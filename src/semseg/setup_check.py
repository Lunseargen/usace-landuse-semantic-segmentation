"""Set-up check: share, roots, GPU state, HF_HOME, pilot layers, NAIP chunk count.

Run: `uv run python -m semseg.setup_check`. It writes `results/setup_check.json`.
Read only (D22, D37): it calls `nvidia-smi` queries and `free -m`, and it allocates nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from semseg import standard, storage
from semseg.config import DEFAULT_PATH

EXPECTED_NAIP = {"77": 157, "392": 227}
PILOTS = (77, 392)
YEARS = range(1985, 2026)
KINDS = ("nlcd", "classf", "tcc")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "setup_check.json"
GPU_FIELDS = "name,memory.total,memory.used,memory.free,utilization.gpu"
APP_FIELDS = "pid,process_name,used_memory"
_C_ENV = {"LC_ALL": "C"}      # FM-49: fixed labels in `free` and `nvidia-smi`


def _run(run, cmd):
    """Return stdout lines of a successful command, or None."""
    try:
        done = run(cmd, capture_output=True, text=True, timeout=30, check=False,
                   env={**os.environ, **_C_ENV})
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def _number(text):
    """Return an int, or None for `[N/A]` or any text that is not an integer."""
    try:
        return int(text)
    except ValueError:
        return None


def _csv(run, query, fields, count):
    cmd = ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"]
    rows = []
    for line in _run(run, cmd) or []:
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == count:
            rows.append(cells)
    return rows


def query_gpu(run=subprocess.run):
    """Return `{"found", "gpus", "apps"}` from read-only `nvidia-smi` queries (D37)."""
    gpus = [
        {"name": c[0], "memory_total_mb": _number(c[1]), "memory_used_mb": _number(c[2]),
         "memory_free_mb": _number(c[3]), "utilization_pct": _number(c[4])}
        for c in _csv(run, "gpu", GPU_FIELDS, 5)
    ]
    apps = []
    if gpus:
        for c in _csv(run, "compute-apps", APP_FIELDS, 3):
            pid, used = _number(c[0]), _number(c[2])
            if pid is not None:
                apps.append({"pid": pid, "process_name": c[1], "used_memory_mb": used})
    return {"found": bool(gpus), "gpus": gpus, "apps": apps}


def read_memory(run=subprocess.run):
    """Return the unified memory from `free -m`. Values are None on a failure."""
    result = {"total_mb": None, "used_mb": None, "available_mb": None}
    for line in _run(run, ["free", "-m"]) or []:
        cells = line.split()
        if cells[0] == "Mem:" and len(cells) >= 7:
            values = [_number(cells[i]) for i in (1, 2, 6)]
            result = dict(zip(result, values))
    return result


def _pilot_dir(root, pid):
    found = sorted(Path(root).glob(f"{pid}_*")) if Path(root).is_dir() else []
    return found[0] if found else None


def _is_chunk_file(folder, name):
    """True for a regular file `r<row>_c<col>.tif` with a standard chunk ID (FM-50)."""
    if not name.endswith(".tif"):
        return False
    try:
        standard.parse_chunk_id(name[:-4])
    except ValueError:
        return False
    return (folder / name).is_file()


def count_naip(fgt_root, pilots=PILOTS):
    """Return `{pid: number of r<row>_c<col>.tif files}` in `<fgt>/<pid>_<NAME>/naip` (D35)."""
    counts = {}
    for pid in pilots:
        folder = _pilot_dir(fgt_root, pid)
        naip = folder / "naip" if folder else None
        names = os.listdir(naip) if naip and naip.is_dir() else []
        counts[str(pid)] = sum(1 for n in names if _is_chunk_file(naip, n))
    return counts


def check_layers(stack_root, pilots=PILOTS):
    """Return the layer counts of each pilot in the stack and an `ok` flag (read only)."""
    result = {}
    for pid in pilots:
        folder = _pilot_dir(stack_root, pid)
        entry = {}
        for kind in KINDS:
            entry[kind] = sum(
                1 for year in YEARS if folder and (folder / f"{kind}_{year}.tif").is_file())
        entry["boundary"] = bool(folder and (folder / "boundary.gpkg").is_file())
        entry["vectors"] = bool(folder and (folder / "vectors.gpkg").is_file())
        entry["ok"] = (all(entry[k] == len(YEARS) for k in KINDS)
                       and entry["boundary"] and entry["vectors"])
        result[str(pid)] = entry
    return result


def _stamp(now):
    """Return `(created, run_id)` for a UTC time (D40)."""
    now = datetime.now(timezone.utc) if now is None else now
    return now.strftime("%Y-%m-%dT%H:%M:%SZ"), standard.run_id(now, "setup_check")


def failure_report(roots, reason, now=None):
    """Return the report of a run that stopped before the checks (D40): `pass` is false."""
    created, run = _stamp(now)
    return {"created": created, "run_id": run, "share_ok": False, "pass": False,
            "reason": reason, "roots": {name: str(path) for name, path in roots.items()}}


def build_report(roots, *, run=subprocess.run, pilots=PILOTS, expected_naip=None,
                 environ=None, now=None):
    """Return the report. Exit code 3 (SystemExit) if the share is missing."""
    storage.guard_share(roots)
    expected = dict(EXPECTED_NAIP if expected_naip is None else expected_naip)
    hf_home = storage.ensure_hf_home(roots, environ)
    gpu = query_gpu(run)
    layers = check_layers(roots["stack"], pilots)
    naip = count_naip(roots["fgt"], pilots)
    ok = gpu["found"] and all(v["ok"] for v in layers.values()) and naip == expected
    created, run_ident = _stamp(now)
    return {
        "created": created,
        "run_id": run_ident,
        "share_ok": True,
        "roots": {name: str(path) for name, path in roots.items()},
        "gpu": gpu,
        "memory": read_memory(run),
        "hf_home": hf_home,
        "layers": layers,
        "naip": naip,
        "expected_naip": expected,
        "naip_total": sum(naip.values()),
        "pass": bool(ok),
        "reason": None if ok else "a check failed: see gpu, layers and naip",
    }


def _check_out_path(path, roots):
    """Refuse an `--out` path under a read-only root, the share root or the semseg root.

    The report stays on the local disk (DS-26). A missing share is never created (FM-5).
    """
    storage.check_not_readonly(path, roots)
    for name in ("share", "semseg"):
        if name in roots and storage._inside(path, roots[name]):
            raise PermissionError(f"the report must not go under the {name} root: {path}")


def write_report(report, path, roots=None):
    """Write the report as JSON on the local disk: temp file, then `os.replace`.

    With `roots`, a path under the share, the semseg root or a read-only root raises
    PermissionError (FM-51, FM-5). It makes at most the final missing folder, with no
    `parents=True`: a missing parent raises FileNotFoundError.
    """
    if roots is not None:
        _check_out_path(path, roots)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"     # TypeError before any file
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        storage._remove(tmp)
        raise
    return path


def main(argv=None, *, roots=None, run=None, expected_naip=None, environ=None):
    """Run the check. Return 0 for a pass and 1 for a fail. Exit code 3 if no share.

    The report is written in each case (D40). A ValueError or an OSError (for example a bad
    `HF_HOME` or a bad registry) writes a failure report and returns 1.
    """
    parser = argparse.ArgumentParser(prog="semseg.setup_check", description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON output file")
    args = parser.parse_args(argv)
    known = roots if roots is not None else {}
    try:
        known = roots if roots is not None else storage.load_roots(config=DEFAULT_PATH)
        report = build_report(known, run=run or subprocess.run, expected_naip=expected_naip,
                              environ=environ)
    except SystemExit:                    # D40: the report is written on a missing share too
        write_report(failure_report(known, "share or semseg root missing"), args.out,
                     known or None)
        raise
    except (ValueError, OSError) as exc:  # D40: no stale `pass: true` after any failure
        write_report(failure_report(known, f"{type(exc).__name__}: {exc}"), args.out,
                     known or None)
        print(f"setup_check: failed: {exc} -> {args.out}", file=sys.stderr)
        return 1
    write_report(report, args.out, known)
    print(f"setup_check: pass={report['pass']} naip={report['naip']} "
          f"total={report['naip_total']} gpu_found={report['gpu']['found']} -> {args.out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
