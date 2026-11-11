"""Chunk grid of a pilot (task 1.1, R2, D13, D43, D44).

The chunk list of a pilot is every DS-3 chunk whose 960 m box has a positive area inside the
footprint (`boundary.gpkg`) buffered by 150 m. A chunk that only touches the edge of the buffer
is not in the list. All row, column, ID and box values come from `semseg.standard`.
`chunks.parquet` is written to `<semseg>/<pid>_<NAME>/` with `semseg.storage.safe_write`.

Use: `python -m semseg.grid <pid>...`. The CLI does not read `chunks.csv` of fine_grid.
"""
import argparse
import io
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely.geometry import MultiPolygon, Polygon

from semseg import standard, storage

BUFFER = 150
NAIP_YEAR = 2022
SHARE_DECIMALS = 4
SHARE_TOLERANCE = 1e-4
COLUMNS = ["chunk", "row", "col", "xmin", "ymin", "xmax", "ymax", "footprint_share", "pid"]
REF_COLUMNS = COLUMNS[:-1]
BOUNDARY_FILE = "boundary.gpkg"
FG_CHUNKS_FILE = "chunks.csv"
METHOD = "chunk_grid"
_BOX_COLUMNS = ["xmin", "ymin", "xmax", "ymax"]
_POLYGONS = (Polygon, MultiPolygon)
MAX_PID = 2 ** 63 - 1
MAX_CANDIDATES = 100_000
_FOLDER_RE = re.compile(r"([0-9]+)_[A-Z0-9_]+", re.ASCII)
_DTYPES = {"row": "int64", "col": "int64", "pid": "int64", "xmin": "float64", "ymin": "float64",
           "xmax": "float64", "ymax": "float64", "footprint_share": "float64"}
_FLOAT_COLUMNS = [name for name, dtype in _DTYPES.items() if dtype == "float64"]


def load_footprint(path):
    """Return the footprint polygon of `boundary.gpkg`.

    Raise ValueError for a CRS other than EPSG:5070, a feature count other than 1, an invalid
    geometry or a geometry that is not a polygon (FM-54). A missing file raises the read error.
    """
    frame = gpd.read_file(path)
    epsg = frame.crs.to_epsg() if frame.crs is not None else None
    if epsg != 5070:
        raise ValueError(f"{path}: the CRS must be {standard.CRS}, got {frame.crs}")
    if len(frame) != 1:
        raise ValueError(f"{path}: expected 1 feature, got {len(frame)}")
    geometry = frame.geometry.iloc[0]
    _check_footprint(geometry, path)
    return geometry


def _check_footprint(geometry, name="footprint"):
    if not isinstance(geometry, _POLYGONS):
        raise ValueError(f"{name}: the footprint must be a polygon, got {type(geometry).__name__}")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{name}: the footprint is empty or not valid")


def _check_pid(pid):
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid <= MAX_PID:
        raise ValueError(f"pid must be an integer from 1 to {MAX_PID}, got {pid!r}")


def _candidates(footprint):
    """Return the sorted (row, col) pairs in the bounds of each buffered polygon part (FM-65).

    Raise ValueError if the ranges hold more than `MAX_CANDIDATES` pairs.
    """
    parts = footprint.geoms if isinstance(footprint, MultiPolygon) else [footprint]
    ranges = []
    total = 0
    for part in parts:
        x_min, y_min, x_max, y_max = part.buffer(BUFFER).bounds
        rows = range(math.floor((standard.Y0 - y_max) / standard.CHUNK),
                     math.floor((standard.Y0 - y_min) / standard.CHUNK) + 1)
        cols = range(math.floor((x_min - standard.X0) / standard.CHUNK),
                     math.floor((x_max - standard.X0) / standard.CHUNK) + 1)
        total += len(rows) * len(cols)
        if total > MAX_CANDIDATES:
            raise ValueError(f"the footprint needs more than {MAX_CANDIDATES} candidate chunks")
        ranges.append((rows, cols))
    return sorted({(row, col) for rows, cols in ranges for row in rows for col in cols})


def chunk_list(footprint, pid):
    """Return the chunk list DataFrame of a footprint (D43, D44).

    A chunk is in the list if the area of (box and footprint buffered by 150 m) is above 0.
    `footprint_share` is the area of the chunk inside the unbuffered footprint / 960**2.
    """
    _check_pid(pid)
    _check_footprint(footprint)
    buffered = footprint.buffer(BUFFER)
    cells = _candidates(footprint)
    records = [(row, col) + standard.chunk_bounds(row, col) for row, col in cells]
    boxes = shapely.box(*zip(*[record[2:] for record in records]))
    keep = shapely.area(shapely.intersection(boxes, buffered)) > 0
    shares = shapely.area(shapely.intersection(boxes, footprint)) / standard.CHUNK ** 2
    frame = pd.DataFrame(records, columns=["row", "col"] + _BOX_COLUMNS)
    frame["footprint_share"] = shares.round(SHARE_DECIMALS)
    frame = frame[keep].sort_values(["row", "col"]).reset_index(drop=True)
    frame["chunk"] = [standard.chunk_id(r, c) for r, c in zip(frame["row"], frame["col"])]
    frame["pid"] = pid
    frame = frame[COLUMNS].astype({"row": "int64", "col": "int64", "pid": "int64"})
    return frame.astype({name: "float64" for name in _BOX_COLUMNS + ["footprint_share"]})


def read_fg_chunks(folder, roots):
    """Return `<fgt>/<folder>/chunks.csv` of fine_grid (read only). Exit 3 if it is missing (FM-57)."""
    storage.guard_share(roots)
    standard.check_name(folder)
    root = Path(roots["fgt"]).resolve()
    inside = (root / folder).resolve()
    if inside.parent != root:
        raise ValueError(f"the folder is not directly inside the fgt root: {folder!r}")
    path = inside / FG_CHUNKS_FILE
    if not path.is_file():
        print(f"{FG_CHUNKS_FILE} not found: {path}", file=sys.stderr)
        raise SystemExit(standard.EXIT_SHARE_MISSING)
    return pd.read_csv(path)


def compare_chunks(df, ref):
    """Return `{"extra": [...], "missing": [...], "moved": [...]}` of chunk IDs (FM-59).

    extra: in `df`, not in `ref`. missing: in `ref`, not in `df`. moved: the same ID with a
    different row, col or bound, or a `footprint_share` that differs by more than 1e-4.
    """
    if len(df) == 0 and len(ref) == 0:
        raise ValueError("both chunk lists are empty")
    for name, frame in (("df", df), ("ref", ref)):
        if not frame["chunk"].is_unique:
            raise ValueError(f"{name} has duplicate chunk IDs")
    known = set(ref["chunk"])
    current = set(df["chunk"])
    merged = df.merge(ref, on="chunk", suffixes=("", "_ref"))
    moved = pd.Series(False, index=merged.index)
    for name in ["row", "col"] + _BOX_COLUMNS:
        moved |= merged[name] != merged[name + "_ref"]
    moved |= merged["footprint_share"].isna() | merged["footprint_share_ref"].isna()
    moved |= (merged["footprint_share"] - merged["footprint_share_ref"]).abs() > SHARE_TOLERANCE + 1e-9
    return {
        "extra": [c for c in df["chunk"] if c not in known],
        "missing": [c for c in ref["chunk"] if c not in current],
        "moved": merged.loc[moved, "chunk"].tolist(),
    }


def _git(*args):
    """Run git in the module folder. Raise RuntimeError if git fails. Return the stdout."""
    try:
        done = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"git {' '.join(args)} failed: {error}") from error
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (code {done.returncode}): {done.stderr.strip()}")
    return done.stdout.strip()


def _code_version():
    """Return the full git SHA of HEAD, plus `-dirty` if a tracked file changed (DS-21, D45).

    Raise RuntimeError if git fails or the repository has no commit: nothing is written then.
    """
    sha = _git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", sha):
        raise RuntimeError(f"git rev-parse HEAD gave no commit: {sha!r}")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return sha + "-dirty" if dirty else sha


def _metadata(folder, now=None):
    """Return the DS-21 key-value metadata of `chunks.parquet` (all values non-empty)."""
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must have a time zone, got {now!r}")
    now = now.astimezone(timezone.utc)
    values = {
        "run_id": standard.run_id(now, METHOD),
        "code_version": _code_version(),
        "data_standard": str(standard.DATA_STANDARD_VERSION),
        "naip_year": str(NAIP_YEAR),
        "sources": f"stack/{folder}/{BOUNDARY_FILE}",
        "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return values


def _check_frame(df, folder):
    """Check the frame against D43 and the folder (DS-6, DS-22). Raise ValueError (FM-61)."""
    standard.check_name(folder)
    match = _FOLDER_RE.fullmatch(folder)
    if match is None:
        raise ValueError(f"the folder must be <pid>_<NAME> (DS-6), got {folder!r}")
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    for name, dtype in _DTYPES.items():
        if str(df[name].dtype) != dtype:
            raise ValueError(f"column {name} must be {dtype}, got {df[name].dtype}")
    chunks = df["chunk"].tolist()
    if not all(isinstance(c, str) for c in chunks):
        raise ValueError("column chunk must hold strings")
    if len(df) == 0:
        raise ValueError("the chunk list has no row")
    for name in _FLOAT_COLUMNS:
        if not all(math.isfinite(v) for v in df[name].tolist()):
            raise ValueError(f"column {name} holds NaN or a value that is not finite")
    if len(set(chunks)) != len(chunks):
        raise ValueError("duplicate chunk IDs")
    rows, cols = df["row"].tolist(), df["col"].tolist()
    keys = list(zip(rows, cols))
    if keys != sorted(keys):
        raise ValueError("the rows must be sorted by row, then col")
    boxes = df[_BOX_COLUMNS].values.tolist()
    for chunk, row, col, box in zip(chunks, rows, cols, boxes):
        if chunk != standard.chunk_id(row, col) or tuple(box) != standard.chunk_bounds(row, col):
            raise ValueError(f"chunk {chunk!r} does not match its row, col or box")
    if not (df["pid"] == int(match.group(1))).all():
        raise ValueError(f"every pid must be {int(match.group(1))} (folder {folder})")


def write_chunk_list(df, roots, folder):
    """Write `chunks.parquet` to `<semseg>/<folder>/` with `storage.safe_write`. Return the path.

    Raise ValueError if the frame is not a valid D43 chunk list for the folder. The checks
    run before any write. Exit 3 if the share is missing (DS-26).
    """
    _check_frame(df, folder)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.set_column(COLUMNS.index("chunk"), "chunk", table["chunk"].cast(pa.string()))
    meta = {**(table.schema.metadata or {}),
            **{k.encode(): v.encode() for k, v in _metadata(folder).items()}}
    buffer = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(meta), buffer)
    target = Path(roots["semseg"]) / folder / standard.CHUNK_LIST
    return storage.safe_write(target, buffer.getvalue(), roots)


def find_folder(pid, roots):
    """Return the one `<pid>_<NAME>` folder name in the stack root (DS-6).

    `pid` is an int. Raise FileNotFoundError for no folder, ValueError for a bad pid, more than
    one folder, or a name that is not DS-6 (`^<pid>_[A-Z0-9_]+$`, ASCII).
    """
    if isinstance(pid, bool) or not isinstance(pid, int):
        raise ValueError(f"pid must be an int, got {pid!r}")
    prefix = f"{pid}_"
    found = sorted(p.name for p in Path(roots["stack"]).iterdir() if p.is_dir() and p.name.startswith(prefix))
    for name in found:
        if re.fullmatch(re.escape(prefix) + "[A-Z0-9_]+", name, re.ASCII) is None:
            raise ValueError(f"not a DS-6 folder name: {name!r}")
    if not found:
        raise FileNotFoundError(f"no folder {pid}_* in {roots['stack']}")
    if len(found) > 1:
        raise ValueError(f"more than one folder {pid}_* in {roots['stack']}: {found}")
    return found[0]


def _pid_arg(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"pid must be 1 or more: {text!r}")
    return value


def main(argv=None, roots=None):
    """Write `chunks.parquet` for each pid. Return 0.

    Check every pid first (folder, footprint, chunk list), then write (FM-66).
    """
    parser = argparse.ArgumentParser(prog="python -m semseg.grid", description=__doc__.splitlines()[0])
    parser.add_argument("pid", nargs="+", type=_pid_arg, help="project IDs, for example 77 392")
    args = parser.parse_args(argv)
    roots = storage.load_roots() if roots is None else roots
    plans = []
    for pid in args.pid:
        folder = find_folder(pid, roots)
        footprint = load_footprint(Path(roots["stack"]) / folder / BOUNDARY_FILE)
        df = chunk_list(footprint, pid)
        _check_frame(df, folder)
        plans.append((pid, folder, df))
    _code_version()
    for pid, folder, df in plans:
        path = write_chunk_list(df, roots, folder)
        print(f"{pid}: {len(df)} chunks -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
