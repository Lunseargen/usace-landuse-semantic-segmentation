"""Set-up check tests (task 0.3: D35, D37, FM-37, FM-42).

API under test (semseg.setup_check):
  query_gpu(run=subprocess.run) -> {"found": bool, "gpus": [..], "apps": [..]}
  read_memory(run=subprocess.run) -> {"total_mb", "used_mb", "available_mb"} or None values
  count_naip(fgt_root, pilots=(77, 392)) -> {"77": int, "392": int}
  check_layers(stack_root, pilots=(77, 392)) -> {"77": {"nlcd","classf","tcc","boundary","vectors","ok"}, ..}
  build_report(roots, *, run=subprocess.run, pilots=(77, 392), expected_naip=None, environ=None) -> dict
      keys: share_ok, roots, gpu, memory, hf_home, layers, naip, naip_total, pass
  write_report(report, path) -> Path   (atomic, local)
  main(argv=None, *, roots=None, run=None, expected_naip=None, environ=None) -> int (0 pass, 1 fail);
      SystemExit(3) when the share is missing (nothing written). argv: ["--out", path].
  EXPECTED_NAIP == {"77": 157, "392": 227}
"""
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from semseg import setup_check
from semseg import storage

GPU_OK = "NVIDIA GB10, [N/A], [N/A], [N/A], 96\n"
APPS_OK = "4321, VLLM::EngineCore, 79903\n"
FREE_OK = (
    "               total        used        free      shared  buff/cache   available\n"
    "Mem:          124546       97683        1000         100       25863       26863\n"
    "Swap:              0           0           0\n"
)


def cp(out="", code=0):
    return subprocess.CompletedProcess([], code, stdout=out, stderr="")


class FakeRun:
    """Mock of subprocess.run that records each command."""

    def __init__(self, gpu=GPU_OK, apps=APPS_OK, free=FREE_OK, gpu_code=0):
        self.gpu, self.apps, self.free, self.gpu_code = gpu, apps, free, gpu_code
        self.calls = []

    def __call__(self, cmd, *a, **k):
        self.calls.append(list(cmd))
        if cmd[0] == "nvidia-smi":
            if self.gpu_code:
                return cp("", self.gpu_code)
            if any(c.startswith("--query-gpu") for c in cmd):
                return cp(self.gpu)
            if any(c.startswith("--query-compute-apps") for c in cmd):
                return cp(self.apps)
        if cmd[0] == "free":
            return cp(self.free)
        raise AssertionError(f"unexpected command {cmd}")


def missing_binary(cmd, *a, **k):
    raise FileNotFoundError(cmd[0])


def make_tree(base, naip=None, layers=True):
    naip = naip or {77: 3, 392: 2}
    roots = {"share": base / "share", "semseg": base / "share" / "semseg",
             "fgt": base / "share" / "fgt", "stack": base / "stack"}
    for r in roots.values():
        r.mkdir(parents=True)
    for pid, name in ((77, "77_GRAPEVINE_LAKE"), (392, "392_RAYSTOWN_LAKE")):
        nd = roots["fgt"] / name / "naip"
        nd.mkdir(parents=True)
        for i in range(naip[pid]):
            (nd / f"r{i}_c{i}.tif").write_bytes(b"x")
        sd = roots["stack"] / name
        sd.mkdir(parents=True)
        if layers:
            for kind in ("nlcd", "classf", "tcc"):
                for y in range(1985, 2026):
                    (sd / f"{kind}_{y}.tif").write_bytes(b"x")
            (sd / "boundary.gpkg").write_bytes(b"x")
            (sd / "vectors.gpkg").write_bytes(b"x")
    return roots


# ---------------------------------------------------------------- GPU parse (D37, FM-37)

def test_d37_query_gpu_parses_unified_memory_na():
    gpu = setup_check.query_gpu(FakeRun())
    assert gpu["found"] is True
    g = gpu["gpus"][0]
    assert g["name"] == "NVIDIA GB10"
    assert g["utilization_pct"] == 96
    assert g["memory_total_mb"] is None and g["memory_free_mb"] is None and g["memory_used_mb"] is None
    assert gpu["apps"] == [{"pid": 4321, "process_name": "VLLM::EngineCore", "used_memory_mb": 79903}]


def test_d37_query_gpu_parses_numbers():
    run = FakeRun(gpu="Tesla V100, 32510, 1000, 31510, 5\n", apps="")
    gpu = setup_check.query_gpu(run)
    g = gpu["gpus"][0]
    assert (g["memory_total_mb"], g["memory_used_mb"], g["memory_free_mb"]) == (32510, 1000, 31510)
    assert gpu["found"] is True and gpu["apps"] == []


def test_d37_query_gpu_two_gpus():
    run = FakeRun(gpu="A, 10, 1, 9, 0\nB, 20, 2, 18, 1\n", apps="")
    gpu = setup_check.query_gpu(run)
    assert [g["name"] for g in gpu["gpus"]] == ["A", "B"]


def test_d37_query_gpu_uses_read_only_queries_only():
    run = FakeRun()
    setup_check.query_gpu(run)
    assert run.calls
    for cmd in run.calls:
        assert cmd[0] == "nvidia-smi"
        flags = [c for c in cmd[1:] if c.startswith("-")]
        assert flags and all(
            f.startswith(("--query-gpu", "--query-compute-apps", "--format")) for f in flags)


def test_edge_nvidia_smi_missing_gives_null():
    gpu = setup_check.query_gpu(missing_binary)
    assert gpu["found"] is False and gpu["gpus"] == [] and gpu["apps"] == []


def test_edge_nvidia_smi_exit_9_not_found():
    gpu = setup_check.query_gpu(FakeRun(gpu_code=9))
    assert gpu["found"] is False and gpu["gpus"] == []


def test_edge_nvidia_smi_garbage_not_found():
    gpu = setup_check.query_gpu(FakeRun(gpu="NVIDIA-SMI has failed\n", apps=""))
    assert gpu["found"] is False


def test_edge_nvidia_smi_empty_output_not_found():
    assert setup_check.query_gpu(FakeRun(gpu="", apps=""))["found"] is False


def test_edge_apps_garbage_ignored_gpu_still_found():
    gpu = setup_check.query_gpu(FakeRun(apps="No devices were found\n"))
    assert gpu["found"] is True and gpu["apps"] == []


def test_d37_read_memory_parses_free_m():
    run = FakeRun()
    mem = setup_check.read_memory(run)
    assert mem == {"total_mb": 124546, "used_mb": 97683, "available_mb": 26863}
    assert run.calls == [["free", "-m"]]


def test_edge_read_memory_missing_binary_null():
    mem = setup_check.read_memory(missing_binary)
    assert mem == {"total_mb": None, "used_mb": None, "available_mb": None}


def test_edge_read_memory_garbage_null():
    mem = setup_check.read_memory(FakeRun(free="nonsense\n"))
    assert mem["available_mb"] is None


def test_d37_integration_real_nvidia_smi_read_only():
    gpu = setup_check.query_gpu()
    assert isinstance(gpu["found"], bool)
    assert isinstance(gpu["gpus"], list) and isinstance(gpu["apps"], list)
    mem = setup_check.read_memory()
    assert isinstance(mem["available_mb"], int) and mem["available_mb"] > 0


def test_setup_check_runs_no_gpu():
    src = inspect.getsource(setup_check)
    assert "torch" not in src and "cuda" not in src.lower() and "samgeo" not in src


# ---------------------------------------------------------------- NAIP count (D35, FM-42)

def test_d35_expected_naip_constants():
    assert setup_check.EXPECTED_NAIP == {"77": 157, "392": 227}
    assert sum(setup_check.EXPECTED_NAIP.values()) == 384


def test_d35_count_naip_per_pilot(tmp_path):
    roots = make_tree(tmp_path, naip={77: 5, 392: 7})
    assert setup_check.count_naip(roots["fgt"]) == {"77": 5, "392": 7}


def test_fm42_count_ignores_non_chunk_files(tmp_path):
    roots = make_tree(tmp_path, naip={77: 4, 392: 1})
    nd = roots["fgt"] / "77_GRAPEVINE_LAKE" / "naip"
    (nd / "r0_c0.tif.aux.xml").write_bytes(b"x")
    (nd / "other.tif").write_bytes(b"x")
    (nd / "Thumbs.db").write_bytes(b"x")
    (nd / "r9_c9.tif.part").write_bytes(b"x")
    (nd / "sub").mkdir()
    assert setup_check.count_naip(roots["fgt"]) == {"77": 4, "392": 1}


def test_d35_count_naip_missing_pilot_is_zero(tmp_path):
    roots = make_tree(tmp_path)
    import shutil
    shutil.rmtree(roots["fgt"] / "392_RAYSTOWN_LAKE")
    assert setup_check.count_naip(roots["fgt"])["392"] == 0


def test_d35_count_naip_does_not_cross_pilots(tmp_path):
    roots = make_tree(tmp_path, naip={77: 2, 392: 3})
    assert setup_check.count_naip(roots["fgt"], pilots=(392,)) == {"392": 3}


def test_nlcd_chunk_count_384():
    roots = storage.load_roots()
    counts = setup_check.count_naip(roots["fgt"])
    assert counts == {"77": 157, "392": 227}
    assert sum(counts.values()) == 384


# ---------------------------------------------------------------- layers (FM-37)

def test_layers_full_set_ok(tmp_path):
    roots = make_tree(tmp_path)
    res = setup_check.check_layers(roots["stack"])
    for pid in ("77", "392"):
        r = res[pid]
        assert (r["nlcd"], r["classf"], r["tcc"]) == (41, 41, 41)
        assert r["boundary"] is True and r["vectors"] is True and r["ok"] is True


def test_edge_missing_layer_sets_ok_false(tmp_path):
    roots = make_tree(tmp_path)
    (roots["stack"] / "77_GRAPEVINE_LAKE" / "tcc_2000.tif").unlink()
    res = setup_check.check_layers(roots["stack"])
    assert res["77"]["tcc"] == 40 and res["77"]["ok"] is False
    assert res["392"]["ok"] is True


def test_edge_missing_gpkg_sets_ok_false(tmp_path):
    roots = make_tree(tmp_path)
    (roots["stack"] / "392_RAYSTOWN_LAKE" / "boundary.gpkg").unlink()
    res = setup_check.check_layers(roots["stack"])
    assert res["392"]["boundary"] is False and res["392"]["ok"] is False


def test_edge_missing_pilot_folder_ok_false(tmp_path):
    roots = make_tree(tmp_path)
    import shutil
    shutil.rmtree(roots["stack"] / "77_GRAPEVINE_LAKE")
    res = setup_check.check_layers(roots["stack"])
    assert res["77"]["ok"] is False and res["77"]["nlcd"] == 0


def test_real_stack_layers_ok():
    roots = storage.load_roots()
    res = setup_check.check_layers(roots["stack"])
    assert res["77"]["ok"] and res["392"]["ok"]


# ---------------------------------------------------------------- report (schema)

def test_report_schema_and_pass(tmp_path):
    roots = make_tree(tmp_path)
    env = {}
    rep = setup_check.build_report(roots, run=FakeRun(), expected_naip={"77": 3, "392": 2}, environ=env)
    for key in ("share_ok", "roots", "gpu", "memory", "hf_home", "layers", "naip", "naip_total", "pass"):
        assert key in rep
    assert rep["share_ok"] is True
    assert rep["naip"] == {"77": 3, "392": 2} and rep["naip_total"] == 5
    assert rep["roots"]["semseg"] == str(roots["semseg"])
    assert rep["hf_home"] == str(roots["semseg"] / "models") == env["HF_HOME"]
    assert rep["gpu"]["found"] is True
    assert rep["memory"]["available_mb"] == 26863
    assert rep["layers"]["77"]["ok"] is True
    assert rep["pass"] is True
    json.dumps(rep)                                    # serialisable


def test_d37_fail_only_when_no_gpu(tmp_path):
    roots = make_tree(tmp_path)
    exp = {"77": 3, "392": 2}
    ok = setup_check.build_report(roots, run=FakeRun(), expected_naip=exp, environ={})
    # busy GPU (vLLM holds memory, little free memory) does not fail the check
    busy = setup_check.build_report(
        roots, run=FakeRun(free=FREE_OK.replace("26863", "10")), expected_naip=exp, environ={})
    none = setup_check.build_report(roots, run=missing_binary, expected_naip=exp, environ={})
    assert ok["pass"] is True and busy["pass"] is True
    assert none["pass"] is False and none["gpu"]["found"] is False
    assert none["memory"]["available_mb"] is None


def test_fm42_wrong_naip_count_fails(tmp_path):
    roots = make_tree(tmp_path, naip={77: 3, 392: 1})
    rep = setup_check.build_report(roots, run=FakeRun(), expected_naip={"77": 3, "392": 2}, environ={})
    assert rep["naip"] == {"77": 3, "392": 1} and rep["naip_total"] == 4
    assert rep["pass"] is False


def test_fm37_missing_layer_fails_report(tmp_path):
    roots = make_tree(tmp_path)
    (roots["stack"] / "77_GRAPEVINE_LAKE" / "nlcd_1990.tif").unlink()
    rep = setup_check.build_report(roots, run=FakeRun(), expected_naip={"77": 3, "392": 2}, environ={})
    assert rep["layers"]["77"]["ok"] is False and rep["pass"] is False


def test_report_default_expected_is_384(tmp_path):
    roots = make_tree(tmp_path)               # small tree: not 384
    rep = setup_check.build_report(roots, run=FakeRun(), environ={})
    assert rep["pass"] is False


def test_build_report_share_missing_exit_3(tmp_path):
    roots = make_tree(tmp_path)
    roots["semseg"].rmdir()
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as exc:
        setup_check.build_report(roots, run=FakeRun(), environ={})
    assert exc.value.code == 3
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_build_report_runs_no_write_to_roots(tmp_path):
    roots = make_tree(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    setup_check.build_report(roots, run=FakeRun(), expected_naip={"77": 3, "392": 2}, environ={})
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


# ---------------------------------------------------------------- writer and main

def test_write_report_roundtrip(tmp_path):
    out = tmp_path / "results" / "setup_check.json"
    out.parent.mkdir()
    rep = {"pass": True, "naip": {"77": 157, "392": 227}, "naip_total": 384}
    assert Path(setup_check.write_report(rep, out)) == out
    assert json.loads(out.read_text(encoding="utf-8")) == rep
    assert out.stat().st_size < 1_000_000
    assert [p.name for p in out.parent.iterdir()] == ["setup_check.json"]


def test_edge_setup_check_json_atomic(tmp_path):
    out = tmp_path / "setup_check.json"
    out.write_text('{"pass": true}', encoding="utf-8")
    with pytest.raises(TypeError):
        setup_check.write_report({"bad": object()}, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"pass": True}
    assert [p.name for p in tmp_path.iterdir()] == ["setup_check.json"]


def test_edge_write_report_failure_leaves_no_file(tmp_path):
    out = tmp_path / "setup_check.json"
    with pytest.raises(TypeError):
        setup_check.write_report({"bad": object()}, out)
    assert list(tmp_path.iterdir()) == []


def test_main_writes_json_and_returns_0(tmp_path):
    roots = make_tree(tmp_path)
    out = tmp_path / "out" / "setup_check.json"
    out.parent.mkdir()
    code = setup_check.main(["--out", str(out)], roots=roots, run=FakeRun(),
                            expected_naip={"77": 3, "392": 2}, environ={})
    assert code == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["pass"] is True and rep["naip_total"] == 5


def test_main_failure_writes_json_and_returns_1(tmp_path):
    roots = make_tree(tmp_path)
    out = tmp_path / "setup_check.json"
    code = setup_check.main(["--out", str(out)], roots=roots, run=missing_binary,
                            expected_naip={"77": 3, "392": 2}, environ={})
    assert code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["pass"] is False


def test_main_share_missing_exit_3_writes_failed_report(tmp_path):
    roots = make_tree(tmp_path)
    roots["share"].rename(tmp_path / "gone")
    out = tmp_path / "setup_check.json"
    with pytest.raises(SystemExit) as exc:
        setup_check.main(["--out", str(out)], roots=roots, run=FakeRun(), environ={})
    assert exc.value.code == 3
    rep = json.loads(out.read_text(encoding="utf-8"))          # D40: the report is written
    assert rep["pass"] is False and rep["share_ok"] is False and rep["reason"]
