"""Fix-round tests for task 0.3 (D39, D40, FM-43 to FM-53). Fake roots in tmp_path only."""
import json
import subprocess
import re
from datetime import datetime, timezone

import pytest

from semseg import setup_check, standard, storage


def no_gpu(cmd, *a, **k):
    return subprocess.CompletedProcess(cmd, 9, "", "")


def make_roots(base):
    roots = {"share": base / "share", "semseg": base / "share" / "semseg",
             "fgt": base / "share" / "fgt", "stack": base / "stack"}
    for path in roots.values():
        path.mkdir(parents=True)
    return roots


def test_write_into_existing_nested_folder(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "a" / "b").mkdir(parents=True)
    storage.safe_write(roots["semseg"] / "a" / "b" / "c" / "f.bin", b"X", roots)
    assert (roots["semseg"] / "a" / "b" / "c" / "f.bin").read_bytes() == b"X"
    assert not list(roots["semseg"].rglob("*.part"))


def test_case_variant_of_semseg_root_refused(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["share"] / "SEMSEG" / "f.bin", b"X", roots)
    assert not (roots["share"] / "SEMSEG").exists()


def test_part_name_is_new_each_time(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    names = []
    real = storage.os.replace
    monkeypatch.setattr(storage.os, "replace", lambda s, d: (names.append(s), real(s, d))[1])
    for _ in range(2):
        storage.safe_write(roots["semseg"] / "f.bin", b"X", roots)
    assert len(set(names)) == 2
    assert all(re.fullmatch(r".*/f\.bin\.[0-9a-f]{8}\.part", str(n)) for n in names)


def test_guard_runs_again_before_replace(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    calls = []
    real = storage.guard_share
    monkeypatch.setattr(storage, "guard_share", lambda r: (calls.append(1), real(r))[1])
    storage.safe_write(roots["semseg"] / "f.bin", b"X", roots)
    assert len(calls) == 2        # at the start and right before the replace


def test_load_roots_default_requires_all_four(tmp_path):
    reg = tmp_path / "r.toml"
    reg.write_text('[roots]\nshare = "/a"\nsemseg = "{share}/s"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="fgt"):
        storage.load_roots(reg)


def test_report_has_created_and_run_id(tmp_path):
    roots = make_roots(tmp_path)
    now = datetime(2026, 11, 11, 8, 5, 59, tzinfo=timezone.utc)
    rep = setup_check.build_report(roots, run=no_gpu, environ={}, now=now)
    assert rep["created"] == "2026-11-11T08:05:59Z"
    assert rep["run_id"] == "20261111-0805-setup_check"
    standard.check_run_id(rep["run_id"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", setup_check.failure_report(
        roots, "x")["created"])


def test_failure_report_fields(tmp_path):
    roots = make_roots(tmp_path)
    rep = setup_check.failure_report(roots, "why")
    assert rep["pass"] is False and rep["reason"] == "why" and rep["share_ok"] is False
    assert rep["roots"]["stack"] == str(roots["stack"])


def test_write_report_refuses_readonly_root(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(PermissionError):
        setup_check.write_report({"a": 1}, roots["fgt"] / "x" / "o.json", roots)
    assert not (roots["fgt"] / "x").exists()


def test_main_failed_run_report_has_reason(tmp_path):
    roots = make_roots(tmp_path)
    out = tmp_path / "o.json"
    code = setup_check.main(["--out", str(out)], roots=roots,
                            run=no_gpu, environ={})
    assert code == 1
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["pass"] is False and rep["reason"] and rep["run_id"].endswith("-setup_check")
