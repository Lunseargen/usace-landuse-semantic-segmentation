"""Extra coverage for 0.3: cleanup helper and report writer failure paths."""
import os

import pytest

from semseg import setup_check, storage


def test_remove_missing_file_is_silent(tmp_path):
    storage._remove(str(tmp_path / "none"))


def test_write_report_replace_failure_removes_temp(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("no")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        setup_check.write_report({"a": 1}, tmp_path / "r.json")
    assert list(tmp_path.iterdir()) == []


def test_main_defaults_use_registry_roots(tmp_path, monkeypatch):
    seen = {}

    def fake_build(roots, **kw):
        seen["roots"] = roots
        return {"pass": True, "naip": {}, "naip_total": 0, "gpu": {"found": True}}

    monkeypatch.setattr(setup_check, "build_report", fake_build)
    out = tmp_path / "o.json"
    assert setup_check.main(["--out", str(out)]) == 0
    assert seen["roots"] == storage.load_roots()
    assert out.exists()
