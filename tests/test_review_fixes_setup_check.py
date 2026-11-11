"""Review fixes 0.3: failure report for ValueError/OSError, and the --out guard."""
import json
import subprocess

import pytest

from semseg import setup_check, storage


def make_roots(base):
    roots = {"share": base / "share", "semseg": base / "share" / "semseg",
             "fgt": base / "share" / "fgt", "stack": base / "stack"}
    for path in roots.values():
        path.mkdir(parents=True)
    return roots


@pytest.mark.parametrize("where", ["share", "semseg"])
def test_out_under_share_or_semseg_refused_nothing_created(tmp_path, where):
    roots = make_roots(tmp_path)
    out = roots[where] / "r" / "deep" / "setup_check.json"
    with pytest.raises(PermissionError):
        setup_check.write_report({"pass": True}, out, roots)
    assert not (roots[where] / "r").exists()


def test_missing_share_not_created_by_failure_report(tmp_path):
    roots = make_roots(tmp_path)
    out = roots["semseg"] / "r" / "setup_check.json"
    import shutil
    shutil.rmtree(roots["share"])
    with pytest.raises(PermissionError):
        setup_check.main(["--out", str(out)], roots=roots, run=subprocess.run)
    assert not roots["share"].exists()


def test_write_report_makes_only_final_folder(tmp_path):
    out = tmp_path / "res" / "r.json"
    setup_check.write_report({"a": 1}, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}


def test_write_report_missing_parent_not_created(tmp_path):
    out = tmp_path / "no" / "res" / "r.json"
    with pytest.raises(FileNotFoundError):
        setup_check.write_report({"a": 1}, out)
    assert not (tmp_path / "no").exists()


def test_main_load_roots_value_error_writes_failure_report(tmp_path, monkeypatch):
    def bad(**kwargs):
        raise ValueError("bad registry")
    monkeypatch.setattr(storage, "load_roots", bad)
    out = tmp_path / "r.json"
    out.write_text('{"pass": true}\n', encoding="utf-8")
    assert setup_check.main(["--out", str(out)]) == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pass"] is False and "bad registry" in data["reason"]
    assert data["roots"] == {}


def test_main_oserror_writes_failure_report(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk gone")
    monkeypatch.setattr(setup_check, "check_layers", boom)
    out = tmp_path / "r.json"
    assert setup_check.main(["--out", str(out)], roots=roots, environ={},
                            run=subprocess.run) == 1
    assert "OSError" in json.loads(out.read_text(encoding="utf-8"))["reason"]
