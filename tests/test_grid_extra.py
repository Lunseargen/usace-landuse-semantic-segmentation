"""Extra tests of `semseg.grid` for the metadata helpers (task 1.1, DS-21)."""
import subprocess
from datetime import datetime, timezone

import pytest

from semseg import grid, standard


def test_code_version_is_full_sha_with_optional_dirty():
    version = grid._code_version()
    assert len(version.removesuffix("-dirty")) in (40, 64)


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 1)])
def test_code_version_raises_without_git(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(grid.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        grid._code_version()


def test_code_version_raises_for_a_failed_command(monkeypatch):
    done = subprocess.CompletedProcess([], 128, stdout="", stderr="not a repository")
    monkeypatch.setattr(grid.subprocess, "run", lambda *a, **k: done)
    with pytest.raises(RuntimeError):
        grid._code_version()


def test_code_version_raises_for_output_that_is_not_a_sha(monkeypatch):
    done = subprocess.CompletedProcess([], 0, stdout="HEAD\n", stderr="")
    monkeypatch.setattr(grid.subprocess, "run", lambda *a, **k: done)
    with pytest.raises(RuntimeError):
        grid._code_version()


def test_metadata_has_all_ds21_keys():
    now = datetime(2026, 11, 11, 8, 30, tzinfo=timezone.utc)
    meta = grid._metadata("77_T", now)
    assert tuple(meta) == standard.METADATA_KEYS
    assert meta["run_id"] == "20261111-0830-chunk_grid"
    assert meta["data_standard"] == str(standard.DATA_STANDARD_VERSION)
    assert meta["created"] == "2026-11-11T08:30:00Z"
    assert all(meta.values())
