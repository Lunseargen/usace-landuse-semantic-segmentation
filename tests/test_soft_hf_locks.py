"""Tests for `storage.use_soft_hf_locks` and `storage.soft_lock` (D58).

The GVFS share refuses `FileLock` on an existing lock file and `fchmod` (errno 95).
"""
import os
import types

from filelock import FileLock, SoftFileLock

from semseg import storage


def test_stand_in_module_gets_soft_lock():
    module = types.SimpleNamespace(FileLock=FileLock)
    assert storage.use_soft_hf_locks(module) is module
    assert module.FileLock is storage.soft_lock


def test_stand_in_module_without_attribute():
    module = types.SimpleNamespace()
    storage.use_soft_hf_locks(module)
    assert module.FileLock is storage.soft_lock


def test_default_module_is_hf_fixes(monkeypatch):
    from huggingface_hub.utils import _fixes

    monkeypatch.setattr(_fixes, "FileLock", _fixes.FileLock)  # restore after the test
    assert storage.use_soft_hf_locks() is _fixes
    assert _fixes.FileLock is storage.soft_lock


def test_soft_lock_type_and_arguments(tmp_path):
    lock = storage.soft_lock(tmp_path / "a.lock", timeout=3)
    assert isinstance(lock, SoftFileLock)
    assert lock.timeout == 3
    assert lock.lock_file == str(tmp_path / "a.lock")


def test_soft_lock_ignores_mode_so_no_fchmod(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "fchmod", lambda *a: calls.append(a) or (_ for _ in ()).throw(OSError(95, "no")))
    with storage.soft_lock(tmp_path / "b.lock", timeout=1, mode=0o664):
        assert (tmp_path / "b.lock").exists()
    assert calls == []


def test_soft_lock_mode_none_and_positional_timeout(tmp_path):
    lock = storage.soft_lock(tmp_path / "c.lock", 2, mode=None)
    assert lock.timeout == 2


def test_weak_file_lock_takes_the_same_lock_two_times(tmp_path, monkeypatch):
    """The hub takes one lock path 2 times for each download; the soft lock allows that and removes the file."""
    from huggingface_hub.utils import _fixes

    monkeypatch.setattr(_fixes, "FileLock", _fixes.FileLock)
    storage.use_soft_hf_locks()
    lock_path = tmp_path / "x.lock"
    for _ in range(2):
        with _fixes.WeakFileLock(lock_path) as lock:
            assert isinstance(lock, SoftFileLock)
            assert lock_path.exists()
    assert not lock_path.exists()
