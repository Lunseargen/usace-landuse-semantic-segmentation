"""Storage module tests (task 0.3: R5, FM-5, FM-36, FM-38 to FM-41, D36, D38).

All write tests use fake roots in tmp_path. No test writes to the real share or stack.
API under test (semseg.storage):
  load_roots(registry=None, config=None) -> {name: Path}      (no disk access)
  guard_share(roots) -> None | SystemExit(3)
  check_writable(path, roots) -> Path | PermissionError
  safe_write(target, data, roots, *, copy=shutil.copyfile, tmp_dir=None) -> Path
  ensure_hf_home(roots, environ=None) -> str
"""
import errno
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from semseg import standard
from semseg import storage

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "REGISTRY.toml"
CONFIG = ROOT / "config.yaml"


def make_roots(base):
    roots = {
        "share": base / "share",
        "semseg": base / "share" / "semseg",
        "fgt": base / "share" / "fgt",
        "stack": base / "stack",
    }
    for p in roots.values():
        p.mkdir(parents=True)
    return roots


def snapshot(base):
    return sorted(str(p.relative_to(base)) for p in base.rglob("*"))


MIN_ROOTS = ("share", "semseg")      # FM-43: these registries have 2 keys


def write_registry(path, **roots):
    lines = ["[roots]"] + [f'{k} = "{v}"' for k, v in roots.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_config(path, share_root):
    path.write_text(f"share_root: {share_root}\naois: [77, 392]\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- load_roots (D36, FM-36)

def test_d36_real_registry_roots_load():
    roots = storage.load_roots()
    for name in ("share", "semseg", "fgt", "stack"):
        assert isinstance(roots[name], Path) and roots[name].is_absolute()
    assert roots["semseg"] == roots["share"] / "usace-landuse-semseg"
    assert roots["fgt"] == roots["share"] / "usace-landuse" / "fine_grid_test"
    assert "{" not in str(roots["semseg"])


def test_d36_stack_tilde_expanded():
    roots = storage.load_roots()
    assert roots["stack"] == Path("~/workspace/usace-landuse/USACE/01_Data/stack").expanduser()


def test_d36_config_share_root_equals_semseg_root():
    from semseg.config import load_config
    roots = storage.load_roots(REGISTRY, CONFIG)
    assert str(roots["semseg"]) == load_config(CONFIG)["share_root"]


def test_d36_config_mismatch_raises_value_error(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share", semseg="{share}/semseg")
    cfg = write_config(tmp_path / "c.yaml", "/a/share/other")
    with pytest.raises(ValueError):
        storage.load_roots(reg, cfg, required=MIN_ROOTS)


def test_d36_config_match_ok(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share", semseg="{share}/semseg")
    cfg = write_config(tmp_path / "c.yaml", "/a/share/semseg")
    assert storage.load_roots(reg, cfg, required=MIN_ROOTS)["semseg"] == Path("/a/share/semseg")


def test_d36_load_roots_does_not_touch_disk(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/does/not/exist", semseg="{share}/s")
    before = snapshot(tmp_path)
    storage.load_roots(reg, required=MIN_ROOTS)
    assert snapshot(tmp_path) == before
    assert not Path("/does").exists()


def test_fm36_relative_root_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share", semseg="relative/semseg")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


def test_fm36_relative_share_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="share", semseg="{share}/s")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


def test_fm36_dotdot_root_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share", semseg="{share}/../escape")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


def test_fm36_leftover_template_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share", semseg="{shre}/semseg")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


def test_fm36_leftover_share_in_share_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="{share}/x", semseg="/a/semseg")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


def test_fm36_empty_root_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share", semseg="")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


def test_fm36_missing_semseg_key_refused(tmp_path):
    reg = write_registry(tmp_path / "r.toml", share="/a/share")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=MIN_ROOTS)


# ---------------------------------------------------------------- share guard (R5, FM-5, FM-41)

def test_share_missing_exit_3(tmp_path, monkeypatch):
    base = tmp_path / "w"
    base.mkdir()
    monkeypatch.chdir(base)
    roots = {"share": base / "nope", "semseg": base / "nope" / "semseg",
             "fgt": base / "nope" / "fgt", "stack": base / "stack"}
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as exc:
        storage.guard_share(roots)
    assert exc.value.code == 3 == standard.EXIT_SHARE_MISSING
    with pytest.raises(SystemExit) as exc:
        storage.safe_write(roots["semseg"] / "a.bin", b"x", roots, tmp_dir=base)
    assert exc.value.code == 3
    assert not (base / "nope").exists()
    assert snapshot(tmp_path) == before          # nothing written anywhere (MOP: 0 local writes)


def test_share_missing_hf_home_exit_3(tmp_path):
    roots = {"share": tmp_path / "nope", "semseg": tmp_path / "nope" / "s",
             "fgt": tmp_path / "nope" / "f", "stack": tmp_path / "stack"}
    env = {}
    with pytest.raises(SystemExit) as exc:
        storage.ensure_hf_home(roots, environ=env)
    assert exc.value.code == 3
    assert "HF_HOME" not in env
    assert snapshot(tmp_path) == []


def test_share_present_guard_passes(tmp_path):
    roots = make_roots(tmp_path)
    assert storage.guard_share(roots) is None


def test_fm41_semseg_missing_share_present_exit_3(tmp_path):
    roots = make_roots(tmp_path)
    roots["semseg"].rmdir()
    with pytest.raises(SystemExit) as exc:
        storage.guard_share(roots)
    assert exc.value.code == 3
    assert not roots["semseg"].exists()


def test_fm41_guard_never_creates_share_root(tmp_path):
    roots = {"share": tmp_path / "a" / "b" / "share", "semseg": tmp_path / "a" / "b" / "share" / "s",
             "fgt": tmp_path / "x", "stack": tmp_path / "y"}
    with pytest.raises(SystemExit):
        storage.guard_share(roots)
    with pytest.raises(SystemExit):
        storage.safe_write(roots["semseg"] / "f", b"1", roots)
    with pytest.raises(SystemExit):
        storage.ensure_hf_home(roots, environ={})
    assert not (tmp_path / "a").exists()


@pytest.mark.parametrize("code", [errno.ESTALE, errno.EIO, errno.EACCES, errno.ENOTCONN])
def test_fm41_stale_mount_exit_3(tmp_path, monkeypatch, code):
    roots = make_roots(tmp_path)
    real = os.listdir
    share = str(roots["share"])

    def fake(path="."):
        if str(path).rstrip("/") in (share, str(roots["semseg"])):
            raise OSError(code, "stale")
        return real(path)

    monkeypatch.setattr(os, "listdir", fake)
    with pytest.raises(SystemExit) as exc:
        storage.guard_share(roots)
    assert exc.value.code == 3


def test_fm41_permission_error_exit_3(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    real = os.listdir

    def fake(path="."):
        if str(path).rstrip("/") in (str(roots["share"]), str(roots["semseg"])):
            raise PermissionError(errno.EACCES, "denied")
        return real(path)

    monkeypatch.setattr(os, "listdir", fake)
    with pytest.raises(SystemExit) as exc:
        storage.guard_share(roots)
    assert exc.value.code == 3


# ---------------------------------------------------------------- read-only roots (R5, FM-38)

def test_semseg_write_allowed(tmp_path):
    roots = make_roots(tmp_path)
    target = roots["semseg"] / "77_X" / "cells" / "a.parquet"
    assert Path(storage.check_writable(target, roots)) == target


def test_readonly_roots(tmp_path):
    roots = make_roots(tmp_path)
    for name in ("fgt", "stack"):
        with pytest.raises(PermissionError):
            storage.check_writable(roots[name] / "a.tif", roots)
        with pytest.raises(PermissionError):
            storage.check_writable(roots[name], roots)
        with pytest.raises(PermissionError):
            storage.safe_write(roots[name] / "a.tif", b"x", roots, tmp_dir=tmp_path)
    assert list(roots["fgt"].iterdir()) == []
    assert list(roots["stack"].iterdir()) == []


def test_readonly_nested_new_folder_refused(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["fgt"] / "new" / "deep" / "a", b"x", roots)
    assert not (roots["fgt"] / "new").exists()


def test_fm38_symlink_dir_escape_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "link").symlink_to(roots["fgt"], target_is_directory=True)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "link" / "a.bin", b"x", roots)
    assert list(roots["fgt"].iterdir()) == []


def test_fm38_symlink_to_stack_escape_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "s").symlink_to(roots["stack"], target_is_directory=True)
    with pytest.raises(PermissionError):
        storage.check_writable(roots["semseg"] / "s" / "x", roots)


def test_fm38_symlink_target_file_refused(tmp_path):
    roots = make_roots(tmp_path)
    victim = roots["fgt"] / "v.tif"
    victim.write_bytes(b"old")
    (roots["semseg"] / "l.tif").symlink_to(victim)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "l.tif", b"new", roots)
    assert victim.read_bytes() == b"old"


def test_fm38_dotdot_escape_refused(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(PermissionError):
        storage.check_writable(roots["semseg"] / ".." / "fgt" / "a", roots)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "x" / ".." / ".." / "fgt" / "a", b"x", roots)
    assert list(roots["fgt"].iterdir()) == []


def test_fm38_prefix_sibling_not_matched(tmp_path):
    roots = make_roots(tmp_path)
    (tmp_path / "stack2").mkdir()
    (roots["share"] / "semseg2").mkdir()
    # a sibling with a common prefix is not under any root: refused, and not treated as semseg
    with pytest.raises(PermissionError):
        storage.check_writable(tmp_path / "stack2" / "a", roots)
    with pytest.raises(PermissionError):
        storage.check_writable(roots["share"] / "semseg2" / "a", roots)
    assert list((tmp_path / "stack2").iterdir()) == []


def test_fm38_case_variant_refused(tmp_path):
    roots = make_roots(tmp_path)
    for variant in (roots["fgt"].parent / "FGT" / "a", roots["stack"].parent / "Stack" / "a"):
        with pytest.raises(PermissionError):
            storage.check_writable(variant, roots)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["fgt"].parent / "Fgt" / "a", b"x", roots)
    assert list(roots["fgt"].iterdir()) == []


def test_local_path_outside_roots_refused(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(PermissionError):
        storage.safe_write(tmp_path / "local.bin", b"x", roots)
    assert not (tmp_path / "local.bin").exists()


def test_semseg_root_itself_not_a_file_target(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises((PermissionError, ValueError, IsADirectoryError)):
        storage.safe_write(roots["semseg"], b"x", roots)


# ---------------------------------------------------------------- safe write (DS-25, FM-39)

def test_safe_write_happy_path(tmp_path):
    roots = make_roots(tmp_path)
    local = tmp_path / "local"
    local.mkdir()
    target = roots["semseg"] / "out.bin"
    result = storage.safe_write(target, b"hello", roots, tmp_dir=local)
    assert Path(result) == target
    assert target.read_bytes() == b"hello"
    assert sorted(p.name for p in roots["semseg"].iterdir()) == ["out.bin"]
    assert list(local.iterdir()) == []


def test_safe_write_empty_payload(tmp_path):
    roots = make_roots(tmp_path)
    storage.safe_write(roots["semseg"] / "e.bin", b"", roots, tmp_dir=tmp_path)
    assert (roots["semseg"] / "e.bin").read_bytes() == b""


def test_safe_write_replaces_existing(tmp_path):
    roots = make_roots(tmp_path)
    target = roots["semseg"] / "o.bin"
    target.write_bytes(b"old")
    storage.safe_write(target, b"new", roots, tmp_dir=tmp_path)
    assert target.read_bytes() == b"new"


def test_safe_write_uses_part_and_os_replace_in_same_folder(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    calls = []
    real = os.replace

    def spy(src, dst, *a, **k):
        calls.append((Path(src), Path(dst)))
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", spy)
    target = roots["semseg"] / "o.bin"
    storage.safe_write(target, b"1", roots, tmp_dir=tmp_path)
    # D39: a unique part name `<name>.<8 hex>.part` in the folder of the target
    assert len(calls) == 1 and calls[0][1] == target
    assert re.fullmatch(r"o\.bin\.[0-9a-f]{8}\.part", calls[0][0].name)
    assert calls[0][0].parent == target.parent


def test_safe_write_copy_gets_part_destination(tmp_path):
    roots = make_roots(tmp_path)
    seen = []

    def copy(src, dst):
        seen.append((Path(src), Path(dst)))
        return shutil.copyfile(src, dst)

    target = roots["semseg"] / "o.bin"
    storage.safe_write(target, b"1", roots, copy=copy, tmp_dir=tmp_path)
    assert re.fullmatch(r"o\.bin\.[0-9a-f]{8}\.part", seen[0][1].name)     # D39
    assert seen[0][1].parent == target.parent
    assert not seen[0][0].is_relative_to(roots["share"])      # the source is a local temp file


def test_edge_failure_during_copy_leaves_no_final_file(tmp_path):
    roots = make_roots(tmp_path)
    local = tmp_path / "local"
    local.mkdir()

    def broken(src, dst):
        Path(dst).write_bytes(b"half")
        raise OSError(errno.EIO, "share went away")

    target = roots["semseg"] / "o.bin"
    with pytest.raises(OSError):
        storage.safe_write(target, b"data", roots, copy=broken, tmp_dir=local)
    assert not target.exists()
    assert list(roots["semseg"].iterdir()) == []           # no .part either
    assert list(local.iterdir()) == []                      # local temp file removed


def test_edge_failure_keeps_old_final_file(tmp_path):
    roots = make_roots(tmp_path)
    target = roots["semseg"] / "o.bin"
    target.write_bytes(b"old")

    def broken(src, dst):
        Path(dst).write_bytes(b"ha")
        raise OSError(errno.EIO, "x")

    with pytest.raises(OSError):
        storage.safe_write(target, b"new", roots, copy=broken, tmp_dir=tmp_path)
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in roots["semseg"].iterdir()) == ["o.bin"]


def test_edge_part_removed_on_error(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)

    def boom(src, dst, *a, **k):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        storage.safe_write(roots["semseg"] / "o.bin", b"d", roots, tmp_dir=tmp_path)
    assert list(roots["semseg"].iterdir()) == []


def test_edge_keyboard_interrupt_cleans_up(tmp_path):
    roots = make_roots(tmp_path)

    def stop(src, dst):
        Path(dst).write_bytes(b"h")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        storage.safe_write(roots["semseg"] / "o.bin", b"d", roots, copy=stop, tmp_dir=tmp_path)
    assert list(roots["semseg"].iterdir()) == []


def test_edge_replace_exdev_not_used(tmp_path):
    # the .part file is next to the target, so the temp file is not the os.replace source
    roots = make_roots(tmp_path)
    local = tmp_path / "local"
    local.mkdir()
    sources = []
    real = os.replace
    orig = os.replace
    try:
        os.replace = lambda s, d, *a, **k: (sources.append(Path(s).parent), real(s, d))[1]
        storage.safe_write(roots["semseg"] / "o.bin", b"1", roots, tmp_dir=local)
    finally:
        os.replace = orig
    assert sources == [roots["semseg"]]


def test_safe_write_rejects_non_bytes(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(TypeError):
        storage.safe_write(roots["semseg"] / "o.bin", "text", roots, tmp_dir=tmp_path)
    assert list(roots["semseg"].iterdir()) == []


# ---------------------------------------------------------------- HF_HOME (D38, FM-40)

def test_hf_home_on_share(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    value = storage.ensure_hf_home(roots)
    assert value == str(roots["semseg"] / standard.MODELS_DIR)
    assert os.environ["HF_HOME"] == value


def test_hf_home_injected_environ(tmp_path):
    roots = make_roots(tmp_path)
    env = {}
    value = storage.ensure_hf_home(roots, environ=env)
    assert env["HF_HOME"] == value == str(roots["semseg"] / "models")


def test_hf_home_already_on_share_kept(tmp_path):
    roots = make_roots(tmp_path)
    mine = str(roots["semseg"] / "models" / "sub")
    env = {"HF_HOME": mine}
    assert storage.ensure_hf_home(roots, environ=env) == mine
    assert env["HF_HOME"] == mine


def test_edge_hf_home_overrides_env_value(tmp_path):
    roots = make_roots(tmp_path)
    env = {"HF_HOME": str(Path.home() / ".cache" / "huggingface")}
    with pytest.raises(ValueError):
        storage.ensure_hf_home(roots, environ=env)
    assert env["HF_HOME"].endswith("huggingface")           # not changed silently


def test_hf_home_outside_via_dotdot_raises(tmp_path):
    roots = make_roots(tmp_path)
    env = {"HF_HOME": str(roots["semseg"] / ".." / "elsewhere")}
    with pytest.raises(ValueError):
        storage.ensure_hf_home(roots, environ=env)


def test_hf_home_prefix_sibling_raises(tmp_path):
    roots = make_roots(tmp_path)
    env = {"HF_HOME": str(roots["share"] / "semseg2" / "models")}
    with pytest.raises(ValueError):
        storage.ensure_hf_home(roots, environ=env)


def test_hf_home_empty_string_treated_as_unset_or_refused(tmp_path):
    roots = make_roots(tmp_path)
    env = {"HF_HOME": ""}
    try:
        value = storage.ensure_hf_home(roots, environ=env)
    except ValueError:
        return
    assert value == str(roots["semseg"] / "models")


def test_hf_home_does_not_import_huggingface(tmp_path):
    code = (
        "import sys; from pathlib import Path; from semseg import storage;"
        f"r={{'share':Path(r'{tmp_path}'),'semseg':Path(r'{tmp_path}'),'fgt':Path(r'{tmp_path}/f'),"
        f"'stack':Path(r'{tmp_path}/s')}};"
        "storage.ensure_hf_home(r, environ={});"
        "sys.exit(1 if 'huggingface_hub' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_storage_never_touches_gpu():
    import inspect
    src = inspect.getsource(storage)
    assert "torch" not in src and "cuda" not in src.lower()
