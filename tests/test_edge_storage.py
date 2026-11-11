"""Edge-hunter tests for task 0.3 (storage and setup_check). Fake roots in tmp_path only.

A test marked `xfail(strict=True)` shows an open fault. The edges table is in
`.factory/0.3/edges.md`.
"""
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from semseg import setup_check, storage

GPU_OK = "NVIDIA GB10, [N/A], [N/A], [N/A], 96\n"


def make_roots(base, with_ro=True):
    roots = {"share": base / "share", "semseg": base / "share" / "semseg"}
    if with_ro:
        roots["fgt"] = base / "share" / "fgt"
        roots["stack"] = base / "stack"
    for p in roots.values():
        p.mkdir(parents=True)
    return roots


def tree(base):
    return sorted(str(p.relative_to(base)) for p in base.rglob("*"))


def planted(roots):
    """A read-only file in the stack with known content."""
    victim = roots["stack"] / "victim.tif"
    victim.write_bytes(b"ORIGINAL")
    return victim


# ------------------------------------------------------------------ read-only roots

def test_edge_symlink_chain_into_stack_refused(tmp_path):
    roots = make_roots(tmp_path)
    victim = planted(roots)
    (roots["semseg"] / "b").symlink_to(roots["stack"])
    (roots["semseg"] / "a").symlink_to(roots["semseg"] / "b")
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "a" / "victim.tif", b"X", roots)
    assert victim.read_bytes() == b"ORIGINAL"


def test_edge_dangling_symlink_target_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "dang").symlink_to(roots["stack"] / "not_yet.tif")
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "dang", b"X", roots)
    assert not (roots["stack"] / "not_yet.tif").exists()


def test_edge_new_target_under_symlinked_parent_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "lnk").symlink_to(roots["fgt"])
    before = tree(tmp_path)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "lnk" / "new" / "deep" / "f.bin", b"X", roots)
    assert tree(tmp_path) == before


def test_edge_dotdot_after_symlink_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["stack"] / "sub").mkdir()
    (roots["semseg"] / "lnk").symlink_to(roots["stack"] / "sub")
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "lnk" / ".." / "f.bin", b"X", roots)
    assert not (roots["stack"] / "f.bin").exists()


def test_edge_hardlink_in_semseg_keeps_stack_inode(tmp_path):
    roots = make_roots(tmp_path)
    victim = planted(roots)
    link = roots["semseg"] / "hard.tif"
    os.link(victim, link)
    storage.safe_write(link, b"NEW", roots)
    assert victim.read_bytes() == b"ORIGINAL"
    assert link.read_bytes() == b"NEW"


def test_edge_symlink_to_stack_file_is_replaced_not_followed(tmp_path):
    roots = make_roots(tmp_path)
    victim = planted(roots)
    (roots["semseg"] / "l.tif").symlink_to(victim)
    with pytest.raises(PermissionError):
        storage.safe_write(roots["semseg"] / "l.tif", b"NEW", roots)
    assert victim.read_bytes() == b"ORIGINAL"


@pytest.mark.parametrize("sibling", ["stack2", "stack-old", "stack_", "Stack2"])
def test_edge_prefix_sibling_of_stack_not_allowed(tmp_path, sibling):
    roots = make_roots(tmp_path)
    other = tmp_path / sibling
    other.mkdir()
    with pytest.raises(PermissionError):
        storage.check_writable(other / "f", roots)


def test_edge_prefix_sibling_of_semseg_refused(tmp_path):
    roots = make_roots(tmp_path)
    other = roots["share"] / "semseg-old"
    other.mkdir()
    with pytest.raises(PermissionError):
        storage.check_writable(other / "f", roots)


def test_edge_trailing_slash_and_dot_segments_resolve(tmp_path):
    roots = make_roots(tmp_path)
    assert storage.check_writable(str(roots["semseg"]) + "/./sub//f.bin", roots)
    with pytest.raises(PermissionError):
        storage.check_writable(str(roots["stack"]) + "/", roots)
    with pytest.raises(PermissionError):
        storage.check_writable(str(roots["semseg"]) + "/", roots)    # the root itself


def test_edge_tilde_path_not_expanded_stays_relative(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    monkeypatch.setenv("HOME", str(roots["stack"]))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PermissionError):
        storage.check_writable("~/f.bin", roots)


def test_edge_relative_path_uses_cwd(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    monkeypatch.chdir(roots["stack"])
    with pytest.raises(PermissionError):
        storage.check_writable("f.bin", roots)
    monkeypatch.chdir(roots["semseg"])
    assert storage.check_writable("f.bin", roots) == Path("f.bin")


def test_edge_str_and_path_agree(tmp_path):
    roots = make_roots(tmp_path)
    t = roots["semseg"] / "a.bin"
    assert storage.check_writable(str(t), roots) == storage.check_writable(t, roots)


def test_edge_bytes_path_fails_closed(tmp_path):
    roots = make_roots(tmp_path)
    before = tree(tmp_path)
    with pytest.raises((TypeError, PermissionError)):
        storage.safe_write(os.fsencode(str(roots["semseg"] / "a.bin")), b"X", roots)
    assert tree(tmp_path) == before


def test_edge_roots_without_fgt_stack_still_refuse_stack(tmp_path):
    """Impl deviation: `load_roots` needs only share and semseg. A real stack path is refused."""
    roots = make_roots(tmp_path, with_ro=False)
    stack = tmp_path / "stack"
    stack.mkdir()
    fgt = roots["share"] / "fgt"
    fgt.mkdir()
    for target in (stack / "f", fgt / "f"):
        with pytest.raises(PermissionError):
            storage.check_writable(target, roots)


def test_edge_load_roots_registry_without_ro_roots_loads(tmp_path):
    reg = tmp_path / "r.toml"
    reg.write_text('[roots]\nshare = "/a"\nsemseg = "{share}/s"\n', encoding="utf-8")
    assert sorted(storage.load_roots(reg, required=("share", "semseg"))) == ["semseg", "share"]


def test_edge_registry_without_ro_roots_refused_at_load(tmp_path):
    reg = tmp_path / "r.toml"
    reg.write_text('[roots]\nshare = "/a"\nsemseg = "{share}/s"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        storage.load_roots(reg)


def test_edge_load_roots_unknown_user_tilde(tmp_path):
    reg = tmp_path / "r.toml"
    reg.write_text('[roots]\nshare = "/a"\nsemseg = "{share}/s"\n'
                   'stack = "~no_such_user_zzz/x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown user"):
        storage.load_roots(reg, required=("share", "semseg"))


def test_edge_load_roots_non_string_root(tmp_path):
    reg = tmp_path / "r.toml"
    reg.write_text('[roots]\nshare = 5\nsemseg = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        storage.load_roots(reg, required=("share", "semseg"))


# ------------------------------------------------------------------ share guard

def test_edge_guard_empty_share_dir_exit_3(tmp_path):
    """Unmounted GVFS mount point: an empty dir. The semseg folder is missing."""
    share = tmp_path / "mnt"
    share.mkdir()
    with pytest.raises(SystemExit) as exc:
        storage.guard_share({"share": share, "semseg": share / "semseg"})
    assert exc.value.code == 3
    assert list(share.iterdir()) == []


def test_edge_guard_file_in_place_of_dir_exit_3(tmp_path):
    share = tmp_path / "mnt"
    share.write_text("x")
    with pytest.raises(SystemExit) as exc:
        storage.guard_share({"share": share, "semseg": share / "semseg"})
    assert exc.value.code == 3


def test_edge_guard_semseg_is_file_exit_3(tmp_path):
    share = tmp_path / "mnt"
    share.mkdir()
    (share / "semseg").write_text("x")
    with pytest.raises(SystemExit) as exc:
        storage.guard_share({"share": share, "semseg": share / "semseg"})
    assert exc.value.code == 3


def test_edge_guard_no_permission_exit_3(tmp_path):
    if os.getuid() == 0:
        pytest.skip("root ignores mode bits")
    roots = make_roots(tmp_path)
    roots["semseg"].chmod(0)
    try:
        with pytest.raises(SystemExit) as exc:
            storage.guard_share(roots)
        assert exc.value.code == 3
    finally:
        roots["semseg"].chmod(0o755)


def test_edge_guard_twice_same_result_no_side_effect(tmp_path):
    roots = make_roots(tmp_path)
    before = tree(tmp_path)
    storage.guard_share(roots)
    storage.guard_share(roots)
    assert tree(tmp_path) == before


def test_edge_guard_missing_key_is_keyerror_not_exit_3(tmp_path):
    with pytest.raises(KeyError):
        storage.guard_share({"share": tmp_path})


def test_edge_safe_write_share_missing_creates_nothing(tmp_path):
    share = tmp_path / "gone" / "share"
    roots = {"share": share, "semseg": share / "semseg"}
    with pytest.raises(SystemExit):
        storage.safe_write(share / "semseg" / "a" / "f.bin", b"X", roots)
    assert not (tmp_path / "gone").exists()


def test_edge_share_vanishes_after_guard_not_recreated(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    real_guard = storage.guard_share

    def guard_then_unmount(r):
        real_guard(r)
        shutil.rmtree(r["share"])           # the mount goes away after the probe

    monkeypatch.setattr(storage, "guard_share", guard_then_unmount)
    with pytest.raises((SystemExit, OSError)):
        storage.safe_write(roots["semseg"] / "sub" / "f.bin", b"X", roots)
    assert not roots["share"].exists()


# ------------------------------------------------------------------ safe write

def test_edge_replace_error_removes_part_and_temp(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    tmp = tmp_path / "loc"
    tmp.mkdir()

    def boom(a, b):
        raise OSError(18, "EXDEV")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        storage.safe_write(roots["semseg"] / "f.bin", b"X", roots, tmp_dir=tmp)
    assert list(roots["semseg"].iterdir()) == []
    assert list(tmp.iterdir()) == []


def test_edge_target_is_directory_cleans_up(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "d").mkdir()
    tmp = tmp_path / "loc"
    tmp.mkdir()
    with pytest.raises(OSError):
        storage.safe_write(roots["semseg"] / "d", b"X", roots, tmp_dir=tmp)
    assert [p.name for p in roots["semseg"].iterdir()] == ["d"]
    assert list(tmp.iterdir()) == []


def test_edge_leftover_part_file_untouched(tmp_path):
    """D39: the part name is new each time, so an old `<name>.part` file is not used."""
    roots = make_roots(tmp_path)
    (roots["semseg"] / "f.bin.part").write_bytes(b"OLD-LONGER-CONTENT")
    storage.safe_write(roots["semseg"] / "f.bin", b"NEW", roots)
    assert (roots["semseg"] / "f.bin").read_bytes() == b"NEW"
    assert (roots["semseg"] / "f.bin.part").read_bytes() == b"OLD-LONGER-CONTENT"


def test_edge_leftover_part_is_directory_ignored(tmp_path):
    """D39: a directory named `<name>.part` no longer blocks the write."""
    roots = make_roots(tmp_path)
    (roots["semseg"] / "f.bin.part").mkdir()
    storage.safe_write(roots["semseg"] / "f.bin", b"NEW", roots)
    assert (roots["semseg"] / "f.bin").read_bytes() == b"NEW"
    assert (roots["semseg"] / "f.bin.part").is_dir()


def test_edge_leftover_part_symlink_into_stack_not_followed(tmp_path):
    roots = make_roots(tmp_path)
    victim = planted(roots)
    (roots["semseg"] / "f.bin.part").symlink_to(victim)
    try:
        storage.safe_write(roots["semseg"] / "f.bin", b"NEW", roots)
    except (PermissionError, OSError):
        pass
    assert victim.read_bytes() == b"ORIGINAL"


def test_edge_leftover_part_hardlink_to_stack_not_written(tmp_path):
    roots = make_roots(tmp_path)
    victim = planted(roots)
    os.link(victim, roots["semseg"] / "f.bin.part")
    try:
        storage.safe_write(roots["semseg"] / "f.bin", b"NEW", roots)
    except (PermissionError, OSError):
        pass
    assert victim.read_bytes() == b"ORIGINAL"


def test_edge_two_writers_final_never_partial(tmp_path):
    roots = make_roots(tmp_path)
    target = roots["semseg"] / "f.bin"
    a_copied = threading.Event()
    b_half = threading.Event()
    a_replaced = threading.Event()
    results = {}

    def copy_a(src, dst):
        shutil.copyfile(src, dst)
        a_copied.set()
        b_half.wait(5)                       # B has truncated the same .part file

    def copy_b(src, dst):
        a_copied.wait(5)
        data = Path(src).read_bytes()
        with open(dst, "wb") as handle:     # same name: same inode that A will move
            handle.write(data[: len(data) // 2])
            handle.flush()
            b_half.set()
            a_replaced.wait(5)
            handle.write(data[len(data) // 2:])

    def run(name, data, copy):
        try:
            results[name] = storage.safe_write(target, data, roots, copy=copy)
        except BaseException as exc:                 # noqa: BLE001
            results[name] = exc

    ta = threading.Thread(target=run, args=("a", b"A" * 100, copy_a))
    tb = threading.Thread(target=run, args=("b", b"B" * 100, copy_b))
    ta.start()
    tb.start()
    ta.join(5)
    seen = target.read_bytes() if target.exists() else None
    a_replaced.set()
    tb.join(5)
    assert seen in (None, b"A" * 100, b"B" * 100), "partial file under the final name"
    if isinstance(results.get("a"), Path):
        # Checked when A is done: B may replace the file later (last writer wins, D39).
        assert seen == b"A" * 100, "A reported success, file holds other data"


def test_edge_cross_filesystem_temp_dir_ok(tmp_path):
    shm = Path("/dev/shm")
    if not shm.is_dir() or os.stat(shm).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("no second filesystem")
    roots = make_roots(tmp_path)
    storage.safe_write(roots["semseg"] / "f.bin", b"X" * 1000, roots, tmp_dir=shm)
    assert (roots["semseg"] / "f.bin").read_bytes() == b"X" * 1000
    assert not list(shm.glob("semseg_*"))


def test_edge_non_ascii_names(tmp_path):
    roots = make_roots(tmp_path)
    for name in ("地図.tif", "карта.tif", "é.tif", "🌲.bin", "‮evil.tif"):
        storage.safe_write(roots["semseg"] / name, b"X", roots)
        assert (roots["semseg"] / name).read_bytes() == b"X"


@pytest.mark.parametrize("name", ["", ".", "..", "a\nb", "a\x00b", "CON", "aux.tif", "nul .", "a\\b"])
def test_edge_refused_name_leaves_nothing(tmp_path, name):
    roots = make_roots(tmp_path)
    tmp = tmp_path / "loc"
    tmp.mkdir()
    before = tree(tmp_path)
    with pytest.raises((ValueError, PermissionError)):
        storage.safe_write(roots["semseg"] / name, b"X", roots, tmp_dir=tmp)
    assert tree(tmp_path) == before


def test_edge_name_too_long_cleans_up(tmp_path):
    roots = make_roots(tmp_path)
    tmp = tmp_path / "loc"
    tmp.mkdir()
    with pytest.raises(OSError):
        storage.safe_write(roots["semseg"] / ("a" * 252), b"X", roots, tmp_dir=tmp)
    assert list(tmp.iterdir()) == []
    assert list(roots["semseg"].iterdir()) == []


@pytest.mark.parametrize("folder", ["CON", "aux", "a\x01b", "nul ."])
def test_edge_parent_folder_name_checked(tmp_path, folder):
    roots = make_roots(tmp_path)
    with pytest.raises((ValueError, PermissionError)):
        storage.safe_write(roots["semseg"] / folder / "f.bin", b"X", roots)
    assert list(roots["semseg"].iterdir()) == []


def test_edge_failed_write_leaves_empty_parent_dirs(tmp_path):
    """Document the state: a failed write keeps the new parent folders (harmless)."""
    roots = make_roots(tmp_path)

    def bad(src, dst):
        raise OSError("disk full")

    with pytest.raises(OSError):
        storage.safe_write(roots["semseg"] / "x" / "f.bin", b"X", roots, copy=bad)
    assert not (roots["semseg"] / "x" / "f.bin").exists()
    assert not (roots["semseg"] / "x" / "f.bin.part").exists()


def test_edge_bytearray_and_memoryview(tmp_path):
    roots = make_roots(tmp_path)
    storage.safe_write(roots["semseg"] / "a", bytearray(b"AB"), roots)
    with pytest.raises(TypeError):
        storage.safe_write(roots["semseg"] / "b", memoryview(b"AB"), roots)
    assert not (roots["semseg"] / "b").exists()


# ------------------------------------------------------------------ HF_HOME

def test_edge_hf_home_symlink_into_semseg_to_outside_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"] / "models").symlink_to(roots["stack"])
    with pytest.raises(ValueError):
        storage.ensure_hf_home(roots, {"HF_HOME": str(roots["semseg"] / "models")})


def test_edge_hf_home_semseg_root_itself_and_sibling(tmp_path):
    roots = make_roots(tmp_path)
    env = {"HF_HOME": str(roots["semseg"])}
    assert storage.ensure_hf_home(roots, env) == str(roots["semseg"])
    (roots["share"] / "semseg2").mkdir()
    with pytest.raises(ValueError):
        storage.ensure_hf_home(roots, {"HF_HOME": str(roots["share"] / "semseg2")})


def test_edge_hf_home_empty_string_is_set_to_models(tmp_path):
    roots = make_roots(tmp_path)
    env = {"HF_HOME": ""}
    assert storage.ensure_hf_home(roots, env) == str(roots["semseg"] / "models")
    assert env["HF_HOME"] == str(roots["semseg"] / "models")


def test_edge_hf_home_relative_refused(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    monkeypatch.chdir(roots["semseg"])
    with pytest.raises(ValueError):
        storage.ensure_hf_home(roots, {"HF_HOME": "models"})


# ------------------------------------------------------------------ setup_check

class Run:
    def __init__(self, gpu=GPU_OK, apps="", free=None, code=0):
        self.gpu, self.apps, self.free, self.code = gpu, apps, free, code
        self.kwargs = []

    def __call__(self, cmd, *a, **k):
        self.kwargs.append(k)
        if self.code:
            return subprocess.CompletedProcess(cmd, self.code, "", "")
        if cmd[0] == "free":
            return subprocess.CompletedProcess(cmd, 0, self.free or "", "")
        out = self.apps if any("compute-apps" in c for c in cmd) else self.gpu
        return subprocess.CompletedProcess(cmd, 0, out, "")


def test_edge_gpu_exit_nonzero_and_garbage_and_empty():
    assert setup_check.query_gpu(Run(code=9))["found"] is False
    for text in ("", "\n\n", "No devices were found\n", "a,b\n", "x,y,z,w,v,u\n"):
        assert setup_check.query_gpu(Run(gpu=text))["found"] is False


def test_edge_gpu_several_and_na_and_crlf():
    text = "A, 100, 10, 90, 5\r\nB, [N/A], [N/A], [N/A], [N/A]\r\n"
    out = setup_check.query_gpu(Run(gpu=text))
    assert [g["name"] for g in out["gpus"]] == ["A", "B"]
    assert out["gpus"][1]["memory_total_mb"] is None
    assert out["gpus"][0]["memory_free_mb"] == 90


def test_edge_gpu_apps_bad_rows_skipped():
    apps = "[N/A], x, 5\n12, name, [N/A]\n13, a, b, c\n"
    out = setup_check.query_gpu(Run(apps=apps))
    assert out["apps"] == [{"pid": 12, "process_name": "name", "used_memory_mb": None}]


def test_edge_gpu_timeout_is_not_found():
    def slow(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, 30)

    assert setup_check.query_gpu(slow)["found"] is False
    assert setup_check.read_memory(slow)["total_mb"] is None


def test_edge_free_runs_in_c_locale():
    german = ("               gesamt    benutzt    frei\n"
              "Speicher:      124546       97683   1000       100       25863       26863\n")
    english = ("               total        used        free      shared  buff/cache   available\n"
               "Mem:          124546       97683        1000         100       25863       26863\n")

    def run(cmd, *a, **k):
        env = k.get("env") or {}
        text = english if env.get("LC_ALL") == "C" else german
        return subprocess.CompletedProcess(cmd, 0, text, "")

    assert setup_check.read_memory(run)["total_mb"] == 124546


def test_edge_free_old_format_six_columns_gives_none():
    old = "Mem:   100 50 50 0 0 0\n"
    assert setup_check.read_memory(Run(free=old))["available_mb"] == 0
    short = "Mem:   100 50 50 0 0\n"
    assert setup_check.read_memory(Run(free=short))["available_mb"] is None


def _fgt(base, pilot=77, names=()):
    """Fake fgt root `base` with a pilot folder and NAIP files. Dirs end with a slash."""
    naip = base / f"{pilot}_NAME" / "naip"
    naip.mkdir(parents=True, exist_ok=True)
    for n in names:
        if n.endswith("/"):
            (naip / n.rstrip("/")).mkdir()
        else:
            (naip / n).write_bytes(b"")
    return base


def test_edge_naip_negative_col_not_counted(tmp_path):
    fgt = _fgt(tmp_path / "fgt", 77, ["r1_c-1.tif"])
    assert setup_check.count_naip(fgt)["77"] == 0


def test_edge_naip_missing_pilot_and_empty_and_extra(tmp_path):
    fgt = _fgt(tmp_path / "fgt", 77, ["r0_c0.tif", "notes.txt", "r1_c1.tif.part", "R1_C1.TIF",
                              "r1_c1.tiff", "x_r1_c1.tif", "r1_c1.tif\n"])
    assert setup_check.count_naip(fgt) == {"77": 1, "392": 0}
    assert setup_check.count_naip(tmp_path / "nowhere") == {"77": 0, "392": 0}


@pytest.mark.parametrize("bad", ["r5_c5.tif/", "r01_c02.tif", "r٥_c٦.tif"])
def test_edge_naip_count_only_standard_chunk_files(tmp_path, bad):
    fgt = _fgt(tmp_path / "fgt", 77, ["r0_c0.tif", bad])
    assert setup_check.count_naip(fgt)["77"] == 1


def test_edge_layers_missing_pilot_ok_false(tmp_path):
    out = setup_check.check_layers(tmp_path)
    assert out["77"]["ok"] is False and out["77"]["nlcd"] == 0


def _full_stack(tmp_path):
    stack = tmp_path / "stack"
    for pid in (77, 392):
        folder = stack / f"{pid}_N"
        folder.mkdir(parents=True)
        for kind in setup_check.KINDS:
            for year in setup_check.YEARS:
                (folder / f"{kind}_{year}.tif").write_bytes(b"")
        (folder / "boundary.gpkg").write_bytes(b"")
        (folder / "vectors.gpkg").write_bytes(b"")
    return stack


def _report_roots(tmp_path):
    roots = make_roots(tmp_path)
    roots["stack"] = _full_stack(tmp_path)
    return roots


def test_edge_pass_false_when_one_layer_missing(tmp_path):
    roots = _report_roots(tmp_path)
    (roots["stack"] / "392_N" / "tcc_2000.tif").unlink()
    naip = {str(p): n for p, n in setup_check.EXPECTED_NAIP.items()}
    rep = setup_check.build_report(roots, run=Run(), expected_naip=naip, environ={})
    assert rep["pass"] is False and rep["layers"]["392"]["ok"] is False


def test_edge_pass_false_when_no_gpu_even_if_rest_ok(tmp_path):
    roots = _report_roots(tmp_path)
    for pid, n in setup_check.EXPECTED_NAIP.items():
        _fgt(roots["fgt"], int(pid), [f"r{i}_c0.tif" for i in range(n)])
    rep = setup_check.build_report(roots, run=Run(code=9), environ={})
    assert rep["naip"] == setup_check.EXPECTED_NAIP
    assert rep["pass"] is False and rep["share_ok"] is True
    rep = setup_check.build_report(roots, run=Run(), environ={})
    assert rep["pass"] is True


def test_edge_pass_false_when_naip_one_short(tmp_path):
    roots = _report_roots(tmp_path)
    for pid, n in setup_check.EXPECTED_NAIP.items():
        _fgt(roots["fgt"], int(pid), [f"r{i}_c0.tif" for i in range(n - (pid == "392"))])
    rep = setup_check.build_report(roots, run=Run(), environ={})
    assert rep["pass"] is False and rep["naip_total"] == 383


def test_edge_bad_hf_home_writes_failure_report_and_replaces_stale_pass(tmp_path):
    roots = _report_roots(tmp_path)
    out = tmp_path / "res" / "o.json"
    out.parent.mkdir()
    out.write_text('{"pass": true}\n', encoding="utf-8")
    code = setup_check.main(["--out", str(out)], roots=roots, run=Run(),
                            environ={"HF_HOME": str(tmp_path / "elsewhere")})
    assert code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pass"] is False and "ValueError" in data["reason"]
    assert data["share_ok"] is False


def test_edge_bad_hf_home_writes_failure_report_without_old_file(tmp_path):
    roots = _report_roots(tmp_path)
    out = tmp_path / "res" / "o.json"
    out.parent.mkdir()
    code = setup_check.main(["--out", str(out)], roots=roots, run=Run(),
                            environ={"HF_HOME": "relative/hf"})
    assert code == 1 and json.loads(out.read_text(encoding="utf-8"))["pass"] is False


@pytest.mark.parametrize("where", ["stack", "fgt"])
def test_edge_out_under_readonly_root_refused(tmp_path, where):
    roots = _report_roots(tmp_path)
    out = roots[where] / "new_dir" / "setup_check.json"
    try:
        setup_check.main(["--out", str(out)], roots=roots, run=Run(), environ={})
    except (PermissionError, SystemExit, ValueError):
        pass
    assert not out.exists()
    assert not out.parent.exists()


def test_edge_failed_run_does_not_leave_stale_pass(tmp_path):
    roots = make_roots(tmp_path)
    out = tmp_path / "res" / "setup_check.json"
    out.parent.mkdir()
    out.write_text('{"pass": true}\n', encoding="utf-8")
    shutil.rmtree(roots["share"])
    with pytest.raises(SystemExit):
        setup_check.main(["--out", str(out)], roots=roots, run=Run(), environ={})
    assert not out.exists() or '"pass": true' not in out.read_text(encoding="utf-8")


def test_edge_write_report_lf_endings_and_utf8(tmp_path):
    p = setup_check.write_report({"name": "地図 "}, tmp_path / "o.json")
    raw = p.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert list(tmp_path.glob("*.tmp")) == []


def test_edge_write_report_nan_serialises_but_is_invalid_json(tmp_path):
    p = setup_check.write_report({"x": float("nan")}, tmp_path / "o.json")
    assert "NaN" in p.read_text(encoding="utf-8")
