"""Storage roots, share guard, read-only roots and safe write (DS-22 to DS-26, DS-28).

Rules (task 0.3, D23, D36, D38):
- The roots come from `data/REGISTRY.toml` `[roots]`. Loading touches no disk.
- A missing or stale share stops the step with exit code 3. No code creates the share root
  and no write goes to another place (FM-5, FM-41).
- Only `semseg` takes writes. `fgt` and `stack` refuse them, also through a link, `..`,
  a prefix sibling or a case variant (FM-38).
- A write goes to a local temp file, then to `<name>.<8 hex>.part` on the target filesystem
  (a new file: `O_EXCL`, `O_NOFOLLOW`), then `os.replace` (DS-25, FM-39, D39).
- No code creates the semseg root or a folder above it (FM-44).
"""
import os
import secrets
import shutil
import tempfile
import tomllib
from pathlib import Path

from semseg import standard
from semseg.config import load_config

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "REGISTRY.toml"
REQUIRED_ROOTS = ("share", "semseg", "fgt", "stack")      # FM-43: all four unless `required=`
KNOWN_ROOTS = ("share", "semseg", "fgt", "stack")


def _expand(name, text, share):
    """Return one registry root as an absolute `Path`. Raise ValueError if it is not valid."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"root {name!r} is empty or not a string")
    if share is not None:
        text = text.replace("{share}", share)
    if "{" in text or "}" in text:
        raise ValueError(f"root {name!r} has an unexpanded template: {text!r}")
    try:
        path = Path(text).expanduser()
    except RuntimeError:                      # FM-53: `~unknownuser`
        raise ValueError(f"root {name!r} has an unknown user: {text!r}") from None
    if not path.is_absolute():
        raise ValueError(f"root {name!r} is not an absolute path: {text!r}")
    if ".." in path.parts:
        raise ValueError(f"root {name!r} has a '..' part: {text!r}")
    return path


def load_roots(registry=None, config=None, *, required=REQUIRED_ROOTS):
    """Return `{name: Path}` for share, semseg, and fgt and stack when present. It does not touch the disk.

    Raise ValueError for a missing, relative, `..` or template root, and (D36) when the
    `share_root` of the config differs from the `semseg` root. By default all four roots
    are required, so a loaded `roots` always refuses a write to fgt and stack (DS-23,
    FM-43). A caller can pass a shorter `required=`.
    """
    registry = Path(registry) if registry is not None else REGISTRY_PATH
    with open(registry, "rb") as handle:
        table = tomllib.load(handle).get("roots", {})
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError(f"{registry}: [roots] misses {missing}")
    share = _expand("share", table["share"], None)
    roots = {"share": share}
    for name in KNOWN_ROOTS[1:]:
        if name in table:
            roots[name] = _expand(name, table[name], str(share))
    if config is not None:
        wanted = Path(load_config(config)["share_root"])
        if wanted != roots["semseg"]:
            raise ValueError(
                f"D36: config share_root {str(wanted)!r} differs from "
                f"registry semseg root {str(roots['semseg'])!r}")
    return roots


def guard_share(roots):
    """Stop with exit code 3 if the share or the semseg root is missing or stale (DS-26).

    The probe is `os.listdir`. It never creates a folder.
    """
    for name in ("share", "semseg"):
        try:
            os.listdir(roots[name])
        except OSError:
            raise SystemExit(standard.EXIT_SHARE_MISSING) from None


def _fold(path):
    """Return the resolved path (links and `..` followed) with case folded."""
    return Path(os.path.realpath(path)).as_posix().casefold()


def _inside(path, root, *, strict=False):
    """True if `path` is under `root` after resolving links and `..`, ignoring case."""
    path, root = Path(_fold(path)), Path(_fold(root))
    return path.is_relative_to(root) and not (strict and path == root)


def check_not_readonly(path, roots):
    """Raise PermissionError if `path` is under a read-only root (links and `..` followed)."""
    for name, mode in standard.ROOTS.items():
        if mode == "ro" and name in roots and _inside(path, roots[name]):
            raise PermissionError(f"{name} is a read-only root: {path}")


def check_writable(path, roots):
    """Return `path` if a write is allowed. Raise PermissionError otherwise (DS-23, FM-38).

    Only a place strictly under the `semseg` root is allowed. `realpath` follows a link at
    any level, also the last one, so a link to a place outside is refused.
    """
    check_not_readonly(path, roots)
    if not _inside(path, roots["semseg"], strict=True):
        raise PermissionError(f"not under the semseg root: {path}")
    return Path(path)


def _remove(name):
    try:
        os.remove(name)
    except OSError:
        pass


def _make_parents(real_parent, roots):
    """Create the missing folders of `real_parent`, one level at a time, below the semseg root.

    The semseg root and anything above it are never created (FM-44). Each new name passes
    `check_name` before the first folder is made (FM-47).
    """
    base = Path(os.path.realpath(roots["semseg"]))
    try:
        parts = real_parent.relative_to(base).parts
    except ValueError:
        raise PermissionError(f"not under the semseg root: {real_parent}") from None
    current, index = base, 0
    while index < len(parts) and (current / parts[index]).exists():
        current = current / parts[index]
        index += 1
    for part in parts[index:]:
        standard.check_name(part)
    for part in parts[index:]:
        current = current / part
        current.mkdir(exist_ok=True)


def safe_write(target, data, roots, *, copy=shutil.copyfile, tmp_dir=None):
    """Write `data` (bytes) to `target` under the semseg root (DS-25, D39). Return the path.

    Local temp file, then a new `<name>.<8 hex>.part` in the folder of the target, then
    `os.replace`. The guard runs again right before the replace. The temp file and the own
    `.part` file go on any error.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    guard_share(roots)
    target = check_writable(target, roots)
    standard.check_name(target.name)
    _make_parents(Path(os.path.realpath(target)).parent, roots)
    part = target.parent / f"{target.name}.{secrets.token_hex(4)}{standard.PART_SUFFIX}"
    fd, tmp = tempfile.mkstemp(dir=tmp_dir, prefix="semseg_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        os.close(os.open(part, flags, 0o666))           # a new file: never a link (FM-45)
        try:
            copy(tmp, part)
            guard_share(roots)                          # FM-44: the share may be gone now
            os.replace(part, target)
        except BaseException:
            _remove(part)
            raise
    finally:
        _remove(tmp)
    return target


def ensure_hf_home(roots, environ=None):
    """Set `HF_HOME` to `<semseg>/models` if unset. Return the value (D38, DS-28).

    Raise ValueError if `HF_HOME` is relative (FM-48) or is set to a place outside the semseg root. It imports
    no Hugging Face module.
    """
    environ = os.environ if environ is None else environ
    guard_share(roots)
    current = environ.get("HF_HOME")
    if not current:
        current = str(standard.models_dir(roots["semseg"]))
        environ["HF_HOME"] = current
    elif not os.path.isabs(current):
        raise ValueError(f"HF_HOME is not an absolute path: {current}")
    elif not _inside(current, roots["semseg"]):
        raise ValueError(f"HF_HOME is outside the semseg root: {current}")
    return current


def use_soft_hf_locks(module=None):
    """Make the Hugging Face hub use `filelock.SoftFileLock` for its cache locks. Return the module (D58).

    The GVFS (SMB) share refuses `os.open(O_RDWR)` on an existing file and `fchmod` (errno 95).
    `FileLock` opens the lock file again on the second lock, so each download on the share fails.
    `SoftFileLock` uses only `O_CREAT | O_EXCL`; `soft_lock` drops `mode`, so no `fchmod` occurs.
    `module` is `huggingface_hub.utils._fixes` by default; tests give a stand-in object.
    """
    if module is None:
        from huggingface_hub.utils import _fixes as module
    module.FileLock = soft_lock
    return module


def soft_lock(lock_file, *args, mode=None, **kwargs):
    """Return a `filelock.SoftFileLock` for `lock_file`. Ignore `mode`: the share refuses `fchmod` (D58)."""
    from filelock import SoftFileLock

    return SoftFileLock(lock_file, *args, **kwargs)
