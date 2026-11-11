"""Guard for FM-71 and D22: no test builds a real SAM 2 model or downloads weights.

The guard replaces the entry points with a function that raises `RealModelError`:
`samgeo.SamGeo2` (and `samgeo.samgeo2.SamGeo2`), `sam2.build_sam.build_sam2*`, and
`hf_hub_download` / `snapshot_download` in `huggingface_hub` and its `file_download`,
`_snapshot_download` and `hub_mixin` modules. The guard installs when this file is imported,
before the collection of any test module. An import hook applies the guard to each of these
modules right after the first import, so the order of the imports does not matter.
"""

import importlib.abc
import importlib.machinery
import sys

import pytest

GUARDED = {
    "samgeo": ("SamGeo2",),
    "samgeo.samgeo2": ("SamGeo2",),
    "sam2.build_sam": ("build_sam2", "build_sam2_video_predictor", "build_sam2_hf",
                       "build_sam2_video_predictor_hf", "_load_checkpoint", "hf_hub_download"),
    "huggingface_hub": ("hf_hub_download", "snapshot_download"),
    "huggingface_hub.file_download": ("hf_hub_download",),
    "huggingface_hub._snapshot_download": ("snapshot_download",),
    "huggingface_hub.hub_mixin": ("hf_hub_download", "snapshot_download"),
}


class RealModelError(AssertionError):
    """A test tried to build a real SAM 2 model or to download weights (D22, FM-71)."""


def _forbidden(name):
    def blocked(*args, **kwargs):
        raise RealModelError(f"{name} is forbidden in a test (D22, FM-71): use a fake model")
    blocked.__name__ = name.rsplit(".", 1)[-1]
    blocked._semseg_guard = True
    return blocked


def patch_module(module):
    """Replace the guarded names of one module. Return the list of names replaced."""
    done = []
    for attr in GUARDED.get(getattr(module, "__name__", ""), ()):
        current = getattr(module, attr, None)
        if current is not None and not getattr(current, "_semseg_guard", False):
            setattr(module, attr, _forbidden(f"{module.__name__}.{attr}"))
            done.append(attr)
    return done


class _GuardLoader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        patch_module(module)


class _GuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in GUARDED:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _GuardLoader(spec.loader)
                return spec
        return None


def install_guard():
    """Patch the modules already imported, then add the import hook. Safe to call twice."""
    for name in GUARDED:
        if name in sys.modules:
            patch_module(sys.modules[name])
    if not any(isinstance(f, _GuardFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _GuardFinder())


install_guard()  # module level: runs before the collection of the test modules (review fix)


@pytest.fixture(scope="session", autouse=True)
def no_real_sam2_model():
    install_guard()  # a second call changes nothing
    yield
