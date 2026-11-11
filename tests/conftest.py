"""Guard for FM-71, FM-89 and D22: no test builds a real SAM 2 or LangSAM model or downloads weights.

The guard replaces the entry points with a function that raises `RealModelError`:
`samgeo.SamGeo2` (and `samgeo.samgeo2.SamGeo2`), `samgeo.text_sam.LangSAM`,
`sam2.build_sam.build_sam2*`, the GroundingDINO loaders, `torch.hub` loaders, and
`hf_hub_download` / `snapshot_download` / `cached_file` in `huggingface_hub`, its `file_download`,
`_snapshot_download` and `hub_mixin` modules, and `transformers.utils.hub` (D55). The guard installs when this
file is imported, before the collection of any test module. An import hook applies the guard to
each of these modules right after the first import, so the order of the imports does not matter.
The Hugging Face offline variables are the second layer: no download.
"""

import importlib.abc
import importlib.machinery
import os
import sys

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"        # D55: second layer, no download
os.environ["TRANSFORMERS_OFFLINE"] = "1"

GUARDED = {
    "samgeo": ("SamGeo2",),
    "samgeo.samgeo2": ("SamGeo2",),
    "sam2.build_sam": ("build_sam2", "build_sam2_video_predictor", "build_sam2_hf",
                       "build_sam2_video_predictor_hf", "_load_checkpoint", "hf_hub_download"),
    "huggingface_hub": ("hf_hub_download", "snapshot_download"),
    "huggingface_hub.file_download": ("hf_hub_download",),
    "huggingface_hub._snapshot_download": ("snapshot_download",),
    "huggingface_hub.hub_mixin": ("hf_hub_download", "snapshot_download"),
    # D55 (task 1.3, FM-89): LangSAM, GroundingDINO, torch.hub and transformers
    "samgeo.text_sam": ("LangSAM", "load_model_hf", "hf_hub_download", "SamGeo2"),
    "groundingdino.util.inference": ("load_model",),
    "groundingdino.util.get_tokenlizer": ("get_tokenlizer", "get_pretrained_language_model"),
    "torch.hub": ("load_state_dict_from_url", "load", "download_url_to_file"),
    "transformers.utils.hub": ("hf_hub_download", "snapshot_download", "cached_file", "cached_files"),
}


class RealModelError(AssertionError):
    """A test tried to build a real SAM 2 or LangSAM model or to download weights (D22, FM-71, FM-89)."""


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
