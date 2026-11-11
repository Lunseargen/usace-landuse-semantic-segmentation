"""Environment checks for task 0.1 (FM-14, FM-15, FM-17). No GPU use (D22)."""
import sys

import pytest


@pytest.fixture(autouse=True)
def forget_sam2_modules():
    """Task 1.2 (FM-71): remove the SAM 2 modules this test imports, so that
    `test_edge_suite_never_imports_real_sam2` in test_segment.py stays true in the full run."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.split(".")[0] in ("samgeo", "sam2"):
            del sys.modules[name]


def test_package_importable():
    import semseg

    assert isinstance(semseg.__version__, str) and semseg.__version__


def test_import_samgeo():
    import samgeo

    assert samgeo.__version__ == "1.4.2"
    from samgeo import SamGeo2  # noqa: F401
    from samgeo.text_sam import LangSAM  # noqa: F401


def test_torch_cuda_build():
    # Reads the build string only. No tensor, no model, no device call.
    import torch

    assert torch.version.cuda is not None
    assert torch.version.cuda.startswith("13")
