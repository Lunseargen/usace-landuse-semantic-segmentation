"""Environment checks for task 0.1 (FM-14, FM-15, FM-17). No GPU use (D22)."""


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
