"""More tests of task 1.3: the LangSAM adapter with a fake samgeo, the loader, and config checks.

D22: no real model. The fake `samgeo.text_sam` module replaces the real one in `sys.modules`.
DS-30, DS-11: the config section `label` (D51) and the run helpers.
"""
import sys
import types

import numpy as np
import pytest

import test_label as base
from semseg import config as semseg_config
from semseg import label, segment


class FakeLangSAM:
    made = []

    def __init__(self, model_type=None):
        self.model_type = model_type
        self.calls = []
        self.result = None
        FakeLangSAM.made.append(self)

    def predict(self, image, prompt, box_threshold, text_threshold, return_results=False):
        self.calls.append((image.size, prompt, box_threshold, text_threshold, return_results))
        return self.result, None, None, None


class FakeTensor:
    """A torch-like tensor: `.cpu()` returns a numpy array."""

    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self.array


def fake_samgeo(monkeypatch):
    parent = types.ModuleType("samgeo")
    parent.__path__ = []
    child = types.ModuleType("samgeo.text_sam")
    child.LangSAM = FakeLangSAM
    monkeypatch.setitem(sys.modules, "samgeo", parent)
    monkeypatch.setitem(sys.modules, "samgeo.text_sam", child)


def test_to_masks_converts_none_numpy_and_tensor_results():
    """DS-30, D51 item 9: LangSAM masks become a list of bool 2-D arrays."""
    assert label._to_masks(None) == []
    three = np.zeros((2, 4, 4), dtype=np.float32)
    three[1, 1, 1] = 1.0
    out = label._to_masks(three)
    assert len(out) == 2 and out[0].dtype == np.bool_ and out[1][1, 1] and not out[0].any()
    four = FakeTensor(np.ones((3, 1, 4, 4), dtype=np.uint8))
    out = label._to_masks(four)
    assert len(out) == 3 and out[0].shape == (4, 4) and out[0].all()


def test_default_loader_builds_the_adapter_from_a_fake_langsam(monkeypatch, tmp_path):
    """D22, D51: the real adapter calls LangSAM with the model id and thresholds of the config."""
    fake_samgeo(monkeypatch)
    settings = {"model_id": "sam2-hiera-tiny", "box_threshold": 0.3, "text_threshold": 0.4, "min_overlap": 0.5}
    model = label._default_loader(settings, {})
    assert model.fake is False and model.model_id == "langsam/sam2-hiera-tiny"
    assert FakeLangSAM.made[-1].model_type == "sam2-hiera-tiny"
    rgb = np.zeros((6, 5, 3), dtype=np.uint8)
    FakeLangSAM.made[-1].result = FakeTensor(np.ones((1, 1, 6, 5), dtype=np.uint8))
    out = model.predict(rgb, "building")
    assert FakeLangSAM.made[-1].calls == [((5, 6), "building", 0.3, 0.4, True)]
    assert len(out) == 1 and out[0].shape == (6, 5) and out[0].dtype == np.bool_


def test_default_loader_refuses_when_the_fake_model_env_is_set():
    """D22, FM-71: SEMSEG_FAKE_MODEL stops the real build."""
    with pytest.raises(RuntimeError):
        label._default_loader(label._label_config({}), {segment.FAKE_ENV: "1"})


def test_main_without_loader_uses_the_default_loader(monkeypatch, tmp_path):
    """D22: `main` with no loader calls the default loader; with SEMSEG_FAKE_MODEL set it stops, nothing is built."""
    roots, config = base.demo_setup(tmp_path)
    monkeypatch.setenv(segment.FAKE_ENV, "1")
    monkeypatch.setenv("HF_HOME", str(roots["semseg"] / "models"))
    with pytest.raises(RuntimeError):
        label.main(["--demo", "--run"], roots=roots, config=config, probe=base.Probe(30000), now=base.NOW)


def test_main_demo_section_must_be_a_mapping(tmp_path):
    """FM-84: a config with no demo mapping raises ValueError before any load."""
    roots, config = base.demo_setup(tmp_path)
    loader = base.Loader()
    for bad in (None, "x"):
        with pytest.raises(ValueError):
            label.main(["--demo", "--run"], roots=roots, config={**config, "demo": bad},
                       probe=base.Probe(30000), loader=loader, now=base.NOW)
    assert loader.calls == 0


def test_run_labels_without_chunks_is_refused(tmp_path):
    """D54: an empty chunk list raises ValueError."""
    roots = base.make_roots(tmp_path)
    with pytest.raises(ValueError):
        label.run_labels([], base.FakeText(), roots, base.FOLDER, base.CFG, run_id=base.LRUN,
                         probe=base.Probe(30000))


def test_labelled_writer_refuses_label_metadata_that_is_not_a_mapping(tmp_path):
    """D54 item 4: label metadata must be a mapping."""
    roots = base.make_roots(tmp_path)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(base.labelled(base.base_polygons()), roots, base.FOLDER, base.CHUNK,
                                      base.META, "x")


def test_label_config_section_of_a_config_file_is_checked(tmp_path):
    """DS-30, D51: load_config refuses a bad `label` section and names the file; a good one loads."""
    good = tmp_path / "good.yaml"
    head = "share_root: /x\naois: [1]\n"
    good.write_text(head + "label:\n  min_overlap: 0.4\n", encoding="utf-8")
    assert semseg_config.load_config(good)["label"]["min_overlap"] == 0.4
    for body in ("label:\n  min_overlap: 2\n", "label: x\n", "label:\n  box_threshold: 0\n"):
        bad = tmp_path / "bad.yaml"
        bad.write_text(head + body, encoding="utf-8")
        with pytest.raises(semseg_config.ConfigError, match="bad.yaml"):
            semseg_config.load_config(bad)
    none = tmp_path / "none.yaml"
    none.write_text(head + "label:\n", encoding="utf-8")
    assert semseg_config.label_settings(semseg_config.load_config(none)["label"]) == semseg_config.LABEL_DEFAULTS


def test_d57_segment_run_writes_the_naip_year_of_the_chunk_file(tmp_path):
    """D57, DS-5, DS-21: the polygon file `naip_year` is the `years` tag of the NAIP chunk, not a constant."""
    import pyogrio
    import test_segment as seg
    roots = seg.make_roots(tmp_path)
    path = seg.write_tif(roots["fgt"] / seg.FOLDER / "naip" / f"{seg.CHUNK}.tif")
    import rasterio
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(years="2019")
    segment.run_chunks([seg.CHUNK], seg.FakeModel(), roots, seg.FOLDER, seg.CFG, run_id=seg.RUN,
                       probe=seg.Probe(30000, 29000))
    gpkg = roots["semseg"] / seg.FOLDER / "polygons" / f"{seg.CHUNK}.gpkg"
    assert pyogrio.read_info(gpkg, layer="polygons")["layer_metadata"]["naip_year"] == "2019"
