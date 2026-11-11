"""Extra tests of task 1.2: config validation (D46, D47), the real-model adapter with a fake
`samgeo` module (D22: the real library is never imported), the empty file, and GPU log values."""

import json
import sys
import types
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from semseg import segment, standard
from semseg.config import load_config, non_negative_number
import test_segment as base


def cfg(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text("share_root: /x\naois: [1]\n" + text)
    return path


def test_real_config_loads_with_segment_and_demo():
    config = load_config()
    assert config["segment"]["model_id"] == "sam2-hiera-large"
    assert config["demo"]["pid"] == 77 and len(config["demo"]["chunks"]) == 6


def test_good_segment_and_demo_sections(tmp_path):
    text = ("segment: {min_free_mb: 8000, simplify_tolerance_m: 0, min_area_m2: 1.5, model_id: m}\n"
            "demo: {pid: 5, chunks: [r1_c1, r0_c2]}\n")
    assert load_config(cfg(tmp_path, text))["demo"]["pid"] == 5
    assert load_config(cfg(tmp_path, "segment: {}\n"))["segment"] == {}


@pytest.mark.parametrize("text", [
    "segment: 3\n", "segment: {min_free_mb: -1}\n", "segment: {min_free_mb: 1.5}\n",
    "segment: {min_free_mb: true}\n", "segment: {simplify_tolerance_m: -1}\n",
    "segment: {min_area_m2: .nan}\n", "segment: {min_area_m2: '2'}\n", "segment: {model_id: ''}\n",
    "segment: {model_id: 3}\n", "demo: 3\n", "demo: {chunks: [r1_c1]}\n",
    "demo: {pid: 0, chunks: [r1_c1]}\n", "demo: {pid: 1}\n", "demo: {pid: 1, chunks: []}\n",
    "demo: {pid: 1, chunks: [r01_c1]}\n", "demo: {pid: 1, chunks: [5]}\n",
    "demo: {pid: 1, chunks: [r1_c1, r1_c1]}\n", "demo: {pid: true, chunks: [r1_c1]}\n",
])
def test_bad_segment_or_demo_section(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(cfg(tmp_path, text))


def test_non_negative_number():
    assert non_negative_number(0) and non_negative_number(2.5)
    assert not any(non_negative_number(v) for v in (-1, True, "1", None, float("inf")))


# ---- the adapter, with a fake samgeo module

class FakeSamGeo2:
    made = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.masks = None
        FakeSamGeo2.made.append(self)

    def generate(self, source):
        self.source = source
        self.masks = [{"segmentation": np.eye(3, dtype=np.uint8)}]


def test_default_loader_builds_the_adapter_from_a_fake_samgeo(monkeypatch, tmp_path):
    fake = types.ModuleType("samgeo")
    fake.SamGeo2 = FakeSamGeo2
    monkeypatch.setitem(sys.modules, "samgeo", fake)
    roots = base.make_roots(tmp_path)
    model = segment.load_sam2(roots, {"segment": {"min_free_mb": 8000, "model_id": "sam2-hiera-tiny"}},
                              probe=base.Probe(9000), environ={})
    assert model.fake is False and model.model_id == "facebook/sam2-hiera-tiny"
    assert FakeSamGeo2.made[-1].kwargs == {"model_id": "sam2-hiera-tiny", "device": "cuda",
                                           "automatic": True}
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    out = model.generate(rgb)
    assert FakeSamGeo2.made[-1].source is rgb
    assert len(out) == 1 and out[0].dtype == np.bool_ and out[0].tolist() == np.eye(3, dtype=bool).tolist()


def test_bad_model_id_rejected(tmp_path):
    roots = base.make_roots(tmp_path)
    for bad in ("", 3):
        with pytest.raises(ValueError):
            segment.load_sam2(roots, {"segment": {"min_free_mb": 8000, "model_id": bad}},
                              probe=base.Probe(9000), loader=base.Loader(), environ={})


def test_polygon_config_none_section_and_bad_bounds():
    assert segment._polygon_config({"segment": None}) == (0.5, 2.0)
    with pytest.raises(ValueError):
        segment.masks_to_polygons([], base.TRANSFORM, (1, 1, 0, 0), base.RUN, base.CHUNK)
    with pytest.raises(ValueError):
        segment.masks_to_polygons([], base.TRANSFORM, (0, 0, 1), base.RUN, base.CHUNK)


def test_write_polygons_rejects_ids_of_another_chunk_or_run(tmp_path):
    roots = base.make_roots(tmp_path)
    gdf = base.polygons([base.mask((10, 40), (10, 40))])
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, base.FOLDER, "r2295_c2395", base.META)
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, base.FOLDER, base.CHUNK,
                               {**base.META, "run_id": "20261111-1300-sam2"})
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, base.FOLDER, base.CHUNK, "not a mapping")
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, "bad", base.CHUNK, base.META)
    assert not (roots["semseg"] / base.FOLDER).exists()


def test_empty_polygon_set_writes_a_valid_file(tmp_path):
    roots = base.make_roots(tmp_path)
    path = segment.write_polygons(base.polygons([]), roots, base.FOLDER, base.CHUNK, base.META)
    assert len(gpd.read_file(path, layer="polygons")) == 0


def test_run_log_holds_the_gpu_values(tmp_path):
    roots, model, probe = base.one_run(tmp_path)
    gpu = {"found": True, "gpus": [], "apps": [{"pid": 1, "process_name": "vllm", "used_memory_mb": 79907}]}
    segment.run_chunks([base.CHUNK], model, roots, base.FOLDER, base.CFG, run_id=base.RUN,
                       probe=probe, gpu_query=lambda: gpu)
    log = json.loads((roots["semseg"] / "runs" / base.RUN / "log.json").read_text())
    assert log["gpu_before"] == gpu and log["gpu_after"] == gpu


def test_run_log_rejects_bad_run_id(tmp_path):
    roots = base.make_roots(tmp_path)
    with pytest.raises(ValueError):
        segment.write_run_log(roots, "bad", model_id="m", fake=True, memory_before_mb=1,
                              memory_after_mb=1, started="s", finished="f", chunks=[], chunks_done=[])


def test_main_does_not_change_the_real_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    roots, config = base.demo_setup(tmp_path, [base.CHUNK])
    segment.main(["--demo", "--run"], roots=roots, config=config, probe=base.Probe(30000),
                 loader=base.Loader(), now=base.NOW)
    import os
    assert "HF_HOME" not in os.environ


def test_fix_round_config_not_a_mapping_and_transform_not_affine():
    """FM-84 and FM-79: a clear ValueError for a non-mapping config and a non-transform."""
    mask = np.zeros((1920, 1920), dtype=bool)
    bounds = standard.chunk_bounds(2295, 2395)
    chunk = "r2295_c2395"
    with pytest.raises(ValueError, match="config must be a mapping"):
        segment.masks_to_polygons([mask], None, bounds, "20261111-1200-sam2", chunk, config="x")
    for bad in (None, 5, ("a", "b")):
        with pytest.raises(ValueError, match="affine"):
            segment.masks_to_polygons([mask], bad, bounds, "20261111-1200-sam2", chunk)
