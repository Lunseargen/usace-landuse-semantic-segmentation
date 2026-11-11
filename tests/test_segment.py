"""Tests of task 1.2 SAM 2 polygons (`semseg.segment`): I1, I2, FM-6, FM-70 to FM-75, D46 to D48.

API under test (names fixed here, see `.factory/1.2/tests.md`):
  MaskModel.generate(rgb) -> list of bool masks (H, W); a model has `model_id` and `fake`
  read_available_mb(run=subprocess.run) -> int      (`free -m`, column "available")
  load_sam2(roots, config, *, probe=None, loader=None, environ=None) -> model
  read_chunk(path) -> (rgb uint8 (H, W, 3), transform, crs)
  masks_to_polygons(masks, transform, chunk_bounds, run_id, chunk, config=None) -> GeoDataFrame
  write_polygons(gdf, roots, folder, chunk, metadata) -> Path
  write_run_log(roots, run_id, *, model_id, fake, memory_before_mb, memory_after_mb, started,
                finished, chunks, chunks_done, error=None) -> Path
  run_chunks(chunks, model, roots, folder, config, *, run_id, probe) -> list of Path
  select_demo_chunks(df, n=6) -> list of chunk IDs
  main(argv=None, roots=None, config=None, probe=None, loader=None, now=None) -> int

The config keys are `segment.min_free_mb`, `segment.simplify_tolerance_m` (default 0.5),
`segment.min_area_m2` (default 2.0), and `demo: {pid, chunks}`.

D22: no test builds a real SAM 2 model, loads weights, or uses the GPU. Every write test uses
fake roots under `tmp_path`. One read-only test reads the profile of one real NAIP chunk.
"""

import json
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import conftest
import guard_probe
from semseg import config as semseg_config
from semseg import segment
from semseg import standard
from semseg import storage

ROOT = Path(__file__).resolve().parent.parent
ROW, COL = 2295, 2394
CHUNK = standard.chunk_id(ROW, COL)
BOUNDS = standard.chunk_bounds(ROW, COL)
XMIN, YMIN, XMAX, YMAX = BOUNDS
N = standard.PIXELS_PER_CHUNK
TRANSFORM = from_origin(XMIN, YMAX, 0.5, 0.5)
RUN = "20261111-1200-sam2"
FOLDER = "77_TEST_LAKE"
CFG = {"segment": {"min_free_mb": 16000, "simplify_tolerance_m": 0.0, "min_area_m2": 0.0}}
META = {"run_id": RUN, "code_version": "a" * 40, "data_standard": "3", "naip_year": "2022",
        "sources": f"naip/{FOLDER}/{CHUNK}.tif", "created": "2026-11-11T12:00:00Z"}
FREE_OUT = ("               total        used        free      shared  buff/cache   available\n"
            "Mem:          124546      101451        1000         200       22000       23095\n"
            "Swap:              0           0           0\n")


# ------------------------------------------------------------------ helpers

def make_roots(base):
    roots = {"share": base / "share", "semseg": base / "share" / "semseg",
             "fgt": base / "share" / "fgt", "stack": base / "stack"}
    for path in roots.values():
        path.mkdir(parents=True)
    return roots


def mask(rows, cols):
    """A bool (N, N) mask, True in the pixel window rows[0]:rows[1], cols[0]:cols[1]."""
    out = np.zeros((N, N), dtype=bool)
    out[rows[0]:rows[1], cols[0]:cols[1]] = True
    return out


def write_tif(path, bounds=BOUNDS, bands=4, dtype="uint8", crs="EPSG:5070", size=N, pixel=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((bands, size, size), dtype=dtype)
    for b in range(bands):
        data[b] = 10 * (b + 1)
    data[0, 5, 7] = 99
    profile = {"driver": "GTiff", "width": size, "height": size, "count": bands, "dtype": dtype,
               "crs": crs, "transform": from_origin(bounds[0], bounds[3], pixel, pixel)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


class FakeModel:
    fake = True
    model_id = "fake-sam2"

    def __init__(self, masks=None, fail_on_call=None):
        self.masks = [mask((100, 140), (200, 260))] if masks is None else masks
        self.fail_on_call = fail_on_call
        self.calls = []

    def generate(self, rgb):
        self.calls.append(rgb)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("fake model failure")
        return self.masks


class Probe:
    def __init__(self, *values):
        self.values = list(values)
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.values[min(self.count, len(self.values)) - 1]


class Loader:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return FakeModel()


def polygons(masks, config=CFG, bounds=BOUNDS):
    return segment.masks_to_polygons(masks, TRANSFORM, bounds, RUN, CHUNK, config=config)


def one_run(tmp_path, chunk_ids=(CHUNK,), model=None, probe=None):
    roots = make_roots(tmp_path)
    for cid in chunk_ids:
        row, col = standard.parse_chunk_id(cid)
        write_tif(roots["fgt"] / FOLDER / "naip" / f"{cid}.tif", standard.chunk_bounds(row, col))
    return roots, model or FakeModel(), probe or Probe(30000, 29000)


def run(roots, model, probe, chunk_ids=(CHUNK,)):
    return segment.run_chunks(list(chunk_ids), model, roots, FOLDER, CFG, run_id=RUN, probe=probe)


# ------------------------------------------------------------------ I1 NAIP chunk profile

def test_naip_chunk_profile(tmp_path):
    """I1: EPSG:5070, 1920 x 1920, 0.5 m, 3 or more uint8 bands; rgb is bands 1 to 3."""
    path = write_tif(tmp_path / f"{CHUNK}.tif")
    rgb, transform, crs = segment.read_chunk(path)
    assert rgb.shape == (N, N, 3) and rgb.dtype == np.uint8
    assert (rgb[0, 0] == [10, 20, 30]).all()          # R, G, B; band 4 (nir) is not used
    assert rgb[5, 7, 0] == 99 and rgb[7, 5, 0] == 10  # row and column are not swapped
    assert transform == TRANSFORM
    assert crs.to_string() == "EPSG:5070"


@pytest.mark.parametrize("kwargs", [
    {"crs": "EPSG:4326"}, {"size": 1919}, {"pixel": 0.6}, {"bands": 2}, {"dtype": "float32"},
    {"dtype": "uint16"}, {"bounds": (XMIN + 0.25, YMIN, XMAX + 0.25, YMAX)},
], ids=["crs", "size", "pixel", "two_bands", "float", "uint16", "half_pixel_origin"])
def test_naip_chunk_profile_rejects_bad_file(tmp_path, kwargs):
    path = write_tif(tmp_path / f"{CHUNK}.tif", **kwargs)
    with pytest.raises(ValueError):
        segment.read_chunk(path)


def test_naip_chunk_profile_rejects_name_that_does_not_match_box(tmp_path):
    path = write_tif(tmp_path / "r2295_c2395.tif")     # the box is the box of c2394
    with pytest.raises(ValueError):
        segment.read_chunk(path)


def test_naip_chunk_profile_three_bands_ok(tmp_path):
    rgb, _, _ = segment.read_chunk(write_tif(tmp_path / f"{CHUNK}.tif", bands=3))
    assert rgb.shape == (N, N, 3)


def test_naip_chunk_profile_real_file_read_only():
    """One real NAIP chunk of 77: read only, no write, never skipped."""
    roots = storage.load_roots()
    storage.guard_share(roots)
    path = Path(roots["fgt"]) / "77_GRAPEVINE_LAKE" / "naip" / f"{CHUNK}.tif"
    rgb, transform, crs = segment.read_chunk(path)
    assert rgb.shape == (N, N, 3) and rgb.dtype == np.uint8
    assert crs.to_string() == "EPSG:5070"
    assert (transform.c, transform.f) == (XMIN, YMAX)


# ------------------------------------------------------------------ I2 polygons schema

def test_polygons_schema():
    gdf = polygons([mask((100, 140), (200, 260)), mask((400, 440), (800, 860))])
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert {"polygon_id", "area_m2", "geometry"} <= set(gdf.columns)
    assert gdf.crs.to_epsg() == 5070
    assert len(gdf) == 2
    assert gdf.geometry.is_valid.all() and not gdf.geometry.is_empty.any()
    assert set(gdf.geometry.geom_type) == {"Polygon"}
    ids = gdf["polygon_id"].tolist()
    assert len(set(ids)) == 2
    for text in ids:
        chunk, run_id, n = standard.parse_polygon_id(text)
        assert (chunk, run_id) == (CHUNK, RUN) and text == f"{chunk}_{run_id}_{n}"
    assert gdf["area_m2"].tolist() == pytest.approx(gdf.geometry.area.tolist())
    assert sorted(gdf["area_m2"]) == pytest.approx([40 * 60 * 0.25] * 2)


def test_polygons_schema_empty_result_keeps_columns_and_crs():
    for masks in ([], [np.zeros((N, N), dtype=bool)]):
        gdf = polygons(masks)
        assert len(gdf) == 0
        assert {"polygon_id", "area_m2", "geometry"} <= set(gdf.columns)
        assert gdf.crs.to_epsg() == 5070


def test_polygons_schema_hole_and_parts():
    ring = mask((100, 200), (100, 200))
    ring[130:170, 130:170] = False
    two = mask((300, 320), (300, 320)) | mask((300, 320), (500, 520))
    gdf = polygons([ring, two])
    assert len(gdf) == 3                                  # one row for each polygon part
    assert len(set(gdf["polygon_id"])) == 3
    assert sorted(gdf["area_m2"]) == pytest.approx([100.0, 100.0, (100 * 100 - 40 * 40) * 0.25])


@pytest.mark.parametrize("bad", [
    np.zeros((N, N), dtype=np.uint8), np.zeros((N,), dtype=bool), np.zeros((3, N, N), dtype=bool),
])
def test_polygons_schema_rejects_bad_mask(bad):
    with pytest.raises(ValueError):
        polygons([bad])


def test_polygons_schema_rejects_bad_ids():
    with pytest.raises(ValueError):
        segment.masks_to_polygons([], TRANSFORM, BOUNDS, "bad run", CHUNK, config=CFG)
    with pytest.raises(ValueError):
        segment.masks_to_polygons([], TRANSFORM, BOUNDS, RUN, "r01_c2", config=CFG)


# ------------------------------------------------------------------ FM-70 transform

def test_edge_fake_mask_corner_maps_to_chunk_corner():
    """FM-70: pixel (0, 0) is the upper-left corner; a non-square window gives exact bounds."""
    one = mask((0, 1), (0, 1))
    gdf = polygons([one])
    assert gdf.geometry.iloc[0].bounds == (XMIN, YMAX - 0.5, XMIN + 0.5, YMAX)
    win = mask((0, 10), (0, 20))                         # 10 rows high, 20 columns wide
    assert polygons([win]).geometry.iloc[0].bounds == (XMIN, YMAX - 5.0, XMIN + 10.0, YMAX)


def test_edge_fake_mask_far_corner_and_offset_window():
    last = mask((N - 1, N), (N - 1, N))
    assert polygons([last]).geometry.iloc[0].bounds == (XMAX - 0.5, YMIN, XMAX, YMIN + 0.5)
    win = mask((30, 38), (100, 120))                     # rows 30..37, cols 100..119
    expect = (XMIN + 50.0, YMAX - 19.0, XMIN + 60.0, YMAX - 15.0)
    assert polygons([win]).geometry.iloc[0].bounds == expect


# ------------------------------------------------------------------ FM-73 validity

def test_edge_polygons_valid_inside_chunk_no_sliver():
    """FM-73: no invalid, empty, outside or sliver polygon, also with a large tolerance."""
    rng = np.random.default_rng(7)
    noise = np.zeros((N, N), dtype=bool)
    noise[0:300, 0:300] = rng.random((300, 300)) > 0.5
    stair = np.zeros((N, N), dtype=bool)
    for i in range(200):
        stair[i, i:i + 3] = True
    edge = mask((0, 50), (N - 50, N))                     # touches the chunk edge
    line = mask((900, 901), (100, 106))                   # 1 x 6 pixels = 1.5 m2
    config = {"segment": {"simplify_tolerance_m": 3.0, "min_area_m2": 2.0}}
    gdf = polygons([noise, stair, edge, line], config=config)
    assert len(gdf) > 0
    chunk_box = box(*BOUNDS)
    assert gdf.geometry.is_valid.all() and not gdf.geometry.is_empty.any()
    assert gdf.geometry.apply(lambda g: chunk_box.buffer(1e-9).contains(g)).all()
    assert (gdf["area_m2"] >= 2.0).all()
    assert gdf["polygon_id"].is_unique


def test_edge_min_area_boundary():
    """Area 2.0 m2 (8 pixels) stays; 7 pixels (1.75 m2) goes. The default is 2 m2."""
    keep = mask((10, 12), (10, 14))                       # 2 x 4 = 8 pixels
    drop = mask((50, 51), (10, 17))                       # 1 x 7 = 7 pixels
    gdf = polygons([keep, drop], config={"segment": {"simplify_tolerance_m": 0.0}})
    assert gdf["area_m2"].tolist() == pytest.approx([2.0])
    gdf = polygons([keep, drop], config={"segment": {"simplify_tolerance_m": 0.0, "min_area_m2": 1.0}})
    assert len(gdf) == 2
    assert len(polygons([keep, drop], config=None)) == 1  # defaults: 0.5 m and 2 m2


def test_edge_clip_to_chunk_box_smaller_than_raster(monkeypatch):
    """A raster shape that reaches outside the chunk box is clipped to it (FM-73)."""
    outside = {"type": "Polygon", "coordinates": [[(XMIN + 100.0, YMAX - 100.0), (XMAX + 10.0, YMAX - 100.0),
                                                   (XMAX + 10.0, YMAX - 140.0), (XMIN + 100.0, YMAX - 140.0),
                                                   (XMIN + 100.0, YMAX - 100.0)]]}
    away = {"type": "Polygon", "coordinates": [[(XMAX + 5.0, YMAX), (XMAX + 10.0, YMAX), (XMAX + 10.0, YMAX - 5.0),
                                                (XMAX + 5.0, YMAX - 5.0), (XMAX + 5.0, YMAX)]]}
    monkeypatch.setattr(segment.features, "shapes", lambda *a, **k: iter([(outside, 1), (away, 1)]))
    gdf = polygons([mask((100, 200), (100, 200))])
    assert len(gdf) == 1 and gdf.geometry.iloc[0].bounds[2] == XMAX
    assert gdf["area_m2"].iloc[0] == pytest.approx((XMAX - XMIN - 100.0) * 40.0)


def test_edge_simplify_tolerance_from_config():
    stair = np.zeros((N, N), dtype=bool)
    for i in range(300):
        stair[i, i:i + 40] = True

    def vertices(config):
        return sum(len(g.exterior.coords) for g in polygons([stair], config=config).geometry)

    rough = vertices({"segment": {"simplify_tolerance_m": 0.0, "min_area_m2": 0.0}})
    smooth = vertices({"segment": {"simplify_tolerance_m": 2.0, "min_area_m2": 0.0}})
    assert smooth < rough / 4


@pytest.mark.parametrize("key,value", [("simplify_tolerance_m", -1.0), ("min_area_m2", -1.0),
                                       ("simplify_tolerance_m", float("nan")), ("min_area_m2", "2")])
def test_edge_bad_polygon_config_rejected(key, value):
    with pytest.raises(ValueError):
        polygons([mask((10, 20), (10, 20))], config={"segment": {key: value}})


# ------------------------------------------------------------------ FM-74 write

def test_edge_write_polygons_gpkg_layer_and_ds21(tmp_path):
    roots = make_roots(tmp_path)
    gdf = polygons([mask((100, 140), (200, 260))])
    path = segment.write_polygons(gdf, roots, FOLDER, CHUNK, META)
    assert path == roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg"
    assert pyogrio.list_layers(path)[:, 0].tolist() == ["polygons"]
    back = gpd.read_file(path, layer="polygons")
    assert back.crs.to_epsg() == 5070 and back["polygon_id"].tolist() == gdf["polygon_id"].tolist()
    meta = pyogrio.read_info(path, layer="polygons")["layer_metadata"]
    assert set(standard.METADATA_KEYS) <= set(meta)
    assert all(meta[k] for k in standard.METADATA_KEYS) and meta["run_id"] == RUN
    assert not list(path.parent.glob("*.part"))


@pytest.mark.parametrize("drop", list(standard.METADATA_KEYS))
def test_edge_write_polygons_rejects_missing_metadata_and_writes_nothing(tmp_path, drop):
    roots = make_roots(tmp_path)
    meta = {k: v for k, v in META.items() if k != drop}
    with pytest.raises(ValueError):
        segment.write_polygons(polygons([mask((10, 40), (10, 40))]), roots, FOLDER, CHUNK, meta)
    assert not (roots["semseg"] / FOLDER).exists()


def test_edge_write_polygons_rejects_wrong_crs_chunk_and_folder(tmp_path):
    roots = make_roots(tmp_path)
    gdf = polygons([mask((10, 40), (10, 40))])
    with pytest.raises(ValueError):
        segment.write_polygons(gdf.to_crs(4326), roots, FOLDER, CHUNK, META)
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, FOLDER, "r01_c1", META)
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, "../x", CHUNK, META)
    assert not (roots["semseg"] / FOLDER).exists()


def test_edge_write_polygons_goes_through_safe_write(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    seen = []
    orig = storage.safe_write

    def spy(target, data, roots_, **kw):
        seen.append((Path(target), type(data)))
        return orig(target, data, roots_, **kw)

    monkeypatch.setattr(storage, "safe_write", spy)
    segment.write_polygons(polygons([mask((10, 40), (10, 40))]), roots, FOLDER, CHUNK, META)
    assert seen == [(roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg", bytes)]


def test_edge_failure_mid_write_leaves_no_gpkg(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    orig = storage.safe_write

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "safe_write", lambda t, d, r, **kw: orig(t, d, r, copy=boom))
    with pytest.raises(OSError):
        segment.write_polygons(polygons([mask((10, 40), (10, 40))]), roots, FOLDER, CHUNK, META)
    folder = roots["semseg"] / FOLDER / "polygons"
    assert not list(folder.glob("*.gpkg")) and not list(folder.glob("*.part")) if folder.exists() else True


def test_edge_failure_keeps_old_file_whole(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)
    path = segment.write_polygons(polygons([mask((10, 40), (10, 40))]), roots, FOLDER, CHUNK, META)
    before = path.read_bytes()
    orig = storage.safe_write

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "safe_write", lambda t, d, r, **kw: orig(t, d, r, copy=boom))
    with pytest.raises(OSError):
        segment.write_polygons(polygons([mask((20, 50), (20, 50))]), roots, FOLDER, CHUNK, META)
    assert path.read_bytes() == before


def test_edge_rerun_replaces_file_ids_unique_and_ds21_present(tmp_path):
    roots = make_roots(tmp_path)
    first = segment.write_polygons(polygons([mask((10, 40), (10, 40)), mask((60, 90), (60, 90))]),
                                   roots, FOLDER, CHUNK, META)
    run2 = "20261111-1300-sam2"
    gdf2 = segment.masks_to_polygons([mask((200, 230), (200, 230))], TRANSFORM, BOUNDS, run2, CHUNK, config=CFG)
    second = segment.write_polygons(gdf2, roots, FOLDER, CHUNK, {**META, "run_id": run2})
    assert first == second
    back = gpd.read_file(second, layer="polygons")
    assert len(back) == 1 and back["polygon_id"].is_unique
    assert all(standard.parse_polygon_id(t)[1] == run2 for t in back["polygon_id"])
    assert pyogrio.read_info(second, layer="polygons")["layer_metadata"]["run_id"] == run2
    assert len(list(second.parent.glob("*"))) == 1


def test_edge_write_polygons_share_missing_exit_3(tmp_path):
    roots = make_roots(tmp_path)
    (roots["semseg"]).rmdir()
    with pytest.raises(SystemExit) as info:
        segment.write_polygons(polygons([mask((10, 40), (10, 40))]), roots, FOLDER, CHUNK, META)
    assert info.value.code == 3


# ------------------------------------------------------------------ FM-75 run log

def test_edge_log_has_time_gpu_memory_and_model_flag_after_failure(tmp_path):
    chunks = ["r2295_c2394", "r2295_c2395"]
    roots, model, probe = one_run(tmp_path, chunks, FakeModel(fail_on_call=2), Probe(30000, 29000))
    with pytest.raises(RuntimeError, match="fake model failure"):
        run(roots, model, probe, chunks)
    log = json.loads((roots["semseg"] / "runs" / RUN / "log.json").read_text())
    assert log["run_id"] == RUN and log["fake"] is True and log["model_id"] == "fake-sam2"
    assert log["memory_before_mb"] == 30000 and isinstance(log["memory_after_mb"], int)
    assert log["started"] and log["finished"]
    assert datetime.fromisoformat(log["started"].replace("Z", "+00:00")) <= \
        datetime.fromisoformat(log["finished"].replace("Z", "+00:00"))
    assert log["chunks"] == chunks and log["chunks_done"] == ["r2295_c2394"]
    assert "fake model failure" in log["error"]
    assert (roots["semseg"] / FOLDER / "polygons" / "r2295_c2394.gpkg").exists()
    assert not (roots["semseg"] / FOLDER / "polygons" / "r2295_c2395.gpkg").exists()


def test_edge_log_success_and_real_flag(tmp_path):
    roots, model, probe = one_run(tmp_path, probe=Probe(30000, 21000))
    paths = run(roots, model, probe)
    assert paths == [roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg"]
    log = json.loads((roots["semseg"] / "runs" / RUN / "log.json").read_text())
    assert (log["memory_before_mb"], log["memory_after_mb"]) == (30000, 21000)
    assert log["error"] is None and log["chunks_done"] == [CHUNK] and log["fake"] is True
    real = types.SimpleNamespace(model_id="facebook/sam2-hiera-large", generate=lambda rgb: [])
    path = segment.write_run_log(roots, "20261111-1400-sam2", model_id=real.model_id, fake=False,
                                 memory_before_mb=1, memory_after_mb=2, started="2026-11-11T14:00:00Z",
                                 finished="2026-11-11T14:01:00Z", chunks=[CHUNK], chunks_done=[])
    assert path == roots["semseg"] / "runs" / "20261111-1400-sam2" / "log.json"
    assert json.loads(path.read_text())["fake"] is False


def test_edge_fake_model_gets_rgb_uint8_of_the_chunk(tmp_path):
    roots, model, probe = one_run(tmp_path)
    run(roots, model, probe)
    assert len(model.calls) == 1
    rgb = model.calls[0]
    assert rgb.shape == (N, N, 3) and rgb.dtype == np.uint8 and rgb[0, 0].tolist() == [10, 20, 30]
    gdf = gpd.read_file(roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg", layer="polygons")
    assert len(gdf) == 1 and gdf.crs.to_epsg() == 5070
    meta = pyogrio.read_info(roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg",
                             layer="polygons")["layer_metadata"]
    assert meta["run_id"] == RUN and meta["data_standard"] == str(standard.DATA_STANDARD_VERSION)


def test_edge_run_chunks_missing_naip_raises_and_logs(tmp_path):
    roots, model, probe = one_run(tmp_path, ())
    with pytest.raises(FileNotFoundError):
        run(roots, model, probe)
    log = json.loads((roots["semseg"] / "runs" / RUN / "log.json").read_text())
    assert log["chunks_done"] == [] and log["error"]


def test_edge_run_chunks_rejects_empty_list_and_bad_run_id(tmp_path):
    roots, model, probe = one_run(tmp_path)
    with pytest.raises(ValueError):
        segment.run_chunks([], model, roots, FOLDER, CFG, run_id=RUN, probe=probe)
    with pytest.raises(ValueError):
        segment.run_chunks([CHUNK], model, roots, FOLDER, CFG, run_id="bad", probe=probe)
    assert model.calls == []


# ------------------------------------------------------------------ FM-6, FM-72, D47 memory check

def test_fm6_read_available_mb_parses_free_output():
    done = types.SimpleNamespace(returncode=0, stdout=FREE_OUT, stderr="")
    assert segment.read_available_mb(run=lambda *a, **k: done) == 23095


@pytest.mark.parametrize("out,code", [("", 0), ("Mem: x\n", 0), (FREE_OUT, 1)])
def test_fm6_read_available_mb_bad_output_raises(out, code):
    done = types.SimpleNamespace(returncode=code, stdout=out, stderr="")
    with pytest.raises(RuntimeError):
        segment.read_available_mb(run=lambda *a, **k: done)


@pytest.mark.parametrize("available,loads", [(15999, False), (16000, True), (16001, True)])
def test_edge_memory_check_refuses_below_need_and_loads_nothing(tmp_path, available, loads):
    """FM-72, D47: available < segment.min_free_mb gives exit 4 and a fake loader with 0 calls."""
    roots = make_roots(tmp_path)
    loader, environ = Loader(), {}
    config = {"segment": {"min_free_mb": 16000}}
    if loads:
        assert isinstance(segment.load_sam2(roots, config, probe=Probe(available), loader=loader,
                                            environ=environ), FakeModel)
        assert loader.calls == 1
    else:
        with pytest.raises(SystemExit) as info:
            segment.load_sam2(roots, config, probe=Probe(available), loader=loader, environ=environ)
        assert info.value.code == 4 and loader.calls == 0


def test_edge_memory_check_runs_after_hf_home_and_before_load(tmp_path):
    roots = make_roots(tmp_path)
    order, environ = [], {}

    def probe():
        order.append(("probe", environ.get("HF_HOME")))
        return 20000

    def loader(*a, **k):
        order.append(("load", environ.get("HF_HOME")))
        return FakeModel()

    segment.load_sam2(roots, {"segment": {"min_free_mb": 16000}}, probe=probe, loader=loader, environ=environ)
    home = str(Path(roots["semseg"]) / standard.MODELS_DIR)
    assert order == [("probe", home), ("load", home)]


def test_edge_loader_refuses_hf_home_off_share(tmp_path):
    """FM-71: HF_HOME outside the share stops the loader before the probe and the load."""
    roots = make_roots(tmp_path)
    probe, loader = Probe(99999), Loader()
    for bad in (str(tmp_path / "cache"), "relative/cache"):
        with pytest.raises(ValueError):
            segment.load_sam2(roots, {"segment": {"min_free_mb": 8000}}, probe=probe, loader=loader,
                              environ={"HF_HOME": bad})
    assert probe.count == 0 and loader.calls == 0


def test_edge_loader_share_missing_exit_3_before_probe(tmp_path):
    roots = make_roots(tmp_path)
    roots["semseg"].rmdir()
    probe, loader = Probe(99999), Loader()
    with pytest.raises(SystemExit) as info:
        segment.load_sam2(roots, {"segment": {"min_free_mb": 8000}}, probe=probe, loader=loader, environ={})
    assert info.value.code == 3 and probe.count == 0 and loader.calls == 0


def test_edge_default_loader_raises_when_fake_model_env_set(tmp_path):
    """FM-71: with SEMSEG_FAKE_MODEL set, the default factory never builds the real model."""
    roots = make_roots(tmp_path)
    with pytest.raises(RuntimeError, match="SEMSEG_FAKE_MODEL"):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 8000}}, probe=Probe(99999),
                          environ={"SEMSEG_FAKE_MODEL": "1"})


def test_edge_bad_memory_config_rejected(tmp_path):
    roots = make_roots(tmp_path)
    loader = Loader()
    for bad in ({}, {"segment": {}}, {"segment": {"min_free_mb": -1}}, {"segment": {"min_free_mb": "x"}}):
        with pytest.raises(ValueError):
            segment.load_sam2(roots, bad, probe=Probe(99999), loader=loader, environ={})
    assert loader.calls == 0


# ------------------------------------------------------------------ FM-71 guard

def test_edge_suite_never_imports_real_sam2():
    """FM-71, D22: the segment module imports no SAM 2 module, and no CUDA context exists."""
    for name in ("sam2", "sam2.build_sam", "samgeo", "samgeo.samgeo2"):
        assert name not in sys.modules, f"{name} was imported by the test suite or by semseg.segment"
    torch = sys.modules.get("torch")
    assert torch is None or not torch.cuda.is_initialized()


def test_edge_guard_blocks_samgeo2_and_build_sam_and_download():
    fake_samgeo = types.ModuleType("samgeo")
    fake_samgeo.SamGeo2 = type("SamGeo2", (), {})
    fake_build = types.ModuleType("sam2.build_sam")
    fake_build.build_sam2 = lambda *a, **k: object()
    fake_build.hf_hub_download = lambda *a, **k: "x"
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda *a, **k: "x"
    for module in (fake_samgeo, fake_build, fake_hf):
        assert conftest.patch_module(module)
    with pytest.raises(conftest.RealModelError):
        fake_samgeo.SamGeo2(model_id="x")
    with pytest.raises(conftest.RealModelError):
        fake_build.build_sam2("cfg", "ckpt")
    with pytest.raises(conftest.RealModelError):
        fake_build.hf_hub_download(repo_id="x", filename="y")
    with pytest.raises(conftest.RealModelError):
        fake_hf.hf_hub_download("x", "y")
    assert conftest.patch_module(fake_samgeo) == []          # a second patch changes nothing


def test_edge_guard_active_for_this_module_and_default_loader_is_guarded(tmp_path):
    """The session guard is installed, and the segment module did not copy a real builder."""
    assert any(isinstance(f, conftest._GuardFinder) for f in sys.meta_path)
    roots = make_roots(tmp_path)
    with pytest.raises((conftest.RealModelError, RuntimeError)):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 8000}}, probe=Probe(99999),
                          environ={"SEMSEG_FAKE_MODEL": "1", "HF_HOME": str(roots["semseg"] / "models")})
    assert "huggingface_hub.file_download" not in sys.modules or \
        getattr(sys.modules["huggingface_hub.file_download"].hf_hub_download, "_semseg_guard", True)


def test_review_name_bound_at_collection_is_the_guarded_one():
    """Review fix: a top-level `from huggingface_hub import hf_hub_download` is guarded, no network."""
    for func in (guard_probe.hf_hub_download, guard_probe.snapshot_download, guard_probe.mixin_download):
        assert getattr(func, "_semseg_guard", False)
        with pytest.raises(conftest.RealModelError):
            func("some/repo", "file.bin")
    import huggingface_hub.file_download as file_download
    import huggingface_hub.hub_mixin as hub_mixin
    assert file_download.hf_hub_download._semseg_guard and hub_mixin.hf_hub_download._semseg_guard


# ------------------------------------------------------------------ review notes (task 1.2)

def test_review_abbreviated_flag_is_refused_nothing_loaded(tmp_path):
    """D50: `--ru` is not `--run`: exit 2, no probe, no load."""
    roots = make_roots(tmp_path)
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo", "--ru"], roots=roots, config={}, probe=probe, loader=loader, now=NOW)
    assert info.value.code == 2 and loader.calls == 0 and probe.count == 0


def test_review_low_memory_log_uses_the_value_that_caused_the_stop(tmp_path):
    """The log shows the value of the first probe (15000), not a second probe (17000)."""
    roots, config = demo_setup(tmp_path, [CHUNK])
    config = {**config, "segment": {"min_free_mb": 16000}}
    probe = Probe(15000, 17000)
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=probe, loader=Loader(), now=NOW)
    assert info.value.code == 4 and probe.count == 1
    log = json.loads(next((roots["semseg"] / "runs").glob("*/log.json")).read_text())
    assert log["memory_before_mb"] == 15000 and log["memory_after_mb"] == 15000


def test_review_run_log_failure_keeps_the_first_error_and_notes_the_second(tmp_path, monkeypatch):
    roots, model, probe = one_run(tmp_path, chunk_ids=())          # the NAIP file is missing
    monkeypatch.setattr(segment, "write_run_log", lambda *a, **k: (_ for _ in ()).throw(OSError("share dropped")))
    with pytest.raises(FileNotFoundError) as info:
        run(roots, model, probe)
    assert any("share dropped" in note for note in info.value.__notes__)


def test_review_run_log_failure_alone_is_raised(tmp_path, monkeypatch):
    roots, model, probe = one_run(tmp_path)
    monkeypatch.setattr(segment, "write_run_log", lambda *a, **k: (_ for _ in ()).throw(OSError("share dropped")))
    with pytest.raises(OSError, match="share dropped"):
        run(roots, model, probe)


# ------------------------------------------------------------------ D46 demo chunks and config

def test_d46_select_demo_chunks_largest_share_ties_by_row_then_col():
    rows = [(1, 5, 0.9), (1, 4, 0.9), (0, 9, 0.9), (2, 2, 1.0), (3, 3, 0.5), (4, 4, 0.8),
            (5, 5, 0.8), (6, 6, 0.1), (7, 7, 0.95)]
    df = pd.DataFrame({"chunk": [standard.chunk_id(r, c) for r, c, _ in rows],
                       "row": [r for r, _, _ in rows], "col": [c for _, c, _ in rows],
                       "footprint_share": [s for _, _, s in rows]})
    picked = segment.select_demo_chunks(df)
    assert picked == ["r2_c2", "r7_c7", "r0_c9", "r1_c4", "r1_c5", "r4_c4"]
    assert segment.select_demo_chunks(df, n=2) == ["r2_c2", "r7_c7"]
    assert segment.select_demo_chunks(df.iloc[::-1]) == picked      # the order of rows is not used


def test_d46_select_demo_chunks_rejects_too_few_rows():
    df = pd.DataFrame({"chunk": ["r0_c0"], "row": [0], "col": [0], "footprint_share": [1.0]})
    with pytest.raises(ValueError):
        segment.select_demo_chunks(df)


def test_d46_config_yaml_demo_matches_the_six_largest_footprints_of_77():
    config = semseg_config.load_config()
    assert config["demo"]["pid"] == 77
    chunks = config["demo"]["chunks"]
    assert len(chunks) == 6 and len(set(chunks)) == 6
    roots = storage.load_roots()
    storage.guard_share(roots)
    df = pd.read_parquet(Path(roots["semseg"]) / "77_GRAPEVINE_LAKE" / standard.CHUNK_LIST)
    ranked = df.sort_values(["footprint_share", "row", "col"], ascending=[False, True, True])
    assert set(chunks) == set(ranked["chunk"].head(6))


def test_d47_config_yaml_segment_section():
    seg = semseg_config.load_config()["segment"]
    assert seg["min_free_mb"] == 16000
    assert seg["simplify_tolerance_m"] == 0.5 and seg["min_area_m2"] == 2


# ------------------------------------------------------------------ D22 CLI and D47 run log

def demo_setup(tmp_path, chunk_ids):
    roots = make_roots(tmp_path)
    (roots["stack"] / FOLDER).mkdir()
    for cid in chunk_ids:
        row, col = standard.parse_chunk_id(cid)
        write_tif(roots["fgt"] / FOLDER / "naip" / f"{cid}.tif", standard.chunk_bounds(row, col))
    config = {**CFG, "demo": {"pid": 77, "chunks": list(chunk_ids)}}
    return roots, config


NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)


def test_d22_cli_demo_without_run_refuses_and_loads_nothing(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo"], roots=roots, config=config, probe=probe, loader=loader, now=NOW)
    assert info.value.code not in (0, None)
    assert loader.calls == 0 and probe.count == 0
    assert not (roots["semseg"] / "runs").exists() and not (roots["semseg"] / FOLDER).exists()


def test_d22_cli_module_demo_without_run_exits_nonzero_and_names_the_flag():
    done = subprocess.run([sys.executable, "-m", "semseg.segment", "--demo"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "SEMSEG_FAKE_MODEL": "1"})
    assert done.returncode != 0
    assert "--run" in done.stderr and "Traceback" not in done.stderr


def test_d22_cli_no_args_refuses(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    with pytest.raises(SystemExit) as info:
        segment.main([], roots=roots, config=config, probe=Probe(99999), loader=Loader(), now=NOW)
    assert info.value.code not in (0, None)


def test_d22_cli_demo_run_with_fake_loader_writes_polygons_and_log(tmp_path):
    chunk_ids = ["r2295_c2394", "r2295_c2395", "r2296_c2394"]
    roots, config = demo_setup(tmp_path, chunk_ids)
    loader = Loader()
    code = segment.main(["--demo", "--run"], roots=roots, config=config,
                        probe=Probe(30000, 30000, 29000), loader=loader, now=NOW)
    assert code == 0 and loader.calls == 1
    for cid in chunk_ids:
        assert (roots["semseg"] / FOLDER / "polygons" / f"{cid}.gpkg").exists()
    logs = list((roots["semseg"] / "runs").glob("*/log.json"))
    assert len(logs) == 1
    log = json.loads(logs[0].read_text())
    standard.check_run_id(log["run_id"])
    assert log["fake"] is True and log["chunks_done"] == chunk_ids
    assert log["memory_before_mb"] == 30000 and log["memory_after_mb"] == 29000


def test_d47_cli_low_memory_exit_4_nothing_loaded_log_has_memory(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    loader = Loader()
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo", "--run"], roots=roots, config=config,
                     probe=Probe(15999), loader=loader, now=NOW)
    assert info.value.code == 4 and loader.calls == 0
    logs = list((roots["semseg"] / "runs").glob("*/log.json"))
    assert len(logs) == 1
    log = json.loads(logs[0].read_text())
    assert log["memory_before_mb"] == 15999 and log["memory_after_mb"] == 15999
    assert log["chunks_done"] == [] and log["error"]
    assert not (roots["semseg"] / FOLDER / "polygons").exists()


def test_d48_run_writes_only_under_semseg_and_nothing_to_fgt_or_stack(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    before = {k: sorted(p.relative_to(roots[k]) for p in Path(roots[k]).rglob("*"))
              for k in ("fgt", "stack")}
    segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(30000),
                 loader=Loader(), now=NOW)
    for k in ("fgt", "stack"):
        assert sorted(p.relative_to(roots[k]) for p in Path(roots[k]).rglob("*")) == before[k]
