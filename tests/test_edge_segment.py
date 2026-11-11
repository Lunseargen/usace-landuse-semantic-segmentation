"""Edge attacks on `semseg.segment` (task 1.2). New FMEA rows FM-76 to FM-84.

An open fault has a strict xfail mark with the reason. Every test uses fake roots under
`tmp_path`, fake models, and fake `free` and `nvidia-smi` output. No test builds a real SAM 2
model, reads the GPU, or writes to the real share (D22). Only `test_edge_real_naip_*` reads one
real NAIP file, read only.
"""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyogrio
import pytest
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely import wkt
from shapely.geometry import box

from semseg import segment, setup_check, standard, storage
from semseg import config as semseg_config
from test_segment import (BOUNDS, CFG, CHUNK, FOLDER, META, N, NOW, RUN, TRANSFORM, XMAX, XMIN,
                          YMAX, YMIN, FakeModel, Loader, Probe, demo_setup, make_roots, mask,
                          one_run, polygons, run, write_tif)

DEFAULT_CFG = {"segment": {"simplify_tolerance_m": 0.5, "min_area_m2": 2}}


def read_log(roots):
    logs = list((Path(roots["semseg"]) / "runs").glob("*/log.json"))
    assert len(logs) == 1
    return json.loads(logs[0].read_text())


# ------------------------------------------------------------------ read_chunk

def test_edge_read_chunk_width_and_height_checked_one_by_one(tmp_path):
    for width, height in ((1919, 1920), (1920, 1919), (1921, 1920)):
        path = tmp_path / f"{width}x{height}" / f"{CHUNK}.tif"
        path.parent.mkdir()
        profile = {"driver": "GTiff", "width": width, "height": height, "count": 3, "dtype": "uint8",
                   "crs": "EPSG:5070", "transform": from_origin(XMIN, YMAX, 0.5, 0.5)}
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.zeros((3, height, width), dtype="uint8"))
        with pytest.raises(ValueError):
            segment.read_chunk(path)


@pytest.mark.parametrize("transform", [
    Affine(0.5, 0.1, XMIN, 0.0, -0.5, YMAX),          # skewed (shear in x)
    Affine(0.5, 0.0, XMIN, 0.1, -0.5, YMAX),          # skewed (shear in y)
    Affine(0.5, 0.0, XMIN, 0.0, 0.5, YMIN),           # south up
    Affine.rotation(5) @ from_origin(XMIN, YMAX, 0.5, 0.5),
], ids=["shear_x", "shear_y", "south_up", "rotated"])
def test_edge_read_chunk_rejects_rotated_or_skewed_transform(tmp_path, transform):
    path = tmp_path / f"{CHUNK}.tif"
    profile = {"driver": "GTiff", "width": N, "height": N, "count": 3, "dtype": "uint8",
               "crs": "EPSG:5070", "transform": transform}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((3, N, N), dtype="uint8"))
    with pytest.raises(ValueError):
        segment.read_chunk(path)


@pytest.mark.parametrize("shift", [(0.5, 0), (0, -0.5), (30.0, 0), (0, 30.0)],
                         ids=["one_pixel_x", "one_pixel_y", "one_cell_x", "one_cell_y"])
def test_edge_read_chunk_rejects_shifted_origin_with_chunk_name(tmp_path, shift):
    """A file that sits one pixel (off lattice) or one cell (on lattice) off the name is refused."""
    bounds = (XMIN + shift[0], YMIN + shift[1], XMAX + shift[0], YMAX + shift[1])
    path = write_tif(tmp_path / f"{CHUNK}.tif", bounds=bounds)
    with pytest.raises(ValueError):
        segment.read_chunk(path)


@pytest.mark.parametrize("bands, ok", [(1, False), (2, False), (3, True), (4, True), (5, True)])
def test_edge_read_chunk_band_counts(tmp_path, bands, ok):
    path = write_tif(tmp_path / f"{CHUNK}.tif", bands=bands)
    if not ok:
        with pytest.raises(ValueError):
            segment.read_chunk(path)
        return
    rgb, _, _ = segment.read_chunk(path)
    assert rgb.shape == (N, N, 3)
    assert (rgb[1, 1] == [10, 20, 30]).all()      # bands 1 to 3 only; bands 4 and 5 never in rgb


@pytest.mark.parametrize("crs", ["EPSG:6350", "EPSG:32614", "EPSG:3857"])
def test_edge_read_chunk_rejects_other_projected_crs(tmp_path, crs):
    with pytest.raises(ValueError):
        segment.read_chunk(write_tif(tmp_path / f"{CHUNK}.tif", crs=crs))


def test_edge_read_chunk_no_crs_rejected(tmp_path):
    path = tmp_path / f"{CHUNK}.tif"
    profile = {"driver": "GTiff", "width": N, "height": N, "count": 3, "dtype": "uint8",
               "transform": from_origin(XMIN, YMAX, 0.5, 0.5)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((3, N, N), dtype="uint8"))
    with pytest.raises(ValueError):
        segment.read_chunk(path)


def test_edge_read_chunk_nodata_is_read_through_and_does_not_change_the_profile(tmp_path):
    """A nodata tag is accepted; the pixels are returned as stored (no mask is made)."""
    path = tmp_path / f"{CHUNK}.tif"
    data = np.full((4, N, N), 7, dtype="uint8")
    data[:, :10, :10] = 0
    profile = {"driver": "GTiff", "width": N, "height": N, "count": 4, "dtype": "uint8", "nodata": 0,
               "crs": "EPSG:5070", "transform": TRANSFORM}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(gsd="0.6", years="2022")  # D57
    rgb, transform, _ = segment.read_chunk(path)
    assert rgb.dtype == np.uint8 and rgb[0, 0].tolist() == [0, 0, 0] and rgb[20, 20].tolist() == [7, 7, 7]
    assert transform == TRANSFORM


def test_edge_read_chunk_five_band_with_unusual_colour_interpretation_is_read_as_bands_1_to_3(tmp_path):
    """Bands 1 to 3 are taken as RGB. The real NAIP files carry no colour tags (gray, undefined),
    so the order cannot be checked from the file: this test records the rule only."""
    path = write_tif(tmp_path / f"{CHUNK}.tif", bands=5)
    with rasterio.open(path, "r+") as dst:
        dst.colorinterp = [ColorInterp.blue, ColorInterp.green, ColorInterp.red,
                           ColorInterp.undefined, ColorInterp.undefined]
    rgb, _, _ = segment.read_chunk(path)
    assert rgb[1, 1].tolist() == [10, 20, 30]


@pytest.mark.parametrize("content", [b"", b"not a tiff"], ids=["empty", "garbage"])
def test_edge_read_chunk_corrupt_file_raises(tmp_path, content):
    path = tmp_path / f"{CHUNK}.tif"
    path.write_bytes(content)
    with pytest.raises(Exception):
        segment.read_chunk(path)


def test_edge_read_chunk_directory_and_missing_raise_file_not_found(tmp_path):
    (tmp_path / f"{CHUNK}.tif").mkdir()
    with pytest.raises(FileNotFoundError):
        segment.read_chunk(tmp_path / f"{CHUNK}.tif")
    with pytest.raises(FileNotFoundError):
        segment.read_chunk(tmp_path / "r1_c1.tif")


@pytest.mark.parametrize("name", ["r02_c3.tif", "R1_C1.tif", "r1_c1.tiff", "tile.tif", "r1_c1 .tif"])
def test_edge_read_chunk_bad_name(tmp_path, name):
    path = write_tif(tmp_path / name)
    with pytest.raises(ValueError):
        segment.read_chunk(path)


# ------------------------------------------------------------------ masks_to_polygons

def test_edge_all_true_mask_is_the_chunk_box_with_default_config():
    gdf = polygons([np.ones((N, N), dtype=bool)], config=DEFAULT_CFG)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].equals(box(*BOUNDS))
    assert gdf["area_m2"].iloc[0] == (N * 0.5) ** 2


def test_edge_all_false_mask_gives_no_row():
    assert len(polygons([np.zeros((N, N), dtype=bool)])) == 0


def test_edge_mask_with_hole_keeps_the_hole():
    ring = mask((100, 300), (100, 300))
    ring[150:250, 150:250] = False
    gdf = polygons([ring], config=DEFAULT_CFG)
    assert len(gdf) == 1
    assert len(gdf.geometry.iloc[0].interiors) == 1
    assert gdf["area_m2"].iloc[0] == pytest.approx((200 * 200 - 100 * 100) * 0.25)


def test_edge_diagonal_one_pixel_line_is_dropped_by_the_default_min_area():
    m = np.zeros((N, N), dtype=bool)
    idx = np.arange(100)
    m[idx, idx] = True
    assert len(polygons([m], config=DEFAULT_CFG)) == 0
    assert len(polygons([m], config=CFG)) == 100        # min area 0: every pixel stays (4-connected)


def test_edge_mask_touching_every_chunk_edge_stays_inside_the_box():
    m = np.zeros((N, N), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True      # a frame of one pixel
    gdf = polygons([m], config=DEFAULT_CFG)
    assert len(gdf) == 1 and gdf.geometry.is_valid.all()
    assert box(*BOUNDS).contains(gdf.geometry.iloc[0]) or box(*BOUNDS).covers(gdf.geometry.iloc[0])
    assert gdf.total_bounds.tolist() == [XMIN, YMIN, XMAX, YMAX]


def test_edge_first_and_last_pixel_map_to_the_corners_and_y_runs_down():
    first = mask((0, 4), (0, 4))
    last = mask((N - 4, N), (N - 4, N))
    gdf = polygons([first, last])
    assert gdf.geometry.iloc[0].bounds == (XMIN, YMAX - 2.0, XMIN + 2.0, YMAX)      # upper left
    assert gdf.geometry.iloc[1].bounds == (XMAX - 2.0, YMIN, XMAX, YMIN + 2.0)      # lower right


def test_edge_mask_larger_than_the_chunk_is_refused():
    """FM-80: the mask must be 1920 x 1920; the old clip of an oversize mask is now an error."""
    with pytest.raises(ValueError):
        polygons([np.ones((N + 40, N + 40), dtype=bool)], config=DEFAULT_CFG)


def test_edge_min_area_uses_greater_or_equal_and_drops_just_below():
    one_pixel = mask((10, 11), (10, 11))                 # 0.25 m2
    eight = mask((10, 12), (10, 14))                     # 8 pixels = 2.0 m2
    seven = mask((10, 11), (10, 17))                     # 7 pixels = 1.75 m2
    config = {"segment": {"simplify_tolerance_m": 0.0, "min_area_m2": 2.0}}
    gdf = polygons([one_pixel, eight, seven], config=config)
    assert gdf["area_m2"].tolist() == [2.0]


def test_edge_bowtie_becomes_one_row_for_each_part_with_distinct_ids():
    bow = wkt.loads("POLYGON((0 0, 10 10, 10 0, 0 10, 0 0))")
    parts = segment._clean(bow, box(-1, -1, 11, 11), 0.0, 0.0)
    assert [p.geom_type for p in parts] == ["Polygon", "Polygon"]
    assert all(p.is_valid for p in parts)


def test_edge_simplify_keeps_polygons_valid_and_never_empty():
    """A huge tolerance cannot make an empty or an invalid polygon (preserve_topology)."""
    star = mask((100, 160), (100, 160)) | mask((120, 140), (160, 400))
    for tolerance in (0.0, 0.5, 5.0, 1e6):
        gdf = polygons([star], config={"segment": {"simplify_tolerance_m": tolerance, "min_area_m2": 0.0}})
        assert len(gdf) >= 1 and gdf.geometry.is_valid.all() and not gdf.geometry.is_empty.any()
        assert all(box(*BOUNDS).covers(g) for g in gdf.geometry)


def test_edge_polygon_ids_start_at_1_and_have_no_gap_after_the_filter():
    masks = [mask((10, 11), (10, 11)), mask((20, 40), (20, 40)), mask((50, 51), (50, 51)),
             mask((60, 80), (60, 80))]
    gdf = polygons(masks, config=DEFAULT_CFG)               # the two 1-pixel masks are dropped
    numbers = [standard.parse_polygon_id(text)[2] for text in gdf["polygon_id"]]
    assert numbers == [1, 2]


def test_edge_polygon_ids_differ_across_chunks_of_one_run():
    a = segment.masks_to_polygons([mask((10, 40), (10, 40))], TRANSFORM, BOUNDS, RUN, CHUNK, CFG)
    row, col = standard.parse_chunk_id(CHUNK)
    other = standard.chunk_id(row, col + 1)
    shifted = from_origin(XMIN + 960, YMAX, 0.5, 0.5)
    b = segment.masks_to_polygons([mask((10, 40), (10, 40))], shifted, standard.chunk_bounds(row, col + 1),
                                  RUN, other, CFG)
    assert not set(a["polygon_id"]) & set(b["polygon_id"])


@pytest.mark.parametrize("bad", [
    np.zeros((N, N), dtype=np.int8), np.zeros((N, N), dtype=float), np.zeros((N, N), dtype=object),
    [[True]], None, 1, "mask",
], ids=["int8", "float", "object", "list", "none", "int", "str"])
def test_edge_masks_of_wrong_type_or_dtype_raise_value_error(bad):
    with pytest.raises(ValueError):
        polygons([bad])


@pytest.mark.parametrize("name, build", [
    ("fortran", lambda: np.asfortranarray(mask((1, 5), (1, 5)))),
    ("reversed_view", lambda: mask((N - 5, N - 1), (1, 5))[::-1]),
    ("read_only", lambda: (lambda a: (a.setflags(write=False), a)[1])(mask((1, 5), (1, 5)))),
])
def test_edge_unusual_memory_layout_gives_the_same_polygon(name, build):
    assert len(polygons([build()])) == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "1"],
                         ids=["nan", "inf", "neg", "bool", "str"])
@pytest.mark.parametrize("key", ["simplify_tolerance_m", "min_area_m2"])
def test_edge_polygon_config_bad_values(key, bad):
    with pytest.raises(ValueError):
        polygons([mask((1, 5), (1, 5))], config={"segment": {key: bad}})


def test_edge_polygon_config_missing_keys_use_the_defaults():
    for config in (None, {}, {"segment": {}}, {"segment": None}):
        gdf = polygons([mask((10, 40), (10, 40)), mask((1, 2), (1, 2))], config=config)
        assert len(gdf) == 1                            # the 0.25 m2 mask is dropped by the default 2.0


def test_edge_10_000_small_masks_scale_with_the_count(monkeypatch):
    """The cost is for each mask, not for each pixel pair. 10 000 masks are simulated by a fake
    `features.shapes` (a real run on 10 000 full rasters needs about 4 minutes: see edges.md)."""
    calls = []

    def fake_shapes(source, mask=None, transform=None):
        calls.append(1)
        k = len(calls)
        x = XMIN + (k % 900) * 1.0
        y = YMAX - (k // 900) * 1.0
        yield {"type": "Polygon", "coordinates": [[(x, y), (x + 0.9, y), (x + 0.9, y - 0.9), (x, y - 0.9), (x, y)]]}, 1

    monkeypatch.setattr(segment.features, "shapes", fake_shapes)
    one = np.ones((N, N), dtype=bool)
    gdf = segment.masks_to_polygons([one] * 10_000, TRANSFORM, BOUNDS, RUN, CHUNK, CFG)
    assert len(gdf) == 10_000 and len(set(gdf["polygon_id"])) == 10_000
    numbers = sorted(standard.parse_polygon_id(t)[2] for t in gdf["polygon_id"])
    assert numbers[0] == 1 and numbers[-1] == 10_000


def test_edge_config_not_a_mapping_gives_value_error():
    for config in ({"segment": "x"}, {"segment": [1]}):
        with pytest.raises(ValueError):
            polygons([mask((1, 5), (1, 5))], config=config)


# ------------------------------------------------------------------ open faults: masks_to_polygons

def test_edge_masks_as_iterator_are_not_silently_dropped():
    gdf = polygons(iter([mask((10, 40), (10, 40))]))
    assert len(gdf) == 1


@pytest.mark.parametrize("transform", [
    Affine(0.5, 0.0, XMIN, 0.0, 0.5, YMIN),                 # y runs up: the picture is mirrored
    Affine(0.5, 0.1, XMIN, 0.1, -0.5, YMAX),                # skewed
    from_origin(XMIN, YMAX, 0.6, 0.6),                      # wrong pixel size
    from_origin(XMIN + 0.25, YMAX, 0.5, 0.5),               # off the lattice
], ids=["mirrored", "skewed", "pixel_0_6", "half_pixel"])
def test_edge_transform_that_is_not_the_chunk_transform_is_refused(transform):
    with pytest.raises(ValueError):
        segment.masks_to_polygons([mask((10, 40), (10, 40))], transform, BOUNDS, RUN, CHUNK, CFG)


def test_edge_bounds_of_another_chunk_are_refused():
    other = standard.chunk_bounds(1, 1)
    with pytest.raises(ValueError):
        segment.masks_to_polygons([mask((10, 40), (10, 40))], TRANSFORM, other, RUN, CHUNK, CFG)


@pytest.mark.parametrize("shape", [(100, 100), (N, N - 1), (0, 0)], ids=["small", "one_col_short", "empty"])
def test_edge_mask_of_wrong_shape_is_refused_with_value_error(shape):
    with pytest.raises(ValueError):
        polygons([np.ones(shape, dtype=bool)])


# ------------------------------------------------------------------ write_polygons

def two_polygons():
    return polygons([mask((10, 40), (10, 40)), mask((100, 140), (100, 140))])


def test_edge_empty_geodataframe_is_written_as_an_empty_layer_with_ds21(tmp_path):
    roots = make_roots(tmp_path)
    out = segment.write_polygons(polygons([]), roots, FOLDER, CHUNK, META)
    info = pyogrio.read_info(out, layer="polygons")
    assert info["features"] == 0
    assert info["layer_metadata"]["run_id"] == RUN
    assert str(info["crs"]).endswith("5070")


def test_edge_hostile_text_in_metadata_survives_the_round_trip(tmp_path):
    roots = make_roots(tmp_path)
    text = "<script>alert(1)</script>&\"'\r\n漢\U0001F600 =cmd|' /C calc'!A0"
    out = segment.write_polygons(two_polygons(), roots, FOLDER, CHUNK, {**META, "sources": text})
    assert pyogrio.read_info(out, layer="polygons")["layer_metadata"]["sources"] == text


@pytest.mark.parametrize("folder", ["../77_X", "77_X/..", "77_X/sub", "77_x", "77_X\n", "", "x", "77_",
                                    "/etc", "77_漢", ".."])
def test_edge_write_polygons_refuses_bad_folder_and_writes_nothing(tmp_path, folder):
    roots = make_roots(tmp_path)
    with pytest.raises(ValueError):
        segment.write_polygons(two_polygons(), roots, folder, CHUNK, META)
    assert not list(Path(tmp_path).rglob("*.gpkg"))
    assert not list((Path(roots["semseg"])).iterdir())


@pytest.mark.parametrize("chunk", ["r02_c3", "../r1_c1", "r1_c1.gpkg", "", None])
def test_edge_write_polygons_refuses_bad_chunk(tmp_path, chunk):
    roots = make_roots(tmp_path)
    with pytest.raises(ValueError):
        segment.write_polygons(two_polygons(), roots, FOLDER, chunk, META)


def test_edge_write_polygons_failure_in_the_copy_leaves_no_file_and_no_part(tmp_path, monkeypatch):
    roots = make_roots(tmp_path)

    def broken(target, data, roots_):
        raise OSError("disk full")

    monkeypatch.setattr(segment.storage, "safe_write", broken)
    with pytest.raises(OSError):
        segment.write_polygons(two_polygons(), roots, FOLDER, CHUNK, META)
    assert not list(Path(tmp_path).rglob("*.gpkg*"))


def test_edge_write_polygons_rerun_replaces_the_file_whole(tmp_path):
    roots = make_roots(tmp_path)
    first = segment.write_polygons(two_polygons(), roots, FOLDER, CHUNK, META)
    again = segment.write_polygons(polygons([mask((10, 40), (10, 40))]), roots, FOLDER, CHUNK, META)
    assert first == again
    assert pyogrio.read_info(again, layer="polygons")["features"] == 1
    assert [p.name for p in again.parent.iterdir()] == [f"{CHUNK}.gpkg"]


def test_edge_write_polygons_ids_of_another_run_or_chunk_are_refused(tmp_path):
    roots = make_roots(tmp_path)
    gdf = two_polygons()
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, FOLDER, CHUNK, {**META, "run_id": "20261111-1201-sam2"})
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, FOLDER, "r2295_c2395", META)
    assert not list(Path(tmp_path).rglob("*.gpkg"))


def test_edge_write_polygons_share_missing_exit_3_and_no_file(tmp_path):
    roots = make_roots(tmp_path)
    gdf = two_polygons()
    Path(roots["share"]).rename(tmp_path / "gone")
    with pytest.raises(SystemExit) as info:
        segment.write_polygons(gdf, roots, FOLDER, CHUNK, META)
    assert info.value.code == 3


@pytest.mark.parametrize("damage", ["duplicate_id", "no_id_column", "outside_chunk", "invalid_geometry"])
def test_edge_write_polygons_refuses_a_damaged_frame_with_value_error(tmp_path, damage):
    roots = make_roots(tmp_path)
    gdf = two_polygons()
    if damage == "duplicate_id":
        gdf["polygon_id"] = gdf["polygon_id"].iloc[0]
    elif damage == "no_id_column":
        gdf = gdf.drop(columns="polygon_id")
    elif damage == "outside_chunk":
        gdf.geometry = [box(0, 0, 10, 10), gdf.geometry.iloc[1]]
        gdf = gdf.set_crs(5070, allow_override=True)
    else:
        gdf.geometry = [wkt.loads("POLYGON((0 0, 10 10, 10 0, 0 10, 0 0))"), gdf.geometry.iloc[1]]
        gdf = gdf.set_crs(5070, allow_override=True)
    with pytest.raises(ValueError):
        segment.write_polygons(gdf, roots, FOLDER, CHUNK, META)
    assert not list(Path(tmp_path).rglob("*.gpkg"))


# ------------------------------------------------------------------ write_run_log and run_chunks

def test_edge_run_log_text_survives_unicode_and_hostile_values(tmp_path):
    roots = make_roots(tmp_path)
    text = "<b>x</b> 漢\U0001F600 =cmd\r\nline2\x00"
    path = segment.write_run_log(roots, RUN, model_id=text, fake=True, memory_before_mb=1,
                                 memory_after_mb=None, started="a", finished="b", chunks=[CHUNK],
                                 chunks_done=[], error=text)
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["model_id"] == text and log["error"] == text and log["memory_after_mb"] is None
    assert path.read_bytes().endswith(b"\n") and b"\r" not in path.read_bytes()


def test_edge_run_log_rejects_bad_run_id_and_path_tricks(tmp_path):
    roots = make_roots(tmp_path)
    for bad in ("../x", "20261111-1200-sam2/../../a", "20261111-2561-sam2", "20260230-1200-sam2", ""):
        with pytest.raises(ValueError):
            segment.write_run_log(roots, bad, model_id="m", fake=True, memory_before_mb=1,
                                  memory_after_mb=1, started="a", finished="b", chunks=[],
                                  chunks_done=[])
    assert not (Path(roots["semseg"]) / "runs").exists()


def test_edge_run_log_gpu_not_available_values_are_logged_as_null(tmp_path):
    """`[N/A]` fields of nvidia-smi on the GB10 become null, and the run still works."""
    def fake_smi(cmd, **kwargs):
        text = ("NVIDIA GB10, [N/A], [N/A], [N/A], [N/A]\n" if "--query-gpu=" + setup_check.GPU_FIELDS in cmd
                else "123, python, [N/A]\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    roots, model, probe = one_run(tmp_path)
    segment.run_chunks([CHUNK], model, roots, FOLDER, CFG, run_id=RUN, probe=probe,
                       gpu_query=lambda: setup_check.query_gpu(fake_smi))
    log = read_log(roots)
    assert log["gpu_before"]["gpus"][0]["memory_used_mb"] is None and log["gpu_after"]["found"] is True


def test_edge_run_log_nvidia_smi_missing_is_not_an_error(tmp_path):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    roots, model, probe = one_run(tmp_path)
    segment.run_chunks([CHUNK], model, roots, FOLDER, CFG, run_id=RUN, probe=probe,
                       gpu_query=lambda: setup_check.query_gpu(missing))
    assert read_log(roots)["gpu_before"]["found"] is False


def test_edge_failure_in_the_middle_logs_the_chunks_done_and_keeps_the_first_file(tmp_path):
    chunks = ("r2295_c2394", "r2295_c2395", "r2296_c2394")
    roots, model, probe = one_run(tmp_path, chunk_ids=chunks, model=FakeModel(fail_on_call=2))
    with pytest.raises(RuntimeError):
        run(roots, model, probe, chunk_ids=chunks)
    log = read_log(roots)
    assert log["chunks_done"] == ["r2295_c2394"] and log["error"].startswith("RuntimeError")
    names = sorted(p.name for p in (Path(roots["semseg"]) / FOLDER / "polygons").iterdir())
    assert names == ["r2295_c2394.gpkg"]


def test_edge_run_chunks_rejects_bad_chunk_in_the_list_before_any_work(tmp_path):
    roots, model, probe = one_run(tmp_path)
    with pytest.raises(ValueError):
        run(roots, model, probe, chunk_ids=(CHUNK, "r01_c2"))
    assert model.calls == [] and probe.count == 0
    assert not (Path(roots["semseg"]) / "runs").exists()


def test_edge_run_chunks_folder_with_path_trick_is_refused(tmp_path):
    roots, model, probe = one_run(tmp_path)
    with pytest.raises(ValueError):
        segment.run_chunks([CHUNK], model, roots, "../77_X", CFG, run_id=RUN, probe=probe)
    assert model.calls == []


def test_edge_interrupt_is_written_in_the_log_error_field(tmp_path):
    class Interrupt(FakeModel):
        def generate(self, rgb):
            raise KeyboardInterrupt()

    roots, _, probe = one_run(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        run(roots, Interrupt(), probe)
    log = read_log(roots)
    assert log["chunks_done"] == [] and log["error"]


@pytest.mark.parametrize("which", ["probe", "gpu"])
def test_edge_log_is_written_when_the_memory_read_after_the_run_fails(tmp_path, which):
    roots, model, _ = one_run(tmp_path, model=FakeModel(fail_on_call=1))
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        if which == "probe" and calls["n"] >= 2:
            raise RuntimeError("free vanished")
        return 30000

    gcalls = {"n": 0}

    def gpu():
        gcalls["n"] += 1
        if which == "gpu" and gcalls["n"] >= 2:
            raise OSError("nvidia-smi vanished")
        return {"found": False, "gpus": [], "apps": []}

    with pytest.raises(RuntimeError, match="fake model failure"):
        segment.run_chunks([CHUNK], model, roots, FOLDER, CFG, run_id=RUN, probe=probe, gpu_query=gpu)
    assert read_log(roots)["error"].startswith("RuntimeError: fake model failure")


def test_edge_two_runs_in_the_same_minute_do_not_destroy_the_first_log(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(30000),
                 loader=Loader(), now=NOW)
    first = (Path(roots["semseg"]) / "runs").glob("*/log.json")
    first_text = {p: p.read_text() for p in first}
    config2 = {**config, "demo": {"pid": 77, "chunks": [CHUNK, "r2300_c2300"]}}
    with pytest.raises(FileExistsError):  # D49: the run folder of the first run exists
        segment.main(["--demo", "--run"], roots=roots, config=config2, probe=Probe(30000),
                     loader=Loader(), now=NOW + timedelta(seconds=30))
    after = {p: p.read_text() for p in (Path(roots["semseg"]) / "runs").glob("*/log.json")}
    assert all(after.get(p) == text for p, text in first_text.items())     # the first log is unchanged


# ------------------------------------------------------------------ memory check and loader

def free_run(stdout, code=0):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")
    return runner


@pytest.mark.parametrize("stdout, code", [
    ("", 0), ("garbage\n", 0), ("Mem: abc def\n", 0), ("Mem: 1 2 3 4\n", 0),
    ("Mem: 1 2 3 4 5 xyz\n", 0), ("Mem: 1 2 3 4 5 [N/A]\n", 0),
    ("Mem: 1 2 3 4 5 6 7\n", 1), ("Mem: 1 2 3 4 5 6 7\n", 127),
], ids=["empty", "garbage", "short", "five_columns", "text_value", "na_value", "exit_1", "exit_127"])
def test_edge_read_available_mb_bad_output_or_exit_raises_runtime_error(stdout, code):
    with pytest.raises(RuntimeError):
        segment.read_available_mb(free_run(stdout, code))


def test_edge_read_available_mb_missing_command_raises_runtime_error():
    def missing(cmd, **kwargs):
        raise FileNotFoundError("free")

    with pytest.raises(RuntimeError):
        segment.read_available_mb(missing)


def test_edge_read_available_mb_timeout_raises_runtime_error():
    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 30)

    with pytest.raises(RuntimeError):
        segment.read_available_mb(slow)


def test_edge_read_available_mb_reads_the_available_column_not_free():
    out = ("              total   used   free  shared  buff/cache  available\n"
           "Mem:          124546 101451  1000   200   22000   23095\n")
    assert segment.read_available_mb(free_run(out)) == 23095


@pytest.mark.parametrize("available, loads", [(15999, False), (16000, True), (16001, True), (0, False), (-1, False)])
def test_edge_memory_limit_exactly_at_the_need_runs_and_below_stops(tmp_path, available, loads):
    roots = make_roots(tmp_path)
    loader = Loader()
    config = {"segment": {"min_free_mb": 16000}}
    if loads:
        segment.load_sam2(roots, config, probe=lambda: available, loader=loader, environ={})
        assert loader.calls == 1
    else:
        with pytest.raises(SystemExit) as info:
            segment.load_sam2(roots, config, probe=lambda: available, loader=loader, environ={})
        assert info.value.code == 4 and loader.calls == 0


def test_edge_memory_probe_failure_loads_nothing(tmp_path):
    roots = make_roots(tmp_path)
    loader = Loader()

    def broken():
        raise RuntimeError("cannot read the available memory")

    with pytest.raises(RuntimeError):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 16000}}, probe=broken, loader=loader,
                          environ={})
    assert loader.calls == 0


@pytest.mark.parametrize("value", [-1, 16000.0, "16000", None, True, [16000]],
                         ids=["negative", "float", "string", "none", "bool", "list"])
def test_edge_min_free_mb_of_wrong_type_or_sign_is_refused_and_loads_nothing(tmp_path, value):
    roots = make_roots(tmp_path)
    loader = Loader()
    with pytest.raises(ValueError):
        segment.load_sam2(roots, {"segment": {"min_free_mb": value}}, probe=lambda: 99999, loader=loader,
                          environ={})
    assert loader.calls == 0


@pytest.mark.parametrize("config", [None, {}, {"segment": None}, {"segment": {}}, {"segment": []}])
def test_edge_missing_segment_section_is_refused_and_loads_nothing(tmp_path, config):
    roots = make_roots(tmp_path)
    loader = Loader()
    with pytest.raises(ValueError):
        segment.load_sam2(roots, config, probe=lambda: 99999, loader=loader, environ={})
    assert loader.calls == 0


def test_edge_min_free_mb_zero_must_not_switch_the_memory_stop_off(tmp_path):
    roots = make_roots(tmp_path)
    loader = Loader()
    with pytest.raises((ValueError, SystemExit)):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 0}}, probe=lambda: 5, loader=loader, environ={})
    assert loader.calls == 0


def test_edge_config_file_with_min_free_mb_zero_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("share_root: /x\naois: [1]\nsegment: {min_free_mb: 0}\n")
    with pytest.raises(semseg_config.ConfigError):
        semseg_config.load_config(path)


def test_edge_loader_not_called_when_hf_home_is_off_the_share_even_with_enough_memory(tmp_path):
    roots = make_roots(tmp_path)
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(ValueError):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 8000}}, probe=probe, loader=loader,
                          environ={"HF_HOME": "/tmp/not-on-the-share"})
    assert loader.calls == 0 and probe.count == 0


@pytest.mark.parametrize("value", ["1", "0", "false", "anything"])
def test_edge_default_loader_refuses_when_fake_env_is_set_even_for_0_or_false(tmp_path, value, monkeypatch):
    """Any non-empty value of SEMSEG_FAKE_MODEL stops the real build (safe side)."""
    built = []
    monkeypatch.setattr(segment, "_build_real_model", lambda model_id: built.append(model_id))
    roots = make_roots(tmp_path)
    with pytest.raises(RuntimeError, match="SEMSEG_FAKE_MODEL"):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 8000}}, probe=lambda: 99999,
                          environ={segment.FAKE_ENV: value})
    assert built == []


def test_edge_default_loader_passes_the_model_id_to_the_builder_when_fake_env_is_unset(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(segment, "_build_real_model", lambda model_id: built.append(model_id) or "model")
    roots = make_roots(tmp_path)
    config = {"segment": {"min_free_mb": 8000, "model_id": "sam2-hiera-small"}}
    assert segment.load_sam2(roots, config, probe=lambda: 99999, environ={}) == "model"
    assert built == ["sam2-hiera-small"]


@pytest.mark.parametrize("model_id", ["", "  ", 5, None, ["x"]])
def test_edge_bad_model_id_is_refused_before_the_probe_and_load(tmp_path, model_id):
    roots = make_roots(tmp_path)
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(ValueError):
        segment.load_sam2(roots, {"segment": {"min_free_mb": 8000, "model_id": model_id}}, probe=probe,
                          loader=loader, environ={})
    assert loader.calls == 0 and probe.count == 0


def test_edge_loader_injected_skips_nothing_the_memory_check_still_runs(tmp_path):
    """FM-72: an injected loader does not skip the memory check."""
    roots = make_roots(tmp_path)
    loader, probe = Loader(), Probe(10)
    with pytest.raises(SystemExit) as info:
        segment.load_sam2(roots, {"segment": {"min_free_mb": 16000}}, probe=probe, loader=loader, environ={})
    assert info.value.code == 4 and probe.count == 1 and loader.calls == 0


# ------------------------------------------------------------------ CLI

def test_edge_cli_run_without_demo_is_refused(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(SystemExit) as info:
        segment.main(["--run"], roots=roots, config=config, probe=probe, loader=loader, now=NOW)
    assert info.value.code == 2 and loader.calls == 0 and probe.count == 0


@pytest.mark.parametrize("extra", [["--chunks", "r1_c1"], ["--run=1"], ["extra"]])
def test_edge_cli_unknown_or_extra_arguments_are_refused(tmp_path, extra):
    roots, config = demo_setup(tmp_path, [CHUNK])
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo", "--run", *extra], roots=roots, config=config, probe=probe,
                     loader=loader, now=NOW)
    assert info.value.code == 2 and loader.calls == 0 and probe.count == 0


def test_edge_cli_repeated_demo_flag_still_runs_only_the_demo_list(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    assert segment.main(["--demo", "--demo", "--run"], roots=roots, config=config, probe=Probe(30000),
                        loader=Loader(), now=NOW) == 0


def test_edge_cli_low_memory_exit_4_then_nothing_in_polygons_and_log_says_why(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    loader = Loader()
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(1), loader=loader, now=NOW)
    assert info.value.code == 4 and loader.calls == 0
    log = read_log(roots)
    assert log["error"] and log["chunks_done"] == [] and log["fake"] is False
    assert not (Path(roots["semseg"]) / FOLDER).exists()


def test_edge_cli_chunk_without_naip_file_logs_and_keeps_earlier_polygons(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    config = {**config, "demo": {"pid": 77, "chunks": [CHUNK, "r2300_c2300"]}}
    with pytest.raises(FileNotFoundError):
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(30000),
                     loader=Loader(), now=NOW)
    log = read_log(roots)
    assert log["chunks_done"] == [CHUNK] and "FileNotFoundError" in log["error"]


def test_edge_cli_chunk_id_outside_the_grid_is_logged_not_written(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    config = {**config, "demo": {"pid": 77, "chunks": ["r99999999_c1"]}}
    with pytest.raises((ValueError, FileNotFoundError)):
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(30000),
                     loader=Loader(), now=NOW)
    assert not list(Path(tmp_path).rglob("*.gpkg"))


def test_edge_cli_year_boundary_run_id():
    now = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert standard.run_id(now, segment.METHOD) == "20261231-2359-sam2"
    leap = datetime(2028, 2, 29, 0, 0, tzinfo=timezone.utc)
    assert standard.run_id(leap, segment.METHOD) == "20280229-0000-sam2"


def test_edge_clock_of_the_cli_is_utc():
    assert segment._now().utcoffset() == timedelta(0)
    plus9 = datetime(2026, 11, 11, 23, 30, tzinfo=timezone(timedelta(hours=9)))
    assert segment._utc_text(plus9) == "2026-11-11T14:30:00Z"


def test_edge_cli_share_missing_exit_3_and_nothing_written(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    Path(roots["share"]).rename(tmp_path / "gone")
    loader = Loader()
    with pytest.raises(SystemExit) as info:
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(99999), loader=loader,
                     now=NOW)
    assert info.value.code == 3 and loader.calls == 0


def test_edge_cli_stack_folder_missing_or_double_is_refused(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    (Path(roots["stack"]) / "77_OTHER").mkdir()
    loader = Loader()
    with pytest.raises(ValueError):
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(99999), loader=loader,
                     now=NOW)
    assert loader.calls == 0


def test_edge_cli_config_without_demo_is_a_clear_error(tmp_path):
    roots, config = demo_setup(tmp_path, [CHUNK])
    config = {k: v for k, v in config.items() if k != "demo"}
    with pytest.raises((ValueError, SystemExit)):
        segment.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(99999),
                     loader=Loader(), now=NOW)


# ------------------------------------------------------------------ config file

def cfg_file(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text("share_root: /x\naois: [77]\n" + text)
    return path


@pytest.mark.parametrize("text", [
    "segment: {min_free_mb: -1}",
    "segment: {min_free_mb: 16000, min_area_m2: .nan}",
    "segment: {min_free_mb: 16000, min_area_m2: .inf}",
    "segment: {min_free_mb: 16000, min_area_m2: -2}",
    "segment: {min_free_mb: 16000, simplify_tolerance_m: -0.1}",
    "segment: {min_free_mb: 16000, simplify_tolerance_m: '0.5'}",
    "segment: {min_free_mb: 16000, simplify_tolerance_m: true}",
    "segment: {min_free_mb: 16000.5}",
    "segment: {min_free_mb: 16000, min_free_mb: 1}",
    "segment: [1, 2]",
    "segment: 5",
    "demo: {pid: 77, chunks: [r1_c1, r1_c1]}",
    "demo: {pid: 77, chunks: [r01_c1]}",
    "demo: {pid: 77, chunks: [r1_c1, 5]}",
    "demo: {pid: 77, chunks: ['r1_c1\\n']}",
    "demo: {pid: 77, chunks: ['r٣_c1']}",
    "demo: {pid: 77, chunks: [R1_C1]}",
    "demo: {pid: 77, chunks: []}",
    "demo: {pid: 77, chunks: r1_c1}",
    "demo: {pid: 77}",
    "demo: {chunks: [r1_c1]}",
    "demo: {pid: 0, chunks: [r1_c1]}",
    "demo: {pid: true, chunks: [r1_c1]}",
    "demo: {pid: '77', chunks: [r1_c1]}",
    "demo: []",
])
def test_edge_config_file_bad_segment_or_demo_is_refused(tmp_path, text):
    with pytest.raises(semseg_config.ConfigError):
        semseg_config.load_config(cfg_file(tmp_path, text))


@pytest.mark.parametrize("text", [
    "", "segment: {}", "segment: {min_free_mb: 16000}",
    "segment: {min_free_mb: 16000, simplify_tolerance_m: 0, min_area_m2: 0}",
    "demo: {pid: 77, chunks: [r1_c1]}",
])
def test_edge_config_file_missing_optional_keys_are_allowed(tmp_path, text):
    assert semseg_config.load_config(cfg_file(tmp_path, text))["aois"] == [77]


# ------------------------------------------------------------------ real NAIP file, read only

def test_edge_real_naip_file_has_no_colour_tags_so_band_order_is_not_checkable():
    """Read only. The real file has 5 uint8 bands with the tags gray, undefined, ...: the code
    cannot verify that bands 1 to 3 are R, G, B. This test pins the fact, to fail if the data change."""
    roots = storage.load_roots()
    storage.guard_share(roots)
    path = Path(roots["fgt"]) / "77_GRAPEVINE_LAKE" / "naip" / f"{CHUNK}.tif"
    with rasterio.open(path) as src:
        assert src.count >= 3
        assert src.colorinterp[1:] == (ColorInterp.undefined,) * (src.count - 1) or src.count == 3
