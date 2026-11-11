"""Edge-hunter tests of task 1.3 (`semseg.label`, D51 to D57). Factory edge hunter, 2026-11-11.

Every test uses fake models, synthetic rasters and `tmp_path` roots (D22). Faults that the code still
has are `xfail(strict=True)` with the FM row; see `.factory/1.3/edges.md`.
"""

import json
import math

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, box

import test_label as base
from semseg import config as semseg_config
from semseg import label, segment, standard
from test_label import (BOUNDS, CFG, CHUNK, FOLDER, LMETA, LRUN, META, N, T05, T30, XMIN, YMAX,
                        FakeText, Probe, do_run, gpkg, grids, make_roots, mask, px, setup_run,
                        unions_for, vote, write_base, write_layer, write_naip, write_stack)


def log_of(roots, run_id=LRUN):
    return json.loads((roots["semseg"] / "runs" / run_id / "log.json").read_text())


# ------------------------------------------------------------------ vote: geometry

def test_edge_vote_polygon_exactly_on_a_pixel_edge_and_corner():
    """A polygon whose side lies on a pixel edge takes only that pixel; a corner pixel has area 0."""
    nlcd, classf = grids(71)
    nlcd[3, 4] = 41
    nlcd[3, 5] = 42                      # right neighbour: only touched
    nlcd[4, 5] = 43                      # diagonal: only the corner touches
    assert vote(px(3, 4), nlcd, classf) == (41, 1.0)
    assert [(r, c) for r, c, _ in label.pixel_weights(px(3, 4), T30, nlcd.shape)] == [(3, 4)]


def test_edge_vote_polygon_over_four_pixels_meeting_at_a_corner_is_a_four_way_tie():
    nlcd, classf = grids(71)
    nlcd[3, 4], nlcd[3, 5], nlcd[4, 4], nlcd[4, 5] = 43, 42, 41, 52
    polygon = box(XMIN + 4 * 30 + 15, YMAX - 5 * 30 - 15, XMIN + 5 * 30 + 15, YMAX - 4 * 30 + 15)
    assert vote(polygon, nlcd, classf) == (41, 0.25)


def test_edge_vote_polygon_on_the_edge_of_the_map_and_chunk_window():
    """The last pixel row and column (31) and the first (0) work."""
    nlcd, classf = grids(71)
    nlcd[31, 31] = 90
    assert vote(px(31, 31), nlcd, classf) == (90, 1.0)
    assert vote(px(0, 0), nlcd, classf) == (71, 1.0)


def test_edge_vote_polygon_outside_the_map_extent_is_refused():
    nlcd, classf = grids(71)
    outside = box(XMIN - 10, YMAX - 10, XMIN + 10, YMAX)
    with pytest.raises(ValueError):
        vote(outside, nlcd, classf)
    far = box(XMIN + 960 + 1, YMAX - 10, XMIN + 960 + 5, YMAX)
    with pytest.raises(ValueError):
        vote(far, nlcd, classf)


def test_edge_vote_polygon_with_a_hole_does_not_count_the_hole():
    nlcd, classf = grids(71)
    nlcd[0, 0], nlcd[0, 1] = 41, 42
    ring = Polygon(box(XMIN, YMAX - 60, XMIN + 60, YMAX).exterior.coords,
                   [list(box(XMIN + 2, YMAX - 28, XMIN + 28, YMAX - 2).exterior.coords)])
    code, p = vote(ring, nlcd, classf)
    area0 = 30 * 30 - 26 * 26             # pixel (0, 0) minus the hole
    area1 = 30 * 30
    nlcd[1, 0] = nlcd[1, 1] = 71          # the second row holds more area, but is 71 (the largest)
    assert code == 71
    assert math.isclose(p, round((2 * 900) / (area0 + area1 + 1800), 9), abs_tol=1e-9)


def test_edge_vote_multipolygon_is_refused():
    nlcd, classf = grids(71)
    multi = MultiPolygon([px(0, 0), px(5, 5)])
    with pytest.raises(ValueError):
        vote(multi, nlcd, classf)


def test_edge_vote_area_tie_within_1e12_is_stable_and_lowest_code():
    """Two codes with areas that differ by 1e-12: the 9-decimal shares tie, the lowest code wins every time."""
    nlcd, classf = grids(71)
    nlcd[0, 0], nlcd[0, 1] = 52, 41
    polygon = box(XMIN + 15, YMAX - 10, XMIN + 45 + 1e-12, YMAX)
    results = {vote(polygon, nlcd, classf) for _ in range(5)}
    assert results == {(41, 0.5)}


def test_edge_vote_share_rounds_to_9_decimals():
    nlcd, classf = grids(71)
    nlcd[0, 0] = 41
    polygon = box(XMIN + 29, YMAX - 10, XMIN + 31, YMAX)
    nlcd[0, 1] = 42
    code, p = vote(polygon, nlcd, classf)
    assert (code, p) == (41, 0.5)
    third = box(XMIN, YMAX - 30, XMIN + 90, YMAX)
    nlcd[0, 2] = 43
    code, p = vote(third, nlcd, classf)
    assert (code, p) == (41, 0.333333333)


# ------------------------------------------------------------------ vote: pixel values

@pytest.mark.parametrize("nlcd,classf,want", [
    (12, 0, 12), (72, 0, 0), (73, 0, 0), (74, 0, 0),          # 12 is a code (D60); the rest none
    (250, 0, None), (250, 5, 33), (250, 2, 20),               # no data in one layer
    (11, 5, 33), (41, 9, 95), (24, 26, 52), (12, 17, 51),    # classf with one code wins over NLCD
    (22, 0, 20), (23, 2, 20), (21, 2, 21), (12, 2, 12), (72, 2, 20),
    (41, 10, 41), (82, 13, 82), (11, 1, 11),
])
def test_edge_vote_code_table(nlcd, classf, want):
    assert standard.vote_code(nlcd, classf) == want


@pytest.mark.parametrize("nlcd,classf", [(0, 0), (255, 0), (200, 0), (71, 30), (71, 255), (71, -1), (-1, 0)])
def test_edge_vote_code_out_of_range_is_refused(nlcd, classf):
    with pytest.raises(ValueError):
        standard.vote_code(np.int16(nlcd), np.int16(classf))


def test_edge_vote_code_numpy_integers_and_float_nan():
    assert standard.vote_code(np.uint8(71), np.uint8(0)) == 71
    for bad in (float("nan"), 71.0, "71", None, True):
        with pytest.raises(ValueError):
            standard.vote_code(bad, 0)
        with pytest.raises(ValueError):
            standard.vote_code(71, bad)


def test_edge_vote_polygon_over_fill_outside_a_small_map_counts_only_valid_area(tmp_path):
    """A map smaller than the window: the outside is no data; a polygon on the outside gets (0, 0.0)."""
    roots = make_roots(tmp_path)
    small = np.full((4, 4), 41, dtype=np.uint8)
    write_stack(roots, nlcd=small, classf=np.zeros((4, 4), dtype=np.uint8))
    nlcd, classf, tr = label.read_vote_layers(roots, FOLDER, CHUNK)
    assert nlcd[0, 0] == 41 and nlcd[10, 10] == 250 and classf[10, 10] == 0
    assert label.vote_polygon(px(10, 10), nlcd, classf, tr) == (0, 0.0)
    both = box(XMIN + 90, YMAX - 30, XMIN + 150, YMAX)
    assert label.vote_polygon(both, nlcd, classf, tr) == (41, 1.0)


def test_edge_vote_nodata_only_in_one_layer_in_a_file(tmp_path):
    roots = make_roots(tmp_path)
    nlcd, classf = grids(250, 0)
    classf[0, 0] = 26
    nlcd[0, 1], classf[0, 1] = 41, 0
    write_stack(roots, nlcd=nlcd, classf=classf)
    n, c, tr = label.read_vote_layers(roots, FOLDER, CHUNK)
    assert label.vote_polygon(px(0, 0), n, c, tr) == (52, 1.0)
    assert label.vote_polygon(px(0, 1), n, c, tr) == (41, 1.0)
    assert label.vote_polygon(px(0, 2), n, c, tr) == (0, 0.0)


# ------------------------------------------------------------------ vote: maps with a wrong grid

@pytest.mark.parametrize("kw", [
    {"transform": from_origin(XMIN, YMAX, 10, 10)},
    {"transform": from_origin(XMIN + 15, YMAX, 30, 30)},                 # off-lattice origin
    {"transform": from_origin(XMIN, YMAX + 7, 30, 30)},
    {"transform": from_origin(XMIN, YMAX, 30, 30) @ Affine.rotation(1)},
    {"crs": "EPSG:4326"},
    {"crs": "EPSG:5069"},
])
def test_edge_map_with_another_grid_is_refused_not_resampled(tmp_path, kw):
    roots = make_roots(tmp_path)
    write_stack(roots, **kw)
    with pytest.raises(ValueError):
        label.read_vote_layers(roots, FOLDER, CHUNK)


def test_edge_map_south_up_or_uint16_or_no_crs_is_refused(tmp_path):
    roots = make_roots(tmp_path)
    nlcd, classf = grids()
    write_layer(roots["stack"] / FOLDER / "nlcd_2022.tif", nlcd, 250)
    write_layer(roots["stack"] / FOLDER / "classf_2022.tif", classf, 0,
                transform=rasterio.Affine(30, 0, XMIN, 0, 30, YMAX - 960))
    with pytest.raises(ValueError):
        label.read_vote_layers(roots, FOLDER, CHUNK)
    path = roots["stack"] / FOLDER / "classf_2022.tif"
    with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1, dtype="uint16",
                       crs="EPSG:5070", transform=T30) as dst:
        dst.write(classf.astype("uint16"), 1)
    with pytest.raises(ValueError):
        label.read_vote_layers(roots, FOLDER, CHUNK)


def test_edge_layers_with_different_extent_but_same_lattice_are_refused(tmp_path):
    roots = make_roots(tmp_path)
    n, c = grids()
    write_layer(roots["stack"] / FOLDER / "nlcd_2022.tif", n, 250)
    write_layer(roots["stack"] / FOLDER / "classf_2022.tif", c, 0, transform=from_origin(XMIN - 30, YMAX, 30, 30))
    with pytest.raises(ValueError):
        label.read_vote_layers(roots, FOLDER, CHUNK)


def test_edge_read_vote_layers_hostile_names(tmp_path):
    roots = make_roots(tmp_path)
    write_stack(roots)
    for folder in ("../77_TEST_LAKE", "77_TEST_LAKE/..", "", None):
        with pytest.raises(ValueError):
            label.read_vote_layers(roots, folder, CHUNK)
    for chunk in ("../x", "", None, CHUNK + "x"):
        with pytest.raises(ValueError):
            label.read_vote_layers(roots, FOLDER, chunk)


def test_edge_vote_layers_with_a_value_not_in_the_legend_inside_the_chunk_are_refused(tmp_path):
    roots = make_roots(tmp_path)
    n, c = grids()
    n[5, 5] = 99
    write_stack(roots, nlcd=n, classf=c)
    nn, cc, tr = label.read_vote_layers(roots, FOLDER, CHUNK)
    with pytest.raises(ValueError):
        label.vote_polygon(px(5, 5), nn, cc, tr)


@pytest.mark.xfail(strict=True, raises=IndexError, reason="FM-93: classf smaller than nlcd gives IndexError, not ValueError")
def test_edge_vote_polygon_with_maps_of_different_shape_is_a_value_error():
    nlcd, _ = grids(71)
    classf = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        vote(px(20, 20), nlcd, classf)


# ------------------------------------------------------------------ prompt

def test_edge_prompt_empty_mask_list_gives_all_false_unions():
    rgb = np.zeros((N, N, 3), dtype=np.uint8)
    unions = label.prompt_unions(FakeText(), rgb, CFG)
    assert len(unions) == len(standard.CODE_LIST) == 32 and not any(u.any() for u in unions.values())


@pytest.mark.xfail(strict=True, raises=TypeError, reason="FM-94: a model that returns None gives TypeError, not ValueError or no mask")
def test_edge_prompt_model_returning_none_is_a_clean_result():
    class NoneModel(FakeText):
        def predict(self, rgb, prompt):
            return None
    rgb = np.zeros((N, N, 3), dtype=np.uint8)
    try:
        unions = label.prompt_unions(NoneModel(), rgb, CFG)
    except ValueError:
        return
    assert not any(u.any() for u in unions.values())


def test_edge_prompt_mask_list_with_a_3d_array_and_generator():
    rgb = np.zeros((N, N, 3), dtype=np.uint8)
    stack = np.stack([mask((0, 5), (0, 5)), mask((10, 15), (10, 15))])
    unions = label.prompt_unions(FakeText({"building": stack}), rgb, CFG)
    assert unions[25].sum() == 50
    unions = label.prompt_unions(FakeText({"building": (m for m in stack)}), rgb, CFG)
    assert unions[25].sum() == 50


def test_edge_prompt_one_bad_mask_among_good_ones_is_refused_whole():
    rgb = np.zeros((N, N, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        label.prompt_unions(FakeText({"dam": [mask((0, 5), (0, 5)), np.zeros((N, N + 1), dtype=bool)]}), rgb, CFG)


def test_edge_prompt_model_returns_masks_for_a_prompt_not_in_the_table():
    """A fake that answers to a prompt that the table lacks changes nothing."""
    rgb = np.zeros((N, N, 3), dtype=np.uint8)
    unions = label.prompt_unions(FakeText({"unicorn": [mask((0, 5), (0, 5))]}), rgb, CFG)
    assert not any(u.any() for u in unions.values())


def test_edge_prompt_polygon_with_a_hole_excludes_hole_pixels():
    ring = Polygon(box(XMIN + 10, YMAX - 30, XMIN + 30, YMAX - 10).exterior.coords,
                   [list(box(XMIN + 15, YMAX - 25, XMIN + 25, YMAX - 15).exterior.coords)])
    pixels = label.polygon_pixels(ring, T05)
    assert pixels.sum() == 40 * 40 - 20 * 20
    inside_hole = mask((30, 50), (30, 50))
    assert label.prompt_polygon(pixels, unions_for(c25=inside_hole), 0.0) == (0, 0.0)


def test_edge_prompt_polygon_multipolygon_and_outside_chunk_are_refused():
    with pytest.raises(ValueError):
        label.polygon_pixels(MultiPolygon([px(0, 0), px(5, 5)]), T05)
    with pytest.raises(ValueError):
        label.polygon_pixels(box(XMIN - 1, YMAX - 5, XMIN + 5, YMAX), T05)


def test_edge_prompt_polygon_on_the_chunk_edge_and_corner_pixels():
    corner = box(XMIN, YMAX - 1, XMIN + 1, YMAX)
    pixels = label.polygon_pixels(corner, T05)
    assert pixels.sum() == 4 and pixels[:2, :2].all()
    last = box(XMIN + 960 - 1, YMAX - 960, XMIN + 960, YMAX - 960 + 1)
    assert label.polygon_pixels(last, T05)[-2:, -2:].all()


def test_edge_prompt_min_overlap_exactly_at_the_limit_and_one_pixel_below():
    pixels = mask((0, 10), (0, 10))              # n = 100
    at = mask((0, 5), (0, 10))                   # m = 50
    below = mask((0, 4), (0, 10))
    assert label.prompt_polygon(pixels, unions_for(c25=at), 0.5) == (25, 0.5)
    assert label.prompt_polygon(pixels, unions_for(c25=below), 0.5) == (0, 0.0)
    assert label.prompt_polygon(pixels, unions_for(c25=below), 0.4) == (25, 0.4)
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 10), (0, 10))), 1.0) == (25, 1.0)
    assert label.prompt_polygon(pixels, unions_for(c25=at), 0.0) == (25, 0.5)


def test_edge_prompt_p_rounding_third():
    pixels = mask((0, 3), (0, 1))
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 1), (0, 1))), 0.0) == (25, 0.333333333)


def test_edge_prompt_two_codes_with_equal_m_lowest_code_wins_with_empty_masks_between():
    pixels = mask((0, 10), (0, 10))
    u = unions_for(c90=mask((0, 5), (0, 10)), c11=mask((5, 10), (0, 10)), c50=ZEROS_LIKE())
    assert label.prompt_polygon(pixels, u, 0.0) == (11, 0.5)


def ZEROS_LIKE():
    return np.zeros((N, N), dtype=bool)


def test_edge_prompt_polygons_empty_frame():
    gdf = base.base_polygons().iloc[0:0]
    out = label.prompt_polygons(gdf, unions_for(), T05, 0.5)
    assert len(out) == 0 and list(out.columns) == ["code_prompt", "p_prompt"]
    out = label.vote_polygons(gdf, *grids(), T30)
    assert len(out) == 0 and list(out.columns) == ["code_vote", "p_vote"]


def test_edge_prompt_polygon_smaller_than_one_pixel_centre_in_the_run(tmp_path):
    roots = setup_run(tmp_path)
    tiny = box(XMIN + 10.05, YMAX - 10.25, XMIN + 10.2, YMAX - 10.05)
    gdf = segment.masks_to_polygons([mask((100, 140), (200, 260))], T05, BOUNDS, base.RUN, CHUNK, config=CFG)
    gdf.loc[len(gdf)] = ["x", 0.0, tiny] if False else gdf.iloc[0]
    gdf.at[gdf.index[-1], "geometry"] = tiny
    gdf.at[gdf.index[-1], "polygon_id"] = standard.polygon_id(CHUNK, base.RUN, 9)
    gdf.at[gdf.index[-1], "area_m2"] = tiny.area
    segment.write_polygons(gdf, roots, FOLDER, CHUNK, META)
    do_run(roots, FakeText({"building": [mask((0, N), (0, N))]}))
    out = gpd.read_file(gpkg(roots), layer="polygons")
    assert out.iloc[-1]["code_prompt"] == 0 and out.iloc[-1]["p_prompt"] == 0.0
    assert out.iloc[0]["code_prompt"] == 25 and out.iloc[0]["p_prompt"] == 1.0


# ------------------------------------------------------------------ output file and run

def replace_layer(roots, gdf, metadata=None):
    path = gpkg(roots)
    path.unlink()
    gdf.to_file(path, layer="polygons", driver="GPKG", engine="pyogrio",
                layer_metadata=META if metadata is None else metadata)
    return path


def test_edge_run_missing_polygon_file_leaves_log_with_34_code_rows(tmp_path):
    roots = setup_run(tmp_path)
    gpkg(roots).unlink()
    with pytest.raises(FileNotFoundError):
        do_run(roots)
    log = log_of(roots)
    assert len(log["codes"]) == 34 and "FileNotFoundError" in log["error"]
    assert not gpkg(roots).exists()


def test_edge_run_polygon_file_without_polygon_id_is_refused_and_kept(tmp_path):
    roots = setup_run(tmp_path)
    old = replace_layer(roots, base.base_polygons().drop(columns=["polygon_id"]))
    before = old.read_bytes()
    with pytest.raises(ValueError):
        do_run(roots)
    assert old.read_bytes() == before
    assert len(log_of(roots)["codes"]) == 34


def test_edge_run_polygon_file_without_ds21_metadata_is_refused_and_kept(tmp_path):
    roots = setup_run(tmp_path)
    old = replace_layer(roots, base.base_polygons(), metadata={"run_id": base.RUN})
    before = old.read_bytes()
    with pytest.raises(ValueError):
        do_run(roots)
    assert old.read_bytes() == before


def test_edge_run_polygon_file_that_is_garbage_is_an_error_and_kept(tmp_path):
    roots = setup_run(tmp_path)
    path = gpkg(roots)
    path.write_bytes(b"not a geopackage")
    with pytest.raises(Exception):
        do_run(roots)
    assert path.read_bytes() == b"not a geopackage"
    assert "error" in log_of(roots) and log_of(roots)["error"]


def test_edge_run_polygon_file_with_zero_rows_works(tmp_path):
    roots = setup_run(tmp_path)
    replace_layer(roots, base.base_polygons().iloc[0:0])
    do_run(roots)
    out = gpd.read_file(gpkg(roots), layer="polygons")
    assert len(out) == 0 and list(out.columns)[-5:-1] == base.FOUR
    assert sum(r["n_prompt"] + r["n_vote"] for r in log_of(roots)["codes"]) == 0


def test_edge_run_polygon_outside_the_chunk_box_is_refused_before_a_write(tmp_path):
    roots = setup_run(tmp_path)
    gdf = base.base_polygons()
    gdf.at[0, "geometry"] = box(XMIN - 50, YMAX - 10, XMIN - 40, YMAX)
    old = replace_layer(roots, gdf)
    before = old.read_bytes()
    with pytest.raises(ValueError):
        do_run(roots)
    assert old.read_bytes() == before


def test_edge_run_polygon_with_wrong_crs_in_the_file_is_refused(tmp_path):
    roots = setup_run(tmp_path)
    gdf = base.base_polygons().to_crs(5069)
    old = replace_layer(roots, gdf)
    before = old.read_bytes()
    with pytest.raises(ValueError):
        do_run(roots)
    assert old.read_bytes() == before


def test_edge_run_twice_replaces_the_columns_and_label_keys_not_duplicates(tmp_path):
    roots = setup_run(tmp_path)
    do_run(roots, FakeText({"building": [mask((100, 140), (200, 260))]}))
    do_run(roots, FakeText({"dam": [mask((100, 140), (200, 260))]}), run_id="20261112-0931-langsam")
    out = gpd.read_file(gpkg(roots), layer="polygons")
    cols = list(out.columns)
    assert cols == ["polygon_id", "area_m2", *base.FOUR, "geometry"]
    assert out.iloc[0]["code_prompt"] == 29
    info = pyogrio.read_info(gpkg(roots), layer="polygons")["layer_metadata"]
    assert info["label_run_id"] == "20261112-0931-langsam" and info["run_id"] == base.RUN
    assert set(info) == set(standard.METADATA_KEYS) | set(label.LABEL_KEYS)


def test_edge_run_ds21_keys_from_another_run_are_kept_not_overwritten(tmp_path):
    roots = setup_run(tmp_path)
    do_run(roots)
    info = pyogrio.read_info(gpkg(roots), layer="polygons")["layer_metadata"]
    for key in standard.METADATA_KEYS:
        assert info[key] == META[key] or key == "created"
    assert info["naip_year"] == "2022"


def test_edge_writer_metadata_extra_keys_do_not_leak_into_the_layer(tmp_path):
    roots = setup_run(tmp_path)
    gdf = gpd.read_file(gpkg(roots), layer="polygons")
    out = base.labelled(gdf)
    meta = {**META, "label_run_id": "evil", "extra": "x"}
    label.write_labelled_polygons(out, roots, FOLDER, CHUNK, meta, {**LMETA, "run_id": "evil", "other": "y"})
    info = pyogrio.read_info(gpkg(roots), layer="polygons")["layer_metadata"]
    assert info["run_id"] == base.RUN and info["label_run_id"] == LRUN
    assert "extra" not in info and "other" not in info


def test_edge_run_log_codes_exact_34_rows_sorted_with_names(tmp_path):
    roots = setup_run(tmp_path)
    do_run(roots, FakeText({"building": [mask((100, 140), (200, 260))]}))
    codes = log_of(roots)["codes"]
    assert [r["code"] for r in codes] == sorted([0, 20, *standard.CODE_LIST]) and len(codes) == 34
    assert all(set(r) == {"code", "name", "n_prompt", "n_vote", "area_prompt_m2", "area_vote_m2"} for r in codes)
    assert sum(r["n_prompt"] for r in codes) == 2 == sum(r["n_vote"] for r in codes)
    json.dumps(codes, allow_nan=False)


def test_edge_run_log_on_failure_in_second_chunk_keeps_codes_of_first(tmp_path):
    roots = setup_run(tmp_path)
    other = standard.chunk_id(base.ROW, base.COL + 1)
    with pytest.raises(FileNotFoundError):
        label.run_labels([CHUNK, other], FakeText(), roots, FOLDER, CFG, run_id=LRUN, probe=Probe(30000, 29000))
    log = log_of(roots)
    assert log["chunks_done"] == [CHUNK] and sum(r["n_vote"] for r in log["codes"]) == 2


def test_edge_run_folder_collision_refuses_before_any_write(tmp_path):
    roots = setup_run(tmp_path)
    (roots["semseg"] / "runs" / LRUN).mkdir(parents=True)
    before = gpkg(roots).read_bytes()
    with pytest.raises(FileExistsError):
        do_run(roots)
    assert gpkg(roots).read_bytes() == before


def test_edge_run_duplicate_chunk_in_list_second_pass_replaces(tmp_path):
    roots = setup_run(tmp_path)
    label.run_labels([CHUNK, CHUNK], FakeText(), roots, FOLDER, CFG, run_id=LRUN, probe=Probe(30000, 29000))
    out = gpd.read_file(gpkg(roots), layer="polygons")
    assert list(out.columns).count("code_prompt") == 1 and len(out) == 2


def test_edge_run_bad_run_id_methods_are_refused(tmp_path):
    """D54: the label run ID has the method `langsam`; a sam2 run ID would look like a 1.2 run."""
    roots = setup_run(tmp_path)
    with pytest.raises(ValueError):
        label.run_labels([CHUNK], FakeText(), roots, FOLDER, CFG, run_id="20261112-0930-sam2",
                         probe=Probe(30000, 29000))


@pytest.mark.parametrize("cp,pp", [(25, 0.0), (0, 0.0)])
def test_edge_writer_code_with_p_zero(tmp_path, cp, pp):
    roots = setup_run(tmp_path)
    gdf = base.labelled(gpd.read_file(gpkg(roots), layer="polygons"), cp=cp, pp=pp)
    if cp:
        with pytest.raises(ValueError):
            label.write_labelled_polygons(gdf, roots, FOLDER, CHUNK, META, LMETA)
    else:
        label.write_labelled_polygons(gdf, roots, FOLDER, CHUNK, META, LMETA)


# ------------------------------------------------------------------ D57

def run_with_tags(tmp_path, tags):
    roots = setup_run(tmp_path)
    write_naip(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif", tags=tags)
    return roots


# " 0.6" is not a case: GDAL strips a leading blank from a GTiff tag value on write (checked 2026-11-11).
@pytest.mark.parametrize("gsd", ["0.60", "0.6 ", "0.5", "", "0,6", "0.6\n", "٠.٦", "0.6m", "6e-1"])
def test_edge_d57_bad_gsd_strings_are_refused_in_read_chunk_and_label_run(tmp_path, gsd):
    roots = run_with_tags(tmp_path, {"gsd": gsd, "years": "2022"})
    with pytest.raises(ValueError):
        segment.read_chunk(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")
    before = gpkg(roots).read_bytes()
    with pytest.raises(ValueError):
        do_run(roots)
    assert gpkg(roots).read_bytes() == before


def test_edge_d57_gsd_as_number_in_the_writer_is_the_same_string_0_6(tmp_path):
    roots = run_with_tags(tmp_path, {"gsd": 0.6, "years": "2022"})
    segment.read_chunk(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")
    roots2 = run_with_tags(tmp_path / "b", {"gsd": 0.60, "years": 2022})
    assert segment.chunk_naip_year(roots2["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif") == 2022


def test_edge_d57_missing_gsd_tag_is_refused(tmp_path):
    roots = run_with_tags(tmp_path, {"years": "2022"})
    with pytest.raises(ValueError):
        segment.read_chunk(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")


# " 2022" is not a case: GDAL strips a leading blank from a GTiff tag value on write (checked 2026-11-11).
@pytest.mark.parametrize("years", ["2022;2021", "", "22", "2022 ", "2022\n", "20222", "２０２２",
                                   "٢٠٢٢", "2022,2021", "2022-2021", "+202", "-202", "20.2", "0x7e6"])
def test_edge_d57_bad_years_are_refused(tmp_path, years):
    roots = run_with_tags(tmp_path, {"gsd": "0.6", "years": years})
    with pytest.raises(ValueError):
        segment.chunk_naip_year(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")


def test_edge_d57_missing_years_tag_is_refused(tmp_path):
    roots = run_with_tags(tmp_path, {"gsd": "0.6"})
    with pytest.raises(ValueError):
        segment.chunk_naip_year(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")


@pytest.mark.parametrize("years", ["0000", "0022", "1066", "9999"])
@pytest.mark.xfail(strict=True, reason="FM-95: a 4-digit year outside the NAIP years (2003 and later) is accepted; '0022' becomes naip_year '22'")
def test_edge_d57_year_outside_the_naip_range_is_refused(tmp_path, years):
    roots = run_with_tags(tmp_path, {"gsd": "0.6", "years": years})
    with pytest.raises(ValueError):
        segment.chunk_naip_year(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")


def test_edge_d57_segment_run_1_2_same_errors_and_log_keys_after_refactor(tmp_path):
    """The shared helpers keep the 1.2 log keys; a bad tag is logged and no polygon file is written."""
    roots = make_roots(tmp_path)
    write_naip(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif", tags={"gsd": "0.5", "years": "2022"})

    class Seg:
        fake = True
        model_id = "fake"

        def generate(self, rgb):
            return [mask((0, 10), (0, 10))]
    with pytest.raises(ValueError):
        segment.run_chunks([CHUNK], Seg(), roots, FOLDER, CFG, run_id=base.RUN, probe=Probe(30000, 29000))
    log = log_of(roots, base.RUN)
    assert set(log) == {"chunks", "chunks_done", "created", "error", "fake", "finished", "gpu_after", "gpu_before",
                        "log_errors", "memory_after_mb", "memory_before_mb", "model_id", "started"} - {"created"} \
        or "codes" not in log
    assert "codes" not in log and log["chunks_done"] == [] and "gsd" in log["error"]
    assert not (roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg").exists()


# ------------------------------------------------------------------ config

def test_edge_label_config_types_and_bounds():
    for bad in ({"min_overlap": True}, {"min_overlap": "0.5"}, {"min_overlap": float("nan")},
                {"min_overlap": -0.01}, {"min_overlap": 1.01}, {"box_threshold": 0}, {"box_threshold": float("inf")},
                {"text_threshold": None}, {"model_id": ""}, {"model_id": "sam2-hiera-x"}, {"model_id": None},
                [], "x", 5):
        with pytest.raises(ValueError):
            semseg_config.label_settings(bad)
    assert semseg_config.label_settings({"min_overlap": 0})["min_overlap"] == 0
    assert semseg_config.label_settings({"min_overlap": 1})["min_overlap"] == 1
    assert semseg_config.label_settings(None) == semseg_config.LABEL_DEFAULTS


@pytest.mark.xfail(strict=True, reason="FM-96: an unknown key in `label` (a typo of min_overlap) is dropped silently and the default 0.5 applies")
def test_edge_label_config_unknown_key_is_refused():
    with pytest.raises(ValueError):
        semseg_config.label_settings({"min_overlp": 0.9})
