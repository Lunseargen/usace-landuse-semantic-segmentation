"""Tests of task 1.3 polygon labels (`semseg.label`): D51 to D56, FM-2, FM-85 to FM-92, R3.

API under test (names fixed here, see `.factory/1.3/tests.md`):
  standard.Code.prompt; standard.PROMPTS {code: prompt}; standard.NLCD_VOTE; standard.CLASSF_VOTE;
  standard.VOTE_NONE (0); standard.VOTE_NODATA (None); standard.vote_code(nlcd, classf) -> int | None;
  standard.check_machine_code(code); standard.check_vote_code(code)
  label.VOTE_LAYERS; label._label_config(config) -> dict
  label.read_vote_layers(roots, folder, chunk) -> (nlcd uint8 (32, 32), classf uint8 (32, 32), transform)
  label.pixel_weights(polygon, transform, shape) -> [(row, col, area_m2)]
  label.vote_polygon(polygon, nlcd, classf, transform) -> (code_vote, p_vote)
  label.vote_polygons(gdf, nlcd, classf, transform) -> DataFrame[code_vote, p_vote]
  label.prompt_unions(model, rgb, config) -> {code: bool (1920, 1920)}
  label.polygon_pixels(polygon, transform) -> bool (1920, 1920)
  label.prompt_polygon(pixels, unions, min_overlap) -> (code_prompt, p_prompt)
  label.prompt_polygons(gdf, unions, transform, min_overlap) -> DataFrame[code_prompt, p_prompt]
  label.write_labelled_polygons(gdf, roots, folder, chunk, metadata, label_metadata) -> Path
  label.count_codes(frames) -> list of 34 dicts
  label.run_labels(chunks, model, roots, folder, config, *, run_id, probe, gpu_query=None) -> [Path]
  label.main(argv=None, roots=None, config=None, probe=None, loader=None, now=None) -> int
  A text model has `fake`, `model_id` and `predict(rgb, prompt) -> sequence of bool masks`.

D22: no test builds a real LangSAM, GroundingDINO or transformers model. All write tests use fake
roots under `tmp_path` and synthetic rasters. The tests that cover DS-11, DS-12, DS-21, DS-22,
DS-29 and DS-30 are named in the docstrings.
"""

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from semseg import config as semseg_config
from semseg import segment
from semseg import standard
from semseg import storage

try:
    from semseg import label
except ImportError:                      # before the task: each test fails on its own (missing name)
    class _MissingLabel:
        def __getattr__(self, name):
            raise AttributeError(f"semseg.label does not exist yet: no name {name!r}")

    label = _MissingLabel()

ROOT = Path(__file__).resolve().parent.parent
ROW, COL = 2295, 2394
CHUNK = standard.chunk_id(ROW, COL)
BOUNDS = standard.chunk_bounds(ROW, COL)
XMIN, YMIN, XMAX, YMAX = BOUNDS
N = standard.PIXELS_PER_CHUNK
T05 = from_origin(XMIN, YMAX, 0.5, 0.5)
T30 = from_origin(XMIN, YMAX, 30, 30)
RUN = "20261111-1200-sam2"
LRUN = "20261112-0930-langsam"
FOLDER = "77_TEST_LAKE"
CFG = {"segment": {"min_free_mb": 16000, "simplify_tolerance_m": 0.0, "min_area_m2": 0.0},
       "label": {"model_id": "sam2-hiera-large", "box_threshold": 0.25, "text_threshold": 0.25,
                 "min_overlap": 0.5}}
META = {"run_id": RUN, "code_version": "a" * 40, "data_standard": "5", "naip_year": "2022",
        "sources": f"naip/{FOLDER}/{CHUNK}.tif", "created": "2026-11-11T12:00:00Z"}
LMETA = {"label_run_id": LRUN, "label_code_version": "b" * 40, "label_created": "2026-11-12T09:30:00Z",
         "label_sources": f"naip/{FOLDER}/{CHUNK}.tif nlcd_2022.tif classf_2022.tif fake"}
NOW = datetime(2026, 11, 12, 9, 30, tzinfo=timezone.utc)
FOUR = ["code_prompt", "p_prompt", "code_vote", "p_vote"]
ALL_PROMPTS = {
    11: "lake", 12: "snow and ice", 13: "river", 14: "pond", 21: "mowed lawn", 25: "building", 26: "paved road",
    27: "dirt road", 28: "parking lot", 29: "dam", 31: "bare ground", 32: "sandy beach",
    33: "rock outcrop", 34: "mud flat", 35: "gravel pit", 36: "boulder field", 41: "deciduous trees", 42: "evergreen trees",
    43: "mixed forest", 44: "tree canopy", 51: "dwarf shrubs", 52: "shrubs", 53: "deciduous shrubs",
    54: "evergreen shrubs", 55: "mixed shrubs", 71: "grassland",
    81: "pasture", 82: "crop field", 83: "bare farm field", 90: "swamp forest", 95: "marsh",
    96: "floating water plants",
}
ZEROS = np.zeros((N, N), dtype=bool)


# ------------------------------------------------------------------ helpers

def make_roots(base):
    roots = {"share": base / "share", "semseg": base / "share" / "semseg",
             "fgt": base / "share" / "fgt", "stack": base / "stack"}
    for path in roots.values():
        path.mkdir(parents=True)
    return roots


def mask(rows, cols):
    out = np.zeros((N, N), dtype=bool)
    out[rows[0]:rows[1], cols[0]:cols[1]] = True
    return out


def px(row, col):
    """The box of the 30 m pixel (row, col) of the chunk window."""
    return box(XMIN + 30 * col, YMAX - 30 * (row + 1), XMIN + 30 * (col + 1), YMAX - 30 * row)


def grids(nlcd=71, classf=0):
    return np.full((32, 32), nlcd, dtype=np.uint8), np.full((32, 32), classf, dtype=np.uint8)


def vote(polygon, nlcd, classf):
    return label.vote_polygon(polygon, nlcd, classf, T30)


def write_naip(path, bounds=BOUNDS, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((4, N, N), 40, dtype="uint8")
    profile = {"driver": "GTiff", "width": N, "height": N, "count": 4, "dtype": "uint8",
               "crs": "EPSG:5070", "transform": from_origin(bounds[0], bounds[3], 0.5, 0.5)}
    with rasterio.open(path, "w", **profile) as dst:   # D57: the NAIP file has the gsd and years tags
        dst.write(data)
        dst.update_tags(**({"gsd": "0.6", "years": "2022"} if tags is None else tags))
    return path


def write_layer(path, array, nodata, transform=T30, crs="EPSG:5070"):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "width": array.shape[1], "height": array.shape[0], "count": 1,
               "dtype": "uint8", "crs": crs, "transform": transform, "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


def write_stack(roots, nlcd=None, classf=None, **kw):
    default_nlcd, default_classf = grids()
    nlcd = default_nlcd if nlcd is None else nlcd
    classf = default_classf if classf is None else classf
    write_layer(roots["stack"] / FOLDER / "nlcd_2022.tif", nlcd, 250, **kw)
    write_layer(roots["stack"] / FOLDER / "classf_2022.tif", classf, 0, **kw)


def base_polygons(masks=None):
    masks = [mask((100, 140), (200, 260)), mask((400, 440), (800, 860))] if masks is None else masks
    return segment.masks_to_polygons(masks, T05, BOUNDS, RUN, CHUNK, config=CFG)


def write_base(roots, masks=None):
    return segment.write_polygons(base_polygons(masks), roots, FOLDER, CHUNK, META)


def labelled(gdf, cp=0, pp=0.0, cv=0, pv=0.0):
    out = gdf.copy()
    out["code_prompt"] = np.full(len(out), cp, dtype="int64")
    out["p_prompt"] = np.full(len(out), pp, dtype="float64")
    out["code_vote"] = np.full(len(out), cv, dtype="int64")
    out["p_vote"] = np.full(len(out), pv, dtype="float64")
    return out


class FakeText:
    fake = True
    model_id = "fake-langsam"

    def __init__(self, hits=None):
        self.hits = hits or {}          # prompt -> list of masks
        self.calls = []

    def predict(self, rgb, prompt):
        self.calls.append(prompt)
        return self.hits.get(prompt, [])


class Probe:
    def __init__(self, *values):
        self.values = list(values)
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.values[min(self.count, len(self.values)) - 1]


class Loader:
    def __init__(self, model=None):
        self.calls = 0
        self.model = model

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.model or FakeText({"building": [mask((100, 140), (200, 260))]})


def unions_for(**by_code):
    """All 32 codes False; `by_code` is `c25=array` style."""
    out = {code: ZEROS for code in standard.CODE_LIST}
    for key, value in by_code.items():
        out[int(key[1:])] = value
    return out


def pix(rows=(0, 10), cols=(0, 10)):
    return mask(rows, cols)


def setup_run(tmp_path, nlcd=None, classf=None, masks=None):
    roots = make_roots(tmp_path)
    write_naip(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif")
    write_stack(roots, nlcd, classf)
    write_base(roots, masks)
    return roots


def do_run(roots, model=None, probe=None, run_id=LRUN):
    return label.run_labels([CHUNK], model or FakeText(), roots, FOLDER, CFG, run_id=run_id,
                            probe=probe or Probe(30000, 29000))


def gpkg(roots):
    return roots["semseg"] / FOLDER / "polygons" / f"{CHUNK}.gpkg"


# ------------------------------------------------------------------ D51 prompt table

def test_d51_prompt_table_is_the_exact_table_of_decisions():
    """DS-11, D51: PROMPTS is the exact table, in table order."""
    assert standard.PROMPTS == ALL_PROMPTS
    assert list(standard.PROMPTS) == list(standard.CODE_LIST)
    assert [entry.prompt for entry in standard.CODES] == [ALL_PROMPTS[c] for c in standard.CODE_LIST]
    assert standard.Code._fields[-1] == "prompt"


def test_prompt_table_has_one_unique_nonempty_prompt_for_each_code_except_0():
    """DS-11, FM-87: 32 keys, pattern of D51 item 3, all different, code 0 is not a key."""
    prompts = standard.PROMPTS
    assert len(prompts) == 32 and 0 not in prompts and set(prompts) == set(standard.CODE_LIST)
    pattern = re.compile(r"\A[a-z]+( [a-z]+){0,3}\Z", re.ASCII)
    for code, text in prompts.items():
        assert isinstance(text, str) and pattern.match(text), code
        assert "." not in text
    assert len(set(prompts.values())) == 32


def test_d51_standard_doc_prompt_column_ds29_ds30_and_version():
    """DS-11, DS-12, DS-21, DS-22, DS-29, DS-30: the standard doc (version 6) has the Prompt column and the rules."""
    text = (ROOT / "docs" / "DATA_STANDARD.md").read_text(encoding="utf-8")
    assert standard.DATA_STANDARD_VERSION == 6
    assert text.splitlines()[2].startswith("Version 6")
    rows = re.findall(r"^\|[^|]*\|\s*(\d{2})\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*`([a-z ]+)`\s*\|\s*$",
                      text, re.MULTILINE)
    assert {int(c): p for c, p in rows} == ALL_PROMPTS
    ids = re.findall(r"^\| (DS-\d+) \|", text, re.MULTILINE)
    assert ids == [f"DS-{n}" for n in range(1, 35)]
    assert "label_run_id" in text and "never a level-1 code" in text


def test_label_config_defaults_and_bad_values():
    """DS-30, D51 item 7: defaults, good values, and each bad value raises ValueError."""
    defaults = {"model_id": "sam2-hiera-large", "box_threshold": 0.25, "text_threshold": 0.25,
                "min_overlap": 0.5}
    assert dict(label._label_config({})) == defaults
    assert dict(label._label_config({"label": {}})) == defaults
    assert dict(label._label_config(CFG)) == defaults
    for model_id in ("sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large"):
        assert label._label_config({"label": {"model_id": model_id}})["model_id"] == model_id
    assert label._label_config({"label": {"box_threshold": 1}})["box_threshold"] == 1
    assert label._label_config({"label": {"min_overlap": 0}})["min_overlap"] == 0
    assert label._label_config({"label": {"min_overlap": 1.0}})["min_overlap"] == 1.0
    bad = [("box_threshold", 0), ("box_threshold", 1.01), ("box_threshold", -0.1), ("text_threshold", 0),
           ("text_threshold", 1.5), ("box_threshold", True), ("text_threshold", "0.25"),
           ("box_threshold", float("nan")), ("min_overlap", -0.01), ("min_overlap", 1.01),
           ("min_overlap", False), ("min_overlap", None), ("min_overlap", float("inf")),
           ("model_id", "sam2-hiera-huge"), ("model_id", ""), ("model_id", 3)]
    for key, value in bad:
        with pytest.raises(ValueError):
            label._label_config({"label": {key: value}})
    with pytest.raises(ValueError):
        label._label_config({"label": "x"})


def test_d51_config_yaml_label_section():
    """D51 item 6: config.yaml has the label section and still loads."""
    section = semseg_config.load_config()["label"]
    assert section == {"model_id": "sam2-hiera-large", "box_threshold": 0.25, "text_threshold": 0.25,
                       "min_overlap": 0.5}


def test_d22_import_label_imports_no_model_library():
    """D22: importing semseg.label imports no samgeo, torch, transformers, groundingdino or sam2."""
    code = ("import sys, semseg.label\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('samgeo', 'torch', 'transformers', 'groundingdino', 'lang_sam', 'sam2'))\n"
            "assert not bad, bad\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


# ------------------------------------------------------------------ D52/D56 reverse maps and vote_code

def test_reverse_maps_agree_with_the_tables_of_d52():
    """DS-29: NLCD_VOTE and CLASSF_VOTE are exactly the tables of D52 and D56."""
    expected = {c: c for c in (11, 12, 21, 31, 41, 42, 43, 51, 52, 71, 81, 82, 90, 95)}
    expected.update({22: 20, 23: 20, 24: 20, 72: 0, 73: 0, 74: 0, 250: None})
    assert standard.NLCD_VOTE == expected
    assert standard.CLASSF_VOTE == {5: 33, 6: 36, 8: 96, 9: 95, 12: 81, 14: 55, 15: 54, 16: 53, 17: 51,
                                    26: 52}
    assert standard.VOTE_NONE == 0 and standard.VOTE_NODATA is None
    for raster, code in standard.CLASSF_VOTE.items():          # the inverse of USACE_M_MAP (DS-16)
        assert [c for c, r in standard.USACE_M_MAP.items() if r == raster] == [code]


@pytest.mark.parametrize("nlcd,want", [
    (11, 11), (21, 21), (31, 31), (41, 41), (42, 42), (43, 43), (51, 51), (52, 52), (71, 71),
    (81, 81), (82, 82), (90, 90), (95, 95), (12, 12), (72, 0), (73, 0), (74, 0), (250, None)])
def test_vote_nlcd_values_alone(nlcd, want):
    """DS-29: each NLCD value with no classf_2022 value (0)."""
    assert standard.vote_code(nlcd, 0) == want


def test_vote_nlcd_22_23_24_give_20():
    """DS-29, D56: developed intensity NLCD values give the level-1 code 20."""
    for nlcd in (22, 23, 24):
        assert standard.vote_code(nlcd, 0) == 20
        assert standard.vote_code(nlcd, 2) == 20


@pytest.mark.parametrize("classf,want", [(5, 33), (6, 36), (8, 96), (9, 95), (12, 81), (14, 55), (15, 54),
                                          (16, 53), (17, 51), (26, 52)])
def test_classf_single_code_wins_over_nlcd(classf, want):
    """DS-29: a classf_2022 value with one code wins over every NLCD value, also 250."""
    for nlcd in (41, 21, 12, 22, 250):
        assert standard.vote_code(nlcd, classf) == want


def test_vote_classf_2_gives_20_only_where_nlcd_has_no_code():
    """DS-29, D56: classf 2 (M02) gives 20 only where NLCD gives none or no data."""
    for nlcd in (11, 12, 21, 31, 41, 42, 43, 51, 52, 71, 81, 82, 90, 95):
        assert standard.vote_code(nlcd, 2) == nlcd
    for nlcd in (72, 73, 74, 250):
        assert standard.vote_code(nlcd, 2) == 20


def test_vote_classf_values_without_one_code_use_nlcd():
    """DS-29: the classf values with no single code (and 0) fall back to the NLCD value."""
    for classf in [0, 1, 3, 4, 7, 10, 11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29]:
        assert standard.vote_code(41, classf) == 41, classf
        assert standard.vote_code(72, classf) == 0, classf
        assert standard.vote_code(250, classf) is None, classf


def test_edge_classf_value_not_in_code_table_is_refused_or_counted_as_0():
    """DS-29, FM-88: a classf_2022 value above 29 raises ValueError."""
    for bad in (30, 31, 99, 254, 255):
        with pytest.raises(ValueError):
            standard.vote_code(41, bad)


def test_edge_nlcd_value_not_in_legend_is_refused():
    """DS-29, FM-88: an NLCD value that is not in the table of D52 raises ValueError."""
    for bad in (0, 1, 13, 25, 53, 70, 75, 91, 100, 249, 251, 255, -1):
        with pytest.raises(ValueError):
            standard.vote_code(bad, 0)


def test_edge_vote_code_refuses_bool_and_non_integer():
    """DS-29: a bool or a non-integer value raises ValueError; numpy integers are accepted."""
    for bad in (True, False, 41.5, "41", None):
        with pytest.raises(ValueError):
            standard.vote_code(bad, 0)
        with pytest.raises(ValueError):
            standard.vote_code(41, bad)
    assert standard.vote_code(np.uint8(41), np.uint8(0)) == 41
    assert standard.vote_code(np.int64(250), np.int64(5)) == 33


# ------------------------------------------------------------------ D52 area weights and the winner

def test_vote_weights_each_map_pixel_by_overlap_area():
    """DS-29, FM-90: pixel_weights gives the exact area of polygon and pixel; sum is the polygon area."""
    polygon = box(XMIN + 100, YMAX - 70, XMIN + 130, YMAX - 50)          # 20 + 10 m wide, 10 + 10 m high
    weights = label.pixel_weights(polygon, T30, (32, 32))
    got = {(r, c): a for r, c, a in weights if a > 0}
    assert got == pytest.approx({(1, 3): 200.0, (2, 3): 200.0, (1, 4): 100.0, (2, 4): 100.0})
    assert math.fsum(a for _, _, a in weights) == pytest.approx(polygon.area)
    nlcd, classf = grids()
    nlcd[1:3, 3] = 41
    nlcd[1:3, 4] = 42
    assert vote(polygon, nlcd, classf) == (41, round(400 / 600, 9))       # by area, not by pixel count


def test_vote_polygon_inside_one_pixel_takes_that_pixel():
    """DS-29, FM-90: a 1 m2 polygon takes the code of the one pixel under it, p 1.0."""
    nlcd, classf = grids()
    nlcd[3, 4] = 43
    tiny = box(XMIN + 125, YMAX - 100, XMIN + 126, YMAX - 99)
    assert vote(tiny, nlcd, classf) == (43, 1.0)
    for r, c, a in label.pixel_weights(tiny, T30, (32, 32)):
        assert (r, c) == (3, 4) or a == 0


def test_vote_polygon_on_two_pixels_takes_the_larger_area():
    """DS-29, FM-90: 20 m on one pixel and 10 m on the next: the larger area wins, p = 2/3."""
    nlcd, classf = grids()
    nlcd[3, 4], nlcd[3, 5] = 41, 42
    polygon = box(XMIN + 130, YMAX - 110, XMIN + 160, YMAX - 100)
    assert vote(polygon, nlcd, classf) == (41, 0.666666667)
    nlcd[3, 4], nlcd[3, 5] = 42, 41                                       # swap: the position does not matter
    assert vote(polygon, nlcd, classf) == (42, 0.666666667)


def test_vote_pixel_that_only_touches_does_not_count():
    """DS-29, FM-90: a polygon equal to one pixel box ignores the 8 neighbours."""
    nlcd, classf = grids(nlcd=42)
    nlcd[5, 5] = 41
    assert vote(px(5, 5), nlcd, classf) == (41, 1.0)


def test_vote_tie_goes_to_the_lowest_code_every_run():
    """DS-29, FM-90: equal areas: the lowest code wins, for both pixel orders and on each run."""
    polygon = box(XMIN + 135, YMAX - 110, XMIN + 165, YMAX - 100)         # 15 m + 15 m
    for left, right in ((71, 21), (21, 71)):
        nlcd, classf = grids()
        nlcd[3, 4], nlcd[3, 5] = left, right
        results = {vote(polygon, nlcd, classf) for _ in range(5)}
        assert results == {(21, 0.5)}


def test_vote_20_ties_follow_code_order():
    """DS-29, D56: 20 competes as a code: 20 beats 21 and loses to 11 in a tie."""
    polygon = box(XMIN + 135, YMAX - 110, XMIN + 165, YMAX - 100)
    for other, want in ((21, 20), (11, 11)):
        nlcd, classf = grids()
        nlcd[3, 4], nlcd[3, 5] = 22, other
        assert vote(polygon, nlcd, classf) == (want, 0.5)


def test_vote_none_wins_a_tie_with_a_code():
    """DS-29: none (code 0) is the lowest key: it wins a tie, and the result is (0, 0.0)."""
    polygon = box(XMIN + 135, YMAX - 110, XMIN + 165, YMAX - 100)
    nlcd, classf = grids()
    nlcd[3, 4], nlcd[3, 5] = 72, 41
    assert vote(polygon, nlcd, classf) == (0, 0.0)


def test_vote_none_larger_than_a_code_gives_0_and_a_smaller_none_counts_in_the_total():
    """DS-29, FM-92: none is in the valid area: 10 m none + 20 m code gives p = 2/3, not 1."""
    nlcd, classf = grids()
    nlcd[3, 4], nlcd[3, 5] = 72, 41
    big_none = box(XMIN + 130, YMAX - 110, XMIN + 160, YMAX - 100)       # 20 m on col 4, 10 m on col 5
    assert vote(big_none, nlcd, classf) == (0, 0.0)
    nlcd[3, 4], nlcd[3, 5] = 41, 72
    assert vote(big_none, nlcd, classf) == (41, 0.666666667)


def test_edge_nodata_250_never_wins_the_vote():
    """DS-29, FM-88: 250 is excluded from the valid area: 20 m of 250 and 10 m of code 41 give (41, 1.0)."""
    nlcd, classf = grids()
    nlcd[3, 4], nlcd[3, 5] = 250, 41
    polygon = box(XMIN + 130, YMAX - 110, XMIN + 160, YMAX - 100)
    assert vote(polygon, nlcd, classf) == (41, 1.0)


def test_edge_polygon_under_only_nodata_gets_code_0():
    """DS-29, FM-92: all pixels under the polygon are no data: (0, 0.0), never NaN."""
    nlcd, classf = grids(nlcd=250)
    code, p = vote(px(5, 5), nlcd, classf)
    assert (code, p) == (0, 0.0) and not math.isnan(p)


def test_vote_classf_value_beats_nlcd_in_the_pixel_sum():
    """DS-29: classf 5 on one pixel (33) and NLCD 41 on the other: area decides."""
    nlcd, classf = grids(nlcd=41)
    classf[3, 4] = 5
    polygon = box(XMIN + 130, YMAX - 110, XMIN + 160, YMAX - 100)
    assert vote(polygon, nlcd, classf) == (33, 0.666666667)


def test_vote_bad_value_under_the_polygon_is_refused():
    """DS-29, FM-88: a bad map value under the polygon raises ValueError."""
    nlcd, classf = grids()
    nlcd[5, 5] = 99
    with pytest.raises(ValueError):
        vote(px(5, 5), nlcd, classf)
    nlcd, classf = grids()
    classf[5, 5] = 77
    with pytest.raises(ValueError):
        vote(px(5, 5), nlcd, classf)


def test_p_vote_is_in_0_1():
    """DS-29, FM-92: random maps and rectangles: code in the allowed set, 0 <= p <= 1, no NaN."""
    rng = np.random.default_rng(11)
    nlcd_values = np.array(sorted(standard.NLCD_VOTE), dtype=np.uint8)
    classf_values = np.arange(0, 30, dtype=np.uint8)
    allowed = {0, 20} | set(standard.CODE_LIST)
    for _ in range(40):
        nlcd = rng.choice(nlcd_values, size=(32, 32)).astype(np.uint8)
        classf = rng.choice(classf_values, size=(32, 32)).astype(np.uint8)
        x0, y0 = rng.uniform(0, 800), rng.uniform(60, 900)
        polygon = box(XMIN + x0, YMAX - y0 - rng.uniform(1, 50), XMIN + x0 + rng.uniform(1, 80), YMAX - y0)
        code, p = vote(polygon, nlcd, classf)
        assert code in allowed and 0.0 <= p <= 1.0 and not math.isnan(p)
        assert (code == 0) == (p == 0.0)


def test_edge_empty_or_invalid_geometry_is_refused_with_clear_error():
    """DS-29, FM-85: empty, invalid, and outside-the-chunk polygons raise ValueError."""
    nlcd, classf = grids()
    bowtie = Polygon([(XMIN + 10, YMAX - 10), (XMIN + 20, YMAX - 20), (XMIN + 20, YMAX - 10), (XMIN + 10, YMAX - 20)])
    assert not bowtie.is_valid
    for bad in (Polygon(), bowtie, box(XMIN - 50, YMAX - 10, XMIN + 10, YMAX - 5),
                box(XMAX + 5, YMIN + 5, XMAX + 20, YMIN + 20), None):
        with pytest.raises(ValueError):
            vote(bad, nlcd, classf)
    frame = gpd.GeoDataFrame({"polygon_id": ["x"]}, geometry=[Polygon()], crs="EPSG:5070")
    with pytest.raises(ValueError):
        label.vote_polygons(frame, nlcd, classf, T30)


def test_edge_polygon_crs_not_the_grid_crs_is_refused():
    """DS-29, FM-85: a frame in another CRS, or with no CRS, raises ValueError."""
    nlcd, classf = grids()
    good = gpd.GeoDataFrame({"polygon_id": ["a"]}, geometry=[px(2, 2)], crs="EPSG:5070")
    out = label.vote_polygons(good, nlcd, classf, T30)
    assert list(out.columns) == ["code_vote", "p_vote"] and len(out) == 1
    with pytest.raises(ValueError):
        label.vote_polygons(good.to_crs(4326), nlcd, classf, T30)
    with pytest.raises(ValueError):
        label.vote_polygons(good.set_crs(None, allow_override=True), nlcd, classf, T30)


def test_vote_polygons_frame_shape_order_and_dtypes():
    """DS-29: one row for each polygon, same index and order; int64 and float64."""
    nlcd, classf = grids()
    nlcd[2, 2], nlcd[6, 6] = 41, 82
    frame = gpd.GeoDataFrame({"polygon_id": ["a", "b", "c"]}, geometry=[px(6, 6), px(2, 2), px(9, 9)],
                             crs="EPSG:5070", index=[7, 8, 9])
    out = label.vote_polygons(frame, nlcd, classf, T30)
    assert out.index.tolist() == [7, 8, 9]
    assert out["code_vote"].tolist() == [82, 41, 71] and out["p_vote"].tolist() == [1.0, 1.0, 1.0]
    assert out["code_vote"].dtype == np.int64 and out["p_vote"].dtype == np.float64
    empty = label.vote_polygons(frame.iloc[0:0], nlcd, classf, T30)
    assert len(empty) == 0 and list(empty.columns) == ["code_vote", "p_vote"]


# ------------------------------------------------------------------ D52 layers

def test_label_layers_names_are_the_two_constants():
    """DS-29, FM-88: VOTE_LAYERS is exactly nlcd_2022 and classf_2022."""
    assert tuple(label.VOTE_LAYERS) == ("nlcd_2022", "classf_2022")


def test_vote_reads_only_nlcd_2022_and_classf_2022(tmp_path, monkeypatch):
    """DS-29, FM-88: read_vote_layers opens only the two layers of the stack folder (decoy files are not opened)."""
    roots = make_roots(tmp_path)
    nlcd, classf = grids(nlcd=41, classf=5)
    write_stack(roots, nlcd, classf)
    for decoy in ("nlcd_2021.tif", "landcover.tif", "nlcd_2022_extra.tif", "classf_2021.tif"):
        write_layer(roots["stack"] / FOLDER / decoy, grids(nlcd=99)[0], 250)
    seen = []
    real_open = rasterio.open

    def spy(path, *args, **kwargs):
        seen.append(Path(str(path)))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", spy)
    got_nlcd, got_classf, transform = label.read_vote_layers(roots, FOLDER, CHUNK)
    assert seen and {p.name for p in seen} == {"nlcd_2022.tif", "classf_2022.tif"}
    assert {p.parent for p in seen} == {roots["stack"] / FOLDER}
    assert got_nlcd.shape == (32, 32) and got_classf.shape == (32, 32)
    assert got_nlcd.dtype == np.uint8 and got_classf.dtype == np.uint8
    assert (got_nlcd == 41).all() and (got_classf == 5).all() and transform == T30


def test_read_vote_layers_window_of_a_larger_raster_and_nodata_outside(tmp_path):
    """DS-29: the 32 x 32 window of the chunk box; a pixel outside the raster gets 250 and 0."""
    roots = make_roots(tmp_path)
    big_n = np.arange(40 * 40, dtype=np.int64).reshape(40, 40) % 40 + 11
    big = from_origin(XMIN - 60, YMAX + 90, 30, 30)                    # the chunk window starts at (3, 2)
    write_stack(roots, big_n.astype(np.uint8), np.full((40, 40), 5, dtype=np.uint8), transform=big)
    nlcd, classf, transform = label.read_vote_layers(roots, FOLDER, CHUNK)
    assert (nlcd == big_n[3:35, 2:34]).all() and (classf == 5).all() and transform == T30
    small = make_roots(tmp_path / "small")
    write_stack(small, np.full((16, 16), 41, dtype=np.uint8), np.full((16, 16), 5, dtype=np.uint8))
    nlcd, classf, _ = label.read_vote_layers(small, FOLDER, CHUNK)
    assert (nlcd[:16, :16] == 41).all() and (nlcd[16:, :] == 250).all() and (nlcd[:, 16:] == 250).all()
    assert (classf[:16, :16] == 5).all() and (classf[16:, :] == 0).all() and (classf[:, 16:] == 0).all()


def test_edge_layers_off_lattice_are_refused(tmp_path):
    """DS-29, FM-85: wrong CRS, wrong pixel size, off lattice, or two different transforms raise ValueError."""
    nlcd, classf = grids()
    cases = {
        "crs": dict(crs="EPSG:4326"),
        "pixel": dict(transform=from_origin(XMIN, YMAX, 10, 10)),
        "shifted": dict(transform=from_origin(XMIN + 15, YMAX, 30, 30)),
        "south_up": dict(transform=from_origin(XMIN, YMAX, 30, -30)),
    }
    for name, kw in cases.items():
        roots = make_roots(tmp_path / name)
        write_stack(roots, nlcd, classf, **kw)
        with pytest.raises(ValueError):
            label.read_vote_layers(roots, FOLDER, CHUNK)
    roots = make_roots(tmp_path / "mixed")
    write_layer(roots["stack"] / FOLDER / "nlcd_2022.tif", nlcd, 250)
    write_layer(roots["stack"] / FOLDER / "classf_2022.tif", classf, 0, transform=from_origin(XMIN - 30, YMAX, 30, 30))
    with pytest.raises(ValueError):
        label.read_vote_layers(roots, FOLDER, CHUNK)


def test_read_vote_layers_missing_layer_is_an_error(tmp_path):
    """DS-29: a missing layer file raises FileNotFoundError."""
    roots = make_roots(tmp_path)
    write_layer(roots["stack"] / FOLDER / "nlcd_2022.tif", grids()[0], 250)
    with pytest.raises(FileNotFoundError):
        label.read_vote_layers(roots, FOLDER, CHUNK)


# ------------------------------------------------------------------ D53 prompt label

def test_prompt_calls_each_code_once_in_table_order():
    """DS-30, D51: 27 calls, one prompt each, in table order, never combined."""
    model = FakeText()
    unions = label.prompt_unions(model, np.zeros((N, N, 3), dtype=np.uint8), CFG)
    assert model.calls == [standard.PROMPTS[c] for c in standard.CODE_LIST]
    assert list(unions) == list(standard.CODE_LIST)
    assert all(not v.any() and v.shape == (N, N) and v.dtype == np.bool_ for v in unions.values())
    assert not any("." in call for call in model.calls)


def test_prompt_union_is_or_of_all_masks_of_a_code():
    """DS-30: the union of one code is the OR of its masks (none: all False)."""
    model = FakeText({"building": [mask((0, 10), (0, 10)), mask((5, 20), (5, 20))], "dam": []})
    unions = label.prompt_unions(model, np.zeros((N, N, 3), dtype=np.uint8), CFG)
    expect = mask((0, 10), (0, 10)) | mask((5, 20), (5, 20))
    assert (unions[25] == expect).all() and not unions[29].any() and not unions[26].any()


@pytest.mark.parametrize("bad", [np.zeros((1919, 1919), dtype=bool), np.zeros((N, N), dtype=np.uint8),
                                 np.zeros((3, N, N), dtype=bool), np.zeros((N,), dtype=bool)])
def test_edge_text_mask_of_wrong_shape_is_refused(bad):
    """DS-30, FM-86: a text mask that is not a 1920 x 1920 bool array raises ValueError."""
    with pytest.raises(ValueError):
        label.prompt_unions(FakeText({"building": [bad]}), np.zeros((N, N, 3), dtype=np.uint8), CFG)


def test_polygon_pixels_uses_the_pixel_centre_rule():
    """DS-30: a 0.5 m pixel belongs to the polygon when its centre is inside."""
    flat = label.polygon_pixels(box(XMIN + 10, YMAX - 20, XMIN + 20, YMAX - 10), T05)
    assert flat.shape == (N, N) and flat.dtype == np.bool_ and flat.sum() == 400
    assert flat[20:40, 20:40].all()
    shifted = label.polygon_pixels(box(XMIN + 10.3, YMAX - 20, XMIN + 20.3, YMAX - 10), T05)
    assert shifted.sum() == 400 and shifted[20:40, 21:41].all() and not shifted[:, 20].any()


def test_edge_polygon_with_no_pixel_centre_gets_code_0():
    """DS-30: a 0.2 m square that holds no pixel centre has 0 pixels: (0, 0.0)."""
    tiny = box(XMIN + 10.05, YMAX - 10.25, XMIN + 10.2, YMAX - 10.05)
    pixels = label.polygon_pixels(tiny, T05)
    assert pixels.sum() == 0
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, N), (0, N))), 0.5) == (0, 0.0)


def test_edge_polygon_pixels_refuses_a_transform_that_is_not_the_chunk_grid():
    """DS-30, D53 item 2: a 30 m transform or a half-pixel shift raises ValueError."""
    polygon = box(XMIN + 10, YMAX - 20, XMIN + 20, YMAX - 10)
    for bad in (T30, from_origin(XMIN + 0.25, YMAX, 0.5, 0.5), from_origin(XMIN, YMAX, 1.0, 1.0)):
        with pytest.raises(ValueError):
            label.polygon_pixels(polygon, bad)


def test_prompt_overlap_two_codes_takes_the_larger_p_and_ties_take_the_lowest_code():
    """DS-30, FM-91: the larger m wins; a tie goes to the lowest code, not to dict order."""
    pixels = pix()
    result = label.prompt_polygon(pixels, unions_for(c25=mask((0, 10), (0, 7)), c26=mask((0, 10), (0, 4))), 0.5)
    assert result == (25, 0.7)
    result = label.prompt_polygon(pixels, unions_for(c25=mask((0, 10), (0, 4)), c26=mask((0, 10), (0, 7))), 0.5)
    assert result == (26, 0.7)
    left, right = mask((0, 10), (0, 5)), mask((0, 10), (5, 10))
    assert label.prompt_polygon(pixels, unions_for(c28=left, c41=right), 0.5) == (28, 0.5)
    reverse = {c: ZEROS for c in reversed(standard.CODE_LIST)}      # the dict order is not the code order
    reverse[28], reverse[41] = right, left
    assert label.prompt_polygon(pixels, reverse, 0.5) == (28, 0.5)


def test_p_prompt_is_in_0_1_for_each_polygon():
    """DS-30, FM-91: a mask larger than the polygon does not push p above 1; random masks stay in 0..1."""
    pixels = pix((10, 20), (10, 20))
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, N), (0, N))), 0.0) == (25, 1.0)
    rng = np.random.default_rng(5)
    for _ in range(25):
        union = rng.random((N, N)) > rng.random()
        code, p = label.prompt_polygon(pix((0, 50), (0, 50)), unions_for(c41=union, c42=~union), 0.0)
        assert code in (0, 41, 42) and 0.0 <= p <= 1.0 and not math.isnan(p)


def test_polygon_with_no_overlap_gets_code_0_and_p_0():
    """DS-30, FM-91: no hit gives (0, 0.0); n = 0 gives (0, 0.0)."""
    assert label.prompt_polygon(pix(), unions_for(c25=mask((100, 120), (100, 120))), 0.0) == (0, 0.0)
    assert label.prompt_polygon(ZEROS, unions_for(c25=mask((0, N), (0, N))), 0.0) == (0, 0.0)
    assert label.prompt_polygon(pix(), unions_for(), 0.5) == (0, 0.0)


def test_prompt_below_min_overlap_gets_code_0_and_p_0():
    """DS-30: p < min_overlap gives (0, 0.0); p == min_overlap gives the code; min 0 and min 1 limits."""
    pixels = pix()                                                  # n = 100
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 7), (0, 7))), 0.5) == (0, 0.0)   # 49
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 10), (0, 5))), 0.5) == (25, 0.5)  # 50
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 1), (0, 1))), 0.0) == (25, 0.01)
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 10), (0, 9))), 1.0) == (0, 0.0)
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 10), (0, 10))), 1.0) == (25, 1.0)


def test_prompt_p_is_rounded_to_9_decimals():
    """DS-30: p = round(m / n, 9)."""
    pixels = mask((0, 1), (0, 3))                                   # n = 3
    assert label.prompt_polygon(pixels, unions_for(c25=mask((0, 1), (0, 1))), 0.0) == (25, 0.333333333)


def test_prompt_polygons_frame_for_all_polygons():
    """DS-30: prompt_polygons returns one row for each polygon with the polygon index."""
    frame = gpd.GeoDataFrame({"polygon_id": ["a", "b", "c"]}, crs="EPSG:5070", index=[4, 5, 6], geometry=[
        box(XMIN + 10, YMAX - 20, XMIN + 20, YMAX - 10),            # pixels 20..39
        box(XMIN + 100, YMAX - 20, XMIN + 110, YMAX - 10),
        box(XMIN + 50, YMAX - 20, XMIN + 50.2, YMAX - 19.8)])       # no pixel centre
    unions = unions_for(c25=mask((20, 40), (20, 40)), c26=mask((20, 40), (200, 205)))
    out = label.prompt_polygons(frame, unions, T05, 0.5)
    assert out.index.tolist() == [4, 5, 6] and list(out.columns) == ["code_prompt", "p_prompt"]
    assert out["code_prompt"].tolist() == [25, 0, 0] and out["p_prompt"].tolist() == [1.0, 0.0, 0.0]
    assert out["code_prompt"].dtype == np.int64 and out["p_prompt"].dtype == np.float64


# ------------------------------------------------------------------ D54 codes and writer

def test_level1_code_is_refused_as_a_machine_label():
    """DS-12, D54: a machine label is 0 or one of the 32 codes; a level-1 code or another number raises."""
    for code in (0, *standard.CODE_LIST):
        standard.check_machine_code(code)
    for bad in (10, 20, 30, 40, 50, 70, 80, 99, 15, 22, 1, 100, -1):
        with pytest.raises(ValueError):
            standard.check_machine_code(bad)


def test_check_vote_code_allows_20_only():
    """DS-12, D56: code_vote is 0, a code, or 20; the other level-1 codes raise ValueError."""
    for code in (0, 20, *standard.CODE_LIST):
        standard.check_vote_code(code)
    for bad in (10, 30, 40, 50, 70, 80, 99, 15, 22, 100, -1):
        with pytest.raises(ValueError):
            standard.check_vote_code(bad)


def test_prompt_code_20_is_refused():
    """DS-12, D56: code_prompt 20 is refused by the machine check and by the writer."""
    with pytest.raises(ValueError):
        standard.check_machine_code(20)


def test_labelled_file_keeps_the_1_2_columns_rows_and_metadata(tmp_path):
    """DS-21, FM-74, D54: the 4 columns are added; the old columns, rows, geometry and 6 keys stay."""
    roots = make_roots(tmp_path)
    base = base_polygons()
    old = write_base(roots)
    old_meta = pyogrio.read_info(old, layer="polygons")["layer_metadata"]
    new = labelled(base, cp=25, pp=0.75, cv=41, pv=0.5)
    path = label.write_labelled_polygons(new, roots, FOLDER, CHUNK, META, LMETA)
    assert path == old == gpkg(roots)
    assert pyogrio.list_layers(path)[:, 0].tolist() == ["polygons"]
    back = gpd.read_file(path, layer="polygons")
    assert [c for c in back.columns if c != "geometry"] == ["polygon_id", "area_m2", *FOUR]
    assert back["polygon_id"].tolist() == base["polygon_id"].tolist()
    assert back.crs.to_epsg() == 5070
    assert all(a.equals_exact(b, 1e-9) for a, b in zip(back.geometry, base.geometry))
    assert back["code_prompt"].dtype == np.int64 and back["code_vote"].dtype == np.int64
    assert back["p_prompt"].dtype == np.float64 and back["p_vote"].dtype == np.float64
    meta = pyogrio.read_info(path, layer="polygons")["layer_metadata"]
    for key in standard.METADATA_KEYS:
        assert meta[key] == old_meta[key] == META[key]


def test_labelled_file_has_label_metadata_keys(tmp_path):
    """DS-21, D54: label_run_id, label_code_version, label_created and label_sources are in the layer."""
    roots = make_roots(tmp_path)
    write_base(roots)
    path = label.write_labelled_polygons(labelled(base_polygons()), roots, FOLDER, CHUNK, META, LMETA)
    meta = pyogrio.read_info(path, layer="polygons")["layer_metadata"]
    for key, value in LMETA.items():
        assert meta[key] == value
    second = {**LMETA, "label_run_id": "20261113-1000-langsam", "label_created": "2026-11-13T10:00:00Z"}
    label.write_labelled_polygons(labelled(base_polygons(), cp=26, pp=1.0), roots, FOLDER, CHUNK, META, second)
    meta = pyogrio.read_info(path, layer="polygons")["layer_metadata"]
    assert meta["label_run_id"] == "20261113-1000-langsam" and meta["run_id"] == RUN
    back = gpd.read_file(path, layer="polygons")
    assert set(back["code_prompt"]) == {26} and len(back.columns) == 7


def test_labelled_file_rewrite_from_a_labelled_file_replaces_the_columns(tmp_path):
    """DS-21, D54: a new label run replaces the 4 columns; it does not add a second set."""
    roots = make_roots(tmp_path)
    write_base(roots)
    label.write_labelled_polygons(labelled(base_polygons(), cp=25, pp=1.0), roots, FOLDER, CHUNK, META, LMETA)
    again = gpd.read_file(gpkg(roots), layer="polygons")
    again["code_prompt"] = 41
    label.write_labelled_polygons(again, roots, FOLDER, CHUNK, META, LMETA)
    back = gpd.read_file(gpkg(roots), layer="polygons")
    assert [c for c in back.columns if c != "geometry"] == ["polygon_id", "area_m2", *FOUR]
    assert set(back["code_prompt"]) == {41}


@pytest.mark.parametrize("column,value", [
    ("code_prompt", 20), ("code_prompt", 10), ("code_prompt", 99), ("code_prompt", 15), ("code_vote", 30),
    ("code_vote", 99), ("code_vote", 15), ("p_prompt", 1.5), ("p_prompt", -0.1), ("p_vote", 1.01),
    ("p_vote", float("nan")), ("p_prompt", float("nan"))])
def test_edge_labelled_writer_refuses_bad_values_and_keeps_old_file(tmp_path, column, value):
    """DS-12, FM-92, D54: a bad code or p raises ValueError before the first write; the old file stays whole."""
    roots = make_roots(tmp_path)
    old = write_base(roots)
    before = old.read_bytes()
    frame = labelled(base_polygons(), cp=25, pp=1.0, cv=41, pv=1.0)
    frame[column] = frame[column].astype("float64") if isinstance(value, float) else value
    frame.loc[frame.index[0], column] = value
    with pytest.raises(ValueError):
        label.write_labelled_polygons(frame, roots, FOLDER, CHUNK, META, LMETA)
    assert old.read_bytes() == before and len(list(old.parent.glob("*"))) == 1


def test_edge_labelled_writer_accepts_code_vote_20_and_refuses_code_0_with_p(tmp_path):
    """DS-12, D54, D56: code_vote 20 is valid; a code 0 with p above 0 raises ValueError."""
    roots = make_roots(tmp_path)
    write_base(roots)
    label.write_labelled_polygons(labelled(base_polygons(), cv=20, pv=0.5), roots, FOLDER, CHUNK, META, LMETA)
    assert set(gpd.read_file(gpkg(roots), layer="polygons")["code_vote"]) == {20}
    with pytest.raises(ValueError):
        label.write_labelled_polygons(labelled(base_polygons(), cp=0, pp=0.5), roots, FOLDER, CHUNK, META, LMETA)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(labelled(base_polygons(), cv=0, pv=0.5), roots, FOLDER, CHUNK, META, LMETA)


@pytest.mark.parametrize("drop", ["label_run_id", "label_code_version", "label_created", "label_sources"])
def test_edge_labelled_writer_refuses_missing_label_metadata(tmp_path, drop):
    """DS-21, D54: a missing or empty label_* key raises ValueError and writes nothing."""
    roots = make_roots(tmp_path)
    meta = {k: v for k, v in LMETA.items() if k != drop}
    with pytest.raises(ValueError):
        label.write_labelled_polygons(labelled(base_polygons()), roots, FOLDER, CHUNK, META, meta)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(labelled(base_polygons()), roots, FOLDER, CHUNK, META, {**LMETA, drop: " "})
    assert not (roots["semseg"] / FOLDER).exists()


def test_edge_labelled_writer_refuses_bad_run_id_crs_ds21_and_missing_columns(tmp_path):
    """DS-10, DS-21, D54: a bad label_run_id, CRS, DS-21 key or missing column raises ValueError."""
    roots = make_roots(tmp_path)
    frame = labelled(base_polygons(), cp=25, pp=1.0)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(frame, roots, FOLDER, CHUNK, META, {**LMETA, "label_run_id": "bad run"})
    with pytest.raises(ValueError):
        label.write_labelled_polygons(frame.to_crs(4326), roots, FOLDER, CHUNK, META, LMETA)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(frame, roots, FOLDER, CHUNK, {k: v for k, v in META.items() if k != "created"}, LMETA)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(frame.drop(columns=["p_vote"]), roots, FOLDER, CHUNK, META, LMETA)
    with pytest.raises(ValueError):
        label.write_labelled_polygons(frame, roots, FOLDER, "r01_c1", META, LMETA)
    assert not (roots["semseg"] / FOLDER).exists()


def test_label_write_goes_through_safe_write(tmp_path, monkeypatch):
    """FM-74, D54: the file is written with storage.safe_write."""
    roots = make_roots(tmp_path)
    write_base(roots)
    seen = []
    orig = storage.safe_write

    def spy(target, data, roots_, **kw):
        seen.append((Path(target), type(data)))
        return orig(target, data, roots_, **kw)

    monkeypatch.setattr(storage, "safe_write", spy)
    label.write_labelled_polygons(labelled(base_polygons()), roots, FOLDER, CHUNK, META, LMETA)
    assert seen == [(gpkg(roots), bytes)]


def test_label_write_failure_keeps_old_file_whole(tmp_path, monkeypatch):
    """FM-74, D54: a failed copy leaves the old file whole and no part file."""
    roots = make_roots(tmp_path)
    old = write_base(roots)
    before = old.read_bytes()
    orig = storage.safe_write

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(storage, "safe_write", lambda t, d, r, **kw: orig(t, d, r, copy=boom))
    with pytest.raises(OSError):
        label.write_labelled_polygons(labelled(base_polygons(), cp=25, pp=1.0), roots, FOLDER, CHUNK, META, LMETA)
    assert old.read_bytes() == before and len(list(old.parent.glob("*"))) == 1


# ------------------------------------------------------------------ D54/D56 code counts (FM-2)

def test_count_codes_has_34_rows_in_code_order_with_zeros():
    """DS-22, FM-2, D56: code 0, the 32 codes and code 20, in code order, also with 0 polygons."""
    rows = label.count_codes([])
    codes = [r["code"] for r in rows]
    assert codes == [0, 11, 12, 13, 14, 20, 21, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 41, 42, 43,
                     44, 51, 52, 53, 54, 55, 71, 81, 82, 83, 90, 95, 96] and len(rows) == 34
    for row in rows:
        assert set(row) == {"code", "name", "n_prompt", "n_vote", "area_prompt_m2", "area_vote_m2"}
        assert row["n_prompt"] == 0 and row["n_vote"] == 0
        assert row["area_prompt_m2"] == 0 and row["area_vote_m2"] == 0
    names = {r["code"]: r["name"] for r in rows}
    assert names[0] == "No label" and names[20] == "Developed" and names[25] == "Building"
    assert names[21].startswith("Developed, open space")


def test_count_codes_counts_polygons_and_sums_areas():
    """DS-22, D54: n and area for each code; areas are fsum rounded to 2 decimals; 20 has n_prompt 0."""
    frame1 = pd.DataFrame({"code_prompt": [25, 25, 0], "code_vote": [20, 41, 41], "area_m2": [10.005, 20.001, 5.5]})
    frame2 = pd.DataFrame({"code_prompt": [26], "code_vote": [0], "area_m2": [0.1]})
    rows = {r["code"]: r for r in label.count_codes([frame1, frame2])}
    assert (rows[25]["n_prompt"], rows[25]["area_prompt_m2"]) == (2, round(10.005 + 20.001, 2))
    assert (rows[26]["n_prompt"], rows[26]["area_prompt_m2"]) == (1, 0.1)
    assert (rows[0]["n_prompt"], rows[0]["n_vote"]) == (1, 1) and rows[0]["area_prompt_m2"] == 5.5
    assert (rows[41]["n_vote"], rows[41]["area_vote_m2"]) == (2, round(20.001 + 5.5, 2))
    assert (rows[20]["n_vote"], rows[20]["n_prompt"], rows[20]["area_prompt_m2"]) == (1, 0, 0)
    assert sum(r["n_vote"] for r in rows.values()) == 4 and sum(r["n_prompt"] for r in rows.values()) == 4


# ------------------------------------------------------------------ run, log and R3

def read_log(roots, run_id=LRUN):
    return json.loads((roots["semseg"] / "runs" / run_id / "log.json").read_text())


def test_every_code_vote_and_code_prompt_is_in_the_code_table(tmp_path):
    """R3, DS-11, D54: after a fake run every code_prompt and code_vote is in the code table."""
    nlcd, classf = grids()
    nlcd[1:3, 3], nlcd[1:3, 4] = 41, 42
    roots = setup_run(tmp_path, nlcd, classf)
    model = FakeText({"building": [mask((100, 140), (200, 260))]})
    paths = do_run(roots, model)
    assert paths == [gpkg(roots)]
    back = gpd.read_file(gpkg(roots), layer="polygons")
    assert len(back) == 2
    assert set(back["code_prompt"]) <= {0, *standard.CODE_LIST}
    assert set(back["code_vote"]) <= {0, 20, *standard.CODE_LIST}
    first, second = back.iloc[0], back.iloc[1]
    assert (first["code_prompt"], first["p_prompt"]) == (25, 1.0)
    assert (first["code_vote"], first["p_vote"]) == (41, 0.666666667)
    assert (second["code_prompt"], second["p_prompt"]) == (0, 0.0)
    assert (second["code_vote"], second["p_vote"]) == (71, 1.0)
    assert model.calls == [standard.PROMPTS[c] for c in standard.CODE_LIST]


def test_run_keeps_polygons_and_writes_label_metadata_and_log_keys(tmp_path):
    """DS-21, DS-22, D54: the run keeps the 1.2 data, adds label_* keys and writes the log with codes."""
    roots = setup_run(tmp_path)
    base = base_polygons()
    do_run(roots)
    back = gpd.read_file(gpkg(roots), layer="polygons")
    assert back["polygon_id"].tolist() == base["polygon_id"].tolist()
    meta = pyogrio.read_info(gpkg(roots), layer="polygons")["layer_metadata"]
    for key in standard.METADATA_KEYS:
        assert meta[key] == META[key]
    assert meta["label_run_id"] == LRUN and meta["label_code_version"].strip()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["label_created"])
    assert all(name in meta["label_sources"] for name in ("nlcd_2022.tif", "classf_2022.tif", CHUNK))
    log = read_log(roots)
    for key in ("run_id", "model_id", "fake", "memory_before_mb", "memory_after_mb", "started", "finished",
                "chunks", "chunks_done", "error", "log_errors", "gpu_before", "gpu_after", "codes"):
        assert key in log, key
    assert log["run_id"] == LRUN and log["fake"] is True and log["chunks_done"] == [CHUNK] and log["error"] is None
    assert log["memory_before_mb"] == 30000


def test_report_lists_each_code_with_0_polygons(tmp_path):
    """FM-2, D54, D56: the log `codes` list has 34 rows; a code with 0 polygons is a row with n = 0."""
    roots = setup_run(tmp_path)
    do_run(roots, FakeText({"building": [mask((100, 140), (200, 260))]}))
    rows = read_log(roots)["codes"]
    assert len(rows) == 34 and [r["code"] for r in rows][:6] == [0, 11, 12, 13, 14, 20]
    by = {r["code"]: r for r in rows}
    assert by[25]["n_prompt"] == 1 and by[25]["area_prompt_m2"] == pytest.approx(600.0, abs=0.01)
    assert by[41]["n_prompt"] == 0 and by[52]["n_prompt"] == 0 and by[52]["n_vote"] == 0
    assert by[0]["n_prompt"] == 1                                    # the second polygon has no hit
    assert by[71]["n_vote"] == 2 and sum(r["n_vote"] for r in rows) == 2


def test_label_run_refuses_existing_run_folder(tmp_path):
    """D49, D54: the label run stops when runs/<label_run_id>/ exists; it changes nothing."""
    roots = setup_run(tmp_path)
    before = gpkg(roots).read_bytes()
    (roots["semseg"] / "runs" / LRUN).mkdir(parents=True)
    probe = Probe(30000)
    with pytest.raises(FileExistsError):
        do_run(roots, probe=probe)
    assert gpkg(roots).read_bytes() == before and not list((roots["semseg"] / "runs" / LRUN).iterdir())


def test_label_run_missing_polygon_file_is_an_error_and_is_logged(tmp_path):
    """D54 item 6: a chunk with no polygon file raises FileNotFoundError; the log records the error."""
    roots = setup_run(tmp_path)
    gpkg(roots).unlink()
    with pytest.raises(FileNotFoundError):
        do_run(roots)
    log = read_log(roots)
    assert log["chunks_done"] == [] and log["error"] and "FileNotFoundError" in log["error"]


def test_label_run_writes_only_under_semseg(tmp_path):
    """DS-23: the run writes nothing to fgt or stack."""
    roots = setup_run(tmp_path)
    before = {k: sorted(p.relative_to(roots[k]) for p in Path(roots[k]).rglob("*")) for k in ("fgt", "stack")}
    do_run(roots)
    for k in ("fgt", "stack"):
        assert sorted(p.relative_to(roots[k]) for p in Path(roots[k]).rglob("*")) == before[k]


def test_label_run_reads_only_the_two_vote_layers_of_the_stack(tmp_path, monkeypatch):
    """FM-88: a whole run opens in the stack only nlcd_2022.tif and classf_2022.tif."""
    roots = setup_run(tmp_path)
    write_layer(roots["stack"] / FOLDER / "nlcd_2021.tif", grids()[0], 250)
    seen = []
    real_open = rasterio.open

    def spy(path, *args, **kwargs):
        seen.append(Path(str(path)))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", spy)
    do_run(roots)
    in_stack = {p.name for p in seen if roots["stack"] in p.parents}
    assert in_stack == {"nlcd_2022.tif", "classf_2022.tif"}


def test_label_run_wrong_naip_gsd_is_refused_before_a_write(tmp_path):
    """D57: a NAIP chunk with gsd 0.5 stops the label run; the polygon file is not changed."""
    roots = setup_run(tmp_path)
    write_naip(roots["fgt"] / FOLDER / "naip" / f"{CHUNK}.tif", tags={"gsd": "0.5", "years": "2022"})
    before = gpkg(roots).read_bytes()
    with pytest.raises(ValueError):
        do_run(roots)
    assert gpkg(roots).read_bytes() == before


# ------------------------------------------------------------------ D22/D47/D50 CLI

def demo_setup(tmp_path):
    roots = setup_run(tmp_path)
    config = {**CFG, "demo": {"pid": 77, "chunks": [CHUNK]}}
    return roots, config


def test_d22_cli_demo_without_run_refuses_and_loads_nothing(tmp_path):
    """D22: no --run: exit not 0, no probe, no load, no run folder."""
    roots, config = demo_setup(tmp_path)
    loader, probe = Loader(), Probe(99999)
    before = gpkg(roots).read_bytes()
    with pytest.raises(SystemExit) as info:
        label.main(["--demo"], roots=roots, config=config, probe=probe, loader=loader, now=NOW)
    assert info.value.code not in (0, None) and loader.calls == 0 and probe.count == 0
    assert not (roots["semseg"] / "runs").exists() and gpkg(roots).read_bytes() == before


def test_d22_cli_no_args_refuses(tmp_path):
    roots, config = demo_setup(tmp_path)
    with pytest.raises(SystemExit) as info:
        label.main([], roots=roots, config=config, probe=Probe(99999), loader=Loader(), now=NOW)
    assert info.value.code not in (0, None)


def test_d50_cli_abbreviated_flag_is_refused_nothing_loaded(tmp_path):
    """D50: `--ru` is not `--run` (allow_abbrev=False): exit 2, no probe, no load."""
    roots, config = demo_setup(tmp_path)
    loader, probe = Loader(), Probe(99999)
    with pytest.raises(SystemExit) as info:
        label.main(["--demo", "--ru"], roots=roots, config=config, probe=probe, loader=loader, now=NOW)
    assert info.value.code == 2 and loader.calls == 0 and probe.count == 0


def test_d22_cli_module_without_run_exits_nonzero_and_names_the_flag():
    done = subprocess.run([sys.executable, "-m", "semseg.label", "--demo"], cwd=ROOT, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode != 0 and "--run" in done.stderr and "Traceback" not in done.stderr


def test_d47_cli_low_memory_exit_4_before_any_langsam_load(tmp_path):
    """D47, D50: available memory below segment.min_free_mb: exit 4, the loader is never called."""
    roots, config = demo_setup(tmp_path)
    loader = Loader()
    before = gpkg(roots).read_bytes()
    with pytest.raises(SystemExit) as info:
        label.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(15999), loader=loader, now=NOW)
    assert info.value.code == 4 and loader.calls == 0
    assert gpkg(roots).read_bytes() == before


def test_d47_cli_memory_at_the_limit_loads(tmp_path):
    """D47: available memory equal to min_free_mb is enough: the loader runs once."""
    roots, config = demo_setup(tmp_path)
    loader = Loader()
    assert label.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(16000), loader=loader,
                      now=NOW) == 0
    assert loader.calls == 1


def test_d22_cli_demo_run_with_fake_loader_labels_the_polygons(tmp_path):
    """D22, D54: --demo --run with a fake loader labels the demo chunk and writes the log."""
    roots, config = demo_setup(tmp_path)
    loader = Loader()
    assert label.main(["--demo", "--run"], roots=roots, config=config, probe=Probe(30000, 29000),
                      loader=loader, now=NOW) == 0
    assert loader.calls == 1
    back = gpd.read_file(gpkg(roots), layer="polygons")
    assert set(FOUR) <= set(back.columns) and len(back) == 2
    logs = list((roots["semseg"] / "runs").glob("*/log.json"))
    assert len(logs) == 1
    log = json.loads(logs[0].read_text())
    standard.check_run_id(log["run_id"])
    assert log["run_id"].endswith("-langsam") and log["fake"] is True and len(log["codes"]) == 34
    assert log["chunks_done"] == [CHUNK]
