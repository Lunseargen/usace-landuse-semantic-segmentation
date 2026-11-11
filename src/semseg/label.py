"""Polygon labels: LangSAM text prompts and the map vote (task 1.3, D51 to D56, D57, FM-2, FM-85 to FM-92).

Run (only after Jack says "run", D22):
`uv run python -m semseg.label --demo --run`

For each chunk the run reads `polygons/<chunk>.gpkg` (task 1.2), the NAIP chunk and the 2 stack
layers `nlcd_2022` and `classf_2022`. It adds 4 columns: `code_prompt`, `p_prompt` (LangSAM,
DS-30) and `code_vote`, `p_vote` (map vote, DS-29). The file is rewritten with `storage.safe_write`.
The run log `<semseg>/runs/<run_id>/log.json` has the key `codes`: the polygon count of each code,
also with 0 polygons (FM-2).

D22: importing this module imports no samgeo, torch, transformers, groundingdino or sam2 module.
The real model loads only inside `_build_real_model`, and only when the user runs the command above.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from semseg import grid, segment, standard, storage
from semseg.config import label_settings, load_config

METHOD = "langsam"
VOTE_LAYERS = ("nlcd_2022", "classf_2022")  # FM-88: the only stack layers that the vote reads
LABEL_KEYS = ("label_run_id", "label_code_version", "label_created", "label_sources")
LABEL_COLUMNS = ("code_prompt", "p_prompt", "code_vote", "p_vote")
_NLCD_FILL, _CLASSF_FILL = 250, 0  # the no-data value of each layer, used outside the raster extent
_WINDOW = standard.CELLS_PER_CHUNK


class TextMaskModel(Protocol):
    """A text-prompt model. `predict` gets an RGB uint8 array (H, W, 3) and a prompt; it returns bool masks (H, W)."""

    model_id: str
    fake: bool

    def predict(self, rgb, prompt):  # pragma: no cover (protocol)
        ...


def _label_config(config):
    """Return the `label` settings of a config as a dict of 4 keys (D51 item 7). Raise ValueError if not valid."""
    section = config.get("label") if isinstance(config, dict) else None
    return label_settings(section)


# ---------------------------------------------------------------- D52 vote layers

def read_vote_layers(roots, folder, chunk):
    """Return `(nlcd uint8 (32, 32), classf uint8 (32, 32), transform)` for the chunk window (DS-29, D52 item 4).

    Reads only `<stack>/<folder>/nlcd_2022.tif` and `<stack>/<folder>/classf_2022.tif`. A pixel outside
    the raster gets the no-data value of its layer (250 and 0). Raise FileNotFoundError for a missing
    layer. Raise ValueError if a layer is not EPSG:5070, 30 m, north up, uint8 and on the DS-2 lattice,
    or if the 2 layers have different transforms (FM-85).
    """
    standard.parse_project_folder(folder)
    row, col = standard.parse_chunk_id(chunk)
    xmin, _, _, ymax = standard.chunk_bounds(row, col)
    want_row, want_col = standard.cell_of_bounds(xmin, ymax)
    arrays, transforms = [], []
    for name, fill in zip(VOTE_LAYERS, (_NLCD_FILL, _CLASSF_FILL)):
        path = Path(roots["stack"]) / folder / f"{name}.tif"
        if not path.is_file():
            raise FileNotFoundError(f"no stack layer: {path}")
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 5070:
                raise ValueError(f"{path}: the CRS must be {standard.CRS}, got {src.crs}")
            a, b, c, d, e, f = tuple(src.transform)[:6]
            if (a, b, d, e) != (standard.CELL, 0, 0, -standard.CELL) or src.dtypes[0] != "uint8":
                raise ValueError(f"{path}: need a 30 m north-up uint8 raster, got {(a, b, d, e)} {src.dtypes[0]}")
            first_row, first_col = standard.cell_of_bounds(c, f)  # raises ValueError off the lattice
            r0, c0 = want_row - first_row, want_col - first_col
            out = np.full((_WINDOW, _WINDOW), fill, dtype=np.uint8)
            r1, c1 = max(r0, 0), max(c0, 0)
            r2, c2 = min(r0 + _WINDOW, src.height), min(c0 + _WINDOW, src.width)
            if r1 < r2 and c1 < c2:
                out[r1 - r0:r2 - r0, c1 - c0:c2 - c0] = src.read(1, window=Window(c1, r1, c2 - c1, r2 - r1))
            arrays.append(out)
            transforms.append(src.transform)
    if transforms[0] != transforms[1]:
        raise ValueError(f"nlcd_2022 and classf_2022 have different transforms in {folder} (FM-85)")
    return arrays[0], arrays[1], from_origin(xmin, ymax, standard.CELL, standard.CELL)


# ---------------------------------------------------------------- polygon checks

def _grid_box(transform, shape):
    """The box of a raster with this transform and shape (rows, cols); the transform is north up."""
    return box(transform.c, transform.f + transform.e * shape[0], transform.c + transform.a * shape[1], transform.f)


def _check_polygon(polygon, area_box):
    """Raise ValueError unless `polygon` is a valid, non-empty Polygon inside `area_box` (FM-85)."""
    if (polygon is None or polygon.is_empty or polygon.geom_type != "Polygon" or not polygon.is_valid
            or not polygon.covered_by(area_box)):
        raise ValueError("every geometry must be a valid, non-empty Polygon inside the chunk box")


def _check_crs(gdf):
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        raise ValueError(f"the CRS must be {standard.CRS} (FM-85)")


# ---------------------------------------------------------------- D52 map vote

def pixel_weights(polygon, transform, shape):
    """Return `[(row, col, area_m2)]`: the exact area of polygon ∩ pixel box for each pixel the polygon covers (DS-29).

    A pixel that only touches the polygon has area 0 and is not in the list (FM-90).
    """
    size_x, size_y = transform.a, -transform.e
    min_x, min_y, max_x, max_y = polygon.bounds
    col0 = max(int(math.floor((min_x - transform.c) / size_x)), 0)
    col1 = min(int(math.ceil((max_x - transform.c) / size_x)), shape[1])
    row0 = max(int(math.floor((transform.f - max_y) / size_y)), 0)
    row1 = min(int(math.ceil((transform.f - min_y) / size_y)), shape[0])
    rows, cols = np.meshgrid(np.arange(row0, row1), np.arange(col0, col1), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    boxes = shapely.box(transform.c + cols * size_x, transform.f - (rows + 1) * size_y,
                        transform.c + (cols + 1) * size_x, transform.f - rows * size_y)
    areas = shapely.area(shapely.intersection(polygon, boxes))
    return [(int(r), int(c), float(a)) for r, c, a in zip(rows, cols, areas) if a > 0]


def vote_polygon(polygon, nlcd, classf, transform):
    """Return `(code_vote, p_vote)` of one polygon (DS-29, D52, D56). Raise ValueError for a bad polygon or map value."""
    if nlcd.shape != classf.shape:
        raise ValueError(f"the maps must have one shape, got {nlcd.shape} and {classf.shape} (FM-93)")
    _check_polygon(polygon, _grid_box(transform, nlcd.shape))
    buckets = {}
    for row, col, area in pixel_weights(polygon, transform, nlcd.shape):
        code = standard.vote_code(nlcd[row, col], classf[row, col])
        if code is not standard.VOTE_NODATA:
            buckets.setdefault(code, []).append(area)
    areas = {code: math.fsum(weights) for code, weights in buckets.items()}
    valid = math.fsum(areas.values())
    if valid == 0:
        return 0, 0.0
    shares = {code: round(area / valid, 9) for code, area in areas.items()}
    winner = min(shares, key=lambda code: (-shares[code], code))  # the largest share; on a tie the lowest code
    if winner == standard.VOTE_NONE:
        return 0, 0.0
    return winner, shares[winner]


def vote_polygons(gdf, nlcd, classf, transform):
    """Return a DataFrame `code_vote` (int64), `p_vote` (float64) with the index of `gdf` (DS-29)."""
    _check_crs(gdf)
    results = [vote_polygon(geom, nlcd, classf, transform) for geom in gdf.geometry]
    return pd.DataFrame({"code_vote": np.array([r[0] for r in results], dtype="int64"),
                         "p_vote": np.array([r[1] for r in results], dtype="float64")}, index=gdf.index)


# ---------------------------------------------------------------- D53 prompt label

def prompt_unions(model, rgb, config):
    """Return `{code: bool (1920, 1920)}`: the union of the masks of each prompt, one call for each code (D51, D53).

    The codes go in table order. A code with no mask has an all-False array. A mask that is not a
    1920 x 1920 bool array raises ValueError (FM-86).
    """
    _label_config(config)
    size = standard.PIXELS_PER_CHUNK
    unions = {}
    for code in standard.CODE_LIST:
        union = np.zeros((size, size), dtype=bool)
        masks = model.predict(rgb, standard.PROMPTS[code])
        for mask in (() if masks is None else list(masks)):  # FM-94: None means no mask
            segment._check_mask(mask)
            union |= mask
        unions[code] = union
    return unions


def polygon_pixels(polygon, transform):
    """Return the bool (1920, 1920) array of the polygon on the NAIP chunk grid; a pixel centre inside counts (DS-30)."""
    size = standard.PIXELS_PER_CHUNK
    standard.check_naip_transform(transform, size, size)
    _check_polygon(polygon, _grid_box(transform, (size, size)))
    return features.rasterize([(polygon, 1)], out_shape=(size, size), transform=transform, fill=0,
                              all_touched=False, dtype="uint8").astype(bool)


def prompt_polygon(pixels, unions, min_overlap):
    """Return `(code_prompt, p_prompt)` of one polygon (DS-30, D53).

    m = the number of polygon pixels inside the union of a code; the largest m wins, on a tie the
    lowest code. p = round(m / n, 9). No hit, n = 0, or p < `min_overlap` gives `(0, 0.0)`.
    """
    rows, cols = np.nonzero(pixels)
    n = len(rows)
    best_code, best_m = 0, 0
    for code in sorted(unions):
        m = int(np.count_nonzero(unions[code][rows, cols]))
        if m > best_m:
            best_code, best_m = code, m
    if n == 0 or best_m == 0:
        return 0, 0.0
    p = round(best_m / n, 9)
    if p < min_overlap:
        return 0, 0.0
    return best_code, p


def prompt_polygons(gdf, unions, transform, min_overlap):
    """Return a DataFrame `code_prompt` (int64), `p_prompt` (float64) with the index of `gdf` (DS-30)."""
    _check_crs(gdf)
    results = [prompt_polygon(polygon_pixels(geom, transform), unions, min_overlap) for geom in gdf.geometry]
    return pd.DataFrame({"code_prompt": np.array([r[0] for r in results], dtype="int64"),
                         "p_prompt": np.array([r[1] for r in results], dtype="float64")}, index=gdf.index)


# ---------------------------------------------------------------- D54 writer and code counts

def _check_label_metadata(label_metadata):
    if not isinstance(label_metadata, dict):
        raise ValueError("label metadata must be a mapping")
    bad = [key for key in LABEL_KEYS if not isinstance(label_metadata.get(key), str) or not label_metadata[key].strip()]
    if bad:
        raise ValueError(f"label metadata is missing or empty: {bad}")
    standard.check_run_id(label_metadata["label_run_id"])


def _check_label_method(run_id):
    """D54: the method of a label run ID is `langsam`; raise ValueError for any other method."""
    method = run_id[len("YYYYMMDD-HHMM-"):]
    if method != METHOD:
        raise ValueError(f"the label run ID must have the method {METHOD!r}, got {run_id!r}")


def _check_probability(values, name):
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)) \
                or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a number from 0 to 1, got {value!r} (FM-92)")


def _check_labels(gdf):
    """Raise ValueError unless the 4 label columns are complete and valid (D54 item 3)."""
    missing = [c for c in LABEL_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"the frame has no column {missing}")
    for code in gdf["code_prompt"].tolist():
        standard.check_machine_code(code)
    for code in gdf["code_vote"].tolist():
        standard.check_vote_code(code)
    for name in ("p_prompt", "p_vote"):
        _check_probability(gdf[name].tolist(), name)
    for code_col, p_col in (("code_prompt", "p_prompt"), ("code_vote", "p_vote")):
        if ((gdf[code_col] == 0) != (gdf[p_col] == 0)).any():
            raise ValueError(f"{code_col} 0 needs {p_col} 0.0, and a code other than 0 needs {p_col} above 0")


def write_labelled_polygons(gdf, roots, folder, chunk, metadata, label_metadata):
    """Rewrite `polygons/<chunk>.gpkg` with the 4 label columns and the `label_*` keys (D54). Return the path.

    `metadata` holds the 6 DS-21 keys of the segment run. Every check runs before the first write.
    The 1.2 columns and their order stay; the 4 columns go after them. The file goes through
    `storage.safe_write`, so a failure keeps the old file whole (FM-74).
    """
    segment.check_polygon_file(gdf, folder, chunk, metadata)
    _check_label_metadata(label_metadata)
    _check_labels(gdf)
    geometry = gdf.geometry.name
    old = [c for c in gdf.columns if c not in LABEL_COLUMNS and c != geometry]
    out = gdf[[*old, *LABEL_COLUMNS, geometry]].copy()
    out["code_prompt"] = out["code_prompt"].astype("int64")
    out["code_vote"] = out["code_vote"].astype("int64")
    out["p_prompt"] = out["p_prompt"].astype("float64")
    out["p_vote"] = out["p_vote"].astype("float64")
    # DS-21: the six 1.2 keys stay; only data_standard becomes the current version (the file now holds label columns).
    layer_metadata = {**{key: metadata[key] for key in standard.METADATA_KEYS},
                      "data_standard": str(standard.DATA_STANDARD_VERSION),
                      **{key: label_metadata[key] for key in LABEL_KEYS}}
    return segment.write_polygon_layer(out, roots, folder, chunk, layer_metadata)


def _count_rows():
    names = {0: "No label", standard.VOTE_DEVELOPED: standard.LEVEL1_CODES[standard.VOTE_DEVELOPED]}
    names.update({entry.code: entry.name for entry in standard.CODES})
    return sorted(names.items())


def count_codes(frames):
    """Return 34 rows (code 0, code 20 and the 32 codes, in code order) with the polygon count and area (D54, D56).

    Row keys: `code`, `name`, `n_prompt`, `n_vote`, `area_prompt_m2`, `area_vote_m2`. A code with
    0 polygons has a row with zeros (FM-2). An area is `math.fsum` rounded to 2 decimals.
    """
    frames = list(frames)
    if frames:
        data = pd.concat([f[["code_prompt", "code_vote", "area_m2"]] for f in frames], ignore_index=True)
    else:
        data = pd.DataFrame({"code_prompt": [], "code_vote": [], "area_m2": []})
    rows = []
    for code, name in _count_rows():
        row = {"code": code, "name": name}
        for kind in ("prompt", "vote"):
            areas = data.loc[data[f"code_{kind}"] == code, "area_m2"].tolist()
            row[f"n_{kind}"] = len(areas)
            row[f"area_{kind}_m2"] = round(math.fsum(areas), 2)
        rows.append({k: row[k] for k in ("code", "name", "n_prompt", "n_vote", "area_prompt_m2", "area_vote_m2")})
    return rows


# ---------------------------------------------------------------- run

def _read_polygon_file(roots, folder, chunk):
    """Return `(gdf, metadata)` of `polygons/<chunk>.gpkg`; metadata holds the 6 DS-21 keys. FileNotFoundError if missing."""
    path = segment.polygon_path(roots, folder, chunk)
    if not path.is_file():
        raise FileNotFoundError(f"no polygon file: {path} (run task 1.2 first)")
    info = pyogrio.read_info(path, layer="polygons")["layer_metadata"] or {}
    metadata = {key: info.get(key) for key in standard.METADATA_KEYS}
    return gpd.read_file(path, layer="polygons"), metadata


def _label_chunk(chunk, model, roots, folder, config, settings, label_run, code_version):
    """Label the polygons of one chunk. Return `(path, labelled frame)`."""
    gdf, metadata = _read_polygon_file(roots, folder, chunk)
    rgb, transform, _ = segment.read_chunk(Path(roots["fgt"]) / folder / "naip" / f"{chunk}.tif")  # D57
    nlcd, classf, transform30 = read_vote_layers(roots, folder, chunk)
    unions = prompt_unions(model, rgb, config)
    gdf = gdf.drop(columns=[c for c in LABEL_COLUMNS if c in gdf.columns])
    labelled = pd.concat([gdf, prompt_polygons(gdf, unions, transform, settings["min_overlap"]),
                          vote_polygons(gdf, nlcd, classf, transform30)], axis=1)
    model_tag = "fake" if getattr(model, "fake", False) else model.model_id
    label_metadata = {
        "label_run_id": label_run, "label_code_version": code_version,
        "label_created": segment._utc_text(segment._now()),
        "label_sources": (f"naip/{folder}/{chunk}.tif stack/{folder}/nlcd_2022.tif "
                          f"stack/{folder}/classf_2022.tif {model_tag}"),
    }
    path = write_labelled_polygons(gpd.GeoDataFrame(labelled, geometry=gdf.geometry.name, crs=gdf.crs),
                                   roots, folder, chunk, metadata, label_metadata)
    return path, labelled


def run_labels(chunks, model, roots, folder, config, *, run_id, probe, gpu_query=None):
    """Label the polygon file of each chunk. Return the list of paths. `run_id` is the label run ID (method `langsam`).

    The NAIP chunk comes from `<fgt>/<folder>/naip/<chunk>.tif` (D57: source GSD 0.6 m). The log
    (with the key `codes`) is written on success and on a failure; the error is re-raised.
    """
    chunks = list(chunks)
    if not chunks:
        raise ValueError("no chunk to run")
    for chunk in chunks:
        standard.parse_chunk_id(chunk)
    standard.check_run_id(run_id)
    _check_label_method(run_id)
    standard.parse_project_folder(folder)
    settings = _label_config(config)
    segment._refuse_existing_run(roots, run_id)
    frames = []

    def work(done):
        paths = []
        code_version = grid._code_version()
        for chunk in chunks:
            path, labelled = _label_chunk(chunk, model, roots, folder, config, settings, run_id, code_version)
            paths.append(path)
            frames.append(labelled)
            done.append(chunk)
        return paths

    return segment.run_logged(roots, run_id, model, chunks, probe, gpu_query, work,
                              extra=lambda: {"codes": count_codes(frames)})


# ---------------------------------------------------------------- real model and CLI

class _LangSamModel:
    """The real LangSAM model (GroundingDINO + SAM 2): one prompt for each call (D22, D51)."""

    fake = False

    def __init__(self, sam, settings):
        self._sam = sam
        self._box = settings["box_threshold"]
        self._text = settings["text_threshold"]
        self.model_id = f"langsam/{settings['model_id']}"

    def predict(self, rgb, prompt):
        from PIL import Image  # imported here: the module import stays light

        result = self._sam.predict(Image.fromarray(rgb), prompt, self._box, self._text, return_results=True)
        return _to_masks(result[0])


def _to_masks(raw):
    """Convert the masks of LangSAM (None, a torch tensor or an array: N x H x W or N x 1 x H x W) to bool arrays."""
    if raw is None:
        return []
    array = np.asarray(raw.cpu() if hasattr(raw, "cpu") else raw)
    if array.ndim == 4:
        array = array[:, 0]
    return [np.asarray(item > 0, dtype=bool) for item in array]


def _build_real_model(settings):
    """Build the real model. It loads weights, so only the user's `--run` calls it (D22)."""
    from samgeo.text_sam import LangSAM  # imported here: importing semseg.label must not import samgeo

    storage.use_soft_hf_locks()  # D58: the share refuses FileLock
    return _LangSamModel(LangSAM(model_type=settings["model_id"]), settings)


def _default_loader(settings, environ):
    if environ.get(segment.FAKE_ENV):
        raise RuntimeError(f"{segment.FAKE_ENV} is set: the real LangSAM model is not built (D22, FM-71)")
    return _build_real_model(settings)


def _parser():
    parser = argparse.ArgumentParser(prog="python -m semseg.label", description=__doc__.splitlines()[0],
                                     allow_abbrev=False)  # D50: `--ru` must not start a run
    parser.add_argument("--demo", action="store_true", required=True,
                        help="the demo chunks of config.yaml (demo.chunks)")
    parser.add_argument("--run", action="store_true",
                        help="really run: loads LangSAM on the GPU (only after Jack says run, D22)")
    return parser


def main(argv=None, roots=None, config=None, probe=None, loader=None, now=None):
    """Label the demo chunks. Return 0. Without `--run` it exits with an error and loads nothing (D22)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("--demo needs --run: nothing starts without --run (D22)")
    roots = storage.load_roots() if roots is None else roots
    config = load_config() if config is None else config
    storage.guard_share(roots)
    probe = probe or segment.read_available_mb
    now = segment._now() if now is None else now
    demo = config.get("demo") if isinstance(config, dict) else None
    if not isinstance(demo, dict):
        raise ValueError("config `demo` is missing or not a mapping (FM-84)")
    folder = grid.find_folder(demo["pid"], roots)
    settings = _label_config(config)
    run_id = standard.run_id(now, METHOD)
    segment._refuse_existing_run(roots, run_id)  # D49: before any write, also before the low-memory log
    started = segment._utc_text(now)
    environ = os.environ if loader is None else dict(os.environ)  # an injected loader must not change the environment

    def load(_segment_model_id, env):  # the memory check and HF_HOME of segment.load_sam2, the model of `label:`
        if loader is None:
            return _default_loader(settings, env)
        return loader(settings["model_id"], env)

    try:
        model = segment.load_sam2(roots, config, probe=probe, loader=load, environ=environ)
    except SystemExit as stop:
        if stop.code == segment.EXIT_LOW_MEMORY:
            memory = stop.available_mb
            segment.write_run_log(roots, run_id, model_id=f"langsam/{settings['model_id']}", fake=False,
                                  memory_before_mb=memory, memory_after_mb=memory, started=started,
                                  finished=segment._utc_text(segment._now()), chunks=demo["chunks"],
                                  chunks_done=[], error="not enough available memory (D47): nothing was loaded",
                                  extra={"codes": count_codes([])})
        raise
    for path in run_labels(demo["chunks"], model, roots, folder, config, run_id=run_id, probe=probe):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
