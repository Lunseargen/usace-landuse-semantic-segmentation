"""SAM 2 polygons for NAIP chunks (task 1.2, I1, I2, FM-6, FM-70 to FM-75, D46 to D48).

Run (only after Jack says "run", D22):
`uv run python -m semseg.segment --demo --run` (the demo chunks of `config.yaml`), or
`uv run python -m semseg.segment --pid 77 --run` (all chunks of `<semseg>/<folder>/chunks.parquet`, D59).

Steps for each run: guard the share, set `HF_HOME`, check the available memory (exit 4 if less
than `segment.min_free_mb`), load the model, then for each chunk read the NAIP file, make the
masks, make the polygons, and write `polygons/<chunk>.gpkg`. The run log is
`<semseg>/runs/<run_id>/log.json`.

D22: importing this module imports no SAM 2, samgeo, torch, or Hugging Face module. The real
model loads only inside `_build_real_model`, and only when the user runs the command above.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features
from shapely.geometry import box, shape

from semseg import grid, setup_check, standard, storage
from semseg.config import MIN_FREE_FLOOR_MB, load_config, non_negative_number

METHOD = "sam2"
EXIT_LOW_MEMORY = 4
DEFAULT_SIMPLIFY_M = 0.5
DEFAULT_MIN_AREA_M2 = 2.0
DEFAULT_MODEL_ID = "sam2-hiera-large"
FAKE_ENV = "SEMSEG_FAKE_MODEL"
_POLYGONS = "polygons"
NAIP_GSD = "0.6"  # D57: the source GSD of every NAIP chunk, in metres, as the file tag `gsd`
_YEAR_RE = re.compile(r"[0-9]{4}", re.ASCII)


class MaskModel(Protocol):
    """A mask model. `generate` gets an RGB uint8 array (H, W, 3) and returns bool masks (H, W)."""

    model_id: str
    fake: bool

    def generate(self, rgb):  # pragma: no cover (protocol)
        ...


def _utc_text(when):
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- memory check (FM-6, D47)

def read_available_mb(run=subprocess.run):
    """Return the available memory in MB from `free -m`. Raise RuntimeError if it is not readable."""
    value = setup_check.read_memory(run)["available_mb"]
    if value is None:
        raise RuntimeError("cannot read the available memory from `free -m`")
    return value


def _min_free_mb(config):
    """Return `segment.min_free_mb`. Raise ValueError if it is missing or not an integer >= the floor (D50)."""
    section = config.get("segment") if isinstance(config, dict) else None
    value = section.get("min_free_mb") if isinstance(section, dict) else None
    if type(value) is not int or value < MIN_FREE_FLOOR_MB:
        raise ValueError(f"config segment.min_free_mb must be an integer of {MIN_FREE_FLOOR_MB} or more (D47, D50)")
    return value


def _model_id(config):
    section = config.get("segment") if isinstance(config, dict) else None
    value = section.get("model_id", DEFAULT_MODEL_ID) if isinstance(section, dict) else DEFAULT_MODEL_ID
    if not isinstance(value, str) or not value.strip():
        raise ValueError("config segment.model_id must be a non-empty string")
    return value


class _Sam2Model:
    """The real SAM 2 model: wraps the samgeo `SamGeo2` automatic mask generator (D22)."""

    fake = False

    def __init__(self, sam, model_id):
        self._sam = sam
        self.model_id = f"facebook/{model_id}"

    def generate(self, rgb):
        self._sam.generate(rgb)
        return [np.asarray(item["segmentation"], dtype=bool) for item in self._sam.masks]


def _build_real_model(model_id):
    """Build the real model. It loads weights, so only the user's `--run` calls it (D22)."""
    from samgeo import SamGeo2  # imported here: importing semseg.segment must not import samgeo

    storage.use_soft_hf_locks()  # D58: the share refuses FileLock
    return _Sam2Model(SamGeo2(model_id=model_id, device="cuda", automatic=True), model_id)


def _default_loader(model_id, environ):
    if environ.get(FAKE_ENV):
        raise RuntimeError(f"{FAKE_ENV} is set: the real SAM 2 model is not built (D22, FM-71)")
    return _build_real_model(model_id)


def load_sam2(roots, config, *, probe=None, loader=None, environ=None):
    """Return the model. Order: `HF_HOME` and share guard, config, memory check, load (FM-71, FM-72).

    Exit 3 if the share is missing. ValueError if `HF_HOME` is off the share or the config is not
    valid. Exit 4 if `probe()` (MB available) is less than `segment.min_free_mb`: nothing loads.
    `loader(model_id, environ)` builds the model; the default builds the real SAM 2 model.
    """
    environ = os.environ if environ is None else environ
    storage.ensure_hf_home(roots, environ)
    need = _min_free_mb(config)
    model_id = _model_id(config)
    available = (probe or read_available_mb)()
    if available < need:
        print(f"memory check: {available} MB available, {need} MB needed (D47): stop",
              file=sys.stderr)
        stop = SystemExit(EXIT_LOW_MEMORY)
        stop.available_mb = available  # the value that caused the stop: `main` logs this one
        raise stop
    return (loader or _default_loader)(model_id, environ)


# ---------------------------------------------------------------- I1 NAIP chunk

def _check_gsd(src, path):
    """D57: raise ValueError unless the tag `gsd` of the file is exactly "0.6" (the source GSD, DS-4)."""
    gsd = src.tags().get("gsd")
    if gsd != NAIP_GSD:
        raise ValueError(f"{path}: the tag gsd must be {NAIP_GSD!r} (NAIP source GSD 0.6 m, D57), got {gsd!r}")


def chunk_naip_year(path):
    """Return the NAIP year of a chunk file as an int, from the tag `years` (D57, DS-5).

    Raise FileNotFoundError for a missing file. Raise ValueError if the tag `gsd` is not "0.6", or
    if `years` is missing or is not one 4-digit year (several years are refused). The call reads
    the tags only, not the pixels.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no NAIP chunk file: {path}")
    with rasterio.open(path) as src:
        _check_gsd(src, path)
        years = src.tags().get("years")
    if not isinstance(years, str) or _YEAR_RE.fullmatch(years) is None:
        raise ValueError(f"{path}: the tag years must be one 4-digit year (D57), got {years!r}")
    return int(years)


def read_chunk(path):
    """Read a NAIP chunk file. Return `(rgb uint8 (H, W, 3), transform, crs)`.

    Raise FileNotFoundError for a missing file. Raise ValueError if the profile is not I1
    (EPSG:5070, 1920 x 1920, 0.5 m on the lattice, 3 or more uint8 bands), if the file name does
    not match the box of the file (DS-4, DS-7), or if the tag `gsd` is missing or not "0.6" (D57).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no NAIP chunk file: {path}")
    row, col = standard.parse_chunk_id(path.stem)
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            raise ValueError(f"{path}: the CRS must be {standard.CRS}, got {src.crs}")
        standard.check_naip_transform(src.transform, src.width, src.height)
        if src.count < 3 or any(dtype != "uint8" for dtype in src.dtypes):
            raise ValueError(f"{path}: need 3 or more uint8 bands, got {src.count} {src.dtypes}")
        xmin, _, _, ymax = standard.chunk_bounds(row, col)
        if (src.transform.c, src.transform.f) != (xmin, ymax):
            raise ValueError(f"{path}: the file name does not match the box of the file")
        _check_gsd(src, path)
        rgb = np.ascontiguousarray(src.read([1, 2, 3]).transpose(1, 2, 0))
        return rgb, src.transform, src.crs


# ---------------------------------------------------------------- I2 polygons

def _polygon_config(config):
    """Return `(simplify_tolerance_m, min_area_m2)`. Raise ValueError for a bad value."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    section = (config or {}).get("segment")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("config segment must be a mapping (FM-84)")
    values = []
    for key, default in (("simplify_tolerance_m", DEFAULT_SIMPLIFY_M),
                         ("min_area_m2", DEFAULT_MIN_AREA_M2)):
        value = section.get(key, default)
        if not non_negative_number(value):
            raise ValueError(f"config segment.{key} must be a finite number of 0 or more, got {value!r}")
        values.append(float(value))
    return tuple(values)


def _check_mask(mask):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("a mask must be a 2-D numpy array of bool")
    size = standard.PIXELS_PER_CHUNK
    if mask.shape != (size, size):
        raise ValueError(f"a mask must have the shape ({size}, {size}), got {mask.shape} (FM-80)")


def _check_transform(transform, chunk_bounds, expected):
    """Raise ValueError unless `transform` is the DS-4 transform of the chunk (FM-79)."""
    xmin, _, _, ymax = expected
    want = (standard.PIXEL, 0.0, xmin, 0.0, -standard.PIXEL, ymax)
    try:
        got = tuple(float(v) for v in tuple(transform)[:6])
    except (TypeError, ValueError):
        raise ValueError(f"not an affine transform: {transform!r}") from None
    if got != want:
        raise ValueError(f"the transform {got} is not the chunk transform {want} (FM-79)")
    if tuple(chunk_bounds) != tuple(expected):
        raise ValueError(f"the bounds {tuple(chunk_bounds)} are not the box of the chunk {expected} (FM-79)")


def _check_bounds(bounds):
    values = tuple(bounds)
    if (len(values) != 4 or not all(math.isfinite(v) for v in values)
            or values[0] >= values[2] or values[1] >= values[3]):
        raise ValueError(f"not a chunk box (xmin, ymin, xmax, ymax): {bounds!r}")
    return values


def _parts(geom):
    """Return the non-empty Polygon parts of a geometry (lines and points are dropped)."""
    return [part for part in shapely.get_parts(geom)
            if part.geom_type == "Polygon" and not part.is_empty]


def _clean(geom, chunk_box, tolerance, min_area):
    """Return the valid polygons of one raster shape: clip, simplify, repair, clip, area (FM-73)."""
    geom = shapely.make_valid(geom).intersection(chunk_box)
    geom = shapely.make_valid(geom.simplify(tolerance, preserve_topology=True)).intersection(chunk_box)
    return [part for part in _parts(geom) if part.area >= min_area]


def masks_to_polygons(masks, transform, chunk_bounds, run_id, chunk, config=None):
    """Return a GeoDataFrame (`polygon_id`, `area_m2`, `geometry`, EPSG:5070), one row for each polygon part.

    `masks` are bool (H, W) arrays on the pixel grid of `transform` (pixel (0, 0) is the upper-left
    corner of the chunk, FM-70). The polygons come from `rasterio.features.shapes`, then they are
    made valid, clipped to the chunk box, simplified (`segment.simplify_tolerance_m`), made valid,
    clipped again, and dropped if the area is less than `segment.min_area_m2`. IDs are DS-9 and
    number from 1. Raise ValueError for a bad mask (not bool, not 1920 x 1920), ID, box, transform
    (not the DS-4 transform of `chunk`) or config value.
    """
    standard.check_run_id(run_id)
    row, col = standard.parse_chunk_id(chunk)
    tolerance, min_area = _polygon_config(config)
    bounds = _check_bounds(chunk_bounds)
    _check_transform(transform, bounds, standard.chunk_bounds(row, col))
    chunk_box = box(*bounds)
    masks = list(masks)  # FM-78: an iterator is read once
    for mask in masks:
        _check_mask(mask)
    geoms = []
    for mask in masks:
        for value, _ in features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
            geoms.extend(_clean(shape(value), chunk_box, tolerance, min_area))
    ids = [standard.polygon_id(chunk, run_id, n) for n in range(1, len(geoms) + 1)]
    return gpd.GeoDataFrame({"polygon_id": ids, "area_m2": np.array([g.area for g in geoms], dtype=float)},
                            geometry=geoms, crs=standard.CRS)


# ---------------------------------------------------------------- writers (FM-74, FM-75)

def _check_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    bad = [key for key in standard.METADATA_KEYS
           if not isinstance(metadata.get(key), str) or not metadata[key].strip()]
    if bad:
        raise ValueError(f"DS-21 metadata is missing or empty: {bad}")
    standard.check_run_id(metadata["run_id"])


def _check_frame(gdf, chunk, run_id):
    """Raise ValueError unless every row has a unique DS-9 ID and a valid Polygon inside the chunk (FM-81)."""
    if "polygon_id" not in gdf.columns:
        raise ValueError("the frame has no `polygon_id` column")
    ids = list(gdf["polygon_id"])
    for text in ids:
        if standard.parse_polygon_id(text)[:2] != (chunk, run_id):
            raise ValueError(f"polygon ID {text!r} does not match the chunk and run ID")
    if len(set(ids)) != len(ids):
        raise ValueError("polygon_id values are not unique")
    box_ = box(*standard.chunk_bounds(*standard.parse_chunk_id(chunk)))
    for geom in gdf.geometry:
        if (geom is None or geom.is_empty or geom.geom_type != "Polygon" or not geom.is_valid
                or not geom.covered_by(box_)):
            raise ValueError("every geometry must be a valid, non-empty Polygon inside the chunk box")


def check_polygon_file(gdf, folder, chunk, metadata):
    """Raise ValueError unless the frame, the names and the DS-21 metadata are valid for a polygon file."""
    _check_metadata(metadata)
    standard.parse_project_folder(folder)
    standard.check_name(folder)
    standard.parse_chunk_id(chunk)
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        raise ValueError(f"the CRS must be {standard.CRS}")
    _check_frame(gdf, chunk, metadata["run_id"])


def polygon_path(roots, folder, chunk):
    """Return `<semseg>/<folder>/polygons/<chunk>.gpkg`."""
    return Path(roots["semseg"]) / folder / _POLYGONS / f"{chunk}.gpkg"


def write_polygon_layer(gdf, roots, folder, chunk, layer_metadata):
    """Write the frame as layer `polygons` with `layer_metadata`, through `storage.safe_write` (FM-74). Return the path."""
    target = polygon_path(roots, folder, chunk)
    with tempfile.TemporaryDirectory(prefix="semseg_") as tmp:
        local = Path(tmp) / target.name
        gdf.to_file(local, layer=_POLYGONS, driver="GPKG", engine="pyogrio", layer_metadata=layer_metadata)
        data = local.read_bytes()
    return storage.safe_write(target, data, roots)


def write_polygons(gdf, roots, folder, chunk, metadata):
    """Write `<semseg>/<folder>/polygons/<chunk>.gpkg` (layer `polygons`, DS-21 in the layer metadata).

    Every check runs before the first write. The file goes through `storage.safe_write`, so a
    failure leaves no part file and keeps an old file whole (FM-74). Return the path.
    """
    check_polygon_file(gdf, folder, chunk, metadata)
    return write_polygon_layer(gdf, roots, folder, chunk, {key: metadata[key] for key in standard.METADATA_KEYS})


def write_run_log(roots, run_id, *, model_id, fake, memory_before_mb, memory_after_mb, started,
                  finished, chunks, chunks_done, error=None, gpu_before=None, gpu_after=None,
                  log_errors=(), extra=None):
    """Write `<semseg>/runs/<run_id>/log.json` with `storage.safe_write`. Return the path.

    `fake` marks a fake model (FM-75). `log_errors` lists failed reads after the run (FM-77). The GPU values come from `setup_check.query_gpu` (they can
    be `None` on the GB10, D37). `extra` is a mapping of more keys (task 1.3: `codes`, D54).
    """
    standard.check_run_id(run_id)
    text = json.dumps({
        "run_id": run_id, "model_id": model_id, "fake": bool(fake),
        "memory_before_mb": memory_before_mb, "memory_after_mb": memory_after_mb,
        "gpu_before": gpu_before, "gpu_after": gpu_after,
        "started": started, "finished": finished,
        "chunks": list(chunks), "chunks_done": list(chunks_done), "error": error,
        "log_errors": list(log_errors), **(extra or {}),
    }, indent=2, sort_keys=True)
    target = Path(roots["semseg"]) / "runs" / run_id / "log.json"
    return storage.safe_write(target, (text + "\n").encode("utf-8"), roots)


# ---------------------------------------------------------------- run

def _metadata(run_id, folder, chunk, code_version, now, naip_year):
    return {"run_id": run_id, "code_version": code_version,
            "data_standard": str(standard.DATA_STANDARD_VERSION), "naip_year": str(naip_year),
            "sources": f"naip/{folder}/{chunk}.tif", "created": _utc_text(now)}


def _refuse_existing_run(roots, run_id):
    """Raise FileExistsError if `<semseg>/runs/<run_id>/` exists: a run never replaces an old run (D49, FM-83)."""
    folder = Path(roots["semseg"]) / "runs" / run_id
    if folder.exists():
        raise FileExistsError(f"the run folder exists: {folder} (D49): wait one minute or remove it")


def run_chunks(chunks, model, roots, folder, config, *, run_id, probe, gpu_query=None):
    """Make the polygon file of each chunk. Return the list of paths.

    NAIP comes from `<fgt>/<folder>/naip/<chunk>.tif`. The log is written on success and on a
    failure (the error is re-raised). The memory is read with `probe()` before and after.
    """
    chunks = list(chunks)
    if not chunks:
        raise ValueError("no chunk to run")
    for chunk in chunks:
        standard.parse_chunk_id(chunk)
    standard.check_run_id(run_id)
    standard.parse_project_folder(folder)
    _refuse_existing_run(roots, run_id)

    def work(done):
        paths = []
        code_version = grid._code_version()
        for chunk in chunks:
            row, col = standard.parse_chunk_id(chunk)
            naip = Path(roots["fgt"]) / folder / "naip" / f"{chunk}.tif"
            rgb, transform, _ = read_chunk(naip)
            naip_year = chunk_naip_year(naip)  # D57, DS-5: the year of the file, not a constant
            gdf = masks_to_polygons(model.generate(rgb), transform, standard.chunk_bounds(row, col),
                                    run_id, chunk, config)
            meta = _metadata(run_id, folder, chunk, code_version, _now(), naip_year)
            paths.append(write_polygons(gdf, roots, folder, chunk, meta))
            done.append(chunk)
        return paths

    return run_logged(roots, run_id, model, chunks, probe, gpu_query, work)


def run_logged(roots, run_id, model, chunks, probe, gpu_query, work, extra=None):
    """Run `work(done)` and write the run log on success and on a failure (FM-76, FM-77). Return its result.

    `work` appends each finished chunk to the list `done`. The memory is read with `probe()` before
    and after. `extra()` (optional) returns more log keys, read after the work (task 1.3: `codes`).
    The first error stays the one raised; a failed read or a failed log write is chained or logged.
    """
    gpu_query = gpu_query or setup_check.query_gpu
    started, memory_before, gpu_before = _utc_text(_now()), probe(), gpu_query()
    done, result, error, first = [], None, None, None
    try:
        result = work(done)
    except BaseException as exc:  # FM-76: also KeyboardInterrupt goes in the log
        error, first = f"{type(exc).__name__}: {exc}", exc
        raise
    finally:
        # FM-77: a failed read after the run goes in the log; the first error stays the one raised.
        log_errors, memory_after, gpu_after = [], None, None
        try:
            memory_after = probe()
        except Exception as exc:
            log_errors.append(f"memory_after: {type(exc).__name__}: {exc}")
        try:
            gpu_after = gpu_query()
        except Exception as exc:
            log_errors.append(f"gpu_after: {type(exc).__name__}: {exc}")
        try:
            write_run_log(roots, run_id, model_id=model.model_id, fake=getattr(model, "fake", False),
                          memory_before_mb=memory_before, memory_after_mb=memory_after, started=started,
                          finished=_utc_text(_now()), chunks=chunks, chunks_done=done, error=error,
                          gpu_before=gpu_before, gpu_after=gpu_after, log_errors=log_errors,
                          extra=extra() if extra else None)
        except Exception as log_exc:
            if first is None:
                raise  # no first error: the log error is the only one
            # The first error stays the one raised; the log error is chained as a note.
            first.add_note(f"the run log also failed: {type(log_exc).__name__}: {log_exc}")
    return result


# ---------------------------------------------------------------- D46 demo chunks and CLI

def select_demo_chunks(df, n=6):
    """Return the `n` chunk IDs with the largest `footprint_share` (tie: lowest row, then lowest col) (D46)."""
    if len(df) < n:
        raise ValueError(f"need {n} chunks, the list has {len(df)}")
    ranked = df.sort_values(["footprint_share", "row", "col"], ascending=[False, True, True])
    return ranked["chunk"].head(n).tolist()


def _parser():
    parser = argparse.ArgumentParser(prog="python -m semseg.segment", description=__doc__.splitlines()[0],
                                     allow_abbrev=False)  # D50: `--ru` must not start a run
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--demo", action="store_true",
                       help="the demo chunks of config.yaml (demo.chunks)")
    which.add_argument("--pid", type=int,
                       help="all chunks of <semseg>/<pid>_<NAME>/chunks.parquet (D59)")
    parser.add_argument("--run", action="store_true",
                        help="really run: loads SAM 2 on the GPU (only after Jack says run, D22)")
    return parser


def pilot_chunks(roots, folder):
    """Return the chunk IDs of `<semseg>/<folder>/chunks.parquet` in file order (D59).

    Raise FileNotFoundError if the file is missing, ValueError if it has no `chunk` column or no row.
    """
    path = Path(roots["semseg"]) / folder / standard.CHUNK_LIST
    if not path.is_file():
        raise FileNotFoundError(f"no chunk list: {path} (task 1.1 writes it)")
    df = pd.read_parquet(path)
    if "chunk" not in df.columns:
        raise ValueError(f"{path}: no column `chunk`")
    chunks = [str(chunk) for chunk in df["chunk"]]
    if not chunks:
        raise ValueError(f"{path}: the chunk list is empty")
    return chunks


def main(argv=None, roots=None, config=None, probe=None, loader=None, now=None):
    """Run the demo chunks or all chunks of one pilot. Return 0.

    Without `--run` it exits with an error and loads nothing (D22).
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run:
        flag = "--demo" if args.demo else "--pid"
        parser.error(f"{flag} needs --run: nothing starts without --run (D22)")
    roots = storage.load_roots() if roots is None else roots
    config = load_config() if config is None else config
    storage.guard_share(roots)
    probe = probe or read_available_mb
    now = _now() if now is None else now
    if args.demo:
        demo = config.get("demo") if isinstance(config, dict) else None
        if not isinstance(demo, dict):
            raise ValueError("config `demo` is missing or not a mapping (FM-84)")
        folder = grid.find_folder(demo["pid"], roots)
        chunks = demo["chunks"]
    else:
        folder = grid.find_folder(args.pid, roots)
        chunks = pilot_chunks(roots, folder)
    run_id = standard.run_id(now, METHOD)
    _refuse_existing_run(roots, run_id)  # D49: before any write, also before the low-memory log
    started = _utc_text(now)
    # An injected loader (a test) must not change the real environment.
    environ = os.environ if loader is None else dict(os.environ)
    try:
        model = load_sam2(roots, config, probe=probe, loader=loader, environ=environ)
    except SystemExit as stop:
        if stop.code == EXIT_LOW_MEMORY:
            memory = stop.available_mb
            write_run_log(roots, run_id, model_id=f"facebook/{_model_id(config)}", fake=False,
                          memory_before_mb=memory, memory_after_mb=memory, started=started,
                          finished=_utc_text(_now()), chunks=chunks, chunks_done=[],
                          error="not enough available memory (D47): nothing was loaded")
        raise
    for path in run_chunks(chunks, model, roots, folder, config, run_id=run_id, probe=probe):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
