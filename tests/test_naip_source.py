"""Tests of D57: `semseg.segment.read_chunk` accepts only a NAIP chunk with source GSD 0.6 m.

API under test (names fixed here, see `.factory/1.3/tests.md`):
  segment.read_chunk(path) -> (rgb, transform, crs)   unchanged; ValueError if tag `gsd` is missing or not "0.6"
  segment.chunk_naip_year(path) -> int                the `years` tag of the file (same `gsd` check)

The chunk file stays on the 0.5 m grid (DS-4); the tag `gsd` is the source GSD (DS-5). The synthetic
NAIP fixtures of the older tests (`write_tif` in tests/test_segment.py and tests/test_edge_segment.py)
need the tags `gsd` = "0.6" and `years` = "2022" once read_chunk checks them.

D22: no model. The synthetic tests write only under `tmp_path`. The real-data test reads the share.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from semseg import segment, standard, storage

ROW, COL = 2295, 2394
CHUNK = standard.chunk_id(ROW, COL)
XMIN, YMIN, XMAX, YMAX = standard.chunk_bounds(ROW, COL)
N = standard.PIXELS_PER_CHUNK
PILOTS = {"77_GRAPEVINE_LAKE": 157, "392_RAYSTOWN_LAKE": 227}


def write_naip(path, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((4, N, N), dtype="uint8")
    for band in range(4):
        data[band] = 10 * (band + 1)
    profile = {"driver": "GTiff", "width": N, "height": N, "count": 4, "dtype": "uint8", "crs": "EPSG:5070",
               "transform": from_origin(XMIN, YMAX, 0.5, 0.5)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    return path


def test_d57_chunk_with_gsd_0_6_is_accepted(tmp_path):
    """D57: gsd "0.6" passes; the rgb, transform and CRS are those of the file."""
    path = write_naip(tmp_path / f"{CHUNK}.tif", {"gsd": "0.6", "years": "2022"})
    rgb, transform, crs = segment.read_chunk(path)
    assert rgb.shape == (N, N, 3) and rgb.dtype == np.uint8 and (rgb[0, 0] == [10, 20, 30]).all()
    assert transform == from_origin(XMIN, YMAX, 0.5, 0.5) and crs.to_epsg() == 5070
    assert segment.chunk_naip_year(path) == 2022


def test_d57_chunk_without_gsd_tag_is_refused(tmp_path):
    """D57: a missing gsd tag raises ValueError that names gsd (no tag at all, or only years)."""
    for name, tags in (("none", None), ("years_only", {"years": "2022"})):
        path = write_naip(tmp_path / name / f"{CHUNK}.tif", tags)
        with pytest.raises(ValueError, match="gsd"):
            segment.read_chunk(path)


@pytest.mark.parametrize("gsd", ["0.5", "1.0", "1", "0.3", "0.9", "", "abc", "0.6.0", "-0.6"])
def test_d57_chunk_with_gsd_not_0_6_is_refused(tmp_path, gsd):
    """D57: any gsd value other than "0.6" raises ValueError."""
    path = write_naip(tmp_path / f"{CHUNK}.tif", {"gsd": gsd, "years": "2022"})
    with pytest.raises(ValueError, match="gsd"):
        segment.read_chunk(path)


@pytest.mark.parametrize("year", ["2022", "2021", "2018"])
def test_d57_chunk_naip_year_comes_from_the_years_tag(tmp_path, year):
    """D57, DS-5: the NAIP year of the chunk is the integer of the `years` tag."""
    path = write_naip(tmp_path / f"{CHUNK}.tif", {"gsd": "0.6", "years": year})
    value = segment.chunk_naip_year(path)
    assert value == int(year) and isinstance(value, int)


def test_d57_chunk_naip_year_refuses_bad_files(tmp_path):
    """D57: a missing or non-numeric years tag, a gsd that is not 0.6, or a missing file raise."""
    cases = {"no_years": {"gsd": "0.6"}, "text_years": {"gsd": "0.6", "years": "abc"},
             "empty_years": {"gsd": "0.6", "years": ""}, "gsd_0_5": {"gsd": "0.5", "years": "2022"}}
    for name, tags in cases.items():
        path = write_naip(tmp_path / name / f"{CHUNK}.tif", tags)
        with pytest.raises(ValueError):
            segment.chunk_naip_year(path)
    with pytest.raises(FileNotFoundError):
        segment.chunk_naip_year(tmp_path / "none" / f"{CHUNK}.tif")


def test_d57_all_384_fine_grid_test_chunks_of_77_and_392_pass():
    """D57: read-only. Each chunk of 77 (157) and 392 (227) has gsd 0.6 and the year 2022.

    Fast: `chunk_naip_year` opens each file once and reads only the tags (no pixel read, no
    directory scan), in parallel threads. The pixel path of `read_chunk` has the synthetic tests
    above. The test never skips. If the lab share is not mounted, guard_share ends with exit code 3.
    """
    roots = storage.load_roots()
    storage.guard_share(roots)
    paths = []
    for folder, expected in PILOTS.items():
        listed = pd.read_parquet(Path(roots["semseg"]) / folder / standard.CHUNK_LIST)["chunk"].tolist()
        assert len(listed) == expected
        paths += [Path(roots["fgt"]) / folder / "naip" / f"{chunk}.tif" for chunk in listed]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), ThreadPoolExecutor(max_workers=16) as pool:
        years = list(pool.map(segment.chunk_naip_year, paths))
    assert len(years) == 384 and set(years) == {2022}
