"""Edge attacks on `semseg.grid` (task 1.1). New FMEA rows FM-60 to FM-69.

The fix round (FM-60 to FM-69) removed every xfail mark. Every test uses
synthetic data under `tmp_path` (fake roots). No test reads or writes the real share, the stack
or `fine_grid_test`. No GPU.
"""
import subprocess
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon, box

from semseg import grid, standard
from test_grid import make_roots, write_boundary

X0, Y0, CHUNK = standard.X0, standard.Y0, standard.CHUNK
FOOTPRINT = box(X0 + 1000, Y0 - 5000, X0 + 3000, Y0 - 4000)       # 8 chunks, rows 4 to 5
ALBERS_5070 = ("+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 "
               "+datum=NAD83 +units=m +no_defs")


def keys(frame):
    return set(zip(frame["row"].tolist(), frame["col"].tolist()))


def good_frame(pid=5):
    return grid.chunk_list(FOOTPRINT, pid)


def write_target(tmp_path, frame, folder="5_X"):
    roots = make_roots(tmp_path)
    return roots, grid.write_chunk_list(frame, roots, folder)


# ------------------------------------------------------------------ footprint shapes (FM-54)

def test_edge_multipolygon_footprint_is_union_of_parts():
    part_a = box(X0 + 1000, Y0 - 5000, X0 + 3000, Y0 - 4000)
    part_b = box(X0 + 9000, Y0 - 9000, X0 + 9500, Y0 - 8800)
    both = grid.chunk_list(MultiPolygon([part_a, part_b]), 1)
    one, two = grid.chunk_list(part_a, 1), grid.chunk_list(part_b, 1)
    assert keys(both) == keys(one) | keys(two)
    assert both["chunk"].is_unique
    assert both["footprint_share"].sum() == pytest.approx(
        one["footprint_share"].sum() + two["footprint_share"].sum(), abs=1e-3)


def test_edge_multipolygon_in_gpkg_loads(tmp_path):
    parts = MultiPolygon([box(X0 + 1000, Y0 - 5000, X0 + 3000, Y0 - 4000),
                          box(X0 + 9000, Y0 - 9000, X0 + 9500, Y0 - 8800)])
    path = write_boundary(tmp_path / "b.gpkg", [parts])
    assert isinstance(grid.load_footprint(path), MultiPolygon)


def test_edge_hole_chunk_far_inside_hole_is_out_and_near_edge_is_in_with_zero_share():
    outer = box(X0 + 960, Y0 - 960 * 20, X0 + 960 * 10, Y0 - 960 * 8)
    wide = box(X0 + 960 * 3, Y0 - 960 * 16, X0 + 960 * 9, Y0 - 960 * 10)
    # a chunk 100 m from every hole edge: in the buffer band, share 0.0 (D43: present in the list)
    hole_small = Polygon(outer.exterior.coords, [box(X0 + 3740, Y0 - 14500, X0 + 4900, Y0 - 13340).exterior.coords])
    small = grid.chunk_list(hole_small, 1)
    row = small[small["chunk"] == "r14_c4"]
    assert len(row) == 1 and row["footprint_share"].iloc[0] == 0.0
    # a chunk more than 150 m from every hole edge: not in the list
    hole_wide = Polygon(outer.exterior.coords, [wide.exterior.coords])
    big = grid.chunk_list(hole_wide, 1)
    assert (12, 5) not in keys(big)
    assert (big["footprint_share"] == 0.0).sum() > 0


def test_edge_hole_share_is_reduced_by_hole_area():
    solid = box(X0 + 960 * 2 - 500, Y0 - 960 * 6 - 500, X0 + 960 * 3 + 500, Y0 - 960 * 5 + 500)
    hole = box(X0 + 960 * 2 + 100, Y0 - 960 * 6 + 100, X0 + 960 * 2 + 300, Y0 - 960 * 6 + 300)
    holed = Polygon(solid.exterior.coords, [hole.exterior.coords])
    full = grid.chunk_list(solid, 1).set_index("chunk")["footprint_share"]
    part = grid.chunk_list(holed, 1).set_index("chunk")["footprint_share"]
    diff = (full - part).round(4)
    assert diff.sum() == pytest.approx(200 * 200 / CHUNK ** 2, abs=1e-4)


def test_edge_3d_footprint_equals_2d():
    z = Polygon([(X0 + 1000, Y0 - 5000, 5), (X0 + 3000, Y0 - 5000, 6),
                 (X0 + 3000, Y0 - 4000, 7), (X0 + 1000, Y0 - 4000, 8)])
    assert z.has_z
    pd.testing.assert_frame_equal(grid.chunk_list(z, 1), grid.chunk_list(FOOTPRINT, 1))


def test_edge_3d_footprint_in_gpkg_loads(tmp_path):
    z = Polygon([(X0 + 1000, Y0 - 5000, 5), (X0 + 3000, Y0 - 5000, 6),
                 (X0 + 3000, Y0 - 4000, 7), (X0 + 1000, Y0 - 4000, 8)])
    path = write_boundary(tmp_path / "b.gpkg", [z])
    pd.testing.assert_frame_equal(grid.chunk_list(grid.load_footprint(path), 1),
                                  grid.chunk_list(FOOTPRINT, 1))


def test_edge_empty_polygon_refused():
    with pytest.raises(ValueError):
        grid.chunk_list(Polygon(), 1)


def test_edge_collinear_ring_refused():
    line = Polygon([(X0 + 1000, Y0 - 5000), (X0 + 2000, Y0 - 5000), (X0 + 3000, Y0 - 5000)])
    with pytest.raises(ValueError):
        grid.chunk_list(line, 1)


def test_edge_self_intersecting_ring_refused():
    bowtie = Polygon([(X0 + 1000, Y0 - 5000), (X0 + 3000, Y0 - 4000),
                      (X0 + 3000, Y0 - 5000), (X0 + 1000, Y0 - 4000)])
    with pytest.raises(ValueError):
        grid.chunk_list(bowtie, 1)


def test_edge_valid_sliver_gives_zero_shares_not_an_error():
    sliver = box(X0 + 1000, Y0 - 5000, X0 + 3000, Y0 - 5000 + 1e-7)
    assert sliver.is_valid and sliver.area > 0
    df = grid.chunk_list(sliver, 1)
    assert len(df) > 0 and (df["footprint_share"] == 0.0).all()


def test_edge_null_geometry_feature_refused(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame({"a": [1]}, geometry=[None], crs="EPSG:5070").to_file(path, driver="GPKG")
    with pytest.raises(ValueError):
        grid.load_footprint(path)


def test_edge_gpkg_without_crs_refused(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame({"a": [1]}, geometry=[FOOTPRINT]).to_file(path, driver="GPKG")
    with pytest.raises(ValueError):
        grid.load_footprint(path)


@pytest.mark.parametrize("crs", [
    CRS.from_epsg(6350),                                             # NAD83(2011) Albers, 1 m shift
    CRS.from_user_input("ESRI:102003"),                              # same Albers, other latitude of origin
    CRS.from_proj4(ALBERS_5070.replace("lon_0=-96", "lon_0=-95.9")),
    CRS.from_proj4(ALBERS_5070.replace("x_0=0", "x_0=100")),
    CRS.from_proj4(ALBERS_5070.replace("units=m", "units=us-ft")),
], ids=["6350", "esri102003", "lon_0", "false_easting", "us_ft"])
def test_edge_near_5070_crs_refused(tmp_path, crs):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame({"a": [1]}, geometry=[FOOTPRINT], crs=crs).to_file(path, driver="GPKG")
    with pytest.raises(ValueError):
        grid.load_footprint(path)


def test_edge_5070_defined_by_proj_string_is_accepted(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame({"a": [1]}, geometry=[FOOTPRINT], crs=CRS.from_proj4(ALBERS_5070)).to_file(
        path, driver="GPKG")
    assert grid.load_footprint(path).equals(FOOTPRINT)


# ------------------------------------------------------------------ grid edges (D30, D44, FM-55)

@pytest.mark.parametrize("footprint", [
    box(X0 + 100, Y0 - 5000, X0 + 3000, Y0 - 4000),                  # buffer crosses the west edge
    box(X0 + 1000, Y0 - 100, X0 + 3000, Y0 - 10),                    # buffer crosses the north edge
    box(X0 - 5000, Y0 - 5000, X0 - 3000, Y0 - 4000),                 # outside the grid
    box(X0 - 500, Y0 - 5000, X0 + 3000, Y0 - 4000),                  # spans the origin column
], ids=["west", "north", "outside", "span"])
def test_edge_footprint_beyond_grid_origin_raises(footprint):
    with pytest.raises(ValueError):
        grid.chunk_list(footprint, 1)


def test_edge_buffer_reaching_exactly_the_origin_is_allowed():
    footprint = box(X0 + 150, Y0 - 5000, X0 + 3000, Y0 - 4000)       # buffer x_min == X0
    df = grid.chunk_list(footprint, 1)
    assert min(df["col"]) == 0


def test_edge_buffer_corner_point_touch_is_out():
    # The buffer arc ends at (b + 150, d) with b + 150 on a chunk edge and d on a chunk corner row.
    # The chunk to the north-east shares only that point with the buffer: area 0, not in the list.
    b = X0 + CHUNK * 7 - 150
    d = Y0 - CHUNK * 3
    footprint = box(b - 2000, d - 1500, b, d)
    assert footprint.buffer(150).bounds[2] == X0 + CHUNK * 7
    df = grid.chunk_list(footprint, 1)
    assert (2, 7) not in keys(df) and (3, 7) not in keys(df) and (4, 7) not in keys(df)
    assert (2, 6) in keys(df)


@pytest.mark.parametrize("area, low, high", [
    (46.0, 0.0, 0.0),
    (46.08, 0.0, 0.0001),                  # exactly 5e-5: either neighbour (within the 1e-4 tolerance)
    (46.2, 0.0001, 0.0001),
    (138.24, 0.0001, 0.0002),
    (9216.0, 0.01, 0.01),
])
def test_edge_footprint_share_rounding_boundaries(area, low, high):
    x0, y0 = X0 + CHUNK * 2 + 10, Y0 - CHUNK * 3 + 10
    df = grid.chunk_list(box(x0, y0, x0 + 100, y0 + area / 100), 1)
    share = df.loc[df["chunk"] == "r2_c2", "footprint_share"]
    assert len(share) == 1 and low <= share.iloc[0] <= high
    assert share.iloc[0] == round(share.iloc[0], 4)


def test_edge_full_chunk_footprint_share_is_exactly_one_and_never_more():
    df = grid.chunk_list(box(X0 + CHUNK * 2, Y0 - CHUNK * 4, X0 + CHUNK * 5, Y0 - CHUNK * 2), 1)
    assert df["footprint_share"].max() == 1.0
    assert (df["footprint_share"] <= 1.0).all() and (df["footprint_share"] >= 0.0).all()
    assert (df["footprint_share"] == 1.0).sum() == 6


def test_edge_sort_and_idempotence():
    first, second = good_frame(), good_frame()
    pd.testing.assert_frame_equal(first, second)
    order = list(zip(first["row"], first["col"]))
    assert order == sorted(order) and len(set(order)) == len(order)
    assert list(first.index) == list(range(len(first)))


def test_edge_pid_above_int64_refused():
    try:
        df = grid.chunk_list(FOOTPRINT, 2 ** 63)
    except (ValueError, OverflowError):
        return
    assert (df["pid"] > 0).all()


def test_edge_pid_int64_max_is_kept():
    df = grid.chunk_list(FOOTPRINT, 2 ** 63 - 1)
    assert df["pid"].dtype == "int64" and (df["pid"] == 2 ** 63 - 1).all()


@pytest.mark.parametrize("pid", [0, -1, True, "77", 77.0, None])
def test_edge_pid_not_a_positive_int_refused(pid):
    with pytest.raises((ValueError, TypeError)):
        grid.chunk_list(FOOTPRINT, pid)


def test_edge_sparse_far_apart_parts_do_not_scan_the_bounding_box(monkeypatch):
    class Budget(Exception):
        pass

    real, calls = standard.chunk_bounds, []

    def counted(row, col):
        calls.append(1)
        if len(calls) > 50_000:
            raise Budget
        return real(row, col)

    monkeypatch.setattr(standard, "chunk_bounds", counted)
    near = box(X0 + 10_000, Y0 - 11_000, X0 + 11_000, Y0 - 10_000)
    far = box(X0 + 2_000_000, Y0 - 2_001_000, X0 + 2_001_000, Y0 - 2_000_000)   # 2000 km away
    try:
        df = grid.chunk_list(MultiPolygon([near, far]), 1)
    except Budget:
        pytest.fail("chunk_list called chunk_bounds more than 50 000 times for about 20 chunks")
    except ValueError:
        return                                                         # a clear refusal is also fine
    assert len(df) < 100


# ------------------------------------------------------------------ Parquet schema (FM-58)

def test_edge_parquet_numeric_types_and_pandas_round_trip(tmp_path):
    frame = good_frame()
    _, path = write_target(tmp_path, frame)
    schema = pq.read_schema(path)
    assert [str(schema.field(n).type) for n in ["row", "col", "pid"]] == ["int64"] * 3
    assert [str(schema.field(n).type) for n in ["xmin", "ymin", "xmax", "ymax", "footprint_share"]] == ["double"] * 5
    back = pd.read_parquet(path)
    assert list(back.columns) == grid.COLUMNS
    for name in ["row", "col", "pid"]:
        assert back[name].dtype == "int64"
    pd.testing.assert_frame_equal(back, frame)


def test_edge_parquet_chunk_column_is_string_not_large_string(tmp_path):
    _, path = write_target(tmp_path, good_frame())
    assert pq.read_schema(path).field("chunk").type == pa.string()


def test_edge_parquet_metadata_has_every_key_with_a_value(tmp_path):
    _, path = write_target(tmp_path, good_frame())
    meta = pq.read_schema(path).metadata
    for key in standard.METADATA_KEYS:
        assert meta[key.encode()].decode().strip()
    assert meta[b"data_standard"] == b"3" and meta[b"naip_year"] == b"2022"
    standard.check_run_id(meta[b"run_id"].decode())
    assert meta[b"sources"] == b"stack/5_X/boundary.gpkg"
    assert b"pandas" in meta                                           # pandas schema metadata kept


def test_edge_write_twice_gives_same_table(tmp_path):
    roots = make_roots(tmp_path)
    frame = good_frame()
    path = grid.write_chunk_list(frame, roots, "5_X")
    first = pq.read_table(path)
    assert grid.write_chunk_list(frame, roots, "5_X") == path
    assert pq.read_table(path).equals(first)
    assert sorted(p.name for p in path.parent.iterdir()) == ["chunks.parquet"]


def test_edge_write_refuses_int32_row(tmp_path):
    with pytest.raises(ValueError):
        write_target(tmp_path, good_frame().astype({"row": "int32"}))


def test_edge_write_refuses_nan_share(tmp_path):
    with pytest.raises(ValueError):
        write_target(tmp_path, good_frame().assign(footprint_share=np.nan))


def test_edge_write_refuses_empty_list(tmp_path):
    with pytest.raises(ValueError):
        write_target(tmp_path, good_frame().iloc[0:0])


def test_edge_write_refuses_duplicate_chunk(tmp_path):
    frame = good_frame()
    with pytest.raises(ValueError):
        write_target(tmp_path, pd.concat([frame, frame], ignore_index=True))


def test_edge_write_refuses_unsorted_rows(tmp_path):
    with pytest.raises(ValueError):
        write_target(tmp_path, good_frame().iloc[::-1].reset_index(drop=True))


def test_edge_write_refuses_pid_that_differs_from_folder(tmp_path):
    with pytest.raises(ValueError):
        write_target(tmp_path, good_frame(5), "392_RAYSTOWN_LAKE")


def test_edge_write_refuses_folder_that_is_not_ds6(tmp_path):
    roots = make_roots(tmp_path)
    with pytest.raises(ValueError):
        grid.write_chunk_list(good_frame(), roots, "foo")


@pytest.mark.parametrize("folder", ["..", "a/b", "a\\b", "", "AUX", "5_X\x00"])
def test_edge_write_refuses_hostile_folder_and_writes_nothing(tmp_path, folder):
    roots = make_roots(tmp_path)
    with pytest.raises(ValueError):
        grid.write_chunk_list(good_frame(), roots, folder)
    assert list(roots["semseg"].iterdir()) == [] and list(tmp_path.glob("*.parquet")) == []


# ------------------------------------------------------------------ compare_chunks (FM-59)

def ref_frame():
    return pd.DataFrame({"chunk": ["r1_c1", "r1_c2"], "row": [1, 1], "col": [1, 2],
                         "xmin": [0.0, 960.0], "ymin": [0.0, 0.0], "xmax": [960.0, 1920.0],
                         "ymax": [960.0, 960.0], "footprint_share": [0.5, 0.25]})


def test_edge_compare_bound_noise_of_1e_9_is_reported_as_moved():
    # Bounds lie on integer metres (DS-3): an exact compare is right; noise must not pass silently.
    df = ref_frame()
    df.loc[0, "xmin"] += 1e-9
    assert grid.compare_chunks(df, ref_frame()) == {"extra": [], "missing": [], "moved": ["r1_c1"]}


def test_edge_compare_nan_bound_is_moved():
    df = ref_frame()
    df.loc[1, "ymax"] = np.nan
    assert grid.compare_chunks(df, ref_frame())["moved"] == ["r1_c2"]


@pytest.mark.parametrize("delta, moved", [(0.0001, False), (0.00011, True), (-0.0001, False), (0.0002, True)])
def test_edge_compare_share_tolerance_boundary(delta, moved):
    df = ref_frame()
    df.loc[0, "footprint_share"] = 0.5 + delta
    assert (grid.compare_chunks(df, ref_frame())["moved"] == ["r1_c1"]) is moved


def test_edge_compare_nan_share_is_moved():
    df = ref_frame()
    df.loc[0, "footprint_share"] = np.nan
    assert grid.compare_chunks(df, ref_frame())["moved"] == ["r1_c1"]


def test_edge_compare_nan_share_in_reference_is_moved():
    ref = ref_frame()
    ref.loc[1, "footprint_share"] = np.nan
    assert grid.compare_chunks(ref_frame(), ref)["moved"] == ["r1_c2"]


def test_edge_compare_order_of_rows_does_not_matter():
    assert grid.compare_chunks(ref_frame().iloc[::-1].reset_index(drop=True), ref_frame()) == {
        "extra": [], "missing": [], "moved": []}


def test_edge_compare_chunk_id_with_blank_is_extra_and_missing():
    df = ref_frame()
    df.loc[0, "chunk"] = "r1_c1 "
    result = grid.compare_chunks(df, ref_frame())
    assert result["extra"] == ["r1_c1 "] and result["missing"] == ["r1_c1"]


# ------------------------------------------------------------------ read_fg_chunks (FM-57)

def test_edge_read_fg_chunks_refuses_dot_dot(tmp_path):
    roots = make_roots(tmp_path)
    (roots["fgt"].parent / "chunks.csv").write_text("chunk\nr1_c1\n")
    with pytest.raises((ValueError, SystemExit)) as info:
        grid.read_fg_chunks("..", roots)
    assert not isinstance(info.value, SystemExit) or info.value.code != 0


# ------------------------------------------------------------------ find_folder and the CLI

def test_edge_find_folder_pid_prefix_is_not_a_match(tmp_path):
    roots = make_roots(tmp_path)
    (roots["stack"] / "77_A").mkdir()
    (roots["stack"] / "8_X").write_text("a file, not a folder")
    with pytest.raises(FileNotFoundError):
        grid.find_folder(7, roots)
    with pytest.raises(FileNotFoundError):
        grid.find_folder(8, roots)


@pytest.mark.parametrize("pid", [0, -1])
def test_edge_find_folder_pid_zero_or_negative_finds_nothing(tmp_path, pid):
    roots = make_roots(tmp_path)
    (roots["stack"] / "77_A").mkdir()
    with pytest.raises(FileNotFoundError):
        grid.find_folder(pid, roots)


def test_edge_find_folder_glob_characters_in_pid_refused(tmp_path):
    roots = make_roots(tmp_path)
    (roots["stack"] / "77_A").mkdir()
    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        grid.find_folder("*", roots)


def test_edge_find_folder_refuses_non_ds6_name(tmp_path):
    roots = make_roots(tmp_path)
    (roots["stack"] / "5_lower case").mkdir()
    with pytest.raises(ValueError):
        grid.find_folder(5, roots)


def pilot_roots(tmp_path, pids=(77, 392)):
    roots = make_roots(tmp_path)
    for pid in pids:
        write_boundary(roots["stack"] / f"{pid}_P{pid}" / "boundary.gpkg", [FOOTPRINT])
    return roots


def written(roots):
    return sorted(p.parent.name for p in roots["semseg"].glob("*/chunks.parquet"))


def test_edge_main_two_good_pids_write_both(tmp_path, capsys):
    roots = pilot_roots(tmp_path)
    assert grid.main(["77", "392"], roots) == 0
    assert written(roots) == ["392_P392", "77_P77"]
    assert pd.read_parquet(roots["semseg"] / "77_P77" / "chunks.parquet")["pid"].eq(77).all()


def test_edge_main_same_pid_twice_is_idempotent(tmp_path):
    roots = pilot_roots(tmp_path, (77,))
    assert grid.main(["77", "77"], roots) == 0
    assert written(roots) == ["77_P77"]


def test_edge_main_bad_first_pid_writes_nothing(tmp_path):
    roots = pilot_roots(tmp_path)
    with pytest.raises(FileNotFoundError):
        grid.main(["999", "77"], roots)
    assert written(roots) == []


@pytest.mark.parametrize("argv", [["77", "abc"], ["77", "0"], ["77", "-3"], ["77", "1.5"], ["77", ""]])
def test_edge_main_bad_argument_among_good_exits_2_and_writes_nothing(tmp_path, argv):
    roots = pilot_roots(tmp_path)
    with pytest.raises(SystemExit) as info:
        grid.main(argv, roots)
    assert info.value.code == 2 and written(roots) == []


def test_edge_main_never_writes_outside_semseg_root(tmp_path):
    roots = pilot_roots(tmp_path)
    before = sorted(p for p in roots["stack"].rglob("*"))
    grid.main(["77"], roots)
    assert sorted(p for p in roots["stack"].rglob("*")) == before
    assert list(roots["fgt"].rglob("*")) == []


@pytest.mark.parametrize("bad", ["missing_folder", "wrong_crs"])
def test_edge_main_one_bad_pid_among_good_writes_nothing(tmp_path, bad):
    roots = pilot_roots(tmp_path, (77, 392))
    if bad == "wrong_crs":
        write_boundary(roots["stack"] / "392_P392" / "boundary.gpkg", [FOOTPRINT], crs="EPSG:4326")
        argv = ["77", "392"]
    else:
        argv = ["77", "999", "392"]
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        grid.main(argv, roots)
    assert written(roots) == []


# ------------------------------------------------------------------ metadata (DS-21)

def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


def fake_module(monkeypatch, repo):
    package = repo / "semseg"
    package.mkdir(exist_ok=True)
    (package / "grid.py").write_text("x = 1\n")
    monkeypatch.setattr(grid, "__file__", str(package / "grid.py"))
    return package / "grid.py"


def test_edge_code_version_in_repo_without_commit_raises(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    fake_module(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        grid._code_version()


def test_edge_code_version_outside_a_repository_raises(tmp_path, monkeypatch):
    fake_module(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        grid._code_version()


def test_edge_code_version_in_clean_repo_is_the_commit(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    path = fake_module(monkeypatch, tmp_path)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "x")
    assert grid._code_version() == git(tmp_path, "rev-parse", "HEAD") != path.name


def test_edge_code_version_marks_a_dirty_tree(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    path = fake_module(monkeypatch, tmp_path)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "x")
    path.write_text("x = 2\n")
    assert grid._code_version() == git(tmp_path, "rev-parse", "HEAD") + "-dirty"


def test_edge_metadata_year_boundary_and_leap_day():
    for when, run in [(datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc), "20261231-2359-chunk_grid"),
                      (datetime(2028, 2, 29, 0, 0, tzinfo=timezone.utc), "20280229-0000-chunk_grid")]:
        meta = grid._metadata("5_X", when)
        assert meta["run_id"] == run
        assert meta["created"] == when.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert all(meta.values())


def test_edge_metadata_converts_non_utc_time_to_utc():
    when = datetime(2026, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=5)))
    meta = grid._metadata("5_X", when)
    assert meta["created"] == "2026-12-31T18:30:00Z"
    assert meta["run_id"] == "20261231-1830-chunk_grid"


# ------------------------------------------------------------------ fix round (D45)

def test_edge_metadata_naive_time_refused():
    with pytest.raises(ValueError):
        grid._metadata("5_X", datetime(2026, 1, 1, 0, 0))


def test_edge_pid_range_message():
    with pytest.raises(ValueError, match="9223372036854775807"):
        grid.chunk_list(FOOTPRINT, 2 ** 63)


def test_edge_too_many_candidates_refused():
    big = box(X0 + 1000, Y0 - 400_000, X0 + 400_000, Y0 - 1000)
    with pytest.raises(ValueError, match="candidate"):
        grid.chunk_list(big, 1)


def test_edge_write_refuses_non_string_chunk(tmp_path):
    frame = good_frame()
    frame["chunk"] = list(range(len(frame)))
    with pytest.raises(ValueError):
        write_target(tmp_path, frame)


def test_edge_write_refuses_inf_bound(tmp_path):
    frame = good_frame()
    frame.loc[0, "xmax"] = np.inf
    with pytest.raises(ValueError):
        write_target(tmp_path, frame)


@pytest.mark.parametrize("column,value", [("chunk", "r9_c9"), ("xmin", 1.0)])
def test_edge_write_refuses_chunk_id_or_box_that_differs_from_row_col(tmp_path, column, value):
    frame = good_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        write_target(tmp_path, frame)


def test_edge_write_validation_writes_nothing(tmp_path):
    roots = make_roots(tmp_path)
    frame = good_frame()
    with pytest.raises(ValueError):
        grid.write_chunk_list(frame, roots, "6_X")
    assert written(roots) == []


def test_edge_find_folder_non_int_pid_refused(tmp_path):
    roots = make_roots(tmp_path)
    for pid in [True, 5.0, None]:
        with pytest.raises(ValueError):
            grid.find_folder(pid, roots)


def test_edge_read_fg_chunks_refuses_absolute_and_symlink_escape(tmp_path):
    roots = make_roots(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "chunks.csv").write_text("chunk\nr1_c1\n")
    (roots["fgt"] / "link").symlink_to(outside)
    for folder in [str(outside), "link", "a/b"]:
        with pytest.raises(ValueError):
            grid.read_fg_chunks(folder, roots)


def test_edge_main_writes_nothing_when_git_fails(tmp_path, monkeypatch):
    roots = pilot_roots(tmp_path)

    def fail():
        raise RuntimeError("git")
    monkeypatch.setattr(grid, "_code_version", fail)
    with pytest.raises(RuntimeError):
        grid.main(["77", "392"], roots)
    assert written(roots) == []
