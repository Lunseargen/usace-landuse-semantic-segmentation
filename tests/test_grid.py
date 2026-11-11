"""Tests of task 1.1 Chunk grid (`semseg.grid`): R2, FM-54 to FM-59, D43, D44.

Real-data tests read the stack `boundary.gpkg` and the fine_grid `chunks.csv` (read only).
They never skip: a missing share stops with exit code 3 (FM-29, FM-57).
Every write test uses fake roots under `tmp_path`. No test writes to the real share.
"""

import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from semseg import grid
from semseg import standard
from semseg import storage

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["chunk", "row", "col", "xmin", "ymin", "xmax", "ymax", "footprint_share", "pid"]
PILOTS = [(77, "77_GRAPEVINE_LAKE", 157, 12), (392, "392_RAYSTOWN_LAKE", 227, 13)]
X0, Y0, CHUNK = standard.X0, standard.Y0, standard.CHUNK


def make_roots(base):
    roots = {
        "share": base / "share",
        "semseg": base / "share" / "semseg",
        "fgt": base / "share" / "fgt",
        "stack": base / "stack",
    }
    for path in roots.values():
        path.mkdir(parents=True)
    return roots


def write_boundary(path, geometries, crs="EPSG:5070"):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame(
        {"pid": list(range(len(geometries))), "name": ["T"] * len(geometries)},
        geometry=list(geometries), crs=crs)
    frame.to_file(path, driver="GPKG")
    return path


def edge_footprint(extra=0.0):
    """A box in row 3, cols 5 to 6. Its 150 m buffer reaches x = xmin of col 7 minus `extra`."""
    x_min = X0 + CHUNK * 5 + 100
    x_max = X0 + CHUNK * 7 - 150 + extra
    return box(x_min, Y0 - 3700, x_max, Y0 - 3000)


def rc(frame):
    return set(zip(frame["row"].tolist(), frame["col"].tolist()))


def reference(frame):
    return frame[["chunk", "row", "col", "xmin", "ymin", "xmax", "ymax", "footprint_share"]].copy()


# ------------------------------------------------------------------ R2, FM-56, FM-59 (real data)

@pytest.mark.parametrize("pid, folder, count, zeros", PILOTS)
def test_chunks_match_fine_grid(pid, folder, count, zeros):
    roots = storage.load_roots()
    storage.guard_share(roots)                                  # exit 3, never a skip
    footprint = grid.load_footprint(roots["stack"] / folder / "boundary.gpkg")
    df = grid.chunk_list(footprint, pid)
    ref = grid.read_fg_chunks(folder, roots)
    assert len(ref) == count and len(df) == count               # FM-59: the counts
    assert list(df.columns) == COLUMNS                          # D43
    assert df["chunk"].is_unique and ref["chunk"].is_unique
    assert set(df["chunk"]) == set(ref["chunk"])                # R2: 0 different IDs
    assert (df["pid"] == pid).all()
    keys = list(zip(df["row"], df["col"]))
    assert keys == sorted(keys)                                 # sorted by row, then col
    for row in df.itertuples():                                 # FM-56: only semseg.standard
        assert row.chunk == standard.chunk_id(row.row, row.col)
        assert (row.xmin, row.ymin, row.xmax, row.ymax) == standard.chunk_bounds(row.row, row.col)
    merged = df.merge(ref, on="chunk", suffixes=("", "_fg"), validate="one_to_one")
    for name in ("row", "col", "xmin", "ymin", "xmax", "ymax"):
        assert (merged[name] == merged[name + "_fg"]).all(), name
    assert (merged["footprint_share"] - merged["footprint_share_fg"]).abs().max() <= 1e-4 + 1e-12
    assert int((df["footprint_share"] == 0).sum()) == zeros     # the buffer-only chunks
    assert df["footprint_share"].between(0, 1).all()
    assert grid.compare_chunks(df, ref) == {"extra": [], "missing": [], "moved": []}


@pytest.mark.parametrize("pid, folder, count, zeros", PILOTS)
def test_real_boundary_loads_as_one_polygon(pid, folder, count, zeros):
    roots = storage.load_roots()
    storage.guard_share(roots)
    footprint = grid.load_footprint(roots["stack"] / folder / "boundary.gpkg")
    assert footprint.is_valid and footprint.area > 1e7


# ------------------------------------------------------------------ FM-57

def test_edge_chunks_csv_missing_exit_3(tmp_path, capsys):
    roots = make_roots(tmp_path)
    (roots["fgt"] / "77_T").mkdir()
    with pytest.raises(SystemExit) as exc:
        grid.read_fg_chunks("77_T", roots)
    assert exc.value.code == 3 == standard.EXIT_SHARE_MISSING
    assert "chunks.csv" in capsys.readouterr().err


def test_edge_share_missing_exit_3_for_chunks_csv(tmp_path):
    roots = {"share": tmp_path / "nope", "semseg": tmp_path / "nope" / "s",
             "fgt": tmp_path / "nope" / "f", "stack": tmp_path / "stack"}
    with pytest.raises(SystemExit) as exc:
        grid.read_fg_chunks("77_T", roots)
    assert exc.value.code == 3
    assert not (tmp_path / "nope").exists()


def test_read_fg_chunks_reads_csv(tmp_path):
    roots = make_roots(tmp_path)
    folder = roots["fgt"] / "77_T"
    folder.mkdir()
    (folder / "chunks.csv").write_text(
        "chunk,row,col,xmin,ymin,xmax,ymax,footprint_share\n"
        "r1_c2,1,2,-1.0,2.0,3.0,4.0,0.5\n", encoding="utf-8")
    ref = grid.read_fg_chunks("77_T", roots)
    assert list(ref.columns) == COLUMNS[:-1]
    assert ref.loc[0, "chunk"] == "r1_c2" and ref.loc[0, "footprint_share"] == 0.5


# ------------------------------------------------------------------ FM-54

def test_load_footprint_ok(tmp_path):
    poly = box(X0 + 5000, Y0 - 5000, X0 + 6000, Y0 - 4000)
    path = write_boundary(tmp_path / "boundary.gpkg", [poly])
    got = grid.load_footprint(path)
    assert got.is_valid and got.equals(poly)


def test_edge_boundary_wrong_crs_refused(tmp_path):
    poly = box(-100, 30, -99, 31)
    path = write_boundary(tmp_path / "boundary.gpkg", [poly], crs="EPSG:4326")
    with pytest.raises(ValueError, match="5070"):
        grid.load_footprint(path)


def test_edge_boundary_two_features_refused(tmp_path):
    polys = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    path = write_boundary(tmp_path / "boundary.gpkg", polys)
    with pytest.raises(ValueError):
        grid.load_footprint(path)


def test_edge_boundary_no_feature_refused(tmp_path):
    path = write_boundary(tmp_path / "boundary.gpkg", [])
    with pytest.raises(ValueError):
        grid.load_footprint(path)


def test_edge_boundary_invalid_geometry_refused(tmp_path):
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
    assert not bowtie.is_valid
    path = write_boundary(tmp_path / "boundary.gpkg", [bowtie])
    with pytest.raises(ValueError):
        grid.load_footprint(path)


def test_edge_boundary_not_a_polygon_refused(tmp_path):
    from shapely.geometry import Point
    path = write_boundary(tmp_path / "boundary.gpkg", [Point(1, 1)])
    with pytest.raises(ValueError):
        grid.load_footprint(path)


def test_edge_boundary_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        grid.load_footprint(tmp_path / "nope.gpkg")


# ------------------------------------------------------------------ FM-55, D44 (synthetic)

def test_edge_chunk_touching_buffer_edge():
    df = grid.chunk_list(edge_footprint(), 5)
    assert (3, 7) not in rc(df)                                 # touch-only: out
    assert rc(df) == {(3, 4), (3, 5), (3, 6), (2, 5), (2, 6), (4, 5), (4, 6)}
    assert df["footprint_share"].between(0, 1).all()


def test_edge_chunk_with_tiny_overlap_is_in():
    df = grid.chunk_list(edge_footprint(extra=1e-3), 5)
    assert (3, 7) in rc(df)                                     # positive area: in
    assert rc(df) == {(3, 4), (3, 5), (3, 6), (3, 7), (2, 5), (2, 6), (4, 5), (4, 6)}
    assert df.set_index("chunk").loc["r3_c7", "footprint_share"] == 0.0


def test_footprint_share_half_chunk():
    x_min = X0 + CHUNK * 10
    footprint = box(x_min, Y0 - CHUNK * 10 - CHUNK, x_min + CHUNK / 2, Y0 - CHUNK * 10)
    df = grid.chunk_list(footprint, 5).set_index("chunk")
    assert df.loc["r10_c10", "footprint_share"] == 0.5
    assert df.loc["r10_c9", "footprint_share"] == 0.0           # in the buffer only
    assert df["pid"].eq(5).all()


def test_chunk_list_dtypes_and_order():
    df = grid.chunk_list(edge_footprint(), 5)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == COLUMNS
    assert df["row"].dtype == "int64" and df["col"].dtype == "int64" and df["pid"].dtype == "int64"
    for name in ("xmin", "ymin", "xmax", "ymax", "footprint_share"):
        assert df[name].dtype == "float64", name
    assert list(df.index) == list(range(len(df)))
    keys = list(zip(df["row"], df["col"]))
    assert keys == sorted(keys)
    assert df["footprint_share"].equals(df["footprint_share"].round(4))


@pytest.mark.parametrize("pid", [0, -1, "77", 7.5, None, True])
def test_edge_chunk_list_bad_pid(pid):
    with pytest.raises((ValueError, TypeError)):
        grid.chunk_list(edge_footprint(), pid)


@pytest.mark.parametrize("bad", [None, Polygon(), box(0, 0, 1, 1).boundary, "box"])
def test_edge_chunk_list_bad_footprint(bad):
    with pytest.raises((ValueError, TypeError)):
        grid.chunk_list(bad, 5)


# ------------------------------------------------------------------ FM-59 compare helper

def make_pair():
    df = grid.chunk_list(edge_footprint(), 5)
    return df, reference(df)


def test_compare_identical_is_empty():
    df, ref = make_pair()
    assert grid.compare_chunks(df, ref) == {"extra": [], "missing": [], "moved": []}


def test_edge_compare_detects_extra_chunk():
    df, ref = make_pair()
    ref = ref[ref["chunk"] != "r3_c4"]
    assert grid.compare_chunks(df, ref)["extra"] == ["r3_c4"]


def test_edge_compare_detects_missing_chunk():
    df, ref = make_pair()
    code = pd.concat([ref, ref.iloc[[0]].assign(chunk="r9_c9", row=9, col=9)], ignore_index=True)
    assert grid.compare_chunks(df, code)["missing"] == ["r9_c9"]


def test_edge_compare_detects_moved_chunk():
    df, ref = make_pair()
    ref = ref.copy()
    ref.loc[ref["chunk"] == "r3_c5", "xmin"] += 960.0
    diff = grid.compare_chunks(df, ref)
    assert diff["moved"] == ["r3_c5"] and diff["extra"] == [] and diff["missing"] == []


def test_edge_compare_detects_share_change():
    df, ref = make_pair()
    ref = ref.copy()
    ref.loc[ref["chunk"] == "r3_c5", "footprint_share"] += 0.01
    assert grid.compare_chunks(df, ref)["moved"] == ["r3_c5"]


def test_edge_compare_share_within_tolerance_is_equal():
    df, ref = make_pair()
    ref = ref.copy()
    ref["footprint_share"] += 5e-5
    assert grid.compare_chunks(df, ref)["moved"] == []


def test_edge_compare_refuses_two_empty_lists():
    df, ref = make_pair()
    with pytest.raises(ValueError):
        grid.compare_chunks(df.iloc[0:0], ref.iloc[0:0])


def test_edge_compare_refuses_duplicate_ids():
    df, ref = make_pair()
    dup = pd.concat([ref, ref.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        grid.compare_chunks(df, dup)


# ------------------------------------------------------------------ FM-58, DS-21, DS-25

def test_chunks_parquet_schema_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    roots = make_roots(tmp_path)
    df = grid.chunk_list(edge_footprint(), 77)
    path = grid.write_chunk_list(df, roots, "77_TEST")
    assert Path(path) == roots["semseg"] / "77_TEST" / standard.CHUNK_LIST
    assert path.read_bytes()[:4] == b"PAR1"
    assert not list(path.parent.glob("*.part"))
    table = pq.read_table(path)
    assert table.column_names == COLUMNS
    types = dict(zip(table.schema.names, table.schema.types))
    assert pa.types.is_string(types["chunk"]) or pa.types.is_large_string(types["chunk"])
    for name in ("row", "col", "pid"):
        assert types[name] == pa.int64(), name
    for name in ("xmin", "ymin", "xmax", "ymax", "footprint_share"):
        assert types[name] == pa.float64(), name
    meta = table.schema.metadata or {}
    for key in standard.METADATA_KEYS:                          # DS-21
        assert key.encode() in meta and meta[key.encode()], key
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    assert list(back["chunk"]) == list(df["chunk"])
    assert back["footprint_share"].tolist() == df["footprint_share"].tolist()


def test_write_chunk_list_writes_only_under_semseg(tmp_path):
    roots = make_roots(tmp_path)
    grid.write_chunk_list(grid.chunk_list(edge_footprint(), 77), roots, "77_TEST")
    assert list(roots["fgt"].rglob("*")) == [] and list(roots["stack"].rglob("*")) == []


def test_write_chunk_list_replaces_old_file(tmp_path):
    roots = make_roots(tmp_path)
    one = grid.chunk_list(edge_footprint(), 77)
    grid.write_chunk_list(one, roots, "77_TEST")
    grid.write_chunk_list(one.iloc[:3], roots, "77_TEST")
    assert len(pd.read_parquet(roots["semseg"] / "77_TEST" / "chunks.parquet")) == 3


def test_edge_write_chunk_list_share_missing_exit_3(tmp_path):
    roots = {"share": tmp_path / "nope", "semseg": tmp_path / "nope" / "s",
             "fgt": tmp_path / "nope" / "f", "stack": tmp_path / "stack"}
    with pytest.raises(SystemExit) as exc:
        grid.write_chunk_list(grid.chunk_list(edge_footprint(), 77), roots, "77_TEST")
    assert exc.value.code == 3
    assert not (tmp_path / "nope").exists()


@pytest.mark.parametrize("folder", ["", "..", "a/b", "77 TEST\n", "con"])
def test_edge_write_chunk_list_bad_folder(tmp_path, folder):
    roots = make_roots(tmp_path)
    with pytest.raises(ValueError):
        grid.write_chunk_list(grid.chunk_list(edge_footprint(), 77), roots, folder)
    assert list(roots["semseg"].rglob("*")) == []


def test_edge_write_chunk_list_wrong_columns_refused(tmp_path):
    roots = make_roots(tmp_path)
    df = grid.chunk_list(edge_footprint(), 77)
    with pytest.raises(ValueError):
        grid.write_chunk_list(df.drop(columns=["pid"]), roots, "77_TEST")
    with pytest.raises(ValueError):
        grid.write_chunk_list(df[list(reversed(COLUMNS))], roots, "77_TEST")
    assert list(roots["semseg"].rglob("*")) == []


# ------------------------------------------------------------------ folder lookup and CLI

def test_find_folder(tmp_path):
    roots = make_roots(tmp_path)
    (roots["stack"] / "77_TEST").mkdir()
    (roots["stack"] / "392_OTHER").mkdir()
    assert grid.find_folder(77, roots) == "77_TEST"
    with pytest.raises(FileNotFoundError):
        grid.find_folder(5, roots)
    (roots["stack"] / "77_TWO").mkdir()
    with pytest.raises(ValueError):
        grid.find_folder(77, roots)


def test_main_writes_chunks_parquet(tmp_path):
    roots = make_roots(tmp_path)
    write_boundary(roots["stack"] / "77_TEST" / "boundary.gpkg", [edge_footprint()])
    assert grid.main(["77"], roots=roots) == 0
    out = pd.read_parquet(roots["semseg"] / "77_TEST" / "chunks.parquet")
    assert list(out.columns) == COLUMNS and (out["pid"] == 77).all()
    assert rc(out) == rc(grid.chunk_list(edge_footprint(), 77))


def test_main_many_pids(tmp_path):
    roots = make_roots(tmp_path)
    write_boundary(roots["stack"] / "77_TEST" / "boundary.gpkg", [edge_footprint()])
    write_boundary(roots["stack"] / "392_TEST" / "boundary.gpkg", [edge_footprint()])
    assert grid.main(["77", "392"], roots=roots) == 0
    assert (roots["semseg"] / "77_TEST" / "chunks.parquet").is_file()
    assert (roots["semseg"] / "392_TEST" / "chunks.parquet").is_file()


@pytest.mark.parametrize("argv", [[], ["abc"], ["0"], ["-5"]])
def test_edge_main_bad_arguments_exit_2(tmp_path, argv):
    roots = make_roots(tmp_path)
    with pytest.raises(SystemExit) as exc:
        grid.main(argv, roots=roots)
    assert exc.value.code == 2
    assert list(roots["semseg"].rglob("*")) == []


def test_edge_main_wrong_crs_writes_nothing(tmp_path):
    roots = make_roots(tmp_path)
    write_boundary(roots["stack"] / "77_TEST" / "boundary.gpkg", [box(-100, 30, -99, 31)],
                   crs="EPSG:4326")
    with pytest.raises(ValueError):
        grid.main(["77"], roots=roots)
    assert list(roots["semseg"].rglob("*")) == []


def test_cli_module_no_arguments_prints_usage():
    done = subprocess.run([sys.executable, "-m", "semseg.grid"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 2
    assert "usage" in done.stderr.lower()
