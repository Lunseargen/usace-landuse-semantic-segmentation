"""Task 0.2: the data standard in code (R1, R3, R8; FM-1, FM-24 to FM-29).

The tests target `semseg.standard` (decisions D27 to D33, section A of
`.factory/0.2/decisions.md`). The docstring of each test starts with the DS IDs that it covers.
`test_every_ds_rule_has_test` (FM-24) checks that each DS ID of `docs/DATA_STANDARD.md` has a test.

Read only: the stack, the share, and `scheme.py` of `usace-landuse`. No test writes outside the
repository. No GPU. `scheme.py` is read as text with `ast` and never imported (D3).
"""
import ast
import csv
import datetime as dt
import math
import random
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from semseg import standard as S
from semseg.config import load_config

ROOT = Path(__file__).resolve().parent.parent

# Read-only reference for FM-25. It is a fixed path, not a data set (D3: never import it).
SCHEME_PATH = Path(
    "/home/spdc_research/workspace/usace-landuse/document/usace-landuse/code/"
    "usace_veg/src/usace_veg/scheme.py"
)

PILOT_FOLDERS = {77: "77_GRAPEVINE_LAKE", 392: "392_RAYSTOWN_LAKE"}
TOP_LEFT_CELL_ID = {77: 73346076515, 392: 41009129998}

# DS rules that have a file-level test in a later task (D31, section A7). The constants and the
# pure functions of these rules are still tested here.
OWNED_ELSEWHERE = {
    "DS-4": "1.2 test_naip_chunk_profile",
    "DS-5": "1.4",
    "DS-9": "1.2 or 2.x",
    "DS-17": "1.4",
    "DS-18": "1.2, 2.3, 5.1",
    "DS-20": "1.4 test_record_schema",
    "DS-21": "each writer task",
    "DS-23": "0.3",
    "DS-25": "0.3",
    "DS-26": "0.3",
    "DS-27": "2.1 or 2.3",
    "DS-28": "0.3",
}

CODE_TABLE = [
    # code, NLCD code?, colour (section B of decisions.md)
    (11, True, "#466b9f"),
    (12, True, "#d1def8"),
    (13, False, "#5a86b8"),
    (14, False, "#2f4f7a"),
    (21, True, "#dec5c5"),
    (25, False, "#8b0000"),
    (26, False, "#ff3b3b"),
    (27, False, "#c9846b"),
    (28, False, "#e3aa9c"),
    (29, False, "#7a3d3d"),
    (31, True, "#b3ac9f"),
    (32, False, "#e8dcb0"),
    (33, False, "#8f8a80"),
    (34, False, "#a89b84"),
    (35, False, "#c7c2b8"),
    (36, False, "#6f6a62"),
    (41, True, "#68ab5f"),
    (42, True, "#1c5f2c"),
    (43, True, "#b5c58f"),
    (44, False, "#4f8a4a"),
    (51, True, "#af963c"),
    (52, True, "#ccb879"),
    (53, False, "#d6c48a"),
    (54, False, "#8c7a3a"),
    (55, False, "#bfa65a"),
    (71, True, "#dfdfc2"),
    (81, True, "#dcd939"),
    (82, True, "#ab6c28"),
    (83, False, "#c9a27a"),
    (90, True, "#b8d9eb"),
    (95, True, "#6c9fb8"),
    (96, False, "#4f7f99"),
]
CODES_IN_ORDER = [c for c, _, _ in CODE_TABLE]

NLCD_LEGEND_EXPECTED = {
    11: "#466b9f", 12: "#d1def8", 21: "#dec5c5", 22: "#d99282", 23: "#eb0000", 24: "#ab0000",
    31: "#b3ac9f", 41: "#68ab5f", 42: "#1c5f2c", 43: "#b5c58f", 51: "#af963c", 52: "#ccb879",
    71: "#dfdfc2", 72: "#d1d182", 73: "#a3cc51", 74: "#82ba9e", 81: "#dcd939", 82: "#ab6c28",
    90: "#b8d9eb", 95: "#6c9fb8",
}
DERIVED_CODES = {0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95}
LEVEL1_EXPECTED = {
    10: "Water", 20: "Developed", 30: "Barren", 40: "Forest", 50: "Shrubland",
    70: "Herbaceous", 80: "Planted or cultivated", 99: "Wetlands",
}

# DS-14 groups: output code <- member codes
NLCD_GROUPS = {
    11: (11, 13, 14), 12: (12,), 21: (21,), 31: (31, 32, 33, 34, 35, 36), 41: (41,), 42: (42,),
    43: (43, 44), 52: (51, 52, 53, 54, 55), 71: (71,), 81: (81,), 82: (82, 83), 90: (90,), 95: (95, 96),
}

USACE_M_MAP_EXPECTED = {
    11: 1, 13: 1, 14: 1, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 12: 3, 31: 3, 35: 3, 32: 4, 34: 4,
    33: 5, 36: 6, 96: 8, 95: 9, 21: 10, 71: 10, 81: 12, 82: 13, 83: 13, 52: 26, 53: 16, 54: 15,
    55: 14, 51: 17,
}
TREE_CODES = (41, 42, 43, 44, 90)
USACE_M_LABELS_EXPECTED = {
    1: "M01", 2: "M02", 3: "M03", 4: "M04", 5: "M05", 6: "M06", 8: "M08", 9: "M09", 10: "M10",
    12: "M12", 13: "M13", 14: "M14", 15: "M15", 16: "M16", 17: "M17", 18: "M18", 19: "M19", 20: "M20", 21: "M21", 22: "M22",
    23: "M23", 24: "M24", 26: "M14-16u", 27: "M19-21u", 28: "M22-24u",
}
NOT_PRODUCED_EXPECTED = {7, 11, 25, 29}


# ---------------------------------------------------------------- helpers


def _registry_roots():
    """Return the resolved [roots] of data/REGISTRY.toml as a dict of Path (read only)."""
    data = tomllib.loads((ROOT / "data" / "REGISTRY.toml").read_text(encoding="utf-8"))
    raw = data["roots"]
    share = raw["share"]
    roots = {}
    for key, value in raw.items():
        if key in ("share", "repo"):
            continue
        roots[key] = Path(value.replace("{share}", share)).expanduser()
    roots["share"] = Path(share)
    return roots


def _need_root(name):
    """Return a registry root. A missing root is a failure, never a skip (FM-29, DS-26)."""
    root = _registry_roots()[name]
    if not root.exists():
        pytest.fail(
            f"root '{name}' does not exist: {root}. The lab share is not mounted. "
            "DS-26: the step stops with exit code 3; this test must not skip."
        )
    return root


def _codes_of(table):
    """Return the code keys of CODES or DERIVED_NLCD (a mapping or a sequence of entries)."""
    if hasattr(table, "keys"):
        return list(table.keys())
    out = []
    for entry in table:
        if isinstance(entry, int):
            out.append(entry)
        elif hasattr(entry, "code"):
            out.append(entry.code)
        elif hasattr(entry, "keys"):
            out.append(entry["code"])
        else:
            out.append(entry[0])
    return out


def _name_of(table, code):
    """Return the name of a code in CODES (mapping value or entry; attribute, key or tuple)."""
    if hasattr(table, "keys"):
        entry = table[code]
    else:
        entry = next(
            e for e in table
            if (getattr(e, "code", None) == code)
            or (hasattr(e, "keys") and e.get("code") == code)
            or (not hasattr(e, "keys") and not hasattr(e, "code") and e[0] == code)
        )
    if hasattr(entry, "name"):
        return entry.name
    if hasattr(entry, "keys"):
        return entry["name"]
    return entry[1] if entry[0] == code else entry[0]


def _shares(**kw):
    """Build a shares mapping: _shares(c41=0.5, c71=0.5)."""
    return {int(k[1:]): v for k, v in kw.items()}


def _doc_text():
    return (ROOT / "docs" / "DATA_STANDARD.md").read_text(encoding="utf-8")


def _scheme_tree():
    return ast.parse(SCHEME_PATH.read_text(encoding="utf-8"))


def _scheme_value(tree, name):
    """Return the literal value of a top-level assignment `name = <literal>` in scheme.py."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {SCHEME_PATH}")


def _scheme_value_resolved(tree, name, env):
    """Like _scheme_value, but replace the names in `env` by their constant values first."""

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(env[node.id]), node)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(Sub().visit(node.value))
    raise AssertionError(f"{name} not found in {SCHEME_PATH}")


def _open_stack_lattice_rasters():
    """Open nlcd_2022.tif of 77 and 392 (read only). Return a list of (pid, path, dataset info)."""
    import rasterio

    stack = _need_root("stack")
    opened = []
    for pid, folder in PILOT_FOLDERS.items():
        path = stack / folder / "nlcd_2022.tif"
        assert path.is_file(), f"stack raster missing (FM-29: a missing file is a failure): {path}"
        with rasterio.open(path) as ds:
            opened.append((pid, path, {
                "epsg": ds.crs.to_epsg() if ds.crs else None,
                "a": ds.transform.a,
                "b": ds.transform.b,
                "d": ds.transform.d,
                "e": ds.transform.e,
                "bounds": tuple(ds.bounds),
                "width": ds.width,
                "height": ds.height,
            }))
    return opened


# ---------------------------------------------------------------- A1: grid


def test_grid_constants():
    """DS-1, DS-2, DS-3, DS-4, DS-8: the grid constants and the nesting of pixel, cell, chunk."""
    assert S.CRS == "EPSG:5070"
    assert (S.X0, S.Y0) == (-2415585, 3314805)
    assert (S.CELL, S.CHUNK, S.CELLS_PER_CHUNK) == (30, 960, 32)
    assert S.CELL * S.CELLS_PER_CHUNK == S.CHUNK
    assert S.PIXEL == 0.5
    assert S.PIXELS_PER_CELL == 60 and S.CELL / S.PIXEL == 60
    assert S.PIXELS_PER_CHUNK == 1920 and S.CHUNK / S.PIXEL == 1920
    assert S.PIXELS_PER_CELL * S.CELLS_PER_CHUNK == S.PIXELS_PER_CHUNK
    assert S.CELL_ID_FACTOR == 1_000_000


def test_data_standard_version():
    """DS-16: the version constant equals the `Version <n>` line of docs/DATA_STANDARD.md (D31)."""
    m = re.search(r"^Version (\d+)", _doc_text(), re.MULTILINE)
    assert m, "no 'Version <n>' line in docs/DATA_STANDARD.md"
    assert S.DATA_STANDARD_VERSION == 6
    assert int(m.group(1)) == S.DATA_STANDARD_VERSION


def test_on_lattice():
    """DS-2: on_lattice is exact, with no tolerance."""
    X0, Y0 = S.X0, S.Y0
    assert S.on_lattice(X0, Y0)
    assert S.on_lattice(X0 + 30, Y0 - 60)
    assert S.on_lattice(X0 + 30 * 76515, Y0 - 30 * 73346)
    assert S.on_lattice(X0 - 30, Y0 + 30)  # a negative offset is still on the lattice
    assert S.on_lattice(float(X0 + 60), float(Y0 - 90))
    for x, y in [(X0 + 1, Y0), (X0 + 0.5, Y0), (X0, Y0 - 1), (X0, Y0 - 0.5),
                 (X0 + 29, Y0), (X0 + 1e-9, Y0), (X0, Y0 - 1e-9)]:
        assert not S.on_lattice(x, y), (x, y)


def test_grid_lattice():
    """DS-2, DS-1, R1, FM-1, I7: the real nlcd_2022.tif of 77 and 392 lie on the DS-2 lattice.

    The stack path comes from data/REGISTRY.toml [roots]. The test never skips (FM-29).
    """
    opened = _open_stack_lattice_rasters()
    assert len(opened) == 2
    assert [p for p, _, _ in opened] == [77, 392]
    for pid, path, info in opened:
        assert info["epsg"] == 5070, path
        assert info["a"] == 30 and info["e"] == -30, path
        assert info["b"] == 0 and info["d"] == 0, path
        left, bottom, right, top = info["bounds"]
        assert S.on_lattice(left, top), path
        assert S.on_lattice(right, bottom), path
        assert (left - S.X0) % 30 == 0 and (S.Y0 - top) % 30 == 0
        row, col = S.cell_of_bounds(left, top)
        assert S.cell_id(row, col) == TOP_LEFT_CELL_ID[pid], path
        # the raster size in cells agrees with its bounds
        assert S.cell_of_bounds(right - 30, bottom + 30) == (
            row + info["height"] - 1, col + info["width"] - 1)


def test_grid_lattice_not_skipped(monkeypatch):
    """DS-2, R1, FM-29: both rasters are opened, and the module has no way to skip a test."""
    opened = _open_stack_lattice_rasters()
    assert sorted(p.parent.name for _, p, _ in opened) == sorted(PILOT_FOLDERS.values())
    assert all(p.name == "nlcd_2022.tif" for _, p, _ in opened)
    source = Path(__file__).read_text(encoding="utf-8")
    for word in ("skip" + "if", "mark." + "skip", "pytest." + "skip(", "import" + "orskip",
                 "xf" + "ail"):
        assert word not in source, f"{word} found in tests/test_standard.py (FM-29)"
    # A missing root is a failure with the exit-3 message, not a skip.
    real = _registry_roots

    def fake():
        roots = real()
        roots["stack"] = Path("/nonexistent/stack/root")
        return roots

    monkeypatch.setattr(sys.modules[__name__], "_registry_roots", fake)
    with pytest.raises(pytest.fail.Exception, match="exit code 3"):
        _need_root("stack")


def test_cell_of_bounds_and_bounds():
    """DS-8, DS-2: cell_of_bounds, cell_bounds and cell_id on known cells."""
    assert S.cell_of_bounds(S.X0, S.Y0) == (0, 0)
    assert S.cell_of_bounds(-120135.0, 1114425.0) == (73346, 76515)
    assert S.cell_of_bounds(1484355, 2084535) == (41009, 129998)
    row, col = S.cell_of_bounds(-120135.0, 1114425.0)
    assert isinstance(row, int) and isinstance(col, int)
    assert S.cell_bounds(0, 0) == (S.X0, S.Y0 - 30, S.X0 + 30, S.Y0)
    assert S.cell_bounds(73346, 76515) == (-120135, 1114395, -120105, 1114425)
    assert S.cell_id(0, 0) == 0
    assert S.cell_id(1, 0) == 1_000_000
    assert S.cell_id(73346, 76515) == 73346076515
    assert S.cell_id(41009, 129998) == 41009129998
    assert S.cell_id(0, 999_999) == 999_999
    assert S.cell_rowcol(73346076515) == (73346, 76515)
    assert S.cell_rowcol(0) == (0, 0)


@pytest.mark.parametrize("row,col", [(-1, 0), (0, -1), (0, 1_000_000), (0, 2_000_000)])
def test_cell_id_rejects_out_of_range(row, col):
    """DS-8: cell_id needs row >= 0 and 0 <= col < 1 000 000."""
    with pytest.raises(ValueError):
        S.cell_id(row, col)


@pytest.mark.parametrize("row,col", [(1.0, 2), (1, 2.0), ("1", 2), (None, 2), (1.5, 2)])
def test_cell_id_rejects_non_integers(row, col):
    """DS-8: cell_id requires integers."""
    with pytest.raises((ValueError, TypeError)):
        S.cell_id(row, col)


def test_cell_rowcol_negative_raises():
    """DS-8: a negative cell ID raises."""
    with pytest.raises(ValueError):
        S.cell_rowcol(-1)


def test_cell_id_roundtrip():
    """DS-8, FM-26, R1: cell ID -> bounds -> cell ID is the identity (corner, edge, random cells)."""
    rng = random.Random(20261111)
    cells = [(0, 0), (0, 999_999), (73346, 76515), (41009, 129998), (2295 * 32, 2394 * 32),
             (2295 * 32 + 31, 2394 * 32 + 31), (1, 0), (0, 1), (200_000, 999_999)]
    cells += [(rng.randrange(0, 200_000), rng.randrange(0, 1_000_000)) for _ in range(200)]
    for row, col in cells:
        cid = S.cell_id(row, col)
        assert S.cell_rowcol(cid) == (row, col)
        xmin, ymin, xmax, ymax = S.cell_bounds(*S.cell_rowcol(cid))
        assert (xmax - xmin, ymax - ymin) == (30, 30)
        assert S.cell_of_bounds(xmin, ymax) == (row, col)
        assert S.cell_of_bounds(float(xmin), float(ymax)) == (row, col)
        assert S.cell_id(*S.cell_of_bounds(xmin, ymax)) == cid
        assert S.cell_of_point((xmin + xmax) / 2, (ymin + ymax) / 2) == (row, col)


def test_cell_of_point_edges():
    """DS-8, FM-26: a cell includes its left edge and its top edge; floor, not round."""
    X0, Y0 = S.X0, S.Y0
    assert S.cell_of_point(X0, Y0) == (0, 0)
    assert S.cell_of_point(X0 + 30, Y0) == (0, 1)
    assert S.cell_of_point(X0, Y0 - 30) == (1, 0)
    assert S.cell_of_point(X0 + 29.999, Y0 - 0.001) == (0, 0)
    assert S.cell_of_point(X0 + 30.5, Y0 - 59.5) == (1, 1)
    xmin, ymax = -120135, 1114425
    assert S.cell_of_point(xmin, ymax) == (73346, 76515)
    assert S.cell_of_point(xmin + 30, ymax) == (73346, 76516)
    assert S.cell_of_point(xmin, ymax - 30) == (73347, 76515)


@pytest.mark.parametrize("x,y", [
    (-2415585 - 0.001, 3314805), (-2415585, 3314805 + 0.001), (-3000000, 3314805),
    (-2415585, 4000000),
])
def test_cell_of_point_negative_raises(x, y):
    """DS-8, DS-3: a point above or left of the grid (negative row or column) raises."""
    with pytest.raises(ValueError):
        S.cell_of_point(x, y)


def test_off_lattice_raises():
    """DS-8, DS-2, FM-28, R1: offsets of 1 m and 0.5 m in x and in y raise ValueError."""
    X0, Y0 = S.X0, S.Y0
    for xmin, ymax in [(X0 + 1, Y0), (X0 + 0.5, Y0), (X0, Y0 - 1), (X0, Y0 - 0.5),
                       (X0 + 30 + 1, Y0 - 30), (X0 + 30, Y0 - 30 - 0.5),
                       (-120135 + 1, 1114425), (-120135 + 0.5, 1114425),
                       (-120135, 1114425 - 1), (-120135, 1114425 - 0.5), (X0 + 29.999, Y0)]:
        with pytest.raises(ValueError):
            S.cell_of_bounds(xmin, ymax)
    # a bound above or left of the origin (negative row or column) raises
    with pytest.raises(ValueError):
        S.cell_of_bounds(X0 - 30, Y0)
    with pytest.raises(ValueError):
        S.cell_of_bounds(X0, Y0 + 30)
    # the chunk function also needs the 30 m lattice
    for xmin, ymax in [(X0 + 1, Y0), (X0 + 0.5, Y0), (X0, Y0 - 1), (X0, Y0 - 0.5)]:
        with pytest.raises(ValueError):
            S.chunk_of_bounds(xmin, ymax)
    # non-numbers raise too
    with pytest.raises((ValueError, TypeError)):
        S.cell_of_bounds("a", Y0)
    with pytest.raises((ValueError, TypeError)):
        S.cell_of_bounds(float("nan"), Y0)


def test_chunk_functions():
    """DS-3, DS-7, FM-26: chunk row and column, bounds, the chunk of a cell, and the chunk ID."""
    assert S.chunk_of_bounds(-117345, 1111605) == (2295, 2394)
    assert S.chunk_of_bounds(1505055, 2082165) == (1284, 4084)
    assert S.chunk_of_bounds(S.X0, S.Y0) == (0, 0)
    # a 30 m aligned corner inside a chunk: floor, not round
    assert S.chunk_of_bounds(S.X0 + 30 * 33, S.Y0 - 30 * 65) == (2, 1)
    assert S.chunk_of_bounds(S.X0 + 30 * 31, S.Y0 - 30 * 31) == (0, 0)
    assert S.chunk_bounds(2295, 2394) == (-117345, 1110645, -116385, 1111605)
    assert S.chunk_bounds(0, 0) == (S.X0, S.Y0 - 960, S.X0 + 960, S.Y0)
    assert S.chunk_of_cell(73346, 76515) == (2292, 2391)  # 73346 / 32 = 2292.06
    assert S.chunk_of_cell(2295 * 32, 2394 * 32) == (2295, 2394)
    assert S.chunk_of_cell(2295 * 32 + 31, 2394 * 32 + 31) == (2295, 2394)
    assert S.chunk_of_cell(2295 * 32 - 1, 2394 * 32 - 1) == (2294, 2393)
    assert S.chunk_id(2295, 2394) == "r2295_c2394"
    assert S.chunk_id(0, 0) == "r0_c0"
    assert re.match(r"^r\d+_c\d+$", S.chunk_id(1284, 4084))
    for row, col in [(0, 0), (2295, 2394), (1284, 4084), (10, 0)]:
        assert S.parse_chunk_id(S.chunk_id(row, col)) == (row, col)
    assert S.parse_chunk_id("r0_c0") == (0, 0)


def test_chunk_rejects_negative():
    """DS-3: a chunk row or column below 0 is outside the grid and is an error."""
    with pytest.raises(ValueError):
        S.chunk_of_bounds(S.X0 - 30, S.Y0)
    with pytest.raises(ValueError):
        S.chunk_of_bounds(S.X0, S.Y0 + 30)


@pytest.mark.parametrize("text", [
    "r02_c3", "r1_c01", "x1_c2", "r1_c", "r_c1", "r-1_c2", "", "r1c2", "r1_c2_", " r1_c2",
    "R1_C2", "r1.0_c2",
])
def test_parse_chunk_id_invalid(text):
    """DS-7: a chunk ID matches ^r(\\d+)_c(\\d+)$ with no leading zero."""
    with pytest.raises(ValueError):
        S.parse_chunk_id(text)


def test_chunk_ids_match_chunks_csv():
    """DS-3, DS-7, FM-26, R1: the chunk IDs of the code agree with chunks.csv of 77 and 392.

    A missing share is a failure with the exit-3 message, not a skip.
    """
    fgt = _need_root("fgt")
    expected_rows = {77: 157, 392: 227}
    sample = {77: ("r2295_c2394", -117345.0, 1111605.0), 392: ("r1284_c4084", 1505055.0, 2082165.0)}
    for pid, folder in PILOT_FOLDERS.items():
        path = fgt / folder / "chunks.csv"
        assert path.is_file(), path
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == expected_rows[pid]
        seen = {}
        for r in rows:
            xmin, ymax = float(r["xmin"]), float(r["ymax"])
            row, col = int(r["row"]), int(r["col"])
            assert S.chunk_of_bounds(xmin, ymax) == (row, col), r["chunk"]
            assert S.chunk_id(row, col) == r["chunk"]
            assert S.parse_chunk_id(r["chunk"]) == (row, col)
            assert S.chunk_bounds(row, col) == (xmin, float(r["ymin"]), float(r["xmax"]), ymax)
            seen[r["chunk"]] = (xmin, ymax)
        name, sx, sy = sample[pid]
        assert seen[name] == (sx, sy)
        assert S.chunk_of_bounds(sx, sy) == S.parse_chunk_id(name)


def test_pixel_functions():
    """DS-4: the pixel of a point, the cell of a pixel, and the bounds of the pixel grid."""
    row, col = 2295, 2394
    xmin, ymax = -117345, 1111605
    assert S.pixel_of_point(xmin, ymax, row, col) == (0, 0)
    assert S.pixel_of_point(xmin + 0.5, ymax - 0.5, row, col) == (1, 1)
    assert S.pixel_of_point(xmin + 0.4999, ymax - 0.4999, row, col) == (0, 0)
    assert S.pixel_of_point(xmin + 30, ymax - 30, row, col) == (60, 60)
    assert S.pixel_of_point(xmin + 959.75, ymax - 959.75, row, col) == (1919, 1919)
    i, j = S.pixel_of_point(xmin + 100.25, ymax - 200.75, row, col)
    assert (i, j) == (401, 200)
    assert (i // 60, j // 60) == (6, 3)


@pytest.mark.parametrize("dx,dy", [(960, 0), (0, -960), (-0.5, 0), (0, 0.5), (1000, -1000)])
def test_pixel_of_point_outside_chunk(dx, dy):
    """DS-4: a point outside the chunk raises ValueError."""
    with pytest.raises(ValueError):
        S.pixel_of_point(-117345 + dx, 1111605 + dy, 2295, 2394)


def test_check_naip_transform():
    """DS-4: the NAIP chunk transform has a 0.5 m pixel and starts on a cell edge."""
    xmin, ymax = -117345, 1111605
    good = (0.5, 0, xmin, 0, -0.5, ymax)
    S.check_naip_transform(good, 1920, 1920)
    bad_transforms = [
        (1.0, 0, xmin, 0, -1.0, ymax),
        (0.5, 0, xmin, 0, 0.5, ymax),
        (0.5, 0.1, xmin, 0, -0.5, ymax),
        (0.5, 0, xmin, 0.1, -0.5, ymax),
        (0.5, 0, xmin + 0.5, 0, -0.5, ymax),
        (0.5, 0, xmin + 1, 0, -0.5, ymax),
        (0.5, 0, xmin, 0, -0.5, ymax - 0.5),
        (0.5, 0, xmin, 0, -0.5, ymax - 1),
    ]
    for t in bad_transforms:
        with pytest.raises(ValueError):
            S.check_naip_transform(t, 1920, 1920)
    for w, h in [(1919, 1920), (1920, 1919), (1921, 1920), (960, 960)]:
        with pytest.raises(ValueError):
            S.check_naip_transform(good, w, h)


# ---------------------------------------------------------------- A2: codes and colours


def test_code_table():
    """DS-11, DS-12, DS-13, R3, FM-27: 32 codes, order, colours, level-1 codes, nodata."""
    assert _codes_of(S.CODES) == CODES_IN_ORDER
    assert len(CODES_IN_ORDER) == 32 and len(set(CODES_IN_ORDER)) == 32
    assert S.NODATA == 0 and 0 not in CODES_IN_ORDER
    assert S.NLCD_LEGEND == NLCD_LEGEND_EXPECTED
    assert S.LEVEL1_CODES == LEVEL1_EXPECTED
    assert "Developed, open space" in _name_of(S.CODES, 21)
    legend_colours = {c.lower() for c in NLCD_LEGEND_EXPECTED.values()}
    seen = []
    for code, is_nlcd, colour in CODE_TABLE:
        assert S.colour(code).lower() == colour, code
        if is_nlcd:
            assert colour == NLCD_LEGEND_EXPECTED[code], code
        else:
            assert S.colour(code).lower() not in legend_colours, code
        seen.append(S.colour(code).lower())
    for code in (22, 23, 24):
        seen.append(S.colour(code).lower())
        assert S.colour(code).lower() == NLCD_LEGEND_EXPECTED[code]
    assert len(seen) == 35 and len(set(seen)) == 35
    # the code table in docs/DATA_STANDARD.md agrees with the code (the doc is edited first)
    rows = re.findall(
        r"^\|[^|]*\|\s*(\d{2})\s*\|[^|]*\|[^|]*\|[^|]*\|\s*`(#[0-9a-fA-F]{6})`\s*\|\s*`([a-z ]+)`\s*\|\s*$",
        _doc_text(), re.MULTILINE)  # 7 columns: the last one is the Prompt (D51)
    assert [(int(c), col.lower()) for c, col, _ in rows] == [(c, col) for c, _, col in CODE_TABLE]
    assert {int(c): prompt for c, _, prompt in rows} == S.PROMPTS


def test_vote_tables_and_machine_codes():
    """DS-29, DS-12, DS-11: the reverse maps, the vote of one pixel and the machine-label checks (D52, D54, D56)."""
    assert S.CLASSF_VOTE == {5: 33, 6: 36, 8: 96, 9: 95, 12: 81, 14: 55, 15: 54, 16: 53, 17: 51, 26: 52}
    assert S.NLCD_VOTE[22] == S.NLCD_VOTE[23] == S.NLCD_VOTE[24] == 20
    assert S.NLCD_VOTE[250] is S.VOTE_NODATA and S.NLCD_VOTE[72] == S.VOTE_NONE == 0
    assert S.NLCD_VOTE[12] == 12 and S.vote_code(12, 0) == 12  # D60: ice and snow is a code
    assert S.vote_code(41, 5) == 33 and S.vote_code(22, 0) == 20 and S.vote_code(72, 2) == 20
    assert S.vote_code(41, 2) == 41 and S.vote_code(250, 0) is None and S.vote_code(72, 0) == 0
    for nlcd, classf in ((41, 30), (41, -1), (0, 0), (41, True), (None, 0)):
        with pytest.raises(ValueError):
            S.vote_code(nlcd, classf)
    assert [S.check_machine_code(c) for c in (0, 11, 96)] == [0, 11, 96]
    assert S.check_vote_code(20) == 20 and S.check_vote_code(0) == 0
    for bad in (20, 99, 15):
        with pytest.raises(ValueError):
            S.check_machine_code(bad)
    for bad in (30, 99):
        with pytest.raises(ValueError):
            S.check_vote_code(bad)


def test_prompt_labels_and_overlap_rule_constants():
    """DS-30, DS-11: one prompt for each code (32), no prompt for code 0; label.min_overlap is in config.yaml (D51, D53)."""
    assert list(S.PROMPTS) == list(S.CODE_LIST) and 0 not in S.PROMPTS
    assert [entry.prompt for entry in S.CODES] == list(S.PROMPTS.values())
    assert len(set(S.PROMPTS.values())) == 32
    assert load_config()["label"]["min_overlap"] == 0.5


def test_level1_rules():
    """DS-12, R3: level-1 code is the tens digit x 10, and 99 for wetlands (D27)."""
    expected = {11: 10, 12: 10, 13: 10, 14: 10, 36: 30, 53: 50, 55: 50, 21: 20, 25: 20, 29: 20, 31: 30, 35: 30, 41: 40,
                44: 40, 51: 50, 52: 50, 71: 70, 81: 80, 83: 80, 90: 99, 95: 99, 96: 99}
    for code, l1 in expected.items():
        assert S.level1(code) == l1, code
    for code in LEVEL1_EXPECTED:
        assert S.level1(code) == code
    for bad in (0, 22, 15, 56, 91, 100, -1, 60, 98):
        with pytest.raises(ValueError):
            S.level1(bad)


def test_is_label_code():
    """DS-12, R3: 40 label codes (32 codes and 8 level-1 codes); the two sets do not overlap."""
    labels = {c for c in range(0, 128) if S.is_label_code(c)}
    assert labels == set(CODES_IN_ORDER) | set(LEVEL1_EXPECTED)
    assert len(labels) == 40
    assert not set(CODES_IN_ORDER) & set(LEVEL1_EXPECTED)
    for bad in (0, 22, 23, 24, 15, 56, 91, 250):
        assert not S.is_label_code(bad)


def test_derived_nlcd_codes_and_colours():
    """DS-13, DS-14, R3: DERIVED_NLCD has 17 output codes and the NLCD legend colours."""
    assert set(_codes_of(S.DERIVED_NLCD)) == DERIVED_CODES
    for code in DERIVED_CODES - {0}:
        assert S.colour(code).lower() == NLCD_LEGEND_EXPECTED[code], code
    for bad in (1, 15, 40, 99, 250, -1):
        with pytest.raises(ValueError):
            S.colour(bad)
    assert S.colour(13).lower() == "#5a86b8"


def test_stack_nlcd_value():
    """DS-11, R3: a stack NLCD value of 250 becomes 0; legend codes stay; others raise."""
    assert S.stack_nlcd_value(250) == 0
    for code in (11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95):
        assert S.stack_nlcd_value(code) == code
    for bad in (0, 1, 249, 255, -1, 13, 25):
        with pytest.raises(ValueError):
            S.stack_nlcd_value(bad)


# ---------------------------------------------------------------- A3: derived NLCD code


def test_derived_nlcd_limits():
    """DS-14, R8, FM-27: each limit (0.199, 0.20, 0.499, 0.50, 0.799, 0.80), each group, a tie."""
    f = S.derived_nlcd
    # developed intensity limits, exact comparison on both sides of each limit
    assert f({25: 0.199, 71: 0.801}, 1.0) == 71
    assert f({25: 0.2, 71: 0.8}, 1.0) == 22
    assert f({26: 0.499, 71: 0.501}, 1.0) == 22
    assert f({26: 0.5, 71: 0.5}, 1.0) == 23
    assert f({27: 0.799, 71: 0.201}, 1.0) == 23
    assert f({27: 0.8, 71: 0.2}, 1.0) == 24
    assert f({29: 1.0}, 1.0) == 24
    # I is the sum over 25 to 29
    assert f({25: 0.1, 26: 0.1, 71: 0.8}, 1.0) == 22
    assert f({25: 0.1, 28: 0.4, 71: 0.5}, 1.0) == 23
    assert f({25: 0.2, 26: 0.2, 27: 0.2, 28: 0.2, 29: 0.2}, 1.0) == 24
    # I >= 0.20 wins even if one NLCD group is larger
    assert f({25: 0.2, 41: 0.8}, 1.0) == 22
    # each group: every member code alone
    for out, members in NLCD_GROUPS.items():
        for m in members:
            assert f({m: 1.0}, 1.0) == out, (out, m)
    # group sums decide, not the single largest code
    assert f({11: 0.15, 13: 0.15, 14: 0.15, 41: 0.4, 71: 0.15}, 1.0) == 11
    assert f({31: 0.1, 32: 0.1, 33: 0.1, 34: 0.1, 35: 0.1, 41: 0.2, 71: 0.3}, 1.0) == 31
    # codes 25 to 29 below the limit are in no group
    assert f({25: 0.19, 41: 0.4, 71: 0.41}, 1.0) == 71
    # exact ties: the lowest group code
    assert f({41: 0.5, 42: 0.5}, 1.0) == 41
    assert f({90: 0.5, 11: 0.5}, 1.0) == 11
    assert f({95: 0.5, 71: 0.5}, 1.0) == 71
    assert f({43: 0.5, 44: 0.5}, 1.0) == 43
    # no photo
    assert f({}, 0) == 0
    assert f({}, 0.0) == 0
    assert f({41: 0.0}, 0.0) == 0
    # a partial photo share still uses the shares of the photo area
    assert f({41: 1.0}, 0.25) == 41
    # the sum rule: 1 +- 0.001
    assert f({71: 0.5005, 81: 0.5}, 1.0) == 71
    assert f({71: 0.4995, 81: 0.5}, 1.0) == 81


def test_derived_nlcd_input_errors():
    """DS-14, DS-20, R8: bad keys, values, sums and types raise."""
    f = S.derived_nlcd
    for key in (0, 22, 23, 24, 40, 99, 10, 15, 56, 250):
        with pytest.raises(ValueError):
            f({key: 1.0}, 1.0)
    for v in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            f({71: v, 81: 0.5}, 1.0)
    for p in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            f({71: 1.0}, p)
    with pytest.raises(ValueError):
        f({41: 0.5}, 0)  # a share with photo_share = 0
    with pytest.raises(ValueError):
        f({71: 0.5}, 1.0)  # the sum is not 1
    with pytest.raises(ValueError):
        f({}, 1.0)
    with pytest.raises(ValueError):
        f({71: 0.502, 81: 0.5}, 1.0)  # sum 1.002
    with pytest.raises(ValueError):
        f({71: 0.498, 81: 0.5}, 1.0)  # sum 0.998
    with pytest.raises((ValueError, TypeError)):
        f([("71", 1.0)], 1.0)
    with pytest.raises((ValueError, TypeError)):
        f({71: "1.0"}, 1.0)
    with pytest.raises((ValueError, TypeError)):
        f({71: 1.0}, "1")
    with pytest.raises((ValueError, TypeError)):
        f({71: 1.0}, None)


# ---------------------------------------------------------------- A4: derived USACE-M code


def test_usace_m_map_complete():
    """DS-16: the map has one row for each of the 32 codes (27 fixed targets, 5 tree codes)."""
    assert dict(S.USACE_M_MAP) == USACE_M_MAP_EXPECTED
    assert set(S.USACE_M_MAP) | set(TREE_CODES) == set(CODES_IN_ORDER)
    assert not set(S.USACE_M_MAP) & set(TREE_CODES)
    assert dict(S.USACE_M_LABELS) == USACE_M_LABELS_EXPECTED
    assert set(S.USACE_M_MAP.values()) <= set(S.USACE_M_LABELS)
    assert set(S.USACE_M_NOT_PRODUCED) == NOT_PRODUCED_EXPECTED
    assert not set(S.USACE_M_NOT_PRODUCED) & set(S.USACE_M_LABELS)


def test_usace_m_limits():
    """DS-15, R8, FM-27, D33: developed first; T limits 0.099, 0.10, 0.149, 0.15, 0.45, 0.451."""
    f = S.derived_usace_m
    # D33: I >= 0.20 -> M02 before the tree step
    assert f({25: 0.2, 41: 0.8}, 1.0) == 2
    assert f({25: 0.199, 41: 0.801}, 1.0) == 24
    assert f({25: 0.2, 71: 0.8}, 1.0) == 2
    assert f({29: 0.1, 28: 0.1, 41: 0.8}, 1.0) == 2
    # tree share limits with I = 0 (the rest is code 71)
    assert f({41: 0.099, 71: 0.901}, 1.0) == 10
    assert f({41: 0.1, 71: 0.9}, 1.0) == 18
    assert f({41: 0.149, 71: 0.851}, 1.0) == 18
    assert f({41: 0.15, 71: 0.85}, 1.0) == 21
    assert f({41: 0.45, 71: 0.55}, 1.0) == 21
    assert f({41: 0.451, 71: 0.549}, 1.0) == 24
    assert f({41: 1.0}, 1.0) == 24
    # M18 does not depend on the leaf type
    for code in TREE_CODES:
        assert f({code: 0.1, 71: 0.9}, 1.0) == 18, code
        assert f({code: 0.099, 71: 0.901}, 1.0) == 10, code
    # T is the sum over 41 to 44 and 90
    assert f({41: 0.05, 42: 0.05, 71: 0.9}, 1.0) == 18
    assert f({44: 0.05, 90: 0.05, 71: 0.9}, 1.0) == 18
    assert f({41: 0.09, 11: 0.05, 71: 0.86}, 1.0) == 10
    # no photo
    assert f({}, 0) == 0
    assert f({}, 0.0) == 0


def test_usace_m_leaf_types():
    """DS-15, R8: leaf type from the largest of 41, 42, 43; ties are mixed; 44 and 90 are u."""
    f = S.derived_usace_m
    closed = {41: 24, 42: 23, 43: 22}
    opened = {41: 21, 42: 20, 43: 19}
    for code, out in closed.items():
        assert f({code: 0.6, 71: 0.4}, 1.0) == out, code
    for code, out in opened.items():
        assert f({code: 0.3, 71: 0.7}, 1.0) == out, code
    # unresolved leaf type: 44 alone and 90 alone
    assert f({44: 0.6, 71: 0.4}, 1.0) == 28
    assert f({90: 0.6, 71: 0.4}, 1.0) == 28
    assert f({44: 0.3, 71: 0.7}, 1.0) == 27
    assert f({90: 0.3, 71: 0.7}, 1.0) == 27
    assert f({44: 0.2, 90: 0.2, 71: 0.6}, 1.0) == 27
    assert f({44: 1.0}, 1.0) == 28
    assert f({90: 1.0}, 1.0) == 28
    # 44 is not a leaf type: 41 wins even with a larger 44
    assert f({41: 0.2, 44: 0.4, 71: 0.4}, 1.0) == 24
    assert f({42: 0.1, 44: 0.2, 71: 0.7}, 1.0) == 20
    assert f({43: 0.05, 90: 0.25, 71: 0.7}, 1.0) == 19
    # ties of two or three codes are mixed
    assert f({41: 0.3, 42: 0.3, 71: 0.4}, 1.0) == 22
    assert f({41: 0.3, 43: 0.3, 71: 0.4}, 1.0) == 22
    assert f({42: 0.3, 43: 0.3, 71: 0.4}, 1.0) == 22
    assert f({41: 0.2, 42: 0.2, 43: 0.2, 71: 0.4}, 1.0) == 22
    assert f({41: 0.1, 42: 0.1, 71: 0.8}, 1.0) == 19
    assert f({41: 0.1, 42: 0.1, 43: 0.1, 71: 0.7}, 1.0) == 19
    # a strict maximum is not a tie
    assert f({41: 0.3, 42: 0.2, 43: 0.1, 71: 0.4}, 1.0) == 24
    assert f({41: 0.1, 42: 0.2, 43: 0.3, 71: 0.4}, 1.0) == 22


def test_usace_m_non_tree_targets():
    """DS-15, DS-16, R8: each non-tree code maps to its raster code; sums; ties -> lowest code."""
    f = S.derived_usace_m
    for code, target in USACE_M_MAP_EXPECTED.items():
        assert f({code: 1.0}, 1.0) == target, code
    for code in (41, 42, 43, 44, 90):
        assert f({code: 1.0}, 1.0) in (22, 23, 24, 28)
    # summed share of the codes with the same target
    assert f({11: 0.3, 13: 0.3, 71: 0.4}, 1.0) == 1
    assert f({21: 0.3, 71: 0.3, 82: 0.4}, 1.0) == 10
    assert f({31: 0.2, 35: 0.2, 33: 0.3, 82: 0.3}, 1.0) == 3
    # exact tie: the lowest raster code
    assert f({11: 0.5, 71: 0.5}, 1.0) == 1
    assert f({33: 0.5, 82: 0.5}, 1.0) == 5
    assert f({52: 0.5, 51: 0.5}, 1.0) == 17
    assert f({95: 0.5, 96: 0.5}, 1.0) == 8
    # tree share below 0.10 does not count as a target
    assert f({41: 0.09, 52: 0.45, 51: 0.46}, 1.0) == 17


def test_usace_m_flags():
    """DS-16, DS-20, R8: bit 1 (woody wetland) and bit 128 (maintained grass), both limits."""
    g = S.usace_m_flags
    # bit 1: tree class and s90 >= sum of 41 to 44
    assert g({90: 1.0}, 1.0) == 1
    assert g({90: 0.3, 41: 0.3, 71: 0.4}, 1.0) == 1  # equal
    assert g({90: 0.31, 41: 0.3, 71: 0.39}, 1.0) == 1  # above
    assert g({90: 0.29, 41: 0.3, 71: 0.41}, 1.0) == 0  # below
    assert g({90: 0.2, 41: 0.1, 42: 0.1, 71: 0.6}, 1.0) == 1  # equal to the sum of 41 to 44
    assert g({90: 0.1, 41: 0.1, 44: 0.1, 71: 0.7}, 1.0) == 0  # 0.1 < 0.1 + 0.1
    assert g({90: 0.1, 71: 0.9}, 1.0) == 1  # M18, T = 0.10
    assert g({41: 1.0}, 1.0) == 0
    # bit 1 is 0 for a non-tree code
    assert g({90: 0.05, 71: 0.95}, 1.0) == 0
    assert g({25: 0.2, 90: 0.8}, 1.0) == 0  # code 2
    assert g({95: 1.0}, 1.0) == 0
    # bit 128: code 10 and s21 >= s71
    assert g({21: 1.0}, 1.0) == 128
    assert g({21: 0.5, 71: 0.5}, 1.0) == 128  # equal
    assert g({21: 0.6, 71: 0.4}, 1.0) == 128  # above
    assert g({21: 0.4, 71: 0.6}, 1.0) == 0  # below
    assert g({71: 1.0}, 1.0) == 0
    assert g({21: 0.5, 82: 0.5}, 1.0) == 128  # tie 10 vs 13: the lowest raster code is 10
    # the bit 128 needs code 10 (here the code is 13)
    assert g({21: 0.2, 82: 0.8}, 1.0) == 0
    # no photo
    assert g({}, 0) == 0
    with pytest.raises(ValueError):
        g({41: 0.5}, 0)
    with pytest.raises(ValueError):
        g({22: 1.0}, 1.0)


def test_usace_m_input_errors():
    """DS-15, R8: the input checks of A3 apply to derived_usace_m, usace_m_flags and top_code."""
    for fn in (S.derived_usace_m, S.usace_m_flags, S.top_code):
        for key in (0, 22, 40, 99):
            with pytest.raises(ValueError):
                fn({key: 1.0}, 1.0)
        with pytest.raises(ValueError):
            fn({71: 1.2}, 1.0)
        with pytest.raises(ValueError):
            fn({71: float("nan")}, 1.0)
        with pytest.raises(ValueError):
            fn({71: 0.5}, 1.0)
        with pytest.raises(ValueError):
            fn({71: 1.0}, -0.1)
        with pytest.raises(ValueError):
            fn({71: 1.0}, 1.1)
        with pytest.raises(ValueError):
            fn({71: 0.4}, 0)


def _scheme_code_for_all():
    tree = _scheme_tree()
    return tree, _scheme_value(tree, "CLASSES")


def test_usace_m_matches_scheme():
    """DS-16, FM-25, R8, D3: the USACE-M map agrees with scheme.py (read as text, never imported)."""
    assert SCHEME_PATH.is_file(), f"read-only reference missing: {SCHEME_PATH}"
    tree, classes = _scheme_code_for_all()
    assert len(classes) == 29
    for raster_code, label in S.USACE_M_LABELS.items():
        assert classes[raster_code][0] == label, (raster_code, label)
    for raster_code in S.USACE_M_NOT_PRODUCED:
        assert raster_code in classes
    # every target of the map and every derived output is a valid scheme.py raster code
    for target in S.USACE_M_MAP.values():
        assert target in classes
    for code in CODES_IN_ORDER:
        out = S.derived_usace_m({code: 1.0}, 1.0)
        assert out in classes and out in S.USACE_M_LABELS, code
    # scheme.py constants
    assert _scheme_value(tree, "NODATA") == 0 == S.NODATA
    f1 = _scheme_value(tree, "F1_WOODY_WETLAND")
    assert S.usace_m_flags({90: 1.0}, 1.0) == f1 == 1
    # bit 128 must be free in scheme.py (bits 1 to 64 are used)
    used = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if re.fullmatch(r"F\d_[A-Z_]+|CONFLICT|UNCONFIRMED", n):
                used.append(ast.literal_eval(node.value))
    assert 128 not in used and max(used) <= 64
    assert S.usace_m_flags({21: 1.0}, 1.0) == 128
    # the tree classes of the flag rule are the scheme.py TREE set plus the unresolved ones
    tree_set = _scheme_value(tree, "TREE")
    assert {18, 19, 20, 21, 22, 23, 24, 27, 28} == set(tree_set)
    # the leaf-type tables of scheme.py agree with DS-15 (names resolved from the tuple assign)
    leaf_names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [t.id for t in node.targets[0].elts]
            if names[0] == "LEAF_DECID":
                leaf_names = dict(zip(names, ast.literal_eval(node.value)))
    assert set(leaf_names) == {"LEAF_DECID", "LEAF_EVERG", "LEAF_MIXED"}
    by_leaf = _scheme_value_resolved(tree, "TREE_BY_LEAF", leaf_names)
    leaf = _scheme_value_resolved(tree, "NLCD_LEAF", leaf_names)
    assert by_leaf["closed"][leaf[41]] == S.derived_usace_m({41: 0.6, 71: 0.4}, 1.0) == 24
    assert by_leaf["closed"][leaf[42]] == S.derived_usace_m({42: 0.6, 71: 0.4}, 1.0) == 23
    assert by_leaf["closed"][leaf[43]] == S.derived_usace_m({43: 0.6, 71: 0.4}, 1.0) == 22
    assert by_leaf["open"][leaf[41]] == S.derived_usace_m({41: 0.3, 71: 0.7}, 1.0) == 21
    assert by_leaf["open"][leaf[42]] == S.derived_usace_m({42: 0.3, 71: 0.7}, 1.0) == 20
    assert by_leaf["open"][leaf[43]] == S.derived_usace_m({43: 0.3, 71: 0.7}, 1.0) == 19
    unresolved = _scheme_value(tree, "TREE_UNRESOLVED")
    assert unresolved == {"closed": 28, "open": 27}
    assert _scheme_value(tree, "SHRUB_UNRESOLVED") == S.USACE_M_MAP[52] == 26


def test_scheme_is_not_imported():
    """DS-16, FM-25, D3: this test module and semseg.standard never import usace_veg."""
    for path in (Path(__file__), ROOT / "src" / "semseg" / "standard.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert not [n for n in names if n.split(".")[0] in ("usace" + "_veg", "fine" + "_grid")]


# ---------------------------------------------------------------- A5: names and records


def test_history_years():
    """DS-5: the history years are 1985 to 2025 (41 years)."""
    assert S.HISTORY_YEARS == tuple(range(1985, 2026))
    assert len(S.HISTORY_YEARS) == 41


def test_project_folder_names():
    """DS-6: pilots, folder name <pid>_<NAME>, and the same names as the stack folders."""
    assert dict(S.PILOTS) == {77: "GRAPEVINE_LAKE", 392: "RAYSTOWN_LAKE"}
    assert S.project_folder(77, "GRAPEVINE_LAKE") == "77_GRAPEVINE_LAKE"
    assert S.project_folder(392, "RAYSTOWN_LAKE") == "392_RAYSTOWN_LAKE"
    assert S.parse_project_folder("77_GRAPEVINE_LAKE") == (77, "GRAPEVINE_LAKE")
    assert S.parse_project_folder(
        "249_MISSISSIPPI_RIVER_BETWEEN_MISSOURI_RIVER_AND_MINNEAPOLIS"
    ) == (249, "MISSISSIPPI_RIVER_BETWEEN_MISSOURI_RIVER_AND_MINNEAPOLIS")
    for pid, name in S.PILOTS.items():
        assert S.parse_project_folder(S.project_folder(pid, name)) == (pid, name)
    for pid, name in [(0, "X"), (-1, "X"), (77, "grape"), (77, ""), (77, "A B"), (77, "A-B"),
                      (1.5, "X"), ("77", "X")]:
        with pytest.raises((ValueError, TypeError)):
            S.project_folder(pid, name)
    for text in ["077_X", "0_X", "77_", "77", "_X", "77_x", "x_X", "77 X", ""]:
        with pytest.raises(ValueError):
            S.parse_project_folder(text)
    stack = _need_root("stack")
    for pid, name in S.PILOTS.items():
        assert (stack / S.project_folder(pid, name)).is_dir()


def test_polygon_ids():
    """DS-9: polygon ID <chunk>_<run_id>_<n>; round trip; not stable between runs."""
    chunk, run_a, run_b = "r2295_c2394", "20261112-0930-sam2", "20261113-1015-sam2"
    pid_a = S.polygon_id(chunk, run_a, 5)
    assert pid_a == "r2295_c2394_20261112-0930-sam2_5"
    assert S.parse_polygon_id(pid_a) == (chunk, run_a, 5)
    assert S.parse_polygon_id(S.polygon_id(chunk, run_b, 0)) == (chunk, run_b, 0)
    assert S.polygon_id(chunk, run_b, 5) != pid_a  # a polygon ID changes with the run
    # the cell ID does not depend on the run: a join between runs uses it
    assert S.cell_id(73346, 76515) == S.cell_id(73346, 76515)
    for bad in [(chunk, run_a, -1), (chunk, run_a, 1.5), (chunk, run_a, "1"), ("x", run_a, 1), (chunk, "2026-sam2", 1), (chunk, "20261112-0930-SAM2", 1)]:
        with pytest.raises((ValueError, TypeError)):
            S.polygon_id(*bad)
    for text in ["", "r1_c2", "r1_c2_20261112-0930-sam2", "r1_c2_20261112-0930-sam2_",
                 "r1_c2_20261112-0930-sam2_x", "r1_c2_bad_1"]:
        with pytest.raises(ValueError):
            S.parse_polygon_id(text)


def test_run_ids():
    """DS-10: run ID <YYYYMMDD-HHMM>-<method> and the regular expression."""
    assert S.RUN_ID_RE == r"\A[0-9]{8}-[0-9]{4}-[a-z0-9_]+\Z"
    assert S.run_id(dt.datetime(2026, 11, 12, 9, 30), "sam2") == "20261112-0930-sam2"
    assert S.run_id(dt.datetime(2026, 1, 2, 3, 4), "lang_sam") == "20260102-0304-lang_sam"
    assert S.run_id(dt.datetime(2026, 12, 31, 23, 59, 59), "b1") == "20261231-2359-b1"
    for method in ["SAM2", "sam-2", "", "sam 2", "sam/2", "sam2\n"]:
        with pytest.raises(ValueError):
            S.run_id(dt.datetime(2026, 11, 12, 9, 30), method)
    S.check_run_id("20261112-0930-sam2")
    for text in ["20261112-930-sam2", "2026112-0930-sam2", "20261112-0930-", "20261112-0930",
                 "x", "", "20261112-0930-Sam2", "20261112_0930_sam2"]:
        with pytest.raises(ValueError):
            S.check_run_id(text)


@pytest.mark.parametrize("kind,dtype,nodata", [("code", "uint8", 0), ("percent", "uint8", None),
                                               ("value", "float32", None)])
def test_raster_profile(kind, dtype, nodata):
    """DS-17, DS-1: GeoTIFF, deflate, tiled 256 x 256, EPSG:5070; dtype by kind; nodata 0 for code."""
    p = S.raster_profile(kind)
    assert p["driver"] == "GTiff"
    assert p["compress"].lower() == "deflate"
    assert p["tiled"] is True
    assert p["blockxsize"] == 256 and p["blockysize"] == 256
    assert p["crs"] == "EPSG:5070"
    assert p["dtype"] == dtype
    assert p.get("nodata") == nodata


@pytest.mark.parametrize("kind", ["other", "", "Code", None, 1])
def test_raster_profile_bad_kind(kind):
    """DS-17: an unknown raster kind raises ValueError."""
    with pytest.raises(ValueError):
        S.raster_profile(kind)


def test_file_format_constants():
    """DS-18, DS-19, DS-21: layer names, the CSV size limit and the metadata keys."""
    assert tuple(S.GPKG_LAYERS) == ("polygons", "labels", "sample")
    assert S.RESULTS_MAX_BYTES == 1_000_000
    assert tuple(S.METADATA_KEYS) == ("run_id", "code_version", "data_standard", "naip_year",
                                      "sources", "created")


def test_results_files_are_small():
    """DS-19: each file in results/ is smaller than 1 MB."""
    results = ROOT / "results"
    for f in (results.rglob("*") if results.is_dir() else []):
        if f.is_file():
            assert f.stat().st_size < S.RESULTS_MAX_BYTES, f


def test_record_columns():
    """DS-20, DS-11, DS-14, DS-15: the cell record has 169 columns in a fixed order."""
    cols = S.record_columns()
    assert isinstance(cols, list)
    assert len(cols) == 169 and len(set(cols)) == 169
    assert cols[:8] == ["cell_id", "row", "col", "pid", "chunk", "naip_year", "naip_date",
                        "photo_share"]
    assert cols[8:40] == [f"share_{c}" for c in CODES_IN_ORDER]
    assert cols[40:45] == ["top_code", "n_poly", "nlcd_derived", "usace_m_derived",
                           "usace_m_flags"]
    years = range(1985, 2026)
    assert cols[45:86] == [f"nlcd_{y}" for y in years]
    assert cols[86:127] == [f"classf_{y}" for y in years]
    assert cols[127:168] == [f"tcc_{y}" for y in years]
    assert cols[168] == "run_id"
    assert "share_0" not in cols and "share_22" not in cols and "share_40" not in cols


def test_top_code():
    """DS-20: the code with the largest share; lowest code on a tie; 0 without a photo."""
    t = S.top_code
    assert t({}, 0) == 0
    assert t({71: 0.6, 41: 0.4}, 1.0) == 71
    assert t({41: 0.5, 42: 0.5}, 1.0) == 41
    assert t({96: 0.5, 11: 0.5}, 1.0) == 11
    assert t({25: 1.0}, 1.0) == 25
    assert t({41: 0.3, 42: 0.3, 43: 0.4}, 1.0) == 43
    assert t({41: 1.0}, 0.5) == 41


def test_folder_names():
    """DS-22: folder functions return PurePosixPath, check each part, and do not touch the disk."""
    assert tuple(S.PROJECT_SUBFOLDERS) == ("polygons", "cells", "rasters", "labels", "sample",
                                          "sources", "baselines")
    assert S.CHUNK_LIST == "chunks.parquet"
    assert set(S.RESERVED_NAMES) == {"aux", "con", "nul", "prn"}
    root = "/nonexistent_root_for_test"
    p = S.project_dir(root, "77_GRAPEVINE_LAKE", "polygons")
    assert isinstance(p, PurePosixPath)
    assert p == PurePosixPath(root, "77_GRAPEVINE_LAKE", "polygons")
    for kind in S.PROJECT_SUBFOLDERS:
        assert S.project_dir(root, "392_RAYSTOWN_LAKE", kind).name == kind
    r = S.run_dir(root, "20261112-0930-sam2")
    assert isinstance(r, PurePosixPath)
    assert r == PurePosixPath(root, "runs", "20261112-0930-sam2")
    m = S.models_dir(root)
    assert isinstance(m, PurePosixPath) and m == PurePosixPath(root, "models")
    assert not Path(root).exists()


@pytest.mark.parametrize("part", ["aux", "AUX", "Con", "nul", "PRN", "aux.txt", "con.tar.gz",
                                  "NUL.parquet", "", ".", "..", "a/b", "/"])
def test_check_name_rejects(part):
    """DS-22: reserved names (any case, any extension), empty, dot parts and slashes raise."""
    with pytest.raises(ValueError):
        S.check_name(part)
    with pytest.raises(ValueError):
        S.project_dir("/r", part, "polygons")
    with pytest.raises(ValueError):
        S.project_dir("/r", "77_GRAPEVINE_LAKE", part)
    with pytest.raises(ValueError):
        S.run_dir("/r", part)


@pytest.mark.parametrize("part", ["auxiliary", "polygons", "77_GRAPEVINE_LAKE", "console",
                                  "models", "a.b", "nulls", "prnt.txt"])
def test_check_name_accepts(part):
    """DS-22: other names pass check_name."""
    S.check_name(part)


def test_storage_constants():
    """DS-23, DS-24, DS-25, DS-26, DS-27, DS-28: constants of the storage rules (D31)."""
    assert dict(S.ROOTS) == {"semseg": "rw", "fgt": "ro", "stack": "ro"}
    assert set(S.ROOTS) <= set(_registry_roots())
    assert S.PART_SUFFIX == ".part"
    assert S.EXIT_SHARE_MISSING == 3
    assert S.APP_DB == "var/app.sqlite"
    assert S.MODELS_DIR == "models"


def test_no_large_tracked_file():
    """DS-24: `git ls-files` has no file larger than 1 000 000 bytes."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    files = [f for f in out.stdout.decode("utf-8").split("\0") if f]
    assert files
    for name in files:
        p = ROOT / name
        if p.is_file():
            assert p.stat().st_size <= 1_000_000, name


# ---------------------------------------------------------------- A6: FM-24


def _test_docstrings():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            out[node.name] = ast.get_docstring(node) or ""
    return out


def test_every_ds_rule_has_test():
    """DS-1, DS-2, DS-3, DS-4, DS-5, DS-6, DS-7, DS-8, DS-9, DS-10, DS-11, DS-12, DS-13, DS-14,
    DS-15, DS-16, DS-17, DS-18, DS-19, DS-20, DS-21, DS-22, DS-23, DS-24, DS-25, DS-26, DS-27,
    DS-28, DS-29, DS-30, DS-31, DS-32, DS-33, DS-34, FM-24: each DS ID of docs/DATA_STANDARD.md is in a test docstring or the mapping."""
    ids = re.findall(r"^\| (DS-\d+) \|", _doc_text(), re.MULTILINE)
    assert ids == [f"DS-{n}" for n in range(1, 35)]
    docs = _test_docstrings()
    in_docstrings = set()
    for name, doc in docs.items():
        if name == "test_every_ds_rule_has_test":
            continue
        in_docstrings |= set(re.findall(r"DS-\d+", doc))
    for rule, owner in OWNED_ELSEWHERE.items():
        assert rule in ids and owner.strip(), rule
    missing = [i for i in ids if i not in in_docstrings and i not in OWNED_ELSEWHERE]
    assert missing == [], f"DS rules without a test: {missing}"
    # D31 / A6: every docstring ID is a real rule; the rules owned elsewhere also have a
    # 0.2 test (constants or pure function) in this module
    assert in_docstrings <= set(ids)
    assert set(ids) <= in_docstrings
    # the named PLAN tests exist
    for name in ("test_grid_lattice", "test_code_table", "test_derived_nlcd_limits",
                 "test_off_lattice_raises", "test_usace_m_matches_scheme",
                 "test_grid_lattice_not_skipped"):
        assert name in docs
    # each test docstring starts with an ID (DS, R or FM)
    for name, doc in docs.items():
        assert re.match(r"(DS|R|FM)-?\d+", doc), name


# ---------------------------------------------------------------- D60: categories from METHOD_v1


def test_new_codes_in_derived_codes():
    """DS-14, DS-15, DS-16, D60: codes 12, 36, 53, 54, 55 in the derived NLCD and USACE-M codes."""
    n, m = S.derived_nlcd, S.derived_usace_m
    assert n({12: 1.0}, 1.0) == 12 and m({12: 1.0}, 1.0) == 3
    assert n({36: 1.0}, 1.0) == 31 and m({36: 1.0}, 1.0) == 6
    for code in (53, 54, 55):
        assert n({code: 1.0}, 1.0) == 52
    assert S.level1(12) == 10 and S.colour(12) == "#d1def8"
    assert S.DERIVED_NLCD[12].name == "Perennial ice/snow"


def test_usace_m_shrub_target():
    """DS-15, DS-16, D60: shrub codes 52 to 55 are one target; leaf type from 53, 54, 55."""
    f = S.derived_usace_m
    assert [f({c: 1.0}, 1.0) for c in (52, 53, 54, 55)] == [26, 16, 15, 14]
    # the shrub codes sum: 0.3 + 0.3 beats 0.4 grass
    assert f({53: 0.3, 54: 0.3, 71: 0.4}, 1.0) == 14  # tie of 53 and 54 -> mixed
    assert f({53: 0.2, 54: 0.2, 55: 0.2, 71: 0.4}, 1.0) == 14  # tie of three -> mixed
    assert f({53: 0.31, 54: 0.29, 71: 0.4}, 1.0) == 16
    assert f({54: 0.31, 53: 0.29, 71: 0.4}, 1.0) == 15
    assert f({55: 0.31, 53: 0.29, 71: 0.4}, 1.0) == 14
    # 52 is not a leaf type: 53 wins even with a larger 52
    assert f({52: 0.5, 53: 0.1, 71: 0.4}, 1.0) == 16
    assert f({52: 0.3, 11: 0.3, 71: 0.4}, 1.0) == 10  # 52 alone at 0.3 < 0.4
    assert f({52: 0.6, 71: 0.4}, 1.0) == 26
    # a tie with another target: the shrub target counts as the code of its result
    assert f({52: 0.5, 51: 0.5}, 1.0) == 17  # 17 < 26
    assert f({55: 0.5, 51: 0.5}, 1.0) == 14  # 14 < 17
    assert f({53: 0.5, 71: 0.5}, 1.0) == 10  # 10 < 16
    # the shrub step needs T < 0.10
    assert f({53: 0.85, 41: 0.15}, 1.0) == 21


def test_usace_m_stack_flags():
    """DS-16, DS-20, D60: bits 2 (F2) and 4 (F3) come from the stack value; other bits are not copied."""
    g = S.usace_m_flags
    assert (S.FLAG_INTRODUCED, S.FLAG_DRAWDOWN) == (2, 4)
    assert g({71: 1.0}, 1.0, 2) == 2 and g({71: 1.0}, 1.0, 4) == 4
    assert g({71: 1.0}, 1.0, 6) == 6
    assert g({71: 1.0}, 1.0, 1 | 8 | 16) == 0  # F1, CONFLICT, UNCONFIRMED: not copied
    assert g({71: 1.0}, 1.0, 255) == 6
    assert g({90: 1.0}, 1.0, 4) == 5  # photo F1 and stack F3
    assert g({21: 1.0}, 1.0, 2) == 130
    assert g({}, 0, 4) == 4  # kept without a photo
    assert g({71: 1.0}, 1.0) == 0
    for bad in (-1, 256, 2.0, True, None, "2"):
        with pytest.raises(ValueError):
            g({71: 1.0}, 1.0, bad)


def test_change_groups():
    """DS-31, D60: the 11 change groups of the 29 raster codes; code 0 has none; others raise."""
    expected = {1: "W", 29: "W", 2: "D", 3: "X", 4: "SP", 5: "SP", 6: "SP", 7: "SP", 25: "SP",
                8: "AB", 9: "EM", 10: "H", 11: "H", 12: "AG", 13: "AG", 14: "S", 15: "S",
                16: "S", 17: "S", 26: "S", 18: "TO", 19: "TO", 20: "TO", 21: "TO", 27: "TO",
                22: "TC", 23: "TC", 24: "TC", 28: "TC"}
    assert {c: S.change_group(c) for c in range(1, 30)} == expected
    assert S.change_group(0) is None
    assert S.VEG_GROUPS == {"SP", "EM", "H", "S", "TO", "TC"}
    for bad in (30, -1, 255, 1.0, True, None):
        with pytest.raises(ValueError):
            S.change_group(bad)


def test_change_groups_match_scheme():
    """DS-31, DS-33, D60, D3: the groups, the assessed set and the F2, F3 bits agree with scheme.py (text)."""
    tree, classes = _scheme_code_for_all()
    for code, row in classes.items():
        group = row[3]
        if code in (18, 19, 20, 21, 27):
            group = "TO"
        elif code in (22, 23, 24, 28):
            group = "TC"
        elif group == "NV":
            group = "SP"  # usace-landuse change GROUP11[7] = "SP"
        assert S.change_group(code) == group, code
    assert set(_scheme_value(tree, "ASSESSED")) == S.ASSESSED
    assert _scheme_value(tree, "F2_INTRODUCED") == S.FLAG_INTRODUCED
    assert _scheme_value(tree, "F3_DRAWDOWN") == S.FLAG_DRAWDOWN


def test_change_category_each_rule():
    """DS-32, D60: one pair for each category, in the rule order."""
    f = S.change_category
    cases = {
        (0, 24): 0, (24, 0): 0, (0, 0): 0,
        (24, 24): 1, (1, 1): 1,
        (10, 9): 3, (4, 9): 3, (1, 8): 3,
        (9, 10): 4, (9, 25): 4, (8, 1): 4,
        (21, 19): 11, (24, 23): 11, (1, 29): 11, (2, 2): 1, (14, 26): 11, (12, 13): 11,
        (24, 1): 5, (2, 1): 5, (12, 29): 5,
        (1, 4): 6, (29, 25): 6,
        (1, 9): 7, (1, 10): 7, (29, 11): 7,
        (1, 2): 8, (1, 24): 8, (1, 13): 8, (1, 3): 8, (1, 16): 8,
        (21, 24): 9, (18, 28): 9,
        (24, 21): 10, (28, 18): 10,
        (10, 16): 12, (10, 21): 12, (11, 24): 12, (16, 24): 12, (17, 18): 12, (4, 10): 12,
        (25, 24): 12, (9, 16): 12, (9, 24): 12, (9, 21): 12,
        (24, 16): 13, (21, 10): 13, (16, 10): 13, (24, 4): 13, (10, 25): 13, (16, 5): 13,
        (24, 2): 14, (9, 2): 14, (4, 2): 14,
        (24, 3): 15, (10, 3): 15,
        (24, 13): 16, (9, 12): 16,
        (12, 10): 17, (13, 24): 17, (12, 9): 17,
        (2, 10): 18, (3, 24): 18, (3, 4): 18,
        (24, 9): 19, (8, 10): 19, (2, 3): 19, (12, 2): 19, (3, 12): 19, (8, 2): 19, (16, 9): 19,
    }
    for (a, b), cat in cases.items():
        assert f(a, b) == cat, (a, b)
    assert set(cases.values()) == set(S.CHANGE_CATEGORIES) - {2}  # 2 needs confirmed=False
    assert len(S.CHANGE_CATEGORIES) == 20


def test_change_category_unconfirmed_and_errors():
    """DS-32, D60: an unconfirmed change is 2 before all other rules; persistence and no data first."""
    f = S.change_category
    assert f(24, 2, confirmed=False) == 2
    assert f(1, 8, confirmed=False) == 2
    assert f(24, 24, confirmed=False) == 1
    assert f(0, 24, confirmed=False) == 0
    for bad in (1, 0, None, "yes"):
        with pytest.raises(ValueError):
            f(24, 2, confirmed=bad)
    for a, b in ((30, 1), (1, 30), (1.0, 1), (True, 1)):
        with pytest.raises(ValueError):
            f(a, b)


def test_change_category_all_pairs():
    """DS-32, DS-31, D60: every pair of the 30 raster codes gives one category of the table."""
    seen = set()
    for a in range(30):
        for b in range(30):
            cat = S.change_category(a, b)
            assert cat in S.CHANGE_CATEGORIES
            seen.add(cat)
            if a and b and a != b and S.change_group(a) == S.change_group(b):
                assert cat == 11, (a, b)
    assert seen == set(S.CHANGE_CATEGORIES) - {2}


def test_condition_state_limits():
    """DS-33, D60: the rating limits, the state of each rating, not assessed and no data."""
    r = S.condition_rating
    assert [r(v) for v in (5.0, 4.5, 4.499, 3.5, 3.499, 2.0, 1.999, 1.0)] == list("AABBCCDD")
    assert r(4.4999999999) == "A"  # rounded to 9 decimals first
    assert r(4) == "B"
    st = S.condition_state
    assert [st(24, v) for v in (4.5, 3.5, 3.499, 2.0, 1.999)] == [1, 1, 2, 2, 3]
    for code in (1, 2, 3, 7, 8, 12, 13, 25, 29):
        assert st(code) == 4, code
        with pytest.raises(ValueError):
            st(code, 3.0)
    assert st(0) == 0
    for code in S.ASSESSED:
        with pytest.raises(ValueError):
            st(code)
    for bad in (0.999, 5.001, -1, float("nan"), float("inf"), "3", None, True):
        with pytest.raises(ValueError):
            r(bad)
    for bad in (30, -1, 2.0):
        with pytest.raises(ValueError):
            st(bad)
    assert S.CONDITION_STATES == {0: "No data", 1: "Sustainable", 2: "Transitioning",
                                  3: "Degraded", 4: "Not assessed"}


def test_trend_slope_and_trajectory():
    """DS-34, D60: Theil-Sen slope of 6 years; the +-0.1 limits; not available; newly assessed."""
    t = S.trend_slope
    assert t([3, 3.1, 3.2, 3.3, 3.4, 3.5]) == 0.1
    assert t([4, 3.9, 3.8, 3.7, 3.6, 3.5]) == -0.1
    assert t([3, 3, 3, 3, 3, 3]) == 0
    assert t([3, 3, 3, 3, 3, 5]) == 0  # one outlier does not move the median
    assert t((1, 2, 3, 4, 5, 5)) == 1.0
    for bad in ([3] * 5, [3] * 7, [], "333333", 3, None, [3, 3, 3, 3, 3, "3"],
                [3, 3, 3, 3, 3, float("nan")]):
        with pytest.raises(ValueError):
            t(bad)
    j = S.trajectory
    assert [j(v) for v in (0.1, 0.0999999, 0, -0.0999999, -0.1, 0.5, -0.5)] == [1, 2, 2, 2, 3, 1, 3]
    assert j(0.09999999999) == 1  # rounded to 9 decimals first
    assert j(None) == 0
    assert j(None, assessed_before=False) == 1
    assert j(-0.5, assessed_before=False) == 1
    for bad in ("0.1", True, float("nan")):
        with pytest.raises(ValueError):
            j(bad)
    with pytest.raises(ValueError):
        j(0.1, assessed_before=None)
    assert S.TRAJECTORIES == {0: "Not available", 1: "Recovering", 2: "Stable", 3: "Declining"}
    assert (S.TREND_WINDOW, S.TREND_LIMIT) == (6, 0.1)


def test_score_change():
    """DS-34, D60: the +-0.5 limits of the score change; the scores must be 1 to 5."""
    c = S.score_change
    assert c(3.0, 3.5) == 1 and c(3.0, 3.4999) == 2 and c(3.0, 3.0) == 2
    assert c(3.5, 3.0) == 3 and c(3.4999, 3.0) == 2
    assert c(1.0, 5.0) == 1 and c(5.0, 1.0) == 3
    assert c(3.1, 3.6) == 1  # binary error: 3.6 - 3.1 = 0.5000000000000004, rounded to 0.5
    assert c(3.6, 3.1) == 3
    for a, b in ((0.5, 3), (3, 5.5), (None, 3), (3, "3")):
        with pytest.raises(ValueError):
            c(a, b)
    assert S.SCORE_CHANGES == {1: "Improved", 2: "Unchanged", 3: "Declined"}
