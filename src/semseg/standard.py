"""The data standard (`docs/DATA_STANDARD.md`, version 2) in code.

Task 0.2. This module has pure functions and constants only. No function reads or writes a
file. The storage code is task 0.3. A bad input raises `ValueError`. A function never rounds an
input to make it valid, and every comparison is exact, except the +- 0.001 of the share sum
(DS-20). Grid math uses exact arithmetic (int or Fraction). Each group share sum is rounded to
9 decimals first (D34). The USACE-M raster codes follow `usace-landuse` `scheme.py`; the tests read that file
as text (D3: this module never imports or copies its code).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime
from fractions import Fraction
from numbers import Integral, Real
from pathlib import PurePosixPath
from typing import NamedTuple

DATA_STANDARD_VERSION = 2

# ---------------------------------------------------------------- grid (DS-1 to DS-4, DS-7, DS-8)

CRS = "EPSG:5070"
X0 = -2415585
Y0 = 3314805
CELL = 30
CHUNK = 960
CELLS_PER_CHUNK = 32
PIXEL = 0.5
PIXELS_PER_CELL = 60
PIXELS_PER_CHUNK = 1920
CELL_ID_FACTOR = 1_000_000

INT64_MAX = 2**63 - 1
_PIXEL_EXACT = Fraction(1, 2)
_NUM = r"(?:0|[1-9][0-9]*)"
_CHUNK_ID_RE = re.compile(rf"r({_NUM})_c({_NUM})", re.ASCII)


def _real(value, name="value"):
    """Return a finite real number. A bool, a string or a NaN raises ValueError."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _exact(value, name="value"):
    """Return the exact value of a finite real number: an int if whole, else a Fraction.

    A numpy scalar works. A bool, an array, a NaN or an inf raises ValueError.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return int(number) if number.is_integer() else Fraction(number)


def _int(value, name="value"):
    """Return an integer. A bool, a float or a string raises ValueError."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _index(value, name):
    """Return an integer that is 0 or more."""
    value = _int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be 0 or more, got {value}")
    return value


def on_lattice(x, y):
    """DS-2: true if (x, y) is on the 30 m lattice. The test is exact."""
    x = _exact(x, "x")
    y = _exact(y, "y")
    return (x - X0) % CELL == 0 and (Y0 - y) % CELL == 0


def _check_lattice(x, y):
    if not on_lattice(x, y):
        raise ValueError(f"({x}, {y}) is not on the 30 m lattice (x0={X0}, y0={Y0})")


def cell_of_bounds(xmin, ymax):
    """DS-8: the (row, col) of the cell with the upper left corner (xmin, ymax)."""
    _check_lattice(xmin, ymax)
    row = int((Y0 - _exact(ymax)) // CELL)
    col = int((_exact(xmin) - X0) // CELL)
    if row < 0 or col < 0:
        raise ValueError(f"({xmin}, {ymax}) is outside the grid (row {row}, col {col})")
    return row, col


def cell_of_point(x, y):
    """DS-8: the (row, col) of the cell that holds the point. A cell has its left and top edge."""
    x = _exact(x, "x")
    y = _exact(y, "y")
    row = int((Y0 - y) // CELL)
    col = int((x - X0) // CELL)
    if row < 0 or col < 0:
        raise ValueError(f"({x}, {y}) is outside the grid (row {row}, col {col})")
    return row, col


def cell_bounds(row, col):
    """DS-8: (xmin, ymin, xmax, ymax) of a cell."""
    row = _index(row, "row")
    col = _index(col, "col")
    xmin = X0 + CELL * col
    ymax = Y0 - CELL * row
    return xmin, ymax - CELL, xmin + CELL, ymax


def cell_id(row, col):
    """DS-8: `row * 1 000 000 + col`, with row >= 0, 0 <= col < 1 000 000 and an ID in int64."""
    row = _index(row, "row")
    col = _index(col, "col")
    if col >= CELL_ID_FACTOR:
        raise ValueError(f"col must be less than {CELL_ID_FACTOR}, got {col}")
    result = row * CELL_ID_FACTOR + col
    if result > INT64_MAX:
        raise ValueError(f"the cell ID is above {INT64_MAX}: row {row}, col {col}")
    return result


def cell_rowcol(cell):
    """DS-8: the (row, col) of a cell ID (0 to 2**63 - 1)."""
    cell = _index(cell, "cell_id")
    if cell > INT64_MAX:
        raise ValueError(f"cell_id must be {INT64_MAX} or less, got {cell}")
    return divmod(cell, CELL_ID_FACTOR)


def chunk_of_bounds(xmin, ymax):
    """DS-3: the (row, col) of the chunk that holds the cell with the corner (xmin, ymax)."""
    _check_lattice(xmin, ymax)
    row = int((Y0 - _exact(ymax)) // CHUNK)
    col = int((_exact(xmin) - X0) // CHUNK)
    if row < 0 or col < 0:
        raise ValueError(f"({xmin}, {ymax}) is outside the grid (chunk {row}, {col})")
    return row, col


def chunk_of_cell(row, col):
    """DS-3: the chunk (row, col) of a cell (row, col)."""
    return _index(row, "row") // CELLS_PER_CHUNK, _index(col, "col") // CELLS_PER_CHUNK


def chunk_bounds(row, col):
    """DS-3: (xmin, ymin, xmax, ymax) of a chunk."""
    row = _index(row, "row")
    col = _index(col, "col")
    xmin = X0 + CHUNK * col
    ymax = Y0 - CHUNK * row
    return xmin, ymax - CHUNK, xmin + CHUNK, ymax


def chunk_id(row, col):
    """DS-7: `r<row>_c<col>`."""
    return f"r{_index(row, 'row')}_c{_index(col, 'col')}"


def parse_chunk_id(text):
    """DS-7: the (row, col) of a chunk ID. A leading zero raises ValueError."""
    match = _CHUNK_ID_RE.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"not a chunk ID: {text!r}")
    return int(match.group(1)), int(match.group(2))


def pixel_of_point(x, y, chunk_row, chunk_col):
    """DS-4: the pixel (i, j) of a point in a chunk. A point outside the chunk raises."""
    x = _exact(x, "x")
    y = _exact(y, "y")
    xmin, _, _, ymax = chunk_bounds(chunk_row, chunk_col)
    i = int((ymax - y) // _PIXEL_EXACT)
    j = int((x - xmin) // _PIXEL_EXACT)
    if not (0 <= i < PIXELS_PER_CHUNK and 0 <= j < PIXELS_PER_CHUNK):
        raise ValueError(f"({x}, {y}) is outside the chunk r{chunk_row}_c{chunk_col}")
    return i, j


def check_naip_transform(transform, width, height):
    """DS-4: a NAIP chunk has a 0.5 m pixel, starts on a cell edge and is 1920 x 1920 pixels.

    `transform` has the 6 numbers (a, b, c, d, e, f) of rasterio/affine.
    """
    a, b, c, d, e, f = tuple(transform)[:6]
    if (a, b, d, e) != (PIXEL, 0, 0, -PIXEL):
        raise ValueError(f"NAIP pixel must be 0.5 m, north up: {(a, b, d, e)}")
    _check_lattice(c, f)
    if (width, height) != (PIXELS_PER_CHUNK, PIXELS_PER_CHUNK):
        raise ValueError(f"NAIP chunk must be 1920 x 1920 pixels, got {width} x {height}")


# ---------------------------------------------------------------- codes (DS-11 to DS-13)

NODATA = 0


class Code(NamedTuple):
    """One row of the code table: name, NLCD flag, USACE-M raster code (None for a tree code)."""

    code: int
    name: str
    nlcd: str
    usace_m: int | None
    colour: str


class Derived(NamedTuple):
    """One output code of DS-14: NLCD name and NLCD legend colour (None for code 0)."""

    name: str
    colour: str | None


# DS-16: photo code -> scheme.py raster code. The tree codes 41, 42, 43, 44 and 90 have no
# fixed target (DS-15).
USACE_M_MAP = {
    11: 1, 13: 1, 14: 1,
    25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
    31: 3, 35: 3,
    32: 4, 34: 4,
    33: 5,
    96: 8,
    95: 9,
    21: 10, 71: 10,
    81: 12,
    82: 13, 83: 13,
    52: 26,
    51: 17,
}
USACE_M_LABELS = {
    1: "M01", 2: "M02", 3: "M03", 4: "M04", 5: "M05", 8: "M08", 9: "M09", 10: "M10",
    12: "M12", 13: "M13", 17: "M17", 18: "M18", 19: "M19", 20: "M20", 21: "M21", 22: "M22",
    23: "M23", 24: "M24", 26: "M14-16u", 27: "M19-21u", 28: "M22-24u",
}
USACE_M_NOT_PRODUCED = frozenset({6, 7, 11, 14, 15, 16, 25, 29})

_TABLE = (
    (11, "Open water", "yes", "#466b9f"),
    (13, "River or stream", "new", "#5a86b8"),
    (14, "Pond", "new", "#2f4f7a"),
    (21, "Developed, open space (maintained grass)", "yes", "#dec5c5"),
    (25, "Building", "new", "#8b0000"),
    (26, "Paved road", "new", "#ff3b3b"),
    (27, "Unpaved road", "new", "#c9846b"),
    (28, "Parking or other paved", "new", "#e3aa9c"),
    (29, "Dam or levee", "new", "#7a3d3d"),
    (31, "Barren, type not known", "yes", "#b3ac9f"),
    (32, "Sand or beach", "new", "#e8dcb0"),
    (33, "Rock or cliff", "new", "#8f8a80"),
    (34, "Mud flat or drawdown shore", "new", "#a89b84"),
    (35, "Gravel or quarry", "new", "#c7c2b8"),
    (41, "Deciduous forest", "yes", "#68ab5f"),
    (42, "Evergreen forest", "yes", "#1c5f2c"),
    (43, "Mixed forest", "yes", "#b5c58f"),
    (44, "Forest, leaf type not known", "new", "#4f8a4a"),
    (51, "Dwarf shrub", "yes", "#af963c"),
    (52, "Shrub or scrub", "yes", "#ccb879"),
    (71, "Grassland or herbaceous", "yes", "#dfdfc2"),
    (81, "Pasture or hay", "yes", "#dcd939"),
    (82, "Cultivated crops", "yes", "#ab6c28"),
    (83, "Fallow or bare field", "new", "#c9a27a"),
    (90, "Woody wetland", "yes", "#b8d9eb"),
    (95, "Emergent herbaceous wetland", "yes", "#6c9fb8"),
    (96, "Aquatic bed", "new", "#4f7f99"),
)
CODES = tuple(Code(c, n, f, USACE_M_MAP.get(c), col) for c, n, f, col in _TABLE)
CODE_LIST = tuple(entry.code for entry in CODES)
_BY_CODE = {entry.code: entry for entry in CODES}

LEVEL1_CODES = {
    10: "Water", 20: "Developed", 30: "Barren", 40: "Forest", 50: "Shrubland",
    70: "Herbaceous", 80: "Planted or cultivated", 99: "Wetlands",
}

NLCD_LEGEND = {
    11: "#466b9f", 12: "#d1def8", 21: "#dec5c5", 22: "#d99282", 23: "#eb0000", 24: "#ab0000",
    31: "#b3ac9f", 41: "#68ab5f", 42: "#1c5f2c", 43: "#b5c58f", 51: "#af963c", 52: "#ccb879",
    71: "#dfdfc2", 72: "#d1d182", 73: "#a3cc51", 74: "#82ba9e", 81: "#dcd939", 82: "#ab6c28",
    90: "#b8d9eb", 95: "#6c9fb8",
}

_DERIVED_NAMES = {
    0: "No photo",
    11: "Open water",
    21: "Developed, open space",
    22: "Developed, low intensity",
    23: "Developed, medium intensity",
    24: "Developed, high intensity",
    31: "Barren land",
    41: "Deciduous forest",
    42: "Evergreen forest",
    43: "Mixed forest",
    52: "Shrub/scrub",
    71: "Grassland/herbaceous",
    81: "Pasture/hay",
    82: "Cultivated crops",
    90: "Woody wetlands",
    95: "Emergent herbaceous wetlands",
}
DERIVED_NLCD = {c: Derived(n, NLCD_LEGEND.get(c)) for c, n in _DERIVED_NAMES.items()}

# The Annual NLCD legend codes that can occur in the stack (DS-11).
_STACK_NLCD_CODES = frozenset({11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95})
_STACK_NLCD_NODATA = 250


def _code_int(code):
    """Return a code as an int. Only an int or a numpy integer is a code."""
    if isinstance(code, bool) or not isinstance(code, Integral):
        raise ValueError(f"a code must be an integer, got {code!r}")
    return int(code)


def level1(code):
    """DS-12: the level-1 code of a code. 99 for wetlands; a level-1 code gives itself."""
    code = _code_int(code)
    if code in LEVEL1_CODES:
        return code
    if code in _BY_CODE:
        return 99 if code // 10 == 9 else (code // 10) * 10
    raise ValueError(f"not a label code: {code!r}")


def is_label_code(code):
    """DS-12: true for the 27 codes and the 8 level-1 codes (35 codes)."""
    return code in _BY_CODE or code in LEVEL1_CODES


def colour(code):
    """DS-13: the colour of a code of the code table or of a derived NLCD code."""
    if code in _BY_CODE:
        return _BY_CODE[code].colour
    derived = DERIVED_NLCD.get(code)
    if derived is None or derived.colour is None:
        raise ValueError(f"no colour for code {code!r}")
    return derived.colour


def stack_nlcd_value(value):
    """DS-11: a stack NLCD value for a cell record. 250 (no data) becomes 0."""
    value = _code_int(value)
    if value == _STACK_NLCD_NODATA:
        return NODATA
    if value in _STACK_NLCD_CODES:
        return value
    raise ValueError(f"not an Annual NLCD value: {value!r}")


# ---------------------------------------------------------------- compositions (DS-14 to DS-16)

_SUM_LIMIT = 10**6  # 0.001 in units of 1e-9 (DS-20, D34)
_DEVELOPED = (25, 26, 27, 28, 29)
_TREE_LEAF = (41, 42, 43)
_TREE_ALL = (41, 42, 43, 44, 90)
_NLCD_GROUPS = {
    11: (11, 13, 14), 21: (21,), 31: (31, 32, 33, 34, 35), 41: (41,), 42: (42,),
    43: (43, 44), 52: (51, 52), 71: (71,), 81: (81,), 82: (82, 83), 90: (90,), 95: (95, 96),
}
_USACE_M_TREE_CLASSES = frozenset({18, 19, 20, 21, 22, 23, 24, 27, 28})
_CLOSED = {41: 24, 42: 23, 43: 22}
_OPEN = {41: 21, 42: 20, 43: 19}
FLAG_WOODY_WETLAND = 1
FLAG_MAINTAINED_GRASS = 128


def _share(value, name):
    value = _real(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return float(value)


def _gsum(values):
    """D34: a group share sum, rounded to 9 decimals."""
    return round(math.fsum(values), 9)


def _check_shares(shares, photo_share):
    """Check the input of a composition. Return (shares of the 27 codes, photo_share)."""
    if not isinstance(shares, Mapping):
        raise ValueError("shares must be a mapping from code to share")
    photo = _share(photo_share, "photo_share")
    full = dict.fromkeys(CODE_LIST, 0.0)
    for key, value in shares.items():
        if isinstance(key, bool) or key not in full:
            raise ValueError(f"not a code of the code table: {key!r}")
        full[key] = _share(value, f"share_{key}")
    if photo == 0:
        if any(full.values()):
            raise ValueError("photo_share is 0, so each share must be 0")
    elif abs(round(_gsum(full.values()) * 1e9) - 10**9) > _SUM_LIMIT:
        raise ValueError("the shares must sum to 1 +- 0.001 where photo_share > 0")
    return full, photo


def _rounded(full):
    """D34: each single share, rounded to 9 decimals, for a comparison or a tie."""
    return {c: round(v, 9) for c, v in full.items()}


def _developed(full):
    return _gsum(full[c] for c in _DEVELOPED)


def derived_nlcd(shares, photo_share):
    """DS-14: the derived NLCD code of a cell (0 without a photo)."""
    full, photo = _check_shares(shares, photo_share)
    if photo == 0:
        return 0
    intensity = _developed(full)
    if intensity >= 0.20:
        return 22 if intensity < 0.50 else 23 if intensity < 0.80 else 24
    sums = {out: _gsum(full[c] for c in members) for out, members in _NLCD_GROUPS.items()}
    return min(sums, key=lambda out: (-sums[out], out))


def derived_usace_m(shares, photo_share):
    """DS-15: the USACE-M raster code of a cell (0 without a photo)."""
    full, photo = _check_shares(shares, photo_share)
    if photo == 0:
        return 0
    if _developed(full) >= 0.20:
        return 2
    tree = _gsum(full[c] for c in _TREE_ALL)
    if tree >= 0.10:
        if tree < 0.15:
            return 18
        table = _CLOSED if tree > 0.45 else _OPEN
        single = _rounded(full)
        largest = max(single[c] for c in _TREE_LEAF)
        if largest == 0:
            return 28 if tree > 0.45 else 27
        leaders = [c for c in _TREE_LEAF if single[c] == largest]
        return table[leaders[0] if len(leaders) == 1 else 43]
    sums = {}
    for code, target in USACE_M_MAP.items():
        sums.setdefault(target, []).append(full[code])
    totals = {target: _gsum(values) for target, values in sums.items()}
    return min(totals, key=lambda target: (-totals[target], target))


def usace_m_flags(shares, photo_share):
    """DS-16: the flags of a cell. Bit 1 = woody wetland, bit 128 = maintained grass."""
    full, photo = _check_shares(shares, photo_share)
    code = derived_usace_m(full, photo)
    single = _rounded(full)
    flags = 0
    if code in _USACE_M_TREE_CLASSES and single[90] >= _gsum(single[c] for c in (41, 42, 43, 44)):
        flags |= FLAG_WOODY_WETLAND
    if code == 10 and single[21] >= single[71]:
        flags |= FLAG_MAINTAINED_GRASS
    return flags


def top_code(shares, photo_share):
    """DS-20: the code with the largest share; the lowest code on a tie; 0 without a photo."""
    full, photo = _check_shares(shares, photo_share)
    if photo == 0:
        return 0
    single = _rounded(full)
    return min(single, key=lambda c: (-single[c], c))


# ---------------------------------------------------------------- names and records (DS-5 to DS-28)

HISTORY_YEARS = tuple(range(1985, 2026))
PILOTS = {77: "GRAPEVINE_LAKE", 392: "RAYSTOWN_LAKE"}

RUN_ID_RE = r"\A[0-9]{8}-[0-9]{4}-[a-z0-9_]+\Z"
_RUN_ID = re.compile(RUN_ID_RE)
_PROJECT_NAME = re.compile(r"[A-Z0-9_]+", re.ASCII)
_PROJECT_FOLDER = re.compile(r"([1-9][0-9]*)_([A-Z0-9_]+)", re.ASCII)
_POLYGON_ID = re.compile(
    rf"(r{_NUM}_c{_NUM})_([0-9]{{8}}-[0-9]{{4}}-[a-z0-9_]+)_({_NUM})", re.ASCII
)
_TRAILING_BLANK = re.compile(r"[\s.]+\Z")

GPKG_LAYERS = ("polygons", "labels", "sample")
RESULTS_MAX_BYTES = 1_000_000
METADATA_KEYS = ("run_id", "code_version", "data_standard", "naip_year", "sources", "created")
PROJECT_SUBFOLDERS = ("polygons", "cells", "rasters", "labels", "sample", "sources", "baselines")
CHUNK_LIST = "chunks.parquet"
RESERVED_NAMES = frozenset({"aux", "con", "nul", "prn"})
ROOTS = {"semseg": "rw", "fgt": "ro", "stack": "ro"}
PART_SUFFIX = ".part"
EXIT_SHARE_MISSING = 3
APP_DB = "var/app.sqlite"
MODELS_DIR = "models"


def project_folder(pid, name):
    """DS-6: `<pid>_<NAME>`."""
    pid = _int(pid, "pid")
    if pid < 1:
        raise ValueError(f"pid must be 1 or more, got {pid}")
    if not isinstance(name, str) or _PROJECT_NAME.fullmatch(name) is None:
        raise ValueError(f"project name must match [A-Z0-9_]+, got {name!r}")
    return f"{pid}_{name}"


def parse_project_folder(text):
    """DS-6: the (pid, NAME) of a project folder name."""
    match = _PROJECT_FOLDER.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"not a project folder name: {text!r}")
    return int(match.group(1)), match.group(2)


def check_run_id(text):
    """DS-10: raise ValueError if the text is not a run ID (pattern and a real date and time)."""
    if not isinstance(text, str) or _RUN_ID.fullmatch(text) is None:
        raise ValueError(f"not a run ID: {text!r}")
    try:
        datetime.strptime(text[:13], "%Y%m%d-%H%M")
    except ValueError:
        raise ValueError(f"not a run ID (impossible date or time): {text!r}") from None


def run_id(when: datetime, method):
    """DS-10: `<YYYYMMDD-HHMM>-<method>`. `when` is a datetime (or a date: time 0000)."""
    if not isinstance(when, date):
        raise ValueError(f"when must be a datetime, got {when!r}")
    text = f"{when.strftime('%Y%m%d-%H%M')}-{method}"
    check_run_id(text)
    return text


def polygon_id(chunk, run, n):
    """DS-9: `<chunk>_<run_id>_<n>`. A polygon ID is not stable from run to run."""
    parse_chunk_id(chunk)
    check_run_id(run)
    return f"{chunk}_{run}_{_index(n, 'n')}"


def parse_polygon_id(text):
    """DS-9: the (chunk, run_id, n) of a polygon ID."""
    match = _POLYGON_ID.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"not a polygon ID: {text!r}")
    chunk, run, n = match.group(1), match.group(2), int(match.group(3))
    check_run_id(run)
    return chunk, run, n


def raster_profile(kind):
    """DS-17: the rasterio profile for `code`, `percent` or `value`."""
    if kind not in ("code", "percent", "value"):
        raise ValueError(f"raster kind must be code, percent or value, got {kind!r}")
    profile = {
        "driver": "GTiff",
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "crs": CRS,
        "dtype": "float32" if kind == "value" else "uint8",
    }
    if kind == "code":
        profile["nodata"] = NODATA
    return profile


def record_columns():
    """DS-20: the 164 columns of a cell record, in order."""
    columns = ["cell_id", "row", "col", "pid", "chunk", "naip_year", "naip_date", "photo_share"]
    columns += [f"share_{c}" for c in CODE_LIST]
    columns += ["top_code", "n_poly", "nlcd_derived", "usace_m_derived", "usace_m_flags"]
    for prefix in ("nlcd", "classf", "tcc"):
        columns += [f"{prefix}_{y}" for y in HISTORY_YEARS]
    columns.append("run_id")
    return columns


def check_name(part):
    """DS-22: raise ValueError for a bad name.

    A bad name is empty, a dot name, a name with a slash, a backslash or a control character
    (NUL too), or a reserved name. SMB strips trailing blanks and dots, so `aux .` is `aux`.
    """
    if not isinstance(part, str) or part in ("", ".", "..") or "/" in part or "\\" in part:
        raise ValueError(f"bad name for a folder or a file: {part!r}")
    if any(unicodedata.category(ch) == "Cc" for ch in part):
        raise ValueError(f"control character in a name: {part!r}")
    if _TRAILING_BLANK.sub("", part).split(".")[0].lower() in RESERVED_NAMES:
        raise ValueError(f"reserved name on the SMB share: {part!r}")


def project_dir(root, folder, kind):
    """DS-22: `<root>/<pid>_<NAME>/<kind>`. It does not touch the disk."""
    check_name(folder)
    check_name(kind)
    return PurePosixPath(root, folder, kind)


def run_dir(root, run):
    """DS-22: `<root>/runs/<run_id>`. It does not touch the disk."""
    check_name(run)
    return PurePosixPath(root, "runs", run)


def models_dir(root):
    """DS-22, DS-28: `<root>/models`. It does not touch the disk."""
    return PurePosixPath(root, MODELS_DIR)
