"""Edge attacks on `semseg.standard` (task 0.2). New FMEA rows FM-30 to FM-35.

A test marked `xfail(strict=True)` records an open fault. The mark must be removed when the
implementer fixes the fault (a strict xfail that passes fails the run). Read only: the stack
(for the extents of 77 and 392). No GPU. No write outside the repository.
"""
import datetime as dt
import math
import random
import re
import tomllib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from semseg import standard as S

ROOT = Path(__file__).resolve().parent.parent
X0, Y0 = S.X0, S.Y0
INT64_MAX = 2**63 - 1


def _stack_root():
    raw = tomllib.loads((ROOT / "data" / "REGISTRY.toml").read_text(encoding="utf-8"))["roots"]
    path = Path(raw["stack"]).expanduser()
    if not path.exists():
        pytest.fail(f"stack root missing: {path}. DS-26: exit code 3; this test must not skip.")
    return path


def _extents():
    """Return {pid: (row0, col0, height, width)} of nlcd_2022.tif of 77 and 392 (read only)."""
    import rasterio

    out = {}
    for pid, folder in ((77, "77_GRAPEVINE_LAKE"), (392, "392_RAYSTOWN_LAKE")):
        path = _stack_root() / folder / "nlcd_2022.tif"
        assert path.is_file(), path
        with rasterio.open(path) as ds:
            left, _, _, top = ds.bounds
            row0, col0 = S.cell_of_bounds(left, top)
            out[pid] = (row0, col0, ds.height, ds.width)
    return out


# ================================================================ grid: origin, negative, huge


def test_edge_origin_cell():
    assert S.cell_of_bounds(X0, Y0) == (0, 0)
    assert S.cell_of_point(X0, Y0) == (0, 0)
    assert S.cell_bounds(0, 0) == (X0, Y0 - 30, X0 + 30, Y0)
    assert S.cell_id(0, 0) == 0
    assert S.cell_rowcol(0) == (0, 0)
    assert S.chunk_of_bounds(X0, Y0) == (0, 0)
    assert S.chunk_bounds(0, 0) == (X0, Y0 - 960, X0 + 960, Y0)
    assert S.pixel_of_point(X0, Y0, 0, 0) == (0, 0)


@pytest.mark.parametrize("x,y", [
    (X0 - 30, Y0), (X0, Y0 + 30), (X0 - 30, Y0 + 30),
    (math.nextafter(float(X0), -math.inf), float(Y0)),
    (float(X0), math.nextafter(float(Y0), math.inf)),
])
def test_edge_outside_grid_raises(x, y):
    """A point or a corner above or left of the origin has a negative index: ValueError."""
    with pytest.raises(ValueError):
        S.cell_of_point(x, y)
    if S.on_lattice(x, y):
        with pytest.raises(ValueError):
            S.cell_of_bounds(x, y)
        with pytest.raises(ValueError):
            S.chunk_of_bounds(x, y)


def test_edge_negative_coordinates_are_fine_when_inside():
    # x is negative for all of 77 and part of 392; the column is still positive
    assert S.cell_of_bounds(-117345.0, 1111605.0) == (73440, 76608)
    assert S.cell_of_point(-117345.0, 1111605.0) == (73440, 76608)
    assert S.cell_of_point(-117345.0 + 29.999, 1111605.0 - 29.999) == (73440, 76608)
    assert S.cell_of_point(-117345.0 + 30, 1111605.0) == (73440, 76609)
    assert S.cell_of_point(-117345.0, 1111605.0 - 30) == (73441, 76608)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, False, None,
                                 "1", b"1", [1], np.array([1.0]), np.array(1.0), 1j])
def test_edge_grid_rejects_bad_scalars(bad):
    for fn in (S.on_lattice, S.cell_of_bounds, S.cell_of_point, S.chunk_of_bounds):
        with pytest.raises((ValueError, TypeError)):
            fn(bad, 0)
        with pytest.raises((ValueError, TypeError)):
            fn(0, bad)
    with pytest.raises((ValueError, TypeError)):
        S.pixel_of_point(bad, 0, 0, 0)


@pytest.mark.parametrize("off", [1e-9, -1e-9, 1e-6, 0.5, 1, 29.999999, 30 - 1e-9])
def test_edge_off_lattice_offsets_raise(off):
    x, y = -120135.0, 1111605.0
    assert S.on_lattice(x, y)
    for fn in (S.cell_of_bounds, S.chunk_of_bounds):
        with pytest.raises(ValueError):
            fn(x + off, y)
        with pytest.raises(ValueError):
            fn(x, y + off)
        with pytest.raises(ValueError):
            fn(x + off, y + off)


@pytest.mark.parametrize("x", [-120135.00000001, -120135.0000001, -120134.99999999,
                               -120135 + (0.1 + 0.2 - 0.3) * 1e9])
def test_edge_binary_error_inputs_raise(x):
    with pytest.raises(ValueError):
        S.cell_of_bounds(x, 1111605.0)


def test_edge_tiny_binary_noise_below_precision_is_the_same_float():
    # 0.1 + 0.2 - 0.3 is 5.5e-17: it vanishes at 1e5 m, so the sum IS the lattice float
    x = -120135.0 + (0.1 + 0.2 - 0.3)
    assert x == -120135.0 and S.on_lattice(x, 1111605.0)


def test_edge_cell_of_point_half_open_cell_with_floats():
    row, col = 5000, 7000
    xmin, _, xmax, ymax = S.cell_bounds(row, col)
    assert S.cell_of_point(float(xmin), float(ymax)) == (row, col)
    assert S.cell_of_point(float(xmax), float(ymax)) == (row, col + 1)
    assert S.cell_of_point(float(xmin), float(ymax) - 30) == (row + 1, col)
    assert S.cell_of_point(xmax - 1e-6, ymax - 30 + 1e-6) == (row, col)


def test_edge_numpy_scalars_and_python_ints():
    cid = S.cell_id(np.int64(73346), np.int32(76515))
    assert cid == 73346076515 and type(cid) is int
    assert S.cell_id(np.uint64(5), np.uint8(7)) == 5_000_007
    assert S.cell_rowcol(np.int64(73346076515)) == (73346, 76515)
    assert S.cell_of_bounds(np.float64(-117345.0), np.float64(1111605.0)) == (73440, 76608)
    assert S.cell_of_bounds(np.int64(-117345), np.int64(1111605)) == (73440, 76608)
    assert S.chunk_id(np.int64(3), np.int64(4)) == "r3_c4"
    with pytest.raises(ValueError):
        S.cell_id(np.float64(1.0), 2)
    with pytest.raises(ValueError):
        S.cell_id(np.array([1]), 2)
    with pytest.raises(ValueError):
        S.cell_id(np.bool_(True), 2)


def test_edge_bool_is_not_an_integer_or_a_number():
    for fn in (S.cell_id, S.cell_bounds, S.chunk_id, S.chunk_of_cell, S.chunk_bounds):
        with pytest.raises(ValueError):
            fn(True, 0)
        with pytest.raises(ValueError):
            fn(0, False)
    with pytest.raises(ValueError):
        S.cell_rowcol(True)
    with pytest.raises(ValueError):
        S.project_folder(True, "X")
    with pytest.raises(ValueError):
        S.polygon_id("r1_c1", "20261111-1200-x", True)


def test_edge_cell_id_col_limit_and_big_values():
    assert S.cell_id(0, 999_999) == 999_999
    with pytest.raises(ValueError):
        S.cell_id(0, 1_000_000)
    assert S.cell_rowcol(1_000_000) == (1, 0)
    assert S.cell_rowcol(999_999) == (0, 999_999)
    with pytest.raises(ValueError):
        S.cell_rowcol(-1)
    # the largest row whose ID is an int64
    row = INT64_MAX // 1_000_000
    assert S.cell_id(row, 0) <= INT64_MAX
    assert S.cell_rowcol(S.cell_id(row, 0)) == (row, 0)
    assert S.cell_id(row, 775_807) == INT64_MAX
    assert S.cell_rowcol(INT64_MAX) == (row, 775_807)
    with pytest.raises(ValueError):
        S.cell_id(row, 775_808)


def test_edge_cell_of_bounds_with_col_past_the_id_range_is_split_from_cell_id():
    """cell_of_bounds does not check col < 1 000 000 (spec A1); cell_id does, so no bad ID forms."""
    xmin = X0 + 30 * 1_000_000
    assert S.cell_of_bounds(xmin, Y0) == (0, 1_000_000)
    with pytest.raises(ValueError):
        S.cell_id(0, 1_000_000)


def test_edge_cell_id_int64_limit():
    """DS-8: the cell ID is int64. A row of 9 223 372 036 855 gives 9.22e18 > 2**63 - 1."""
    row = INT64_MAX // 1_000_000 + 1
    with pytest.raises(ValueError):
        S.cell_id(row, 0)


def test_edge_cell_rowcol_int64_limit():
    with pytest.raises(ValueError):
        S.cell_rowcol(INT64_MAX + 1)


def test_edge_cell_id_absurd_row():
    with pytest.raises(ValueError):
        S.cell_id(10**30, 5)


# ---- float rounding inside the lattice test and the floor (FM-28, FM-26)


def test_edge_on_lattice_one_ulp_off_x():
    x = math.nextafter(-120135.0, 0.0)
    assert Fraction(x) != Fraction(-120135)
    assert not S.on_lattice(x, 1111605.0)


def test_edge_on_lattice_one_ulp_off_y():
    y = math.nextafter(1111605.0, 0.0)
    assert not S.on_lattice(-120135.0, y)


def test_edge_on_lattice_float32_off_lattice():
    x = np.float32(-120134.99)  # 0.0078 m off the lattice
    assert float(x) != -120135.0
    assert not S.on_lattice(x, np.float32(1111605.0))
    with pytest.raises(ValueError):
        S.cell_of_bounds(x, np.float32(1111605.0))


@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_edge_on_lattice_int64_overflow():
    x = INT64_MAX - ((INT64_MAX - X0) % 30)  # truly on the lattice
    assert (x - X0) % 30 == 0
    assert S.on_lattice(np.int64(x), Y0)


def test_edge_cell_of_point_row_one_ulp_inside_cell():
    k = 100_000
    y = math.nextafter(float(Y0 - 30 * k), math.inf)  # just above the top edge of row k
    exact = math.floor((Fraction(Y0) - Fraction(y)) / 30)
    assert exact == k - 1
    assert S.cell_of_point(float(X0), y)[0] == k - 1


def test_edge_cell_of_point_col_one_ulp_inside_cell():
    k = 100_000
    x = math.nextafter(float(X0 + 30 * k), -math.inf)  # just left of the left edge of col k
    exact = math.floor((Fraction(x) - Fraction(X0)) / 30)
    assert exact == k - 1
    assert S.cell_of_point(x, float(Y0))[1] == k - 1


def test_edge_cell_of_point_agrees_with_exact_arithmetic_on_random_floats():
    """Away from the 1 ULP band the floor is right (random points, exact oracle)."""
    rnd = random.Random(7)
    for _ in range(20000):
        x = rnd.uniform(X0, X0 + 30 * 200_000)
        y = rnd.uniform(Y0 - 30 * 200_000, Y0)
        want = (math.floor((Fraction(Y0) - Fraction(y)) / 30),
                math.floor((Fraction(x) - Fraction(X0)) / 30))
        assert S.cell_of_point(x, y) == want, (x, y)


# ================================================================ round trips on 77 and 392


def test_edge_cell_roundtrip_over_extents():
    for pid, (row0, col0, height, width) in _extents().items():
        rnd = random.Random(pid)
        cells = {(row0, col0), (row0 + height - 1, col0 + width - 1),
                 (row0, col0 + width - 1), (row0 + height - 1, col0)}
        cells |= {(row0 + rnd.randrange(height), col0 + rnd.randrange(width)) for _ in range(5000)}
        cells |= {(row0 + r, col0) for r in range(0, height, 7)}
        cells |= {(row0, col0 + c) for c in range(0, width, 7)}
        for row, col in cells:
            cid = S.cell_id(row, col)
            assert S.cell_rowcol(cid) == (row, col)
            xmin, ymin, xmax, ymax = S.cell_bounds(row, col)
            assert S.cell_of_bounds(xmin, ymax) == (row, col)
            assert S.cell_of_bounds(float(xmin), float(ymax)) == (row, col)
            assert S.cell_of_point(xmin, ymax) == (row, col)
            assert S.cell_of_point(xmin + 15, ymax - 15) == (row, col)
            assert S.cell_of_point(xmax - 0.25, ymin + 0.25) == (row, col)
            assert (xmax - xmin, ymax - ymin) == (30, 30)
            crow, ccol = S.chunk_of_cell(row, col)
            assert S.chunk_of_bounds(xmin, ymax) == (crow, ccol)
            cx0, cy0, cx1, cy1 = S.chunk_bounds(crow, ccol)
            assert cx0 <= xmin and xmax <= cx1 and cy0 <= ymin and ymax <= cy1
            assert S.parse_chunk_id(S.chunk_id(crow, ccol)) == (crow, ccol)
            assert S.on_lattice(cx0, cy1)


def test_edge_every_cell_of_a_chunk_maps_back_to_its_chunk():
    for crow, ccol in ((2295, 2394), (1284, 4084), (0, 0)):
        seen = set()
        for r in range(32):
            for c in range(32):
                row, col = crow * 32 + r, ccol * 32 + c
                assert S.chunk_of_cell(row, col) == (crow, ccol)
                seen.add(S.cell_id(row, col))
        assert len(seen) == 1024


def test_edge_chunk_pixel_corners():
    xmin, ymin, xmax, ymax = S.chunk_bounds(1284, 4084)
    assert S.pixel_of_point(xmin, ymax, 1284, 4084) == (0, 0)
    assert S.pixel_of_point(xmax - 0.5, ymin + 0.5, 1284, 4084) == (1919, 1919)
    assert S.pixel_of_point(xmax - 0.25, ymin + 0.25, 1284, 4084) == (1919, 1919)
    for x, y in ((xmax, ymax), (xmin, ymin), (xmax, ymin)):
        with pytest.raises(ValueError):
            S.pixel_of_point(x, y, 1284, 4084)


# ================================================================ naip transform


def test_edge_naip_transform_variants():
    xmin, ymin, xmax, ymax = S.chunk_bounds(10, 10)
    good = (0.5, 0, float(xmin), 0, -0.5, float(ymax))
    S.check_naip_transform(good, 1920, 1920)
    S.check_naip_transform(np.array(good), 1920, 1920)
    S.check_naip_transform(good + (0, 0, 1), 1920, 1920)  # affine.Affine is a 9-tuple
    for bad in [(0.5000001, 0, xmin, 0, -0.5, ymax), (0.5, 0, xmin, 0, 0.5, ymax),
                (0.5, 1e-9, xmin, 0, -0.5, ymax), (0.5, 0, xmin, 1e-9, -0.5, ymax),
                (0.5, 0, xmin + 1, 0, -0.5, ymax), (0.5, 0, xmin, 0, -0.5, ymax + 0.5),
                (0.5, 0, float("nan"), 0, -0.5, ymax), (0.5, 0, xmin, 0, -0.5),
                (xmin, 0.5, 0, ymax, 0, -0.5)]:  # GDAL order
        with pytest.raises(ValueError):
            S.check_naip_transform(bad, 1920, 1920)
    for w, h in ((1919, 1920), (1920, 1921), (0, 0), (-1920, 1920), (None, 1920), ("1920", 1920)):
        with pytest.raises(ValueError):
            S.check_naip_transform(good, w, h)


# ================================================================ regex IDs


@pytest.mark.parametrize("text", [
    "r1_c2\n", "r1_c2\r\n", "\nr1_c2", " r1_c2", "r1_c2 ", "r1_c2\x00", "﻿r1_c2", "R1_c2",
    "r1_C2", "r_c2", "r1_c", "r-1_c2", "r1_c-2", "r1.0_c2", "r01_c2", "r1_c02", "r00_c0",
    "r1_c2_", "r1__c2", "r1_c2​", "r１_c2", "١", "r1_c2<script>", "r1_c2;rm -rf",
    "=cmd", "", "r" * 100_000, "r" + "9" * 100_000 + "_c1", "r1_c" + "٣" * 3,
])
def test_edge_parse_chunk_id_rejects(text):
    with pytest.raises(ValueError):
        S.parse_chunk_id(text)


@pytest.mark.parametrize("bad", [None, 1, b"r1_c2", ["r1_c2"], 1.5])
def test_edge_parse_chunk_id_wrong_type(bad):
    with pytest.raises(ValueError):
        S.parse_chunk_id(bad)


def test_edge_parse_chunk_id_zero_and_big():
    assert S.parse_chunk_id("r0_c0") == (0, 0)
    assert S.parse_chunk_id("r10_c100") == (10, 100)
    big = 10**18
    assert S.parse_chunk_id(S.chunk_id(big, big)) == (big, big)


def test_edge_parse_chunk_id_unicode_digit_after_ascii():
    """'r1' + ARABIC-INDIC DIGIT THREE parses as row 13: the ID does not round trip."""
    with pytest.raises(ValueError):
        S.parse_chunk_id("r1٣_c0")


def test_edge_parse_project_folder_unicode_digit():
    with pytest.raises(ValueError):
        S.parse_project_folder("1٣_GRAPEVINE_LAKE")


def test_edge_run_id_unicode_digits():
    with pytest.raises(ValueError):
        S.check_run_id("٢٠٢٦١١١١-1200-x")


def test_edge_polygon_id_roundtrip_is_exact():
    """polygon_id(*parse_polygon_id(t)) == t for each accepted t, or parse raises."""
    for text in ("r01_c2_20261111-1200-x_5", "r1_c2_20261111-1200-x_007",
                 "r1٣_c2_20261111-1200-x_5"):
        try:
            parts = S.parse_polygon_id(text)
        except ValueError:
            continue
        assert S.polygon_id(*parts) == text, text


@pytest.mark.parametrize("text", [
    "r1_c2_20261111-1200-x_5\n", "r1_c2_20261111-1200-x_", "r1_c2_20261111-1200-X_5",
    "r1_c2_2026111-1200-x_5", "r1_c2_20261111-120-x_5", "r1_c2_20261111-1200-_5",
    "r1_c2_20261111-1200-x_-1", "r1_c2_20261111-1200-x_5 ", "", "r1_c2", None,
])
def test_edge_parse_polygon_id_rejects(text):
    with pytest.raises(ValueError):
        S.parse_polygon_id(text)


def test_edge_polygon_id_round_trip_with_underscore_method():
    pid = S.polygon_id("r1_c2", "20261111-1200-sam2_auto_5", 3)
    assert pid == "r1_c2_20261111-1200-sam2_auto_5_3"
    assert S.parse_polygon_id(pid) == ("r1_c2", "20261111-1200-sam2_auto_5", 3)
    assert S.polygon_id("r0_c0", "20261111-1200-x", 0).endswith("_0")
    for bad_n in (-1, 1.0, "1", None, True):
        with pytest.raises(ValueError):
            S.polygon_id("r1_c2", "20261111-1200-x", bad_n)
    for bad_chunk in ("r01_c2", "r1_c2\n", "x", ""):
        with pytest.raises(ValueError):
            S.polygon_id(bad_chunk, "20261111-1200-x", 1)


@pytest.mark.parametrize("text", [
    "20261111-1200-x\n", "20261111-1200-x ", " 20261111-1200-x", "20261111-1200-", "20261111-1200",
    "20261111-1200-X", "20261111-1200-a-b", "20261111-1200-a b", "20261111-1200-é",
    "20261111_1200-x", "2026111-1200-x", "202611111-1200-x", "20261111-12000-x", "",
    "20261111-1200-x\r\n", "﻿20261111-1200-x", "20261111-1200-../x", "20261111-1200-x/y",
    "20261111-1200-" + "a" * 5 + "\x00",
])
def test_edge_check_run_id_rejects(text):
    with pytest.raises(ValueError):
        S.check_run_id(text)


@pytest.mark.parametrize("bad", [None, 20261111, b"20261111-1200-x", ["a"]])
def test_edge_check_run_id_wrong_type(bad):
    with pytest.raises(ValueError):
        S.check_run_id(bad)


def test_edge_run_id_dates_and_formats():
    assert S.run_id(dt.datetime(2024, 2, 29, 23, 59), "m") == "20240229-2359-m"
    assert S.run_id(dt.datetime(2025, 12, 31, 23, 59, 59), "m") == "20251231-2359-m"
    assert S.run_id(dt.datetime(2026, 1, 1, 0, 0), "m_1") == "20260101-0000-m_1"
    assert S.run_id(dt.datetime(2026, 3, 8, 2, 30, tzinfo=dt.timezone.utc), "m")
    assert S.run_id(dt.date(2026, 11, 11), "m") == "20261111-0000-m"
    for year in (1, 999):  # strftime does not pad: 7 digits, the regex refuses
        with pytest.raises(ValueError):
            S.run_id(dt.datetime(year, 1, 1), "m")
    for method in ("", "X", "a-b", "a b", "é", "a/b", "..", "a\n"):
        with pytest.raises(ValueError):
            S.run_id(dt.datetime(2026, 1, 1), method)


@pytest.mark.parametrize("text", ["20261341-1200-x", "20260230-1200-x", "20260101-2561-x",
                                  "00000000-0000-x", "20261111-2400-x"])
def test_edge_check_run_id_impossible_date(text):
    with pytest.raises(ValueError):
        S.check_run_id(text)


@pytest.mark.parametrize("when", ["2026-11-11", 20261111, None])
def test_edge_run_id_wrong_type_is_value_error(when):
    with pytest.raises((ValueError, TypeError)):
        S.run_id(when, "m")


def test_edge_run_id_re_constant_with_re_match():
    """RUN_ID_RE is public (spec A5). A consumer that uses re.match accepts a trailing newline."""
    assert re.match(S.RUN_ID_RE, "20261111-1200-x\n") is None


@pytest.mark.parametrize("pid,name", [(0, "X"), (-1, "X"), (1.0, "X"), ("1", "X"), (1, ""),
                                      (1, "x"), (1, "A B"), (1, "A-B"), (1, "A\n"), (1, "Ä"),
                                      (1, "<script>"), (1, "=cmd"), (1, ".."), (1, "A/B"),
                                      (1, None), (1, "田中"), (1, "A\x00")])
def test_edge_project_folder_rejects(pid, name):
    with pytest.raises(ValueError):
        S.project_folder(pid, name)


@pytest.mark.parametrize("text", ["01_X", "0_X", "1_", "_X", "1X", "1_x", "1_X\n", " 1_X", "1_X ",
                                  "-1_X", "1_X/..", "1_A B", "", None, 77, "1.5_X", "１_X"])
def test_edge_parse_project_folder_rejects(text):
    with pytest.raises(ValueError):
        S.parse_project_folder(text)


def test_edge_project_folder_roundtrip_and_numpy():
    assert S.parse_project_folder(S.project_folder(np.int64(77), "GRAPEVINE_LAKE")) == (
        77, "GRAPEVINE_LAKE")
    assert S.parse_project_folder("10_X_1_2") == (10, "X_1_2")
    assert S.project_folder(1, "AUX") == "1_AUX"  # a digit prefix makes it a legal name


# ================================================================ codes and colours


def test_edge_derived_outputs_all_have_colour_and_label():
    rnd = random.Random(3)
    outs_n, outs_m, outs_top = set(), set(), set()
    for _ in range(4000):
        k = rnd.randint(1, 5)
        codes = rnd.sample(S.CODE_LIST, k)
        w = [rnd.random() + 0.01 for _ in codes]
        total = sum(w)
        sh = {c: x / total for c, x in zip(codes, w)}
        outs_n.add(S.derived_nlcd(sh, 1.0))
        outs_m.add(S.derived_usace_m(sh, 1.0))
        outs_top.add(S.top_code(sh, 1.0))
        flags = S.usace_m_flags(sh, 1.0)
        assert flags in (0, 1, 128)
    for code in S.CODE_LIST:  # every pure cell
        outs_n.add(S.derived_nlcd({code: 1.0}, 1.0))
        outs_m.add(S.derived_usace_m({code: 1.0}, 1.0))
    assert outs_n <= set(S.DERIVED_NLCD)
    assert all(S.colour(c) for c in outs_n - {0})
    assert outs_m <= set(S.USACE_M_LABELS)
    assert not outs_m & S.USACE_M_NOT_PRODUCED
    assert outs_top <= set(S.CODE_LIST)


@pytest.mark.parametrize("code", [0, -1, 1, 15, 22.5, 99, 10, 250, None, "11", True, float("nan")])
def test_edge_colour_rejects(code):
    with pytest.raises(ValueError):
        S.colour(code)


def test_edge_colour_hex_format_and_derived_shares_nlcd_legend():
    for entry in S.CODES:
        assert re.fullmatch(r"#[0-9a-f]{6}", entry.colour)
    for code in (22, 23, 24):
        assert S.colour(code) == S.NLCD_LEGEND[code]
        assert S.colour(code) not in {e.colour for e in S.CODES}
    assert S.colour(np.int64(11)) == "#466b9f"


@pytest.mark.parametrize("bad", [0, -1, 1, 16, 91, 100, 15, 56, None, "41", True, float("nan"),
                                 22, 23, 24, 98, 60])
def test_edge_level1_rejects(bad):
    with pytest.raises(ValueError):
        S.level1(bad)


def test_edge_level1_values_and_numpy():
    assert S.level1(np.int64(96)) == 99 and S.level1(90) == 99 and S.level1(99) == 99
    assert S.level1(np.int64(41)) == 40
    assert S.is_label_code(99) and not S.is_label_code(91) and not S.is_label_code(0)
    assert not S.is_label_code(None) and not S.is_label_code("41")


def test_edge_level1_float_code_returns_int():
    """FM-35: only an int (or a numpy integer) is a code; a float raises."""
    with pytest.raises(ValueError):
        S.level1(41.0)


def test_edge_stack_nlcd_value_float_returns_int():
    with pytest.raises(ValueError):
        S.stack_nlcd_value(11.0)


def test_edge_stack_nlcd_value_array():
    with pytest.raises((ValueError, TypeError)):
        S.stack_nlcd_value(np.array([250]))


def test_edge_stack_nlcd_nodata_and_legend():
    assert S.stack_nlcd_value(250) == 0 and S.stack_nlcd_value(np.uint8(250)) == 0
    assert S.stack_nlcd_value(np.int64(41)) == 41
    for bad in (0, 1, 10, 249, 251, 255, -1, 256, 2**64, None, "250", "41", float("nan"),
                float("inf"), True, 91, 96, 13, 44):
        with pytest.raises((ValueError, TypeError)):
            S.stack_nlcd_value(bad)
    with pytest.raises(ValueError):  # FM-35: a float is not a code, even 250.0
        S.stack_nlcd_value(250.0)


# ================================================================ derived classes: limits


def _two(code_a, a, code_b, b):
    return {code_a: a, code_b: b}


@pytest.mark.parametrize("limit,below,at", [(0.20, 11, 22), (0.50, 22, 23), (0.80, 23, 24)])
def test_edge_derived_nlcd_limit_and_ulp(limit, below, at):
    """One developed code at the limit and 1 ULP each side (the rest is one code, no sum rounds)."""
    other = 11  # the rest is water: it wins the group step only when I < 0.20
    up, dn = math.nextafter(limit, 1.0), limit - 2e-9  # D34: sums round to 9 decimals
    assert S.derived_nlcd({26: limit, other: 1 - limit}, 1.0) == at
    assert S.derived_nlcd({26: up, other: 1 - up}, 1.0) == at
    assert S.derived_nlcd({26: math.nextafter(limit, 0.0), other: 1 - limit}, 1.0) == at
    assert S.derived_nlcd({26: dn, other: 1 - dn}, 1.0) == below


@pytest.mark.parametrize("t,want", [
    (0.45, 21), (math.nextafter(0.45, 1.0), 21), (0.45 + 2e-9, 24), (math.nextafter(0.45, 0.0), 21),
    (0.15, 21), (0.15 - 2e-9, 18), (math.nextafter(0.15, 0.0), 21), (math.nextafter(0.15, 1.0), 21),
    (0.10, 18), (0.10 - 2e-9, None), (math.nextafter(0.10, 0.0), 18), (math.nextafter(0.10, 1.0), 18),
])
def test_edge_usace_m_tree_limits_and_ulp(t, want):
    """T from code 41 alone; the rest is water 11, so the non-tree step gives M01 (1)."""
    got = S.derived_usace_m({41: t, 11: 1 - t}, 1.0)
    assert got == (1 if want is None else want)


@pytest.mark.parametrize("i,want", [(0.20, 2), (math.nextafter(0.20, 1.0), 2),
                                    (math.nextafter(0.20, 0.0), 2), (0.20 - 2e-9, 24)])
def test_edge_usace_m_developed_first_at_ulp(i, want):
    """D33: I >= 0.20 gives M02 even when the tree share is large."""
    assert S.derived_usace_m({25: i, 41: 1 - i}, 1.0) == want


def test_edge_usace_m_flags_and_top_code_ties():
    assert S.top_code({41: 0.5, 42: 0.5}, 1.0) == 41
    assert S.top_code({82: 0.5, 11: 0.5}, 1.0) == 11
    assert S.top_code({90: 1.0}, 1.0) == 90
    assert S.top_code({}, 0.0) == 0
    assert S.top_code({41: 0.0}, 0.0) == 0
    assert S.derived_nlcd({41: 0.5, 42: 0.5}, 1.0) == 41
    assert S.derived_nlcd({90: 1.0}, 1.0) == 90
    assert S.derived_nlcd({44: 1.0}, 1.0) == 43
    assert S.derived_nlcd({41: 0.4, 44: 0.6}, 1.0) == 43
    assert S.derived_nlcd({41: 0.6, 44: 0.4}, 1.0) == 41
    assert S.derived_usace_m({90: 1.0}, 1.0) == 28
    assert S.derived_usace_m({44: 1.0}, 1.0) == 28
    assert S.derived_usace_m({44: 0.3, 11: 0.7}, 1.0) == 27
    assert S.derived_usace_m({41: 0.2, 44: 0.5, 11: 0.3}, 1.0) == 24
    assert S.derived_usace_m({41: 0.5, 42: 0.5}, 1.0) == 22
    assert S.derived_usace_m({41: 0.04, 42: 0.04, 43: 0.04, 11: 0.88}, 1.0) == 18
    assert S.usace_m_flags({90: 1.0}, 1.0) == 1
    assert S.usace_m_flags({41: 0.5, 90: 0.5}, 1.0) == 1  # 90 equals 41..44
    assert S.usace_m_flags({41: 0.6, 90: 0.4}, 1.0) == 0
    assert S.usace_m_flags({21: 0.5, 71: 0.5}, 1.0) == 128
    assert S.usace_m_flags({21: 0.4, 71: 0.6}, 1.0) == 0
    assert S.usace_m_flags({21: 1.0}, 1.0) == 128
    assert S.usace_m_flags({71: 1.0}, 1.0) == 0
    assert S.usace_m_flags({}, 0.0) == 0


def test_edge_not_produced_codes_never_come_out_for_pure_cells():
    for code in S.CODE_LIST:
        assert S.derived_usace_m({code: 1.0}, 1.0) not in S.USACE_M_NOT_PRODUCED


# ---- binary error on the limits and ties (FM-27)


def _dec(shares):
    return {c: Decimal(str(v)) for c, v in shares.items()}


def _oracle_nlcd(shares):
    d = _dec(shares)
    g = lambda cs: sum((d.get(c, Decimal(0)) for c in cs), Decimal(0))  # noqa: E731
    i = g((25, 26, 27, 28, 29))
    if i >= Decimal("0.20"):
        return 22 if i < Decimal("0.50") else 23 if i < Decimal("0.80") else 24
    sums = {o: g(m) for o, m in S._NLCD_GROUPS.items()}
    return min(sums, key=lambda o: (-sums[o], o))


def _oracle_m(shares):
    d = _dec(shares)
    g = lambda cs: sum((d.get(c, Decimal(0)) for c in cs), Decimal(0))  # noqa: E731
    if g((25, 26, 27, 28, 29)) >= Decimal("0.20"):
        return 2
    t = g((41, 42, 43, 44, 90))
    if t >= Decimal("0.10"):
        if t < Decimal("0.15"):
            return 18
        closed = t > Decimal("0.45")
        m = max(d.get(c, Decimal(0)) for c in (41, 42, 43))
        if m == 0:
            return 28 if closed else 27
        lead = [c for c in (41, 42, 43) if d.get(c, Decimal(0)) == m]
        tab = S._CLOSED if closed else S._OPEN
        return tab[lead[0] if len(lead) == 1 else 43]
    sums = {}
    for code, tgt in S.USACE_M_MAP.items():
        sums[tgt] = sums.get(tgt, Decimal(0)) + d.get(code, Decimal(0))
    return min(sums, key=lambda k: (-sums[k], k))


_NLCD_BINARY = [
    {25: 0.02, 26: 0.18, 11: 0.80},             # I = 0.20 in decimal, 0.19999999999999998 float
    {25: 0.70, 26: 0.10, 11: 0.20},             # I = 0.80 -> 24; float 0.7999999999999999 -> 23
    {35: 0.10, 33: 0.35, 13: 0.10, 90: 0.45},   # tie 31 / 90 -> 31; float sum < 0.45 -> 90
]
_M_BINARY = [
    {41: 0.01, 42: 0.09, 11: 0.90},             # T = 0.10 -> 18
    {41: 0.03, 42: 0.14, 43: 0.28, 11: 0.55},   # T = 0.45 -> open (19); float 0.45000000000000007
    {41: 0.01, 42: 0.07, 43: 0.07, 11: 0.85},   # T = 0.15 -> open; float 0.15000000000000002 ok
    {21: 0.45, 71: 0.0, 14: 0.45, 35: 0.1},     # tie of M10 / M01 -> 1 (lowest); exact floats
]


@pytest.mark.parametrize("shares", _NLCD_BINARY)
def test_edge_derived_nlcd_decimal_shares_at_limits(shares):
    """The decimal sum is the limit or the tie; the float sum is 1 ULP off."""
    assert S.derived_nlcd(shares, 1.0) == _oracle_nlcd(shares)


@pytest.mark.parametrize("shares", _M_BINARY[:2])
def test_edge_usace_m_decimal_shares_at_limits(shares):
    assert S.derived_usace_m(shares, 1.0) == _oracle_m(shares)


def test_edge_usace_m_decimal_shares_that_float_gets_right():
    for shares in _M_BINARY[2:]:
        assert S.derived_usace_m(shares, 1.0) == _oracle_m(shares)


def test_edge_pixel_count_shares_agree_with_integer_arithmetic():
    """Shares are pixel counts / 3600. Compare with exact integer arithmetic (fuzz, seed fixed)."""
    n = 3600
    rnd = random.Random(1)
    bad = 0
    for _ in range(20000):
        k = rnd.choice([2, 3, 4])
        cs = rnd.sample(S.CODE_LIST, k)
        cuts = sorted(rnd.sample(range(1, n // 180), k - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [n // 180])]
        counts = {c: p * 180 for c, p in zip(cs, parts)}
        fr = {c: Fraction(v, n) for c, v in counts.items()}
        sh = {c: v / n for c, v in counts.items()}
        i = sum((fr.get(c, 0) for c in (25, 26, 27, 28, 29)), Fraction(0))
        if i >= Fraction(1, 5):
            want = 22 if i < Fraction(1, 2) else 23 if i < Fraction(4, 5) else 24
        else:
            sums = {o: sum((fr.get(c, 0) for c in m), Fraction(0))
                    for o, m in S._NLCD_GROUPS.items()}
            want = min(sums, key=lambda o: (-sums[o], o))
        bad += S.derived_nlcd(sh, 1.0) != want
    assert bad == 0, f"{bad} of 20000 cells are classed differently from the exact integer rule"


# ================================================================ shares input


def test_edge_shares_input_errors():
    ok = {11: 1.0}
    assert S.derived_nlcd(ok, 1.0) == 11
    for bad_shares in (None, [], [(11, 1.0)], "11", 5, np.array([1.0])):
        with pytest.raises(ValueError):
            S.derived_nlcd(bad_shares, 1.0)
    for key in (0, 22, 23, 24, 10, 40, 99, 15, 56, 91, "11", None, (11,), True, -11, 110, 5):
        with pytest.raises(ValueError):
            S.derived_nlcd({11: 0.5, key: 0.5}, 1.0)
    for value in (-0.1, -1e-300, 1.0000001, 2, float("nan"), float("inf"), -float("inf"), True,
                  "0.5", None, np.array([0.5]), [0.5], 1j):
        with pytest.raises((ValueError, TypeError)):
            S.derived_nlcd({11: value, 13: 0.5}, 1.0)
    for ps in (-0.1, 1.0000001, float("nan"), float("inf"), True, None, "1", np.array([1.0])):
        with pytest.raises((ValueError, TypeError)):
            S.derived_nlcd(ok, ps)


def test_edge_shares_sum_tolerance():
    assert S.derived_nlcd({11: 0.5, 13: 0.4995}, 1.0) == 11   # sum 0.9995
    assert S.derived_nlcd({11: 0.5, 13: 0.5005}, 1.0) == 11   # sum 1.0005
    assert S.derived_nlcd({11: 0.501, 13: 0.5}, 1.0) == 11    # sum 1.001 (accepted, float low)
    for a, b in ((0.5, 0.498), (0.5, 0.502), (0.4, 0.4), (0.9, 0.2), (0.0, 0.0), (0.001, 0.0)):
        with pytest.raises(ValueError):
            S.derived_nlcd({11: a, 13: b}, 1.0)
    with pytest.raises(ValueError):
        S.derived_nlcd({}, 1.0)
    # 2.7 % of 3 * 0.1: the sum 0.30000000000000004 case, three thirds
    third = 1 / 3
    assert S.derived_nlcd({11: third, 13: third, 14: third}, 1.0) == 11


def test_edge_shares_sum_boundary_is_symmetric():
    """Sum 1.001 is accepted and sum 0.999 is rejected: the bound is not symmetric."""
    hi = lo = True
    try:
        S.derived_nlcd({11: 0.501, 13: 0.5}, 1.0)
    except ValueError:
        hi = False
    try:
        S.derived_nlcd({11: 0.499, 13: 0.5}, 1.0)
    except ValueError:
        lo = False
    assert hi == lo


def test_edge_photo_share_zero_and_small():
    assert S.derived_nlcd({}, 0.0) == 0
    assert S.derived_nlcd({}, -0.0) == 0
    assert S.derived_nlcd({}, np.float32(0.0)) == 0
    assert S.derived_usace_m({}, 0) == 0
    assert S.usace_m_flags({}, 0.0) == 0
    assert S.derived_nlcd({41: 0.0, 11: 0.0}, 0.0) == 0
    with pytest.raises(ValueError):
        S.derived_nlcd({41: 1e-300}, 0.0)
    assert S.derived_nlcd({41: 1.0}, 1e-300) == 41
    assert S.derived_usace_m({41: 1.0}, 5e-324) == 24
    with pytest.raises(ValueError):
        S.derived_nlcd({}, 1e-300)  # photo > 0 but no shares


def test_edge_compositions_do_not_mutate_input_and_are_repeatable():
    sh = {41: 0.5, 42: 0.25, 25: 0.0, 44: 0.25}
    snap = dict(sh)
    for fn in (S.derived_nlcd, S.derived_usace_m, S.usace_m_flags, S.top_code):
        first = fn(sh, 1.0)
        assert fn(sh, 1.0) == first
        assert sh == snap and list(sh) == list(snap)
    # key order does not change the answer
    rev = dict(reversed(list(sh.items())))
    for fn in (S.derived_nlcd, S.derived_usace_m, S.usace_m_flags, S.top_code):
        assert fn(rev, 1.0) == fn(sh, 1.0)


def test_edge_numpy_values_and_keys_in_shares():
    sh = {np.int64(41): np.float64(0.5), np.int32(42): np.float32(0.5)}
    assert S.derived_nlcd(sh, np.float64(1.0)) == 41
    assert S.top_code(sh, np.float32(1.0)) == 41
    from types import MappingProxyType
    assert S.derived_nlcd(MappingProxyType({11: 1.0}), 1.0) == 11


def test_edge_raster_profile_is_a_fresh_dict():
    p = S.raster_profile("code")
    p["nodata"] = 99
    p["crs"] = "x"
    again = S.raster_profile("code")
    assert again["nodata"] == 0 and again["crs"] == "EPSG:5070"
    assert S.raster_profile("code") is not S.raster_profile("code")
    cols = S.record_columns()
    cols.append("x")
    assert len(S.record_columns()) == 169 and S.record_columns() is not S.record_columns()


def test_edge_record_columns_unique_and_ordered():
    cols = S.record_columns()
    assert len(cols) == len(set(cols)) == 169
    assert cols[0] == "cell_id" and cols[-1] == "run_id"
    assert cols.index("share_11") == 8 and cols.index("share_96") == 39
    assert cols.index("top_code") == 40
    assert cols.index("nlcd_1985") < cols.index("nlcd_2025") < cols.index("classf_1985")
    assert cols.index("classf_2025") < cols.index("tcc_1985") < cols.index("tcc_2025")
    assert all(isinstance(c, str) and c == c.strip() and c.isascii() for c in cols)


# ================================================================ names


@pytest.mark.parametrize("part", [
    "aux", "AUX", "Aux", "con", "CON", "nul", "NUL", "prn", "PRN", "aux.txt", "AUX.TXT", "con.txt",
    "nul.tar.gz", "prn.", "aux..", "Con.Tmp", "aux.x.y", "nul.txt ",
])
def test_edge_check_name_reserved(part):
    with pytest.raises(ValueError):
        S.check_name(part)


@pytest.mark.parametrize("part", ["", ".", "..", "a/b", "/", "/a", "a/", "../x", "a/..", "//",
                                  None, 5, b"aux", ["a"]])
def test_edge_check_name_bad(part):
    with pytest.raises(ValueError):
        S.check_name(part)


@pytest.mark.parametrize("part", ["...", ".aux", "a.aux", "auxx", "xaux", "au", "aux_", "_aux",
                                  "my.con", "console", "77_GRAPEVINE_LAKE", "田中",
                                  "\U0001f600", "a b", "AUX1", "COM1", "LPT1"])
def test_edge_check_name_accepts(part):
    S.check_name(part)


@pytest.mark.parametrize("part", ["a\x00b", "\x00", "x\x00", "aux\x00"])
def test_edge_check_name_nul(part):
    with pytest.raises(ValueError):
        S.check_name(part)


@pytest.mark.parametrize("part", ["a\\b", "..\\x", "\\", "x\\.."])
def test_edge_check_name_backslash(part):
    with pytest.raises(ValueError):
        S.check_name(part)


@pytest.mark.parametrize("part", ["aux ", "con  ", "prn\t", "aux\n"])
def test_edge_check_name_reserved_with_trailing_blank(part):
    with pytest.raises(ValueError):
        S.check_name(part)


def test_edge_paths_are_pure_and_checked():
    root = "/share/semseg"
    p = S.project_dir(root, "77_GRAPEVINE_LAKE", "cells")
    assert str(p) == "/share/semseg/77_GRAPEVINE_LAKE/cells"
    assert str(S.run_dir(root, "20261111-1200-x")) == "/share/semseg/runs/20261111-1200-x"
    assert str(S.models_dir(root)) == "/share/semseg/models"
    for bad in ("..", "a/b", "", "aux", "CON.txt"):
        with pytest.raises(ValueError):
            S.project_dir(root, bad, "cells")
        with pytest.raises(ValueError):
            S.project_dir(root, "77_X", bad)
        with pytest.raises(ValueError):
            S.run_dir(root, bad)
    # an absolute part cannot reach PurePosixPath, since '/' is refused
    with pytest.raises(ValueError):
        S.run_dir(root, "/etc")


# ================================================================ module state and size


def test_edge_import_has_no_side_effects_on_inputs_and_constants_are_consistent():
    assert len(S.CODES) == len(S.CODE_LIST) == 32 and len(set(S.CODE_LIST)) == 32
    assert set(S.CODE_LIST).isdisjoint(S.LEVEL1_CODES)
    assert len({e.colour for e in S.CODES}) == 32
    assert S.HISTORY_YEARS[0] == 1985 and S.HISTORY_YEARS[-1] == 2025 and len(S.HISTORY_YEARS) == 41
    assert S.CELL_ID_FACTOR == 10**6 and S.CELL * S.CELLS_PER_CHUNK == S.CHUNK
    assert set(S.USACE_M_MAP) | set(S._TREE_ALL) == set(S.CODE_LIST)
    assert not set(S.USACE_M_MAP) & set(S._TREE_ALL)
    assert set(S.USACE_M_MAP.values()) <= set(S.USACE_M_LABELS)
    assert set(S._CLOSED.values()) | set(S._OPEN.values()) | {18, 27, 28} <= set(S.USACE_M_LABELS)
    members = [c for ms in S._NLCD_GROUPS.values() for c in ms]
    assert len(members) == len(set(members))
    assert set(members) | {25, 26, 27, 28, 29} == set(S.CODE_LIST)


def test_edge_long_hostile_text_is_fast_and_refused():
    big = "A" * 1_000_000
    for fn in (S.parse_chunk_id, S.parse_project_folder, S.parse_polygon_id, S.check_run_id):
        with pytest.raises(ValueError):
            fn(big)
    with pytest.raises(ValueError):
        S.check_name("/" * 1_000_000)
    with pytest.raises(ValueError):
        S.project_folder(1, big + "\n")
    assert S.project_folder(1, big) == "1_" + big  # legal: no length rule in DS-6


def test_review_single_shares_rounded_before_tie():
    """D34: single shares are rounded to 9 decimals before a tie (review fix 2)."""
    assert 0.7 - 0.4 != 0.3
    # 41 and 42 tie after rounding: mixed, T = 0.6 > 0.45, closed -> M22 (22)
    assert S.derived_usace_m({41: 0.7 - 0.4, 42: 0.3, 71: 0.4}, 1.0) == 22
    # 90 ties the sum of 41-44 after rounding: the woody wetland flag is set
    assert S.usace_m_flags({90: 0.7 - 0.4, 41: 0.3, 71: 0.4}, 1.0) == 1
    # maintained grass: 21 ties 71 after rounding
    assert S.usace_m_flags({21: 0.7 - 0.4, 71: 0.3, 11: 0.4}, 1.0) == 128


def test_review_top_code_tie_after_rounding():
    assert S.top_code({71: 0.7 - 0.4, 41: 0.3, 11: 0.4}, 1.0) == 11
    assert S.top_code({71: 0.7 - 0.4, 41: 0.3, 11: 0.2, 21: 0.2}, 1.0) == 41
    assert S.top_code({82: 0.7 - 0.4, 71: 0.3, 11: 0.2, 21: 0.2}, 1.0) == 71
