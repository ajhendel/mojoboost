"""The ingestion transpose: exactness, determinism, and the infinity flag.

`trainset.transpose_to_column_major` is the whole of ingestion for a caller
holding a C-ordered matrix, which is what NumPy hands out by default. It is
the one pass that stands between `fit(X, y)` and `binning.fit_bins`, and
these are the three things that have to be true of it.

1. **It moves values, not bits.** A transpose relocates a double and does no
   arithmetic on it, so every destination slot must hold the source double
   exactly. Asserted on `to_bits()`, with no tolerance anywhere: a tolerance
   here would pass for a version that rounded, which is precisely the failure
   worth catching.

2. **It is identical at every worker count.** The row blocks are disjoint and
   each destination slot is written by exactly one task, so nothing is
   reassociated. That is an argument; this is the test of it, at
   `MOJOTREES_NUM_WORKERS` of 1, 3 and 8, and the shapes are large enough
   that the plan actually fans out rather than falling through to the serial
   path. **The fan-out is proved rather than assumed**: `plan_row_blocks` is
   asked for the block count under the same environment and the multi-worker
   settings are asserted to produce more than one block, so a run in which
   the parallel path never opened cannot pass as agreement between two serial
   runs.

3. **The fused infinity flag says exactly what a separate pass would say.**
   `+inf` and `-inf` are reported; `NaN` is not, because `NaN` is the
   missing-value marker and the binner reserves a bin for it. The flag is
   checked against an independent serial scan over the same matrix, not
   against a hand-written expectation.

Shapes are deliberately not multiples of the tile: `INGEST_TILE_BYTES` is
divided by the row width to get the tile height, so a matrix whose row count
is not a whole number of tiles is what exercises the ragged last tile of the
last block.
"""

from std.math import isinf
from std.os import setenv
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from std.utils.numerics import inf, nan

from mojotrees.parallel import plan_row_blocks
from mojotrees.trainset import (
    has_infinite,
    to_column_major,
    transpose_to_column_major,
)

from support import _uniform


comptime NAN = nan[DType.float64]()
comptime INF = inf[DType.float64]()


def _row_major(n_rows: Int, n_features: Int) -> List[Float64]:
    """A caller's C-ordered matrix: `raw[r * n_features + f]`."""
    var raw = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        raw.append(_uniform(UInt64(k)))
    return raw^


def _reference(
    raw: List[Float64], n_rows: Int, n_features: Int
) -> List[Float64]:
    """The transpose written the obvious way, serially, as the oracle."""
    var out = List[Float64](unsafe_uninit_length=n_rows * n_features)
    for r in range(n_rows):
        for f in range(n_features):
            out[f * n_rows + r] = raw[r * n_features + f]
    return out^


def test_transpose_is_exact() raises:
    """Every slot holds the caller's double, bit for bit, at several shapes.

    The shapes cover a single row, a single feature, a row count below one
    tile, and a row count that leaves a ragged tail. Fifty features is
    3,200 source bytes per tile row, so a 32 KiB tile is ten rows and 4,097
    rows is 409 whole tiles plus seven.
    """
    var shapes: List[Int] = [1, 50, 1, 1, 7, 3, 4097, 50, 1000, 1]
    for i in range(0, len(shapes), 2):
        var n_rows = shapes[i]
        var n_features = shapes[i + 1]
        var raw = _row_major(n_rows, n_features)
        var want = _reference(raw, n_rows, n_features)
        var got = to_column_major(raw, n_rows, n_features)
        assert_equal(len(got), n_rows * n_features)
        for k in range(len(want)):
            assert_equal(want[k].to_bits(), got[k].to_bits())


def test_transpose_is_worker_invariant() raises:
    """Identical output at 1, 3 and 8 workers, with the fan-out proved.

    `plan_row_blocks` is asked the same question the transpose asks it, under
    the same environment, so the assertion below is about a plan that really
    did split. Without it this test would compare three serial runs and pass
    whatever the parallel path does.
    """
    var n_rows = 20_003
    var n_features = 17
    var raw = _row_major(n_rows, n_features)
    var want = _reference(raw, n_rows, n_features)

    var workers: List[String] = ["1", "3", "8"]
    var blocks_seen = List[Int]()
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        blocks_seen.append(
            plan_row_blocks(n_rows, n_rows * n_features).n_blocks
        )
        var got = to_column_major(raw, n_rows, n_features)
        for k in range(len(want)):
            assert_equal(want[k].to_bits(), got[k].to_bits())
    _ = setenv("MOJOTREES_NUM_WORKERS", "")

    # One block at one worker, and more than one at three and at eight: the
    # gate opened, so the agreement above is agreement across task counts.
    assert_equal(blocks_seen[0], 1)
    assert_true(blocks_seen[1] > 1)
    assert_true(blocks_seen[2] > 1)


def test_infinity_flag_matches_a_separate_scan() raises:
    """The fused flag equals an independent serial scan, and NaN is not it.

    Four matrices: clean, one `+inf`, one `-inf`, and one `NaN`. The `NaN`
    case is the one that matters most, because reporting it would reject
    every matrix with a missing value in it.
    """
    var n_rows = 601
    var n_features = 11
    var poison: List[Float64] = [
        0.0,
        INF,
        -INF,
        NAN,
    ]
    var expect_flag: List[Bool] = [False, True, True, False]

    for which in range(len(poison)):
        var raw = _row_major(n_rows, n_features)
        if which != 0:
            # A slot in the interior, so it lands inside a tile rather than
            # at a boundary that a fencepost error would also hit.
            raw[437 * n_features + 6] = poison[which]

        var oracle = False
        for k in range(len(raw)):
            if isinf(raw[k]):
                oracle = True
        assert_equal(oracle, expect_flag[which])

        var dst = List[Float64](unsafe_uninit_length=n_rows * n_features)
        var found = transpose_to_column_major(raw, dst, n_rows, n_features)
        assert_equal(found, oracle)

        # And the value still crossed: a rejected matrix is rejected by the
        # caller, not by this function refusing to write it.
        var want = _reference(raw, n_rows, n_features)
        for k in range(len(want)):
            assert_equal(want[k].to_bits(), dst[k].to_bits())


def test_has_infinite_agrees_with_the_fused_flag() raises:
    """The standalone scan, for the already-column-major caller.

    It answers the same question `transpose_to_column_major` folds in, and it
    is what the Python wrapper calls when there is no transpose to fold it
    into.
    """
    var values = _row_major(4_099, 3)
    assert_false(has_infinite(values))
    values[9_001] = -INF
    assert_true(has_infinite(values))
    values[9_001] = NAN
    assert_false(has_infinite(values))

    # And at more than one worker, where the flag is an OR over block slots.
    _ = setenv("MOJOTREES_NUM_WORKERS", "8")
    assert_false(has_infinite(values))
    values[12_296] = INF
    assert_true(has_infinite(values))
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def test_shape_disagreements_are_refused() raises:
    """A length that does not match the shape is an error, not a truncation.

    Both sides are checked, because the destination is a buffer the caller
    allocated on the other side of the Python boundary and a short one is a
    memory error in their process.
    """
    var raw = _row_major(10, 4)
    var ok = List[Float64](unsafe_uninit_length=40)
    var short = List[Float64](unsafe_uninit_length=39)

    var raised = False
    try:
        _ = transpose_to_column_major(raw, ok, 10, 5)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = transpose_to_column_major(raw, short, 10, 4)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = transpose_to_column_major(raw, ok, 0, 4)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
