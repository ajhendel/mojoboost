"""The dominant-bin skip changes no number in a sparse GPU histogram.

`gpu_sparse.mojo` stops accumulating a column's stored entries when they bin
to that column's default bin and a large enough share of them do, and lets
`_sparse_default_fill_kernel` recover them from the node totals along with
the implicit zeros. That is LightGBM's `FixHistogramKernel` on our fixed
point, and it is exact rather than approximate, so the assertion this file
makes is equality and not a tolerance: the same builder, the same tree, the
same gradients, both arms, cell for cell in the raw Int32 planes.

Comparing the *raw* planes rather than the converted `Histogram` is
deliberate. The claim is about integers, and `histogram_from_host` multiplies
by `1 / scale` before anyone can look; comparing after that conversion would
be comparing two Float64 products of the same constant and would pass on a
pair of Int32 planes that differed by a rounding step. `download_raw` leaves
the fixed point alone, and the fixed point is what the identity is about.

Every test drives one builder through both arms with
`set_skip_freq_percent`, which is why that setter exists: an arm read from
the environment cannot be interleaved, and an arm that cannot be interleaved
cannot be compared against itself inside a single tree.

Every test needs an accelerator and prints `skipped` without one.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.gpu_sparse import (
    DEFAULT_SKIP_FREQ_PERCENT,
    GpuSparseHistogramBuilder,
    env_skip_freq_percent,
)
from mojotrees.sparse import SparseBinnedMatrix


comptime N_ROWS = 700
comptime N_FEATURES = 4
comptime N_BINS = 32

# The bin every feature's implicit zeros fall in. Deliberately not bin 0:
# nothing in the recovery privileges bin 0, and a test that used it would not
# notice a kernel that did.
comptime DEFAULT_BIN = 5

# Feature 1's reserved missing bin. `SparseBinnedMatrix.validate` forbids it
# from equalling the default bin, which is what keeps the skip from ever
# swallowing a missing row; this file exercises a column that has both.
comptime MISSING_BIN = 31

# The four columns, by what they are for.
comptime F_DOMINANT = 0
comptime F_UNIFORM = 1
comptime F_ALL_DEFAULT = 2
comptime F_EMPTY = 3


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _grads() -> List[Float64]:
    var g = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        g.append(2.0 * _uniform(UInt64(3_100_000 + r)) - 1.0)
    return g^


def _hessians() -> List[Float64]:
    var h = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        h.append(_uniform(UInt64(3_200_000 + r)) + 0.01)
    return h^


def _matrix() raises -> SparseBinnedMatrix:
    """Four columns chosen so the threshold has to decide each one
    differently, built by hand rather than binned from dense values.

    The share of a column that lands on its default bin is exactly what the
    rule reads, so a test that produced it by fitting a binning would be
    asserting against whatever `binning.mojo` happened to choose that day.
    Writing the bins out states the four cases and nothing else:

    - `F_DOMINANT`: nine rows in ten carry a stored entry and all but one in
      twenty of those bins to `DEFAULT_BIN`, a share near 94 percent. The
      case the trick exists for, and the case with contention to remove.
    - `F_UNIFORM`: stored entries spread over thirty bins, a share near 4
      percent, plus a band of rows in the reserved missing bin. Below any
      sane threshold, so it takes the old accumulation, and it is here to
      prove the two arms coexist in one launch.
    - `F_ALL_DEFAULT`: every stored entry on the default bin, a share of 100
      percent, which is the only column a threshold of 100 admits.
    - `F_EMPTY`: no stored entries at all. A column with nothing to skip,
      which must be refused at every threshold rather than dividing by its
      own length.
    """
    var row_index = List[Int]()
    var bin = List[UInt8]()
    var col_offsets = List[Int]()
    col_offsets.append(0)

    for r in range(N_ROWS):
        if r % 10 == 7:
            continue
        row_index.append(r)
        if r % 20 == 3:
            bin.append(UInt8(1 + (r // 20) % 4))
        else:
            bin.append(UInt8(DEFAULT_BIN))
    col_offsets.append(len(bin))

    for r in range(N_ROWS):
        if r % 5 == 0:
            continue
        row_index.append(r)
        if r % 37 == 11:
            bin.append(UInt8(MISSING_BIN))
        else:
            bin.append(UInt8((r * 7) % 30 + 1))
    col_offsets.append(len(bin))

    for r in range(N_ROWS):
        if r % 3 != 0:
            continue
        row_index.append(r)
        bin.append(UInt8(DEFAULT_BIN))
    col_offsets.append(len(bin))

    col_offsets.append(len(bin))

    var default_bin = List[UInt8](capacity=N_FEATURES)
    for _ in range(N_FEATURES):
        default_bin.append(UInt8(DEFAULT_BIN))
    var missing_bin: List[Int] = [-1, MISSING_BIN, -1, -1]

    var data = SparseBinnedMatrix(
        row_index^,
        bin^,
        col_offsets^,
        default_bin^,
        N_ROWS,
        N_FEATURES,
        N_BINS,
        missing_bin=missing_bin^,
    )
    data.validate()
    return data^


def _raw(mut builder: GpuSparseHistogramBuilder, leaf: Int) raises -> List[Int]:
    """One node's histogram as the three fixed-point Int32 planes, in the
    `[grad | hess | count]` order the device wrote them and before any
    conversion has had a chance to hide a difference."""
    builder.enqueue_leaf(leaf)
    builder.download_raw()
    var n = 3 * builder.n_features * builder.n_bins
    var out = List[Int](capacity=n)
    var src = builder.host_out.unsafe_ptr()
    for i in range(n):
        out.append(Int(src.unsafe_load(i)))
    return out^


def _assert_planes_equal(a: List[Int], b: List[Int]) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def _opened(data: SparseBinnedMatrix) raises -> GpuSparseHistogramBuilder:
    """A builder whose threshold is the default whatever the environment
    holds, so a run under `MOJOTREES_GPU_SPARSE_SKIP_FREQ` still tests the
    default rule."""
    var builder = GpuSparseHistogramBuilder(data, 8)
    builder.set_skip_freq_percent(DEFAULT_SKIP_FREQ_PERCENT)
    return builder^


def test_the_threshold_decides_each_column_separately() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _matrix()
        var builder = _opened(data)

        # The shares the four columns were built to have.
        assert_true(builder.default_bin_share_percent(F_DOMINANT) >= 90)
        assert_true(builder.default_bin_share_percent(F_UNIFORM) <= 10)
        assert_equal(builder.default_bin_share_percent(F_ALL_DEFAULT), 100)
        assert_equal(builder.default_bin_share_percent(F_EMPTY), 0)

        assert_equal(builder.skip_freq_percent(), DEFAULT_SKIP_FREQ_PERCENT)
        assert_true(builder.skips_default_bin(F_DOMINANT))
        assert_false(builder.skips_default_bin(F_UNIFORM))
        assert_true(builder.skips_default_bin(F_ALL_DEFAULT))
        assert_false(builder.skips_default_bin(F_EMPTY))
        assert_equal(builder.n_skipped_features(), 2)

        # The escape hatch, and the strictest rule that still admits a
        # column. An empty column is admitted by neither.
        builder.set_skip_freq_percent(0)
        assert_equal(builder.n_skipped_features(), 0)
        builder.set_skip_freq_percent(100)
        assert_equal(builder.n_skipped_features(), 1)
        assert_true(builder.skips_default_bin(F_ALL_DEFAULT))
        assert_false(builder.skips_default_bin(F_EMPTY))

        # Above 100 clamps rather than admitting nothing by accident, and a
        # negative threshold is refused because "off" already has a spelling.
        builder.set_skip_freq_percent(250)
        assert_equal(builder.skip_freq_percent(), 100)
        var raised = False
        try:
            builder.set_skip_freq_percent(-1)
        except:
            raised = True
        assert_true(raised)

        # The environment reader agrees with the constant when unset, and
        # never reports something a threshold cannot be.
        var p = env_skip_freq_percent()
        assert_true(p >= 0)
        assert_true(p <= 100)


def test_root_histogram_is_bit_identical_across_the_skip() raises:
    """The whole dataset, both arms, one builder, one tree. The dominant and
    the all-default columns take the skip, the uniform one does not, and the
    empty one has nothing either way, so a single launch covers all four."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _matrix()
        var builder = _opened(data)
        builder.upload_gradients(_grads(), _hessians())
        builder.begin_tree()

        assert_equal(builder.n_skipped_features(), 2)
        var with_skip = _raw(builder, 0)
        builder.set_skip_freq_percent(0)
        assert_equal(builder.n_skipped_features(), 0)
        var without_skip = _raw(builder, 0)
        _assert_planes_equal(with_skip, without_skip)

        # A histogram of zeros would satisfy the equality above, so say that
        # the arms agreed on something: every column's counts sum to the
        # node's rows, because every row lands in exactly one bin of it.
        for f in range(N_FEATURES):
            var total = 0
            for b in range(N_BINS):
                total += with_skip[2 * N_FEATURES * N_BINS + f * N_BINS + b]
            assert_equal(total, N_ROWS)


def test_a_subrange_node_is_bit_identical_across_the_skip() raises:
    """A node whose rows are a strict subset of the dataset, whose entry
    windows are a strict subset of the columns, and whose totals are its own.
    The recovery subtracts against those totals, so this is where a skip that
    leaked into the wrong node's leftover would show."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _matrix()
        var builder = _opened(data)
        builder.upload_gradients(_grads(), _hessians())

        builder.begin_tree()
        builder.apply_split(F_UNIFORM, 15, 0, 1, 2, MISSING_BIN, False)
        var left_rows = builder.row_range(1).count()
        var right_rows = builder.row_range(2).count()
        assert_true(left_rows > 0)
        assert_true(right_rows > 0)
        assert_equal(left_rows + right_rows, N_ROWS)
        var left_skip = _raw(builder, 1)
        var right_skip = _raw(builder, 2)

        builder.set_skip_freq_percent(0)
        builder.begin_tree()
        builder.apply_split(F_UNIFORM, 15, 0, 1, 2, MISSING_BIN, False)
        assert_equal(builder.row_range(1).count(), left_rows)
        var left_off = _raw(builder, 1)
        var right_off = _raw(builder, 2)

        _assert_planes_equal(left_skip, left_off)
        _assert_planes_equal(right_skip, right_off)


def test_sibling_subtraction_survives_the_skip() raises:
    """A parent's bins are the exact integer sum of its two children's, with
    the skip on and in the fixed point, not to a tolerance.

    This is the identity the grower builds the sibling histogram by. It
    survives because both sides of the recovery are additive over a split:
    the node totals are, the accumulated bins are, and the skipped set is
    partitioned between the children by the same mask as everything else.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _matrix()
        var builder = _opened(data)
        builder.upload_gradients(_grads(), _hessians())
        builder.begin_tree()
        assert_equal(builder.n_skipped_features(), 2)

        var parent = _raw(builder, 0)
        builder.apply_split(F_DOMINANT, 3, 0, 1, 2, -1, False)
        assert_true(builder.row_range(1).count() > 0)
        assert_true(builder.row_range(2).count() > 0)
        var left = _raw(builder, 1)
        var right = _raw(builder, 2)

        assert_equal(len(parent), len(left))
        for i in range(len(parent)):
            assert_equal(parent[i], left[i] + right[i])


def test_the_count_plane_is_recovered_exactly() raises:
    """The count plane specifically, against a host count of the same rows.

    Gradient and hessian cells are only checkable against the other arm, but
    a count is checkable against the matrix itself: the default bin of a
    column holds every row with no stored entry for it plus every stored
    entry that binned there, whether the accumulation visited those entries
    or the subtraction recovered them.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _matrix()
        var builder = _opened(data)
        builder.upload_gradients(_grads(), _hessians())
        builder.begin_tree()
        var raw = _raw(builder, 0)
        var counts = 2 * N_FEATURES * N_BINS

        for f in range(N_FEATURES):
            # What the matrix says each bin of this column holds at the root.
            var want = List[Int](capacity=N_BINS)
            want.resize(N_BINS, 0)
            var stored = 0
            for i in range(data.col_offsets[f], data.col_offsets[f + 1]):
                want[Int(data.bin[i])] += 1
                stored += 1
            want[DEFAULT_BIN] += N_ROWS - stored
            for b in range(N_BINS):
                assert_equal(raw[counts + f * N_BINS + b], want[b])

        # And the same after the skip is turned off, so the check is a
        # statement about both arms and not only the one that ran first.
        builder.set_skip_freq_percent(0)
        var off = _raw(builder, 0)
        for i in range(counts, 3 * N_FEATURES * N_BINS):
            assert_equal(raw[i], off[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
