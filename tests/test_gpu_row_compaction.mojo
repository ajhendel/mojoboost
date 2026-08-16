"""The row-compaction arm: what it promises, and the gate that it engaged.

CatBoost's GPU grower physically reorders the data at every split so a leaf's
rows are contiguous rather than named by a scattered index
(`catboost/cuda/methods/greedy_subsets_searcher/split_properties_helper.cpp`,
`MakeSplit`). `GpuActiveRows.set_row_compaction` is that mechanism here. It is
off by default, it is a trade rather than a saving, and this file is what
stops it being a trade that also changes an answer.

Four claims, in the order that makes each one worth making.

1. **The invariant holds.** `cbins[f * n_rows + j] == bins[f * n_rows +
   rows[j]]` and `cgq[2j] == gq[2 * rows[j]]`, for every position of the row
   buffer, after a rebuild and after a chain of splits has maintained it
   incrementally. Everything else in the arm rests on this: it is what makes
   the compacted launch read the identical byte at the identical step.

2. **The permutation does not move.** The same fixture is grown twice, once
   with the arm off and once on, and `download_rows` is compared position for
   position. This is the claim the lane was told to state explicitly, and it
   is stated here as a check rather than as prose: the arm reorders *data*,
   never the index, so every consumer of `rows_dev` outside this module --
   `_publish_row_ranges`, the segment table `update_raw_device` reads, the
   speculation's set-preservation argument -- sees a buffer that did not
   change.

3. **The histogram is bit-identical.** Not close, not within a tolerance:
   equal cell for cell, across both strategies and both feature-group rungs,
   against a host sum that is the same integer arithmetic.

4. **The forest is identical, node for node**, end to end through `train_gpu`
   on the device-resident plane, and the arm is proved to have engaged rather
   than assumed to have. That last half is the whole reason the compaction
   trace exists: this arm is reachable end to end only through an environment
   variable, and a variable that silently did nothing would make claim 4 a
   comparison of the default against itself.

**No timing is claimed anywhere in this file and none was measured.** The arm
exists to be A/B'd by the benchmark harness in an interleaved window; what is
established here is only that the two arms are the same model.

Every test that needs a device is wrapped so the file runs green on a machine
without one, which is the convention the rest of the GPU tests in this
directory follow.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.boosting import BoosterParams
from mojotrees.gpu_active_rows import GpuActiveRows, RowRouting
from mojotrees.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    histogram_bin_capacity,
    histogram_shared_bytes,
    query_device_caps,
)
from mojotrees.objective_registry import SQUARED_ERROR
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _splitmix64, _uniform


comptime _TRACE_PATH = "/tmp/mojotrees_compaction_test_trace.txt"
"""Where the per-tree compaction record goes for the end-to-end test.

Truncated before each arm and read after it, so an arm's record is that arm's
and not the file's history.
"""

comptime _PLANE_TRACE_PATH = "/tmp/mojotrees_compaction_plane_trace.txt"
"""Where `MOJOTREES_GPU_TREE_RESIDENT_TRACE` writes, for the same test.

A second sink rather than the compaction's, so that counting one token cannot
be satisfied by the other instrument having written a line.
"""

comptime _PLANE_MARK = "plane=device-resident"
"""The token `grow_tree_device_resident` writes once per tree it grows.

Named the same way `test_gpu_tree_resident.mojo` names it, and asserted for
the same reason: without it, "the forests agree" would be a comparison of the
host-driven search loop against itself, which is not the plane this arm has
to be identical on.
"""


def _truncate(path: String) raises:
    with open(path, "w") as handle:
        handle.write(String(""))


def _read(path: String) raises -> String:
    return open(path, "r").read()


def _truncate_trace() raises:
    _truncate(_TRACE_PATH)
    _truncate(_PLANE_TRACE_PATH)


def _read_trace() raises -> String:
    return _read(_TRACE_PATH)


# --- Fixtures -------------------------------------------------------------


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, built the way
    test_gpu_active_rows.mojo and test_gpu_hist_row_unroll.mojo build one so
    the three files look at the same kind of data."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x51ED2701)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _upload_bins(
    mut ctx: DeviceContext, data: BinnedMatrix
) raises -> DeviceBuffer[DType.uint8]:
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.uint8](len(data.bins))
        ctx.enqueue_copy(dst_buf=buf, src_ptr=data.bins.unsafe_ptr())
        ctx.synchronize()
        return buf^


def _reversed_slots(
    mut ctx: DeviceContext, n_features: Int
) raises -> DeviceBuffer[DType.int32]:
    """Slot `s` holds feature `n_features - 1 - s`, so a kernel that wrote at
    the slot index rather than the feature id would be caught."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.int32](n_features)
        with buf.map_to_host() as host:
            for s in range(n_features):
                host.unsafe_ptr().unsafe_store(s, Int32((n_features - 1) - s))
        return buf^


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[r]
            - 1.5 * features[n_rows + r]
            + _uniform(UInt64(r))
        )
    return target^


# --- Claim 1: the invariant ------------------------------------------------


def _assert_invariant(
    mut rows: GpuActiveRows,
    data: BinnedMatrix,
    n_rows: Int,
    n_features: Int,
    label: String,
) raises:
    """`cbins[f * n_rows + j] == bins[f * n_rows + rows[j]]` everywhere, and
    the same for the quantized gradient pair.

    Checked over the **whole** row buffer rather than the active prefix,
    because the rebuild covers the whole buffer and a maintenance step that
    corrupted a position past the active prefix would be a real bug that only
    showed up on a later, larger tree.
    """
    var perm = rows.download_rows()
    var cbins = rows.download_compacted_bins()
    var cgq = rows.download_compacted_grads()
    var gq = rows.download_quantized_grads()
    assert_equal(len(perm), n_rows, label + ": permutation length")
    assert_equal(len(cbins), n_rows * n_features, label + ": plane length")
    # How many Int32 words one row's staged derivative pair occupies. Two on
    # the Int32 arm; one on the gradient-staging lane's Int16 arm, where the
    # pair is two 16-bit words inside the first `4 * n_rows` bytes of the same
    # allocation. Comparing whole words either way means this check never has
    # to know what is *inside* them, only that the compacted copy holds the
    # source's bytes for the row the permutation names -- which is exactly the
    # invariant, at either width.
    var words = 1 if rows.compaction_packed_gradients() else 2
    var checked = 0
    for j in range(n_rows):
        var r = Int(perm[j])
        assert_true(
            r >= 0 and r < n_rows, label + ": permutation escaped the buffer"
        )
        for f in range(n_features):
            assert_equal(
                Int(cbins[f * n_rows + j]),
                Int(data.bins[f * n_rows + r]),
                label + ": compacted bin at position " + String(j),
            )
            checked += 1
        for w in range(words):
            assert_equal(
                Int(cgq[words * j + w]),
                Int(gq[words * r + w]),
                label
                + ": compacted derivative word "
                + String(w)
                + " at position "
                + String(j),
            )
    assert_equal(
        checked, n_rows * n_features, label + ": the sweep skipped cells"
    )


def _grad_planes(
    mut ctx: DeviceContext, n_rows: Int
) raises -> Tuple[
    DeviceBuffer[DType.float32],
    DeviceBuffer[DType.float32],
    List[Int],
    List[Int],
]:
    """Integer-valued gradients and hessians carried as Float32.

    Under a scale of exactly 1.0, `Int32(round(x * 1.0))` is `x`, so the host
    reference below is the same integer sum the device computes and not an
    approximation of it.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var gvals = List[Int](capacity=n_rows)
        var hvals = List[Int](capacity=n_rows)
        for r in range(n_rows):
            gvals.append(Int(_splitmix64(UInt64(r) + 3) % 401) - 200)
            hvals.append(Int(_splitmix64(UInt64(r) + 77) % 97) + 1)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        with grad_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(gvals[r]))
        with hess_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(hvals[r]))
        return (grad_dev^, hess_dev^, gvals^, hvals^)


def _invariant_case(
    n_bins: Int, n_features: Int, n_rows: Int, packed_grads: Bool = False
) raises:
    """Rebuild, then three splits, checking the invariant after each.

    The three splits are what makes this more than a test of the rebuild. The
    first is over the root, so the compaction scatter's source is the plane
    the rebuild wrote; the second and third are over windows the first
    produced, so their source is a plane the incremental path wrote. A
    maintenance step that were correct only against a freshly rebuilt plane
    would pass the first check and fail the second, which is exactly the fault
    a single-split test could not see.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _reversed_slots(ctx, n_features)
        var planes = _grad_planes(ctx, n_rows)
        var grad_dev = planes[0].copy()
        var hess_dev = planes[1].copy()

        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        # The gradient-staging lane's Int16 width, when asked for. Set before
        # the arm so the first staging is at that width and the planes are
        # built from it, which is the ordering a real fit has.
        rows.set_packed_gradients(packed_grads)
        rows.set_row_compaction(True)
        assert_true(rows.row_compaction_requested())
        # Requested is not the same as live: nothing has built the planes yet.
        assert_true(not rows.row_compaction_live())
        rows.begin_tree()

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var out_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var part_dev = ctx.enqueue_create_buffer[DType.int32](3)

        # One histogram to make the planes live. This is the only thing that
        # builds them, deliberately: it is where the `bins` pointer is.
        var tiling = rows.range_tiling(
            caps, 0, n_features, STRATEGY_ATOMIC, 1 << 20
        )
        rows.enqueue_range_histogram(
            tiling,
            0,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            out_dev.unsafe_ptr(),
            part_dev.unsafe_ptr(),
            n_features,
            Float32(1.0),
            Float32(1.0),
        )
        rows.synchronize()
        assert_true(rows.row_compaction_live())
        assert_equal(
            rows.compaction_packed_gradients(),
            packed_grads,
            "the planes were built at the other gradient width",
        )
        var after_build = rows.compaction_counts()
        assert_equal(after_build[0], 1, "the rebuild did not run")
        assert_equal(after_build[1], 0, "a split ran before any split")
        var tag = String(
            "n=",
            n_rows,
            " f=",
            n_features,
            " b=",
            n_bins,
            " packed=",
            packed_grads,
            " ",
        )
        _assert_invariant(
            rows, data, n_rows, n_features, tag + "after the rebuild"
        )

        # Split the root, then both of its children. Every one of these has to
        # leave the invariant standing.
        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, n_bins // 2)
        )
        rows.synchronize()
        _assert_invariant(
            rows, data, n_rows, n_features, tag + "after split 0"
        )
        _ = rows.partition(
            bins.unsafe_ptr(), 1, 3, 4, RowRouting.numerical(1, n_bins // 3)
        )
        rows.synchronize()
        _assert_invariant(
            rows, data, n_rows, n_features, tag + "after split 1"
        )
        _ = rows.partition(
            bins.unsafe_ptr(),
            2,
            5,
            6,
            RowRouting.numerical(n_features - 1, (2 * n_bins) // 3),
        )
        rows.synchronize()
        _assert_invariant(
            rows, data, n_rows, n_features, tag + "after split 2"
        )

        var after_splits = rows.compaction_counts()
        assert_equal(
            after_splits[0], 1, "the planes were rebuilt from scratch mid-tree"
        )
        assert_equal(
            after_splits[1], 3, "the incremental path did not run per split"
        )


def test_the_compacted_planes_hold_what_the_permutation_names() raises:
    """The invariant, on three shapes, over a chain of three splits each.

    2049 rows is one past two whole 1024-row groups at the 256-thread block
    width this backend derives, so the partition's tile walk ends on the
    smallest possible ragged remainder; 900 rows is below one group, so every
    thread's walk is tail. Between them the compaction scatter's chunk
    arithmetic is exercised on both sides of the tiling boundary.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _invariant_case(32, 3, 900)
        _invariant_case(64, 5, 2049)
        _invariant_case(256, 4, 1200)
        # And the same three against the gradient-staging lane's Int16 pair,
        # which the compaction kernels move four bytes of instead of eight.
        # A width the compacted plane got wrong would decode to a legal
        # gradient belonging to another row, so this is checked as bytes
        # rather than inferred from the histogram agreeing.
        _invariant_case(32, 3, 900, True)
        _invariant_case(64, 5, 2049, True)
        _invariant_case(256, 4, 1200, True)


# --- Claim 2 and 3: the permutation and the histogram ----------------------


def _crossarm_case(
    n_bins: Int, n_features: Int, n_rows: Int, packed_grads: Bool = False
) raises:
    """The same fixture through both arms, comparing everything observable.

    The permutation is compared position for position and the histogram cell
    for cell, over both strategies and both feature-group rungs, and both are
    also compared against a host sum so that "the two arms agree" cannot be
    satisfied by both arms being wrong in the same way.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _reversed_slots(ctx, n_features)
        var planes = _grad_planes(ctx, n_rows)
        var grad_dev = planes[0].copy()
        var hess_dev = planes[1].copy()
        var gvals = planes[2].copy()
        var hvals = planes[3].copy()

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var out_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_out = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var cap = histogram_bin_capacity(n_bins)

        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.set_packed_gradients(packed_grads)
        var perm_off = List[Int32]()
        var arms = 0
        for a in range(2):
            var on = a == 1
            rows.set_row_compaction(on)
            rows.begin_tree()

            # A histogram at the root to make the planes live, then two
            # splits, then the node under comparison. The node is a
            # grandchild, so its rows are a quarter of the dataset and are
            # scattered: exactly the shape whose gather the arm is about.
            var root_tiling = rows.range_tiling(
                caps, 0, n_features, STRATEGY_ATOMIC, 1 << 20
            )
            var part_seed = ctx.enqueue_create_buffer[DType.int32](3)
            rows.enqueue_range_histogram(
                root_tiling,
                0,
                bins.unsafe_ptr(),
                grad_dev.unsafe_ptr(),
                hess_dev.unsafe_ptr(),
                feat_dev.unsafe_ptr(),
                out_dev.unsafe_ptr(),
                part_seed.unsafe_ptr(),
                n_features,
                Float32(1.0),
                Float32(1.0),
            )
            assert_equal(
                rows.row_compaction_live(),
                on,
                "the arm did not follow the request",
            )
            _ = rows.partition(
                bins.unsafe_ptr(),
                0,
                1,
                2,
                RowRouting.numerical(0, n_bins // 2),
            )
            _ = rows.partition(
                bins.unsafe_ptr(),
                1,
                3,
                4,
                RowRouting.numerical(1, n_bins // 3),
            )
            var node = 3
            var window = rows.range_of(node)
            assert_true(window.count() > 0)
            assert_true(window.begin >= 0)
            assert_true(window.count() < rows.n_active())

            # Claim 2. The permutation is a data structure the rest of the
            # backend reads, and the arm must not have moved it.
            var perm = rows.download_rows()
            if not on:
                perm_off = perm.copy()
            else:
                assert_equal(
                    len(perm), len(perm_off), "permutation length moved"
                )
                for j in range(len(perm)):
                    assert_equal(
                        Int(perm[j]),
                        Int(perm_off[j]),
                        "the compaction arm moved the permutation at "
                        + String(j),
                    )

            # The host reference for this node, from the permutation the
            # device holds.
            var idx = rows.download_range(node)
            assert_equal(len(idx), window.count())
            var golden = List[Int](capacity=cells)
            for _ in range(cells):
                golden.append(0)
            for jj in range(len(idx)):
                var r = idx[jj]
                for s in range(n_features):
                    var f = (n_features - 1) - s
                    var b = Int(data.bins[f * n_rows + r])
                    var cell = f * n_bins + b
                    golden[cell] += gvals[r]
                    golden[hist_size + cell] += hvals[r]
                    golden[2 * hist_size + cell] += 1

            var strategies = [STRATEGY_ATOMIC, STRATEGY_TILED]
            var groups = [1, 4]
            for si in range(len(strategies)):
                var tiling = rows.range_tiling(
                    caps, node, n_features, strategies[si], 1 << 20
                )
                var part_cells = tiling.partial_cells
                if part_cells < 1:
                    part_cells = 1
                var part_dev = ctx.enqueue_create_buffer[DType.int32](
                    3 * part_cells
                )
                for gi in range(len(groups)):
                    var group = groups[gi]
                    if (
                        histogram_shared_bytes(cap, group)
                        > caps.max_shared_memory_per_block
                    ):
                        continue
                    rows.set_feature_group(group)
                    rows.enqueue_range_histogram(
                        tiling,
                        node,
                        bins.unsafe_ptr(),
                        grad_dev.unsafe_ptr(),
                        hess_dev.unsafe_ptr(),
                        feat_dev.unsafe_ptr(),
                        out_dev.unsafe_ptr(),
                        part_dev.unsafe_ptr(),
                        n_features,
                        Float32(1.0),
                        Float32(1.0),
                    )
                    ctx.enqueue_copy(
                        dst_ptr=host_out.unsafe_ptr(), src_buf=out_dev
                    )
                    ctx.synchronize()
                    var got = host_out.unsafe_ptr()
                    for i in range(cells):
                        assert_equal(
                            Int(got.unsafe_load(i)),
                            golden[i],
                            "compaction="
                            + String(on)
                            + " strategy="
                            + String(strategies[si])
                            + " group="
                            + String(group)
                            + " cell "
                            + String(i),
                        )
                    arms += 1
                    rows.set_feature_group(1)
        # Two compaction arms, two strategies, at least one group rung each.
        # A silently skipped sweep would otherwise pass.
        assert_true(arms >= 4, "the sweep did not reach both arms")
        assert_true(
            rows.compaction_counts()[1] >= 2,
            "the incremental path never ran, so the on arm proved nothing",
        )
        rows.set_row_compaction(False)


def test_the_two_arms_agree_on_the_permutation_and_the_histogram() raises:
    """Claims 2 and 3 together, on three shapes.

    They are one test because the second is worthless without the first: a
    histogram that agreed while the permutation had moved would mean the two
    arms had built the same numbers over different rows, which is the failure
    this lane's hard constraint is about.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _crossarm_case(32, 3, 2049)
        _crossarm_case(64, 6, 1500)
        _crossarm_case(256, 4, 900)
        # Once more at the Int16 staged width. The host reference is the same
        # integer sum either way -- under a unit scale the stored word is the
        # value -- so this is the two arms and the host all agreeing at the
        # width the compaction kernels move four bytes of.
        _crossarm_case(64, 6, 1500, True)


# --- Claim 4: the forest, end to end --------------------------------------


def _assert_same_forest(a: List[Tree], b: List[Tree], label: String) raises:
    """Every tree, node for node, with no tolerance anywhere.

    Lifted deliberately from `test_gpu_tree_resident._assert_same_forest`, so
    that the bar this arm is held to is the bar the device-resident plane is
    held to and not a weaker one written for the occasion. `value` and the
    thresholds are compared as bit patterns: the question is not whether the
    two arms produce similar models, it is whether they make the same
    decisions, and a decision is discrete.
    """
    assert_equal(len(a), len(b), label + ": tree count")
    for t in range(len(a)):
        var want = a[t].copy()
        var got = b[t].copy()
        assert_equal(got.n_leaves, want.n_leaves, label + ": n_leaves")
        assert_equal(len(got.feature), len(want.feature), label + ": n_nodes")
        for i in range(len(want.feature)):
            assert_equal(got.feature[i], want.feature[i], label + ": feature")
            assert_equal(
                got.threshold_bin[i],
                want.threshold_bin[i],
                label + ": threshold_bin",
            )
            assert_equal(got.left[i], want.left[i], label + ": left")
            assert_equal(got.right[i], want.right[i], label + ": right")
            assert_equal(
                got.value[i].to_bits(),
                want.value[i].to_bits(),
                label + ": value bits",
            )


def test_the_two_arms_grow_the_identical_forest() raises:
    """Both arms of a real fit, node for node, with the gate asserted open.

    `MOJOTREES_GPU_SPLIT_STRATEGY=device` is set for both arms and is not
    optional, for the reason `test_gpu_tree_resident._both_planes` gives: the
    automatic policy would otherwise send a fixture this size to the host
    histogram scan, which reaches neither `GpuActiveRows` histogram entry
    point, and the comparison would become the default against itself.

    **The two-sided gate.** The on arm's trace has to say `arm=on` once per
    tree and the off arm's has to say `arm=off`, and the on arm's last record
    has to show launches actually issued. Without that this test would pass
    just as happily if `MOJOTREES_GPU_ROW_COMPACTION` were misspelled, which
    is precisely how an environment-variable arm gets run under the other
    arm's label -- something this repository has already done once.

    Every variable is cleared afterwards whichever way the assertions go: a
    leaked one would change every later test in the file, and would change it
    in the direction of passing.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 700
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var params = BoosterParams(4, 0.1, TreeParams(8, 20, 1.0, 1e-3))

        _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "device")
        _ = setenv("MOJOTREES_GPU_COMPACTION_TRACE", _TRACE_PATH)
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", _PLANE_TRACE_PATH)

        _truncate_trace()
        _ = setenv("MOJOTREES_GPU_ROW_COMPACTION", "0")
        var off = train_gpu(data, target, SQUARED_ERROR, params)
        var off_trace = _read_trace()
        var off_plane = _read(_PLANE_TRACE_PATH)

        _truncate_trace()
        _ = setenv("MOJOTREES_GPU_ROW_COMPACTION", "1")
        var on = train_gpu(data, target, SQUARED_ERROR, params)
        var on_trace = _read_trace()
        var on_plane = _read(_PLANE_TRACE_PATH)

        _ = setenv("MOJOTREES_GPU_ROW_COMPACTION", "")
        _ = setenv("MOJOTREES_GPU_COMPACTION_TRACE", "")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
        _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "")

        # The plane, before the arm. Both arms have to have grown every tree
        # on the device-resident plane, or the node-for-node comparison below
        # is being made somewhere this arm was never meant to be judged.
        assert_equal(
            off_plane.count(_PLANE_MARK),
            len(off.trees),
            "the off arm did not grow every tree on the device-resident plane",
        )
        assert_equal(
            on_plane.count(_PLANE_MARK),
            len(on.trees),
            "the on arm did not grow every tree on the device-resident plane",
        )
        assert_true(len(on.trees) > 0, "the fit grew no trees")

        # The gate, both sides, before anything is compared.
        assert_true(
            off_trace.count("arm=off") > 0,
            (
                "the off arm never reached GpuActiveRows.begin_tree, so this"
                " comparison proves nothing"
            ),
        )
        assert_equal(
            off_trace.count("arm=on"),
            0,
            "the off arm turned the compaction on",
        )
        assert_true(
            on_trace.count("arm=on") > 0,
            (
                "the on arm never reached the compaction; the environment"
                " variable did not select it"
            ),
        )
        assert_equal(
            on_trace.count("arm=off"),
            0,
            "the on arm ran with the compaction off",
        )
        # Launches issued, not launches requested. The record is written at
        # the *start* of a tree, so the second tree's record is the first
        # tree's tally and is what shows the wire carried current.
        assert_true(
            on_trace.count("builds=0 scatters=0") < on_trace.count("arm=on"),
            "the compaction was requested and never issued a launch",
        )

        _assert_same_forest(
            off.trees.copy(), on.trees.copy(), "row compaction"
        )


# --- The refusals ----------------------------------------------------------


def test_the_arm_refuses_what_it_cannot_do() raises:
    """Two preconditions, refused rather than worked around.

    Both failure modes are silent if they are not refused. Without the
    quantized gradient arm the compacted planes hold no gradient at all, and
    an identity index would be read against the un-compacted Float32 planes,
    which is a wrong histogram that no shape check would catch. With the
    blocked layout on, two arms re-arrange the same matrix and one of them is
    ignored.

    The withdrawal of the quantized arm *after* compaction is on is checked
    too, and it is the case that is not refused: `set_quantized_gradients` has
    no `raises` and growing one would change a signature this lane does not
    own, so the launch predicate carries it instead. What the test asserts is
    that the result is the safe fallback and not a live compaction over a
    plane that is not there.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 256, 3, 32, caps)

        rows.set_quantized_gradients(False)
        var refused_quant = False
        try:
            rows.set_row_compaction(True)
        except:
            refused_quant = True
        assert_true(
            refused_quant,
            "compaction was accepted without the quantized gradient arm",
        )
        assert_true(not rows.row_compaction_requested())

        rows.set_quantized_gradients(True)
        rows.set_row_compaction(True)
        assert_true(rows.row_compaction_requested())

        var refused_blocked = False
        try:
            rows.set_blocked_layout(4)
        except:
            refused_blocked = True
        assert_true(
            refused_blocked,
            "the blocked layout was accepted beside row compaction",
        )

        # The packed bin layout, refused from both directions. This one is not
        # tidiness: the packed decode reads the *same* `bins` pointer the byte
        # gather reads, so a live compaction would hand a bit-stream decoder a
        # byte plane and every bin id it produced would be legal and wrong.
        var narrow_widths = List[Int](capacity=3)
        for _ in range(3):
            narrow_widths.append(5)
        var refused_packed_bins = False
        try:
            rows.set_packed_bins(narrow_widths.copy())
        except:
            refused_packed_bins = True
        assert_true(
            refused_packed_bins,
            "the packed bin layout was accepted beside row compaction",
        )

        # And the other order: compaction refused while packed bins are on.
        rows.set_row_compaction(False)
        rows.set_packed_bins(narrow_widths.copy())
        var refused_compaction = False
        try:
            rows.set_row_compaction(True)
        except:
            refused_compaction = True
        assert_true(
            refused_compaction,
            "row compaction was accepted beside the packed bin layout",
        )
        rows.set_packed_bins(List[Int]())
        rows.set_row_compaction(True)

        # The un-refusable direction, carried by the launch predicate.
        rows.set_quantized_gradients(False)
        assert_true(
            not rows.row_compaction_live(),
            "compaction stayed live after the quantized arm was withdrawn",
        )
        rows.set_row_compaction(False)


def test_a_disabled_arm_allocates_nothing_and_downloads_nothing() raises:
    """The default costs no device memory, and says so when asked.

    `download_compacted_bins` raises on an instance that never enabled the
    arm rather than returning a plausible buffer of zeros, which is what a
    one-element placeholder would have made easy to return by accident.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 256, 3, 32, caps)
        assert_true(not rows.row_compaction_requested())
        assert_true(not rows.row_compaction_live())
        var counts = rows.compaction_counts()
        assert_equal(counts[0], 0)
        assert_equal(counts[1], 0)
        var raised = False
        try:
            _ = rows.download_compacted_bins()
        except:
            raised = True
        assert_true(
            raised, "a never-enabled instance handed out a compacted plane"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
