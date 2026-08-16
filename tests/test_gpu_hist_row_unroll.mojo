"""The unrolled histogram row walk, against a host reference and against the
walk it replaces.

`gpu_active_rows._hist_accumulate_rows` walks `HIST_ROW_UNROLL` rows of a
node's range per loop iteration, issuing the loads of each stage together, in
place of the one-row-per-iteration loop the two histogram kernels used to
carry inline. `GpuActiveRows.set_row_unroll(False)` puts the old shape back at
run time, without a second kernel instantiation, so both arms are reachable in
one process.

The claim being tested is exact and not approximate. Accumulation is
fixed-point Int32 and integer addition is associative and commutative, so
changing how many rows a thread keeps in flight can only change the order in
which the adds are issued, and the order of integer adds cannot change a sum.
Every comparison in this file is therefore `assert_equal` on raw Int32 cells
of the histogram planes, not a tolerance.

Two kinds of check, because they fail in different ways.

The first compares the device against a histogram computed on the host from
the same binned matrix and the same rows. That is the check a cross-arm
comparison cannot make: an unroll that dropped or double-counted the same rows
in *every* arm would leave all the arms agreeing with each other and all of
them wrong. To keep the comparison free of any question about whether host and
device round a Float32 product the same way, this half uses small integer
gradients and hessians under unit scales, where `Int32(round(x * 1.0))` is `x`
exactly on both sides; any disagreement is then a disagreement about which
rows were visited, which is the only thing the unroll could get wrong.

The second holds the shipped shape (`set_row_unroll(False)`, group 1, atomic
accumulation, Float32 gradient planes) as the reference and compares every
other arm against it under realistic gradients and real fixed-point scales:
both accumulation strategies, every feature-group rung the device can hold,
both gradient sources, and the constant-hessian specialization. That is the
half that would catch the unrolled path having drifted in the one place it
touches floating point, namely the Float32 arm's `Int32(round(x * scale))`,
whose exact written form `docs/NUMERICS.md` section 5.6 depends on.

What the shapes are chosen to cover:

- Row counts on both sides of one thread-stride group. A thread walks rows
  `tid, tid + block_dim.x, ...`, so the unrolled loop runs only while a full
  group of `HIST_ROW_UNROLL` of them remains and the rest are taken one at a
  time by the tail. Counts near and below `HIST_ROW_UNROLL * block_threads`
  exercise a walk that is all tail, a walk that is all body, and walks that
  end with one, two, and three rows over.
- Feature counts that leave a ragged group block: 3, 5, 8 and 17 slots
  against group widths 1, 2, 4, 8 and 16, so a tail block owning one slot and
  a tail block owning several are both reached.
- Both accumulation strategies, since the row walk is shared by
  `_range_hist_atomic_kernel` and `_range_hist_partial_kernel` and a mistake
  in the shared code has to show in both.
- A node whose range is a strict subset of the active prefix, reached by
  splitting the root first, so `begin > 0` and `count < n_active`. A walk that
  ignored the window would still pass a root-only test.
- A non-identity feature-slot permutation, so a kernel that indexed by
  `block_idx.x` rather than following `feat_ids` would show.

The device half skips (passing) with no accelerator. The one host-side check
runs everywhere.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    HIST_ROW_UNROLL,
    GpuActiveRows,
    RowRouting,
)
from mojotrees.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    histogram_bin_capacity,
    histogram_shared_bytes,
    query_device_caps,
)
from support import _splitmix64


def test_the_unroll_depth_is_a_usable_one() raises:
    """A depth below one has no meaning, and a depth of one is the shipped
    shape rather than an unroll, which the rest of this file would then be
    testing against itself. Runs without an accelerator because it is
    arithmetic over a compile-time constant."""
    assert_true(HIST_ROW_UNROLL >= 1)


# --- Shared fixtures ---


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, built the way
    test_gpu_active_rows.mojo and test_gpu_kernel_family.mojo build one so the
    three files are looking at the same kind of data."""
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
    """Slot `s` holds feature `n_features - 1 - s`. Not the identity, so a
    kernel that wrote its output at the slot index rather than at the feature
    id would be caught."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.int32](n_features)
        with buf.map_to_host() as host:
            for s in range(n_features):
                host.unsafe_ptr().unsafe_store(
                    s, Int32((n_features - 1) - s)
                )
        return buf^


def _fixed_scale(values: List[Float64]) -> Float32:
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if total < 1e-12:
        total = 1e-12
    return Float32(Float64(1 << 30) / total)


# --- The host reference half ---


def _golden_case(
    n_bins: Int,
    n_features: Int,
    n_rows: Int,
    subset: Bool,
    expect_a_full_group: Bool,
    const_hess: Bool = False,
) raises:
    """One shape, checked against a histogram summed on the host.

    The gradients are integers in [-200, 200] and the hessians integers in
    [1, 97], carried as Float32 and quantized under a scale of exactly 1.0. In
    that regime `Int32(round(x * 1.0))` is `x`, on the host and on the device
    and for both gradient sources, because multiplying a Float32 by one cannot
    round and every one of these values is exactly representable. So the
    reference below is not an approximation of what the kernel should compute;
    it is the same integer sum, and the only thing the comparison can be
    sensitive to is which (row, feature) pairs the kernel visited.

    Every bin sum is bounded by `200 * n_rows`, which is under 700,000 at the
    largest shape here, so nothing in the comparison is near an Int32 edge and
    a failure cannot be an overflow.

    With `const_hess` every hessian is exactly 1.0, which is both what makes
    the declaration true and, under a unit scale, what makes `hq_const` equal
    to one, so the reconstructed hessian plane is the count plane and the same
    host reference covers it. This is the arm worth having a host reference
    for: a walk that dropped the same rows in every arm would leave the
    reconstructed hessian plane consistent with the count plane it was built
    from, so a check of the device against itself could not see it.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _reversed_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()

        rows.set_constant_hessian(const_hess)
        if const_hess:
            if not rows.const_hessian_allowed:
                print("skipped: MOJOTREES_CONST_HESSIAN=0")
                return
            assert_true(rows.constant_hessian)

        var gvals = List[Int](capacity=n_rows)
        var hvals = List[Int](capacity=n_rows)
        for r in range(n_rows):
            gvals.append(Int(_splitmix64(UInt64(r) + 3) % 401) - 200)
            if const_hess:
                hvals.append(1)
            else:
                hvals.append(Int(_splitmix64(UInt64(r) + 77) % 97) + 1)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        with grad_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(gvals[r]))
        with hess_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(hvals[r]))

        var node = 0
        if subset:
            _ = rows.partition(
                bins.unsafe_ptr(),
                0,
                1,
                2,
                RowRouting.numerical(0, n_bins // 2),
            )
            node = 2
            var w = rows.range_of(node)
            assert_true(w.begin > 0)
            assert_true(w.count() > 0)
            assert_true(w.count() < rows.n_active())

        var window = rows.range_of(node)
        assert_true(window.count() > 0)
        if expect_a_full_group:
            # Large enough that the unrolled body can run rather than the
            # whole walk falling to the tail. This is a gate on the shape, not
            # a proof that the body ran: how a range is cut into tiles is the
            # tiling policy's business and this test does not reach into it.
            assert_true(
                window.count() > HIST_ROW_UNROLL * rows.block_threads
            )

        # The rows this node owns, in the order the device holds them. Order
        # does not matter to the sum, but the *set* does, and this is the set.
        var idx = rows.download_range(node)
        assert_equal(len(idx), window.count())

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
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

        var out_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_out = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var cap = histogram_bin_capacity(n_bins)
        var strategies = [STRATEGY_ATOMIC, STRATEGY_TILED]
        var groups = [1, 4]
        var arms = 0
        for si in range(len(strategies)):
            var tiling = rows.range_tiling(
                caps, node, n_features, strategies[si], 1 << 20
            )
            assert_equal(tiling.strategy, strategies[si])
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
                for q in range(2):
                    rows.set_quantized_gradients(q == 1)
                    for u in range(2):
                        rows.set_row_unroll(u == 1)
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
                            assert_equal(Int(got.unsafe_load(i)), golden[i])
                        arms += 1
        # Both strategies, both group rungs that fit, both gradient sources,
        # both row walks. A silently skipped sweep would otherwise pass.
        assert_true(arms >= 8)
        rows.set_feature_group(1)
        rows.set_quantized_gradients(False)
        rows.set_constant_hessian(False)
        rows.set_row_unroll(True)


def test_the_unrolled_walk_matches_a_host_sum() raises:
    """Row counts chosen against the thread stride.

    A thread's walk length is about `count / block_threads`, and the unrolled
    body needs `HIST_ROW_UNROLL` of a thread's rows to remain before it will
    run. At the 256-thread block size this backend derives, one full group is
    1024 rows, so:

    - 900 rows is below one group: every thread's walk is all tail.
    - 2049 and 3073 are one and two rows past whole multiples of 1024, so the
      walk ends with the smallest possible ragged remainder.
    - 2600 is 552 past a multiple, which leaves a remainder of two rows for
      some threads and three for others.

    The split halves each of them again, which lands the subset counts on a
    different set of residues than the root counts, for free.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _golden_case(32, 3, 900, False, False)
        _golden_case(32, 3, 900, True, False)
        _golden_case(64, 17, 2049, False, True)
        _golden_case(64, 17, 2049, True, False)
        _golden_case(256, 5, 3073, True, False)
        _golden_case(32, 8, 2600, False, True)
        # The same reference against the elided hessian plane.
        _golden_case(64, 17, 2049, True, False, True)
        _golden_case(32, 8, 2600, False, True, True)


# --- The cross-arm half ---


def _crossarm_case(
    n_bins: Int, n_features: Int, n_rows: Int, const_hess: Bool
) raises:
    """One shape under realistic gradients and real fixed-point scales, with
    the shipped row walk as the reference.

    The reference arm is `set_row_unroll(False)` at group 1 on the atomic
    strategy reading the Float32 gradient planes, which is the shape this
    module ran before the hist-kernel-margin lane. Every other arm is
    therefore compared against what shipped and not merely against its
    neighbour.

    Unlike the host-reference half above, the scales here are the real
    `2^30 / sum|x|` the trainer uses and the values are not exactly
    representable, so the quantization does round. That is the point: it is
    the arm in which a change to the written form of
    `Int32(round(x * scale))` could move a result by one unit in the last
    place of the product, and one unit is a different integer and therefore a
    different histogram.

    With `const_hess` the hessians are all exactly
    `histogram.CONSTANT_HESSIAN`, which is what makes the declaration true;
    the kernel then accumulates two planes instead of three and reconstructs
    the hessian plane from the count, and the reference arm carries the same
    declaration so the comparison is between two row walks and not between
    two plane counts.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(UInt64(r) + 7) % 2000)) * 0.001 - 1.0
            )
            if const_hess:
                hess.append(1.0)
            else:
                hess.append(
                    Float64(Int(_splitmix64(UInt64(r) + 991) % 1000)) * 0.001
                    + 0.25
                )
        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)

        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _reversed_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()
        rows.set_constant_hessian(const_hess)
        if const_hess:
            if not rows.const_hessian_allowed:
                # `MOJOTREES_CONST_HESSIAN=0` withdrew the permission. Say so
                # rather than returning quietly, because a silent return would
                # leave this case passing without having compared anything.
                print("skipped: MOJOTREES_CONST_HESSIAN=0")
                return
            assert_true(rows.constant_hessian)

        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        with grad_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(grad[r]))
        with hess_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(hess[r]))

        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, n_bins // 2)
        )
        var node = 2
        var window = rows.range_of(node)
        assert_true(window.begin > 0)
        assert_true(window.count() > 0)
        assert_true(window.count() < rows.n_active())

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var ref_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var arm_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_ref = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var host_arm = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var ref_part = ctx.enqueue_create_buffer[DType.int32](1)

        var ref_tiling = rows.range_tiling(
            caps, node, n_features, STRATEGY_ATOMIC, 1
        )
        rows.set_feature_group(1)
        rows.set_quantized_gradients(False)
        rows.set_row_unroll(False)
        rows.enqueue_range_histogram(
            ref_tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            ref_dev.unsafe_ptr(),
            ref_part.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )
        ctx.enqueue_copy(dst_ptr=host_ref.unsafe_ptr(), src_buf=ref_dev)
        ctx.synchronize()

        # Two buffers of zeros would compare equal for the wrong reason, and a
        # walk that ignored the window would count every active row.
        var populated = 0
        var total_count = 0
        for i in range(cells):
            if host_ref.unsafe_ptr().unsafe_load(i) != 0:
                populated += 1
        for b in range(hist_size):
            total_count += Int(
                host_ref.unsafe_ptr().unsafe_load(2 * hist_size + b)
            )
        assert_true(populated > 0)
        assert_equal(total_count, n_features * window.count())

        var cap = histogram_bin_capacity(n_bins)
        var strategies = [STRATEGY_ATOMIC, STRATEGY_TILED]
        var groups = [1, 2, 4, 8, 16]
        var arms = 0
        for si in range(len(strategies)):
            var tiling = rows.range_tiling(
                caps, node, n_features, strategies[si], 1 << 20
            )
            assert_equal(tiling.strategy, strategies[si])
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
                for q in range(2):
                    rows.set_quantized_gradients(q == 1)
                    rows.set_row_unroll(True)
                    rows.enqueue_range_histogram(
                        tiling,
                        node,
                        bins.unsafe_ptr(),
                        grad_dev.unsafe_ptr(),
                        hess_dev.unsafe_ptr(),
                        feat_dev.unsafe_ptr(),
                        arm_dev.unsafe_ptr(),
                        part_dev.unsafe_ptr(),
                        n_features,
                        g_scale,
                        h_scale,
                    )
                    ctx.enqueue_copy(
                        dst_ptr=host_arm.unsafe_ptr(), src_buf=arm_dev
                    )
                    ctx.synchronize()
                    var a = host_ref.unsafe_ptr()
                    var b2 = host_arm.unsafe_ptr()
                    for i in range(cells):
                        assert_equal(
                            Int(a.unsafe_load(i)), Int(b2.unsafe_load(i))
                        )
                    arms += 1
        assert_true(arms >= 8)
        rows.set_feature_group(1)
        rows.set_quantized_gradients(False)
        rows.set_constant_hessian(False)
        rows.set_row_unroll(True)


def test_both_row_walks_build_the_same_histogram() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _crossarm_case(32, 3, 1500, False)
        _crossarm_case(64, 17, 2600, False)
        _crossarm_case(256, 8, 1100, False)


def test_both_row_walks_agree_with_the_hessian_plane_elided() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _crossarm_case(64, 17, 2600, True)
        _crossarm_case(256, 5, 1100, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
