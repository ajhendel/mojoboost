"""The three hist-latency arms, against the shape they shipped with.

The K1 lane added three runtime arms to the histogram path, each switchable
inside one process so that a benchmark can interleave them on a machine whose
device timings drift several-fold between time windows:

- `GpuActiveRows.set_narrow_index` forms the row loop's two data-dependent
  indices in Int32 rather than in Int.
- `GpuActiveRows.set_pair_alignment` states the 8-byte alignment the
  quantized gradient pair's address actually has, instead of leaving the
  width-2 load annotated `align 4`.
- `GpuActiveRows.set_row_tiling` requests a row-tile floor or a rows-per-tile
  length in the call rather than through an environment variable.

What is being tested is exactness, and the three claims are not the same
shape of claim, which is why they are tested differently.

`set_pair_alignment` and `set_row_tiling` are exact whatever the data.
Alignment is an assertion to the code generator over the same eight bytes,
and tiling only changes the order in which fixed-point Int32 adds are issued,
which integer addition does not care about. For those two, a cross-arm
comparison is the whole check.

`set_narrow_index` is exact only *given a bound on the dataset shape*, and
that difference is the interesting one. Above the bound the narrow arm would
wrap an index and accumulate into the wrong bin, silently. So this file
checks the bound at its own boundary on the host, where a two-billion-row
shape costs nothing to state, and separately checks that the setter refuses a
shape the bound rejects rather than honoring it.

The device half compares every arm against the shape that shipped before this
lane: wide indices, `align 4` pair load, no tile request, at group 1 on the
atomic strategy. Comparisons are `assert_equal` on raw Int32 cells and never a
tolerance. It skips (passing) with no accelerator, and the two host-arithmetic
tests then carry the file.

This file takes no `# run_tests: cpu-safe` marker and is therefore classified
GPU-only by name, following `test_gpu_hist_row_unroll.mojo`, which guards its
device work the same way and is classified the same way. The guards are
believed sufficient and the classification is belt: an accelerator entry point
that loses its guard fails only on the x86-64 half of the matrix, which an
Apple machine structurally cannot reproduce, so the cheap answer is to not
compile the file there at all.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    MAX_ROWS,
    GpuActiveRows,
    RowRouting,
    narrow_index_fits,
)
from mojotrees.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    histogram_bin_capacity,
    histogram_shared_bytes,
    query_device_caps,
    resolve_tiling,
    row_tile_floor,
)
from support import _splitmix64


# --- The bound, on the host ---


def test_the_narrow_index_bound_holds_at_its_boundary() raises:
    """`narrow_index_fits` at the exact shapes that straddle it.

    `MAX_ROWS` is `Int32.MAX`, so the bound is `n_features * n_rows` at or
    below 2,147,483,647 and `2 * n_rows` likewise. Both sides of both are
    stated here as shapes rather than as arithmetic, because the failure this
    guards against is a wrapped index and the only convincing check is that
    the predicate flips exactly where the wrap begins.

    Opens no device: it is host arithmetic over two integers, so it is the
    half of this file that still runs when the accelerator half skips.
    """
    # The product exactly at the limit. One row and `MAX_ROWS` features is
    # the only way to spell it that does not also trip the `2 * n_rows`
    # condition, because `MAX_ROWS` is 2^31 - 1 and is prime.
    assert_true(narrow_index_fits(1, MAX_ROWS))

    # The column-offset term, straddled at two features.
    assert_true(narrow_index_fits(MAX_ROWS // 2, 2))
    assert_false(narrow_index_fits(MAX_ROWS // 2 + 1, 2))

    # The `2 * n_rows` term is the binding one at a single feature, and it is
    # the case the column-offset term alone cannot see: `MAX_ROWS * 1` fits
    # the first condition and `2 * MAX_ROWS` fails the second, so a
    # single-feature fit at the row limit is refused by the second condition
    # alone. That is the whole reason the second condition is checked rather
    # than inferred from the first.
    assert_false(narrow_index_fits(MAX_ROWS, 1))
    assert_true(narrow_index_fits(MAX_ROWS // 2, 1))
    assert_false(narrow_index_fits(MAX_ROWS // 2 + 1, 1))

    # The shapes this library is actually run at, on both registered axes,
    # are far inside it. These are the numbers the docstring quotes.
    assert_true(narrow_index_fits(1_000_000, 50))
    assert_true(narrow_index_fits(1_000_000, 2_147))
    assert_false(narrow_index_fits(1_000_000, 2_148))
    assert_true(narrow_index_fits(42_949_672, 50))
    assert_false(narrow_index_fits(42_949_673, 50))

    # A shape that is not a dataset admits nothing.
    assert_false(narrow_index_fits(0, 4))
    assert_false(narrow_index_fits(4, 0))
    assert_false(narrow_index_fits(-1, 4))


# --- The tile requests, on the host ---


def test_a_tile_request_overrides_the_environment_and_zero_does_not() raises:
    """`resolve_tiling` and `row_tile_floor` with and without a request.

    Pure arithmetic and opens no device. Two things are asserted and they are
    the two ways this could have been wired wrongly: a request must reach the
    tile count, and a zero request must leave the answer exactly where it
    was, since zero is what every caller that does not care passes.

    The absolute tile counts below assume `MOJOTREES_GPU_MIN_TILES` and
    `MOJOTREES_GPU_ROW_TILE` are unset, which is what `tools/run_tests.sh`
    runs with. A session that exported either one will see this fail, and
    that is the right outcome: the same export would silently move every
    geometry the suite exercises.

    The bounds below are the M4's: 80 threadgroups wanted device-wide, 256
    threads, 256 bins, a million rows, fifty features. That is the shape the
    registration names, and at it the occupancy term alone gives
    `ceil(80 / 50) = 2` tiles.
    """
    var no_request = resolve_tiling(
        1_000_000, 50, 256, 256, 256, 80, 1 << 24, STRATEGY_ATOMIC
    )
    var zero_request = resolve_tiling(
        1_000_000, 50, 256, 256, 256, 80, 1 << 24, STRATEGY_ATOMIC, 0, 0
    )
    assert_equal(no_request.n_tiles, zero_request.n_tiles)
    assert_equal(no_request.rows_per_tile, zero_request.rows_per_tile)
    assert_equal(no_request.n_tiles, 2)

    # A floor raises the tile count as far as the row-amortization bound
    # allows, which at a million rows and a 2,048-row minimum is 488 tiles,
    # so 4 and 8 both land exactly.
    var four = resolve_tiling(
        1_000_000, 50, 256, 256, 256, 80, 1 << 24, STRATEGY_ATOMIC, 4, 0
    )
    var eight = resolve_tiling(
        1_000_000, 50, 256, 256, 256, 80, 1 << 24, STRATEGY_ATOMIC, 8, 0
    )
    assert_equal(four.n_tiles, 4)
    assert_equal(eight.n_tiles, 8)

    # A floor below the occupancy term is raised to it rather than honored,
    # which is the rule `row_tile_floor` has always applied to the
    # environment variable and now applies to the argument.
    var one = resolve_tiling(
        1_000_000, 50, 256, 256, 256, 80, 1 << 24, STRATEGY_ATOMIC, 1, 0
    )
    assert_equal(one.n_tiles, 2)

    # The rows-per-tile request is the only way to ask for FEWER tiles than
    # the occupancy term gives, which is the arm the tile re-test needs: the
    # earlier experiment only ever moved in one direction.
    var single = resolve_tiling(
        1_000_000,
        50,
        256,
        256,
        256,
        80,
        1 << 24,
        STRATEGY_ATOMIC,
        0,
        1_000_000,
    )
    assert_equal(single.n_tiles, 1)

    # And the floor function itself, at the same bounds.
    assert_equal(row_tile_floor(80, 50), 2)
    assert_equal(row_tile_floor(80, 50, 0), 2)
    assert_equal(row_tile_floor(80, 50, 8), 8)
    assert_equal(row_tile_floor(80, 50, 1), 2)
    with assert_raises():
        _ = row_tile_floor(80, 50, -1)


# --- Shared device fixtures ---


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, built exactly as
    test_gpu_hist_row_unroll.mojo builds one so the two files compare the same
    kind of data."""
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
    """Feature slot `s` carries feature `n_features - 1 - s`.

    A non-identity permutation, so a kernel that indexed a column by
    `block_idx.x` rather than by `feat_ids` would show. The narrow arm
    truncates that column base to Int32, which is precisely the value this
    permutation makes non-trivial.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.int32](n_features)
        with buf.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for s in range(n_features):
                dst.unsafe_store(s, Int32(n_features - 1 - s))
        return buf^


def _fixed_scale(values: List[Float64]) -> Float32:
    """`2^30 / sum|x|`, the scale the trainer uses, so the quantization in
    this file rounds the way a real round rounds."""
    var total = 0.0
    for i in range(len(values)):
        var v = values[i]
        total += v if v >= 0.0 else -v
    if total <= 0.0:
        return Float32(1.0)
    return Float32(Float64(1 << 30) / total)


# --- The device half ---


def _crossarm_case(
    n_bins: Int, n_features: Int, n_rows: Int, const_hess: Bool
) raises:
    """One shape, with the pre-lane shape as the reference.

    The reference arm is wide indices, the weaker pair-load alignment, and no
    tile request, at group 1 on the atomic strategy reading the Float32
    gradient planes. Every arm below is therefore compared against what
    shipped and not merely against its neighbour, which is the mistake a
    round-robin over new arms alone would make.

    The gradients are not exactly representable and the scales are the real
    `2^30 / sum|x|`, so quantization rounds. The pair-alignment arm is the one
    that could plausibly move a value if it were not what it claims to be,
    since it is the only arm that touches how the quantized words are read.
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
                print("skipped: MOJOTREES_CONST_HESSIAN=0")
                return
            assert_true(rows.constant_hessian)

        # Every shape this file runs is small, so the bound holds and the
        # narrow arm is reachable. Asserting it here rather than assuming it
        # keeps a future shape change from turning the narrow arms below into
        # silently skipped repeats of the wide ones.
        assert_true(rows.narrow_index_supported())

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
        rows.set_narrow_index(False)
        rows.set_pair_alignment(False)
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

        # Two buffers of zeros would compare equal for the wrong reason, and
        # a walk that ignored the window would count every active row.
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
        var groups = [1, 4]
        # Tile requests as (min_tiles, rows_per_tile) pairs: no request, the
        # two floors the registration names, and one tile, which only the
        # rows-per-tile spelling can reach.
        var tile_min = [0, 4, 8, 0]
        var tile_rows = [0, 0, 0, n_rows]
        var arms = 0
        for si in range(len(strategies)):
            for gi in range(len(groups)):
                var group = groups[gi]
                if (
                    histogram_shared_bytes(cap, group)
                    > caps.max_shared_memory_per_block
                ):
                    continue
                rows.set_feature_group(group)
                for ti in range(len(tile_min)):
                    rows.set_row_tiling(tile_min[ti], tile_rows[ti])
                    # The geometry has to be re-derived after the request,
                    # because it is `range_tiling` that reads it.
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
                    for q in range(2):
                        rows.set_quantized_gradients(q == 1)
                        for ni in range(2):
                            rows.set_narrow_index(ni == 1)
                            for ai in range(2):
                                rows.set_pair_alignment(ai == 1)
                                rows.set_row_unroll(arms % 2 == 0)
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
                                    dst_ptr=host_arm.unsafe_ptr(),
                                    src_buf=arm_dev,
                                )
                                ctx.synchronize()
                                var a = host_ref.unsafe_ptr()
                                var b2 = host_arm.unsafe_ptr()
                                for i in range(cells):
                                    assert_equal(
                                        Int(a.unsafe_load(i)),
                                        Int(b2.unsafe_load(i)),
                                    )
                                arms += 1
        # Two strategies, two group rungs, four tile requests, two gradient
        # sources, two index widths, two alignments. A sweep that silently
        # skipped a dimension would otherwise pass with nothing compared.
        assert_true(arms >= 64)
        rows.set_feature_group(1)
        rows.set_quantized_gradients(False)
        rows.set_constant_hessian(False)
        rows.set_row_unroll(True)
        rows.set_narrow_index(False)
        rows.set_pair_alignment(True)
        rows.set_row_tiling(0, 0)


def test_every_index_arm_builds_the_same_histogram() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _crossarm_case(32, 3, 1500, False)
        _crossarm_case(64, 17, 2600, False)


def test_every_index_arm_agrees_with_the_hessian_plane_elided() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _crossarm_case(64, 17, 2600, True)


def test_the_defaults_are_the_ones_the_lane_registered() raises:
    """Off for the index width, on for the pair alignment, no tile request.

    Stated as a test because the three defaults were argued separately and
    for different reasons, and a later edit that flipped one to match another
    would be a change to what every fit runs rather than a tidy-up. The
    arguments are at the fields in `gpu_active_rows.mojo`.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 1024, 4, 32, caps)
        assert_false(rows.narrow_index)
        assert_true(rows.pair_alignment)
        assert_equal(rows.min_tiles_request, 0)
        assert_equal(rows.rows_per_tile_request, 0)


def test_the_setters_refuse_what_they_cannot_honor() raises:
    """A negative tile request raises, and so does an index-width request on
    a shape the bound rejects.

    The second cannot be reached through a constructed `GpuActiveRows` on any
    machine, since a shape that fails the bound needs a binned matrix above
    2.1 GB, which is why `narrow_index_fits` is tested separately above. What
    is checked here is the half that is reachable: the setter raises rather
    than clamping, so a caller who asked for something impossible hears about
    it instead of getting a quietly different launch.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 1024, 4, 32, caps)
        with assert_raises():
            rows.set_row_tiling(-1, 0)
        with assert_raises():
            rows.set_row_tiling(0, -1)
        # A shape well inside the bound is accepted, both ways.
        rows.set_narrow_index(True)
        assert_true(rows.narrow_index)
        rows.set_narrow_index(False)
        assert_false(rows.narrow_index)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
