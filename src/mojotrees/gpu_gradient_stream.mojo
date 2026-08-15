"""Gradient planes for a GPU round: their layout, their staging, and the
row sample that scales them.

This module holds the pieces of the start of a boosting round that are
neither the objective itself (gpu_objectives_native.mojo) nor the histogram
(histogram_gpu.mojo, gpu_active_rows.mojo), and that today force a round
onto a slower path than the hardware needs.

There are three, and the third has its own banner further down: a device
image of a row sample, a one-pass host staging path for gradients that
genuinely originate on the host, and the interleaved derivative plane,
which changes how every histogram in a tree reads the two derivatives.

`GpuRowSelection` is the device image of a row sample. A sample is two
arrays, the selected row ids and their compensation multipliers, and the
GPU trainer currently has no device image of either, so `device_gradients`
in train_gpu.mojo refuses the device objective path whenever bagging or
GOSS is configured and the whole round falls back to host-side derivatives
plus a `2 * n_rows` Float32 upload. Nothing about a sample requires that.
The gradients still come from `GpuObjectiveState`, over every row, exactly
as they do without sampling; the sample only decides which rows the root
range covers (already device side, `begin_tree(bag)`) and which rows carry
a multiplier. This struct supplies the second half: the row ids and the
multipliers as device buffers, and one kernel that applies the multipliers
in place, in the same operand order as `apply_goss_scaling` in goss.mojo,
so the gradients the histogram reads already carry the compensation.

`GpuRowSelection` also owns the device-side GOSS ranking score,
`|grad * hess|`, so a trainer that wants to sample from device-resident
gradients downloads one Float32 plane instead of needing host-side
derivatives at all. That download is the only reason the score exists;
the selection rule itself (quickselect threshold, forward pass, counter
stream) stays in goss.mojo, unchanged and host side, because it is a
ranking over the whole row set and a partial-order pass, not a per-row map.

`HostGradientStage` is the other end. Where the gradients genuinely
originate on the host, which is the custom-objective contract and nothing
else, `GpuHistogramBuilder.stage_gradients` walks the two Float64 lists
three times: once for the gradient magnitude sum, once for the hessian
magnitude sum, and once to convert into the pinned Float32 staging
buffers. Those are the same three reads of the same two arrays. This
struct fuses them into one pass that accumulates both magnitude sums in
Float64 while it converts, which is where the "streaming" design in the
handoff actually pays: not on the device, where the planes have to be
materialized anyway, but on the host-origin path, where a value is
touched three times for no reason. The sums accumulate in index order over
the Float64 inputs, which is exactly what `_fixed_scale` does, so the
scale that comes out of `device_fixed_scale` is bit-identical to the one
`stage_gradients` computes today.

What is deliberately not here
-----------------------------
No objective arms. A row-restricted gradient kernel would have to repeat
the per-objective derivative chain that `_grad_hess_kernel` already owns,
and a second definition of the objectives is worse than the pass it would
save. Gradients are produced for every row by the existing kernel and the
sample is applied afterwards, which is also what the host path does
(`_fill_grad_hess` over all rows, then `apply_goss_scaling` over the
sample), so the two stay comparable term by term.

No chunked upload. True overlap of conversion with transfer needs a copy
into a sub-range of a device buffer, and the copy entry points reached
from here take whole buffers. The single fused pass is the part that does
not need it; the handoff names the sub-range copy as the follow-up.

No edit to the shipped histogram kernels. The interleaved path below is
additive and off by default, so the split-plane path it competes with is
byte for byte the one that shipped, and the two can be run against each
other on one build. That is a requirement of measuring them, not only a
courtesy to the lane that owns those files.

Precision
---------
The device carries gradients, hessians, and multipliers as Float32. The
host path multiplies Float64 gradients by a Float64 multiplier and rounds
once, at staging; this path rounds the gradient and the multiplier first
and multiplies in Float32. The products therefore agree to Float32, not
bit-exactly, which is the same trade every other device stage makes. It
matters in one specific place and the handoff says so: the GOSS *ranking*
score is a function of the gradients, so a Float32 score can order two
near-equal rows differently from the Float64 one and put a different row
on the wrong side of the top-k threshold. Row bagging has no such
exposure, because a bag is drawn from a counter stream keyed by
(seed, bag index, row) and never looks at a gradient.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import isfinite, round
from std.memory import stack_allocation
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .binning import MAX_BINS
from .goss import GossSelection
from .gpu_active_rows import (
    GpuActiveRows,
    _range_reduce_kernel,
    _zero_int32_kernel,
)
from .gpu_objectives_native import GradMagnitudes
from .gpu_tiling import (
    STRATEGY_TILED,
    HistogramTiling,
    derive_block_threads,
    query_device_caps,
)
from .histogram_gpu import GpuHistogramBuilder


# Which layout the derivative planes are in when a histogram reads them.
# SPLIT is the shipped one, two Float32 planes indexed by row. INTERLEAVED
# is the pair layout below. The default is SPLIT: the interleaved path is
# unmeasured, and this module's contract is that nothing changes until a
# benchmark says it should.
comptime LAYOUT_SPLIT = 0
comptime LAYOUT_INTERLEAVED = 1


def env_grad_layout() -> Int:
    """`MOJOTREES_GPU_GRAD_LAYOUT` as a layout constant. Anything other
    than `interleaved` is the shipped split layout, so an unset or
    misspelled variable cannot silently move a run onto the new path."""
    if getenv("MOJOTREES_GPU_GRAD_LAYOUT") == "interleaved":
        return LAYOUT_INTERLEAVED
    return LAYOUT_SPLIT


def stream_layout_name(layout: Int) -> String:
    if layout == LAYOUT_INTERLEAVED:
        return String("interleaved")
    return String("split")


def _row_scale_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    count: Int32,
):
    """Apply one compensation multiplier per selected row, in place.

    One thread per entry of the selection, not per row of the dataset: a
    GOSS round selects a fraction of the rows and this kernel touches
    exactly those. `apply_goss_scaling` skips an entry whose multiplier is
    1.0 and so does this, which is a bandwidth choice rather than a
    numerical one, since multiplying a finite Float32 by exactly 1.0 is the
    identity. The high-gradient rows GOSS keeps are all at 1.0, so the skip
    covers most of the selection.

    Both planes are scaled by the same multiplier and in the same operand
    order the host does (`grad[r] * s`, `hess[r] * s`), so a row's
    compensated derivatives here are the Float32 image of the host's.
    """
    var j = global_idx.x
    if j >= Int(count):
        return
    var s = scale[unsafe_offset=j][0]
    if s == 1.0:
        return
    var r = Int(rows[unsafe_offset=j][0])
    grad[unsafe_offset=r] = grad[unsafe_offset=r][0] * s
    hess[unsafe_offset=r] = hess[unsafe_offset=r][0] * s


def _importance_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    imp: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """LightGBM's GOSS ranking score `|grad * hess|`, one row per thread.

    The same expression as `goss_importance`, over every row, so a host
    sampler fed from here ranks the rows it would have ranked. Sample
    weights are already folded into both derivatives by the gradient
    kernel, so a zero-weight row scores zero and ranks last, which is the
    documented behavior of the host score.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    imp[unsafe_offset=r] = abs(
        grad[unsafe_offset=r][0] * hess[unsafe_offset=r][0]
    )


struct GpuRowSelection(Movable):
    """One round's row sample, device side.

    Construct once per training session with the capacity the sampler can
    ask for (`max_selected`, in practice `n_rows`, since a bag is Binomial
    and a GOSS selection can exceed `top_k + other_k` on importance ties).
    A session that never samples constructs with `max_selected = 0` and
    pays one element per placeholder buffer.

    Per round: `select_all` for an unsampled round, or `set_selection`
    (or `from_goss`) with the sampler's output, then `apply_compensation`
    after the gradients are filled and before their magnitudes are
    reduced. That order is the host path's order, where `goss_round`
    scales the gradients and `upload_gradients` then derives the
    fixed-point scale from the scaled values.

    The selection is a device image of a host decision, not a device
    decision. Bags come from bagging.mojo and GOSS selections from
    goss.mojo, both unchanged, so both backends still sample identical
    rows from identical seeds.
    """

    var rows_dev: DeviceBuffer[DType.int32]
    """Selected row ids, ascending, in the sampler's order. The first
    `n_selected` entries are live."""
    var scale_dev: DeviceBuffer[DType.float32]
    """Compensation multiplier per entry of `rows_dev`, 1.0 for a row the
    sampler kept outright."""
    var imp_dev: DeviceBuffer[DType.float32]
    """`|grad * hess|` per row, or a one-element placeholder when the
    session did not ask for the ranking score."""
    var stage_rows: HostBuffer[DType.int32]
    var stage_scale: HostBuffer[DType.float32]
    var host_imp: HostBuffer[DType.float32]
    var n_rows: Int
    var max_selected: Int
    var n_selected: Int
    var block_threads: Int
    var has_importance: Bool
    var has_compensation: Bool
    """Whether this round's selection carried multipliers at all. A bag
    does not, so a bagged round never launches the scaling kernel."""

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        max_selected: Int = 0,
        with_importance: Bool = False,
    ) raises:
        """Allocate the selection buffers once for the whole session.

        `max_selected` of 0 means this session never samples, and both
        selection buffers become one-element placeholders (zero-length
        device buffers are not portable). `with_importance` allocates the
        `n_rows` ranking plane and its pinned readback buffer, which only
        a GOSS session needs.
        """
        if n_rows < 1:
            raise Error("row selection requires at least one row")
        if max_selected < 0:
            raise Error("max_selected must be nonnegative")
        if max_selected > n_rows:
            raise Error("max_selected cannot exceed n_rows")

        self.n_rows = n_rows
        self.max_selected = max_selected
        self.n_selected = 0
        self.has_importance = with_importance
        self.has_compensation = False
        self.block_threads = derive_block_threads(query_device_caps(ctx))

        var cap = max_selected if max_selected > 0 else 1
        var imp_cap = n_rows if with_importance else 1
        self.rows_dev = ctx.enqueue_create_buffer[DType.int32](cap)
        self.scale_dev = ctx.enqueue_create_buffer[DType.float32](cap)
        self.imp_dev = ctx.enqueue_create_buffer[DType.float32](imp_cap)
        self.stage_rows = ctx.enqueue_create_host_buffer[DType.int32](cap)
        self.stage_scale = ctx.enqueue_create_host_buffer[DType.float32](cap)
        self.host_imp = ctx.enqueue_create_host_buffer[DType.float32](imp_cap)

        # The copies below take whole buffers, so the tail beyond a
        # round's selection is transferred whatever it holds. No kernel
        # reads past `n_selected`, so this is hygiene rather than
        # correctness, and it is the same choice gpu_active_rows.mojo
        # makes for its staging buffer.
        var dst_rows = self.stage_rows.unsafe_ptr()
        var dst_scale = self.stage_scale.unsafe_ptr()
        for i in range(cap):
            dst_rows.unsafe_store(i, Int32(0))
            dst_scale.unsafe_store(i, Float32(1.0))

    def select_all(mut self):
        """Train this round on every row with no compensation. An empty
        selection means the same thing here that an empty bag means
        everywhere else in the library."""
        self.n_selected = 0
        self.has_compensation = False

    def is_all_rows(self) -> Bool:
        return self.n_selected == 0

    def selected(self) -> Int:
        """Rows in the current selection, 0 meaning every row."""
        return self.n_selected

    def set_selection(
        mut self,
        ctx: DeviceContext,
        rows: List[Int],
        scale: List[Float64] = [],
    ) raises:
        """Upload this round's sample.

        `rows` holds the selected row ids in the sampler's order, which is
        the order the root range is seeded in, so the compacted rows track
        the CPU grower's row list index for index. `scale` is either empty
        (no compensation, which is what row bagging wants) or one
        multiplier per entry of `rows`.

        An empty `rows` is `select_all`, so a `GossSelection.all_rows()`
        passed straight through means what it means on the host.
        """
        if len(rows) == 0:
            if len(scale) != 0:
                raise Error(
                    "a compensation multiplier without a selected row has"
                    " nothing to scale"
                )
            self.select_all()
            return
        if len(rows) > self.max_selected:
            raise Error(
                "selection is larger than the capacity this session was"
                " constructed with"
            )
        if len(scale) != 0 and len(scale) != len(rows):
            raise Error(
                "compensation multipliers must be one per selected row"
            )
        for i in range(len(rows)):
            if rows[i] < 0 or rows[i] >= self.n_rows:
                raise Error("selected row index out of range")
        for i in range(len(scale)):
            if not isfinite(scale[i]) or scale[i] < 0.0:
                raise Error(
                    "compensation multipliers must be finite and"
                    " nonnegative"
                )

        # Any copy still reading the staging buffers has to finish before
        # they are overwritten.
        ctx.synchronize()
        var dst_rows = self.stage_rows.unsafe_ptr()
        var dst_scale = self.stage_scale.unsafe_ptr()
        for i in range(len(rows)):
            dst_rows.unsafe_store(i, Int32(rows[i]))
        if len(scale) == 0:
            for i in range(len(rows)):
                dst_scale.unsafe_store(i, Float32(1.0))
        else:
            for i in range(len(scale)):
                dst_scale.unsafe_store(i, Float32(scale[i]))
        ctx.enqueue_copy(dst_buf=self.rows_dev, src_ptr=dst_rows)
        ctx.enqueue_copy(dst_buf=self.scale_dev, src_ptr=dst_scale)
        self.n_selected = len(rows)
        self.has_compensation = len(scale) > 0

    def from_goss(
        mut self, ctx: DeviceContext, selection: GossSelection
    ) raises:
        """`set_selection` from a `GossSelection`, so the sampler's output
        reaches the device without a caller unpacking it. The no-sampling
        selection (`GossSelection.all_rows()`) becomes `select_all`."""
        self.set_selection(ctx, selection.rows, selection.scale)

    def from_bag(mut self, ctx: DeviceContext, bag: List[Int]) raises:
        """`set_selection` from a row bag: the ids, and deliberately no
        multipliers.

        Bagging changes which rows a tree is grown on and never what a
        row's derivatives are, so a bagged round carries no compensation
        and `apply_compensation` stays a no-op for it. That is the whole
        difference from `from_goss`, and it is why a bagged round is free
        of the Float32 ranking caveat: the bag comes from a counter stream
        keyed by (seed, bag index, row) and never reads a gradient.

        An empty bag is `select_all`, which is what an empty bag means
        everywhere else in the library.
        """
        self.set_selection(ctx, bag)

    def apply_compensation(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Scale the selected rows' gradients and hessians in place.

        A no-op when the selection is every row or when no multipliers
        were supplied, which is the row-bagging case: bagging changes
        which rows a tree sees, never what a row's derivatives are.

        Call after the gradient kernel and before the magnitude reduction.
        The scale the histograms quantize with then describes the values
        the histograms will actually read, which is the invariant the
        fixed-point accumulation depends on for its overflow bound.
        """
        if self.n_selected <= 0 or not self.has_compensation:
            return
        var blocks = (
            self.n_selected + self.block_threads - 1
        ) // self.block_threads
        ctx.enqueue_function[_row_scale_kernel](
            self.rows_dev.unsafe_ptr(),
            self.scale_dev.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            Int32(self.n_selected),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )

    def enqueue_importance(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Fill the ranking plane from the round's uncompensated
        gradients. Does not transfer or synchronize.

        Call before `apply_compensation`: the host sampler ranks rows by
        the derivatives the objective produced, and the multiplier it
        hands back is a consequence of that ranking, not an input to it.
        """
        if not self.has_importance:
            raise Error(
                "this selection was constructed without the ranking plane;"
                " construct with with_importance=True"
            )
        var blocks = (
            self.n_rows + self.block_threads - 1
        ) // self.block_threads
        ctx.enqueue_function[_importance_kernel](
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            self.imp_dev.unsafe_ptr(),
            Int32(self.n_rows),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )

    def download_importance(
        mut self, ctx: DeviceContext
    ) raises -> List[Float64]:
        """The ranking plane, host side, ready for `goss_select`.

        One `n_rows` Float32 transfer and one synchronization, against the
        `2 * n_rows` Float32 upload the host-gradient path pays every
        round, and it replaces a full host-side objective evaluation
        rather than adding to one. The values are the Float32 image of
        `goss_importance`, so a selection drawn from them can differ from
        the CPU sampler's on rows whose scores tie to within Float32; see
        the module docstring.
        """
        if not self.has_importance:
            raise Error(
                "this selection was constructed without the ranking plane;"
                " construct with with_importance=True"
            )
        ctx.enqueue_copy(
            dst_ptr=self.host_imp.unsafe_ptr(), src_buf=self.imp_dev
        )
        ctx.synchronize()
        var src = self.host_imp.unsafe_ptr()
        var out = List[Float64](capacity=self.n_rows)
        for r in range(self.n_rows):
            out.append(Float64(src.unsafe_load(r)))
        return out^


def selection_capacity(
    n_rows: Int, bagging_on: Bool, goss_on: Bool
) raises -> Int:
    """`max_selected` for a session, from what the run is configured to
    sample.

    `n_rows` whenever anything samples, and 0 (placeholder buffers only)
    when nothing does. Neither sampler has a tighter bound that holds every
    round: a bag is a Binomial draw, so its size is only `bagging_fraction`
    in expectation, and a GOSS selection can exceed `top_k + other_k` when
    importances tie. Sizing to the looser bound costs `8 * n_rows` bytes
    once per session and removes a reallocation the trainer would otherwise
    have to reason about mid-run.
    """
    if n_rows < 1:
        raise Error("row selection requires at least one row")
    return n_rows if (bagging_on or goss_on) else 0


def selection_wants_ranking(
    goss_on: Bool, allow_device_ranking: Bool
) -> Bool:
    """Whether a session needs the `|grad * hess|` ranking plane.

    Only a GOSS run whose caller has accepted a Float32 ranking, which is
    the same condition `round_eligibility` gates on: without that
    acceptance the sample is drawn from host-side Float64 gradients and the
    plane would never be read. Passing this as `with_importance` is what
    keeps the `n_rows` Float32 plane and its pinned readback buffer out of
    every run that does not sample.
    """
    return goss_on and allow_device_ranking


struct HostGradientStage(Movable):
    """Single-pass staging for gradients that genuinely originate on the
    host.

    The custom-objective trainers hand back two `List[Float64]` per round
    and there is no device image of the callback that produced them, so
    those values have to be converted and uploaded. `stage_gradients` in
    histogram_gpu.mojo reads both lists three times to do it. This reads
    them once.

    Not a replacement for the device objective path and not reachable
    from it: a built-in objective produces its derivatives on the device
    and nothing per row is staged at all. This is the fallback's fast
    path, which is why it lives next to the selection rather than inside
    the builder.
    """

    var stage_g: HostBuffer[DType.float32]
    var stage_h: HostBuffer[DType.float32]
    var n_rows: Int
    var staged: Bool

    def __init__(out self, ctx: DeviceContext, n_rows: Int) raises:
        if n_rows < 1:
            raise Error("gradient staging requires at least one row")
        self.n_rows = n_rows
        self.staged = False
        self.stage_g = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        self.stage_h = ctx.enqueue_create_host_buffer[DType.float32](n_rows)

    def stage(
        mut self,
        ctx: DeviceContext,
        grad: List[Float64],
        hess: List[Float64],
    ) raises -> GradMagnitudes:
        """Convert both planes into the pinned Float32 buffers and return
        the two magnitude sums, in one pass over the inputs.

        The sums accumulate `abs(values[i])` in Float64 in ascending index
        order, which is exactly what `_fixed_scale` accumulates, so
        `device_fixed_scale` applied to these totals returns the Float32
        scale `stage_gradients` would have computed, bit for bit. Feeding
        the result to a builder therefore quantizes identically; only the
        number of times the two lists were read has changed.

        Finiteness is checked by the scale, not here, so that a
        non-finite gradient produces the same error message it produces on
        every other path.
        """
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")

        # Any copy still reading the staging buffers has to finish before
        # they are overwritten.
        ctx.synchronize()
        self.staged = False

        var dst_g = self.stage_g.unsafe_ptr()
        var dst_h = self.stage_h.unsafe_ptr()
        var src_g = grad.unsafe_ptr()
        var src_h = hess.unsafe_ptr()
        var g_total = 0.0
        var h_total = 0.0
        for r in range(self.n_rows):
            var g = src_g.unsafe_load(r)
            var h = src_h.unsafe_load(r)
            g_total += abs(g)
            h_total += abs(h)
            dst_g.unsafe_store(r, Float32(g))
            dst_h.unsafe_store(r, Float32(h))
        self.staged = True
        return GradMagnitudes(g_total, h_total)

    def upload(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Copy the staged planes into caller-owned device buffers, which
        must hold at least `n_rows` Float32 on this context. Enqueued, not
        synchronized."""
        if not self.staged:
            raise Error("call stage before upload")
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=self.stage_g.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=self.stage_h.unsafe_ptr())


# --------------------------------------------------------------------------
# The interleaved derivative plane.
#
# A histogram reads a row's gradient and its hessian together, always, and
# never one without the other. The shipped layout keeps them in two planes
# `n_rows` apart, so a thread issues two loads to two addresses that share
# no cache line. Below the root the row indices are a permutation, so those
# loads are a gather: a warp of 32 threads touches up to 32 lines in the
# gradient plane and 32 more in the hessian plane.
#
# Interleaved, a row's pair is `gh[2r]` and `gh[2r + 1]`, eight bytes that
# are eight-byte aligned and therefore always inside one cache line. The
# same warp touches 32 lines instead of 64.
#
# That halving is unconditional, and it is worth being precise about why,
# because an earlier draft of this lane's handoff overstated the
# uncertainty. Whether the gather hits or misses, the transaction count
# halves: on a miss it halves the lines fetched from memory, on a hit it
# halves the lookups. What a measurement is still needed for is how much
# wall clock that buys, which depends on whether the gather is the
# bottleneck at a given shape, and on nothing here.
#
# The root is the one node where it is neutral rather than a win: the
# unbagged root's permutation is the identity, so both layouts read
# consecutive rows and both coalesce perfectly.
#
# Why this lives here rather than in the histogram modules. The kernels
# below are `_range_hist_partial_kernel` and `_range_hist_atomic_kernel`
# from gpu_active_rows.mojo with exactly two lines changed, the two loads.
# Those files belong to another lane and were mid-flight when this was
# written, so this path is additive: nothing existing changes, no signature
# moves, no test is invalidated, and the two layouts stay side by side,
# which is what comparing them needs anyway. THE DUPLICATION IS A DEBT.
# Integration should collapse the two pairs into one pair parameterized by
# layout and delete these, exactly as `_range_reduce_kernel`'s own
# docstring asks for the copy that module already carries. Until then, a
# change to the accumulation rule has to be made in both places.
#
# The reduction and the zeroing kernels are not duplicated: they depend on
# the partial layout and not on how the derivatives were read, so this
# imports and reuses them.
#
# Results are bit-identical to the split path. `pack` copies Float32
# values with no arithmetic, so `gh[2r]` is exactly `grad[r]`, and the
# accumulation order, the fixed-point rounding, and the atomics are
# unchanged. A histogram built either way is the same integer histogram.
# --------------------------------------------------------------------------


def _pack_gh_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gh: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """Interleave the two derivative planes into pairs. One row per
    thread, a pure copy: no arithmetic, so no value changes."""
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var i = 2 * r
    gh[unsafe_offset=i] = grad[unsafe_offset=r][0]
    gh[unsafe_offset = i + 1] = hess[unsafe_offset=r][0]


def _range_hist_partial_gh_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    gh: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_slots: Int32,
    n_bins: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """`_range_hist_partial_kernel` reading one interleaved plane instead
    of two. Same partial layout, same write-once-no-atomics contract, so
    the existing reduction kernel combines these partials unchanged."""
    var slot = Int(block_idx.x)
    var f = Int(feat_ids[unsafe_offset=slot][0])
    var t = block_idx.y
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var n = Int(count)
    var plane = Int(n_slots) * nb

    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = t * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = Int(begin) + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        # The one change from the split-plane kernel: both derivatives
        # come from one eight-byte aligned pair rather than from two
        # planes n_rows apart.
        var gi = 2 * r
        var gq = Int32(round(gh[unsafe_offset=gi][0] * g_scale))
        var hq = Int32(round(gh[unsafe_offset = gi + 1][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var base = t * 3 * plane + slot * nb
    b = tid
    while b < nb:
        partials[unsafe_offset = base + b] = sg[unsafe_offset=b][0]
        partials[unsafe_offset = base + plane + b] = sh[unsafe_offset=b][0]
        partials[unsafe_offset = base + 2 * plane + b] = sc[
            unsafe_offset=b
        ][0]
        b += block_dim.x


def _range_hist_atomic_gh_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    gh: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_bins: Int32,
    hist_size: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """`_range_hist_atomic_kernel` reading one interleaved plane instead
    of two."""
    var f = Int(feat_ids[unsafe_offset = Int(block_idx.x)][0])
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var n = Int(count)

    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = block_idx.y * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = Int(begin) + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gi = 2 * r
        var gq = Int32(round(gh[unsafe_offset=gi][0] * g_scale))
        var hq = Int32(round(gh[unsafe_offset = gi + 1][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var base = f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(base + b), sg[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(hs + base + b), sh[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(2 * hs + base + b),
                sc[unsafe_offset=b][0],
            )
        b += block_dim.x


def enqueue_range_histogram_interleaved[
    bins_origin: MutOrigin,
    gh_origin: MutOrigin,
    feat_origin: MutOrigin,
    out_origin: MutOrigin,
    part_origin: MutOrigin, //,
](
    mut rows: GpuActiveRows,
    tiling: HistogramTiling,
    node: Int,
    bins: MutPointer[UInt8, bins_origin],
    gh: MutPointer[Float32, gh_origin],
    feat_ids: MutPointer[Int32, feat_origin],
    out_hist: MutPointer[Int32, out_origin],
    partials: MutPointer[Int32, part_origin],
    n_slots: Int,
    g_scale: Float32,
    h_scale: Float32,
) raises:
    """`GpuActiveRows.enqueue_range_histogram` over the interleaved plane.

    Same arguments, same output layout, same two strategies, same zeroing
    rule, and the same reduction kernel; only the plane the derivatives
    come from differs. The histogram it produces is bit-identical to the
    one the split-plane path produces from the same values.
    """
    if n_slots < 1 or n_slots > rows.n_features:
        raise Error("active feature count out of range")
    var window = rows.ranges.get(node)
    var hist_size = rows.n_features * rows.n_bins
    var threads = tiling.block_threads

    # The reduction of the tiled path writes every active feature's slice,
    # so that path only needs zeroing when some feature is inactive; the
    # atomic path always does, and so does a node with no rows.
    if (
        tiling.strategy != STRATEGY_TILED
        or n_slots < rows.n_features
        or window.count() <= 0
    ):
        var cells = 3 * hist_size
        var zero_blocks = (cells + threads - 1) // threads
        rows.ctx.enqueue_function[_zero_int32_kernel](
            out_hist,
            Int32(cells),
            grid_dim=zero_blocks,
            block_dim=threads,
        )
    if window.count() <= 0:
        return

    if tiling.strategy == STRATEGY_TILED:
        rows.ctx.enqueue_function[_range_hist_partial_gh_kernel](
            bins,
            rows.rows_dev.unsafe_ptr(),
            gh,
            feat_ids,
            partials,
            Int32(rows.n_rows),
            Int32(n_slots),
            Int32(rows.n_bins),
            Int32(tiling.rows_per_tile),
            Int32(window.begin),
            Int32(window.count()),
            g_scale,
            h_scale,
            grid_dim=(n_slots, tiling.n_tiles),
            block_dim=threads,
        )
        var n_cells = 3 * n_slots * rows.n_bins
        var blocks = (n_cells + threads - 1) // threads
        # No fused subtraction here: the interleaved plane serves the
        # gradient stream, which builds one node at a time and holds no
        # resident sibling to derive.
        rows.ctx.enqueue_function[_range_reduce_kernel](
            partials,
            feat_ids,
            out_hist,
            Int32(n_slots),
            Int32(rows.n_bins),
            Int32(hist_size),
            Int32(tiling.n_tiles),
            Int32(0),
            Int32(0),
            grid_dim=blocks,
            block_dim=threads,
        )
    else:
        rows.ctx.enqueue_function[_range_hist_atomic_gh_kernel](
            bins,
            rows.rows_dev.unsafe_ptr(),
            gh,
            feat_ids,
            out_hist,
            Int32(rows.n_rows),
            Int32(rows.n_bins),
            Int32(hist_size),
            Int32(tiling.rows_per_tile),
            Int32(window.begin),
            Int32(window.count()),
            g_scale,
            h_scale,
            grid_dim=(n_slots, tiling.n_tiles),
            block_dim=threads,
        )


struct InterleavedGradients(Movable):
    """The interleaved derivative plane and the pass that fills it.

    Construct once per training session on the builder's context, then per
    round, after the gradients are final and their magnitudes reduced,
    call `pack`. Histograms then read this plane through
    `enqueue_leaf_interleaved` instead of `builder.enqueue_leaf`.

    Costs `8 * n_rows` bytes of device memory (8 MB at a million rows) and
    one `16 * n_rows` byte pass per round to fill. Both are paid once per
    round against a gather that happens once per (node, feature), so at
    fifty features and six histogram passes per tree the pack is under one
    percent of what it reorganizes. The pack disappears entirely if the
    derivative kernels are ever taught to write pairs directly, which is a
    change to `_grad_hess_kernel` in gpu_objectives_native.mojo and
    belongs to that lane.

    Order matters in one place: pack after the magnitude reduction and
    after any compensation, so the plane holds the values the histograms
    will actually read.
    """

    var gh_dev: DeviceBuffer[DType.float32]
    """`[grad_0, hess_0, grad_1, hess_1, ...]`, `2 * n_rows` Float32."""
    var n_rows: Int
    var block_threads: Int
    var packed: Bool

    def __init__(out self, ctx: DeviceContext, n_rows: Int) raises:
        if n_rows < 1:
            raise Error("interleaved gradients require at least one row")
        self.n_rows = n_rows
        self.packed = False
        self.block_threads = derive_block_threads(query_device_caps(ctx))
        self.gh_dev = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)

    def pack(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Fill the interleaved plane from the two split planes. Enqueued,
        not synchronized, and a pure copy, so the plane holds exactly the
        Float32 values the split planes hold."""
        var blocks = (
            self.n_rows + self.block_threads - 1
        ) // self.block_threads
        ctx.enqueue_function[_pack_gh_kernel](
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            self.gh_dev.unsafe_ptr(),
            Int32(self.n_rows),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )
        self.packed = True


def enqueue_leaf_interleaved(
    mut builder: GpuHistogramBuilder,
    mut planes: InterleavedGradients,
    leaf: Int,
) raises:
    """`GpuHistogramBuilder.enqueue_leaf` reading the interleaved plane.

    A drop-in for that method: same node, same active feature set, same
    per-node launch geometry, same output buffer, so `download_raw` and
    `histogram_from_host` follow unchanged. Call `planes.pack` once for
    the round first.
    """
    if not builder.has_gradients:
        raise Error("fill the gradients before building a leaf")
    if not planes.packed:
        raise Error("call pack before building a leaf from the pair plane")
    if planes.n_rows != builder.n_rows:
        raise Error(
            "the interleaved plane and the builder disagree on the row count"
        )
    if leaf < 0:
        raise Error("leaf id must be nonnegative")

    var n_slots = len(builder.active)
    var tiling = builder.rows.range_tiling(
        builder.caps,
        leaf,
        n_slots,
        builder.tiling.strategy,
        builder.part_capacity,
    )
    enqueue_range_histogram_interleaved(
        builder.rows,
        tiling,
        leaf,
        builder.bins_dev.unsafe_ptr(),
        planes.gh_dev.unsafe_ptr(),
        builder.feat_dev.unsafe_ptr(),
        builder.out_dev.unsafe_ptr(),
        builder.part_dev.unsafe_ptr(),
        n_slots,
        Float32(builder.g_scale),
        Float32(builder.h_scale),
    )
