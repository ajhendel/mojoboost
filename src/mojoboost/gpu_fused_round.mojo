"""The beginning of a boosting round, fused where fusing is sound.

A GPU round starts with four things: this round's gradients and hessians,
the fixed-point scale their histograms quantize with, the root's active-row
permutation, and the root histogram itself. This module holds the
primitives that sequence those, and the cost model that says which of the
three candidate fusions is worth building.

The three designs, and what happened to each
--------------------------------------------
**Materialized gradient planes** (what the trainer does today, and what
this module keeps). One kernel writes `grad[n]` and `hess[n]`, a second
reduces their magnitudes, and every histogram in the tree reads the two
planes. RECOMMENDED, and not as a compromise: every node below the root
gathers `grad[r]` and `hess[r]` at permuted row indices, so the planes have
to exist for the whole tree whatever the root does. Nothing that removes
them can be correct.

**Tiled production consumed immediately by the root histogram.** Produce
the gradients a row tile at a time and hand each tile straight to the
histogram accumulation for those rows. REJECTED. It saves no bytes,
because the planes are still written for the nodes below the root, and it
saves no time on an in-order queue, where a tile's histogram launch cannot
start before the tile's gradient launch has retired anyway. It costs one
launch per tile. The streaming idea does pay, just not here: it pays on
the host-origin path, where the same two Float64 lists are read three
times, and `HostGradientStage` in gpu_gradient_stream.mojo collects that.

**Kernel fusion**, the objective evaluated inside the histogram kernel.
REJECTED, and not on bandwidth grounds. The histogram accumulates in
fixed point, and the fixed-point scale is a global reduction over the very
gradients being accumulated (`_fixed_scale`, `device_fixed_scale`). Every
gradient in the dataset therefore has to exist before the first bin of the
first histogram can be quantized. A kernel that produced gradients and
consumed them in the same launch would need its own scale before it had
computed the sum the scale comes from. That is a circular dependency, not
a tuning problem, and it is what the task's "do not fuse operations whose
synchronization semantics require materialized arrays" is pointing at.

The bandwidth arithmetic agrees, and `RoundTraffic` below reproduces it
for any shape. At a million rows and fifty features the gradient
production is 2.4% of the root's traffic on an unweighted objective, so a
perfect fusion could remove 2.4% while evaluating the objective 51 times
per row instead of once. On a weighted objective the fused kernel reads
three planes per feature where the materialized one reads two, and fusion
costs 27% more traffic than it saves. Neither number is worth a second
definition of the objectives.

What is left, and what this module implements
---------------------------------------------
Two things survive, and both are about scheduling and residency rather
than arithmetic.

`MagnitudeReader` splits the magnitude reduction into an enqueue and a
read. `GpuObjectiveState.magnitude_sums` does the kernel, the copy, and
the synchronization in one call, which forces the round's only
device-to-host wait to happen before anything else in the round can be
enqueued. Split, the caller can enqueue every launch that does not depend
on the scale (seeding the root permutation, in the trainer) ahead of the
wait, so that work executes while the host is blocked instead of after it.
The reduction itself is the same grid-stride over the same fixed block
count with the same shared-memory tree, so the partials, their host-side
Float64 total, and every scale and histogram derived from them are
bit-identical to what the unsplit call produces.

The round drivers below then sequence a round start in the one order that
is correct under sampling: gradients, then compensation, then magnitudes.
The scale has to describe the values the histograms will read, which under
GOSS are the compensated ones, and that is also the order the host path
runs in (`goss_round` scales, then `upload_gradients` derives the scale
from the scaled values).

`round_eligibility` records which configurations the device round can
serve. It is deliberately not a copy of `device_gradients` in
train_gpu.mojo, because that function refuses bagging and GOSS with one
shared reason ("row sampling draws its sample from host-side gradients")
that is only true of one of them. A GOSS selection is a ranking over
gradient values. A bag is not: `sample_rows` draws from a counter stream
keyed by (seed, bag index, row) and never reads a derivative. Bagging is
blocked by something else entirely, and by exactly one thing, which the
codes below name: bagging requires every row's raw score to advance after
each tree, in bag or not, and the device update walks leaf ranges that
cover only the in-bag rows. The handoff spells out the two ways to close
that.
"""

from std.gpu import block_idx, thread_idx
from std.math import isfinite
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .boosting import CUSTOM
from .gpu_gradient_stream import GpuRowSelection
from .gpu_objectives_native import (
    SUM_BLOCKS,
    SUM_THREADS,
    GpuObjectiveState,
    GradMagnitudes,
    device_fixed_scale,
    supports_device_objective,
)

# Element sizes the cost model counts in. The device carries gradients,
# hessians, raw scores, targets, and weights as Float32, row ids and
# histogram cells as Int32, and bins as UInt8.
comptime BYTES_F32 = 4
comptime BYTES_I32 = 4
comptime BYTES_U8 = 1

# Bytes of gradient plane per row: one Float32 gradient, one Float32
# hessian.
comptime GRAD_PLANE_BYTES = 2 * BYTES_F32


def _magnitude_partials_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """Per-threadgroup magnitude sums of the gradients and hessians, laid
    out `[grad partials | hess partials]`.

    This is `_abs_sum_kernel` from gpu_objectives_native.mojo, expression
    for expression and in the same grid-stride and tree-reduction order,
    so its partials are bit-identical to that kernel's. It exists here
    only so that the reduction can be enqueued without the copy and the
    synchronization that `magnitude_sums` performs in the same call.
    Integration should delete one of the two definitions rather than keep
    them in step by hand; the handoff says which and why, and the
    duplication is the same one gpu_active_rows.mojo carries for
    `_range_reduce_kernel`.
    """
    var tid = thread_idx.x
    var sg = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var acc_g = Float32(0.0)
    var acc_h = Float32(0.0)
    var nr = Int(n_rows)
    var r = Int(block_idx.x) * SUM_THREADS + tid
    var stride = SUM_BLOCKS * SUM_THREADS
    while r < nr:
        acc_g += abs(grad[unsafe_offset=r][0])
        acc_h += abs(hess[unsafe_offset=r][0])
        r += stride
    sg[unsafe_offset=tid] = acc_g
    sh[unsafe_offset=tid] = acc_h
    barrier()

    # Uniform trip count across the threadgroup, so every thread reaches
    # every barrier.
    var active = SUM_THREADS // 2
    while active > 0:
        if tid < active:
            sg[unsafe_offset=tid] = (
                sg[unsafe_offset=tid][0] + sg[unsafe_offset = tid + active][0]
            )
            sh[unsafe_offset=tid] = (
                sh[unsafe_offset=tid][0] + sh[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2

    if tid == 0:
        var slot = Int(block_idx.x)
        partials[unsafe_offset=slot] = sg[unsafe_offset=0][0]
        partials[unsafe_offset = SUM_BLOCKS + slot] = sh[unsafe_offset=0][0]


@fieldwise_init
struct RoundScales(Copyable, Movable):
    """A round's fixed-point scales and the magnitude sums they came
    from. The two scales are what `GpuHistogramBuilder.g_scale` and
    `h_scale` hold; the sums are kept so a caller can report or assert on
    them without a second reduction."""

    var g_scale: Float32
    var h_scale: Float32
    var grad_magnitude: Float64
    var hess_magnitude: Float64


struct MagnitudeReader(Movable):
    """The magnitude reduction, split into an enqueue and a read.

    Construct once per training session on the same context as the
    gradient buffers it will reduce. Per round: `enqueue` after the
    gradients are final (which under sampling means after the
    compensation has been applied), then any launch that does not depend
    on the scale, then `read`.

    The partial buffer is `2 * SUM_BLOCKS` Float32 whatever the row count,
    so the readback is 2 KB per round rather than a function of `n_rows`,
    and the final sum happens on the host in Float64, which is more
    accurate than a Float32 device-side finish would be.
    """

    var part_dev: DeviceBuffer[DType.float32]
    var host_part: HostBuffer[DType.float32]
    var n_rows: Int
    var pending: Bool

    def __init__(out self, ctx: DeviceContext, n_rows: Int) raises:
        if n_rows < 1:
            raise Error("magnitude reduction requires at least one row")
        self.n_rows = n_rows
        self.pending = False
        self.part_dev = ctx.enqueue_create_buffer[DType.float32](
            2 * SUM_BLOCKS
        )
        self.host_part = ctx.enqueue_create_host_buffer[DType.float32](
            2 * SUM_BLOCKS
        )

    def enqueue(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Enqueue the reduction and the readback. Does not synchronize.

        The copy is enqueued here rather than in `read` so that it sits
        immediately behind its kernel in the queue and the host's later
        wait covers both. Everything the caller enqueues between this and
        `read` executes while the host is blocked.
        """
        if self.pending:
            raise Error(
                "a previous magnitude reduction has not been read; the"
                " readback buffer cannot be reused until it has"
            )
        ctx.enqueue_function[_magnitude_partials_kernel](
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            self.part_dev.unsafe_ptr(),
            Int32(self.n_rows),
            grid_dim=SUM_BLOCKS,
            block_dim=SUM_THREADS,
        )
        ctx.enqueue_copy(
            dst_ptr=self.host_part.unsafe_ptr(), src_buf=self.part_dev
        )
        self.pending = True

    def read(mut self, ctx: DeviceContext) raises -> GradMagnitudes:
        """Wait for the enqueued reduction and sum its partials.

        Ascending block index, gradient plane then hessian plane, in
        Float64. That is the order `magnitude_sums` sums in, over partials
        this kernel produced in the same order, so the totals agree bit
        for bit.
        """
        if not self.pending:
            raise Error("call enqueue before read")
        ctx.synchronize()
        self.pending = False
        var src = self.host_part.unsafe_ptr()
        var g_total = 0.0
        var h_total = 0.0
        for i in range(SUM_BLOCKS):
            g_total += Float64(src.unsafe_load(i))
            h_total += Float64(src.unsafe_load(SUM_BLOCKS + i))
        if not isfinite(g_total) or not isfinite(h_total):
            raise Error("gradients and hessians must be finite")
        return GradMagnitudes(g_total, h_total)

    def read_scales(mut self, ctx: DeviceContext) raises -> RoundScales:
        """`read`, converted into the two fixed-point scales a histogram
        builder needs."""
        var sums = self.read(ctx)
        var g = device_fixed_scale(sums.grad)
        var h = device_fixed_scale(sums.hess)
        return RoundScales(g, h, sums.grad, sums.hess)


def enqueue_gradients(
    ctx: DeviceContext,
    mut state: GpuObjectiveState,
    objective: Int,
    alpha: Float64,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut hess_dev: DeviceBuffer[DType.float32],
) raises:
    """This round's uncompensated gradients for a single-output
    objective, straight into the histogram buffers. A thin pass-through to
    `GpuObjectiveState.fill_grad_hess`, kept so a caller sequences a round
    entirely through this module and so the compensated and uncompensated
    stages of a sampled round have names."""
    state.fill_grad_hess(ctx, objective, alpha, grad_dev, hess_dev)


def enqueue_softmax_gradients(
    ctx: DeviceContext,
    mut state: GpuObjectiveState,
    k: Int,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut hess_dev: DeviceBuffer[DType.float32],
) raises:
    """Class `k`'s one-vs-rest gradients. Call
    `state.refresh_softmax(ctx)` once per round first, before the first
    class: the probabilities are shared by every class of the round and
    are normalized across classes inside that one kernel, so refreshing
    per class would not just cost `n_classes` times the exponentials, it
    would read raw scores that trees grown earlier in the same round have
    already advanced and break the shared-sample semantics the host path
    has."""
    state.fill_softmax_grad_hess(ctx, k, grad_dev, hess_dev)


def enqueue_magnitudes(
    ctx: DeviceContext,
    mut selection: GpuRowSelection,
    mut mags: MagnitudeReader,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut hess_dev: DeviceBuffer[DType.float32],
) raises:
    """Apply the round's compensation, then enqueue the magnitude
    reduction over the values the histograms will actually read.

    The order is the point. A GOSS round scales its sampled rows' two
    derivatives up so that the histogram of a subset still estimates the
    histogram of the whole dataset, and the fixed-point scale has to bound
    the scaled values, not the unscaled ones, or the Int32 accumulation
    loses its overflow guarantee. Reducing before compensating would
    produce a scale that is too large by up to the multiplier.

    The reduction covers every row, including rows this round's sample
    left out. That is what the host path does too (`_fixed_scale` runs
    over the whole gradient list, not over the bag), and it is the
    conservative side: the bound only has to cover the rows a histogram
    reads, which are a subset.
    """
    selection.apply_compensation(ctx, grad_dev, hess_dev)
    mags.enqueue(ctx, grad_dev, hess_dev)


def round_start(
    ctx: DeviceContext,
    mut state: GpuObjectiveState,
    objective: Int,
    alpha: Float64,
    mut selection: GpuRowSelection,
    mut mags: MagnitudeReader,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut hess_dev: DeviceBuffer[DType.float32],
) raises -> RoundScales:
    """A whole single-output round start, for a caller with nothing to
    interleave: gradients, compensation, magnitudes, scales.

    A caller that does have scale-independent work to enqueue (the
    trainer has one, seeding the root's active-row permutation) should
    call `enqueue_gradients`, `enqueue_magnitudes`, its own launches, and
    then `mags.read_scales` instead. The result is identical; only what
    the device does while the host waits differs.

    Sampling that has to be drawn from this round's gradients cannot use
    this entry point at all, because the sampler runs on the host between
    the gradient kernel and the compensation. That round is
    `enqueue_gradients`, `selection.enqueue_importance`,
    `selection.download_importance`, `goss_select` on the host,
    `selection.from_goss`, `enqueue_magnitudes`, `read_scales`.
    """
    enqueue_gradients(ctx, state, objective, alpha, grad_dev, hess_dev)
    enqueue_magnitudes(ctx, selection, mags, grad_dev, hess_dev)
    return mags.read_scales(ctx)


def softmax_round_start(
    ctx: DeviceContext,
    mut state: GpuObjectiveState,
    k: Int,
    mut selection: GpuRowSelection,
    mut mags: MagnitudeReader,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut hess_dev: DeviceBuffer[DType.float32],
) raises -> RoundScales:
    """`round_start` for one class of a softmax round. The caller has
    already called `state.refresh_softmax(ctx)` for this round, and a
    sampled softmax round shares one selection across every class, which
    is what the host path does and what keeps the classes of a round
    comparable."""
    enqueue_softmax_gradients(ctx, state, k, grad_dev, hess_dev)
    enqueue_magnitudes(ctx, selection, mags, grad_dev, hess_dev)
    return mags.read_scales(ctx)


# Why a configuration can or cannot take the device round. These are
# reasons, not a ranking: a caller that asked for the device path
# explicitly should be told which one applies rather than being quietly
# downgraded, which is the contract `device_gradients` already has.
comptime ROUND_OK = 0
comptime ROUND_CUSTOM_OBJECTIVE = 1
comptime ROUND_NO_DEVICE_KERNEL = 2
comptime ROUND_BAGGING_OUT_OF_BAG = 3
comptime ROUND_GOSS_RANK_PRECISION = 4


def round_eligibility(
    objective: Int,
    n_classes: Int,
    bagging_on: Bool,
    goss_on: Bool,
    allow_device_ranking: Bool = False,
) raises -> Int:
    """Which reason, if any, keeps this configuration off the device
    round.

    `ROUND_OK` means every per-row stage of the round can run on the
    device: derivatives from `GpuObjectiveState`, compensation from
    `GpuRowSelection`, magnitudes from `MagnitudeReader`, raw scores from
    `update_raw_ranges`.

    `ROUND_BAGGING_OUT_OF_BAG` is the bagging blocker, and it is not the
    sample. A bag is drawn from a counter stream and needs no gradients,
    so the sample itself is free to be device side. What is not yet
    device side is the raw-score update for the rows the bag left out:
    bagging semantics update every row's score after every tree, in bag
    or not, so that an out-of-bag row carries a correct gradient into
    later rounds, and the device update walks leaf ranges that by
    construction cover only in-bag rows. Until those rows are scored on
    the device the bagged round is not equivalent to the host one, and a
    trainer that took this path anyway would train later rounds on stale
    derivatives rather than merely different ones.

    `ROUND_GOSS_RANK_PRECISION` is a warning turned into a gate. Every
    stage of a GOSS round can run on the device, but the selection is a
    ranking of `|grad * hess|`, and ranking the Float32 device scores can
    put a different row across the top-k threshold than ranking the host's
    Float64 ones. That changes which rows a tree grows on, not just the
    last bits of a value, so it is opt-in: `allow_device_ranking` is the
    caller saying it accepts a sample that need not match the CPU
    trainer's row for row. Left False, a GOSS run keeps the host path and
    the two backends keep sampling identically.
    """
    if n_classes < 1:
        raise Error("n_classes must be positive")
    if objective == CUSTOM:
        return ROUND_CUSTOM_OBJECTIVE
    # Softmax is served by the probability and per-class kernels rather
    # than by the single-output derivative chain, so it is supported
    # without appearing in `supports_device_objective`.
    if n_classes == 1 and not supports_device_objective(objective):
        return ROUND_NO_DEVICE_KERNEL
    if bagging_on:
        return ROUND_BAGGING_OUT_OF_BAG
    if goss_on and not allow_device_ranking:
        return ROUND_GOSS_RANK_PRECISION
    return ROUND_OK


def round_eligibility_reason(code: Int) -> String:
    """A sentence a caller can raise or log verbatim."""
    if code == ROUND_OK:
        return String("the device round can serve this configuration")
    if code == ROUND_CUSTOM_OBJECTIVE:
        return String(
            "custom objectives have no device image of the callback, so the"
            " gradients originate on the host by construction; stage them"
            " with HostGradientStage and keep tree growth on the device"
        )
    if code == ROUND_NO_DEVICE_KERNEL:
        return String(
            "this objective has no device derivative kernel; train it with"
            " host-computed gradients and grow the trees on the device"
        )
    if code == ROUND_BAGGING_OUT_OF_BAG:
        return String(
            "row bagging updates every row's raw score after every tree,"
            " and the device update covers only the rows inside a leaf"
            " range, so the out-of-bag rows would carry stale derivatives"
            " into later rounds"
        )
    if code == ROUND_GOSS_RANK_PRECISION:
        return String(
            "GOSS ranks rows by |grad * hess|, and the device scores are"
            " Float32, so the sample can differ from the CPU trainer's near"
            " the threshold; pass allow_device_ranking=True to accept that"
        )
    return String("unknown round eligibility code")


@fieldwise_init
struct RoundTraffic(Copyable, Movable):
    """What one tree's round start moves and costs, under one design.

    Bytes are counted as the traffic a kernel issues, reads plus writes,
    with no cache modeled. That is deliberate: the whole question fusion
    asks is whether re-reading the gradient planes once per feature costs
    anything, and a model that assumed those reads were served from cache
    would answer the question by assumption. What the model gives is the
    upper bound and the ratio between designs; which fraction of the
    gradient-plane reads actually reach memory is a counter reading, and
    the bench plan names it.

    `objective_rows` counts per-row derivative evaluations, which is the
    recomputation axis: it is `n_rows` when the planes are materialized
    and a multiple of `n_rows` when they are not.
    """

    var gradient_bytes: Int
    """Producing the two planes: reading the objective's inputs and
    writing the gradients and hessians."""
    var magnitude_bytes: Int
    """The pass the fixed-point scale is reduced from."""
    var seed_bytes: Int
    """Writing the root's active-row permutation."""
    var root_hist_bytes: Int
    """The root histogram: row ids, bins, derivatives, and the partial
    buffer it writes."""
    var objective_rows: Int
    var launches: Int

    def start_bytes(self) -> Int:
        """The part of the round a fusion could touch: everything before
        the histogram."""
        return self.gradient_bytes + self.magnitude_bytes + self.seed_bytes

    def total_bytes(self) -> Int:
        return self.start_bytes() + self.root_hist_bytes


def objective_input_bytes(weighted: Bool) -> Int:
    """Bytes per row a derivative kernel reads: the raw score, the target,
    and the sample weight when there is one. The softmax kernels read one
    probability and one label instead, which is the same two Float32, so
    this covers both."""
    return 3 * BYTES_F32 if weighted else 2 * BYTES_F32


def _partial_bytes(n_features: Int, n_bins: Int, n_tiles: Int) -> Int:
    """The tiled strategy's partial buffer, written once per (tile,
    feature, bin) across three Int32 planes. The atomic strategy writes
    the output once instead and then folds partials into it with global
    atomics, whose traffic depends on contention and is not modeled;
    passing `n_tiles = 1` gives that path's output-sized write."""
    return 3 * n_tiles * n_features * n_bins * BYTES_I32


def _root_scan_bytes(
    n_rows: Int, n_features: Int, per_row_read: Int
) -> Int:
    """Per active feature, the root histogram reads one Int32 row id, one
    UInt8 bin, and `per_row_read` bytes of whatever it accumulates, for
    every row of the node."""
    return n_features * n_rows * (BYTES_I32 + BYTES_U8 + per_row_read)


def materialized_round_traffic(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_tiles: Int = 1,
    weighted: Bool = False,
) raises -> RoundTraffic:
    """The shipped design: one gradient kernel, one magnitude kernel, one
    seed kernel, one root histogram reading the two planes.

    Launches counted as gradient, magnitude, seed, and two for the
    histogram (a zeroing or a reduction alongside the accumulation),
    which is what the trainer issues today.
    """
    if n_rows < 1 or n_features < 1 or n_bins < 1 or n_tiles < 1:
        raise Error("traffic model needs positive rows, features, bins, tiles")
    var inputs = objective_input_bytes(weighted)
    return RoundTraffic(
        n_rows * (inputs + GRAD_PLANE_BYTES),
        n_rows * GRAD_PLANE_BYTES,
        n_rows * BYTES_I32,
        _root_scan_bytes(n_rows, n_features, GRAD_PLANE_BYTES)
        + _partial_bytes(n_features, n_bins, n_tiles),
        n_rows,
        5,
    )


def streamed_round_traffic(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_tiles: Int = 1,
    weighted: Bool = False,
) raises -> RoundTraffic:
    """Tiled production consumed immediately by the root histogram.

    Every byte term is the materialized design's, because the planes are
    still written: the nodes below the root gather them at permuted row
    indices and cannot be served from a tile that has already retired.
    The difference is one gradient launch per row tile instead of one, so
    the model returns the same traffic and a larger launch count, which
    is the whole finding.
    """
    var base = materialized_round_traffic(
        n_rows, n_features, n_bins, n_tiles, weighted
    )
    return RoundTraffic(
        base.gradient_bytes,
        base.magnitude_bytes,
        base.seed_bytes,
        base.root_hist_bytes,
        base.objective_rows,
        base.launches + n_tiles - 1,
    )


def fused_round_traffic(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_tiles: Int = 1,
    weighted: Bool = False,
) raises -> RoundTraffic:
    """The objective evaluated inside the histogram kernel.

    Modeled at its most favorable, and it still needs a prior full pass
    over the rows, because the fixed-point scale is a reduction over the
    gradients the histogram is about to quantize and cannot be computed
    from values that do not exist yet. With no planes to reduce, that pass
    recomputes the derivatives, so it reads the objective's inputs and
    evaluates the objective once per row; the histogram then evaluates it
    again once per (row, feature).

    The model gives this design the benefit of never writing the planes at
    all, which no real tree can have, since every node below the root
    reads them. Read it as the ceiling on what fusion could ever save,
    not as an achievable number.
    """
    if n_rows < 1 or n_features < 1 or n_bins < 1 or n_tiles < 1:
        raise Error("traffic model needs positive rows, features, bins, tiles")
    var inputs = objective_input_bytes(weighted)
    return RoundTraffic(
        0,
        n_rows * inputs,
        n_rows * BYTES_I32,
        _root_scan_bytes(n_rows, n_features, inputs)
        + _partial_bytes(n_features, n_bins, n_tiles),
        n_rows * (n_features + 1),
        4,
    )


def fusion_delta_bytes(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_tiles: Int = 1,
    weighted: Bool = False,
) raises -> Int:
    """Fused total minus materialized total, at the fused design's
    ceiling. Negative means fusion would move fewer bytes, positive means
    it would move more.

    It is negative only when the objective reads no more per row than the
    gradient planes it replaces, which means unweighted. Add a sample
    weight and the fused histogram reads three Float32 per row per feature
    where the materialized one reads two, and the sign flips at every
    feature count above one.
    """
    var fused = fused_round_traffic(
        n_rows, n_features, n_bins, n_tiles, weighted
    )
    var kept = materialized_round_traffic(
        n_rows, n_features, n_bins, n_tiles, weighted
    )
    return fused.total_bytes() - kept.total_bytes()


def fusion_recompute_factor(n_features: Int) raises -> Int:
    """Per-row derivative evaluations under fusion for every one under
    materialization. One pass to reduce the magnitudes plus one per active
    feature inside the histogram."""
    if n_features < 1:
        raise Error("feature count must be positive")
    return n_features + 1
