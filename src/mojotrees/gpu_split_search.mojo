"""Device-side split search over GPU histograms.

`split.mojo` finds a node's best split on the host, from a `Histogram` that
`histogram_gpu.mojo` has just downloaded. That download is the expensive part
of a GPU tree node: `3 * n_features * n_bins` Int32 words plus one host
synchronization, per node, whatever the node's size. A 100-feature, 256-bin
dataset pays 300 KB and a full pipeline drain to choose one (feature, bin)
pair, so the transfer is three orders of magnitude larger than the answer it
is used to compute.

This module searches the histogram where it already lives. The kernels here
consume the same fixed-point `[grad | hess | count]` Int32 buffer
`GpuHistogramBuilder` produces and emit one compact **split record**: the
winning feature, threshold or category set, gain, runner-up gain, missing
direction, both children's statistics, and both children's leaf values. That
record is the only thing that crosses back to the host, so a node costs a
fixed 136 bytes instead of a histogram.

`train_gpu.mojo` drives this per node today, under
`SPLIT_SEARCH_DEVICE`. The module is also standalone and testable on its
own: it owns a histogram buffer that a caller can upload to directly, and
`enqueue` also accepts an external device buffer for the zero-copy path.

One node, or a whole frontier
-----------------------------
Every per-node table has a slot per record: the feature set, the allow
mask, the float parameters (which carry the node's monotone bounds), and
the histogram offset. `enqueue_frontier` stages a bounded set of leaves,
issues one copy per table, and runs one scan and one reduction over all of
them; `download_frontier` is the single wait that brings every decision
back. That is one host synchronization per tree level rather than the two
per node the incremental loop pays, and it is what the record layout was
designed for. The batch changes nothing about a decision: each node reads
only its own slots, the scan order inside a node is unchanged, and the
records are the ones the same nodes searched one at a time would produce.

Float32 near ties, and the host-scan fallback
---------------------------------------------
The scan is Float32 (point 1 below), so two candidates whose exact gains
differ by less than a few ulps can come back in either order, and that is a
different *tree*, not a different last bit. Every record therefore carries
`runner_gain`, the best gain of every candidate the node scored except the
winner, over every scanned feature, so the margin the decision was made by
is a number the host can see. `GpuSplitRecord.is_near_tie` tests the margin
against a relative tolerance (`SPLIT_TIE_RELATIVE`, deliberately several
ulps wide), and `host_rescan_recommended` is the policy: a run that needs
CPU/GPU agreement redoes exactly those nodes with the host scan, one node
at a time, and keeps the device decision everywhere else. Tracking the
runner-up costs one compare per candidate and cannot change which candidate
wins. `frontier_margin` reports the same quantity one level up, where it
decides which leaf splits next.

Semantics
---------
Every decision this module makes is the one `find_best_split` would make, in
the same order:

- Features are scanned in active-slot order, bins ascending, and within a bin
  the missing-left candidate is scored before the missing-right one. Both the
  per-feature scan and the cross-feature reduction accept a new best only on
  a strictly greater gain, so the first candidate in that order wins every
  tie, exactly as on the host. Composed, the two stages select the
  lexicographically smallest (slot, bin, direction) among the maximum-gain
  candidates, which is what the host's single loop selects.
- L1 soft-thresholding, the L2 denominator, `min_child_hess`,
  `min_data_in_leaf`, the reserved missing bin and its `default_left`
  direction, the "every ordinary bin left" top threshold, monotone candidate
  rejection and output clamping, and both categorical searches (one-vs-rest
  and the sorted many-vs-many walk) are reproduced candidate for candidate
  from `split.mojo`, `gain.mojo`, `monotone.mojo`, and `categorical.mojo`.

Two deliberate numeric differences from the host path, neither of which
changes the shape of the arithmetic:

1. **Float32.** Apple GPUs have no Float64, so gains, hessian tests, leaf
   values, and the categorical sort keys are computed in Float32, as the
   histogram kernels already are. Split decisions therefore agree with the
   host to Float32 precision, not bit-exactly; a candidate pair whose gains
   differ below Float32 resolution may resolve the other way.
2. **Exact accumulation.** A child's gradient and hessian sums accumulate in
   the histogram's fixed-point Int32 and are dequantized once, at the end,
   rather than being summed from already-dequantized values as the host does.
   Integer addition is associative and the fixed-point scale bounds every
   partial sum (see `histogram_gpu._fixed_scale`), so the device sums are
   exact and reproducible, and row counts are exact integers throughout.

Both together mean the device is bit-deterministic run to run, which is the
property the GPU backend already guarantees, but is not bit-identical to the
host scan. Equivalence tests against the CPU trainer must stay
tolerance-based.

Layout
------
The search runs as two kernels and never allocates per node:

- `_scan_slot_kernel`, one threadgroup per (node, active feature), writes
  that feature's best candidate for that node into a per-slot record.
- `_reduce_slots_kernel`, one thread per node, folds that node's per-slot
  records into one record in ascending slot order and fills in the child
  statistics, child leaf values, the parent's leaf value, and the node's
  runner-up gain. `_reduce_slots_block_kernel` is the same fold on a
  threadgroup, which is what runs unless
  `MOJOTREES_GPU_SPLIT_PRIMITIVES=0`.

Two launches cover a whole frontier, not two per node and not one per
feature: the grid is `(widest feature slot, node)`, which is the shape
LightGBM's CUDA best-split finder uses when it runs the frontier's tasks as
one grid of `num_tasks_` blocks. There is no per-feature or per-leaf launch
left in this module to merge away.

Collective primitives, and where they are not allowed
----------------------------------------------------
The reductions here are `gpu.primitives.block` collectives (`sum`, `max`,
`min`, `prefix_sum`) rather than hand-rolled shared-memory loops, which is
portable across NVIDIA, AMD, and Apple Metal and so keeps this module's
one-source rule. A collective may replace a loop here only where
reassociation cannot move a bit:

- Every quantity accumulated along a feature's bins is fixed-point Int32,
  and integer addition is associative, so `block.sum` over a feature's
  totals and `block.prefix_sum[exclusive=True]` over the per-thread chunk
  sums return the serial walk's words exactly.
- Gains are only ever *compared* across threads, never summed, and `max`
  and `min` are associative and commutative on the values these kernels
  produce, so a tree-shaped argmax returns the serial walk's winner exactly
  once the tie-break is carried alongside it (highest gain, then lowest
  candidate ordinal inside a feature and lowest feature slot across
  features).

What is deliberately left serial: the gain arithmetic itself, and the
per-thread walk along a chunk of bins. A gain is a difference of three
Float32 quotients, and no collective reassociates one. There is no
floating-point atomic and no floating-point sum crossing a thread boundary
anywhere in this module, which is what keeps the device path
bit-deterministic run to run.

The scan is sequential within a feature because the candidate order *is* the
tie-breaking rule, and because a threshold scan is a prefix sum. Features are
the parallel dimension. That is deliberately the cheap half of the win: the
scan is `O(n_features * n_bins)` against the histogram build's
`O(n_rows * n_features)`, so removing the download is what matters, and
parallelizing within a feature (one thread per bin over a shared-memory
prefix scan) is a later refinement that cannot change any result, since the
prefix sums are exact integers.

Toward a device-side queue
--------------------------
`max_records` lets one searcher hold several leaves' records at once, and
`enqueue_pick_best` reduces a set of them to the single best-gain leaf,
tie-broken by ascending record index. The staircase this module was built
for is: the host downloads one record per node (the incremental loop); the
host downloads the records for a whole split at once (`enqueue_frontier`
plus `download_frontier`, which is now here, and which the resident grower
in `train_gpu` uses to pay one wait per split rather than two); the
frontier itself lives in `rec_i_dev`/`rec_f_dev` and the host only reads
the finished tree.

Note what the middle step does *not* become under leaf-wise growth: one
download per tree level. Leaf-wise splitting picks the best-gain leaf in
the whole frontier, so only the two new children need searching after each
split, and there is no level of siblings to batch. A level-wise grower is
what would turn `enqueue_frontier` into one wait per level; for the grower
we ship, one wait per split is the floor while the host is still deciding.

`enqueue_pick_best` is the piece that would lift that floor, and it is
built and tested but unused, because the rest of the last step is not in
this module and is larger than it looks. The device-side row partition it
was waiting on now exists: `GpuHistogramBuilder.apply_split` partitions a
parent's row range entirely on the device and stays fully enqueued when the
caller passes the left count it already has from the parent histogram. What
remains is everything else the host still does per split, none of which is
about finding a split: the leaf-value commit, the monotone output bounds
threaded down each branch, the `min_data_in_leaf` and `max_depth` shape
rules, per-node feature subsampling, and writing the tree itself. Those are
also what the CPU/GPU equivalence tests pin, so moving them is a
correctness project and not a latency patch.

Nothing in the record layout has to change for any of it, which is why the
record carries child statistics and leaf values rather than making the host
recompute them from a histogram it no longer has.

On where the device path's remaining cost actually is: on the evidence so
far it is not the scan kernel's shape. Packing feature slots a SIMD group
at a time instead of one threadgroup each, and moving the categorical sort
scratch out of threadgroup memory to lift the occupancy that allocation
caps, were both measured on an M4 at 50000 x 100 and both came back inside
noise of the one-thread-per-threadgroup launch this module still defaults
to. Whatever the per-split overhead is, those two did not touch it.

`_scan_slot_wide_kernel` is a third attempt at the same target and it is
off by default for that reason and not for any doubt about the result: it
splits one feature's bins across a threadgroup and returns the serial
kernel's record bit for bit, which `tests/test_gpu_split_search.mojo`
asserts against the serial kernel on the same histograms. It is reached
only through `MOJOTREES_GPU_SPLIT_WIDE=1`, and the two measurements above
are what it has to be read against: a run that cannot separate it from the
serial scan is the expected outcome, not a surprise. The default flips when
an interleaved benchmark resolves it and not before.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier

from .categorical import (
    CatBitset,
    CategoricalParams,
    CategoricalSpec,
    cat_add,
    cat_empty,
)
from .monotone import (
    MONOTONE_DECREASING,
    MONOTONE_FREE,
    MONOTONE_INCREASING,
    OutputBounds,
)
from .split import SplitInfo

# The widest histogram the GPU backend accepts (`histogram_gpu.MAX_BINS`).
comptime MAX_SPLIT_BINS = 256

# --- Split record layout -------------------------------------------------
#
# A record is one slice of an Int32 buffer and one slice of a Float32 buffer,
# rather than one packed struct, so no value is ever bit-cast between an
# integer and a float on the device. Counts, bin ids, and the category set
# are exact integers; gains, sums, and leaf values are Float32.

comptime IREC_FEATURE = 0
comptime IREC_BIN = 1
comptime IREC_FLAGS = 2
comptime IREC_ORDINAL = 3
comptime IREC_LEFT_COUNT = 4
comptime IREC_RIGHT_COUNT = 5
comptime IREC_TOTAL_COUNT = 6
comptime IREC_CAT0 = 7

# A category set is 256 bits held 16 bits to an Int32 word, so no bit is ever
# written into an Int32's sign position and the host needs no unsigned
# reinterpretation to read it back.
comptime CAT_WORD_BITS = 16
comptime CAT_WORDS = MAX_SPLIT_BINS // CAT_WORD_BITS

comptime SPLIT_IWORDS = IREC_CAT0 + CAT_WORDS

comptime FREC_GAIN = 0
comptime FREC_LEFT_GRAD = 1
comptime FREC_LEFT_HESS = 2
comptime FREC_RIGHT_GRAD = 3
comptime FREC_RIGHT_HESS = 4
comptime FREC_TOTAL_GRAD = 5
comptime FREC_TOTAL_HESS = 6
comptime FREC_LEFT_VALUE = 7
comptime FREC_RIGHT_VALUE = 8
comptime FREC_PARENT_VALUE = 9
comptime FREC_RUNNER_GAIN = 10
"""The best gain among every candidate this node scored *except* the
winner, across every scanned feature. The margin `gain - runner_gain` is
what a caller measures a Float32 near-tie against; see
`GpuSplitRecord.is_near_tie` and the near-tie section of the module
docstring."""

comptime SPLIT_FWORDS = 11

comptime FLAG_FOUND = 1
comptime FLAG_DEFAULT_LEFT = 2
comptime FLAG_CATEGORICAL = 4

# --- Per-node integer parameter block ------------------------------------
#
# The two node-varying integers that cannot be kernel arguments once a whole
# frontier is searched by one launch: how many feature slots this node
# scans, and where its histogram starts in the buffer the launch was given.
# One row per record.

comptime NODE_SLOTS = 0
comptime NODE_HIST_BASE = 1
"""Offset, in Int32 words, of this node's `[grad | hess | count]`
histogram inside the buffer passed to the launch. Zero for a caller that
hands over one node's histogram (`GpuHistogramBuilder.out_dev`), and
`slot * 3 * n_features * n_bins` for a caller whose leaves share one
multi-slot buffer, which is exactly the layout `GpuLeafBatcher.out_dev`
holds and `slot_cells` measures."""

comptime NODE_WORDS = 2

# Relative margin below which a node's winning gain and its runner-up are
# not distinguishable in Float32 with any confidence.
#
# Float32 carries about 1.2e-7 of relative resolution. A gain is a
# difference of three quotients, each accumulated from dequantized sums, so
# a handful of roundings separate a computed gain from the exact one; 1e-6
# is roughly eight of those, which is the conservative side of the only
# number that matters here, since being too eager costs a host rescan of
# one node and being too lax silently accepts a split the host would not
# have chosen.
comptime SPLIT_TIE_RELATIVE = Float64(1e-6)

# --- Per-node float parameter block --------------------------------------
#
# The floating-point half of a node's search parameters travels as one small
# device buffer instead of as kernel arguments, which keeps both kernels at a
# launch arity comparable to the histogram kernels'.

comptime PF_G_INV = 0
comptime PF_H_INV = 1
comptime PF_LAMBDA_L2 = 2
comptime PF_LAMBDA_L1 = 3
comptime PF_MIN_CHILD_HESS = 4
comptime PF_BOUND_LO = 5
comptime PF_BOUND_HI = 6
comptime PF_CAT_SMOOTH = 7
comptime PF_CAT_L2 = 8

comptime PF_WORDS = 9


# --- Shared scalar arithmetic --------------------------------------------
#
# These are the Float32 counterparts of `gain.mojo` and `monotone.mojo`, and
# they are the only place the gain formula is written. Both the kernels and
# the host reference below call them, so the two searches can only disagree
# in loop structure, which is what the tests compare.


@always_inline
def gpu_soft_threshold_l1(s: Float32, lambda_l1: Float32) -> Float32:
    """`gain.soft_threshold_l1` in Float32."""
    if lambda_l1 <= 0.0:
        return s
    var mag = abs(s) - lambda_l1
    if mag <= 0.0:
        return Float32(0.0)
    return mag if s > 0.0 else -mag


@always_inline
def gpu_leaf_score(
    g: Float32, h: Float32, lambda_l1: Float32, lambda_l2: Float32
) -> Float32:
    """`gain.leaf_score` in Float32."""
    var t = gpu_soft_threshold_l1(g, lambda_l1)
    return t * t / (h + lambda_l2)


@always_inline
def gpu_leaf_value(
    g: Float32, h: Float32, lambda_l1: Float32, lambda_l2: Float32
) -> Float32:
    """`tree._leaf_value` for one leaf's totals, in Float32. The host clamps
    the result into the node's monotone interval; this is the raw Newton
    value, exactly as `_leaf_value` returns it."""
    return -gpu_soft_threshold_l1(g, lambda_l1) / (h + lambda_l2)


@always_inline
def gpu_clamp(value: Float32, lo: Float32, hi: Float32) -> Float32:
    """`OutputBounds.clamp` in Float32."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@always_inline
def gpu_violates(
    sign: Int32, left_output: Float32, right_output: Float32
) -> Bool:
    """`monotone.violates` in Float32."""
    if sign == Int32(MONOTONE_INCREASING):
        return left_output > right_output
    if sign == Int32(MONOTONE_DECREASING):
        return left_output < right_output
    return False


@always_inline
def gpu_output_score(
    grad_sum: Float32, hess_sum: Float32, lambda_l2: Float32, output: Float32
) -> Float32:
    """`monotone.output_score` in Float32."""
    return -(
        2.0 * grad_sum * output + (hess_sum + lambda_l2) * output * output
    )


@always_inline
def gpu_split_gain(
    left_g: Float32,
    left_h: Float32,
    right_g: Float32,
    right_h: Float32,
    lambda_l2: Float32,
    parent_score: Float32,
    sign: Int32,
    bound_lo: Float32,
    bound_hi: Float32,
    constrained: Bool,
) -> Float32:
    """`split._split_gain` in Float32. `left_g` and `right_g` are already
    soft-thresholded; a candidate running against `sign` scores 0.0, which
    no caller accepts."""
    if not constrained:
        return (
            left_g * left_g / (left_h + lambda_l2)
            + right_g * right_g / (right_h + lambda_l2)
            - parent_score
        )
    var left_out = gpu_clamp(
        -left_g / (left_h + lambda_l2), bound_lo, bound_hi
    )
    var right_out = gpu_clamp(
        -right_g / (right_h + lambda_l2), bound_lo, bound_hi
    )
    if gpu_violates(sign, left_out, right_out):
        return Float32(0.0)
    return (
        gpu_output_score(left_g, left_h, lambda_l2, left_out)
        + gpu_output_score(right_g, right_h, lambda_l2, right_out)
        - parent_score
    )


# --- Collective primitives, and the switch that holds both arms -----------
#
# Mojo's `gpu.primitives.block` collectives (`sum`, `max`, `min`,
# `prefix_sum`) are supported on NVIDIA, AMD, and Apple Metal alike, so using
# them keeps this module's one-portable-source rule intact: there is still no
# per-backend code path here, and there are still no floating-point atomics.
#
# What they are allowed to replace is decided by associativity and nothing
# else. Every quantity these kernels accumulate along a feature's bins is
# fixed-point Int32, and integer addition is associative, so a tree-shaped
# `prefix_sum` or `sum` over those returns the serial walk's value bit for
# bit. The Float32 quantities are only ever *compared*, never summed across
# threads: `max` and `min` are associative and commutative on the values
# these kernels produce (no NaN, no signed zero), so a tree-shaped `max` over
# gains also returns the serial walk's value bit for bit. No collective in
# this module reassociates a floating-point sum, and none may: a gain is a
# difference of three Float32 quotients, and reassociating that would move a
# last bit and therefore, at a near tie, move a decision.

comptime NO_CANDIDATE = Int32(2147483647)
"""The identity a thread holding no candidate contributes to a `block.min`
over candidate positions. `Int32.MAX`, spelled out because every real
position is a candidate ordinal or a feature slot, both of which are far
below it."""

comptime REDUCE_SLOT_THREADS = 64
"""Threads per threadgroup in the primitive cross-feature reduction. A warp
multiple on every supported backend (it is `gpu_tiling.WARP_GRANULARITY`),
which is what `block.max` and `block.min` want; the collectives allocate
their own threadgroup scratch, one word per warp, so this kernel reserves no
shared memory of its own and raises no device floor."""


def split_primitives_requested() -> Bool:
    """`MOJOTREES_GPU_SPLIT_PRIMITIVES=0`, the switch back to the
    hand-rolled reductions.

    On unless refused, which is the opposite posture from
    `MOJOTREES_GPU_SPLIT_WIDE` and for a reason: the wide scan changes which
    kernel shape does the scanning, while the collectives change only how a
    reduction is spelled. Both arms return the same record by construction
    (integer sums and float maxima are both associative), and
    `tests/test_gpu_split_scan.mojo` asserts field for field that they do,
    so what the switch preserves is a measurement handle and an escape
    hatch, not a doubt about the answer.

    Read once, at construction, and stored on the searcher, where
    `GpuSplitSearcher.set_primitives` can override it. A benchmark that wants
    to hold both arms in one process sets the field rather than re-execing
    with a different environment, which is the same shape
    `MOJOTREES_GPU_SPLIT_RESIDENT` and `MOJOTREES_GPU_SPLIT_WIDE` already
    have: one environment variable that decides the default, one explicit
    handle that overrides it.
    """
    return getenv("MOJOTREES_GPU_SPLIT_PRIMITIVES") != "0"


# --- Kernels --------------------------------------------------------------


def _scan_slot_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    cat_n: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    record_base: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
    cat_onehot_max: Int32,
    cat_max_threshold: Int32,
    cat_min_group: Int32,
):
    """One threadgroup per (node, active feature slot): scan that feature's
    candidates for that node and write its best one as a per-slot record.

    The scan runs on one thread because the candidate order is the
    tie-breaking rule and a threshold scan is a prefix sum; features are the
    parallel dimension, and nodes are the second one. A launch covers
    records `[record_base, record_base + grid_dim.y)`, each with its own
    feature set, allow mask, float parameters, and histogram offset, which
    is what lets a whole frontier be scanned by one launch instead of one
    launch and one host wait per node. Every accumulation is exact
    fixed-point Int32, so the sums do not depend on either choice and a
    later per-bin parallel scan cannot change a result."""
    # Allocated at entry rather than inside the categorical branch, so the
    # threadgroup's shared allocation is unconditional and static.
    var keys = stack_allocation[
        MAX_SPLIT_BINS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sorted_bins = stack_allocation[
        MAX_SPLIT_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var slot = Int(block_idx.x)
    var record = Int(record_base) + Int(block_idx.y)
    var nt = record * NODE_WORDS
    # A launch is as wide as the widest node in the batch, so a node with a
    # narrower feature set leaves the tail slots alone. The reduction reads
    # only this node's own slots, so what those hold does not matter.
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = Int(node_tab[unsafe_offset = nt + NODE_HIST_BASE][0]) + f * nb
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS

    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = io + i] = Int32(0)
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = fo + i] = Float32(0.0)
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var cat_smooth = fparams[unsafe_offset = pf + PF_CAT_SMOOTH][0]
    var cat_l2 = fparams[unsafe_offset = pf + PF_CAT_L2][0]

    # Totals over this feature's bins. Every accumulated feature has the same
    # totals bit for bit, because a row contributes the same quantized value
    # to exactly one bin of each, so slot 0's copy is the one the reduction
    # takes the parent's leaf value from.
    var tg = Int32(0)
    var th = Int32(0)
    var tc = Int32(0)
    for b in range(nb):
        tg += hist[unsafe_offset = base + b][0]
        th += hist[unsafe_offset = hs + base + b][0]
        tc += hist[unsafe_offset = 2 * hs + base + b][0]
    var total_g = tg.cast[DType.float32]() * g_inv
    var total_h = th.cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = total_g
    out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = total_h
    out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = tc

    # A feature the node's interaction constraints disallow is skipped before
    # any of its candidates is scored, exactly as in `find_best_split`.
    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)
    var parent_score = gpu_leaf_score(
        total_g, total_h, lambda_l1, lambda_l2
    )

    var best_gain = Float32(0.0)
    # The best gain of every candidate this feature scored except the
    # winner, kept so the reduction can report the node's margin. Updated at
    # every acceptance site and nowhere else, so it costs one compare per
    # candidate and cannot change which candidate wins.
    var runner_gain = Float32(0.0)
    var best_bin = -1
    var best_ordinal = -1
    var best_default_left = False
    var best_left_g = Int32(0)
    var best_left_h = Int32(0)
    var best_left_c = Int32(0)
    var found = False
    var is_categorical = False

    var n_cat = Int(cat_n[unsafe_offset=f][0])
    if n_cat >= 2:
        # --- Category partition search ---------------------------------
        #
        # Bin 0 (missing, unseen, dropped) is never a member of a candidate
        # set, so those rows always route right and this feature's missing
        # bin plays no part.
        if n_cat <= Int(cat_onehot_max):
            # One-vs-rest: the single category goes left.
            for t in range(1, n_cat + 1):
                var lg = hist[unsafe_offset = base + t][0]
                var lh = hist[unsafe_offset = hs + base + t][0]
                var lc = hist[unsafe_offset = 2 * hs + base + t][0]
                if lc < min_data_in_leaf:
                    continue
                var lhf = lh.cast[DType.float32]() * h_inv
                if lhf < min_child_hess:
                    continue
                var rc = tc - lc
                if rc < min_data_in_leaf:
                    continue
                var rhf = total_h - lhf
                if rhf < min_child_hess:
                    continue
                var lgf = lg.cast[DType.float32]() * g_inv
                var rgf = total_g - lgf
                var gain = (
                    gpu_leaf_score(lgf, lhf, lambda_l1, lambda_l2)
                    + gpu_leaf_score(rgf, rhf, lambda_l1, lambda_l2)
                    - parent_score
                )
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_left_g = lg
                    best_left_h = lh
                    best_left_c = lc
                    found = True
                    is_categorical = True
                    for w in range(CAT_WORDS):
                        out_i[unsafe_offset = io + IREC_CAT0 + w] = Int32(0)
                    var word = io + IREC_CAT0 + (t // CAT_WORD_BITS)
                    out_i[unsafe_offset=word] = Int32(
                        1 << (t % CAT_WORD_BITS)
                    )
                elif gain > runner_gain:
                    runner_gain = gain
        else:
            # Many-vs-many over prefixes of the gradient/hessian ordering,
            # walked from both ends.
            var used = 0
            for t in range(1, n_cat + 1):
                var lc = hist[unsafe_offset = 2 * hs + base + t][0]
                if lc.cast[DType.float32]() < cat_smooth:
                    continue
                var lg = hist[unsafe_offset = base + t][0]
                var lh = hist[unsafe_offset = hs + base + t][0]
                keys[unsafe_offset=used] = (
                    lg.cast[DType.float32]()
                    * g_inv
                    / (lh.cast[DType.float32]() * h_inv + cat_smooth)
                )
                sorted_bins[unsafe_offset=used] = Int32(t)
                used += 1

            if used >= 2:
                # Stable ascending insertion sort, which is the ordering
                # `metrics._argsort` gives the host: ties keep the lower
                # category.
                for i in range(1, used):
                    var kv = keys[unsafe_offset=i][0]
                    var bv = sorted_bins[unsafe_offset=i][0]
                    var j = i - 1
                    while j >= 0 and keys[unsafe_offset=j][0] > kv:
                        keys[unsafe_offset = j + 1] = keys[unsafe_offset=j][0]
                        sorted_bins[unsafe_offset = j + 1] = sorted_bins[
                            unsafe_offset=j
                        ][0]
                        j -= 1
                    keys[unsafe_offset = j + 1] = kv
                    sorted_bins[unsafe_offset = j + 1] = bv

                var l2c = lambda_l2 + cat_l2
                var max_num_cat = Int(cat_max_threshold)
                if (used + 1) // 2 < max_num_cat:
                    max_num_cat = (used + 1) // 2
                var steps = used if used < max_num_cat else max_num_cat

                for d in range(2):
                    var direction = 1 if d == 0 else -1
                    var start_pos = 0 if d == 0 else used - 1
                    var pos = start_pos
                    var group = Int32(0)
                    var lg = Int32(0)
                    var lh = Int32(0)
                    var lc = Int32(0)
                    for i in range(steps):
                        var t = Int(sorted_bins[unsafe_offset=pos][0])
                        pos += direction
                        lg += hist[unsafe_offset = base + t][0]
                        lh += hist[unsafe_offset = hs + base + t][0]
                        var cnt = hist[unsafe_offset = 2 * hs + base + t][0]
                        lc += cnt
                        group += cnt

                        var lhf = lh.cast[DType.float32]() * h_inv
                        if lc < min_data_in_leaf or lhf < min_child_hess:
                            continue
                        var rc = tc - lc
                        if rc < min_data_in_leaf or rc < cat_min_group:
                            break
                        var rhf = total_h - lhf
                        if rhf < min_child_hess:
                            break
                        if group < cat_min_group:
                            continue
                        group = Int32(0)

                        var lgf = lg.cast[DType.float32]() * g_inv
                        var rgf = total_g - lgf
                        var gain = (
                            gpu_leaf_score(lgf, lhf, lambda_l1, l2c)
                            + gpu_leaf_score(rgf, rhf, lambda_l1, l2c)
                            - parent_score
                        )
                        if gain > best_gain:
                            runner_gain = best_gain
                            best_gain = gain
                            best_left_g = lg
                            best_left_h = lh
                            best_left_c = lc
                            found = True
                            is_categorical = True
                            for w in range(CAT_WORDS):
                                out_i[
                                    unsafe_offset = io + IREC_CAT0 + w
                                ] = Int32(0)
                            var p = start_pos
                            for _ in range(i + 1):
                                var m = Int(sorted_bins[unsafe_offset=p][0])
                                var word = (
                                    io + IREC_CAT0 + (m // CAT_WORD_BITS)
                                )
                                out_i[unsafe_offset=word] = out_i[
                                    unsafe_offset=word
                                ][0] | Int32(1 << (m % CAT_WORD_BITS))
                                p += direction
                        elif gain > runner_gain:
                            runner_gain = gain
    else:
        # --- Ordinal threshold scan ------------------------------------
        var missing_bin = Int(missing[unsafe_offset=f][0])
        var n_scan = missing_bin if missing_bin >= 0 else nb
        var miss_g = Int32(0)
        var miss_h = Int32(0)
        var miss_c = Int32(0)
        if missing_bin >= 0:
            miss_g = hist[unsafe_offset = base + missing_bin][0]
            miss_h = hist[unsafe_offset = hs + base + missing_bin][0]
            miss_c = hist[unsafe_offset = 2 * hs + base + missing_bin][0]

        var left_g = Int32(0)
        var left_h = Int32(0)
        var left_c = Int32(0)
        for b in range(n_scan):
            # The top threshold puts every ordinary bin left, so it is only a
            # split at all when missing rows fill the right child.
            if b == n_scan - 1 and miss_c == Int32(0):
                break
            left_g += hist[unsafe_offset = base + b][0]
            left_h += hist[unsafe_offset = hs + base + b][0]
            left_c += hist[unsafe_offset = 2 * hs + base + b][0]

            # Missing to the left, scored first so an exact tie keeps
            # default_left, as in LightGBM and as on the host.
            if missing_bin >= 0:
                var dl_g = left_g + miss_g
                var dl_h = left_h + miss_h
                var dl_c = left_c + miss_c
                var dl_hf = dl_h.cast[DType.float32]() * h_inv
                var dr_hf = total_h - dl_hf
                if not (
                    dl_hf < min_child_hess
                    or dr_hf < min_child_hess
                    or dl_c < min_data_in_leaf
                    or tc - dl_c < min_data_in_leaf
                ):
                    var dl_gf = dl_g.cast[DType.float32]() * g_inv
                    var dr_gf = total_g - dl_gf
                    var gain = gpu_split_gain(
                        gpu_soft_threshold_l1(dl_gf, lambda_l1),
                        dl_hf,
                        gpu_soft_threshold_l1(dr_gf, lambda_l1),
                        dr_hf,
                        lambda_l2,
                        parent_score,
                        sign,
                        bound_lo,
                        bound_hi,
                        is_constrained,
                    )
                    if gain > best_gain:
                        runner_gain = best_gain
                        best_gain = gain
                        best_bin = b
                        best_ordinal = 2 * b
                        best_default_left = True
                        best_left_g = dl_g
                        best_left_h = dl_h
                        best_left_c = dl_c
                        found = True
                    elif gain > runner_gain:
                        runner_gain = gain

            # Missing to the right. For a feature with no missing bin this is
            # the only candidate and the scan is exactly the ordinal one.
            if missing_bin < 0 or miss_c > Int32(0):
                var lhf = left_h.cast[DType.float32]() * h_inv
                var rhf = total_h - lhf
                if lhf < min_child_hess or rhf < min_child_hess:
                    continue
                if (
                    left_c < min_data_in_leaf
                    or tc - left_c < min_data_in_leaf
                ):
                    continue
                var lgf = left_g.cast[DType.float32]() * g_inv
                var rgf = total_g - lgf
                var gain = gpu_split_gain(
                    gpu_soft_threshold_l1(lgf, lambda_l1),
                    lhf,
                    gpu_soft_threshold_l1(rgf, lambda_l1),
                    rhf,
                    lambda_l2,
                    parent_score,
                    sign,
                    bound_lo,
                    bound_hi,
                    is_constrained,
                )
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_bin = b
                    best_ordinal = 2 * b + 1
                    best_default_left = False
                    best_left_g = left_g
                    best_left_h = left_h
                    best_left_c = left_c
                    found = True
                elif gain > runner_gain:
                    runner_gain = gain

    if not found:
        return

    var flags = Int32(FLAG_FOUND)
    if best_default_left:
        flags += Int32(FLAG_DEFAULT_LEFT)
    if is_categorical:
        flags += Int32(FLAG_CATEGORICAL)
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(best_bin)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(best_ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = best_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = tc - best_left_c
    var lgf = best_left_g.cast[DType.float32]() * g_inv
    var lhf = best_left_h.cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_GAIN] = best_gain
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = runner_gain
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = lgf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = lhf
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = total_g - lgf
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = total_h - lhf


# Threads per threadgroup in the wide ordinal scan. A warp multiple on every
# supported backend (it is `gpu_tiling.WARP_GRANULARITY`), and the width the
# shared-memory budget below is stated at.
comptime WIDE_SCAN_THREADS = 64

# Threadgroup memory `_scan_slot_wide_kernel` reserves: twelve
# `WIDE_SCAN_THREADS`-long Int32 or Float32 arrays, which at 64 threads is
# 3072 bytes, the same reservation the histogram kernels already make. The
# wide scan therefore raises no device floor, and
# `gpu_portability.MIN_SHARED_MEMORY_PER_BLOCK` covers it unchanged.
comptime WIDE_SCAN_SHARED_BYTES = 12 * WIDE_SCAN_THREADS * 4


def wide_scan_requested() -> Bool:
    """`MOJOTREES_GPU_SPLIT_WIDE=1`, the switch for the wide scan.

    Off unless asked for, which is this package's rule for a path no
    benchmark has priced rather than a doubt about the result: the wide
    kernel returns the serial kernel's records bit for bit (see
    `_scan_slot_wide_kernel`) and `tests/parallel/test_gpu_split_search.mojo`
    asserts that, so what is unmeasured is only whether it is faster. The
    scan is a small share of a split's cost on the one device this
    repository has run on -- `bench-launch-cost` prices a split's fixed
    overhead at roughly 280us -- so the honest expectation is a small win,
    and the default flips when a run says so and not before.
    """
    return getenv("MOJOTREES_GPU_SPLIT_WIDE") == "1"


def wide_scan_for(has_categorical: Bool) -> Bool:
    """Whether a searcher over this dataset scans wide: requested, and no
    categorical feature to scan.

    The categorical bar is the kernel's, not a policy: `_scan_slot_wide_kernel`
    implements the ordinal threshold scan and nothing else. Refusing per
    dataset rather than per feature keeps one kernel per launch, and a
    dataset that declares a categorical feature is one the serial kernel
    already serves correctly.
    """
    return wide_scan_requested() and not has_categorical


def _scan_slot_wide_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    record_base: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
):
    """`_scan_slot_kernel`'s ordinal scan, spread over a threadgroup instead
    of run on one thread, and it writes the same per-slot record.

    Same grid, `WIDE_SCAN_THREADS` threads to a threadgroup rather than one:
    `_scan_slot_kernel` puts a whole feature on a single lane because a
    threshold scan is a prefix sum, which is right about the dependency and
    wrong about it being serial. A prefix sum splits: each thread takes one
    contiguous chunk of the bins, the chunk sums are combined, and every
    thread starts its own walk from the exact sum of the chunks before it.

    Why the result is the serial one, bit for bit:

    - The running left sums are fixed-point Int32 and integer addition is
      associative, so a thread's starting sums are the ones the serial walk
      would have reached at that bin, whatever order the chunks were summed
      in. Every candidate is then scored by the same expressions over the
      same Float32 inputs.
    - The serial scan takes a candidate on a strict `>`, so its winner is
      the highest gain and, among equal gains, the earliest in scan order.
      Candidate order is the ordinal (`2 * bin` for missing-left, `2 * bin +
      1` for missing-right), which ascends with the scan, so the reduction
      below picks by gain and breaks ties on the lower ordinal and lands on
      the same candidate.
    - `runner_gain` is the second largest accepted gain counted with
      multiplicity, which is order independent, so merging each thread's top
      two is the whole-feature top two.

    Categorical features are not scanned here. Their many-vs-many search
    sorts categories by a gradient ratio and walks prefixes of that order,
    which is a different algorithm with its own scratch, and it stays in
    `_scan_slot_kernel`. `GpuSplitSearcher` refuses this kernel outright for
    a dataset that declares any categorical feature rather than branching
    per feature inside the launch, so nothing here can meet one.
    """
    # Totals reduction and chunk sums use separate arrays, so the read of the
    # totals and the write of the chunk sums need no barrier between them.
    var t_g = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var t_h = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var t_c = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_g = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_h = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_c = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_gain = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_runner = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_ord = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_lg = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_lh = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var a_lc = stack_allocation[
        WIDE_SCAN_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var tid = Int(thread_idx.x)
    var slot = Int(block_idx.x)
    var record = Int(record_base) + Int(block_idx.y)
    var nt = record * NODE_WORDS
    # Every early return in this kernel is decided by the grid position, the
    # node table, or a per-feature table, never by `tid`, so a threadgroup
    # takes it whole and no barrier below is reached by a subset of it.
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = Int(node_tab[unsafe_offset = nt + NODE_HIST_BASE][0]) + f * nb
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS

    var pg = Int32(0)
    var ph = Int32(0)
    var pc = Int32(0)
    var bb = tid
    while bb < nb:
        pg += hist[unsafe_offset = base + bb][0]
        ph += hist[unsafe_offset = hs + base + bb][0]
        pc += hist[unsafe_offset = 2 * hs + base + bb][0]
        bb += WIDE_SCAN_THREADS
    t_g[unsafe_offset=tid] = pg
    t_h[unsafe_offset=tid] = ph
    t_c[unsafe_offset=tid] = pc
    barrier()
    var active = WIDE_SCAN_THREADS // 2
    while active > 0:
        if tid < active:
            t_g[unsafe_offset=tid] = (
                t_g[unsafe_offset=tid][0] + t_g[unsafe_offset = tid + active][0]
            )
            t_h[unsafe_offset=tid] = (
                t_h[unsafe_offset=tid][0] + t_h[unsafe_offset = tid + active][0]
            )
            t_c[unsafe_offset=tid] = (
                t_c[unsafe_offset=tid][0] + t_c[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2
    var tg = t_g[unsafe_offset=0][0]
    var th = t_h[unsafe_offset=0][0]
    var tc = t_c[unsafe_offset=0][0]

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var total_g = tg.cast[DType.float32]() * g_inv
    var total_h = th.cast[DType.float32]() * h_inv

    # The record belongs to one thread throughout: nothing else in the
    # threadgroup writes `out_i` or `out_f`, so the initial clear and the
    # final winner need no barrier between them and the scan.
    if tid == 0:
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = io + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = fo + i] = Float32(0.0)
        out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)
        out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = total_h
        out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = tc

    # A feature the node's interaction constraints disallow is skipped before
    # any of its candidates is scored, exactly as in `find_best_split`.
    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)
    var parent_score = gpu_leaf_score(total_g, total_h, lambda_l1, lambda_l2)

    var missing_bin = Int(missing[unsafe_offset=f][0])
    var n_scan = missing_bin if missing_bin >= 0 else nb
    if n_scan < 1:
        return
    var miss_g = Int32(0)
    var miss_h = Int32(0)
    var miss_c = Int32(0)
    if missing_bin >= 0:
        miss_g = hist[unsafe_offset = base + missing_bin][0]
        miss_h = hist[unsafe_offset = hs + base + missing_bin][0]
        miss_c = hist[unsafe_offset = 2 * hs + base + missing_bin][0]

    # One contiguous chunk of the scan per thread. `per` is the same for
    # every thread, so a chunk's start is its thread index times it and the
    # partition is a function of `n_scan` alone.
    var per = (n_scan + WIDE_SCAN_THREADS - 1) // WIDE_SCAN_THREADS
    var lo = tid * per
    if lo > n_scan:
        lo = n_scan
    var hi = lo + per
    if hi > n_scan:
        hi = n_scan

    var cg = Int32(0)
    var ch = Int32(0)
    var cc = Int32(0)
    for i in range(lo, hi):
        cg += hist[unsafe_offset = base + i][0]
        ch += hist[unsafe_offset = hs + base + i][0]
        cc += hist[unsafe_offset = 2 * hs + base + i][0]
    s_g[unsafe_offset=tid] = cg
    s_h[unsafe_offset=tid] = ch
    s_c[unsafe_offset=tid] = cc
    barrier()
    # Exclusive prefix over the chunk sums, summed low index first. Every
    # thread reads the same shared values in the same order, so this is one
    # integer sum with one answer.
    var left_g = Int32(0)
    var left_h = Int32(0)
    var left_c = Int32(0)
    for j in range(tid):
        left_g += s_g[unsafe_offset=j][0]
        left_h += s_h[unsafe_offset=j][0]
        left_c += s_c[unsafe_offset=j][0]

    var best_gain = Float32(0.0)
    var runner_gain = Float32(0.0)
    var best_ordinal = -1
    var best_left_g = Int32(0)
    var best_left_h = Int32(0)
    var best_left_c = Int32(0)

    for b in range(lo, hi):
        # The top threshold puts every ordinary bin left, so it is only a
        # split at all when missing rows fill the right child. Serial breaks
        # out of the loop here; the bin is the last one either way, so
        # skipping it is the same thing.
        if b == n_scan - 1 and miss_c == Int32(0):
            continue
        left_g += hist[unsafe_offset = base + b][0]
        left_h += hist[unsafe_offset = hs + base + b][0]
        left_c += hist[unsafe_offset = 2 * hs + base + b][0]

        # Missing to the left, scored first so an exact tie keeps
        # default_left, as in LightGBM and as on the host.
        if missing_bin >= 0:
            var dl_g = left_g + miss_g
            var dl_h = left_h + miss_h
            var dl_c = left_c + miss_c
            var dl_hf = dl_h.cast[DType.float32]() * h_inv
            var dr_hf = total_h - dl_hf
            if not (
                dl_hf < min_child_hess
                or dr_hf < min_child_hess
                or dl_c < min_data_in_leaf
                or tc - dl_c < min_data_in_leaf
            ):
                var dl_gf = dl_g.cast[DType.float32]() * g_inv
                var dr_gf = total_g - dl_gf
                var gain = gpu_split_gain(
                    gpu_soft_threshold_l1(dl_gf, lambda_l1),
                    dl_hf,
                    gpu_soft_threshold_l1(dr_gf, lambda_l1),
                    dr_hf,
                    lambda_l2,
                    parent_score,
                    sign,
                    bound_lo,
                    bound_hi,
                    is_constrained,
                )
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_ordinal = 2 * b
                    best_left_g = dl_g
                    best_left_h = dl_h
                    best_left_c = dl_c
                elif gain > runner_gain:
                    runner_gain = gain

        # Missing to the right. For a feature with no missing bin this is
        # the only candidate and the scan is exactly the ordinal one.
        if missing_bin < 0 or miss_c > Int32(0):
            var lhf = left_h.cast[DType.float32]() * h_inv
            var rhf = total_h - lhf
            if lhf < min_child_hess or rhf < min_child_hess:
                continue
            if left_c < min_data_in_leaf or tc - left_c < min_data_in_leaf:
                continue
            var lgf = left_g.cast[DType.float32]() * g_inv
            var rgf = total_g - lgf
            var gain = gpu_split_gain(
                gpu_soft_threshold_l1(lgf, lambda_l1),
                lhf,
                gpu_soft_threshold_l1(rgf, lambda_l1),
                rhf,
                lambda_l2,
                parent_score,
                sign,
                bound_lo,
                bound_hi,
                is_constrained,
            )
            if gain > best_gain:
                runner_gain = best_gain
                best_gain = gain
                best_ordinal = 2 * b + 1
                best_left_g = left_g
                best_left_h = left_h
                best_left_c = left_c
            elif gain > runner_gain:
                runner_gain = gain

    a_gain[unsafe_offset=tid] = best_gain
    a_runner[unsafe_offset=tid] = runner_gain
    a_ord[unsafe_offset=tid] = Int32(best_ordinal)
    a_lg[unsafe_offset=tid] = best_left_g
    a_lh[unsafe_offset=tid] = best_left_h
    a_lc[unsafe_offset=tid] = best_left_c
    barrier()
    if tid != 0:
        return

    # Winner: highest gain, ties to the lower ordinal, which is what the
    # serial scan's strict `>` over an ascending candidate order gives.
    var win = -1
    for j in range(WIDE_SCAN_THREADS):
        if a_gain[unsafe_offset=j][0] <= Float32(0.0):
            continue
        if win < 0:
            win = j
        elif a_gain[unsafe_offset=j][0] > a_gain[unsafe_offset=win][0]:
            win = j
        elif (
            a_gain[unsafe_offset=j][0] == a_gain[unsafe_offset=win][0]
            and a_ord[unsafe_offset=j][0] < a_ord[unsafe_offset=win][0]
        ):
            win = j
    if win < 0:
        return

    # Top two of the union of the per-thread top twos, which is the top two
    # of every accepted gain: the serial `runner_gain` counted with
    # multiplicity.
    var m1 = Float32(0.0)
    var m2 = Float32(0.0)
    for j in range(WIDE_SCAN_THREADS):
        var v = a_gain[unsafe_offset=j][0]
        if v > m1:
            m2 = m1
            m1 = v
        elif v > m2:
            m2 = v
        var u = a_runner[unsafe_offset=j][0]
        if u > m1:
            m2 = m1
            m1 = u
        elif u > m2:
            m2 = u

    var ordinal = Int(a_ord[unsafe_offset=win][0])
    var flags = Int32(FLAG_FOUND)
    if ordinal % 2 == 0:
        flags += Int32(FLAG_DEFAULT_LEFT)
    var won_left_c = a_lc[unsafe_offset=win][0]
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(ordinal // 2)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = won_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = tc - won_left_c
    var lgf = a_lg[unsafe_offset=win][0].cast[DType.float32]() * g_inv
    var lhf = a_lh[unsafe_offset=win][0].cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_GAIN] = m1
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = m2
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = lgf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = lhf
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = total_g - lgf
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = total_h - lhf


def _scan_slot_wide_primitive_kernel(
    hist: MutPointer[Int32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    allow: MutPointer[Int32, MutAnyOrigin],
    missing: MutPointer[Int32, MutAnyOrigin],
    mono: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    n_bins: Int32,
    hist_size: Int32,
    record_base: Int32,
    feat_stride: Int32,
    min_data_in_leaf: Int32,
    constrained: Int32,
):
    """`_scan_slot_wide_kernel` with its four hand-rolled reductions written
    as `gpu.primitives.block` collectives, and returning its record bit for
    bit.

    The four, and why each substitution is exact:

    - The feature's totals were a strided partial sum per thread followed by
      a shared-memory halving tree. They are now `block.sum` over the same
      per-thread partials. The accumulated quantity is fixed-point Int32 and
      integer addition is associative, so the tree the collective happens to
      use returns the same word the halving tree did.
    - The running left sums across chunk boundaries were an exclusive prefix
      each thread computed by walking every lower thread's chunk sum out of
      shared memory. They are now `block.prefix_sum[exclusive=True]` over
      the same chunk sums. Int32 again, so again exact.
    - The winner across threads was a serial walk on thread 0 over a shared
      array of per-thread bests, taking the highest gain and, among equal
      gains, the lowest candidate ordinal. It is now `block.max` over the
      gains followed by `block.min` over the ordinals of the threads holding
      the maximum. `max` and `min` reassociate exactly on these values, and
      the pair (gain, ordinal) reproduces the serial rule exactly, because
      candidate ordinals ascend with the scan and are unique across threads:
      one thread and only one holds the maximum gain at the minimum ordinal,
      and it is the thread the serial walk would have stopped on.
    - The node's runner-up was a serial top-two merge over the same shared
      array. `runner_gain` is the second largest, counted with multiplicity,
      of the union of every thread's best and every thread's own runner-up.
      Removing one occurrence of the maximum is the same as excluding the
      winning thread's best, so the second largest is
      `max(max over non-winning threads of their best, max over all threads
      of their runner-up)`, which is two more `block.max` calls.

    Not a substitution: nothing here sums a Float32 across threads, and
    nothing may. A gain is a difference of three Float32 quotients, and a
    tree-shaped float sum would move its last bit, which at a near tie is a
    different split and not a different rounding.

    Everything else -- the candidate order inside a chunk, the missing-left
    before missing-right ordering, the `min_data_in_leaf` and
    `min_sum_hessian_in_leaf` gates, the monotone rejection, the top
    threshold rule -- is copied unchanged from `_scan_slot_wide_kernel`,
    because those are the semantics and not the reduction."""
    # The winning thread's fixed-point left sums, published once. Allocated
    # at entry so the threadgroup's shared reservation is unconditional and
    # static, as in the kernels above; the collectives allocate their own
    # scratch on top of this, one word per warp per reduction.
    var won = stack_allocation[
        3,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var tid = Int(thread_idx.x)
    var slot = Int(block_idx.x)
    var record = Int(record_base) + Int(block_idx.y)
    var nt = record * NODE_WORDS
    # Every early return in this kernel is decided by the grid position, the
    # node table, or a per-feature table, never by `tid`, so a threadgroup
    # takes it whole and no collective below is reached by a subset of it.
    # That is the same rule the hand-rolled barriers needed, and the
    # collectives need it for the same reason.
    if slot >= Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0]):
        return
    var stride = Int(feat_stride)
    var table = record * stride
    var f = Int(feat_ids[unsafe_offset = table + slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = Int(node_tab[unsafe_offset = nt + NODE_HIST_BASE][0]) + f * nb
    var io = (table + slot) * SPLIT_IWORDS
    var fo = (table + slot) * SPLIT_FWORDS
    var pf = record * PF_WORDS

    var pg = Int32(0)
    var ph = Int32(0)
    var pc = Int32(0)
    var bb = tid
    while bb < nb:
        pg += hist[unsafe_offset = base + bb][0]
        ph += hist[unsafe_offset = hs + base + bb][0]
        pc += hist[unsafe_offset = 2 * hs + base + bb][0]
        bb += WIDE_SCAN_THREADS
    var tg = block.sum[block_size=WIDE_SCAN_THREADS](pg)
    var th = block.sum[block_size=WIDE_SCAN_THREADS](ph)
    var tc = block.sum[block_size=WIDE_SCAN_THREADS](pc)

    var g_inv = fparams[unsafe_offset = pf + PF_G_INV][0]
    var h_inv = fparams[unsafe_offset = pf + PF_H_INV][0]
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var min_child_hess = fparams[unsafe_offset = pf + PF_MIN_CHILD_HESS][0]
    var bound_lo = fparams[unsafe_offset = pf + PF_BOUND_LO][0]
    var bound_hi = fparams[unsafe_offset = pf + PF_BOUND_HI][0]
    var total_g = tg.cast[DType.float32]() * g_inv
    var total_h = th.cast[DType.float32]() * h_inv

    # The record belongs to one thread throughout: nothing else in the
    # threadgroup writes `out_i` or `out_f`, so the initial clear and the
    # final winner need no barrier between them and the scan.
    if tid == 0:
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = io + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = fo + i] = Float32(0.0)
        out_i[unsafe_offset = io + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = io + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(-1)
        out_f[unsafe_offset = fo + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = fo + FREC_TOTAL_HESS] = total_h
        out_i[unsafe_offset = io + IREC_TOTAL_COUNT] = tc

    # A feature the node's interaction constraints disallow is skipped before
    # any of its candidates is scored, exactly as in `find_best_split`.
    if allow[unsafe_offset = table + slot][0] == Int32(0):
        return

    var sign = Int32(MONOTONE_FREE)
    if constrained != Int32(0):
        sign = mono[unsafe_offset=f][0]
    var is_constrained = constrained != Int32(0)
    var parent_score = gpu_leaf_score(total_g, total_h, lambda_l1, lambda_l2)

    var missing_bin = Int(missing[unsafe_offset=f][0])
    var n_scan = missing_bin if missing_bin >= 0 else nb
    if n_scan < 1:
        return
    var miss_g = Int32(0)
    var miss_h = Int32(0)
    var miss_c = Int32(0)
    if missing_bin >= 0:
        miss_g = hist[unsafe_offset = base + missing_bin][0]
        miss_h = hist[unsafe_offset = hs + base + missing_bin][0]
        miss_c = hist[unsafe_offset = 2 * hs + base + missing_bin][0]

    # One contiguous chunk of the scan per thread. `per` is the same for
    # every thread, so a chunk's start is its thread index times it and the
    # partition is a function of `n_scan` alone.
    var per = (n_scan + WIDE_SCAN_THREADS - 1) // WIDE_SCAN_THREADS
    var lo = tid * per
    if lo > n_scan:
        lo = n_scan
    var hi = lo + per
    if hi > n_scan:
        hi = n_scan

    var cg = Int32(0)
    var ch = Int32(0)
    var cc = Int32(0)
    for i in range(lo, hi):
        cg += hist[unsafe_offset = base + i][0]
        ch += hist[unsafe_offset = hs + base + i][0]
        cc += hist[unsafe_offset = 2 * hs + base + i][0]
    # Exclusive prefix over the chunk sums. Fixed-point Int32, so the
    # collective's tree and the serial low-index-first walk it replaces
    # agree word for word.
    var left_g = block.prefix_sum[
        block_size=WIDE_SCAN_THREADS, exclusive=True
    ](cg)
    var left_h = block.prefix_sum[
        block_size=WIDE_SCAN_THREADS, exclusive=True
    ](ch)
    var left_c = block.prefix_sum[
        block_size=WIDE_SCAN_THREADS, exclusive=True
    ](cc)

    var best_gain = Float32(0.0)
    var runner_gain = Float32(0.0)
    var best_ordinal = -1
    var best_left_g = Int32(0)
    var best_left_h = Int32(0)
    var best_left_c = Int32(0)

    for b in range(lo, hi):
        # The top threshold puts every ordinary bin left, so it is only a
        # split at all when missing rows fill the right child. Serial breaks
        # out of the loop here; the bin is the last one either way, so
        # skipping it is the same thing.
        if b == n_scan - 1 and miss_c == Int32(0):
            continue
        left_g += hist[unsafe_offset = base + b][0]
        left_h += hist[unsafe_offset = hs + base + b][0]
        left_c += hist[unsafe_offset = 2 * hs + base + b][0]

        # Missing to the left, scored first so an exact tie keeps
        # default_left, as in LightGBM and as on the host.
        if missing_bin >= 0:
            var dl_g = left_g + miss_g
            var dl_h = left_h + miss_h
            var dl_c = left_c + miss_c
            var dl_hf = dl_h.cast[DType.float32]() * h_inv
            var dr_hf = total_h - dl_hf
            if not (
                dl_hf < min_child_hess
                or dr_hf < min_child_hess
                or dl_c < min_data_in_leaf
                or tc - dl_c < min_data_in_leaf
            ):
                var dl_gf = dl_g.cast[DType.float32]() * g_inv
                var dr_gf = total_g - dl_gf
                var gain = gpu_split_gain(
                    gpu_soft_threshold_l1(dl_gf, lambda_l1),
                    dl_hf,
                    gpu_soft_threshold_l1(dr_gf, lambda_l1),
                    dr_hf,
                    lambda_l2,
                    parent_score,
                    sign,
                    bound_lo,
                    bound_hi,
                    is_constrained,
                )
                if gain > best_gain:
                    runner_gain = best_gain
                    best_gain = gain
                    best_ordinal = 2 * b
                    best_left_g = dl_g
                    best_left_h = dl_h
                    best_left_c = dl_c
                elif gain > runner_gain:
                    runner_gain = gain

        # Missing to the right. For a feature with no missing bin this is
        # the only candidate and the scan is exactly the ordinal one.
        if missing_bin < 0 or miss_c > Int32(0):
            var lhf = left_h.cast[DType.float32]() * h_inv
            var rhf = total_h - lhf
            if lhf < min_child_hess or rhf < min_child_hess:
                continue
            if left_c < min_data_in_leaf or tc - left_c < min_data_in_leaf:
                continue
            var lgf = left_g.cast[DType.float32]() * g_inv
            var rgf = total_g - lgf
            var gain = gpu_split_gain(
                gpu_soft_threshold_l1(lgf, lambda_l1),
                lhf,
                gpu_soft_threshold_l1(rgf, lambda_l1),
                rhf,
                lambda_l2,
                parent_score,
                sign,
                bound_lo,
                bound_hi,
                is_constrained,
            )
            if gain > best_gain:
                runner_gain = best_gain
                best_gain = gain
                best_ordinal = 2 * b + 1
                best_left_g = left_g
                best_left_h = left_h
                best_left_c = left_c
            elif gain > runner_gain:
                runner_gain = gain

    # Winner: highest gain, ties to the lower ordinal. Every accepted gain
    # is strictly positive (a thread's `best_gain` starts at zero and only a
    # strictly greater candidate replaces it), so zero is a safe identity
    # for a thread that found nothing, and a maximum of zero means the whole
    # threadgroup found nothing, which is the serial kernel's `win < 0`.
    var top = block.max[block_size=WIDE_SCAN_THREADS](best_gain)
    if top <= Float32(0.0):
        return
    var my_ordinal = NO_CANDIDATE
    if best_gain == top:
        my_ordinal = Int32(best_ordinal)
    var win_ordinal = block.min[block_size=WIDE_SCAN_THREADS](my_ordinal)
    # Candidate ordinals are unique across threads, because the chunks are
    # disjoint ranges of bins, so exactly one thread satisfies both halves.
    var is_winner = best_gain == top and Int32(best_ordinal) == win_ordinal
    if is_winner:
        won[unsafe_offset=0] = best_left_g
        won[unsafe_offset=1] = best_left_h
        won[unsafe_offset=2] = best_left_c

    # `runner_gain` for the whole feature is the second largest, counted
    # with multiplicity, of every thread's best together with every thread's
    # own runner-up. One occurrence of the maximum is exactly the winning
    # thread's best, so excluding that thread from the first maximum and
    # folding in the maximum runner-up gives the serial merge's answer.
    var excluding_winner = best_gain
    if is_winner:
        excluding_winner = Float32(0.0)
    var other_best = block.max[block_size=WIDE_SCAN_THREADS](
        excluding_winner
    )
    var best_runner = block.max[block_size=WIDE_SCAN_THREADS](runner_gain)
    # The collectives fence threadgroup memory themselves, but the write to
    # `won` above is this kernel's own and is spelled out rather than
    # inferred from theirs.
    barrier()
    if tid != 0:
        return

    var m2 = other_best if other_best > best_runner else best_runner
    var ordinal = Int(win_ordinal)
    var flags = Int32(FLAG_FOUND)
    if ordinal % 2 == 0:
        flags += Int32(FLAG_DEFAULT_LEFT)
    var won_left_c = won[unsafe_offset=2][0]
    out_i[unsafe_offset = io + IREC_FEATURE] = Int32(f)
    out_i[unsafe_offset = io + IREC_BIN] = Int32(ordinal // 2)
    out_i[unsafe_offset = io + IREC_FLAGS] = flags
    out_i[unsafe_offset = io + IREC_ORDINAL] = Int32(ordinal)
    out_i[unsafe_offset = io + IREC_LEFT_COUNT] = won_left_c
    out_i[unsafe_offset = io + IREC_RIGHT_COUNT] = tc - won_left_c
    var lgf = won[unsafe_offset=0][0].cast[DType.float32]() * g_inv
    var lhf = won[unsafe_offset=1][0].cast[DType.float32]() * h_inv
    out_f[unsafe_offset = fo + FREC_GAIN] = top
    out_f[unsafe_offset = fo + FREC_RUNNER_GAIN] = m2
    out_f[unsafe_offset = fo + FREC_LEFT_GRAD] = lgf
    out_f[unsafe_offset = fo + FREC_LEFT_HESS] = lhf
    out_f[unsafe_offset = fo + FREC_RIGHT_GRAD] = total_g - lgf
    out_f[unsafe_offset = fo + FREC_RIGHT_HESS] = total_h - lhf


def _reduce_slots_kernel(
    slot_i: MutPointer[Int32, MutAnyOrigin],
    slot_f: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    record_base: Int32,
    feat_stride: Int32,
):
    """Fold one node's per-slot records into one, in ascending slot order,
    and fill in the child and parent leaf values.

    One thread per node, launched over the same records the scan covered.
    A single thread walking the slots in order, accepting a new best only on
    a strictly greater gain, is what makes the winner the first candidate of
    the first slot holding the maximum gain: the same rule, in the same
    order, as the host's one loop. No atomics are involved, so the result
    cannot depend on scheduling, and a batched frontier reduces exactly as a
    one-node launch does since no node reads another node's slots."""
    var record = Int(record_base) + Int(block_idx.x)
    var table = record * Int(feat_stride)
    var nt = record * NODE_WORDS
    var n_slots = Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0])
    var pf = record * PF_WORDS
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var oi = record * SPLIT_IWORDS
    var of = record * SPLIT_FWORDS

    var best = -1
    var best_gain = Float32(0.0)
    # The best gain among the slots that did not win, folded together with
    # the winning slot's own runner-up below. Ties do not move it: a slot
    # whose gain equals the current best is not the winner and is a genuine
    # runner-up, which is exactly the case a near-tie test has to see.
    var runner_gain = Float32(0.0)
    for s in range(n_slots):
        var si = (table + s) * SPLIT_IWORDS
        var sf = (table + s) * SPLIT_FWORDS
        var flags = slot_i[unsafe_offset = si + IREC_FLAGS][0]
        if (flags & Int32(FLAG_FOUND)) == Int32(0):
            continue
        var gain = slot_f[unsafe_offset = sf + FREC_GAIN][0]
        if best < 0 or gain > best_gain:
            if best >= 0 and best_gain > runner_gain:
                runner_gain = best_gain
            best = s
            best_gain = gain
        elif gain > runner_gain:
            runner_gain = gain

    # The parent's totals come from this node's slot 0, which is the feature
    # the host grower already uses for leaf values (`tree_features[0]`).
    # Every accumulated feature carries the same totals bit for bit.
    var zi = table * SPLIT_IWORDS
    var zf = table * SPLIT_FWORDS
    var total_g = slot_f[unsafe_offset = zf + FREC_TOTAL_GRAD][0]
    var total_h = slot_f[unsafe_offset = zf + FREC_TOTAL_HESS][0]
    var total_c = slot_i[unsafe_offset = zi + IREC_TOTAL_COUNT][0]

    if best < 0:
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = oi + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = of + i] = Float32(0.0)
        out_i[unsafe_offset = oi + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_ORDINAL] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
        out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h
        out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
            total_g, total_h, lambda_l1, lambda_l2
        )
        return

    var bi = (table + best) * SPLIT_IWORDS
    var bf = (table + best) * SPLIT_FWORDS
    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = oi + i] = slot_i[unsafe_offset = bi + i][0]
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = of + i] = slot_f[unsafe_offset = bf + i][0]

    # The winner's own totals are bit-identical to slot 0's, but the record
    # is defined to carry slot 0's, so a caller reading the parent's
    # statistics gets the same numbers whether or not a split was found.
    out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
    out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
    out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h

    # The node's runner-up is the better of the best losing feature and the
    # winning feature's own second candidate, so the margin a caller tests
    # covers both ways a decision can be close: two features that scored
    # nearly the same, and two bins of one feature that did.
    var own_runner = slot_f[unsafe_offset = bf + FREC_RUNNER_GAIN][0]
    if own_runner > runner_gain:
        runner_gain = own_runner
    out_f[unsafe_offset = of + FREC_RUNNER_GAIN] = runner_gain

    var left_g = slot_f[unsafe_offset = bf + FREC_LEFT_GRAD][0]
    var left_h = slot_f[unsafe_offset = bf + FREC_LEFT_HESS][0]
    var right_g = slot_f[unsafe_offset = bf + FREC_RIGHT_GRAD][0]
    var right_h = slot_f[unsafe_offset = bf + FREC_RIGHT_HESS][0]
    out_f[unsafe_offset = of + FREC_LEFT_VALUE] = gpu_leaf_value(
        left_g, left_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_RIGHT_VALUE] = gpu_leaf_value(
        right_g, right_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
        total_g, total_h, lambda_l1, lambda_l2
    )


def _reduce_slots_block_kernel(
    slot_i: MutPointer[Int32, MutAnyOrigin],
    slot_f: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    node_tab: MutPointer[Int32, MutAnyOrigin],
    fparams: MutPointer[Float32, MutAnyOrigin],
    record_base: Int32,
    feat_stride: Int32,
):
    """`_reduce_slots_kernel` on a threadgroup instead of on one thread, and
    it writes the same record.

    The serial fold is the one place the default path spends time
    proportional to the feature count on a single lane: one thread per node
    walking every one of that node's feature slots in order. A hundred
    features is a hundred dependent iterations while the rest of the device
    is idle. Here each thread takes a strided subset of the slots, and the
    cross-thread fold is three `gpu.primitives.block` collectives.

    Why the winner is the serial winner. The serial rule is "highest gain,
    and among equal gains the lowest slot", because it walks slots ascending
    and accepts only on a strictly greater gain. Split in two, that is a
    `block.max` over the gains and then a `block.min` over the slot indices
    of the threads holding the maximum. Both reassociate exactly: `max` and
    `min` are associative and commutative, and no gain is recomputed here,
    only compared. Within a thread the same strict `>` over its own
    ascending slots keeps the lowest of its own ties, so the pair really is
    the global lexicographic minimum among the maximum-gain slots.

    Why the runner-up is the serial runner-up. `_reduce_slots_kernel` ends
    with `runner_gain` equal to the best gain over every found slot that is
    not the winner: a slot displaced from `best` is folded in at the
    displacement, and a slot rejected against `best` is folded in on the
    spot, so every non-winning slot is compared exactly once, and a slot
    whose gain ties the winner's is a genuine runner-up rather than a
    winner. Excluding one slot is decomposable, because exactly one thread
    owns the winning slot: that thread contributes the best of its *other*
    slots, every other thread contributes its own best, and one `block.max`
    finishes it.

    No floating-point sum crosses a thread boundary here. The gains being
    reduced were computed by the scan kernel and are only compared; the leaf
    values below are computed once, by one thread, from the winning slot's
    already-reduced sums, exactly as the serial kernel computes them."""
    var tid = Int(thread_idx.x)
    var record = Int(record_base) + Int(block_idx.x)
    var table = record * Int(feat_stride)
    var nt = record * NODE_WORDS
    var n_slots = Int(node_tab[unsafe_offset = nt + NODE_SLOTS][0])
    var pf = record * PF_WORDS
    var lambda_l2 = fparams[unsafe_offset = pf + PF_LAMBDA_L2][0]
    var lambda_l1 = fparams[unsafe_offset = pf + PF_LAMBDA_L1][0]
    var oi = record * SPLIT_IWORDS
    var of = record * SPLIT_FWORDS

    # This thread's own best and second best over its strided share of the
    # slots, in ascending slot order, so a tie inside one thread keeps the
    # lower slot exactly as the serial walk does. A found slot's gain is
    # always strictly positive, so zero is a safe identity for a thread with
    # no slots at all, which is every thread past `n_slots`.
    var my_gain = Float32(0.0)
    var my_second = Float32(0.0)
    var my_slot = NO_CANDIDATE
    var s = tid
    while s < n_slots:
        var si = (table + s) * SPLIT_IWORDS
        var sf = (table + s) * SPLIT_FWORDS
        var flags = slot_i[unsafe_offset = si + IREC_FLAGS][0]
        if (flags & Int32(FLAG_FOUND)) != Int32(0):
            var gain = slot_f[unsafe_offset = sf + FREC_GAIN][0]
            if gain > my_gain:
                my_second = my_gain
                my_gain = gain
                my_slot = Int32(s)
            elif gain > my_second:
                my_second = gain
        s += REDUCE_SLOT_THREADS

    var top = block.max[block_size=REDUCE_SLOT_THREADS](my_gain)
    var mine = NO_CANDIDATE
    if my_gain == top:
        mine = my_slot
    var best = block.min[block_size=REDUCE_SLOT_THREADS](mine)
    var contribution = my_gain
    if my_slot == best:
        contribution = my_second
    var runner = block.max[block_size=REDUCE_SLOT_THREADS](contribution)

    if tid != 0:
        return

    # The parent's totals come from this node's slot 0, which is the feature
    # the host grower already uses for leaf values (`tree_features[0]`).
    # Every accumulated feature carries the same totals bit for bit.
    var zi = table * SPLIT_IWORDS
    var zf = table * SPLIT_FWORDS
    var total_g = slot_f[unsafe_offset = zf + FREC_TOTAL_GRAD][0]
    var total_h = slot_f[unsafe_offset = zf + FREC_TOTAL_HESS][0]
    var total_c = slot_i[unsafe_offset = zi + IREC_TOTAL_COUNT][0]

    if top <= Float32(0.0):
        for i in range(SPLIT_IWORDS):
            out_i[unsafe_offset = oi + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            out_f[unsafe_offset = of + i] = Float32(0.0)
        out_i[unsafe_offset = oi + IREC_FEATURE] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_BIN] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_ORDINAL] = Int32(-1)
        out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
        out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
        out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h
        out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
            total_g, total_h, lambda_l1, lambda_l2
        )
        return

    var bi = (table + Int(best)) * SPLIT_IWORDS
    var bf = (table + Int(best)) * SPLIT_FWORDS
    for i in range(SPLIT_IWORDS):
        out_i[unsafe_offset = oi + i] = slot_i[unsafe_offset = bi + i][0]
    for i in range(SPLIT_FWORDS):
        out_f[unsafe_offset = of + i] = slot_f[unsafe_offset = bf + i][0]

    # The winner's own totals are bit-identical to slot 0's, but the record
    # is defined to carry slot 0's, so a caller reading the parent's
    # statistics gets the same numbers whether or not a split was found.
    out_i[unsafe_offset = oi + IREC_TOTAL_COUNT] = total_c
    out_f[unsafe_offset = of + FREC_TOTAL_GRAD] = total_g
    out_f[unsafe_offset = of + FREC_TOTAL_HESS] = total_h

    # The node's runner-up is the better of the best losing feature and the
    # winning feature's own second candidate, so the margin a caller tests
    # covers both ways a decision can be close.
    var node_runner = runner
    var own_runner = slot_f[unsafe_offset = bf + FREC_RUNNER_GAIN][0]
    if own_runner > node_runner:
        node_runner = own_runner
    out_f[unsafe_offset = of + FREC_RUNNER_GAIN] = node_runner

    var left_g = slot_f[unsafe_offset = bf + FREC_LEFT_GRAD][0]
    var left_h = slot_f[unsafe_offset = bf + FREC_LEFT_HESS][0]
    var right_g = slot_f[unsafe_offset = bf + FREC_RIGHT_GRAD][0]
    var right_h = slot_f[unsafe_offset = bf + FREC_RIGHT_HESS][0]
    out_f[unsafe_offset = of + FREC_LEFT_VALUE] = gpu_leaf_value(
        left_g, left_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_RIGHT_VALUE] = gpu_leaf_value(
        right_g, right_h, lambda_l1, lambda_l2
    )
    out_f[unsafe_offset = of + FREC_PARENT_VALUE] = gpu_leaf_value(
        total_g, total_h, lambda_l1, lambda_l2
    )


def _pick_best_record_kernel(
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    n_records: Int32,
    out_index: Int32,
):
    """Reduce a set of finished records to the single best-gain one, ties
    going to the lower record index.

    This is the frontier selection the host currently does in `grow_tree_gpu`
    (`for i in range(len(frontier))`, strictly greater gain wins). Running it
    here is what lets a whole tree level, and eventually a whole tree, be
    grown without a host round trip per node."""
    var best = -1
    var best_gain = Float32(0.0)
    for r in range(Int(n_records)):
        var flags = rec_i[unsafe_offset = r * SPLIT_IWORDS + IREC_FLAGS][0]
        if (flags & Int32(FLAG_FOUND)) == Int32(0):
            continue
        var gain = rec_f[unsafe_offset = r * SPLIT_FWORDS + FREC_GAIN][0]
        if best < 0 or gain > best_gain:
            best = r
            best_gain = gain

    # The destination slot lives in the same buffer as the sources, so one
    # pointer pair carries both and no two kernel arguments alias.
    var oi = Int(out_index) * SPLIT_IWORDS
    var of = Int(out_index) * SPLIT_FWORDS
    if best < 0:
        for i in range(SPLIT_IWORDS):
            rec_i[unsafe_offset = oi + i] = Int32(0)
        for i in range(SPLIT_FWORDS):
            rec_f[unsafe_offset = of + i] = Float32(0.0)
        rec_i[unsafe_offset = oi + IREC_FEATURE] = Int32(-1)
        rec_i[unsafe_offset = oi + IREC_BIN] = Int32(-1)
        rec_i[unsafe_offset = oi + IREC_ORDINAL] = Int32(-1)
        return
    var bi = best * SPLIT_IWORDS
    var bf = best * SPLIT_FWORDS
    if bi == oi:
        return
    for i in range(SPLIT_IWORDS):
        rec_i[unsafe_offset = oi + i] = rec_i[unsafe_offset = bi + i][0]
    for i in range(SPLIT_FWORDS):
        rec_f[unsafe_offset = of + i] = rec_f[unsafe_offset = bf + i][0]


# --- Host-side record and parameters -------------------------------------


@fieldwise_init
struct ChildStats(Copyable, Movable):
    """One child's statistics as the device computed them: gradient and
    hessian sums dequantized once from exact fixed-point accumulation, and an
    exact row count."""

    var grad: Float64
    var hess: Float64
    var count: Int


struct GpuSplitRecord(Copyable, Movable, Writable):
    """The single value a node's search returns to the host.

    Everything the grower needs from a node: the split itself, both
    children's statistics and Newton leaf values, and the parent's leaf
    value. The host still clamps the leaf values into the node's monotone
    interval and collapses an inverted pair to its midpoint, exactly as
    `grow_tree_gpu` does today; those are host-side bookkeeping over the
    tree's own bounds, not properties of the histogram."""

    var feature: Int
    var bin: Int
    var gain: Float64
    var found: Bool
    var default_left: Bool
    var is_categorical: Bool
    var cat_bitset: CatBitset
    var left: ChildStats
    var right: ChildStats
    var total: ChildStats
    var left_value: Float64
    var right_value: Float64
    var parent_value: Float64
    var ordinal: Int
    """Position of the winning candidate within its feature's scan:
    `2 * bin` with the missing rows left, `2 * bin + 1` with them right, and
    -1 for a categorical partition or no split. Diagnostic only; the
    tie-breaking rule is the scan order itself."""
    var runner_gain: Float64
    """The best gain of every candidate this node scored except the winner,
    over every scanned feature. `gain - runner_gain` is the margin the
    decision was made by, and `is_near_tie` is the test a caller applies to
    it."""

    def __init__(out self):
        """The absence of a split, with zero statistics."""
        self.feature = -1
        self.bin = -1
        self.gain = 0.0
        self.found = False
        self.default_left = False
        self.is_categorical = False
        self.cat_bitset = cat_empty()
        self.left = ChildStats(0.0, 0.0, 0)
        self.right = ChildStats(0.0, 0.0, 0)
        self.total = ChildStats(0.0, 0.0, 0)
        self.left_value = 0.0
        self.right_value = 0.0
        self.parent_value = 0.0
        self.ordinal = -1
        self.runner_gain = 0.0

    def margin(self) -> Float64:
        """How far ahead of the runner-up the winning candidate scored.

        Zero when nothing was found and when the two best candidates scored
        exactly alike, in which case the scan order decided, which is
        deterministic but is a decision the host would have made on
        Float64 gains instead.
        """
        if not self.found:
            return 0.0
        var m = self.gain - self.runner_gain
        return m if m > 0.0 else 0.0

    def is_near_tie(
        self, relative: Float64 = SPLIT_TIE_RELATIVE
    ) -> Bool:
        """Whether this node's decision is inside Float32's resolution.

        The device scans in Float32, so two candidates whose exact gains
        differ by less than a few Float32 ulps can come back in either
        order; when they do, the split chosen here can differ from the one
        the host's Float64 scan would have chosen, and the difference is a
        different tree, not a different last bit of a value.

        This is the test, and `host_rescan_recommended` is the policy built
        on it: a node that answers True is a node to redo with the host
        scan when the caller needs CPU/GPU agreement. A node with no
        runner-up (one candidate, or one feature with one admissible bin)
        is never near a tie whatever the margin.
        """
        if not self.found or self.runner_gain <= 0.0:
            return False
        var scale = abs(self.gain)
        if scale < abs(self.runner_gain):
            scale = abs(self.runner_gain)
        return self.margin() <= relative * scale

    def to_split_info(self) -> SplitInfo:
        """The `SplitInfo` the existing growers consume, so the device path
        is a drop-in for `_search`'s return value."""
        if not self.found:
            return SplitInfo(-1, -1, 0.0, False)
        if self.is_categorical:
            return SplitInfo.categorical(
                self.feature, self.gain, self.cat_bitset
            )
        return SplitInfo(
            self.feature, self.bin, self.gain, True, self.default_left
        )

    def write_to(self, mut writer: Some[Writer]):
        if not self.found:
            writer.write("GpuSplitRecord(none)")
        elif self.is_categorical:
            writer.write(
                "GpuSplitRecord(feature=",
                self.feature,
                ", categorical, gain=",
                self.gain,
                ")",
            )
        else:
            writer.write(
                "GpuSplitRecord(feature=",
                self.feature,
                ", bin<=",
                self.bin,
                ", gain=",
                self.gain,
                ", default_left=",
                self.default_left,
                ")",
            )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def host_rescan_recommended(
    record: GpuSplitRecord,
    tie_relative: Float64 = SPLIT_TIE_RELATIVE,
    enabled: Bool = True,
) raises -> Bool:
    """The conservative fallback contract, in one place.

    True when this node's decision should be redone by the host scan
    because the device could not separate its two best candidates in
    Float32. The caller's part of the contract is what it does with a True:
    download that node's histogram (`GpuHistogramBuilder.download_raw` and
    `histogram_from_host`, which the host-search path already calls for
    every node) and take `_search`'s answer for that node instead of this
    record's. Nothing else about the node changes: the row counts, the
    child statistics, and the partition are recomputed by the host scan
    from the same histogram.

    The fallback is per node, not per tree. A tie at one node says nothing
    about the next one, and a whole-tree fallback would give up the
    transfer saving on every node to fix the handful that are close.

    `enabled` is the caller's switch. A run that does not need CPU/GPU
    agreement (the default posture of the device search, whose gains are
    documented as Float32) passes False and keeps every decision on the
    device; a parity run passes True. Left as a parameter rather than read
    from the environment here, because the trainer already owns the split
    strategy resolution and one place should decide it.

    Raises for a nonpositive tolerance, which would silently disable the
    check.
    """
    if tie_relative <= 0.0:
        raise Error("the near-tie tolerance must be positive")
    if not enabled:
        return False
    return record.is_near_tie(tie_relative)


def frontier_margin(records: List[GpuSplitRecord]) raises -> Float64:
    """How much better the best leaf of a frontier is than the next best.

    The node-level margin's counterpart at the level above: which leaf a
    leaf-wise grower splits next is also decided by comparing gains, so a
    frontier whose two best leaves are within Float32 of each other can
    grow a differently *shaped* tree than the host would, even when every
    node's own decision is unambiguous.

    Returns 0.0 when fewer than two leaves offer a split, which is the
    case where the order cannot matter. The trainer's tie-break (lowest
    frontier index wins) is unchanged and deterministic either way; this
    only reports how close the call was.
    """
    var best = 0.0
    var second = 0.0
    var seen = 0
    for i in range(len(records)):
        if not records[i].found:
            continue
        var g = records[i].gain
        seen += 1
        if seen == 1 or g > best:
            if seen > 1 and best > second:
                second = best
            best = g
        elif g > second:
            second = g
    if seen < 2:
        return 0.0
    var m = best - second
    return m if m > 0.0 else 0.0


# --- Which configurations the device search can serve ---------------------

comptime SEARCH_OK = 0
comptime SEARCH_EXTRA_PARAMS = 1
comptime SEARCH_FEATURE_BYLEVEL = 2
comptime SEARCH_TOO_MANY_BINS = 3


def device_search_eligibility(
    n_bins: Int,
    extra_active: Bool,
    feature_fraction_bylevel_active: Bool,
) raises -> Int:
    """Which reason, if any, keeps a configuration off the device split
    search.

    The kernels score from `GpuSplitParams` alone: the two lambdas, the two
    child floors, and the categorical parameters. There is nowhere in them
    to charge a gain floor, a per-feature multiplier, a CEGB cost, a
    monotone penalty, a drawn threshold, or a capped and smoothed child
    output, and nowhere to draw a per-level feature subset, so a
    configuration that asks for any of those is refused rather than served
    a tree that quietly ignores it. The host scan honors all of them and is
    the fallback for every code below.

    Scalars rather than a `TreeParams`, so this module stays free of the
    tree-parameter graph and the trainer can call it with
    `params.extra.is_active()` and
    `params.feature_fraction_bylevel != 1.0` without a new import in either
    direction. `train_gpu._check_device_search_supported` is the caller
    that should consume it; the handoff carries that patch.
    """
    if n_bins < 1:
        raise Error("split search requires at least one bin")
    if n_bins > MAX_SPLIT_BINS:
        return SEARCH_TOO_MANY_BINS
    if extra_active:
        return SEARCH_EXTRA_PARAMS
    if feature_fraction_bylevel_active:
        return SEARCH_FEATURE_BYLEVEL
    return SEARCH_OK


def device_search_reason(code: Int) -> String:
    """A sentence a caller can raise verbatim, worded as the trainer words
    its own refusals today."""
    if code == SEARCH_OK:
        return String("the device split search can serve this configuration")
    if code == SEARCH_EXTRA_PARAMS:
        return String(
            "the device split search does not implement min_gain_to_split,"
            " max_delta_step, path_smooth, extra_trees, monotone_penalty,"
            " feature_contri, or the CEGB costs; the kernel scores from"
            " GpuSplitParams alone. Use the host split scan, which is the"
            " default (MOJOTREES_GPU_SPLIT_STRATEGY=host, or"
            " split_search=SPLIT_SEARCH_HOST)"
        )
    if code == SEARCH_FEATURE_BYLEVEL:
        return String(
            "the device split search does not implement"
            " feature_fraction_bylevel; the per-node draw it stages is taken"
            " from the tree's feature set directly. Use the host split scan,"
            " which is the default"
        )
    if code == SEARCH_TOO_MANY_BINS:
        return String(
            "the device split search supports at most 256 bins, which is the"
            " widest histogram a threadgroup's shared scratch holds"
        )
    return String("unknown split search eligibility code")


@fieldwise_init
struct GpuSplitParams(Copyable, Movable):
    """The node-independent half of a search's parameters, named as in
    `TreeParams`."""

    var lambda_l2: Float64
    var lambda_l1: Float64
    var min_child_hess: Float64
    var min_data_in_leaf: Int
    var cat: CategoricalParams

    @staticmethod
    def default() -> GpuSplitParams:
        return GpuSplitParams(1.0, 0.0, 1e-3, 0, CategoricalParams.default())


def _f32_bound(value: Float64) -> Float32:
    """A monotone output bound in the device's Float32. `monotone.NO_BOUND`
    is the largest finite Float64, which is not representable, so both
    sentinels map to the largest finite Float32 and stay sentinels."""
    var limit = Float64(Float32.MAX_FINITE)
    if value >= limit:
        return Float32.MAX_FINITE
    if value <= -limit:
        return -Float32.MAX_FINITE
    return Float32(value)


def _bitset_from_words(words: List[Int32], offset: Int) -> CatBitset:
    """Reassemble a 256-bit category set from the record's 16-bit words."""
    var bitset = cat_empty()
    for b in range(1, MAX_SPLIT_BINS):
        var w = words[offset + IREC_CAT0 + b // CAT_WORD_BITS]
        if (w & Int32(1 << (b % CAT_WORD_BITS))) != Int32(0):
            cat_add(bitset, b)
    return bitset


# --- Host-side searcher ---------------------------------------------------


struct SplitNodeRequest(Copyable, Movable):
    """One leaf of a frontier, as a batched search takes it.

    Everything here is per node; everything that is per tree or per run
    (the two lambdas, the child floors, the categorical parameters, the
    monotone sign vector, the missing-bin table) stays on the searcher or
    on `GpuSplitParams`, so a batch stages only what actually varies.

    The record slot a node's answer lands in is its position in the list
    the batch was given, so a caller keeps its frontier and its records in
    the same order and reads them back by index.
    """

    var hist_slot: Int
    """Which histogram inside the buffer handed to the batch this node's
    scan reads, in units of `3 * n_features * n_bins` Int32 words. Zero for
    a single-node buffer; the pool slot for a caller holding a level's
    histograms at once."""
    var features: List[Int]
    """This node's active features, global ids in scan order. Empty leaves
    the record's current set alone, which is the tree-level set a caller
    broadcast with `set_features`; a per-node draw
    (`feature_fraction_bynode`) passes its own."""
    var allowed: List[Bool]
    """The interaction-constraint mask by global feature id, empty for
    "every feature allowed"."""
    var bounds: OutputBounds
    """The node's monotone output interval, which is the one float
    parameter that differs between the leaves of one tree."""

    def __init__(
        out self,
        hist_slot: Int = 0,
        var features: List[Int] = [],
        var allowed: List[Bool] = [],
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.hist_slot = hist_slot
        self.features = features^
        self.allowed = allowed^
        self.bounds = bounds^


def _launch_search(
    mut ctx: DeviceContext,
    mut hist: DeviceBuffer[DType.int32],
    mut node: DeviceBuffer[DType.int32],
    mut feat: DeviceBuffer[DType.int32],
    mut allow: DeviceBuffer[DType.int32],
    mut missing: DeviceBuffer[DType.int32],
    mut catn: DeviceBuffer[DType.int32],
    mut mono: DeviceBuffer[DType.int32],
    mut fparam: DeviceBuffer[DType.float32],
    mut slot_i: DeviceBuffer[DType.int32],
    mut slot_f: DeviceBuffer[DType.float32],
    mut rec_i: DeviceBuffer[DType.int32],
    mut rec_f: DeviceBuffer[DType.float32],
    n_bins: Int,
    hist_size: Int,
    feat_stride: Int,
    widest_slots: Int,
    record_base: Int,
    n_records: Int,
    min_data_in_leaf: Int,
    constrained: Bool,
    cat_onehot_max: Int,
    cat_max_threshold: Int,
    cat_min_group: Int,
    wide: Bool = False,
    primitives: Bool = True,
) raises:
    """Enqueue the two kernels of one search, over `n_records` consecutive
    nodes starting at `record_base`.

    One node and a whole frontier take the same two launches; the batch is
    the grid's second dimension. Everything that varies per node (the
    feature set, the allow mask, the float parameters, the histogram
    offset) is read from a per-record slot, and everything that does not
    (the bin count, the tree's monotone vector, the categorical and
    minimum-rows parameters) stays a kernel argument.

    A free function over the context and the buffers rather than a method,
    so the histogram buffer is an ordinary argument whether it belongs to
    this searcher, to the histogram builder next to it, or to a multi-slot
    batcher holding a whole level's histograms at once.

    `wide` runs the same scan on `WIDE_SCAN_THREADS` threads per feature
    instead of one, writing the same per-slot records, so the reduction
    below is the same kernel over the same slots either way. Only
    `GpuSplitSearcher` decides it: the wide kernel scans ordinal features
    only, and the searcher is what knows whether the dataset has a
    categorical one.

    `primitives` swaps the hand-rolled shared-memory reductions for
    `gpu.primitives.block` collectives in both the wide scan and the
    cross-feature reduction. It changes no record: every reduction it
    replaces is over fixed-point Int32 sums or over Float32 maxima, both of
    which reassociate exactly. It does change the reduction's launch shape,
    which is why it is a parameter here and not a constant: the collective
    reduction runs `REDUCE_SLOT_THREADS` threads per node where the serial
    one runs a single thread, and the two launches must agree with the
    kernels they carry. The launch *count* is the same either way, two, and
    it is already the LightGBM shape: one grid over every (leaf, feature)
    task, not one launch per feature or per leaf. See
    `split_primitives_requested` for the switch and
    `GpuSplitSearcher.set_primitives` for the in-process override."""
    # The whole dispatch sits behind a compile-time accelerator test, so a
    # CPU-only extension build never instantiates any of these kernels and
    # never asks the backend for a GPU architecture it was not built with.
    # An accelerator machine cannot reproduce that failure, which is why it
    # is a guard here rather than a test somewhere.
    comptime if not has_accelerator():
        raise Error(
            "the device split search needs an accelerator; this binary was"
            " built without one"
        )
    else:
        if wide and primitives:
            ctx.enqueue_function[_scan_slot_wide_primitive_kernel](
                hist.unsafe_ptr(),
                node.unsafe_ptr(),
                feat.unsafe_ptr(),
                allow.unsafe_ptr(),
                missing.unsafe_ptr(),
                mono.unsafe_ptr(),
                fparam.unsafe_ptr(),
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                Int32(n_bins),
                Int32(hist_size),
                Int32(record_base),
                Int32(feat_stride),
                Int32(min_data_in_leaf),
                Int32(1) if constrained else Int32(0),
                grid_dim=(widest_slots, n_records),
                block_dim=WIDE_SCAN_THREADS,
            )
        elif wide:
            ctx.enqueue_function[_scan_slot_wide_kernel](
                hist.unsafe_ptr(),
                node.unsafe_ptr(),
                feat.unsafe_ptr(),
                allow.unsafe_ptr(),
                missing.unsafe_ptr(),
                mono.unsafe_ptr(),
                fparam.unsafe_ptr(),
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                Int32(n_bins),
                Int32(hist_size),
                Int32(record_base),
                Int32(feat_stride),
                Int32(min_data_in_leaf),
                Int32(1) if constrained else Int32(0),
                grid_dim=(widest_slots, n_records),
                block_dim=WIDE_SCAN_THREADS,
            )
        else:
            ctx.enqueue_function[_scan_slot_kernel](
                hist.unsafe_ptr(),
                node.unsafe_ptr(),
                feat.unsafe_ptr(),
                allow.unsafe_ptr(),
                missing.unsafe_ptr(),
                catn.unsafe_ptr(),
                mono.unsafe_ptr(),
                fparam.unsafe_ptr(),
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                Int32(n_bins),
                Int32(hist_size),
                Int32(record_base),
                Int32(feat_stride),
                Int32(min_data_in_leaf),
                Int32(1) if constrained else Int32(0),
                Int32(cat_onehot_max),
                Int32(cat_max_threshold),
                Int32(cat_min_group),
                grid_dim=(widest_slots, n_records),
                block_dim=1,
            )
        if primitives:
            ctx.enqueue_function[_reduce_slots_block_kernel](
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                node.unsafe_ptr(),
                fparam.unsafe_ptr(),
                Int32(record_base),
                Int32(feat_stride),
                grid_dim=n_records,
                block_dim=REDUCE_SLOT_THREADS,
            )
        else:
            ctx.enqueue_function[_reduce_slots_kernel](
                slot_i.unsafe_ptr(),
                slot_f.unsafe_ptr(),
                rec_i.unsafe_ptr(),
                rec_f.unsafe_ptr(),
                node.unsafe_ptr(),
                fparam.unsafe_ptr(),
                Int32(record_base),
                Int32(feat_stride),
                grid_dim=n_records,
                block_dim=1,
            )

struct GpuSplitSearcher(Movable):
    """Device-resident split search for one dataset shape.

    Construct once per training session next to the histogram builder, call
    `set_features` once per tree, `set_allowed`/`set_monotone` per node, and
    `search` (or `enqueue` + `download`) per node. Nothing is allocated per
    node: every buffer is sized at construction from `n_features`, `n_bins`,
    and `max_records`.

    One node at a time, or a whole frontier
    ---------------------------------------
    Every per-node table has one slot per record: the feature set, the allow
    mask, the float parameters (which carry the node's monotone output
    bounds), and the histogram offset. That is what lets `enqueue_frontier`
    stage a bounded set of leaves, launch them all, and wait once, instead
    of waiting once per node as the node-at-a-time loop does. The two paths
    run the same two kernels over the same slots and return the same
    records; the batch only changes how many nodes one launch covers and
    how often the host blocks.

    The staging contract, which is what a caller has to respect:

    - `enqueue_frontier` writes every record's slot first and issues one
      copy per table, so nothing is overwritten while a copy of it is in
      flight. `download_frontier` is the batch's single wait, and no new
      staging may begin before it (or an explicit `synchronize`).
    - `enqueue` and `search` stage one record and copy, which is the
      node-at-a-time loop's existing contract: a node's `enqueue` is
      followed by its `download` before the next node stages.
    """

    var ctx: DeviceContext
    var n_features: Int
    var n_bins: Int
    var max_records: Int
    # The histogram this searcher owns, in `GpuHistogramBuilder`'s layout, for
    # callers that stage a histogram through the host. The zero-copy path
    # passes the builder's own buffer to `enqueue` instead.
    #
    # Allocated on first use rather than at construction, because neither
    # device search path touches it: `enqueue` and `enqueue_frontier` are
    # handed the builder's buffer, and only `upload_histogram` and `search`,
    # which exist so the search is exercisable on its own, read this one.
    # At the default 255 bins and a 50-feature fit that is 153 KB of device
    # memory per searcher that a fit allocated and never addressed. Until
    # `_ensure_hist` runs it holds a single placeholder element, since a
    # zero-length device buffer is not portable.
    var hist_dev: DeviceBuffer[DType.int32]
    var hist_owned: Bool
    """Whether `hist_dev` has been sized for a `3 * n_features * n_bins`
    histogram yet. False on a searcher that has only ever been driven by the
    trainer."""
    # Per-record tables. `feat_dev` and `allow_dev` are strided by
    # `n_features` rather than by a batch's slot count, so narrowing one
    # node's feature set never moves another node's row, which is the same
    # choice `GpuLeafBatcher` makes for its item tables.
    var node_dev: DeviceBuffer[DType.int32]
    var feat_dev: DeviceBuffer[DType.int32]
    var allow_dev: DeviceBuffer[DType.int32]
    var missing_dev: DeviceBuffer[DType.int32]
    var catn_dev: DeviceBuffer[DType.int32]
    var mono_dev: DeviceBuffer[DType.int32]
    var fparam_dev: DeviceBuffer[DType.float32]
    var slot_i_dev: DeviceBuffer[DType.int32]
    var slot_f_dev: DeviceBuffer[DType.float32]
    var rec_i_dev: DeviceBuffer[DType.int32]
    var rec_f_dev: DeviceBuffer[DType.float32]
    # Pinned staging for the per-node tables, so a node's parameters upload
    # as an ordinary one-way copy rather than through `map_to_host`, which
    # moves the buffer both ways on every use. One slot per record: see the
    # staging contract above.
    #
    # One-way, not asynchronous. On Metal `enqueue_copy` is a synchronous
    # full-queue drain in both directions (measured by disassembly,
    # `docs/GPU_PORTABILITY.md` section 6.1), so an upload here is a host
    # synchronization and the copies below have to be counted as waits, not
    # as enqueues.
    var stage_node: HostBuffer[DType.int32]
    var stage_feat: HostBuffer[DType.int32]
    var stage_allow: HostBuffer[DType.int32]
    var stage_param: HostBuffer[DType.float32]
    var host_i: HostBuffer[DType.int32]
    var host_f: HostBuffer[DType.float32]
    var active: List[Int]
    """The most recently broadcast feature set, and what `n_active`
    reports. A record whose own set was narrowed by `set_features(...,
    record=r)` carries its slot count in `active_len` instead."""
    var active_len: List[Int]
    """Feature slots per record, one entry per record slot."""
    var missing_bin: List[Int]
    var cat_n: List[Int]
    var constrained: Bool
    var wide_scan: Bool
    """Whether the per-feature scan runs on a threadgroup rather than on one
    thread. Decided once at construction by `wide_scan_for`, because both
    inputs are fixed there: the environment switch, and whether this
    dataset declares a categorical feature, which the wide kernel does not
    scan. Reported by `describe_scan`."""
    var use_primitives: Bool
    """Whether the reductions run as `gpu.primitives.block` collectives
    rather than as the hand-rolled shared-memory loops. Read once at
    construction from `MOJOTREES_GPU_SPLIT_PRIMITIVES` and settable
    afterwards, so one process can hold both arms; see `set_primitives`.
    Unlike `wide_scan` this has no dataset precondition, because both
    reductions it replaces are exact under reassociation on every dataset:
    the collectives choose the same split as the loops, whatever the
    histogram."""

    def __init__(
        out self,
        n_features: Int,
        n_bins: Int,
        missing_bins: List[Int] = [],
        cats: CategoricalSpec = CategoricalSpec.none(),
        max_records: Int = 1,
    ) raises:
        """Size every buffer and upload the per-feature tables that do not
        change during training: the missing-bin table and the category
        counts.

        Opens a private `DeviceContext`. A searcher reading another owner's
        device buffer (the trainer integration reads the histogram
        builder's) must share that owner's context instead, so the two
        enqueue into one in-order queue; see the context overload below."""
        var ctx = DeviceContext()
        self = Self(ctx, n_features, n_bins, missing_bins, cats, max_records)

    def __init__(
        out self,
        ctx: DeviceContext,
        n_features: Int,
        n_bins: Int,
        missing_bins: List[Int] = [],
        cats: CategoricalSpec = CategoricalSpec.none(),
        max_records: Int = 1,
    ) raises:
        """Build on a caller-supplied context; the private-context form
        above lands here. Sharing the histogram builder's context is what
        makes `enqueue` over the builder's own buffer safe without a fence:
        one queue orders the histogram kernels before the scan."""
        if n_features < 1:
            raise Error("split search requires at least one feature")
        if n_bins < 1:
            raise Error("split search requires at least one bin")
        if n_bins > MAX_SPLIT_BINS:
            raise Error("split search supports at most 256 bins")
        if len(missing_bins) > 0 and len(missing_bins) != n_features:
            raise Error("missing_bins length must equal n_features")
        if max_records < 1:
            raise Error("max_records must be at least one")
        for f in range(n_features):
            if len(missing_bins) > 0 and missing_bins[f] >= n_bins:
                raise Error("missing bin index out of range")
            if cats.is_cat(f) and cats.n_categories(f) >= n_bins:
                raise Error(
                    "categorical feature has more categories than bins"
                )

        self.ctx = ctx
        self.n_features = n_features
        self.n_bins = n_bins
        self.max_records = max_records
        self.constrained = False
        self.missing_bin = List[Int](capacity=n_features)
        self.cat_n = List[Int](capacity=n_features)
        self.active = List[Int](capacity=n_features)
        self.active_len = List[Int](capacity=max_records)
        var any_cat = False
        for f in range(n_features):
            self.missing_bin.append(
                missing_bins[f] if len(missing_bins) > 0 else -1
            )
            self.cat_n.append(cats.n_categories(f) if cats.is_cat(f) else 0)
            if self.cat_n[f] >= 2:
                any_cat = True
            self.active.append(f)
        # Fixed here and not revisited: `cats` is a construction-time fact,
        # and narrowing a record's feature set later can only remove
        # features, never introduce a categorical one.
        self.wide_scan = wide_scan_for(any_cat)
        self.use_primitives = split_primitives_requested()
        for _ in range(max_records):
            self.active_len.append(n_features)

        var table_cells = max_records * n_features
        # A placeholder until `_ensure_hist` sizes it, because a
        # zero-length device buffer is not portable. See the field.
        self.hist_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.hist_owned = False
        self.node_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_records * NODE_WORDS
        )
        self.feat_dev = self.ctx.enqueue_create_buffer[DType.int32](
            table_cells
        )
        self.allow_dev = self.ctx.enqueue_create_buffer[DType.int32](
            table_cells
        )
        self.missing_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_features
        )
        self.catn_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.mono_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.fparam_dev = self.ctx.enqueue_create_buffer[DType.float32](
            max_records * PF_WORDS
        )
        self.slot_i_dev = self.ctx.enqueue_create_buffer[DType.int32](
            table_cells * SPLIT_IWORDS
        )
        self.slot_f_dev = self.ctx.enqueue_create_buffer[DType.float32](
            table_cells * SPLIT_FWORDS
        )
        self.rec_i_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_records * SPLIT_IWORDS
        )
        self.rec_f_dev = self.ctx.enqueue_create_buffer[DType.float32](
            max_records * SPLIT_FWORDS
        )
        self.stage_node = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_records * NODE_WORDS
        )
        self.stage_feat = self.ctx.enqueue_create_host_buffer[DType.int32](
            table_cells
        )
        self.stage_allow = self.ctx.enqueue_create_host_buffer[DType.int32](
            table_cells
        )
        self.stage_param = self.ctx.enqueue_create_host_buffer[DType.float32](
            max_records * PF_WORDS
        )
        self.host_i = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_records * SPLIT_IWORDS
        )
        self.host_f = self.ctx.enqueue_create_host_buffer[DType.float32](
            max_records * SPLIT_FWORDS
        )

        # Every record starts as "every feature, all allowed, one histogram
        # at offset zero", so a caller that never narrows a set and never
        # batches (the node-at-a-time loop, and every existing caller) sees
        # exactly the behavior it saw when these tables held one slot.
        var dst_node = self.stage_node.unsafe_ptr()
        var dst_feat = self.stage_feat.unsafe_ptr()
        var dst_allow = self.stage_allow.unsafe_ptr()
        var dst_param = self.stage_param.unsafe_ptr()
        for r in range(max_records):
            var nt = r * NODE_WORDS
            dst_node.unsafe_store(nt + NODE_SLOTS, Int32(n_features))
            dst_node.unsafe_store(nt + NODE_HIST_BASE, Int32(0))
            for f in range(n_features):
                dst_feat.unsafe_store(r * n_features + f, Int32(f))
                dst_allow.unsafe_store(r * n_features + f, Int32(1))
            for w in range(PF_WORDS):
                dst_param.unsafe_store(r * PF_WORDS + w, Float32(0.0))
        self.ctx.enqueue_copy(dst_buf=self.node_dev, src_ptr=dst_node)
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst_feat)
        self.ctx.enqueue_copy(dst_buf=self.allow_dev, src_ptr=dst_allow)

        with self.missing_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                dst.unsafe_store(f, Int32(self.missing_bin[f]))
        with self.catn_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                var n_cat = cats.n_categories(f) if cats.is_cat(f) else 0
                dst.unsafe_store(f, Int32(n_cat))
        with self.mono_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                dst.unsafe_store(f, Int32(MONOTONE_FREE))
        # The three table copies above read pinned buffers those tables
        # reuse every node, so the session starts with them retired rather
        # than in flight. On Metal each copy already drained the queue
        # (`docs/GPU_PORTABILITY.md` section 6.1) and this call is therefore
        # redundant there; it is kept because it is what makes the sequence
        # correct on a backend where the copy is asynchronous, and because
        # it costs nothing once the queue is empty.
        self.ctx.synchronize()

    def synchronize(self) raises:
        """Block until every enqueued device operation has completed."""
        self.ctx.synchronize()

    def n_active(self) -> Int:
        """How many feature slots the next search scans."""
        return len(self.active)

    def set_primitives(mut self, enabled: Bool):
        """Force the collective reductions on or off for this searcher,
        overriding `MOJOTREES_GPU_SPLIT_PRIMITIVES`.

        The handle an interleaved benchmark needs: the two arms have to be
        alternated inside one process and one thermal state to be comparable
        at all on this machine, which re-execing with a different
        environment cannot do. Safe to change between searches, because it
        selects a kernel at launch time and owns no state; the records
        either arm returns are the same records."""
        self.use_primitives = enabled

    def describe_scan(self) -> String:
        """One line for benchmark output and bug reports: which scan kernel
        this searcher launches, how wide its threadgroup is, and whether its
        reductions are collectives or hand-rolled loops."""
        var reduction = String(
            " reduce=block-primitives threads=",
            REDUCE_SLOT_THREADS,
        ) if self.use_primitives else String(" reduce=serial threads=1")
        if self.wide_scan:
            return String("scan=wide threads=", WIDE_SCAN_THREADS, reduction)
        return String("scan=serial threads=1", reduction)

    def _check_record(self, record: Int) raises:
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")

    def _stage_features(
        mut self, features: List[Int], record: Int
    ) raises:
        """Write one record's feature slots and slot count into the pinned
        tables. Copies nothing: the caller decides when the tables go
        across, which is what lets a frontier stage every node first."""
        var dst = self.stage_feat.unsafe_ptr()
        var base = record * self.n_features
        for i in range(len(features)):
            dst.unsafe_store(base + i, Int32(features[i]))
        self.active_len[record] = len(features)
        var node = self.stage_node.unsafe_ptr()
        node.unsafe_store(
            record * NODE_WORDS + NODE_SLOTS, Int32(len(features))
        )

    def _stage_allowed(
        mut self, allowed: List[Bool], record: Int
    ) raises:
        """Write one record's allow mask, translated from global feature
        ids into this record's slot order."""
        var dst = self.stage_allow.unsafe_ptr()
        var base = record * self.n_features
        var n_slots = self.active_len[record]
        var feat = self.stage_feat.unsafe_ptr()
        for i in range(n_slots):
            var f = Int(feat.unsafe_load(base + i))
            var ok = True
            if len(allowed) > 0:
                ok = f < len(allowed) and allowed[f]
            dst.unsafe_store(base + i, Int32(1) if ok else Int32(0))
        for i in range(n_slots, self.n_features):
            dst.unsafe_store(base + i, Int32(0))

    def _copy_tables(mut self) raises:
        """One copy per per-node table, covering every record slot."""
        self.ctx.enqueue_copy(
            dst_buf=self.node_dev, src_ptr=self.stage_node.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.feat_dev, src_ptr=self.stage_feat.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.allow_dev, src_ptr=self.stage_allow.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.fparam_dev, src_ptr=self.stage_param.unsafe_ptr()
        )

    def set_features(
        mut self, features: List[Int], record: Int = -1
    ) raises:
        """Restrict later searches to `features` (global feature ids,
        ascending, one entry each), the same subsampled set
        `GpuHistogramBuilder.set_features` accumulated. Slot order is scan
        order, so it also fixes the cross-feature tie-breaking.

        `record` of -1, the default, applies the set to every record slot,
        which is what a tree-level or node-at-a-time caller means and what
        this method has always done. A frontier that draws a different
        subset per node (`feature_fraction_bynode`) passes the record it is
        staging instead, and only that node's slots move.

        Staged through pinned memory rather than a mapping, so it costs no
        host synchronization; the copy goes out with the node's `enqueue`.
        """
        if len(features) == 0:
            raise Error("active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
        if record >= 0:
            self._check_record(record)
            self._stage_features(features, record)
            # The allow mask is indexed by slot, so a new slot order
            # invalidates it; reset this record's to "every listed feature
            # allowed" and let its `set_allowed` narrow it again.
            self._stage_allowed(List[Bool](), record)
            return
        self.active = features.copy()
        for r in range(self.max_records):
            self._stage_features(features, r)
            self._stage_allowed(List[Bool](), r)

    def set_allowed(
        mut self, allowed: List[Bool] = [], record: Int = -1
    ) raises:
        """This node's interaction-constraint allow mask, indexed by global
        feature id. Empty (the default) allows every feature; a mask shorter
        than `n_features` disallows the features past its end, exactly as
        `find_best_split` reads it.

        Indexed by slot on the device, so it is re-staged after every
        `set_features`. `record` of -1 applies the mask to every record
        slot, as the node-at-a-time loop wants; a frontier passes the record
        it is staging.

        The copy is issued by `enqueue` or `enqueue_frontier` rather than
        here, so a batch stages every node's mask before any of them
        crosses.
        """
        if record >= 0:
            self._check_record(record)
            self._stage_allowed(allowed, record)
            return
        for r in range(self.max_records):
            self._stage_allowed(allowed, r)

    def set_monotone(mut self, signs: List[Int] = []) raises:
        """This tree's active monotone constraint vector. Empty (the default)
        keeps the unconstrained scoring path, exactly as on the host."""
        if len(signs) > 0 and len(signs) != self.n_features:
            raise Error("monotone length must equal n_features")
        for f in range(self.n_features):
            if (
                len(signs) > 0
                and self.cat_n[f] > 0
                and signs[f] != MONOTONE_FREE
            ):
                raise Error(
                    "monotonic constraints are not supported on categorical"
                    " features"
                )
        self.constrained = len(signs) > 0
        with self.mono_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(self.n_features):
                var s = signs[f] if len(signs) > 0 else MONOTONE_FREE
                dst.unsafe_store(f, Int32(s))

    def _ensure_hist(mut self) raises:
        """Size the searcher's own histogram buffer, once, on first use.

        Both callers are the standalone path (`upload_histogram` and
        `search`); a searcher the trainer drives never reaches either and
        therefore never pays for the buffer. Sizing is a construction-time
        fact (`n_features` and `n_bins` never change), so this can only run
        once, and it runs before the copy or the launch that needs it."""
        if self.hist_owned:
            return
        self.hist_dev = self.ctx.enqueue_create_buffer[DType.int32](
            3 * self.n_features * self.n_bins
        )
        self.hist_owned = True

    def upload_histogram(mut self, words: List[Int32]) raises:
        """Stage a fixed-point `[grad | hess | count]` histogram into this
        searcher's own buffer. The trainer integration uses the zero-copy
        `enqueue` overload against the builder's buffer instead; this exists
        so the search is exercisable, and benchmarkable, on its own."""
        if len(words) != 3 * self.n_features * self.n_bins:
            raise Error(
                "histogram must hold 3 * n_features * n_bins Int32 words"
            )
        self._ensure_hist()
        self.ctx.enqueue_copy(
            dst_buf=self.hist_dev, src_ptr=words.unsafe_ptr()
        )
        self.ctx.synchronize()

    def _stage_params(
        mut self,
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds,
        record: Int,
    ) raises:
        """Write one record's float parameter block, including its monotone
        output bounds, which are the only per-node value in it. Copies
        nothing; `_copy_tables` does."""
        if g_scale <= 0.0 or h_scale <= 0.0:
            raise Error("fixed-point scales must be positive")
        var dst = self.stage_param.unsafe_ptr()
        var base = record * PF_WORDS
        dst.unsafe_store(base + PF_G_INV, Float32(1.0 / g_scale))
        dst.unsafe_store(base + PF_H_INV, Float32(1.0 / h_scale))
        dst.unsafe_store(base + PF_LAMBDA_L2, Float32(params.lambda_l2))
        dst.unsafe_store(base + PF_LAMBDA_L1, Float32(params.lambda_l1))
        dst.unsafe_store(
            base + PF_MIN_CHILD_HESS, Float32(params.min_child_hess)
        )
        dst.unsafe_store(base + PF_BOUND_LO, _f32_bound(bounds.lo))
        dst.unsafe_store(base + PF_BOUND_HI, _f32_bound(bounds.hi))
        dst.unsafe_store(
            base + PF_CAT_SMOOTH, Float32(params.cat.cat_smooth)
        )
        dst.unsafe_store(base + PF_CAT_L2, Float32(params.cat.cat_l2))

    def _launch(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        params: GpuSplitParams,
        record_base: Int,
        n_records: Int,
        widest_slots: Int,
    ) raises:
        """Copy the staged tables and launch the two kernels over
        `n_records` consecutive record slots. The one place either entry
        point reaches the device."""
        self._copy_tables()
        _launch_search(
            self.ctx,
            hist,
            self.node_dev,
            self.feat_dev,
            self.allow_dev,
            self.missing_dev,
            self.catn_dev,
            self.mono_dev,
            self.fparam_dev,
            self.slot_i_dev,
            self.slot_f_dev,
            self.rec_i_dev,
            self.rec_f_dev,
            self.n_bins,
            self.n_features * self.n_bins,
            self.n_features,
            widest_slots,
            record_base,
            n_records,
            params.min_data_in_leaf,
            self.constrained,
            params.cat.max_cat_to_onehot,
            params.cat.max_cat_threshold,
            params.cat.min_data_per_group,
            self.wide_scan,
            self.use_primitives,
        )

    def enqueue(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds = OutputBounds.unbounded(),
        record: Int = 0,
        hist_slot: Int = 0,
    ) raises:
        """Enqueue the scan and reduction over `hist`, writing record slot
        `record`.

        It does transfer, and on Metal it therefore waits. `_launch` copies
        the four staged per-node tables across before either kernel is
        enqueued, and `enqueue_copy` on Metal is a synchronous full-queue
        drain (measured by disassembly, `docs/GPU_PORTABILITY.md` section
        6.1). An earlier version of this line said "does not transfer or
        synchronize" and was wrong on both halves. Nothing about the launch
        changed; what changed is what a wait count may claim.

        `hist` is a device buffer of `3 * n_features * n_bins` Int32 words in
        `GpuHistogramBuilder`'s `[grad | hess | count]` layout, and `g_scale`
        / `h_scale` are the fixed-point scales that histogram was accumulated
        with (`builder.g_scale`, `builder.h_scale`). `hist_slot` names which
        histogram inside a multi-slot buffer to read, for a caller holding a
        whole level's histograms at once (`GpuLeafBatcher.out_dev`, whose
        slot stride is exactly `3 * n_features * n_bins`); it stays 0 for
        the builder's single-node buffer.

        Ordering contract: this stages one record and copies the tables, so
        a node's `enqueue` is followed by its `download` (or an explicit
        `synchronize`) before the next node stages. That is the
        node-at-a-time loop, unchanged. A caller that wants a whole
        frontier without a wait per node uses `enqueue_frontier`, which
        stages every node before any table crosses."""
        self._check_record(record)
        if hist_slot < 0:
            raise Error("histogram slot must be nonnegative")
        self._stage_params(params, g_scale, h_scale, bounds, record)
        self.stage_node.unsafe_ptr().unsafe_store(
            record * NODE_WORDS + NODE_HIST_BASE,
            Int32(hist_slot * 3 * self.n_features * self.n_bins),
        )
        self._launch(hist, params, record, 1, self.active_len[record])

    def enqueue_pick_best(
        mut self, n_records: Int, record: Int = 0
    ) raises:
        """Reduce record slots `[0, n_records)` into slot `record`: the
        best-gain leaf of a frontier, ties going to the lower slot. The step
        that lets a whole tree level be selected without a host round trip
        per node.

        `record` must be outside `[0, n_records)` unless the frontier is
        being consumed, since the destination slot is overwritten in place."""
        if n_records < 1 or n_records > self.max_records:
            raise Error("n_records out of range")
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")
        self.ctx.enqueue_function[_pick_best_record_kernel](
            self.rec_i_dev.unsafe_ptr(),
            self.rec_f_dev.unsafe_ptr(),
            Int32(n_records),
            Int32(record),
            grid_dim=1,
            block_dim=1,
        )

    def download_words(
        mut self, mut words_i: List[Int32], mut words_f: List[Float32]
    ) raises:
        """Copy every record slot into two host lists. One host
        synchronization and `max_records * 136` bytes, whatever the histogram
        shape; a 100-feature, 256-bin node's histogram is 300 KB."""
        var n_i = self.max_records * SPLIT_IWORDS
        var n_f = self.max_records * SPLIT_FWORDS
        words_i.resize(n_i, Int32(0))
        words_f.resize(n_f, Float32(0.0))
        self.ctx.enqueue_copy(
            dst_ptr=self.host_i.unsafe_ptr(), src_buf=self.rec_i_dev
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_f.unsafe_ptr(), src_buf=self.rec_f_dev
        )
        self.ctx.synchronize()
        var src_i = self.host_i.unsafe_ptr()
        var src_f = self.host_f.unsafe_ptr()
        for i in range(n_i):
            words_i[i] = src_i.unsafe_load(i)
        for i in range(n_f):
            words_f[i] = src_f.unsafe_load(i)

    def download(mut self, record: Int = 0) raises -> GpuSplitRecord:
        """Copy the record buffer to the host and decode slot `record`."""
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")
        var words_i = List[Int32]()
        var words_f = List[Float32]()
        self.download_words(words_i, words_f)
        return decode_record(words_i, words_f, record)

    def search(
        mut self,
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds = OutputBounds.unbounded(),
        record: Int = 0,
    ) raises -> GpuSplitRecord:
        """Search the histogram staged by `upload_histogram` and return its
        record."""
        self._check_record(record)
        # Ordinarily a no-op, since `upload_histogram` has run and sized the
        # buffer; it is here so that this method never launches against the
        # placeholder, whatever order a caller uses.
        self._ensure_hist()
        self._stage_params(params, g_scale, h_scale, bounds, record)
        self.stage_node.unsafe_ptr().unsafe_store(
            record * NODE_WORDS + NODE_HIST_BASE, Int32(0)
        )
        # Do not call `_launch(self.hist_dev, ...)`: that borrows all of
        # `self` mutably for the method receiver while also borrowing one of
        # its fields mutably as an argument, which Mojo correctly rejects as
        # aliasing.  The owned-histogram path is the one place the histogram
        # belongs to the searcher, so spell the disjoint field borrows out at
        # the free-function boundary after staging the tables.
        #
        # `wide` is passed here, which it was not until the lane merge.
        # Omitting it made `search` run the serial scan even on a searcher
        # whose `wide_scan` was set, while `enqueue` and `enqueue_frontier`
        # (both through `_launch`) honored it. The trainer only ever uses
        # those two, so no fit was ever affected; what the omission did cost
        # was a test, because `test_gpu_split_search.test_wide_scan_matches
        # _the_serial_wide_scan` reaches the wide kernel through `search`
        # and had therefore been comparing the serial scan against itself.
        self._copy_tables()
        _launch_search(
            self.ctx,
            self.hist_dev,
            self.node_dev,
            self.feat_dev,
            self.allow_dev,
            self.missing_dev,
            self.catn_dev,
            self.mono_dev,
            self.fparam_dev,
            self.slot_i_dev,
            self.slot_f_dev,
            self.rec_i_dev,
            self.rec_f_dev,
            self.n_bins,
            self.n_features * self.n_bins,
            self.n_features,
            self.active_len[record],
            record,
            1,
            params.min_data_in_leaf,
            self.constrained,
            params.cat.max_cat_to_onehot,
            params.cat.max_cat_threshold,
            params.cat.min_data_per_group,
            wide=self.wide_scan,
            primitives=self.use_primitives,
        )
        return self.download(record)

    def enqueue_frontier(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        nodes: List[SplitNodeRequest],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
    ) raises:
        """Enqueue a whole frontier's searches in one pair of launches,
        writing record slots `[0, len(nodes))`.

        This is the entry point that removes the *download* per node. Every
        node's feature set, allow mask, monotone bounds, and histogram slot
        are written into their own record's staging slot first, then each
        table crosses once, then one scan covers the whole batch and one
        reduction writes every record. The host's next download is
        `download_frontier`, which is one for the level rather than one per
        leaf.

        It does transfer, and on Metal it therefore waits. Four table
        copies cross in `_copy_tables` before the launches, and on Metal an
        `enqueue_copy` is a synchronous full-queue drain in both directions,
        measured by disassembly and recorded in `docs/GPU_PORTABILITY.md`
        section 6.1. An earlier version of this line said "does not transfer
        and does not synchronize", which was wrong on both halves. What this
        entry point buys is one table crossing for a whole level instead of
        one per node, which is still the point of it; what it does not buy
        is a level with no host wait in it at all.

        `hist` holds every node's histogram: the builder's single-node
        buffer when the batch has one node, and a multi-slot buffer
        (`GpuLeafBatcher.out_dev`, slot stride `3 * n_features * n_bins`)
        when it has more. A node's `hist_slot` names its slot, and two
        nodes may name the same slot only if they really do share a
        histogram.

        What stays the caller's: which leaves are in the frontier at all,
        the depth and minimum-row rules (properties of the tree, not of a
        histogram), and the split it commits. What the batch does not
        change: the scan order inside a node, the cross-feature
        tie-breaking, and therefore every record it returns, which is
        identical to what the same nodes searched one at a time would
        return.
        """
        if len(nodes) < 1:
            raise Error("a frontier batch needs at least one node")
        if len(nodes) > self.max_records:
            raise Error(
                "the frontier is larger than the record capacity this"
                " searcher was constructed with"
            )
        var widest = 1
        var slot_cells = 3 * self.n_features * self.n_bins
        for i in range(len(nodes)):
            if nodes[i].hist_slot < 0:
                raise Error("histogram slot must be nonnegative")
            if len(nodes[i].features) > 0:
                self.set_features(nodes[i].features, record=i)
            self.set_allowed(nodes[i].allowed, record=i)
            self._stage_params(
                params, g_scale, h_scale, nodes[i].bounds, i
            )
            self.stage_node.unsafe_ptr().unsafe_store(
                i * NODE_WORDS + NODE_HIST_BASE,
                Int32(nodes[i].hist_slot * slot_cells),
            )
            if self.active_len[i] > widest:
                widest = self.active_len[i]
        self._launch(hist, params, 0, len(nodes), widest)

    def download_frontier(
        mut self, n_records: Int
    ) raises -> List[GpuSplitRecord]:
        """The batch's one wait: copy every record back and decode slots
        `[0, n_records)`.

        `max_records * 136` bytes and one synchronization, against one
        histogram download and one synchronization per node, which is what
        the node-at-a-time loop pays and what the whole record layout
        exists to avoid.
        """
        if n_records < 1 or n_records > self.max_records:
            raise Error("n_records out of range")
        var words_i = List[Int32]()
        var words_f = List[Float32]()
        self.download_words(words_i, words_f)
        var out = List[GpuSplitRecord](capacity=n_records)
        for r in range(n_records):
            out.append(decode_record(words_i, words_f, r))
        return out^

    def search_frontier(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        nodes: List[SplitNodeRequest],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
    ) raises -> List[GpuSplitRecord]:
        """`enqueue_frontier` followed by `download_frontier`: a bounded
        frontier's decisions, in one launch pair and one wait."""
        self.enqueue_frontier(hist, nodes, params, g_scale, h_scale)
        return self.download_frontier(len(nodes))


def decode_record(
    words_i: List[Int32], words_f: List[Float32], record: Int
) raises -> GpuSplitRecord:
    """Decode one record slot out of a downloaded record buffer pair. Public
    because the device-side queue will download many records at once and
    decode the slots it needs."""
    var io = record * SPLIT_IWORDS
    var fo = record * SPLIT_FWORDS
    if len(words_i) < io + SPLIT_IWORDS or len(words_f) < fo + SPLIT_FWORDS:
        raise Error("record buffers are too short for that record index")
    var flags = words_i[io + IREC_FLAGS]
    var out = GpuSplitRecord()
    out.feature = Int(words_i[io + IREC_FEATURE])
    out.bin = Int(words_i[io + IREC_BIN])
    out.ordinal = Int(words_i[io + IREC_ORDINAL])
    out.found = (flags & Int32(FLAG_FOUND)) != Int32(0)
    out.default_left = (flags & Int32(FLAG_DEFAULT_LEFT)) != Int32(0)
    out.is_categorical = (flags & Int32(FLAG_CATEGORICAL)) != Int32(0)
    out.gain = Float64(words_f[fo + FREC_GAIN])
    out.left = ChildStats(
        Float64(words_f[fo + FREC_LEFT_GRAD]),
        Float64(words_f[fo + FREC_LEFT_HESS]),
        Int(words_i[io + IREC_LEFT_COUNT]),
    )
    out.right = ChildStats(
        Float64(words_f[fo + FREC_RIGHT_GRAD]),
        Float64(words_f[fo + FREC_RIGHT_HESS]),
        Int(words_i[io + IREC_RIGHT_COUNT]),
    )
    out.total = ChildStats(
        Float64(words_f[fo + FREC_TOTAL_GRAD]),
        Float64(words_f[fo + FREC_TOTAL_HESS]),
        Int(words_i[io + IREC_TOTAL_COUNT]),
    )
    out.left_value = Float64(words_f[fo + FREC_LEFT_VALUE])
    out.right_value = Float64(words_f[fo + FREC_RIGHT_VALUE])
    out.parent_value = Float64(words_f[fo + FREC_PARENT_VALUE])
    out.runner_gain = Float64(words_f[fo + FREC_RUNNER_GAIN])
    if out.is_categorical:
        out.cat_bitset = _bitset_from_words(words_i, io)
    return out^


# --- Host reference -------------------------------------------------------


def reference_search(
    hist: List[Int32],
    n_features: Int,
    n_bins: Int,
    g_scale: Float64,
    h_scale: Float64,
    params: GpuSplitParams,
    features: List[Int] = [],
    allowed: List[Bool] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    cats: CategoricalSpec = CategoricalSpec.none(),
    bounds: OutputBounds = OutputBounds.unbounded(),
) raises -> GpuSplitRecord:
    """The kernels' arithmetic, run on the host over the same fixed-point
    histogram.

    This is not an independent implementation of split finding: it calls the
    same Float32 helpers the kernels do and walks the candidates in the same
    order, so what the tests compare is that the two loop structures agree,
    and what they assert against hand-computed gains is the shared
    arithmetic. It is also what makes the analytical tests runnable on a
    machine with no accelerator, and it is the reference the device path is
    validated against before it replaces the host scan.
    """
    if len(hist) != 3 * n_features * n_bins:
        raise Error("histogram must hold 3 * n_features * n_bins Int32 words")
    if g_scale <= 0.0 or h_scale <= 0.0:
        raise Error("fixed-point scales must be positive")
    if len(missing_bins) > 0 and len(missing_bins) != n_features:
        raise Error("missing_bins length must equal n_features")
    if len(monotone) > 0 and len(monotone) != n_features:
        raise Error("monotone length must equal n_features")

    var hs = n_features * n_bins
    var g_inv = Float32(1.0 / g_scale)
    var h_inv = Float32(1.0 / h_scale)
    var lambda_l2 = Float32(params.lambda_l2)
    var lambda_l1 = Float32(params.lambda_l1)
    var min_child_hess = Float32(params.min_child_hess)
    var min_data_in_leaf = Int32(params.min_data_in_leaf)
    var bound_lo = _f32_bound(bounds.lo)
    var bound_hi = _f32_bound(bounds.hi)
    var cat_smooth = Float32(params.cat.cat_smooth)
    var cat_l2 = Float32(params.cat.cat_l2)
    var constrained = len(monotone) > 0

    var active = features.copy()
    if len(active) == 0:
        active = List[Int](capacity=n_features)
        for f in range(n_features):
            active.append(f)

    var out = GpuSplitRecord()
    var best_slot = -1
    var best_gain = Float32(0.0)
    var slot_records = List[GpuSplitRecord]()
    var slot_gains = List[Float32]()
    var slot_runners = List[Float32]()

    for slot in range(len(active)):
        var f = active[slot]
        if f < 0 or f >= n_features:
            raise Error("feature index out of range")
        var base = f * n_bins
        var rec = GpuSplitRecord()

        var tg = Int32(0)
        var th = Int32(0)
        var tc = Int32(0)
        for b in range(n_bins):
            tg += hist[base + b]
            th += hist[hs + base + b]
            tc += hist[2 * hs + base + b]
        var total_g = tg.cast[DType.float32]() * g_inv
        var total_h = th.cast[DType.float32]() * h_inv
        rec.total = ChildStats(
            Float64(total_g), Float64(total_h), Int(tc)
        )

        var permitted = True
        if len(allowed) > 0:
            permitted = f < len(allowed) and allowed[f]
        if not permitted:
            slot_records.append(rec^)
            slot_gains.append(Float32(0.0))
            slot_runners.append(Float32(0.0))
            continue

        var sign = Int32(MONOTONE_FREE)
        if constrained:
            sign = Int32(monotone[f])
        var parent_score = gpu_leaf_score(
            total_g, total_h, lambda_l1, lambda_l2
        )
        var gain_here = Float32(0.0)
        # The same runner-up rule the kernel applies: the best gain among
        # every candidate this feature scored except the winner.
        var runner_here = Float32(0.0)
        var left_best_g = Int32(0)
        var left_best_h = Int32(0)
        var left_best_c = Int32(0)

        var n_cat = cats.n_categories(f) if cats.is_cat(f) else 0
        if n_cat >= 2:
            if n_cat >= n_bins:
                raise Error(
                    "categorical feature has more categories than bins"
                )
            if n_cat <= params.cat.max_cat_to_onehot:
                for t in range(1, n_cat + 1):
                    var lg = hist[base + t]
                    var lh = hist[hs + base + t]
                    var lc = hist[2 * hs + base + t]
                    if lc < min_data_in_leaf:
                        continue
                    var lhf = lh.cast[DType.float32]() * h_inv
                    if lhf < min_child_hess:
                        continue
                    var rc = tc - lc
                    if rc < min_data_in_leaf:
                        continue
                    var rhf = total_h - lhf
                    if rhf < min_child_hess:
                        continue
                    var lgf = lg.cast[DType.float32]() * g_inv
                    var rgf = total_g - lgf
                    var gain = (
                        gpu_leaf_score(lgf, lhf, lambda_l1, lambda_l2)
                        + gpu_leaf_score(rgf, rhf, lambda_l1, lambda_l2)
                        - parent_score
                    )
                    if gain > gain_here:
                        runner_here = gain_here
                        gain_here = gain
                        left_best_g = lg
                        left_best_h = lh
                        left_best_c = lc
                        rec.found = True
                        rec.is_categorical = True
                        rec.feature = f
                        rec.cat_bitset = cat_empty()
                        cat_add(rec.cat_bitset, t)
                    elif gain > runner_here:
                        runner_here = gain
            else:
                var keys = List[Float32]()
                var sorted_bins = List[Int]()
                for t in range(1, n_cat + 1):
                    var lc = hist[2 * hs + base + t]
                    if lc.cast[DType.float32]() < cat_smooth:
                        continue
                    var lg = hist[base + t]
                    var lh = hist[hs + base + t]
                    keys.append(
                        lg.cast[DType.float32]()
                        * g_inv
                        / (lh.cast[DType.float32]() * h_inv + cat_smooth)
                    )
                    sorted_bins.append(t)
                var used = len(sorted_bins)
                if used >= 2:
                    for i in range(1, used):
                        var kv = keys[i]
                        var bv = sorted_bins[i]
                        var j = i - 1
                        while j >= 0 and keys[j] > kv:
                            keys[j + 1] = keys[j]
                            sorted_bins[j + 1] = sorted_bins[j]
                            j -= 1
                        keys[j + 1] = kv
                        sorted_bins[j + 1] = bv

                    var l2c = lambda_l2 + cat_l2
                    var max_num_cat = params.cat.max_cat_threshold
                    if (used + 1) // 2 < max_num_cat:
                        max_num_cat = (used + 1) // 2
                    var steps = used if used < max_num_cat else max_num_cat
                    var min_group = Int32(params.cat.min_data_per_group)

                    for d in range(2):
                        var direction = 1 if d == 0 else -1
                        var start_pos = 0 if d == 0 else used - 1
                        var pos = start_pos
                        var group = Int32(0)
                        var lg = Int32(0)
                        var lh = Int32(0)
                        var lc = Int32(0)
                        for i in range(steps):
                            var t = sorted_bins[pos]
                            pos += direction
                            lg += hist[base + t]
                            lh += hist[hs + base + t]
                            var cnt = hist[2 * hs + base + t]
                            lc += cnt
                            group += cnt

                            var lhf = lh.cast[DType.float32]() * h_inv
                            if lc < min_data_in_leaf or lhf < min_child_hess:
                                continue
                            var rc = tc - lc
                            if rc < min_data_in_leaf or rc < min_group:
                                break
                            var rhf = total_h - lhf
                            if rhf < min_child_hess:
                                break
                            if group < min_group:
                                continue
                            group = Int32(0)

                            var lgf = lg.cast[DType.float32]() * g_inv
                            var rgf = total_g - lgf
                            var gain = (
                                gpu_leaf_score(lgf, lhf, lambda_l1, l2c)
                                + gpu_leaf_score(rgf, rhf, lambda_l1, l2c)
                                - parent_score
                            )
                            if gain > gain_here:
                                runner_here = gain_here
                                gain_here = gain
                                left_best_g = lg
                                left_best_h = lh
                                left_best_c = lc
                                rec.found = True
                                rec.is_categorical = True
                                rec.feature = f
                                rec.cat_bitset = cat_empty()
                                var p = start_pos
                                for _ in range(i + 1):
                                    cat_add(rec.cat_bitset, sorted_bins[p])
                                    p += direction
                            elif gain > runner_here:
                                runner_here = gain
        else:
            var missing_bin = -1
            if len(missing_bins) > 0:
                missing_bin = missing_bins[f]
            var n_scan = missing_bin if missing_bin >= 0 else n_bins
            var miss_g = Int32(0)
            var miss_h = Int32(0)
            var miss_c = Int32(0)
            if missing_bin >= 0:
                miss_g = hist[base + missing_bin]
                miss_h = hist[hs + base + missing_bin]
                miss_c = hist[2 * hs + base + missing_bin]

            var left_g = Int32(0)
            var left_h = Int32(0)
            var left_c = Int32(0)
            for b in range(n_scan):
                if b == n_scan - 1 and miss_c == Int32(0):
                    break
                left_g += hist[base + b]
                left_h += hist[hs + base + b]
                left_c += hist[2 * hs + base + b]

                if missing_bin >= 0:
                    var dl_g = left_g + miss_g
                    var dl_h = left_h + miss_h
                    var dl_c = left_c + miss_c
                    var dl_hf = dl_h.cast[DType.float32]() * h_inv
                    var dr_hf = total_h - dl_hf
                    if not (
                        dl_hf < min_child_hess
                        or dr_hf < min_child_hess
                        or dl_c < min_data_in_leaf
                        or tc - dl_c < min_data_in_leaf
                    ):
                        var dl_gf = dl_g.cast[DType.float32]() * g_inv
                        var dr_gf = total_g - dl_gf
                        var gain = gpu_split_gain(
                            gpu_soft_threshold_l1(dl_gf, lambda_l1),
                            dl_hf,
                            gpu_soft_threshold_l1(dr_gf, lambda_l1),
                            dr_hf,
                            lambda_l2,
                            parent_score,
                            sign,
                            bound_lo,
                            bound_hi,
                            constrained,
                        )
                        if gain > gain_here:
                            runner_here = gain_here
                            gain_here = gain
                            left_best_g = dl_g
                            left_best_h = dl_h
                            left_best_c = dl_c
                            rec.found = True
                            rec.feature = f
                            rec.bin = b
                            rec.ordinal = 2 * b
                            rec.default_left = True
                        elif gain > runner_here:
                            runner_here = gain

                if missing_bin < 0 or miss_c > Int32(0):
                    var lhf = left_h.cast[DType.float32]() * h_inv
                    var rhf = total_h - lhf
                    if lhf < min_child_hess or rhf < min_child_hess:
                        continue
                    if (
                        left_c < min_data_in_leaf
                        or tc - left_c < min_data_in_leaf
                    ):
                        continue
                    var lgf = left_g.cast[DType.float32]() * g_inv
                    var rgf = total_g - lgf
                    var gain = gpu_split_gain(
                        gpu_soft_threshold_l1(lgf, lambda_l1),
                        lhf,
                        gpu_soft_threshold_l1(rgf, lambda_l1),
                        rhf,
                        lambda_l2,
                        parent_score,
                        sign,
                        bound_lo,
                        bound_hi,
                        constrained,
                    )
                    if gain > gain_here:
                        runner_here = gain_here
                        gain_here = gain
                        left_best_g = left_g
                        left_best_h = left_h
                        left_best_c = left_c
                        rec.found = True
                        rec.feature = f
                        rec.bin = b
                        rec.ordinal = 2 * b + 1
                        rec.default_left = False
                    elif gain > runner_here:
                        runner_here = gain

        if rec.found:
            var lgf = left_best_g.cast[DType.float32]() * g_inv
            var lhf = left_best_h.cast[DType.float32]() * h_inv
            rec.gain = Float64(gain_here)
            rec.left = ChildStats(
                Float64(lgf), Float64(lhf), Int(left_best_c)
            )
            rec.right = ChildStats(
                Float64(total_g - lgf),
                Float64(total_h - lhf),
                Int(tc - left_best_c),
            )
        slot_records.append(rec^)
        slot_gains.append(gain_here)
        slot_runners.append(runner_here)

    # The same fold `_reduce_slots_kernel` performs: the winner by strictly
    # greater gain in ascending slot order, and the node's runner-up as the
    # better of the best losing feature and the winner's own second
    # candidate.
    var runner = Float32(0.0)
    for slot in range(len(slot_records)):
        if not slot_records[slot].found:
            continue
        if best_slot < 0 or slot_gains[slot] > best_gain:
            if best_slot >= 0 and best_gain > runner:
                runner = best_gain
            best_slot = slot
            best_gain = slot_gains[slot]
        elif slot_gains[slot] > runner:
            runner = slot_gains[slot]

    var total = slot_records[0].total.copy()
    if best_slot >= 0:
        out = slot_records[best_slot].copy()
        if slot_runners[best_slot] > runner:
            runner = slot_runners[best_slot]
        out.runner_gain = Float64(runner)
    out.total = total.copy()
    out.parent_value = Float64(
        gpu_leaf_value(
            Float32(total.grad), Float32(total.hess), lambda_l1, lambda_l2
        )
    )
    if out.found:
        out.left_value = Float64(
            gpu_leaf_value(
                Float32(out.left.grad),
                Float32(out.left.hess),
                lambda_l1,
                lambda_l2,
            )
        )
        out.right_value = Float64(
            gpu_leaf_value(
                Float32(out.right.grad),
                Float32(out.right.hess),
                lambda_l1,
                lambda_l2,
            )
        )
    return out^
