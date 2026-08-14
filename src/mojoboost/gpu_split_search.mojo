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
winning feature, threshold or category set, gain, missing direction, both
children's statistics, and both children's leaf values. That record is the
only thing that crosses back to the host, so a node costs a fixed ~100 bytes
instead of a histogram.

Nothing here is wired into the trainer yet; `handoffs/apple_a2_split_search.md`
carries the exact integration. The module is standalone and testable on its
own: it owns a histogram buffer that a caller can upload to directly, and
`enqueue` also accepts an external device buffer for the zero-copy path.

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

- `_scan_slot_kernel`, one threadgroup per active feature, writes that
  feature's best candidate into a per-slot record.
- `_reduce_slots_kernel`, a single thread, folds the per-slot records into
  one record in ascending slot order and fills in the child statistics,
  child leaf values, and the parent's leaf value.

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
`pick_best_record` reduces a set of them to the single best-gain leaf,
tie-broken by ascending record index. That is the seed of a fully
device-side leaf-wise frontier: today the host downloads one record per node
and drives growth; next the host downloads one record per *tree level*; last
the frontier itself lives in `rec_i_dev`/`rec_f_dev` and the host only reads
the finished tree. Nothing in the record layout has to change for that, which
is why the record carries child statistics and leaf values rather than making
the host recompute them from a histogram it no longer has.
"""

from std.gpu import block_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace

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


def _launch_search(
    mut ctx: DeviceContext,
    mut hist: DeviceBuffer[DType.int32],
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
    n_slots: Int,
    min_data_in_leaf: Int,
    constrained: Bool,
    cat_onehot_max: Int,
    cat_max_threshold: Int,
    cat_min_group: Int,
    record: Int,
) raises:
    """Enqueue the two kernels of one node's search.

    A free function over the context and the buffers rather than a method, so
    the histogram buffer is an ordinary argument whether it belongs to this
    searcher or to the histogram builder next to it."""
    ctx.enqueue_function[_scan_slot_kernel](
        hist.unsafe_ptr(),
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
        Int32(min_data_in_leaf),
        Int32(1) if constrained else Int32(0),
        Int32(cat_onehot_max),
        Int32(cat_max_threshold),
        Int32(cat_min_group),
        grid_dim=n_slots,
        block_dim=1,
    )
    ctx.enqueue_function[_reduce_slots_kernel](
        slot_i.unsafe_ptr(),
        slot_f.unsafe_ptr(),
        rec_i.unsafe_ptr(),
        rec_f.unsafe_ptr(),
        fparam.unsafe_ptr(),
        Int32(n_slots),
        Int32(record),
        grid_dim=1,
        block_dim=1,
    )


struct GpuSplitSearcher(Movable):
    """Device-resident split search for one dataset shape.

    Construct once per training session next to the histogram builder, call
    `set_features` once per tree, `set_allowed`/`set_monotone` per node, and
    `search` (or `enqueue` + `download`) per node. Nothing is allocated per
    node: every buffer is sized at construction from `n_features`, `n_bins`,
    and `max_records`.
    """

    var ctx: DeviceContext
    var n_features: Int
    var n_bins: Int
    var max_records: Int
    # The histogram this searcher owns, in `GpuHistogramBuilder`'s layout, for
    # callers that stage a histogram through the host. The zero-copy path
    # passes the builder's own buffer to `enqueue` instead.
    var hist_dev: DeviceBuffer[DType.int32]
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
    # Pinned staging for the two per-node tables, so a node's parameters
    # upload as an ordinary asynchronous copy rather than through
    # `map_to_host`, which synchronizes on every use. Reused every node: see
    # the ordering contract on `enqueue`.
    var stage_allow: HostBuffer[DType.int32]
    var stage_param: HostBuffer[DType.float32]
    var host_i: HostBuffer[DType.int32]
    var host_f: HostBuffer[DType.float32]
    var active: List[Int]
    var missing_bin: List[Int]
    var cat_n: List[Int]
    var constrained: Bool

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
        for f in range(n_features):
            self.missing_bin.append(
                missing_bins[f] if len(missing_bins) > 0 else -1
            )
            self.cat_n.append(cats.n_categories(f) if cats.is_cat(f) else 0)
            self.active.append(f)

        var hist_size = n_features * n_bins
        self.hist_dev = self.ctx.enqueue_create_buffer[DType.int32](
            3 * hist_size
        )
        self.feat_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.allow_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_features
        )
        self.missing_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_features
        )
        self.catn_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.mono_dev = self.ctx.enqueue_create_buffer[DType.int32](n_features)
        self.fparam_dev = self.ctx.enqueue_create_buffer[DType.float32](
            PF_WORDS
        )
        self.slot_i_dev = self.ctx.enqueue_create_buffer[DType.int32](
            n_features * SPLIT_IWORDS
        )
        self.slot_f_dev = self.ctx.enqueue_create_buffer[DType.float32](
            n_features * SPLIT_FWORDS
        )
        self.rec_i_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_records * SPLIT_IWORDS
        )
        self.rec_f_dev = self.ctx.enqueue_create_buffer[DType.float32](
            max_records * SPLIT_FWORDS
        )
        self.stage_allow = self.ctx.enqueue_create_host_buffer[DType.int32](
            n_features
        )
        self.stage_param = self.ctx.enqueue_create_host_buffer[DType.float32](
            PF_WORDS
        )
        self.host_i = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_records * SPLIT_IWORDS
        )
        self.host_f = self.ctx.enqueue_create_host_buffer[DType.float32](
            max_records * SPLIT_FWORDS
        )

        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                dst.unsafe_store(f, Int32(f))
        with self.allow_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(n_features):
                dst.unsafe_store(f, Int32(1))
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

    def synchronize(self) raises:
        """Block until every enqueued device operation has completed."""
        self.ctx.synchronize()

    def n_active(self) -> Int:
        """How many feature slots the next search scans."""
        return len(self.active)

    def set_features(mut self, features: List[Int]) raises:
        """Restrict later searches to `features` (global feature ids,
        ascending, one entry each), the same subsampled set
        `GpuHistogramBuilder.set_features` accumulated. Slot order is scan
        order, so it also fixes the cross-feature tie-breaking."""
        if len(features) == 0:
            raise Error("active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
        self.active = features.copy()
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(features)):
                dst.unsafe_store(i, Int32(features[i]))
        # The allow mask is indexed by slot, so a new slot order invalidates
        # it; reset it to "every listed feature allowed" and let the node's
        # `set_allowed` narrow it again.
        self.set_allowed()

    def set_allowed(mut self, allowed: List[Bool] = []) raises:
        """This node's interaction-constraint allow mask, indexed by global
        feature id. Empty (the default) allows every feature; a mask shorter
        than `n_features` disallows the features past its end, exactly as
        `find_best_split` reads it.

        Indexed by slot on the device, so it is re-staged after every
        `set_features`."""
        var dst = self.stage_allow.unsafe_ptr()
        for i in range(len(self.active)):
            var f = self.active[i]
            var ok = True
            if len(allowed) > 0:
                ok = f < len(allowed) and allowed[f]
            dst.unsafe_store(i, Int32(1) if ok else Int32(0))
        for i in range(len(self.active), self.n_features):
            dst.unsafe_store(i, Int32(0))
        self.ctx.enqueue_copy(
            dst_buf=self.allow_dev, src_ptr=self.stage_allow.unsafe_ptr()
        )

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

    def upload_histogram(mut self, words: List[Int32]) raises:
        """Stage a fixed-point `[grad | hess | count]` histogram into this
        searcher's own buffer. The trainer integration uses the zero-copy
        `enqueue` overload against the builder's buffer instead; this exists
        so the search is exercisable, and benchmarkable, on its own."""
        if len(words) != 3 * self.n_features * self.n_bins:
            raise Error(
                "histogram must hold 3 * n_features * n_bins Int32 words"
            )
        self.ctx.enqueue_copy(
            dst_buf=self.hist_dev, src_ptr=words.unsafe_ptr()
        )
        self.ctx.synchronize()

    def _upload_params(
        mut self,
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds,
    ) raises:
        if g_scale <= 0.0 or h_scale <= 0.0:
            raise Error("fixed-point scales must be positive")
        var dst = self.stage_param.unsafe_ptr()
        dst.unsafe_store(PF_G_INV, Float32(1.0 / g_scale))
        dst.unsafe_store(PF_H_INV, Float32(1.0 / h_scale))
        dst.unsafe_store(PF_LAMBDA_L2, Float32(params.lambda_l2))
        dst.unsafe_store(PF_LAMBDA_L1, Float32(params.lambda_l1))
        dst.unsafe_store(PF_MIN_CHILD_HESS, Float32(params.min_child_hess))
        dst.unsafe_store(PF_BOUND_LO, _f32_bound(bounds.lo))
        dst.unsafe_store(PF_BOUND_HI, _f32_bound(bounds.hi))
        dst.unsafe_store(PF_CAT_SMOOTH, Float32(params.cat.cat_smooth))
        dst.unsafe_store(PF_CAT_L2, Float32(params.cat.cat_l2))
        self.ctx.enqueue_copy(
            dst_buf=self.fparam_dev, src_ptr=self.stage_param.unsafe_ptr()
        )

    def enqueue(
        mut self,
        mut hist: DeviceBuffer[DType.int32],
        params: GpuSplitParams,
        g_scale: Float64,
        h_scale: Float64,
        bounds: OutputBounds = OutputBounds.unbounded(),
        record: Int = 0,
    ) raises:
        """Enqueue the scan and reduction over `hist`, writing record slot
        `record`. Does not transfer or synchronize.

        `hist` is a device buffer of `3 * n_features * n_bins` Int32 words in
        `GpuHistogramBuilder`'s `[grad | hess | count]` layout, and `g_scale`
        / `h_scale` are the fixed-point scales that histogram was accumulated
        with (`builder.g_scale`, `builder.h_scale`).

        Ordering contract: the allow mask and the parameter block are staged
        through one pinned buffer each, reused by every node, so a node's
        `enqueue` must be followed by its `download` (or an explicit
        `synchronize`) before the next node's `set_allowed`/`enqueue`
        overwrites them. That is exactly the incremental integration's
        one-node-at-a-time loop; a fully device-side queue needs a per-leaf
        parameter slot instead."""
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")
        self._upload_params(params, g_scale, h_scale, bounds)
        _launch_search(
            self.ctx,
            hist,
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
            len(self.active),
            params.min_data_in_leaf,
            self.constrained,
            params.cat.max_cat_to_onehot,
            params.cat.max_cat_threshold,
            params.cat.min_data_per_group,
            record,
        )

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
        if record < 0 or record >= self.max_records:
            raise Error("record index out of range")
        self._upload_params(params, g_scale, h_scale, bounds)
        _launch_search(
            self.ctx,
            self.hist_dev,
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
            len(self.active),
            params.min_data_in_leaf,
            self.constrained,
            params.cat.max_cat_to_onehot,
            params.cat.max_cat_threshold,
            params.cat.min_data_per_group,
            record,
        )
        return self.download(record)


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
            continue

        var sign = Int32(MONOTONE_FREE)
        if constrained:
            sign = Int32(monotone[f])
        var parent_score = gpu_leaf_score(
            total_g, total_h, lambda_l1, lambda_l2
        )
        var gain_here = Float32(0.0)
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
                        gain_here = gain
                        left_best_g = lg
                        left_best_h = lh
                        left_best_c = lc
                        rec.found = True
                        rec.is_categorical = True
                        rec.feature = f
                        rec.cat_bitset = cat_empty()
                        cat_add(rec.cat_bitset, t)
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
                            gain_here = gain
                            left_best_g = dl_g
                            left_best_h = dl_h
                            left_best_c = dl_c
                            rec.found = True
                            rec.feature = f
                            rec.bin = b
                            rec.ordinal = 2 * b
                            rec.default_left = True

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
                        gain_here = gain
                        left_best_g = left_g
                        left_best_h = left_h
                        left_best_c = left_c
                        rec.found = True
                        rec.feature = f
                        rec.bin = b
                        rec.ordinal = 2 * b + 1
                        rec.default_left = False

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

    for slot in range(len(slot_records)):
        if not slot_records[slot].found:
            continue
        if best_slot < 0 or slot_gains[slot] > best_gain:
            best_slot = slot
            best_gain = slot_gains[slot]

    var total = slot_records[0].total.copy()
    if best_slot >= 0:
        out = slot_records[best_slot].copy()
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
