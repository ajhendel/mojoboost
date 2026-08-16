"""Device-side gradients, hessians, and raw-score updates.

The GPU trainer in train_gpu.mojo computes every round's derivatives on the
host (`_fill_grad_hess` over Float64 lists) and then uploads `2 * n_rows`
Float32 to the device. That upload is pure overhead: the labels never change,
the sample weights never change, and the raw scores are the only thing that
moves between rounds, by an amount the device already knows (each row's leaf
and that leaf's value). This module keeps all three device-resident so a
boosting round costs no per-row host-to-device traffic at all.

What lives here:

  `GpuObjectiveState`   the per-session device buffers: target (or class
                        label), sample weight, raw scores, and softmax
                        probabilities. Uploaded once at construction, never
                        again.
  gradient kernels      one thread per row, one straight-line body per
                        objective, writing into gradient/hessian buffers the
                        caller owns (in practice `GpuHistogramBuilder`'s, so
                        the histogram kernels read what these wrote without
                        anything crossing the boundary).
  `update_raw`          the device-resident prediction update:
                        `raw[r] += learning_rate * value[leaf_id[r]]`, from
                        the leaf-assignment array tree growth already left on
                        the device and a node-value table of `n_nodes`
                        floats.
  `magnitude_sums`      the two magnitude sums the fixed-point histogram
                        scale is derived from, reduced on the device, so the
                        host reads back 8 bytes per round instead of
                        `8 * n_rows`.

Objectives covered: squared error, binary logistic, cross entropy, poisson,
gamma, tweedie, huber, quantile, L1, MAPE, fair, and softmax multiclass.
Every one of those has a closed-form per-row derivative in the raw score,
which is exactly the interface a one-thread-per-row kernel can serve. CUSTOM
is the exception and stays on the host by construction: the callback is
Python or Mojo code over host-side `List[Float64]`, there is no device image
of it, and `train_custom_gpu`'s contract (one call per round over the whole
row set, then `check_custom_grad_hess`) is unchanged by anything here. The
host path is preserved, not replaced; `supports_device_objective` is the one
place that says which objective goes which way, and it now answers from
`objective_gradients_on_device` in objective_registry.mojo rather than from a
second if-chain over the same codes, so the capability table has one
definition and the GPU module is the dependent side of it.

Precision. Apple GPUs have no Float64, so the device carries raw scores,
labels, weights, gradients, and hessians as Float32, where the CPU trainer
carries Float64. Agreement with `fill_grad_hess` is to Float32 precision, not
bit-exact, which is the same trade the Float32 histogram path already makes
(see histogram_gpu.mojo). Within the device path the results are
bit-deterministic run to run: every kernel here is a per-row map except the
magnitude reduction, which sums in a fixed grid-stride and tree order over a
fixed block count.

Stability and clipping. Float32 overflows at ~3.4e38, so every exponent
argument is clamped to +/- `EXP_ARG_LIMIT` before `exp`; the CPU path does
not clamp because Float64 does not need it below the same raw scores. The
softmax kernel subtracts the row max first, as the CPU one does, so its
exponent arguments are already non-positive. Hessians are floored at
`HESS_FLOOR`, the same 1e-16 the CPU logistic and softmax objectives use, so
a saturated row still has positive-semidefinite curvature. See
`docs/LIGHTGBM_PARITY.md` for where the CPU objectives themselves diverge
from LightGBM; nothing here adds a divergence beyond the Float32 carrier and
the exponent clamp.

Sample weights follow the built-in convention exactly: the kernel multiplies
both derivatives by the row weight in the same operand order as
`_fill_grad_hess_into`, so a zero-weight row produces a zero gradient and a
zero hessian and is invisible to every histogram, on either backend.
"""

from std.gpu import block_idx, global_idx, thread_idx
from std.math import exp, isfinite
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    TWEEDIE,
    _POISSON_MAX_DELTA_STEP,
)
from .gpu_active_rows import GpuActiveRows
from .gpu_tiling import derive_block_threads, query_device_caps
from .monotone import NO_BOUND, OutputBounds
from .objective_registry import objective_gradients_on_device
from .quantized_gradient import fixed_point_scale

# Clamp on every `exp` argument. exp(60) is 1.1e26, four orders of magnitude
# inside the Float32 maximum, so the poisson hessian's extra
# `_POISSON_MAX_DELTA_STEP` and the tweedie exponents cannot push a clamped
# argument over. A raw score that reaches this has diverged; clamping keeps
# the gradient finite so the fixed-point scale reports a real magnitude
# instead of failing on an infinity.
comptime EXP_ARG_LIMIT = Float32(60.0)

# The same hessian floor the CPU logistic and softmax objectives apply.
comptime HESS_FLOOR = Float32(1e-16)

# Threads for the magnitude reduction. Fixed rather than device-derived
# because the shared-memory tree reduction needs a compile-time size; the
# per-row kernels take their geometry from `derive_block_threads` like the
# histogram kernels do.
comptime SUM_THREADS = 256
comptime SUM_BLOCKS = 256

comptime SCALE_WINDOW_MAX = 64
"""How many rounds' magnitude partials `GpuObjectiveState` can hold before
the host has to fold them.

The ceiling on `GpuHistogramBuilder.set_scale_refresh`, and the reason it is
a ceiling rather than a limit nobody states: each unfolded round is one more
round quantized on a scale derived from magnitudes that are that many rounds
old, and the resolution given up grows with the staleness. Sixty-four is
already far past anything defensible; it exists so the buffer arithmetic has
a bound, not as an invitation.

The cost is `SCALE_WINDOW_MAX * 2 * SUM_BLOCKS` Float32 of pinned host
memory, 128 KB, once per training session and independent of `n_rows`."""

# Default capacity of the node-value table `update_raw` uploads. A tree has
# `2 * num_leaves - 1` nodes, so this covers num_leaves up to 1024 without a
# caller having to think about it.
comptime DEFAULT_MAX_NODES = 2048

# Leaf id of a row tree growth never routed (any negative id, and any id
# past the tree's node count). `update_raw` leaves those rows alone.
comptime UNROUTED_LEAF = Int32(-1)


def supports_device_objective(objective: Int) -> Bool:
    """Whether `objective` has a closed-form per-row derivative these kernels
    implement. False for CUSTOM, whose callback lives on the host, and for
    any code the built-in trainer does not define.

    The answer comes from `objective_gradients_on_device` in
    objective_registry.mojo, which is the one table of objective facts; this
    function is the name the GPU modules already import and stays as the
    device-side spelling of that question. The two used to be independent
    if-chains over the same eleven codes, which is a capability table
    maintained twice: the registry's docstring named this file as the
    mirror to delete, and this delegation is that deletion, in the
    direction the registry asked for (the GPU module depends on the
    metadata module, never the reverse, so nothing drags `max.gpu.*` into
    params.mojo or the CLI).

    The set is unchanged, value for value: `objective_gradients_on_device`
    returns `objective_is_builtin`, whose eleven codes are exactly the arms
    the chain here enumerated, and softmax multiclass is still absent from
    both because it is served by `refresh_softmax` and
    `fill_softmax_grad_hess` rather than by `_grad_hess_kernel`.
    """
    return objective_gradients_on_device(objective)


def device_fixed_scale(total: Float64) raises -> Float32:
    """The fixed-point histogram scale for a device magnitude sum.

    `quantized_gradient.fixed_point_scale` is the definition; this name
    remains because GPU modules already import it and because it identifies
    which side of the boundary produced the magnitude sum.
    """
    return fixed_point_scale(total)


@always_inline
def _dev_exp(x: Float32) -> Float32:
    """`exp` with its argument clamped into the Float32-safe range."""
    var a = x
    if a > EXP_ARG_LIMIT:
        a = EXP_ARG_LIMIT
    elif a < -EXP_ARG_LIMIT:
        a = -EXP_ARG_LIMIT
    return exp(a)


@always_inline
def _dev_sigmoid(x: Float32) -> Float32:
    """The same branch-on-sign form as `_sigmoid` in boosting.mojo, so the
    exponent argument is never positive and the result never overflows."""
    if x >= 0.0:
        var e = exp(-x) if x < EXP_ARG_LIMIT else Float32(0.0)
        return 1.0 / (1.0 + e)
    var e = _dev_exp(x)
    return e / (1.0 + e)


@always_inline
def _dev_sign(x: Float32) -> Float32:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@always_inline
def _dev_mape_weight(y: Float32) -> Float32:
    """LightGBM's MAPE label weight `1 / max(1, |y|)`."""
    var m = abs(y)
    return 1.0 / m if m > 1.0 else Float32(1.0)


def _grad_hess_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    objective: Int32,
    alpha: Float32,
    weighted: Int32,
):
    """One row per thread: the first and second derivative of `objective` at
    that row's raw score, weighted, written into `grad` and `hess`.

    The objective test is uniform across the whole grid, so the branch costs
    no divergence: every thread of every threadgroup takes the same arm. The
    arms are in the same order and compute the same expressions as
    `_fill_grad_hess_into` in boosting.mojo, in Float32 and with the exponent
    clamp; anything else would be a second definition of the objectives.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return

    var w = Float32(1.0)
    if weighted != 0:
        w = weight[unsafe_offset=r][0]
    var raw_r = raw[unsafe_offset=r][0]
    var y = target[unsafe_offset=r][0]

    if objective == Int32(BINARY_LOGISTIC) or objective == Int32(
        CROSS_ENTROPY
    ):
        # One arm for both: cross entropy is the logistic loss with the
        # {0, 1} label relaxed to a probability, so the derivatives are the
        # same expression and only the label validation differs (host side).
        var p = _dev_sigmoid(raw_r)
        grad[unsafe_offset=r] = w * (p - y)
        var h = p * (1.0 - p)
        if h < HESS_FLOOR:
            h = HESS_FLOOR
        hess[unsafe_offset=r] = w * h
    elif objective == Int32(GAMMA):
        var y_over_mu = y * _dev_exp(-raw_r)
        grad[unsafe_offset=r] = w * (1.0 - y_over_mu)
        hess[unsafe_offset=r] = w * y_over_mu
    elif objective == Int32(TWEEDIE):
        # alpha is the variance power in (1, 2), so 1 - alpha < 0 and
        # 2 - alpha > 0 and the hessian stays nonnegative.
        var e1 = _dev_exp((1.0 - alpha) * raw_r)
        var e2 = _dev_exp((2.0 - alpha) * raw_r)
        grad[unsafe_offset=r] = w * (-y * e1 + e2)
        hess[unsafe_offset=r] = w * (
            -y * (1.0 - alpha) * e1 + (2.0 - alpha) * e2
        )
    elif objective == Int32(MAPE):
        var lw = w * _dev_mape_weight(y)
        grad[unsafe_offset=r] = lw * _dev_sign(raw_r - y)
        hess[unsafe_offset=r] = lw
    elif objective == Int32(FAIR):
        var d = raw_r - y
        var denom = abs(d) + alpha
        grad[unsafe_offset=r] = w * alpha * d / denom
        hess[unsafe_offset=r] = w * alpha * alpha / (denom * denom)
    elif objective == Int32(POISSON):
        grad[unsafe_offset=r] = w * (_dev_exp(raw_r) - y)
        hess[unsafe_offset=r] = w * _dev_exp(
            raw_r + Float32(_POISSON_MAX_DELTA_STEP)
        )
    elif objective == Int32(HUBER):
        var d = raw_r - y
        if abs(d) <= alpha:
            grad[unsafe_offset=r] = w * d
        else:
            grad[unsafe_offset=r] = w * _dev_sign(d) * alpha
        hess[unsafe_offset=r] = w
    elif objective == Int32(QUANTILE):
        var d = raw_r - y
        if d >= 0.0:
            grad[unsafe_offset=r] = w * (1.0 - alpha)
        else:
            grad[unsafe_offset=r] = w * -alpha
        hess[unsafe_offset=r] = w
    elif objective == Int32(L1):
        grad[unsafe_offset=r] = w * _dev_sign(raw_r - y)
        hess[unsafe_offset=r] = w
    else:
        # Squared error, and the only arm an unlisted code could reach.
        # `fill_grad_hess` refuses every code `supports_device_objective`
        # rejects before the launch, so nothing else arrives here.
        grad[unsafe_offset=r] = w * (raw_r - y)
        hess[unsafe_offset=r] = w


def _softmax_prob_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    prob: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
):
    """Row-major softmax over `n_classes` raw scores per row, max-subtracted
    exactly as `_softmax_inplace` does on the host. Every exponent argument
    is therefore non-positive and the denominator is at least 1, so no clamp
    is needed here."""
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var k = Int(n_classes)
    var base = r * k

    var m = raw[unsafe_offset=base][0]
    for i in range(1, k):
        var v = raw[unsafe_offset = base + i][0]
        if v > m:
            m = v
    var total = Float32(0.0)
    for i in range(k):
        var e = exp(raw[unsafe_offset = base + i][0] - m)
        prob[unsafe_offset = base + i] = e
        total += e
    for i in range(k):
        prob[unsafe_offset = base + i] = (
            prob[unsafe_offset = base + i][0] / total
        )


def _softmax_class_kernel(
    prob: MutPointer[Float32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
    k: Int32,
    weighted: Int32,
):
    """One-vs-rest derivatives for class `k` from the probabilities
    `_softmax_prob_kernel` left behind, matching `_fill_softmax_grad_hess`:
    gradient `p - y`, hessian `(k / (k - 1)) p (1 - p)` floored.

    `target` holds the integer class label as a Float32. Class counts are far
    below 2^24, so the label is exact in Float32 and the equality test is
    exact too.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var w = Float32(1.0)
    if weighted != 0:
        w = weight[unsafe_offset=r][0]
    var p = prob[unsafe_offset = r * Int(n_classes) + Int(k)][0]
    var y = Float32(0.0)
    if target[unsafe_offset=r][0] == Float32(Int(k)):
        y = 1.0
    grad[unsafe_offset=r] = w * (p - y)
    # LightGBM's factor_ = k / (k - 1), not the old hardcoded 2 (exact only
    # at two classes); see _fill_softmax_grad_hess in boosting.mojo.
    var factor = Float32(Int(n_classes)) / Float32(Int(n_classes) - 1)
    var h = factor * p * (1.0 - p)
    if h < HESS_FLOOR:
        h = HESS_FLOOR
    hess[unsafe_offset=r] = w * h


def _init_raw_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    base: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
):
    """Set every row's raw scores to the per-class base scores. Written as a
    kernel rather than a memset so the multiclass case, whose base scores are
    per-class log priors, goes through the same path as the single-output
    one."""
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var k = Int(n_classes)
    for i in range(k):
        raw[unsafe_offset = r * k + i] = base[unsafe_offset=i][0]


def _update_raw_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    leaf_ids: MutPointer[Int32, MutAnyOrigin],
    steps: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
    k: Int32,
    n_nodes: Int32,
):
    """The device-resident prediction update: each row's raw score for class
    `k` advances by `step[leaf]`, the already-shrunk value of the node the
    row sits in on the device.

    This is the whole reason the leaf-assignment array is worth keeping
    around after a tree is grown. Every row ends a tree assigned to a leaf,
    and leaf ids are node ids, so the tree's own per-node step is the lookup
    table; no traversal, no feature reads, no host round trip.

    A row whose id is out of range (`OUT_OF_BAG`, or any id at or past the
    tree's node count) is left untouched: tree growth never routed it, so the
    device does not know its leaf. Under row bagging that is every
    out-of-bag row, and the caller has to score those rows some other way.

    Why this takes a step and not a value and a learning rate
    ---------------------------------------------------------
    It used to compute `raw[i] + learning_rate * value[node]` inline, and
    because `node` is read per-thread out of `leaf_ids` the product varied
    across the launch and the device compiler contracted it into the add.
    The two range kernels below do not contract: the per-leaf one takes
    `node` as a launch argument, so its product is uniform and gets hoisted
    and rounded on its own, and the range-table one has no multiply left in
    it at all. So the same arithmetic on the same tree produced one answer
    from this kernel and a different one, by one unit in the last place,
    from either range kernel, and which one a fit got depended on nothing
    but whether it was bagged.

    The multiply now happens on the host, exactly as
    `_range_table_add_raw_kernel` documents at length: `Float32(lr) *
    Float32(value)` there is the same IEEE 754 single-precision multiply the
    per-leaf kernel performs, with the same rounding, so the step that
    crosses is the same bits. This kernel contains no multiply and there is
    nothing left for any compiler to fuse, on any backend, at any release.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var node = leaf_ids[unsafe_offset=r][0]
    if node < 0 or node >= n_nodes:
        return
    var i = r * Int(n_classes) + Int(k)
    raw[unsafe_offset=i] = (
        raw[unsafe_offset=i][0] + steps[unsafe_offset = Int(node)][0]
    )


def _range_add_raw_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    raw: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    node: Int32,
    n_classes: Int32,
    k: Int32,
    learning_rate: Float32,
):
    """`_update_raw_kernel` for one leaf of a compacted tree: every row in
    the leaf's contiguous slice of the active-row permutation advances by
    `learning_rate * value[node]`.

    Under active-row compaction (gpu_active_rows.mojo) there is no per-row
    leaf-assignment array to look a value up in; instead each live leaf owns
    a range of the permutation, so the update is one launch per leaf over
    exactly that leaf's rows. A row outside every range (out of bag) is
    never touched, the same contract `_update_raw_kernel` has for unrouted
    rows.
    """
    var j = global_idx.x
    if j < Int(count):
        var r = Int(rows[unsafe_offset = Int(begin) + j][0])
        var i = r * Int(n_classes) + Int(k)
        raw[unsafe_offset=i] = (
            raw[unsafe_offset=i][0]
            + learning_rate * values[unsafe_offset = Int(node)][0]
        )


# One device-side range descriptor, in Int32 words. `SEG_START` is where this
# segment's rows begin in the flattened thread index space (the running sum of
# the preceding segments' counts), `SEG_BEGIN` is where they begin in the
# active-row permutation, and `SEG_STEP` is the amount every row in the
# segment adds to its raw score: `learning_rate * value[node]`, multiplied and
# rounded on the host, carried here as the Float32's own 32 bits reinterpreted
# as an Int32. The fourth word is unused and exists only to keep the stride a
# power of two, so a descriptor never straddles a cache line.
#
# Why the step lives in the descriptor and not in a node-indexed plane of its
# own
# ---------------------------------------------------------------------------
# It used to. The host staged a Float32 step per node into `step_dev` and an
# Int32 descriptor per live leaf into `seg_dev`, and the kernel looked the
# step up by the node id the descriptor carried in this word. That is two
# device buffers and therefore two `enqueue_copy` calls per tree, and section
# 6.1 of `docs/GPU_PORTABILITY.md` establishes **by measurement** (disassembly
# of the shipped Metal runtime) that an `enqueue_copy` on that backend is a
# synchronous full-queue drain in both directions rather than a byte movement
# whose behavior scales with the count. So the second buffer was a second
# drain to carry a few hundred bytes.
#
# A second drain is not a second wait. Section 6.1.1, withdrawn 2026-08-16,
# took back the step that priced a drain at the per-synchronization constant:
# that constant is **derived** and it is the price of a round trip, and
# neither of these uploads is one. The nearest **measured** point is thirteen
# copies per tree removed on the device-resident plane for 0.016 seconds at
# 1,000,000 x 50, a null under M0. So what the second buffer was buying is a
# second ordering point and a second staging lifetime, which is reason enough
# to be rid of it and is not a time.
#
# Every live leaf has exactly one segment and exactly one node, so the step is
# a function of the segment and belongs in the segment record. Moving it here
# costs nothing in bytes (it takes a word that was padding), removes the whole
# `step_dev` plane from this arm, removes the kernel's second indirect load,
# and leaves the node id with no remaining reader, which is why this word is
# now the step instead of the node.
#
# The mixed types are the only real obstacle, and they are not a conversion.
# A Float32 and an Int32 are both 32 bits, so the step travels as its own bit
# pattern and comes back out of it unchanged; `bitcast` is a reinterpretation
# and no value is altered in either direction. `update_raw_ranges` states the
# equality argument in full.
comptime SEG_WORDS = 4
comptime SEG_START = 0
comptime SEG_BEGIN = 1
comptime SEG_STEP = 2


def _range_table_add_raw_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    raw: MutPointer[Float32, MutAnyOrigin],
    segs: MutPointer[Int32, MutAnyOrigin],
    n_segments: Int32,
    total: Int32,
    n_classes: Int32,
    k: Int32,
):
    """`_range_add_raw_kernel` for every live leaf of a compacted tree at
    once, from a device-resident table of that tree's ranges.

    The per-leaf kernel above needs one launch per leaf because the leaf's
    `begin`, `count`, and node value arrive as launch arguments. Here the
    same numbers arrive as a table the host staged once, so a thread can
    find its own leaf and the whole tree closes in one launch. At the
    31-leaf tree the trainer grows by default that is 1 launch where there
    were 31.

    Thread `t` covers the `t`-th row of the concatenation of the live
    ranges, in the order the host staged them. It finds its segment by
    binary search for the last descriptor whose `SEG_START` is at or below
    `t`, which is exact because the host writes `SEG_START` as the running
    sum of the counts and therefore strictly ascending (empty ranges are
    never staged). With at most a few hundred live leaves that is at most
    nine iterations over a table of a few kilobytes, and every thread of a
    threadgroup that lies inside one leaf, which is all but at most one per
    leaf, searches to the same descriptor.

    Why this takes a step and not a value and a learning rate
    ---------------------------------------------------------
    The per-leaf kernel computes `raw[i] + learning_rate * value[node]`.
    Writing the same expression here produced a different last bit, and the
    reason is instructive: in the per-leaf kernel `node` is a launch
    argument, so `learning_rate * value[node]` is uniform across the launch
    and is computed and rounded to Float32 on its own; here the leaf a
    thread lands on is per-thread, so the product varied across the launch
    and the device compiler contracted the multiply and the add into a
    single fused multiply-add, which rounds once instead of twice.
    Both are legitimate Float32 evaluations of the same expression and they
    differ by one unit in the last place, which is enough to make a model
    not byte-identical to the one this lane started from.

    Rather than fight the contraction, the multiply is moved to the host,
    where `Float32(learning_rate) * Float32(value)` is the same IEEE 754
    single-precision multiply the per-leaf kernel performs, with the same
    rounding, and is therefore the same bits. What reaches the device is
    already the step, so the kernel contains no multiply at all and there is
    nothing left for a compiler to fuse. The result is equal to the per-leaf
    kernel's by construction rather than by the optimizer's agreement,
    which is the stronger of the two guarantees.

    Where the step comes from, and why it is an Int32 here
    ------------------------------------------------------
    The step used to arrive in a Float32 plane of its own, indexed by a node
    id this descriptor carried. It now arrives in the descriptor, as the
    Float32's own bit pattern reinterpreted as an Int32 and reinterpreted
    back by the `bitcast` below. The reason is transfer count, not
    arithmetic: two device buffers meant two `enqueue_copy` calls per tree,
    and on Metal a copy is a queue drain rather than a byte movement
    (`docs/GPU_PORTABILITY.md` section 6.1, **measured** by disassembly). The
    layout comment above `SEG_WORDS` argues the choice in full.

    A `bitcast` between two 32-bit types is a reinterpretation, so the value
    that comes out is the value that went in, to the bit. Nothing about the
    arithmetic moved: the host still computes `Float32(lr) * Float32(value)`,
    in that form and in that place, for the contraction reason above, and
    this kernel still contains no multiply.

    Ranges are pairwise disjoint (`LeafRangeTable` checks that invariant),
    so no row is written by two threads and no row is written twice; a row
    in no range keeps its old score, which is the out-of-bag contract both
    kernels have.
    """
    var t = Int(global_idx.x)
    if t >= Int(total):
        return
    var lo = 0
    var hi = Int(n_segments) - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(segs[unsafe_offset = mid * SEG_WORDS + SEG_START][0]) <= t:
            lo = mid
        else:
            hi = mid - 1
    var base = lo * SEG_WORDS
    var j = t - Int(segs[unsafe_offset = base + SEG_START][0])
    var slot = Int(segs[unsafe_offset = base + SEG_BEGIN][0]) + j
    var step = bitcast[DType.float32, 1](
        segs[unsafe_offset = base + SEG_STEP][0]
    )
    var r = Int(rows[unsafe_offset=slot][0])
    var i = r * Int(n_classes) + Int(k)
    raw[unsafe_offset=i] = raw[unsafe_offset=i][0] + step


def _abs_sum_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """Per-threadgroup magnitude sums of the gradients and hessians, laid out
    `[grad partials | hess partials]`, one entry per threadgroup.

    Both planes reduce in one pass over the rows, so the round's scale costs
    one kernel and one 2 KB readback rather than two passes and two
    downloads. The grid stride, the block count, and the shared-memory tree
    reduction are all fixed, so the partials and their host-side total are
    bit-identical run to run, which is what keeps the fixed-point scale (and
    therefore every histogram derived from it) deterministic.
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
struct GradMagnitudes(Copyable, Movable):
    """The two magnitude sums a round's fixed-point scales come from."""

    var grad: Float64
    var hess: Float64


def enqueue_abs_sum(
    ctx: DeviceContext,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut hess_dev: DeviceBuffer[DType.float32],
    mut part_dev: DeviceBuffer[DType.float32],
    n_rows: Int,
) raises:
    """Enqueue `_abs_sum_kernel` over `n_rows` rows into a caller-owned
    partial buffer of `2 * SUM_BLOCKS` Float32. Does not copy and does not
    synchronize.

    Split out of `magnitude_sums` so that the staged round driver
    (`MagnitudeReader` in gpu_fused_round.mojo) can enqueue the reduction,
    enqueue work that does not depend on the scale, and only then wait. It
    is the same launch with the same fixed grid and block counts, so both
    callers produce bit-identical partials.
    """
    ctx.enqueue_function[_abs_sum_kernel](
        grad_dev.unsafe_ptr(),
        hess_dev.unsafe_ptr(),
        part_dev.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=SUM_BLOCKS,
        block_dim=SUM_THREADS,
    )


def sum_abs_partials[partials_origin: MutOrigin, //](
    partials: MutPointer[Float32, partials_origin],
) raises -> GradMagnitudes:
    """Fold a downloaded partial buffer into the two totals.

    Ascending block index, gradient plane then hessian plane, accumulated in
    Float64. Every caller sums in this one order over partials the one
    kernel produced, so the totals, the scales derived from them, and every
    histogram quantized with those scales agree bit for bit whichever driver
    enqueued the reduction.
    """
    var g_total = 0.0
    var h_total = 0.0
    for i in range(SUM_BLOCKS):
        g_total += Float64(partials.unsafe_load(i))
        h_total += Float64(partials.unsafe_load(SUM_BLOCKS + i))
    if not isfinite(g_total) or not isfinite(h_total):
        raise Error("gradients and hessians must be finite")
    return GradMagnitudes(g_total, h_total)


struct GpuObjectiveState(Movable):
    """Device-resident labels, weights, and raw scores for one training run.

    Construct once per training session, alongside the `GpuHistogramBuilder`
    and from the same `DeviceContext`; call `init_raw` once, then per round
    `fill_grad_hess` (or `refresh_softmax` + `fill_softmax_grad_hess`) and
    `magnitude_sums`, and per tree `update_raw`. Nothing per-row is uploaded
    after construction.

    The context is passed to each method rather than held, so these buffers
    and the histogram builder's can be driven by the one context that owns
    them both.
    """

    var target_dev: DeviceBuffer[DType.float32]
    """The regression target, or the integer class label for softmax."""
    var weight_dev: DeviceBuffer[DType.float32]
    """Sample weights, or a one-element placeholder when unweighted:
    zero-length device buffers are not portable."""
    var raw_dev: DeviceBuffer[DType.float32]
    """Row-major raw scores, `raw[r * n_classes + k]`."""
    var prob_dev: DeviceBuffer[DType.float32]
    """Softmax probabilities in the same layout, or a placeholder when
    `n_classes == 1`."""
    var value_dev: DeviceBuffer[DType.float32]
    """The current tree's node values, the lookup table the per-leaf range
    kernel reads. It is the only kernel that still applies the learning rate
    itself, and it may because its `node` is a launch argument; every other
    update arm reads `step_dev` instead."""
    var seg_dev: DeviceBuffer[DType.int32]
    """The current tree's live-range descriptors, `SEG_WORDS` Int32 apiece,
    which is what lets `update_raw_ranges` close a whole tree in one launch
    and, since the descriptor carries its own step, in one copy. Sized once
    at construction from `max_nodes`, never per tree."""
    var part_dev: DeviceBuffer[DType.float32]
    var base_dev: DeviceBuffer[DType.float32]
    var host_part: HostBuffer[DType.float32]
    var host_raw: HostBuffer[DType.float32]
    var step_dev: DeviceBuffer[DType.float32]
    """The current tree's per-node steps, `learning_rate * value[node]`
    already multiplied and rounded on the host, which is why
    `_update_raw_kernel` contains no multiply; see that kernel for the
    rounding argument.

    Read by `_update_raw_kernel` only. The range-table arm used to read it
    too and now carries its step inside the range descriptor instead, which
    is what took that arm from two copies per tree to one; the node-indexed
    plane survives here because `_update_raw_kernel`'s node ids arrive
    per-row out of a leaf-assignment array and there is no descriptor to
    hang a step on."""
    var stage_value: HostBuffer[DType.float32]
    """Pinned staging for the node-value table. `map_to_host` copies in both
    directions on every use and blocks (the reasoning is written out in
    histogram_gpu.mojo), so the per-tree upload goes through an ordinary
    one-way copy out of this buffer instead.

    One-way, not asynchronous. On Metal `enqueue_copy` is a synchronous
    full-queue drain in both directions, **measured** by disassembly and
    recorded in `docs/GPU_PORTABILITY.md` section 6.1, so what this buys over
    the mapping is the second direction's bytes and not the drain. The drain
    is still there, once per tree, and under section 6.1.1 it is an ordering
    point rather than a time: nothing is queued behind it and no host decision
    reads a device answer through it."""
    var stage_seg: HostBuffer[DType.int32]
    """Pinned staging for `seg_dev`, on the same grounds. Since the step
    moved into the descriptor this is the whole of what
    `update_raw_ranges` sends, so the arm stages once and copies once.

    There is no `stage_step` beside it: `update_raw` is the only remaining
    reader of `step_dev` and it still writes through a `map_to_host`
    mapping, which is a per-tree bidirectional transfer this lane counted
    and deliberately did not touch."""
    var part_pending: Int
    """How many magnitude reductions have been enqueued into `host_part` and
    not yet folded.

    Zero on every path that calls `magnitude_sums`, which enqueues and reads
    in one call. Nonzero only between `enqueue_magnitudes` and
    `read_magnitudes`, which is the pair that lets a caller amortize the
    round's one device-to-host wait over several rounds. It is also the slot
    index the next reduction lands in, which is why one counter serves as
    both: slot `p` is written by the `p`-th enqueue since the last read, and
    `read_magnitudes` folds slots `0 .. p-1` in that order and resets it.

    Never larger than `SCALE_WINDOW_MAX`; `enqueue_magnitudes` refuses rather
    than wrapping, because wrapping would silently fold one round's partials
    as another round's and every scale after it would be wrong in a way no
    assertion would catch."""

    var n_rows: Int
    var n_classes: Int
    var max_nodes: Int
    var weighted: Bool
    var block_threads: Int
    var has_raw: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        target: List[Float64],
        sample_weight: List[Float64] = [],
        n_classes: Int = 1,
        max_nodes: Int = DEFAULT_MAX_NODES,
    ) raises:
        """Upload the labels and weights, which never change again, and
        allocate the raw-score and scratch buffers.

        For softmax, `target` holds the class labels as whole numbers and
        `n_classes` is the class count; for every other objective
        `n_classes` is 1 and `target` is the regression target.
        """
        if len(target) < 1:
            raise Error("device objectives require at least one row")
        if n_classes < 1:
            raise Error("n_classes must be positive")
        if max_nodes < 1:
            raise Error("max_nodes must be positive")
        if len(sample_weight) > 0 and len(sample_weight) != len(target):
            raise Error("sample_weight length must equal the target length")
        for r in range(len(target)):
            if not isfinite(target[r]):
                raise Error("target must be finite")
        for r in range(len(sample_weight)):
            if not isfinite(sample_weight[r]) or sample_weight[r] < 0.0:
                raise Error("sample_weight must be finite and nonnegative")
        if n_classes > 1:
            for r in range(len(target)):
                var label = Int(target[r])
                if Float64(label) != target[r] or label < 0 or (
                    label >= n_classes
                ):
                    raise Error(
                        "multiclass target must hold whole class labels in"
                        " 0..n_classes-1"
                    )

        self.n_rows = len(target)
        self.n_classes = n_classes
        self.max_nodes = max_nodes
        self.weighted = len(sample_weight) > 0
        self.has_raw = False
        self.block_threads = derive_block_threads(query_device_caps(ctx))

        var n_scores = self.n_rows * n_classes
        self.target_dev = ctx.enqueue_create_buffer[DType.float32](self.n_rows)
        self.weight_dev = ctx.enqueue_create_buffer[DType.float32](
            self.n_rows if self.weighted else 1
        )
        self.raw_dev = ctx.enqueue_create_buffer[DType.float32](n_scores)
        self.prob_dev = ctx.enqueue_create_buffer[DType.float32](
            n_scores if n_classes > 1 else 1
        )
        self.value_dev = ctx.enqueue_create_buffer[DType.float32](max_nodes)
        self.seg_dev = ctx.enqueue_create_buffer[DType.int32](
            max_nodes * SEG_WORDS
        )
        self.step_dev = ctx.enqueue_create_buffer[DType.float32](max_nodes)
        self.base_dev = ctx.enqueue_create_buffer[DType.float32](n_classes)
        self.part_dev = ctx.enqueue_create_buffer[DType.float32](
            2 * SUM_BLOCKS
        )
        # `SCALE_WINDOW_MAX` slots rather than one, allocated unconditionally
        # because the window is a runtime arm the builder chooses after this
        # state is constructed and 64 KB of pinned memory is not worth a
        # second allocation path. The device partial buffer stays ONE slot:
        # the queue is in order, so kernel(j) writes it, copy(j) drains it
        # into slot j, and only then does kernel(j+1) overwrite it.
        self.host_part = ctx.enqueue_create_host_buffer[DType.float32](
            SCALE_WINDOW_MAX * 2 * SUM_BLOCKS
        )
        self.part_pending = 0
        self.host_raw = ctx.enqueue_create_host_buffer[DType.float32](n_scores)
        self.stage_value = ctx.enqueue_create_host_buffer[DType.float32](
            max_nodes
        )
        self.stage_seg = ctx.enqueue_create_host_buffer[DType.int32](
            max_nodes * SEG_WORDS
        )

        # One-time uploads. Both buffers are written through the mapping
        # rather than staged, because this runs once per session and the
        # mapping is the shorter path; the per-round transfers below are the
        # ones that had to be cheap.
        with self.target_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(self.n_rows):
                dst.unsafe_store(r, Float32(target[r]))
        if self.weighted:
            with self.weight_dev.map_to_host() as host:
                var dst = host.unsafe_ptr()
                for r in range(self.n_rows):
                    dst.unsafe_store(r, Float32(sample_weight[r]))

    def _row_blocks(self) -> Int:
        return (self.n_rows + self.block_threads - 1) // self.block_threads

    def init_raw(
        mut self, ctx: DeviceContext, base_scores: List[Float64]
    ) raises:
        """Set every row's raw score to its class's base score. Once per
        training run, before the first round."""
        if len(base_scores) != self.n_classes:
            raise Error("base_scores length must equal n_classes")
        for k in range(len(base_scores)):
            if not isfinite(base_scores[k]):
                raise Error("base scores must be finite")
        with self.base_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for k in range(self.n_classes):
                dst.unsafe_store(k, Float32(base_scores[k]))
        ctx.enqueue_function[_init_raw_kernel](
            self.raw_dev.unsafe_ptr(),
            self.base_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_classes),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )
        self.has_raw = True

    def set_raw(mut self, ctx: DeviceContext, raw: List[Float64]) raises:
        """Overwrite the device-resident raw scores with arbitrary per-row
        values, row-major over classes.

        `init_raw` covers the ordinary start of a run; this covers the two
        cases that do not start flat: continued training, whose starting
        scores are an existing ensemble's predictions, and an explicit
        per-row init score. It is also how a test can put the device at a
        chosen point of the loss surface without training to get there.
        """
        var n = self.n_rows * self.n_classes
        if len(raw) != n:
            raise Error("raw length must equal n_rows * n_classes")
        for i in range(n):
            if not isfinite(raw[i]):
                raise Error("raw scores must be finite")
        # Any enqueued update still writing the raw scores has to finish
        # before the host overwrites them.
        ctx.synchronize()
        with self.raw_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(n):
                dst.unsafe_store(i, Float32(raw[i]))
        self.has_raw = True

    def fill_grad_hess(
        mut self,
        ctx: DeviceContext,
        objective: Int,
        alpha: Float64,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Write this round's gradients and hessians for a single-output
        objective into `grad_dev` and `hess_dev`, which must be device
        buffers of at least `n_rows` Float32 belonging to the same context.

        In the trainer those are the histogram builder's own gradient and
        hessian buffers, so the values the histogram kernels read are the
        ones this kernel wrote and nothing crosses to the host in between.
        """
        if objective == CUSTOM:
            raise Error(
                "custom objectives have no device kernel; keep them on the"
                " host path (train_custom_gpu)"
            )
        if not supports_device_objective(objective):
            raise Error("unknown objective code ", objective)
        if self.n_classes != 1:
            raise Error(
                "multiclass state: use refresh_softmax and"
                " fill_softmax_grad_hess"
            )
        if not self.has_raw:
            raise Error("call init_raw before filling gradients")
        if not isfinite(alpha):
            raise Error("alpha must be finite")
        ctx.enqueue_function[_grad_hess_kernel](
            self.raw_dev.unsafe_ptr(),
            self.target_dev.unsafe_ptr(),
            self.weight_dev.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(objective),
            Float32(alpha),
            Int32(1) if self.weighted else Int32(0),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )

    def refresh_softmax(mut self, ctx: DeviceContext) raises:
        """Recompute the softmax probabilities from the current raw scores.
        Once per multiclass round, before the per-class gradient calls, which
        is where the host trainer computes them too."""
        if self.n_classes < 2:
            raise Error("refresh_softmax requires n_classes >= 2")
        if not self.has_raw:
            raise Error("call init_raw before computing probabilities")
        ctx.enqueue_function[_softmax_prob_kernel](
            self.raw_dev.unsafe_ptr(),
            self.prob_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_classes),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )

    def fill_softmax_grad_hess(
        mut self,
        ctx: DeviceContext,
        k: Int,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Write class `k`'s one-vs-rest gradients and hessians. Call
        `refresh_softmax` first; the probabilities are shared by every class
        of the round, exactly as on the host."""
        if self.n_classes < 2:
            raise Error("fill_softmax_grad_hess requires n_classes >= 2")
        if k < 0 or k >= self.n_classes:
            raise Error("class index out of range")
        if not self.has_raw:
            raise Error("call init_raw before filling gradients")
        ctx.enqueue_function[_softmax_class_kernel](
            self.prob_dev.unsafe_ptr(),
            self.target_dev.unsafe_ptr(),
            self.weight_dev.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_classes),
            Int32(k),
            Int32(1) if self.weighted else Int32(0),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )

    def update_raw(
        mut self,
        ctx: DeviceContext,
        mut leaf_dev: DeviceBuffer[DType.int32],
        values: List[Float64],
        learning_rate: Float64,
        k: Int = 0,
    ) raises:
        """Advance the device-resident raw scores by one grown tree.

        `leaf_dev` is the leaf-assignment array tree growth left behind (the
        histogram builder's), and `values` is the tree's node-value array,
        already renewed and not yet shrunk: this method applies
        `learning_rate`, so it consumes exactly `tree.value` and the
        booster's learning rate.

        The shrinkage is applied here on the host rather than in the kernel,
        which is what makes this arm's answer equal to the two range arms'
        rather than one unit in the last place away from it. `Float32(lr) *
        Float32(value)` is the same single-precision multiply the per-leaf
        range kernel performs on the device, so the step that crosses is the
        same bits, and the kernel then does nothing but add it.
        `_update_raw_kernel` and `_range_table_add_raw_kernel` both give the
        argument in full.

        Only rows the device routed are updated. With no row bagging that is
        every row and the update is complete. Under bagging the out-of-bag
        rows sit at `OUT_OF_BAG` and keep their old scores, so the caller has
        to either replay the tree's splits over a full leaf-assignment reset
        first or score those rows on the host; the handoff spells both out.
        """
        if len(values) < 1:
            raise Error("node values must not be empty")
        if len(values) > self.max_nodes:
            raise Error(
                "tree has more nodes than the node-value table holds;"
                " construct with a larger max_nodes"
            )
        if k < 0 or k >= self.n_classes:
            raise Error("class index out of range")
        if not self.has_raw:
            raise Error("call init_raw before updating raw scores")
        if not isfinite(learning_rate):
            raise Error("learning_rate must be finite")
        for i in range(len(values)):
            if not isfinite(values[i]):
                raise Error("node values must be finite")
        var lr32 = Float32(learning_rate)
        with self.step_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(values)):
                dst.unsafe_store(i, lr32 * Float32(values[i]))
        ctx.enqueue_function[_update_raw_kernel](
            self.raw_dev.unsafe_ptr(),
            leaf_dev.unsafe_ptr(),
            self.step_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_classes),
            Int32(k),
            Int32(len(values)),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )

    def _check_range_update(
        self, values: List[Float64], learning_rate: Float64, k: Int
    ) raises:
        """The preconditions both range-update arms share, so the two
        cannot drift apart on what they refuse."""
        if len(values) < 1:
            raise Error("node values must not be empty")
        if len(values) > self.max_nodes:
            raise Error(
                "tree has more nodes than the node-value table holds;"
                " construct with a larger max_nodes"
            )
        if k < 0 or k >= self.n_classes:
            raise Error("class index out of range")
        if not self.has_raw:
            raise Error("call init_raw before updating raw scores")
        if not isfinite(learning_rate):
            raise Error("learning_rate must be finite")
        for i in range(len(values)):
            if not isfinite(values[i]):
                raise Error("node values must be finite")

    def _stage_values(mut self, ctx: DeviceContext, values: List[Float64]
    ) raises:
        """Upload the tree's node values through pinned staging.

        This replaces a `map_to_host` on `value_dev`. The two are not the
        same transfer: a mapping is bidirectional, so it moved the buffer
        both ways every time it was opened. A staged copy is one-way, which
        is the convention histogram_gpu.mojo documents and the split
        searcher already follows for its per-node tables.

        One-way is the whole difference on Metal, and the earlier version of
        this docstring claimed more. It said the staged copy was
        asynchronous where the mapping blocked. It is not: on Metal
        `enqueue_copy` is a synchronous full-queue drain in both directions,
        **measured** by disassembly of the shipped runtime and recorded in
        `docs/GPU_PORTABILITY.md` section 6.1. So this call still drains once
        per tree, exactly as the mapping did, and a round's **hazard** budget
        has to carry it.

        Its **time** budget does not, and section 6.1.1 is why. A drain of a
        queue holding nothing costs nothing; this upload blocks on no device
        answer and nothing enqueued is waiting behind it, so it is an ordering
        point and not a round trip. The two counts are separate and only the
        round-trip count predicts seconds.

        The staging contract is the usual one and is kept for the backend
        where the copy really is asynchronous: the pinned buffer must not be
        rewritten while a copy out of it is in flight. It is rewritten once
        per tree, and the next tree's growth blocks on the device well
        before it reaches this point (the round's magnitude reduction alone
        synchronizes), so the copy has long retired.

        Words past `len(values)` are whatever the previous tree left, which
        is exactly what the mapping left there too: no kernel reads a node
        id at or beyond the current tree's node count.
        """
        var dst = self.stage_value.unsafe_ptr()
        for i in range(len(values)):
            dst.unsafe_store(i, Float32(values[i]))
        ctx.enqueue_copy(dst_buf=self.value_dev, src_ptr=dst)

    def update_raw_ranges(
        mut self,
        ctx: DeviceContext,
        mut rows: GpuActiveRows,
        values: List[Float64],
        learning_rate: Float64,
        k: Int = 0,
    ) raises:
        """`update_raw` for a compacted tree: advance the raw scores from
        the leaf ranges the grown tree left in `rows` instead of a per-row
        leaf-assignment array.

        Call after `grow_tree_gpu` returns and before the next tree's
        `begin_tree`, which is what resets the ranges. Every live range
        belongs to a leaf of the finished tree; rows outside every range
        (out of bag) keep their old scores, the same contract `update_raw`
        has for unrouted rows.

        One launch, not one per leaf
        ----------------------------
        This used to open a `map_to_host` mapping on the node-value buffer
        and then issue one small launch per live leaf. On the default
        31-leaf tree that was 31 launches and one hidden host
        synchronization per tree; it is now one launch and no
        synchronization of its own. The descriptors are read by
        `_range_table_add_raw_kernel`, which finds a thread's leaf by binary
        search over them.

        One copy, not two
        -----------------
        The launch count was already right and the transfer count was not.
        Between the rewrite above and this one the method issued two
        `enqueue_copy` calls per tree, one for a Float32 plane of per-node
        steps and one for the Int32 range descriptors, each a few hundred
        bytes. Section 6.1 of `docs/GPU_PORTABILITY.md` establishes
        **by measurement** (disassembly of the shipped Metal runtime, plus a
        second measurement from outside the process in
        `docs/METAL_TIMELINE.md`) that on that backend an `enqueue_copy`
        drains the whole queue and then memcpys, in both directions, and that
        the drain is very nearly independent of the byte count. Two buffers
        therefore meant two drains to move a few hundred bytes. Two drains and
        not two waits: section 6.1.1, withdrawn 2026-08-16, is explicit that a
        copy count predicts portability risk and ordering hazards while a
        round-trip count predicts time, and neither of these is a round trip.
        The paragraph below already declined to convert this into seconds, and
        that was the right call.

        The step is now carried inside the range descriptor, in the word
        that was padding, as the Float32's own bits reinterpreted as an
        Int32; the layout comment above `SEG_WORDS` argues why that is the
        right home for it and why the reinterpretation is free. So this arm
        stages one buffer and copies once. **Counted in source**, that is two
        `enqueue_copy` calls per tree before and one after.

        What that saves in seconds is **not measured here and is not
        estimated here**. It cannot be: section 6.4 of the same document
        records a live factor-of-five tension between the wait count derived
        from the source and the wait count a Metal System Trace actually
        observed, and concludes that the wait count of any path must be
        measured rather than derived. One of the candidate resolutions there
        is that a drain on an already-empty queue is cheap. This lane
        removes a copy that source says is there; how much wall clock it was
        worth is the coordinator's to measure.

        What did not change is the arithmetic
        -------------------------------------
        Every row still receives `learning_rate * value[leaf]` added to its
        own Float32 raw score, and the multiply is still
        `Float32(learning_rate) * Float32(values[node])` evaluated on the
        host, in that expression and in that place. It has to be: doing it
        in the kernel let the device compiler contract it into an FMA and
        produced a one-bit divergence from the per-leaf arm, which is the
        whole reason the multiply is here at all
        (`_range_table_add_raw_kernel` gives the argument in full). This
        change moved which buffer the resulting bits travel in and nothing
        else. A `bitcast` between two 32-bit types is a reinterpretation,
        not a conversion, so the Float32 the kernel adds is the Float32 the
        host computed, bit for bit.

        The ranges are disjoint, so each row is still written exactly once
        by exactly one thread. `update_raw_ranges_per_leaf` keeps the old
        launch shape and the old node-value lookup, so a test can assert the
        agreement rather than a docstring asserting it; it is deliberately
        not packed, which is what makes it an independent arm rather than a
        second spelling of this one. `tests/test_gpu_fma_consistency.mojo`
        and `tests/test_gpu_split_launch_overhead.mojo` both run the two
        against the same state and compare the raw scores bit for bit.
        """
        self._check_range_update(values, learning_rate, k)
        # Flatten the live ranges into ascending descriptors, each carrying
        # its own step. Empty ranges are the tree's internal nodes, whose
        # rows their children own; the per-leaf loop skipped them the same
        # way, and a node with no live rows contributes no step because
        # nothing would read it.
        #
        # `Float32(lr) * Float32(value)` here is the same IEEE 754 single
        # multiply the per-leaf kernel performs on the device, so the step
        # that crosses is bit-identical to the one that kernel would have
        # computed; the kernel then does nothing but add it. See
        # `_range_table_add_raw_kernel` for why the multiply had to leave
        # the kernel at all, and note that it has not moved since: only the
        # buffer it is stored into has.
        var dst = self.stage_seg.unsafe_ptr()
        var lr32 = Float32(learning_rate)
        var n_segments = 0
        var total = 0
        for node in range(rows.ranges.n_nodes()):
            if node >= len(values):
                break
            var window = rows.ranges.get(node)
            var n = window.count()
            if n <= 0:
                continue
            var step = lr32 * Float32(values[node])
            var base = n_segments * SEG_WORDS
            dst.unsafe_store(base + SEG_START, Int32(total))
            dst.unsafe_store(base + SEG_BEGIN, Int32(window.begin))
            dst.unsafe_store(
                base + SEG_STEP, bitcast[DType.int32, 1](step)
            )
            dst.unsafe_store(base + SEG_WORDS - 1, Int32(0))
            n_segments += 1
            total += n
        if n_segments == 0:
            return
        ctx.enqueue_copy(dst_buf=self.seg_dev, src_ptr=dst)
        var blocks = (
            total + self.block_threads - 1
        ) // self.block_threads
        ctx.enqueue_function[_range_table_add_raw_kernel](
            rows.rows_dev.unsafe_ptr(),
            self.raw_dev.unsafe_ptr(),
            self.seg_dev.unsafe_ptr(),
            Int32(n_segments),
            Int32(total),
            Int32(self.n_classes),
            Int32(k),
            grid_dim=blocks,
            block_dim=self.block_threads,
        )

    def update_raw_ranges_per_leaf(
        mut self,
        ctx: DeviceContext,
        mut rows: GpuActiveRows,
        values: List[Float64],
        learning_rate: Float64,
        k: Int = 0,
    ) raises:
        """The launch-per-leaf range update, kept as the reference arm.

        This is what `update_raw_ranges` issued before the range table
        existed, minus the `map_to_host` on the node-value buffer, which the
        staged copy replaces here as well so that the only difference
        between the two arms is the launch shape. Nothing in the trainer
        calls it; it exists so a test can run both over the same state and
        compare the resulting raw scores bit for bit, which is a stronger
        statement about the rewrite than any argument about it.
        """
        self._check_range_update(values, learning_rate, k)
        self._stage_values(ctx, values)
        for node in range(rows.ranges.n_nodes()):
            if node >= len(values):
                break
            var window = rows.ranges.get(node)
            var n = window.count()
            if n <= 0:
                continue
            var blocks = (
                n + self.block_threads - 1
            ) // self.block_threads
            ctx.enqueue_function[_range_add_raw_kernel](
                rows.rows_dev.unsafe_ptr(),
                self.raw_dev.unsafe_ptr(),
                self.value_dev.unsafe_ptr(),
                Int32(window.begin),
                Int32(n),
                Int32(node),
                Int32(self.n_classes),
                Int32(k),
                Float32(learning_rate),
                grid_dim=blocks,
                block_dim=self.block_threads,
            )

    def magnitude_sums(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises -> GradMagnitudes:
        """Sum `|grad|` and `|hess|` over the rows on the device and return
        both totals.

        This is what replaces the host pass `_fixed_scale` makes over the
        gradient lists. The threadgroup partials come back to the host (2 KB,
        independent of `n_rows`) and are summed in Float64 there, so the
        total is more accurate than a Float32 device-side final reduction
        would be, and the readback is the round's only device-to-host
        transfer.

        The kernel launch and the host-side fold are `enqueue_abs_sum` and
        `sum_abs_partials`, so a caller that needs the reduction split
        across a wait (`MagnitudeReader`) runs the same two halves this
        method runs back to back rather than a second copy of them.

        **This is the unwindowed arm, kept expression for expression.** The
        windowed pair below (`enqueue_magnitudes` / `read_magnitudes`) is
        what a caller amortizing the wait uses, and at a window of one it is
        this call split in half: same launch, same copy, same synchronize,
        same fold, in that order. Keeping both reachable is what lets
        `tests/test_gpu_scale_refresh.mojo` compare them at the tree level
        rather than assert their equivalence in a docstring, which is the
        only way an off-by-one in the slot arithmetic would ever be caught:
        folding the wrong slot produces a *plausible* scale, not an error.
        """
        if self.part_pending != 0:
            raise Error(
                "a windowed magnitude reduction has not been read;"
                " magnitude_sums cannot share the readback buffer with it"
            )
        enqueue_abs_sum(
            ctx, grad_dev, hess_dev, self.part_dev, self.n_rows
        )
        ctx.enqueue_copy(
            dst_ptr=self.host_part.unsafe_ptr(), src_buf=self.part_dev
        )
        # Load-bearing. The destination is a pinned `HostBuffer`, and on
        # Metal a copy into pinned memory is asynchronous (**measured** by
        # execution, `gpu_tree_tables.download`: 64 of 64 stale words behind
        # a slow kernel, 0 of 64 behind a fast one). Reading without this
        # passes on a small fixture and corrupts a real fit.
        ctx.synchronize()
        return sum_abs_partials(self.host_part.unsafe_ptr())

    def enqueue_magnitudes(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises -> Int:
        """Enqueue this round's magnitude reduction and its readback into the
        next free window slot. **Does not synchronize.** Returns how many
        slots are now pending.

        The half of `magnitude_sums` that costs no wait. The copy is enqueued
        here rather than in `read_magnitudes` so that it sits immediately
        behind its own kernel in the queue: the device partial buffer is one
        slot wide and every round overwrites it, so the copy that carries
        round `j`'s partials out of it has to be ordered between kernel `j`
        and kernel `j+1`. An in-order queue gives that; nothing else is
        needed, and nothing else would be enough.

        Each round lands in its own host slot, so a caller may leave up to
        `SCALE_WINDOW_MAX` rounds unread and still recover every round's
        magnitudes exactly. That is what makes the amortized arm *verifiable*
        rather than merely cheaper: the rounds that reused a stale scale are
        not rounds whose magnitudes were never measured, they are rounds
        whose magnitudes were measured and read late.

        The pinned-write hazard runs the other way here and is worth naming.
        The usual rule is that a pinned buffer must not be rewritten while a
        copy *out of* it is in flight; these are copies *into* it, each into a
        distinct slot, and no slot is read until `read_magnitudes` has
        synchronized. A caller that read a slot before that synchronize would
        get the stale-word failure this file's other waits exist to prevent,
        which is why the read is a method and the buffer is private to it.
        """
        if self.part_pending >= SCALE_WINDOW_MAX:
            raise Error(
                "the magnitude window is full; read it before enqueuing"
                " another reduction"
            )
        enqueue_abs_sum(
            ctx, grad_dev, hess_dev, self.part_dev, self.n_rows
        )
        ctx.enqueue_copy(
            dst_ptr=self.host_part.unsafe_ptr().unsafe_offset(
                self.part_pending * 2 * SUM_BLOCKS
            ),
            src_buf=self.part_dev,
        )
        self.part_pending += 1
        return self.part_pending

    def read_magnitudes(mut self, ctx: DeviceContext) raises -> List[
        GradMagnitudes
    ]:
        """Fold every pending window slot, oldest first, and empty the window.

        **One synchronize whatever the window holds**, which is the whole
        point: a window of `N` rounds pays the round trip once instead of `N`
        times. The wait is load-bearing for the reason `magnitude_sums` gives
        and for every slot at once, since one drain covers every copy behind
        it in an in-order queue.

        Each slot is folded by `sum_abs_partials` over that slot's own 2 KB,
        ascending block index, gradient plane then hessian plane, in Float64.
        That is the same function, the same order, and the same width the
        unwindowed call uses, so slot `j`'s totals are bit for bit the totals
        `magnitude_sums` would have returned for round `j` had it waited
        there. **The window changes when the host learns a round's
        magnitudes. It does not change what they are.** Everything the
        amortized arm gives up, it gives up in the scale *derivation*, which
        is `GpuHistogramBuilder`'s decision and is argued there.

        Returns oldest first, so element 0 is the oldest unread round and the
        last element is the round that just enqueued. A caller deriving a
        scale for the next window wants the maximum over the list; a caller
        checking that the window it just closed was safe wants the maximum
        over everything but the last. Both are in the caller because both are
        policy, and this method is a transfer.
        """
        if self.part_pending < 1:
            raise Error("no magnitude reductions are pending")
        # See `magnitude_sums`: the destination is pinned, so this is the
        # wait and not a formality.
        ctx.synchronize()
        var out = List[GradMagnitudes](capacity=self.part_pending)
        var base = self.host_part.unsafe_ptr()
        for slot in range(self.part_pending):
            out.append(
                sum_abs_partials(base.unsafe_offset(slot * 2 * SUM_BLOCKS))
            )
        self.part_pending = 0
        return out^

    def magnitudes_pending(self) -> Int:
        """How many enqueued reductions `read_magnitudes` would fold."""
        return self.part_pending

    def download_raw(mut self, ctx: DeviceContext) raises -> List[Float64]:
        """The current raw scores, row-major, as Float64.

        The device path needs this only where the host still owns a decision
        the raw scores feed: quantile and L1 leaf renewal, validation
        metrics, and early stopping. It is a full `n_rows * n_classes`
        transfer, so it is not part of a plain round.
        """
        var n = self.n_rows * self.n_classes
        ctx.enqueue_copy(
            dst_ptr=self.host_raw.unsafe_ptr(), src_buf=self.raw_dev
        )
        ctx.synchronize()
        var src = self.host_raw.unsafe_ptr()
        var out = List[Float64](capacity=n)
        for i in range(n):
            out.append(Float64(src.unsafe_load(i)))
        return out^

    def download_grad_hess(
        mut self,
        ctx: DeviceContext,
        grad_dev: DeviceBuffer[DType.float32],
        hess_dev: DeviceBuffer[DType.float32],
    ) raises -> List[Float64]:
        """The gradients followed by the hessians, `2 * n_rows` Float64.

        Not part of a training round: it exists for the tests, which compare
        the kernels against `fill_grad_hess`, and for GOSS, which ranks rows
        by gradient magnitude on the host and so cannot stay device-side
        without a device-side ranking pass.
        """
        var g = List[Float64](capacity=2 * self.n_rows)
        ctx.enqueue_copy(
            dst_ptr=self.host_raw.unsafe_ptr(), src_buf=grad_dev
        )
        ctx.synchronize()
        var src = self.host_raw.unsafe_ptr()
        for r in range(self.n_rows):
            g.append(Float64(src.unsafe_load(r)))
        ctx.enqueue_copy(
            dst_ptr=self.host_raw.unsafe_ptr(), src_buf=hess_dev
        )
        ctx.synchronize()
        for r in range(self.n_rows):
            g.append(Float64(src.unsafe_load(r)))
        return g^


# ---------------------------------------------------------------------------
# CatBoost's `leaf_estimation_iterations` on the device-resident plane.
# ---------------------------------------------------------------------------
#
# One device-side leaf record, in Int32 words. `EST_START` is where this leaf's
# rows begin in the flattened thread index space (the running sum of the
# preceding leaves' counts), `EST_BEGIN` is where they begin in the active-row
# permutation, and `EST_COUNT` is how many there are. The fourth word is unused
# and keeps the stride a power of two, exactly as `SEG_WORDS` does.
#
# The count is carried here and is not carried in `SEG_WORDS`, because the two
# tables answer different questions. `_range_table_add_raw_kernel` maps a
# *thread* to a leaf, so it needs only the running start and finds its row from
# the difference; `_leaf_newton_kernel` maps a *block* to a leaf and then
# sweeps that leaf's rows, so it needs the leaf's own extent. Reusing `seg_dev`
# would mean either recovering the count by differencing the next descriptor
# (which has no next at the last leaf) or making the two kernels' layouts
# co-vary for no gain.
comptime EST_WORDS = 4
comptime EST_START = 0
comptime EST_BEGIN = 1
comptime EST_COUNT = 2

# The Float32 plane parallel to it: the leaf's current value, and the closed
# monotone interval that value must stay inside. `EST_V` is the only word the
# device writes; the bounds are read-only for the whole tree. Fourth word
# padding again, for the same alignment reason.
comptime EST_VALS = 4
comptime EST_V = 0
comptime EST_LO = 1
comptime EST_HI = 2


def _leaf_shift_raw_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    raw: MutPointer[Float32, MutAnyOrigin],
    shifted: MutPointer[Float32, MutAnyOrigin],
    segs: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    n_segments: Int32,
    total: Int32,
):
    """`shifted[r] = raw[r] + v[leaf(r)]` for every row of every live leaf.

    The score an extra Newton step differentiates at is the score the row
    would sit at if the tree stopped here, which is its current raw score plus
    the value its leaf currently holds. This kernel materializes that point so
    that `_grad_hess_kernel` -- the *same* kernel the round's own derivatives
    come from, argument for argument -- can be run over it unchanged. That
    reuse is the whole reason this arm is three launches per iteration rather
    than one: a fused kernel would have to restate every objective's
    derivative and its weight multiplier, and a second definition of the
    objectives is precisely what `supports_device_objective`'s docstring says
    this module has already paid for once.

    **The shift is `v`, not `learning_rate * v`.** The ensemble adds
    `learning_rate * v` to each row, so evaluating here at `raw[r] + v` asks
    what the best *full* step for this leaf is and lets the raw-score update
    shrink that answer once, at the end. It is the same division of labour the
    host implementation makes and states at length
    (`boosting._estimate_leaf_values`), and it is CatBoost's: its walker
    carries no rate and `NormalizeLeafValues` multiplies the accumulated leaf
    value by the rate once, after the last iteration.

    Thread `t` finds its leaf by the same binary search
    `_range_table_add_raw_kernel` uses, over the same strictly ascending
    `EST_START` column. Ranges are pairwise disjoint, so each row is written
    by exactly one thread. Rows in no range -- out of bag -- are never written
    and keep the zero `GpuLeafEstimator` put there once at construction;
    nothing reads them, and the zero is there so that `_grad_hess_kernel`'s
    full-grid sweep never differentiates uninitialized memory.
    """
    var t = Int(global_idx.x)
    if t >= Int(total):
        return
    var lo = 0
    var hi = Int(n_segments) - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(segs[unsafe_offset = mid * EST_WORDS + EST_START][0]) <= t:
            lo = mid
        else:
            hi = mid - 1
    var base = lo * EST_WORDS
    var j = t - Int(segs[unsafe_offset = base + EST_START][0])
    var slot = Int(segs[unsafe_offset = base + EST_BEGIN][0]) + j
    var r = Int(rows[unsafe_offset=slot][0])
    shifted[unsafe_offset=r] = (
        raw[unsafe_offset=r][0]
        + vals[unsafe_offset = lo * EST_VALS + EST_V][0]
    )


def _leaf_newton_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    segs: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    lambda_l1: Float32,
    lambda_l2: Float32,
    max_delta_step: Float32,
):
    """One threadgroup per live leaf: sum that leaf's gradients and hessians
    and take one more Newton step, in place, without the host seeing either
    number.

    This is the launch that makes the iteration a *device* iteration. The
    dependency an extra Newton step introduces is genuine -- iteration `k`'s
    sums are taken at the scores iteration `k - 1` wrote -- and it is resolved
    here, inside the device queue, rather than by handing the sums to the host
    and taking the value back. A block owns one leaf and no other block reads
    or writes that leaf's record, so the step needs no cross-block
    communication and no second launch: thread 0 of the block writes the new
    value straight back into `EST_V`, where the next iteration's shift kernel
    reads it.

    What is recomputed and what is not
    ----------------------------------
    **The histogram is not rebuilt and the tree's structure does not move.**
    Nothing here reads a bin, a threshold, or a feature. What moves between
    iterations is exactly two numbers per leaf, `G` and `H`, and they move
    only because the point they are evaluated at moved. The row membership is
    the one the grower left in the active-row permutation and is fixed for the
    whole call.

    The fold order
    --------------
    Thread `t` accumulates rows `t, t + SUM_THREADS, t + 2 * SUM_THREADS, ...`
    of its leaf's slice, then the threadgroup folds those 256 partials in a
    fixed binary tree. Both the stride and the tree are compile-time constants
    and the slice is the grower's, so the sum is the same Float32 for a given
    leaf whatever the device schedules -- deterministic run to run, in the
    sense the module docstring already claims for `_abs_sum_kernel`.

    It is **not** the host's fold. `boosting._estimate_leaf_values` sums the
    same rows sequentially in ascending row order in Float64. This is a
    strided Float32 tree reduction, so the two agree to Float32 and not to the
    bit, which is the same trade every other number on this plane already
    makes and is stated in the module docstring. Agreement with the host
    implementation is asserted in `tests/test_gpu_leaf_estimation.mojo` rather
    than argued here.

    The guard, the cap and the clamp
    --------------------------------
    A non-positive `H + lambda_l2` leaves the value alone. On the host that is
    a `break` out of the iteration loop; here it is a skip, and the two end at
    the same value because a skipped iteration changes nothing the next
    iteration reads, so every later iteration skips too.

    `max_delta_step` and the monotone interval are re-applied after every
    step, not only after the last, because both are projections onto a fixed
    set: applying one to an already-projected value is the identity, so this
    costs nothing and keeps the value the *next* iteration differentiates at
    inside the cap and inside the constraint the tree was grown under.
    `path_smooth` is not a projection and is refused beside this parameter in
    `ExtraTreeParams.check_leaf_estimation`, which is why no smoothing appears
    here.

    No fused multiply-add is at stake in the step. `-T(G) / (H + lambda_l2)`
    is a divide and the accumulation `v + step` adds its quotient, so there is
    no multiply for the device compiler to contract into the following add --
    unlike `learning_rate * value[node]` in `_range_table_add_raw_kernel`,
    whose contraction cost that lane a bit and is documented there. The
    reduction loop is a chain of plain adds for the same reason.
    """
    var tid = thread_idx.x
    var seg = Int(block_idx.x)
    var sg = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var base = seg * EST_WORDS
    var begin = Int(segs[unsafe_offset = base + EST_BEGIN][0])
    var count = Int(segs[unsafe_offset = base + EST_COUNT][0])

    var acc_g = Float32(0.0)
    var acc_h = Float32(0.0)
    var j = tid
    while j < count:
        var r = Int(rows[unsafe_offset = begin + j][0])
        acc_g += grad[unsafe_offset=r][0]
        acc_h += hess[unsafe_offset=r][0]
        j += SUM_THREADS
    sg[unsafe_offset=tid] = acc_g
    sh[unsafe_offset=tid] = acc_h
    barrier()

    # Uniform trip count across the threadgroup, so every thread reaches every
    # barrier, exactly as in `_abs_sum_kernel`.
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
        var vbase = seg * EST_VALS
        var g_sum = sg[unsafe_offset=0][0]
        var h_sum = sh[unsafe_offset=0][0]
        var denom = h_sum + lambda_l2
        if denom > 0.0:
            # `gain.soft_threshold_l1`, arm for arm. `g_sum == 0` reaches the
            # `mag <= 0` arm whenever `lambda_l1 > 0`, so the sign test below
            # is never asked about a zero.
            var s = g_sum
            if lambda_l1 > 0.0:
                var mag = abs(g_sum) - lambda_l1
                if mag <= 0.0:
                    s = 0.0
                elif g_sum > 0.0:
                    s = mag
                else:
                    s = -mag
            var v = vals[unsafe_offset = vbase + EST_V][0] + (-s / denom)
            # `tree_parameters_extra.cap_leaf_output`.
            if max_delta_step > 0.0:
                if v > max_delta_step:
                    v = max_delta_step
                elif v < -max_delta_step:
                    v = -max_delta_step
            # `monotone.OutputBounds.clamp`.
            var b_lo = vals[unsafe_offset = vbase + EST_LO][0]
            var b_hi = vals[unsafe_offset = vbase + EST_HI][0]
            if v < b_lo:
                v = b_lo
            elif v > b_hi:
                v = b_hi
            vals[unsafe_offset = vbase + EST_V] = v


struct GpuLeafEstimator(Movable):
    """CatBoost's `leaf_estimation_iterations` for the device-resident round.

    After a tree's structure is fixed, re-estimate each leaf's value `k` times
    instead of once, each iteration recomputing that leaf's gradient and
    hessian sums against the raw scores the previous iteration produced. The
    host implementation is `boosting._estimate_leaf_values` and is the
    definition; this is the same iteration on the plane where the raw scores
    live on the device, node-identical to that implementation to Float32.

    Which shape this is, and why
    ----------------------------
    Two shapes were available. This is the **per-iteration device reduction
    inside the tree**: the launches go into the schedule and the host stays
    out of the loop. The alternative -- lifting leaf estimation out of the
    device round, downloading the raw scores, iterating on the host and
    uploading the values -- was rejected because it converts a fixed launch
    cost into `k - 1` *round trips* per tree, and section 6.1.1 of
    `docs/GPU_PORTABILITY.md` is explicit that a round-trip count predicts
    seconds while a launch or copy count predicts ordering hazard. A hundred
    rounds at `k = 10` would be nine hundred waits on a plane whose entire
    design is one wait per tree.

    What it costs, counted in source
    --------------------------------
    Three launches per extra iteration, independent of the leaf count, plus
    one device-to-host copy and one synchronization per tree:

      shift    `_leaf_shift_raw_kernel`, one thread per live row
      grads    `_grad_hess_kernel`, the round's own derivative kernel,
               unmodified, over the shifted scores
      step     `_leaf_newton_kernel`, one threadgroup per live leaf

    So `k` iterations add `3 * (k - 1)` launches to a tree's schedule. The
    campaign's measured enqueue cost is flat at 6-7 microseconds through 64
    command buffers and 14-17 beyond, and the two schedules sit on opposite
    sides of that knee, so the same six launches are not the same cost:

    - **Leaf-wise**, a tree already issues 278 launches, well past the knee.
      `k = 3` takes it to 284, a 2.2% increase entirely inside the expensive
      regime: about 84-102 microseconds of enqueue per tree.
    - **Oblivious**, a depth-6 tree issues 62, inside the flat regime. `k = 3`
      takes it to 68, a 9.7% increase, and it **crosses the knee**: six
      launches that would each have cost 6-7 microseconds instead push the
      schedule past 64 buffers. Proportionally the added launches cost more on
      the oblivious schedule, and the crossing is the reason, not the count.

    The one round trip per tree is unavoidable in either shape: `Tree.value`
    is a host list and the ensemble is scored from it, so the finished values
    have to come home. What this shape buys is that it is **one** and not
    `k - 1`.

    Memory
    ------
    Three `n_rows` Float32 planes -- the shifted scores and the derivative
    pair -- so 12 MB at a million rows, allocated only when a fit actually
    sets the parameter above 1. The derivative planes are the estimator's own
    rather than the histogram builder's: the builder's hold the round's
    gradients, and although nothing reads them after the tree is grown,
    borrowing them would make this feature's correctness depend on that
    remaining true.

    Construct once per fit, alongside `GpuObjectiveState`, and call `estimate`
    once per tree between growth and the raw-score update.
    """

    var shift_dev: DeviceBuffer[DType.float32]
    """`raw[r] + v[leaf(r)]`, the point this iteration differentiates at."""
    var grad_dev: DeviceBuffer[DType.float32]
    var hess_dev: DeviceBuffer[DType.float32]
    var seg_dev: DeviceBuffer[DType.int32]
    """`EST_WORDS` Int32 per live leaf, staged once per tree."""
    var val_dev: DeviceBuffer[DType.float32]
    """`EST_VALS` Float32 per live leaf: the value the iteration carries, and
    the monotone interval it stays inside."""
    var stage_seg: HostBuffer[DType.int32]
    var stage_val: HostBuffer[DType.float32]
    """Pinned staging for `val_dev`, **and** the destination of the one
    readback. Both directions can share it because the queue is in order: the
    upload copy has retired before the last iteration's kernels run, and the
    readback is enqueued behind all of them."""

    var n_rows: Int
    var max_nodes: Int
    var block_threads: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        max_nodes: Int = DEFAULT_MAX_NODES,
    ) raises:
        if n_rows < 1:
            raise Error("leaf estimation requires at least one row")
        if max_nodes < 1:
            raise Error("max_nodes must be positive")
        self.n_rows = n_rows
        self.max_nodes = max_nodes
        self.block_threads = derive_block_threads(query_device_caps(ctx))
        self.shift_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.seg_dev = ctx.enqueue_create_buffer[DType.int32](
            max_nodes * EST_WORDS
        )
        self.val_dev = ctx.enqueue_create_buffer[DType.float32](
            max_nodes * EST_VALS
        )
        self.stage_seg = ctx.enqueue_create_host_buffer[DType.int32](
            max_nodes * EST_WORDS
        )
        self.stage_val = ctx.enqueue_create_host_buffer[DType.float32](
            max_nodes * EST_VALS
        )
        # Once per fit, not once per tree. `_grad_hess_kernel` sweeps the whole
        # grid, so it differentiates rows the shift kernel never wrote --
        # out-of-bag rows, under a sampler. Their derivatives are read by
        # nothing, since no leaf owns them, but an allocation's contents are
        # not defined and a NaN sitting in a buffer is the kind of thing that
        # is harmless until it is not. Zero is a point every device objective
        # is finite at.
        with self.shift_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(n_rows):
                dst.unsafe_store(r, Float32(0.0))

    def estimate(
        mut self,
        ctx: DeviceContext,
        mut state: GpuObjectiveState,
        mut rows: GpuActiveRows,
        mut values: List[Float64],
        is_leaf: List[Bool],
        bounds: List[OutputBounds],
        objective: Int,
        alpha: Float64,
        iterations: Int,
        lambda_l1: Float64,
        lambda_l2: Float64,
        max_delta_step: Float64,
    ) raises:
        """Take `iterations - 1` extra Newton steps on every live leaf of the
        tree `values` belongs to, and write the results back into `values`.

        Call after growth and before the next `begin_tree`, which is what
        resets the ranges this reads, and before the raw-score update, which
        is what consumes the values this writes.

        `values` is the tree's node-value array, **finished**: capped and
        clamped, exactly as the grower left it and exactly as
        `update_raw_ranges` will consume it. That is the value whose
        derivatives the first extra iteration evaluates, because it is the
        value the leaf actually holds. `is_leaf` and `bounds` are parallel to
        it: `is_leaf[node]` is `tree.feature[node] < 0`, and `bounds` is
        `tree.node_bounds`'s output, or empty when no monotone constraint is
        active.

        **`iterations <= 1` returns before it stages a word, and that early
        return is the bit-identity guarantee.** It mirrors, statement for
        statement, the early return `boosting._estimate_leaf_values` opens
        with: iteration 1 is never recomputed here either, so the value the
        grower wrote from the histogram's own sums is the value the tree
        keeps, and the default path enqueues nothing, copies nothing and waits
        on nothing.

        Order of operations, per tree
        -----------------------------
        1. Flatten the live leaf ranges into ascending-node-order records on
           the host, one per leaf with a live range, each carrying the leaf's
           current value and its monotone interval. Internal nodes are skipped
           twice over: their ranges are empty once their children own their
           rows, and `is_leaf` is checked as well, so a range a grower failed
           to clear cannot be mistaken for a leaf.
        2. Upload both planes. Two copies, once, not once per iteration.
        3. For each of the `iterations - 1` extra steps: shift, differentiate,
           reduce-and-step. Nothing crosses to the host inside this loop.
        4. Read the value plane back, once, and write it into `values`.

        The membership does not move between steps and neither does the
        structure. Only `G` and `H` move, and only because the point they are
        taken at moved.
        """
        if iterations <= 1:
            return
        if state.n_classes != 1:
            raise Error(
                "leaf estimation iterations are single-output only; the"
                " multiclass trainers refuse the setting by name"
            )
        if state.n_rows != self.n_rows:
            raise Error(
                "objective state and leaf estimator disagree on n_rows"
            )
        if not state.has_raw:
            raise Error("call init_raw before estimating leaf values")
        if objective == CUSTOM or not supports_device_objective(objective):
            raise Error(
                "leaf estimation iterations need a device objective kernel;"
                " custom objectives stay on the host path"
            )
        if not isfinite(alpha):
            raise Error("alpha must be finite")
        if not isfinite(lambda_l1) or not isfinite(lambda_l2):
            raise Error("regularization must be finite")
        if not isfinite(max_delta_step):
            raise Error("max_delta_step must be finite")
        if len(values) > self.max_nodes:
            raise Error(
                "tree has more nodes than the leaf-estimation table holds;"
                " construct with a larger max_nodes"
            )
        for i in range(len(values)):
            if not isfinite(values[i]):
                raise Error("node values must be finite")

        var dseg = self.stage_seg.unsafe_ptr()
        var dval = self.stage_val.unsafe_ptr()
        var nodes = List[Int]()
        var n_segments = 0
        var total = 0
        for node in range(rows.ranges.n_nodes()):
            if node >= len(values):
                break
            if node < len(is_leaf) and not is_leaf[node]:
                continue
            var window = rows.ranges.get(node)
            var n = window.count()
            if n <= 0:
                continue
            var base = n_segments * EST_WORDS
            dseg.unsafe_store(base + EST_START, Int32(total))
            dseg.unsafe_store(base + EST_BEGIN, Int32(window.begin))
            dseg.unsafe_store(base + EST_COUNT, Int32(n))
            dseg.unsafe_store(base + EST_WORDS - 1, Int32(0))
            var vbase = n_segments * EST_VALS
            dval.unsafe_store(vbase + EST_V, Float32(values[node]))
            # An inactive bound is `Float64.MAX_FINITE`, which is not a finite
            # Float32; it is staged as the Float32 maximum instead, so the
            # comparison in the kernel is against a real number and the clamp
            # is the identity either way.
            var b_lo = -Float32.MAX_FINITE
            var b_hi = Float32.MAX_FINITE
            if node < len(bounds):
                if bounds[node].lo > -NO_BOUND:
                    b_lo = Float32(bounds[node].lo)
                if bounds[node].hi < NO_BOUND:
                    b_hi = Float32(bounds[node].hi)
            dval.unsafe_store(vbase + EST_LO, b_lo)
            dval.unsafe_store(vbase + EST_HI, b_hi)
            dval.unsafe_store(vbase + EST_VALS - 1, Float32(0.0))
            nodes.append(node)
            n_segments += 1
            total += n
        if n_segments == 0:
            return

        ctx.enqueue_copy(dst_buf=self.seg_dev, src_ptr=dseg)
        ctx.enqueue_copy(dst_buf=self.val_dev, src_ptr=dval)

        var l1 = Float32(lambda_l1)
        var l2 = Float32(lambda_l2)
        var cap = Float32(max_delta_step)
        var shift_blocks = (
            total + self.block_threads - 1
        ) // self.block_threads
        var row_blocks = (
            self.n_rows + self.block_threads - 1
        ) // self.block_threads
        for _ in range(iterations - 1):
            ctx.enqueue_function[_leaf_shift_raw_kernel](
                rows.rows_dev.unsafe_ptr(),
                state.raw_dev.unsafe_ptr(),
                self.shift_dev.unsafe_ptr(),
                self.seg_dev.unsafe_ptr(),
                self.val_dev.unsafe_ptr(),
                Int32(n_segments),
                Int32(total),
                grid_dim=shift_blocks,
                block_dim=self.block_threads,
            )
            # The round's own derivative kernel, over the shifted scores and
            # the state's own labels and weights. Not a copy of it and not a
            # variant of it: the same function, so the weight multiplier and
            # every objective arm keep one definition on this backend.
            ctx.enqueue_function[_grad_hess_kernel](
                self.shift_dev.unsafe_ptr(),
                state.target_dev.unsafe_ptr(),
                state.weight_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(objective),
                Float32(alpha),
                Int32(1) if state.weighted else Int32(0),
                grid_dim=row_blocks,
                block_dim=self.block_threads,
            )
            ctx.enqueue_function[_leaf_newton_kernel](
                rows.rows_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                self.seg_dev.unsafe_ptr(),
                self.val_dev.unsafe_ptr(),
                l1,
                l2,
                cap,
                grid_dim=n_segments,
                block_dim=SUM_THREADS,
            )

        # The tree's one round trip. `Tree.value` is a host list and the
        # ensemble is scored from it, so the finished values have to come
        # home; what this shape buys is that they come home once and not once
        # per iteration.
        ctx.enqueue_copy(
            dst_ptr=self.stage_val.unsafe_ptr(), src_buf=self.val_dev
        )
        ctx.synchronize()
        var src = self.stage_val.unsafe_ptr()
        for s in range(n_segments):
            values[nodes[s]] = Float64(src.unsafe_load(s * EST_VALS + EST_V))
