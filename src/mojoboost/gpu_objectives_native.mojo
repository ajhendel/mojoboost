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
place that says which objective goes which way.

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
from std.memory import stack_allocation
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
    SQUARED_ERROR,
    TWEEDIE,
    _POISSON_MAX_DELTA_STEP,
)
from .gpu_active_rows import GpuActiveRows
from .gpu_tiling import derive_block_threads, query_device_caps

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

# Mirrors `_FIXED_ONE` in histogram_gpu.mojo: half the Int32 range, the bound
# every partial sum of scaled values has to stay inside.
comptime FIXED_ONE = Float64(1 << 30)

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
    any code the built-in trainer does not define."""
    return (
        objective == SQUARED_ERROR
        or objective == BINARY_LOGISTIC
        or objective == CROSS_ENTROPY
        or objective == POISSON
        or objective == GAMMA
        or objective == TWEEDIE
        or objective == HUBER
        or objective == QUANTILE
        or objective == L1
        or objective == MAPE
        or objective == FAIR
    )


def device_fixed_scale(total: Float64) raises -> Float32:
    """The fixed-point histogram scale for a magnitude sum, the scalar core
    of `_fixed_scale` in histogram_gpu.mojo with the host-side pass over the
    values replaced by `magnitude_sums`.

    Same expression, same Float32 result, so a builder fed from the device
    reduction quantizes identically to one fed from a host list. The two must
    stay in step; the handoff names the one-line refactor that would leave a
    single definition.
    """
    var t = total
    if not isfinite(t):
        raise Error("gradients and hessians must be finite")
    if t < 1e-12:
        t = 1e-12
    var scale = Float32(FIXED_ONE / t)
    if not isfinite(scale) or scale <= 0.0:
        raise Error(
            "gradient/hessian magnitudes are out of range for the GPU"
            " fixed-point histogram"
        )
    return scale


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
    gradient `p - y`, hessian `2 p (1 - p)` floored.

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
    var h = 2.0 * p * (1.0 - p)
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
    values: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
    k: Int32,
    n_nodes: Int32,
    learning_rate: Float32,
):
    """The device-resident prediction update: each row's raw score for class
    `k` advances by `learning_rate * value[leaf]`, where `leaf` is the node
    the row already sits in on the device.

    This is the whole reason the leaf-assignment array is worth keeping
    around after a tree is grown. Every row ends a tree assigned to a leaf,
    and leaf ids are node ids, so the tree's own `value` array is the lookup
    table; no traversal, no feature reads, no host round trip.

    A row whose id is out of range (`OUT_OF_BAG`, or any id at or past the
    tree's node count) is left untouched: tree growth never routed it, so the
    device does not know its leaf. Under row bagging that is every
    out-of-bag row, and the caller has to score those rows some other way.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var node = leaf_ids[unsafe_offset=r][0]
    if node < 0 or node >= n_nodes:
        return
    var i = r * Int(n_classes) + Int(k)
    raw[unsafe_offset=i] = (
        raw[unsafe_offset=i][0]
        + learning_rate * values[unsafe_offset = Int(node)][0]
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
    """The current tree's node values, the lookup table `update_raw` reads."""
    var part_dev: DeviceBuffer[DType.float32]
    var base_dev: DeviceBuffer[DType.float32]
    var host_part: HostBuffer[DType.float32]
    var host_raw: HostBuffer[DType.float32]
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
        self.base_dev = ctx.enqueue_create_buffer[DType.float32](n_classes)
        self.part_dev = ctx.enqueue_create_buffer[DType.float32](
            2 * SUM_BLOCKS
        )
        self.host_part = ctx.enqueue_create_host_buffer[DType.float32](
            2 * SUM_BLOCKS
        )
        self.host_raw = ctx.enqueue_create_host_buffer[DType.float32](n_scores)

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
        already renewed and not yet shrunk: the kernel applies
        `learning_rate` itself, so it consumes exactly `tree.value` and the
        booster's learning rate.

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
        with self.value_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(values)):
                dst.unsafe_store(i, Float32(values[i]))
        ctx.enqueue_function[_update_raw_kernel](
            self.raw_dev.unsafe_ptr(),
            leaf_dev.unsafe_ptr(),
            self.value_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_classes),
            Int32(k),
            Int32(len(values)),
            Float32(learning_rate),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )

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
        belongs to a leaf of the finished tree, so the update is one small
        launch per leaf over exactly that leaf's rows; rows outside every
        range (out of bag) keep their old scores, the same contract
        `update_raw` has for unrouted rows."""
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
        with self.value_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(values)):
                dst.unsafe_store(i, Float32(values[i]))
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
        """
        ctx.enqueue_function[_abs_sum_kernel](
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
        ctx.synchronize()
        var src = self.host_part.unsafe_ptr()
        var g_total = 0.0
        var h_total = 0.0
        for i in range(SUM_BLOCKS):
            g_total += Float64(src.unsafe_load(i))
            h_total += Float64(src.unsafe_load(SUM_BLOCKS + i))
        if not isfinite(g_total) or not isfinite(h_total):
            raise Error("gradients and hessians must be finite")
        return GradMagnitudes(g_total, h_total)

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
