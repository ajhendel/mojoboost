"""Device-side gradients, hessians, and raw-score updates.

The GPU trainer in train_gpu.mojo computes every round's derivatives on the
host (`_fill_grad_hess` over Float64 lists) and then uploads `2 * n_rows`
Float32 to the device. That upload is pure overhead: the labels never change,
the user's sample weights never change, and the raw scores are the only thing
that moves between rounds, by an amount the device already knows (each row's
leaf and that leaf's value). This module keeps all three device-resident so a
boosting round costs no per-row host-to-device traffic at all.

What lives here:

  `GpuObjectiveState`   the per-session device buffers: target (or class
                        label), sample weight, raw scores, and softmax
                        probabilities. Uploaded once at construction. The
                        weight plane is the one exception, because CatBoost's
                        Bayesian bootstrap redraws it per tree; see
                        `refresh_weights`.
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
which is exactly the interface a one-thread-per-row kernel can serve.

CatBoost's ranking objectives -- QueryRMSE, PairLogit, YetiRank -- are also
covered, at the bottom of this file, and they are a different interface rather
than three more arms of `_grad_hess_kernel`: a row's ranking gradient is a
function of the *other rows in its query group*, so neither the kernel shape
nor the state is the same. `GpuRankingState` holds their two planes, and
`ranking_pairwise.mojo` -- which imports nothing from `max.gpu.*` and is
therefore reachable from a CPU-only build -- holds the derivative definitions,
the group and pair conventions, the refusals, and the fixed-point arithmetic a
ranking gradient profile implies. Nothing in this paragraph's subject touches
`_grad_hess_kernel`, `_softmax_prob_kernel`, `_softmax_class_kernel`, the four
raw-score update kernels, `_abs_sum_kernel`, or `GpuObjectiveState`: the
ranking path is new symbols beside them, which is what keeps every
non-ranking fit bit-identical.

CUSTOM
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

Where the weight is applied, and why it is not in the histogram
--------------------------------------------------------------
A weight is a per-row multiplier on the gradient and on the hessian, and it
could be applied in either of two places. The CPU applies it in the objective,
before anything is quantized: `boosting._fill_grad_hess_into` loads
`w = weights[r]` as the first statement of every objective's row body and
stores `w * (...)` into `grad` and `hess`. Nothing downstream of that carries a
weight at all -- `histogram.build_histogram` takes `grad`, `hess`, the bins and
the constant-hessian flag, and there is no weight argument anywhere in the
histogram API on either backend.

The device matches that exactly. `_grad_hess_kernel` and
`_softmax_grad_hess_kernel` apply the weight, and the histogram accumulation is
untouched: no weight plane is fetched per (row, feature) visit and no multiply
is added to the hottest loop in the codebase. The alternative -- carrying the
weight into the accumulation -- would put the two backends on different
arithmetic (a weighted product formed once per row against one formed once per
visit) and buy nothing, since the row's weight is loop-invariant across its
features.

The corollary for the host-gradient path: `GpuHistogramBuilder.stage_gradients`
needs nothing added to it. The Float64 gradients it stages already have the
weight multiplied in by `_fill_grad_hess_into`, so a weighted fit on that path
was already correct and is unchanged by anything here.

What a weighted fit costs the staging arm
-----------------------------------------
`boosting.round_has_constant_hessian` ends in
`objective_has_constant_hessian(objective, len(sample_weight) > 0)`, so a
non-empty weight vector refuses the constant-hessian declaration for every
objective. That is a definition rather than a conservatism: under squared
error, L1, huber and quantile the CPU stores the bare weight as the hessian
(`hp.unsafe_store(r, derivative[NARROW](w))`), so the hessian *is* the weight
and a builder told to rebuild that plane from the row count would rebuild the
wrong plane.

Priced on the Int16 gradient-staging arm, per
`GpuActiveRows.staged_gradient_bytes_per_row`, that costs the better half of
the arm. A constant-hessian round stages the gradient alone, 2 bytes per row;
a weighted round stages both planes, 4. At the default feature group of one,
each (row, feature) visit fetches 4 bytes of row index plus the staged
derivative plus 1 bin byte, so a weighted fit is on **9 bytes per visit where
an unweighted one is on 7**. Every fit under a per-row weight -- a user's
`sample_weight`, a class weight folded into it, or a Bayesian bootstrap draw
-- is on the 9-byte width by construction. That is the arithmetic of the
declaration and not a regression in anything measured here.

That arithmetic is not special to the single-output case and is *not*
relaxed by a multi-target fit. A weighted multi-target round stages both
derivative planes per output for the same reason a weighted scalar round
stages both: under squared error the CPU stores the bare weight as the
hessian, so the hessian is the weight and cannot be rebuilt from a row count.
Nine bytes per (row, feature) visit at the default feature group, per output,
by construction.

Multi-target fits (CatBoost's MultiRMSE), and the tree shape they take
---------------------------------------------------------------------
A multi-target fit predicts a vector rather than a scalar. Two tree shapes
can carry that, and they are different trees, not different derivative
formulas:

  K trees per round   one scalar-leaf tree per output, each grown on its own
                      output's derivatives, the vector assembled by summing
                      K independent traversals. `objective_is_multi_output`
                      in objective_registry.mojo already *defines*
                      multi-output this way ("whether one boosting iteration
                      grows more than one tree").
  one vector tree     one tree whose split is scored against a gain summed
                      over outputs and whose leaf holds one value per output.
                      This is CatBoost's shape for MultiRMSE.

**mojotrees takes the first, and this module implements the first.** That is
not a preference; it is the only shape the rest of the codebase can express.
`tree.Tree.value` is a `List[Float64]` indexed by node id and
`Tree.predict_row` returns one `Float64`, so a leaf cannot hold a vector
without a new field on `Tree`; and softmax multiclass, the one multi-output
trainer that exists, already grows K trees per round
(`boosting._boost_rounds_multiclass`, `trees[round * n_classes + k]`). A
device path that grew vector-leaf trees would have no host model to put them
in. What is NOT here, and what the other shape would cost, is written out
under `fill_multi_grad_hess`.

Given that shape, MultiRMSE is the *easier* of the two multi-output
derivative problems, and deliberately so. Softmax couples the outputs: a
round needs a probability pass over the whole row before any class's gradient
exists, because class `k`'s gradient reads a denominator summed over every
class. MultiRMSE does not couple them at all -- output `j`'s gradient is
`w * (raw[r, j] - y[r, j])` and reads nothing of output `j'` -- so the
probability pass, `prob_dev`, and `refresh_softmax` all disappear and the
derivative kernel is a pure per-(row, output) map. Everything downstream is
the multiclass machinery unchanged: `_multi_grad_hess_kernel` writes the same
class-major plane `grad[slot * n_rows + r]` that `_batch_softmax_grad_kernel`
writes, so `GpuClassBatch.magnitude_sums`, `set_scales`, `scatter_slot` and
every batched histogram kernel take a multi-target round without one line
changed, and `update_raw(k=j)` advances output `j`'s slot of the row-major
raw scores exactly as it advances class `k`'s.

One fixed-point scale per output, never one shared
--------------------------------------------------
With K outputs there are K gradient-magnitude profiles, and a multi-target
fit is the case where they genuinely differ: an output in units of 1 beside
an output in units of 1000 produces magnitude sums three orders of magnitude
apart on the very first round. `GpuClassBatch.set_scales` already answers
this per slot -- `device_fixed_scale(mags[c].grad)` for each `c`, from that
slot's own reduction -- and the multi-target path inherits that answer
unchanged, which is the whole reason the derivative kernel writes into the
batch's plane layout rather than a layout of its own. A shared scale would
size the small output's lattice by the large output's magnitudes and quantize
every one of its gradients toward zero, and it would do so silently: the fit
would converge on the large output and barely move on the small one.
`tests/test_gpu_multitarget.mojo` fixes outputs three orders of magnitude
apart for exactly this reason. Equal-scale outputs would pass a shared-scale
implementation, and they would hide an output-index error entirely, because
swapping two identically-scaled planes changes nothing observable.

REACHABILITY: the multi-target path is UNREACHABLE from any user-facing API
---------------------------------------------------------------------------
Stated here rather than left to be discovered, because this project has
already shipped two features that were complete, correct, tested and not
callable (CatBoost mode and `grow_policy=oblivious`, both invisible until
somebody walked a path end to end instead of checking its entry point). A
feature nobody can call is as absent as one nobody wrote, and it passes every
Mojo test either way. So, walked end to end as of this lane:

  `python/mojotrees/sklearn.py`   `MTRegressor._OBJECTIVES` has no
                                  multi-target name and no estimator accepts
                                  a 2-D `y`. Not an allow-list gap: a
                                  multi-target estimator is a new class, not
                                  a new dictionary entry.
  `params.mojo`                   No `num_targets` key. `num_class` parses,
                                  and `_validate` **refuses** it for every
                                  objective but `multiclass`, so a
                                  multi-target fit cannot even be spelled in
                                  a parameter string.
  `trainset.Dataset`              One `List[Float64]` label column. A target
                                  matrix cannot be expressed at the dataset
                                  level at all.
  `objective_registry`            No multi-target objective code exists.
                                  `objective_is_multi_output` is True for
                                  `MULTICLASS` alone.
  trainers                        Nothing calls `fill_multi_grad_hess`. It is
                                  reached only by
                                  `tests/test_gpu_multitarget.mojo`.

What is built here is therefore the **device derivative plane and its
per-output scale**, tested against the scalar device path bit for bit, and
nothing above it. Making it reachable is four further pieces of work, none of
them in this file and none of them this lane's: a label matrix on `Dataset`,
a `num_targets` parameter, a `train_multi_target` round loop shaped like
`_boost_rounds_multiclass`, and a `MTMultiOutputRegressor` on the Python
side. Until all four land, a user asking for MultiRMSE gets a name error from
sklearn.py, which is a refusal -- the failure mode to avoid is the opposite
one, an accepted parameter with no reader.

**Every ranking round is on the 9-byte width too, and unconditionally.**
PairLogit and YetiRank have a hessian that is a per-round sum over each row's
pairs, so they can never be declared constant. QueryRMSE's hessian is the row
weight, so a weighted one cannot either, and an unweighted one -- whose hessian
really is exactly 1.0 -- is refused anyway, because
`histogram.objective_has_constant_hessian` is the one statement of which
objectives qualify and this lane does not extend it.
`GpuHistogramBuilder.fill_rank_gradients_device` is where that is refused and
argued. Again: arithmetic of the declaration, not a regression.
"""

from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import exp, isfinite, sqrt
from std.memory import bitcast, stack_allocation
from std.sys import has_accelerator
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
from .monotone import NO_BOUND, OutputBounds
from .objective_registry import objective_gradients_on_device
from .quantized_gradient import fixed_point_scale
from .ranking import RankGroups, check_groups
from .ranking_pairwise import (
    RANK_PAIR_LOGIT,
    RANK_QUERY_RMSE,
    RANK_YETI_RANK,
    PairAdjacency,
    check_rank_kind,
    check_rank_sample_weight,
    check_yeti_rank_pairs,
    describe_rank_kind,
    rank_kind_is_pairwise,
    rank_kind_regenerates_pairs,
)
from .rng import splitmix64

# sampling.mojo imports nothing from `max.gpu.*` and nothing from any GPU file
# -- only `std`, `bagging`, `rng` and `objective_registry` -- so naming it here
# adds no cycle and no accelerator dependency to a CPU-only build. The three
# names are the MVS block geometry and the keep-probability epsilon; taking
# them from the sampler rather than restating them is what keeps the device
# draw blocked exactly as the host draw blocks.
from .sampling import (
    MVS_BLOCK_SHIFT,
    MVS_BLOCK_SIZE,
    MVS_PROBABILITY_EPS_F32,
)

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


def supports_multi_output_objective(objective: Int) -> Bool:
    """Whether `objective` has a per-output device kernel for a multi-target
    fit.

    True for `SQUARED_ERROR` alone, which under a vector target is CatBoost's
    MultiRMSE: the loss is separable over outputs and each output's
    derivative is the scalar squared-error derivative at that output's raw
    score. Every other code is False, and `fill_multi_grad_hess` **refuses**
    it by name rather than falling through to a per-output squared error,
    which is the failure mode this predicate exists to make impossible.

    Deliberately not routed through `objective_gradients_on_device` the way
    `supports_device_objective` is. That table answers "does this objective
    have a closed-form per-row derivative these kernels implement", and every
    one of its eleven codes does; the question here is the different one of
    "is this objective *separable over a vector target*", which poisson,
    gamma, tweedie and the rest are not -- not because their derivative is
    hard, but because CatBoost defines no multi-target form of them and
    mojotrees has no multi-target label to feed one. Answering the second
    question out of the first table would silently accept ten codes nothing
    has defined.

    CatBoost's other multi-target losses (MultiLogloss, MultiCrossEntropy,
    MultiQuantile) are absent from **both** backends, so a refusal here has
    no `device='cpu'` fallback to point at and the error message does not
    pretend otherwise.
    """
    return objective == SQUARED_ERROR


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


def _multi_grad_hess_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_outputs: Int32,
    j_begin: Int32,
    j_count: Int32,
    weighted: Int32,
):
    """MultiRMSE derivatives for a contiguous run of outputs, in one launch.
    `grid.x` tiles the rows and `grid.y` indexes the output slot, so the whole
    (row, output-in-batch) plane is one launch.

    The arithmetic per (row, output) is character for character the squared
    error arm of `_grad_hess_kernel` -- `grad = w * (raw_r - y)`, `hess = w`,
    in that operand order -- and there is no second definition of it: a
    multi-target fit's `j`-th output is a scalar squared-error fit against
    `y[:, j]`, and the only thing that differs is where the two operands live.

    Only the addressing differs, and it differs on both sides, exactly as
    `gpu_multiclass_batch._batch_softmax_grad_kernel`'s does. The reads are
    row-major (`raw[r * n_outputs + j]`, `target[r * n_outputs + j]`, the
    layout every consumer of a score reduces over outputs within a row in,
    and the layout `update_raw` and `gpu_predict` already contract for) and
    the write is class-major (`grad[slot * n_rows + r]`, the layout the
    histogram kernels and the magnitude reduction need). This kernel is the
    transpose, and it is the only one in a multi-target round.

    No coupling term appears anywhere: output `j` reads index
    `r * n_outputs + j` of two planes and nothing else. That is the whole
    difference from the softmax kernel, which reads a probability whose
    denominator summed over every class.
    """
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nr = Int(n_rows)
    if r >= nr:
        return
    var slot = Int(block_idx.y)
    if slot >= Int(j_count):
        return
    var j = Int(j_begin) + slot

    var w = Float32(1.0)
    if weighted != 0:
        w = weight[unsafe_offset=r][0]
    var i = r * Int(n_outputs) + j
    var raw_r = raw[unsafe_offset=i][0]
    var y = target[unsafe_offset=i][0]
    var out = slot * nr + r
    grad[unsafe_offset=out] = w * (raw_r - y)
    hess[unsafe_offset=out] = w


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
    # Guarded 2026-08-17 by the CPU-only build audit
    # (docs/design/CPU_ONLY_BUILD_AUDIT.md). On a build with no accelerator
    # ANY reachable `enqueue_function` elaborates a GPU kernel and fails the
    # compile with `Unknown GPU architecture detected`, whatever the kernel
    # does. This is a module-level launcher, so one `from
    # mojotrees.gpu_objectives_native import enqueue_abs_sum` in a CPU-set
    # test is all it takes to reach it, and an Apple machine can never
    # reproduce the failure.
    comptime if not has_accelerator():
        raise Error(
            "the gradient magnitude reduction needs an accelerator; this"
            " build has none"
        )
    else:
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


# --- CatBoost's `CalcDerivativesStDevFromZero`, on the device --------------
#
# `random_score_scale` wants the RMS of this round's derivatives about zero:
# `sqrt(sum(g^2) / n_rows)`, which is `derivatives_stdev_from_zero` in
# tree_parameters_extra.mojo. That is a plain sum of squares -- no mean, no
# second pass, no cross terms -- so it is the same shape as the magnitude
# reduction above and reuses its grid, its block count and its host-side
# Float64 fold.
#
# **A SEPARATE KERNEL AND A SEPARATE BUFFER, NOT A THIRD PLANE ON
# `_abs_sum_kernel`.** Adding a plane there was the obvious move and it is
# the wrong one, for three reasons that are all about coupling rather than
# cost:
#
#   1. That buffer's layout is shared. `gpu_multiclass_batch` writes the same
#      `slot * 2 * SUM_BLOCKS + block` arithmetic from its own kernel, and
#      `gpu_fused_round` allocates to the same shape. Widening the stride
#      means editing three parallel implementations in step, and a
#      slot-arithmetic mistake there produces a PLAUSIBLE scale rather than
#      an error -- `magnitude_sums` says exactly that about its own fold.
#   2. Those magnitudes feed the fixed-point histogram scales, so every
#      histogram in the run depends on them. The RMS has one consumer and no
#      histogram reads it. Putting an unrelated quantity in the same struct
#      makes `GradMagnitudes` -- documented as "the two magnitude sums a
#      round's fixed-point scales come from" -- mean two things, which is the
#      one-name-two-meanings defect this campaign keeps paying for.
#   3. It is guarded. At the shipped `random_strength = 0` nothing here is
#      launched and nothing is read, exactly as `set_random_score` is guarded
#      in train_gpu, so a default fit issues the launches it issued before
#      this existed. A third plane on the magnitude kernel would be computed
#      on every round of every fit for the one in a hundred that reads it.
#
# The cost of keeping them apart is one extra launch and one extra 1 KB
# readback per ROUND (not per node) on fits that set `random_strength`, which
# is nothing beside the ~279 launches a round already issues.


def _sq_sum_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """Per-threadgroup sum of `grad^2`, one entry per threadgroup.

    `_abs_sum_kernel`'s shape with one plane instead of two and `g * g` in
    place of `abs(g)`. The grid stride, the block count and the shared-memory
    tree reduction are identical and fixed, so the partials and their
    host-side total are bit-identical run to run -- the property the noise
    scale needs, since it multiplies every candidate's draw.

    Float32 accumulation per thread, folded in Float64 on the host, which is
    the same split `_abs_sum_kernel` uses. Each thread accumulates
    `n_rows / (SUM_BLOCKS * SUM_THREADS)` terms -- about 15 at a million rows
    -- so the per-thread error stays far inside the tolerance the
    cross-backend test asserts, and squaring cannot overflow Float32 for any
    gradient a finite objective produces.
    """
    var tid = thread_idx.x
    var sq = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var acc = Float32(0.0)
    var nr = Int(n_rows)
    var r = Int(block_idx.x) * SUM_THREADS + tid
    var stride = SUM_BLOCKS * SUM_THREADS
    while r < nr:
        var g = grad[unsafe_offset=r][0]
        acc += g * g
        r += stride
    sq[unsafe_offset=tid] = acc
    barrier()

    # Uniform trip count across the threadgroup, so every thread reaches
    # every barrier. Same reduction as `_abs_sum_kernel`.
    var active = SUM_THREADS // 2
    while active > 0:
        if tid < active:
            sq[unsafe_offset=tid] = (
                sq[unsafe_offset=tid][0] + sq[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2

    if tid == 0:
        partials[unsafe_offset = Int(block_idx.x)] = sq[unsafe_offset=0][0]


def enqueue_sq_sum(
    ctx: DeviceContext,
    mut grad_dev: DeviceBuffer[DType.float32],
    mut part_dev: DeviceBuffer[DType.float32],
    n_rows: Int,
) raises:
    """Enqueue `_sq_sum_kernel` over `n_rows` rows into a caller-owned partial
    buffer of `SUM_BLOCKS` Float32. Does not copy and does not synchronize.

    Split from the read for the same reason `enqueue_abs_sum` is: a caller
    that wants to overlap the round trip enqueues here and waits later.
    """
    # Guarded 2026-08-17, same reason as `enqueue_abs_sum` above.
    comptime if not has_accelerator():
        raise Error(
            "the squared-gradient reduction needs an accelerator; this build"
            " has none"
        )
    else:
        ctx.enqueue_function[_sq_sum_kernel](
            grad_dev.unsafe_ptr(),
            part_dev.unsafe_ptr(),
            Int32(n_rows),
            grid_dim=SUM_BLOCKS,
            block_dim=SUM_THREADS,
        )


def sum_sq_partials[partials_origin: MutOrigin, //](
    partials: MutPointer[Float32, partials_origin],
) raises -> Float64:
    """Fold a downloaded square-partial buffer into one total.

    Ascending block index, accumulated in Float64, exactly as
    `sum_abs_partials` folds its two planes. Every caller sums in this one
    order over partials the one kernel produced, so the total and the scale
    derived from it agree bit for bit whichever driver enqueued the
    reduction.

    **This total will not equal the host replica's bit for bit, and that is
    expected rather than a defect.** `derivatives_stdev_from_zero` sums in ROW
    order in Float64; this sums Float32 per thread and folds 256 block
    partials in Float64. Float addition is not associative, so the two are two
    summation orders of the same quantity and neither approximates the other.
    The cross-backend test asserts agreement to a stated relative tolerance
    with an anti-vacuity check, never bit-identity. See the DIVERGENCE table.
    """
    var total = 0.0
    for i in range(SUM_BLOCKS):
        total += Float64(partials.unsafe_load(i))
    if not isfinite(total):
        raise Error("gradients must be finite")
    return total


def _check_weight_vector(weights: List[Float64], n_rows: Int) raises:
    """The one definition of a valid per-row weight vector on this side: one
    finite, nonnegative entry per row.

    Shared by the constructor and by `refresh_weights` so a bootstrap draw
    cannot reach the device through a weaker check than a user's
    `sample_weight` does. Nonnegative rather than positive because zero is a
    meaningful weight -- it is how a row is excluded -- and the kernels carry
    it through to an exactly zero gradient and hessian.
    """
    if len(weights) != n_rows:
        raise Error("sample_weight length must equal the target length")
    for r in range(n_rows):
        if not isfinite(weights[r]) or weights[r] < 0.0:
            raise Error("sample_weight must be finite and nonnegative")


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
    """The regression target, or the integer class label for softmax, or --
    under `multi_output` -- the row-major target matrix
    `y[r * n_outputs + j]`.

    The one field whose length is not always `n_rows`: it is
    `n_rows * n_classes` on a multi-target state and `n_rows` everywhere
    else, which is why the constructor sizes it from `len(target)` rather
    than from `self.n_rows`."""
    var weight_dev: DeviceBuffer[DType.float32]
    """Sample weights, or a one-element placeholder when unweighted:
    zero-length device buffers are not portable.

    The one plane here that a caller may rewrite after construction, because
    CatBoost's Bayesian bootstrap draws a fresh per-row weight every tree and
    the round's effective weight is that draw times the user's own
    `sample_weight` (`sampling.refresh_bayesian_bootstrap`). See
    `refresh_weights`, which is also what grows this buffer from the
    placeholder on a fit that started unweighted."""
    var raw_dev: DeviceBuffer[DType.float32]
    """Row-major raw scores, `raw[r * n_classes + k]`."""
    var prob_dev: DeviceBuffer[DType.float32]
    """Softmax probabilities in the same layout, or a placeholder when
    `n_classes == 1` and on a `multi_output` state, whose outputs are
    uncoupled and so have nothing to normalize over."""
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
    var sq_part_dev: DeviceBuffer[DType.float32]
    """`SUM_BLOCKS` Float32 of square-partials, `_sq_sum_kernel`'s output.

    Its own buffer rather than a third plane on `part_dev`: that layout is
    shared with `gpu_multiclass_batch`'s kernel and `gpu_fused_round`'s
    allocation, and it feeds the fixed-point histogram scales, which this
    quantity has nothing to do with. See `_sq_sum_kernel`.

    One slot, not `SCALE_WINDOW_MAX` of them. The magnitude path windows
    because it runs every round and the wait is worth amortizing; the RMS is
    read once per tree by a fit that asked for `random_strength`, and it is
    read immediately, so there is nothing to amortize."""
    var sq_host_part: HostBuffer[DType.float32]
    """Pinned destination for `sq_part_dev`, `SUM_BLOCKS` Float32 = 1 KB."""
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
    var stage_weight: HostBuffer[DType.float32]
    """Pinned staging for `weight_dev`, on the same grounds as `stage_value`,
    and allocated by the first `refresh_weights` rather than at construction:
    an unbootstrapped fit never refreshes its weights and must not pay
    `4 * n_rows` of pinned memory for a buffer it will not write.

    A one-element placeholder until then. `weight_stage_rows` is what says
    which of the two it currently is."""
    var weight_stage_rows: Int
    """Rows `stage_weight` is sized for: 0 while it is the placeholder,
    `n_rows` once `refresh_weights` has grown it. One number rather than a
    second flag, so the buffer and the claim about it cannot disagree."""
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
    var multi_output: Bool
    """Whether this state carries a vector target (MultiRMSE) rather than a
    scalar target or a class label.

    A separate flag from `n_classes > 1` and not derivable from it, because
    the two multi-output states have the *same* `n_classes` shape and
    opposite meanings: on a softmax state `n_classes` counts coupled classes,
    `target` is one label column, and `prob_dev` is live; on a multi-target
    state `n_classes` counts uncoupled outputs, `target` is an
    `n_rows * n_classes` matrix, and there is no probability plane at all.
    Every method that can only serve one of the two tests this flag and
    raises, rather than inferring an answer from the class count and
    computing the wrong thing."""

    def __init__(
        out self,
        ctx: DeviceContext,
        target: List[Float64],
        sample_weight: List[Float64] = [],
        n_classes: Int = 1,
        max_nodes: Int = DEFAULT_MAX_NODES,
        multi_output: Bool = False,
    ) raises:
        """Upload the labels and weights, which never change again, and
        allocate the raw-score and scratch buffers.

        For softmax, `target` holds the class labels as whole numbers and
        `n_classes` is the class count; for every other single-output
        objective `n_classes` is 1 and `target` is the regression target.

        Under `multi_output` (MultiRMSE), `n_classes` is the output count and
        `target` is the row-major target matrix, `n_rows * n_classes` long,
        with row `r`'s output `j` at `target[r * n_classes + j]` -- the same
        layout `raw_dev` already carries, so the derivative kernel reads both
        planes at one index. `n_rows` is then derived from the two rather
        than being `len(target)`.

        **The default is `False` and the `False` path is unchanged.** Every
        statement below that a single-output or softmax state executes
        computes what it computed before this parameter existed: `n_rows` is
        `len(target)`, `target_dev` is `len(target)` == `n_rows` long,
        `prob_dev` is live exactly when `n_classes > 1`, and the class-label
        validation runs on exactly the states it ran on.
        """
        if len(target) < 1:
            raise Error("device objectives require at least one row")
        if n_classes < 1:
            raise Error("n_classes must be positive")
        if max_nodes < 1:
            raise Error("max_nodes must be positive")
        if multi_output:
            if n_classes < 2:
                raise Error(
                    "a multi-output state needs at least two outputs; a"
                    " one-output fit is a scalar fit and belongs on the"
                    " ordinary single-output state"
                )
            if len(target) % n_classes != 0:
                raise Error(
                    "multi-output target length must be n_rows * n_outputs"
                )
        var n_rows = len(target) // n_classes if multi_output else len(target)
        if len(sample_weight) > 0:
            # One weight per ROW, not per (row, output): a sample weight
            # weights the observation, and every one of its outputs with it,
            # which is what lets the one weight plane serve every output slot
            # of a launch.
            _check_weight_vector(sample_weight, n_rows)
        for r in range(len(target)):
            if not isfinite(target[r]):
                raise Error("target must be finite")
        if n_classes > 1 and not multi_output:
            for r in range(len(target)):
                var label = Int(target[r])
                if Float64(label) != target[r] or label < 0 or (
                    label >= n_classes
                ):
                    raise Error(
                        "multiclass target must hold whole class labels in"
                        " 0..n_classes-1"
                    )

        self.n_rows = n_rows
        self.n_classes = n_classes
        self.max_nodes = max_nodes
        self.weighted = len(sample_weight) > 0
        self.has_raw = False
        self.multi_output = multi_output
        self.block_threads = derive_block_threads(query_device_caps(ctx))

        var n_scores = self.n_rows * n_classes
        self.target_dev = ctx.enqueue_create_buffer[DType.float32](len(target))
        self.weight_dev = ctx.enqueue_create_buffer[DType.float32](
            self.n_rows if self.weighted else 1
        )
        self.raw_dev = ctx.enqueue_create_buffer[DType.float32](n_scores)
        # No probability plane on a multi-target state: MultiRMSE outputs are
        # uncoupled, so nothing ever reduces over them and the
        # `4 * n_rows * n_outputs` a softmax state spends here would be a
        # buffer with no reader.
        self.prob_dev = ctx.enqueue_create_buffer[DType.float32](
            n_scores if (n_classes > 1 and not multi_output) else 1
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
        # 1 KB device and 1 KB host, allocated unconditionally on the same
        # argument the 64 KB above is: the alternative is a second
        # construction path keyed on a parameter this state does not carry,
        # and `random_strength` is a tree setting the objective state never
        # sees. Nothing is LAUNCHED unless a caller asks (see
        # `derivative_sum_squares`), which is where the real cost is.
        self.sq_part_dev = ctx.enqueue_create_buffer[DType.float32](SUM_BLOCKS)
        self.sq_host_part = ctx.enqueue_create_host_buffer[DType.float32](
            SUM_BLOCKS
        )
        self.part_pending = 0
        self.host_raw = ctx.enqueue_create_host_buffer[DType.float32](n_scores)
        self.stage_value = ctx.enqueue_create_host_buffer[DType.float32](
            max_nodes
        )
        self.stage_seg = ctx.enqueue_create_host_buffer[DType.int32](
            max_nodes * SEG_WORDS
        )
        # Placeholder until a caller refreshes the weights; see the field.
        self.stage_weight = ctx.enqueue_create_host_buffer[DType.float32](1)
        self.weight_stage_rows = 0

        # One-time uploads. Both buffers are written through the mapping
        # rather than staged, because this runs once per session and the
        # mapping is the shorter path; the per-round transfers below are the
        # ones that had to be cheap.
        with self.target_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(len(target)):
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

    def refresh_weights(
        mut self, ctx: DeviceContext, weights: List[Float64]
    ) raises:
        """Replace the device-resident per-row weights, for the one sampler
        whose weights are not fixed for the whole fit.

        CatBoost's Bayesian bootstrap draws a fresh weight per row per tree,
        and the round's *effective* weight is that draw times the user's own
        `sample_weight` -- `sampling.refresh_bayesian_bootstrap` is where the
        product is formed, and it is formed on the host, in Float64, because
        the draw is a `splitmix64` stream the device has no image of. What
        crosses to the device is the finished product, one Float32 per row,
        once per tree.

        **THE PARENTHETICAL THAT STOOD HERE WAS STALE AND NAMED THE WRONG
        MECHANISM**, corrected 2026-08-17. It read "MVS is refused by name in
        `sampling.canonical_bootstrap_type` and produces no weights to carry;
        when it does, it produces them the same way and arrives here unchanged."
        The first clause was true before MVS landed and is not true now:
        `canonical_bootstrap_type` accepts `"mvs"`, refusing only `Bernoulli`
        and `Poisson`, and an MVS bundle is honored on a GPU fit today. What
        keeps MVS out of THIS method is narrower and is a routing fact rather
        than a refusal -- `gpu_fused_round.ROUND_MVS_HOST_MAGNITUDES` sends an
        MVS round to `train_gpu._train_gpu_rounds`'s host-gradient arm, where
        `sampling.bootstrap_round` scales the host derivatives in place and no
        weight plane is involved at all. The second clause still holds and is
        the design a device MVS draw would take: the product would arrive here
        unchanged, and the comment block above `sampling.mvs_bootstrap_weights`
        is what the rest of it would cost. One difference worth stating, since
        it is the reason `_check_weight_vector` admits zero: an MVS draw carries
        exact ZEROS, where a Bayesian draw is positive with probability one.

        Where the weight lands, and where it deliberately does not
        ----------------------------------------------------------
        Into the *objective*, which is where the CPU puts it. The plane this
        writes is read only by `_grad_hess_kernel` and
        `_softmax_grad_hess_kernel`, which multiply both derivatives by it
        exactly as `boosting._fill_grad_hess_into` does, before anything is
        quantized. The histogram accumulation never sees a weight on either
        backend and gains no multiply and no fetch from this. The module
        docstring argues why that is the only defensible half of the choice.

        Cost and cadence
        ----------------
        `4 * n_rows` host to device per tree, plus one drain. That is the
        same shape and the same cadence as `_stage_values`, and a third of
        the traffic a host-gradient round already pays; it is not on the
        default path at all, because nothing calls this unless a bootstrap is
        configured.

        The drain is the difference from `_stage_values`, which argues it can
        skip one because the round's magnitude reduction synchronizes before
        the staging arena is rewritten. This cannot borrow that argument:
        `GpuHistogramBuilder.set_scale_refresh` can defer that reduction by up
        to `SCALE_WINDOW_MAX` rounds, so on a windowed fit there is no
        guaranteed drain between two calls here and the pinned buffer could be
        rewritten under a copy still reading it. On Metal the hazard is
        unreachable anyway (`enqueue_copy` is itself a synchronous full-queue
        drain, `docs/GPU_PORTABILITY.md` section 6.1), so this costs an
        ordering point and not a wait there; on a backend whose copies are
        genuinely asynchronous it is the whole guarantee. The same trade
        `stage_gradients` makes, for the same reason, and the alternative is
        the silent staleness this project has already been bitten by.

        Growing the plane
        -----------------
        A fit with no `sample_weight` constructed a one-element placeholder,
        so the first call here allocates the real `n_rows` buffer and its
        pinned staging and flips `weighted`. That happens once per fit, under
        the same synchronize, and every later call is the copy alone.

        An empty vector is refused rather than read as "go back to
        unweighted". `refresh_bayesian_bootstrap` returns an empty list only
        when the bundle is disabled, which is a fit that should not be
        calling this per tree at all, and silently un-weighting a round would
        be a wrong model rather than an error.
        """
        if len(weights) == 0:
            raise Error(
                "refresh_weights needs one weight per row; an empty vector is"
                " the unweighted convention and there is nothing to refresh"
            )
        _check_weight_vector(weights, self.n_rows)

        # Two hazards, one drain: a kernel still holding the placeholder
        # buffer that the growth below drops, and a copy still reading the
        # staging arena that the fill below overwrites. See the docstring.
        ctx.synchronize()

        if not self.weighted:
            self.weight_dev = ctx.enqueue_create_buffer[DType.float32](
                self.n_rows
            )
        if self.weight_stage_rows != self.n_rows:
            self.stage_weight = ctx.enqueue_create_host_buffer[DType.float32](
                self.n_rows
            )
            self.weight_stage_rows = self.n_rows

        var dst = self.stage_weight.unsafe_ptr()
        for r in range(self.n_rows):
            dst.unsafe_store(r, Float32(weights[r]))
        ctx.enqueue_copy(dst_buf=self.weight_dev, src_ptr=dst)
        self.weighted = True

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
        if self.multi_output:
            raise Error(
                "multi-output state: use fill_multi_grad_hess, which reads"
                " one output's column of the target matrix; this call would"
                " read the matrix as if it were a scalar target"
            )
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
        if self.multi_output:
            raise Error(
                "a multi-output state has no softmax coupling: MultiRMSE"
                " outputs are independent and there is no probability plane"
                " to refresh; use fill_multi_grad_hess"
            )
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
        if self.multi_output:
            raise Error(
                "a multi-output state carries a vector target, not class"
                " labels; use fill_multi_grad_hess"
            )
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

    def fill_multi_grad_hess(
        mut self,
        ctx: DeviceContext,
        objective: Int,
        j_begin: Int,
        j_count: Int,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
    ) raises:
        """Write outputs `j_begin .. j_begin+j_count-1` of a MultiRMSE round
        into `grad_dev` and `hess_dev`, class-major, in one launch.

        Slot `c` of the destination carries output `j_begin + c` at
        `[c * n_rows .. (c+1) * n_rows)`, so the buffers must hold at least
        `j_count * n_rows` Float32 each. That is exactly
        `GpuClassBatch.grad_dev`'s layout and exactly what
        `GpuClassBatch.magnitude_sums`, `set_scales` and `scatter_slot`
        consume, which is why a multi-target round gets its per-output
        fixed-point scales out of the multiclass batch machinery without a
        line of new reduction or scaling code.

        Slot-to-output is `j_begin + c` and nothing reorders it, the same
        contract `gpu_output_planes` fixes for classes, so results collected
        by ascending slot are results in ascending output order.

        What this does not do, and what the other tree shape would cost
        ------------------------------------------------------------------
        This serves the **K-trees-per-round** shape: output `j`'s plane is a
        scalar gradient/hessian pair per row, and a grower consumes it as an
        ordinary single-output tree, exactly as a class tree is grown. It
        does not serve the vector-leaf shape, and it cannot be made to from
        here. That shape needs two things this lane did not write and does
        not own:

        1. **A gain summed over outputs**, in `gpu_split_search.mojo`. A
           candidate would have to score `sum_j gain(G_j, H_j)` over
           `n_outputs` histogram planes at one split point instead of over
           one, which changes the reduction the split kernel performs and
           the argmax it feeds -- not the histogram accumulation, which
           already builds one plane per slot. A concurrent lane owns that
           expression.
        2. **A vector leaf value**, in `tree.mojo` and `model.mojo`.
           `Tree.value` is `List[Float64]` indexed by node id and
           `Tree.predict_row` returns one `Float64`; a vector leaf needs a
           second dimension on both, and every serializer, dumper and
           predictor that reads them. The CPU campaign owns those files.

        Neither is refused here as a runtime error, because neither is
        reachable: no caller can ask for a vector leaf, since no type in
        this codebase can hold one.
        """
        if not self.multi_output:
            raise Error(
                "fill_multi_grad_hess requires a multi-output state"
                " (GpuObjectiveState(..., multi_output=True)); this state"
                " carries a scalar target or class labels"
            )
        if objective == CUSTOM:
            raise Error(
                "custom objectives have no device kernel, multi-target or"
                " otherwise; the callback is host code over host lists"
            )
        if not supports_multi_output_objective(objective):
            raise Error(
                "objective code ",
                objective,
                " has no multi-target device kernel; only squared error"
                " (CatBoost MultiRMSE) is separable over a vector target"
                " here. The other CatBoost multi-target losses"
                " (MultiLogloss, MultiCrossEntropy, MultiQuantile) are"
                " implemented on neither backend, so there is no"
                " device='cpu' fallback for them to fall back to",
            )
        if not self.has_raw:
            raise Error("call init_raw before filling gradients")
        if j_count < 1:
            raise Error("output batch size must be positive")
        if j_begin < 0 or j_begin + j_count > self.n_classes:
            raise Error("output batch is outside the output range")
        ctx.enqueue_function[_multi_grad_hess_kernel](
            self.raw_dev.unsafe_ptr(),
            self.target_dev.unsafe_ptr(),
            self.weight_dev.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_classes),
            Int32(j_begin),
            Int32(j_count),
            Int32(1) if self.weighted else Int32(0),
            grid_dim=(self._row_blocks(), j_count),
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

    def derivative_sum_squares(
        mut self,
        ctx: DeviceContext,
        mut grad_dev: DeviceBuffer[DType.float32],
    ) raises -> Float64:
        """`sum(g^2)` over this round's gradients, on the device.

        The device half of `random_score_scale`: the caller divides by the row
        count, takes the root, and multiplies by `model_size_decrease`, which
        together are `tree_parameters_extra.random_score_scale_from_gradients`
        with its `derivatives_stdev_from_zero` computed here instead of over a
        host gradient list. Sum of squares rather than the finished RMS
        because the division is by rows per OUTPUT DIMENSION, which this state
        does not own -- a multiclass caller passes the flat plane and divides
        by one dimension's row count, exactly as CatBoost does.

        Independent of the magnitude path in every respect: its own kernel,
        its own buffer, its own readback, and no interaction with the window.
        In particular it does NOT touch `part_pending`, so it is safe to call
        with a windowed magnitude reduction outstanding -- unlike
        `magnitude_sums`, which refuses in that state because it would share
        the readback buffer.

        One launch and one 1 KB readback per call, and callers are expected to
        guard it on `random_strength > 0` the way `train_gpu` guards
        `set_random_score`, so a default fit never issues either.
        """
        enqueue_sq_sum(ctx, grad_dev, self.sq_part_dev, self.n_rows)
        ctx.enqueue_copy(
            dst_ptr=self.sq_host_part.unsafe_ptr(), src_buf=self.sq_part_dev
        )
        # Load-bearing, for the reason `magnitude_sums` states: the
        # destination is pinned, and on Metal a copy into pinned memory is
        # asynchronous. Reading without this passes on a small fixture and
        # corrupts a real fit.
        ctx.synchronize()
        return sum_sq_partials(self.sq_host_part.unsafe_ptr())

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
# CatBoost's MVS bootstrap, solved on the device.
# ---------------------------------------------------------------------------
#
# Built 2026-08-17 to the design recorded above `sampling.mvs_bootstrap_weights`,
# which is where the reasoning lives; this header states only what the code
# does and what it may be claimed to be.
#
# WHY IT EXISTS. An MVS bundle routes a GPU fit to the host-gradient arm
# (`gpu_fused_round.ROUND_MVS_HOST_MAGNITUDES`), because MVS solves its keep
# threshold from the round's own per-row gradient magnitudes and the device
# round does not hand those to the host. That arm computes every round's
# derivatives on the host in Float64 while the tree grows on the device, and
# it was measured on 2026-08-17 to cost 1.62x against the same symmetric fit
# with no sampler -- the single largest named line item in the CatBoost-mode
# default set, with every other member of that set inside a tenth. Solving the
# threshold here is what removes it.
#
# **IT IS NOT BIT-IDENTICAL TO THE HOST DRAW AND MUST NEVER BE CALLED ONE.**
# Five separate reasons, none of which is a rounding to be tightened away:
#
#   1. The host solves over magnitudes built from FLOAT64 derivatives; a device
#      round holds Float32 ones. The magnitudes differ before any solve runs.
#   2. `sqrt(g * g + lambda)` is written unfused here on purpose, but a
#      compiler is free to contract it into an fma anyway, and the host
#      expression and this one are not the same source text.
#   3. The host sums the small side in pivot-partition order; this sums it in
#      grid-stride order and folds it through a shared-memory tree. Float
#      addition is not associative.
#   4. The threshold is solved in Float32 here and in Float64 there, so a row
#      within an ulp of the threshold can land on either side.
#   5. `MVS_PROBABILITY_EPS_F32` is not `_MVS_PROBABILITY_EPS`.
#
# So this is EQUIVALENT IN DISTRIBUTION, not exact. `LANE_RULES` rule 5's
# same-session default flip does NOT reach it, it takes the accuracy gate in
# `docs/design/ACCURACY_BUDGET.md`, and it is therefore behind a DEFAULT-OFF
# switch (`train_gpu.device_mvs_enabled`, `MOJOTREES_GPU_MVS_DEVICE=1`). The
# precedent is `gpu_fused_round.ROUND_GOSS_RANK_PRECISION`, which gates on
# exactly this class of difference: ranking Float32 scores can put a different
# row across a threshold.
#
# WHAT *IS* EXACT, stated separately so the two are not blurred together.
#
#   - The SOLVE IS NOT AN APPROXIMATION. It is not a bisection to a tolerance
#     and it is not a quantile read off a histogram. `_mvs_threshold_kernel`
#     iterates `mu <- (sum of g below mu) / (S - count of g at or above mu)`
#     from `mu_0 = (sum of all g) / S` and stops when the partition repeats.
#     A repeated partition IS the fixed point: at that point `mu` solves
#     `sum_r min(1, g_r / mu) = S` exactly for the partition it induces, which
#     is the same equation and the same partition `sampling._mvs_threshold`
#     lands on by pivoting. The two differ in arithmetic, not in what they
#     compute.
#   - The KEEP DRAW IS THE HOST'S DRAW, given the same `p`. Row `r` reads
#     `splitmix64(stream + r) >> 11`, the identical 53-bit integer
#     `rng.uniform` divides by `2^53`, and `_mvs_keep_draw` compares it against
#     `p * 2^53` formed in exact integer arithmetic from the Float32 `p`. There
#     is no device RNG state, no per-thread seeding, and no second uniform: the
#     decision is a pure function of `(seed, tree_index, row)`, so it is
#     independent of worker count, of block size, of grid shape, and of how
#     many other rows were drawn. **CatBoost's own device MVS does not have
#     this property** -- `MvsBootstrapRadixSortImpl` seeds per thread and
#     advances -- and it is the one thing this implementation keeps that theirs
#     gives up.
#
# WHAT IT DELIBERATELY DOES NOT DO: COMPACT. A dropped row keeps its slot with
# weight 0, which the derivative kernel turns into an exactly zero gradient and
# an exactly zero hessian, so it contributes to no histogram. The host draw
# instead hands `mvs_kept_rows` to `GpuActiveRows` as a bag. The two agree on
# every sum and disagree on one thing: `min_data_in_leaf` counts ROWS, not
# mass, so an uncompacted device draw and a compacted host draw can disagree
# about whether a leaf is legal. At the shipped symmetric `min_data_in_leaf`
# of 1 that binds only on an empty leaf. It is not inert in general.
#
# THE CATBOOST CITATION, and what could not be verified. The comments in
# `sampling.mojo` attribute a CUDA MVS bootstrap to
# `cuda/cuda_util/kernel/mvs.cu::MvsBootstrapRadixSortImpl` with an 8192 block
# size. **There is no CatBoost source in this checkout to verify that
# against.** `.pixi/envs/bench` carries the `catboost` 1.2.10 *wheel*, which is
# a compiled binary and no `.cu`, `.cpp` or `.h` file; a repository-wide search
# for `mvs.cu` finds only our own two comments quoting it. Those attributions
# were made by a lane that had the source open and are left standing as that
# lane's record; nothing in THIS file depends on them, and no claim here should
# be read as verified from CatBoost's tree. The only CatBoost fact this code
# needs is the 8192 block size, and that one is independently pinned by
# `sampling.MVS_BLOCK_SIZE`, whose own comment cites `TMvsSampler::BlockSize`
# on the CPU side.

# Threads per threadgroup in the threshold solve. Fixed rather than derived,
# for `SUM_THREADS`'s reason: the shared-memory tree reduction needs a
# compile-time size. One threadgroup handles one whole `MVS_BLOCK_SIZE` block
# and grid-strides over it, so this is independent of the block size and of the
# row count.
comptime MVS_SOLVE_THREADS = 256

# Hard ceiling on the fixed-point iteration.
#
# **THE ITERATION COUNT IS DATA DEPENDENT AND THIS IS THE ANSWER TO THAT**, the
# one open risk the design block in sampling.mojo named. The iterate is
# monotone decreasing and the large set it induces is monotone growing, so the
# sequence of partitions is nested and strictly grows until it stops; the loop
# can therefore run at most once per distinct magnitude in the block, and in
# practice stops in single digits. Sixty-four is slack by construction, not a
# tuned number.
#
# What happens if it is ever hit, said plainly because a silent cap would be a
# silently different sampler: the kernel keeps the last iterate, which is an
# UPPER bound on the true threshold (the sequence decreases toward it), so a
# capped block keeps slightly fewer rows than it should rather than producing
# nonsense. `GpuMvsSampler.iterations` is what shows whether the cap was
# approached; a test that wants to prove the bound slack reads it.
comptime MVS_SOLVE_MAX_ITERS = 64

# Largest threshold the weight kernel will divide by. Anything above this is
# an infinity produced by a degenerate quotient, and it is rejected on the same
# grounds `sampling._mvs_threshold_is_usable` rejects a non-finite one.
#
# A comparison rather than an `isfinite` call, so the kernel needs nothing from
# `std.math` beyond `sqrt`: `inf > 3.0e38` is true and every finite Float32
# below the maximum is not.
comptime MVS_MU_MAX = Float32(3.0e38)


def _mvs_magnitude_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    mag: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    lam: Float32,
):
    """`thresholdCandidates[r] = sqrt(lambda + grad_r^2)`, one thread per row.

    The device image of the magnitude loop in `sampling.mvs_bootstrap_weights`,
    at `n_outputs == 1`, which is the only shape `GpuMvsSampler` admits.

    `g * g + lam` is written as a separate multiply and add, deliberately and
    for the reason the host loop's comment gives: CatBoost's own kernel writes
    `sqrtf(fmaf(d, d, lambda))` and that is a different number in the last
    bits. **This cannot be guaranteed** -- a compiler is free to contract the
    expression anyway -- and that is one of the five reasons listed in the
    section header for why the device draw is not the host draw.

    Written once per round into a plane of its own rather than recomputed,
    which is the same trade the host loop takes and for a stronger reason
    here: the solve reads the magnitudes once per iteration, so recomputing
    them would multiply the square roots by the iteration count.
    """
    var r = Int(global_idx.x)
    if r >= Int(n_rows):
        return
    var g = grad[unsafe_offset=r][0]
    var sq = g * g
    mag[unsafe_offset=r] = sqrt(sq + lam)


def _mvs_restore_kernel(
    base: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """Put the user's own `sample_weight` back into the round's weight plane.

    **THIS IS NOT TIDYING AND LEAVING IT OUT IS A SILENT WRONG MODEL.** The
    weight plane is `GpuMvsSampler`'s output, so at the top of round `i` it
    still holds round `i - 1`'s draw. The pre-pass that gives the draw its
    magnitudes runs the derivative kernel, which multiplies by that plane, so
    without this the round would solve its threshold over derivatives already
    scaled by the previous tree's draw -- and every row dropped once would
    carry an exactly zero gradient forever after and never be drawn again. The
    fit would train on a monotonically shrinking row set and report nothing.

    One launch and `4 * n_rows` of device-to-device traffic per tree, which is
    a fifth of what the pre-pass beside it costs.
    """
    var r = Int(global_idx.x)
    if r >= Int(n_rows):
        return
    weight[unsafe_offset=r] = base[unsafe_offset=r][0]


def _mvs_threshold_kernel(
    mag: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    iters: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    subsample: Float32,
):
    """One threadgroup solves one `MVS_BLOCK_SIZE` block's threshold.

    The equation is `sampling._mvs_threshold`'s:
    `sum_i min(1, g_i / mu) == subsample * m` over the block's `m` rows. The
    method is not: this iterates rather than pivoting, because a pivot walk
    permutes its input and a threadgroup cannot permute 8192 Float32 without
    32 KB of shared memory it does not have on this device.

    THE ITERATION, and why its fixed point is the exact solution rather than an
    approximation of it. Let `L(t) = {i : g_i >= t}` and
    `S_small(t) = sum of g_i over the complement`. If `mu` were known, it would
    satisfy `mu = S_small(mu) / (target - |L(mu)|)` -- that is the equation
    restated, with the saturated rows contributing 1 apiece. So iterate exactly
    that map from `t_0 = total / target`, which is an upper bound because
    `min(1, x) <= x` makes `F(t_0) <= target` and `F` is decreasing. Each step
    lowers `t`, so `L` grows, so the partitions are NESTED and the count
    `|L|` alone identifies the partition. When the count repeats, the map has
    reproduced its own input partition, and the `mu` computed from that
    partition solves the equation on it. That is a fixed point, not a
    tolerance, and no epsilon appears anywhere in this kernel.

    DEGENERATE CASES, matched to the host's guard rather than invented here.
    `target - |L|` at or below zero means the block already saturates its own
    sample size, and a NaN magnitude poisons the sums; both write a threshold
    the weight kernel rejects, and rejection there means "keep this block
    whole at weight 1", which is `sampling.mvs_bootstrap_weights`'s
    `blocks_guarded` branch. CatBoost drops every row instead; we do not, and
    that difference predates this kernel.

    Every barrier is reached by every thread. The two grid-stride sweeps have
    per-thread trip counts but contain no barrier; the tree reductions have a
    uniform trip count; and the outer loop's exit conditions are read out of
    shared memory or off loop counters, so they are the same for every thread
    in the group.
    """
    # Declared first, before any conditional return, which is where
    # `_abs_sum_kernel` declares its own and is the shape to copy: threadgroup
    # memory is a static allocation and a reader should not have to work out
    # whether a branch above it changes that. 2 KB, well inside the 32 KB an
    # Apple threadgroup gets -- and the reason the block's 8192 magnitudes are
    # NOT cached here is that they would be exactly 32 KB on their own, with
    # nothing left for the reduction.
    var ss = stack_allocation[
        MVS_SOLVE_THREADS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sn = stack_allocation[
        MVS_SOLVE_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var nr = Int(n_rows)
    var begin = b * MVS_BLOCK_SIZE
    if begin >= nr:
        # Uniform across the whole threadgroup, so no barrier is skipped by
        # some threads and reached by others.
        return
    var end = begin + MVS_BLOCK_SIZE
    if end > nr:
        end = nr
    var m = end - begin

    # `t_0 = total / target`: one sweep and one tree reduction.
    var acc = Float32(0.0)
    var i = begin + tid
    while i < end:
        acc += mag[unsafe_offset=i][0]
        i += MVS_SOLVE_THREADS
    ss[unsafe_offset=tid] = acc
    barrier()
    var active = MVS_SOLVE_THREADS // 2
    while active > 0:
        if tid < active:
            ss[unsafe_offset=tid] = (
                ss[unsafe_offset=tid][0] + ss[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2
    var total = ss[unsafe_offset=0][0]
    barrier()

    var target = subsample * Float32(m)
    var t = Float32(0.0)
    if target > 0.0:
        t = total / target

    var prev_large = Int32(-1)
    var used = 0
    while used < MVS_SOLVE_MAX_ITERS:
        var s_acc = Float32(0.0)
        var n_acc = Int32(0)
        var j = begin + tid
        while j < end:
            var v = mag[unsafe_offset=j][0]
            # `>=` puts an exact tie on the large side, which is the side the
            # host's keep rule puts it on too (`if g > mu: p = 1` leaves a tie
            # at `p = g / mu == 1`). A NaN fails both comparisons and falls
            # into the small sum, which is how it reaches the guard.
            if v >= t:
                n_acc += 1
            else:
                s_acc += v
            j += MVS_SOLVE_THREADS
        ss[unsafe_offset=tid] = s_acc
        sn[unsafe_offset=tid] = n_acc
        barrier()
        var a = MVS_SOLVE_THREADS // 2
        while a > 0:
            if tid < a:
                ss[unsafe_offset=tid] = (
                    ss[unsafe_offset=tid][0] + ss[unsafe_offset = tid + a][0]
                )
                sn[unsafe_offset=tid] = (
                    sn[unsafe_offset=tid][0] + sn[unsafe_offset = tid + a][0]
                )
            barrier()
            a //= 2
        var sum_small = ss[unsafe_offset=0][0]
        var n_large = sn[unsafe_offset=0][0]
        barrier()
        used += 1
        if n_large == prev_large:
            # The map reproduced its own partition: `t` already solves the
            # equation on it. Stop, and keep `t` as it stands.
            break
        prev_large = n_large
        var denom = target - Float32(Int(n_large))
        if not (denom > 0.0):
            t = Float32(0.0)
            break
        t = sum_small / denom

    if tid == 0:
        mu[unsafe_offset=b] = t
        iters[unsafe_offset=b] = Int32(used)


@always_inline
def _mvs_keep_draw(counter: UInt64, p: Float32) -> Bool:
    """The host's keep test, decided in integers so it needs no Float64.

    `sampling.mvs_bootstrap_weights` asks `uniform(stream + r) < p`, and
    `rng.uniform` is `Float64(splitmix64(counter) >> 11) * 2^-53`. That left
    side is EXACT: a 53-bit integer converts to Float64 without rounding and
    scaling by a power of two is exact. So the host's test is, in exact real
    arithmetic, `u < p * 2^53` for the integer `u`, and this evaluates that
    same inequality without ever forming a Float64 -- which matters because
    Apple GPUs have none.

    `p` is a normal Float32 in `(0, 1)` by the caller's guards, so
    `p = m * 2^(e - 23)` with `m` a 24-bit integer, and `p * 2^53` is
    `m * 2^(e + 30)`. Left shift when that exponent is non-negative; when it is
    negative, compare `u * 2^k < m` instead, which is the same inequality
    cleared of its denominator. Both sides are bounded well inside 64 bits
    under the caller's guards, and the two early rejections below are what
    keep them bounded when they are not.

    The counter is `stream + r` and nothing advances, so this is a pure
    function of `(seed, tree_index, row)` and reproduces on any grid, at any
    worker count, on either backend. That is the per-row reproducibility the
    host draw has and CatBoost's device draw gives up.
    """
    var u = splitmix64(counter) >> 11
    var bits = bitcast[DType.uint32, 1](p)
    var ex = Int((bits >> 23) & 0xFF)
    if ex == 0:
        # Subnormal: below 2^-126, far under any epsilon the caller admits.
        return False
    var m = UInt64(Int((bits & 0x7FFFFF) | 0x800000))
    var shift = ex - 127 + 30
    if shift >= 0:
        if shift >= 40:
            # `p * 2^53 >= 2^63 > u`. Unreachable for `p < 1`; kept so the
            # shift below can never overflow whatever the caller passes.
            return True
        return u < (m << UInt64(shift))
    var k = -shift
    if u == 0:
        # `p * 2^53 > 0`, so the strict inequality holds.
        return True
    if k >= 40:
        return False
    if u >= (UInt64(1) << 24):
        # `p * 2^53 < 2^24 <= u`.
        return False
    return (u << UInt64(k)) < m


def _mvs_weight_kernel(
    mag: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    base: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    s0: Int32,
    s1: Int32,
    s2: Int32,
    s3: Int32,
):
    """One thread per row: the keep decision, the amplification, and the
    product with the user's own `sample_weight`.

    This is `sampling.mvs_bootstrap_weights`'s weight loop followed by
    `refresh_mvs_bootstrap`'s `weights[r] * base_weight[r]`, in one pass. The
    product is formed in that order because that is the order the host forms
    it in and the two factors do not commute in floating point.

    The counter stream arrives as four 16-bit limbs rather than as one UInt64,
    and that is not fussiness. Every kernel in this package passes its scalars
    as `Int32` or `Float32`, and no kernel here has ever taken a 64-bit scalar
    argument; sixteen bits at a time is a split that is certainly a
    non-negative `Int32` on every backend, and
    reassembling it here costs three shifts and three ors once per row against
    a `splitmix64` that costs more than that. `GpuMvsSampler.draw` is the one
    place the split is made.
    """
    var r = Int(global_idx.x)
    if r >= Int(n_rows):
        return
    var b = r >> MVS_BLOCK_SHIFT
    var t = mu[unsafe_offset=b][0]
    var w0 = base[unsafe_offset=r][0]
    if not (t > 0.0) or t > MVS_MU_MAX:
        # `sampling._mvs_threshold_is_usable` said no: keep the block whole.
        # `not (t > 0.0)` covers NaN and the negative zero the undershoot
        # branch can produce, exactly as the host guard does.
        weight[unsafe_offset=r] = w0
        return
    var g = mag[unsafe_offset=r][0]
    var p = Float32(1.0)
    if not (g > t):
        p = g / t
    var w = Float32(0.0)
    if p > MVS_PROBABILITY_EPS_F32:
        if p >= 1.0:
            # Certain keep, and no draw is burned: the stream is keyed by row,
            # so skipping a row's draw cannot shift any other row's.
            w = Float32(1.0)
        else:
            var stream = (
                (UInt64(Int(s3) & 0xFFFF) << UInt64(48))
                | (UInt64(Int(s2) & 0xFFFF) << UInt64(32))
                | (UInt64(Int(s1) & 0xFFFF) << UInt64(16))
                | UInt64(Int(s0) & 0xFFFF)
            )
            if _mvs_keep_draw(stream + UInt64(r), p):
                w = Float32(1.0) / p
    weight[unsafe_offset=r] = w * w0


struct GpuMvsSampler(Movable):
    """CatBoost's MVS draw, solved and applied entirely on the device.

    One per fit, constructed beside the `GpuObjectiveState` and from the same
    `DeviceContext`; call `draw` once per tree, before the round's derivatives
    are filled for the histogram.

    HOW IT LANDS IN A ROUND, because the ordering is the whole design and it is
    not obvious. The draw needs the round's gradient magnitudes, and the
    round's fixed-point histogram scale is derived from the magnitude sums of
    whatever the gradient plane holds when `histogram_gpu._refresh_scales`
    runs. Those two want opposite orders. Applying the draw to the gradient
    plane AFTER `fill_gradients_device` would leave the scale derived from an
    unsampled plane and, worse, would leave the window overflow check
    measuring a plane no histogram was ever built from -- a silent wrong
    answer rather than a loud one. So the draw goes into the WEIGHT plane
    instead, which the derivative kernel already multiplies both derivatives
    by, and the trainer's round reads:

        sampler.restore_base_weights(...)  # undo the PREVIOUS tree's draw
        state.fill_grad_hess(...)          # unsampled, magnitudes only
        sampler.draw(...)                  # writes state.weight_dev
        builder.fill_gradients_device(...) # sampled, and the scale with it

    All four steps are required and the first is the one that looks optional
    and is not: the weight plane is this sampler's output, so without the
    restore the pre-pass would read the previous tree's draw and every dropped
    row would stay dropped for the rest of the fit.

    One extra derivative pass and one copy kernel per round, and NO extra round
    trip. The extra pass would disappear if `fill_gradients_device` had a seam
    between its fill and its scale refresh; it does not, and adding one is an
    edit to histogram_gpu.mojo rather than to this file.

    WHAT IT OWNS. Four device planes: the magnitudes, the per-block threshold,
    the per-block iteration count, and a private copy of the user's own
    `sample_weight` -- private because `state.weight_dev` is this sampler's
    OUTPUT every tree and would otherwise be squared into itself by the second
    tree. `4 * n_rows` bytes twice, plus a few bytes per 8192 rows.

    Single output only. A multiclass or multi-target state has `n_outputs`
    derivatives per row and a magnitude that reduces across them, and the
    device round refuses a bootstrap on those shapes for reasons that predate
    this file; rather than half-implement it, `draw` raises.
    """

    var mag_dev: DeviceBuffer[DType.float32]
    var mu_dev: DeviceBuffer[DType.float32]
    var iter_dev: DeviceBuffer[DType.int32]
    var base_dev: DeviceBuffer[DType.float32]
    """The user's `sample_weight`, or ones, uploaded once at construction.

    Read every tree and never written, so the round's weight plane is always
    `draw(tree) * sample_weight` and never `draw(tree) * draw(tree - 1) *
    sample_weight`. This is the buffer whose absence would make an MVS fit
    silently collapse its own weights over a hundred rounds."""
    var host_iter: HostBuffer[DType.int32]
    var n_rows: Int
    var n_blocks: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        base_weight: List[Float64],
    ) raises:
        """Allocate the planes and upload the base weights.

        `base_weight` is the user's own `sample_weight`, or empty for the
        unweighted convention this package uses everywhere, in which case the
        plane is filled with ones. It is uploaded here rather than copied off
        `state.weight_dev` because a device-to-device copy of a plane this
        sampler is about to overwrite is the kind of aliasing that reads
        correct and is not.
        """
        if n_rows < 1:
            raise Error("GpuMvsSampler needs at least one row")
        if (1 << MVS_BLOCK_SHIFT) != MVS_BLOCK_SIZE:
            raise Error(
                "MVS_BLOCK_SHIFT and MVS_BLOCK_SIZE have drifted apart; the"
                " device draw maps a row to its block with a shift and would"
                " solve a different threshold per row than the host draw"
            )
        if len(base_weight) != 0 and len(base_weight) != n_rows:
            raise Error("sample_weight length must match the row count")
        self.n_rows = n_rows
        self.n_blocks = (n_rows + MVS_BLOCK_SIZE - 1) // MVS_BLOCK_SIZE
        self.mag_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.base_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        self.mu_dev = ctx.enqueue_create_buffer[DType.float32](self.n_blocks)
        self.iter_dev = ctx.enqueue_create_buffer[DType.int32](self.n_blocks)
        self.host_iter = ctx.enqueue_create_host_buffer[DType.int32](
            self.n_blocks
        )
        # Validate before the mapping opens, so a bad weight raises without
        # leaving a half-written plane behind it.
        if len(base_weight) != 0:
            for r in range(n_rows):
                if not isfinite(base_weight[r]) or base_weight[r] < 0.0:
                    raise Error("sample_weight must be finite and nonnegative")
        # Written through the mapping rather than staged, which is what
        # `GpuObjectiveState.__init__` does with its own one-time uploads and
        # for the same reason: this runs once per fit and the mapping is the
        # shorter path. A temporary `HostBuffer` would be the other way to do
        # it and is the wrong one -- the buffer's last use would be the
        # `unsafe_ptr()` call, so nothing would keep it alive across the copy
        # that reads through that pointer.
        with self.base_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            if len(base_weight) == 0:
                for r in range(n_rows):
                    dst.unsafe_store(r, Float32(1.0))
            else:
                for r in range(n_rows):
                    dst.unsafe_store(r, Float32(base_weight[r]))

    def restore_base_weights(
        mut self, ctx: DeviceContext, mut state: GpuObjectiveState
    ) raises:
        """Undo the previous tree's draw, leaving the plane at the user's own
        `sample_weight`.

        **Call this immediately before the pre-pass that fills the gradients
        the draw will read**, every round, including the first. `draw`'s own
        docstring says why; `_mvs_restore_kernel`'s says what it costs if it is
        skipped, which is a fit that trains on a row set that can only shrink.

        Separate from `draw` rather than folded into it because the two land on
        opposite sides of the caller's derivative fill, and a method that had
        to be called twice with a fill in between would be one method with two
        meanings.
        """
        if state.n_rows != self.n_rows:
            raise Error(
                "GpuMvsSampler was built for a different row count than this"
                " objective state carries"
            )
        if not state.weighted:
            raise Error(
                "the device MVS draw writes into the objective state's weight"
                " plane, and this state still carries the unweighted"
                " placeholder; call refresh_weights once before the round"
                " loop so the plane is allocated at full width"
            )
        comptime if not has_accelerator():
            raise Error(
                "the device MVS draw needs an accelerator; this build has none"
            )
        else:
            ctx.enqueue_function[_mvs_restore_kernel](
                self.base_dev.unsafe_ptr(),
                state.weight_dev.unsafe_ptr(),
                Int32(self.n_rows),
                grid_dim=state._row_blocks(),
                block_dim=state.block_threads,
            )

    def draw(
        mut self,
        ctx: DeviceContext,
        mut state: GpuObjectiveState,
        mut grad_dev: DeviceBuffer[DType.float32],
        subsample: Float64,
        lam: Float64,
        stream: UInt64,
    ) raises:
        """One tree's draw, written into `state.weight_dev`.

        `grad_dev` must hold THIS round's derivatives already, carrying the
        user's `sample_weight` and NO earlier draw -- in the trainer that is
        the histogram builder's own gradient buffer, filled by
        `GpuObjectiveState.fill_grad_hess` immediately before this call, which
        is itself preceded by `restore_base_weights`. That ordering is a
        correctness condition and not a convention; see the restore kernel.

        `lam` is `MvsBootstrapParams.resolve_reg(auto_lambda)`, resolved by the
        caller because the auto branch reads the previous tree's leaf values
        and neither this file nor sampling.mojo can see the ensemble.

        `stream` is `sampling._mvs_stream(seed, tree_index)`. It is a stream
        START and nothing advances it, which is what makes row `r`'s decision
        a pure function of `(seed, tree_index, r)`.

        Three launches, no readback, no synchronize. The caller's next call
        into the builder is what orders this work against the derivative
        kernel that reads the plane.
        """
        if state.n_rows != self.n_rows:
            raise Error(
                "GpuMvsSampler was built for a different row count than this"
                " objective state carries"
            )
        if state.n_classes != 1 or state.multi_output:
            raise Error(
                "the device MVS draw serves single-output rounds only: a"
                " multiclass or multi-target row has one magnitude per output"
                " and reduces across them, and the device round refuses a"
                " bootstrap on those shapes anyway"
            )
        if not state.weighted:
            raise Error(
                "the device MVS draw writes into the objective state's weight"
                " plane, and this state still carries the unweighted"
                " placeholder; call refresh_weights once before the round"
                " loop so the plane is allocated at full width"
            )
        if not (subsample > 0.0) or subsample > 1.0:
            raise Error("mvs subsample must be in (0, 1]")
        if not (lam >= 0.0) or not isfinite(lam):
            raise Error("mvs lambda must be finite and nonnegative")
        # Guarded 2026-08-17, same reason as `enqueue_abs_sum`: on a build
        # with no accelerator ANY reachable `enqueue_function` elaborates a
        # GPU kernel and fails the compile with "Unknown GPU architecture",
        # whatever the kernel does, and an Apple machine never reproduces it.
        comptime if not has_accelerator():
            raise Error(
                "the device MVS draw needs an accelerator; this build has none"
            )
        else:
            ctx.enqueue_function[_mvs_magnitude_kernel](
                grad_dev.unsafe_ptr(),
                self.mag_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Float32(lam),
                grid_dim=state._row_blocks(),
                block_dim=state.block_threads,
            )
            ctx.enqueue_function[_mvs_threshold_kernel](
                self.mag_dev.unsafe_ptr(),
                self.mu_dev.unsafe_ptr(),
                self.iter_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Float32(subsample),
                grid_dim=self.n_blocks,
                block_dim=MVS_SOLVE_THREADS,
            )
            ctx.enqueue_function[_mvs_weight_kernel](
                self.mag_dev.unsafe_ptr(),
                self.mu_dev.unsafe_ptr(),
                self.base_dev.unsafe_ptr(),
                state.weight_dev.unsafe_ptr(),
                Int32(self.n_rows),
                # `.cast[DType.int32]()` off a masked UInt64, which is the
                # spelling `gpu_split_search._random_score_key_kernel` already
                # uses to move 64-bit words across this boundary in the other
                # direction. Sixteen bits a limb, so every limb is a
                # non-negative Int32 and the kernel's reassembly is the same
                # `UInt64(Int(word) & mask)` that module's host side uses.
                (stream & UInt64(0xFFFF)).cast[DType.int32](),
                ((stream >> UInt64(16)) & UInt64(0xFFFF)).cast[DType.int32](),
                ((stream >> UInt64(32)) & UInt64(0xFFFF)).cast[DType.int32](),
                ((stream >> UInt64(48)) & UInt64(0xFFFF)).cast[DType.int32](),
                grid_dim=state._row_blocks(),
                block_dim=state.block_threads,
            )

    def iterations(mut self, ctx: DeviceContext) raises -> List[Int]:
        """How many fixed-point steps each block's solve took on the last
        `draw`, one entry per `MVS_BLOCK_SIZE` block.

        Not on any training path: it costs a readback and a synchronize, and
        it exists so `MVS_SOLVE_MAX_ITERS` can be shown to be slack rather
        than asserted to be. A test that reads this and finds a block at the
        cap has found a real defect, not a tuning opportunity.
        """
        ctx.enqueue_copy(
            dst_ptr=self.host_iter.unsafe_ptr(), src_buf=self.iter_dev
        )
        ctx.synchronize()
        var out = List[Int](capacity=self.n_blocks)
        var src = self.host_iter.unsafe_ptr()
        for b in range(self.n_blocks):
            out.append(Int(src.unsafe_load(b)))
        return out^


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
                " multiclass trainers refuse the setting by name, and a"
                " multi-target state reaches this refusal through the same"
                " test because its outputs are separate trees with separate"
                " leaf values"
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


# ---------------------------------------------------------------------------
# CatBoost's group-and-pair ranking objectives on the device-resident plane.
# ---------------------------------------------------------------------------
#
# QueryRMSE, PairLogit, and YetiRank. `ranking_pairwise.mojo` is the definition
# -- the derivative formulas in CatBoost's own sign convention, the group and
# pair conventions and where they were read from, the refusals, the weighting
# rule, and the fixed-point arithmetic a ranking gradient profile implies. This
# section is those derivatives on the plane where the raw scores already live,
# and it restates none of that reasoning; it records only what is true of the
# device shape.
#
# WHAT THE GROUPING PLANE IS, AND WHAT IT IS NOT
# ----------------------------------------------
# It is the `n_groups + 1` boundary array `ranking.RankGroups.starts`, uploaded
# once per fit, and **it is not a per-row group id**. Rows of a group are
# contiguous -- that is `RankGroups`'s invariant, and `groups_from_query_ids`
# refuses data that violates it -- so a group is a window and a kernel block
# owns one. No row ever looks its own group up: `_query_rmse_kernel` is indexed
# by group, and the pairwise kernel is indexed by row and never needs the group
# at all, because the pairs were already validated to lie inside one.
#
# A per-row id would have been `4 * n_rows` bytes of upload plus one gather per
# row per round, to answer a question the row's position already answers. The
# only shape in which it would have been necessary is unsorted rows, which is
# the convention the CPU does not have.
#
# WHY THE PAIR PLANE IS AN ADJACENCY AND NOT A PAIR LIST
# ------------------------------------------------------
# **Metal has no floating-point atomic add.** A one-thread-per-pair kernel
# writes into `grad[i]` and `grad[j]` from as many threads as the row appears
# in pairs, which needs an atomic this backend does not have and which would be
# non-deterministic on the backends that do. So the host expands the pair list
# into the per-row CSR `ranking_pairwise.PairAdjacency` and the kernel is one
# thread per row over its own slice: no atomic, no contention, and a fold order
# fixed by the host rather than by the scheduler. Each pair is read twice,
# which is the price.
#
# One device word pair per entry, `PAIR_WORDS` Int32 apiece: the other
# endpoint, and the pair's weight as a Float32 whose *sign* carries whether
# this row was the winner, travelling as its own bit pattern. Same
# reinterpretation `SEG_STEP` uses and for the same reason -- two buffers are
# two `enqueue_copy` calls and on Metal a copy is a full-queue drain whatever
# its byte count (`docs/GPU_PORTABILITY.md` section 6.1, **measured**). A
# `bitcast` between two 32-bit types alters no value in either direction.
comptime PAIR_WORDS = 2
comptime PAIR_OTHER = 0
comptime PAIR_WEIGHT = 1


def _query_rmse_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    starts: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    weighted: Int32,
):
    """One threadgroup per query group: the group's weighted mean residual,
    then every row's derivative against it.

    `ranking_pairwise.query_rmse_grad_hess` is the definition and carries the
    CatBoost correspondence. Here:

        r_i    = target_i - raw_i
        avg    = sum_i w_i r_i / sum_i w_i        (0 when sum_i w_i is 0)
        grad_i = w_i * (avg - r_i)
        hess_i = w_i

    The block sweeps its group twice -- once to reduce, once to write -- rather
    than caching the residuals, because a group has no bounded size and shared
    memory does. Both sweeps are `SUM_THREADS`-strided over the same window in
    the same direction, so the second reads exactly what the first summed.

    **The reduction is inside the block and the barriers are unconditional.**
    Every thread of the block runs the same fixed `SUM_THREADS`-wide tree
    whatever the group's size, so every thread reaches every barrier, and the
    fold order is a compile-time constant. That makes the sum the same Float32
    for a given group whatever the device schedules -- deterministic run to
    run, in the sense `_abs_sum_kernel` already claims. It is **not** the
    host's fold: the reference sums sequentially in ascending row order in
    Float64, so the two agree to Float32 and not to the bit, which is the trade
    every number on this plane makes.

    A group of one is not a special case here and gets no branch. Its single
    thread sums `w_0` and `w_0 r_0`, `avg` comes out exactly `r_0`, and the
    write produces `w_0 * (r_0 - r_0)`, an exact zero in Float32 because it is
    a subtraction of a value from itself. A group whose weights sum to zero
    takes the `sum_w > 0` branch to `avg = 0` and then writes `0 * (...)`,
    which is zero rather than the NaN a division would have produced.

    The shape is chosen for correctness and not for occupancy, and that is
    worth saying rather than leaving to be discovered: a group of ten leaves
    246 of 256 threads idle in both sweeps. A thread-per-group or
    subgroup-per-group variant is the obvious answer for many small groups and
    is **not measured here** -- this is an accuracy lane and it publishes no
    timing.
    """
    var tid = Int(thread_idx.x)
    var q = Int(block_idx.x)
    var sw = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sr = stack_allocation[
        SUM_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var start = Int(starts[unsafe_offset=q][0])
    var stop = Int(starts[unsafe_offset = q + 1][0])

    var acc_w = Float32(0.0)
    var acc_r = Float32(0.0)
    var i = start + tid
    while i < stop:
        var w = Float32(1.0)
        if weighted != 0:
            w = weight[unsafe_offset=i][0]
        acc_w += w
        acc_r += w * (target[unsafe_offset=i][0] - raw[unsafe_offset=i][0])
        i += SUM_THREADS
    sw[unsafe_offset=tid] = acc_w
    sr[unsafe_offset=tid] = acc_r
    barrier()

    var active = SUM_THREADS // 2
    while active > 0:
        if tid < active:
            sw[unsafe_offset=tid] = (
                sw[unsafe_offset=tid][0] + sw[unsafe_offset = tid + active][0]
            )
            sr[unsafe_offset=tid] = (
                sr[unsafe_offset=tid][0] + sr[unsafe_offset = tid + active][0]
            )
        barrier()
        active //= 2

    # The last iteration's barrier is what publishes slot 0 to every thread,
    # so no further synchronization is needed before this read.
    var sum_w = sw[unsafe_offset=0][0]
    var sum_wr = sr[unsafe_offset=0][0]
    var avg = Float32(0.0)
    if sum_w > 0.0:
        avg = sum_wr / sum_w

    var j = start + tid
    while j < stop:
        var w = Float32(1.0)
        if weighted != 0:
            w = weight[unsafe_offset=j][0]
        var resid = target[unsafe_offset=j][0] - raw[unsafe_offset=j][0]
        grad[unsafe_offset=j] = w * (avg - resid)
        hess[unsafe_offset=j] = w
        j += SUM_THREADS


def _pair_logit_kernel(
    raw: MutPointer[Float32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    entries: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
):
    """One thread per row: the row's pairwise-logit derivatives, summed over
    the pairs it takes part in.

    `ranking_pairwise.pairwise_grad_hess` is the definition and carries the
    CatBoost correspondence, and it is the same expression for PairLogit and
    for YetiRank because it is the same expression in CatBoost. Per entry,
    with `d` the winner-minus-loser raw score difference and `w` the pair's
    weight:

        rho     = 1 / (1 + exp(d))
        grad   -= w * rho                  if this row is the winner
        grad   += w * rho                  if this row is the loser
        hess   += w * rho * (1 - rho)      either way

    `rho` is `_dev_sigmoid(-d)`, which is `ranking._pair_sigmoid(d, 1.0)`
    branch for branch: at `d > 0` both evaluate `e = exp(-d); e / (1 + e)`, at
    `d < 0` both evaluate `e = exp(d); 1 / (1 + e)`, and at `d == 0` both reach
    the first arm and return exactly `1/2`. That is why the host reference
    imports the CPU's function instead of restating it and why this kernel
    calls the module's existing sigmoid instead of adding a second one: the
    overflow guard has one definition per backend and the two are the same
    expression.

    **No atomic, and no thread writes a row another thread writes.** The
    accumulators are registers, the slice is the row's own, and the two stores
    at the end are the only writes. That is the whole reason the pair plane is
    an adjacency; the block comment above gives it.

    A row with an empty slice -- every row of a singleton group, and any row no
    pair mentions -- exits the loop having touched nothing and stores exactly
    zero into both planes. There is no division in this kernel, so there is
    nothing a degenerate group could divide by.

    The trip count is the row's pair count and varies across a threadgroup, so
    this kernel diverges where `_grad_hess_kernel` does not. It is the shape
    the absence of a float atomic leaves, and the alternative is not a faster
    kernel, it is a wrong one.
    """
    var r = global_idx.x
    if r >= Int(n_rows):
        return
    var lo = Int(offsets[unsafe_offset=r][0])
    var hi = Int(offsets[unsafe_offset = r + 1][0])
    var raw_r = raw[unsafe_offset=r][0]
    var g = Float32(0.0)
    var h = Float32(0.0)
    for e in range(lo, hi):
        var base = e * PAIR_WORDS
        var o = Int(entries[unsafe_offset = base + PAIR_OTHER][0])
        var sw = bitcast[DType.float32, 1](
            entries[unsafe_offset = base + PAIR_WEIGHT][0]
        )
        var raw_o = raw[unsafe_offset=o][0]
        var w = abs(sw)
        var d = (raw_r - raw_o) if sw > 0.0 else (raw_o - raw_r)
        var rho = _dev_sigmoid(-d)
        var contrib = w * rho
        if sw > 0.0:
            g -= contrib
        else:
            g += contrib
        h += w * rho * (1.0 - rho)
    grad[unsafe_offset=r] = g
    hess[unsafe_offset=r] = h


struct GpuRankingState(Movable):
    """Device-resident query boundaries and pair adjacency for one ranking fit.

    Construct once per fit from the same `DeviceContext` as the
    `GpuObjectiveState` whose labels, weights and raw scores it reads, and
    beside it; `fill_grad_hess` takes that state rather than duplicating any of
    it, which is the arrangement `GpuLeafEstimator` already has.

    Two planes, with different lifetimes, and the difference is the whole of
    what separates the three objectives on this side:

    - The **group boundaries** are a property of the *dataset*. Uploaded once,
      in the constructor, and never again. Rows of a group are contiguous
      (`ranking.RankGroups`), so this is an `n_groups + 1` boundary array and
      not a per-row column; the block comment above argues why that is not
      merely smaller but structurally different.
    - The **pair adjacency** is a property of the *pair set*. For PairLogit
      that is the fit's, uploaded once; for YetiRank it is the round's, and
      `refresh_pairs` is the per-round upload, the twin of
      `GpuObjectiveState.refresh_weights` and for the same kind of reason.

    QueryRMSE reads only the first plane, and this state allocates the second
    at one word until something needs it, so a QueryRMSE fit pays no pair
    memory at all.

    What this state deliberately does not hold: the raw scores, the labels, the
    weights, and the gradient buffers. All four belong to `GpuObjectiveState`
    and `GpuHistogramBuilder` and are read through them, so a ranking round
    writes into exactly the buffers the histogram kernels read, with nothing
    crossing the boundary, on the same terms as every other device objective.
    """

    var starts_dev: DeviceBuffer[DType.int32]
    """`n_groups + 1` ascending row boundaries. The grouping plane."""
    var off_dev: DeviceBuffer[DType.int32]
    """`n_rows + 1` CSR offsets into `pair_dev`, or a placeholder."""
    var pair_dev: DeviceBuffer[DType.int32]
    """`PAIR_WORDS` Int32 per adjacency entry, or a placeholder."""
    var stage_off: HostBuffer[DType.int32]
    var stage_pair: HostBuffer[DType.int32]
    """Pinned staging for the two pair planes, on the same grounds as
    `GpuObjectiveState.stage_value`: `map_to_host` copies in both directions on
    every use, so a per-round upload goes through a one-way copy instead.
    Allocated by the first `refresh_pairs`, never at construction, so a
    QueryRMSE fit pays nothing for a buffer it will not write."""

    var n_rows: Int
    var n_groups: Int
    var pair_capacity: Int
    """Adjacency entries `pair_dev` and `stage_pair` are sized for. Zero while
    they are placeholders. One number rather than a second flag, so the buffer
    and the claim about it cannot disagree."""
    var n_entries: Int
    """Entries currently staged. Zero when no pair set has been uploaded."""
    var pairs_round: Int
    """The round index the staged pairs were generated for, or -1 when none
    have been. Read only by `_check_pair_plane`, which is what makes a YetiRank
    round that forgot to redraw an error instead of a model trained against an
    ordering it has already left behind."""
    var has_pairs: Bool
    var block_threads: Int

    def __init__(out self, ctx: DeviceContext, groups: RankGroups) raises:
        """Upload the query boundaries, which never change again.

        `groups` is validated by `ranking.check_groups` -- the CPU's validator,
        not a second one -- so a boundary array that does not start at zero,
        does not ascend strictly, or does not end at `n_rows` is refused here
        rather than producing a block that sweeps the wrong window.
        """
        check_groups(groups, groups.n_rows)
        if groups.n_rows < 1:
            raise Error("ranking objectives require at least one row")
        self.n_rows = groups.n_rows
        self.n_groups = groups.n_queries()
        self.block_threads = derive_block_threads(query_device_caps(ctx))
        self.pair_capacity = 0
        self.n_entries = 0
        self.pairs_round = -1
        self.has_pairs = False

        self.starts_dev = ctx.enqueue_create_buffer[DType.int32](
            self.n_groups + 1
        )
        # Placeholders: zero-length device buffers are not portable, and a
        # QueryRMSE fit never grows these.
        self.off_dev = ctx.enqueue_create_buffer[DType.int32](1)
        self.pair_dev = ctx.enqueue_create_buffer[DType.int32](1)
        self.stage_off = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.stage_pair = ctx.enqueue_create_host_buffer[DType.int32](1)

        # One-time upload, written through the mapping rather than staged, on
        # the same grounds as `GpuObjectiveState`'s label upload: it runs once
        # per session and the mapping is the shorter path.
        with self.starts_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for q in range(self.n_groups + 1):
                dst.unsafe_store(q, Int32(groups.starts[q]))

    def _row_blocks(self) -> Int:
        return (self.n_rows + self.block_threads - 1) // self.block_threads

    def refresh_pairs(
        mut self,
        ctx: DeviceContext,
        adjacency: PairAdjacency,
        round_index: Int = 0,
    ) raises:
        """Replace the device-resident pair adjacency.

        Called once per fit for PairLogit, whose pairs the caller supplies and
        which do not move, and once per *round* for YetiRank, whose pairs are
        redrawn against the current scores. `round_index` records which round
        the staged pairs belong to; `fill_grad_hess` refuses a YetiRank round
        whose pairs carry a different one, which is the difference between
        training YetiRank and training PairLogit on a stale draw.

        Cost and cadence
        ----------------
        Two `enqueue_copy` calls -- the offsets and the interleaved entries --
        plus one drain, per upload. Two rather than three because the entry's
        weight travels inside the entry record as a reinterpreted Float32; the
        block comment above `PAIR_WORDS` argues that choice. On the PairLogit
        path this happens once for the whole fit. On the YetiRank path it is
        per round, at `4 * (n_rows + 1) + 8 * entries` bytes.

        The drain is the same trade `refresh_weights` makes and is argued
        there: on Metal `enqueue_copy` is itself a synchronous full-queue drain
        (`docs/GPU_PORTABILITY.md` section 6.1), so the explicit synchronize
        costs an ordering point rather than a wait there, and on a backend
        whose copies are genuinely asynchronous it is the whole guarantee that
        a kernel still reading the old planes has finished before the buffers
        are dropped and the staging arena is rewritten.

        Validation
        ----------
        Every field is checked against the row count and against the buffer it
        will index, because a malformed adjacency does not crash a kernel, it
        produces a plausible wrong gradient: an offset that does not ascend
        gives a row a negative trip count and silently drops its pairs, and an
        endpoint out of range reads another row's raw score. Weights must be
        finite and nonzero, because zero is the one value the sign encoding
        cannot carry a direction for; `pair_adjacency` drops zero-weight pairs
        rather than emitting them, so an adjacency built by that function
        always passes.
        """
        if adjacency.n_rows != self.n_rows:
            raise Error(
                "pair adjacency and query boundaries disagree on the row count"
            )
        if len(adjacency.offsets) != self.n_rows + 1:
            raise Error("pair adjacency offsets must have n_rows + 1 entries")
        if adjacency.offsets[0] != 0:
            raise Error("pair adjacency offsets must start at 0")
        var total = len(adjacency.other)
        if len(adjacency.signed_weight) != total:
            raise Error(
                "pair adjacency index and weight planes must have equal length"
            )
        if adjacency.offsets[self.n_rows] != total:
            raise Error("pair adjacency offsets must end at the entry count")
        for r in range(self.n_rows):
            if adjacency.offsets[r + 1] < adjacency.offsets[r]:
                raise Error("pair adjacency offsets must be nondecreasing")
        for e in range(total):
            var o = adjacency.other[e]
            if o < 0 or o >= self.n_rows:
                raise Error("pair adjacency endpoint out of range")
            var w = adjacency.signed_weight[e]
            if not isfinite(w) or w == 0.0:
                raise Error(
                    "pair weights must be finite and nonzero; the sign carries"
                    " which endpoint won and zero has no sign"
                )

        # Two hazards, one drain: a kernel still holding a placeholder buffer
        # the growth below drops, and a copy still reading the staging arena
        # the fill below overwrites. See the docstring.
        ctx.synchronize()

        if self.pair_capacity == 0:
            self.off_dev = ctx.enqueue_create_buffer[DType.int32](
                self.n_rows + 1
            )
            self.stage_off = ctx.enqueue_create_host_buffer[DType.int32](
                self.n_rows + 1
            )
        var want = total if total > 0 else 1
        if want > self.pair_capacity:
            self.pair_dev = ctx.enqueue_create_buffer[DType.int32](
                want * PAIR_WORDS
            )
            self.stage_pair = ctx.enqueue_create_host_buffer[DType.int32](
                want * PAIR_WORDS
            )
            self.pair_capacity = want

        var doff = self.stage_off.unsafe_ptr()
        for r in range(self.n_rows + 1):
            doff.unsafe_store(r, Int32(adjacency.offsets[r]))
        var dpair = self.stage_pair.unsafe_ptr()
        for e in range(total):
            var base = e * PAIR_WORDS
            dpair.unsafe_store(base + PAIR_OTHER, Int32(adjacency.other[e]))
            dpair.unsafe_store(
                base + PAIR_WEIGHT,
                bitcast[DType.int32, 1](Float32(adjacency.signed_weight[e])),
            )
        ctx.enqueue_copy(dst_buf=self.off_dev, src_ptr=doff)
        if total > 0:
            ctx.enqueue_copy(dst_buf=self.pair_dev, src_ptr=dpair)
        self.n_entries = total
        self.pairs_round = round_index
        self.has_pairs = True

    def _check_pair_plane(self, kind: Int, round_index: Int) raises:
        """The two refusals a pairwise round can hit, stated once so the
        YetiRank arm and the PairLogit arm cannot drift apart on them."""
        if not rank_kind_is_pairwise(kind):
            return
        if not self.has_pairs:
            raise Error(
                "objective '",
                describe_rank_kind(kind),
                "' is a pairwise loss and no pair plane has been uploaded;"
                " call refresh_pairs, or train with device='cpu'",
            )
        if rank_kind_regenerates_pairs(kind):
            check_yeti_rank_pairs(kind, self.pairs_round == round_index)

    def fill_grad_hess(
        mut self,
        ctx: DeviceContext,
        mut state: GpuObjectiveState,
        kind: Int,
        mut grad_dev: DeviceBuffer[DType.float32],
        mut hess_dev: DeviceBuffer[DType.float32],
        round_index: Int = 0,
    ) raises:
        """Write this round's ranking gradients and hessians into `grad_dev`
        and `hess_dev`, which must be device buffers of at least `n_rows`
        Float32 belonging to the same context.

        In the trainer those are the histogram builder's own buffers, so the
        values the histogram kernels read are the ones these kernels wrote and
        nothing per-row crosses to the host, exactly as
        `GpuObjectiveState.fill_grad_hess` arranges for the built-in
        objectives.

        Every refusal is here and every one names a reason
        --------------------------------------------------
        An unknown kind, a multiclass state, a row-count disagreement,
        uninitialized raw scores, a per-row `sample_weight` on a pairwise kind
        (the weight belongs on the pair -- `check_rank_sample_weight`), a
        pairwise kind with no pair plane, and a YetiRank round whose pairs were
        drawn for a different round. None of them is a silent fallback and none
        of them quietly computes something adjacent: this file's standing rule,
        after `leaf_estimation_iterations` was ignored without comment by every
        GPU entry point for months, is that a device which cannot honour a
        configuration says so and names `device='cpu'`.

        What this does *not* refuse is a group whose rows carry no signal -- a
        singleton group, a perfectly ordered group, a group of zero-weight
        rows. Those are not unsupported configurations, they are configurations
        whose correct gradient is zero, and both kernels produce an exact zero
        for them without a branch. `ranking_pairwise` works each one through.
        """
        check_rank_kind(kind)
        if state.n_classes != 1:
            raise Error(
                "ranking objectives are single-output; a multiclass objective"
                " state cannot serve one. Use device='cpu' for a multi-output"
                " ranking loss"
            )
        if state.n_rows != self.n_rows:
            raise Error("objective state and ranking state disagree on n_rows")
        if not state.has_raw:
            raise Error("call init_raw before filling ranking gradients")
        check_rank_sample_weight(kind, state.weighted)
        self._check_pair_plane(kind, round_index)

        if kind == RANK_QUERY_RMSE:
            ctx.enqueue_function[_query_rmse_kernel](
                state.raw_dev.unsafe_ptr(),
                state.target_dev.unsafe_ptr(),
                state.weight_dev.unsafe_ptr(),
                self.starts_dev.unsafe_ptr(),
                grad_dev.unsafe_ptr(),
                hess_dev.unsafe_ptr(),
                Int32(1) if state.weighted else Int32(0),
                grid_dim=self.n_groups,
                block_dim=SUM_THREADS,
            )
            return

        ctx.enqueue_function[_pair_logit_kernel](
            state.raw_dev.unsafe_ptr(),
            self.off_dev.unsafe_ptr(),
            self.pair_dev.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            Int32(self.n_rows),
            grid_dim=self._row_blocks(),
            block_dim=self.block_threads,
        )
