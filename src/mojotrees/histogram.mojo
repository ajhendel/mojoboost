"""Histogram accumulation.

For each (feature, bin) pair, accumulates the sum of gradients, the sum of
hessians, and the row count. Split finding then scans these fixed-size
histograms instead of the raw data.

The accumulation loops are scatter-bound (bin indices collide), so they use
pointer-based scalar stores; the elementwise kernels (zeroing, sibling
subtraction) are SIMD-vectorized. `SIMD_LANES` is sized above the hardware
width so the compiler can keep several vector operations in flight.

Accumulation parallelizes across features: each feature owns the
`[f * n_bins, (f + 1) * n_bins)` slice of the output, so per-feature workers
never write the same location and need no atomics. Nodes too small to amortize
task-scheduling overhead take the serial path.

Every builder takes an optional list of feature ids (empty means all). Under
feature subsampling only those features are accumulated; the output keeps its
full `n_features * n_bins` shape with the excluded slices left at zero, so no
dataset is copied or re-indexed and sibling subtraction stays exact. The ids
must be distinct: two tasks handed the same feature would write the same
slice, which is the one thing the no-atomics argument above rules out.

Each builder comes in two forms. The plain one allocates and returns a fresh
`Histogram` and is what callers outside tree growth want. The `_into` one
writes a caller-owned buffer, so a grower that visits hundreds of nodes can
recycle a handful of histograms instead of allocating three arrays per node
(see `Histogram.zeroed` / `Histogram.reset`). The two forms run the same
kernels and produce bit-identical results; only the allocation differs.

Three things about the CPU shape of these kernels, all of them scheduling or
memory traffic and none of them arithmetic (see
`handoffs/performance_13_apple_cpu.md` for the mechanism behind each and for
what still needs measuring):

- **Zeroing is fused into the feature pass, not run before it.** A build
  writes every one of the `n_features * n_bins` cells and then accumulates
  into a subset of them. Done as a separate pass, the write streams a buffer
  that at 100 features and 255 bins is around 600 KB out of cache and back
  in; done inside each feature's task it happens on the slice that task is
  about to accumulate into, in parallel with every other task. It is also
  counted in the work estimate, so a small node whose accumulation is tiny
  but whose output is large no longer runs a multi-megabyte write serially.
- **A node's gradients and hessians are gathered once, not once per
  feature.** Accumulating a tree node reads `grad[r]` and `hess[r]` through
  a row id, so each feature pays two indirect loads per row over buffers far
  larger than cache. `build_histogram_subset_into_scratch` gathers them into
  one contiguous `(g, h)` pair buffer first, which turns those two loads per
  feature into two sequential loads per feature and leaves one gather for
  the whole node. The threshold and the scratch reuse are the caller's, via
  the scratch form.
- **N features are accumulated in one inner loop.** Their slices are
  disjoint, so the N read-modify-write chains are independent, and the row id
  and the gradient pair are loaded once for all N. Whether the hardware
  actually overlaps them is a claim for a profiler, not for this comment; the
  traffic arithmetic is not, and it is the reason the width is a ladder rather
  than a constant. Feature-major accumulation walks the node's gradient and
  hessian stream once per group, so at 50 features and a million rows the
  16 MB of (g, h) pairs is streamed 50 times at width 1 (800 MB), 13 times at
  width 4 (208 MB), and 4 times at width 16 (64 MB), against 50 MB of binned
  data that is read once at every width. `apple_cpu_policy.plan_feature_group`
  chooses the width and says what bounds it.

None of that changes a value or an order: each feature still sums its bins
over rows in ascending order inside a single task. The argument is worth
stating precisely, because widening an interleave is exactly the kind of
change that quietly reassociates a Float64 sum, and this one does not.

  A histogram cell `(f, b)` receives one `+= grad[r]` for every row `r` of the
  node whose bin for feature `f` is `b`. Which cells exist, and which rows
  land in each, is fixed by the data. What a schedule can change is the ORDER
  of those additions, and Float64 addition is not associative, so the order is
  part of the result.

  Feature `f` is owned by exactly one group, the group is owned by exactly one
  task, and that task walks `i_row` from 0 to the node's row count in
  ascending order, adding into `f`'s slice as it goes. So the sequence of
  additions into `(f, b)` is the node's rows in ascending order, filtered to
  those that fall in bin `b`. That sentence contains no reference to the group
  width, to which other features share the group, to how many groups a task
  holds, or to how many tasks there are, and that is the whole proof: the same
  sequence of additions in the same order at every width, hence the same
  bytes. It rests on the feature ids being distinct, which is the contract
  stated above, since two copies of one feature would share a slice.

  The two dispatch shapes preserve it in the same way `parallel.mojo` states:
  a group is never split across tasks, and a feature is never split across
  groups.

**Why the group width, and not row blocks.** Widening the group divides both
the gradient traffic and the task count by the same factor, so it trades
memory traffic against parallel slack and neither term can be made large
without shrinking the other. The decomposition that removes the trade is the
obvious one: split a group's rows into blocks, give each block a private
histogram, and fold the partials. Both counts go up together, and it is what
the GPU tiled kernel already does.

It is ruled out here, and the reason is the bit-identity proof above rather
than a preference. A fold over contiguous ascending row blocks gives cell
`(f, b)` the value

    (sum of block 0's rows in b) + (sum of block 1's rows in b) + ...

and Float64 addition is not associative, so that is a *different value* from
adding the same rows one at a time in ascending order. It is not less
accurate, and for many inputs it is more so; it is simply not the same bytes,
and this project's histograms are bit-identical across every backend and every
worker setting by construction rather than by tolerance. The block count would
enter the result, `MOJOTREES_NUM_WORKERS` would stop being a scheduling knob,
and sibling subtraction would stop being exact. Anyone reaching for row blocks
is proposing to give all of that up, and should say so explicitly rather than
discover it by breaking parity. The GPU path escapes the same argument only
because it accumulates in fixed-point Int32, where addition *is* associative;
see `_range_hist_partial_kernel` in `gpu_active_rows.mojo`.

The constant-hessian plane
--------------------------
A cell is three planes, and for four of the built-in objectives one of them
carries no information. `SQUARED_ERROR`, `L1`, `HUBER` and `QUANTILE` all
write the same hessian into every row when the fit has no sample weights, and
that value is exactly 1.0, so the hessian plane is the count plane and storing
both is waste. `objective_has_constant_hessian` is the predicate, and it is
deliberately about the objective rather than about a round's numbers: hessians
that happen to be equal once are not a guarantee, and a weighted or GOSS round
breaks the guarantee while leaving the objective code unchanged.

The elision is a **declaration**, threaded as the trailing `const_hessian`
argument of every builder here and as `GpuActiveRows.set_constant_hessian` on
the device. Nothing infers it, nothing defaults to it, and
`MOJOTREES_CONST_HESSIAN=0` forces every declaration back onto the three-plane
path. When it is on, the accumulation loop and the device kernel stop touching
the hessian plane entirely and the plane is refilled from the count at the end
of the pass, so the assembled `Histogram` still presents `grad`, `hess` and
`count` and no consumer changes.

The declaration is made in exactly one place per backend, and both go through
`boosting.round_has_constant_hessian`, which is where the exclusions this
predicate cannot see (GOSS above all) are applied. On the CPU it travels as
the `const_hessian` argument of `tree.grow_tree_leaves_profiled` and its
sibling entry points, down through `tree._hist_full` and `tree._hist_subset`
to the builders here and to the sibling subtraction. On the device it is
builder state, set once per fit with
`GpuHistogramBuilder.set_constant_hessian`. Every other trainer in the
package -- multiclass, custom objectives, DART, random forest, the sparse and
distributed paths -- leaves the argument at its default of False and runs the
three-plane path that shipped.

Exactness on this backend, which is the whole argument for shipping it. A cell
that received `count` rows accumulated `1.0 + 1.0 + ... + 1.0`, count times, in
Float64. Every partial sum of that series is one of the integers 1, 2, ...,
count, each of which is exactly representable in Float64 while count stays
below 2^53, so no rounding occurs anywhere in the series and the total is
exactly `Float64(count)`. The refill writes `Float64(count)` and is therefore
the identical Float64, not an approximation of it.

Note what the choice of 1.0 buys, because it is not incidental. For a general
constant `h` the series `h + h + ... + h` is *not* `h * count` in floating
point, and reconstructing it would need a multiply next to the additions that
already exist in this file, which is exactly the neighborhood
`docs/NUMERICS.md` says a fusion can appear in and cannot be refused at a
single site. At `h = 1.0` there is no multiply at all: the refill is an
integer-to-float conversion, and no optimizer has anything to contract. The
device path does have a multiply and it is an Int32 one, argued in
`_range_hist_atomic_kernel`.
"""

from std.math import round
from std.sys.info import simd_width_of

from .apple_cpu_policy import (
    cpu_profile,
    derive_accumulation_plan,
    subtract_ops,
    subtract_ops_for_planes,
)
from .binning import BinnedMatrix
from .objective_registry import HUBER, L1, QUANTILE, SQUARED_ERROR
from .parallel import (
    _env_int,
    dispatch_feature_ranges,
    dispatch_features,
    dispatch_rows,
)

comptime SIMD_LANES = 4 * simd_width_of[DType.float64]()


# ---------------------------------------------------------------------------
# The constant-hessian specialization
# ---------------------------------------------------------------------------

comptime CONSTANT_HESSIAN = 1.0
"""The only per-row hessian this package specializes on.

Not a tunable and not a general constant. It is 1.0 because that is the
value four built-in objectives write into every row of `hess` when there
are no sample weights, and because 1.0 is the one constant for which the
reconstruction below is exact in Float64 without a multiply. See
`objective_has_constant_hessian` for the objectives and
`_accumulate_full_at` for the exactness argument.
"""


def objective_has_constant_hessian(objective: Int, weighted: Bool) -> Bool:
    """Whether `objective` *guarantees* a per-row hessian of exactly
    `CONSTANT_HESSIAN` for every row, at every raw score, on every dataset.

    This is a statement about the objective's second derivative, read off
    `boosting._fill_grad_hess_into`, and deliberately not a statement about
    any particular round's numbers. A run whose hessians happen to be equal
    on one round is not the same thing, cannot be relied on for the next
    round, and is not what this returns.

    The four that qualify, and why each is safe. Every one of them ends its
    row body with `hp.unsafe_store(r, w)` and nothing else, where `w` is
    `weights[r]` when the fit has sample weights and the Float64 literal
    `1.0` when it does not:

    - `SQUARED_ERROR`. The loss is `(raw - y)^2 / 2`, second derivative 1.
    - `L1`. The gradient is `sign(raw - y)`; LightGBM (and this package)
      carry the second derivative as 1 so the Newton step reduces to the
      gradient step, and `renew_tree_output` corrects the leaf afterwards.
    - `HUBER`. Quadratic inside the band and linear outside, and the linear
      arm carries the same hessian of 1 for the same reason as L1.
    - `QUANTILE`. The pinball gradient with a hessian of 1, same reason.

    That set is exactly LightGBM's `IsConstantHessian()`, which is
    `return !weights_` on `RegressionL2loss` and is inherited unchanged by
    its L1, Huber, and quantile subclasses. `MAPE` inherits it too in this
    package's shape but overrides the hessian to `w * label_weight(y)`,
    which varies per row, so it is excluded here as it is there.

    Everything else is excluded, and the exclusions matter more than the
    inclusions:

    - **Sample weights of any kind.** The hessian *is* the weight, so a
      weighted fit has a per-row hessian by construction. That covers an
      explicit `sample_weight`, `class_weight` (which becomes one), and any
      other path that folds a weight into `hess`.
    - **GOSS.** `goss.apply_goss_scaling` multiplies the sampled
      small-gradient rows' hessians by an amplification factor and leaves
      the top rows alone, so a GOSS round has two distinct hessian values
      even under squared error. A caller that turns GOSS on must not
      declare a constant hessian, and this function cannot see that it did:
      the objective is still `SQUARED_ERROR`. The declaration is the
      caller's and this is the reason it has to be.
    - Every objective with a curvature that depends on the raw score or the
      label: logistic, cross entropy, Poisson, gamma, tweedie, fair, MAPE,
      and softmax multiclass.
    - `CUSTOM` and `LAMBDARANK`, whose hessians come from a caller-supplied
      callable and from query groups. Neither is a code this function can
      reason about, so both return False.

    Who calls this. `boosting.round_has_constant_hessian` is the single
    binding of it, and it adds the one exclusion this function cannot make
    (GOSS) before answering. Every trainer that declares the specialization
    goes through that function: `boosting._boost_rounds` and
    `boosting.train_with_valid` on the CPU, and the `train_gpu` and
    `train_gpu_with_valid` entry points in `train_gpu.mojo` on the device.
    The multiclass and custom-objective trainers on both backends do not
    call it at all, and their builders keep the three-plane path.

    Calling this directly, at a site that is not one of those, means taking
    on the exclusions in the list above by hand. Prefer the binding.
    """
    if weighted:
        return False
    return (
        objective == SQUARED_ERROR
        or objective == L1
        or objective == HUBER
        or objective == QUANTILE
    )


def const_hessian_allowed() -> Bool:
    """Whether the constant-hessian specialization may run at all.

    `MOJOTREES_CONST_HESSIAN=0` forces every builder back onto the
    three-plane path regardless of what a caller declared, which is the
    off switch a bisection wants. Anything else, including unset, leaves
    the specialization available; it still does nothing until a caller
    declares an objective that guarantees it.

    The convention is `parallel.mojo`'s: an integer read through `_env_int`,
    a default that means "unchanged behavior", and zero meaning off. The
    explicit setters (`GpuActiveRows.set_constant_hessian`,
    `GpuHistogramBuilder.set_constant_hessian`, and the `const_hessian`
    argument the CPU builders take) exist alongside it for the reason
    `set_feature_group` gives: an A/B that reads its arm from the
    environment can run one arm under the other's label, which has happened
    in this repository once.
    """
    return _env_int("MOJOTREES_CONST_HESSIAN", 1) != 0


def const_hessian_verify() -> Bool:
    """Whether a declared constant hessian is checked against the data.

    `MOJOTREES_CONST_HESSIAN_VERIFY=1` makes every CPU builder that was
    told the hessians are constant walk the hessian array once and raise if
    any entry is not exactly `CONSTANT_HESSIAN`. Off by default because it
    is one extra pass over `n_rows` per build, which on a node with a
    handful of rows means a pass over the whole dataset.

    It is a diagnostic, not the safety mechanism. The safety mechanism is
    that the declaration comes from the objective. This exists so that a
    test, or somebody wiring a new trainer up, can find out immediately that
    a declaration is wrong instead of discovering it as a bit difference six
    rounds later.
    """
    return _env_int("MOJOTREES_CONST_HESSIAN_VERIFY", 0) != 0


def _check_constant_hessian(hess: List[Float64], n_rows: Int) raises:
    """Raise unless every one of the first `n_rows` hessians is exactly
    `CONSTANT_HESSIAN`. Only called under `const_hessian_verify`."""
    for r in range(n_rows):
        if hess[r] != CONSTANT_HESSIAN:
            raise Error(
                "a constant hessian was declared but row ",
                r,
                " has hessian ",
                hess[r],
                "; the declaration must come from an objective that"
                " guarantees it, and a weighted or GOSS round does not",
            )


def _resolve_const_hessian(declared: Bool) -> Bool:
    """The specialization's on/off decision for one build: the caller's
    declaration, ANDed with the environment's permission."""
    return declared and const_hessian_allowed()


@fieldwise_init
struct Histogram(Copyable, Movable):
    """Per-(feature, bin) statistics, flattened as `[f * n_bins + b]`."""

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]
    var n_features: Int
    var n_bins: Int

    @staticmethod
    def zeroed(n_features: Int, n_bins: Int) -> Histogram:
        """An all-zero histogram of the given shape, ready to accumulate
        into. Callers that build many histograms of one shape allocate once
        with this and recycle with `reset`."""
        var size = n_features * n_bins
        return Histogram(
            _zeroed_f64(size), _zeroed_f64(size), _zeroed_int(size),
            n_features, n_bins,
        )

    def reset(mut self):
        """Zero every bin in place, keeping the allocation. Cheaper than a
        fresh `zeroed` by exactly one malloc/free per buffer, which is what
        tree growth spends most of its allocator time on.

        Serial by construction, and the builders below no longer call it:
        they zero each feature's slice inside that feature's task instead. It
        stays because it is the right shape for a caller that needs a zeroed
        buffer without a build (`histogram_gpu`, `distributed`) and for any
        caller already running inside a parallel task, where a dispatching
        zero pass would nest one `sync_parallelize` inside another.
        """
        var size = self.n_features * self.n_bins
        var gp = self.grad.unsafe_ptr()
        var hp = self.hess.unsafe_ptr()
        var cp = self.count.unsafe_ptr()
        comptime W = SIMD_LANES
        var i = 0
        while i + W <= size:
            gp.unsafe_store(i, SIMD[DType.float64, W](0.0))
            hp.unsafe_store(i, SIMD[DType.float64, W](0.0))
            cp.unsafe_store(i, SIMD[DType.int, W](0))
            i += W
        while i < size:
            gp.unsafe_store(i, 0.0)
            hp.unsafe_store(i, 0.0)
            cp.unsafe_store(i, 0)
            i += 1

    def matches(self, n_features: Int, n_bins: Int) -> Bool:
        return self.n_features == n_features and self.n_bins == n_bins


@fieldwise_init
struct FeatureTotals(Copyable, Movable):
    """One feature's totals over its bins."""

    var grad: Float64
    var hess: Float64
    var count: Int


def _zeroed_f64(size: Int) -> List[Float64]:
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    return g^


def _zeroed_int(size: Int) -> List[Int]:
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    return c^


def _check_features(features: List[Int], n_features: Int) raises:
    """Feature ids must be in range; an empty list means every feature."""
    for i in range(len(features)):
        if features[i] < 0 or features[i] >= n_features:
            raise Error("feature index out of range")


def ensure_pair_capacity(mut pairs: List[Float64], n_rows: Int):
    """Size a gradient/hessian pair buffer for a node of `n_rows` rows.

    Grows only. A grower that keeps one buffer across a whole tree allocates
    it once, at the size of the largest node it meets (the root), and every
    later node reuses that allocation: the buffer is written before it is
    read, so stale contents beyond the current node are never observed.
    """
    var wanted = 2 * n_rows
    if len(pairs) < wanted:
        pairs.resize(wanted, 0.0)


def _zero_excluded(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    n_features: Int,
    n_bins: Int,
    features: List[Int],
    total_ops: Int,
) raises:
    """Zero the slices of every feature this build will not accumulate.

    Only reached under feature subsampling. The active features' slices are
    zeroed by the accumulation pass itself, on the task that is about to fill
    them, so this covers exactly the rest.
    """
    if len(features) == 0 or n_features <= 0 or n_bins <= 0:
        return

    # Mojo's scalar pointer API cannot load `Bool`. A byte preserves the
    # compact active mask and keeps the parallel zeroing pass unchanged.
    var active = List[UInt8](capacity=n_features)
    active.resize(n_features, UInt8(0))
    for i in range(len(features)):
        active[features[i]] = UInt8(1)

    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var active_p = active.unsafe_ptr()
    comptime W = SIMD_LANES

    def zero_range(f_start: Int, f_end: Int) {imm}:
        for f in range(f_start, f_end):
            if active_p.unsafe_load(f) == UInt8(0):
                var base = f * n_bins
                var b = 0
                while b + W <= n_bins:
                    gp.unsafe_store(
                        base + b, SIMD[DType.float64, W](0.0)
                    )
                    hp.unsafe_store(
                        base + b, SIMD[DType.float64, W](0.0)
                    )
                    cp.unsafe_store(base + b, SIMD[DType.int, W](0))
                    b += W
                while b < n_bins:
                    gp.unsafe_store(base + b, 0.0)
                    hp.unsafe_store(base + b, 0.0)
                    cp.unsafe_store(base + b, 0)
                    b += 1

    dispatch_feature_ranges(zero_range, n_features, total_ops)


def build_histogram(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
    const_hessian: Bool = False,
) raises -> Histogram:
    """Build a full-dataset histogram from per-row gradients and hessians.
    With a non-empty `features`, only those features are accumulated and the
    rest of the output stays zero.

    `const_hessian` declares that every entry of `hess` is exactly
    `CONSTANT_HESSIAN`, which is a property of the objective and never of
    the array; see `objective_has_constant_hessian`. The result is
    bit-identical either way.
    """
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_into(out, data, grad, hess, features, const_hessian)
    return out^


def build_histogram_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
    const_hessian: Bool = False,
) raises:
    """`build_histogram` into a caller-owned buffer, every cell of which is
    written: the accumulation pass zeroes each slice before filling it, so a
    buffer holding another node's histogram comes back holding exactly this
    one. Identical results to the allocating form, one fewer allocation."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    _check_features(features, data.n_features)
    var const_h = _resolve_const_hessian(const_hessian)
    if const_h and const_hessian_verify():
        _check_constant_hessian(hess, data.n_rows)

    # The three output buffers are passed as separate `mut` lists rather than
    # reached through `out`: a pointer taken from a struct field carries that
    # field's origin, which a worker closure cannot capture.
    _accumulate_full(
        out.grad, out.hess, out.count, data, grad, hess, features, const_h
    )


def _accumulate_full(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int],
    const_h: Bool,
) raises:
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = derive_accumulation_plan(
        cpu_profile(), n_features, n_active, n_bins, n_rows, False
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops,
    )

    # The ladder, dispatched from the host exactly as the GPU histogram family
    # is dispatched in `gpu_active_rows._enqueue_atomic_family`: one static
    # arm per instantiated width, `>=` rather than `==` so a width that is
    # somehow off the ladder still lands on a kernel that exists rather than
    # on none. The policy floors to a rung, so in practice each arm is hit
    # exactly.
    var group = plan.group_width
    if group >= 16:
        _accumulate_full_at[16](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h,
        )
    elif group >= 8:
        _accumulate_full_at[8](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h,
        )
    elif group >= 4:
        _accumulate_full_at[4](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h,
        )
    elif group >= 2:
        _accumulate_full_at[2](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h,
        )
    else:
        _accumulate_full_at[1](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h,
        )


def _accumulate_full_at[
    GROUP: Int
](
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int],
    n_active: Int,
    n_groups: Int,
    active_ops: Int,
    const_h: Bool,
) raises:
    """The full-dataset accumulation at one interleave width.

    `GROUP` features share one ascending walk of every row, so the gradient
    and the hessian are loaded once and spent `GROUP` times. The dispatch is
    over groups rather than over features, which is what guarantees a task
    holds whole groups and can therefore actually interleave what the policy
    asked for; splitting by feature let the task splitter hand a task a single
    feature and silently drop the interleave for it.

    A tail group owning fewer than `GROUP` features is not a second code path:
    `owned` bounds the unrolled body, exactly as the `owned` guard does in the
    device kernels in `gpu_active_rows.mojo`. The slot arrays are SIMD lanes
    rather than a stack array so that a comptime lane index stays in a
    register instead of becoming a load the compiler cannot prove does not
    alias the histogram it is storing into.

    Bit-identity across widths is argued in this module's docstring. Nothing
    in this function reads `GROUP` except to decide how many features share a
    walk; the per-feature summation order is the row order either way.

    **Why this builder does not gather (g, h) into an interleaved buffer, the
    way the subset builder does.** It was proposed on the grounds that the
    root reads two 8 MB Float64 streams per group rather than one 16 MB
    stream, and the arithmetic does not support it. Here `r` is the loop
    counter, so both reads are sequential: `grad` and `hess` each supply 8
    bytes per row, one 64-byte line per eight rows, 16 bytes of traffic per
    row in total. An interleaved buffer supplies the same 16 bytes per row as
    one line per four rows. Identical traffic, identical line count, and the
    interleaving would additionally cost a full 16 MB write and one more pass
    over the data. The gather earns its pass in `_accumulate_subset_at` for a
    different reason entirely, which does not apply here: there the rows
    arrive through a row-id list, so `grad[r]` and `hess[r]` are two separate
    lines *per row* rather than two streams, and gathering turns 2 * n_active
    indirect loads per row into two.

    That refutation is about the gradient INPUT and says nothing about the
    histogram OUTPUT. Interleaving the output cells, so that a bin's gradient
    and hessian are adjacent and one update touches one line instead of two,
    and narrowing `count` from a 64-bit `Int`, are separate proposals about
    `Histogram`'s layout. They are exact, they are not refuted here, and they
    are not this lane's to make: the layout is read by every consumer,
    including the GPU download path and the C ABI, so it is its own change
    behind its own profile.

    **The elided hessian plane.** With `const_h` the row loop performs two
    read-modify-writes per (row, feature) instead of three, the zeroing pass
    covers two planes instead of three, and the hessian slice is written once
    per bin at the end of the group's work as `Float64(count)`. The module
    docstring carries the exactness argument; the shape of it is that the
    three-plane path sums the Float64 literal 1.0 into the cell `count` times,
    every partial sum of which is an exactly representable integer, so the
    plane it produces *is* `Float64(count)` cell for cell. There is no
    multiply in the refill, so this introduces no contraction site of the kind
    `docs/NUMERICS.md` catalogues.

    The refill is written as a separate pass over the group's bins rather than
    folded into the flush of each cell, because a bin's count is not final
    until the whole row walk is done. It runs on the task that owns the slice,
    so it neither adds a dispatch nor crosses a task boundary.

    The trade it makes is honest and small. Per (row, feature) the row loop
    drops one read and one write of a Float64, which is 16 of the 48 bytes of
    scattered read-modify-write traffic a visit costs. Per bin the pass now
    writes two zeroed planes instead of three but then reads a count and
    writes a hessian, so bin-side traffic goes from 24 bytes to 32. The first
    term is multiplied by `n_rows` and the second by `n_bins`, which is why
    this is a saving and not a wash, and it is also why it is a saving that
    shrinks as a node gets smaller. `active_ops` is left alone: the estimate
    counts one op per zeroed cell and one per accumulated row, and three
    cell-touches per bin is what both arms do.

    The branch is taken once per group, outside the row walk, which is the
    same reason `_accumulate_subset_at` has two row loops for `compact`
    rather than one loop with a branch in it.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var use_all = len(features) == 0
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var bins_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    comptime W = SIMD_LANES

    def accumulate_groups(g_start: Int, g_end: Int) {imm}:
        for grp in range(g_start, g_end):
            var slot0 = grp * GROUP
            var owned = n_active - slot0
            if owned > GROUP:
                owned = GROUP
            var base = SIMD[DType.int, GROUP](0)
            var col = SIMD[DType.int, GROUP](0)
            comptime for k in range(GROUP):
                if k < owned:
                    var f = (
                        (slot0 + k) if use_all
                        else feat_p.unsafe_load(slot0 + k)
                    )
                    base[k] = f * n_bins
                    col[k] = f * n_rows

            # Zeroing stays fused into the pass that fills the slice, as the
            # module docstring explains, and now covers the whole group before
            # the shared row walk begins. The hessian plane is skipped under
            # `const_h`: it is overwritten wholesale by the refill below, so
            # zeroing it would be a write nothing reads.
            comptime for k in range(GROUP):
                if k < owned:
                    var z0 = Int(base[k])
                    var zb = 0
                    while zb + W <= n_bins:
                        gp.unsafe_store(z0 + zb, SIMD[DType.float64, W](0.0))
                        if not const_h:
                            hp.unsafe_store(
                                z0 + zb, SIMD[DType.float64, W](0.0)
                            )
                        cp.unsafe_store(z0 + zb, SIMD[DType.int, W](0))
                        zb += W
                    while zb < n_bins:
                        gp.unsafe_store(z0 + zb, 0.0)
                        if not const_h:
                            hp.unsafe_store(z0 + zb, 0.0)
                        cp.unsafe_store(z0 + zb, 0)
                        zb += 1

            if const_h:
                # Two planes per (row, feature), and the hessian is not read
                # at all: it is the same value on every row by the objective's
                # construction, so the accumulation carries no information the
                # count does not.
                for r in range(n_rows):
                    var g = grad_p.unsafe_load(r)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_p.unsafe_load(Int(col[k]) + r)
                            )
                            gp.unsafe_store(b, gp.unsafe_load(b) + g)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                for r in range(n_rows):
                    # Contiguous: the whole dataset is one ascending walk, so
                    # the gradients, the hessians, and all `GROUP` binned
                    # columns are sequential streams.
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_p.unsafe_load(Int(col[k]) + r)
                            )
                            gp.unsafe_store(b, gp.unsafe_load(b) + g)
                            hp.unsafe_store(b, hp.unsafe_load(b) + h)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

            # The boundary at which the elided plane is refilled, and the only
            # place this loop touches the hessian plane at all under
            # `const_h`. It runs on the task that owns these slices, after
            # that task's row walk has finalized the counts, so it neither
            # adds a dispatch nor crosses a task boundary. Exact by the
            # argument in this function's docstring: a cell that received `n`
            # rows holds `Float64(n)` on the three-plane path, and
            # `Float64(n)` is what is written here.
            if const_h:
                comptime for k in range(GROUP):
                    if k < owned:
                        var f0 = Int(base[k])
                        var fb = 0
                        while fb + W <= n_bins:
                            hp.unsafe_store(
                                f0 + fb,
                                cp.unsafe_load[width=W](f0 + fb).cast[
                                    DType.float64
                                ](),
                            )
                            fb += W
                        while fb < n_bins:
                            hp.unsafe_store(
                                f0 + fb, Float64(cp.unsafe_load(f0 + fb))
                            )
                            fb += 1

    dispatch_feature_ranges(accumulate_groups, n_groups, active_ops)


def build_histogram_subset(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
    const_hessian: Bool = False,
) raises -> Histogram:
    """Build a histogram over a subset of rows (one tree node's rows). With a
    non-empty `features`, only those features are accumulated and the rest of
    the output stays zero.

    `const_hessian` declares the objective's guarantee that every hessian is
    exactly `CONSTANT_HESSIAN`; the result is bit-identical either way.
    """
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into(
        out, data, grad, hess, rows, 0, len(rows), features, const_hessian
    )
    return out^


def build_histogram_subset_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int] = [],
    const_hessian: Bool = False,
) raises:
    """`build_histogram_subset` over the window `rows[row_start :
    row_start + row_count]`, into a caller-owned buffer every cell of which
    is written.

    The window lets tree growth keep every node's row ids in one shared arena
    instead of allocating a fresh `List[Int]` per node; passing
    `(0, len(rows))` is exactly the whole-list behaviour.

    This form owns the gradient/hessian pair buffer for the duration of the
    call, which means allocating and freeing it per node. A grower visiting
    hundreds of nodes should hold one buffer and call
    `build_histogram_subset_into_scratch` instead; the two produce identical
    results.
    """
    var pairs = List[Float64]()
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian,
    )


def build_histogram_subset_into_scratch(
    mut out: Histogram,
    mut pairs: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int] = [],
    const_hessian: Bool = False,
) raises:
    """`build_histogram_subset_into` with a caller-owned scratch buffer.

    `pairs` is the node's gradients and hessians, interleaved, and is grown
    (never shrunk) to fit. Passing the same list across every node of a tree
    turns a per-node allocation of `16 * row_count` bytes into one allocation
    per tree; passing a fresh empty list is exactly
    `build_histogram_subset_into`. Its contents on entry are irrelevant and
    its contents on exit are unspecified, so it carries no state between
    calls beyond its capacity.

    Whether the buffer is filled at all is a policy decision
    (`apple_cpu_policy.derive_accumulation_plan`): a node with one active
    feature, or too few rows for the gather to pay for itself, reads the
    gradients through the row ids as before. Both paths accumulate the same
    values in the same order.

    Under `const_hessian` the gather still writes the pair buffer as it
    always did. Halving it to a gradient-only buffer is a change to the
    scratch contract this lane deliberately does not make: the buffer's
    stride is `2 * row_count` in this function's documented interface, a
    grower holds one across a whole tree, and a stride that depends on the
    objective is a worse thing to own than one wasted store per row. The
    saving that matters is in the feature loop, which visits the buffer
    `n_active` times.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    _check_features(features, data.n_features)
    var const_h = _resolve_const_hessian(const_hessian)
    if const_h and const_hessian_verify():
        _check_constant_hessian(hess, data.n_rows)

    _accumulate_subset(
        out.grad, out.hess, out.count, pairs,
        data, grad, hess, rows, row_start, row_count, features, const_h,
    )


def _accumulate_subset(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    mut pairs: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int],
    const_h: Bool,
) raises:
    var n_bins = data.n_bins
    var n_features = data.n_features
    var n_sub = row_count
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = derive_accumulation_plan(
        cpu_profile(), n_features, n_active, n_bins, n_sub, True
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops,
    )
    if plan.compact_rows:
        ensure_pair_capacity(pairs, n_sub)

    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var pairs_p = pairs.unsafe_ptr()

    # One gather for the whole node instead of one per feature. Disjoint
    # ascending blocks, elementwise, so the buffer comes out the same
    # whatever the block count.
    def fill_pairs(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = rows_p.unsafe_load(i)
            pairs_p.unsafe_store(2 * i, grad_p.unsafe_load(r))
            pairs_p.unsafe_store(2 * i + 1, hess_p.unsafe_load(r))

    if plan.compact_rows:
        dispatch_rows(fill_pairs, n_sub, plan.gather_ops)

    # The ladder, dispatched as in `_accumulate_full` above.
    var group = plan.group_width
    if group >= 16:
        _accumulate_subset_at[16](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h,
        )
    elif group >= 8:
        _accumulate_subset_at[8](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h,
        )
    elif group >= 4:
        _accumulate_subset_at[4](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h,
        )
    elif group >= 2:
        _accumulate_subset_at[2](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h,
        )
    else:
        _accumulate_subset_at[1](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h,
        )


def _accumulate_subset_at[
    GROUP: Int
](
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    pairs: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int],
    compact: Bool,
    n_active: Int,
    n_groups: Int,
    active_ops: Int,
    const_h: Bool,
) raises:
    """The subset accumulation at one interleave width.

    The twin of `_accumulate_full_at`, over `rows[row_start : row_start +
    row_count]` instead of over every row, and with two row loops instead of
    one. `compact` selects between the gathered `(g, h)` pair buffer and the
    two indirect loads through the row id; they read the same Float64 values
    in the same order, so which one runs cannot change a bin. They are two
    loops rather than one loop with a branch so the branch is not re-evaluated
    a million times, which is the shape `gpu_active_rows`'s `use_quant` pair
    uses for the same reason.

    The gather decision and the interleave width are now independent, where
    before the pairing branch also required `compact` and so a small node ran
    one feature at a time whatever the policy said. Widening the non-gathered
    path is strictly less work: the two indirect loads per row are spent
    `GROUP` times instead of once.

    Tail groups, the SIMD-lane slot arrays, and the bit-identity argument are
    as in `_accumulate_full_at`, and so is the elided hessian plane: with
    `const_h` the row loops drop the hessian read-modify-write, the zeroing
    pass drops that plane, and the slice is refilled from the count once per
    bin after the walk. That makes four row loops rather than two, which is
    the same trade the `compact` pair already took: a branch evaluated once
    per group instead of once per (row, feature).
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_sub = row_count
    var use_all = len(features) == 0
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var pairs_p = pairs.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    comptime W = SIMD_LANES

    def accumulate_groups(g_start: Int, g_end: Int) {imm}:
        for grp in range(g_start, g_end):
            var slot0 = grp * GROUP
            var owned = n_active - slot0
            if owned > GROUP:
                owned = GROUP
            var base = SIMD[DType.int, GROUP](0)
            var col = SIMD[DType.int, GROUP](0)
            comptime for k in range(GROUP):
                if k < owned:
                    var f = (
                        (slot0 + k) if use_all
                        else feat_p.unsafe_load(slot0 + k)
                    )
                    base[k] = f * n_bins
                    col[k] = f * n_rows

            comptime for k in range(GROUP):
                if k < owned:
                    var z0 = Int(base[k])
                    var zb = 0
                    while zb + W <= n_bins:
                        gp.unsafe_store(z0 + zb, SIMD[DType.float64, W](0.0))
                        if not const_h:
                            hp.unsafe_store(
                                z0 + zb, SIMD[DType.float64, W](0.0)
                            )
                        cp.unsafe_store(z0 + zb, SIMD[DType.int, W](0))
                        zb += W
                    while zb < n_bins:
                        gp.unsafe_store(z0 + zb, 0.0)
                        if not const_h:
                            hp.unsafe_store(z0 + zb, 0.0)
                        cp.unsafe_store(z0 + zb, 0)
                        zb += 1

            if const_h:
                if compact:
                    for i_row in range(n_sub):
                        var r = rows_p.unsafe_load(i_row)
                        # The gather still interleaves (g, h); only the
                        # gradient half is read here.
                        var g = pairs_p.unsafe_load(2 * i_row)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var b = Int(base[k]) + Int(
                                    bins_all_p.unsafe_load(Int(col[k]) + r)
                                )
                                gp.unsafe_store(b, gp.unsafe_load(b) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
                else:
                    for i_row in range(n_sub):
                        var r = rows_p.unsafe_load(i_row)
                        var g = grad_p.unsafe_load(r)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var b = Int(base[k]) + Int(
                                    bins_all_p.unsafe_load(Int(col[k]) + r)
                                )
                                gp.unsafe_store(b, gp.unsafe_load(b) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            elif compact:
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    # Adjacent, so one cache line carries both.
                    var g = pairs_p.unsafe_load(2 * i_row)
                    var h = pairs_p.unsafe_load(2 * i_row + 1)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            gp.unsafe_store(b, gp.unsafe_load(b) + g)
                            hp.unsafe_store(b, hp.unsafe_load(b) + h)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                # Small node, or one active feature: the gather would cost a
                # pass and save nothing, so the gradients are read through the
                # row ids as they always were.
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            gp.unsafe_store(b, gp.unsafe_load(b) + g)
                            hp.unsafe_store(b, hp.unsafe_load(b) + h)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

            # The refill, exactly as in `_accumulate_full_at`: the elided
            # plane is written once per bin as `Float64(count)`, which is bit
            # for bit what the three-plane path accumulated.
            if const_h:
                comptime for k in range(GROUP):
                    if k < owned:
                        var f0 = Int(base[k])
                        var fb = 0
                        while fb + W <= n_bins:
                            hp.unsafe_store(
                                f0 + fb,
                                cp.unsafe_load[width=W](f0 + fb).cast[
                                    DType.float64
                                ](),
                            )
                            fb += W
                        while fb < n_bins:
                            hp.unsafe_store(
                                f0 + fb, Float64(cp.unsafe_load(f0 + fb))
                            )
                            fb += 1

    dispatch_feature_ranges(accumulate_groups, n_groups, active_ops)


def build_histogram_subset_replica_into[
    grad_origin: ImmOrigin, hess_origin: ImmOrigin, //
](
    mut out: Histogram,
    mut fixed: List[Int32],
    data: BinnedMatrix,
    grad_f32: Span[Float32, grad_origin],
    hess_f32: Span[Float32, hess_origin],
    rows: List[Int32],
    row_start: Int,
    row_count: Int,
    g_scale: Float64,
    h_scale: Float64,
    features: List[Int] = [],
    const_hessian: Bool = False,
) raises:
    """The host replica of the device's fixed-point histogram build, over
    the window `rows[row_start : row_start + row_count]`.

    This is the builder `docs/design/HYBRID_TRAINING.md` §8.2 specifies: it
    must reproduce the device pipeline number for number, so a histogram it
    produces can substitute for a downloaded one without changing a split.
    Per (row, feature) it computes

        gq = Int32(round(grad_f32[r] * Float32(g_scale)))

    exactly as `_range_hist_atomic_kernel` does — the *same* Float32 inputs
    the kernel reads (the staged conversions, not the Float64 originals) and
    the same Float32 scale — accumulates in Int32, and dequantizes with the
    same `Float64(sum) * (1.0 / scale)` as `histogram_from_host`. Integer
    addition is associative and commutative, so the per-feature tasks and
    their order cannot change a sum; whether the host's multiply-and-round
    matches the device's bit for bit is a hardware claim this function
    cannot make, and `tests/test_host_replica.mojo` is where it is
    established on the target hardware. (`MODE_MIRROR` in
    hybrid_leaf_scheduler.mojo established it in-run until that module was
    deleted on 2026-08-16; this function outlives it, because the replica is
    the CPU/GPU oracle and was never the scheduler's.)

    `fixed` is the caller-owned Int32 scratch holding the three planes in
    the device's `[grad | hess | count]` layout, followed by the node's
    quantized (g, h) pairs; it is grown to
    `3 * n_features * n_bins + 2 * row_count` as needed and its contents on
    entry are irrelevant. Excluded features' output slices are zeroed, as every
    builder here guarantees, so the result has the full dataset shape.

    `const_hessian` elides the fixed-point hessian plane exactly as the
    device kernels do, and must be set to whatever the device build it is
    replicating was set to -- although, and this is the point, the two agree
    even when it is not. On the elided path this function quantizes the
    constant once as `Int32(round(Float32(CONSTANT_HESSIAN) * h_scale))` and
    multiplies by the count; on the three-plane path it adds that same Int32
    into the bin once per row. The two are equal as integers, and equal even
    under Int32 wraparound, since repeated addition and multiplication agree
    modulo 2^32. So the elision cannot move this replica relative to the
    device, whichever arm each of them is on. What it does do is shrink the
    claim that comparison has to establish: instead of the host and the device
    agreeing on `round` at every one of `n_rows` products, they need only
    agree on it at one, because `Float32(1.0) * h_scale` is exactly
    `h_scale` on any IEEE-754 unit and both sides therefore round the same
    number.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    if len(grad_f32) != n_rows or len(hess_f32) != n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(n_features, n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    if g_scale <= 0.0 or h_scale <= 0.0:
        raise Error("fixed-point scales must be positive")
    _check_features(features, n_features)
    for j in range(row_start, row_start + row_count):
        var r = Int(rows[j])
        if r < 0 or r >= n_rows:
            raise Error("row id out of range")

    var hist_size = n_features * n_bins
    if len(fixed) < 3 * hist_size + 2 * row_count:
        fixed.resize(3 * hist_size + 2 * row_count, Int32(0))

    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    _zero_excluded(
        out.grad, out.hess, out.count,
        n_features, n_bins, features,
        (n_features - n_active) * n_bins,
    )

    # The three output buffers are passed as separate `mut` lists rather than
    # reached through `out`, exactly as `build_histogram_into` explains: a
    # pointer taken from a struct field carries that field's origin, which a
    # worker closure cannot capture.
    _accumulate_replica(
        out.grad, out.hess, out.count, fixed,
        data, grad_f32, hess_f32, rows, row_start, row_count,
        g_scale, h_scale, features,
        _resolve_const_hessian(const_hessian),
    )


def _accumulate_replica[
    grad_origin: ImmOrigin, hess_origin: ImmOrigin, //
](
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    mut fixed: List[Int32],
    data: BinnedMatrix,
    grad_f32: Span[Float32, grad_origin],
    hess_f32: Span[Float32, hess_origin],
    rows: List[Int32],
    row_start: Int,
    row_count: Int,
    g_scale: Float64,
    h_scale: Float64,
    features: List[Int],
    const_h: Bool,
) raises:
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    var hist_size = n_features * n_bins
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)

    var gsf = Float32(g_scale)
    var hsf = Float32(h_scale)
    var g_inv = 1.0 / g_scale
    var h_inv = 1.0 / h_scale
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var fixed_p = fixed.unsafe_ptr()
    var grad_p = grad_f32.unsafe_ptr()
    var hess_p = hess_f32.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()

    # Quantize once per row, not once per (row, feature). The kernel rounds
    # inside its accumulation loop, but the rounded value is a pure function
    # of the row, so hoisting it changes no Int32 the sums are made of --
    # while removing 2 * (n_active - 1) multiply-and-rounds per row, which
    # measured as the difference between this builder running at the Float64
    # builder's rate and running twenty times slower. The pairs land in the
    # tail of the caller's scratch, interleaved like the gather in
    # `_accumulate_subset`.
    var pairs_off = 3 * hist_size

    # The elided plane's one quantized value, computed once for the whole
    # build. `Float32(CONSTANT_HESSIAN) * hsf` is exactly `hsf` on any
    # IEEE-754 unit, since multiplying by one is the identity and cannot
    # round, so this is `Int32(round(hsf))` and it is the same Int32 the
    # three-plane path below would have produced for every row.
    var hq_const = Int32(round(Float32(CONSTANT_HESSIAN) * hsf))

    def fill_quantized(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = Int(rows_p.unsafe_load(i))
            fixed_p.unsafe_store(
                pairs_off + 2 * i,
                Int32(round(grad_p.unsafe_load(r) * gsf)),
            )
            if not const_h:
                fixed_p.unsafe_store(
                    pairs_off + 2 * i + 1,
                    Int32(round(hess_p.unsafe_load(r) * hsf)),
                )

    dispatch_rows(fill_quantized, row_count, row_count)

    def accumulate_feature(i: Int) {imm}:
        var f = i if use_all else feat_p.unsafe_load(i)
        var base = f * n_bins
        for b in range(n_bins):
            fixed_p.unsafe_store(base + b, Int32(0))
            if not const_h:
                fixed_p.unsafe_store(hist_size + base + b, Int32(0))
            fixed_p.unsafe_store(2 * hist_size + base + b, Int32(0))
        var col = bins_all_p.unsafe_offset(f * n_rows)
        if const_h:
            for j in range(row_count):
                var r = Int(rows_p.unsafe_load(j))
                var slot = base + Int(col.unsafe_load(r))
                var gq = fixed_p.unsafe_load(pairs_off + 2 * j)
                fixed_p.unsafe_store(slot, fixed_p.unsafe_load(slot) + gq)
                fixed_p.unsafe_store(
                    2 * hist_size + slot,
                    fixed_p.unsafe_load(2 * hist_size + slot) + Int32(1),
                )
        else:
            for j in range(row_count):
                var r = Int(rows_p.unsafe_load(j))
                var slot = base + Int(col.unsafe_load(r))
                var gq = fixed_p.unsafe_load(pairs_off + 2 * j)
                var hq = fixed_p.unsafe_load(pairs_off + 2 * j + 1)
                fixed_p.unsafe_store(slot, fixed_p.unsafe_load(slot) + gq)
                fixed_p.unsafe_store(
                    hist_size + slot,
                    fixed_p.unsafe_load(hist_size + slot) + hq,
                )
                fixed_p.unsafe_store(
                    2 * hist_size + slot,
                    fixed_p.unsafe_load(2 * hist_size + slot) + Int32(1),
                )
        for b in range(n_bins):
            var cq = fixed_p.unsafe_load(2 * hist_size + base + b)
            gp.unsafe_store(
                base + b, Float64(fixed_p.unsafe_load(base + b)) * g_inv
            )
            # `hq_const * cq` is the exact Int32 the loop above would have
            # accumulated: `cq` copies of `hq_const` added together. The
            # dequantization is then the same `Float64(q) * (1 / scale)` the
            # three-plane path applies, so the Float64 that lands is the same
            # Float64.
            var hq_sum = (
                hq_const * cq
                if const_h
                else fixed_p.unsafe_load(hist_size + base + b)
            )
            hp.unsafe_store(base + b, Float64(hq_sum) * h_inv)
            cp.unsafe_store(base + b, Int(cq))

    dispatch_features(
        accumulate_feature,
        n_active,
        row_count * n_active + 2 * n_active * n_bins,
    )


def feature_totals(hist: Histogram, feature: Int) raises -> FeatureTotals:
    """Totals over one feature's bins.

    Split finding computes exactly this inline before it scans a feature, and
    tree growth computes it again for the leaf value. The order here is the
    order `split.mojo` uses today and must stay that way: `SIMD_LANES` lanes
    accumulated in parallel over the whole vector blocks, reduced, then the
    scalar tail added in ascending bin order. Anything else would move a
    result, not just its cost.

    Nothing calls this yet. It exists so that a split scan parallelized
    across features has a totals pass it can hoist without changing a single
    gain (see the handoff for the call sites).
    """
    if feature < 0 or feature >= hist.n_features:
        raise Error("feature index out of range")
    comptime W = SIMD_LANES
    var grad_p = hist.grad.unsafe_ptr()
    var hess_p = hist.hess.unsafe_ptr()
    var count_p = hist.count.unsafe_ptr()
    var base = feature * hist.n_bins
    var vg = SIMD[DType.float64, W](0.0)
    var vh = SIMD[DType.float64, W](0.0)
    var vc = SIMD[DType.int, W](0)
    var b = 0
    while b + W <= hist.n_bins:
        vg += grad_p.unsafe_load[width=W](base + b)
        vh += hess_p.unsafe_load[width=W](base + b)
        vc += count_p.unsafe_load[width=W](base + b)
        b += W
    var total_g = vg.reduce_add()
    var total_h = vh.reduce_add()
    var total_c = Int(vc.reduce_add())
    while b < hist.n_bins:
        total_g += grad_p.unsafe_load(base + b)
        total_h += hess_p.unsafe_load(base + b)
        total_c += count_p.unsafe_load(base + b)
        b += 1
    return FeatureTotals(total_g, total_h, total_c)


def subtract_histogram(
    parent: Histogram, child: Histogram, const_hessian: Bool = False
) raises -> Histogram:
    """Sibling histogram via the subtraction trick: build the smaller child
    directly, get the larger one as parent - child for free."""
    var out = Histogram.zeroed(parent.n_features, parent.n_bins)
    subtract_histogram_into(out, parent, child, const_hessian)
    return out^


def _subtract_histogram_arrays(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    parent_grad: List[Float64],
    parent_hess: List[Float64],
    parent_count: List[Int],
    child_grad: List[Float64],
    child_hess: List[Float64],
    child_count: List[Int],
    size: Int,
    const_h: Bool,
) raises:
    """Parallel SIMD sibling subtraction over independent array borrows.

    Keeping this worker outside `Histogram` field access gives each buffer a
    stable origin that Mojo's ownership checker can carry into the parallel
    closure. The dispatch, range partitioning, SIMD width, and arithmetic are
    identical to the original optimized implementation.

    With `const_h` the hessian plane is not subtracted at all. Both operands
    hold `Float64(count)` cell for cell, so the difference the general path
    would compute is `Float64(parent_count) - Float64(child_count)`; both
    operands of that subtract are exactly representable integers below 2^53
    and so is their difference, so it is exactly `Float64(parent_count -
    child_count)`, which is `Float64` of the count this same loop is already
    computing. Writing it from the integer difference is therefore the same
    Float64.

    Two of the pass's nine memory streams go, and it is worth being exact
    about which two rather than calling it a third. The general pass runs a
    load, a load and a store on each of three planes. The elided one still
    stores the hessian plane, because the output histogram has three planes
    like every other; what it stops doing is *reading* the two operands'
    hessian planes, since the value it needs is the integer difference it
    already has in a register. Nine streams become seven, and 72 bytes per
    cell become 56. `subtract_ops_for_planes` is told two planes rather than
    three so the parallel crossover is priced on traffic that resembles what
    happens.
    """
    var pg = parent_grad.unsafe_ptr()
    var ph = parent_hess.unsafe_ptr()
    var pc = parent_count.unsafe_ptr()
    var cg = child_grad.unsafe_ptr()
    var ch = child_hess.unsafe_ptr()
    var cc = child_count.unsafe_ptr()
    var og = out_grad.unsafe_ptr()
    var oh = out_hess.unsafe_ptr()
    var oc = out_count.unsafe_ptr()

    def subtract_block(start: Int, end: Int) {imm}:
        comptime W = SIMD_LANES
        var i = start
        if const_h:
            while i + W <= end:
                og.unsafe_store(
                    i, pg.unsafe_load[width=W](i) - cg.unsafe_load[width=W](i)
                )
                var dc = pc.unsafe_load[width=W](i) - cc.unsafe_load[width=W](
                    i
                )
                oc.unsafe_store(i, dc)
                oh.unsafe_store(i, dc.cast[DType.float64]())
                i += W
            while i < end:
                og.unsafe_store(i, pg.unsafe_load(i) - cg.unsafe_load(i))
                var dc = pc.unsafe_load(i) - cc.unsafe_load(i)
                oc.unsafe_store(i, dc)
                oh.unsafe_store(i, Float64(dc))
                i += 1
            return
        while i + W <= end:
            og.unsafe_store(
                i, pg.unsafe_load[width=W](i) - cg.unsafe_load[width=W](i)
            )
            oh.unsafe_store(
                i, ph.unsafe_load[width=W](i) - ch.unsafe_load[width=W](i)
            )
            oc.unsafe_store(
                i, pc.unsafe_load[width=W](i) - cc.unsafe_load[width=W](i)
            )
            i += W
        while i < end:
            og.unsafe_store(i, pg.unsafe_load(i) - cg.unsafe_load(i))
            oh.unsafe_store(i, ph.unsafe_load(i) - ch.unsafe_load(i))
            oc.unsafe_store(i, pc.unsafe_load(i) - cc.unsafe_load(i))
            i += 1

    var ops = subtract_ops_for_planes(size, 2) if const_h else subtract_ops(
        size
    )
    dispatch_rows(subtract_block, size, ops)


def subtract_histogram_into(
    mut out: Histogram,
    parent: Histogram,
    child: Histogram,
    const_hessian: Bool = False,
) raises:
    """`subtract_histogram` into a caller-owned buffer. Every element is
    written, so unlike the accumulating builders this one needs no zeroing
    pass at all.

    `const_hessian` declares that **both operands** were built under the
    constant-hessian specialization, so each holds `Float64(count)` in its
    hessian plane. It is the caller's declaration and not something this
    function can see: a histogram carries no record of how it was built, on
    purpose, since the layout is read by the C ABI and by the GPU download
    path and is not this lane's to widen. Declaring it for a pair of
    histograms that were not built that way produces a wrong hessian plane,
    which is why the declaration belongs at the trainer, next to the
    objective, and travels down rather than being guessed here.

    "For free" was always an overstatement: the pass reads two histograms and
    writes a third, six streams over `n_features * n_bins` cells each, and at
    the default shape that is megabytes per node against a node whose
    accumulation the trick just avoided. It is elementwise, so it splits into
    contiguous blocks that reproduce the serial result exactly, and the work
    estimate counts one op per cell per stream rather than per cell (see
    `apple_cpu_policy.subtract_ops`), which is what lets a shape this
    memory-bound reach the parallel path at all.
    """
    if (
        parent.n_features != child.n_features
        or parent.n_bins != child.n_bins
    ):
        raise Error("histogram shapes must match")
    if not out.matches(parent.n_features, parent.n_bins):
        raise Error("output histogram shape must match the operands")

    _subtract_histogram_arrays(
        out.grad,
        out.hess,
        out.count,
        parent.grad,
        parent.hess,
        parent.count,
        child.grad,
        child.hess,
        child.count,
        parent.n_features * parent.n_bins,
        _resolve_const_hessian(const_hessian),
    )
