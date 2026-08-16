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
task-scheduling overhead take the serial path. A node large enough to amortize
a private histogram per row block also parallelizes across *rows*, on a second
axis that multiplies the first; see "The group width, and row blocks
alongside it" below, and read it before assuming any statement about
summation order in this file still holds unqualified.

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

None of those three changes a value or an order: each feature still sums its
bins over rows in ascending order inside a single task. The argument is worth
stating precisely, because widening an interleave is exactly the kind of
change that quietly reassociates a Float64 sum, and this one does not. It is
also the argument the row-blocked kernel deliberately breaks, which is why it
is stated here in full and then answered rather than quietly amended: what
follows is true of the unblocked accumulation, and the row-block section below
says exactly which sentence of it stops being true and on what authority.

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

**The group width, and row blocks alongside it.** Widening the group divides
both the gradient traffic and the task count by the same factor, so on its own
it trades memory traffic against parallel slack and neither term can be made
large without shrinking the other. Worse, the task count it leaves has a hard
ceiling: `ceil(n_active / width)` groups, which at 50 features and width 2 is
25 units whatever the node's row count is, and which does not fall as nodes get
smaller, so a node with a thousandth of the rows pays the same 25 scheduling
events for a thousandth of the work.

The decomposition that removes both is the obvious one, and it is now
implemented in `_accumulate_subset_blocked_at`: split a node's rows into
contiguous ascending blocks, give each block a private histogram over the
active features, and fold the partials. The dispatch unit becomes a
`(block, group)` pair, so the two axes multiply instead of trading, and the
block count grows with the row count instead of being capped by the feature
count. It is what the GPU tiled kernel already does.

**It moves bits, and this is the argument that it is allowed to.** The
reasoning that used to rule it out here was correct about the mechanism and
wrong about the conclusion, so the mechanism is kept. A fold over contiguous
ascending row blocks gives cell `(f, b)` the value

    (sum of block 0's rows in b) + (sum of block 1's rows in b) + ...

and Float64 addition is not associative, so that is a *different value* from
adding the same rows one at a time in ascending order. Every word of that
stands. What has changed is the standard it is measured against. This project
used to promise bit-identity with its own past output; it now promises
something narrower and more useful:

  **Deterministic on a given toolchain**, meaning identical at every
  `MOJOTREES_NUM_WORKERS`, at every task count, at every dispatch grain, and
  on every machine this toolchain targets. **Identity with past output is not
  a promise.**

The blocked kernel meets that standard by construction, and the construction
is worth stating precisely because it is the only thing keeping it true:

  The block count, the block boundaries, and the fold order are *values*, not
  schedule. `apple_cpu_policy.plan_row_block_count` derives the block count
  from the node's row count, the bin count, the active feature count and an
  explicit request, and from nothing else. No core count, no core pool, no
  worker count, no task count, no machine fact reaches it. Blocks are
  contiguous and ascending, a block's rows are walked in ascending order, a
  unit is never split across tasks, and the fold sums block 0, then block 1,
  and so on regardless of how many tasks the fold is cut into. So two runs
  that differ only in how many cores were woken perform the identical sequence
  of Float64 additions.

And the direction of the movement is not neutral. A fold of `B` partial sums
is a two-level summation tree, so the worst-case rounding error of a cell that
received `n` rows goes from `O(n * eps)` for the flat ascending sum to
`O((n / B + B) * eps)` for the blocked one. Blocking cannot make the error
bound worse for `B` in `[1, n]` and for `B` near `sqrt(n)` it is
asymptotically better. Bit movement here is movement *toward* the exact sum
in bound, not away from it. It is not a proof about any particular cell --
non-associativity means an individual value can move either way -- and the
commit that landed this states the largest movement actually observed across
the golden fixtures rather than resting on the bound.

**What the count plane and sibling subtraction do, which is not affected.**
The count plane is integer addition, which *is* associative, so a blocked
count is the same integer at any block count. Under the constant-hessian
specialization the hessian plane is `Float64(count)` on both operands of a
sibling subtraction, and both are exactly representable integers below 2^53,
so that subtraction is exact under blocking exactly as it was without it. The
gradient plane's subtraction was never bit-equal to a direct build of the
sibling and is not claimed to be; blocking does not change that either way.
See `_subtract_histogram_arrays`.

The GPU path escapes the reassociation argument entirely, rather than
accepting it, because it accumulates in fixed-point Int32 where addition *is*
associative; see `_range_hist_partial_kernel` in `gpu_active_rows.mojo`.

**Where blocking does not run.** The full-dataset builder never blocks: it
has no caller-owned scratch to keep the private histograms in, and allocating
them per call is the cost this decomposition exists to avoid. A node too small
for a block to amortize its own private histogram never blocks either --
`apple_cpu_policy.row_block_min_rows` is the floor and its derivation is
there -- so small and tiny nodes keep the accumulation, and the bytes, they
already had.

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
of the pass, so the assembled `Histogram` still presents a gradient, a hessian
and a count for every cell and no consumer changes.

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
from std.time import perf_counter_ns

from .apple_cpu_policy import (
    ASSUMED_CACHE_LINE_BYTES,
    BIN_LAYOUT_AUTO,
    BIN_LAYOUT_FEATURE_MAJOR,
    BIN_LAYOUT_ROW_MAJOR,
    ROW_BLOCK_PLANES,
    AccumulationPlan,
    align_cells_up,
    derive_accumulation_plan_with,
    env_bin_layout,
    histogram_line_floats,
    resolve_bin_layout,
    subtract_ops,
    subtract_ops_for_planes,
)
from .binning import BinnedMatrix
from .objective_registry import HUBER, L1, QUANTILE, SQUARED_ERROR
from .parallel import (
    DispatchSettings,
    _env_int,
    dispatch_feature_ranges_with,
    dispatch_features,
    dispatch_rows,
    dispatch_rows_with,
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
    """Per-(feature, bin) statistics, flattened as `[f * n_bins + b]`.

    The three planes are named with a leading underscore **on purpose**. They
    were `grad`, `hess` and `count`, and call sites across the package, the
    tests and the benches read them directly, which pins the storage: three
    separate `List`s, an `Int` count, three streams touched per cell.
    Changing that layout (an interleaved array of structs, an `Int32` count)
    is a later lane, and what stood in its way was not the arithmetic but the
    impossibility of *knowing* every reader had been found. Renaming the
    fields turns that from a belief into a build error list: the compiler
    enumerates the couplings, exhaustively and mechanically, and a new direct
    reader cannot appear without someone typing the underscore on purpose.

    Read a cell through `grad_at` / `hess_at` / `count_at` and write one
    through `set_grad_at` / `set_hess_at` / `set_count_at`; `n_cells` is the
    old `len(hist.grad)`. There is deliberately no accessor that hands out a
    whole plane, so `grep -rn '\\._grad\\b'` is the exact and complete list of
    what a storage change must rewrite by hand.

    Construction goes through `from_planes`. The fieldwise constructor still
    exists and Mojo has no way to hide it, so `Histogram(g, h, c, nf, nb)`
    would still compile; every construction site in the tree goes through
    `from_planes` instead, which is what makes it the one interception point.

    Storage here is unchanged from before the rename: three planes, `Float64`
    gradients and hessians, `Int` counts, same flattening, same arithmetic.
    """

    var _grad: List[Float64]
    var _hess: List[Float64]
    var _count: List[Int]
    var n_features: Int
    var n_bins: Int

    @staticmethod
    def from_planes(
        var grad: List[Float64],
        var hess: List[Float64],
        var count: List[Int],
        n_features: Int,
        n_bins: Int,
    ) -> Histogram:
        """Take ownership of three already-built planes, in the order the
        fieldwise constructor took them. The named entry point for
        construction, so that a storage change has one place to intercept."""
        return Histogram(grad^, hess^, count^, n_features, n_bins)

    @staticmethod
    def zeroed(n_features: Int, n_bins: Int) -> Histogram:
        """An all-zero histogram of the given shape, ready to accumulate
        into. Callers that build many histograms of one shape allocate once
        with this and recycle with `reset`."""
        var size = n_features * n_bins
        return Histogram.from_planes(
            _zeroed_f64(size), _zeroed_f64(size), _zeroed_int(size),
            n_features, n_bins,
        )

    # --- cell accessors ---------------------------------------------------
    #
    # Flat index, `f * n_bins + b`, exactly as the fields were indexed. No
    # bounds check beyond what `List.__getitem__` already did, so these are
    # the same code the direct reads were.

    @always_inline
    def grad_at(self, i: Int) -> Float64:
        return self._grad[i]

    @always_inline
    def hess_at(self, i: Int) -> Float64:
        return self._hess[i]

    @always_inline
    def count_at(self, i: Int) -> Int:
        return self._count[i]

    @always_inline
    def set_grad_at(mut self, i: Int, v: Float64):
        self._grad[i] = v

    @always_inline
    def set_hess_at(mut self, i: Int, v: Float64):
        self._hess[i] = v

    @always_inline
    def set_count_at(mut self, i: Int, v: Int):
        self._count[i] = v

    def n_cells(self) -> Int:
        """Number of cells, `n_features * n_bins`. Was `len(hist.grad)`."""
        return len(self._grad)

    # --- raw plane access -------------------------------------------------
    #
    # There is deliberately no accessor for a whole plane. The few callers
    # that must hand a `List` to something else (the accumulate helpers in
    # this module, the allreduce in `distributed`, the pointer taken for a hot
    # scan) name `hist._grad` / `hist._hess` / `hist._count` directly. The
    # underscore is the marker: `grep -rn '\._grad\b'` is the exact, complete
    # list of what a storage change has to rewrite by hand, and everything
    # else is already going through the cell accessors above.

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
        var gp = self._grad.unsafe_ptr()
        var hp = self._hess.unsafe_ptr()
        var cp = self._count.unsafe_ptr()
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """Zero the slices of every feature this build will not accumulate.

    Only reached under feature subsampling. The active features' slices are
    zeroed by the accumulation pass itself, on the task that is about to fill
    them, so this covers exactly the rest.

    `settings` is the fit's dispatch snapshot; the default sentinel reads the
    environment live, which is what `build_histogram_subset_replica_into`
    (whose signature is frozen for the hybrid scheduler) still does.
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

    dispatch_feature_ranges_with(settings, zero_range, n_features, total_ops)


def build_histogram(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
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
    build_histogram_into(
        out, data, grad, hess, features, const_hessian, settings
    )
    return out^


def build_histogram_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """`build_histogram` into a caller-owned buffer, every cell of which is
    written: the accumulation pass zeroes each slice before filling it, so a
    buffer holding another node's histogram comes back holding exactly this
    one. Identical results to the allocating form, one fewer allocation.

    `settings` is the fit's `parallel.DispatchSettings` snapshot. Passing one
    means this build reads no environment variable and detects no core count:
    it plans its three dispatches from the values the fit resolved once. The
    default sentinel reads them live per build, exactly as this function did
    before the parameter existed, and produces the same histogram cell for
    cell either way -- a snapshot changes only how many tasks the work is cut
    into, and every dispatch shape here keeps each feature's summation inside
    one task."""
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
        out._grad, out._hess, out._count, data, grad, hess, features, const_h,
        settings,
    )


def _plan_accumulation(
    settings: DispatchSettings,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
) raises -> AccumulationPlan:
    """The one place a histogram build decides its shape.

    Both builders go through here so there is a single answer to "did this
    build read the environment". With a resolved snapshot it did not, and the
    `CpuProfile.detect()` that used to sit inside the argument list of every
    `derive_accumulation_plan` call is gone with it; with the sentinel it did,
    once, exactly as before.
    """
    return derive_accumulation_plan_with(
        settings.policy,
        n_features,
        n_active,
        n_bins,
        n_rows_touched,
        rows_are_indirect,
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = _plan_accumulation(
        settings, n_features, n_active, n_bins, n_rows, False
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops, settings,
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
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    elif group >= 8:
        _accumulate_full_at[8](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    elif group >= 4:
        _accumulate_full_at[4](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    elif group >= 2:
        _accumulate_full_at[2](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    else:
        _accumulate_full_at[1](
            out_grad, out_hess, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
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

    dispatch_feature_ranges_with(
        settings, accumulate_groups, n_groups, active_ops
    )


def build_histogram_subset(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> Histogram:
    """Build a histogram over a subset of rows (one tree node's rows). With a
    non-empty `features`, only those features are accumulated and the rest of
    the output stays zero.

    `const_hessian` declares the objective's guarantee that every hessian is
    exactly `CONSTANT_HESSIAN`; the result is bit-identical either way.
    """
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into(
        out, data, grad, hess, rows, 0, len(rows), features, const_hessian,
        settings,
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
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
        const_hessian, settings,
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
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

    `settings` is the fit's dispatch snapshot, and this is the call that
    matters most for it: a grower reaches this function once per node, and
    without a snapshot each of those calls re-detects the machine and
    re-reads five environment variables to plan three dispatches whose answer
    was fixed when the fit started. Passing one reads nothing. It cannot move
    a cell: the gather is elementwise over ascending disjoint blocks and the
    accumulation keeps each feature inside one task at every task count.
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
        out._grad, out._hess, out._count, pairs,
        data, grad, hess, rows, row_start, row_count, features, const_h,
        settings,
    )


def _gather_pairs(
    mut pairs: List[Float64],
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    n_sub: Int,
    gather_ops: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """One gather of the node's (gradient, hessian) pairs into `pairs[0 : 2 *
    n_sub)`, instead of one indirect load per (row, feature).

    Its own function so the mutable pointer it takes into `pairs` dies with
    the call. The blocked accumulation writes into the *tail* of the same
    list, and two live mutable borrows of one list is a thing Mojo's origin
    checker will not carry into a parallel closure.

    Elementwise over disjoint ascending blocks, so the buffer comes out the
    same whatever the task count.
    """
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var pairs_p = pairs.unsafe_ptr()

    def fill_pairs(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = rows_p.unsafe_load(i)
            pairs_p.unsafe_store(2 * i, grad_p.unsafe_load(r))
            pairs_p.unsafe_store(2 * i + 1, hess_p.unsafe_load(r))

    dispatch_rows_with(settings, fill_pairs, n_sub, gather_ops)


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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    var n_bins = data.n_bins
    var n_features = data.n_features
    var n_sub = row_count
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = _plan_accumulation(
        settings, n_features, n_active, n_bins, n_sub, True
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops, settings,
    )

    # The scratch layout, which is this function's to define: the gather
    # occupies `[0, 2 * n_sub)` when it runs, and the row-blocked private
    # histograms occupy `[part_off, part_off + plan.block_scratch_floats())`
    # after it. `pairs` is documented to carry no state between calls beyond
    # its capacity, so extending its tail costs a grower one growth per fit
    # rather than an allocation per node -- which is the whole reason the
    # blocked path can exist without a new parameter on a signature three
    # other files call.
    if plan.compact_rows:
        ensure_pair_capacity(pairs, n_sub)
    var part_off = (2 * n_sub) if plan.compact_rows else 0
    if plan.blocked():
        var wanted = part_off + plan.block_scratch_floats()
        if len(pairs) < wanted:
            pairs.resize(wanted, 0.0)

    if plan.compact_rows:
        _gather_pairs(
            pairs, grad, hess, rows, row_start, n_sub, plan.gather_ops,
            settings,
        )

    if plan.blocked():
        # The same ladder, over the blocked kernel. `plan.row_blocks` is a
        # value decision and not a scheduling one, so it is taken here and
        # never re-derived inside the kernel.
        var bgroup = plan.group_width
        if bgroup >= 16:
            _accumulate_subset_blocked_at[16](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        elif bgroup >= 8:
            _accumulate_subset_blocked_at[8](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        elif bgroup >= 4:
            _accumulate_subset_blocked_at[4](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        elif bgroup >= 2:
            _accumulate_subset_blocked_at[2](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        else:
            _accumulate_subset_blocked_at[1](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        return

    # The ladder, dispatched as in `_accumulate_full` above.
    var group = plan.group_width
    if group >= 16:
        _accumulate_subset_at[16](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    elif group >= 8:
        _accumulate_subset_at[8](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    elif group >= 4:
        _accumulate_subset_at[4](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    elif group >= 2:
        _accumulate_subset_at[2](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    else:
        _accumulate_subset_at[1](
            out_grad, out_hess, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            plan.compact_rows, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
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

    dispatch_feature_ranges_with(
        settings, accumulate_groups, n_groups, active_ops
    )


def _accumulate_subset_blocked_at[
    GROUP: Int
](
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
    plan: AccumulationPlan,
    part_off: Int,
    n_active: Int,
    const_h: Bool,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The row-blocked subset accumulation at one interleave width.

    The node's rows are cut into `plan.row_blocks` contiguous ascending
    blocks. Each block owns a private histogram over the active features only,
    accumulates its own rows into it, and the partials are then folded into
    the caller's output in ascending block order.

    **Two axes, not a replacement for one.** The dispatch unit is a
    `(block, group)` pair and there are `row_blocks * group_count` of them, so
    blocking multiplies the available parallelism rather than trading it: at
    the shape this exists for -- 50 features, width 2, 25 groups -- a node that
    blocks nine ways offers 225 units where the feature partition alone
    offered 25, and the 25 was the same 25 at every node size because it is
    `ceil(n_active / width)` and nothing else. The unit id is `block *
    group_count + group`, so a task holding a contiguous run of units holds one
    block's groups consecutively, which is what keeps that block's slice of the
    gather buffer and its private histogram live in one core's cache across the
    groups that walk them.

    **What is a value here and what is a schedule.** The block count, the
    block boundaries, and the fold order are values: they are fixed by
    `apple_cpu_policy.plan_row_block_count` from the row count, the bin count
    and the active feature count, and by nothing about the machine. How many
    tasks the units are cut into, and which core runs which, are schedule:
    a unit is never split, a block's rows are always walked in ascending
    order, and the fold always runs block 0 first. So the output is identical
    at every `MOJOTREES_NUM_WORKERS`, at every task count, and on every
    machine on this toolchain -- and it is a *different* Float64 from the
    unblocked path, which is the trade this module's docstring now states.

    **The private scratch.** Three planes of `row_blocks * n_active * n_bins`
    Float64 in the tail of the caller's gather buffer, in the order gradient,
    count, hessian. The count plane is Float64 rather than `Int` because the
    buffer is one `List[Float64]`; a count is an integer below 2^53 at every
    step, so every partial sum is exactly representable and the conversion
    back to `Int` in the fold is exact rather than approximate. The hessian
    plane is last so that `const_h`, which never writes it, simply stops
    before it.

    **The elided hessian plane, under blocking.** The three-plane path gives a
    block's hessian cell the sum of `count_b` copies of the Float64 literal
    1.0, which is exactly `Float64(count_b)`, and the fold then sums those
    across blocks -- again exactly, since every partial is an integer below
    2^53. So the three-plane hessian a blocked build produces is
    `Float64(total count)`, which is exactly what the elided path writes from
    the folded count. The elision is bit-identical to the non-elided path on
    the blocked kernel for the same reason it is on the unblocked one, and
    critically the *block count does not depend on `const_h`*: the scratch is
    sized for three planes either way, so a fit cannot move a bin by turning
    the specialization off.

    A tail group owning fewer than `GROUP` features, the SIMD-lane slot
    arrays, and the `compact` pair of row loops are all as in
    `_accumulate_subset_at`.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_sub = row_count
    var n_blocks = plan.row_blocks
    var n_groups = plan.group_count
    var block_rows = plan.block_rows
    var block_cells = plan.block_cells
    var compact = plan.compact_rows
    var use_all = len(features) == 0
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    # ONE pointer into `pairs`, not two. The gather occupies `[0, 2 * n_sub)`
    # and the private histograms `[part_off, ...)`, and two pointers carrying
    # the same origin into one parallel closure is an aliasing error Mojo
    # refuses to compile. So `part_off` is folded into every private index
    # instead, at `base[k]` and at `in0` below, and there is one `pp`.
    var pp = pairs.unsafe_ptr()
    # Distance between the three private planes, which all sit after
    # `part_off`: gradient first, then count, then hessian.
    var plane = n_blocks * block_cells
    comptime W = SIMD_LANES

    def accumulate_units(u_start: Int, u_end: Int) {imm}:
        for u in range(u_start, u_end):
            var blk = u // n_groups
            var grp = u - blk * n_groups
            var r0 = blk * block_rows
            var r1 = r0 + block_rows
            if r1 > n_sub:
                r1 = n_sub
            var slot0 = grp * GROUP
            var owned = n_active - slot0
            if owned > GROUP:
                owned = GROUP
            # `base` indexes the block's private slice by *active slot*, not
            # by feature id: the partials are compact over the active
            # features, so an excluded feature costs no scratch and no fold.
            # `col` still indexes the binned matrix by feature id.
            var base = SIMD[DType.int, GROUP](0)
            var col = SIMD[DType.int, GROUP](0)
            comptime for k in range(GROUP):
                if k < owned:
                    var f = (
                        (slot0 + k) if use_all
                        else feat_p.unsafe_load(slot0 + k)
                    )
                    base[k] = (
                        part_off + blk * block_cells + (slot0 + k) * n_bins
                    )
                    col[k] = f * n_rows

            # Zeroing stays fused into the pass that fills the slice. Under
            # `const_h` the private hessian plane is never written and never
            # read, so it is not zeroed either.
            comptime for k in range(GROUP):
                if k < owned:
                    var z0 = Int(base[k])
                    var zb = 0
                    while zb + W <= n_bins:
                        pp.unsafe_store(
                            z0 + zb, SIMD[DType.float64, W](0.0)
                        )
                        pp.unsafe_store(
                            plane + z0 + zb, SIMD[DType.float64, W](0.0)
                        )
                        if not const_h:
                            pp.unsafe_store(
                                2 * plane + z0 + zb,
                                SIMD[DType.float64, W](0.0),
                            )
                        zb += W
                    while zb < n_bins:
                        pp.unsafe_store(z0 + zb, 0.0)
                        pp.unsafe_store(plane + z0 + zb, 0.0)
                        if not const_h:
                            pp.unsafe_store(2 * plane + z0 + zb, 0.0)
                        zb += 1

            if const_h:
                if compact:
                    for i_row in range(r0, r1):
                        var r = rows_p.unsafe_load(i_row)
                        var g = pp.unsafe_load(2 * i_row)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var b = Int(base[k]) + Int(
                                    bins_all_p.unsafe_load(Int(col[k]) + r)
                                )
                                pp.unsafe_store(
                                    b, pp.unsafe_load(b) + g
                                )
                                pp.unsafe_store(
                                    plane + b,
                                    pp.unsafe_load(plane + b) + 1.0,
                                )
                else:
                    for i_row in range(r0, r1):
                        var r = rows_p.unsafe_load(i_row)
                        var g = grad_p.unsafe_load(r)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var b = Int(base[k]) + Int(
                                    bins_all_p.unsafe_load(Int(col[k]) + r)
                                )
                                pp.unsafe_store(
                                    b, pp.unsafe_load(b) + g
                                )
                                pp.unsafe_store(
                                    plane + b,
                                    pp.unsafe_load(plane + b) + 1.0,
                                )
            elif compact:
                for i_row in range(r0, r1):
                    var r = rows_p.unsafe_load(i_row)
                    var g = pp.unsafe_load(2 * i_row)
                    var h = pp.unsafe_load(2 * i_row + 1)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(
                                plane + b, pp.unsafe_load(plane + b) + 1.0
                            )
                            pp.unsafe_store(
                                2 * plane + b,
                                pp.unsafe_load(2 * plane + b) + h,
                            )
            else:
                for i_row in range(r0, r1):
                    var r = rows_p.unsafe_load(i_row)
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(
                                plane + b, pp.unsafe_load(plane + b) + 1.0
                            )
                            pp.unsafe_store(
                                2 * plane + b,
                                pp.unsafe_load(2 * plane + b) + h,
                            )

    dispatch_feature_ranges_with(
        settings, accumulate_units, n_blocks * n_groups, plan.block_ops
    )

    # The fold. One task per contiguous run of active slots; a slot writes
    # only its own output slice, and inside it every cell sums the blocks in
    # ascending order. That inner order is what the task count cannot touch,
    # and it is the whole of the determinism argument for this kernel.
    def fold_slots(s_start: Int, s_end: Int) {imm}:
        for j in range(s_start, s_end):
            var f = j if use_all else feat_p.unsafe_load(j)
            var out0 = f * n_bins
            var in0 = part_off + j * n_bins
            var b = 0
            while b + W <= n_bins:
                var sg = pp.unsafe_load[width=W](in0 + b)
                var sc = pp.unsafe_load[width=W](plane + in0 + b)
                var sh = SIMD[DType.float64, W](0.0)
                if not const_h:
                    sh = pp.unsafe_load[width=W](2 * plane + in0 + b)
                for blk in range(1, n_blocks):
                    var off = blk * block_cells + in0 + b
                    sg += pp.unsafe_load[width=W](off)
                    sc += pp.unsafe_load[width=W](plane + off)
                    if not const_h:
                        sh += pp.unsafe_load[width=W](2 * plane + off)
                gp.unsafe_store(out0 + b, sg)
                cp.unsafe_store(out0 + b, sc.cast[DType.int]())
                hp.unsafe_store(out0 + b, sc if const_h else sh)
                b += W
            while b < n_bins:
                var sg1 = pp.unsafe_load(in0 + b)
                var sc1 = pp.unsafe_load(plane + in0 + b)
                var sh1 = Float64(0.0)
                if not const_h:
                    sh1 = pp.unsafe_load(2 * plane + in0 + b)
                for blk in range(1, n_blocks):
                    var off = blk * block_cells + in0 + b
                    sg1 += pp.unsafe_load(off)
                    sc1 += pp.unsafe_load(plane + off)
                    if not const_h:
                        sh1 += pp.unsafe_load(2 * plane + off)
                gp.unsafe_store(out0 + b, sg1)
                cp.unsafe_store(out0 + b, Int(sc1))
                hp.unsafe_store(out0 + b, sc1 if const_h else sh1)
                b += 1

    dispatch_feature_ranges_with(
        settings, fold_slots, n_active, plan.fold_ops
    )


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
        out._grad, out._hess, out._count,
        n_features, n_bins, features,
        (n_features - n_active) * n_bins,
    )

    # The three output buffers are passed as separate `mut` lists rather than
    # reached through `out`, exactly as `build_histogram_into` explains: a
    # pointer taken from a struct field carries that field's origin, which a
    # worker closure cannot capture.
    _accumulate_replica(
        out._grad, out._hess, out._count, fixed,
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
    var grad_p = hist._grad.unsafe_ptr()
    var hess_p = hist._hess.unsafe_ptr()
    var count_p = hist._count.unsafe_ptr()
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
    parent: Histogram,
    child: Histogram,
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> Histogram:
    """Sibling histogram via the subtraction trick: build the smaller child
    directly, get the larger one as parent - child for free."""
    var out = Histogram.zeroed(parent.n_features, parent.n_bins)
    subtract_histogram_into(out, parent, child, const_hessian, settings)
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
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
    dispatch_rows_with(settings, subtract_block, size, ops)


def subtract_histogram_into(
    mut out: Histogram,
    parent: Histogram,
    child: Histogram,
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
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
        out._grad,
        out._hess,
        out._count,
        parent._grad,
        parent._hess,
        parent._count,
        child._grad,
        child._hess,
        child._count,
        parent.n_features * parent.n_bins,
        _resolve_const_hessian(const_hessian),
        settings,
    )


# ---------------------------------------------------------------------------
# The row-major accumulation
# ---------------------------------------------------------------------------
#
# The same histogram, read out of `BinnedMatrix.row_bins` instead of
# `BinnedMatrix.bins`. LightGBM calls this pair `force_row_wise` and
# `force_col_wise` and picks between them by timing one construction of each
# at the start of the fit; `choose_bin_layout_timed` below is that probe.
#
# **Which builder reads which array, stated here because a GPU reader will
# look in this file for it.** Every builder in this module except the
# `_row_major` ones reads `bins`, the feature-major column array, and always
# has: `build_histogram`, `build_histogram_subset` and all their `_into`
# forms, the sibling subtraction, and `build_histogram_subset_replica_into`
# (the hybrid scheduler's host replica). **Every GPU path also reads `bins`**
# -- the device histogram kernels upload and scatter the column array,
# `train_gpu`'s resident data plane holds the column array, and nothing on the
# device is ever handed `row_bins`. A device scatter wants the coalesced
# column, not the record, so the group-major / feature-blocked device layout
# question is a *separate* decision from this one and neither constrains the
# other. Turning `MOJOTREES_CPU_ROW_MAJOR` on cannot change a device result;
# it costs the fit one extra copy of the bin matrix and nothing else. The
# byte figure is on `binning.BinnedMatrix`.
#
# **These kernels move no bits, and that is the whole correctness claim.** A
# row-major build visits the same rows in the same order and adds the same
# Float64 into the same cell as its feature-major twin; only the address the
# bin id is loaded from changes. Three things make that hold rather than
# merely be intended:
#
# 1. **One plan, two arms.** Both arms are run against the *same*
#    `AccumulationPlan`, taken from the same `_plan_accumulation` call on the
#    same arguments. `plan.row_blocks` is the one field that is part of the
#    value (see `apple_cpu_policy`'s docstring), so the layout choice must
#    never be allowed to reach it, and here it cannot: the layout is not an
#    argument to the plan.
# 2. **Ascending rows inside a block, ascending blocks in the fold.** Both are
#    preserved verbatim from `_accumulate_subset_blocked_at`.
# 3. **The group width is free.** The interleave width partitions *features*
#    across tasks and never reassociates a cell's sum, which is why the
#    blocked row-major kernel below can put every active feature in one inner
#    loop -- the thing that makes the layout worth having -- without that
#    being a numerics change.
#
# What the two arms actually differ in is memory traffic, and the argument is
# in the lane report rather than in a cost function here: LightGBM's entire
# auto rule is the timed shot, and a cost model nobody measured would be
# tuning by another name.

comptime FOLD_BIN_CHUNK = 4 * SIMD_LANES
"""Bins one fold task takes from one feature's slice.

The row-major fold dispatches over `(active slot, bin chunk)` pairs rather
than over whole slots, which is LightGBM's `HistMerge` shape -- they reduce
their per-thread buffers with `schedule(static, 1)` over bin ranges.

**Why parallelizing the fold is legal here and does not reassociate
anything.** A unit owns the output range `[f * n_bins + b0, f * n_bins + b1)`
and nothing else: no other unit reads it and no other unit writes it. Inside
the unit, each cell sums its blocks in ascending block order, in a loop the
task count cannot enter. So the fold's parallel structure decides *which core*
sums a cell and never *in what order*, and the histogram is identical at every
`MOJOTREES_NUM_WORKERS`. That is exactly the argument the feature-major fold
makes for its slot ranges; splitting the bins as well only makes the units
smaller.
"""


def _row_major_slot_meta(
    data: BinnedMatrix,
    features: List[Int],
    n_active: Int,
    mut meta: List[Int],
) raises -> Int:
    """Per-active-slot constants for the row-major kernels, and the compact
    cell count.

    Four Ints per slot, interleaved so one cache line carries several slots:
    the slot's offset into a compact histogram, the byte its feature occupies
    in a row record, the nibble shift, and the nibble mask. A sentinel slot at
    the end holds the total, so slot `j`'s width is
    `meta[4 * (j + 1)] - meta[4 * j]` with no special case for the last one.

    The offsets are LightGBM's `group_bin_boundaries_` idea applied to the
    private accumulator: a compact histogram over the active features costs
    `sum_j feature_bins[f_j]` cells instead of `n_active * n_bins`. On a
    dataset whose features average 40 realized bins out of a 255-bin budget
    that is a sixth of the storage, which is the difference between a private
    partial that lives in L2 and one that does not. It never reaches the
    output, which keeps its `f * n_bins + b` shape.

    One `List[Int]` per build, `4 * (n_active + 1)` entries -- 1.6 KB at 50
    features. In the same class as the active mask `_zero_excluded` already
    allocates per call, and three orders of magnitude below the gather buffer
    the same call reuses.
    """
    meta.resize(4 * (n_active + 1), 0)
    var use_all = len(features) == 0
    var off = 0
    for j in range(n_active):
        var f = j if use_all else features[j]
        meta[4 * j] = off
        meta[4 * j + 1] = data.row_byte[f]
        meta[4 * j + 2] = data.row_shift[f]
        meta[4 * j + 3] = data.row_mask[f]
        off += data.feature_bins[f]
    meta[4 * n_active] = off
    return off


def row_major_line_floats() -> Int:
    """Float64 slots one private partial is padded to a multiple of.

    Taken from `ASSUMED_CACHE_LINE_BYTES`, a compilation-target fact, rather
    than from the detected profile: padding to a line that is too *large*
    costs a few hundred bytes of scratch and padding to one too small costs
    two cores a contested line on every store, so the conservative direction
    is up, and the target's line is never smaller than the machine's.
    """
    return histogram_line_floats(ASSUMED_CACHE_LINE_BYTES)


def row_major_scratch_floats(
    plan: AccumulationPlan, compact_cells: Int
) raises -> Int:
    """Float64 slots the row-major blocked kernel needs in the caller's
    scratch, gather region included. Zero for an unblocked plan.

    Deliberately *not* `AccumulationPlan.block_scratch_floats`: that sizes the
    feature-major partials, which are `n_active * n_bins` cells with no
    padding. These are compact (`compact_cells`) and line-padded, so they are
    a different number, and a scratch sized from one and indexed by the other
    is the failure this function exists to make impossible.
    """
    if not plan.blocked():
        return 0
    var line = row_major_line_floats()
    var part_off = row_major_part_offset(plan)
    return part_off + ROW_BLOCK_PLANES * plan.row_blocks * align_cells_up(
        compact_cells, line
    )


def row_major_part_offset(plan: AccumulationPlan) raises -> Int:
    """Where the private partials start in the caller's scratch: after the
    gather region, rounded up to a cache line so block 0's partials do not
    share a line with the last gathered pair."""
    var line = row_major_line_floats()
    if not plan.compact_rows:
        return 0
    return align_cells_up(2 * plan.block_rows * plan.row_blocks, line)


def build_histogram_subset_row_major(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> Histogram:
    """`build_histogram_subset` over the row-major view of the same data.

    Bit-identical to `build_histogram_subset` on the same arguments. Raises if
    `data` has no row-major view, because silently reading the other layout
    would make a benchmark arm indistinguishable from its control -- which is
    the failure this project has already had twice.
    """
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    var pairs = List[Float64]()
    build_histogram_subset_row_major_into_scratch(
        out, pairs, data, grad, hess, rows, 0, len(rows), features,
        const_hessian, settings,
    )
    return out^


def build_histogram_subset_row_major_into_scratch(
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The row-major twin of `build_histogram_subset_into_scratch`.

    Same signature, same scratch contract (`pairs` grows and never shrinks,
    carries no state between calls beyond its capacity), same output. The
    scratch it wants for a blocked plan is `row_major_scratch_floats`, which
    is a different number from the feature-major kernel's; a grower that runs
    both arms simply keeps the larger, since the buffer is written before it
    is read.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    if not data.has_row_major():
        raise Error(
            "the row-major builder needs BinnedMatrix.build_row_major();"
            " set MOJOTREES_CPU_ROW_MAJOR=1 to have fit-time binning build it"
        )
    _check_features(features, data.n_features)
    var const_h = _resolve_const_hessian(const_hessian)
    if const_h and const_hessian_verify():
        _check_constant_hessian(hess, data.n_rows)

    _accumulate_subset_row_major(
        out._grad, out._hess, out._count, pairs,
        data, grad, hess, rows, row_start, row_count, features, const_h,
        settings,
    )


def _accumulate_subset_row_major(
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The row-major twin of `_accumulate_subset`, plan for plan.

    The plan comes from the same `_plan_accumulation` call on the same
    arguments, so `row_blocks`, `block_rows` and `compact_rows` are whatever
    the feature-major build would have used. Only the kernel differs.
    """
    var n_bins = data.n_bins
    var n_features = data.n_features
    var n_sub = row_count
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = _plan_accumulation(
        settings, n_features, n_active, n_bins, n_sub, True
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops, settings,
    )
    if n_active <= 0 or n_sub <= 0:
        # Nothing active: the excluded pass above has already zeroed every
        # slice this build owns, and an empty row window leaves the active
        # ones zero too. Handled here rather than inside a kernel so neither
        # kernel needs an empty case.
        if n_active > 0:
            _zero_active_slices(
                out_grad, out_hess, out_count, n_features, n_bins, features,
                n_active, plan.active_ops, settings,
            )
        return

    if plan.compact_rows:
        ensure_pair_capacity(pairs, n_sub)

    if not plan.blocked():
        if plan.compact_rows:
            _gather_pairs(
                pairs, grad, hess, rows, row_start, n_sub, plan.gather_ops,
                settings,
            )
        var group = plan.group_width
        if group >= 16:
            _accumulate_subset_row_major_at[16](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan.compact_rows,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        elif group >= 8:
            _accumulate_subset_row_major_at[8](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan.compact_rows,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        elif group >= 4:
            _accumulate_subset_row_major_at[4](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan.compact_rows,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        elif group >= 2:
            _accumulate_subset_row_major_at[2](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan.compact_rows,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        else:
            _accumulate_subset_row_major_at[1](
                out_grad, out_hess, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan.compact_rows,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        return

    var meta = List[Int]()
    var cells = _row_major_slot_meta(data, features, n_active, meta)
    var part_off = row_major_part_offset(plan)
    var wanted = row_major_scratch_floats(plan, cells)
    if len(pairs) < wanted:
        pairs.resize(wanted, 0.0)
    if plan.compact_rows:
        _gather_pairs(
            pairs, grad, hess, rows, row_start, n_sub, plan.gather_ops,
            settings,
        )
    _accumulate_subset_row_major_blocked(
        out_grad, out_hess, out_count, pairs, data, grad, hess, rows,
        row_start, row_count, features, plan, part_off, n_active, cells,
        meta, const_h, settings,
    )


def _zero_active_slices(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    n_features: Int,
    n_bins: Int,
    features: List[Int],
    n_active: Int,
    total_ops: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """Zero the active features' slices, for the degenerate empty-window case
    the kernels are not asked to handle.

    The accumulating kernels fuse their zeroing into the pass that fills the
    slice, so with no rows to walk there is no pass and the slices would keep
    whatever the caller's buffer held. `build_histogram_subset_into` documents
    that every cell of the output is written, and an empty node is exactly the
    case where a caller recycling a histogram would notice it was not.
    """
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var use_all = len(features) == 0
    comptime W = SIMD_LANES

    def zero_slots(s_start: Int, s_end: Int) {imm}:
        for j in range(s_start, s_end):
            var f = j if use_all else feat_p.unsafe_load(j)
            var base = f * n_bins
            var b = 0
            while b + W <= n_bins:
                gp.unsafe_store(base + b, SIMD[DType.float64, W](0.0))
                hp.unsafe_store(base + b, SIMD[DType.float64, W](0.0))
                cp.unsafe_store(base + b, SIMD[DType.int, W](0))
                b += W
            while b < n_bins:
                gp.unsafe_store(base + b, 0.0)
                hp.unsafe_store(base + b, 0.0)
                cp.unsafe_store(base + b, 0)
                b += 1

    dispatch_feature_ranges_with(settings, zero_slots, n_active, total_ops)


def _accumulate_subset_row_major_at[
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The unblocked row-major accumulation at one interleave width.

    `_accumulate_subset_at` with three loads changed. Where that kernel reads
    `bins[col[k] + r]`, one byte per (row, feature) from `GROUP` columns
    `n_rows` apart, this one reads byte `rbyte[k]` of row `r`'s record and
    shifts out a nibble. The `GROUP` bytes a group needs come from one record,
    so they are one cache line rather than `GROUP` of them -- which is worth
    everything on a node whose rows are scattered and nothing on a node whose
    rows are the whole dataset, since the feature-major columns are then read
    sequentially and every byte of every line is used.

    The shift and mask are per-feature constants hoisted into SIMD lane
    arrays exactly as `base` and `col` are, so the row loop pays two integer
    operations per (row, feature) and no branch. An unpacked feature carries
    shift 0 and mask 255, so the packed and unpacked cases are one code path.

    Every structural claim `_accumulate_subset_at` makes -- fused zeroing,
    tail groups, the `compact` pair of row loops, the elided hessian plane and
    its exact refill from the count -- holds here unchanged, because none of
    them is about where a bin id was loaded from.
    """
    var n_bins = data.n_bins
    var n_sub = row_count
    var stride = data.row_stride
    var use_all = len(features) == 0
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var rm_p = data.row_bins.unsafe_ptr()
    var byte_p = data.row_byte.unsafe_ptr()
    var shift_p = data.row_shift.unsafe_ptr()
    var mask_p = data.row_mask.unsafe_ptr()
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
            var rbyte = SIMD[DType.int, GROUP](0)
            var rshift = SIMD[DType.int, GROUP](0)
            var rmask = SIMD[DType.int, GROUP](0)
            comptime for k in range(GROUP):
                if k < owned:
                    var f = (
                        (slot0 + k) if use_all
                        else feat_p.unsafe_load(slot0 + k)
                    )
                    base[k] = f * n_bins
                    rbyte[k] = byte_p.unsafe_load(f)
                    rshift[k] = shift_p.unsafe_load(f)
                    rmask[k] = mask_p.unsafe_load(f)

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
                        var rec = rows_p.unsafe_load(i_row) * stride
                        var g = pairs_p.unsafe_load(2 * i_row)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var raw = Int(
                                    rm_p.unsafe_load(rec + Int(rbyte[k]))
                                )
                                var b = Int(base[k]) + (
                                    (raw >> Int(rshift[k])) & Int(rmask[k])
                                )
                                gp.unsafe_store(b, gp.unsafe_load(b) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
                else:
                    for i_row in range(n_sub):
                        var r = rows_p.unsafe_load(i_row)
                        var rec = r * stride
                        var g = grad_p.unsafe_load(r)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var raw = Int(
                                    rm_p.unsafe_load(rec + Int(rbyte[k]))
                                )
                                var b = Int(base[k]) + (
                                    (raw >> Int(rshift[k])) & Int(rmask[k])
                                )
                                gp.unsafe_store(b, gp.unsafe_load(b) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            elif compact:
                for i_row in range(n_sub):
                    var rec = rows_p.unsafe_load(i_row) * stride
                    var g = pairs_p.unsafe_load(2 * i_row)
                    var h = pairs_p.unsafe_load(2 * i_row + 1)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var raw = Int(
                                rm_p.unsafe_load(rec + Int(rbyte[k]))
                            )
                            var b = Int(base[k]) + (
                                (raw >> Int(rshift[k])) & Int(rmask[k])
                            )
                            gp.unsafe_store(b, gp.unsafe_load(b) + g)
                            hp.unsafe_store(b, hp.unsafe_load(b) + h)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    var rec = r * stride
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var raw = Int(
                                rm_p.unsafe_load(rec + Int(rbyte[k]))
                            )
                            var b = Int(base[k]) + (
                                (raw >> Int(rshift[k])) & Int(rmask[k])
                            )
                            gp.unsafe_store(b, gp.unsafe_load(b) + g)
                            hp.unsafe_store(b, hp.unsafe_load(b) + h)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

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

    dispatch_feature_ranges_with(
        settings, accumulate_groups, n_groups, active_ops
    )


def _accumulate_subset_row_major_blocked(
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
    plan: AccumulationPlan,
    part_off: Int,
    n_active: Int,
    compact_cells: Int,
    meta: List[Int],
    const_h: Bool,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The blocked row-major accumulation. This is LightGBM's row-wise
    builder, and it is the shape the layout exists for.

    **One axis, not two, and that is the point.** The feature-major blocked
    kernel dispatches over `(block, group)` pairs because a group must be
    narrow enough that its histogram slices stay in L1. This one dispatches
    over **blocks alone**: a block owns a private histogram over *every*
    active feature and walks its rows once, reading each row's whole record.
    Three consequences, and only the third is a trade:

    - The node's gradients are read **once**, not once per group. The
      feature-major kernel re-streams the gather buffer `ceil(n_active /
      group_width)` times, which at 50 features and width 2 is 25 passes.
    - A row's record is one cache line and serves all `n_active` features,
      where the feature-major kernel pays one line per (row, feature) on a
      node whose rows are scattered.
    - The private histogram is `sum_j feature_bins[f_j]` cells rather than
      `group_width * n_bins`, so it is bigger than L1 where the feature-major
      one was sized to fit. That is the cost, and it is why the compact
      cumulative offsets above matter rather than being tidiness: they are
      what keeps the accumulator in L2 instead of past it.

    **Parallelism comes from blocks only, so `plan.row_blocks` is the whole
    fan-out** for the accumulate pass. The plan is not allowed to grow it for
    this kernel's benefit -- the block count is part of the value, and a
    layout that changed it would move bits. A node the plan does not block is
    therefore not a node for this kernel; `_accumulate_subset_row_major` sends
    those to the unblocked row-major kernel, which partitions by feature as
    before.

    **Padding.** Each block's partials are padded up to a cache line
    (`row_major_line_floats`), which is LightGBM's `num_bin_aligned_`. Without
    it two blocks' partials share the line at their boundary and two cores
    contest it on every store. The gather region is padded to a line for the
    same reason before the partials start.

    **Determinism**, unchanged from the feature-major blocked kernel: a
    block's rows are walked ascending, a block is never split across tasks,
    and the fold sums blocks ascending inside each cell. The fold's own
    parallel split is over disjoint output ranges and cannot reassociate; see
    `FOLD_BIN_CHUNK`.

    **The elided hessian plane** is exact for the reason
    `_accumulate_subset_blocked_at` gives: a block's hessian cell under
    `const_h` would have been `count_b` copies of 1.0, which is exactly
    `Float64(count_b)`, every partial is an integer below 2^53, and the fold
    of those is exact. The block count does not depend on `const_h`, so a fit
    cannot move a bin by turning the specialization off.

    **The tail bins.** A compact slot holds `feature_bins[f]` cells and the
    output slice holds `n_bins`. The bins above the highest one the feature
    realizes are written as zero by the fold, which is bit for bit what the
    feature-major kernel leaves there: no row ever lands in them, so its
    accumulated gradient is `0.0`, its count is `0`, and its hessian is `0.0`
    on both the three-plane and the elided path.
    """
    var n_bins = data.n_bins
    var n_sub = row_count
    var n_blocks = plan.row_blocks
    var block_rows = plan.block_rows
    var compact = plan.compact_rows
    var stride = data.row_stride
    var use_all = len(features) == 0
    var aligned = align_cells_up(compact_cells, row_major_line_floats())
    var plane = n_blocks * aligned
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var rm_p = data.row_bins.unsafe_ptr()
    var meta_p = meta.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    # ONE pointer into `pairs`: the gather occupies `[0, 2 * n_sub)` and the
    # partials `[part_off, ...)`, and two mutable pointers with the same
    # origin in one parallel closure is an aliasing error Mojo refuses.
    var pp = pairs.unsafe_ptr()
    comptime W = SIMD_LANES

    def accumulate_blocks(u_start: Int, u_end: Int) {imm}:
        for blk in range(u_start, u_end):
            var r0 = blk * block_rows
            var r1 = r0 + block_rows
            if r1 > n_sub:
                r1 = n_sub
            var b0 = part_off + blk * aligned

            # Zeroing stays fused into the pass that fills the slice. Only the
            # `compact_cells` in use are zeroed; the line padding beyond them
            # is never read.
            var zb = 0
            while zb + W <= compact_cells:
                pp.unsafe_store(b0 + zb, SIMD[DType.float64, W](0.0))
                pp.unsafe_store(plane + b0 + zb, SIMD[DType.float64, W](0.0))
                if not const_h:
                    pp.unsafe_store(
                        2 * plane + b0 + zb, SIMD[DType.float64, W](0.0)
                    )
                zb += W
            while zb < compact_cells:
                pp.unsafe_store(b0 + zb, 0.0)
                pp.unsafe_store(plane + b0 + zb, 0.0)
                if not const_h:
                    pp.unsafe_store(2 * plane + b0 + zb, 0.0)
                zb += 1

            if const_h:
                if compact:
                    for i_row in range(r0, r1):
                        var rec = rows_p.unsafe_load(i_row) * stride
                        var g = pp.unsafe_load(2 * i_row)
                        for j in range(n_active):
                            var m = 4 * j
                            var raw = Int(
                                rm_p.unsafe_load(
                                    rec + meta_p.unsafe_load(m + 1)
                                )
                            )
                            var b = (
                                b0
                                + meta_p.unsafe_load(m)
                                + (
                                    (raw >> meta_p.unsafe_load(m + 2))
                                    & meta_p.unsafe_load(m + 3)
                                )
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(
                                plane + b, pp.unsafe_load(plane + b) + 1.0
                            )
                else:
                    for i_row in range(r0, r1):
                        var r = rows_p.unsafe_load(i_row)
                        var rec = r * stride
                        var g = grad_p.unsafe_load(r)
                        for j in range(n_active):
                            var m = 4 * j
                            var raw = Int(
                                rm_p.unsafe_load(
                                    rec + meta_p.unsafe_load(m + 1)
                                )
                            )
                            var b = (
                                b0
                                + meta_p.unsafe_load(m)
                                + (
                                    (raw >> meta_p.unsafe_load(m + 2))
                                    & meta_p.unsafe_load(m + 3)
                                )
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(
                                plane + b, pp.unsafe_load(plane + b) + 1.0
                            )
            elif compact:
                for i_row in range(r0, r1):
                    var rec = rows_p.unsafe_load(i_row) * stride
                    var g = pp.unsafe_load(2 * i_row)
                    var h = pp.unsafe_load(2 * i_row + 1)
                    for j in range(n_active):
                        var m = 4 * j
                        var raw = Int(
                            rm_p.unsafe_load(rec + meta_p.unsafe_load(m + 1))
                        )
                        var b = (
                            b0
                            + meta_p.unsafe_load(m)
                            + (
                                (raw >> meta_p.unsafe_load(m + 2))
                                & meta_p.unsafe_load(m + 3)
                            )
                        )
                        pp.unsafe_store(b, pp.unsafe_load(b) + g)
                        pp.unsafe_store(
                            plane + b, pp.unsafe_load(plane + b) + 1.0
                        )
                        pp.unsafe_store(
                            2 * plane + b, pp.unsafe_load(2 * plane + b) + h
                        )
            else:
                for i_row in range(r0, r1):
                    var r = rows_p.unsafe_load(i_row)
                    var rec = r * stride
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    for j in range(n_active):
                        var m = 4 * j
                        var raw = Int(
                            rm_p.unsafe_load(rec + meta_p.unsafe_load(m + 1))
                        )
                        var b = (
                            b0
                            + meta_p.unsafe_load(m)
                            + (
                                (raw >> meta_p.unsafe_load(m + 2))
                                & meta_p.unsafe_load(m + 3)
                            )
                        )
                        pp.unsafe_store(b, pp.unsafe_load(b) + g)
                        pp.unsafe_store(
                            plane + b, pp.unsafe_load(plane + b) + 1.0
                        )
                        pp.unsafe_store(
                            2 * plane + b, pp.unsafe_load(2 * plane + b) + h
                        )

    dispatch_feature_ranges_with(
        settings, accumulate_blocks, n_blocks, plan.block_ops
    )

    var n_chunks = (n_bins + FOLD_BIN_CHUNK - 1) // FOLD_BIN_CHUNK
    if n_chunks < 1:
        n_chunks = 1

    def fold_units(u_start: Int, u_end: Int) {imm}:
        for u in range(u_start, u_end):
            var j = u // n_chunks
            var c = u - j * n_chunks
            var f = j if use_all else feat_p.unsafe_load(j)
            var in0 = part_off + meta_p.unsafe_load(4 * j)
            var width = meta_p.unsafe_load(4 * (j + 1)) - meta_p.unsafe_load(
                4 * j
            )
            var out0 = f * n_bins
            var lo = c * FOLD_BIN_CHUNK
            var hi = lo + FOLD_BIN_CHUNK
            if hi > n_bins:
                hi = n_bins
            var lim = width if width < hi else hi
            var b = lo
            while b + W <= lim:
                var sg = pp.unsafe_load[width=W](in0 + b)
                var sc = pp.unsafe_load[width=W](plane + in0 + b)
                var sh = SIMD[DType.float64, W](0.0)
                if not const_h:
                    sh = pp.unsafe_load[width=W](2 * plane + in0 + b)
                for blk in range(1, n_blocks):
                    var off = blk * aligned + in0 + b
                    sg += pp.unsafe_load[width=W](off)
                    sc += pp.unsafe_load[width=W](plane + off)
                    if not const_h:
                        sh += pp.unsafe_load[width=W](2 * plane + off)
                gp.unsafe_store(out0 + b, sg)
                cp.unsafe_store(out0 + b, sc.cast[DType.int]())
                hp.unsafe_store(out0 + b, sc if const_h else sh)
                b += W
            while b < lim:
                var sg1 = pp.unsafe_load(in0 + b)
                var sc1 = pp.unsafe_load(plane + in0 + b)
                var sh1 = Float64(0.0)
                if not const_h:
                    sh1 = pp.unsafe_load(2 * plane + in0 + b)
                for blk in range(1, n_blocks):
                    var off = blk * aligned + in0 + b
                    sg1 += pp.unsafe_load(off)
                    sc1 += pp.unsafe_load(plane + off)
                    if not const_h:
                        sh1 += pp.unsafe_load(2 * plane + off)
                gp.unsafe_store(out0 + b, sg1)
                cp.unsafe_store(out0 + b, Int(sc1))
                hp.unsafe_store(out0 + b, sc1 if const_h else sh1)
                b += 1
            # The bins this feature never realizes. Zero on both paths, which
            # is what the feature-major kernel leaves there.
            while b < hi:
                gp.unsafe_store(out0 + b, 0.0)
                hp.unsafe_store(out0 + b, 0.0)
                cp.unsafe_store(out0 + b, 0)
                b += 1

    dispatch_feature_ranges_with(
        settings, fold_units, n_active * n_chunks, plan.fold_ops
    )


def build_histogram_subset_by_layout_into_scratch(
    mut out: Histogram,
    mut pairs: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    layout: Int,
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> Int:
    """Build one node's histogram in the requested layout, and **return the
    layout that actually ran**.

    The return value is the observable. A layout that is requested but not
    available degrades to feature-major (`resolve_bin_layout`), and a caller
    or a test that only asserted the request would be asserting nothing --
    this project has already shipped one test comparing two arms that were
    equal whether or not the optimization fired, and one whose six fixtures
    all ran below the gate. So the gate reports itself.

    `BIN_LAYOUT_AUTO` runs feature-major here rather than timing anything. The
    timed choice is a once-per-fit decision (`choose_bin_layout_timed`), which
    is where LightGBM makes it; timing it per node would spend the measurement
    on the thing it is trying to speed up.
    """
    var chosen = resolve_bin_layout(layout, data.has_row_major())
    if chosen == BIN_LAYOUT_ROW_MAJOR:
        build_histogram_subset_row_major_into_scratch(
            out, pairs, data, grad, hess, rows, row_start, row_count,
            features, const_hessian, settings,
        )
        return BIN_LAYOUT_ROW_MAJOR
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian, settings,
    )
    return BIN_LAYOUT_FEATURE_MAJOR


def choose_bin_layout_timed(
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> Int:
    """LightGBM's auto rule: build the node once each way, keep the faster.

    Their whole `force_row_wise` / `force_col_wise` decision is one timed
    construction with each builder at the start of the fit, kept for the
    remainder of it and printed. This is that probe, and a fit should call it
    **once**, on its first node, and thread the answer through
    `build_histogram_subset_by_layout_into_scratch` for every node after.

    `MOJOTREES_CPU_BIN_LAYOUT` is honored first: an explicit arm is an
    instruction, not a hint, and a benchmark that asked for one layout and
    silently got the other is a discarded result.

    **`out` holds a correct histogram on return whichever arm won**, because
    the two arms are bit-identical -- which is what makes a probe that runs
    both affordable at all. It costs one extra build of one node per fit, of
    the thousands a fit performs.

    One clock, `perf_counter_ns`, around one build each, on the machine the
    fit is running on. Two builds are not a benchmark and this function does
    not pretend otherwise: it is a tie-break with a real bias (the second arm
    runs with the caches the first one warmed). LightGBM accepts the same
    bias. It is still strictly better information than a cost model nobody
    measured, and the arms are exchangeable enough that the bias costs a wrong
    answer only where the two are close, which is where it does not matter.
    """
    if not data.has_row_major():
        return BIN_LAYOUT_FEATURE_MAJOR
    var requested = resolve_bin_layout(env_bin_layout(), True)
    if requested != BIN_LAYOUT_AUTO:
        return requested

    var t0 = perf_counter_ns()
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian, settings,
    )
    var t1 = perf_counter_ns()
    build_histogram_subset_row_major_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian, settings,
    )
    var t2 = perf_counter_ns()
    if (t2 - t1) < (t1 - t0):
        return BIN_LAYOUT_ROW_MAJOR
    return BIN_LAYOUT_FEATURE_MAJOR
