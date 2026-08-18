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
recycle a handful of histograms instead of allocating two arrays per node
(see `Histogram.zeroed` / `Histogram.reset`). The two forms run the same
kernels and produce bit-identical results; only the allocation differs.

Three things about the CPU shape of these kernels, all of them scheduling or
memory traffic and none of them arithmetic:

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

**What the count and sibling subtraction do, which is not affected.**
A blocked count is counted, not summed from hessians, and integer addition
*is* associative, so it is the same integer at any block count. Under the constant-hessian
specialization the hessian plane is `Float64(count)` on both operands of a
sibling subtraction, and both are exactly representable integers below 2^53,
so that subtraction is exact under blocking exactly as it was without it. The
gradient plane's subtraction was never bit-equal to a direct build of the
sibling and is not claimed to be; blocking does not change that either way.
See `_subtract_histogram_arrays`.

The GPU path escapes the reassociation argument entirely, rather than
accepting it, because it accumulates in fixed-point Int32 where addition *is*
associative; see `_range_hist_partial_kernel` in `gpu_active_rows.mojo`.

**Both builders block, on one plan, and that is a correctness property.**
The full-dataset builder used to be excluded, on the grounds that it has no
caller-owned scratch and would have to allocate the private histograms per
call. That was a cost argument standing in front of a correctness one: while
only the subset builder blocked, "growing on a bag is growing on the dataset
of those rows" was false, because the bagged tree reached the subset builder
and folded row blocks while its whole-dataset reference summed flat. It was
measured at four ulp on a leaf value in
`test_bagged_tree_equals_tree_on_subset_dataset`, which now asserts bit
identity again. The cost is answered rather than accepted:
`build_histogram_into_scratch` lets a caller own the buffer, and this builder
is reached once per tree where the subset builder is reached once per node.
The alternative -- have the subset builder skip blocking when its row list
happens to cover the whole matrix -- puts a discontinuity at `len(bag) ==
n_rows`, which is worse.

A node too small for a block to amortize its own private histogram still never
blocks -- `apple_cpu_policy.row_block_min_rows` is the floor and its derivation
is there -- so small and tiny nodes keep the accumulation, and the bytes, they

**And so does the CPU quantized path, which is now built on this same
decomposition.** `quantized_gradient.build_histogram_subset_quantized_into_scratch`
is the row-blocked kernel above over interleaved Int64 cells: the same
`plan_row_block_count`, the same contiguous ascending blocks, the same private
partials, the same ascending fold. Integer addition is associative, so on that
path the fold is **exact** rather than merely deterministic, and every
paragraph above that had to argue the block count is a *value* stops applying:
`MOJOTREES_CPU_ROW_BLOCKS` cannot move a quantized cell at any setting, so it
is a pure scheduling A/B there and an answer-changing knob only here. It also
means the full-dataset builder and the subset builder, which sum the same rows
in two orders and so differ by a few ulp in Float64, agree bit for bit once
both are quantized. That kernel lives in `quantized_gradient.mojo` and not in
this file only because this file is on the wrong side of the import edge --
`quantized_gradient` imports `Histogram` from here, so the reverse import
would be a cycle.
already had.

The LightGBM cell, and the per-row derivative
---------------------------------------------
Two things about this file are now LightGBM's rather than this package's, and
both are about bytes rather than about arithmetic.

**A per-row gradient and hessian is Float32.** `include/LightGBM/meta.h`
defines `typedef float score_t`, and every derivative LightGBM carries -- from
the objective's output through `ordered_gradients` into the accumulate -- is
one. Raw scores, leaf values, gains and every histogram cell stay Float64 on
both sides. `score_t` in this module is the narrowing and
`boosting.fill_grad_hess` is where it is applied; the containers stay
`List[Float64]` because their type is fixed by signatures `tree.mojo` owns, so
what travels is a Float32 quantity in a Float64 word. The narrowing is
idempotent and every read site in this file applies it, which is what keeps
the gathered path and the direct path adding identical Float64s -- and
therefore keeps `compact_rows`, a policy decision, from moving a bin.

**That is the default and not the only setting.** `derivative_precision`
(`MOJOTREES_DERIVATIVE_PRECISION`) selects `float32`, everything above, or
`float64`, which keeps per-row derivatives at full Float64 through the
objective and every read site. It exists because the narrowing was a
*measured* trade rather than a free win -- four orders of magnitude of
CPU-versus-device agreement on `dense_regression`, and 9.4 percent of
average precision on `imbalanced_binary` -- and this project ships a
measured trade behind a switch. `DERIVATIVE_PRECISION_FLOAT32` and
`DERIVATIVE_PRECISION_FLOAT64` carry the numbers and the mechanism;
`docs/NUMERICS.md` section 1.6 carries the decision. The mechanism is a
compile-time `NARROW` parameter on every kernel that reads a derivative, so
the default arm's row loops are the instruction stream they were before the
switch existed, and the `float64` arm gives up the gathered pair buffer and
the row-blocked histograms because both are Float32 shapes.

**A histogram cell is an interleaved `(gradient, hessian)` pair, in the
private partials and in the output alike.** `include/LightGBM/bin.h` has
`typedef double hist_t`, `kHistEntrySize = 2 * sizeof(hist_t)`, `GET_GRAD(
hist, i) = hist[(i) << 1]` and `GET_HESS(hist, i) = hist[((i) << 1) + 1]`: 16
bytes a cell, no count plane. `Histogram._gh` is that array, and the
row-blocked kernel's partials are the same shape, so one row's visit to one
feature touches one contiguous cell rather than two planes a whole histogram
apart. `_accumulate_blocked_at` carries the partials' arithmetic and the two
divergences (a third slot for an exact count off the constant-hessian path,
and a stride of two rather than a `<< 1`, because the general arm needs
three).

The interleave used to stop at the fold: the partials were interleaved, then
scattered back into separate `grad` and `hess` planes on the way out. Both
ends now agree, and three consequences follow directly from that.

- The fold stopped being a scatter. On the general arm the private cell's
  first two floats ARE the output cell, so `fold_slots` copies 16 bytes
  rather than writing two scalars into two streams.
- The unblocked kernels' read-modify-write per (row, feature) touches **two**
  cache lines rather than three: the pair, and the count. That is the same
  argument the private cell already made, applied to the buffer the small and
  medium nodes accumulate into directly.
- The sibling subtraction, `Histogram.reset`, the zeroing passes and the
  distributed allreduce each lost one of their streams, and the allreduce
  lost a whole collective: gradients and hessians are one message now because
  they are one array.

**The count is not in the interleave, and that is a decision with evidence
rather than an omission.** `Histogram`'s docstring states it: different type,
different reduction, different access pattern, and a 24-byte cell straddles
cache lines where a 16-byte one never does. LightGBM's histogram entry is two
`hist_t` and nothing else.

The constant-hessian plane
--------------------------
A cell is a `(gradient, hessian)` pair beside a count, and for four of the
built-in objectives the hessian carries no information. `SQUARED_ERROR`,
`L1`, `HUBER` and `QUANTILE` all write the same hessian into every row when
the fit has no sample weights, and that value is exactly 1.0, so the hessian
slot is the count plane and storing both is waste.
`objective_has_constant_hessian` is the predicate, and it is
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
from std.os import getenv
from std.sys.info import simd_width_of
from std.sys.intrinsics import PrefetchOptions, prefetch
from std.time import perf_counter_ns
from .apple_cpu_policy import (
    ASSUMED_CACHE_LINE_BYTES,
    BIN_LAYOUT_AUTO,
    BIN_LAYOUT_FEATURE_MAJOR,
    BIN_LAYOUT_ROW_MAJOR,
    AccumulationPlan,
    PREFETCH_ROW_DISTANCE,
    ROW_BLOCK_CELL_FLOATS,
    ROW_BLOCK_CELL_FLOATS_CONST_H,
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


comptime SCATTER_PREFETCH = (
    PrefetchOptions().for_read().high_locality().to_data_cache()
)
"""The scatter loop's software prefetch, matching LightGBM's `PREFETCH_T0`.

`PREFETCH_T0` in `include/LightGBM/utils/common.h` expands to
`__builtin_prefetch(addr, 0, 3)` (or `_mm_prefetch(..., _MM_HINT_T0)`), which
is a read with maximum temporal locality into the data cache. These four
options say the same thing.
"""


@always_inline
def derivative[narrow: Bool](v: Float64) -> Float64:
    """One per-row derivative at the precision `derivative_precision` selected.

    `narrow` is a **compile-time** parameter, and that is the whole mechanism
    of the switch. At `True` this is `score_t` below and the emitted code is
    the `fcvt`-pair the accumulate kernels have always had; at `False` it is
    the identity and the compiler emits nothing at all. There is no runtime
    test in any row loop, so the default path's instruction stream is
    unchanged rather than merely-probably-unchanged, which is the property
    this lane is required to hold and cannot measure.

    The cost is paid in instantiations instead: every kernel that reads a
    derivative carries `NARROW` alongside its interleave width, so the
    accumulate ladder is compiled twice. See `DERIVATIVE_PRECISION_FLOAT32`
    for the whole argument and for what `float64` gives up.
    """
    comptime if narrow:
        return Float64(Float32(v))
    else:
        return v


@always_inline
def score_t(v: Float64) -> Float64:
    """One per-row derivative rounded to single precision, which is the
    precision LightGBM carries derivatives in.

    `derivative[True]`, under the name the rest of the package (and four
    tests) already import. Callers that have to honor `derivative_precision`
    call `derivative[NARROW]` instead; this name means the narrowing
    unconditionally and is what the `float32` default does everywhere.

    `include/LightGBM/meta.h` defines `typedef float score_t` unless
    `SCORE_T_USE_DOUBLE` is set, and every gradient and hessian in LightGBM
    -- from the objective's output through `ordered_gradients` to the
    histogram's accumulate -- is a `score_t`. Only the histogram cell, the
    leaf value and the gain are `double` (`typedef double hist_t`).

    This package computes derivatives in Float64 and then rounds here, so the
    value is a Float32 quantity living in a Float64 container. Two reasons it
    is a round rather than a retype:

    - The containers are `List[Float64]` in signatures `tree.mojo` owns, and
      this lane does not own that file. A Float64 holding a Float32-exact
      value is the same 24 bits of significand either way, so the *arithmetic*
      is LightGBM's now and the *container* narrows when that signature does.
    - It makes the narrowing idempotent, which is what lets the gathered pair
      buffer hold Float32 while the un-gathered path reads the array directly
      and still produce the identical Float64 sum. Without that, `compact_rows`
      -- a policy decision -- would change a bin, and no policy decision in
      this file is allowed to.

    `CONSTANT_HESSIAN` is 1.0, which is exactly representable in Float32, so
    the constant-hessian guarantee survives this untouched.
    """
    return derivative[True](v)


comptime DERIVATIVE_PRECISION_FLOAT32 = String("float32")
"""The default `derivative_precision`, and LightGBM's precision profile.

A per-row gradient and hessian is a Float32 quantity, narrowed at the
objective (`boosting.fill_grad_hess`) and re-narrowed at every read site
here, in `histogram_sparse.mojo` and in the row-major kernels. The
re-narrowing is not redundant: `goss.apply_goss_scaling` and a weighted fit
multiply a stored derivative into a value that is no longer
Float32-representable, so a read site that trusted the store would be
reading a Float64 the gather could not reproduce.

**What the default costs, measured, and why it is still the default.** The
first real-data run after the narrowing landed moved two things in opposite
directions. Agreement between this package's own CPU and accelerator arms
improved by four orders of magnitude on `dense_regression` (RMSE
differential 1.6e-04 to 7.9e-09). Accuracy on `imbalanced_binary` fell on
the CPU arm alone: average precision 0.0136 to 0.0123, a 9.4 percent
relative drop, and AUC 0.7339 to 0.7279. The decision recorded in
`docs/NUMERICS.md` is that Float32 stays the default because it is
LightGBM's own profile and it puts this package at LightGBM's numbers,
which is the parity the project is aiming at -- but a measured trade
becomes a switch, and this is the switch.

`DERIVATIVE_PRECISION_FLOAT64` restores full Float64 derivatives end to
end. It is opt-in, it moves bits by design, and it is slower: see there.
"""

comptime DERIVATIVE_PRECISION_FLOAT64 = String("float64")
"""`derivative_precision = "float64"`: per-row derivatives stay Float64.

The objective stores what it computed and every read site reads it, so no
derivative is rounded anywhere between the loss and the histogram cell.

**Three things it gives up, all of them stated rather than hidden.**

- **The PACKED gathered pair buffer, and by default the gather with it.**
  `_gather_pairs` packs a row's `(gradient, hessian)` into one Float64 word
  as two Float32 halves, which is exactly the thing this setting refuses.
  So the packed *word* is genuinely unavailable here. The GATHER is not,
  and treating the two as the same thing was a mistake worth naming,
  because it made this the most expensive line item in a CPU fit rather
  than a rounding error. The cost is not the one sequential pass forgone:
  with no gather, the un-gathered row loops read both derivative arrays
  through the row id once per (row, FEATURE GROUP), so at 100 active
  features and width 8 a node pays **13** random-access passes over both
  arrays, and every worker pays its own. `MOJOTREES_CPU_FLOAT64_GATHER`
  gathers into two unconverted Float64 words per row instead, which makes
  the other 12 sequential and cannot move a bit; it is off by default only
  until it is measured. See `float64_gather_arm`.
- **The row-blocked private histograms.** The blocked kernel's only row
  source is that pair buffer on the subset arm, so blocking is off under
  `float64` in *both* builders. Both, deliberately: while only one of them
  blocked, "growing on a bag is growing on the dataset of those rows" was
  false by four ulp on a leaf value (see `_accumulate_full`), and turning
  blocking off on one side only would reintroduce exactly that.
- **Speed, therefore.** This arm is a correctness and accuracy instrument,
  not a performance configuration, and a timing taken under it is not a
  timing of this package's CPU path.

What it does *not* give up: determinism across `MOJOTREES_NUM_WORKERS`,
which the unblocked kernels hold more simply than the blocked ones do
(every feature's summation stays inside one task at every task count, in
row order).
"""


def derivative_precision_narrows() -> Bool:
    """Whether per-row derivatives are narrowed to Float32, read from
    `MOJOTREES_DERIVATIVE_PRECISION`.

    `float64` selects Float64 derivatives; `float32`, unset, and anything
    else select the narrowing default. It does not raise, because two of its
    three callers sit in contexts that cannot
    (`boosting._fill_softmax_grad_hess` is reached from
    `boosting_rf._multiclass_rf_gradients`, which is not this lane's file to
    add a `raises` to). A typo is diagnosed by
    `check_derivative_precision` instead, which is called from every entry
    that can raise; see there for the one gap that leaves and the one word
    that closes it.

    Read **once per fit** at `ConstHessianSettings.resolve()` and once per
    round at `boosting.fill_grad_hess`. Never per node and never per row.
    """
    return getenv("MOJOTREES_DERIVATIVE_PRECISION") != (
        DERIVATIVE_PRECISION_FLOAT64
    )


def check_derivative_precision() raises:
    """Raise unless `MOJOTREES_DERIVATIVE_PRECISION` names a real setting.

    A value that is neither `float32` nor `float64` nor unset is a typo, and
    a typo that silently selected the default is precisely how an A/B runs
    one arm under the other's label -- which has happened in this repository
    once, and is why `const_hessian_allowed` above states the same rule.
    Refusing is the package's rule; falling back is not.

    Called from `ConstHessianSettings.resolve()` (once per fit) and from
    `boosting.fill_grad_hess` (once per round). **The one path it does not
    cover** is a multiclass or random-forest fit that resolves no snapshot
    and never reaches `fill_grad_hess`, because its only derivative site is
    `boosting._fill_softmax_grad_hess`, which cannot raise while
    `boosting_rf._multiclass_rf_gradients` does not declare `raises`. That
    is a one-word change in a file this lane does not own, and the report
    names it.
    """
    var s = getenv("MOJOTREES_DERIVATIVE_PRECISION")
    if (
        s.byte_length() == 0
        or s == DERIVATIVE_PRECISION_FLOAT32
        or s == DERIVATIVE_PRECISION_FLOAT64
    ):
        return
    raise Error(
        "MOJOTREES_DERIVATIVE_PRECISION must be '",
        DERIVATIVE_PRECISION_FLOAT32,
        "' or '",
        DERIVATIVE_PRECISION_FLOAT64,
        "', got '",
        s,
        "'",
    )


def check_device_derivative_precision(param_float64: Bool) raises:
    """Refuse Float64 derivatives on a backend that cannot carry them.

    **The device cannot, and the refusal is the point.** Gradients and
    hessians reach an accelerator as Float32 and nothing else:
    `histogram_gpu` states it as a hardware fact ("Apple GPUs have no
    Float64") and `gpu_gradient_stream.stage_gradients` performs the
    narrowing with a literal `Float32(g)` per row on upload. So a fit that
    asked for `float64` and was routed to the device would get the Float32
    answer, computed more slowly, with no error and nothing in the model to
    say which arm produced it.

    That is the same accepted-then-ignored failure the sparse and distributed
    growers had, and it is refused here for the same reason rather than
    worked around: this package says so or does it, and does not do neither.
    It is not fixable by threading, because the missing thing is a datatype
    the hardware does not have.

    **Both entries are refused**, the parameter through `param_float64` and
    the environment through the live read, because both are equally ignored
    once the derivatives are on the device. An environment variable exported
    for a CPU A/B and then left set for a GPU run is the likelier of the two
    and the harder to notice.

    Called from the two accelerator growers, beside the `ExtraTreeParams`
    validation they already run, so the refusal lands before the first
    histogram rather than part way through a fit.
    """
    check_derivative_precision()
    if derivative_precision_narrows() and not param_float64:
        return
    raise Error(
        "derivative_precision='",
        DERIVATIVE_PRECISION_FLOAT64,
        "' is not available on the accelerator: gradients and hessians are"
        " carried as Float32 on the device (there is no Float64 there), so"
        " the fit would silently produce the float32 answer. Train this"
        " configuration on the CPU path, which honors it end to end, or"
        " leave derivative_precision at '",
        DERIVATIVE_PRECISION_FLOAT32,
        "'. Note that MOJOTREES_DERIVATIVE_PRECISION reaches this too, so an"
        " environment variable left over from a CPU comparison is refused"
        " here rather than quietly ignored",
    )


@always_inline
def cnt_factor(n_data: Int, sum_hessian: Float64) -> Float64:
    """LightGBM's `cnt_factor`: rows per unit of hessian in one node.

    `src/treelearner/feature_histogram.cpp:186` and the four sites in
    `feature_histogram.hpp` all compute `const double cnt_factor = num_data /
    sum_hessian;` from the leaf's row count and the leaf's total hessian, and
    then recover a bin's row count from its hessian with `derived_count`
    below. That is why a LightGBM histogram cell has no count plane.

    Under the constant-hessian specialization `sum_hessian` is
    `CONSTANT_HESSIAN * n_data` and `CONSTANT_HESSIAN` is 1.0, so this is
    exactly 1.0 and `derived_count` is exact. See there.
    """
    if sum_hessian == 0.0:
        return 0.0
    return Float64(n_data) / sum_hessian


@always_inline
def derived_count(hess: Float64, factor: Float64) -> Int:
    """A bin's row count recovered from its hessian, rounded as LightGBM
    rounds.

    `Common::RoundInt` in `include/LightGBM/utils/common.h:911` is

        inline int RoundInt(double x) { return static_cast<int>(x + 0.5f); }

    -- add a half, then truncate toward zero. `0.5f` widens to the exactly
    representable double 0.5, so the `f` suffix changes nothing; it is
    reproduced here as written rather than as `round`, because `round` is
    round-half-away-from-zero and the two differ on every negative
    half-integer. A hessian sum is nonnegative on every objective that
    reaches this, so in practice they agree, and this still says what
    LightGBM says.

    **Exact under the constant-hessian specialization, and only there.** When
    every row's hessian is exactly `CONSTANT_HESSIAN = 1.0`, a bin holding
    `n` rows accumulated `1.0 + 1.0 + ... + 1.0`, whose every partial sum is
    one of the integers 1 .. n and is therefore exactly representable in
    Float64 below 2^53. So `hess` is exactly `Float64(n)`, `factor` is
    exactly 1.0, the product is exactly `Float64(n)`, and `Int(n + 0.5)`
    truncates back to `n`. No tolerance, no drift, at any n below 2^53.

    Off that path it is LightGBM's approximation and nothing better: a
    weighted or GOSS or logistic round has a per-row hessian, and a bin's
    count is not a function of its hessian sum. That is why the general arm
    of the blocked kernel keeps an exact count slot instead of calling this
    (see `apple_cpu_policy.ROW_BLOCK_CELL_FLOATS`).
    """
    return Int(hess * factor + 0.5)


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


@fieldwise_init
struct ConstHessianSettings(Copyable, Movable):
    """The two constant-hessian environment answers, read once.

    `parallel.DispatchSettings` does this for the seven variables the dispatch
    rule reads, and states the argument: a value that is the same for the
    length of a fit should be read once for the length of a fit. These two are
    the same shape of question and were left out of it. `_resolve_const_hessian`
    calls `const_hessian_allowed()`, and a build that resolves to on then calls
    `const_hessian_verify()`, so a squared-error fit -- where the declaration
    is on -- pays two `getenv` calls and the two `String` allocations that name
    them on **every** histogram build and **every** sibling subtraction, which
    is twice per node, ignoring the snapshot the same call already carries.

    Snapshot semantics are `DispatchSettings`'s, deliberately: `unresolved()`
    is the sentinel every defaulted parameter constructs, and it reads the
    environment live exactly as the call did before this type existed. There
    is no cache and no global, so nothing goes stale; a holder invalidates by
    resolving again.

    Where it should live: on `DispatchSettings` itself, resolved once per fit
    with the rest. It is a separate type only because `parallel.mojo` belongs
    to another owner this round, and a per-tree resolve was reachable without
    it. See the report for the field this would fold into.

    **It is now misnamed, and knowingly so.** `narrow` below is
    `derivative_precision`, which has nothing to do with the hessian plane.
    It is here because this is the snapshot that already reaches every
    histogram builder from every trainer that resolves one, and adding a
    second parameter to those signatures would have meant editing
    `tree.mojo`, which this lane does not own -- while adding a *field*
    changes only `resolve()`, which this file owns, and every existing
    threading site picks it up for free. The right name for the type is
    `HistogramSettings`, or it folds into `DispatchSettings` with the rest;
    both are renames across files this lane cannot touch, and the report
    names them.
    """

    var allowed: Bool
    """`const_hessian_allowed()`, i.e. `MOJOTREES_CONST_HESSIAN != 0`."""

    var verify: Bool
    """`const_hessian_verify()`, i.e. `MOJOTREES_CONST_HESSIAN_VERIFY != 0`."""

    var resolved: Bool
    """False for `unresolved()`, which sends the two const-hessian readers
    back to the live environment read. Those two fields are then not
    consulted at all. `narrow` is the exception and says why."""

    var narrow: Bool
    """`derivative_precision_narrows()`, i.e. `float32` rather than `float64`.

    **Read from the snapshot unconditionally, even under the sentinel**,
    which is the one place this type's semantics differ per field, so it is
    worth stating exactly why rather than leaving it to be discovered.

    The const-hessian fields read live under the sentinel because the read
    is skipped entirely on the common path: `_resolve_const_hessian` returns
    before touching the environment whenever the caller did not declare a
    constant hessian, which is every multiclass, DART, random-forest, sparse
    and distributed build. A derivative precision has no such early out --
    every build needs it -- so a live read under the sentinel would put one
    `getenv` and one `String` on **every histogram build** of exactly those
    trainers. That is the per-node environment read `ConstHessianSettings`
    was created to remove and the allocation class the alloc-churn lane
    removed 42,300 of; paying it back to support an opt-in diagnostic is not
    a trade this lane is willing to make.

    **Two entries write this field, and `widened` states which wins.** The
    environment writes it at `resolve()`; the `derivative_precision`
    parameter writes it at `widened()`, which every dense CPU grower applies
    from `params.extra` at `tree.grow_tree_leaves_profiled`. `float64` from
    either entry wins, which is why widening a sentinel is meaningful: an
    unwired grower still honors the parameter even though it honors the
    environment only through the per-tree `resolve()` fallback.

    So the sentinel means `float32`, the documented default, and
    `derivative_precision` is honored wherever a fit resolves a snapshot --
    which is every trainer that threads one, and any caller that hands one
    in. A direct builder call that passes the sentinel while the environment
    says `float64` is **not** a silently wrong answer: the objective stored
    an un-narrowed Float64, the read site narrows it, and
    `Float64(Float32(g))` at the read is bit for bit the value the store
    would have written. Such a build produces the `float32` histogram, which
    is what the sentinel says it produces.
    """

    @staticmethod
    def resolve() raises -> ConstHessianSettings:
        """One read of each of the three variables, once per fit, and the
        one place a mistyped `MOJOTREES_DERIVATIVE_PRECISION` is refused on
        the trainer paths that resolve a snapshot."""
        check_derivative_precision()
        return ConstHessianSettings(
            const_hessian_allowed(),
            const_hessian_verify(),
            True,
            derivative_precision_narrows(),
        )

    @staticmethod
    def resolve_with(param_float64: Bool) raises -> ConstHessianSettings:
        """`resolve()`, then the parameter entry folded in.

        `param_float64` is `ExtraTreeParams.wants_float64_derivatives()`, and
        the fold is `widened` below: `float64` wins from either entry.
        """
        return ConstHessianSettings.resolve().widened(param_float64)

    def widened(self, param_float64: Bool) -> ConstHessianSettings:
        """This snapshot with the `derivative_precision` **parameter** folded
        onto it, which is the hop that makes the parameter and the
        environment the same switch rather than two.

        **The precedence, decided explicitly: `float64` wins from either
        entry.** If the parameter asks for Float64 derivatives, or the
        environment does, the fit does not narrow. Neither entry can reduce
        the precision the other asked for.

        Three reasons it is this way round rather than
        parameter-beats-environment.

        - It is **monotone**, so it does not matter whether a trainer widens
          before or after it resolves, and a caller that widens twice gets
          the same snapshot. A precedence that could narrow would make the
          order of two reads observable in a fit's numbers.
        - It needs **no UNSET code**. `ExtraTreeParams.derivative_precision`
          defaults to `DERIV_PRECISION_FLOAT32` and therefore cannot tell
          "the caller chose float32" from "the caller said nothing"; a rule
          where the parameter overrides would have to tell them apart, which
          means a third code and a third arm at every existing reader of that
          field. `wants_float64_derivatives` states what that would cost.
        - It **cannot silently downgrade**. The documented environment entry
          (`MOJOTREES_DERIVATIVE_PRECISION=float64`, which is the entry
          `bench/results/cpu_float32_lambda0_2026-08-16` was taken through)
          keeps working under a parameter set to anything, and the direction
          it can only ever move is toward more precision. A switch whose two
          entries can cancel is not a switch anyone can A/B.

        The const-hessian fields are untouched: they are a different
        question, and `resolved` is not cleared, because widening does not
        make a resolved snapshot unresolved or an unresolved one resolved.
        Widening a **sentinel** is still meaningful and is the case that
        matters most: `narrow` is the one field every reader consults even
        under the sentinel (see its docstring), so a grower that was handed
        no snapshot still honors the parameter.
        """
        if not param_float64:
            return self.copy()
        var out = self.copy()
        out.narrow = False
        return out^

    @staticmethod
    def unresolved() -> ConstHessianSettings:
        """The sentinel: no `getenv`, four integer stores. `allowed` is set
        to the default the live read would return so that a misuse that
        ignored `resolved` would still fail safe, but no reader ignores it;
        `narrow` is set to the `float32` default and every reader *does*
        consult it, for the reason its docstring gives."""
        return ConstHessianSettings(True, False, False, True)


def _resolve_const_hessian(
    declared: Bool,
    env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) -> Bool:
    """The specialization's on/off decision for one build: the caller's
    declaration, ANDed with the environment's permission.

    With a resolved `env` the permission comes from the snapshot and this
    reads no environment variable; with the sentinel it reads it live, which
    is what every call site did before the parameter existed. The answer is
    the same either way for a fit that did not `setenv` mid-run, and a fit
    that did is exactly what a snapshot is defined not to observe."""
    if not declared:
        return False
    if env.resolved:
        return env.allowed
    return const_hessian_allowed()


def _resolve_const_hessian_verify(
    env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) -> Bool:
    """`const_hessian_verify()` under the same snapshot rule. Only ever
    asked when the specialization resolved to on, which is why it sits
    behind the `and` at every call site."""
    if env.resolved:
        return env.verify
    return const_hessian_verify()


@always_inline
def _resolve_narrow(
    env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) -> Bool:
    """`derivative_precision` for one build: one field read, no branch on
    `resolved`, no environment read. `ConstHessianSettings.narrow` carries
    the whole argument for why this one field is not a live read under the
    sentinel."""
    return env.narrow


@fieldwise_init
struct Histogram(Copyable, Movable):
    """Per-(feature, bin) statistics, flattened as `[f * n_bins + b]`.

    **Storage is LightGBM's `hist_t`: one interleaved `(gradient, hessian)`
    pair per cell, plus a separate count plane.** `include/LightGBM/bin.h`
    has `typedef double hist_t`, `kHistEntrySize = 2 * sizeof(hist_t)`,
    `GET_GRAD(hist, i) = hist[(i) << 1]` and `GET_HESS(hist, i) =
    hist[((i) << 1) + 1]`. `_gh` is exactly that array: `2 * n_features *
    n_bins` Float64, cell `i` at `[2 * i]` and `[2 * i + 1]`. The private
    accumulation cell in `_accumulate_blocked_at` was already laid out this
    way; before this lane the fold then *scattered* it back into two separate
    planes, so the interleave stopped at the task boundary and every consumer
    that wanted a bin's `(g, h)` paid two lines to get them. It no longer
    does.

    **The count is deliberately NOT in the interleave.** It is a different
    type, a different reduction and a different access pattern: the split
    scan SIMD-accumulates it as `SIMD[DType.int]` while the gradient pair is
    Float64, `distributed` moves it through `allreduce_sum_int` while the
    pair goes through `allreduce_sum_f64`, and under `const_h` it is not
    accumulated at all but derived. Folding it in would mean either an
    odd 24-byte cell that straddles cache lines (against a 16-byte pair, four
    to a line, never straddling) or storing it as Float64 and converting on
    every read. LightGBM reaches the same conclusion by construction: its
    histogram entry is two `hist_t` and nothing else.

    The planes are named with a leading underscore **on purpose**, and that
    is what made this change checkable rather than believed: the compiler
    enumerates every direct reader, exhaustively and mechanically, and a new
    one cannot appear without someone typing the underscore.

    Read a cell through `grad_at` / `hess_at` / `count_at` and write one
    through `set_grad_at` / `set_hess_at` / `set_count_at`; those take the
    CELL index and do the `<< 1` themselves, so every one of the several
    hundred accessor call sites is layout-agnostic and none of them moved.
    `n_cells` is the old `len(hist.grad)`.

    Construction goes through `from_planes`, which still takes three separate
    planes and interleaves the two float ones on the way in. Keeping that
    signature is what confines this change: the GPU download, the sparse
    builders, `distributed` and thirty test helpers all build their planes
    separately and hand them over unchanged. Callers already holding an
    interleaved buffer use `from_pairs` and pay no copy. The fieldwise
    constructor still exists and Mojo has no way to hide it, but it now takes
    `(gh, count, nf, nb)`, so the old four-plane spelling is a build error
    rather than a silent misread.
    """

    var _gh: List[Float64]
    var _count: List[Int]
    var n_features: Int
    var n_bins: Int

    @staticmethod
    def from_pairs(
        var gh: List[Float64],
        var count: List[Int],
        n_features: Int,
        n_bins: Int,
    ) -> Histogram:
        """Take ownership of an already-interleaved `(g, h)` array of
        `2 * n_features * n_bins` floats and a count plane. No copy, no
        repack: the named entry point for anything that builds in the storage
        layout."""
        return Histogram(gh^, count^, n_features, n_bins)

    @staticmethod
    def from_planes(
        var grad: List[Float64],
        var hess: List[Float64],
        var count: List[Int],
        n_features: Int,
        n_bins: Int,
    ) -> Histogram:
        """Take ownership of three separate planes and interleave the two
        float ones, in the order the fieldwise constructor used to take them.

        The interleave is one sequential pass over `2 * n_cells` floats,
        which is the same order of traffic the caller just spent building the
        planes, and it is why every producer that naturally emits separate
        planes (the GPU download, the sparse builders, the distributed
        reduce, the test fixtures) needed no change. A producer on a hot path
        should emit pairs and call `from_pairs` instead; none of the current
        ones is on a hot path."""
        var n = len(grad)
        var gh = List[Float64](capacity=2 * n)
        gh.resize(2 * n, 0.0)
        var p = gh.unsafe_ptr()
        var gsrc = grad.unsafe_ptr()
        var hsrc = hess.unsafe_ptr()
        for i in range(n):
            p.unsafe_store(2 * i, gsrc.unsafe_load(i))
            p.unsafe_store(2 * i + 1, hsrc.unsafe_load(i))
        return Histogram(gh^, count^, n_features, n_bins)

    @staticmethod
    def zeroed(n_features: Int, n_bins: Int) -> Histogram:
        """An all-zero histogram of the given shape, ready to accumulate
        into. Callers that build many histograms of one shape allocate once
        with this and recycle with `reset`.

        Two allocations where there were three, for every histogram the
        grower ever holds."""
        var size = n_features * n_bins
        return Histogram.from_pairs(
            _zeroed_f64(2 * size), _zeroed_int(size), n_features, n_bins,
        )

    # --- cell accessors ---------------------------------------------------
    #
    # Flat CELL index, `f * n_bins + b`, exactly as the fields used to be
    # indexed. The pair accessors shift it; the count accessor does not. No
    # bounds check beyond what `List.__getitem__` already did.

    @always_inline
    def grad_at(self, i: Int) -> Float64:
        return self._gh[i << 1]

    @always_inline
    def hess_at(self, i: Int) -> Float64:
        return self._gh[(i << 1) + 1]

    @always_inline
    def count_at(self, i: Int) -> Int:
        return self._count[i]

    @always_inline
    def set_grad_at(mut self, i: Int, v: Float64):
        self._gh[i << 1] = v

    @always_inline
    def set_hess_at(mut self, i: Int, v: Float64):
        self._gh[(i << 1) + 1] = v

    @always_inline
    def set_count_at(mut self, i: Int, v: Int):
        self._count[i] = v

    def n_cells(self) -> Int:
        """Number of cells, `n_features * n_bins`. Was `len(hist.grad)`."""
        return len(self._gh) >> 1

    # --- raw plane access -------------------------------------------------
    #
    # There is deliberately no accessor for a whole plane. The few callers
    # that must hand a `List` to something else (the accumulate helpers in
    # this module, the allreduce in `distributed`, the pointer taken for a hot
    # scan) name `hist._gh` / `hist._count` directly. The underscore is the
    # marker: `grep -rn '\._gh\b'` is the exact, complete list of what a
    # storage change has to rewrite by hand, and everything else is already
    # going through the cell accessors above.

    def reset(mut self):
        """Zero every bin in place, keeping the allocation. Cheaper than a
        fresh `zeroed` by exactly one malloc/free per buffer, which is what
        tree growth spends most of its allocator time on.

        Two streams where there were three, and the pair stream is one
        contiguous run of `2 * size` floats rather than two runs of `size` a
        whole histogram apart.

        Serial by construction, and the builders below no longer call it:
        they zero each feature's slice inside that feature's task instead. It
        stays because it is the right shape for a caller that needs a zeroed
        buffer without a build (`histogram_gpu`, `distributed`) and for any
        caller already running inside a parallel task, where a dispatching
        zero pass would nest one `sync_parallelize` inside another.
        """
        var size = self.n_features * self.n_bins
        var ghp = self._gh.unsafe_ptr()
        var cp = self._count.unsafe_ptr()
        comptime W = SIMD_LANES
        var i = 0
        while i + W <= size:
            ghp.unsafe_store(2 * i, SIMD[DType.float64, 2 * W](0.0))
            cp.unsafe_store(i, SIMD[DType.int, W](0))
            i += W
        while i < size:
            ghp.unsafe_store(2 * i, SIMD[DType.float64, 2](0.0))
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
    """Feature ids must be in range; an empty list means every feature.

    Range **only**. This does not check that the ids are distinct or
    ascending, and a repeated id passes here: `[0, 0, 0]` against three
    features is accepted by this function.

    **A repeated id is not harmless, and the sentence that used to stand here
    saying it was is what let the defect below live.** The old text claimed
    "nothing downstream is unsound for it, the accumulation would simply do
    feature 0 three times". Both halves were wrong. It *is* unsound, and the
    number of times feature 0 is accumulated is not the number of times it
    appears: the accumulation kernels dispatch over feature *groups* of
    `plan.group_width` lanes, they re-zero each lane's slice at the top of the
    group, and every lane of one group that names the same feature adds into
    the same cells. So a list of `k` copies of one id yields that feature's
    histogram multiplied by the group width, not by `k`, and at a width of 1
    it yields it once. The width is machine-dependent (it reads
    `dispatch_cores`), so the wrong answer was different on different
    machines. Two groups that name the same feature also land in different
    tasks, so once the dispatch goes parallel the shared slice takes
    unsynchronized read-modify-writes from two workers as well.

    So duplicates are *removed*, not merely tolerated: every entry point in
    this module runs `_has_duplicate_features` after this check and rebuilds
    its list through `_unique_features` when it fires. Nothing downstream ever
    sees a repeat. This function keeps range-only validation because it is
    also imported by `histogram_sparse.mojo`, and because the covering
    argument in `_zero_excluded` is a separate one: `len(features) ==
    n_features` still does not establish covering on its own for a caller
    that reaches that helper directly.
    """
    for i in range(len(features)):
        if features[i] < 0 or features[i] >= n_features:
            raise Error("feature index out of range")


def _has_duplicate_features(features: List[Int], n_features: Int) -> Bool:
    """Whether any feature id appears twice. Ids are assumed range-checked.

    Two passes, and the first one is the whole point of the shape. A strictly
    ascending list cannot repeat, that test is `len(features)` integer
    compares with no allocation, and **every id list this project generates
    internally is strictly ascending**: `sampling.sample_without_replacement`
    keeps its pool's order and draws no entry twice,
    `sampling.check_feature_pool` refuses a pool that is not ascending and
    unique, and the unsubsampled path passes `[0, n_features)` or the empty
    list. So the production path pays one scan and nothing else, and the
    `List[UInt8]` mask is only ever allocated for a list that is out of order,
    which is a hand-built one from outside this module.
    """
    var n = len(features)
    if n < 2 or n_features <= 0:
        return False
    var ascending = True
    for i in range(1, n):
        if features[i] <= features[i - 1]:
            ascending = False
            break
    if ascending:
        return False

    # Mojo's scalar pointer API cannot load `Bool`; a byte mask is what
    # `_zero_excluded` uses for the same job a few lines below.
    var seen = List[UInt8](capacity=n_features)
    seen.resize(n_features, UInt8(0))
    for i in range(n):
        var f = features[i]
        if f < 0 or f >= n_features:
            # Unreachable after `_check_features`. Skipped rather than
            # indexed so this helper is total on its own arguments.
            continue
        if seen[f] != UInt8(0):
            return True
        seen[f] = UInt8(1)
    return False


def _unique_features(features: List[Int], n_features: Int) -> List[Int]:
    """`features` with every repeat after the first occurrence dropped.

    **First-occurrence order, not sorted**, and that is a requirement rather
    than a convenience. A caller is allowed to hand in an unordered complete
    list and expect the same histogram an ordered one produces, which is what
    `tests/test_feature_sampling.mojo` asserts; sorting here would still
    satisfy that, but it would silently reorder the active-slot numbering that
    the blocked kernel's private partials are indexed by, and reordering is
    not this helper's job. Dropping is.

    Never returns an empty list for a non-empty input. That matters because an
    empty list means *every* feature to every builder in this module, so a
    dedupe that emptied its input would turn a subsampled build into a full
    one. It cannot happen: a non-empty `features` has been through
    `_check_features`, so every id is inside `[0, n_features)` and the first
    one is always appended.
    """
    var out = List[Int](capacity=len(features))
    if n_features <= 0:
        return out^
    var seen = List[UInt8](capacity=n_features)
    seen.resize(n_features, UInt8(0))
    for i in range(len(features)):
        var f = features[i]
        if f < 0 or f >= n_features:
            continue
        if seen[f] != UInt8(0):
            continue
        seen[f] = UInt8(1)
        out.append(f)
    return out^


def ensure_pair_capacity(mut pairs: List[Float64], n_words: Int):
    """Grow a gradient/hessian pair buffer to hold `n_words` Float64 words.

    **The caller decides how many words a row costs, because that now depends
    on the derivative precision.** The parameter was named `n_rows` while there
    was only one answer; it is a word count, and the two answers are:

    - Packed, the shipped Float32 default: **one** word per row, holding the
      row's `(gradient, hessian)` as two Float32 halves. This is LightGBM's
      `score_t` precision and half the bytes the pair used to cost; `score_t`
      above is the whole argument and `_gather_pairs` is where the packing
      happens. Callers pass `n_sub`.
    - Float64, behind `MOJOTREES_CPU_FLOAT64_GATHER`: **two** words per row,
      the derivatives unconverted, because a Float64 does not fit in half a
      word. Callers pass `2 * n_sub`.

    The buffer stays typed `List[Float64]` because its type is fixed by
    `tree.GrowScratch.pairs` and by three signatures `tree.mojo` owns, and a
    Float64 word is exactly two Float32 slots.

    Grows only. A grower that keeps one buffer across a whole tree allocates
    it once, at the size of the largest node it meets (the root), and every
    later node reuses that allocation: the buffer is written before it is
    read, so stale contents beyond the current node are never observed.
    """
    if len(pairs) < n_words:
        pairs.resize(n_words, 0.0)


def _zero_excluded(
    mut out_gh: List[Float64],
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

    # Nothing is excluded when the list already names every feature, and the
    # grower reaches here with exactly that list on the default path: at
    # `feature_fraction = 1.0` the tree's set is `[0, n_features)`, so the
    # mask below is allocated, filled, dispatched over, and zeroes nothing.
    #
    # `len(features) == n_features` is not on its own enough to conclude that:
    # `_check_features` checks range only, so `[0, 0, 0]` against three
    # features would pass a length test while excluding two features. The
    # builders in this module no longer reach here with such a list -- their
    # entry points strip repeats through `_unique_features` first -- but this
    # helper is called with the caller's list rather than with a promise about
    # it, so the shortcut proves its own precondition. Strictly ascending is
    # what settles it -- with every id range-checked by the caller,
    # `n_features` strictly ascending ids inside `[0, n_features)` can only be
    # `features[i] == i`.
    # A complete but unordered list falls through and does the work, which is
    # correct, just not shortened.
    if len(features) == n_features:
        var covers_every_feature = True
        for i in range(1, n_features):
            if features[i] <= features[i - 1]:
                covers_every_feature = False
                break
        if covers_every_feature:
            return

    # Mojo's scalar pointer API cannot load `Bool`. A byte preserves the
    # compact active mask and keeps the parallel zeroing pass unchanged.
    var active = List[UInt8](capacity=n_features)
    active.resize(n_features, UInt8(0))
    for i in range(len(features)):
        active[features[i]] = UInt8(1)

    var ghp = out_gh.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var active_p = active.unsafe_ptr()
    comptime W = SIMD_LANES

    def zero_range(f_start: Int, f_end: Int) {imm}:
        for f in range(f_start, f_end):
            if active_p.unsafe_load(f) == UInt8(0):
                var base = f * n_bins
                var b = 0
                while b + W <= n_bins:
                    # One pair store at double the width where there were two
                    # single-width stores into planes a whole histogram apart.
                    ghp.unsafe_store(
                        2 * (base + b), SIMD[DType.float64, 2 * W](0.0)
                    )
                    cp.unsafe_store(base + b, SIMD[DType.int, W](0))
                    b += W
                while b < n_bins:
                    ghp.unsafe_store(
                        2 * (base + b), SIMD[DType.float64, 2](0.0)
                    )
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
        out, data, grad, hess, features, const_hessian, settings,
        const_hessian_env,
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
    one task.

    `const_hessian_env` is the same idea for the two constant-hessian
    variables; see `ConstHessianSettings`."""
    var scratch = List[Float64]()
    build_histogram_into_scratch(
        out, scratch, data, grad, hess, features, const_hessian, settings,
        const_hessian_env,
    )


def build_histogram_into_scratch(
    mut out: Histogram,
    mut scratch: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) raises:
    """`build_histogram_into` with a caller-owned scratch buffer, the twin of
    `build_histogram_subset_into_scratch`.

    `scratch` holds this build's private row-block histograms and is grown,
    never shrunk, to `plan.block_scratch_floats()`. Its contents on entry are
    irrelevant and on exit unspecified, so it carries no state between calls
    beyond its capacity, and an unblocked plan never touches it at all.

    It exists because this builder now blocks (see `_accumulate_full`), and a
    grower that reaches it once per tree should not allocate the private
    histograms afresh each time. `build_histogram_into` passes a fresh empty
    list, which is exactly the allocate-per-call behaviour and is what every
    caller that has not been wired gets.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    _check_features(features, data.n_features)
    # A repeated feature id is accumulated once, which is the only answer that
    # is a histogram of anything. See `_check_features` for what the kernels do
    # with a repeat instead (multiply by the group width, and race), and see
    # `_unique_features` for why the fix is here rather than in the kernel.
    #
    # **Why strip rather than raise.** Refusing looks cheaper and is what the
    # first reading of this bug proposed, but the accepted contract of this
    # entry point is that a feature list names a *set* -- an unordered one, as
    # `tests/test_feature_sampling` asserts by building a reversed complete
    # list -- and the natural reading of a set given twice is the set. Raising
    # would also be a behaviour change for a caller that is asking for
    # something well defined, and it would convert today's silently wrong
    # number into a crash rather than into a right number. What makes the
    # choice easy is that neither branch costs anything on the production
    # path: no id list this project generates can repeat, and
    # `_has_duplicate_features` names the three generators, so the strip is a
    # scan that always says no.
    #
    # Re-entry rather than a local rebind so that the strip cannot drift out of
    # step with the dispatch below: the second pass takes exactly the path the
    # caller would have taken had it handed in the deduplicated list itself,
    # and terminates because `_unique_features` cannot return a list with a
    # repeat in it.
    if _has_duplicate_features(features, data.n_features):
        var unique = _unique_features(features, data.n_features)
        build_histogram_into_scratch(
            out, scratch, data, grad, hess, unique, const_hessian, settings,
            const_hessian_env,
        )
        return
    var const_h = _resolve_const_hessian(const_hessian, const_hessian_env)
    if const_h and _resolve_const_hessian_verify(const_hessian_env):
        _check_constant_hessian(hess, data.n_rows)

    # The two output buffers are passed as separate `mut` lists rather than
    # reached through `out`: a pointer taken from a struct field carries that
    # field's origin, which a worker closure cannot capture.
    #
    # The one runtime test the switch costs, taken once per build and never
    # inside a loop: it selects between two compile-time instantiations of
    # the whole accumulate, so neither arm's row loop contains a branch that
    # the other arm put there.
    if _resolve_narrow(const_hessian_env):
        _accumulate_full[True](
            out._gh, out._count, scratch, data, grad, hess,
            features, const_h, settings,
        )
    else:
        _accumulate_full[False](
            out._gh, out._count, scratch, data, grad, hess,
            features, const_h, settings,
        )


def _plan_accumulation(
    settings: DispatchSettings,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
    const_h: Bool = False,
) raises -> AccumulationPlan:
    """The one place a histogram build decides its shape.

    Both builders go through here so there is a single answer to "did this
    build read the environment". With a resolved snapshot it did not, and the
    `CpuProfile.detect()` that used to sit inside the argument list of every
    `derive_accumulation_plan` call is gone with it; with the sentinel it did,
    once, exactly as before.

    `const_h` reaches the width clamp and nothing else. **All three callers
    pass their real value**, which is the point: the parameter defaults to
    `False` for the reporting helpers and the tests that have no objective to
    hand, and a default that every call site took would leave the clamp
    exactly as wrong as it was. See `apple_cpu_policy._cache_group`.

    It cannot desynchronize the two builders. The width is a schedule; what
    the two layouts must agree on is the *block count*, and that is derived
    by `plan_row_block_count_at` from the row count, the bin count, the
    active feature count and the amortization ratio -- `const_h` reaches none
    of them, and `ROW_BLOCK_PLANES` stays at 3 unconditionally so the
    allocation the count is bounded by does not move either.
    """
    return derive_accumulation_plan_with(
        settings.policy,
        n_features,
        n_active,
        n_bins,
        n_rows_touched,
        rows_are_indirect,
        const_h,
    )


def _accumulate_full[
    NARROW: Bool
](
    mut out_gh: List[Float64],
    mut out_count: List[Int],
    mut scratch: List[Float64],
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
        settings, n_features, n_active, n_bins, n_rows, False, const_h
    )

    _zero_excluded(
        out_gh, out_count,
        n_features, n_bins, features, plan.excluded_ops, settings,
    )

    # Blocking is a `float32` shape. The subset builder's blocked kernel has
    # the packed Float32 pair buffer as its only row source, so it cannot run
    # under `float64`; and the two builders have to block *together* or not at
    # all, because the four-ulp defect the paragraph below records is exactly
    # what one-sided blocking produces. `DERIVATIVE_PRECISION_FLOAT64` states
    # this as one of the three things that arm gives up.
    var may_block = False
    comptime if NARROW:
        may_block = plan.blocked()

    if may_block:
        # The full-dataset builder blocks on the same plan the subset builder
        # blocks on, which is a correctness requirement and not a speed one.
        # While it did not, "growing on a bag is growing on the dataset of
        # those rows" was false: the bagged tree reached the subset builder
        # and folded row blocks, its reference reached this builder and summed
        # flat, and `test_bagged_tree_equals_tree_on_subset_dataset` measured
        # the disagreement at four ulp on a leaf value. Blocking both on one
        # plan makes them the same sequence of Float64 additions again. The
        # alternative -- have the subset builder skip blocking when its row
        # list happens to cover the whole matrix -- puts a discontinuity at
        # `len(bag) == n_rows`, which is worse.
        #
        # The scratch is local and per call, which is the objection the
        # row-block lane raised against blocking here. It stands, and it is
        # answered by frequency rather than by refutation: this builder is
        # reached once per tree (`tree._hist_full`, at the root) where the
        # subset builder is reached once per node, so one allocation per tree
        # is 61 allocations fewer per tree at the shape this round measures.
        # `build_histogram_into_scratch` exists so a grower can hand its own
        # buffer in and pay none; wiring `tree.GrowScratch.pairs` into
        # `_hist_full` is a one-line change in a file this lane does not own.
        var wanted = plan.block_scratch_floats()
        if len(scratch) < wanted:
            scratch.resize(wanted, 0.0)
        _accumulate_full_blocked(
            out_gh, out_count, scratch, data, grad, hess,
            features, plan, n_active, const_h, settings,
        )
        return

    # The ladder, dispatched from the host exactly as the GPU histogram family
    # is dispatched in `gpu_active_rows._enqueue_atomic_family`: one static
    # arm per instantiated width, `>=` rather than `==` so a width that is
    # somehow off the ladder still lands on a kernel that exists rather than
    # on none. The policy floors to a rung, so in practice each arm is hit
    # exactly.
    var group = plan.group_width
    if group >= 16:
        _accumulate_full_at[16, NARROW](
            out_gh, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    elif group >= 8:
        _accumulate_full_at[8, NARROW](
            out_gh, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    elif group >= 4:
        _accumulate_full_at[4, NARROW](
            out_gh, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    elif group >= 2:
        _accumulate_full_at[2, NARROW](
            out_gh, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )
    else:
        _accumulate_full_at[1, NARROW](
            out_gh, out_count, data, grad, hess, features,
            n_active, plan.group_count, plan.active_ops, const_h, settings,
        )


def _accumulate_full_blocked(
    mut out_gh: List[Float64],
    mut out_count: List[Int],
    mut scratch: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int],
    plan: AccumulationPlan,
    n_active: Int,
    const_h: Bool,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The interleave ladder over the blocked kernel, for the whole dataset.

    `INDIRECT=False`: rows are the loop counter, so the binned columns and
    the two derivative streams are read sequentially and there is no row-id
    list to load. `rows` is passed empty and never dereferenced.
    """
    var empty = List[Int]()
    var group = plan.group_width
    if group >= 16:
        _accumulate_blocked_at[16, False](
            out_gh, out_count, scratch, data, grad, hess,
            empty, 0, data.n_rows, features, plan, 0, n_active, const_h,
            settings,
        )
    elif group >= 8:
        _accumulate_blocked_at[8, False](
            out_gh, out_count, scratch, data, grad, hess,
            empty, 0, data.n_rows, features, plan, 0, n_active, const_h,
            settings,
        )
    elif group >= 4:
        _accumulate_blocked_at[4, False](
            out_gh, out_count, scratch, data, grad, hess,
            empty, 0, data.n_rows, features, plan, 0, n_active, const_h,
            settings,
        )
    elif group >= 2:
        _accumulate_blocked_at[2, False](
            out_gh, out_count, scratch, data, grad, hess,
            empty, 0, data.n_rows, features, plan, 0, n_active, const_h,
            settings,
        )
    else:
        _accumulate_blocked_at[1, False](
            out_gh, out_count, scratch, data, grad, hess,
            empty, 0, data.n_rows, features, plan, 0, n_active, const_h,
            settings,
        )


def _accumulate_full_at[
    GROUP: Int, NARROW: Bool
](
    mut out_gh: List[Float64],
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

    Two things have changed under that refutation without disturbing it. The
    pair buffer now holds two Float32 rather than two Float64, which halves
    what the *subset* builder streams per group and leaves this builder's
    arithmetic exactly where it was; and this builder now reads its
    derivatives through `score_t`, so the Float64 it adds is the Float64 the
    gathered path adds, which is what lets the two builders be compared bit
    for bit at all.

    That refutation is about the gradient INPUT and says nothing about the
    histogram OUTPUT. Interleaving the output cells is done -- in the private
    partials of `_accumulate_blocked_at`, which is where the scatter actually
    runs -- but the `Histogram` this builder *returns* still has three
    separate planes and a 64-bit `Int` count. Narrowing that is a different
    change: the layout is read by the GPU download path, by the distributed
    integer allreduce and by the C ABI, so it is its own change behind its own
    profile.

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
    var ghp = out_gh.unsafe_ptr()
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
                    # The `const_h` skip of the hessian plane is gone, and
                    # the branch with it: a hessian shares a 16-byte cell and
                    # therefore a cache line with its gradient, so the store
                    # that zeroes the gradient owns the line either way and
                    # the elided lanes were never a saved line -- only a
                    # saved 8 bytes inside one already being written. The
                    # refill below still overwrites them.
                    while zb + W <= n_bins:
                        ghp.unsafe_store(
                            2 * (z0 + zb), SIMD[DType.float64, 2 * W](0.0)
                        )
                        cp.unsafe_store(z0 + zb, SIMD[DType.int, W](0))
                        zb += W
                    while zb < n_bins:
                        ghp.unsafe_store(
                            2 * (z0 + zb), SIMD[DType.float64, 2](0.0)
                        )
                        cp.unsafe_store(z0 + zb, 0)
                        zb += 1

            if const_h:
                # Two planes per (row, feature), and the hessian is not read
                # at all: it is the same value on every row by the objective's
                # construction, so the accumulation carries no information the
                # count does not.
                for r in range(n_rows):
                    var g = derivative[NARROW](grad_p.unsafe_load(r))
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_p.unsafe_load(Int(col[k]) + r)
                            )
                            var p = b << 1
                            ghp.unsafe_store(p, ghp.unsafe_load(p) + g)
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                for r in range(n_rows):
                    # Contiguous: the whole dataset is one ascending walk, so
                    # the gradients, the hessians, and all `GROUP` binned
                    # columns are sequential streams.
                    var g = derivative[NARROW](grad_p.unsafe_load(r))
                    var h = derivative[NARROW](hess_p.unsafe_load(r))
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_p.unsafe_load(Int(col[k]) + r)
                            )
                            var p = b << 1
                            ghp.unsafe_store(
                                p,
                                ghp.unsafe_load[width=2](p)
                                + SIMD[DType.float64, 2](g, h),
                            )
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
                        # Writing only the odd lanes of an interleaved run
                        # means reading the even ones back to rejoin them.
                        # The read is of lines this task just wrote and that
                        # are therefore in its own L1, and `deinterleave` /
                        # `interleave` are lane permutations with no
                        # arithmetic, so the Float64 that lands is the Float64
                        # the plane store landed.
                        while fb + W <= n_bins:
                            var cur = ghp.unsafe_load[width = 2 * W](
                                2 * (f0 + fb)
                            )
                            # `rebind` only tells the elaborator that
                            # `(2 * W) / 2` is `W`; it is not a conversion
                            # and emits nothing.
                            var gvec = rebind[SIMD[DType.float64, W]](
                                cur.deinterleave()[0]
                            )
                            ghp.unsafe_store(
                                2 * (f0 + fb),
                                gvec.interleave(
                                    cp.unsafe_load[width=W](f0 + fb).cast[
                                        DType.float64
                                    ]()
                                ),
                            )
                            fb += W
                        while fb < n_bins:
                            ghp.unsafe_store(
                                2 * (f0 + fb) + 1,
                                Float64(cp.unsafe_load(f0 + fb)),
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
        settings, const_hessian_env,
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
        const_hessian, settings, const_hessian_env,
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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

    `const_hessian_env` is the same argument for the two constant-hessian
    variables, and this is likewise the call that matters most for it: the
    declaration is on for every squared-error round, so without a snapshot
    this function reads `MOJOTREES_CONST_HESSIAN` and
    `MOJOTREES_CONST_HESSIAN_VERIFY` once per node, allocating a `String` for
    each name, to re-derive a decision the fit made before its first tree. See
    `ConstHessianSettings`.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    _check_features(features, data.n_features)
    # Repeats stripped before anything reads `len(features)` as an active
    # count; `build_histogram_into_scratch` carries the argument for stripping
    # rather than refusing, and `_check_features` carries what the kernels do
    # with a repeat that survives.
    if _has_duplicate_features(features, data.n_features):
        var unique = _unique_features(features, data.n_features)
        build_histogram_subset_into_scratch(
            out, pairs, data, grad, hess, rows, row_start, row_count, unique,
            const_hessian, settings, const_hessian_env,
        )
        return
    var const_h = _resolve_const_hessian(const_hessian, const_hessian_env)
    if const_h and _resolve_const_hessian_verify(const_hessian_env):
        _check_constant_hessian(hess, data.n_rows)

    # One runtime test per build, selecting between two compile-time
    # instantiations; see `build_histogram_into_scratch` for why it is here
    # and not inside a row loop.
    if _resolve_narrow(const_hessian_env):
        _accumulate_subset[True](
            out._gh, out._count, pairs,
            data, grad, hess, rows, row_start, row_count, features, const_h,
            settings,
        )
    else:
        _accumulate_subset[False](
            out._gh, out._count, pairs,
            data, grad, hess, rows, row_start, row_count, features, const_h,
            settings,
        )


def _gather_pairs[
    NARROW: Bool = True
](
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

    **The pair is two Float32, packed into one Float64 word.** That is
    LightGBM's precision for a per-row derivative (`score_t` is `float`; see
    `score_t` above for why the narrowing is a round here rather than a
    retype) and it halves what the feature loop streams: the accumulation
    walks this buffer once per feature group, so at 50 features and width 2 a
    million-row node reads it 25 times, and 25 passes over 8 MB is 200 MB
    where 25 passes over 16 MB was 400 MB. Nothing else in the build reads
    this buffer.

    The narrowing is idempotent -- `Float32(Float64(Float32(x)))` is
    `Float32(x)` -- and every un-gathered read site applies the same
    `score_t`, which is what keeps `compact_rows`, a policy decision, from
    changing a bin.

    **`NARROW` selects the word, and the two arms cost different buffers.**
    This function used to have no parameter, on the argument that the packing
    *is* the narrowing: a Float64 derivative does not fit in half a Float64
    word, so there was no version of this that carried one. The first half of
    that is still true and the conclusion was too strong. It does not fit in
    half a word, so under `NARROW = False` it takes a **whole** word and the
    pair takes two, `[0, 2 * n_sub)` instead of `[0, n_sub)`. Nothing about the
    packed form changes.

    - `NARROW = True`, the shipped default: one Float64 word per row holding
      `(gradient, hessian)` as two Float32, as described above.
    - `NARROW = False`, behind `MOJOTREES_CPU_FLOAT64_GATHER`: two Float64
      words per row, the derivatives stored unconverted. See
      `float64_gather_arm` for why the gather is worth having on that arm at
      all, which is the 12 redundant random-access passes over the derivative
      arrays that `use_pairs = False` costs, not the one pass it saves.

    **Neither arm can move a bin.** The packed arm applies `score_t`, which is
    idempotent and which every un-gathered read site also applies. The Float64
    arm applies nothing, and `derivative[False]` is the identity, so a stored
    and reloaded Float64 is bit-equal to the direct read it replaces. That is
    what keeps `compact_rows`, a policy decision, from changing a bin on either
    arm.

    `_accumulate_subset_row_major` stays packed-only and passes `True`
    explicitly: it is not the default layout, it was measured slower, and
    widening it here would buy nothing.

    ONE pointer into `pairs`, with the Float32 view taken inside the closure
    from that same pointer. Two pointers carrying one origin into a parallel
    closure is an aliasing error Mojo refuses to compile, which
    `_accumulate_blocked_at` documents and works around the same way. Only one
    arm of the `comptime if` is emitted, so the compiled closure holds a single
    live view either way.
    """
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var pairs_p = pairs.unsafe_ptr()

    def fill_pairs(start: Int, end: Int) {imm}:
        var p32 = pairs_p.unsafe_bitcast[Float32]()
        for i in range(start, end):
            var r = rows_p.unsafe_load(i)
            comptime if NARROW:
                p32.unsafe_store(2 * i, Float32(grad_p.unsafe_load(r)))
                p32.unsafe_store(2 * i + 1, Float32(hess_p.unsafe_load(r)))
            else:
                pairs_p.unsafe_store(2 * i, grad_p.unsafe_load(r))
                pairs_p.unsafe_store(2 * i + 1, hess_p.unsafe_load(r))

    dispatch_rows_with(settings, fill_pairs, n_sub, gather_ops)


def _accumulate_subset[
    NARROW: Bool
](
    mut out_gh: List[Float64],
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
        settings, n_features, n_active, n_bins, n_sub, True, const_h
    )

    _zero_excluded(
        out_gh, out_count,
        n_features, n_bins, features, plan.excluded_ops, settings,
    )

    # The scratch layout, which is this function's to define: the gather
    # occupies `[0, n_sub)` when it runs -- one Float64 word per row, holding
    # the pair as two Float32 -- and the row-blocked private histograms occupy
    # `[part_off, part_off + plan.block_scratch_floats())` after it. `pairs`
    # is documented to carry no state between calls beyond its capacity, so
    # extending its tail costs a grower one growth per fit rather than an
    # allocation per node -- which is the whole reason the blocked path can
    # exist without a new parameter on a signature three other files call.
    #
    # A blocked node always gathers, even below `compact_min_rows`. The
    # blocked kernel has one row source rather than two, which halves what it
    # has to instantiate now that it also carries the row-indirection arm, and
    # a node large enough to block is by construction large enough for the
    # gather to be a rounding error against its own accumulation. It cannot
    # move a bin: the gather and the direct read deliver the same Float64,
    # because `score_t` is idempotent and both apply it.
    #
    # BLOCKING is a `float32` shape and does not run under `float64`: the
    # blocked kernel's only row source on this arm *is* the packed buffer, and
    # its two derivative reads go through a Float32 view unconditionally
    # (`_accumulate_blocked_at`'s `g32`, and the direct arm's `score_t` applied
    # with a comment saying the setting that would switch it cannot reach that
    # code). `DERIVATIVE_PRECISION_FLOAT64` states it, and `_accumulate_full`
    # states why blocking has to go off in both builders together rather than
    # in one. That is unchanged here and `blocked` stays False below.
    #
    # THE GATHER IS A DIFFERENT QUESTION AND IT WAS ANSWERED WRONG. It was off
    # under `float64` for the same reason blocking is, and the reason does not
    # transfer: blocking needs the packed word, the gather only needs *a*
    # contiguous buffer. Under `float64` the pair takes two Float64 words
    # instead of half of one, and everything else about the gather holds. What
    # leaving it off actually cost is not the one saved pass it looks like:
    # with `use_pairs` False, `_accumulate_subset_at`'s un-gathered row loops
    # read `grad_p.unsafe_load(r)` and `hess_p.unsafe_load(r)` through the row
    # id once per (row, FEATURE GROUP), so at 100 active features and width 8
    # that is 13 random-access passes over both derivative arrays per node,
    # repeated by every worker. Gathering once makes the other 12 sequential.
    #
    # Behind `MOJOTREES_CPU_FLOAT64_GATHER`, default off, and off means the
    # previous behavior exactly. `float64_gather_arm` carries the argument for
    # why this cannot move a bin and why the read is once per node rather than
    # in `DispatchSettings`. The read is inside the `else` arm of a comptime
    # branch, so the shipped Float32 path does not even pay the `getenv`.
    var use_pairs = False
    var blocked = False
    comptime if NARROW:
        use_pairs = plan.compact_rows or plan.blocked()
        blocked = plan.blocked()
    else:
        use_pairs = plan.compact_rows and float64_gather_arm()
    if use_pairs:
        # Words, not rows, and the two arms differ: one packed word per row
        # against two unconverted Float64 words. See `ensure_pair_capacity`.
        comptime if NARROW:
            ensure_pair_capacity(pairs, n_sub)
        else:
            ensure_pair_capacity(pairs, 2 * n_sub)
    # Where the row-blocked private histograms start, past whatever the gather
    # occupies. Inert on the `float64` arm because `blocked` is False there and
    # nothing else reads it, but kept correct rather than left at `n_sub` so a
    # future blocked Float64 path does not silently overlap the gather.
    var part_off = 0
    if use_pairs:
        comptime if NARROW:
            part_off = n_sub
        else:
            part_off = 2 * n_sub
    if blocked:
        var wanted = part_off + plan.block_scratch_floats()
        if len(pairs) < wanted:
            pairs.resize(wanted, 0.0)

    if use_pairs:
        _gather_pairs[NARROW](
            pairs, grad, hess, rows, row_start, n_sub, plan.gather_ops,
            settings,
        )

    if blocked:
        # The same ladder, over the blocked kernel. `plan.row_blocks` is a
        # value decision and not a scheduling one, so it is taken here and
        # never re-derived inside the kernel.
        var bgroup = plan.group_width
        if bgroup >= 16:
            _accumulate_blocked_at[16, True](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        elif bgroup >= 8:
            _accumulate_blocked_at[8, True](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        elif bgroup >= 4:
            _accumulate_blocked_at[4, True](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        elif bgroup >= 2:
            _accumulate_blocked_at[2, True](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        else:
            _accumulate_blocked_at[1, True](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, plan, part_off,
                n_active, const_h, settings,
            )
        return

    # The ladder, dispatched as in `_accumulate_full` above.
    var group = plan.group_width
    if group >= 16:
        _accumulate_subset_at[16, NARROW](
            out_gh, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            use_pairs, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    elif group >= 8:
        _accumulate_subset_at[8, NARROW](
            out_gh, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            use_pairs, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    elif group >= 4:
        _accumulate_subset_at[4, NARROW](
            out_gh, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            use_pairs, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    elif group >= 2:
        _accumulate_subset_at[2, NARROW](
            out_gh, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            use_pairs, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )
    else:
        _accumulate_subset_at[1, NARROW](
            out_gh, out_count, pairs, data, grad, hess,
            rows, row_start, row_count, features,
            use_pairs, n_active, plan.group_count, plan.active_ops,
            const_h, settings,
        )


def _accumulate_subset_at[
    GROUP: Int, NARROW: Bool
](
    mut out_gh: List[Float64],
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
    var ghp = out_gh.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    # ONE pointer into `pairs`, typed as the list is, with the Float32 view
    # taken inside the closure from this same pointer. That is
    # `_accumulate_blocked_at`'s shape and it is taken for its reason: two
    # pointers carrying one origin into a parallel closure is an aliasing error
    # Mojo refuses to compile. It also lets one closure serve both gather
    # layouts, since `NARROW` selects the view at compile time and only the
    # selected arm is emitted.
    var pp = pairs.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    comptime W = SIMD_LANES

    def accumulate_groups(g_start: Int, g_end: Int) {imm}:
        var g32 = pp.unsafe_bitcast[Float32]()
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
                    # The `const_h` skip of the hessian plane is gone, and
                    # the branch with it: a hessian shares a 16-byte cell and
                    # therefore a cache line with its gradient, so the store
                    # that zeroes the gradient owns the line either way and
                    # the elided lanes were never a saved line -- only a
                    # saved 8 bytes inside one already being written. The
                    # refill below still overwrites them.
                    while zb + W <= n_bins:
                        ghp.unsafe_store(
                            2 * (z0 + zb), SIMD[DType.float64, 2 * W](0.0)
                        )
                        cp.unsafe_store(z0 + zb, SIMD[DType.int, W](0))
                        zb += W
                    while zb < n_bins:
                        ghp.unsafe_store(
                            2 * (z0 + zb), SIMD[DType.float64, 2](0.0)
                        )
                        cp.unsafe_store(z0 + zb, 0)
                        zb += 1

            if const_h:
                if compact:
                    for i_row in range(n_sub):
                        var r = rows_p.unsafe_load(i_row)
                        # The gather still interleaves (g, h); only the
                        # gradient half is read here. The slot index is the
                        # same on both arms because both store the pair at
                        # `2 * i_row` and `2 * i_row + 1`; only the element
                        # type differs, so `NARROW` selects the view and not
                        # the arithmetic.
                        var g = 0.0
                        comptime if NARROW:
                            g = Float64(g32.unsafe_load(2 * i_row))
                        else:
                            g = pp.unsafe_load(2 * i_row)
                        comptime for k in range(GROUP):
                            if k < owned:
                                var b = Int(base[k]) + Int(
                                    bins_all_p.unsafe_load(Int(col[k]) + r)
                                )
                                var p = b << 1
                                ghp.unsafe_store(p, ghp.unsafe_load(p) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
                else:
                    for i_row in range(n_sub):
                        var r = rows_p.unsafe_load(i_row)
                        var g = derivative[NARROW](grad_p.unsafe_load(r))
                        comptime for k in range(GROUP):
                            if k < owned:
                                var b = Int(base[k]) + Int(
                                    bins_all_p.unsafe_load(Int(col[k]) + r)
                                )
                                var p = b << 1
                                ghp.unsafe_store(p, ghp.unsafe_load(p) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            elif compact:
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    # Adjacent, so one cache line carries both. One width-2
                    # load rather than two scalar loads, which is the shape
                    # `_accumulate_blocked_at`'s `STRIDE == 3` arm already
                    # uses on the packed view; it reads the same two values in
                    # the same order, so it cannot move a bin.
                    var gh = SIMD[DType.float64, 2](0.0)
                    comptime if NARROW:
                        var p32 = g32.unsafe_load[width=2](2 * i_row)
                        gh = SIMD[DType.float64, 2](
                            Float64(p32[0]), Float64(p32[1])
                        )
                    else:
                        gh = pp.unsafe_load[width=2](2 * i_row)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            var p = b << 1
                            ghp.unsafe_store(
                                p, ghp.unsafe_load[width=2](p) + gh
                            )
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                # Small node, or one active feature: the gather would cost a
                # pass and save nothing, so the gradients are read through the
                # row ids as they always were.
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    var g = derivative[NARROW](grad_p.unsafe_load(r))
                    var h = derivative[NARROW](hess_p.unsafe_load(r))
                    comptime for k in range(GROUP):
                        if k < owned:
                            var b = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            var p = b << 1
                            ghp.unsafe_store(
                                p,
                                ghp.unsafe_load[width=2](p)
                                + SIMD[DType.float64, 2](g, h),
                            )
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

            # The refill, exactly as in `_accumulate_full_at`: the elided
            # plane is written once per bin as `Float64(count)`, which is bit
            # for bit what the three-plane path accumulated.
            if const_h:
                comptime for k in range(GROUP):
                    if k < owned:
                        var f0 = Int(base[k])
                        var fb = 0
                        # Writing only the odd lanes of an interleaved run
                        # means reading the even ones back to rejoin them.
                        # The read is of lines this task just wrote and that
                        # are therefore in its own L1, and `deinterleave` /
                        # `interleave` are lane permutations with no
                        # arithmetic, so the Float64 that lands is the Float64
                        # the plane store landed.
                        while fb + W <= n_bins:
                            var cur = ghp.unsafe_load[width = 2 * W](
                                2 * (f0 + fb)
                            )
                            # `rebind` only tells the elaborator that
                            # `(2 * W) / 2` is `W`; it is not a conversion
                            # and emits nothing.
                            var gvec = rebind[SIMD[DType.float64, W]](
                                cur.deinterleave()[0]
                            )
                            ghp.unsafe_store(
                                2 * (f0 + fb),
                                gvec.interleave(
                                    cp.unsafe_load[width=W](f0 + fb).cast[
                                        DType.float64
                                    ]()
                                ),
                            )
                            fb += W
                        while fb < n_bins:
                            ghp.unsafe_store(
                                2 * (f0 + fb) + 1,
                                Float64(cp.unsafe_load(f0 + fb)),
                            )
                            fb += 1

    dispatch_feature_ranges_with(
        settings, accumulate_groups, n_groups, active_ops
    )


# ---------------------------------------------------------------------------
# The serial kernel arms
# ---------------------------------------------------------------------------
#
# `_accumulate_blocked_at`'s inner scatter is the loop the whole CPU fit is
# made of, and until this round nothing had been done to it beyond the
# prefetch. Four things were changed, and they are held as arms rather than
# folded into one commit because a bundle that is faster tells you that the
# bundle is faster and nothing else. The arms are **cumulative**, so each
# adjacent pair isolates one change:
#
# | arm      | cell stride | cell write   | inactive lane |
# |----------|-------------|--------------|---------------|
# | `base`   | runtime     | two scalars  | branch        |
# | `stride` | comptime    | two scalars  | branch        |
# | `packed` | comptime    | one 16-byte  | branch        |
# | `full`   | comptime    | one 16-byte  | none, when the group is full |
#
# **Every arm computes the same Float64 sums in the same order**, which is why
# they can be arms at all. The stride is the same number either way; the
# packed write is one SIMD add over two independent lanes where the scalar
# write was two adds of the same two lanes; and the branch removal only
# deletes a test whose answer is already known for the whole unit. The bench
# checks that rather than asserting it: `bench/bench_serial_kernel.mojo`
# prints a bitwise model digest per arm and refuses to call any of them equal
# without it.
#
# `full` is the default. The others exist so the decomposition can be re-taken.
#
# **Measured, 2026-08-16, 799,110 x 100, twelve repeats, arms interleaved in
# one process with LightGBM 4.7.0 as the in-window anchor.** The ratios below
# are medians of the *per-repeat paired* ratio, which is the form that is
# immune to the thermal ramp this box has: both arms of a pair are measured
# seconds apart under the same heat.
#
# | step             | 1 thread | 10 threads | verdict                     |
# |------------------|----------|------------|-----------------------------|
# | base -> stride   | 1.035x   | 1.030x     | consistent, not resolved    |
# | stride -> packed | 1.213x   | 1.120x     | **resolved**, 12/12 repeats |
# | packed -> full   | 1.000x   | 1.001x     | **null**                    |
# | base -> full     | 1.259x   | 1.152x     | resolved                    |
#
# Against LightGBM in the same process: at one thread we went from 1.331x
# behind to 1.050x behind, and in the run's hot regime 1.022x, which is a
# tie. At ten threads, 1.600x behind to 1.380x.
#
# **The one-write cell is the whole of it, and the branch removal is a null.**
# `packed -> full` deletes eight tests per row on twelve of thirteen groups
# and measured 1.000x at one thread and 1.001x at ten, with the sign flipping
# between the run's cool and hot regimes. It is kept as the default because it
# is the shape LightGBM's templates have and it costs nothing, but nobody
# should expect a number from it. The prediction that it would matter was
# wrong, and it is recorded here rather than quietly dropped.
comptime SERIAL_KERNEL_BASE = 0
comptime SERIAL_KERNEL_STRIDE = 1
comptime SERIAL_KERNEL_PACKED = 2
comptime SERIAL_KERNEL_FULL = 3


def serial_kernel_arm() -> Int:
    """Which scatter `_accumulate_blocked_at` runs, from
    `MOJOTREES_CPU_SERIAL_KERNEL`.

    Read once per accumulation -- that is, once per node histogram build, on
    the order of six thousand times in a hundred-tree fit -- and never inside
    the row loop. At roughly two hundred nanoseconds a read that is about two
    milliseconds against a fit measured in seconds, and it is the *same* two
    milliseconds on every arm including `base`, so it cannot tilt a
    comparison between them. It is deliberately not carried in
    `DispatchSettings`: that would add a field to `ResolvedCpuPolicy` and
    touch every construction of it, in files other lanes are in.

    An unrecognized value is the default rather than an error, because this
    variable is a measurement knob and a benchmark that dies three hours into
    a window on a typo is worse than one that runs the shipped kernel.

    **Where it reaches, and the two places it does not.** The only caller is
    `_accumulate_blocked_at`, and that kernel is the ROW-BLOCKED
    FEATURE-MAJOR one,
    reached from `_accumulate_full_blocked` and from `_accumulate_subset`'s
    blocked ladder. All three growth policies reach it on the same terms.
    It does not reach the unblocked ladder (`_accumulate_full_at`,
    `_accumulate_subset_at`), which is what a node below the amortization
    floor runs, so at 255 bins and the shipped 8/1 ratio a node under 8,160
    rows measures nothing here. It does not reach either row-major kernel
    either. An arm swept over a shape whose nodes are mostly small, or swept
    with `MOJOTREES_CPU_BIN_LAYOUT=row`, is measuring the shipped kernel on
    both sides of the comparison.
    """
    var s = getenv("MOJOTREES_CPU_SERIAL_KERNEL")
    if s == "base":
        return SERIAL_KERNEL_BASE
    if s == "stride":
        return SERIAL_KERNEL_STRIDE
    if s == "packed":
        return SERIAL_KERNEL_PACKED
    return SERIAL_KERNEL_FULL


def float64_gather_arm() -> Bool:
    """Whether the Float64 derivative path gathers its pairs once per node,
    from `MOJOTREES_CPU_FLOAT64_GATHER`. Off unless set to `1`.

    **What it fixes.** Under `derivative_precision = "float64"` the gather is
    off, because the gather's word is two Float32 and a Float64 derivative does
    not fit in half of one (`_gather_pairs`). Losing it is recorded as one of
    the costs of that setting. But the cost is larger than "one pass saved":
    with `use_pairs` False, `_accumulate_subset_at`'s un-gathered row loop
    reads `grad_p.unsafe_load(r)` and `hess_p.unsafe_load(r)` through the row
    id **once per (row, feature group)**. At 100 active features and group
    width 8 that is 13 random-access passes over both derivative arrays per
    node, and every worker makes its own. Gathering once into a contiguous
    two-Float64-per-row buffer makes the other 12 passes sequential.

    **It cannot move a bin, which is what separates it from the row-blocking
    knobs beside it.** `env_row_block_amortize` moves bits and says so, because
    a block count is a summation order. This changes no order and no value: the
    same rows are visited in the same sequence, the same groups accumulate in
    the same sequence, and a Float64 stored and loaded back is the same Float64
    (`derivative[False]` is the identity, so the gathered and un-gathered reads
    deliver bit-equal values). It is therefore a pure performance A/B and needs
    no golden regeneration, and the default is off only so that nothing
    published moves before it is measured.

    Read once per accumulation, which is once per node histogram build, and
    never inside a row loop. The cost argument and the reason this is not
    carried in `DispatchSettings` are `serial_kernel_arm`'s directly above,
    unchanged and for the same reasons: a field there would touch every
    construction of `ResolvedCpuPolicy`, in files other lanes are in. Under
    Float32 this function is not called at all, because that arm's gather
    decision is a comptime `NARROW` branch that never reaches here, so the
    shipped default pays not even the read.

    An unrecognized value is off rather than an error, matching
    `serial_kernel_arm`: this is a measurement knob and a window that dies on a
    typo is worse than one that runs the shipped path.

    **Where it reaches, stated so that a null is not read as a result.** One
    call site, `_accumulate_subset`'s `else` arm, which is the FEATURE-MAJOR
    subset builder under `float64`. It is the same on all three growth
    policies, because every grower reaches that builder through
    `tree._hist_subset`. It does not reach two other builds and cannot.

    - **The row-major subset builder.** `_accumulate_subset_row_major` has no
      `else` arm at all, so under `float64` its `use_pairs` is unconditionally
      False. Setting this variable together with
      `MOJOTREES_CPU_BIN_LAYOUT=row`, or with
      the per-node layout rule (default ON) on a node that does not block, changes
      nothing about that node's build. Nothing is wrong with those histograms;
      the switch is simply inert there.
    - **The whole-dataset builder.** `_accumulate_full` never gathers on
      either precision, because it reads the derivative arrays sequentially
      and has no row ids to gather through, so the root of an unbagged tree is
      outside this switch on every policy.
    """
    return getenv("MOJOTREES_CPU_FLOAT64_GATHER") == "1"


def serial_kernel_arm_name(arm: Int) -> String:
    if arm == SERIAL_KERNEL_BASE:
        return String("base")
    if arm == SERIAL_KERNEL_STRIDE:
        return String("stride")
    if arm == SERIAL_KERNEL_PACKED:
        return String("packed")
    return String("full")


def _accumulate_blocked_at[
    GROUP: Int, INDIRECT: Bool
](
    mut out_gh: List[Float64],
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
    """The row-blocked accumulation at one interleave width, for both
    builders.

    The node's rows are cut into `plan.row_blocks` contiguous ascending
    blocks. Each block owns a private histogram over the active features only,
    accumulates its own rows into it, and the partials are then folded into
    the caller's output in ascending block order.

    `INDIRECT` selects the row source and nothing else. True is the subset
    builder: a row is `rows[row_start + i]` and its derivatives come from the
    gathered pair buffer. False is the whole-dataset builder: a row *is* the
    loop counter, `rows` is empty and never dereferenced, and the derivatives
    are read from the arrays directly. Both arms feed the accumulate the same
    Float64s, because `score_t` is idempotent and the gather applies it too;
    that identity is what lets the two builders be compared bit for bit.

    **Both builders block, on one plan, and that is a correctness property.**
    Until they did, `test_bagged_tree_equals_tree_on_subset_dataset` was
    false: the bagged tree reached the subset builder and folded row blocks
    while its whole-dataset reference summed flat, and the two leaf values
    differed by four ulp. The block count, the boundaries and the fold order
    come from `apple_cpu_policy.plan_row_block_count`, which reads the row
    count, the bin count, the active feature count and an explicit request
    and nothing else -- so the two builders agree, and both are invariant
    under `MOJOTREES_NUM_WORKERS` and under any task count.

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

    **The private cell is LightGBM's: interleaved, and one cache line.**
    `include/LightGBM/bin.h` defines `typedef double hist_t`,
    `kHistEntrySize = 2 * sizeof(hist_t)`, `GET_GRAD(hist, i) = hist[(i) << 1]`
    and `GET_HESS(hist, i) = hist[((i) << 1) + 1]`: a bin's gradient and
    hessian are adjacent Float64 and there is no count plane. The partials
    here are laid out the same way, `stride` floats per cell, so one row's
    visit to one feature reads and writes one contiguous 16- or 24-byte cell
    instead of touching separate planes a `n_blocks * block_cells` stride
    apart. Three scattered lines per (row, feature) become one. The zeroing
    pass and the fold both become contiguous streams for the same reason.

    `stride` is 2 under `const_h` and 3 otherwise, and the difference is what
    the second slot means:

    - Under `const_h` the cell is exactly LightGBM's 16 bytes and the second
      slot is a **count**, as it is in LightGBM: `DenseBin::
      ConstructHistogramInner`'s `!USE_HESSIAN` arm aliases the hessian slot
      as `hist_cnt_t*` and does `++cnt[ti]`, and `Dataset::
      ConstructHistogramsInner` then rewrites it as `data_ptr[i + 1] =
      static_cast<double>(cnt_dst[i]) * hessians[0]`. This file does the same
      thing in Float64 rather than through a union: `CONSTANT_HESSIAN` is 1.0,
      the sum of `n` copies of 1.0 is exactly `Float64(n)` because every
      partial is an exactly representable integer below 2^53, and the refill
      is that same Float64. So the count that comes out is **exact, not
      estimated** -- `derived_count` is called on it to state the rule, and
      the rule is exact at a factor of 1.0.
    - Otherwise the cell keeps a third slot for an exact count where LightGBM
      would derive one from the hessian sum. That divergence is argued at
      `derived_count`: a derived count is exact only when every hessian is
      `CONSTANT_HESSIAN`, and `Histogram.count_at` is read by a distributed
      integer allreduce, by the C ABI, and by the split scan's
      `min_data_in_leaf` test, none of which should take a rounded number
      when an exact one costs eight bytes of scratch that is already
      allocated. LightGBM's authoritative leaf count comes from the data
      partition (`leaf_count`), not from its histogram, and the same is true
      here: `split.find_best_split`'s `n_rows` argument is the partition's
      count and is preferred over the histogram's total wherever the caller
      supplies one.

    **The allocation is unconditional at three slots even when the stride is
    two.** `apple_cpu_policy.ROW_BLOCK_PLANES` stays at 3 for the byte budget
    so that the *block count* cannot depend on `const_h`; a fit that turned
    the specialization off would otherwise fold a different number of
    partials and move a bin. Only the addressing narrows.

    **The prefetch.** `DenseBin::ConstructHistogramInner`
    (`src/io/dense_bin.hpp:109-130`) runs its loop in two parts: while more
    than `pf_offset = 64 / sizeof(VAL_T)` rows remain it issues
    `PREFETCH_T0(data_.data() + data_indices[i + pf_offset])` before each
    accumulate, then finishes without one. `SCATTER_PREFETCH` is that hint
    and `PREFETCH_ROW_DISTANCE` is that distance. It is issued only on the
    `INDIRECT` arm, because it is the row-id indirection that stalls: on the
    sequential arm the binned column is a unit-stride stream and the hardware
    prefetcher already owns it.

    None of the prefetch moves a bit. It is a hint with no architectural
    effect, and the two loop halves perform the same additions in the same
    order.

    **"LightGBM's loop is scalar and this one is too" was half a reading of
    one of LightGBM's two kernels, and it is withdrawn.** LightGBM has two
    dense construction kernels and they have different shapes.

    - `DenseBin::ConstructHistogramInner` is feature-major, one feature per
      call, and there its row loop really is scalar with no unroll: the only
      thing between `i` and `i + 1` is the prefetch. That is the kernel the
      old sentence described, and about that kernel it was right.
    - `MultiValDenseBin::ConstructHistogramInner`
      (`src/io/multi_val_dense_bin.hpp:59-103`) is row-major, and its body is
      `for (int j = 0; j < num_feature_; ++j) { bin = data_ptr[j]; ti = (bin +
      offsets_[j]) << 1; grad[ti] += gradient; hess[ti] += hessian; }`. One
      row's derivatives are loaded once into registers and then scattered
      across **every** feature. That is an unroll over features in all but
      name, and it is the shape this kernel's `comptime for k in range(GROUP)`
      already has -- so the part of the sentence that mattered was already
      false about our own code when it was written.

    Two further things LightGBM does that the sentence obscured, both now
    taken here:

    - **The 16-byte cell is written as one word, not two.** The Int
      histogram kernels (`ConstructHistogramIntInner`, `dense_bin.hpp:175-221`
      and `multi_val_dense_bin.hpp:128-174`) pack gradient and hessian into a
      single `PACKED_HIST_T` (`int64_t`, `int32_t` or `int16_t`) and do
      `out_ptr[ti] += gradient_packed`: **one** add per cell. Those kernels
      are reached only under `use_quantized_grad`, which is off by default, so
      LightGBM's default fit does take the two-add Float64 path -- but the
      one-write cell is LightGBM's own idea and not an invention here. The
      `PACKED` arm is the Float64 spelling of it: one 16-byte load, one
      two-lane add, one 16-byte store, which is the same two additions of the
      same two Float64 and therefore the same bits.
    - **There is no runtime branch inside the loop at all.** `USE_INDICES`,
      `USE_PREFETCH`, `ORDERED`, `USE_HESSIAN`, `IS_4BIT` and `HIST_BITS` are
      every one of them template parameters, resolved before the loop exists.
      This kernel had `if k < owned` inside its unrolled body, once per
      feature per row, testing a value that is fixed for the whole dispatch
      unit. At 100 features and width 8 the answer is "yes" for twelve of the
      thirteen groups. `FULL` is the specialization that deletes the test on
      those twelve, and it deletes nothing else.

    Both are arms; see `serial_kernel_arm`.

    A tail group owning fewer than `GROUP` features and the SIMD-lane slot
    arrays are as in `_accumulate_subset_at`.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_sub = row_count
    var n_blocks = plan.row_blocks
    var n_groups = plan.group_count
    var block_rows = plan.block_rows
    var block_cells = plan.block_cells
    var use_all = len(features) == 0
    var ghp = out_gh.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    # ONE pointer into `pairs`, not two. The gather occupies `[0, n_sub)` and
    # the private histograms `[part_off, ...)`, and two pointers carrying the
    # same origin into one parallel closure is an aliasing error Mojo refuses
    # to compile. So `part_off` is folded into every private index instead,
    # and the Float32 view of the gather is taken inside the closure from this
    # same pointer.
    var pp = pairs.unsafe_ptr()
    # Floats per private cell, and where the count lives inside one.
    var stride = (
        ROW_BLOCK_CELL_FLOATS_CONST_H if const_h else ROW_BLOCK_CELL_FLOATS
    )
    var count_slot = 1 if const_h else 2
    # Floats reserved per block. Always `ROW_BLOCK_CELL_FLOATS` per cell, so
    # the block count is a value independent of `const_h`; the stride above
    # only says how many of them are addressed.
    var region = block_cells * ROW_BLOCK_CELL_FLOATS
    comptime W = SIMD_LANES
    # Once per node histogram build, never inside the row loop. See
    # `serial_kernel_arm`.
    var arm = serial_kernel_arm()

    def accumulate_units(u_start: Int, u_end: Int) {imm}:
        # The Float32 view of the gather is taken inside each nested scatter
        # from `pp` rather than once out here, for the reason `pp`'s own
        # comment above gives: two pointers carrying the same origin into one
        # closure is an aliasing error Mojo refuses to compile, and a nested
        # parametric scatter captures both. The bitcast emits nothing.

        # One dispatch unit's rows.
        #
        # Flat rather than a row-visit called from a row loop, and that is a
        # language constraint rather than a preference: a nested parametric
        # closure that captures another nested closure is two mutable
        # captures of one origin and Mojo refuses to compile it. The k-loop
        # is therefore written out in each of the three row loops.
        #
        # The parameters:
        #
        # - `STRIDE` is the private cell's float count, 2 under a constant
        #   hessian and 3 otherwise, and it is a *parameter* here where the
        #   `base` arm keeps it a variable. The multiply that turns a bin id
        #   into a cell offset is on every feature of every row, and a shift
        #   is not an integer multiply.
        # - `PACKED` writes the cell as one 16-byte word instead of two
        #   scalar read-modify-writes. Two independent SIMD lanes, so the two
        #   Float64 that land are the two the scalar spelling landed.
        # - `FULL` says every lane of the unrolled group is active, which
        #   deletes the per-lane test. It is chosen per dispatch unit.
        # - `G` is `GROUP`, passed explicitly because a nested definition
        #   cannot take the enclosing function's parameter as a SIMD width.
        #
        # `pair` is the row's contribution and it is where the two objectives
        # merge. Under `STRIDE == 2` the second slot counts rows, as it does
        # in LightGBM's `!USE_HESSIAN` arm, so the pair is `(g, 1.0)`;
        # otherwise it is `(g, h)` and the count is the third slot. The
        # `base` arm wrote the same two Float64 with two stores.
        #
        # The alignment on the packed load is stated and not defaulted.
        # `part_off` is the node's row count, which is odd about half the
        # time, so a private cell is 8-byte aligned and not 16-byte aligned;
        # `alignment=8` is what the address actually has. An overstated
        # alignment here would be undefined behavior, and an understated one
        # invites a backend to split the vector back into the two scalar
        # loads it was written to replace.
        #
        # Two halves on the indirect arm exactly as LightGBM has them: the
        # first issues the prefetch and the second is the tail that has
        # nothing left to prefetch. Same additions, same order, same halves
        # as the `base` arm below.
        def scatter[STRIDE: Int, FULL: Bool, PACKED: Bool, G: Int](
            base: SIMD[DType.int, G],
            col: SIMD[DType.int, G],
            owned_n: Int,
            r0: Int,
            r1: Int,
        ) {imm}:
            var g32 = pp.unsafe_bitcast[Float32]()
            var i_row = r0
            comptime if INDIRECT:
                var pf_end = r1 - PREFETCH_ROW_DISTANCE
                while i_row < pf_end:
                    var r = rows_p.unsafe_load(i_row)
                    var pair = SIMD[DType.float64, 2](
                        Float64(g32.unsafe_load(2 * i_row)), 1.0
                    )
                    comptime if STRIDE == 3:
                        var gh = g32.unsafe_load[width=2](2 * i_row)
                        pair = SIMD[DType.float64, 2](
                            Float64(gh[0]), Float64(gh[1])
                        )
                    var rf = rows_p.unsafe_load(i_row + PREFETCH_ROW_DISTANCE)
                    comptime for k in range(G):
                        if FULL or k < owned_n:
                            prefetch[SCATTER_PREFETCH](
                                bins_all_p.unsafe_offset(Int(col[k]) + rf)
                            )
                            var b = Int(base[k]) + STRIDE * Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            comptime if PACKED:
                                pp.unsafe_store(
                                    b,
                                    pp.unsafe_load[width=2, alignment=8](b)
                                    + pair,
                                )
                            else:
                                pp.unsafe_store(
                                    b, pp.unsafe_load(b) + pair[0]
                                )
                                pp.unsafe_store(
                                    b + 1, pp.unsafe_load(b + 1) + pair[1]
                                )
                            comptime if STRIDE == 3:
                                pp.unsafe_store(
                                    b + 2, pp.unsafe_load(b + 2) + 1.0
                                )
                    i_row += 1
                while i_row < r1:
                    var r = rows_p.unsafe_load(i_row)
                    var pair = SIMD[DType.float64, 2](
                        Float64(g32.unsafe_load(2 * i_row)), 1.0
                    )
                    comptime if STRIDE == 3:
                        var gh = g32.unsafe_load[width=2](2 * i_row)
                        pair = SIMD[DType.float64, 2](
                            Float64(gh[0]), Float64(gh[1])
                        )
                    comptime for k in range(G):
                        if FULL or k < owned_n:
                            var b = Int(base[k]) + STRIDE * Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            comptime if PACKED:
                                pp.unsafe_store(
                                    b,
                                    pp.unsafe_load[width=2, alignment=8](b)
                                    + pair,
                                )
                            else:
                                pp.unsafe_store(
                                    b, pp.unsafe_load(b) + pair[0]
                                )
                                pp.unsafe_store(
                                    b + 1, pp.unsafe_load(b + 1) + pair[1]
                                )
                            comptime if STRIDE == 3:
                                pp.unsafe_store(
                                    b + 2, pp.unsafe_load(b + 2) + 1.0
                                )
                    i_row += 1
            else:
                # `score_t` and not `derivative[NARROW]`, on purpose: this
                # kernel carries no `NARROW` parameter because it runs only
                # under `float32`. Blocking is off under `float64`
                # (`_accumulate_full` and `_accumulate_subset` both say why),
                # so the narrowing here is unconditional because the setting
                # that would switch it cannot reach this code.
                while i_row < r1:
                    var pair = SIMD[DType.float64, 2](
                        score_t(grad_p.unsafe_load(i_row)), 1.0
                    )
                    comptime if STRIDE == 3:
                        pair[1] = score_t(hess_p.unsafe_load(i_row))
                    comptime for k in range(G):
                        if FULL or k < owned_n:
                            var b = Int(base[k]) + STRIDE * Int(
                                bins_all_p.unsafe_load(Int(col[k]) + i_row)
                            )
                            comptime if PACKED:
                                pp.unsafe_store(
                                    b,
                                    pp.unsafe_load[width=2, alignment=8](b)
                                    + pair,
                                )
                            else:
                                pp.unsafe_store(
                                    b, pp.unsafe_load(b) + pair[0]
                                )
                                pp.unsafe_store(
                                    b + 1, pp.unsafe_load(b + 1) + pair[1]
                                )
                            comptime if STRIDE == 3:
                                pp.unsafe_store(
                                    b + 2, pp.unsafe_load(b + 2) + 1.0
                                )
                    i_row += 1


        # The `base` arm: the scatter exactly as it shipped, kept whole so
        # that the decomposition can be re-taken rather than trusted. Runtime
        # stride, two scalar stores per cell, one test per lane per row.
        # `CH` is a parameter only because the shipped kernel already hoisted
        # `const_h` out of the row loop; keeping it runtime here would price
        # the baseline for a branch it never had.
        def scatter_base[CH: Bool, G: Int](
            base: SIMD[DType.int, G],
            col: SIMD[DType.int, G],
            owned_n: Int,
            r0: Int,
            r1: Int,
            stride_rt: Int,
        ) {imm}:
            var g32 = pp.unsafe_bitcast[Float32]()
            var pf_end = r1 - PREFETCH_ROW_DISTANCE
            var i_row = r0
            comptime if INDIRECT:
                while i_row < pf_end:
                    var r = rows_p.unsafe_load(i_row)
                    var g: Float64
                    var h = 1.0
                    comptime if CH:
                        g = Float64(g32.unsafe_load(2 * i_row))
                    else:
                        var gh = g32.unsafe_load[width=2](2 * i_row)
                        g = Float64(gh[0])
                        h = Float64(gh[1])
                    var rf = rows_p.unsafe_load(i_row + PREFETCH_ROW_DISTANCE)
                    comptime for k in range(G):
                        if k < owned_n:
                            prefetch[SCATTER_PREFETCH](
                                bins_all_p.unsafe_offset(Int(col[k]) + rf)
                            )
                            var b = Int(base[k]) + stride_rt * Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + h)
                            comptime if not CH:
                                pp.unsafe_store(
                                    b + 2, pp.unsafe_load(b + 2) + 1.0
                                )
                    i_row += 1
                while i_row < r1:
                    var r = rows_p.unsafe_load(i_row)
                    var g: Float64
                    var h = 1.0
                    comptime if CH:
                        g = Float64(g32.unsafe_load(2 * i_row))
                    else:
                        var gh = g32.unsafe_load[width=2](2 * i_row)
                        g = Float64(gh[0])
                        h = Float64(gh[1])
                    comptime for k in range(G):
                        if k < owned_n:
                            var b = Int(base[k]) + stride_rt * Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + h)
                            comptime if not CH:
                                pp.unsafe_store(
                                    b + 2, pp.unsafe_load(b + 2) + 1.0
                                )
                    i_row += 1
            else:
                while i_row < r1:
                    var g = score_t(grad_p.unsafe_load(i_row))
                    var h = 1.0
                    comptime if not CH:
                        h = score_t(hess_p.unsafe_load(i_row))
                    comptime for k in range(G):
                        if k < owned_n:
                            var b = Int(base[k]) + stride_rt * Int(
                                bins_all_p.unsafe_load(Int(col[k]) + i_row)
                            )
                            pp.unsafe_store(b, pp.unsafe_load(b) + g)
                            pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + h)
                            comptime if not CH:
                                pp.unsafe_store(
                                    b + 2, pp.unsafe_load(b + 2) + 1.0
                                )
                    i_row += 1

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
                        part_off
                        + blk * region
                        + (slot0 + k) * n_bins * stride
                    )
                    col[k] = f * n_rows

            # Zeroing stays fused into the pass that fills the slice, and is
            # now one contiguous run per feature rather than one run per
            # plane. Under `const_h` the run is `n_bins * 2` floats and the
            # third slot of every cell is neither zeroed nor read.
            comptime for k in range(GROUP):
                if k < owned:
                    var z0 = Int(base[k])
                    var zn = n_bins * stride
                    var zb = 0
                    while zb + W <= zn:
                        pp.unsafe_store(
                            z0 + zb, SIMD[DType.float64, W](0.0)
                        )
                        zb += W
                    while zb < zn:
                        pp.unsafe_store(z0 + zb, 0.0)
                        zb += 1

            # The scatter, dispatched to the arm this build is running.
            # The test is per dispatch unit -- there are 351 of them at the
            # shape this round measures -- and not per row.
            #
            # `full` is `owned == GROUP`. At 100 active features and width 8
            # that is true for twelve of the thirteen groups, so the branchy
            # spelling below is reached by the tail group only.
            var full = owned == GROUP
            if const_h:
                if arm == SERIAL_KERNEL_BASE:
                    scatter_base[True, GROUP](base, col, owned, r0, r1, stride)
                elif arm == SERIAL_KERNEL_STRIDE:
                    scatter[2, False, False, GROUP](base, col, owned, r0, r1)
                elif arm == SERIAL_KERNEL_PACKED:
                    scatter[2, False, True, GROUP](base, col, owned, r0, r1)
                elif full:
                    scatter[2, True, True, GROUP](base, col, owned, r0, r1)
                else:
                    scatter[2, False, True, GROUP](base, col, owned, r0, r1)
            else:
                if arm == SERIAL_KERNEL_BASE:
                    scatter_base[False, GROUP](base, col, owned, r0, r1, stride)
                elif arm == SERIAL_KERNEL_STRIDE:
                    scatter[3, False, False, GROUP](base, col, owned, r0, r1)
                elif arm == SERIAL_KERNEL_PACKED:
                    scatter[3, False, True, GROUP](base, col, owned, r0, r1)
                elif full:
                    scatter[3, True, True, GROUP](base, col, owned, r0, r1)
                else:
                    scatter[3, False, True, GROUP](base, col, owned, r0, r1)

    dispatch_feature_ranges_with(
        settings, accumulate_units, n_blocks * n_groups, plan.block_ops
    )

    # The node's count factor, LightGBM's `num_data / sum_hessian`. Under
    # `const_h` the second slot of a private cell counted rows rather than
    # summing hessians, so the node's total is `CONSTANT_HESSIAN * n_sub` and
    # this is exactly 1.0 -- which is what makes `derived_count` below exact
    # rather than an estimate. It is computed once for the whole build, from
    # the row count the caller already knows, and never from a reduction.
    var factor = cnt_factor(n_sub, CONSTANT_HESSIAN * Float64(n_sub))

    # The fold. One task per contiguous run of active slots; a slot writes
    # only its own output slice, and inside it every cell sums the blocks in
    # ascending order. That inner order is what the task count cannot touch,
    # and it is the whole of the determinism argument for this kernel.
    #
    # It runs in two passes over one slot. The first sums blocks 1..n-1 into
    # block 0's slots *in place*, which is elementwise over a contiguous run
    # of `n_bins * stride` floats and therefore vectorizes at full width; the
    # interleaved cell is what makes that run contiguous, where the separate
    # planes it replaced forced strided streams. The second walks the folded
    # cells once per bin and copies each cell's pair into the output's own
    # interleaved plane -- a 16-byte move on the general arm, not a scatter --
    # alongside the one count store. The addition order per slot is block 0,
    # then block 1, and so on, which is the order the previous per-cell fold
    # used.
    def fold_slots(s_start: Int, s_end: Int) {imm}:
        for j in range(s_start, s_end):
            var f = j if use_all else feat_p.unsafe_load(j)
            var out0 = f * n_bins
            var in0 = part_off + j * n_bins * stride
            var span = n_bins * stride
            for blk in range(1, n_blocks):
                var off = in0 + blk * region
                var i = 0
                while i + W <= span:
                    pp.unsafe_store(
                        in0 + i,
                        pp.unsafe_load[width=W](in0 + i)
                        + pp.unsafe_load[width=W](off + i),
                    )
                    i += W
                while i < span:
                    pp.unsafe_store(
                        in0 + i, pp.unsafe_load(in0 + i) + pp.unsafe_load(
                            off + i
                        )
                    )
                    i += 1
            # THE SCATTER IS GONE. On the general arm the private cell's
            # first two floats ARE the output cell, so this is a 16-byte
            # copy from one interleaved buffer to another rather than two
            # scalar stores into planes a whole histogram apart. Under
            # `const_h` the private cell is `(g, count)` and the output pair
            # is `(g, Float64(count))`, so one word is copied and one is a
            # rename of the value already in the register.
            for b in range(n_bins):
                var c0 = in0 + b * stride
                var sc = pp.unsafe_load(c0 + count_slot)
                if const_h:
                    # `sc` counted rows, so it is exactly `Float64(count)`;
                    # `factor` is exactly 1.0 and `derived_count` truncates
                    # `count + 0.5` back to `count`. The hessian slot is that
                    # same Float64, which is what the separate-plane path
                    # wrote cell for cell.
                    cp.unsafe_store(out0 + b, derived_count(sc, factor))
                    ghp.unsafe_store(
                        2 * (out0 + b),
                        SIMD[DType.float64, 2](pp.unsafe_load(c0), sc),
                    )
                else:
                    cp.unsafe_store(out0 + b, Int(sc))
                    ghp.unsafe_store(
                        2 * (out0 + b), pp.unsafe_load[width=2](c0)
                    )

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

    # Repeats stripped, as at the other three entry points and for the reason
    # `build_histogram_into_scratch` spells out. This one rebinds a local
    # instead of re-entering itself, because the function is parametric over
    # the two span origins and this path runs once per elected split rather
    # than inside a row loop, so one list copy is cheaper than making the
    # recursion re-infer them. The replica has to match the device number for
    # number, and a device build that named a feature twice would be wrong in
    # its own way; both sides accumulate a set.
    var used = features.copy()
    if _has_duplicate_features(features, n_features):
        used = _unique_features(features, n_features)

    var use_all = len(used) == 0
    var n_active = n_features if use_all else len(used)
    _zero_excluded(
        out._gh, out._count,
        n_features, n_bins, used,
        (n_features - n_active) * n_bins,
    )

    # The two output buffers are passed as separate `mut` lists rather than
    # reached through `out`, exactly as `build_histogram_into` explains: a
    # pointer taken from a struct field carries that field's origin, which a
    # worker closure cannot capture.
    _accumulate_replica(
        out._gh, out._count, fixed,
        data, grad_f32, hess_f32, rows, row_start, row_count,
        g_scale, h_scale, used,
        _resolve_const_hessian(const_hessian),
    )


def _accumulate_replica[
    grad_origin: ImmOrigin, hess_origin: ImmOrigin, //
](
    mut out_gh: List[Float64],
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
    var ghp = out_gh.unsafe_ptr()
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
            var gval = Float64(fixed_p.unsafe_load(base + b)) * g_inv
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
            ghp.unsafe_store(
                2 * (base + b),
                SIMD[DType.float64, 2](gval, Float64(hq_sum) * h_inv),
            )
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
    var gh_p = hist._gh.unsafe_ptr()
    var count_p = hist._count.unsafe_ptr()
    var base = feature * hist.n_bins
    # One 2W-wide pair accumulator where there were two W-wide ones. Lane
    # `2k` of `vp` receives the gradient of bins `base + k`, `base + k + W`,
    # ... in ascending order, which is exactly what lane `k` of the old `vg`
    # received; `deinterleave` is a permutation, not arithmetic, so the half
    # it hands back is that vector and `reduce_add` folds it in the same tree.
    # No bit moves.
    var vp = SIMD[DType.float64, 2 * W](0.0)
    var vc = SIMD[DType.int, W](0)
    var b = 0
    while b + W <= hist.n_bins:
        vp += gh_p.unsafe_load[width = 2 * W](2 * (base + b))
        vc += count_p.unsafe_load[width=W](base + b)
        b += W
    var halves = vp.deinterleave()
    var total_g = halves[0].reduce_add()
    var total_h = halves[1].reduce_add()
    var total_c = Int(vc.reduce_add())
    while b < hist.n_bins:
        var cell = gh_p.unsafe_load[width=2](2 * (base + b))
        total_g += cell[0]
        total_h += cell[1]
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
    mut out_gh: List[Float64],
    mut out_count: List[Int],
    parent_gh: List[Float64],
    parent_count: List[Int],
    child_gh: List[Float64],
    child_count: List[Int],
    size: Int,
    const_h: Bool,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    widths: List[Int] = [],
    n_bins: Int = 0,
) raises:
    """Parallel SIMD sibling subtraction over independent array borrows.

    **`widths` STORES ZERO in the cells no row can occupy instead of
    subtracting there.** `widths[f]` is feature f's realized bin count. On
    covertype 44 of 54 features have two bins against a 255-bin rectangle, so
    81 percent of the cells this pass touches are padding, and today each of
    them costs six streams -- read parent gh and count, read child gh and
    count, write out gh and count -- to compute `0.0 - 0.0`. With a width
    table they cost two.

    The zero must still be STORED. `out` is a pooled buffer whose contents are
    undefined, which is exactly why this function's contract is that it writes
    every element; skipping a cell would leave another node's statistics in it
    and the split scan reads all `n_bins` of every feature.

    Bit-identical under one invariant: every histogram in circulation holds
    zero in its trailing cells. That holds by construction, since the
    accumulate kernels zero the full rectangle before scattering and
    `Histogram.zeroed` starts there, so the arithmetic being replaced is
    `0.0 - 0.0`, whose result is the `+0.0` the store writes. An empty
    `widths` keeps the whole-rectangle path.

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

    **What the interleave did to each arm, in bytes per cell, and it is not
    the same sign on both.**

    The general arm improves in streams and is flat in bytes. It used to run
    a load, a load and a store on each of three planes: nine streams, 72
    bytes per cell. It now runs one 16-byte pair load from each operand, one
    16-byte pair store, and the three count accesses: six streams, the same
    72 bytes. Nothing was saved from DRAM; three fewer sequential streams
    were, which is three fewer prefetch trains and three fewer pages in
    flight.

    The elided arm gets **worse in bytes** and this is the honest price of
    the change. It used to read only the two gradient planes and the two
    count planes -- 32 bytes per cell in -- because the hessian value it
    needed was an integer difference already in a register. A gradient and a
    hessian now share a 16-byte cell and therefore a cache line, so reading
    the gradient fetches the hessian whether or not it is wanted: 48 bytes
    per cell in. Writes are unchanged at 24 (the pair store writes exactly
    the two words the two separate stores wrote). 56 bytes per cell becomes
    72, a 28.6% increase, on `const_h` sibling subtraction only. There is no
    addressing trick that avoids it: the unwanted word is in the line the
    wanted word is in. LightGBM does not pay it because LightGBM has no
    elided arm here at all -- `SubtractHistogram` walks the interleaved
    `hist_t` array and subtracts both words.

    Both arms keep their vector width. The elided one reaches it through
    `deinterleave` (to separate the gradient half of a pair load) and
    `interleave` (to rejoin it with the count difference cast to Float64);
    both are lane permutations with no arithmetic, so the vector stored is
    exactly the pair of values the two separate wide stores used to store.

    `subtract_ops_for_planes` is still told two planes on the elided arm.
    That now *understates* its traffic rather than describing it, which
    biases the crossover toward running it serially at a size where the
    general arm would dispatch; it is left alone because retuning a
    crossover is a measurement, and this lane has no clock.
    """
    var pgh = parent_gh.unsafe_ptr()
    var pc = parent_count.unsafe_ptr()
    var cgh = child_gh.unsafe_ptr()
    var cc = child_count.unsafe_ptr()
    var ogh = out_gh.unsafe_ptr()
    var oc = out_count.unsafe_ptr()

    def subtract_block(start: Int, end: Int) {imm}:
        comptime W = SIMD_LANES
        var i = start
        if const_h:
            # The elided arm's odd lanes are not a subtraction: they are the
            # integer count difference cast to Float64, which is the whole
            # point of eliding. `interleave` is a lane permutation with no
            # arithmetic, so the vector written here holds exactly the values
            # the two separate wide stores wrote, and the arm keeps its width.
            while i + W <= end:
                var pp2 = pgh.unsafe_load[width = 2 * W](2 * i)
                var cp2 = cgh.unsafe_load[width = 2 * W](2 * i)
                var dg = rebind[SIMD[DType.float64, W]](
                    pp2.deinterleave()[0]
                ) - rebind[SIMD[DType.float64, W]](cp2.deinterleave()[0])
                var dc = pc.unsafe_load[width=W](i) - cc.unsafe_load[width=W](
                    i
                )
                oc.unsafe_store(i, dc)
                ogh.unsafe_store(
                    2 * i, dg.interleave(dc.cast[DType.float64]())
                )
                i += W
            while i < end:
                ogh.unsafe_store(
                    2 * i, pgh.unsafe_load(2 * i) - cgh.unsafe_load(2 * i)
                )
                var dc = pc.unsafe_load(i) - cc.unsafe_load(i)
                oc.unsafe_store(i, dc)
                ogh.unsafe_store(2 * i + 1, Float64(dc))
                i += 1
            return
        # The general arm is now ONE elementwise stream over the pair plane
        # instead of two, at double the width. Six memory streams become
        # four, and `out`, `parent` and `child` each contribute one contiguous
        # run rather than two runs a whole histogram apart. The subtractions
        # are the same subtractions in the same pairing, so no bit moves.
        while i + W <= end:
            ogh.unsafe_store(
                2 * i,
                pgh.unsafe_load[width = 2 * W](2 * i)
                - cgh.unsafe_load[width = 2 * W](2 * i),
            )
            oc.unsafe_store(
                i, pc.unsafe_load[width=W](i) - cc.unsafe_load[width=W](i)
            )
            i += W
        while i < end:
            ogh.unsafe_store(
                2 * i,
                pgh.unsafe_load[width=2](2 * i)
                - cgh.unsafe_load[width=2](2 * i),
            )
            oc.unsafe_store(i, pc.unsafe_load(i) - cc.unsafe_load(i))
            i += 1

    var ops = subtract_ops_for_planes(size, 2) if const_h else subtract_ops(
        size
    )

    # Dispatched over FEATURES, because a feature's realized bins are a
    # contiguous run and its padding is the rest of its stride.
    #
    # Self-contained rather than reusing `subtract_block`: a closure calling
    # another closure that writes the same buffers is an aliasing error Mojo
    # refuses. The arithmetic is repeated literally -- same operands, same
    # order, same `const_h` elision -- because the extent is the only thing
    # meant to change.
    if n_bins > 0 and len(widths) * n_bins == size:
        var n_feat = len(widths)
        var wp = widths.unsafe_ptr()

        def subtract_features(fstart: Int, fend: Int) {imm}:
            for f in range(fstart, fend):
                var base = f * n_bins
                var w = wp.unsafe_load(f)
                if w > n_bins:
                    w = n_bins
                if w < 0:
                    w = 0
                for j in range(base, base + w):
                    var dc = pc.unsafe_load(j) - cc.unsafe_load(j)
                    oc.unsafe_store(j, dc)
                    if const_h:
                        ogh.unsafe_store(
                            2 * j,
                            SIMD[DType.float64, 2](
                                pgh.unsafe_load(2 * j)
                                - cgh.unsafe_load(2 * j),
                                Float64(dc),
                            ),
                        )
                    else:
                        ogh.unsafe_store(
                            2 * j,
                            pgh.unsafe_load[width=2](2 * j)
                            - cgh.unsafe_load[width=2](2 * j),
                        )
                for j in range(base + w, base + n_bins):
                    ogh.unsafe_store(2 * j, SIMD[DType.float64, 2](0.0))
                    oc.unsafe_store(j, 0)

        dispatch_feature_ranges_with(
            settings, subtract_features, n_feat, ops
        )
        return

    dispatch_rows_with(settings, subtract_block, size, ops)


def subtract_histogram_into(
    mut out: Histogram,
    parent: Histogram,
    child: Histogram,
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
    widths: List[Int] = [],
) raises:
    """`subtract_histogram` into a caller-owned buffer. Every element is
    written, so unlike the accumulating builders this one needs no zeroing
    pass at all.

    `const_hessian_env` is the fit's constant-hessian snapshot; the sentinel
    reads the environment live, once per subtraction, exactly as before. The
    grower reaches this once per split, which is the other half of the two
    reads per node `ConstHessianSettings` exists to remove.

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
        out._gh,
        out._count,
        parent._gh,
        parent._count,
        child._gh,
        child._count,
        parent.n_features * parent.n_bins,
        _resolve_const_hessian(const_hessian, const_hessian_env),
        settings,
        widths,
        parent.n_bins,
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


def row_major_block_region(compact_cells: Int) raises -> Int:
    """Float64 slots reserved for one block's private partials.

    `compact_cells * ROW_BLOCK_CELL_FLOATS`, padded up to a cache line. Two
    things about that expression are load-bearing and both are copied from
    `_accumulate_blocked_at`:

    - **`ROW_BLOCK_CELL_FLOATS`, not the addressed stride.** A cell is
      addressed at two floats under `const_h` and three otherwise, but it is
      always *reserved* at three, so the byte budget and therefore the block
      count cannot depend on `const_h`. The block count is part of the value;
      a fit that turned the specialization off must not fold a different
      number of partials.
    - **Padded per block, not per slot.** In this kernel an accumulate unit is
      a whole block, so one task owns a block's entire region and no two tasks
      contend inside one. Only the boundary between blocks needs isolating.
    """
    return align_cells_up(
        compact_cells * ROW_BLOCK_CELL_FLOATS, row_major_line_floats()
    )


def row_major_scratch_floats(
    plan: AccumulationPlan, compact_cells: Int
) raises -> Int:
    """Float64 slots the row-major blocked kernel needs in the caller's
    scratch, gather region included. Zero for an unblocked plan.

    Deliberately *not* `AccumulationPlan.block_scratch_floats`: that sizes the
    feature-major partials, which are `n_active * n_bins` cells. These are
    compact over the features' *realized* bin counts and line-padded, so they
    are a different number, and a scratch sized from one and indexed by the
    other is the failure this function exists to make impossible.
    """
    if not plan.blocked():
        return 0
    return (
        row_major_part_offset(plan)
        + plan.row_blocks * row_major_block_region(compact_cells)
    )


def row_major_part_offset(plan: AccumulationPlan) raises -> Int:
    """Where the private partials start in the caller's scratch: after the
    gather region, rounded up to a cache line so block 0's partials do not
    share a line with the last gathered pair.

    **One Float64 word per row, not two.** The gather holds a row's
    `(gradient, hessian)` as two Float32 halves of one word (`_gather_pairs`),
    which is LightGBM's `score_t` precision. `block_rows * row_blocks` rather
    than the row count because it is derived from the plan alone, and the
    ceiling geometry makes it never smaller than the rows it has to cover.
    """
    return align_cells_up(
        plan.block_rows * plan.row_blocks, row_major_line_floats()
    )


def build_histogram_subset_row_major(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
        const_hessian, settings, const_hessian_env,
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
    # Repeats stripped here too, and not only for correctness: this builder is
    # asserted bit-identical to `build_histogram_subset` on the same
    # arguments, so it has to strip exactly what that one strips or the two
    # would disagree on any list a caller repeated an id in.
    if _has_duplicate_features(features, data.n_features):
        var unique = _unique_features(features, data.n_features)
        build_histogram_subset_row_major_into_scratch(
            out, pairs, data, grad, hess, rows, row_start, row_count, unique,
            const_hessian, settings, const_hessian_env,
        )
        return
    var const_h = _resolve_const_hessian(const_hessian, const_hessian_env)
    if const_h and _resolve_const_hessian_verify(const_hessian_env):
        _check_constant_hessian(hess, data.n_rows)

    if _resolve_narrow(const_hessian_env):
        _accumulate_subset_row_major[True](
            out._gh, out._count, pairs,
            data, grad, hess, rows, row_start, row_count, features, const_h,
            settings,
        )
    else:
        _accumulate_subset_row_major[False](
            out._gh, out._count, pairs,
            data, grad, hess, rows, row_start, row_count, features, const_h,
            settings,
        )


def _accumulate_subset_row_major[
    NARROW: Bool
](
    mut out_gh: List[Float64],
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
        settings, n_features, n_active, n_bins, n_sub, True, const_h
    )

    _zero_excluded(
        out_gh, out_count,
        n_features, n_bins, features, plan.excluded_ops, settings,
    )
    if n_active <= 0 or n_sub <= 0:
        # Nothing active: the excluded pass above has already zeroed every
        # slice this build owns, and an empty row window leaves the active
        # ones zero too. Handled here rather than inside a kernel so neither
        # kernel needs an empty case.
        if n_active > 0:
            _zero_active_slices(
                out_gh, out_count, n_features, n_bins, features,
                n_active, plan.active_ops, settings,
            )
        return

    # A blocked node always gathers, even below `compact_min_rows`, exactly as
    # `_accumulate_subset` decided for the feature-major kernel: the blocked
    # kernel then has one row source rather than two, and a node large enough
    # to block is by construction large enough for the gather to be a rounding
    # error against its own accumulation. It cannot move a bin -- `score_t` is
    # idempotent and both the gather and the direct read apply it, so the two
    # deliver the same Float64.
    #
    # Both are off under `float64`. Blocking is off for the reason
    # `_accumulate_subset` gives and that reason still holds, since the
    # gathered word is two Float32 and this kernel's blocked arm reads
    # nothing else. `_accumulate_subset_row_major_blocked` is not even handed
    # `grad` and `hess`.
    #
    # THE GATHER IS NO LONGER THE SAME ON BOTH SIDES, and this comment used to
    # say it was. `_accumulate_subset` now consults
    # `MOJOTREES_CPU_FLOAT64_GATHER` in its `else` arm and will gather under
    # `float64` when asked; the `comptime if NARROW` below has no `else`, so
    # this layout never does, at any setting of that variable. That is a
    # deliberate scope and not an oversight: row-major is not the shipped
    # layout, forcing it measured slower on this machine, and its kernels read
    # a Float32 view unconditionally, so the gathered word would have to be
    # retyped here before the switch could mean anything.
    #
    # **The two layouts still produce the same histogram under either
    # setting**, which is the property this paragraph exists to protect, and
    # it no longer rests on the two arms agreeing about the gather. It rests
    # on the gather being unable to move a cell at all: gathered and direct
    # reads deliver the same Float64, so a build that gathers and a build that
    # does not add the same value into the same bin. `float64_gather_arm`
    # carries that argument in full.
    var use_pairs = False
    var blocked = False
    comptime if NARROW:
        use_pairs = plan.compact_rows or plan.blocked()
        blocked = plan.blocked()
    if use_pairs:
        ensure_pair_capacity(pairs, n_sub)

    if not blocked:
        if use_pairs:
            # `True` explicitly, not by default. The row-major layout stays
            # packed-only: it is not the shipped layout, forcing it measured
            # slower on this machine, and its kernel reads a Float32 view
            # unconditionally. `use_pairs` is False here under `float64`
            # anyway, so this arm is unreachable on that precision; the
            # parameter is spelled so that a later change to `_gather_pairs`'s
            # default cannot quietly retype this buffer.
            _gather_pairs[True](
                pairs, grad, hess, rows, row_start, n_sub, plan.gather_ops,
                settings,
            )
        var group = plan.group_width
        if group >= 16:
            _accumulate_subset_row_major_at[16, NARROW](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, use_pairs,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        elif group >= 8:
            _accumulate_subset_row_major_at[8, NARROW](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, use_pairs,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        elif group >= 4:
            _accumulate_subset_row_major_at[4, NARROW](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, use_pairs,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        elif group >= 2:
            _accumulate_subset_row_major_at[2, NARROW](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, use_pairs,
                n_active, plan.group_count, plan.active_ops, const_h,
                settings,
            )
        else:
            _accumulate_subset_row_major_at[1, NARROW](
                out_gh, out_count, pairs, data, grad, hess,
                rows, row_start, row_count, features, use_pairs,
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
    # After the resize, so the gather is not written into a buffer that is
    # about to be reallocated. `True` explicitly for the reason given at the
    # other row-major gather above: this layout is packed-only.
    _gather_pairs[True](
        pairs, grad, hess, rows, row_start, n_sub, plan.gather_ops, settings,
    )
    _accumulate_subset_row_major_blocked(
        out_gh, out_count, pairs, data, rows,
        row_start, row_count, features, plan, part_off, n_active, cells,
        meta, const_h, settings,
    )


def _zero_active_slices(
    mut out_gh: List[Float64],
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
    var ghp = out_gh.unsafe_ptr()
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
                ghp.unsafe_store(
                    2 * (base + b), SIMD[DType.float64, 2 * W](0.0)
                )
                cp.unsafe_store(base + b, SIMD[DType.int, W](0))
                b += W
            while b < n_bins:
                ghp.unsafe_store(2 * (base + b), SIMD[DType.float64, 2](0.0))
                cp.unsafe_store(base + b, 0)
                b += 1

    dispatch_feature_ranges_with(settings, zero_slots, n_active, total_ops)


def _accumulate_subset_row_major_at[
    GROUP: Int, NARROW: Bool
](
    mut out_gh: List[Float64],
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
    var ghp = out_gh.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var rm_p = data.row_bins.unsafe_ptr()
    var byte_p = data.row_byte.unsafe_ptr()
    var shift_p = data.row_shift.unsafe_ptr()
    var mask_p = data.row_mask.unsafe_ptr()
    # The gather holds two Float32 per Float64 word, so it is read through a
    # Float32 view. `_gather_pairs` is where the packing is argued.
    var pairs_p = pairs.unsafe_ptr().unsafe_bitcast[Float32]()
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
                    # The `const_h` skip of the hessian plane is gone, and
                    # the branch with it: a hessian shares a 16-byte cell and
                    # therefore a cache line with its gradient, so the store
                    # that zeroes the gradient owns the line either way and
                    # the elided lanes were never a saved line -- only a
                    # saved 8 bytes inside one already being written. The
                    # refill below still overwrites them.
                    while zb + W <= n_bins:
                        ghp.unsafe_store(
                            2 * (z0 + zb), SIMD[DType.float64, 2 * W](0.0)
                        )
                        cp.unsafe_store(z0 + zb, SIMD[DType.int, W](0))
                        zb += W
                    while zb < n_bins:
                        ghp.unsafe_store(
                            2 * (z0 + zb), SIMD[DType.float64, 2](0.0)
                        )
                        cp.unsafe_store(z0 + zb, 0)
                        zb += 1

            if const_h:
                if compact:
                    for i_row in range(n_sub):
                        var rec = rows_p.unsafe_load(i_row) * stride
                        var g = Float64(pairs_p.unsafe_load(2 * i_row))
                        comptime for k in range(GROUP):
                            if k < owned:
                                var raw = Int(
                                    rm_p.unsafe_load(rec + Int(rbyte[k]))
                                )
                                var b = Int(base[k]) + (
                                    (raw >> Int(rshift[k])) & Int(rmask[k])
                                )
                                var p = b << 1
                                ghp.unsafe_store(p, ghp.unsafe_load(p) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
                else:
                    for i_row in range(n_sub):
                        var r = rows_p.unsafe_load(i_row)
                        var rec = r * stride
                        var g = derivative[NARROW](grad_p.unsafe_load(r))
                        comptime for k in range(GROUP):
                            if k < owned:
                                var raw = Int(
                                    rm_p.unsafe_load(rec + Int(rbyte[k]))
                                )
                                var b = Int(base[k]) + (
                                    (raw >> Int(rshift[k])) & Int(rmask[k])
                                )
                                var p = b << 1
                                ghp.unsafe_store(p, ghp.unsafe_load(p) + g)
                                cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            elif compact:
                for i_row in range(n_sub):
                    var rec = rows_p.unsafe_load(i_row) * stride
                    var g = Float64(pairs_p.unsafe_load(2 * i_row))
                    var h = Float64(pairs_p.unsafe_load(2 * i_row + 1))
                    comptime for k in range(GROUP):
                        if k < owned:
                            var raw = Int(
                                rm_p.unsafe_load(rec + Int(rbyte[k]))
                            )
                            var b = Int(base[k]) + (
                                (raw >> Int(rshift[k])) & Int(rmask[k])
                            )
                            var p = b << 1
                            ghp.unsafe_store(
                                p,
                                ghp.unsafe_load[width=2](p)
                                + SIMD[DType.float64, 2](g, h),
                            )
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    var rec = r * stride
                    var g = derivative[NARROW](grad_p.unsafe_load(r))
                    var h = derivative[NARROW](hess_p.unsafe_load(r))
                    comptime for k in range(GROUP):
                        if k < owned:
                            var raw = Int(
                                rm_p.unsafe_load(rec + Int(rbyte[k]))
                            )
                            var b = Int(base[k]) + (
                                (raw >> Int(rshift[k])) & Int(rmask[k])
                            )
                            var p = b << 1
                            ghp.unsafe_store(
                                p,
                                ghp.unsafe_load[width=2](p)
                                + SIMD[DType.float64, 2](g, h),
                            )
                            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

            if const_h:
                comptime for k in range(GROUP):
                    if k < owned:
                        var f0 = Int(base[k])
                        var fb = 0
                        # Writing only the odd lanes of an interleaved run
                        # means reading the even ones back to rejoin them.
                        # The read is of lines this task just wrote and that
                        # are therefore in its own L1, and `deinterleave` /
                        # `interleave` are lane permutations with no
                        # arithmetic, so the Float64 that lands is the Float64
                        # the plane store landed.
                        while fb + W <= n_bins:
                            var cur = ghp.unsafe_load[width = 2 * W](
                                2 * (f0 + fb)
                            )
                            # `rebind` only tells the elaborator that
                            # `(2 * W) / 2` is `W`; it is not a conversion
                            # and emits nothing.
                            var gvec = rebind[SIMD[DType.float64, W]](
                                cur.deinterleave()[0]
                            )
                            ghp.unsafe_store(
                                2 * (f0 + fb),
                                gvec.interleave(
                                    cp.unsafe_load[width=W](f0 + fb).cast[
                                        DType.float64
                                    ]()
                                ),
                            )
                            fb += W
                        while fb < n_bins:
                            ghp.unsafe_store(
                                2 * (f0 + fb) + 1,
                                Float64(cp.unsafe_load(f0 + fb)),
                            )
                            fb += 1

    dispatch_feature_ranges_with(
        settings, accumulate_groups, n_groups, active_ops
    )


def _accumulate_subset_row_major_blocked(
    mut out_gh: List[Float64],
    mut out_count: List[Int],
    mut pairs: List[Float64],
    data: BinnedMatrix,
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

    **One axis, not two, and that is the point.** `_accumulate_blocked_at`
    dispatches over `(block, group)` pairs because a group must be narrow
    enough that its histogram slices stay in L1. This one dispatches over
    **blocks alone**: a block owns a private histogram over *every* active
    feature and walks its rows once, reading each row's whole record. Three
    consequences, and only the third is a trade:

    - The node's gathered derivatives are read **once**, not once per group.
      The feature-major kernel re-streams the gather buffer `ceil(n_active /
      group_width)` times, which at 50 features and width 2 is 25 passes.
    - A row's record is one cache line and serves all `n_active` features,
      where the feature-major kernel pays one line per (row, feature) on a
      node whose rows are scattered. It also needs **one** prefetch per row
      rather than one per (row, feature), for the same reason.
    - The private histogram is `sum_j feature_bins[f_j]` cells rather than
      `group_width * n_bins`, so it is bigger than L1 where the feature-major
      one was sized to fit. That is the cost, and it is why the compact
      cumulative offsets matter rather than being tidiness: they are what
      keeps the accumulator in L2 instead of past it.

    **The private cell is LightGBM's interleaved `hist_t`, the same cell
    `_accumulate_blocked_at` uses**, and it must be, because these two
    decompositions are compared bit for bit. `include/LightGBM/bin.h` gives
    `GET_GRAD(hist, i) = hist[(i) << 1]` and `GET_HESS(hist, i) = hist[((i) <<
    1) + 1]`: gradient and hessian adjacent, no count plane. A cell here is
    `stride` floats -- two under `const_h`, where the second slot is a count
    exactly as LightGBM aliases it, and three otherwise, where the third slot
    keeps the exact count `derived_count` would have had to round. The address
    is `part_off + blk * region + (bin_offset[f] + bin) * stride`, so the
    compact offsets and the interleave compose: one contiguous cell per (row,
    feature) instead of three planes a region apart, over a compact rather
    than a `n_active * n_bins` extent.

    **The allocation is unconditional at `ROW_BLOCK_CELL_FLOATS`** even when
    the addressed stride is two, for the reason `row_major_block_region`
    gives: the block count must not depend on `const_h`.

    **Parallelism comes from blocks only, so `plan.row_blocks` is the whole
    fan-out** for the accumulate pass. The plan is not allowed to grow it for
    this kernel's benefit -- the block count is part of the value, and a
    layout that changed it would move bits. A node the plan does not block is
    therefore not a node for this kernel; `_accumulate_subset_row_major` sends
    those to the unblocked row-major kernel, which partitions by feature.

    **Determinism**, unchanged: a block's rows are walked ascending, a block is
    never split across tasks, and the fold sums blocks ascending inside every
    cell. That per-cell order -- block 0, then block 1, and so on -- is
    identical to `_accumulate_blocked_at`'s, which is what makes the two
    layouts bit-identical rather than merely close. The fold's own parallel
    split is over disjoint output ranges and cannot reassociate; see
    `FOLD_BIN_CHUNK`.

    **The elided hessian plane** is exact for the reason `_accumulate_blocked_at`
    gives: under `const_h` the second slot counted rows, `CONSTANT_HESSIAN` is
    1.0, the sum of `n` copies of 1.0 is exactly `Float64(n)` because every
    partial is an exactly representable integer below 2^53, and `cnt_factor`
    is then exactly 1.0 so `derived_count` truncates `count + 0.5` back to
    `count`. Both kernels call the same two functions on the same value.

    **The tail bins.** A compact slot holds `feature_bins[f]` cells and the
    output slice holds `n_bins`. The bins above the highest one the feature
    realizes are written as zero by the fold, which is bit for bit what the
    feature-major kernel leaves there: no row ever lands in them, so the
    gradient is `0.0`, the count is `0`, and the hessian is `0.0` on both the
    interleaved and the elided path.
    """
    var n_bins = data.n_bins
    var n_sub = row_count
    var n_blocks = plan.row_blocks
    var block_rows = plan.block_rows
    var rec_stride = data.row_stride
    var use_all = len(features) == 0
    var ghp = out_gh.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var rm_p = data.row_bins.unsafe_ptr()
    var meta_p = meta.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    # ONE pointer into `pairs`. The gather occupies `[0, n_sub)` and the
    # private partials `[part_off, ...)`, and two pointers carrying the same
    # origin into one parallel closure is an aliasing error Mojo refuses to
    # compile. So `part_off` is folded into every private index instead, and
    # the Float32 view of the gather is taken inside the closure from this
    # same pointer.
    var pp = pairs.unsafe_ptr()
    # Floats per private cell, and where the count lives inside one. Both are
    # `_accumulate_blocked_at`'s, and they have to be.
    var stride = (
        ROW_BLOCK_CELL_FLOATS_CONST_H if const_h else ROW_BLOCK_CELL_FLOATS
    )
    var count_slot = 1 if const_h else 2
    var region = row_major_block_region(compact_cells)
    comptime W = SIMD_LANES

    def accumulate_blocks(u_start: Int, u_end: Int) {imm}:
        var g32 = pp.unsafe_bitcast[Float32]()
        for blk in range(u_start, u_end):
            var r0 = blk * block_rows
            var r1 = r0 + block_rows
            if r1 > n_sub:
                r1 = n_sub
            var b0 = part_off + blk * region

            # Zeroing stays fused into the pass that fills the slice, and is
            # one contiguous run over the whole compact extent rather than one
            # run per plane. Under `const_h` the run is `compact_cells * 2`
            # floats and the third slot of every cell is neither zeroed nor
            # read; the padding past the extent is never touched at all.
            var zn = compact_cells * stride
            var zb = 0
            while zb + W <= zn:
                pp.unsafe_store(b0 + zb, SIMD[DType.float64, W](0.0))
                zb += W
            while zb < zn:
                pp.unsafe_store(b0 + zb, 0.0)
                zb += 1

            # The scatter, in two halves as LightGBM has them: the first
            # issues the prefetch and the second is the tail that has nothing
            # left to prefetch. Same additions, same order. **One prefetch per
            # row**, on the record, because one record carries every feature
            # this loop is about to read -- where the feature-major kernel
            # issues one per (row, feature).
            var pf_end = r1 - PREFETCH_ROW_DISTANCE
            var i_row = r0
            if const_h:
                while i_row < pf_end:
                    var rec = rows_p.unsafe_load(i_row) * rec_stride
                    var g = Float64(g32.unsafe_load(2 * i_row))
                    prefetch[SCATTER_PREFETCH](
                        rm_p.unsafe_offset(
                            rows_p.unsafe_load(i_row + PREFETCH_ROW_DISTANCE)
                            * rec_stride
                        )
                    )
                    for j in range(n_active):
                        var m = 4 * j
                        var raw = Int(
                            rm_p.unsafe_load(rec + meta_p.unsafe_load(m + 1))
                        )
                        var b = b0 + stride * (
                            meta_p.unsafe_load(m)
                            + (
                                (raw >> meta_p.unsafe_load(m + 2))
                                & meta_p.unsafe_load(m + 3)
                            )
                        )
                        pp.unsafe_store(b, pp.unsafe_load(b) + g)
                        pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + 1.0)
                    i_row += 1
                while i_row < r1:
                    var rec = rows_p.unsafe_load(i_row) * rec_stride
                    var g = Float64(g32.unsafe_load(2 * i_row))
                    for j in range(n_active):
                        var m = 4 * j
                        var raw = Int(
                            rm_p.unsafe_load(rec + meta_p.unsafe_load(m + 1))
                        )
                        var b = b0 + stride * (
                            meta_p.unsafe_load(m)
                            + (
                                (raw >> meta_p.unsafe_load(m + 2))
                                & meta_p.unsafe_load(m + 3)
                            )
                        )
                        pp.unsafe_store(b, pp.unsafe_load(b) + g)
                        pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + 1.0)
                    i_row += 1
            else:
                while i_row < pf_end:
                    var rec = rows_p.unsafe_load(i_row) * rec_stride
                    var gh = g32.unsafe_load[width=2](2 * i_row)
                    var g = Float64(gh[0])
                    var h = Float64(gh[1])
                    prefetch[SCATTER_PREFETCH](
                        rm_p.unsafe_offset(
                            rows_p.unsafe_load(i_row + PREFETCH_ROW_DISTANCE)
                            * rec_stride
                        )
                    )
                    for j in range(n_active):
                        var m = 4 * j
                        var raw = Int(
                            rm_p.unsafe_load(rec + meta_p.unsafe_load(m + 1))
                        )
                        var b = b0 + stride * (
                            meta_p.unsafe_load(m)
                            + (
                                (raw >> meta_p.unsafe_load(m + 2))
                                & meta_p.unsafe_load(m + 3)
                            )
                        )
                        pp.unsafe_store(b, pp.unsafe_load(b) + g)
                        pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + h)
                        pp.unsafe_store(b + 2, pp.unsafe_load(b + 2) + 1.0)
                    i_row += 1
                while i_row < r1:
                    var rec = rows_p.unsafe_load(i_row) * rec_stride
                    var gh = g32.unsafe_load[width=2](2 * i_row)
                    var g = Float64(gh[0])
                    var h = Float64(gh[1])
                    for j in range(n_active):
                        var m = 4 * j
                        var raw = Int(
                            rm_p.unsafe_load(rec + meta_p.unsafe_load(m + 1))
                        )
                        var b = b0 + stride * (
                            meta_p.unsafe_load(m)
                            + (
                                (raw >> meta_p.unsafe_load(m + 2))
                                & meta_p.unsafe_load(m + 3)
                            )
                        )
                        pp.unsafe_store(b, pp.unsafe_load(b) + g)
                        pp.unsafe_store(b + 1, pp.unsafe_load(b + 1) + h)
                        pp.unsafe_store(b + 2, pp.unsafe_load(b + 2) + 1.0)
                    i_row += 1

    dispatch_feature_ranges_with(
        settings, accumulate_blocks, n_blocks, plan.block_ops
    )

    # The node's count factor, exactly as `_accumulate_blocked_at` computes
    # it: under `const_h` the node's hessian total is `CONSTANT_HESSIAN *
    # n_sub`, so this is exactly 1.0 and `derived_count` is exact rather than
    # estimated. Computed once for the build, never from a reduction.
    var factor = cnt_factor(n_sub, CONSTANT_HESSIAN * Float64(n_sub))

    var n_chunks = (n_bins + FOLD_BIN_CHUNK - 1) // FOLD_BIN_CHUNK
    if n_chunks < 1:
        n_chunks = 1

    # The fold, in `_accumulate_blocked_at`'s two passes but over `(slot, bin
    # chunk)` units rather than whole slots. Pass one sums blocks 1..n-1 into
    # block 0's cells *in place*, which is elementwise over a contiguous run
    # of `(lim - lo) * stride` floats and therefore vectorizes at full width;
    # the interleaved cell is what makes that run contiguous. Pass two walks
    # the folded cells once per bin and copies each cell's pair into the
    # output's own interleaved plane, alongside the one count store. The
    # addition order per cell is block 0, then block 1, and so on, which is
    # the order the feature-major fold uses and the reason the two agree.
    def fold_units(u_start: Int, u_end: Int) {imm}:
        for u in range(u_start, u_end):
            var j = u // n_chunks
            var c = u - j * n_chunks
            var f = j if use_all else feat_p.unsafe_load(j)
            var slot_off = meta_p.unsafe_load(4 * j)
            var width = meta_p.unsafe_load(4 * (j + 1)) - slot_off
            var out0 = f * n_bins
            var lo = c * FOLD_BIN_CHUNK
            var hi = lo + FOLD_BIN_CHUNK
            if hi > n_bins:
                hi = n_bins
            var lim = width if width < hi else hi
            if lim > lo:
                var in0 = part_off + (slot_off + lo) * stride
                var span = (lim - lo) * stride
                for blk in range(1, n_blocks):
                    var off = in0 + blk * region
                    var i = 0
                    while i + W <= span:
                        pp.unsafe_store(
                            in0 + i,
                            pp.unsafe_load[width=W](in0 + i)
                            + pp.unsafe_load[width=W](off + i),
                        )
                        i += W
                    while i < span:
                        pp.unsafe_store(
                            in0 + i,
                            pp.unsafe_load(in0 + i) + pp.unsafe_load(off + i),
                        )
                        i += 1
                # The same 16-byte cell copy as the feature-major fold; see
                # `_accumulate_blocked_at` for why the scatter is gone.
                for b in range(lo, lim):
                    var c0 = part_off + (slot_off + b) * stride
                    var sc = pp.unsafe_load(c0 + count_slot)
                    if const_h:
                        cp.unsafe_store(out0 + b, derived_count(sc, factor))
                        ghp.unsafe_store(
                            2 * (out0 + b),
                            SIMD[DType.float64, 2](pp.unsafe_load(c0), sc),
                        )
                    else:
                        cp.unsafe_store(out0 + b, Int(sc))
                        ghp.unsafe_store(
                            2 * (out0 + b), pp.unsafe_load[width=2](c0)
                        )
            # The bins this feature never realizes, and the whole chunk when
            # it starts past the realized width. Zero on both paths, which is
            # what the feature-major kernel leaves there.
            var b2 = lim if lim > lo else lo
            while b2 < hi:
                ghp.unsafe_store(2 * (out0 + b2), SIMD[DType.float64, 2](0.0))
                cp.unsafe_store(out0 + b2, 0)
                b2 += 1

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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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
            features, const_hessian, settings, const_hessian_env,
        )
        return BIN_LAYOUT_ROW_MAJOR
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian, settings, const_hessian_env,
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
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
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

    `const_hessian_env` is threaded through for the reason `settings` is, and
    it is not cosmetic: the two constant-hessian variables decide the private
    cell's stride (two floats or three) and therefore how much traffic each
    arm streams. A probe that read them live while the fit ran on a snapshot
    would be timing a configuration the fit is not in, which is the shape of
    defect this campaign has already shipped four times under the name
    "accepted and then quietly ignored".

    **The order the two arms run in is a bias, stated rather than corrected.**
    Feature-major runs first and row-major second, so row-major inherits the
    caches feature-major warmed -- including the gather buffer, which both
    arms read and only the first arm pays to fill. LightGBM's own probe
    carries the same bias in the same direction. What it means for a reader:
    a row-major win by a small margin is worth less than a feature-major win
    by the same margin, and this function is a tie-break rather than a
    measurement. The A/B that decides whether the rule is right at all is
    `bench/bench_cpu_bin_layout.mojo`, which interleaves whole fits.
    """
    if not data.has_row_major():
        return BIN_LAYOUT_FEATURE_MAJOR
    var requested = resolve_bin_layout(env_bin_layout(), True)
    if requested != BIN_LAYOUT_AUTO:
        return requested

    var t0 = perf_counter_ns()
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian, settings, const_hessian_env,
    )
    var t1 = perf_counter_ns()
    build_histogram_subset_row_major_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features,
        const_hessian, settings, const_hessian_env,
    )
    var t2 = perf_counter_ns()
    if (t2 - t1) < (t1 - t0):
        return BIN_LAYOUT_ROW_MAJOR
    return BIN_LAYOUT_FEATURE_MAJOR
