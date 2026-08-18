"""Apple-silicon CPU tuning policy.

A pure policy layer for the multicore CPU backend, in the same shape as
`apple_gpu_policy.mojo`: detected machine facts and a workload shape in, a
plan out. It allocates nothing, dispatches nothing, and touches no dataset,
so every decision below can be exercised and asserted on any machine.

**One quantity here changes a result and the rest do not**, and the split is
worth reading before touching anything in this file.

Everything except the row-block count is a scheduling or a memory-traffic
decision: how many tasks a loop is split into, whether an accumulation
gathers its gradients into a scratch buffer first, how many features one
inner loop interleaves. The kernels in `histogram.mojo` and the dispatch
shapes in `parallel.mojo` keep each feature's summation inside one task and
each row block in ascending order whatever this module answers, so the output
is bit-identical at every one of those settings.

`AccumulationPlan.row_blocks` is the exception. A blocked accumulation folds
per-block partial sums, and Float64 addition is not associative, so the block
count is part of the value. That is why `plan_row_block_count` takes the
workload shape and nothing else: not the core count, not the core pool, not
`MOJOTREES_NUM_WORKERS`, not the task count. Determinism across worker counts
and across machines is preserved by keeping every machine fact out of that one
function, and by the fold running in ascending block order however many tasks
the blocks are dispatched over. `MOJOTREES_CPU_ROW_BLOCKS` and
`MOJOTREES_CPU_ROW_BLOCK_AMORTIZE` are explicit requests for a different
summation order and are the only two variables named in this file that move
bits. Both stay workload-only: neither can be set to a value that makes the
block count depend on the worker count, the task count or the machine.

What this module encodes
------------------------
- **Asymmetric cores, handled conservatively.** Apple silicon reports a mix
  of performance and efficiency cores, and `num_physical_cores()` counts both.
  A barrier-synchronized fan-out finishes when its slowest task finishes, so
  a task landing on an efficiency core sets the pace. The two portable
  responses are to over-decompose (more tasks than cores, so a fast core can
  take a second chunk) and to shrink the pool to the performance cores. The
  default is the first, unchanged from what this repository already shipped,
  because the second throws away real cores on the strength of an assumption
  and no measurement here justifies it. `MOJOTREES_CPU_CORE_POOL=performance`
  selects the second for a benchmark that wants to measure the difference.
  Neither path pins a thread: no portable affinity control is used, and none
  is assumed to exist.
- **Cache-sized inner loops.** How many features one accumulation loop may
  interleave is bounded by an assumed L1 data-cache size, so the interleaved
  group's histogram slices stay resident. The bound is derived from
  `n_bins` and the histogram's bytes-per-cell, not from a fixed feature count.
  The width is then floored to a rung of the ladder `histogram.mojo`
  instantiates its accumulation kernel at (1, 2, 4, 8, 16), because a width
  with no instantiation cannot run.
- **A group width the schedule can actually deliver, and can afford.** The
  interleave only happens if one task holds a whole group, so the dispatch is
  over *groups* rather than over features. Before that, `parallel.plan_tasks`
  fanned 50 features over 40 tasks, 30 of which held a single feature and
  therefore could not interleave anything, so a nominal width of 2 ran at an
  effective 1.25 and the knob that changed the task count silently changed the
  width too.
  Dispatching over groups makes the width real and makes its cost real with
  it: the task count is now divided by exactly the factor the memory traffic
  is. `TASK_BALANCE_FACTOR` is the price of that, a floor on how many groups
  the width must leave per core, because a width chosen from bytes alone would
  have put 13 tasks on a 10-core machine and idled seven of them through the
  second barrier round. Both numbers are reported by `describe_cpu_policy`,
  since a width without its utilization is half a decision.
- **Work estimates that include zeroing.** A tree node's histogram build
  costs one write of the whole `n_features * n_bins` output plus one
  accumulate per (active feature, row). On a small node the first term
  dominates, so an estimate counting only rows sends a multi-megabyte write
  down the serial path. `derive_accumulation_plan` reports the two terms
  separately, and each dispatch is sized by the term it actually runs.

What this module deliberately does NOT encode
---------------------------------------------
- **No development-machine constants.** No M-series part number appears
  anywhere in this file, and no per-chip table is consulted. Everything is
  derived from what `std.sys.info` reports on the running machine, plus the
  three assumptions named below, which are conservative floors rather than
  the development Mac's values.
- **No claim that any of it is measured.** Every constant here is a starting
  point with a stated mechanism, not a tuned optimum.
  `bench/apple/cpu_plan.json` lists the sweep that would turn each one into
  a measured value, and the handoff lists which claims need profiler or
  disassembly evidence before anyone repeats them.
- **No affinity, no QoS class, no thread pinning, no core-type
  detection beyond the counts `std.sys.info` reports.**
- **No Python.** The policy is consumed by Mojo hot paths; a Python
  performance policy would be unreachable from them.

Assumed constants
-----------------
Three numbers are not reported by any portable API and are assumed:
`ARM_CACHE_LINE_BYTES`, `GENERIC_CACHE_LINE_BYTES`, and `ASSUMED_L1D_BYTES`.
Each is set to a conservative floor, because underestimating a cache costs
some locality while overestimating it produces a plan that thrashes. They are
named, not inlined, so a machine that turns out to differ has one place to
change and the handoff has something to point a measurement at.

`ASSUMED_L1D_BYTES` stays at its 64 KiB floor, and that is a decision rather
than an oversight. The development M4's L1 data cache is widely reported to be
larger, and raising the constant would widen the interleave a rung at 255
bins. Two things stop it. The number is not read from any API this project
calls, so raising it would replace a stated assumption with a remembered
figure; and there is no CPU equivalent of the device report that
`gpu_split_policy._is_observed_m4` guards its hardware-specific threshold
with, because nothing portable in `std.sys.info` identifies the part. Guessing
the part from core counts is exactly the per-chip table the section above
refuses. So the floor stands, the ladder is selected against it, and
`MOJOTREES_CPU_FEATURE_GROUP` is the way to run the wider rungs the floor
declines to choose. A measurement that resolves this belongs in
`bench/apple/cpu_plan.json`, not in a constant nobody verified.

Environment contract (all optional; all scheduling-only but the last two)
-------------------------------------------------------------------------
- `MOJOTREES_CPU_TASKS_PER_CORE`: positive integer, auto-mode fan-out per
  core. Default `DEFAULT_TASKS_PER_CORE`.
- `MOJOTREES_CPU_CORE_POOL`: `all` (default) counts every physical core;
  `performance` counts only the reported performance cores.
- `MOJOTREES_CPU_FEATURE_GROUP`: a rung of the ladder (`1`, `2`, `4`, `8`,
  `16`), how many features one accumulation loop interleaves. Unset means the
  derived width: `DEFAULT_FEATURE_GROUP` clamped by the L1 estimate, by the
  active feature count, and by the core count. An explicit request overrides
  all three, because the estimate is the thing the knob exists to test; an
  off-ladder value (`3`, `32`, a word) is *refused*, not rounded, because no
  kernel is instantiated at it.
- `MOJOTREES_CPU_COMPACT_MIN_ROWS`: row count below which a subset
  accumulation skips the gradient/hessian gather. Default
  `DEFAULT_COMPACT_MIN_ROWS`.
- `MOJOTREES_CPU_ROW_BLOCKS`: how many contiguous row blocks a subset
  accumulation splits a node into, each with a private histogram folded in
  ascending block order. Unset (or `0`) means the derived count; `1` turns
  blocking off and reproduces the feature-partition accumulation exactly; `N
  >= 2` forces N blocks, bypassing the amortization floor and the byte budget
  and clamped only by the node's row count and `MAX_ROW_BLOCKS`.
  **This one moves bits.** Two different block counts are two different
  summation orders and produce two different Float64 histograms. `1` is the
  off switch a bisection wants.
- `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`: the amortization ratio the derived
  block count comes from, `N` or `N/D`, default `8` (which is `8/1`). It sets
  `row_block_min_rows(bins) = floor(2 * N * bins / D)`, so `8` is 4,080 rows
  a block at 255 bins and `1/8` is 63. **This one moves bits too**, and by
  exactly the same mechanism: it changes the block count, and a block count
  is a summation order. It is the *rule* where `MOJOTREES_CPU_ROW_BLOCKS` is
  the *answer*, which is what a sweep across node sizes needs, since one
  forced count is right at one node size and wrong at every other. Unset it
  is byte-identical to what shipped. An unparsable or non-positive value is
  refused at the raising planner entry points rather than silently meaning
  the default, because an arm recorded under the wrong label is worse than an
  arm that did not run.

`MOJOTREES_NUM_WORKERS` and `MOJOTREES_PARALLEL_MIN_OPS` keep their meanings
from `parallel.mojo` and override everything here: an explicit worker count
still bypasses the grain floor and the core cap.

Resolving the environment once
------------------------------
Every function above reads its variable with `getenv` at the moment it is
asked, and `cpu_profile()` re-detects the machine on every call. That is one
`getenv` per question and three core-count queries per detection, and the
tree grower asks these questions once per dispatch, several times per node,
tens of thousands of times per fit. `ResolvedCpuPolicy.resolve()` answers all
of them once and hands back a value that can be carried for the length of a
fit; `parallel.DispatchSettings` wraps it with the two variables that module
owns. Nothing about a plan changes: the resolved forms below compute the same
answers from the same numbers, and the unresolved forms are kept and still
read the environment live, so a test that flips a variable mid-process and
calls the unresolved form sees the flip exactly as it always did.

A resolved value is a snapshot. It does not observe a later `setenv`, and
there is deliberately no global cache and no invalidation hook to forget:
whoever resolves it owns it, and dropping it and resolving again is the
whole of the invalidation story. A caller that must see a mid-process change
either resolves again or uses the unresolved form.

The unresolved sentinel, and why it exists
------------------------------------------
`ResolvedCpuPolicy.unresolved()` is a policy that says "I am not a snapshot".
Every resolved method on it, and `derive_accumulation_plan_with` given it,
falls back to reading the environment live, so it answers exactly what the
unresolved free functions answer.

It exists so that a threading parameter can be *defaulted*. The call chain
from the booster down to a histogram build passes through modules that
different lanes and different campaigns own, and a snapshot parameter with no
default would force every one of those call sites to change at once. With the
sentinel as the default, a caller that has a snapshot passes it and reads
nothing; a caller that has not been wired yet passes nothing and behaves
exactly as it did before, down to which `getenv` runs. That is deliberately a
staging device and not a destination: a hot call site left on the sentinel is
still paying per-dispatch, and the point of the parameter is to be filled in.

The sentinel carries a zeroed profile rather than a detected one, because
detecting the machine to build a value nobody will consult would be the cost
this whole mechanism exists to remove. Nothing may read those zeros: every
method checks `resolved` first and re-reads the machine when it is False.
"""

from std.sys.info import (
    CompilationTarget,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    simd_width_of,
)
from std.os import getenv

# Auto-mode fan-out per core, applied on top of the grain rule in
# `parallel.plan_tasks`. Above 1 the extra tasks absorb the jitter of unequal
# per-item cost (features differ in bin occupancy, blocks in cache behaviour)
# and give a performance core somewhere to go when an efficiency core is
# still working; far above it the scheduling events cost more than the
# imbalance they hide. 4 is a starting point, not a measured optimum.
comptime DEFAULT_TASKS_PER_CORE = 4

# Core pools. `all` is the default and is what this repository has always
# used; `performance` exists so a benchmark can measure whether excluding the
# efficiency cores from a barrier-synchronized fan-out is worth the cores it
# gives up. Neither pins anything.
comptime CORE_POOL_ALL = 0
comptime CORE_POOL_PERFORMANCE = 1

# How many features one accumulation inner loop interleaves. N independent
# features give the out-of-order engine N independent read-modify-write chains
# into N disjoint histogram slices, which is what hides the store-to-load
# latency of consecutive rows landing in the same bin, and -- the reason this
# ladder exists at all -- it divides by N the number of times the node's
# gradient and hessian stream is walked. Feature-major accumulation reads that
# stream once per group, so at 50 features and 1M rows a width of 1 streams
# 50 * 16 MB = 800 MB of gradient traffic per node histogram against the
# 50 MB of binned data it actually needs; a width of 4 streams 208 MB.
# That ratio is why LightGBM's `force_row_wise` builder reads each row's
# gradient exactly once (`src/io/multi_val_dense_bin.hpp`), and widening this
# is the cheapest available probe of how much of the gap that accounts for.
#
# The rungs are 1, 2, 4, 8, 16, matching the GPU histogram family's ladder in
# `gpu_tiling.HIST_FEATURE_GROUP_LADDER` for the same reason: the accumulation
# kernel is instantiated once per width, and a width with no instantiation
# cannot run. `MAX_FEATURE_GROUP` is the top of the ladder, not a policy
# ceiling; `plan_feature_group` is where the width is actually decided, and it
# is bounded by the L1 estimate rather than by a hand-set number.
#
# Nothing here is measured. Wider is strictly less gradient traffic and
# strictly more live histogram slices and register pressure, and where those
# two cross on any particular core is a benchmark's answer, not this comment's.
#
# And wider is NOT free even before the hardware is consulted, which is the
# correction that produced `TASK_BALANCE_FACTOR` below. A group is the dispatch
# unit, so widening divides the task count by exactly the factor it divides the
# traffic by. Byte counts cannot see that, and a plan chosen from byte counts
# alone would have shipped 13 tasks on a 10-core machine here.
comptime FEATURE_GROUP_LADDER = 5
"""Rungs in the feature-group ladder, counting 1."""

comptime MAX_FEATURE_GROUP = 16
comptime DEFAULT_FEATURE_GROUP = MAX_FEATURE_GROUP
"""What the policy asks for before the clamps: the widest rung there is. The
L1 estimate, the core count, and the active feature count are what actually
choose, which is the point of deriving a width instead of naming one."""

# Dispatch units per core the resolved width must leave, when the feature
# count allows it. The balance rule, and the reason width is not chosen from
# memory traffic alone.
#
# **A unit is a `(block, group)` pair.** It was a group, back when the feature
# partition was the only axis; row blocking made the accumulation dispatch over
# `row_blocks * group_count` units and the rule was not revisited, so it went
# on demanding `TASK_BALANCE_FACTOR * cores` *groups* from an axis that no
# longer had to supply them alone. On 50 features and 14 cores that demanded 28
# groups from 50 features and returned width 1: no interleaving at all, on the
# default path, and worse on a bigger machine than on a smaller one. The rule
# now divides the demand by the block count first.
#
# The arithmetic. `T` equal-cost tasks over `C` cores under a barrier finish
# in `ceil(T / C)` rounds, so utilization is `T / (C * ceil(T / C))`. That is
# 100% only when C divides T, and over all `T >= F * C` its worst case is at
# `T = F * C + 1`, one task past a whole number of rounds, giving
# `(F * C + 1) / ((F + 1) * C)`. On ten cores that is 70% at F = 2, 77% at
# F = 3 and 82% at F = 4; as C grows it tends to `F / (F + 1)`. So the factor
# buys a floor on how much of the machine the last round can waste, and the
# floor is not steep: going from 2 to 4 buys twelve points of guarantee.
#
# Why 2 and not 4. The floor is a guarantee, not the typical case, and at the
# shape this lane exists for it is not what binds: 50 features on 10 cores at
# width 2 is 25 groups, 3 rounds, 25/30 = 83%. Raising the factor to 4 would
# demand 40 units and force width 1 on an unblocked node of that shape, which
# is to say it would revert the lane on the only shape it was written for. 3
# would demand 30 and force width 1 as well. 2 is therefore the largest factor
# that leaves any interleave at all at 50 features on one block, and that is
# the whole of its derivation: it is a boundary, not an optimum.
#
# That derivation is now much less load-bearing than it was. Once the demand is
# counted in `(block, group)` units, a node that blocks 49 ways satisfies even
# a factor of 4 at the widest rung, so the factor stops choosing the width on
# any node large enough to block and the L1 estimate chooses instead. It still
# decides on nodes too small to block, which is most nodes deep in a tree.
#
# It is unmeasured in both directions. Two things it does not model: cores here
# are not equal (a task on an efficiency core sets the pace of its round, so
# the real cost of a partial round is worse than the fraction suggests), and
# the accumulation may be memory-bound well before every core is busy (in which
# case idle cores cost less than this rule prices them at). Only an interleaved
# wall-time A/B settles it. `MOJOTREES_CPU_FEATURE_GROUP` is how it is swept,
# and `bench/apple/cpu_plan.json` states the comparison that would move it.
comptime TASK_BALANCE_FACTOR = 2

# Row blocking: the second axis of the accumulation decomposition.
#
# The feature group is the first axis and it has a hard ceiling. At width `W`
# over `A` active features the dispatch has `ceil(A / W)` units and no more,
# which at the default shape (50 features, width 2) is 25 whatever the node's
# row count is. That ceiling does not fall with node size either, so a node
# with a thousandth of the rows still pays 25 scheduling events. Splitting a
# node's rows into blocks and giving each block a private histogram adds an
# independent axis: the unit count becomes `blocks * groups`, and it grows
# with the row count instead of being capped by the feature count.
#
# **The block count is a pure function of the workload shape.** Rows, bins,
# and active features go in; nothing about the machine, the core count, the
# core pool, `MOJOTREES_NUM_WORKERS`, or the task count is consulted. That is
# not a stylistic choice. A block fold sums `(block 0's rows) + (block 1's
# rows) + ...` and Float64 addition is not associative, so the block count is
# part of the *value*, not part of the schedule. A block count that varied
# with the worker count would make the output vary with the worker count,
# which is the one thing this round will not trade. Everything else about the
# dispatch -- how many tasks the units are cut into, which core runs which
# unit -- stays free to vary, because a unit is never split and the fold is
# always in ascending block order.
#
# What the constants below bound.
#
# `ROW_BLOCK_AMORTIZE` is the ratio of accumulate work to per-block overhead
# a block must clear. One block costs `active * bins` cells zeroed and
# `active * bins` cells read back in the fold, and earns `block_rows *
# active` accumulates, so requiring
#
#     block_rows * active >= ROW_BLOCK_AMORTIZE * 2 * active * bins
#
# gives `block_rows >= 2 * ROW_BLOCK_AMORTIZE * bins`, which is
# `row_block_min_rows`. The `active` cancels, which is why the floor is
# stated in bins alone. At 255 bins and a ratio of 8 that is 4,080 rows per
# block, so a node needs 8,160 rows before it blocks at all, and the
# zero-and-fold overhead is at most an eighth of the accumulation it buys.
# The ratio is not measured; it is the fraction of the work this decomposition
# is allowed to spend on itself.
#
# `MAX_ROW_BLOCK_SCRATCH_BYTES` bounds the private histograms. Blocks hold
# `blocks * active * bins` cells of three Float64 planes, which at 50
# features and 255 bins is 306 KB per block, so this cap is what stops a
# million-row node from asking for 245 blocks and 75 MB. It is a byte budget
# rather than a block count because the shape that matters is the product.
# `MAX_ROW_BLOCKS` is the absolute ceiling underneath it, for a narrow shape
# whose cells are cheap enough that the byte budget would allow thousands.
# `ROW_BLOCK_AMORTIZE` is a *ratio*, and the two constants below are its
# numerator and denominator so that the ratio can be swept below 1 without the
# derivation losing its units. The default is 8/1, which is exactly the 8 this
# repository shipped, so `row_block_min_rows(255)` is 4,080 unchanged.
#
# Why sub-unit values have to be reachable. LightGBM's comparable number is
# `min_block_size = min(0.3 * num_bin / elems_per_row + 1, 1024)` floored at
# 32, which at 255 bins is 77 rows, 53 times smaller than ours. That is not
# the same quantity -- theirs is a floor on a *work-stealing chunk* whose
# private histogram count comes from the thread count, ours is the private
# histogram count itself -- so it cannot be adopted, because a block count
# that came from the thread count would make the output depend on
# `MOJOTREES_NUM_WORKERS`. But it does say that nobody has established which
# side of 4,080 the answer is on, and a knob that bottomed out at 510 rows
# (ratio 1) could not even reach the region the question is about.
# `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE` therefore accepts `N` and `N/D`, so
# `1/8` is a 63-row floor at 255 bins and the whole span is sweepable.
#
# The ratio remains a statement about work only. Nothing about the machine
# enters it, so every setting still gives a block count that is a function of
# `rows`, `n_bins` and `n_active` alone.
comptime ROW_BLOCK_AMORTIZE = 8
comptime ROW_BLOCK_AMORTIZE_DEN = 1
comptime MAX_ROW_BLOCKS = 64
comptime MAX_ROW_BLOCK_SCRATCH_BYTES = 16 * 1024 * 1024

# Planes and bytes per plane in the private block scratch. Three planes:
# gradient, hessian, and count. The count plane is carried as Float64 rather
# than as `Int` because the scratch is one `List[Float64]` (the caller's
# gradient/hessian gather buffer, whose tail this borrows), and a count is an
# integer below 2^53 at every step of the fold, so every partial sum is
# exactly representable and the conversion back to `Int` is exact. The
# hessian plane is allocated even under the constant-hessian specialization,
# which does not write it: sizing the scratch from `const_hessian` would make
# the *block count* depend on it through the byte budget, and the block count
# is part of the value.
comptime ROW_BLOCK_PLANES = 3
comptime ROW_BLOCK_PLANE_BYTES = 8

# How a private block cell is *laid out*, which is a different question from
# how much room it is allocated. `ROW_BLOCK_PLANES` above is the allocation
# and the byte budget, and it stays at 3 unconditionally so that the block
# count remains a value independent of `const_hessian` (the paragraph above
# says why that matters). These two say how many of those three slots the
# kernel actually addresses, and that they are addressed *interleaved*:
# cell `c` occupies `[c * stride, c * stride + stride)` rather than living on
# three planes a megabyte apart.
#
# This is LightGBM's `hist_t` layout. `include/LightGBM/bin.h` declares
# `typedef double hist_t`, `kHistEntrySize = 2 * sizeof(hist_t)` and
# `GET_GRAD(hist, i) = hist[(i) << 1]`, `GET_HESS(hist, i) = hist[((i) << 1)
# + 1]`: one 16-byte (gradient, hessian) cell, no count plane. Its
# constant-hessian arm keeps the same 16 bytes and spends the second slot on
# a count (`hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(hess)` in
# `src/io/dense_bin.hpp`), which is the arm `ROW_BLOCK_CELL_FLOATS_CONST_H`
# names.
#
# The general arm keeps a third slot for an exact count where LightGBM
# derives one from the hessian. That divergence is deliberate and is argued
# at `histogram.derived_count`: derived counts are exact only when every
# hessian is `CONSTANT_HESSIAN`, and mojotrees exposes `Histogram.count_at`
# to a distributed integer allreduce and to the C ABI, neither of which can
# take a rounded count. Both arms still get the property the interleave is
# for -- one cache line touched per (row, feature) instead of three.
comptime ROW_BLOCK_CELL_FLOATS = 3
comptime ROW_BLOCK_CELL_FLOATS_CONST_H = 2

# LightGBM's cell, for the arithmetic in reports and for the one place the
# constant-hessian arm's stride is compared against it.
comptime LGBM_HIST_ENTRY_FLOATS = 2
comptime LGBM_HIST_ENTRY_BYTES = 16

# One gathered per-row derivative pair, `(gradient, hessian)` as two Float32
# rather than two Float64. LightGBM's `score_t` is `float`
# (`include/LightGBM/meta.h`), so its `ordered_gradients` / `ordered_hessians`
# are 4 bytes per row per plane and this is the same 8 bytes per row. The
# buffer they live in is still typed `List[Float64]`, because its type is
# fixed by a signature `tree.mojo` owns; two Float32 per Float64 word is how
# the narrowing is taken without that signature moving. See
# `histogram._gather_pairs`.
comptime SCORE_T_BYTES = 4
comptime GATHERED_PAIR_BYTES = 2 * SCORE_T_BYTES

# One binned value, `UInt8` in `binning.BinnedMatrix`.
comptime BINNED_VALUE_BYTES = 1

# How far ahead the scatter loop issues its software prefetch, in rows.
# LightGBM's `DenseBin::ConstructHistogramInner` uses `const data_size_t
# pf_offset = 64 / sizeof(VAL_T)` and prefetches `data_[data_indices[i +
# pf_offset]]`, so at one byte per binned value this is 64 rows. A fixed 64
# rather than `ASSUMED_CACHE_LINE_BYTES` on purpose: through a row-id list
# the address is scattered, so the distance is buying latency cover and not
# line coverage, and the number that has been tuned against real hardware is
# LightGBM's.
comptime PREFETCH_ROW_DISTANCE = 64 // BINNED_VALUE_BYTES

# There is deliberately no unroll constant here. An earlier version of this
# lane's brief called for a four-row unroll of the scatter loop alongside the
# prefetch, on the understanding that LightGBM had one.
# `DenseBin::ConstructHistogramInner` does not: its prefetching loop and its
# tail loop are both scalar, one row per iteration. The unroll was withdrawn
# rather than kept as an unattributed invention.


# Below this many rows a node's gradients and hessians are small enough to
# stay in cache across every feature's pass, so gathering them into a
# contiguous scratch buffer first would add a pass without removing any
# misses. Above it the gather is paid once instead of once per feature.
comptime DEFAULT_COMPACT_MIN_ROWS = 256

# Gathering is only worth a pass when more than one feature will read the
# result; with a single active feature the gather reads exactly what the
# accumulation would have read.
comptime COMPACT_MIN_FEATURES = 2

# One host histogram cell: Float64 gradient + Float64 hessian + Int count per
# (feature, bin), the layout of `Histogram` in histogram.mojo on a 64-bit
# target. This is the one definition.
comptime HISTOGRAM_BYTES_PER_CELL = 24

# Assumed, not reported. Apple silicon uses a 128-byte line; 64 is the
# portable floor elsewhere. Only used for locality arithmetic and reporting,
# never for correctness.
comptime ARM_CACHE_LINE_BYTES = 128
comptime GENERIC_CACHE_LINE_BYTES = 64

# Resolved once at compile time rather than queried per call: it is a
# property of the target, it costs nothing to fold, and it keeps every
# function below non-raising, which is what lets `parallel.plan_tasks` stay
# non-raising for its non-raising callers.
comptime TARGET_HAS_NEON = CompilationTarget.has_neon()
comptime ASSUMED_CACHE_LINE_BYTES = (
    ARM_CACHE_LINE_BYTES if TARGET_HAS_NEON else GENERIC_CACHE_LINE_BYTES
)

# Assumed, not reported: a conservative L1 data cache floor. Apple's
# efficiency cores have the smaller L1 of the two core types, and this is
# meant to be at or below it, so a group sized against it stays resident on
# whichever core type the task lands on.
comptime ASSUMED_L1D_BYTES = 64 * 1024

# Fraction of the assumed L1 an interleaved feature group may claim. The rest
# is left for the row index stream, the gradient pairs, and the binned column
# the loop is walking, all of which are streaming through the same cache.
comptime L1_GROUP_DIVISOR = 2


def _env_int(name: String, default: Int) -> Int:
    """Deliberately a private copy of `parallel._env_int` rather than an
    import of it: `parallel.mojo` imports this module, and a policy that
    imported back would make the two mutually dependent. Ten lines of
    duplication buys a one-way dependency."""
    var s = getenv(name)
    if s.byte_length() == 0:
        return default
    try:
        var n = Int(s)
        if n < 0:
            return default
        return n
    except:
        return default


def _positive_or(value: Int, default: Int) -> Int:
    return value if value > 0 else default


def env_tasks_per_core() -> Int:
    """Auto-mode fan-out per core; 0 or unparsable means the default."""
    return _positive_or(
        _env_int("MOJOTREES_CPU_TASKS_PER_CORE", DEFAULT_TASKS_PER_CORE),
        DEFAULT_TASKS_PER_CORE,
    )


def env_core_pool() -> Int:
    """Which cores auto mode COUNTS. Anything unrecognized means `all`, so a
    typo loses the experiment rather than the parallelism.

    **Counts, not runs on, and the difference decides what an experiment with
    this variable is allowed to conclude.** The only consumer is
    `dispatch_cores`, which feeds `max_auto_tasks`, so `performance` lowers
    the task ceiling from `tasks_per_core * physical` to
    `tasks_per_core * performance` and does nothing else. It sets no
    affinity, no QoS class and no pinning; this module's header states that
    none is used and none is assumed to exist. The scheduler is still free to
    place any of the remaining tasks on an efficiency core.

    So a `performance` arm answers "does a smaller fan-out run faster", which
    is a real question and the one the header poses. It does NOT answer "is
    the efficiency cluster what makes ten workers slower than four", because
    the arm cannot keep work off that cluster. Reading the first result as an
    answer to the second is the failure this docstring exists to prevent."""
    var s = getenv("MOJOTREES_CPU_CORE_POOL")
    if s == "performance" or s == "PERFORMANCE" or s == "p":
        return CORE_POOL_PERFORMANCE
    return CORE_POOL_ALL


def is_feature_group_width(group: Int) -> Bool:
    """Whether `group` is a rung of the ladder, so `histogram.mojo` has an
    accumulation kernel instantiated at it. 3 is not, and is refused rather
    than rounded, for the same reason `gpu_tiling.is_feature_group_width`
    refuses it: a width with no instantiation cannot run, and silently running
    a different one loses whatever experiment asked for it."""
    var rung = 1
    for _ in range(FEATURE_GROUP_LADDER):
        if rung == group:
            return True
        rung *= 2
    return False


def feature_group_floor(n: Int) -> Int:
    """The widest rung at or below `n`, never below 1.

    Rounding *down* is the only safe direction for every clamp below: a rung
    above the bound would violate the bound the clamp was expressing, while a
    rung below it merely leaves some of the budget unused. `feature_group_floor(5)`
    is 4, which is what the L1 estimate at 255 bins comes to.
    """
    var best = 1
    var rung = 1
    for _ in range(FEATURE_GROUP_LADDER):
        if rung <= n:
            best = rung
        rung *= 2
    return best


def feature_group_count(n_active: Int, group: Int) -> Int:
    """How many groups `n_active` features split into at this width.

    The dispatch unit of a histogram build. Stated here rather than derived at
    each call site because it is the number that has to match between the
    policy, the kernel's group loop, and the task split: a task holds whole
    groups, so a group is the smallest thing that can be scheduled, and the
    last group owning fewer than `group` features is the ragged tail the
    kernel handles with the same code path as a full one.
    """
    if n_active <= 0 or group <= 0:
        return 0
    return (n_active + group - 1) // group


def env_feature_group() raises -> Int:
    """The requested interleave width, or 0 when nothing is requested.

    Zero rather than `DEFAULT_FEATURE_GROUP` because the two answers mean
    different things downstream: an explicit request bypasses the derived
    clamps (it exists to test the estimate those clamps encode) while the
    default does not.

    Raises on anything that is not a rung. That includes an unparsable value,
    which the other knobs in this file quietly ignore. The difference is
    deliberate and narrow: the others are scheduling numbers where any value
    produces some valid plan, whereas this one names a kernel instantiation,
    and a benchmark arm that asked for 3 and silently got 2 has recorded a
    result under the wrong label. That has happened in this repository before,
    with the GPU split strategy, and `gpu_active_rows.set_feature_group`
    refuses for the same reason.
    """
    var s = getenv("MOJOTREES_CPU_FEATURE_GROUP")
    if s.byte_length() == 0:
        return 0
    try:
        var n = Int(s)
        if not is_feature_group_width(n):
            raise Error("off the ladder")
        return n
    except:
        raise Error(
            'MOJOTREES_CPU_FEATURE_GROUP must be 1, 2, 4, 8, or 16: those are'
            ' the widths an accumulation kernel is instantiated at. Got "',
            s,
            '"',
        )


def env_compact_min_rows() -> Int:
    return _env_int("MOJOTREES_CPU_COMPACT_MIN_ROWS", DEFAULT_COMPACT_MIN_ROWS)


def env_row_blocks() -> Int:
    """The requested row-block count, or 0 when nothing is requested.

    Zero rather than a default, on the same grounds as `env_feature_group`:
    an explicit request means something different from the derived answer. It
    bypasses the amortization floor and the byte budget, because those are
    exactly the estimates the knob exists to test, and it is still clamped by
    the node's row count (a block cannot be shorter than a row) and by
    `MAX_ROW_BLOCKS`.

    **This knob moves bits, and it is one of the two in this file that do**
    (`MOJOTREES_CPU_ROW_BLOCK_AMORTIZE` is the other, and reaches the same
    block count through the rule rather than through the answer).
    Every other variable here is a scheduling decision that leaves the output
    identical. The block count is a summation order, so `1` and `4` produce
    two different Float64 histograms -- neither less accurate than the other,
    see `histogram.mojo`'s module docstring -- and a fit that sets it is not
    comparable cell for cell with a fit that does not. `1` is the off switch
    and reproduces the pre-blocking accumulation exactly, which is what a
    bisection wants; `0` and unset both mean "derive".
    """
    return _env_int("MOJOTREES_CPU_ROW_BLOCKS", 0)


@fieldwise_init
struct RowBlockAmortize(Copyable, Movable):
    """The amortization ratio, as a rational and a note on where it came from.

    Two integers rather than one because the interesting half of the sweep is
    below 1: the default 8 puts 4,080 rows in a block at 255 bins, and the
    question this exists to settle is whether that is roughly right or fifty
    times too conservative, which is a ratio near 1/6. An integer-only knob
    could not express the answer, so it could not ask the question.

    `recognized` is False only for a value the parser refused. The value it
    then carries is the default, so a non-raising caller still plans something
    sane, and `check_row_block_amortize` is what turns the refusal into an
    error at the raising entry points. That split exists because
    `row_block_min_rows` and `plan_row_block_count` are non-raising public
    functions and adding `raises` to them would propagate into callers this
    lane does not own -- while a benchmark arm that asked for `1/8`, was
    silently given 8, and got recorded under the wrong label is the exact
    failure this project has already thrown two results away for.
    """

    var num: Int
    var den: Int

    var recognized: Bool
    """False when the environment held something this parser refused."""

    @staticmethod
    def default() -> RowBlockAmortize:
        """8/1, the shipped ratio."""
        return RowBlockAmortize(ROW_BLOCK_AMORTIZE, ROW_BLOCK_AMORTIZE_DEN, True)

    def describe(self) -> String:
        if self.den == 1:
            return String(self.num)
        return String(self.num, "/", self.den)


def parse_row_block_amortize(spec: String) -> RowBlockAmortize:
    """`N` or `N/D` into a ratio; anything else is refused.

    Empty means the default and is *recognized*, since unset is a valid state
    and the whole point of the default is that it is byte-identical to today.
    A zero or negative numerator or denominator is refused rather than
    clamped: `0` would mean an infinitely small block and `-1` means the
    caller mistyped, and neither is an arm anybody meant to run.
    """
    if spec.byte_length() == 0:
        return RowBlockAmortize.default()
    var slash = spec.find("/")
    var num_text: String
    var den_text: String
    if slash < 0:
        num_text = spec.copy()
        den_text = String("1")
    else:
        num_text = String(spec[byte=0:slash])
        den_text = String(spec[byte = slash + 1 :])
    try:
        var num = Int(num_text)
        var den = Int(den_text)
        if num <= 0 or den <= 0:
            return RowBlockAmortize(
                ROW_BLOCK_AMORTIZE, ROW_BLOCK_AMORTIZE_DEN, False
            )
        return RowBlockAmortize(num, den, True)
    except:
        return RowBlockAmortize(
            ROW_BLOCK_AMORTIZE, ROW_BLOCK_AMORTIZE_DEN, False
        )


def env_row_block_amortize() -> RowBlockAmortize:
    """`MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`, or the shipped 8/1 when unset.

    **This moves bits**, for the same reason `MOJOTREES_CPU_ROW_BLOCKS` does
    and by the same mechanism: it changes the block count, a block count is a
    summation order, and Float64 addition is not associative. It is a second
    spelling of that one knob rather than a new kind of thing -- `ROW_BLOCKS`
    names a count directly, this names the rule that derives one -- and the
    rule is what a sweep across node sizes actually wants, since a single
    forced count is wrong at every node size but one.

    What it emphatically does not do is admit a machine fact. The ratio goes
    into `row_block_min_rows_at`, which sees `n_bins` and nothing else, and
    the count it produces is still a function of `rows`, `n_bins` and
    `n_active`. There is no value of this variable that makes the block count
    depend on the worker count, the task count, or the core count, and
    `test_row_block_size.mojo` asserts that at every swept setting.
    """
    return parse_row_block_amortize(getenv("MOJOTREES_CPU_ROW_BLOCK_AMORTIZE"))


def checked_row_block_amortize() raises -> RowBlockAmortize:
    """`env_row_block_amortize`, raising if the value was refused.

    Called from the raising planner entry points, so a mistyped arm fails at
    the first histogram plan of the fit rather than running the default under
    the wrong label. `env_feature_group` refuses in the same spirit; the only
    difference is that this one refuses from a separate function, because the
    arithmetic helpers it feeds must stay non-raising.

    It returns the ratio rather than only checking it, so the raising path
    reads the environment once rather than twice.
    """
    var a = env_row_block_amortize()
    if a.recognized:
        return a^
    raise Error(
        "MOJOTREES_CPU_ROW_BLOCK_AMORTIZE must be a positive integer `N` or a"
        ' positive ratio `N/D`, e.g. "8" (the default), "1", or "1/8". Got "',
        getenv("MOJOTREES_CPU_ROW_BLOCK_AMORTIZE"),
        '"',
    )


def row_block_min_rows_at(amortize: RowBlockAmortize, n_bins: Int) -> Int:
    """`row_block_min_rows` with the ratio already read.

    `floor(2 * num * n_bins / den)`, derived in the constant's comment: a
    block zeroes `active * n_bins` cells and the fold reads them back, against
    `block_rows * active` accumulates, and `active` cancels out of the ratio.
    Never below 1, so a degenerate `n_bins` cannot divide by zero downstream.

    Floor and not round, because the floor is the direction that makes the
    knob monotone: a smaller ratio must never produce a larger block.
    """
    if n_bins <= 0:
        return 1
    var den = amortize.den if amortize.den > 0 else 1
    var num = amortize.num if amortize.num > 0 else 1
    var m = (2 * num * n_bins) // den
    return m if m > 0 else 1


def row_block_min_rows(n_bins: Int) -> Int:
    """Fewest rows a block must hold to be worth its private histogram.

    The live-reading form: it consults `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE` at
    the moment it is asked, exactly as the other unresolved free functions in
    this module consult theirs. Unset, this is `2 * ROW_BLOCK_AMORTIZE *
    n_bins` and therefore 4,080 at 255 bins, unchanged.
    """
    return row_block_min_rows_at(env_row_block_amortize(), n_bins)


def max_row_blocks_for_cells(cells: Int) -> Int:
    """Blocks the scratch budget allows for a private histogram of `cells`
    cells, never below 1 and never above `MAX_ROW_BLOCKS`.

    `cells` is `active * n_bins`, the compact active-slice shape the blocked
    kernel keeps its partials in -- not `n_features * n_bins`, since an
    excluded feature's slice is never accumulated and never folded.
    """
    if cells <= 0:
        return 1
    var per_block = cells * ROW_BLOCK_PLANES * ROW_BLOCK_PLANE_BYTES
    var by_bytes = MAX_ROW_BLOCK_SCRATCH_BYTES // per_block
    if by_bytes < 1:
        by_bytes = 1
    return by_bytes if by_bytes < MAX_ROW_BLOCKS else MAX_ROW_BLOCKS


def plan_row_block_count(
    requested: Int,
    rows: Int,
    n_bins: Int,
    n_active: Int,
    rows_are_indirect: Bool,
) -> Int:
    """How many row blocks one accumulation splits its node into.

    1 means no blocking: the accumulation runs the feature-partition kernel
    that shipped, cell for cell as it always did. Anything above 1 selects the
    blocked kernel and is therefore a statement about the *value*, so read the
    argument list twice: `rows`, `n_bins`, `n_active`, `rows_are_indirect`,
    and an explicit request. No core count, no worker count, no task count,
    no machine. That is what makes the result independent of
    `MOJOTREES_NUM_WORKERS` and identical on every machine on this toolchain.

    `rows_are_indirect` no longer reaches this decision, and the parameter is
    kept only because it is part of the plan's argument list. **Both builders
    block, on this one rule.** The full-dataset builder used to be excluded
    here, on the grounds that it has no caller-owned scratch and would have to
    allocate the private histograms per call. That was a cost argument against
    a *correctness* property: while only one builder blocked, the two
    disagreed by a few ulp on the same rows, which made "growing on a bag is
    growing on the dataset of those rows" false and was measured as a
    four-ulp leaf value in `test_bagged_tree_equals_tree_on_subset_dataset`.
    The cost is answered instead of accepted:
    `histogram.build_histogram_into_scratch` lets a caller hand its buffer in,
    and the full builder is reached once per tree rather than once per node.
    """
    return plan_row_block_count_at(
        env_row_block_amortize(),
        requested,
        rows,
        n_bins,
        n_active,
        rows_are_indirect,
    )


def plan_row_block_count_at(
    amortize: RowBlockAmortize,
    requested: Int,
    rows: Int,
    n_bins: Int,
    n_active: Int,
    rows_are_indirect: Bool,
) -> Int:
    """`plan_row_block_count` with the amortization ratio already read.

    The one copy of the rule. Read the arguments once more: a ratio, a
    request, and three numbers about the workload. Still no core count, no
    worker count, no task count, no machine, at any setting of the ratio --
    which is the property that has to survive this knob existing, since a
    knob that could make the block count machine-dependent would have traded
    away determinism across `MOJOTREES_NUM_WORKERS` to ask a question about
    block sizes.
    """
    if rows <= 0 or n_bins <= 0 or n_active <= 0:
        return 1
    var ceiling = max_row_blocks_for_cells(n_active * n_bins)
    var blocks: Int
    if requested > 0:
        blocks = requested
        if blocks > MAX_ROW_BLOCKS:
            blocks = MAX_ROW_BLOCKS
    else:
        blocks = rows // row_block_min_rows_at(amortize, n_bins)
        if blocks > ceiling:
            blocks = ceiling
    if blocks > rows:
        blocks = rows
    if blocks < 2:
        return 1
    return blocks


def row_block_geometry(rows: Int, blocks: Int) -> Int:
    """Rows per block for `blocks` contiguous ascending blocks over `rows`.

    Ceiling division, exactly as `parallel._blocks_for` does it, so a caller
    that recounts `ceil(rows / chunk)` gets a block count with no empty
    trailing block. The recount is `row_block_count_from_chunk`.
    """
    if blocks <= 1 or rows <= 0:
        return rows if rows > 0 else 1
    return (rows + blocks - 1) // blocks


def row_block_count_from_chunk(rows: Int, chunk: Int) -> Int:
    """Blocks of `chunk` rows that `rows` actually needs.

    Ceiling division can leave a trailing block empty (10 rows over 4 blocks
    gives a chunk of 3 and a fourth block with nothing in it), and an empty
    block is a zeroed private histogram that the fold then reads back. Recount
    so every block has work.
    """
    if rows <= 0 or chunk <= 0:
        return 1
    return (rows + chunk - 1) // chunk


def core_pool_name(pool: Int) -> String:
    if pool == CORE_POOL_PERFORMANCE:
        return String("performance")
    return String("all")


def _parse_first_two_ints(text: String) -> Tuple[Int, Int]:
    """The first two whitespace-separated integers, or (0, 0)."""
    var vals = List[Int]()
    var cur = 0
    var have = False
    for i in range(text.byte_length()):
        var b = Int(text.as_bytes()[i])
        if b >= 48 and b <= 57:
            cur = cur * 10 + (b - 48)
            have = True
        else:
            if have:
                vals.append(cur)
                if len(vals) == 2:
                    return (vals[0], vals[1])
                cur = 0
                have = False
    if have:
        vals.append(cur)
    if len(vals) >= 2:
        return (vals[0], vals[1])
    if len(vals) == 1:
        return (vals[0], 0)
    return (0, 0)


def cgroup_cpu_limit() -> Int:
    """CPUs this process is actually allowed, or 0 when there is no limit.

    **NOTHING IN THIS REPOSITORY READ A CGROUP QUOTA BEFORE 2026-08-18, AND
    THAT IS A DEFECT ON EVERY CONTAINER RATHER THAN AN EXOTIC ONE.**
    `num_physical_cores()` reports the HOST's topology. A cgroup limit is
    enforced by the scheduler and is invisible to it. So in Docker, in
    Kubernetes, on GitHub Actions, and on any customer cluster with a CPU
    limit, every count derived from that number describes a machine we do not
    have.

    It was found on an NVIDIA container where `nproc` returned **256** against
    a quota of **27.2 CPUs**, a factor of nine. Every task count, every grain
    size, and every crossover threshold in this file was being computed against
    that 256.

    Cgroup v2 puts `"<quota> <period>"` in `/sys/fs/cgroup/cpu.max`, with the
    literal `max` for no limit. v1 splits it across `cpu.cfs_quota_us` and
    `cpu.cfs_period_us`, with `-1` for no limit. Both are microseconds, so the
    allowance is `quota / period`.

    **Rounds DOWN, and never below 1.** A quota of 27.2 CPUs becomes 27 rather
    than 28, because rounding up asks the scheduler for more than it will give
    and the whole point is to stop oversubscribing. A quota under one full CPU
    still gets 1, since a pool of zero cannot run anything.

    Returns 0 on macOS, on any unlimited cgroup, on a missing or unreadable
    file, and on anything it cannot parse. **0 means "no opinion" and the
    caller keeps the topology**, which is the right failure: a misparse must
    not silently shrink a real machine's pool to nothing.

    WHAT THIS DOES NOT DO. It does not resize MAX's own runtime pool, which
    `sync_parallelize` dispatches into and which this package does not own.
    `docs/design/DECLINED_OPTIMIZATIONS.md` F6 measured that pool converting
    about 3.5x on a 10-core M4 and FLAT from 4 to 16 tasks, with six
    environment variables changing nothing. So this fixes OUR arithmetic and
    the question of whether `MOJOTREES_NUM_WORKERS` can reach MAX's pool at all
    is separate and still open.
    """
    # v2 first: one file, and it is the modern layout.
    try:
        var v2 = open("/sys/fs/cgroup/cpu.max", "r").read()
        var qp = _parse_first_two_ints(v2)
        # `max <period>` parses as (period, 0) because `max` yields no digits,
        # so a zero period is how "unlimited" arrives here.
        if qp[1] > 0 and qp[0] > 0:
            var n = qp[0] // qp[1]
            return 1 if n < 1 else n
        return 0
    except:
        pass
    try:
        var q = _parse_first_two_ints(
            open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r").read()
        )
        var pr = _parse_first_two_ints(
            open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r").read()
        )
        if q[0] > 0 and pr[0] > 0:
            var n = q[0] // pr[0]
            return 1 if n < 1 else n
    except:
        pass
    return 0


@fieldwise_init
struct CpuProfile(Copyable, Movable):
    """Everything the CPU policy is allowed to know about the machine.

    Only the four counts are read from the system. The rest are the assumed
    constants above, carried on the profile so a caller can print exactly
    what a decision was made against.
    """

    var physical_cores: Int
    var logical_cores: Int

    var performance_cores: Int
    """Reported performance cores, sanitized into `[1, physical_cores]`. On a
    symmetric machine this equals `physical_cores`, which is how
    `asymmetric` is decided; no core-type detection beyond that is done."""

    var float64_lanes: Int
    """Compile-time SIMD width for Float64 on this target."""

    var has_neon: Bool
    var cache_line_bytes: Int
    var l1d_bytes: Int


    @staticmethod
    def detect() -> CpuProfile:
        """The running machine, with every count sanitized. A machine that
        reports nonsense (zero cores, more performance cores than physical)
        degrades to the conservative reading rather than producing a plan
        with a zero or negative task count."""
        var physical = _positive_or(num_physical_cores(), 1)
        var logical = _positive_or(num_logical_cores(), physical)
        var perf = _positive_or(num_performance_cores(), physical)
        if perf > physical:
            perf = physical
        # THE CGROUP QUOTA CLAMPS THE TOPOLOGY, and see `cgroup_cpu_limit` for
        # why this was a defect on every container. A limit of 0 means no
        # opinion and leaves all three counts alone. A real limit clamps
        # physical, and then logical and perf are re-clamped against it,
        # because a machine that reports 256 cores behind a 27-CPU quota must
        # not end up with more performance cores than physical ones.
        var allowed = cgroup_cpu_limit()
        if allowed > 0 and allowed < physical:
            physical = allowed
            if logical > physical:
                logical = physical
            if perf > physical:
                perf = physical
        return CpuProfile(
            physical,
            logical,
            perf,
            simd_width_of[DType.float64](),
            TARGET_HAS_NEON,
            ASSUMED_CACHE_LINE_BYTES,
            ASSUMED_L1D_BYTES,
        )

    @staticmethod
    def synthetic(physical_cores: Int, performance_cores: Int) -> CpuProfile:
        """A constructed profile, for exercising the rules below on a machine
        that is not the one being planned for. Never used as a fallback:
        `detect` always reads the running machine. The SIMD width and the
        cache assumptions stay the compiling target's, because those are
        properties of the build rather than of the invented core layout."""
        var physical = _positive_or(physical_cores, 1)
        var perf = _positive_or(performance_cores, physical)
        if perf > physical:
            perf = physical
        return CpuProfile(
            physical,
            physical,
            perf,
            simd_width_of[DType.float64](),
            TARGET_HAS_NEON,
            ASSUMED_CACHE_LINE_BYTES,
            ASSUMED_L1D_BYTES,
        )

    def asymmetric(self) -> Bool:
        """Whether the machine reports fewer performance cores than physical
        cores, which is the only signal used to recognize a big.LITTLE-style
        layout. It is a fact about the report, not a chip identification."""
        return self.performance_cores < self.physical_cores

    def efficiency_cores(self) -> Int:
        return self.physical_cores - self.performance_cores

    def dispatch_cores(self) -> Int:
        """Cores auto mode plans against, under `MOJOTREES_CPU_CORE_POOL`."""
        if env_core_pool() == CORE_POOL_PERFORMANCE:
            return _positive_or(self.performance_cores, 1)
        return _positive_or(self.physical_cores, 1)

    def max_auto_tasks(self) -> Int:
        """Ceiling on the task count auto mode may choose. The grain rule in
        `parallel.plan_tasks` applies on top of this and usually binds
        first."""
        return _positive_or(env_tasks_per_core() * self.dispatch_cores(), 1)

    def describe(self) -> String:
        """One line for benchmark headers and bug reports."""
        var neon = String("yes") if self.has_neon else String("no")
        return String(
            "cores=",
            self.physical_cores,
            " (perf ",
            self.performance_cores,
            ", eff ",
            self.efficiency_cores(),
            ", logical ",
            self.logical_cores,
            ") pool=",
            core_pool_name(env_core_pool()),
            " tasks_per_core=",
            env_tasks_per_core(),
            " max_auto_tasks=",
            self.max_auto_tasks(),
            " f64_lanes=",
            self.float64_lanes,
            " neon=",
            neon,
            " assumed_line=",
            self.cache_line_bytes,
            " assumed_l1d=",
            self.l1d_bytes,
        )


def cpu_profile() -> CpuProfile:
    """The running machine. Cheap enough to call once per histogram build,
    which is where every caller calls it; it is not cached, because this
    project has no once-initialized global facility and a stale cache would
    hide an environment change a benchmark just made."""
    return CpuProfile.detect()


@fieldwise_init
struct ResolvedCpuPolicy(Copyable, Movable):
    """One reading of the machine and of this module's six variables.

    The same six questions `env_tasks_per_core`, `env_core_pool`,
    `env_feature_group`, `env_compact_min_rows`, `env_row_blocks`, and
    `env_row_block_amortize` answer,
    plus one
    `CpuProfile.detect()`, resolved together so a fit pays for them once
    instead of once per dispatch. Every method here is the resolved twin of a
    free function above and computes the identical answer from the identical
    numbers; the difference is only when the environment was read.

    Held by value and copied freely: it is five machine counts and five small
    integers, so passing it down a call chain costs nothing worth measuring
    and no lifetime has to be reasoned about. It is a snapshot, so a
    `setenv` after `resolve()` is not observed by it; that is the documented
    contract rather than a defect, and it is why the unresolved functions are
    still here for anything that wants a live read.
    """

    var profile: CpuProfile
    var tasks_per_core: Int
    var core_pool: Int
    var feature_group: Int
    var compact_min_rows: Int

    var row_blocks: Int
    """The snapshot of `MOJOTREES_CPU_ROW_BLOCKS`, 0 for "derive".

    Carried here for the same reason as the other four -- one read per fit
    instead of one per node -- and with one difference worth stating: this and
    `row_block_amortize` are the two values on this snapshot that can change
    an output. A fit that
    resolves the snapshot and then `setenv`s this variable keeps accumulating
    at the resolved block count, which is the documented snapshot contract and
    is also the safer of the two behaviours: a block count that changed
    mid-fit would move bits between one node and the next."""

    var row_block_amortize: RowBlockAmortize
    """The snapshot of `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`, default 8/1.

    The second value on this snapshot that can change an output, and the same
    snapshot contract applies: a `setenv` after `resolve()` is not observed,
    so the block count cannot move between one node of a fit and the next."""

    var resolved: Bool
    """Whether the five fields above and the profile were actually read.

    False only for `unresolved()`, the sentinel a defaulted threading
    parameter carries. Every method below tests it before touching a field,
    so an unresolved policy answers what a live read answers rather than
    answering from the zeros it carries. See "The unresolved sentinel" in the
    module docstring."""

    @staticmethod
    def resolve() raises -> ResolvedCpuPolicy:
        """Detect the machine and read all five variables, once.

        Raises for an off-ladder `MOJOTREES_CPU_FEATURE_GROUP` or an
        unparsable `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`, at the one point the
        snapshot is taken, rather than once per node."""
        return ResolvedCpuPolicy(
            CpuProfile.detect(),
            env_tasks_per_core(),
            env_core_pool(),
            env_feature_group(),
            env_compact_min_rows(),
            env_row_blocks(),
            checked_row_block_amortize(),
            True,
        )

    @staticmethod
    def of(profile: CpuProfile) raises -> ResolvedCpuPolicy:
        """A policy around an already-detected profile, reading the five
        variables now. For a caller that has a `CpuProfile` in hand (a
        synthetic one in a test, or one it detected itself) and wants the
        resolved methods without a second detection."""
        return ResolvedCpuPolicy(
            profile.copy(),
            env_tasks_per_core(),
            env_core_pool(),
            env_feature_group(),
            env_compact_min_rows(),
            env_row_blocks(),
            checked_row_block_amortize(),
            True,
        )

    @staticmethod
    def unresolved() -> ResolvedCpuPolicy:
        """The sentinel: no detection, no `getenv`, no answer of its own.

        Cheap by construction, because it is what a defaulted parameter
        builds at every unwired call site: a handful of integer stores and
        not one system query. It is not raising, which is what lets it be a
        default argument at all -- `resolve()` raises on an off-ladder
        feature group, and a default that could raise would move that refusal
        to a place no caller asked for it.
        """
        return ResolvedCpuPolicy(
            CpuProfile(0, 0, 0, 0, False, 0, 0),
            0,
            0,
            0,
            0,
            0,
            RowBlockAmortize(0, 0, False),
            False,
        )

    def dispatch_cores(self) -> Int:
        """`CpuProfile.dispatch_cores` against the resolved pool."""
        if not self.resolved:
            return CpuProfile.detect().dispatch_cores()
        if self.core_pool == CORE_POOL_PERFORMANCE:
            return _positive_or(self.profile.performance_cores, 1)
        return _positive_or(self.profile.physical_cores, 1)

    def max_auto_tasks(self) -> Int:
        """`CpuProfile.max_auto_tasks` against the resolved knobs."""
        if not self.resolved:
            return CpuProfile.detect().max_auto_tasks()
        return _positive_or(
            _positive_or(self.tasks_per_core, DEFAULT_TASKS_PER_CORE)
            * self.dispatch_cores(),
            1,
        )

    def feature_group_for(
        self,
        n_bins: Int,
        n_active: Int,
        row_blocks: Int = 1,
        const_h: Bool = False,
    ) raises -> Int:
        """`plan_feature_group` against the resolved knobs.

        Takes `n_active` and `row_blocks` because the width is not a property
        of the cache estimate alone: the balance rule bounds it by how many
        `(block, group)` units the dispatch can still spread over the cores,
        and both numbers are needed to count one.
        """
        if not self.resolved:
            return plan_feature_group(
                CpuProfile.detect(), n_bins, n_active, row_blocks, const_h
            )
        return _plan_group(
            self.profile.l1d_bytes,
            self.dispatch_cores(),
            self.feature_group,
            n_bins,
            n_active,
            row_blocks,
            const_h,
        )

    def describe(self) -> String:
        """One line naming the snapshot, for benchmark headers. Distinct from
        `CpuProfile.describe`, which re-reads the environment to print it and
        so would report a live value next to a resolved plan."""
        if not self.resolved:
            return String("unresolved (live reads)")
        return String(
            "cores=",
            self.profile.physical_cores,
            " (perf ",
            self.profile.performance_cores,
            ", eff ",
            self.profile.efficiency_cores(),
            ") pool=",
            core_pool_name(self.core_pool),
            " tasks_per_core=",
            self.tasks_per_core,
            " max_auto_tasks=",
            self.max_auto_tasks(),
            " feature_group=",
            self.feature_group,
            " compact_min_rows=",
            self.compact_min_rows,
            " row_blocks=",
            self.row_blocks,
            " row_block_amortize=",
            self.row_block_amortize.describe(),
        )


def private_cell_bytes(const_h: Bool) -> Int:
    """Bytes one cell of the **private accumulation** histogram occupies.

    Not `HISTOGRAM_BYTES_PER_CELL`, and the distinction is the whole point of
    this function existing. That constant is 24 and describes the *output*
    `Histogram`: Float64 gradient, Float64 hessian, and a separate Int count
    plane. The cell the accumulation kernels actually read and write is the
    private one in `histogram._accumulate_blocked_at` and
    `_accumulate_subset_row_major_blocked`, and it is `stride` floats:
    `ROW_BLOCK_CELL_FLOATS_CONST_H` (two, LightGBM's exact `hist_t`) under a
    constant hessian, `ROW_BLOCK_CELL_FLOATS` (three) otherwise.

    Under a constant hessian -- squared error, and every other objective
    `histogram.objective_has_constant_hessian` admits -- that is **16 bytes
    and not 24**, so a cache clamp sized with the output cell believes the
    working set is half again as large as it is.

    Kept separate from `ROW_BLOCK_PLANES`, which stays at 3 unconditionally
    and deliberately: that one sizes the *allocation*, so that the block
    count cannot depend on `const_h` and a summation order cannot move when
    the specialization is switched off. This one sizes only how much of that
    allocation is addressed, which is a schedule and moves nothing.
    """
    var floats = (
        ROW_BLOCK_CELL_FLOATS_CONST_H if const_h else ROW_BLOCK_CELL_FLOATS
    )
    return floats * ROW_BLOCK_PLANE_BYTES


def feature_slice_bytes(n_bins: Int) -> Int:
    """Bytes one feature's histogram slice occupies across the three
    arrays."""
    return n_bins * HISTOGRAM_BYTES_PER_CELL


def feature_slice_bytes_at(n_bins: Int, cell_bytes: Int) -> Int:
    """One feature's slice at an explicit cell size, which is what the cache
    clamp needs. `feature_slice_bytes` is this at the output cell, kept
    because the reporting and the tests that predate the distinction ask for
    exactly that number."""
    return n_bins * cell_bytes


def _cache_group(l1d_bytes: Int, n_bins: Int, cell_bytes: Int) -> Int:
    """The cache clamp over raw integers, so the resolved snapshot and the
    live path cannot drift. `cache_feature_group` is the profile-shaped
    spelling of it.

    `cell_bytes` is `private_cell_bytes(const_h)`, not
    `HISTOGRAM_BYTES_PER_CELL`. It was the latter, hard-coded through
    `feature_slice_bytes`, and that was a defect rather than a conservative
    choice: at 255 bins on the 64 KiB assumed L1 the budget is 32,768 bytes,
    and 32,768 / (255 * 24) is 5 which floors to rung **4**, where
    32,768 / (255 * 16) is 8 which floors to rung **8**. So every constant
    hessian fit -- which is every squared-error fit, the default objective --
    ran an interleave one rung narrower than its own working set allows, and
    the width is the divisor on the number of times the accumulate re-walks
    the node's row-id list and its gathered derivatives. One rung narrower is
    twice the re-walks.

    This does not touch `ASSUMED_L1D_BYTES`, which is a separate question and
    is a deliberate portable floor; the module docstring's argument for
    leaving it alone stands and is untouched by this. The clamp was wrong
    about the cell, not about the cache.
    """
    var slice_bytes = feature_slice_bytes_at(n_bins, cell_bytes)
    if slice_bytes <= 0:
        return 1
    var budget = l1d_bytes // L1_GROUP_DIVISOR
    return feature_group_floor(budget // slice_bytes)


def _schedule_group(cores: Int, n_active: Int, row_blocks: Int = 1) -> Int:
    """The balance clamp over raw integers; see `schedule_feature_group`.

    `row_blocks` is the second axis of the decomposition. The dispatch unit is
    a `(block, group)` pair, so the units this clamp is protecting number
    `row_blocks * ceil(n_active / width)` and not `ceil(n_active / width)`.
    Counting groups alone was correct only while the feature partition was the
    only axis; it is the default when a caller has no block count to hand in,
    and 1 reproduces the group-only arithmetic exactly.
    """
    if cores < 1:
        return 1
    var blocks = row_blocks if row_blocks > 1 else 1
    var wanted_units = TASK_BALANCE_FACTOR * cores
    if wanted_units < 1:
        wanted_units = 1
    # Groups per block still needed to reach `wanted_units` units in total.
    # Ceiling, so a block count that does not divide the demand still buys the
    # whole demand rather than one unit short of it.
    var wanted_groups = (wanted_units + blocks - 1) // blocks
    if wanted_groups < 1:
        wanted_groups = 1
    return feature_group_floor(n_active // wanted_groups)


def _plan_group(
    l1d_bytes: Int,
    cores: Int,
    requested: Int,
    n_bins: Int,
    n_active: Int,
    row_blocks: Int = 1,
    const_h: Bool = False,
) raises -> Int:
    """The one copy of the width rule, with every input already read.

    Both entry points land here: `plan_feature_group` after a `getenv`, and
    `ResolvedCpuPolicy.feature_group_for` from a snapshot taken once per fit.
    That matters more than it looks. A cached policy that answered a width
    the live path would refuse, or refused one the live path would answer,
    would put the accumulation kernel and the plan that sized its buffers at
    different widths, and the two are not separately checked anywhere.

    `const_h` reaches only the cache clamp, through `private_cell_bytes`; see
    `_cache_group` for what it corrects. It defaults to `False`, the
    three-plane cell, so that every caller which has no objective to hand --
    the reporting helpers, and the tests that predate the distinction -- gets
    the answer it always got. **The default is not the production path.**
    `_derive_plan` passes the fit's real value, which is what the accumulate
    kernels then run at, and `tests/test_cpu_feature_group.mojo` asserts both
    arms by literal rung. A default that every call site took would be the
    same defect this fix is closing, wearing a different hat.

    It is bit-neutral for the reason every other input here is: the width
    changes how many features share one walk of the rows, never the order in
    which any one feature's bins are summed. It is also machine-independent
    and worker-independent, since `const_h` is a property of the objective.
    """
    if requested > 0:
        return requested
    var group = DEFAULT_FEATURE_GROUP
    var by_cache = _cache_group(l1d_bytes, n_bins, private_cell_bytes(const_h))
    if by_cache < group:
        group = by_cache
    var by_schedule = _schedule_group(cores, n_active, row_blocks)
    if by_schedule < group:
        group = by_schedule
    var by_active = feature_group_floor(n_active)
    if by_active < group:
        group = by_active
    return group if group >= 1 else 1


def cache_feature_group(
    profile: CpuProfile, n_bins: Int, const_h: Bool = False
) -> Int:
    """The widest rung whose group of histogram slices fits the L1 budget.

    Interleaving is only useful while every slice in the group stays in L1:
    the whole point is that the N read-modify-write chains do not stall on
    memory. `L1_GROUP_DIVISOR` leaves the rest of the assumed cache for the
    row index stream, the gradient pairs, and the N binned columns the loop
    walks, all of which stream through the same cache.

    At the default shape this is the arithmetic that decides everything, and
    it now has two answers because the cell has two sizes:

    - three-plane cell, `const_h` false: slice is `255 * 24 = 6120`, budget is
      `65536 / 2 = 32768`, `32768 / 6120 = 5` slices fit, and
      `feature_group_floor(5)` is **4**.
    - two-plane cell, `const_h` true: slice is `255 * 16 = 4080`,
      `32768 / 4080 = 8`, and the rung is **8**.

    The second is what a squared-error fit actually runs and it was getting
    the first. Every one of those numbers is still an assumption or a floor,
    none is measured, and `ASSUMED_L1D_BYTES` in particular is still a
    portable floor rather than this machine's cache; the module docstring
    says why it stays that way and this change does not disturb it.
    """
    return _cache_group(profile.l1d_bytes, n_bins, private_cell_bytes(const_h))


def dispatch_rounds(n_tasks: Int, cores: Int) -> Int:
    """Barrier rounds `n_tasks` equal-cost tasks take on `cores` cores.

    The crude model the balance rule is priced with, written down so it can be
    disagreed with: greedy scheduling, equal task costs, equal cores, one
    barrier at the end. Real cores here are not equal, so this understates the
    cost of a partial round rather than overstating it.
    """
    if n_tasks <= 0 or cores <= 0:
        return 0
    return (n_tasks + cores - 1) // cores


def dispatch_utilization_percent(n_tasks: Int, cores: Int) -> Int:
    """Core-seconds used over core-seconds elapsed, as a percentage.

    `n_tasks / (cores * dispatch_rounds(...))`, truncated. 13 tasks on 10
    cores is 65: two rounds, seven cores idle through the second. 25 on 10 is
    83, 40 on 10 is 100. This is the number a memory-traffic table cannot see,
    and reporting a width without it is how a change that halves traffic and
    halves the task count gets recorded as a win.
    """
    var rounds = dispatch_rounds(n_tasks, cores)
    if rounds <= 0:
        return 0
    return (100 * n_tasks) // (cores * rounds)


def schedule_feature_group(
    profile: CpuProfile, n_active: Int, row_blocks: Int = 1
) -> Int:
    """The widest rung that still leaves `TASK_BALANCE_FACTOR` units per core.

    **A unit is a `(block, group)` pair, not a group.** `histogram.
    _accumulate_blocked_at` dispatches over `row_blocks * group_count` of them
    and indexes them as `block * group_count + group`, so a node cut into `B`
    blocks offers `B * ceil(n_active / width)` units of parallelism. This rule
    counted groups alone, which was right when the feature partition was the
    only axis and became wrong the moment row blocking landed: it was refusing
    a width to protect a task count the other axis had already multiplied.

    Worked, on the shape this lane exists for: 50 active features, 255 bins, a
    node of 200,000 rows, on a 14-core profile. The demand is `2 * 14 = 28`
    units. Blocked 49 ways the rule needs `ceil(28 / 49) = 1` group per block,
    `50 // 1 = 50`, floored to the rung 16 -- so this clamp stops binding
    entirely and the L1 estimate (4 at 255 bins) decides. Counting groups
    alone the same shape asked for 28 groups, `50 // 28 = 1`, and returned
    width **1**: no interleaving at all, and the gradient stream walked 50
    times per block instead of 13.

    With `row_blocks` at its default of 1 this is the old arithmetic to the
    integer: `2 * 10 = 20` groups wanted, `50 // 20 = 2`, width 2, 25 groups in
    3 rounds at 83% utilization. That is the right answer for an unblocked
    node and it is still the answer given.

    `dispatch_cores()` and not `max_auto_tasks()`: the 4x over-decomposition is
    itself a balance device against unequal cores, and counting it here as well
    would demand 400 units and cap the width at 1 on every shape this project
    cares about. The two floors are for the same problem and applying both
    would double-charge for it.

    Fewer features than the per-block group demand gives 1, which is what the
    dispatch already did on such a shape and gives up nothing measurable:
    there was never a wide group for a task to hold.

    None of this moves a bit. The width decides how many features share one
    walk of the rows; it never changes which Float64 are added into a cell or
    in what order, and the private-histogram layout is indexed by absolute
    active slot rather than by position within a group. The block count is the
    part of this decomposition that *is* a value, and it is derived by
    `plan_row_block_count` from the workload alone -- this function reads it
    and cannot influence it.
    """
    return _schedule_group(profile.dispatch_cores(), n_active, row_blocks)


def plan_feature_group(
    profile: CpuProfile,
    n_bins: Int,
    n_active: Int,
    row_blocks: Int = 1,
    const_h: Bool = False,
) raises -> Int:
    """How many features one accumulation inner loop interleaves.

    An explicit `MOJOTREES_CPU_FEATURE_GROUP` is returned as asked, having
    been checked against the ladder. It deliberately bypasses all three clamps
    below: those encode an assumed cache size and an unmeasured guess about
    what parallelism is worth, and a knob that could not exceed them could not
    be used to test them. It may exceed `n_active`, which costs nothing: the
    kernel's tail group owns only the features that remain.

    Otherwise the width is the narrowest of what the cache estimate allows,
    what the balance rule leaves schedulable, and what `n_active` features can
    fill, floored to a rung throughout. The three are independent bounds and
    which one binds depends on the shape. `row_blocks` is how many blocks the
    node will be cut into, because the balance rule counts `(block, group)`
    units; at its default of 1 the bound is the group-only one it always was.

    At 255 bins and 50 features on ten cores with one block the balance rule
    binds at 2 while the cache estimate would have allowed 4; at 49 blocks --
    what a 200,000-row node plans at those bins -- the balance rule stops
    binding and the cache estimate's 4 is the answer. At 32 bins the cache
    estimate allows 16, so the balance rule or the active-feature floor
    decides.

    Whatever it returns, the result is bit-identical: the width changes how
    many features share one walk of the rows, never the order in which any one
    feature's bins are summed.
    """
    return _plan_group(
        profile.l1d_bytes,
        profile.dispatch_cores(),
        env_feature_group(),
        n_bins,
        n_active,
        row_blocks,
        const_h,
    )



@fieldwise_init
struct AccumulationPlan(Copyable, Movable):
    """The scheduling shape of one histogram build.

    Three dispatches can run: the excluded features' slices are zeroed, the
    node's gradients and hessians are optionally gathered, and the active
    features are zeroed and accumulated. Each carries its own work estimate,
    because sizing all three by the row count is what used to send a
    multi-megabyte zeroing pass down the serial path on a small node.
    """

    var group_width: Int
    """Features one inner loop interleaves; 1 is the plain per-feature
    loop. Always a rung of the ladder, because the kernel is instantiated per
    rung."""

    var group_count: Int
    """Groups the active features split into, and therefore the number of
    units the accumulation dispatches over. A task takes whole groups, which
    is what stops the task splitter from handing a task fewer features than
    the width it was told to interleave."""

    var compact_rows: Bool
    """Whether to gather the node's (gradient, hessian) pairs into a
    contiguous scratch buffer before the feature loop. False for the
    full-dataset builder, whose rows are already contiguous."""

    var active_ops: Int
    """Work estimate for the zero-and-accumulate dispatch, in histogram-op
    equivalents: one zeroed cell and one accumulated row each count as one."""

    var excluded_ops: Int
    """Work estimate for zeroing the slices of features this build does not
    accumulate (feature subsampling leaves them zero)."""

    var gather_ops: Int
    """Work estimate for the gradient/hessian gather, zero when it is
    skipped. Two indirect loads and two sequential stores per row, counted
    as two ops."""

    var row_blocks: Int
    """Contiguous ascending row blocks the node is split into, each with its
    own private histogram. **1 means the feature-partition kernel**, which is
    what shipped and which this field leaves untouched. Above 1 selects the
    blocked kernel, and it is the one field on this plan that is part of the
    result rather than part of the schedule: a fold over blocks is a different
    Float64 from a single ascending sum. `plan_row_block_count` derives it
    from the workload shape alone, never from the machine."""

    var block_rows: Int
    """Rows in each block but the last; the last holds the remainder. Equal to
    `n_rows_touched` when `row_blocks` is 1."""

    var block_cells: Int
    """Cells in one block's private histogram, `n_active * n_bins`. The
    compact active-slice shape, not the full `n_features * n_bins` output: an
    excluded feature is never accumulated and never folded, so it gets no
    private storage. Zero when `row_blocks` is 1."""

    var block_ops: Int
    """Work estimate for the blocked zero-and-accumulate dispatch: one op per
    accumulated (row, active feature) plus one per zeroed private cell, so
    `n_active * rows + row_blocks * block_cells`. Zero when `row_blocks` is
    1, where `active_ops` is the estimate instead."""

    var fold_ops: Int
    """Work estimate for the fold, in histogram-op equivalents.

    `ROW_BLOCK_PLANES * (row_blocks + 1) * block_cells`: every partial cell is
    read once and every output cell written once, on each of three planes.
    Counted per plane rather than per cell for exactly the reason
    `subtract_ops` gives -- the pass is memory-bound, and a per-cell estimate
    keeps it serial well past the point where three streams saturate one
    core. That is not a hypothetical here: at 50 features, 255 bins and three
    blocks a per-cell estimate is 51,000, which is below the 65,536 crossover,
    so the fold would run single-threaded on the critical path of every node
    in the band where blocking first engages. Zero when `row_blocks` is 1."""

    def total_ops(self) -> Int:
        return (
            self.active_ops
            + self.excluded_ops
            + self.gather_ops
            + self.block_ops
            + self.fold_ops
        )

    def blocked(self) -> Bool:
        """Whether this plan selects the row-blocked accumulation. The one
        predicate any caller should branch on, so "more than one block" is
        spelled once."""
        return self.row_blocks > 1

    def block_scratch_floats(self) -> Int:
        """Float64 slots the private histograms need, `row_blocks *
        block_cells * ROW_BLOCK_PLANES`. Zero for an unblocked plan.

        Named here rather than at the kernel because the plan is what decides
        the block count, and a scratch sized from a different number than the
        kernel indexes with is the failure mode this whole struct exists to
        make impossible."""
        if self.row_blocks <= 1:
            return 0
        return self.row_blocks * self.block_cells * ROW_BLOCK_PLANES

    def describe(self) -> String:
        var compact = String("yes") if self.compact_rows else String("no")
        return String(
            "group_width=",
            self.group_width,
            " group_count=",
            self.group_count,
            " compact_rows=",
            compact,
            " active_ops=",
            self.active_ops,
            " excluded_ops=",
            self.excluded_ops,
            " gather_ops=",
            self.gather_ops,
            " row_blocks=",
            self.row_blocks,
            " block_rows=",
            self.block_rows,
            " block_cells=",
            self.block_cells,
            " block_ops=",
            self.block_ops,
            " fold_ops=",
            self.fold_ops,
        )


def derive_accumulation_plan(
    profile: CpuProfile,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
    const_h: Bool = False,
) raises -> AccumulationPlan:
    """Plan one histogram build.

    `n_active` is how many features will be accumulated (all of them unless
    the caller passed a feature subset), `n_rows_touched` how many rows each
    of them walks, and `rows_are_indirect` whether those rows arrive through
    a row-id list (a tree node) rather than as the whole contiguous dataset.

    The gather is chosen when it can pay for itself: it costs one pass over
    the node's rows and saves the remaining features an indirect load of the
    gradient and the hessian per row, so it wants at least two active
    features and enough rows that the pass is not pure overhead.

    Raises only for an off-ladder `MOJOTREES_CPU_FEATURE_GROUP` or an
    unparsable `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`; every workload shape,
    including a degenerate one, produces a plan.
    """
    return _derive_plan(
        profile.l1d_bytes,
        profile.dispatch_cores(),
        env_feature_group(),
        env_compact_min_rows(),
        env_row_blocks(),
        checked_row_block_amortize(),
        n_features,
        n_active,
        n_bins,
        n_rows_touched,
        rows_are_indirect,
        const_h,
    )


def derive_accumulation_plan_with(
    policy: ResolvedCpuPolicy,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
    const_h: Bool = False,
) raises -> AccumulationPlan:
    """`derive_accumulation_plan` against an already-resolved policy.

    Identical arithmetic on identical inputs; the only difference is that the
    two variables come from the snapshot rather than from a fresh `getenv`,
    and no `CpuProfile.detect()` happens here at all.

    Given the unresolved sentinel this is `derive_accumulation_plan` over a
    fresh detection, so a call site that has not been handed a snapshot yet
    keeps exactly the behaviour it had, including the `getenv` it had.
    """
    if not policy.resolved:
        return derive_accumulation_plan(
            CpuProfile.detect(),
            n_features,
            n_active,
            n_bins,
            n_rows_touched,
            rows_are_indirect,
            const_h,
        )
    return _derive_plan(
        policy.profile.l1d_bytes,
        policy.dispatch_cores(),
        policy.feature_group,
        policy.compact_min_rows,
        policy.row_blocks,
        policy.row_block_amortize.copy(),
        n_features,
        n_active,
        n_bins,
        n_rows_touched,
        rows_are_indirect,
        const_h,
    )


def _derive_plan(
    l1d_bytes: Int,
    dispatch_cores: Int,
    feature_group: Int,
    compact_min_rows: Int,
    row_blocks: Int,
    row_block_amortize: RowBlockAmortize,
    n_features: Int,
    n_active: Int,
    n_bins: Int,
    n_rows_touched: Int,
    rows_are_indirect: Bool,
    const_h: Bool = False,
) raises -> AccumulationPlan:
    """The one copy of the plan arithmetic, with every variable already
    read. `dispatch_cores` is here because the width now depends on how many
    groups the schedule can balance, not only on what L1 holds.

    Note which arguments reach the row-block decision and which do not.
    `dispatch_cores` feeds `_plan_group` and nothing else; the block count is
    derived from `rows`, `bins`, `active`, the amortization ratio and the
    explicit request alone. A machine fact must never reach it, because the
    block count is a summation order rather than a schedule (see
    `plan_row_block_count`). `row_block_amortize` is a workload rule and not a
    machine fact, which is the whole reason it was allowed in here.

    The dependency between the two axes runs **one way, and only one way**:
    the settled block count is an input to the width, and the width is not an
    input to anything the block count reads. That direction is what keeps the
    determinism argument intact. The width is machine-dependent (it reads
    `dispatch_cores`) and moves nothing but the schedule; the block count is
    machine-independent and is part of the value. Were the arrow reversed, a
    core count would reach a summation order and the histogram would stop
    being the same on two machines.
    """
    var active = n_active if n_active > 0 else 0
    var rows = n_rows_touched if n_rows_touched > 0 else 0
    var bins = n_bins if n_bins > 0 else 0
    var excluded = n_features - active
    if excluded < 0:
        excluded = 0

    var compact = (
        rows_are_indirect
        and active >= COMPACT_MIN_FEATURES
        and rows >= compact_min_rows
    )
    # The block count comes first, because the width depends on it and it does
    # not depend on the width. The requested count, then the geometry, then a
    # recount from the chunk so no block comes out empty. The recount can only
    # lower the count, and it can lower it to 1, which is the unblocked path
    # again -- so the width is planned against the *settled* count, never the
    # provisional one.
    var blocks = plan_row_block_count_at(
        row_block_amortize, row_blocks, rows, bins, active, rows_are_indirect
    )
    var chunk = row_block_geometry(rows, blocks)
    if blocks > 1:
        blocks = row_block_count_from_chunk(rows, chunk)
    var cells = active * bins
    if blocks <= 1:
        blocks = 1
        chunk = rows
        cells = 0

    var group = _plan_group(
        l1d_bytes, dispatch_cores, feature_group, bins, active, blocks,
        const_h,
    )

    return AccumulationPlan(
        group,
        feature_group_count(active, group),
        compact,
        active * (bins + rows),
        excluded * bins,
        2 * rows if compact else 0,
        blocks,
        chunk,
        cells,
        (active * rows + blocks * cells) if blocks > 1 else 0,
        (ROW_BLOCK_PLANES * (blocks + 1) * cells) if blocks > 1 else 0,
    )


def subtract_ops(n_cells: Int) -> Int:
    """Work estimate for a sibling subtraction over `n_cells` histogram
    cells. Each cell is a load, a subtract, and a store in each of three
    independent streams, so it is counted as three ops rather than one: the
    pass is memory-bound and a single-op estimate keeps it serial well past
    the point where the streams saturate one core."""
    return subtract_ops_for_planes(n_cells, 3)


def subtract_ops_for_planes(n_cells: Int, n_planes: Int) -> Int:
    """`subtract_ops` for a subtraction that touches `n_planes` of the three.

    The constant-hessian specialization in `histogram.mojo` subtracts the
    gradient and count planes and derives the hessian plane from the count it
    just wrote, so its pass runs two of the three read-modify-write streams
    rather than three. The estimate is the traffic, so it has to follow: a
    two-plane pass that told `parallel.plan_tasks` it was a three-plane one
    would clear the crossover at two thirds of the size that actually
    saturates a core.

    Nothing about this can move a value. The subtraction is elementwise over
    disjoint ascending blocks, so the task count changes which core writes a
    cell and nothing else, which is the guarantee `parallel.mojo` states for
    `dispatch_rows` and the reason a work estimate is allowed to be a
    judgment call at all. A plane count below one is clamped to one, so a
    degenerate caller still gets a positive estimate rather than a zero that
    would read as no work.
    """
    var planes = n_planes if n_planes > 0 else 1
    return planes * n_cells


# What one scored split candidate costs, in the histogram-op equivalents
# `parallel.plan_tasks` compares against its threshold. A histogram op is a
# scattered read-modify-write into three L1-resident arrays; a split
# candidate is two Float64 divisions, an L1 soft-threshold on each child, and
# a handful of compares. A division is the term that dominates and it is the
# one instruction on this class of core that is neither single-cycle nor well
# pipelined, so a candidate is counted as several ops rather than one.
#
# 8 is an order-of-magnitude reading of that instruction mix, not a
# measurement: nothing in this repository has timed a split candidate against
# a histogram op. It matters only through the threshold, so getting it wrong
# moves the size at which the scan starts fanning out and moves nothing else.
# bench/bench_threshold.mojo is the harness that would settle it.
comptime SPLIT_CANDIDATE_OPS = 8


def split_scan_ops(n_active: Int, n_bins: Int, two_sided: Bool) -> Int:
    """Work estimate for one node's split scan over `n_active` features.

    Every ordinary bin of every scanned feature is a threshold, and a feature
    with a reserved missing bin scores each threshold twice, once with the
    missing rows left and once with them right (`two_sided`). The totals pass
    that precedes each feature's scan is one streaming read of the same
    slice, cheap next to the candidates, and is counted as one op per bin.

    Categorical features are not this shape at all: their candidate count is
    a sorted partition search over categories rather than a walk over bins.
    They are estimated here as if they were ordinal, which under-counts a
    wide categorical node. That only ever makes the scan more conservative
    about fanning out.
    """
    var active = n_active if n_active > 0 else 0
    var bins = n_bins if n_bins > 0 else 0
    var per_bin = 2 * SPLIT_CANDIDATE_OPS if two_sided else SPLIT_CANDIDATE_OPS
    return active * bins * (per_bin + 1)


def describe_cpu_policy(profile: CpuProfile, plan: AccumulationPlan) -> String:
    """One line carrying both halves of the width decision.

    The utilization figure is appended here rather than in
    `AccumulationPlan.describe` because it needs the core count, which the plan
    does not carry. It is the number that makes a wide group's cost visible
    next to its benefit; a benchmark header that prints only the width is
    reporting half of the trade.

    It is priced over `row_blocks * group_count`, which is what
    `histogram._accumulate_blocked_at` actually dispatches
    (`dispatch_feature_ranges_with(..., n_blocks * n_groups, ...)`). Printing
    `group_count` alone under-reported a blocked build by the block count and
    would have made a wider group look like a scheduling loss it is not.
    """
    var cores = profile.dispatch_cores()
    var units = plan.row_blocks * plan.group_count
    return String(
        profile.describe(),
        " | ",
        plan.describe(),
        " units=",
        units,
        " rounds=",
        dispatch_rounds(units, cores),
        " utilization=",
        dispatch_utilization_percent(units, cores),
        "%",
    )


# ---------------------------------------------------------------------------
# Bin layout: which copy of the binned matrix an accumulation reads
# ---------------------------------------------------------------------------
#
# `BinnedMatrix` can carry two views of the same bytes. The feature-major one
# is the original, `bins[f * n_rows + r]`, one contiguous column per feature.
# The row-major one is `binning.BinnedMatrix.build_row_major`'s record array,
# one contiguous fixed-width record per row holding every feature's bin, with
# features of at most 16 bins packed two to a byte. LightGBM calls the second
# a `MultiValBin` and selects between them with `force_row_wise` /
# `force_col_wise`.
#
# **Nothing in this section moves a bit.** The two layouts hold the same bin
# ids and an accumulation over either visits the same rows in the same order
# and adds the same Float64 into the same cell, so the histogram is identical
# to the bit. In particular the layout choice must never reach
# `plan_row_block_count`: the block count *is* part of the value (see this
# module's docstring), so the two arms are run against **one** plan and differ
# only in which array their inner loop loads from.
#
# How the choice is made is deliberately not arithmetic here. LightGBM's
# entire auto rule is one timed histogram construction with each builder at
# the start of the fit, keeping the faster for the whole fit and printing
# which it chose; `histogram.choose_bin_layout_timed` is that probe. A cost
# model over traffic was drafted for this decision and is recorded in the
# lane report rather than in code, because a model nobody measured would be
# tuning by another name -- and because the timed shot is strictly better
# information than the model on the machine that is actually running.

comptime BIN_LAYOUT_FEATURE_MAJOR = 0
"""Accumulate from `bins[f * n_rows + r]`, the column-major matrix."""

comptime BIN_LAYOUT_ROW_MAJOR = 1
"""Accumulate from `row_bins[r * row_stride + row_byte[f]]`, the record
array."""

comptime BIN_LAYOUT_AUTO = 2
"""Unresolved: the caller should time both and keep the faster. Never
returned by `resolve_bin_layout` when the row-major view is unavailable."""


def bin_layout_name(layout: Int) -> String:
    """The name a benchmark header or a trace line prints for a layout."""
    if layout == BIN_LAYOUT_ROW_MAJOR:
        return String("row")
    if layout == BIN_LAYOUT_AUTO:
        return String("auto")
    return String("feature")


def env_bin_layout() raises -> Int:
    """`MOJOTREES_CPU_BIN_LAYOUT`: `auto` (default), `feature`, or `row`.

    Scheduling-only, like every variable in this file but the two row-block
    ones (`MOJOTREES_CPU_ROW_BLOCKS` and
    `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`): it selects which array the inner loop loads
    from and cannot change a cell. `feature` is the off switch for the whole
    row-major path and reproduces the accumulation that shipped, which is what
    a bisection wants.

    Raises on an unrecognized value rather than silently meaning `auto`,
    because a benchmark arm that quietly ran the other layout is exactly the
    result this campaign has already had to throw away twice.
    """
    var s = getenv("MOJOTREES_CPU_BIN_LAYOUT")
    if s.byte_length() == 0 or s == "auto":
        return BIN_LAYOUT_AUTO
    if s == "feature" or s == "col" or s == "0":
        return BIN_LAYOUT_FEATURE_MAJOR
    if s == "row" or s == "1":
        return BIN_LAYOUT_ROW_MAJOR
    raise Error(
        'MOJOTREES_CPU_BIN_LAYOUT must be "auto", "feature" or "row". Got "',
        s,
        '"',
    )


def resolve_bin_layout(requested: Int, row_major_available: Bool) -> Int:
    """The layout a build should run, given the request and what exists.

    Availability wins over the request in one direction only: without a
    row-major view there is nothing to read, so `row` degrades to
    `feature` rather than raising -- a matrix that was binned before the view
    was ever asked for is a normal thing for a caller to hold, and refusing to
    build its histogram would be a worse answer than building it the way it
    was always built. `auto` with no view available resolves the same way, so
    the timed probe is only ever reached with both arms runnable.
    """
    if not row_major_available:
        return BIN_LAYOUT_FEATURE_MAJOR
    if requested == BIN_LAYOUT_ROW_MAJOR:
        return BIN_LAYOUT_ROW_MAJOR
    if requested == BIN_LAYOUT_FEATURE_MAJOR:
        return BIN_LAYOUT_FEATURE_MAJOR
    return BIN_LAYOUT_AUTO


def histogram_line_floats(cache_line_bytes: Int) -> Int:
    """Float64 slots in one cache line, never below 1.

    The padding unit for a per-block private histogram. LightGBM pads the same
    buffers to `num_bin_aligned_` for the same reason: two blocks whose
    partials share a line make two cores fight over it on every store, and
    `n_bins * ROW_BLOCK_PLANE_BYTES` is 2,040 bytes at 255 bins, which is not
    a multiple of a 128-byte line.
    """
    var n = cache_line_bytes // ROW_BLOCK_PLANE_BYTES
    return n if n > 0 else 1


def align_cells_up(cells: Int, unit: Int) -> Int:
    """`cells` rounded up to a multiple of `unit`, for the padding above."""
    if unit <= 1 or cells <= 0:
        return cells if cells > 0 else 0
    return ((cells + unit - 1) // unit) * unit
