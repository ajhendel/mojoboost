# Task 13: Apple CPU performance, SIMD, cache, and scheduling

Status: landed in the four files this lane owns. Nothing was compiled, run,
tested, or benchmarked, per the lane's constraints. Every claim below is a
mechanism with a stated falsification, not a measurement.

This lane ran no `git` command and staged nothing. It should be said anyway
that the checkout is shared with the other lanes of this round, and a
concurrent lane's sweeping commit (`b04b5f0`, "Integrate parallel release and
accelerator work") already swept up `parallel.mojo` and the first version of
`apple_cpu_policy.mojo` mid-flight; the rest of this lane's edits are still in
the working tree. So "uncommitted" describes this lane's intent, not the state
of the repository, and whoever integrates should diff rather than assume.

Files changed or added (all owned by this lane):

- `src/mojoboost/apple_cpu_policy.mojo` (new)
- `src/mojoboost/histogram.mojo`
- `src/mojoboost/binning.mojo`
- `src/mojoboost/parallel.mojo`
- `bench/apple/cpu_plan.json` (new)
- `handoffs/performance_13_apple_cpu.md` (this file)

Nothing else was touched. Every change that would land outside these files is
in "Required external edits" below, unapplied.

## The one thing to check first

None of this has been through a compiler, so every construct it uses was
checked against a construct this repository already ships. Two could not be,
and they are the only dialect risks left:

1. `comptime TARGET_HAS_NEON = CompilationTarget.has_neon()` and
   `comptime ASSUMED_CACHE_LINE_BYTES = ... if ... else ...` at module scope
   in `apple_cpu_policy.mojo`. The nearest precedent is
   `comptime SIMD_LANES = 4 * simd_width_of[DType.float64]()` in
   `histogram.mojo`, which folds a target query at module scope; a static
   method on `CompilationTarget` is the same class of query but not the same
   spelling. If it is not foldable there, move both inside
   `CpuProfile.detect` and mark `detect`, `cpu_profile`, `max_auto_tasks`,
   `plan_tasks`, and `plan_row_blocks` `raises`. That cascade is why they are
   `comptime` in the first place, and the blast radius of getting it wrong is
   small either way: both values are report-only (`has_neon` and
   `cache_line_bytes` are printed by `describe` and read by no decision).
2. `num_logical_cores()` and `num_performance_cores()` are called from
   non-raising functions. `num_physical_cores()` already was, in the old
   `plan_tasks` body; the assumption is that its two neighbours in
   `std.sys.info` have the same signature. If not, mark `detect` and its
   callers `raises` as in (1), or drop `logical_cores` (report-only) and read
   `performance_cores` only under the `performance` pool.

Retired after checking the repository, recorded so nobody re-raises them:
`comptime TASKS_PER_CORE = DEFAULT_TASKS_PER_CORE` (a module-scope `comptime`
bound to an imported one) is exactly what `device.mojo:54-56` and
`gpu_sparse.mojo:170` already do; and the per-task sort buffer in `fit_bins`
is emptied with `col.clear()`, the idiom `boosting.fill_grad_hess`,
`objective.mojo`, and `bagging.mojo` already rely on for capacity-preserving
reuse. An earlier draft used `resize(0, 0.0)`, which nothing else in the
repository does.

The equivalence tests in "Commands" are the gate for all of it, and they are
the first thing to run.

## Map of hot loops changed

| Loop | File / anchor | Before | After |
| --- | --- | --- | --- |
| Full-dataset accumulation | `histogram._accumulate_full` | one `func(f)` per feature; separate `out.reset()` pass first | task takes a feature *range*; zeroes each slice then accumulates it; two features share one row walk |
| Node accumulation | `histogram._accumulate_subset` | same, plus two indirect gradient/hessian loads per feature per row | optional one-time gather of `(g, h)` pairs; paired features on the gathered path; fused zeroing |
| Output zeroing | `histogram.build_*_into` | `Histogram.reset`, serial, whole buffer | fused into the feature tasks; excluded features zeroed by their own dispatch (`_zero_excluded`) |
| Sibling subtraction | `histogram.subtract_histogram_into` | serial vector sweep | same kernel over contiguous blocks via `dispatch_rows`, work estimate weighted by three streams |
| Quantile fit | `binning.fit_bins` | one `List` allocation per feature inside the task | one sort buffer per task, emptied and refilled; work estimate carries the comparison count |
| Bin transform | `binning.BinMapper.transform` | work estimate of one op per (feature, row) | estimate carries the binary-search depth |
| Equal-width binning | `binning.bin_equal_width` | serial appending loop over all features | sized up front, written by index, dispatched across features |
| Task split | `parallel.dispatch_features` | ceiling chunk, trailing tasks can be empty | even split; new `dispatch_feature_ranges` is the primitive and `dispatch_features` is written on it |
| Task ceiling | `parallel.plan_tasks` | `TASKS_PER_CORE * num_physical_cores()` inline | `cpu_profile().max_auto_tasks()`, same default, now a policy decision |

New public surface, all additive: `parallel.dispatch_feature_ranges`,
`histogram.build_histogram_subset_into_scratch`,
`histogram.ensure_pair_capacity`, `histogram.FeatureTotals`,
`histogram.feature_totals`, and the whole of `apple_cpu_policy`.
No existing signature changed, so no other lane's file needs an edit to keep
compiling.

## Mechanism, one per improvement

**Fused zeroing.** A build writes all `n_features * n_bins` cells and then
accumulates into some of them. As a separate pass that is a streaming write of
`n_features * n_bins * 24` bytes (about 600 KB at 100 features and 255 bins,
several MB at 500) which is then read back in by the accumulation, evicted in
between. Done inside the task that is about to fill the slice, the zeroed line
is still in L1 when the first accumulate lands on it, and the pass costs
nothing extra to parallelize because it rides the dispatch that was already
happening. It also stops the zeroing from being invisible to the scheduler:
`derive_accumulation_plan` counts it, so a node with 20 rows and a 3 MB output
no longer runs 3 MB of stores on one core because "20 rows is below the
threshold".

**Gradient/hessian gather.** Accumulating a node reads `grad[rows[i]]` and
`hess[rows[i]]`. Those two loads are indirect, over buffers of `8 * n_rows`
bytes each, and *every active feature repeats them* because the feature loop is
outside the row loop. At 100 features that is 200 indirect loads per row of the
node. Gathering once into an interleaved `(g, h)` buffer costs one pass of the
same two indirect loads and turns the other 99 features' reads into a
sequential walk of one buffer, where consecutive rows share a cache line and
the prefetcher can see the stream. Interleaved rather than two buffers so one
line carries both halves of a row.

The gather is skipped when it cannot pay: one active feature (nothing to
amortize over) or fewer than `DEFAULT_COMPACT_MIN_ROWS` rows (the node's
gradients already fit in cache across the whole feature loop, so there are no
misses to remove). `bin_equal_width`-shaped full builds never gather, because
their rows are the whole dataset in order and already contiguous.

**Feature pairing.** Two features' histogram slices are disjoint, so their
read-modify-write chains are independent. Walking both in one row loop loads
the row id and the gradient pair once instead of twice and gives the core two
chains to overlap, which is what hides the store-to-load forwarding latency
when consecutive rows land in the same bin. The compiler cannot prove
`base0 + b0` and `base1 + b1` disjoint (they are offsets into the same
allocation), so it must keep program order; the overlap has to come from the
hardware's memory disambiguation. That makes this the weakest claim in the
lane. See "Claims requiring evidence".

**Scratch reuse.** The pair buffer is the caller's
(`build_histogram_subset_into_scratch`), grown and never shrunk, so a grower
holding one buffer allocates once per tree at the size of its largest node
instead of once per node. The existing entry point still works and simply owns
a temporary buffer for the call, which is what makes this lane's change
non-breaking; the reuse only arrives when `tree.mojo` adopts the scratch form
(below).

**Parallel sibling subtraction.** The subtraction trick avoids an accumulation
but the pass it substitutes is not free: six streams over
`n_features * n_bins` cells per node, memory-bound, and it was serial. It is
elementwise, so contiguous blocks reproduce it exactly. The work estimate is
three ops per cell rather than one (`apple_cpu_policy.subtract_ops`), because
each cell is a load-subtract-store in each of three arrays; at one op per cell
the default shape sits just under the grain floor and never parallelizes.

**Work estimates.** `plan_tasks` compares an estimate in histogram-op
equivalents against a grain. A quantile fit sorts each column (about
`log2(n_rows)` comparisons per element) and a transform binary-searches each
value (about `log2(n_bins)` dependent compares), so counting one op per
(feature, row) understated both by 8 to 20 times and kept them serial well past
the point where a dispatch pays for itself. Each estimate now carries its
stage's real per-row cost. This changes only which path runs.

**Even task split.** A ceiling-divided chunk can leave trailing tasks empty
(6 tasks over 10 features: chunk 2, five tasks of work, one scheduling event
for nothing) and puts the whole remainder on the last task. `w * n // tasks`
splits within one item and spreads the remainder. On a machine whose cores are
not all the same speed, a task that finishes early can only help if there is
another task to take; empty tasks are the opposite of that.

**Core pool.** `num_physical_cores()` counts efficiency cores, and a
`sync_parallelize` is a barrier: the fan-out finishes when its slowest task
does. The two portable responses are over-decomposition (already the default:
`TASKS_PER_CORE = 4`) and shrinking the pool. The default is unchanged, and
`MOJOBOOST_CPU_CORE_POOL=performance` exists so the question can be measured
rather than assumed. No affinity or QoS API is used, and none is assumed to
exist: nothing here pins a thread, and nothing here would break if the OS
migrated every task to a different core type mid-run.

## Semantic invariants

These hold at every setting of every environment variable, on every machine,
and are what makes all of the above safe to tune:

1. **No floating-point reassociation.** A feature's bins are accumulated over
   its rows in ascending order inside exactly one task. Pairing two features
   changes neither feature's order. The gather copies `Float64` values
   bit-exactly and changes only where they are read from. No partial
   histograms, no per-block accumulators, no cross-task reduction.
2. **Every output cell is written exactly once per build.** Active features'
   slices are zeroed by the task that fills them; excluded features' slices by
   `_zero_excluded`. Together those cover `[0, n_features)` with no overlap, so
   an `_into` buffer holding a previous node's histogram comes back holding
   exactly this one. This is what `test_into_builders_match_allocating_and_survive_reuse`
   checks, and it is the invariant that replaced `out.reset()`.
3. **Feature ids must be distinct.** Two tasks handed the same feature would
   zero and accumulate the same slice concurrently. The previous code had the
   same requirement (concurrent accumulation into one slice) and neither
   version checks it; it is now stated in the module docstring. Callers get
   their ids from feature subsampling, which draws without replacement.
4. **Elementwise passes are order-free.** Subtraction, the gather, and
   `bin_equal_width`'s assignment compute each output from one input position,
   so a block split cannot move a value.
5. **`plan_tasks` remains a pure scheduling decision.** Its documented rules
   (grain floor, per-core cap, `MOJOBOOST_NUM_WORKERS` override, clamp to item
   count) are unchanged and its defaults are numerically what they were, which
   is what keeps `test_plan_tasks_respects_grain_and_cap` passing.
6. **No dispatch inside a dispatch.** The three dispatches of a build run in
   sequence, never nested. `Histogram.reset` stays serial precisely so a
   caller already inside a task has something safe to call.
7. **No `n_jobs` parameter was added anywhere.** The native control is
   `MOJOBOOST_NUM_WORKERS`, as it was. See "Future binding" below.

## Required external edits (none applied)

1. **`tree.mojo`: adopt the scratch form.** Add one `var pairs =
   List[Float64]()` next to the `_HistPool` in the grower and call
   `build_histogram_subset_into_scratch(hist, pairs, data, grad, hess, rows,
   start, count, tree_features)` in place of
   `build_histogram_subset_into(...)` at all three call sites: the bagged
   root, and the two children either side of the subtraction trick. (Cited by
   name rather than by line: `tree.mojo` is being edited by other lanes in
   this round and the line numbers have already moved once while this was
   written.) Without this the gather still happens, but its buffer is
   allocated and freed per node, which costs most of what the gather saves on
   small nodes. Same results either way. `tree_sparse.mojo` and
   `train_gpu.mojo` reach only `subtract_histogram` and are unaffected.
2. **`split.mojo`: hoist the totals pass.** `find_best_split` recomputes each
   feature's totals inline (the `vg`/`vh`/`vc` lane accumulators) and scans with
   `hist.grad[base + b]` list indexing rather than the pointer loads the rest
   of the package uses. `histogram.feature_totals` reproduces that computation
   in exactly the current order (vector lanes, `reduce_add`, then the scalar
   tail) so it can be swapped in without moving a gain. Two follow-ons, both
   for the split lane and both needing this same bit-exactness argument:
   switch the prefix scan to pointer loads, and parallelize the feature loop
   with `dispatch_feature_ranges` plus a deterministic best-of reduction (fold
   candidates in ascending feature order, so ties resolve as they do today).
3. **Padded bin stride.** Each feature's slice starts at `f * n_bins`, so with
   `n_bins = 255` no slice after the first is aligned to anything, and every
   vector store in the zeroing and subtraction loops straddles cache lines.
   A `bin_stride = round_up(n_bins, SIMD_LANES)` field on `Histogram`, used
   everywhere `f * n_bins` appears, would fix it. That touches `histogram.mojo`
   (mine), plus `split.mojo`, `tree.mojo`, `histogram_gpu.mojo`,
   `histogram_sparse.mojo`, `tree_sparse.mojo`, `train_gpu.mojo`,
   `distributed.mojo`, `backend.mojo`, `inspection.mojo` if it reads bins, and
   the tests that index histograms directly. It is a cross-cutting change, it
   is not obviously a win, and it must not be attempted before the alignment
   claim below has profiler evidence.
4. **Future narrow binding, not now.** If a Python-level worker control is
   ever wanted, it should be a thin setter over the existing environment
   contract (one call that sets `MOJOBOOST_NUM_WORKERS` for the process),
   never a parameter threaded through the training API, and never a Python
   policy: every decision in this lane has to be reachable from the Mojo hot
   path with no interpreter on the stack. Out of scope here by instruction.

## Commands

Equivalence gate, run first, before any timing is believed. These tests are
owned by other lanes and were not modified:

```
pixi run mojo run -I src tests/test_cpu_parallel.mojo
pixi run mojo run -I src tests/test_histogram_reference.mojo
pixi run mojo run -I src tests/test_mojoboost.mojo
pixi run mojo run -I src tests/test_feature_sampling.mojo
pixi run mojo run -I src tests/test_missing.mojo
pixi run mojo run -I src tests/test_categorical.mojo
```

The first two are the ones that matter: they assert bit-identical results
between the forced-serial and parallel paths and between the allocating and
`_into` builders. The last three cover the feature-subset, missing-bin, and
categorical paths through the rewritten accumulation.

Microbenchmarks:

```
pixi run bench-profile          # machine line, per-stage serial vs auto
pixi run bench-hist             # build and subtract
pixi run bench-hist-scaling     # full and subset builders across shapes
pixi run bench-threshold        # the driver PARALLEL_MIN_OPS came from
```

End to end:

```
pixi run bench                  # bench/bench_train.mojo, CPU training
pixi run mojo run -I src bench/bench_goss.mojo
```

Sweeps (three cells separate the gather from the pairing; the fourth cell does
not exist, because pairing is only used on the gathered path):

```
MOJOBOOST_CPU_COMPACT_MIN_ROWS=1000000000 MOJOBOOST_CPU_FEATURE_GROUP=1 pixi run bench
MOJOBOOST_CPU_COMPACT_MIN_ROWS=256        MOJOBOOST_CPU_FEATURE_GROUP=1 pixi run bench
MOJOBOOST_CPU_COMPACT_MIN_ROWS=256        MOJOBOOST_CPU_FEATURE_GROUP=2 pixi run bench

MOJOBOOST_CPU_TASKS_PER_CORE=1  pixi run bench-profile
MOJOBOOST_CPU_TASKS_PER_CORE=8  pixi run bench-profile
MOJOBOOST_CPU_CORE_POOL=performance pixi run bench-profile
MOJOBOOST_CPU_CORE_POOL=performance pixi run bench
```

`bench/apple/cpu_plan.json` holds the full matrix, the preconditions (idle
machine, warm-up discarded, five repetitions, median and spread), and the
falsification condition for each change.

## Notes for whoever integrates the round

- `parallel.mojo` gained one import (`apple_cpu_policy`) and one new function.
  Every name other lanes reach for is still exported and unchanged: `_env_int`
  (used by `gpu_tiling`, `gpu_runtime`, `gpu_active_rows`, `gpu_output_planes`,
  `apple_histogram_policy`, `histogram_cache_policy`, `hybrid_leaf_scheduler`),
  `dispatch_features` (`binning`, `sparse`, `histogram_sparse`),
  `dispatch_rows` (`boosting`), `plan_row_blocks` / `run_row_blocks` (`tree`),
  `PARALLEL_MIN_OPS`, `TASKS_PER_CORE`, `env_parallel_min_ops`, `plan_tasks`
  (tests and `bench/bench_profile.mojo`).
- The environment namespace is disjoint from the GPU lanes': everything this
  lane added is `MOJOBOOST_CPU_*`. `apple_histogram_policy.mojo`, which landed
  in this same round, is a GPU launch policy despite the similar name and
  shares nothing with `apple_cpu_policy.mojo`.
- Import direction is one-way: `apple_cpu_policy` imports only `std`;
  `parallel` imports it; `histogram` and `binning` import both. That is why
  `_env_int` is duplicated in the policy rather than imported from `parallel`.

## Claims requiring profiler or assembly evidence

Nothing below has been verified. Each is stated as a hypothesis with the
evidence that would settle it.

1. **That any of these loops vectorize at all.** `SIMD[DType.float64, W]`
   stores are written explicitly in the zeroing and subtraction kernels, but
   whether they lower to NEON stores, and whether the accumulation loops
   auto-vectorize anything, needs disassembly of the built module. No ARM
   instruction is claimed to be emitted anywhere in this lane's code or
   comments.
2. **That feature pairing overlaps two dependency chains.** Requires either a
   cycle-level profile showing reduced store-to-load stalls, or disassembly
   showing the two chains interleaved. If it shows nothing, set
   `DEFAULT_FEATURE_GROUP = 1`; the paired loops then become dead and should be
   deleted rather than left as an untaken branch.
3. **That the gather removes cache misses rather than merely moving them.**
   Needs L1/L2 miss counters on the subset builder, at a node large enough that
   the gradient buffer exceeds L2, with and without the gather.
4. **The assumed constants.** `ASSUMED_L1D_BYTES` (64 KB) and
   `ARM_CACHE_LINE_BYTES` (128) are assumptions, labelled as such in the
   module. They gate `plan_feature_group`, so if the real L1 is larger the only
   consequence is a group of 2 where 4 might have fit, which is bounded by
   `MAX_FEATURE_GROUP` anyway.
5. **That misalignment of feature slices costs anything.** The padded-stride
   proposal above rests on it. Measure before touching ten files.
6. **That `TASKS_PER_CORE = 4` is right, on any machine.** It was a starting
   point before this lane and it still is.
7. **Whether the runtime's fan-out is work-stealing or static.** Everything
   said above about over-decomposition absorbing slow cores depends on it, and
   the `MOJOBOOST_CPU_TASKS_PER_CORE` sweep is what answers it: if 1 performs
   like 8, the pool is static and the core-pool question becomes much more
   interesting.
8. **The serial-vs-parallel crossover itself.** `PARALLEL_MIN_OPS` was measured
   against the old estimates. The estimates changed; the constant did not. Rerun
   `bench-threshold` before defending either.
