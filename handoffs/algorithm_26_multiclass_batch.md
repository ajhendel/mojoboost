# Handoff: multiclass and multi-output tree batching (algorithm task 26)

Primitives for growing or evaluating several independent class trees inside
one softmax boosting round: batched gradient planes, a batched magnitude
reduction, two batched histogram kernels, a class-batch planner with the
memory formulas, and a host-side guard for the round's one real
synchronization barrier. Implemented, wired into nothing, and **not compiled
and not tested** (this lane was scoped to static reasoning and isolated
implementation, with tests and builds explicitly out of scope). Read the
"Unverified" section before you trust a line of it.

The point of the lane. A softmax round on the device path costs, per class:
one gradient fill, one magnitude reduction, one host synchronization to read
two floats back, one whole tree's worth of histogram launches, and one score
update. At ten classes that is ten full re-reads of the binned matrix for the
ten root histograms alone, and ten queue drains that each idle the device for
the length of a round trip. The classes of a round are independent by
construction, so almost none of that has to be serial. What the classes share
at a round's root is exactly what makes batching possible: the same rows, in
the same order, from the same pre-round score matrix.

## Files this lane owns

| File | State |
|---|---|
| `src/mojoboost/gpu_output_planes.mojo` | New. Layout, eligibility, memory formulas, the class-batch planner, and the ordering invariants. Pure host arithmetic, no device. |
| `src/mojoboost/gpu_multiclass_batch.mojo` | New. Four kernels, `GpuClassBatch`, `MulticlassRoundGuard`. |
| `bench/apple/multiclass_batch_plan.json` | New. The benchmark plan: cases at 3, 10, 100, and imbalanced classes, with every measurement null and a reason. |
| `handoffs/algorithm_26_multiclass_batch.md` | This file. |

Nothing else was touched. No edit to `train_gpu.mojo`, `boosting.mojo`, any
objective module, `model.mojo`, `histogram_gpu.mojo`, `gpu_active_rows.mojo`,
`gpu_objectives_native.mojo`, the Python package, the bindings, any test, any
doc, packaging, or a workflow. This lane committed nothing.

Note on `git status`. This is a shared checkout and another lane committed
while these files were being written: commit `b04b5f0` ("Integrate parallel
release and accelerator work") swept up `gpu_output_planes.mojo` whole and a
mid-write snapshot of `gpu_multiclass_batch.mojo`. The latter therefore shows
as *modified* rather than untracked, and the pending diff is this lane's own
post-write corrections and nothing else: `Int(b)` at the shared-memory
indices that combine a bin with a class offset, the trailing-comma fix on the
two inferred-parameter lists, an explicit transfer out of `for_plan`, the
`last_counts_shared` field and its two assignments, a dequantization guard in
`histogram_for`, and one line rewrapped under 80 columns. No other lane's
work is touched by it.

## Unverified

Nothing here has been compiled, run, or measured. Specifically:

- The four kernels have never been launched. They were written against the
  proven shapes in `gpu_active_rows._range_hist_atomic_kernel` and
  `gpu_objectives_native._abs_sum_kernel` / `_softmax_class_kernel`, and the
  arithmetic per (row, class) is a transcription of those, but a
  transcription is not a test.
- The one structural risk in the shared kernel is its shared-memory
  allocation: `SHARED_CLASS_CAP * MAX_BINS` Int32 twice plus `MAX_BINS`
  Int32, which is 9216 bytes at the cap. That is inside the 16 KiB fallback
  in `gpu_tiling`, but whether Metal accepts three `stack_allocation`s of
  that size in one threadgroup alongside the launch's other state is a
  question a build answers, not this file.
- Index-type mixing in the kernels (`Int` against the `block_idx`/`thread_idx`
  types) follows the existing kernels' patterns and adds `Int(b)` where a
  shared-memory index is combined with a class offset. If the compiler
  disagrees anywhere, it will disagree at those exact lines.
- Every number in the benchmark plan's `derived` blocks was computed by hand
  from the formulas in `gpu_output_planes.mojo`. They are arithmetic, not
  measurements, and a first run should check a few of them against
  `class_batch_bytes` rather than assume.

## The layout decision

Two per-(row, class) quantities, opposite layouts, and the module makes the
split explicit rather than by convention:

| Quantity | Layout | Why |
|---|---|---|
| raw scores, probabilities, predictions | row-major `x[r * n_classes + k]` | every consumer reduces over classes within a row: the softmax max-subtraction and denominator, the log loss of the true class, the argmax. Also the serialized contract: `MulticlassBooster`, `GpuObjectiveState.raw_dev`, and `gpu_predict._predict_kernel` all already use it. Unchanged and unchangeable. |
| gradients, hessians | class-major `x[k * n_rows + r]` | every consumer reduces over rows within a class: the histogram kernels, the magnitude reduction, the leaf-value sums. A class's plane is then the contiguous `Float32[n_rows]` those kernels already take, at offset `k * n_rows`, so batching costs them a base-pointer offset and nothing else. |

Row-major gradients were the alternative and they lose twice: every histogram
kernel would need a stride argument threaded through it, and the row loop
would read gradients with stride `n_classes`, which is the one access pattern
a coalesced Float32 load cannot absorb. The transpose therefore happens
exactly once per round, inside `_batch_softmax_grad_kernel`, which reads
row-major probabilities and writes class-major derivatives. That is the whole
layout design.

## Achievable parallelism, and where it stops

Three things can be shared across a batch of class trees, and they are worth
separating because they have very different sizes and very different
preconditions.

**1. Bin reads (large, root only).** Available only when the batched classes
read the same rows in the same order through the same active feature set.
That holds at a round's root: `_boost_rounds_multiclass` draws one bag or one
GOSS sample per round *before any class's tree*, precisely so every class is
grown on the same rows. It stops holding one split down, because each class
chooses its own split. When it holds and `classes_per_block` classes share a
threadgroup, the round reads the binned matrix
`ceil(n_classes / classes_per_block)` times instead of `n_classes` times.
This is the only saving that scales with `n_rows * n_features`.

The ceiling is shared memory: each class needs a gradient and a hessian bin
plane, `2 * n_bins * 4` bytes, and the kernel statically allocates for
`SHARED_CLASS_CAP = 4`. At 256 bins that is 2 KiB per class; a 32 KiB device
would fit 15 by arithmetic and gets 4, because the cap is what the kernel
allocates. Raising it needs a bin-capacity-parameterized kernel, which is
`gpu_histogram_specializations.mojo`'s subject, not a constant here. At 64
bins the same cap would be worth revisiting first: that is where the ceiling
is furthest from the arithmetic.

**2. Bin counts (a third of the output, root only).** A histogram's count
plane does not depend on the class, so a batch over a shared row window needs
one count plane, not `k_count`. That is a third of the output bytes and a
third of the shared memory, and the shared memory is what sets the ceiling
above, so this is not merely a byte saving.

**3. Launches and host synchronizations (always).** Independent of any
sharing. `n_classes` magnitude reductions with `n_classes` queue drains
become one launch and one readback of 2 KiB per class. `n_classes` small
deep-node histogram launches become one launch with `n_classes` times the
blocks. At depth, where a node's own row count is far too small to fill the
device, this is the only lever left.

**Where it stops.** Feature subsampling kills bin sharing outright: the
per-tree draw is seeded `round * n_classes + k`, so the classes of a round
scan different columns and no threadgroup can serve several of them from one
bin read. That is not a bug to route around; it is what the seed means, and
changing it would change which features every tree sees on both backends. A
subsampled run therefore gets launch sharing only, which is the honest lower
bound for most real training runs and is why
`bench/apple/multiclass_batch_plan.json` measures it as its own case
(`mc10_root_subsampled`) rather than folding it into an average.

**What does not vary with the class prior.** Softmax grows one tree per class
over *every* row, whatever the class frequencies, so a rare class's tree does
not read fewer rows and does not cost less to histogram. Imbalance changes
gradient magnitudes, and through them the per-class fixed-point scale and how
early a class's tree stops splitting. The load-balance problem in an
imbalanced batch is therefore raggedness in *depth*, not in width. See the
class policy below.

## Memory formulas

All in `gpu_output_planes.mojo`, all pure arithmetic. For a batch of `K`
classes over `n_rows` rows, `F` features, `B` bins, `T` row tiles:

```
gradient planes    8 * n_rows * K                       (Float32 grad + hess)
row indices        12 * n_rows * K, or 0 if shared      (rows, scratch, offsets)
histogram output   4 * P * F * B                        P = 2K+1 shared counts
                                                        P = 3K   otherwise
partial histogram  4 * T * P * F * B                    0 for the shipped
                                                        atomic-only batched path
magnitude partials 2048 * K                             2 * SUM_BLOCKS Float32
score matrix       8 * n_rows * n_classes               resident all run, not
                                                        per batch
```

The shape of that is one fact: the first two terms scale with `n_rows` and
the rest do not. At a million rows the gradient planes and the row indices
outweigh the histogram output by two orders of magnitude, so
`plan_class_batches` bounds the batch by rows and never by the histogram
shape. Worked values, from the benchmark plan:

| Case | K | rows | shared rows | total |
|---|---|---|---|---|
| 3 classes, root | 3 | 100k | yes | 2.8 MB |
| 10 classes, root | 10 | 1M | yes | 82 MB |
| 10 classes, depth | 10 | 1M | no | 203 MB |
| 100 classes, root | 100 | 1M | yes | 821 MB, over budget |

The last row is the case that matters: at a hundred classes and a million
rows the round does not fit the default 512 MiB budget, and the planner
splits it into batches of 65 and 35. The split is deterministic (contiguous
ascending runs), so it changes when work happens and not what is computed.
A caller that wants the whole round resident raises
`MOJOBOOST_GPU_CLASS_BATCH_BYTES`; a caller that forces
`MOJOBOOST_GPU_CLASS_BATCH` past the budget gets an error rather than a
silently smaller batch, because a benchmark that asked for a geometry must
not be handed a different one.

The score matrix is deliberately outside the batch budget. A round's
probabilities are a reduction over every class of a row, so the whole
`n_rows * n_classes` matrix is resident for the whole run whatever the batch
size. At a hundred classes and a million rows that is 800 MB of raw scores
and probabilities before any batching, which is the real ceiling on class
counts on a unified-memory machine and is a fact about softmax on the device,
not about this lane.

## Synchronization points

The contract, stated once: **every class gradient of a round derives from the
same pre-round score matrix.**

The mechanism is the probability snapshot, not the ordering of score updates.
`refresh_softmax` materializes `prob` from `raw` once per round; the per-class
gradient kernel reads `prob` and never `raw`; so a class tree that commits its
own score update mid-round cannot perturb any other class's gradients. That is
already true of the sequential path, which commits per class immediately, and
it stays true under batching.

That leaves exactly one barrier and one freedom.

- **Barrier.** No `refresh_softmax` may run until every class tree of the
  previous round has committed into `raw`. `MulticlassRoundGuard.open_round`
  is that rule and refuses to open a round with pending commits.
- **Freedom.** Within a round, per-class commits may happen in any order or
  all at the end. Each class writes the disjoint slots
  `raw[r * n_classes + k]`, so concurrent commits neither race nor reorder a
  floating-point addition. Determinism does not require committing in class
  order and this lane does not impose one.

The other synchronizations, in the order a round meets them:

| Point | Sequential | Batched |
|---|---|---|
| probability snapshot | 1 per round | 1 per round, unchanged |
| magnitude readback (drains the queue) | `n_classes` | 1 per batch |
| histogram download per node | 1 per class per node | 1 per batch per node |
| score commit | 1 per class | 1 per class, order free |
| round boundary | implicit | `close_round` / `abandon_round` |

`abandon_round` exists because both trainers drop a whole round's trees when
no class made progress. Dropped trees never reached the raw scores, so nothing
is owed; the guard refuses to abandon a round that already committed one.

## Integration requirements

This is the wiring for whoever integrates. None of it was done here, because
`train_gpu.mojo` is outside this lane's ownership.

1. **Allocate once per session.** Build a `ClassBatchPlan` with
   `plan_class_batches(n_classes, n_rows, n_features, n_bins, n_tiles,
   caps.max_shared_memory_per_block, eligibility)` and a `GpuClassBatch` with
   `GpuClassBatch.for_plan(ctx, ...)`, on the histogram builder's context so
   the batch, the objective state, and the builder share one queue. Call
   `check_batch_order(plan)` once; it is cheap and it is the invariant
   everything else rests on.

2. **Allocate for the harder case.** Construct with `shared_counts False`
   (the `3 * cap` output) if the session will use both histogram kernels. The
   shared kernel writes into that layout too and `GpuClassBatch` tracks which
   kernel wrote last, so extraction stays correct; the reverse is not true,
   because a `2 * cap + 1` output has no room for per-class counts.

3. **Per round.** `guard.open_round()`, `state.refresh_softmax(ctx)`,
   `guard.note_probs()`. Then per batch `b`: `batch.fill_gradients(state,
   plan.batch_begin(b), plan.batch_count(b))`,
   `guard.note_gradients(...)`, `batch.refresh_scales(k_count)`. That is the
   round's one readback for the batch.

4. **Per level.** At the root, if `eligibility.bin_reads_shared()`, call
   `set_features_uniform` and then `enqueue_shared_histogram` with the
   builder's `bins_dev` pointer and the round's active-row buffer, then
   `download()` and `histograms(k_count)`. Below the root, call
   `set_features` per slot, `set_windows` with each class's current
   `LeafRange`, and `enqueue_ranged_histogram` with `rows_stride = n_rows`.

5. **What the integrator has to build that this lane did not.** Growing `K`
   trees concurrently needs `K` active-row permutations and `K` frontiers.
   `GpuActiveRows` is per-tree today (`GpuHistogramBuilder` owns exactly
   one), and `grow_tree_gpu` drives one tree from root to leaves inside one
   call. The batched primitives assume a caller that has restructured tree
   growth into a level-synchronous loop over `K` trees; that restructuring is
   the integration, and it is larger than these primitives. The cheap first
   integration is therefore **root-only**: batch the root histogram of a
   round's classes, then hand each class's root histogram to the existing
   sequential `grow_tree_gpu`, which re-derives it. That wastes one histogram
   per class and still removes `n_classes - ceil(n_classes / per_block)` full
   matrix reads. It is a strictly smaller change and it is where a first
   measurement should come from.

6. **Per-class scales must reach the split search.** A device-side split
   search reads the fixed-point planes directly, so it needs the class's
   inverse scale; `scale_of(slot)` and `hess_scale_of(slot)` are there for
   that. Handing it the wrong slot's scale is silent and produces a plausible
   wrong split, so this is the first thing to assert in an integration test.

7. **What must not change.** `round_tree_slot(round, k, n_classes)` and
   `tree_feature_seed(round, k, n_classes)` are both `round * n_classes + k`
   and both are contract: the serialized tree order, the prediction shapes,
   and the feature subsample each tree draws. Batching changes when a tree is
   grown, never where it is stored or what it sees.

## Quality invariants

Exact, and checkable:

1. **Histogram parity.** A batched class histogram is cell-for-cell identical
   to the sequential path's for that class at that node, in fixed-point Int32,
   before dequantization. Accumulation is integer throughout and integer
   addition is associative, so neither the batched shared-memory layout nor
   the order blocks flush in can change a cell.
2. **Scale parity.** A class's fixed-point scale is bit-identical whether it
   was reduced alone or in a batch of a hundred. `_batch_abs_sum_kernel` uses
   the same `SUM_BLOCKS` blocks, `SUM_THREADS` threads, grid stride, and
   shared-memory tree reduction as `_abs_sum_kernel`, per class, and the host
   sums the partials in ascending block order in Float64 as before.
3. **Model identity.** The serialized ensemble is byte-identical to the
   sequential path's, at any batch size, including a split round.
4. **Shape identity.** Raw scores and probabilities stay row-major
   `x[r * n_classes + k]` on host and device. No prediction shape moves.
5. **Ordering.** Results leave `GpuClassBatch.histograms` in ascending slot
   order, which `check_batch_order` proves is ascending class order, whatever
   order slots complete in.
6. **Snapshot integrity.** No class's gradients derive from a score matrix
   any of the round's own trees advanced. `MulticlassRoundGuard` refuses the
   orderings that would break it.
7. **Weights.** A zero-weight row produces a zero gradient and a zero
   hessian and is invisible to every batched histogram, exactly as in the
   single-class kernel; the batched kernel multiplies in the same operand
   order.

The one thing that is *not* invariant, and should not be claimed as one:
agreement with the CPU trainer is to Float32 precision, not bit-exact,
because the device carries scores and derivatives as Float32. That is the
existing trade in `gpu_objectives_native.mojo` and this lane neither widens
nor narrows it.

## Small-class versus large-class policy

Stated as findings rather than as a knob, because the crossover is a
measurement this lane was not allowed to take.

- **Class frequency does not change per-class work.** Every class's tree is
  grown over every row. A batch is therefore never ragged in *width*.
- **Class frequency changes tree depth.** A rare class has small
  probabilities, hence a small hessian `2p(1-p)`, hence candidates that fail
  `min_child_hess` and `min_data_in_leaf` earlier. Its tree stops splitting
  while the majority class's is still growing.
- **A ragged batch wastes grid, not memory.** `enqueue_ranged_histogram`
  sizes `grid.y` by the largest window in the batch, so a class with a
  quarter of the rows leaves three quarters of its tiles idle, and a class
  that has finished entirely leaves all of them idle (its `count` is zero and
  its blocks return immediately, which is cheap but not free).
- **The obvious fix is compaction, and it is not obviously worth it.**
  Dropping finished classes from the batch means re-packing slots, which
  means the slot-to-class map is no longer `k_begin + slot`, which is the
  thing every ordering guarantee in this lane rests on. It can be done (keep
  an explicit slot-to-class table and sort results by class before
  collecting), but it should be done only against a measurement that says the
  idle tiles cost more than the bookkeeping. `mc10_imbalanced` in the
  benchmark plan sizes it.
- **Small class counts do not need a policy.** At three classes the whole
  round fits one threadgroup and one batch; the interesting shapes start
  where `classes_per_block` binds (above 4) and again where the memory budget
  binds (around 65 classes at a million rows).

## Benchmark cases

`bench/apple/multiclass_batch_plan.json` is the plan. Seven cases:

| Case | What it decides |
|---|---|
| `mc03_root_shared` (3 classes, 100k rows) | Whether launch and synchronization savings alone show at the smallest real multiclass shape. One bin pass instead of three. |
| `mc10_root_shared` (10 classes, 1M rows) | The first shape where `classes_per_block` binds rather than the class count: three bin passes instead of ten, one readback instead of ten. |
| `mc100_root_budget_bound` (100 classes, 1M rows) | Whether the memory formula splits the round where it says it does, and whether a split round costs anything beyond the class count. Runs again at a doubled budget to separate the two. |
| `mc10_root_subsampled` (colsample 0.5) | What launch sharing is worth with bin sharing unavailable. The honest lower bound for real runs. |
| `mc10_depth_frontier` (levels 1 and below) | The crossover: at what node size the batched ranged kernel stops beating the sequential per-class loop. Nothing in the Mojo picks a depth until this answers. |
| `mc10_imbalanced` (97 percent in one class) | Per-class depth spread, per-class scale spread, and the idle-tile fraction. The evidence for or against batch compaction. |
| `determinism_repeat` | Byte-identical models across repeats and against `MOJOBOOST_GPU_CLASS_BATCH=1`. Not a timing case. |

The plan refuses to report a speedup for any configuration whose batched and
sequential models are not byte-identical, and refuses to average across
cases: they move different quantities (bin passes, synchronizations,
launches) and an average would hide which one paid.

## Scope discipline

Softmax multiclass only. Nothing here was generalized to arbitrary
multi-output objectives, and it should not be until such an objective has a
real contract, because the primitives lean on three softmax-specific facts:
one shared probability matrix per round, per-class one-vs-rest derivatives of
that matrix, and one tree per class per round in a fixed serialized slot. A
multi-output objective that broke any of the three would need a different
design, not a wider parameter. The place that would have to change first is
`OutputPlanes`, which is written in terms of outputs rather than classes
precisely so that conversation can start from something concrete.
