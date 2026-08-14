# Handoff: fused objective, sampling, and root histogram pipeline (task 23)

The start of a GPU boosting round, examined for fusion, and the two
primitives that survived the examination. Two designs were rejected on
structural grounds rather than on timings, one was already right, and two
narrow primitives were built.

**Read this first.** Nothing in this lane was compiled, run, tested, or
benchmarked. The task forbade executing Mojo, Pixi, Python, builds,
benchmarks, and profilers, so the two new modules have never been through
the compiler and the numbers in `bench/apple/fused_round_plan.json` are
arithmetic from a model, not measurements. Treat the code as a reviewed
draft and expect a compile pass to be the first thing integration does.
Everything below that is a claim about behavior is a claim about what the
source says, not about what a run showed.

## Files this lane owns

| File | State |
|---|---|
| `src/mojoboost/gpu_fused_round.mojo` | New. Round-start drivers, the split magnitude reduction, the eligibility gate, and the traffic model. |
| `src/mojoboost/gpu_gradient_stream.mojo` | New. Device row-selection metadata, the GOSS ranking plane, and single-pass host staging. |
| `bench/apple/fused_round_plan.json` | New. What would have to be measured before any of this is called faster. |
| `handoffs/algorithm_23_fused_round.md` | This file. |

Nothing else was touched. No edit to `gpu_objectives_native.mojo`,
`histogram_gpu.mojo`, `gpu_active_rows.mojo`, `train_gpu.mojo`,
`objective.mojo`, `boosting.mojo`, `goss.mojo`, `bagging.mojo`,
`sampling.mojo`, `gpu_runtime.mojo`, `__init__.mojo`, any test, any Python,
any binding, any workflow.

This lane committed nothing. `gpu_gradient_stream.mojo` is nonetheless
already in commit `b04b5f0` ("Integrate parallel release and accelerator
work"), which a concurrent lane made with a whole-tree add while this one
was still writing; the edits after that commit, and the other three files,
are working-tree only. Nothing was lost and nothing needs undoing, but do
not read that commit as a review of this code.

The two new modules are not exported from `src/mojoboost/__init__.mojo`,
because that file belongs to another lane; adding the two lines is part of
integration.

## The shape of a round, as it stands

For a built-in objective with no row sampling, `_train_gpu_rounds` issues
this per round and per tree:

1. `fill_gradients_device` runs `_grad_hess_kernel` over every row, writing
   `grad[n]` and `hess[n]` into the histogram builder's own buffers.
2. The same call runs `magnitude_sums`, which reduces `|grad|` and `|hess|`
   into 2 KB of threadgroup partials, copies them back, **and
   synchronizes**, then derives the two fixed-point scales on the host.
3. `begin_tree` writes the root's active-row permutation with an iota
   kernel.
4. The root histogram reads, per active feature, one Int32 row id, one
   UInt8 bin, and both Float32 derivatives, for every row of the node.
5. Every node below the root does step 4 again over its own compacted
   range, gathering the derivative planes at permuted row indices.

Step 5 is the fact that decides this whole lane.

## The three designs

### Materialized gradient planes: recommended, and not as a compromise

Step 5 gathers `grad[r]` and `hess[r]` at permuted row indices for every
node of the tree. The planes have to exist for the whole tree whatever the
root does. No design that removes them can be correct, so every alternative
below is competing for the root alone, which is one node out of
`2 * num_leaves - 1`.

### Tiled production consumed immediately by the root histogram: rejected

Produce the gradients a row tile at a time and hand each tile straight to
the root histogram for those rows.

It saves no bytes. The planes are still written, because the nodes below
the root read them, so the gradient term is unchanged and the histogram
term is unchanged.

It saves no time on the queue the trainer runs on. Everything is enqueued
in order on one `DeviceContext`, so a tile's histogram launch cannot begin
before that tile's gradient launch has retired. The overlap the design
exists for is not available to ask for.

It costs one launch per tile. `streamed_round_traffic` returns the
materialized design's byte terms and a larger launch count, which is the
whole finding stated in code.

Reconsider only if the backend grows concurrent queues with explicit
events, **and** a trace shows a gap between the gradient kernel and the
first histogram kernel large enough to be worth hiding. The plan file names
both.

The streaming idea does pay, just not on the device. On the host-origin
path (custom objectives) `stage_gradients` reads the same two `Float64`
lists three times: once for the gradient magnitude sum, once for the
hessian magnitude sum, once to convert into pinned Float32. That is where
`HostGradientStage` went.

### Kernel fusion: rejected, and not on bandwidth grounds

The refusal is a circular dependency, and no measurement can move it.

The histogram accumulates in fixed point. The fixed-point scale is a global
reduction over the very gradients being accumulated (`_fixed_scale`,
`device_fixed_scale`). Every gradient in the dataset therefore has to exist
before the first bin of the first histogram can be quantized. A kernel that
produced gradients and consumed them in the same launch would need its
scale before it had computed the sum the scale comes from.

A second hazard sits behind that one. `grid.x` is the active feature, so a
fused kernel visits each row once per feature. Any magnitude accumulation
folded into it would over-count every row by the active feature count
unless restricted to one feature slot, and restricting it to one slot
reintroduces a cross-threadgroup ordering dependency that a single launch
cannot express.

The bandwidth arithmetic agrees, at the model's most favorable framing
(fusion credited with never writing the planes at all, which no real tree
can have). At a million rows, fifty features, unweighted:

| | materialized | fused ceiling |
|---|---|---|
| gradient bytes | 16.0 MB | 0 |
| magnitude bytes | 8.0 MB | 8.0 MB |
| root histogram bytes | 650.3 MB | 650.3 MB |
| total | 678.3 MB | 662.3 MB |
| per-row objective evaluations | 1x | 51x |

Fusion removes 2.4% of the round's bytes and multiplies objective
evaluations by 51. Add a sample weight and the fused histogram reads three
Float32 per row per feature where the materialized one reads two: the total
goes from 682.3 MB to 866.3 MB, a 27% **increase**. Narrow the dataset to
ten features and the ceiling rises to 10.1%, still against eleven times the
objective work. Amortized over a 31-leaf tree, whose histogram passes are
roughly six root-sized passes, the whole round start is about 0.7% of the
tree.

None of that is worth a second definition of the objectives, which is what
a fused kernel would be: the per-objective derivative chain lives in
`_grad_hess_kernel` and would have to be repeated inside the histogram
kernel.

Reconsider only if histograms stop accumulating in fixed point, which would
give up bit-determinism across run orders. That is a much larger decision
than this lane.

## What was implemented

### 1. `MagnitudeReader`: the magnitude reduction, split

`GpuObjectiveState.magnitude_sums` does the kernel, the copy, and
`ctx.synchronize()` in one call. That forces the round's only
device-to-host wait to happen before anything else in the round is
enqueued, so the seeding kernel for the root permutation, which depends on
nothing the scale produces, runs after the wait instead of during it.

`MagnitudeReader.enqueue` issues the kernel and the readback and returns.
`read` waits and sums. Between them the caller enqueues whatever does not
depend on the scale.

The reduction kernel is `_abs_sum_kernel` reproduced expression for
expression, with the same fixed grid-stride, the same fixed block count,
and the same shared-memory tree, so its partials and their host-side
Float64 total are bit-identical and every scale and histogram derived from
them is unchanged. **That duplication is a debt, not a feature.**
Integration should keep one definition. The repository already carries the
same shape of debt in `gpu_active_rows._range_reduce_kernel`, whose
docstring says the same thing.

The honest expected win is small and bounded by one launch's latency per
round. The plan file makes it conditional on a trace showing a nonzero idle
interval, and says to keep the split for the ordering freedom rather than
for a speed claim if the interval is already zero.

### 2. `GpuRowSelection`: row-selection metadata on the device

This is the primitive that removes a whole-path fallback rather than a
pass, and it is the largest thing in the lane.

`device_gradients` in `train_gpu.mojo` refuses the device objective path for
any sampled run:

> row sampling draws its sample from host-side gradients, so bagging and
> GOSS cannot take the device objective path

**That reason is true of GOSS and false of bagging.** `sample_rows` in
`bagging.mojo` draws from a counter stream keyed by
`(seed, bag index, row)` and never reads a derivative. A bag needs no
gradients at all. The two are refused together for one shared stated
reason, and only one of them has it.

Bagging is blocked, but by something else, and the lane found exactly one
thing: bagging semantics update **every** row's raw score after every tree,
in bag or not, so that out-of-bag rows carry correct gradients into later
rounds (`bagging.mojo` module docstring says so explicitly). The device
update walks leaf ranges, which by construction cover only in-bag rows.
`update_raw`'s own docstring already flags this. Until out-of-bag rows are
scored on the device, a bagged device round would train later rounds on
stale derivatives, which is worse than merely different ones.

`round_eligibility` in `gpu_fused_round.mojo` records the two reasons
separately, as `ROUND_BAGGING_OUT_OF_BAG` and `ROUND_GOSS_RANK_PRECISION`,
with `round_eligibility_reason` giving a sentence a caller can raise
verbatim.

What the selection provides:

- `rows_dev` and `scale_dev`, the selected row ids and their compensation
  multipliers, staged through pinned buffers and uploaded once per round.
- `apply_compensation`, one kernel over the selection (not over the
  dataset), scaling both derivatives in the same operand order as
  `apply_goss_scaling`. Entries at exactly 1.0 are skipped, which is a
  bandwidth choice and not a numerical one; the kept high-gradient rows are
  all at 1.0, so the skip covers most of a GOSS selection.
- `enqueue_importance` and `download_importance`, the device-side
  `|grad * hess|` ranking plane, so a host sampler ranks rows from
  device-resident gradients instead of needing a host-side objective
  evaluation to produce them.

The selection rule itself stays in `goss.mojo`, untouched: the quickselect
threshold, the forward pass, the counter stream, the multiplier. That is a
ranking over the whole row set and an order-dependent forward pass, not a
per-row map, and moving it would change the sample.

### 3. `HostGradientStage`: one pass instead of three

Where gradients genuinely originate on the host, which is the
custom-objective contract and nothing else, `stage` converts both planes
into pinned Float32 while accumulating both magnitude sums in Float64, in
one pass. The sums accumulate `abs(values[i])` in ascending index order,
which is exactly what `_fixed_scale` accumulates, so `device_fixed_scale`
applied to them returns the same Float32 scale bit for bit. Only the number
of reads changed: 40 bytes per row down to 24.

Chunked overlap of conversion with transfer was **not** built. It needs a
copy into a sub-range of a device buffer, and the copy entry points reached
from here take whole buffers. That is the follow-up.

### 4. `RoundTraffic` and the three traffic functions

The paper comparison, executable. `materialized_round_traffic`,
`streamed_round_traffic`, and `fused_round_traffic` return byte terms,
per-row objective evaluations, and launch counts for any shape;
`fusion_delta_bytes` and `fusion_recompute_factor` are the two headline
numbers. Pure host arithmetic, no device needed, so a reviewer can
re-derive the table above for their own shape rather than trusting this
document.

The model counts issued traffic with no cache modeled, deliberately. The
entire fusion question is whether re-reading the derivative planes once per
feature costs anything, and a model that assumed those reads were
cache-served would answer the question by assumption. Which fraction
actually reaches memory is a counter reading, and the plan file demands it
before the interleaved-plane follow-up is built.

## Wiring it in

None of this is called from anywhere. The integration is in `train_gpu.mojo`
and `histogram_gpu.mojo`, neither of which this lane may edit.

### The plain round, with the seeding launch moved ahead of the wait

Replace the `builder.fill_gradients_device(state, objective, alpha)` line
in `_train_gpu_rounds` with:

```mojo
enqueue_gradients(
    builder.ctx, state, objective, alpha, builder.grad_dev, builder.hess_dev
)
enqueue_magnitudes(
    builder.ctx, selection, mags, builder.grad_dev, builder.hess_dev
)
builder.begin_tree()          # does not depend on the scale
var scales = mags.read_scales(builder.ctx)
builder.g_scale = Float64(scales.g_scale)
builder.h_scale = Float64(scales.h_scale)
builder.has_gradients = True
```

`grow_tree_gpu` calls `builder.begin_tree` itself today, so this ordering
needs either a `begin_tree` that is idempotent within a tree or a grower
entry point that skips it. That is the one piece of the wiring this lane
could not make clean from outside, and it is the reason the drivers also
ship a `round_start` convenience that does the whole thing in one call for
a caller with nothing to interleave.

Cleaner still, and what integration should probably do: give
`GpuHistogramBuilder` a `fill_gradients_device_deferred` /
`finish_gradients_device` pair that owns the `MagnitudeReader`, so the
builder's `g_scale`, `h_scale`, and `has_gradients` stay private. The
drivers here work on buffers rather than on the builder precisely so that
either shape is available.

### A GOSS round on the device

```mojo
enqueue_gradients(ctx, state, objective, alpha, grad_dev, hess_dev)
selection.enqueue_importance(ctx, grad_dev, hess_dev)   # before compensation
var imp = selection.download_importance(ctx)            # one sync, 4n bytes
var sel = goss_select(imp, goss, i)                     # unchanged host rule
selection.from_goss(ctx, sel)
enqueue_magnitudes(ctx, selection, mags, grad_dev, hess_dev)
var scales = mags.read_scales(ctx)
builder.begin_tree(sel.rows)
```

Three ordering constraints, all load-bearing:

- The ranking plane is filled **before** compensation. The sampler ranks
  the derivatives the objective produced; the multiplier it hands back is a
  consequence of that ranking, not an input to it.
- Compensation is applied **before** the magnitude reduction. The
  fixed-point scale has to bound the values the histograms will read, which
  under GOSS are the scaled ones. Reducing first would produce a scale too
  large by up to the multiplier and give up the Int32 overflow guarantee.
  This is also the host path's order: `goss_round` scales, then
  `upload_gradients` derives the scale from the scaled values.
- The reduction covers **every** row, including rows this round's sample
  left out, which is what `_fixed_scale` does over the whole gradient list.
  That is the conservative side, since a histogram only reads a subset.

### Multiclass

`state.refresh_softmax(ctx)` once per round, before the first class, then
`softmax_round_start` (or the split pair) per class. Do not refresh per
class: the probabilities are normalized across classes inside that one
kernel, they are shared by every class of the round on the host path too,
and by the time class `k+1` runs, trees grown earlier in the same round
have already advanced the raw scores, so a per-class refresh would change
the shared-sample semantics rather than just cost `n_classes` times the
exponentials.

A sampled softmax round shares one selection across every class, which is
what `_multiclass_goss_select` does on the host.

**Rejected while looking at multiclass:** batching all `n_classes` gradient
plane pairs so one reduction and one synchronization covers the round. It
costs `8 * n_classes * n_rows` bytes of device memory (80 MB at a million
rows and ten classes) to remove `n_classes - 1` synchronizations out of the
`n_classes * num_leaves` a round already pays for its histogram downloads,
which is a few percent of the round's waits. Not worth the memory.

## Parity and precision

What is preserved exactly:

- **Sample weights.** Untouched. The weight multiplies both derivatives
  inside `_grad_hess_kernel`, which this lane does not modify, and the
  compensation kernel multiplies on top of that, which is the host order.
- **Class weights and softmax normalization.** Untouched. The probability
  kernel subtracts the row max and normalizes across classes exactly as
  `_softmax_inplace` does, and the drivers only call it.
- **Bagging determinism.** Untouched. Bags come from `bagging.mojo` and are
  consumed by `begin_tree(bag)` as before; the selection is a device image
  of a host decision, never a device decision.
- **GOSS compensation.** The multiplier, the threshold rule, the counts,
  and the warmup are all still `goss.mojo`.
- **Custom-objective fallback.** Untouched and explicitly gated:
  `round_eligibility` returns `ROUND_CUSTOM_OBJECTIVE` for it, and
  `HostGradientStage` is the fast staging for that path rather than a
  replacement for it.
- **Fixed-point scales.** `MagnitudeReader` and `HostGradientStage` both
  reproduce the existing summation orders, so the scales are bit-identical
  to today's on both paths.

What is not exact, and where it matters:

- The device carries derivatives and multipliers as Float32. The host path
  multiplies Float64 by Float64 and rounds once at staging; the device
  rounds first and multiplies in Float32. Compensated gradients agree to
  Float32, which is the trade every device stage already makes.
- **The one place it changes a decision rather than a value:** the GOSS
  ranking score is a function of the gradients, so a Float32 score can put
  a different row across the top-k threshold than the Float64 one. That
  changes which rows a tree grows on. It is therefore opt-in:
  `round_eligibility(..., allow_device_ranking=False)` keeps a GOSS run on
  the host path and keeps both backends sampling identically. Anyone
  turning it on should report the per-round count of differing rows, and
  the plan file makes that a condition of accepting the change at all.
- Row bagging has no such exposure, because a bag never looks at a
  gradient.

## What a later profiler run has to answer

Full detail in `bench/apple/fused_round_plan.json`. The four questions,
and which claim each one decides:

1. **The idle interval between the magnitude readback and the first root
   histogram kernel** (Metal System Trace). Decides whether the deferred
   magnitude split is worth anything. If it is already zero, the split is
   neutral and should be kept for the ordering freedom, not for a speed
   claim.
2. **Per-round wall clock for a GOSS run, device path against host path,
   equal seed and rounds**, reported alongside the per-round count of
   differing sampled rows. Decides whether device row selection ships. A
   speedup reported without the sample-difference count is not a result.
3. **Bytes read from device memory by the root histogram kernel, at two
   feature counts an order of magnitude apart.** Decides the interleaved
   derivative plane, the largest remaining win and the one this lane could
   not build because it needs the histogram kernels. If the measured bytes
   do not scale with the feature count, the gather is cache-served and the
   change buys nothing. The cost model deliberately refuses to assume
   either way.
4. **Nothing decides the objective-into-histogram fusion.** It is refused
   on the circular dependency between the fixed-point scale and the
   gradients it quantizes. Reopening it means changing how histograms
   accumulate, not running a benchmark.

## Open items, in the order they are worth doing

1. **Compile the two modules.** They have never been through the compiler.
   Expect the usual `mut` on borrowed `DeviceBuffer` arguments to be the
   first thing that bites, since that is what it was for the objectives
   lane, and the signatures here were written to match that lesson.
2. **Collapse the duplicated reduction kernel.** Either give
   `GpuObjectiveState` an enqueue/read split and delete
   `_magnitude_partials_kernel`, or delete `_abs_sum_kernel` and route
   `magnitude_sums` through `MagnitudeReader`. The second is smaller.
3. **Score out-of-bag rows on the device**, which is the single thing
   standing between row bagging and the device round. Two candidates: reset
   the permutation to the identity and replay the tree's splits so every
   row lands in a leaf range, or walk the tree on the device with the
   predictor that already exists in `gpu_predict.mojo` and skip the range
   update entirely. The second looks cheaper and reuses shipped code.
4. **Sub-range device copies**, which would let `HostGradientStage` overlap
   conversion with transfer instead of only fusing its passes.
5. **The interleaved derivative plane**, after question 3 above is
   answered and not before.
