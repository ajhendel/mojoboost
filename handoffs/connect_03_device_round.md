# Handoff: fused device round, device objectives, device split search (connect task 03)

Nothing in this round was built, run, tested, benchmarked, or committed by
this lane. Every claim below is about code structure, not about behavior on
hardware. Every command in the last section is UNRUN.

## Files this lane owns and changed

- `src/mojoboost/gpu_fused_round.mojo`
- `src/mojoboost/gpu_gradient_stream.mojo`
- `src/mojoboost/gpu_objectives_native.mojo`
- `src/mojoboost/gpu_split_search.mojo`
- `handoffs/connect_03_device_round.md` (this file)

Nothing else was edited. Two changes that other files need are written out
verbatim under "Cross-lane patch requests" instead.

**A concurrent lane committed all four of these files mid-edit.** Commits
`dc21f03`, `860b1cf`, and `e6f3959` (2026-08-14) swept
`gpu_fused_round.mojo`, `gpu_gradient_stream.mojo`,
`gpu_objectives_native.mojo`, and then `gpu_split_search.mojo` into the
tree while this lane was working in them, so every change is already in
`HEAD` and `git status` shows the files clean. **This lane committed
nothing itself, staged nothing, and reverted nothing.** A reviewer cannot
use `git status` to see this work; read those commits, or read the files.
Because the sweep happened mid-edit, at least one of those commits contains
an intermediate state of this lane's work rather than the finished one.

## Implementations found before editing

The capability existed in five places, none of them wired to each other.

| What | Where | State found |
| --- | --- | --- |
| Device derivatives, softmax, raw-score updates, magnitude reduction | `gpu_objectives_native.mojo` | Complete; `supports_device_objective` its own if-chain |
| Objective capability table (second copy) | `objective_registry.objective_gradients_on_device` | Mirror of the above, documented as such, unconsumed |
| Round staging, magnitude split, tree router, eligibility rules, traffic model | `gpu_fused_round.mojo` | Complete primitives, no single entry point; carried a duplicate of `_abs_sum_kernel` |
| Row selection, GOSS ranking plane, host staging, interleaved planes | `gpu_gradient_stream.mojo` | Complete; no sizing or bag helpers for a caller |
| Device split search | `gpu_split_search.mojo` | Complete per node; one pinned staging slot forced one host wait per node |
| Round/search eligibility (third copy) | `train_gpu.device_gradients`, `train_gpu._check_device_search_supported` | Refuses bagging for a reason true only of GOSS |

Related but deliberately untouched: `gpu_leaf_batching.GpuLeafBatcher`
(multi-slot histogram pool), `gpu_frontier` (frontier bookkeeping),
`histogram_gpu._fixed_scale` (host twin of `device_fixed_scale`).

## Call path, before

    train_gpu._train_gpu_rounds
      device_gradients(...)                    # third eligibility copy
      state.fill_grad_hess / magnitude_sums    # one call, one wait, inline
      builder.upload_gradients or set_scales
      grow_tree_gpu -> _grow_tree_gpu_device_search
        per node: builder.enqueue_leaf
                  searcher.set_features        # map_to_host  -> host wait 1
                  searcher.set_allowed         # pinned copy
                  searcher.enqueue             # scan + reduce, record 0
                  searcher.download            # copy + sync  -> host wait 2
      state.update_raw_ranges                  # bagging refused upstream

Two host waits per node, one eligibility rule per file, and the magnitude
reduction unable to overlap with anything.

## Call path, after (what the code now supports)

    GpuDeviceRound(ctx, n_rows, objective, ...)     # eligibility settled once
      open_round(ctx, state)                        # softmax probabilities
      fill_gradients(ctx, state, grad, hess, k)
      [read_ranking(...) -> goss_select -> selection.from_goss]   # GOSS only
      reduce_magnitudes(ctx, selection, grad, hess) # compensate + enqueue
      ... caller enqueues scale-independent work here ...
      read_scales(ctx) -> RoundScales               # the round's only wait
      ... grow the tree ...
      advance_scores(ctx, state, rows, tree, bins, lr, k)
      end_round()

    searcher.enqueue_frontier(hist, nodes, params, g, h)   # whole level
    searcher.download_frontier(n)                          # one wait

One wait per round start and one per tree level, instead of one per node.
Both old entry points (`enqueue`/`download` per node, `search`) still exist
with their old signatures and their old contract.

## Connections completed

### 1. One objective capability table (`gpu_objectives_native.mojo`)

`supports_device_objective` now returns
`objective_registry.objective_gradients_on_device(objective)`. The registry
named this file as the mirror to delete and said the dependency has to run
GPU-module-to-metadata-module, never the reverse, so `max.gpu.*` stays out
of `params.mojo` and the CLI; that is the direction taken.
`objective_registry` imports only `.boosting`, which
`gpu_objectives_native` already imported, so no cycle is introduced.

The set is unchanged value for value: the registry answers
`objective_is_builtin`, whose eleven codes are exactly the arms the deleted
chain enumerated. `SQUARED_ERROR` is no longer imported here (it was only
ever the kernel's `else` arm); the arm is unchanged and commented.

### 2. One magnitude reduction (`gpu_objectives_native.mojo`, `gpu_fused_round.mojo`)

`gpu_fused_round._magnitude_partials_kernel` was a hand-maintained copy of
`_abs_sum_kernel`, and its own docstring asked for one of the two to be
deleted. Deleted. The kernel is now split into two reusable halves in
`gpu_objectives_native.mojo`:

- `enqueue_abs_sum(ctx, grad, hess, part, n_rows)` — the launch,
- `sum_abs_partials(ptr) -> GradMagnitudes` — the Float64 host fold.

`GpuObjectiveState.magnitude_sums` calls both back to back;
`MagnitudeReader.enqueue`/`read` call them across a wait. There is one
kernel, one grid geometry, and one summation order, so the split and
unsplit paths produce the same totals by construction rather than by
review.

### 3. The staged round (`gpu_fused_round.GpuDeviceRound`)

The production-consumable interface. It **owns** only what belongs to no
other component (the magnitude reader, and the tree router when the caller
routes all rows) and **borrows** all trainer state (`GpuObjectiveState`,
`GpuRowSelection`, the builder's gradient/hessian buffers, the binned
matrix, `GpuActiveRows`). It cannot become a second trainer: every stage
delegates to the function that already owns that step.

- Eligibility is settled in the constructor via `round_eligibility`, which
  raises `round_eligibility_reason(code)` for a configuration it rejects.
- `device_round_supported(...)` is the yes/no a strategy switch wants, so a
  trainer picks the device round or its established host path without
  catching an error.
- The stage machine (`closed → open → gradients → [ranked] → magnitudes →
  scaled`) is enforced, not documented. The order it protects is the one
  that matters numerically: compensation before reduction, because the
  fixed-point scale has to bound the values the histograms read. Reducing
  first produces a scale too large by up to the GOSS multiplier and
  silently weakens the Int32 overflow bound rather than failing.
- `advance_scores` picks exactly one raw-score update, which is what stops
  a caller from applying both (each adds a full `learning_rate * value`
  step, so both would advance in-bag rows twice).
- Names line up with `gpu_runtime.RoundLifecycle` (`open_round`/`end_round`
  ↔ `begin_round`/`end_round`); the two do not know about each other and
  this one owns no counters.

When routing is off the router is constructed at one row and one node, so
an unbagged session pays a handful of elements for a router it never
launches, and `advance_scores` never reaches it.

### 4. Session sizing helpers (`gpu_gradient_stream.mojo`)

- `GpuRowSelection.from_bag(ctx, bag)` — a bag's ids with deliberately no
  multipliers, since bagging changes which rows a tree sees and never what
  a row's derivatives are.
- `selection_capacity(n_rows, bagging_on, goss_on)` — `n_rows` when
  anything samples, 0 otherwise. Neither sampler has a tighter bound that
  holds every round (a bag is Binomial; a GOSS selection can exceed
  `top_k + other_k` on importance ties).
- `selection_wants_ranking(goss_on, allow_device_ranking)` — keeps the
  `n_rows` ranking plane out of every run that does not sample.

### 5. Batched frontier split search (`gpu_split_search.mojo`)

Every per-node table now has one slot per record: the feature set, the
allow mask, the float parameters (which carry the node's monotone bounds),
and a new two-word node table holding the node's slot count and its
histogram offset. This is exactly the widening
`handoffs/apple_a2_split_search.md` section 6 asked for.

- `_scan_slot_kernel` is now launched over `(feature slot, node)`; the
  reduction is one thread per node. A launch covers records
  `[record_base, record_base + n_records)`.
- `SplitNodeRequest` describes one leaf: `hist_slot`, optional per-node
  feature draw, optional allow mask, monotone bounds.
- `enqueue_frontier` stages every node **first**, then issues one copy per
  table, then one scan and one reduction. `download_frontier` is the
  batch's single wait. `search_frontier` composes them.
- `hist_slot` is in units of `3 * n_features * n_bins`, which is exactly
  `GpuLeafBatcher.slot_cells()` and exactly its `out_dev` slot stride, so a
  level's histograms built by the batcher are searchable in place with no
  new layout and no copy. `enqueue` takes the same `hist_slot` for the
  one-node path.
- `set_features` no longer uses `map_to_host`, so it no longer synchronizes;
  it stages into pinned memory and the copy goes out with the launch. That
  removes one host wait per node from the **existing** shipped path without
  changing a result.
- `set_features(features)` and `set_allowed(allowed)` with no `record`
  broadcast to every record slot, which is what today's callers mean, so
  their behavior is unchanged. A frontier passes `record=i`.

Determinism is unchanged in both directions: each node reads only its own
slots, the within-feature scan order still decides ties, the cross-feature
fold still accepts only a strictly greater gain, and no atomics are
involved. A batched frontier returns the records the same nodes searched
one at a time would return.

### 6. Float32 near ties and the host-scan fallback (`gpu_split_search.mojo`)

New `FREC_RUNNER_GAIN` word, `SPLIT_FWORDS` 10 → 11 (a record is now
exactly the 136 bytes the docstrings already claimed).

- Every acceptance site in `_scan_slot_kernel` and in `reference_search`
  now also keeps the best gain among the candidates it rejected, so a
  slot's runner-up is the best non-winning candidate of that feature.
- `_reduce_slots_kernel` folds the best losing *feature* together with the
  winning feature's own second candidate, so `record.runner_gain` covers
  both ways a node's decision can be close.
- `GpuSplitRecord.margin()` and `.is_near_tie(relative)` are the test;
  `SPLIT_TIE_RELATIVE = 1e-6` is roughly eight Float32 ulps of relative
  resolution, chosen on the conservative side because being too eager
  costs one node's host rescan and being too lax silently accepts a split
  the host would not have chosen.
- `host_rescan_recommended(record, tie_relative, enabled)` is the policy,
  per node and never per tree, with `enabled` as the caller's switch (a run
  that does not need CPU/GPU agreement passes False and keeps every
  decision on the device).
- `frontier_margin(records)` reports the same quantity one level up, where
  the gain comparison decides which leaf splits next and therefore the
  tree's shape.

Tracking the runner-up costs one compare per candidate and cannot change
which candidate wins.

### 7. Unsupported configurations, stated once (`gpu_split_search.mojo`)

`device_search_eligibility(n_bins, extra_active, feature_fraction_bylevel_active)`
and `device_search_reason(code)` carry the refusals `train_gpu` spells out
inline today, word for word, plus the 256-bin ceiling. Scalars rather than
a `TreeParams`, so this module stays out of the tree-parameter graph and no
new import is needed in either direction.

## Duplicates fused or quarantined

| Duplicate | Disposition |
| --- | --- |
| `_magnitude_partials_kernel` vs `_abs_sum_kernel` | **Fused.** Copy deleted, one kernel behind two reusable halves. |
| `supports_device_objective` vs `objective_gradients_on_device` | **Fused.** Delegates to the registry. |
| `device_fixed_scale` vs `histogram_gpu._fixed_scale` | **Quarantined**, patch requested below. Cross-lane. |
| `device_gradients` vs `round_eligibility` | **Quarantined**, patch requested below. Cross-lane. |
| `_check_device_search_supported` vs `device_search_eligibility` | **Quarantined**, patch requested below. Cross-lane. |
| `_range_hist_*_gh_kernel` vs `gpu_active_rows._range_hist_*_kernel` | **Left standing.** The interleaved path is off by default and exists to be measured against the split-plane path on one build; collapsing them needs the layout parameter to land in the other lane's kernels first. Debt, already flagged in the module. |

## Cross-lane patch requests

### A. `train_gpu.device_gradients` → `round_eligibility` (owner: task 01)

Replace the body's sampling test. The current one refuses bagging and GOSS
with one reason ("row sampling draws its sample from host-side gradients")
that is true only of GOSS: a bag is drawn from a counter stream keyed by
(seed, bag index, row) and never reads a derivative. What actually blocks
bagging is the out-of-bag rows' raw scores, which `GpuTreeRouter` closes.

```mojo
def device_gradients(
    supported: Bool,
    source: Int,
    bagging: BaggingParams,
    goss: GossParams,
    routes_all_rows: Bool = False,
    allow_device_ranking: Bool = False,
    n_classes: Int = 1,
    objective: Int = SQUARED_ERROR,
) raises -> Bool:
    var s = resolve_objective_source(source)
    if s == OBJECTIVE_SOURCE_HOST:
        return False
    var code = round_eligibility(
        objective,
        n_classes,
        bagging_enabled(bagging),
        goss.enabled,
        allow_device_ranking,
        routes_all_rows,
    )
    if code != ROUND_OK:
        if s == OBJECTIVE_SOURCE_DEVICE:
            raise Error(round_eligibility_reason(code))
        return False
    return True
```

The `supported` argument becomes redundant once the objective code is
passed (`round_eligibility` asks `supports_device_objective` itself);
keeping it as an ignored parameter or dropping it is task 01's call.

**This widens which configurations take the device path** — a bagged run
that routes all rows becomes eligible — so it changes behavior for existing
bagged GPU runs and wants the parity check in the plan file before it ships.

### B. `train_gpu._check_device_search_supported` → `device_search_eligibility` (owner: task 01)

```mojo
def _check_device_search_supported(params: TreeParams, n_bins: Int) raises:
    params.extra.check_scalars(params.min_data_in_leaf)
    var code = device_search_eligibility(
        n_bins,
        params.extra.is_active(),
        params.feature_fraction_bylevel != 1.0,
    )
    if code != SEARCH_OK:
        raise Error(device_search_reason(code))
```

The range checks still run first, so an out-of-range value is still
reported as the bad number it is. The messages are unchanged, character for
character, so no user-visible text moves.

### C. Frontier-batched node search in `_grow_tree_gpu_device_search` (owner: task 01, needs task 02's histogram slots)

The loop searches the two children of a split one at a time, each with its
own `download`. Both are known at once, so both can go in one batch:

```mojo
var batch = List[SplitNodeRequest]()
batch.append(
    SplitNodeRequest(
        left_slot,
        select_node_features(tree_features, ..., left_node),
        allowed.copy(),
        children.left.copy(),
    )
)
batch.append(
    SplitNodeRequest(
        right_slot,
        select_node_features(tree_features, ..., right_node),
        allowed.copy(),
        children.right.copy(),
    )
)
var recs = searcher.search_frontier(
    hist_buffer, batch, split_params, builder.g_scale, builder.h_scale
)
```

Two conditions, both outside this lane:

1. `searcher` must be constructed with `max_records >= len(batch)` (it is
   built with the default 1 today, in `_grow_tree_gpu_device_search`).
2. `hist_buffer` must hold both children's histograms at once. With
   `builder.out_dev` it holds one, so the batch is one node until the leaf
   batcher's pool is on this path; with `GpuLeafBatcher.out_dev` the two
   slots are already laid out the way `hist_slot` expects.

The depth and minimum-row rules stay where they are (`_search_leaf_device`
applies them to the record after the search); they are properties of the
tree, not of a histogram.

### D. `__init__.mojo` re-exports (owner: task 01)

```mojo
from .gpu_fused_round import (
    ROUND_OK,
    GpuDeviceRound,
    GpuTreeRouter,
    device_round_supported,
    round_eligibility,
    round_eligibility_reason,
)
from .gpu_split_search import (
    SEARCH_OK,
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    SplitNodeRequest,
    device_search_eligibility,
    device_search_reason,
    host_rescan_recommended,
)
```

`reference_search`, `decode_record`, `frontier_margin`, and the `gpu_*`
arithmetic helpers stay internal.

### E. One definition of the fixed-point scale (owner: whoever owns `histogram_gpu.mojo`)

`device_fixed_scale` (here) and `_fixed_scale` (there) are the same
expression; the only difference is that the host one makes its own pass
over a `List[Float64]` to get the magnitude first. The one-line refactor:
have `_fixed_scale` compute its total and then `return
device_fixed_scale(total)`. It is a pure-arithmetic function with no GPU
dependency, so importing it into `histogram_gpu.mojo` drags nothing new in
(that file already imports `device_fixed_scale`).

### F. Registry docstring (owner: task 08)

`objective_registry.mojo`'s "Temporary mirrors" section says
`objective_gradients_on_device` mirrors `supports_device_objective` and
that "the dependency runs the other way after wiring". It now does. The
mirror bullet can be replaced with a statement that
`gpu_objectives_native.supports_device_objective` delegates here.

## Remaining disconnections

1. **`GpuDeviceRound` has no caller.** The trainer still calls
   `state.fill_grad_hess` and `magnitude_sums` directly. Wiring it is task
   01's; nothing in this lane can reach `train_gpu.mojo`.
2. **`enqueue_frontier` has no caller**, for the two reasons in patch C.
   Until a multi-slot histogram buffer is on the device-search path, the
   batch is bounded at one node by the histogram, not by this module.
3. **`host_rescan_recommended` has no caller.** The trainer has to decide
   the near-tie posture (a `SPLIT_SEARCH_*` sibling, or a parity-mode flag)
   and own the per-node host rescan.
4. **GOSS on the device is opt-in and unreachable from the public API.**
   `allow_device_ranking` has no parameter or environment variable behind
   it; that is deliberate until someone decides whether a Float32-ranked
   sample is acceptable by default (it is not, for parity).
5. **The interleaved derivative plane stays off by default** and its two
   kernels stay duplicated, as above.
6. **`GpuTreeRouter` is reachable only through `GpuDeviceRound`** or
   directly; the trainer's bagged path still refuses the device objective
   until patch A lands.

## Fallbacks preserved

- Custom objectives never touch any of this: `round_eligibility` returns
  `ROUND_CUSTOM_OBJECTIVE`, `GpuDeviceRound` refuses to construct, and
  `train_custom_gpu`'s host-callback contract is untouched. `HostGradientStage`
  remains the fast path for host-origin gradients.
- An objective with no device kernel, a bagged run that does not route all
  rows, and a GOSS run without `allow_device_ranking` all keep the host
  gradient path.
- The node-at-a-time search API (`set_features`/`set_allowed`/`enqueue`/
  `download`/`search`) is unchanged in signature and in contract, so the
  shipped `SPLIT_SEARCH_DEVICE` path runs exactly as before, and
  `SPLIT_SEARCH_HOST` remains the default resolution.
- `params.extra`, `feature_fraction_bylevel`, and >256 bins are still
  refused rather than silently ignored, now with one definition of the
  refusal.

## Serialization and public API effects

None. The split record never reaches a model file: `serialize.mojo`, the C
ABI, the CPython bindings, and the Python estimators are untouched, and no
model state changed shape. `SPLIT_FWORDS` and the node table are internal
device layouts with no on-disk image. No Python-visible parameter, default,
or error message changes. `docs/LIGHTGBM_PARITY.md` needs no entry from
this lane; patch A would need one, because it widens which configurations
train on the device.

## Risks

1. **Nothing is compiled.** Four files changed, one of them structurally
   (kernel signatures, buffer sizes, a grid dimension). A syntax or arity
   error would be found by the first `mojo run`, which this lane did not
   do. This is the largest risk on the list by a wide margin.
2. **Memory grows with `max_records`.** The per-record tables are
   `max_records * n_features` Int32 for features and for the allow mask,
   and `max_records * n_features * (SPLIT_IWORDS + SPLIT_FWORDS)` words of
   slot records. At 100 features and 32 records that is ~450 KB, which is
   small against a histogram pool, but it is now a function of
   `max_records` where it was not.
3. **The batched staging contract is a contract.** `enqueue_frontier`
   stages every node before any table crosses, and `download_frontier` is
   the wait that releases the staging buffers. A caller that stages a
   second batch before the first is downloaded overwrites pinned memory a
   copy may still be reading. The single-node path keeps its old contract.
4. **`SPLIT_FWORDS` changed.** Any code that assumed 10 float words per
   record, or the old 132-byte size, is wrong. Nothing in the tree assumes
   it outside this file and its test, which decodes through
   `decode_record`.
5. **The near-tie tolerance is a judgment, not a measurement.** 1e-6
   relative is chosen from Float32's resolution and the shape of the gain
   expression. How often it fires on real data is unknown and is exactly
   what the first parity run should report.
6. **Patch A changes behavior**, as stated there.
7. **A concurrent lane committed all four of these files mid-edit**, so a
   reviewer cannot use `git status` to see this lane's work, and an
   intermediate state of it is in the history.

## Smallest later checks — ALL UNRUN

Nothing below was executed. They are listed smallest first; each is one
command.

```
# 1. Does the package still compile at all? Cheapest possible signal.
pixi run mojo build --emit=object -I src src/mojoboost/gpu_split_search.mojo

# 2. The one focused test for the split search, which covers the record
#    layout, the reference/device agreement, and the frontier reduction.
pixi run mojo run -I src tests/parallel/test_gpu_split_search.mojo

# 3. The one focused test for the device objectives, which covers
#    supports_device_objective and the magnitude reduction.
pixi run mojo run -I src tests/parallel/test_gpu_objectives_native.mojo

# 4. Only if 2 and 3 pass: the GPU training path end to end.
pixi run mojo run -I src tests/test_gpu_training.mojo
```

Per the repository's test budget, run **one** of these per change, not the
suite, and never a build/bench loop.

What a later test should add, once someone is writing tests again:

- A frontier of N nodes searched by `search_frontier` returns record for
  record what the same N nodes searched by `enqueue`/`download` return.
- A node with two exactly equal candidates reports `margin() == 0.0` and
  `is_near_tie()` True, and one with a single admissible candidate reports
  `is_near_tie()` False whatever its gain.
- `reference_search`'s `runner_gain` equals the device record's to Float32
  tolerance, which is the same shape as the existing record comparison.
- A `GpuDeviceRound` driven out of order (magnitudes before gradients)
  raises, and a custom objective refuses to construct one.
