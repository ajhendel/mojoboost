# Connect 01: the GPU training stack into the production trainer

Scope of this lane: `src/mojoboost/train_gpu.mojo`, `src/mojoboost/gpu_runtime.mojo`,
`src/mojoboost/histogram_gpu.mojo`, `src/mojoboost/__init__.mojo`, and this file.
Nothing outside them was edited. Nothing was built, run, tested, benchmarked,
formatted, or committed, so no claim below is a correctness, performance,
parity, packaging, or hardware result. Every statement about behavior is a
statement about the source as written.

`src/mojoboost/gpu_session.mojo` does not exist and was not created:
`GpuSession` already lives in `gpu_runtime.mojo` and a second module holding
a second session would be exactly the duplicate this lane exists to avoid.

---

## 1. What already existed

### Reached by the trainer before this lane

| Module | How the trainer reached it |
| --- | --- |
| `gpu_active_rows` | through `GpuHistogramBuilder.rows`: root seeding, per-split partition, per-node range histogram |
| `gpu_objectives_native` | `builder.objective_state` / `fill_gradients_device` / `fill_softmax_gradients_device` / `update_raw_device` |
| `gpu_split_search` | `_grow_tree_gpu_device_search` under `SPLIT_SEARCH_DEVICE` |
| `gpu_predict` | `_DeviceValidScorer` under `VALID_SCORE_DEVICE` |
| `gpu_runtime` | the `GpuSession` overloads and the `RoundLifecycle` seam |
| `gpu_tiling` | `derive_tiling`, `DeviceCaps` |

### Not reached by anything (orphans)

`gpu_leaf_batching`, `gpu_frontier`, `gpu_binned_layout`,
`gpu_histogram_specializations`, `gpu_fused_round`, `gpu_gradient_stream`,
`gpu_multiclass_batch`, `gpu_levelwise`, `gpu_output_planes`,
`apple_histogram_policy`, `hybrid_leaf_scheduler`, `initialization`,
`unified_memory_policy`.

Verified by importer search: each was imported only by another orphan, by a
bench, or by a handoff. `unified_memory_policy` had exactly one importer,
`bench/apple/unified_memory.mojo`.

---

## 2. Call path, before and after

### Before

```
train_gpu
  device_gradients                      <- train_gpu's own 2-rule blocker list
  GpuHistogramBuilder(data)             <- staged copy, route never consulted
  per round: fill_gradients_device | upload_gradients
    grow_tree_gpu
      builder.build_leaf(node)          <- always one leaf per launch
      subtract_histogram(parent, child) <- host, always
      builder.apply_split(...)
    builder.update_raw_device(...)      <- leaf ranges only => bagging blocked
```

### After

```
train_gpu
  device_gradients -> gpu_fused_round.round_eligibility        [FUSED]
  GpuHistogramBuilder(data)
    unified_memory_policy.resolve_from_env(ROLE_BINS, ...)     [CONNECTED]
    apple_histogram_policy.env_specialization_level()          [CONNECTED]
  per round: fill_gradients_device | upload_gradients
    grow_tree_gpu
      builder.batches_nodes([left, right])
        -> apple_histogram_policy.derive_histogram_plan
        -> apple_histogram_policy.batching_declined_reason     [CONNECTED]
      yes -> builder.build_leaves([left, right])
               gpu_active_rows range -> gpu_frontier.LeafWorkItem
               gpu_leaf_batching.plan_batch / uniform_scales
               gpu_leaf_batching.subtraction_stamp
               GpuLeafBatcher.enqueue_frontier_batch           [CONNECTED]
               GpuLeafBatcher.download_slots
      no  -> builder.build_leaf(smaller) + subtract_histogram  [PRESERVED]
      builder.apply_split(...)
    bagged + explicit device request
      -> GpuTreeRouter.update_all_rows                         [CONNECTED]
    otherwise
      -> builder.update_raw_device                             [PRESERVED]

GpuSession()
  initialization.StartupTrace / FitLatency / WarmupPlan        [CONNECTED]
  session.begin_fit / end_fit around every session-overload fit
  session.session_state() -> initialization.SessionState
```

---

## 3. Connections completed

### 3.1 Round eligibility, fused (`train_gpu.mojo`)

`device_gradients` held its own list of blockers (no device kernel, any row
sampling). `gpu_fused_round.round_eligibility` held a strictly richer version
of the same policy, unreached. The trainer's list is gone;
`device_gradients` now delegates and raises `round_eligibility_reason(code)`
under an explicit `OBJECTIVE_SOURCE_DEVICE`.

- Signature changed from `(supported, source, bagging, goss)` to
  `(objective, n_classes, source, bagging, goss, routes_all_rows=False)`.
  `supports_device_objective` is no longer called from `train_gpu`;
  `round_eligibility` calls it.
- Default behavior is unchanged: `ROUND_OK` holds exactly where the old
  function returned `True`, and the same two configurations fall back under
  `AUTO`.
- GOSS stays blocked: `allow_device_ranking` is passed `False`, so both
  backends keep sampling from the same Float64 gradients.

### 3.2 Bagged rounds on the device objective path (`train_gpu.mojo`)

`GpuTreeRouter.update_all_rows` is the documented answer to the one thing
that kept bagging off the device round: the leaf ranges cover in-bag rows
only, so out-of-bag rows kept stale raw scores. `_train_gpu_rounds` now takes
`route_all_rows` and, when it is set *and* bagging is on, holds one router
per fit and advances every row's raw score through it.

- Opt-in only: `routes_all` is True only when the caller resolved to
  `OBJECTIVE_SOURCE_DEVICE` explicitly. Under `AUTO` a bagged run still takes
  the host path and its Float64 raw scores, byte for byte as before.
- One router per fit, not per tree or per node.
- Unbagged runs allocate no router even under an explicit device request, so
  they keep the cheaper range update.
- The bag comes from `refresh_bag` with the same seed and schedule as the
  host path and the CPU trainer, and the degenerate-tree rule now follows
  bagging semantics on this path too (`continue`, not `break`).

### 3.3 Multi-leaf batching (`histogram_gpu.mojo`, `train_gpu.mojo`)

New builder surface: `range_of`, `leaf_rows`, `histogram_plan`,
`open_batching`, `batching_live`, `batches`, `batches_nodes`, `build_leaves`.

- The row windows come from `GpuActiveRows`' own range table, so a batch
  reads exactly the rows a single-leaf build would have. No second row model.
- `apple_histogram_policy` is the decision, in one place (`batches`), so the
  grower and `build_leaves` cannot answer differently.
- `MOJOBOOST_GPU_HIST_SPECIALIZATION=batched` is the request; there is no new
  environment switch for the feature itself. `MOJOBOOST_GPU_BATCH_SLOTS`
  only sizes the pool, and is read only once batching was requested.
- The leaf-wise grower feeds it a split's two children, which is the one
  frontier the host-search grower can offer (`gpu_frontier.leaves_per_launch`
  says `FEEDER_HOST_SEARCH` offers one leaf per commit *for the subtraction
  path*; building both children instead offers two).
- **Why this is not a different fit.** Accumulation is fixed-point Int32
  under one scale for the whole tree, so a parent's bins are the exact
  integer sum of its children's. Building both children and subtracting one
  built child from the parent therefore agree bin for bin, not to a
  tolerance. This is an argument from the source, not a measurement.
- Fallback: unrequested, declined, or unaffordable, every node goes through
  `build_leaf`, which is the shipped path unchanged.
- Provider reuse rather than local glue: the slot stamp is
  `gpu_leaf_batching.subtraction_stamp`, the per-item scale table is
  `uniform_scales`, the launch is `enqueue_frontier_batch` (which range-checks
  every item against the tree's live active prefix), and the readback is
  `download_slots` (one mapping per chunk, not one per leaf).

### 3.4 Transfer route (`histogram_gpu.mojo`)

`unified_memory_policy.resolve_from_env(ROLE_BINS, ...)` is resolved once per
builder. Every route but `ROUTE_COPY_STAGED` is structurally blocked or
unproven today, so the resolved route is the staged copy and the upload is
unchanged; an explicit `MOJOBOOST_GPU_TRANSFER` the builder cannot honor now
raises with `describe_decision(route)` instead of being silently ignored.

`unified_memory` is passed `False` because `DeviceCaps` does not report it,
which is the same conservative answer `profile_from_caps` gives. It can only
widen what the resolver allows, so `False` cannot enable an unproven route.

### 3.5 Startup and warm-up (`gpu_runtime.mojo`)

`GpuSession` now owns a `StartupTrace`, a `FitLatency`, and a `WarmupPlan`.

- Context creation and device discovery are timed where they happen.
- `note_kernel` attributes a kernel's first launch to `kernel_create` and, if
  the warm-up plan named it, records what that creation cost.
- `note_transfer` / `note_alloc` feed `first_transfer` / `first_alloc`.
- `begin_fit` / `end_fit` route each fit to `first_fit` or `warm_fit`; all
  three session trainer overloads call them.
- `session_state()` answers `initialization.SessionState` truthfully, which
  is what that struct's docstring names `GpuSession` as the owner of and what
  `device_policy.decide_device` takes as data.
- `trace()` reports the startup and warm-up sections.
- `MOJOBOOST_STARTUP_TRACE=1` and `MOJOBOOST_GPU_WARMUP` are the switches;
  both off means no clock reads and no behavior change.

---

## 4. Duplicates: fused, or selected against

| Duplicate | Outcome |
| --- | --- |
| `device_gradients`'s blocker list vs `round_eligibility` | **Fused.** The trainer's list is deleted; `round_eligibility` is the single policy. |
| local stamp counter vs `subtraction_stamp` | **Fused.** The builder keeps two counters (`round_epoch`, `feat_epoch`) and the encoding lives only in `gpu_leaf_batching`. |
| local scale-table loop vs `uniform_scales` | **Fused.** |
| raw-pointer `enqueue_batch` vs `enqueue_frontier_batch` | **Fused** onto the range-checked entry. |
| per-slot `download_slot` loop vs `download_slots` | **Fused** onto the one-mapping entry. |
| `gpu_frontier.LeafFrontier` vs `_GpuLeafState` / `_GpuRecordLeafState` | **Selected against, quarantined.** `LeafFrontier` is a complete alternate frontier with its own commit planning and monotone clamping. Adopting it would rewrite both growers' tie-breaking, node-id assignment, and clamp order, which is exactly the `Tree` semantics this lane must preserve. Only `LeafWorkItem` (the frontier's declared contract with the batcher) is consumed. Migrating the growers onto `LeafFrontier` is a separate, testable change. |
| `gpu_fused_round.round_start` / `softmax_round_start` vs `builder.fill_gradients_device` | **Selected against.** They are convenience sequencers over `GpuObjectiveState` plus `GpuRowSelection` compensation, and their own docstrings say a caller with scale-independent work to interleave (which the trainer has: root row seeding) should call the stages directly. The trainer already calls the stages. `round_eligibility` and `GpuTreeRouter` from the same module *are* connected. |
| `gpu_gradient_stream.HostGradientStage` vs `builder.stage_gradients`/`upload_staged` | **Selected against.** The builder's pinned staging is the shipped path and is what `StagingRing` models. |
| `gpu_binned_layout` / `gpu_histogram_specializations` packing | **Not connected.** Both are planners and host-side packers; no kernel in the package reads a packed or blocked layout. `KernelFeatures.packed_bin_loads` and `specialized_bin_kernels` are declared `False` in `build_kernel_features`, which is the honest statement that the variants do not exist. |

---

## 5. Remaining disconnections

1. **Device split search cannot read a batched histogram.** The
   `SPLIT_SEARCH_DEVICE` grower builds both children and is the frontier
   batching most wants, but a batched build lands in the batcher's slot pool
   and `GpuSplitSearcher.enqueue` takes a whole `DeviceBuffer` with no word
   offset. Patch request in §6.
2. **Interleaved gradient planes** (`gpu_gradient_stream.InterleavedGradients`
   + `enqueue_leaf_interleaved`, `MOJOBOOST_GPU_GRAD_LAYOUT=interleaved`)
   are unreached. `gpu_gradient_stream` imports `histogram_gpu`, so
   `histogram_gpu` cannot import it back; the drop-in free function cannot be
   called from inside `build_leaf`. Patch request in §6.
3. **Multiclass class batching** (`gpu_multiclass_batch.GpuClassBatch`) is
   unreached. Its contract holds only where several classes' histograms are
   wanted over the *same* row ranges, which after each class's first split is
   no longer true, because each class grows its own tree and its own
   partition. The compatible window is the per-class root, and taking it
   needs `n_classes` gradient planes resident at once plus a `plane`-aware
   `LeafWorkItem` batch. Not attempted rather than faked.
4. **`hybrid_leaf_scheduler`** is unreached. `place_leaf` declines everything
   today (`MODE_OFF` by default, and `DECLINE_COSTS_UNMEASURED` whenever it
   is not), and a host placement additionally needs the host `BinnedMatrix`,
   which `GpuHistogramBuilder` deliberately does not retain. Wiring a policy
   that can only answer `PLACE_GPU` would be an import, not an integration.
5. **Synchronization ownership is still split.** `GpuHistogramBuilder` calls
   `ctx.synchronize()` directly; `GpuSession.sync_for_host_read/write` exist
   and are unused by it, because the builder copies the `DeviceContext` and
   holds no session reference. Routing the builder's drains through the
   session is the change `handoffs/apple_a5_runtime.md` describes and needs a
   session-borrowing builder.
6. **Nothing owns a `GpuSession` across two fits.** The pool and residency
   ledgers still only record what a second fit could have skipped. The fit
   latency added here is the first thing that observes the cold/warm split.
7. **`gpu_levelwise`, `gpu_output_planes`, `gpu_binned_layout`** remain
   unreached from the trainer.

---

## 6. Exact cross-lane patch requests

**R1 — `src/mojoboost/gpu_split_search.mojo`: a word offset on the histogram.**

Add a defaulted parameter to `GpuSplitSearcher.enqueue` and thread it into
`_launch_search` and the scan kernel's base index:

```
def enqueue(
    mut self,
    mut hist: DeviceBuffer[DType.int32],
    params: GpuSplitParams,
    g_scale: Float64,
    h_scale: Float64,
    bounds: OutputBounds = OutputBounds.unbounded(),
    record: Int = 0,
    hist_offset: Int = 0,          # NEW: Int32 words into `hist`
) raises:
```

`hist_offset` must default to 0 so every existing call is unchanged, and must
be validated against `hist` (`hist_offset >= 0` and
`hist_offset + 3 * n_features * n_bins <= len(hist)`). With it,
`_search_leaf_device` can hand the searcher
`batcher.out_dev` at `slot * batcher.slot_cells()` and the device-search
grower can search two batch-built children without a readback.

**R2 — `src/mojoboost/gpu_gradient_stream.mojo`: break the import cycle for the
interleaved layout.** Either

- move `enqueue_leaf_interleaved` (and only it) into `histogram_gpu.mojo`,
  leaving `InterleavedGradients` and `enqueue_range_histogram_interleaved`
  where they are, so `histogram_gpu` can import the plane struct without
  `gpu_gradient_stream` importing the builder; or
- change `enqueue_leaf_interleaved` to take the pieces it uses
  (`mut rows: GpuActiveRows`, `bins`, `feat`, `out`, `part` pointers,
  `caps`, `tiling.strategy`, `part_capacity`, `n_slots`, the two scales)
  instead of `mut builder: GpuHistogramBuilder`, which drops the
  `from .histogram_gpu import GpuHistogramBuilder` line entirely.

The second is preferred: it is the same shape every other kernel entry in the
package already has, and it leaves the layout switch selectable from
`histogram_gpu.enqueue_leaf` behind `env_grad_layout()`.

**R3 — `src/mojoboost/gpu_multiclass_batch.mojo`: state the tree-ordering
contract explicitly.** `GpuClassBatch` batches classes over shared row
ranges. Add to its docstring (or as a checked precondition) the window in
which that holds for a trainer: all classes of one round share a row range
only at their trees' roots, because each class then partitions independently.
Without that stated, a later integration will batch past the root and read
another class's rows.

---

## 7. Fallbacks preserved

Every stage this lane touched keeps the path it replaced, and none of them is
the default:

| Stage | Default | Opt-in |
| --- | --- | --- |
| histogram launch | one leaf per launch + host subtraction | `MOJOBOOST_GPU_HIST_SPECIALIZATION=batched`, and only where `batching_declined_reason` agrees |
| bagged device round | host gradients, Float64 raw scores | `objective_source=OBJECTIVE_SOURCE_DEVICE` |
| transfer route | `ROUTE_COPY_STAGED` | `MOJOBOOST_GPU_TRANSFER` (every other route raises today) |
| startup trace | off, no clock reads | `MOJOBOOST_STARTUP_TRACE=1` |
| warm-up plan | off | `MOJOBOOST_GPU_WARMUP=train|all` |
| split search | host scan | `SPLIT_SEARCH_DEVICE` (unchanged) |
| validation scoring | host tree walk | `VALID_SCORE_DEVICE` (unchanged) |

The known trainer was not deleted, replaced, or restructured. `grow_tree_gpu`,
`_grow_tree_gpu_device_search`, `_train_gpu_rounds`,
`_train_custom_gpu_rounds`, `_train_multiclass_gpu_rounds`, and
`_train_gpu_valid_rounds` all keep their shape.

---

## 8. Public API and serialization effects

**Serialization: none.** No model state changed. `Tree`, `Booster`,
`MulticlassBooster`, and everything in `serialize.mojo` / `model.mojo` /
`lgbm_model_io.mojo` are untouched. Every new piece of state
(`bins_route`, `spec_level`, `round_epoch`, `feat_epoch`, `batch_feat_stamp`,
the batcher, the startup trace) is per-fit device or bookkeeping state that
never enters a model, so a model trained on the batched path serializes
identically to one trained on the single-leaf path.

**Breaking signature change (one).** `device_gradients` — exported from
`__init__.mojo` — changed arity and parameter meaning (§3.1). No caller was
found in `tests/`, `bench/`, `python/`, `bindings/`, or `examples/`; the only
occurrences of the name outside `train_gpu.mojo` were prose in
`bench/apple/fused_round_plan.json`.

**Additive signature change (one).** `_train_gpu_rounds` gained a defaulted
`route_all_rows`. It is module-private.

**New exports** in `__init__.mojo`: `BATCH_POOL_BUDGET_BYTES`,
`DEFAULT_BATCH_SLOTS`, `MAX_BATCH_SLOTS`, `build_kernel_features`,
`env_batch_slots` (histogram_gpu); `SPEC_LEVEL_BASELINE`, `SPEC_LEVEL_BATCHED`,
`HistogramWorkload`, `batching_declined_reason`, `derive_histogram_plan`,
`env_specialization_level` (apple_histogram_policy); `LeafWorkItem`
(gpu_frontier); `ROUND_OK`, `GpuTreeRouter`, `round_eligibility`,
`round_eligibility_reason` (gpu_fused_round); `BatchPlan`, `GpuLeafBatcher`,
`plan_batch`, `slots_for_budget` (gpu_leaf_batching); `FitLatency`,
`SessionState`, `StartupTrace`, `WarmupPlan`, `env_warmup_level`,
`session_state_from_trace` (initialization).

`apple_histogram_policy.HistogramPlan` was deliberately **not** exported:
`distributed_transport.mojo` defines a struct of the same name, and a
re-export would collide. `unified_memory_policy` names were not exported
either; `resolve_from_env` is too generic for a flat package namespace.

**New environment variables owned by this lane:** `MOJOBOOST_GPU_BATCH_SLOTS`
only. The other three switches touched (`MOJOBOOST_GPU_HIST_SPECIALIZATION`,
`MOJOBOOST_GPU_TRANSFER`, `MOJOBOOST_STARTUP_TRACE`, `MOJOBOOST_GPU_WARMUP`)
were already defined by their own modules and are now read on a real path.

---

## 9. Risks

1. **Nothing here has been compiled.** Static inspection only, as required.
   The likeliest failure modes are Mojo type/ownership details rather than
   logic: calling `mut self` methods through `List` subscripts
   (`self.batcher[0].enqueue_frontier_batch(...)`), which follows the
   precedent at `distributed_transport.mojo:1512`
   (`self.peers[index].send_all(...)`); and holding a move-only
   `GpuLeafBatcher` / `GpuTreeRouter` in a zero-or-one `List`, which follows
   `List[_GpuLeafState]` in `train_gpu.mojo`.
2. **Concurrent lane dependency.** `gpu_leaf_batching.mojo` is being extended
   by another lane in this same worktree right now, and this lane calls four
   of the entries that landed there during it: `subtraction_stamp`,
   `uniform_scales`, `enqueue_frontier_batch`, `download_slots`. Those calls
   break if that lane renames or re-signatures them. The additive fallbacks
   (`plan_batch`, `enqueue_batch`, `download_slot`) are still present and this
   file could be moved back onto them.
3. **`build_kernel_features()` reports `batched_leaf_kernel = True`.** That
   flag's docstring in `gpu_histogram_specializations.mojo` says "compiled in
   *and validated*". The compilation half is true (`build_leaves` instantiates
   those kernels). The validation half is not, and nothing here claims it: the
   flag can only be acted on when `SPEC_LEVEL_BATCHED` was asked for
   explicitly, which never resolves from `auto`. A reviewer who wants that
   flag to mean validated should gate `open_batching` on a second
   acknowledgement instead.
4. **Batched vs subtracted histograms are argued equal, not measured equal.**
   The argument is exact-integer (§3.3) and rests on the fixed-point scale
   being constant within a tree, which `stage_gradients` /
   `fill_gradients_device` establish per round. If a future change re-derives
   scales mid-tree, the argument fails; `subtraction_stamp` is what would
   catch that on the subtraction path, and this lane's batched path does not
   subtract at all.
5. **The bagged device round changes raw-score precision** from Float64 (host
   path) to Float32 (device), which can move `best_iteration` and leaf values
   within Float32 noise relative to the CPU trainer. It is opt-in for exactly
   that reason.
6. **`_build_leaves_batched` chunking is untested against `max_items`.** A
   frontier larger than `min(max_items, pool_capacity)` is served by several
   launches; the ordering contract that makes that safe is the download
   between chunks, which is stated in the code and not enforced.
7. **`session_state()` reports `kernels_ready` from
   `kernels.warm_count >= N_KERNELS`.** The registry is only fed by
   `note_kernel`, which no shipped code path calls yet, so this reads `False`
   on every real session today. That is conservative and correct as a report;
   it is not yet a measurement.

---

## 10. Smallest later focused commands — ALL UNRUN

Not run by this lane. One at a time, per
`handoffs`/the repository's one-focused-test-per-change budget.

```
UNRUN  pixi run mojo build src/mojoboost/histogram_gpu.mojo
UNRUN  pixi run mojo build src/mojoboost/train_gpu.mojo
UNRUN  pixi run mojo build src/mojoboost/gpu_runtime.mojo
UNRUN  pixi run mojo test tests/parallel/test_gpu_active_rows.mojo
UNRUN  pixi run mojo test tests/test_gpu_strategies.mojo
UNRUN  MOJOBOOST_GPU_HIST_SPECIALIZATION=batched \
         pixi run mojo test tests/test_gpu_strategies.mojo
UNRUN  MOJOBOOST_GPU_TRANSFER=map_write \
         pixi run mojo test tests/test_gpu_strategies.mojo   # expect a raise
```

The first three are the only ones that answer the open question, which is
whether this lane's source compiles. The batched run is the first that would
exercise `build_leaves` at all; a batched and an unbatched run of the same fit
should produce identical trees, and that equality is the assertion this lane's
central claim (§3.3) deserves.
