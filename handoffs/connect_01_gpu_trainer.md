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
| `grow_tree_gpu`'s inline `n_left <= n_right` vs `subtraction_builds_left` | **Fused.** The predicate is `gpu_frontier`'s; the inline test is gone. |
| builder's hand-rolled shape validation vs `check_layout_support` | **Fused, additively.** The specific messages run first; the layout check adds the two overflow shapes they missed. |
| the multiclass loops' informal round contract vs `MulticlassRoundGuard` | **Fused.** The contract was a comment in two loops and is now one checked state machine driven by both. |
| `gpu_frontier.LeafFrontier` vs `_GpuLeafState` / `_GpuRecordLeafState` | **Selected against, quarantined.** `LeafFrontier` is a complete alternate frontier with its own commit planning and monotone clamping. Adopting it would rewrite both growers' tie-breaking, node-id assignment, and clamp order, which is exactly the `Tree` semantics this lane must preserve. Only `LeafWorkItem` (the frontier's declared contract with the batcher) is consumed. Migrating the growers onto `LeafFrontier` is a separate, testable change, and §6.4 is the patch for it. Only `LeafWorkItem` and `subtraction_builds_left` are consumed. |
| `gpu_fused_round.round_start` / `softmax_round_start` vs `builder.fill_gradients_device` | **Selected against.** They are convenience sequencers over `GpuObjectiveState` plus `GpuRowSelection` compensation, and their own docstrings say a caller with scale-independent work to interleave (which the trainer has: root row seeding) should call the stages directly. The trainer already calls the stages. `round_eligibility` and `GpuTreeRouter` from the same module *are* connected. |
| `gpu_gradient_stream.HostGradientStage` vs `builder.stage_gradients`/`upload_staged` | **Selected against.** The builder's pinned staging is the shipped path and is what `StagingRing` models. |
| `gpu_binned_layout` / `gpu_histogram_specializations` packing | **Not connected.** Both are planners and host-side packers; no kernel in the package reads a packed or blocked layout. `KernelFeatures.packed_bin_loads` and `specialized_bin_kernels` are declared `False` in `build_kernel_features`, which is the honest statement that the variants do not exist. |

---
## 5. Second pass: what the first pass wrote off, revisited

The first pass listed six modules as unconnectable. Four of them had a
connection inside this lane's owned files after all, and they are now made.
The revised list is below; §5.7 says what is still genuinely out of reach and
why, and §6 turns every remaining blocker into an applicable patch.

### 5.1 `gpu_frontier.subtraction_builds_left` — CONNECTED (fused)

`grow_tree_gpu` tested `n_left <= n_right` inline to decide which child is
built and which is subtracted. `subtraction_builds_left` is that predicate,
and its docstring names `grow_tree` and `grow_tree_gpu` as the two tests it
matches. The inline test is gone; the grower calls it. A batched grower and
this one can no longer pick different children and then disagree about which
histogram a slot holds. Pure host arithmetic, no behavior change.

### 5.2 `gpu_binned_layout.check_layout_support` — CONNECTED (fused)

`GpuHistogramBuilder`'s constructor hand-rolled its shape validation.
`layout_support` asks two questions it did not: a feature count past the
Int32 index range, and an `n_rows * n_features` cell count that overflows
Int32 even though each factor fits. Those are exactly the shapes whose flat
bin index would wrap, and the builder indexes `bins[f * n_rows + r]`
throughout.

`check_layout_support` now runs in the constructor **after** the existing
specific raises, so a bad row, feature, or bin count is still reported as the
number it is; only the shapes the old checks missed reach the layout message.
This is the ordering `_check_device_search_supported` already uses in
`train_gpu.mojo` for the same reason. Strictly additive coverage; no existing
error message changed, so nothing asserting on those strings can break.

### 5.3 `gpu_multiclass_batch.MulticlassRoundGuard` — CONNECTED

Both softmax loops in `_train_multiclass_gpu_rounds` now drive one guard. The
mapping is exact and was checked gate by gate against the guard's stated
ordering:

| Guard call | Device path | Host path |
| --- | --- | --- |
| `open_round()` | top of round `i` | top of round `i` |
| `note_probs()` | after `state.refresh_softmax` | after the `_softmax_inplace` pass |
| `note_gradients(k, 1)` | after `fill_softmax_gradients_device(state, k)` | after `upload_gradients(grad, hess)` |
| `note_tree(k)` | after `grow_tree_gpu` | after `grow_tree_gpu` |
| `note_commit(k)` | after `update_raw_device(..., k)` | after the `raw[r * n_classes + k] +=` pass |
| `close_round()` | after the class loop | after the class loop |

`close_round`, not `abandon_round`, in both — including the no-progress round
that drops its trees. That is deliberate and is a real distinction the guard
surfaced: both loops advance the raw scores for every class *before* testing
`made_progress`, so by the time the trees are popped they have already landed.
`abandon_round` refuses a round with a committed class for exactly that
reason, and calling it here would be the wrong claim about what happened.

What this buys: the one rule that a batched or reordered class schedule can
break silently — the probability snapshot must be taken when the raw scores
hold every previous round's trees and none of this round's — is now checked
rather than commented. Ordering is unchanged, so a correct run is unaffected.

**Risk, stated plainly.** This is the one change in either pass that can turn
a working trainer into one that raises, because the guard has no fallback: a
misread precondition is a fatal error, not a downgrade. The mapping above was
derived from the guard's implementation (`open_round` requires no pending
commits and resets the flags; `note_probs` requires `ROUND_OPEN` and refuses a
second call; `note_gradients` requires `probs_fresh` and an in-range class
run; `note_tree` requires `probs_fresh` and refuses a second tree per class;
`note_commit` requires that class's tree and refuses a second commit;
`close_round` requires every class grown and committed), not from its
docstring alone. It is still unverified by execution. The two-line revert is
to delete the `guard` declaration and its six call sites.

### 5.4 `hybrid_leaf_scheduler` — CONNECTED (as a report, which is what it is)

`GpuSession.note_hybrid` resolves `MOJOBOOST_HYBRID_LEAVES` against the run's
real facts (whether `SPLIT_SEARCH_DEVICE` resolved, whether the gradients are
host resident, that the trainer holds the `BinnedMatrix` for the whole fit,
the active row count and dataset shape) and keeps `describe_context` plus
`decline_name(decline_reason(...))` for `trace()`. All three session trainer
overloads call it.

Deliberately **not** a raise, unlike the transfer route in §3.4. That route's
alternatives are unimplemented, so honoring a request silently with the
default would mislead. This mode's alternatives are implemented and merely
unlicensed — no run has measured the coefficients its comparison needs — and
the module's own docstring says the switch exists before the numbers do
precisely so the decline reason is *observable*. Making it fatal would
contradict the design it is being connected to.

So the honest statement of what changed: the variable was read by nothing and
is now answered. A caller who sets it on a `SPLIT_SEARCH_DEVICE` run sees
`no_host_parent`; on a device-objective run, `gradients_on_device`; otherwise
`costs_unmeasured`. Those are three different pieces of work, and before this
they were indistinguishable from the variable having no effect.

The representative node is the root over every feature. That is sufficient
because every gate this can reach is a property of the run: the per-node
arithmetic in `decline_reason` sits behind the unmeasured-costs gate.

### 5.5 `gpu_frontier.LeafFrontier` — still selected against, and now for a
second reason

The first reason stands: adopting it rewrites both growers' tie-breaking,
node-id assignment, and clamp order, which is the `Tree` semantics this lane
must preserve.

The second reason is new and decisive for *now*. `gpu_frontier.mojo` is being
actively rewritten by another lane in this worktree during this task. Between
the first and second pass of this lane, `CommitPlan` gained a `missing_bin`
field and a `smaller_child_slot_is_left` method, a `FRONTIER_*` completion
status was added, and `GpuActiveRows.apply_commit` appeared as a new
consumer. Swapping the shipped grower onto a target that is moving, without
the ability to compile or test, is the wrong risk to take. The patch that
would do it is §6.4, written against the interface as it stands, and it
should be applied when that lane settles.

### 5.6 `gpu_multiclass_batch.GpuClassBatch` — still not connected, but the
stated obstacle was wrong

The first pass said it would duplicate the dataset. That is false and worth
correcting: `GpuClassBatch` holds no bins buffer at all (its fields are
gradients, hessians, scales, partials, output, features, and the two range
arrays), and the binned matrix arrives as a pointer exactly as it does for
`GpuLeafBatcher`. There is no second residency.

The real obstacle is the one that remains: its contract is several classes'
histograms over the *same* row ranges, which after each class's first split
is no longer true, because each class grows its own tree and its own
partition. The compatible window is the per-class root. Taking it needs
`n_classes` gradient planes resident at once and a `plane`-aware batch, plus a
`grow_tree_gpu` that can accept a precomputed root histogram. That is a
restructure of the multiclass loop, not a call site. §6.3 is the interface
note it needs first.

### 5.7 Still out of reach from this lane

1. **Device split search cannot read a batched histogram.** Patch §6.1.
2. **Interleaved gradient planes.** `gpu_gradient_stream` imports
   `histogram_gpu`, so the cycle blocks the call. Patch §6.2.
3. **Packed and blocked bin layouts.** `gpu_binned_layout`'s planners and
   `gpu_histogram_specializations`' packing helpers describe layouts no kernel
   in the package reads. `build_kernel_features` declares
   `packed_bin_loads = False` and `specialized_bin_kernels = False`, which is
   the honest statement. Connecting them means writing kernels, not call
   sites, and that is a different lane.
4. **`gpu_fused_round.round_start` / `softmax_round_start`,
   `gpu_gradient_stream.HostGradientStage`.** Selected against on the merits
   (§4), not blocked. No patch is owed.
5. **Synchronization ownership is still split** between the builder's direct
   `ctx.synchronize()` and `GpuSession.sync_for_host_read/write`. Needs a
   session-borrowing builder; `handoffs/apple_a5_runtime.md` owns it.
6. **Nothing owns a `GpuSession` across two fits.** The pool and residency
   ledgers still only record what a second fit could have skipped.
7. **`gpu_levelwise`, `gpu_output_planes`** remain unreached.

---

## 6. Ready-to-apply integration patches

Each is mechanically applicable by the lane that owns the target file. None
was applied here: every target is outside this lane's ownership.

### 6.1 `hist_offset` on the device split search

- **Target file / symbol:** `src/mojoboost/gpu_split_search.mojo`,
  `GpuSplitSearcher.enqueue`, and `_launch_search`.
- **Ownership:** not this lane's. Blocks §5.7.1.
- **Signature:**

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

  `_launch_search` takes the same new trailing `hist_offset: Int = 0` and adds
  it to the base index the scan kernel reads `hist` from. The scan kernel
  already receives `n_features * n_bins` as its plane stride, so the change is
  one added base offset, not a new indexing scheme.
- **Validation inside `enqueue`, before `_upload_params`:**

  ```
  if hist_offset < 0:
      raise Error("histogram offset must be nonnegative")
  if hist_offset + 3 * self.n_features * self.n_bins > len(hist):
      raise Error("histogram offset escapes the buffer")
  ```
- **Call site this unblocks:** `train_gpu._search_leaf_device` (this lane's
  file), which would become: acquire a pool slot, build the node through
  `builder.build_leaves`, then
  `searcher.enqueue(batcher.out_dev, split_params, g_scale, h_scale, bounds,
  0, slot * batcher.slot_cells())`. With it,
  `_grow_tree_gpu_device_search` builds both children in one packed launch
  and searches both without a readback, which is the frontier
  `gpu_frontier.leaves_per_launch` reports as `FEEDER_DEVICE_SEARCH = 2`.
- **State flow:** the fixed-point scales already travel as `g_scale`/`h_scale`
  arguments and are per round, so a pooled slot needs no new scale plumbing.
  The slot's stamp (`gpu_leaf_batching.subtraction_stamp`) is host-side and
  does not cross.
- **Errors:** the two raises above. An out-of-range offset must raise rather
  than clamp: clamping would search a different node's histogram and return a
  plausible record.
- **Fallback:** the default `hist_offset = 0` makes every existing call
  byte-identical, so the patch is inert until a caller passes one.
- **Serialization effect:** none.
- **Public API effect:** additive defaulted parameter on an exported symbol.
- **Dependency:** none; applicable immediately.
- **Minimal later validation — UNRUN:**
  `pixi run mojo test tests/parallel/test_gpu_split_search.mojo`, and a case
  asserting that a record searched at `hist_offset = k * slot_cells()` equals
  the same histogram searched at offset 0 in its own buffer.

### 6.2 Break the `gpu_gradient_stream` → `histogram_gpu` cycle

- **Target file / symbol:** `src/mojoboost/gpu_gradient_stream.mojo`,
  `enqueue_leaf_interleaved`, and the module's
  `from .histogram_gpu import GpuHistogramBuilder` line.
- **Ownership:** not this lane's. Blocks §5.7.2.
- **Signature** (preferred of the two options; it deletes the import
  outright):

  ```
  def enqueue_leaf_interleaved[
      bins_origin: MutOrigin,
      feat_origin: MutOrigin,
      out_origin: MutOrigin,
      part_origin: MutOrigin, //
  ](
      mut rows: GpuActiveRows,
      mut planes: InterleavedGradients,
      leaf: Int,
      caps: DeviceCaps,
      bins: MutPointer[UInt8, bins_origin],
      feat: MutPointer[Int32, feat_origin],
      out: MutPointer[Int32, out_origin],
      part: MutPointer[Int32, part_origin],
      n_slots: Int,
      strategy: Int,
      part_capacity: Int,
      g_scale: Float32,
      h_scale: Float32,
  ) raises:
  ```

  The body is unchanged except that `builder.X` becomes the corresponding
  parameter; it already computes `rows.range_tiling(...)` and calls
  `enqueue_range_histogram_interleaved`. This is the shape every other kernel
  entry in the package has (compare
  `GpuActiveRows.enqueue_range_histogram`), so it is a normalization as much
  as an unblocking.

  The `builder.has_gradients` and `planes.n_rows != builder.n_rows` checks
  move to the caller, which is where the builder is.
- **Call site this unblocks:** `histogram_gpu.GpuHistogramBuilder.enqueue_leaf`
  gains a branch on `gpu_gradient_stream.env_grad_layout()`, and the builder
  gains a `List[InterleavedGradients]` holder packed once per round, exactly
  as it holds `List[GpuLeafBatcher]` today.
- **State flow:** the interleaved plane is `2 * n_rows` Float32 filled by
  `planes.pack(ctx, grad_dev, hess_dev)` once per round, after the magnitudes
  are reduced and after any compensation, so it holds the values the
  histograms will read. The builder is the only object that knows when that
  moment is, which is why the switch belongs there.
- **Errors:** `pack` before build is already enforced by `planes.packed`;
  the row-count agreement check moves to the builder's constructor of the
  holder, where it is a construction-time invariant rather than a per-node
  test.
- **Ownership note:** `histogram_gpu.mojo` is this lane's, so the caller half
  is applicable here the moment the callee half lands. It was not written
  speculatively, because a call to a signature that does not exist is worse
  than no call.
- **Fallback:** `env_grad_layout()` defaults to `LAYOUT_SPLIT`, the shipped
  two-plane path, and anything but the exact string `interleaved` resolves
  there.
- **Serialization effect:** none. The plane is a re-layout of the same
  Float32 values, so histograms are unchanged.
- **Public API effect:** breaking signature change on
  `enqueue_leaf_interleaved`, which has no caller in the repository today.
- **Dependency:** none.
- **Minimal later validation — UNRUN:**
  `MOJOBOOST_GPU_GRAD_LAYOUT=interleaved pixi run mojo test tests/test_gpu_strategies.mojo`,
  asserting the interleaved and split layouts produce identical histograms —
  which they must, since `pack` is a pure copy.

### 6.3 State `GpuClassBatch`'s tree-ordering window

- **Target file / symbol:** `src/mojoboost/gpu_multiclass_batch.mojo`,
  `GpuClassBatch` docstring, and ideally `enqueue_ranged_histogram`.
- **Ownership:** not this lane's. Blocks §5.6.
- **Change:** add to the struct docstring, and enforce in
  `enqueue_ranged_histogram` if the begin/count arrays make it checkable:

  > Classes of one round share a row range only at their trees' roots. Each
  > class then grows its own tree and its own partition, so from the first
  > split onward class `k`'s node ranges are not class `j`'s. A batch that
  > spans classes past the root reads another class's rows, which is not a
  > precision difference but a different dataset. Callers batching per level
  > must batch within one class's tree, or restrict a cross-class batch to
  > the roots.

- **Why a note and not a check:** the struct takes per-slot `begin`/`count`
  arrays and cannot tell whether two slots' ranges came from the same
  partition. If the owning lane wants it checked, the cheapest form is a
  caller-supplied partition epoch per slot, refused when they differ — the
  same shape as `HistogramSlotPool`'s stamp.
- **State flow / errors / fallback / serialization / public API:** none;
  documentation, or one added precondition.
- **Dependency:** none.
- **Minimal later validation — UNRUN:** none needed for the note. For the
  epoch check, `pixi run mojo test tests/parallel/test_gpu_multiclass_batch.mojo`
  (create if absent) asserting a cross-epoch batch raises.

### 6.4 Move the growers onto `LeafFrontier` (deferred, not blocked)

- **Target files:** `src/mojoboost/train_gpu.mojo` (this lane's) against
  `src/mojoboost/gpu_frontier.mojo` (not this lane's, and currently moving).
- **Ownership:** the call site is ownable here; the interface is not, and
  §5.5 is why this waits.
- **Change:** replace `_GpuRecordLeafState` in `_grow_tree_gpu_device_search`
  with `LeafFrontier`, mapping the loop as:

  | Today | `LeafFrontier` |
  | --- | --- |
  | `frontier.append(_GpuRecordLeafState(...))` | `frontier.begin_tree(n_root)` then `set_candidate` |
  | the `best_gain` scan | `select_best()` |
  | `tree._add_node` pair + clamp + `child_bounds` | `plan_commit(slot, signs)` |
  | `frontier[best_i] = left; frontier.append(right)` | `apply_commit(plan)` |
  | `if n_left <= n_right` | `plan.build_left` (already fused, §5.1) |
  | `builder.apply_split(...)` | `plan.missing_bin` + `RowRouting.from_split` |

- **The precondition that must be verified before applying:** `plan_commit`
  must assign node ids in the same order `Tree._add_node` does (left then
  right), and `select_best` must break ties toward the lower frontier index,
  because both growers' node ids are the device's leaf ids and a different
  order is a different tree. Both appear to hold as written; neither is
  checked here.
- **Fallback:** keep `_grow_tree_gpu_device_search` as written and add the
  frontier version behind the existing `split_search` switch rather than
  replacing it, until a differential run shows identical trees.
- **Serialization effect:** none if the precondition holds; a changed node
  order would change every serialized model, which is why it is a
  precondition and not a detail.
- **Public API effect:** none.
- **Dependency:** the concurrent `gpu_frontier.mojo` rewrite must settle
  first.
- **Minimal later validation — UNRUN:**
  `MOJOBOOST_GPU_SPLIT_STRATEGY=device pixi run mojo test tests/parallel/test_gpu_split_search.mojo`,
  asserting the frontier grower and the record grower produce identical trees
  on the same seed.

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

**New `GpuSession` methods.** `note_alloc`, `begin_fit`, `end_fit`,
`session_state`, `note_hybrid`. All additive; `trace()` gained startup,
warm-up, paid-state, and (only when `MOJOBOOST_HYBRID_LEAVES` is set) hybrid
lines, so anything parsing `trace()` output by line count would see more
lines. The docstring already says it is not for parsing by anything shipped.

**New exports** in `__init__.mojo`: `BATCH_POOL_BUDGET_BYTES`,
`DEFAULT_BATCH_SLOTS`, `MAX_BATCH_SLOTS`, `build_kernel_features`,
`env_batch_slots` (histogram_gpu); `SPEC_LEVEL_BASELINE`, `SPEC_LEVEL_BATCHED`,
`HistogramWorkload`, `batching_declined_reason`, `derive_histogram_plan`,
`env_specialization_level` (apple_histogram_policy); `LeafWorkItem`
(gpu_frontier); `ROUND_OK`, `GpuTreeRouter`, `round_eligibility`,
`round_eligibility_reason` (gpu_fused_round); `BatchPlan`, `GpuLeafBatcher`,
`plan_batch`, `slots_for_budget` (gpu_leaf_batching); `FitLatency`,
`SessionState`, `StartupTrace`, `WarmupPlan`, `env_warmup_level`,
`session_state_from_trace` (initialization); `subtraction_builds_left`
(gpu_frontier); `check_layout_support`, `layout_support` (gpu_binned_layout);
`MulticlassRoundGuard` (gpu_multiclass_batch); `HybridContext`, `LeafWork`,
`Placement`, `decline_name`, `decline_reason`, `place_leaf`
(hybrid_leaf_scheduler).

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
6. **`MulticlassRoundGuard` has no fallback.** It is the only change in
   either pass that converts a wiring mistake into a fatal error rather than a
   downgrade, and it now sits on the shipped multiclass path, both the device
   and the host variant. The call mapping was derived from the guard's
   implementation gate by gate (§5.3) and is still unverified by execution.
   Revert is deleting the `guard` declaration and its six call sites in
   `_train_multiclass_gpu_rounds`.
7. **`check_layout_support` can refuse a shape the builder previously
   accepted** — specifically `n_features > Int32.MAX` or an
   `n_rows * n_features` product that overflows Int32. Those shapes would have
   produced a wrapped flat bin index, so refusing is the correct behavior, but
   it is a behavior change and not only added coverage.
8. **`note_hybrid` reports, it does not decide.** Nothing routes a histogram
   to the host. A reader who sees a `hybrid` line in `trace()` is seeing a
   resolved decline, not a placement that happened.
9. **`_build_leaves_batched` chunking is untested against `max_items`.** A
   frontier larger than `min(max_items, pool_capacity)` is served by several
   launches; the ordering contract that makes that safe is the download
   between chunks, which is stated in the code and not enforced.
10. **`session_state()` reports `kernels_ready` from
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
UNRUN  pixi run mojo test tests/test_gpu_objectives.mojo     # round guard
UNRUN  MOJOBOOST_HYBRID_LEAVES=replica \
         pixi run mojo test tests/parallel/test_gpu_active_rows.mojo
```

`tests/test_gpu_objectives.mojo` is the one that matters most after the second
pass (it is the file that calls `train_multiclass_gpu`): `MulticlassRoundGuard` is the only added code with no fallback, so a
single softmax GPU fit either passes or names the gate it violated.

The first three are the only ones that answer the open question, which is
whether this lane's source compiles. The batched run is the first that would
exercise `build_leaves` at all; a batched and an unbatched run of the same fit
should produce identical trees, and that equality is the assertion this lane's
central claim (§3.3) deserves.
