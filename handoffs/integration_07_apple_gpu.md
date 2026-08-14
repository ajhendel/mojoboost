# Task 07: integrating the isolated Apple GPU core into the trainer

Status. Written by static reasoning only. **Nothing in this task was
compiled, run, tested, or benchmarked**, which the task forbade. No
performance claim is made anywhere below, and no default was changed to a
device path that no benchmark has justified.

Files this task owns and changed.

- `src/mojoboost/train_gpu.mojo` (rewritten in place, no trainer deleted)
- `src/mojoboost/gpu_predict.mojo` (context ownership and session
  registration)
- `src/mojoboost/gpu_runtime.mojo` (the lifecycle seam the trainers use)
- `src/mojoboost/__init__.mojo` (exports)
- `handoffs/integration_07_apple_gpu.md` (this file)

Files this task owns and did **not** change, with the reason.

- `src/mojoboost/histogram_gpu.mojo`. It already consumes A1 (active rows),
  A3 (device objectives), and A5 (the session constructor). The remaining
  A5 item inside it, routing its three `ctx.synchronize()` calls through
  `session.sync_for_host_read` / `sync_for_host_write`, is gated on four
  unanswered questions about queue ordering that A5 lists and that no
  reading of this tree can settle. See "Not done, and why" below.
- `src/mojoboost/gpu_active_rows.mojo`, `gpu_objectives_native.mojo`,
  `gpu_split_search.mojo`. All three are already reached from the trainer
  through the builder. Nothing in them needed to change for the stages
  below, and changing a lane module to suit an integration that has not run
  is how a working kernel gets broken.

Files this task did not touch and must not: `device.mojo`,
`apple_gpu_policy.mojo`, `bindings/`, `python/`, `tests/`, `bench/`,
`docs/`, `README.md`, `pixi.toml`, `tree.mojo`, `boosting.mojo`,
`custom_metric.mojo`, `packaging/`. Every edit those need is written out
verbatim in "Central integration still required".

---

## 1. Stage by stage diff map

Five stages. Each is independently revertable, and each names the switch
that returns the run to the path it replaced.

### Stage 1: persistent context and buffers (the session seam)

**Was.** Each of `train_gpu`, `train_custom_gpu`, and
`train_multiclass_gpu` constructed `GpuHistogramBuilder(data)`, which opens
a private `DeviceContext`. `GpuSession` existed, `GpuHistogramBuilder` had a
constructor that borrows one, and nothing in the tree constructed a session.

**Now.**

`gpu_runtime.mojo`

| change | what it is |
| --- | --- |
| `trait RoundLifecycle` | the four boundaries a boosting loop crosses: `begin_round`, `begin_tree`, `end_tree`, `end_round`. Signatures are exactly `GpuSession`'s existing ones, so conformance is a declaration, not a rewrite. |
| `struct NoLifecycle` | the no-op conformer. Two Int counters, no device, cannot raise. |
| `struct GpuSession(RoundLifecycle, Movable)` | one word added to the declaration. No method body changed. |
| `GpuSession` docstring | updated: the builder and the predictor now take a session, and what is still missing is an owner that holds one across two fits. |

`train_gpu.mojo`

| change | what it is |
| --- | --- |
| `_check_train_gpu` | the five input checks `train_gpu` made inline, extracted so both entry points share one copy and the session overload cannot drift. |
| `_multiclass_base_scores` | the log-prior computation and the label range check, extracted for the same reason. Both run **before** the builder is constructed, so a bad input still raises before the binned matrix is uploaded. |
| `_train_gpu_rounds[S: RoundLifecycle]` | the body of the old `train_gpu`, moved verbatim except for the hook calls and the `device_grads` predicate. Takes an already-constructed builder. |
| `_train_custom_gpu_rounds[S, F]` | same, for `train_custom_gpu`. |
| `_train_multiclass_gpu_rounds[S]` | same, for `train_multiclass_gpu`. |
| `train_gpu`, `train_custom_gpu`, `train_multiclass_gpu` | each is now two overloads: the original signature (plus new trailing defaults), and one taking `mut session: GpuSession` as the first argument. |

Every extracted `_*_rounds` function carries the same
`comptime if not has_accelerator(): raise ... else:` guard the trainer
bodies sat inside before the extraction, and so do the three device methods
of `_DeviceValidScorer` (through `_open_valid_predictor`, since a
constructor cannot leave `self` uninitialized in the raising branch). This
is not decoration. Commit `3d560bd` in this repository fixed CPU-only Linux
CI by adding exactly that guard to four module-level test helpers that
called into `GpuObjectiveState`: an unguarded module-level function that
launches device kernels failed the GPU-architecture constraint on a machine
with no accelerator. Moving a trainer body out from behind the guard would
have reintroduced that failure mode on the CPU half of the CI matrix.
`grow_tree_gpu` stays unguarded, as it already was and as it already
compiles.

The two overloads of each trainer differ in exactly one line, the builder
construction:

```mojo
var builder = GpuHistogramBuilder(data)           # private context
var builder = GpuHistogramBuilder(session, data)  # session's context
```

**Escape hatch.** The session-free overload is the default and passes
`NoLifecycle`, whose four methods are two increments and two no-ops. The
device call sequence it issues is byte for byte the one the trainer issued
before this stage. Deleting the session overloads and the `mut life: S`
parameter returns the file to its previous shape without touching a kernel.

**What a session buys today, honestly.** The lifecycle assertions, the
`PHASE_*` counters under `MOJOBOOST_GPU_TRACE=1`, and one context instead of
two when a predictor shares it (stage 4). It does **not** yet skip an
upload: the pool and residency ledgers record what a pooled path could skip,
but the buffers live in the builder, so a second builder on the same session
re-uploads. Fixing that needs the buffers to move into the session, which
A5 explicitly recommended deferring, and an owner on the estimator side,
which is central integration this task may not perform.

### Stage 2: active row ranges

**No change was needed and none was made.** The builder already routes
`begin_tree`, `apply_split`, and `enqueue_leaf` through `GpuActiveRows`, and
`grow_tree_gpu` already hands `apply_split` its `expected_left` from the
parent histogram's exact integer counts, which is what keeps a split fully
enqueued. This stage is recorded here so the map is complete and so the next
reader does not go looking for a missing wire.

One consequence worth stating for stage 4: row compaction reorders the
device's row permutation, and the validation predictor never reads it, so
order-preserving compaction (the open A1 item) cannot change a validation
score. The two are independent.

### Stage 3: device-side supported objectives

**Was.** `train_gpu` took the device objective path whenever
`supports_device_objective(objective)` held and neither bagging nor GOSS was
configured. There was no way to turn it off, and an unsupported objective
fell back silently with no way to ask why.

**Now.** `train_gpu.mojo` gains, next to the existing `SPLIT_SEARCH_*`
block:

- `OBJECTIVE_SOURCE_AUTO` / `_HOST` / `_DEVICE`
- `env_objective_source()`, reading `MOJOBOOST_GPU_OBJECTIVE`
- `resolve_objective_source(source)`, explicit outranks environment
- `device_gradients(supported, source, bagging, goss) raises -> Bool`

`device_gradients` is where the decision and the error messages live. Under
AUTO it returns exactly the predicate the trainer used before, so the
default path is unchanged. Under `OBJECTIVE_SOURCE_DEVICE` it raises with
the specific reason instead of downgrading:

- no device kernel for this objective, naming the switch to set instead
- row sampling configured, explaining that the sample is drawn host-side

`train_gpu` and `train_multiclass_gpu` take `objective_source` as a new
trailing parameter with an `AUTO` default. `train_multiclass_gpu` passes
`supported=True`, since `fill_softmax_gradients_device` covers softmax.
`train_custom_gpu` takes no such parameter by construction: the callback
lives on the host, which is where the raw scores it reads live.

**Escape hatch.** `objective_source=OBJECTIVE_SOURCE_HOST`, or
`MOJOBOOST_GPU_OBJECTIVE=host` with no rebuild, puts the run back on
`_fill_grad_hess` plus `upload_gradients`, which is the path every GPU round
took before the device objectives landed and the one bagging and GOSS still
take.

### Stage 4: GPU prediction primitives, reused for validation

This is the stage that was not wired at all before this task.

**`gpu_predict.mojo`.** `GpuPredictor.__init__` opened its own
`DeviceContext`, which A4 itself flagged as the one divergence to resolve
once A5 landed: a training run that also predicted opened two contexts, and
two contexts on one device are two queues. Now there are three
constructors, in the same shape `GpuHistogramBuilder` already uses:

| constructor | use |
| --- | --- |
| `GpuPredictor(n_features, n_outputs)` | private context. Unchanged behavior, unchanged signature, so every existing caller and the whole of `tests/parallel/test_gpu_predict.mojo` is untouched. |
| `GpuPredictor(session, n_features, n_outputs)` | the session's context, charged to `PHASE_ALLOC`. |
| `GpuPredictor(ctx, caps, n_features, n_outputs)` | a caller-supplied context and its already-queried capabilities. The two above land here. This is the one the trainer uses, with `builder.ctx` and `builder.caps`. |

Plus a `set_validation(session, data, target, weight)` overload that records
the held-out matrix under `ROLE_VALID` and its buffers under
`SLOT_VALID_BINS` / `SLOT_VALID_SCORE`, the slots A5 reserved for exactly
this. Bookkeeping only, no behavior change, and it re-uploads today for the
same reason the builder's session constructor does.

**`train_gpu.mojo`.** A new trainer, `train_gpu_with_valid`, mirroring
`train_with_valid` in boosting.mojo with `grow_tree` replaced by
`grow_tree_gpu`. It is not a second trainer in the sense the task warned
about: the loop is `train_with_valid`'s, and the only piece that varies is
the running validation score, behind a three-method trait.

```
trait GpuValidScorer          start / observe / loss
  _HostValidScorer            List[Float64], predict_row per row per round
  _DeviceValidScorer          GpuPredictor on the builder's context
```

Both scorers hand their raw scores to the same `_mean_loss` from
boosting.mojo, so the stopping rule is one definition rather than two. The
host scorer touches no device code at all, which is what makes it a usable
fallback rather than a second path through the same new modules.

The device scorer uploads the validation matrix, labels, and running score
vector once, and per round uploads only the tree that round grew (kilobytes)
and folds it in with one kernel. Scoring round `i` costs one tree walk per
row instead of `i` of them. It downloads the raw scores for the loss.

**Where the loss is computed, which is the A4 decision made explicitly.**
A4 offered two options, device metric reduction or host scoring from
`validation_raw()`, and called host scoring the conservative default. Taking
either one wholesale is wrong, so `device_loss_metric(objective)` decides
per objective and the rule is exact agreement, not availability:

| objective | loss reduced | why |
| --- | --- | --- |
| `SQUARED_ERROR` | device, `METRIC_L2` + `RESPONSE_IDENTITY` | `_mean_loss` sums `(raw - y)^2 / n`; the kernel sums `w*d*d` and divides by `check_metric_weight([], n)`, which is `n`. Term for term the same. |
| `L1` | device, `METRIC_L1` + `RESPONSE_IDENTITY` | same correspondence. |
| `BINARY_LOGISTIC` | host | `METRIC_BINARY_LOG_LOSS` exists and looks like a match. It is not: `_clamp_prob` floors at 1e-15 and `_clamp32` at 1e-7, because Float32 cannot hold `1 - 1e-15` apart from 1. A confidently wrong row is worth `-log(1e-15)` to the host and saturates at `-log(1e-7)` on the device, and confidently wrong rows are what a log-loss stopping decision turns on. |
| the other eight | host | no device kernel for cross entropy, gamma, tweedie, mape, fair, poisson, huber, or quantile. |

On the two device rows a round moves `n_valid / REDUCE_BLOCK` floats home
instead of `n_valid`, and two host passes over the validation rows and one
allocation per round go away. On every other row the run stops on exactly
the loss definition the CPU trainer stops on, which is the property worth
more than the transfer. The residual difference on the device rows is the
one this whole path already carries: Float32 terms over Float32 labels, so
the value agrees to Float32 tolerance rather than bit for bit.

Anyone extending `device_loss_metric` should be able to put the host
expression and the kernel expression side by side and see them agree,
clamps included. That test is written into its docstring because the next
reader will otherwise add binary log loss to it.

**Escape hatch.** `valid_scoring=VALID_SCORE_HOST` (the default) or
`MOJOBOOST_GPU_VALID_SCORING=host`. On that setting `train_gpu_with_valid`
is `train_with_valid` with GPU tree growth and nothing else, and it does not
construct a `GpuPredictor` at all.

### Stage 5: device-side split search

**Was, and still is.** `grow_tree_gpu` already routed to
`_grow_tree_gpu_device_search` when `resolve_split_search` said DEVICE, with
the searcher sharing the builder's context, so A2's blocking item (one
shared `DeviceContext`) is already closed and the per-node fence A2
described as the interim is already unnecessary.

**Now.** The three trainers take a `split_search` parameter and pass it to
every `grow_tree_gpu` call, so selecting the device scan no longer requires
an environment variable. The default is `SPLIT_SEARCH_AUTO`, which resolves
through `MOJOBOOST_GPU_SPLIT_STRATEGY` and then to the host scan, exactly as
before. No line of `gpu_split_search.mojo` changed.

---

## 2. Default path

What a plain `train_gpu(data, target, SQUARED_ERROR, params)` does today,
end to end, with no environment variable set. This is the path that has been
run on an M4 and it is unchanged by this task.

```
GpuHistogramBuilder(data)      private DeviceContext, bins uploaded once
device gradients               yes (built-in objective, no row sampling)
per round
  fill_gradients_device        gradients and hessians on the device
  begin_tree                   identity row permutation, kernel, no transfer
  per node
    enqueue_leaf               histogram over the node's own row range
    download_raw               one 3-plane copy, one synchronize
    _search                    host scan, Float64, identical to the CPU
    apply_split                stable device partition, expected_left known
  update_raw_device            raw scores advanced from the leaf ranges
```

Split selection is on the host. Validation, if any, is on the host.
Gradients are on the device. That is the shipped configuration and every
switch this task added defaults to it.

## 3. Fallback switches

| stage | parameter | environment | default resolves to |
| --- | --- | --- | --- |
| gradients | `objective_source` | `MOJOBOOST_GPU_OBJECTIVE=host\|device` | device where available, host otherwise (the shipped behavior) |
| split search | `split_search` | `MOJOBOOST_GPU_SPLIT_STRATEGY=host\|device` | host scan |
| validation | `valid_scoring` | `MOJOBOOST_GPU_VALID_SCORING=host\|device` | host tree walk |
| session | overload | none | no session, `NoLifecycle` |
| histogram accumulation | `strategy` (pre-existing) | `MOJOBOOST_GPU_HIST_STRATEGY` | device capability policy |
| tracing | none | `MOJOBOOST_GPU_TRACE=1` | off |
| staging depth | `staging_slots` | `MOJOBOOST_GPU_STAGING_SLOTS` | 2 |

In every row an explicit parameter outranks the environment, matching how
`strategy` already outranks `MOJOBOOST_GPU_HIST_STRATEGY`. Every switch is
runtime, so an A/B needs no rebuild.

## 4. Buffer lifetime

Vertical extent is lifetime. Everything in one column shares one
`DeviceContext` and therefore one in-order queue, which is the whole point
of stage 1 and stage 4.

```
                    fit ─────────────────────────────────────────────► end
DeviceContext       ████████████████████████████████████████████████████
  (session's, or the builder's own when no session is passed)

bins_dev            ████████████████████████████████████████████████████
  uploaded once at construction, read by every histogram and every
  partition kernel, never rewritten

objective state     ████████████████████████████████████████████████████
  labels, weights, device raw scores, softmax probabilities. Built once
  per fit when device gradients are on; absent otherwise.

validation set      ████████████████████████████████████████████████████
  valid_bins / valid_label / valid_weight / valid_raw / valid_resp.
  Built once by set_validation, only under VALID_SCORE_DEVICE.

grad_dev, hess_dev  ████│████│████│████│████│████│████│████│████│████│███
  rewritten once per round, by a kernel (device gradients) or by a
  staged copy (host gradients)

ensemble buffers        │▓   │▓   │▓   │▓   │▓   │▓   │▓   │▓   │▓   │▓
  nodes / values / cat_pool / roots. Reallocated per round by
  upload_ensemble, holding that round's one tree. Kilobytes.

row permutation     ─┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬──
  reseeded at every begin_tree; each live leaf owns a contiguous range
  inside it for the life of that tree

searcher buffers     ▒   ▒   ▒   ▒   ▒   ▒   ▒   ▒   ▒   ▒   ▒   ▒   ▒
  allocated per tree inside _grow_tree_gpu_device_search, only under
  SPLIT_SEARCH_DEVICE. See the follow-up below: this should be hoisted
  to the trainer once the path is measured.

out_dev, part_dev   ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ··
  overwritten per node, downloaded per node under host split search,
  read in place by the search kernels under device split search
```

Host synchronizations in the default path, per node: one, the histogram
download. Under `SPLIT_SEARCH_DEVICE`: one, the 136 byte record download.
Under `VALID_SCORE_DEVICE`, add two per round (`upload_ensemble` and
`validation_raw`), both of which also drain the training queue because the
queue is shared. That is correct and it is the cost of one context; it is
also why the device validation path is not the default until measured.

## 5. Known uncompiled risks

Ordered by how likely they are to bite on the first build.

1. **`trait RoundLifecycle` and the generic trainers.** Three trainer bodies
   are now `def _..._rounds[S: RoundLifecycle](mut builder, mut life: S,
   ...)`. The pattern was copied from `grow_tree_distributed[C: Collective]`
   and `allreduce_histogram[C: Collective]` in distributed.mojo, which
   compile today, including `mut comm: C` and trait methods declared with
   `def ... raises:` and a `...` body. `_train_custom_gpu_rounds` is the
   only two-parameter form (`[S: RoundLifecycle, F: GradHessFn]`) and is the
   most likely to need a syntax fix.
2. **`NoLifecycle` mutating its own Int fields from a `def`.** Matched
   against `SessionLifecycle._move_to`, which does the same.
3. **Overload resolution on the trainers.** Each trainer now has two
   overloads differing by a leading `mut session: GpuSession`. A
   `BinnedMatrix` cannot bind to a `GpuSession`, so resolution should be
   unambiguous, but Mojo overload rules interacting with eight trailing
   default arguments have not been checked against a compiler.
4. **The three `GpuPredictor.__init__` overloads.** Same shape as
   `GpuHistogramBuilder`'s three, which compile. `self = Self(...)`
   delegation is the pattern that file already uses.
5. **`set_validation` overloaded on a leading `mut session`.** Same
   reasoning, and the recursive call `self.set_validation(data, target,
   weight)` inside the session form has to resolve to the plain one.
6. **`GpuSession` and `GpuHistogramBuilder(session, data)` have never been
   run.** A5 states plainly that its module was never compiled, and the
   builder's session constructor has not been exercised by any test. The
   session overloads of the trainers are therefore the least trustworthy
   code in this diff. They are opt-in, and nothing reaches them unless a
   caller constructs a `GpuSession`.
7. **`_DeviceValidScorer` sharing the builder's context.** The predictor's
   `upload_bins`, `set_validation`, `upload_ensemble`, and `validation_raw`
   all call `ctx.synchronize()`, which now drains training work too. That is
   conservative and correct, never a race. What has not been checked on
   hardware is whether `enqueue_create_buffer` on a context that has
   training kernels in flight behaves the way both modules assume.
8. **Float32 validation scores changing an early-stopping outcome.** Real,
   documented in `_DeviceValidScorer`'s docstring, and the reason
   `VALID_SCORE_DEVICE` is not the default. Two rounds within Float32 noise
   of each other can order differently than on the host and pick a different
   `best_iteration`.
9. **CPU-only compilation of the extracted loops.** Addressed by the
   guards described in stage 1, but the evidence is mixed and worth writing
   down. `histogram_gpu.mojo` contains unguarded methods that construct a
   `GpuObjectiveState` and call `fill_grad_hess`, and it compiles on the
   CPU-only runners today; the test helpers that commit `3d560bd` had to
   guard did the same thing and did not. Nothing in a static read settles
   which property distinguishes them, so the conservative reading was
   taken: the extracted loops are guarded exactly as the bodies they came
   from were. If the guard turns out to be unnecessary, removing it is four
   dedents; if it turns out to be necessary somewhere else, the CPU CI job
   says so on the first build.
10. **Nested `var raw` shadowing** inside `_train_gpu_rounds` (the device
   branch declares one inside `if renews:` and the host path declares one at
   function scope). The nesting relationship is identical to the code that
   compiles today; it moved, it did not change.

## 6. The smallest future test and benchmark, per stage

None of these were run. All commands take the repository's build lock and
pin the worker count, matching the convention in the A-lane handoffs.

**Stage 1, session.** Smallest test: a new
`tests/parallel/test_gpu_session_training.mojo` that trains the same problem
twice, once through `train_gpu(data, ...)` and once through
`train_gpu(session, data, ...)`, asserts the predictions are equal row by
row (they should be bit-identical: same kernels, same order, same context
count), and asserts `session.life.rounds` and `session.life.trees` equal
`params.n_estimators`. It must skip cleanly with no accelerator.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_gpu_session_training.mojo
```

Benchmark: the existing GPU trainer bench with tracing on. It reports
nothing new by itself; the point is `session.trace()` output on a real
shape, which is the first measurement anyone can argue a drain removal from.

```
MOJOBOOST_NUM_WORKERS=1 MOJOBOOST_GPU_TRACE=1 nice -n 19 \
  tools/with_build_lock.sh pixi run bench-train-gpu
```

**Stage 3, gradient source.** Smallest test: add to
`tests/test_gpu_objectives.mojo` one case asserting that
`train_gpu(..., objective_source=OBJECTIVE_SOURCE_HOST)` and
`train_gpu(...)` agree to the Float32 tolerance that file already uses, and
two cases asserting `OBJECTIVE_SOURCE_DEVICE` raises under bagging and
raises for an objective without a kernel.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_gpu_objectives.mojo
```

Benchmark: the same shape twice, host and device gradients, which is the
first honest measurement of what the device objective path is worth.

```
MOJOBOOST_GPU_OBJECTIVE=host   nice -n 19 tools/with_build_lock.sh \
  pixi run bench-train-gpu
MOJOBOOST_GPU_OBJECTIVE=device nice -n 19 tools/with_build_lock.sh \
  pixi run bench-train-gpu
```

**Stage 4, validation.** Smallest test: a new
`tests/parallel/test_gpu_valid.mojo` with three assertions. One,
`train_gpu_with_valid(..., valid_scoring=VALID_SCORE_HOST)` and
`train_with_valid` on the same data stop at the same round and hold the same
tree count (both walk the same host arithmetic; only the histograms differ,
so use a problem whose stopping round is not on a knife edge). Two,
`VALID_SCORE_DEVICE` reaches the same tree count on a problem whose
per-round losses are separated well beyond Float32 noise. Three, the device
scorer's `validation_raw()` after round `i` matches a host-computed running
raw vector to `atol=1e-4`, which is the tolerance
`tests/parallel/test_gpu_predict.mojo` already uses. Four, the dispatch in
`device_loss_metric`: on a `SQUARED_ERROR` run the device-reduced loss and
`_mean_loss` over that round's `validation_raw()` agree to Float32
tolerance, and on a `BINARY_LOGISTIC` run `device_loss_metric` returns -1,
so the clamp difference cannot reach a stopping decision. That fourth
assertion is the one that fails loudest if someone widens the mapping.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_gpu_valid.mojo
```

Benchmark: there is no early-stopping benchmark in `bench/` today. The
smallest useful one is a new `bench/bench_gpu_valid.mojo` timing
`train_gpu_with_valid` on one shape under both scorers, reporting wall clock
and the round each stopped at. Until it exists there is no basis for making
the device scorer the default, and it must not become one.

```
MOJOBOOST_GPU_VALID_SCORING=host   nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src bench/bench_gpu_valid.mojo
MOJOBOOST_GPU_VALID_SCORING=device nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src bench/bench_gpu_valid.mojo
```

**Stage 5, split search.** Smallest test: add to
`tests/test_gpu_training.mojo` one case training the same problem with
`split_search=SPLIT_SEARCH_HOST` and `SPLIT_SEARCH_DEVICE` and asserting
both produce a usable model with the same node covers on a problem with no
near-tie splits. Node-for-node equality is not a valid assertion: the device
scan's gains are Float32 and A2 says so.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_gpu_training.mojo
```

Benchmark, the comparison A2 asked for and nobody has run. The device path
builds both children instead of subtracting the sibling, in exchange for
removing the per-node histogram transfer; that is a tradeoff, not a win.

```
MOJOBOOST_GPU_SPLIT_STRATEGY=host   nice -n 19 tools/with_build_lock.sh \
  pixi run bench-train-gpu
MOJOBOOST_GPU_SPLIT_STRATEGY=device nice -n 19 tools/with_build_lock.sh \
  pixi run bench-train-gpu
```

**The one command to run first**, before any of the above, is the existing
GPU training suite, because every stage above sits on the path it covers:

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_gpu_training.mojo
```

## 7. Central integration still required

Each item is a file this task does not own, with the exact edit.

### 7.1 `pixi.toml`, test registration

The new test files named in section 6 have to be appended to the `test`
task string, and the device-bound ones to `test-gpu`, in the same form as
the entries already there:

```
&& mojo run -I src tests/parallel/test_gpu_session_training.mojo
&& mojo run -I src tests/parallel/test_gpu_valid.mojo
```

### 7.2 `tests/parallel/api_snapshot_manifest.json` (task 20)

Its `mojo.exports_by_module` block is already stale for this area: the
`train_gpu` entry lists four names, and the tree exports far more. After
this task the entry should read

```json
"train_gpu": [
  "OBJECTIVE_SOURCE_AUTO", "OBJECTIVE_SOURCE_DEVICE", "OBJECTIVE_SOURCE_HOST",
  "SPLIT_SEARCH_AUTO", "SPLIT_SEARCH_DEVICE", "SPLIT_SEARCH_HOST",
  "VALID_SCORE_AUTO", "VALID_SCORE_DEVICE", "VALID_SCORE_HOST",
  "device_gradients", "device_loss_metric", "grow_tree_gpu",
  "resolve_objective_source", "resolve_split_search",
  "resolve_valid_scoring", "train_custom_gpu",
  "train_gpu", "train_gpu_with_valid", "train_multiclass_gpu"
]
```

and `gpu_runtime` needs `NoLifecycle` and `RoundLifecycle` added to whatever
list it grows. The manifest is hand written and nothing verifies it, so this
is a documentation debt, not a build break.

### 7.3 `src/mojoboost/device.mojo` and `apple_gpu_policy.mojo` (task 20)

**No edit is requested and none was made.** The one call this task would
have wanted does not exist yet, so it is written here instead of guessed at.

`resolve_device(device, n_rows, n_features, n_outputs)` today answers "CPU
or GPU" for training. Stages 3, 4, and 5 are each a second question with the
same shape: given this device and this workload, do the device gradients,
the device split scan, and the device validation scorer pay? The right home
for those is the authoritative policy module, as something like

```mojo
def gpu_stage_policy(caps: DeviceCaps, n_rows: Int, n_features: Int,
                     n_bins: Int) -> GpuStagePolicy
```

returning one field per stage. Until a benchmark exists for each stage, such
a function would have nothing to say, which is exactly why the switches
above are explicit and default to the established path. Whoever owns the
policy should take these three questions when the measurements land, and at
that point the `*_AUTO` constants in `train_gpu.mojo` become the place the
policy is consulted rather than the place the environment is read.

### 7.4 `src/mojoboost/custom_metric.mojo`, the metric training loop

`train_with_callbacks` is the one loop that owns validation sets, metric
suites, and early stopping for the whole library, and it is not this task's
file. `train_gpu_with_valid` deliberately does not try to replace it: it
covers the single-output objective-loss case, which is `train_with_valid`'s
contract, and stops there.

The exact edit that lane needs, once stage 4 has been run:

1. Per validation set, build one `GpuPredictor(session, data.n_features,
   n_outputs)` before the loop, then `set_validation(session, valid.data,
   valid.target, valid.weight)` and `reset_validation(base_scores)`.
2. Replace `_update_valid_raw` with `upload_ensemble(flatten_trees(
   round_trees, zeros, n_outputs, learning_rate))` followed by
   `accumulate_round()`.
3. Leave `_eval_round` alone and feed it `validation_raw()`. That vector is
   exactly what it already receives as `valid_raw[v]`, so this is a
   one-line substitution and keeps every metric in the suite, including the
   ones the device cannot reduce.
4. `_StopState`, `observe`, and `exhausted` are untouched.

The guard in `python/mojoboost/__init__.py` that refuses an `eval_set`
combined with a non-CPU device comes out only after that lands and is
tested, not before.

### 7.5 `src/mojoboost/model.mojo`, `bindings/`, `python/`

Unchanged from A4's section 3 to 5: batched `Model.predict_batch`, the two
new binding functions, and the `device=` keyword on the estimators'
`predict`. Nothing in this task moves those forward or makes them harder.
One correction to A4's note, now that this task has landed: the estimator
should hold a `GpuSession` and build its `GpuPredictor` from it, rather than
letting the predictor open its own context. The constructor for that now
exists.

### 7.6 `bench/bench_gpu_valid.mojo`

Does not exist. Specified in section 6. Without it the device validation
scorer stays off by default, which is the correct state.

## 8. Not done, and why

- **Routing the builder's three `ctx.synchronize()` calls through the
  session's hazard tracker.** A5 lists four assumptions that have to be
  settled first, the load-bearing one being whether `enqueue_memset`,
  `enqueue_copy`, and `enqueue_function` share one in-order queue on Metal,
  CUDA, and HIP. That is answered by reading the `DeviceContext`
  implementation for the installed toolchain, not by testing, and not by
  this task. A5's own recommended first step, adding `note_device_read` /
  `note_device_write` calls with the drains left in place, is a sweep across
  every enqueue in `histogram_gpu.mojo` and `gpu_predict.mojo`; it is
  measurable and risk-free and it is the obvious next task, but it is a
  large mechanical diff that would have buried the five stages above.
- **Hoisting the `GpuSplitSearcher` out of the per-tree scope.** A2 asked
  for one searcher per fit with `max_records=num_leaves`. Today
  `_grow_tree_gpu_device_search` constructs one per tree. That is one small
  device allocation per tree, not per node, and hoisting it means either a
  third `grow_tree_gpu` overload or a signature change that
  `tests/` and `bench/` call. Not worth the API churn before the device
  scan has been benchmarked at all.
- **Composing device gradients with validation.** `train_gpu_with_valid`
  keeps its gradients on the host because early stopping reads the host raw
  scores `_fill_grad_hess` needs. Composing them means either downloading
  the device raw scores per round (which is what the renewing objectives
  already pay) or teaching the device objective state to serve
  `_fill_grad_hess`'s inputs. Either is a stage of its own.
- **Multiclass validation.** `train_multiclass_with_valid`'s GPU
  counterpart. The predictor handles `n_outputs > 1` already and
  `_multiclass_mean_loss` exists in boosting.mojo, so this is a small
  follow-up, but it doubles the surface of stage 4 before any of it has
  compiled.
- **Feature contributions, sparse prediction, device-side binning.** A4's
  follow-ups, unchanged. Device-side binning in particular is a correctness
  problem before it is a performance one, since a Float32 edge search can
  move a row into a different bin and change its prediction by a whole leaf
  value.
- **Any `auto` policy that picks a device path on size.** Explicitly not
  added. There is no crossover measurement for any of the three stages, and
  `python/mojoboost/device_selection.py` already records that the one GPU
  training measurement that exists came out slower than the CPU on an M4.
