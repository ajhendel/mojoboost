# Apple A5: persistent GPU runtime and async scheduling

Lane A5. Files added, all new, no existing file touched:

- `src/mojoboost/gpu_runtime.mojo`
- `tests/parallel/test_gpu_runtime.mojo`
- `handoffs/apple_a5_runtime.md` (this file)

Nothing in the repository imports `gpu_runtime` yet, so this lane changes no
behavior on its own. Everything below marked "integration" is a central edit
this lane is not permitted to make.

**Test status: not run.** The A5 task explicitly forbids running Mojo, pixi,
builds, or tests, so `tests/parallel/test_gpu_runtime.mojo` has been written
but never compiled or executed. Nothing here is claimed to compile, pass, or
be faster.

## What the module provides

`GpuSession` is the per-estimator object that outlives a single `fit`. It
owns exactly one `DeviceContext` plus the bookkeeping that makes reuse safe:

| Piece | Answers |
| --- | --- |
| `SessionLifecycle` | is this call legal here? (`new`/`open`/`round`/`tree`/`closed`) |
| `HazardTracker` | does this host access need a drain? |
| `StagingRing` | which pinned staging slot do I fill, and must I wait first? |
| `ResidencyLedger` | is this matrix already on the device? |
| `PoolLedger` | allocate, grow, or reuse this buffer? |
| `KernelRegistry` | is this the first launch of this kernel? |
| `PhaseCounters` | compile / alloc / transfer / kernel / sync / cleanup |

Every one of those is a plain value type with no device dependency, which is
why the whole lifecycle including illegal transitions is testable on a
CPU-only machine. `GpuSession` is the only struct in the file that touches
`DeviceContext`, and the test file deliberately does not construct it.

The module also contains a description of what `histogram_gpu.mojo` enqueues
today (`model_upload_gradients`, `model_set_features`, `model_begin_tree`,
`model_build_leaf`, `model_apply_split`) and a replay of one boosting round
through it (`audit_round`). That exists so the synchronization claims below
are executable rather than prose, and so a later change to the pipeline
updates one description instead of two.

### Environment contract

Follows the `MOJOBOOST_` convention in `parallel.mojo` and `gpu_tiling.mojo`:

- `MOJOBOOST_GPU_TRACE=1` enables the phase counters. Off by default; a
  disabled session never reads the clock. Call counts are always kept, since
  they cost an integer add and are what the tests assert on.
- `MOJOBOOST_GPU_STAGING_SLOTS=N` sets the staging ring depth. Default 2,
  clamped to `[1, 8]`. `1` reproduces today's single-buffer behavior exactly,
  which is the way to A/B the ring without a rebuild.

An explicit `staging_slots` argument to `GpuSession.__init__` outranks the
environment, matching how `strategy` outranks `MOJOBOOST_GPU_HIST_STRATEGY`.

## The dependency analysis

The model: a `DeviceContext` queue is in order, so device work never needs a
host drain to see earlier device work. A host drain is needed in exactly two
cases, and `HazardTracker` tracks both per resource:

- the host **reads** memory with an unretired device write, or
- the host **writes** memory with an unretired device read or write.

`ctx.synchronize()` drains everything, so one required drain clears every
resource. That is why the tracker counts *elided* checks: an elided check is
a place today's code drains unconditionally and the model says it need not.

### The three `ctx.synchronize()` calls in histogram_gpu.mojo

**1. `__init__`, after uploading the binned matrix (histogram_gpu.mojo:508).
Not removable as written.** The copy reads `data.bins`, host memory owned by
the caller, and the constructor cannot outlive the caller's willingness to
keep it alive. Removing it requires either (a) the session staging the bins
through pinned memory it owns, or (b) a documented lifetime contract that
`data` outlives the session. (a) doubles peak host memory for the matrix;
(b) is a public API change to `GpuHistogramBuilder`. Neither is free, and
this drain happens once per session, so it is the least valuable of the
three to attack. With a session it is also amortized: a second fit on the
same matrix skips both the copy and the drain via `ResidencyLedger`.

**2. `stage_gradients`, before overwriting the pinned staging buffers
(histogram_gpu.mojo:583). Redundant today, and must be replaced by a hazard
check, not simply deleted.** Its job is real: the host is about to overwrite
memory a queued copy may still be reading. `StagingRing` plus
`sync_for_host_write(RES_STAGE)` is the correct replacement.

The honest finding, which `test_audit_with_one_staging_slot_still_needs_no_waits`
pins down, is that deepening the ring buys nothing on its own. Every
`build_leaf` drains the queue, so by the time the next round stages its
gradients, the previous copy has long retired and the check elides. A
two-slot ring and the removal of the per-node download drain are worth
something *together* and nothing separately, and they must not be credited
to each other.

**3. `download_raw`, after copying the histogram to pinned host memory
(histogram_gpu.mojo:766). Not removable while split search runs on the
host.** The host reads the downloaded histogram to search it for a split, and
the download is device work: this is the definitional case for a drain. It
is one drain per tree node, so it is also the expensive one. It goes away
when split search moves onto the device, which is lane A2's territory, not a
scheduling change. `audit_round` counts it as `required`, and will keep
doing so until the model of `model_build_leaf` changes.

### Implicit synchronizations: `map_to_host()`

Three sites use `map_to_host()` to write device buffers from the host:
`__init__` (initial `feat_dev` fill), `set_features`, and the bagged branch
of `begin_tree`. `map_to_host` copies in both directions on every use and
synchronizes internally, which the module docstring already calls out as the
reason gradients stage through pinned buffers instead.

Converting these to pinned staging plus a one-way `enqueue_copy` is the same
transformation gradients already had. The model says the hazard check that
replaces the implicit drain (`sync_for_host_write(RES_FEAT)` and
`(RES_LEAF)`) elides in practice, because every tree is preceded by a
histogram download that drained. See
`test_set_features_is_free_after_a_download`.

That elision is only correct if the implicit drain inside `map_to_host` is
the *only* thing making those writes safe today. It is, under the in-order
queue assumption, but see the open questions below.

### What is already correct and should not be changed

`apply_split` followed by `build_leaf` needs no host involvement: the
partition kernel writes the leaf ids and the histogram kernel reads them,
both on the device, in order. Today's code correctly does not drain between
them, and `test_device_only_operations_queue_without_waiting` asserts the
model agrees.

### Open questions that must be answered before any drain is removed

The model is only as good as these four assumptions, none of which this lane
verified:

1. **Queue ordering across operation kinds.** The model assumes
   `enqueue_memset`, `enqueue_copy`, and `enqueue_function` all issue into one
   in-order queue on Metal, CUDA, and HIP. If any backend puts copies on a
   separate queue, every elision in this document is wrong. This is the load
   bearing assumption and it should be settled first, by reading the
   `DeviceContext` implementation for the toolchain in use, not by testing:
   an out-of-order queue that happens to complete in order is exactly the bug
   that survives testing.
2. **When a device-to-host `enqueue_copy` is visible.** `download_raw` copies
   into a `HostBuffer` and then synchronizes. If completion is only
   guaranteed at `synchronize()`, the drain is required as written; if a
   finer-grained wait exists, this is where it belongs.
3. **What `map_to_host()` guarantees on each backend.** If it synchronizes
   only where it must, the three sites above may already be cheaper than the
   model assumes, and the conversion to pinned staging buys less.
4. **Lifetime of `data.bins`** across `GpuHistogramBuilder.__init__`, for
   drain 1.

Separately: there is no measurement anywhere in this lane. `audit_round`
counts host synchronizations, not time. A drain that the model calls
unnecessary may still be cheap, and removing it may show nothing. The counts
say what is *safe* to remove; `bench/bench_train_gpu.mojo` and the phase
counters have to say whether it is *worth* removing.

## Integration required (central edits this lane may not make)

### 1. Run the test

`pixi.toml`, the `test` task, needs one more entry. It is device-free, so it
belongs in the CPU list next to `test_gpu_tiling.mojo`, not in `test-gpu`:

```
&& mojo run -I src tests/parallel/test_gpu_runtime.mojo
```

`tests/parallel/` is a new directory created by this round. Whoever wires the
first lane in should confirm `mojo run -I src` resolves a test in a
subdirectory the same way it does one in `tests/` (it should: `-I src` is the
only include path that matters and the test path is explicit).

### 2. Export the module

`src/mojoboost/__init__.mojo`, next to the existing `gpu_tiling` and
`histogram_gpu` imports. Prefer a selective list over a blanket re-export:
several names here (`state_name`, `phase_name`, `resource_name`,
`kernel_name`, `role_name`) are generic enough to collide with another
lane's, and A6 lands a policy module in the same round.

```mojo
from .gpu_runtime import (
    GpuSession,
    HazardTracker,
    MatrixIdentity,
    PhaseCounters,
    PoolLedger,
    ResidencyLedger,
    SessionLifecycle,
    StagingRing,
    audit_round,
    bins_fingerprint,
)
```

Nothing else in the package needs to see the constants unless the builder
integration below happens.

### 3. Borrow the session in `GpuHistogramBuilder` (the real integration)

`src/mojoboost/histogram_gpu.mojo`. The builder currently constructs its own
`DeviceContext` in `__init__`. The change is a second constructor that takes
a session instead, keeping the existing one as a thin wrapper so no caller
breaks:

- Add `def __init__(out self, mut session: GpuSession, data: BinnedMatrix,
  strategy: Int = STRATEGY_AUTO) raises`, which uses `session.ctx` rather
  than `DeviceContext()`, `session.caps` rather than a fresh
  `query_device_caps`, and `session.request_buffer(SLOT_*, n_elems,
  elem_bytes)` before each `enqueue_create_buffer` so that a `POOL_REUSE`
  keeps the buffer the session already has.
- Route `session.admit_matrix(ROLE_TRAIN, MatrixIdentity(data.n_rows,
  data.n_features, data.n_bins, bins_fingerprint(data.bins, data.n_rows,
  data.n_features, data.n_bins)))` around the binned-matrix upload; skip the
  copy and its drain when it returns `False`.
- Replace the three `self.ctx.synchronize()` calls with
  `session.sync_for_host_read(...)` / `session.sync_for_host_write(...)`
  **only after the open questions above are settled.** Until then, keep the
  drains and add the `note_device_read` / `note_device_write` calls anyway:
  the tracker then reports what could have been elided, with
  `MOJOBOOST_GPU_TRACE=1`, on real workloads and with no behavior change.
  That is the intended first step, and it is measurable without risk.
- Wrap each `enqueue_function` in `session.note_kernel(KERNEL_*, started)`
  and each `enqueue_copy` in `session.note_transfer(started)`.

Buffer ownership needs a decision that this lane deliberately did not make:
`PoolLedger` tracks sizes, not buffers, so either the builder keeps owning
the `DeviceBuffer` values and consults the ledger, or the buffers move into
the session. The first is a much smaller change and keeps the builder
usable standalone; the second is what actually makes two builders share
memory. Start with the first.

### 4. Thread the session through the trainers

`src/mojoboost/train_gpu.mojo`. `train_gpu`, `train_custom_gpu`, and
`train_multiclass_gpu` each construct `GpuHistogramBuilder(data)`. Add an
optional session parameter and, when present, call
`session.begin_round()` per boosting round, `session.begin_tree()` /
`session.end_tree()` around each `grow_tree_gpu`, and `session.end_round()`
at the end of the round. `train_multiclass_gpu`'s per-class loop is exactly
the `tree -> tree` transition the state machine allows without an
intervening round boundary.

No trainer should call `session.close()`: the session outlives the fit. The
estimator that owns it closes it.

### 5. Estimator ownership

`python/mojoboost/` and `bindings/_mojoboost.mojo`. The point of a
per-estimator session is that `fit` twice, or `fit` then `predict`, reuses
the device state. That needs an owner on the Python side with a defined
teardown, which means the ABI grows a create/destroy pair and the estimator
closes the session in `__del__` (or on refit with an incompatible dataset).
This is the largest piece of the integration and it should not start until
step 3 is landed and traced.

### 6. Validation matrices (coordination with lane A4)

`ROLE_VALID`, `SLOT_VALID_BINS`, `SLOT_VALID_SCORE`, `RES_VALID_BINS`, and
`RES_VALID_SCORE` exist so the GPU prediction and validation-scoring path
does not re-upload the held-out matrix once per early-stopping round. A4
should call `session.admit_matrix(ROLE_VALID, ...)` and take its buffers from
the same pool. Nothing in A5 depends on A4 landing.

### 7. Documentation

No `docs/` change is required by this lane. If `docs/GPU_VALIDATION.md`
grows a scheduling section later, the two facts worth recording there are
the environment contract above and that `audit_round` is the model the
synchronization claims come from.

## Risks

- **Unrun code.** The module and its test have never been compiled. Expect
  syntax and API fixes on first build. The device-bound half (`GpuSession`)
  is the riskiest: it uses `DeviceContext()`, `query_device_caps`, and
  `ctx.synchronize()`, mirroring `histogram_gpu.mojo`, but it has not been
  checked against the toolchain.
- **`KernelRegistry` does not hold device function handles.** The task asked
  for precompiled kernel handles. Binding real handles needs a
  `DeviceContext` compile API that this lane could not verify against the
  installed toolchain (`docs.modular.com` returned nothing for it and the
  packaged `.mojoc` files carry no readable symbols), and inventing a
  signature would have been worse than counting. What is there is the
  attribution point: the first launch of each kernel is charged to
  `PHASE_COMPILE`, and the registry is where handles hang off once the API
  is confirmed. This is the one part of the A5 brief that is stubbed rather
  than built, and it is called out here rather than buried.
- **Fingerprint cost.** `bins_fingerprint` hashes every cell. That is one
  host pass over the same bytes an upload would move, so it is cheaper than
  the upload it can skip, but it is not free and it is paid even on the miss.
  If residency turns out to miss almost always, gate it behind an explicit
  "same dataset" assertion from the estimator instead. A sampled hash is not
  an acceptable substitute: the failure mode is training on the wrong data.
- **The elision counts are a model, not a measurement.** They say a drain is
  unnecessary under the in-order-queue assumption. They do not say it is
  expensive, and no code in this lane got faster.
