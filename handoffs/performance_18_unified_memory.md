# Task 18: Apple unified-memory integration seam

## What this lane produced

Three files, one new.

- `src/mojoboost/unified_memory_policy.mojo` (**new**) is the seam: a pure
  policy layer that answers "which allocation route does this buffer use, and
  what does that route oblige its owner to do". It opens no device, allocates
  nothing, holds no pointer, and returns `copy_staged` for every buffer role in
  every configuration this repository can currently construct.
- `bench/apple/unified_memory.mojo` (strengthened) now measures the
  device-to-host direction as well as host-to-device, models both of
  mojoboost's transfer lifetimes (`rewrite` for gradients, `resident` for the
  binned matrix, with an explicit CPU-writes-after-GPU-reads round), reports
  the queue drains each route owed and the bytes each route asked this library
  to copy, can hold a second device-resident buffer to model a resident
  validation matrix, and imports its route vocabulary from the policy module so
  the experiment and the policy cannot drift apart.
- `docs/APPLE_UNIFIED_MEMORY.md` (rewritten) carries the methodology, the
  mandatory external capture, the per-role structural table, the evidence
  ladder, the output grammar, and the table of what each possible outcome does
  and does not license.

## What this lane did NOT do

Stated first because it is the part most likely to be misread later.

- **Nothing was compiled and nothing was run.** No `mojo build`, no `mojo run`,
  no benchmark, no profiler, no `powermetrics`. The round's contract forbids
  it, and the driver in particular is a heavy job that must be serialized on a
  machine running parallel lanes.
- **No route was enabled, and none can be.** `EvidenceLedger.installed()`
  reports `none` for every route, `ENABLE_LEVEL` is the end-to-end trainer
  rung, and `resolve_from_env` therefore returns `copy_staged` for every role.
  The only way to get anything else is an explicit `MOJOBOOST_GPU_TRANSFER`
  request plus an explicit `MOJOBOOST_GPU_TRANSFER_UNPROVEN=1`, and nothing
  reads either variable yet.
- **The phrase "zero copy" appears nowhere as a conclusion.** The document
  separates unified physical memory (true, free, implies nothing), "mojoboost
  issued no copy" (a fact about our source, reported as
  `copy_bytes_issued_total`), and "no duplication happened" (unproven, needs
  the external capture and the trace).
- **No central file was touched.** `histogram_gpu.mojo`, `train_gpu.mojo`,
  `gpu_runtime.mojo`, `gpu_predict.mojo`, `gpu_active_rows.mojo`,
  `binning.mojo`, `gpu_tiling.mojo`, `device.mojo`, Python, bindings, tests,
  other benchmarks, packaging, and workflows are all unmodified. Everything
  they would need is in "Required central changes" below.
- **Routes that were considered and not compiled are reported `not_probed`,
  never `unsupported` and never `ok`.** That is two routes: the non-owning
  `DeviceBuffer` over a host pointer, in each direction.

## Structural findings that need no run

These come from reading the code, not from measuring it, and none of them can
be overturned by a benchmark. They are listed first because three of them bound
what the whole experiment could be worth.

### 1. No kernel signature changes anywhere

Every kernel in the GPU path takes `MutPointer[T, MutAnyOrigin]`, not a buffer
type. A shared or mapped route therefore changes only *which pointer the launch
site passes*. That is why one checksum kernel in the driver can serve every
route, and it is why an eventual integration is a change to buffer ownership in
`GpuHistogramBuilder` and nothing deeper.

### 2. The binned matrix cannot take a shared route as the data is owned today

`BinnedMatrix.bins` is a plain heap `List[UInt8]` built by `binning.mojo` and
owned by the caller. `GpuHistogramBuilder.__init__` copies it to the device
(`enqueue_copy(dst_buf=self.bins_dev, src_ptr=data.bins.unsafe_ptr())`) and
keeps no reference to it. To hand a kernel that host pointer, the bytes would
first have to be copied into a runtime allocation, which is exactly the copy
the route was supposed to remove: two host-side copies instead of one host copy
and one device buffer, which on a unified pool is not an improvement.

So the largest buffer in the system, the one where duplication actually costs
gigabytes, cannot be de-duplicated by any route choice. It requires
`binning.mojo` to bin *into* a device-visible allocation. That is a change to
another module's data ownership and it is listed as a required external edit.

Today's duplication, recorded in the policy module:

| Path | Copies of the matrix held | Where |
|---|---|---|
| Training (`ROLE_BINS`) | 2 | caller's `List[UInt8]`, `bins_dev` |
| Resident validation (`ROLE_VALID_BINS`) | 2 | caller's `List[UInt8]`, `valid_bins_dev` |
| Batch scoring (`ROLE_BATCH_BINS`) | 3 | caller's `List[UInt8]`, pinned `stage_bins`, `bins_dev` |

The first two are the same shape: both `GpuHistogramBuilder.__init__` and
`GpuPredictor.upload_validation` allocate a buffer sized exactly to their matrix
and copy straight out of the caller's list, with no pinned staging copy. They
are separate rows because they are resident *at the same time*: a fit that
scores a held-out set holds four copies of two matrices, which is what
`MOJOBOOST_UM_HOLD_MIB` models in the driver.

The batch-scoring path's third copy needs care, and my first reading of it was
wrong. `GpuPredictor.upload_bins` stages the batch into a pinned `stage_bins`
before uploading, and it is tempting to call that redundant and delete it. It is
not: `bins_dev` is sized to the **high-water batch** rather than to this batch,
and `enqueue_copy(dst_buf=..., src_ptr=...)` moves the whole destination buffer,
so copying from the caller's exactly-sized list would read past its end.
`_check_matrix` guarantees the source is exactly `n_rows * n_features`, so this
is a real out-of-bounds read, not a theoretical one.

Removing that copy therefore needs one of three things: a sub-range copy API (no
such API is verified here), an exact-size reallocation of `bins_dev` per batch
(trading the copy for an allocation), or a shared route. Which makes
`ROLE_BATCH_BINS` the one bins-shaped role where a shared route is structurally
available today, because unlike the other two its bytes already sit in a runtime
allocation the session owns. It is still gated on the same evidence as
everything else.

### 3. The shipped default gradient path already transfers nothing

`train_gpu` uses `fill_gradients_device` for built-in objectives without
bagging or GOSS, which computes gradients on the device straight into
`grad_dev` / `hess_dev`. On that path `stage_gradients` and `upload_staged` are
never called and no per-row byte crosses the boundary at all.

A shared gradient route therefore only affects the bagging, GOSS,
custom-objective, and renewal paths. It is worth `2 * n_rows * 4` bytes per
round on those paths (8 MB per round at 1M rows), and worth exactly nothing on
the default one. Any hypothesis about gradient transfers has to name which path
it is about.

### 4. The histogram download is the frequent transfer, and it is the one
structurally blocked

`download_raw` copies `3 * n_features * n_bins` Int32 back and drains the queue
once per node. At 100 features and 256 bins that is 307 KB per node, and a
200-round fit growing 31 leaves per tree performs it about six thousand times,
against one binned-matrix upload for the whole session. If any transfer is
worth attacking on frequency, it is this one.

It is also the one the policy module refuses structurally
(`BLOCK_DEVICE_WRITTEN_ATOMICS`). `STRATEGY_ATOMIC` folds every threadgroup's
partial into `out_dev` with `Atomic.fetch_add`, `STRATEGY_TILED` reduces
partials into it with plain stores, and which of the two runs is decided per
node from that node's row count. Whether a global atomic is coherent against
host-visible memory is unverified on Metal, CUDA, and HIP alike here, the
failure mode is a silently wrong histogram rather than a raise, and a route
resolved once per session cannot be safe under only one of the two strategies. The driver's `out_host_direct`
route exists to answer exactly this question, with the same atomic the real
kernels use, and a `wrong` there closes the question outright, which is a
result worth having.

### 5. A shared route invalidates the staging ring's overlap argument

`StagingRing` in `gpu_runtime.mojo` assumes a staging slot is free once the
copy reading it has retired, which happens early in a round. On a shared route
there is no copy: the slot is read by *every histogram kernel of the whole
tree*, so it is not free until the round drains. A two-slot ring then buys
nothing and, worse, would report an overlap that is not there. On `map_write`
the ring does not apply at all, because there is no staging buffer to rotate:
the host writes through the device buffer's own mapping.

`SyncContract.staging_ring_applies` is False on every route but the two copy
ones for this reason, and `note_read_on_each_launch` is True there instead of
`note_read_on_publish`. Whoever implements a shared route must move the
`note_device_read(RES_STAGE)` call from publish time to launch time, or the
dependency model will say the host may overwrite a buffer the device is still
reading.

### 6. A route change cannot remove a synchronization

`host_rewrite_needs_drain` is True on every route including the default. The
`ctx.synchronize()` at the top of `stage_gradients` exists because the host is
about to overwrite memory the device may still be reading, and a route change
only moves *which event* the wait is for (`RETIRE_ON_COPY` to
`RETIRE_ON_KERNEL`), always later, never earlier. Every route except the two
copy ones retires on the kernel, `map_write` included: its publish may cost a
transfer and its next write still waits for the kernels, which is why
`sync_contract` branches per route family rather than deriving everything from
whether the route publishes by copy. Lane A5's proposal to elide
synchronizations and any non-default route interact here and must be reviewed
together, not merged independently.

## The integration seam, precisely

Everything in this section is contingent on evidence that does not exist. None
of it should be implemented until `docs/APPLE_UNIFIED_MEMORY.md`'s status table
has a route at `ok` **and** the external capture backs it **and**
`bench/bench_train_gpu.mojo` confirms it end to end. That is what
`ENABLE_LEVEL` encodes.

### Where the decision is made

In `GpuSession` (`gpu_runtime.mojo`), once per session, next to the pool and
the residency ledger. Not in `GpuHistogramBuilder`, and not per buffer: two
modules must not each grow their own device-memory policy. The session calls

```mojo
var route = resolve_from_env(ROLE_GRAD, self.caps.unified_memory)
```

once per role it owns and hands the resolved `RouteDecision` to the builder.
`unified_memory` comes from the device profile the caller already queries;
`unified_memory_policy` deliberately takes it as a plain `Bool` so it never
needs a `DeviceContext` and stays testable without an accelerator.

### The wrapper the builder would hold

`GpuHistogramBuilder` hardcodes the staged-copy route in five places:

1. the constructor's one-time binned-matrix upload
   (`histogram_gpu.mojo`, `enqueue_create_buffer` + `enqueue_copy` +
   `synchronize`);
2. `stage_gradients`, the per-round Float64-to-Float32 conversion into
   `self.stage_g` / `self.stage_h`;
3. `upload_staged`, the two `enqueue_copy` calls into `grad_dev` / `hess_dev`;
4. `enqueue_leaf`, which passes `bins_dev.unsafe_ptr()`,
   `grad_dev.unsafe_ptr()`, `hess_dev.unsafe_ptr()`, `out_dev.unsafe_ptr()`;
5. the field declarations themselves.

The minimal seam is a per-buffer wrapper that owns whichever allocation the
resolved route needs and answers one question, "what pointer do I hand the
kernel":

```mojo
# Sketch only. Not written, not compiled.
struct Payload[dtype: DType](Movable):
    var dev: DeviceBuffer[Self.dtype]     # copy routes only
    var host: HostBuffer[Self.dtype]      # every route
    var decision: RouteDecision

    def kernel_ptr(self) -> MutPointer[Scalar[Self.dtype], MutAnyOrigin]:
        if self.decision.contract.publish_is_copy:
            return self.dev.unsafe_ptr()
        return self.host.unsafe_ptr()

    def publish(mut self, mut ctx: DeviceContext) raises:
        if self.decision.contract.publish_is_copy:
            ctx.enqueue_copy(dst_buf=self.dev, src_ptr=self.host.unsafe_ptr())
```

With that, `stage_gradients` keeps its conversion loop unchanged (it already
writes into a `HostBuffer`), `upload_staged` becomes `self.grad.publish(...)`
plus the same for hessians and is a no-op on a shared route, and `enqueue_leaf`
substitutes `kernel_ptr()` for `.unsafe_ptr()`. No kernel changes.

`ROLE_BINS` does not get a `Payload` until finding 2 is addressed, and
`ROLE_HIST_OUT` does not get one until the driver answers the atomics question.

### Pointer and buffer lifetimes

The rule the seam has to preserve, in one sentence: **on a copy route the host
allocation must outlive the copy; on a shared route it must outlive every
kernel launch that took its pointer, and the buffer that owns it must outlive
the session.**

| Buffer | Allocated by | Owned by | Host source | Must outlive | Freed at |
|---|---|---|---|---|---|
| `bins_dev` | builder constructor | builder | caller's `data.bins`, borrowed only during the constructor's copy | every kernel launch of the session | builder drop |
| `stage_g` / `stage_h` | builder constructor | builder | caller's `grad` / `hess` lists, borrowed only during conversion | copy route: the enqueued copy. shared route: every histogram kernel of the tree | builder drop |
| `grad_dev` / `hess_dev` | builder constructor | builder | `stage_*` on the host path, nothing on the device-objective path | every histogram kernel of the round | builder drop |
| `out_dev` | builder constructor | builder | none (device written) | the download copy | builder drop |
| `host_out` | builder constructor | builder | none | copy route: the download. shared route: every kernel that accumulated into it | builder drop |
| `stage_rows` | `GpuActiveRows` | rows | caller's `bag` list, borrowed during staging | copy route: the enqueued copy. shared route: every partition and histogram launch of the tree | rows drop |
| `stage_bins` (batch scoring) | `GpuPredictor` | predictor | caller's `data.bins` | copy route: the upload copy. shared route: every predict launch over the batch | predictor drop |
| `valid_bins_dev` | `GpuPredictor` | predictor | caller's `data.bins`, borrowed during `upload_validation` | every validation scoring launch | predictor drop or the next `upload_validation` |

Three lifetime hazards a shared route introduces, none of which exist today:

1. **Caller-owned host memory must never become a kernel pointer.** Every
   `List` in the table above belongs to a caller whose lifetime the session
   does not control. The `Payload` wrapper owns runtime allocations only, and
   `structural_support` refuses `host_direct` for exactly the roles whose
   source is caller-owned.
2. **A buffer grow frees memory a kernel may hold a pointer to.**
   `PoolLedger.request` returning `POOL_GROW` means the old buffer is replaced.
   On a copy route the drain before the free is enough; on a shared route the
   free must additionally wait for every launch that took the old pointer,
   which is the same `RETIRE_ON_KERNEL` wait.
3. **Session teardown order is load-bearing.** `GpuSession.close` drains, then
   releases the pool. On a shared route releasing a host buffer the device is
   still reading is a use-after-free rather than a wasted copy, so the drain in
   `close` becomes mandatory rather than conditional on `any_pending()`.

### Invalidation rules

A resolved `RouteDecision` is not permanent. It must be re-resolved, and any
buffer it governs rebuilt, when any of these happen:

- **The environment changes between sessions.** Resolution is per session, so a
  process that constructs a second session picks up a changed
  `MOJOBOOST_GPU_TRANSFER`. A decision is never cached across sessions.
- **Residency is evicted or re-admitted.** `ResidencyLedger.admit` returning
  True for a role whose matrix identity changed means the device copy is being
  replaced; the route for that role is re-resolved before the new upload,
  because a shared route's buffer identity is the host buffer, not the device
  one.
- **A pool slot grows.** `POOL_GROW` invalidates every pointer into that slot,
  so any kernel argument derived from it is stale. See hazard 2 above.
- **`set_features` narrows the active set.** This does not invalidate a route
  (it re-derives tiling only), and it is listed because it looks like it might.
- **The device changes.** Not currently possible in one session, and if it ever
  becomes possible, `unified_memory` changes and every decision is void.

What does *not* invalidate a decision: a new tree, a new boosting round, a new
node, or a `begin_tree` reseed. Those all reuse the same buffers, which is the
entire reason a per-session decision is the right granularity.

## Measurable hypotheses

Each of these is falsifiable by a specific run, and each names what a negative
result would settle. Nothing here is a prediction with confidence attached.

**H1. `host_direct` compiles and passes the checksum.** Run: the default
driver invocation. Falsified by `unsupported` (the runtime rejects a host
pointer as a kernel argument) or `wrong` (it accepts it and the device reads
something else). Either negative closes the input-side question for this Mojo
version and is worth documenting loudly.

**H2. On a route that issues no copy, one payload-sized allocation is
resident, not two.** Run: the default invocation under `/usr/bin/time -l`
with `vm_stat` brackets. Falsified by a peak RSS showing two. A negative here
means the runtime duplicates regardless of what our source does, which settles
Claim 2 in the negative on this stack.

**H3. No blit encoder runs between the host write and the kernel on
`host_direct`.** Run: Instruments, Metal System Trace, default invocation.
This is the only direct evidence available. Nothing else substitutes.

**H4. A global integer atomic against host-visible memory is coherent.** Run:
the `um.out_host_direct` scope of the default invocation. Falsified by `wrong`,
which closes the histogram-output question outright and is the cheapest
valuable result in the whole experiment.

**H5. First touch and the CPU-writes-after-GPU-reads transition are cheap
relative to the steady state.** Run: `MOJOBOOST_UM_MODE=resident`, reading
`round0_over_steady` and `retouch_over_steady`. A large `retouch_over_steady`
would mean gradients pay a migration cost every round on a shared route, which
would matter more than the copy it saves.

**H6. A route win survives the trainer.** Run: `bench/bench_train_gpu.mojo`
with `MOJOBOOST_GPU_TRANSFER` set and `MOJOBOOST_GPU_TRANSFER_UNPROVEN=1`,
against the same benchmark on the default. This is the only hypothesis whose
answer may change a default. Note the standing context: the one end-to-end GPU
training measurement in this repository (M4) is slower than the CPU trainer and
is dominated by per-node launches and full-dataset scans, so H6 can easily be
false while H1 through H5 are all true.

**H7. Holding a resident validation matrix moves the practical ceiling.** Run:
the ladder with and without `MOJOBOOST_UM_HOLD_MIB`. Answers what a fit that
also scores a held-out set can size, which is a support question rather than a
performance one.

## Profiler and capture evidence required

| Evidence | Tool | Answers |
|---|---|---|
| peak resident set size | `/usr/bin/time -l` | whether two payload-sized allocations exist |
| compressor and swap deltas | `vm_stat` before and after | whether the run is void |
| dirty / compressed / swapped page split | `footprint -p <pid>` during | where the memory actually went |
| blit encoder presence between write and kernel | Instruments, Metal System Trace | Claim 2, directly |
| kernel duration versus host wait | Metal System Trace | whether `sync_ns` is device work or scheduling |
| end-to-end fit time and model equality | `bench/bench_train_gpu.mojo` | H6 |

A run without the first two is a strictly smaller answer than the one asked
for. A claim about duplication without the fourth is not supported.

## What each result licenses

The full table is in `docs/APPLE_UNIFIED_MEMORY.md`, "Reading the results". The
three rows most likely to be over-read, restated here so a reviewer sees them
without opening the document:

- `copy_bytes_issued_total: 0` with a passing checksum licenses "the device
  read the correct bytes with no copy issued by mojoboost" and **nothing about
  duplication**. The runtime may have migrated pages or blitted behind the
  enqueue.
- Any route beating `copy_staged` in the driver licenses "that route was faster
  in this driver, on this payload, on this machine, at this drain count" and
  **nothing about training**. The driver's consumer is a synthetic
  bandwidth-bound kernel over a four-byte accumulator.
- A `resident`-mode run with a flat steady state licenses "this route pays
  nothing measurable per launch for memory the device already holds" and **not**
  "there is no migration", which needs the trace, and **not** a statement about
  gradients, which are the `rewrite` mode's question.

## Required central changes

None from this lane right now. All three files are new or self-contained, the
policy module is imported by nothing in `src/`, and the driver is a benchmark.
What a future integration would need, by owner:

**Task 07 (`histogram_gpu.mojo`, `train_gpu.mojo`, `gpu_runtime.mojo`,
`gpu_predict.mojo`, `gpu_active_rows.mojo`)**

1. Resolve the route once per session in `GpuSession` and hand the
   `RouteDecision` to the builder. Do not resolve per buffer.
2. Introduce the `Payload` wrapper and route the five hardcoded sites in
   `GpuHistogramBuilder` through it.
3. Move `note_device_read(RES_STAGE)` from publish time to launch time when
   `SyncContract.note_read_on_each_launch` is set, and stop trusting
   `StagingRing` when `staging_ring_applies` is False.
4. Make the drain in `GpuSession.close` unconditional rather than conditional on
   `any_pending()` before any shared route ships, and make a `POOL_GROW` free
   wait on kernel retirement rather than copy retirement.
5. The batch-scoring staging copy in `gpu_predict.mojo` is the one place a
   shared route would pay off structurally rather than only possibly. Do
   **not** simply delete the copy and upload from `data.bins`: `bins_dev` is
   high-water sized and `enqueue_copy` moves the whole destination, so that
   change reads past the end of the caller's list. The options are a sub-range
   copy if the API supports one (unverified), an exact-size reallocation per
   batch, or `ROLE_BATCH_BINS` on a shared route once the evidence exists.

**Task 13 (`binning.mojo`)**

6. If the binned matrix is ever to avoid duplication, `fit_bins` /
   `bin_equal_width` must be able to write into a device-visible allocation
   rather than a heap `List[UInt8]`. This is a data-ownership change, it is the
   only way finding 2 is addressed, and it should not be attempted before H1
   through H3 come back positive, because it is invasive and worthless if they
   do not.

**Whoever owns `bench/apple/schema.json` and `bench/apple/suite.py`**

7. Extend the Apple benchmark schema for the additive grammar changes:
   `um.mode`, `um.hold_bytes`, the `um.policy.*` scope, the
   `um.out_host_direct` and `um.out_wrapped_host_buffer` scopes, the
   `um.policy.role.<role>.host_direct` lines, and the
   per-route keys `input_route`, `out_route`, `mode`,
   `copy_bytes_issued_total`, `enqueues_copy`, `drains_per_round`, and the five
   `retouch_*` keys. Every previously emitted key is unchanged in name and
   meaning, so an existing parser still works and simply sees fewer keys than
   exist. `docs/APPLE_UNIFIED_MEMORY.md`, "Output grammar", is the source of
   truth.

**`pixi.toml` (shared hotspot, not edited here)**

8. A `bench-um = "mojo run -I src bench/apple/unified_memory.mojo"` task would
   be convenient and is not required; the invocation in the document works as
   is.
9. The future policy test named below has to be appended to the `test` task's
   chain, which is the only place tests are enumerated.

**Task 20 (`device_policy.mojo`, `apple_gpu_policy.mojo`)**

10. Boundary, so the two do not overlap: Task 20 owns *which device runs*, and
    the memory *estimates* that feed that choice. This lane owns *how a buffer
    reaches the device that was chosen*. The only value that crosses is
    `unified_memory`, which this module takes as a plain `Bool` argument
    precisely so it never has to import a profile type or open a device. If
    Task 20's profile grows a memory budget, `role_footprint` here is the
    function that should feed it, not a second estimate.

## Coordination with sibling lanes

- **Task 07** owns every file the seam lands in. Nothing here was written into
  those files; items 1 through 5 above are the whole ask, and every one of them
  is gated on evidence that does not exist yet. Item 5 is the exception worth
  doing now.
- **Task 13** owns `binning.mojo` and therefore owns finding 2's only real fix.
- **Task 14** owns histogram specialization. This driver's `BLOCK_THREADS = 256`
  and `MAX_BLOCKS = 4096` are deliberately fixed and untuned so the kernel
  cancels out of every route comparison. They are not a geometry proposal and
  Task 14 should ignore them.
- **Task 17** owns the thermal and energy protocol. Its capture and this
  driver's protocol both bracket a run with external tools on an idle machine
  and both are invalidated by a busy one; a combined capture would be
  convenient and neither lane should assume the other's brackets.
- **Task 20** owns device policy. See item 10.

## Focused test that should exist later

None was written: this round forbids new tests, and this lane compiled nothing.
The one that should exist is `tests/parallel/test_unified_memory_policy.mojo`,
covering, all of which are pure host arithmetic and need no accelerator:

- `resolve_from_env` returns `copy_staged` for every role with the environment
  unset, which is the shipped-default assertion;
- an explicit request for every non-default route raises with the installed
  (empty) ledger, and names the missing evidence rung;
- the same request succeeds under `ack_unproven` and the returned decision has
  `ack_unproven` set;
- `structural_support` refuses `host_direct` for `ROLE_BINS` and
  `ROLE_VALID_BINS`, and refuses every shared route for `ROLE_HIST_OUT`,
  regardless of evidence;
- `structural_support` *allows* `host_direct` for `ROLE_BATCH_BINS`, which is
  the asymmetry with `ROLE_BINS` most likely to look like a bug to a reader
  who has not read why;
- a copy-skipping route is refused when `unified_memory` is False even with a
  full ledger;
- `RouteEvidence.level` truncates at the first missing rung (a route with
  `no_blit_in_trace` but no `checksum_ok` scores `none`);
- `EvidenceLedger.audit` flags a rung set with no record identifier;
- `sync_contract` reports `RETIRE_ON_KERNEL` and `staging_ring_applies ==
  False` for every route except the two copy ones, `map_write` included. That
  last inclusion is the assertion worth having: `map_write` publishes by copy
  and still retires on the kernel, because the buffer it writes next is the
  one the kernels read;
- `publishes_by_copy(ROUTE_MAP_WRITE)` is True while the driver's
  `enqueues_copy` key is `0` for the same route. That disagreement is
  deliberate (the block exit is not an `enqueue_copy` call but may still be a
  full upload) and it is the single most likely thing for a later reader to
  "fix" into agreement, so it gets an assertion and a comment rather than a
  shared helper that would have to pick one meaning;
- `explain_route` returns the default with the same reason `resolve_route`
  raises on, for every (role, route) pair, which is the assertion that keeps
  the reporting path and the acting path from drifting apart;
- `role_footprint` counts host and device bytes separately and never sums them
  into a saving;
- `training_bins_duplication().total() == 2`,
  `validation_bins_duplication().total() == 2`, and
  `batch_scoring_bins_duplication().total() == 3`.

## Exact commands for the later validation pass

In order, and never on a busy machine.

```sh
# 1. Compile the policy module (host arithmetic only, no device needed).
mojo run -I src tests/parallel/test_unified_memory_policy.mojo   # once written

# 2. Compile the driver. This is the first time it will ever be compiled and
#    it is expected to need fixes; see Risks.
mojo build -I src bench/apple/unified_memory.mojo

# 3. The measurement, rewrite mode, with the mandatory external capture.
vm_stat > /tmp/um_vmstat_before.txt
/usr/bin/time -l mojo run -I src bench/apple/unified_memory.mojo 256 8 \
    > /tmp/um_rewrite_256.txt 2> /tmp/um_time_256.txt
vm_stat > /tmp/um_vmstat_after.txt

# 4. Repeat step 3 twice more, then at 1024 and 4096 MiB.

# 5. Resident mode, same brackets.
MOJOBOOST_UM_MODE=resident mojo run -I src bench/apple/unified_memory.mojo 256 8

# 6. Contention, as a whole separate set.
MOJOBOOST_UM_CONTEND=1 mojo run -I src bench/apple/unified_memory.mojo 256 8

# 7. Resident validation matrix held throughout.
MOJOBOOST_UM_HOLD_MIB=512 mojo run -I src bench/apple/unified_memory.mojo 256 8

# 8. Ladder, alone, last, with vm_stat bracketing.
MOJOBOOST_UM_LADDER=1 mojo run -I src bench/apple/unified_memory.mojo 256 8

# 9. Only if 3 through 8 come back positive: the trainer, which is the only
#    run that may change a default.
mojo run -I src bench/bench_train_gpu.mojo
MOJOBOOST_GPU_TRANSFER=host_direct MOJOBOOST_GPU_TRANSFER_UNPROVEN=1 \
    mojo run -I src bench/bench_train_gpu.mojo
```

Step 9 does nothing until Task 07 wires the seam; it is listed so the order is
unambiguous.

## Risks

1. **The driver is still uncompiled, and it now has more surface.** This is the
   largest risk and it is inherent to the round's instructions. Every API it
   uses is one `histogram_gpu.mojo` already uses in the same form, which is why
   it is built from that vocabulary and nothing else. The places most likely to
   need a fix on first compile:
   - passing `HostBuffer.unsafe_ptr()` as a kernel argument, in both the
     payload position (`host_direct`) and now the accumulator position
     (`out_host_direct`); the accumulator position is the newer of the two and
     has no precedent anywhere in the repository;
   - the new import from `mojoboost.unified_memory_policy`, which couples the
     benchmark to a module that has also never been compiled. If the policy
     module fails to compile, the driver fails with it. The coupling is
     deliberate (one route vocabulary, not two) and this is its cost;
   - `RunPlan` and `FirstTouch` as `Copyable, Movable` structs passed by value
     into runners;
   - `Consumer` holding both a `DeviceBuffer[DType.int32]` and a
     `HostBuffer[DType.int32]` and branching between them at launch time.
2. **The held buffer must actually stay alive.** Mojo destroys a value at its
   last use, so the driver ends with a real use of the held buffer rather than
   a discard. If that use is ever removed as dead code, the hold silently stops
   holding and every number after the first route is measured in a different
   memory state.
3. **`resident` mode's staleness check is weaker than `rewrite` mode's, by
   construction.** Before the retouch round it only proves the route published
   once. The document says so, the module docstring says so, and the protocol
   says run `rewrite` first. A `resident`-only result should not be accepted.
4. **`copy_bytes_issued_total: 0` will be misread as zero copy.** It is
   reported next to `enqueues_copy` and explained in three places for that
   reason. It is the single most likely misreading of this whole experiment.
5. **A route win would not imply a training win.** The current M4 end-to-end GPU
   measurement is slower than the CPU trainer and is dominated by per-node
   launches and full-dataset scans, not transfers. `ENABLE_LEVEL` is the
   trainer rung specifically so a driver result cannot promote itself.
6. **The unproven override is a loaded gun.** It has to exist, because the top
   evidence rung cannot be climbed without running the trainer on the route.
   Every decision taken under it carries `ack_unproven`, and any benchmark
   result produced under it must report the flag beside the number. A result
   that does not is not a result.
