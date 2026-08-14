# A7: Unified-memory experiment design

## What this lane produced

Two files, both new.

- `bench/apple/unified_memory.mojo` -- an unexecuted experiment driver that
  runs one fixed checksum kernel over one payload through every host-to-device
  route this Mojo version is known to provide, and reports allocation, first
  touch, publish, kernel, synchronization, readback, repeated-round steady
  state, per-route allocated bytes, CPU/GPU contention, and a size ladder, with
  a correctness gate on every route.
- `docs/APPLE_UNIFIED_MEMORY.md` -- the methodology, the mandatory external
  measurement protocol, the output grammar, and an explicit table of which
  claims each possible outcome licenses.

## What this lane did NOT do

Stated first because it is the part most likely to be misread later.

- **The driver has never been compiled.** This is a documentation and design
  lane under the round's parallelism contract, which forbids compiling. No
  `mojo build`, no `mojo run`, no test.
- **The driver has never been executed.** There are no results, no timings,
  and no numbers anywhere in either file.
- **No zero-copy claim is made anywhere.** `docs/APPLE_UNIFIED_MEMORY.md`
  separates "Apple has unified physical memory" (true, free, implies nothing)
  from "our buffer is the bytes the kernel reads" (unproven), and every status
  row in that document reads **none**.
- **No GPU module, kernel, or central file was touched.** `histogram_gpu.mojo`,
  `train_gpu.mojo`, `gpu_tiling.mojo`, `device.mojo`, `pixi.toml`,
  `bench/README.md`, and `docs/GPU_VALIDATION.md` are all unmodified.
- **Route 5 (non-owning `DeviceBuffer` over a host pointer) is not
  implemented.** It is reported as `not_probed`, with the candidate snippet and
  a three-outcome promotion procedure in the doc. An unverified route is not
  reported as an available one.

## The one structural finding that does not need a run

The kernels already take `MutPointer[T, MutAnyOrigin]`, not buffer types. So a
shared or mapped route needs **no kernel signature change at all** -- only a
change in which pointer gets passed at the launch site. That is why the driver
can compare five routes under one kernel, and it is why an eventual integration
is a change to `GpuHistogramBuilder`'s buffer ownership and nothing deeper.

The second structural point, also independent of any run: in Mojo a missing
method is a compile error, not a catchable one. `try`/`except` can report "this
API exists and refused"; it cannot report "this API does not exist". The driver
therefore has two distinct negative statuses, `unsupported` (raised at runtime)
and `not_probed` (never compiled), and the doc's promotion procedure treats a
compile failure as a recordable result rather than a blocker.

## Two methodology decisions a reviewer should check

Neither is obvious from reading the driver quickly, and both are places where
the easy version of the code produces a number that means something other than
what it is labelled.

1. **`copy_direct` performs a full-size host write, it does not skip one.**
   The tempting version copies straight from the caller's `List` and does no
   staging write at all. That varies two things at once against `copy_staged`
   (pinned versus unpinned staging, *and* write versus no write), so the
   resulting difference cannot be attributed to either. It also would not model
   the trainer, whose `stage_gradients` must write its Float64-to-Float32
   conversion into some buffer regardless. So `copy_direct` stages into a plain
   heap `List` of the same size and the single variable is pinning. The
   round-0 comparison between these two routes is still not clean, because the
   heap `List` is faulted in by its own construction while the pinned buffer's
   residency is the runtime's business; that caveat is stated in both the
   driver and the doc, and the steady-state comparison is the one to read.

2. **Contention time is timed separately from the synchronize, and is counted
   in the round.** The host contention work runs on the same thread that then
   waits, so wrapping both in one span would book host time as device wait and
   manufacture a contention effect out of nothing. `contend_ns` and `sync_ns`
   are therefore disjoint. They are both in the round total, because host work
   on the round's own thread is round time; excluding it would make a contended
   run look cheaper than an uncontended one. The consequence to know before
   reading a contended run is that if the host work outlasts the kernel,
   `sync_ns` collapses toward zero, and that is not evidence the device was
   unaffected. Contention is only answerable by comparing two whole runs.

## Proposed API changes

**All of these are contingent on evidence that does not exist yet.** Nothing
below should be implemented until `docs/APPLE_UNIFIED_MEMORY.md`'s status table
has a route at `ok` *and* the external capture (peak resident set showing one
payload-sized allocation rather than two, plus a Metal System Trace with no
blit encoder between the host write and the kernel) backs it. A passing
checksum on its own is not sufficient and the doc says why.

### 1. A transfer-route seam in `histogram_gpu.mojo`

Today `GpuHistogramBuilder` hardcodes the staged-copy route in five places.

- Constructor, the one-time binned-matrix upload:
  `self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](n_cells)`
  followed by `self.ctx.enqueue_copy(dst_buf=self.bins_dev,
  src_ptr=data.bins.unsafe_ptr())` and `self.ctx.synchronize()`.
- `stage_gradients`, the per-round Float64-to-Float32 conversion into
  `self.stage_g` / `self.stage_h`.
- `upload_staged`, the two `enqueue_copy` calls into `self.grad_dev` /
  `self.hess_dev`.
- `enqueue_leaf`, which passes `self.bins_dev.unsafe_ptr()`,
  `self.grad_dev.unsafe_ptr()`, `self.hess_dev.unsafe_ptr()` to the kernels.
- The field declarations themselves, `bins_dev: DeviceBuffer[DType.uint8]`,
  `grad_dev` / `hess_dev: DeviceBuffer[DType.float32]`, `stage_g` /
  `stage_h: HostBuffer[DType.float32]`.

The minimal seam is a per-buffer wrapper that owns whichever allocation the
resolved route needs and answers one question, "what pointer do I hand the
kernel":

```mojo
# Sketch only. Not written, not compiled.
struct Payload[dtype: DType](Copyable, Movable):
    var dev: DeviceBuffer[Self.dtype]     # staged route only
    var host: HostBuffer[Self.dtype]      # both routes
    var shared: Bool

    def kernel_ptr(self) -> MutPointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.host.unsafe_ptr() if self.shared else self.dev.unsafe_ptr()

    def publish(mut self, mut ctx: DeviceContext) raises:
        if not self.shared:
            ctx.enqueue_copy(dst_buf=self.dev, src_ptr=self.host.unsafe_ptr())
```

With that, `stage_gradients` keeps its Float64-to-Float32 conversion loop
unchanged (it already writes into a `HostBuffer`), `upload_staged` becomes
`self.grad.publish(self.ctx)` plus the same for hessians and is a no-op on the
shared route, and `enqueue_leaf` substitutes `kernel_ptr()` for
`.unsafe_ptr()`. No kernel changes.

Note the invariant that makes the shared route *not* a drop-in: today
`stage_gradients` calls `self.ctx.synchronize()` before overwriting the staging
buffers, because a queued copy may still be reading them. On a shared route the
kernel itself reads that memory, so the synchronization is not removable and
may need to be stronger, not weaker. Whoever implements this must redo that
dependency analysis rather than deleting the synchronize because a benchmark
got faster. Coordinate with lane A5 (`gpu_runtime.mojo`), which is separately
proposing to remove synchronizations.

### 2. Route selection lives with the runtime, not the histogram builder

If lane A5's persistent `gpu_runtime.mojo` session lands, the route belongs
there with the buffer pool, resolved once per session and handed to
`GpuHistogramBuilder` rather than re-decided per buffer. Two lanes should not
each grow their own device-memory policy. This lane makes no claim on that
file.

### 3. Environment knob, following the existing convention

`MOJOBOOST_GPU_TRANSFER` with values `staged` (default) and `shared`, parsed
next to `MOJOBOOST_GPU_HIST_STRATEGY` in `gpu_tiling.mojo`'s `env_strategy`
neighborhood, defaulting to `staged` until the evidence exists. The precedent
is `MOJOBOOST_AUTO_MIN_CELLS` in `device.mojo`, which is deliberately shipped
disabled because no measurement justifies a default. The same reasoning applies
here and for the same reason.

### 4. Nothing changes in the model, ABI, serialization, or Python API

No parameter, no field, no C ABI entry point, no serialized byte. The routes
produce identical results by construction, which the driver's checksum gate
enforces; a route that changed results would be reported `wrong`, not adopted.

## Central integration required from this lane, right now

**None.** Both files are new and self-contained, and neither is imported by
anything. Two optional wiring items belong to other owners.

- `pixi.toml` (shared hotspot, not edited): a task would read
  `bench-um = "mojo run -I src bench/apple/unified_memory.mojo"`. Not required;
  the `mojo run` line in the doc works as is.
- `bench/README.md` (not assigned to this lane): if benchmarks get indexed
  there, this driver should be listed with the words "unexecuted, no results"
  next to it.

## Coordination with sibling lanes

- **A8 (`bench/apple/suite.py`, `bench/apple/schema.json`)** shares the
  `bench/apple/` directory. This lane created only `unified_memory.mojo` there
  and touched neither of A8's files. The `um.<scope>.<key>: <value>` output
  grammar is specified in `docs/APPLE_UNIFIED_MEMORY.md` under "Output grammar"
  and is what A8's schema should consume; status values are exactly `ok`,
  `unsupported`, `wrong`, `not_probed`, and a `comparable: 0` flag marks
  results that must not be compared.
- **A5 (`gpu_runtime.mojo`)** owns synchronization removal and buffer pooling.
  See the invariant note in proposal 1.
- **A6 (`apple_gpu_policy.mojo`)** owns launch geometry. This driver's
  `BLOCK_THREADS = 256` and `MAX_BLOCKS = 4096` are deliberately fixed and
  untuned so the kernel cancels out of every route comparison. They are not a
  geometry proposal and A6 should ignore them.

## Focused test

None run, and none exists. This is a documentation and design lane; the round's
contract lists A7 among the lanes that never compile, and the task explicitly
says this lane writes the experiment but does not execute a potentially heavy
benchmark. `git diff --check` was run over the assigned paths and is clean.

## Risks

1. **The driver is uncompiled, so it may not build.** This is the largest risk
   and it is inherent to the lane's instructions, not an oversight. Every API
   the driver uses is one `histogram_gpu.mojo` already uses in the same form
   (`enqueue_create_buffer`, `enqueue_create_host_buffer`, `enqueue_copy` in
   both directions, `enqueue_memset`, `enqueue_function`, `map_to_host`,
   `synchronize`, `get_attribute`, `unsafe_load` / `unsafe_store`,
   `Atomic.fetch_add`, `MutPointer[T, MutAnyOrigin]` kernel arguments), which
   is why the driver is built from that vocabulary and nothing else. The
   specific places most likely to need a fix on first compile:
   - passing `HostBuffer.unsafe_ptr()` as a kernel argument in the
     `host_direct` route, which is the one pointer flow with no precedent in
     the repository;
   - `Consumer` holding `DeviceBuffer[DType.int32]` and
     `HostBuffer[DType.int32]` as struct fields with `Copyable, Movable`,
     mirroring `GpuHistogramBuilder`'s fields but in a smaller struct;
   - `mut ctx: DeviceContext` parameters, chosen over `imm` because
     `histogram_gpu.mojo` only ever calls these methods through `mut self`.
2. **No measurement of resident memory, compression, or swap happens in
   process.** The driver has no honest way to read them, so the doc makes the
   external `/usr/bin/time -l` and `vm_stat` capture mandatory and the driver
   prints begin and end markers to line up against. A run without that capture
   answers a strictly smaller question than the one asked.
3. **`host_direct` could pass its checksum and still not be zero copy.** A
   correct checksum shows the device read the right bytes, not that it read
   them in place. The doc's reading table makes this the central caveat.
4. **The ladder mode can push a machine into swap.** It is off by default,
   documented as the memory-pressure mode, and its answer is explicitly a
   property of one machine in one memory state, not of the hardware.
5. **A route win here would not imply a training win.** The current M4
   end-to-end GPU measurement is slower than the CPU trainer and is dominated
   by per-node launches and full-dataset scans, not transfers. The doc says so
   in the reading table so nobody promotes a transfer result into a training
   claim.
