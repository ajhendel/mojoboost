# GPU portability contract

Written: 2026-08-14. Section 6 added 2026-08-15.

mojotrees has one GPU source. `docs/GPU_VALIDATION.md` states that
commitment and holds the record of what has actually run on hardware. This
document is the other half: **what the one source requires of a backend**,
which of those requirements every supported API specifies, which assumptions
in the tree are Apple-shaped rather than portable, and where a backend may
specialize without forking the source.

Nothing here is a claim about NVIDIA or AMD behavior. Only Metal has
executed a kernel from this repository, on one Apple M4. Everything below
about CUDA and HIP is a statement about what those APIs specify, or about
what the code in this tree does, not about anything anyone observed.

The Mojo modules that carry this contract are:

| Module | Holds |
|---|---|
| `src/mojotrees/gpu_backend_policy.mojo` | Which backend is in front of us, and how far this repository has taken it |
| `src/mojotrees/gpu_portability.mojo` | The primitive contract, the launch gate, and the specialization gate |
| `src/mojotrees/gpu_tiling.mojo` | The launch geometry itself, and the three device attributes it is derived from |
| `src/mojotrees/gpu_histogram_specializations.mojo` | The specialization primitives and the real threadgroup footprints |
| `src/mojotrees/device_policy.mojo` | Whether training runs on a device at all |

## 1. The primitive contract

The shared kernels reach for a small, fixed set of device primitives. This
set was read off the imports of every `src/mojotrees/gpu_*.mojo` and
`histogram_gpu.mojo`, not assumed. There is nothing else: no vendor
intrinsic, no subgroup operation, no cooperative group, no dynamic
threadgroup memory, no device-side allocation, no float atomic.

| Requirement | Spelled | Used by | Metal | CUDA | HIP |
|---|---|---|---|---|---|
| `threadgroup-barrier` | `max.gpu.sync.barrier()` | every histogram, scan, and partition kernel | specified | specified | specified |
| `static-threadgroup-allocation` | `stack_allocation[..., AddressSpace.SHARED]` | partial histograms, prefix-sum scratch | specified | specified | specified |
| `threadgroup-int32-atomic-add` | `Atomic.fetch_add` on a shared `Int32` | per-block partial histogram | specified | specified | specified |
| `device-int32-atomic-add` | `Atomic.fetch_add` on a device `Int32` | `STRATEGY_ATOMIC` flush only | specified | specified | specified |
| `two-dimensional-grid` | `grid_dim=(x, y)` | features or leaf slots on `x`, row tiles or classes on `y` | specified | specified | specified |
| `in-order-queue` | `DeviceContext` enqueue order | the hazard model in `gpu_runtime.mojo` | specified | specified | specified |

"specified" means the API documents the primitive. It does not mean this
project ran it there. The one column that has been exercised is Metal, and
`gpu_backend_policy.backend_support` is where that difference is recorded so
a reader cannot confuse the two.

`gpu_portability.required_primitives(strategy)` returns this list per
resolved histogram strategy. The two strategies differ by exactly one entry:
`STRATEGY_TILED` writes each partial to a slot nothing else writes and
reduces in a second kernel, so it needs no device-memory atomic;
`STRATEGY_ATOMIC` folds its partials into the output and does.

## 2. Backend support levels

`gpu_backend_policy.mojo` carries three levels, and the distinction between
the last two is the whole point of the module.

| Level | Meaning | Backends |
|---|---|---|
| `unsupported` | No contract exists. An API code outside the covered set, which is a caller bug and is refused | none in normal operation |
| `portable-untested` | The shared source targets it and the contract covers it. This repository has never executed a kernel on it | CUDA, HIP, and a device that did not name its API |
| `exercised` | Kernels from this repository have run on it and the run is recorded in `docs/GPU_VALIDATION.md` | Metal, on one Apple M4 |

`portable-untested` does not block training. Blocking it would refuse the
design commitment: one source, compiled for whatever device MAX opens. What
it blocks is **specialization**, through
`gpu_portability.require_specializations_allowed`, because a kernel variant
chosen for an architecture nobody has run on rests on a measurement that
does not exist.

An unidentified device is `portable-untested` rather than `unsupported` on
purpose. Nothing in the shipping code reads an API name before it launches,
so `API_UNKNOWN` is the value every launch carries today, and every bound in
its contract is the tightest of the three backends.

## 3. Apple-shaped assumptions in the shared code

Found by inspection, listed whether or not they are defects. Several are
deliberate and correct; naming them is what keeps them deliberate.

### 3.1 Unified memory is inferred from the API name

`apple_gpu_policy.mojo:327` sets `GpuProfile.unified_memory` to
`api == API_METAL`. That is right for Metal and wrong as a general rule: an
integrated AMD part, an NVIDIA Jetson, CUDA managed memory, and an
XNACK-enabled HIP device are all unified in ways an API name cannot
distinguish from a discrete card. Because `unified_memory` feeds
`apple_gpu_policy.partial_budget_bytes` (which divides a reported memory
budget by 32 instead of 16) and `unified_memory_policy.plan_session_routes`,
a unified non-Apple device currently gets the discrete partial-buffer
fraction.

The consequence today is bounded: the reported memory budget is 0 in every
profile this tree constructs, so `partial_budget_bytes` returns the portable
ceiling either way. `gpu_portability.contract_from_profile` takes
`unified_memory` from the profile rather than from the API, so the value has
one place to be corrected. The correction itself belongs to the module that
builds profiles, and is filed as a patch request in
`handoffs/connect_20_gpu_portability.md`.

### 3.2 Float32 gradients and no Float64 on device

`histogram_gpu.mojo:58` and `gpu_predict.mojo:37` both record it: Apple GPUs
have no Float64, so leaf values, base scores, gradients, and hessians are
Float32 on the device and histogram accumulation is fixed-point Int32.

This is Apple's floor imposed on every backend, and it is the right call for
one source, but it is a real cost on CUDA and HIP, which have Float64. It is
recorded as `BackendContract.device_float64_permitted = False` on every
backend rather than left as a comment, and
`gpu_portability.require_device_float64` is the one place a future Float64
variant would be gated from. Such a variant would have to reproduce the
fixed-point integers exactly, because those integers are what make a
histogram bit-identical run to run.

### 3.3 The specialization planner is Apple-named

`apple_histogram_policy.mojo` is the only module that turns device facts and
a node's shape into a specialized launch plan, and until now the only
constructor anything called for its capability input was
`DeviceHistogramCapabilities.portable()`, which answers "nothing reported"
for every device. So the specialization layer was reachable in practice only
through an Apple-named module, and every non-Apple device planned as if it
had reported nothing.

`gpu_portability.histogram_capabilities(contract)` is the connection: it
builds that same existing record from a backend contract, so a CUDA or AMD
device reaches the existing planner with its own capabilities instead of the
portable blank. Nothing in the planner is Apple-specific; the file name is.
Renaming it is filed as a patch request.

### 3.4 Subgroup width is never known and never assumed

Metal rejects the `WARP_SIZE` attribute query, and this repository has never
run the query on CUDA or HIP. `BackendContract.subgroup_width` is therefore
0, meaning unknown, on every backend in the table, and nothing in the
package divides by it.

`WARP_GRANULARITY = 64` in `gpu_tiling.mojo` is a different kind of value
and must not be read as a width claim. It is the multiple a threadgroup
width is rounded to, chosen because 64 is AMD's CDNA wavefront and a
multiple of the 32-wide warp everywhere else, so one constant is launchable
on all three backends. `BackendContract.launch_granularity` carries it under
a name that says so.

### 3.5 Three device attributes, and the shared-memory guard is optimistic

`gpu_tiling.query_device_caps` reads exactly three attributes:
`MULTIPROCESSOR_COUNT`, `MAX_THREADS_PER_BLOCK`, and
`MAX_SHARED_MEMORY_PER_BLOCK`, each falling back to a conservative constant
when a backend does not implement the query. That is the entire capability
surface every shipping GPU module sees.

`gpu_tiling.shared_bytes_for(n_bins)` returns
`histogram_bin_capacity(n_bins) * 12`, the footprint of one feature slot in a
kernel sized to its bin capacity. The kernels that ship are now sized that
way: they take the capacity as a comptime parameter and allocate three
`GROUP * BIN_CAP`-wide Int32 planes, where `BIN_CAP` is `n_bins` rounded up
the ladder 32, 64, 128, 256. A 64-bin dataset therefore allocates 768 bytes
per slot rather than the 3072 it allocated at every bin count before, and the
model and the launch agree at every bin count rather than only at 256.

One caveat, and it is the reason `KernelFeatures.specialized_bin_kernels` is
still false: this function and `kernel_shared_request` below both price ONE
feature slot, and a threadgroup owning a group of G occupies G times as much.
Neither knows the group. The bound that actually refuses a width the device
cannot hold is `GpuActiveRows.set_feature_group`, which does know it. Giving
these two the group is the change that would let the geometry gate check a
number that bounds the launch.

`gpu_portability.kernel_shared_request(n_bins, features)` returns the
footprint the compiled kernel really has, choosing between the two according
to whether the build has bin-capacity specialized kernels, and
`require_launch_geometry` checks that against the reported device.
`require_device_can_host_kernels` asks the same question once at session
open, against `MIN_SHARED_MEMORY_PER_BLOCK`.

No supported backend advertises less than 3 KiB. The checks exist so that
stays a fact rather than an assumption on hardware nobody here has opened.

### 3.6 Duplicate portable constants

`MAX_GRID_DIM_Y = 65535` is CUDA's `grid.y` cap and the bound the whole
package is written to, since Metal and HIP allow more and one source targets
all three. It is defined in `gpu_tiling.mojo:115` and mirrored in
`apple_gpu_policy.mojo:96`. `gpu_leaf_batching.mojo` used to carry a third
copy and now imports it. The remaining mirror is filed as a patch request.

`gpu_portability.mojo` imports the `gpu_tiling.mojo` definition, so the
portability contract and the geometry it gates cannot drift into disagreeing
about the bound.

### 3.7 One unchecked `grid.y`

`gpu_multiclass_batch.mojo:1041` launches `grid_dim=(row_blocks, k_count)`,
putting the class count on the `grid.y` axis rather than a tile count.
`derive_tiling` clamps its tile count to `MAX_GRID_DIM_Y`; nothing clamps
`k_count`. No realistic class count approaches 65535, so this is a missing
check rather than a live defect, and
`gpu_portability.require_launch_geometry` is the check to add. Filed as a
patch request.

## 4. Launch geometry contract

| Axis | Carries | Bound checked against |
|---|---|---|
| `grid.x` | features, active features under subsampling, or leaf slots | positive only. No portable upper bound is established here, and CUDA's is over two billion |
| `grid.y` | row tiles, leaf-batch tiles, or classes | `BackendContract.max_grid_dim_y`, which is 65535 |
| `grid.z` | nothing | not used. No portable bound for it has ever been established in this repository, and an earlier batched draft that wanted one was removed rather than given a guess |
| threadgroup width | lanes per block | the device's reported `MAX_THREADS_PER_BLOCK`. Rounding to `launch_granularity` is `gpu_tiling.clamp_block_threads`' job, not the gate's |
| threadgroup memory | the partial histogram | the device's reported `MAX_SHARED_MEMORY_PER_BLOCK`, against the real compiled footprint |

## 5. Synchronization and allocation contract

**Queue ordering.** A `DeviceContext` queue is in order, so device work never
needs a host synchronization to observe earlier device work. `HazardTracker`
in `gpu_runtime.mojo` is the model built on that, and it says a host
synchronization is required in exactly two cases: the host is about to read
memory the device has an unretired write to, or the host is about to write
memory the device has an unretired read or write to. That model is
backend-independent by construction, because it rests on queue ordering
rather than on any device's memory model.

**Allocation.** Every buffer is an explicit copy today, on every platform, in
every build. `unified_memory_policy.mojo` is the layer that could return
anything else and cannot currently do so without both an environment request
and an explicit acknowledgment that the request is unproven. A backend with
a unified address space does not change that default, and a route that skips
a copy changes when the host may next write the buffer, which is a
synchronization obligation and is returned with every decision there.

**Numerics.** Gradients and hessians cross to the device as Float32.
Accumulation is fixed-point Int32 with a scale chosen on the host, so
histogram addition is exact and associative and the result is bit-identical
run to run and strategy to strategy. Counts are exact. Agreement with the
CPU builder is to Float32 precision, not bit-exact. This is the property
most likely to differ first on a backend whose atomics or scheduling differ,
which is why `docs/GPU_VALIDATION.md` gates every timing behind a
determinism check.

**Queue ordering is not the whole synchronization contract on Metal.**
Section 6 records two things Metal provides differently from what the code
above was written against. Both were established this week and neither had
been written down anywhere in this repository.

## 6. Two Metal facts established by disassembly

Everything else in this document is either what an API specifies or what the
code in this tree does. This section is a third kind of thing: what one
backend's shipped implementation actually does, read out of the binary. The
provenance vocabulary is the one the rest of the project now uses, stated
each time, because the unusual part here is that both facts are **measured**
without a clock. A disassembly is a measurement of the code that runs. It is
exact for the build it was taken from, it says nothing about the next build,
and it is not a timing.

Neither fact is a defect in MAX and neither is a reason to stop using it.
Both change what a design is allowed to assume, which is the only reason
they are in this document rather than in a benchmark report.

### 6.1 `enqueue_copy` is a synchronous full-queue drain in both directions

**Read 6.1.1 first. Exactly one half of this section was withdrawn on
2026-08-16 and the other half was not.** What stands: the mechanism below, the
disassembly that established it, and every ordering consequence that follows
from it, including the staging-buffer lifetime arguments and the hazard
reasoning. A copy really does drain the queue, and that is still load-bearing.
What was withdrawn: a **cost** model built on top of the mechanism during the
2026-08-15 round, which said that because a copy drains, every copy is a host
wait worth the ~458 microsecond per-synchronization constant, so removing a
copy saves that much. That inference was refuted by measurement. Nothing in
"What it is", "How it was established", or "What this is not" is retracted.

**What it is.** On Metal, `DeviceContext.enqueue_copy` blocks the calling
thread until the whole queue has drained, and then copies. Host to device as
well as device to host. The name says enqueue and the code waits.

**How it was established. Measured**, by disassembling the shipped MAX Metal
runtime (`libMGPRT.dylib`). `enqueueCopyToDevice` and `enqueueCopyFromDevice`
have the identical body, and it is not an enqueue:

```
[queue commandBuffer]      ; an empty command buffer, no encoder
[cmdbuf commit]
[cmdbuf waitUntilCompleted]
memcpy                     ; host memcpy against the shared MTLBuffer
```

`enqueueSetMemory` and `synchronize` have the same shape, so
`enqueue_memset` is a drain too. Those empty command buffers are visible
from outside the process as well: `docs/METAL_TIMELINE.md:174` counted 3,225
command buffers carrying no work beside the 22,107 carrying work, which it
recorded without explaining. That is the same fact **measured** a second
way, by an instrument that never saw the binary.

**What this is not.** It is not a claim about CUDA or HIP. Those APIs
specify an asynchronous copy family, and nothing in this repository has
established what MAX does on top of them. Their behavior here is
**unestablished**, which is different from asynchronous, and a design that
needs an asynchronous upload has to establish it before relying on it.

#### 6.1.1 The cost model derived from this was refuted. The mechanism was not.

Withdrawn 2026-08-16, on the data in
`bench/results/session3_2026-08-16/RESULTS.md`. Both figures below are
**measured**: medians of five in-process repeats, alternating processes, quiet
machine, 1,000,000 x 50, under the decision rules registered in
`bench/results/PROFILE_PROTOCOL.md` before any of it ran.

| what was removed | predicted | **measured** | verdict |
|---|---|---|---|
| thirteen copies per tree, ~1,300 per fit, from the device-resident plane | 0.64 s | **0.016 s** | not resolved under M0; the gap sits inside the arm's own spread |
| ~thirty round trips per tree, the resident plane against the shipping loop | none registered | **0.75 s** | resolved by a wide margin, same shape, same session |

The 0.75 is the fast regime, where it is 24% of the fit; in a slow window the
same A/B fell to 0.35, or 8%, so the effect size moves with machine state and
only its direction is regime-independent. The 0.016 is a null under M0 either
way.

Same session, same machine, same instrument. The two rows differ by a factor of
about fifty, and what separates them is not how many copies were removed. It is
whether anything was waiting.

**Draining a queue that holds nothing costs nothing.** What costs is the
**round trip**: host code blocking on a device answer that it needs before it
can decide what to enqueue next. The ~458 microsecond per-synchronization
constant is **derived**, from the depthwise A/B, and in that A/B what was
removed were per-level round trips. It remains valid for what it was derived
from. It was then extended to copies by analogy, and the extension is what the
table above refutes.

**The rule that replaces it:**

> **Count round trips to predict time. Count copies to predict portability
> risk and to reason about ordering hazards.** They are two different counts
> with two different uses, and the 2026-08-15 round conflated them.

A copy that drains a queue holding only already-finished work is very nearly
free on Metal. A copy or a `synchronize` that waits on unfinished device work
costs whatever that work had left to do. A round trip costs that, plus the
refill of a pipeline it just emptied, plus the host's own decision.

**What this does not license.** It does not make copies free in general, it
does not make the drain go away, and it does not weaken anything in this
section above. A copy still drains, so it still orders device work against host
work, it still ends the usable depth of a launch stream (section 6.2 rule 3),
and a staging buffer's lifetime still has to be argued against the drain rather
than against an in-flight copy. Removing copies is still worth doing on the
grounds it was actually earned on: fewer places for a stale byte to hide, fewer
buffers to keep alive, less to get wrong on a backend where the copy is
asynchronous. Those are correctness and portability arguments. They were always
the good arguments. Only the stopwatch attached to them was wrong.

**What in this repository assumes otherwise.** Docstrings that state the
opposite have been corrected in place and are listed in the commit that adds
this section. What could not be corrected by editing a comment:

- `gpu_runtime.StagingRing` exists to let the host convert round `i+1`'s
  gradients while round `i`'s copy is still in flight. On Metal no copy is
  ever still in flight, so the ring's second slot can never be the thing
  that avoids a wait, and `MOJOTREES_GPU_STAGING_SLOTS` above 1 buys nothing
  on this backend. The ring is correct, cheap, and portable, and it should
  stay; what is wrong is the expected value of turning it up.
- `HazardTracker`'s host-write hazard class (the host is about to overwrite
  a staging buffer whose copy may still be reading it) is vacuous on Metal
  for the same reason. The model is still the right model, because it is
  written against queue ordering rather than against Metal, and it is
  conservative in the safe direction. But an "elided check" it reports
  against a staging buffer is not a synchronization anyone could have
  removed, since the copy that follows drains anyway.
- Every `ctx.synchronize()` that immediately follows an `enqueue_copy` with
  no enqueue in between is redundant on Metal. It is what keeps the code
  correct on a backend where the copy really is asynchronous, so it should
  stay. What must stop is counting it as a second wait: the copy before it
  already drained, so there is nothing left for it to wait on. Under 6.1.1
  this cuts the other way as well. If that copy itself found an empty queue,
  neither of the two cost anything, and the pair is one ordering point worth
  no time at all rather than one wait worth 458 microseconds.

**What a design must do about it.**

1. **Count round trips to predict time; count copies to predict portability
   risk and ordering hazards.** Rewritten 2026-08-16; see 6.1.1 for the
   measurement that forced it. This rule used to read "count every copy as a
   synchronization, the wait is the cost", and that is the sentence the data
   refutes. The two counts still both matter and are still both worth writing
   down. What is no longer permitted is quoting the copy count and a
   per-synchronization constant and calling the product a predicted saving.
   Where a design does predict a time, the prediction is over round trips,
   and it is **estimated** until it has been run interleaved.
2. **Stage per fit wherever the data does not change per round.** The binned
   matrix already does this. The per-node parameter tables, the feature set,
   the allow mask, the monotone vector and the categorical parameters are
   per-tree or per-fit constants under the common configuration and should
   cross once, not per node and not per batch.
3. **Where a stage must change per round, say so and count it.** The
   per-round stages that exist today are the gradient and hessian planes,
   the bag mask on a bagged tree, the GOSS row and scale vectors, the
   per-round fixed-point scale words, and the per-tree node value and step
   tables. Each is one drain, and a drain is an ordering point whether or not
   it is a cost. A round's **hazard** budget is not complete until all of them
   are in it. Its **time** budget is a different list, made only of the ones
   that block on unfinished device work.
4. **A device-resident control plane states two counts, not one, and labels
   which is which.** Its round-trip count is the one that predicts time. Its
   copy count is the one that predicts what breaks on another backend and
   where a stale byte could hide. A plane that makes one round trip per tree
   and also uploads a bag mask per tree is one round trip and two copies, and
   both numbers should be written, each under its own heading.
5. **Do not argue for new pinned staging buffers on Metal grounds.** One
   buffer per destination is fine and costs nothing, but the argument that a
   shared buffer would force a drain between copies is vacuous when every
   copy has already drained. The argument is a portability argument, not a
   Metal one.

### 6.2 The command queue is 64 command buffers deep and MAX never raises it

**What it is.** One Metal command queue per `DeviceContext`, created once,
holding at most 64 command buffers in flight, with no MAX-level knob,
environment variable, or `DeviceContext` parameter that changes it. And one
command buffer per launch: there is no encoder batching anywhere on this
path, so 64 in flight means 64 launches in flight.

**How it was established. Measured**, by reading how MAX creates its queue
in the same shipped runtime. The queue is created with a bare
`[device newCommandQueue]` (`MetalDeviceContext.cpp:397`). The two selectors
that would raise the limit, `newCommandQueueWithMaxCommandBufferCount:` and
`setMaxCommandBufferCount:`, have zero load sites anywhere in the 38.6 MB
binary, and no `MTLCommandQueueDescriptor` is constructed on that path. So
Apple's default of 64 applies by absence, which is a stronger reading than a
constant would have been: there is no code that could set it to anything
else.

The one-buffer-per-launch half is **measured** from the same disassembly.
`enqueueFunctionExecDirect` is a closed sequence: `[queue commandBuffer]`,
`computeCommandEncoder`, `setComputePipelineState:`, the argument loop,
`dispatchThreadgroups:`, `endEncoding`, `commit`. One buffer, one encoder,
one dispatch, committed immediately, per `enqueue_function`, with no seam to
batch at. `docs/METAL_TIMELINE.md:174` observed the same thing from outside
the process: 22,107 command buffers carrying work, one encoder each.

**What happens when it fills is a derived bound, not a measurement.** Apple
documents that `MTLCommandQueue.commandBuffer` blocks the calling thread
when the queue is full rather than returning nil, so MAX's null check never
fires and no error is raised. From that plus the 64, the steady state is
**derived**: 64 launches go in, the host blocks on the 65th until the first
completes, and thereafter the stream runs at one in and one out. That is a
throttled pipeline and not a queue overrun. Nothing is dropped and nothing
fails. No one has measured it on this hardware.

**The stall is invisible to every instrument in this repository.** The block
happens inside `objc_msgSend`, which `phase_profile.mojo`, `PhaseCounters`
in `gpu_runtime.mojo`, and any wall-clock benchmark all count as host enqueue
time with no attribution. In a Metal System Trace it lands in the
"completion signal to next commit" bucket that `docs/METAL_TIMELINE.md`
already lists as uninvestigated host code. This is **derived** from where
the block occurs, not observed.

**What in this repository assumes otherwise.**

- `docs/design/CLEANSHEET_GPU.md:279` reasons that 187 command buffers
  submitted back to back with no `synchronize()` behave like one. Up to 64
  they do.
- `gpu_resident_round.mojo` enqueues about ten launches per split (commit,
  stage the search, three partition, two child histogram, two search, file
  the records) over `num_leaves - 1` splits, plus roughly five per tree.
  At the default 32-leaf budget that is on the order of 315 launches between
  waits, which is nearly five times the depth. **Derived bound**, from the 64
  and from a launch count read off the source; not measured.
- `HybridCosts.launch_nanos`, in the hybrid leaf scheduler deleted on
  2026-08-16 and preserved in
  `bench/results/apple_m4_hybrid_costs_2026-08-15.md`, was documented as the
  fixed cost of enqueuing and running one histogram kernel. Under
  backpressure the enqueue half is not fixed: it is zero while the queue has
  room and the completion time of an older buffer once it does not. A cost
  model calibrated on a short launch stream will underestimate a long one.

**What a design must do about it.**

1. **Measure enqueue time separately from wall time.** Without that split, a
   queue-full stall reads as "the GPU got slower" when it is the host being
   blocked, and the two call for opposite fixes.
2. **Bound any "N launches with no host wait" claim at 64 on Metal, and say
   so in the same sentence.** The claim is still worth making. It is the
   count that has to be honest.
3. **Compose the two facts before counting.** A copy in the middle of a
   launch stream drains the queue, so the usable depth is not 64 launches
   per tree, it is 64 launches between copies. A per-round upload resets the
   pipeline as well as costing its own wait.
4. **Expect no assist.** MAX exposes no asynchronous copy on Metal,
   `"Metal stream not implemented"` is a literal string in the runtime so
   there is no second queue to overlap with, and there is no
   `MetalDeviceGraphBuilder` although `CUDADeviceGraphBuilder` and
   `HIPDeviceGraphBuilder` both ship. Whatever pipelining a design wants on
   this backend, it takes from the 64-deep queue alone. The graph half of
   that sentence was derived from a symbol table when it was written; it has
   since been **verified by execution** and the raise it produces is quoted
   in section 6.5.

### 6.3 What each backend provides, on these two points

**Two Metal rows were corrected on 2026-08-16 by execution. See 6.5.** The
copy rows used to read "**no**, measured by disassembly", from the same
reading that produced 6.1, and both are wrong for the destination this
library actually uses.

| Requirement | Metal | CUDA | HIP |
|---|---|---|---|
| `asynchronous-host-to-device-copy` | **yes** into a pinned `HostBuffer`, **no** into an arbitrary host pointer; measured by execution, 6.5 | unestablished; the API specifies one, MAX's use of it has not been read | unestablished, same reason |
| `asynchronous-device-to-host-copy` | **yes** into a pinned `HostBuffer`, **no** into an arbitrary host pointer; measured by execution, 6.5 | unestablished | unestablished |
| `queue-depth` | 64 command buffers, one per launch, measured by disassembly; not raisable, and 6.5 verifies the "not raisable" two further ways | unestablished | unestablished |
| `second-queue-or-stream` | none; `"Metal stream not implemented"` in the runtime, and `create_stream()` raises `createStream is not supported on this device` | unestablished | unestablished |
| `capture-and-replay` | none, **verified by execution**: `DeviceGraph.create` raises `createGraphBuilder() not supported on this device context` on an M4; `MetalDeviceGraphBuilder.cpp` absent from a driver set containing the CUDA and HIP equivalents. Section 6.5 | `CUDADeviceGraphBuilder` ships; never exercised here | `HIPDeviceGraphBuilder` ships; never exercised here |
| `event-or-fence` | **none**; `create_event()` raises, and every `MTLSharedEvent` and `MTLFence` selector has zero load sites. 6.5 | unestablished | unestablished |
| `host-visible-device-allocation` | **none reachable**; every buffer is shared storage, but the Mojo API hands out `gpuAddress` for a `DeviceBuffer` and `contents` for a `HostBuffer`, never both names for one allocation. 6.5 | unestablished | unestablished |
| `runtime-handle-out` | **none**; `DeviceContext` exposes no queue, command-buffer, or native handle, so no vendor shim can hook it. 6.5 | unestablished | unestablished |

"unestablished" is deliberately not "specified" here. Section 1's table can
say "specified" because it is quoting what an API documents about a kernel
primitive. These rows are about what the runtime *does* between the API and
the device, which is exactly the layer where Metal turned out to differ, so
quoting the API would be the wrong kind of answer.

### 6.5 What the readback path can and cannot reach, and the correction to 6.1

Added 2026-08-16 by the `cheap-sync` lane, whose question was whether one host
readback of a 136-byte split record can be made to cost what it costs on CUDA
rather than what a queue drain costs. Everything below is **measured**, by
execution on an Apple M4 unless it says disassembly. The reproductions are
`probes/readback_cost.mojo`, which runs every claim as a gated arm.

#### 6.5.1 `enqueue_copy` has two implementations and the destination picks one

**This is a correction to 6.1, and it is the load-bearing one.** 6.1 says
`enqueue_copy` on Metal blocks until the whole queue has drained and then
copies, in both directions, and it established that by disassembling
`enqueueCopyToDevice` / `enqueueCopyFromDevice`. The disassembly is not
retracted. What it missed is that it is not the only path.

| destination | behavior | how established |
|---|---|---|
| pinned `HostBuffer` (`enqueue_create_host_buffer`) | **asynchronous.** Enqueues a blit and returns | execution |
| arbitrary host pointer (a `List`'s `unsafe_ptr()`) | **synchronous.** Commits, waits, memcpys, exactly as 6.1 disassembled | execution |
| `map_to_host()` | **synchronous**, and it is a fresh host allocation with a copy in on entry and a copy out on exit, not an address remap | execution |

Download, **measured**: a kernel slow enough to be unambiguous, then a copy
into a pinned `HostBuffer`, read with no intervening `synchronize()`. All 64
of 64 words stale, four attempts out of four, and all 64 correct after the
`synchronize()`. The same kernel and the same read into a plain heap pointer:
0 of 64 stale, with no `synchronize()` anywhere.

Upload, **measured**: a pinned buffer filled with ones, uploaded, and then
overwritten with sevens with no drain in between; a kernel summed what the
device received. Across four repetitions the device saw the sevens once and
the ones three times. **Non-deterministic**, which is the signature of a live
race and not of either fixed ordering.

**What this changes, in the unsafe direction.**

- 6.1's bullet "every `ctx.synchronize()` that immediately follows an
  `enqueue_copy` with no enqueue in between is redundant on Metal" is
  **wrong** for the pinned destination, which is the one every transfer in
  this library uses. That `synchronize` is the only thing making the readback
  correct. `probes/readback_cost.mojo` carries `pinned_pair_nosync` and
  `pinned_one_nosync` as arms for exactly this reason: they execute the edit
  the bullet licensed and report 34 of 34 words wrong.
- 6.1's bullet that `StagingRing` can never save a wait on Metal, and that
  `HazardTracker`'s host-write hazard class is vacuous there, is **wrong**.
  The hazard is real and was reproduced. Both docstrings in
  `gpu_runtime.mojo` are corrected in place.
- **The race is latency-dependent, which is what makes it dangerous.** Under
  a fast kernel the unsafe shapes pass every time, because the blit wins;
  under a slow one they fail every time. A change validated on a small
  fixture will look correct and break on a large one. This is the same shape
  as the host-side table that once made tree 0 bit-identical and every tree
  after it diverge.
- It does **not** disturb 6.1.1. Removing thirteen pinned copies per tree
  measured 0.016 seconds, and this section supplies the mechanism that null
  was missing: a pinned copy is nearly free because it never waited, not
  because the drain it performed was cheap. Count round trips to predict time
  still holds, and is now better founded.

**What it does not change.** The synchronous path is real and 6.1 described
it correctly. A copy into an arbitrary host pointer still drains.

**What shipped on it, 2026-08-16.** `GpuSplitSearcher.download_words` takes
the synchronous path now, as `READBACK_PLAIN_ONE`: one `enqueue_copy` out of
one device allocation into a `List[Int32]`, and no trailing `synchronize()`,
because there is nothing left to wait for. It replaced two copies into two
pinned `HostBuffer`s followed by a `synchronize()`, four command buffers a
trip against two, and the probe measured 202.14 us against 124.85 with the
arms interleaved in one process. The record's two planes became one
allocation with two `create_sub_buffer` windows to make the single copy
possible; `gpu_split_search.SPLIT_RECORD_WORDS` states that layout and
`tests/test_gpu_readback_transport.mojo` checks the windows land where it
says, against a fixture whose every word differs from every other.

The gate on that edit is this section, and it is worth stating as a rule
because the next lane will want to drop a `synchronize()` somewhere else.
**A wait may be removed only where the destination is provably not pinned,
and the proof has to be about the type of the destination and not about what
the call site appears to pass.** `download_words` branches on
`ReadbackTransport.pinned_destination` rather than on an arm code for exactly
that reason: the branch that omits the wait is unreachable with a pinned
destination, so pointing a plain arm at a `HostBuffer` to save an allocation
cannot compile into a silent corruption. Every remaining `synchronize()` in
that file now says at its call site which of the two paths makes it
load-bearing.

#### 6.5.2 There is no event, fence, stream, callback, or completion query

Every one of these compiles and then fails at run time, which is why the
compiler is not sufficient here and the probe executes them:

| API | result |
|---|---|
| `DeviceContext.create_event()` | raises `eventCreate is not supported on this device` |
| `DeviceContext.create_stream()` | raises `createStream is not supported on this device`; `num_streams()` answers 1 |
| `DeviceContext.enqueue_cpu_function(f)` | raises `enqueue_cpu_function is only supported on CPU DeviceContexts` |
| `DeviceContext.handle/unsafe_ptr/native_handle/stream_handle` | do not exist; the compiler rejects each |
| `DeviceContext.query/is_idle/poll/is_complete` | do not exist; the compiler rejects each |

Independently, **measured by disassembly** of `libMGPRT.dylib`:
`newSharedEvent`, `newSharedEventHandle`, `newSharedEventWithHandle:`,
`newEvent`, `newFence`, `encodeSignalEvent:value:`,
`encodeWaitForEvent:value:`, `notifyListener:atValue:block:` and
`addCompletedHandler:` are each registered in the runtime's metal-cpp
selector table and each has **zero** load sites. The only synchronization
selectors with load sites are `commit` (8) and `waitUntilCompleted` (6). So
the runtime does not use a single Metal synchronization primitive beyond
committing a command buffer and blocking on it.

**A vendor `MTLSharedEvent` shim is therefore not constructible, and the
reason is not effort.** To make a wait cheaper than `waitUntilCompleted`, a
shim must either encode `encodeSignalEvent:value:` into the command buffer
MAX committed, or poll memory MAX's kernel wrote. The first needs MAX's queue
or command buffer, and the table above shows no handle comes out. The second
needs a host address for a device allocation, and 6.5.3 shows none is
reachable. Allocating the `MTLBuffer` outside MAX and handing its
`gpuAddress` to `enqueue_function` fails on residency: MAX calls
`useResource:usage:` for its own allocations only, and a bindless argument to
a non-resident buffer is undefined. The gap is upstream, and
`handoffs/upstream_max_metal_gaps.md` is the report.

#### 6.5.3 Every buffer is shared storage, and no allocation exposes both names

**Measured by disassembly.** Every Metal allocation MAX makes goes through
one `newBufferWithLength:options:` call site, and the one caller reached from
the device-context allocator passes `options == 0`, which is
`MTLStorageModeShared`. So the hardware premise of a zero-copy readback holds:
the memory really is host-addressable.

**Measured by execution**, and this is where it stops:

- `DeviceBuffer.unsafe_ptr()` and `DeviceBuffer.device_ptr()` return the same
  value, and it is the buffer's `gpuAddress`. An M4 returned 0x10000080000
  for a fresh device buffer and **faulted** on the first host load;
  consecutive allocations were 256 bytes apart in that space. Host
  allocations in the same process sat near 0x100b4c000. (`gpuAddress` has one
  load site, `setBuffer:offset:atIndex:` has none, and `useResource:usage:`
  has one, so kernel arguments are bound bindlessly by GPU address.)
- `HostBuffer.unsafe_ptr()` returns the buffer's `contents`, a real host
  address. A kernel handed it wrote nothing: the destination still held its
  pre-fill sentinel in all 34 words after a full drain. A kernel handed it as
  a source read zeros. Both directions, and both before and after the buffer
  had been used as an `enqueue_copy` destination, which rules out "the
  runtime had never seen it".

**So the 2026-08-15 `host_direct WRONG` result in
`bench/results/apple_m4_unified_memory_2026-08-15/` was never an ordering
bug.** It is an addressing mismatch: `enqueue_function` binds by GPU address
and the Mojo API hands out a CPU address for that allocation kind. The
ordering requirement for a direct read is satisfiable and already paid, since
one command buffer completion is Metal's visibility point for shared storage
and every readback pays one. What is missing is an address, and no amount of
fencing supplies it.

#### 6.5.4 The queue depth has no knob, verified three ways

6.2 established by disassembly that the queue is created with a bare
`[device newCommandQueue]`. Re-verified here from the same binary by counting
load sites rather than by reading the call site: `newCommandQueue` has one
store (the metal-cpp selector registration) and **one load**, at 0xe2b7c,
inside the Metal device context's initialization and immediately before its
`MODULAR_DISABLE_METAL_GPU_PRINT` lookup. `newCommandQueueWithDescriptor:`,
`newCommandQueueWithMaxCommandBufferCount:` and `setMaxCommandBufferCount:`
each have their registration store and **zero** loads.

Two further checks 6.2 did not make. MAX reads no environment variable
resembling a queue depth; the complete set of `MODULAR_*` and `METAL_*` names
in the runtime is context, telemetry, logging, affinity, cache, compiler-path
and health-check settings, and none of them is a queue parameter. And
`DeviceContext.__init__` has exactly four overloads, none of which takes one;
the compiler lists all four when handed a bad keyword.

**So the depth cannot be raised, and the derived one-in-one-out claim in
`gpu_resident_round.mojo` cannot be tested by raising it.**
`probes/readback_cost.mojo` tests the consequence instead, with a ladder of
launch-stream lengths on both sides of 64 that times only the enqueue loop.
A depth that blocks shows up as a knee in per-launch enqueue cost. That
measurement is the orchestrator's; this section only records that the ladder
is the available instrument and why.

#### 6.5.5 A toolchain trap that can invalidate a worktree's compile check

Not about the device, and recorded here because it can silently invalidate
everything above. On this machine, `mojo run -I <path>` does **not** resolve
the `mojotrees` package to `<path>` when another checkout of the same package
name exists. **Measured**: a fresh copy of `src/` at an unrelated path, with a
new module-level constant added, compiled without the constant being visible;
appending deliberately invalid syntax to that copy produced no error at all,
so the file was never read. The same tree copied to a package named
`mtprobe` compiled and exposed everything. Clearing the cache directory, a
private `MODULAR_HOME`, and a private `cache_dir` each changed nothing.

**Consequence for anyone working in a git worktree: `mojo precompile -I
src src/mojotrees` may be elaborating the shared checkout's source, not
yours, and it will report success.** The workaround that is known to work is
to copy the package under a different name and compile that. Whatever the
resolution mechanism is, it has not been identified, and identifying it is
worth someone's hour.

## 7. Gates and environment variables

| Gate | Refuses |
|---|---|
| `require_backend_covered(api)` | an API code outside the covered set |
| `require_device_can_host_kernels(contract, caps)` | a device that cannot host the kernels at all, checked once at session open |
| `require_bins_supported(n_bins)` | a bin count outside `[1, MAX_BINS]` |
| `require_primitives(contract, strategy)` | a strategy whose primitives the backend does not promise |
| `require_specializations_allowed(contract, selected)` | a kernel variant a plan *selected* on a named backend nobody has run |
| `require_launch_geometry(...)` | a launch the reported device cannot run |
| `require_histogram_launchable(...)` | all of the above for one resolved histogram launch |
| `require_device_float64(contract)` | Float64 device arithmetic, on every backend |

Two details of the specialization gate that are easy to get backwards and
that change whether it is correct:

**Selected, not compiled-in.** `histogram_gpu.build_kernel_features()`
reports `batched_leaf_kernel = True`, because linking that module
instantiates the batched kernels from `gpu_leaf_batching.mojo`. That is a
fact about compilation. Selecting them is a separate decision that
`apple_histogram_policy` makes only when `SPEC_LEVEL_BATCHED` was asked for
by name and never from `auto`. The gate takes what the plan chose, so it
fires on the choice. Handing it the compiled set instead would refuse every
run on every backend. `require_histogram_launchable` takes both values
separately for this reason: the compiled set decides the threadgroup
footprint, the selected set faces the validation gate.

**The gate is only as strong as the backend identification.** An
unidentified backend (`API_UNKNOWN`) is not refused. It is the value every
launch carries today, because nothing in the shipping code reads an API name
before it launches and `MOJOTREES_GPU_BACKEND` is unset on an ordinary run,
so an unidentified device is indistinguishable here from the Apple part this
repository was developed on. Refusing it would refuse every run that ships,
including every Metal one. A CUDA device therefore escapes the gate until
the reported API name reaches the policy layer, which is filed as a patch
request in `handoffs/connect_20_gpu_portability.md` and is the single change
that turns this gate from a declaration into an enforcement.

| Variable | Effect |
|---|---|
| `MOJOTREES_GPU_BACKEND` | Names the API for reporting. Read by `device_policy.env_declared_api`, never a capability number. A declaration, not a detection |
| `MOJOTREES_GPU_BACKEND_UNVALIDATED=1` | Acknowledges selecting a specialization on a backend with no validation record, and runs it anyway. Any number measured under it must be reported with it |
| `MOJOTREES_GPU_READBACK` | Names the split-record readback transport (`plain_one` by default; also `plain_pair`, `pinned_one_sync`, `pinned_pair_sync`). Named transports only, never a number. Refuses `map` and the two `nosync` arms, which 6.5.1 measured wrong. `GpuSplitSearcher.set_readback_transport` is the in-process handle a window uses, since two arms have to be interleaved in one process to be comparable at all |

The second follows `MOJOTREES_GPU_TRANSFER_UNPROVEN` in
`unified_memory_policy.mojo` deliberately: the evidence that would lift the
gate can only be produced by running the thing the gate blocks, so without
an override the gate is unsatisfiable by construction.

## 8. Specialization points, and why none is taken

Each of these is a place a backend could diverge without forking the source.
None is enabled anywhere, on any backend, because none has been measured on
any backend.

| Point | Where | What would have to be true first |
|---|---|---|
| Bin-capacity kernels | `gpu_histogram_specializations.bin_capacity_for`, `KernelFeatures.specialized_bin_kernels` | The kernels take a capacity parameter. Then a 32-bin histogram stops occupying 3 KiB of threadgroup memory, which raises residency on any device where threadgroup memory is the binding constraint |
| Packed four-byte bin loads | `plan_packed_window`, `pack4_bins`, `KernelFeatures.packed_bin_loads` | A measurement that a four-byte aligned load beats four one-byte loads on the device in hand. The portable implementation beside it produces the identical integers and is the definition |
| Batched multi-leaf launches | `gpu_leaf_batching.mojo`, `KernelFeatures.batched_leaf_kernel` | The batched kernels are compiled in and validated. Tiles pack onto `grid.y`, whose portable bound is known, rather than onto a `grid.z` whose bound is not |
| Float64 accumulation | `require_device_float64` | A variant that reproduces the fixed-point Int32 integers exactly on a backend that has Float64 |
| Subgroup reductions | nothing exists | A width read from the device, on a backend that answers the query, plus a portable implementation producing identical integers |

The rule the specialization modules enforce structurally, and that this
document restates so it is not lost: a non-portable operation may be reached
only through a flag on a capability record, and only where a portable
implementation of the same arithmetic sits next to it producing the
identical integers. The portable implementation is the definition. The fast
path is an optimization that has to reproduce it.

## 9. Per-backend notes

`docs/NVIDIA_GPU.md` and `docs/AMD_GPU.md` carry what to expect and what to
check first on each. `docs/GPU_VALIDATION.md` carries the procedure that
turns either from `portable-untested` into `exercised`, and the empty record
sections that procedure fills in.

### 6.4 A tension with the timeline, resolved 2026-08-16

Recorded first as unresolved, on the grounds that resolving it by picking the
more convenient half would be the wrong move. It has since been resolved by
measurement rather than by picking, and the resolution is kept below the
original statement so that the reasoning is visible and not just the answer.

Section 6.1 establishes by disassembly that every `enqueue_copy` drains the
queue. Section 6.2's audit found that `gpu_split_search._launch` calls
`_copy_tables`, which issues **four** copies, on every launch. Taken together
those say a per-split search costs four upload drains plus one download drain,
so a thirty-one leaf tree should show on the order of **155 serialization
points**.

The Metal System Trace measured **32.1 per round** (`docs/METAL_TIMELINE.md`),
which is almost exactly one per split, and it measured that by counting where
the host actually blocked rather than by reading source.

Both are measurements and they disagree by a factor of five. Possible
resolutions, none of them established at the time:

- The measured path was `_device_search_resident`, which searches a whole
  frontier through `enqueue_frontier` rather than a node at a time, so
  `_copy_tables` may run far less often than once per split on that path while
  running once per launch on the node-at-a-time `enqueue` path that section
  6.2's audit read.
- A drain on an already-empty queue may be cheap enough that the trace's
  blocking-blit criterion does not count it, in which case the count is right
  and the cost is not five times larger.
- The disassembled path may not be the one MAX takes for small buffers.

**The second candidate is the one that holds.** Section 6.1.1's A/B removed
thirteen of exactly these copies per tree and **measured** 0.016 seconds, so a
drain on a queue that holds nothing costs so little that an instrument
selecting for blocking blits is right not to count it. The count the trace
reported was never five times low; it was counting the thing that costs, which
is the round trip, and the source-derived 155 was counting a different thing
and pricing it wrongly. The first candidate may also be true about how often
`_copy_tables` runs, and nothing here establishes it either way; it is no
longer needed to explain the gap.

What follows from the tension regardless, and what still stands: **the wait
count of any path must be measured rather than derived from the source.** This
project produced a source-derived count and an instrument-derived count for the
same code that differed five-fold, and the instrument was right. The instrument
is the one to believe about what the machine did. The disassembly is the one to
believe about what the API guarantees. They answer different questions, and the
gap between them was exactly where the design assumption 6.1.1 withdraws had
been hiding: the disassembly's answer, which is about ordering, was read as an
answer about time.

### 6.5 Device graphs: the Mojo API is complete, Metal does not back it

**Verified by execution, 2026-08-16, MAX 26.5.0 / Mojo 1.0.0 (ed45d567),
Apple M4.** `tools/probe_device_graph.mojo` reproduces it in one command.

`DeviceGraph.create(ctx, build)` raises on this device:

```
At max/mojo/max/gpu/host/device_graph.mojo:285:17:
createGraphBuilder() not supported on this device context
```

That is the whole answer, and it arrives at the first call. Nothing is
captured, nothing is replayed, and there is no partially-working path to
characterize: the failure is at *builder creation*, so every one of
`add_function`, `add_copy`, `add_memset`, `add_empty`, `region`,
`create_buffer`, `recording_context` and `replay` is unreachable on Metal.

**The exact missing API.** Not a Mojo symbol. `max.gpu.host.device_graph`
compiles, its types construct, and `DeviceGraph.create` type-checks and
elaborates against a real kernel; the gap is one layer below, in the C++
driver. The shipped `libAsyncRTMojoBindings.dylib` names its driver sources
in its string table, and the device-context implementations it contains are:

```
MLRT/lib/Driver/DeviceContext/CPU/CpuDeviceContext.cpp
MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp
MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceGraphBuilder.cpp
MLRT/lib/Driver/DeviceContext/DeviceContext.cpp
MLRT/lib/Driver/DeviceContext/HIP/HIPCompilationDevice.cpp
MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp
MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceGraphBuilder.cpp
MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.cpp
MLRT/lib/Driver/DeviceContext/Virtual/VirtualDeviceContext.cpp
```

CUDA has a graph builder. HIP has a graph builder. Metal has a device context
and no graph builder, so `MetalDeviceContext` inherits the base
`DeviceContext::createGraphBuilder()`, whose entire body is the throw quoted
above. The missing API is therefore **`MLRT/lib/Driver/DeviceContext/Metal/
MetalDeviceGraphBuilder.cpp`, a file that does not exist in this build**, and
nothing on our side of the boundary can substitute for it. There is no
environment variable that changes this: the only `MODULAR_*` and `MTL_*`
knobs in the binary are `MODULAR_DEBUG`, `MODULAR_DERIVED_PATH`,
`MODULAR_DISABLE_METAL_GPU_PRINT`, `MODULAR_DRIVER_PLUGINS`,
`MODULAR_ENABLE_AFFINITY`, `MODULAR_ENABLE_CACHE_LOGGING`, `MODULAR_HOME`,
`MODULAR_NVPTX_COMPILER_PATH`, `MODULAR_PROFILER_PLUGIN` and
`MODULAR_THREAD_BUSY_WAIT_US`, none of which is a graph gate.

**This upgrades a claim the repository already made, it does not overturn
it.** Section 6.2's fourth point and section 6.3's `capture-and-replay` row
already said "no `MetalDeviceGraphBuilder`", derived from the same string
table. What was never established is the thing that actually matters, and it
is not the same claim: a missing implementation file is consistent with a
silent no-op, with a fallback that batches submissions, and with a hard
raise, and those three would have led a design to three different places.
It raises. Loudly, at the first call, before any node is added. Of the three
that is the best of the bad outcomes, because no design can accidentally
build on it and no benchmark can accidentally measure a graph that captured
nothing.

**What the API is, for whoever revisits this on CUDA or HIP.** Recorded
because the question will come back and the surface is not in any source
this install ships; it was recovered from the compiled package's docstrings
and mangled signatures, so treat the shapes as read rather than as run.

- `DeviceGraph.create(ctx, build)` is the only entry point. It calls `build`
  with a fresh `DeviceGraphBuilder`, then instantiates. The builder and every
  `DeviceGraphNode` it hands out are origin-branded to the `create` scope, so
  a handle cannot escape the callback or be mixed into another graph. The
  borrow checker enforces that, which is why there is no way to hold a
  half-built graph across a host decision.
- Nodes: `add_function` (compiled `DeviceFunction`, `DeviceExternalFunction`,
  a `{var}`-capturing closure, or a kernel passed as a parameter),
  `add_copy` in all three directions, `add_memset`, and `add_empty`.
- Ordering is an explicit DAG, not enqueue order: every `add_*` takes a
  required `dependencies=` list of predecessor handles, and an empty list
  means "graph root". `add_empty` exists to express an `m`-to-`n` barrier in
  `m + n` edges instead of `m * n`, and `builder.region(work)` runs a closure
  and returns one handle joining everything it added.
- `builder.recording_context()` returns a `DeviceContext` view on which
  ordinary `enqueue_*` calls record into the graph instead of executing.
  That is the migration path that would matter to us: existing launch code
  records unmodified. Host-visible waits -- `synchronize()`, events, timers
  -- raise inside it, by design, because they observe a device result that
  does not exist until replay.
- `builder.create_buffer[dtype](size, is_host)` allocates with a lifetime
  tied to the graph, and the docstring is explicit that graph allocations
  must go through it rather than `DeviceContext.enqueue_create_buffer`.
- `DeviceGraph` itself exposes `replay()`, a refcounting `copy`, and
  `take_handle`. **That is the whole surface, and the absence is the
  important part: there is no node-update call.** CUDA's
  `cudaGraphExecKernelNodeSetParams` is not exposed, so between two replays
  of an instantiated graph *nothing* may change except the contents of
  device memory. Grid dimensions, block dimensions, shared-memory bytes and
  every kernel argument are frozen at capture.

**Does our step-descriptor design satisfy those constraints? Half of it does,
and the half that does not is the expensive half.** The device-resident plane
already had kernels read their decisions out of device memory rather than take
them as launch arguments, and that is exactly what the frozen-argument rule
demands: pointers into session-owned buffers are stable across a whole fit, so
every *argument* our per-tree sequence passes would survive capture unchanged.
That much is a real dividend from work already done. What does not survive is
launch *geometry*. Our per-node and per-frontier launches size their grids to
the live frontier, and a captured graph freezes `grid_dim`, so a captured
per-tree sequence would have to launch max-sized grids that early-out on a
device-read descriptor, paying for the largest frontier on every round.
Whether that trade is worth taking is unmeasurable here and must not be
guessed at; it is the first thing to measure on a CUDA or HIP device, and it
is the reason this section stops rather than recommending anything.

**What this does not say.** It says nothing about whether capture-and-replay
would be *faster* on a backend that has it -- no timing was taken and none
should be quoted from this work. It says nothing about the CPU or Virtual
device contexts, which were not probed. And it does not close the question
for MAX versions after 26.5.0: a `MetalDeviceGraphBuilder.cpp` is a file
Modular can add, and `tools/probe_device_graph.mojo` exists so that the day
it appears, nobody has to re-derive any of this.
