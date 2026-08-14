# Connect 05: device policy, Apple policy, unified memory, initialization

Lane 05. Owned files: `src/mojoboost/device.mojo`,
`src/mojoboost/device_policy.mojo`, `src/mojoboost/apple_gpu_policy.mojo`,
`src/mojoboost/unified_memory_policy.mojo`,
`src/mojoboost/initialization.mojo`,
`python/mojoboost/device_selection.py`,
`python/mojoboost/diagnostics.py`, and this file.

Nothing outside that list was edited. Everything that needs to change
elsewhere is a patch request below, quoted exactly.

**Nothing was compiled, run, or tested.** No claim here is a claim about
correctness, performance, parity, packaging, or hardware. The commands that
would establish any of that are at the end and are marked UNRUN.

---

## 0. A preservation note that is not about code

While this lane was working, another lane committed the whole working tree
as `dc21f03 Connect accelerator and public API foundations`, which swept
this lane's in-progress edits to all six owned files into that commit. This
lane did not commit and did not ask for that.

Verified afterwards, by `git diff HEAD~1..HEAD` restricted to the six owned
files: every deletion in that commit's version of those files is a line this
lane deleted, and no other lane's content appears in them. Nothing was lost
and nothing of anyone else's was overwritten. It is recorded because the
checkout is shared and a whole-tree commit can capture another lane's
half-finished state, which is the failure mode rather than this particular
outcome. **Commit by explicit path.**

---

## 1. Implementations found

Inventory taken before any edit. Six places answer some part of "where does
this run, on what hardware, moving bytes how".

| # | Where | What it implements | State found |
| --- | --- | --- | --- |
| 1 | `device_policy.mojo` | The whole decision engine: vocabulary, GPU support gates, memory estimate, capability detection, crossover evidence table, refusal semantics. `DeviceRequest` / `DeviceCapabilities` / `DeviceDecision`. | Complete and authoritative. Struct-only interface, so unreachable from any non-Mojo caller. |
| 2 | `device.mojo` | Thin forwarding facade over #1. | Already a facade, no policy of its own. Forwarded only the four-argument shape-only entry. |
| 3 | `apple_gpu_policy.mojo` | `GpuProfile`, tuning plan, partial-histogram budget, `CROSSOVER_DISABLED`. | Leaf, imported by #1. Already fused: #1 imports `CROSSOVER_DISABLED` and `partial_budget_bytes` rather than restating them. |
| 4 | `unified_memory_policy.mojo` | Per-role transfer routes, structural eligibility, evidence ladder, sync contracts, footprint accounting. | **Fully isolated.** Imported by nothing in `src/`. Not in `src/mojoboost/__init__.mojo`. Only `bench/apple/unified_memory.mojo` referenced it. |
| 5 | `initialization.mojo` | Ten startup phases, `StartupTrace`, `FitLatency`, `WarmupPlan`, `BuildIdentity`. | **Fully isolated.** Imported by nothing in `src/`. Not exported. `WarmupPlan.report()` emitted lines no parser read. |
| 6 | `python/mojoboost/device_selection.py` | Workload extraction from `X`/`y`, decision formatting, `DeviceReport`. | Already policy-free by construction. Stuck on the `"narrow"` contract because the binding it needs does not exist. |

There is **no second decision engine**. An earlier round already collapsed
the Python one into #1; the docstrings in #1 and #6 record that. What was
left was not duplication but disconnection: three of the six modules had no
path to a caller.

Also found, and not a duplicate: `apple_histogram_policy.mojo` (lane 04)
carries `caps_from_profile` / `profile_from_caps` converters against
`GpuProfile`. That is a conversion, not a second policy, and it stays where
it is.

---

## 2. Call path, before and after

### Before

```
estimator.fit
  -> MojoBoostBase._resolve_device            python/mojoboost/__init__.py
     -> device_selection.select_device        (already wired by lane 07)
        -> _NarrowNativePolicy                 because decide_device is unbound
           -> _mojoboost.resolve_device        bindings/_mojoboost.mojo
              -> device.resolve_device         device.mojo
                 -> device_policy.resolve_device
                    -> DeviceRequest(shape only) + DeviceCapabilities.detect()
                    -> decide_device            <- the whole engine, asked 1/3 of the question

unified_memory_policy.mojo   -- no caller
initialization.mojo          -- no caller
```

Three consequences of that shape, all of them real:

- The objective gate, the bin-limit gate, the sparse block, the validation
  block, and the memory gate **never ran** for any estimator call, because
  the request carried only rows, features, and outputs. `device='gpu'` on a
  sparse matrix passed the policy and had to be caught by a hand-written
  Python raise further down.
- The report — blocking reasons, warnings, memory estimate, policy version,
  evidence identifier — was computed natively on every call and then
  discarded at the binding, which returns a bare device name.
- Transfer routes and startup state were not in the picture at all.

### After

```
estimator.fit
  -> MojoBoostBase._resolve_device
     -> device_selection.select_device
        -> _FullNativePolicy                   once the binding in §5.1 lands
           -> _mojoboost.decide_device
              -> device.decide_device_report          device.mojo          [NEW]
                 -> device_policy.decide_device_report                     [NEW]
                    -> DeviceRequest(full workload)
                    -> DeviceCapabilities.detect(SessionState.from_env())
                       |- plan_session_routes()  <- unified_memory_policy   [NEW EDGE]
                       |- SessionState           <- initialization          [NEW EDGE]
                    -> decide_device
                    -> DeviceDecision.serialize()  -> key=value lines
        -> DeviceReport                        parses, formats, never decides

trainer / predictor (Mojo)
  -> device.resolve_device_full                                            [NEW]
     -> device_policy.resolve_device_full                                  [NEW]
        -> decide_device -> raise_if_blocked

caller holding an open DeviceContext
  -> device.decide_device_report_reported                                  [NEW]
     -> device_policy.capabilities_from_reported                           [NEW]
        -> GpuProfile.from_reported -> PROFILE_REPORTED
```

`decide_device` itself is unchanged and still pure: it opens no device,
reads no environment, and takes no dataset. Everything new that reads the
environment reads it in `DeviceCapabilities`, which is where the environment
was already being folded in.

---

## 3. Connections completed (owned files only)

### 3.1 A flat seam, so non-Mojo callers stop needing one of their own

`decide_device` takes structs. A CPython binding builds arguments out of
`PythonObject`s and cannot construct a `DeviceRequest`; a C API and a CLI
are in the same position. Without a flat form each of them grows its own
marshalling, and the one that gets it slightly wrong is a second policy
nobody meant to write.

Added to `device_policy.mojo`, mirrored in `device.mojo`:

- `decide_device_report(device: String, n_rows, n_features, n_outputs=1,
  n_bins=BINS_UNSPECIFIED, objective=OBJECTIVE_UNSPECIFIED, sparse=False,
  categorical=False, has_missing=False, uses_validation=False) raises ->
  String` — the serialized decision. Ten parameters, in exactly the order
  `_FullNativePolicy.decide` in `device_selection.py` already sends them.
- `decide_device_report_reported(... , reported_api, reported_arch,
  core_count, max_threads_per_block, max_shared_memory_per_block,
  memory_budget_bytes=0, context_open=True, kernels_ready=False,
  warmup_level=0) raises -> String` — the same engine over `PROFILE_REPORTED`
  capabilities.
- `resolve_device_full(device: Int, ... same ten fields) raises -> Int` —
  the raising form for a trainer that knows its whole workload, as against
  `resolve_device`, which knows only the shape.
- `capabilities_from_reported(...)` — the call site `GpuProfile.from_reported`
  and `PROFILE_REPORTED` both describe, in one function.

Sentinel normalization is `_normalized_bins` / `_normalized_objective`, and
both are marshalling, not policy. `_normalized_objective` deliberately does
**not** fold `-1` into `OBJECTIVE_UNSPECIFIED`: `-1` is the multiclass
marker and a real code, and folding it would silently skip the objective
gate for every multiclass run.

### 3.2 Unified memory connected to the policy contract

`unified_memory_policy.mojo` gained a session-wide value and
`device_policy.mojo` now carries it.

New in `unified_memory_policy.mojo`:

- `SessionMemoryPlan` — one `RouteDecision` per buffer role, plus the
  requested route, the unified-memory flag, and the acknowledgment flag.
  Predicates: `all_default`, `honored_count`, `any_unproven`,
  `needs_kernel_retirement`, `for_role`. Wire format: `report()`.
- `plan_session_routes(unified_memory) raises -> SessionMemoryPlan`.
- `describe_session_plan(plan)` — the one-line form.
- `SessionMemoryPlan.staged(unified_memory=False)` — the shipped
  all-`copy_staged` plan, built from `sync_contract(DEFAULT_ROUTE)` rather
  than from a repeated literal.

The plan is built with `explain_route`, not `resolve_route`, and the
difference is deliberate: a *report* must describe all eight roles, so a
role that is structurally blocked appears as the default with its reason
rather than erasing the plan for the other seven. The *refusal* still
happens, at the allocation site, where `resolve_from_env` raises.
`route_block_reason` is the single gate and both go through it, so the
report that explains and the allocator that refuses are not two policies.

In `device_policy.mojo`, `DeviceCapabilities` gained
`var transfer: SessionMemoryPlan`, populated by `detect()` and
`from_profile()`. It is **carried, never consulted for a backend choice**.
Two warnings read it and neither changes the selected device:

- `WARN_UNPROVEN_TRANSFER_ROUTE` (11) — some buffer is on a route enabled
  only by `MOJOBOOST_GPU_TRANSFER_UNPROVEN=1`.
- `WARN_KERNEL_RETIREMENT_ROUTE` (12) — some buffer is on a route the
  kernels read directly, so the host cannot refill it until the kernels
  retire and `StagingRing`'s overlap argument does not hold. This is a
  synchronization change, not only an allocation one, and it is the finding
  most likely to be missed.

**No zero-copy claim is made anywhere.** `Footprint.resident_bytes_unknown`
is still unconditionally True, `EvidenceLedger.installed()` is still empty,
so `plan_session_routes` returns all-`copy_staged` in every context this
repository controls, and the serialized decision reports requested bytes
only. Whether two allocations are the same physical pages is not visible
from inside the process and nothing added here pretends otherwise.

### 3.3 Initialization connected to the policy contract

New in `initialization.mojo`:

- `SessionState(context_open, kernels_ready, warmup_level)` with `cold()`,
  `from_env()`, `is_cold()`, `paid_nothing()`, `describe()`.
- `session_state_from_trace(trace, warmup_level=WARMUP_OFF)` — reads the
  state off what a `StartupTrace` has *observed*. Counts, not durations, so
  it answers correctly on an untraced run, which is the run a device
  decision is usually made on.

`initialization.mojo` remains a leaf: it imports only the standard library.
The dependency runs one way, `device_policy` importing it and never the
reverse.

`DeviceCapabilities` gained `var session: SessionState`. Again carried, not
consulted: the only rule that reads it is `WARN_COLD_SESSION` (10), which
says that a GPU run on a process with no open context also pays context
creation and first-launch kernel creation, and that comparing that against a
warm CPU run measures the wrong thing.

`from_env()` is deliberately still *cold*: `MOJOBOOST_GPU_WARMUP` says what
a caller intends to front-load, not what it has already done, and reading an
intention as an accomplishment reports an unpaid cost as paid.

**Why none of this selects a device.** A warm session and a fast transfer
route are performance facts. `crossover_rules()` is the only place a
performance fact is allowed to change a device, it is empty, and the module
docstring says why. Letting a warm session tip `auto` to the GPU would be
inventing a threshold, which is exactly what this policy refuses to do.

### 3.4 Python: formatting the two new sections

`device_selection.py`:

- `TransferRoute` value type; `_split_transfer`; `transfer` added to the
  repeated-key handling in `_parse_decision`.
- `DeviceReport.transfer_routes`, `_transfer_line()`, `_session_line()`,
  two new lines in `explanation`, `transfer_routes` in `to_dict()`.
- No new judgment. Every field is read out of the native lines.

`diagnostics.py`:

- `WarmupSummary` and `parse_warmup()`. `WarmupPlan.report()` has been
  emitting `warmup.*` lines since that module was written and **nothing
  parsed them**, so a run with `MOJOBOOST_GPU_WARMUP=train` produced a
  record no report could show. `report_from_trace` now picks them up from
  the same text, so a trace and a plan can be concatenated and pasted in
  together.
- `WATCHED_ENV` gained `MOJOBOOST_STARTUP_REPORT_FD`,
  `MOJOBOOST_GPU_TRANSFER`, `MOJOBOOST_GPU_TRANSFER_UNPROVEN`. Listed, not
  interpreted — the device ones still belong to `device_selection`.
- `WarmupSummary` keeps the created/timed distinction: `total_ns` is zero on
  an untraced run even when kernels were created, because
  `WarmupPlan.note_created` records a duration only when the owning trace is
  enabled. Creation is tracked by the flag, never by a nonzero duration.

---

## 4. Duplicates fused, quarantined, and corrected

**Fused — the objective lists.** `device_policy.mojo` carried three copies of
facts that `objective_registry.mojo` declares itself the one table for:

| Was | Now |
| --- | --- |
| `comptime LAMBDARANK = 7`, mirroring `ranking.mojo` | imported from `objective_registry` |
| `is_builtin_objective`, an eleven-way `or` chain | delegates to `objective_is_builtin` (byte-for-byte the same chain) |
| `gpu_objective_is_device_resident`, returning `is_builtin_objective` | delegates to `objective_gradients_on_device` |

The third one also closes a real semantic gap, not just a duplicate: the old
version answered `is_builtin_objective`, which is the wrong question.
`objective_gradients_on_device` additionally excludes `MULTICLASS`, whose
softmax derivatives have a device kernel of their own reached through
`train_multiclass_gpu` rather than through the single-output
`fill_gradients_device`. `gpu_objectives_native.supports_device_objective`
already deferred to that function, so all three predicates now agree by
construction rather than by three lists being edited in step.

Safe by inspection: `objective_registry` imports only `boosting` (which
`device_policy` already imported) and `metrics` (which imports nothing
local), so it reaches neither `model.mojo` nor the GPU kernel stack. The
`from .boosting import (...)` block shrank to `CUSTOM`, the only code
`_collect_blocks` still names directly.

**Fused — the rest.**

- The staged `SyncContract` literal that `SessionMemoryPlan.staged()` would
  have needed now comes from `sync_contract(DEFAULT_ROUTE)`. Two spellings
  of the staged contract is exactly the drift `unified_memory_policy.mojo`
  was written to prevent.
- `unified_memory_policy`'s `block_reason_name` and `EVIDENCE_NONE` collide
  by name with `device_policy`'s. They are imported under
  `transfer_block_name` and `transfer_evidence_name`. Importing them bare
  would shadow this module's and make a report name a *route* refusal as the
  reason a *GPU request* was denied.

**Not fused, deliberately.** `MAX_GPU_ROWS`, `MIN_GPU_BINS`, and
`MAX_GPU_BINS` stay copies of `histogram_gpu.mojo`. Importing that module
drags the whole `max.gpu.*` kernel stack into a layer that has to stay
compilable on a machine with no accelerator, which is the same reason
`objective_registry.mojo` does not import `gpu_objectives_native.mojo`.
Three mirrors down to three from six, and the three that remain are the ones
with a hard reason.

**Corrected, and this one matters.** Two comments in `device_policy.mojo`
claimed `tests/parallel/test_device_policy.mojo` pins the mirrors and the
`MemoryEstimate` invariants. **That file does not exist** and nothing in the
repository asserts either. Both comments now say so. A false claim of a
guarantee is worse than no guarantee: a drifted `MAX_GPU_ROWS` admits a
workload the kernels cannot index, and a reader who believes a test is
watching will not check.

---

## 5. Cross-lane patch requests

Exact, minimal, and in dependency order. None of these files was touched.

### 5.1 Task 06 — `bindings/_mojoboost.mojo` (the one that unblocks the rest)

This is the single change that moves Python off the `"narrow"` contract.

**(a)** In the `from mojoboost.device import (...)` block at line 71, add
two names:

```mojo
from mojoboost.device import (
    decide_device_report as mojo_decide_device_report,   # ADD
    device_name as mojo_device_name,
    gpu_available as mojo_gpu_available,
    parse_device,
    resolve_device as mojo_resolve_device,
    resolve_device_full as mojo_resolve_device_full,     # ADD
)
```

**(b)** Add the wrapper. Ten parameters, in this order — it is the order
`_FullNativePolicy.decide` in `python/mojoboost/device_selection.py` already
sends and must not be reordered:

```mojo
def decide_device(
    device: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    n_outputs: PythonObject,
    n_bins: PythonObject,
    objective: PythonObject,
    sparse: PythonObject,
    categorical: PythonObject,
    has_missing: PythonObject,
    uses_validation: PythonObject,
) raises -> PythonObject:
    """The whole device decision as `key=value` lines. See `serialize` in
    src/mojoboost/device_policy.mojo for the format.

    Does not raise for a workload it refuses: the refusal is `blocked=true`
    with `message=` saying why, and Python turns that into
    DeviceUnavailableError. It does raise for an unknown device name, a
    shape with no rows or features, and an unparsable
    MOJOBOOST_GPU_TRANSFER. `n_bins` negative and `objective` below -1 mean
    undeclared; -1 is the multiclass marker and is a real objective code."""
    return PythonObject(
        mojo_decide_device_report(
            String(py=device),
            Int(py=n_rows),
            Int(py=n_features),
            Int(py=n_outputs),
            Int(py=n_bins),
            Int(py=objective),
            Int(py=sparse) != 0,
            Int(py=categorical) != 0,
            Int(py=has_missing) != 0,
            Int(py=uses_validation) != 0,
        )
    )
```

**(c)** Register it beside the existing device functions at line 187:

```mojo
        m.def_function[gpu_available]("gpu_available")
        m.def_function[resolve_device]("resolve_device")
        m.def_function[decide_device]("decide_device")          # ADD
```

**(d)** Optional, and only if lane 08 exposes a name-to-code function:
`objective_code(name) -> Int` bound to `objective_from_name` in
`params.mojo`. `_code_for_objective_name` in `device_selection.py` already
looks for exactly this name and returns None when it is absent, so a
name-only caller gets its objective gated once it exists and is reported as
undeclared until then. Do not add a name table in Python; that is the second
implementation this whole layer exists to not have.

Keep `resolve_device` bound. `device_selection.py` falls back to it, and
`python/mojoboost/__init__.py` calls it directly when `device_selection`
cannot be imported at all.

### 5.2 Task 07 — `python/mojoboost/__init__.py`

**(a) Pass a full workload.** `_resolve_device` (line 1613) builds a
shape-only `Workload` at line 1648:

```python
            if workload is None:
                workload = _policy.Workload(
                    n_rows, n_features, n_classes=n_outputs
                )
```

Every caller leaves `workload=None`, so *every* estimator decision skips the
objective, bin-count, sparse, missing, categorical, validation, and memory
gates. The fix is at the four call sites, not here: each should pass what it
already knows.

```python
        workload = _policy.Workload(
            n_rows,
            n_features,
            objective_code=self._objective_code(),
            n_classes=n_outputs,
            max_bin=int(self.max_bin),
            sparse=is_sparse,
            categorical=bool(categorical_features),
            has_missing=bool(self.use_missing),
            has_eval_set=eval_set is not None,
        )
        device = self._resolve_device(
            n_rows, n_features, n_outputs, workload=workload
        )
```

**(b) Retire four duplicate Python-side refusals.** Each of these restates a
block the native engine already computes and returns with a reason, a code,
and a report:

| Line | Python raise | Native block it duplicates |
| --- | --- | --- |
| 1876 | "validation metrics are scored on the CPU" | `BLOCK_VALIDATION_SET` (7) |
| 2566 | "sparse input trains on the CPU" | `BLOCK_SPARSE_INPUT` (3) |
| 2995 | "custom objectives train on the CPU" | `BLOCK_CUSTOM_OBJECTIVE` (4) |
| 3902 | "lambdarank trains on the CPU" | `BLOCK_RANKING_OBJECTIVE` (5) |

Once (a) lands, `select_device` raises `DeviceUnavailableError` (a
`RuntimeError` subclass) carrying the native message and the whole report,
for the same inputs. Retire them **only after** 5.1 and (a) are both in: on
the narrow contract they are the only thing enforcing those refusals, and
removing them first would let `device='gpu'` on a sparse matrix reach the
trainer. Sequencing matters more than the cleanup.

**(c)** Nothing else. The `device_selection` routing lane 07 already added is
the right shape and this lane did not change its contract.

### 5.3 Task 01 — `src/mojoboost/__init__.mojo` and `gpu_runtime.mojo`

**(a) Export the contract.** The `from .device import (...)` block at line
124 exports seven names; the flat entry points and the sentinels are not
among them, so no Mojo caller outside this package can reach them:

```mojo
from .device import (
    AUTO_DEVICE,
    BINS_UNSPECIFIED,                  # ADD
    CPU_DEVICE,
    GPU_DEVICE,
    NO_DEVICE,                         # ADD
    OBJECTIVE_UNSPECIFIED,             # ADD
    decide_device_report,              # ADD
    decide_device_report_reported,     # ADD
    device_name,
    gpu_available,
    parse_device,
    resolve_device,
    resolve_device_full,               # ADD
)
```

Optionally also export the structs for a caller that would rather hold a
decision than parse one:

```mojo
from .device_policy import (
    DeviceCapabilities,
    DeviceDecision,
    DeviceRequest,
    decide_device,
    describe_decision,
)
from .initialization import SessionState, StartupTrace, session_state_from_trace
from .unified_memory_policy import SessionMemoryPlan, plan_session_routes
```

**(b) Report real capabilities from `GpuSession`.** `GpuSession` owns the one
`DeviceContext` and is the only thing that can answer truthfully. Today every
decision runs on `PROFILE_FALLBACK`, so the memory gate, the unified-memory
warning, and any future hardware-scoped crossover rule are all inert. The
call site, using the entry point added in §3.1:

```mojo
# on GpuSession, once ctx is open
def device_report(
    self, device: String, n_rows: Int, n_features: Int, n_outputs: Int,
    n_bins: Int, objective: Int, sparse: Bool, categorical: Bool,
    has_missing: Bool, uses_validation: Bool,
) raises -> String:
    return decide_device_report_reported(
        device, n_rows, n_features, n_outputs, n_bins, objective,
        sparse, categorical, has_missing, uses_validation,
        self.ctx.api(),               # or whatever spells the API name
        self.ctx.arch_name(),         # architecture string
        self.ctx.core_count(),
        self.ctx.max_threads_per_block(),
        self.ctx.max_shared_memory_per_block(),
        self.ctx.memory_budget_bytes(),   # 0 if unavailable; 0 means unreported
        True,                              # context_open
        self.kernels_created(),            # kernels_ready
        env_warmup_level(),
    )
```

The attribute accessor spellings are lane 01's to confirm against the MAX
API; this lane did not open a device and does not know them. Pass `0` for
any attribute the backend does not report — `GpuProfile.from_reported`
sanitizes each one to the conservative portable constant, and a zero memory
budget is read as *unreported*, which disables the memory gate rather than
blocking on a made-up number.

**(c) Feed the startup trace back.** `GpuSession` should hold a
`StartupTrace` and a `FitLatency`, bracket `context_create`,
`kernel_create`, `first_alloc`, and `first_transfer` with `clock()`/`record`,
and build its `SessionState` with `session_state_from_trace(trace,
env_warmup_level())`. Disabled tracing reads no clock, so these calls cost a
call count and nothing else — that property is why they are safe on paths
that also run in steady state, and it is worth not breaking.

**(d)** Trainers should move to the full entry point. In `model.mojo` (lines
100, 193, 256, 316) and `trainset.mojo` (lines 785, 857), each
`resolve_device(device, n_rows, n_features, n_outputs)` becomes
`resolve_device_full(...)` with the objective, `max_bin`, and input flags
the caller already has in scope. The shape-only form is not wrong, but it
lets an explicit `device='gpu'` pass the policy and fail deeper in, which is
the outcome the refusal exists to prevent. `model.mojo` and `trainset.mojo`
are lane 01/12 territory; `resolve_device` stays exported and unchanged, so
this can land whenever.

### 5.4 Task 08 — `src/mojoboost/objective_registry.mojo` (optional)

No patch needed for `gpu_objectives_native.mojo`: another lane already made
`supports_device_objective` delegate to `objective_gradients_on_device`, and
this lane followed by making `device_policy` delegate to the same function
(§4). All three predicates now agree by construction.

One duplicate is left and it is lane 08's: `objective_registry.mojo` carries
its own `comptime LAMBDARANK = 7` (line 189) mirroring `ranking.mojo`
(line 93), for the same cycle-avoidance reason `device_policy` used to. That
is now the package's single cycle-avoiding copy, which is the right number,
so this is a note rather than a request — but a test pinning
`objective_registry.LAMBDARANK == ranking.LAMBDARANK` would be worth more
than the comment currently guarding it.

---

## 6. Remaining disconnections

Stated plainly. None of these was fixed and none is claimed to be.

1. **The binding does not exist yet.** Until §5.1 lands,
   `native_contract()` returns `"narrow"`, every estimator decision is
   shape-only, and `decide_device_report` has no caller outside Mojo. This
   is the top of the list.
2. **Every decision runs on `PROFILE_FALLBACK`.** Nothing opens a device and
   reports attributes, so `memory_budget_bytes` is 0, the memory gate never
   fires (correctly — it must not block on an unreported budget), and
   `WARN_UNKNOWN_HARDWARE` is on every GPU-relevant decision. §5.3(b).
3. **`unified_memory_policy` still has no allocation-site caller.** It is
   now *reported* through the device decision, which is real integration for
   the reporting half, but no buffer in `histogram_gpu.mojo` or
   `gpu_predict.mojo` asks it which route to take. Since the answer is
   `copy_staged` for every role in every context, wiring it changes no
   behavior today; it changes behavior the moment the ledger fills, which is
   the point of wiring it before then.
4. **`initialization.mojo` still records nothing.** `StartupTrace` has no
   `clock()`/`record` call anywhere in `src/`. §5.3(c). Until then
   `session_state_from_trace` always answers cold, which is conservative and
   correct but uninformative.
5. **`crossover_rules()` is empty, so `auto` is the CPU everywhere.** Not a
   disconnection — it is the designed state. No measurement in this
   repository has found a shape where GPU training beats CPU training, and
   the one end-to-end measurement (M4, `bench/bench_train_gpu.mojo`) came
   out slower. Adding a rule is a benchmarking result, not a code change.
6. **No test covers any of this.** No `test_device_policy.mojo`, no test for
   `unified_memory_policy.mojo`, none for `initialization.mojo`.
   `tests/test_device.mojo` exercises the `device.mojo` facade only. §7.

---

## 7. The test that should exist (UNWRITTEN, UNRUN)

Not written — this lane may not write tests. Specified so the next lane that
can does not have to re-derive it. `tests/parallel/test_device_policy.mojo`:

- **Mirror pinning.** `MAX_GPU_ROWS` equals `histogram_gpu.MAX_ROWS`;
  `MIN_GPU_BINS` / `MAX_GPU_BINS` equal `histogram_gpu.MAX_BINS` and the
  binner's range. These are the three mirrors left after §4, and a test is
  the only thing that can watch them, since the import that would make them
  unnecessary is the one that closes the cycle. The test must be able to see
  both sides; a test that cannot is not this test, and belongs wherever it
  can. (`LAMBDARANK` no longer needs pinning here — it is imported.)
- **`MemoryEstimate` invariants**, all five from the struct docstring, and
  invariant 4 (monotone in each of rows, features, outputs, bins) by
  property over a small grid. A shrinking estimate admits a run that does
  not fit.
- **`decide_device` against injected capabilities**, which is the whole
  reason `DeviceCapabilities` is injectable: an explicit `gpu` on
  `unavailable()` is `blocked` with `NO_DEVICE` and never `CPU_DEVICE`;
  `auto` on an available fixture with an empty rule table is
  `DECISION_AUTO_CPU_NO_EVIDENCE`; each of the nine `BLOCK_*` codes fires on
  a request built to trigger exactly it.
- **`serialize()` round trip.** Every key `device_selection.py` reads is
  present: `selected`, `blocked`, `message`, `decision`, `validated`,
  `policy_version`, `evidence_id`, `memory_estimate_complete`,
  `gpu_available`, `api`, `apple_generation`, `profile_source`,
  `memory_device_bytes`, `memory_upper_bound_bytes`, `session_context_open`,
  `session_kernels_ready`, `session_warmup_level`, `transfer_all_default`,
  `transfer_ack_unproven`, and one `transfer=` line per role. No value
  contains a newline.
- **`plan_session_routes` returns all-`copy_staged`** and
  `SessionMemoryPlan.all_default()` is True while `EvidenceLedger.installed()`
  is empty. This is the assertion that fails loudly the day somebody sets a
  rung without a record, which is the failure `RouteEvidence.audit` exists
  for.

---

## 8. Fallbacks preserved

Nothing was made stricter and no established path was removed.

- `resolve_device` (four-argument, shape-only) is unchanged and still
  exported. Every current caller keeps working.
- `device.mojo` still defines every symbol it has always defined, as
  wrappers rather than re-exports, for the reason its docstring gives.
- `_NarrowNativePolicy` in `device_selection.py` is untouched. It is the
  path a build without the new binding takes, it runs the same native
  engine, and it is confined to one class so it can be deleted whole.
- `python/mojoboost/__init__.py`'s direct `_mojoboost.resolve_device` call
  survives as the fallback for a build where `device_selection` cannot be
  imported at all.
- `NativePolicyUnavailable` still refuses to invent a Python answer when the
  extension is missing. That is deliberate and was not softened.
- `DEFAULT_ROUTE` is `copy_staged` and `ENABLE_LEVEL` is still
  `EVIDENCE_TRAINER`, the top rung. No gate was loosened to make anything
  reachable.
- Transfer routes and session state are reported, never acted on. Turning
  either into a selection input is a separate, evidenced change.

---

## 9. Serialization and public-API effects

**Model serialization: none.** No model format, no tree, no booster field
changed. Nothing added here is persisted with a model. A device is a
property of a run, not of a fitted ensemble, which is why the estimators
have `device_` and the model file does not.

**Decision wire format: additive.** `DeviceDecision.serialize()` gained
`session_context_open`, `session_kernels_ready`, `session_warmup_level`,
`transfer_requested`, `transfer_all_default`, `transfer_ack_unproven`,
`transfer_unified_memory`, and repeated `transfer=` lines. Every existing key
is unchanged, in the same place, with the same spelling.
`_parse_decision` in `device_selection.py` keeps unrecognized keys in
`native`, so an older Python against a newer native layer surfaces the new
keys in `to_dict()` rather than dropping them, and a newer Python against an
older native layer reports "not reported" rather than fabricating an answer.

**`POLICY_VERSION` was not bumped.** It versions the *rules*: it is bumped
when a crossover rule is added, removed, or retuned, or when a gate changes
what it admits. No gate changed what it admits — the two new warnings do not
select, and the new entry points call the same `decide_device`. Bumping it
for an additive report field would make the one signal that a decision's
rules changed stop meaning that.

**Public Python API: additive.** `TransferRoute` and
`DeviceReport.transfer_routes` in `device_selection`; `WarmupSummary`,
`parse_warmup`, and `StartupReport.warmup` in `diagnostics`. Both `__all__`
lists updated. `StartupReport.__init__` and `report_from_values` gained a
trailing optional `warmup` parameter; every existing positional call is
unaffected. `to_dict()` output gained `transfer_routes` and `warmup` keys.

**Mojo signature changes, all inside this lane:**

- `DeviceCapabilities.__init__` gained two required parameters, `session` and
  `transfer`, before the three defaulted limit parameters. Every construction
  site is in `device_policy.mojo`.
- `DeviceCapabilities.detect`, `from_profile`, and `unavailable` are now
  `raises` (they read `MOJOBOOST_GPU_TRANSFER`, and an unparsable value must
  raise rather than silently resolve to a different path). `detect` and
  `from_profile` gained a trailing defaulted `session` parameter.

Grepped: nothing outside `device_policy.mojo` constructs a
`DeviceCapabilities` or calls those three. If lane 01 adds such a call site,
it needs `raises`.

---

## 10. Risks

1. **Nothing was compiled.** The Mojo additions are unverified. The specific
   things to watch, in the order they are likely to bite:
   - `SessionState.cold()` as a default argument value for a `var`
     parameter. Precedent exists in this repository
     (`CategoricalSpec.none()` in `split.mojo`, `BaggingParams.disabled()`
     in `trainset.mojo`), which is why it was written that way, but it was
     not compiled.
   - Implicit copies of non-`ImplicitlyCopyable` list elements. Two sites
     use explicit `.copy()` for exactly this (`serialize`'s `transfer` loop,
     `SessionMemoryPlan.report`); if there is a third the compiler will name
     it.
   - Aliased imports (`block_reason_name as transfer_block_name` and
     friends) resolving as intended rather than shadowing.
2. **Import cycle.** `device_policy.mojo` gained three imports:
   `initialization.mojo`, `unified_memory_policy.mojo`, and
   `objective_registry.mojo`. The first two were leaves importing only the
   standard library, and the third imports only `boosting` and `metrics`, so
   the direction is safe by inspection — but only by inspection, and another
   lane adding an import to any of the three could close a cycle through
   `model.mojo` or the GPU stack. Worth re-checking before the first build.
   `objective_registry.mojo` in particular is lane 08's and was being edited
   concurrently, so `objective_is_builtin`, `objective_gradients_on_device`,
   and `LAMBDARANK` are names this lane now depends on and does not own. All
   three are described in that file as canonical, which is why they were
   chosen; if one is renamed, this import is the thing that breaks.
3. **§5.2(b) ordering is a real hazard.** Retiring the four Python raises
   before §5.1 and §5.2(a) both land removes the only enforcement those
   refusals have on the narrow contract. `device='gpu'` on a sparse matrix
   would then reach the trainer. Sequence, or leave them.
4. **`session_warmup_level` can mislead.** It reports what
   `MOJOBOOST_GPU_WARMUP` asks for. On a process that never opened a device,
   `session_kernels_ready` is False beside it, and a reader who takes the
   level as evidence that warm-up *happened* will be wrong. The prose in
   `_session_line` prints both together for this reason.
5. **Report growth.** The serialized decision is now around sixty lines.
   `select_device` produces one per fit, and nothing caches it. Fine for a
   fit; worth noticing if something ever calls it per row.
6. **The `transfer=` lines describe a plan, not an allocation.** No buffer
   consults `SessionMemoryPlan` yet (§6.3). A reader could take the plan as
   a description of what the allocator did. It is a description of what the
   allocator *would* do; the two coincide today only because both are
   `copy_staged`.
7. **The shared checkout.** §0.

---

## 11. Smallest later commands — ALL UNRUN

Not run, per this lane's constraints. Smallest useful set, in order.

Type-check the layer first — cheapest, catches everything in risk 1:

```
# UNRUN
pixi run mojo build --emit=object -o /dev/null src/mojoboost/device_policy.mojo
```

Then the one focused test, once §7 is written (one test per change, never
the full suite):

```
# UNRUN
pixi run mojo run tests/parallel/test_device_policy.mojo
```

Then the existing facade test, which is the regression guard for the
`device.mojo` signature changes:

```
# UNRUN
pixi run mojo run tests/test_device.mojo
```

After §5.1 lands, the assertion that the binding wiring actually took —
this is the one-line check that `native_contract()` flipped:

```
# UNRUN
pixi run python -c "from mojoboost.device_selection import native_contract; print(native_contract())"
# expect: full
```

And the end-to-end report, which exercises the new sections:

```
# UNRUN
pixi run python -c "
import numpy as np
from mojoboost.device_selection import explain_device_choice
print(explain_device_choice(np.zeros((1000, 20)), device='gpu', max_bin=255, objective='regression'))
"
```

Do not run the benchmark or build loops to check any of this. The questions
above are answered by a type-check and one test.
