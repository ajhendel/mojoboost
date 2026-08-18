# Resolving the nine dead switches

Date 2026-08-17. Companion to [SWITCH_GRID.md](SWITCH_GRID.md), which
catalogued 85 live `MOJOTREES_*` switches under `src/` and `bindings/` and
found nine dead. This document records what happened to each of the nine.

The rule this lane worked to. A switch that nobody can use is worse than no
switch, because it reads as a supported feature. Each of the nine therefore
ends either wired up or deleted, and a docstring never survives a deletion
that promised the behavior.

**Outcome, nine deleted and zero wired up.** Every one of the nine failed the
same test in the same way, which is set out per switch below. Nothing else
was removed, no kernel changed, no signature moved, and no behavior of any
run changed, because none of the nine changed the behavior of any run before
this either. That last point is the whole finding. Deleting them is
observationally inert and was only ever a documentation fix.

Nothing here was built, compiled, or run. This lane held no compiler. What
that leaves unverified is listed at the end.

---

## 1. `MOJOTREES_STARTUP_REPORT_FD`

**Verdict: deleted.** The docstring bullet in `src/mojotrees/initialization.mojo`
is gone, replaced by a tombstone.

**Evidence.** The audit's claim was that the name is read by nothing, and it
is stronger than that. There is no `getenv` call anywhere in the repository
that names it, so there was never a read to delete, only a promise. What
existed was one bullet inside the module docstring's "Environment contract"
list, sitting between two bullets that describe switches that genuinely work
(`MOJOTREES_STARTUP_TRACE` and `MOJOTREES_GPU_WARMUP`), which is the position
that makes a reader believe it. The bullet read "reserved, unread here. See
the handoff; emitting the report is a call-site decision, not this module's."
No handoff mentioned the name. No call site exists.

**Why not wired.** There is nothing to wire it to. `initialization.mojo`
measures phases and renders no report; it has no serializer for
`StartupTrace`, no `write(2)` through FFI, and no dependency that would give
it one. Wiring the switch means writing an emitter, choosing a wire format,
and agreeing that format with `python/mojotrees/diagnostics.py`, which
already owns the phase-name schema. That is building a diagnostic feature, not
connecting a call, and it cannot be validated without a compiler.

**What would bring it back.** An emitter, a call site, and a format agreed
with `diagnostics.py`, in that order. The tombstone in the file says the
same.

**Left behind, outside this lane's write scope.**
`python/mojotrees/diagnostics.py` still carries the name in its `WATCHED_ENV`
tuple. `compatibility/api_snapshot.json` and `compatibility/DRIFT_REPORT.md`
still list it. Those three should drop it.

---

## 2 to 8. The seven `MOJOTREES_DIST_*` switches

`MOJOTREES_DIST_JOB_ID`, `MOJOTREES_DIST_MACHINES`, `MOJOTREES_DIST_MODE`,
`MOJOTREES_DIST_RANK`, `MOJOTREES_DIST_RESTART_EPOCH`,
`MOJOTREES_DIST_TIMEOUT_S`, `MOJOTREES_DIST_WORLD_SIZE`.

**Verdict: all seven deleted, consistently.** `runtime_from_env` and its
private `_env_int` helper are gone from
`src/mojotrees/distributed_transport.mojo`, and the module's `from std.os
import getenv` import went with them, since it had no other use in that file.
A tombstone stands where the function was.

### The prior question, answered first

**Is distributed training a shipped feature of this library, or an unfinished
one? It is unfinished, and the repository says so in five places, none of
which this lane had to change.**

- `docs/distributed.md`, line 3, quoted verbatim. "Status: design plus CPU
  prototype. Not a shipped feature. The prototype runs every rank inside one process, so
  nothing here has been exercised over a network, and no distributed
  performance number is claimed anywhere in this document."
- `README.md`, in the feature list, "Every rank runs in one process and
  nothing has run over a network, so no distributed performance is claimed
  and this is a prototype rather than a feature."
- `docs/LIGHTGBM_PARITY.md`, three separate rows. The distributed transport
  row reads "partial" and ends "no process has connected to another." The
  machine-list parameter row (`num_machines` / `local_listen_port` /
  `time_out` / `machine_list_filename` / `machines`) reads "deferred", "No
  transport is wired up." The transport row adds "**No distributed
  performance or scaling claim is made anywhere**."
- `python/mojotrees/dask.py`, module docstring, "Today every published build
  is the second kind ... `DaskMojoTreesRegressor(...).fit(...)` raises
  `DistributedNotAvailable` before it touches the cluster, naming what is
  absent." And "Nothing in this module ever trains, in any state."
- The code's own predicate. `bindings/distributed_bindings.mojo` reports
  `multi_process` from `transport_validated()`, which is False, with the
  reason string "multi-process distributed training is not validated in this
  build ... no two processes have trained together yet."

Three further facts this lane checked directly, none of which is written down
elsewhere in that form.

- **No CI job runs a distributed fit.** The six workflows under
  `.github/workflows/` (`ci.yml`, `contributor-access.yml`,
  `gpu-validation.yml`, `release-linux.yml`, `release-macos.yml`,
  `release-provenance.yml`) contain no occurrence of the string
  "distributed" and no `MOJOTREES_DIST` name.
- **No test exercises a distributed fit through a runtime spec.**
  `tests/test_distributed_transport.mojo` imports 50 names from the transport
  module and none of them is `RuntimeSpec`, `local_runtime`,
  `transport_runtime`, or `runtime_from_env`. It drives the protocol over
  `MemoryEndpoint`, which the module's own header calls a fake.
- **`src/mojotrees/distributed_transport.mojo` is not exported from
  `src/mojotrees/__init__.mojo` at all.** The package exports
  `train_distributed`, `train_distributed_run`, `grow_tree_distributed` and
  the strategy vocabulary; it exports no transport symbol, so no
  `RuntimeSpec` reaches a user of the package.

So what does ship is single-process distributed training over
`LocalCollective`, reached from `bindings/distributed_bindings.mojo`, which
builds `LocalCollective(world_size)` from an explicit integer argument and
calls `train_distributed_run`. That path never touched these seven variables
and does not need them.

### Why deletion rather than wiring

The consumer that would take a `RuntimeSpec` out of `runtime_from_env` is
`distributed.run_distributed`, described in its own docstring as "the one
entry point for bindings and Dask". **`run_distributed` has no caller either,
and is not exported from `__init__.mojo`.** So the surface is dead two levels
deep, and wiring the switches means supplying the missing level as well as the
missing call.

Beyond that, the variables only mean anything to a process that some launcher
started with them set, and this repository ships no launcher. In local mode
the only variable that does work is `MOJOTREES_DIST_WORLD_SIZE`, which would
duplicate an argument the bindings already take explicitly, and taking a world
size from the environment instead of from an argument is a downgrade. In
transport mode every path ends at socket code whose own header says
"Implemented and never run", and `transport_validated()` returns False.
Wiring the seven would therefore have meant shipping a configuration surface
for a run that has never happened, which is the defect being fixed rather than
a fix for it.

Consistency was a requirement and is satisfied. All seven were read in one
function; the function is gone; none survives.

### What was deliberately kept

The deletion is narrow on purpose. `RuntimeSpec`, `local_runtime`,
`transport_runtime`, `parse_runtime_mode`, `runtime_mode_name`,
`require_transport`, `open_local_collective`, `open_transport_collective`,
`open_socket_collective`, `connect_world`, `SocketEndpoint` and the whole wire
protocol are untouched. They take configuration as arguments and promise
nothing about the environment. The transport suite still imports the same 50
names and none of them moved.

One consequence worth naming rather than hiding. `local_runtime` and
`transport_runtime` were called only by `runtime_from_env`, so after this
change they have no in-repository caller. They are the documented
explicit-argument builders for the type `run_distributed` takes, they are
described in `docs/DISTRIBUTED_TRANSPORT.md`, and deleting them would be
deleting the transport's construction API rather than a dead switch. That is
a different decision and it belongs to whoever owns the distributed roadmap.
The same applies to `parse_runtime_mode`, whose last caller was the deleted
function.

**What would bring the seven back.** A launcher or a shipping
`run_distributed` call site; `transport_validated()` returning True, which
needs the two-process procedure in `docs/DISTRIBUTED_TRANSPORT.md` section 7
to have been run; and a test that sets the variables and asserts the spec
that comes out. There was no such test, which is how all seven stayed dead
without anyone noticing.

**Left behind, outside this lane's write scope.**
`compatibility/api_snapshot.json` lists all seven in three places and
`compatibility/DRIFT_REPORT.md` lists them in its distributed row. Both should
drop them. `bench/results/INSTRUCTION_AUDIT.md` sections 9g and its
recommendation table refer to them as live; that file is a dated audit and
probably wants the same snapshot banner this lane put on `SWITCH_GRID.md`.

---

## 9. `MOJOTREES_GPU_GRAD_LAYOUT`

**Verdict: deleted.** `env_grad_layout` is gone from
`src/mojotrees/gpu_gradient_stream.mojo`, together with that module's `from
std.os import getenv`, which had no other use there. A tombstone stands in its
place.

**Evidence.** `env_grad_layout` returned `LAYOUT_INTERLEAVED` for the exact
text `interleaved` and `LAYOUT_SPLIT` for anything else. A repository-wide
grep for `env_grad_layout` returns two hits, its own definition and the audit
row that reported it. `stream_layout_name`, the only other consumer of the two
constants, has no caller either, so the value the function computed was never
compared against anything. Setting the variable changed nothing observable
about any run.

**Why not wired, and this one is a hard blocker rather than a judgment call.**
The layout only matters at the moment a histogram kernel loads the two
derivatives. Selecting it means branching between `builder.enqueue_leaf` and
`enqueue_leaf_interleaved` at the trainer's leaf-enqueue site, which lives in
`src/mojotrees/histogram_gpu.mojo`, `src/mojotrees/gpu_active_rows.mojo` and
`src/mojotrees/train_gpu.mojo`. All three belong to other lanes and are
outside this lane's write scope. A variable read inside
`gpu_gradient_stream.mojo` cannot steer a kernel launched from a module that
never asks it, which is exactly why the switch was inert. The additive design
that kept this path out of the shipped kernels also put the branch point out
of its own reach.

The interleaved path is real and none of it was deleted. `_pack_gh_kernel`,
`_range_hist_partial_gh_kernel`, `_range_hist_atomic_gh_kernel`,
`enqueue_range_histogram_interleaved`, `InterleavedGradients` and
`enqueue_leaf_interleaved` all stand unchanged. What was missing was never the
kernels. It was a caller, and an environment variable is not one.

It is also unmeasured. The module banner argues that interleaving halves the
memory transaction count for the gather below the root, which is arithmetic
and is sound, and the same banner says what a measurement is still owed for,
namely how much wall clock that buys. `bench/apple/fused_round_plan.json`
records the intended comparison and names
`MOJOTREES_GPU_GRAD_LAYOUT=interleaved` as how it would be switched. That plan
entry is now stale; the comparison needs a layout parameter threaded from the
trainer, not this variable.

**What would bring it back, and the recommendation is that it should not.**
When the branch lands in the trainer, the layout should be a parameter carried
in the trainer's own configuration, decided once, and not re-read from the
environment at two sites that can disagree. `SWITCH_GRID.md` section 5 already
documents what that failure looks like, in the shadowed
`MOJOTREES_GPU_TREE_RESIDENT` pair, where two predicates over one variable
disagree about its default.

---

## Edits this lane could not make

Listed for whoever owns the files.

1. `python/mojotrees/diagnostics.py`, the `WATCHED_ENV` tuple. Drop
   `"MOJOTREES_STARTUP_REPORT_FD"`. The tuple's own comment says it lists
   variables "that change what a startup measurement means", and this one
   never could.
2. `compatibility/api_snapshot.json`. Drop `MOJOTREES_STARTUP_REPORT_FD`
   (2 occurrences), `MOJOTREES_GPU_GRAD_LAYOUT` (3), and the seven
   `MOJOTREES_DIST_*` names (16 across three blocks). If this file is
   generated by a scan, regenerating it after this change is enough.
3. `compatibility/DRIFT_REPORT.md`, the distributed row and the GPU tuning
   row and the diagnostics row. Same three sets of names.
4. `bench/apple/fused_round_plan.json`, two entries that name
   `MOJOTREES_GPU_GRAD_LAYOUT` as the way to switch layouts. The mechanism is
   now a call-site choice.
5. `bench/results/INSTRUCTION_AUDIT.md` sections 9g and 431 and the
   recommendation table. Dated audit; wants a resolved banner rather than a
   rewrite.

No change is needed in any file owned by the GPU or CPU perf lanes. Nothing
this lane did requires an edit in `train_gpu.mojo`, `gpu_resident_round.mojo`,
`gpu_tree_tables.mojo`, `device_policy.mojo`, `histogram.mojo`, `tree.mojo`,
`histogram_gpu.mojo`, `gpu_leaf_batching.mojo`, `gpu_split_search.mojo` or
`gpu_active_rows.mojo`.

---

## What is unverified, ranked

No compiler was available to this lane. In descending order of risk.

1. **The two unused-import removals.** `from std.os import getenv` was dropped
   from both `distributed_transport.mojo` and `gpu_gradient_stream.mojo` after
   grepping each file for every remaining `getenv` occurrence and finding
   none. If either file reaches `getenv` through a name this lane did not grep
   for, the build breaks with an unresolved reference at that site. This is
   the most likely place for an error and the cheapest to fix.
2. **Dangling references to the three deleted functions.** Repository-wide
   greps for `runtime_from_env`, `_env_int` inside
   `distributed_transport.mojo`, and `env_grad_layout` were run before and
   after the edits and return no code hits, only prose. A reference reached
   through a re-export or a trait would not appear in those greps.
   `distributed_transport` is not exported from `src/mojotrees/__init__.mojo`,
   which closes the re-export path for the transport pair; the grad-layout
   function was module-private in practice and never exported.
3. **`parse_runtime_mode` and `stream_layout_name` now have no caller.** Both
   were left in place deliberately. If this toolchain errors rather than warns
   on an uncalled module-level `def`, both would need deleting. Nothing else
   in this repository suggests it does; there are other uncalled public
   helpers, `local_runtime` among them now.
4. **Docstring-only edits.** The changes to `initialization.mojo`, to the
   `RuntimeSpec` docstring, and to the `gpu_gradient_stream.mojo` module
   header are all inside triple-quoted strings or comments and cannot change
   behavior, but a stray quote character would break parsing. The added text
   contains no triple quotes and no backslashes.
5. **Markdown only, no risk.** `docs/DEVICE_SELECTION.md`,
   `docs/design/SWITCH_GRID.md`, and this file.
