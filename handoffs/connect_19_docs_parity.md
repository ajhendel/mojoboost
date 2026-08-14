# Connect 19: documentation and parity claims against actual reachability

Lane 19. Owned files, and the only ones this lane wrote:

- `README.md`
- `python/README.md`
- `docs/LIGHTGBM_PARITY.md` (contract version 3)
- `docs/CAPABILITY_LEVELS.md`
- `docs/ARCHITECTURE.md`
- `docs/INTEGRATION_INVENTORY.md`
- `tools/check_parity.py`
- `tools/audit_integration.py`
- `handoffs/connect_19_docs_parity.md` (this file)

Nothing else was edited, staged, reverted, cleaned, reformatted, committed,
or pushed. Every change another file needs is a patch request in §8, quoted
precisely enough that its owner can apply it without re-deriving it.

**Snapshot: derived at `860b1cf`, re-derived at `63aad82`, re-verified at
`9c1e771` and again at `29d76e4`, all on 2026-08-14.** Those are four
different trees, and a stamp is a reading rather than a promise: the closure
held across the last three, and the next commit is another lane's. Between the
first and the second, 72 files under `src/mojoboost`, `bindings`, `capi`,
and `python/mojoboost` changed by 35,143 insertions, and the change closed
the single largest disconnection this lane had documented. That is the
central fact about this lane and it is why §9 is as long as it is.

---

## 0. The constraint this lane worked under

Static inspection only: `rg`, `sed`, `awk`, `wc`, `git status`, `git diff`,
`git diff --check`. No Mojo, no Python, no pixi, no build, no test, no
formatter. **Both owned tools are UNRUN**, and so is the graph engine they
sit on. Every count, every cell, and every path in the documents was
gathered by hand and cross-checked against the source; the tools' *output*
is a prediction until someone runs §10.

Nothing here claims correctness, performance, parity, packaging, or hardware
validation. Where a document says a capability runs, it says which line
calls it.

---

## 1. Implementations found, and the fusion decision

Four things in this repository answer some version of "is this claim true":

| Where | What it owns | Verdict |
|---|---|---|
| `tools/connectivity_audit.py` | the import graph from five roots, and the `CLASSIFICATION` judgment table | **authoritative.** Not owned by this lane |
| `tools/audit_integration.py` | `docs/INTEGRATION_INVENTORY.md` against that graph | kept, layered on the above |
| `tools/check_parity.py` | `docs/LIGHTGBM_PARITY.md` and `docs/CAPABILITY_LEVELS.md` | kept, owns the claims |
| the four documents | the prose a user reads | rewritten against the tree |

The fusion decision, taken once and held: **`tools/connectivity_audit.py`
is the only import graph in this repository, and this lane did not write a
second one.** `tools/audit_integration.py` imports it and asks it questions;
it computes no edges of its own. Two import graphs would be exactly the
duplication all three scripts exist to find. No new module, registry,
policy engine, trainer, or model representation was created by this lane.

The vocabulary was fused the same way. `docs/CAPABILITY_LEVELS.md` defines
seven levels (implemented, integrated, publicly reachable, focused-tested,
differential-tested, hardware-validated, release-packaged) and five status
words, once. `docs/LIGHTGBM_PARITY.md` section 0 scores rows against them,
`docs/INTEGRATION_INVENTORY.md` carries the evidence behind every `no` in
the two graph columns, `docs/ARCHITECTURE.md` is the map, and
`README.md` cites the levels rather than restating them.

---

## 2. The evidence path, before and after

This lane connects claims, not code, so the path that matters runs from a
sentence a user reads to a line in the tree.

**Before.** A status word in the parity contract was supported by prose in
the same file. Nothing checked it against the tree. Three rows had rotted in
the direction that costs a user the most: `cv`, `inspection`, and
`device_selection` were all in `mojoboost.__all__` and all three tables said
they were out of reach. Two more claimed a fallback was the only path after
the binding behind it had landed.

**After.**

```
docs/LIGHTGBM_PARITY.md  §0 cell
      -> tools/check_parity.py           (status/level consistency,
                                          PUBLIC_REACHABILITY_PROBES,
                                          STALE_DEFERRED_WATCHES,
                                          cited paths exist)
      -> python/mojoboost/__init__.py __all__, class methods, arguments
      -> src/mojoboost/__init__.mojo re-exports
      -> MOJOBOOST_* literals in shipping source          [new this pass]

docs/INTEGRATION_INVENTORY.md table row
      -> tools/audit_integration.py      (every row gated)
      -> tools/connectivity_audit.py     (the one graph)
      -> the import closure from the five roots
```

Neither direction imports or builds the package. `check_parity.py` parses
Python with `ast` and Mojo with regexes; `audit_integration.py` asks
`connectivity_audit` and never re-derives.

---

## 3. Connections completed

**a. A reachability re-derivation, three times.** The closure was computed
from four native roots (`bindings/_mojoboost.mojo`,
`src/mojoboost/__init__.mojo`, `capi/mojoboost_capi.mojo`,
`cli/mojoboost_cli.mojo`) over 516 relative-import edges. The closure holds
78 module names; of the 90 files in `src/mojoboost`, 69 are reached and 21
are orphans, and the orphans are listed by name rather than counted. At
`9c1e771` and again at `29d76e4` the only new edges since `63aad82` are
internal to already reachable modules or point into modules that were
already orphans, so the closure is unchanged and both documents say so in
their stamp.

**b. Section 0 of the parity contract now scores 27 contested capabilities**
against the seven levels, and the levels are checked rather than asserted.

**c. Five rows corrected in the direction that matters.** `cv`,
`inspection`, and `device_selection` were publicly reachable and marked
otherwise. Model inspection and explainable device selection were described
as blocked by unregistered bindings that another lane had since registered.

**d. One row promoted on evidence.** Class-batched GPU multiclass rounds
moved `deferred -> partial`, `integrated: no -> yes`, `publicly reachable:
no -> yes`, `hardware-validated: n/a -> no`, on this call chain:

```
src/mojoboost/train_gpu.mojo:1703   builder.class_schedule(...)
src/mojoboost/train_gpu.mojo:1707   if not schedule.is_sequential():
                                        _train_multiclass_gpu_batched(...)
src/mojoboost/train_gpu.mojo:1579   GpuClassBatch.for_plan(...)
src/mojoboost/gpu_output_planes.mojo:514  MOJOBOOST_GPU_CLASS_BATCH
```

The default schedule is sequential, so no default changed. The env var is
public under section 2 of `docs/COMPATIBILITY_POLICY.md`, which is the whole
of the reach, and the row says exactly that.

**e. Two rows held at `deferred` and given honest evidence.** GPU packed-bin
layout is now *reached* (`histogram_gpu.mojo` imports `check_layout_support`
and calls it when the builder opens) and still not *planned*: nothing builds
a `BinLayoutPlan` or calls `pack_binned_matrix`. Hybrid leaf placement is
reached from `GpuSession.note_hybrid`, which builds a `HybridContext` and
reports; the module's own comment is "Nothing here moves a histogram." An
import is not a call and a report is not a placement, and neither row was
promoted.

**f. `env:` probes.** `tools/check_parity.py` grew one probe kind,
`env:MOJOBOOST_NAME`, resolved by `env_var_names()` scanning string literals
under `src/mojoboost`, `bindings`, and `python/mojoboost`. It exists because
a capability whose only route is an environment variable is publicly
reachable, and check 9 previously had no way to hold that claim to anything.
The first watch using it is the class-batch row from (d).

**g. Prose reconnected to the tree.** `docs/ARCHITECTURE.md` no longer says
every `getattr` fallback is a live disconnection (most now find their hook,
and the distinction is which extension you built). `python/README.md` no
longer says split gains are never serialized (format v4 writes them) and no
longer says `report.contract` reads `"narrow"` (a current build registers
`decide_device` and reports `"full"`).

**h. The inventory's tables were rebuilt** against `63aad82`: 21 orphan
rows, the binding-registration table down from five rows to one, the
unbound-hook table down to `split_gains` alone, three of four duplicate
policies rewritten, and two rows added to "Reachable, but with no default
effect" for `gpu_multiclass_batch` and `hybrid_leaf_scheduler`.

---

## 4. Duplicates fused or quarantined

- **Fused:** one import graph (`connectivity_audit.py`), one judgment table
  (`CLASSIFICATION`), one inventory gate, one claims checker. This lane
  added none of these and duplicated none of them.
- **Fused:** one capability vocabulary, in `docs/CAPABILITY_LEVELS.md`.
  `check_parity.py` fails if the contract's level table uses any other
  names, so the two files cannot drift into two vocabularies.
- **Quarantined, not deleted:** the four "policy that exists twice" rows in
  `docs/INTEGRATION_INVENTORY.md` (device choice, objective/metric naming,
  model dump, class weights). Three of them now have a native side that is
  bound, so the Python side is a compatibility path rather than a second
  policy; the fourth (`class_weight`) has no seam open at all. Deleting any
  of them is another lane's edit, and §8 asks for it.
- **Documented as stale, not edited:** `CLASSIFICATION` in
  `tools/connectivity_audit.py` still carries entries for seven modules that
  stopped being orphans. They are judgments about non-findings. Patch
  request §8.3.

---

## 5. Remaining disconnections

Full detail is in `docs/INTEGRATION_INVENTORY.md`; this is the summary a
reader of this handoff needs.

1. **21 orphan native modules.** `alternate_boosting`, `backend`,
   `boosting_dart`, `boosting_rf`, `cegb`, `distributed_gpu`,
   `distributed_strategies`, `external_memory`, `gpu_amd_policy`,
   `gpu_categorical`, `gpu_cuda_policy`, `gpu_levelwise`, `gpu_sparse`,
   `gpu_sparse_layout`, `levelwise_policy`, `lgbm_model_io`, `linear_tree`,
   `model_editing`, `ranking_advanced`, `sequence`, `validation`. Several
   are chains, so the count overstates the number of decisions.
2. **`split_gains` / `split_gains_multiclass`.** The last hook
   `python/mojoboost/inspection.py` probes for that exists on neither side
   of the seam. It is what keeps model inspection `partial`.
3. **`_Base._resolve_device`** calls `_mojoboost.resolve_device` directly
   while `explain_device_choice` goes through `decide_device`, so a report
   and a `fit` are two decisions into one engine.
4. **`src/mojoboost/class_weight.mojo` has no caller.** The estimators
   compute class weights in Python. Two implementations, no test that would
   notice them disagreeing.
5. **Parameters that parse and do nothing.** `linear_tree` and the `cegb_*`
   controls are accepted by `src/mojoboost/params.mojo` and reach a config
   field nothing reads.
6. **Reached but not exercised:** `gpu_binned_layout` (one guard),
   `gpu_bin_packing` (nothing), `hybrid_leaf_scheduler` and
   `histogram_cache_policy` (a report).

---

## 6. Fallbacks preserved

Every degraded path in `python/mojoboost` was left exactly as it is, and
this was deliberate. `getattr(_mojoboost, name, None)` with a slower pure
Python path is what lets the package work against an extension built before
its binding landed, and the binding for the dump, the registry, and the
device report landed *during* this lane. Deleting a fallback the day its
binding lands breaks every user who has not rebuilt.

What changed is only the claim: `_eval.registry_source()` still returns
`"native"` or `"compat"`, `DeviceReport.contract` still returns `"full"` or
`"narrow"`, and the documents now say that a *current build* takes the
native route rather than saying the fallback is the path. The conservative
default behaviors this lane documented are also unchanged and were
re-verified: `AUTO_MIN_CELLS = CROSSOVER_DISABLED` with an empty
`crossover_rules()`, `MOJOBOOST_GPU_SPLIT_STRATEGY` resolving to the host
scan, `MOJOBOOST_GPU_HIST_SPECIALIZATION` resolving to baseline, and a
sequential class schedule.

---

## 7. Serialization and public API effects

**None.** This lane changed no serialization format, no public symbol, no
parameter name, no env var, and no C ABI declaration. `tools/check_parity.py`
gained a probe kind and one watch entry; it is a checker, is not imported by
the package, and is not in any pixi task this lane could add (§8.1).

The documents now *describe* two public-surface facts more accurately, which
is a claim change rather than an API change: model format v4 serializes
per-node split gains, so `has_split_gain` is `True` for models this version
writes and `False` for older files; and `MOJOBOOST_GPU_CLASS_BATCH` is
named as the public route to a capability the contract previously called
unreachable.

---

## 8. Cross-lane patch requests

This lane owns none of these files and edited none of them.

**8.1 `pixi.toml`** — add the audit as a task, next to `check-parity`:

```toml
audit-integration = "python3 tools/audit_integration.py"
```

Not with `--strict`. GAP severities are expected while the connect round is
in flight; ERROR (the inventory stating something false) already exits 1.

**8.2 `.github/workflows/ci.yml`** — optionally add one step to the existing
`parity` job. It is stdlib-only and builds nothing:

```yaml
      - name: Integration inventory
        run: python3 tools/audit_integration.py
```

**8.3 `tools/connectivity_audit.py`** — `CLASSIFICATION` maintenance, for
that file's owner:

- *Remove* the seven entries that are no longer findings, because their
  modules are now reached: `gpu_binned_layout`, `gpu_bin_packing`,
  `gpu_portability`, `gpu_backend_policy`, `gpu_multiclass_batch`,
  `hybrid_leaf_scheduler`, `histogram_cache_policy`.
- *Add* entries for the eleven orphans that have none, so they stop
  rendering as `PENDING | unassigned`: `cegb`, `distributed_gpu`,
  `distributed_strategies`, `external_memory`, `gpu_amd_policy`,
  `gpu_cuda_policy`, `linear_tree`, `model_editing`, `ranking_advanced`,
  `sequence`, `validation`.

**8.4 `bindings/`** — implement and register `split_gains` and
`split_gains_multiclass`. This is the last unbound hook, and the only reason
model inspection is `partial` rather than `supported`. When it lands, the
"Native names Python reaches for that no binding registers" table in
`docs/INTEGRATION_INVENTORY.md` becomes empty, which is its healthy state.

**8.5 `python/mojoboost/__init__.py`** — route `_Base._resolve_device`
through `device_selection`'s full `decide_device` contract, now that
`bindings/_mojoboost.mojo` registers it, so a report and a `fit` are one
decision. Keep the `resolve_device` path as the fallback for an older
extension; `DeviceReport.contract` already reports which answered.

**8.6 `src/mojoboost/`** — decide `class_weight.mojo`: give it a caller, or
state that the Python arithmetic is authoritative and quarantine the module
with a comment saying so. Either resolves the row; leaving it is what does
not.

**8.7 `tests/parallel/`** — `src/mojoboost/device_policy.mojo`'s docstring
says its mirrored constants are "Pinned by"
`tests/parallel/test_device_policy.mojo`, and that file does not exist. Add
the suite or correct the docstring.

**Already done by another lane, recorded so nobody does it twice:** the
auxiliary binding modules are registered. `bindings/_mojoboost.mojo` imports
`basic_bindings`, `dataset_bindings`, `distributed_bindings`,
`inspection_bindings`, and `objective_bindings`, and registers `dump_model`,
`dump_model_multiclass`, `split_values`, `dump_leaf_index`,
`dump_raw_scores`, `objective_code`, `decide_device`, and six `registry_*`
entries. This was the highest-leverage request this lane had queued.

---

## 9. Risks

**The tree moves faster than a document can.** This is the real risk and it
already materialized: between `860b1cf` and `63aad82` three whole tables in
`docs/INTEGRATION_INVENTORY.md` asserted disconnections that no longer
existed, and two parity rows cited unregistered bindings that had been
registered. A document that overstates a disconnection is a document that
tells a user a working feature is broken. Three mitigations are in place:
every owned document is stamped with the exact commit it was read at, the
inventory says in its own text that the tool is the authority, and
`audit_integration.py` keeps GAP non-fatal so a moving tree does not turn CI
red for being ahead of the prose.

**Both tools are UNRUN.** Their regexes were exercised by hand; their
control flow was not. A traceback on first run is expected work, not a
finding. Where a tool's output disagrees with a document, the tool is right,
because it reads today's tree.

**`env:` probes are weaker than symbol probes.** `env_var_names()` proves
that a shipping file reads the variable, not that the code path behind it
still dispatches. If a lane deletes the `_train_multiclass_gpu_batched`
dispatch and leaves the variable parsed, the probe still resolves and the
class-batch row keeps a `yes` it no longer deserves. The evidence cell names
the call site so a human re-audit has somewhere to look; nothing automatic
catches that case.

**Judgment calls a reviewer should check.** Two rows were held at `deferred`
while their modules became reachable (packed-bin layout, hybrid leaf
placement), on the reading that the *capability* is not called even though
the module is imported. One row was promoted to `partial` on an env-gated
call site. Both readings follow `docs/CAPABILITY_LEVELS.md` as written; both
are the kind of call another reader could make differently, and the evidence
cells are written so that disagreeing requires only reading them.

**Other lanes committed this lane's owned files.** `README.md`,
`python/README.md`, and all six other owned files were swept into commits
`e6f3959`, `5085097`, and `e28a24d` by concurrent lanes. The content was
verified intact after each. This lane committed nothing.

---

## 10. The smallest later commands, all UNRUN

Static, stdlib-only, and none of them build or import the package:

```
python3 tools/check_parity.py                    # UNRUN
python3 tools/audit_integration.py               # UNRUN
python3 tools/audit_integration.py --table       # UNRUN
python3 tools/connectivity_audit.py              # UNRUN
```

Run them in that order. `check_parity.py` fails first if a status word, a
level cell, or a cited path is wrong; `audit_integration.py` fails next if
the inventory disagrees with the graph; `connectivity_audit.py` is the graph
itself and answers what the other two argued about.

Expected on a first run, and not findings: a traceback from either owned
tool, GAP severities from `audit_integration.py` for tree state the
inventory has not caught up with, and `PENDING | unassigned` classifications
for the eleven orphans in §8.3.

Nothing in this lane was committed.
