# Connect 22: static connection audit and the cross-lane patch queue

Lane 22. Owned files, and the only three this lane wrote:

- `tools/connectivity_audit.py` (new, 1,549 lines)
- `docs/CONNECTION_AUDIT.md` (new)
- `handoffs/connect_22_audit.md` (this file)

Nothing else was edited, staged, reverted, reformatted, or committed. Every
change another file needs is a patch request in §7, addressed to its owner
and quoted precisely enough to apply without re-deriving it.

**Snapshot: commit `860b1cf`, 2026-08-14.** The tree moved substantially
during this lane: at the start of the audit `bindings/` held one Mojo file
and nine native modules had no importer at all; by the end `bindings/` held
seven and those nine had importers. Every count here is a reading at
`860b1cf`. That volatility is the argument for the script.

---

## 0. The constraint this lane worked under, stated first

The prompt was explicit: **author the audit, do not execute it.** So
`tools/connectivity_audit.py` has never been run. Not once, not on a subset,
not with `--section`. It has been read, and its patterns were developed by
running the equivalent `rg` and `sed` queries by hand against the working
tree, but the script as a program is UNRUN.

Everything in `docs/CONNECTION_AUDIT.md` and in this handoff was gathered by
hand with `rg`, `sed`, `wc`, and `git status`. The findings are therefore
real observations about the source. The *script's* output is a prediction,
and §11 lists the one command that turns it into an observation.

Two consequences worth naming:

- The script may fail on first run. Its regexes were exercised, its control
  flow was not. Treat a traceback as expected work, not as a finding.
- Where the script's output ever disagrees with `docs/CONNECTION_AUDIT.md`,
  the script is right, because it is looking at that day's tree and the
  document is looking at `860b1cf`.

---

## 1. Implementations found

An inventory of what already existed before this lane, because the brief
was to fuse and connect rather than to add.

**Auditing that already exists, and that this lane did not duplicate:**

| Tool | Owner | What it checks | Overlap with this lane |
| --- | --- | --- | --- |
| `tools/check_parity.py` (1,139 lines) | 19 | the LightGBM parity contract: level definitions, evidence columns, claim schema, that every suite offered as evidence is also run | **Authoritative. Not re-implemented.** |
| `tools/audit_python_compat.py` | 18 | version-gated syntax and stdlib calls under `python/mojoboost/` | none; different question |
| `tools/inspect_startup_artifacts.py` | 15 | startup artifacts | none |
| `tests/parallel/api_snapshot_manifest.json` | 19 | declared native exports per module | read as input, never rewritten |

The single most important design decision in this lane was **not** to grow a
second parity checker. `check_parity.py` owns `docs/LIGHTGBM_PARITY.md` and
`docs/CAPABILITY_LEVELS.md` and runs as its own CI job. `connectivity_audit.py`
asks exactly one parity question - do the files named as evidence exist as
files - and defers everything about what evidence *means*. Section 8 of the
audit document complains about duplicate registries; shipping a duplicate
parity engine in the same commit would have been absurd.

`tools/api_snapshot.py` is specified by `docs/COMPATIBILITY_POLICY.md:770`
and does not exist, while `tests/parallel/api_snapshot_manifest.json` does
and nothing reads it. That pair is reported, not fixed: it is lane 19's.

---

## 2. Call path, before

There was no static reachability check of any kind. The five entry points
existed and nothing measured what they reached:

```
python/mojoboost/__init__.py  ->  _mojoboost.so  ->  ?
bindings/_mojoboost.mojo      ->  src/mojoboost/*  ->  ?
src/mojoboost/__init__.mojo   ->  32 re-exported modules  ->  ?
capi/mojoboost_capi.mojo      ->  14 @export fns  ->  ?
cli/mojoboost_cli.mojo        ->  5 commands  ->  ?
```

Consequence, measured: 77 native modules, and at the start of this lane
**16 of them had no importer anywhere in `src/`**, several with complete
test suites of their own. The question "is this reachable" had no answer
short of reading every import in the repository, which is why it went
unasked for as long as it did.

---

## 3. Call path, after

Still five entry points. The addition is a way to ask:

```
tools/connectivity_audit.py
  roots := { python/mojoboost/__init__.py,
             bindings/_mojoboost.mojo,
             src/mojoboost/__init__.mojo,
             capi/mojoboost_capi.mojo,
             cli/mojoboost_cli.mojo }
  |
  +-- native import graph      -> orphan modules and orphan clusters
  +-- python import graph      -> unreachable package modules
  +-- def_function table       -> exported and uncalled / called and missing
  +-- bindings/ directory      -> sibling modules the entry point never imports
  +-- serialize.mojo tokens    -> written-not-read, read-not-written
  +-- struct fields            -> model state serialization drops
  +-- top-level declarations   -> names defined by two modules
  +-- estimator __init__        -> parameters with no downstream consumer
  +-- capi header vs @export   -> C ABI drift, both directions
  +-- docs and pixi.toml paths -> references to files that do not exist
  |
  +-> CLASSIFICATION -> {DEAD | EXPERIMENTAL | PENDING} + owning lane
```

Eleven sections, `--section` to run one, `--json` to consume it, `--fail-on`
to gate on it. The default is report-and-succeed: this is a map, not a gate,
until someone decides to gate on it.

The classification table is the only opinion in the file. Everything above
it is mechanical, and an unclassified finding defaults to `PENDING`, which
is the conservative answer: it lands in the queue and a human decides.

---

## 4. Connections completed

None, and that is the correct outcome for this lane rather than a shortfall.

Lane 22 owns no implementation, no bindings, no public API, no tests, no
packaging, and no workflows. Every connection this audit identified is an
edit in a file another lane owns exclusively. Making any of them here would
have been the exact cross-lane overwrite the brief forbids, and would have
raced the four lanes that were actively writing to those files during this
session.

What was completed instead: the queue in §7, with an owner and an exact edit
for every unresolved finding.

---

## 5. Duplicates fused or quarantined

Nothing owned by this lane duplicated anything, so nothing was fused. What
the audit *found* about duplication:

**Already fused, correctly, by other lanes:**

- `device.mojo` / `device_policy.mojo`. Five names are defined in both, and
  it is not a duplicate: `device.mojo` is a documented facade that forwards
  every call, and its module docstring says so in the first paragraph. This
  is the resolved shape of the problem. The script's `duplicate-registries`
  section will keep flagging the name collision, correctly, and the answer
  each time is "read the docstring".
- `objective_registry.mojo` is now imported by `objective.mojo`,
  `custom_metric.mojo`, `gpu_objectives_native.mojo`, `lgbm_model_io.mojo`,
  and `objective_bindings.mojo`. Lane 05 records in its own handoff that the
  three device-objective predicates now agree by construction.

**Not fused, and reported:**

- **Three objective and metric tables.** `objective_registry.mojo` is the
  authority. `params.mojo` still carries its own `objective_from_name` and
  `objective_display_name`. `python/mojoboost/_eval.py` still carries a
  metric-code table whose own comment calls it "the mirror of the table in
  bindings/_mojoboost.mojo" - mirrored by hand, in another language, because
  `registry_metrics` is unreachable. Owners: 08 and 07.
- **`objective_registry.LAMBDARANK` vs `ranking.LAMBDARANK`.** One deliberate
  cycle-avoiding copy, guarded by a comment rather than a test. Lane 05's
  handoff §5.4 already flags it; this audit concurs, and notes the general
  form: a duplicated constant guarded by a comment is a duplicate that will
  drift on a schedule nobody controls.

**Quarantined:** nothing. This lane's files are new, so there was no obsolete
glue to remove or wall off.

---

## 6. Remaining disconnections

The full list is `docs/CONNECTION_AUDIT.md`. The six that matter, in the
order a reader should care about them:

### 6.1 Five binding modules that do not exist at runtime

`bindings/` holds seven Mojo files. `_mojoboost.mojo` imports none of the
other six. The 45 public functions in `basic_bindings.mojo`,
`dataset_bindings.mojo`, `distributed_bindings.mojo`,
`inspection_bindings.mojo`, and `objective_bindings.mojo` are attributes of
nothing, and `bindings/build.sh` compiles only the entry point with `-I src`,
so `objective_bindings.mojo`'s `from binding_support import py_dict` could
not resolve even if the entry point did import it.

Four separate Python fallbacks are running right now for this one reason:

| Python asks for | Implemented in | Currently falls back to |
| --- | --- | --- |
| `dump_model`, `split_values*`, `dump_raw_scores*`, `dump_leaf_index*` | `inspection_bindings.mojo` | ~580 lines of Python that re-parse the saved text model |
| `objective_code` | `inspection_bindings.mojo` | `None`, then a Python-side guess |
| `registry_metrics` | `objective_bindings.mojo` | the hand-maintained mirror table above |
| `decide_device` | `basic_bindings.mojo` | `device_selection.py`'s `"narrow"` contract |

`python/mojoboost/inspection.py` carries a `# DELETION POINT` banner naming
that exact set of functions and stating that everything below it exists only
until they appear. They have appeared. They are still not in the table.

### 6.2 A registered GPU surface with no Python caller

The mirror failure. `_mojoboost.mojo` exports `gpu_predict_capability`, eight
`gpu_validation_*` entries, and four `predict_*_batch` entries. No module
under `python/mojoboost/` mentions any of them. `MojoBoostRegressor.predict`
calls `predict_range`, so a fitted model never asks whether GPU prediction
covers it and never takes that path. Registering is not connecting either.

### 6.3 Nine native modules no entry point reaches

In seven clusters, so that fixing the head of a cluster fixes all of it:
`alternate_boosting` (with `boosting_dart`, `boosting_rf`),
`gpu_binned_layout` (with `gpu_bin_packing`), `gpu_levelwise` (with
`levelwise_policy`), `gpu_multiclass_batch`, `hybrid_leaf_scheduler` (with
`histogram_cache_policy`), `gpu_categorical` (with `gpu_sparse`,
`gpu_sparse_layout`), and `gpu_portability` (with `gpu_backend_policy`).

`tests/test_gpu_portability.mojo` deserves its own sentence: it imports
`gpu_tiling` and `histogram_gpu`, not `gpu_portability`. The test named for
the module does not exercise the module.

### 6.4 Two parameter surfaces, one unreachable from Python

`params.mojo` is the only production writer of `ExtraTreeParams`
(`min_gain_to_split`, `max_delta_step`, `path_smooth`, `extra_trees`,
`extra_seed`, `monotone_penalty`, `monotone_method`, feature penalties,
forced splits). `parse_params` is imported by the C ABI and the CLI and not
by the extension, which builds its argument list positionally. So the C and
CLI surfaces can set nine tree parameters the Python estimators cannot
express at all. `basic_bindings.mojo`'s `extra_params_check` and
`forced_splits_check` are the intended Python route, and they are behind
6.1.

### 6.5 EFB is a guard with no engine behind it

`efb.mojo` is reachable. What is reachable is `check_bundling_supported` and
`check_bundling_params`, which *raise* when a caller asks for bundling. The
machinery - `fit_bundles`, `bundle_csc`, `unbundle_histogram` - has no
production caller; `boosting.mojo` and `tree.mojo` mention a bundle only to
say it "defaults to inactive". The refusal is the right failure mode, and
the module still reads as connected in an import graph while being connected
to nothing behaviorally. This is the clearest instance of the distinction the
whole audit exists to draw.

### 6.6 `python/mojoboost/_compat.py` is unreferenced, and that is a defect

The module holds `unsupported_interpreter()` and `import_extension()`, the
checks that must run *before* `mojoboost._mojoboost` is imported. The reason
is measured and recorded in its docstring: on CPython 3.9 the Mojo runtime
resolves `Py_NewRef` out of libpython at load time, fails, and **aborts the
process**. An abort cannot be caught, so a `try` around the import cannot
help. The only place a check can do any good is in front of the import.

`rg _compat python/` returns exactly one hit, inside `_compat.py`'s own
docstring. `python/mojoboost/__init__.py:260` imports `_mojoboost` with no
guard ahead of it. A user on an unsupported interpreter gets `ABORT: symbol
not found: Py_NewRef` instead of the message this module was written to
print. Patch request 7.2(c).

---

## 7. Cross-lane patch queue

Every unresolved finding, assigned. In dependency order: 7.1 unblocks more
than the rest combined. **This lane made none of these edits.**

### 7.1 Lane 06 - `bindings/_mojoboost.mojo` and `bindings/build.sh`

**The one that unblocks the rest.** Lane 05's handoff §5.1 independently
asked for part (a) and called it "the single change that moves Python off the
`narrow` contract"; this audit agrees and extends it to all five modules.

**(a)** Import the five sibling modules at the top of `_mojoboost.mojo`:

```mojo
from basic_bindings import (
    decide_device,
    efb_check,
    efb_defaults,
    extra_option_supported,
    extra_params_check,
    forced_splits_check,
    native_clock_ns,
    startup_environment,
    startup_phase_contract,
)
from dataset_bindings import (...)        # 9 functions
from distributed_bindings import (...)    # 4 functions
from inspection_bindings import (...)     # 13 functions
from objective_bindings import (...)      # 10 functions
```

Take the exact name lists from each module's own `def` block, and from
lane 14's handoff when it lands, rather than from this file: these counts
were read at `860b1cf` and lane 14 was still writing.

**(b)** Register each in the `PythonModuleBuilder` block beside the existing
64 `m.def_function[...]("...")` lines. Registration is what creates the
attribute; the import alone does nothing.

**(c)** `bindings/build.sh` currently runs

```sh
pixi run mojo build --emit shared-lib -I src \
    bindings/_mojoboost.mojo -o python/mojoboost/_mojoboost.so
```

It needs `-I bindings` as well, or `objective_bindings.mojo`'s
`from binding_support import py_dict, py_pair` cannot resolve.
`bindings/build.sh` is **unowned** by any lane in this round; lane 06 is the
nearest owner and should take it, or say who does.

**(d)** Prefer the four names Python is already reaching for -
`decide_device`, `dump_model`, `objective_code`, `registry_metrics` - if any
part of this has to be split across commits. Each one deletes a live
fallback.

### 7.2 Lane 07 - `python/mojoboost/__init__.py`

**(a)** After 7.1 lands, route prediction through the batch and GPU entries.
`gpu_predict_capability` plus the four `predict_*_batch` functions are
registered and uncalled; `predict`/`predict_proba`/`predict_raw` are
superseded by their `_range` forms and still exported. Decide which set is
the real one and make the other unreachable on purpose.

**(b)** `_eval.py`'s metric-code table is a hand mirror of the native table.
Once `registry_metrics` is registered, take the table from it.

**(c)** **Call the interpreter guard.** Ahead of the
`from . import _arrays, _eval, _mojoboost, ...` at line 260:

```python
from . import _compat as _compat_checks

_reason = _compat_checks.unsupported_interpreter()
if _reason is not None:
    raise ImportError(_reason)
```

Confirm the return shape against `_compat.py` before applying; the point of
the request is that *something* must call it, not the exact spelling. This is
the only finding in this audit that changes what a user sees on a supported
failure path, and today that failure path is an uncatchable abort.

### 7.3 Lane 17 - `src/mojoboost/alternate_boosting.mojo`

`alternate_boosting`, `boosting_dart`, and `boosting_rf` form a closed
cluster nothing imports. Either connect the head to a real entry point, or
mark the cluster `EXPERIMENTAL` in each module's own docstring and add rows
to `CLASSIFICATION`. The lane's brief allows either; what it does not allow
is leaving the question unanswered, because `boosting="dart"` currently has
no route and the estimator accepts the string.

### 7.4 Lane 02 - `gpu_binned_layout`, `gpu_levelwise`

Both are cluster heads with a second module reachable only through them
(`gpu_bin_packing`, `levelwise_policy`). Neither `train_gpu` nor
`histogram_gpu` asks either for a plan. Connect behind a conservative
internal switch, or classify.

### 7.5 Lane 04 - `gpu_multiclass_batch`, `hybrid_leaf_scheduler`

Same shape. `hybrid_leaf_scheduler` carries `histogram_cache_policy` with it.
Multiclass GPU training is still per-class, so `gpu_multiclass_batch` is a
faster path nothing can select.

### 7.6 Lane 10 - `gpu_categorical`

Carries `gpu_sparse` and `gpu_sparse_layout`. The GPU trainer refuses
categorical features, so its category kernels cannot run. If the refusal is
the intended shipping state, say so in `gpu_categorical.mojo`'s docstring and
the cluster becomes `EXPERIMENTAL` rather than `PENDING`.

### 7.7 Lane 20 - `gpu_portability`, and `tests/test_gpu_portability.mojo`

`gpu_portability` carries `gpu_backend_policy`. Separately, and worth its own
line: the test named `test_gpu_portability.mojo` imports `gpu_tiling` and
`histogram_gpu`, not `gpu_portability`. Whoever owns that test file (lane 20
owns the module, not the test) should either point it at the module or rename
it, because right now it reads as coverage that does not exist.

### 7.8 Lane 16 - `src/mojoboost/lgbm_model_io.mojo`

1,400 lines of LightGBM text model reader and writer, reached only from its
own test. No entry point offers LightGBM interop to any caller.
`inspection_bindings.mojo` has a `model_file_kind` that could carry it.

### 7.9 Lane 08 - `src/mojoboost/params.mojo` and `objective_registry.mojo`

**(a)** `params.mojo` still carries `objective_from_name` and
`objective_display_name` alongside `objective_registry`'s canonical table.
Two tables of the same fact.

**(b)** Lane 05 §5.4 asks for a test pinning
`objective_registry.LAMBDARANK == ranking.LAMBDARANK`. This audit seconds it:
the copy is deliberate and cycle-avoiding, and a comment is not a mechanism.

### 7.10 Lane 09 / 06 / 07 jointly - the parameter split

Nine `ExtraTreeParams` fields are settable from C and the CLI and not from
Python (§6.4). Resolution is one of:

- route the extension's argument list through `parse_params` (lane 06 edits
  `_mojoboost.mojo`, lane 09 keeps `params.mojo` authoritative), or
- add the nine to `_Base.__init__` and thread them (lane 07 edits the
  estimators, lane 06 widens the binding signature).

The first keeps one parameter engine, which is what the brief prefers. The
choice is not this lane's to make; the fact that it is unmade is.

### 7.11 Lane 09 - EFB

`fit_bundles`, `bundle_csc`, `unbundle_histogram` have no production caller
(§6.5). Either call them from `binning.mojo`/`boosting.mojo` behind
`enable_bundle`, or state in `efb.mojo`'s docstring that the guards are the
whole shipping surface and the machinery is `EXPERIMENTAL` pending
validation.

### 7.12 Lane 19 - `docs/COMPATIBILITY_POLICY.md` and `tools/api_snapshot.py`

`docs/COMPATIBILITY_POLICY.md:770` calls itself the specification for
`tools/api_snapshot.py`, which does not exist.
`tests/parallel/api_snapshot_manifest.json` exists and nothing reads it.
Write the script, or mark the manifest as unchecked in the document.
(`bench/bench_startup.mojo` is the other missing path and needs nothing:
`docs/STARTUP_LATENCY.md:265` says it does not exist in the same sentence.)

### 7.13 Unowned - `python/build/lib.macosx-11.0-arm64-cpython-314/`

A stale copy of the Python package from an earlier build, tracked or
untracked, sitting where every `rg` over `python/` will find it and report
each hit twice. `connectivity_audit.py` skips `build/` by name. Somebody
should add it to `.gitignore` or delete it; this lane did neither, because
deleting another lane's working files is exactly what the brief forbids.

---

## 8. Dead, experimental, and awaiting validation

The brief asked for these to be separated. They are, and the split is not
even:

**True dead code: none.** Not one finding in this audit is code that nothing
reaches and nothing was ever meant to reach. That is a real result. The
unreachable code in this repository is not abandoned, it is unconnected, and
the two call for opposite responses. A cleanup pass that treated §6.3 as dead
would delete seven working GPU subsystems.

**Intentionally experimental (2), both already self-documenting:**

- `src/mojoboost/backend.mojo` - a one-function dispatch shim reached only
  from `tests/test_backend_equivalence.mojo`, which is in the `test` pixi
  task. It is the reference the equivalence test compares against. Test-only
  by design. Leave it.
- `python/mojoboost/_public_api_plan.py` - a proposal expressed as data. Its
  docstring says nothing in the package imports it and means it; importing it
  would be the bug. Leave it, and leave the docstring, because the docstring
  is what stops the next audit from re-finding it.

Borderline and classified `EXPERIMENTAL` with a reason rather than
`PENDING`: `unified_memory_policy.mojo`, now reached from `device_policy` and
`histogram_gpu`. The routes it scores are not implemented in any trainer, so
its decision has one live outcome. Connected, and not yet meaningful.

**Awaiting validation (everything else, 30-odd findings).** Implemented,
tested where a test exists, and blocked on an edit in a file the implementing
lane does not own. Every one is in §7 with an owner. The distinguishing
evidence is consistent: each has a module docstring written by someone who
expected it to be called, and most have a test.

The asymmetry is the finding. This repository's problem is not accumulated
dead code. It is that parallel ownership makes the last edit - the one in
somebody else's file - the one that never happens.

---

## 9. Fallbacks preserved

This lane changed no behavior, so no fallback needed preserving. What it did
was make the surviving fallbacks legible, which is the prerequisite for
removing them safely:

- `python/mojoboost/inspection.py`'s text-parsing path stays until 7.1 lands.
  Its `# DELETION POINT` banner already names the exact functions and the
  exact lines to delete. This audit's contribution is confirming those
  functions now exist in Mojo.
- `device_selection.py`'s `"narrow"` contract stays until `decide_device` is
  registered. It degrades by announcing itself, which is the right design and
  is why the gap was findable at all.
- `_eval.py`'s mirrored metric table stays until `registry_metrics` is
  registered.

The script itself defaults to `--fail-on none`: reporting and exiting 0.
Nobody's CI changes because this landed. Gating is a later, separate,
deliberate decision.

---

## 10. Serialization and public-API effects

**Serialization: none from this lane, and one finding closed by another.**

Two hours before the snapshot, `Tree.split_gain` was filled in by training
(`tree.mojo:327`, `distributed.mojo:670`), read by `importance.mojo` and
`inspection.mojo`, and written by neither `save_model` nor `_write_trees`, so
`importance_type="gain"` returned zeros after any save/load round trip. The
reader documented it as intended: "Split gains are a training artifact and
are not serialized."

At `860b1cf` the format is `v4`, `_write_trees` emits per-node gains behind a
`_has_split_gains` guard, and `_read_trees` restores them. Lane 11 closed it.
Recorded here because it was the audit's highest-severity finding for about
ninety minutes, and because a `# this is intentional` comment on a real bug
is worth remembering as a pattern.

The general check survives in the script: for every field of `Tree`,
`Booster`, and `BinMapper`, does `serialize.mojo` mention it at all? A field
training fills and serialization drops is state a saved model loses, and
prediction is the wrong place to notice, because prediction usually does not
need it. Importance, inspection, and contributions do.

**Public API: none.** No Python export, no `def_function` entry, no
`@export`, no native re-export changed. `tools/connectivity_audit.py` is a
developer tool with no import-time relationship to the package: it imports
`argparse`, `json`, `os`, `re`, `sys`, and `collections`, and never imports
`mojoboost`. It cannot appear in a wheel's runtime path and does not belong
in one.

---

## 11. Risks

**The script is UNRUN.** Stated in §0 and repeated here because it is the
largest risk in this handoff. Its first execution may traceback. That is
expected work.

**The snapshot is already stale.** Nine native modules gained importers
during this session and `bindings/` grew from one file to seven. Any specific
row in `docs/CONNECTION_AUDIT.md` may have been resolved between writing and
reading. The document says so at the top; the script is the durable artifact.

**`CLASSIFICATION` will rot.** It is a hand-maintained table of judgments
about modules other lanes own. A stale `EXPERIMENTAL` row is worse than no
row, because it silences a finding. The script's docstring says a
`CONNECTED` classification on something the graph calls an orphan means the
row is stale rather than the graph wrong, and the default for unknown
subjects is `PENDING` for exactly this reason.

**Regex parsing of Mojo under-reports.** The script matches imports and
top-level declarations with patterns tuned to this repository's style. A
file written differently contributes no edges, so it would be reported as an
orphan when it is not - a false positive on unreachability, never a false
negative. That direction is the safe one, but it means a finding should be
confirmed by reading before it is acted on.

**`strip_mojo_comments` is heuristic.** It removes `#` to end of line and
triple-quoted blocks without tracking string context, so a `#` inside a
string literal truncates that line. Docstrings must be stripped - this
repository's docstrings name other modules constantly, and leaving them in
would make every module reachable from every other one - so some heuristic is
required. This one errs toward stripping too much, which again means false
orphans rather than missed ones.

**A concurrent commit swept this lane's work in.** Commit `dc21f03`
("Connect accelerator and public API foundations"), authored by another
agent, committed the entire working tree including this lane's in-progress
`tools/connectivity_audit.py`. `handoffs/connect_18_release.md` records the
same incident from its own side, so it hit at least two lanes. Nothing was
lost and nothing was reverted here. The mechanism is worth naming: in a
shared worktree, `git commit` without an explicit pathspec commits every
lane's uncommitted work, and the repository memory's rule about committing by
explicit path exists for this.

**Findings are about existence, never correctness.** A connected path is a
path that exists. This audit cannot tell you that GPU training produces the
same trees as CPU training, that a registered binding has the right
signature, that a serialized field round-trips to the same value, or that any
performance claim holds. Those need the test suites, `check_parity.py`, and
hardware. Reachability is the precondition for all of it and the only thing
measured here.

---

## 12. Smallest later commands - ALL UNRUN

Nothing below has been executed. Ordered cheapest first; each is
individually useful.

```sh
# 1. Does the script parse at all? The cheapest possible check, and the
#    right first command, because everything after it assumes this passes.
#    UNRUN
python3 -m py_compile tools/connectivity_audit.py

# 2. First real run. Expect a traceback or a wrong count; both are work,
#    not findings.
#    UNRUN
python3 tools/connectivity_audit.py

# 3. The one section worth running before any other, because it is the
#    finding that unblocks four Python fallbacks at once (§6.1).
#    Expect: five rows, one per unregistered sibling under bindings/,
#    plus one for build.sh missing -I bindings.
#    UNRUN
python3 tools/connectivity_audit.py --section binding-modules

# 4. Confirm the orphan clusters of §6.3 against today's tree before
#    anyone acts on §7.3 through §7.8.
#    Expect at 860b1cf: alternate_boosting, boosting_dart, boosting_rf,
#    gpu_binned_layout, gpu_bin_packing, gpu_categorical, gpu_levelwise,
#    gpu_multiclass_batch, gpu_portability, gpu_backend_policy,
#    gpu_sparse, gpu_sparse_layout, histogram_cache_policy,
#    hybrid_leaf_scheduler, levelwise_policy, lgbm_model_io, backend.
#    UNRUN
python3 tools/connectivity_audit.py --section orphans

# 5. The two-directional binding gap. Expect decide_device, dump_model,
#    objective_code, registry_metrics on the missing side.
#    UNRUN
python3 tools/connectivity_audit.py --section missing-bindings
python3 tools/connectivity_audit.py --section unused-bindings

# 6. Machine-readable, for whoever wants to diff two runs and see what a
#    day of lane work actually connected.
#    UNRUN
python3 tools/connectivity_audit.py --json > /tmp/audit-$(git rev-parse --short HEAD).json

# 7. Only after the queue in §7 is worked and someone wants a gate. Not
#    before: today this exits 1 on roughly thirty findings.
#    UNRUN
python3 tools/connectivity_audit.py --fail-on PENDING

# 8. Unchanged and unaffected by this lane. Named here only so nobody
#    mistakes §9's parity paragraph for a replacement.
#    UNRUN
pixi run check-parity
```

**Not run, and deliberately not listed as a next step:** any `mojo build`,
any `mojo run`, any test suite, `bindings/build.sh`, any benchmark, any
formatter. This lane's brief was static inspection, its files are one Python
script and two documents, and no claim here rests on a compiler.

---

## 13. Handoffs read

`connect_05_device_policy.md`, `connect_12_dataset_cv.md`, and
`connect_18_release.md` were the only `connect_*` handoffs present at
`860b1cf`; all three were read and are folded into §7. The other eighteen
lanes had not filed by the time this lane finished, which is the one gap in
the brief this lane could not close: the queue in §7 is derived from the
source, not from eighteen handoffs, and a lane that filed after `860b1cf`
may have resolved or superseded a row in it.

The remedy is the script. Whoever assembles the next round should run §12
command 2 first and treat its output as the current queue, using §7 for the
owner assignments and the exact edits rather than for the finding list.

The forty-odd earlier handoffs (`algorithm_*`, `apple_*`, `integration_*`,
`migration_*`, `performance_*`, `release_*`, `task*`) were consulted where a
specific module's intent was unclear - they are how `unified_memory_policy`
and `backend.mojo` got classified `EXPERIMENTAL` rather than `PENDING`.
