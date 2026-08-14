# Drift report

What a static reading of the tree found while the schemas above it were
being written. Every finding is a disagreement between two things in the
repository that are supposed to agree, or between the repository and a
document that claims to describe it.

**Nothing here was changed.** Every file named is outside this lane's
ownership. Each finding carries the fix, and the ones that are mechanical
appear as ready-to-apply patches in
`handoffs/remaining_13_compatibility.md`.

**Nothing here was executed.** These are readings of source text. Where a
finding predicts a runtime failure, the prediction is marked and is
**UNRUN**: it follows from the code as read and has not been observed.

Findings are ordered by consequence, not by discovery.

---

## F1. The pure-Python model parser cannot read what the writer writes

**Severity: breaks a public path. UNRUN prediction.**

| Where | Says |
|---|---|
| `src/mojoboost/serialize.mojo:84` | `comptime _VERSION = "v4"` |
| `src/mojoboost/serialize.mojo:89` | `comptime CURRENT_FORMAT_VERSION = 4` |
| `src/mojoboost/serialize.mojo:_read_version` | accepts `v1`, `v2`, `v3`, `v4` |
| `python/mojoboost/inspection.py:93` | `SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3)` |

`parse_model_string` raises on any version outside that tuple, with the
message "model format v4 is newer than this build reads". So a model saved
by the current build and handed to the pure-Python parse path is rejected
by the same build that wrote it.

The path this reaches is `dump_model`'s fallback, `_dump_from_text`, which
is what runs when the native dump binding is not available. `dump_model`
is in `mojoboost.__all__` as of the current tree, so this is a public
surface failing on a model the public API just produced.

This is invariant **I5** in [SNAPSHOT_SCHEMA.md](SNAPSHOT_SCHEMA.md)
section 6, and it is the reason I5 exists.

**Fix.** `SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3, 4)`, and the v4
additions handled in `_parse_trees`: per-node split gains, the covers
presence flag, and the optional `feature_names` section. The tuple change
alone is not enough and would turn a clean refusal into a misparse, which
is worse. Owner: whoever owns `python/mojoboost/inspection.py`.

**Not verified.** No parser was run and no model was saved. The claim is
that the constant excludes 4 and that the writer emits `v4`, both of which
are visible in the source. That the failure occurs in practice depends on
which dump path is taken at runtime, which was not exercised.

---

## F2. The dump reports a model format version the writer no longer writes

**Severity: silently wrong answer to a documented question.**

| Where | Says |
|---|---|
| `src/mojoboost/model_dump.mojo:68` | `comptime MODEL_FORMAT_VERSION = 3` |
| `src/mojoboost/serialize.mojo:87` | comment: `MODEL_FORMAT_VERSION` in model_dump.mojo "reports this number to a dump consumer and has to track it" |
| `src/mojoboost/serialize.mojo:89` | `CURRENT_FORMAT_VERSION = 4` |

The two constants are documented as coupled and they are not equal. A
native dump therefore reports `model_format_version: 3` for a model that
would serialize to v4.

`docs/MODEL_INSPECTION_SCHEMA.md` makes `model_format_version` the key a
consumer branches on to know which optional facts a model of that vintage
can carry at all. A consumer told 3 concludes that split gains cannot be
present and skips them, on a model that has them.

Nothing raises. This is invariant **I4**.

**Fix.** `MODEL_FORMAT_VERSION = 4` in `src/mojoboost/model_dump.mojo`,
and check `has_split_gain` is reported true where gains exist. Owner:
whoever owns `model_dump.mojo`.

---

## F3. Three documents state the model format version as v3

**Severity: documentation, but it is the normative documentation.**

| Document | Location | Says | Should say |
|---|---|---|---|
| `docs/COMPATIBILITY_POLICY.md` | Section 1.4 table | Model format version `v3` | `v4` |
| `docs/COMPATIBILITY_POLICY.md` | Section 7.1 | "current version `v3`" | `v4` |
| `docs/COMPATIBILITY_POLICY.md` | Section 7.2 table | Rows for v1, v2, v3 | A v4 row: split gains, the covers presence flag, and the optional `feature_names` section |
| `docs/MODEL_INSPECTION_SCHEMA.md` | Versioning section | "current values are `dump_format_version: 1` and `model_format_version: 3`" | `model_format_version: 4` |
| `docs/MODEL_INSPECTION_SCHEMA.md` | Top level table | "`model_format_version` ... 1, 2, or 3" | 1, 2, 3, or 4 |
| `tests/parallel/api_snapshot_manifest.json` | `versions.model_format` and the `model_format` block | `v3`, readable `["v1","v2","v3"]` | Superseded by `compatibility/api_snapshot.json` |

Section 7.2 of the compatibility policy is the read-back matrix, and
release gate item B3 requires it to be extended whenever the format
version bumps. The bump happened and the matrix did not move, which means
gate item B3 has an unrecorded failure sitting in the tree.

`docs/COMPATIBILITY_POLICY.md` and `docs/MODEL_INSPECTION_SCHEMA.md` are
both outside this lane. The v4 row text is drafted in the handoff, ready
to paste.

---

## F4. The C ABI version bumped and nothing that records it was updated

**Severity: the snapshot's whole purpose, demonstrated on itself.**

`capi/mojoboost.h:83` is `#define MOJOBOOST_ABI_VERSION 2`. Both documents
that state the ABI version say 1:

- `docs/COMPATIBILITY_POLICY.md` section 1.4 table, and section 6.3,
  "`MOJOBOOST_ABI_VERSION` is 1"
- `tests/parallel/api_snapshot_manifest.json`, `versions.c_abi: 1` and
  `c_abi.abi_version: 1`

The header also grew, in ways the manifest's fourteen-declaration list
does not cover. Four new functions and two new constant groups:

| Added | Kind |
|---|---|
| `mojoboost_gpu_available` | function |
| `mojoboost_predict_ex` | function |
| `mojoboost_model_num_iterations` | function |
| `mojoboost_model_dump_json` | function |
| `mojoboost_string_free` | function |
| `MOJOBOOST_DEVICE_CPU`, `_GPU`, `_AUTO` | `#define` |
| `MOJOBOOST_PREDICT_RESPONSE`, `_RAW` | `#define` |

Adding a function is additive under compatibility policy section 6.3 and
needs no ABI bump. Something in this set was judged to need one, and
whatever that was, it is a **breaking** change under that section: a major
release and a break note. Neither exists, because there is no release yet
and section 1.3's pre-1.0 allowance covers it. What is missing is the
record.

This finding is what the snapshot is for. It is exactly the diff
`tools/api_snapshot.py --check` reports as `breaking` under section 11.2's
row "A C ABI declaration changes".

**Fix.** Update both documents, and record in the release notes for the
first tagged release which change took the ABI from 1 to 2 and what a
caller compiled against 1 sees.

---

## F5. `mojoboost.__all__` grew by nine names, and gate item C5 is resolved

**Severity: additive, and one release gate item can be closed.**

The manifest lists seventeen names. `python/mojoboost/__init__.py:274` now
lists twenty-six. Added:

`cv`, `CVBooster`, `build_info`, `show_versions`, `explain_device_choice`,
`dump_model`, `trees_to_dataframe`, `trees_to_records`,
`get_split_value_histogram`.

Every one is additive and lands correctly in a minor release. Two things
follow that are not just a bigger list.

**Gate item C5 is resolved, and section 6.1 has not caught up.** C5 asked
that `mojoboost.inspection` be either re-exported at the top level or
added to the supported import paths. It was re-exported: four inspection
names are in `__all__`, resolved lazily through `_LAZY_ATTRS`. Section 6.1
of the compatibility policy still says no submodule other than
`mojoboost.callback` is a supported import path, and section 8.1 still
describes the question as open. Both should now describe the resolution.

**Five names in `__all__` are resolved by `__getattr__`, not bound
eagerly.** `_LAZY_ATTRS` maps `explain_device_choice`, `dump_model`,
`trees_to_dataframe`, `trees_to_records`, and `get_split_value_histogram`
to their submodules, and `_LAZY_SUBMODULES` holds `dask`,
`device_selection`, `diagnostics`, and `inspection`.

That distinction is caller-visible in at least two ways and is why
`python.lazy_attributes` is its own block in the snapshot schema. A lazy
name fails at first access rather than at import if its optional
dependency is missing, and `hasattr` on the module resolves it, paying the
import. A snapshot that recorded only `__all__` would call an eager name
becoming lazy a no-op, and it is not.

---

## F6. Seven environment variables are documented; twenty-one are named in code

**Severity: an undocumented surface people will find and use.**

Compatibility policy section 9.5 documents seven. A scan for double-quoted
`MOJOBOOST_*` literals under `src/`, `bindings/`, `python/`, `capi/`, and
`cli/` finds twenty-one, of which nineteen are read by Mojo and two by
Python.

Declared and read (5 of the 7):
`MOJOBOOST_DISABLE_GPU`, `MOJOBOOST_GPU_HIST_STRATEGY`,
`MOJOBOOST_AUTO_MIN_CELLS`, and, through the `_env_int` wrapper in
`parallel.mojo`, `MOJOBOOST_NUM_WORKERS` and
`MOJOBOOST_PARALLEL_MIN_OPS`.

Declared, and **not found by any literal scan** (2 of the 7):
`MOJOBOOST_GPU_BLOCK_THREADS` and `MOJOBOOST_GPU_ROW_TILE`. Either they
are read through a computed name, or they are documented and unread. A
static reading cannot tell which, and the difference matters: the second
is a documented control that does nothing.

Read, and not declared (16):
`MOJOBOOST_CPU_CORE_POOL`, `MOJOBOOST_DASK_BACKEND`,
`MOJOBOOST_DIST_MACHINES`, `MOJOBOOST_DIST_MODE`,
`MOJOBOOST_GPU_BACKEND`, `MOJOBOOST_GPU_BACKEND_UNVALIDATED`,
`MOJOBOOST_GPU_GRAD_LAYOUT`, `MOJOBOOST_GPU_HIST_SPECIALIZATION`,
`MOJOBOOST_GPU_OBJECTIVE`, `MOJOBOOST_GPU_SPLIT_STRATEGY`,
`MOJOBOOST_GPU_TRACE`, `MOJOBOOST_GPU_TRANSFER`,
`MOJOBOOST_GPU_TRANSFER_UNPROVEN`, `MOJOBOOST_GPU_VALID_SCORING`,
`MOJOBOOST_GPU_WARMUP`, `MOJOBOOST_HYBRID_LEAVES`,
`MOJOBOOST_STARTUP_TRACE`.

An undeclared variable is not a bug. Most of these read as diagnostic and
tuning knobs, and section 2 of the compatibility policy already says only
what is listed is public. But an environment variable is discoverable by
grep and is the easiest surface in the project to depend on accidentally,
and two of them, `MOJOBOOST_GPU_BACKEND_UNVALIDATED` and
`MOJOBOOST_GPU_TRANSFER_UNPROVEN`, name in their own spelling the reason
nobody should rely on them.

**Fix.** Not a policy change. Either declare each one or say in one place
that the undeclared `MOJOBOOST_*` variables are diagnostics and carry no
promise. The snapshot's `environment.undeclared` block keeps the count
honest either way, and `environment.stale` is what would fail if a
documented variable stopped being read.

---

## F7. `SUPPORTED_KEYS` grew from nineteen keys to thirty-two

**Severity: additive, and the record is stale.**

`src/mojoboost/params.mojo:66` now lists thirty-two keys. The manifest
lists nineteen. The thirteen added:

`feature_fraction_bylevel`, `min_gain_to_split`, `max_delta_step`,
`path_smooth`, `extra_trees`, `extra_seed`, `monotone_penalty`,
`monotone_constraints_method`, `cegb_tradeoff`, `cegb_penalty_split`,
`enable_bundle`, `max_conflict_rate`, `data_sample_strategy`.

Every one is additive under compatibility policy section 4.4: a key moving
from unsupported to accepted never changes what an existing program does.
Nothing is wrong here except that no record moved, which is the pattern
F3, F4, F5, and F6 also show.

Note for the tool author: the literal is one implicitly concatenated
string split across eleven lines, and the separating commas sit at the
ends of the fragments. Joining before splitting is required, and it is
parsing rule 6 in the snapshot schema.

---

## F8. A cross-reference in the reset-slot contract names a symbol that does not exist

**Severity: cosmetic, and it sits on the one contract that fails silently.**

`bindings/_mojoboost.mojo:622` says the slot order is "the contract the
Python side mirrors in `_RESET_SLOTS`". The Python side names it
`RESETTABLE`, in `python/mojoboost/callback.py:105`. There is no
`_RESET_SLOTS` anywhere under `python/`.

The ordering itself agrees, entry for entry, and `RESET_SLOTS` is 9 while
`len(RESETTABLE)` is 9. Only the name in the comment is wrong.

This was already reported in `handoffs/task20_compatibility.md` and is
still true. It is repeated here because of where it sits: section 9.3 of
the compatibility policy calls this pair a wire format between two files,
and says a change to one side alone reassigns parameters silently. A
comment that sends a reader looking for the wrong symbol is a small cost
on the one contract in the project whose failure mode is wrong numbers
rather than an error.

Invariants **I6** and **I7** are what make the pair checkable, and they
pass on the tree as read.

---

## F9. The snapshot manifest is in the wrong place and says it is proposed

**Severity: superseded by this lane.**

`tests/parallel/api_snapshot_manifest.json` has `status: "proposed"`,
`generated_by: null`, and an `about` block explaining that it was written
by hand. Its own handoff says the location was a round convention, that
`tests/parallel/` holds no other JSON, and that no test runner reads it.

Findings F3 through F7 are all cases where it is now wrong about the tree,
which is what a hand-written record does over four months of parallel
work. It was cross-checked once by a throwaway script that no longer
exists, and nothing re-runs.

**Fix.** `compatibility/api_snapshot.json`, generated by
`tools/api_snapshot.py --write`, at `schema_version` 2. The old file is
superseded wholesale rather than migrated, which is what its own handoff
said would happen. The deletion is a ready-to-apply patch in the handoff;
this lane does not own `tests/`.

---

## What was checked and found to agree

Stated so the report is not read as a list of everything that was looked
at.

| Checked | Result |
|---|---|
| The three library version locations | All three are `0.1.0` |
| `len(RESETTABLE)` against `RESET_SLOTS` | Both 9 |
| `RESETTABLE` order against the slot order in `bindings/_mojoboost.mojo` | Agrees, entry for entry |
| `CallbackEnv` field list and order | Six fields, unchanged from the manifest |
| `_RESET_ALIASES` | Eleven entries, unchanged |
| `_FITTED_ATTRS` | Eleven names, unchanged |
| `pixi.toml` declared platforms | `osx-arm64`, `linux-64`, `linux-aarch64`, unchanged |
| CI runner matrix | `ubuntu-latest` and `ubuntu-24.04-arm`, which is what the tier 1 rows claim |
| `requires-python` | `>=3.14`, unchanged |
| `DUMP_FORMAT_VERSION` | 1, and no dump key was found removed or retyped |
| Estimator classes and their `_OBJECTIVES` tables | Present, three classes, unchanged shape |

Two of these deserve a caveat. The dump key check was a reading of
`inspection.py`'s `__all__` and its schema document, not a comparison of
two generated dumps, so it establishes that nothing obvious was removed
and not that no key's meaning moved. And the estimator check was a
structural one: class present, table present. Parameter defaults were not
compared value by value, because that is precisely the job of the tool
this lane wrote and did not run.
