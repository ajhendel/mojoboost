# Handoff, remaining round task 13: backward-compatibility fixtures and deprecation infrastructure

Nothing outside the five assigned paths was touched. No build, test, bench,
tool, or profiler was run. Nothing was committed or staged.

## What this lane produced

| Path | What it is |
| --- | --- |
| `compatibility/README.md` | The map. What each register is for, why the artifacts live here and not under `tests/`, and the order the commands have to run in |
| `compatibility/SNAPSHOT_SCHEMA.md` | Normative shape of the API snapshot at `schema_version` 2. Eleven invariants, nine parsing rules, and what changed from the hand-written version 1 |
| `compatibility/MODEL_FORMAT_MIGRATIONS.md` | Normative per-version rules, R1 through R7, and what a version bump owes the fixture set |
| `compatibility/deprecations.toml` | The deprecation register. Zero ordinary entries, which is a fact about the project, and two `[[candidate]]` rows for decisions owed before the tag |
| `compatibility/fixtures/README.md` | What a fixture is, the three provenance kinds, the twelve that have to exist, and the five assertions a fixture supports |
| `compatibility/fixtures/manifest.toml` | Those twelve, declared with provenance and without checksums |
| `compatibility/DRIFT_REPORT.md` | Nine findings from a static reading, F1 through F9, plus what was checked and agreed |
| `docs/DEPRECATION_POLICY.md` | The mechanism: four states, the overlap floor, the register schema, what a removed surface does, three exemptions, and five requested gate items |
| `tools/api_snapshot.py` | The generator and checker. Standard library only, no build, no import of mojoboost |
| `tools/model_fixture_manifest.py` | The fixture checker and checksum recorder |
| `handoffs/remaining_13_compatibility.md` | This file |

## Read this before trusting any of it

**Neither tool has been run.** Not once, not with `--check`, not with
`--write`. Both are written from a reading of the sources they parse, and
every parsing decision in them is documented against the source construct
it handles, but a parser that has never met its input is a hypothesis.
Section "Validation, all of it UNRUN" at the end says exactly what to run
and what each command should print.

**No fixture, checksum, or snapshot was fabricated.**
`compatibility/api_snapshot.json` does not exist. `compatibility/fixtures/checksums.json`
does not exist. No `.mbst` file was written. Every field that would have
to come from running something is absent, and the file that would hold it
says why. A plausible-looking checksum nobody computed is worse than a
missing one: the missing one fails.

**The drift in `DRIFT_REPORT.md` is real and predates this lane.** F1 and
F2 are live inconsistencies between files in the tree. Both are outside
this lane's ownership and both have patches below.

## What the two tools are, in one line each

`tools/api_snapshot.py --check` parses the public surface out of the tree,
compares it against `compatibility/api_snapshot.json`, classifies every
difference under compatibility policy section 11.2, and separately
enforces eleven invariants that are facts about the tree rather than about
the file. `--write` regenerates the snapshot, carrying forward the two
blocks that cannot be derived.

`tools/model_fixture_manifest.py --check` verifies that every fixture the
manifest declares exists, that its digest is unchanged, that every model
format version the build can read has a fixture, and that every
provenance row carries the fields its kind requires. `--write` records
sizes and digests into `checksums.json`, and refuses to overwrite a
recorded digest without `--accept-new`.

---

# READY-TO-APPLY INTEGRATION PATCHES

Ten patches. Each is blocked only by ownership: every one is in a file
this lane may not edit. They are ordered so that a patch never depends on
a later one.

Every "minimal later validation" line is **UNRUN**.

---

## P1. The pixi tasks

**Target file:** `pixi.toml`, `[tasks]` table.
**Owner:** whoever owns `pixi.toml` (shared).
**Depends on:** nothing.
**Public API effect:** none. A task is a developer entry point.
**Serialization effect:** none.

Add alongside `check-parity`, which these deliberately mirror:

```toml
# The public API snapshot. Standard library only and no build, like
# check-parity, so it runs on a bare runner in seconds. Fails when a public
# name, default, signature, or version constant changes without the
# snapshot being regenerated, and separately when one of the eleven
# invariants in compatibility/SNAPSHOT_SCHEMA.md is violated.
api-snapshot = "python3 tools/api_snapshot.py --check"
api-snapshot-write = "python3 tools/api_snapshot.py --write"

# The model format fixture set. Verifies that every fixture still hashes
# to what it hashed to, and that every format version the reader accepts
# has one. A fixture is never regenerated; see compatibility/fixtures/README.md.
fixture-manifest = "python3 tools/model_fixture_manifest.py --check"
fixture-manifest-write = "python3 tools/model_fixture_manifest.py --write"
```

**State flow:** none. Each task is a process invocation.
**Errors:** exit 1 on any difference, any problem, or a missing snapshot.
**Fallback:** none needed. If `python3` is absent the task fails loudly,
which is the same behavior `check-parity` already has.
**Minimal later validation, UNRUN:** `pixi run api-snapshot` prints the
missing-snapshot message and exits 1 on a tree where `--write` has not
been run.

---

## P2. The CI job

**Target file:** `.github/workflows/ci.yml`, `jobs:` mapping.
**Owner:** whoever owns the workflows (shared).
**Depends on:** P1 is not required; the job calls the script directly, as
the `parity` job does.
**Public API effect:** none.

Add after the `parity` job:

```yaml
  # The public API snapshot and the model format fixture set. Standard
  # library only and no build, so this runs on a bare runner in seconds
  # alongside parity. It fails when a public name, default, signature, or
  # version constant moves without the snapshot being regenerated.
  compatibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: API snapshot
        run: python3 tools/api_snapshot.py --check
      - name: Model format fixtures
        run: python3 tools/model_fixture_manifest.py --check
```

**A sequencing note that matters.** Both steps fail on the tree as it
stands: the first because the snapshot file does not exist and because
invariants I4 and I5 are violated (findings F1 and F2), the second because
no fixture exists. **Do not add this job before P3, P4, and the first
`--write` have landed**, or `main` goes red on the commit that adds the
job and the job gets deleted rather than fixed. The correct order is P3,
P4, `--write`, commit the snapshot, then P2.

Adding the fixture step can be deferred separately: it stays red until a
lane that may build produces the twelve fixtures. Landing the first step
alone is fine and is the recommended first move.

**Errors:** a non-zero exit fails the job with the tool's own output.
**Fallback:** none. This is a gate.
**Minimal later validation, UNRUN:** the job is green on a commit where
`compatibility/api_snapshot.json` was regenerated, and red on a commit
that changes `python/mojoboost/__init__.py`'s `__all__` without it.

---

## P3. The pure-Python model parser cannot read v4 (finding F1)

**Target file:** `python/mojoboost/inspection.py`.
**Target symbols:** `SUPPORTED_MODEL_FORMAT_VERSIONS` (line 93),
`_parse_trees`, `parse_model_string`.
**Owner:** whoever owns `python/mojoboost/inspection.py`.
**Depends on:** nothing.
**Public API effect:** fixes a public path. `dump_model` is in
`mojoboost.__all__`.

**The defect.** `src/mojoboost/serialize.mojo` writes `v4`.
`SUPPORTED_MODEL_FORMAT_VERSIONS` is `(1, 2, 3)`. `parse_model_string`
raises `ValueError("model format v4 is newer than this build reads")` on
any v4 file, so the fallback dump path rejects a model the same build just
saved.

**The patch is two parts and the second is the one that matters.**

```python
#: ... (docstring unchanged above this line)
SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3, 4)
```

That line alone is **not sufficient and must not land alone.** It turns a
clean refusal into a misparse, which is worse: `_parse_trees` would read
v4's per-node split gain array as the next field it expects. The v4
additions the reader has to handle, from `serialize.mojo`'s version
history:

| Addition | Where in the stream | Effect on the parse |
| --- | --- | --- |
| Per-node split gains (`Tree.split_gain`) | Per tree, alongside the existing per-node arrays | A new array to read; sets `has_split_gain: True` |
| A presence flag on covers | Per tree, before the covers array | Covers become conditional. A v4 file written from a loaded v1 or v2 model says "no covers here", and `has_node_count` must then be False even though the file is v4 |
| Optional `feature_names` section | After the mapper, written only when a caller passed names | Read if present; names are one whitespace-free token each with five characters escaped, and an empty name travels as `\e` |

**State flow:** `parse_model_string` returns a dict whose
`model_format_version` feeds `_dump_from_text`, which sets
`has_split_gain` and `has_node_count`. With gains now present in the file,
`has_split_gain` becomes True for a v4 file parsed from text, where today
it is False for every text-parsed dump. That is a behavior change visible
to a dump consumer and it is the intended one.

**Errors:** keep the existing refusal for any version above the supported
tuple, with its current message shape. A newer file must still be reported
by version rather than misparsed (migration rule, forward compatibility).

**Fallback:** none. A parser that cannot read the current format has no
useful degraded mode.

**Serialization effect:** read side only. This patch writes nothing.

**Dependency:** none, and it does not depend on P4.

**Minimal later validation, UNRUN:** save a model with the current build,
call `mojoboost.dump_model` on it through the text path, and confirm it
returns rather than raising, that `dump["model_format_version"] == 4`, and
that `dump["has_split_gain"]` is True. Then confirm invariant I5 stops
firing: `python3 tools/api_snapshot.py --check` no longer prints the
`I5:` line.

---

## P4. The dump reports a format version the writer no longer writes (finding F2)

**Target file:** `src/mojoboost/model_dump.mojo`, line 68.
**Target symbol:** `comptime MODEL_FORMAT_VERSION`.
**Owner:** whoever owns `model_dump.mojo`.
**Depends on:** nothing.
**Public API effect:** fixes a value a documented schema key reports.

```mojo
comptime MODEL_FORMAT_VERSION = 4
```

**Why.** `src/mojoboost/serialize.mojo:87` says in its own comment that
`MODEL_FORMAT_VERSION` in `model_dump.mojo` "reports this number to a dump
consumer and has to track it", and `CURRENT_FORMAT_VERSION` there is 4.
The two are documented as coupled and are not equal.

**State flow:** `model_dump.mojo:257` assigns it to
`self.model_format_version`, which surfaces as the dump's
`model_format_version` key. `docs/MODEL_INSPECTION_SCHEMA.md` makes that
key the one a consumer branches on to know which optional facts a model of
that vintage can carry. A consumer told 3 concludes that split gains
cannot be present, on a model that has them.

**Errors:** none. Nothing raises today, which is the problem.

**Check while you are in the file:** that `has_split_gain` is reported
True where gains exist. The native dump reads the in-memory `Model`, so
gains were always available to it; what changes with v4 is that they now
survive a save, which is the text path's problem (P3) rather than this
one's.

**Fallback:** none.
**Serialization effect:** none. This constant is reported, not written to
a model file.
**Dependency:** none. Independent of P3, and both are needed: P3 fixes the
text parser, P4 fixes the native reporter.

**Minimal later validation, UNRUN:** `python3 tools/api_snapshot.py
--check` no longer prints the `I4:` line.

---

## P5. The compatibility policy has six stale statements

**Target file:** `docs/COMPATIBILITY_POLICY.md`.
**Owner:** whoever owns that document. This lane owns
`docs/DEPRECATION_POLICY.md` only, deliberately: the contract and the
mechanism are different documents and merging them would give one file two
authorities.
**Depends on:** P4 should land first for item 2, so the document and the
tree agree at the moment the edit is made.
**Public API effect:** none directly; this is the document that defines
what the public API is, so it is the highest-value patch in the list.

### P5.1, section 1.4 table

| Row | Current | Should be |
| --- | --- | --- |
| C ABI version | `1` | `2` |
| Model format version | `v3` | `v4` |
| Snapshot schema version, location | `schema_version` in `tests/parallel/api_snapshot_manifest.json`, `1` | `meta.schema_version` in `compatibility/api_snapshot.json`, `2` |

### P5.2, section 6.3

"`MOJOBOOST_ABI_VERSION` is 1" becomes 2. The header is at 2 already
(`capi/mojoboost.h:83`). The release notes for the first tagged release
must say which change took it from 1 to 2 and what a caller compiled
against 1 sees, because that is a breaking change under this very section
and the record does not exist. The header also grew five functions
(`mojoboost_gpu_available`, `mojoboost_predict_ex`,
`mojoboost_model_num_iterations`, `mojoboost_model_dump_json`,
`mojoboost_string_free`) and two constant groups (`MOJOBOOST_DEVICE_*`,
`MOJOBOOST_PREDICT_*`), all of which are additive on this section's own
terms.

### P5.3, section 7.1 and 7.2

Section 7.1: "current version `v3`" becomes `v4`.

Section 7.2 read-back table gains a row. Release gate item B3 requires
this whenever the format version bumps, so this row is an unrecorded B3
failure sitting in the tree:

```markdown
| v4 | Per-node split gains; a presence flag on covers; an optional feature-names section | Yes |
```

And a sentence after the table, because v4's covers flag changes what
"an older file loads as what it is" means in one case:

> A v4 file does not necessarily carry covers. The flag exists so that a
> model loaded from a v1 or v2 file can be re-saved without its absent
> covers being written as zeros, which v3 did and which produced a file
> v3's own reader rejected.

### P5.4, sections 6.1 and 8.1: gate item C5 is resolved

C5 asked that `mojoboost.inspection` be either re-exported at the top
level or added to the supported import paths. **It was re-exported.**
`python/mojoboost/__init__.py` now has `dump_model`,
`trees_to_dataframe`, `trees_to_records`, and
`get_split_value_histogram` in `__all__`, resolved through `_LAZY_ATTRS`.
Section 6.1 still says no submodule other than `mojoboost.callback` is a
supported import path and section 8.1 still calls the question open.

Both should describe the resolution. A sentence for 6.1:

> Four inspection entry points are re-exported at the top level and are
> public there. They are resolved on first access by a module-level
> `__getattr__` rather than bound at import, so `import mojoboost` stays
> free of pandas. The submodule path `mojoboost.inspection` reaches the
> full schema and is not, by itself, a supported import path; whether to
> make it one is a candidate in `compatibility/deprecations.toml`.

Mark C5 as met, and note that `__all__` grew by nine names, all additive:
`cv`, `CVBooster`, `build_info`, `show_versions`, `explain_device_choice`,
and the four inspection entry points.

### P5.5, section 9.5: the environment table is a selection, not an enumeration

Seven variables are documented and forty-four `MOJOBOOST_*` names appear
as literals in the source. One sentence after the table:

> The table is the public set. Other `MOJOBOOST_*` variables exist and are
> read; they are diagnostics and tuning controls, they carry no promise,
> and they may be renamed or removed in any release.
> `compatibility/api_snapshot.json` records the full observed set under
> `environment.undeclared`, so the ratio stays visible without every new
> knob becoming a documentation change.

### P5.6, section 11.3 and section 12

Section 11.3 says the generator, the pixi task, and the CI job do not
exist and points at `handoffs/task20_compatibility.md`. Repoint it at
`tools/api_snapshot.py`, `compatibility/SNAPSHOT_SCHEMA.md`, and this
handoff, and say that the manifest's home moved to
`compatibility/api_snapshot.json` at schema version 2.

Section 12 gains five items, in the numbering
`docs/DEPRECATION_POLICY.md` section 8 already uses so the two documents
agree on the labels:

```markdown
- **C6.** `pixi run api-snapshot` green: the public surface matches the
  snapshot, or every difference has been classified and accepted.
- **C7.** Every `compatibility/deprecations.toml` entry whose `remove_in`
  is at or below the release being cut is either removed with a break
  note, or moved out with the extension in the release notes.
- **C8.** No register entry has a `remove_in` that violates the overlap
  floor of docs/DEPRECATION_POLICY.md section 1 against its `since`.
- **C9.** `pixi run fixture-manifest` green: every declared fixture
  exists, its digest is unchanged, and every model format version the
  build can read has one.
- **C10.** Every surface removed since the previous release has a register
  entry in state `removed` with `removed_in` set to this release.
```

C7 is the only gate item in the whole document that fails on inaction
rather than on a change somebody made, which is why it is worth having.

**Minimal later validation, UNRUN:** none of P5 is machine-checkable
today except through `check_parity.py`'s existing citation checks. Read it
against `DRIFT_REPORT.md` findings F3, F4, F5, and F6.

---

## P6. The inspection schema states the model format version as 3

**Target file:** `docs/MODEL_INSPECTION_SCHEMA.md`.
**Owner:** whoever owns that document.
**Depends on:** P4, so the document and `model_dump.mojo` agree.

Two edits, both in the Versioning section and the top-level key table:

- "The current values are `dump_format_version: 1` and
  `model_format_version: 3`" becomes `model_format_version: 4`.
- "`model_format_version` | int | mojoboost save format version, 1, 2, or
  3" becomes "1, 2, 3, or 4".

And the `has_node_count` bullet needs a clause, because v4 broke the
"v1 or v2" shorthand:

> `has_node_count` (bool). False for a model read from a v1 or v2 file,
> whose nodes predate covers, and false for a v4 file whose covers
> presence flag says it carries none, which is what a v1 or v2 model
> re-saved at v4 produces.

**Public API effect:** this document is normative for the dump, so a
consumer reads it rather than either implementation. A consumer that
branches on `model_format_version <= 3` today is reading a document that
told it 3 was the maximum.

**`DUMP_FORMAT_VERSION` does not bump.** No dump key was removed,
retyped, or redefined. `has_node_count` gains a second way of being false,
which is a widening of an existing key's condition and not a redefinition
of what the key means. This is worth stating in the release notes because
it is the closest call in the set.

**Minimal later validation, UNRUN:** `python3 tools/api_snapshot.py
--check` shows `versions.dump_format` unchanged at 1 and
`versions.model_format_dump_reports` moved to 4.

---

## P7. Retire the superseded snapshot manifest

**Target file:** `tests/parallel/api_snapshot_manifest.json`, deleted.
**Owner:** whoever owns `tests/`.
**Depends on:** the first successful `tools/api_snapshot.py --write`, and
on the resulting `compatibility/api_snapshot.json` being committed.
**Public API effect:** none. Nothing imports or reads the file; its own
handoff records that `tests/parallel/` holds no other JSON and no test
runner reads it.

**Do not delete it before the replacement is committed.** The tool reads
it as a fallback when `compatibility/api_snapshot.json` is missing, and
prints a note saying the shapes differ. That fallback is what lets the
first `--check` produce a legible message instead of an unexplained
absence, and it is the only remaining use for the file.

**Fallback after deletion:** `load_previous` in `tools/api_snapshot.py`
handles the file being gone. The `if path == SNAPSHOT and
LEGACY_SNAPSHOT.exists()` branch simply stops firing. No code change is
needed when the file goes; leaving the branch in place costs one
`exists()` call and documents the history.

**Minimal later validation, UNRUN:** `grep -rn api_snapshot_manifest`
returns only `compatibility/` and `handoffs/` prose after the deletion.

---

## P8. A cross-reference names a symbol that does not exist (finding F8)

**Target file:** `bindings/_mojoboost.mojo`, line 622, the `RESET_SLOTS`
docstring.
**Owner:** whoever owns `bindings/`.
**Depends on:** nothing.

```mojo
comptime RESET_SLOTS = 9
"""Length of the `reset_addr` buffer. The slot order below is the contract
the Python side mirrors in `RESETTABLE` (python/mojoboost/callback.py);
changing either alone silently reassigns hyperparameters, so change both
together."""
```

The current text says `_RESET_SLOTS`, which does not exist anywhere under
`python/`. The ordering itself agrees entry for entry and `RESET_SLOTS` is
9 while `len(RESETTABLE)` is 9, so this is a comment fix and nothing else.

It is worth doing because of where it sits. Policy section 9.3 calls this
pair a wire format between two files and says a change to one side alone
reassigns parameters silently, and a comment that sends a reader looking
for the wrong symbol is a real cost on the one contract in the project
whose failure mode is wrong numbers rather than an error.

**Public API effect:** none. **Serialization effect:** none.
**Minimal later validation, UNRUN:** invariants I6 and I7 in
`tools/api_snapshot.py` already check the substance; they pass on the tree
as read, and this patch does not change what they check.

---

## P9. Wire the compatibility gate into the release workflows

**Target files:** `.github/workflows/release-macos.yml` and
`.github/workflows/release-linux.yml`, in the `build` job, after the
existing "The commit is a tag and the tag matches the version" step.
**Owner:** whoever owns the release workflows.
**Depends on:** P2, and on a committed snapshot.

```yaml
      - name: Compatibility gate
        run: |
          python3 tools/api_snapshot.py --check
          python3 tools/model_fixture_manifest.py --check
```

**Why here as well as in CI.** The CI job proves the snapshot was
regenerated on the commit that changed the surface. This step proves it at
the moment the artifact is built, which is the moment the surface becomes
something a user can hold. They are the same command and they answer
different questions, and the second is the one that maps onto gate items
C6 and C9.

**Errors:** a non-zero exit stops the release before the wheel is built,
which is the intended severity: gate items are hard gates.

**Fallback:** none, and deliberately none. A release gate with a bypass is
a suggestion.

**Do not add this before the fixture set exists**, or every release is
blocked by a step nobody can satisfy. Adding the `api_snapshot` line alone
first is the right sequencing, as with P2.

**Minimal later validation, UNRUN:** trigger the workflow on a branch with
a deliberately stale snapshot and confirm the build job fails at this step
rather than at the upload.

---

## P10. The fixture round-trip suite

**Target file:** a new `tests/test_compat_fixtures.mojo`, and its entry in
`pixi.toml`'s `test` task.
**Owner:** whoever owns `tests/`. This lane may not write tests and did
not.
**Depends on:** the twelve fixture files existing, which needs a lane that
may build and run.

**What it reads.** `compatibility/fixtures/manifest.toml` for the row
list, and each row's three files by stem. The manifest is TOML because
`packaging/matrix/validate_matrix.py` already established `tomllib` as the
house convention for a machine-readable register; a Mojo test that cannot
parse TOML can instead iterate the directory and take the version from the
file's own header token, which is the more robust reading anyway.

**The five assertions**, from `compatibility/fixtures/README.md`:

1. `model_file_kind` dispatches from the header alone, and the matching
   loader succeeds.
2. The dump's `model_format_version` equals the manifest's.
3. Raw scores on `<stem>.input.tsv` equal `<stem>.expected.tsv`, compared
   as IEEE-754 bit patterns. Not to a tolerance: the guarantee is bit
   exactness (policy section 8.3 item 3).
4. The capability flags are false for every fact the row's `carries` list
   omits, and asking for an omitted fact raises rather than returning a
   plausible number (migration rules R2 and R3).
5. Saving the loaded model at the current version and reloading it
   preserves every prediction and adds no fact the original did not carry
   (rule R4). This is the assertion that would have caught the v3 bug v4
   fixed.

**State flow:** each fixture is independent. No shared state between rows,
and no temporary file outside the runner's scratch directory. The suite
must not write into `compatibility/fixtures/`, ever, for any reason.

**Errors:** a load failure is a test failure and never a skip. A missing
fixture file is a test failure, not a skip, because
`tools/model_fixture_manifest.py --check` is what reports absence and the
suite's job is to report breakage.

**Serialization effect:** assertion 5 writes a model. It writes to scratch,
never next to the fixture.

**Fallback:** none.

**Minimal later validation, UNRUN:** the suite fails if a single byte of
any `.mbst` is changed, and `tools/model_fixture_manifest.py --check`
reports the same change as a digest mismatch. Those two failing together
is the evidence that the manifest and the suite are reading the same
files.

---

# Ownership boundaries this lane did not cross

- No file under `src/`, `python/`, `bindings/`, `capi/`, `cli/`, `tests/`,
  `bench/`, `packaging/`, or `.github/` was edited. Findings F1, F2, F4,
  and F8 all live in those trees and all are patches above.
- `docs/COMPATIBILITY_POLICY.md` was not edited. It is the contract;
  `docs/DEPRECATION_POLICY.md`, which this lane owns, is the mechanism,
  and it says in its own opening that where the two disagree the contract
  wins and the mechanism is the bug.
- `docs/LIGHTGBM_PARITY.md` was not touched. No parity status was set,
  raised, or lowered.
- `pixi.toml` and `.github/workflows/ci.yml` were not touched. P1 and P2
  are the text, ready to paste.
- `tools/check_parity.py` was not edited. `tools/api_snapshot.py` imports
  it and calls `mojo_export_names()`; that is a read, and invariant I11 is
  what turns the reuse into a check rather than a coincidence.

# Integration that was made rather than deferred

Everything below is inside this lane's own files and is live in the tree
as written, not proposed.

1. **One parser for the model format constants, used by both tools.**
   `tools/model_fixture_manifest.py` imports `model_format_block`, `ROOT`,
   `rel`, and `Deriver` from `tools/api_snapshot.py` rather than
   reimplementing any of them. The fixture coverage check therefore
   derives the readable version set from `serialize.mojo`'s own
   `if token == "vN"` chain, which is what makes a new format version fail
   the fixture check on the commit that adds it.
2. **One error-accumulation type.** `Deriver` is defined once and both
   tools report through it, so a problem prints the same way whichever
   tool found it.
3. **The deprecation register is read by the snapshot tool, not just by
   humans.** `deprecations_block` mirrors it into the snapshot so a
   register movement shows in the diff, and `check_register` enforces
   invariants I9 and I10 against the surface the same run just derived.
   The register and the tree cannot disagree about whether a name exists.
4. **`check_parity.py`'s Mojo export parser is reused and cross-checked.**
   Invariant I11 compares this lane's by-module parser against it, and
   degrades to a recorded gap rather than a failure if the import does not
   work, because a snapshot that refuses to generate is worse than one
   that generates with a named gap.
5. **The policy document is a parsed input, not just a citation.**
   `environment_block` reads the backticked `MOJOBOOST_*` names out of
   `docs/COMPATIBILITY_POLICY.md` section 9.5 to build `declared`, so
   invariant I8 fires when the documented set and the read set diverge.
   The document is the source of the promise and the code is the source of
   the behavior, and the tool compares them directly.
6. **The two documents this lane owns cite each other in both
   directions.** `docs/DEPRECATION_POLICY.md` section 7 defers model
   formats to `compatibility/MODEL_FORMAT_MIGRATIONS.md`, which defers the
   five-constant invariants to `compatibility/SNAPSHOT_SCHEMA.md` section
   6, which names the tool that enforces them.
7. **The fixture manifest's declared half and the tool's expectations are
   the same vocabulary.** `PROVENANCE_REQUIRED` in the tool and the
   provenance table in `fixtures/README.md` are the same three kinds with
   the same required fields, and a row naming a fourth kind is an error
   rather than a shrug.

# Validation, all of it UNRUN

In order. Nothing below was executed.

| # | Command | Expected on the tree as it stands |
| --- | --- | --- |
| V1 | `python3 tools/api_snapshot.py --check` | Exit 1. Prints that `compatibility/api_snapshot.json` does not exist, notes that it is reading the schema-1 draft at `tests/parallel/api_snapshot_manifest.json`, and prints `PROBLEM:` lines for invariants I4 and I5 |
| V2 | `python3 tools/api_snapshot.py --write` | Exit 1, and the file is written anyway. Non-zero because I4 and I5 are violated; the file is written because a snapshot of a tree with a known problem is still the record of that tree. Read `meta.underived` before trusting any block |
| V3 | `python3 tools/api_snapshot.py --check` again | Exit 1, with only the I4 and I5 problems and no differences. If any difference is reported here, `--write` and `--check` disagree, which is a bug in the tool and not in the tree |
| V4 | Apply P3 and P4, then V3 again | Exit 0, `ok` |
| V5 | `python3 tools/model_fixture_manifest.py --check` | Exit 1. Thirty-six missing files, plus `checksums.json` absent, plus a coverage problem for each of v1 through v4 |
| V6 | After the fixtures exist: `--write` then `--check` | `--write` records twelve fixtures; `--check` exits 0 |

**V2 and V3 are the pair that matters.** A generator whose output does not
satisfy its own checker is worthless, and that is the single most likely
defect in a tool written without being run. Run them back to back before
anything else is built on the output.

Two known behaviors of V1 and V2 that are correct and will look like bugs:

- The diff against the schema-1 draft reports nearly every path as a
  difference, because the shapes differ. Section 8 of
  `compatibility/SNAPSHOT_SCHEMA.md` lists what changed and why. Judge the
  first `--write` against `compatibility/DRIFT_REPORT.md`, which was
  written to be that expectation, and not against the draft.
- `platforms.tiers` and `numerical_contracts` are carried forward from
  whatever previous file was found. On the very first run that is the
  schema-1 draft, so they arrive with the draft's values, and `meta.carried`
  says so. Seed them by hand once from compatibility policy sections 10.2
  and 8.3, then they carry forever.
