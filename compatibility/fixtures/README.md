# compatibility/fixtures/

Model files an older format version wrote, kept so a newer build can prove
it still reads them. The promise they exist to defend is section 7.2 of
[docs/COMPATIBILITY_POLICY.md](../../docs/COMPATIBILITY_POLICY.md): a
release reads every model file any earlier release wrote.

**No fixture exists yet.** This directory holds the register
(`manifest.toml`), the rules, and the commands that would produce them.
Producing a fixture means authoring or building a model file, which this
lane does not do.

The register is split in two, because `tomllib` reads TOML and the
standard library cannot write it.

| File | Half | Maintained by |
|---|---|---|
| `manifest.toml` | Declared. Which fixtures must exist, what each is, where it came from | Hand |
| `checksums.json` | Computed. Size and SHA-256 of each of the three files per fixture | `tools/model_fixture_manifest.py --write` |

`checksums.json` does not exist. `--check` fails until the fixture files
are there and `--write` has produced it. **That failure is the correct
state, not a bug**: a manifest that passed while the evidence was missing
would be worth nothing, and a checksum that nobody computed would be worse
than worthless.

## The one rule

**A fixture is never regenerated.** It is evidence about what a build
wrote at a moment. Regenerating it destroys the only thing it proves. When
a fixture stops loading, the reader is broken, and the fix goes in the
reader.

Everything else in this document follows from that rule.

## What already exists, informally

`tests/test_missing.mojo:test_v1_file_still_loads_and_routes_nothing`
writes a hand-authored v1 file to `.test_missing_v1.tmp`, loads it, checks
that it routes no missing values, and throws it away. That is the whole of
mojoboost's backward-compatibility evidence today: one version, one shape,
one assertion, inline in a test, and gitignored so it never lands in a
commit.

The precedent it sets is the right one. A model file is a plain text token
stream, so a fixture can be authored by hand from the format spec and read
by a human. What this directory adds is that the file is kept, that its
provenance is recorded, that the other three versions are covered, and
that the assertions go beyond "it loaded".

## Provenance: where a fixture may come from

Every fixture declares one `provenance.kind`. The three are not
interchangeable and the manifest requires different fields for each.

| `kind` | Means | Required fields | Trust |
|---|---|---|---|
| `released` | Written by a build of a tagged release | `release`, `platform`, `produced_at` | Highest. This is real evidence about a file a user could hold |
| `built_at_commit` | Written by a build of a named commit that was never tagged | `commit`, `toolchain`, `platform`, `produced_at` | High. Reproducible if the commit still builds |
| `synthetic` | Authored by hand from the format specification, never written by any build | `authored_from`, `authored_by`, `produced_at` | Lowest, and unavoidable for v1 through v3 |

**Every fixture in this directory starts as `synthetic`, and that is a
statement about the project rather than a shortcut.** There is no git tag
and no published release, so no build has ever written a file that a user
could hold. There is nothing for a `released` fixture to be evidence
about. A `synthetic` v1 file proves that the reader handles the format as
specified, which is the strongest claim available before there is a
release, and it is exactly the claim `test_missing.mojo` already makes.

A `synthetic` fixture's `authored_from` names the section of
`src/mojoboost/serialize.mojo` or of
[../MODEL_FORMAT_MIGRATIONS.md](../MODEL_FORMAT_MIGRATIONS.md) the bytes
were written against, so that a later reader can tell whether the fixture
encodes the format or encodes a misreading of it.

**At the first tagged release, the fixture set gains `released` rows and
loses none.** A `synthetic` fixture is never upgraded to `released` and
never deleted in favor of one; they answer different questions, and both
answers stay.

## What a fixture is, concretely

Three files per fixture, sharing a stem.

| File | Contents | Why |
|---|---|---|
| `<stem>.mbst` | The model | The artifact under test |
| `<stem>.input.tsv` | The prediction input, one row per line, tab separated, `nan` for a missing value | Fixed so the expected output is meaningful |
| `<stem>.expected.tsv` | Raw scores for those rows, as IEEE-754 bit patterns in decimal, one per line | The assertion |

Raw scores travel as bit patterns for the same reason the model does: it
makes the comparison exact and removes every locale and precision
question. A round-trip check that compares decimal text to a tolerance is
checking something weaker than the guarantee, which is bit exactness
(compatibility policy section 8.3, item 3).

`.input.tsv` and `.expected.tsv` are checksummed alongside the model. All
three are immutable together; changing the input silently changes what the
expected output means.

## Naming

`<version>/<kind>_<shape>.mbst`, for example `v1/single_regression.mbst`
or `v2/multiclass_3.mbst`.

Two naming hazards from the repository's `.gitignore`, both of which
silently produce an untracked fixture, which is the failure mode most
likely to go unnoticed:

- `capi_test_*.mbst` is ignored **at any depth**. Never start a fixture
  stem with `capi_test_`.
- `.test_*.tmp` is ignored at any depth. Fixtures use `.mbst`, `.tsv`, and
  no leading dot, so this only bites if someone copies a scratch file in.

`/*.mbst` is anchored to the repository root and does **not** ignore
anything here. `tools/model_fixture_manifest.py --check` reports a fixture
that git does not track, which is what catches all of the above.

## The set that has to exist

Twelve fixtures, in four groups of three. Rows are declared in
`manifest.toml` with `status = "declared"` and no checksum.

| Group | Fixtures | Proves |
|---|---|---|
| v1 | `single_regression`, `single_binary`, `multiclass_3` | The oldest format still loads. No missing routing, no covers, no gains |
| v2 | `single_missing`, `single_monotone`, `single_categorical` | Missing routing round trips; each optional section loads, and its absence loads as the unconstrained or fully numerical case |
| v3 | `single_regression`, `multiclass_3`, `single_contrib` | Covers travel, so feature contributions work; the two shapes v1 covered still work at v3 |
| v4 | `single_gains`, `single_named_features`, `resaved_from_v1` | Gains travel; the optional `feature_names` section round trips; and rule R4, a v1 model re-saved at v4 is a v4 file that still has no covers and no gains |

`v4/resaved_from_v1` is the one that is not obvious and it is the most
valuable of the twelve. It is the regression test for the bug v4 fixed: v3
wrote a loaded-from-v1 model's absent covers as zeros and then could not
read the file back. Its `provenance.derived_from` names `v1/single_regression`,
and `tools/model_fixture_manifest.py` checks that the parent row exists.

## Assertions a fixture supports

The round-trip suite that reads these is **not this lane's to write**. For
whoever writes it, a fixture supports exactly these claims and no others:

1. **It loads.** `load_model` or `load_multiclass_model` succeeds, and
   `model_file_kind` dispatches to the right one from the header alone.
2. **It reports its own version.** The dump's `model_format_version`
   equals the manifest's `model_format_version`.
3. **It predicts bit-identically.** Raw scores on `.input.tsv` equal
   `.expected.tsv`, compared as bit patterns.
4. **Its absences are absences.** The capability flags of migration rule
   R2 are false where the manifest says `carries = [...]` omits the fact,
   and asking for the absent fact raises rather than returning a plausible
   number (rule R3).
5. **A re-save does not fabricate.** Saving the loaded model at the
   current version and reloading it preserves every prediction, and adds
   no fact the original did not carry (rule R4).

A fixture does not prove anything about performance, about the trainer, or
about a platform. It is evidence about one reader and one file.

## Producing them, which has not been done

```
# 1. Author or build the model files, per the provenance kind of each row.
#    v1 through v3 cannot be produced by the current build: its writer
#    emits v4 only. They are authored by hand from the format spec, the
#    way tests/test_missing.mojo already authors its v1 file.
# 2. Produce the input and expected files for each.
# 3. Fill in the register:
python3 tools/model_fixture_manifest.py --write
# 4. Confirm it is now a no-op:
python3 tools/model_fixture_manifest.py --check
```

Step 1 is the work and it needs a lane that may build and run. Steps 3 and
4 are mechanical. Nothing here has been run, and the manifest says so in
its own `status` field.
