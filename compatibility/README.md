# compatibility/

The machine-readable half of mojoboost's compatibility contract. The prose
half is [docs/COMPATIBILITY_POLICY.md](../docs/COMPATIBILITY_POLICY.md)
and [docs/DEPRECATION_POLICY.md](../docs/DEPRECATION_POLICY.md); those two
say what is promised, and the files here are what a tool can fail on.

Nothing in this directory is generated yet. Every file is either a schema,
a register a human maintains, or a placeholder that names what has to be
produced and by which command. **No checksum, no fixture, and no snapshot
in this directory was fabricated.** Where a value would have to come from
running something, the field is absent and the reason is stated, because a
plausible-looking checksum that nobody computed is worse than a missing
one: the missing one fails, and the invented one passes.

## What is here

| Path | Kind | Read by | State |
|---|---|---|---|
| `README.md` | Map | Humans | This file |
| `SNAPSHOT_SCHEMA.md` | Normative schema | `tools/api_snapshot.py` authors | Complete |
| `deprecations.toml` | Register | `tools/api_snapshot.py --check` | Complete, and empty of ordinary entries by design |
| `MODEL_FORMAT_MIGRATIONS.md` | Normative rules | Humans, and the fixture manifest's `migration` rows | Complete |
| `fixtures/README.md` | What must exist and how it is made | Humans | Complete |
| `fixtures/manifest.toml` | Fixture register, declared half: rows and provenance | `tools/model_fixture_manifest.py` | Twelve rows declared |
| `fixtures/checksums.json` | Fixture register, computed half: sizes and SHA-256 | `tools/model_fixture_manifest.py --check` | **Does not exist.** Produced by `--write` |
| `fixtures/*.mbst` | The fixtures themselves | The round-trip suite | **None exist.** See `fixtures/README.md` |
| `DRIFT_REPORT.md` | Findings | Humans | Complete, static reading only |
| `api_snapshot.json` | The snapshot | `tools/api_snapshot.py --check` | **Does not exist.** Produced by `--write` |

## The three registers and what each is for

They are easy to confuse because all three describe change over time.

**`api_snapshot.json`** is what the public surface *is*, right now,
derived from the source by a tool. It answers "what would break". It is
regenerated, never edited. It does not exist yet; `tools/api_snapshot.py
--write` is what creates it, and that command has not been run.

**`deprecations.toml`** is what was *promised* about surfaces that are
going away. It answers "what did we say, and is it still true". It is
maintained by hand, because a promise is not derivable from source.

**`fixtures/manifest.toml`** is what an *older release wrote*, kept so a
newer release can prove it still reads it. It answers "does the format
promise hold". Its rows are maintained by hand and its checksums are
computed.

A change to the public surface touches the first. A change that retires
something touches the first and the second. A change to the model format
touches all three.

## Why this directory exists rather than living under tests/

`tests/parallel/api_snapshot_manifest.json` is the hand-written draft this
work supersedes. Its own handoff (`handoffs/task20_compatibility.md`) says
the location was a round convention rather than a choice, that
`tests/parallel/` holds no other JSON, and that no test runner reads it.

Three reasons the artifacts belong here instead.

1. **They are not tests.** The snapshot is read by a release gate and by
   anyone asking what changed between two tags. Putting it under `tests/`
   makes it look like an input to a suite that does not read it.
2. **The fixtures have to outlive the suite that checks them.** A fixture
   is evidence about a release that shipped years ago. Its value comes
   from never being regenerated, which is the opposite of how everything
   else under `tests/` is treated.
3. **One directory, one owner.** The snapshot, the register, and the
   fixtures are checked against each other. Splitting them across
   `tests/`, `docs/`, and `packaging/` is how the three drift.

The old file is left where it is. Moving it is not this lane's to do, and
the migration is a ready-to-apply patch in
`handoffs/remaining_13_compatibility.md`. Until it is applied,
`tools/api_snapshot.py` reads the old path as a fallback and says so.

## Running order

None of this has been run. In the order it has to happen:

```
python3 tools/api_snapshot.py --write            # creates compatibility/api_snapshot.json
python3 tools/api_snapshot.py --check            # must then be a no-op, exit 0
# then, on a machine that can build and train:
#   produce the fixture models named in fixtures/README.md
python3 tools/model_fixture_manifest.py --write  # fills in sizes and checksums
python3 tools/model_fixture_manifest.py --check  # must then be a no-op, exit 0
```

The first command is the one to run first and to read carefully. It will
report differences against the hand-written draft, and several of them are
real drift that predates this lane rather than a bug in the tool.
`DRIFT_REPORT.md` lists the ones a static reading already found, so the
first `--write` can be judged against a written expectation instead of
being accepted because it ran.
