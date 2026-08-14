# Model format migration rules

Normative. What a reader must do with a file of each version, what is
recoverable and what is not, and what a new version is allowed to change.

The format itself is defined by `src/mojoboost/serialize.mojo`, whose
version history is the source this document is written from. Section 7 of
[docs/COMPATIBILITY_POLICY.md](../docs/COMPATIBILITY_POLICY.md) is the
promise; this document is the per-step rule that makes the promise
executable.

## 1. The two directions

**Backward, which is guaranteed.** A release reads every model file any
earlier release wrote. Unconditional, within a major version and across
major versions unless a release note says otherwise.

**Forward, which is not.** An older release does not read a newer file.
A reader that meets a version it does not know reports the version rather
than misparsing. Sections are therefore added, never removed and never
repurposed: a section name means one thing for the life of the format.

The asymmetry is why the fixture set of `fixtures/` is one-directional.
Fixtures prove the backward direction. Nothing can prove the forward
direction, because there is nothing to prove: it is a refusal.

## 2. The versions

| Version | Adds | Optional | Recoverable from an older file |
|---|---|---|---|
| v1 | Mapper edges and offsets; per-node feature, threshold, children, value | n/a | n/a |
| v2 | Missing-value routing: per-feature missing bins, per-node `default_left` and `missing_bin` | The `monotone` section, the `categorical` section, and per-tree `cat` sections | No. A v1 model was trained without missing support and routes nothing |
| v3 | Per-node covers (`Tree.count`) | No, unconditional | No. Covers are training row counts and a fitted tree does not carry them |
| v4 | Per-node split gains (`Tree.split_gain`); a presence flag on covers | The `feature_names` section | No. The gradient sums a gain was scored from are gone by the time the tree exists |

## 3. The rules, per step

### R1. An older file loads as what it is, never as what it would have been

A v1 file describes a model trained without missing support. It routes no
missing values because the model it describes routed none. A v2 file
carries no covers because the model it describes never recorded them.

A reader must not synthesize the absent fact. Filling covers with zeros,
or gains with zeros, and then presenting them as data is the failure this
rule exists to prevent, and v3 committed a version of it; see R4.

### R2. An absence is testable, not silent

Every fact a version added is reachable through a flag a consumer can
branch on, rather than through a sentinel value it has to know about.

| Absent fact | How a consumer tests for it |
|---|---|
| Split gains | `has_split_gain` in a dump; `false` for v1, v2, and v3 |
| Node covers | `has_node_count` in a dump; `false` for v1 and v2 |
| Feature names | `load_feature_names` returns nothing; the consumer falls back to `Column_0`, `Column_1`, ... |
| Missing routing | v1 only, and it is not separately flagged; `model_format_version == 1` is the test |
| Monotone constraints | The section's absence loads as unconstrained, which is a real state and not an absence |

Missing routing is the one gap in this rule. A consumer must compare
`model_format_version` rather than read a flag. That is not worth a format
change to fix, and it is recorded here so nobody later concludes the flag
was lost.

### R3. Asking for an absent fact raises; it does not return a plausible number

A v1 or v2 model asked for feature contributions raises, because exact
contributions condition on covers and the model has none. A v3 model's
gain importance is the case that was gotten wrong: before v4, gains were
dropped on save, so every loaded model reported zero gain importance
rather than refusing. Zero is a plausible number and it was wrong.

The rule generalizes: when a version adds a fact, the release that adds it
also fixes every path that previously returned a plausible value in its
absence, so that the fix lands with the format change rather than a
version later.

### R4. A re-save is lossless in the only sense available to it

Loading a file of version N and saving it at version M > N must not
fabricate the facts N did not carry.

This is the rule v3 broke and v4 fixed. A model loaded from a v1 or v2
file has no covers. v3's writer wrote its zeros as if they were covers,
producing a file v3's own reader then rejected. v4 records "no covers
here" instead, so a re-save of an old model round trips.

**A re-save is therefore an upgrade in version and not in information.**
A v1 file re-saved by a v4 build is a v4 file that still has no missing
routing, no covers, and no gains. It is not more useful than it was; it is
only readable by fewer old builds. There is no reason to run a bulk
re-save, and one is not part of any migration.

### R5. Byte stability within a version

Within one version, a model that uses none of an optional section's
machinery serializes to exactly the bytes it did before that section
existed. A model with no categorical features writes no categorical
section. A model with no monotone constraints writes no monotone section.
A model saved without feature names writes no `feature_names` section.

A change that alters the bytes of a model that uses none of the new
machinery is a breaking change even though every file still loads. This
is what makes a checksum a usable fixture invariant at all: without R5,
a fixture's checksum would change on every unrelated feature and the
manifest would be noise.

### R6. What a new version may and may not do

A version bump may:

- add a section, optional or unconditional
- add a field to a section, at the end of that section's token run
- add a presence flag for a fact an older file could not carry

A version bump may not:

- remove a section or a field
- repurpose a section or a field name
- change the encoding of a value already written, including the
  IEEE-754-bits-as-decimal-`UInt64` convention for floats
- change what the magic or the kind token means

A change in the last group is not a version bump. It is a new format, and
it needs a new magic, so that an old reader fails at the first token
rather than at the tenth.

### R7. Five constants, one version

The format's version number appears in five places and they are not free
to disagree.

| Constant | File | Says |
|---|---|---|
| `_VERSION` | `src/mojoboost/serialize.mojo` | The token the writer writes |
| `CURRENT_FORMAT_VERSION` | same | That token as an integer |
| the `_read_version` chain | same | Which tokens the reader accepts |
| `MODEL_FORMAT_VERSION` | `src/mojoboost/model_dump.mojo` | What a dump reports |
| `SUPPORTED_MODEL_FORMAT_VERSIONS` | `python/mojoboost/inspection.py` | What the pure-Python fallback parser accepts |

The invariants are I2 through I5 in
[SNAPSHOT_SCHEMA.md](SNAPSHOT_SCHEMA.md) section 6, and
`tools/api_snapshot.py` enforces them. **Three of the five disagree in the
tree as it stands**; see [DRIFT_REPORT.md](DRIFT_REPORT.md), findings F1
and F2.

The reason for the enforcement rather than a comment is that four of the
five failures are silent. A dump that reports the wrong
`model_format_version` gives a consumer the wrong answer about which
optional facts a model can carry, and nothing raises.

## 4. What a version bump owes the fixture set

Adding a version is not complete until:

1. A fixture exists for the **new** version, covering each kind the format
   distinguishes (single-output and multiclass at minimum), added to
   `fixtures/manifest.toml` with its provenance and checksum.
2. Every **existing** fixture still loads and still predicts identically.
   The old fixtures are not regenerated. Regenerating them destroys the
   only evidence that the backward promise holds, and it is the single
   most likely mistake in this whole directory.
3. `SUPPORTED_MODEL_FORMAT_VERSIONS` in `python/mojoboost/inspection.py`
   includes the new version, or the pure-Python parser cannot read what
   the same build just wrote (invariant I5).
4. `MODEL_FORMAT_VERSION` in `src/mojoboost/model_dump.mojo` matches
   (invariant I4).
5. The read-back table in section 7.2 of the compatibility policy gains a
   row, which is release gate item B3.
6. This document gains a row in section 2 and, if the step is not a plain
   addition, a rule in section 3.

Item 2 is the one to be careful about. A fixture is evidence about a
release that already shipped. It has no maintainer, it is never improved,
and the correct response to a fixture that no longer loads is to fix the
reader, not the fixture.
