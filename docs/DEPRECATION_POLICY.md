# Deprecation policy

How a public surface in mojotrees is retired: the states it passes
through, the overlap it is owed, where the fact is recorded so a tool can
read it, and what the release gate does with it.

## Status of this document

Section 3 of [docs/COMPATIBILITY_POLICY.md](COMPATIBILITY_POLICY.md) is
the contract. It says which surfaces are public, how long a deprecation
runs, and what a break note contains. **This document does not set,
lengthen, or shorten any period it states.** Where a period appears below
it is quoted from there, and if the two ever disagree, that document wins
and this one is the bug.

What this document adds is the mechanism: a register with a defined
schema, four states a deprecation can be in, the rule that decides when a
state may advance, and the checks that read the register so that "we said
we would remove it in 0.4" is a fact a tool can fail on rather than a
sentence in a release note nobody grepped.

mojotrees is at version 0.1.0. There is no git tag, no published release,
and no PyPI distribution. Every rule here takes effect at the first tagged
release. Before that, nothing has shipped, so nothing can be owed an
overlap; section 6 says what that does and does not license.

## 1. The overlap floor

**An ordinary deprecation gets at least one full release of overlap.** The
old surface and its replacement both work, in the same release, for at
least one release before the old one goes. A surface that is announced and
removed in the same release has not been deprecated; it has been broken,
and it is governed by the break rules rather than by this document.

The floor is one release. It is a floor and not the period. Section 3.1 of
the compatibility policy sets longer periods for every surface it names,
and those are what actually apply:

| Surface | Minimum notice | Calendar minimum |
|---|---|---|
| Python name, argument, or documented behavior | 2 minor releases | 90 days |
| Mojo export | 2 minor releases | 90 days |
| C ABI declaration | 1 major release | 180 days |
| Parameter name or alias | 2 minor releases | 90 days |
| Model format section | Never removed | n/a |
| Environment variable | 2 minor releases | 90 days |

Two consequences worth stating plainly, because both have been guessed
wrong elsewhere.

1. **Two minor releases means two, counted from the release that shipped
   the warning.** A name deprecated in 0.3.0 is removable in 0.5.0, not in
   0.4.0. The release that carries the announcement is not one of the two.
2. **A patch release never removes anything.** Not a name, not an alias,
   not an environment variable, not a status code. A patch release may add
   a deprecation warning to a path that did not have one, because that is
   additive, and the clock starts at the minor release it lands in rather
   than at the patch.

Before 1.0 the calendar minimums are advisory rather than binding, because
the release cadence is not established. The release counts are binding
now. This is the compatibility policy's rule (section 3.1) and not a
relaxation invented here.

## 2. The four states

Every entry in the register is in exactly one state, and the states only
move forward.

| State | The surface | The caller sees | Register carries |
|---|---|---|---|
| `active` | Works, and is the recommended spelling | Nothing | Nothing; unregistered surfaces are `active` by definition |
| `soft` | Works, unchanged, and is documented as going | A docstring note and a release note. No runtime warning | `since`, `replacement`, `reason` |
| `deprecated` | Works, unchanged | A `DeprecationWarning` naming the replacement and the removal release, where the surface has a warning channel | `since`, `remove_in`, `replacement`, `reason` |
| `removed` | Gone | The error of section 4 | everything above, plus `removed_in` |

`soft` exists because two of mojotrees's three surfaces have no runtime
warning channel. A Mojo export cannot raise a `DeprecationWarning` without
a cost in a loop that may be hot, and a C declaration cannot warn at all
until the caller recompiles. For those, `soft` and `deprecated` differ
only in the register and in the wording of the note, and the clock runs
the same.

A `soft` entry that has never been `deprecated` still serves the full
period. Skipping the warning does not shorten the overlap; it changes only
where the announcement is visible.

### 2.1 What may advance a state

- `active` to `soft` or `deprecated`: any minor release. This is additive.
- `soft` to `deprecated`: any minor release. Does not restart the clock,
  because the clock started at `since`.
- `deprecated` or `soft` to `removed`: only when **both** the release count
  and, after 1.0, the calendar minimum have elapsed, **and** the removal
  lands in a release whose kind the surface's row permits. A Python name
  may go in a minor release before 1.0 and in a major release after; a C
  declaration may go only in a major release, ever.
- Anything to `active`: allowed, and it is how a deprecation is withdrawn.
  Withdrawing one is additive and needs a release note saying so, because
  callers who migrated on the announcement need to know the old spelling
  is staying.

There is no state for "removed in the next release, we promise". Either
the register says `remove_in` and the gate enforces it, or the surface is
not deprecated.

## 3. The register

`compatibility/deprecations.toml` is the machine-readable record. One
`[[deprecation]]` table per surface. It is read by
`tools/api_snapshot.py --check`, which is what makes the dates and the
release numbers enforceable rather than aspirational.

### 3.1 Required keys

| Key | Type | Meaning |
|---|---|---|
| `id` | string | Stable, unique, kebab-case. Never reused, never renamed, cited by release notes |
| `surface` | string | One of `python`, `mojo`, `c_abi`, `parameter`, `env`, `cli`, `model_format`, `dump_schema` |
| `name` | string | The exact spelling being retired, as it appears in the snapshot |
| `state` | string | `soft`, `deprecated`, or `removed` |
| `since` | string | The version that first announced it. The clock starts here |
| `replacement` | string or `""` | The spelling to migrate to. Empty only when the exemption of section 5 applies |
| `reason` | string | One sentence. Why it is going, not what it did |
| `migration` | string | What a caller changes. A sentence, or a before/after pair |

### 3.2 Conditional keys

| Key | Required when | Meaning |
|---|---|---|
| `remove_in` | `state = "deprecated"` | The earliest release the removal may land in. Must satisfy section 1 against `since` |
| `removed_in` | `state = "removed"` | The release that removed it. Must be at or after `remove_in` |
| `exempt_reason` | `replacement = ""` | Which clause of section 5 applies, verbatim |
| `announced_at` | always, once dates are meaningful | ISO 8601 date of the release that carried `since`. The calendar minimum is computed from this, and it is what makes the 90-day rule checkable after 1.0 |

`announced_at` is a date and not a release number on purpose. Release
numbers do not carry time, and the calendar half of section 3.1 cannot be
checked from a version string alone.

### 3.3 Candidates, which are not deprecations

The register carries a second array, `[[candidate]]`. A candidate is a
surface whose shape is unresolved and where resolving it after the first
tagged release costs a deprecation window that resolving it before costs
nothing.

Nothing about a candidate is announced. It does not warn, no clock runs,
and it is not a promise to anyone outside the project. It exists so that
"we meant to settle that" is a row rather than a memory.

| Key | Meaning |
|---|---|
| `id` | Stable, kebab-case, as for a deprecation |
| `surface`, `name` | As for a deprecation |
| `owed_by` | The release the decision is owed by. Today, always the first tagged release |
| `gate_item` | The release gate item that fails if it is not settled, where one exists |
| `question` | The decision, stated so that either answer is a sentence |
| `current` | What the tree does today |
| `why_now` | What it costs to defer. If deferring costs nothing, it is not a candidate |

`tools/api_snapshot.py --check` reports the candidate count and does not
fail on it. Failing is the release gate's job, through the item each
candidate names.

A candidate is resolved by deleting the row, in the same commit as the
change or the decision to keep things as they are. A candidate resolved in
favor of changing the surface produces a `[[deprecation]]` row if the
change lands after the tag, and no row at all if it lands before.

### 3.4 What the register is not

It is not a changelog and it is not a list of everything that ever
changed. Only surfaces that are public under section 2 of the
compatibility policy appear in it. An internal helper that was renamed
does not get an entry, because it was never owed an overlap.

It is also not the source of truth for whether the surface still exists.
The snapshot is. The register says what was promised; the snapshot says
what is there. The gate is what compares them, and a disagreement between
them is the bug the pair exists to catch.

## 4. What a removed surface does

A removal is not silence. For as long as the message is useful, and at
minimum through the major release that follows the removal:

- **Python.** The name raises rather than being absent, where the surface
  admits it. An argument removed from a signature raises `TypeError` from
  an explicit check naming the replacement, not from Python's own
  unexpected-keyword message. A module-level name removed from `__all__`
  is answered by `__getattr__` with an `AttributeError` that names the
  replacement.
- **Parameters.** A removed parameter name is reported with its
  replacement, not with the generic unknown-key message. This is the
  compatibility policy's section 3.2 rule and the register is what lets
  the parser's message table stay right.
- **C ABI.** Nothing. A removed declaration is gone from the header, the
  ABI version has been bumped, and a caller that was linked against the
  old one fails to load. That is the intended outcome and it is why the C
  period is a major release rather than two minors.
- **Environment variable.** Reading a removed variable and finding a value
  reports that the variable no longer does anything, once, and names the
  replacement. Silently ignoring a variable a user set is the failure this
  clause exists to prevent.

## 5. Exemptions

The overlap of section 1 assumes there is something to migrate to. It is
waived, with `exempt_reason` in the register and the reason stated in the
release notes, when one of these is true. The list is the compatibility
policy's section 3.3, restated so the register's field has a closed
vocabulary to draw from.

| `exempt_reason` | Means |
|---|---|
| `never_released` | The surface never appeared in a tagged release. Nothing can have been written against it |
| `always_raised` | The behavior was documented as unavailable and raised on every call, so no working program depends on it |
| `silently_wrong` | Keeping it would produce silently wrong numbers. A wrong answer is worse than a break, and this is the only clause that can justify removing something that works |

`silently_wrong` is the one that will be reached for wrongly. It covers a
surface whose output is incorrect, not one whose output is inconvenient,
undesirable, or slower than it should be. A performance regression is not
`silently_wrong`. A default whose value someone now disagrees with is not
`silently_wrong`; that is a default change, governed by section 4.3 of the
compatibility policy, and it gets a major release.

## 6. Before the first tagged release

Nothing has shipped. Until there is a tag, the register is empty of
ordinary entries and every removal is `never_released`.

That is honest, and it is also the thing most likely to be abused, so two
rules bound it.

1. **The exemption expires at the tag.** The moment a release is tagged,
   every public surface in it is owed the full overlap. There is no
   grace period after the tag and no "we only just shipped it" clause.
2. **The register is populated before the tag, not after.** Any surface
   the project already intends to retire gets its `soft` entry in the
   release that tags, with `since` set to that release. Discovering after
   1.0.0 that something should have been deprecated in 1.0.0 costs a major
   release, and the way to not pay that is to write the entries while the
   exemption still covers the mistake.

Section 5 of this document and section 1.3 of the compatibility policy are
the whole of the pre-1.0 flexibility. A 0.x minor release may break a
documented surface, and every such break is listed in the release notes
with its migration. That is a narrower promise than SemVer's 0.x, which
promises nothing, and it is deliberately narrower.

## 7. Model formats, which are not deprecated

The model format never deprecates a section. Section 7.3 of the
compatibility policy is unconditional: a section name means one thing for
the life of the format, sections are added and never removed, and a
release reads every file any earlier release wrote.

So the model format has no `soft` state and no removal window. What it has
instead is a migration rule per version step, and those rules are in
[compatibility/MODEL_FORMAT_MIGRATIONS.md](../compatibility/MODEL_FORMAT_MIGRATIONS.md),
which is normative for them. A `model_format` row may appear in the
register only in the `removed` state and only under the `silently_wrong`
exemption, which is to say only if a released format version was found to
encode something incorrectly. No such row exists and none is expected.

The same holds for `dump_schema`, one level up: `DUMP_FORMAT_VERSION`
bumps when a key is removed, retyped, or redefined, and a consumer must
ignore keys it does not know. A dump key that is going gets a register
entry like any other public name, because a consumer can be written
against it.

## 8. The gate

These are additions to section 12 of the compatibility policy, in its
numbering scheme, and they are requested rather than made: that document
is not this lane's to edit.

- **C6.** `tools/api_snapshot.py --check` green. The public surface matches
  the snapshot, or every difference has been classified and accepted.
- **C7.** Every register entry whose `remove_in` is at or below the release
  being cut is either removed with a break note, or moved out to a later
  `remove_in` with the extension in the release notes. Silently letting a
  `remove_in` pass is the failure this item exists to catch.
- **C8.** No register entry has a `remove_in` that would violate section 1
  against its `since`.
- **C9.** `tools/model_fixture_manifest.py --check` green: every fixture
  the manifest names exists, its checksum matches, and every model format
  version the current build can write has a fixture.
- **C10.** Every surface removed since the previous release has a register
  entry in state `removed` with `removed_in` set to the release being cut.

Item C7 is the one that matters most, because it is the only one that
fails on inaction. Every other item fails on a change somebody made.
