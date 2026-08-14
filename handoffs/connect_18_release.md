# Connect 18. Release packaging, platform matrix, and install experience

Static inspection, plus one command. Nothing here was built, installed, or
published. No wheel was produced and no workflow was dispatched. The single
exception is `packaging/matrix/validate_matrix.py`, a read-only
standard-library check that was run against this lane's own edits and passed;
it is recorded at the end of this document, including what it does not prove.
Every other claim below comes from reading files.

## What this lane found, and what it actually did

Five earlier lanes had already built the release machinery, and they built it
well. `python/pyproject.toml`, `python/setup.py`, `python/MANIFEST.in`, both
release workflows, both wheel builders, three validators, two smoke fixtures,
and four documents were all present and mostly consistent. This lane added
almost no new capability, because almost none was missing.

What was missing was a connection, and it was a single one with several
symptoms. **The machine-readable matrix had drifted out of step with the
repository it describes, and the drift ran in the direction that produces a
false artifact rather than a caught error.**

`packaging/matrix/platform_matrix.toml` declared exactly two Linux targets,
both tagged `manylinux_2_28`, both carrying `builder = "does not exist yet"`.
By the time this lane read it, all three of those facts were wrong.

## The call path, before

    release-linux.yml (build job)
      -> check_metadata_ready.py          ok
      -> validate_matrix.py               ok, matrix vs repository
      -> build_wheel_linux.sh             produces linux_x86_64 by default
      -> inspect_wheel.py                 ELF inspection
      -> (nothing)                        <-- the artifact met no matrix rule
      -> stage, upload

    packaging/matrix/platform_matrix.toml
      linux-x86_64-cp314   wheel_tag = cp314-cp314-manylinux_2_28_x86_64
      linux-aarch64-cp314  wheel_tag = cp314-cp314-manylinux_2_28_aarch64
      both                 builder   = "does not exist yet"

Three independent things were true at once and could not all be right.

1. `packaging/linux/build_wheel_linux.sh` exists, is complete, and is invoked
   at line 195 of `.github/workflows/release-linux.yml`. The matrix said no
   builder existed. `packaging/linux/README.md` and
   `packaging/matrix/smoke/clean_install_linux.sh` said the same thing in their
   own headers.
2. That builder reads `MOJOBOOST_TAG_POLICY`, **defaults to `plain`**, and
   emits `linux_<arch>`. The workflow's `tag_policy` input **also defaults to
   `plain`**. So a dispatch accepting every default produces
   `mojoboost-0.1.0-cp314-cp314-linux_x86_64.whl`, a filename that appeared
   nowhere in the matrix.
3. `validate_matrix.py` line 150 **required** every Linux row to carry a
   `manylinux_` tag, and failed the row otherwise. The honest artifact was not
   merely undeclared, it was unrepresentable. The rule that looked like the
   strict one was the rule forcing the matrix to name a manylinux file nobody
   builds.

The consequence is the part that matters. `validate_artifact.py` line 189 binds
a wheel to a target by exact `wheel_tag` string match, and rule R1 fails when
no target matches, with the note "Either the build is wrong or the matrix is out
of date." Every default Linux build would have hit that. It never surfaced only
because the Linux workflow never called `validate_artifact.py` at all, which the
macOS lane has done since it was written, through
`packaging/macos/build_release_wheel.sh` line 203.

`docs/INSTALLATION.md` was already correct and had been for a while. It
documented the plain `linux_x86_64` filename, the index rejection, and the musl
hazard. The install lane, the builder lane, and the workflow lane all agreed
with reality. The matrix was the only file still describing the old story, and
the validator was enforcing the old story against the other four.

## The call path, after

    release-linux.yml (build job)
      -> check_metadata_ready.py
      -> validate_matrix.py               matrix vs repository, now path-checked
      -> build_wheel_linux.sh             linux_<arch> by default, unchanged
      -> inspect_wheel.py
      -> validate_artifact.py             ADDED. wheel vs the target it claims
      -> stage (now also stages validate-<arch>.txt), upload

    packaging/matrix/platform_matrix.toml, four Linux rows
      linux-x86_64-cp314              cp314-cp314-linux_x86_64        publishable = false
      linux-aarch64-cp314             cp314-cp314-linux_aarch64       publishable = false
      linux-x86_64-cp314-manylinux    cp314-cp314-manylinux_2_28_x86_64   publishable = true
      linux-aarch64-cp314-manylinux   cp314-cp314-manylinux_2_28_aarch64  publishable = true
      all four                        status = designed, evidence = ""

Every row now names the script and the workflow that produce it, and both of
those fields are paths that get checked.

## Connections completed

1. **The matrix can now describe the artifact that gets built.** Four Linux
   rows instead of two. The bare id means the default artifact, which is the
   plain-tag wheel; the `-manylinux` suffix means the promoted one. A default
   build now matches `linux-x86_64-cp314` on the nose instead of failing R1.

2. **`publishable` is a declared field rather than an assumption.** Wheel rows
   must set it to true or false. `validate_matrix.py` now checks the tag
   against what the row claims about itself instead of demanding manylinux
   everywhere. A bare `linux_` tag is legal only on a row that admits no index
   will take it; a `manylinux_` tag may not sit on a row calling itself
   unpublishable. This is the "do not label wheels manylinux without
   justification" rule, expressed as a check rather than as prose.

3. **`builder_script` and `workflow` are new path-checked fields on every
   target.** This is the fix for the actual root cause. `builder` was prose,
   prose is never verified, and `"does not exist yet"` survived long after it
   stopped being true. Both new fields are in `PATH_FIELDS`, so the same drift
   now fails `validate_matrix.py` instead of quietly misinforming a reader.
   Wheel rows must fill both.

4. **The Linux workflow validates its artifact against the matrix.** Added the
   step the macOS lane always had. Its report is staged into the uploaded
   artifact next to the ELF inspection.

5. **The install checksum instructions now match what the release runs emit.**
   `docs/INSTALLATION.md` told users Linux emits "a .sha256 per wheel". It does
   not. `release-linux.yml` writes `SHA256SUMS-<arch>.txt` and its own
   clean-install job verifies with `sha256sum -c`. macOS writes `SHA256SUMS`
   via `packaging/macos/hash_artifacts.sh`. Both examples now use `-c` against
   the real manifest names, so the user runs the same check the workflow runs.

6. **Stale "no builder exists" claims removed from the three owned files that
   carried them**, being `packaging/linux/README.md`,
   `packaging/matrix/smoke/clean_install_linux.sh`, and the classifier
   rationale in `python/pyproject.toml`. In each case the replacement states
   the thing that is still true, which is that no Linux wheel has ever been
   built, rather than the thing that stopped being true.

7. **`docs/PLATFORM_MATRIX.md` reconciled with the TOML**, since
   `validate_matrix.py check_doc` requires every target id to appear there. The
   targets table gained the publishable column and the two new rows, and the
   Linux section now describes a builder that exists and a measurement that
   does not.

## What was deliberately not changed

- **The plain tag stays the default**, in the builder and in the workflow.
  Promoting to manylinux stays a deliberate dispatch input. This lane made the
  default representable, not publishable.
- **No status was promoted.** All four Linux rows and both macOS rows are still
  `designed` with `evidence = ""`. Nothing has been built or installed, so
  nothing earned `validated`.
- **Publication stays gated.** The TestPyPI job still requires all of
  `publish_testpypi`, the repository variable `MOJOBOOST_TESTPYPI_ENABLED`, and
  a manylinux tag. Production PyPI publishing still does not exist in either
  workflow, which is correct until clean-install evidence exists.
- **`installer_floor` on the plain rows says `unmeasured`** rather than
  repeating glibc 2.28. That number is what `pixi.lock` was solved against,
  which is a fact about the build environment and not a measurement of the
  shipped objects. The manylinux rows say "CLAIMED AND NOT YET MEASURED".
- **No runtime source, Python library module, test, or benchmark was touched.**

## Duplicates fused or quarantined

None found, and this is worth stating rather than skipping. There is no second
matrix, no competing platform registry, and no alternate wheel builder. The two
builders are genuinely two programs for two linkers, and
`packaging/macos/build_release_wheel.sh` explicitly documents at its line 9 that
it is not a second builder but a release entry point that reaches
`packaging/build_wheel.sh` through `pixi run -e pkg test-wheel`. The matrix now
records that distinction in the `builder` and `builder_script` fields rather
than collapsing it.

## A deviation from the prior lane's request, stated plainly

`handoffs/release_03_linux_wheels.md` section 5 asked for exactly this row
split and proposed the ids `linux-x86_64-cp314-plain` and
`linux-x86_64-cp314-manylinux`. This lane used `linux-x86_64-cp314` for the
plain row instead of `-plain`, on the reasoning that the bare id should mean
the artifact a default build produces, and that `docs/INSTALLATION.md`,
`docs/PYPI_RELEASE.md`, and `docs/LIGHTGBM_PARITY.md` already reference the
bare id and should keep pointing at the real default wheel. If the symmetry is
preferred, renaming is a mechanical change to the TOML plus the three documents,
and `check_doc` will catch any reference missed.

## Remaining disconnections

1. **No Linux wheel has ever been built.** The builder, the workflow, the
   validators, and the smoke fixture are all wired to each other and none has
   run. This is the largest remaining gap and it is not fixable by editing.
2. **The Linux glibc floor is unmeasured**, so neither `-manylinux` row may be
   published. R5f in `validate_artifact.py` and `container_elf_report.sh` in
   the clean-install job are the two places that would measure it.
3. **`bundled_dylibs` is unchecked on Linux.** `validate_artifact.py` R2b
   compares the declared set on macOS only. The Linux lists are the macOS names
   with a different extension and have never been confirmed against a real
   build, so on Linux that field is documentation and not a contract. Recorded
   in both the TOML notes and `docs/PLATFORM_MATRIX.md`.
4. **`macos-arm64-cp314-lowered` remains a hypothesis.** Whether the Mojo
   compiler honors a lowered deployment target is untested, and lowering the
   tag without lowering the binary's `minos` produces a wheel that installs and
   then fails to import. `setup.py` and the TOML both say so.
5. **Action pins.** `.github/workflows/release-linux.yml` still carries
   `actions/checkout@v4`, `actions/download-artifact@v4`, and
   `actions/upload-artifact@v4` with `# TODO(pin)` markers, while
   `release-macos.yml` uses `REPLACE_WITH_SHA` placeholders.
   `packaging/security/check_action_pins.py` and
   `packaging/macos/check_action_pins.py` exist to enforce pinning. Neither
   workflow is dispatchable as written, which is a safe failure, but it is a
   gap.
6. **`requires-python = ">=3.14"` is still under review.**
   `docs/PYTHON_SUPPORT.md` finds the toolchain publishes builds for 3.10
   through 3.14 and that 3.14 is what the solver chose rather than what the
   toolchain requires. Lowering it changes wheel tags, classifier rows, and the
   matrix python rows together, so it is one decision and not five. Not this
   lane's to make.

## Cross-lane patch requests

This lane edited nothing outside its ownership. Two requests, both small.

**To the owner of `packaging/macos/report_accelerator.mojo`.** Its line 19
notes that the file was proposed as `tools/report_accelerator.mojo`, which does
not exist. `packaging/macos/provenance.sh` line 67 correctly invokes the
`packaging/macos/` path, so nothing is broken. If the file is meant to move to
`tools/`, `provenance.sh` and `packaging/macos/README.md` move with it.

**To the owner of `docs/LIGHTGBM_PARITY.md`.** Its Linux wheel row states
"`docs/PLATFORM_MATRIX.md` still says `designed`". The rows are still
`designed`, so the sentence remains true, but the same row's claim that the
default tag policy is plain is now reflected in the matrix rather than
contradicting it. Suggested replacement for the trailing clause, being "the
matrix now carries both tag policies as separate rows, the plain rows marked
`publishable = false`; all four remain `designed` because no Linux wheel has
been built."

## An incident to record

Partway through this lane, another agent's commit `dc21f03` ("Connect
accelerator and public API foundations") swept up this lane's in-progress edits
to `packaging/matrix/platform_matrix.toml`,
`packaging/matrix/validate_matrix.py`, `docs/PLATFORM_MATRIX.md`,
`docs/INSTALLATION.md`, and `.github/workflows/release-linux.yml`, evidently
through a broad `git add`. This lane did not commit anything and did not amend,
revert, or restage that commit.

All content was verified present on disk afterwards and is intact. The only
practical effect is that these packaging changes are attributed to a commit
whose message describes unrelated accelerator work. Anyone bisecting release
packaging should know to look there. `packaging/matrix/smoke/clean_install_linux.sh`
and this handoff were edited after that commit and remain uncommitted.

## Serialization and public API effects

None. No model format, no serialized field, no Python symbol, and no default
parameter value was touched. `python/mojoboost/` was not edited. The only
metadata change in `python/pyproject.toml` is a comment block explaining why no
Linux operating-system classifier is claimed; the classifier list, the
dependency lists, `requires-python`, the version, and the package-data patterns
are byte-for-byte unchanged.

The wheel's contents and tag are unchanged on both platforms. What changed is
which artifacts the matrix admits exist and which checks run against them.

## Risks

- ~~The new `validate_matrix.py` rules were not executed.~~ **Retired.** They
  were run and passed, on all eight targets. See the command section below.
- **`publishable` widens what the matrix accepts.** A future Linux row could
  now legally carry a bare `linux_` tag. That is the intent, and the guard is
  that such a row must declare itself unpublishable and the TestPyPI job
  refuses it, but it is a loosened rule and worth knowing.
- **The added workflow step runs before staging and is not `if: always()`.** A
  validation failure now blocks the upload of that architecture's wheel, which
  is the intended behavior for a release gate and matches how the existing
  inspection step behaves, but it does mean a matrix error stops a build later
  than `validate_matrix.py` would.

## Smallest later commands

### Command 1, RUN, and it passed

The matrix contract check was run in the session that produced this handoff,
after the repository owner lifted the static-inspection restriction for this one
read-only, standard-library, no-network command.

```sh
python3 packaging/matrix/validate_matrix.py
```

```
release matrix ok: 8 targets, 3 source installs, 25 devices, 1 with any recorded evidence
```

This is the only executed evidence behind anything in this document, and it is
worth being precise about what it does and does not establish.

**What it proves.** The TOML parses, so both new `[[target]]` blocks and their
multi-line notes are well formed, and the target count of 8 is the expected 6
plus the two new `-manylinux` rows. The edited `validate_matrix.py` compiles and
every rule in it ran. The new `builder_script` and `workflow` fields on all
eight targets resolve to files that exist in the tree, which is the check that
would have caught the original `builder = "does not exist yet"` drift. The new
`publishable` rules accept the four Linux rows, meaning the two plain rows carry
`publishable = false` alongside their bare `linux_` tags and the two manylinux
rows do not. `check_doc` found both new target ids in
`docs/PLATFORM_MATRIX.md`, so the TOML and that document are back in step.

**What it does not prove.** Nothing about a wheel, a build, a platform, or any
performance or parity claim. It reads files and compares them to each other. No
artifact exists for it to have inspected, and no status in the matrix was
promoted on the strength of it. Every target is still `designed` with empty
evidence.

### Commands 2 and 3, still UNRUN

```sh
# Does the Linux workflow still parse as YAML, after the added step?
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release-linux.yml'))"

# Only after a Linux wheel exists, and expected to pass now that the
# plain-tag rows are declared. This is the check that would have failed
# against a default build before this lane.
python3 packaging/matrix/validate_artifact.py python/dist/*.whl
```

The YAML parse is the remaining cheap one. The added step was written to match
the indentation and shape of the two steps around it, and `git diff --check` is
clean, but no parser has read the file. `validate_artifact.py` was not edited by
this lane and cannot run until a Linux wheel exists.
