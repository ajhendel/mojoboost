# Handoff: self-contained macOS arm64 wheels

Release lane 02 of a parallel round. This lane owns `packaging/macos/`,
`.github/workflows/release-macos.yml`, and this file. Everything it needed to
change anywhere else is written out below, exactly, and **none of it has been
applied**.

## What this lane added

| Path | What it is |
|---|---|
| `packaging/macos/README.md` | What the directory is, what it reuses, and the code signing and notarization decision |
| `packaging/macos/build_release_wheel.sh` | The release build: preflight, build through the existing builder, provenance, three checkers, hashes |
| `packaging/macos/provenance.sh` | Writes `<wheel>.provenance.json` and refuses a build whose accelerator answer and Metal toolchain disagree |
| `packaging/macos/inspect_wheel.py` | Describes the wheel and checks the release-only rules the matrix does not cover |
| `packaging/macos/hash_artifacts.sh` | Writes and verifies `SHA256SUMS` |
| `packaging/macos/check_action_pins.py` | Every third-party action in the release workflow is a commit hash with a version comment |
| `packaging/macos/report_accelerator.mojo` | Prints the build's compile-time `has_accelerator()`, for the provenance sidecar |
| `.github/workflows/release-macos.yml` | The release: tagged commit, native Apple silicon runner, artifacts, opt-in TestPyPI publish |

## What was executed

**Nothing.** Not one command in this lane produced or touched an artifact.

No wheel was built. No wheel was inspected. No hash was computed. No Mach-O
object was parsed. No Mojo file was compiled, including
`packaging/macos/report_accelerator.mojo`, which has never been through the
compiler. No pixi task ran, no test ran, no workflow was dispatched, no runner
was registered, and nothing was uploaded to TestPyPI, PyPI, or any other index.
`codesign`, `install_name_tool`, `otool`, `lipo`, `pip`, and `python` were not
run.

Every check ID in `inspect_wheel.py` (C1 through C13) names a future result.
Every command in the "Exact later commands" section below is unverified.

What was actually done is reading: `packaging/build_wheel.sh`,
`packaging/test_wheel.sh`, `packaging/matrix/` in full, `python/pyproject.toml`,
`python/setup.py`, `pixi.toml`, both existing workflows, `.gitignore`,
`src/mojoboost/device.mojo`, and `handoffs/task18_platform.md`.

## The shape of the release

```
tag v0.1.0 exists on a commit
        |
        v
workflow_dispatch(ref=v0.1.0)  or  push of the tag
        |
        v
[build job, self-hosted macOS arm64 with Metal]
  action pins are commit hashes      check_action_pins.py
  runner is native Apple silicon     no Rosetta, no mojo on PATH, python3.14 present
  HEAD is a tag and matches version  git describe, python/pyproject.toml
        |
        v
  packaging/macos/build_release_wheel.sh
        |-- SOURCE_DATE_EPOCH from the commit
        |-- pixi run -e pkg test-wheel         (the existing builder, then the
        |                                       existing two-venv wheel test)
        |-- packaging/macos/provenance.sh      (sidecar + accelerator/Metal gate)
        |-- packaging/matrix/validate_matrix.py     \
        |-- packaging/matrix/validate_artifact.py    > all three run; the exit
        |-- packaging/macos/inspect_wheel.py  (C1..C13)  status is collected
        |-- otool record
        `-- packaging/macos/hash_artifacts.sh  (SHA256SUMS)
        |
        v
  clean install, outside pixi        packaging/matrix/smoke/clean_install_macos.sh
        |
        v
  re-hash, so the manifest covers the clean install record
        |
        v
  two artifacts, 90 day retention    `wheel` alone; `release-metadata` separately
        |
        v
[publish job, only if publish_testpypi was ticked]
  round-trip hash check against the build output and SHA256SUMS
  TestPyPI trusted publishing, id-token: write, environment `testpypi`
```

There is no PyPI job. Not a disabled one, not a commented one.

## Decisions, and why

**One builder, not two.** `packaging/build_wheel.sh` stays the builder.
`build_release_wheel.sh` reaches it through `pixi run -e pkg test-wheel`, which
depends on `build-wheel`, so the release also runs `packaging/test_wheel.sh`
rather than skipping it. That split matters: `test_wheel.sh` answers "does this
wheel work", the three checkers answer "does it say true things about itself",
and the clean-install fixture answers "does it work without the toolchain". A
release that runs only the middle one has checked the label and not the
contents. Two builders that can produce a file with the same name and different
contents is the exact failure `packaging/matrix/` was written to prevent, and it
would have been the easy thing to do here.

**All three checkers run, and the exit status is collected.** Aborting on the
first failure would leave the artifact directory with less evidence in it than a
passing run produces, which is backwards: a failing release is exactly when the
inspection report, the otool dump, and the hashes are worth having. The script
still exits non-zero, so the workflow still fails.

**`SOURCE_DATE_EPOCH` is set from the commit.** Zip entry timestamps otherwise
come from the wall clock, so two builds of one commit differ in every member
header and therefore in the wheel's digest. `docs/PYPI_RELEASE.md` names this
file as the place that gap lives, and `packaging/linux/build_wheel_linux.sh`
already does the same thing. It removes one source of nondeterminism and not all
of them: whether the Mojo compiler emits a byte-identical extension across two
builds of one commit is untested, so the macOS wheel still must not be described
as reproducible, and two matching digests still prove nothing.

**"Bundle only what is required" is enforced as a check, not as a discovery
step.** The builder bundles a hardcoded list of four MAX dylibs.
`inspect_wheel.py` C4 fails when anything in the wheel needs a library that is
not bundled, and C5 fails when a bundled library is not needed by anything. That
covers both directions without a second implementation of the bundling logic,
and when C5 fires the answer might be that the matrix's `bundled_dylibs` list is
what is wrong. That decision is not this script's to make.

**Truthful platform tags are a two-variable contract, and this lane owns the
verification half.** `python/setup.py` (lane 01) writes the tag from
`MOJOBOOST_MACOS_DEPLOYMENT_TARGET` and deliberately does not read
`MACOSX_DEPLOYMENT_TARGET`, stating that keeping tag and binary in step is the
release procedure's job. `build_release_wheel.sh` therefore sets both variables
from one input, and `inspect_wheel.py` C1 requires the tag's floor to *equal* the
extension's `LC_BUILD_VERSION` minos. Equality, not the matrix's
`minos <= floor`: a wheel built for macOS 12 and tagged `macosx_26_0_arm64`
passes rule R5b and is still wrong to publish, because pip then refuses it on
every Mac in between.

**Ad-hoc signing is necessary; Developer ID and notarization are not.** The full
argument is in `packaging/macos/README.md`. In short: on Apple silicon every
Mach-O image needs a valid signature to be mapped and `install_name_tool`
invalidates one, so the existing ad-hoc re-sign is load-bearing and C3 checks it
survived. Notarization is enforced against quarantined code, pip does not
quarantine what it extracts, and a `.whl` is not a container a ticket can be
stapled to. Adding Developer ID would mean a signing identity and App Store
Connect credentials living next to a self-hosted runner, which is the credential
posture `docs/RELEASE_SECURITY.md` exists to avoid. Two things flip this and
neither is true today: shipping a downloadable installer or a standalone binary,
and the quarantine assumption below turning out to be wrong.

**The runner is self-hosted because the build machine decides what the wheel
says about GPUs.** `has_accelerator()` is resolved at compile time, so a wheel
built where an accelerator was visible reports one as available everywhere it is
installed. A GitHub-hosted Apple silicon runner has a visible accelerator and no
Metal toolchain, which is the one combination that cannot be reasoned about.
`provenance.sh` records the answer and exits 3 on that combination.

**TestPyPI only, opt in per run, never on a tag push.** The publish job is
skipped unless `publish_testpypi` was ticked on a `workflow_dispatch`. It is the
only job with `id-token: write`, it holds no `contents` permission at all, and it
re-verifies the artifact's SHA-256 against both the build job's output and
`SHA256SUMS` after the artifact-storage round trip, before handing the directory
to anything that can upload.

**Action pins are placeholders, deliberately.** This lane had no network access.
A fabricated hash would look right and be wrong; `@REPLACE_WITH_SHA` fails
loudly. It is the spelling `.github/workflows/release-provenance.yml` (lane 10)
already uses, so one pass fills in both files.

## Cross-lane reconciliation

Read this before integrating the round. `.github/workflows/release-provenance.yml`
(lane 10) landed in the same round and overlaps this lane substantially. Both
files were read at the time of writing; neither lane could edit the other.

**The overlap, item by item.**

| This lane | Lane 10 | Same thing? |
|---|---|---|
| `release-macos.yml` build job, `[self-hosted, macos, arm64, metal]` | `release-provenance.yml` build job, same labels | Yes. Two jobs competing for one runner that does not exist yet |
| `packaging/macos/hash_artifacts.sh` | `packaging/security/hash_manifest.py` | Yes. Both write `python/dist/SHA256SUMS` |
| `packaging/macos/check_action_pins.py` | `packaging/security/check_action_pins.py` | Yes, including the `@REPLACE_WITH_SHA` convention |
| `packaging/macos/provenance.sh` writes the sidecar | Lane 10's build job requires the sidecar to already exist | No. These are complementary, and see the defect below |
| TestPyPI publish job, environment `testpypi` | TestPyPI **and** PyPI publish jobs, environment from an input | Overlapping, and mutually exclusive in practice |
| SBOM and attestations | `provenance` job | Lane 10 only. Nothing here duplicates it |

**The defect this lane found in lane 10's workflow.** Its build job runs
`pixi run -e pkg test-wheel` and then has a step asserting that
`<wheel>.provenance.json` exists, with a message pointing at
`handoffs/task18_platform.md` edit 4. That edit has not been applied:
`packaging/build_wheel.sh` does not write a sidecar, and nothing else in that
job does either. As committed, lane 10's build job fails at that step on every
run. The one-line fix, in lane 10's file, is to call the script that does write
it:

```yaml
      - name: Provenance sidecar
        run: packaging/macos/provenance.sh "$(ls python/dist/mojoboost-*.whl)"
```

Which also picks up the accelerator and Metal consistency gate, which that job
otherwise approximates with `xcrun --find metal` alone. Not applied here:
`.github/workflows/` outside `release-macos.yml` is not this lane's.

**The publishing conflict, which has to be decided before either file
publishes.** A PyPI or TestPyPI trusted publisher is bound to a repository, a
workflow *filename*, and an environment name. Two workflows with publish jobs
are two publisher identities. `docs/PYPI_RELEASE.md` (lane 01) tells the owner
to register the one belonging to `release-provenance.yml`. So today, dispatching
this lane's workflow with `publish_testpypi` ticked would fail at upload with a
token TestPyPI does not accept, and registering both would widen the publishing
surface for no gain.

**Recommended resolution, and what each choice costs.** Keep *this* lane's build
job and *lane 10's* attestation and publish chain. That is not lane loyalty: the
build job here does strictly more (tag and version agreement, Rosetta and PATH
preflight, `SOURCE_DATE_EPOCH`, the release-only inspection rules, the otool
record), and lane 10's `provenance` and `publish` jobs do things this lane has
nothing equivalent to (SBOM, build provenance attestation, a `pypi` path with an
environment gate). Concretely:

1. Delete the `build` job from `release-provenance.yml` and have its
   `provenance` job take `needs: []` with a `workflow_run` or dispatch input
   naming the run to attest. Or, simpler and probably better first: delete the
   `publish-testpypi` job from `release-macos.yml` and let this file be the
   builder and verifier only, with lane 10's file owning every upload.
   **This lane's artifact names were already changed to `wheel` and
   `release-metadata` to match what lane 10's jobs download**, so either
   direction is a deletion rather than a rewrite.
2. Keep one action-pin checker. `packaging/security/check_action_pins.py` is the
   better home, because pin policy is a security concern and lane 10's checklist
   already cites it. Then the guard step here becomes
   `python3 packaging/security/check_action_pins.py .github/workflows/release-macos.yml`
   and `packaging/macos/check_action_pins.py` is deleted.
3. Keep one hash manifest writer. `hash_manifest.py` covers the wheel and the
   sidecar; `hash_artifacts.sh` also covers the inspection report, the otool
   dump, and the clean-install record. Either extend the Python one to take a
   directory or keep the shell one; do not run both, because the second to run
   silently rewrites `SHA256SUMS` with a different file list.
4. Whichever workflow keeps the publish job, its filename and environment names
   are what the owner registers with the index. Changing the filename afterwards
   breaks publishing until the publisher is re-registered, which
   `docs/PYPI_RELEASE.md` already says.

None of the above is applied. Every file named in it except this lane's three
belongs to another lane.

## Required edits outside this lane

Each is written out ready to apply. None has been applied.

### E1. `pixi.toml`, two tasks in `[feature.pkg.tasks]`

Next to the existing `build-wheel` and `test-wheel`. Deliberately no task for
`build_release_wheel.sh` itself: it refuses to run with `CONDA_PREFIX` set,
because it calls pixi and a nested environment changes which interpreter and
which libraries the build sees.

```toml
# Describes a built wheel and checks the release-only rules: exact platform
# tag, bundle minimality, install names, and source-tree leakage. Complements
# packaging/matrix/validate_artifact.py rather than repeating it.
inspect-wheel = { cmd = "python3 packaging/macos/inspect_wheel.py python/dist/*.whl", depends-on = [
    "build-wheel",
] }

# Writes SHA256SUMS over whatever release artifacts are in python/dist.
hash-artifacts = "packaging/macos/hash_artifacts.sh python/dist"
```

### E2. `.github/workflows/ci.yml`, one job

Cheap, standard library, no build. It fails the moment a release workflow gains
an unpinned action, which is the only time it matters.

```yaml
  # Release workflows must pin every third-party action to a commit hash. A tag
  # is a moving pointer, and these workflows hold an OIDC identity.
  action-pins:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Release action pins
        run: python3 packaging/macos/check_action_pins.py
```

This will fail until the pins are filled in. Add it after that, or add it now
and treat the red as the reminder. Do not add it and then fill the pins with
whatever the tag resolves to today without reading the release notes for the
version in the comment.

### E3. `packaging/matrix/smoke/clean_install_macos.sh`, one line

The fixture makes the wheel path absolute and then `cd`s into a temp directory,
but leaves the record path exactly as given, so a relative one is written
relative to a directory that does not contain it and `tee` fails. The workflow
works around it by passing `"$PWD/python/dist/clean-install.txt"`, which is a
workaround and not a fix. The fix, next to the existing `WHEEL=` line:

```sh
LOG=${2:-/dev/null}
[ "$LOG" = /dev/null ] || LOG=$(cd "$(dirname "$LOG")" && pwd)/$(basename "$LOG")
```

Not applied here: `packaging/matrix/` belongs to another lane and this is its
file. Unverified; found by reading, not by running the fixture.

### E4. `packaging/build_wheel.sh`, conditional on what C6a and C7 report

The builder rewrites the rpath of `_mojoboost.so` and re-signs everything, but
does nothing to the four copied dylibs' own load commands. Two of the release
checks look at those, and neither has ever run:

- **C7, install names.** If a copied dylib's `LC_ID_DYLIB` is an absolute path
  into the build machine's pixi environment, it ships that path. The repair is
  one line per library, before the re-sign:

  ```sh
  install_name_tool -id "@loader_path/$lib.dylib" "$PKG/.dylibs/$lib.dylib"
  ```

- **C6a, rpaths.** If a copied dylib carries an absolute `LC_RPATH`, the same
  applies with `-delete_rpath`/`-add_rpath`.

Whether either is needed is unknown, because nobody has looked. The command that
answers it, on the existing wheel in `python/dist` of whichever working copy has
one:

```sh
for f in python/mojoboost/.dylibs/*.dylib; do
    echo "== $f"; otool -D "$f"; otool -l "$f" | grep -A 2 LC_RPATH
done
```

Do not apply the edit before running that. If the install names are already
`@rpath/...`, adding a rewrite changes a working artifact for no reason.

### E5. `packaging/matrix/platform_matrix.toml`, two fields on one row

For the matrix owner, once this lane's workflow is real. The row is
`macos-arm64-cp314`, and the status stays `designed`: a builder existing is not
evidence.

```toml
builder = "packaging/macos/build_release_wheel.sh (release) or packaging/build_wheel.sh (local)"
release_workflow = ".github/workflows/release-macos.yml"
```

`release_workflow` is a new key. `validate_matrix.py` was not read closely enough
to know whether it rejects unknown keys; check before adding it, and drop it if
so. The `builder` change alone is safe.

### E6. No change needed, recorded so nobody looks for one

- **`.gitignore`.** `python/dist/` is ignored, which is correct, and it means
  `SHA256SUMS`, `inspection.json`, `otool.txt`, `clean-install.txt`, and the
  provenance sidecar are ignored too. They travel as workflow artifacts, which is
  where they belong. Do not un-ignore that directory to keep a release record.
- **`python/setup.py`.** Already reads `MOJOBOOST_MACOS_DEPLOYMENT_TARGET`, which
  is what `build_release_wheel.sh` sets. The two agree as written.
- **`python/pyproject.toml`.** Already declares `license-files = ["LICENSE"]`,
  which is what C10b checks for in the wheel.

## Settings only the repository owner can make

None of these are files, so none of them are in this lane's diff.

1. **Register the runner** with the labels `self-hosted`, `macos`, `arm64`,
   `metal`, on an Apple silicon Mac with Xcode, the Metal toolchain, and a
   Python 3.14 on `PATH` that did not come from this repository's pixi
   environment. The runner account must not have `mojo` on its `PATH`. The
   workflow fails on the first step if it does, which is correct: the
   clean-install fixture cannot prove anything next to the toolchain the wheel
   must not need.
2. **Fill in the action pins.** One command each, and read the release notes for
   the version in the trailing comment rather than taking whatever is newest:
   ```sh
   gh api repos/actions/checkout/commits/v4.2.2 --jq .sha
   gh api repos/prefix-dev/setup-pixi/commits/v0.10.1 --jq .sha
   gh api repos/actions/upload-artifact/commits/v4.4.3 --jq .sha
   gh api repos/actions/download-artifact/commits/v4.1.8 --jq .sha
   gh api repos/pypa/gh-action-pypi-publish/commits/v1.12.2 --jq .sha
   ```
   The version numbers in the comments are the versions this lane wrote down as
   intended, not versions it verified exist. Check each one resolves before
   pasting the hash. Then `python3 packaging/macos/check_action_pins.py`.
3. **Create the `testpypi` environment** in repository settings, with a required
   reviewer and a deployment branch/tag restriction. The job references it by
   name and will not run until it exists.
4. **Configure the TestPyPI trusted publisher** for the project, with the
   repository owner and name, workflow filename `release-macos.yml`, and
   environment name `testpypi`. The exact procedure, and the separation between
   reserving the name and making a production release, is lane 01's
   `docs/PYPI_RELEASE.md`.
5. **Decide which workflow holds the publisher identity**, before registering
   anything. See "Cross-lane reconciliation" above: `release-macos.yml` and
   `release-provenance.yml` both have publish jobs, an index binds a publisher to
   one workflow filename and one environment name, and `docs/PYPI_RELEASE.md`
   currently names the other file. Registering both is not the fix.
6. **Decide whether release tags must be signed.** The workflow prints the tag
   object and does not require a signature, because that is a security-posture
   decision belonging to `docs/RELEASE_SECURITY.md` (lane 10) rather than one
   this lane should invent.

## Exact later commands

**Every result below is unverified. None of these has been run.** They are
written for a Mac that is not the build machine, from a directory that is not the
source tree, using a Python that did not come from this repository's pixi
environment. Substitute the real filename for `$WHEEL`.

### Verify what you were given, before installing it

The run produces two artifacts, `wheel` and `release-metadata`, and `SHA256SUMS`
is in the second one while the file it describes is in the first. Put them in one
directory before checking, or `shasum -c` reports every line as missing:

```sh
cd /path/to/downloads
unzip -o wheel.zip -d artifacts
unzip -o release-metadata.zip -d artifacts
cd artifacts
shasum -a 256 -c SHA256SUMS
cat mojoboost-0.1.0-*.whl.provenance.json
```

`clean-install.txt` is listed in the manifest only when the release re-hashed
after the smoke step. A file absent from `SHA256SUMS` is not a failure: `-c`
checks the lines it is given and says nothing about files it was never told
about.

Expected: `OK` per file, and a sidecar whose `git_commit` is the tagged commit and
whose `has_accelerator_at_build` is `true` or `false` rather than `unknown`.

### Clean venv install

```sh
WORK=$(mktemp -d) && cd "$WORK"
test -z "$CONDA_PREFIX" || echo "WARNING: inside a conda or pixi environment; this proves nothing"
command -v mojo && echo "WARNING: mojo is on PATH; this proves nothing"
python3.14 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install --no-index --no-cache-dir "$WHEEL"
./venv/bin/pip list
```

`--no-index` is the point: if the wheel silently needs something it does not
declare, this is where it shows up instead of pip quietly fetching it.

Expected failure on the wrong machine: `ERROR: mojoboost-0.1.0-cp314-cp314-
macosx_26_0_arm64.whl is not a supported wheel on this platform.` That is the
correct refusal on an Intel Mac, on macOS older than the tag's floor, and on any
interpreter that is not CPython 3.14.

### Import

```sh
cd "$WORK"
./venv/bin/python -c "import mojoboost, sys; print(mojoboost.__version__); print(mojoboost.__file__); print(sys.version)"
```

Expected: `0.1.0`, a path inside `venv/lib/python3.14/site-packages/mojoboost/`,
and CPython 3.14. A path outside `site-packages` invalidates the whole run.

### Tiny fit, predict, save, load

```sh
cd "$WORK"
./venv/bin/python - <<'PY'
from mojoboost import MojoBoostRegressor

X = [[i / 20.0, (i % 5) / 5.0] for i in range(20)]
y = [3.0 * r[0] + r[1] for r in X]

m = MojoBoostRegressor(n_estimators=10, min_data_in_leaf=2, device="cpu").fit(X, y)
before = m.predict(X[:3])
print("device_:          ", m.device_)
print("n_features_in_:   ", m.n_features_in_)
print("predict:          ", before)

m.save("tiny.mbst")
again = MojoBoostRegressor.load("tiny.mbst")
after = again.predict(X[:3])
print("after load:       ", after)
print("round trip exact: ", list(after) == list(before))
PY
```

Expected: `device_` is `cpu`, `n_features_in_` is 2, and the round trip is exact.
The save format holds the model and not the estimator, so hyperparameters and
feature names do not come back; that is documented behavior, not a defect.

### Architecture verification

```sh
cd "$WORK"
SITE=$(./venv/bin/python -c "import mojoboost, os; print(os.path.dirname(mojoboost.__file__))")
file "$SITE/_mojoboost.so"
lipo -archs "$SITE/_mojoboost.so"
otool -l "$SITE/_mojoboost.so" | grep -A 4 LC_BUILD_VERSION
./venv/bin/python -c "import platform, sysconfig; print(platform.machine(), sysconfig.get_platform())"
sw_vers -productVersion
```

Expected: `Mach-O 64-bit bundle arm64`, `lipo` prints `arm64` and nothing else,
`minos` equals the wheel filename's platform tag floor exactly, and the
interpreter reports `arm64`. A universal binary here is a defect: this project
ships single-arch and `validate_artifact.py` rejects fat objects.

### Dependency inspection

```sh
cd "$WORK"
otool -L "$SITE/_mojoboost.so"
otool -l "$SITE/_mojoboost.so" | grep -A 2 LC_RPATH
ls -l "$SITE/.dylibs/"
for f in "$SITE"/.dylibs/*.dylib; do echo "== $(basename "$f")"; otool -D "$f"; otool -L "$f"; done
codesign -dv --verbose=2 "$SITE/_mojoboost.so" 2>&1 | grep -i signature
xattr -l "$SITE/_mojoboost.so" || true
```

Expected: every non-system dependency is `@rpath/...` and names a file present in
`.dylibs/`; the only rpath is `@loader_path/.dylibs`; every absolute dependency
is under `/usr/lib` or `/System/Library`; `codesign` reports `Signature=adhoc`;
`xattr` prints nothing, and in particular no `com.apple.quarantine`.

The same facts without Apple's tools, from the wheel rather than the install:

```sh
python3 packaging/macos/inspect_wheel.py "$WHEEL" --report-only
```

A load-time trace, which is the only one of these that observes rather than
declares:

```sh
DYLD_PRINT_LIBRARIES=1 ./venv/bin/python -c "import mojoboost" 2>&1 | grep -i -E 'mojoboost|AsyncRT|KGEN|MSupport'
```

Caveat, unverified: `DYLD_*` variables are stripped for hardened-runtime
processes, and a python.org or Homebrew interpreter may be one. If the trace is
empty, that is why, and it is not evidence about the wheel.

### Uninstall

```sh
cd "$WORK"
./venv/bin/pip uninstall -y mojoboost
./venv/bin/python -c "import mojoboost" ; echo "exit status $? (expect non-zero, ModuleNotFoundError)"
ls "$SITE" 2>&1 || echo "package directory gone, as expected"
cd / && rm -rf "$WORK"
```

Expected: `pip uninstall` removes the extension and `.dylibs/` with it, the
import fails with `ModuleNotFoundError`, and nothing is left in site-packages. A
leftover `.dylibs/` directory means the package data was installed outside the
RECORD, which is a packaging defect worth chasing.

### After a TestPyPI publish

```sh
python3.14 -m venv tvenv
./tvenv/bin/pip download --no-deps --dest . \
    --index-url https://test.pypi.org/simple/ mojoboost==0.1.0
shasum -a 256 mojoboost-0.1.0-*.whl
```

Expected: the hash equals `wheel_sha256` in the provenance sidecar and the entry
in `SHA256SUMS`. `--no-deps` because TestPyPI does not mirror numpy, and
`download` rather than `install` because the point is to compare the bytes, not
to exercise the package again.

## Open questions and unverified assumptions

Ordered by what would cost the most to discover late.

1. **Does the Mojo compiler honor `MACOSX_DEPLOYMENT_TARGET`?** Unknown, and it
   gates whether `pip install mojoboost` works for Mac users generally or only
   for those on the newest macOS. The experiment is one command
   (`handoffs/task18_platform.md`, edit 3) and this lane's C1 is what turns its
   result into a checked fact. Until it is run, every release produces a
   `macosx_26_0_arm64` wheel.
2. **Does the quarantine attribute propagate through pip's extraction?** The
   notarization decision rests on it not propagating. Test: download a wheel in
   Safari, `xattr -l` it (expect `com.apple.quarantine`), install it, then
   `xattr -l` the installed `_mojoboost.so` (expect nothing). If the attribute
   does survive, revisit Developer ID signing before the first public release.
3. **Do the bundled MAX dylibs carry absolute install names or rpaths?** E4
   above. Cheap to answer, and it decides whether the wheel loads on any machine
   other than the one that built it.
4. **Do the vendored MAX dylibs contain build-host path strings?** They are
   Modular's binaries, built in Modular's environment, and `conda-bld` is one of
   the markers `validate_artifact.py` rule R6 scans for across every object in
   the wheel. If R6 fires on a vendored library, this repository cannot fix it,
   and the choice is between shipping the string and not shipping the wheel.
   `inspect_wheel.py` C8 reports such hits separately for exactly this reason.
   Nobody has looked.
5. **Does the `runs-on` fallback work on a tag push?** `fromJSON(inputs.runner ||
   '[...]')` reads an input that does not exist on a `push` event. This is the
   one expression in the workflow that behaves differently between the two
   triggers, and it has not been exercised.
6. **Is `python3` inside the pixi `pkg` environment the right name?** The build
   script runs every standard library helper as `pixi run -e pkg python3`,
   because `validate_artifact.py` needs `tomllib` and a macOS system `python3`
   is too old. If that environment exposes only `python`, one line in
   `build_release_wheel.sh` changes.
7. **Should `setup-pixi` itself be version pinned?** The action is SHA pinned but
   the pixi binary it installs is not. The toolchain is pinned by `pixi.lock`
   regardless, and the provenance sidecar records the lock's hash, so this is a
   determinism improvement rather than a correctness one. `setup-pixi` has
   inputs for it; this lane used only `cache` and `environments`, which are the
   two this repository's CI already proves work.
8. **Does `gh-action-pypi-publish` generate attestations by default at the
   version pinned here, and is that wanted for a TestPyPI run?** Lane 10 owns
   attestations (`.github/workflows/release-provenance.yml`). If both workflows
   attest the same artifact, decide which one does it before the first publish.
9. **Does the Mojo compiler emit a byte-identical extension across two builds of
   one commit?** `SOURCE_DATE_EPOCH` now fixes the zip's timestamps, which was
   the known source of digest drift, but nobody has built the same commit twice
   and compared the `.so`. Until someone has, "reproducible" is not a word this
   project may use about the macOS wheel, and the TestPyPI-to-PyPI digest
   comparison stays invalid as `docs/PYPI_RELEASE.md` says. The test is two
   runs of `packaging/macos/build_release_wheel.sh` on one commit and
   `shasum -a 256` on both wheels and both extensions.
10. **Is the `.dylibs` directory name right for a non-delocated wheel?** It is the
   convention delocate uses and the existing builder follows it, so pip and the
   RECORD handle it as ordinary package data. Nothing checks that auditing tools
   outside this repository interpret it the same way.

## Deliberately not done

- No wheel was built, inspected, hashed, installed, or published.
- No file outside `packaging/macos/`, `.github/workflows/release-macos.yml`, and
  this handoff was edited. Every cross-file change is E1 through E5 above, as
  text.
- No platform status in `packaging/matrix/platform_matrix.toml` was changed.
  `macos-arm64-cp314` stays `designed`, and this lane produced no evidence that
  could move it.
- No production PyPI job exists, and no code path in this lane can reach
  pypi.org.
- No secret, token placeholder, or credential of any kind was added. The publish
  job's only credential is a short-lived OIDC token minted at run time.
- No action was pinned to a real commit hash, because that would have meant
  writing down a hash this lane could not resolve.
- `packaging/build_wheel.sh` was not changed, including the hardcoded list of
  four bundled dylibs. The release checks the list rather than replacing it.
- No second Mach-O parser was written. `inspect_wheel.py` imports
  `macho_info` from `packaging/matrix/validate_artifact.py` and adds only the one
  load command that script does not read, `LC_ID_DYLIB`.
