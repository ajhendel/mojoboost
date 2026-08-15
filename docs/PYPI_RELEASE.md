# Publishing mojotrees to PyPI

The exact steps to claim the name `mojotrees`, publish an artifact, and
undo a bad one. Account enrollment and approval steps require the owner;
repository preparation and read-only verification can be automated.

## Status

**`mojotrees 0.1.0a2` is live on PyPI as of 2026-08-15T14:14:48Z.** The
name is claimed, the pending publisher converted to a real one on that
upload, and the artifact is
`mojotrees-0.1.0a2-cp314-cp314-macosx_26_0_arm64.whl` (2,237,257 bytes)
built from tag `v0.1.0a2` at commit `142e32f`, with SBOM and attestations.
Verified after the fact from a clean CPython 3.14 venv against the real
index (`pip install --pre mojotrees`, import, train, and
`packaging/smoke_test.py`, which reports the stdlib fallback path).

**The TestPyPI rehearsal for this name was SKIPPED.** Section 3 was not
run under `mojotrees`; the release went straight to PyPI at the owner's
instruction. So the TestPyPI publisher for `mojotrees` is still PENDING
and has never been exercised, and `mojotrees` does not exist on TestPyPI.
The next person to run section 3 will be proving that publisher for the
first time. The justification for skipping was that a rejected upload does
not consume the version number, so the downside was a retry rather than a
burned release; that reasoning holds only while the version is one nobody
depends on.

Note that section 4's sequencing argument below is therefore no longer
describing what happened. Read it as the intended procedure, not the
record.

The GitHub environments `testpypi` and `pypi` exist and are restricted to
the `main` branch. Neither requires a reviewer: required reviewers were
removed from `pypi` on 2026-08-15 (see section 1 step 4). No API token
exists anywhere and none is needed.

THE 2026-08-15 RENAME RESET THE PUBLISHER STATE. The trusted publishers
that had each published once belong to the `mojoboost` projects and are
scoped to the literal path `mojoboost-ml/mojoboost`, which no longer
exists; trusted-publisher matching is on literal owner and repository
strings, and GitHub's redirects do not apply to it. Those publishers are
dead. `mojotrees` was registered as a PENDING publisher on both indexes
(owner `mojotrees`, repository `mojotrees`, workflow
`release-provenance.yml`, environment `pypi` on PyPI and `testpypi` on
TestPyPI). A pending publisher is invisible from outside the account and
becomes real only on first upload. The PyPI one has since been proven by
the 0.1.0a2 upload and is now a normal publisher; the TestPyPI one is
still pending and unproven.

`mojoboost 0.1.0.dev1` is live on TestPyPI (the section 3 rehearsal, tag
`v0.1.0.dev1`) and `mojoboost 0.1.0a1` is live on PyPI (the pre-release
name claim, tag `v0.1.0a1`). Those are the FORMER name and must not be
deleted: a deleted PyPI name is reclaimable by anyone, and
`pip install mojoboost` is a live code path, so holding the old project
is the squat. Neither tag may be reused or moved, because each marks the
commit that produced a published artifact. Both were built by
`.github/workflows/release-provenance.yml` on the self-hosted
Apple-silicon Metal runner, with the SBOM, provenance sidecar, and GitHub
attestations attached, and both installs were verified from a clean
CPython 3.14 venv against the real index followed by
`packaging/smoke_test.py`. The rehearsal caught and fixed two workflow
bugs (the provenance sidecar path in the SBOM job, and a pypi-publish pin
too old for Metadata-Version 2.4) before any real version was spent.

The version in the repository is back at 0.1.0, which has not been
published. The 0.1.0a2 version commit was reverted per section 3 step 6;
the tag `v0.1.0a2` stays, because it records the commit the published
artifact came from. `0.1.0a1` was NOT reused for this release: tag
`v0.1.0a1` already marks the commit that produced the published
`mojoboost` wheel, and pointing it at a different artifact would make the
provenance record lie.

Plain `pip install mojotrees` still resolves to nothing installable,
because 0.1.0a2 is a pre-release and pip skips those. That is intended.
The install line everywhere in this repository is
`pip install --pre mojotrees`, and it stays that way until a final version
ships under the section 12 release gate.

## What this document does not decide

| Question | Where it is answered |
|---|---|
| Which artifacts exist, for which platform, with which exact filename | `packaging/matrix/platform_matrix.toml` and [PLATFORM_MATRIX.md](PLATFORM_MATRIX.md) |
| What a release promises, and what has to be true before one is cut | [COMPATIBILITY_POLICY.md](COMPATIBILITY_POLICY.md), especially the release gate in section 12 |
| What the package says about itself | `python/pyproject.toml` |
| Which CPython versions are supported, and whether the floor is real | [PYTHON_SUPPORT.md](PYTHON_SUPPORT.md) |
| Repository and workflow security posture, and the full owner setup list | [RELEASE_SECURITY.md](RELEASE_SECURITY.md) and `handoffs/release_10_security.md` |

Where this document and the platform matrix disagree about an artifact,
the matrix wins. Where it and the compatibility policy disagree about
whether a release may be cut, the policy wins. Where it and
`docs/RELEASE_SECURITY.md` disagree about a security decision, that
document wins. This one covers getting a decided release onto an index and
off it again, and nothing else.

## The rule that governs everything below

**A published artifact comes from a release workflow, never from a
laptop.**

Not because a laptop build is likely to be wrong, though the macOS wheel
is tagged with a hardcoded deployment target that only a human keeps
honest, but because a laptop build is unattributable. Trusted publishing
binds an upload to a specific repository, a specific workflow file, and a
specific environment, and PyPI records that binding as provenance
alongside the file. An upload from a laptop is a person asserting where a
binary came from. Once one artifact on the index has that property, every
artifact on the index has to be checked by hand forever.

The one thing a laptop is still for is building a candidate wheel and
running `packaging/test_wheel.sh` against it before any of this starts.
That artifact is a test subject and is never uploaded.

## Sequence

The order matters, and the two claims are deliberately separated.

1. [Bind the name](#1-bind-the-name-pending-trusted-publishers) with
   pending trusted publishers on TestPyPI and PyPI. No upload, no version
   number, reversible.
2. [Prove the pipeline](#3-the-testpypi-release) on TestPyPI with a
   throwaway version.
3. [Claim the name](#4-claiming-the-name-on-pypi) on PyPI with a
   pre-release version. This creates the project. It is not a release
   anyone is asked to install.
4. [Cut a production release](#5-the-production-release) only after the
   section 12 release gate passes.

The repository-side preparation for step 1 is complete, but the owner must
still register the two pending publishers on the package indexes. Steps 2 and
3 are mechanically available after that account-side setup, but each is an
external publication and must be started deliberately. Step 4 remains gated
by the compatibility policy and by target validation in the platform matrix.

## 1. Bind the name: pending trusted publishers

PyPI has no way to reserve a name without publishing. `setup.py register`
was removed years ago and there is no API that replaces it. What PyPI does
have is a **pending publisher**, a trusted-publisher binding created for a
project that does not exist yet. The first upload that arrives through
that publisher creates the project and makes it yours.

Be clear about what this buys. A pending publisher does not hold the name
against anyone else. Until the first upload lands, another account can
still take `mojotrees` and your pending publisher becomes inert. What it
buys is that when you do claim the name, you claim it through OIDC with no
token in existence, and there is never a window in which a long-lived
credential could have published as you.

This is step 6 of the owner-action list in
`handoffs/release_10_security.md`, which is the complete sequence and the
one to work through. Repeated here because the reasoning below is about
claiming a name rather than about security posture, and because getting
the workflow filename wrong is the one mistake that produces a publisher
which silently never matches.

Do this on **TestPyPI first**, then PyPI, with identical values.

1. Enable two-factor authentication on both accounts. PyPI requires it for
   anyone who owns a project, and a recovery-code-only account is a
   single point of failure for the whole namespace. Store the recovery
   codes somewhere that is not the machine that builds wheels.
2. TestPyPI: https://test.pypi.org/manage/account/publishing/
   PyPI: https://pypi.org/manage/account/publishing/
3. Under "Add a new pending publisher", GitHub tab, enter exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `mojotrees` |
   | Owner | `mojotrees` |
   | Repository name | `mojotrees` |
   | Workflow name | `release-provenance.yml` |
   | Environment name | `pypi` on PyPI, `testpypi` on TestPyPI |

   The workflow name is the filename, not the `name:` inside the file,
   which is `Release provenance`. The environment names come from the
   `publish` input of that workflow, whose `environment.name` is the input
   value itself, so `testpypi` and `pypi` are the only two possible and
   they are spelled exactly like that.

   All five fields are matched exactly on every upload. A rename of the
   workflow file, of the repository, or of the GitHub account breaks
   publishing until the publisher is edited, which is the point.
4. In GitHub, Settings then Environments, create `pypi` and `testpypi`,
   and restrict deployment on both to the `main` branch and to tags, so
   the release identity cannot be exercised from an arbitrary branch.

   **Required reviewers were removed from `pypi` on 2026-08-15**, at the
   owner's instruction, after the 0.1.0a2 release. It had been set to the
   owner, which made a production upload something a human approved in the
   GitHub UI at the moment it happened. It is gone because on a
   single-maintainer project the approver and the dispatcher are the same
   person, so the click authenticated nothing and only added friction to a
   workflow that is already manual-dispatch-only.

   Know what that costs. A production upload is now a consequence of
   dispatching the run rather than a separate act. The remaining controls
   are that the workflow has no `push` or `release` trigger so every
   publication is still started by a human on purpose, the branch policy
   above, and trusted publishing itself. If a second maintainer ever gains
   push access, put required reviewers back: at that point it starts
   authenticating something real.

   The environments hold no secrets. That is the intended end state.

## 2. Preflight

Before any upload, on the machine that builds the candidate:

```
pixi run -e pkg build-wheel
pixi run -e pkg test-wheel
python3 packaging/matrix/validate_matrix.py
```

`packaging/macos/build_release_wheel.sh` is the release-grade wrapper
around that: it refuses an untagged commit, applies the deployment target
to both halves of the tag, writes the provenance sidecar, and runs the
release-only rules. Use it for anything that might be published. The three
commands above are for a candidate that certainly will not be.

Then check the metadata itself. These read the built artifact and touch no
network:

```
pixi run -e pkg python -m twine check python/dist/*.whl
unzip -p python/dist/mojotrees-*.whl \
    'mojotrees-*.dist-info/METADATA' | head -40
unzip -l python/dist/mojotrees-*.whl
```

What to look for, in order:

- `twine check` passes. This is where an unregistered trove classifier, a
  malformed license expression, or an unrenderable README shows up. PyPI
  rejects all three at upload, and finding out then means burning a
  version number.
- The filename is a tag that appears in `packaging/matrix/platform_matrix.toml`.
  Today the only one is `mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl`.
- `unzip -l` lists `mojotrees/__init__.py`, every other module in
  `python/mojotrees/`, `mojotrees/_mojotrees.so`, four
  `mojotrees/.dylibs/*.dylib`, and
  `mojotrees-*.dist-info/licenses/LICENSE`. Nothing else. No `tests/`, no
  `__pycache__`, no `.pytest_cache`. Check C9 of
  `packaging/macos/inspect_wheel.py` is the automated form of this.
- The METADATA `Requires-Dist` lines are extras only. A bare
  `Requires-Dist: numpy` with no `; extra ==` marker means the
  dependency-free promise in section 6.1 of the compatibility policy has
  been broken.

Confirm the deployment target in the tag matches the binary. The tag comes
from `DEFAULT_MACOS_TARGET` in `python/setup.py`, or from
`MOJOTREES_MACOS_DEPLOYMENT_TARGET` when it is set, and neither is a
measurement of anything. Check C1 of `packaging/macos/inspect_wheel.py` is
what compares the two, and `packaging/macos/build_release_wheel.sh` runs
it. By hand:

```
otool -l python/mojotrees/_mojotrees.so | grep -A4 LC_BUILD_VERSION
```

The `minos` there must equal the version in the wheel filename. If they
disagree, fix `python/setup.py` and rebuild. Do not rename the file: the
tag comes from the build, and renaming makes the label a lie rather than
a mistake.

Record the digest of the candidate. It is not what gets published, but it
is what tells you whether the workflow built the same thing you tested:

```
shasum -a 256 python/dist/*.whl
```

## 3. The TestPyPI release

TestPyPI exists so the first thing that ever goes through the publishing
path is something whose version number does not matter.

Use a throwaway version. `0.1.0.dev1`, then `.dev2` and so on if you need
another attempt. **A version number on an index can never be reused**,
even after deletion, so burn dev versions freely and never a real one.

1. Set the version to `0.1.0.dev1` in the three locations of
   compatibility policy section 1.1 (`pixi.toml`, `python/pyproject.toml`,
   `python/mojotrees/__init__.py`) and in
   `packaging/matrix/platform_matrix.toml`, whose `version` and `filename`
   rows must agree with the wheel or `validate_artifact.py` rule R1b fails
   the build. Run `python3 packaging/matrix/validate_matrix.py` to check.
2. Commit on `main` and tag `v0.1.0.dev1`. Both halves are forced, and an
   earlier revision of this document got both wrong:
   `packaging/macos/build_release_wheel.sh` refuses an untagged HEAD, and
   the `testpypi` environment's deployment branch policy only admits runs
   dispatched from `main`, so a side branch never reaches the publish job.
3. Run `Release provenance` from the Actions tab (`workflow_dispatch` is
   its only trigger), dispatched **from `main`**, with the `ref` input set
   to `v0.1.0.dev1` (its default is the production tag, so it must be
   overridden here) and `publish` set to `testpypi`. Leave `macos_target`
   empty unless you are deliberately building the lowered-floor wheel.
4. Approve the `testpypi` environment if it is gated.
5. Verify against the installed package, never against the source tree.
   Run this from a directory that is not the checkout, so that
   `import mojotrees` cannot resolve to `python/mojotrees/`:

   ```
   REPO=$PWD
   cd /tmp && python -m venv tpv && tpv/bin/pip install \
       --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       "mojotrees==0.1.0.dev1"
   tpv/bin/python "$REPO/packaging/smoke_test.py"
   ```

   The `--extra-index-url` is required because TestPyPI does not mirror
   numpy or anything else an extra pulls in.
6. Revert the version commit on `main`. Nothing about `0.1.0.dev1` stays
   in the tree; the `v0.1.0.dev1` tag stays, because it records the commit
   the published artifact came from.

TestPyPI prunes projects and is not a backup of anything. Treat every
TestPyPI upload as disposable.

## 4. Claiming the name on PyPI

This step creates the PyPI project and permanently associates the name
with the account. It is deliberately not the production release.

Publish a **pre-release**: `0.1.0a2`. pip will not install a pre-release
unless the user passes `--pre` or pins the exact version, so the name is
claimed, the trusted publisher stops being pending, the provenance chain
is proved end to end on the real index, and nobody who types
`pip install mojotrees` gets an alpha they did not ask for.

1. Set the four version locations (the three of compatibility policy
   section 1.1, plus `packaging/matrix/platform_matrix.toml` and its
   `filename` rows) to `0.1.0a2`. Commit to `main` and tag
   `v0.1.0a2`. `packaging/macos/build_release_wheel.sh` refuses to build
   an untagged commit unless `MOJOTREES_ALLOW_UNTAGGED=1` is set, and a
   published artifact should never be built with that set.
2. Run `Release provenance` with `publish` set to `pypi`. The tag does not
   trigger it. There is deliberately no `push` or `release` trigger on
   that workflow, so every publication is something a human started.
3. No approval step. Required reviewers were removed on 2026-08-15, so
   the publish job runs as soon as its dependencies finish.
4. Verify, as in step 3 but against PyPI:

   ```
   REPO=$PWD
   cd /tmp && python -m venv pv && pv/bin/pip install "mojotrees==0.1.0a2"
   pv/bin/python "$REPO/packaging/smoke_test.py"
   ```

5. Confirm PyPI shows the publisher and the attestations on the file's
   page. If it does not, the upload did not go through trusted publishing
   and you should find out why before anything else is uploaded.
6. On PyPI, Manage project, Publishing: confirm the publisher is now a
   real publisher rather than pending, and that no API token exists on the
   account.
7. Restore the version in the repository to what it was.

After this, `pip install mojotrees` resolves to nothing installable, which
is the correct state until there is a release worth installing.

## 5. The production release

Only after every item of section 12 of
[COMPATIBILITY_POLICY.md](COMPATIBILITY_POLICY.md) passes. That gate is
not restated here so that it cannot drift; read it there.

Two additional preconditions belong to publishing specifically:

- **P1.** Every artifact the release will ship has a row in
  `packaging/matrix/platform_matrix.toml` whose `filename` matches
  exactly, and `python3 packaging/matrix/validate_matrix.py` passes.
- **P2.** The trusted publisher on PyPI is a real publisher, not pending,
  and the account holds no API token.

Then:

1. Set the four version locations (the three of compatibility policy
   section 1.1, plus `packaging/matrix/platform_matrix.toml` and its
   `filename` rows) to the release version. Commit to
   `main` and tag `vX.Y.Z`.
2. Run `Release provenance` with `publish` set to `testpypi`. Install from
   TestPyPI in a clean venv and confirm it works, exactly as in step 3.
3. Run it again with `publish` set to `pypi`, from the same commit.
4. No approval step; required reviewers were removed on 2026-08-15. The
   publish job runs straight through. It downloads the wheel
   built earlier in that run, checks nothing else is in the directory, and
   uploads. It checks out no source and runs no repository code.
5. Verify hashes, below.
6. Write the release note. Compatibility policy items D2 and 10.4 say what
   it has to name: every artifact with its exact tag, every platform that
   gets none, and every platform's tier. Include the digests.

One thing to know about step 3. Each `workflow_dispatch` run builds its
own wheel, so the file that goes to PyPI is not literally the file that
went to TestPyPI. See the next section for what that costs and what to
check.

## Promoting the identical artifact

The goal is that the file on PyPI is the file that was tested. Say
precisely how much of that is true today, because it is not all of it.

**Within one run, it holds.** The build job computes the digest on the
machine that produced the wheel and carries it forward as a job output.
The attestation job re-hashes what it downloaded and refuses to sign
unless the two agree. The publish job downloads the same artifact and
uploads it without checking out any source. Artifact upload and download
are not a trust boundary the run relies on.

**Across two runs, it does not.** The workflow's `publish` input takes one
index per run, so the TestPyPI rehearsal and the PyPI publication are two
runs and two builds. Wheels are not reproducible by default: zip entry
timestamps differ between builds of identical source, so the two files
will have different digests even when nothing is wrong.
`packaging/linux/build_wheel_linux.sh` sets `SOURCE_DATE_EPOCH` from the
commit for exactly this reason; `packaging/macos/build_release_wheel.sh`
does not, so the macOS wheel is not reproducible today.

What that means in practice:

- Comparing the TestPyPI and PyPI digests is **not** a valid check right
  now. A mismatch is expected. Do not treat one as an incident, and do not
  treat a match as proof of anything either.
- The check that does hold is the attestation. `gh attestation verify`
  says the file on PyPI came out of a run of
  `.github/workflows/release-provenance.yml` in this repository, which is
  the question worth asking.
  `packaging/security/verify_release.sh` is the consumer-side script for
  it.
- Run the TestPyPI rehearsal from the same commit and confirm it installs
  and smoke-tests. It proves the source is publishable. It does not prove
  the two binaries are the same file, and this document does not claim it
  does.

Two ways to close the gap, neither of which is this document's to make.
Either set `SOURCE_DATE_EPOCH` in `packaging/macos/build_release_wheel.sh`
as the Linux builder already does, or give the workflow a publish mode
that uploads one build to both indexes in a single run. Until one of them
lands, the honest statement in a release note is "built by run N, attested"
and not "identical to the tested artifact".

Verify what is verifiable, from a clean directory:

```
mkdir -p /tmp/verify && cd /tmp/verify
pip download --no-deps --only-binary :all: "mojotrees==X.Y.Z"
pip download --no-deps --only-binary :all: \
    --index-url https://test.pypi.org/simple/ "mojotrees==X.Y.Z"
shasum -a 256 *.whl
```

Compare the PyPI file against the digest the run that published it
logged, and against the `SHA256SUMS` manifest in that run's
`release-metadata` artifact. Those must agree. The TestPyPI file is from a
different build and is not expected to match, per the caveat above. PyPI
also publishes the digest it recorded:

```
curl -s https://pypi.org/pypi/mojotrees/X.Y.Z/json \
    | python3 -c 'import json,sys; [print(f["filename"], f["digests"]["sha256"]) for f in json.load(sys.stdin)["urls"]]'
```

Three numbers, all equal: the publishing run's `SHA256SUMS`, the digest
that run logged, and PyPI's recorded digest. Any disagreement means the
file on the index is not the file that run built, and the release is
yanked rather than investigated in place.

Then check provenance, which is the stronger claim:

```
bash packaging/security/verify_release.sh mojotrees-X.Y.Z-*.whl
```

Put the digests in the release note. Together with the attestation they
are the only thing a user has that does not depend on trusting the
index.

## Yanking a bad release

Section 10 of [RELEASE_SECURITY.md](RELEASE_SECURITY.md) is the authority
on choosing between yanking, deleting, and revoking, and it covers the
compromised-release case this section does not. What follows is the
operational short form for the ordinary case, a release that is wrong but
not dangerous.

**Never delete. Yank.**

Deleting a release frees nothing: the version number and the filename are
burned forever and cannot be re-uploaded. Worse, deleting breaks every
lockfile that pinned it, turning other people's working builds into
resolution failures. Yanking leaves the file installable for anyone who
pinned that exact version and removes it from every resolution that did
not.

To yank:

1. PyPI, Manage project, Releases, the version, Options, Yank.
2. Give a reason. It is shown to anyone who installs the pinned version
   and it is the only place a user finds out why.
3. Release the fix as a new version. A yanked version is never re-cut
   under its own number.
4. Amend the release note of the yanked version to say it was yanked, when,
   and what replaced it.

Yank when the artifact is wrong: it fails to import, it is tagged for a
platform it cannot run on, it ships the wrong binary, or it produces wrong
numbers. Do not yank for a documentation error or a missing classifier.

Delete only for something that must not remain downloadable at all, such
as a credential or private data inside the artifact. In that case the
credential is compromised the moment it is uploaded and must be rotated
regardless of whether the file is deleted.

## Tokens: how to have none

Trusted publishing means no long-lived credential exists. The OIDC token
GitHub mints is valid for minutes, scoped to one project, and cannot be
replayed from anywhere else. There is nothing to leak, nothing to rotate,
and nothing to find in a repository years later.

Do not create an API token to get started faster. A token created "just
for the first upload" is a token that exists.

If a token becomes genuinely unavoidable, for an index that does not
support OIDC:

- Scope it to the single project. Never account-scoped.
- Store it as a GitHub Environment secret on `pypi`, with required
  reviewers, so it cannot be used by a workflow run nobody approved.
  Never a repository-level secret, never an organization secret, never
  `~/.pypirc` on a laptop, never a `.env`.
- Use it, then revoke it in the same session. A release token that
  outlives its release is the failure mode.
- Its lifetime is the release, not the calendar. There is no rotation
  schedule because there is no long-lived token.

If a token is ever exposed, in a log, a screenshot, a commit, or a pasted
terminal: revoke it on PyPI first and investigate second. Then check
whether any release appeared that the release workflow did not produce,
and yank anything you cannot account for.

## The release workflow

`.github/workflows/release-provenance.yml` is the only thing that may
publish. It is owned by the release-security lane, not by this document,
and its own reasoning lives in `docs/RELEASE_SECURITY.md` and
`handoffs/release_10_security.md`. What matters here is the handful of
facts a publisher has to know.

- **`workflow_dispatch` only.** No `push`, no `release`, and above all no
  `pull_request_target`. A fork pull request must never reach a job with
  `id-token: write`, a self-hosted runner, or an environment, and the only
  reliable way to guarantee that is to have no event a fork can cause.
  Every publication is therefore something a human started by hand.
- **Three things stop it from running today**, all deliberate. Every
  `uses:` is `@REPLACE_WITH_SHA`, which is not a ref, and
  `packaging/security/check_action_pins.py` is the guard that keeps
  pinning by tag from becoming an acceptable interim state. The build job
  wants a `[self-hosted, macos, arm64, metal]` runner, because a wheel
  built without a Metal toolchain answers `has_accelerator()` differently
  and is a different product under the same filename. And publication
  wants the `pypi` environment and the pending publisher from step 1.
- **`permissions: {}` at the top**, with each job adding back only what it
  needs. The publish job holds `id-token: write` and nothing else. It does
  not check the repository out, so it runs no code from this repository
  and no third-party action other than the publisher itself.
- **No secret appears in the file.** If `secrets.` ever does, something has
  gone wrong.
- **`concurrency: release-provenance`, no cancel.** Two runs could
  otherwise publish two different wheels under one version.
- **The publish job asserts `dist/` holds exactly one file**, the wheel it
  built. That assertion is what keeps a broken sdist off the index, and it
  is stronger than a `*.whl` glob because it also fails on an unexpected
  extra file.

The pending publisher created in step 1 is scoped to that filename and to
those environment names. Renaming the file breaks publishing until the
publisher is edited, which is the intended behavior and not an
inconvenience to route around.

## If the name is already taken

Check before doing any of this. If `mojotrees` is registered by someone
else and unused, PyPI's name-retention policy (PEP 541) is the only route
and it is slow. Plan on the alternative name rather than on winning the
dispute, and decide the name before publishing anything, because the
package name is in `python/pyproject.toml`, in the import path, and in
every document in this repository.
