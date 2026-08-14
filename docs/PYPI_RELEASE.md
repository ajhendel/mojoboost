# Publishing mojoboost to PyPI

The exact steps to claim the name `mojoboost`, publish an artifact, and
undo a bad one. Every step here is owner-only: it needs administrative
access to the GitHub repository and to a PyPI account, and no step in it
can be delegated to a contributor or to an agent.

## Status

Nothing in this document has been executed. There is no PyPI project named
`mojoboost` owned by this repository, no TestPyPI project, no trusted
publisher, no API token, and no git tag.

The release workflow does exist, as
`.github/workflows/release-provenance.yml`, and it cannot run yet on
purpose. It fails closed in three ways: every `uses:` is pinned to a
`@REPLACE_WITH_SHA` placeholder rather than a ref, the build job needs a
`[self-hosted, macos, arm64, metal]` runner that is not registered, and
publication needs a `pypi` environment and a PyPI pending publisher that
do not exist. The first two belong to the lane that owns that file. The
third is this document, step 1.

The version in the repository is 0.1.0 and no artifact of it has been
published anywhere.

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

Step 1 can happen today and costs nothing. Steps 2 and 3 cannot yet: the
release workflow is pinned to placeholder SHAs and needs a self-hosted
Apple silicon runner with a Metal toolchain, neither of which is this
document's to fix. Step 4 is further out still, because the release gate
has open items and the macOS wheel target is `designed` rather than
`validated` in the matrix.

Do step 1 anyway, and do it before anything else. It is reversible, it
requires no artifact, and the value of a pending publisher is entirely in
having created it before the first upload rather than after.

## 1. Bind the name: pending trusted publishers

PyPI has no way to reserve a name without publishing. `setup.py register`
was removed years ago and there is no API that replaces it. What PyPI does
have is a **pending publisher**, a trusted-publisher binding created for a
project that does not exist yet. The first upload that arrives through
that publisher creates the project and makes it yours.

Be clear about what this buys. A pending publisher does not hold the name
against anyone else. Until the first upload lands, another account can
still take `mojoboost` and your pending publisher becomes inert. What it
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
   | PyPI Project Name | `mojoboost` |
   | Owner | `mojoboost-ml` |
   | Repository name | `mojoboost` |
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
4. In GitHub, Settings then Environments, create `pypi` and `testpypi`.
   On `pypi` set **Required reviewers** to yourself and restrict
   deployment to the `main` branch and to tags. That turns a production
   upload into something a human approves in the GitHub UI at the moment
   it happens, and it is the only interactive gate in this procedure.

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
unzip -p python/dist/mojoboost-*.whl \
    'mojoboost-*.dist-info/METADATA' | head -40
unzip -l python/dist/mojoboost-*.whl
```

What to look for, in order:

- `twine check` passes. This is where an unregistered trove classifier, a
  malformed license expression, or an unrenderable README shows up. PyPI
  rejects all three at upload, and finding out then means burning a
  version number.
- The filename is a tag that appears in `packaging/matrix/platform_matrix.toml`.
  Today the only one is `mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl`.
- `unzip -l` lists `mojoboost/__init__.py`, every other module in
  `python/mojoboost/`, `mojoboost/_mojoboost.so`, four
  `mojoboost/.dylibs/*.dylib`, and
  `mojoboost-*.dist-info/licenses/LICENSE`. Nothing else. No `tests/`, no
  `__pycache__`, no `.pytest_cache`. Check C9 of
  `packaging/macos/inspect_wheel.py` is the automated form of this.
- The METADATA `Requires-Dist` lines are extras only. A bare
  `Requires-Dist: numpy` with no `; extra ==` marker means the
  dependency-free promise in section 6.1 of the compatibility policy has
  been broken.

Confirm the deployment target in the tag matches the binary. The tag comes
from `DEFAULT_MACOS_TARGET` in `python/setup.py`, or from
`MOJOBOOST_MACOS_DEPLOYMENT_TARGET` when it is set, and neither is a
measurement of anything. Check C1 of `packaging/macos/inspect_wheel.py` is
what compares the two, and `packaging/macos/build_release_wheel.sh` runs
it. By hand:

```
otool -l python/mojoboost/_mojoboost.so | grep -A4 LC_BUILD_VERSION
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

1. Set the three version locations of compatibility policy section 1.1 to
   `0.1.0.dev1`: `pixi.toml`, `python/pyproject.toml`, and
   `python/mojoboost/__init__.py`. Commit on a branch. Do not tag.
2. Run `Release provenance` from the Actions tab (`workflow_dispatch` is
   its only trigger) with `publish` set to `testpypi`. Leave
   `macos_target` empty unless you are deliberately building the
   lowered-floor wheel.
3. Approve the `testpypi` environment if it is gated.
4. Verify against the installed package, never against the source tree.
   Run this from a directory that is not the checkout, so that
   `import mojoboost` cannot resolve to `python/mojoboost/`:

   ```
   REPO=$PWD
   cd /tmp && python -m venv tpv && tpv/bin/pip install \
       --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       "mojoboost==0.1.0.dev1"
   tpv/bin/python "$REPO/packaging/smoke_test.py"
   ```

   The `--extra-index-url` is required because TestPyPI does not mirror
   numpy or anything else an extra pulls in.
5. Revert the version commit. Nothing about `0.1.0.dev1` gets merged.

TestPyPI prunes projects and is not a backup of anything. Treat every
TestPyPI upload as disposable.

## 4. Claiming the name on PyPI

This step creates the PyPI project and permanently associates the name
with the account. It is deliberately not the production release.

Publish a **pre-release**: `0.1.0a1`. pip will not install a pre-release
unless the user passes `--pre` or pins the exact version, so the name is
claimed, the trusted publisher stops being pending, the provenance chain
is proved end to end on the real index, and nobody who types
`pip install mojoboost` gets an alpha they did not ask for.

1. Set the three version locations to `0.1.0a1`. Commit to `main` and tag
   `v0.1.0a1`. `packaging/macos/build_release_wheel.sh` refuses to build
   an untagged commit unless `MOJOBOOST_ALLOW_UNTAGGED=1` is set, and a
   published artifact should never be built with that set.
2. Run `Release provenance` with `publish` set to `pypi`. The tag does not
   trigger it. There is deliberately no `push` or `release` trigger on
   that workflow, so every publication is something a human started.
3. Approve the `pypi` environment when GitHub asks.
4. Verify, as in step 3 but against PyPI:

   ```
   REPO=$PWD
   cd /tmp && python -m venv pv && pv/bin/pip install "mojoboost==0.1.0a1"
   pv/bin/python "$REPO/packaging/smoke_test.py"
   ```

5. Confirm PyPI shows the publisher and the attestations on the file's
   page. If it does not, the upload did not go through trusted publishing
   and you should find out why before anything else is uploaded.
6. On PyPI, Manage project, Publishing: confirm the publisher is now a
   real publisher rather than pending, and that no API token exists on the
   account.
7. Restore the version in the repository to what it was.

After this, `pip install mojoboost` resolves to nothing installable, which
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

1. Set the three version locations to the release version. Commit to
   `main` and tag `vX.Y.Z`.
2. Run `Release provenance` with `publish` set to `testpypi`. Install from
   TestPyPI in a clean venv and confirm it works, exactly as in step 3.
3. Run it again with `publish` set to `pypi`, from the same commit.
4. Approve the `pypi` environment. The publish job downloads the wheel
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
pip download --no-deps --only-binary :all: "mojoboost==X.Y.Z"
pip download --no-deps --only-binary :all: \
    --index-url https://test.pypi.org/simple/ "mojoboost==X.Y.Z"
shasum -a 256 *.whl
```

Compare the PyPI file against the digest the run that published it
logged, and against the `SHA256SUMS` manifest in that run's
`release-metadata` artifact. Those must agree. The TestPyPI file is from a
different build and is not expected to match, per the caveat above. PyPI
also publishes the digest it recorded:

```
curl -s https://pypi.org/pypi/mojoboost/X.Y.Z/json \
    | python3 -c 'import json,sys; [print(f["filename"], f["digests"]["sha256"]) for f in json.load(sys.stdin)["urls"]]'
```

Three numbers, all equal: the publishing run's `SHA256SUMS`, the digest
that run logged, and PyPI's recorded digest. Any disagreement means the
file on the index is not the file that run built, and the release is
yanked rather than investigated in place.

Then check provenance, which is the stronger claim:

```
bash packaging/security/verify_release.sh mojoboost-X.Y.Z-*.whl
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

Check before doing any of this. If `mojoboost` is registered by someone
else and unused, PyPI's name-retention policy (PEP 541) is the only route
and it is slow. Plan on the alternative name rather than on winning the
dispute, and decide the name before publishing anything, because the
package name is in `python/pyproject.toml`, in the import path, and in
every document in this repository.
