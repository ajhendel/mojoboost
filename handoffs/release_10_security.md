# Handoff: release security, provenance, and recovery

Lane 10 of a parallel round. This lane defined the security posture that has to
exist before mojoboost publishes an executable native wheel to a package index,
and wrote the configuration for it.

**Nothing in this lane was executed.** No script was run, no workflow was
triggered, no wheel was built, no scanner or signing tool was invoked, and no
network call was made. Every command in this document is written to be run
later, by a person, in the order given.

## What this lane added

| Path | What it is |
|---|---|
| `SECURITY.md` | Reporting policy. Private channel, supported versions, response targets, scope, disclosure expectations, and the model file trust boundary stated plainly |
| `docs/RELEASE_SECURITY.md` | The design. Threat model, release owner model, trusted publishing, action pinning, permissions, dependency policy, why auto-merge is unsafe, hashes and attestations and SBOM, compromised release response, yank and revoke |
| `.github/dependabot.yml` | Weekly grouped version updates for GitHub Actions and for `python/pyproject.toml`, with the conda and pixi blind spot written into the file |
| `.github/workflows/release-provenance.yml` | Four jobs: guard, build, attest, publish. `workflow_dispatch` only, `permissions: {}` at the top, trusted publishing, no secret of any kind |
| `packaging/security/check_action_pins.py` | Enforces SHA pinning. Standard library, bare checkout, runs in under a second |
| `packaging/security/hash_manifest.py` | `SHA256SUMS` write, verify, and single digest. Standard library, so a consumer needs nothing installed |
| `packaging/security/sbom_supplement.py` | Adds the compiled extension and the bundled MAX runtime to a Syft CycloneDX SBOM, hashed out of the wheel itself |
| `packaging/security/verify_release.sh` | The consumer side. Digest, then `gh attestation verify` with `--signer-workflow` |
| `packaging/security/release_checklist.md` | Gate items F through I, continuing the lettering of `docs/COMPATIBILITY_POLICY.md` section 12 rather than restarting it |
| `packaging/security/README.md` | What runs, what does not, and the two questions this directory keeps apart |

Nothing outside that list was edited. `README.md`, `pixi.toml`, `ci.yml`,
`gpu-validation.yml`, `packaging/build_wheel.sh`, `python/pyproject.toml`, and
`packaging/matrix/` are all untouched, and the edits they need are written out
below rather than applied.

## The one thing to take away

The release workflow **cannot run**, in three independent ways, and each one is
deliberate.

1. Every `uses:` says `@REPLACE_WITH_SHA`, which is not a ref, so the run fails
   before a job starts. Pinning by tag was not an acceptable interim state for
   a workflow that mints an OIDC token, and pinning to a SHA requires looking
   each action up, which is a network operation this lane did not perform.
2. The build job needs a `[self-hosted, macos, arm64, metal]` runner. None is
   registered.
3. Publication needs a `pypi` environment and a PyPI pending publisher. Neither
   exists.

A release pipeline that fails closed is not a half finished pipeline. The
alternative, a workflow that runs today against whatever `@v4` points to, is
worse in exactly the way section 5.1 of `docs/RELEASE_SECURITY.md` describes.

## Implemented in configuration

These are done and are in the tree.

- Least privilege. `permissions: {}` at the workflow level, and four jobs
  granting `contents: read`, `contents: read`, `contents: read` plus
  `id-token: write` plus `attestations: write`, and `id-token: write` alone.
  The self hosted build job has the weakest token in the run.
- No fork pull request can reach a privileged job, structurally, because
  `workflow_dispatch` is the only trigger. There is no `pull_request_target`
  anywhere in this repository, and adding one is the change to argue hardest
  about.
- `persist-credentials: false` on every checkout, which matters most on the
  self hosted runner where the workspace outlives the job.
- No secret and no token placeholder, anywhere. Trusted publishing only.
- Build and publish separated, so the job holding the publishing identity runs
  no repository code and checks the repository out at all.
- The digest is computed on the build machine, carried as a job output, and
  re-checked after every artifact download, so artifact storage is not a trust
  boundary the run depends on.
- The publish job asserts that exactly one file, the wheel, is in the upload
  directory before the publisher sees it.
- SBOM supplemented with the shipped native runtime and attested to the same
  digest as the wheel.
- `concurrency` on the workflow, so two runs cannot publish two different
  wheels under one version.

## Owner actions

None of these is a file, so none of them could be done by this lane. They are
in the order they should be done. The first two are worth doing today, before
any release question is decided, because they cost minutes and change what
happens when somebody finds a bug next week.

1. **Enable private vulnerability reporting.** Settings, Advanced Security (or
   Code security and analysis), Private vulnerability reporting, Enable.
   Without this, the link in `SECURITY.md` returns a 404 and reporters fall
   back to plaintext email. Confirm afterwards that
   `https://github.com/ajhendel/mojoboost/security/advisories/new` loads.
2. **Enable the dependency graph, Dependabot alerts, and Dependabot security
   updates**, same settings page. `.github/dependabot.yml` configures scheduled
   version updates only; security updates are driven by alerts, which are a
   setting. Note the coverage gap this does not close, in section 6.1 of
   `docs/RELEASE_SECURITY.md`: nothing watches `pixi.lock`, which is where the
   libraries that ship inside the wheel come from.
3. **Restrict which actions can run.** Settings, Actions, General. Allow
   actions created by GitHub and specified actions, and list the four or five
   the release path uses. Set the default `GITHUB_TOKEN` permissions to read
   only, and require approval for all outside collaborators' workflow runs.
4. **Pin the actions.** For each `@REPLACE_WITH_SHA` in
   `.github/workflows/release-provenance.yml`, open the action's repository,
   read the release you intend to use, copy the commit SHA of that tag, and
   replace the placeholder, keeping the trailing version comment. Then run
   `python3 packaging/security/check_action_pins.py .github/workflows/release-provenance.yml`
   and expect it to pass.
5. **Create the `testpypi` and `pypi` environments.** Settings, Environments.
   Required reviewers set to yourself, deployment branches limited to `main`
   and tags. Add no secret to either.
6. **Create the PyPI pending publishers.** On PyPI and TestPyPI, Your projects,
   Publishing, add a pending publisher with owner `ajhendel`, repository
   `mojoboost`, workflow `release-provenance.yml`, and environment `pypi` or
   `testpypi`. Do not create a project API token, and if one already exists,
   delete it.
7. **Register the self hosted macOS runner**, with the labels
   `self-hosted, macos, arm64, metal`, on a machine with Xcode and the Metal
   toolchain. Read section 3.3 of `docs/RELEASE_SECURITY.md` first. A self
   hosted runner in a repository that accepts fork pull requests is a standing
   hazard, and the mitigation here is that the release workflow has no event a
   fork can cause. Do not add one.
8. **Add a second maintainer with PyPI ownership**, when there is one. This is
   the gap most likely to matter first, and it is the only one on this list
   that a person, rather than a setting, closes.

Rehearse to TestPyPI before PyPI. That is item H3 of the checklist and it is
the only way any of this gets tested end to end.

## Edits to files this lane does not own

Written out, ready to apply, and **not applied**.

### Edit A: `.github/workflows/ci.yml`, pin the actions

`ci.yml` uses `actions/checkout@v4` and `prefix-dev/setup-pixi@v0.10.1`, and
`gpu-validation.yml` adds `actions/upload-artifact@v4`. Those are tags. CI holds
no secret and gets a read only token, so the exposure is smaller than the
release path's, but a compromised action still runs on a runner with the
repository checked out.

Replace each with the commit SHA of the release you intend, keeping a trailing
comment, the same way as owner action 4.

### Edit B: `.github/workflows/ci.yml`, add the pin check

Modeled on the existing `parity` job, which already runs a standard library
checker on a bare runner:

```yaml
  # Action pinning. Standard library only and no build, so it runs on a bare
  # runner in seconds and fails when a workflow starts trusting a tag.
  pins:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<sha> # actions/checkout v4
        with:
          persist-credentials: false
      - name: Action pins
        run: python3 packaging/security/check_action_pins.py .github/workflows
```

Apply this **after** edit A and after owner action 4, not before. Added first,
it fails on the workflows it is meant to protect and the fix is to delete the
job, which is the wrong lesson.

### Edit C: `pixi.toml`, one task

Next to `check-parity`, which it deliberately resembles:

```toml
# Checks that every GitHub Action is pinned to a commit SHA rather than a tag.
# Standard library only, and it builds nothing, so it is cheap enough to run on
# every change. See docs/RELEASE_SECURITY.md section 5.
check-pins = "python3 packaging/security/check_action_pins.py .github/workflows"
```

Do not add a pixi task for `verify_release.sh`. It is run by a consumer who has
downloaded a wheel and has no checkout, which is the situation it is written
for.

### Edit D: the provenance sidecar

This is a dependency, not a suggestion. The release workflow stops hard when
`<wheel>.provenance.json` is missing, and `sbom_supplement.py` reads the Mojo
and MAX versions out of it, because they cannot be recovered from the wheel
afterwards.

No edit may be needed. The sidecar was specified in
`handoffs/task18_platform.md` edit 4, and while this lane was writing, another
lane landed `packaging/macos/provenance.sh` and `packaging/macos/report_accelerator.mojo`,
which appear to implement it. Confirm two things and then either point the
workflow's build job at that script or apply task18 edit 4 to
`packaging/build_wheel.sh`:

1. The sidecar lands at `<wheel>.provenance.json`, next to the wheel.
2. Its keys are the ones `packaging/matrix/validate_artifact.py` requires under
   rule R7, which are the same ones `sbom_supplement.py` reads.

This lane deliberately did not write a second implementation. Two lanes writing
the same facts differently is how they end up disagreeing, and
`validate_artifact.py` rule R7 is already the authority on which fields are
required.

One addition worth making while applying it. `python/dist/` is in
`.gitignore`, so both the wheel and the sidecar are ignored, which is right for
the repository and wrong for a release. The workflow uploads them as separate
artifacts for exactly that reason, and they must travel together anywhere else
too.

### Edit E: `README.md`, a link

Other lanes are editing `README.md` in this round, so this is a replacement for
a specific paragraph rather than a diff. In the "Tests and wheels" section,
after the sentence about wheels targeting macOS on Apple silicon, add:

> No release has been published. What a release will have to prove about itself
> before one is, meaning trusted publishing, artifact digests, build provenance
> attestations, an SBOM covering the bundled MAX runtime, and the yank and
> revoke procedure, is in
> [docs/RELEASE_SECURITY.md](docs/RELEASE_SECURITY.md). To report a
> vulnerability, see [SECURITY.md](SECURITY.md).

### Edit F: `.github/ISSUE_TEMPLATE/config.yml`, a contact link

The existing file already routes parity and hardware questions away from the
issue tracker. Security reports need the same treatment, and more urgently,
because the cost of a misrouted one is a public zero day rather than a
duplicate issue.

```yaml
  - name: Report a security vulnerability
    url: https://github.com/ajhendel/mojoboost/security/advisories/new
    about: Private disclosure. Do not open a public issue for a suspected vulnerability.
```

This link 404s until owner action 1 is done. Do that first.

### Edit G: `CONTRIBUTING.md`, one line

Under "Pull requests", after the list of what a description must cover:

> A pull request that touches `.github/**` or `packaging/**` is a change to
> what gets built and published, not only to what gets tested. Say so in the
> description and expect it to be reviewed as such. See
> [docs/RELEASE_SECURITY.md](docs/RELEASE_SECURITY.md) section 7.

## Verification commands

Exact, and **none of them was run**. Ordered so that the ones that work on a
bare checkout today come first.

Runnable now, on any machine with a checkout:

```sh
# Expected to FAIL today. ci.yml and gpu-validation.yml pin by tag, and
# release-provenance.yml carries placeholders. That failure is the finding.
python3 packaging/security/check_action_pins.py .github/workflows

# Expected to PASS once owner action 4 is done
python3 packaging/security/check_action_pins.py .github/workflows/release-provenance.yml

# Read the matches, do not count them. Expected today: exactly one, the
# detection pattern in packaging/macos/inspect_wheel.py, which is another
# lane's secret scanner and has to contain the string it looks for. An
# assignment, a workflow input, or a value is the finding. The --include
# filters matter: docs/RELEASE_SECURITY.md and the release checklist both name
# those strings in order to prohibit them
grep -rInE 'PYPI_API_TOKEN|TWINE_PASSWORD|pypi-AgEIcHlwaS5vcmc' \
    --include='*.yml' --include='*.yaml' --include='*.sh' --include='*.py' \
    --include='*.toml' --include='*.mojo' \
    --exclude-dir=.git --exclude-dir=.pixi .

# Expected: no matches. A fork triggered event must never reach a privileged
# job. Matches the trigger key only, not the comment in release-provenance.yml
# that explains why it is absent
grep -rnE '^\s*pull_request_target\s*:' .github/workflows/

# Expected: an empty list, twice
gh secret list --repo ajhendel/mojoboost
gh secret list --repo ajhendel/mojoboost --env pypi
```

Runnable once a wheel exists:

```sh
WHEEL=$(ls python/dist/mojoboost-*.whl)

python3 packaging/security/hash_manifest.py write python/dist/SHA256SUMS \
    "$WHEEL" "$WHEEL.provenance.json"
python3 packaging/security/hash_manifest.py verify python/dist/SHA256SUMS
python3 packaging/security/hash_manifest.py digest "$WHEEL"

# Needs a Syft CycloneDX SBOM and the sidecar from edit D
python3 packaging/security/sbom_supplement.py \
    sbom.cyclonedx.json "$WHEEL.provenance.json" "$WHEEL" sbom.merged.cyclonedx.json
```

Runnable once a release exists, from a machine that is not the build machine:

```sh
bash packaging/security/verify_release.sh ./mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl ./SHA256SUMS

gh attestation verify ./mojoboost-0.1.0-cp314-cp314-macosx_26_0_arm64.whl \
    --repo ajhendel/mojoboost \
    --signer-workflow ajhendel/mojoboost/.github/workflows/release-provenance.yml
```

The `--signer-workflow` flag is not optional. Without it the check passes for
an artifact attested by any workflow in the repository, including one an
attacker with write access could add.

## Decisions worth re-opening if you disagree

- **Placeholders rather than SHAs.** Looking a SHA up is a network operation
  and this lane made none. The alternatives were to pin by tag, which the whole
  of section 5.1 argues against, or to write a plausible looking hex string,
  which would be a fabricated fact in a file whose entire purpose is
  verifiable ones. A loud placeholder that fails closed was the least bad
  option, and `check_action_pins.py` is what keeps it from becoming permanent.
- **The release workflow builds as well as attests.** `handoffs/task18_platform.md`
  edit 5 proposes `.github/workflows/release.yml` doing the build, and that
  file does not exist. Splitting build and attestation across two workflows
  would mean attesting an artifact this run did not produce, which weakens the
  claim to the point of being misleading. **Do not apply edit 5 as well.**
  `release-provenance.yml` supersedes it and contains every step it specified,
  plus the guard, the digest handoff, the SBOM, the attestations, and the
  gated publish. If you prefer edit 5's file name, rename this one and update
  the PyPI publisher configuration, which is scoped to the workflow filename.
- **No auto-merge, including for Dependabot.** Section 7.2. The short version
  is that CI measures training correctness and cannot notice a workflow gaining
  a permission, and that automatically merging an action bump is automatically
  accepting whatever that action's maintainer pushed, which is the failure mode
  SHA pinning exists to prevent.
- **The `pip` Dependabot entry will find nothing for a while.** That is
  expected, not a misconfiguration, and the comment in the file says so.
- **`SECURITY.md` at the repository root** rather than under `.github/`.
  GitHub reads either. The root is where a person looks.

## Cross-lane reconciliation

Read at the end of this lane's work, from files other lanes landed while it was
running. Everything below is a statement about the tree at that moment and
needs re-checking by whoever integrates the round.

**Three lanes wrote a release workflow.** `.github/workflows/release-macos.yml`,
`.github/workflows/release-linux.yml`, and this lane's
`release-provenance.yml`. Two lanes also added a packaging directory,
`packaging/macos/` and `packaging/linux/`, alongside this lane's
`packaging/security/`.

**The agreement is worth recording first, because it is unusual.** Three lanes
that could not see each other's work independently landed on `permissions: {}`
at the workflow level, `workflow_dispatch` rather than any fork triggerable
event, a publish job gated by a protected environment, `id-token: write` as
that job's only permission, trusted publishing with no API token anywhere,
TestPyPI before PyPI, and SHA placeholders instead of invented pins. Three
derivations converging on the same posture is good evidence for all three.

Four things to settle before the round merges.

1. **Three publishing workflows means three trusted publishers.** A PyPI
   publisher is scoped to a workflow filename, so every workflow that can
   upload is a separate standing grant to upload, and each one has to be
   audited and revoked separately. That is the opposite of section 3.2's
   intent. The consolidation worth trying is one publish path called by the
   platform builds, with attestation and SBOM in it, so there is one identity
   and one thing to revoke. **Verify before relying on it:** which OIDC claim
   PyPI matches when a reusable workflow is involved is not obvious, and this
   lane did not check it. If the answer makes reuse awkward, the fallback is
   one publish workflow that the platform builds hand artifacts to, rather than
   a call.
2. **`release-macos.yml` triggers on a tag push as well as on dispatch.** With
   an environment gate in front of publication that is defensible, and it is
   still a wider door than this lane's design assumes, because a tag push
   starts a run that reaches a gate rather than a run that cannot start. Pick
   one convention across the workflows and write down which.
3. **`release-linux.yml` line 362 uses `pypa/gh-action-pypi-publish@release/v1`
   with a `TODO(pin)` comment.** That is the one unpinned reference in the
   round that will actually resolve and run, because `release/v1` is a live
   branch, and it sits in the job holding the publishing identity. The
   placeholder convention this lane used fails closed instead. Pin it, or
   replace it with a placeholder, before that workflow is runnable.
4. **Two `check_action_pins.py` and two hash tools.**
   `packaging/macos/check_action_pins.py` defaults to that lane's workflow and
   deliberately skips files it does not own; `packaging/security/check_action_pins.py`
   takes paths and defaults to all of `.github/workflows`, so it reports the
   tag pinned legacy workflows as failures. Both readings are defensible and
   the repository should have one script. This lane's argument for keeping the
   repository-wide default is that a checker which never mentions the files it
   does not own cannot tell you the repository is unpinned, which is the
   question worth asking before a release. Whichever survives, delete the
   other rather than leaving two that can disagree. The same choice applies to
   `packaging/macos/hash_artifacts.sh` and
   `packaging/security/hash_manifest.py`; the argument for the Python one is
   that a consumer verifying a download has a Python and may not have
   coreutils.

**The natural ownership split, if the directories stay separate.**
`packaging/macos` and `packaging/linux` own how a platform's artifact is built
and inspected, which is where the Mach-O, ELF, and toolchain specifics belong.
`packaging/security` owns the trust chain that is the same for every platform,
which is pinning, hashes, SBOM, attestation, publication, and revocation. No
attestation or SBOM step exists in either platform workflow today, so that
division is already what the files describe rather than a reorganization.

## Open questions

- **Who is the backup release owner?** Everything in section 3 has the same
  single point of failure and no file closes it.
- **Does the wheel get published at all before a self hosted runner exists?**
  The consistent answer from this lane's design is no, because a build
  provenance attestation from a personal laptop attests a personal laptop. The
  counter-argument, that an alpha wheel is worth more to users than a strong
  provenance claim, is a real one, and if it wins it belongs in
  `docs/RELEASE_SECURITY.md` as a stated exception rather than as a quiet one.
- **Which Syft version, and is `anchore/sbom-action` acceptable in a job with
  `id-token: write`?** It is a third party action in the one job that holds a
  signing identity. The alternatives are moving the SBOM into a separate
  unprivileged job and passing the file, or generating the SBOM from
  `pixi.lock` and the wheel with the standard library, which would be a
  second implementation of something Syft already does well. Pinning it to a
  SHA is the minimum; splitting the job is the stronger answer if the review of
  that action's diff ever stops being readable.
- **Should `sbom_supplement.py` also record the four dylibs' conda package
  provenance?** It records what is in the wheel and what the sidecar says. The
  conda package name, version, and channel URL that produced each library are
  in `pixi.lock` and would make the components matchable rather than merely
  identified. That is a real improvement and it needs a `pixi.lock` parser,
  which is a piece of work rather than a field.
- **Is there a Linux story for any of this?** The workflow is macOS only
  because the wheel is. A manylinux build would need its own runner, its own
  bundling step, and its own artifact validation rules, and none of the
  provenance design changes, which is the good news.
- **Does the model file parser deserve fuzzing before the first release?**
  `SECURITY.md` states the trust boundary honestly, which is the minimum. A
  corpus and a fuzz target would let it state something stronger, and it is the
  highest value security work in the repository that is not release plumbing.

## Deliberately not done

- No script was executed, no test was written or run, no build, no scanner, no
  signing tool, no workflow trigger, and no network call.
- No secret, no token, and no placeholder for one was added.
- No existing workflow, source file, Python metadata file, README, or other
  packaging directory was edited.
- No action was pinned to an invented SHA.
- No claim was made that anything here has been validated. Section 11 of
  `docs/RELEASE_SECURITY.md` lists every gap, and it is long on purpose.
