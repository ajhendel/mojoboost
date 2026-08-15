# Release security and provenance

How a mojotrees release is built, recorded, published, and revoked, and what a
stranger can check about an artifact without trusting the person who published
it.

[SECURITY.md](../SECURITY.md) is the other half of this subject. It is for
people who found a bug and need to report it. This document is for whoever cuts
a release, and for whoever has to clean up after a bad one.

## Status of this document

Nothing described here has run. There is no published release, no PyPI project,
no attestation, no SBOM, and no signed artifact
([docs/PLATFORM_MATRIX.md](PLATFORM_MATRIX.md)). One macOS wheel has been built
locally, on a development machine, and never distributed.

That is the reason to write this now. Every control below is cheaper to install
before the first publication than after it, and the first release is the only
one that can be designed rather than retrofitted. Where the design depends on
something that does not exist, the gap is named in section 11 rather than
smoothed over.

The configuration this document specifies lives in
[.github/workflows/release-provenance.yml](../.github/workflows/release-provenance.yml),
[.github/dependabot.yml](../.github/dependabot.yml), and
[packaging/security/](../packaging/security/). The settings it depends on, which
are not files, are listed in
[handoffs/release_10_security.md](../handoffs/release_10_security.md).

## 1. What is actually at risk

Take the threat model seriously and it stays short, because most of the usual
answers do not apply here. mojotrees has no server, no service, no account
system, no credential handling, and no network access at runtime.

What it does have is the thing that matters most in a supply chain, which is
that **`pip install mojotrees` runs a compiled native library inside somebody
else's process, with their privileges, usually on a machine that also holds
their data.** Nobody reads a `.so`. Whatever ships in the wheel is what runs.

### 1.1 The attacker worth defending against

One attacker, three paths to the same objective, which is getting their code
into the artifact a user installs.

| Path | What it looks like | What stops it here |
|---|---|---|
| Publish a bad artifact directly | A stolen PyPI credential, or an account takeover | No long lived PyPI credential exists to steal. Section 4. |
| Get bad code into the build | A pull request that changes a workflow or a packaging script, merged without a human reading it | No auto-merge, and no path from a fork pull request to a privileged job. Section 7. |
| Get bad code into a dependency | A compromised action, or a compromised package the build pulls in | Actions pinned by commit SHA, and a small, deliberately visible dependency surface. Sections 5 and 6. |

An alpha with a handful of users is not an uninteresting target. It is a
cheaper one, because the controls tend not to exist yet and the maintainer
tends to be one person who can be waited out. Being small is not a mitigation.

### 1.2 What is deliberately not in scope

- Protecting a user from a model file they chose to load. That is the trust
  boundary in SECURITY.md, and it is a code hardening problem, not a release
  problem.
- Reproducible builds, in the bit for bit sense. See section 11.
- Protecting against a compromise of the maintainer's own development machine
  while it is the machine that builds the artifact. See section 3.3, which says
  what that costs and does not pretend to solve it.

## 2. The chain a release has to produce

A release is credible when a stranger can walk this chain from the artifact
back to the source without asking anyone to vouch for anything.

```
source commit
  -> a workflow run whose definition is in that commit
    -> a build on a machine with the recorded toolchain
      -> a wheel with a recorded digest
        -> an attestation binding that digest to that workflow run
          -> an SBOM saying what is inside the wheel, attested to the same digest
            -> publication by an identity that only that workflow can hold
```

Every link is a file or a check in
[release-provenance.yml](../.github/workflows/release-provenance.yml). The
links break at different costs, which is why they are separate:

- A **digest** proves two files are identical. It says nothing about origin. A
  manifest published next to the artifact it describes is a courtesy to people
  with flaky networks and is not a security control.
- An **attestation** proves origin. It binds the digest to a workflow identity
  through a certificate that GitHub issued to an OIDC token, and it is recorded
  in a public transparency log. This is the link an attacker with write access
  to a download page cannot forge.
- An **SBOM** proves nothing at all by itself. It is a claim about contents,
  and it is worth exactly as much as the process that produced it, which is why
  it is attested to the same digest rather than published loose. Section 8.3
  says what this project's SBOM does and does not cover.

## 3. Who may release

### 3.1 The owner model

One person, named, with a named alternate when one exists.

| Role | Today | Responsibility |
|---|---|---|
| Release owner | Andrew Hendel (`ajhendel`), an owner of `mojotrees` | Approves the environment gate, and is accountable for what got published |
| Backup owner | None | Nobody else can publish, and nobody else can revoke |
| Security contact | The same person | Triage under SECURITY.md |

Every one of those being the same person is the largest weakness in this
document and it is not fixable by writing more of this document.

### 3.2 What the gate is

Publication is not a step in a build. It is a job with an `environment`
attached, which means GitHub holds the run and waits for a human to approve it,
and the approval is recorded. That is the only reason the split between the
`provenance` job and the `publish` job exists.

The environment is where the controls that cannot live in a workflow file go,
because a workflow file can be edited by whoever can push:

- required reviewers, REMOVED from `pypi` on 2026-08-15 at the owner's
  instruction. It had made approval a second action rather than a consequence
  of dispatching the run. On a single-maintainer project the approver and the
  dispatcher are the same person, so it authenticated nothing. Restore it the
  moment a second person has push access
- a deployment branch rule limiting the environment to `main` and to tags, so
  the release identity cannot be exercised from an arbitrary branch
- a wait timer, optional, and useful for exactly one reason, which is that it
  gives you time to cancel after realizing something is wrong

A single maintainer approving their own release is a formality in the sense
that it stops nobody malicious. It is not a formality in the sense that it
stops a dispatch made by an unattended process, a stale browser tab, or a token
that got somewhere it should not be.

### 3.3 What one person cannot do

Stated plainly rather than mitigated with a paragraph that sounds like a
mitigation.

- **No four eyes.** Nobody reviews the release owner's own commits. A
  compromise of the maintainer's account or laptop compromises the release, and
  none of the controls in this document prevent that. What they do is make it
  visible afterwards, because the attestation names the workflow run and the
  run names the commit.
- **The build host is a development machine.** The wheel must be built on
  Apple silicon with a Metal toolchain, GitHub hosted macOS runners do not have
  one, and no self hosted runner is registered. Until one is, the honest
  description of a locally built wheel is that its provenance is a personal
  assurance. Do not publish under that condition. Section 11.
- **Revocation is single threaded.** If the release owner is unreachable, a bad
  artifact stays up. The only real fix is a second person with PyPI ownership,
  which is an owner action, not a file.

## 4. Publishing without a credential

### 4.1 Trusted publishing, and why there is no token

PyPI trusted publishing exchanges a short lived OIDC token, minted by GitHub
for one workflow run, for an equally short lived upload token. Nothing is
stored. There is no secret in this repository, no secret in an environment, and
nothing to leak, rotate, or exfiltrate from a compromised runner.

A long lived PyPI API token is the opposite of all of that. It works from
anywhere, forever, for anyone who has the string, and it is only as safe as
every place it has ever been pasted. A repository secret is readable by any
workflow the repository runs, which includes a workflow added by a pull request
under some configurations, and by anyone with admin access.

**Therefore, and this is a rule rather than a preference: no `PYPI_API_TOKEN`,
no `TWINE_PASSWORD`, no `password:` input, and no placeholder for any of them
appears in this repository.** A placeholder is not neutral. It is a shaped hole
that the next person fills, and the way it gets filled is with a token.

If trusted publishing cannot be configured for some reason, the correct action
is to not publish. An alpha does not need to be on PyPI badly enough to be
worth a permanent credential.

### 4.2 The publisher configuration

Configured on PyPI, not here. Before the project exists on PyPI this is a
*pending publisher*, which is what lets the first upload be a trusted one
rather than a manual upload that establishes the project.

| Field | Value |
|---|---|
| Owner | `mojotrees` |
| Repository | `mojotrees` |
| Workflow | `release-provenance.yml` |
| Environment | `pypi` |

The environment name is part of the publisher identity, which is what makes
section 3.2's gate load bearing rather than decorative. A run that skips the
environment gets a token PyPI will not accept.

TestPyPI gets its own pending publisher with environment `testpypi`, and the
first end to end rehearsal goes there.

### 4.3 What the publish job may contain

- `permissions: id-token: write` and nothing else. It does not check the
  repository out, so it needs no `contents` at all.
- No step that runs code from this repository. The fewer things happen in the
  job holding the publishing identity, the smaller the question "what could
  have run here" is.
- One directory containing exactly one wheel and nothing else, checked before
  the upload rather than assumed, because the publisher uploads what it finds.

## 5. Actions, pins, and permissions

### 5.1 Pin by commit SHA

Every `uses:` in a workflow that touches a release must be
`owner/repo@<40 hex>` with a trailing comment naming the version. A tag is a
name its owner can move, so `@v4` is a promise from a third party to keep being
trustworthy, renewed silently on every run. A commit SHA is a fact.

This is not hypothetical for actions in particular. The 2025 `tj-actions`
compromise worked exactly this way, by repointing tags at malicious code, and
repositories that had pinned by SHA were unaffected while repositories that had
pinned by tag were not.

Pinning is enforced mechanically by
[packaging/security/check_action_pins.py](../packaging/security/check_action_pins.py),
which is standard library only and runs on a bare checkout, in the same spirit
as `tools/check_parity.py`. The release workflow runs it on itself as the first
job, before a self hosted runner or an OIDC token is involved.

The trailing version comment is required and not decoration. Nobody can review
a diff of two hex strings. Dependabot writes and updates that comment for you,
which is what makes SHA pinning survivable for one maintainer.

**Current state, stated because the checker will say it anyway.** `ci.yml` and
`gpu-validation.yml` pin by tag, and neither is owned by this lane.
`release-provenance.yml` carries `@REPLACE_WITH_SHA` placeholders rather than
SHAs, so it fails closed until somebody looks each action up. Handing over a
release workflow that cannot run is better than handing over one that runs
against whatever a tag points to today.

### 5.2 Permissions

The workflow declares `permissions: {}` at the top level, which denies
everything, and each job adds back exactly what it needs.

| Job | Permissions | Why |
|---|---|---|
| `guard` | `contents: read` | Checks out and runs one standard library script |
| `build` | `contents: read` | Builds. It never signs and never publishes |
| `provenance` | `contents: read`, `id-token: write`, `attestations: write` | Mints the OIDC token the attestation exchanges for a certificate, and writes the result to the repository's attestation store |
| `publish` | `id-token: write` | Exchanges an OIDC token for a PyPI upload token. Nothing else |

Two properties follow from that table and both are deliberate. The job on the
machine this project does not own the image of, the self hosted build, has the
weakest token in the run. And the two jobs that hold an identity run on hosted
runners whose contents are GitHub's responsibility rather than this project's.

`checkout` is called with `persist-credentials: false` everywhere, and it
matters most on the self hosted runner, where the workspace survives the job
and a credential left in `.git/config` outlives the run that created it.

### 5.3 Self hosted runners

A self hosted runner is a machine that executes whatever a workflow tells it
to, keeps state between runs, and usually sits on somebody's home network. The
rule that makes one survivable is that **no untrusted code may ever reach it**,
which here means the release workflow has no event a fork can trigger. Section
7.

## 6. Dependencies

### 6.1 What the supply chain actually is

| Layer | Where it is pinned | Visible to Dependabot |
|---|---|---|
| GitHub Actions | `uses:` lines in `.github/workflows` | Yes |
| Python distribution metadata | `python/pyproject.toml` | Yes, and there is almost nothing there |
| Mojo, MAX, and every conda package the build uses | `pixi.toml`, `pixi.lock` | **No** |
| The four MAX runtime libraries shipped inside the wheel | `pixi.lock`, copied by `packaging/build_wheel.sh` | **No** |
| CPython | The MAX package's own dependency, `python 3.14.*` | **No** |

The two rows that ship code to users are the two rows nothing watches. That is
the honest summary of this project's dependency posture, and no configuration
file changes it, because Dependabot has no conda or pixi ecosystem.

### 6.2 What automation does

[.github/dependabot.yml](../.github/dependabot.yml) opens weekly grouped pull
requests for GitHub Actions, and watches `python/pyproject.toml` so that the
first pinned runtime dependency added there is already covered.

Repository settings, not files, enable the dependency graph, Dependabot alerts,
and Dependabot security updates. Those are owner actions and they are listed in
the handoff.

### 6.3 What stays manual

The toolchain. Updating it is `pixi update`, reviewing the `pixi.lock` diff,
running the suite, and rebuilding the wheel, because a MAX version change moves
the libraries bundled inside the artifact and can move the Python floor the
whole matrix is derived from
([packaging/matrix/platform_matrix.toml](../packaging/matrix/platform_matrix.toml)).

Do it on a schedule rather than on an alert, because there will be no alert.
Before every release is the natural point, and it is item **F3** of
[packaging/security/release_checklist.md](../packaging/security/release_checklist.md).

### 6.4 How an update is reviewed

- An action bump is read as a diff of the upstream repository between the two
  SHAs, not as a version number going up. If the diff is not readable in a
  reasonable time, that is information about whether the action belongs in a
  privileged job.
- A dependency added to `python/pyproject.toml` is a decision about what every
  user installs, and it gets the scrutiny of a public surface change under
  [docs/COMPATIBILITY_POLICY.md](COMPATIBILITY_POLICY.md) section 2.
- A `pixi.lock` change is reviewed for what moved, not that something moved.
  The interesting lines are `mojo`, `max`, `python`, and anything whose file
  ends up inside the wheel.

## 7. Untrusted contributions

### 7.1 No fork pull request reaches anything privileged

The release workflow triggers on `workflow_dispatch` and on nothing else. There
is no `push`, no `release`, no `pull_request`, and above all no
`pull_request_target`.

`pull_request_target` is the specific hazard. It runs the workflow definition
from the base branch, which sounds safe, but it does so **with a read write
token and access to secrets, in the context of the base repository**, while the
pull request's own code sits in the checkout. Any step that checks out the pull
request head, or runs a build script, or executes a task the fork's files
define, hands that token to the fork's author. It is the single most common way
repositories are compromised through Actions.

The defense is structural rather than careful. If no event a fork can cause
reaches a privileged job, no review mistake in a privileged job can be
triggered by a fork.

The existing `ci.yml` does run on `pull_request`, which is correct: that
workflow gets a read only token, holds no secrets, and runs on hosted runners.
A fork pull request running the test suite is the point of CI. The line this
document draws is between that and anything holding an identity.

### 7.2 Why arbitrary changes are never merged automatically

Auto-merge is worth wanting. One maintainer, a queue of small dependency bumps,
and a green CI run is exactly the situation it was built for. It is still not
enabled here, and the reason is not a general nervousness about automation.

**A workflow file is code that runs with the repository's permissions, and CI
passing is not a review of it.** A pull request that edits
`.github/workflows/` can add a step that reads a secret, request extra
permissions, add a trigger, or point an action at a different SHA, and the test
suite will pass, because the test suite tests mojotrees and not the workflow.
The signal auto-merge consumes does not measure the risk auto-merge takes.

The same is true of `packaging/`. A change to `build_wheel.sh` changes what
goes into the artifact users install. A change to a `codesign` or
`install_name_tool` invocation changes what the artifact loads at runtime. A
change to a validation script changes which checks a release is allowed to
skip. None of that is exercised by CI today, because no CI job builds a wheel.

Three more reasons specific to this repository:

- **CI is not a security check.** Green means the suite passed, and the suite
  measures training correctness. There is no job that would notice a workflow
  gaining `contents: write`.
- **A bot author is not a trusted author.** A Dependabot pull request is a
  proposal to run somebody else's code, and its trustworthiness is the
  trustworthiness of the upstream it is bumping to, which is precisely the
  thing that gets compromised in an action supply chain attack. Automatically
  merging an action bump is automatically accepting whatever that action's
  maintainer pushed, which is the entire failure mode SHA pinning exists to
  prevent. Pinning and auto-merging together buy nothing.
- **Alpha is a reason for more care, not less.** The users of an alpha are
  early adopters running unfamiliar binaries because they trust the project.
  The argument that a compromise would not matter much because the user count
  is low is an argument about the maintainer's exposure, not the users'.

What is acceptable, when the volume justifies it, is a narrow rule rather than
a general one: auto-merge for patch level bumps of dependencies that never
reach a privileged job or a shipped artifact, with an explicit path filter that
excludes `.github/**` and `packaging/**`, and never for a dependency that is
part of the build. That is a change to this section and to the release checklist
when it happens, not an on switch in a settings page.

## 8. Hashes, provenance, and SBOM

### 8.1 Hashes

[packaging/security/hash_manifest.py](../packaging/security/hash_manifest.py)
writes a coreutils compatible `SHA256SUMS` next to the artifacts, using the
standard library so a consumer needs nothing installed to check it. The digest
is computed on the machine that built the file, carried forward as a job
output, and re-checked after every artifact download inside the workflow, so
artifact storage is not a trust boundary the run depends on.

What it is worth is in section 2. Integrity, not origin.

### 8.2 Build provenance attestation

`actions/attest-build-provenance` produces a signed SLSA style statement
binding the artifact's digest to the workflow run that made it. GitHub mints a
short lived certificate for the run's OIDC identity, the signature goes into a
public transparency log, and the bundle is stored with the repository. No key
material exists in this project, so none can leak.

The consumer side is `gh attestation verify`, and the flag that matters is
`--signer-workflow`, because without it any workflow in the repository
satisfies the check, including one an attacker with write access could add.
[packaging/security/verify_release.sh](../packaging/security/verify_release.sh)
uses it.

PyPI attestations (PEP 740) are a second, independent record on the index
itself. `pypa/gh-action-pypi-publish` generates them from the same trusted
publishing identity; confirm the input's default when the action is pinned,
rather than assuming it.

### 8.3 SBOM, and what this one is missing without help

The SBOM is generated by Syft over the built wheel in CycloneDX JSON, then
supplemented by
[packaging/security/sbom_supplement.py](../packaging/security/sbom_supplement.py),
and attested to the same digest as the wheel.

The supplement is not tidying. A scanner reads a wheel as a Python
distribution, finds `mojotrees 0.1.0`, and stops. It does not report that the
artifact contains a compiled Mojo extension and four MAX runtime libraries
copied out of a conda environment, which is most of the wheel by bytes and all
of it by risk. An SBOM that omits the shipped native runtime is worse than no
SBOM, because it reads as complete to whoever is ticking a box.

The supplement opens the wheel, hashes each native member, and adds a component
per member, plus the Mojo and MAX versions read from the provenance sidecar,
which cannot be recovered from the artifact afterwards. It records `unknown`
where the sidecar recorded nothing, and never a plausible guess.

Known limits, so nobody reads more into the file than is there:

- There is no upstream vulnerability feed keyed to a MAX runtime library, so
  these components are identification rather than a matchable coordinate. The
  `pkg:generic` purls say what is present, not what is known about it.
- The SBOM does not enumerate CPython or the operating system libraries the
  extension links from the platform.
- It is not a license inventory. No license is asserted for any added
  component.

### 8.4 The provenance sidecar

`<wheel>.provenance.json` records the toolchain versions, the `pixi.lock`
digest, the commit, the build host, and the compile time accelerator answer.
None of that survives in the wheel, so it is written at build time or lost.
`packaging/matrix/validate_artifact.py` rule R7 is the authority on which
fields are required, and the sidecar was specified in
[handoffs/task18_platform.md](../handoffs/task18_platform.md) edit 4.

The release workflow treats its absence as a hard stop rather than filling the
fields itself, because two places writing the same facts differently is how
they end up disagreeing. Which script writes it is a packaging question rather
than a security one; see the cross-lane note in
[handoffs/release_10_security.md](../handoffs/release_10_security.md).

## 9. When a release is compromised

"Compromised" means a published artifact is not what it should be, or was
produced by a process that cannot be shown to be clean. A build machine
compromise, a stolen account, an unexplained artifact, or a dependency found to
have been malicious at build time all land here.

### 9.1 First hour

1. **Do not delete anything yet.** Evidence first. Record the artifact
   filenames, digests, upload times, and the workflow run IDs and their logs,
   and take a copy of any suspect artifact.
2. **Stop further publication.** Remove the trusted publisher on PyPI and
   disable the `pypi` environment. Both take seconds and both are reversible,
   and together they mean no further upload can succeed regardless of what an
   attacker controls in the repository.
3. **Yank the affected versions.** Section 10. Yanking is fast, reversible, and
   does not break anyone who pinned. It is always the correct first move and it
   is never the last one.
4. **Say something.** A one paragraph GitHub Security Advisory that names the
   version, the digests, and says the investigation is in progress beats a
   perfect statement issued a day later. Users are installing in the meantime.

### 9.2 Same day

5. **Establish the blast radius.** Which versions, which files, which digests,
   and which uploads. Compare every published digest against the attestation
   for that digest. An artifact with no attestation, or one whose attestation
   names a workflow run that does not exist or does not match, is the finding.
6. **Establish the entry point.** Account, laptop, runner, dependency, or
   workflow change. The commit history of `.github/**` and `packaging/**` is
   the first place to look, because it is where an attacker has to go to make a
   change stick.
7. **Rotate everything the entry point touched.** No PyPI credential exists to
   rotate, which is the point of section 4. GitHub account credentials, SSH
   keys, personal access tokens, and any self hosted runner registration token
   do exist and get rotated.
8. **Rebuild the self hosted runner from scratch** if it was in scope. A
   runner's workspace persists, so cleaning it is not credible; replacing it is.

### 9.3 Before anything is published again

9. **Fix the entry point**, and record the fix in this document if it is a
   process gap rather than a bug.
10. **Publish a new version.** Never re-upload a fixed file under an old
    version number. PyPI forbids reusing a filename, and even where a registry
    allows it, a version that changed contents is undetectable to everyone who
    already installed it.
11. **Publish the full advisory.** Affected versions, digests of the bad
    artifacts, the fixed version and its digest, the entry point, the timeline,
    and what a user should do to check whether they got a bad copy. Naming the
    digests is the part that lets a user answer that question themselves.
12. **Update the checklist**
    ([packaging/security/release_checklist.md](../packaging/security/release_checklist.md))
    so the next release checks for whatever this one missed.

### 9.4 What cannot be undone

- **A signature is not revocable.** Sigstore signs and logs; there is no
  revocation list. If a malicious artifact was signed by a legitimate workflow
  run, its attestation will verify forever. Revocation is a statement, meaning
  an advisory naming the bad digests, not a cryptographic operation.
- **A deleted PyPI file does not come back**, and its version number can never
  be reused.
- **An installed wheel stays installed.** Users have to act. That is the entire
  reason step 4 happens in the first hour rather than after the analysis.

## 10. Yank, delete, and revoke

Three different actions, routinely confused, with different consequences.

| Action | What happens | Reversible | When |
|---|---|---|---|
| **Yank** (PEP 592) | The file stays on PyPI. Resolvers skip it unless a requirement pins that exact version, in which case it still installs, with a warning | Yes | Almost always the right first action |
| **Delete** | The file is gone. Anything pinned to it breaks immediately. The version number is burned permanently | **No** | Only for an artifact that is actively harmful to install |
| **Revoke the publisher** | The trusted publisher and the environment are removed, so no further upload can happen | Yes | Immediately on any suspected compromise |

### 10.1 Yanking

Yank when a release is wrong but not dangerous. A broken wheel, a bad platform
tag, an artifact that does not match its attestation, a version published by
mistake.

1. PyPI, the project's Manage page, Releases, the version, Options, Yank, with
   a one line reason. The reason is shown to users by some tools, so write it
   for them and not for yourself.
2. Say so in the release notes and in a GitHub advisory when the cause is a
   security issue.
3. Ship the replacement as a new version. Yanking is not a fix, it is a way to
   stop the wrong thing being chosen by default while the fix is prepared.

Yanking is reversible, which is why the bar for doing it is low. A yank that
turns out to be unnecessary costs a click.

### 10.2 Deleting

Delete only when leaving the file installable is worse than breaking every
pinned dependency on it, which in practice means the artifact contains
malicious code or leaks a credential. A yanked file is still installable by
anyone who pins it, and an attacker who can influence a pin can therefore still
serve it.

Deletion burns the version number forever, so the replacement is `0.N.P+1` and
never a re-upload. Record in the release notes that the version was deleted and
why, because a gap in the version sequence with no explanation is exactly the
kind of thing that erodes trust in a project.

### 10.3 Revoking the ability to publish

Fast, and worth doing on suspicion rather than on confirmation.

1. PyPI, Manage, Publishing, remove the trusted publisher.
2. GitHub, Settings, Environments, delete or disable `pypi`, which also removes
   its reviewers and branch rules.
3. Confirm no `secrets.PYPI_*` exists to remove, which under section 4 should
   be a five second check with a foregone conclusion. If one does exist,
   somebody violated section 4 and that is its own finding.

Restoring means re-creating the publisher and the environment deliberately,
which is the correct amount of friction for turning publication back on.

### 10.4 GitHub side

Yanking has no GitHub equivalent. What exists instead:

- Mark the GitHub release as a pre-release or delete it, and edit its notes to
  say plainly at the top what happened.
- Delete the artifacts attached to the release, since they are copies of the
  same bytes.
- Do not delete or force push the tag. A moved tag makes every existing
  reference to that commit wrong, silently, which is the same problem as an
  action tag being repointed.
- Publish the advisory. It is the durable record and it is what shows up in
  tooling.

## 11. Not implemented

Every item here is a real gap. None of it is scheduled, and listing it is not a
commitment to close it.

- **No release has been published**, so every mechanism in this document is
  untested end to end. The first rehearsal goes to TestPyPI.
- **No self hosted macOS runner exists**, so the build job cannot run. Until it
  does, a wheel is built on a personal machine, and a personal machine cannot
  produce a build provenance attestation worth the name.
- **Actions are not yet pinned to real SHAs.** The placeholders in the release
  workflow fail closed, and `ci.yml` and `gpu-validation.yml` are pinned by tag
  and are outside this lane's ownership.
- **The provenance sidecar has never been produced by a build**, and which
  script writes it was still being settled across lanes when this was written.
  Everything downstream of it, meaning the SBOM supplement and rule R7, is
  blocked on that. Section 8.4.
- **Builds are not reproducible.** Two builds of the same commit are not
  expected to produce identical bytes, and nothing checks. That means the
  attestation is the only link between the artifact and the source, with no
  independent way to confirm it.
- **No fuzzing, no adversarial corpus, and no memory safety testing** of the
  model file parsers. SECURITY.md states this as a trust boundary rather than
  hiding it.
- **No Linux artifact exists**, so nothing here has been exercised against
  `auditwheel`, `patchelf`, or an ELF runtime bundling step, all of which will
  raise questions this document does not answer.
- **No second maintainer**, which section 3.3 covers, and which is the gap most
  likely to matter first.

## 12. Verification commands

Exact, and **none of them has been run**, because nothing they inspect exists
yet. They are written down so the first release is checked against a
pre-existing standard rather than against whatever was convenient afterwards.

Repository side, before a release:

```sh
# Every action in the release workflow is pinned to a commit SHA
python3 packaging/security/check_action_pins.py .github/workflows/release-provenance.yml

# The same check over every workflow. Expected to fail today: ci.yml and
# gpu-validation.yml pin by tag
python3 packaging/security/check_action_pins.py .github/workflows

# No credential shaped string in anything that executes. The include filters
# are what keep this useful: this document and the release checklist both name
# those strings in order to prohibit them, and a check that always matches is a
# check nobody runs twice.
#
# Expected: every match is a detection pattern rather than a credential. A
# secret scanner has to contain the string it looks for, so read each match and
# do not merely count them. Anything that is an assignment, an input, or a
# value is the finding.
grep -rInE 'PYPI_API_TOKEN|TWINE_PASSWORD|pypi-AgEIcHlwaS5vcmc' \
    --include='*.yml' --include='*.yaml' --include='*.sh' --include='*.py' \
    --include='*.toml' --include='*.mojo' \
    --exclude-dir=.git --exclude-dir=.pixi .

# No fork triggered event reaches a privileged job. Matches the trigger key
# only, not the comments that explain why it is absent. Expected: no matches
grep -rnE '^\s*pull_request_target\s*:' .github/workflows/

# The release matrix still agrees with the repository
python3 packaging/matrix/validate_matrix.py
```

Artifact side, after a build and before publishing:

```sh
WHEEL=$(ls python/dist/mojotrees-*.whl)

# What is in the wheel, and what it links
python3 packaging/matrix/validate_artifact.py "$WHEEL"
unzip -l "$WHEEL"
otool -l python/mojotrees/_mojotrees.so | grep -A 4 LC_BUILD_VERSION

# The manifest, and a re-check of it
python3 packaging/security/hash_manifest.py write python/dist/SHA256SUMS \
    "$WHEEL" "$WHEEL.provenance.json"
python3 packaging/security/hash_manifest.py verify python/dist/SHA256SUMS
```

Consumer side, after a release exists:

```sh
# Both questions, in order, with an explanation of which is which
bash packaging/security/verify_release.sh ./mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl ./SHA256SUMS

# Provenance on its own. --signer-workflow is not optional: without it, any
# workflow in the repository satisfies the check
gh attestation verify ./mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl \
    --repo mojotrees/mojotrees \
    --signer-workflow mojotrees/mojotrees/.github/workflows/release-provenance.yml

# The attested SBOM, as JSON
gh attestation verify ./mojotrees-0.1.0-cp314-cp314-macosx_26_0_arm64.whl \
    --repo mojotrees/mojotrees \
    --predicate-type https://cyclonedx.org/bom \
    --format json

# Install only what a hash file allows, which is the strongest thing pip offers
python3 -m pip install --require-hashes -r requirements.txt
```

Incident side:

```sh
# What was published, and when
gh api /repos/mojotrees/mojotrees/actions/workflows/release-provenance.yml/runs \
    --jq '.workflow_runs[] | {id, created_at, event, actor: .actor.login, conclusion}'

# Every attestation this repository holds for a digest
gh api /repos/mojotrees/mojotrees/attestations/sha256:<digest>

# Confirm no publishing secret exists. Expected: an empty list
gh secret list --repo mojotrees/mojotrees
gh secret list --repo mojotrees/mojotrees --env pypi
```
