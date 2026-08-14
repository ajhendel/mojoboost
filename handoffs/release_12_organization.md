# Task 12 handoff: organization and community launch

Lane files (the only ones this lane touched, all new):

- `GOVERNANCE.md`
- `SUPPORT.md`
- `CODE_OF_CONDUCT.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/profile/README.md`
- `launch/ORG_MIGRATION_CHECKLIST.txt`
- `launch/CONTRIBUTOR_INVITE.txt`
- `handoffs/release_12_organization.md` (this file)

`README.md`, `CONTRIBUTING.md`, `SECURITY.md`, the six existing launch posts,
`launch/README.txt`, the issue forms, the workflows, source, and packaging were
read and not edited. This lane committed and staged nothing.

> **Take the working tree, not the commit.** A concurrent session committed a
> mid-task snapshot of this lane's files in `9a9c8d1 "Prepare packaging and
> parallel optimization work"`, bundled with many other lanes' work. That
> snapshot is not wrong, it is early. It predates the reconciliation with the
> hardware contributor lane, so its `SUPPORT.md` still routes hardware results
> to the demoted `hardware_validation.yml` form, and its
> `.github/profile/README.md` and `launch/CONTRIBUTOR_INVITE.txt` still
> describe a generic validation procedure rather than the capture scripts that
> now exist. Four files hold an uncommitted delta over `9a9c8d1`:
> `SUPPORT.md`, `.github/profile/README.md`, `launch/CONTRIBUTOR_INVITE.txt`,
> and this handoff. The other four are committed and current, and the
> reconciliation with the security lane made it into the snapshot. Nothing from
> any other lane was overwritten; the only files this lane wrote are the eight
> listed above.

> **No GitHub state was changed.** No organization exists, no repository was
> transferred, no setting was touched, no team was created, no Discussion was
> enabled, and no message was sent. Every action in this lane's output is a
> written instruction for a human. The full inventory of what remains undone is
> in [Settings not performed](#settings-not-performed), which is the section to
> read if you read only one.

## What landed

**`GOVERNANCE.md`** is the substantive document. It states the mission, the
Apache-2.0 inbound-equals-outbound position with no contributor license
agreement, the benevolent-maintainer decision model and its three limits
(decisions in public, reasons given, the evidence rule binding the maintainer
too), the merge policy, the path to Write access, how maintainers are added and
removed, the four-step conflict path ending in disagree-and-commit or fork, a
short security section that defers to `SECURITY.md`, and how the document
itself changes.

**`SUPPORT.md`** routes questions. A table maps intent to destination,
including three Discussions categories by slug, and the rest sets expectations
honestly: nobody is on call, a week of silence is not a rejection, and there is
no commercial support of any kind.

**`CODE_OF_CONDUCT.md`** adapts Contributor Covenant 2.1 under CC BY 4.0 with
attribution, and adds a section on what the standard means on a technical
project, since the realistic failure mode here is impatience wearing technical
clothing rather than overt abuse. It names the specific limitation that a
one-maintainer project cannot handle a report about its only maintainer, and
gives GitHub's own abuse reporting as the independent route.

**`.github/PULL_REQUEST_TEMPLATE.md`** is comment-only prompts, five short
sections, with the first line of the file stating that no approval is needed
and an unfinished template never blocks a merge. It mirrors what
`CONTRIBUTING.md` already asks a pull request to say.

**`.github/profile/README.md`** is the organization profile text, staged in
this repository for review and non-functional at this path. It renders only
from `profile/README.md` in a repository named `.github` owned by the
organization, which the migration checklist covers. An HTML comment at the top
of the file says so, and should be deleted when the file is copied across.

**`launch/ORG_MIGRATION_CHECKLIST.txt`** is fourteen sections of exact manual
steps, A through N, from creating the organization to what happens after the
move.

**`launch/CONTRIBUTOR_INVITE.txt`** is six copy-and-paste variants with
`<PLACEHOLDER>` slots: two cold invites split by background (Mojo and GPU
versus gradient boosting), a hardware-validation ask, a short form for chat, a
message for after somebody's first contribution merges, and one offering Write
access to an active contributor.

## The merge policy, and why it is shaped this way

The brief was that no ordinary alpha contribution should wait on a human
approval, while untrusted workflow, packaging, and release changes must not
merge automatically. Those pull in opposite directions and the resolution is a
path list rather than a person list.

Contributors with Write access push directly to `main`. Everyone else opens a
fork pull request that merges on a machine check with three conditions:
continuous integration green on every required job, no reserved path touched,
and the branch not stale. Reserved paths are `.github/workflows/**`,
`.github/actions/**`, `.github/dependabot.yml`, `pixi.toml`, `pixi.lock`,
`python/pyproject.toml`, `packaging/**`, `tools/**`, `LICENSE`,
`GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`. A reserved-path
change merges when a maintainer has read the diff, which is the only queue in
the process, and it applies to dependency bot pull requests exactly as it
applies to a stranger.

Two mechanics matter and are easy to get wrong.

**GitHub auto-merge needs something to wait for.** A pull request with no
pending requirement is immediately mergeable, and the auto-merge affordance
does not appear. So the ruleset has to require the status checks even though it
requires no review.

**Requiring status checks would also block trusted direct pushes**, which is
the whole point of giving people Write access. The fix is a ruleset rather than
classic branch protection, with `maintainers` and `contributors` in the bypass
list. Section I of the checklist has the exact configuration and tells the
operator to verify the bypass with a trivial direct push, because a bypass list
that did not take looks identical to one that did until somebody pushes.

**The automatic part does not exist yet.** GitHub's auto-merge must be turned
on per pull request, by a person or by a workflow. Workflow files were out of
scope for this lane, so today a maintainer clicks "Enable auto-merge", which is
one second of human involvement and not an approval. Section I5 specifies the
workflow that would remove even that, including the constraint that a
`pull_request_target` job must not check out or execute the fork's code and
must compute the reserved-path check itself. Whoever owns workflows should read
that before writing it.

## Reconciliation with concurrent lanes

Two lanes landed files in the working tree while this one was writing, and both
changed what these documents should say. Both were reconciled by editing this
lane's files only. Nothing belonging to either lane was touched.

### The hardware contributor lane

`docs/HARDWARE_CONTRIBUTORS.md`, `hardware/capture/`, `hardware/templates/`,
and a fourth issue form `hardware_result.yml` appeared after the first drafts
were written. That form positions itself as the primary route and demotes the
existing `hardware_validation.yml` to the prose alternative it explicitly still
accepts.

Three files here were routing people to the demoted form, so they were
repointed. `SUPPORT.md` now sends hardware results through the protocol
document and the machine-readable form, and says plainly that the validated set
is one Apple M4, so "is my hardware supported" is usually a finding rather than
a support question. `CONTRIBUTOR_INVITE.txt` variant 3 now names the capture
script and the record template instead of describing a generic procedure, which
matters because that variant is the highest-value ask in the file and its
credibility rests on the work being genuinely bounded. It also warns about the
"skipped, no accelerator" trap that the new form calls out. The profile README
row says the protocol was written for people who do not want to learn Mojo,
which is that document's own framing and a better recruiting line than anything
this lane would have invented.

### The security lane

`SECURITY.md`, `.github/dependabot.yml`, and
`.github/workflows/release-provenance.yml` appeared in the working tree from
another lane partway through this one, after `GOVERNANCE.md` and `SUPPORT.md`
had been drafted with their own security sections.

Those drafts were rewritten to defer rather than compete. `SECURITY.md` is now
named as authoritative and holds the reporting channel, scope, supported
versions, response targets, and disclosure terms. `GOVERNANCE.md` keeps only
the three governance-level facts (maintainers act on reports, security fixes
take the reserved-path route regardless of urgency, a reckless security failure
is grounds for removal), and `SUPPORT.md` points at the file. The earlier
drafts had different response times, seven and thirty days against that file's
three and ten business days, and two published policies disagreeing about how
fast a vulnerability gets acknowledged is worse than either one alone.

One consequence for the contact address. These documents originally used a
`MAINTAINER_CONTACT_EMAIL` placeholder to avoid publishing a personal address,
but `SECURITY.md` and `pixi.toml` already publish `ajhendel@gmail.com`, so
`CODE_OF_CONDUCT.md` now uses the same address rather than shipping a
placeholder or introducing a second contact route. Step A2 of the checklist
raises moving both to a role alias as a deliberate decision, with the command
to find every occurrence.

Two loose ends belong to that lane and not this one. `SECURITY.md` references
`docs/RELEASE_SECURITY.md` repeatedly and that file is not in the tree.
`.github/workflows/release-provenance.yml` has `@REPLACE_WITH_SHA` on every
`uses:` line, so it is a draft rather than a runnable workflow. Both are noted
in the checklist rather than fixed here.

## Settings not performed

Everything below is GitHub state that no file in a repository can express. None
of it has been done. The checklist section is given for the exact steps.

### Organization

| Setting | Status | Checklist |
| --- | --- | --- |
| `mojoboost-ml` organization created | **NOT PERFORMED** | B1 |
| Organization name availability confirmed | **NOT VERIFIED** | A1 |
| Profile display name, description, URL, avatar | **NOT PERFORMED** | B2 |
| Two-factor requirement for all members | **NOT PERFORMED** | B3 |
| Base permissions set to No permission | **NOT PERFORMED** | B4 |
| Repository creation restricted to owners | **NOT PERFORMED** | B4 |
| `mojoboost-ml/.github` repository created | **NOT PERFORMED** | C1 |
| `profile/README.md` published to it | **NOT PERFORMED** | C2 |

### Transfer

| Setting | Status | Checklist |
| --- | --- | --- |
| Repository transferred to `mojoboost-ml` | **NOT PERFORMED** | D1, D2 |
| Redirect from the old location confirmed | **NOT VERIFIED** | D3 |
| Actions secrets and variables checked after transfer | **NOT VERIFIED** | E1 |
| Organization Actions allow list configured | **NOT PERFORMED** | E2 |
| A full green continuous integration run at the new location | **NOT VERIFIED** | E3, L2 |
| Repository description and topics | **NOT PERFORMED** | E5 |
| Local and collaborator remotes updated | **NOT PERFORMED** | F1 |
| In-repository links updated to the new owner | **NOT PERFORMED** | G |

### Teams

| Setting | Status | Checklist |
| --- | --- | --- |
| `maintainers` team created, Admin on the repository | **NOT PERFORMED** | H1 |
| `contributors` team created, Write on the repository | **NOT PERFORMED** | H2 |
| Team notification defaults | **NOT PERFORMED** | H3 |
| A second maintainer identified or added | **NOT PERFORMED** | H, and `GOVERNANCE.md` |

### Merge policy

| Setting | Status | Checklist |
| --- | --- | --- |
| "Allow auto-merge" enabled | **NOT PERFORMED** | I1 |
| "Automatically delete head branches" enabled | **NOT PERFORMED** | I1 |
| Ruleset on `main` with required status checks | **NOT PERFORMED** | I2 |
| Bypass list containing both teams | **NOT PERFORMED** | I3 |
| Required check names matched against a real run | **NOT VERIFIED** | I4 |
| Per-pull-request auto-merge workflow | **NOT WRITTEN**, out of scope | I5 |
| Fork workflow approval set to the low-friction option | **NOT PERFORMED** | I7 |
| Workflow token permissions confirmed read-only | **NOT VERIFIED** | I8 |
| Secrets withheld from fork pull request workflows | **NOT VERIFIED** | I8 |

### Discussions

| Setting | Status | Checklist |
| --- | --- | --- |
| Discussions enabled on the repository | **NOT PERFORMED** | J1 |
| Categories `q-a`, `ideas`, `show-and-tell`, Announcements | **NOT PERFORMED** | J2 |
| The three slugs in `SUPPORT.md` confirmed to resolve | **NOT VERIFIED** | J3 |
| Seed Announcement and Q&A posts | **NOT PERFORMED** | J4 |

### Security

| Setting | Status | Checklist |
| --- | --- | --- |
| Private vulnerability reporting enabled | **NOT PERFORMED** | K1 |
| Secret scanning and push protection enabled | **NOT PERFORMED** | K2 |
| Dependabot alerts and security updates enabled | **NOT PERFORMED** | K3 |

### Outward-facing actions

| Action | Status | Checklist |
| --- | --- | --- |
| Any launch post published | **NOT PERFORMED** | L3 |
| Any contributor invitation sent | **NOT PERFORMED** | L4 |
| Any self-hosted runner registered | **NOT PERFORMED** | I9 |
| Any package name reserved on any index | **NOT PERFORMED**, out of scope | M1 |

## Known gaps

**The profile README links are dead until the transfer.** Every URL in
`.github/profile/README.md` and the three Discussions links in `SUPPORT.md`
point at `github.com/mojoboost-ml`, per the brief's instruction to assume that
location. GitHub's redirect works from the old location to the new one and not
the other way, so these resolve only after step D2 and step J2. If the transfer
is deferred, these two files are the ones that read as broken.

**The `contributors` team is a policy with no members.** The Write-access path
in `GOVERNANCE.md` describes a mechanism nobody has used yet, and the honest
version of the current state is one person with an open invitation. Variant 6
of the invitation file exists because the first grant is the one that will not
happen by itself.

**Nothing here is enforced by a machine.** The reserved-path list, the merge
conditions, and the maintainer-adds-maintainer rule are prose. Until the
workflow in I5 exists, `GOVERNANCE.md` describes what a person is supposed to
do rather than what the repository will make them do. That is stated in the
checklist rather than papered over.

**Required status check names are guessed.** Section I2 lists `test
(ubuntu-latest)`, `test (ubuntu-24.04-arm)`, `python (ubuntu-latest)`, `python
(ubuntu-24.04-arm)`, and `parity`, derived by reading `ci.yml` rather than by
observing a run. GitHub reports matrix job names in a form that has changed
before, and a required check whose name matches nothing is never satisfied and
blocks every pull request silently. Step I4 tells the operator to copy the
names from a completed run instead of trusting the list.

**Two files this lane may not edit need a one-line change each.**
`launch/README.txt` calls itself the index of this directory and lists six
posts; it does not mention `ORG_MIGRATION_CHECKLIST.txt` or
`CONTRIBUTOR_INVITE.txt`, so a person following the index will not find them.
`.github/ISSUE_TEMPLATE/config.yml` sets `blank_issues_enabled: true` and
carries two contact links; once Discussions exists it should route questions
there, since `SUPPORT.md` sends people to Discussions while the issue picker
still offers a blank issue as the path of least resistance. Adding
`docs/HARDWARE_CONTRIBUTORS.md` as a third contact link belongs in the same
edit. Both files are owned by other lanes and neither was touched.

**`CONTRIBUTING.md` and these documents do not reference each other.** It
covers how to make a change and which tests to run, and predates all four of
`GOVERNANCE.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, and the pull request
template. Nothing in it contradicts them, which was checked; it simply does not
know they exist, and its pull request section would be the natural place to say
that no approval is required. Not this lane's file.

**One maintainer is the largest open risk**, and it is a governance problem
rather than an administrative one. A single Admin means no recovery from a lost
account, no second reader for reserved-path changes, and no route for a conduct
report about the maintainer. `GOVERNANCE.md` and `CODE_OF_CONDUCT.md` both name
this explicitly rather than describing a project that does not exist yet.

## What was and was not run

Nothing was executed. This lane read files, searched the tree for the old owner
string, and wrote eight files. No test, no build, no benchmark, no Mojo, no
pixi, no Python, no GitHub API call, no network request, and no commit.

Three static checks were run over this lane's own output and all three pass.
Every relative markdown link in the six markdown files resolves to a file that
exists. Every reserved path in the `GOVERNANCE.md` table exists, with the one
deliberate exception of `.github/actions/**`, which is reserved ahead of the
first composite action rather than in response to one. No em dash or en dash
appears in any of the eight files.

Consequently nothing here is verified by execution. The claims that carry risk
are the GitHub mechanics in checklist sections E, I, and J, which were written
from documented behavior and not confirmed against the interface. Each step
that could fail silently says how to check it. Where the interface disagrees
with this checklist, the interface is right.
