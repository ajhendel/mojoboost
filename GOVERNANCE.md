# mojoboost governance

This document says who decides what, how contributors get commit access, and
how disagreements end. It is short on purpose. mojoboost is an experimental
public alpha maintained by volunteers, and process that outweighs the project
is worse than no process at all.

## Mission

Build a gradient-boosted decision tree library, written natively in Mojo, that
is honest about what it can do.

Three commitments follow from that.

1. **A familiar surface.** LightGBM-shaped API and semantics, so that a person
   who knows LightGBM can read mojoboost code without a translation guide.
   Deliberate differences are documented in
   [docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md), which is the behavior
   contract.
2. **Accelerators that are actually present.** The GPU in an Apple Silicon Mac
   is the first-class target, because it is the accelerator most contributors
   already own. NVIDIA and AMD matter and are not yet validated.
3. **Evidence before claims.** No performance, parity, portability, or
   correctness claim ships without a reproducible command and a recorded
   environment. An explicit unsupported error beats a silent fallback. This is
   the rule most likely to get a change sent back, and it applies to
   maintainers exactly as it applies to first-time contributors.

## What this project is not

mojoboost is not a company, a product, or a funded program. `mojoboost-ml` is a
GitHub organization that exists so the repository has a neutral home, so more
than one person can hold administrative access, and so the project outlives any
single account. There is no legal entity behind it, nobody is employed by it,
and nobody is on call. The software is provided as-is under Apache-2.0 with no
warranty, exactly as the license says.

Nothing here creates an obligation on any contributor, including the
maintainers. Response times below are intentions, not commitments.

## License and copyright

mojoboost is licensed under [Apache License 2.0](LICENSE), and that governs
both directions.

- **Inbound equals outbound.** Every contribution is offered under Apache-2.0
  under the terms in section 5 of the license, which covers the patent grant.
  Opening a pull request is the act of offering it. There is no separate
  contributor license agreement to sign and no developer certificate of origin
  bot to satisfy.
- **Contributors keep their copyright.** No copyright assignment is requested
  or accepted. There is no per-file author list to maintain; git history is the
  record.
- **Third-party code needs its provenance stated.** If a contribution copies or
  adapts code from another project, say so in the pull request, name the source
  and its license, and keep any required attribution in the file. Code under a
  license incompatible with Apache-2.0 cannot be merged, and code with unclear
  provenance is treated as incompatible.
- **Relicensing requires unanimity.** Since copyright stays with contributors,
  a license change would require every contributor to agree. Assume the license
  is permanent and plan accordingly.

## Who decides

During the alpha, mojoboost uses a benevolent maintainer model. Andrew Hendel
([@ajhendel](https://github.com/ajhendel)) created mojoboost, serves as lead
maintainer, and holds the final call on technical direction, scope, and
anything unresolved.

This is a description of the current reality, not an ambition. Andrew started
the project, and pretending a committee exists would slow contributors down
without protecting them. The model is deliberately temporary, and the section
on evolving this document says what replaces it.

Three limits keep the model honest.

- **Decisions happen in public.** Technical decisions are made in issues, pull
  requests, and Discussions on the repository, not in private messages. A
  decision reached in a private conversation gets written back into the public
  thread before it takes effect.
- **Reasons are given.** A rejected approach comes with the reason it was
  rejected and, where one exists, the alternative that would be accepted.
- **The evidence rule binds the maintainer.** A maintainer claim about
  performance or parity needs the same reproducible command as anyone else's.
  Any contributor may ask for it, and asking is never rude.

Most changes never reach a decision. Merging a bug fix, adding a test, fixing
documentation, and improving an error message are ordinary contributor work,
not maintainer prerogatives.

## Pull requests during the alpha

**No contribution needs a human approval to merge.** Waiting on a maintainer to
click a button is the single most common way an alpha loses a contributor, so
mojoboost does not do it. Two paths exist, and both end in merged code without
a review queue.

### If you have Write access

Push directly. Small, self-contained work such as a bug fix, a test, a
docstring, a benchmark script, or an example does not need a pull request at
all. Use a pull request when you want a second pair of eyes, when the change is
large enough that others should see it coming, or when it touches a reserved
path.

Three rules come with direct push.

- Run the focused tests described in [CONTRIBUTING.md](CONTRIBUTING.md) before
  pushing. Not the full suite, the relevant file.
- Never force-push or rewrite history on `main`.
- Do not self-merge a change to a reserved path. Reserved paths are listed
  below and are the one place a second person is required.

### If you do not have Write access yet

Open a pull request from a fork. It merges automatically once a machine check
passes, with no human in the loop. The check is deliberately minimal, and it
asks only questions a machine can answer.

- Continuous integration is green on every required job, which currently means
  the Mojo test matrix, the Python API and estimator suites, and the parity
  contract check.
- The pull request touches no reserved path.
- The branch is not behind `main` in a way that invalidates the run.

If all three hold, it merges. If continuous integration fails, read the log and
push a fix; nobody needs to re-approve anything. If the change touches a
reserved path, the automatic merge stops and a maintainer looks at it, which is
the only queue in the process.

A merged change is not a permanent one. Reverting is cheap, and the project
would rather merge quickly and revert occasionally than hold contributions
hostage to attention that may not arrive for days. A revert is a statement
about a change, never about the person who wrote it.

### Reserved paths

Automatic merge is disabled for changes that can compromise the project or its
users regardless of whether the tests pass. Somebody who cannot yet be trusted
with the repository also cannot be trusted to rewrite what runs inside it.

| Path | Why it is reserved |
| --- | --- |
| `.github/workflows/**`, `.github/actions/**`, `.github/dependabot.yml` | Workflow code executes with repository context and can be written to exfiltrate tokens or poison later runs. |
| `pixi.toml`, `pixi.lock`, `python/pyproject.toml` | Dependency and lock changes are the supply chain. A single altered pin is enough. |
| `packaging/**`, wheel and release tooling | Determines what is published under the project name. |
| `tools/**` scripts invoked by continuous integration | Runs on project infrastructure. |
| `LICENSE`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | Changes the terms everyone else agreed to. |

A reserved-path change from an outside contributor is welcome and gets read on
its merits. It merges when a maintainer has actually read the diff, not when
the tests go green. Expect this to be slower, and say in the pull request why
the change is needed so the reading is quick.

Contributors with Write access are trusted to change these files, and the only
added requirement is that somebody else merges it.

## Getting Write access

Write access is the normal outcome of contributing, not a promotion. The bar is
evidence that you will not break the project carelessly, which usually means:

- two or three merged contributions of some substance, or one substantial one;
- changes that came with tests and did not require someone else to finish them;
- respect for the evidence rule, meaning claims arrived with commands attached.

That is the whole bar. It is not gated on a time period, an amount of code, or
knowing the maintainer.

**Ask.** Open a Discussion or comment on your most recent merged pull request
and say you would like Write access. Being asked is not an imposition, and
self-nomination carries no penalty. A maintainer will also offer access
unprompted when the pattern above appears, but nobody should have to wait to be
noticed.

The maintainer adds you to the `contributors` team, which carries the Write
role on the repository. Access is expected to be granted, and a refusal comes
with the specific reason and what would change it.

Write access does not mean ownership of an area, an obligation to keep
contributing, or a duty to review other people's work. Contributors who go
quiet are not doing anything wrong.

## Maintainers

Maintainers hold the Admin role and can change repository settings, manage
teams, publish releases, and act on security reports. Today there is one. Two
or three is the target before the alpha ends, because a single administrator is
a real risk to the project rather than a governance style.

**Adding a maintainer.** The maintainer proposes an active contributor, in
public, on the repository. The proposal names what the person has done and why
the added administrative access is warranted. Existing maintainers and the
proposed person have seven days to respond. Absent an unresolved objection, the
person joins the `maintainers` team. Once more than one maintainer exists,
adding another requires that all current maintainers agree.

What is being assessed is judgment under uncertainty and care with other
people's trust, not volume of code. Somebody who reviews carefully, reproduces
other people's failures, and says "I do not know" is a better maintainer than
somebody who writes more lines.

**Stepping down.** Say so, in public or privately, and move to emeritus. No
notice period, no justification, no ill will. Emeritus maintainers are listed
as past maintainers and keep no elevated access, which is a security measure
and not a comment on them. Returning is a matter of asking.

**Inactivity.** A maintainer who has not been reachable for six months may be
moved to emeritus by the remaining maintainers, after a public note on the
repository and a direct attempt to reach them. If every maintainer is inactive,
active contributors may propose a new maintainer in a public Discussion; after
thirty days with no response from any maintainer, GitHub organization
administrators may act on that proposal. This exists so an abandoned project
can be adopted rather than quietly die.

**Removal for cause.** A maintainer may be removed for a serious Code of
Conduct violation, for acting against the project's users, or for a security
failure that was deliberate or reckless. Removal requires agreement of all
other maintainers, or, if only one other exists, of that maintainer together
with a public explanation. Access is revoked first and explained second when
credentials are at risk.

## When people disagree

Disagreement about technical direction is normal and is usually the most useful
thing happening on the repository. The path is short.

1. **Argue it in the thread.** State the tradeoff, not the conclusion. Most
   disagreements end here once each side sees what the other is optimizing for.
2. **Ask for evidence.** A remarkable share of GBDT disputes dissolve into a
   benchmark or a differential test against LightGBM. If a claim can be
   settled by running something, run it and post the command, the output, and
   the environment.
3. **Ask the maintainer to decide.** Say so explicitly in the thread. A
   decision arrives with its reasoning. This is the end of the line during the
   alpha, and the decision holds.
4. **Disagree and commit, or fork.** Apache-2.0 guarantees the fork, and taking
   it is a legitimate move rather than a hostile one. A contributor who is
   overruled is not required to pretend to agree, and is asked not to relitigate
   the same question in unrelated threads.

Two things are outside this path. Conduct is handled through
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), not through technical debate.
Security is handled privately, below.

Reverting a merged change is part of ordinary disagreement, not an escalation.
Anyone with Write access may revert a change that breaks `main`, and the
expected courtesy is a note in the thread saying what broke and how to
reproduce it.

## Security

**Do not open a public issue for a vulnerability.**
[SECURITY.md](SECURITY.md) is the authoritative policy and holds the reporting
channel, the scope boundaries, the supported-version rule, and the response and
disclosure timelines. It governs where it and this document differ.

Only the governance-level facts belong here.

- **Maintainers act on security reports.** Private advisories are visible to
  the `maintainers` team, which is one more reason the project wants a second
  maintainer.
- **A security fix reaches `main` through the reserved-path route**, with a
  maintainer reading the diff, even when it is urgent and even when the tests
  pass. The automatic merge path is never used for security work, and this
  holds regardless of who or what authored the change, a dependency bot
  included.
- **A security failure that was deliberate or reckless is grounds for removing
  a maintainer**, under the removal-for-cause rule above.

## Changing this document

Governance changes through a pull request against `GOVERNANCE.md`, announced in
Discussions, left open at least seven days for comment. During the alpha the
maintainer decides after that window, applying the same obligation to give
reasons that every other decision carries.

Two changes are expected rather than hypothetical.

- **A maintainer vote replaces the benevolent maintainer model** once three
  maintainers exist. At that point technical decisions that cannot be resolved
  in a thread go to a simple majority, ties are resolved by not making the
  change, and this section is rewritten to say so.
- **The lightweight merge policy tightens when the alpha ends.** Automatic
  merging on a machine check is right for a project where reverting is cheap
  and nothing depends on the current release. It stops being right the moment
  people run mojoboost on something that matters. The trigger to revisit it is
  a stable release, not a contributor count.

Governance that outgrows the project is a failure mode, so a change that
removes process is as welcome as one that adds it.

## Related documents

| Document | What it covers |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to make a change, which tests to run, what a pull request should say |
| [SUPPORT.md](SUPPORT.md) | Where to ask a question and what to include |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Expected behavior and how to report a problem |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability, scope, and disclosure |
| [docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md) | The public behavior contract |
| [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) | The procedure behind any hardware claim |
