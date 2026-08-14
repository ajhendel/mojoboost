# Security policy

mojoboost is an experimental public alpha maintained by one person. This
document says how to report a vulnerability privately, what is in scope, which
versions get a fix, and what you can expect back and when.

The engineering side of the same subject, meaning how a release is built,
signed, attested, and revoked, is in
[docs/RELEASE_SECURITY.md](docs/RELEASE_SECURITY.md).

## Status

No mojoboost release has been published. There is no git tag, no PyPI
distribution, and no artifact that anyone can install by name. One macOS wheel
has been built locally and never distributed
([docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md)).

That matters for two reasons. Nothing published can be compromised yet, which
is the cheapest moment to fix the release path rather than the last. And the
supported-versions table below is currently empty, not as an oversight but
because there is nothing in it.

## Reporting a vulnerability

**Do not open a public issue, discussion, or pull request for a suspected
vulnerability.**

Use GitHub private vulnerability reporting, which is the preferred channel:

1. Go to <https://github.com/ajhendel/mojoboost/security/advisories/new>
2. Or from the repository, Security, Advisories, Report a vulnerability

That form creates a private draft advisory visible only to you and the
maintainer. It supports private discussion, a private fork for the fix, a CVE
request, and coordinated publication, which is why it is preferred over email.

If that page returns a 404, private reporting has not been enabled on the
repository yet. In that case, and only in that case, email
**ajhendel@gmail.com**, the address in `pixi.toml`, with `mojoboost security`
in the subject line, and say in the first line that the report is private. No
PGP key is published, so treat email as a plaintext channel and send only what
is needed to establish the issue, holding a full exploit until a private
advisory exists.

There is no bug bounty and no payment. Credit in the advisory and in the
release notes is offered to every reporter who wants it.

### What to include

A report that has these is triaged faster and misjudged less often.

- The affected surface. Python API, Mojo API, C ABI, command line tool, model
  file format, the wheel and its bundled runtime, or the release pipeline.
- The version, commit, and platform, plus `mojo --version` and the MAX version
  if the report involves a build. For a wheel, the exact filename and its
  SHA-256.
- The trust boundary you believe is crossed. Which input is attacker
  controlled, and who the attacker is relative to whoever runs mojoboost.
- The smallest reproduction you have, as a command and a file. A dataset that
  is smaller and synthetic is more useful than a real one that is large.
- The impact you claim, and how confident you are of it. A suspected
  out-of-bounds write reported as suspected is welcome; one reported as
  confirmed when it has not been confirmed costs everyone a day.

Please do not run tests against infrastructure you do not own. Nothing in this
project needs a hosted service to reproduce.

## Supported versions

| Version | Supported |
|---|---|
| Any | **No.** No release exists. |

From the first tagged release onward, the rule is the one a single maintainer
can actually keep:

- **Only the latest release gets a security fix.** There are no backports and
  no long term support branch.
- **The fix ships in a normal version bump**, patch if nothing else moved,
  minor if the fix has to break a documented surface. A security fix is never a
  reason to break a surface silently. It gets a break note like any other
  break, under section 3.4 of
  [docs/COMPATIBILITY_POLICY.md](docs/COMPATIBILITY_POLICY.md).
- **`main` is not a supported version.** It is where fixes land first. Running
  it is fine and reporting against it is welcome, but no advisory is issued for
  an unreleased commit.
- **Versions before 1.0 are alpha.** Under section 1.3 of the compatibility
  policy, a break may land in a minor bump, which is the same policy a security
  fix inherits.

When this table stops saying "no release exists", it lists exactly one row.

## Response expectations

These are targets from one maintainer working on this part time, published so
that silence is measurable rather than mysterious. They are not a contractual
service level.

| Stage | Target |
|---|---|
| Acknowledge the report | 3 business days |
| Initial assessment, in scope or not, with reasoning | 10 business days |
| Fix or a written plan with a date | 30 days for high impact, best effort otherwise |
| Public disclosure | 90 days from the report, or on the fix release, whichever is first |

If you have not heard back in 10 business days, send one follow up on the same
channel. If a report is silent for 30 days, you are free to disclose publicly
and the delay is the maintainer's failure, not a violation of anything.

The 90 day clock can be extended by agreement, and by nothing else. A request
for an indefinite embargo will be declined.

## Disclosure

Coordinated disclosure, with these expectations on both sides.

- The maintainer will not ask you to stay quiet past 90 days, will name the
  affected versions and the fixed version in the advisory, and will publish the
  advisory even when the fix is trivial and even when the bug is embarrassing.
- You are asked to hold public detail until the earlier of a fix release or the
  90 day mark, and to not use the finding against a third party's deployment.
- A vulnerability already public, already exploited, or already fixed upstream
  is disclosed immediately rather than held.
- Advisories are published as GitHub Security Advisories on this repository. A
  CVE is requested through GitHub when the issue affects a published artifact.

## Scope

### In scope

- Memory safety in Mojo, in the C ABI, or in the CPython extension, reachable
  from data a caller could plausibly receive from elsewhere. Training and
  prediction data, a model file, a serialized dump, a LightGBM model text file.
- Parsing of a model file or a LightGBM model file that a hostile author
  controls. See the trust boundary note below before assuming this is a strong
  guarantee.
- Anything in a distributed artifact that reaches outside the artifact. An
  absolute rpath into a build machine, a bundled library loaded from a
  writable path, an executable member of a wheel that does not belong there.
  `packaging/matrix/validate_artifact.py` exists to catch this class, and a
  case it misses is a good report.
- The release and CI pipeline. A path by which an untrusted pull request, a
  compromised dependency, or a workflow misconfiguration could publish an
  artifact, obtain a token, or alter a published one.
- Any credential, private path, or host detail that leaks into a committed
  file, a published artifact, or a workflow log.

### Out of scope

These are real bugs when they are bugs, and they belong in a normal public
issue rather than a private advisory.

- A crash, hang, or exhaustion caused by parameters or data the caller chose
  for their own process. Passing `num_leaves = 2**40` is a usage bug.
- Any behavior difference from LightGBM. That is a parity matter, governed by
  [docs/LIGHTGBM_PARITY.md](docs/LIGHTGBM_PARITY.md).
- Performance, numerical accuracy, and determinism, which are contracts in
  [docs/COMPATIBILITY_POLICY.md](docs/COMPATIBILITY_POLICY.md) section 8 rather
  than security properties.
- Unpickling a model from an untrusted source. Python pickle executes arbitrary
  code by design and no library can make that safe. Do not do it, with
  mojoboost or with anything else.
- Missing hardening that has no demonstrated impact, results from an automated
  scanner with no analysis attached, or amounts to a version number being lower
  than the newest one.

### The trust boundary, stated plainly

**Model files and dumps are treated as trusted input today.** The parsers have
not been fuzzed, no adversarial corpus exists, and no test in this repository
feeds a deliberately malformed model to a loader. Treat a `.mbst` file the way
you would treat a shared object, meaning load one only from a source you would
already trust to run code on your machine.

That is a statement about evidence, not an exemption. A crash or a memory
safety failure from a malformed model file is in scope and will be fixed. What
is not offered is a claim that the loader is robust, because nothing in the
repository would justify it.

## What this project does not do

Recorded because absence is easy to misread as an unstated feature.

- No network access at runtime. Training, prediction, and serialization read
  and write local files and nothing else.
- No telemetry, no analytics, no crash reporting, no phone home, in any binary
  or in any script.
- No credential handling, no cryptography of its own, and no authentication
  surface.
- No sandbox. A custom objective or a callback is ordinary Python or Mojo
  running in your process with your privileges.

## Dependencies

Dependency policy, the update mechanism, and the parts of this project's supply
chain that automated tooling cannot see are in
[docs/RELEASE_SECURITY.md](docs/RELEASE_SECURITY.md) section 6, with the
configuration in [.github/dependabot.yml](.github/dependabot.yml).

A vulnerability in Mojo, MAX, CPython, or any other dependency is reported to
that project, not here. Report it here as well when mojoboost's use of it makes
the impact materially worse, or when mojoboost ships the affected component
inside an artifact, which is true of the four MAX runtime libraries bundled in
the macOS wheel.
