# packaging/security

The checks and fixtures behind a mojotrees release. What an artifact has to
prove about itself before it is published, and what a stranger can check about
it afterwards.

[`docs/RELEASE_SECURITY.md`](../../docs/RELEASE_SECURITY.md) is the reasoning.
[`SECURITY.md`](../../SECURITY.md) is the reporting policy. This directory is
the executable part, and it sits next to
[`packaging/matrix`](../matrix/README.md), which answers the adjacent question
of whether an artifact is the right shape for the platform it claims.

```
check_action_pins.py     every action in a workflow is pinned to a commit SHA
hash_manifest.py         write and verify SHA256SUMS, standard library only
sbom_supplement.py       add the shipped native runtime to a CycloneDX SBOM
verify_release.sh        the consumer side: digest, then provenance
release_checklist.md     the release gate, items F through I
```

## What runs and what does not

**Nothing here has ever been executed.** No mojotrees release exists, so there
is no artifact to hash, no attestation to verify, and no SBOM to supplement.
These were written before the first publication on purpose, so that the
acceptance criteria for a release are settled and reviewable now rather than
invented afterwards to fit whatever the build happened to produce.

`check_action_pins.py` is the one that can run today on a bare checkout, and it
is expected to fail when pointed at the whole workflow directory: `ci.yml` and
`gpu-validation.yml` pin actions by tag, and
`.github/workflows/release-provenance.yml` carries `@REPLACE_WITH_SHA`
placeholders. That failure is the finding, not a bug in the checker.

Every script is standard library Python or POSIX shell. No `pip install`, no
network, no third party module, for the same reason `tools/check_parity.py` and
`packaging/matrix/validate_matrix.py` are: a check that needs an environment
built before it can run is a check that stops being run.

## The two questions, which are not the same question

Most of this directory exists to keep these apart, because conflating them is
the usual way a project ends up believing it has a supply chain story.

**Is this file intact?** A SHA-256 answers that. It proves two files are the
same file and says nothing about who made either one. A `SHA256SUMS` published
next to the artifact it describes is signed by the same nobody as the artifact.
`hash_manifest.py` is this question, and it is the weaker one.

**Where did this file come from?** A GitHub artifact attestation answers that.
It binds the digest to a workflow run through a certificate issued to that
run's OIDC identity and recorded in a public transparency log. There is no key
in this repository to steal, because there is no key.
`verify_release.sh` asks both questions in that order, and prints which is
which, because a green digest check reads like more assurance than it is.

## The SBOM, and why it needs help

A dependency scanner reads a mojotrees wheel as a Python distribution, finds
one component, and stops. It does not report the compiled Mojo extension or the
four MAX runtime libraries copied into the wheel by
`packaging/build_wheel.sh`, which are most of the artifact by bytes and all of
it by risk.

`sbom_supplement.py` opens the wheel, hashes every native member, and adds a
component for each, plus the Mojo and MAX versions read from the build's
provenance sidecar, since those cannot be recovered from the artifact
afterwards. It writes `unknown` where the sidecar recorded nothing rather than a
plausible guess, and it adds no timestamp, so two runs over the same inputs
produce the same bytes.

The limits of the result are stated in `docs/RELEASE_SECURITY.md` section 8.3.
Briefly: the added components identify what is present, and no vulnerability
feed is keyed to them.

## Things this directory deliberately does not contain

- **No secret, no token, and no placeholder for one.** Publication uses PyPI
  trusted publishing, which stores nothing. A `PYPI_API_TOKEN` shaped hole in a
  file is not neutral, it is an invitation, and section 4.1 of the release
  security document makes leaving one a policy violation rather than a style
  preference.
- **No scanner wrapper.** Running a vulnerability scanner over this project
  would produce a report about the Python packaging metadata, which is nearly
  empty, and would miss the conda toolchain, which is the actual supply chain.
  A tool that reports on the wrong thing is worse than no tool.
- **No signing key management.** There is no key. That is the design, not an
  omission.
