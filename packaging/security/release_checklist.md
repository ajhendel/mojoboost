# Release security checklist

The supply chain half of the release gate. Every item is a hard gate, in the
same sense as section 12 of
[docs/COMPATIBILITY_POLICY.md](../../docs/COMPATIBILITY_POLICY.md), and the
letters continue that list rather than restarting it, so **F3** here and **A3**
there are never confused. That document's A through E cover tests, versions,
surface, and honesty. This one covers what the artifact is made of and how it
gets published.

The reasoning behind each item is
[docs/RELEASE_SECURITY.md](../../docs/RELEASE_SECURITY.md), cited by section.
This file is the runbook. If an item here and a section there disagree, the
section is the authority and this line is the bug.

**Nothing on this list has been run, because no release exists.**

## F. Supply chain

- [ ] **F1.** `python3 packaging/security/check_action_pins.py .github/workflows/release-provenance.yml`
      passes. Every action in the release path is a commit SHA with a version
      comment. Section 5.1.
- [ ] **F2.** Every open Dependabot pull request is merged or explicitly
      declined with a reason. None is merged automatically. Section 7.2.
- [ ] **F3.** The toolchain was refreshed deliberately for this release.
      `pixi update`, the `pixi.lock` diff read for `mojo`, `max`, `python`, and
      anything that ends up inside the wheel, and the suite re-run. Nothing
      will alert you to this, ever. Section 6.3.
- [ ] **F4.** No credential shaped string is in anything that executes. Run the
      grep in section 12 of `docs/RELEASE_SECURITY.md` as written, with the
      `--include` filters, and read every match rather than counting them. A
      detection pattern inside a secret scanner is expected and fine; an
      assignment, a workflow input, or a value is the finding. Section 4.1.
- [ ] **F5.** `gh secret list --repo mojotrees/mojotrees` and the same for the
      `pypi` environment are both empty. A publishing secret existing at all is
      a finding, not a convenience. Section 4.1.
- [ ] **F6.** No workflow gained a trigger, a permission, or a step since the
      last release without a human reading that diff. `git log -p` over
      `.github/**` and `packaging/**` since the last tag. Section 7.

## G. The build

- [ ] **G1.** The build ran on a machine with the Metal toolchain. If that was
      a personal machine rather than a registered self hosted runner, the
      provenance is a personal assurance and the release does not go to PyPI.
      Section 3.3.
- [ ] **G2.** `pixi run -e pkg test-wheel` green, from a clean checkout.
- [ ] **G3.** `python3 packaging/matrix/validate_artifact.py python/dist/*.whl`
      green, so no build host path, no absolute rpath, no stray member, and no
      broken signature ships.
- [ ] **G4.** `<wheel>.provenance.json` exists and no required field reads
      empty. `has_accelerator_at_build` may read `unknown`; it may not be
      blank. Section 8.4.
- [ ] **G5.** The clean install fixture ran outside pixi, with no `mojo` on
      `PATH`, and its output is saved.
- [ ] **G6.** `SHA256SUMS` written, and verified in a second command rather
      than trusted from the first. Section 8.1.

## H. Publish

- [ ] **H1.** The PyPI trusted publisher exists and names this repository, the
      workflow filename `release-provenance.yml`, and the environment. No API
      token is configured on the project. Section 4.2.
- [ ] **H2.** The `pypi` environment has required reviewers and a deployment
      branch rule limiting it to `main` and tags. Section 3.2.
- [ ] **H3.** The run went to TestPyPI first, and installing from there into a
      clean venv worked on a machine that is not the build machine.
- [ ] **H4.** The build provenance attestation and the SBOM attestation both
      succeeded, and both name the same digest as `SHA256SUMS`. Section 8.2.
- [ ] **H5.** The SBOM lists the compiled extension and every bundled runtime
      library, each with a SHA-256. A component count that matches a bare Syft
      run means the supplement did not run. Section 8.3.
- [ ] **H6.** The publish job uploaded exactly one file and it was the wheel.
      Section 4.3.
- [ ] **H7.** The release notes name the artifact, its exact tag, its SHA-256,
      and the platforms that get no artifact. That is **D2** of the
      compatibility policy, restated here because it is the line a user
      actually reads.

## I. After

- [ ] **I1.** `bash packaging/security/verify_release.sh <wheel> SHA256SUMS`
      passes against the published file, downloaded fresh from PyPI rather than
      copied from the build directory. Verifying the file you just made proves
      nothing about the file users get.
- [ ] **I2.** `gh attestation verify` with `--signer-workflow` passes on that
      same downloaded file. Section 8.2.
- [ ] **I3.** The digest in the release notes matches the digest of the
      published file.
- [ ] **I4.** Section 11 of `docs/RELEASE_SECURITY.md` is accurate for what
      this release actually did. An item that got closed is removed; an item
      that stayed open stays listed.

## If something is wrong after publication

Do not improvise. Section 9 of `docs/RELEASE_SECURITY.md` is the order of
operations, and the first four steps are evidence, stop publication, yank, and
say something, in that order. Yanking is cheap and reversible. Deleting is
neither, and it burns the version number permanently.
