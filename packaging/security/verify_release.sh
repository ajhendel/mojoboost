#!/usr/bin/env bash
# Verify a downloaded mojotrees wheel before installing it.
#
#   bash packaging/security/verify_release.sh <wheel> [SHA256SUMS]
#
# This is the consumer side of docs/RELEASE_SECURITY.md. It answers two
# different questions in the order that makes sense, and it is worth knowing
# which is which, because the first one is much weaker than it looks.
#
#   1. Integrity. Is this file the file the manifest describes? A digest proves
#      two files are the same file. It proves nothing about who made either
#      one, and a manifest downloaded from the same place as the wheel is
#      signed by the same absence of a signature. On its own this catches a
#      truncated download and an accidental substitution, and no attacker.
#
#   2. Provenance. Did this exact file come out of a run of
#      .github/workflows/release-provenance.yml in github.com/mojotrees/mojotrees?
#      That is what `gh attestation verify` checks, against a certificate issued
#      to that workflow's OIDC identity and logged in a public transparency log.
#      This is the question worth asking, and it is the one an attacker with
#      write access to a download page cannot forge.
#
# NOTHING HERE HAS EVER RUN. No mojotrees release exists, so there is no wheel
# to verify and no attestation to verify it against. This script is written now
# so that the acceptance criteria are settled before the first artifact, rather
# than invented afterwards to fit whatever was published.
#
# Requirements: python3, and the GitHub CLI (`gh`, version 2.49 or newer) for
# step 2. The attestation check needs network access to reach the transparency
# log, unless you pass `--bundle` with a bundle you downloaded earlier.
set -euo pipefail

REPO=mojotrees/mojotrees
WORKFLOW=.github/workflows/release-provenance.yml

WHEEL=${1:-}
SUMS=${2:-}

if [ -z "$WHEEL" ]; then
    echo "usage: verify_release.sh <wheel> [SHA256SUMS]" >&2
    exit 2
fi
if [ ! -f "$WHEEL" ]; then
    echo "not a file: $WHEEL" >&2
    exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)

echo "== 1. digest =="
python3 "$HERE/hash_manifest.py" digest "$WHEEL"

if [ -n "$SUMS" ]; then
    echo
    echo "== 1b. against the manifest =="
    python3 "$HERE/hash_manifest.py" verify "$SUMS" "$(dirname "$WHEEL")"
else
    echo "no SHA256SUMS given; digest printed but not compared"
fi

echo
echo "== 2. provenance =="
if ! command -v gh >/dev/null 2>&1; then
    cat >&2 <<EOF
gh is not installed, so the provenance of this file has NOT been checked.
The digest above says only that the file is intact. Install the GitHub CLI and
run:

    gh attestation verify "$WHEEL" \\
        --repo $REPO \\
        --signer-workflow $REPO/$WORKFLOW

Until then, treat this wheel as an unverified download.
EOF
    exit 1
fi

# --signer-workflow is the part that matters. Without it, the check passes for
# an artifact attested by any workflow in the repository, which includes a
# workflow an attacker with write access could add. With it, the certificate
# must name this release workflow.
gh attestation verify "$WHEEL" \
    --repo "$REPO" \
    --signer-workflow "$REPO/$WORKFLOW"

echo
echo "== 3. what is inside it =="
echo "The SBOM attestation lists the bundled MAX runtime libraries and the"
echo "toolchain that built the extension. To read it:"
echo
echo "    gh attestation verify \"$WHEEL\" --repo $REPO --predicate-type https://cyclonedx.org/bom --format json"
echo
echo "verified: $WHEEL"
