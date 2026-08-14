#!/usr/bin/env bash
# Write or verify SHA256SUMS for the release artifacts in a directory.
#
#   packaging/macos/hash_artifacts.sh python/dist            # write
#   packaging/macos/hash_artifacts.sh python/dist --verify   # verify
#
# NOT EXECUTED. No manifest has been written and nothing has been verified.
#
# The manifest covers the files that ship or that are evidence for what shipped:
# the wheel, its provenance sidecar, the inspection report, and the otool dump.
# It deliberately does not cover itself.
#
# What this is for, since a wheel already carries per-file hashes in its RECORD:
# RECORD hashes the members against each other, so it detects a tampered member
# inside a wheel it cannot detect the substitution of. A hash over the whole
# artifact, published next to it and repeated in the release notes and in the
# provenance sidecar, is the value a user or a mirror can compare against
# something the build published rather than something the file says about
# itself. TestPyPI and PyPI show their own sha256 for an uploaded file; that
# number and this one must be the same, and the compare is written out in
# handoffs/release_02_macos_wheels.md.
set -euo pipefail

DIR=${1:?usage: hash_artifacts.sh <directory> [--verify]}
MODE=${2:-write}
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 2; }
cd "$DIR"

MANIFEST=SHA256SUMS

if [ "$MODE" = "--verify" ]; then
    [ -f "$MANIFEST" ] || { echo "no $MANIFEST in $DIR" >&2; exit 2; }
    shasum -a 256 -c "$MANIFEST"
    echo "verified against $DIR/$MANIFEST"
    exit 0
fi

shopt -s nullglob
FILES=(*.whl *.provenance.json inspection.json otool.txt)
shopt -u nullglob
[ "${#FILES[@]}" -gt 0 ] || { echo "nothing to hash in $DIR" >&2; exit 2; }

# Sorted, so two builds of the same set of files produce byte-identical
# manifests and a diff between them is about the artifacts rather than about
# directory order.
printf '%s\n' "${FILES[@]}" | sort | xargs shasum -a 256 >"$MANIFEST"

echo "wrote $DIR/$MANIFEST"
cat "$MANIFEST"
