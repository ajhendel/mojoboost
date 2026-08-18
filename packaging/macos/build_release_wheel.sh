#!/usr/bin/env bash
# Release build for the macOS arm64 wheel.
#
#   packaging/macos/build_release_wheel.sh
#
# NOT EXECUTED. Nothing in packaging/macos has been run: no wheel was built by
# it, no artifact inspected, no hash computed. Read it before trusting it.
#
# This is not a second wheel builder. `packaging/build_wheel.sh` is the builder
# and stays the builder; this script reaches it through `pixi run -e pkg
# test-wheel`, which builds and then installs the result into two clean venvs
# and runs the suites there. That task, not a bare `build-wheel`, is the entry
# point on purpose: a release that skips `packaging/test_wheel.sh` skips the one
# check that answers "does this wheel work", and the checkers below only answer
# "does it say true things about itself".
#
# This script is what a release does around that:
#
#   1. refuse to build unless the checkout is a clean, tagged commit whose tag
#      matches the version in python/pyproject.toml
#   2. refuse to build on a host that cannot produce the artifact the matrix
#      declares (wrong architecture, Rosetta, no pixi)
#   3. build, with the deployment target applied to both halves of the tag
#      contract at once
#   4. record provenance that the wheel cannot carry itself
#   5. verify the result against packaging/matrix and against the release-only
#      rules in inspect_wheel.py
#   6. hash everything that ships
#
# Environment:
#   MOJOTREES_MACOS_TARGET   macOS deployment target for the extension, e.g.
#                            "12.0". Empty means the SDK default, which is
#                            today's behavior and today's macosx_26_0 tag.
#   MOJOTREES_ALLOW_UNTAGGED set to 1 to build from an untagged commit. The
#                            wheel is then a test artifact and must never be
#                            published; provenance records git_tag as "none".
#   MOJOTREES_RELEASE_PYTHON not used here. It belongs to the clean-install
#                            fixture, which runs outside pixi.
set -euo pipefail
cd "$(dirname "$0")/../.."

MACOS_DIR=packaging/macos
DIST=python/dist

say() { printf '\n== %s\n' "$*"; }
die() { printf 'refusing: %s\n' "$*" >&2; exit 2; }

# Every stdlib helper runs under the pinned interpreter rather than whatever
# `python3` resolves to on the host. packaging/matrix/validate_artifact.py needs
# tomllib, so a macOS system python3 is not sufficient, and the pixi environment
# is the one interpreter this repository actually pins.
# Which pixi environment builds the wheel. Defaults to `pkg`, so a laptop build
# and every existing caller behave exactly as before. The release matrix sets
# it to `pkg-py310` .. `pkg-py314` to produce one wheel per interpreter; see
# the [environments] block in pixi.toml.
: "${MOJOTREES_PKG_ENV:=pkg}"
py() { pixi run -e "$MOJOTREES_PKG_ENV" python3 "$@"; }

# --- 1. The host ------------------------------------------------------------

say "host"
[ "$(uname -s)" = "Darwin" ] || die "this builder is macOS only, found $(uname -s)."
[ "$(uname -m)" = "arm64" ] || die "expected arm64, found $(uname -m). The
macos-x86_64 target of packaging/matrix/platform_matrix.toml is unsupported."
# A native arm64 shell running under Rosetta reports arm64 for uname but builds
# an x86_64 world. The wheel would be tagged arm64 and would not be.
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
    die "this shell is translated by Rosetta. Use a native arm64 shell."
fi
[ -z "${CONDA_PREFIX:-}" ] || die "CONDA_PREFIX is set ($CONDA_PREFIX). Run this
from a plain shell; it calls pixi itself and the nested environment changes
which interpreter and which libraries the build sees."
command -v pixi >/dev/null 2>&1 || die "pixi is not on PATH."
command -v codesign >/dev/null 2>&1 || die "codesign is not on PATH; the Xcode
command line tools are required."
command -v install_name_tool >/dev/null 2>&1 || die "install_name_tool is not on
PATH; the Xcode command line tools are required."

sw_vers
uname -m
xcodebuild -version 2>/dev/null || echo "xcodebuild: not available"

# --- 2. The commit ----------------------------------------------------------

say "commit"
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' python/pyproject.toml | head -1)
[ -n "$VERSION" ] || die "no version found in python/pyproject.toml."

DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
    printf '%s\n' "$DIRTY" >&2
    die "the working tree is dirty. A release artifact must correspond to a
commit that exists, and provenance records the commit, not the tree."
fi

if TAG=$(git describe --exact-match --tags HEAD 2>/dev/null); then
    [ "$TAG" = "v$VERSION" ] || die "HEAD is tagged $TAG but
python/pyproject.toml says version $VERSION. Expected tag v$VERSION. Fix one of
the two; do not publish an artifact whose tag and metadata disagree."
elif [ "${MOJOTREES_ALLOW_UNTAGGED:-0}" = "1" ]; then
    TAG=none
    echo "warning: building from an untagged commit. This artifact is a test"
    echo "build. Do not publish it to any index."
else
    die "HEAD is not a tag. Check out v$VERSION, or set
MOJOTREES_ALLOW_UNTAGGED=1 for a test build that must not be published."
fi
echo "tag:     $TAG"
echo "commit:  $(git rev-parse HEAD)"
echo "version: $VERSION"

# --- 3. The build -----------------------------------------------------------
#
# The deployment target has two halves and they are set together on purpose.
#
#   MACOSX_DEPLOYMENT_TARGET          what the Mojo compile step is asked to
#                                     emit as LC_BUILD_VERSION minos
#   MOJOTREES_MACOS_DEPLOYMENT_TARGET what python/setup.py writes into the
#                                     wheel's platform tag
#
# python/setup.py deliberately does not read MACOSX_DEPLOYMENT_TARGET, because a
# conda-style environment exports that variable for its own compilers at values
# unrelated to what Mojo emitted. Setting only the first produces a lowered
# binary with a macosx_26_0 tag; setting only the second produces a published
# lie. Setting both is a request, not a result: whether the Mojo compiler honors
# the variable at all is an open question (handoffs/task18_platform.md, edit 3),
# and inspect_wheel.py check C1 is what turns the request into a verified fact by
# comparing the tag against the binary.

say "build"

# Zip entry timestamps otherwise come from the wall clock, so two builds of one
# commit differ in every member's header and therefore in the wheel's digest.
# `wheel` reads SOURCE_DATE_EPOCH and stamps entries from it instead, which
# removes that source of difference. Taken from the commit, so it is a property
# of what is being built rather than of when the build ran.
#
# This is one source of nondeterminism, not all of them. Whether the Mojo
# compiler emits a byte-identical extension across two builds of the same commit
# has not been tested, and until it has, do not describe the macOS wheel as
# reproducible or treat two matching digests as proof of anything. The Linux
# builder (packaging/linux/build_wheel_linux.sh) sets this for the same reason.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git log -1 --pretty=%ct HEAD)}
echo "SOURCE_DATE_EPOCH: $SOURCE_DATE_EPOCH"

# THE DEPLOYMENT TARGET NOW DEFAULTS TO 12.0 RATHER THAN TO THE SDK, AND THAT
# IS A USABILITY FIX RATHER THAN A PREFERENCE.
#
# Without an explicit target the compiler stamps the SDK of whatever machine
# ran the build. This project builds on an M4 running macOS 26, so the wheel
# came out tagged `macosx_26_0_arm64`, which pip reads as "needs macOS 26 or
# newer". That is not a requirement of Mojo or of this library. It is a
# fingerprint of the build machine, and it made the wheel installable only on
# the newest macOS for no reason anybody chose.
#
# **AND THE FLOOR IS THE HALF THAT MATTERS, BECAUSE `requires-python` DOES NOT
# HELP.** `python/pyproject.toml` already declares `>=3.10`, correctly, and its
# own comment says why that buys nothing on its own: pip matches the WHEEL TAG
# before it consults `requires-python`. A wheel tagged
# `cp314-macosx_26_0_arm64` is invisible to a user on 3.12 or on macOS 14 no
# matter how permissive the metadata is. The tag is the gate.
#
# WHY 12.0 AND NOT SOMETHING OLDER. Monterey is the oldest release that runs on
# every Apple Silicon Mac ever shipped, so it costs nothing in reach, and
# `docs/PLATFORM_MATRIX.md` already carries a `macos-arm64-cp314-lowered` row
# targeting exactly `macosx_12_0_arm64`.
#
# VERIFIED BY BUILDING IT ON 2026-08-18, not assumed. The open question was
# whether anything in the build genuinely needs a macOS 26 API, and the only
# way to answer it is to lower the target and compile. Result: the extension
# builds clean, `vtool -show-build` reports `minos 12.0` against `sdk 26.5`,
# and both backends train on it, CPU and GPU agreeing at rmse 0.323278. So
# nothing here needed macOS 26 and the tag was pure accident.
#
# Set MOJOTREES_MACOS_TARGET to override, including to the SDK default by
# passing the current major, and C1 below still checks what the compiler
# actually emitted rather than trusting the request.
: "${MOJOTREES_MACOS_TARGET:=12.0}"
echo "requested deployment target: $MOJOTREES_MACOS_TARGET"
echo "(a request. C1 below checks what the compiler actually emitted.)"
export MACOSX_DEPLOYMENT_TARGET="$MOJOTREES_MACOS_TARGET"
export MOJOTREES_MACOS_DEPLOYMENT_TARGET="$MOJOTREES_MACOS_TARGET"

# Builds through packaging/build_wheel.sh (test-wheel depends on build-wheel),
# then installs the wheel into a bare venv and a full one and runs the suites
# against the install rather than against the source tree. It runs inside pixi,
# where the Mojo runtime is on the library path whether the wheel bundles it or
# not, which is why it cannot be the last word on self-containment. The
# clean-install fixture is, and it runs outside pixi in the release workflow.
pixi run -e "$MOJOTREES_PKG_ENV" test-wheel

shopt -s nullglob
WHEELS=("$DIST"/mojotrees-*.whl)
shopt -u nullglob
[ "${#WHEELS[@]}" -eq 1 ] || die "expected exactly one wheel in $DIST, found
${#WHEELS[@]}: ${WHEELS[*]:-none}."
WHEEL=${WHEELS[0]}
echo "built: $WHEEL"

# --- 4. Provenance ----------------------------------------------------------
#
# Written before the checks, so a failing verification still leaves behind a
# record of what was built and where. provenance.sh exits non-zero when the
# build's accelerator answer and the host's Metal toolchain disagree, which is a
# release stopper rather than a verification detail: two wheels with this
# filename and different answers to has_accelerator() are different products.

say "provenance"
"$MACOS_DIR/provenance.sh" "$WHEEL"

# --- 5. Verification --------------------------------------------------------
#
# Three checkers, in widening scope, each printing its own verdict:
#   validate_matrix.py     does the matrix still agree with the repository
#   validate_artifact.py   does the wheel match the target the matrix declares
#   inspect_wheel.py       the release-only rules, which the matrix does not
#                          cover (tag exactness, bundle minimality, install
#                          names, source-tree leakage)
#
# What is deliberately not here: packaging/matrix/smoke/clean_install_macos.sh.
# It refuses to run with CONDA_PREFIX set, and every command in this script goes
# through pixi. The workflow runs it as a separate step outside the environment,
# which is the only way its result means anything.

# Each checker runs even when an earlier one failed, and the exit status is
# collected rather than propagated immediately. A release decision wants the
# whole list, and a failing run is exactly when the evidence below (the
# inspection report, the otool dump, the hashes) is worth having: aborting on
# the first failure would leave the artifact directory with less in it than a
# passing run produces.
rc=0

say "matrix contract"
py packaging/matrix/validate_matrix.py || rc=1

say "artifact against the matrix"
py packaging/matrix/validate_artifact.py "$WHEEL" || rc=1

say "release inspection"
py "$MACOS_DIR/inspect_wheel.py" "$WHEEL" --json "$DIST/inspection.json" || rc=1

# The load commands as the system tool reports them, kept as evidence next to
# the artifact. inspect_wheel.py parses the same bytes with the standard
# library; a disagreement between these two files is worth more than either.
say "otool record"
{
    echo "=== _mojotrees.so ==="
    otool -l python/mojotrees/_mojotrees.so | grep -A 4 -E 'LC_BUILD_VERSION|LC_RPATH|LC_ID_DYLIB' || true
    otool -L python/mojotrees/_mojotrees.so || true
    for lib in python/mojotrees/.dylibs/*.dylib; do
        echo "=== $(basename "$lib") ==="
        otool -l "$lib" | grep -A 4 -E 'LC_BUILD_VERSION|LC_RPATH|LC_ID_DYLIB' || true
        otool -L "$lib" || true
    done
} >"$DIST/otool.txt" 2>&1
echo "wrote $DIST/otool.txt"

# --- 6. Hashes --------------------------------------------------------------

say "hashes"
"$MACOS_DIR/hash_artifacts.sh" "$DIST"

say "done"
echo "$WHEEL"

if [ "$rc" -ne 0 ]; then
    echo
    echo "One or more checkers failed. Read the whole log above rather than the"
    echo "last failure: each checker prints its own verdict and they cover"
    echo "different questions. The artifact and every report about it are in"
    echo "$DIST, which is where they need to be to work out which of the wheel"
    echo "and the matrix is the one that is wrong."
    exit "$rc"
fi

echo
echo "Built and verified. Not validated: no platform in"
echo "packaging/matrix/platform_matrix.toml may be moved off 'designed' by this"
echo "script's output alone. That takes a clean-install run on a machine with no"
echo "toolchain, pasted into a record. The fixture is"
echo "packaging/matrix/smoke/clean_install_macos.sh and it must run outside pixi."
