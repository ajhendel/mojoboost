#!/usr/bin/env bash
# Write the provenance sidecar for a built wheel.
#
#   packaging/macos/provenance.sh python/dist/<wheel>
#
# NOT EXECUTED. No sidecar has been written by this script.
#
# Produces <wheel>.provenance.json, which is the filename
# packaging/matrix/validate_artifact.py rule R7 looks for. R7 requires eight
# keys and treats an empty value as a failure while accepting a recorded
# "unknown", so an unavailable fact is written down as unknown rather than
# omitted.
#
# Why a sidecar at all. None of this is recoverable from the wheel afterwards,
# and one field changes what the artifact does on the user's machine:
# has_accelerator() is resolved at compile time (src/mojoboost/device.mojo), so
# a wheel built where an accelerator was visible reports one as available, and
# a `device="gpu"` request on that build fails when the device is opened rather
# than when it is resolved. Two wheels with the same filename and different
# answers to that question are different products.
#
# The sidecar is written even when the consistency gate at the end fails, so a
# refused build still leaves a record of what it was.
set -euo pipefail
cd "$(dirname "$0")/../.."

WHEEL=${1:?usage: provenance.sh python/dist/<wheel>}
[ -f "$WHEEL" ] || { echo "no such wheel: $WHEEL" >&2; exit 2; }
OUT="$WHEEL.provenance.json"

# JSON string values, defensively. Command output can contain quotes,
# backslashes, and newlines, and a sidecar that does not parse is worse than a
# missing one because R7 reports it as unreadable rather than absent.
esc() {
    printf '%s' "$1" | tr '\n' ' ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' \
        -e 's/[[:cntrl:]]//g' -e 's/  */ /g' -e 's/ $//'
}
field() { printf '  "%s": "%s",\n' "$1" "$(esc "$2")"; }

MOJO_VERSION=$(pixi run mojo --version 2>/dev/null | head -1 || true)
MAX_VERSION=$(pixi list --environment default 2>/dev/null \
    | awk '$1=="max"{print $2; exit}' || true)
PIXI_LOCK_SHA=$(shasum -a 256 pixi.lock | cut -d' ' -f1)
COMMIT=$(git rev-parse HEAD)
TAG=$(git describe --exact-match --tags HEAD 2>/dev/null || echo none)
HOST_OS=$(sw_vers -productVersion 2>/dev/null || echo unknown)
HOST_BUILD=$(sw_vers -buildVersion 2>/dev/null || echo unknown)
HOST_ARCH=$(uname -m)
XCODE=$(xcodebuild -version 2>/dev/null | tr '\n' ' ' || echo unknown)
SDK=$(xcrun --show-sdk-version 2>/dev/null || echo unknown)
WHEEL_SHA=$(shasum -a 256 "$WHEEL" | cut -d' ' -f1)

# The Metal toolchain. Its absence is what makes a GitHub-hosted Apple silicon
# runner the wrong machine to build a release on: it reports an accelerator at
# compile time and cannot build the GPU-touching sources (see the comment at the
# top of .github/workflows/ci.yml).
if METAL=$(xcrun --find metal 2>/dev/null); then
    METAL_VERSION=$(xcrun metal --version 2>&1 | head -1 || true)
else
    METAL=absent
    METAL_VERSION=absent
fi

# The compile-time accelerator answer, from a program that prints exactly that
# and nothing else. Anything other than "true" or "false" is recorded as
# unknown rather than guessed.
ACCEL=$(pixi run mojo run -I src packaging/macos/report_accelerator.mojo 2>/dev/null \
    | tr -d '[:space:]' || true)
case "$ACCEL" in
    true|false) ;;
    *) ACCEL=unknown ;;
esac

# Deployment target as requested, which is not the same as what was emitted.
# What was emitted is in the binary, and inspect_wheel.py check C1 reads it
# there. Recording the request separately is how a compiler that ignores the
# variable becomes visible instead of confusing.
REQUESTED_TARGET=${MOJOBOOST_MACOS_TARGET:-sdk-default}

# What the wheel's zip entries were stamped from. Recorded because it is the
# one input a later rebuild has to match before its digest can be compared with
# this one, and because "unset" is the answer that explains a mismatch.
SDE=${SOURCE_DATE_EPOCH:-unset}

{
    printf '{\n'
    printf '  "schema": 1,\n'
    field builder "packaging/macos/build_release_wheel.sh via packaging/build_wheel.sh"
    field wheel "$(basename "$WHEEL")"
    field wheel_sha256 "$WHEEL_SHA"
    field mojo_version "${MOJO_VERSION:-unknown}"
    field max_version "${MAX_VERSION:-unknown}"
    field pixi_lock_sha256 "$PIXI_LOCK_SHA"
    field git_commit "$COMMIT"
    field git_tag "$TAG"
    printf '  "git_dirty": %s,\n' "$(test -n "$(git status --porcelain)" && echo true || echo false)"
    field build_host_os "$HOST_OS ($HOST_BUILD)"
    field build_host_arch "$HOST_ARCH"
    field xcode "${XCODE:-unknown}"
    field sdk_version "${SDK:-unknown}"
    field metal_toolchain "$METAL"
    field metal_version "${METAL_VERSION:-unknown}"
    field requested_deployment_target "$REQUESTED_TARGET"
    field source_date_epoch "$SDE"
    printf '  "has_accelerator_at_build": "%s"\n' "$ACCEL"
    printf '}\n'
} >"$OUT"

echo "wrote $OUT"
cat "$OUT"

# --- The consistency gate ---------------------------------------------------
#
# An accelerator visible at compile time on a host with no Metal toolchain is
# the specific combination that produces a wheel nobody can reason about: the
# build claims a GPU is available, the GPU-touching sources could not have been
# built and tested there, and the failure surfaces on the user's machine when
# the device is opened. Refuse it here rather than publish it.
#
# MOJOBOOST_ALLOW_INCONSISTENT_GPU_BUILD=1 overrides, for the case where the
# artifact is being produced deliberately to study that failure. The override is
# not recorded in the sidecar on purpose: an artifact built with it is not a
# release artifact, and the sidecar should not be made to look like it is.
if [ "$ACCEL" = "true" ] && [ "$METAL" = "absent" ]; then
    if [ "${MOJOBOOST_ALLOW_INCONSISTENT_GPU_BUILD:-0}" = "1" ]; then
        echo "warning: accelerator visible at compile time and no Metal toolchain."
        echo "warning: override set. This artifact must not be published."
    else
        echo >&2
        echo "refusing: this build reports has_accelerator() = true and the host" >&2
        echo "has no Metal toolchain (xcrun --find metal failed)." >&2
        echo >&2
        echo "That is the GitHub-hosted Apple silicon runner's configuration and" >&2
        echo "it produces a wheel whose GPU story cannot be tested on the machine" >&2
        echo "that built it. Build on a Mac with Xcode and the Metal toolchain" >&2
        echo "installed, or set MOJOBOOST_ALLOW_INCONSISTENT_GPU_BUILD=1 for an" >&2
        echo "artifact that is explicitly not for publication." >&2
        exit 3
    fi
fi

if [ "$ACCEL" = "unknown" ]; then
    echo
    echo "note: has_accelerator_at_build is unknown. The reporter did not print"
    echo "true or false. R7 accepts unknown as a recorded answer, and the"
    echo "consistency gate above cannot run on it, so a release built this way"
    echo "is a release whose GPU behavior nobody checked."
fi
