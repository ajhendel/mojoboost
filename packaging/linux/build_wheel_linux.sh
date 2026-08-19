#!/usr/bin/env bash
# Build a self-contained Linux wheel for the mojotrees Python API.
#
#     pixi run -e pkg packaging/linux/build_wheel_linux.sh
#
# THIS SCRIPT HAS NEVER BEEN EXECUTED. It is the Linux counterpart of
# packaging/build_wheel.sh, written from a static reading of that script, of
# pixi.lock, and of the ELF loader's rules. Every step below is a hypothesis
# until the first run, and the first run should happen on a throwaway machine
# with the output kept.
#
# What it does, in the order it does it:
#
#   1. refuse to run anywhere the result would be misleading
#   2. refuse to run while the Python metadata cannot describe a Linux wheel
#   3. build the extension with the Mojo toolchain
#   4. walk the extension's DT_NEEDED closure and stage every object that
#      resolves inside the pixi environment into python/mojotrees/.libs
#   5. rewrite RUNPATH so the loader finds them relative to the wheel
#   6. build the wheel with an explicit, honest platform tag
#   7. write a SHA-256 manifest and a provenance sidecar
#   8. inspect what it just produced and refuse to leave a bad artifact behind
#
# It does NOT install the wheel and does NOT test it. That is
# packaging/matrix/smoke/clean_install_linux.sh, on a different machine, in a
# container with no toolchain. A wheel this script accepts is a candidate, not
# a release.
#
# Environment:
#   MOJOTREES_TAG_POLICY     plain (default) | manylinux
#   MOJOTREES_MANYLINUX      glibc floor for the manylinux policy, default 2_28
#   MOJOTREES_ALLOW_DIRTY    set to 1 to build from a dirty tree (never for a release)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

TAG_POLICY=${MOJOTREES_TAG_POLICY:-plain}
MANYLINUX_FLOOR=${MOJOTREES_MANYLINUX:-2_28}
PKG=python/mojotrees
LIBDIR=$PKG/.libs
DIST=python/dist

die() { printf 'build_wheel_linux: %s\n' "$*" >&2; exit 1; }
note() { printf '\n== %s\n' "$*"; }

# --- 1. Refuse to run where the answer would not mean anything -------------

[ "$(uname -s)" = "Linux" ] || die "this builder is for Linux, found $(uname -s).
Use packaging/build_wheel.sh on macOS."

HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
    x86_64|aarch64) ;;
    *) die "unsupported architecture $HOST_ARCH.
pixi.toml declares linux-64 and linux-aarch64 only, and cross building is not
attempted here: the extension is compiled by the Mojo toolchain for the host." ;;
esac

[ -n "${CONDA_PREFIX:-}" ] || die "CONDA_PREFIX is not set.
Run this inside the pixi pkg environment:
    pixi run -e pkg packaging/linux/build_wheel_linux.sh"

command -v patchelf >/dev/null 2>&1 || die "patchelf not found.
It is the Linux counterpart of install_name_tool and no pixi environment in
this repository provides it yet. See handoffs/release_03_linux_wheels.md for
the pixi.toml edit that adds it. Do not substitute a system patchelf for a
release build without recording its version in the provenance sidecar."

command -v readelf >/dev/null 2>&1 || die "readelf not found (binutils)."

if [ "${MOJOTREES_ALLOW_DIRTY:-0}" != "1" ] && ! git diff --quiet HEAD 2>/dev/null; then
    die "working tree is dirty.
A release artifact has to be attributable to a commit. Commit, or set
MOJOTREES_ALLOW_DIRTY=1 for a throwaway build you will not publish."
fi

# An accelerator visible at compile time changes the product, not just the
# build: has_accelerator() in src/mojotrees/device.mojo resolves at compile
# time, so a wheel built here would tell every user that a GPU is available and
# then fail when the device is opened.
ACCEL=no
if command -v nvidia-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
    ACCEL=unknown
    [ "${MOJOTREES_ALLOW_ACCEL_HOST:-0}" = "1" ] || die \
"an accelerator tool is on PATH (nvidia-smi or rocm-smi).
Build release wheels on a machine with no visible accelerator, because
has_accelerator() resolves at compile time and this host would bake a GPU claim
into every install of the resulting file. If this host genuinely has no device
and only the vendor tooling is installed, re-run with
MOJOTREES_ALLOW_ACCEL_HOST=1 and expect has_accelerator_at_build to be recorded
as 'unknown', which is what it is."
fi

# --- 2. Refuse to run while the metadata cannot describe a Linux wheel ------

note "metadata preflight"
python3 packaging/linux/check_metadata_ready.py \
    || die "the Python metadata cannot produce a correct Linux wheel yet.
Fix the blockers listed above (they are Task 01's files, and the exact edits
are in handoffs/release_03_linux_wheels.md) and re-run."

# --- Platform tag ----------------------------------------------------------

case "$TAG_POLICY" in
    plain)
        PLAT="linux_${HOST_ARCH}"
        ;;
    manylinux)
        PLAT="manylinux_${MANYLINUX_FLOOR}_${HOST_ARCH}"
        ;;
    *)
        die "MOJOTREES_TAG_POLICY must be 'plain' or 'manylinux', got '$TAG_POLICY'"
        ;;
esac

if [ "$TAG_POLICY" = "manylinux" ]; then
    cat >&2 <<EOF

  !! Building with a manylinux tag.
  !!
  !! That tag is a promise that this file works on any glibc ${MANYLINUX_FLOOR//_/.}
  !! system, and this script cannot verify it: it can only report what the
  !! objects reference. The promise is only true after a clean install in a
  !! container of that glibc has passed. See the four conditions in
  !! packaging/linux/README.md.
  !!
  !! Nothing forces you to be right here. Being wrong produces installs that
  !! fail at import on machines you will never see.

EOF
fi

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' python/pyproject.toml | head -1)
[ -n "$VERSION" ] || die "could not read version from python/pyproject.toml"

# --- 3. Build the extension ------------------------------------------------

note "building the extension"
rm -rf "$LIBDIR" python/build "$DIST" python/mojotrees.egg-info
bindings/build.sh

EXT=$PKG/_mojotrees.so
[ -f "$EXT" ] || die "bindings/build.sh did not produce $EXT"

# --- 3b. The instruction set, checked here rather than on the wheel ---------
#
# `mojo build` defaults --target-cpu to the HOST. bindings/build.sh now pins a
# baseline through packaging/build_target.sh, and this is the check that the
# pin held. It runs here, on the freshly built object, for one reason: the
# check needs a disassembler, and this build host has binutils while the
# machine a wheel is later inspected on may not. packaging/linux/inspect_wheel.py
# is deliberately stdlib-only so that a Linux wheel can be examined from the
# macOS laptop this project is developed on, and adding a check that shells out
# to objdump would take that property away. So the ELF side of the ISA check
# lives on the build path, and the macOS side lives in
# packaging/macos/inspect_wheel.py as C14, where otool is always present.
#
# GitHub's ubuntu-22.04 x86 fleet mixes Intel (AVX-512 capable) with AMD EPYC
# (not), and its ARM runners are Neoverse-class with SVE and bf16. Without the
# pin, two runs of .github/workflows/release-linux.yml on one commit can produce
# two different products, and neither the wheel nor its ELF headers record which
# one you got.
note "checking the instruction set baseline of $EXT"
python3 packaging/isa_baseline.py "$EXT" --verbose \
    || die "the extension contains instructions outside the target baseline.
Read the report above: it names the feature and quotes the instruction. The fix
is packaging/build_target.sh, not this script. Do not publish this wheel."

# --- 4. Stage the runtime closure ------------------------------------------
#
# The macOS builder copies a hard-coded list of four dylibs. This one does not,
# because the Linux runtime set has never been inspected and a hard-coded list
# would be a guess wearing the costume of a fact. Walk DT_NEEDED to a fixed
# point instead, and stage everything that resolves inside the pixi prefix.
#
# What is deliberately not staged: the C library and its siblings, the dynamic
# loader, and libpython. Those come from the user's system by design. A wheel
# that bundles libc does not work, it merely fails later and more strangely.

needed_of() {
    readelf -d "$1" 2>/dev/null \
        | sed -n 's/.*(NEEDED).*Shared library: \[\(.*\)\]$/\1/p'
}

is_system_lib() {
    # One line per group, and no line continuations inside the pattern list:
    # a backslash-newline there splices the next line's indentation into the
    # pattern, and every pattern after the first silently stops matching.
    case "$1" in
        libc.so*|libm.so*|libdl.so*|librt.so*|libpthread.so*) return 0 ;;
        libresolv.so*|libutil.so*|libnsl.so*|libcrypt.so*) return 0 ;;
        ld-linux*|ld64.so*|linux-vdso.so*) return 0 ;;
        libpython*) return 0 ;;
    esac
    return 1
}

note "walking the DT_NEEDED closure of $EXT"
mkdir -p "$LIBDIR"
queue=$(needed_of "$EXT")
seen=""
external=""
while [ -n "$queue" ]; do
    next=""
    for soname in $queue; do
        case " $seen " in *" $soname "*) continue ;; esac
        seen="$seen $soname"

        if is_system_lib "$soname"; then
            echo "  system   $soname   (left to the target machine)"
            external="$external $soname"
            continue
        fi

        src=""
        for candidate in "$CONDA_PREFIX/lib/$soname" "$CONDA_PREFIX/lib64/$soname"; do
            [ -e "$candidate" ] && { src=$candidate; break; }
        done

        if [ -z "$src" ]; then
            # Not in the pixi prefix and not a library the target is expected to
            # have. Either the toolchain grew a dependency nobody knows about,
            # or the build host is leaking one of its own libraries in.
            die "DT_NEEDED '$soname' resolves to neither the pixi prefix nor the
system allowlist. Find out what it is before shipping anything that needs it:
    readelf -d $EXT
    ldd $EXT"
        fi

        # Follow the symlink chain so the real object is what lands in the
        # wheel, but keep the soname as the filename so the loader finds it.
        real=$(readlink -f "$src")
        cp -L "$real" "$LIBDIR/$soname"
        chmod u+w "$LIBDIR/$soname"
        echo "  bundled  $soname   <- $real"

        case "$soname" in
            libstdc++.so*|libgcc_s.so*)
                echo "  !! $soname is GPL-3 with the GCC Runtime Library Exception."
                echo "  !! Its license text has to ship in the wheel. See the"
                echo "  !! licensing section of packaging/linux/README.md."
                ;;
        esac

        next="$next $(needed_of "$LIBDIR/$soname")"
    done
    queue=$next
done

BUNDLED=$(cd "$LIBDIR" 2>/dev/null && ls 2>/dev/null | tr '\n' ' ' || true)
[ -n "$BUNDLED" ] || die "nothing was bundled.
The extension linked no library from the pixi prefix, which contradicts what
the macOS build does (it needs libKGENCompilerRTShared and
libAsyncRTMojoBindings). Something is wrong with the build, not with the wheel."

# --- 5. Rewrite RUNPATH ----------------------------------------------------
#
# The freshly built extension carries an absolute RPATH into this checkout's
# pixi environment. That path does not exist on a user's machine, and shipping
# it also leaks the build host's directory layout into the artifact.

note "rewriting RUNPATH"
patchelf --remove-rpath "$EXT"
patchelf --set-rpath '$ORIGIN/.libs' "$EXT"
for lib in "$LIBDIR"/*; do
    patchelf --remove-rpath "$lib"
    patchelf --set-rpath '$ORIGIN' "$lib"
done
readelf -d "$EXT" | grep -E 'RUNPATH|RPATH|NEEDED' || true

# --- 6. Build the wheel ----------------------------------------------------
#
# --plat-name on the command line overrides the plat_name that python/setup.py
# sets in setup()'s options dict, which is hard-coded to macOS. That override
# is a workaround, not the fix; check_metadata_ready.py explains the fix and
# this script verifies the filename afterwards either way.
#
# SOURCE_DATE_EPOCH comes from the commit, so two builds of the same commit on
# the same host produce zip entries with the same timestamps.

note "building the wheel, platform tag $PLAT"
cp LICENSE python/LICENSE
cp NOTICE python/NOTICE
SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
export SOURCE_DATE_EPOCH
(
    cd python
    python -m build --wheel --no-isolation \
        --config-setting=--build-option=--plat-name="$PLAT"
)

WHEEL=$(ls "$DIST"/mojotrees-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || die "no wheel in $DIST"
BASE=$(basename "$WHEEL")

case "$BASE" in
    *-"$PLAT".whl) ;;
    *) die "the wheel is tagged '$BASE' and the requested platform was '$PLAT'.
The --plat-name override did not take. Do not rename the file: the tag comes
from the build and renaming it makes the label a lie rather than a mistake.
Fix python/setup.py (see check_metadata_ready.py) and rebuild." ;;
esac
case "$BASE" in
    mojotrees-"$VERSION"-*) ;;
    *) die "wheel version does not match python/pyproject.toml ($VERSION): $BASE" ;;
esac

# --- 7. Manifest and provenance --------------------------------------------

note "manifest and provenance"
(cd "$DIST" && sha256sum "$BASE" > "$BASE.sha256")
cat "$DIST/$BASE.sha256"

GLIBC_BUILD=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo unknown)
OS_NAME=$( [ -f /etc/os-release ] && (. /etc/os-release && echo "$PRETTY_NAME") || echo unknown )
MOJO_VERSION=$(mojo --version 2>/dev/null | head -1 || echo unknown)
MAX_VERSION=$(sed -n 's/.*conda\.modular\.com\/max\/[a-z0-9-]*\/max-\([0-9.]*\)-.*/\1/p' \
    pixi.lock | head -1 || echo unknown)
LOCK_SHA=$(sha256sum pixi.lock | cut -d' ' -f1)

cat > "$DIST/$BASE.provenance.json" <<EOF
{
  "mojo_version": "$MOJO_VERSION",
  "max_version": "$MAX_VERSION",
  "pixi_lock_sha256": "$LOCK_SHA",
  "git_commit": "$(git rev-parse HEAD)",
  "build_host_os": "$OS_NAME",
  "build_host_arch": "$HOST_ARCH",
  "has_accelerator_at_build": "$ACCEL",
  "metal_toolchain": "n/a (linux)",
  "build_host_glibc": "$GLIBC_BUILD",
  "patchelf_version": "$(patchelf --version 2>/dev/null || echo unknown)",
  "tag_policy": "$TAG_POLICY",
  "platform_tag": "$PLAT",
  "bundled": "$BUNDLED",
  "external_left_to_target": "$(echo "$external" | tr -s ' ')",
  "container_image_digest": "${MOJOTREES_BUILD_IMAGE_DIGEST:-none (native runner)}",
  "source_date_epoch": "$SOURCE_DATE_EPOCH",
  "validated": "no. This file records how the wheel was built. It is not evidence that the wheel works anywhere."
}
EOF
cat "$DIST/$BASE.provenance.json"

# --- 8. Inspect what was produced ------------------------------------------

note "inspecting the artifact"
python3 packaging/linux/inspect_wheel.py "$WHEEL" \
    || die "the wheel it just built does not pass inspection.
Do not publish it and do not hand it to anyone. The failures above are about
this file, not about the machine it would be installed on."

cat <<EOF

wheel built: $WHEEL

It has been inspected and NOT tested. Nothing has installed it, imported it, or
trained a model with it. Next, on a machine that is not this one and has no
toolchain:

    packaging/matrix/smoke/clean_install_linux.sh $DIST/$BASE record.txt

The exact container commands for both architectures are in
handoffs/release_03_linux_wheels.md.
EOF
