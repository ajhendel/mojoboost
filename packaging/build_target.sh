#!/bin/sh
# The CPU target every mojotrees build emits code for. Sourced, not executed.
#
#     . "$(dirname "$0")/../packaging/build_target.sh"
#     mojotrees_resolve_target
#     mojo build $MOJOTREES_TARGET_FLAGS ...
#
# WHY THIS FILE EXISTS
# --------------------
# `mojo build` defaults `--target-cpu` and `--target-features` to the HOST. Not
# to a portable baseline, and not to nothing: to whatever chip ran the compiler.
# Measured on this project's development machine with `pixi run print-target`:
#
#     --target-cpu apple-m4
#     --target-features +aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,
#       +fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,
#       +rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2
#
# `bindings/build.sh`, `capi/build.sh` and `cli/build.sh` emit real machine code
# and, before this file, passed no target flags. So the shipped artifact was
# compiled for the build machine's CPU.
#
# On this project that is not hypothetical. `.github/workflows/release-macos.yml`
# and `.github/workflows/release-provenance.yml` both build on a SELF-HOSTED M4.
# `+bf16` and `+i8mm` do not exist on M1; `+sme2` does not exist on M1, M2 or M3.
# Compare (`pixi run mojo build --print-effective-target --target-cpu <cpu>`):
#
#     apple-m1   no bf16, no i8mm, no sme, no sme2
#     apple-m2   +bf16 +i8mm, no sme2
#     apple-m4   +bf16 +i8mm +sme +sme2
#
# LLVM does not need to be asked to use an enabled feature. It emits `bfdot`,
# `smmla` and `bfmmla` from ordinary loops when the feature bit is set, and the
# result is SIGILL on a machine that lacks it, at the instruction, inside the
# extension, with no diagnostic a user can act on.
#
# Nothing downstream catches it. `packaging/macos/inspect_wheel.py` reads Mach-O
# `cputype` and `minos`; arm64 `cpusubtype` stays `ARM64_ALL` whatever `-mcpu`
# was, so a native-built wheel LOOKS portable. The check that can see it is
# `packaging/isa_baseline.py`, added alongside this file, and it works by
# disassembling.
#
# THE COST, STATED RATHER THAN DISCOVERED
# ---------------------------------------
# A baseline target may produce a shipped artifact that is slower than a native
# build would have been on a new machine. That is the correct trade: a wheel
# that runs everywhere it claims to run beats a wheel that is faster on the one
# machine that built it. The wheel has no way to negotiate a microarchitecture
# with pip, so the floor is the ceiling.
#
# How large the cost is, is NOT known and is not claimed here, because this was
# established by reading two binaries and not by running either of them. No
# timing was taken and none should be inferred from what follows.
#
# The extension was built twice on 2026-08-16 on the same M4, once at the host
# default and once with `--target-cpu apple-m1`, and both were disassembled with
# `otool -tv`:
#
#   same size                6,620,448 bytes both times
#   same mnemonic set        205 distinct mnemonics, symmetric difference empty
#   instruction totals       719,057 native, 718,585 at apple-m1
#   of that difference       483 is `nop` (497 native, 14 at apple-m1), i.e.
#                            function alignment padding
#   non-padding difference   about 11 instructions out of 718,000
#
# And the native build, the one the M4 host default produced, contains NO
# instruction above ARMv8.0-A NEON plus LSE atomics. No `bfdot`, no `smmla`, no
# `bfmmla`, no SME, no SVE, not even `sdot`/`udot`. The auto-vectorizer was not
# using the features the host default enabled, which is why lowering the floor
# to apple-m1 removes an option that today's code does not exercise.
#
# What that does NOT establish: instruction scheduling, register allocation and
# addressing can differ without moving a histogram, so "the same instructions in
# the same proportions" is not "the same speed". The honest statement is that
# the difference in what is emitted is small and that nobody has measured the
# difference in what it costs.
#
# None of this stays true by itself. It is a property of the current source, and
# it is exactly why the disassembly check is committed alongside this file
# rather than left as a one-off observation: the day someone writes a loop the
# vectorizer can turn into `bfdot`, the native build starts differing and the
# check is what notices.
#
# THE SPLIT: BASELINE BY DEFAULT, NATIVE ON REQUEST
# -------------------------------------------------
#   MOJOTREES_BUILD_TARGET=baseline   (default) portable floor, per the table
#   MOJOTREES_BUILD_TARGET=native     the host, i.e. the old behavior
#
# Baseline is the default and native is the opt-in, rather than the other way
# round, for three reasons.
#
#   1. The failure directions are not comparable. Forgetting `native` costs a
#      developer some speed on a build they can rerun. Forgetting `baseline`
#      costs a user a crash in a published wheel that has to be yanked.
#   2. A release must not depend on remembering an environment variable. Every
#      release path reaches these three scripts (`packaging/build_wheel.sh`,
#      `packaging/linux/build_wheel_linux.sh`, and through them
#      `packaging/macos/build_release_wheel.sh`), so making the safe value the
#      default makes every one of them safe without a new step in any of them.
#   3. No benchmark in this repository goes through these scripts. Every
#      `bench-*` task in pixi.toml is a `mojo run -I src bench/...`, which is
#      unaffected by this file and stays native. "Local benchmarking keeps the
#      native target" is therefore true by construction, not by this default.
#
# Every build echoes the target it resolved, so a number taken from a build is
# never taken from an unlabelled one.
#
# THE BASELINES, AND WHY EACH
# ---------------------------
#   Darwin arm64      --target-cpu apple-m1
#       The oldest Apple silicon that exists. There is no M0, and this project
#       ships no macOS x86_64 wheel (packaging/matrix/platform_matrix.toml marks
#       macos-x86_64 unsupported, and packaging/macos/build_release_wheel.sh
#       refuses to run on it), so apple-m1 is exactly "the oldest hardware the
#       project intends to support". packaging/matrix/accelerators/index.toml
#       carries apple-m1 as a target, which is the documented intent.
#       Keeps: dotprod, fullfp16, fp16fml, LSE, crypto, rcpc, complxnum, jsconv.
#       Drops: bf16, i8mm, sme, sme2. Those four are the whole defect.
#
#   Linux x86_64      --target-cpu x86-64-v2
#       SSE3 through SSE4.2, POPCNT, CMPXCHG16B. Every x86-64 part since Nehalem
#       (2009) and Bulldozer (2011). v3 (AVX2/FMA/BMI2) would exclude pre-2013
#       Intel and pre-2015 AMD and is still common in service; there is no wheel
#       tag that lets pip choose between a v2 and a v3 build, so shipping v3
#       means shipping only v3. If the CPU path is ever shown to want AVX2 badly
#       enough to matter, the answer is a runtime dispatch or a second wheel with
#       a mechanism to select it, not a raised floor here.
#       GitHub's ubuntu-22.04 x86 fleet mixes Intel Xeon (AVX-512 capable) with
#       AMD EPYC (no AVX-512), so without this flag two runs of
#       release-linux.yml on the same commit can produce two different products.
#
#   Linux aarch64     --target-cpu generic --target-features +lse
#       ARMv8.0-A plus large-system atomics, i.e. an ARMv8.1-A floor. Covers
#       Graviton2 and later, Ampere Altra, Azure Cobalt 100, Cortex-A55/A75 and
#       later, and Apple silicon under Linux. Excludes Cortex-A72 (Raspberry
#       Pi 4) and Graviton1.
#       `+lse` is not decoration and it is the one place this file deliberately
#       does not take the lowest floor available. The natively built extension
#       contains 4,716 LSE atomic instructions (`ldaddal`, `ldaddl`) from the
#       runtime's reference counting. Dropping to plain ARMv8.0-A replaces every
#       one of them with an `ldxr`/`stxr` retry loop on a contended refcount
#       path. That is a cost this lane has no instrument to measure and no
#       business incurring blind, and ARMv8.1-A hardware is nine years old.
#       `generic` rather than `neoverse-n1` because the two differ only in
#       dotprod, fullfp16 and rcpc, none of which appears in the object today,
#       so `generic +lse` is strictly more portable at no observed cost.
#
# WHAT THIS FILE DELIBERATELY DOES NOT SET
# ----------------------------------------
#   --fp-mode          Default is `contract=fast`, which fuses `a + b*c` into an
#                      FMA across statements. Turning it off moves bits in every
#                      objective, split gain and leaf value. That is a golden
#                      re-baseline, sequenced by the orchestrator, never a
#                      build-file edit. See docs/NUMERICS.md section 8. Note
#                      that the DEFAULT is `contract=fast` and no build script
#                      says so anywhere; this comment is the first place it is
#                      written down at the point of use.
#   --optimization-level
#                      Already defaults to 3. There is no missing -O3.
#   --target-triple    Left at the host. The macOS deployment target is a
#                      separate contract with its own two halves, set by
#                      packaging/macos/build_release_wheel.sh through
#                      MACOSX_DEPLOYMENT_TARGET and
#                      MOJOTREES_MACOS_DEPLOYMENT_TARGET. Pinning a triple here
#                      would fight it.
#   --target-accelerator
#                      GPU target selection is a different question with a
#                      different owner (see .github/workflows/release-macos.yml's
#                      header on has_accelerator() and the self-hosted runner).
#                      This file is the CPU half only.
#
# Setting MOJOTREES_TARGET_FLAGS directly overrides everything above. It is the
# escape hatch for a target this file has not been taught, and it is echoed.

# Sets MOJOTREES_TARGET_FLAGS. Safe to call more than once.
#
# The flags are spliced unquoted into the `mojo build` command line by the
# callers. That is correct here and only here: every value this function can
# produce is a fixed literal with no whitespace, no glob character and no
# variable in it, so word splitting is the intended behavior and there is
# nothing for it to split wrongly. POSIX sh has no arrays; this is the
# alternative.
mojotrees_resolve_target() {
    _mt_mode=${MOJOTREES_BUILD_TARGET:-baseline}

    if [ -n "${MOJOTREES_TARGET_FLAGS+x}" ] && [ "$_mt_mode" != native ]; then
        echo "target: MOJOTREES_TARGET_FLAGS set explicitly: $MOJOTREES_TARGET_FLAGS"
        return 0
    fi

    case "$_mt_mode" in
        native)
            MOJOTREES_TARGET_FLAGS=""
            echo "target: native (host CPU). NOT PORTABLE. Do not publish an"
            echo "        artifact from this build. Run 'pixi run print-target'"
            echo "        to record what this machine actually resolved to."
            ;;
        baseline)
            case "$(uname -s)/$(uname -m)" in
                Darwin/arm64)
                    MOJOTREES_TARGET_FLAGS="--target-cpu apple-m1" ;;
                Linux/x86_64)
                    MOJOTREES_TARGET_FLAGS="--target-cpu x86-64-v2" ;;
                Linux/aarch64|Linux/arm64)
                    MOJOTREES_TARGET_FLAGS="--target-cpu generic --target-features +lse" ;;
                *)
                    echo "packaging/build_target.sh: no baseline is defined for" >&2
                    echo "$(uname -s)/$(uname -m). This is refused rather than" >&2
                    echo "defaulted to the host, because defaulting to the host is" >&2
                    echo "the defect this file exists to remove: it would produce a" >&2
                    echo "publishable-looking artifact compiled for one machine." >&2
                    echo "Add the platform to the table above, or set" >&2
                    echo "MOJOTREES_BUILD_TARGET=native for a local build that must" >&2
                    echo "not be published." >&2
                    return 2 ;;
            esac
            echo "target: baseline ($MOJOTREES_TARGET_FLAGS)"
            ;;
        *)
            echo "packaging/build_target.sh: MOJOTREES_BUILD_TARGET=$_mt_mode is" >&2
            echo "not one of: baseline, native." >&2
            return 2 ;;
    esac
    export MOJOTREES_TARGET_FLAGS
}
