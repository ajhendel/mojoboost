#!/bin/sh
# run_validation.sh - one paste-and-go validation run, any vendor.
#
# Executes the procedure in docs/GPU_VALIDATION.md section "Procedure" on
# whatever device this machine has, and writes a single result bundle in the
# format the "Recording a result" section expects.
#
#   sh hardware/run_validation.sh                 # detect vendor, full run
#   sh hardware/run_validation.sh --vendor amd    # force a vendor
#   sh hardware/run_validation.sh --quick         # skip the full suite
#   sh hardware/run_validation.sh --commands-only # print, run nothing
#
# Properties, matching hardware/capture/*.sh:
#   * No root. Installs nothing except pixi's own environment via `pixi install`.
#   * Never fails the run on a failing step. A failure IS a result; it is
#     recorded and the script continues. Exit status is always 0.
#   * Writes one file, `<vendor>-validation-<date>.txt`, in the current
#     directory. Nothing is uploaded.
#
# The one thing this script is strict about: docs/GPU_VALIDATION.md says a GPU
# suite that prints "skipped: no accelerator" is NOT a validation. This script
# greps for that and marks the run INVALID rather than letting a green pass
# stand in for a device that was never seen.

set -u

VENDOR=""
QUICK=0
COMMANDS_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --vendor) VENDOR="$2"; shift 2 ;;
        --vendor=*) VENDOR="${1#*=}"; shift ;;
        --quick) QUICK=1; shift ;;
        --commands-only) COMMANDS_ONLY=1; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- vendor ----
# Adding a vendor when Mojo grows a backend is three edits and no new
# infrastructure, because the portable source does not change:
#
#   1. a detection branch below, keyed on that stack's smi-equivalent
#   2. a `case` arm in the environment section naming its query commands
#   3. hardware/capture/capture_<vendor>.sh and templates/result_<vendor>.json
#
# Mojo emits for NVIDIA (PTX), AMD (HIP), and Apple (Metal) as of Mojo 1.0.
# There is no Intel, Mali, Adreno, or Tenstorrent target, so those are absent
# here because nothing can be compiled for them, not because they were skipped.
# Free developer clouds exist for each vendor (AMD Developer Cloud, Intel Tiber)
# and are the intended way to reach a new backend's hardware without paying
# retail. See docs/HARDWARE_CONTRIBUTORS.md.
if [ -z "$VENDOR" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        VENDOR=nvidia
    elif command -v rocm-smi >/dev/null 2>&1; then
        VENDOR=amd
    elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        VENDOR=apple
    else
        VENDOR=none
    fi
fi

DATE="$(date +%Y-%m-%d)"
OUT="${VENDOR}-validation-${DATE}.txt"

# run <label> <command...>  -- echo the command, run it, record output.
run() {
    label="$1"; shift
    printf '\n=== %s ===\n$ %s\n' "$label" "$*"
    [ "$COMMANDS_ONLY" -eq 1 ] && return 0
    "$@" 2>&1
    printf '[exit %s]\n' "$?"
}

# runsh <label> <shell string> -- same, for pipelines and env prefixes.
runsh() {
    label="$1"; shift
    printf '\n=== %s ===\n$ %s\n' "$label" "$*"
    [ "$COMMANDS_ONLY" -eq 1 ] && return 0
    sh -c "$*" 2>&1
    printf '[exit %s]\n' "$?"
}

emit() {

printf 'mojotrees GPU validation bundle\n'
printf 'vendor: %s\n' "$VENDOR"
printf 'date:   %s\n' "$DATE"
printf 'script: hardware/run_validation.sh\n'
printf 'repo:   %s\n' "$(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
printf 'dirty:  %s\n' "$(git status --porcelain 2>/dev/null | head -1 | grep -q . && echo YES || echo no)"

printf '\n----------------------------------------------------------------\n'
printf 'A run from a dirty tree is exploration and is not quotable.\n'
printf '%s\n' '----------------------------------------------------------------'

# ------------------------------------------------------- 1. environment ----
run  "uname"        uname -a
if [ "$(uname -s)" = "Linux" ]; then
    runsh "lscpu"   "lscpu | grep -iE 'model name|architecture|^cpu\\(s\\)'"
else
    runsh "sysctl"  "sysctl -n machdep.cpu.brand_string hw.ncpu"
fi
run  "mojo version" pixi run mojo --version
runsh "max version" "pixi list --environment default | grep -Ei '^(mojo|max)'"

case "$VENDOR" in
    nvidia)
        run   "nvidia-smi" nvidia-smi
        runsh "nvidia-smi query" \
              "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv"
        [ -f hardware/capture/capture_nvidia.sh ] && \
            run "capture_nvidia.sh" sh hardware/capture/capture_nvidia.sh
        ;;
    amd)
        run   "rocm-smi" rocm-smi
        runsh "rocminfo"  "rocminfo | grep -iE 'name|gfx|compute unit'"
        runsh "rocm ver"  "cat /opt/rocm/.info/version"
        [ -f hardware/capture/capture_amd.sh ] && \
            run "capture_amd.sh" sh hardware/capture/capture_amd.sh
        ;;
    apple)
        runsh "system_profiler" "system_profiler SPDisplaysDataType | head -20"
        [ -f hardware/capture/capture_apple.sh ] && \
            run "capture_apple.sh" sh hardware/capture/capture_apple.sh
        ;;
    none)
        printf '\n=== no accelerator detected ===\n'
        printf 'Neither nvidia-smi nor rocm-smi is on PATH and this is not\n'
        printf 'Apple silicon. Per PLATFORM_MATRIX.md, "unsupported" is a\n'
        printf 'status the schema accepts: this bundle is still worth filing.\n'
        ;;
esac

# --------------------------------------------- 2. correctness, determinism --
# Pinned first so the CPU comparison is reproducible (GPU_VALIDATION.md s3).
MOJOTREES_NUM_WORKERS=8
MOJOTREES_PARALLEL_MIN_OPS=65536
export MOJOTREES_NUM_WORKERS MOJOTREES_PARALLEL_MIN_OPS
printf '\n=== threading pinned ===\nMOJOTREES_NUM_WORKERS=%s MOJOTREES_PARALLEL_MIN_OPS=%s\n' \
       "$MOJOTREES_NUM_WORKERS" "$MOJOTREES_PARALLEL_MIN_OPS"

if [ "$QUICK" -eq 0 ]; then
    run "full suite"    pixi run test
fi
run "gpu suite"         pixi run test-gpu
runsh "cpu-pinned path" "MOJOTREES_DISABLE_GPU=1 pixi run test-gpu"

# ------------------------------------------------------ 3. shapes, phases --
run "gpu-validate sweep"  pixi run gpu-validate
if [ "$QUICK" -eq 0 ]; then
    run "gpu-validate 250k"  pixi run gpu-validate 20 250000 200
    run "gpu-validate multi" pixi run gpu-validate 20 10000 20 100000 100 50000 400 1000000 20
fi

printf '\n=== end of bundle ===\n'
}

# ------------------------------------------------------------------ main ----
if [ "$COMMANDS_ONLY" -eq 1 ]; then
    emit
    exit 0
fi

echo "mojotrees validation: vendor=$VENDOR -> $OUT"
echo "this runs a full test suite and a benchmark sweep; expect several minutes."
emit | tee "$OUT"

# --------------------------------------------------------------- verdict ----
# GPU_VALIDATION.md: "A pass that says skipped is not a validation."
echo
echo "================ verdict ================"
if [ "$VENDOR" = "none" ]; then
    echo "NO DEVICE. File as 'unsupported'."
elif grep -qi 'skipped: no accelerator' "$OUT"; then
    echo "INVALID: the GPU suite printed 'skipped: no accelerator' on a machine"
    echo "that reports a $VENDOR device. The build did not see the GPU, so"
    echo "nothing was tested. Do not file this as a validation."
    echo "Check: driver version, and whether pixi resolved a GPU-capable MAX."
else
    echo "GPU suite saw a device. Now read the file before filing it:"
    echo "  1. every [exit N] line, N must be 0"
    echo "  2. the training MSE next to every timing (a throughput number"
    echo "     without a loss number next to it is not a result)"
    echo "  3. the device attribute block from gpu-validate"
fi
echo
echo "Next: paste into docs/GPU_VALIDATION.md 'Recording a result', fill"
echo "hardware/templates/result_${VENDOR}.json, drop both in hardware/results/,"
echo "and move the row in docs/PLATFORM_MATRIX.md off 'designed'."
echo "Absolute paths in tool output can carry your username. Read before sharing."
echo "========================================="
