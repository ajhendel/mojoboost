#!/bin/sh
# mojotrees environment capture: NVIDIA, CUDA.
#
# Prints the machine, driver, and toolchain metadata that
# hardware/templates/result_nvidia.json asks for. Run it once, keep the
# output, attach it to your report.
#
#   sh hardware/capture/capture_nvidia.sh > nvidia-capture.txt 2>&1
#   sh hardware/capture/capture_nvidia.sh --commands-only
#
# What this script does: run read-only informational commands, printing each
# command before its output so a reviewer can see exactly what produced each
# line.
#
# What this script does not do, by construction: install anything, run a
# package manager, use sudo, write any file, set any variable outside its own
# shell, contact the network, upload anything, or build any part of mojotrees.
# Read it before you run it; it is short on purpose.
#
# It does not call `pixi run` either. `pixi run` solves and installs the
# environment on first use, which is a mutation, so the Mojo and MAX versions
# are read only if those tools are already on PATH. Take them from your own
# `pixi run mojo --version` in the validation session instead.
#
# Secrets and identifying data. The command list is fixed and narrow. It does
# not print the hostname, the username, your environment as a whole, the GPU
# UUID or serial, or any git remote URL, which is the one field in this area
# that has been known to carry an access token. The only environment variables
# printed are the named mojotrees and threading ones below. Absolute paths in
# tool output can still contain your username, so read the file before you
# attach it, and say what you changed if you change anything.
#
# Exit status is 0 whenever the script itself ran, including when no NVIDIA
# tooling exists on the machine. A capture that finds no device is a complete
# result: file it with verdict `unsupported` and the reason.

set -u

CAPTURE_VERSION="1.0.0"
COMMANDS_ONLY=0

usage() {
    printf 'usage: sh capture_nvidia.sh [--commands-only] [--help]\n'
    printf '  --commands-only   print the command list and run nothing\n'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --commands-only) COMMANDS_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

section() {
    printf '\n===== %s =====\n' "$1"
}

run() {
    printf '\n$ %s\n' "$1"
    if [ "$COMMANDS_ONLY" -eq 1 ]; then
        return 0
    fi
    sh -c "$1" 2>&1 || printf '[command failed or is not installed, exit %s]\n' "$?"
}

show_var() {
    eval "value=\"\${$1-__mojotrees_unset__}\""
    if [ "$value" = "__mojotrees_unset__" ]; then
        printf '%s=<unset>\n' "$1"
    else
        printf '%s=%s\n' "$1" "$value"
    fi
}

printf 'mojotrees hardware capture\n'
printf 'vendor:           nvidia\n'
printf 'api:              cuda\n'
printf 'capture script:   hardware/capture/capture_nvidia.sh\n'
printf 'capture version:  %s\n' "$CAPTURE_VERSION"
printf 'protocol:         docs/HARDWARE_CONTRIBUTORS.md 1.0.0\n'
if [ "$COMMANDS_ONLY" -eq 1 ]; then
    printf 'mode:             commands only, nothing was executed\n'
fi

section "when"
run 'date -u +%Y-%m-%dT%H:%M:%SZ'

section "host"
run 'uname -srm'
run 'cat /etc/os-release'
run 'lscpu | grep -iE "model name|^architecture|^cpu\(s\)|^socket|core\(s\) per socket|^thread"'
run 'grep -c ^processor /proc/cpuinfo'
run 'grep -i MemTotal /proc/meminfo'

section "virtualization and containers"
run 'systemd-detect-virt'
run 'test -f /.dockerenv && echo "/.dockerenv present" || echo "/.dockerenv absent"'

section "nvidia driver and device"
run 'command -v nvidia-smi'
run 'nvidia-smi'
run 'nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv'
run 'cat /proc/driver/nvidia/version'
run 'lsmod | grep -i nvidia'

section "cuda toolkit and profilers, none of which are required to run mojotrees"
run 'command -v nvcc'
run 'nvcc --version'
run 'command -v ncu'
run 'command -v nsys'

section "toolchain, read only if already on PATH"
run 'command -v pixi'
run 'pixi --version'
run 'command -v mojo'
run 'mojo --version'
run 'command -v python3'
run 'python3 --version'

section "mojotrees source"
run 'git rev-parse HEAD'
run 'git rev-parse --abbrev-ref HEAD'
run 'git status --porcelain'

section "environment variables that change results"
if [ "$COMMANDS_ONLY" -eq 1 ]; then
    printf '\n(the fixed variable allowlist, printed one per line)\n'
else
    printf '\n'
    show_var MOJOTREES_NUM_WORKERS
    show_var MOJOTREES_PARALLEL_MIN_OPS
    show_var MOJOTREES_DISABLE_GPU
    show_var MOJOTREES_GPU_HIST_STRATEGY
    show_var MOJOTREES_GPU_ROW_TILE
    show_var MOJOTREES_GPU_BLOCK_THREADS
    show_var MOJOTREES_AUTO_MIN_CELLS
    show_var OMP_NUM_THREADS
    show_var MKL_NUM_THREADS
    show_var CUDA_VISIBLE_DEVICES
fi

section "notes for the record"
cat <<'NOTE'

Deliberately not collected: hostname, username, full environment, GPU UUID and
serial number, git remote URL.

Fill hardware/templates/result_nvidia.json from the output above. Fields that no
command answered stay null. `unavailable`, `not installed`, and `no device` are
answers; none of them is a blank.

If nvidia-smi is absent or reports no device, stop here. Set verdict.correctness
to `unsupported`, put the exact error in verdict.unsupported_reason, and file it.
A machine that cannot run this code is worth recording once so nobody spends an
afternoon rediscovering it.

If nvidia-smi does see a device, continue with docs/GPU_VALIDATION.md, which is
the authoritative procedure. Watch for a GPU suite that prints
`skipped: no accelerator`: on this machine that means the build did not see the
device, and it is a finding rather than a pass.
NOTE

exit 0
