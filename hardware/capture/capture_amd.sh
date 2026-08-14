#!/bin/sh
# mojoboost environment capture: AMD, HIP.
#
# Prints the machine, driver, and toolchain metadata that
# hardware/templates/result_amd.json asks for. Run it once, keep the output,
# attach it to your report.
#
#   sh hardware/capture/capture_amd.sh > amd-capture.txt 2>&1
#   sh hardware/capture/capture_amd.sh --commands-only
#
# What this script does: run read-only informational commands, printing each
# command before its output so a reviewer can see exactly what produced each
# line.
#
# What this script does not do, by construction: install anything, run a
# package manager, use sudo, write any file, set any variable outside its own
# shell, contact the network, upload anything, or build any part of mojoboost.
# Read it before you run it; it is short on purpose.
#
# It does not call `pixi run` either. `pixi run` solves and installs the
# environment on first use, which is a mutation, so the Mojo and MAX versions
# are read only if those tools are already on PATH. Take them from your own
# `pixi run mojo --version` in the validation session instead.
#
# Secrets and identifying data. The command list is fixed and narrow. It does
# not print the hostname, the username, your environment as a whole, the GPU
# serial number, or any git remote URL, which is the one field in this area
# that has been known to carry an access token. It does print your group names,
# because render-group membership is the single most common cause of a machine
# with a working card reporting no accelerator. The only environment variables
# printed are the named mojoboost and threading ones below. Absolute paths in
# tool output can still contain your username, so read the file before you
# attach it, and say what you changed if you change anything.
#
# Exit status is 0 whenever the script itself ran, including when no ROCm
# tooling exists on the machine. A capture that finds no device is a complete
# result: file it with verdict `unsupported` and the reason.

set -u

CAPTURE_VERSION="1.0.0"
COMMANDS_ONLY=0

usage() {
    printf 'usage: sh capture_amd.sh [--commands-only] [--help]\n'
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
    eval "value=\"\${$1-__mojoboost_unset__}\""
    if [ "$value" = "__mojoboost_unset__" ]; then
        printf '%s=<unset>\n' "$1"
    else
        printf '%s=%s\n' "$1" "$value"
    fi
}

printf 'mojoboost hardware capture\n'
printf 'vendor:           amd\n'
printf 'api:              hip\n'
printf 'capture script:   hardware/capture/capture_amd.sh\n'
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

section "rocm runtime and device"
run 'command -v rocm-smi'
run 'rocm-smi'
run 'rocm-smi --showproductname'
run 'rocm-smi --showdriverversion'
run 'rocm-smi --showmeminfo vram'
run 'command -v rocminfo'
run 'rocminfo | grep -iE "^  name|gfx|compute unit|marketing"'
run 'cat /opt/rocm/.info/version'
run 'ls -d /opt/rocm*'
run 'lsmod | grep -i amdgpu'

section "device node permissions, the usual cause of a false skip"
run 'ls -l /dev/kfd /dev/dri'
run 'id -nG'

section "hip toolchain and profilers, none of which are required to run mojoboost"
run 'command -v hipcc'
run 'hipcc --version'
run 'command -v rocprofv3'
run 'command -v rocprof-compute'

section "toolchain, read only if already on PATH"
run 'command -v pixi'
run 'pixi --version'
run 'command -v mojo'
run 'mojo --version'
run 'command -v python3'
run 'python3 --version'

section "mojoboost source"
run 'git rev-parse HEAD'
run 'git rev-parse --abbrev-ref HEAD'
run 'git status --porcelain'

section "environment variables that change results"
if [ "$COMMANDS_ONLY" -eq 1 ]; then
    printf '\n(the fixed variable allowlist, printed one per line)\n'
else
    printf '\n'
    show_var MOJOBOOST_NUM_WORKERS
    show_var MOJOBOOST_PARALLEL_MIN_OPS
    show_var MOJOBOOST_DISABLE_GPU
    show_var MOJOBOOST_GPU_HIST_STRATEGY
    show_var MOJOBOOST_GPU_ROW_TILE
    show_var MOJOBOOST_GPU_BLOCK_THREADS
    show_var MOJOBOOST_AUTO_MIN_CELLS
    show_var OMP_NUM_THREADS
    show_var MKL_NUM_THREADS
    show_var HIP_VISIBLE_DEVICES
    show_var ROCR_VISIBLE_DEVICES
    show_var HSA_OVERRIDE_GFX_VERSION
fi

section "notes for the record"
cat <<'NOTE'

Deliberately not collected: hostname, username, full environment, GPU serial
number, git remote URL.

Fill hardware/templates/result_amd.json from the output above. Fields that no
command answered stay null. `unavailable`, `not installed`, and `no device` are
answers; none of them is a blank.

Record the gfx target, not only the board name. gfx1100 and gfx942 are different
machines wearing the same vendor logo, and a consumer RDNA result transfers to
datacenter CDNA about as well as a Metal result does.

If HSA_OVERRIDE_GFX_VERSION printed anything other than <unset>, the runtime was
told to treat this card as a different one. That belongs in
findings.workarounds_used, and the record must say which target actually ran. An
overridden target is a different device from the one the record names.

If the installed ROCm does not list this card as a supported target, or rocm-smi
is absent, stop here. Set verdict.correctness to `unsupported`, put the exact
error and the ROCm version that refused it in verdict.unsupported_reason, and
file it. That refusal is a real result and nobody has recorded one yet.

If ROCm does see the device, continue with docs/GPU_VALIDATION.md, which is the
authoritative procedure. Watch for a GPU suite that prints
`skipped: no accelerator`: on this machine that means the build did not see the
device, usually the render group or the support list, and it is a finding rather
than a pass.
NOTE

exit 0
