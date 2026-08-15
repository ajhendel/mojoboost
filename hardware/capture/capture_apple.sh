#!/bin/sh
# mojotrees environment capture: Apple silicon, Metal.
#
# Prints the machine, Metal toolchain, and thermal metadata that
# hardware/templates/result_apple.json asks for. Run it once, keep the output,
# attach it to your report.
#
#   sh hardware/capture/capture_apple.sh > apple-capture.txt 2>&1
#   sh hardware/capture/capture_apple.sh --commands-only
#
# What this script does: run read-only informational commands, printing each
# command before its output so a reviewer can see exactly what produced each
# line.
#
# What this script does not do, by construction: install anything, run a
# package manager, use sudo, write any file, set any variable outside its own
# shell, contact the network, upload anything, or build any part of mojotrees.
# It never runs powermetrics, which needs root. Read it before you run it; it is
# short on purpose.
#
# It does not call `pixi run` either. `pixi run` solves and installs the
# environment on first use, which is a mutation, so the Mojo and MAX versions
# are read only if those tools are already on PATH. Take them from your own
# `pixi run mojo --version` in the validation session instead.
#
# Secrets and identifying data. The command list is fixed and narrow. It does
# not print the hostname, the username, your environment as a whole, or any git
# remote URL, which is the one field in this area that has been known to carry
# an access token. It reads the machine identity from `sysctl hw.model` rather
# than from `system_profiler SPHardwareDataType`, because that report contains
# the serial number and the hardware UUID and this record needs neither. The
# display report is asked for at `-detailLevel mini` for the same reason. The
# only environment variables printed are the named mojotrees and threading ones
# below. Absolute paths in tool output can still contain your username, so read
# the file before you attach it, and say what you changed if you change
# anything.
#
# Exit status is 0 whenever the script itself ran, including when the Metal
# compiler is absent. A Mac that cannot build the GPU path is a complete result:
# file it with verdict `unsupported` and the reason.

set -u

CAPTURE_VERSION="1.0.0"
COMMANDS_ONLY=0

usage() {
    printf 'usage: sh capture_apple.sh [--commands-only] [--help]\n'
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
printf 'vendor:           apple\n'
printf 'api:              metal\n'
printf 'capture script:   hardware/capture/capture_apple.sh\n'
printf 'capture version:  %s\n' "$CAPTURE_VERSION"
printf 'protocol:         docs/HARDWARE_CONTRIBUTORS.md 1.0.0\n'
if [ "$COMMANDS_ONLY" -eq 1 ]; then
    printf 'mode:             commands only, nothing was executed\n'
fi

section "when"
run 'date -u +%Y-%m-%dT%H:%M:%SZ'

section "machine"
run 'sw_vers'
run 'uname -srm'
run 'sysctl -n hw.model'
run 'sysctl -n machdep.cpu.brand_string'
run 'sysctl -n hw.physicalcpu hw.logicalcpu'
run 'sysctl -n hw.perflevel0.physicalcpu'
run 'sysctl -n hw.perflevel1.physicalcpu'
run 'sysctl -n hw.memsize'

section "gpu"
run 'system_profiler -detailLevel mini SPDisplaysDataType'

section "metal toolchain, which is a separate installation from having a gpu"
run 'xcode-select -p'
run 'xcrun --find metal'
run 'xcrun metal --version'
run 'xcodebuild -version'

section "thermal and power state, which decide whether a timing means anything"
run 'pmset -g therm'
run 'pmset -g batt'
run 'pmset -g | grep -i lowpowermode'
run 'sysctl -n vm.loadavg'

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
    show_var VECLIB_MAXIMUM_THREADS
fi

section "notes for the record"
cat <<'NOTE'

Deliberately not collected: hostname, username, full environment, serial number,
hardware UUID, git remote URL. `system_profiler SPHardwareDataType` is the report
that carries the first two of those and this script does not run it.

Fill hardware/templates/result_apple.json from the output above. Fields that no
command answered stay null. If the GPU core count is absent from the display
report on this macOS release, leave it null; never fill it in from the chip name.

If `xcrun --find metal` failed, this Mac has no Metal compiler and cannot build
the GPU path, however good its GPU is. That is a legitimate and useful record:
fill in the environment, set verdict.correctness to `unsupported`,
write `build failed, Metal toolchain absent` with the exact error in
verdict.unsupported_reason, and stop. It is the difference between "the GPU path
is broken on this chip" and "this machine could never have built it", and the
next person on the same macOS version should not have to rediscover which one
they are looking at.

If the toolchain is present, continue with docs/GPU_VALIDATION.md for
correctness and determinism. For timings, energy, and anything you intend to
quote, docs/APPLE_GPU_BENCHMARK_PROTOCOL.md is the authoritative procedure and
its idle gate is not optional: `pmset -g therm` must read CPU_Speed_Limit = 100,
the machine must be on AC power with Low Power Mode off, and nothing else may be
running. A laptop timing taken next to a browser is an anecdote.
NOTE

exit 0
