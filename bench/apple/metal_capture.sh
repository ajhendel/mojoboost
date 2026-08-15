#!/usr/bin/env bash
#
# metal_capture.sh, version 1.0.0
#
# Records an Instruments "Metal System Trace" of one mojotrees GPU training
# run on Apple silicon and reduces it with bench/apple/metal_timeline.py.
# The protocol and the results are docs/METAL_TIMELINE.md.
#
# Unlike bench/apple/thermal_capture.sh, this script does measure. It can,
# because nothing it runs is privileged: `xctrace record` needs no root, opens
# no authorization prompt on a machine with Xcode installed, and touches
# nothing outside the output directory. The thermal script refuses to execute
# because its plan contains `sudo powermetrics`; this one contains no such
# command and there is nothing to protect a reader from except the disk cost.
#
# WHAT IT CAPTURES
#
#   Every Metal command buffer the run submits, with four timestamps each:
#   when the host's `commit` call started and ended, when the GPU began
#   executing the encoder, when the GPU finished it, and when the completion
#   notification came back to the process. From those five hundred thousand
#   or so timestamps the reader derives GPU busy and idle time, the size of
#   every gap between kernels, the cost of one enqueue, the queue latency
#   before a kernel starts, and which command buffers the host actually
#   blocked on rather than pipelining past.
#
#   It also captures the GPU hardware's own Active/Idle record for the whole
#   machine, which is an independent cross-check on the per-process numbers,
#   and the GPU performance state, which is the reason two captures of the
#   same shape can disagree by a factor of three.
#
# WHAT IT CANNOT CAPTURE
#
#   Occupancy, ALU utilization, memory bandwidth, cache behavior, register
#   pressure, and anything else that would say whether a kernel is
#   latency-bound or bandwidth-bound. The Apple M4 in this machine offers
#   Instruments exactly one GPU counter, "RT Unit Active", which reports the
#   raytracing unit and is therefore worthless to a histogram. Adding
#   `--instrument 'Metal GPU Counters'` does not help and actively hurts: the
#   request is refused with "Selected counter profile is not supported on
#   target device" and the capture then contains zero counter samples instead
#   of the useless one. Do not add it back.
#
#   Kernel names. MAX sets no labels and pushes no debug groups on its Metal
#   encoders, so every dispatch in the trace is called "Compute Command 0" and
#   every copy "Blit Command 0". Which kernel ran is recoverable only by
#   position and duration, and the reader marks every such claim INFERRED.
#
#   Shader Timeline, which would give per-line cost inside a kernel, is a
#   recording setting the command line cannot reach. It is off in every trace
#   this script takes.
#
# WHAT IT COSTS
#
#   About 120 MB of disk per four seconds of traced run, and roughly 15% added
#   to the run's wall clock at 200,000 rows. The GPU-side timestamps come from
#   the GPU's own clock and are not distorted by that overhead; the CPU-side
#   ones are measured on an instrumented process and are upper bounds.
#
# WHAT IT DOES NOT COMPARE
#
#   Two traces taken minutes apart. The GPU performance state is a device
#   decision, and captures taken here at 100k and 400k rows sat at Minimum for
#   100% of their recorded state time while the 200k capture sat at Maximum
#   for 78% of its. Kernel durations across such a pair differ by the clock,
#   not by the workload. The reader prints the state breakdown for exactly
#   this reason; if it does not match between two captures, they are not
#   comparable and no scaling claim may be made from them.
#
# USAGE
#
#   bash bench/apple/metal_capture.sh --self-check
#   bash bench/apple/metal_capture.sh --print-plan
#   bash bench/apple/metal_capture.sh                       # 200000 50 reg 1 gpu
#   bash bench/apple/metal_capture.sh --rows 200000 --features 50 --arm gpu
#   bash bench/apple/metal_capture.sh --analyze-only <trace-path>
#
# EXIT CODES
#
#   0  capture taken and reduced, or self-check passed, or plan printed
#   2  usage or validation error
#   3  a required tool is missing
#   4  self-check failed
#   5  the capture ran but produced no GPU intervals for the target process

set -euo pipefail

SCRIPT_VERSION="1.0.0"
TEMPLATE="Metal System Trace"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
READER="${HERE}/metal_timeline.py"
DOC_PATH="${REPO_ROOT}/docs/METAL_TIMELINE.md"
MANIFEST="${MOJOTREES_PIXI_MANIFEST:-${REPO_ROOT}/pixi.toml}"

ROWS=200000
FEATURES=50
OBJECTIVE=reg
REPEATS=1
ARM=gpu
OUT_DIR="${HERE}/metal_traces"
RUN_ID=""
BUILD_DIR="${REPO_ROOT}/build"
PRINT_PLAN=0
SELF_CHECK=0
ANALYZE_ONLY=""
KEEP_EXPORT=1
REBUILD=0

die() {
    printf 'metal_capture.sh: %s\n' "$1" >&2
    printf 'run with --help for usage\n' >&2
    exit 2
}

usage() { sed -n '3,96p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

# 0 the template is there, 1 it is not, 2 the listing itself failed.
# `xctrace list templates` intermittently returns nothing at all, and a check
# that cannot tell that apart from an absent template reports the wrong repair.
have_template() {
    local listing
    listing="$(xcrun xctrace list templates 2>/dev/null || true)"
    [[ -n "$listing" ]] || return 2
    grep -qx "$TEMPLATE" <<<"$listing"
}

need_value() { [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"; }

quote_cmd() {
    local out="" arg esc
    for arg in "$@"; do
        if [[ "$arg" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
            out+="${arg} "
        else
            esc="${arg//\'/\'\\\'\'}"
            out+="'${esc}' "
        fi
    done
    printf '%s' "${out% }"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --version) printf 'metal_capture.sh %s\n' "$SCRIPT_VERSION"; exit 0 ;;
        --self-check) SELF_CHECK=1; shift ;;
        --print-plan) PRINT_PLAN=1; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --rows) need_value "$1" "${2:-}"; is_uint "$2" || die "--rows must be a whole number"
            [[ "$2" -ge 1000 ]] || die "--rows below 1000 trains nothing worth tracing"
            [[ "$2" -le 500000 ]] || die "--rows above 500000 produces a trace too large to read; the dispatch count does not grow with rows, so a bigger shape buys no extra structure"
            ROWS="$2"; shift 2 ;;
        --features) need_value "$1" "${2:-}"; is_uint "$2" || die "--features must be a whole number"
            [[ "$2" -ge 1 ]] || die "--features must be at least 1"
            FEATURES="$2"; shift 2 ;;
        --objective) need_value "$1" "${2:-}"
            [[ "$2" == reg || "$2" == binary ]] || die "--objective must be reg or binary"
            OBJECTIVE="$2"; shift 2 ;;
        --repeats) need_value "$1" "${2:-}"; is_uint "$2" || die "--repeats must be a whole number"
            [[ "$2" -eq 1 ]] || die "--repeats above 1 puts N runs in one trace and makes every per-round number an average over arms; capture once per question instead"
            REPEATS="$2"; shift 2 ;;
        --arm) need_value "$1" "${2:-}"
            case "$2" in gpu|gpu-host|gpu-device) ;; *) die "--arm must be gpu, gpu-host, or gpu-device; a cpu arm has no Metal work to trace" ;; esac
            ARM="$2"; shift 2 ;;
        --out-dir) need_value "$1" "${2:-}"; OUT_DIR="$2"; shift 2 ;;
        --run-id) need_value "$1" "${2:-}"
            [[ "$2" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id may contain letters, digits, dot, underscore, and hyphen only"
            RUN_ID="$2"; shift 2 ;;
        --analyze-only) need_value "$1" "${2:-}"; ANALYZE_ONLY="$2"; shift 2 ;;
        --) shift; break ;;
        -*) die "unknown option: $1" ;;
        *) die "unexpected argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Self-check.
# ---------------------------------------------------------------------------

if [[ "$SELF_CHECK" -eq 1 ]]; then
    failures=0
    note() { printf '  %s\n' "$1"; }
    fail() { printf '  FAIL %s\n' "$1"; failures=$((failures + 1)); }

    printf 'metal_capture.sh --self-check\n'

    for f in "$READER" "$DOC_PATH" "${REPO_ROOT}/bench/bench_train_gpu.mojo"; do
        if [[ -f "$f" ]]; then note "found ${f#"${REPO_ROOT}/"}"; else fail "missing ${f#"${REPO_ROOT}/"}"; fi
    done

    if [[ "$(uname -s)" == "Darwin" ]]; then
        note "host is macOS"
    else
        fail "host is not macOS; Metal System Trace exists nowhere else"
    fi

    if command -v xcrun >/dev/null 2>&1 && xcrun xctrace version >/dev/null 2>&1; then
        note "xctrace present: $(xcrun xctrace version 2>&1 | head -1)"
        case "$(have_template; echo $?)" in
            0) note "template '${TEMPLATE}' is installed" ;;
            1) fail "template '${TEMPLATE}' is not installed; a full Xcode is required, Command Line Tools alone do not ship it" ;;
            *) fail "could not list templates; xctrace returned nothing. It does this intermittently, so try again before believing it" ;;
        esac
    else
        fail "xctrace is unavailable; install Xcode"
    fi

    if command -v python3 >/dev/null 2>&1; then
        note "python3 present: $(python3 --version 2>&1)"
    else
        fail "python3 is unavailable; the reader needs it"
    fi

    # The reader must not have acquired a third-party import. It runs against
    # whatever python3 the machine has, deliberately, so that a trace can be
    # read without the pixi environment.
    if grep -nE '^\s*(import|from)\s+' "$READER" 2>/dev/null \
        | grep -vE '(argparse|collections|os|re|statistics|subprocess|sys|tempfile|xml)' | grep -q .; then
        fail "the reader imports something outside the standard library:"
        grep -nE '^\s*(import|from)\s+' "$READER" | grep -vE '(argparse|collections|os|re|statistics|subprocess|sys|tempfile|xml)' || true
    else
        note "reader imports the standard library only"
    fi

    # The counter instrument must stay out. Adding it back silently empties
    # the one counter table the default template does record. The flag name is
    # assembled at runtime so that this check cannot match the line performing
    # it; the comment block above is excluded by the comment filter.
    instrument_flag="--in""strument"
    if grep -nE -- "$instrument_flag" "${BASH_SOURCE[0]}" | grep -v '^[0-9]*: *#' | grep -q .; then
        fail "an instrument flag is in command position; see the WHAT IT CANNOT CAPTURE block"
        grep -nE -- "$instrument_flag" "${BASH_SOURCE[0]}" | grep -v '^[0-9]*: *#' || true
    else
        note "no instrument flag; the default template's counter set is left alone"
    fi

    priv_pattern='(^|[;&|(`]|\$\()[[:space:]]*(sudo|powermetrics|pmset)[[:space:]]'
    if grep -nE "$priv_pattern" "${BASH_SOURCE[0]}" | grep -v '^[0-9]*: *#' | grep -q .; then
        fail "a privileged command appears in command position"
    else
        note "no privileged command anywhere in this script"
    fi

    printf '\n'
    if [[ "$failures" -eq 0 ]]; then
        printf 'self-check ok. No capture was taken.\n'
        exit 0
    fi
    printf 'self-check failed: %s problem(s). No capture was taken.\n' "$failures"
    exit 4
fi

# ---------------------------------------------------------------------------
# Analyze an existing trace and stop.
# ---------------------------------------------------------------------------

if [[ -n "$ANALYZE_ONLY" ]]; then
    [[ -e "$ANALYZE_ONLY" ]] || die "no such trace: $ANALYZE_ONLY"
    exec python3 "$READER" "$ANALYZE_ONLY" --process bench_train_gpu --rounds 100
fi

# ---------------------------------------------------------------------------
# Plan.
# ---------------------------------------------------------------------------

[[ -n "$RUN_ID" ]] || RUN_ID="metal-${ROWS}x${FEATURES}-${OBJECTIVE}-${ARM}-$(date +%Y%m%dT%H%M%S)"
TRACE_PATH="${OUT_DIR}/${RUN_ID}.trace"
BINARY="${BUILD_DIR}/bench_train_gpu"

BUILD_PKG_CMD=(pixi run --manifest-path "$MANIFEST" mojo precompile -I "${REPO_ROOT}/src" "${REPO_ROOT}/src/mojotrees" -o "${BUILD_DIR}/mojotrees.mojopkg")
BUILD_BIN_CMD=(pixi run --manifest-path "$MANIFEST" mojo build -I "$BUILD_DIR" "${REPO_ROOT}/bench/bench_train_gpu.mojo" -o "$BINARY")
RECORD_CMD=(xcrun xctrace record --template "$TEMPLATE" --output "$TRACE_PATH" --no-prompt --target-stdout - --launch -- "$BINARY" "$ROWS" "$FEATURES" "$OBJECTIVE" "$REPEATS" "$ARM")
READ_CMD=(python3 "$READER" "$TRACE_PATH" --process bench_train_gpu --rounds 100)

if [[ "$PRINT_PLAN" -eq 1 ]]; then
    printf 'metal_capture.sh %s plan\n\n' "$SCRIPT_VERSION"
    printf 'shape       %s rows x %s features, %s, %s repeat, arm %s\n' "$ROWS" "$FEATURES" "$OBJECTIVE" "$REPEATS" "$ARM"
    printf 'template    %s\n' "$TEMPLATE"
    printf 'would write %s\n\n' "$TRACE_PATH"
    printf 'It would run, in order:\n\n'
    printf '    %s\n' "$(quote_cmd "${BUILD_PKG_CMD[@]}")"
    printf '    %s\n' "$(quote_cmd "${BUILD_BIN_CMD[@]}")"
    printf '    %s\n' "$(quote_cmd "${RECORD_CMD[@]}")"
    printf '    %s\n\n' "$(quote_cmd "${READ_CMD[@]}")"
    printf 'NOTHING WAS RUN AND NO FILE WAS WRITTEN.\n'
    exit 0
fi

# ---------------------------------------------------------------------------
# Capture.
# ---------------------------------------------------------------------------

command -v xcrun >/dev/null 2>&1 || { printf 'metal_capture.sh: xcrun not found; install Xcode\n' >&2; exit 3; }
have_template || { printf 'metal_capture.sh: template "%s" is unavailable (a full Xcode is required); xctrace also returns an empty template list intermittently, so try again\n' "$TEMPLATE" >&2; exit 3; }
command -v python3 >/dev/null 2>&1 || { printf 'metal_capture.sh: python3 not found\n' >&2; exit 3; }

mkdir -p "$OUT_DIR" "$BUILD_DIR"

# The binary is built ahead of the capture rather than traced through
# `mojo run`, because `mojo run` compiles in the same process and would put
# several seconds of a Mojo compile at the head of the trace. It changes
# nothing about the measurement and it halves the file.
if [[ "$REBUILD" -eq 1 || ! -x "$BINARY" ]]; then
    printf '==> building %s\n' "$BINARY"
    "${BUILD_PKG_CMD[@]}" >/dev/null
    "${BUILD_BIN_CMD[@]}" >/dev/null
fi
[[ -x "$BINARY" ]] || { printf 'metal_capture.sh: build produced no binary at %s\n' "$BINARY" >&2; exit 3; }

printf '==> recording %s\n' "$TRACE_PATH"
printf '    %s\n' "$(quote_cmd "${RECORD_CMD[@]}")"
rm -rf "$TRACE_PATH"
"${RECORD_CMD[@]}"

printf '\n==> reducing\n'
if ! "${READ_CMD[@]}"; then
    printf 'metal_capture.sh: the trace has no GPU intervals for bench_train_gpu.\n' >&2
    printf 'The run may have fallen back to the CPU path. Check the arm and the device.\n' >&2
    exit 5
fi

printf '\ntrace kept at %s\n' "$TRACE_PATH"
printf 'exported tables cached in %s.export\n' "$TRACE_PATH"
printf 'Open it in Instruments with: open %s\n' "$(quote_cmd "$TRACE_PATH")"
