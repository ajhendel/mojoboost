#!/usr/bin/env bash
#
# thermal_capture.sh, version 1.0.0
#
# Plans a run of the MacBook thermal, energy, and sustained-performance
# protocol in docs/APPLE_THERMAL_ENERGY.md and prints, in order, the exact
# commands such a run would issue.
#
# This script does not measure anything. It starts no sampler, runs no
# privileged command, fits no model, and writes no record. It validates its
# arguments and prints a plan. That is the whole of it.
#
# The reason it is built this way is that the commands a thermal run wants to
# issue include `sudo powermetrics`, and a script that can be talked into
# running that by a typo is a script that eventually will. The plan is
# printed so a person can read it, decide, and run the parts they want by
# hand. `--execute` is parsed and refused; the exact commands a human may
# deliberately run later are listed in
# handoffs/performance_17_thermal_energy.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_17_thermal_energy.md).
#
# Usage:
#   bash bench/apple/thermal_capture.sh --list-phases
#   bash bench/apple/thermal_capture.sh --self-check
#   bash bench/apple/thermal_capture.sh --phase cold_fit --phase warm_fit
#   bash bench/apple/thermal_capture.sh --phase sustained --duration 1200 --energy
#   bash bench/apple/thermal_capture.sh --print-plan --phase sustained
#
# Exit codes:
#   0  plan printed, or self-check passed
#   2  usage or validation error
#   3  --execute refused; this version has no measurement path
#   4  self-check failed

set -euo pipefail

SCRIPT_VERSION="1.0.0"
THERMAL_PROTOCOL_VERSION="1.0.0"
THERMAL_SCHEMA_VERSION="1.0.0"
PARENT_PROTOCOL_VERSION="1.0.0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
SCHEMA_PATH="${HERE}/thermal_schema.json"
PROTOCOL_PATH="${REPO_ROOT}/docs/APPLE_THERMAL_ENERGY.md"
PARENT_PROTOCOL_PATH="${REPO_ROOT}/docs/APPLE_GPU_BENCHMARK_PROTOCOL.md"
HANDOFF_PATH="${REPO_ROOT}/handoffs/performance_17_thermal_energy.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_17_thermal_energy.md)"

# Phase catalog. Fields are id, default window in seconds, and a one-line
# description. The ids must match the phase enum in thermal_schema.json; the
# self-check verifies that rather than assuming it.
PHASE_IDS=(
    idle_baseline
    cold_fit
    warm_fit
    repeat_series
    sustained
    throttle_probe
    cooldown_recovery
    background_control
)
PHASE_WINDOWS=(60 0 0 0 1200 3600 1800 120)
PHASE_DESCRIPTIONS=(
    "sample the idle machine with the sampler the workload phases will use"
    "one fit, fresh process, machine idle and thermally clean"
    "one more fit in the same process, immediately, no cooldown"
    "ten fits back to back in one process, all timed, all scored"
    "identical fits back to back for the declared window, bucketed per minute"
    "continue the sustained load until a thermal limit appears or the ceiling is reached"
    "idle sampling until the thermal state returns to its pre-run value or the ceiling is reached"
    "negative control: apply a declared competing load, which the idle gate must reject"
)
REPEAT_SERIES_FITS=10
BUCKET_SECONDS=60

# Defaults.
OUT_DIR="${HERE}/thermal_results"
RUN_ID=""
INTERVAL_MS=200
DURATION_OVERRIDE=""
BASELINE_OVERRIDE=""
COOLDOWN_CEILING_OVERRIDE=""
WORKLOAD="w2_medium_dense"
DEVICE="cpu"
THREADS=""
POWER_SOURCE="ac"
LID="open"
SURFACE="hard_desk"
LINK_RECORD=""
ENERGY=0
PROCESS_ENERGY=0
PRINT_PLAN=0
LIST_PHASES=0
SELF_CHECK=0
EXECUTE=0
NOTES=()
SELECTED_PHASES=()

die() {
    printf 'thermal_capture.sh: %s\n' "$1" >&2
    printf 'run with --help for usage\n' >&2
    exit 2
}

usage() {
    sed -n '3,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Shell-quote a command so the printed plan can be copied without a reader
# having to guess where a word ends.
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

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

in_list() {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

phase_index() {
    local needle="$1" i
    for i in "${!PHASE_IDS[@]}"; do
        [[ "${PHASE_IDS[$i]}" == "$needle" ]] && { printf '%s' "$i"; return 0; }
    done
    return 1
}

# The window a phase would run for. --duration applies to the load phases
# only. Widening the idle baseline or the cooldown ceiling with the same flag
# would silently change what the baseline is a baseline of, so those have
# their own options.
phase_window() {
    local idx
    idx="$(phase_index "$1")" || return 1
    case "$1" in
        sustained|throttle_probe)
            printf '%s' "${DURATION_OVERRIDE:-${PHASE_WINDOWS[$idx]}}" ;;
        idle_baseline)
            printf '%s' "${BASELINE_OVERRIDE:-${PHASE_WINDOWS[$idx]}}" ;;
        cooldown_recovery)
            printf '%s' "${COOLDOWN_CEILING_OVERRIDE:-${PHASE_WINDOWS[$idx]}}" ;;
        *)
            printf '%s' "${PHASE_WINDOWS[$idx]}" ;;
    esac
}

# Selected phases, in the protocol's order rather than the order they were
# typed. The order matters: an idle baseline taken after the workload is a
# baseline of a machine the run just heated, which is the defect this
# repository's benchmark lane already had to fix once.
canonical_order() {
    local id
    for id in "${PHASE_IDS[@]}"; do
        if in_list "$id" "${SELECTED_PHASES[@]:-}"; then
            printf '%s\n' "$id"
        fi
    done
}

need_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage; exit 0 ;;
        --version)
            printf 'thermal_capture.sh %s, protocol %s, schema %s\n' \
                "$SCRIPT_VERSION" "$THERMAL_PROTOCOL_VERSION" "$THERMAL_SCHEMA_VERSION"
            exit 0 ;;
        --list-phases)
            LIST_PHASES=1; shift ;;
        --self-check)
            SELF_CHECK=1; shift ;;
        --print-plan)
            PRINT_PLAN=1; shift ;;
        --execute)
            EXECUTE=1; shift ;;
        --phase)
            need_value "$1" "${2:-}"
            in_list "$2" "${PHASE_IDS[@]}" || die "unknown phase: $2"
            in_list "$2" "${SELECTED_PHASES[@]:-}" || SELECTED_PHASES+=("$2")
            shift 2 ;;
        --all-phases)
            SELECTED_PHASES=("${PHASE_IDS[@]}"); shift ;;
        --duration)
            need_value "$1" "${2:-}"
            is_uint "$2" || die "--duration must be a whole number of seconds"
            [[ "$2" -ge 60 ]] || die "--duration below 60 s cannot fill one $BUCKET_SECONDS s bucket"
            [[ "$2" -le 14400 ]] || die "--duration above 14400 s is refused; split the run instead"
            DURATION_OVERRIDE="$2"; shift 2 ;;
        --baseline-seconds)
            need_value "$1" "${2:-}"
            is_uint "$2" || die "--baseline-seconds must be a whole number of seconds"
            [[ "$2" -ge 10 ]] || die "--baseline-seconds below 10 is too few samples to be a baseline"
            [[ "$2" -le 1800 ]] || die "--baseline-seconds above 1800 is refused"
            BASELINE_OVERRIDE="$2"; shift 2 ;;
        --cooldown-ceiling)
            need_value "$1" "${2:-}"
            is_uint "$2" || die "--cooldown-ceiling must be a whole number of seconds"
            [[ "$2" -ge 60 ]] || die "--cooldown-ceiling below 60 s cannot observe a recovery"
            [[ "$2" -le 7200 ]] || die "--cooldown-ceiling above 7200 s is refused"
            COOLDOWN_CEILING_OVERRIDE="$2"; shift 2 ;;
        --interval-ms)
            need_value "$1" "${2:-}"
            is_uint "$2" || die "--interval-ms must be a whole number of milliseconds"
            [[ "$2" -ge 50 ]] || die "--interval-ms below 50 makes the sampler a load of its own"
            [[ "$2" -le 10000 ]] || die "--interval-ms above 10000 cannot resolve a fit"
            INTERVAL_MS="$2"; shift 2 ;;
        --workload)
            need_value "$1" "${2:-}"
            [[ "$2" =~ ^[a-z0-9_]+$ ]] || die "--workload must be a lowercase identifier"
            WORKLOAD="$2"; shift 2 ;;
        --device)
            need_value "$1" "${2:-}"
            in_list "$2" cpu gpu || die "--device must be cpu or gpu"
            DEVICE="$2"; shift 2 ;;
        --threads)
            need_value "$1" "${2:-}"
            is_uint "$2" || die "--threads must be a whole number"
            [[ "$2" -ge 1 ]] || die "--threads must be at least 1"
            THREADS="$2"; shift 2 ;;
        --power-source)
            need_value "$1" "${2:-}"
            in_list "$2" ac battery || die "--power-source must be ac or battery"
            POWER_SOURCE="$2"; shift 2 ;;
        --lid)
            need_value "$1" "${2:-}"
            in_list "$2" open closed_clamshell || die "--lid must be open or closed_clamshell"
            LID="$2"; shift 2 ;;
        --surface)
            need_value "$1" "${2:-}"
            in_list "$2" hard_desk stand soft lap || die "--surface must be hard_desk, stand, soft, or lap"
            SURFACE="$2"; shift 2 ;;
        --link-record)
            need_value "$1" "${2:-}"
            LINK_RECORD="$2"; shift 2 ;;
        --run-id)
            need_value "$1" "${2:-}"
            [[ "$2" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id may contain letters, digits, dot, underscore, and hyphen only"
            RUN_ID="$2"; shift 2 ;;
        --out-dir)
            need_value "$1" "${2:-}"
            OUT_DIR="$2"; shift 2 ;;
        --energy)
            ENERGY=1; shift ;;
        --process-energy)
            PROCESS_ENERGY=1; shift ;;
        --note)
            need_value "$1" "${2:-}"
            NOTES+=("$2"); shift 2 ;;
        --)
            shift; break ;;
        -*)
            die "unknown option: $1" ;;
        *)
            die "unexpected argument: $1" ;;
    esac
done

if [[ "$EXECUTE" -eq 1 ]]; then
    cat >&2 <<EOF
thermal_capture.sh: --execute is refused.

This version has no measurement path. It cannot start a sampler, cannot fit a
model, and will not run a privileged command on your behalf. The flag is
parsed so that passing it produces this message rather than a silent no-op
that reads like a completed run.

What to do instead:

  1. Run this script without --execute and read the plan it prints.
  2. Run the parts you want by hand, from
     handoffs/performance_17_thermal_energy.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_17_thermal_energy.md), which lists them with the
     privileged ones marked.

Nothing was measured and no file was written.
EOF
    exit 3
fi

if [[ "$LIST_PHASES" -eq 1 ]]; then
    printf 'Phases defined by docs/APPLE_THERMAL_ENERGY.md %s\n\n' "$THERMAL_PROTOCOL_VERSION"
    printf '%-20s %10s  %s\n' "id" "window(s)" "what it does"
    for i in "${!PHASE_IDS[@]}"; do
        w="${PHASE_WINDOWS[$i]}"
        [[ "$w" == "0" ]] && w="by fits"
        printf '%-20s %10s  %s\n' "${PHASE_IDS[$i]}" "$w" "${PHASE_DESCRIPTIONS[$i]}"
    done
    printf '\nNo measurement was taken.\n'
    exit 0
fi

if [[ "$SELF_CHECK" -eq 1 ]]; then
    failures=0
    note() { printf '  %s\n' "$1"; }
    fail() { printf '  FAIL %s\n' "$1"; failures=$((failures + 1)); }

    printf 'thermal_capture.sh --self-check\n'

    # 1. Companion files exist. A plan that points at a protocol nobody can
    #    read is not a plan.
    for f in "$SCHEMA_PATH" "$PROTOCOL_PATH" "$PARENT_PROTOCOL_PATH" "$HANDOFF_PATH"; do
        if [[ -f "$f" ]]; then
            note "found ${f#"${REPO_ROOT}/"}"
        else
            fail "missing ${f#"${REPO_ROOT}/"}"
        fi
    done

    # 2. Catalog arrays are the same length. A mismatch here would print a
    #    window against the wrong phase.
    if [[ "${#PHASE_IDS[@]}" -eq "${#PHASE_WINDOWS[@]}" && "${#PHASE_IDS[@]}" -eq "${#PHASE_DESCRIPTIONS[@]}" ]]; then
        note "phase catalog consistent: ${#PHASE_IDS[@]} phases"
    else
        fail "phase catalog arrays differ in length"
    fi

    # 3. Every phase id this script knows appears in the schema's phase enum.
    #    This is the check that keeps the two files from drifting apart.
    if [[ -f "$SCHEMA_PATH" ]]; then
        for p in "${PHASE_IDS[@]}"; do
            if grep -q "\"${p}\"" "$SCHEMA_PATH"; then
                note "schema knows phase ${p}"
            else
                fail "schema has no phase id ${p}"
            fi
        done
        # 4. The prohibition the protocol rests on, checked in the file
        #    rather than trusted: no estimated energy method.
        if grep -q '"estimated"' "$SCHEMA_PATH"; then
            fail "schema contains an estimated value; energy is measured or null"
        else
            note "schema defines no estimated energy method"
        fi
    fi

    # 5. The schema parses, when a JSON reader is available. jq is not
    #    required to be installed, and its absence is reported rather than
    #    quietly treated as a pass.
    if command -v jq >/dev/null 2>&1; then
        if jq empty "$SCHEMA_PATH" >/dev/null 2>&1; then
            note "schema parses as JSON (jq)"
        else
            fail "schema does not parse as JSON"
        fi
    else
        note "jq not installed; schema was not parsed, only searched"
    fi

    # 6. This script has no path that runs a privileged or sampling command.
    #    The patterns look for a command position, meaning the start of a line
    #    or immediately after a separator, so that the word appearing inside a
    #    string, a comment, or an emit_priv argument is not a hit. Anything
    #    this script prints goes through emit_cmd or emit_priv and can never
    #    be in command position.
    priv_pattern='(^|[;&|(`]|\$\()[[:space:]]*(sudo|powermetrics|pmset|caffeinate|yes)[[:space:]]'
    if grep -nE "$priv_pattern" "${BASH_SOURCE[0]}" | grep -v '^[0-9]*: *#' | grep -q .; then
        fail "a privileged or sampling command appears in command position:"
        grep -nE "$priv_pattern" "${BASH_SOURCE[0]}" | grep -v '^[0-9]*: *#' || true
    else
        note "no sampler or privileged command in command position; they are printed only"
    fi

    printf '\n'
    if [[ "$failures" -eq 0 ]]; then
        printf 'self-check ok: %s phases, schema %s, protocol %s. No measurement was taken.\n' \
            "${#PHASE_IDS[@]}" "$THERMAL_SCHEMA_VERSION" "$THERMAL_PROTOCOL_VERSION"
        exit 0
    fi
    printf 'self-check failed: %s problem(s). No measurement was taken.\n' "$failures"
    exit 4
fi

# Default selection: everything except the two long tail phases, which are
# deliberately opt-in because between them they can occupy ninety minutes.
if [[ "${#SELECTED_PHASES[@]}" -eq 0 ]]; then
    SELECTED_PHASES=(idle_baseline cold_fit warm_fit repeat_series sustained background_control)
fi

ORDERED_PHASES=()
while IFS= read -r ordered_id; do
    ORDERED_PHASES+=("$ordered_id")
done < <(canonical_order)
SELECTED_PHASES=("${ORDERED_PHASES[@]}")

# Cross-argument validation. These are the combinations that produce a record
# nobody can use, and they are refused at plan time rather than discovered
# after an afternoon of sampling.
if [[ "$POWER_SOURCE" == "battery" ]]; then
    for p in "${SELECTED_PHASES[@]}"; do
        if [[ "$p" == "idle_baseline" ]]; then
            printf 'note: an idle baseline on battery is only subtractable from workload phases taken on battery in the same session.\n' >&2
        fi
    done
fi
if [[ "$LID" == "closed_clamshell" && "$POWER_SOURCE" == "battery" ]]; then
    die "a closed lid on battery with no external power is a sleeping machine, not a condition"
fi
if [[ "$ENERGY" -eq 0 && "$PROCESS_ENERGY" -eq 1 ]]; then
    die "--process-energy needs --energy; both come from powermetrics"
fi
if [[ "$ENERGY" -eq 1 ]]; then
    has_baseline=0
    for p in "${SELECTED_PHASES[@]}"; do
        [[ "$p" == "idle_baseline" ]] && has_baseline=1
    done
    if [[ "$has_baseline" -eq 0 ]]; then
        die "--energy without the idle_baseline phase yields absolute joules with nothing to subtract; add --phase idle_baseline"
    fi
fi
if [[ -n "$LINK_RECORD" && ! "$LINK_RECORD" =~ \.json$ ]]; then
    die "--link-record expects a bench/apple/results/<run_id>.json path"
fi
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="thermal-PENDING"
fi

RECORD_PATH="${OUT_DIR}/${RUN_ID}.json"

# ---------------------------------------------------------------------------
# Plan output.
# ---------------------------------------------------------------------------

emit() { printf '%s\n' "$*"; }
emit_cmd() { printf '    %s\n' "$(quote_cmd "$@")"; }
emit_priv() { printf '    [ROOT] %s\n' "$(quote_cmd "$@")"; }
emit_todo() { printf '    [NOT IMPLEMENTED] %s\n' "$1"; }

if [[ "$PRINT_PLAN" -eq 1 ]]; then
    # A machine-readable plan. This is a plan document, not a thermal record:
    # it contains no sample, it is not written by a run, and it does not
    # validate against thermal_schema.json.
    printf '{\n'
    printf '  "document": "thermal_capture_plan",\n'
    printf '  "not_a_record": "This is a plan. It contains no measurement and does not validate against bench/apple/thermal_schema.json.",\n'
    printf '  "capture_script_version": "%s",\n' "$SCRIPT_VERSION"
    printf '  "thermal_protocol_version": "%s",\n' "$THERMAL_PROTOCOL_VERSION"
    printf '  "thermal_schema_version": "%s",\n' "$THERMAL_SCHEMA_VERSION"
    printf '  "parent_protocol_version": "%s",\n' "$PARENT_PROTOCOL_VERSION"
    printf '  "executed": false,\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "would_write": "%s",\n' "$RECORD_PATH"
    printf '  "sample_interval_ms": %s,\n' "$INTERVAL_MS"
    printf '  "bucket_seconds": %s,\n' "$BUCKET_SECONDS"
    printf '  "energy_requested": %s,\n' "$([[ $ENERGY -eq 1 ]] && echo true || echo false)"
    printf '  "process_attribution_requested": %s,\n' "$([[ $PROCESS_ENERGY -eq 1 ]] && echo true || echo false)"
    printf '  "conditions": {"power_source": "%s", "lid_state": "%s", "surface": "%s"},\n' \
        "$POWER_SOURCE" "$LID" "$SURFACE"
    printf '  "workload": {"id": "%s", "device": "%s", "requested_threads": %s},\n' \
        "$WORKLOAD" "$DEVICE" "${THREADS:-null}"
    printf '  "benchmark_link": {"record_path": %s},\n' \
        "$([[ -n "$LINK_RECORD" ]] && printf '"%s"' "$LINK_RECORD" || printf 'null')"
    printf '  "phases": [\n'
    last=$(( ${#SELECTED_PHASES[@]} - 1 ))
    for i in "${!SELECTED_PHASES[@]}"; do
        p="${SELECTED_PHASES[$i]}"
        w="$(phase_window "$p")"
        printf '    {"id": "%s", "requested_window_seconds": %s, "status": "planned"}%s\n' \
            "$p" "$w" "$([[ $i -eq $last ]] && printf '' || printf ',')"
    done
    printf '  ],\n'
    printf '  "measurement_path": "absent; no fit driver exists for these phases"\n'
    printf '}\n'
    exit 0
fi

emit "mojotrees thermal capture plan"
emit "=============================="
emit ""
emit "script            thermal_capture.sh ${SCRIPT_VERSION}"
emit "protocol          docs/APPLE_THERMAL_ENERGY.md ${THERMAL_PROTOCOL_VERSION}"
emit "parent protocol   docs/APPLE_GPU_BENCHMARK_PROTOCOL.md ${PARENT_PROTOCOL_VERSION}"
emit "record schema     bench/apple/thermal_schema.json ${THERMAL_SCHEMA_VERSION}"
emit ""
emit "run id            ${RUN_ID}"
emit "would write       ${RECORD_PATH}"
emit "workload          ${WORKLOAD}, device ${DEVICE}, threads ${THREADS:-unset}"
emit "conditions        power ${POWER_SOURCE}, lid ${LID}, surface ${SURFACE}"
emit "sampler interval  ${INTERVAL_MS} ms"
emit "energy            $([[ $ENERGY -eq 1 ]] && echo "requested (needs root)" || echo "not requested")"
emit "linked record     ${LINK_RECORD:-none}"
for n in "${NOTES[@]:-}"; do
    [[ -n "$n" ]] && emit "note              ${n}"
done
emit ""
emit "NOTHING BELOW WAS RUN. Lines marked [ROOT] would need root. Lines marked"
emit "[NOT IMPLEMENTED] have no implementation in this repository yet."
emit ""

emit "--- 0. Before anything, by hand ---"
emit ""
emit "    Quit browsers, editors, chat clients, and file sync. Let Spotlight and"
emit "    Time Machine finish. Do not touch the machine while a phase runs."
emit "    Set the lid and surface to the declared condition: lid ${LID}, surface ${SURFACE}."
emit "    Leave the machine idle for at least 20 minutes before the cold fit."
emit "    Read the privacy section of the protocol before publishing any record."
emit ""

emit "--- 1. Identity and provenance, unprivileged ---"
emit ""
emit_cmd sysctl -n machdep.cpu.brand_string hw.model hw.memsize \
    hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.logicalcpu
emit_cmd sw_vers -productVersion
emit_cmd sw_vers -buildVersion
# uname -a would include the host name, which is commonly a person's name.
emit_cmd uname -srm
emit_cmd system_profiler SPDisplaysDataType -json
emit_cmd git -C "$REPO_ROOT" rev-parse HEAD
emit_cmd git -C "$REPO_ROOT" status --porcelain
emit ""
emit "    system_profiler SPHardwareDataType and SPSoftwareDataType are NOT run."
emit "    They carry the serial number, hardware UUID, computer name, and user"
emit "    name, and the record schema has no field for any of them."
emit ""

emit "--- 2. Idle gate, unprivileged ---"
emit ""
emit_cmd sysctl -n vm.loadavg
emit_cmd pmset -g therm
emit_cmd pmset -g batt
emit_cmd pmset -g
emit_cmd ps -Ao comm=
emit ""
emit "    Gate rules, inherited from the parent protocol: one-minute load at or"
emit "    below 0.75, no mojo/pixi/pytest/cmake/clang/Xcode process, Low Power"
emit "    Mode off, CPU_Speed_Limit = 100. Power source must be ${POWER_SOURCE}."
emit "    A failed gate stops the run. It does not downgrade it."
emit ""

emit "--- 3. Phases ---"
emit ""

for p in "${SELECTED_PHASES[@]}"; do
    idx="$(phase_index "$p")"
    w="$(phase_window "$p")"
    emit "  phase ${p}"
    emit "    ${PHASE_DESCRIPTIONS[$idx]}"
    if [[ "$w" == "0" ]]; then
        emit "    window: bounded by fits, not by time"
    else
        emit "    window: ${w} s"
    fi
    emit ""

    emit "    thermal sampling for the whole phase, unprivileged, every $((INTERVAL_MS >= 1000 ? INTERVAL_MS / 1000 : 1)) s:"
    emit_cmd pmset -g therm
    if [[ "$ENERGY" -eq 1 ]]; then
        emit ""
        emit "    machine power sampler, started before the phase and stopped after it:"
        emit_priv powermetrics --samplers cpu_power,gpu_power,thermal \
            -i "$INTERVAL_MS" -f plist
        if [[ "$PROCESS_ENERGY" -eq 1 ]]; then
            emit_priv powermetrics --samplers tasks --show-process-energy \
                -i "$INTERVAL_MS"
            emit "    (energy impact scores only; unitless, never joules, never published as power)"
        fi
    fi
    emit ""

    case "$p" in
        idle_baseline)
            emit "    workload: none. The machine is left alone for the window."
            emit "    This is what makes energy_above_idle_j meaningful in every other phase."
            ;;
        cooldown_recovery)
            emit "    workload: none. Sampling continues until the thermal state returns"
            emit "    to its pre-run value or the ${w} s ceiling is reached. Failure to"
            emit "    recover before the ceiling is a result, recorded in stopping_rule."
            ;;
        background_control)
            emit "    deliberate competing load, declared in the record. One busy core"
            emit "    is enough to move a histogram timing, and this form is stoppable"
            emit "    by pid rather than by a pattern match over every process on the"
            emit "    machine, which is why it is written this way:"
            emit_cmd sh -c 'yes > /dev/null & echo $! > /tmp/mojotrees_thermal_load.pid'
            emit "    then re-run the idle gate. It MUST fail. A gate that passes here"
            emit "    proves nothing about any other phase in this record."
            emit_cmd sh -c 'kill "$(cat /tmp/mojotrees_thermal_load.pid)"'
            emit_cmd rm -f /tmp/mojotrees_thermal_load.pid
            ;;
        cold_fit|warm_fit|repeat_series|sustained|throttle_probe)
            case "$p" in
                cold_fit)     fitspec="1 fit in a fresh process" ;;
                warm_fit)     fitspec="1 fit in the process that just did the cold fit" ;;
                repeat_series) fitspec="${REPEAT_SERIES_FITS} fits back to back in one process" ;;
                sustained)    fitspec="fits back to back for ${w} s, bucketed every ${BUCKET_SECONDS} s" ;;
                *)            fitspec="fits back to back until a thermal limit appears or ${w} s elapses" ;;
            esac
            emit "    workload: ${fitspec}, ${WORKLOAD} on ${DEVICE}."
            emit_todo "fit driver for phase ${p}"
            emit "    No driver exists. See handoffs/performance_17_thermal_energy.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_17_thermal_energy.md),"
            emit "    section \"The measurement path\", for the two candidate homes for it"
            emit "    and why neither belongs to this lane."
            emit "    Environment the driver would set before importing anything."
            emit "    PCORES stands in for sysctl -n hw.perflevel0.physicalcpu when"
            emit "    --threads was not given; the protocol matches at 1 and at that count."
            emit_cmd env "MOJOTREES_NUM_WORKERS=${THREADS:-PCORES}" \
                "OMP_NUM_THREADS=${THREADS:-PCORES}" \
                "MKL_NUM_THREADS=${THREADS:-PCORES}" \
                "VECLIB_MAXIMUM_THREADS=${THREADS:-PCORES}"
            emit "    Every fit is scored on the same held-out split. A fit whose quality"
            emit "    is not recorded makes every throughput number in this phase unquotable."
            ;;
    esac
    emit ""
done

emit "--- 4. After the run ---"
emit ""
emit_cmd pmset -g therm
emit_cmd sysctl -n vm.loadavg
emit_cmd pmset -g batt
emit ""
emit "    Then write ${RECORD_PATH} against"
emit "    bench/apple/thermal_schema.json ${THERMAL_SCHEMA_VERSION}, fill"
emit "    validity.invalid_reasons honestly, and read the record for machine"
emit "    identifiers and paths before publishing it."
emit ""
emit "--- 5. What this plan does not include ---"
emit ""
emit "    No temperature in degrees. Apple silicon exposes no public, documented"
emit "    die temperature, and this protocol does not read undocumented SMC keys."
emit "    No per-process joules. powermetrics reports the machine."
emit "    No estimated energy. Absent is recorded as absent."
emit "    No sysdiagnose, no ioreg battery dump, no nvram."
emit ""
emit "NOTHING WAS MEASURED AND NO FILE WAS WRITTEN."
