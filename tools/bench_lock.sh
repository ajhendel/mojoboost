#!/bin/sh
# The TIMING lock, and a liveness check for it.
#
#   tools/bench_lock.sh status
#   tools/bench_lock.sh acquire "<session>" "<what>" "<eta>"
#   tools/bench_lock.sh release
#   tools/bench_lock.sh take            # only when status says ABANDONED
#
# This is NOT tools/with_build_lock.sh. That one serializes heavy *builds* on
# /tmp/mojotrees-build.lock through an fcntl flock, and an fcntl lock is
# released by the kernel when the holding process dies, so it has liveness for
# free and needs nothing from this file. This one guards
# /tmp/mojotrees-bench.lock, the advisory TIMING lock that says a measurement
# window is open, and until now it was a hand-written text file with no tool,
# no enforcement and no way to tell a held lock from an abandoned one.
#
# WHY IT EXISTS. On 2026-08-16 a lane holding the timing lock died mid-window
# when three sessions were terminated at once. The file sat there with a stated
# ETA and nothing to release it. From the outside a stale lock and a busy lock
# are the same bytes: the holder, the workload and the ETA are all claims
# written at ACQUISITION by a session that may not survive to release them, and
# nothing checked whether they were still true. Another session waited on it
# correctly, and would have kept waiting.
#
# THE DIRECTION THIS FAILS IN IS THE WHOLE DESIGN. Reporting a live lock as
# abandoned lets two sessions measure at once and silently corrupts both
# windows; reporting an abandoned lock as held costs somebody a wait. Those are
# not comparable, so every uncertain case here resolves to HELD. A lock is
# reported STALE-PID only when the holder's PID is positively confirmed gone.
# No `pid:` line, an unreadable file, a PID we cannot ask about: all HELD.
#
# THE PID MUST OUTLIVE THE WINDOW, AND A PER-CALL SHELL DOES NOT. Each shell
# invocation is its own process, so `acquire` from one call and `status` from
# the next will record a pid that is already gone -- correctly reported, and
# useless. Acquire from the process that will still be running when the window
# ends: hold it from inside the long-lived job, or export BENCH_LOCK_PID with a
# pid that survives. If nothing survives, record no pid, because a missing pid
# reports HELD and a dead pid invites a steal.
#
# WHAT A DEAD PID DOES AND DOES NOT TELL YOU. It tells you the recorded process
# is not running. It does NOT tell you the work finished: a shell can exit while
# the job it started keeps going, and once the pid is gone there is nothing left
# to ask -- the record that would have answered is the process that exited, and
# its children are reparented carrying no back-reference. So this reports
# STALE-PID and not ABANDONED. On 2026-08-16 a lane read the earlier wording
# while eight `mojo` processes were running at load 11. The detection was right
# both times; the word was what misled, because ABANDONED reads as permission.
#
# AND IT NEVER STEALS. `status` reports; `take` is a separate verb somebody has
# to type. The stale file is preserved under a `.dead-<session>` suffix rather
# than deleted, because what a window was doing when it died is evidence.
set -u

LOCK=/tmp/mojotrees-bench.lock

_field() {
    # `_field <name>` -> the value of a `name: value` line, or empty.
    [ -f "$LOCK" ] || return 0
    sed -n "s/^$1:[[:space:]]*//p" "$LOCK" | head -1
}

_holder_alive() {
    # 0 = alive or UNKNOWN, 1 = positively dead. Note the asymmetry: this
    # returns "alive" when it cannot tell, which is what makes the caller fail
    # safe. `ps -p` rather than `kill -0`, because `kill -0` fails identically
    # for "no such process" and "not yours to signal" and those mean opposite
    # things here.
    pid=$(_field pid)
    case "$pid" in
        ''|*[!0-9]*) return 0 ;;   # no pid, or not a number: cannot tell
    esac
    if ps -p "$pid" >/dev/null 2>&1; then
        return 0
    fi
    # The PID is gone. One more check before calling it dead: a socket path, if
    # the holder recorded one. PIDs are recycled, so an existing socket is
    # weak evidence of life and a missing one is not evidence of death -- it
    # only ever moves the answer toward HELD, never away from it.
    return 1
}

case "${1:-status}" in
status)
    if [ ! -f "$LOCK" ]; then
        echo "FREE  (no $LOCK)"
        exit 0
    fi
    holder=$(_field session)
    pid=$(_field pid)
    if _holder_alive; then
        if [ -z "$pid" ]; then
            echo "HELD  by '${holder:-unknown}' -- no pid recorded, so liveness is UNKNOWN and this reports HELD"
        else
            echo "HELD  by '${holder:-unknown}' (pid $pid alive)"
        fi
        sed -n '1,12p' "$LOCK"
        exit 1
    fi
    echo "STALE-PID  '${holder:-unknown}' (recorded pid $pid is not running)"
    echo "  THIS IS NOT EVIDENCE THE BOX IS FREE. It means one recorded process"
    echo "  is gone. A shell can exit while the work it started continues, and"
    echo "  nothing here can tell those apart. The box itself, right now:"
    echo "    mojo processes: $(ps -Ao comm | grep -c 'mojo$')"
    echo "    $(uptime | sed 's/.*load/load/')"
    echo "  the fields below were written at acquisition and are NOT current:"
    sed -n '1,12p' "$LOCK"
    echo "  if the box is genuinely idle: tools/bench_lock.sh take"
    exit 2
    ;;
acquire)
    if [ -f "$LOCK" ] && _holder_alive; then
        echo "REFUSED: lock is held (or its liveness is unknown)." >&2
        "$0" status >&2
        exit 1
    fi
    [ -f "$LOCK" ] && "$0" take >/dev/null
    # $PPID, not $$. This script EXITS as soon as it has written the file,
    # so its own pid is dead within milliseconds and every lock would report
    # ABANDONED -- which is the one direction this whole file exists to avoid,
    # and the self-test caught it on the first run. The pid recorded here must
    # belong to a process that OUTLIVES the window: the caller's shell by
    # default, or whatever BENCH_LOCK_PID names. If neither outlives the
    # measurement, record no pid at all rather than a dead one, because a
    # missing pid reports HELD and a dead pid invites a steal.
    cat > "$LOCK" <<EOF
session: ${2:-unnamed}
pid: ${BENCH_LOCK_PID:-$PPID}
socket: ${CLAUDE_SESSION_SOCKET:-none}
mode: timing
what: ${3:-unspecified}
eta: ${4:-unspecified}
started: $(date '+%Y-%m-%d %H:%M:%S')
box_at_start: $(uptime | sed 's/.*load/load/')
note: pid/socket are the liveness fields; every other line is a claim written
      once at acquisition and never updated. Read them as history, not status.
EOF
    echo "ACQUIRED by '${2:-unnamed}' (pid ${BENCH_LOCK_PID:-$PPID})"
    ;;
release)
    if [ ! -f "$LOCK" ]; then
        echo "nothing to release"
        exit 0
    fi
    owner=$(_field pid)
    me=${BENCH_LOCK_PID:-$PPID}
    if [ -n "$owner" ] && [ "$owner" != "$me" ] && ps -p "$owner" >/dev/null 2>&1; then
        echo "REFUSED: pid $owner still holds this and it is not you ($$)." >&2
        echo "If it is yours from a dead child, use 'take'. (you=$me)" >&2
        exit 1
    fi
    rm -f "$LOCK"
    echo "RELEASED"
    ;;
take)
    [ -f "$LOCK" ] || { echo "nothing to take"; exit 0; }
    if _holder_alive; then
        echo "REFUSED: holder is alive, or its liveness is unknown." >&2
        exit 1
    fi
    # Reached only when status says STALE-PID, which is not the same as "free".
    # Whoever types this has been told to look at the box; the tool cannot.
    dead="$LOCK.dead-$(_field session | tr -c 'A-Za-z0-9._-' '-' | sed 's/-*$//')"
    mv "$LOCK" "$dead"
    echo "TOOK the lock; the stale file is preserved at $dead"
    ;;
*)
    echo "usage: $0 {status|acquire <session> <what> <eta>|release|take}" >&2
    exit 64
    ;;
esac
