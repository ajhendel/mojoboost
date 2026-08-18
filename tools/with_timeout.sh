#!/usr/bin/env bash
# Run a command under a wall-clock cap, portably.
#
# WHY THIS EXISTS
#
# On 2026-08-18, `pixi run gpu-validate` deadlocked inside `train_gpu` on an
# NVIDIA RTX 5090. Not slowly: the main thread and all ten worker threads sat
# in `futex_wait_queue` and the process used ONE CPU tick in a five-second
# window, with the GPU at 0%. It never returned.
#
# It happened three times in a row, always stopping after the same printed
# line, and each time it was diagnosed as something else (a slow CPU oracle,
# then a large shape, then a busy machine) because a hung process and a slow
# process look identical from the outside. Hours went into that. The suite had
# already grown a timeout that morning for exactly this reason; the eighteen
# `mojo run` bench tasks in pixi.toml had none, so the one that deadlocked was
# the one with no upper bound.
#
# A benchmark that can hang is a benchmark that cannot be run unattended, and
# unattended on leased hardware billed by the hour is precisely how these get
# run.
#
# WHY NOT JUST `timeout`
#
# `timeout` is coreutils. Linux CI has it; macOS does not ship it at all, and
# Homebrew coreutils installs it as `gtimeout`. Since every development machine
# here is macOS and CI is Linux, a bare `timeout` would protect CI and silently
# do nothing where the work actually happens. So: probe for both, then fall
# back to a POSIX job-control implementation.
#
# USAGE
#
#   tools/with_timeout.sh <seconds> <command> [args...]
#
# Exit codes match coreutils `timeout`: 124 means the deadline fired. Anything
# else is the command's own status. A `<seconds>` of 0 disables the cap and
# runs the command directly, which is the escape hatch for a benchmark that
# genuinely needs longer.

set -u

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <seconds> <command> [args...]" >&2
  exit 2
fi

secs="$1"
shift

# `auto` reads MOJOTREES_BENCH_TIMEOUT, defaulting to 1800s. The pixi task
# strings pass `auto` rather than `${MOJOTREES_BENCH_TIMEOUT:-1800}` because
# pixi runs tasks through its own shell, and depending on that shell's
# parameter-expansion support to set a safety limit would be a bad bet: if the
# expansion silently produced an empty string, the cap would vanish and the
# task would look protected while being unprotected. Resolving it here means
# one implementation, in a shell whose behavior is known.
if [ "$secs" = "auto" ]; then
  secs="${MOJOTREES_BENCH_TIMEOUT:-1800}"
fi

case "$secs" in
  ''|*[!0-9]*)
    echo "$0: timeout must be a whole number of seconds, 0, or 'auto'" >&2
    echo "$0: got: '$secs'" >&2
    exit 2 ;;
esac

if [ "$secs" = "0" ]; then
  exec "$@"
fi

if command -v timeout >/dev/null 2>&1; then
  timeout "$secs" "$@"
  rc=$?
elif command -v gtimeout >/dev/null 2>&1; then
  gtimeout "$secs" "$@"
  rc=$?
else
  # No coreutils. Run in the background and poll for the deadline.
  #
  # SIGKILL rather than SIGTERM is deliberate: the case this exists for is a
  # futex deadlock, and a process wedged on a futex does not necessarily run a
  # signal handler. A polite signal that the target cannot act on would leave
  # the hang in place and report a clean exit, which is worse than no timeout
  # because it looks like success.
  "$@" &
  pid=$!
  waited=0
  rc=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$secs" ]; then
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      rc=124
      break
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  if [ "$rc" -ne 124 ]; then
    wait "$pid"
    rc=$?
  fi
fi

if [ "$rc" -eq 124 ]; then
  echo "" >&2
  echo "TIMEOUT: killed after ${secs}s: $*" >&2
  echo "  The command did not finish. Its output above is INCOMPLETE and any" >&2
  echo "  timing it printed is not a measurement of a completed run." >&2
  echo "  If it was merely slow, re-run with MOJOTREES_BENCH_TIMEOUT raised," >&2
  echo "  or 0 to disable. If it was wedged, check whether the process was" >&2
  echo "  burning CPU or parked: 'ps -o stat,wchan,pcpu -p <pid>' answers it," >&2
  echo "  and a futex_wait_queue with ~0% CPU is a deadlock, not slowness." >&2
fi

exit "$rc"
