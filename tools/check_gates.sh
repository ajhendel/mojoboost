#!/usr/bin/env bash
#
# Run the cheap gates: the ones that read the repository and build nothing.
#
# Every gate here is standard-library Python over files already in the tree.
# None of them compiles Mojo, builds the extension module, starts a benchmark,
# or touches the network, which is what makes it reasonable to run the whole
# set before a commit and on a bare CI runner in seconds.
#
# Every gate runs even when an earlier one fails, and the failures are
# reported together at the end. Stopping at the first one hides the state of
# the rest, which is the same reason `tools/run_tests.sh` collects failures
# instead of chaining suites with `&&`.
#
# Usage:
#   tools/check_gates.sh              # every cheap gate
#   tools/check_gates.sh api pixi     # only the named ones
#
# Names: api, parity, pixi, connectivity, integration
#
# Exit status is 0 when every selected gate passes and 1 otherwise.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "tools/check_gates.sh: $PY is not on PATH" >&2
  exit 1
fi

# gate name -> command. Keep this table and the pre-commit hook's file map
# (tools/hooks/pre-commit) agreeing about what owns what.
run_gate() {
  case "$1" in
    api)          "$PY" tools/api_snapshot.py --check ;;
    parity)       "$PY" tools/check_parity.py ;;
    pixi)         "$PY" tools/check_pixi_tasks.py ;;
    connectivity) "$PY" tools/connectivity_audit.py ;;
    # --strict, so an unrecorded disconnection FAILS here rather than being
    # printed and passed over. Without it this gate exits 0 while printing
    # the GAP it found, which makes the pre-commit hook report a problem and
    # allow the commit anyway -- the exact report-but-do-not-fail shape this
    # repository has now been caught by four times. CI runs it with the same
    # flag (.github/workflows/ci.yml, connectivity-audit).
    integration)  "$PY" tools/audit_integration.py --strict ;;
    *)            echo "unknown gate: $1" >&2; return 2 ;;
  esac
}

ALL="api parity pixi connectivity integration"
SELECTED="${*:-$ALL}"

failed=""
for gate in $SELECTED; do
  echo "== $gate =="
  if run_gate "$gate"; then
    :
  else
    failed="$failed $gate"
  fi
  echo
done

if [ -n "$failed" ]; then
  echo "FAILED gates:$failed" >&2
  echo "Each one prints what drifted and which file to regenerate." >&2
  exit 1
fi

echo "all gates passed:$( for g in $SELECTED; do printf ' %s' "$g"; done )"
