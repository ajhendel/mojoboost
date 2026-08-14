#!/usr/bin/env bash
# Clean-install smoke test for a macOS arm64 mojoboost wheel.
#
# NOT WIRED INTO ANYTHING AND NOT EXECUTED. This is a fixture: it is the
# procedure a release run should follow, written down so the procedure is
# reviewable before it is trusted. Wiring it into pixi and into a release
# workflow is specified in handoffs/task18_platform.md.
#
#   packaging/matrix/smoke/clean_install_macos.sh <wheel> [record file]
#
# What makes this different from packaging/test_wheel.sh, which already builds
# a wheel and tests it in two venvs: that script runs inside the pixi
# environment, with the Mojo toolchain on PATH and CONDA_PREFIX pointing at the
# libraries the extension links. It proves the wheel works. It cannot prove the
# wheel is self-contained, because the thing the wheel must not need is sitting
# right there. This script refuses to run in that environment, which is most of
# the value in it.
#
# Nothing here is a claim. The output is evidence, and evidence counts only
# once it is pasted into a record.
set -euo pipefail

WHEEL=${1:?usage: clean_install_macos.sh <wheel> [record file]}
LOG=${2:-/dev/null}
WHEEL=$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")
HERE=$(cd "$(dirname "$0")" && pwd)

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
run() { say "\$ $*"; "$@" 2>&1 | tee -a "$LOG"; }

# --- 1. The environment must be hostile ------------------------------------
# A pass in an environment that has the toolchain proves nothing about a user
# who does not have it.
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "refusing: CONDA_PREFIX is set ($CONDA_PREFIX)." >&2
    echo "Run from a plain shell, not from inside pixi run or pixi shell." >&2
    exit 2
fi
if command -v mojo >/dev/null 2>&1; then
    echo "refusing: mojo is on PATH ($(command -v mojo))." >&2
    echo "The point of this test is a machine without the toolchain." >&2
    exit 2
fi
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "refusing: this fixture is for macOS arm64, found $(uname -s) $(uname -m)." >&2
    exit 2
fi

PY=${PYTHON:-python3.14}
command -v "$PY" >/dev/null 2>&1 || {
    echo "refusing: no $PY on PATH. Set PYTHON to a 3.14 interpreter that did" >&2
    echo "not come from this repository's pixi environment." >&2
    exit 2
}

# --- 2. Record the host, before anything is installed ----------------------
say "=== host ==="
run sw_vers
run uname -a
run "$PY" -c 'import sys, sysconfig; print(sys.version); print(sysconfig.get_platform()); print(sys.executable)'
say "=== wheel ==="
run shasum -a 256 "$WHEEL"

# --- 3. Install into a throwaway venv, offline -----------------------------
# --no-index is deliberate. If the wheel silently needs numpy, or anything else
# it does not declare, this is where that shows up, rather than pip quietly
# fetching it and the gap shipping.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/../../smoke_test.py" "$HERE/probe_platform.py" "$WORK/"
"$PY" -m venv "$WORK/bare"
say "=== install (bare, offline) ==="
run "$WORK/bare/bin/pip" install --no-index --no-cache-dir "$WHEEL"
run "$WORK/bare/bin/pip" list

# Everything below runs from the temp directory, so an accidental import of a
# source checkout cannot make this pass. probe_platform.py and smoke_test.py
# both report where mojoboost was imported from; an import outside
# site-packages invalidates the whole run.
cd "$WORK"

say "=== import provenance and platform ==="
run ./bare/bin/python probe_platform.py
say "=== smoke test (stdlib fallback, no numpy) ==="
run ./bare/bin/python smoke_test.py

# --- 4. The failure behaviors, which are part of the contract --------------
# device="gpu" must raise rather than fall back silently. On a Mac with a GPU
# and a Metal toolchain in the build it should train; where it raises, the
# exact message is the interesting part, so it is recorded rather than
# summarized.
say "=== device selection ==="
./bare/bin/python - 2>&1 <<'PY' | tee -a "$LOG"
from mojoboost import MojoBoostRegressor

X = [[i / 20.0, (i % 5) / 5.0] for i in range(20)]
y = [3.0 * r[0] + r[1] for r in X]

m = MojoBoostRegressor(n_estimators=5, min_data_in_leaf=2, device="cpu").fit(X, y)
print("cpu device_:", m.device_)

try:
    g = MojoBoostRegressor(n_estimators=5, min_data_in_leaf=2, device="gpu").fit(X, y)
    print("gpu device_:", g.device_)
except Exception as exc:
    print(f"gpu raised: {type(exc).__name__}: {exc}")
PY

say "=== device selection, GPU pinned off ==="
MOJOBOOST_DISABLE_GPU=1 ./bare/bin/python - 2>&1 <<'PY' | tee -a "$LOG"
from mojoboost import MojoBoostRegressor

X = [[i / 20.0, (i % 5) / 5.0] for i in range(20)]
y = [3.0 * r[0] + r[1] for r in X]
try:
    MojoBoostRegressor(n_estimators=5, min_data_in_leaf=2, device="gpu").fit(X, y)
    print("FAIL: device=gpu succeeded with MOJOBOOST_DISABLE_GPU=1")
except Exception as exc:
    print(f"ok, gpu raised: {type(exc).__name__}: {exc}")
PY

# --- 5. The full install, with the optional dependencies -------------------
say "=== install (full) ==="
"$PY" -m venv full
run ./full/bin/pip install --no-cache-dir "$WHEEL" numpy pytest
run ./full/bin/python smoke_test.py

say "=== done ==="
say "This output is not a validation until it is pasted into a record and the"
say "status in packaging/matrix/platform_matrix.toml is changed to match it."
