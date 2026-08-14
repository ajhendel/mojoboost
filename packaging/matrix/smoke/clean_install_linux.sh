#!/usr/bin/env bash
# Clean-install smoke test for a Linux mojoboost wheel, x86_64 or aarch64.
#
# NOT WIRED INTO ANYTHING AND NOT EXECUTED, and it currently has nothing to
# test: no Linux wheel builder exists. packaging/build_wheel.sh is macOS only,
# because install_name_tool and codesign have no Linux counterpart and the ELF
# equivalent is a different program rather than a flag. See
# handoffs/task18_platform.md for what building one would take.
#
# This fixture exists now so the acceptance criteria are settled before anyone
# writes that builder. A Linux wheel that passes this is shippable; one that
# does not, is not, whatever it does on the machine that built it.
#
#   packaging/matrix/smoke/clean_install_linux.sh <wheel> [record file]
#
# Intended host: a clean manylinux_2_28 container of the wheel's architecture,
# or any glibc 2.28 machine with no Mojo, no MAX, and no conda environment. The
# container is the better test because it has the oldest glibc the tag permits,
# which is the constraint the tag is a promise about:
#
#   docker run --rm -v "$PWD:/io" -w /io \
#       quay.io/pypa/manylinux_2_28_x86_64 \
#       packaging/matrix/smoke/clean_install_linux.sh dist/<wheel>
#
# A pass on the developer's own Ubuntu proves the wheel works on that Ubuntu.
# It does not test the tag.
set -euo pipefail

WHEEL=${1:?usage: clean_install_linux.sh <wheel> [record file]}
LOG=${2:-/dev/null}
WHEEL=$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")
HERE=$(cd "$(dirname "$0")" && pwd)

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
run() { say "\$ $*"; "$@" 2>&1 | tee -a "$LOG"; }

# --- 1. The environment must be hostile ------------------------------------
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "refusing: CONDA_PREFIX is set ($CONDA_PREFIX)." >&2
    exit 2
fi
if command -v mojo >/dev/null 2>&1; then
    echo "refusing: mojo is on PATH ($(command -v mojo))." >&2
    exit 2
fi
if [ "$(uname -s)" != "Linux" ]; then
    echo "refusing: this fixture is for Linux, found $(uname -s)." >&2
    exit 2
fi

PY=${PYTHON:-python3.14}
command -v "$PY" >/dev/null 2>&1 || {
    echo "refusing: no $PY on PATH. Set PYTHON to a 3.14 interpreter." >&2
    exit 2
}

# --- 2. Record the host, including the glibc the tag claims to support ------
# The tag is a promise about this number. Record it every time, because a pass
# on glibc 2.39 says nothing about the manylinux_2_28 claim.
say "=== host ==="
run uname -a
run "$PY" -c 'import sys, sysconfig; print(sys.version); print(sysconfig.get_platform()); print(sys.executable)'
say "glibc:"
ldd --version 2>&1 | head -1 | tee -a "$LOG"
[ -f /etc/os-release ] && (. /etc/os-release && say "os-release: $PRETTY_NAME")
say "=== wheel ==="
run sha256sum "$WHEEL"

# --- 3. Install into a throwaway venv, offline -----------------------------
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/../../smoke_test.py" "$HERE/probe_platform.py" "$WORK/"
"$PY" -m venv "$WORK/bare"
say "=== install (bare, offline) ==="
run "$WORK/bare/bin/pip" install --no-index --no-cache-dir "$WHEEL"

cd "$WORK"
say "=== import provenance and platform ==="
run ./bare/bin/python probe_platform.py

# --- 4. What the ELF objects actually require ------------------------------
# The Linux analogue of the macOS install_name and rpath checks, and the reason
# a Linux wheel is not just "the macOS builder with different flags":
#
#   * every bundled .so must be found through an $ORIGIN-relative RUNPATH, not
#     an absolute path into the build machine's pixi environment
#   * the highest GLIBC_ version any object references is the real floor, and
#     it must be at or below the one the platform tag claims
#
# Both are recorded here rather than asserted, because the first Linux wheel
# that exists is the one that establishes what these values are.
PKG=$(./bare/bin/python -c 'import mojoboost, pathlib; print(pathlib.Path(mojoboost.__file__).parent)')
say "=== ELF inspection: $PKG ==="
for so in "$PKG"/*.so "$PKG"/.libs/*.so* "$PKG"/mojoboost.libs/*.so*; do
    [ -e "$so" ] || continue
    say "--- $so"
    run readelf -d "$so" | grep -Ei 'runpath|rpath|needed' || true
    say "highest glibc symbol version referenced:"
    readelf -V "$so" 2>/dev/null \
        | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 | tee -a "$LOG"
done

# --- 5. Import and use it --------------------------------------------------
say "=== smoke test (stdlib fallback, no numpy) ==="
run ./bare/bin/python smoke_test.py

# --- 6. Device behavior ----------------------------------------------------
# On a Linux host with no accelerator, and a wheel built on a host with no
# accelerator, device="gpu" must raise. On a wheel built where a GPU was
# present, availability was resolved at compile time, so the failure moves to
# where the device is opened. Record which of the two happened; that difference
# is a property of the build machine, not of this machine.
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

say "=== done ==="
say "This output is not a validation until it is pasted into a record and the"
say "status in packaging/matrix/platform_matrix.toml is changed to match it."
