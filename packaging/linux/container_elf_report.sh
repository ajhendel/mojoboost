#!/usr/bin/env bash
# Install a wheel into a throwaway venv inside a container and report what its
# ELF objects declare and whether this machine's loader can satisfy them.
#
#     packaging/linux/container_elf_report.sh <wheel> <output file>
#
# Runs INSIDE the container, not on the host. The host side is one docker run;
# see .github/workflows/release-linux.yml and the handoff.
#
# THIS SCRIPT HAS NEVER BEEN EXECUTED.
#
# It exists next to packaging/matrix/smoke/clean_install_linux.sh rather than
# inside it because the two answer different questions and are owned by
# different lanes. The fixture answers "does this wheel work on a clean
# machine": install, import, fit, predict, device behavior. This answers "what
# does it require of the machine, and does the machine have it": full version
# requirements per library, and `ldd -r`, which resolves symbols rather than
# just libraries. A wheel can pass the fixture on a modern distribution and
# still be mislabeled; only the numbers here show it.
#
# Reads and installs into a temporary directory. Changes nothing else.
set -euo pipefail

WHEEL=${1:?usage: container_elf_report.sh <wheel> <output file>}
OUT=${2:?usage: container_elf_report.sh <wheel> <output file>}
HERE=$(cd "$(dirname "$0")" && pwd)

PY=${PYTHON:-python3.14}
command -v "$PY" >/dev/null 2>&1 || {
    echo "container_elf_report: no $PY on PATH." >&2
    echo "Set PYTHON to a CPython 3.14 interpreter. In a manylinux image that" >&2
    echo "is a path under /opt/python; in a plain distribution image it has to" >&2
    echo "be installed first." >&2
    exit 2
}

if [ -n "${CONDA_PREFIX:-}" ] || command -v mojo >/dev/null 2>&1; then
    echo "container_elf_report: refusing to report from a machine that has the" >&2
    echo "toolchain on it. The numbers would be about this environment rather" >&2
    echo "than about the wheel." >&2
    exit 2
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --no-index --no-cache-dir "$WHEEL"

PKG=$("$WORK/venv/bin/python" -c \
    'import mojoboost, pathlib; print(pathlib.Path(mojoboost.__file__).parent)')

"$HERE/inspect_elf.sh" "$PKG" "$OUT"

echo "elf report written to $OUT"
echo "It is a recording of one machine. It becomes evidence only when it is"
echo "attached to a record that says which image and which wheel produced it."
