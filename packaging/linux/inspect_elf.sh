#!/usr/bin/env bash
# Inspect installed mojoboost ELF objects on the machine they have to run on.
#
#     packaging/linux/inspect_elf.sh <directory or object> [record file]
#
# Typically the installed package inside a clean container:
#
#     inspect_elf.sh "$(python -c 'import mojoboost,pathlib;print(pathlib.Path(mojoboost.__file__).parent)')"
#
# THIS SCRIPT HAS NEVER BEEN EXECUTED. Nothing it prints has been observed.
#
# Why this exists next to packaging/linux/inspect_wheel.py, which parses the
# same structures without any tools: that one reads a wheel file and can run
# anywhere, including on the macOS laptop this project is developed on. It
# cannot answer the question that only the target can answer, which is whether
# the dynamic loader on THIS machine can actually resolve everything. `ldd -r`
# can, and it needs a real loader, real system libraries, and the package
# installed. Run both. They disagree in exactly the interesting cases.
#
# This script only reads. It installs nothing, changes nothing, and touches no
# network.
set -euo pipefail

TARGET=${1:?usage: inspect_elf.sh <directory or object> [record file]}
LOG=${2:-/dev/null}

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
run() { say ""; say "\$ $*"; "$@" 2>&1 | tee -a "$LOG" || say "(exit $?)"; }

[ "$(uname -s)" = "Linux" ] || {
    echo "inspect_elf: this is a Linux tool, found $(uname -s)." >&2
    echo "Use packaging/linux/inspect_wheel.py to inspect a wheel anywhere." >&2
    exit 2
}

missing=""
for tool in readelf ldd file; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
[ -z "$missing" ] || {
    echo "inspect_elf: missing tools:$missing" >&2
    echo "Install binutils and file. On a manylinux image both are present." >&2
    exit 2
}

# --- The host, because every number below is relative to it ----------------
say "=== host ==="
run uname -a
say "glibc:"
ldd --version 2>&1 | head -1 | tee -a "$LOG"
[ -f /etc/os-release ] && (. /etc/os-release && say "os-release: $PRETTY_NAME")
if [ -n "${CONDA_PREFIX:-}" ]; then
    say ""
    say "!! CONDA_PREFIX is set ($CONDA_PREFIX)."
    say "!! Whatever this reports, it was measured next to a toolchain the"
    say "!! wheel is supposed to not need. Re-run somewhere clean."
fi

# --- Collect the objects ---------------------------------------------------
if [ -d "$TARGET" ]; then
    OBJECTS=$(find "$TARGET" -type f \( -name '*.so' -o -name '*.so.*' \) | sort)
else
    OBJECTS=$TARGET
fi
[ -n "$OBJECTS" ] || { echo "inspect_elf: no ELF objects under $TARGET" >&2; exit 1; }

say ""
say "=== objects ==="
printf '%s\n' "$OBJECTS" | tee -a "$LOG"

for so in $OBJECTS; do
    say ""
    say "================================================================"
    say "$so"
    say "================================================================"

    # Architecture and object kind. A wheel that installed on this machine and
    # holds an object for another architecture means the tag was wrong.
    run file "$so"

    # DT_NEEDED, DT_SONAME, DT_RUNPATH, DT_RPATH.
    #
    # RUNPATH must be $ORIGIN-relative. An absolute path here is the build
    # machine's pixi environment, and it is the single most common way a wheel
    # works on the machine that built it and nowhere else.
    #
    # DT_RPATH rather than DT_RUNPATH is a lesser problem worth noticing:
    # RPATH cannot be overridden with LD_LIBRARY_PATH, which makes debugging a
    # broken install harder for the user.
    run readelf -d "$so"

    # Symbol versions required from each library. The highest GLIBC_ version
    # across all objects is the wheel's real glibc floor, and the number a
    # manylinux tag has to be at or above. Reading it per library matters:
    # GLIBCXX_ and CXXABI_ come from libstdc++ and are a separate constraint
    # with a separate answer (bundle it, or require it).
    say ""
    say "-- version requirements (readelf -V, Version needs section)"
    readelf -V "$so" 2>&1 | sed -n '/Version needs/,/^$/p' | tee -a "$LOG"
    say ""
    say "-- highest GLIBC_ requirement"
    readelf -V "$so" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' \
        | sort -uV | tail -1 | tee -a "$LOG"

    # Resolution on this machine, including unresolved symbols. This is the
    # part inspect_wheel.py cannot do: it needs this loader and these
    # libraries. An unresolved symbol here is an ImportError waiting to happen.
    say ""
    say "-- ldd -r (resolution on this host)"
    ldd -r "$so" 2>&1 | tee -a "$LOG"

    # Anything absolute in the strings that points into a build machine.
    say ""
    say "-- build host paths in strings"
    if grep -aoE '/(Users|home|root)/[A-Za-z0-9_.-]+|/\.pixi/|/opt/hostedtoolcache/' "$so" \
        | sort -u | tee -a "$LOG" | grep -q .; then
        say "!! build host paths present. The artifact leaks where it was made,"
        say "!! and if they are in RUNPATH it also does not work here by luck."
    else
        say "(none)"
    fi
done

say ""
say "=== done ==="
say "This is a recording, not a verdict. Nothing above becomes a validated"
say "status until it is written into a record file and the matching row in"
say "packaging/matrix/platform_matrix.toml is changed to point at it."
