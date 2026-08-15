#!/usr/bin/env bash
#
# Run the Mojo test suite: one package build, then a bounded pool of test
# processes.
#
# What this replaces.  `pixi run test` used to be sixty `mojo run` commands
# chained with `&&`.  That is serial, so the suite cost the sum of sixty
# compiles rather than the slowest one, and it is fail-fast, so a failure in
# the first file hid the state of the other fifty-nine.  It also passed
# `-I src`, which recompiles all sixty modules of the package inside every
# one of the sixty processes.
#
# What this does instead.  `mojo precompile` builds the package once into
# `build/mojotrees.mojopkg`, every test runs against `-I build`, and the
# files run concurrently with failures collected rather than aborting the
# run.  Precompiling is also a stronger check than the suite: it elaborates
# every module in `src/mojotrees/`, including the ones no test imports.
#
# Usage:
#   tools/run_tests.sh [all|cpu|gpu|list] [name ...]
#
#     all   every test file (the default)
#     cpu   everything that does not need an accelerator
#     gpu   the accelerator subset, for a runner that has one
#     list  print the selection and exit, running nothing
#
# Naming files after the mode runs exactly those, which is the focused
# command CONTRIBUTING.md asks for during implementation, with the package
# build in front of it:
#
#   tools/run_tests.sh cpu test_binning test_sparse
#
# Environment:
#   MOJOTREES_TEST_JOBS   concurrent test processes.  Defaults to two fewer
#                         than the CPU count, so a suite run leaves a
#                         development machine usable; CONTRIBUTING.md asks
#                         for that and this is the knob for it.  CI sets it
#                         to the full count.
#   MOJOTREES_TEST_PKG    set to 0 to compile from `src/` instead of the
#                         package, which is slower but skips the build step
#
# This parallelizes *within* one suite run.  `tools/with_build_lock.sh` is
# the opposite concern, serializing whole runs against other sessions in the
# same checkout, and the two compose: run this under that lock.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

# `mojo` is on PATH inside a pixi environment and nowhere else, so a pixi
# task reaches it and a bare shell does not.  Re-enter the environment
# rather than fail on `mojo: command not found`.
if ! command -v mojo >/dev/null 2>&1; then
  if command -v pixi >/dev/null 2>&1 && [ -z "${MOJOTREES_TEST_REEXEC:-}" ]; then
    export MOJOTREES_TEST_REEXEC=1
    exec pixi run bash "$ROOT/tools/run_tests.sh" "$@"
  fi
  echo "mojo is not on PATH; run this as 'pixi run test' or inside 'pixi shell'" >&2
  exit 1
fi

MODE="${1:-all}"
case "$MODE" in
  all|cpu|gpu|list) shift || true ;;
  *) echo "usage: tools/run_tests.sh [all|cpu|gpu|list] [name ...]" >&2
     exit 2 ;;
esac
NAMED=("$@")

if command -v nproc >/dev/null 2>&1; then
  NCPU=$(nproc)
else
  NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi
DEFAULT_JOBS=$(( NCPU > 2 ? NCPU - 2 : 1 ))
JOBS="${MOJOTREES_TEST_JOBS:-$DEFAULT_JOBS}"
USE_PKG="${MOJOTREES_TEST_PKG:-1}"

# Tests that instantiate accelerator kernels.  CPU-only CI must not compile
# these: an Apple-silicon or CUDA-less runner reports no GPU architecture at
# compile time and the kernel fails to build rather than skipping at run
# time.  Everything not listed here is CPU-safe and runs in every mode
# except `gpu`.
GPU_ONLY="
test_apple_gpu_policy
test_backend_equivalence
test_device
test_gpu_active_rows
test_gpu_objectives
test_gpu_objectives_native
test_gpu_portability
test_gpu_predict
test_gpu_runtime
test_gpu_sparse
test_gpu_split_search
test_gpu_strategies
test_gpu_tiling
test_gpu_training
test_gpu_vendor_policy
test_hybrid_replica
"

# CPU-safe, but exercises a GPU path when one exists, so a GPU runner should
# see it too.
GPU_ALSO="test_interaction"

# Tests that reach past the package into another source tree.
extra_includes() {
  case "$1" in
    test_capi) echo "-I capi" ;;
    test_cli)  echo "-I cli" ;;
    *)         echo "" ;;
  esac
}

in_list() {
  local needle="$1" hay="$2"
  [[ " $(echo $hay) " == *" $needle "* ]]
}

# Discovery is a glob, not a hand-maintained list.  The `&&` chains named
# every file explicitly and drifted: `test_gpu_split_policy.mojo` was in the
# tree, passing, and named by no task at all.
SELECTED=()
if [ "${#NAMED[@]}" -gt 0 ]; then
  for name in "${NAMED[@]}"; do
    name="${name%.mojo}"
    name="${name#tests/}"
    if [ ! -e "tests/$name.mojo" ]; then
      echo "no such test: tests/$name.mojo" >&2
      exit 2
    fi
    SELECTED+=("$name")
  done
else
  for path in tests/test_*.mojo; do
    [ -e "$path" ] || continue
    name="$(basename "$path" .mojo)"
    case "$MODE" in
      all|list) SELECTED+=("$name") ;;
      cpu) in_list "$name" "$GPU_ONLY" || SELECTED+=("$name") ;;
      gpu) if in_list "$name" "$GPU_ONLY" || in_list "$name" "$GPU_ALSO"
           then SELECTED+=("$name")
           fi ;;
    esac
  done
fi

if [ "${#SELECTED[@]}" -eq 0 ]; then
  echo "no test files matched mode '$MODE'" >&2
  exit 1
fi

if [ "$MODE" = "list" ]; then
  printf '%s\n' "${SELECTED[@]}"
  exit 0
fi

if [ "$USE_PKG" = "1" ]; then
  echo "building build/mojotrees.mojopkg"
  mkdir -p build
  if ! mojo precompile -I src src/mojotrees -o build/mojotrees.mojopkg; then
    echo "package build failed; the suite cannot run against a package it" >&2
    echo "could not build.  Re-run with MOJOTREES_TEST_PKG=0 to compile" >&2
    echo "each test from src/ instead." >&2
    exit 1
  fi
  PKG_INCLUDE="-I build"
else
  PKG_INCLUDE="-I src"
fi

RESULTS="$(mktemp -d)"
trap 'rm -rf "$RESULTS"' EXIT

echo "running ${#SELECTED[@]} test files, mode=$MODE, jobs=$JOBS"

run_one() {
  local name="$1"
  local log="$RESULTS/$name.log"
  local start=$SECONDS
  if mojo run $PKG_INCLUDE -I tests $(extra_includes "$name") \
        "tests/$name.mojo" >"$log" 2>&1; then
    echo "  ok   $name ($((SECONDS - start))s)"
  else
    echo "fail" >"$RESULTS/$name.failed"
    echo "  FAIL $name ($((SECONDS - start))s)"
  fi
}
export -f run_one extra_includes
export RESULTS PKG_INCLUDE ROOT

printf '%s\n' "${SELECTED[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

FAILED=()
for name in "${SELECTED[@]}"; do
  [ -e "$RESULTS/$name.failed" ] && FAILED+=("$name")
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "all ${#SELECTED[@]} test files passed"
  exit 0
fi

# Every failure, not just the first one, which is the point of not chaining
# these with `&&`.
echo "${#FAILED[@]} of ${#SELECTED[@]} test files failed:"
for name in "${FAILED[@]}"; do
  echo
  echo "=== $name ==="
  tail -40 "$RESULTS/$name.log"
done
exit 1
