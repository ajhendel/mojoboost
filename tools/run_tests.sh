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
#
# Where the time goes.  Measured, not assumed: almost all of a suite run is
# `mojo` compiling, not tests executing.  `tests/test_tree_parameters_extra.mojo`
# runs its 47 tests in 0.061 ms against roughly three seconds of wall clock
# on an M4 and seventeen in CI.  `TestSuite` itself is not slow; a
# compute-heavy function costs the same called directly, called through a
# function pointer, and called by `TestSuite` (231.9 / 220.4 / 221.2 ms).
# So the levers that matter are compile-side, and the two that work are
# already here: one `mojo precompile` of the package instead of sixty, and
# concurrent files.  Lowering the optimization level is NOT a lever, it is
# worse on both counts (`-O0` compiled the file above in 4.68 s against
# 3.09 s at the default `-O3`, and ran the tests ten times slower).  The
# one lever left is the on-disk compile cache under
# `$MODULAR_HOME/cache/.mojo_cache`, worth 3.10 s cold against 0.57 s warm
# on that same file; it survives locally and CI throws it away every run.
# docs/design/TEST_HARNESS_COST.md has the numbers and the CI recipe.

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
# time.
#
# This list is BELT, not braces.  It was hand-maintained, and in one round two
# new accelerator tests were added without it and were silently classified
# CPU-safe, which is a break that only a CPU-only runner can see and that no
# amount of local testing on an Apple machine reproduces.  `gpu_by_content`
# below now derives the same answer from the file, so a test that reaches the
# accelerator is GPU-only whether or not anyone remembered to name it here.
# Keeping the explicit list as well costs nothing and documents intent.
GPU_ONLY="
test_apple_gpu_policy
test_backend_equivalence
test_device
test_gpu_active_rows
test_gpu_fma_consistency
test_gpu_kernel_family
test_gpu_objectives
test_gpu_objectives_native
test_gpu_portability
test_gpu_predict
test_gpu_row_compaction
test_gpu_random_score_noise
test_gpu_runtime
test_gpu_scale_refresh
test_gpu_scan_primitives
test_gpu_sparse
test_gpu_sparse_skip
test_gpu_speculation_build
test_gpu_split_scan
test_gpu_split_search
test_gpu_strategies
test_gpu_tiling
test_gpu_training
test_gpu_vendor_policy
test_host_replica
"

# `test_gpu_tile_floor` is deliberately NOT above. It asserts the tiling
# geometry as pure host arithmetic over synthetic `DeviceCaps` and opens no
# device, so it belongs in the CPU set where it also guards the rule on a
# runner that has no accelerator to plan for.
#
# `test_golden_bits` is deliberately NOT above, and must never be added to
# it. It is the package's bit-exactness contract: checked-in IEEE-754 bit
# patterns for six fits, compared as integers. A CPU-only runner has to be
# able to enforce that contract, so the file opens no device and runs in
# both the `cpu` and `all` sets, which the glob above already selects it
# into. Accelerator exactness is a separate claim and belongs in the GPU
# files.
#
# `test_gpu_phase_profile` is deliberately NOT above either, despite its name.
# It exercises the node size classes, the counters, and the CPU grower's
# charge sites, opens no device, and asserts that the instrument moves no
# model on the host backend, so it belongs in the CPU set.

# CPU-safe, but exercises a GPU path when one exists, so a GPU runner should
# see it too.
GPU_ALSO="test_interaction"

# Derive GPU-only status from the file NAME, so the list above cannot drift.
#
# Naming, not content.  The first attempt at this classified any file
# mentioning `has_accelerator`, `DeviceContext`, or a `mojotrees.gpu_` import,
# and it moved seven genuinely CPU-safe files out of the CPU set, which is a
# coverage loss rather than a fix.  Those files are safe precisely because they
# wrap their device work in `comptime if not has_accelerator()`, so mentioning
# the symbol is what safety looks like, not what danger looks like.
#
# Every file that actually broke was named `test_gpu_*` and simply had not been
# added to the list.  So that is the rule: a `test_gpu_*` file is GPU-only
# unless it declares itself with a marker comment on any line:
#
#     # run_tests: cpu-safe
#
# `test_gpu_tile_floor` uses that marker; it asserts tiling geometry over
# synthetic `DeviceCaps` and opens nothing.  `test_golden_bits` needs no marker
# because of its name, and carries one anyway so that a future rename cannot
# quietly pull the bit-exactness contract out of CPU-only CI.
gpu_by_content() {
  case "$1" in test_gpu_*) ;; *) return 1 ;; esac
  grep -q '^[[:space:]]*#[[:space:]]*run_tests: cpu-safe' "tests/$1.mojo" 2>/dev/null \
    && return 1
  return 0
}

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
      cpu) { in_list "$name" "$GPU_ONLY" || gpu_by_content "$name"; } \
             || SELECTED+=("$name") ;;
      gpu) if in_list "$name" "$GPU_ONLY" || gpu_by_content "$name" \
             || in_list "$name" "$GPU_ALSO"
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

# `TestSuite` closes with a line of the form
#
#   Summary [ 0.061 ] 47 tests run: 47 passed , 0 failed , 0 skipped
#
# and that bracketed number is MILLISECONDS, not seconds.  Measured against
# a test that sleeps for exactly one second, the harness prints 1000.598.
# Reporting it next to the wall clock is the whole point: for almost every
# file in this tree the two numbers differ by four or five orders of
# magnitude, which says the wall time is `mojo` compiling and not the tests
# running.  See docs/design/TEST_HARNESS_COST.md.
suite_ms() {
  local ms
  ms="$(sed -n 's/^Summary \[ *\([0-9.][0-9.]*\) \].*/\1/p' "$1" | tail -1)"
  if [ -n "$ms" ]; then echo "$ms"; else echo "?"; fi
}

run_one() {
  local name="$1"
  local log="$RESULTS/$name.log"
  local start=$SECONDS
  if mojo run $PKG_INCLUDE -I tests $(extra_includes "$name") \
        "tests/$name.mojo" >"$log" 2>&1; then
    echo "  ok   $name ($((SECONDS - start))s wall, $(suite_ms "$log")ms in tests)"
  else
    echo "fail" >"$RESULTS/$name.failed"
    echo "  FAIL $name ($((SECONDS - start))s wall, $(suite_ms "$log")ms in tests)"
  fi
}
export -f run_one extra_includes suite_ms
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
