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
#                         package, which skips the build step.  Slower for a
#                         suite and much faster for one file; see below.
#   MOJOTREES_TEST_REBUILD  set to 1 to rebuild the package even when it is
#                         already newer than everything under `src/`
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
# concurrent files.
#
# That argument is about a SIXTY-file suite and does not carry to one file,
# which is worth stating because it was carried anyway and cost every lane
# real time.  Precompiling elaborates all sixty modules; `-I src` elaborates
# only the modules the file being run imports, which for a light test is a
# handful the cache serves immediately.  Measured on one 49-test file, warm,
# interleaved, two repeats each: 12.2 s through the package against 0.75 s
# with `MOJOTREES_TEST_PKG=0`, and at two files 16.3 s against 5.4 s.  The
# precompile does not pay for itself until several files, so the rule is
# `MOJOTREES_TEST_PKG=0` for one or two files and the default for more.
# Skipping the rebuild when `src/` has not changed (below) removes most of
# the difference without anyone having to remember the variable.
#
# Lowering the optimization level is NOT a lever, it is
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
#
# In a LANE WORKTREE, borrow the main checkout's environment before reaching
# for `pixi run`.  `pixi run` here would treat the worktree as its own
# project and install a second complete copy of the environment into
# `<worktree>/.pixi`, roughly 1.1 GB, before compiling anything -- and with
# it a second, empty Mojo compile cache, because `MODULAR_HOME` follows the
# environment.  Measured 2026-08-16: 46 lane worktrees had done that, 49 GB
# of duplicated environments, their caches running 1.1 MB to 243 MB against
# the main checkout's 8.7 GB.  `tools/lane_env.sh` points at the main one
# instead, declines when the manifests differ, and is a no-op in the main
# checkout, so the `pixi run` fallback below still covers every case it
# cannot serve.
if ! command -v mojo >/dev/null 2>&1; then
  if [ -r "$ROOT/tools/lane_env.sh" ]; then
    # shellcheck source=/dev/null
    . "$ROOT/tools/lane_env.sh" || true
  fi
fi
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
# The pool is sized by CPU count, but a GPU test is not bounded by the CPU:
# it is bounded by the one accelerator every process in the pool has to share.
# On the 10-core M4 this project is developed on, `NCPU - 2` is 8 and the
# distinction never surfaces.  On a 256-core host it is 254, and 254 processes
# each opening a device context against a single card do not run 254 times
# faster; they thrash.  Measured on a RunPod RTX 5090 (256 vCPU), 2026-08-18:
# GPU utilization sat at 0% with 7.1 GB of VRAM consumed by contexts alone,
# one test was killed at 503s by MAX's own watchdog, and the suite made no
# progress.  The same suite at MOJOTREES_TEST_JOBS=4 is what produced the
# first NVIDIA result this repository holds.
#
# GPU_MAX_JOBS is a device-contention bound, not a throughput tuning knob, so
# it is deliberately small and does not scale with NCPU.  An explicit
# MOJOTREES_TEST_JOBS still wins: this clamps the DEFAULT, it does not
# override a number somebody chose on purpose.
GPU_MAX_JOBS="${MOJOTREES_GPU_TEST_JOBS:-4}"
if [ "$MODE" = "gpu" ] && [ "$DEFAULT_JOBS" -gt "$GPU_MAX_JOBS" ]; then
  DEFAULT_JOBS="$GPU_MAX_JOBS"
fi
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
test_gpu_ranking_device
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
# `test_cosine_device_split` is the second entry and is the same shape: its
# replica-against-host assertions run anywhere, and its last two tests open a
# device to check that the kernels choose what the host chooses under
# `score_function=Cosine`. Naming it `test_gpu_*` would have hidden the
# host-side half from CPU-only CI, which is where most of its content is.
GPU_ALSO="test_interaction test_cosine_device_split"

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
    # Reaches into bindings/ because the thing under test IS a binding: the
    # objective sentinel fold that used to live in `decide_device_workload`.
    # Calling it needs a real CPython dict, which is what makes the test a
    # test of the boundary rather than of a model of the boundary.
    test_objective_marshalling) echo "-I bindings" ;;
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
  PKG="build/mojotrees.mojopkg"
  # Rebuild only when there is something to rebuild.  This used to be
  # unconditional, which put a floor of roughly eleven seconds under every
  # invocation of this script -- measured at 12.2 s for one 49-test file
  # against 0.75 s for the same file with the build skipped, on a warm cache
  # that the build does not benefit from.  A lane running one file after one
  # edit paid that floor every time, and it is the reason focused runs felt
  # like they cost minutes.
  #
  # The test is mtime against `src/`, which is the precompile's entire input
  # (`mojo precompile -I src src/mojotrees`).  Nothing else can invalidate the
  # package: a branch switch rewrites the mtimes of the files it changes, so
  # moving between commits invalidates exactly the way an edit does, and a
  # missing `build/` invalidates by the `-f` test.  `find | head -1` rather
  # than `find -quit`, which is not portable.
  #
  # This does not weaken the check the header describes.  Precompiling
  # elaborates every module in `src/mojotrees/`, including the ones no test
  # imports, and skipping it when no module changed means the elaboration
  # that already succeeded still stands.  `MOJOTREES_TEST_REBUILD=1` forces
  # the build anyway, which is what to reach for if a package is ever
  # suspected of being stale for a reason mtime cannot see.
  if [ "${MOJOTREES_TEST_REBUILD:-0}" = "0" ] &&
     [ -f "$PKG" ] &&
     [ -z "$(find src -newer "$PKG" | head -1)" ]; then
    echo "build/mojotrees.mojopkg is up to date with src/, skipping the build"
  else
    echo "building build/mojotrees.mojopkg"
    mkdir -p build
    if ! mojo precompile -I src src/mojotrees -o "$PKG"; then
      echo "package build failed; the suite cannot run against a package it" >&2
      echo "could not build.  Re-run with MOJOTREES_TEST_PKG=0 to compile" >&2
      echo "each test from src/ instead." >&2
      exit 1
    fi
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

# Run one test under a wall-clock cap.
#
# Why this exists: `run_one` used to invoke `mojo run` bare, so a test that
# hung hung the whole suite with no upper bound and no diagnosis.  That is a
# real failure mode rather than a hypothetical one.  On a RunPod RTX 5090
# (2026-08-18) an over-subscribed GPU pool left tests parked on a contended
# device; the only thing that ever stopped one was MAX's own watchdog at 503s,
# which reports as a bare "Alarm clock" with no indication of which test or
# why.  A suite that cannot distinguish "slow" from "wedged" cannot be run
# unattended, which is precisely how it gets run on leased hardware.
#
# `timeout` is NOT portable: it is coreutils, so Linux runners have it and
# macOS does not ship it at all (Homebrew coreutils installs it as `gtimeout`).
# Since the development machines here are macOS and CI is Linux, a bare
# `timeout` would silently protect CI and nothing else.  Hence the probe below
# and the shell fallback, which uses only POSIX job control.
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout"
else
  TIMEOUT_CMD=""
fi
# Generous by design.  This is a hang detector, not a performance budget: it
# must never fire on a test that is merely slow on a cold compile cache, or it
# would turn a green suite red for a reason that has nothing to do with the
# code.  The slowest legitimate GPU test observed so far is well under a
# minute of device time; the compile in front of it is what makes the wall
# time long.  Set to 0 to disable.
TEST_TIMEOUT="${MOJOTREES_TEST_TIMEOUT:-900}"

# Fallback for hosts with neither `timeout` nor `gtimeout`: run the command in
# the background, poll for the deadline, and kill it if it outlives one.
# Returns 124 on timeout, matching coreutils, so the caller needs no special
# case.
run_with_deadline() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$secs" ]; then
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  wait "$pid"
}

run_one() {
  local name="$1"
  local log="$RESULTS/$name.log"
  local start=$SECONDS
  local rc=0

  if [ "$TEST_TIMEOUT" -gt 0 ] 2>/dev/null; then
    if [ -n "$TIMEOUT_CMD" ]; then
      "$TIMEOUT_CMD" "$TEST_TIMEOUT" mojo run $PKG_INCLUDE -I tests \
        $(extra_includes "$name") "tests/$name.mojo" >"$log" 2>&1 || rc=$?
    else
      run_with_deadline "$TEST_TIMEOUT" mojo run $PKG_INCLUDE -I tests \
        $(extra_includes "$name") "tests/$name.mojo" >"$log" 2>&1 || rc=$?
    fi
  else
    mojo run $PKG_INCLUDE -I tests $(extra_includes "$name") \
      "tests/$name.mojo" >"$log" 2>&1 || rc=$?
  fi

  if [ "$rc" -eq 0 ]; then
    echo "  ok   $name ($((SECONDS - start))s wall, $(suite_ms "$log")ms in tests)"
  elif [ "$rc" -eq 124 ]; then
    # Distinguished from a plain failure on purpose: a timeout means the test
    # never reported, so its log is truncated and its assertions are unknown.
    # Reading it as "the assertions failed" would be wrong.
    echo "timeout" >"$RESULTS/$name.failed"
    echo "  TIMEOUT $name (killed after ${TEST_TIMEOUT}s; log is incomplete)" \
      >>"$log"
    echo "  TIMEOUT $name (killed after ${TEST_TIMEOUT}s, no result reported)"
  else
    echo "fail" >"$RESULTS/$name.failed"
    echo "  FAIL $name ($((SECONDS - start))s wall, $(suite_ms "$log")ms in tests)"
  fi
}
# Everything `run_one` touches has to cross into the `bash -c` subshell that
# xargs spawns per test, and a shell variable that is merely SET does not.
#
# `run_with_deadline` and the two timeout variables were missing from these two
# lines when the timeout was first added, and the failure was silent in the
# worst way: in the subshell `TEST_TIMEOUT` expanded to the empty string, so
# `[ "" -gt 0 ]` failed, the `2>/dev/null` swallowed the error, and every test
# took the untimed branch. The suite reported zero timeouts while five tests
# sat in futex_wait_queue for twenty-five minutes against a 420s cap, on the
# very run that was supposed to prove the timeout worked.
#
# A safety mechanism that silently does nothing is worse than none, because it
# is trusted. Anything run_one needs goes here.
export -f run_one extra_includes suite_ms run_with_deadline
export RESULTS PKG_INCLUDE ROOT TEST_TIMEOUT TIMEOUT_CMD

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
