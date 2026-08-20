#!/usr/bin/env bash
# Run NVIDIA's gbm-bench with the mojotrees arms added.
#
# This script sets a run up and then runs it. It does not decide what is
# worth publishing; bench/external/README.md holds that, and the short
# version is that no number from a machine other than a validated one is a
# publishable number.
#
# usage:
#   bench/external/run_gbm_bench.sh <dataset> <ntrees> [algorithms]
# example:
#   bench/external/run_gbm_bench.sh year 500 mojotrees-gpu,lgbm-cpu,cat-cpu
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${GBM_BENCH_HOME:-$REPO_ROOT/bench/external/.gbm-bench}"
DATA="${GBM_BENCH_DATA:-$REPO_ROOT/bench/external/.gbm-datasets}"

DATASET="${1:?dataset required (airline, year, higgs, epsilon, football, fraud, bosch, url)}"
NTREES="${2:?ntrees required}"
# Tier 1 (ours only, read against competitor_baselines.json) is what a
# development loop wants; tier 2 (every arm interleaved) is what anything
# publishable requires. See bench/external/README.md.
if [ "${3:-}" = "ours" ]; then
  ALGOS="mojotrees-cpu,mojotrees-gpu"
elif [ "${3:-}" = "full" ] || [ -z "${3:-}" ]; then
  ALGOS="mojotrees-cpu,mojotrees-gpu,lgbm-cpu,lgbm-cpu-det,cat-cpu,xgb-cpu"
else
  ALGOS="$3"
fi

STAMP="$(date +%Y-%m-%d_%H%M%S)"
OUT="$REPO_ROOT/bench/results/gbm_bench_${DATASET}_${STAMP}.json"

if [ ! -d "$WORK" ]; then
  echo "==> cloning gbm-bench into $WORK"
  git clone --depth 1 https://github.com/NVIDIA/gbm-bench.git "$WORK"
fi

# ENVIRONMENT PRECONDITIONS, CHECKED BEFORE ANYTHING IS DOWNLOADED.
#
# Added 2026-08-18 after following the usage block above from a clean shell
# failed twice in sequence: first `ModuleNotFoundError: No module named
# 'pandas'`, because the bare `python3` below picks up the system interpreter
# rather than the bench environment, and then `ModuleNotFoundError: No module
# named 'mojotrees'`, because the bench environment does not install this
# package and nothing put python/ on the path. It runs on the third attempt
# with both fixed, and neither requirement appeared in the usage block, in
# this script, or in the README.
#
# That matters more here than it would anywhere else in the tree. This
# directory exists to answer "was the harness shaped around the result", and
# its entire value is that a skeptical reader can rerun it. A script whose
# documented invocation does not work, and whose working invocation lives
# only in the shell history of whoever wrote it, does not have that property.
# The numbers already in bench/results/ are good numbers; the point is that
# nobody else could reproduce them from the repository alone.
#
# The check runs BEFORE the download rather than after it, because these
# datasets are tens of gigabytes and failing on an import after fetching one
# is the expensive ordering.
PY_BIN="${GBM_BENCH_PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PY_BIN" -c "import pandas, sklearn, mojotrees" 2>/dev/null; then
  cat >&2 <<'PRECHECK'
run_gbm_bench.sh: the interpreter cannot import pandas, sklearn and mojotrees.

This script needs the bench environment AND this repository's python/ on the
path. It does not resolve either for you, because it does not know how you
manage environments. Two ways that work here:

  pixi run -e bench bash bench/external/run_gbm_bench.sh <dataset> <ntrees> [algos]

  GBM_BENCH_PYTHON=/path/to/python bash bench/external/run_gbm_bench.sh ...

PYTHONPATH is set for you and already points at the repository's python/.
If mojotrees is the missing one, `pixi run build-python` writes the extension;
note that `pixi run build-pkg` does NOT, which has caught people.
PRECHECK
  exit 2
fi

echo "==> pinning the harness commit for the record"
GBM_SHA="$(git -C "$WORK" rev-parse HEAD)"

echo "==> patching (idempotent)"
"$PY_BIN" "$REPO_ROOT/bench/external/patch_gbm_bench.py" "$WORK"

mkdir -p "$DATA" "$REPO_ROOT/bench/results"

echo "==> recording the box"
"$REPO_ROOT/bench/external/record_environment.sh" > "${OUT%.json}.env.txt" 2>&1 || true
echo "gbm_bench_commit=$GBM_SHA" >> "${OUT%.json}.env.txt"

echo "==> running: dataset=$DATASET ntrees=$NTREES algorithms=$ALGOS"
echo "    the harness downloads $DATASET into $DATA on first use; some of"
echo "    these datasets are tens of GB, so check free space before airline"
echo "    or bosch."
# THREAD COUNT, WHICH IS NOT OPTIONAL IN A CONTAINER.
#
# gbm-bench's -cpus defaults to 0, which it resolves to the processor count
# the OS reports. On a leased container that is the HOST core count, not the
# cgroup quota: RunPod advertises 128 to 256 CPUs against quotas of 15 to 27.
# Every arm then oversubscribes by an order of magnitude and no number from
# the run means anything.
#
# GBM_BENCH_CPUS is passed straight through to gbm-bench's own -cpus, which
# reaches LightGBM's nthread, XGBoost's nthread, CatBoost's thread_count and
# our MOJOTREES_NUM_WORKERS from one place. It is an argument the harness
# already exposes and it lands on every arm identically, so it does not
# forfeit the restraint this directory exists to demonstrate. Setting one
# library's thread count and not another's would.
#
# On a bare-metal box leave it unset and the upstream default applies.
CPUS_ARG=()
if [ -n "${GBM_BENCH_CPUS:-}" ]; then
  echo "==> pinning every arm to $GBM_BENCH_CPUS threads (container cgroup quota)"
  CPUS_ARG=(-cpus "$GBM_BENCH_CPUS")
fi

cd "$WORK"
"$PY_BIN" runme.py \
  -root "$DATA" \
  -dataset "$DATASET" \
  -algorithm "$ALGOS" \
  -ntrees "$NTREES" \
  "${CPUS_ARG[@]}" \
  -output "$OUT" \
  -verbose

echo
echo "==> wrote $OUT"
echo "    and the box record beside it at ${OUT%.json}.env.txt"
echo
echo "Before any of this leaves the machine, read the publication rules in"
echo "bench/external/README.md. The two that catch people:"
echo "  1. one run of each arm is not a measurement; repeat and interleave"
echo "  2. no NVIDIA or AMD number is publishable from an unvalidated backend"
