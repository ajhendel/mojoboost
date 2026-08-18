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

echo "==> pinning the harness commit for the record"
GBM_SHA="$(git -C "$WORK" rev-parse HEAD)"

echo "==> patching (idempotent)"
python3 "$REPO_ROOT/bench/external/patch_gbm_bench.py" "$WORK"

mkdir -p "$DATA" "$REPO_ROOT/bench/results"

echo "==> recording the box"
"$REPO_ROOT/bench/external/record_environment.sh" > "${OUT%.json}.env.txt" 2>&1 || true
echo "gbm_bench_commit=$GBM_SHA" >> "${OUT%.json}.env.txt"

echo "==> running: dataset=$DATASET ntrees=$NTREES algorithms=$ALGOS"
echo "    the harness downloads $DATASET into $DATA on first use; some of"
echo "    these datasets are tens of GB, so check free space before airline"
echo "    or bosch."
cd "$WORK"
python3 runme.py \
  -root "$DATA" \
  -dataset "$DATASET" \
  -algorithm "$ALGOS" \
  -ntrees "$NTREES" \
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
