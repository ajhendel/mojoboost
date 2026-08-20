#!/usr/bin/env bash
# Run gbm-bench covtype on a leased GPU box, from a clean container.
#
# Written 2026-08-20. This is the container's main process: it installs a
# toolchain, clones this repository, builds, smoke-tests the GPU, and then
# runs NVIDIA's gbm-bench three times with every arm interleaved. It ends in
# a long sleep because a RunPod container whose main process exits is a
# crash loop, and the log is the only way results come back.
#
# It lives in the repository rather than in a base64 environment variable
# because bench/external/README.md exists to let a skeptic rerun what we ran,
# and a benchmark invocation that survives only in someone's shell history
# does not have that property. That complaint is already written about this
# directory; this file is the fix.
#
# Backend-agnostic on purpose. It reports whichever accelerator it finds and
# does not assume one, so the same script is the NVIDIA run and the AMD run.
#
#   MJT_BRANCH   branch to clone           (default main)
#   MJT_DATASET  gbm-bench dataset         (default covtype)
#   MJT_NTREES   trees per arm             (default 100)
#   MJT_REPEATS  interleaved repeats       (default 3)
set -uo pipefail
exec 2>&1

BRANCH="${MJT_BRANCH:-main}"
DATASET="${MJT_DATASET:-covtype}"
NTREES="${MJT_NTREES:-100}"
REPEATS="${MJT_REPEATS:-3}"

echo "MJT===== ENVIRONMENT ====="
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname 2>/dev/null | head -12
  cat /opt/rocm/.info/version 2>/dev/null || echo "no /opt/rocm/.info/version"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null | head -4
fi
. /etc/os-release 2>/dev/null; echo "OS: ${PRETTY_NAME:-unknown}"
grep -m1 "model name" /proc/cpuinfo

# THE CONTAINER CPU COUNT, WHICH IS NOT WHAT nproc SAYS.
#
# RunPod advertises the host's core count, 128 to 256, against a cgroup quota
# of 15 to 27. nproc reads the host. src/mojotrees/parallel.mojo reads nproc,
# gbm-bench's -cpus default resolves to nproc, and OpenMP reads nproc, so
# every arm in the run oversubscribes by an order of magnitude unless this is
# computed and passed down. This is the single line that decides whether the
# numbers from this box mean anything.
echo "NPROC_REPORTED: $(nproc)"
CPUMAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo 'max 100000')"
echo "cgroup cpu.max: $CPUMAX"
QUOTA="$(echo "$CPUMAX" | awk '{ if ($1=="max") print 0; else printf "%d", $1/$2 }')"
if [ "${QUOTA:-0}" -lt 1 ]; then QUOTA="$(nproc)"; echo "MJT no cgroup quota, using nproc"; fi
echo "MJT CPUS_PINNED=$QUOTA"
free -g 2>/dev/null | head -2

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git curl build-essential >/dev/null 2>&1
export HOME=/root
curl -fsSL https://pixi.sh/install.sh | bash >/dev/null 2>&1
export PATH="/root/.pixi/bin:$PATH"
echo "PIXI: $(pixi --version)"

echo "MJT===== CLONE ====="
cd /root && rm -rf mojotrees
git clone --branch "$BRANCH" --depth 1 https://github.com/mojolearn/mojotrees.git 2>&1 | tail -2
cd /root/mojotrees || exit 1
# Capture the commit into the log. The 2026-08-19 AMD record had to INFER its
# commit from push timing because the pod printed it and nobody kept the line,
# and the pod was gone by the time anyone noticed.
echo "MJT COMMIT: $(git log --oneline -1)"

echo "MJT===== PIXI INSTALL default ====="
time pixi install 2>&1 | tail -4
echo "MOJO: $(pixi run mojo --version 2>&1 | tail -1)"

echo "MJT===== BUILD PKG ====="
time pixi run build-pkg 2>&1 | grep -viE "^ +\^|warning:|Included from|^ *$" | tail -12
echo "MJT BUILD_PKG_RC=$?"

# A green build is not evidence the GPU path exists: has_accelerator() is
# comptime, so on a runtime MAX does not recognize the whole GPU half compiles
# out and the build still exits 0. This test is what distinguishes them.
echo "MJT===== SMOKE: GPU TRAINING TEST ====="
s=$(date +%s)
timeout 900 pixi run mojo run -I src -I tests tests/test_gpu_training.mojo > /tmp/train.log 2>&1
echo "MJT TRAINING rc=$? secs=$(( $(date +%s) - s ))"
tail -4 /tmp/train.log

echo "MJT===== BUILD PYTHON EXTENSION ====="
time pixi run build-python 2>&1 | grep -viE "^ +\^|warning:|Included from|^ *$" | tail -8
ls -la python/mojotrees/_mojotrees.so 2>&1 | tail -1

echo "MJT===== PIXI INSTALL bench ====="
time pixi install -e bench 2>&1 | tail -4
pixi run -e bench python -c "import lightgbm,xgboost,sklearn,pandas;print('lgbm',lightgbm.__version__,'xgb',xgboost.__version__)" 2>&1 | tail -3
# conda-forge's linux-64 catboost is a CUDA build. It is expected to import on
# a CPU-only or AMD box and fall back, but if it does not, the run drops that
# arm rather than losing the other five.
pixi run -e bench python -c "import catboost;print('catboost',catboost.__version__)" 2>&1 | tail -3
CATOK=$?
echo "MJT CATBOOST_IMPORT_RC=$CATOK"
PYTHONPATH=/root/mojotrees/python pixi run -e bench python -c \
  "import mojotrees;print('mojotrees',mojotrees.__version__,'gpu_available',mojotrees.gpu_available())" 2>&1 | tail -4

if [ "$CATOK" -eq 0 ]; then
  ARMS="full"
else
  ARMS="mojotrees-cpu,mojotrees-gpu,lgbm-cpu,lgbm-cpu-det,xgb-cpu"
fi
echo "MJT ARMS=$ARMS"

for rep in $(seq 1 "$REPEATS"); do
  echo "MJT===== GBM-BENCH $DATASET REPEAT $rep ====="
  s=$(date +%s)
  GBM_BENCH_CPUS="$QUOTA" timeout 2400 pixi run -e bench bash \
    bench/external/run_gbm_bench.sh "$DATASET" "$NTREES" "$ARMS" > "/tmp/gbm$rep.log" 2>&1
  echo "MJT GBM$rep rc=$? secs=$(( $(date +%s) - s ))"
  grep -E "pinning|Error|error|Traceback" "/tmp/gbm$rep.log" | head -6
  echo "MJT---JSON REPEAT $rep---"
  ls -t "/root/mojotrees/bench/results/gbm_bench_${DATASET}"_*.json 2>/dev/null | head -1 | xargs -r cat
done

echo "MJT===== DONE ====="
sleep 100000
