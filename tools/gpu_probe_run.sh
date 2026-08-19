#!/usr/bin/env bash
# One-command GPU validation run for a leased machine. VENDOR NEUTRAL.
#
# WHY THIS IS IN THE REPOSITORY RATHER THAN PASTED INTO AN SSH SESSION
#
# Leased GPU boxes are reached two ways and both fight you. A RunPod proxy
# session is an interactive PTY that ignores remote command arguments and
# truncates any line past roughly 4 KiB, so a base64'd setup script does not
# survive it. Direct SSH avoids that but only exists on images that honor the
# PUBLIC_KEY convention, which the official ROCm images do not. On 2026-08-19
# that combination cost a working NVIDIA pod and about fifteen minutes of paid
# time before the obvious answer surfaced: put the script where `git clone`
# can reach it, then the command that starts a run is short enough to survive
# any terminal.
#
#   git clone --depth 40 https://github.com/mojotrees/mojotrees.git
#   nohup bash mojotrees/tools/gpu_probe_run.sh > run.log 2>&1 &
#
# It detaches, so the SSH session can die without taking the run with it, and
# every phase prints a MARKER line so progress can be read by grepping rather
# than by watching.
#
# WHAT IT RUNS, IN THE ORDER THAT ANSWERS THE MOST PER MINUTE
#
#  1. The variety probe TWICE. The difference between the two runs is the
#     measurement: on an M4 the second run is 18x faster, which means
#     something amortizes kernel compilation between processes there. If the
#     second run here is identical to the first, this backend recompiles
#     everything every time and that asymmetry is the sharpest difference
#     between the backend that works and the one that hangs.
#  2. The four MAX-only probes, which pass on Metal and on CUDA and establish
#     that the machine and toolchain are basically sound.
#  3. Two known-good suites, to confirm this machine matches the recorded one.
#  4. One training fit under a hard cap, which is the thing that hangs.
#
# Everything is wrapped in `timeout`. A benchmark that can hang cannot be run
# unattended, and unattended on hardware billed by the hour is exactly how
# these get run.
set -u
export PATH="$HOME/.pixi/bin:$PATH"
LOG=/root/run.log
say() { echo "=== MARKER $* ==="; }

say ENV_START
uname -a
(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null) || \
  (rocm-smi --showproductname 2>/dev/null | head -20) || echo "no gpu tool"
echo "nproc: $(nproc)"
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)"
say ENV_OK

say PIXI_INSTALL
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash >/dev/null 2>&1
  export PATH="$HOME/.pixi/bin:$PATH"
fi
pixi --version || { say PIXI_FAILED; exit 1; }
say PIXI_OK

# NO CLONE. This script lives INSIDE a checkout, so it runs from the one it
# was fetched with and works from there.
#
# It used to `rm -rf mojotrees` and re-clone, which deleted the file bash was
# still reading and killed the run mid-flight with "No such file or
# directory". A script that removes its own directory is a script that cannot
# be re-run, and this one cost a paid NVIDIA pod eight minutes to learn that.
say REPO
cd "$(dirname "$0")/.." || { say REPO_FAILED; exit 1; }
pwd
git log --oneline -1
say REPO_OK

say BUILD
timeout 1500 pixi run build-pkg > /root/build.log 2>&1
echo "build rc=$?"
ls -la build/mojotrees.mojopkg 2>&1 | tail -1
grep -ci "error" /root/build.log || true
say BUILD_DONE

# ---- The cache question. Run the variety probe TWICE and compare. ----
say VARIETY_COLD
MOJOTREES_BENCH_TIMEOUT=900 timeout 1000 pixi run probe-cuda-variety > /root/variety1.log 2>&1
echo "rc=$?"
grep -E "^(device|api|n_kernels|PHASE|VARIETY)" /root/variety1.log | tail -20
echo "--- first/last kernels ---"
grep "KERNEL" /root/variety1.log | head -3
grep "KERNEL" /root/variety1.log | tail -3
echo "--- nv compute cache after run ---"
ls -la ~/.nv/ComputeCache 2>&1 | head -3
find ~/.nv -type f 2>/dev/null | wc -l
say VARIETY_COLD_DONE

say VARIETY_WARM
MOJOTREES_BENCH_TIMEOUT=900 timeout 1000 pixi run probe-cuda-variety > /root/variety2.log 2>&1
echo "rc=$?"
grep -E "^(PHASE_G_OK|PHASE_H_OK|VARIETY)" /root/variety2.log
grep "KERNEL" /root/variety2.log | head -3
grep "KERNEL" /root/variety2.log | tail -3
find ~/.nv -type f 2>/dev/null | wc -l
say VARIETY_WARM_DONE

# ---- The four existing probes. Should all pass. ----
for p in alloc launch subbuffer parallel; do
  say "PROBE_$p"
  MOJOTREES_BENCH_TIMEOUT=420 timeout 500 pixi run probe-cuda-$p > /root/probe_$p.log 2>&1
  echo "rc=$?"
  grep -E "PHASE_|COMPLETED|DEADLOCK" /root/probe_$p.log | tail -8
done
say PROBES_DONE

# ---- The known-good suites, to confirm the machine matches the old record ----
say KNOWN_GOOD
for t in test_gpu_scan_primitives test_gpu_raw_update_packing; do
  s=$(date +%s)
  timeout 300 pixi run mojo run -I src -I tests tests/$t.mojo > /root/$t.log 2>&1
  echo "$t rc=$? secs=$(( $(date +%s) - s ))"
  grep -E "tests run|Summary" /root/$t.log | tail -2
done
say KNOWN_GOOD_DONE

# ---- The hang itself, under a hard cap ----
say TRAINING_FIT
s=$(date +%s)
timeout 600 pixi run mojo run -I src -I tests tests/test_gpu_training.mojo > /root/training.log 2>&1
echo "training rc=$? secs=$(( $(date +%s) - s ))"
tail -5 /root/training.log
say TRAINING_DONE

say ALL_DONE
