#!/usr/bin/env bash
# Everything about the box that a reader needs in order to discount a number
# correctly. Written beside every external-harness result.
set -uo pipefail
echo "recorded_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "uname=$(uname -a)"
if [ "$(uname -s)" = "Darwin" ]; then
  echo "model=$(sysctl -n hw.model 2>/dev/null)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "cores=$(sysctl -n hw.ncpu 2>/dev/null)"
  echo "memory_bytes=$(sysctl -n hw.memsize 2>/dev/null)"
  echo "macos=$(sw_vers -productVersion 2>/dev/null)"
  echo "thermal=$(pmset -g therm 2>/dev/null | tr '\n' ';')"
  echo "power=$(pmset -g batt 2>/dev/null | head -1)"
fi
echo "python=$(python3 --version 2>&1)"
for pkg in mojotrees lightgbm xgboost catboost numpy; do
  v=$(python3 -c "import $pkg,sys;sys.stdout.write(getattr($pkg,'__version__','?'))" 2>/dev/null) \
    && echo "$pkg=$v"
done
python3 -c "import mojotrees; mojotrees.show_versions()" 2>/dev/null || true
