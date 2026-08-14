#!/usr/bin/env bash
# Install the built wheel into a clean venv and run the API tests against
# the installed package (not the source tree), from a neutral directory.
# Run via: pixi run test-wheel
set -euo pipefail
cd "$(dirname "$0")/.."

WHEEL=$(ls python/dist/mojoboost-*.whl)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet "$WHEEL"
"$WORK/venv/bin/pip" install --quiet numpy \
    || echo "numpy install failed; tests will use the stdlib fallback"

cp python/test_python_api.py "$WORK/"
(cd "$WORK" && ./venv/bin/python test_python_api.py)
echo "wheel ok: $WHEEL"
