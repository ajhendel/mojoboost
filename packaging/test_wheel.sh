#!/usr/bin/env bash
# Install the built wheel into clean venvs and test it there, never against
# the source tree, from a neutral directory.
# Run via: pixi run test-wheel
#
# Two installs, because both are shipped configurations:
#   bare  wheel only, so the stdlib fallback and the dependency-free suite
#         run exactly as a user with no numpy would get them
#   full  wheel plus numpy and pytest (and scikit-learn and pandas when they
#         install), which runs the estimator suite in python/tests
set -euo pipefail
cd "$(dirname "$0")/.."

WHEEL=$(ls python/dist/mojotrees-*.whl)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cp packaging/smoke_test.py python/test_python_api.py "$WORK/"
cp -R python/tests "$WORK/tests"

echo "== bare install (no numpy) =="
python -m venv "$WORK/bare"
"$WORK/bare/bin/pip" install --quiet "$WHEEL"
(cd "$WORK" && ./bare/bin/python smoke_test.py)
(cd "$WORK" && ./bare/bin/python test_python_api.py)

echo "== full install =="
python -m venv "$WORK/full"
"$WORK/full/bin/pip" install --quiet "$WHEEL" numpy pytest
# Optional: the scikit-learn and pandas tests skip themselves when these do
# not install for this interpreter.
"$WORK/full/bin/pip" install --quiet scikit-learn pandas \
    || echo "scikit-learn/pandas unavailable; those tests will skip"
(cd "$WORK" && ./full/bin/python smoke_test.py)
(cd "$WORK" && ./full/bin/python -m pytest -q tests)

echo "wheel ok: $WHEEL"
