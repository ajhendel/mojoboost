#!/bin/sh
# Build the CPython extension module into python/mojoboost/_mojoboost.so.
# Run from anywhere; requires pixi.
set -e
cd "$(dirname "$0")/.."
pixi run mojo build --emit shared-lib -I src \
    bindings/_mojoboost.mojo -o python/mojoboost/_mojoboost.so
echo "built python/mojoboost/_mojoboost.so"
