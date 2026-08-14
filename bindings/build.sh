#!/bin/sh
# Build the CPython extension module into python/mojoboost/_mojoboost.so.
# Run from anywhere; requires pixi.
#
# Two include paths, both required. `-I src` is the mojoboost package.
# `-I bindings` is this directory: _mojoboost.mojo is the entry point and
# imports its capability modules (binding_support, objective_bindings,
# dataset_bindings, inspection_bindings, distributed_bindings,
# basic_bindings) as top-level modules, which only resolve when the
# directory holding them is on the import path. Dropping it fails the
# build loudly at the first import rather than producing a smaller module.
#
# packaging/build_wheel.sh and packaging/linux/build_wheel_linux.sh both
# run this script rather than repeating the command, so this is the only
# place the flags are written.
set -e
cd "$(dirname "$0")/.."
pixi run mojo build --emit shared-lib -I src -I bindings \
    bindings/_mojoboost.mojo -o python/mojoboost/_mojoboost.so
echo "built python/mojoboost/_mojoboost.so"
