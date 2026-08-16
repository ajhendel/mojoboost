#!/bin/sh
# Build the CPython extension module into python/mojotrees/_mojotrees.so.
# Run from anywhere; requires pixi.
#
# Two include paths, both required. `-I src` is the mojotrees package.
# `-I bindings` is this directory: _mojotrees.mojo is the entry point and
# imports its capability modules (binding_support, objective_bindings,
# dataset_bindings, inspection_bindings, distributed_bindings,
# basic_bindings) as top-level modules, which only resolve when the
# directory holding them is on the import path. Dropping it fails the
# build loudly at the first import rather than producing a smaller module.
#
# packaging/build_wheel.sh and packaging/linux/build_wheel_linux.sh both
# run this script rather than repeating the command, so this is the only
# place the flags are written.
#
# The CPU target is NOT one of the flags written here. It comes from
# packaging/build_target.sh, which is shared with capi/build.sh and
# cli/build.sh so that the three artifacts this project ships cannot drift
# apart. Read that file before changing anything about the target: it is
# where the reasoning lives, including why the default is a portable
# baseline rather than the host, and what that costs.
#
# This is the script that matters most for the target, because it is the one
# every wheel goes through: build_release_wheel.sh -> test-wheel ->
# build-wheel -> packaging/build_wheel.sh -> here. The wheel published as
# mojotrees 0.1.0a3 came down that chain, on a self-hosted M4, with no target
# flags set anywhere on it.
set -e
cd "$(dirname "$0")/.."
. packaging/build_target.sh
mojotrees_resolve_target

# $MOJOTREES_TARGET_FLAGS is deliberately unquoted; see build_target.sh.
# shellcheck disable=SC2086
pixi run mojo build --emit shared-lib $MOJOTREES_TARGET_FLAGS \
    -strip-file-prefix "$PWD/" -I src -I bindings \
    bindings/_mojotrees.mojo -o python/mojotrees/_mojotrees.so
echo "built python/mojotrees/_mojotrees.so"
