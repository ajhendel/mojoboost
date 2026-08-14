#!/bin/sh
# Build the C ABI shared library into capi/libmojoboost.{dylib,so}.
# Run from anywhere; requires pixi.
set -e
cd "$(dirname "$0")/.."

case "$(uname -s)" in
    Darwin) lib="capi/libmojoboost.dylib" ;;
    *) lib="capi/libmojoboost.so" ;;
esac

pixi run mojo build --emit shared-lib -I src -I capi \
    capi/mojoboost_capi.mojo -o "$lib"
echo "built $lib"
