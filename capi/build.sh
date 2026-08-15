#!/bin/sh
# Build the C ABI shared library into capi/libmojotrees.{dylib,so}.
# Run from anywhere; requires pixi.
set -e
cd "$(dirname "$0")/.."

case "$(uname -s)" in
    Darwin) lib="capi/libmojotrees.dylib" ;;
    *) lib="capi/libmojotrees.so" ;;
esac

pixi run mojo build --emit shared-lib -I src -I capi \
    capi/mojotrees_capi.mojo -o "$lib"
echo "built $lib"
