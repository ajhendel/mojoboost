#!/bin/sh
# Build the C ABI shared library into capi/libmojotrees.{dylib,so}.
# Run from anywhere; requires pixi.
#
# The CPU target comes from packaging/build_target.sh, shared with
# bindings/build.sh and cli/build.sh. It defaults to a portable baseline
# rather than the host; that file says why, and what it costs.
#
# This artifact is redistributed by packaging/native (see
# packaging/native/layout.toml), so it has the same portability contract the
# wheel has, and the same reason to not be compiled for the build machine.
set -e
cd "$(dirname "$0")/.."
. packaging/build_target.sh
mojotrees_resolve_target

case "$(uname -s)" in
    Darwin) lib="capi/libmojotrees.dylib" ;;
    *) lib="capi/libmojotrees.so" ;;
esac

# $MOJOTREES_TARGET_FLAGS is deliberately unquoted; see build_target.sh.
# shellcheck disable=SC2086
pixi run mojo build --emit shared-lib $MOJOTREES_TARGET_FLAGS -I src -I capi \
    capi/mojotrees_capi.mojo -o "$lib"
echo "built $lib"
