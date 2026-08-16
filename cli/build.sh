#!/bin/sh
# Build the command line tool into cli/mojotrees.
# Run from anywhere; requires pixi.
#
# The CPU target comes from packaging/build_target.sh, shared with
# bindings/build.sh and capi/build.sh. It defaults to a portable baseline
# rather than the host; that file says why, and what it costs. A CLI binary
# is the artifact a user is most likely to copy from one machine to another,
# so a host-tuned build of it fails in the most confusing way available.
set -e
cd "$(dirname "$0")/.."
. packaging/build_target.sh
mojotrees_resolve_target

# $MOJOTREES_TARGET_FLAGS is deliberately unquoted; see build_target.sh.
# shellcheck disable=SC2086
pixi run mojo build $MOJOTREES_TARGET_FLAGS -I src \
    cli/mojotrees_cli.mojo -o cli/mojotrees
echo "built cli/mojotrees"
