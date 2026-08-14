#!/bin/sh
# Build the command line tool into cli/mojoboost.
# Run from anywhere; requires pixi.
set -e
cd "$(dirname "$0")/.."
pixi run mojo build -I src cli/mojoboost_cli.mojo -o cli/mojoboost
echo "built cli/mojoboost"
