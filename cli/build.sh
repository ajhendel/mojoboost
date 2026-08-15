#!/bin/sh
# Build the command line tool into cli/mojotrees.
# Run from anywhere; requires pixi.
set -e
cd "$(dirname "$0")/.."
pixi run mojo build -I src cli/mojotrees_cli.mojo -o cli/mojotrees
echo "built cli/mojotrees"
