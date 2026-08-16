#!/usr/bin/env bash
# Build a self-contained macOS wheel for the mojotrees Python API.
# Run via: pixi run build-wheel   (uses the pkg pixi environment)
#
# The Mojo-built extension links four MAX runtime dylibs via @rpath. This
# script bundles them into the package (delocate-style) and adds an
# @loader_path rpath so the wheel works without a Mojo/MAX installation.
set -euo pipefail
cd "$(dirname "$0")/.."

bindings/build.sh

# The extension is compiled for the baseline CPU that packaging/build_target.sh
# pins, not for the build machine. This is the check that the pin held, run on
# the object rather than on the wheel because it is the cheapest place to catch
# it and because the failure it catches (a native build) is invisible in every
# Mach-O header. packaging/macos/inspect_wheel.py runs the same check on the
# finished wheel as C14, covering the bundled runtime as well.
python3 packaging/isa_baseline.py python/mojotrees/_mojotrees.so

PKG=python/mojotrees
LIBS=(
    libKGENCompilerRTShared
    libAsyncRTMojoBindings
    libMSupportGlobals
    libAsyncRTRuntimeGlobals
)

rm -rf "$PKG/.dylibs" python/build python/dist python/mojotrees.egg-info
mkdir -p "$PKG/.dylibs"
for lib in "${LIBS[@]}"; do
    cp "$CONDA_PREFIX/lib/$lib.dylib" "$PKG/.dylibs/"
done

# The fresh .so carries an absolute rpath into this checkout's pixi env;
# replace it with the bundled dir so no local path ships in the wheel.
# The bundled dylibs sit next to the .so, so dev runs keep working too.
# install_name_tool invalidates the code signature on arm64, so
# everything is re-signed.
install_name_tool -rpath "$CONDA_PREFIX/lib" "@loader_path/.dylibs" "$PKG/_mojotrees.so"
codesign --force --sign - "$PKG/_mojotrees.so"
for lib in "${LIBS[@]}"; do
    codesign --force --sign - "$PKG/.dylibs/$lib.dylib"
done

cp LICENSE python/LICENSE
(cd python && python -m build --wheel --no-isolation)
ls -l python/dist/
