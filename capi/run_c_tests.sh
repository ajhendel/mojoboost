#!/bin/sh
# Build the C ABI shared library, compile capi/test_capi.c against the
# public header, and run it. Skips cleanly, with status 0, when no C
# compiler is installed: the Mojo suite covers the same behavior, so a
# machine without a toolchain should not fail the build.
#
# Set MOJOTREES_CC to choose a compiler. Set MOJOTREES_LEAK_CHECK=1 to run
# the test under the platform leak checker instead of directly.
set -e
cd "$(dirname "$0")/.."

cc_bin="${MOJOTREES_CC:-cc}"
if ! command -v "$cc_bin" >/dev/null 2>&1; then
    echo "no C compiler ($cc_bin); skipping the C ABI tests"
    exit 0
fi

case "$(uname -s)" in
    Darwin) lib="capi/libmojotrees.dylib" ;;
    *) lib="capi/libmojotrees.so" ;;
esac

capi/build.sh

# The shared library links the Mojo runtime from the pixi environment, so
# the test binary needs that directory on its runtime search path.
prefix="$(pixi run printenv CONDA_PREFIX)"

"$cc_bin" -std=c99 -Wall -Wextra -Werror -O1 \
    -o capi/test_capi capi/test_capi.c \
    -Icapi "$lib" -lm \
    -Wl,-rpath,"$prefix/lib" -Wl,-rpath,"$(pwd)/capi"

if [ "${MOJOTREES_LEAK_CHECK:-0}" = "1" ]; then
    case "$(uname -s)" in
        Darwin) exec leaks --atExit -- ./capi/test_capi ;;
        *) exec valgrind --leak-check=full --error-exitcode=1 \
            ./capi/test_capi ;;
    esac
fi

exec ./capi/test_capi
