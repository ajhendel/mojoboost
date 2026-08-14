"""Wheel configuration for the prebuilt Mojo extension.

setuptools does not compile anything here; the extension is built by
`bindings/build.sh` (Mojo) before packaging. has_ext_modules is overridden
so the wheel is tagged for the interpreter and platform instead of
py3-none-any, and the platform tag is pinned to the Mojo toolchain's
macOS deployment target (minos in the .so's LC_BUILD_VERSION), which is
what actually constrains where the binary runs.
"""

import platform

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    options={
        "bdist_wheel": {
            "plat_name": "macosx_26_0_" + platform.machine().lower(),
        },
    },
)
