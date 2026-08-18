"""Wheel shape for the prebuilt Mojo extension.

setuptools compiles nothing here. `bindings/build.sh` builds the extension
with the Mojo toolchain and `packaging/build_wheel.sh` stages it, plus the
four MAX runtime dylibs it links through @rpath, into python/mojotrees/
before this file ever runs. There is no source build, and there is no way
to make one that works without the Mojo toolchain, which is why
packaging/matrix/platform_matrix.toml publishes no sdist.

All distribution metadata is in pyproject.toml. This file carries only the
two facts PEP 621 has no field for, and nothing else.

1. The wheel is not pure Python. `has_ext_modules` is forced True so the
   wheel is tagged for the interpreter and ABI, `cp314-cp314`, instead of
   `py3-none-any`. The extension is not built against the limited API, so
   one wheel serves exactly one CPython minor version. That is the
   `abi3 -> unsupported` row of packaging/matrix/platform_matrix.toml.

2. On macOS the platform tag is pinned to the Mojo compile step's
   deployment target, not to the OS of the machine doing the build. The
   `minos` in the extension's LC_BUILD_VERSION is what actually decides
   where the binary loads, and the tag is what pip compares against
   before it tries. A tag that disagrees with the binary is a published
   lie in one of two directions: too low and the wheel installs onto Macs
   where it cannot load, too high and it is refused by Macs that could
   have run it.

Both of those are release-contract claims, so the procedure that consumes
them is docs/PYPI_RELEASE.md.
"""

import os
import platform
import sys

from setuptools import setup
from setuptools.dist import Distribution

# The macOS deployment target the Mojo compile step produces, as it appears
# in a wheel tag. This must equal the `minos` that
#
#     otool -l python/mojotrees/_mojotrees.so
#
# reports for LC_BUILD_VERSION. Nothing here reads the Mach-O header, so
# the two are kept in step by the release procedure, not by this file.
DEFAULT_MACOS_TARGET = "26.0"

# Override for the lowered-floor wheel, the `macos-arm64-cp314-lowered`
# target of packaging/matrix/platform_matrix.toml. Set it only after
# otool confirms the rebuilt extension actually carries the lower floor;
# lowering the tag alone produces a wheel that installs and then fails to
# import.
TARGET_ENV_VAR = "MOJOTREES_MACOS_DEPLOYMENT_TARGET"

# MACOSX_DEPLOYMENT_TARGET is deliberately not consulted. conda-style
# environments, which is what pixi gives this build, export it for their
# own compilers at values unrelated to what the Mojo toolchain emitted.
# Inheriting it would silently tag a wheel with a floor its binary does
# not honor, and that failure only shows up on a user's machine.
#
# packaging/macos/build_release_wheel.sh sets both variables together and
# for different jobs: MACOSX_DEPLOYMENT_TARGET asks the Mojo compile step
# to emit a lower minos, and the variable above tells this file what to
# write into the tag. Setting only the first gives a lowered binary with
# a 26.0 tag; setting only the second gives a published lie. Neither
# variable is a result: check C1 of packaging/macos/inspect_wheel.py is
# what compares the tag against the binary afterwards.


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools >= 70 vendors it
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    """`bdist_wheel` that tags the interpreter `py3` rather than `cp3xx`.

    A `python_tag` in the options dict is not enough. `has_ext_modules`
    returning True makes `bdist_wheel` derive the implementation tag from the
    running interpreter, and it overwrites the option. Overriding `get_tag` is
    the only place the decision is actually made.

    The platform half is deliberately left as the base class computed it, so
    `--plat-name` and `macos_plat_name` above keep working. Only the
    interpreter and ABI halves are widened.
    """

    def get_tag(self):
        _, _, plat = _bdist_wheel.get_tag(self)
        return ("py3", "none", plat)


def macos_plat_name():
    """The macOS wheel platform tag, or a hard stop for an arch that has
    no artifact."""
    arch = platform.machine().lower()
    if arch != "arm64":
        raise SystemExit(
            "mojotrees builds no macOS wheel for {!r}. The `macos-x86_64` "
            "target of packaging/matrix/platform_matrix.toml is "
            "`unsupported`: pixi.toml declares no osx-64 platform and the "
            "pinned channel ships no Intel macOS toolchain, so there is "
            "nothing to build with and universal2 has no x86_64 half. If "
            "this is Apple silicon, the build is running under Rosetta; "
            "use a native arm64 interpreter.".format(arch)
        )
    target = os.environ.get(TARGET_ENV_VAR, DEFAULT_MACOS_TARGET)
    return "macosx_{}_{}".format(target.replace(".", "_"), arch)


# On anything other than macOS the tag is left alone. bdist_wheel then
# derives it from the interpreter, or takes it from an explicit
# --plat-name, which is what packaging/linux/build_wheel_linux.sh passes
# after it has staged the ELF closure and set RPATH. Without this
# condition that script's override would be fighting a hardcoded macOS
# tag, and a hand-run `python -m build` on Linux would produce a wheel
# named macosx_26_0_x86_64 containing an ELF object: a file pip on macOS
# would accept and then fail to import.
#
# The bare `linux_x86_64` tag that results from no override is refused by
# PyPI, which is correct. A Linux wheel becomes publishable by being
# repaired into a manylinux tag, not by being named one.
# THE INTERPRETER TAG, AND WHY IT IS `py3` RATHER THAN `cp3xx`.
#
# `bdist_wheel` derives the interpreter tag from whichever CPython is running
# the build, so a build on 3.14 emitted `cp314` and the wheel became invisible
# to every user on 3.10 through 3.13. **Nobody chose 3.14.** Nothing in this
# project pinned an interpreter, so pixi's solver took the newest available,
# the extension was compiled against it, and the tag recorded the accident.
#
# `requires-python` cannot fix it and already does not: `pyproject.toml`
# declares `>=3.10`, correctly, and its own comment gives the reason that buys
# nothing. **pip matches the wheel TAG before it consults `requires-python`.**
# The tag is the gate.
#
# AND A BUILD MATRIX IS NOT NEEDED, WHICH IS THE USEFUL PART.
# `docs/PYTHON_SUPPORT.md` section 7 establishes it and `otool -L` confirms it:
# this extension LINKS NO LIBPYTHON. It resolves CPython by name at runtime.
# abi3 exists so that one binary can serve many interpreters DESPITE being
# linked against libpython, so it is not the mechanism this needs; a binary
# that was never linked can already serve them. Section 4 records the
# measurement, one artifact importing, fitting, predicting and round-tripping
# on 3.10 through 3.14, run M1a.
#
# So the honest tag is the one that admits every interpreter the artifact was
# proven on. `py3-none-<platform>` says exactly that: any Python 3, this
# platform, not pure Python. The platform half stays specific because the
# object genuinely is architecture and OS bound, which is the whole reason
# `macos_plat_name` above exists.
#
# WHAT WOULD FALSIFY THIS. An interpreter in 3.10 through 3.14 on which the
# same artifact fails to import or produces a different prediction. That is
# what `docs/PYTHON_SUPPORT.md`'s `proven` label rests on, and if a future
# CPython breaks the by-name resolution this tag becomes a lie and the floor
# has to move. It is a claim with a date on it, like every other "cannot" in
# this repository.
options = {}
if sys.platform == "darwin":
    options["bdist_wheel"] = {"plat_name": macos_plat_name()}

setup(
    distclass=BinaryDistribution,
    options=options,
    cmdclass={"bdist_wheel": bdist_wheel},
)
