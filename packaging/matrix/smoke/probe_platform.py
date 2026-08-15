#!/usr/bin/env python3
"""Print the platform facts a release record has to contain.

NOT EXECUTED BY ANY TASK OR WORKFLOW. It is a fixture, meant to be run by hand
on a target machine, inside the venv the wheel was installed into, from a
directory that is not the source tree:

    python packaging/matrix/smoke/probe_platform.py

Standard library only, and it imports mojotrees only if mojotrees is installed,
so it also runs on a machine that is being checked *before* an install.

Everything printed here is something a person would otherwise retype from
memory into a record, which is where the errors come from. Two facts it exists
to make hard to fake:

1. **Where the import came from.** A wheel test that accidentally imports the
   source checkout tests the checkout. The `origin` line is how that is caught.
2. **Which wheel tags this interpreter accepts.** The platform tag is the whole
   macOS install story. If pip would refuse the wheel on this machine, the tag
   list printed here says so directly, without needing pip.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    section("interpreter")
    print(f"executable:   {sys.executable}")
    print(f"version:      {sys.version.splitlines()[0]}")
    print(f"implementation: {platform.python_implementation()}")
    # The free-threaded build is a different ABI and a different wheel tag.
    # max 26.5.0 depends on python-gil, so a True here means this interpreter
    # cannot be a mojotrees target at all.
    gil_disabled = getattr(sys, "_is_gil_enabled", None)
    if gil_disabled is None:
        print("free-threaded: no (interpreter has no GIL toggle)")
    else:
        print(f"free-threaded: {'yes' if not gil_disabled() else 'no'}")
    print(f"abi tag:      {sysconfig.get_config_var('SOABI')}")
    print(f"ext suffix:   {sysconfig.get_config_var('EXT_SUFFIX')}")

    section("platform")
    print(f"system:       {platform.system()} {platform.release()}")
    print(f"machine:      {platform.machine()}")
    print(f"sysconfig:    {sysconfig.get_platform()}")
    if sys.platform == "darwin":
        print(f"mac_ver:      {platform.mac_ver()[0]}")
        # The deployment target the interpreter itself was built for. It is not
        # the wheel's floor, but a mismatch here explains a surprising refusal.
        print(f"MACOSX_DEPLOYMENT_TARGET: "
              f"{sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')}")
    if sys.platform.startswith("linux"):
        try:
            print(f"glibc:        {os.confstr('CS_GNU_LIBC_VERSION')}")
        except (ValueError, OSError):
            print("glibc:        unavailable")

    section("wheel tags this interpreter accepts")
    # packaging.tags is not in the standard library, so this prints the two
    # pieces the tag is built from and leaves the join to the reader rather
    # than pulling in a dependency the bare install must not have.
    print("Compare the wheel filename against these by hand:")
    print(f"  interpreter/abi: cp{sys.version_info.major}{sys.version_info.minor}")
    print(f"  platform:        {sysconfig.get_platform().replace('-', '_').replace('.', '_')}")
    print("A wheel whose platform tag names a macOS newer than mac_ver above,")
    print("or an architecture other than machine above, is refused by pip with")
    print("'not a supported wheel on this platform'. That is the correct")
    print("failure, and it is a tag problem, not an install problem.")

    section("mojotrees")
    try:
        import mojotrees
    except ImportError as exc:
        print(f"not installed: {exc}")
        return 0

    origin = getattr(mojotrees, "__file__", "unknown")
    print(f"version:      {getattr(mojotrees, '__version__', 'unknown')}")
    print(f"origin:       {origin}")
    if "site-packages" not in str(origin):
        print("WARNING: not imported from site-packages. Whatever this run")
        print("proves, it does not prove anything about the wheel.")

    pkg_dir = os.path.dirname(str(origin))
    print("package contents:")
    for root, _dirs, files in os.walk(pkg_dir):
        rel = os.path.relpath(root, pkg_dir)
        for name in sorted(files):
            if name.endswith((".so", ".dylib", ".dll")):
                path = os.path.join(root, name)
                size = os.path.getsize(path)
                shown = name if rel == "." else os.path.join(rel, name)
                print(f"  {shown}  {size} bytes")

    section("mojotrees environment overrides in effect")
    # These change what the library does. A record taken without them noted is
    # a record of an unknown configuration.
    any_set = False
    for key, value in sorted(os.environ.items()):
        if key.startswith("MOJOTREES_"):
            print(f"  {key}={value}")
            any_set = True
    if not any_set:
        print("  none set (defaults)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
