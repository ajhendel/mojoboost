#!/usr/bin/env python3
"""Can the repository's Python metadata produce a correct Linux wheel today?

    python3 packaging/linux/check_metadata_ready.py

NOT EXECUTED BY THIS LANE and never run. Standard library only, reads files as
text and TOML, imports nothing from the project, builds nothing.

The answer today is no, and the point of this script is that the answer is
checkable rather than remembered. `python/pyproject.toml` and `python/setup.py`
were written for a macOS wheel, and two of their settings do not merely fail to
help on Linux, they produce a *wrong artifact quietly*:

  * a wheel tagged macosx, built on Linux, on the wrong architecture
  * a wheel with the bundled runtime silently missing from it

Both files belong to another lane (Task 01). This script does not fix them and
must not: it reports, with the exact edit, and
packaging/linux/build_wheel_linux.sh refuses to build while a blocker stands.

Exit status is 0 when no blocker remains, 1 otherwise. Warnings do not fail.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT = ROOT / "python" / "pyproject.toml"
SETUP_PY = ROOT / "python" / "setup.py"
PIXI = ROOT / "pixi.toml"

# Where build_wheel_linux.sh stages the runtime closure. A package-data pattern
# has to cover these names, including the versioned ones, or setuptools drops
# them from the wheel without a word.
STAGED = (".libs/libFoo.so", ".libs/libFoo.so.1", ".libs/libFoo.so.1.2.3")


class Report:
    def __init__(self) -> None:
        self.blockers = 0
        self.warnings = 0

    def ok(self, cid: str, detail: str) -> None:
        print(f"ok    {cid}  {detail}")

    def blocker(self, cid: str, detail: str, fix: str) -> None:
        self.blockers += 1
        print(f"BLOCK {cid}  {detail}")
        for line in fix.strip().splitlines():
            print(f"        {line}")

    def warn(self, cid: str, detail: str, fix: str = "") -> None:
        self.warnings += 1
        print(f"warn  {cid}  {detail}")
        for line in fix.strip().splitlines():
            print(f"        {line}")


def _fnmatch_one(pattern: str, name: str) -> bool:
    """setuptools package-data globbing, close enough for these patterns.

    fnmatch is not used because its `*` crosses `/`, and package-data patterns
    are path globs where it does not. Only `*` and `?` appear in the patterns
    this project uses, so translate those and nothing else.
    """
    parts = pattern.split("/")
    target = name.split("/")
    if len(parts) != len(target):
        return False
    for pat, seg in zip(parts, target):
        rx = "".join(
            "[^/]*" if ch == "*" else "[^/]" if ch == "?" else re.escape(ch)
            for ch in pat
        )
        if not re.fullmatch(rx, seg):
            return False
    return True


def check_package_data(data: dict, rep: Report) -> None:
    """C1. Does package-data carry the bundled runtime into the wheel?"""
    patterns = (
        data.get("tool", {})
        .get("setuptools", {})
        .get("package-data", {})
        .get("mojotrees", [])
    )
    uncovered = [
        staged
        for staged in STAGED
        if not any(_fnmatch_one(p, staged) for p in patterns)
    ]
    if not uncovered:
        rep.ok("C1", f"package-data covers the staged runtime: {patterns}")
        return
    rep.blocker(
        "C1",
        f"package-data {patterns} does not match {uncovered}",
        """
The bundled MAX runtime would be dropped from the wheel and the failure would
appear as an ImportError on a user's machine, not here. Required edit, in
python/pyproject.toml (Task 01 owns this file):

    [tool.setuptools.package-data]
    mojotrees = ["*.so", ".dylibs/*.dylib", ".libs/*.so", ".libs/*.so.*"]

Both .libs patterns are needed: sonames carry version suffixes and "*.so" does
not match "libfoo.so.1".
""",
    )


def check_plat_name(text: str, rep: Report) -> None:
    """C2. Does setup.py hard-code a macOS platform tag?"""
    if "macosx" not in text:
        rep.ok("C2", "setup.py does not hard-code a macOS platform tag")
        return
    hits = [
        line.strip()
        for line in text.splitlines()
        if "macosx" in line
    ]
    rep.blocker(
        "C2",
        f"setup.py hard-codes a macOS platform tag: {hits}",
        """
Run unchanged on Linux this produces a wheel named ...-macosx_26_0_x86_64.whl:
a macOS tag, on a Linux binary, and pip on macOS would then accept it and fail
at import. build_wheel_linux.sh overrides it with an explicit --plat-name and
verifies the resulting filename, so a build is possible today, but the default
stays wrong for anyone who runs `python -m build` by hand.

Required edit, in python/setup.py (Task 01 owns this file): make the tag
platform-conditional rather than unconditional, for example

    import sys
    if sys.platform == "darwin":
        options = {"bdist_wheel": {"plat_name": "macosx_26_0_" + machine}}
    else:
        options = {}          # let --plat-name or the default decide

and keep the macOS deployment-target reasoning in the docstring where it is.
""",
    )


def check_patchelf(text: str, rep: Report) -> None:
    """C3. Is patchelf available to any pixi environment?"""
    if re.search(r"^\s*patchelf\s*=", text, re.MULTILINE):
        rep.ok("C3", "pixi.toml provides patchelf")
        return
    rep.blocker(
        "C3",
        "no pixi environment provides patchelf",
        """
patchelf is to a Linux wheel what install_name_tool is to a macOS one: without
it the extension keeps an absolute RPATH into the build machine's pixi
environment, which does not exist on any user's machine.

Required edit, in pixi.toml (owned by no lane in this round, so it goes through
the integration owner):

    [feature.pkg.target.linux-64.dependencies]
    patchelf = "*"

    [feature.pkg.target.linux-aarch64.dependencies]
    patchelf = "*"

Target-scoped so that macOS builds do not acquire a dependency they have no use
for. Adding it changes pixi.lock, which is the reason this is not a one-line
change anyone should make casually mid-round.
""",
    )


def check_classifiers(data: dict, rep: Report) -> None:
    """C4. Do the classifiers claim Linux? (warning, not a blocker)"""
    classifiers = data.get("project", {}).get("classifiers", [])
    linux = [c for c in classifiers if "Linux" in c or "POSIX" in c]
    if linux:
        rep.ok("C4", f"classifiers mention Linux: {linux}")
        return
    rep.warn(
        "C4",
        f"classifiers claim {[c for c in classifiers if c.startswith('Operating System')]} only",
        """
Cosmetic, and it should stay wrong until a Linux wheel is real. Add

    "Operating System :: POSIX :: Linux",

to python/pyproject.toml (Task 01) at the same time as the first Linux artifact
is published, not before. A classifier is a claim like any other.
""",
    )


def check_requires_python(data: dict, rep: Report) -> None:
    """C5. Is requires-python consistent with the cp314-only toolchain?"""
    req = data.get("project", {}).get("requires-python", "")
    if req.replace(" ", "") == ">=3.14":
        rep.ok("C5", f"requires-python {req!r} matches the cp314 toolchain pin")
    else:
        rep.warn(
            "C5",
            f"requires-python is {req!r}",
            """
The Linux wheel is tagged for whatever interpreter builds it, and max 26.5.0
pins python 3.14.*. If Task 05 lowers this bound, the Linux wheel tags do not
follow automatically: one wheel is built per interpreter series and the release
workflow's matrix has to grow a row for each.
""",
        )


def check_gitignore(rep: Report) -> None:
    """C6. Artifacts are ignored, which is right and has a consequence."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if "python/dist/" in ignore:
        rep.warn(
            "C6",
            "python/dist/ is gitignored, so wheels, .sha256 files and "
            "provenance sidecars are untracked",
            """
Correct for the repository, and it means the three files travel together only
if the release workflow uploads them together. A wheel published without its
provenance sidecar cannot answer which toolchain built it or whether an
accelerator was visible at compile time, and those facts are unrecoverable
afterwards.
""",
        )
    else:
        rep.ok("C6", "python/dist/ is not gitignored")


def main() -> int:
    print(f"metadata readiness for a Linux wheel, checked against {ROOT}")
    rep = Report()

    try:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"BLOCK C0  cannot read {PYPROJECT}: {exc}")
        return 1

    check_package_data(data, rep)
    check_plat_name(SETUP_PY.read_text(encoding="utf-8"), rep)
    check_patchelf(PIXI.read_text(encoding="utf-8"), rep)
    check_classifiers(data, rep)
    check_requires_python(data, rep)
    check_gitignore(rep)

    print(
        f"\n{rep.blockers} blocker(s), {rep.warnings} warning(s). "
        + ("Linux wheel builds are blocked." if rep.blockers else "No blockers.")
    )
    return 1 if rep.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
