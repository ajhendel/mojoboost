#!/usr/bin/env python3
"""Check the release matrix against itself and against the repository.

    python3 packaging/matrix/validate_matrix.py

`packaging/matrix/platform_matrix.toml` and
`packaging/matrix/accelerators/index.toml` make claims. This script checks the
ones a script can check, so that a status cannot quietly become false:

1. every status uses the documented vocabulary, in both files
2. every repository path either file cites exists
3. no target or device claims more than its evidence supports, which is the
   rule the whole directory is built around: `validated` requires an evidence
   file, and an accelerator claiming `validated` requires all four recorded
   steps to pass
4. expected wheel filenames agree with their own tags, and the tags agree with
   the target's os, arch, and interpreter
5. exactly one interpreter is the build target and it is the one pixi.lock
   pins, every runnable interpreter is at or above the toolchain floor, and
   none of them claims to work without the GIL
6. docs/PLATFORM_MATRIX.md names every target in the metadata, so the prose
   cannot drift away from the data

Standard library only. It builds nothing, installs nothing, and imports no part
of mojoboost, so it runs on a bare checkout and in CPU-only CI in well under a
second. It deliberately does NOT check whether a wheel exists: a matrix is a
statement of intent plus evidence, and it has to be checkable before any
artifact is built.

Exit status is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MATRIX = ROOT / "packaging" / "matrix" / "platform_matrix.toml"
ACCEL = ROOT / "packaging" / "matrix" / "accelerators" / "index.toml"
DOC = ROOT / "docs" / "PLATFORM_MATRIX.md"
LOCK = ROOT / "pixi.lock"

TARGET_KEYS = {
    "id", "os", "arch", "python", "artifact", "wheel_tag", "filename",
    "status", "build_host", "builder", "installer_floor", "bundled_dylibs",
    "smoke", "evidence",
}
DEVICE_KEYS = {
    "id", "vendor", "chip", "api", "status", "correctness", "determinism",
    "phase_timings", "profiler", "record", "template",
}
STEPS = ("correctness", "determinism", "phase_timings", "profiler")
STEP_WORDS = {"pass", "fail", "partial", "not-run"}

# Fields whose value is a repository path when it is not empty.
PATH_FIELDS = ("smoke", "evidence", "record", "template")

failures: list[str] = []


def fail(where: str, message: str) -> None:
    failures.append(f"{where}: {message}")


def check_vocabulary(data: dict, rows: list[dict], kind: str, where: str) -> None:
    allowed = set(data.get("vocabulary", {}))
    if not allowed:
        fail(where, "no [vocabulary] table, so no status can be checked")
        return
    for row in rows:
        status = row.get("status")
        if status not in allowed:
            fail(f"{where}:{row.get('id', '?')}",
                 f"{kind} status {status!r} is not one of {sorted(allowed)}")


def check_paths(rows: list[dict], where: str) -> None:
    for row in rows:
        for field in PATH_FIELDS:
            value = row.get(field, "")
            if not value:
                continue
            # An evidence pointer may name a section inside a document.
            path = ROOT / value.split("#", 1)[0].split(",", 1)[0].strip()
            if not path.exists():
                fail(f"{where}:{row.get('id', '?')}",
                     f"{field} names {value!r}, which does not exist")


def check_target_evidence(targets: list[dict]) -> None:
    """The core rule. A target may not say `validated` on a promise."""
    for t in targets:
        tid = t.get("id", "?")
        missing = TARGET_KEYS - set(t)
        if missing:
            fail(f"target:{tid}", f"missing keys {sorted(missing)}")
        if t.get("status") == "validated" and not t.get("evidence", "").strip():
            fail(f"target:{tid}",
                 "status is `validated` with no evidence file. A platform is "
                 "validated when hardware ran the artifact and somebody wrote "
                 "down what happened, not when the matrix says so.")
        if t.get("artifact") == "wheel":
            for field in ("wheel_tag", "filename", "smoke"):
                if not t.get(field, "").strip():
                    fail(f"target:{tid}", f"artifact is a wheel but {field} is empty")
        else:
            if t.get("wheel_tag") or t.get("filename"):
                fail(f"target:{tid}",
                     "artifact is not a wheel but a wheel tag or filename is set")


def check_tags(targets: list[dict], pythons: list[dict], matrix: dict) -> None:
    """A tag is a promise about where pip will install the artifact. Check that
    the promise matches the row that makes it."""
    # Interpreters the project claims the code runs on. A wheel tag naming
    # anything else is a promise the matrix does not back.
    build_targets = {p["tag"] for p in pythons if p["status"] != "unsupported"}
    for t in targets:
        tid = t.get("id", "?")
        tag = t.get("wheel_tag", "")
        if not tag:
            continue
        parts = tag.split("-")
        if len(parts) != 3:
            fail(f"target:{tid}", f"wheel tag {tag!r} is not interpreter-abi-platform")
            continue
        python, abi, plat = parts

        if python not in build_targets:
            fail(f"target:{tid}",
                 f"tag names interpreter {python!r}, which is not a supported "
                 f"build target ({sorted(build_targets)})")
        if python != t.get("python"):
            fail(f"target:{tid}",
                 f"tag interpreter {python!r} does not match python {t.get('python')!r}")
        if abi != python:
            fail(f"target:{tid}",
                 f"abi tag {abi!r} does not match interpreter {python!r}; this "
                 "project ships no abi3 and no cross-ABI wheel")

        arch = t.get("arch", "")
        if not plat.endswith(arch):
            fail(f"target:{tid}", f"platform tag {plat!r} does not end in arch {arch!r}")
        os_name = t.get("os")
        if os_name == "macos" and not plat.startswith("macosx_"):
            fail(f"target:{tid}", f"macOS target with platform tag {plat!r}")
        if os_name == "linux" and not plat.startswith("manylinux_"):
            fail(f"target:{tid}",
                 f"linux target with platform tag {plat!r}; publish manylinux, "
                 "not a bare linux_ tag, which PyPI rejects and which promises "
                 "nothing about glibc")

        expected = f"{matrix['project']}-{matrix['version']}-{tag}.whl"
        if t.get("filename") != expected:
            fail(f"target:{tid}",
                 f"filename {t.get('filename')!r} does not follow from the tag; "
                 f"expected {expected!r}")


def _minor(version: str) -> tuple[int, int] | None:
    """(major, minor) from a row's `version`, or None when it is prose.

    Rows like "3.9 and earlier" and "any" are deliberately not parsed to a
    number: they describe a range or a mechanism, not one interpreter.
    """
    m = re.fullmatch(r"(\d+)\.(\d+)", version.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_python_rows(pythons: list[dict], toolchain: dict) -> None:
    """Two separate invariants, which this check used to conflate.

    Exactly one interpreter BUILDS the extension, because setuptools tags a
    wheel for whichever interpreter ran the build and the toolchain pin picks
    that one. Any number of interpreters may RUN the result, because the
    extension links no libpython and resolves CPython entry points by name at
    runtime. Requiring one row of each kind, as an earlier version of this
    function did, made a true matrix fail: 3.10 through 3.14 all run the
    suite. See docs/PYTHON_SUPPORT.md.
    """
    runnable = [p for p in pythons if p["status"] != "unsupported"]
    if not runnable:
        fail("python", "no interpreter row is anything but `unsupported`, so "
                       "the matrix says nothing runs at all")

    builders = [p for p in pythons if p.get("build_target")]
    if len(builders) != 1:
        fail("python", f"expected exactly one row with `build_target = true`, "
                       f"found {[p['tag'] for p in builders]}. The toolchain pin "
                       "selects one interpreter to compile against, and the wheel "
                       "carries its tag.")
    for row in builders:
        pin = toolchain.get("python_pin", "")
        if not pin.startswith(row["version"]):
            fail("python", f"build target {row['version']} does not match the "
                           f"toolchain pin {pin!r}")
        if row["status"] == "unsupported":
            fail("python", f"build target {row['tag']} is marked unsupported")

    floor = _minor(toolchain.get("python_toolchain_floor", ""))
    for row in runnable:
        if row.get("gil") != "required" or not toolchain.get("requires_gil"):
            fail("python", f"the toolchain depends on python-gil, so runnable "
                           f"interpreter {row['tag']} must say the GIL is required")
        version = _minor(row.get("version", ""))
        if floor and version and version < floor:
            fail("python", f"interpreter {row['tag']} is runnable at "
                           f"{row['version']}, below the toolchain floor "
                           f"{toolchain['python_toolchain_floor']}. One of the two "
                           "is wrong and neither may be guessed at.")


def check_lock(toolchain: dict) -> None:
    """The toolchain block is derived from pixi.lock. Check it still is."""
    if not LOCK.exists():
        fail("toolchain", "pixi.lock is missing, so nothing here is derived")
        return
    text = LOCK.read_text()
    pin = toolchain.get("python_pin", "").rstrip(".*")
    if f"python-{pin}." not in text:
        fail("toolchain", f"pixi.lock has no python {pin} package; the matrix's "
                          "python_pin no longer follows from the lock")
    build = toolchain.get("max_build_string", "")
    if f"max-{toolchain.get('max')}-{build}" not in text:
        fail("toolchain", f"pixi.lock has no max-{toolchain.get('max')}-{build}; "
                          "re-derive the toolchain block from the lock")
    if f"mojo-{toolchain.get('mojo')}-" not in text:
        fail("toolchain", f"pixi.lock has no mojo-{toolchain.get('mojo')}")
    for platform in toolchain.get("pixi_platforms", []):
        if f"- name: {platform}" not in text:
            fail("toolchain", f"pixi.lock does not solve for {platform}")


def check_devices(accel: dict) -> None:
    devices = accel.get("device", [])
    if not devices:
        fail("accelerators", "no device rows")
    seen = set()
    for d in devices:
        did = d.get("id", "?")
        if did in seen:
            fail(f"device:{did}", "duplicate id")
        seen.add(did)
        missing = DEVICE_KEYS - set(d)
        if missing:
            fail(f"device:{did}", f"missing keys {sorted(missing)}")
        for step in STEPS:
            if d.get(step) not in STEP_WORDS:
                fail(f"device:{did}",
                     f"{step} is {d.get(step)!r}, not one of {sorted(STEP_WORDS)}")
        status = d.get("status")
        recorded = bool(d.get("record", "").strip())
        if status != "not-run" and not recorded:
            fail(f"device:{did}",
                 f"status is {status!r} with no record. Every device starts at "
                 "`not-run` and moves only when a record exists.")
        if status == "not-run" and any(d.get(s) != "not-run" for s in STEPS):
            fail(f"device:{did}",
                 "status is `not-run` but a step claims a result; one of the two "
                 "is wrong")
        if status == "validated":
            bad = [s for s in STEPS if d.get(s) != "pass"]
            if bad:
                fail(f"device:{did}",
                     f"claims `validated` while {bad} is not `pass`. Validated "
                     "means all four steps ran and passed on real hardware.")


def check_doc(targets: list[dict], devices: list[dict]) -> None:
    if not DOC.exists():
        fail("docs", f"{DOC.relative_to(ROOT)} is missing")
        return
    text = DOC.read_text()
    for t in targets:
        if t["id"] not in text:
            fail("docs", f"target {t['id']} is not mentioned in "
                         f"{DOC.relative_to(ROOT)}")
    for vendor in {d["vendor"] for d in devices}:
        if vendor not in text.lower():
            fail("docs", f"accelerator vendor {vendor} is not mentioned in "
                         f"{DOC.relative_to(ROOT)}")
    # The one sentence that must survive every edit of that document.
    if not re.search(r"no\s+NVIDIA\b.*\bAMD\b", text, re.I | re.S):
        fail("docs", "the document does not state that no NVIDIA or AMD device "
                     "has run this code. That statement is the point of the "
                     "document and it stays until it stops being true.")


def main() -> int:
    matrix = tomllib.loads(MATRIX.read_text())
    accel = tomllib.loads(ACCEL.read_text())
    targets = matrix.get("target", [])
    pythons = matrix.get("python", [])
    sources = matrix.get("source_install", [])
    devices = accel.get("device", [])

    ids = [t["id"] for t in targets]
    if len(ids) != len(set(ids)):
        fail("target", "duplicate target ids")

    check_vocabulary(matrix, targets, "target", "matrix")
    check_vocabulary(matrix, pythons, "python", "matrix")
    check_vocabulary(matrix, sources, "source_install", "matrix")
    check_vocabulary(accel, devices, "device", "accelerators")
    check_paths(targets, "matrix")
    check_paths(devices, "accelerators")
    check_target_evidence(targets)
    check_tags(targets, pythons, matrix)
    check_python_rows(pythons, matrix.get("toolchain", {}))
    check_lock(matrix.get("toolchain", {}))
    check_devices(accel)
    check_doc(targets, devices)

    index = matrix.get("accelerators", {}).get("index", "")
    if not (ROOT / index).exists():
        fail("matrix", f"accelerators.index names {index!r}, which does not exist")

    if failures:
        print(f"{len(failures)} problem(s):\n")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"release matrix ok: {len(targets)} targets, {len(sources)} source "
          f"installs, {len(devices)} devices, "
          f"{sum(1 for d in devices if d['status'] != 'not-run')} with any "
          f"recorded evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
