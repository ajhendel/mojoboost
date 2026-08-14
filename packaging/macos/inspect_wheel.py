#!/usr/bin/env python3
"""Describe a macOS arm64 mojoboost wheel, and check the release-only rules.

    python3 packaging/macos/inspect_wheel.py python/dist/<wheel> [--json out.json]
                                             [--report-only]

NOT EXECUTED. This script has never been run: it needs a wheel, and no wheel has
been built by this lane. Every number it will print is a future number.

Standard library only, and it reads the wheel as a zip, so it never installs or
imports what it is inspecting and it runs on a bare checkout.

Relationship to packaging/matrix/validate_artifact.py
----------------------------------------------------
That script is the matrix contract: does this wheel match a target that
platform_matrix.toml declares. Its rules are not re-implemented here, and its
Mach-O parser is imported rather than copied, so there is one such parser in the
repository and one place to fix it.

This script asks the four questions a release asks that a matrix check does not.

C1  Is the platform tag exactly right, not merely not-a-lie? Rule R5b passes
    when the extension's minos is at or below the tag's floor. A wheel built for
    macOS 12 and tagged macosx_26_0_arm64 satisfies R5b and is still wrong to
    publish, because pip then refuses it on every Mac between the two. python/
    setup.py writes the tag from an environment variable and states plainly that
    keeping it in step with the binary is the release procedure's job. This
    check is that job.

C5  Is the bundle minimal? "Bundle only the required runtime libraries" fails in
    two directions. A missing library is caught by R5e. A library that nothing
    in the wheel loads is caught nowhere, and shipping one means shipping bytes
    and a license obligation for no reason.

C7  What do the bundled libraries declare about themselves? Their LC_ID_DYLIB
    install name and their own LC_RPATH entries are load-time inputs that no
    existing check reads.

C8, C9, C11
    Is anything from the source tree in the artifact? Across every member of the
    zip, not only the Mach-O objects: caches, test data, scratch files,
    machine-specific paths, and secret-shaped strings.

Exit status is 0 when every check passes and 1 otherwise, unless --report-only
is given, which prints the same report and always exits 0. A check that cannot
run is a failure, not a skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR = ROOT / "packaging" / "matrix"

sys.path.insert(0, str(MATRIX_DIR))
try:
    # The repository's only Mach-O load command parser, and its list of strings
    # that must not survive a build. Imported, never copied.
    from validate_artifact import BUILD_HOST_MARKERS, macho_info
except ImportError as exc:  # pragma: no cover - a broken checkout, not a case
    raise SystemExit(
        f"cannot import packaging/matrix/validate_artifact.py ({exc}). "
        "This script is a companion to that one and does not duplicate it."
    )

CPU_TYPE_ARM64 = 0x0100000C
CPU_NAMES = {CPU_TYPE_ARM64: "arm64", 0x01000007: "x86_64"}
MH_MAGIC_64 = 0xFEEDFACF
LC_ID_DYLIB = 0x0D

WHEEL_NAME = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>.+)\.whl$"
)
MACOS_TAG = re.compile(r"^macosx_(?P<major>\d+)_(?P<minor>\d+)_(?P<arch>.+)$")

# Members that have no business in a distribution. Wider than the matrix
# script's list, because this one also covers what this repository in
# particular leaves lying around: Mojo scratch models, the serialization test
# temporaries, and the pixi environment.
FORBIDDEN_MEMBERS = re.compile(
    r"(^|/)("
    r"__pycache__/|\.pytest_cache/|\.pixi/|\.git/|\.github/|"
    r"build/|dist/|tests?/|bench/|docs/|"
    r"\.DS_Store$|\.test_.*\.tmp$|.*\.pyc$|.*\.mbst$|.*\.mojopkg$|"
    r"\.env$|.*\.pem$|.*\.key$|id_rsa.*"
    r")"
)

# Absolute dependency prefixes that are legitimate on macOS: they name the
# system's own libraries, which are present on every Mac and are not something a
# wheel may bundle.
SYSTEM_DYLIB_PREFIXES = ("/usr/lib/", "/System/Library/")

# A screen, not a proof. It catches a credential pasted into a file that got
# packaged; it cannot catch an encoded or unusual one. The real defense is that
# the wheel contains only python/mojoboost, which is why C9 exists.
SECRET_PATTERNS = (
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(rb"ghp_[A-Za-z0-9]{36}"), "GitHub personal access token"),
    (re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(rb"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{10,}"), "PyPI API token"),
    (re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(rb"AIza[0-9A-Za-z_\-]{35}"), "Google API key"),
)

# Extra machine-specific markers on top of the matrix script's set. The temp
# directory one matters here because packaging/test_wheel.sh and the clean
# install fixture both build under mktemp -d, and a path baked in from there is
# as unusable on a user's machine as a home directory is.
EXTRA_HOST_MARKERS = (
    b"/private/var/folders/",
    b"/var/folders/",
    b"/Applications/Xcode",
    b"/Library/Developer/CommandLineTools",
)


# ---------------------------------------------------------------------------
# Mach-O, the one piece validate_artifact.py does not report
# ---------------------------------------------------------------------------


def macho_install_name(blob: bytes) -> str | None:
    """LC_ID_DYLIB's name, or None when the object declares no install name.

    An executable or a bundle has no LC_ID_DYLIB; a dylib always does, and its
    value is what other objects will have recorded as their dependency. A
    bundled library whose install name is an absolute path into the build
    machine is a library that was copied without being repaired.
    """
    if len(blob) < 32 or struct.unpack_from("<I", blob, 0)[0] != MH_MAGIC_64:
        return None
    ncmds = struct.unpack_from("<I", blob, 16)[0]
    off = 32
    for _ in range(ncmds):
        if off + 8 > len(blob):
            return None
        cmd, cmdsize = struct.unpack_from("<II", blob, off)
        if cmdsize < 8 or off + cmdsize > len(blob):
            return None
        if cmd == LC_ID_DYLIB:
            (stroff,) = struct.unpack_from("<I", blob, off + 8)
            raw = blob[off + stroff : off + cmdsize]
            return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
        off += cmdsize
    return None


# ---------------------------------------------------------------------------
# Report accumulation
# ---------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failed = False

    def check(self, cid: str, ok: bool, detail: str, notes: list[str] | None = None) -> None:
        if not ok:
            self.failed = True
        self.checks.append({"id": cid, "ok": ok, "detail": detail, "notes": notes or []})
        print(f"{'ok  ' if ok else 'FAIL'} {cid}  {detail}")
        for note in notes or []:
            print(f"          {note}")

    def section(self, title: str) -> None:
        print(f"\n--- {title}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(zf: zipfile.ZipFile, names: list[str]) -> tuple[dict, str]:
    """The dist-info METADATA headers, as a dict of first occurrences."""
    for name in names:
        if name.endswith(".dist-info/METADATA"):
            out: dict[str, str] = {}
            for line in zf.read(name).decode("utf-8", "replace").splitlines():
                if not line.strip():
                    break
                key, _, value = line.partition(":")
                out.setdefault(key.strip(), value.strip())
            return out, name
    return {}, ""


def read_wheel_tags(zf: zipfile.ZipFile, names: list[str]) -> tuple[list[str], dict]:
    for name in names:
        if name.endswith(".dist-info/WHEEL"):
            tags, fields = [], {}
            for line in zf.read(name).decode("utf-8", "replace").splitlines():
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "Tag":
                    tags.append(value)
                elif key:
                    fields.setdefault(key, value)
            return tags, fields
    return [], {}


# ---------------------------------------------------------------------------
# The inspection
# ---------------------------------------------------------------------------


def inspect(path: Path, rep: Report) -> dict:
    data: dict = {"wheel": path.name, "sha256": sha256(path), "size": path.stat().st_size}

    rep.section("identity")
    print(f"file:    {path}")
    print(f"size:    {data['size']} bytes")
    print(f"sha256:  {data['sha256']}")

    match = WHEEL_NAME.match(path.name)
    if not match:
        rep.check("C0", False, "the filename is not a wheel name")
        return data
    tag = f"{match['python']}-{match['abi']}-{match['platform']}"
    data["tag"] = tag
    print(f"tag:     {tag}")

    plat = MACOS_TAG.match(match["platform"])
    if not plat:
        rep.check(
            "C0",
            False,
            f"platform tag {match['platform']!r} is not a macOS tag; this "
            "inspector is macOS only (packaging/linux/ owns the ELF side)",
        )
        return data
    tag_floor = (int(plat["major"]), int(plat["minor"]))
    tag_arch = plat["arch"]
    data["tag_floor"] = list(tag_floor)
    data["tag_arch"] = tag_arch

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkg = f"{match['name']}/"

        # --- structure --------------------------------------------------
        rep.section("members")
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            print(f"  {info.file_size:>12}  {info.filename}")
        data["members"] = [
            {"name": i.filename, "size": i.file_size, "compressed": i.compress_size}
            for i in zf.infolist()
        ]

        strays = sorted(n for n in names if FORBIDDEN_MEMBERS.search(n))
        rep.check(
            "C9",
            not strays,
            f"no source-tree members: {strays or 'none found'}",
            ["Caches, test data, scratch models, and VCS directories are one",
             "failure with several names: the wheel was packaged from a working",
             "directory rather than from a declared file list. python/",
             "pyproject.toml names the package and its package-data, so a stray",
             "here means something was copied into python/mojoboost/ and left."]
            if strays else [],
        )

        # --- metadata ---------------------------------------------------
        rep.section("metadata")
        meta, meta_name = read_metadata(zf, names)
        for key in ("Metadata-Version", "Name", "Version", "Summary",
                    "Requires-Python", "License-Expression", "License",
                    "License-File"):
            if key in meta:
                print(f"  {key}: {meta[key]}")
        data["metadata"] = meta

        required_fields = ["Name", "Version", "Requires-Python", "Summary"]
        missing = [k for k in required_fields if not meta.get(k)]
        has_license = bool(meta.get("License-Expression") or meta.get("License"))
        rep.check(
            "C10a",
            not missing and has_license and bool(meta_name),
            "package metadata present: "
            + (f"missing {missing}" if missing else "Name, Version, "
               "Requires-Python, Summary")
            + ("" if has_license else "; no License or License-Expression"),
        )

        license_members = [
            n for n in names
            if re.search(r"\.dist-info/(licenses/)?(LICENSE|COPYING|NOTICE)", n)
        ]
        sizes = {n: zf.getinfo(n).file_size for n in license_members}
        rep.check(
            "C10b",
            bool(license_members) and all(v > 0 for v in sizes.values()),
            f"license file shipped: {sizes or 'NONE'}",
            [] if license_members else [
                "python/pyproject.toml declares license-files = [\"LICENSE\"],"
                " and packaging/build_wheel.sh copies LICENSE into python/"
                " before the build. A wheel without it means that copy did not"
                " happen."
            ],
        )

        wheel_tags, wheel_fields = read_wheel_tags(zf, names)
        print(f"  WHEEL Tag: {wheel_tags or 'MISSING'}")
        print(f"  Root-Is-Purelib: {wheel_fields.get('Root-Is-Purelib', 'MISSING')}")
        rep.check(
            "C13",
            wheel_tags == [tag] and wheel_fields.get("Root-Is-Purelib") == "false",
            f"WHEEL agrees with the filename: tags {wheel_tags}, "
            f"Root-Is-Purelib {wheel_fields.get('Root-Is-Purelib', 'MISSING')}",
            ["A purelib wheel installs into the wrong directory and carries a"
             " tag that promises portability the extension does not have."]
            if wheel_fields.get("Root-Is-Purelib") != "false" else [],
        )

        # --- Mach-O -----------------------------------------------------
        rep.section("Mach-O objects")
        objects = sorted(
            n for n in names
            if n.startswith(pkg) and n.endswith((".so", ".dylib"))
        )
        ext_name = f"{pkg}_mojoboost.so"
        bundled = {
            Path(n).name: n
            for n in names
            if n.startswith(f"{pkg}.dylibs/") and n.endswith(".dylib")
        }

        parsed: dict[str, dict] = {}
        blobs: dict[str, bytes] = {}
        for name in objects:
            blob = zf.read(name)
            blobs[name] = blob
            info = macho_info(blob)
            info["install_name"] = macho_install_name(blob)
            parsed[name] = info
            short = name[len(pkg):]
            if "error" in info:
                print(f"  {short}: {info['error']}")
                continue
            print(f"  {short}")
            print(f"      arch:         {CPU_NAMES.get(info['cputype'], hex(info['cputype']))}")
            print(f"      minos:        {fmt_version(info['minos'])}"
                  f"   sdk: {fmt_version(info['sdk'])}")
            print(f"      signed:       {info['signed']}")
            print(f"      install name: {info['install_name'] or '(none)'}")
            print(f"      rpaths:       {info['rpaths'] or '(none)'}")
            for dep in info["dylibs"]:
                kind = classify_dep(dep, bundled)
                print(f"      needs:        {dep}   [{kind}]")
        data["macho"] = {
            k[len(pkg):]: {kk: vv for kk, vv in v.items()} for k, v in parsed.items()
        }

        broken = [n for n, i in parsed.items() if "error" in i]
        if broken:
            rep.check("C2", False,
                      f"unparseable Mach-O objects: {[b[len(pkg):] for b in broken]}")
            return data

        # C1. The tag is exactly the extension's floor.
        ext = parsed.get(ext_name)
        if ext is None:
            rep.check("C1", False,
                      f"no extension module at {ext_name}; found {objects or 'nothing'}")
        else:
            minos = ext["minos"]
            rep.check(
                "C1",
                minos is not None and tuple(minos[:2]) == tag_floor,
                f"platform tag floor {tag_floor[0]}.{tag_floor[1]} equals the "
                f"extension's minos {fmt_version(minos)}",
                [] if minos is not None and tuple(minos[:2]) == tag_floor else [
                    "A tag below the binary's floor installs onto Macs where the",
                    "extension cannot load. A tag above it is refused by Macs that",
                    "could have run it. packaging/matrix/validate_artifact.py rule",
                    "R5b only rejects the first of those, which is why this check",
                    "is equality. The tag comes from",
                    "MOJOBOOST_MACOS_DEPLOYMENT_TARGET in python/setup.py and the",
                    "binary comes from the Mojo compile step; set both together",
                    "(packaging/macos/build_release_wheel.sh) or neither.",
                ],
            )

        # C2. One architecture, the one on the label.
        want_cpu = {"arm64": CPU_TYPE_ARM64}.get(tag_arch)
        wrong = [n[len(pkg):] for n, i in parsed.items() if i["cputype"] != want_cpu]
        rep.check("C2", want_cpu is not None and not wrong,
                  f"every object is {tag_arch}: {wrong or 'yes'}")

        # C3. Signed. install_name_tool invalidates a signature and an object
        # with an invalid one does not load on Apple silicon at all.
        unsigned = [n[len(pkg):] for n, i in parsed.items() if not i["signed"]]
        rep.check("C3", not unsigned,
                  f"every object carries a code signature: {unsigned or 'yes'}",
                  ["packaging/build_wheel.sh re-signs ad-hoc after rewriting the",
                   "rpath. An unsigned object here means that step was skipped or",
                   "the object was rewritten afterwards."] if unsigned else [])

        # C4 and C5. The dependency closure, both directions.
        required: set[str] = set()
        unresolved: list[str] = []
        for name, info in parsed.items():
            for dep in info["dylibs"]:
                base = Path(dep).name
                if dep.startswith("@"):
                    if base in bundled:
                        required.add(base)
                    else:
                        unresolved.append(f"{name[len(pkg):]} needs {dep}")
        rep.check("C4", not unresolved,
                  f"every @rpath or @loader_path dependency is bundled: "
                  f"{unresolved or 'yes'}")

        unused = sorted(set(bundled) - required)
        rep.check(
            "C5",
            not unused,
            f"every bundled library is required by something in the wheel: "
            f"{unused or 'yes'}",
            ["Bundle only what is loaded. An unrequired library is bytes the",
             "user downloads and a redistribution obligation the project takes",
             "on for nothing. Check it against the bundled_dylibs list of the",
             "matching target in packaging/matrix/platform_matrix.toml before",
             "removing it: the list may be what is wrong."] if unused else [],
        )

        # C6. No absolute rpath anywhere, and the extension points at the
        # bundle directory.
        absolute = [
            f"{n[len(pkg):]}: {p}"
            for n, i in parsed.items() for p in i["rpaths"] if not p.startswith("@")
        ]
        rep.check("C6a", not absolute,
                  f"no absolute rpath: {absolute or 'yes'}")
        ext_rpaths = ext["rpaths"] if ext else []
        rep.check("C6b", "@loader_path/.dylibs" in ext_rpaths,
                  f"the extension's rpath is the bundle directory: {ext_rpaths}")
        dylib_rpaths = {
            n[len(pkg):]: i["rpaths"] for n, i in parsed.items()
            if n != ext_name and i["rpaths"]
        }
        if dylib_rpaths:
            print("\n  note: bundled libraries carry their own rpaths:")
            for k, v in dylib_rpaths.items():
                print(f"        {k}: {v}")
            print("  dyld resolves @rpath through the load chain, so a bundled")
            print("  library's own @rpath dependency can resolve through the")
            print("  extension's @loader_path/.dylibs entry. That is a claim about")
            print("  dyld's search order and it has not been verified for this")
            print("  wheel; the clean-install fixture is what settles it.")

        # C7. Install names.
        bad_ids = []
        for name, info in parsed.items():
            iname = info["install_name"]
            if iname and (iname.startswith("/") or any(
                    m.decode() in iname for m in BUILD_HOST_MARKERS)):
                bad_ids.append(f"{name[len(pkg):]}: {iname}")
        rep.check(
            "C7",
            not bad_ids,
            f"no bundled library declares an absolute install name: "
            f"{bad_ids or 'yes'}",
            ["An install name is what other objects record as the dependency.",
             "An absolute one pointing into the build machine's pixi environment",
             "is the single most common way a copied dylib stops working",
             "somewhere else."] if bad_ids else [],
        )

        # C12. Absolute dependencies, which are legitimate only for the
        # system's own libraries.
        foreign = []
        for name, info in parsed.items():
            for dep in info["dylibs"]:
                if dep.startswith("@"):
                    continue
                if not dep.startswith(SYSTEM_DYLIB_PREFIXES):
                    foreign.append(f"{name[len(pkg):]} needs {dep}")
        rep.check(
            "C12",
            not foreign,
            f"absolute dependencies are system libraries only: {foreign or 'yes'}",
            ["A dependency outside /usr/lib and /System/Library is a library",
             "the user's Mac has no reason to have. Either bundle it or link",
             "it differently; do not publish a wheel that needs it."]
            if foreign else [],
        )

        # --- content scans ---------------------------------------------
        rep.section("content scans")
        host_hits: list[str] = []
        secret_hits: list[str] = []
        markers = tuple(BUILD_HOST_MARKERS) + EXTRA_HOST_MARKERS
        for info in zf.infolist():
            if info.is_dir():
                continue
            blob = blobs.get(info.filename) or zf.read(info.filename)
            vendored = info.filename.startswith(f"{pkg}.dylibs/")
            for mark in markers:
                if mark in blob:
                    tail = " (vendored MAX runtime)" if vendored else ""
                    host_hits.append(f"{info.filename}: {mark.decode()}{tail}")
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(blob):
                    secret_hits.append(f"{info.filename}: {label}")

        rep.check(
            "C8",
            not host_hits,
            f"no build-machine paths in any member: {host_hits or 'none found'}",
            ["A hit in _mojoboost.so is this project's bug: the rpath rewrite or",
             "the build did not clean up. A hit inside a vendored MAX library is",
             "a property of Modular's build, not of this repository, and it",
             "cannot be fixed here. It still blocks a release, because the",
             "string ships either way; the decision is recorded in",
             "handoffs/release_02_macos_wheels.md rather than made by this",
             "script."] if host_hits else [],
        )
        rep.check("C11", not secret_hits,
                  f"no secret-shaped strings: {secret_hits or 'none found'}",
                  ["This is a screen and not a proof. It finds a credential that",
                   "was packaged by accident; it does not certify that none was."])

    data["checks"] = rep.checks
    return data


def classify_dep(dep: str, bundled: dict) -> str:
    if dep.startswith("@"):
        return "bundled" if Path(dep).name in bundled else "UNRESOLVED"
    if dep.startswith(SYSTEM_DYLIB_PREFIXES):
        return "system"
    return "ABSOLUTE, not a system path"


def fmt_version(v) -> str:
    return "unknown" if v is None else ".".join(str(x) for x in v)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Inspect a macOS mojoboost wheel and check the release rules.",
    )
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full report as JSON, for the release record")
    ap.add_argument("--report-only", action="store_true",
                    help="print everything and exit 0 even when checks fail")
    args = ap.parse_args(argv)

    if not args.wheel.exists():
        print(f"no such file: {args.wheel}", file=sys.stderr)
        return 2

    rep = Report()
    data = inspect(args.wheel, rep)

    passed = sum(1 for c in rep.checks if c["ok"])
    print(f"\n{passed}/{len(rep.checks)} release checks passed"
          f"{' (report-only)' if args.report_only else ''}")
    if rep.failed:
        print("This wheel is not releasable as it stands.")
    print("Nothing above is a validation either way. A clean run says the")
    print("artifact is self-consistent, not that it works on a machine without")
    print("the toolchain, which is packaging/matrix/smoke/clean_install_macos.sh.")

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")

    return 0 if args.report_only or not rep.failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
