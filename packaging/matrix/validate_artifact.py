#!/usr/bin/env python3
"""Check a built mojoboost wheel against the release matrix.

    python3 packaging/matrix/validate_artifact.py python/dist/<wheel> [...]

NOT EXECUTED BY ANY TASK OR WORKFLOW, and it has not been run: it needs a
wheel, and building one is a release operation. Wiring it into
`pixi run build-wheel` is specified in handoffs/task18_platform.md.

Standard library only. It reads the wheel as a zip and parses the Mach-O load
commands itself, so it runs on a bare checkout with no delocate, no auditwheel,
and no packaging module, and it never has to install what it is inspecting.

What it is for. `packaging/test_wheel.sh` already answers "does this wheel
work", by installing it and running the suites. It answers that from inside the
pixi environment, where the Mojo runtime the extension links is on the library
path whether the wheel bundles it or not, and it says nothing about what the
wheel claims on its label. This script answers the other two questions:

    does the wheel say true things about where it can be installed
    does it carry everything it needs, and nothing from the build machine

Each rule below is independent and each prints its own verdict, because a
release decision wants the whole list, not the first failure.

Exit status is 0 when every rule passes and 1 otherwise. A rule that cannot
run (no matching target, unreadable member) is a failure, not a skip: an
unchecked wheel is not a checked one.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MATRIX = ROOT / "packaging" / "matrix" / "platform_matrix.toml"

# Mach-O, little endian 64 bit. The only object kind this project ships on
# macOS; a fat binary or a 32 bit object is a bug, not a case to handle.
MH_MAGIC_64 = 0xFEEDFACF
# Fat headers are big endian on disk, so these are the values a little endian
# read of the first four bytes produces.
FAT_MAGIC_LE = 0xBEBAFECA
FAT_MAGIC_64_LE = 0xBFBAFECA
CPU_TYPE_ARM64 = 0x0100000C
CPU_TYPE_X86_64 = 0x01000007
LC_LOAD_DYLIB = 0x0C
LC_CODE_SIGNATURE = 0x1D
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
LC_RPATH = 0x8000001C

# Anything from the build machine that must not survive into a wheel. The rpath
# in a freshly built extension points at the checkout's pixi environment, which
# is exactly the kind of string this catches.
BUILD_HOST_MARKERS = (
    b"/.pixi/",
    b"/Users/",
    b"/home/",
    b"/opt/homebrew/",
    b"conda-bld",
)

# Members that have no business in a distribution.
FORBIDDEN_MEMBERS = re.compile(
    r"(^|/)(__pycache__/|\.pytest_cache/|build/|dist/|tests?/|\.DS_Store$|.*\.pyc$)"
)

WHEEL_NAME = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>.+)\.whl$"
)

# Required keys in the provenance sidecar. These are the facts that cannot be
# recovered from the wheel afterwards, which is why they have to be written
# down at build time or lost.
PROVENANCE_KEYS = (
    "mojo_version",
    "max_version",
    "pixi_lock_sha256",
    "git_commit",
    "build_host_os",
    "build_host_arch",
    "has_accelerator_at_build",
    "metal_toolchain",
)


class Result:
    def __init__(self) -> None:
        self.failed = False
        self.checked = 0

    def rule(self, rid: str, ok: bool, detail: str) -> None:
        self.checked += 1
        if not ok:
            self.failed = True
        print(f"{'ok  ' if ok else 'FAIL'} {rid}  {detail}")

    def note(self, text: str) -> None:
        print(f"     {text}")


# ---------------------------------------------------------------------------
# Mach-O
# ---------------------------------------------------------------------------


def macho_info(blob: bytes) -> dict:
    """Load commands worth knowing about, or {'error': ...}.

    Only what the rules below need: architecture, deployment target, rpaths,
    linked libraries, and whether the object carries a code signature.
    """
    if len(blob) < 32:
        return {"error": "too short to be a Mach-O object"}
    magic = struct.unpack_from("<I", blob, 0)[0]
    if magic in (FAT_MAGIC_LE, FAT_MAGIC_64_LE):
        return {"error": "universal (fat) binary; this project ships single-arch"}
    if magic != MH_MAGIC_64:
        return {"error": f"not a 64 bit little endian Mach-O (magic {magic:#x})"}

    cputype, _sub, _ftype, ncmds, _sizeofcmds, _flags, _res = struct.unpack_from(
        "<iiIIIII", blob, 4
    )
    info: dict = {
        "cputype": cputype & 0xFFFFFFFF,
        "rpaths": [],
        "dylibs": [],
        "minos": None,
        "sdk": None,
        "signed": False,
    }
    off = 32
    for _ in range(ncmds):
        if off + 8 > len(blob):
            return {"error": "truncated load commands"}
        cmd, cmdsize = struct.unpack_from("<II", blob, off)
        if cmdsize < 8 or off + cmdsize > len(blob):
            return {"error": "bad load command size"}
        if cmd == LC_RPATH:
            (stroff,) = struct.unpack_from("<I", blob, off + 8)
            info["rpaths"].append(_lc_str(blob, off, cmdsize, stroff))
        elif cmd == LC_LOAD_DYLIB:
            (stroff,) = struct.unpack_from("<I", blob, off + 8)
            info["dylibs"].append(_lc_str(blob, off, cmdsize, stroff))
        elif cmd == LC_BUILD_VERSION:
            _plat, minos, sdk, _ntools = struct.unpack_from("<IIII", blob, off + 8)
            info["minos"] = _version(minos)
            info["sdk"] = _version(sdk)
        elif cmd == LC_VERSION_MIN_MACOSX and info["minos"] is None:
            version, sdk = struct.unpack_from("<II", blob, off + 8)
            info["minos"] = _version(version)
            info["sdk"] = _version(sdk)
        elif cmd == LC_CODE_SIGNATURE:
            info["signed"] = True
        off += cmdsize
    return info


def _lc_str(blob: bytes, off: int, cmdsize: int, stroff: int) -> str:
    raw = blob[off + stroff : off + cmdsize]
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _version(packed: int) -> tuple[int, int, int]:
    return (packed >> 16, (packed >> 8) & 0xFF, packed & 0xFF)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def check_wheel(path: Path, matrix: dict, res: Result) -> None:
    print(f"\n--- {path.name}")
    m = WHEEL_NAME.match(path.name)
    if not m:
        res.rule("R1", False, "filename is not a wheel name")
        return

    tag = f"{m['python']}-{m['abi']}-{m['platform']}"
    targets = [t for t in matrix.get("target", []) if t.get("wheel_tag") == tag]

    # R1. The tag has to be one the matrix expects. A wheel with a tag nobody
    # declared is the failure mode this whole directory exists to prevent: it
    # is how a build machine's incidental configuration becomes a published
    # promise.
    res.rule(
        "R1",
        bool(targets),
        f"tag {tag} " + (f"matches target {targets[0]['id']}" if targets
                         else "matches NO target in platform_matrix.toml"),
    )
    if not targets:
        res.note("Either the build is wrong or the matrix is out of date. Do")
        res.note("not resolve this by adding the tag the build happened to")
        res.note("produce; work out which of the two is the error first.")
        return
    target = targets[0]

    if m["name"] != matrix["project"] or m["version"] != matrix["version"]:
        res.rule("R1b", False,
                 f"name/version {m['name']}-{m['version']} does not match "
                 f"{matrix['project']}-{matrix['version']}")
    else:
        res.rule("R1b", True, f"{m['name']}-{m['version']}")

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkg = f"{matrix['project']}/"

        # R2. Everything the extension needs travels with it. The wheel's whole
        # claim is that it installs without a Mojo or MAX toolchain, and the
        # bundled runtime is what makes that true.
        ext = [n for n in names if n.endswith((".so", ".pyd"))]
        res.rule("R2a", len(ext) == 1 and ext[0] == f"{pkg}_mojoboost.so",
                 f"extension module: {ext or 'MISSING'}")

        expected = set(target.get("bundled_dylibs", []))
        if target["os"] == "macos":
            bundled = {Path(n).stem for n in names if n.startswith(f"{pkg}.dylibs/")}
            res.rule("R2b", bundled == expected,
                     f"bundled dylibs: {sorted(bundled) or 'NONE'}")
            if bundled != expected:
                res.note(f"expected exactly {sorted(expected)}")

        # R3. No member that only exists on the build machine.
        strays = [n for n in names if FORBIDDEN_MEMBERS.search(n)]
        res.rule("R3", not strays, f"stray members: {strays or 'none'}")

        # R4. Metadata agrees with the target.
        meta = _read_metadata(zf, names)
        req = meta.get("Requires-Python", "")
        want = _python_row(matrix, target["python"])
        res.rule("R4a", req.replace(" ", "") == f">={want['version']}",
                 f"Requires-Python: {req or 'MISSING'}")
        res.rule("R4b", "License-File" in meta or any(
            "licenses/" in n for n in names), "license file present")

        # R5 and R6, per Mach-O object.
        if target["os"] == "macos":
            _check_macho(zf, names, pkg, target, res)
        elif target["os"] == "linux":
            _check_elf(zf, names, pkg, res)

    # R7. Provenance. The wheel cannot tell you what toolchain built it, which
    # macOS it was built on, or whether an accelerator was visible at compile
    # time, and that last one changes the binary's behavior on the user's
    # machine (src/mojoboost/device.mojo: has_accelerator() is resolved at
    # compile time, so a wheel built on a GPU machine reports a GPU as
    # available and fails later, when the device is opened).
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    if not sidecar.exists():
        res.rule("R7", False, f"no provenance sidecar at {sidecar.name}")
    else:
        try:
            prov = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            res.rule("R7", False, f"provenance unreadable: {exc}")
        else:
            missing = [k for k in PROVENANCE_KEYS if not str(prov.get(k, "")).strip()]
            res.rule("R7", not missing,
                     f"provenance keys: {'complete' if not missing else missing}")


def _check_macho(zf, names, pkg, target, res) -> None:
    want_arch = {"arm64": CPU_TYPE_ARM64, "x86_64": CPU_TYPE_X86_64}[target["arch"]]
    tag_floor = _platform_floor(target["wheel_tag"])
    objects = [n for n in names
               if n.startswith(pkg) and n.endswith((".so", ".dylib"))]
    for name in sorted(objects):
        blob = zf.read(name)
        info = macho_info(blob)
        short = name[len(pkg):]
        if "error" in info:
            res.rule("R5", False, f"{short}: {info['error']}")
            continue

        # R5a. One architecture, the one on the label.
        res.rule("R5a", info["cputype"] == want_arch,
                 f"{short}: cputype {info['cputype']:#x} "
                 f"(want {want_arch:#x} = {target['arch']})")

        # R5b. The deployment target is the real install floor, and the
        # platform tag is a promise about it. An object built for a newer macOS
        # than the tag claims installs and then fails to load, which is the
        # worst of the available failures.
        minos = info["minos"]
        res.rule("R5b", minos is not None and minos[:2] <= tag_floor,
                 f"{short}: minos {_fmt(minos)} against tag floor "
                 f"{tag_floor[0]}.{tag_floor[1]}")

        # R5c. No absolute rpath, and no path into the build machine. A
        # freshly built extension carries an rpath into the checkout's pixi
        # environment; build_wheel.sh replaces it with @loader_path/.dylibs and
        # this is the check that it worked.
        bad = [p for p in info["rpaths"] if not p.startswith("@")]
        res.rule("R5c", not bad, f"{short}: rpaths {info['rpaths'] or 'none'}")

        # R5d. install_name_tool invalidates the signature on arm64. An
        # unsigned object that was rewritten will not load on Apple silicon.
        res.rule("R5d", info["signed"], f"{short}: code signature present")

        # R5e. Every @rpath dependency resolves to something in the wheel.
        for dep in info["dylibs"]:
            if not dep.startswith("@rpath/"):
                continue
            stem = Path(dep).name
            present = f"{pkg}.dylibs/{stem}" in names
            res.rule("R5e", present, f"{short}: needs {dep} "
                     f"{'(bundled)' if present else '(NOT IN WHEEL)'}")

        # R6. Nothing from the build machine anywhere in the bytes, not only in
        # the load commands. Debug paths and embedded strings count.
        hits = [mark.decode() for mark in BUILD_HOST_MARKERS if mark in blob]
        res.rule("R6", not hits, f"{short}: build host strings {hits or 'none'}")


def _check_elf(zf, names, pkg, res) -> None:
    # A byte scan, not a dynamic-section parse. It is enough to catch the two
    # failures that matter and honest about being a screen rather than a proof:
    # readelf -d in packaging/matrix/smoke/clean_install_linux.sh is the
    # authority, and it runs on the target rather than here.
    objects = [n for n in names if n.startswith(pkg) and ".so" in n]
    for name in sorted(objects):
        blob = zf.read(name)
        short = name[len(pkg):]
        if blob[:4] != b"\x7fELF":
            res.rule("R5", False, f"{short}: not an ELF object")
            continue
        hits = [mark.decode() for mark in BUILD_HOST_MARKERS if mark in blob]
        res.rule("R6", not hits, f"{short}: build host strings {hits or 'none'}")
        glibc = sorted({v.decode() for v in re.findall(rb"GLIBC_2\.\d+", blob)},
                       key=lambda s: int(s.rsplit(".", 1)[1]))
        res.rule("R5f", bool(glibc),
                 f"{short}: glibc symbol versions referenced, highest "
                 f"{glibc[-1] if glibc else 'none found'}")
        res.note("Compare that against the manylinux tag by hand; this scan")
        res.note("does not know which symbols are actually reachable.")


def _read_metadata(zf, names) -> dict:
    for name in names:
        if name.endswith(".dist-info/METADATA"):
            out = {}
            for line in zf.read(name).decode("utf-8", "replace").splitlines():
                if not line.strip():
                    break
                key, _, value = line.partition(":")
                out.setdefault(key.strip(), value.strip())
            return out
    return {}


def _platform_floor(tag: str) -> tuple[int, int]:
    m = re.search(r"macosx_(\d+)_(\d+)_", tag)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _python_row(matrix: dict, tag: str) -> dict:
    for row in matrix.get("python", []):
        if row["tag"] == tag:
            return row
    return {"version": "?"}


def _fmt(v) -> str:
    return "unknown" if v is None else ".".join(str(x) for x in v)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    matrix = tomllib.loads(MATRIX.read_text())
    res = Result()
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            res.rule("R0", False, f"{arg}: no such file")
            continue
        check_wheel(path, matrix, res)
    print(f"\n{res.checked} rules checked, {'FAILED' if res.failed else 'all passed'}")
    return 1 if res.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
