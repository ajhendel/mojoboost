#!/usr/bin/env python3
"""Inspect a Linux mojotrees wheel: its label, its contents, and its ELF objects.

    python3 packaging/linux/inspect_wheel.py python/dist/<wheel> [...]

THIS SCRIPT HAS NEVER BEEN EXECUTED. It was written against the ELF and wheel
formats rather than against an artifact, because no Linux wheel exists yet. The
first run should be treated as a test of this file as much as of the wheel, and
its output kept.

Standard library only. It reads the wheel as a zip and parses ELF headers,
dynamic sections and version requirements itself, so it needs no auditwheel, no
patchelf, no binutils, and no target machine. It runs on macOS, which is where
this project is developed and where a Linux wheel cannot otherwise be looked at.

Where it sits among the other checks:

    packaging/matrix/validate_artifact.py   matrix conformance; its ELF branch
                                            is a byte scan and says so, and
                                            defers the real parse to here
    this file                               what the objects actually declare
    packaging/matrix/smoke/clean_install_linux.sh
                                            whether it works, on a real target

None of the three substitutes for the others, and this one is the only one that
can run before a wheel is installed anywhere. It answers two questions:

    does the filename tell the truth about where this file can be installed
    does it carry what it needs and nothing from the machine that built it

Exit status is 0 when every rule passes and 1 otherwise. A rule that cannot be
evaluated is a failure, not a skip.
"""

from __future__ import annotations

import posixpath
import re
import struct
import sys
import zipfile
from pathlib import Path

WHEEL_NAME = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>.+)\.whl$"
)

# EI_CLASS, e_machine, and the section and dynamic tags this needs.
ELFCLASS64 = 2
ELFDATA2LSB = 1
EM_X86_64 = 62
EM_AARCH64 = 183
MACHINE_NAMES = {EM_X86_64: "x86_64", EM_AARCH64: "aarch64"}
SHT_DYNAMIC = 6
SHT_STRTAB = 3
SHT_GNU_VERNEED = 0x6FFFFFFE
DT_NULL = 0
DT_NEEDED = 1
DT_SONAME = 14
DT_RPATH = 15
DT_RUNPATH = 29

# PEP 600's external allowlist, minus the X11/GL/graphics entries that cannot
# apply to this project. A DT_NEEDED entry outside this set has to be bundled,
# or the wheel is quietly requiring the user to have something.
EXTERNAL_ALLOWED = re.compile(
    r"^(libc|libm|libdl|librt|libpthread|libresolv|libutil|libcrypt|libnsl|"
    r"libgcc_s|libstdc\+\+|ld-linux-x86-64|ld-linux-aarch64|ld64)\.so[.0-9]*$"
)

# Strings from the machine that built the wheel. The RPATH a freshly built Mojo
# extension carries is an absolute path into the build checkout's pixi
# environment, which is exactly what this catches.
HOST_MARKERS = (
    b"/.pixi/",
    b"/Users/",
    b"/home/",
    b"/root/",
    b"/opt/hostedtoolcache/",
    b"/opt/homebrew/",
    b"conda-bld",
    b"CONDA_PREFIX",
)

FORBIDDEN_MEMBER = re.compile(
    r"(^|/)(__pycache__/|\.pytest_cache/|tests?/|build/|dist/|\.git/|"
    r"\.DS_Store$|.*\.pyc$|.*\.egg-info/)"
)

VERSION_TOKEN = re.compile(rb"(GLIBC|GLIBCXX|CXXABI|GCC)_[0-9][0-9A-Za-z._]*")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


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
# ELF
# ---------------------------------------------------------------------------


def elf_info(blob: bytes) -> dict:
    """What a 64 bit little endian ELF object declares, or {'error': ...}.

    Only what the rules need: architecture, soname, DT_NEEDED, RPATH, RUNPATH,
    and the symbol versions the object requires from its dependencies. 32 bit
    and big endian objects are reported as errors rather than handled, because
    this project ships neither and pretending otherwise would add code that can
    never be exercised.
    """
    if len(blob) < 64 or blob[:4] != b"\x7fELF":
        return {"error": "not an ELF object"}
    if blob[4] != ELFCLASS64:
        return {"error": f"not a 64 bit object (EI_CLASS {blob[4]})"}
    if blob[5] != ELFDATA2LSB:
        return {"error": f"not little endian (EI_DATA {blob[5]})"}

    (e_machine,) = struct.unpack_from("<H", blob, 18)
    e_shoff, = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3A)

    info: dict = {
        "machine": e_machine,
        "soname": None,
        "needed": [],
        "rpath": [],
        "runpath": [],
        "versions": set(),
    }
    if e_shoff == 0 or e_shnum == 0:
        return {"error": "no section headers; object is stripped of them"}

    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if off + 64 > len(blob):
            return {"error": "truncated section header table"}
        sh_name, sh_type = struct.unpack_from("<II", blob, off)
        sh_offset, sh_size = struct.unpack_from("<QQ", blob, off + 24)
        (sh_link,) = struct.unpack_from("<I", blob, off + 40)
        (sh_entsize,) = struct.unpack_from("<Q", blob, off + 56)
        sections.append(
            {
                "name": sh_name,
                "type": sh_type,
                "offset": sh_offset,
                "size": sh_size,
                "link": sh_link,
                "entsize": sh_entsize,
            }
        )

    def body(sec: dict) -> bytes:
        return blob[sec["offset"] : sec["offset"] + sec["size"]]

    def cstr(table: bytes, at: int) -> str:
        end = table.find(b"\x00", at)
        raw = table[at:] if end < 0 else table[at:end]
        return raw.decode("utf-8", "replace")

    dynamic = next((s for s in sections if s["type"] == SHT_DYNAMIC), None)
    if dynamic is None:
        return {"error": "no .dynamic section; not a shared object"}
    if dynamic["link"] >= len(sections):
        return {"error": "'.dynamic' points at a section that does not exist"}
    dynstr_sec = sections[dynamic["link"]]
    if dynstr_sec["type"] != SHT_STRTAB:
        return {"error": "'.dynamic' does not link to a string table"}
    dynstr = body(dynstr_sec)

    raw = body(dynamic)
    for off in range(0, len(raw) - 15, 16):
        d_tag, d_val = struct.unpack_from("<Qq", raw, off)
        if d_tag == DT_NULL:
            break
        if d_tag == DT_NEEDED:
            info["needed"].append(cstr(dynstr, d_val))
        elif d_tag == DT_SONAME:
            info["soname"] = cstr(dynstr, d_val)
        elif d_tag == DT_RPATH:
            info["rpath"] = cstr(dynstr, d_val).split(":")
        elif d_tag == DT_RUNPATH:
            info["runpath"] = cstr(dynstr, d_val).split(":")

    info["versions"] = _verneed(blob, sections, body, cstr, dynstr)
    return info


def _verneed(blob, sections, body, cstr, dynstr) -> set:
    """Symbol versions this object requires, as {(library, version)}.

    Parsed out of .gnu.version_r when that is readable, because the version
    alone ("GLIBC_2.29") does not say which library it has to come from. Falls
    back to scanning the dynamic string table, which holds the same version
    strings without the attribution: a weaker answer, not a wrong one.
    """
    verneed = next((s for s in sections if s["type"] == SHT_GNU_VERNEED), None)
    out: set = set()
    if verneed is not None:
        try:
            strtab = dynstr
            if verneed["link"] < len(sections):
                strtab = body(sections[verneed["link"]])
            raw = body(verneed)
            at = 0
            while at + 16 <= len(raw):
                _ver, cnt, file_off, aux_off, next_off = struct.unpack_from(
                    "<HHIII", raw, at
                )
                library = cstr(strtab, file_off)
                aux = at + aux_off
                for _ in range(cnt):
                    if aux + 16 > len(raw):
                        break
                    _hash, _flags, _other, name_off, aux_next = struct.unpack_from(
                        "<IHHII", raw, aux
                    )
                    out.add((library, cstr(strtab, name_off)))
                    if aux_next == 0:
                        break
                    aux += aux_next
                if next_off == 0:
                    break
                at += next_off
            if out:
                return out
        except (struct.error, IndexError, UnicodeDecodeError):
            pass
    for match in VERSION_TOKEN.finditer(dynstr):
        out.add(("(unattributed)", match.group(0).decode("ascii", "replace")))
    return out


def _version_tuple(token: str) -> tuple:
    """'GLIBC_2.28' -> (2, 28). Unparseable pieces sort low."""
    _, _, tail = token.partition("_")
    parts = []
    for piece in tail.split("."):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def inspect(path: Path, res: Result) -> None:
    print(f"\n--- {path.name}")
    match = WHEEL_NAME.match(path.name)
    if not match:
        res.rule("L1", False, "filename is not a wheel name")
        return

    plat = match["platform"]
    res.rule(
        "L1",
        True,
        f"{match['name']} {match['version']}, "
        f"{match['python']}-{match['abi']}-{plat}",
    )

    tag_arch, tag_floor = _platform_claim(plat, res)

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        distinfo = f"{match['name']}-{match['version']}.dist-info/"
        pkg = "mojotrees/"

        _check_layout(zf, names, distinfo, pkg, res)
        _check_elf(zf, names, pkg, tag_arch, tag_floor, res)
        _check_host_paths(zf, names, res)


def _platform_claim(plat: str, res: Result) -> tuple:
    """What the platform tag promises. Returns (arch, glibc floor or None)."""
    many = re.fullmatch(r"manylinux_(\d+)_(\d+)_(x86_64|aarch64)", plat)
    plain = re.fullmatch(r"linux_(x86_64|aarch64)", plat)
    legacy = re.fullmatch(r"manylinux(1|2010|2014)_(x86_64|aarch64)", plat)

    if plain:
        res.rule("L2", True, f"plain tag, no glibc promise, arch {plain[1]}")
        res.note("PyPI and TestPyPI reject this tag on upload, by design. This")
        res.note("file is distributable as a release asset and by direct path.")
        return plain[1], None
    if many:
        floor = (int(many[1]), int(many[2]))
        res.rule("L2", True, f"manylinux tag: glibc {floor[0]}.{floor[1]}, arch {many[3]}")
        res.note("A tag is a promise. Rules L7 and L9 below check what the")
        res.note("objects reference; only a clean install in a container of")
        res.note("that exact glibc checks the promise itself.")
        return many[3], floor
    if legacy:
        res.rule("L2", False, f"legacy manylinux alias '{plat}'")
        res.note("Use the PEP 600 form (manylinux_2_28_x86_64), not the")
        res.note("manylinux2014 alias. The builder never produces this.")
        return legacy[2], None

    res.rule("L2", False, f"unrecognized platform tag '{plat}'")
    return "", None


def _check_layout(zf, names, distinfo, pkg, res: Result) -> None:
    """L3-L6: does the wheel contain what it should and nothing else."""
    wheel_meta = distinfo + "WHEEL"
    if wheel_meta not in names:
        res.rule("L3", False, f"no {wheel_meta}")
    else:
        fields = {}
        for line in zf.read(wheel_meta).decode("utf-8", "replace").splitlines():
            key, _, value = line.partition(":")
            fields.setdefault(key.strip(), []).append(value.strip())
        purelib = (fields.get("Root-Is-Purelib") or ["true"])[0].lower()
        res.rule(
            "L3",
            purelib == "false",
            f"Root-Is-Purelib: {purelib} (a wheel with a compiled "
            "extension must be platlib)",
        )

    res.rule("L4", distinfo + "METADATA" in names, f"{distinfo}METADATA")

    licenses = [
        n for n in names
        if n.startswith(distinfo) and "LICENSE" in n.upper()
    ]
    res.rule("L5", bool(licenses), f"license files in dist-info: {licenses or 'none'}")
    res.note("Every bundled runtime object needs its license text here too, and")
    res.note("the MAX runtime is proprietary. See packaging/linux/README.md.")

    strays = sorted(n for n in names if FORBIDDEN_MEMBER.search(n))
    res.rule("L6", not strays, f"stray members: {strays or 'none'}")

    dylibs = sorted(n for n in names if ".dylibs/" in n or n.endswith(".dylib"))
    res.rule("L6b", not dylibs, f"Mach-O leftovers: {dylibs or 'none'}")

    record = distinfo + "RECORD"
    if record not in names:
        res.rule("L6c", False, f"no {record}")
    else:
        listed = {
            line.rsplit(",", 2)[0]
            for line in zf.read(record).decode("utf-8", "replace").splitlines()
            if line.strip()
        }
        missing = sorted(set(names) - listed - {record})
        res.rule("L6c", not missing, f"members absent from RECORD: {missing or 'none'}")


def _check_elf(zf, names, pkg, tag_arch, tag_floor, res: Result) -> None:
    """L7-L9: what the shipped objects declare about themselves."""
    objects = sorted(
        n for n in names
        if not n.endswith("/") and re.search(r"\.so(\.\d+)*$", n)
    )
    if not objects:
        res.rule("L7", False, "no ELF objects in the wheel")
        return

    bundled = {posixpath.basename(n) for n in objects}
    sonames = set()
    highest = None

    for name in objects:
        short = name[len(pkg):] if name.startswith(pkg) else name
        info = elf_info(zf.read(name))
        if "error" in info:
            res.rule("L7", False, f"{short}: {info['error']}")
            continue

        arch = MACHINE_NAMES.get(info["machine"], f"e_machine {info['machine']}")
        res.rule(
            "L7a",
            arch == tag_arch,
            f"{short}: {arch}, tag claims {tag_arch or 'nothing usable'}",
        )
        if info["soname"]:
            sonames.add(info["soname"])

        # DT_NEEDED. Every entry is either inside this wheel or a library the
        # target machine is allowed to be assumed to have.
        unmet = [
            dep for dep in info["needed"]
            if dep not in bundled and not EXTERNAL_ALLOWED.fullmatch(dep)
        ]
        res.rule(
            "L7b",
            not unmet,
            f"{short}: needs {info['needed'] or 'nothing'}"
            + (f"; NOT SATISFIED: {unmet}" if unmet else ""),
        )

        # RPATH and RUNPATH. Anything absolute is the build machine leaking in.
        paths = info["rpath"] + info["runpath"]
        absolute = [p for p in paths if p.startswith("/")]
        res.rule(
            "L8a",
            not absolute,
            f"{short}: search paths {paths or 'none'}"
            + (f"; ABSOLUTE: {absolute}" if absolute else ""),
        )
        res.rule(
            "L8b",
            not info["rpath"],
            f"{short}: DT_RPATH {info['rpath'] or 'absent'}"
            + (" (deprecated and not overridable by LD_LIBRARY_PATH; "
               "patchelf --set-rpath should have produced DT_RUNPATH)"
               if info["rpath"] else ""),
        )

        glibc = sorted(
            (v for _lib, v in info["versions"] if v.startswith("GLIBC_")),
            key=_version_tuple,
        )
        other = sorted({v for _lib, v in info["versions"] if not v.startswith("GLIBC_")},
                       key=_version_tuple)
        top = glibc[-1] if glibc else None
        if top and (highest is None or _version_tuple(top) > _version_tuple(highest)):
            highest = top
        res.rule(
            "L9a",
            bool(glibc),
            f"{short}: highest glibc requirement {top or 'none found'}"
            + (f"; also {other}" if other else ""),
        )

    # Bundled objects have to be findable by soname, which is the name the
    # loader looks for, not the name on disk.
    mismatched = sorted(
        s for s in sonames
        if s not in bundled and not EXTERNAL_ALLOWED.fullmatch(s)
    )
    res.rule(
        "L7c",
        not mismatched,
        f"sonames with no matching filename in the wheel: {mismatched or 'none'}",
    )

    if tag_floor is None:
        res.rule(
            "L9b",
            True,
            f"no glibc claim to check; the artifact's real floor is "
            f"{highest or 'unknown'}",
        )
        res.note("That number is what a manylinux tag would have to be at or")
        res.note("above. It is a finding, not a permission.")
    else:
        claim = f"GLIBC_{tag_floor[0]}.{tag_floor[1]}"
        ok = highest is not None and _version_tuple(highest) <= _version_tuple(claim)
        res.rule(
            "L9b",
            ok,
            f"tag claims {claim}, objects require up to {highest or 'unknown'}",
        )


def _check_host_paths(zf, names, res: Result) -> None:
    """L10: nothing from the build machine survives into the artifact."""
    dirty = {}
    for name in names:
        if name.endswith("/"):
            continue
        blob = zf.read(name)
        hits = sorted({m.decode() for m in HOST_MARKERS if m in blob})
        if hits:
            dirty[name] = hits
    res.rule("L10", not dirty, f"build host strings: {dirty or 'none'}")
    if dirty:
        res.note("A path into the build machine's pixi environment in a shipped")
        res.note("object usually means RUNPATH was never rewritten. Rebuild;")
        res.note("do not patch the wheel.")


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    res = Result()
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            res.rule("L0", False, f"{arg}: no such file")
            continue
        try:
            inspect(path, res)
        except (zipfile.BadZipFile, OSError) as exc:
            res.rule("L0", False, f"{arg}: unreadable ({exc})")
    print(f"\n{res.checked} rules checked, {'FAILED' if res.failed else 'all passed'}")
    print("Passing here means the label and the contents agree. It is not")
    print("evidence that the wheel imports, trains, or predicts anywhere.")
    return 1 if res.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
