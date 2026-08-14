#!/usr/bin/env python3
"""Report what the dynamic loader will have to do at first import.

`ext_load` is the second phase of the startup contract in
docs/STARTUP_LATENCY.md, and it is the one phase whose cost is fixed
entirely by artifacts sitting on disk: how many shared libraries the
extension pulls in, how large they are, how their paths are spelled, and
whether the loader can find them without leaving the package. All of that
is readable from the files themselves.

So this script reads them. It parses the Mach-O or ELF headers of
`_mojoboost.so` and of any runtime libraries bundled beside it, and prints
the dependency list, the search paths, the platform minimum, the code
signature status, and the sizes.

    python3 tools/inspect_startup_artifacts.py
    python3 tools/inspect_startup_artifacts.py --json
    python3 tools/inspect_startup_artifacts.py --strict
    python3 tools/inspect_startup_artifacts.py path/to/other.so

Standard library only. It imports nothing from the package, executes
nothing, and does not need `otool`, `objdump`, `patchelf`, pixi, or a Mojo
toolchain, which is the point: the question "why is importing this slow,
or why does importing it fail" has to be answerable on the machine where
it went wrong, and that machine may have none of those.

Relationship to the release tooling
-----------------------------------
`packaging/matrix/validate_artifact.py` asks whether a *wheel* matches a
target that `platform_matrix.toml` declares, and it owns the repository's
Mach-O reader. This script asks a different question, about an *installed*
package on a machine, and it does not own a second reader: it loads
`macho_info` from that file. There is one Mach-O parser here and one place
to fix it. The ELF reader below is this file's own, because the existing
Linux path (`packaging/linux/inspect_elf.sh`) needs a real loader on a real
Linux box and cannot answer anything from a macOS checkout.

What it checks with `--strict`
------------------------------
Exit status is 0 unless `--strict` is given, in which case a failed
expectation exits 1:

1. the extension exists and parses
2. every `@rpath`-relative dependency resolves against some search path
   that exists on this filesystem
3. a wheel layout carries all four MAX runtime libraries
4. a wheel layout has no absolute search path pointing outside the
   package, which would mean the rpath rewrite in
   `packaging/build_wheel.sh` did not happen and the wheel only works on
   the machine that built it
5. on arm64 macOS every Mach-O carries a code signature, because
   `install_name_tool` invalidates the signature it was built with and an
   unsigned arm64 image is killed on load rather than being reported as a
   bad import

Check 4 is the one that matters for a release. A source build is *expected*
to point at the environment that built it and is not faulted for it.

Limits, stated because a report that overstates its coverage is worse than
no report: only 64-bit little-endian Mach-O and 64-bit ELF are read, which
covers every target in docs/PLATFORM_MATRIX.md and nothing else. Universal
binaries are rejected, not summarized, because this project ships
single-arch. A bundled library's own `LC_ID_DYLIB` install name is not
reported, because the shared reader does not keep that load command.
`@executable_path` is reported but not resolved, since the executable is
whichever interpreter runs later. The dependency walk follows only
libraries this script can find on disk, so the closure it prints is a lower
bound on what the loader will open, never an upper one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_DIR = ROOT / "python" / "mojoboost"

# The four MAX runtime libraries packaging/build_wheel.sh bundles. Keep in
# sync with the LIBS array there and with BUNDLED_RUNTIME_LIBS in
# python/mojoboost/diagnostics.py.
BUNDLED_RUNTIME_LIBS = (
    "libKGENCompilerRTShared",
    "libAsyncRTMojoBindings",
    "libMSupportGlobals",
    "libAsyncRTRuntimeGlobals",
)

# --- Mach-O ---------------------------------------------------------------
#
# This script does not read Mach-O itself. packaging/matrix/validate_artifact.py
# already has a reader, and its own docstring states the rule followed here:
# there is one such parser in the repository and one place to fix it. It is
# loaded by path rather than imported, because packaging/ is not a package.

CPU_TYPES = {
    0x0100000C: "arm64",
    0x01000007: "x86_64",
}

VALIDATE_ARTIFACT = ROOT / "packaging" / "matrix" / "validate_artifact.py"

# --- ELF ------------------------------------------------------------------

PT_LOAD = 1
PT_DYNAMIC = 2

DT_NULL = 0
DT_NEEDED = 1
DT_STRTAB = 5
DT_STRSZ = 10
DT_SONAME = 14
DT_RPATH = 15
DT_RUNPATH = 29

ELF_MACHINES = {
    0x3E: "x86_64",
    0xB7: "aarch64",
}


class ParseError(Exception):
    """The file is not an image this script knows how to read."""


def _cstring(blob: bytes, offset: int) -> str:
    end = blob.find(b"\0", offset)
    if end < 0:
        end = len(blob)
    return blob[offset:end].decode("utf-8", "replace")


def _load_macho_info():
    """`macho_info` from packaging/matrix/validate_artifact.py, or None.

    None rather than an exception: an image this script cannot describe is
    still worth reporting the size, digest, and install kind of, and a
    checkout where the release tooling has moved should degrade to a
    smaller report rather than to a traceback.

    That module imports only the standard library at module scope and
    guards its `main`, so loading it here runs nothing.
    """
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_mojoboost_validate_artifact", VALIDATE_ARTIFACT
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "macho_info", None)
    except Exception:
        return None


_MACHO_INFO = _load_macho_info()


def _parse_macho(blob: bytes) -> dict:
    """Adapt `macho_info`'s vocabulary to this script's record shape.

    Its rejections are kept rather than worked around. A universal binary
    is an error there because this project ships single-arch wheels, and
    reporting one here as merely interesting would hide a packaging bug
    that the release checks already treat as fatal.
    """
    if _MACHO_INFO is None:
        raise ParseError(
            "no Mach-O reader available: %s could not be loaded"
            % VALIDATE_ARTIFACT
        )
    info = _MACHO_INFO(blob)
    if "error" in info:
        raise ParseError(info["error"])
    minos = info.get("minos")
    return {
        "format": "mach-o",
        "arch": CPU_TYPES.get(info["cputype"], "0x%08x" % info["cputype"]),
        "dependencies": [
            {"name": name, "kind": "load"} for name in info["dylibs"]
        ],
        "search_paths": list(info["rpaths"]),
        # LC_ID_DYLIB is not among the load commands that reader keeps, so
        # a bundled library's own install name is not reported here.
        "install_name": None,
        "code_signed": bool(info["signed"]),
        "platform": "macos" if minos else None,
        "minimum_os": ".".join(str(part) for part in minos) if minos else None,
    }


# --- ELF ------------------------------------------------------------------


def _parse_elf(blob: bytes) -> dict:
    if blob[4] != 2:
        raise ParseError("only 64-bit ELF is parsed")
    endian = "<" if blob[5] == 1 else ">"
    machine, = struct.unpack_from(endian + "H", blob, 18)
    e_phoff, = struct.unpack_from(endian + "Q", blob, 32)
    e_phentsize, e_phnum = struct.unpack_from(endian + "HH", blob, 54)

    info = {
        "format": "elf",
        "arch": ELF_MACHINES.get(machine, "0x%x" % machine),
        "dependencies": [],
        "search_paths": [],
        "install_name": None,
        "code_signed": False,
        "platform": "linux",
        "minimum_os": None,
    }

    loads = []
    dynamic = None
    for i in range(e_phnum):
        at = e_phoff + i * e_phentsize
        p_type, = struct.unpack_from(endian + "I", blob, at)
        p_offset, p_vaddr = struct.unpack_from(endian + "QQ", blob, at + 8)
        p_filesz, = struct.unpack_from(endian + "Q", blob, at + 32)
        if p_type == PT_LOAD:
            loads.append((p_vaddr, p_filesz, p_offset))
        elif p_type == PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        return info

    def to_file_offset(vaddr: int):
        for p_vaddr, p_filesz, p_offset in loads:
            if p_vaddr <= vaddr < p_vaddr + p_filesz:
                return p_offset + (vaddr - p_vaddr)
        return None

    # Two passes: the string table address is itself a dynamic entry, so
    # the tags that index into it cannot be resolved until it is known.
    entries = []
    strtab_vaddr = None
    strsz = 0
    off, size = dynamic
    for at in range(off, off + size, 16):
        d_tag, d_val = struct.unpack_from(endian + "qQ", blob, at)
        if d_tag == DT_NULL:
            break
        entries.append((d_tag, d_val))
        if d_tag == DT_STRTAB:
            strtab_vaddr = d_val
        elif d_tag == DT_STRSZ:
            strsz = d_val
    if strtab_vaddr is None:
        return info
    strtab_off = to_file_offset(strtab_vaddr)
    if strtab_off is None:
        return info
    strtab = blob[strtab_off : strtab_off + strsz]

    for d_tag, d_val in entries:
        if d_tag == DT_NEEDED:
            info["dependencies"].append(
                {"name": _cstring(strtab, d_val), "kind": "load"}
            )
        elif d_tag == DT_SONAME:
            info["install_name"] = _cstring(strtab, d_val)
        elif d_tag in (DT_RPATH, DT_RUNPATH):
            for part in _cstring(strtab, d_val).split(":"):
                if part:
                    info["search_paths"].append(part)
    return info


# --- artifact inspection --------------------------------------------------


def inspect_image(path: Path) -> dict:
    """Everything the loader will consult, for one file on disk."""
    blob = path.read_bytes()
    record = {
        "path": str(path),
        "exists": True,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "error": None,
    }
    try:
        if len(blob) < 64:
            raise ParseError("file is too small to be a shared library")
        if blob[:4] == b"\x7fELF":
            record.update(_parse_elf(blob))
        else:
            record.update(_parse_macho(blob))
    except (ParseError, struct.error) as exc:
        record["error"] = str(exc)
        record.setdefault("format", "unknown")
        record.setdefault("dependencies", [])
        record.setdefault("search_paths", [])
    return record


def expand_search_path(entry: str, image_dir: Path) -> str:
    """A search path with `@loader_path` / `$ORIGIN` resolved.

    `@executable_path` is left alone: the executable is whichever
    interpreter imports the package later, which this script has no way to
    know and should not guess at.
    """
    if entry.startswith("@loader_path"):
        return os.path.normpath(
            str(image_dir) + entry[len("@loader_path") :]
        )
    if entry.startswith("$ORIGIN"):
        return os.path.normpath(str(image_dir) + entry[len("$ORIGIN") :])
    if entry.startswith("${ORIGIN}"):
        return os.path.normpath(str(image_dir) + entry[len("${ORIGIN}") :])
    return entry


def resolve_dependencies(record: dict) -> list:
    """For each dependency, where the loader would look and what is there.

    Only `@rpath`-relative and `@loader_path`-relative names are resolved.
    An absolute name is reported with whether that exact path exists; a
    bare SONAME (the ELF case) is reported unresolved, because resolving
    it means reimplementing the whole `ld.so` search order and being
    subtly wrong about it.
    """
    image_dir = Path(record["path"]).resolve().parent
    expanded = [
        expand_search_path(entry, image_dir)
        for entry in record.get("search_paths", [])
    ]
    resolved = []
    for dep in record.get("dependencies", []):
        name = dep["name"]
        entry = {"name": name, "kind": dep["kind"], "resolved": None}
        if name.startswith("@rpath/"):
            tail = name[len("@rpath/") :]
            candidates = [os.path.join(p, tail) for p in expanded]
            entry["candidates"] = candidates
            for candidate in candidates:
                if os.path.exists(candidate):
                    entry["resolved"] = candidate
                    break
        elif name.startswith("@loader_path"):
            candidate = expand_search_path(name, image_dir)
            entry["candidates"] = [candidate]
            if os.path.exists(candidate):
                entry["resolved"] = candidate
        elif os.path.isabs(name):
            entry["candidates"] = [name]
            if os.path.exists(name):
                entry["resolved"] = name
        else:
            entry["candidates"] = []
        resolved.append(entry)
    return resolved


def find_extension(package_dir: Path):
    """The compiled extension in `package_dir`, or None."""
    if not package_dir.is_dir():
        return None
    for entry in sorted(package_dir.iterdir()):
        if not entry.name.startswith("_mojoboost"):
            continue
        if entry.suffix in (".so", ".pyd", ".dylib"):
            return entry
    return None


def bundled_runtime_dir(package_dir: Path):
    for name in (".dylibs", ".libs"):
        path = package_dir / name
        if path.is_dir():
            return path
    return None


def collect(package_dir: Path, extra: list) -> dict:
    """Inspect the package's artifacts plus any explicitly named files."""
    extension = find_extension(package_dir)
    runtime_dir = bundled_runtime_dir(package_dir)

    if extension is None:
        install_kind = "absent"
    elif runtime_dir is not None:
        install_kind = "wheel"
    else:
        install_kind = "source"

    images = []
    if extension is not None:
        images.append(inspect_image(extension))
    bundled_names = []
    if runtime_dir is not None:
        for entry in sorted(runtime_dir.iterdir()):
            if entry.suffix in (".dylib", ".so") or ".so." in entry.name:
                bundled_names.append(entry.stem)
                images.append(inspect_image(entry))
    for path in extra:
        images.append(inspect_image(Path(path)))

    for record in images:
        record["resolved_dependencies"] = resolve_dependencies(record)

    total_bytes = sum(r["size_bytes"] for r in images)
    return {
        "package_dir": str(package_dir),
        "install_kind": install_kind,
        "extension": str(extension) if extension else None,
        "runtime_dir": str(runtime_dir) if runtime_dir else None,
        "bundled": bundled_names,
        "images": images,
        "totals": {
            "images": len(images),
            "bytes": total_bytes,
        },
    }


# --- expectations ---------------------------------------------------------


def check(result: dict) -> list:
    """Problems worth failing a release over, as `(level, message)`.

    `error` is something that will break an import or a redistribution.
    `warning` is something a reader should know and a build should not
    necessarily fail on.
    """
    problems = []
    kind = result["install_kind"]
    package_dir = os.path.abspath(result["package_dir"])

    if kind == "absent":
        problems.append(
            (
                "error",
                "no compiled extension in %s; build it with"
                " bindings/build.sh" % package_dir,
            )
        )
        return problems

    for record in result["images"]:
        name = os.path.basename(record["path"])
        if record.get("error"):
            problems.append(
                ("error", "%s could not be parsed: %s" % (name, record["error"]))
            )
            continue

        for dep in record["resolved_dependencies"]:
            if dep["resolved"] is not None:
                continue
            if not dep["candidates"]:
                # A bare SONAME on ELF. The loader's own search order
                # decides, and guessing at it here would be noise.
                continue
            problems.append(
                (
                    "error",
                    "%s needs %s and no search path on this machine has it"
                    % (name, dep["name"]),
                )
            )

        if record.get("arch") == "arm64" and not record.get("code_signed"):
            problems.append(
                (
                    "error",
                    "%s is arm64 and carries no code signature; it will be"
                    " killed on load, not reported as a bad import" % name,
                )
            )

        if kind == "wheel":
            for entry in record["search_paths"]:
                if entry.startswith(("@loader_path", "@executable_path")):
                    continue
                inside = os.path.abspath(entry).startswith(
                    package_dir + os.sep
                ) or os.path.abspath(entry) == package_dir
                if os.path.isabs(entry) and not inside:
                    problems.append(
                        (
                            "error",
                            "%s searches %s, which is outside the package;"
                            " the rpath rewrite in"
                            " packaging/build_wheel.sh did not take, and"
                            " this artifact only loads on the machine that"
                            " built it" % (name, entry),
                        )
                    )

    if kind == "wheel":
        have = set(result["bundled"])
        for lib in BUNDLED_RUNTIME_LIBS:
            if lib not in have:
                problems.append(
                    (
                        "error",
                        "bundled runtime library %s is missing from %s"
                        % (lib, result["runtime_dir"]),
                    )
                )
    else:
        problems.append(
            (
                "warning",
                "source install: the extension resolves the MAX runtime"
                " through the environment that built it, so moving or"
                " removing that environment breaks the import. This is"
                " expected for a development checkout.",
            )
        )

    return problems


# --- rendering ------------------------------------------------------------


def render(result: dict, problems: list) -> str:
    lines = []
    lines.append("install kind:  %s" % result["install_kind"])
    lines.append("package dir:   %s" % result["package_dir"])
    lines.append("extension:     %s" % (result["extension"] or "none"))
    lines.append("runtime dir:   %s" % (result["runtime_dir"] or "none"))
    lines.append(
        "images:        %d, %.1f MiB total"
        % (result["totals"]["images"], result["totals"]["bytes"] / 1048576.0)
    )
    lines.append("")
    lines.append(
        "Every image below is opened, mapped, and relocated during the"
    )
    lines.append(
        "ext_load phase of the startup contract (docs/STARTUP_LATENCY.md)."
    )

    for record in result["images"]:
        lines.append("")
        lines.append("== %s" % os.path.basename(record["path"]))
        lines.append("   path        %s" % record["path"])
        lines.append(
            "   size        %d bytes" % record["size_bytes"]
        )
        lines.append("   sha256      %s" % record["sha256"])
        if record.get("error"):
            lines.append("   error       %s" % record["error"])
            continue
        lines.append(
            "   format      %s %s" % (record["format"], record.get("arch"))
        )
        if record.get("minimum_os"):
            lines.append(
                "   minimum os  %s %s"
                % (record.get("platform"), record["minimum_os"])
            )
        if record.get("install_name"):
            lines.append("   install as  %s" % record["install_name"])
        lines.append(
            "   signed      %s" % ("yes" if record.get("code_signed") else "no")
        )
        paths = record.get("search_paths") or []
        lines.append("   search paths (%d)" % len(paths))
        for entry in paths:
            expanded = expand_search_path(
                entry, Path(record["path"]).resolve().parent
            )
            marker = "exists" if os.path.isdir(expanded) else "MISSING"
            if expanded == entry:
                lines.append("     %-44s %s" % (entry, marker))
            else:
                lines.append(
                    "     %-44s %s -> %s" % (entry, marker, expanded)
                )
        deps = record.get("resolved_dependencies") or []
        lines.append("   dependencies (%d)" % len(deps))
        for dep in deps:
            if dep["resolved"]:
                state = "ok"
            elif dep["candidates"]:
                state = "UNRESOLVED"
            else:
                state = "loader search"
            lines.append("     %-52s %s" % (dep["name"], state))

    lines.append("")
    if not problems:
        lines.append("No problems found.")
    else:
        for level, message in problems:
            lines.append("%s: %s" % (level.upper(), message))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report the shared libraries, search paths, and sizes that"
            " determine first-import cost. Reads files only; imports,"
            " builds, and runs nothing."
        )
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="additional shared libraries to inspect",
    )
    parser.add_argument(
        "--package-dir",
        default=str(DEFAULT_PACKAGE_DIR),
        help="package directory holding the extension (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full record as JSON instead of a report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when an expectation fails",
    )
    args = parser.parse_args(argv)

    result = collect(Path(args.package_dir), args.extra)
    problems = check(result)

    if args.json:
        result["problems"] = [
            {"level": level, "message": message} for level, message in problems
        ]
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render(result, problems))

    if args.strict and any(level == "error" for level, _ in problems):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
