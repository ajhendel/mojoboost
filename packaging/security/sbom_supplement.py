#!/usr/bin/env python3
"""Add the shipped native runtime to a CycloneDX SBOM of a mojoboost wheel.

A dependency scanner reads a wheel the way a package manager does. It finds one
Python distribution called mojoboost and, correctly by its own lights, stops.
What it does not find is that the wheel contains a compiled Mojo extension and
four MAX runtime libraries copied out of a conda environment
(`packaging/build_wheel.sh`), that those libraries are the largest thing in the
artifact by bytes, and that they are the part a consumer would actually want to
match against a vulnerability feed.

An SBOM that omits the shipped native runtime is not a partial bill of
materials. It is a document that reads as complete and is not, which is worse
than none at all, because it is the kind of file people tick a box against.

This script fixes that with facts rather than a list. It reads the wheel and
hashes what is really inside it, and it reads the provenance sidecar for the
toolchain versions, which are not recoverable from the artifact afterwards.
Nothing here is hardcoded except the property names.

Usage:

    sbom_supplement.py <sbom.cyclonedx.json> <wheel.provenance.json> <wheel> <out.json>

Standard library only, no network, and deterministic: same inputs, same bytes
out. It adds no timestamp of its own, because a differing timestamp in an
otherwise identical document destroys the one useful property of a
reproducible artifact, which is that you can diff two of them.

What it does not do. It does not invent a version it cannot read, it does not
resolve a MAX library to an upstream source repository, and it does not claim a
license for anything. An unknown stays the string "unknown", which a reader can
act on, unlike a plausible guess.
"""

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

# Wheel members that are native code rather than Python. The extension is built
# by bindings/build.sh; the .dylibs directory is created by
# packaging/build_wheel.sh, which copies the MAX runtime libraries the
# extension links through @rpath.
EXTENSION_SUFFIXES = (".so", ".pyd")
BUNDLED_DIR = ".dylibs/"
BUNDLED_SUFFIXES = (".dylib", ".so")

# From the provenance sidecar. The same names are required by
# packaging/matrix/validate_artifact.py rule R7; if that list changes, this one
# follows it rather than the other way around.
TOOLCHAIN_FIELDS = ("mojo_version", "max_version")
CONTEXT_FIELDS = (
    "pixi_lock_sha256",
    "git_commit",
    "git_dirty",
    "build_host_os",
    "build_host_arch",
    "xcode",
    "metal_toolchain",
    "has_accelerator_at_build",
)

VERSION = re.compile(r"\d+\.\d+(?:\.\d+)?")
UNKNOWN = "unknown"


def version_of(raw: object) -> str:
    """The version inside a tool's banner, or "unknown". Never a guess."""
    if not isinstance(raw, str) or not raw.strip():
        return UNKNOWN
    match = VERSION.search(raw)
    return match.group(0) if match else UNKNOWN


def prop(name: str, value: object) -> dict:
    return {"name": f"mojoboost:{name}", "value": str(value)}


def component(name: str, version: str, purl: str, role: str, sha256: str | None,
              extra: list[dict]) -> dict:
    body = {
        "type": "library",
        "bom-ref": f"mojoboost:{role}:{name}@{version}",
        "name": name,
        "version": version,
        "purl": purl,
        "properties": [prop("role", role)] + extra,
    }
    if sha256:
        body["hashes"] = [{"alg": "SHA-256", "content": sha256}]
    return body


def native_members(wheel: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(extensions, bundled) as (member name, sha256), each sorted by name."""
    extensions: list[tuple[str, str]] = []
    bundled: list[tuple[str, str]] = []
    with zipfile.ZipFile(wheel) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            name = info.filename
            if name.endswith("/"):
                continue
            content = zf.read(name)
            sha = hashlib.sha256(content).hexdigest()
            if BUNDLED_DIR in name and name.endswith(BUNDLED_SUFFIXES):
                bundled.append((name, sha))
            elif name.endswith(EXTENSION_SUFFIXES):
                extensions.append((name, sha))
    return extensions, bundled


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "usage: sbom_supplement.py <sbom.json> <provenance.json> <wheel> <out.json>",
            file=sys.stderr,
        )
        return 2

    sbom_path, prov_path, wheel_path, out_path = (Path(a) for a in argv[1:5])
    for path in (sbom_path, prov_path, wheel_path):
        if not path.is_file():
            print(f"not a file: {path}", file=sys.stderr)
            return 1

    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    if sbom.get("bomFormat") != "CycloneDX":
        print(
            f"{sbom_path} is not CycloneDX (bomFormat={sbom.get('bomFormat')!r}). "
            "Generate the SBOM with format cyclonedx-json.",
            file=sys.stderr,
        )
        return 1

    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        print(f"{prov_path} is not a JSON object", file=sys.stderr)
        return 1
    absent = [k for k in TOOLCHAIN_FIELDS if k not in provenance]
    if absent:
        print(
            f"{prov_path} is missing {', '.join(absent)}; the sidecar is the only "
            "record of what built the wheel and this script will not guess",
            file=sys.stderr,
        )
        return 1

    mojo = version_of(provenance.get("mojo_version"))
    max_version = version_of(provenance.get("max_version"))
    extensions, bundled = native_members(wheel_path)

    if not extensions:
        print(
            f"{wheel_path} contains no compiled extension. Either this is not a "
            "mojoboost wheel or the build produced a pure Python one.",
            file=sys.stderr,
        )
        return 1

    added: list[dict] = []

    added.append(
        component(
            "mojo",
            mojo,
            f"pkg:generic/mojo@{mojo}",
            "toolchain",
            None,
            [
                prop("toolchain.raw", provenance.get("mojo_version", "")),
                prop("toolchain.channel", "https://conda.modular.com/max"),
                prop("note", "compiled the extension; not shipped in the wheel"),
            ],
        )
    )
    added.append(
        component(
            "max",
            max_version,
            f"pkg:generic/max@{max_version}",
            "toolchain",
            None,
            [
                prop("toolchain.raw", provenance.get("max_version", "")),
                prop("toolchain.channel", "https://conda.modular.com/max"),
                prop("note", "source of the bundled runtime libraries below"),
            ],
        )
    )

    for member, sha in extensions:
        name = Path(member).name
        added.append(
            component(
                name,
                mojo,
                f"pkg:generic/{name}@{mojo}",
                "extension",
                sha,
                [
                    prop("wheel.member", member),
                    prop("built.by", "bindings/build.sh"),
                    prop("has_accelerator_at_build",
                         provenance.get("has_accelerator_at_build", UNKNOWN)),
                ],
            )
        )

    for member, sha in bundled:
        name = Path(member).name
        added.append(
            component(
                name,
                max_version,
                f"pkg:generic/{name}@{max_version}",
                "bundled-runtime",
                sha,
                [
                    prop("wheel.member", member),
                    prop("bundled.by", "packaging/build_wheel.sh"),
                    prop("bundled.from", "the MAX conda package, $CONDA_PREFIX/lib"),
                ],
            )
        )

    components = sbom.setdefault("components", [])
    existing = {
        (c.get("name"), h.get("content"))
        for c in components
        if isinstance(c, dict)
        for h in (c.get("hashes") or [{}])
    }
    refs = {c.get("bom-ref") for c in components if isinstance(c, dict)}

    appended = 0
    for item in added:
        sha = (item.get("hashes") or [{}])[0].get("content")
        if sha and (item["name"], sha) in existing:
            continue
        if item["bom-ref"] in refs:
            continue
        components.append(item)
        refs.add(item["bom-ref"])
        appended += 1

    metadata = sbom.setdefault("metadata", {})
    properties = metadata.setdefault("properties", [])
    properties.append(prop("supplemented", "packaging/security/sbom_supplement.py"))
    for field in TOOLCHAIN_FIELDS + CONTEXT_FIELDS:
        if field in provenance:
            properties.append(prop(f"provenance.{field}", provenance[field]))
    properties.append(prop("wheel.sha256", hashlib.sha256(wheel_path.read_bytes()).hexdigest()))
    properties.append(prop("wheel.filename", wheel_path.name))

    out_path.write_text(json.dumps(sbom, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  components before      {len(components) - appended}")
    print(f"  components added       {appended}")
    print(f"  compiled extensions    {len(extensions)}")
    print(f"  bundled runtime libs   {len(bundled)}")
    print(f"  mojo                   {mojo}")
    print(f"  max                    {max_version}")
    if mojo == UNKNOWN or max_version == UNKNOWN:
        print("  NOTE a toolchain version read as unknown; the sidecar did not record one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
