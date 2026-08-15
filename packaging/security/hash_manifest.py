#!/usr/bin/env python3
"""SHA-256 manifest for release files, written and checked with the standard library.

Why this exists rather than `shasum -a 256 *`: the manifest has to be written on
macOS, checked on Linux, and checked again by whoever downloads the wheel on a
machine with neither coreutils nor a package manager you can assume. Every
platform that can run mojotrees has a Python, so the tool that proves what was
downloaded should need nothing else.

The output is the coreutils format, `<64 hex><two spaces><name>`, so
`sha256sum -c SHA256SUMS` and `shasum -a 256 -c SHA256SUMS` both accept the
file. It carries no comment lines, because those two programs do not agree on
what a comment is.

Usage:

    hash_manifest.py write <manifest> <file> [<file> ...]
    hash_manifest.py verify <manifest> [<directory>]
    hash_manifest.py digest <file>

`write` records base names only. Release files sit in one directory and a
manifest with a build machine's path in it leaks that path to everyone who
downloads it. `verify` resolves those names next to the manifest, or in the
directory given.

What this is not. A digest is an integrity check, not a provenance claim: it
says two files are the same file, and says nothing about who made either one.
The claim about origin comes from the GitHub attestation, verified with
`gh attestation verify`. See packaging/security/verify_release.sh, which does
both in the order that makes sense.
"""

import hashlib
import sys
from pathlib import Path

CHUNK = 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def cmd_write(manifest: Path, files: list[Path]) -> int:
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        print("not a file: " + ", ".join(missing), file=sys.stderr)
        return 1

    by_name: dict[str, Path] = {}
    for path in files:
        if path.name in by_name and by_name[path.name] != path:
            print(
                f"two different files share the base name {path.name}; "
                "a flat manifest cannot describe that",
                file=sys.stderr,
            )
            return 1
        by_name[path.name] = path

    lines = [f"{digest(path)}  {name}" for name, path in sorted(by_name.items())]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {manifest} with {len(lines)} entr{'y' if len(lines) == 1 else 'ies'}")
    for line in lines:
        print(f"  {line}")
    return 0


def cmd_verify(manifest: Path, directory: Path) -> int:
    if not manifest.is_file():
        print(f"no manifest at {manifest}", file=sys.stderr)
        return 1

    failed = 0
    checked = 0
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        expected, sep, name = line.partition("  ")
        if not sep or len(expected) != 64:
            print(f"FAIL {manifest}:{number}  not a sha256sum line: {raw!r}")
            failed += 1
            continue
        target = directory / name.lstrip("*")
        checked += 1
        if not target.is_file():
            print(f"FAIL {name}  missing from {directory}")
            failed += 1
            continue
        actual = digest(target)
        if actual != expected:
            print(f"FAIL {name}  expected {expected}, got {actual}")
            failed += 1
        else:
            print(f"ok   {name}  {actual}")

    print()
    if failed:
        print(f"hash manifest FAILED: {failed} of {checked} file(s) did not match")
        print("do not install, publish, or attest anything in this directory")
        return 1
    print(f"hash manifest ok: {checked} file(s)")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(
            "usage: hash_manifest.py write <manifest> <file>...\n"
            "       hash_manifest.py verify <manifest> [<directory>]\n"
            "       hash_manifest.py digest <file>",
            file=sys.stderr,
        )
        return 2

    command = argv[1]
    if command == "write":
        return cmd_write(Path(argv[2]), [Path(a) for a in argv[3:]])
    if command == "verify":
        manifest = Path(argv[2])
        directory = Path(argv[3]) if len(argv) > 3 else (manifest.parent or Path("."))
        return cmd_verify(manifest, directory)
    if command == "digest":
        path = Path(argv[2])
        if not path.is_file():
            print(f"not a file: {path}", file=sys.stderr)
            return 1
        print(digest(path))
        return 0

    print(f"unknown command {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
