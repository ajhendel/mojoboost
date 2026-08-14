#!/usr/bin/env python3
"""Check that every GitHub Action a workflow uses is pinned to a commit SHA.

A tag is a name the tag's owner can move. `actions/checkout@v4` today and
`actions/checkout@v4` next Tuesday are not required to be the same code, and
nothing in a workflow run notices the difference. A 40 character commit SHA is
the only reference in git that an upstream repository cannot repoint. In a
workflow that mints an OIDC token and publishes an artifact, that distinction
is the whole of the difference between "GitHub vouches for what built this" and
"GitHub vouches for whatever ran".

Usage:

    python3 packaging/security/check_action_pins.py [path ...]

Each path is a workflow file or a directory of them. With no path, this checks
`.github/workflows`. Exit status is 0 when every rule passes and 1 otherwise,
so it works as a CI step or a pre-release gate.

Rules:

    P1  Every `uses:` names a commit SHA, 40 lowercase hex characters.
    P2  Every pinned `uses:` carries a trailing comment naming the version the
        SHA is supposed to be, because a reviewer cannot read a diff of two
        hex strings and Dependabot writes that comment for you.
    P3  A `docker://` reference is pinned by `@sha256:` digest.

Exempt: a local action path (`./...`), which is versioned by this repository's
own commit and cannot be repointed by anyone else.

This is a text scan and not a YAML parse, on purpose. It has to run on a bare
runner with the standard library and no `pip install`, exactly like
`tools/check_parity.py` and `packaging/matrix/validate_matrix.py`. The cost is
that a `uses:` written inside a `run:` block would be read as a real one. That
has not happened, and a false failure here is cheap.

Running this over the whole directory today fails, and should. `ci.yml` and
`gpu-validation.yml` pin by tag, and `release-provenance.yml` carries
placeholders rather than SHAs. See docs/RELEASE_SECURITY.md section 5 and
handoffs/release_10_security.md.
"""

import re
import sys
from pathlib import Path

USES = re.compile(r"^\s*-?\s*uses\s*:\s*(?P<ref>[^\s#]+)\s*(?P<comment>#.*)?$")
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class Result:
    def __init__(self) -> None:
        self.failed = False
        self.checked = 0

    def rule(self, rid: str, ok: bool, detail: str) -> None:
        self.checked += 1
        if not ok:
            self.failed = True
        print(f"{'ok  ' if ok else 'FAIL'} {rid}  {detail}")


def workflow_files(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(q for q in p.iterdir() if q.suffix in (".yml", ".yaml")))
        else:
            found.append(p)
    return found


def check_file(path: Path, result: Result) -> None:
    if not path.exists():
        result.rule("P0", False, f"{path} does not exist")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    seen = 0

    for number, line in enumerate(lines, start=1):
        match = USES.match(line)
        if not match:
            continue
        seen += 1
        ref = match.group("ref").strip("\"'")
        comment = (match.group("comment") or "").strip()
        where = f"{path}:{number}  {ref}"

        if ref.startswith("./"):
            result.rule("P1", True, f"{where}  local action, versioned with this repo")
            continue

        if ref.startswith("docker://"):
            _, _, tail = ref.partition("docker://")
            _, _, digest = tail.partition("@")
            result.rule(
                "P3",
                bool(DIGEST.match(digest)),
                f"{where}  docker reference must end in @sha256:<64 hex>",
            )
            continue

        action, sep, version = ref.partition("@")
        if not sep:
            result.rule("P1", False, f"{where}  no version at all, floats on the default branch")
            continue
        if "/" not in action:
            result.rule("P1", False, f"{where}  not an owner/repo reference")
            continue

        pinned = bool(SHA.match(version))
        result.rule(
            "P1",
            pinned,
            f"{where}  pinned to a commit SHA" if pinned else f"{where}  '{version}' is a tag or a placeholder, not a commit SHA",
        )
        if pinned:
            result.rule(
                "P2",
                comment.startswith("#") and len(comment) > 1,
                f"{where}  trailing comment naming the intended version",
            )

    if seen == 0:
        print(f"     {path}  no `uses:` lines")


def main(argv: list[str]) -> int:
    paths = argv[1:] or [".github/workflows"]
    files = workflow_files(paths)
    if not files:
        print("no workflow files found", file=sys.stderr)
        return 1

    result = Result()
    for path in files:
        check_file(path, result)

    print()
    if result.failed:
        print(f"action pins FAILED: {result.checked} checks over {len(files)} file(s)")
        print("every action must be `owner/repo@<40 hex>` with a trailing version comment")
        print("see docs/RELEASE_SECURITY.md section 5")
        return 1
    print(f"action pins ok: {result.checked} checks over {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
