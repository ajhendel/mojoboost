#!/usr/bin/env python3
"""Check that every action a release workflow uses is pinned to a commit.

    python3 packaging/macos/check_action_pins.py
    python3 packaging/macos/check_action_pins.py .github/workflows/release-macos.yml

NOT EXECUTED. It has not been run against any workflow.

Default target is .github/workflows/release-macos.yml, which is the only
workflow this lane owns. The other workflows in this repository pin by tag
(`actions/checkout@v4`) and are deliberately not checked here: changing them
belongs to whoever owns them, and a checker that fails on files it does not own
gets disabled rather than fixed.

The rule
--------
Every `uses:` that names a third-party action must be

    owner/repo@<40 hex characters>   # <human readable version>

A tag is a moving pointer. `actions/checkout@v4` is whatever the `v4` tag points
at when the job starts, and whoever can move that tag can run code inside a job
that has this repository's OIDC identity. For a workflow that publishes to a
package index, that is the whole security boundary. The trailing comment is not
decoration: a bare hash is unreviewable, so the version it corresponds to is
recorded next to it, and this checker requires it.

Placeholders
------------
The workflow ships with `@REPLACE_WITH_SHA` in place of every real hash, because
the lane that wrote it had no network access and a fabricated hash is worse than
an obvious hole: it would be wrong, it would look right, and it would fail
confusingly at dispatch time. That spelling is shared with
.github/workflows/release-provenance.yml so both files are filled in by the same
pass. This checker rejects it, and also rejects a hash of forty zeros, and
prints the exact command that resolves each one. Until they are filled in the
workflow cannot run, which is the intended state.

Standard library only, and it reads the workflow as text rather than parsing
YAML, so it runs on a bare checkout with no PyYAML. The cost of that choice is
that a `uses:` inside a block scalar or a quoted string would be treated as a
real one. There are none, and a false positive here is a visible failure rather
than a silent pass.

Exit status is 0 when every pin is a commit hash and 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / ".github" / "workflows" / "release-macos.yml"

# Two spellings of "not filled in yet". The text one is what this repository's
# release workflows carry; the zeros one is rejected too, because it is the
# other obvious way to write a hash nobody resolved.
TEXT_PLACEHOLDER = "REPLACE_WITH_SHA"
PLACEHOLDER = "0" * 40

USES = re.compile(
    r"^\s*-?\s*uses:\s*(?P<ref>[^\s#]+)\s*(?:#\s*(?P<comment>.*?))?\s*$"
)
PINNED = re.compile(r"^(?P<action>[^@]+)@(?P<sha>[0-9a-f]{40})$")


def check(path: Path) -> bool:
    if not path.exists():
        print(f"FAIL  no such workflow: {path}")
        return False

    ok = True
    found = 0
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        match = USES.match(line)
        if not match:
            continue
        ref = match["ref"].strip("'\"")
        comment = (match["comment"] or "").strip()
        found += 1

        # A local action lives in this repository and is reviewed with it.
        if ref.startswith(("./", "docker://")):
            print(f"ok    {path.name}:{lineno}  {ref}  (local)")
            continue

        pin = PINNED.match(ref)
        if not pin:
            ok = False
            action, _, version = ref.partition("@")
            print(f"FAIL  {path.name}:{lineno}  {ref}")
            if version == TEXT_PLACEHOLDER:
                print("      placeholder; this workflow cannot run until it is filled in.")
                print(f"      {resolve_hint(action + '@' + (comment or 'TAG'))}")
            else:
                print("      not pinned to a 40 character commit hash.")
                print(f"      {resolve_hint(ref)}")
            continue
        if pin["sha"] == PLACEHOLDER:
            ok = False
            print(f"FAIL  {path.name}:{lineno}  {pin['action']}@<placeholder>"
                  f"  # {comment or 'no version comment'}")
            print("      the placeholder is still in place; this workflow cannot run.")
            print(f"      {resolve_hint(pin['action'] + '@' + (comment or 'TAG'))}")
            continue
        if not comment:
            ok = False
            print(f"FAIL  {path.name}:{lineno}  {ref}")
            print("      pinned, but with no comment saying which version this is.")
            print("      A bare hash cannot be reviewed or updated by a human.")
            continue
        print(f"ok    {path.name}:{lineno}  {pin['action']}@{pin['sha'][:12]}...  # {comment}")

    if found == 0:
        print(f"FAIL  {path}: no `uses:` lines found. Either the workflow "
              "changed shape or this checker's regex did.")
        return False
    return ok


def resolve_hint(ref: str) -> str:
    action, _, version = ref.partition("@")
    version = version or "TAG"
    # `commits/<tag>` rather than `git/ref/tags/<tag>`: an annotated tag's ref
    # points at the tag object, not at the commit, and pinning to a tag object
    # hash is a different thing that Actions does not resolve the same way.
    return f"resolve with: gh api repos/{action}/commits/{version} --jq .sha"


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [DEFAULT]
    ok = all(check(p) for p in paths)
    print()
    if ok:
        print("every action is pinned to a commit hash with a version comment.")
        print("That is a check on the reference, not on what it points at. Verify")
        print("that each hash is really the commit the named tag points at:")
        print("  gh api repos/<owner>/<repo>/commits/<tag> --jq .sha")
    else:
        print("unpinned or placeholder actions above. A workflow with an OIDC")
        print("identity must not resolve a mutable reference at dispatch time.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
