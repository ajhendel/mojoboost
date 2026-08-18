#!/usr/bin/env python3
"""Mark citations of handoff files that no longer exist.

WHY THIS EXISTS. On 2026-08-18 this tree carried 118 citations of
`handoffs/*.md` files across 61 files, and every one pointed at a file that
had been deleted. The handoffs directory itself held six survivors out of
sixty-one paths cited.

A citation pointing at a file that does not exist is worse than no citation,
because a reader who cannot check it assumes somebody did. That is not a
theoretical harm here. On the same day, twelve places were found where this
repository contradicted itself and THREE of the twelve pointed at real bugs:
an FMA divergence that made NVIDIA disagree with Metal, a kernel silently
corrupting every oblivious tree deeper than 7, and a sampling parameter
accepted and discarded on the ranker. Every one of those three was a
confident claim nobody had checked.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT. It appends a marker so a
dead citation reads as dead. It does not delete the citation, because the
path is still the key you look the content up by:

    git log --all --diff-filter=D -- handoffs/<name>.md
    git show <sha>^:handoffs/<name>.md

And it does not touch the surrounding claim. A claim whose support is gone is
an unsupported claim, and saying so is honest; silently dropping the citation
would launder it into a bare assertion, which is the failure this tool exists
to prevent rather than commit.

REPOINTING IS BETTER AND THIS IS NOT IT. Where the content moved to a live
document, the right fix is to cite that document. That needs judgment per
site and cannot be mechanical. This tool is the floor, not the ceiling: it
converts a silent lie into a visible one so the repoint can be done later by
someone who knows where the content went.

usage:
    tools/mark_dead_handoff_citations.py --dry-run [path ...]
    tools/mark_dead_handoff_citations.py --apply   [path ...]

With no paths it sweeps src/, bindings/, python/, tools/, tests/, bench/ and
docs/. Pass paths to sweep a subset, which is what you want when other
sessions hold files in a shared checkout.
"""
import os, re, sys

MARKER = " (deleted, recover with git log --all --diff-filter=D --"
CITE = re.compile(r'(handoffs/[A-Za-z0-9_./-]*\.md)')
EXTS = {".mojo", ".py", ".md", ".sh", ".txt", ".json", ".toml"}
DEFAULT_ROOTS = ["src", "bindings", "python", "tools", "tests", "bench", "docs"]


def files(roots):
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for dirpath, _, names in os.walk(root):
            if ".git" in dirpath or "/.pixi" in dirpath:
                continue
            for n in names:
                p = os.path.join(dirpath, n)
                if os.path.splitext(n)[1] in EXTS:
                    yield p


def main(argv):
    apply = "--apply" in argv
    dry = "--dry-run" in argv or not apply
    roots = [a for a in argv[1:] if not a.startswith("--")] or DEFAULT_ROOTS
    total = touched = 0
    for p in files(roots):
        if os.path.basename(p) == os.path.basename(__file__):
            continue
        try:
            s = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        out, n = s, 0
        for path in sorted(set(CITE.findall(s)), key=len, reverse=True):
            if os.path.isfile(path):
                continue
            # already marked, or the marker itself would be a second citation
            if path + MARKER in out:
                continue
            out = out.replace(path, path + MARKER + " " + path + ")")
            n += out.count(path + MARKER) and 1
        if n:
            total += n
            touched += 1
            print(("would mark" if dry else "marked") + " %-56s %d" % (p, n))
            if apply:
                open(p, "w", encoding="utf-8").write(out)
    print("\n%s %d citation group(s) across %d file(s)"
          % ("would mark" if dry else "marked", total, touched))
    if dry:
        print("nothing written; pass --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
