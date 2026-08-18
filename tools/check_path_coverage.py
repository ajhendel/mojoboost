#!/usr/bin/env python3
"""Fail when a training or scoring path appears, or moves, without the path
coverage table being updated.

WHY THIS EXISTS.  This project has three growth policies, two backends, a
dense and a sparse data plane, a single-output and a multiclass plane, and a
host-stepped and a device-owned control plane.  An optimization is written
inside ONE of the bodies that results, it is measured there, and it is never
carried to the others.  Nobody notices, because every body still produces the
right tree; the ones that missed out are only slower.  Six separate incidents
of that shape are catalogued in `docs/design/PATH_COVERAGE.md`.

A document alone rots.  This is the mechanical half of it, and it is
deliberately narrow: it checks the two facts a text file cannot keep true by
itself.

WHAT IT CHECKS.

  R1  BODY INVENTORY.  Every function in `src/mojotrees/` whose name matches
      a grower or batch-scorer pattern must be classified in
      `tools/path_coverage_baseline.txt`, as either a BODY (a real execution
      body, which the coverage table owes a column) or a NOTBODY (a
      forwarder, a predicate, or an unrelated name the patterns caught).  A
      name in neither list FAILS.  That is the rule that fires when somebody
      adds a fourth grower and does not tell the table about it.

  R2  MECHANISM REACH.  For each mechanism the baseline names an entry
      SYMBOL and records the exact set of `file::owner` sites that reference
      it.  The set is recomputed and any difference FAILS, in either
      direction.  Growing the set means the mechanism reached a new body and
      the table's verdict for that cell is now stale.  Shrinking it means a
      body lost the mechanism.  Both need a human to re-issue a verdict; the
      check cannot issue one and does not try.

  R3  NON-REACH CLAIMS, ADVISORY ONLY, never fails the build.  Reports
      docstrings that assert a symbol has no callers.  Run with `--claims`.
      See LIMITS for why this one does not gate.

WHAT IT DOES NOT CHECK, stated plainly because a checker that implies more
than it does would recreate the defect it exists to end.

  * It cannot tell DELIBERATE from UNCARRIED.  That is a judgement about a
    reason, it lives in the prose column of `docs/design/PATH_COVERAGE.md`,
    and a human writes it.  This file only detects that a cell CHANGED.
  * It cannot see a mechanism that is inline rather than a named function.
    The branchless NaN identity in `predict.mojo` is fourteen lines of
    arithmetic with no symbol to count, so no rule here can tell whether a
    walker has it.  Mechanisms of that shape are marked NOT MECHANICALLY
    CHECKABLE in the table and this file makes no claim about them.
  * It matches text, not types.  Mojo methods on different structs may share
    a name, so an R2 site set can contain sites belonging to a same-named
    method elsewhere.  That is harmless for a RATCHET, which only compares
    the set to itself, and it is fatal for any rule that tried to count
    "callers of this method", which is exactly why R3 does not gate.
  * A green run does not mean the paths agree.  It means nothing moved since
    a human last looked.

Usage:
    python3 tools/check_path_coverage.py             # check (R1 + R2)
    python3 tools/check_path_coverage.py --list      # print what it sees
    python3 tools/check_path_coverage.py --claims    # advisory R3 report
    python3 tools/check_path_coverage.py --write-baseline
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "mojotrees")
BASELINE = os.path.join(ROOT, "tools", "path_coverage_baseline.txt")

# Name shapes that a tree grower or a batch scorer has had in this tree.
# Broad on purpose: a false positive costs one NOTBODY line in the baseline,
# a false negative costs an uncovered path nobody hears about.
BODY_PATTERNS = [
    r"_?grow_tree\w*",
    r"_grow_\w+",
    r"_device_search_\w+",
    r"predict_\w*batch\w*",
    r"_?build_tree\w*",
]
BODY_RE = re.compile(
    r"^(\s*)(?:def|fn)\s+((?:" + r"|".join(BODY_PATTERNS) + r"))\s*[\(\[]"
)

DEF = re.compile(r"^(\s*)(?:def|fn)\s+(\w+)")
STRUCT = re.compile(r"^struct\s+(\w+)")

# Phrases that assert a symbol has no callers.  R3 only.
CLAIM_RE = re.compile(
    r"(no caller sets this|nothing calls this|has no callers?|no caller at all"
    r"|no call site|never called|dead at head|no trainer calls"
    r"|this has no caller|had no caller|unreached by any caller)",
    re.I,
)


def source_files():
    return sorted(f for f in os.listdir(SRC) if f.endswith(".mojo"))


def read_lines(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as fh:
        return fh.read().split("\n")


def owners(lines):
    """Per line, the enclosing `Struct.method` or `function` name."""
    out = []
    struct = None
    func = None
    for line in lines:
        m = STRUCT.match(line)
        if m:
            struct, func = m.group(1), None
        else:
            m = DEF.match(line)
            if m:
                if len(m.group(1)) == 0:
                    struct = None
                func = m.group(2)
        if struct and func:
            out.append(struct + "." + func)
        else:
            out.append(func or struct or "<module>")
    return out


def load_source():
    text = {}
    own = {}
    for name in source_files():
        lines = read_lines(name)
        text[name] = lines
        own[name] = owners(lines)
    return text, own


# ---------------------------------------------------------------- R1


def find_bodies(text):
    found = []
    for name in sorted(text):
        for line in text[name]:
            m = BODY_RE.match(line)
            if m:
                found.append(name + "::" + m.group(2))
    return sorted(set(found))


# ---------------------------------------------------------------- R2


def reference_sites(symbol, text, own):
    """`file::owner` for every non-comment, non-docstring reference to
    `symbol` used as a call or a subscript.  Includes the definition site,
    which is what makes a moved definition visible too."""
    use = re.compile(r"(?<![\w])" + re.escape(symbol) + r"\s*[\(\[]")
    sites = set()
    for name in sorted(text):
        in_doc = False
        for i, line in enumerate(text[name]):
            stripped = line.strip()
            quotes = stripped.count('"""')
            if in_doc:
                if quotes:
                    in_doc = False
                continue
            if quotes == 1:
                in_doc = True
                continue
            if stripped.startswith("#"):
                continue
            if use.search(line):
                sites.add(name + "::" + own[name][i])
    return sorted(sites)


# ---------------------------------------------------------------- R3


def claim_report(text):
    rows = []
    for name in sorted(text):
        current = None
        for i, line in enumerate(text[name]):
            m = DEF.match(line)
            if m:
                current = m.group(2)
            hit = CLAIM_RE.search(line)
            if current and hit:
                rows.append((name, current, i + 1, hit.group(0)))
                current = None
    return rows


# ---------------------------------------------------------------- baseline


def parse_baseline(path):
    bodies, notbodies, mechanisms = set(), set(), {}
    section = None
    current = None
    if not os.path.exists(path):
        return bodies, notbodies, mechanisms
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if section == "BODIES":
                bodies.add(line.strip())
            elif section == "NOTBODIES":
                notbodies.add(line.strip())
            elif section == "MECHANISMS":
                if line.startswith("  "):
                    if current:
                        mechanisms[current]["sites"].append(line.strip())
                else:
                    label, _, symbol = line.partition("=")
                    current = label.strip()
                    mechanisms[current] = {
                        "symbol": symbol.strip(),
                        "sites": [],
                    }
    return bodies, notbodies, mechanisms


def write_baseline(path, bodies, notbodies, mechanisms, text, own):
    out = [
        "# Baseline for tools/check_path_coverage.py.  Companion to",
        "# docs/design/PATH_COVERAGE.md, which carries the verdicts; this",
        "# file carries only the facts the script can recompute.",
        "#",
        "# [BODIES]     execution bodies the coverage table owes a column.",
        "# [NOTBODIES]  names the body patterns caught that are not bodies.",
        "#              A name in NEITHER list fails the check.",
        "# [MECHANISMS] `label = symbol`, then one indented `file::owner`",
        "#              per reference site.  The set may not change without",
        "#              this file changing with it.",
        "",
        "[BODIES]",
    ]
    out += sorted(bodies)
    out += ["", "[NOTBODIES]"]
    out += sorted(notbodies)
    out += ["", "[MECHANISMS]"]
    for label in sorted(mechanisms):
        symbol = mechanisms[label]["symbol"]
        out.append("%s = %s" % (label, symbol))
        for site in reference_sites(symbol, text, own):
            out.append("  " + site)
        out.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out).rstrip("\n") + "\n")


# ---------------------------------------------------------------- main


def main(argv):
    text, own = load_source()
    bodies, notbodies, mechanisms = parse_baseline(BASELINE)

    if "--claims" in argv:
        rows = claim_report(text)
        print("R3 advisory: %d docstring non-reach claims" % len(rows))
        for name, symbol, line, phrase in rows:
            print("  %s::%s  :%d  %r" % (name, symbol, line, phrase))
        print(
            "\nADVISORY ONLY.  The claim's subject is often a symbol other"
            "\nthan the enclosing one, and same-named methods on different"
            "\nstructs cannot be told apart by text, so this never gates."
        )
        return 0

    found = find_bodies(text)

    if "--list" in argv:
        print("body candidates (%d)" % len(found))
        for f in found:
            tag = "BODY" if f in bodies else (
                "notbody" if f in notbodies else "UNCLASSIFIED"
            )
            print("  %-12s %s" % (tag, f))
        print("\nmechanisms (%d)" % len(mechanisms))
        for label in sorted(mechanisms):
            sites = reference_sites(mechanisms[label]["symbol"], text, own)
            print("  %s = %s  (%d sites)" % (
                label, mechanisms[label]["symbol"], len(sites)))
            for s in sites:
                print("      " + s)
        return 0

    if "--write-baseline" in argv:
        if not mechanisms:
            print(
                "refusing to write: no [MECHANISMS] block to recompute.\n"
                "This file records a HUMAN's mechanism list; it cannot"
                " invent one.",
                file=sys.stderr,
            )
            return 2
        unclassified = [f for f in found
                        if f not in bodies and f not in notbodies]
        if unclassified:
            print(
                "refusing to write: %d body candidates are unclassified.\n"
                "Classify each as BODY or NOTBODY by hand first, because"
                " that\nis the judgement this file exists to record:"
                % len(unclassified),
                file=sys.stderr,
            )
            for f in unclassified:
                print("  " + f, file=sys.stderr)
            return 2
        write_baseline(BASELINE, bodies, notbodies, mechanisms, text, own)
        print("wrote %s" % os.path.relpath(BASELINE, ROOT))
        return 0

    failures = []

    # R1
    for f in found:
        if f not in bodies and f not in notbodies:
            failures.append(
                "R1 unclassified path: %s\n"
                "    A new grower or scorer appeared.  Decide whether it is"
                " an execution\n"
                "    body.  If it is, give it a column in"
                " docs/design/PATH_COVERAGE.md and\n"
                "    add it to [BODIES]; if it is a forwarder or an unrelated"
                " name, add\n"
                "    it to [NOTBODIES]." % f
            )
    for b in sorted(bodies):
        if b not in found:
            failures.append(
                "R1 recorded body has vanished: %s\n"
                "    Its column in docs/design/PATH_COVERAGE.md is now about"
                " code that\n"
                "    does not exist.  Remove both, together." % b
            )

    # R2
    for label in sorted(mechanisms):
        symbol = mechanisms[label]["symbol"]
        want = sorted(set(mechanisms[label]["sites"]))
        got = reference_sites(symbol, text, own)
        if want != got:
            gained = [s for s in got if s not in want]
            lost = [s for s in want if s not in got]
            detail = []
            for s in gained:
                detail.append("      + %s" % s)
            for s in lost:
                detail.append("      - %s" % s)
            failures.append(
                "R2 reach changed for %s (%s)\n%s\n"
                "    The coverage table's verdict for this mechanism is now"
                " stale.\n"
                "    Re-read the cells in docs/design/PATH_COVERAGE.md, then"
                " rerun\n"
                "    with --write-baseline." % (label, symbol, "\n".join(detail))
            )

    if failures:
        print("FAIL: %d path coverage problem(s)\n" % len(failures))
        for f in failures:
            print("  " + f + "\n")
        return 1

    print(
        "ok: %d body candidates classified, %d mechanisms at their recorded"
        " reach" % (len(found), len(mechanisms))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
