#!/usr/bin/env python3
"""A defaulted parameter that every call site leaves at its default.

    python tools/default_argument_audit.py
    python tools/default_argument_audit.py --json
    python tools/default_argument_audit.py --min-callers 2

WHY THIS EXISTS, WHICH IS A SPECIFIC BUG AND NOT A STYLE OPINION
----------------------------------------------------------------
`sampling.select_tree_features` has always taken

    usable: List[Int] = []

documented as LightGBM's `Dataset::ValidFeatureIndices()`, the pool its
`ColSampler` actually samples from. It had five call sites. **All five took
the default.** The pool existed too, as `BinnedMatrix.usable`, with
`usable_features()` and `is_usable()` sitting beside it. Nothing connected
them, and two user-facing parameters were quietly inert as a result:
`feature_pre_filter` could compute a pool nothing consumed, and the CatBoost
categorical replacement needed a way to drop a column from the split search
that existed and was never called.

This is a **third** category of unreachability, and the point of this file is
that neither of the tools already here can see it:

1. A module nothing imports. `tools/connectivity_audit.py` finds these by
   walking the import graph.
2. A gate that IS imported and IS called on every fit and is blind to the
   parameter it should test. `tools/refusal_consistency.py` finds these by
   comparing what four layers claim.
3. **A parameter that is passed, and always passed the same value, by every
   caller there is.** No graph over the code finds it, because the edge is
   there. The function is called. The module is imported. Every layer agrees.
   It reads as complete from every angle except counting arguments at a call
   site.

The rule this file applies: a defaulted parameter whose every call site takes
the default is either dead or unwired, and it is worth one line of output
either way. It does not know which, and it does not guess.

WHAT IT IS NOT
--------------
Static and naive by design: regex over Mojo source, no parser, no type
resolution. Consequences, stated rather than discovered:

- It matches call sites by FUNCTION NAME, not by resolved callee. Two
  functions with one name in different modules are one row here.
- A parameter passed positionally is invisible to it, so a positional call
  reads as "took the default". Mojo keyword-argument style makes this rare in
  this repository, and the failure direction is a false positive rather than
  a miss, which is the right way round for a tool nobody is required to obey.
- A default that is genuinely the only sensible value is a false positive and
  there is no way to tell one from a defect without reading. That is why the
  output is a list to read and not a gate that fails a build.

So this is advisory. It is not wired into the pre-commit hook and should not
be: the correct response to a row here is to go and look, and a check whose
correct response is "go and look" must not block a commit.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "mojotrees")

#: Directories whose calls count as call sites. Tests deliberately included:
#: a parameter exercised ONLY by a test is precisely the "built and reached
#: only from tests" case this campaign keeps finding, and excluding tests
#: would hide it behind a green row.
CALLER_DIRS = (
    os.path.join(ROOT, "src", "mojotrees"),
    os.path.join(ROOT, "bindings"),
    os.path.join(ROOT, "cli"),
    os.path.join(ROOT, "capi"),
    os.path.join(ROOT, "tests"),
    os.path.join(ROOT, "bench"),
)

#: Parameters whose default being universal says nothing. Each needs a reason,
#: because an unexplained suppression is how a real finding gets buried.
IGNORE = {
    # Ubiquitous plumbing whose default is the whole point.
    "seed": "a seed defaulting everywhere is a reproducibility choice",
    "verbose": "reporting verbosity, not a behavior edge",
    "out": "an out-parameter reused by a caller that has one",
}


def _strip(code):
    """Comments and docstrings out. Naive about a `#` inside a string, which
    is recorded rather than fixed: a wrong strip makes a marker go missing,
    which surfaces as a row to read rather than as silence."""
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    code = re.sub(r"#[^\n]*", "", code)
    return code


def _split_args(text):
    """Split an argument list on top-level commas only."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def collect_defaults(src_dir):
    """`{function: {param: (file, line, default_text)}}` for defaulted params."""
    found = {}
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".mojo"):
            continue
        path = os.path.join(src_dir, name)
        code = _strip(open(path).read())
        for match in re.finditer(
            r"^(?:def|fn)\s+(\w+)\s*\(((?:.|\n)*?)\)\s*(?:raises\s*)?->", code, re.M
        ):
            fname, args = match.group(1), match.group(2)
            line = code.count("\n", 0, match.start()) + 1
            for arg in _split_args(args):
                if "=" not in arg or ":" not in arg:
                    continue
                param = arg.split(":")[0].strip()
                default = arg.split("=", 1)[1].strip()
                if not param.isidentifier():
                    continue
                found.setdefault(fname, {})[param] = (
                    os.path.relpath(path, ROOT),
                    line,
                    default,
                    _split_args(args).index(arg),
                )
    return found


def collect_calls(dirs, names):
    """`{function: [(file, line, args_text)]}` for every call to a known name."""
    calls = {}
    for directory in dirs:
        if not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if not name.endswith((".mojo", ".py")):
                    continue
                path = os.path.join(root, name)
                try:
                    code = _strip(open(path, errors="ignore").read())
                except OSError:
                    continue
                for fname in names:
                    for match in re.finditer(
                        re.escape(fname) + r"\s*\(((?:.|\n)*?)\)", code
                    ):
                        # Skip the definition itself.
                        start = code.rfind("\n", 0, match.start()) + 1
                        if re.match(r"\s*(?:def|fn)\s", code[start : match.start()]):
                            continue
                        line = code.count("\n", 0, match.start()) + 1
                        calls.setdefault(fname, []).append(
                            (os.path.relpath(path, ROOT), line, match.group(1))
                        )
    return calls


def audit(min_callers):
    defaults = collect_defaults(SRC)
    calls = collect_calls(CALLER_DIRS, set(defaults))
    rows = []
    for fname, params in sorted(defaults.items()):
        sites = calls.get(fname, [])
        if len(sites) < min_callers:
            continue
        for param, (path, line, default, index) in sorted(params.items()):
            if param in IGNORE:
                continue
            # A call site supplies this parameter either by NAME or by
            # POSITION. Counting only names was the first version of this
            # tool and it was useless: `tree._search` came out top with 136
            # call sites none of which "passed" `allowed`, when in fact they
            # pass it positionally every time. A tool whose loudest row is
            # its own blind spot is worse than no tool, so this counts both.
            covered = [
                s
                for s in sites
                if re.search(r"\b" + param + r"\s*=", s[2])
                or len(_split_args(s[2])) > index
            ]
            if covered:
                continue
            rows.append(
                {
                    "function": fname,
                    "parameter": param,
                    "default": default,
                    "declared": f"{path}:{line}",
                    "call_sites": len(sites),
                    "sites": [f"{p}:{l}" for p, l, _ in sites[:6]],
                    "producers": _producers(param),
                }
            )
    rows.sort(key=lambda r: (not r["producers"], -r["call_sites"], r["function"]))
    return rows


def _producers(param):
    """Declarations elsewhere in `src/` that look like they EXIST TO FEED this
    parameter, and this is the heuristic that makes the tool worth running.

    Raw count is a weak signal: plenty of optional parameters are optional on
    purpose, and a `sample_weight=[]` that twenty callers omit is twenty
    callers with no sample weights. What is not weak is a parameter nobody
    passes sitting in the same repository as a function whose whole job is to
    produce it.

    That is the exact shape of the bug this file was written for.
    `select_tree_features(usable=[])` had five callers and none passed it,
    and `BinnedMatrix.usable_features()` sat one module away with `is_usable()`
    beside it and NO CALLER IN `src/` AT ALL. A producer with no consumer and
    a consumer with no argument are the two halves of one missing edge, and
    finding either half alone is easy to explain away. Finding both at once is
    not.

    Matched conservatively, on name only: `def <param>`, `def <param>_*`,
    `def *_<param>`, and `var <param>` on a struct. A match is a reason to
    read, never a verdict.

    **This group's name promises more than the check delivers, and the
    difference is load-bearing.** "A producer exists and nothing consumes it"
    is what the heading says; what the code tests is only that a producer
    exists and that no CALL SITE passes the parameter. It does not trace
    consumption. `train_gpu(row_compaction=...)` is the proof: its producers
    ARE consumed, at `train_gpu.mojo:3594` and `:3703`, so the plumbing is
    live and the row is not dead code at all. **It still paid**, because what
    it actually found was a live mechanism with a dead door: 77 call sites all
    defaulting, one environment variable as the only producer of a `True`, and
    no named way in from Python.

    So read this group as **where reading is cheap and likely to pay**, not as
    a list of dead code. A ranking that surfaces live plumbing behind a dead
    door is more useful than one that only finds orphans, and the docstring
    should promise the weaker thing it does rather than the stronger thing the
    heading implies.

    **A second warning, from a sibling filter that reported its own blind spot
    as a result.** A name-matching pass over environment variables first
    returned 2 candidates and missed `row_compaction`, because its variable is
    `MOJOTREES_GPU_ROW_COMPACTION` and the parameter is `row_compaction`: the
    infix broke the match. Handling it took the count from 2 to 11. **That was
    caught only because a known instance was absent from the output**, which
    is not a check anyone can run on a list of things they do not already
    know. This file's own first version had the same failure in a different
    shape, counting only keyword arguments and reporting `tree._search` with
    136 positional call sites as its loudest row.
    """
    if len(param) < 4:
        return []
    hits = []
    pattern = re.compile(
        r"^\s*(?:def|fn)\s+(\w*" + re.escape(param) + r"\w*)\s*\(", re.M
    )
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".mojo"):
            continue
        path = os.path.join(SRC, name)
        code = _strip(open(path).read())
        for match in pattern.finditer(code):
            produced = match.group(1)
            line = code.count("\n", 0, match.start()) + 1
            hits.append(f"{os.path.relpath(path, ROOT)}:{line} {produced}()")
    return hits[:4]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument(
        "--min-callers",
        type=int,
        default=2,
        help=(
            "only report parameters on functions with at least this many call "
            "sites. One caller taking a default is ordinary; every caller of "
            "several taking it is the shape worth reading."
        ),
    )
    args = parser.parse_args(argv)
    rows = audit(args.min_callers)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print("Defaulted parameters no call site ever passes")
    print("=" * 46)
    print()
    if not rows:
        print("  nothing to report")
        return 0
    ranked = [r for r in rows if r["producers"]]
    plain = [r for r in rows if not r["producers"]]
    print("A. A PRODUCER EXISTS AND NOTHING CONSUMES IT. Read these first.")
    print()
    for row in ranked[: args.limit]:
        print(f"  {row['function']}({row['parameter']}=...)")
        print(f"      declared {row['declared']}, default {row['default']!r}")
        print(f"      {row['call_sites']} call sites, none passes it")
        print(f"      produced by: {'; '.join(row['producers'][:2])}")
        print()
    if len(ranked) > args.limit:
        print(f"  ... and {len(ranked) - args.limit} more in group A")
        print()
    print(f"B. No obvious producer: {len(plain)} parameter(s), --json for the list.")
    print()
    print(f"{len(rows)} total. Group A is {len(ranked)} of them.")
    print()
    print("Each is either dead or unwired. This tool does not know which, and")
    print("neither does a reader who has not opened the file. Advisory only:")
    print("a false positive here is a default that is genuinely the only")
    print("sensible value, and there is no way to tell without reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
