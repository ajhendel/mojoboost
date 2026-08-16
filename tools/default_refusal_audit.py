#!/usr/bin/env python3
"""Which refusals does a proposed DEFAULT parameter set walk into?

    python tools/default_refusal_audit.py
    python tools/default_refusal_audit.py --json
    python tools/default_refusal_audit.py --set catboost_defaults

WHY THIS EXISTS
---------------
On 2026-08-16 Andrew decided that mojotrees's shipped defaults become
CatBoost's CPU defaults. Within an hour that proposal had collided with three
separate refusals, each of which was individually correct:

1. `score_function=Cosine` on the GPU. The per-node device search cannot score
   a Cosine ratio, and `_launch_oblivious_search` raises on anything but
   `SCORE_L2` for a real mathematical reason: a level's Cosine score is a ratio
   of two cross-leaf accumulators with one square root at the end, and summing
   per-leaf Cosine gains is not the Cosine of the level.
2. `grow_policy=symmetrictree` against `gpu_tree_tables.mojo:424`, which
   returns `TREE_RESIDENT_DEPTHWISE` for anything that is not leafwise.
3. `bootstrap_type=MVS` on multiclass and sparse, where the sparse arm refuses
   the bundle by name and the multiclass trainer takes no bundle at all, so
   even CatBoost's own Bayesian fallback has nowhere to land.

**Every one of those refusals is right. They are jointly unsatisfiable with the
proposed default set.** That is the insight this file exists to mechanize, and
it is not the same question any other tool here asks:

- `connectivity_audit.py` asks whether a module is imported.
- `refusal_consistency.py` asks whether four layers AGREE about a refusal.
- `default_argument_audit.py` asks whether a parameter is ever passed.
- `surface_parity.py` asks whether four entry surfaces answer alike.

None of them asks **"if this were the default, what would refuse?"** Changing a
default is a compatibility change against every refusal in the tree, and until
now nothing checked a proposed default set against the refusals it would hit.
The three collisions above were each found by a person, one at a time, hours
apart, after the decision was already made.

WHAT IT DOES
------------
For each parameter in a proposed default set, it finds every `raise` in the
native and binding sources whose message names that parameter, and reports them
grouped by parameter with the file, the line and the message. A reader then
decides which fire.

WHAT IT IS NOT
--------------
It does not evaluate conditions. It cannot tell you that a refusal fires only
on the sparse path, or only above a row count, because that needs the call
graph and a type checker and this is a regex over text. **Every row is a site
to read, not a prediction.** The output is deliberately a reading list.

It also cannot see a refusal that does not name its parameter in the message.
That is a real blind spot and it is the same one `refusal_consistency.py` has:
a `BLOCK_*` constant whose message says "this configuration" rather than naming
the key is invisible here. Refusals in this repository are unusually good about
naming the parameter, which is what makes the approach work at all.

HOW IT GATES WITHOUT BECOMING NOISE
-----------------------------------
`--check` is the gated mode and it does NOT fail on every row. Failing on
every row is how a gate gets disabled: this tool finds seventy-odd sites for
the current proposal and most of them are conditions the default never
reaches. A check whose honest output is "go and read seventy things" cannot
block a commit.

So `--check` fails only on a site that has NOT been acknowledged. `ACKNOWLEDGED`
is the same shape as `MONOTONE_EXEMPT` in `check_parity.py` and
`CATBOOST_UNMATCHABLE` in the harness: an entry is an argument, not a
suppression, and it records which of three things is true.

    RESOLVED   the refusal cannot fire for the default, and why
    DIVERGENCE the default resolves to something other than the proposed
               value on that path, recorded as OURS rather than as parity
    BLOCKING   it fires, it is not yet fixed, and the default cannot ship
               until it is

**A BLOCKING entry still fails the check.** That is the point: the four
collisions found by hand today were each discovered after the decision, and a
gate that lets a known-blocking collision sit silently is the same failure one
layer up. What the acknowledgement buys is that a NEW collision is
distinguishable from an old one.

Advisory in its default mode; gated in `--check`.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEARCH_DIRS = (
    os.path.join(ROOT, "src", "mojotrees"),
    os.path.join(ROOT, "bindings"),
    os.path.join(ROOT, "python", "mojotrees"),
)

#: Proposed default sets, by name. Each maps the parameter name a refusal
#: message would use to the value being proposed. The value is reported but
#: not evaluated; it is there so a reader knows what is being asked for.
DEFAULT_SETS = {
    # Andrew, 2026-08-16: mojotrees's shipped defaults become CatBoost's CPU
    # defaults, except n_estimators, where CatBoost's 1000 becomes 500.
    "catboost_defaults": {
        "grow_policy": "symmetrictree",
        "max_depth": 6,
        "n_estimators": 500,
        "learning_rate": "auto",
        "auto_learning_rate": True,
        "boosting_type": "plain",
        "bootstrap_type": "MVS",
        "subsample": 0.8,
        "bagging_temperature": "(Bayesian fallback)",
        "random_strength": 1.0,
        "score_function": "cosine",
        "lambda_l2": 3.0,
        "leaf_estimation_iterations": "(per objective)",
        "max_bin": 254,
        "max_cat_to_onehot": 2,
        "max_ctr_complexity": 1,
        "min_data_in_leaf": 1,
    },
}

#: Sites a human has read and ruled on, keyed by (parameter, file, message
#: prefix). The message prefix rather than a line number because line numbers
#: churn on every edit above them and a gate that fails on unrelated edits is
#: a gate that gets skipped.
#:
#: Seeded 2026-08-16 with the four collisions found by hand. Everything not
#: listed here is unreviewed, which is the state this table exists to make
#: visible.
ACKNOWLEDGED = {
    ("bootstrap_type", "src/mojotrees/model.mojo"): (
        "BLOCKING",
        "train_gpu takes no bootstrap bundle, so MVS as a default breaks "
        "every GPU fit. f9 owns the GPU round loop and is building the draw "
        "and refresh; the per-row weight plane already exists there. Interim: "
        "the default resolves to bootstrap_type=No on the GPU, recorded as "
        "OUR divergence, never a raise",
    ),
    ("bootstrap_type", "src/mojotrees/trainset.mojo"): (
        "BLOCKING",
        "the sparse arm refuses the bundle by name and the multiclass trainer "
        "takes no bundle at all, so CatBoost's own Bayesian fallback has "
        "nowhere to land. lane/bootstrap-multiclass-sparse is building both "
        "round loops. Interim: default resolves to No on those two paths",
    ),
    ("score_function", "src/mojotrees/gpu_split_search.mojo"): (
        "BLOCKING",
        "the oblivious level search raises on anything but L2, and earns it: "
        "a level's Cosine score is a ratio of two cross-leaf accumulators "
        "with one square root at the end, so summing per-leaf Cosine gains is "
        "not the Cosine of the level. No merge order fixes it; f9 is building "
        "oblivious-Cosine as its item (1)",
    ),
    ("grow_policy", "src/mojotrees/gpu_tree_tables.mojo"): (
        "BLOCKING",
        "the resident tree tables return TREE_RESIDENT_DEPTHWISE for anything "
        "that is not leafwise, and symmetrictree is the proposed default. "
        "Whether the oblivious device path goes through these tables at all "
        "is the first question f9's build lane answers",
    ),
    ("boosting_type", "src/mojotrees/ordered_boosting.mojo"): (
        "RESOLVED",
        "every refusal naming boosting_type in this file is about "
        "'ordered'. The proposed default is 'plain', which is what these "
        "paths already do",
    ),
    ("boosting_type", "src/mojotrees/train_gpu.mojo"): (
        "RESOLVED",
        "refuses boosting_type='ordered'; the default is 'plain'",
    ),
    ("boosting_type", "src/mojotrees/train_gpu_sparse.mojo"): (
        "RESOLVED",
        "refuses boosting_type='ordered'; the default is 'plain'",
    ),
    ("boosting_type", "src/mojotrees/boosting.mojo"): (
        "RESOLVED",
        "refuses boosting_type='ordered'; the default is 'plain'",
    ),
}

#: Words whose presence in a raise message means it is a refusal rather than a
#: validation error. Not used to filter, only to label, because the difference
#: matters to a reader and guessing it wrong should not hide a row.
REFUSAL_WORDS = (
    "not implemented",
    "not honored",
    "not reachable",
    "not supported",
    "cannot",
    "refus",
    "does not",
)


def _sources():
    for directory in SEARCH_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith((".mojo", ".py")):
                yield os.path.join(directory, name)


def _raise_blocks(text):
    """(line, message) for each raise, message flattened to one line.

    Mojo's `raise Error(` takes a comma-separated list of parts that are
    concatenated, and Python's `raise ValueError(` takes implicit string
    concatenation across lines. Both flatten the same way for this purpose:
    pull every double-quoted run out of the argument list and join them.
    """
    out = []
    for match in re.finditer(r"raise\s+\w*Error\(", text):
        # Depth counting must SKIP string literals. A message containing
        # "(0, 1]" is common in this repository's range checks, and counting
        # the paren inside it ran the scan past the closing paren and merged
        # several unrelated raises plus the following docstring into one row.
        # The first version of this tool did exactly that and its output was
        # unreadable in a way that looked like the source was unreadable.
        depth, i, quote = 0, match.end() - 1, ""
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if text.startswith(quote, i):
                    i += len(quote)
                    quote = ""
                    continue
            elif text.startswith('"""', i):
                quote = '"""'
                i += 3
                continue
            elif ch == '"':
                quote = '"'
                i += 1
                continue
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[match.end() : i]
        parts = re.findall(r'"([^"]*)"', body)
        if not parts:
            continue
        line = text.count("\n", 0, match.start()) + 1
        out.append((line, " ".join(p.strip() for p in parts if p.strip())))
    return out


def audit(default_set):
    params = DEFAULT_SETS[default_set]
    rows = {name: [] for name in params}
    for path in _sources():
        try:
            text = open(path, errors="ignore").read()
        except OSError:
            continue
        blocks = _raise_blocks(text)
        for name in params:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b")
            for line, message in blocks:
                if pattern.search(message):
                    rows[name].append(
                        {
                            "file": os.path.relpath(path, ROOT),
                            "line": line,
                            "message": message,
                            "looks_like_refusal": any(
                                w in message.lower() for w in REFUSAL_WORDS
                            ),
                        }
                    )
    return params, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--set", default="catboost_defaults", choices=sorted(DEFAULT_SETS))
    parser.add_argument(
        "--check",
        action="store_true",
        help="gated mode: exit 1 on an unacknowledged or BLOCKING collision",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include raises that do not read as refusals",
    )
    args = parser.parse_args(argv)
    params, rows = audit(args.set)

    if args.check:
        unreviewed, blocking = [], []
        for name, sites in sorted(rows.items()):
            for site in sites:
                if not site["looks_like_refusal"]:
                    continue
                verdict = ACKNOWLEDGED.get((name, site["file"]))
                if verdict is None:
                    unreviewed.append((name, site))
                elif verdict[0] == "BLOCKING":
                    blocking.append((name, site, verdict[1]))
        for name, site in unreviewed:
            print(
                f"UNREVIEWED {name} -> {site['file']}:{site['line']}\n"
                f"    {site['message'][:160]}\n"
                f"    Read it, then add ({name!r}, {site['file']!r}) to "
                "ACKNOWLEDGED as RESOLVED, DIVERGENCE or BLOCKING with the "
                "reason."
            )
        seen = set()
        for name, site, why in blocking:
            if (name, site["file"]) in seen:
                continue
            seen.add((name, site["file"]))
            print(f"BLOCKING {name} -> {site['file']}: {why}")
        total = len(unreviewed) + len(seen)
        if total == 0:
            print(f"default set '{args.set}': no unreviewed or blocking collisions")
            return 0
        print()
        print(
            f"{len(unreviewed)} unreviewed, {len(seen)} blocking. A BLOCKING "
            "entry fails on purpose: it is a collision somebody has read and "
            "not yet fixed, and the default cannot ship over it."
        )
        return 1

    if args.json:
        print(json.dumps({"set": args.set, "params": params, "sites": rows}, indent=2))
        return 0

    print(f"Refusals a default set of '{args.set}' would walk into")
    print("=" * 52)
    print()
    total = 0
    for name in sorted(rows):
        sites = rows[name]
        if not args.all:
            sites = [s for s in sites if s["looks_like_refusal"]]
        if not sites:
            continue
        print(f"  {name} = {params[name]!r}   ({len(sites)} site(s))")
        for site in sites[:6]:
            print(f"      {site['file']}:{site['line']}")
            print(f"        {site['message'][:150]}")
        if len(sites) > 6:
            print(f"      ... and {len(sites) - 6} more")
        print()
        total += len(sites)
    quiet = [n for n in sorted(rows) if not rows[n]]
    if quiet:
        print("  No raise names these at all: " + ", ".join(quiet))
        print()
    print(f"{total} site(s) to read.")
    print()
    print("Every row is a site to READ, not a prediction. This does not")
    print("evaluate conditions: it cannot tell you a refusal fires only on the")
    print("sparse path or only above a row count. A default set is compatible")
    print("with the tree when a person has read these and said so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
