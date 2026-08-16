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

Advisory. Not wired into the pre-commit hook: the correct response to a row is
to go and read it, and a check whose correct answer is "go and read" must not
block a commit.
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
        "--all",
        action="store_true",
        help="include raises that do not read as refusals",
    )
    args = parser.parse_args(argv)
    params, rows = audit(args.set)
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
