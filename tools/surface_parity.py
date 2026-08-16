#!/usr/bin/env python3
"""Does every entry surface answer the same question the same way?

    python tools/surface_parity.py
    python tools/surface_parity.py --json
    python tools/surface_parity.py --parameter random_strength

WHY THIS EXISTS
---------------
This repository has four ways in, and on 2026-08-16 it answered the same
question differently through them **four separate times in one day**:

- `auto_learning_rate` is honored from the CLI and the C ABI, both of which
  build a `TrainConfig` and call `resolved_learning_rate`. The Python
  extension never builds a `TrainConfig`, so the parameter is unreachable on
  the surface every benchmark and every pip user goes through.
- `random_strength` was honored by `fit` and refused by `train(params,
  Dataset)`, because `_parse_params` declared `random_strength_ok` at exactly
  one call site. The benchmark arm silently lost the parameter.
- `max_bin` in `MOJOTREES_CATBOOST_MODE` reached `mojotrees.train` and raised,
  because that dict is applied over the TRAINING parameters and `max_bin`
  belongs to the `Dataset`.
- `score_function` reaches `device_policy` on the Python full path
  (`bindings/basic_bindings.mojo:158` puts it on the `DeviceRequest`) and does
  NOT reach it on the Mojo native paths, where `resolve_device`'s
  `score_function: Int = SCORE_L2` is supplied by none of its eight callers.
  **The same block was wrong in opposite directions depending on the caller**:
  on Python it refused a configuration that worked, and on Mojo it was inert
  and would have waved that configuration through.

The recurring defect is not a missing gate. It is that there are four entry
surfaces and no habit of asking a question at all four.
`tools/connectivity_audit.py` walks imports. `tools/refusal_consistency.py`
compares the four refusal LAYERS within a surface.
`tools/default_argument_audit.py` finds a parameter no caller ever passes.
**None of them compares surfaces**, which is what this does.

THE FOUR SURFACES
-----------------
1. `estimator`  -- `python/mojotrees/sklearn.py`, an `__init__` keyword. What
   a scikit-learn user can express.
2. `paramstr`   -- `src/mojotrees/params.mojo` `SUPPORTED_KEYS`, the
   whitespace `key=value` surface. Shared by the CLI and the C ABI.
3. `refused`    -- `params.mojo` `_MOJO_API_ONLY`, names that surface REFUSES
   with a reason rather than calling unknown. A refusal is a good state and
   this tool counts it as such; silence is the defect.
4. `trainconfig` -- read off `TrainConfig` and therefore honored by the CLI
   and C ABI, which build one. The Python extension's `_parse_params` returns
   `BoosterParams` and builds no `TrainConfig`, so a parameter that ONLY
   `TrainConfig` reads is CLI-and-C-ABI-only. That asymmetry is exactly the
   `auto_learning_rate` bug and is the row shape worth reading first.

WHAT IT IS NOT
--------------
Static, regex over source, no parser and no type resolution. It reports where
a NAME appears, not what the code does with it, so every row is a question and
not a verdict. It cannot see a parameter that is expressible under one
spelling on one surface and another spelling elsewhere, beyond the aliases
`params.mojo` lists. Advisory: not wired into the pre-commit hook, because the
correct response to a row is to go and read.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PARAMS = os.path.join(ROOT, "src", "mojotrees", "params.mojo")
SKLEARN = os.path.join(ROOT, "python", "mojotrees", "sklearn.py")
CLI = os.path.join(ROOT, "cli", "mojotrees_cli.mojo")
CAPI = os.path.join(ROOT, "capi", "mojotrees_capi.mojo")
BINDINGS = os.path.join(ROOT, "bindings", "_mojotrees.mojo")


def _read(path):
    try:
        return open(path, errors="ignore").read()
    except OSError:
        return ""


def _comptime_string(code, name):
    """The words of a `comptime NAME = String("...")` block."""
    match = re.search(
        r"comptime\s+" + name + r"\s*=\s*String\(((?:.|\n)*?)\)\s*\n", code
    )
    if not match:
        return set()
    body = " ".join(re.findall(r'"([^"]*)"', match.group(1)))
    return {w.strip().strip(",") for w in body.split() if w.strip().strip(",")}


def _estimator_keywords(code):
    """`__init__` keywords of the estimator base."""
    names = set()
    for match in re.finditer(r"^\s{8}(\w+)\s*=\s*(?:None|[^,\n]+),\s*$", code, re.M):
        names.add(match.group(1))
    return names


def _trainconfig_fields(code):
    """Fields declared on `TrainConfig`."""
    match = re.search(r"struct\s+TrainConfig\b((?:.|\n)*?)\n(?=struct |def |fn )", code)
    if not match:
        return set()
    return set(re.findall(r"^\s*var\s+(\w+)\s*:", match.group(1), re.M))


def audit(low_confidence=False):
    params_code = _read(PARAMS)
    supported = _comptime_string(params_code, "SUPPORTED_KEYS")
    refused = _comptime_string(params_code, "_MOJO_API_ONLY")
    estimator = _estimator_keywords(_read(SKLEARN))
    trainconfig = _trainconfig_fields(params_code)
    bindings_code = _read(BINDINGS)
    cli_code = _read(CLI) + _read(CAPI)

    names = sorted(supported | refused | estimator)
    rows = []
    for name in names:
        in_est = name in estimator
        in_par = name in supported
        in_ref = name in refused
        # A parameter the CLI/C ABI mention by name but the Python extension
        # never does is the auto_learning_rate shape.
        in_cli = bool(re.search(r"\b" + re.escape(name) + r"\b", cli_code))
        in_bind = bool(re.search(r"\b" + re.escape(name) + r"\b", bindings_code))
        finding = None
        if in_cli and not in_bind and not in_ref:
            finding = "cli-and-capi-only"
        elif in_est and not in_par and not in_ref and not in_bind:
            # LOW CONFIDENCE and off by default. `_estimator_keywords` is a
            # regex over eight-space-indented `name = default,` lines, which
            # over-collects badly: it picks up `fit()` keywords (`eval_X`,
            # `eval_group`, `eval_names`), internal plumbing (`contri_addr`,
            # `encode`) and pure aliases (`eta`, `depth`) alongside real
            # parameters. Shipping that as signal would bury the one category
            # this tool has actually validated, which is the failure mode the
            # sibling tool already had once: a loud top row that is the tool's
            # own blind spot. Behind --low-confidence until the extraction is
            # a parser rather than a pattern.
            finding = "estimator-only-and-unbound" if low_confidence else None
        elif in_par and not in_est and not in_ref:
            finding = "paramstr-only"
        rows.append(
            {
                "parameter": name,
                "estimator": in_est,
                "paramstr": in_par,
                "refused_by_name": in_ref,
                "in_bindings": in_bind,
                "in_cli_or_capi": in_cli,
                "trainconfig_field": name in trainconfig,
                "finding": finding,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--parameter")
    parser.add_argument(
        "--low-confidence",
        action="store_true",
        help="include the noisy estimator-only category; see audit()",
    )
    args = parser.parse_args(argv)
    rows = audit(args.low_confidence)
    if args.parameter:
        rows = [r for r in rows if r["parameter"] == args.parameter]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    flagged = [r for r in rows if r["finding"]]
    print("Parameters whose surfaces disagree")
    print("=" * 34)
    print()
    if not flagged:
        print("  nothing to report")
    for row in flagged:
        print(f"  [{row['finding']}] {row['parameter']}")
        print(
            f"      estimator={row['estimator']} paramstr={row['paramstr']} "
            f"refused={row['refused_by_name']} bindings={row['in_bindings']} "
            f"cli/capi={row['in_cli_or_capi']}"
        )
    print()
    print(f"{len(flagged)} to read, of {len(rows)} parameters across four surfaces.")
    print()
    print("A row is a question, not a verdict: this reports where a NAME")
    print("appears, never what the code does with it. Being REFUSED by name on")
    print("a surface is a good state and is not flagged. Silence is the defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
