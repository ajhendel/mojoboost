#!/usr/bin/env python3
"""Do the four refusal layers agree about what the accelerator can honor?

    python tools/refusal_consistency.py
    python tools/refusal_consistency.py --json
    python tools/refusal_consistency.py --exit-zero

The same question -- "may a fit that set this parameter run on the GPU?" --
is answered independently in four places, and nothing in this repository
checks that the four answers agree:

1. **estimator**, `python/mojotrees/sklearn.py`. Decides what to tell the
   device policy, by building a `device_selection.Workload`. A parameter it
   does not put on the workload is a parameter the policy cannot see.
2. **policy**, `src/mojotrees/device_policy.mojo`. Decides WHERE a run goes.
   A `BLOCK_*` here sends `device='auto'` to the CPU and makes an explicit
   `device='gpu'` raise.
3. **bindings**, `bindings/_mojotrees.mojo`. The `*_ok` flags on
   `_parse_params`, which refuse a parameter the next trainer does not
   implement. They default False, so a forgotten call site fails closed,
   which is the right default and is why this layer is worth reading.
4. **trainer**, `src/mojotrees/train_gpu.mojo` and `train_gpu_sparse.mojo`.
   Refuses a run that arrived anyway. This is the only protection a caller
   who reaches `train_gpu` directly, from Mojo, has.

Four layers that must agree, with no gate that checks agreement, has already
produced a real contradiction. This is the gate.

HOW A DIFFERENCE IN KIND IS TOLD FROM A CONTRADICTION
-----------------------------------------------------
Layers 2 and 4 are deliberately not the same rule and must not be collapsed.
`device_policy` is a routing decision and the trainer is a doorman; a
parameter can legitimately be blocked at routing AND refused at the door, and
the trainer refusal is not redundant because a Mojo caller skips routing
entirely. So this tool does not report "policy blocks it and the trainer also
refuses it" as a disagreement. It reports four named findings instead, and
each one names the layer pair and the user-visible consequence:

* `contradiction`     one layer says the GPU may honor this and another says
                      it may not. At most one of them is right.
* `unrouted-refusal`  a layer refuses it on the GPU and the policy does not
                      block it, so `device='auto'` routes to the GPU and then
                      raises, instead of routing to the CPU that can honor it.
* `unguarded-device`  the policy blocks it and no GPU trainer refuses it, so a
                      caller who reaches the trainer directly gets a fit that
                      ignored the parameter and reported success.
* `unforwarded-block` the policy has a request field for it and the estimator
                      never puts it on the `Workload`, so the block is
                      unreachable from Python whatever it says.

The first is a defect in what we believe. The other three are defects in
where we check it. Keeping them apart is the whole reason this file has a
taxonomy rather than a boolean.

WHAT THIS IS NOT
----------------
Static analysis only. It reads the four files as text, strips comments and
docstrings, and matches declared markers. It imports nothing from the built
extension, opens no device, and trains nothing, so it runs on a machine with
no accelerator and in a checkout that has never been built. The cost of that
is that it cannot follow a call: it knows `fit` passes `score_function_ok=
True` and it does not know what the trainer then does with the field. Every
claim it makes is therefore a claim about what a layer SAYS, and the findings
are disagreements between statements. A disagreement is a thing to go read;
it is not by itself proof of which side is wrong.

The comment stripper is naive about a `#` inside a string literal. That is
recorded rather than fixed because none of the four files has one on a line
that matters here, and a wrong strip shows up as a marker going missing,
which surfaces as a finding rather than as silence.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

POLICY = "src/mojotrees/device_policy.mojo"
BINDINGS = "bindings/_mojotrees.mojo"
ESTIMATOR = "python/mojotrees/sklearn.py"
TRAINERS = ("src/mojotrees/train_gpu.mojo", "src/mojotrees/train_gpu_sparse.mojo")

#: The three claims a layer can make about one parameter on the GPU.
REFUSES = "refuses-on-gpu"
ALLOWS = "allows-on-gpu"
SILENT = "silent"


class Feature:
    """One parameter, and where each of the four layers speaks about it.

    Every field is a NAME to look for, never a verdict. The verdicts are
    derived from the source at run time, so a lane that changes a layer
    changes this tool's output without editing this table.
    """

    def __init__(
        self,
        name,
        policy_guard=None,
        policy_block=None,
        workload_kwarg=None,
        ok_flag=None,
        trainer_refusals=(),
        note="",
    ):
        #: The parameter, spelled as a user spells it.
        self.name = name
        #: The `request.<field>` expression `device_policy` tests, as text.
        self.policy_guard = policy_guard
        #: The `BLOCK_*` constant that test adds.
        self.policy_block = policy_block
        #: The `device_selection.Workload` keyword the estimator must pass
        #: for the policy to be able to see this parameter at all.
        self.workload_kwarg = workload_kwarg
        #: The `_parse_params` reachability flag, if this parameter has one.
        self.ok_flag = ok_flag
        #: Code expressions whose presence in a GPU trainer is that trainer
        #: refusing this parameter by name.
        self.trainer_refusals = tuple(trainer_refusals)
        #: Anything a reader of a finding needs that the source does not say.
        self.note = note


#: The parameters this tool covers. Every one of them carries a device or
#: feature restriction that more than one layer states.
FEATURES = (
    Feature(
        name="boosting_type='ordered'",
        policy_guard="request.ordered_boosting",
        policy_block="BLOCK_ORDERED_BOOSTING",
        workload_kwarg="ordered_boosting",
        ok_flag="ordered_ok",
        trainer_refusals=("params.ordered.enabled",),
        note=(
            "the rung planes live in ordered_boosting.mojo and are advanced "
            "by boosting.train only"
        ),
    ),
    Feature(
        name="score_function",
        policy_guard="request.score_function != SCORE_L2",
        policy_block="BLOCK_SCORE_FUNCTION",
        workload_kwarg="score_function",
        ok_flag="score_function_ok",
        trainer_refusals=("params.tree.extra.score_function", "extra.score_function"),
        note=(
            "the device SPLIT SEARCH cannot score Cosine and refuses on "
            "ExtraTreeParams.is_active(), but it is not the default: the "
            "host split scan is, it routes through tree._search, and it "
            "reads the field. So 'the GPU cannot honor Cosine' and 'a GPU "
            "fit honors Cosine through the host scan' are both in the tree"
        ),
    ),
    Feature(
        name="random_strength",
        policy_guard=None,
        policy_block=None,
        workload_kwarg=None,
        ok_flag="random_strength_ok",
        trainer_refusals=(),
        note=(
            "the per-split noise is staged on the device by GpuSplitSearcher "
            "but its per-tree scale is computed only by the dense CPU round "
            "loops (boosting._round_random_score_scale). Three _parse_params "
            "call sites declare random_strength_ok as of 2026-08-16 (fit, "
            "train_dataset, booster_update), each chosen by which round loop "
            "it reaches; the rest inherit the False default and refuse by "
            "name. This tool reports what a layer SAYS, and the ok_flag layer "
            "now says three different things at three call sites for good "
            "reason, which is a case it does not model: it reads the flag, "
            "not the routing behind it"
        ),
    ),
    Feature(
        name="leaf_estimation_iterations",
        policy_guard=None,
        policy_block=None,
        workload_kwarg=None,
        ok_flag="leaf_estimation_ok",
        trainer_refusals=("_refuse_leaf_estimation",),
        note=(
            "train_gpu implements it (_check_leaf_estimation_config plus "
            "GpuLeafEstimator); the sparse and custom trainers refuse it"
        ),
    ),
    Feature(
        name="enable_bundle",
        policy_guard="request.bundling",
        policy_block="BLOCK_FEATURE_BUNDLING",
        workload_kwarg="bundling",
        ok_flag=None,
        trainer_refusals=("check_bundling_honored",),
        note="POLICY_VERSION 5 added this block",
    ),
    Feature(
        name="linear_tree",
        policy_guard="request.linear_tree",
        policy_block="BLOCK_LINEAR_TREE",
        workload_kwarg="linear_tree",
        ok_flag=None,
        trainer_refusals=("check_linear_tree_unconnected",),
        note="POLICY_VERSION 5 added this block",
    ),
    Feature(
        name="forced_splits",
        policy_guard="request.forced_splits",
        policy_block="BLOCK_FORCED_SPLITS",
        workload_kwarg="forced_splits",
        ok_flag=None,
        trainer_refusals=(
            "_check_gpu_forced_splits",
            "_check_gpu_forced_splits_sparse",
        ),
        note="POLICY_VERSION 5 added this block",
    ),
)

#: Named here rather than left to be noticed as an absence. Every one of
#: these is a device restriction that at least one layer states and that this
#: tool does NOT check, with what would have to be read to cover it.
NOT_COVERED = (
    (
        "the objective gates (BLOCK_CUSTOM_OBJECTIVE, BLOCK_RANKING_OBJECTIVE, "
        "BLOCK_UNKNOWN_OBJECTIVE)",
        "they are decided from an objective CODE against a table of trainers, "
        "not from a named parameter, so there is no marker to match on the "
        "trainer side",
    ),
    (
        "the shape limits (BLOCK_ROW_LIMIT, BLOCK_BIN_LIMIT, "
        "BLOCK_OUTPUT_LIMIT, BLOCK_MEMORY_BUDGET)",
        "they compare a request against a capability profile, so agreement is "
        "a numeric question and not a yes/no one",
    ),
    (
        "BLOCK_SPARSE_INPUT and BLOCK_VALIDATION_SET",
        "both are routing facts about the call shape rather than about a "
        "parameter the four layers each name",
    ),
    (
        "BLOCK_DERIVATIVE_PRECISION and BLOCK_CONST_HESSIAN_VERIFY",
        "both are driven by an environment variable as well as by a "
        "parameter, and an environment read is a fifth layer this tool does "
        "not model",
    ),
    (
        "the distributed trainers",
        "distributed_strategies calls split.find_best_split itself; it is a "
        "fifth refusal surface and is out of scope here",
    ),
    (
        "whether a refusal MESSAGE is accurate",
        "this tool checks that the layers agree, not that what they agree on "
        "is true",
    ),
)


# --- reading the source ------------------------------------------------


def read(rel):
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


_TRIPLE = re.compile(r'"""(?:.|\n)*?"""')


def strip_comments(text):
    """The source with docstrings and `#` comments blanked, lines preserved.

    Blanked rather than deleted so that a match still reports the line number
    it has in the file a reader will open. A `#` inside a string literal is
    stripped too; see the module docstring for why that is recorded rather
    than fixed.
    """

    def blank(match):
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    text = _TRIPLE.sub(blank, text)
    out = []
    for line in text.split("\n"):
        hash_at = line.find("#")
        out.append(line if hash_at < 0 else line[:hash_at])
    return "\n".join(out)


def find_lines(code, needle):
    """1-based line numbers where `needle` appears in stripped code."""
    return [
        index + 1
        for index, line in enumerate(code.split("\n"))
        if needle in line
    ]


# --- layer 2: the device policy ----------------------------------------


def policy_claims(code, features):
    """What `device_policy` says about each feature.

    A feature is refused when its guard expression is present AND the
    `BLOCK_*` constant it adds is present in a `blocks.add` call. Both are
    required: the constant alone is only a declaration, and the guard alone
    is a test that adds nothing.
    """
    claims = {}
    for feature in features:
        if not feature.policy_block:
            claims[feature.name] = (SILENT, "no block declared for it")
            continue
        guard_lines = find_lines(code, feature.policy_guard or "")
        add_lines = [
            line
            for line in find_lines(code, feature.policy_block)
            if _is_in_blocks_add(code, line)
        ]
        # The guard nearest ABOVE the `blocks.add`, not the last one in the
        # file. `request.<field>` is also read by the report formatter far
        # below the gate, and citing that line sends a reader to a string
        # builder instead of to the decision.
        guarding = [
            line for line in guard_lines if add_lines and line <= add_lines[-1]
        ]
        if guarding and add_lines:
            claims[feature.name] = (
                REFUSES,
                f"{POLICY}:{guarding[-1]} adds {feature.policy_block} "
                f"at :{add_lines[-1]}",
            )
        elif add_lines or guard_lines:
            claims[feature.name] = (
                SILENT,
                f"{feature.policy_block} is declared but its guard "
                f"{feature.policy_guard!r} was not found beside a blocks.add",
            )
        else:
            claims[feature.name] = (SILENT, "no block declared for it")
    return claims


def _is_in_blocks_add(code, line):
    """Whether the `BLOCK_*` on `line` is an argument to `blocks.add(`.

    The constant is also declared once and named once by `block_reason_name`,
    and neither of those is the policy firing. `blocks.add(` opens on the
    line before its first argument in this file's formatting, so the two
    lines above are enough.
    """
    lines = code.split("\n")
    window = "\n".join(lines[max(0, line - 3):line])
    return "blocks.add(" in window


# --- layer 3: the bindings ---------------------------------------------


CALL = re.compile(r"_parse_params\(")


def binding_call_sites(code):
    """Every `_parse_params` call, as (entry point, line, arguments text).

    The definition itself is skipped: its argument list is the defaults, not
    a claim by a caller.
    """
    lines = code.split("\n")
    sites = []
    for index, line in enumerate(lines):
        if not CALL.search(line) or line.lstrip().startswith("def "):
            continue
        entry = "?"
        for back in range(index, -1, -1):
            match = re.match(r"def (\w+)\(", lines[back])
            if match:
                entry = match.group(1)
                break
        depth = 0
        chunk = []
        for forward in range(index, min(index + 24, len(lines))):
            chunk.append(lines[forward])
            depth += lines[forward].count("(") - lines[forward].count(")")
            if forward > index and depth <= 0:
                break
        sites.append((entry, index + 1, "\n".join(chunk)))
    return sites


def _kwarg(args, name):
    """The text of `name=<value>` in a call's argument text, or None."""
    match = re.search(
        name + r"\s*=\s*([^,\n]+(?:\n\s*[^,\n)]+)*?)\s*(?:,|\))", args
    )
    return match.group(1).strip() if match else None


def _resolve_local_bool(code, call_line, value):
    """One level of `var NAME = <expr>` substitution, looking backwards from a
    call site.

    Why this exists, stated because it is a limit and not a feature. A call
    site is allowed to pass a named local rather than an inline expression,
    and one did: `train_dataset` computes

        var scale_is_computed = device == CPU_DEVICE and not d[].is_sparse

    and passes `random_strength_ok=scale_is_computed`, deliberately, so the
    conjunction is readable and easy to correct. Before this function, the
    reader saw a bare identifier, could not match `CPU_DEVICE` in it, and
    reported `unrecognized value` -- which is to say **it went blind at
    exactly the call site somebody had just thought hardest about**. A tool
    whose coverage drops when the code gets clearer is worse than no tool,
    because the gap moves with the attention.

    Deliberately shallow: one hop, same file, nearest preceding assignment,
    no chains and no control flow. A value it cannot resolve is returned
    unchanged and still reported as unrecognized, which is the loud failure
    and is the correct outcome for anything more complicated than this.
    """
    if value is None or "CPU_DEVICE" in value or value in ("True", "False"):
        return value
    name = value.strip()
    if not name.isidentifier():
        return value
    pattern = re.compile(
        r"^\s*var\s+" + re.escape(name) + r"\s*=\s*(.+?)\s*$", re.M
    )
    best = None
    for match in pattern.finditer(code):
        if code.count("\n", 0, match.start()) + 1 < call_line:
            best = match.group(1)
    return best if best is not None else value


def binding_claims(code, features):
    """What the `*_ok` flags say, at the call sites that can reach the GPU.

    A call site reaches the GPU exactly when it hands `_parse_params` a
    device-dependent `cpu=`; every other entry point in the file is CPU-only
    by construction and its flags say nothing about an accelerator.
    """
    sites = binding_call_sites(strip_comments(code))
    gpu_sites = [
        site for site in sites if (_kwarg(site[2], "cpu") or "").startswith("device")
    ]
    claims = {}
    for feature in features:
        if not feature.ok_flag:
            claims[feature.name] = (SILENT, "no reachability flag for it")
            continue
        verdicts = []
        for entry, line, args in gpu_sites:
            value = _kwarg(args, feature.ok_flag)
            value = _resolve_local_bool(code, line, value)
            if value is None:
                verdicts.append((REFUSES, entry, line, "flag omitted, defaults False"))
            elif value == "True":
                verdicts.append((ALLOWS, entry, line, "True on every device"))
            elif "CPU_DEVICE" in value:
                verdicts.append((REFUSES, entry, line, value))
            else:
                verdicts.append((SILENT, entry, line, f"unrecognized value {value!r}"))
        if not verdicts:
            claims[feature.name] = (SILENT, "no GPU-reachable call site")
            continue
        allows = [v for v in verdicts if v[0] == ALLOWS]
        # ALLOWS wins the summary when any GPU-reachable entry point grants
        # it, because one entry point that lets a parameter through to an
        # accelerator is the whole exposure.
        chosen = allows[0] if allows else verdicts[0]
        claims[feature.name] = (
            chosen[0],
            "; ".join(
                f"{BINDINGS}:{line} {entry} {feature.ok_flag}={why}"
                for _, entry, line, why in verdicts
            ),
        )
    return claims


# --- layer 4: the GPU trainers -----------------------------------------


#: The guard functions the DENSE and SPARSE GPU training paths actually run.
#: A refusal inside one of these is the GPU trainer refusing the parameter.
#: A refusal anywhere else in the same file belongs to a different GPU round
#: loop -- the custom-objective one and the multiclass one each have their
#: own -- and saying "the GPU trainer refuses it" on the strength of that
#: would be wrong in the direction that matters: it would manufacture a
#: contradiction with a bindings flag that is scoped to the dense path.
#:
#: Tagged dense or sparse, and compared only against a bindings entry point
#: of the same family. Every GPU-reachable `_parse_params` call site is a
#: DENSE one -- the sparse entry points leave `cpu` at its default, so the
#: sparse GPU trainer is reachable from Mojo and not from this binding -- and
#: comparing a sparse guard's refusal against a dense entry point's flag
#: manufactures a contradiction between two paths that never meet. That is
#: not a hypothetical: `leaf_estimation_iterations` is refused by
#: `train_gpu_sparse._refuse_unhonored` and allowed by `fit`, and both are
#: correct.
GPU_TRAINER_GUARDS = {
    "train_gpu": "dense",
    "train_gpu_with_valid": "dense",
    "grow_tree_gpu": "dense",
    "_check_gpu_booster_params": "dense",
    "_check_gpu_forced_splits": "dense",
    "grow_tree_gpu_sparse": "sparse",
    "train_gpu_sparse": "sparse",
    "_check_gpu_forced_splits_sparse": "sparse",
    "_refuse_unhonored": "sparse",
    "_refuse_bundling": "sparse",
}

_DEF = re.compile(r"^(?:def|fn)\s+(\w+)")


def enclosing_function(code, line):
    """The `def`/`fn` a 1-based line sits in, or None at module level.

    None is how an import is told from a call: a marker listed in a
    multi-line `from .x import (...)` block sits at module level, and
    counting it as a refusal is how `_refuse_leaf_estimation` appeared to be
    refused by a trainer that implements it.
    """
    name = None
    for index, text in enumerate(code.split("\n")[:line], start=1):
        match = _DEF.match(text)
        if match:
            name = match.group(1)
    return name


def trainer_claims(codes, features):
    """Whether either GPU trainer refuses each feature on the path a fit takes.

    The claim is the DENSE GPU path's, because that is the only GPU trainer
    the bindings in this repository can route to. A refusal that lives only
    in the sparse guards, or only in the custom-objective or multiclass GPU
    round loops, is reported as evidence beside the claim and does not become
    a disagreement with a dense flag.
    """
    claims = {}
    for feature in features:
        dense = []
        other = []
        for rel, code in codes.items():
            for marker in feature.trainer_refusals:
                for line in find_lines(code, marker):
                    where = enclosing_function(code, line)
                    if where is None:
                        continue  # an import, not a refusal
                    family = GPU_TRAINER_GUARDS.get(where)
                    cite = f"{rel}:{line} {marker} in {where}"
                    if family == "dense":
                        dense.append(cite)
                    else:
                        other.append(cite + f" [{family or 'other round loop'}]")
        trailer = ("  other GPU paths: " + "; ".join(other)) if other else ""
        if dense:
            claims[feature.name] = (REFUSES, "; ".join(dense) + trailer)
        elif other:
            claims[feature.name] = (
                SILENT,
                "no dense GPU guard names it." + trailer,
            )
        else:
            claims[feature.name] = (
                SILENT,
                "no GPU trainer names it outside comments",
            )
    return claims


# --- layer 1: the estimator --------------------------------------------


WORKLOAD = re.compile(r"Workload\(")


def estimator_claims(code, features):
    """Whether the estimator forwards each feature to the device policy.

    The estimator does not decide; it decides what the policy is allowed to
    know. So its claim mirrors the policy's when the parameter is on the
    `Workload`, and is `silent` when it is not, which is the state that makes
    a policy block unreachable from Python.
    """
    stripped = strip_comments(code)
    lines = stripped.split("\n")
    spans = []
    for index, line in enumerate(lines):
        if not WORKLOAD.search(line):
            continue
        depth = 0
        chunk = []
        for forward in range(index, min(index + 60, len(lines))):
            chunk.append(lines[forward])
            depth += lines[forward].count("(") - lines[forward].count(")")
            if forward > index and depth <= 0:
                break
        spans.append((index + 1, "\n".join(chunk)))
    claims = {}
    for feature in features:
        if not feature.workload_kwarg:
            claims[feature.name] = (
                SILENT,
                "the policy has no request field for it, so there is nothing "
                "to forward",
            )
            continue
        found = [
            line
            for line, args in spans
            if re.search(r"\b" + feature.workload_kwarg + r"\s*=", args)
        ]
        if found:
            claims[feature.name] = (
                "forwarded",
                f"{ESTIMATOR}:{found[0]} puts {feature.workload_kwarg} on the "
                "Workload",
            )
        else:
            claims[feature.name] = (
                SILENT,
                f"no Workload(...) in {ESTIMATOR} passes "
                f"{feature.workload_kwarg}",
            )
    return claims


# --- the findings ------------------------------------------------------


def findings_for(feature, policy, bindings, trainer, estimator):
    out = []
    verdicts = {
        "policy": policy[0],
        "bindings": bindings[0],
        "trainer": trainer[0],
    }
    allows = [name for name, verdict in verdicts.items() if verdict == ALLOWS]
    refuses = [name for name, verdict in verdicts.items() if verdict == REFUSES]

    if allows and refuses:
        out.append(
            {
                "kind": "contradiction",
                "feature": feature.name,
                "says_allowed": allows,
                "says_refused": refuses,
                "consequence": (
                    "at most one of these is right. Either the accelerator "
                    "path honors this parameter and a block is keeping fits "
                    "off a backend that can run them, or it does not and a "
                    "reachability flag is letting a wrong answer through"
                ),
            }
        )

    for layer, claim in (("bindings", bindings), ("trainer", trainer)):
        if verdicts[layer] == REFUSES and policy[0] != REFUSES:
            out.append(
                {
                    "kind": "unrouted-refusal",
                    "feature": feature.name,
                    "refused_by": layer,
                    "consequence": (
                        "device='auto' is not steered away from the GPU, so "
                        "it routes there and then raises, instead of taking "
                        "the CPU that can honor the parameter"
                    ),
                }
            )
            break

    # Deliberately not raised when this feature already has a contradiction.
    # The consequence sentence below asserts that the policy is right and the
    # parameter really is unhonorable on the device; a contradiction is
    # exactly the state in which that has not been settled, and printing both
    # would state a conclusion the tool has not earned.
    contradicted = any(f["kind"] == "contradiction" for f in out)
    if (
        policy[0] == REFUSES
        and trainer[0] != REFUSES
        and bindings[0] != REFUSES
        and not contradicted
    ):
        out.append(
            {
                "kind": "unguarded-device",
                "feature": feature.name,
                "consequence": (
                    "the policy is the only gate. A caller reaching the GPU "
                    "trainer directly, from Mojo or past the estimator, gets "
                    "a fit that ignored the parameter and reported success"
                ),
            }
        )

    if policy[0] == REFUSES and estimator[0] == SILENT and feature.workload_kwarg:
        out.append(
            {
                "kind": "unforwarded-block",
                "feature": feature.name,
                "consequence": (
                    f"the block tests request.{feature.workload_kwarg} and no "
                    "Workload in the estimator sets it, so the block cannot "
                    "fire for any fit that came through Python"
                ),
            }
        )
    return out


def analyze():
    policy_code = strip_comments(read(POLICY))
    bindings_raw = read(BINDINGS)
    estimator_raw = read(ESTIMATOR)
    trainer_codes = {rel: strip_comments(read(rel)) for rel in TRAINERS}

    policy = policy_claims(policy_code, FEATURES)
    bindings = binding_claims(bindings_raw, FEATURES)
    trainer = trainer_claims(trainer_codes, FEATURES)
    estimator = estimator_claims(estimator_raw, FEATURES)

    rows = []
    findings = []
    for feature in FEATURES:
        row = {
            "feature": feature.name,
            "note": feature.note,
            "estimator": estimator[feature.name],
            "policy": policy[feature.name],
            "bindings": bindings[feature.name],
            "trainer": trainer[feature.name],
        }
        rows.append(row)
        findings.extend(
            findings_for(
                feature,
                policy[feature.name],
                bindings[feature.name],
                trainer[feature.name],
                estimator[feature.name],
            )
        )
    return rows, findings


def render(rows, findings):
    out = []
    out.append("refusal consistency: four layers, one question per parameter")
    out.append("=" * 72)
    for row in rows:
        out.append("")
        out.append(f"{row['feature']}")
        for layer in ("estimator", "policy", "bindings", "trainer"):
            verdict, evidence = row[layer]
            out.append(f"    {layer:<10} {verdict:<14} {evidence}")
        if row["note"]:
            out.append(f"    note       {row['note']}")
    out.append("")
    out.append("=" * 72)
    if not findings:
        out.append("no disagreement found across the covered parameters.")
    else:
        out.append(f"{len(findings)} disagreement(s):")
        for finding in findings:
            out.append("")
            out.append(f"  [{finding['kind']}] {finding['feature']}")
            if "says_allowed" in finding:
                out.append(
                    f"    allowed by: {', '.join(finding['says_allowed'])}"
                    f"   refused by: {', '.join(finding['says_refused'])}"
                )
            if "refused_by" in finding:
                out.append(f"    refused by: {finding['refused_by']}")
            out.append(f"    {finding['consequence']}")
    out.append("")
    out.append("=" * 72)
    out.append("NOT COVERED by this tool:")
    for what, why in NOT_COVERED:
        out.append(f"  - {what}")
        out.append(f"      {why}")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument(
        "--exit-zero", action="store_true",
        help="report findings but always exit 0, for a caller that wants the "
             "text without the verdict",
    )
    args = parser.parse_args(argv)
    rows, findings = analyze()
    if args.json:
        print(json.dumps({"claims": rows, "findings": findings}, indent=2))
    else:
        print(render(rows, findings))
    if args.exit_zero:
        return 0
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
