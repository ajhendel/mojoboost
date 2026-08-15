#!/usr/bin/env python3
"""Generate and check mojotrees's public API snapshot.

    python3 tools/api_snapshot.py --check   # exit 0 if the snapshot matches
    python3 tools/api_snapshot.py --write   # regenerate it in place

The snapshot is `compatibility/api_snapshot.json`. Its normative shape is
`compatibility/SNAPSHOT_SCHEMA.md`, which is the authority; this file is
one implementation of that document and not a second definition of it.
`docs/COMPATIBILITY_POLICY.md` section 11.2 is the table that says which
kind of difference is breaking and which is additive, and section 2 is
what "public" means.

Modeled on `tools/check_parity.py`, and constrained the same three ways,
for the same reason. A check that needs a build gets skipped the first day
it is inconvenient.

- **Standard library only.** `ast` for Python sources, `re` for Mojo and C,
  `tomllib` for TOML. Nothing installed, nothing vendored.
- **Builds nothing, imports no part of mojotrees.** It parses text. It
  never needs a compiled extension module and runs on a bare runner in
  seconds. The one module it does import is `check_parity`, a sibling in
  this directory, and section "Reuse" below says why.
- **Deterministic output.** `indent=2, sort_keys=True`, one trailing
  newline. Regenerating with no source change produces byte-identical
  output, so the diff is the whole signal.

Reuse
-----
`mojo_export_names()` in `check_parity.py` already parses
`src/mojotrees/__init__.mojo`, and it already handles both spellings of
the import block. This tool needs the same names grouped by module, so it
has its own parser for the grouping and then checks the flattened result
against `check_parity`'s (invariant I11). Two independent parsers over one
file drift; a parser and a cross-check do not. If the import fails the run
continues with I11 recorded as underived, because a snapshot that refuses
to generate is worse than one that generates with a named gap.

What this tool does NOT do
--------------------------
It does not decide anything. `--check` reports each difference with its
classification and exits 1 if there were any. Whether a `breaking` line
blocks a release is the release gate's call, and whether it is acceptable
is a human's. A tool that decided would be ignored the first time it was
wrong.

It also runs no git command. `--commit <sha>` is how a caller records the
revision, so that the tool behaves identically inside a release tarball
with no `.git` directory.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

# Bumped when a parser here changes what it derives, so that a diff caused
# by this tool is separable from a diff caused by the tree. It is not the
# snapshot's schema version; that is SCHEMA_VERSION and it lives in
# compatibility/SNAPSHOT_SCHEMA.md.
TOOL_VERSION = 1
SCHEMA_VERSION = 2

ROOT = Path(__file__).resolve().parent.parent

SNAPSHOT = ROOT / "compatibility" / "api_snapshot.json"
LEGACY_SNAPSHOT = ROOT / "tests" / "parallel" / "api_snapshot_manifest.json"
DEPRECATIONS = ROOT / "compatibility" / "deprecations.toml"

PY_PKG = ROOT / "python" / "mojotrees"
PY_API = PY_PKG / "__init__.py"
PY_BASIC = PY_PKG / "basic.py"
PY_CALLBACK = PY_PKG / "callback.py"
PY_EVAL = PY_PKG / "_eval.py"
PY_INSPECTION = PY_PKG / "inspection.py"
PYPROJECT = ROOT / "python" / "pyproject.toml"

MOJO_INIT = ROOT / "src" / "mojotrees" / "__init__.mojo"
MOJO_BOOSTING = ROOT / "src" / "mojotrees" / "boosting.mojo"
MOJO_PARAMS = ROOT / "src" / "mojotrees" / "params.mojo"
MOJO_SERIALIZE = ROOT / "src" / "mojotrees" / "serialize.mojo"
MOJO_DUMP = ROOT / "src" / "mojotrees" / "model_dump.mojo"
BINDINGS = ROOT / "bindings" / "_mojotrees.mojo"

CAPI_HEADER = ROOT / "capi" / "mojotrees.h"
PIXI = ROOT / "pixi.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
POLICY = ROOT / "docs" / "COMPATIBILITY_POLICY.md"

# Directories scanned for MOJOTREES_* string literals. `tools/`, `bench/`,
# and `packaging/` are deliberately absent: a variable only a benchmark
# reads is not a surface of the library.
ENV_SCAN_DIRS = ("src", "bindings", "python", "capi", "cli")
ENV_SCAN_SUFFIXES = (".mojo", ".py", ".h", ".c")

# The estimator classes whose surface is public under policy section 2.
ESTIMATORS = ("MojoTreesRegressor", "MojoTreesClassifier", "MojoTreesRanker")

# The Mojo side of the reset slot contract spells two parameters
# differently from the Python side. The pair is a wire format (policy
# section 9.3), so the translation is written down here rather than
# inferred, and invariant I7 compares through it.
RESET_NAME_MAP = {
    "min_child_hess": "min_sum_hessian_in_leaf",
    "lambda_reg": "lambda_l2",
}

# A field the tool could not derive. Distinct from None, which is a real
# JSON value some fields legitimately take, and never written to the file:
# it is replaced by None and its path is recorded in meta.underived.
UNDERIVED = object()


class Deriver:
    """Accumulates the snapshot and the complaints raised while building
    it. Every parser reports through this rather than raising, because one
    unparseable block should not cost the other twenty."""

    def __init__(self):
        self.underived: list[str] = []
        self.carried: list[str] = []
        self.problems: list[str] = []
        self.notes: list[str] = []

    def gap(self, path: str, why: str) -> None:
        self.underived.append(path)
        self.notes.append(f"underived {path}: {why}")

    def problem(self, message: str) -> None:
        self.problems.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def read(path: Path, d: Deriver) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        d.problem(f"cannot read {rel(path)}: {exc}")
        return ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# -- Python parsing ------------------------------------------------------


def parse_py(path: Path, d: Deriver):
    text = read(path, d)
    if not text:
        return None
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        d.problem(f"cannot parse {rel(path)}: {exc}")
        return None


def module_constants(tree) -> dict:
    """Module-level `NAME = <literal>` assignments, for resolving a default
    that is written as a name rather than as a literal.

    Parsing rule 2 in the schema: `_Base.__init__` has
    `lambda_l2=_LAMBDA_L2`. Recording the string "_LAMBDA_L2" would either
    report drift forever or, worse, never notice when the constant's value
    changed.
    """
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
    return out


def value_of(node, consts: dict):
    """A literal, a module-level constant by name, or UNDERIVED."""
    if node is None:
        return None
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return UNDERIVED


def dunder_all(tree, d: Deriver, where: str) -> list:
    """`__all__` in source order. Order is preserved deliberately: sorting
    would hide a rewrite, and a reordering is a visible, harmless diff."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "__all__"
            for t in node.targets
        ):
            continue
        elts = getattr(node.value, "elts", [])
        return [e.value for e in elts if isinstance(e, ast.Constant)]
    d.gap(f"{where}.all", "no __all__ assignment found")
    return []


def assign_in(body, name: str):
    """The value node of `name = ...` in a class or module body."""
    for node in body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    return None


def class_named(tree, name: str):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def func_named(body, name: str):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def signature(fn, consts: dict) -> dict:
    """Argument names to defaults, in declaration order.

    An argument with no default records the string "required", which is a
    value a caller can read as such and which no real default collides
    with. `*args` and `**kwargs` record their own markers, because an
    estimator that forwards shared hyperparameters through `**kwargs` is a
    documented deviation from scikit-learn and its presence is a fact
    about the surface.
    """
    out = {}
    positional = list(fn.args.posonlyargs) + list(fn.args.args)
    defaults = list(fn.args.defaults)
    first_default = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if arg.arg == "self":
            continue
        if i < first_default:
            out[arg.arg] = "required"
        else:
            out[arg.arg] = value_of(defaults[i - first_default], consts)
    if fn.args.vararg is not None:
        out["*" + fn.args.vararg.arg] = "varargs"
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        out[arg.arg] = "required" if default is None else value_of(
            default, consts
        )
    if fn.args.kwarg is not None:
        out["**" + fn.args.kwarg.arg] = "varkw"
    return out


def decorator_names(fn) -> set:
    names = set()
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
        elif isinstance(dec, ast.Call):
            inner = dec.func
            if isinstance(inner, ast.Name):
                names.add(inner.id)
            elif isinstance(inner, ast.Attribute):
                names.add(inner.attr)
    return names


def public_methods(cls, consts: dict) -> tuple:
    """(methods, properties, classmethods) for a class body. A name
    starting with an underscore is internal by policy section 2 and is not
    recorded, with the dunder protocol methods the exception: they are the
    pickle and scikit-learn hooks, and their presence is public."""
    protocols = {
        "__getstate__",
        "__setstate__",
        "__sklearn_tags__",
        "__sklearn_is_fitted__",
        "__repr__",
        "__len__",
        "__iter__",
    }
    methods, properties, classmethods = {}, [], []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name.startswith("_") and name not in protocols:
            continue
        decs = decorator_names(node)
        if "property" in decs:
            properties.append(name)
            continue
        if "classmethod" in decs:
            classmethods.append(name)
        methods[name] = signature(node, consts)
    return methods, sorted(properties), sorted(classmethods)


# -- Python blocks -------------------------------------------------------


def python_block(d: Deriver) -> dict:
    tree = parse_py(PY_API, d)
    if tree is None:
        d.gap("python", "python/mojotrees/__init__.py did not parse")
        return {}
    consts = module_constants(tree)
    out = {"module": "mojotrees"}
    out["all"] = dunder_all(tree, d, "python")

    lazy_subs = value_of(assign_in(tree.body, "_LAZY_SUBMODULES"), consts)
    lazy_attrs = value_of(assign_in(tree.body, "_LAZY_ATTRS"), consts)
    if lazy_subs is UNDERIVED or lazy_subs is None:
        d.gap("python.lazy_submodules", "_LAZY_SUBMODULES not a literal")
        lazy_subs = []
    if lazy_attrs is UNDERIVED or lazy_attrs is None:
        d.gap("python.lazy_attributes", "_LAZY_ATTRS not a literal")
        lazy_attrs = {}
    out["lazy_submodules"] = sorted(lazy_subs)
    out["lazy_attributes"] = dict(sorted(lazy_attrs.items()))

    base = class_named(tree, "_Base")
    if base is None:
        d.gap("python.shared_estimator_parameters", "_Base is gone")
        out["shared_estimator_parameters"] = {}
        out["fitted_attributes"] = []
        out["parameter_aliases"] = {}
    else:
        init = func_named(base.body, "__init__")
        if init is None:
            d.gap("python.shared_estimator_parameters", "_Base.__init__ gone")
            out["shared_estimator_parameters"] = {}
        else:
            out["shared_estimator_parameters"] = signature(init, consts)
        fitted = value_of(assign_in(base.body, "_FITTED_ATTRS"), consts)
        if fitted is UNDERIVED or fitted is None:
            d.gap("python.fitted_attributes", "_FITTED_ATTRS not a literal")
            out["fitted_attributes"] = []
        else:
            out["fitted_attributes"] = list(fitted)
        out["parameter_aliases"] = alias_pairs(base, consts, d)

    out["estimators"] = {}
    for name in ESTIMATORS:
        cls = class_named(tree, name)
        if cls is None:
            d.gap(f"python.estimators.{name}", "class is gone")
            continue
        init = func_named(cls.body, "__init__")
        methods, properties, classmethods = public_methods(cls, consts)
        objectives = value_of(assign_in(cls.body, "_OBJECTIVES"), consts)
        entry = {
            "bases": [
                b.id for b in cls.bases if isinstance(b, ast.Name)
            ],
            "own_parameters": signature(init, consts) if init else {},
            "methods": methods,
            "properties": properties,
            "classmethods": classmethods,
        }
        if objectives is UNDERIVED or objectives is None:
            entry["objective_names"] = []
        else:
            entry["objective_names"] = sorted(objectives)
        out["estimators"][name] = entry

    out["callbacks"] = callbacks_block(d)
    out["eval_metric_names"] = eval_block(d)
    out["functional_api"] = functional_block(d)
    out["inspection"] = inspection_block(d)
    return out


def alias_pairs(base, consts: dict, d: Deriver) -> dict:
    """Alias to canonical, from the `self._resolve_alias(...)` call sites.

    There is no alias table in `__init__.py`; the pairs are expressed as
    calls, and each carries a third argument that is the default used when
    neither spelling is set. That default is a second place a default
    lives, so it is recorded, and a pair whose call sites disagree about
    it is an error rather than a merge.
    """
    pairs: dict = {}
    for node in ast.walk(base):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "_resolve_alias":
            continue
        if len(node.args) < 3:
            continue
        canonical, alias = node.args[0], node.args[1]
        if not (
            isinstance(canonical, ast.Constant)
            and isinstance(alias, ast.Constant)
        ):
            continue
        default = value_of(node.args[2], consts)
        if default is UNDERIVED:
            d.gap(
                f"python.parameter_aliases.{alias.value}.fallback_default",
                "third argument of _resolve_alias is not a literal",
            )
            default = None
        entry = pairs.setdefault(
            alias.value,
            {
                "canonical": canonical.value,
                "fallback_default": default,
                "sites": 0,
            },
        )
        entry["sites"] += 1
        if entry["canonical"] != canonical.value:
            d.problem(
                f"alias {alias.value!r} resolves to "
                f"{entry['canonical']!r} at one site and "
                f"{canonical.value!r} at another"
            )
        if entry["fallback_default"] != default:
            d.problem(
                f"alias {alias.value!r} falls back to "
                f"{entry['fallback_default']!r} at one site and "
                f"{default!r} at another; a default that differs between "
                f"a first fit and a continued one is a silent divergence"
            )
    return dict(sorted(pairs.items()))


def callbacks_block(d: Deriver) -> dict:
    tree = parse_py(PY_CALLBACK, d)
    if tree is None:
        d.gap("python.callbacks", "callback.py did not parse")
        return {}
    consts = module_constants(tree)
    out = {"module": "mojotrees.callback"}
    out["all"] = dunder_all(tree, d, "python.callbacks")

    # Parsing rule 3: CallbackEnv is a namedtuple() call, not a class, so
    # the field list is the second positional argument of the call and is
    # not reachable from a ClassDef walk.
    env = assign_in(tree.body, "CallbackEnv")
    fields = UNDERIVED
    if isinstance(env, ast.Call) and len(env.args) >= 2:
        fields = value_of(env.args[1], consts)
    if fields is UNDERIVED or fields is None:
        d.gap("python.callbacks.env_fields", "CallbackEnv fields not literal")
        out["env_fields"] = []
    else:
        out["env_fields"] = list(fields)

    for key, name in (
        ("resettable", "RESETTABLE"),
        ("reset_aliases", "_RESET_ALIASES"),
        ("integral_slots", "_INTEGRAL"),
    ):
        value = value_of(assign_in(tree.body, name), consts)
        if value is UNDERIVED or value is None:
            d.gap(f"python.callbacks.{key}", f"{name} not a literal")
            value = [] if key != "reset_aliases" else {}
        if key == "reset_aliases":
            out[key] = dict(sorted(value.items()))
        elif key == "integral_slots":
            out[key] = sorted(value)
        else:
            out[key] = list(value)

    out["reset_slots"] = mojo_comptime_int(BINDINGS, "RESET_SLOTS", d)
    out["reset_slot_order"] = mojo_reset_slot_order(d)

    factories = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            factories[node.name] = signature(node, consts)
    out["factories"] = factories
    return out


def eval_block(d: Deriver) -> dict:
    tree = parse_py(PY_EVAL, d)
    if tree is None:
        d.gap("python.eval_metric_names", "_eval.py did not parse")
        return {}
    consts = module_constants(tree)
    out = {"source": rel(PY_EVAL)}

    # Parsing rule 1: _METRICS is not literal_eval-able. Its values are
    # module-level integer constants, so the dict raises as a whole. The
    # keys are what the snapshot needs, and they read fine individually.
    metrics = assign_in(tree.body, "_METRICS")
    if metrics is None or not hasattr(metrics, "keys"):
        d.gap("python.eval_metric_names.metrics", "_METRICS not a dict")
        out["metrics"] = []
    else:
        names = []
        for key in metrics.keys:
            value = value_of(key, consts)
            if value is UNDERIVED:
                d.gap("python.eval_metric_names.metrics", "non-literal key")
                continue
            names.append(value)
        out["metrics"] = names

    aliases = value_of(assign_in(tree.body, "_ALIASES"), consts)
    if aliases is UNDERIVED or aliases is None:
        d.gap("python.eval_metric_names.aliases", "_ALIASES not a literal")
        out["aliases"] = {}
    else:
        out["aliases"] = dict(sorted(aliases.items()))
    return out


def functional_block(d: Deriver) -> dict:
    tree = parse_py(PY_BASIC, d)
    if tree is None:
        d.gap("python.functional_api", "basic.py did not parse")
        return {}
    consts = module_constants(tree)
    out = {"module": "mojotrees.basic"}
    out["all"] = dunder_all(tree, d, "python.functional_api")
    train = func_named(tree.body, "train")
    out["train"] = signature(train, consts) if train else {}
    for name in ("Dataset", "Booster"):
        cls = class_named(tree, name)
        if cls is None:
            d.gap(f"python.functional_api.{name}", "class is gone")
            continue
        methods, properties, classmethods = public_methods(cls, consts)
        out[name] = {
            "methods": methods,
            "properties": properties,
            "classmethods": classmethods,
        }
    return out


def inspection_block(d: Deriver) -> dict:
    tree = parse_py(PY_INSPECTION, d)
    if tree is None:
        d.gap("python.inspection", "inspection.py did not parse")
        return {}
    consts = module_constants(tree)
    out = {"module": "mojotrees.inspection"}
    out["all"] = dunder_all(tree, d, "python.inspection")
    for key, name in (
        ("dump_format_version", "DUMP_FORMAT_VERSION"),
        ("supported_model_format_versions", "SUPPORTED_MODEL_FORMAT_VERSIONS"),
        ("objective_names", "OBJECTIVE_NAMES"),
    ):
        value = value_of(assign_in(tree.body, name), consts)
        if value is UNDERIVED or value is None:
            d.gap(f"python.inspection.{key}", f"{name} not a literal")
            value = None
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = {str(k): v for k, v in sorted(value.items())}
        out[key] = value
    return out


# -- Mojo, C, and configuration parsing ----------------------------------


def mojo_comptime_int(path: Path, name: str, d: Deriver):
    text = read(path, d)
    m = re.search(rf"comptime\s+{re.escape(name)}\s*=\s*(-?\d+)\b", text)
    if m is None:
        d.gap(f"{rel(path)}:{name}", "no comptime integer assignment found")
        return None
    return int(m.group(1))


def mojo_comptime_str(path: Path, name: str, d: Deriver):
    text = read(path, d)
    m = re.search(rf'comptime\s+{re.escape(name)}\s*=\s*"([^"]*)"', text)
    if m is None:
        d.gap(f"{rel(path)}:{name}", "no comptime string assignment found")
        return None
    return m.group(1)


def mojo_exports_by_module(d: Deriver) -> dict:
    """The `from .mod import ...` blocks of src/mojotrees/__init__.mojo.

    Parsing rule 4: exports come in two spellings. Most are parenthesized
    and span lines; some are a single bare line, with one name or with
    several. A regex that handles only the parenthesized form silently
    drops whole modules, which is the worst possible failure for a tool
    whose job is detecting change. Both forms are handled here and the
    flattened result is checked against check_parity's parser (I11).
    """
    text = read(MOJO_INIT, d)
    out: dict = {}

    def add(module: str, names: str) -> None:
        bucket = out.setdefault(module, set())
        for raw in names.replace("\n", " ").split(","):
            name = raw.strip()
            if name:
                bucket.add(name)

    for module, names in re.findall(
        r"from\s+\.(\w+)\s+import\s*\(([^)]*)\)", text
    ):
        add(module, names)
    for line in text.splitlines():
        m = re.match(r"from\s+\.(\w+)\s+import\s+([^(].*)$", line.strip())
        if m:
            add(m.group(1), m.group(2))

    if not out:
        d.gap("mojo.exports_by_module", "no import blocks matched")
    return {module: sorted(names) for module, names in sorted(out.items())}


def mojo_objective_codes(d: Deriver) -> dict:
    """Uppercase `comptime NAME = <int>` constants in boosting.mojo, plus
    MULTICLASS from params.mojo. Restricted to uppercase names so that
    tuning constants written in lower case do not enter the snapshot as
    though they were part of the objective vocabulary."""
    out = {}
    for path in (MOJO_BOOSTING, MOJO_PARAMS):
        text = read(path, d)
        for name, value in re.findall(
            r"comptime\s+([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+)\b", text
        ):
            out.setdefault(name, int(value))
    if not out:
        d.gap("mojo.objective_codes", "no comptime integer constants matched")
    return dict(sorted(out.items()))


def mojo_reset_slot_order(d: Deriver) -> list:
    """The slot order of `_write_reset` in bindings/_mojotrees.mojo,
    translated into the Python side's spelling.

    Each line is `p.unsafe_store(<i>, <expr>)`, and the parameter is the
    last attribute of the expression: slot 1 is
    `Float64(params.tree.num_leaves)` and slot 4 is
    `params.tree.min_child_hess`. Two names differ from the Python side by
    design and RESET_NAME_MAP carries the translation; this is the wire
    format of policy section 9.3, and a reordering here produces wrong
    numbers rather than an error.
    """
    text = read(BINDINGS, d)
    start = text.find("def _write_reset")
    if start < 0:
        d.gap("python.callbacks.reset_slot_order", "_write_reset not found")
        return []
    end = text.find("\ndef ", start + 1)
    body = text[start : end if end > 0 else len(text)]
    slots: dict = {}
    for index, expr in re.findall(
        r"p\.unsafe_store\(\s*(\d+)\s*,\s*(.+?)\)\s*$", body, re.MULTILINE
    ):
        attrs = re.findall(r"\.(\w+)", expr)
        if not attrs:
            continue
        name = attrs[-1]
        slots[int(index)] = RESET_NAME_MAP.get(name, name)
    if not slots:
        d.gap("python.callbacks.reset_slot_order", "no unsafe_store matched")
        return []
    return [slots[i] for i in sorted(slots)]


def model_format_block(d: Deriver) -> dict:
    """Five constants in three files, recorded separately rather than
    reconciled. A schema with one "model format version" field would have
    hidden the four-way disagreement findings F1 and F2 describe."""
    text = read(MOJO_SERIALIZE, d)
    readable = sorted(
        int(v) for v in re.findall(r'token\s*==\s*"v(\d+)"', text)
    )
    if not readable:
        d.gap("versions.model_format_readable", "no version chain matched")
    return {
        "writer": mojo_comptime_int(
            MOJO_SERIALIZE, "CURRENT_FORMAT_VERSION", d
        ),
        "writer_token": mojo_comptime_str(MOJO_SERIALIZE, "_VERSION", d),
        "readable": readable,
        "dump_reports": mojo_comptime_int(
            MOJO_DUMP, "MODEL_FORMAT_VERSION", d
        ),
    }


def c_abi_block(d: Deriver) -> dict:
    """The header, normalized.

    Declarations are collapsed to one space so that rewrapping a long
    signature is not a diff, but parameter names are kept: a name is
    documentation a caller reads, and losing one silently is exactly what
    this file exists to show. Parsing rule 5: negative defines are
    parenthesized, `(-3)` and not `-3`.
    """
    text = read(CAPI_HEADER, d)
    stripped = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    stripped = re.sub(r"//[^\n]*", " ", stripped)

    defines = {}
    for name, value in re.findall(
        r"#define\s+(MOJOTREES_[A-Z0-9_]+)\s+\(?\s*(-?\d+)\s*\)?\s*$",
        stripped,
        re.MULTILINE,
    ):
        defines[name] = int(value)
    if "MOJOTREES_ABI_VERSION" not in defines:
        d.gap("c_abi.abi_version", "MOJOTREES_ABI_VERSION not found")

    opaque = sorted(
        set(re.findall(r"typedef\s+struct\s+\w+\s+(\w+)\s*;", stripped))
    )

    functions = []
    for match in re.finditer(
        r"(?<![\w*])((?:const\s+)?\w+[\w\s*]*?)\b(mojotrees_\w+)\s*\("
        r"([^;{]*?)\)\s*;",
        stripped,
        re.DOTALL,
    ):
        ret = " ".join(match.group(1).split())
        name = match.group(2)
        args = " ".join(match.group(3).split())
        joiner = "" if ret.endswith("*") else " "
        functions.append(f"{ret}{joiner}{name}({args})")
    if not functions:
        d.gap("c_abi.functions", "no declarations matched")

    return {
        "header": rel(CAPI_HEADER),
        "abi_version": defines.get("MOJOTREES_ABI_VERSION"),
        "defines": dict(sorted(defines.items())),
        "opaque_types": opaque,
        "functions": sorted(set(functions)),
    }


def parameter_string_block(d: Deriver) -> dict:
    """SUPPORTED_KEYS, which is one implicitly concatenated string literal
    split across lines. Parsing rule 6: the separating commas sit at the
    ends of the fragments, so the fragments are joined before the split or
    the key straddling a boundary is lost."""
    text = read(MOJO_PARAMS, d)
    m = re.search(r"comptime\s+SUPPORTED_KEYS\s*=\s*String\((.*?)\)", text,
                  re.DOTALL)
    if m is None:
        d.gap("parameter_string.supported_keys", "SUPPORTED_KEYS not found")
        return {"supported_keys": []}
    joined = "".join(re.findall(r'"([^"]*)"', m.group(1)))
    keys = [k.strip() for k in joined.split(",") if k.strip()]
    return {
        "parser": "src/mojotrees/params.mojo parse_params",
        "used_by": ["capi", "cli"],
        "supported_keys": keys,
    }


def environment_block(d: Deriver) -> dict:
    """Declared against observed.

    `observed` is the double-quoted literal scan and not a scan of getenv
    call sites, and the difference is load-bearing: four of the seven
    documented variables are read through `_env_int(name, default)`
    wrappers, so a call scan would mark them stale and fail invariant I8
    on a correct tree. Parsing rule 9: discard the bare prefix literal,
    which is a filter and not a variable.
    """
    declared = sorted(
        set(re.findall(r"`(MOJOTREES_[A-Z0-9_]+)`", read(POLICY, d)))
    )
    observed: set = set()
    direct: set = set()
    for directory in ENV_SCAN_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in ENV_SCAN_SUFFIXES:
                continue
            text = path.read_text(errors="replace")
            observed.update(re.findall(r'"(MOJOTREES_[A-Z0-9_]+)"', text))
            direct.update(
                re.findall(
                    r'(?:getenv|environ\.get)\(\s*"(MOJOTREES_[A-Z0-9_]+)"',
                    text,
                )
            )
    observed.discard("MOJOTREES_")
    direct.discard("MOJOTREES_")
    if not declared:
        d.gap("environment.declared", "no backtick names found in the policy")
    return {
        "declared": declared,
        "observed": sorted(observed),
        "read_directly": sorted(direct),
        "undeclared": sorted(observed - set(declared)),
        "stale": sorted(set(declared) - observed),
    }


def versions_block(d: Deriver, inspection: dict) -> dict:
    """`inspection` is the already-parsed python.inspection block. It is
    passed in rather than re-derived so that inspection.py is parsed once
    and a gap in it is reported once."""
    pixi = read(PIXI, d)
    pyproject = read(PYPROJECT, d)
    init = read(PY_API, d)

    def first(pattern: str, text: str, label: str):
        m = re.search(pattern, text, re.MULTILINE)
        if m is None:
            d.gap(f"versions.{label}", "pattern did not match")
            return None
        return m.group(1)

    locations = {
        "pixi.toml": first(
            r'^\s*version\s*=\s*"([^"]+)"', pixi, "library_locations.pixi"
        ),
        "python/pyproject.toml": first(
            r'^\s*version\s*=\s*"([^"]+)"',
            pyproject,
            "library_locations.pyproject",
        ),
        "python/mojotrees/__init__.py": first(
            r'^__version__\s*=\s*"([^"]+)"',
            init,
            "library_locations.dunder",
        ),
    }
    values = {v for v in locations.values() if v is not None}
    library = values.pop() if len(values) == 1 else None
    if library is None and locations:
        d.gap("versions.library", "the three locations do not agree")

    fmt = model_format_block(d)
    return {
        "library": library,
        "library_locations": locations,
        "c_abi": None,  # filled from the c_abi block, which owns the parse
        "model_format_writer": fmt["writer"],
        "model_format_writer_token": fmt["writer_token"],
        "model_format_readable": fmt["readable"],
        "model_format_dump_reports": fmt["dump_reports"],
        "model_format_python_reads": inspection.get(
            "supported_model_format_versions"
        ),
        "dump_format": inspection.get("dump_format_version"),
        "snapshot_schema": SCHEMA_VERSION,
        "requires_python": first(
            r'^requires-python\s*=\s*"([^"]+)"',
            pyproject,
            "requires_python",
        ),
        "mojo_toolchain": first(
            r'^mojo\s*=\s*"([^"]+)"', pixi, "mojo_toolchain"
        ),
        "max_toolchain": first(
            r'^max\s*=\s*"([^"]+)"', pixi, "max_toolchain"
        ),
    }


def platforms_block(d: Deriver) -> dict:
    pixi = read(PIXI, d)
    m = re.search(r"^platforms\s*=\s*\[([^\]]*)\]", pixi, re.MULTILINE)
    declared = (
        sorted(re.findall(r'"([^"]+)"', m.group(1))) if m else []
    )
    if not declared:
        d.gap("platforms.declared", "pixi.toml platforms did not match")
    runners: set = set()
    for block in re.findall(r"runner:\s*\[([^\]]*)\]", read(CI, d)):
        runners.update(r.strip() for r in block.split(",") if r.strip())
    if not runners:
        d.gap("platforms.ci_runners", "no runner matrix matched")
    return {"declared": declared, "ci_runners": sorted(runners)}


def deprecations_block(d: Deriver) -> dict:
    """A digest of compatibility/deprecations.toml, so that a movement in
    the register is visible in the snapshot diff. The register is the
    authority; this is a mirror."""
    try:
        data = tomllib.loads(DEPRECATIONS.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        d.problem(f"cannot read {rel(DEPRECATIONS)}: {exc}")
        return {"entries": [], "candidates": []}
    entries = []
    for row in data.get("deprecation", []):
        entries.append(
            {
                "id": row.get("id"),
                "surface": row.get("surface"),
                "name": row.get("name"),
                "state": row.get("state"),
                "since": row.get("since"),
                "remove_in": row.get("remove_in"),
                "removed_in": row.get("removed_in"),
            }
        )
    candidates = [
        {"id": row.get("id"), "name": row.get("name"),
         "gate_item": row.get("gate_item")}
        for row in data.get("candidate", [])
    ]
    return {
        "entries": sorted(entries, key=lambda r: str(r["id"])),
        "candidates": sorted(candidates, key=lambda r: str(r["id"])),
    }


# -- Invariants ----------------------------------------------------------


def check_invariants(snap: dict, d: Deriver) -> None:
    """Facts about the tree, not about the file. A violation is reported
    and exits non-zero even when the snapshot itself matches, because the
    tree is what is wrong."""
    v = snap.get("versions", {})
    cb = snap.get("python", {}).get("callbacks", {})

    locations = v.get("library_locations", {})
    distinct = {value for value in locations.values() if value is not None}
    if len(distinct) > 1:
        d.problem(
            "I1: the three library version locations disagree: "
            + ", ".join(f"{k}={value!r}" for k, value in locations.items())
        )

    writer = v.get("model_format_writer")
    token = v.get("model_format_writer_token")
    if writer is not None and token is not None and token != f"v{writer}":
        d.problem(
            f"I2: serialize.mojo writes {token!r} but "
            f"CURRENT_FORMAT_VERSION is {writer}"
        )
    readable = v.get("model_format_readable") or []
    if writer is not None and readable and writer not in readable:
        d.problem(
            f"I3: the writer emits v{writer} and the reader accepts "
            f"{readable}; the build cannot read what it writes"
        )
    reports = v.get("model_format_dump_reports")
    if writer is not None and reports is not None and reports != writer:
        d.problem(
            f"I4: model_dump.mojo reports model_format_version {reports} "
            f"and serialize.mojo writes v{writer}; a dump consumer branches "
            f"on that number to know which optional facts a model can carry"
        )
    py_reads = v.get("model_format_python_reads") or []
    if writer is not None and py_reads and max(py_reads) < writer:
        d.problem(
            f"I5: inspection.py reads {py_reads} and the writer emits "
            f"v{writer}; the pure-Python parser rejects a model this build "
            f"just saved"
        )

    resettable = cb.get("resettable") or []
    slots = cb.get("reset_slots")
    if slots is not None and len(resettable) != slots:
        d.problem(
            f"I6: len(RESETTABLE) is {len(resettable)} and RESET_SLOTS is "
            f"{slots}; the two sides of the reset wire format disagree"
        )
    order = cb.get("reset_slot_order") or []
    if resettable and order and list(resettable) != list(order):
        for i, (py, mojo) in enumerate(zip(resettable, order)):
            if py != mojo:
                d.problem(
                    f"I7: reset slot {i} is {py!r} on the Python side and "
                    f"{mojo!r} on the Mojo side; a reordering here produces "
                    f"wrong numbers rather than an error"
                )

    stale = snap.get("environment", {}).get("stale") or []
    if stale:
        d.problem(
            "I8: documented environment variables that nothing reads: "
            + ", ".join(stale)
        )

    check_register(snap, d)
    check_mojo_export_agreement(snap, d)


def _version_tuple(text):
    if not isinstance(text, str):
        return None
    parts = text.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def check_register(snap: dict, d: Deriver) -> None:
    """I9 and I10. The register says what was promised; the snapshot says
    what is there, and they cannot disagree about existence."""
    entries = snap.get("deprecations", {}).get("entries", [])
    if not entries:
        return
    present = present_names(snap)
    for row in entries:
        name, state = row.get("name"), row.get("state")
        if state == "removed" and name in present:
            d.problem(
                f"I9: {row['id']} is recorded as removed and {name} is "
                f"still in the snapshot"
            )
        if state in ("soft", "deprecated") and name not in present:
            d.problem(
                f"I9: {row['id']} is recorded as {state} and {name} is not "
                f"in the snapshot; a deprecated name has to still work"
            )
        since = _version_tuple(row.get("since"))
        remove_in = _version_tuple(row.get("remove_in"))
        if since and remove_in:
            surface = row.get("surface")
            if surface == "c_abi":
                ok = remove_in[0] > since[0]
                rule = "a C ABI declaration goes only in a major release"
            else:
                ok = remove_in[0] > since[0] or remove_in[1] >= since[1] + 2
                rule = "two minor releases, counted from the announcement"
            if not ok:
                d.problem(
                    f"I10: {row['id']} announced in {row['since']} and "
                    f"promised for removal in {row['remove_in']}; {rule}"
                )


def present_names(snap: dict) -> set:
    """Every public name the snapshot records, in the spellings the
    register uses. Deliberately generous: a false positive here means a
    register row is accepted, a false negative means a correct row is
    reported as broken, and the second is the worse error."""
    out: set = set()
    py = snap.get("python", {})
    out.update(py.get("all", []))
    for cls, entry in py.get("estimators", {}).items():
        out.add(cls)
        for method in entry.get("methods", {}):
            out.add(f"{cls}.{method}")
        for prop in entry.get("properties", []):
            out.add(f"{cls}.{prop}")
    for attr in py.get("fitted_attributes", []):
        out.add(attr)
        out.add(f"_Base.{attr}")
    out.update(py.get("parameter_aliases", {}))
    out.update(py.get("callbacks", {}).get("all", []))
    for module, names in snap.get("mojo", {}).get(
        "exports_by_module", {}
    ).items():
        out.update(names)
        out.update(f"{module}.{n}" for n in names)
    for decl in snap.get("c_abi", {}).get("functions", []):
        m = re.search(r"\b(mojotrees_\w+)\s*\(", decl)
        if m:
            out.add(m.group(1))
    out.update(snap.get("c_abi", {}).get("defines", {}))
    out.update(snap.get("parameter_string", {}).get("supported_keys", []))
    out.update(snap.get("environment", {}).get("observed", []))
    return out


def check_mojo_export_agreement(snap: dict, d: Deriver) -> None:
    """I11. Two parsers over src/mojotrees/__init__.mojo must agree, or one
    of them is dropping a module. Degrades to a recorded gap rather than a
    failure if check_parity cannot be imported."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import check_parity  # noqa: E402  (deliberately late and guarded)
    except Exception as exc:  # pragma: no cover - environment dependent
        d.gap("invariants.I11", f"check_parity could not be imported: {exc}")
        return
    try:
        theirs = set(check_parity.mojo_export_names())
    except Exception as exc:
        d.gap("invariants.I11", f"check_parity.mojo_export_names(): {exc}")
        return
    ours: set = set()
    for names in snap.get("mojo", {}).get("exports_by_module", {}).values():
        ours.update(names)
    if ours != theirs:
        only_ours = sorted(ours - theirs)
        only_theirs = sorted(theirs - ours)
        d.problem(
            "I11: the two parsers of src/mojotrees/__init__.mojo disagree; "
            f"only here: {only_ours or 'none'}; "
            f"only in check_parity: {only_theirs or 'none'}"
        )


# -- Diff and classification ---------------------------------------------

# Policy section 11.2, as rules over dotted paths. The first pattern that
# matches a path wins, so order is significance: the specific rules come
# before the general ones.
CLASSIFY = (
    (r"^python\.callbacks\.resettable", "breaking", "policy 9.3, and silently wrong"),
    (r"^python\.callbacks\.reset_slot", "breaking", "policy 9.3, and silently wrong"),
    (r"^python\.callbacks\.env_fields", "breaking", "policy 9.1, positional contract"),
    (r"^versions\.c_abi", "breaking", "an ABI version bump records a break"),
    (r"^c_abi\.", "breaking", "policy 6.3"),
    (r"^versions\.model_format", "review", "policy 7; extend the read-back matrix"),
    (r"^versions\.dump_format", "breaking", "a dump key was removed, retyped, or redefined"),
    (r"^versions\.library", "review", "a release happened"),
    (r"^python\.fitted_attributes", "breaking", "policy 5.2"),
    (r"^python\.shared_estimator_parameters\.", "breaking", "policy 4.3, a default changed"),
    (r"^python\.estimators\.[^.]+\.own_parameters\.", "breaking", "policy 4.3"),
    (r"^python\.estimators\.[^.]+\.methods\.", "breaking", "an argument moved, was renamed, or was retyped"),
    (r"^python\.parameter_aliases\.", "breaking", "policy 4.2"),
    (r"^parameter_string\.supported_keys", "review", "policy 4.4; accepted is additive, the reverse is breaking"),
    (r"^platforms\.", "breaking", "policy 10.4, a tier or a target moved"),
    (r"^environment\.stale", "breaking", "a documented variable stopped being read"),
    (r"^environment\.", "review", "an undeclared variable appeared or went"),
    (r"^numerical_contracts", "breaking", "policy 8.3"),
    (r"^meta\.", "ignore", "bookkeeping"),
)


def classify(path: str, kind: str) -> tuple:
    """(severity, why). `kind` is "added", "removed", or "changed".

    Additive is the default for an appearance and breaking is the default
    for a disappearance, which is policy section 11.2's first two rows.
    The rules table overrides both, because some paths are breaking even
    when something appears: a `CallbackEnv` field is additive only at the
    end, and a new `RESETTABLE` entry is additive only if the slot count
    moved with it, which invariant I6 is what actually checks.
    """
    for pattern, severity, why in CLASSIFY:
        if re.match(pattern, path):
            if severity == "ignore":
                return "ignore", why
            if kind == "added" and severity == "review":
                return "additive", why
            return severity, why
    if kind == "added":
        return "additive", "a name or a key appeared"
    if kind == "removed":
        return "breaking", "a name disappeared; policy 11.2"
    return "review", "policy 11.2"


def flatten(obj, prefix="") -> dict:
    """Dotted path to leaf value. Lists are compared whole, because for
    every list in this snapshot the order is either the contract or a
    meaningful rewrite."""
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, path))
    else:
        out[prefix] = obj
    return out


def diff(old: dict, new: dict) -> list:
    a, b = flatten(old), flatten(new)
    rows = []
    for path in sorted(set(a) | set(b)):
        if path not in b:
            rows.append((path, "removed", a[path], None))
        elif path not in a:
            rows.append((path, "added", None, b[path]))
        elif a[path] != b[path]:
            rows.append((path, "changed", a[path], b[path]))
    return rows


# -- Assembly and entry point --------------------------------------------


def normalize(obj, d: Deriver, prefix=""):
    """Replace every UNDERIVED sentinel with None and record its path.

    `signature()` and the other readers return the sentinel where a value
    is a name they could not resolve, which is right: recording the
    spelling of an unresolved constant would report drift forever, and
    recording a guess would hide it. But the sentinel is not JSON, so it
    is converted here, once, on the way out. A leak would crash the dump,
    which is a loud failure and therefore the acceptable one, but this
    turns it into a named gap instead.
    """
    if obj is UNDERIVED:
        d.gap(prefix or "<root>", "value is not a literal and did not resolve")
        return None
    if isinstance(obj, dict):
        return {
            key: normalize(
                value, d, f"{prefix}.{key}" if prefix else str(key)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [
            normalize(value, d, f"{prefix}[{i}]")
            for i, value in enumerate(obj)
        ]
    return obj


def build(commit: str | None) -> tuple:
    d = Deriver()
    snap: dict = {}
    snap["python"] = python_block(d)
    snap["mojo"] = {
        "package": "mojotrees",
        "export_source": rel(MOJO_INIT),
        "abi_promise": (
            "none; Mojo has no stable ABI, so the guarantee is source "
            "compatibility only"
        ),
        "exports_by_module": mojo_exports_by_module(d),
        "objective_codes": mojo_objective_codes(d),
    }
    snap["mojo"]["export_count"] = sum(
        len(v) for v in snap["mojo"]["exports_by_module"].values()
    )
    snap["c_abi"] = c_abi_block(d)
    snap["parameter_string"] = parameter_string_block(d)
    snap["environment"] = environment_block(d)
    snap["platforms"] = platforms_block(d)
    snap["deprecations"] = deprecations_block(d)
    snap["versions"] = versions_block(
        d, snap["python"].get("inspection", {})
    )
    snap["versions"]["c_abi"] = snap["c_abi"]["abi_version"]

    snap = normalize(snap, d)
    snap["meta"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "generated",
        "generated_by": "tools/api_snapshot.py",
        "tool_version": TOOL_VERSION,
        "library_version": snap["versions"]["library"],
        "source_commit": commit,
        "underived": sorted(set(d.underived)),
        "carried": [],
    }
    check_invariants(snap, d)
    return snap, d


def carry_forward(snap: dict, previous: dict, d: Deriver) -> None:
    """platforms.tiers and numerical_contracts cannot be parsed out of
    anything: a tier is a claim about what evidence exists and a numerical
    contract is a claim about behavior. Carry them, say so, and never
    invent them."""
    for path, container, key in (
        ("platforms.tiers", snap["platforms"], "tiers"),
        ("numerical_contracts", snap, "numerical_contracts"),
    ):
        source = previous
        for part in path.split("."):
            source = source.get(part) if isinstance(source, dict) else None
        if source is None:
            d.note(
                f"{path} has no previous value to carry; seed it by hand "
                f"from docs/COMPATIBILITY_POLICY.md, once"
            )
            continue
        container[key] = source
        snap["meta"]["carried"].append(path)
    snap["meta"]["carried"].sort()


def load_previous(path: Path, d: Deriver):
    if path.exists():
        try:
            return json.loads(path.read_text()), path
        except json.JSONDecodeError as exc:
            d.problem(f"{rel(path)} is not valid JSON: {exc}")
            return None, path
    if path == SNAPSHOT and LEGACY_SNAPSHOT.exists():
        d.note(
            f"{rel(SNAPSHOT)} does not exist; reading "
            f"{rel(LEGACY_SNAPSHOT)} as the superseded schema-1 draft. Its "
            f"shape differs, so every path will report as a difference. "
            f"See compatibility/SNAPSHOT_SCHEMA.md section 8"
        )
        try:
            return json.loads(LEGACY_SNAPSHOT.read_text()), LEGACY_SNAPSHOT
        except json.JSONDecodeError as exc:
            d.problem(f"{rel(LEGACY_SNAPSHOT)} is not valid JSON: {exc}")
    return None, path


def dump(snap: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(snap, fp, indent=2, sort_keys=True)
        fp.write("\n")


def report(d: Deriver) -> None:
    for note in d.notes:
        print(f"  note: {note}")
    for problem in d.problems:
        print(f"  PROBLEM: {problem}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check mojotrees's public API snapshot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument(
        "--commit",
        default=None,
        help="revision to record. The tool runs no git command, so that it "
        "behaves identically inside a release tarball with no .git",
    )
    args = parser.parse_args(argv)

    snap, d = build(args.commit)
    previous, previous_path = load_previous(args.snapshot, d)

    if args.write:
        if previous is not None:
            carry_forward(snap, previous, d)
        dump(snap, args.snapshot)
        print(f"wrote {rel(args.snapshot)}")
        if snap["meta"]["carried"]:
            print(
                "  carried, not derived: "
                + ", ".join(snap["meta"]["carried"])
            )
        if snap["meta"]["underived"]:
            print(
                "  underived: " + ", ".join(snap["meta"]["underived"])
            )
        report(d)
        return 1 if d.problems else 0

    if previous is None:
        print(f"{rel(args.snapshot)} does not exist; run --write first")
        report(d)
        return 1

    carry_forward(snap, previous, d)
    rows = diff(previous, snap)
    # meta moves on every run by design; it is not a surface change.
    rows = [r for r in rows if not r[0].startswith("meta.")]

    if rows:
        print(f"{len(rows)} difference(s) against {rel(previous_path)}:")
        for path, kind, old, new in rows:
            severity, why = classify(path, kind)
            if severity == "ignore":
                continue
            print(f"  [{severity}] {path} ({kind}): {old!r} -> {new!r}")
            print(f"      {why}")
    report(d)

    if rows or d.problems:
        print(
            "\nA difference is not automatically a failure. Classify each "
            "one under docs/COMPATIBILITY_POLICY.md section 11.2, and "
            "regenerate with --write in the same commit as the change that "
            "caused it. A breaking row needs a break note under section 3.4 "
            "and, if the surface is going away, a row in "
            "compatibility/deprecations.toml."
        )
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
