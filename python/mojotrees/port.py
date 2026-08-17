"""What happened to a LightGBM, XGBoost, or CatBoost parameter dict.

The estimators already do the hard half. Every vendor spelling is accepted,
two spellings of one parameter that disagree raise instead of one silently
winning, unimplemented objectives raise with the trainer's own sentence, and
value strings are case-insensitive. What was missing is the report, so a
user porting a script found out what we did with their dict by trial and
error. `port()` answers it in one call.

    from mojotrees import port

    report = port({"num_leaves": 63, "feature_pre_filter": True},
                  source="lightgbm")
    print(report)                       # the readable table
    model = MojoTreesRegressor(**report.params)

THREE NAMES, NOT TWO
--------------------
A parameter here has a CANONICAL spelling, which is the scikit-learn one
and is what every error message, every docstring and this report name; a
WIRE spelling, which is LightGBM's and is what the native layer is sent and
what a saved model holds; and one or more vendor ALIASES. Often all three
coincide. For eleven parameters they do not, and `subsample` is the example
to keep in mind: you type `subsample`, the wire carries `bagging_fraction`,
and a report that answered "we call it bagging_fraction" would be naming
the wire when you asked what to type. So `we call it` is the canonical
spelling and `on the wire` is beside it, both, rather than one of them
under a label that fits the other.

`docs/PARAMETER_NAMING.md` is the record of which spelling is canonical. It
is read, not transcribed, and a parameter it does not cover keeps its wire
spelling here rather than acquiring a canonical this file invented.

Every key the caller passed lands in exactly one of four categories.

- `honored`, ours under the same name, used as given. Kept as given even
  where the canonical spelling differs, because "used as given" is the
  promise the category makes.
- `aliased_to`, a vendor spelling of one of our canonical names. The entry
  carries the canonical name, the wire name, and the value.
- `defaulted_differently`, we accept it, the caller did NOT name it, and our
  default is not that vendor's stock default. This is the category that
  prevents a silent surprise, because nothing in the caller's dict points at
  it. It is computed against each vendor's recorded defaults, never against
  anybody's memory of them.
- `refused`, we do not implement it. The entry carries the REASON, taken
  from the place in this repository that already states it.

A fifth list, `unsourced`, is not a category. It is every fact this report
wanted and could not read, named one by one, so an answer this function
could not source shows up as an absence rather than as a guess.

WHERE THE FACTS COME FROM
-------------------------
Nothing here is transcribed. Every table is read, at call time, from the
file that owns it.

- accepted parameter names and our defaults, from `_Base.__init__` and the
  three estimator constructors in `sklearn.py`, parsed as source. That file
  ships in the wheel beside this one, so this half always resolves.
- the alias table, from the literal arguments of the `self._resolve_alias`
  call sites in `_Base`. That is the same derivation
  `tools/api_snapshot.py:alias_pairs` performs to produce the
  `parameter_aliases` block of `compatibility/api_snapshot.json`, over the
  same call sites, so the two cannot disagree about a pair. It is read here
  rather than read out of the snapshot because the snapshot is not in the
  wheel and `sklearn.py` is.
- the canonical spelling of each parameter, from the OURS column of
  `docs/PARAMETER_NAMING.md`. That document is the source of truth for
  which spelling is canonical, and `tools/api_snapshot.py` reads it the
  same three ways for the same purpose. It lives in the checkout and not in
  the wheel, so an installed `port()` reports the wire spelling and says so
  in `unsourced` rather than guessing which of a parameter's spellings the
  document would have called canonical.
- LightGBM's stock defaults and our DECLARED departures from them, from
  `LIGHTGBM_STOCK` and `STOCK_DIVERGENCES` in `tools/check_parity.py`. That
  table is a gate. A default that drifts off it fails CI, so reusing it here
  is what keeps `defaulted_differently` true tomorrow.
- XGBoost's and CatBoost's resolved defaults, from
  `XGBOOST_RESOLVED_DEFAULTS` and `CATBOOST_LEFT_AT_STOCK` in
  `bench/real_data/scenarios.py`, each of which records the library call it
  was read back from rather than a documentation page.
- refusal reasons, from the row for that parameter in section 7 of
  `docs/LIGHTGBM_PARITY.md`, from `_DATASET_PARAMS` in `basic.py` for the
  keys that belong to a `Dataset`, and from the extension's own
  `extra_option_supported` message by way of `mojotrees.preflight` when the
  extension is importable.

The last four of those live in the checkout and not in the wheel. Installed
from a wheel, this function still classifies every key, still resolves every
alias and still reports our own defaults. What it cannot do is compare them
against a vendor's, and it says so in `unsourced` instead of inventing the
comparison. `report.fact_sources` names which files were read.

WHY A DATACLASS AND NOT A DICT OF LISTS
---------------------------------------
The four categories do not carry the same fields. An alias entry needs two
names, a divergence needs two values plus a reason plus the file the reason
came from, and a refusal needs a status word. A plain dict of lists would
force either the union of all of those on every row, most of them empty, or
four differently shaped dicts whose keys are documented nowhere. So the rows
are small frozen dataclasses, the container is one `PortReport`, and
`to_dict()` hands back the plain nested dict for a program that wants JSON.
`render()` builds the table, `__str__` calls `render()`, and the docs page
`docs/PORTING.md` is generated by `tools/port_report.py` through that same
`render()`, so the page and the console cannot drift.

`port()` returns one object rather than a `(params, report)` pair, and
`report.params` is the canonical dict. One object, because a pair invites
`params, _ = port(...)`, which throws away the half that exists to be read.

`port()` never raises for a configuration the estimator would reject. It is
a report. Two spellings of one parameter that disagree are recorded in
`report.notes` and left for `fit` to raise on, because the user asking what
happened to their config is exactly the user whose config is already wrong.

IMPORTS NOTHING FROM THIS PACKAGE
---------------------------------
Only the standard library, and the one optional `preflight` import is inside
a function and inside a `try`. That is deliberate. It lets
`tools/port_report.py` load this file by path and regenerate the docs page
on a checkout that has never been built, which is the same rule every other
gate under `tools/` follows.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "SOURCES",
    "AliasedTo",
    "DefaultedDifferently",
    "Honored",
    "PortReport",
    "Refused",
    "Unsourced",
    "port",
]

#: The vendors `port()` can be asked about. Each one has a recorded table of
#: that vendor's own resolved defaults in this repository; a vendor with no
#: such table could not be answered without guessing, so it is not offered.
SOURCES = ("lightgbm", "xgboost", "catboost")


class _Sentinel:
    """A named singleton, so a missing fact prints as a word."""

    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    def __bool__(self):
        return False


#: A fact this function could not read. Never equal to any real value, and
#: falsey, so `value is UNKNOWN` and a plain truth test both behave.
UNKNOWN = _Sentinel("unknown")

#: A constructor argument with no default at all. `_Base.__init__` has none
#: today; the sentinel exists so that adding one is reported rather than
#: read as `None`.
REQUIRED = _Sentinel("required")

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Honored:
    """A key that is ours under the same name, used as given."""

    name: str
    value: object
    owner: str = "shared"
    note: str = ""


@dataclass(frozen=True)
class AliasedTo:
    """A vendor spelling of one of our canonical names.

    `canonical` is the spelling to type, from `docs/PARAMETER_NAMING.md`.
    `wire` is the spelling the native layer is sent and the model file
    holds, which is LightGBM's. They differ for eleven parameters, and
    `canonical` falls back to `wire` where the naming document is silent or
    cannot be read.
    """

    name: str
    canonical: str
    value: object
    note: str = ""
    wire: str = ""


@dataclass(frozen=True)
class DefaultedDifferently:
    """A parameter the caller did not name, whose default is not theirs."""

    name: str
    vendor_name: str
    vendor_default: object
    our_default: object
    declared: bool
    reason: str
    citation: str


@dataclass(frozen=True)
class Refused:
    """A key we do not implement, with the reason we already state."""

    name: str
    value: object
    status: str
    reason: str
    citation: str


@dataclass(frozen=True)
class Unsourced:
    """A fact this report wanted and could not read."""

    what: str
    wanted_from: str


# ---------------------------------------------------------------------------
# Reading source files
# ---------------------------------------------------------------------------

#: `{path: (tree, module constants)}`. Every table this module reads is a
#: literal in a file that does not change while the process runs, and two of
#: those files are several thousand lines, so each is parsed once.
_PARSED = {}


def _literal(node, consts):
    """An AST node as a Python value, resolving bare names through
    `consts`, or UNKNOWN when this reader cannot decide it."""
    if node is None:
        return UNKNOWN
    if isinstance(node, ast.Name):
        return consts.get(node.id, UNKNOWN)
    try:
        return ast.literal_eval(node)
    except Exception:
        return UNKNOWN


def _parse(path):
    """`(tree, module constants)` for a Python file, or `(None, {})`."""
    key = str(path)
    if key in _PARSED:
        return _PARSED[key]
    try:
        text = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError):
        _PARSED[key] = (None, {})
        return _PARSED[key]
    consts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _literal(node.value, consts)
        if value is not UNKNOWN:
            consts[target.id] = value
    _PARSED[key] = (tree, consts)
    return _PARSED[key]


def _class_named(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _func_named(body, name):
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _signature_defaults(fn, consts):
    """`{argument: default}` for a function definition, `self` dropped."""
    out = {}
    if fn is None:
        return out
    args = fn.args
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults = list(args.defaults)
    first_defaulted = len(positional) - len(defaults)
    for index, arg in enumerate(positional):
        if arg.arg in ("self", "cls"):
            continue
        if index < first_defaulted:
            out[arg.arg] = REQUIRED
        else:
            out[arg.arg] = _literal(defaults[index - first_defaulted], consts)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            out[arg.arg] = REQUIRED
        else:
            out[arg.arg] = _literal(default, consts)
    return out


def _alias_pairs(scope, consts):
    """`{alias: {wire, fallback_default, sites}}` from the
    `self._resolve_alias(primary, alias, default)` call sites.
    `_estimator_facts` adds `canonical` to each entry afterwards.

    The same three literal leading arguments `tools/api_snapshot.py` reads,
    walked over the same class, so the pair reported here and the pair in
    `compatibility/api_snapshot.json` are one fact and not two.

    The first argument is `primary` at the call site and is the WIRE name,
    not the canonical one. `_resolve_alias`'s own docstring says so:
    `num_leaves` is the primary and `max_leaves` is the canonical name
    resolved onto it. The canonical spelling is added by
    `_canonical_map`, from `docs/PARAMETER_NAMING.md`, and is a separate
    key precisely because these two are separate facts.
    """
    pairs = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "_resolve_alias" or len(node.args) < 3:
            continue
        wire, alias = node.args[0], node.args[1]
        if not isinstance(wire, ast.Constant):
            continue
        if not isinstance(alias, ast.Constant):
            continue
        default = _literal(node.args[2], consts)
        entry = pairs.setdefault(
            alias.value,
            {
                "wire": wire.value,
                "fallback_default": default,
                "sites": 0,
            },
        )
        entry["sites"] += 1
        # A chained resolution passes the running value as its third
        # argument at every link but the first, so a later site can be
        # unreadable where the first was readable. Keep the readable one
        # rather than letting the chain erase it.
        if entry["fallback_default"] is UNKNOWN and default is not UNKNOWN:
            entry["fallback_default"] = default
    return pairs


def _repo_root():
    """The checkout this file sits in, or None when installed from a wheel.

    Identified by the files whose tables this report reads, not by a
    directory name, so a partial tree counts as no tree.
    """
    for parent in _HERE.parents:
        parity = parent / "tools" / "check_parity.py"
        contract = parent / "docs" / "LIGHTGBM_PARITY.md"
        if parity.is_file() and contract.is_file():
            return parent
    return None


def _display_path(path):
    """A path as this report should print it.

    Relative to the checkout when there is one, absolute otherwise. The
    generated `docs/PORTING.md` embeds these strings, so an absolute path
    here would make the page differ between two machines and make
    `tools/port_report.py --check` fail on a clean tree.
    """
    path = Path(path)
    root = _repo_root()
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


#: Our parameters whose constructor default is `None` because "the user did
#: not name this" has to survive to fit time, and the module constant an
#: unset one resolves to. Names only. No value is repeated here, so this
#: cannot carry a stale number.
#:
#: The same pairing `STOCK_PYTHON_CONSTANTS` in `tools/check_parity.py`
#: carries, which is the gate that asserts these three constants against
#: LightGBM's stock table.
_CONSTANT_DEFAULTS = {
    "lambda_l1": "_LAMBDA_L1",
    "lambda_l2": "_LAMBDA_L2",
    "learning_rate": "_LEARNING_RATE",
}

_ESTIMATORS = (
    "MojoTreesRegressor",
    "MojoTreesClassifier",
    "MojoTreesRanker",
)

_ESTIMATOR_FACTS = {}


def _estimator_facts():
    """Accepted parameters, our defaults, the alias table, and the module
    constants, all read out of `sklearn.py` next to this file."""
    if _ESTIMATOR_FACTS:
        return _ESTIMATOR_FACTS
    path = _HERE / "sklearn.py"
    tree, consts = _parse(path)
    facts = {
        "path": path,
        "accepted": {},
        "aliases": {},
        "canonicals": {},
        "constants": consts,
        "gaps": [],
    }
    if tree is None:
        facts["gaps"].append(
            Unsourced(
                "the accepted parameter names and our own defaults",
                _display_path(path),
            )
        )
        _ESTIMATOR_FACTS.update(facts)
        return _ESTIMATOR_FACTS
    base = _class_named(tree, "_Base")
    if base is None:
        facts["gaps"].append(
            Unsourced(
                "the shared estimator parameters",
                "%s, class _Base" % _display_path(path),
            )
        )
    else:
        init = _func_named(base.body, "__init__")
        shared = _signature_defaults(init, consts)
        for name, default in shared.items():
            facts["accepted"][name] = {"default": default, "owner": "shared"}
        facts["aliases"] = _alias_pairs(base, consts)
        if not facts["aliases"]:
            facts["gaps"].append(
                Unsourced(
                    "the vendor alias table",
                    "%s, the _Base._resolve_alias call sites"
                    % _display_path(path),
                )
            )
    for estimator in _ESTIMATORS:
        node = _class_named(tree, estimator)
        if node is None:
            continue
        own = _signature_defaults(_func_named(node.body, "__init__"), consts)
        for name, default in own.items():
            if name in facts["accepted"]:
                continue
            facts["accepted"][name] = {"default": default, "owner": estimator}
    root = _repo_root()
    facts["canonicals"] = _canonical_map(
        root, facts["accepted"], facts["aliases"]
    )
    for entry in facts["aliases"].values():
        # Beside the wire name, so one entry has the same shape the
        # `parameter_aliases` block of the snapshot has. It falls back to
        # the wire name here where the snapshot writes `null`, because this
        # value is printed and splatted rather than diffed, and a report
        # that named no spelling at all would be worse than one that named
        # the spelling that certainly works.
        entry["canonical"] = facts["canonicals"].get(
            entry["wire"], entry["wire"]
        )
    if facts["aliases"] and not facts["canonicals"]:
        # One row, not one per parameter. Either the document is not here,
        # which is every wheel install, or it is here and this reader
        # understood none of it, which is a defect. The report cannot tell
        # those apart, so it names the file and says what is missing.
        facts["gaps"].append(
            Unsourced(
                "which spelling of each parameter is the canonical one, so "
                "every name below is the wire spelling",
                "docs/PARAMETER_NAMING.md, which is not in the installed "
                "wheel; run port() from a checkout for the canonical "
                "spellings",
            )
        )
    _ESTIMATOR_FACTS.update(facts)
    return _ESTIMATOR_FACTS


def _dataset_parameters():
    """`{key: the Dataset argument it belongs to}`, from `basic.py`.

    `basic._DATASET_PARAMS` already refuses these by name and names the
    argument they belong to, so its sentence is the refusal reason here too.
    """
    value = _module_values(_HERE / "basic.py", ("_DATASET_PARAMS",))[
        "_DATASET_PARAMS"
    ]
    return value if isinstance(value, dict) else {}


def _module_values(path, names):
    """Named module-level assignments of one file, as Python values."""
    tree, consts = _parse(path)
    out = dict.fromkeys(names, UNKNOWN)
    if tree is None:
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in out:
            out[target.id] = _literal(node.value, consts)
    return out


_PIPE = re.compile(r"(?<!\\)\|")
_BACKTICKED = re.compile(r"`([^`]+)`")


def _one_line(text):
    """Whitespace collapsed and table pipes escaped, so a cell stays a
    cell."""
    return " ".join(str(text).replace("\\|", "|").split()).replace("|", r"\|")


def _first_sentence(text, limit=320):
    """A bounded, readable piece of a contract note. Some notes in section 7
    run to thousands of characters; every row is cited, so the rest is one
    lookup away."""
    text = _one_line(text)
    if len(text) <= limit:
        return text
    cut = text.rfind(". ", 0, limit)
    if cut == -1:
        cut = text.rfind(" ", 0, limit)
    if cut == -1:
        cut = limit
    return text[:cut].rstrip(" .") + " ..."


_PARITY_ROWS = {}


def _parity_rows(root):
    """`{LightGBM parameter: (status, note)}` from section 7 of
    `docs/LIGHTGBM_PARITY.md`, the contract `tools/check_parity.py` checks."""
    if root is None:
        return {}
    key = str(root)
    if key in _PARITY_ROWS:
        return _PARITY_ROWS[key]
    rows = {}
    _PARITY_ROWS[key] = rows
    try:
        text = (root / "docs" / "LIGHTGBM_PARITY.md").read_text(
            encoding="utf-8"
        )
    except OSError:
        return rows
    start = text.find("\n## 7. ")
    if start == -1:
        return rows
    end = text.find("\n## ", start + 1)
    section = text[start : end if end != -1 else len(text)]
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = _PIPE.split(line)
        if len(parts) < 5:
            continue
        parts = parts[1:-1]
        first = parts[0].strip()
        if not first or set(first) <= set("-: "):
            continue
        if first == "Parameter":
            continue
        status = parts[1].strip().strip("*").strip("`").strip().lower()
        note = _first_sentence("|".join(parts[2:]))
        for name in _BACKTICKED.findall(first):
            rows[name.strip()] = (status, note)
    return rows


_NAMING_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAMING_PAREN = re.compile(r"\([^)]*\)")
_NAMING_INDEX = {}


def _naming_cell(cell):
    """Every parameter name one cell of the naming table holds.

    A cell may hold several (`eta / learning_rate`), may carry a
    parenthetical that is a note and not a name (`device (cpu | gpu)`,
    `(implicit 1)`), may be a dash for "this vendor has no such
    parameter", and may escape the table pipe. Only a bare identifier is a
    name.
    """
    text = cell.replace("\\|", "|").strip()
    text = _NAMING_PAREN.sub(" ", text)
    out = []
    for token in re.split(r"[/,]", text):
        token = token.strip().strip("`").strip()
        if _NAMING_IDENT.match(token):
            out.append(token)
    return out


def _naming_index(root):
    """`docs/PARAMETER_NAMING.md` as `(ours, by_any_name, by_lightgbm)`.

    `ours` is the canonical set, the OURS column. `by_lightgbm` maps the
    LightGBM column of a row to that row's canonical name, and `by_any`
    maps every vendor spelling in the row to the same. A name two rows
    claim is dropped from both, because an ambiguous canonical is not a
    canonical.

    Only the first five columns are read; the sixth is prose that names
    values rather than parameters. Rows whose OURS cell begins with `=` are
    values of `grow_policy` and are skipped for the same reason.
    """
    if root is None:
        return set(), {}, {}
    key = str(root)
    if key in _NAMING_INDEX:
        return _NAMING_INDEX[key]
    ours, by_any, by_lightgbm, ambiguous = set(), {}, {}, set()
    _NAMING_INDEX[key] = (ours, by_any, by_lightgbm)
    try:
        text = (root / "docs" / "PARAMETER_NAMING.md").read_text(
            encoding="utf-8"
        )
    except OSError:
        return _NAMING_INDEX[key]
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _PIPE.split(line)[1:-1]
        if len(cells) < 5:
            continue
        first = cells[0].strip()
        if not first or set(first) <= set("-: ") or first == "OURS":
            continue
        if first.startswith("="):
            continue
        names = _naming_cell(cells[0])
        if not names:
            continue
        ours.update(names)
        for name in names:
            by_any[name] = name
        if len(names) > 1:
            # Several parameters on one row (`top_rate / other_rate`), so
            # the vendor columns cannot be attributed to one of them.
            continue
        canonical = names[0]
        for column in cells[1:5]:
            for name in _naming_cell(column):
                if by_any.get(name, canonical) != canonical:
                    ambiguous.add(name)
                by_any.setdefault(name, canonical)
        for name in _naming_cell(cells[1]):
            if by_lightgbm.get(name, canonical) != canonical:
                ambiguous.add(name)
            by_lightgbm.setdefault(name, canonical)
    for name in ambiguous:
        by_any.pop(name, None)
        by_lightgbm.pop(name, None)
    return _NAMING_INDEX[key]


def _canonical_map(root, accepted, aliases):
    """`{wire name: canonical name}` for the pairs whose two spellings
    differ, from `docs/PARAMETER_NAMING.md`.

    Three lookups per pair, most specific first, and no invention.

    1. The wire name is itself an OURS name, so the two coincide and the
       pair is not in this map at all.
    2. The wire name is in the LightGBM column, so that row's OURS name is
       the canonical (`bagging_fraction` to `subsample`).
    3. The ALIAS is somewhere in the table, which is how a wire name that
       is ours rather than LightGBM's is recovered: `min_child_hess`
       appears in no column, and `min_sum_hessian_in_leaf` finds the row.

    Two guards, because this map decides the spelling a user is told to
    type. A canonical the estimator does not accept as a keyword is
    dropped, so nothing here can produce a `report.params` an estimator
    would reject. A canonical claimed by two different wire names is
    dropped from both, since that would collapse two parameters into one.
    """
    ours, by_any, by_lightgbm = _naming_index(root)
    if not ours:
        return {}
    mapped = {}
    for alias, entry in aliases.items():
        wire = entry["wire"]
        if wire in mapped or wire in ours:
            continue
        canonical = by_lightgbm.get(wire) or by_any.get(alias) or by_any.get(
            wire
        )
        if canonical is None or canonical == wire:
            continue
        if canonical not in accepted and canonical not in aliases:
            continue
        mapped[wire] = canonical
    claimed = {}
    for wire, canonical in mapped.items():
        claimed.setdefault(canonical, []).append(wire)
    for canonical, wires in claimed.items():
        if len(wires) > 1:
            for wire in wires:
                mapped.pop(wire, None)
    return mapped


def _native_refusal(name):
    """The extension's own "not implemented, here is what it would take"
    message for an option name, or None.

    Optional by construction. This module has to import on a checkout that
    has never been built, and `tools/port_report.py` loads it by path with
    no package around it, so both failures are swallowed on purpose.
    """
    try:
        from . import preflight as _preflight
    except Exception:
        return None
    try:
        return _preflight.unimplemented_option_message(str(name))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Vendor tables
# ---------------------------------------------------------------------------

#: `xgboost.Booster.save_config()` reports XGBoost's C++ parameter names.
#: A porting user types the Python spelling, and four of them differ. This
#: maps one string onto another; it holds no value and no default.
#:
#: Each pairing is confirmed inside this repository rather than recalled.
#: `eta`, `reg_lambda`, `reg_alpha` and `gamma` are all aliases in our own
#: `_resolve_alias` table, resolving to `learning_rate`, `lambda_l2`,
#: `lambda_l1` and `min_gain_to_split`, which is the same four parameters the
#: C++ names below denote.
#:
#: `alpha` is the one that matters. XGBoost's `alpha` is L1 regularization
#: and mojotrees's `alpha` is the objective's own scalar, the quantile level
#: and the Huber transition point. Two libraries, one word, two parameters,
#: and `_collision` reports it rather than letting a ported `alpha` land
#: silently on the wrong one.
_XGBOOST_CONFIG_SPELLINGS = {
    "eta": "learning_rate",
    "lambda": "reg_lambda",
    "alpha": "reg_alpha",
    "min_split_loss": "gamma",
}


def _vendor_defaults(source, root, notes):
    """`({vendor name: their stock default}, provenance sentence)`.

    Read from the tables that record each vendor's own resolved
    configuration. `({}, "")` when the checkout is not present, which is what
    puts the whole comparison into `unsourced` instead of into a guess.
    """
    if root is None:
        return {}, ""
    if source == "lightgbm":
        stock = _module_values(
            root / "tools" / "check_parity.py", ("LIGHTGBM_STOCK",)
        )["LIGHTGBM_STOCK"]
        if not isinstance(stock, dict):
            return {}, ""
        return dict(stock), (
            "tools/check_parity.py LIGHTGBM_STOCK, read from "
            "microsoft/LightGBM include/LightGBM/config.h at tag v4.7.0"
        )
    scenarios = root / "bench" / "real_data" / "scenarios.py"
    if source == "xgboost":
        values = _module_values(
            scenarios,
            ("XGBOOST_RESOLVED_DEFAULTS", "XGBOOST_DEFAULTS_SOURCE"),
        )
        table = values["XGBOOST_RESOLVED_DEFAULTS"]
        if not isinstance(table, dict):
            return {}, ""
        out = {}
        for name, entry in table.items():
            if isinstance(entry, dict) and "value" in entry:
                out[_XGBOOST_CONFIG_SPELLINGS.get(name, name)] = entry["value"]
            else:
                notes.append(
                    "XGBOOST_RESOLVED_DEFAULTS[%r] is not the (path, value) "
                    "shape this reader expects, so it was skipped" % (name,)
                )
        provenance = values["XGBOOST_DEFAULTS_SOURCE"]
        return out, (
            provenance
            if isinstance(provenance, str)
            else "bench/real_data/scenarios.py XGBOOST_RESOLVED_DEFAULTS"
        )
    values = _module_values(
        scenarios, ("CATBOOST_LEFT_AT_STOCK", "CATBOOST_DEFAULTS_SOURCE")
    )
    table = values["CATBOOST_LEFT_AT_STOCK"]
    if not isinstance(table, dict):
        return {}, ""
    provenance = values["CATBOOST_DEFAULTS_SOURCE"]
    return dict(table), (
        provenance
        if isinstance(provenance, str)
        else "bench/real_data/scenarios.py CATBOOST_LEFT_AT_STOCK"
    )


def _declared_divergences(root):
    """`{LightGBM parameter: (their stock, ours, why)}` from
    `STOCK_DIVERGENCES` in `tools/check_parity.py`.

    Every entry there is a departure this library argued for on the page and
    gated in CI, which is exactly the `defaulted_differently` category for
    `source="lightgbm"`. Nothing is restated; the reason is the gate's own.
    """
    if root is None:
        return {}
    table = _module_values(
        root / "tools" / "check_parity.py", ("STOCK_DIVERGENCES",)
    )["STOCK_DIVERGENCES"]
    return dict(table) if isinstance(table, dict) else {}


def _lightgbm_name_map(root):
    """`{LightGBM parameter: our parameter}` for the names that differ.

    From `STOCK_PYTHON_SIGNATURE` and `STOCK_PYTHON_CONSTANTS` in
    `tools/check_parity.py`, which is the mapping that gate already walks to
    check our signature against LightGBM's stock table.
    """
    mapping = {}
    if root is None:
        return mapping
    values = _module_values(
        root / "tools" / "check_parity.py",
        ("STOCK_PYTHON_SIGNATURE", "STOCK_PYTHON_CONSTANTS"),
    )
    signature = values["STOCK_PYTHON_SIGNATURE"]
    if isinstance(signature, dict):
        for ours, theirs in signature.items():
            mapping.setdefault(theirs, ours)
    constants = values["STOCK_PYTHON_CONSTANTS"]
    if isinstance(constants, dict):
        for constant, theirs in constants.items():
            for ours, named in _CONSTANT_DEFAULTS.items():
                if named == constant:
                    mapping.setdefault(theirs, ours)
    return mapping


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("1", "true", "yes", "on"):
            return True
        if word in ("0", "false", "no", "off"):
            return False
    return UNKNOWN


def _as_number(value):
    if isinstance(value, bool):
        return UNKNOWN
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return UNKNOWN
    return UNKNOWN


def _same(ours, theirs):
    """True, False, or UNKNOWN for two defaults that may be spelled in two
    types.

    `save_config()` returns every scalar as a string, so `"1"` and `1.0` are
    one default and `"depthwise"` and `"lossguide"` are two. Anything this
    cannot decide comes back UNKNOWN and is reported as unsourced, never
    resolved by picking a side.
    """
    if ours is UNKNOWN or theirs is UNKNOWN:
        return UNKNOWN
    if ours is REQUIRED or theirs is REQUIRED:
        return UNKNOWN
    if ours is None or theirs is None:
        # `None` in our signature means "unset, resolved at fit time" for
        # every parameter whose default is not a value. That is not a
        # number, so it is not compared to one.
        return UNKNOWN
    if isinstance(ours, bool) or isinstance(theirs, bool):
        left, right = _as_bool(ours), _as_bool(theirs)
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        return left == right
    left, right = _as_number(ours), _as_number(theirs)
    if left is not UNKNOWN and right is not UNKNOWN:
        scale = max(1.0, abs(left), abs(right))
        return abs(left - right) <= 1e-9 * scale
    if isinstance(ours, str) and isinstance(theirs, str):
        return ours.strip().lower() == theirs.strip().lower()
    if isinstance(ours, str) != isinstance(theirs, str):
        return UNKNOWN
    try:
        return bool(ours == theirs)
    except Exception:
        return UNKNOWN


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _show(value):
    if isinstance(value, _Sentinel):
        return str(value)
    try:
        return _one_line(repr(value))
    except Exception:
        return "unprintable"


def _table(headings, rows):
    """A markdown table.

    Markdown because a console reads it fine and the docs page needs it,
    which is what keeps the two from becoming two copies of the truth.
    """
    if not rows:
        return ""
    cells = [[_one_line(cell) for cell in row] for row in rows]
    widths = [len(head) for head in headings]
    for row in cells:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "| "
        + " | ".join(
            head.ljust(widths[index]) for index, head in enumerate(headings)
        )
        + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    for row in cells:
        lines.append(
            "| "
            + " | ".join(
                cell.ljust(widths[index]) for index, cell in enumerate(row)
            )
            + " |"
        )
    return "\n".join(lines)


@dataclass
class PortReport:
    """What `port()` did with one parameter dict.

    `params` is the canonical mojotrees dict, ready to splat into an
    estimator. The four lists are the classification, `unsourced` is every
    fact this report could not read, `notes` is everything worth saying that
    is neither, and `fact_sources` names the files that were read.
    """

    source: str
    params: dict = field(default_factory=dict)
    honored: list = field(default_factory=list)
    aliased_to: list = field(default_factory=list)
    defaulted_differently: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    unsourced: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    fact_sources: dict = field(default_factory=dict)

    @property
    def counts(self):
        """`{category: how many}`, for a caller that wants the shape before
        the detail."""
        return {
            "honored": len(self.honored),
            "aliased_to": len(self.aliased_to),
            "defaulted_differently": len(self.defaulted_differently),
            "refused": len(self.refused),
            "unsourced": len(self.unsourced),
        }

    def to_dict(self):
        """The whole report as plain dicts, lists, and scalars. A fact this
        report could not read is the string `"unknown"`, which is what it
        is."""

        def plain(value):
            return str(value) if isinstance(value, _Sentinel) else value

        def rows(items):
            return [
                {
                    name: plain(getattr(item, name))
                    for name in item.__dataclass_fields__
                }
                for item in items
            ]

        return {
            "source": self.source,
            "params": {key: plain(v) for key, v in self.params.items()},
            "honored": rows(self.honored),
            "aliased_to": rows(self.aliased_to),
            "defaulted_differently": rows(self.defaulted_differently),
            "refused": rows(self.refused),
            "unsourced": rows(self.unsourced),
            "notes": list(self.notes),
            "fact_sources": dict(self.fact_sources),
            "counts": self.counts,
        }

    def render(self, heading="mojotrees port report"):
        """The readable report.

        Markdown tables, so this one string serves the console and
        `docs/PORTING.md` alike.
        """
        counts = self.counts
        out = [
            "%s, source=%s" % (heading, self.source),
            "",
            "%d honored, %d aliased, %d defaulted differently, %d refused, "
            "%d unsourced."
            % (
                counts["honored"],
                counts["aliased_to"],
                counts["defaulted_differently"],
                counts["refused"],
                counts["unsourced"],
            ),
        ]

        if self.honored:
            out += [
                "",
                "**honored.** Ours under the same name, used as given.",
                "",
                _table(
                    ["parameter", "value", "owner", "note"],
                    [
                        [row.name, _show(row.value), row.owner, row.note]
                        for row in self.honored
                    ],
                ),
            ]

        if self.aliased_to:
            out += [
                "",
                "**aliased_to.** A vendor spelling of one of our names. The "
                "value is used as given, under the canonical name. `we call "
                "it` is the spelling to type; `on the wire` is the spelling "
                "the native layer is sent and the model file holds, and the "
                "two differ for eleven parameters.",
                "",
                _table(
                    ["you passed", "we call it", "on the wire", "value",
                     "note"],
                    [
                        [
                            row.name,
                            row.canonical,
                            row.wire or row.canonical,
                            _show(row.value),
                            row.note,
                        ]
                        for row in self.aliased_to
                    ],
                ),
            ]

        if self.defaulted_differently:
            out += [
                "",
                "**defaulted_differently.** You did not name these and our "
                "default is not %s's, so your model will differ from %s's "
                "unless you set them." % (self.source, self.source),
                "",
                _table(
                    [
                        "parameter",
                        "their name",
                        "their default",
                        "our default",
                        "declared",
                        "reason",
                    ],
                    [
                        [
                            row.name,
                            row.vendor_name,
                            _show(row.vendor_default),
                            _show(row.our_default),
                            "yes" if row.declared else "no",
                            row.reason or ("see " + row.citation),
                        ]
                        for row in self.defaulted_differently
                    ],
                ),
            ]

        if self.refused:
            out += [
                "",
                "**refused.** We do not implement these, and why.",
                "",
                _table(
                    ["parameter", "value", "status", "reason", "cited from"],
                    [
                        [
                            row.name,
                            _show(row.value),
                            row.status,
                            row.reason,
                            row.citation,
                        ]
                        for row in self.refused
                    ],
                ),
            ]

        if self.notes:
            out += ["", "**notes.**", ""]
            out += ["- " + _one_line(note) for note in self.notes]

        if self.unsourced:
            out += [
                "",
                "**unsourced.** Facts this report wanted and could not read. "
                "Nothing below is a claim; each line is an absence.",
                "",
                _table(
                    ["what", "wanted from"],
                    [[row.what, row.wanted_from] for row in self.unsourced],
                ),
            ]

        if self.fact_sources:
            out += [
                "",
                "**fact sources.**",
                "",
                _table(
                    ["fact", "read from"],
                    [
                        [name, str(where)]
                        for name, where in sorted(self.fact_sources.items())
                    ],
                ),
            ]

        return "\n".join(out).rstrip() + "\n"

    def __str__(self):
        return self.render()


# ---------------------------------------------------------------------------
# The function
# ---------------------------------------------------------------------------


def _our_default(name, facts, seen=None):
    """Our effective default for a parameter or for one of its aliases.

    Resolves through the WIRE name, because the wire name is the spelling
    that holds the stock default; every other spelling of the parameter
    carries `None` in the signature so that "unset" stays knowable.
    """
    seen = set() if seen is None else seen
    if name in seen:
        return UNKNOWN
    seen.add(name)
    entry = facts["accepted"].get(name)
    if entry is not None:
        default = entry["default"]
        if default is None and name in _CONSTANT_DEFAULTS:
            resolved = facts["constants"].get(
                _CONSTANT_DEFAULTS[name], UNKNOWN
            )
            if resolved is not UNKNOWN:
                return resolved
        if default is None and name in facts["aliases"]:
            return _our_default(facts["aliases"][name]["wire"], facts, seen)
        return default
    alias = facts["aliases"].get(name)
    if alias is not None:
        return _our_default(alias["wire"], facts, seen)
    return UNKNOWN


def _wire_name(name, facts):
    """The mojotrees parameter a key lands on, in the spelling the native
    layer is sent, or None when we have no such parameter.

    The wire name and not the canonical one, because this is the identity
    of the parameter: two spellings that land on one wire name are one
    parameter, which is what the conflict check and the
    `defaulted_differently` bookkeeping both need. What a user should TYPE
    is `_canonical_display`.
    """
    alias = facts["aliases"].get(name)
    if alias is not None:
        return alias["wire"]
    if name in facts["accepted"]:
        return name
    return None


def _canonical_display(name, facts):
    """The spelling a user should type for a parameter, given any spelling.

    The canonical name from `docs/PARAMETER_NAMING.md` where that document
    covers the parameter and names a keyword the estimator accepts, and the
    name given otherwise. It never returns a spelling `fit` would reject.
    """
    if name is None:
        return None
    wire = _wire_name(name, facts) or name
    return facts["canonicals"].get(wire, wire)


def _collision(vendor_name, facts, source):
    """`(what it lands on here, what the vendor means, the vendor's other
    spelling of it)` when one word names two different parameters.

    Derived rather than listed. A name we accept, whose vendor meaning
    resolves onto a different parameter of ours, is a word two libraries
    spell the same and mean differently. XGBoost only, because
    `_XGBOOST_CONFIG_SPELLINGS` is the only spelling map here and LightGBM's
    `alpha` is the same quantile-and-Huber scalar ours is.
    """
    if source != "xgboost":
        return None
    meant_as = _XGBOOST_CONFIG_SPELLINGS.get(vendor_name)
    if meant_as is None:
        return None
    # Compared on the wire names, because the question is whether the two
    # are one parameter, and reported with the canonical ones, because the
    # note tells the user what to type instead.
    typed = _wire_name(vendor_name, facts)
    meant = _wire_name(meant_as, facts)
    if typed is None or meant is None or typed == meant:
        return None
    return (
        _canonical_display(typed, facts),
        _canonical_display(meant, facts),
        meant_as,
    )


def _refusal(key, parity, dataset_params):
    """`(status, reason, citation)` for a key we do not accept, or
    `(status, None, citation)` when this repository states no reason."""
    status = "unimplemented"
    reason = None
    citation = "unsourced"
    if key in parity:
        status, reason = parity[key]
        citation = "docs/LIGHTGBM_PARITY.md section 7"
    if key in dataset_params:
        reason = (
            "%r describes the data, not the training run; pass it as %s"
            % (key, dataset_params[key])
        )
        citation = "python/mojotrees/basic.py _DATASET_PARAMS"
        status = "belongs to Dataset"
    native = _native_refusal(key)
    if native:
        reason = _first_sentence(native, 600)
        citation = (
            "src/mojotrees/tree_parameters_extra.mojo "
            "check_extra_option_supported"
        )
    return status, reason, citation


def port(params, source="lightgbm"):
    """Classify a vendor parameter dict and translate it.

    `params` is the dict you would have passed to LightGBM, XGBoost, or
    CatBoost. `source` is which of those it came from, and it decides one
    thing only, which is whose stock defaults `defaulted_differently` is
    computed against.

    Returns a `PortReport`. `report.params` is the canonical mojotrees dict,
    `print(report)` is the table, and `report.to_dict()` is the same content
    for a program. Nothing here raises for a configuration `fit` would
    reject; see this module's docstring for why.
    """
    if source not in SOURCES:
        raise ValueError(
            "unknown source %r; expected one of %s"
            % (source, ", ".join(SOURCES))
        )
    if params is None:
        given = {}
    elif isinstance(params, dict):
        given = dict(params)
    else:
        raise TypeError(
            "params must be a dict of vendor parameters, not %s"
            % type(params).__name__
        )

    facts = _estimator_facts()
    root = _repo_root()
    report = PortReport(source=source)
    report.unsourced.extend(facts["gaps"])
    report.fact_sources["accepted parameters and our defaults"] = (
        _display_path(facts["path"])
    )
    report.fact_sources["vendor alias table"] = (
        "%s, the _Base._resolve_alias call sites"
        % _display_path(facts["path"])
    )
    if facts["canonicals"]:
        report.fact_sources["canonical spellings"] = (
            "docs/PARAMETER_NAMING.md, the OURS column"
        )
    if root is None:
        report.unsourced.append(
            Unsourced(
                "%s's own stock defaults, so nothing here can be classified "
                "as defaulted_differently" % source,
                "tools/check_parity.py and bench/real_data/scenarios.py, "
                "neither of which is in the installed wheel; run port() from "
                "a checkout for this half of the report",
            )
        )
        report.unsourced.append(
            Unsourced(
                "the reason behind each refusal",
                "docs/LIGHTGBM_PARITY.md section 7, which is not in the "
                "installed wheel",
            )
        )

    parity = _parity_rows(root)
    dataset_params = _dataset_parameters()
    if parity:
        report.fact_sources["refusal reasons"] = (
            "docs/LIGHTGBM_PARITY.md, section 7"
        )

    # -- the keys the caller passed ------------------------------------------
    landed = {}
    for name, value in given.items():
        if name in facts["accepted"] or name in facts["aliases"]:
            key = name
        else:
            key = str(name).strip().lower()
        note = ""
        collision = _collision(key, facts, source)
        if collision is not None:
            typed, meant, spelled = collision
            note = (
                "%s spells this parameter %r and means %r by it; here %r is "
                "a different parameter and lands on %r. Pass %r for the %s "
                "meaning." % (source, key, meant, key, typed, spelled, source)
            )
            report.notes.append(note)

        alias = facts["aliases"].get(key)
        if alias is not None:
            wire = alias["wire"]
            spelled = _canonical_display(wire, facts)
            report.aliased_to.append(
                AliasedTo(
                    name=name,
                    canonical=spelled,
                    value=value,
                    note=note,
                    wire=wire,
                )
            )
        elif key in facts["accepted"]:
            wire = key
            # Kept as the caller spelled it, even where the canonical
            # spelling differs. `honored` promises "used as given", and a
            # report that renamed the key here would break that promise to
            # make a point about naming. The canonical spelling of a key
            # that is already ours is one lookup away in `docs/PORTING.md`.
            spelled = key
            report.honored.append(
                Honored(
                    name=name,
                    value=value,
                    owner=facts["accepted"][key]["owner"],
                    note=note,
                )
            )
        else:
            wire = None
            spelled = None
            status, reason, citation = _refusal(key, parity, dataset_params)
            if reason is None:
                status = "unknown"
                reason = (
                    "not an estimator parameter, and no row in this "
                    "repository names it, so no reason can be quoted here"
                )
                report.unsourced.append(
                    Unsourced(
                        "why %r is not accepted" % (key,),
                        "docs/LIGHTGBM_PARITY.md section 7 has no row for it",
                    )
                )
            report.refused.append(
                Refused(
                    name=name,
                    value=value,
                    status=status,
                    reason=reason,
                    citation=citation,
                )
            )

        if wire is None:
            continue
        # Keyed by the wire name, which is the parameter's identity: two
        # spellings collide when they land on one wire name, whatever the
        # caller typed and whatever the canonical spelling is.
        if wire in landed:
            first_name, first_value = landed[wire]
            if _same(first_value, value) is not True:
                report.notes.append(
                    "%r and %r are two spellings of %r and you gave them "
                    "different values. fit() raises on that rather than "
                    "letting one win, so set only one."
                    % (first_name, name, _canonical_display(wire, facts))
                )
            continue
        landed[wire] = (name, value)
        report.params[spelled] = value

    # -- the parameters the caller did not name ------------------------------
    named = set(landed)
    vendor_defaults, provenance = _vendor_defaults(source, root, report.notes)
    if provenance:
        report.fact_sources["%s stock defaults" % source] = provenance
    declared = _declared_divergences(root)

    if source == "lightgbm":
        if root is not None and not declared:
            report.unsourced.append(
                Unsourced(
                    "the declared departures from LightGBM's stock defaults",
                    "tools/check_parity.py STOCK_DIVERGENCES",
                )
            )
        name_map = _lightgbm_name_map(root)
        for vendor_name in sorted(declared):
            entry = declared[vendor_name]
            if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                report.unsourced.append(
                    Unsourced(
                        "our declared departure for LightGBM's %r"
                        % (vendor_name,),
                        "tools/check_parity.py STOCK_DIVERGENCES entry is "
                        "not the (stock, ours, why) shape this reader "
                        "expects",
                    )
                )
                continue
            stock, asserted, why = entry[0], entry[1], entry[2]
            if vendor_name in facts["accepted"]:
                ours_name = vendor_name
            else:
                ours_name = name_map.get(vendor_name)
            if ours_name is None:
                report.unsourced.append(
                    Unsourced(
                        "which mojotrees parameter LightGBM's %r is"
                        % (vendor_name,),
                        "tools/check_parity.py STOCK_PYTHON_SIGNATURE and "
                        "STOCK_PYTHON_CONSTANTS name no counterpart",
                    )
                )
                continue
            if ours_name in named:
                continue
            ours = _our_default(ours_name, facts)
            if _same(ours, asserted) is False:
                report.notes.append(
                    "tools/check_parity.py declares our %r as %r and %s "
                    "reads %r. The gate is the authority; this row shows "
                    "what the code says."
                    % (ours_name, asserted, facts["path"].name, ours)
                )
            report.defaulted_differently.append(
                DefaultedDifferently(
                    # `ours_name` is `tools/check_parity.py`'s spelling,
                    # which is the wire one; the column is what to type.
                    name=_canonical_display(ours_name, facts),
                    vendor_name=vendor_name,
                    vendor_default=stock,
                    our_default=asserted if ours is UNKNOWN else ours,
                    declared=True,
                    reason=_first_sentence(why, 600),
                    citation="tools/check_parity.py STOCK_DIVERGENCES",
                )
            )
        if vendor_defaults:
            report.notes.append(
                "Every LightGBM stock default not listed above is asserted "
                "equal to ours by tools/check_parity.py, on the Mojo and the "
                "Python surface both, so an undeclared difference fails CI "
                "rather than reaching this report."
            )
        return report

    if root is not None and not vendor_defaults:
        report.unsourced.append(
            Unsourced(
                "%s's own resolved defaults" % source,
                "bench/real_data/scenarios.py",
            )
        )
    for vendor_name in sorted(vendor_defaults):
        theirs = vendor_defaults[vendor_name]
        # The wire name throughout, because `named` is keyed by it and
        # `declared` below is keyed by LightGBM's spelling, which is the
        # same one. Only what the reader is shown is canonical.
        ours_name = _wire_name(vendor_name, facts)
        if ours_name is None or ours_name in named:
            continue
        shown = _canonical_display(ours_name, facts)
        ours = _our_default(ours_name, facts)
        verdict = _same(ours, theirs)
        if verdict is UNKNOWN:
            report.unsourced.append(
                Unsourced(
                    "whether our %r matches %s's %r, which is %s"
                    % (shown, source, vendor_name, _show(theirs)),
                    "our default reads %s, which is not a value this report "
                    "can compare; an unset parameter is resolved at fit time"
                    % _show(ours),
                )
            )
            continue
        if verdict:
            continue
        entry = declared.get(ours_name)
        is_declared = (
            isinstance(entry, (tuple, list)) and len(entry) >= 3
        )
        if is_declared:
            # The same departure LightGBM's own row carries. Our default is
            # one number whatever vendor is asking about it, so the reason
            # it moved is the same reason here.
            reason = _first_sentence(entry[2], 600)
            citation = "tools/check_parity.py STOCK_DIVERGENCES"
        else:
            reason = (
                "no per-parameter reason for this difference is recorded in "
                "this repository; only %s's own value is" % source
            )
            citation = "bench/real_data/scenarios.py"
            report.unsourced.append(
                Unsourced(
                    "why our %r differs from %s's %r"
                    % (shown, source, vendor_name),
                    "no table here states a reason; "
                    "tools/check_parity.py STOCK_DIVERGENCES covers LightGBM "
                    "only",
                )
            )
        report.defaulted_differently.append(
            DefaultedDifferently(
                name=shown,
                vendor_name=vendor_name,
                vendor_default=theirs,
                our_default=ours,
                declared=is_declared,
                reason=reason,
                citation=citation,
            )
        )
    return report
