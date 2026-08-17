"""An alias must be indistinguishable in behavior from its canonical.

That is a property of the ESTIMATOR CLASS and not of any one parameter, so
this file sweeps it instead of hand writing a case per pair. The pairs are
read at runtime from the derived table in `compatibility/api_snapshot.json`,
so a pair added later is swept without anybody remembering to add it here.

NAMING. The snapshot field this file reads is `parameter_aliases.<name>.wire`,
and until schema version 3 it was called `canonical`, which it never was. It
is the FIRST argument of the `_resolve_alias` call site, which that function
calls the `primary` and which its own docstring says "is not necessarily the
canonical user-facing name", giving `num_leaves` as the primary that the
canonical `max_leaves` resolves onto. Eleven of the forty five entries
disagreed with docs/PARAMETER_NAMING.md, which is why the field was renamed
and a real `canonical`, read from that document, was added beside it.

This file calls the two sides PRIMARY (the spelling that holds the stock
default, which is the `wire` field) and ALIAS (the spelling that defaults to
None), and every message prints both names. The equivalence being asserted
does not care which one is canonical, so the `canonical` field is read here
only to check that the two derivations of it agree; see
`test_port_table_agrees_with_snapshot`.

Two live bugs of exactly this shape reached a tagged release candidate.

1. `bagging_freq` carried a signature default of 0 while `subsample_freq`
   carried `None`. Since "None means the user did not name this" is the
   convention, an explicit `bagging_freq=0` could not be told apart from an
   omission and was silently overridden to 1, which moved predictions by
   0.257 on a configuration documented to be inert. The other spelling of
   the same parameter was honored correctly. One pair, two behaviors.
2. `_resolve_alias` substituted the stock default for ANY explicitly passed
   `None`, including for parameters whose signature default is a real
   value, so `device=None` became `device="cpu"` instead of raising.

Resolution is checked WITHOUT TRAINING. `_params`, `_callback_params`,
`_resolve_seeds`, `_resolve_device`, `_check_n_jobs`, `_resolve_boosting`,
`_resolve_grow_policy` and `_multi_target.wire_params` are all pure
functions of the constructor arguments, so two estimators that differ only
in which spelling was used can be compared on the record they resolve to.
The only test here that fits is `test_bagging_freq_zero_is_a_statement`,
because bug 1 was a prediction difference and the assertion that closes it
is a prediction comparison.

Usage, after building the extension with bindings/build.sh,
    pixi run python python/test_alias_equivalence.py
"""

import ast
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mojotrees import (
    MojoTreesClassifier,
    MojoTreesRanker,
    MojoTreesRegressor,
)

try:
    from mojotrees import _multi_target
except ImportError:  # pragma: no cover - the module ships with the package
    _multi_target = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SNAPSHOT = os.path.join(_REPO_ROOT, "compatibility", "api_snapshot.json")
_PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mojotrees"
)

#: The estimator classes the sweep runs over. `MojoTreesRegressor` is the
#: only one with a multi-target path, so it is the only one whose record
#: includes `_multi_target.wire_params`.
_CLASSES = (MojoTreesRegressor, MojoTreesClassifier, MojoTreesRanker)

#: The contexts every pair is swept in. A pair that agrees at the defaults
#: can still disagree elsewhere, which is what bug 1 was. `bagging_freq=0`
#: and `subsample_freq=0` agreed until a fraction below 1 was also named,
#: and then one of them was overridden and the other was not. The CatBoost
#: mode is here for the same reason, since three stock defaults change under
#: it and each one is resolved from a different member of its pair.
#:
#: A base key that names either member of the pair being probed is dropped
#: for that pair, so the base can never collide with the probe and turn an
#: honest conflict refusal into a false difference.
_BASE_CONFIGS = (
    ("defaults", {}),
    ("catboost mode", {"grow_policy": "symmetrictree"}),
    ("row bagging named", {"bagging_fraction": 0.5}),
)

#: A shape well under `AUTO_GPU_MIN_ROWS`, so `device='auto'` resolves to
#: the CPU on a machine with an accelerator and on one without, and the
#: record is the same either way.
_PROBE_ROWS = 1000
_PROBE_FEATURES = 8


# ------------------------------------------------------------------ part 1
#
# Discover the pairs. Never a hardcoded list.


def alias_pairs():
    """Every (alias, primary, stock default) the snapshot records.

    `tools/api_snapshot.py:alias_pairs` derives this table from the
    `_resolve_alias` call sites, so it is the same set of pairs the library
    actually resolves rather than a second list that can drift. The key is
    the alias (the second argument at the call site) and the `wire` field
    is the primary (the first), which is the spelling the native layer is
    sent and not necessarily the canonical user-facing one; see the module
    docstring.

    Some entries record `fallback_default` as null, which means the third
    argument at the call site was not a literal (`learning_rate` and
    `lambda_l2` take a mode-dependent default), so a null is a missing fact
    and not a value. Callers must treat it as unknown rather than as None.
    """
    with open(_SNAPSHOT) as handle:
        snapshot = json.load(handle)
    table = snapshot["python"]["parameter_aliases"]
    pairs = []
    for alias in sorted(table):
        entry = table[alias]
        pairs.append((alias, entry["wire"], entry.get("fallback_default")))
    return pairs


def test_alias_table_is_discovered():
    pairs = alias_pairs()
    assert len(pairs) >= 40, f"alias table looks truncated: {len(pairs)}"
    for alias, primary, _fallback in pairs:
        assert isinstance(alias, str) and alias
        assert isinstance(primary, str) and primary
        assert alias != primary, f"{alias} is its own primary"
    # The pair that carried bug 1 has to be in here, or the sweep below is
    # sweeping the wrong table.
    names = {alias for alias, _p, _f in pairs}
    assert "subsample_freq" in names, "the bug-1 pair is missing"
    unresolved = [a for a, _p, f in pairs if f is None]
    print(
        f"alias table ok ({len(pairs)} pairs, "
        f"{len(unresolved)} with an underived stock default)"
    )


def test_port_table_agrees_with_snapshot():
    """The two derivations of the alias table are one fact, not two.

    `tools/api_snapshot.py:alias_pairs` and `port._alias_pairs` walk the
    same `_resolve_alias` call sites, and both read the canonical spelling
    out of `docs/PARAMETER_NAMING.md` the same three ways. Neither imports
    the other, because the snapshot is not in the wheel and the tool
    imports no part of the package, so the only thing holding them together
    is this comparison.

    A canonical the estimator does not accept as a keyword is dropped by
    `port` and kept by the snapshot, deliberately: `port` hands back a dict
    a caller splats into a constructor and must never name a keyword that
    would raise. That case is printed rather than failed.
    """
    # THE MODULE, not the function, and getting this takes `importlib`.
    #
    # `__init__.py` does `from .port import port` eagerly, so the attribute
    # `mojotrees.port` is the CALLABLE and it shadows the submodule of the
    # same name. Both `from mojotrees import port` and
    # `import mojotrees.port as port_module` bind that attribute, because the
    # `as` form reads the attribute after the import rather than taking the
    # module object, so both fail with "'function' object has no attribute
    # '_estimator_facts'". `import_module` returns `sys.modules` entry itself
    # and is unaffected by the shadowing.
    #
    # This is the same name collision `_public_api_plan.TOP_LEVEL_ADDITIONS`
    # records for `cv` and `CVBooster`, met a third time, and it is worth the
    # paragraph because the two wrong spellings look correct.
    import importlib

    port_module = importlib.import_module("mojotrees.port")

    with open(_SNAPSHOT) as handle:
        table = json.load(handle)["python"]["parameter_aliases"]
    facts = port_module._estimator_facts()
    theirs = facts["aliases"]
    failures = []
    notes = []
    only_snapshot = sorted(set(table) - set(theirs))
    only_port = sorted(set(theirs) - set(table))
    if only_snapshot or only_port:
        failures.append(
            "the two derivations disagree about which aliases exist; only "
            f"in the snapshot: {only_snapshot or 'none'}; only in port.py: "
            f"{only_port or 'none'}"
        )
    for alias in sorted(set(table) & set(theirs)):
        snap, mine = table[alias], theirs[alias]
        if snap["wire"] != mine["wire"]:
            failures.append(
                f"{alias}: the snapshot resolves it onto "
                f"{snap['wire']!r} and port.py onto {mine['wire']!r}"
            )
            continue
        snap_canonical = snap.get("canonical")
        mine_canonical = mine.get("canonical")
        if snap_canonical is None:
            notes.append(
                f"{alias}: docs/PARAMETER_NAMING.md names no canonical "
                f"spelling for the wire name {mine['wire']!r}, so the "
                "snapshot records null and port.py reports the wire name"
            )
        elif snap_canonical != mine_canonical:
            if snap_canonical not in signature_defaults(MojoTreesRegressor):
                notes.append(
                    f"{alias}: the naming document calls "
                    f"{snap_canonical!r} canonical, and it is not a "
                    "constructor keyword, so port.py keeps "
                    f"{mine_canonical!r}"
                )
            else:
                failures.append(
                    f"{alias}: the snapshot calls {snap_canonical!r} "
                    f"canonical and port.py calls it {mine_canonical!r}, "
                    "from the same document"
                )
    for line in notes:
        print(f"  NOTE {line}")
    for line in failures:
        print(f"  {line}")
    assert not failures, (
        f"{len(failures)} disagreements between the snapshot's alias table "
        "and port.py's"
    )
    print(f"port and snapshot alias tables agree ok ({len(theirs)} pairs)")


# ------------------------------------------------------------------ part 2
#
# The equivalence sweep.


#: Non-default probe values for pairs where the generic rule below has
#: nothing to work from, which is every pair whose stock default the
#: snapshot could not derive plus the two string-valued ones. Keyed by the
#: PRIMARY name, because that is what decides the type.
_NON_DEFAULT_BY_PRIMARY = {
    "learning_rate": 0.05,
    "lambda_l2": 2.0,
    "random_state": 7,
    # A positive worker count is refused by name through every spelling,
    # which is a resolved behavior and therefore an observable one. -1 is
    # accepted by doing nothing, which no surface can see.
    "n_jobs": 4,
    "verbose": 1,
    "early_stopping_rounds": 5,
    "interaction_constraints": [[0, 1]],
    "monotone_constraints": [1, -1],
    # 'auto' is a second accepted value and resolves to 'cpu' at this shape.
    "device": "auto",
    # LightGBM's index-list form, accepted through every spelling.
    "categorical_feature": [0],
}


def non_default_probe(primary, fallback):
    """A legal value that is not the stock default, or None when there is no
    honest way to build one for this parameter."""
    if primary in _NON_DEFAULT_BY_PRIMARY:
        return _NON_DEFAULT_BY_PRIMARY[primary]
    if fallback is None:
        return None
    if isinstance(fallback, bool):
        return not fallback
    if isinstance(fallback, int):
        # A count. A non-positive stock value means "no limit", so a small
        # positive integer is what differs from it; a positive stock value
        # halves to something still legal for every count in the table.
        return 5 if fallback <= 0 else max(2, fallback // 2)
    if isinstance(fallback, float):
        # A fraction. Everything float-valued in this table is either a rate
        # in (0, 1] or a nonnegative penalty, and 0.5 is legal for both.
        if fallback <= 0.0 or fallback >= 1.0:
            return 0.5
        return fallback / 2.0
    return None


def signature_defaults(cls):
    """The signature default of every constructor parameter of `cls`.

    THE WHOLE MRO, not `cls.__init__`. The parameters are split across two
    signatures, since `MojoTreesRegressor.__init__` declares about ten that
    are specific to the task and `_Base.__init__` declares the other hundred
    and thirty six. Reading one class's `__init__` returns a set missing
    almost everything, which is the mistake `_Base._none_defaulted`
    documents having made.
    """
    out = {}
    for klass in reversed(cls.__mro__):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            parameters = inspect.signature(init).parameters
        except (TypeError, ValueError):  # pragma: no cover
            continue
        for name, parameter in parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                continue
            out[name] = parameter.default
    return out


def _capture(thunk):
    """The value a resolution surface produces, or the error it raises.

    Two spellings of one parameter have to agree about the refusals too, so
    an exception is part of the record rather than the end of the
    comparison.
    """
    try:
        value = thunk()
    except Exception as exc:  # noqa: BLE001 - the type is part of the record
        return ("raise", type(exc).__name__, str(exc))
    return ("ok", value)


def _strip_addresses(params):
    """`_params` without its pointer fields.

    The `*_addr` keys carry the addresses of buffers this file never builds,
    so they are zero here, and comparing them would compare identities
    rather than resolved values. The matching `*_len` keys stay, which is
    where anything interesting about those buffers shows up.
    """
    return {k: v for k, v in params.items() if not k.endswith("_addr")}


def resolution_record(est, with_multi_target):
    """Every parameter-resolving surface a fit reads, without fitting.

    None of these trains. `_params` is the wire dict the trainer is handed,
    `_callback_params` is what a before-iteration callback sees,
    `_resolve_seeds` is the seed fan-out, `_resolve_device` is the backend
    choice, `_check_n_jobs`, `_resolve_boosting` and `_resolve_grow_policy`
    are the remaining resolvers, and `_multi_target.wire_params` is the
    second wire dict, the one a `num_targets` fit is sent. Two spellings
    that produce the same record here are two spellings no fit can tell
    apart.
    """
    out = {}
    device = _capture(
        lambda: est._resolve_device(_PROBE_ROWS, _PROBE_FEATURES, 1)
    )
    out["device"] = device
    if device[0] == "ok":
        out["params"] = _capture(
            lambda: _strip_addresses(est._params(0, device[1]))
        )
    else:
        # A fit stops at the device error, so there is nothing further to
        # compare and both spellings have to arrive here the same way.
        out["params"] = ("blocked by the device error",)
    out["callback_params"] = _capture(est._callback_params)
    out["seeds"] = _capture(est._resolve_seeds)
    out["n_jobs"] = _capture(est._check_n_jobs)
    out["boosting"] = _capture(est._resolve_boosting)
    out["grow_policy"] = _capture(est._resolve_grow_policy)
    if with_multi_target and _multi_target is not None:
        out["multi_target"] = _capture(lambda: _multi_target.wire_params(est))
    return out


def test_alias_equivalence_sweep():
    """For every pair, every base configuration and every probe, the two
    spellings must resolve to the same record."""
    pairs = alias_pairs()
    failures = []
    skipped = []
    # Pairs whose probe moved SOMETHING in the record, on any class under any
    # base. Tracked globally and not per class, because a surface that is
    # blind for one estimator can be the one that sees the pair for another,
    # and a note claiming otherwise would be wrong.
    observed = set()
    swept = 0
    for cls in _CLASSES:
        defaults = signature_defaults(cls)
        with_multi_target = cls is MojoTreesRegressor
        for base_name, base in _BASE_CONFIGS:
            for alias, primary, fallback in pairs:
                if alias not in defaults or primary not in defaults:
                    skipped.append(
                        f"{cls.__name__}: {alias}/{primary} is not a "
                        "constructor parameter of this class"
                    )
                    continue
                # A base key naming either spelling of the pair under test
                # would be a second value for the parameter being probed,
                # and the conflict refusal would fire on one arm only.
                context = {
                    k: v
                    for k, v in base.items()
                    if k not in (alias, primary)
                }
                probes = []
                if fallback is not None:
                    probes.append(("stock default", fallback))
                else:
                    skipped.append(
                        f"{alias}/{primary} has no stock default in the "
                        "snapshot (the call site passes a computed "
                        "default), so the stock probe is skipped"
                    )
                other = non_default_probe(primary, fallback)
                if other is None:
                    skipped.append(
                        f"{alias}/{primary} has no legal non-default probe "
                        "this file can build, so only the unset probe ran"
                    )
                else:
                    probes.append(("non-default", other))
                # The unset probe omits the keyword rather than passing
                # None. An explicit None is NOT an omission for a parameter
                # whose signature default is a real value, which is bug 2
                # and is asserted by `test_explicit_none_is_not_swallowed`.
                baseline = resolution_record(
                    cls(**context), with_multi_target
                )
                repeat = resolution_record(cls(**context), with_multi_target)
                swept += 1
                for surface in sorted(baseline):
                    if baseline[surface] != repeat[surface]:
                        failures.append(
                            f"{cls.__name__} [{base_name}] {alias} vs "
                            f"{primary} [unset] surface {surface} is not "
                            "reproducible"
                        )
                for label, value in probes:
                    by_alias = resolution_record(
                        cls(**dict(context, **{alias: value})),
                        with_multi_target,
                    )
                    by_primary = resolution_record(
                        cls(**dict(context, **{primary: value})),
                        with_multi_target,
                    )
                    swept += 1
                    for surface in sorted(by_alias):
                        got = by_alias[surface]
                        want = by_primary[surface]
                        if got != want:
                            failures.append(
                                f"{cls.__name__} [{base_name}] "
                                f"{alias}={value!r} vs {primary}={value!r} "
                                f"[{label}] surface {surface}\n"
                                f"      {alias:<24} {got}\n"
                                f"      {primary:<24} {want}"
                            )
                    if by_alias != baseline:
                        observed.add((alias, primary))
    for line in sorted(set(skipped)):
        print(f"  SKIPPED {line}")
    for alias, primary, _fallback in pairs:
        if (alias, primary) not in observed:
            print(
                f"  NOTE {primary} is resolved by no surface this file can "
                "reach without fitting, so its pair with "
                f"{alias} was compared only for agreement"
            )
    if failures:
        print(f"  {len(failures)} SPELLINGS THAT RESOLVE DIFFERENTLY")
        for line in failures:
            print(f"    {line}")
    assert not failures, (
        f"{len(failures)} alias/primary comparisons resolve differently; "
        "see the listing above"
    )
    print(
        f"alias equivalence sweep ok ({swept} comparisons over "
        f"{len(pairs)} pairs, {len(_BASE_CONFIGS)} base configurations and "
        f"{len(_CLASSES)} classes)"
    )


# ------------------------------------------------------------------ part 3
#
# The two shipped bugs, guarded by name.


def _rand_stream(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (state >> 11) / 9007199254740992.0


def make_regression(n_rows, n_features=4, seed=7):
    rng = _rand_stream(seed)
    X = [[next(rng) for _ in range(n_features)] for _ in range(n_rows)]
    y = [3.0 * r[0] + 2.0 * r[1] * r[2] + 0.05 * (next(rng) - 0.5) for r in X]
    return X, y


def resolved_params(est):
    """The wire dict a fit would be sent, resolved without fitting."""
    return _strip_addresses(
        est._params(0, est._resolve_device(_PROBE_ROWS, _PROBE_FEATURES, 1))
    )


def test_bagging_freq_zero_is_a_statement():
    """Bug 1. An explicit frequency of 0 disables bagging through EITHER
    spelling, and both stay distinguishable from an omission."""
    # An unnamed frequency beside a fraction below 1 implies 1, which is the
    # deliberate divergence from LightGBM's silent no-op.
    for fraction in ("bagging_fraction", "subsample"):
        implied = resolved_params(MojoTreesRegressor(**{fraction: 0.5}))
        assert implied["bagging_freq"] == 1, (
            f"{fraction}=0.5 with no frequency should imply 1, got "
            f"{implied['bagging_freq']}"
        )
        assert implied["bagging_fraction"] == 0.5
        # An explicit 0 is a statement and not an omission, whichever
        # spelling of the frequency and whichever spelling of the fraction.
        for frequency in ("bagging_freq", "subsample_freq"):
            off = resolved_params(
                MojoTreesRegressor(**{fraction: 0.5, frequency: 0})
            )
            assert off["bagging_freq"] == 0, (
                f"{frequency}=0 was overridden to {off['bagging_freq']}"
            )
            assert off["bagging_fraction"] == 0.5

    # And the same at the level the regression was measured at. The bug moved
    # predictions by 0.257 on a configuration documented to be inert.
    X, y = make_regression(400)
    baseline = list(MojoTreesRegressor(n_estimators=20).fit(X, y).predict(X))
    for fraction in ("bagging_fraction", "subsample"):
        for frequency in ("bagging_freq", "subsample_freq"):
            kwargs = {fraction: 0.5, frequency: 0}
            off = MojoTreesRegressor(n_estimators=20, **kwargs).fit(X, y)
            worst = max(abs(a - b) for a, b in zip(baseline, off.predict(X)))
            assert worst == 0.0, (
                f"{kwargs} is documented to be inert but moved predictions "
                f"by {worst}"
            )
    # The fixture is not vacuous. With neither frequency named, the same
    # fraction does bag and does move the fit.
    bagged = MojoTreesRegressor(n_estimators=20, bagging_fraction=0.5).fit(
        X, y
    )
    assert any(a != b for a, b in zip(baseline, bagged.predict(X))), (
        "an unnamed frequency did not bag, so the checks above prove nothing"
    )
    print("bagging_freq=0 and subsample_freq=0 agree ok")


def test_explicit_none_is_not_swallowed():
    """Bug 2. `None` means "unset" only for a parameter whose SIGNATURE
    default is `None`. Everywhere else it is a value the user typed and it
    reaches validation."""
    for cls in _CLASSES:
        defaults = signature_defaults(cls)
        assert defaults["device"] == "cpu", (
            "device's signature default is no longer a real value, so an "
            "explicit device=None now means 'unset' and this test is "
            "guarding the wrong parameter"
        )
        try:
            cls(device=None)._resolve_device(_PROBE_ROWS, _PROBE_FEATURES, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{cls.__name__}(device=None) should raise ValueError, not "
                "resolve to the stock default"
            )

    # The other direction. `learning_rate`, `lambda_l2` and `bagging_freq`
    # carry `None` in the signature precisely so that "unset" stays
    # knowable, so an explicit None there IS an omission and must resolve
    # exactly as one.
    defaults = signature_defaults(MojoTreesRegressor)
    baseline = resolved_params(MojoTreesRegressor())
    for name in ("learning_rate", "lambda_l2", "bagging_freq"):
        assert defaults[name] is None, (
            f"{name} no longer carries None, so 'unset' is no longer "
            "knowable for it"
        )
        explicit = resolved_params(MojoTreesRegressor(**{name: None}))
        assert explicit == baseline, (
            f"{name}=None did not resolve the way an omission does"
        )
    assert baseline["learning_rate"] == 0.1, (
        "an unset learning rate should be 0.1, got "
        f"{baseline['learning_rate']}"
    )
    print("explicit None is honored per signature ok")


# ------------------------------------------------------------------ part 4
#
# The signature-level invariants, which is where bug 1 was born.


def test_alias_signature_defaults():
    """One spelling of a parameter may hold the stock default and the other
    must hold None.

    The shape the code follows, verified across the table, is that the
    PRIMARY carries the stock default and the ALIAS carries `None`, meaning
    "not used". A primary may also carry `None`, and three of them do
    (`learning_rate`, `lambda_l2` and, since bug 1 was fixed,
    `bagging_freq`), because that is the only way "the user did not name
    this" survives to fit time; `_resolve_alias` substitutes the stock value
    for them. What is never allowed is an ALIAS carrying a real default,
    since then an explicit stock value through that spelling cannot be told
    apart from an omission while through the other spelling it can, which is
    the inequivalence bug 1 was.
    """
    pairs = alias_pairs()
    failures = []
    for cls in _CLASSES:
        defaults = signature_defaults(cls)
        for alias, primary, fallback in pairs:
            if alias not in defaults or primary not in defaults:
                continue
            alias_default = defaults[alias]
            primary_default = defaults[primary]
            if alias_default is not None:
                failures.append(
                    f"{cls.__name__}: {alias} defaults to "
                    f"{alias_default!r} and {primary} defaults to "
                    f"{primary_default!r}. The spelling that is resolved "
                    "onto the other must default to None, or an explicit "
                    "value through it is indistinguishable from an "
                    "omission, which is the shape of the bagging_freq bug."
                )
                continue
            if (
                fallback is not None
                and primary_default is not None
                and primary_default != fallback
            ):
                failures.append(
                    f"{cls.__name__}: {primary} defaults to "
                    f"{primary_default!r} but _resolve_alias substitutes "
                    f"{fallback!r} for an unset value, so the signature and "
                    "the resolver disagree about the stock default of the "
                    f"{alias}/{primary} pair."
                )
    for line in failures:
        print(f"  {line}")
    assert not failures, f"{len(failures)} signature-default defects"
    print(f"alias signature defaults ok ({len(pairs)} pairs)")


def _provenance_reads():
    """Every `<self|est|estimator>.NAME is [not] None` in the package, as
    NAME -> [(file, line)].

    Read with `ast` and not with a regex, so a comment that quotes the old
    code (sklearn.py has two of those) is not mistaken for the code.

    A HIT IS DROPPED WHEN THE ENCLOSING CLASS OWNS THE NAME ITSELF, and that
    exclusion is load bearing rather than a convenience. `self` is not always
    an estimator. `device_selection.py` defines a workload-description class
    whose own `__init__` takes `max_bin=None`, so `self.max_bin is None` there
    is a real question with a real answer, and matching that name against the
    ESTIMATOR's signature default of 255 reported a defect that does not
    exist. The first version of this function did exactly that and the false
    positive survived until somebody read the other class. So the test asks
    which class the attribute belongs to before judging it, and a name a class
    declares in its own constructor is that class's parameter and not ours.
    """
    hits = {}
    for entry in sorted(os.listdir(_PACKAGE_DIR)):
        if not entry.endswith(".py"):
            continue
        path = os.path.join(_PACKAGE_DIR, entry)
        try:
            with open(path) as handle:
                tree = ast.parse(handle.read(), filename=path)
        except (OSError, SyntaxError):  # pragma: no cover
            continue
        # name -> the set of parameters the class declaring it owns, so an
        # attribute can be attributed to the class it actually belongs to.
        owned = {}
        for klass in ast.walk(tree):
            if not isinstance(klass, ast.ClassDef):
                continue
            own = set()
            for item in klass.body:
                if not isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if item.name != "__init__":
                    continue
                args = item.args
                for argument in (
                    list(args.posonlyargs)
                    + list(args.args)
                    + list(args.kwonlyargs)
                ):
                    own.add(argument.arg)
            for inner in ast.walk(klass):
                owned[id(inner)] = own
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if len(node.ops) != 1 or not isinstance(
                node.ops[0], (ast.Is, ast.IsNot)
            ):
                continue
            right = node.comparators[0]
            if not (isinstance(right, ast.Constant) and right.value is None):
                continue
            left = node.left
            if not isinstance(left, ast.Attribute):
                continue
            owner = left.value
            if not isinstance(owner, ast.Name):
                continue
            if owner.id not in ("self", "est", "estimator"):
                continue
            if left.attr in owned.get(id(node), ()):
                # The enclosing class declares this name in its own
                # `__init__`, so `self.<name>` is that class's parameter and
                # its default governs, not the estimator's. See the docstring.
                continue
            hits.setdefault(left.attr, []).append((entry, node.lineno))
    return hits


def test_provenance_is_askable():
    """A parameter whose provenance is READ must be able to answer.

    `X.name is None` is a question about whether the caller named the
    parameter. If that parameter's signature default is a real value, the
    question has one answer forever, the branch is decided before any caller
    is heard, and the fit acts on a fact nobody established. That is bug 2
    (`self.device is not None`, with `device='cpu'` in the signature, is
    always true) and it is the condition that made bug 1 possible.

    Restricted to members of the alias table, because that is the surface
    this file owns; a constant-folded provenance test on a parameter with no
    aliases is the same defect but somebody else's lane.
    """
    pairs = alias_pairs()
    reads = _provenance_reads()
    members = set()
    for alias, primary, _fallback in pairs:
        members.add(alias)
        members.add(primary)
    failures = []
    asymmetric = []
    checked = 0
    defaults = signature_defaults(MojoTreesRegressor)
    for name in sorted(members):
        sites = reads.get(name)
        if not sites:
            continue
        if name not in defaults:
            continue
        checked += 1
        where = ", ".join(f"{f}:{n}" for f, n in sorted(set(sites)))
        if defaults[name] is not None:
            failures.append(
                f"{name} is tested against None at {where}, but its "
                f"signature default is {defaults[name]!r}, so the test can "
                "never change and the branch is decided before any caller "
                "is heard."
            )
    # Not a failure, and it is the judgment call this file deliberately
    # leaves to a reader. When one spelling can report an omission and the
    # other cannot, the two are equivalent only as long as nothing
    # downstream asks. Bug 1 lived in exactly that gap for a release, so the
    # pairs sitting in it are printed for review rather than left silent.
    for alias, primary, _fallback in pairs:
        if alias not in defaults or primary not in defaults:
            continue
        for named, partner in ((alias, primary), (primary, alias)):
            if not reads.get(named):
                continue
            if defaults[partner] is not None:
                asymmetric.append(
                    f"{named} is read for provenance at "
                    + ", ".join(
                        f"{f}:{n}" for f, n in sorted(set(reads[named]))
                    )
                    + f", but its other spelling {partner} defaults to "
                    f"{defaults[partner]!r} and cannot report an omission"
                )
    for line in sorted(set(asymmetric)):
        print(f"  REVIEW {line}")
    for line in failures:
        print(f"  {line}")
    assert not failures, (
        f"{len(failures)} parameters are asked a question they cannot answer"
    )
    print(f"provenance is askable ok ({checked} table members read for it)")


if __name__ == "__main__":
    test_alias_table_is_discovered()
    test_port_table_agrees_with_snapshot()
    test_alias_equivalence_sweep()
    test_bagging_freq_zero_is_a_statement()
    test_explicit_none_is_not_swallowed()
    test_alias_signature_defaults()
    test_provenance_is_askable()
    print("all alias equivalence tests passed")
