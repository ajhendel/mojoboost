"""Workload extraction and device-decision formatting for Python.

`device="auto"` has to answer one question, "CPU or GPU for this run", and
it has to be able to say why. **Neither half of that answer is computed
here.** The decision, the rules behind it, the memory estimate, the
hardware capabilities, and the refusal an explicit `device="gpu"` gets all
live in `src/mojotrees/device_policy.mojo`, which is the one authoritative
implementation. This module does the two things a native policy cannot do
from inside Mojo:

1. **Extraction.** Read `X` and `y` far enough to fill in plain workload
   metadata: rows, features, sparseness, class count. No judgment, no
   thresholds, no capability probing. `Workload` is that metadata.
2. **Formatting.** Take the decision the native layer returns and render
   it as prose a user can read or as a dict a support ticket can carry.
   `DeviceReport` is that renderer.

    from mojotrees.device_selection import explain_device_choice

    print(explain_device_choice(X, y, device="auto"))

Three requested values, the same vocabulary the Mojo layer uses:

- `"cpu"`, the default and the dependable path. Always resolves to itself.
- `"gpu"`, an explicit request. It runs or it raises. There is no silent
  fallback to the CPU, because a fallback turns "my GPU run" into "a CPU
  run that took the same wall clock and I never knew".
- `"auto"`, which picks the GPU only when the GPU path covers the workload
  and evidence says the GPU is the faster choice for that shape on that
  device. With no such evidence it picks the CPU and says so.

Why `auto` is the CPU everywhere today, what `MOJOTREES_AUTO_MIN_CELLS`
and `MOJOTREES_DISABLE_GPU` do, and what evidence a crossover rule has to
carry are documented in `src/mojotrees/device_policy.mojo`. Do not restate
those rules here, and do not add a threshold here: a rule that exists in
this file and not in that one is a rule the Mojo API, the CLI, and the C
API do not have.

The native boundary
-------------------
`_NATIVE` is the single, deliberately narrow seam through which every
decision arrives. It has two modes, and `DeviceReport.contract` says which
one produced a given report:

- `"full"`: `_mojotrees.decide_device(...)` was available. That binding is
  `decide_device_report` in `src/mojotrees/device_policy.mojo`, whose ten
  parameters are exactly the ten this module sends. The whole contract
  crossed, and the report carries the blocking reasons, the warnings, the
  memory estimate, the transfer route each device buffer is on, how much
  of the one-time startup cost the process has paid, the policy version,
  and the evidence identifier the native layer produced.
- `"narrow"`: only the older `_mojotrees.resolve_device(...)` was
  available. That entry point runs the *same* native engine, so the
  selected backend is still the native answer and never a Python one; it
  just carries the shape and nothing else, so the gates that need an
  objective, a bin count, or the input flags were skipped natively and the
  report says so.

`"narrow"` is a temporary state. It exists only until the binding named in
`handoffs/migration_20_device_policy.md` is added, and the code that
implements it is confined to `_NarrowNativePolicy` so it can be deleted in
one piece. It computes nothing: on a refusal it reports the native error
text, and on a success it reports the native backend.

Reports, not booleans
---------------------
`select_device` returns a `DeviceReport`: the resolution, the ordered
reasons behind it, the warnings, the workload as it was understood, and
the native decision's own fields. `report.explanation` renders the same
content as prose, and `str(report)` is that explanation. `report.to_dict()`
is JSON-serializable, which is what a support ticket or a CI log wants.

`explain_device_choice` is the same call in a form that never raises: for a
request that would fail it sets `would_raise` and `error` instead, so a
user can ask "what would `device='gpu'` do here" without handling an
exception. `report.raise_if_unsupported()` turns it back into the raise.
"""

import json

__all__ = [
    "CONTRACT_FULL",
    "CONTRACT_NARROW",
    "DEVICES",
    "DeviceReport",
    "DeviceUnavailableError",
    "NativePolicyUnavailable",
    "Reason",
    "TransferRoute",
    "PredictSupport",
    "Workload",
    "explain_device_choice",
    "explain_predict_device",
    "native_contract",
    "select_device",
]

#: The public device vocabulary, as in `mojotrees._DEVICES` and as
#: `parse_device` in src/mojotrees/device_policy.mojo accepts it.
DEVICES = ("cpu", "gpu", "auto")


class DeviceUnavailableError(RuntimeError):
    """An explicit `device="gpu"` cannot run this workload.

    A subclass of `RuntimeError` so it is caught by code written against
    what the estimators raise today.
    """

    def __init__(self, message, report=None):
        RuntimeError.__init__(self, message)
        #: The `DeviceReport` behind the refusal, when one was built.
        self.report = report


class NativePolicyUnavailable(RuntimeError):
    """The compiled extension could not be loaded, so no decision can be
    made at all.

    Deliberately fatal rather than falling back to a Python answer. A
    Python answer is exactly what this module no longer has, and inventing
    one to paper over a missing build would put a second policy back.
    """


class Reason:
    """One ordered step of the decision, a stable `code` and prose.

    Both fields come from the native decision. The codes are the
    `block_reason_name` and `warning_name` strings in
    src/mojotrees/device_policy.mojo, which are stable because they end up
    in `to_dict()` output that something else may match on.
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message

    def to_dict(self):
        return {"code": self.code, "message": self.message}

    def __repr__(self):
        return "Reason(%r, %r)" % (self.code, self.message)

    def __eq__(self, other):
        if not isinstance(other, Reason):
            return NotImplemented
        return self.code == other.code and self.message == other.message

    def __hash__(self):
        return hash((self.code, self.message))


class TransferRoute:
    """Which route one device buffer is on, as the native layer reported it.

    Every field is a stable name from
    `src/mojotrees/unified_memory_policy.mojo`, and nothing here decides or
    infers: whether a buffer may skip a copy is a statement about pointer
    lifetime and device queue ordering, which Python cannot see, so Python
    is not the layer that answers it.

    - `role`: which buffer, e.g. `bins`, `grad`, `hist_out`.
    - `requested`: what `MOJOTREES_GPU_TRANSFER` asked for.
    - `selected`: what this role actually got. Different from `requested`
      means the request was refused for this role and the default was used.
    - `reason`: `eligible` when the request was honored, otherwise why not.
    - `evidence`: the rung of the evidence ladder the selected route sits
      on. `none` for every route in this repository today.
    - `retire_on`: `copy` when the host may refill the buffer once the
      upload retires, `kernel` when it must wait for every kernel that read
      it. This is the field that changes what a caller must *do*, not only
      what it allocates.

    A `selected` of `copy_staged` on every role is the shipped state and the
    only state any measurement in this repository was taken under.
    """

    __slots__ = (
        "role",
        "requested",
        "selected",
        "reason",
        "evidence",
        "retire_on",
    )

    def __init__(
        self,
        role,
        requested="",
        selected="",
        reason="",
        evidence="",
        retire_on="",
    ):
        self.role = role
        self.requested = requested
        self.selected = selected
        self.reason = reason
        self.evidence = evidence
        self.retire_on = retire_on

    @property
    def honored(self):
        """Whether this role got the route that was asked for."""
        return self.reason == "eligible"

    def to_dict(self):
        return {
            "role": self.role,
            "requested": self.requested,
            "selected": self.selected,
            "reason": self.reason,
            "evidence": self.evidence,
            "retire_on": self.retire_on,
        }

    def __repr__(self):
        return "TransferRoute(%r, selected=%r)" % (self.role, self.selected)

    def __eq__(self, other):
        if not isinstance(other, TransferRoute):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(tuple(sorted(self.to_dict().items())))


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------


class Workload:
    """The shape and the declared features of one training run.

    Plain metadata, and nothing else: every field is something read off the
    call or off `X` and `y`, and no field is a judgment about what the GPU
    can do with it. The native policy makes those judgments from these
    numbers.

    `n_outputs` is what the native contract takes: 1 for single-output
    training and for binary classification, the class count beyond that,
    because that is how many trees a boosting round grows.

    The objective arrives in two forms, and only one of them gates:

    - `objective_code` is the native objective code from
      `src/mojotrees/boosting.mojo` (and `LAMBDARANK` from
      `ranking.mojo`). It is what the native policy reads, and the
      estimators already have it: `_objective_code()` in
      `python/mojotrees/__init__.py` returns exactly this. `None` means
      undeclared, in which case the native layer skips the objective gate
      and marks the decision incomplete rather than assuming one.
    - `objective` is the public *name*, and it is display only. Names map
      to codes through `objective_from_name` in
      `src/mojotrees/params.mojo`, which is a Mojo function; when the
      build exposes it (see `_code_for_objective_name`), a name-only
      caller gets its code resolved and gated too, and when it does not,
      the name is still shown and the gate is still skipped. Passing an
      int for `objective` is accepted and read as a code.

    `max_bin` is the estimator's parameter of that name, and it is the bin
    count the kernels see. `None` means the caller did not declare one, in
    which case the native layer skips the gates that need it and marks the
    decision incomplete rather than assuming a value.
    """

    def __init__(
        self,
        n_rows,
        n_features,
        objective=None,
        objective_code=None,
        n_classes=1,
        max_bin=None,
        sparse=False,
        categorical=False,
        has_missing=False,
        has_eval_set=False,
    ):
        self.n_rows = int(n_rows)
        self.n_features = int(n_features)
        if isinstance(objective, int) and not isinstance(objective, bool):
            if objective_code is None:
                objective_code = objective
            objective = None
        self.objective = None if objective is None else str(objective)
        if objective_code is None and self.objective is not None:
            objective_code = _code_for_objective_name(self.objective)
        self.objective_code = (
            None if objective_code is None else int(objective_code)
        )
        self.n_classes = int(n_classes)
        self.max_bin = None if max_bin is None else int(max_bin)
        self.sparse = bool(sparse)
        self.categorical = bool(categorical)
        self.has_missing = bool(has_missing)
        self.has_eval_set = bool(has_eval_set)
        if self.n_rows < 1 or self.n_features < 1:
            raise ValueError(
                "a workload needs at least one row and one feature; got "
                "n_rows=%d, n_features=%d" % (self.n_rows, self.n_features)
            )
        if self.n_classes < 1:
            raise ValueError(
                "n_classes must be at least 1; got %d" % self.n_classes
            )

    @property
    def n_outputs(self):
        """Trees grown per boosting round: 1 for single-output training
        and for binary classification, the class count beyond that."""
        return self.n_classes if self.n_classes > 2 else 1

    @property
    def cells(self):
        """`n_rows * n_features`, the unit the native crossover rules and
        `MOJOTREES_AUTO_MIN_CELLS` are written in."""
        return self.n_rows * self.n_features

    @classmethod
    def from_data(cls, X, y=None, **kwargs):
        """A `Workload` read off a dataset.

        `X` supplies rows, features, and sparseness. Anything with a
        two-element `shape` works, which covers numpy arrays, pandas and
        polars frames, and scipy sparse matrices; a list of rows works
        too. `y` supplies the class count when `n_classes` was not passed
        and the labels are countable. Everything else comes from
        `kwargs`, which take precedence over what was inferred.
        """
        n_rows, n_features = _shape_of(X)
        inferred = {
            "n_rows": n_rows,
            "n_features": n_features,
            "sparse": _is_sparse(X),
        }
        if y is not None and "n_classes" not in kwargs:
            n_classes = _count_classes(y)
            if n_classes is not None:
                inferred["n_classes"] = n_classes
        inferred.update(kwargs)
        return cls(**inferred)

    def describe(self):
        """One line of shape and features, for the explanation."""
        parts = [
            "%s rows x %s features"
            % (_fmt_int(self.n_rows), _fmt_int(self.n_features))
        ]
        if self.objective:
            parts.append("objective '%s'" % self.objective)
        elif self.objective_code is not None:
            parts.append("objective code %d" % self.objective_code)
        else:
            parts.append("objective undeclared")
        if self.n_classes > 1:
            parts.append("%d classes" % self.n_classes)
        parts.append("%d output(s) per round" % self.n_outputs)
        if self.max_bin is None:
            parts.append("max_bin undeclared")
        else:
            parts.append("max_bin=%d" % self.max_bin)
        parts.append("sparse" if self.sparse else "dense")
        if self.categorical:
            parts.append("categorical features")
        if self.has_missing:
            parts.append("missing values")
        if self.has_eval_set:
            parts.append("with eval_set")
        return ", ".join(parts)

    def to_dict(self):
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "cells": self.cells,
            "objective": self.objective,
            "objective_code": self.objective_code,
            "n_classes": self.n_classes,
            "n_outputs": self.n_outputs,
            "max_bin": self.max_bin,
            "sparse": self.sparse,
            "categorical": self.categorical,
            "has_missing": self.has_missing,
            "has_eval_set": self.has_eval_set,
        }

    def __repr__(self):
        return "Workload(n_rows=%d, n_features=%d, objective=%r)" % (
            self.n_rows,
            self.n_features,
            self.objective,
        )


def _shape_of(X):
    """(n_rows, n_features) for a dataset-like object."""
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if shape is not None:
        raise ValueError(
            "X must be two dimensional; got shape %r" % (tuple(shape),)
        )
    try:
        n_rows = len(X)
        first = X[0]
        n_features = len(first)
    except Exception:
        raise ValueError(
            "cannot read a shape from %r; pass a Workload instead"
            % type(X).__name__
        ) from None
    return int(n_rows), int(n_features)


def _is_sparse(X):
    """Whether `X` is a scipy sparse matrix or array, by duck typing, so
    nothing here imports scipy."""
    if hasattr(X, "tocsr") and hasattr(X, "nnz"):
        return True
    return getattr(X, "format", None) in ("csr", "csc", "coo")


def _count_classes(y):
    """Distinct label count, or None when `y` is not countable cheaply."""
    try:
        unique = getattr(y, "unique", None)
        if callable(unique):
            return int(len(unique()))
        return int(len({_hashable(v) for v in y}))
    except Exception:
        return None


def _hashable(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def _code_for_objective_name(name):
    """The native objective code for a public objective name, or None.

    Asks the extension, because the name-to-code mapping is
    `objective_code_from_name` in src/mojotrees/objective_registry.mojo
    and belongs there. Returns None for every failure, including a build
    with no name resolver bound and a name the registry refuses: an
    unresolved name is reported as an undeclared objective, which skips
    the gate, and is never guessed at here.

    `objective_code_of_name` is asked for by that name, and not as
    `objective_code`, because those are two different questions that two
    earlier plans gave one name. `_mojotrees.objective_code` takes a
    *model handle* and answers what a fitted model was trained for
    (`python/mojotrees/inspection.py` calls it that way); the resolver
    here takes a name and answers what a caller asked for. Binding both
    under one name would silently disable whichever caller lost. The
    older name is still tried second, so a build that predates the split
    keeps whatever it had.

    Do not replace this with a dict. A name table in Python is exactly the
    second implementation this module exists to not have.
    """
    try:
        ext = _extension()
    except NativePolicyUnavailable:
        return None
    resolve = getattr(ext, "objective_code_of_name", None)
    if resolve is None:
        resolve = getattr(ext, "objective_code", None)
    if resolve is None:
        return None
    try:
        return int(resolve(str(name)))
    except Exception:
        return None


# ---------------------------------------------------------------------
# The native boundary
# ---------------------------------------------------------------------

#: Values `DeviceReport.contract` takes. See the module docstring.
CONTRACT_FULL = "full"
CONTRACT_NARROW = "narrow"

#: Wire sentinels for "the caller did not declare this". The boundary
#: carries plain ints, so an undeclared value needs a value; the binding
#: normalizes any negative `n_bins` to the native `BINS_UNSPECIFIED` (which
#: is 0 natively, so a caller must not send 0 to mean undeclared) and any
#: negative `objective` to `OBJECTIVE_UNSPECIFIED`.
#:
#: That second folding takes in the multiclass marker (-1) deliberately.
#: `MULTICLASS` is not one of the built-in single-output objectives
#: `device_policy.is_builtin_objective` recognizes, so sending it would
#: gate the run as "not a built-in objective", which is the wrong reason:
#: multiclass is a tree count, and `n_outputs` already carries it.
#:
#: These are transport, not policy: no rule here reads them, and the
#: native layer decides what an undeclared field means.
_BINS_UNSPECIFIED = -1
_OBJECTIVE_UNSPECIFIED = -2


def _extension():
    """The compiled extension, or a `NativePolicyUnavailable`."""
    try:
        try:
            from . import _mojotrees
        except ImportError:
            import mojotrees._mojotrees as _mojotrees
        return _mojotrees
    except Exception as exc:
        raise NativePolicyUnavailable(
            "the compiled mojotrees extension could not be loaded (%s), so "
            "no device decision can be made; this module does not have a "
            "Python fallback policy, by design" % exc
        ) from None


def _parse_decision(text):
    """The native `key=value` lines as a dict.

    Repeated keys are lists in order, which is how `block`, `warning`, and
    `transfer` arrive; every other key appears once. Values are left as
    strings here and coerced by the accessors that know their types, so an
    unrecognized key from a newer native layer survives into `to_dict()`
    instead of being dropped.
    """
    fields = {}
    blocks = []
    warnings = []
    transfers = []
    for line in text.splitlines():
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key == "block":
            blocks.append(_split_reason(value))
        elif key == "warning":
            warnings.append(_split_reason(value))
        elif key == "transfer":
            transfers.append(_split_transfer(value))
        else:
            fields[key] = value
    fields["_blocks"] = blocks
    fields["_warnings"] = warnings
    fields["_transfers"] = transfers
    return fields


def _split_reason(value):
    """`<code>:<name>:<message>` as a `Reason`, keeping the stable name as
    the code and leaving any colon in the message alone."""
    parts = value.split(":", 2)
    if len(parts) == 3:
        return Reason(parts[1], parts[2])
    return Reason("unknown", value)


def _split_transfer(value):
    """`<role>:<requested>:<selected>:<reason>:<evidence>:<retire_on>` as a
    `TransferRoute`.

    A short line with no free text in it, so unlike `_split_reason` there is
    nothing to keep un-split. A line with the wrong field count is kept as a
    role-only entry rather than dropped: a newer native layer that adds a
    field should degrade the report, not empty it.
    """
    parts = value.split(":")
    if len(parts) == 6:
        return TransferRoute(*parts)
    return TransferRoute(parts[0] if parts else "unknown")


class _FullNativePolicy:
    """The intended path: the whole contract crosses in one call.

    The binding is `decide_device_report` in
    `src/mojotrees/device_policy.mojo`, exposed as `decide_device`. It never
    raises for a workload it refuses; the refusal is `blocked=true` in the
    returned lines, and `select_device` is what turns that into an exception.
    It does raise for a device name outside the vocabulary, a shape with no
    rows or features, and an unparsable `MOJOTREES_GPU_TRANSFER`, all of
    which are caller or operator errors rather than policy outcomes, and all
    of which reach the user as-is rather than being reinterpreted here.
    """

    contract = CONTRACT_FULL

    def __init__(self, decide):
        self._decide = decide

    def decide(self, requested, workload):
        # The device name is positional and the workload is one mapping,
        # which is the shape `_parse_params` in bindings/_mojotrees.mojo
        # already reads and the shape the binding registers. A workload has
        # a dozen fields; sending them positionally would fix their order
        # in two languages at once and would bet on an argument count no
        # entry point in the module has tried.
        #
        # Inside the mapping: flags cross as 0/1 ints and the bin count and
        # the objective code as their `_UNSPECIFIED` sentinels when
        # undeclared, so the boundary carries no Python bool conversion and
        # no optional. That is the convention the rest of the bindings
        # already use (see `goss` in bindings/_mojotrees.mojo).
        text = self._decide(
            requested,
            {
                "n_rows": int(workload.n_rows),
                "n_features": int(workload.n_features),
                "n_outputs": int(workload.n_outputs),
                "n_bins": _BINS_UNSPECIFIED
                if workload.max_bin is None
                else int(workload.max_bin),
                "objective": _OBJECTIVE_UNSPECIFIED
                if workload.objective_code is None
                else int(workload.objective_code),
                "sparse": 1 if workload.sparse else 0,
                "categorical": 1 if workload.categorical else 0,
                "has_missing": 1 if workload.has_missing else 0,
                "uses_validation": 1 if workload.has_eval_set else 0,
            },
        )
        return _parse_decision(str(text))


class _NarrowNativePolicy:
    """Temporary compatibility path. Delete this class whole once
    `decide_device` is bound.

    It is not a second policy. `resolve_device` runs the same native
    engine through `src/mojotrees/device.mojo`, so the backend it returns
    is the native answer; what it cannot carry is the objective, the bin
    count, the input flags, and the report. This class fills in only what
    the call itself proves: which backend came back, or what the native
    refusal said.

    Nothing here decides, thresholds, estimates, or infers. If you find
    yourself wanting to add such a thing to this class, it belongs in
    src/mojotrees/device_policy.mojo instead.
    """

    contract = CONTRACT_NARROW

    def __init__(self, resolve):
        self._resolve = resolve

    def decide(self, requested, workload):
        skipped = (
            "the narrow native entry point carries only the shape, so the "
            "objective, bin-count, input-flag, and memory gates were not "
            "evaluated; bind decide_device for the full contract"
        )
        fields = {
            "requested": requested,
            "policy_version": "",
            "evidence_id": "",
            "validated": "false",
            "_blocks": [],
            "_warnings": [Reason("narrow-contract", skipped)],
        }
        try:
            resolved = str(
                self._resolve(
                    requested,
                    int(workload.n_rows),
                    int(workload.n_features),
                    int(workload.n_outputs),
                )
            )
        except Exception as exc:
            fields["selected"] = "none"
            fields["blocked"] = "true"
            fields["decision"] = "gpu-refused"
            fields["message"] = str(exc)
            fields["_blocks"] = [Reason("native-refusal", str(exc))]
            return fields
        fields["selected"] = resolved
        fields["blocked"] = "false"
        fields["decision"] = "native-resolved"
        fields["message"] = "the native device policy resolved %r to %s" % (
            requested,
            resolved.upper(),
        )
        return fields


def _policy():
    """The narrowest native seam available, most capable first."""
    ext = _extension()
    decide = getattr(ext, "decide_device", None)
    if decide is not None:
        return _FullNativePolicy(decide)
    resolve = getattr(ext, "resolve_device", None)
    if resolve is None:
        raise NativePolicyUnavailable(
            "the compiled extension exposes neither decide_device nor "
            "resolve_device, so it is too old to answer a device question"
        )
    return _NarrowNativePolicy(resolve)


def native_contract():
    """Which native seam this build has, `"full"` or `"narrow"`.

    Useful in a bug report, and useful in CI as the assertion that the
    binding wiring actually landed."""
    return _policy().contract


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


class DeviceReport:
    """The native decision, rendered.

    Holds no policy. Every field is read out of what the native layer
    returned, and the only work done here is turning it into prose, into
    JSON, or into an exception.

    `resolved` is `"cpu"` or `"gpu"` when a backend was chosen and None
    when the request cannot run, in which case `would_raise` is True and
    `error` holds the message `select_device` raises.
    """

    def __init__(self, requested, workload, fields, contract):
        self.requested = requested
        self.workload = workload
        self.contract = contract
        #: Every key the native layer sent, as strings, unfiltered.
        self.native = {
            k: v for k, v in fields.items() if not k.startswith("_")
        }
        self.reasons = tuple(fields.get("_blocks", ()))
        self.warnings = tuple(fields.get("_warnings", ()))
        #: One `TransferRoute` per device buffer role, in native role order.
        #: Empty on the `"narrow"` contract, which carries no such key.
        self.transfer_routes = tuple(fields.get("_transfers", ()))
        selected = fields.get("selected", "none")
        self.would_raise = fields.get("blocked", "false") == "true"
        self.resolved = None if self.would_raise else selected
        self.error = fields.get("message") if self.would_raise else None
        self.decision = fields.get("decision", "")
        self.policy_version = fields.get("policy_version", "")
        self.evidence_id = fields.get("evidence_id", "")
        self.message = fields.get("message", "")

    @property
    def validated(self):
        """Whether a GPU choice rests on benchmark-derived evidence. False
        for every CPU choice, and False for a GPU chosen through
        `MOJOTREES_AUTO_MIN_CELLS` or requested explicitly."""
        return self.native.get("validated") == "true"

    @property
    def complete(self):
        """Whether every native gate was evaluated. False for a `"narrow"`
        contract, and False when the caller left the objective or the bin
        count undeclared."""
        if self.contract != CONTRACT_FULL:
            return False
        return self.native.get("memory_estimate_complete") == "true"

    def raise_if_unsupported(self):
        """Raise `DeviceUnavailableError` when the request cannot run,
        and return the report otherwise. This is what turns an
        `explain_device_choice` report back into `select_device`
        behavior."""
        if self.would_raise:
            raise DeviceUnavailableError(self.error, self)
        return self

    @property
    def explanation(self):
        """The report as prose, one section per thing that mattered."""
        lines = []
        if self.would_raise:
            lines.append(
                "device=%r cannot run this workload." % self.requested
            )
        else:
            lines.append(
                "device=%r resolved to %s."
                % (self.requested, self.resolved.upper())
            )
        lines.append("")
        lines.append("Workload    %s" % self.workload.describe())
        lines.append("Device      %s" % self._device_line())
        lines.append("Memory      %s" % self._memory_line())
        lines.append("Transfer    %s" % self._transfer_line())
        lines.append("Startup     %s" % self._session_line())
        lines.append("Policy      %s" % self._policy_line())
        if self.message:
            lines.append("")
            lines.append("Why")
            lines.append("  [%s] %s" % (self.decision, self.message))
        if self.reasons:
            lines.append("")
            lines.append("Blocked by")
            for reason in self.reasons:
                lines.append("  [%s] %s" % (reason.code, reason.message))
        if self.warnings:
            lines.append("")
            lines.append("Warnings")
            for warning in self.warnings:
                lines.append("  [%s] %s" % (warning.code, warning.message))
        return "\n".join(lines)

    def _device_line(self):
        available = self.native.get("gpu_available")
        if available is None:
            # The narrow contract carries no capability keys. Absent is
            # not the same as absent-hardware, and saying "no accelerator"
            # here would be this layer inventing an answer.
            return "not reported by this build's native contract"
        if available != "true":
            return "no accelerator"
        parts = ["accelerator available"]
        api = self.native.get("api")
        if api:
            parts.append("api %s" % api)
        generation = self.native.get("apple_generation")
        if generation and generation != "unknown":
            parts.append("apple %s" % generation)
        source = self.native.get("profile_source")
        if source:
            parts.append("capabilities %s" % source)
        return ", ".join(parts)

    def _memory_line(self):
        device = self.native.get("memory_device_bytes")
        if device is None:
            return "not estimated"
        text = "%s device, %s including the tiled partial-histogram budget" % (
            _fmt_bytes(device),
            _fmt_bytes(self.native.get("memory_upper_bound_bytes", device)),
        )
        if self.native.get("memory_estimate_complete") != "true":
            text += " (partial: no bin count was declared)"
        else:
            text += " (estimate)"
        return text

    def _transfer_line(self):
        """Which transfer routes the device buffers are on.

        Reports the shipped all-default case in one phrase and names the
        exceptions individually, because a plan where seven roles are staged
        and one is not is exactly the state a summary would hide. Says
        nothing about copies avoided or memory saved: whether two allocations
        are the same physical pages is not visible from inside the process,
        and `src/mojotrees/unified_memory_policy.mojo` is where that is
        argued out.
        """
        if not self.transfer_routes:
            if self.contract != CONTRACT_FULL:
                return "not reported by this build's native contract"
            return "not reported"
        if self.native.get("transfer_all_default") == "true":
            text = "copy_staged for all %d buffer roles (the shipped route)" % len(
                self.transfer_routes
            )
        else:
            changed = [
                "%s=%s" % (r.role, r.selected)
                for r in self.transfer_routes
                if r.selected != "copy_staged"
            ]
            text = "copy_staged except %s" % ", ".join(changed)
        refused = [r.role for r in self.transfer_routes if not r.honored]
        if refused:
            text += "; request refused for %s" % ", ".join(refused)
        if self.native.get("transfer_ack_unproven") == "true":
            text += (
                "; running on MOJOTREES_GPU_TRANSFER_UNPROVEN=1, so no"
                " evidence is behind it"
            )
        return text

    def _session_line(self):
        """How much of the one-time startup cost this process has paid.

        The number a first-fit timing has to be read against. It is reported
        and never acted on: nothing about a warm session selects a backend,
        for the same reason nothing about a cell count does without a
        measurement behind it.
        """
        open_ = self.native.get("session_context_open")
        if open_ is None:
            if self.contract != CONTRACT_FULL:
                return "not reported by this build's native contract"
            return "not reported"
        if open_ != "true":
            text = "cold: no device context open in this process"
        else:
            text = "warm: a device context is already open"
        if self.native.get("session_kernels_ready") == "true":
            text += ", kernels already created"
        else:
            text += ", kernels not yet created"
        level = self.native.get("session_warmup_level")
        if level:
            text += " (MOJOTREES_GPU_WARMUP=%s)" % level
        return text

    def _policy_line(self):
        version = self.policy_version or "unreported"
        evidence = self.evidence_id or "unreported"
        text = "version %s, evidence %s, contract %s" % (
            version,
            evidence,
            self.contract,
        )
        if not self.complete:
            text += ", incomplete"
        return text

    def to_dict(self):
        """The report as JSON-serializable data.

        `native` is the decision exactly as the policy sent it, so a
        consumer that wants a field this class does not surface can read it
        without waiting for this file to grow an accessor.
        """
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "would_raise": self.would_raise,
            "error": self.error,
            "decision": self.decision,
            "message": self.message,
            "validated": self.validated,
            "complete": self.complete,
            "contract": self.contract,
            "policy_version": self.policy_version,
            "evidence_id": self.evidence_id,
            "reasons": [r.to_dict() for r in self.reasons],
            "warnings": [w.to_dict() for w in self.warnings],
            "transfer_routes": [t.to_dict() for t in self.transfer_routes],
            "workload": self.workload.to_dict(),
            "native": dict(self.native),
        }

    def to_json(self, **kwargs):
        """`to_dict()` as a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    def __str__(self):
        return self.explanation

    def __repr__(self):
        return "DeviceReport(requested=%r, resolved=%r)" % (
            self.requested,
            self.resolved,
        )


def _fmt_int(n):
    return "{:,}".format(int(n))


def _fmt_bytes(n):
    """Bytes as the largest binary unit that keeps the number readable."""
    try:
        value = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return "%d B" % int(value)
            return "%.1f %s" % (value, unit)
        value /= 1024.0


def _normalize_device(device):
    """The requested device, lowercased as LightGBM treats `device_type`.
    Raises `ValueError` for anything outside the vocabulary, which is what
    the estimators do.

    The one check this module still makes on its own, and it is a spelling
    check rather than a policy one: `parse_device` in
    src/mojotrees/device_policy.mojo rejects the same set, and doing it
    here only means the caller gets a `ValueError` naming the alternatives
    instead of a `RuntimeError` from across the boundary.
    """
    if device is None:
        return "cpu"
    if not isinstance(device, str) or device.lower() not in DEVICES:
        raise ValueError(
            "unknown device %r; expected one of %s"
            % (device, ", ".join(DEVICES))
        )
    return device.lower()


def _as_workload(workload, kwargs):
    if isinstance(workload, Workload):
        if kwargs:
            raise TypeError(
                "workload keyword(s) %s cannot be combined with a Workload"
                % ", ".join(sorted(kwargs))
            )
        return workload
    raise TypeError("expected a Workload; got %r" % type(workload).__name__)


def _report(device, workload):
    # Spelling first, so a typo is a ValueError naming the alternatives
    # even on a machine where the extension is missing entirely.
    requested = _normalize_device(device)
    policy = _policy()
    fields = policy.decide(requested, workload)
    return DeviceReport(requested, workload, fields, policy.contract)


def select_device(device, workload):
    """Resolve `device` for `workload` and report why.

    Returns a `DeviceReport` whose `resolved` is "cpu" or "gpu". Raises
    `ValueError` for a device name outside `DEVICES`,
    `DeviceUnavailableError` when `device="gpu"` cannot run, and
    `NativePolicyUnavailable` when the compiled extension is missing.

    An explicit GPU request either runs on the GPU or fails, never quietly
    on the CPU. That rule is enforced natively; this function only turns
    the native refusal into the Python exception.
    """
    return _report(device, _as_workload(workload, {})).raise_if_unsupported()


def explain_device_choice(X, y=None, device="auto", **workload_kwargs):
    """Explain what `device` would do with this data, without training.

    `X` is a dataset (anything with a two dimensional `shape`, or a
    sequence of rows) or a ready-made `Workload`; `y` supplies the class
    count when it can be counted. Keyword arguments matching `Workload`
    fields override what was inferred, which is how a caller declares
    `objective="poisson"`, `max_bin=63`, or `has_eval_set=True`.

    Unlike `select_device`, this never raises for an unsupported GPU
    request. The returned report carries `would_raise=True` and the
    message in `error`, so "what would happen if I asked for the GPU" is
    answerable without try/except. Call `report.raise_if_unsupported()`
    to get the exception instead.

        report = explain_device_choice(X, y, device="gpu")
        print(report)                 # the prose explanation
        report.to_dict()["resolved"]  # the structured answer
    """
    if isinstance(X, Workload):
        workload = _as_workload(X, workload_kwargs)
    else:
        workload = Workload.from_data(X, y, **workload_kwargs)
    return _report(device, workload)


class PredictSupport:
    """Whether the GPU prediction path covers a request of one shape.

    `supported` is the answer; `block_code` and `reason` are
    device_policy.mojo's stable refusal vocabulary (0 and "" when
    supported), and `message` the prose an explicit `device="gpu"` predict
    would raise with. This is the question form of that refusal, read from
    the same native record (`gpu_predict_capability`), so an estimator can
    branch on the code without asking for the device.
    """

    __slots__ = ("supported", "block_code", "reason", "message", "shape")

    def __init__(self, supported, block_code, reason, message, shape):
        self.supported = bool(supported)
        self.block_code = int(block_code)
        self.reason = str(reason)
        self.message = str(message)
        self.shape = dict(shape)

    def to_dict(self):
        return {
            "supported": self.supported,
            "block_code": self.block_code,
            "reason": self.reason,
            "message": self.message,
            "shape": dict(self.shape),
        }

    def raise_if_unsupported(self):
        if not self.supported:
            raise DeviceUnavailableError(self.message)
        return self

    def __repr__(self):
        if self.supported:
            return "PredictSupport(supported=True)"
        return "PredictSupport(supported=False, reason=%r)" % self.reason


def explain_predict_device(X, n_outputs=1, n_bins=0, sparse=None):
    """Whether predicting on `X` could run on the GPU, without asking it to.

    `X` is the matrix (anything with a two dimensional `shape`, or a
    sequence of rows); `n_outputs` is 1 for a single-output model or the
    class count for a multiclass one; `n_bins` the binner's reserved bin
    count when known (0 otherwise); `sparse` overrides the duck-typed
    sparsity check. Returns a `PredictSupport`; never raises for an
    uncovered shape (call `raise_if_unsupported()` for the exception).
    """
    n_rows, n_features = _shape_of(X)
    is_sparse = _is_sparse(X) if sparse is None else bool(sparse)
    ext = _extension()
    shape = {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "n_outputs": int(n_outputs),
        "n_bins": int(n_bins),
        "sparse": 1 if is_sparse else 0,
    }
    supported, block_code, reason, message = ext.gpu_predict_capability(shape)
    return PredictSupport(supported, block_code, reason, message, shape)

