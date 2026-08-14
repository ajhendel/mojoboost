"""Explainable device selection for the Python estimators.

`device="auto"` has to answer one question, "CPU or GPU for this run",
and it has to be able to say why. This module is that policy, kept apart
from the estimators so the answer can be inspected, tested with injected
capabilities, and read by a user who is deciding whether an accelerator is
worth using at all.

    from mojoboost.device_selection import explain_device_choice

    print(explain_device_choice(X, y, device="auto"))

Three requested values, the same vocabulary the Mojo layer uses (see
src/mojoboost/device.mojo):

- `"cpu"`, the default and the dependable path. Always resolves to itself.
- `"gpu"`, an explicit request. It runs or it raises. There is no silent
  fallback to the CPU, because a fallback turns "my GPU run" into "a CPU
  run that took the same wall clock and I never knew".
- `"auto"`, the policy this module implements. It picks the GPU only when
  the GPU path covers the workload *and* a validated, benchmark-derived
  rule says the GPU is the faster choice for that shape on that backend.
  With no such rule it picks the CPU and says so.

What "validated" means
----------------------
`CROSSOVER_RULES` is a versioned table of measured crossovers. It is
**empty**. Nothing in this repository has measured a workload size where
GPU training beats CPU training: the one end-to-end measurement that
exists (Apple M4, `bench/bench_train_gpu.mojo`) came out slower than the
CPU trainer, and no NVIDIA or AMD device has ever run this code at all
(docs/GPU_VALIDATION.md). A crossover threshold invented here would be a
performance claim with no evidence under it, so the table ships empty and
`auto` conservatively resolves to the CPU everywhere. Adding a rule is a
benchmarking result, not a code change, and the `evidence` field is where
that result gets cited.

`MOJOBOOST_AUTO_MIN_CELLS` is the escape hatch for running that benchmark:
an integer cell count (`n_rows * n_features`) at or above which `auto`
selects the GPU, `0` meaning "whenever the GPU path covers the workload",
unset or negative meaning the heuristic is off. It is device independent
and mirrors `env_auto_min_cells` in src/mojoboost/device.mojo exactly, so
the two layers cannot disagree about what a given environment does. A run
that reaches the GPU through it is reported as unvalidated.

Hard blocks and soft uncertainty
--------------------------------
Two different things keep a workload off the GPU, and conflating them
would either refuse runs that work or promise runs that do not.

A **hard block** is something that will actually fail. No accelerator,
sparse input, a custom objective callable, an `eval_set` (validation is
scored on the CPU), lambdarank, a row count past the Int32 ceiling the
kernels index with, a bin count outside [2, 256], an estimate that does
not fit in device memory. Each of these is a rule the estimators or the
kernels already enforce; explicit `"gpu"` raises on them and `"auto"`
takes the CPU.

**Soft uncertainty** is a workload nobody has measured or documented as
covered, most often an objective outside the set device.mojo names. It
never blocks an explicit `"gpu"` request, because refusing a run the
native layer would have accepted is its own kind of lie. It does keep
`"auto"` on the CPU, since choosing the GPU on an uncharacterized path is
exactly the guess `auto` is supposed to not make.

Reports, not booleans
---------------------
`select_device` returns a `DeviceReport`: the resolution, the ordered
reasons behind it, the capabilities that were detected or injected, the
workload as it was understood, the memory estimate and its components,
the rules table version, and the rule that matched if one did.
`report.explanation` renders the same content as prose, and `str(report)`
is that explanation. `report.to_dict()` is JSON-serializable, which is
what a support ticket or a CI log wants.

`explain_device_choice` is the same policy in a form that never raises:
for a request that would fail it sets `would_raise` and `error` instead,
so a user can ask "what would `device='gpu'` do here" without handling an
exception. `report.raise_if_unsupported()` turns it back into the raise.

Integration note
----------------
The estimators pass an already-resolved concrete device name to the
native layer, which resolves it a second time (see `_parse_device` in
bindings/_mojoboost.mojo). That is what makes this module safe: a
`"gpu"` chosen here is requested as `"gpu"` natively and therefore runs
or raises on native terms too. Passing `"auto"` through to the native
layer instead would discard this policy, because the native `auto` gate
is only `MOJOBOOST_AUTO_MIN_CELLS`.
"""

import json
import os
import platform

__all__ = [
    "CROSSOVER_RULES",
    "DEVICES",
    "GPU_OBJECTIVES",
    "MAX_GPU_BINS",
    "MAX_GPU_ROWS",
    "MIN_GPU_BINS",
    "RULES_VERSION",
    "Capabilities",
    "CrossoverRule",
    "DeviceReport",
    "DeviceUnavailableError",
    "MemoryEstimate",
    "Reason",
    "Workload",
    "detect_capabilities",
    "estimate_gpu_memory",
    "explain_device_choice",
    "select_device",
]

#: The public device vocabulary, as in `mojoboost._DEVICES`.
DEVICES = ("cpu", "gpu", "auto")

#: Rows are indexed as Int32 by the histogram and partition kernels
#: (`MAX_ROWS` in src/mojoboost/histogram_gpu.mojo), which is also where
#: the fixed-point accumulator stops being exact.
MAX_GPU_ROWS = 2**31 - 1

#: Bin counts the binner accepts (src/mojoboost/binning.mojo) and the
#: kernels reserve shared memory for (`MAX_BINS`).
MIN_GPU_BINS = 2
MAX_GPU_BINS = 256

#: Objectives src/mojoboost/device.mojo documents the GPU path as
#: covering, under the estimator's spelling of each name. Anything outside
#: this set is soft uncertainty, not a hard block: the native layer may
#: well accept it, but nothing here documents that it does, so `auto`
#: will not choose the GPU for it.
GPU_OBJECTIVES = frozenset(
    {
        "regression",
        "mae",
        "regression_l1",
        "huber",
        "quantile",
        "poisson",
        "binary",
        "multiclass",
    }
)

#: Objectives whose Python training path is CPU-only, each one a rule the
#: estimators already enforce (see `_fit` in python/mojoboost/__init__.py).
CPU_ONLY_OBJECTIVES = frozenset({"lambdarank"})

# Reason codes. Stable strings, because they end up in `to_dict()` output
# that something else may match on.
EXPLICIT_CPU = "explicit-cpu"
EXPLICIT_GPU = "explicit-gpu"
NO_ACCELERATOR = "no-accelerator"
GPU_DISABLED_ENV = "gpu-disabled-env"
UNSUPPORTED_FEATURE = "unsupported-feature"
UNSUPPORTED_OBJECTIVE = "unsupported-objective"
WORKLOAD_LIMIT = "workload-limit"
INSUFFICIENT_MEMORY = "insufficient-memory"
UNVALIDATED_PATH = "unvalidated-path"
NO_VALIDATED_RULE = "no-validated-rule"
RULE_MATCHED = "rule-matched"
BELOW_RULE_THRESHOLD = "below-rule-threshold"
ENV_THRESHOLD = "env-threshold"
BELOW_ENV_THRESHOLD = "below-env-threshold"


class DeviceUnavailableError(RuntimeError):
    """An explicit `device="gpu"` cannot run this workload.

    A subclass of `RuntimeError` so it is caught by code written against
    what the estimators raise today.
    """

    def __init__(self, message, report=None):
        RuntimeError.__init__(self, message)
        #: The `DeviceReport` behind the refusal, when one was built.
        self.report = report


class Reason:
    """One ordered step of the decision, a stable `code` and prose."""

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


class Capabilities:
    """What the running build and machine can do, as the policy sees it.

    Every field is data, never a probe, so a test injects a machine it
    does not have and the policy cannot tell the difference. Use
    `detect_capabilities()` to fill one in from the real environment.

    - `gpu_available`: training can run on an accelerator, the same
      question `mojoboost.gpu_available()` answers.
    - `backend`: "metal", "cuda", "hip", or None when unknown. Detection
      is a heuristic and says so through `backend_source`; the policy
      only uses it to scope crossover rules.
    - `chip`: the device or host chip name when one could be read, such
      as "Apple M4". None when unknown.
    - `device_memory_bytes`: the memory budget a training run may use, or
      None when unknown, in which case memory never blocks anything.
    - `unified_memory`: True when `device_memory_bytes` is host memory
      shared with the GPU, as on Apple silicon.
    - `gpu_objectives`: objective names the GPU path is documented to
      cover.
    - `supports_multiclass`: whether the GPU path covers more than two
      classes, mirroring `gpu_supports` in src/mojoboost/device.mojo.
    - `supports_sparse`, `supports_custom_objective`, `supports_eval_set`:
      the three Python-level gates the estimators enforce today. All
      False, and each one is a hard block.
    - `auto_min_cells`: the `MOJOBOOST_AUTO_MIN_CELLS` value in effect,
      or None when the heuristic is off.
    - `disabled_by_env`: `MOJOBOOST_DISABLE_GPU=1` was set.
    - `build_has_accelerator`: whether the build itself was compiled with
      an accelerator present, when that is knowable separately from
      `gpu_available`. Availability is a build property in Mojo, so a
      redistributed wheel can claim a device the running machine lacks.
    - `source`: where these values came from, for the report.
    """

    def __init__(
        self,
        gpu_available=False,
        backend=None,
        chip=None,
        device_memory_bytes=None,
        unified_memory=False,
        gpu_objectives=GPU_OBJECTIVES,
        supports_multiclass=True,
        supports_sparse=False,
        supports_custom_objective=False,
        supports_eval_set=False,
        max_rows=MAX_GPU_ROWS,
        min_bins=MIN_GPU_BINS,
        max_bins=MAX_GPU_BINS,
        auto_min_cells=None,
        disabled_by_env=False,
        build_has_accelerator=None,
        backend_source="injected",
        source="injected",
        notes=(),
    ):
        self.gpu_available = bool(gpu_available)
        self.backend = backend
        self.chip = chip
        self.device_memory_bytes = device_memory_bytes
        self.unified_memory = bool(unified_memory)
        self.gpu_objectives = frozenset(gpu_objectives)
        self.supports_multiclass = bool(supports_multiclass)
        self.supports_sparse = bool(supports_sparse)
        self.supports_custom_objective = bool(supports_custom_objective)
        self.supports_eval_set = bool(supports_eval_set)
        self.max_rows = int(max_rows)
        self.min_bins = int(min_bins)
        self.max_bins = int(max_bins)
        self.auto_min_cells = (
            None if auto_min_cells is None else int(auto_min_cells)
        )
        self.disabled_by_env = bool(disabled_by_env)
        self.build_has_accelerator = build_has_accelerator
        self.backend_source = backend_source
        self.source = source
        self.notes = tuple(notes)

    def replace(self, **changes):
        """A copy with `changes` applied. Handy for building a fixture
        family from one baseline."""
        fields = dict(
            gpu_available=self.gpu_available,
            backend=self.backend,
            chip=self.chip,
            device_memory_bytes=self.device_memory_bytes,
            unified_memory=self.unified_memory,
            gpu_objectives=self.gpu_objectives,
            supports_multiclass=self.supports_multiclass,
            supports_sparse=self.supports_sparse,
            supports_custom_objective=self.supports_custom_objective,
            supports_eval_set=self.supports_eval_set,
            max_rows=self.max_rows,
            min_bins=self.min_bins,
            max_bins=self.max_bins,
            auto_min_cells=self.auto_min_cells,
            disabled_by_env=self.disabled_by_env,
            build_has_accelerator=self.build_has_accelerator,
            backend_source=self.backend_source,
            source=self.source,
            notes=self.notes,
        )
        unknown = set(changes) - set(fields)
        if unknown:
            raise TypeError(
                "unknown Capabilities field(s) "
                + ", ".join(sorted(unknown))
            )
        fields.update(changes)
        return Capabilities(**fields)

    def describe(self):
        """One line naming the backend and chip, for the explanation."""
        if not self.gpu_available:
            if self.disabled_by_env:
                return "no accelerator (MOJOBOOST_DISABLE_GPU=1)"
            return "no accelerator"
        parts = [self.backend or "unknown backend"]
        if self.chip:
            parts.append(self.chip)
        return "accelerator available, " + ", ".join(parts)

    def to_dict(self):
        return {
            "gpu_available": self.gpu_available,
            "backend": self.backend,
            "backend_source": self.backend_source,
            "chip": self.chip,
            "device_memory_bytes": self.device_memory_bytes,
            "unified_memory": self.unified_memory,
            "gpu_objectives": sorted(self.gpu_objectives),
            "supports_multiclass": self.supports_multiclass,
            "supports_sparse": self.supports_sparse,
            "supports_custom_objective": self.supports_custom_objective,
            "supports_eval_set": self.supports_eval_set,
            "max_rows": self.max_rows,
            "min_bins": self.min_bins,
            "max_bins": self.max_bins,
            "auto_min_cells": self.auto_min_cells,
            "disabled_by_env": self.disabled_by_env,
            "build_has_accelerator": self.build_has_accelerator,
            "source": self.source,
            "notes": list(self.notes),
        }

    def __repr__(self):
        return "Capabilities(gpu_available=%r, backend=%r, chip=%r)" % (
            self.gpu_available,
            self.backend,
            self.chip,
        )


class Workload:
    """The shape and the features of one training run.

    `n_outputs` is what the native `gpu_supports` check takes: 1 for
    single-output training and the class count for multiclass, which is
    how the classifier calls it (2 classes are one binary tree per round).

    `max_bin` is the estimator's parameter of that name, and it is the bin
    count the kernels see, so it feeds both the limit check and the
    histogram term of the memory estimate.
    """

    def __init__(
        self,
        n_rows,
        n_features,
        objective="regression",
        n_classes=1,
        max_bin=255,
        sparse=False,
        custom_objective=False,
        has_eval_set=False,
    ):
        self.n_rows = int(n_rows)
        self.n_features = int(n_features)
        self.objective = None if objective is None else str(objective)
        self.n_classes = int(n_classes)
        self.max_bin = int(max_bin)
        self.sparse = bool(sparse)
        self.custom_objective = bool(custom_objective)
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
        """`n_rows * n_features`, the unit `MOJOBOOST_AUTO_MIN_CELLS` and
        the crossover rules are written in."""
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
        if self.custom_objective:
            parts.append("custom objective")
        elif self.objective:
            parts.append("objective '%s'" % self.objective)
        if self.n_classes > 1:
            parts.append("%d classes" % self.n_classes)
        parts.append("%d output(s) per round" % self.n_outputs)
        parts.append("max_bin=%d" % self.max_bin)
        parts.append("sparse" if self.sparse else "dense")
        if self.has_eval_set:
            parts.append("with eval_set")
        return ", ".join(parts)

    def to_dict(self):
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "cells": self.cells,
            "objective": self.objective,
            "n_classes": self.n_classes,
            "n_outputs": self.n_outputs,
            "max_bin": self.max_bin,
            "sparse": self.sparse,
            "custom_objective": self.custom_objective,
            "has_eval_set": self.has_eval_set,
        }

    def __repr__(self):
        return "Workload(n_rows=%d, n_features=%d, objective=%r)" % (
            self.n_rows,
            self.n_features,
            self.objective,
        )


class MemoryEstimate:
    """What one GPU training session is estimated to allocate.

    Derived from the buffers `GpuHistogramBuilder.__init__` creates in
    src/mojoboost/histogram_gpu.mojo, one term per buffer:

        binned matrix     n_rows * n_features * 1     uint8
        leaf ids          n_rows * 4                  int32
        gradients         n_rows * 4 * n_outputs      float32
        hessians          n_rows * 4 * n_outputs      float32
        histograms        n_features * n_bins * 12    3 int32 planes
        feature ids       n_features * 4              int32

    plus, on the host, two pinned float32 staging planes of `n_rows` and
    one pinned copy of the histogram buffer.

    The tiled accumulation strategy also allocates a partial-histogram
    buffer whose size depends on device attributes read at runtime, so it
    cannot be computed here. `PARTIAL_BUDGET_BYTES` in
    src/mojoboost/gpu_tiling.mojo caps it at 64 MiB, and that cap is what
    `upper_bound_bytes` adds.

    This is an estimate, and it is labeled one everywhere it appears. It
    counts the training buffers, not the allocator's own overhead, and
    the `n_outputs` factor on the gradient planes is an upper bound that
    assumes every class plane is resident at once.
    """

    #: `PARTIAL_BUDGET_BYTES` in src/mojoboost/gpu_tiling.mojo.
    PARTIAL_BUDGET_BYTES = 64 << 20

    def __init__(self, components, host_components, partial_budget_bytes):
        self.components = dict(components)
        self.host_components = dict(host_components)
        self.partial_budget_bytes = int(partial_budget_bytes)

    @property
    def device_bytes(self):
        """Device allocations excluding the tiled partial buffer."""
        return sum(self.components.values())

    @property
    def upper_bound_bytes(self):
        """Device allocations with the partial-histogram cap included."""
        return self.device_bytes + self.partial_budget_bytes

    @property
    def host_bytes(self):
        """Pinned host staging buffers."""
        return sum(self.host_components.values())

    def describe(self):
        return (
            "%s device, %s including the tiled partial-histogram budget, "
            "%s pinned host"
            % (
                _fmt_bytes(self.device_bytes),
                _fmt_bytes(self.upper_bound_bytes),
                _fmt_bytes(self.host_bytes),
            )
        )

    def to_dict(self):
        return {
            "device_bytes": self.device_bytes,
            "upper_bound_bytes": self.upper_bound_bytes,
            "host_bytes": self.host_bytes,
            "partial_budget_bytes": self.partial_budget_bytes,
            "components": dict(self.components),
            "host_components": dict(self.host_components),
        }

    def __repr__(self):
        return "MemoryEstimate(device_bytes=%d)" % self.device_bytes


def estimate_gpu_memory(workload):
    """The `MemoryEstimate` for `workload`. See that class for the terms
    and for what the estimate does and does not count."""
    n_rows = workload.n_rows
    n_features = workload.n_features
    n_bins = workload.max_bin
    planes = workload.n_outputs
    components = {
        "binned_matrix": n_rows * n_features,
        "leaf_ids": n_rows * 4,
        "gradients": n_rows * 4 * planes,
        "hessians": n_rows * 4 * planes,
        "histograms": n_features * n_bins * 12,
        "feature_ids": n_features * 4,
    }
    host_components = {
        "staged_gradients": n_rows * 4,
        "staged_hessians": n_rows * 4,
        "histogram_readback": n_features * n_bins * 12,
    }
    return MemoryEstimate(
        components, host_components, MemoryEstimate.PARTIAL_BUDGET_BYTES
    )


class CrossoverRule:
    """One benchmark-derived rule saying "the GPU wins from here up".

    A rule is a claim about measured performance, so it carries the
    measurement with it. `evidence` cites where the numbers live (a
    document section, a benchmark file, a commit), `measured_on` names
    the device they came from, and `speedup` records what was seen. A
    rule without evidence is not a rule; `CROSSOVER_RULES` is empty for
    exactly that reason.

    Scope narrows a rule to what was actually measured. `backend` and
    `chip` limit it to one device family or one chip, `objectives` to the
    objectives that were benchmarked, and `min_rows`, `min_features`, and
    `min_cells` are the thresholds themselves. An unset field does not
    constrain. A rule matches only when every set field matches, so
    widening a rule to hardware nobody measured takes a deliberate edit.
    """

    def __init__(
        self,
        name,
        evidence,
        measured_on=None,
        backend=None,
        chip=None,
        objectives=None,
        min_rows=0,
        min_features=0,
        min_cells=0,
        max_classes=None,
        speedup=None,
    ):
        if not evidence:
            raise ValueError(
                "a crossover rule needs evidence; cite the benchmark that "
                "measured it"
            )
        self.name = str(name)
        self.evidence = str(evidence)
        self.measured_on = measured_on
        self.backend = backend
        self.chip = chip
        self.objectives = (
            None if objectives is None else frozenset(objectives)
        )
        self.min_rows = int(min_rows)
        self.min_features = int(min_features)
        self.min_cells = int(min_cells)
        self.max_classes = (
            None if max_classes is None else int(max_classes)
        )
        self.speedup = speedup

    def matches(self, capabilities, workload):
        """Whether this rule covers the (device, workload) pair."""
        if self.backend is not None and capabilities.backend != self.backend:
            return False
        if self.chip is not None and capabilities.chip != self.chip:
            return False
        if (
            self.objectives is not None
            and workload.objective not in self.objectives
        ):
            return False
        if self.max_classes is not None:
            if workload.n_classes > self.max_classes:
                return False
        if workload.n_rows < self.min_rows:
            return False
        if workload.n_features < self.min_features:
            return False
        if workload.cells < self.min_cells:
            return False
        return True

    def describe(self):
        scope = []
        if self.backend:
            scope.append(self.backend)
        if self.chip:
            scope.append(self.chip)
        where = " on " + " ".join(scope) if scope else ""
        return "%s%s (%s)" % (self.name, where, self.evidence)

    def to_dict(self):
        return {
            "name": self.name,
            "evidence": self.evidence,
            "measured_on": self.measured_on,
            "backend": self.backend,
            "chip": self.chip,
            "objectives": (
                None if self.objectives is None else sorted(self.objectives)
            ),
            "min_rows": self.min_rows,
            "min_features": self.min_features,
            "min_cells": self.min_cells,
            "max_classes": self.max_classes,
            "speedup": self.speedup,
        }

    def __repr__(self):
        return "CrossoverRule(%r)" % self.name


#: Version of the crossover table below. Bump it whenever a rule is
#: added, removed, or retuned, so a report from one release can be told
#: apart from a report from another.
RULES_VERSION = 1

#: The benchmark-derived crossover rules, in priority order. Empty, and
#: the module docstring says why: no measurement in this repository has
#: found a shape where GPU training beats CPU training, and the only
#: device that has ever run the GPU trainer end to end came out slower.
#: Do not add a rule from reasoning. Add one from a recorded sweep, cite
#: it in `evidence`, and bump `RULES_VERSION`.
CROSSOVER_RULES = ()


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


def _fmt_int(n):
    return "{:,}".format(int(n))


def _fmt_bytes(n):
    """Bytes as the largest binary unit that keeps the number readable."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return "%d B" % int(value)
            return "%.1f %s" % (value, unit)
        value /= 1024.0


def _env_auto_min_cells(environ):
    """`MOJOBOOST_AUTO_MIN_CELLS` as the native layer reads it: unset,
    unparsable, or negative all mean the heuristic is off, which this
    layer spells None. Mirrors `env_auto_min_cells` in device.mojo."""
    raw = environ.get("MOJOBOOST_AUTO_MIN_CELLS")
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return None if value < 0 else value


def _detect_backend(environ, system, machine):
    """(backend, source) by the cheapest honest means available.

    `MOJOBOOST_GPU_BACKEND` wins when it is set, because a user who names
    their backend knows better than any heuristic here. Otherwise Apple
    silicon means Metal, and on Linux the presence of a vendor driver
    node is the only filesystem-visible hint. Anything else is None, and
    None is reported as unknown rather than guessed at, since the backend
    only scopes crossover rules.
    """
    named = environ.get("MOJOBOOST_GPU_BACKEND")
    if named:
        return named.strip().lower(), "MOJOBOOST_GPU_BACKEND"
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "metal", "platform"
    if system == "Linux":
        if os.path.exists("/proc/driver/nvidia/version"):
            return "cuda", "/proc/driver/nvidia/version"
        if os.path.exists("/sys/module/amdgpu"):
            return "hip", "/sys/module/amdgpu"
    return None, "unknown"


def _detect_chip(system, machine):
    """The chip name when the platform will name it.

    macOS keeps it in a sysctl and nowhere `platform` reads, so that one
    costs a short subprocess; everywhere else `platform.processor()` is
    consulted and nothing is spawned. Any failure returns None, because
    an unknown chip is reported as unknown rather than guessed at.
    """
    if system == "Darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                timeout=2,
            )
            name = out.stdout.decode("utf-8", "replace").strip()
            return name or None
        except Exception:
            return None
    processor = platform.processor()
    if processor and processor != machine:
        return processor
    return None


def _physical_memory_bytes():
    """Installed host memory, or None when it cannot be read."""
    try:
        return int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, ValueError, OSError):
        return None


def _native_gpu_available():
    """(available, note) from the compiled extension, without making its
    absence fatal: the policy layer is useful on a machine that has not
    built the extension yet."""
    try:
        try:
            from . import _mojoboost
        except ImportError:
            import mojoboost._mojoboost as _mojoboost
        return bool(_mojoboost.gpu_available()), None
    except Exception as exc:
        return (
            False,
            "the compiled extension could not be loaded (%s), so no "
            "accelerator is reported" % exc,
        )


def detect_capabilities(environ=None, gpu_available=None):
    """`Capabilities` for the machine and build this process is running.

    `gpu_available` overrides the probe of the compiled extension, which
    is the one piece of detection that needs a built module; pass it to
    describe a machine other than this one. `environ` defaults to
    `os.environ` and supplies `MOJOBOOST_DISABLE_GPU`,
    `MOJOBOOST_AUTO_MIN_CELLS`, and `MOJOBOOST_GPU_BACKEND`.

    Every field this cannot determine comes back None, and the report
    says "unknown" for it rather than filling in a plausible value.
    """
    environ = os.environ if environ is None else environ
    notes = []
    if gpu_available is None:
        available, note = _native_gpu_available()
        if note:
            notes.append(note)
    else:
        available = bool(gpu_available)
    disabled = environ.get("MOJOBOOST_DISABLE_GPU") == "1"
    system = platform.system()
    machine = platform.machine()
    backend, backend_source = _detect_backend(environ, system, machine)
    chip = _detect_chip(system, machine)
    memory = _physical_memory_bytes()
    unified = backend == "metal"
    if not unified:
        # Host memory is not a device budget anywhere else, and nothing
        # here can query VRAM, so the budget stays unknown.
        memory = None
    elif memory is not None:
        notes.append(
            "device memory is host memory on this backend (unified), so "
            "the budget shown is installed RAM, not a reservation"
        )
    return Capabilities(
        gpu_available=available and not disabled,
        backend=backend,
        chip=chip,
        device_memory_bytes=memory,
        unified_memory=unified,
        auto_min_cells=_env_auto_min_cells(environ),
        disabled_by_env=disabled,
        build_has_accelerator=available,
        backend_source=backend_source,
        source="detected",
        notes=notes,
    )


class DeviceReport:
    """The decision, everything it rested on, and how to read it aloud.

    `resolved` is "cpu" or "gpu" when a device was chosen and None when
    the request cannot run, in which case `would_raise` is True and
    `error` holds the message `select_device` raises.
    """

    def __init__(
        self,
        requested,
        resolved,
        reasons,
        capabilities,
        workload,
        memory,
        rules_version,
        rules_considered,
        matched_rule=None,
        would_raise=False,
        error=None,
        warnings=(),
    ):
        self.requested = requested
        self.resolved = resolved
        self.reasons = tuple(reasons)
        self.capabilities = capabilities
        self.workload = workload
        self.memory = memory
        self.rules_version = rules_version
        self.rules_considered = int(rules_considered)
        self.matched_rule = matched_rule
        self.would_raise = bool(would_raise)
        self.error = error
        self.warnings = tuple(warnings)

    @property
    def validated(self):
        """Whether a GPU choice rests on a benchmark-derived rule. False
        for every CPU choice, and False for a GPU chosen through the
        `MOJOBOOST_AUTO_MIN_CELLS` knob or requested explicitly."""
        return self.resolved == "gpu" and self.matched_rule is not None

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
        lines.append("Device      %s" % self.capabilities.describe())
        lines.append("Workload    %s" % self.workload.describe())
        lines.append("Memory      %s (estimate)" % self.memory.describe())
        budget = self.capabilities.device_memory_bytes
        lines.append(
            "Budget      %s"
            % (
                "unknown, so memory is not a factor"
                if budget is None
                else _fmt_bytes(budget)
            )
        )
        lines.append(
            "Rules       version %s, %d rule(s), %s"
            % (
                self.rules_version,
                self.rules_considered,
                (
                    "matched %s" % self.matched_rule.describe()
                    if self.matched_rule is not None
                    else "none matched"
                ),
            )
        )
        lines.append("")
        lines.append("Why")
        for reason in self.reasons:
            lines.append("  [%s] %s" % (reason.code, reason.message))
        if self.would_raise:
            lines.append("")
            lines.append("Error")
            lines.append("  %s" % self.error)
        if self.warnings:
            lines.append("")
            lines.append("Warnings")
            for warning in self.warnings:
                lines.append("  %s" % warning)
        if self.capabilities.notes:
            lines.append("")
            lines.append("Notes")
            for note in self.capabilities.notes:
                lines.append("  %s" % note)
        return "\n".join(lines)

    def to_dict(self):
        """The report as JSON-serializable data."""
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "would_raise": self.would_raise,
            "error": self.error,
            "validated": self.validated,
            "rules_version": self.rules_version,
            "rules_considered": self.rules_considered,
            "matched_rule": (
                None
                if self.matched_rule is None
                else self.matched_rule.to_dict()
            ),
            "reasons": [r.to_dict() for r in self.reasons],
            "warnings": list(self.warnings),
            "capabilities": self.capabilities.to_dict(),
            "workload": self.workload.to_dict(),
            "memory": self.memory.to_dict(),
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


def _normalize_device(device):
    """The requested device, lowercased as LightGBM treats `device_type`.
    Raises `ValueError` for anything outside the vocabulary, which is
    what the estimators do."""
    if device is None:
        return "cpu"
    if not isinstance(device, str) or device.lower() not in DEVICES:
        raise ValueError(
            "unknown device %r; expected one of %s"
            % (device, ", ".join(DEVICES))
        )
    return device.lower()


def _hard_blocks(capabilities, workload, memory):
    """Reasons the GPU path will actually fail for this workload.

    Each one corresponds to a rule already enforced somewhere else: the
    estimator's own guards in python/mojoboost/__init__.py, the device
    policy in src/mojoboost/device.mojo, or the kernels' indexing limits
    in src/mojoboost/histogram_gpu.mojo. Ordered from the cheapest and
    most fundamental to the most workload-specific, so the first one is
    the one worth telling the user about.
    """
    blocks = []
    if not capabilities.gpu_available:
        if capabilities.disabled_by_env:
            blocks.append(
                Reason(
                    GPU_DISABLED_ENV,
                    "MOJOBOOST_DISABLE_GPU=1 pins this process to the CPU "
                    "backend, so no accelerator is available",
                )
            )
        else:
            blocks.append(
                Reason(
                    NO_ACCELERATOR,
                    "no accelerator is available to this build",
                )
            )
        return blocks
    if workload.sparse and not capabilities.supports_sparse:
        blocks.append(
            Reason(
                UNSUPPORTED_FEATURE,
                "sparse input trains on the CPU; there is no sparse GPU "
                "kernel",
            )
        )
    if workload.custom_objective and not (
        capabilities.supports_custom_objective
    ):
        blocks.append(
            Reason(
                UNSUPPORTED_FEATURE,
                "custom objectives train on the CPU through the Python "
                "estimators",
            )
        )
    if workload.has_eval_set and not capabilities.supports_eval_set:
        blocks.append(
            Reason(
                UNSUPPORTED_FEATURE,
                "validation metrics are scored on the CPU, so a run with "
                "an eval_set trains there too",
            )
        )
    if workload.objective in CPU_ONLY_OBJECTIVES:
        blocks.append(
            Reason(
                UNSUPPORTED_FEATURE,
                "objective '%s' trains on the CPU only"
                % workload.objective,
            )
        )
    if workload.n_outputs > 1 and not capabilities.supports_multiclass:
        blocks.append(
            Reason(
                UNSUPPORTED_FEATURE,
                "multiclass grows one tree per class per round and this "
                "build's GPU path does not cover it",
            )
        )
    if workload.n_rows > capabilities.max_rows:
        blocks.append(
            Reason(
                WORKLOAD_LIMIT,
                "%s rows is past the %s the GPU kernels can index"
                % (
                    _fmt_int(workload.n_rows),
                    _fmt_int(capabilities.max_rows),
                ),
            )
        )
    if not (
        capabilities.min_bins <= workload.max_bin <= capabilities.max_bins
    ):
        blocks.append(
            Reason(
                WORKLOAD_LIMIT,
                "max_bin=%d is outside the [%d, %d] the GPU histogram "
                "kernels support"
                % (
                    workload.max_bin,
                    capabilities.min_bins,
                    capabilities.max_bins,
                ),
            )
        )
    budget = capabilities.device_memory_bytes
    if budget is not None and memory.device_bytes > budget:
        blocks.append(
            Reason(
                INSUFFICIENT_MEMORY,
                "the estimated %s of training buffers does not fit the %s "
                "budget" % (_fmt_bytes(memory.device_bytes),
                            _fmt_bytes(budget)),
            )
        )
    return blocks


def _soft_notes(capabilities, workload):
    """Reasons to distrust the GPU for this workload without refusing it.

    These keep `auto` on the CPU and travel with an explicit `"gpu"` run
    as warnings. Nothing here blocks: refusing a run the native layer
    would have accepted would be this layer overreaching.
    """
    notes = []
    if workload.custom_objective:
        return notes
    if (
        workload.objective is not None
        and workload.objective not in capabilities.gpu_objectives
    ):
        notes.append(
            Reason(
                UNSUPPORTED_OBJECTIVE,
                "objective '%s' is outside the set the GPU path is "
                "documented to cover (%s), so auto will not choose the "
                "GPU for it"
                % (
                    workload.objective,
                    ", ".join(sorted(capabilities.gpu_objectives)),
                ),
            )
        )
    if capabilities.backend is None and capabilities.gpu_available:
        notes.append(
            Reason(
                UNVALIDATED_PATH,
                "the accelerator backend could not be identified, so no "
                "crossover rule can be scoped to it",
            )
        )
    return notes


def _decide(requested, capabilities, workload, rules, rules_version):
    """The policy itself. Returns a `DeviceReport`, and never raises for
    an unsupported GPU request; `select_device` does the raising."""
    memory = estimate_gpu_memory(workload)
    blocks = _hard_blocks(capabilities, workload, memory)
    soft = _soft_notes(capabilities, workload)
    common = dict(
        requested=requested,
        capabilities=capabilities,
        workload=workload,
        memory=memory,
        rules_version=rules_version,
        rules_considered=len(rules),
    )

    if requested == "cpu":
        return DeviceReport(
            resolved="cpu",
            reasons=[
                Reason(
                    EXPLICIT_CPU,
                    "device='cpu' was requested, and the CPU path covers "
                    "every objective and every input",
                )
            ],
            **common
        )

    if requested == "gpu":
        if blocks:
            first = blocks[0]
            message = (
                "device 'gpu' requested but %s; use device='cpu' or "
                "device='auto'" % first.message
            )
            return DeviceReport(
                resolved=None,
                reasons=blocks,
                would_raise=True,
                error=message,
                **common
            )
        return DeviceReport(
            resolved="gpu",
            reasons=[
                Reason(
                    EXPLICIT_GPU,
                    "device='gpu' was requested and nothing blocks it, so "
                    "training runs on the accelerator; an explicit request "
                    "never falls back to the CPU",
                )
            ]
            + soft,
            warnings=[
                "device='gpu' was requested explicitly, so this run is not "
                "backed by a crossover rule and may be slower than the CPU"
            ],
            **common
        )

    # device="auto"
    if blocks:
        return DeviceReport(resolved="cpu", reasons=blocks, **common)
    if soft:
        return DeviceReport(
            resolved="cpu",
            reasons=soft
            + [
                Reason(
                    NO_VALIDATED_RULE,
                    "auto chooses the CPU when anything about the GPU "
                    "path for this workload is uncharacterized",
                )
            ],
            **common
        )

    min_cells = capabilities.auto_min_cells
    if min_cells is not None:
        if workload.cells >= min_cells:
            return DeviceReport(
                resolved="gpu",
                reasons=[
                    Reason(
                        ENV_THRESHOLD,
                        "MOJOBOOST_AUTO_MIN_CELLS=%d and this workload has "
                        "%s cells, so the size heuristic selects the GPU"
                        % (min_cells, _fmt_int(workload.cells)),
                    )
                ]
                + soft,
                warnings=[
                    "MOJOBOOST_AUTO_MIN_CELLS is the knob for running the "
                    "crossover benchmark, not a validated threshold; this "
                    "GPU choice rests on no measurement"
                ],
                **common
            )
        return DeviceReport(
            resolved="cpu",
            reasons=[
                Reason(
                    BELOW_ENV_THRESHOLD,
                    "MOJOBOOST_AUTO_MIN_CELLS=%d and this workload has "
                    "only %s cells"
                    % (min_cells, _fmt_int(workload.cells)),
                )
            ],
            **common
        )

    for rule in rules:
        if rule.matches(capabilities, workload):
            return DeviceReport(
                resolved="gpu",
                matched_rule=rule,
                reasons=[
                    Reason(
                        RULE_MATCHED,
                        "crossover rule %s covers this device and "
                        "workload, so auto selects the GPU"
                        % rule.describe(),
                    )
                ]
                + soft,
                **common
            )

    if rules:
        return DeviceReport(
            resolved="cpu",
            reasons=[
                Reason(
                    BELOW_RULE_THRESHOLD,
                    "none of the %d crossover rule(s) in version %s covers "
                    "this device and workload, so auto keeps the CPU"
                    % (len(rules), rules_version),
                )
            ],
            **common
        )

    return DeviceReport(
        resolved="cpu",
        reasons=[
            Reason(
                NO_VALIDATED_RULE,
                "the crossover table (version %s) is empty: no benchmark "
                "has established a workload size where GPU training beats "
                "CPU training, so auto conservatively keeps the CPU. Set "
                "MOJOBOOST_AUTO_MIN_CELLS to run that benchmark, or "
                "device='gpu' to force the accelerator" % rules_version,
            )
        ],
        **common
    )


def _as_workload(workload, kwargs):
    if isinstance(workload, Workload):
        if kwargs:
            raise TypeError(
                "workload keyword(s) %s cannot be combined with a Workload"
                % ", ".join(sorted(kwargs))
            )
        return workload
    raise TypeError(
        "expected a Workload; got %r" % type(workload).__name__
    )


def select_device(
    device,
    workload,
    capabilities=None,
    rules=None,
    rules_version=None,
):
    """Resolve `device` for `workload` and report why.

    Returns a `DeviceReport` whose `resolved` is "cpu" or "gpu". Raises
    `ValueError` for a device name outside `DEVICES`, and
    `DeviceUnavailableError` when `device="gpu"` cannot run: an explicit
    GPU request either runs on the GPU or fails, never quietly on the
    CPU.

    `capabilities` defaults to `detect_capabilities()`. Pass one to
    describe a machine other than this one, which is how the tests cover
    hardware nobody here owns. `rules` defaults to `CROSSOVER_RULES`;
    pass a table (with `rules_version`) to try a candidate crossover rule
    without editing this module.
    """
    requested = _normalize_device(device)
    workload = _as_workload(workload, {})
    if capabilities is None:
        capabilities = detect_capabilities()
    if rules is None:
        rules = CROSSOVER_RULES
        if rules_version is None:
            rules_version = RULES_VERSION
    elif rules_version is None:
        rules_version = "custom"
    report = _decide(
        requested, capabilities, workload, tuple(rules), rules_version
    )
    return report.raise_if_unsupported()


def explain_device_choice(
    X,
    y=None,
    device="auto",
    capabilities=None,
    rules=None,
    rules_version=None,
    **workload_kwargs
):
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
    requested = _normalize_device(device)
    if capabilities is None:
        capabilities = detect_capabilities()
    if rules is None:
        rules = CROSSOVER_RULES
        if rules_version is None:
            rules_version = RULES_VERSION
    elif rules_version is None:
        rules_version = "custom"
    return _decide(
        requested, capabilities, workload, tuple(rules), rules_version
    )
