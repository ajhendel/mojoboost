"""Device-resident validation sets: the Python door to the `gpu_validation_*`
bindings.

The extension can keep a validation set's binned matrix and its raw scores
on the accelerator (`gpu_predict.mojo`, `GpuValidation` in
`bindings/_mojotrees.mojo`), fold model iterations into those scores one
range at a time, and score them with a device metric kernel. This module
wraps that surface once, so a caller carries one metric vocabulary
(`_eval`'s names and codes) rather than a device-side one:

    from mojotrees.gpu_validation import GpuValidation
    resident = GpuValidation.open(booster, X, y)
    resident.accumulate(booster)          # every iteration
    resident.score("l2")                  # on the device
    resident.raw()                        # the scores, back on the host

`Booster.eval(..., device="gpu")` is the reader that uses it; a metric the
device has no kernel for is scored on the host from `raw()`, which is what
the binding docstring calls the escape hatch. Nothing here decides where a
fit runs; that stays with `device.mojo`.
"""

from . import _arrays, _eval, _mojotrees

__all__ = [
    "GpuValidation",
    "device_metric_support",
    "metric_scored_on_device",
]

_addr = _arrays.addr
_out_buffer = _arrays.out_buffer


def device_metric_support(metric, task=_eval.REGRESSION):
    """`(has_kernel, matches_host)` for a built-in metric name.

    `has_kernel` says the device can compute it at all; `matches_host` says
    the device definition is metrics.mojo's term for term rather than
    merely close (`gpu_validation_metric_matches_host`). Both answers come
    from gpu_predict.mojo, which owns the kernels."""
    _canonical, code, _higher = _eval.resolve(metric, task)
    has_kernel, matches = _mojotrees.gpu_validation_metric_matches_host(code)
    return bool(has_kernel), bool(matches)


def metric_scored_on_device(metric, task=_eval.REGRESSION):
    """Whether `GpuValidation.score` computes this metric on the device with
    a definition identical to the host's. False means `score` falls back to
    the host suite over `raw()`."""
    has_kernel, matches = device_metric_support(metric, task)
    return has_kernel and matches


class GpuValidation:
    """A validation set resident on the accelerator, bound to one model.

    Build one with `open`; the object holds the native handle, the label
    and weight buffers (the extension reads them by address, so they must
    outlive it), and the model's task so metric names resolve the way
    `Booster.eval` resolves them.
    """

    def __init__(self, handle, n_rows, n_outputs, task, objective_code, keep):
        self._handle = handle
        self.n_rows = int(n_rows)
        self.n_outputs = int(n_outputs)
        self._task = task
        self._objective_code = int(objective_code)
        self._keep = keep

    @classmethod
    def open(cls, booster, X, y, weight=None):
        """Bin `X` with the model's own mapper and make it resident.

        `booster` is a `mojotrees.Booster` (or anything with `_handle`,
        `_n_classes`, `_task`, `_config`, and `_check_X`); `X` is a dense
        matrix; `y` the labels (class codes for a multiclass model); `weight`
        optional per-row weights. Raises the extension's message when there
        is no accelerator or the request is outside what the GPU path
        covers.
        """
        matrix, n_rows = booster._check_X(X)
        n_features = booster.num_feature()
        y_buf = _arrays.f64_vector(y, n_rows, "y")
        w_buf = (
            None
            if weight is None
            else _arrays.f64_vector(weight, n_rows, "weight")
        )
        params = {
            "y_addr": _addr(y_buf),
            "weight_addr": 0 if w_buf is None else _addr(w_buf),
        }
        n_classes = int(booster._n_classes or 0)
        if n_classes:
            handle = _mojotrees.gpu_validation_open_multiclass(
                booster._handle, _addr(matrix), n_rows, n_features, params
            )
        else:
            handle = _mojotrees.gpu_validation_open(
                booster._handle, _addr(matrix), n_rows, n_features, params
            )
        shape_rows, n_outputs = _mojotrees.gpu_validation_shape(handle)
        config = getattr(booster, "_config", None)
        objective_code = 0 if config is None else int(config.objective_code)
        task = getattr(booster, "_task", None) or _eval.REGRESSION
        return cls(
            handle,
            shape_rows,
            n_outputs,
            task,
            objective_code,
            keep=(matrix, y_buf, w_buf),
        )

    def shape(self):
        """`(n_rows, n_outputs)` of the resident score vector."""
        rows, outputs = _mojotrees.gpu_validation_shape(self._handle)
        return int(rows), int(outputs)

    def reset(self, base_scores):
        """Set every resident raw score to the per-output base score.
        `base_scores` has one entry per output; this is where a boosting
        run starts and the only place the base score enters."""
        buf = _arrays.f64_vector(base_scores, self.n_outputs, "base_scores")
        _mojotrees.gpu_validation_reset(self._handle, _addr(buf))
        return self

    def accumulate(self, booster, start=0, stop=None):
        """Fold the model's iterations `[start, stop)` into the resident
        scores; `stop=None` means through the last iteration. Empty ranges
        do nothing."""
        n_iter = int(booster.num_trees())
        if booster._n_classes:
            n_iter = n_iter // int(booster._n_classes)
        stop = n_iter if stop is None else int(stop)
        if booster._n_classes:
            _mojotrees.gpu_validation_accumulate_multiclass(
                self._handle, booster._handle, int(start), stop
            )
        else:
            _mojotrees.gpu_validation_accumulate(
                self._handle, booster._handle, int(start), stop
            )
        return self

    def raw(self):
        """The resident raw scores on the host: a float64 buffer of length
        `n_rows * n_outputs`, row-major, or an `(n_rows, n_outputs)` array
        when numpy is present."""
        out = _out_buffer(self.n_rows * self.n_outputs)
        _mojotrees.gpu_validation_raw(self._handle, _addr(out))
        if _arrays.np is not None and self.n_outputs > 1:
            return _arrays.np.asarray(out).reshape(self.n_rows, self.n_outputs)
        return out

    def score(self, metric):
        """The metric over the resident scores, computed on the device when
        the device has a kernel whose definition matches the host's, else
        raised as `ValueError` naming the metric so a caller can score
        `raw()` with the host suite (which is what `Booster.eval` does)."""
        canonical, code, _higher = _eval.resolve(metric, self._task)
        has_kernel, matches = _mojotrees.gpu_validation_metric_matches_host(
            code
        )
        if not (has_kernel and matches):
            raise ValueError(
                f"metric {canonical!r} has no device kernel that matches "
                "the host definition; score raw() with the host suite"
            )
        return float(
            _mojotrees.gpu_validation_metric(
                self._handle, code, self._objective_code
            )
        )
