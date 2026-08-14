"""scikit-learn style Python API for mojoboost.

Build the extension first: `bindings/build.sh` (produces `_mojoboost.so`
next to this file). numpy is used when available; plain Python sequences
(lists of rows) work without it.

    from mojoboost import MojoBoostRegressor
    model = MojoBoostRegressor().fit(X, y)
    pred = model.predict(X)

Estimators take `device="cpu"` (the default and the dependable backend),
`device="gpu"`, or `device="auto"`. `"gpu"` raises when no accelerator is
available or when the GPU path does not cover the workload, rather than
falling back silently; `"auto"` picks a backend for you and currently
always picks the CPU. `gpu_available()` reports whether this build can
train on an accelerator. Fitting records the backend that ran on
`device_`. See src/mojoboost/device.mojo for the full policy.
"""

import array as _array

from . import _mojoboost

try:
    import numpy as _np
except ImportError:
    _np = None

# Keep in sync with python/pyproject.toml.
__version__ = "0.1.0"

__all__ = ["MojoBoostRegressor", "MojoBoostClassifier", "gpu_available"]

_DEVICES = ("cpu", "gpu", "auto")

_SQUARED_ERROR = 0
_BINARY_LOGISTIC = 1
_HUBER = 3
_QUANTILE = 4
_L1 = 5


def _as_column_major(X):
    """Return (buffer, n_rows, n_features): float64 column-major buffer plus
    shape. The caller must keep the buffer referenced while using its
    address."""
    if _np is not None:
        Xa = _np.asfortranarray(X, dtype=_np.float64)
        if Xa.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        n_rows, n_features = Xa.shape
        return Xa, n_rows, n_features
    rows = [list(map(float, r)) for r in X]
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("X must not be empty")
    n_features = len(rows[0])
    for r in rows:
        if len(r) != n_features:
            raise ValueError("X rows must have equal length")
    flat = _array.array("d", bytes(8 * n_rows * n_features))
    for f in range(n_features):
        base = f * n_rows
        for r in range(n_rows):
            flat[base + r] = rows[r][f]
    return flat, n_rows, n_features


def _as_f64_vector(y, n_rows, name="y"):
    if _np is not None:
        ya = _np.ascontiguousarray(y, dtype=_np.float64)
        if ya.shape != (n_rows,):
            raise ValueError(f"{name} must have shape ({n_rows},)")
        return ya
    ya = _array.array("d", (float(v) for v in y))
    if len(ya) != n_rows:
        raise ValueError(f"{name} must have length {n_rows}")
    return ya


def _addr(buf):
    if _np is not None and isinstance(buf, _np.ndarray):
        return buf.ctypes.data
    return buf.buffer_info()[0]


def _out_buffer(n):
    if _np is not None:
        return _np.empty(n, dtype=_np.float64)
    return _array.array("d", bytes(8 * n))


def _finish(buf):
    return buf if _np is not None else list(buf)


def gpu_available():
    """True when this build can train on an accelerator. False on a
    CPU-only build and when `MOJOBOOST_DISABLE_GPU=1` is set."""
    return bool(_mojoboost.gpu_available())


class _Base:
    """Shared hyperparameters, mojoboost defaults (LightGBM-matched)."""

    def __init__(
        self,
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
        min_data_in_leaf=20,
        lambda_l2=1.0,
        lambda_l1=0.0,
        min_child_hess=1e-3,
        max_bin=255,
        device="cpu",
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.min_child_hess = min_child_hess
        self.max_bin = max_bin
        self.device = device
        self._model = None

    def _params(self, sample_weight_addr, device):
        if float(self.lambda_l1) < 0.0:
            raise ValueError("lambda_l1 must be nonnegative")
        return {
            "num_leaves": int(self.num_leaves),
            "learning_rate": float(self.learning_rate),
            "n_estimators": int(self.n_estimators),
            "min_data_in_leaf": int(self.min_data_in_leaf),
            "lambda_l2": float(self.lambda_l2),
            "lambda_l1": float(self.lambda_l1),
            "min_child_hess": float(self.min_child_hess),
            "max_bin": int(self.max_bin),
            "sample_weight_addr": int(sample_weight_addr),
            "alpha": float(getattr(self, "alpha", 0.9)),
            "device": device,
        }

    def _resolve_device(self, n_rows, n_features, n_outputs):
        """The backend that will actually run, "cpu" or "gpu". Raises
        ValueError for an unknown `device` and RuntimeError when "gpu" is
        requested but unavailable or unsupported; "gpu" never falls back to
        the CPU."""
        device = self.device
        if not isinstance(device, str) or device.lower() not in _DEVICES:
            raise ValueError(
                f"unknown device {device!r}; expected one of "
                + ", ".join(_DEVICES)
            )
        try:
            return _mojoboost.resolve_device(
                device.lower(), int(n_rows), int(n_features), int(n_outputs)
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from None

    def _weight_buffer(self, sample_weight, n_rows):
        """Validated weight buffer and its address (buffer must stay
        referenced while the address is in use); (None, 0) when absent."""
        if sample_weight is None:
            return None, 0
        wb = _as_f64_vector(sample_weight, n_rows, "sample_weight")
        if not any(wb):
            raise ValueError("sample_weight must not be all zeros")
        return wb, _addr(wb)

    def _require_fitted(self):
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")


class MojoBoostRegressor(_Base):
    """Objective names follow LightGBM: "regression" (squared error),
    "huber", "quantile", and "mae" (alias "regression_l1"). `alpha` is the
    quantile level for "quantile" and the transition point for "huber";
    the other objectives ignore it."""

    _OBJECTIVES = {
        "regression": _SQUARED_ERROR,
        "huber": _HUBER,
        "quantile": _QUANTILE,
        "mae": _L1,
        "regression_l1": _L1,
    }

    def __init__(self, objective="regression", alpha=0.9, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.alpha = alpha

    def _objective_code(self):
        code = self._OBJECTIVES.get(self.objective)
        if code is None:
            raise ValueError(
                f"unknown objective {self.objective!r}; expected one of "
                + ", ".join(sorted(self._OBJECTIVES))
            )
        alpha = float(self.alpha)
        if code == _HUBER and alpha <= 0.0:
            raise ValueError("huber requires alpha > 0")
        if code == _QUANTILE and not 0.0 < alpha < 1.0:
            raise ValueError("quantile requires 0 < alpha < 1")
        return code

    def fit(self, X, y, sample_weight=None):
        objective = self._objective_code()
        Xb, n_rows, n_features = _as_column_major(X)
        yb = _as_f64_vector(y, n_rows)
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        device = self._resolve_device(n_rows, n_features, 1)
        self._model = _mojoboost.fit(
            _addr(Xb),
            n_rows,
            n_features,
            _addr(yb),
            objective,
            self._params(w_addr, device),
        )
        self.n_features_in_ = n_features
        self.device_ = device
        return self

    def predict(self, X):
        self._require_fitted()
        Xb, n_rows, n_features = _as_column_major(X)
        out = _out_buffer(n_rows)
        _mojoboost.predict(
            self._model, _addr(Xb), n_rows, n_features, _addr(out)
        )
        return _finish(out)

    def save(self, path):
        self._require_fitted()
        _mojoboost.save(self._model, str(path))

    @classmethod
    def load(cls, path):
        """Load a saved model. The training device is not part of the model
        file (the ensemble is the same either way), so a loaded estimator
        has no `device_`; `device` governs the next `fit`."""
        est = cls()
        est._model = _mojoboost.load(str(path))
        return est


class MojoBoostClassifier(_Base):
    """Binary (logistic) for 2 classes, softmax for more. Labels must be
    integers 0..n_classes-1. Multiclass training is CPU-only, so
    `device="gpu"` raises for 3 or more classes and `device="auto"`
    resolves to the CPU."""

    def fit(self, X, y, sample_weight=None):
        Xb, n_rows, n_features = _as_column_major(X)
        yb = _as_f64_vector(y, n_rows)
        labels = sorted({int(v) for v in (y if _np is None else yb.tolist())})
        n_classes = (labels[-1] + 1) if labels else 0
        if labels != list(range(n_classes)) or n_classes < 2:
            raise ValueError(
                "labels must be consecutive integers starting at 0, "
                "with at least 2 classes present"
            )
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        device = self._resolve_device(n_rows, n_features, n_classes)
        if n_classes == 2:
            self._model = _mojoboost.fit(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                _BINARY_LOGISTIC,
                self._params(w_addr, device),
            )
        else:
            self._model = _mojoboost.fit_multiclass(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                n_classes,
                self._params(w_addr, device),
            )
        self.n_classes_ = n_classes
        self.n_features_in_ = n_features
        self.device_ = device
        return self

    def predict_proba(self, X):
        self._require_fitted()
        Xb, n_rows, n_features = _as_column_major(X)
        if self.n_classes_ == 2:
            out = _out_buffer(n_rows)
            _mojoboost.predict(
                self._model, _addr(Xb), n_rows, n_features, _addr(out)
            )
            if _np is not None:
                return _np.column_stack([1.0 - out, out])
            return [[1.0 - p, p] for p in out]
        out = _out_buffer(n_rows * self.n_classes_)
        _mojoboost.predict_proba(
            self._model, _addr(Xb), n_rows, n_features, _addr(out)
        )
        if _np is not None:
            return out.reshape(n_rows, self.n_classes_)
        k = self.n_classes_
        return [list(out[r * k : (r + 1) * k]) for r in range(n_rows)]

    def predict(self, X):
        proba = self.predict_proba(X)
        if _np is not None:
            return _np.argmax(proba, axis=1)
        return [max(range(len(p)), key=p.__getitem__) for p in proba]

    def save(self, path):
        self._require_fitted()
        if self.n_classes_ == 2:
            _mojoboost.save(self._model, str(path))
        else:
            _mojoboost.save_multiclass(self._model, str(path))

    @classmethod
    def load(cls, path):
        """Load a saved model. As with the regressor, the training device
        is not stored, so a loaded estimator has no `device_`."""
        est = cls()
        try:
            est._model = _mojoboost.load(str(path))
            est.n_classes_ = 2
        except Exception:
            est._model = _mojoboost.load_multiclass(str(path))
            est.n_classes_ = _mojoboost.n_classes(est._model)
        return est
