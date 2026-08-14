"""scikit-learn style Python API for mojoboost.

Build the extension first: `bindings/build.sh` (produces `_mojoboost.so`
next to this file). numpy is used when available; plain Python sequences
(lists of rows) work without it.

    from mojoboost import MojoBoostRegressor
    model = MojoBoostRegressor().fit(X, y)
    pred = model.predict(X)
"""

import array as _array

from . import _mojoboost

try:
    import numpy as _np
except ImportError:
    _np = None

__all__ = ["MojoBoostRegressor", "MojoBoostClassifier"]

_SQUARED_ERROR = 0
_BINARY_LOGISTIC = 1


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


class _Base:
    """Shared hyperparameters, mojoboost defaults (LightGBM-matched)."""

    def __init__(
        self,
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
        min_data_in_leaf=20,
        lambda_l2=1.0,
        min_child_hess=1e-3,
        max_bin=255,
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.min_child_hess = min_child_hess
        self.max_bin = max_bin
        self._model = None

    def _params(self, sample_weight_addr):
        return {
            "num_leaves": int(self.num_leaves),
            "learning_rate": float(self.learning_rate),
            "n_estimators": int(self.n_estimators),
            "min_data_in_leaf": int(self.min_data_in_leaf),
            "lambda_l2": float(self.lambda_l2),
            "min_child_hess": float(self.min_child_hess),
            "max_bin": int(self.max_bin),
            "sample_weight_addr": int(sample_weight_addr),
        }

    def _require_fitted(self):
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")


class MojoBoostRegressor(_Base):
    def fit(self, X, y, sample_weight=None):
        Xb, n_rows, n_features = _as_column_major(X)
        yb = _as_f64_vector(y, n_rows)
        wb = None
        w_addr = 0
        if sample_weight is not None:
            wb = _as_f64_vector(sample_weight, n_rows, "sample_weight")
            w_addr = _addr(wb)
        self._model = _mojoboost.fit(
            _addr(Xb),
            n_rows,
            n_features,
            _addr(yb),
            _SQUARED_ERROR,
            self._params(w_addr),
        )
        self.n_features_in_ = n_features
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
        est = cls()
        est._model = _mojoboost.load(str(path))
        return est


class MojoBoostClassifier(_Base):
    """Binary (logistic) for 2 classes, softmax for more. Labels must be
    integers 0..n_classes-1."""

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
        wb = None
        w_addr = 0
        if sample_weight is not None:
            wb = _as_f64_vector(sample_weight, n_rows, "sample_weight")
            w_addr = _addr(wb)
        if n_classes == 2:
            self._model = _mojoboost.fit(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                _BINARY_LOGISTIC,
                self._params(w_addr),
            )
        else:
            self._model = _mojoboost.fit_multiclass(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                n_classes,
                self._params(w_addr),
            )
        self.n_classes_ = n_classes
        self.n_features_in_ = n_features
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
        est = cls()
        try:
            est._model = _mojoboost.load(str(path))
            est.n_classes_ = 2
        except Exception:
            est._model = _mojoboost.load_multiclass(str(path))
            est.n_classes_ = _mojoboost.n_classes(est._model)
        return est
