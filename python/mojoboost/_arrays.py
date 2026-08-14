"""Buffer plumbing and input validation for the Python estimators.

Two jobs live here. The first is moving data across the extension-module
boundary: the bindings take raw float64 buffer addresses plus lengths, so
every array becomes a contiguous float64 buffer (column-major for feature
matrices) that the caller keeps referenced for the duration of the call.
numpy is used when available and `array.array` otherwise, so the estimators
work on a plain Python install.

The second is validating what the caller passed before any of it reaches
Mojo, which is where a bad shape or dtype would otherwise surface as an
opaque error. The rules follow LightGBM's scikit-learn wrapper, which
validates with `force_all_finite="allow-nan"`:

- `X` may contain `NaN`. `NaN` is mojoboost's missing-value marker and the
  binner reserves a bin for it (see src/mojoboost/binning.mojo).
- `X` may not contain `+inf` or `-inf`.
- `y` and `sample_weight` must be finite throughout. A missing label or a
  missing weight has no defined meaning.
- `sample_weight` must be nonnegative and not all zeros, since a zero
  weight drops a row and an all-zero vector drops every row.
"""

import array as _array
import math

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on numpy-free installs
    np = None


def have_numpy():
    """True when the numpy paths are in use."""
    return np is not None


def addr(buf):
    """Address of a float64 buffer's first element."""
    if np is not None and isinstance(buf, np.ndarray):
        return buf.ctypes.data
    return buf.buffer_info()[0]


def out_buffer(n):
    """An uninitialized float64 output buffer of length `n`."""
    if np is not None:
        return np.empty(n, dtype=np.float64)
    return _array.array("d", bytes(8 * n))


def finish(buf):
    """Convert an output buffer to what the public API returns: a numpy
    array when numpy is present, a list otherwise."""
    return buf if np is not None else list(buf)


def _is_sparse(X):
    """True for a scipy sparse matrix or array, which is not supported."""
    return hasattr(X, "toarray") and hasattr(X, "shape") and hasattr(X, "nnz")


def feature_names(X):
    """Column names when `X` carries them and all of them are strings
    (a pandas DataFrame, typically), None otherwise. Matching what
    scikit-learn records in `feature_names_in_`, a frame with non-string
    columns contributes no names."""
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return names


def name_array(names):
    """Feature names in the type `feature_names_in_` should hold."""
    if np is not None:
        return np.asarray(names, dtype=object)
    return list(names)


def _as_column_major_numpy(X, name):
    try:
        Xa = np.asfortranarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} could not be converted to a float64 array: {exc}"
        ) from None
    if Xa.ndim == 1:
        raise ValueError(
            f"{name} must be 2-dimensional, got a 1-dimensional array of "
            f"length {Xa.shape[0]}; reshape it to (-1, 1) for a single "
            "feature or (1, -1) for a single sample"
        )
    if Xa.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got {Xa.ndim}")
    n_rows, n_features = Xa.shape
    if n_rows == 0:
        raise ValueError(f"{name} must have at least one row")
    if n_features == 0:
        raise ValueError(f"{name} must have at least one feature")
    if np.isinf(Xa).any():
        raise ValueError(
            f"{name} must not contain infinite values (NaN is allowed and "
            "is treated as missing)"
        )
    return Xa, n_rows, n_features


def _as_column_major_stdlib(X, name):
    try:
        rows = [[float(v) for v in r] for r in X]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} could not be converted to a float64 array: {exc}"
        ) from None
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError(f"{name} must have at least one row")
    n_features = len(rows[0])
    if n_features == 0:
        raise ValueError(f"{name} must have at least one feature")
    for r in rows:
        if len(r) != n_features:
            raise ValueError(f"{name} rows must have equal length")
    flat = _array.array("d", bytes(8 * n_rows * n_features))
    for f in range(n_features):
        base = f * n_rows
        for r in range(n_rows):
            v = rows[r][f]
            if math.isinf(v):
                raise ValueError(
                    f"{name} must not contain infinite values (NaN is "
                    "allowed and is treated as missing)"
                )
            flat[base + r] = v
    return flat, n_rows, n_features


def column_major(X, name="X"):
    """Return (buffer, n_rows, n_features): a validated float64
    column-major buffer plus its shape. The caller must keep the buffer
    referenced while using its address."""
    if _is_sparse(X):
        raise TypeError(
            f"{name} is a sparse matrix, which mojoboost does not support; "
            "convert it with .toarray() first"
        )
    if np is not None:
        return _as_column_major_numpy(X, name)
    return _as_column_major_stdlib(X, name)


def check_X(X, name="X"):
    """(buffer, n_rows, n_features, names) for a feature matrix, with the
    column names when it carries them."""
    names = feature_names(X)
    buf, n_rows, n_features = column_major(X, name)
    return buf, n_rows, n_features, names


def f64_vector(y, n_rows, name="y"):
    """A float64 vector of length `n_rows`, without a finiteness check."""
    if np is not None:
        try:
            ya = np.ascontiguousarray(y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} could not be converted to a float64 array: {exc}"
            ) from None
        if ya.shape != (n_rows,):
            raise ValueError(
                f"{name} must have shape ({n_rows},), got {ya.shape}"
            )
        return ya
    try:
        ya = _array.array("d", (float(v) for v in y))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} could not be converted to a float64 array: {exc}"
        ) from None
    if len(ya) != n_rows:
        raise ValueError(f"{name} must have length {n_rows}, got {len(ya)}")
    return ya


def _require_finite(values, name):
    if np is not None:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must not contain NaN or infinite values")
        return
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{name} must not contain NaN or infinite values")


def check_target(y, n_rows, name="y"):
    """A finite float64 target vector of length `n_rows`."""
    yb = f64_vector(y, n_rows, name)
    _require_finite(yb, name)
    return yb


def check_sample_weight(sample_weight, n_rows):
    """A validated float64 weight vector: finite, nonnegative, and not all
    zeros (an all-zero vector drops every row from training)."""
    wb = f64_vector(sample_weight, n_rows, "sample_weight")
    _require_finite(wb, "sample_weight")
    if np is not None:
        if (wb < 0.0).any():
            raise ValueError("sample_weight must be nonnegative")
        if not wb.any():
            raise ValueError("sample_weight must not be all zeros")
        return wb
    if any(v < 0.0 for v in wb):
        raise ValueError("sample_weight must be nonnegative")
    if not any(wb):
        raise ValueError("sample_weight must not be all zeros")
    return wb


def encode_labels(y, n_rows):
    """Return (codes, classes) for a classification target.

    `classes` is the sorted unique labels, exactly as passed, and `codes`
    is a float64 buffer of their indices into it. Labels of any comparable
    type work, so strings and gappy integers are fine; the trainer only
    ever sees 0..n_classes-1, which is what it requires.
    """
    if np is not None:
        ya = np.asarray(y)
        if ya.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got {ya.ndim}")
        if ya.shape[0] != n_rows:
            raise ValueError(
                f"y must have shape ({n_rows},), got {ya.shape}"
            )
        if ya.dtype.kind in "fc":
            _require_finite(ya, "y")
        try:
            classes, inverse = np.unique(ya, return_inverse=True)
        except TypeError:
            raise ValueError(
                "y must hold labels of a single comparable type"
            ) from None
        codes = np.ascontiguousarray(
            inverse.reshape(-1), dtype=np.float64
        )
    else:
        values = list(y)
        if len(values) != n_rows:
            raise ValueError(
                f"y must have length {n_rows}, got {len(values)}"
            )
        for v in values:
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("y must not contain NaN or infinite values")
        try:
            classes = sorted(set(values))
        except TypeError:
            raise ValueError(
                "y must hold labels of a single comparable type"
            ) from None
        index = {label: i for i, label in enumerate(classes)}
        codes = _array.array("d", (float(index[v]) for v in values))
    if len(classes) < 2:
        raise ValueError(
            f"y must contain at least 2 classes, got {len(classes)}"
        )
    return codes, classes
