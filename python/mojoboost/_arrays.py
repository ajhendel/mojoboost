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
    """True for a SciPy sparse matrix or array."""
    return hasattr(X, "toarray") and hasattr(X, "shape") and hasattr(X, "nnz")


is_sparse = _is_sparse


class SparseBuffers:
    """A canonical CSC or CSR matrix as float64 data with int64 indices.

    Holds the three arrays so the caller can keep them referenced while
    their addresses are in flight, exactly as the dense path keeps its
    column-major buffer alive.
    """

    __slots__ = ("data", "indices", "indptr", "n_rows", "n_features", "layout")

    def __init__(self, data, indices, indptr, n_rows, n_features, layout):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.n_rows = n_rows
        self.n_features = n_features
        self.layout = layout

    @property
    def nnz(self):
        return int(self.data.shape[0])

    def params(self):
        """The keys the extension module reads a sparse matrix from."""
        return {
            "sparse_data_addr": addr(self.data),
            "sparse_indices_addr": addr(self.indices),
            "sparse_indptr_addr": addr(self.indptr),
            "sparse_nnz": self.nnz,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
        }


def _canonical_sparse(X, layout, name):
    """`X` as a SciPy matrix in `layout` with sorted, deduplicated indices.

    Converts only when it has to, and copies before canonicalizing so a
    caller's matrix is never mutated in place.
    """
    if getattr(X, "format", None) != layout:
        X = X.tocsc() if layout == "csc" else X.tocsr()
    if not getattr(X, "has_canonical_format", False):
        X = X.copy()
        # sum_duplicates sorts the indices as well.
        X.sum_duplicates()
    if getattr(X, "ndim", 2) != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    return X


def check_X_sparse(X, layout="csc", name="X"):
    """(SparseBuffers, n_rows, n_features, names) for a SciPy sparse matrix.

    Implicit zeros are numerical zeros, so this validates exactly what the
    dense path validates: NaN is allowed and means missing, infinities are
    not. Explicitly stored zeros are left alone, because they mean the same
    thing as the gaps.
    """
    if np is None:
        raise TypeError(
            f"{name} is a sparse matrix, which needs numpy; install numpy or "
            "pass a dense sequence"
        )
    Xs = _canonical_sparse(X, layout, name)
    n_rows, n_features = Xs.shape
    if n_rows == 0:
        raise ValueError(f"{name} must have at least one row")
    if n_features == 0:
        raise ValueError(f"{name} must have at least one feature")
    data = np.ascontiguousarray(Xs.data, dtype=np.float64)
    if data.size and np.isinf(data).any():
        raise ValueError(
            f"{name} must not contain infinite values (NaN is allowed and "
            "is treated as missing)"
        )
    buffers = SparseBuffers(
        data,
        np.ascontiguousarray(Xs.indices, dtype=np.int64),
        np.ascontiguousarray(Xs.indptr, dtype=np.int64),
        int(n_rows),
        int(n_features),
        layout,
    )
    return buffers, int(n_rows), int(n_features), feature_names(X)




def take_rows(data, rows, name="X"):
    """`data` restricted to `rows`, in the order `rows` gives them.

    The layouts a feature matrix arrives in are not one type, so this
    dispatches the way the rest of the package does: on what the object
    offers rather than on what it is. A frame keeps its columns, so a
    selection out of a pandas or polars frame still carries the feature
    names and the category dtypes the whole one had.

    This is row selection on the *raw* matrix, which is the only place it is
    safe to do: bin edges are fitted from data, so selecting rows out of a
    matrix that was already binned would carry the whole matrix's quantiles
    into the part. `mojoboost.cv` builds every fold through here for exactly
    that reason, and `src/mojoboost/raw_data.mojo` does the same selection on
    the Mojo side.

    Sparse input is refused rather than sliced, because the `Dataset` the
    selection would be handed does not accept it yet (see
    handoffs/connect_12_dataset_cv.md); scipy would slice it happily, and the
    failure would then surface two calls later with a worse message.
    """
    if _is_sparse(data):
        raise TypeError(
            f"{name} is a sparse matrix, and rows cannot be selected out of "
            "it here because Dataset does not accept sparse data; densify "
            "with .toarray()"
        )
    index = list(rows)
    iloc = getattr(data, "iloc", None)
    if iloc is not None:  # pandas
        return iloc[index]
    if np is not None and isinstance(data, np.ndarray):
        return data[np.asarray(index, dtype=np.intp)]
    take = getattr(data, "take", None)
    if take is not None and getattr(data, "column_names", None) is not None:
        return take(index)  # pyarrow
    if getattr(data, "columns", None) is not None:
        try:
            return data[index]  # polars
        except (TypeError, ValueError):
            pass
    if np is not None and hasattr(data, "__array__"):
        return np.asarray(data)[np.asarray(index, dtype=np.intp)]
    return [data[i] for i in index]


def take_column(column, rows):
    """One per-row column restricted to `rows`, or None for a column the
    caller does not have."""
    if column is None:
        return None
    if np is not None:
        return np.asarray(column)[np.asarray(list(rows), dtype=np.intp)]
    return [column[i] for i in rows]


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


def frame_categories(X):
    """`{column_index: [category, ...]}` for the pandas category-dtype
    columns of `X`, in column order; empty for anything else.

    The categories are the column's own, in its own order, which is what
    `.cat.codes` numbers from 0. Recording them is what lets a later
    prediction frame be encoded through the *fitted* mapping instead of its
    own, which may order or extend the categories differently.
    """
    iloc = getattr(X, "iloc", None)
    columns = getattr(X, "columns", None)
    if iloc is None or columns is None:
        return {}
    out = {}
    for i in range(len(columns)):
        categories = getattr(getattr(iloc[:, i], "cat", None), "categories", None)
        if categories is None:
            continue
        out[i] = list(categories)
    return out


def _codes_from_labels(column, categories):
    """A column's values as float64 category codes under `categories`.

    A value absent from the table, including a missing one, becomes -1,
    which mojoboost's binner reads as the unknown category (see
    src/mojoboost/categorical.mojo).
    """
    cat = getattr(column, "cat", None)
    own = getattr(cat, "categories", None)
    if own is not None and list(own) == list(categories):
        # Same table in the same order, so pandas' own codes are the answer
        # and already use -1 for missing.
        return np.asarray(cat.codes, dtype=np.float64)
    index = {}
    for i, label in enumerate(categories):
        index[label] = float(i)
    values = np.asarray(column, dtype=object)
    out = np.empty(len(values), dtype=np.float64)
    for r in range(len(values)):
        try:
            out[r] = index.get(values[r], -1.0)
        except TypeError:  # an unhashable value cannot be a category
            out[r] = -1.0
    return out


def frame_to_array(X, encoders, name="X"):
    """`X` as a float64 array, with the columns in `encoders` replaced by
    their category codes under the given tables.

    `encoders` maps a column index to the fitted category list. Every other
    column converts as it would on its own, so a frame that mixes encoded
    and numeric columns stays one matrix.
    """
    iloc = getattr(X, "iloc", None)
    shape = getattr(X, "shape", None)
    if iloc is None or shape is None or len(shape) != 2:
        raise ValueError(
            f"{name} must be a pandas DataFrame to carry categorical columns"
        )
    n_rows, n_features = shape
    out = np.empty((n_rows, n_features), dtype=np.float64, order="F")
    for i in range(n_features):
        column = iloc[:, i]
        if i in encoders:
            out[:, i] = _codes_from_labels(column, encoders[i])
            continue
        try:
            out[:, i] = np.asarray(column, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} column {i} could not be converted to float64: {exc}"
            ) from None
    return out


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


def column_major(X, name="X", encoders=None):
    """Return (buffer, n_rows, n_features): a validated float64
    column-major buffer plus its shape. The caller must keep the buffer
    referenced while using its address.

    A non-empty `encoders` maps column indices to fitted category tables;
    those columns become integer codes before the conversion, so a frame
    holding labels rather than numbers still becomes a numeric matrix.
    """
    if _is_sparse(X):
        raise TypeError(
            f"{name} is a sparse matrix; the sparse path takes it directly "
            "(see check_X_sparse) and this dense entry point will not "
            "densify it silently"
        )
    if encoders:
        X = frame_to_array(X, encoders, name)
    if np is not None:
        return _as_column_major_numpy(X, name)
    return _as_column_major_stdlib(X, name)


def check_X(X, name="X", encoders=None):
    """(buffer, n_rows, n_features, names) for a feature matrix, with the
    column names when it carries them. `encoders` is as in `column_major`;
    the names come from the original `X`, before any encoding."""
    names = feature_names(X)
    buf, n_rows, n_features = column_major(X, name, encoders)
    return buf, n_rows, n_features, names


def column_view(buf, n_rows, index):
    """One column of a validated column-major feature buffer."""
    if np is not None and isinstance(buf, np.ndarray):
        return buf[:, index]
    return buf[index * n_rows : (index + 1) * n_rows]


def first_bad_code(column, limit):
    """The first value in `column` that cannot be a category code, or None.

    A code is a whole number below `limit`. `NaN` and negative values are
    missing rather than codes, so they are never bad; a fractional
    nonnegative value is, because rounding it would silently merge two
    categories.
    """
    if np is not None and isinstance(column, np.ndarray):
        bad = (column >= limit) | (
            (column >= 0.0) & (column != np.floor(column))
        )
        found = np.flatnonzero(bad)
        if found.size == 0:
            return None
        return float(column[found[0]])
    for v in column:
        if v >= limit or (v >= 0.0 and v != math.floor(v)):
            return float(v)
    return None


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
