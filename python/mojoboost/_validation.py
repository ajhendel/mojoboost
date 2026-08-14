"""Ecosystem-object structure checks, and nothing else.

This module answers one question about a Python object: can it be turned
into the buffers the extension module takes, and what shape are they. It
does not answer whether the numbers in those buffers are acceptable.

Why the line is drawn there
---------------------------
mojoboost's numeric rules live in ``src/mojoboost/validation.mojo``, which
is the layer every trainer, dataset, serializer, and binding calls. A rule
implemented twice is a rule with two answers, and the Python copy is the one
that drifts, because it is written against numpy's vocabulary rather than
against the loop that will actually read the values. The concrete case: the
Python layer rejects a ``sample_weight`` that is all zeros, and the Mojo
layer rejects one whose sum is not positive. Those are the same rule until a
vector of ``[1e-320, -1e-320]`` arrives, at which point one accepts and the
other does not, and which error a caller sees depends on whether they came
through the estimators or through ``mojoboost.Dataset``.

So the split is by *what the check needs to know*:

- **Structure** is a property of the Python object. Is it two-dimensional.
  Does it have the row count its label claims. Are its columns named with
  strings. Is the scipy matrix canonical. Can this dtype become float64 at
  all. None of that survives the crossing, because what crosses is an
  address and a length, so it has to be settled here.
- **Domain** is a property of the numbers. Finite, nonnegative, in range,
  summing to something positive, whole. All of it survives the crossing
  intact, so it is settled once, natively, where the training loop that
  depends on it lives.

The practical consequence is that this module never calls ``np.isfinite``,
``np.isinf``, or ``np.isnan``, never compares a value against a bound, and
never sums anything. If you find yourself reaching for one of those, the
check belongs in ``validation.mojo``.

Exceptions
----------
``ValueError`` for a well-typed object of the wrong shape or content-type,
``TypeError`` for an object of a kind that cannot be used at all. That is
the split scikit-learn uses and the one the estimators already raise, so
moving a check in here does not change what a caller catches.

numpy
-----
Optional, exactly as in ``_arrays``. Every function below works without it,
on sequences of sequences, and says so where the behavior differs.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on numpy-free installs
    np = None


# Mirrors of the ceilings in src/mojoboost/validation.mojo. They are checked
# on this side as well because the binding hands over an address and a
# length: a shape that overflows is a buffer the native side cannot size and
# this side has already allocated. Keep the two in step; the native values
# are the authority and these are the early warning.
MAX_FEATURES = 1 << 31
MAX_ROWS = 1 << 44
MAX_ALLOC_ELEMS = 1 << 46


def have_numpy():
    """True when the numpy paths are in use."""
    return np is not None


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def check_shape(n_rows, n_features, name="X"):
    """A feature matrix's shape, as two Python ints.

    Nonempty on both axes and small enough that ``n_rows * n_features`` is a
    number the native side will accept. Returns the pair as ``int``, so a
    numpy integer from ``.shape`` does not travel any further: the binding
    signature takes Python ints, and a ``np.int64`` there is a conversion
    the extension module has to guess at.
    """
    n_rows = int(n_rows)
    n_features = int(n_features)
    if n_rows < 1:
        raise ValueError(f"{name} must have at least one row, got {n_rows}")
    if n_features < 1:
        raise ValueError(
            f"{name} must have at least one feature, got {n_features}"
        )
    if n_rows > MAX_ROWS:
        raise ValueError(
            f"{name} has {n_rows} rows, above the limit of {MAX_ROWS}"
        )
    if n_features > MAX_FEATURES:
        raise ValueError(
            f"{name} has {n_features} features, above the limit of "
            f"{MAX_FEATURES}"
        )
    if n_rows > MAX_ALLOC_ELEMS // n_features:
        raise ValueError(
            f"{name} of {n_rows} by {n_features} needs more than "
            f"{MAX_ALLOC_ELEMS} elements, which mojoboost will not allocate"
        )
    return n_rows, n_features


def check_ndim(obj, expected, name="X"):
    """``obj.ndim`` when it has one, with the one-dimensional case given the
    message that says how to fix it.

    A 1-D array passed where a matrix belongs is the single most common
    shape mistake, and ``must be 2-dimensional`` does not tell the caller
    whether they meant one feature or one sample. Only they know, so the
    message offers both reshapes rather than choosing.
    """
    ndim = getattr(obj, "ndim", None)
    if ndim is None:
        return None
    ndim = int(ndim)
    if ndim == expected:
        return ndim
    if expected == 2 and ndim == 1:
        length = int(getattr(obj, "shape", (0,))[0])
        raise ValueError(
            f"{name} must be 2-dimensional, got a 1-dimensional array of "
            f"length {length}; reshape it to (-1, 1) for a single feature "
            "or (1, -1) for a single sample"
        )
    raise ValueError(
        f"{name} must be {expected}-dimensional, got {ndim}"
    )


def check_rectangular(rows, name="X"):
    """A sequence of sequences, as the numpy-free path receives it: nonempty,
    and every row the same length. Returns ``(n_rows, n_features)``.

    numpy would raise its own error for a ragged sequence, and in recent
    versions that error is about object dtype rather than about the ragged
    row, so the stdlib path checks it directly and names the row.
    """
    n_rows = len(rows)
    if n_rows < 1:
        raise ValueError(f"{name} must have at least one row")
    n_features = len(rows[0])
    if n_features < 1:
        raise ValueError(f"{name} must have at least one feature")
    for r in range(1, n_rows):
        if len(rows[r]) != n_features:
            raise ValueError(
                f"{name} rows must have equal length: row {r} has "
                f"{len(rows[r])}, row 0 has {n_features}"
            )
    return check_shape(n_rows, n_features, name)


# ---------------------------------------------------------------------------
# Per-row columns
# ---------------------------------------------------------------------------


def check_length(values, n_rows, name):
    """A per-row column has one entry per row. Returns its length.

    Length agreement is structure: it is a property of the two Python
    objects, it cannot be recovered from the buffers once they cross, and
    getting it wrong reads past the end of an allocation rather than
    producing a bad number. So it is checked here as well as natively, which
    is the one deliberate overlap in this module.
    """
    shape = getattr(values, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError(
                f"{name} must be 1-dimensional, got shape {tuple(shape)}"
            )
        length = int(shape[0])
    else:
        try:
            length = len(values)
        except TypeError:
            raise TypeError(
                f"{name} must be a sequence with a length, got "
                f"{type(values).__name__}"
            ) from None
    if length != n_rows:
        raise ValueError(
            f"{name} must have one entry per row: got {length} for "
            f"{n_rows} rows"
        )
    return length


def check_optional_length(values, n_rows, name):
    """The same rule for a column that may be absent. ``None`` passes."""
    if values is None:
        return 0
    return check_length(values, n_rows, name)


def check_float64_convertible(values, name):
    """The values can become a contiguous float64 buffer at all.

    This is a dtype question, not a domain one: a column of ``Decimal``, of
    strings, or of a pandas extension type with no float cast cannot cross
    the binding no matter what numbers it holds. numpy raises for those, and
    the raw message names neither the argument nor what it was asked to do,
    so it is caught and rewritten.

    Returns the converted array. Nothing here inspects a single value.
    """
    if np is not None:
        try:
            return np.ascontiguousarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} could not be converted to a float64 array: {exc}"
            ) from None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} could not be converted to a float64 array: {exc}"
        ) from None


# ---------------------------------------------------------------------------
# scipy sparse
# ---------------------------------------------------------------------------


def is_sparse(X):
    """True for a SciPy sparse matrix or array, by capability rather than by
    isinstance, so scipy stays an optional import."""
    return hasattr(X, "toarray") and hasattr(X, "shape") and hasattr(X, "nnz")


def check_sparse_layout(X, layout="csc", name="X"):
    """``X`` as a canonical scipy matrix in ``layout``, without mutating it.

    Canonical means deduplicated and index-sorted, which is what
    ``sum_duplicates()`` produces. The native reader requires strictly
    ascending inner indices within each outer slice
    (``validation.check_compressed``), and a non-canonical matrix is the one
    input that is structurally fine on this side and rejected on the other,
    so it is repaired here rather than reported.

    The copy before the repair is not optional: ``sum_duplicates`` works in
    place, and a caller's matrix is not this function's to modify.
    """
    if np is None:
        raise TypeError(
            f"{name} is a sparse matrix, which needs numpy; install numpy "
            "or pass a dense sequence"
        )
    if getattr(X, "format", None) != layout:
        X = X.tocsc() if layout == "csc" else X.tocsr()
    if not getattr(X, "has_canonical_format", False):
        X = X.copy()
        # sum_duplicates sorts the indices as well.
        X.sum_duplicates()
    if int(getattr(X, "ndim", 2)) != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    return X


def check_sparse_index_width(X, name="X"):
    """The index arrays fit the int64 the binding hands over.

    scipy picks int32 or int64 per matrix, and both widen to int64 cleanly,
    so this is a bound on the *values* an index array holds rather than on
    its dtype: ``indptr[-1]`` is the stored-entry count and is about to size
    an allocation on the far side. Returns the stored-entry count.
    """
    nnz = int(getattr(X, "nnz", 0))
    if nnz < 0:
        raise ValueError(f"{name} reports a negative stored-entry count")
    if nnz > MAX_ALLOC_ELEMS:
        raise ValueError(
            f"{name} has {nnz} stored entries, above the limit of "
            f"{MAX_ALLOC_ELEMS}"
        )
    return nnz


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def frame_column_names(X):
    """Column names when ``X`` carries them and all of them are strings, and
    ``None`` otherwise.

    Matching what scikit-learn records in ``feature_names_in_``, a frame with
    non-string columns contributes no names at all rather than a mix: a
    partially named feature axis is worse than an unnamed one, because
    positional and named lookups then disagree about the same matrix.
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return names


def check_feature_names_match(fitted, incoming, name="X"):
    """A prediction frame's columns match the fitted ones, in order.

    Order, not just membership. The binned matrix is positional, so a frame
    whose columns are the same set in a different order predicts confidently
    from the wrong features. Both sides being unnamed is fine and is how a
    plain array is handled; one side named and the other not is fine too,
    because a caller may reasonably fit on a frame and predict on an array.
    """
    if fitted is None or incoming is None:
        return
    if list(fitted) == list(incoming):
        return
    if set(fitted) == set(incoming):
        raise ValueError(
            f"{name} has the fitted feature names in a different order; "
            "mojoboost matches features by position, so reorder the columns "
            f"to {list(fitted)}"
        )
    missing = [n for n in fitted if n not in set(incoming)]
    extra = [n for n in incoming if n not in set(fitted)]
    raise ValueError(
        f"{name} does not carry the fitted feature names: missing "
        f"{missing}, unexpected {extra}"
    )


def check_frame_index_aligned(X, y, name="y"):
    """A pandas ``y`` is indexed like ``X``.

    Only checked when both carry an index, which means both are pandas. A
    misaligned index is the one Python-side mistake that produces a
    perfectly valid buffer of the right length holding the wrong labels,
    which no native check can see and no metric will flag. Objects without an
    index pass, since a plain array has no alignment to violate.
    """
    xi = getattr(X, "index", None)
    yi = getattr(y, "index", None)
    if xi is None or yi is None:
        return
    if len(xi) != len(yi):
        raise ValueError(
            f"{name} has {len(yi)} rows and X has {len(xi)}"
        )
    if not xi.equals(yi):
        raise ValueError(
            f"{name} is indexed differently from X; align it with "
            f"{name}.reindex(X.index) or pass {name}.to_numpy()"
        )


def check_category_tables(encoders, n_features, name="X"):
    """A ``{column index: [category, ...]}`` mapping for a frame's category
    dtypes: indices in range, tables nonempty and free of duplicates.

    Duplicates are the interesting case. A category list with the same label
    twice gives two codes to one value, and which one a row gets depends on
    whether pandas' own codes or this package's fallback lookup produced it,
    so the fitted mapping and the prediction mapping would disagree on
    exactly the rows the duplicate touches.
    """
    for index, table in sorted(encoders.items()):
        index = int(index)
        if index < 0 or index >= n_features:
            raise ValueError(
                f"{name} categorical column index {index} is out of range "
                f"for {n_features} features"
            )
        labels = list(table)
        if not labels:
            raise ValueError(
                f"{name} column {index} has an empty category table"
            )
        seen = set()
        for position, label in enumerate(labels):
            try:
                duplicate = label in seen
            except TypeError:  # an unhashable label cannot be a category
                raise ValueError(
                    f"{name} column {index} has an unhashable category at "
                    f"position {position}"
                ) from None
            if duplicate:
                raise ValueError(
                    f"{name} column {index} lists category {label!r} twice, "
                    f"the second time at position {position}"
                )
            seen.add(label)


# ---------------------------------------------------------------------------
# Parameter objects
# ---------------------------------------------------------------------------


def check_param_mapping(params, name="params"):
    """A parameter object is a mapping with string keys.

    The binding reads parameters by name out of a dict, so a non-mapping or a
    non-string key is a structural failure at the boundary rather than a bad
    value. What each parameter's legal range is stays native, in
    ``validation.check_booster_ranges`` and ``params.parse_params``.
    """
    items = getattr(params, "items", None)
    if items is None:
        raise TypeError(
            f"{name} must be a mapping of parameter names to values, got "
            f"{type(params).__name__}"
        )
    for key in params:
        if not isinstance(key, str):
            raise TypeError(
                f"{name} keys must be strings, got a "
                f"{type(key).__name__} key {key!r}"
            )
    return params


def check_int_param(value, name, minimum=None):
    """A parameter that must reach the binding as a Python ``int``.

    ``bool`` is refused first and on its own, even though it is an ``int``
    subclass and even though ``True.is_integer()`` is true on Python 3.12 and
    later: ``True`` for a count is always a mistake, and silently reading it
    as 1 is how a caller ends up training a single tree and reporting a bug
    about it. A whole-valued ``float`` is accepted, because ``100.0`` from a
    grid search or a JSON round trip means 100. Range checking past
    ``minimum`` is the native layer's, and ``minimum`` here exists only for
    the bounds that decide a buffer size on this side.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be an integer, got the boolean {value!r}"
        )
    if not isinstance(value, int):
        integral = getattr(value, "is_integer", None)
        if integral is None or not integral():
            raise TypeError(
                f"{name} must be an integer, got "
                f"{type(value).__name__} {value!r}"
            )
        value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def check_float_param(value, name):
    """A parameter that must reach the binding as a Python ``float``.

    Convertibility only. Whether the number is positive, finite, or inside
    ``(0, 1]`` is decided natively, so that a value set through the
    estimators and the same value set through a parameter string are
    rejected by the same rule with the same message.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got the boolean {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{name} must be a number, got "
            f"{type(value).__name__} {value!r}"
        ) from None


# ---------------------------------------------------------------------------
# Native-domain handoff
# ---------------------------------------------------------------------------


def check_fit_structure(
    X,
    y,
    n_rows,
    n_features,
    sample_weight=None,
    group=None,
    init_score=None,
    name="X",
):
    """Every structural rule a fit call has, in the order a caller hits them.

    One call so that adopting this module is a one-line edit at each
    estimator rather than a rewrite. It runs after ``X`` has been turned into
    a buffer, because the shape is what the earlier conversion produced, and
    it checks the per-row columns against that shape.

    ``group`` is checked only for length agreement with itself, not for
    contents: whether the counts are positive and sum to ``n_rows`` is
    arithmetic on the numbers, which is ``validation.check_group_counts``'s.

    Returns the validated ``(n_rows, n_features)`` pair.
    """
    n_rows, n_features = check_shape(n_rows, n_features, name)
    check_length(y, n_rows, "y")
    check_frame_index_aligned(X, y, "y")
    check_optional_length(sample_weight, n_rows, "sample_weight")
    check_optional_length(init_score, n_rows, "init_score")
    if group is not None:
        try:
            n_queries = len(group)
        except TypeError:
            raise TypeError(
                "group must be a sequence of per-query row counts, got "
                f"{type(group).__name__}"
            ) from None
        if n_queries < 1:
            raise ValueError("group must contain at least one query")
        if n_queries > n_rows:
            raise ValueError(
                f"group has {n_queries} queries but X has only {n_rows} "
                "rows, and every query needs at least one row"
            )
    return n_rows, n_features


def check_feature_count_matches(
    fitted_n_features, incoming_n_features, name="X"
):
    """A matrix is as wide as the one the model was fitted on.

    The binned matrix is positional, so a matrix of a different width is
    binned into bins that mean different features and predicts a number with
    no relationship to the input.
    """
    fitted = int(fitted_n_features)
    incoming = int(incoming_n_features)
    if incoming != fitted:
        raise ValueError(
            f"{name} has {incoming} features, but this model was fitted on "
            f"{fitted}"
        )
    return incoming


def check_predict_structure(
    X, n_features, fitted_n_features, fitted_names=None, n_rows=None, name="X"
):
    """Every structural rule a predict call has, in one call.

    Width first, because it is the failure that produces nonsense rather than
    an error. Then names, when both sides carry them, which catches the case
    a width check cannot: the same columns, rearranged.
    """
    incoming = check_feature_count_matches(
        fitted_n_features, n_features, name
    )
    if n_rows is not None:
        check_shape(n_rows, incoming, name)
    check_feature_names_match(fitted_names, frame_column_names(X), name)
    return incoming


def domain_checks_are_native():
    """True, always, and here to be imported rather than called.

    It documents the contract at the one place a reader looking for a
    finiteness check in this module will land: there isn't one, and
    ``src/mojoboost/validation.mojo`` is where it went. See
    ``docs/VALIDATION_CONTRACT.md`` for the full division.
    """
    return True


def describe_domain_owner(what):
    """Which native check owns a domain rule, for an error message that has
    to explain why this layer did not catch something.

    Unknown names return ``None`` rather than raising: this is a lookup for
    building a message, and failing to build a message is worse than
    building a vaguer one.
    """
    owners = {
        "features": "validation.check_features_finite",
        "label": "validation.check_labels_finite",
        "sample_weight": "validation.check_weights",
        "gradients": "validation.check_gradient_pair",
        "hessians": "validation.check_gradient_pair",
        "class_codes": "validation.check_class_codes",
        "relevance": "validation.check_relevance_labels",
        "group": "validation.check_group_counts",
        "categorical": "validation.check_category_code",
        "sparse": "validation.check_compressed",
        "booster_params": "validation.check_booster_ranges",
        "objective": "boosting._check_objective",
    }
    return owners.get(what)


__all__ = [
    "MAX_ALLOC_ELEMS",
    "MAX_FEATURES",
    "MAX_ROWS",
    "check_category_tables",
    "check_feature_count_matches",
    "check_feature_names_match",
    "check_fit_structure",
    "check_float64_convertible",
    "check_float_param",
    "check_frame_index_aligned",
    "check_int_param",
    "check_length",
    "check_ndim",
    "check_optional_length",
    "check_param_mapping",
    "check_predict_structure",
    "check_rectangular",
    "check_shape",
    "check_sparse_index_width",
    "check_sparse_layout",
    "describe_domain_owner",
    "domain_checks_are_native",
    "frame_column_names",
    "have_numpy",
    "is_sparse",
]
