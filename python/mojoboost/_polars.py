"""Polars input adapters: DataFrames, Series, and the LazyFrame refusal.

Internal, on the terms of section 2 of `docs/COMPATIBILITY_POLICY.md`.
`docs/ECOSYSTEM_INPUTS.md` is the prose, and
`handoffs/remaining_10_ecosystem_inputs.md` holds the patches that wire this
into `_arrays.check_X`, `Dataset`, and the estimators.

One adapter, not two
--------------------

A polars frame is Arrow underneath, and `DataFrame.to_arrow()` is the door
to it. So this module is deliberately thin: it decides what kind of object
it was handed, it refuses what polars can express and mojoboost cannot
train on, and then it hands an Arrow table to `_arrow`, which owns the
conversion, the null rule, the dictionary unification, the 2**53 integer
check, and the column-major layout. There is exactly one implementation of
each of those in this package and it is not here.

What is left, and it is not nothing, is the part that is genuinely polars:

- **LazyFrames are refused, by name.** A `LazyFrame` has not read anything
  yet. Calling `.collect()` on the caller's behalf would run a query plan
  of unknown cost inside a function they think is validating an argument.
- **Categorical and Enum are not the same declaration.** `Enum` fixes its
  categories when the column is created and the order is the declared one.
  `Categorical` numbers its categories in the order the string cache
  happened to see them, which is a property of the session and not of the
  data, so the same labels in two frames can carry different physical
  codes. Both reach the estimator as *labels* here, never as physical
  codes, which is what makes a model fitted on one frame able to score
  another. `polars_frame_categories` is the seam that does it.
- **The dtype names in the error are polars' names.** A caller who wrote
  `pl.Datetime` should be told about `pl.Datetime`, not about
  `timestamp[us]`. The preflight below exists for that and for nothing
  else: it never refuses a dtype it does not recognize, because `_arrow`
  is the authority on what converts and a second opinion here could only
  drift from it.

Nothing is imported from polars
-------------------------------

Not at import time and not lazily, on the same terms as `_arrow`:
recognition is structural, `polars_available()` uses
`importlib.util.find_spec`, and every operation is a method the object
itself offers (`to_arrow`, `get_column`, `columns`, `dtypes`, `schema`).

`to_arrow()` is cheap, and the rest is not
------------------------------------------

Polars stores its columns in Arrow layout, so `to_arrow()` on a numeric
column without nulls is close to free. That is worth knowing and it is not
zero copy for mojoboost: `_arrow` then materializes every column into a
fresh float64 buffer, and the binding copies that buffer again into a
`List[Float64]`. See the "Zero copy is a contract, not a claim" section of
`_arrow.py`, which applies here unchanged.
"""

from . import _arrow

#: Polars dtype names, as `str(dtype)` spells them (parameters stripped),
#: that reach a numeric Arrow type. Membership here buys a better error
#: message than the Arrow one, and nothing else: the conversion itself is
#: `_arrow`'s.
_NUMERIC_DTYPES = frozenset(
    (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
    )
)

_BOOL_DTYPES = frozenset(("Boolean",))

#: The two category dtypes. They differ in where their category order comes
#: from, which matters for encoders and is spelled out in the module
#: docstring, and not at all in how they convert.
_CATEGORY_DTYPES = frozenset(("Categorical", "Enum"))

_NULL_DTYPES = frozenset(("Null",))

#: Refused polars dtypes, with the polars expression that fixes each. The
#: advice is polars-flavored on purpose: `_arrow` would give correct advice
#: in Arrow's vocabulary, and a polars user should not have to translate it.
_REFUSED_DTYPES = {
    "String": (
        "cast it with .cast(pl.Categorical) or .cast(pl.Enum([...])) and "
        "declare the column categorical, or encode it to integer codes "
        "yourself"
    ),
    "Utf8": (
        "cast it with .cast(pl.Categorical) or .cast(pl.Enum([...])) and "
        "declare the column categorical, or encode it to integer codes "
        "yourself"
    ),
    "Binary": "encode it to numbers yourself; there is no numeric reading "
    "of an opaque byte string",
    "Date": "cast it with .dt.epoch(...) or .cast(pl.Int64), which makes "
    "the epoch and the unit your choice rather than this adapter's",
    "Datetime": "cast it with .dt.epoch('us') or .cast(pl.Int64), which "
    "makes the epoch, the unit, and the time zone your choice rather than "
    "this adapter's",
    "Duration": "cast it with .dt.total_seconds() or .cast(pl.Int64), "
    "which makes the unit your choice",
    "Time": "cast it with .cast(pl.Int64), which makes the unit your "
    "choice",
    "Decimal": "cast it with .cast(pl.Float64), which is the precision "
    "loss you would be accepting anyway, so it should be visible in your "
    "code",
    "Int128": "cast it with .cast(pl.Float64) or .cast(pl.Int64); a "
    "128-bit integer has no exact float64 reading and mojoboost trains on "
    "float64",
    "List": "a feature is one number per row; use .list.to_struct() and "
    ".unnest(), or explode the column into features first",
    "Array": "a feature is one number per row; use .arr.to_struct() and "
    ".unnest() first",
    "Struct": "a feature is one number per row; use .unnest() first",
    "Object": "an object column holds arbitrary Python values; build the "
    "numeric columns you want to train on first",
    "Unknown": "polars could not type this column; build it with an "
    "explicit dtype before passing it",
}


# -- availability and recognition ----------------------------------------


def polars_available():
    """True when polars is importable, without importing it.

    As in `_arrow.arrow_available`, nothing here needs the answer:
    recognition is structural. It exists for diagnostics and for messages
    that want to say "you have no polars installed" rather than "that is
    not a frame".
    """
    from importlib.util import find_spec

    try:
        return find_spec("polars") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken metadata
        return False


def is_polars_frame(obj):
    """True for a polars `DataFrame`.

    `to_arrow` plus `height` and `width` is the distinguishing pair: a
    pandas frame has neither, and a pandas frame is separately excluded by
    `iloc`, which polars does not have and which `_arrays.frame_to_array`
    uses to recognize pandas. Keeping those two apart matters, because both
    answer `columns` and `dtypes` and the pandas path must keep getting the
    pandas frame.
    """
    if hasattr(obj, "iloc"):
        return False
    for attribute in ("to_arrow", "columns", "dtypes", "height", "width"):
        if not hasattr(obj, attribute):
            return False
    return callable(getattr(obj, "to_arrow"))


def is_polars_series(obj):
    """True for a polars `Series`, the shape a label, a weight, or a group
    column arrives in."""
    if hasattr(obj, "iloc") or hasattr(obj, "columns"):
        return False
    for attribute in ("to_arrow", "dtype", "name", "__len__"):
        if not hasattr(obj, attribute):
            return False
    return callable(getattr(obj, "to_arrow"))


def is_polars_lazyframe(obj):
    """True for a polars `LazyFrame`.

    Recognized so it can be refused with the one-word fix rather than
    falling through to "not a supported input". A LazyFrame has `collect`
    and `explain` and, notably, no `to_arrow`.
    """
    if hasattr(obj, "to_arrow"):
        return False
    return hasattr(obj, "collect") and (
        hasattr(obj, "explain") or hasattr(obj, "collect_schema")
    )


def polars_kind(obj):
    """`"frame"`, `"series"`, `"lazyframe"`, or None. One question for a
    dispatcher that would otherwise ask three; `_sequence.adapter_for` is
    the caller."""
    if is_polars_frame(obj):
        return "frame"
    if is_polars_series(obj):
        return "series"
    if is_polars_lazyframe(obj):
        return "lazyframe"
    return None


# -- dtypes ---------------------------------------------------------------


def dtype_name(dtype):
    """The base name of a polars dtype, without its parameters.

    `Categorical(ordering='physical')` is `Categorical`,
    `Datetime(time_unit='us', time_zone=None)` is `Datetime`,
    `List(Int64)` is `List`. Read off `str()` for the same reason `_arrow`
    reads Arrow type names off `str()`: it is the one representation that
    does not require importing the library.
    """
    text = str(dtype)
    for separator in ("(", "["):
        cut = text.find(separator)
        if cut != -1:
            text = text[:cut]
    return text.strip()


def dtype_kind(dtype):
    """How a polars column converts: `"numeric"`, `"bool"`, `"category"`,
    `"null"`, `"refused"`, or `"unknown"`.

    `"unknown"` is not a failure. It means this table has not heard of the
    dtype, in which case `_arrow` decides after `to_arrow()`, which is the
    correct place for the decision and the only place it is made once.
    """
    name = dtype_name(dtype)
    if name in _NUMERIC_DTYPES:
        return "numeric"
    if name in _BOOL_DTYPES:
        return "bool"
    if name in _CATEGORY_DTYPES:
        return "category"
    if name in _NULL_DTYPES:
        return "null"
    if name in _REFUSED_DTYPES:
        return "refused"
    return "unknown"


def _schema_pairs(frame):
    """`[(name, dtype), ...]` for a polars frame, from whichever of the two
    schema spellings this polars has.

    `DataFrame.schema` is a mapping in current polars and was an ordered
    dict before that; `columns` and `dtypes` are the pair that has always
    been there. Trying the mapping first and falling back keeps this
    working across the versions in `pixi.toml` without pinning one.
    """
    schema = getattr(frame, "schema", None)
    if schema is not None:
        try:
            pairs = list(schema.items())
        except AttributeError:
            pairs = None
        if pairs:
            return [(str(name), dtype) for name, dtype in pairs]
    names = [str(name) for name in frame.columns]
    dtypes = list(frame.dtypes)
    return list(zip(names, dtypes))


def preflight(frame, name="X"):
    """Refuse the columns polars can express and mojoboost cannot train on,
    with polars' own vocabulary in the message.

    Runs before `to_arrow()`, so a frame with a `pl.Object` column reports
    the column rather than whatever `to_arrow()` makes of it, and so a
    large frame is not converted before being refused. Every check here is
    a message improvement over the Arrow one, never a different verdict:
    a dtype this does not recognize is left to `_arrow`.
    """
    for index, (column_name, dtype) in enumerate(_schema_pairs(frame)):
        kind = dtype_kind(dtype)
        if kind != "refused":
            continue
        advice = _REFUSED_DTYPES[dtype_name(dtype)]
        raise ValueError(
            f"{name} column {index} ({column_name!r}) has polars dtype "
            f"{dtype}, which mojoboost does not read as a feature: {advice}"
        )


def describe_schema(frame):
    """`[(index, name, dtype string, kind), ...]` for a polars frame.

    Diagnostics, and the table `docs/ECOSYSTEM_INPUTS.md` is written
    against. Converts nothing.
    """
    return [
        (index, column_name, str(dtype), dtype_kind(dtype))
        for index, (column_name, dtype) in enumerate(_schema_pairs(frame))
    ]


# -- the matrix path ------------------------------------------------------


def _require_frame(X, name):
    """The frame, or the error that names what was passed instead."""
    if is_polars_lazyframe(X):
        raise TypeError(
            f"{name} is a polars LazyFrame, which has not read any data "
            "yet; call .collect() and pass the DataFrame, so that the cost "
            "of running your query plan is in your code rather than inside "
            "an argument check"
        )
    if not is_polars_frame(X):
        raise TypeError(f"{name} is not a polars DataFrame")
    return X


def to_arrow_table(X, name="X"):
    """A polars frame as an Arrow table, after the polars-side refusals.

    The single door from this module to `_arrow`. Everything that follows
    a call to this function is Arrow's problem, which is what keeps the two
    adapters from growing two answers to the same question.
    """
    _require_frame(X, name)
    preflight(X, name)
    table = X.to_arrow()
    if not _arrow.is_arrow_table(table):
        raise TypeError(
            f"{name}.to_arrow() did not return an Arrow table; this "
            "adapter converts polars frames through Arrow and cannot "
            f"convert a {type(table).__name__}"
        )
    return table


def polars_feature_names(X):
    """Column names for a polars frame, or None for anything else.

    The counterpart of `_arrays.feature_names`. Polars forbids duplicate
    column names in a frame, so the uniqueness check `_arrow` has to make
    for Arrow tables cannot fire on this path.
    """
    if not is_polars_frame(X):
        return None
    names = [str(name) for name in X.columns]
    return names or None


def polars_frame_categories(X):
    """`{column_index: [label, ...]}` for the `Categorical` and `Enum`
    columns of a polars frame, in column order; empty for anything else.

    The counterpart of `_arrays.frame_categories`, and it carries the same
    meaning into the same two places: `categorical_feature="auto"` means
    exactly these columns, and a prediction frame's categories are checked
    against the fitted ones.

    The labels come from the Arrow dictionary rather than from
    `Series.cat.get_categories()`, which is what makes `Categorical` safe:
    physical codes depend on the string cache and the labels do not, so a
    model fitted on one frame scores another frame's `Categorical` column
    correctly even when the two sessions numbered the same strings
    differently. That is the same reason `_arrow` encodes a prediction
    table through the fitted labels instead of through its own dictionary.
    """
    if not is_polars_frame(X):
        return {}
    return _arrow.arrow_frame_categories(to_arrow_table(X))


def column_major_polars(X, name="X", encoders=None):
    """`(buffer, n_rows, n_features)` for a polars frame.

    The polars counterpart of `_arrays.column_major`, with the same
    contract: the caller keeps the returned buffer referenced while its
    address is in flight, and the frame itself is not part of that
    lifetime because every value has been copied out of it.
    """
    return _arrow.column_major_arrow(
        to_arrow_table(X, name), name, encoders
    )


def check_X_polars(X, name="X", encoders=None):
    """`(buffer, n_rows, n_features, names)` for a polars frame.

    The polars counterpart of `_arrays.check_X`, returning the same
    4-tuple in the same order. The names come from the frame rather than
    from the Arrow table, which is the same list; reading them from the
    frame keeps the answer available even if a future polars changes what
    `to_arrow()` names an unnamed column.
    """
    names = polars_feature_names(X)
    buf, n_rows, n_features = column_major_polars(X, name, encoders)
    return buf, n_rows, n_features, names


# -- the column path ------------------------------------------------------


def polars_f64_vector(values, n_rows, name="y"):
    """A polars Series as a float64 vector of length `n_rows`.

    The counterpart of `_arrays.f64_vector`, for `label`, `weight`,
    `init_score`, and `group`. Nulls become NaN, and a NaN in a field that
    may not hold one is refused by `_arrays.check_target` or
    `check_sample_weight` a moment later, with the message that names the
    field.
    """
    if not is_polars_series(values):
        raise TypeError(f"{name} is not a polars Series")
    kind = dtype_kind(values.dtype)
    if kind == "refused":
        raise ValueError(
            f"{name} has polars dtype {values.dtype}, which is not a "
            f"numeric column: {_REFUSED_DTYPES[dtype_name(values.dtype)]}"
        )
    if kind == "category":
        raise ValueError(
            f"{name} has polars dtype {values.dtype}; pass the labels "
            "themselves (a classifier encodes them) or their codes as an "
            "integer column, because a physical category code is a "
            "property of the string cache and not of the data"
        )
    return _arrow.arrow_f64_vector(values.to_arrow(), n_rows, name)


def polars_to_pylist(values, name="y"):
    """A polars Series as a Python list, nulls included as None.

    For `_arrays.encode_labels`, which classifies on the labels as passed:
    a `pl.Enum` label column should give `classes_` holding your category
    names, and it does when the labels arrive as labels. `to_list()` is
    polars' own spelling and is used in preference to a round trip through
    Arrow because it is the shorter path to the same list.
    """
    if not is_polars_series(values):
        raise TypeError(f"{name} is not a polars Series")
    to_list = getattr(values, "to_list", None)
    if to_list is None:  # pragma: no cover - every polars Series has one
        return _arrow.arrow_to_pylist(values.to_arrow(), name)
    return list(to_list())
