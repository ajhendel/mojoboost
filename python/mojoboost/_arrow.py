"""Apache Arrow input adapters: tables, record batches, and arrays.

Internal, on the terms of section 2 of `docs/COMPATIBILITY_POLICY.md`, which
puts every underscore-prefixed module in `python/mojoboost/` outside the
public surface in its entirety. `docs/ECOSYSTEM_INPUTS.md` is the prose
version of everything here, and `handoffs/remaining_10_ecosystem_inputs.md`
holds the patches that would wire it into `_arrays.check_X`, `Dataset`, and
the estimators.

What this module is
-------------------

`_arrays.py` takes a feature matrix apart into the one thing the extension
module reads: a contiguous float64 buffer, column-major, whose address is
handed across the boundary. It knows numpy arrays, pandas frames, plain
sequences, and SciPy sparse matrices. This module adds Arrow to that list,
in exactly the same currency, so that an Arrow table reaches the binner
through the same validation the dense path already applies rather than
through a second one.

Every entry point here mirrors an `_arrays` entry point by name and by
return shape, which is what makes the integration a dispatch rather than a
rewrite:

| `_arrays`          | here                    |
| ------------------ | ----------------------- |
| `feature_names`    | `arrow_feature_names`   |
| `frame_categories` | `arrow_frame_categories`|
| `check_X`          | `check_X_arrow`         |
| `f64_vector`       | `arrow_f64_vector`      |

Nothing is imported from pyarrow
--------------------------------

Not at import time and not lazily. Every check here asks the object what it
offers rather than asking the library what the object is, the way
`mojoboost.dask.is_dask_collection()` does, so this module is import safe on
an install with no pyarrow and it costs nothing on `import mojoboost`.
`arrow_available()` answers the "is it installed" question with
`importlib.util.find_spec`, which does not execute the package either.

The practical consequence is that this module works against the Arrow
*protocol*: `Table.column`, `ChunkedArray.chunk`, `Array.to_pylist`,
`Array.buffers`, `DictionaryArray.dictionary`, `DataType.__str__`. Anything
implementing that protocol works, which includes `pyarrow`, and the adapter
does not have to track a pyarrow version to keep doing so.

Zero copy is a contract, not a claim
------------------------------------

Arrow's whole point is that a column is already a flat, typed, contiguous
buffer, and a caller reading this module will reasonably wonder whether
mojoboost trains on it in place. It does not, and this module never says it
does. Two separate facts stand in the way, and both are recorded per column
in `BufferPlan` rather than assumed away:

1. **The Arrow side may or may not be eligible.** A column is a candidate
   only if it is a single chunk of `double` values, at offset 0, with no
   validity bitmap or with a bitmap the reader is prepared to honor.
   Chunking, a narrower numeric type, a dictionary encoding, and a slice
   offset each rule it out. `describe_columns()` reports which of those
   applies, per column, with the address and length the column does expose.

2. **The mojoboost side is not ready for it in any case.** The binding
   entry point (`dataset_create` in `bindings/_mojoboost.mojo`) takes one
   address for the whole matrix and immediately copies it element by
   element into a `List[Float64]` through `_f64_list`. There is no path
   today that reads N per-column pointers, and no path that reads a
   validity bitmap. So even a perfectly eligible Arrow column is copied
   twice on the way in: once here, into the column-major buffer, and once
   in Mojo. The handoff carries the binding request that would remove the
   second copy and then the first.

`BufferPlan.zero_copy_eligible` therefore means "the Arrow buffer could be
read in place by a reader that accepted per-column pointers", and says so in
its docstring. It never means "mojoboost read it in place".

Which types convert, and which are refused
------------------------------------------

The rule is that a conversion that could change a value, or that could pick
a unit on the caller's behalf, is refused rather than performed:

- **Signed and unsigned integers, floats, and half floats** convert.
  `int64` and `uint64` are checked against 2**53 first, because past that a
  float64 cannot hold the value and the binner would split on a number the
  caller never supplied.
- **Booleans** convert to 0.0 and 1.0.
- **Dictionary columns** become their integer codes, under the same rule
  the pandas `category` path uses in `_arrays._codes_from_labels`: a
  fitted category table wins over the column's own, and a value the table
  does not hold becomes -1, which the binner reads as the unknown
  category. Chunks carrying different dictionaries are unified into one
  table, which is a case pandas cannot produce and Arrow can.
- **Null-typed columns** become all-NaN. That is not a lossy conversion,
  it is the exact one: a `null` column holds nothing but nulls, and NaN is
  mojoboost's missing marker.
- **Strings, binary, temporal, decimal, interval, and every nested type**
  are refused, each with the cast that would make it acceptable. A
  timestamp is the clearest case: casting it to `int64` is a choice of
  epoch and unit, and it is the caller's choice, not this module's.

Nulls
-----

A null becomes NaN, which is mojoboost's missing marker end to end (see
`src/mojoboost/binning.mojo`, which reserves a bin for it, and the
`use_missing` parameter). This is the one place where Arrow and the dense
path genuinely agree already: a numpy float column of NaN and an Arrow
column of nulls describe the same rows to the binner.

Ownership and lifetime
----------------------

Because every column is materialized into a buffer this module allocates,
the returned buffer does not borrow from the Arrow object at all, and the
table may be dropped as soon as `check_X_arrow` returns. That is a weaker
promise than Arrow could support and a much simpler one to keep: the caller
holds the returned buffer for the duration of the binding call, exactly as
`_arrays.column_major`'s caller does, and holds nothing else.
"""

import array as _array
import math

from . import _arrays

_np = _arrays.np

#: Beyond this an integer is no longer exactly representable in float64.
#: `int64` and `uint64` are the only Arrow integer types that can reach it.
INT_EXACT_LIMIT = 1 << 53

#: The largest category code the binner accepts, mirroring
#: `mojoboost._Base._CATEGORY_LIMIT`. A dictionary with more entries than
#: this cannot be encoded, and saying so here is better than letting
#: `_check_category_codes` report it as a bad value later.
CATEGORY_LIMIT = 1 << 31

#: Arrow type names, as `str(DataType)` spells them, that convert to
#: float64 without changing a value. `str(pa.float16())` is `halffloat`,
#: `str(pa.float32())` is `float`, and `str(pa.float64())` is `double`,
#: which is why this is a table rather than a prefix test.
_NUMERIC_TYPES = frozenset(
    (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "halffloat",
        "float",
        "double",
    )
)

#: The two that need the 2**53 check before they can be trusted in float64.
_WIDE_INT_TYPES = frozenset(("int64", "uint64"))

_BOOL_TYPES = frozenset(("bool", "boolean"))

#: Refused types, by base name, with what the caller can do about each. The
#: message is the whole value of refusing: "unsupported type" tells nobody
#: anything, and every one of these has a cast that makes it work.
_REFUSED_TYPES = {
    "string": (
        "cast it to dictionary and declare the column categorical "
        "(pa.compute.dictionary_encode, or Table.cast with a "
        "pa.dictionary(pa.int32(), pa.string()) field), or encode it to "
        "integer codes yourself"
    ),
    "large_string": (
        "cast it to dictionary and declare the column categorical, or "
        "encode it to integer codes yourself"
    ),
    "string_view": (
        "cast it to dictionary and declare the column categorical, or "
        "encode it to integer codes yourself"
    ),
    "binary": "encode it to numbers yourself; there is no numeric reading "
    "of an opaque byte string",
    "large_binary": "encode it to numbers yourself",
    "binary_view": "encode it to numbers yourself",
    "fixed_size_binary": "encode it to numbers yourself",
    "date32": "cast it to int64 first, which fixes the epoch and the unit "
    "as your choice rather than this adapter's",
    "date64": "cast it to int64 first, which fixes the epoch and the unit "
    "as your choice rather than this adapter's",
    "timestamp": "cast it to int64 first, which fixes the epoch, the unit, "
    "and the time zone as your choice rather than this adapter's",
    "time32": "cast it to int32 or int64 first, which fixes the unit as "
    "your choice",
    "time64": "cast it to int64 first, which fixes the unit as your choice",
    "duration": "cast it to int64 first, which fixes the unit as your "
    "choice",
    "interval": "there is no single number a calendar interval converts "
    "to; split it into the components you want to train on",
    "month_day_nano_interval": "there is no single number a calendar "
    "interval converts to; split it into the components you want to train "
    "on",
    "decimal": "cast it to double, which is the precision loss you would "
    "be accepting anyway, so it should be visible in your code",
    "decimal32": "cast it to double, which is the precision loss you "
    "would be accepting anyway",
    "decimal64": "cast it to double, which is the precision loss you "
    "would be accepting anyway",
    "decimal128": "cast it to double, which is the precision loss you "
    "would be accepting anyway",
    "decimal256": "cast it to double, which is the precision loss you "
    "would be accepting anyway",
    "list": "a feature is one number per row; flatten the list into "
    "columns first",
    "large_list": "a feature is one number per row; flatten the list into "
    "columns first",
    "fixed_size_list": "a feature is one number per row; flatten the list "
    "into columns first",
    "list_view": "a feature is one number per row; flatten the list into "
    "columns first",
    "large_list_view": "a feature is one number per row; flatten the list "
    "into columns first",
    "map": "a feature is one number per row; flatten the map into columns "
    "first",
    "struct": "a feature is one number per row; flatten the struct into "
    "columns first",
    "union": "a column with one type per row has no single conversion; "
    "split it into one column per type",
    "sparse_union": "a column with one type per row has no single "
    "conversion; split it into one column per type",
    "dense_union": "a column with one type per row has no single "
    "conversion; split it into one column per type",
    "run_end_encoded": "call .combine_chunks() or cast it to its value "
    "type; run-end encoding is a layout this adapter does not decode",
    "extension": "an extension type carries meaning this adapter cannot "
    "read; use .storage to hand over the underlying column",
}


# -- availability and recognition ----------------------------------------


def arrow_available():
    """True when pyarrow is importable, without importing it.

    Nothing in this module needs the answer: recognition is structural, so
    an object that walks like an Arrow table is treated as one whether or
    not pyarrow is the thing that made it. This exists for diagnostics and
    for error messages that want to distinguish "you passed something odd"
    from "you have no Arrow installed".
    """
    from importlib.util import find_spec

    try:
        return find_spec("pyarrow") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken metadata
        return False


def _looks_like_type(obj):
    """True for an Arrow `DataType`, which every column exposes as `.type`
    and which is the object this module classifies."""
    return hasattr(obj, "id") and hasattr(obj, "num_fields")


def is_arrow_array(obj):
    """True for an Arrow `Array` or `ChunkedArray`.

    An Array carries `buffers()`, a ChunkedArray carries `num_chunks`, and
    both carry a `type` and a `null_count`. Requiring the type object to
    look like an Arrow type as well is what keeps a numpy array, which also
    has a `.type`-free `dtype`, and a pandas Series, which has neither,
    from matching by accident.
    """
    if not hasattr(obj, "type") or not hasattr(obj, "null_count"):
        return False
    if not _looks_like_type(getattr(obj, "type")):
        return False
    return hasattr(obj, "buffers") or hasattr(obj, "num_chunks")


def is_arrow_chunked(obj):
    """True for a `ChunkedArray`, the multi-chunk column of a Table."""
    return (
        is_arrow_array(obj)
        and hasattr(obj, "num_chunks")
        and hasattr(obj, "chunk")
    )


def is_arrow_table(obj):
    """True for an Arrow `Table` or `RecordBatch`.

    Both answer `column(i)`, `column_names`, `num_rows`, and `num_columns`,
    and the difference between them (a Table's columns are chunked, a
    RecordBatch's are not) is handled by `_chunks_of` rather than by
    branching on which one this is.
    """
    for attribute in ("column", "column_names", "num_rows", "num_columns"):
        if not hasattr(obj, attribute):
            return False
    return callable(getattr(obj, "column"))


def is_arrow_dataset(obj):
    """True for a `pyarrow.dataset.Dataset` or a `RecordBatchReader`, the
    two lazy Arrow inputs that are not tables.

    Recognized so they can be refused precisely. Both stream, and streaming
    into a bounded-memory binner is `_sequence.py`'s subject and the native
    core's (`src/mojoboost/sequence.mojo`, task 07); neither is something
    this module should quietly materialize.
    """
    if is_arrow_table(obj):
        return False
    if hasattr(obj, "to_batches") and hasattr(obj, "schema"):
        return True
    return hasattr(obj, "read_next_batch") and hasattr(obj, "schema")


def is_arrow_sparse_tensor(obj):
    """True for one of Arrow's sparse tensor types (`SparseCSRMatrix`,
    `SparseCSCMatrix`, `SparseCOOTensor`, `SparseCSFTensor`).

    They are not table-like and do not convert here. `sparse_to_scipy`
    turns the two that mojoboost's sparse path can use into what that path
    takes.
    """
    return (
        hasattr(obj, "to_scipy")
        and hasattr(obj, "shape")
        and hasattr(obj, "non_zero_length")
    )


def arrow_kind(obj):
    """`"table"`, `"array"`, `"dataset"`, `"sparse_tensor"`, or None.

    One question, one answer, for a dispatcher that would otherwise ask
    four. `_sequence.adapter_for` is the caller.
    """
    if is_arrow_table(obj):
        return "table"
    if is_arrow_array(obj):
        return "array"
    if is_arrow_sparse_tensor(obj):
        return "sparse_tensor"
    if is_arrow_dataset(obj):
        return "dataset"
    return None


# -- type classification --------------------------------------------------


def type_name(arrow_type):
    """The base name of an Arrow type, without its parameters.

    `timestamp[us, tz=UTC]` is `timestamp`, `decimal128(10, 2)` is
    `decimal128`, `list<item: int64>` is `list`. Reading the name off
    `str()` rather than off `pa.types.is_*` is what keeps this module from
    importing pyarrow; the names are part of Arrow's printed form and are
    as stable as the type system itself.
    """
    text = str(arrow_type)
    for separator in ("[", "<", "("):
        cut = text.find(separator)
        if cut != -1:
            text = text[:cut]
    return text.strip()


def is_dictionary_type(arrow_type):
    """True for a dictionary type, asked structurally.

    A dictionary type is the only one carrying both a `value_type` and an
    `index_type`, so this does not depend on how the type prints.
    """
    return hasattr(arrow_type, "value_type") and hasattr(
        arrow_type, "index_type"
    )


def column_kind(arrow_type):
    """How a column converts: `"numeric"`, `"bool"`, `"dictionary"`,
    `"null"`, or `"refused"`."""
    if is_dictionary_type(arrow_type):
        return "dictionary"
    name = type_name(arrow_type)
    if name in _NUMERIC_TYPES:
        return "numeric"
    if name in _BOOL_TYPES:
        return "bool"
    if name == "null":
        return "null"
    return "refused"


def _refusal(arrow_type, column_name, index, matrix_name):
    """The message a refused column raises with."""
    name = type_name(arrow_type)
    advice = _REFUSED_TYPES.get(name)
    if advice is None:
        advice = (
            "convert it to a numeric or dictionary column; this adapter "
            "converts integers, floats, booleans, dictionaries, and nulls"
        )
    return ValueError(
        f"{matrix_name} column {index} ({column_name!r}) has Arrow type "
        f"{arrow_type}, which mojoboost does not read as a feature: {advice}"
    )


# -- structure ------------------------------------------------------------


def _chunks_of(column):
    """A column as a list of contiguous Arrow arrays.

    A Table's column is a ChunkedArray and may be any number of them,
    including zero for an empty table. A RecordBatch's column is a single
    Array and is its own only chunk.
    """
    if hasattr(column, "num_chunks") and hasattr(column, "chunk"):
        return [column.chunk(i) for i in range(int(column.num_chunks))]
    return [column]


def _row_count(table):
    """Rows in a Table or RecordBatch."""
    count = getattr(table, "num_rows", None)
    if count is None:
        return len(table)
    return int(count)


def _columns_of(table):
    """`[(name, column), ...]` in schema order."""
    names = [str(name) for name in table.column_names]
    return [(names[i], table.column(i)) for i in range(len(names))]


def arrow_feature_names(X):
    """Column names for an Arrow table, or None for anything else.

    The counterpart of `_arrays.feature_names`, and deliberately stricter
    in the same way: names that are not all strings contribute nothing, so
    `feature_names_in_` never holds a mixture. Arrow schema names are
    always strings, so in practice this is "the names, or None because this
    is not a table".
    """
    if not is_arrow_table(X):
        return None
    names = [name for name in X.column_names]
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return list(names)


# -- dictionary columns ---------------------------------------------------


def _hashable(value):
    """`value` if it can be a dictionary key, else a marker that is not
    equal to anything a real category could be.

    Mirrors the `TypeError` guard in `_arrays._codes_from_labels`: an
    unhashable label cannot be looked up, so it can only be unknown.
    """
    try:
        hash(value)
    except TypeError:
        return _Unhashable(value)
    return value


class _Unhashable:
    """A stand-in for a label that cannot be a dict key. Two of these are
    never equal, which is exactly the "no category matches" answer."""

    __slots__ = ("kind",)

    def __init__(self, value):
        self.kind = type(value).__name__

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return False


def _unify_dictionaries(chunks, column_name, index, matrix_name):
    """`(labels, remaps)` for a possibly multi-chunk dictionary column.

    Arrow permits every chunk of a ChunkedArray to carry its own
    dictionary, and a table read from several files or several row groups
    routinely does. `labels` is the union in first-appearance order and
    `remaps[c][i]` is the code chunk `c`'s local index `i` maps to, so the
    codes this module emits are comparable across the whole column.
    Unifying is not an optimization: without it, code 3 would mean a
    different category in chunk 2 than in chunk 1, and the binner would
    split on the union of two unrelated categories.

    A null *inside* a dictionary's values (a category that is itself null)
    remaps to -1, the same answer a null index gets, because both mean the
    row has no category.
    """
    labels = []
    position = {}
    remaps = []
    for chunk in chunks:
        values = _dictionary_values(chunk, column_name, index, matrix_name)
        remap = []
        for value in values:
            if value is None:
                remap.append(-1)
                continue
            key = _hashable(value)
            slot = position.get(key)
            if slot is None:
                slot = len(labels)
                position[key] = slot
                labels.append(value)
            remap.append(slot)
        remaps.append(remap)
    if len(labels) >= CATEGORY_LIMIT:
        raise ValueError(
            f"{matrix_name} column {index} ({column_name!r}) has "
            f"{len(labels)} dictionary values, and a category code must be "
            f"below {CATEGORY_LIMIT}"
        )
    return labels, remaps


def _dictionary_values(chunk, column_name, index, matrix_name):
    """A dictionary chunk's category labels as a Python list."""
    dictionary = getattr(chunk, "dictionary", None)
    if dictionary is None:
        raise ValueError(
            f"{matrix_name} column {index} ({column_name!r}) has a "
            "dictionary type but a chunk that carries no dictionary; "
            "call .combine_chunks() on the table first"
        )
    return list(dictionary.to_pylist())


def _fitted_remaps(chunks, categories, column_name, index, matrix_name):
    """`remaps` for a dictionary column encoded through a *fitted* category
    table rather than its own.

    This is the prediction-time path, and it is the whole reason the
    category tables are recorded at fit time: a prediction table may order
    its dictionary differently, may hold extra values, and may be missing
    some, and none of that may move a category. A value the fitted table
    does not hold becomes -1, the unknown category, exactly as
    `_arrays._codes_from_labels` does for pandas.
    """
    fitted = {}
    for position, label in enumerate(categories):
        fitted[_hashable(label)] = position
    remaps = []
    for chunk in chunks:
        values = _dictionary_values(chunk, column_name, index, matrix_name)
        remaps.append(
            [
                -1 if value is None else fitted.get(_hashable(value), -1)
                for value in values
            ]
        )
    return remaps


def arrow_frame_categories(X):
    """`{column_index: [label, ...]}` for the dictionary columns of an
    Arrow table, in column order; empty for anything else.

    The counterpart of `_arrays.frame_categories`, and it feeds the same
    two things: `_Base._resolve_categorical`, where `categorical_feature=
    "auto"` means exactly these columns, and `_Base._matrix_encoders`,
    where a prediction table's dictionary columns are checked against the
    fitted ones. A pandas `category` column and an Arrow `dictionary`
    column are the same declaration in two libraries, and they should
    reach the estimator as the same fact.
    """
    if not is_arrow_table(X):
        return {}
    out = {}
    for index, (name, column) in enumerate(_columns_of(X)):
        if not is_dictionary_type(column.type):
            continue
        labels, _ = _unify_dictionaries(
            _chunks_of(column), name, index, "X"
        )
        out[index] = labels
    return out


# -- value conversion -----------------------------------------------------


def _check_wide_int_chunk(values, column_name, index, matrix_name):
    """Refuse an int64 or uint64 chunk holding a value float64 cannot
    represent exactly.

    Past 2**53 the conversion is lossy in a way that matters: two adjacent
    identifiers collapse onto one float, and the binner puts them in the
    same bin. LightGBM has the same exposure through numpy and does not
    check; mojoboost refuses, on the same principle that refuses
    infinities in `_arrays._as_column_major_numpy`.
    """
    limit = INT_EXACT_LIMIT
    if _np is not None and hasattr(values, "dtype"):
        if values.size == 0:
            return
        if not ((values > limit).any() or (values < -limit).any()):
            return
        offender = None
        for candidate in (values > limit), (values < -limit):
            found = _np.flatnonzero(candidate)
            if found.size:
                offender = int(values[found[0]])
                break
    else:
        offender = None
        for value in values:
            if value is None:
                continue
            if value > limit or value < -limit:
                offender = int(value)
                break
        if offender is None:
            return
    raise ValueError(
        f"{matrix_name} column {index} ({column_name!r}) holds {offender}, "
        f"which float64 cannot represent exactly (the limit is {limit}); "
        "mojoboost converts every feature to float64, so cast the column "
        "to double yourself if the rounding is acceptable, or scale it"
    )


def _chunk_to_numpy(chunk):
    """A null-free chunk as a numpy array in its own dtype, or None when
    Arrow will not produce one.

    Only ever called for chunks with `null_count == 0`, where `to_numpy`
    has an unambiguous answer. A chunk with nulls goes through
    `to_pylist()` instead, because what `to_numpy` does with a null
    depends on the type (a float column gets NaN, an integer column gets
    an object array or an error), and this module cannot afford to be
    unsure which of those it received.
    """
    if _np is None:
        return None
    to_numpy = getattr(chunk, "to_numpy", None)
    if to_numpy is None:
        return None
    try:
        values = to_numpy(zero_copy_only=False)
    except TypeError:
        try:
            values = to_numpy()
        except Exception:
            return None
    except Exception:
        return None
    if getattr(values, "dtype", None) is None:
        return None
    if values.dtype == object:
        return None
    return values


def _write_numeric_chunk(
    out, offset, chunk, wide_int, column_name, index, matrix_name
):
    """Write one numeric or boolean chunk into `out` at `offset`, as
    float64, with nulls as NaN. Returns the number of rows written."""
    n = len(chunk)
    if n == 0:
        return 0
    if int(chunk.null_count) == 0:
        values = _chunk_to_numpy(chunk)
        if values is not None:
            if wide_int:
                _check_wide_int_chunk(
                    values, column_name, index, matrix_name
                )
            out[offset : offset + n] = values.astype(_np.float64, copy=False)
            return n
    values = chunk.to_pylist()
    if wide_int:
        _check_wide_int_chunk(values, column_name, index, matrix_name)
    for position, value in enumerate(values):
        out[offset + position] = (
            float("nan") if value is None else float(value)
        )
    return n


def _write_dictionary_chunk(out, offset, chunk, remap):
    """Write one dictionary chunk into `out` at `offset` as category codes,
    with a null index and a null category alike as -1."""
    n = len(chunk)
    if n == 0:
        return 0
    indices = chunk.indices.to_pylist()
    for position, code in enumerate(indices):
        out[offset + position] = (
            -1.0 if code is None else float(remap[code])
        )
    return n


def _new_column(n_rows):
    """A float64 scratch column of `n_rows`, numpy or not."""
    if _np is not None:
        return _np.empty(n_rows, dtype=_np.float64)
    return _array.array("d", bytes(8 * n_rows))


def column_values(column, n_rows, name="column", index=0, categories=None,
                  matrix_name="X"):
    """One Arrow column as a float64 buffer of length `n_rows`.

    `categories`, when given, is a fitted category table and the column
    must be a dictionary one: its values are encoded through that table
    rather than through its own. This is the only difference between the
    fit path and the predict path, and it is the difference that keeps a
    category meaning the same thing in both.
    """
    arrow_type = column.type
    kind = column_kind(arrow_type)
    if categories is not None and kind != "dictionary":
        raise ValueError(
            f"{matrix_name} column {index} ({name!r}) has Arrow type "
            f"{arrow_type}, but this model was fitted with a category "
            "table for that column; pass a dictionary column, or pass its "
            "integer codes as a numeric column"
        )
    if kind == "refused":
        raise _refusal(arrow_type, name, index, matrix_name)
    out = _new_column(n_rows)
    chunks = _chunks_of(column)
    written = 0
    if kind == "null":
        # A null-typed column holds nothing but nulls, so this is the exact
        # conversion rather than a lossy one: every row is missing.
        if _np is not None and hasattr(out, "fill"):
            out.fill(float("nan"))
        else:
            for position in range(n_rows):
                out[position] = float("nan")
        return out
    if kind == "dictionary":
        if categories is None:
            _, remaps = _unify_dictionaries(chunks, name, index, matrix_name)
        else:
            remaps = _fitted_remaps(
                chunks, categories, name, index, matrix_name
            )
        for chunk, remap in zip(chunks, remaps):
            written += _write_dictionary_chunk(out, written, chunk, remap)
    else:
        wide_int = type_name(arrow_type) in _WIDE_INT_TYPES
        for chunk in chunks:
            written += _write_numeric_chunk(
                out, written, chunk, wide_int, name, index, matrix_name
            )
    if written != n_rows:
        raise ValueError(
            f"{matrix_name} column {index} ({name!r}) has {written} rows "
            f"but the table has {n_rows}"
        )
    return out


def _require_no_infinities(values, name, index, matrix_name):
    """Refuse an infinite feature value, matching the dense path.

    `_arrays` rejects infinities in `X` on the grounds that LightGBM's own
    wrapper lets them into the binner as extreme finite values by accident.
    An Arrow column reaches the same binner, so it answers to the same rule.
    """
    if _np is not None and hasattr(values, "dtype"):
        if values.size and _np.isinf(values).any():
            bad = True
        else:
            bad = False
    else:
        bad = any(math.isinf(value) for value in values)
    if bad:
        raise ValueError(
            f"{matrix_name} column {index} ({name!r}) contains infinite "
            "values; NaN is allowed and is treated as missing, but an "
            "infinity is not a value the binner can place"
        )


# -- buffer description ---------------------------------------------------


class BufferPlan:
    """What one Arrow column would offer a reader that read it in place.

    A description, and only a description. Nothing in mojoboost reads an
    Arrow buffer today (see the module docstring), and this class exists so
    that the requirements of doing so are written down in terms of the
    actual columns a caller passes rather than in the abstract:

    - `values_addr` / `values_length`: the values buffer of the single
      chunk, when there is a single chunk. Zero and zero otherwise.
    - `validity_addr`: the validity bitmap, or 0 when the chunk has none.
      A reader honoring this would not need the NaN materialization this
      module performs, which is the larger half of the copy.
    - `offset`: Arrow slices are views with a nonzero offset, and a reader
      that ignored it would read the wrong rows.
    - `zero_copy_eligible`: whether the *Arrow* side could be read in
      place, which is a strictly weaker statement than "mojoboost reads it
      in place". `blocked_by` says which condition failed.
    """

    __slots__ = (
        "index",
        "name",
        "arrow_type",
        "kind",
        "n_chunks",
        "n_rows",
        "null_count",
        "values_addr",
        "values_length",
        "validity_addr",
        "offset",
        "zero_copy_eligible",
        "blocked_by",
        "dictionary_size",
    )

    def __init__(self, **fields):
        for slot in self.__slots__:
            setattr(self, slot, fields.get(slot))

    def as_dict(self):
        """The plan as a plain dict, for diagnostics and for a handoff that
        wants to quote it."""
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self):
        state = (
            "eligible"
            if self.zero_copy_eligible
            else f"copied ({self.blocked_by})"
        )
        return (
            f"BufferPlan({self.name!r}, {self.arrow_type}, "
            f"{self.n_chunks} chunk(s), {state})"
        )


def _buffer_addresses(chunk):
    """`(validity_addr, values_addr, values_length)` for one chunk, or
    zeros when the layout is not the two-buffer numeric one."""
    buffers = getattr(chunk, "buffers", None)
    if buffers is None:
        return 0, 0, 0
    try:
        parts = buffers()
    except Exception:  # pragma: no cover - a layout with no buffer view
        return 0, 0, 0
    if len(parts) < 2:
        return 0, 0, 0
    validity, values = parts[0], parts[1]
    return (
        0 if validity is None else int(validity.address),
        0 if values is None else int(values.address),
        0 if values is None else int(values.size),
    )


def describe_columns(X):
    """A `BufferPlan` per column of an Arrow table.

    The honest inventory of what a table offers, which is what
    `docs/ECOSYSTEM_INPUTS.md` quotes and what a future in-place reader
    would be written against. It converts nothing and allocates nothing
    beyond the plans themselves.
    """
    if not is_arrow_table(X):
        raise TypeError("describe_columns takes an Arrow table")
    n_rows = _row_count(X)
    plans = []
    for index, (name, column) in enumerate(_columns_of(X)):
        arrow_type = column.type
        kind = column_kind(arrow_type)
        chunks = _chunks_of(column)
        validity_addr = values_addr = values_length = 0
        offset = 0
        blocked = None
        if len(chunks) != 1:
            blocked = f"{len(chunks)} chunks"
        else:
            chunk = chunks[0]
            offset = int(getattr(chunk, "offset", 0) or 0)
            validity_addr, values_addr, values_length = _buffer_addresses(
                chunk
            )
            if kind == "refused":
                blocked = f"unsupported type {arrow_type}"
            elif kind == "dictionary":
                blocked = "dictionary encoding needs code materialization"
            elif kind == "null":
                blocked = "null type has no values buffer"
            elif type_name(arrow_type) != "double":
                blocked = f"{arrow_type} is not double"
            elif offset != 0:
                blocked = f"sliced view at offset {offset}"
            elif values_addr == 0:
                blocked = "no values buffer"
        dictionary_size = None
        if kind == "dictionary" and chunks:
            try:
                labels, _ = _unify_dictionaries(chunks, name, index, "X")
                dictionary_size = len(labels)
            except ValueError:
                dictionary_size = None
        plans.append(
            BufferPlan(
                index=index,
                name=name,
                arrow_type=str(arrow_type),
                kind=kind,
                n_chunks=len(chunks),
                n_rows=n_rows,
                null_count=int(column.null_count),
                values_addr=values_addr,
                values_length=values_length,
                validity_addr=validity_addr,
                offset=offset,
                zero_copy_eligible=blocked is None,
                blocked_by=blocked,
                dictionary_size=dictionary_size,
            )
        )
    return plans


# -- the matrix path ------------------------------------------------------


def _require_unique_names(names, matrix_name):
    """Refuse a schema with a repeated field name.

    Arrow permits duplicate field names and a join produces them routinely,
    which pandas makes much harder to do by accident. They cannot be
    allowed through: `categorical_feature=["price"]` resolves a name to a
    position with `names.index(...)`, so with two columns called `price`
    the declaration would land on the first one and quietly leave the
    second numeric. This is one of the two places the Arrow path is
    stricter than the pandas path, and `docs/ECOSYSTEM_INPUTS.md` records
    both.
    """
    seen = set()
    for position, name in enumerate(names):
        if name in seen:
            raise ValueError(
                f"{matrix_name} has more than one column named {name!r} "
                f"(the second is at index {position}); feature names "
                "resolve categorical declarations and must be unique, so "
                "rename the columns before passing the table"
            )
        seen.add(name)


def _allocate_matrix(n_rows, n_features):
    """The column-major float64 buffer the binding reads, in whichever of
    the two shapes `_arrays` uses: a Fortran-ordered 2-D numpy array when
    numpy is present, a flat `array.array` otherwise. Both answer
    `_arrays.addr` and `_arrays.column_view`, which is the whole
    requirement."""
    if _np is not None:
        return _np.empty((n_rows, n_features), dtype=_np.float64, order="F")
    return _array.array("d", bytes(8 * n_rows * n_features))


def _store_column(out, values, n_rows, index):
    """Write one converted column into the matrix buffer."""
    if _np is not None and hasattr(out, "shape"):
        out[:, index] = values
        return
    base = index * n_rows
    for position in range(n_rows):
        out[base + position] = values[position]


def column_major_arrow(X, name="X", encoders=None):
    """`(buffer, n_rows, n_features)` for an Arrow table.

    The Arrow counterpart of `_arrays.column_major`, with the same
    contract: the buffer is validated float64 in column-major order, and
    the caller keeps it referenced while its address is in flight. Unlike
    the dense path, the source object is not part of that lifetime, because
    every value was copied out of it (see the module docstring).

    `encoders` maps a column index to a fitted category table, exactly as
    in `_arrays.column_major`.
    """
    if not is_arrow_table(X):
        raise TypeError(f"{name} is not an Arrow table or record batch")
    n_rows = _row_count(X)
    columns = _columns_of(X)
    n_features = len(columns)
    _require_unique_names([column_name for column_name, _ in columns], name)
    if n_rows == 0:
        raise ValueError(f"{name} must have at least one row")
    if n_features == 0:
        raise ValueError(f"{name} must have at least one feature")
    tables = encoders or {}
    out = _allocate_matrix(n_rows, n_features)
    for index, (column_name, column) in enumerate(columns):
        values = column_values(
            column,
            n_rows,
            name=column_name,
            index=index,
            categories=tables.get(index),
            matrix_name=name,
        )
        if index not in tables and column_kind(column.type) != "dictionary":
            _require_no_infinities(values, column_name, index, name)
        _store_column(out, values, n_rows, index)
    return out, n_rows, n_features


def check_X_arrow(X, name="X", encoders=None):
    """`(buffer, n_rows, n_features, names)` for an Arrow table.

    The Arrow counterpart of `_arrays.check_X`, returning the same 4-tuple
    in the same order, so a dispatcher in `_arrays.check_X` is one `if`
    and one call. The names come from the schema, which every Arrow table
    has, so an Arrow-fitted estimator always has `feature_names_in_`.
    """
    names = arrow_feature_names(X)
    buf, n_rows, n_features = column_major_arrow(X, name, encoders)
    return buf, n_rows, n_features, names


def arrow_f64_vector(values, n_rows, name="y"):
    """An Arrow array as a float64 vector of length `n_rows`.

    The counterpart of `_arrays.f64_vector`, for the columns a `Dataset`
    takes beside the matrix: `label`, `weight`, `init_score`, and `group`.
    Nulls become NaN here as they do everywhere, which means a null label
    or a null weight is refused a moment later by
    `_arrays.check_target` / `check_sample_weight` rather than here; that
    is the right division, because the message those raise ("must not
    contain NaN or infinite values") is about the field, and this function
    knows only about the conversion.
    """
    if not is_arrow_array(values):
        raise TypeError(f"{name} is not an Arrow array")
    kind = column_kind(values.type)
    if kind == "dictionary":
        raise ValueError(
            f"{name} is a dictionary column; pass the labels themselves "
            "(a classifier encodes them) or their codes as a numeric column"
        )
    if kind == "refused":
        raise _refusal(values.type, name, 0, name)
    length = len(values)
    if length != n_rows:
        raise ValueError(
            f"{name} must have length {n_rows}, got {length}"
        )
    return column_values(
        values, n_rows, name=name, index=0, matrix_name=name
    )


def arrow_to_pylist(values, name="y"):
    """An Arrow array as a Python list, nulls included as None.

    For `_arrays.encode_labels`, which classifies on the labels as passed
    rather than on numbers: a dictionary column of strings should train the
    same classifier a Python list of the same strings trains, and it does
    if it arrives as that list. A dictionary column is expanded to its
    labels here rather than to its codes, which is the difference between
    `classes_` holding your category names and holding 0..k-1.
    """
    if not is_arrow_array(values):
        raise TypeError(f"{name} is not an Arrow array")
    out = []
    for chunk in _chunks_of(values):
        out.extend(chunk.to_pylist())
    return out


def sparse_to_scipy(obj, name="X"):
    """An Arrow sparse tensor as the SciPy matrix `_arrays.check_X_sparse`
    takes.

    Arrow's `SparseCSRMatrix` and `SparseCSCMatrix` carry the same three
    arrays mojoboost's sparse path carries, and both expose `to_scipy()`,
    so the bridge is one call and a shape check rather than a second
    canonicalizer. COO and CSF have no direct equivalent and say so.
    """
    if not is_arrow_sparse_tensor(obj):
        raise TypeError(f"{name} is not an Arrow sparse tensor")
    kind = type(obj).__name__
    if "CSR" not in kind and "CSC" not in kind:
        raise TypeError(
            f"{name} is a {kind}, which mojoboost's sparse path does not "
            "take; convert it to a SparseCSRMatrix or SparseCSCMatrix, or "
            "to a scipy.sparse matrix, first"
        )
    shape = tuple(obj.shape)
    if len(shape) != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {shape}")
    return obj.to_scipy()


def refuse_lazy(obj, name="X"):
    """The error a `pyarrow.dataset.Dataset` or a `RecordBatchReader`
    raises, naming the two things a caller can do with it.

    Materializing it here would be the wrong answer twice: it is unbounded
    memory, and it hides the fact that mojoboost has a real streaming
    story in progress (`_sequence.py` on this side,
    `src/mojoboost/sequence.mojo` on the other).
    """
    raise TypeError(
        f"{name} is a streaming Arrow input ({type(obj).__name__}), which "
        "this adapter will not materialize behind your back; call "
        ".to_table() if the data fits in memory, or pass its batches to "
        "mojoboost's batched input path"
    )
