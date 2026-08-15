"""Batched input, and the one dispatcher over every ecosystem adapter.

Internal, on the terms of section 2 of `docs/COMPATIBILITY_POLICY.md`.
`docs/ECOSYSTEM_INPUTS.md` is the prose, and
`handoffs/remaining_10_ecosystem_inputs.md` holds the patches that wire this
into `_arrays`, `Dataset`, and the estimators.

Two jobs, and they belong together
----------------------------------

**The dispatcher.** `_arrays.check_X` knows numpy, pandas, sequences, and
SciPy sparse. `_arrow` and `_polars` each add a kind of input in exactly
the shape `_arrays` uses. Something has to decide which of them a given
object belongs to, and that decision must be made in one place or the
estimators, `Dataset`, and `cv` will each grow their own version of it.
`adapter_for`, `categories_for`, `names_for`, and `vector_for` are that
place. Each returns a callable (or None), each callable has the signature
of the `_arrays` function it stands in for, and the patch that puts them in
front of `_arrays.check_X` is four lines long. It is in the handoff.

**Batches.** A caller who already has their data in pieces (Arrow record
batches off a Parquet reader, a `lgb.Sequence`, a list of frames per file)
should not have to concatenate them in their own code just to hand
mojotrees one matrix. `Batches` and `materialize` take the pieces, check
that they describe the same columns, unify their category dictionaries, and
assemble one column-major buffer, with the label, the weight, and the query
ids pulled out of the batches if they travel inside them.

What this is not
----------------

**It is not bounded-memory training.** `materialize` holds every batch and
the assembled matrix at once, so peak memory is higher than handing over
one matrix, not lower. That is worth having anyway (the pieces are usually
already in memory, and the alternative is the caller writing this loop
themselves), and it is worth being blunt about, because the LightGBM
feature it resembles, `lgb.Dataset(lgb.Sequence(...))`, is a bounded-memory
feature.

Bounded memory needs a binner that accepts data in pieces, which is a
native question: two passes over the batches, quantiles built incrementally,
bins fixed before any row is written. That core is `src/mojotrees/
sequence.mojo` and `src/mojotrees/external_memory.mojo` (task 07), and the
Python side of it is a thin loop over `Batches` that hands one batch at a
time to a binding this package does not have yet. `Batches` is deliberately
shaped so that loop can be written against it without changing anything
here: it exposes the batches, their row counts, and their schemas
separately from the assembly step.

**It does not guess that a list is batches.** `[[1.0, 2.0], [3.0, 4.0]]` is
a matrix of two rows, and `_arrays` has always read it as one. A list of
frames is batches. Since both are lists, the difference cannot be inferred
from the object, and inferring it wrong would silently transpose a
caller's data. So batched input is opt-in: either the object implements the
`lgb.Sequence` protocol (`__len__`, `__getitem__`, and `batch_size`), or the
caller wraps it in `Batches(...)`. Nothing else is ever read as batches.
"""

import array as _array

from . import _arrays, _arrow, _polars

_np = _arrays.np


# -- dispatch -------------------------------------------------------------


def input_kind(data):
    """What kind of input `data` is, as one of the names below, or None
    when it is one `_arrays` already handles (numpy, pandas, a sequence of
    rows) or one nothing handles.

    - `"arrow_table"`: an Arrow `Table` or `RecordBatch`
    - `"arrow_array"`: an Arrow `Array` or `ChunkedArray`
    - `"arrow_dataset"`: a streaming Arrow source
    - `"arrow_sparse"`: an Arrow sparse tensor
    - `"polars_frame"`, `"polars_series"`, `"polars_lazyframe"`
    - `"batches"`: an `lgb.Sequence`-style object or a `Batches` wrapper

    Answering None for numpy and pandas is the point: this function says
    "not mine", and the existing `_arrays` path stays the default rather
    than becoming one case of a dispatcher.
    """
    if isinstance(data, Batches):
        return "batches"
    kind = _arrow.arrow_kind(data)
    if kind is not None:
        return {
            "table": "arrow_table",
            "array": "arrow_array",
            "dataset": "arrow_dataset",
            "sparse_tensor": "arrow_sparse",
        }[kind]
    kind = _polars.polars_kind(data)
    if kind is not None:
        return "polars_" + kind
    if is_sequence_protocol(data):
        return "batches"
    return None


def is_sequence_protocol(obj):
    """True for an object implementing LightGBM's `lgb.Sequence`.

    `batch_size` is what makes the recognition safe: `__len__` and
    `__getitem__` alone describe a list of rows just as well as a list of
    batches, and reading one as the other would transpose a caller's data
    without an error. LightGBM's own ABC declares all three, so requiring
    the third costs a real `Sequence` nothing.
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    if not hasattr(obj, "batch_size"):
        return False
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


def adapter_for(data):
    """The `_arrays.check_X`-shaped callable for `data`, or None.

    The returned callable takes `(X, name="X", encoders=None)` and returns
    `(buffer, n_rows, n_features, names)`, which is `check_X`'s contract
    exactly, so the dispatch in `_arrays.check_X` is an `if` and a call.
    An input this package recognizes but cannot convert (a LazyFrame, a
    streaming Arrow source) returns a callable that raises the message
    naming the fix, rather than None, so the caller does not fall through
    to "could not be converted to a float64 array".
    """
    kind = input_kind(data)
    if kind is None:
        return None
    if kind == "arrow_table":
        return _arrow.check_X_arrow
    if kind == "polars_frame":
        return _polars.check_X_polars
    if kind == "batches":
        return check_X_batches
    if kind == "arrow_dataset":
        return _refuse_streaming
    if kind == "polars_lazyframe":
        return _refuse_lazyframe
    if kind == "arrow_sparse":
        return _refuse_arrow_sparse
    if kind in ("arrow_array", "polars_series"):
        return _refuse_one_dimensional
    return None


def categories_for(data):
    """The `_arrays.frame_categories`-shaped callable for `data`, or None.

    Returns `{column_index: [label, ...]}` for the columns that declare
    themselves categorical. `categorical_feature="auto"` means exactly
    those columns, so an Arrow `dictionary` column and a polars
    `Categorical` column become categorical features on the same terms a
    pandas `category` column already does.
    """
    kind = input_kind(data)
    if kind == "arrow_table":
        return _arrow.arrow_frame_categories
    if kind == "polars_frame":
        return _polars.polars_frame_categories
    if kind == "batches":
        return batch_categories
    return None


def names_for(data):
    """The `_arrays.feature_names`-shaped callable for `data`, or None."""
    kind = input_kind(data)
    if kind == "arrow_table":
        return _arrow.arrow_feature_names
    if kind == "polars_frame":
        return _polars.polars_feature_names
    if kind == "batches":
        return batch_feature_names
    return None


def vector_for(values):
    """The `_arrays.f64_vector`-shaped callable for `values`, or None.

    For the columns a `Dataset` takes beside the matrix: `label`,
    `weight`, `init_score`, and `group`. The callable takes
    `(values, n_rows, name)` and returns a float64 buffer, which is
    `f64_vector`'s contract.
    """
    kind = input_kind(values)
    if kind == "arrow_array":
        return _arrow.arrow_f64_vector
    if kind == "polars_series":
        return _polars.polars_f64_vector
    return None


def labels_for(values):
    """The callable that turns `values` into the plain Python list
    `_arrays.encode_labels` classifies on, or None.

    A classifier's `classes_` should hold the caller's labels, so a
    dictionary or `Enum` label column arrives as its labels rather than as
    its codes. Everything else about `encode_labels` stays where it is.
    """
    kind = input_kind(values)
    if kind == "arrow_array":
        return _arrow.arrow_to_pylist
    if kind == "polars_series":
        return _polars.polars_to_pylist
    return None


def _refuse_streaming(X, name="X", encoders=None):
    """`adapter_for`'s answer for a streaming Arrow source."""
    _arrow.refuse_lazy(X, name)


def _refuse_lazyframe(X, name="X", encoders=None):
    raise TypeError(
        f"{name} is a polars LazyFrame, which has not read any data yet; "
        "call .collect() and pass the DataFrame, so that the cost of "
        "running your query plan is in your code rather than inside an "
        "argument check"
    )


def _refuse_arrow_sparse(X, name="X", encoders=None):
    raise TypeError(
        f"{name} is an Arrow sparse tensor. mojotrees's sparse path takes "
        "SciPy matrices: convert it with .to_scipy() and pass that, which "
        "keeps it sparse end to end rather than densifying it here"
    )


def _refuse_one_dimensional(X, name="X", encoders=None):
    raise ValueError(
        f"{name} is a single column, and a feature matrix is "
        "2-dimensional; wrap it in a table or a frame with one column, or "
        "pass it as the label, the weight, or the init score, which are "
        "the arguments a single column belongs to"
    )


def describe_input(data):
    """A plain dict describing what `data` is and how it would convert.

    Diagnostics: what `mojotrees.show_versions()` and a bug report want,
    and what `docs/ECOSYSTEM_INPUTS.md` is written against. Converts
    nothing and allocates nothing beyond the description.
    """
    kind = input_kind(data)
    out = {
        "kind": kind or "native",
        "type": type(data).__name__,
        "adapter": None,
        "columns": None,
        "arrow_installed": _arrow.arrow_available(),
        "polars_installed": _polars.polars_available(),
    }
    if kind == "arrow_table":
        out["adapter"] = "mojotrees._arrow"
        out["columns"] = [
            plan.as_dict() for plan in _arrow.describe_columns(data)
        ]
    elif kind == "polars_frame":
        out["adapter"] = "mojotrees._polars"
        out["columns"] = [
            {
                "index": index,
                "name": column_name,
                "dtype": dtype,
                "kind": column_kind,
            }
            for index, column_name, dtype, column_kind in (
                _polars.describe_schema(data)
            )
        ]
    elif kind == "batches":
        out["adapter"] = "mojotrees._sequence"
        out["columns"] = _wrap(data).describe()
    elif kind is not None:
        out["adapter"] = (
            "mojotrees._arrow" if kind.startswith("arrow") else
            "mojotrees._polars"
        )
    return out


# -- one batch ------------------------------------------------------------


def _batch_kind(batch):
    """The converter key for one batch: `"arrow"`, `"polars"`, or
    `"dense"`, the last covering everything `_arrays` already reads.

    A batch is anything a feature matrix can be, which is why this reads
    almost like `input_kind` and does not share its body: `input_kind`
    answers None for numpy and pandas because `_arrays` owns them, and here
    they are ordinary batches with a converter like any other.
    """
    if _arrow.is_arrow_table(batch):
        return "arrow"
    if _polars.is_polars_frame(batch):
        return "polars"
    if _polars.is_polars_lazyframe(batch):
        raise TypeError(
            "a batch is a polars LazyFrame; call .collect() on it first"
        )
    if _arrow.is_arrow_dataset(batch):
        raise TypeError(
            "a batch is a streaming Arrow source; pass its batches rather "
            "than the source, or call .to_table() on it"
        )
    if _arrays.is_sparse(batch):
        raise TypeError(
            "a batch is a SciPy sparse matrix. Batched input assembles one "
            "dense matrix, so a sparse batch would be densified, which "
            "this will not do behind your back; use the sparse fit path "
            "with one matrix instead"
        )
    return "dense"


def _batch_names(batch):
    """Feature names for one batch, or None."""
    kind = _batch_kind(batch)
    if kind == "arrow":
        return _arrow.arrow_feature_names(batch)
    if kind == "polars":
        return _polars.polars_feature_names(batch)
    return _arrays.feature_names(batch)


def _batch_categories(batch):
    """`{column_index: [label, ...]}` for one batch."""
    kind = _batch_kind(batch)
    if kind == "arrow":
        return _arrow.arrow_frame_categories(batch)
    if kind == "polars":
        return _polars.polars_frame_categories(batch)
    return _arrays.frame_categories(batch)


def _batch_convert(batch, name, encoders):
    """`(buffer, n_rows, n_features, names)` for one batch, through
    whichever adapter owns its kind.

    This is the whole reason the three adapters return the same 4-tuple:
    a batch of any supported kind converts through one call, and batched
    input did not need a fourth converter written for it.
    """
    kind = _batch_kind(batch)
    if kind == "arrow":
        return _arrow.check_X_arrow(batch, name, encoders)
    if kind == "polars":
        return _polars.check_X_polars(batch, name, encoders)
    return _arrays.check_X(batch, name, encoders)


def _batch_rows(batch):
    """Rows in one batch, without converting it."""
    shape = getattr(batch, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0])
    rows = getattr(batch, "num_rows", None)
    if rows is not None:
        return int(rows)
    return len(batch)


def _batch_column(batch, column, name):
    """One named column out of a batch, as the object the column's own
    library represents it with.

    Used for `label_column`, `weight_column`, and `query_column`, which are
    the three columns that usually travel inside the data file rather than
    beside it.
    """
    kind = _batch_kind(batch)
    if kind == "arrow":
        names = list(batch.column_names)
        if column not in names:
            raise ValueError(
                f"{name} {column!r} is not a column of this batch; it has "
                f"{names}"
            )
        return _arrow.arrow_to_pylist(batch.column(names.index(column)))
    if kind == "polars":
        if column not in list(batch.columns):
            raise ValueError(
                f"{name} {column!r} is not a column of this batch; it has "
                f"{list(batch.columns)}"
            )
        return _polars.polars_to_pylist(batch.get_column(column))
    names = _arrays.feature_names(batch)
    if names is None or column not in names:
        raise ValueError(
            f"{name} {column!r} cannot be read from a batch that carries "
            "no column names; pass the column separately"
        )
    return list(batch[column])


def _batch_drop(batch, columns, name):
    """A batch without the named columns, using the batch's own dropper.

    Every kind that can carry a label column inline can also drop one:
    Arrow has `drop_columns` (and `drop` before it), polars has `drop`, and
    pandas has `drop(columns=...)`. A batch kind that cannot is refused
    rather than silently trained on its own label.
    """
    if not columns:
        return batch
    kind = _batch_kind(batch)
    if kind == "arrow":
        dropper = getattr(batch, "drop_columns", None) or getattr(
            batch, "drop", None
        )
        if dropper is None:
            raise TypeError(
                f"{name} names columns to take out of each batch, but this "
                f"{type(batch).__name__} cannot drop columns; project them "
                "out before passing the batches"
            )
        return dropper(list(columns))
    if kind == "polars":
        return batch.drop(list(columns))
    dropper = getattr(batch, "drop", None)
    if dropper is None:
        raise TypeError(
            f"{name} names columns to take out of each batch, but a batch "
            f"of type {type(batch).__name__} cannot drop columns; pass "
            "frames, or project the columns out yourself"
        )
    return dropper(columns=list(columns))


# -- the batch source -----------------------------------------------------


class Batches:
    """A caller's batches, in the order they will be concatenated.

    Wrap anything that yields feature matrices: a list of Arrow record
    batches, a list of polars frames, an `lgb.Sequence`, a
    `pyarrow.dataset.Dataset`, a `RecordBatchReader`, or a generator of any
    of those. Wrapping is required rather than inferred, for the reason in
    the module docstring: a list of rows and a list of batches are both
    lists.

    A source that can be indexed is held as it is and read twice, which is
    what a two-pass binner would need. A source that can only be iterated
    (a generator, a reader) is drained into a list on construction, because
    the alternative is reading a stream twice and getting nothing the
    second time. `drained` says which happened.
    """

    __slots__ = (
        "_batches",
        "drained",
        "source_type",
        "_schema",
        "_categories",
    )

    def __init__(self, source):
        self.source_type = type(source).__name__
        self._batches, self.drained = _collect_batches(source)
        if not self._batches:
            raise ValueError("a batched input must have at least one batch")
        # Both are derived from the batches and both are asked for more
        # than once on a fit (`feature_names`, `frame_categories`, and
        # `check_X` each want one of them), so they are computed once. The
        # batches are held, not consumed, and an Arrow table or a polars
        # frame does not change under its holder, so the answers do not go
        # stale.
        self._schema = None
        self._categories = None

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)

    def __getitem__(self, index):
        return self._batches[index]

    @property
    def batches(self):
        """The batches themselves, in order."""
        return list(self._batches)

    def schema(self, name="X"):
        """`(names, n_features)` for the whole input, checked across every
        batch and computed once."""
        if self._schema is None:
            self._schema = _check_consistency(self._batches, name)
        return self._schema

    def categories(self, name="X"):
        """The unified category table per column, computed once."""
        if self._categories is None:
            self._categories = unify_categories(self._batches, name)
        return self._categories

    @property
    def shape(self):
        """`(n_rows, n_features)`, so a batched input answers the shape
        question everything else answers.

        `mojotrees.device_selection.Workload.from_data` reads `shape` off
        anything two-dimensional, which is how an Arrow table and a polars
        frame already reach it; this is what puts `Batches` in the same
        position without a change in that module. It costs one cheap pass
        over the batches (each reports its own row count) and no
        conversion.
        """
        _, n_features = self.schema()
        return self.num_data(), n_features

    def row_counts(self):
        """Rows per batch, without converting anything."""
        return [_batch_rows(batch) for batch in self._batches]

    def num_data(self):
        """Rows across every batch."""
        return sum(self.row_counts())

    def offsets(self):
        """The first row index of each batch in the assembled matrix, plus
        the total, so `offsets()[i]:offsets()[i + 1]` is batch `i`.

        This is the mapping a later streaming binner needs in order to
        report which batch a row came from, and the mapping `materialize`
        writes each batch through.
        """
        out = [0]
        for count in self.row_counts():
            out.append(out[-1] + count)
        return out

    def describe(self):
        """One dict per batch: its type, its rows, its names, and its
        categorical columns. Diagnostics, and the consistency check's
        input."""
        out = []
        for position, batch in enumerate(self._batches):
            out.append(
                {
                    "index": position,
                    "type": type(batch).__name__,
                    "kind": _batch_kind(batch),
                    "n_rows": _batch_rows(batch),
                    "names": _batch_names(batch),
                    "categorical": sorted(_batch_categories(batch)),
                }
            )
        return out

    def __repr__(self):
        return (
            f"Batches({len(self._batches)} batches, {self.num_data()} rows,"
            f" from {self.source_type})"
        )


def _collect_batches(source):
    """`(batches, drained)` for whatever a caller wrapped.

    The four shapes a batch source arrives in, in the order they are
    tried: an already-wrapped `Batches`, a streaming Arrow source, an
    indexable sequence (which includes `lgb.Sequence`), and a plain
    iterable.
    """
    if isinstance(source, Batches):
        return source.batches, source.drained
    to_batches = getattr(source, "to_batches", None)
    if to_batches is not None and hasattr(source, "schema"):
        return list(to_batches()), True
    if hasattr(source, "read_next_batch") and hasattr(source, "schema"):
        return list(_drain_reader(source)), True
    if hasattr(source, "__len__") and hasattr(source, "__getitem__"):
        return [source[i] for i in range(len(source))], False
    try:
        return list(source), True
    except TypeError:
        raise TypeError(
            f"a batched input must be iterable or indexable; a "
            f"{type(source).__name__} is neither"
        ) from None


def _drain_reader(reader):
    """Every batch of an Arrow `RecordBatchReader`, in order."""
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


# -- assembly -------------------------------------------------------------


def _check_consistency(source, name):
    """Refuse batches that do not describe the same columns.

    Three ways they can disagree, and all three are silent corruption if
    they are let through: different widths (the matrix would be ragged),
    different names (the model would record one batch's names for another
    batch's columns), and names in one batch and not another (half the
    matrix would be identified and half positional). The width is the only
    one LightGBM checks.
    """
    names = None
    width = None
    for position, batch in enumerate(source):
        batch_names = _batch_names(batch)
        batch_width = _batch_width(batch, batch_names)
        if position == 0:
            names = batch_names
            width = batch_width
            continue
        if batch_width != width:
            raise ValueError(
                f"{name} batch {position} has {batch_width} features but "
                f"batch 0 has {width}"
            )
        if (names is None) != (batch_names is None):
            raise ValueError(
                f"{name} batch {position} "
                + (
                    "carries no column names"
                    if batch_names is None
                    else "carries column names"
                )
                + " but batch 0 "
                + ("does" if batch_names is None else "does not")
                + "; either every batch names its columns or none does, "
                "because half a named matrix cannot be identified"
            )
        if names is not None and list(batch_names) != list(names):
            raise ValueError(
                f"{name} batch {position} has columns {list(batch_names)} "
                f"but batch 0 has {list(names)}; batches are concatenated "
                "by position, so their columns must match in order as well "
                "as in name"
            )
    return names, width


def _batch_width(batch, names):
    """Features in one batch, without converting it.

    A batch that cannot say how wide it is fails here rather than at the
    concatenation, because a 1-dimensional batch is the likely cause and
    "rows must have equal length" would be the wrong thing to be told.
    """
    if names is not None:
        return len(names)
    shape = getattr(batch, "shape", None)
    if shape is not None:
        if len(shape) == 2:
            return int(shape[1])
        raise ValueError(
            f"a batch must be 2-dimensional, got shape {tuple(shape)}; a "
            "batch is a block of rows of the feature matrix, so one row is "
            "shape (1, n_features) and one feature is (n_rows, 1)"
        )
    columns = getattr(batch, "num_columns", None)
    if columns is not None:
        return int(columns)
    try:
        return len(batch[0])
    except TypeError:
        raise ValueError(
            f"a batch of type {type(batch).__name__} does not describe its "
            "width; pass batches as arrays, frames, tables, or lists of "
            "rows"
        ) from None


def unify_categories(source, name="X"):
    """`{column_index: [label, ...]}` across every batch.

    A category table has to be one table for the whole matrix: batch 2's
    third category and batch 5's third category must be the same category,
    or the binner splits on the union of two unrelated things. Arrow makes
    this easy to get wrong, because every record batch carries its own
    dictionary and two batches read from two files routinely disagree.

    The union is in first-appearance order, batch by batch and within a
    batch in the column's own order, which is the same rule
    `_arrow._unify_dictionaries` uses across the chunks of one column. A
    column that is categorical in one batch and not in another is refused:
    that is a schema difference, not an ordering one.
    """
    tables = {}
    seen = {}
    for position, batch in enumerate(source):
        found = _batch_categories(batch)
        for index in sorted(found):
            if index not in tables:
                if position:
                    raise ValueError(
                        f"{name} batch {position} declares column {index} "
                        "categorical but batch 0 does not; a column is "
                        "categorical for the whole matrix or not at all"
                    )
                tables[index] = []
                seen[index] = {}
            table = tables[index]
            known = seen[index]
            for label in found[index]:
                key = _key(label)
                if key in known:
                    continue
                known[key] = None
                table.append(label)
        missing = sorted(set(tables) - set(found))
        if missing:
            raise ValueError(
                f"{name} batch {position} does not declare column "
                f"{missing[0]} categorical but batch 0 does; a column is "
                "categorical for the whole matrix or not at all"
            )
    for index, table in tables.items():
        if len(table) >= _arrow.CATEGORY_LIMIT:
            raise ValueError(
                f"{name} column {index} has {len(table)} categories across "
                f"its batches, and a category code must be below "
                f"{_arrow.CATEGORY_LIMIT}"
            )
    return tables


def _key(label):
    """A dict key for a category label, unhashable labels included.

    The same function the Arrow adapter unifies chunk dictionaries with, so
    "the same label" means the same thing across chunks and across batches.
    """
    return _arrow.hashable_label(label)


def batch_feature_names(source):
    """Feature names for a batched input, or None. The
    `_arrays.feature_names` counterpart, for `Batches`."""
    names, _ = _wrap(source).schema()
    return names


def batch_categories(source):
    """`{column_index: [label, ...]}` for a batched input. The
    `_arrays.frame_categories` counterpart, for `Batches`."""
    return _wrap(source).categories()


def _wrap(source):
    """`source` as a `Batches`, wrapping it only if it is not one already.

    Reusing the wrapper is what makes the caches on it worth having: a fit
    asks for the names, then the categories, then the matrix, and an
    already-wrapped input answers the first two from what it worked out the
    first time.
    """
    return source if isinstance(source, Batches) else Batches(source)


def _allocate_matrix(n_rows, n_features):
    """The column-major float64 buffer, in whichever of the two shapes
    `_arrays` uses. Same allocation as `_arrow._allocate_matrix`, kept
    here rather than imported so the batched path does not depend on the
    Arrow adapter for a matrix that may hold no Arrow at all."""
    if _np is not None:
        return _np.empty((n_rows, n_features), dtype=_np.float64, order="F")
    return _array.array("d", bytes(8 * n_rows * n_features))


def _write_batch(out, buffer, start, rows, n_rows, n_features):
    """Copy one converted batch into the assembled matrix at row `start`."""
    for index in range(n_features):
        column = _arrays.column_view(buffer, rows, index)
        if _np is not None and hasattr(out, "shape"):
            out[start : start + rows, index] = column
            continue
        base = index * n_rows + start
        out[base : base + rows] = _array.array("d", column)


class BatchedInput:
    """One assembled matrix, plus whatever travelled with it.

    `matrix` is the column-major float64 buffer, in the shape `_arrays`
    uses, so `_arrays.addr` and `_arrays.column_view` read it and the
    binding could take its address directly. `as_rows()` and
    `dataset_kwargs()` are the two ways to spend it today; the handoff
    carries the `Dataset` patch that would take `matrix` as it is.
    """

    __slots__ = (
        "matrix",
        "n_rows",
        "n_features",
        "names",
        "categories",
        "offsets",
        "label",
        "weight",
        "query_ids",
    )

    def __init__(self, **fields):
        for slot in self.__slots__:
            setattr(self, slot, fields.get(slot))

    def group(self):
        """Per-query row counts from the query id column, or None.

        A query split across two batches is fine: the batches are
        concatenated in order, so its rows are still one unbroken run, and
        that is the only thing `group_from_query_ids` requires. A query
        whose rows are split by a *different* query's rows raises there, as
        it should.
        """
        if self.query_ids is None:
            return None
        from . import group_from_query_ids

        return group_from_query_ids(self.query_ids)

    def as_rows(self):
        """The matrix as a list of rows.

        The shape `Dataset` and the estimators take today on an install
        without numpy, where a flat column-major `array.array` would be
        read as a 1-dimensional sequence and refused. It costs a full copy,
        which is why `dataset_kwargs` only reaches for it when there is no
        numpy.
        """
        rows = []
        for r in range(self.n_rows):
            rows.append(
                [
                    float(_row_value(self.matrix, self.n_rows, r, f))
                    for f in range(self.n_features)
                ]
            )
        return rows

    def dataset_kwargs(self, **extra):
        """Keyword arguments for `mojotrees.Dataset`.

            data = mojotrees._sequence.materialize(batches, label_column="y")
            train_set = mojotrees.Dataset(**data.dataset_kwargs())

        Works today, with no change anywhere else in the package. With
        numpy the matrix is passed as the 2-D array it already is, which
        `_arrays.check_X` revalidates and does not copy; without numpy it
        goes through `as_rows()`, which does copy. `extra` is merged last,
        so a caller can add `params=` or override a column.
        """
        out = {
            "data": self.matrix if _np is not None else self.as_rows(),
            "feature_name": self.names,
            "categorical_feature": sorted(self.categories) or None,
        }
        if self.label is not None:
            out["label"] = self.label
        if self.weight is not None:
            out["weight"] = self.weight
        group = self.group()
        if group is not None:
            out["group"] = group
        out.update(extra)
        return out

    def __repr__(self):
        return (
            f"BatchedInput({self.n_rows} rows, {self.n_features} features, "
            f"{len(self.offsets) - 1} batches)"
        )


def _row_value(matrix, n_rows, row, feature):
    """One element of a column-major buffer, whichever shape it is in."""
    if _np is not None and hasattr(matrix, "shape"):
        return matrix[row, feature]
    return matrix[feature * n_rows + row]


def materialize(
    source,
    name="X",
    encoders=None,
    label_column=None,
    weight_column=None,
    query_column=None,
):
    """Assemble batches into one `BatchedInput`.

    `source` is a `Batches`, an `lgb.Sequence`, or anything `Batches`
    wraps. `encoders` maps a column index to a fitted category table and
    turns this into the prediction path: with it, every batch is encoded
    through the fitted categories, and the batches' own dictionaries are
    used only to look labels up. Without it the batches' dictionaries are
    unified into one table per column, which becomes the fitted table.

    `label_column`, `weight_column`, and `query_column` name columns that
    travel inside the batches. Each is read out of every batch, dropped
    from the features, and returned on the result, so a Parquet dataset
    with the label in it needs no separate array.

    Two passes over the batches: one to check the schemas and unify the
    categories, one to convert. That is the same two-pass shape a
    bounded-memory binner needs, which is why `Batches` keeps the batches
    rather than consuming them.
    """
    batches = _wrap(source)
    taken = [
        c
        for c in (label_column, weight_column, query_column)
        if c is not None
    ]
    if len(set(taken)) != len(taken):
        raise ValueError(
            "label_column, weight_column, and query_column must name "
            f"different columns; got {taken}"
        )

    label = [] if label_column is not None else None
    weight = [] if weight_column is not None else None
    query_ids = [] if query_column is not None else None
    feature_batches = []
    for batch in batches:
        if label is not None:
            label.extend(_batch_column(batch, label_column, "label_column"))
        if weight is not None:
            weight.extend(
                _batch_column(batch, weight_column, "weight_column")
            )
        if query_ids is not None:
            query_ids.extend(
                _batch_column(batch, query_column, "query_column")
            )
        feature_batches.append(_batch_drop(batch, taken, "materialize"))

    # With nothing taken out, the feature batches are the batches, so the
    # wrapper's cached schema and category tables are the answer. With a
    # label or a weight column removed they are not, and both are worked
    # out again on what is left.
    if taken:
        names, n_features = _check_consistency(feature_batches, name)
        found = unify_categories(feature_batches, name)
    else:
        names, n_features = batches.schema(name)
        found = batches.categories(name)
    if n_features == 0:
        raise ValueError(f"{name} must have at least one feature")
    tables = dict(encoders) if encoders else found
    counts = [_batch_rows(batch) for batch in feature_batches]
    n_rows = sum(counts)
    if n_rows == 0:
        raise ValueError(f"{name} must have at least one row")

    out = _allocate_matrix(n_rows, n_features)
    offsets = [0]
    for position, batch in enumerate(feature_batches):
        buffer, rows, width, _ = _batch_convert(
            batch, f"{name} batch {position}", tables
        )
        if width != n_features:
            raise ValueError(
                f"{name} batch {position} converted to {width} features "
                f"but was described as {n_features}"
            )
        if rows != counts[position]:
            raise ValueError(
                f"{name} batch {position} converted to {rows} rows but "
                f"reported {counts[position]}"
            )
        _write_batch(out, buffer, offsets[-1], rows, n_rows, n_features)
        offsets.append(offsets[-1] + rows)

    return BatchedInput(
        matrix=out,
        n_rows=n_rows,
        n_features=n_features,
        names=names,
        categories=tables,
        offsets=offsets,
        label=label,
        weight=weight,
        query_ids=query_ids,
    )


def check_X_batches(X, name="X", encoders=None):
    """`(buffer, n_rows, n_features, names)` for a batched input.

    The `_arrays.check_X` counterpart, so `Batches` is a feature matrix
    everywhere a feature matrix is taken. The label, weight, and query
    columns are not reachable through this signature, which is
    `check_X`'s, so a caller who needs them calls `materialize` directly
    and gets them on the result.
    """
    result = materialize(X, name=name, encoders=encoders)
    return result.matrix, result.n_rows, result.n_features, result.names
