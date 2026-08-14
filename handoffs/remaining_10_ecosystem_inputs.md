# Remaining task 10 handoff: ecosystem input adapters

Lane files (the only ones this lane wrote):

- `python/mojoboost/_arrow.py` (new)
- `python/mojoboost/_polars.py` (new)
- `python/mojoboost/_sequence.py` (new)
- `docs/ECOSYSTEM_INPUTS.md` (new)
- `handoffs/remaining_10_ecosystem_inputs.md` (this file)

Read and not edited: `python/mojoboost/_arrays.py`,
`python/mojoboost/basic.py`, `python/mojoboost/__init__.py`,
`python/mojoboost/cv.py`, `python/mojoboost/dask.py`,
`python/mojoboost/_compat.py`, `python/mojoboost/device_selection.py`,
`bindings/_mojoboost.mojo`, `docs/LIGHTGBM_PARITY.md`,
`python/pyproject.toml`, `pixi.toml`.

This lane committed nothing and staged nothing. Nothing was executed: no
test was written or run, no build, no Python, no benchmark, and pyarrow and
polars were never imported. Every claim below is a reading of the code as
written.

> **Read this before integrating: a concurrent session committed this
> lane's files mid-task.** Four commits landed while this lane ran
> (`dc21f03`, `860b1cf`, `e6f3959`, `5085097`), and one of them swept up
> `_arrow.py`, `_polars.py`, `_sequence.py`, and `docs/ECOSYSTEM_INPUTS.md`
> as they stood at that moment, together with many other lanes' work.
> `_arrow.py` and `_polars.py` were finished by then and the committed
> versions are correct. **`_sequence.py` and `docs/ECOSYSTEM_INPUTS.md`
> were not**: the committed snapshot predates `Batches.shape`, the cached
> `schema()` / `categories()`, the `_wrap` helper, the `_batch_width`
> dimension check, and the `describe_input` availability fields. The
> working tree holds the correct version as an uncommitted delta
> (`git diff -- python/mojoboost/_sequence.py docs/ECOSYSTEM_INPUTS.md`
> is exactly those changes and nothing else). **Commit the working tree,
> not the committed snapshot.** No other lane's file was touched by this
> one.

Other lanes also changed `python/mojoboost/_arrays.py` (a new
`i64_vector`), `__init__.py`, `basic.py`, `device_selection.py`,
`_eval.py`, and many `src/mojoboost/*.mojo` files while this lane ran. None
were touched here, and none of them move the anchors the patches below
quote. Anchor on the quoted text, not on a line number.

---

## 1. What landed

Three internal modules, on the terms of section 2 of
`docs/COMPATIBILITY_POLICY.md` (every underscore-prefixed module in
`python/mojoboost/` is outside the public surface).

### `_arrow.py`

Arrow tables, record batches, arrays, chunked arrays, sparse tensors, and
streaming sources. Converts a table to the same validated column-major
float64 buffer `_arrays.column_major` produces, with the same NaN-as-missing
rule, the same infinity refusal, and the same category-code semantics as
the pandas `category` path.

| Symbol | Signature | Mirrors |
|---|---|---|
| `arrow_available()` | `() -> bool` | - |
| `is_arrow_table` / `is_arrow_array` / `is_arrow_chunked` / `is_arrow_dataset` / `is_arrow_sparse_tensor` | `(obj) -> bool` | - |
| `arrow_kind` | `(obj) -> str \| None` | - |
| `type_name` | `(arrow_type) -> str` | - |
| `is_dictionary_type` | `(arrow_type) -> bool` | - |
| `column_kind` | `(arrow_type) -> str` | - |
| `arrow_feature_names` | `(X) -> list[str] \| None` | `_arrays.feature_names` |
| `arrow_frame_categories` | `(X) -> dict[int, list]` | `_arrays.frame_categories` |
| `column_values` | `(column, n_rows, name, index, categories, matrix_name) -> buffer` | - |
| `column_major_arrow` | `(X, name="X", encoders=None) -> (buf, n_rows, n_features)` | `_arrays.column_major` |
| `check_X_arrow` | `(X, name="X", encoders=None) -> (buf, n_rows, n_features, names)` | `_arrays.check_X` |
| `arrow_f64_vector` | `(values, n_rows, name="y") -> buffer` | `_arrays.f64_vector` |
| `arrow_to_pylist` | `(values, name="y") -> list` | input to `_arrays.encode_labels` |
| `sparse_to_scipy` | `(obj, name="X") -> scipy matrix` | input to `_arrays.check_X_sparse` |
| `describe_columns` | `(X) -> list[BufferPlan]` | - |
| `refuse_lazy` | `(obj, name="X") -> NoReturn` | - |
| `hashable_label` | `(value) -> key` | `_arrays._codes_from_labels`'s `TypeError` guard |
| `BufferPlan` | 14 slots, `as_dict()` | - |
| `INT_EXACT_LIMIT`, `CATEGORY_LIMIT` | `1 << 53`, `1 << 31` | `_Base._CATEGORY_LIMIT` |

### `_polars.py`

polars frames, series, and lazyframes. Thin by design: refuses in polars'
vocabulary, then hands `to_arrow()` to `_arrow`, which owns every
conversion rule.

| Symbol | Signature | Mirrors |
|---|---|---|
| `polars_available()` | `() -> bool` | - |
| `is_polars_frame` / `is_polars_series` / `is_polars_lazyframe` | `(obj) -> bool` | - |
| `polars_kind` | `(obj) -> str \| None` | - |
| `dtype_name` / `dtype_kind` | `(dtype) -> str` | - |
| `preflight` | `(frame, name="X") -> None` | - |
| `describe_schema` | `(frame) -> list[tuple]` | - |
| `to_arrow_table` | `(X, name="X") -> arrow table` | - |
| `polars_feature_names` | `(X) -> list[str] \| None` | `_arrays.feature_names` |
| `polars_frame_categories` | `(X) -> dict[int, list]` | `_arrays.frame_categories` |
| `column_major_polars` | `(X, name="X", encoders=None) -> (buf, n_rows, n_features)` | `_arrays.column_major` |
| `check_X_polars` | `(X, name="X", encoders=None) -> (buf, n_rows, n_features, names)` | `_arrays.check_X` |
| `polars_f64_vector` | `(values, n_rows, name="y") -> buffer` | `_arrays.f64_vector` |
| `polars_to_pylist` | `(values, name="y") -> list` | input to `_arrays.encode_labels` |

### `_sequence.py`

Batched input, and the single dispatcher over all three adapters.

| Symbol | Signature | Purpose |
|---|---|---|
| `input_kind` | `(data) -> str \| None` | None for numpy/pandas/sequences, which `_arrays` owns |
| `adapter_for` | `(data) -> callable \| None` | a `check_X`-shaped callable |
| `categories_for` | `(data) -> callable \| None` | a `frame_categories`-shaped callable |
| `names_for` | `(data) -> callable \| None` | a `feature_names`-shaped callable |
| `vector_for` | `(values) -> callable \| None` | an `f64_vector`-shaped callable |
| `labels_for` | `(values) -> callable \| None` | the list `encode_labels` classifies on |
| `describe_input` | `(data) -> dict` | diagnostics |
| `is_sequence_protocol` | `(obj) -> bool` | LightGBM `lgb.Sequence` |
| `Batches` | `(source)` | the batch source; `shape`, `num_data()`, `row_counts()`, `offsets()`, `schema()`, `categories()`, `describe()`, `drained` |
| `unify_categories` | `(source, name="X") -> dict[int, list]` | one category table per column across batches |
| `materialize` | `(source, name, encoders, label_column, weight_column, query_column) -> BatchedInput` | assembly |
| `check_X_batches` | `(X, name="X", encoders=None) -> (buf, n_rows, n_features, names)` | `_arrays.check_X` |
| `batch_feature_names` / `batch_categories` | `(source) -> ...` | the other two seams |
| `BatchedInput` | 9 slots, `group()`, `as_rows()`, `dataset_kwargs(**extra)` | the result |

## 2. Integration done inside the lane

The three modules are fused, not three isolated files:

- `_polars` holds no conversion of its own. `to_arrow_table` is its single
  door to `_arrow`, so the null rule, the dictionary unification, the
  2**53 check, the infinity refusal, and the column-major layout have one
  implementation.
- `_sequence._batch_convert` dispatches one batch to
  `_arrow.check_X_arrow`, `_polars.check_X_polars`, or
  `_arrays.check_X`. Batched input needed no fourth converter, which is
  the payoff of all three returning `check_X`'s 4-tuple.
- `_sequence.unify_categories` unifies category labels across batches with
  `_arrow.hashable_label`, the same key function
  `_arrow._unify_dictionaries` uses across the chunks of one column, so
  "the same label" means one thing in both.
- `_sequence.adapter_for` routes recognized-but-unconvertible inputs to
  refusal callables (`_refuse_streaming`, `_refuse_lazyframe`,
  `_refuse_arrow_sparse`, `_refuse_one_dimensional`) rather than to None,
  so a caller gets the message naming the fix instead of falling through to
  `_arrays`'s "could not be converted to a float64 array".
- `_arrow.sparse_to_scipy` bridges Arrow's CSR/CSC tensors to the existing
  `_arrays.check_X_sparse` path rather than densifying them.
- `BatchedInput.dataset_kwargs()` works today against the unmodified
  `mojoboost.Dataset`, and `BatchedInput.group()` calls the existing public
  `group_from_query_ids` through a function-level import (the `dask.py`
  pattern), so nothing here imports the package at module scope.

What could not be done inside the lane: nothing calls these modules. The
five dispatch points are in `_arrays.py`, `basic.py`, `__init__.py`,
`cv.py`, and `device_selection.py`, none of which this lane owns.

## 3. READY-TO-APPLY INTEGRATION PATCHES

Each patch is independent unless a dependency is named. All validation is
marked UNRUN because this lane ran nothing.

---

### P1. `_arrays.check_X`: dispatch to the adapters

- **Target file/symbol**: `python/mojoboost/_arrays.py`, `check_X`.
- **Ownership**: whoever owns `_arrays.py`. Not this lane.
- **Signature**: unchanged, `check_X(X, name="X", encoders=None)`.
- **Call site**: replace the body.

```python
def check_X(X, name="X", encoders=None):
    """(buffer, n_rows, n_features, names) for a feature matrix, with the
    column names when it carries them. `encoders` is as in `column_major`;
    the names come from the original `X`, before any encoding."""
    # Arrow tables, polars frames, and batched input convert through their
    # own adapters, which return this same 4-tuple. The import is inside
    # the function because `_sequence` imports this module.
    from . import _sequence

    adapter = _sequence.adapter_for(X)
    if adapter is not None:
        return adapter(X, name, encoders)
    names = feature_names(X)
    buf, n_rows, n_features = column_major(X, name, encoders)
    return buf, n_rows, n_features, names
```

- **State flow**: the returned buffer is owned by the caller and must stay
  referenced while its address is in flight, exactly as today. The source
  object is *not* part of that lifetime on the adapter paths, because
  every value is copied out of it.
- **Errors**: `ValueError` for a refused dtype, a duplicate column name, an
  int64 past 2**53, an infinity, a ragged batch set, or an inconsistent
  categorical declaration. `TypeError` for a LazyFrame, a streaming Arrow
  source, or an Arrow sparse tensor. Every message names the fix.
- **Fallback**: `adapter_for` returns None for numpy, pandas, sequences,
  and anything unrecognized, so the existing path is unchanged for them.
- **Dependency**: none.
- **Import-cycle note**: `_sequence` imports `_arrays` at module scope, so
  this import must stay inside the function. `basic.py` and `cv.py` already
  use function-level imports for the same reason.
- **Serialization effect**: none. The model format does not record how a
  matrix arrived.
- **Public API effect**: `Dataset(arrow_table)`, `Dataset(polars_frame)`,
  and every estimator `fit`/`predict` accept the new inputs, because all of
  them reach `check_X`. No new public name.
- **Minimal later validation (UNRUN)**:
  - `mb.Dataset(pa.table({"a": [1.0, 2.0]}), label=[0.0, 1.0]).num_data() == 2`
  - a numpy fit and an Arrow fit of the same values produce equal
    predictions
  - `mb.Dataset([[1.0, 2.0], [3.0, 4.0]])` still reads as two rows

---

### P2. `_arrays.feature_names` and `_arrays.frame_categories`: dispatch

- **Target file/symbol**: `python/mojoboost/_arrays.py`, `feature_names`
  and `frame_categories`.
- **Ownership**: whoever owns `_arrays.py`.
- **Signatures**: unchanged.
- **Call site**: prepend to each body.

```python
def feature_names(X):
    from . import _sequence

    adapter = _sequence.names_for(X)
    if adapter is not None:
        return adapter(X)
    columns = getattr(X, "columns", None)
    ...  # unchanged
```

```python
def frame_categories(X):
    from . import _sequence

    adapter = _sequence.categories_for(X)
    if adapter is not None:
        return adapter(X)
    iloc = getattr(X, "iloc", None)
    ...  # unchanged
```

- **State flow**: `_Base._fit_X` calls `feature_names` and
  `frame_categories` before `check_X` and records the result on
  `_cat_encoders` and `feature_names_in_`. With this patch an Arrow
  `dictionary` column and a polars `Categorical` / `Enum` column become
  categorical features under `categorical_feature="auto"` on exactly the
  terms a pandas `category` column already does, including the rule in
  `_resolve_categorical` that a declared-categorical column left out of an
  explicit list raises.
- **Errors**: `_matrix_encoders` raises today when a prediction frame has a
  category column the model holds no mapping for, and when the model was
  fitted on category columns and `X` has no `iloc`. **That second check is
  pandas-specific and will misfire on Arrow and polars**; see P3.
- **Fallback**: None for every input `_arrays` already handles.
- **Dependency**: P1 should land with these; `check_X` and the two
  describers must agree about what a matrix is.
- **Public API effect**: `feature_names_in_` is populated from an Arrow
  schema and a polars frame. `categorical_feature_` covers their category
  columns.
- **Minimal later validation (UNRUN)**:
  - fit on a table with a `dictionary` column, then predict on a table
    whose dictionary lists the same labels in a different order, and check
    the predictions match a fit/predict on the pandas equivalent
  - `est.feature_names_in_` equals the Arrow schema names

---

### P3. `_Base._matrix_encoders`: the pandas-only guard

- **Target file/symbol**: `python/mojoboost/__init__.py`,
  `_Base._matrix_encoders`, the `if encoders and not hasattr(X, "iloc")`
  branch.
- **Ownership**: whoever owns `__init__.py`.
- **Problem**: the guard says "a DataFrame is the only thing that carries
  category labels". After P2 that is false: an Arrow table and a polars
  frame carry them too, and an estimator fitted on an Arrow dictionary
  column would be told to "pass X as a DataFrame".
- **Call site**: replace the condition.

```python
        if encoders and not _carries_labels(X):
            raise ValueError(
                f"{type(self).__name__} was fitted on categorical columns "
                f"{sorted(encoders)}, whose labels only a labelled frame "
                f"carries; pass {name} as a pandas DataFrame, an Arrow "
                "table, or a polars frame, or fit on integer codes instead"
            )
```

with, next to it:

```python
def _carries_labels(X):
    """True when `X` can carry category labels rather than codes: a pandas
    frame, an Arrow table, or a polars frame."""
    from . import _sequence

    return hasattr(X, "iloc") or _sequence.categories_for(X) is not None
```

- **State flow**: `_matrix_encoders` feeds `encoders` into
  `check_X(..., encoders=...)`, which the adapters honor through
  `_arrow._fitted_remaps`.
- **Errors**: unchanged in kind; the message widens.
- **Fallback**: identical behavior for pandas and for numpy.
- **Dependency**: P2.
- **Public API effect**: predicting on an Arrow table with a fitted
  dictionary column works instead of raising.
- **Minimal later validation (UNRUN)**: fit on a polars frame with an
  `Enum` column, predict on another polars frame with the same `Enum`, and
  confirm no `ValueError`.

---

### P4. `_arrays.f64_vector` and `encode_labels`: the columns beside the matrix

- **Target file/symbol**: `python/mojoboost/_arrays.py`, `f64_vector` and
  `encode_labels`.
- **Ownership**: whoever owns `_arrays.py`.
- **Signatures**: unchanged.
- **Call site**: prepend to `f64_vector`:

```python
def f64_vector(y, n_rows, name="y"):
    from . import _sequence

    adapter = _sequence.vector_for(y)
    if adapter is not None:
        return adapter(y, n_rows, name)
    if np is not None:
        ...  # unchanged
```

and to `encode_labels`:

```python
def encode_labels(y, n_rows):
    from . import _sequence

    adapter = _sequence.labels_for(y)
    if adapter is not None:
        y = adapter(y, "y")
    if np is not None:
        ...  # unchanged
```

- **State flow**: `Dataset.__init__` passes `label`, `weight`,
  `init_score`, and `group` through `check_target` /
  `check_sample_weight` / `f64_vector`, all of which reach `f64_vector`.
  `MojoBoostClassifier.fit` reaches `encode_labels`.
- **Errors**: a null label becomes NaN in the adapter and is then refused
  by `check_target` with "y must not contain NaN or infinite values",
  which is the message that names the field. A dictionary column passed as
  a *weight* is refused by `arrow_f64_vector` by name.
- **Fallback**: None for numpy, lists, and pandas Series.
- **Dependency**: none, but pointless without P1.
- **Public API effect**: `Dataset(X, label=arrow_array)` and
  `clf.fit(X, polars_series)` work. `classes_` holds the caller's labels,
  not codes, because `labels_for` expands a dictionary column to its
  labels.
- **Minimal later validation (UNRUN)**:
  - `clf.fit(X, pa.array(["a", "b", "a"]).dictionary_encode())` gives
    `classes_ == ["a", "b"]`
  - a null in a label array raises with "must not contain NaN"

---

### P5. `_arrays.check_X_sparse`: the Arrow sparse bridge

- **Target file/symbol**: `python/mojoboost/_arrays.py`, `check_X_sparse`.
- **Ownership**: whoever owns `_arrays.py`.
- **Signature**: unchanged.
- **Call site**: prepend, before `_canonical_sparse`:

```python
    from . import _arrow

    if _arrow.is_arrow_sparse_tensor(X):
        X = _arrow.sparse_to_scipy(X, name)
```

- **State flow**: the SciPy matrix then goes through the existing
  canonicalization, which copies before it sorts and deduplicates, so the
  caller's tensor is never mutated.
- **Errors**: a COO or CSF tensor raises `TypeError` naming CSR and CSC.
- **Fallback**: unchanged for SciPy input.
- **Dependency**: none.
- **Serialization effect**: none.
- **Public API effect**: `est.fit(arrow_sparse_csr, y)` works, with the
  same restrictions the SciPy sparse path already documents (no GPU, no
  Python objective, no `eval_set`, no ranking, no `pred_leaf` /
  `pred_contrib`, no iteration slicing).
- **Minimal later validation (UNRUN)**: an Arrow `SparseCSRMatrix` fit
  equals the SciPy CSR fit of the same values.

---

### P6. `cv._take_column`: fold slicing for Arrow and polars columns

- **Target file/symbol**: `python/mojoboost/cv.py`, `_take_column`.
- **Ownership**: whoever owns `cv.py`.
- **Problem**: `_take_rows` already dispatches to pyarrow's `take` and to
  polars indexing, so a fold's *matrix* is sliced correctly. `_take_column`
  does `_np.asarray(column)[...]`, which for an Arrow array with nulls
  either produces an object array or raises, and which loses a polars
  `Enum`'s labels.
- **Signature**: unchanged, `_take_column(column, rows)`.
- **Call site**: prepend.

```python
def _take_column(column, rows):
    """One of a dataset's columns restricted to `rows`, or None."""
    if column is None:
        return None
    take = getattr(column, "take", None)
    if take is not None and hasattr(column, "type"):
        return take(list(rows))  # pyarrow
    if take is not None and hasattr(column, "dtype") and hasattr(
        column, "to_arrow"
    ):
        return take(list(rows))  # polars
    if _np is not None:
        ...  # unchanged
```

- **State flow**: the sliced column goes back into a per-fold `Dataset`,
  where P4 converts it.
- **Errors**: unchanged.
- **Fallback**: numpy and lists take the existing path.
- **Dependency**: P4, or the sliced Arrow column reaches an `f64_vector`
  that cannot read it.
- **Public API effect**: `mojoboost.cv` works on a `Dataset` built from an
  Arrow table with an Arrow label.
- **Minimal later validation (UNRUN)**: `cv` on an Arrow-backed dataset
  returns the same fold means as `cv` on the numpy equivalent.

---

### P7. `device_selection`: no patch needed

**Nothing to apply.** Recorded because it is the obvious next place to
look. `device_selection._shape_of` reads a two-element `shape` off
anything that has one, which an Arrow table and a polars frame both do, so
`Workload.from_data` and `explain_device_choice` already accept them.
`Batches` was given a `shape` property inside this lane for the same
reason, so it works too, and `device_selection.py` needs no edit at all.

- **Minimal later validation (UNRUN)**:
  `Workload.from_data(arrow_table).n_rows` is the table's row count;
  `Workload.from_data(batches).n_rows` is the summed batch row count.

---

### P8. Public export: `mojoboost.Batches`

- **Target file/symbol**: `python/mojoboost/__init__.py`, `__all__` and the
  import block after it.
- **Ownership**: whoever owns `__init__.py` (see
  `python/mojoboost/_public_api_plan.py` for the rules this follows).
- **Request**: exactly one new public name.

```python
__all__ = [
    ...
    "Batches",
    ...
]

from ._sequence import Batches  # noqa: E402
```

- **Why one name**: Arrow tables and polars frames need no public name at
  all, because they arrive through `Dataset` and `fit` as any other matrix
  does (P1). Batched input needs one, because wrapping is deliberately
  required rather than inferred: a list of rows and a list of batches are
  both lists, and guessing wrong would transpose a caller's data.
- **Cost of the import**: `_sequence` imports `_arrays`, `_arrow`, and
  `_polars`, none of which import numpy beyond what `_arrays` already
  imports, and none of which import pyarrow or polars at all. So
  `import mojoboost` gains three module objects and no dependency. It does
  not need the PEP 562 `__getattr__` treatment `_public_api_plan.py`
  reserves for costly submodules.
- **Collides with**: nothing. There is no `mojoboost.Batches` today and no
  submodule of that name.
- **Docstring to use**: the `Batches` class docstring, which already
  documents the opt-in rule and the drained-source behavior.
- **Dependency**: none, but `Batches` is only useful as a matrix once P1
  lands.
- **Minimal later validation (UNRUN)**: `mb.Batches` imports; `import
  mojoboost` still succeeds with pyarrow and polars uninstalled.

---

### P9. `Dataset`: take the assembled buffer without a second validation

- **Target file/symbol**: `python/mojoboost/basic.py`,
  `Dataset.__init__`, the `_arrays.check_X(data)` call.
- **Ownership**: whoever owns `basic.py`.
- **Status**: **optional**, and it is an efficiency patch, not a
  correctness one. `BatchedInput.dataset_kwargs()` already works against
  the unmodified `Dataset`.
- **What it saves**: with numpy, `Dataset(data=batched.matrix)` re-runs
  `np.asfortranarray` (a no-op on an already-Fortran float64 array) and
  `np.isinf(...).any()` (a full pass). Without numpy it is much worse:
  the flat column-major `array.array` cannot be re-read as a matrix at all,
  so `BatchedInput.as_rows()` builds a list of lists and `Dataset` converts
  that back into a flat buffer, which is two full copies.
- **Proposed call site**: accept a `BatchedInput` directly.

```python
        if isinstance(data, _sequence.BatchedInput):
            Xb = data.matrix
            n_rows, n_features = data.n_rows, data.n_features
            frame_names = data.names
            if categorical_feature is None and data.categories:
                categorical_feature = sorted(data.categories)
        else:
            Xb, n_rows, n_features, frame_names = _arrays.check_X(data)
```

- **State flow**: `self._x` holds the assembled buffer and `self._raw`
  holds the `BatchedInput`, so `get_data()` returns something meaningful
  and `free_raw_data=True` drops both.
- **Errors**: none new. The buffer was validated during assembly by the
  same functions `check_X` would call.
- **Fallback**: every other input takes the existing path.
- **Serialization effect**: none.
- **Public API effect**: `Dataset(batched_input)` in addition to
  `Dataset(**batched_input.dataset_kwargs())`.
- **Dependency**: P8 if `BatchedInput` should also be public; it need not
  be, since `materialize` is reached through `Batches` in this design only
  if a public `materialize` is also exported. **If P9 is taken, export
  `materialize` too, or `BatchedInput` is unreachable from public code.**
- **Minimal later validation (UNRUN)**: a `Dataset` built from a
  `BatchedInput` and one built from the equivalent numpy matrix have equal
  `num_bin()` and train to equal predictions.

---

### B1. Binding: per-column buffers and validity bitmaps

- **Target file/symbol**: `bindings/_mojoboost.mojo`, a new
  `dataset_create_columns` beside `dataset_create`.
- **Ownership**: whoever owns the bindings and `src/mojoboost/trainset.mojo`.
- **Status**: **request only**. Nothing in this lane depends on it, and it
  is the only change that would make the phrase "zero copy" true.
- **Problem**: `dataset_create` takes one address for the whole matrix and
  `_f64_list` copies it element by element into a `List[Float64]`. So even
  a perfectly eligible Arrow column is copied twice: once by
  `_arrow.column_major_arrow` into the column-major buffer, once in Mojo.
  Arrow columns are per-column contiguous and never one contiguous matrix,
  so the first copy exists only to satisfy the second's layout assumption.
- **Proposed signature**:

```mojo
def dataset_create_columns(
    addrs: PythonObject,      # n_features int addresses, one per column
    validity: PythonObject,   # n_features int addresses, 0 for "no nulls"
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,     # exactly today's params dict
) raises -> PythonObject
```

- **State flow**: the Python caller keeps every column buffer referenced
  for the duration of the call, as it keeps the single buffer today.
  `_arrow.describe_columns` already produces exactly the addresses,
  lengths, offsets, and bitmaps this would read, per column, with a
  `zero_copy_eligible` verdict and a `blocked_by` reason.
- **Semantics required**: a set bit in the validity bitmap means valid; a
  clear bit means missing and must bin as NaN does, so `use_missing`
  applies unchanged. A column with `validity == 0` has no nulls.
- **Errors**: a null bitmap for a column whose values buffer is absent, or
  a length shorter than `ceil(n_rows / 8)`, must raise rather than read
  past the end.
- **Fallback**: `dataset_create` stays, and the Python side uses the new
  entry point only when every column reports `zero_copy_eligible`.
- **Serialization effect**: none. Bins are fitted from values, and the
  values are the same values.
- **Public API effect**: none, if the Python side chooses between the two
  entry points internally.
- **Dependency**: `src/mojoboost/trainset.mojo` must accept per-column
  spans instead of one flat `List[Float64]`.
- **Minimal later validation (UNRUN)**: a dataset built through
  `dataset_create_columns` and one built through `dataset_create` from the
  same values have the same `num_bin()` and train to identical trees.

---

### B2. Native streaming: what `_sequence` would call

- **Target file/symbol**: `src/mojoboost/sequence.mojo` and
  `src/mojoboost/external_memory.mojo` (task 07's lane), plus bindings.
- **Ownership**: task 07. **Coordinate before designing anything**: this is
  a request for the Python side of an interface that lane owns.
- **Why**: `materialize` holds every batch and the assembled matrix at
  once. LightGBM's `lgb.Sequence` is a bounded-memory feature and this is
  not one; `docs/ECOSYSTEM_INPUTS.md` section 6 says so plainly.
- **Shape the Python side is ready for**, with no change to `_sequence.py`:

```python
handle = _mojoboost.dataset_begin(n_features, params)   # bins not yet fixed
for batch in batches:                                    # pass 1: quantiles
    buf, rows, _, _ = _sequence._batch_convert(batch, name, tables)
    _mojoboost.dataset_scan_batch(handle, _arrays.addr(buf), rows)
_mojoboost.dataset_fix_bins(handle)
for batch in batches:                                    # pass 2: write
    buf, rows, _, _ = _sequence._batch_convert(batch, name, tables)
    _mojoboost.dataset_push_batch(handle, _arrays.addr(buf), rows)
dataset = _mojoboost.dataset_finish(handle)
```

- **What `Batches` already provides for it**: the batches themselves
  (`__iter__`, `__getitem__`), their row counts (`row_counts()`), their
  offsets in the final matrix (`offsets()`), whether the source was drained
  (`drained`, which decides whether a second pass is possible), and the
  unified category tables (`unify_categories`, which must be computed
  before pass 1 because a category code must mean the same thing in every
  batch).
- **Constraint to design around**: a drained source cannot be read twice.
  Either the native side buffers pass 1 itself, or `Batches` must refuse a
  drained source for the streaming path. The second is one line in this
  lane's files and needs no decision here.
- **Errors**: cancellation and cleanup of a partially built dataset are
  task 07's to define; the Python side needs a `dataset_abort(handle)` to
  call from a `finally`.
- **Public API effect**: `Dataset(Batches(...), streaming=True)`, or a
  separate constructor. Not proposed here.
- **Minimal later validation (UNRUN)**: a streamed dataset and a
  materialized one from the same batches have equal `num_bin()` and train
  to identical trees.

---

### D1. `docs/LIGHTGBM_PARITY.md`: the three deferred rows

- **Target file/symbol**: `docs/LIGHTGBM_PARITY.md`, section 6 "Data
  inputs".
- **Ownership**: whoever owns the parity document.
- **Status**: apply **after P1**, not before. The rows are correct today.
- **Replacement rows**:

| `Sequence` / batched construction | partial | `mojoboost.Batches` takes Arrow record batches, polars frames, pandas frames, numpy arrays, or lists of rows and assembles one matrix, checking that the batches describe the same columns and unifying their category tables. It is not bounded memory, which LightGBM's is: see docs/ECOSYSTEM_INPUTS.md section 6 and task 07 |
| pyarrow tables and arrays | supported | Tables and record batches as a feature matrix, arrays and chunked arrays as `label`, `weight`, `init_score`, and `group`. Dictionary columns are categorical features carrying their labels, unified across chunks. Strings, temporal, decimal, and nested types are refused with the cast that fixes each, and `int64` past 2**53 is refused rather than rounded. Sparse CSR/CSC tensors bridge to the SciPy sparse path. Nothing is read in place: docs/ECOSYSTEM_INPUTS.md section 3 |
| polars frames | supported | Through Arrow. `Categorical` and `Enum` carry their labels rather than their physical codes, so a model fitted on one frame scores another. LazyFrames are refused; call `.collect()` |

---

### D2. `python/pyproject.toml`: the two extras

- **Target file/symbol**: `python/pyproject.toml`,
  `[project.optional-dependencies]`.
- **Ownership**: whoever owns packaging (`handoffs/release_01_pypi.md`
  raised the same request and deferred it for lack of a guarded import;
  there are now two).
- **Patch**:

```toml
# Arrow tables, record batches, and arrays. The adapter duck-types the
# Arrow protocol and imports nothing from pyarrow; pyarrow is what a
# caller uses to build the table in the first place.
arrow = ["pyarrow"]
# polars frames, converted through Arrow by the same adapter.
polars = ["polars"]
all = ["mojoboost[numpy,pandas,scipy,scikit-learn,dask,arrow,polars]"]
```

- **Precedent**: the `scipy` extra is worded the same way and for the same
  reason: the wrapper duck-types the interface and imports nothing.
- **Dependency**: none. `pixi.toml`'s `pytest` feature already installs
  both, so the test environment needs no change.
- **Minimal later validation (UNRUN)**: `pip install mojoboost[arrow]`
  resolves; `packaging/matrix/validate_matrix.py` still passes.

---

## 4. Decisions worth reviewing

1. **Refusing `int64` past 2**53.** Stricter than LightGBM, stricter than
   the pandas path here, and the same principle as the existing infinity
   refusal. An id column is the realistic case and silently merging two ids
   into one bin is a wrong split. Reversible: delete
   `_check_wide_int_chunk`'s two call sites.
2. **Refusing duplicate Arrow column names.** Arrow permits them and a
   join produces them; `categorical_feature` by name would land on the
   first. Reversible: delete `_require_unique_names`.
3. **Refusing temporal and decimal rather than casting.** Casting a
   timestamp picks an epoch, a unit, and a time zone on the caller's
   behalf. The message names the cast.
4. **A `null`-typed column becomes all-NaN rather than raising.** It is the
   exact conversion, not a lossy one, and an all-missing feature is legal
   in the dense path already.
5. **Batched input is opt-in.** A list of rows and a list of batches are
   both lists. Auto-detecting would transpose data.
6. **No pyarrow or polars import anywhere.** Structural recognition only,
   the `mojoboost.dask` pattern. Costs one risk: an object that answers
   `column_names`, `num_rows`, `num_columns`, and a callable `column`
   without being an Arrow table would be treated as one. No such object is
   known.
7. **`_polars` holds no conversion.** Every rule has one implementation, in
   `_arrow`. The cost is that a polars-only bug fix is an Arrow fix.

## 5. Known gaps in this lane's own files

- A caller who hands the *same unwrapped* `lgb.Sequence` to
  `frame_categories` and then to `check_X` pays for the schema check and
  the category unification twice, because each call wraps it in a fresh
  `Batches`. Wrapping once in `Batches(...)` avoids it: the wrapper caches
  both. Nothing forces the caller to, and nothing breaks if they do not.
- An unhashable dictionary label (a list-valued category) is never equal to
  itself, so it produces one category per chunk it appears in. This matches
  `_arrays._codes_from_labels`, which treats an unhashable label as always
  unknown, and no Arrow scalar type produces one in practice.
- `_arrow.describe_columns` reports `dictionary_size` as None when the
  dictionary is too large to encode, because it swallows the `ValueError`
  that `column_major_arrow` would raise. A describer should not raise; the
  converter still does.
- Nothing here reads the Arrow C Data Interface (`__arrow_c_array__`,
  `__arrow_c_stream__`), which is how a non-pyarrow producer would hand
  over a column without pyarrow in the picture at all. It is the natural
  next step for the structural-recognition approach and needs no new
  dependency.

## 6. Validation status

Every item is UNRUN. Nothing in this lane was executed.

| Check | Status |
|---|---|
| `python -m py_compile` on the three modules | UNRUN |
| `import mojoboost` with pyarrow and polars absent | UNRUN |
| Arrow fit equals numpy fit on the same values | UNRUN |
| polars fit equals pandas fit on the same values | UNRUN |
| dictionary column fit equals pandas `category` fit | UNRUN |
| chunked dictionary with disagreeing chunks unifies correctly | UNRUN |
| nulls bin as NaN does | UNRUN |
| every refused dtype raises with its cast named | UNRUN |
| `int64` past 2**53 raises | UNRUN |
| duplicate Arrow column names raise | UNRUN |
| batches with different widths, names, or category declarations raise | UNRUN |
| `BatchedInput.dataset_kwargs()` builds a `Dataset` that trains | UNRUN |
| `group()` on query ids spanning a batch boundary | UNRUN |
| numpy-free install: `as_rows()` path | UNRUN |
