# Ecosystem inputs: Arrow, polars, and batches

What mojotrees does with an Arrow table, a polars frame, or data that
arrives in pieces. The code is `python/mojotrees/_arrow.py`,
`python/mojotrees/_polars.py`, and `python/mojotrees/_sequence.py`; the
integration state, and the patches that finish it, are in
`handoffs/remaining_10_ecosystem_inputs.md`.

**Status.** The adapters are written and self-contained. They are not yet
reachable from `mojotrees.Dataset`, from the estimators, or from any public
name, because the four dispatch points that would reach them live in files
this lane does not own. Read every "converts to" below as a statement about
the adapter, not about `mb.Dataset(arrow_table)`, until the handoff's
patches land. Nothing here has been executed: see "What has not been run".

## 1. The shape of the thing

mojotrees's extension module reads one thing: a contiguous float64 buffer,
column-major, whose address crosses the boundary
(`dataset_create` in `bindings/_mojotrees.mojo`). `_arrays.py` is what
turns numpy arrays, pandas frames, plain sequences, and SciPy sparse
matrices into that. The three adapters add Arrow, polars, and batches in
exactly the same currency, mirroring the `_arrays` functions by name and by
return shape:

| `_arrays` | `_arrow` | `_polars` | `_sequence` |
|---|---|---|---|
| `feature_names` | `arrow_feature_names` | `polars_feature_names` | `batch_feature_names` |
| `frame_categories` | `arrow_frame_categories` | `polars_frame_categories` | `batch_categories` |
| `check_X` | `check_X_arrow` | `check_X_polars` | `check_X_batches` |
| `f64_vector` | `arrow_f64_vector` | `polars_f64_vector` | n/a |
| `encode_labels` input | `arrow_to_pylist` | `polars_to_pylist` | n/a |

`_sequence.adapter_for`, `categories_for`, `names_for`, `vector_for`, and
`labels_for` pick the right one for an object, and each returns a callable
with the signature of the `_arrays` function it stands in for. That is the
whole dispatcher, and it exists so that `Dataset`, the estimators, and `cv`
do not each grow their own.

## 2. No optional dependency is ever imported

Neither adapter imports pyarrow or polars, at import time or lazily.
Recognition is structural (does this object answer `column_names`? does it
answer `to_arrow`?) the way `mojotrees.dask.is_dask_collection()` is, and
every operation is a method the object itself offers. `arrow_available()`
and `polars_available()` answer the "is it installed" question with
`importlib.util.find_spec`, which does not execute the package either.

Two consequences worth stating:

- `import mojotrees` stays cheap and total, which is the rule in
  `python/mojotrees/_public_api_plan.py`.
- The adapters work against the Arrow and polars *protocols*, so they do
  not have to track a library version to keep working.

## 3. Zero copy: what is true

Arrow columns are already flat, typed, contiguous buffers, so the obvious
question is whether mojotrees trains on them in place. It does not, and
these modules never say it does. Two independent things stand in the way.

**The Arrow side is often ineligible.** A column could be read in place only
if it is a single chunk of `double` at offset 0. Chunking, a narrower
numeric type, a dictionary encoding, and a sliced view each rule it out.
`_arrow.describe_columns(table)` reports this per column, with the address
and length the column does expose, as a `BufferPlan`:

| field | meaning |
|---|---|
| `values_addr`, `values_length` | the single chunk's values buffer, or 0 |
| `validity_addr` | the validity bitmap, or 0 when the chunk has none |
| `offset` | a slice's start; a reader ignoring it reads the wrong rows |
| `n_chunks`, `null_count`, `dictionary_size` | why it is or is not eligible |
| `zero_copy_eligible`, `blocked_by` | the verdict and the reason |

**The mojotrees side is not ready for it in any case.** `dataset_create`
takes one address for the whole matrix and copies it element by element
into a `List[Float64]` through `_f64_list`. There is no path that reads N
per-column pointers and none that reads a validity bitmap. So today an
Arrow column is copied twice on the way in: once by the adapter, into the
column-major buffer, and once in Mojo. That second copy is paid by the
numpy path too and is not an Arrow tax.

`BufferPlan.zero_copy_eligible` therefore means "the Arrow buffer could be
read in place by a reader that accepted per-column pointers". It never
means "mojotrees read it in place". The binding change that would make the
first meaning matter is request **B1** in the handoff.

## 4. Arrow types

`_arrow.column_kind(type)` sorts every Arrow type into one of five kinds.
The rule behind the table: a conversion that could change a value, or that
picks a unit on the caller's behalf, is refused rather than performed.

| Arrow type | Kind | Result |
|---|---|---|
| `int8` … `int32`, `uint8` … `uint32` | numeric | exact in float64 |
| `int64`, `uint64` | numeric | exact in float64 **after a 2**53** check; see below |
| `halffloat`, `float`, `double` | numeric | exact (widening) |
| `bool` | bool | 0.0 and 1.0 |
| `dictionary<...>` | dictionary | integer category codes, -1 for missing or unknown |
| `null` | null | every row NaN, which is exact for a column that holds only nulls |
| `string`, `large_string`, `string_view` | refused | cast to `dictionary` and declare the column categorical, or encode codes yourself |
| `binary`, `large_binary`, `binary_view`, `fixed_size_binary` | refused | no numeric reading of opaque bytes |
| `date32`, `date64`, `timestamp`, `time32`, `time64`, `duration` | refused | cast to `int64` yourself, which makes the epoch, unit, and time zone your choice |
| `interval`, `month_day_nano_interval` | refused | no single number; split into components |
| `decimal32/64/128/256` | refused | cast to `double`, so the precision loss is visible in your code |
| `list`, `large_list`, `fixed_size_list`, `list_view`, `map`, `struct` | refused | a feature is one number per row; flatten first |
| `union`, `sparse_union`, `dense_union` | refused | split into one column per type |
| `run_end_encoded` | refused | `combine_chunks()` or cast to the value type |
| extension types | refused | hand over `.storage` |

Every refusal names the cast that fixes it. `_REFUSED_TYPES` in `_arrow.py`
is that table.

### The 2**53 check

`int64` and `uint64` can hold values float64 cannot represent exactly. Past
2**53 two adjacent ids collapse onto one float, and the binner puts them in
the same bin, which is a wrong split rather than a slow one. The adapter
refuses those columns and says so. LightGBM has the same exposure through
numpy and does not check. This is one of the two places the Arrow path is
deliberately stricter than the pandas path.

### Nulls

A null becomes NaN, which is mojotrees's missing marker end to end: the
binner reserves a bin for it, each node carries a default direction, and
`use_missing` (on by default) controls the whole business. A numpy float
column of NaN and an Arrow column of nulls describe the same rows to the
binner.

A null in `label`, `weight`, or `init_score` is a different matter and is
refused, by `_arrays.check_target` and `check_sample_weight`, with the
message that names the field. The adapter converts; those functions judge.

### Chunked dictionaries

Arrow lets every chunk of a `ChunkedArray` carry its own dictionary, and a
table read from several files or row groups routinely does. The adapter
unifies them into one table per column, in first-appearance order, and
remaps each chunk's indices into it. Without that, code 3 would mean a
different category in chunk 2 than in chunk 1 and the binner would split on
the union of two unrelated things. pandas cannot produce this case; Arrow
can, which is why it is handled explicitly.

### Duplicate column names

Arrow permits two fields with the same name and a join produces them
routinely. The adapter refuses them, because `categorical_feature=["price"]`
resolves a name to a position with `names.index(...)`, so with two `price`
columns the declaration would land on the first and quietly leave the
second numeric. This is the second place the Arrow path is stricter than
the pandas path.

### Arrow inputs that are not tables

| Input | Result |
|---|---|
| `Array`, `ChunkedArray` | a single column, so not a feature matrix. Refused as `X` with the message naming `label` / `weight` / `init_score`, which is where a single column belongs. Accepted *as* those, through `arrow_f64_vector` |
| `pyarrow.dataset.Dataset`, `RecordBatchReader` | streaming. Refused rather than materialized behind your back: call `.to_table()` if it fits, or hand the batches to `_sequence` |
| `SparseCSRMatrix`, `SparseCSCMatrix` | bridged by `_arrow.sparse_to_scipy`, which returns the SciPy matrix mojotrees's sparse path already takes. Nothing is densified |
| `SparseCOOTensor`, `SparseCSFTensor` | refused; convert to CSR or CSC first |

## 5. polars

A polars frame is Arrow underneath, so `_polars` is thin on purpose: it
decides what it was handed, refuses what polars can express and mojotrees
cannot train on, and hands `to_arrow()` to `_arrow`, which owns the
conversion, the null rule, the dictionary unification, the 2**53 check, and
the layout. There is one implementation of each of those and it is not in
`_polars.py`.

| polars dtype | Result |
|---|---|
| `Int8` … `Int64`, `UInt8` … `UInt64`, `Float32`, `Float64` | numeric |
| `Boolean` | 0.0 and 1.0 |
| `Categorical`, `Enum` | category codes, through the *labels* (see below) |
| `Null` | every row NaN |
| `String` / `Utf8` | refused: `.cast(pl.Categorical)` or `.cast(pl.Enum([...]))`, or encode codes yourself |
| `Binary` | refused |
| `Date`, `Datetime`, `Duration`, `Time` | refused: `.dt.epoch('us')` or `.cast(pl.Int64)` |
| `Decimal` | refused: `.cast(pl.Float64)` |
| `Int128` | refused: no exact float64 reading |
| `List`, `Array`, `Struct` | refused: `.unnest()` or explode first |
| `Object`, `Unknown` | refused |
| anything unrecognized | left to `_arrow` after `to_arrow()` |

The last row is a rule, not an omission: the polars preflight exists to put
polars' own vocabulary in the error message, never to reach a different
verdict from Arrow's. A dtype it has not heard of is Arrow's to judge.

**`Categorical` versus `Enum`, and why labels matter.** `Enum` fixes its
categories when the column is created, in the declared order. `Categorical`
numbers its categories in the order the string cache happened to see them,
which is a property of the session and not of the data, so the same labels
in two frames can carry different physical codes. Both reach the estimator
as *labels*, never as physical codes, and a prediction frame is encoded
through the fitted labels. That is what lets a model fitted on one frame
score another correctly.

**LazyFrames are refused, by name.** A `LazyFrame` has read nothing yet.
Calling `.collect()` on the caller's behalf would run a query plan of
unknown cost inside a function they think is checking an argument.

**`to_arrow()` is cheap and the rest is not.** polars stores its columns in
Arrow layout, so `to_arrow()` on a null-free numeric column is close to
free. Everything after it copies, exactly as in section 3.

## 6. Batches

`_sequence.Batches` takes data that arrives in pieces and `materialize`
assembles it:

```python
from mojotrees import _sequence   # internal today; see handoff request P1

data = _sequence.materialize(
    _sequence.Batches(record_batches),
    label_column="target",
)
train_set = mojotrees.Dataset(**data.dataset_kwargs())
```

A batch may be an Arrow table or record batch, a polars frame, a pandas
frame, a numpy array, or a list of rows, and the batches need not all be
the same kind. `label_column`, `weight_column`, and `query_column` name
columns that travel inside the batches: each is read out of every batch,
dropped from the features, and returned on the result, so a Parquet dataset
with the label in it needs no separate array.

What is checked, batch by batch, before anything is converted:

- **Width.** Batches with different widths would make a ragged matrix.
  This is the only one LightGBM checks.
- **Names.** Batches are concatenated by position, so their columns must
  match in order as well as in name.
- **Named or not, consistently.** Half a named matrix cannot be identified.
- **Categorical declarations.** A column is categorical for the whole
  matrix or not at all.
- **Category tables.** Unified across batches by label, on the same rule
  `_arrow` uses across the chunks of one column, so batch 5's third
  category is batch 2's third category.

`Batches` answers `shape`, `num_data()`, `row_counts()`, and `offsets()`
without converting anything, and caches the schema check and the unified
category tables, so a fit that asks for the names, then the categories,
then the matrix pays for the first two once. Answering `shape` is also what
puts a batched input into `mojotrees.device_selection.Workload.from_data`
and `explain_device_choice`, which read a two-element `shape` off whatever
they are given; an Arrow table and a polars frame already have one.

`BatchedInput` carries the assembled `matrix` (column-major float64, the
shape `_arrays.addr` and `column_view` read), the `names`, the unified
`categories`, the `offsets` of each batch in the matrix, and the `label`,
`weight`, and `query_ids` if they were named. `group()` turns query ids
into LightGBM's per-query row counts; a query split across two batches is
fine, because the batches are concatenated in order and its rows are still
one unbroken run.

### It is not bounded memory

`materialize` holds every batch and the assembled matrix at once, so peak
memory is *higher* than handing over one matrix, not lower. It is worth
having anyway, because the pieces are usually already in memory and the
alternative is the caller writing this loop themselves. It is also worth
being blunt about, because the LightGBM feature it resembles,
`lgb.Dataset(lgb.Sequence(...))`, is a bounded-memory feature.

Bounded memory needs a binner that accepts data in pieces: two passes over
the batches, quantiles built incrementally, bins fixed before any row is
written. That is a native question, and the core is
`src/mojotrees/sequence.mojo` with `src/mojotrees/external_memory.mojo`
(task 07). The Python side of it is a thin loop over `Batches`, which is
shaped for it: the batches, their row counts (`row_counts()`), and their
offsets (`offsets()`) are all available separately from the assembly step.

### A list is not batches

`[[1.0, 2.0], [3.0, 4.0]]` is a matrix of two rows, and `_arrays` has
always read it as one. A list of frames is batches. Both are lists, so the
difference cannot be inferred from the object, and inferring it wrong would
silently transpose a caller's data. Batched input is therefore opt-in:
either the object implements LightGBM's `Sequence` protocol (`__len__`,
`__getitem__`, and `batch_size`, all three), or the caller wraps it in
`Batches(...)`. Nothing else is ever read as batches.

A source that can be indexed is held and read twice, which is what a
two-pass binner needs. A source that can only be iterated (a generator, a
`RecordBatchReader`) is drained into a list on construction, because
reading a stream twice returns nothing the second time. `Batches.drained`
says which happened.

## 7. What has not been run

Nothing in this lane has been executed. No test was written or run, no
build was made, no Python was started, and pyarrow and polars were never
imported. Every behavior above is a reading of the code as written and is
marked UNRUN in the handoff, which lists the minimal checks that would
confirm each of them. `pixi.toml`'s `pytest` feature already installs
pyarrow and polars, so the checks have an environment to run in.

`docs/LIGHTGBM_PARITY.md` still records "pyarrow tables and arrays",
"polars frames", and "`Sequence` / batched construction" as deferred to
task 10. That stays correct until the dispatch patches land; the handoff
carries the replacement rows.
