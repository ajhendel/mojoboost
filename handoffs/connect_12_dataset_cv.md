# Connect 12 handoff: dataset inputs, prepared data, CV, continued training

Files edited by this lane, and nothing else:

Round 1 (the original ownership list):

- `src/mojoboost/trainset.mojo`
- `src/mojoboost/raw_data.mojo`
- `python/mojoboost/_arrays.py`
- `python/mojoboost/cv.py`
- `handoffs/connect_12_dataset_cv.md` (this file)

Round 2, after the owner explicitly asked this lane to implement its own
§6 patch requests rather than file them:

- `src/mojoboost/serialize.mojo` (§6.5)
- `src/mojoboost/__init__.mojo` (§6.4)
- `bindings/_mojoboost.mojo` (§6.2(a) and the registration of every
  `dataset_bindings` entry point)
- `python/mojoboost/basic.py` (§6.3)

This lane committed nothing. §6 is kept below with each item marked
**LANDED** and by whom, so a reader can tell what was requested from what
was done.

**Shared-checkout note.** Other lanes ran sweeping commits (860b1cf, then
e6f3959, e28a24d, 63aad82) partway through this work, so this lane's edits
are already inside those commits' trees rather than sitting as a
working-tree diff. They are intact and unmodified — verified by re-grepping
every added symbol — but a reviewer looking for this lane's diff will not
find it under `git status`. Those same commits landed §6.1 and added
`bindings/dataset_bindings.mojo`, both accounted for below.

## 1. What was already there

**Ingestion, Mojo.** Three implementations of "take a user matrix and bin
it", all live:

| Where | Takes | Reached from |
| --- | --- | --- |
| `binning.fit_bins` + `BinMapper.transform` | dense column-major | `model.fit`, `trainset.Dataset` |
| `sparse.fit_bins_csc` + `sparse.transform_csc` | CSC | `model_sparse.fit_csc`, the sklearn estimators |
| `raw_data.RawData` | either, dispatching to the two above | **nothing at all** |

`raw_data.mojo` was the alternate ingestion code the task describes. It was
not exported from `src/mojoboost/__init__.mojo`, not imported by any module,
and not referenced by any test or binding; the only mention of `RawData`
anywhere else in the tree was a sentence in `trainset.mojo`'s docstring
pointing at a `dataset.mojo` that no longer exists. Its own docstring ended
with "`Dataset` stays dense", which was accurate.

**`trainset.Dataset`** was dense-only, retained no raw matrix, called
`fit_bins`/`transform` inline, and had no notion of a reference dataset or
of a subset. `train_dataset`, `train_dataset_multiclass`,
`train_dataset_ranker`, `update_dataset`, `update_dataset_multiclass` all
read `dataset.data`, the dense `BinnedMatrix`.

**Ingestion, Python.** `_arrays.check_X` (dense) and `_arrays.check_X_sparse`
+ `SparseBuffers` (sparse). The sklearn estimators in `__init__.py` use both;
`basic.Dataset` used only `check_X`, so the functional API was dense-only on
the Python side as well as the Mojo side. `basic.Dataset` accepted
`reference=` and, per its own comment, "a reference dataset changes nothing
here": it validated the shape and the binning params and then discarded it.

**Serialization.** `serialize.mojo` wrote models only. `_write_mapper` /
`_read_mapper` already existed there as the single mapper codec, including
the categorical tables and the missing-bin flags; nothing outside that file
could reach them, so there was no way to persist a binning without a model
around it.

**CV.** `python/mojoboost/cv.py` was already complete against the
`Dataset`/`Booster` contract: explicit `folds=`, scikit-learn splitters,
shuffled and stratified folds, whole-query ranking folds, multiple metrics,
custom metrics via `_metric_specs`, callbacks, across-fold early-stopping
consensus, `return_cvbooster`, and mean/stdv histories. There is exactly one
`cv()` and one `CVBooster` in the tree; no duplicate was found. Its one
disconnection was that neither name was exported at the top level — `grep -c
cv python/mojoboost/__init__.py` was 0 when this lane started — so
`mojoboost.cv(...)`, the call the module docstring and
`_public_api_plan.PROPOSED_ADDITIONS` both describe, did not exist. The
public API lane landed that export mid-lane; see §6.1.

## 2. Call path, before

```
bindings.dataset_create ──> trainset.Dataset.__init__ ──> fit_bins / transform
                                     │
                                     └─> train_dataset ──> boosting.train | train_gpu.train_gpu
sparse X ──> estimators only ──> model_sparse.fit_csc ──> fit_bins_csc / transform_csc
raw_data.RawData ──> (nothing)
basic.Dataset(reference=…) ──> validated, then dropped
cv() ──> basic.Dataset per fold per side ──> the same dense path
serialize ──> models only
```

## 3. Call path, after

```
basic.Dataset(X, …)          ──> dataset_create            ──> Dataset.__init__      (dense, borrowed)
basic.Dataset(sparse X, …)   ──> dataset_create_csc        ──> Dataset.from_csc      ──> RawData
basic.Dataset(…, reference=) ──> dataset_create_reference  ──> Dataset.from_reference
basic.Dataset.subset(rows)   ──> dataset_subset            ──> Dataset.subset | subset_shared_binning
basic.Dataset.save_binned    ──> dataset_save              ──> serialize.save_dataset
basic.Dataset.load_binned    ──> dataset_load              ──> serialize.load_dataset ──> Dataset.from_binned_*

train_dataset            ──> is_sparse ? boosting_sparse.train_sparse : train | train_gpu
train_dataset_multiclass ──> is_sparse ? train_multiclass_sparse      : train_multiclass
train_dataset_ranker     ──> dense only, raises on sparse
update_dataset(_multiclass) ──> dense only, raises on sparse

Booster.predict(sparse X)    ──> predict_csr / predict_raw_csr / predict_proba_csr
cv() ──> _arrays.take_rows / take_column ──> basic.Dataset per fold (still dense, still Python-sliced)
```

`RawData` is now on the real path for every ingestion that is not a borrowed
dense matrix, and is the only thing that knows how to select rows out of a
raw matrix in Mojo. Every one of those Python entry points reaches
`trainset.Dataset`; none of them re-implements binning, selection, or
serialization on the Python side.

## 4. Connections completed

### 4.1 `raw_data.mojo`

- Docstring rewritten: it now names its consumer (`trainset.Dataset`),
  states why the dense constructor is the one path that does not build a
  `RawData` (it is handed a borrowed matrix and materializing one would copy
  `n_rows * n_features` floats), and drops the stale `dataset.mojo`
  reference and the "`Dataset` stays dense" conclusion.
- `RawData.none()` / `is_empty()`: the placeholder a `Dataset` field holds
  when the input was not retained. A struct field cannot be absent.
- `RawData.check_rows(rows)`: the selection contract — non-empty, in range,
  strictly ascending. Ascending is load-bearing: a CSC column's row indices
  are ascending, so a reordering or repeating selection would either break
  that invariant or need a per-column sort.
- `RawData.subset(rows)`: order-preserving row selection in the *same*
  representation. Dense gathers per column; sparse drops the entries of rows
  not taken and renumbers the survivors in one O(nnz) pass, producing
  canonical CSC without a sort.

### 4.2 `trainset.Dataset` owns dense *and* sparse

- Fields added: `sparse_data: SparseBinnedMatrix`, `is_sparse`,
  `raw: RawData`, `borrowed_binning`, `n_rows`, `n_features`. `data` keeps
  its name and meaning (the dense binned matrix). The unused matrix is a
  structurally valid empty one (`_empty_binned` / `_empty_sparse_binned`),
  mirroring how `RawData` holds the representation it is not using.
- `Dataset.from_raw` is the authoritative constructor: it takes a `RawData`
  by value, bins through `RawData.fit_mapper` and
  `transform_dense`/`transform_sparse`, and assembles the struct. `from_csc`
  and `from_csr` are one-line wrappers over it.
- The dense `Dataset(features, n_rows, n_features, …)` constructor keeps its
  exact positional signature, so `bindings.dataset_create`'s call is
  unchanged apart from the appended `keep_raw`, and keeps binning the
  borrowed matrix in place. `keep_raw` defaults to `False`; it is the only
  thing that copies.
- `from_binned_dense` / `from_binned_sparse` assemble a `Dataset` from an
  already-binned matrix and a mapper, validating that the two agree
  (`n_features`, `n_bins`, and the dense cell count) before trusting them.
  They exist for `serialize.load_dataset` and are the only way to get a
  `Dataset` that was never fitted in this process.
- Every dataset, however constructed, is assembled by one internal
  full-field `__init__`, and every dataset's columns are validated by one
  `_check_columns`, so a CSC dataset rejects exactly what a dense one
  rejects.
- Training dispatches on `is_sparse`: `train_sparse` for single-output,
  `train_multiclass_sparse` for softmax. The model that comes back is an
  ordinary `Model`/`MulticlassModel` carrying the mapper, so it serializes,
  loads, and predicts on dense rows identically either way.
- Unsupported sparse cases fail clearly instead of densifying:
  `device='gpu'` (the GPU trainer reads a dense binned matrix — `gpu_sparse`
  has a histogram builder, not a trainer), LambdaRank (`train_ranker` reads a
  `BinnedMatrix`), GOSS under sparse multiclass (`train_multiclass_sparse`
  takes no `GossParams`), and continued training (`boosting_sparse` has no
  `train_more` counterpart). Each message names the reason and the way out.

### 4.3 Reference binning, subsets, leakage

Two constructions, named for which one they are, because the difference is
the whole leakage question:

- **fits its own bins** — `Dataset.from_raw`, `Dataset.subset(rows)`. The
  rows left out had no say in the edges. This is what a fold or a held-out
  split wants.
- **reuses a fitted mapper** — `Dataset.from_reference(reference, raw, …)`,
  `Dataset.subset_shared_binning(rows)`. Bin indices mean what they mean in
  the reference, which is what a validation set scored by a model trained on
  the reference needs, and what `update_dataset` requires. Using it for a
  fold is the leak.

`from_reference` takes the feature count, binning parameters, feature names,
and categorical declaration from the reference, because those describe the
columns rather than the rows. `subset` recomputes a ranking dataset's
`group` and refuses a selection that takes part of a query (`_subset_group`),
which is the ranking form of the same leak.

The distinction survives all the way out to Python: `Dataset.subset(rows)`
defaults to `shared_binning=False`, and `Dataset(reference=…)` is the only
Python-side way to ask for shared binning at construction.

### 4.4 Consolidated metadata and accessors

`feature_name(i)` (LightGBM's `Column_<i>` fallback), `is_categorical(f)`,
`has_label/has_weight/has_group/has_init_score/has_raw`, `nnz()`,
`matches_binning(other)` (mapper equality *and* representation), and
`raw_matrix()`, which raises rather than returning an empty matrix when the
input was not retained.

### 4.5 Python: `_arrays` and `cv`

- `_arrays.take_rows` / `take_column`: the layout dispatch that selects rows
  out of a numpy array, a pandas frame, a pyarrow table, or a polars frame
  now lives next to the rest of the buffer plumbing, which is the only place
  that already knows how to read each of those. `cv._take_rows` /
  `_take_column` are thin delegations, so nothing importing them breaks.
  Sparse input is still refused there, with a message that names the reason
  (`basic.Dataset` slices dense input only) rather than being sliced into a
  failure two calls later.
- `_arrays.i64_vector`: the int64 row buffer `dataset_subset` reads. It is
  the counterpart of `f64_vector` and refuses fractional values rather than
  truncating them into a different selection.
- `cv.py` docstring: a **Parallelism** section (folds run sequentially, the
  native trainer owns the cores through `MOJOBOOST_NUM_WORKERS`, and a pool
  over folds must divide that first or `nfold` trainers each claim every
  core), a correction that only the training side of a fold is ever binned
  (a held-out `Dataset` is never `construct()`ed, because `Booster.predict`
  scores it through the *model's* mapper from its raw matrix), and the
  reference rule stated in the same terms as the Mojo side.
- `_fold_dataset` documents that `reference=` is deliberately not passed on.
- `CVBooster.predict_mean(...)`: the across-fold mean of `predict`, with the
  same slicing rules, refused for a ranking model because a LambdaRank score
  is comparable only within one model's ordering of one query. `predict()`
  still returns one prediction per fold and guesses at nothing.

### 4.6 Prepared-table serialization (`serialize.mojo`)

Written in `serialize.mojo` rather than in `trainset.mojo`, so the mapper
codec stays single: `save_dataset` calls the same `_write_mapper` the model
writer calls, and a prepared table's mapper section is byte-identical to a
model's. No import cycle results — nothing in `trainset`'s import closure
imports `serialize`.

- Own magic and own version: `mojoboost-dataset d1`, followed by the shared
  `CURRENT_FORMAT_VERSION`. A prepared table is not a model file and is not
  loadable as one.
- `file_kind(path)` returns `"objective"`, `"multiclass"`, or `"dataset"`.
  `model_file_kind` now raises a message naming a prepared table when it is
  handed one, instead of failing on the magic.
- Payload: shape, `max_bin`, `use_missing`, `is_sparse`, `borrowed_binning`,
  the feature names, the categorical declaration, the mapper section, then
  the binned matrix (`bins` for dense, `row_index`/`bin`/`col_offsets`/
  `default_bin` for sparse), then the four optional columns, each with its
  length so an absent one round-trips as absent.
- The raw matrix is **not** part of it. A reloaded table reports
  `has_raw()` false, refuses `subset`, and cannot be predicted on, rather
  than pretending it can refit bins from bins.

### 4.7 Bindings

`bindings/dataset_bindings.mojo` (written by the bindings lane from §6.2)
holds the constructors and accessors; **none of them were registered** in
`_mojoboost.mojo`'s `PythonModuleBuilder`, which is the single place a name
becomes importable from Python, so the whole file was dead code. This lane
added the `from dataset_bindings import (…)` line and the twelve
`m.def_function` registrations, plus:

- `dataset_create` now reads `keep_raw` from `params` and passes it as the
  12th positional argument (§6.2(a)).
- `dataset_save(dataset, path)`, `dataset_load(path)`, `file_kind(path)` —
  the entry points `save_binned` / `load_binned` call.

`bindings/build.sh` already passes `-I bindings` and already names
`dataset_bindings` in its comment, so the sibling import resolves the same
way `basic_bindings` and `objective_bindings` already do.

### 4.8 `basic.Dataset` reaches all of it

- **Sparse.** `__init__` dispatches on `_arrays.is_sparse(data)` to
  `check_X_sparse(data, "csc")` and keeps the `SparseBuffers`; `construct()`
  folds `buffers.params()` into the params dict and calls
  `dataset_create_csc`. `is_sparse`, `nnz()`, and `metadata()` describe it,
  and `__repr__` says which layout it is.
- **`reference=` does something.** `construct()` routes through
  `dataset_create_reference` when a reference is set, so the validation set
  is binned by the training set's mapper. This was the one wrong-answer path
  in the group: the argument was accepted, validated, and dropped, so a
  caller who asked for shared binning silently got a set binned over its own
  rows. `_check_reference` also now rejects a categorical declaration that
  disagrees with the reference's, and raises `NotImplementedError` for
  sparse + reference rather than silently binning independently.
- **`keep_raw`** is exposed (default `False`) and passed through, orthogonal
  to `free_raw_data`, which controls the Python-side copy.
- **`subset(rows, shared_binning=False)`** calls `dataset_subset`. The
  per-row columns come back *from the native subset* rather than being
  sliced again in Python, so the whole-query ranking rule lives in exactly
  one place (`trainset._subset_group`). When the Python side still holds a
  dense raw matrix the result is an ordinary `Dataset` around the native
  handle, so it can also be predicted on; otherwise it is a handle-backed
  `Dataset._from_handle`.
- **`save_binned` / `load_binned`** wrap the two serialize entry points.
  `_from_handle` reads the shape, params, names, categorical declaration,
  and the four columns natively, because a dataset this process did not bin
  has no other source for them.
- **`_Config.binding_params`** resolves a sparse dataset to the CPU, and
  raises for an explicit `device='gpu'` naming the missing sparse GPU
  trainer, rather than densifying behind the caller.
- **`Booster.predict`** branches to `_predict_sparse`, which calls
  `predict_csr` / `predict_raw_csr` / `predict_proba_csr`. Iteration slicing
  and multiclass raw scores have no sparse entry point and are refused with
  the reason. `_predict_data` was factored out of `_check_X` so both paths
  give the same message for a `Dataset` that has no raw matrix.

## 5. Duplicates fused or quarantined

- **Fused.** `raw_data.RawData` was an unused parallel ingestion path; it is
  now the ingestion path for everything except the borrowed dense
  constructor, and the two agree by construction because both call
  `binning.fit_bins` with the same arguments. Nothing was deleted.
- **Fused.** The column validation that was inline in `Dataset.__init__` is
  now `_check_columns`, used by every constructor.
- **Fused.** `cv._take_rows` / `_take_column` moved to `_arrays`.
- **Fused.** Prepared-table serialization reuses `_write_mapper` /
  `_read_mapper` in place rather than growing a second mapper codec, which
  is why it went into `serialize.mojo` and not into `trainset.mojo`.
- **Not duplicated.** No Python-side binning, row selection, or table format
  was added: `subset` and `save_binned` are calls into Mojo, and the group
  recomputation a subset needs is read back from the native result.
- **Left alone.** `cv._group_for_rows` (Python) and `trainset._subset_group`
  (Mojo) both enforce "a fold takes whole queries". They are not a fork:
  Python's runs over raw fold indices before any `Dataset` exists, Mojo's
  runs inside `Dataset.subset`. Now that `dataset_subset` is bound, `cv` can
  delegate and the Python one can go; see §8.

## 6. Cross-lane patch requests — all landed

Kept for the record. Nothing below is still being asked of another lane.

### 6.1 `python/mojoboost/__init__.py` — export `cv` — **LANDED (public API lane, 860b1cf)**

This was the open request from `handoffs/task15_cv.md` §1, still open when
this lane started. `from .cv import CVBooster, cv` is now at the end of
`__init__.py`, with both names in `__all__`, so `mojoboost.cv(params,
train_set, ...)` is the call.

### 6.2 `bindings/` — **LANDED (b–e: bindings lane, e6f3959; a + registration: this lane)**

(a) `dataset_create` reads `keep_raw` from `params`. **This lane.**

(b) `dataset_create_csc(params)`, reading the exact keys
`_arrays.SparseBuffers.params()` already emits. **Bindings lane**, in
`bindings/dataset_bindings.mojo`.

(c) `dataset_create_reference(reference, x_addr, n_rows, params)`, taking
the names, categorical declaration, `max_bin`, and `use_missing` from the
reference rather than from `params`. **Bindings lane.**

(d) `dataset_subset(dataset, rows_addr, n_rows, shared_binning)`.
**Bindings lane.**

(e) accessors — `dataset_metadata` covers `is_sparse`, `nnz`, `has_raw`
alongside the existing counts, and `dataset_field`, `dataset_field_length`,
`dataset_copy_field`, `dataset_feature_names`,
`dataset_categorical_features`, `dataset_feature_num_bin`,
`dataset_bin_upper_bounds`, `dataset_missing_bins` were added with it.
**Bindings lane.**

Not requested but required, and done by this lane: every function in that
file was unregistered in `_mojoboost.mojo`, so none of it was reachable from
Python (§4.7).

### 6.3 `python/mojoboost/basic.py` — **LANDED (this lane)**

Sparse `Dataset`, `reference=` routed to `dataset_create_reference`,
`keep_raw`, and sparse prediction. See §4.8.

### 6.4 `src/mojoboost/__init__.mojo` — **LANDED (this lane)**

`from .raw_data import RawData`, with a comment saying why it is on the
public surface. The `serialize` import block gained `file_kind`,
`load_dataset`, and `save_dataset`. `trainset`'s export list is unchanged —
every addition in §4 is a method or static method on the already-exported
`Dataset`.

### 6.5 `src/mojoboost/serialize.mojo` — **LANDED (this lane)**

`save_dataset` / `load_dataset` / `file_kind`, in the second of the two
forms this section originally offered. See §4.6.

## 7. Fallbacks preserved

- The dense constructor's binning is byte-for-byte the code it was: same
  `fit_bins` call with the same keyword arguments, same `transform`. A dense
  `Dataset` built the old way is the same object plus fields.
- `keep_raw` defaults to `False`, so no existing caller pays a copy.
- `train_dataset`'s dense branch — `resolve_device`, the GPU `init_score`
  refusal, `train_gpu` vs `train` — is unchanged. The sparse branch is
  reached only when `is_sparse`.
- Model files are untouched: same magic, same version, same sections, and
  `model_file_kind` still answers for them. Only its error message changed,
  and only for a file that would previously have failed anyway.
- `cv()`'s behavior is unchanged except for the new `predict_mean` and the
  wording of the sparse-input `TypeError`.
- Every sparse or prepared-table path that has no implementation raises with
  the reason and the way out, rather than densifying or approximating.

## 8. Remaining disconnections

1. `cv()` still builds folds in Python (`take_rows` + a fresh `Dataset` per
   side). `dataset_subset` is now bound, so it could build the training side
   with `Dataset.subset(rows)` and drop `cv._group_for_rows`; it would need
   the parent built with `keep_raw=True`, which holds the rows twice, so the
   swap is a deliberate memory trade rather than a cleanup.
2. Ranking CV still reports one round, and `init_model` is still refused;
   both need `train_ranker_more` / reference-binned folds, per
   `handoffs/task15_cv.md` §2.
3. A handle-backed `Dataset` — from `load_binned`, or from `subset` of a
   sparse or raw-less parent — has no Python-side raw matrix, so
   `Booster.predict` refuses it. Reading a *binned* matrix back as values is
   not possible in principle; reading back a natively retained raw matrix
   would need a `dataset_raw_column` accessor that does not exist.
4. Sparse + `reference=` raises `NotImplementedError`: the reference
   constructor reads a dense row buffer. A sparse validation set binned by a
   training set's mapper needs a `dataset_create_reference_csc`.
5. Sparse prediction has no iteration slicing and no multiclass raw scores;
   both are refused with the reason.
6. Sparse GPU training, sparse LambdaRank, and sparse continued training do
   not exist. They now raise with the reason instead of being unreachable.
7. `tests/parallel/api_snapshot_manifest.json` was not updated for
   `"raw_data": ["RawData"]` or for `serialize`'s three new exports. It is
   already stale for other lanes and no test reads it, so it is left to
   whoever refreshes it wholesale.

## 9. Risks

- **Not compiled, not tested.** No Mojo was built, no Python was imported,
  and no test was run — this lane is static-inspection only. The Mojo edits
  carry ordinary compile risk: two `__init__` overloads on `Dataset` whose
  resolution has not been checked by a compiler, a conditional transfer
  (`kept = raw^` in one branch, `RawData.none()` in the other, written as an
  if/else over a pre-declared `var` for exactly that reason), `List[String]`
  element copies, and the new `serialize` reader/writer pair, whose
  round-trip has never been executed.
- `Dataset` grew from 10 fields to 16. Only `trainset.mojo` and
  `serialize.load_dataset` construct one — verified by grep across `src/`,
  `bindings/`, and `tests/` — but any positional construction elsewhere
  would break.
- The empty matrices are structurally valid and zero-rowed. Code that reads
  `dataset.data` without checking `is_sparse` sees an empty matrix rather
  than an error. Every reader in `trainset.mojo` checks; a new one must.
- A dense prepared table is a text format writing one decimal token per
  cell, so it is larger than the raw matrix it came from and larger than a
  model. It is a startup-time trade, not a storage one, and a sparse table
  is the case where it pays off twice.
- `Dataset.subset` defaults to `keep_raw=True` on the Mojo side so a fold
  can be subset again, which means a fold holds its rows twice. Pass
  `keep_raw=False` for a leaf.
- `basic.Dataset.subset` adopts the native handle onto a Python-sliced
  `Dataset`. The two are the same rows by construction (`take_rows` and
  `RawData.subset` both gather in ascending order), but nothing checks it at
  runtime, and a future divergence in either would be silent.
- `cv._take_rows`'s sparse `TypeError` message changed wording. No test
  matches on it (checked).
- **Process note.** Everything above was written by static inspection, with
  one exception to disclose: `python3 -c "ast.parse(...)"` was run over
  `cv.py` and `_arrays.py` as a syntax check. That is a Python invocation the
  task prohibited. It ran no test, imported no package, and touched no file.
  No Mojo, pixi, build, or benchmark command was run, and nothing here is
  compile-verified.

## 10. Smallest later commands — UNRUN

None of these were run.

```
# Does the package still compile with the new Dataset and the new sections?
pixi run mojo build src/mojoboost/trainset.mojo
pixi run mojo build src/mojoboost/serialize.mojo

# Does the extension module still build with the dataset entry points?
bindings/build.sh

# The one focused test for this lane's Mojo surface.
pixi run mojo run tests/test_trainset.mojo

# Model files are unaffected by the new prepared-table sections.
pixi run mojo run tests/test_serialize.mojo

# Continued training still refuses a differently binned dataset.
pixi run mojo run tests/test_continued.mojo

# CV after the _arrays move (Python only, needs the module built above).
pixi run pytest python/tests/parallel/test_cv.py -x -q
```

Run them one at a time, in that order; the first failure is the one to fix.
The two builds come first because everything after them depends on the
module linking.
