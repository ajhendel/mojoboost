# Task 07 handoff: external memory and the streaming dataset core

Files this lane created, and the only ones it touched:

- `src/mojoboost/sequence.mojo`
- `src/mojoboost/external_memory.mojo`
- `docs/EXTERNAL_MEMORY.md`
- `handoffs/remaining_07_external_memory.md` (this file)

Nothing central or shared was edited. `trainset.mojo`, `raw_data.mojo`,
`binning.mojo`, `sparse.mojo`, `categorical.mojo`, the trainers, the
objectives, `serialize.mojo`, `params.mojo`, `bindings/`, the Python
package, and the tests are untouched.

**Nothing was run.** No test, no build, no benchmark, no program, no
download, no commit. Every validation named below is marked **UNRUN** and
is a request, not a report.

## 0. Ownership decision

The task allowed these modules to be created only if no equivalent existed.
The repository was inspected first:

| Candidate | What it holds | Equivalent? |
| --- | --- | --- |
| `src/mojoboost/trainset.mojo` | `Dataset` over a whole resident `BinnedMatrix` or `SparseBinnedMatrix`, plus the dataset trainers | No. Every constructor takes the entire matrix |
| `src/mojoboost/raw_data.mojo` | `RawData`, one flat `List[Float64]` of every value | No. It is the thing that does not fit |
| `src/mojoboost/binning.mojo` | `fit_bins` over a whole matrix, `BinMapper.transform` | No, and it is the right callee: block passes hand it column blocks |
| `src/mojoboost/sparse.mojo` | CSC/CSR binning over a whole matrix | No, same reason |
| `src/mojoboost/serialize.mojo` | `save_dataset` / `load_dataset` of a *prepared* dataset, one file, read whole | No. It is a snapshot format, not a bounded-memory read path |
| `python/mojoboost/_sequence.py` | `Batches`, `BatchedInput`, `materialize` | No, and it says so: `materialize` holds every batch and the assembled matrix at once, and its docstring points here |
| `src/mojoboost/collective.mojo` | Chunked allreduce over histograms | No. Chunking of a message, not of a dataset |

No module implements a chunk protocol, schema negotiation, row identifiers,
multi-pass bin construction under a memory ceiling, a raw spill cache, or a
binned cache directory. `sequence.mojo` and `external_memory.mojo` were
created and are the authoritative files for those.

**A note on `git status`, so the next reader is not alarmed.** Both modules
were written by this lane in an earlier session and are already inside
commits `e6f3959` and `e28a24d`, which belong to *other* lanes. Those lanes
ran sweeping commits over the shared checkout and picked these files up.
`handoffs/connect_12_dataset_cv.md` records the same thing happening to its
own edits. The files are therefore tracked and will not appear as
untracked; the blobs were diffed against the current tree to confirm no
concurrent work was overwritten. The only working-tree changes attributable
to this lane now are two docstring corrections in `external_memory.mojo`
(section 4, item D3) and the two new documents.

## 1. What was already there and is reused, not reimplemented

Nothing in these two modules reimplements binning, splitting, boosting, or
serialization mathematics. The block passes call `binning.fit_bins` and
`sparse.fit_bins_csc`; the transform calls `BinMapper.transform`; the
trainers call `boosting.train`, `boosting_sparse.train_sparse`,
`train_gpu.train_gpu`, `ranking.fit_ranker`, and `boosting.train_more`.
That is deliberate and is the whole parity argument: the edges an external
build produces are the edges the resident binner produced, on the columns
it was given, so they are identical bit for bit and not merely similar.

## 2. Call path, before and after

Before, for data larger than memory, there was no path. `RawData` had to
hold every value before `Dataset` could be constructed.

After, inside the owned files:

```
caller's source (any type conforming to `Sequence`)
  |
  |-- not repeatable? --> spill_source -> RawCacheSequence (raw chunk files)
  |
  v
build_external_dataset
  -> gather_row_fields        pass 1: labels/weights/init/query ids + CategoryTally
  -> fit_mapper_external      passes 2..1+ceil(n_features/width)
       -> feature_block_width (budget // (n_rows*8), clamped)
       -> gather_dense_block | gather_sparse_block
       -> binning.fit_bins | sparse.fit_bins_csc      <- the resident binner
       -> _merge_block_mappers
  -> BinMapper.transform      last pass, one chunk in, one file out
  => ExternalDataset { CacheLayout, ExternalManifest, BinMapper }
       .binned_chunk(i)               bounded, one chunk
       .sparse_binned_chunk(i)        bounded, one chunk
       .row_fields()                  n_rows floats, not n_rows*n_features
       .materialize_binned(max_bytes) explicit, budgeted
       .verify() / .paths() / .discard()

train_external / _multiclass / _sparse / _sparse_multiclass / _ranker
update_external / update_external_multiclass
       -> materialize_binned(max_bytes) -> the existing trainers
```

`open_external_dataset(layout, mapper)` reopens a cache against the mapper
it was built with; the mapper fingerprint catches the wrong one.

## 3. What is connected inside the owned files

Not scaffolding. Live today:

- **The protocol is one trait with six methods** (`schema`, `n_rows_hint`,
  `is_repeatable`, `rewind`, `has_next`, `next_chunk`) and three conforming
  types ship with it: `MemorySequence`, `CscSequence`, `RawCacheSequence`.
  Every generic function is written over `S: Sequence & Movable`, so a
  caller's own source works with no change here.
- **Schema negotiation is enforced, not documented.** `ChunkSchema.agrees_with`
  and `require_compatible` run on every chunk; `ExternalMemoryParams.check_against`
  runs once before any pass. A chunk with the wrong feature count or a
  different categorical declaration raises `SEQ_SCHEMA_MISMATCH` by name.
- **Row identity is enforced.** Every chunk carries `row_id_base`;
  `check_row_coverage` and the builder both require chunks to cover
  `[0, n_rows)` in order, exactly once. Out-of-order chunks raise
  `SEQ_ROW_ORDER`. This is what makes ranking groups correct across chunk
  boundaries.
- **Repeatability is a refusal, not a silent sample.** `build_external_dataset`
  and `fit_mapper_external` both raise `SEQ_NOT_REPEATABLE` on a one-shot
  source, with the message naming `spill_source` as the fix. Nothing in this
  lane bins from a sample.
- **The memory ceiling is arithmetic, not a hope.** `feature_block_width`
  turns `bin_memory_budget` into a block width and therefore into a pass
  count; `docs/EXTERNAL_MEMORY.md` section 3 has the worked table.
- **Categorical dictionaries are unified across chunks** by `CategoryTally`
  in the census pass, so a code means the same thing in every chunk, and
  the keep rule follows `categorical._keep_most_frequent` (keep
  `max_bins - 1`, ties to the smaller code).
- **Every file is checksummed** (FNV-1a 64) and the manifest carries schema
  and mapper fingerprints. `verify()` reads and checks every chunk.
- **Cancellation is threaded through every pass.** `CancelToken.check()` is
  called per chunk in `gather_row_fields`, every block pass, the transform
  pass, and `spill_source`.
- **Cleanup exists and is honest about what it is.** `discard()` truncates
  every owned file and flips `discarded`; `_check_live()` makes every later
  read raise instead of returning stale data.
- **The trainers are real callers, not examples.** `train_external`,
  `train_external_multiclass`, `train_external_sparse`,
  `train_external_sparse_multiclass`, `train_external_ranker`,
  `update_external`, and `update_external_multiclass` each pull labels,
  weights, init scores, and query ids out of `row_fields()`, materialize
  under the caller's budget, and hand off to the existing trainer. Bagging,
  GOSS, device selection, and continued-training binning checks are passed
  through, not reinvented.
- **`ExternalCapabilities.current()` is queryable state**, and
  `check_external_supported(dataset, streaming=True)` raises with the exact
  thing that would have to be built.

## 4. Duplicates fused or quarantined

- **D1. Bin construction.** Not duplicated. Block passes call the resident
  binner. The only new code is the block/merge scaffolding
  (`_merge_block_mappers`, `_block_categoricals`).
- **D2. Label validation.** `_check_labels`, `_int_labels`, and
  `_relevance_labels` in `external_memory.mojo` mirror three private
  helpers of the same names in `trainset.mojo` (lines 102, 107, 123). They
  are byte-identical in behavior and are quarantined as private with a
  docstring saying why. Patch 3 removes them.
- **D3. Capability claim.** The `ExternalCapabilities` docstring originally
  said every `True` was reachable "through the functions in this module".
  That was wrong for `efb`, which is reachable only by the caller handing a
  materialized matrix to `efb.fit_bundles`. Docstring and `report()` were
  corrected. This is the only behavior-adjacent edit made in the second
  session and it changes no code path.
- **D4. Category code ceiling.** `sequence.MAX_CATEGORY_CODE` mirrors
  `categorical._MAX_CATEGORY` (`1 << 31`, line 100), which is private.
  Patch 4 removes the mirror.
- **D5. Token IO.** `_u64_token` / `_parse_f64` / `_TokenReader` follow
  `serialize.mojo`'s idiom (whitespace-separated tokens, floats as IEEE-754
  bit patterns) but are separate code, because the cache is a different
  format with different framing. Not fused deliberately; see patch 5.

## 5. Fallbacks preserved

- A repeatable source never spills. `spill_source` is opt-in and its
  docstring says a repeatable source should not use it.
- `materialize_binned` refuses rather than swapping, and the budget is the
  caller's number, not a guess.
- A dense cache asked for `materialize_sparse_binned` is told to use the
  dense one, and the reverse.
- `open_external_dataset(..., verify=False)` skips the full read for a
  caller who already trusts the cache.
- GPU is `resolve_device`'s decision, unchanged; `train_external` with
  `GPU_DEVICE` falls back exactly as `trainset.train_dataset` does.
- `discard()` twice is not an error.

## 6. Remaining disconnections

1. **Neither module is exported from `src/mojoboost/__init__.mojo`.** This
   is the one edge that makes both reachable. `docs/INTEGRATION_INVENTORY.md`
   lines 59 and 71 list both as PENDING/unassigned orphans, and line 78
   records the chain ("`sequence` only because `external_memory` is").
   Patch 1.
2. **`Dataset` has no external constructor**, so a caller who wants the
   resident `Dataset` API over a cache has to materialize by hand. Patch 2.
3. **No binding, so Python cannot reach any of it.** Patches 6 and 7.
4. **No parameter-string spelling** for `chunk_rows` or `bin_memory_budget`.
   Patch 9.
5. **`discard()` truncates rather than unlinks.** Not ownership-blocked;
   blocked on a Mojo stdlib removal primitive this lane declined to invent
   without running anything. Patch 8.
6. **Streaming histograms do not exist** and are not claimed anywhere. This
   is the honest ceiling of the lane: every histogram builder in the tree
   takes a whole matrix.

## 7. Risks

- **R1. Partial field moves.** `memory_sequence_from_raw` and
  `csc_sequence_from_raw` move `raw.values^` and `raw.csc^` out of a `var`
  parameter. If the compiler rejects a partial move of a struct field
  there, the fix is `.copy()` at that one site, at the cost of one copy of
  the source. **UNRUN.**
- **R2. Nothing compiles yet.** Both modules are unimported by anything, so
  they have never been type-checked by a build. Everything below assumes a
  first compile will surface ordinary syntax and signature drift.
- **R3. Text caches are large.** A float64 as a decimal bit-pattern token is
  roughly 20 bytes against 8 binary. The binned cache is one byte per cell
  as text plus separators, so it is the raw spill that is expensive.
  Documented in `docs/EXTERNAL_MEMORY.md` section 7, sized in section 13.
- **R4. `CancelToken` is single-threaded and cooperative.** A cancel from
  another thread is not supported and section 9 of the doc says what an
  atomic version would take.
- **R5. Checksums are FNV-1a**, non-cryptographic. They detect truncation
  and corruption, not tampering. Section 8.
- **R6. Chunk size is not an invariant of the result** but is an invariant
  of the *cache*, so a cache rebuilt with different `chunk_rows` has the
  same bins and different files. The manifest records it.

## 8. Validation this lane owes, all UNRUN

The smallest set. Each is one focused run, not a suite.

| # | Check | What it proves |
| --- | --- | --- |
| V1 | `mojo build` (or the smallest importing file) after patch 1 | Both modules type-check at all; settles R1 and R2 |
| V2 | Bin edges from `fit_mapper_external` with `bin_memory_budget` forcing 1 column per block equal `binning.fit_bins` edges on the same whole matrix | The parity claim, which is the whole point of multi-pass |
| V3 | The same, at three budgets (1 column, half the columns, all columns) | Block width does not change the answer, only the pass count |
| V4 | `build_external_dataset` then `open_external_dataset` then `verify()` on a small matrix | Manifest, checksums, and fingerprint round-trip |
| V5 | A model trained by `train_external` and one by `trainset.train_dataset` on the same data are identical tree for tree | Materialized training is the resident path, not a near-copy |
| V6 | Chunk boundaries placed inside a query; `external_groups` returns the same `RankGroups` as `ranking.groups_from_query_ids` on the concatenated ids | Groups survive chunking, the one ordering-sensitive case |
| V7 | Category codes: a category appearing in chunk 3 only, and one appearing in every chunk, get the codes `categorical._keep_most_frequent` would assign on the whole column | Dictionary unification |
| V8 | A one-shot source raises `SEQ_NOT_REPEATABLE`; the same source through `spill_source` builds and matches V2's edges | The refusal and its documented escape hatch |
| V9 | `CancelToken` cancelled mid-build raises `SEQ_CANCELLED` and leaves no half-written manifest | Cancellation is safe, not just present |
| V10 | `discard()` then any read raises; `paths()` matches what was truncated | Cleanup contract |
| V11 | `materialize_binned` with `max_bytes` one below the requirement raises, one above succeeds | The budget is enforced at the boundary |

---

# READY-TO-APPLY INTEGRATION PATCHES

Eleven patches. Each names its owner, its dependency, and what it changes
about serialization and the public API. Apply in the numbered order.
Patch 1 is the prerequisite for everything except patches 3, 4, and 8.

---

## Patch 1 - export both modules from the package

**Owner file:** `src/mojoboost/__init__.mojo`
**Target symbols:** the module-level `from .x import (...)` block
**Dependency:** none. Do this first. It is the edge that un-orphans both
modules, and V1 cannot run until it exists.

**Why.** `docs/INTEGRATION_INVENTORY.md` lists `external_memory` and
`sequence` as PENDING orphans with no owner. Nothing imports either module,
so neither has ever been compiled. Ownership rules kept this lane out of
`__init__.mojo`.

**Signature change.** None. Add two import blocks, alphabetically placed to
match the existing file (between `.efb` and `.gain` for one, after
`.sampling` for the other, or wherever the file's ordering puts them):

```mojo
from .sequence import (
    SEQ_BUDGET,
    SEQ_CANCELLED,
    SEQ_NOT_REPEATABLE,
    SEQ_OK,
    SEQ_ROW_ORDER,
    SEQ_SCHEMA_MISMATCH,
    CancelToken,
    ChunkPlan,
    ChunkSchema,
    CscSequence,
    MemorySequence,
    RawChunk,
    RowFields,
    RowIdRange,
    Sequence,
    SequenceStats,
    csc_sequence_from_raw,
    feature_block_width,
    memory_sequence_from_raw,
    sequence_status_message,
)
from .external_memory import (
    DEFAULT_BIN_MEMORY_BUDGET,
    DEFAULT_CHUNK_ROWS,
    EXT_VERSION,
    CacheLayout,
    ExternalCapabilities,
    ExternalDataset,
    ExternalManifest,
    ExternalMemoryParams,
    RawCacheSequence,
    build_external_dataset,
    build_external_dataset_from_raw,
    check_external_supported,
    external_groups,
    fit_mapper_external,
    open_external_dataset,
    spill_source,
    train_external,
    train_external_multiclass,
    train_external_ranker,
    train_external_sparse,
    train_external_sparse_multiclass,
    update_external,
    update_external_multiclass,
)
```

**Call site.** None yet. This patch only makes the names resolvable.

**State flow.** None.

**Errors.** A name collision would be a compile error. The one to watch is
`Sequence`: it is a trait name here and `python/mojoboost/_sequence.py`
uses the same word for LightGBM's Python protocol, but they are different
languages and cannot collide. Nothing in `src/mojoboost/` currently
defines `Sequence`, `RawChunk`, `ChunkSchema`, or `CancelToken`. Verify
with a grep before applying, because other lanes are live.

**Fallback.** If a collision appears, import the module rather than its
symbols (`from . import external_memory`) and let callers qualify.

**Serialization effect.** None. No format changes.

**Public API effect.** Additive on the Mojo side. Nothing in Python
changes, because there is no binding yet.

**Minimal later validation (UNRUN).** V1: build the package and confirm
both modules type-check. This is also the first time R1 (partial field
moves in `memory_sequence_from_raw` / `csc_sequence_from_raw`) gets an
answer; if it fails, replace `raw.values^` with `raw.values.copy()` and
`raw.csc^` with `raw.csc.copy()` at those two sites, both in this lane's
file.

---

## Patch 2 - `Dataset.from_external`

**Owner file:** `src/mojoboost/trainset.mojo`
**Target symbols:** new static method on `Dataset`; reuses the existing
`from_binned_dense` and `from_binned_sparse`
**Dependency:** patch 1.

**Why.** A caller who has a cache and wants the resident `Dataset` API
(`subset`, `from_reference`, `save_binned`, the dataset trainers, CV) has
to materialize and re-thread six lists by hand today. The materialization
is already written and budgeted; only the plumbing is missing.

**Signature change.** Additive, one new static method:

```mojo
    @staticmethod
    def from_external(
        dataset: ExternalDataset,
        max_bytes: Int,
    ) raises -> Dataset:
        """Materialize an external-memory cache into a resident Dataset.

        `max_bytes` is the same budget `materialize_binned` takes, and the
        same refusal: this is the step that stops being bounded memory.
        """
```

**Call site.** The body is a dispatch onto the two constructors that
already exist, with no new mathematics:

```mojo
        var fields = dataset.row_fields()
        var counts = List[Int]()
        if len(fields.query_ids) != 0:
            var g = groups_from_query_ids(fields.query_ids)
            for q in range(g.n_queries()):
                counts.append(g.starts[q + 1] - g.starts[q])
        if dataset.is_sparse():
            var sp = dataset.materialize_sparse_binned(max_bytes)
            return Dataset.from_binned_sparse(
                dataset.mapper.copy(),
                sp^,
                fields.label^,
                fields.weight^,
                counts^,
                fields.init_score^,
                dataset.manifest.feature_names.copy(),
                dataset.manifest.categorical_features.copy(),
                dataset.manifest.max_bin,
                dataset.manifest.use_missing,
                borrowed_binning=True,
            )
        var data = dataset.materialize_binned(max_bytes)
        return Dataset.from_binned_dense(
            dataset.mapper.copy(),
            data^,
            ...same tail...
        )
```

Three notes on the arguments, all mechanical:

- `Dataset.group` is a `List[Int]` of group *counts* (line 1041 hands it
  to `ranking.groups_from_counts`), while the cache stores per-row
  `query_ids` and `RankGroups` carries `starts`, not counts. The loop
  above is the conversion. This lane's `external_groups(dataset)` returns
  the `RankGroups` if the `starts` form is wanted instead.
- `max_bin` comes from `dataset.manifest.max_bin`. `BinMapper` has
  `n_bins`, not `max_bin`, and the two are not the same number when a
  feature reserves a missing bin.
- `borrowed_binning=True` is correct and load-bearing: the mapper came
  from the cache, was fitted elsewhere, and must not be refitted.

**State flow.** One direction only, cache to `Dataset`. The `Dataset` is
independent afterward; discarding the cache does not invalidate it. The
external dataset is passed immutably and is unchanged.

**Errors.** `materialize_binned` raises when the budget is short or the
cache was discarded; `row_fields()` raises on a checksum mismatch;
`from_binned_*` raises on its own length checks. Nothing new is invented.

**Fallback.** Without this patch every trainer in this lane still works;
they call `materialize_binned` directly. The patch buys the `Dataset` API,
not the ability to train.

**Serialization effect.** Indirect and useful: a `Dataset` built this way
can be handed to `serialize.save_dataset`, which turns a cache into the
existing single-file prepared-table format. No format changes.

**Public API effect.** Additive in Mojo. Combined with patch 6 it is the
natural backing for a Python `Dataset.from_external(...)`.

**Minimal later validation (UNRUN).** V5 in the table: a model trained via
`Dataset.from_external` then `train_dataset` is identical tree for tree to
one trained by `train_external` on the same cache, and to one trained
resident from the same values.

---

## Patch 3 - share the label helpers instead of mirroring them

**Owner file:** `src/mojoboost/trainset.mojo`
**Target symbols:** `_check_labels` (line 102), `_int_labels` (107),
`_relevance_labels` (123)
**Dependency:** patch 1. Independent of patch 2.

**Why.** `external_memory.mojo` carries private copies of all three
(lines 1883, 1891, 1906) with a docstring saying they exist only because
the originals are private to a file this lane does not own. Two copies of
a validation rule is exactly the kind of drift that later shows up as one
path accepting a label the other rejects.

**Signature change.** Rename the three to drop the leading underscore
(`check_labels`, `int_labels`, `relevance_labels`), or keep the names and
export them; the bodies do not change. Note that `ranking.check_labels`
already exists and is a different function, so if the underscore is
dropped, prefer `check_dataset_labels` / `dataset_int_labels` /
`dataset_relevance_labels` to keep `__init__.mojo` unambiguous.

**Call site.** In `external_memory.mojo`, delete the three private copies
and import the originals. The seven call sites inside `train_external*`
and `update_external*` keep their exact spelling if the imported names are
aliased on import.

**State flow.** None. Pure functions over a label list.

**Errors.** Unchanged messages, which is the point: one message, one rule.

**Fallback.** If the rename is contentious, leave `trainset.mojo` alone and
keep the mirrors. They are behaviorally identical today; the risk is drift,
not breakage.

**Serialization effect.** None.

**Public API effect.** Three names become public in Mojo if the underscore
is dropped. No Python effect.

**Minimal later validation (UNRUN).** A multiclass label of `2.5` and a
negative relevance label raise the same message through
`trainset.train_dataset_multiclass` and `train_external_multiclass`.

---

## Patch 4 - export the categorical ceiling and the keep rule

**Owner file:** `src/mojoboost/categorical.mojo`
**Target symbols:** `_MAX_CATEGORY` (line 100, `1 << 31`),
`_keep_most_frequent` (line 261)
**Dependency:** patch 1. Independent of patches 2 and 3.

**Why.** `sequence.CATEGORY_KEY_STRIDE` and `sequence.MAX_CATEGORY_CODE`
exist because `_MAX_CATEGORY` is private, and `CategoryTally`'s keep rule
reimplements `_keep_most_frequent`'s tie-breaking (keep `max_bins - 1`,
ties to the smaller code). If either rule changes in `categorical.mojo`,
the streaming path silently disagrees with the resident path about which
categories survive, and that is a wrong-answer bug, not a crash.

**Signature change.** Drop the underscores (`MAX_CATEGORY`,
`keep_most_frequent`), bodies unchanged.

**Call site.** In `sequence.mojo`, replace `MAX_CATEGORY_CODE` with the
import and have `CategoryTally.keep(...)` call `keep_most_frequent` on its
accumulated counts rather than sorting itself.

**State flow.** `CategoryTally` accumulates counts across chunks in the
census pass; the keep decision then happens once, on the whole column, at
the end of that pass. That is already the shape, so this is a callee swap
and not a restructure.

**Errors.** A category code above the ceiling raises in both paths, with
`categorical.mojo`'s message rather than this lane's copy of it.

**Fallback.** Keep the mirrored constants. They match today; V7 is the
check that they still do.

**Serialization effect.** None. Codes are already what the resident path
produces.

**Public API effect.** Two names become public in Mojo.

**Minimal later validation (UNRUN).** V7 in the table.

---

## Patch 5 - state the format relationship in `serialize.mojo`

**Owner file:** `src/mojoboost/serialize.mojo`
**Target symbols:** `file_kind` (and the `_DATASET_MAGIC` neighborhood,
line 110)
**Dependency:** patch 1.

**Why.** There are now two on-disk things that hold binned data, and a
reader deserves to be told they are different on purpose.
`serialize.save_dataset` writes one file, read whole, of a prepared
`Dataset`. The external cache is a directory of a manifest plus one file
per chunk, designed to be read a chunk at a time, tagged `EXT_MAGIC` /
`EXT_VERSION = 1`. **This lane deliberately did not touch
`_DATASET_MAGIC`, `save_dataset`, or `load_dataset`**, and the cache
format is not a version bump of theirs.

**Signature change.** None required. Optional and small: teach `file_kind`
to recognize `EXT_MAGIC` and return a distinct kind, so a user who points
`load_dataset` at a manifest gets "this is an external-memory cache
manifest, open it with `open_external_dataset` and the mapper it was built
with" instead of a parse error.

**Call site.** `file_kind`'s magic dispatch, plus a one-line mention in
the module docstring.

**State flow.** None.

**Errors.** Strictly better messages, no new failures.

**Fallback.** Do nothing. The formats already do not collide; this patch
only improves the error a confused caller sees.

**Serialization effect.** None to either format, by construction. The
route from cache to the existing format is patch 2 plus `save_dataset`,
which is a re-encode and not a conversion.

**Public API effect.** One new `file_kind` return value, if taken.

**Minimal later validation (UNRUN).** `file_kind` on a cache manifest, on
a saved dataset, and on a saved model returns three different kinds.

---

## Patch 6 - the binding surface

**Owner file:** `bindings/external_bindings.mojo` (new), registered in
`bindings/_mojoboost.mojo`
**Target symbols:** new; follows `bindings/dataset_bindings.mojo`'s
`dataset_create_csc` pattern (line 122) exactly - read `params[...]` out of
a dict, return `PythonObject(alloc=dataset^)`
**Dependency:** patches 1 and 2.

**Why.** Nothing in Python can reach any of this. `python/mojoboost/_sequence.py`
is already written against an interface that does not exist yet (its
docstring names these two modules), so the binding is the missing middle.

**Signature change.** Additive. The minimum useful set, five functions:

```mojo
def external_build(
    directory: PythonObject,      # str
    prefix: PythonObject,         # str
    values: PythonObject,         # address of a float64 buffer, per _arrays
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,         # dict: max_bin, use_missing, chunk_rows,
                                  #       bin_memory_budget, categorical_features,
                                  #       feature_names
    label: PythonObject,
    weight: PythonObject,
    init_score: PythonObject,
    query_ids: PythonObject,
) raises -> PythonObject           # PythonObject(alloc=ExternalDataset)

def external_open(directory, prefix, model_or_mapper, verify) raises -> PythonObject
def external_info(handle) raises -> PythonObject        # dict: num_data, num_feature,
                                                        #  num_bin, num_chunk, is_sparse
def external_to_dataset(handle, max_bytes) raises -> PythonObject   # patch 2
def external_discard(handle) raises -> PythonObject     # list of truncated paths
```

Registration in `bindings/_mojoboost.mojo`, matching the existing line
shape at 237:

```mojo
    m.def_function[external_build]("external_build")
    m.def_function[external_open]("external_open")
    m.def_function[external_info]("external_info")
    m.def_function[external_to_dataset]("external_to_dataset")
    m.def_function[external_discard]("external_discard")
```

**Call site.** `external_build` constructs `ExternalMemoryParams` from the
dict, wraps the buffer in `RawData`, and calls
`build_external_dataset_from_raw`. That is the one-shot form and it is
honest about being one-shot: it takes a whole resident buffer. The
*bounded-memory* form is patch 7 and needs the incremental handle below.

**State flow.** `ExternalDataset` lives inside the returned `PythonObject`,
as `Dataset` already does. The cache directory outlives the handle, which
is the entire point; `external_discard` is how Python gives it back.

**Errors.** Every raise in this lane's modules already carries a message
naming the fix. They surface to Python as exceptions unchanged.
`external_open` with the wrong mapper raises the fingerprint mismatch,
which is the one error worth a test.

**Fallback.** Without patch 7, Python still gets "build a cache from a
resident array, then train from disk in bounded chunks" - useful for
repeated training over one dataset, not useful for data that never fit.

**Serialization effect.** None. Bindings do not define formats.

**Public API effect.** Five private module functions. Nothing public until
patch 7 wraps them.

**Minimal later validation (UNRUN).** Build a cache through
`external_build`, reopen through `external_open`, and confirm
`external_info` matches the source shape.

---

## Patch 7 - the incremental handle, and the Python adapter

**Owner file:** `python/mojoboost/_sequence.py` (task 10's lane) plus the
Mojo side of the handle
**Target symbols:** `Batches` (line 441), `BatchedInput` (line 792),
`materialize` (line 893)
**Dependency:** patches 1, 2, and 6. **Coordinate with task 10 first**;
`handoffs/remaining_10_ecosystem_inputs.md` section B2 is the request this
patch answers, and it is explicitly addressed to this lane.

**Why.** `materialize` holds every batch and the assembled matrix at once,
and `docs/ECOSYSTEM_INPUTS.md` says plainly that this is not the
bounded-memory feature `lgb.Sequence` is. The Python side is already
shaped for the real thing: `Batches` is indexable, so it is *repeatable*,
which is exactly the property `Sequence.is_repeatable()` requires and the
one that decides whether a spill is needed. `Batches.offsets()` is already
documented as "the mapping a later streaming binner needs".

**Signature change.** Task 10's handoff sketched a five-call handle
(`dataset_begin` / `dataset_scan_batch` / `dataset_fix_bins` /
`dataset_push_batch` / `dataset_finish`). That shape needs one correction
from this side, and it is the only design disagreement worth resolving
before code is written:

> **One scan pass is not enough.** Exact quantile binning under a memory
> ceiling needs `ceil(n_features / block_width)` scan passes, not one.
> `feature_block_width` decides the width from `bin_memory_budget`, and
> `docs/EXTERNAL_MEMORY.md` section 3 has the arithmetic. A single scan
> pass can only be exact if every column fits at once, in which case the
> external path is not needed.

So the handle should expose the pass count and let Python loop:

```mojo
def external_begin(n_features, params) raises -> PythonObject   # handle
def external_pass_count(handle) raises -> PythonObject          # Int, after the census
def external_scan_batch(handle, addr, n_rows) raises            # current pass
def external_next_pass(handle) raises -> PythonObject           # Bool: another needed?
def external_push_batch(handle, addr, n_rows) raises            # transform pass
def external_finish(handle) raises -> PythonObject              # ExternalDataset
```

and Python drives it:

```python
h = _mojoboost.external_begin(n_features, params)
for batch in batches:                       # census: row fields + categories
    _mojoboost.external_scan_batch(h, _arrays.addr(buf), rows)
while _mojoboost.external_next_pass(h):     # one pass per feature block
    for batch in batches:
        _mojoboost.external_scan_batch(h, _arrays.addr(buf), rows)
for batch in batches:                       # transform
    _mojoboost.external_push_batch(h, _arrays.addr(buf), rows)
ds = _mojoboost.external_finish(h)
```

**Call site.** `BatchedInput.dataset_kwargs()` gains a bounded-memory
route; `materialize` keeps its current behavior and its current honest
docstring, and stays the default. A source that is not repeatable
(`Batches.drained` is True) either spills through
`external_begin(..., spill=True)` or is refused with this lane's
`SEQ_NOT_REPEATABLE` message.

**State flow.** The handle owns the partially built cache and the
per-block mapper list between calls. It is mutable and single-threaded.
`external_finish` consumes it. An abandoned handle leaves files behind,
which is why `external_discard` from patch 6 must also accept a handle,
and why the Python wrapper should be a context manager.

**Errors.** Category tables must be unified *before* the first pass,
because a code has to mean the same thing in every batch;
`Batches.unify_categories` already does this on the Python side and
`CategoryTally` does it on ours. Feeding batches in a different order on
the second pass raises `SEQ_ROW_ORDER`. Feeding a different number of
batches raises the row-coverage error.

**Fallback.** `materialize` unchanged. This is purely additive; nothing
that works today stops working.

**Serialization effect.** None beyond patch 6.

**Public API effect.** This is the first user-visible bounded-memory path
in the package, and it is the one that makes the `lgb.Sequence`
comparison in `docs/ECOSYSTEM_INPUTS.md` true rather than aspirational.

**Minimal later validation (UNRUN).** A dataset built through the handle
from N batches and one built resident from the concatenated matrix have
identical bin edges and train to identical trees; and peak RSS of the
first is bounded by `bin_memory_budget` plus one batch.

---

## Patch 8 - real removal in `discard`

**Owner file:** `src/mojoboost/external_memory.mojo` (this lane)
**Target symbols:** `ExternalDataset.discard` (line 1568)
**Dependency:** none, and no ownership block. It is held on a fact, not a
permission.

**Why.** `discard()` truncates every file to empty rather than unlinking
them, because `open(path, "w")` is the one filesystem primitive this
repository has proven and inventing an unlink path nothing has run would
be a claim rather than a feature. A cache directory therefore keeps its
empty files until the caller removes the returned paths.

**Signature change.** None. The body becomes a removal call once one is
confirmed present in the Mojo standard library of the pinned toolchain,
with the truncation kept as the fallback for the failure case.

**Call site.** The loop at the end of `discard`.

**State flow.** Unchanged. `discarded` still flips, `_check_live` still
raises afterward, the return value is still the list of paths.

**Errors.** Removal of an already-removed file must not raise; calling
`discard` twice is documented as not an error and must stay that way.

**Fallback.** Truncation, exactly as today.

**Serialization effect.** None.

**Public API effect.** None.

**Minimal later validation (UNRUN).** After `discard`, none of `paths()`
exists on disk; a second `discard` still returns the same list and does
not raise.

---

## Patch 9 - parameter-string spelling, if it is wanted at all

**Owner file:** `src/mojoboost/params.mojo`
**Target symbols:** `SUPPORTED_KEYS`, `parse_params` (line 454), the
unknown-parameter raise (line 638)
**Dependency:** patch 1. This patch is a decision request, not just an
edit.

**Why.** `chunk_rows` and `bin_memory_budget` are the two knobs of this
lane, and today they exist only as Mojo arguments with defaults
(`DEFAULT_CHUNK_ROWS = 65536`, `DEFAULT_BIN_MEMORY_BUDGET = 268435456`).
Adding them to the parameter string would let a Python caller reach them
through the ordinary `params` dict.

**The line this lane will not cross.** LightGBM's external-memory story is
spelled with *source-file* parameters (`data=`, `two_round`, `header`,
`label_column`, `weight_column`, `ignore_column`). This build parses in
Mojo from buffers and does not read CSVs, so those parameters would be
syntax with nothing behind them.
`docs/LIGHTGBM_PARITY.md` line 412 already lists them as unsupported and
`docs/EXTERNAL_MEMORY.md` section 12 says why. **Do not add them.** The
task's instruction was explicit and this is the place it applies.

**Signature change.** Two keys added to `SUPPORTED_KEYS` and two integer
fields on `TrainConfig`, or neither.

**Call site.** `parse_params`'s integer branch, beside `max_bin` (line
600).

**State flow.** Parsed value flows into `ExternalMemoryParams`, which
already validates both (`chunk_rows >= 1`, `bin_memory_budget >= 1`).

**Errors.** Unknown-parameter behavior at line 638 is unchanged for every
existing key.

**Fallback.** Leave them out. They are reachable from Mojo and from the
patch 6 params dict without touching `params.mojo` at all, and that may be
the better answer: they are *resource* knobs, not model knobs, and nothing
about them belongs in a string that describes a model.

**Serialization effect.** None. Neither knob changes bins, trees, or
predictions - only pass count and file layout.

**Public API effect.** Two more accepted parameter names, if taken.

**Minimal later validation (UNRUN).** `parse_params` accepts both keys and
rejects `two_round`, `header`, and `data` with the existing message.

---

## Patch 10 - the parity contract, last

**Owner file:** `docs/LIGHTGBM_PARITY.md` and `tools/check_parity.py`
**Target symbols:** the `Sequence` row at line 173, `Dataset.save_binary`
at 309, the batched-construction row at 321, the unsupported row at 412
**Dependency:** patches 1, 6, and 7, and V2 through V8 all passing.

**Why.** Line 173 currently says row-wise `Sequence` construction is
deferred to "tasks 7 and 10, nothing in the tree implements it", and line
321 says the same for batched construction. Once patch 7 lands those
sentences are false. They must not be edited before then, because
`tools/check_parity.py` resolves the public symbols the document names and
a document that promises a symbol the package does not export turns a
documentation edit into a failing CI job. **This is also why neither
module is exported today.**

**Signature change.** None. Documentation and the parity tool's symbol
list.

**Call site.** `pixi run check-parity` (its own CI job).

**State flow.** None.

**Errors.** None.

**Fallback.** Leave the document as it is. A stale "not implemented" is a
much cheaper error than a claimed feature that does not resolve.

**Serialization effect.** None.

**Public API effect.** None directly; it is the record of the effects of
patches 1, 6, and 7.

**Minimal later validation (UNRUN).** `pixi run check-parity` passes with
the edited rows.

---

## Patch 11 - the inventory and the audit tool

**Owner file:** `tools/connectivity_audit.py` and
`docs/INTEGRATION_INVENTORY.md`
**Target symbols:** the `CLASSIFICATION` dict (line 208), consulted at
lines 452 and 453; inventory lines 59, 71, and 78
**Dependency:** patch 1 must have landed.

**Why.** Neither `sequence` nor `external_memory` has a `CLASSIFICATION`
entry, so this patch **adds** entries rather than editing them. The
inventory lists both as PENDING with owner `unassigned`, and line 78
records the orphan chain. Patch 1 breaks that chain; the record should say
so.

**Signature change.** Two new dict entries keyed by module name, following
whatever shape the neighboring entries use, with owner `task 07`.

**Call site.** The lookup at line 452.

**State flow.** None.

**Errors.** None.

**Fallback.** Leave both as unassigned orphans. The audit tool is
descriptive; a stale entry misleads a reader but breaks nothing.

**Serialization effect.** None.

**Public API effect.** None.

**Minimal later validation (UNRUN).** The audit reports zero orphaned
modules for this lane after patch 1.

---

## Appendix - things deliberately not done

- **Streaming histograms.** Every histogram builder in the tree takes a
  whole `BinnedMatrix`. A chunked accumulator is a real feature and is
  sized in `docs/EXTERNAL_MEMORY.md` section 13; it is not claimed
  anywhere here, and `check_external_supported(..., streaming=True)`
  raises with what would have to exist.
- **A binary cache format.** The text format follows `serialize.mojo`'s
  proven idiom. Binary would be four to five times smaller for the raw
  spill and is the right next step, but it would have been a second
  unproven IO path in a lane that ran nothing.
- **Cross-thread cancellation.** `CancelToken` is cooperative and
  single-threaded. An atomic flag is described in section 9 of the doc.
- **Distributed external memory.** `collective.mojo` chunks messages, not
  datasets. Combining the two is a real design question and is out of
  scope for a lane that owns two files.
- **Custom objectives from a cache.** `train_external` takes an objective
  code. A Python callback would have to cross the binding per round, which
  is patch 6's territory and a different lane's decision.
- **Source-file parameters.** See patch 9. Not an oversight.
