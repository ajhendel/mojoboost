# External memory and the streaming dataset core

`src/mojotrees/trainset.mojo` builds a `Dataset` by holding the whole matrix.
That is the right implementation when the data fits, and nothing in this
document replaces it.

This document describes the two modules that exist for the case where it does
not fit:

- `src/mojotrees/sequence.mojo` - the chunk protocol. A `Sequence` hands out
  `RawChunk`s in ascending row order. Nothing in it holds the whole matrix,
  and the one function that materializes takes a byte budget and raises
  rather than exceeding it.
- `src/mojotrees/external_memory.mojo` - the multi-pass binner, the chunk
  cache, and the trainers over it. It turns a `Sequence` into an
  `ExternalDataset`: a fitted `BinMapper` plus a directory of binned chunk
  files plus a manifest that says what they are and what their bytes hash to.

## 0. Status, in one place

Both modules are exported from `src/mojotrees/__init__.mojo` and are run by
`tests/test_external_memory.mojo` (in the `test` and `test-cpu` suites)
since 2026-08-15. What that test establishes, and what it does not:

| Claim | Status |
| --- | --- |
| mojotrees supports external-memory training | Delivered natively: `build_external_dataset_from_raw` / `build_external_dataset` plus `train_external*` are public and tested. There is no Python binding for the cache path yet; the Python side's chunked ingestion (`Dataset(lgb.Sequence)`, `bindings/sequence_bindings.mojo`) bins the accumulated float64 matrix once and is documented as not bounded-memory |
| The bin edges equal the resident path's, bit for bit | **Verified**: `test_streamed_bins_equal_resident_bins` compares `edges` and `edge_offsets` to `binning.fit_bins` on the same matrix, with a block width of one feature |
| A chunk binned alone equals its slice | **Verified**: the same test compares every `binned_chunk(i)` to its rows of `mapper.transform` |
| The chunk files round trip exactly | **Verified** for dense caches: `test_cache_reopens_and_refuses_the_wrong_mapper` reopens through the checksums and compares the materialized bins; the wrong mapper is refused by fingerprint |
| Training from a cache equals resident training | **Verified**: `test_external_training_matches_resident_training` compares `train_external` to `train_dataset` prediction for prediction |
| Row coverage rejects gaps and overlaps | **Verified**: `test_row_coverage_rejects_gaps_and_overlaps` |
| Training runs without materializing | **No.** Explicitly no. See section 11 |
| `ExternalDataset` is a `Dataset` | No. Separate type; `materialize_binned` is the bridge |
| Sparse cache equals dense cache | Not yet tested |

## 1. The chunk protocol

A `RawChunk` is a contiguous block of rows in the caller's own representation.

Dense chunks are column-major *within the chunk*, `values[f * n_rows + r]`.
That is not a new layout: it is exactly what `binning.fit_bins` and
`BinMapper.transform` already read, so a chunk can be handed to the resident
binner with no repacking. Sparse chunks are a chunk-local `CscMatrix` whose
row indices are relative to the chunk, and they stay sparse the whole way
into `sparse.fit_bins_csc`, exactly as `raw_data.RawData` keeps them.

A chunk also carries the row fields belonging to its rows (label, weight,
init score, query ids) and `row_id_base`, the global id of its first row.

The trait is six methods:

```mojo
trait Sequence:
    def schema(self) -> ChunkSchema: ...
    def n_rows_hint(self) -> Int: ...
    def is_repeatable(self) -> Bool: ...
    def rewind(mut self) raises: ...
    def has_next(self) -> Bool: ...
    def next_chunk(mut self) raises -> RawChunk: ...
```

Three implementations ship in this lane. `MemorySequence` and `CscSequence`
chunk a resident matrix by row blocks, which is what makes the streaming path
testable without a file, and what
`memory_sequence_from_raw` / `csc_sequence_from_raw` produce from the
ingestion type the rest of the tree already uses. `RawCacheSequence` replays
raw chunk files written by `spill_source`.

### Schema negotiation

The source declares one `ChunkSchema` and every delivered chunk is checked
against it by `require_chunk_schema`: the feature count, the representation,
and which optional row fields it brought. A source whose third chunk stops
carrying weights would otherwise silently train a differently weighted model,
so it raises at that chunk.

Feature names and the categorical declaration are the caller's policy rather
than any chunk's property, so they are compared schema against schema by
`ChunkSchema.require_compatible`, which happens when there really are two
schemas (a cache manifest's and a caller's). `ChunkSchema.fingerprint` folds
all of it into one 64-bit integer, and that integer is what a manifest stores
and re-checks on reopen.

### Row identity

Row identifiers are the driver's, not the source's. A pass assigns global ids
`0, 1, 2, ...` in delivery order and checks each chunk's `row_id_base`
against the running total. A source that skips, repeats, or reorders rows is
rejected at the chunk that does it. `SequenceStats.observe` is where that
check lives, and `check_row_coverage` is the end-of-pass assertion that the
delivered `RowIdRange`s tile `[0, n_rows)` exactly once with no gap and no
overlap.

This is why "covers every row exactly once" is a property the code enforces
rather than a promise this document makes. It is also what makes the second
and later passes safe: the transform pass re-checks that it cut the source
the same way the census pass did, and raises if it did not.

## 2. Repeatable iteration, and sources that are not

Exact multi-pass binning reads the source more than once, so a `Sequence`
says whether it can be read again (`is_repeatable`) and how (`rewind`).

`build_external_dataset` refuses a non-repeatable source by name. It does not
fall back to binning from the first chunk, or from a sample, because a
silently sampled binning is a different model that looks like the same one.
The fix it names is `spill_source`, which drains the source once into raw
chunk files and returns a `RawCacheSequence` over them. That is one extra
pass and one extra copy of the raw data on disk, and it is the caller's
decision to pay it.

## 3. The passes, and the memory ceiling

Bin edges are global quantiles. A quantile of a column needs the whole
column, so exact binning of a dataset that does not fit is a question of
which whole thing you are willing to hold. This module holds a *block of
columns*.

1. **Census**, one pass, `sequence.gather_row_fields`. Counts the rows, fixes
   the row identity, and concatenates label, weight, init score, and query
   ids. Those are `n_rows` long each and are the caller's own data at any
   size, so holding them is not a new cost. The same pass runs
   `CategoryTally` over the declared categorical columns, so a column with
   more distinct codes than the binning can keep is reported before the
   expensive passes rather than after them.

2. **Bin construction**, `ceil(n_features / block_width)` passes. Each pass
   gathers one block of columns for every row (`gather_dense_block` or
   `gather_sparse_block`) and hands it to the *unmodified* `binning.fit_bins`
   or `sparse.fit_bins_csc`. `_merge_block_mappers` concatenates the per-block
   edges, offsets, missing reservations, and categorical tables into one
   `BinMapper` over all features.

3. **Transform**, one pass. Each chunk is binned by the fitted mapper and
   written to its own cache file with a checksum.

Total reads of the source: `2 + ceil(n_features / block_width)`.

### Why the edges are the resident path's

Not by agreement, by construction. Feature `f`'s edges come from calling the
same quantile binner on the same `n_rows` values of feature `f`, in the same
order. The only thing this module contributes is the order in which those
values were collected, and they are collected in ascending row order, which
is the order a resident column is already in.

The same argument covers the transform pass. Binning is per element, so a
chunk binned on its own equals its slice of the whole matrix binned at once.
This is the same argument `sparse.transform_csc` already makes about absent
entries, and the implicit-zero rule is unchanged: an absent sparse entry is
numeric `0.0`, not missing.

Fewer passes are possible only by giving up exact quantiles. This lane does
not do that. There is no sampled or sketched binner here, and
`subsample_for_bin` remains unimplemented in the tree.

### Choosing the ceiling

`feature_block_width(n_features, n_rows, budget_bytes)` is
`budget_bytes // (n_rows * 8)`, clamped to `[1, n_features]`. A block of
`w` columns costs `w * n_rows * 8` bytes for a dense source, which is the
only allocation the bin-construction pass makes.

| `bin_memory_budget` | 10M rows, 200 features, dense | passes |
| --- | --- | --- |
| 256 MiB (default) | 3 columns per block | 67 bin passes, 69 total |
| 1 GiB | 13 columns per block | 16 bin passes, 18 total |
| 8 GiB | 107 columns per block | 2 bin passes, 4 total |

The memory ceiling is the parameter and the number of passes is what it buys.
That is the whole trade, and it is exposed rather than guessed.

`chunk_rows` is a separate knob and a smaller one. It sets the transform
pass's working set (one chunk of raw values in, one binned chunk out) and the
number of cache files. It does not affect the fitted bins at all, which is
section 4.

For a sparse source the block cost is the stored entries of those columns
rather than `w * n_rows * 8`, so the same budget buys a much wider block. The
arithmetic in `feature_block_width` is the dense worst case and is used for
both, deliberately: it under-uses memory on sparse data rather than
over-committing on it.

## 4. Determinism, and what is not invariant

Two builds of the same source with the same parameters produce the same bin
edges, the same chunk boundaries, the same row identifiers, and the same
checksums, byte for byte.

Nothing here samples. Nothing here depends on the thread count either: the
parallelism is inside `fit_bins` and `BinMapper.transform`, both already
documented as bit-identical to their serial paths, and the chunk cut is
`ChunkPlan`, a pure function of `(n_rows, chunk_rows)`.

What is **not** invariant is the chunk size. The same data cut into different
chunks produces the same bins and the same trained model, but a different set
of cache files with different checksums. The manifest records the cut, so a
cache built with one `chunk_rows` cannot be silently mistaken for one built
with another.

What is also not invariant across builds is the *source*. Two different
sources that deliver the same rows produce the same cache; a source that
delivers different rows produces a different cache and says so through the
schema fingerprint and the row-coverage check, not through a wrong model.

## 5. Categorical dictionaries

Declared categorical columns are tallied during the census pass by
`CategoryTally`, which stores one sorted list of packed
`slot * 2**32 + code` keys. Packing is what lets one list serve every
categorical column without a per-column allocation; `CATEGORY_KEY_STRIDE` and
`MAX_CATEGORY_CODE` are the two constants that make it unambiguous, and
`MAX_CATEGORY_CODE` mirrors `categorical._MAX_CATEGORY`, which is private
today (sharing it instead of mirroring is a one-line follow-up).

The tally counts distinct codes and missing rows per column. On a sparse
source the implicit zeros are counted as code `0`, which is what
`sparse._distinct_codes_and_counts_csc` already does, so a sparse and a dense
encoding of the same data tally the same.

What the tally is *for* is the early report. The dictionary that actually
ships in the model is still built by `fit_bins` / `fit_bins_csc` on the block
pass, through `categorical._keep_most_frequent`, with LightGBM's rule
unchanged: keep `max_bin - 1` codes, ties broken toward the smaller code.
`CategoryTally.cap_exceeded` and `report()` tell a caller *before* the
expensive passes that a column has more distinct codes than the binning can
keep, so that a caller who cares can raise `max_bin` or recode rather than
discover the truncation in the fitted model.

## 6. What a cache is

```
<directory>/<prefix>.manifest.mbx     the manifest
<directory>/<prefix>.fields.mbx       label, weight, init score, query ids
<directory>/<prefix>.bins.<i>.mbx     one binned chunk
<directory>/<prefix>.raw.<i>.mbx      one spilled raw chunk (spill_source)
```

Paths are derived from `CacheLayout` at open time and are never stored in the
manifest, so a cache directory can be moved or renamed and still open. What
the manifest stores instead is the count and the checksums, which is what
actually has to survive the move.

The manifest holds the format version, the schema fingerprint, the mapper
fingerprint, the shape, the binning parameters, the feature names, the
categorical declaration, the row-field checksum, and one record per chunk
(index, base row, row count, stored entries, checksum).

A cache does **not** store its `BinMapper`. That is deliberate. Bins are only
meaningful under the binning that produced them, a trained `Model` already
carries that binning in `Model.mapper`, and a second copy in the cache is a
second thing that can disagree. `open_external_dataset` therefore takes the
mapper from the caller and checks `mapper_fingerprint` against the manifest,
so handing over the wrong one is an error at open rather than a run that
quietly means something else. Serializing the mapper *alongside* a cache is
`serialize.mojo`'s existing job and needs nothing new from this lane.

## 7. Why the files are text

Every file is a whitespace-separated token stream in the style of
`serialize.mojo`, with floats written as their IEEE-754 bit patterns
(`String(x.to_bits())` out, `bitcast[DType.float64, 1]` back). A round trip
is therefore exact and locale-free: no decimal formatting, no parsing
tolerance, no rounding.

Text costs roughly four bytes per bin where one byte would do. That is a
deliberate trade for using the one file primitive this repository has already
proven, `open(path, "w")` and `open(path, "r").read()`. Inventing a binary
block format would mean inventing a byte-level IO path that nothing in this
tree has run. The binary format is recorded as future work in section 12
rather than written here, and the on-disk format carries `EXT_VERSION` so
that it can be introduced as version 2 without ambiguity.

One file per chunk, rather than one file with an offset table, is what keeps
every read bounded. Reopening chunk 4000 reads chunk 4000 and nothing else.

## 8. Checksums, and what they are not

`checksum_text` is FNV-1a 64 over the file's bytes. It is a **non-
cryptographic** integrity check. It detects a truncated write, a partially
flushed file, a corrupted disk block, and a chunk file edited between the
build and the run. It does not detect a deliberate forgery, and nothing here
should be read as saying it does. A cache directory is as trusted as the
filesystem it sits on.

`ExternalDataset.verify()` re-reads every file and re-checks every checksum.
It is the expensive check and it is on by default at open
(`ExternalMemoryParams.verify_on_open`), because opening a cache is rare and
training from a corrupted one is expensive. Per-chunk checks also run on
every individual chunk read, which is the cheap check and is unconditional.

The same FNV-1a construction gives `ChunkSchema.fingerprint` and
`mapper_fingerprint`. Those are identity checks over structure rather than
integrity checks over bytes, and they exist so that a mismatch is reported as
"this is not the same schema" instead of appearing later as wrong splits.

## 9. Cancellation

`CancelToken` is cooperative and single-threaded. Every driver in these two
modules polls it at chunk boundaries and raises `SEQ_CANCELLED` when it is
set, so a cancelled pass stops within one chunk of the request rather than at
the end of the data.

What it deliberately is not: a flag another thread can set. The token is
passed `mut` down one call stack, which is exactly what a caller-driven loop
or a chunk callback can use, and exactly what a background canceller cannot.

**What a cross-thread token would take.** An atomic flag, `poll` becoming an
atomic load, and `cancel` an atomic store, plus a decision about whether
cancellation is observable inside `fit_bins` and `BinMapper.transform`, which
are the parts of a pass that do not return to this code until they finish. A
chunk-granular cancel that cannot interrupt a single block's `fit_bins` is
still a chunk-granular cancel; the honest version of the feature says which
granularity it has. `collective.STATUS_CANCELLED` already exists for the
distributed case and is the code a reduced cancellation is reported through
there; the two vocabularies are separate on purpose, because this one is not
reduced across ranks.

## 10. Cleanup

`ExternalDataset.discard()` truncates every file the cache owns, marks the
dataset dead so every later read raises rather than returning stale data, and
returns the list of paths.

It truncates rather than unlinks because this module writes files through
`open`, the only filesystem primitive already in use in this repository, and
inventing an unlink path that nothing has run would be a claim rather than a
feature. The result is correct if untidy: a truncated cache cannot be read
back as data, since its manifest is gone and its chunk files are empty, and
the returned paths are exactly what a caller's own cleanup removes.

**What a real unlink would take.** A confirmed removal primitive in the Mojo
standard library, a decision about the failure mode when a file is already
gone or the directory is read-only, and a decision about whether `discard`
removes the directory itself (it should not: the directory is the caller's,
and this module never created it). That change is not made.

Nothing here removes files on destruction. A cache outlives the process
unless a caller discards it, which is the behavior a cache should have.

## 11. What can be trained from an external dataset

`ExternalCapabilities.current()` is this table in code, so that a caller can
ask instead of reading a document, and `check_external_supported` raises with
a message that names what would have to exist.

| Capability | Supported | How, and what it costs |
| --- | --- | --- |
| Streaming histograms (train without materializing) | **No** | Every histogram builder in the tree takes a whole `BinnedMatrix` or `SparseBinnedMatrix`. This is the one real limit and it is stated as one |
| CPU training, materialized | Yes | `train_external`. `materialize_binned` is `n_rows * n_features` bytes, one eighth of the raw float64 matrix, which is why an external build can train data a resident build cannot even read |
| Sparse CPU training, materialized | Yes | `train_external_sparse`, over the stored entries rather than every cell, so a sparse cache trains within a budget a dense one could not |
| GPU training, materialized | Yes | `train_external` with `device=GPU_DEVICE`, through the same `device.resolve_device` every other trainer uses. Device transfer is of the materialized matrix, not of chunks: there is no chunk-by-chunk upload path, and the GPU sees exactly what `trainset.train_dataset` would have given it |
| Multiclass | Yes | `train_external_multiclass`, `train_external_sparse_multiclass`. CPU only, as `trainset.train_dataset_multiclass` is |
| Ranking groups | Yes | `external_groups` assembles groups from the whole query-id column, not per chunk, so a query whose rows straddle a chunk boundary is one query and not two. `ranking.groups_from_query_ids` still enforces contiguity, which is the property that actually matters. `train_external_ranker` is the trainer |
| Bagging | Yes | `BaggingParams` passes straight through to the trainers. Row indices are over the materialized matrix, so a bag is drawn from all `n_rows` and not per chunk. Bagging never changes the binning, which was fitted before any row was sampled |
| GOSS | Yes | `GossParams`, same route as bagging |
| EFB | Yes, by the caller | `efb.fit_bundles` on a `materialize_sparse_binned` result, or `efb.fit_bundles_dense` on a `materialize_binned` result. This module does not call them; it produces exactly the input they take |
| Init score | Yes | Carried per row through the census pass and stored in the fields file. Refused for multiclass and ranking for the reason it is refused everywhere else in the tree: one offset per row cannot say what each class starts from. The GPU path refuses it exactly as `train_dataset` does |
| Continued training | Yes | `update_external`, `update_external_multiclass`. `Model.mapper.matches(dataset.mapper)` is required, because a bin index has to mean the same thing to the trees already in the model and to the ones about to be grown. CPU, as continued training is from a resident dataset too |
| Custom objectives, metrics, callbacks | Not wired | Nothing blocks them. They take a `BinnedMatrix` like everything else, so a caller materializes and calls `model.fit_custom` or `custom_metric`'s trainers directly. No external-memory-specific function was added for them because it would be a rename of an existing one |
| Distributed | Not wired | Out of this lane. The row identifiers are the piece a partitioned external build would need, and `distributed.mojo`'s partition contract is the piece it would have to satisfy |

The chunk files, the row identifiers, and the per-chunk checksums are exactly
the substrate a chunked histogram accumulator would need. None of them is a
claim that one exists.

## 12. What is deliberately not supported

LightGBM spells external memory as **text-parsing configuration**, because
its dataset is built by its own parser: `two_round`, `header`,
`label_column`, `weight_column`, `group_column`, `ignore_column`,
`data=/path/to/file`.

None of those appear in `ExternalMemoryParams`, and none should be added.
Parsing in mojotrees lives in the caller, which is what makes an Arrow batch,
a NumPy view, and a CSV reader equally valid sources of a chunk. A parameter
named `two_round` with no parser behind it would be spelling for its own
sake, and `docs/LIGHTGBM_PARITY.md` already classifies that whole family as
`unsupported` with the same reasoning.

What `ExternalMemoryParams` does hold is exactly two kinds of thing: what
changes the fitted binning (`max_bin`, `use_missing`, the categorical
declaration, the feature names) and what changes the memory ceiling
(`chunk_rows`, `bin_memory_budget`), plus `verify_on_open`.

The corresponding parity rows are `Sequence`, `Dataset.save_binary`, and the
external-memory entries; moving them is the parity lane's edit to
`docs/LIGHTGBM_PARITY.md`, now that section 0's checks have run.

## 13. Future work, sized

- **A chunked histogram accumulator.** The one thing that would turn
  `streaming_histograms` to `True`. It means a histogram builder that takes a
  chunk and a row-index subset and accumulates into a shared bin array, plus
  a split search that reads that array. The bagging and GOSS row indices are
  global today, so they would need a chunk-local translation, which the
  `RowIdRange` table already supports.
- **A binary chunk format**, as `EXT_VERSION = 2`. One byte per bin instead
  of about four, and a fixed header. It needs a byte-level file primitive
  this tree has not exercised, and it is worth doing only after the text
  format has been validated end to end, because a format bug and an IO bug
  look the same.
- **A cross-thread `CancelToken`**, section 9.
- **A real unlink in `discard`**, section 10.
- **Prefetch of chunk `i+1` while chunk `i` is being binned.** The transform
  pass is IO-bound by construction and single-threaded by choice. This is a
  performance change with no effect on the output, which makes it exactly the
  kind of change to make after there is a benchmark rather than before.
- **`Dataset` unification.** `ExternalDataset` is a separate type;
  `materialize_binned` plus `Dataset`'s internal assembling constructor is
  the bridge, and a `Dataset.from_external` that spells it is not written.

## 14. What is unverified

Section 0's table is the record. Still unrun: that a sparse cache and a
dense cache of the same data train the same model, and any comparison
against LightGBM's own external-memory build. The remaining checks the
original design listed (edges, chunk slices, round trip, row coverage, the
partial field moves in `memory_sequence_from_raw` and
`csc_sequence_from_raw`, which are now copies) are in
`tests/test_external_memory.mojo`.
