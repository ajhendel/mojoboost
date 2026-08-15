# connect_21: native interfaces

## K4: sequence ownership audit (2026-08-15)

Lane K4 of the consolidation round. Scope was an audit and a written
recommendation first, then only the safe subset of edits. Outcome: **park
both, change nothing**. No code was edited or deleted this round.

### The two files

| | `src/mojotrees/sequence.mojo` (1,612 lines) | `python/mojotrees/_sequence.py` (1,009 lines) |
| --- | --- | --- |
| Reachability (audit) | orphan; imported only by `external_memory.mojo`, itself an orphan | orphan; only `_arrow.py` and `_polars.py` name it in prose, nothing imports it |
| Compiles / imports today | yes (probe: `mojo run -I src` importing `MemorySequence`, `RawChunk`, `ChunkSchema`, `CancelToken`, `materialize_dense` runs) | yes (probe: `Batches([...numpy blocks...])`, `materialize(...)` returns a `BatchedInput`) |
| Landed in | e6f3959 / 5085097 / e28a24d, same commits as the Python file | same commits as the native file |
| Handoff that specified it | `handoffs/remaining_07_external_memory.md`, deleted in 21ff9fa | `handoffs/remaining_10_ecosystem_inputs.md`, deleted in 21ff9fa |
| Docs | `docs/EXTERNAL_MEMORY.md` (section 0 says every claim is UNRUN; modules deliberately unexported) | `docs/ECOSYSTEM_INPUTS.md` (documents `_sequence.Batches` / `materialize` as internal, "see handoff request P1") |
| Tests | none under `tests/` | none under `python/tests/` |
| Benchmarks | none | none |
| Parity doc | not cited | not cited |

### 1. Which LightGBM surface each targets

Both sit under the same LightGBM feature, `lgb.Dataset(lgb.Sequence(...))`,
but they target different halves of it.

- LightGBM's `lgb.Sequence` is a Python ABC (`__getitem__`, `__len__`,
  `batch_size`). `Dataset` consumes it in two steps: sample rows to build
  the bin mappers (`LGBM_DatasetCreateFromSampledColumn`) and then push the
  batches into the binned dataset (`LGBM_DatasetPushRows`), so the raw
  matrix is never resident. That is a bounded-memory ingestion path.
- `_sequence.py` targets the **Python-facing recognition** of that
  surface: `is_sequence_protocol` accepts anything with the three ABC
  members, `Batches` wraps it (or Arrow record batches, polars frames,
  lists of numpy blocks), `unify_categories` merges per-batch category
  dictionaries, and `materialize` assembles one column-major buffer plus
  label/weight/query pulled from the batches. Its own docstring is blunt
  that this is **not** bounded memory (peak is higher than one matrix) and
  that the bounded-memory core is `sequence.mojo` + `external_memory.mojo`.
  It also carries the ecosystem dispatcher (`adapter_for`,
  `categories_for`, `names_for`, `vector_for`) that fronts `_arrays.check_X`
  for Arrow and polars inputs; that half has nothing to do with sequences
  and is why `_arrow.py` and `_polars.py` refer to it.
- `sequence.mojo` targets the **native ingestion protocol**: a `Sequence`
  trait (`schema`, `n_rows_hint`, `is_repeatable`, `rewind`, `has_next`,
  `next_chunk`), `RawChunk` (dense column-major within the chunk, or a
  chunk-local `CscMatrix`), `ChunkSchema` negotiation with an FNV
  fingerprint for cache manifests, row-coverage enforcement, in-memory
  `MemorySequence` / `CscSequence` adapters over `RawData`, `CategoryTally`,
  and budgeted `materialize_dense` / `gather_*_block` drivers.
  `external_memory.mojo` builds the two-pass binner and chunk cache on
  top of it. This is the analogue of LightGBM's push-rows path, except it
  bins from full passes rather than a sample.

### 2. Same design or competing designs

Neither. They are **two layers of one design**, written to be stacked:
`_sequence.py` explicitly says the Python side of bounded memory "is a thin
loop over `Batches` that hands one batch at a time to a binding this package
does not have yet", and `Batches` exposes batches, row counts, and schemas
separately from `materialize` so that loop can be written without changing
it. There is no function implemented in both files. The only overlapping
concept is "a batch/chunk with an optional label, weight, and query column",
and the two representations differ on purpose (Python objects in
`_sequence.py`; `List[Float64]` column-major and `CscMatrix` in
`sequence.mojo`) because one is the ecosystem boundary and the other is the
binner's input.

### 3. Which is further along, and which direction the architecture wants

- Further along: `sequence.mojo`, by a wide margin as a *design*. It has a
  consumer (`external_memory.mojo`), a full trait with two adapters, schema
  and row-order enforcement, and a documented cache contract. It is also,
  per `docs/EXTERNAL_MEMORY.md` section 0, entirely UNRUN beyond compiling.
- `_sequence.py` is complete and runs today for what it claims
  (materialize-then-train), but it is honestly a convenience layer plus a
  dispatcher, and it is unwired for the same reason as its siblings
  `_arrow.py` and `_polars.py`: the four-line patch in front of
  `_arrays.check_X` was never applied, and that file belongs to connect_07.
- Direction: the house pattern is native core with a thin Python wrapper,
  and this pair already has that shape. Wiring order when someone does the
  feature work is (a) export `sequence` + `external_memory` from
  `src/mojotrees/__init__.mojo` (patch 1 of the deleted remaining_07
  handoff), (b) run the UNRUN checks listed in `docs/EXTERNAL_MEMORY.md`
  section 14, (c) add a binding that accepts one chunk at a time, (d) make
  `_sequence.Batches` the Python loop over that binding, keeping
  `materialize` as the fallback for callers who want one matrix. None of
  that is this round.

### 4. References by tests, benchmarks, docs, handoffs

Docs only. `docs/EXTERNAL_MEMORY.md` cites `sequence.mojo` throughout;
`docs/ECOSYSTEM_INPUTS.md` cites `_sequence.py` and shows
`from mojotrees import _sequence` as an internal-only example. Both docs
point at handoffs (`remaining_07_external_memory.md`,
`remaining_10_ecosystem_inputs.md`) that were removed in 21ff9fa; the
docstring of `_sequence.py` does the same. Those are stale pointers for C0
(docs are C0-only). No test, benchmark, or `tools/` script touches either
module.

### Decision under the deletion bar

Neither file is a superseded duplicate of the other, so condition (b) of the
deletion bar ("the authority demonstrably covers the behavior") cannot be
met in either direction: deleting `sequence.mojo` would orphan
`external_memory.mojo` and remove the only bounded-memory design in the
repo; deleting `_sequence.py` would remove the ecosystem dispatcher that
`_arrow.py` / `_polars.py` are written against. Both are coherent,
unreachable features.

**Disposition: PARK BOTH, connect later.**
- `src/mojotrees/sequence.mojo`: park; connect later via export +
  external_memory validation. Not superseded.
- `python/mojotrees/_sequence.py`: park; connect later via connect_07
  (`_arrays.check_X` front) and a chunk binding. Not superseded.

Audit before and after: unchanged for this section by design (`sequence`,
`external_memory`, `mojotrees._sequence` remain PENDING orphans). This
note is the explanation the coordinator asked for when a lane's section
does not shrink.

### For the coordinator: CancelToken duplication (recorded, not touched)

`CancelToken` is defined in `src/mojotrees/sequence.mojo:115` and
`src/mojotrees/validation.mojo:1654`. They are not identical:

| | sequence.mojo | validation.mojo |
| --- | --- | --- |
| fields | `cancelled: Bool`, `polls: Int` | `cancelled: Bool`, `reason: String` |
| constructor | `none()` | `live()` |
| raise path | `check()` raises `sequence_status_message(SEQ_CANCELLED)` | validation's own error wording |

Same cooperative single-writer semantics, different payload (poll counter
vs reason string). Both are in orphan modules today. `validation.mojo` is
remaining_12 adjacency, so nothing was changed. Suggested resolution when
that lane lands: one `CancelToken` carrying both `polls` and `reason`,
owned by whichever module is exported first, with the other importing it;
`sequence.mojo`'s `check()` would then wrap the shared token to keep the
`SEQ_CANCELLED` wording. Not urgent while both modules are unexported.

### Export-list changes for C0

None.

### What was deferred

Everything that is feature work: exports, the chunk binding, wiring
`Batches` into `Dataset`, the UNRUN checks in EXTERNAL_MEMORY.md, and the
`CancelToken` merge.
