# Task 09: feature-parallel, voting-parallel, and distributed GPU contracts

Status. Two new modules and one new document landed inside this lane's
exclusive ownership. Nothing outside those four files was touched. Nothing
was built, run, tested, benchmarked, staged, or committed. No test file was
written and no process was started, per the lane's instructions.

Files owned and written.

- `src/mojoboost/distributed_strategies.mojo` (new, about 1440 lines)
- `src/mojoboost/distributed_gpu.mojo` (new, about 780 lines)
- `docs/DISTRIBUTED_STRATEGIES.md` (new)
- `handoffs/remaining_09_distributed_strategies.md` (this file)

**Nothing here is operational.** Feature parallel and voting parallel are
cores with no grower calling them, and `require_strategy` refuses both
unconditionally through `UNSUPPORTED_NO_DRIVER`. Distributed GPU is a
contract with five open gates and no device collective behind it, and
`require_distributed_gpu` refuses it. No multi-process run and no
multi-device run has happened in this repository, so no claim in either
module or in the document is backed by an execution.

## Coordination with CONNECT_EVERYTHING Task 13

That lane was live in this shared checkout during this task and landed at
least commit 860b1cf while it ran, growing `distributed.mojo` from 930 to
1196 lines and `distributed_transport.mojo` from 1844 to 2311. This lane
built on that work rather than beside it. It defines no transport, no
endpoint, no runtime spec, no shard plan, and no status code. Symbols
imported from Task 13's files and from `collective.mojo`, all confirmed
present after that commit.

| Symbol | Owner file | Used for |
|---|---|---|
| `Collective`, `agree_equal_ints`, `agree_status`, `zeros_f64`, `zeros_int` | `collective.mojo` | every collective in both modules |
| `hosts_whole_world` | `collective.mojo` | deriving `multi_process` in `require_strategy` |
| `STATUS_LAYOUT_MISMATCH`, `STATUS_SHAPE_MISMATCH`, `STATUS_UNSUPPORTED` | `collective.mojo` | `strategy_statuses` |
| `f64_bits`, `f64_from_bits`, `digest_ints` | `distributed_transport.mojo` | candidate gain encoding, partition digest |
| `transport_available` | `distributed_transport.mojo` | the multi-process gate |
| `Histogram` | `histogram.mojo` | packing and unpacking |
| `SplitInfo`, `find_best_split` | `split.mojo` | `search_owned_features` forwards, never reimplements |
| `TreeParams`, `OutputBounds`, `CategoricalSpec`, `CAT_BITSET_WORDS`, `cat_empty` | `tree.mojo`, `monotone.mojo`, `categorical.mojo` | parameter and constraint pass-through |

Risk to this lane. If Task 13 renames `transport_available`,
`hosts_whole_world`, `f64_bits`, `f64_from_bits`, or `digest_ints` before it
lands, both new modules stop compiling. The fix is an import rename in two
files and nothing else. No logic in either module depends on the spelling,
and neither module reads a transport field directly.

## What is implemented, and what each mode refuses

| Capability | State | Refused by |
|---|---|---|
| Feature partition arithmetic, ownership, digest | implemented | n/a |
| Candidate encode, all-gather, deterministic election | implemented | n/a |
| Per-rank owned-feature split search | implemented, forwards to `find_best_split` | `search_owned_features` refuses `params.extra.needs_grower_support()` |
| Feature-parallel training | **not implemented, no driver** | `require_strategy` |
| Top-k selection, vote reduction, elected-feature reduction | implemented | n/a |
| Voting-parallel training | **not implemented, no driver** | `require_strategy` |
| Voting exactness | **not exact by construction** | `voting_is_exact()` returns False |
| Global fixed-point scale agreement | implemented | n/a |
| Fixed-point word widen, narrow, plane check, reduction | implemented | `narrow_words` refuses anything outside Int32 |
| Distributed GPU training | **not implemented, five open gates** | `require_distributed_gpu`, `check_gpu_strategy` |
| Device-resident collective | **declared as a trait, never bound** | `UnavailableDeviceCollective`, `resolve_device_collective` |

The election is deterministic because feature partitions are contiguous
ascending blocks, so ascending rank order equals ascending feature order, and
a strict-improvement scan over gathered candidates reproduces
`find_best_split`'s tie-break exactly. The all-gather is emulated with
`allreduce_max_int` over per-rank slots holding non-negative encoded words,
so no new `Collective` trait method was needed. Gains cross as two 32-bit
halves of their IEEE-754 bit pattern. All of this is argued in
`docs/DISTRIBUTED_STRATEGIES.md` sections 1 and 3.

The distributed GPU result worth carrying forward. Per-rank fixed-point
scales are **not additive**, so summing two ranks' Int32 planes that were
quantized against different scales is silently wrong. The scale has to be
globally agreed before the kernel runs, which costs one two-element Float64
reduction per boosting round, not per node. With a global scale the summed
words are bounded by `2^30 + total_rows / 2`, inside `Int32.MAX`,
independent of world size, and integer reduction is order-independent, so
the GPU path makes a *weaker* determinism demand on a transport than the CPU
Float64 path does. `check_fixed_point_headroom` enforces the bound and
`agree_fixed_scales` does the agreement.

One number that is deliberately not claimed. The fixed-point exchange is one
integer reduction against the Float64 path's three, which is a third of the
round trips. It is **not** fewer bytes today, because `Collective` reduces
`List[Int]` and each 32-bit word stages as 64 bits.
`staged_saving_ratio` therefore returns 1.0 and `native_saving_ratio` returns
2.0, and the second one is only reachable through patch 5 below.

## Integration performed inside owned files

Everything below is already wired, not scaffolded.

- `check_strategy_world` is the single entry point a driver needs. It calls
  `require_strategy` (which reads `hosts_whole_world` and
  `transport_available` itself, so a caller cannot pass a `transport_ready`
  its build does not have), then `agree_status` over `strategy_statuses`,
  then `agree_strategy`, in the order `_grow_tree_distributed` already uses.
- `elect_split_collective` is the whole per-node feature-parallel seam. It
  stands exactly where a single-node grower's `find_best_split` call stands.
- `search_owned_features` forwards to the one real `find_best_split`, so
  monotone constraints, categorical partitions, missing routing, the
  interaction allow mask, the CEGB cost, and the gain floor hold by
  construction. It also applies the two guards `tree._search` applies, because
  a rank that skipped them would propose a split its peers refused and the
  election cannot tell the difference.
- `pack_selected` / `allreduce_selected` / `unpack_selected` form the whole
  voting-parallel node seam, and `allreduce_selected` uses the same three
  reductions `allreduce_histogram` uses so the two modes share one shape.
- `distributed_gpu` imports the strategy codes from `distributed_strategies`
  rather than restating them, and `check_gpu_strategy` names the mode it
  refuses using `strategy_name`.
- `check_fixed_point_contract` and `fixed_scale_from_total` mirror
  `histogram_gpu._FIXED_ONE` and `_fixed_scale` exactly, and the mirror is
  checkable at runtime rather than by comment. Patch 4 removes the mirror.

## READY-TO-APPLY INTEGRATION PATCHES

Each patch is blocked only by file ownership. None requires a design
decision from this lane. Every validation line is **UNRUN**.

---

### Patch 1. Export the two modules

**Target file / symbol** `src/mojoboost/__init__.mojo`, the import block at
lines 121 to 128 (currently `from .distributed import (...)`).

**Signature** Add two blocks, alphabetically after `.distributed`.

```mojo
from .distributed_gpu import (
    DeviceCollective,
    GpuExchangePlan,
    GpuFixedScales,
    agree_fixed_scales,
    check_fixed_point_headroom,
    distributed_gpu_available,
    distributed_gpu_unavailable_detail,
    gpu_exchange_plan,
    reduce_fixed_words,
    require_distributed_gpu,
)
from .distributed_strategies import (
    STRATEGY_DATA_PARALLEL,
    STRATEGY_FEATURE_PARALLEL,
    STRATEGY_SERIAL,
    STRATEGY_VOTING_PARALLEL,
    FeaturePartition,
    StrategyCapabilities,
    check_strategy_world,
    elect_split_collective,
    parse_strategy,
    search_owned_features,
    strategy_capabilities,
    strategy_cost_plan,
    strategy_name,
    voting_is_exact,
)
```

**Call site** None. This is a re-export only.

**State flow** None.

**Errors** None added. Every exported function that raises keeps its own
message.

**Ownership** `__init__.mojo` is shared and is edited by several lanes. Apply
as one contiguous block so a merge conflict is a block move rather than a
line interleave.

**Fallback** If the package prefers narrow exports, export only
`check_strategy_world`, `strategy_capabilities`, `distributed_gpu_available`,
and `distributed_gpu_unavailable_detail`. Those four are what an outside
caller needs to ask whether a mode exists.

**Serialization effect** None. Neither module touches the model format.

**Public API effect** Additive. Fourteen plus ten new names at package
scope, all of which either describe or refuse.

**Dependency** None beyond the two modules being present.

**Minimal later validation, UNRUN** `mojo run -I src` on any file that does
`from mojoboost import strategy_name` compiles. No behavior to assert.

---

### Patch 2. Strategy selection on the distributed trainer

**Target file / symbol** `src/mojoboost/distributed.mojo`,
`DistributedRunOptions` and `train_distributed_run` (line 1478).

**Signature** Add one field, defaulted so no existing caller changes.

```mojo
# in DistributedRunOptions
var strategy: Int = STRATEGY_DATA_PARALLEL
```

**Call site** In `train_distributed_run`, immediately after the existing
configuration agreement and before the first tree, one line.

```mojo
check_strategy_world(
    comm,
    options.strategy,
    FeaturePartition(n_features, comm.world_size()),
    n_features,
    DEFAULT_TOP_K,
)
```

**State flow** `options.strategy` is read once per run and never per node.
`check_strategy_world` performs two collectives, both once per run. It
returns nothing and mutates nothing. Rank state is unchanged on success.

**Errors** Raises the identical message on every rank for an unknown
strategy, a world size of 1 with a parallel strategy, a world size above 1
with `STRATEGY_SERIAL`, a multi-process world with no transport, or any
selection of feature or voting parallel (`UNSUPPORTED_NO_DRIVER`). Because
the gate's inputs are identical on every rank it raises everywhere or
nowhere, so no rank is left waiting in a collective a refused peer will never
enter. This is the property the gate ordering in `check_strategy_world`
exists to preserve, and it must not be reordered behind the two agreements.

**Ownership** `distributed.mojo` belongs to CONNECT_EVERYTHING Task 13.

**Fallback** With the field defaulted to `STRATEGY_DATA_PARALLEL`, the gate
is a no-op for every existing caller and every existing test. A build that
does not want the field at all can call `check_strategy_world` with a literal
`STRATEGY_DATA_PARALLEL` and still get the world-size and transport checks.

**Serialization effect** None on the model. If `DistributedRunOptions` is
ever serialized into a job description, the new field is an Int and should
default to 1 when absent so an older description means data parallel.

**Public API effect** Additive and defaulted. No existing signature changes.

**Dependency** `from .distributed_strategies import DEFAULT_TOP_K,
FeaturePartition, STRATEGY_DATA_PARALLEL, check_strategy_world`. That import
makes `distributed.mojo` depend on `distributed_strategies.mojo`, which in
turn depends on `distributed_transport.mojo` for `f64_bits`,
`f64_from_bits`, `digest_ints`, and `transport_available`. **Check for an
import cycle before applying.** `distributed_strategies` does not import
`distributed`, so the cycle does not exist today, and it must not be created
by moving anything from `distributed.mojo` into `distributed_strategies.mojo`
later.

**Minimal later validation, UNRUN** A four-rank `LocalCollective` run with
the default options trains as it does today, byte-identical model. A run with
`strategy = STRATEGY_FEATURE_PARALLEL` raises on every rank with a message
containing "not operational". A single-rank run with
`STRATEGY_DATA_PARALLEL` raises "world size of at least 2".

---

### Patch 3. Voting-parallel node seam in the grower

**Target file / symbol** `src/mojoboost/distributed.mojo`,
`_grow_tree_distributed` (line 1098), at the point where it currently calls
`allreduce_histogram(comm, hist)`.

**Signature** No signature change. A branch on `options.strategy`.

```mojo
if options.strategy == STRATEGY_VOTING_PARALLEL:
    var local_best = local_gains_per_feature(hist, params)   # see below
    var votes = allreduce_votes(comm, local_votes, n_features)
    var elected = elect_voted_features(votes, top_k)
    var packed = pack_selected(hist, elected)
    allreduce_selected(comm, packed)
    unpack_selected(hist, elected, packed)
    # then search with `features = elected`
else:
    allreduce_histogram(comm, hist)
```

**State flow** `hist` goes in local and comes out with the elected features'
cells globally summed and every unelected feature's slice **zeroed**, not
stale. The subsequent split search must pass `elected` as its `features`
list, because an unelected feature now reads as a feature with no rows. That
coupling is the one thing a driver can get wrong, and it is why
`unpack_selected` zeroes rather than leaves.

`local_gains_per_feature` does not exist and is the one piece this lane
cannot supply, because producing a per-feature best gain means a per-feature
scan the grower owns. The two options are a `find_best_split` variant that
returns a gain per feature, or `select_top_k` fed from the grower's existing
per-feature loop. `select_top_k(gains, found, k)` already takes exactly
`List[Float64]` and `List[Bool]` of length `n_features` and returns the
ascending feature ids.

**Errors** `allreduce_votes` raises on a vote list that is not ascending or
out of range. `elect_voted_features` raises on `n_select < 1`.
`pack_selected` raises on a non-ascending or out-of-range selection and
`unpack_selected` raises on a shape mismatch. All are local and all are
raised identically on every rank, because every rank runs the same check on
the same globally reduced vote counts.

**Ownership** `distributed.mojo`, Task 13.

**Fallback** Voting parallel is refused by patch 2's gate, so this branch is
dead code until `UNSUPPORTED_NO_DRIVER` is lifted for it. Land the branch and
leave the gate in place, and the mode costs nothing but is one line from
being testable.

**Serialization effect** None on the model format. The trees a voting run
produces will differ from a data-parallel run's trees whenever the globally
best feature was voted out, which is why `voting_is_exact()` returns False
and why no parity claim may be made for this mode.

**Public API effect** None beyond patch 2's field.

**Dependency** Patch 2. Plus `from .distributed_strategies import
allreduce_votes, allreduce_selected, elect_voted_features, pack_selected,
select_top_k, unpack_selected`.

**Minimal later validation, UNRUN** On a four-rank `LocalCollective` with
`top_k >= n_features`, a voting run must produce a model **identical** to the
data-parallel run, because electing every feature makes the mode exact. That
single assertion is the strongest cheap test of the whole voting path and it
needs no transport and no second process. With `top_k = 1`, the run must
still complete and produce a valid model with strictly fewer distinct split
features.

---

### Patch 4. A public node search wrapper in `tree.mojo`

**Target file / symbol** `src/mojoboost/tree.mojo`, the private `_search`
forwarder.

**Signature** Expose the guards, not the whole grower.

```mojo
def search_node(
    hist: Histogram,
    params: TreeParams,
    n_rows: Int,
    features: List[Int],
    depth: Int,
    node: Int,
    tree_index: Int,
    parent_output: Float64,
    allowed: List[Bool] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    bounds: OutputBounds = OutputBounds.unbounded(),
    cats: CategoricalSpec = CategoricalSpec.none(),
) raises -> SplitInfo
```

**Call site** `distributed_strategies.search_owned_features` would forward
to this instead of restating the two guards and calling `find_best_split`
directly.

**State flow** Pure. Reads a histogram, returns a `SplitInfo`.

**Errors** Unchanged from `find_best_split`.

**Ownership** `tree.mojo` belongs to the tree lane.

**Fallback** Not applying this patch costs nothing today.
`search_owned_features` already copies the two guards verbatim
(`params.max_depth > 0 and depth >= params.max_depth`, and
`n_rows < 2 * params.min_data_in_leaf or n_rows < 2`) and the copy is marked
as a copy in its docstring. The risk is drift, not incorrectness. **If the
tree lane adds a third guard to `_search`, this lane's copy must be updated
in the same change or feature-parallel ranks will disagree with the
single-node grower.**

**Serialization effect** None.

**Public API effect** One new public function in `tree.mojo`. It exposes
nothing that `find_best_split` does not already expose.

**Dependency** None.

**Minimal later validation, UNRUN** `search_node` and
`search_owned_features` on a single-rank world with a one-rank partition
return the same `SplitInfo` as `find_best_split` over the full feature list,
field for field.

---

### Patch 5. A 32-bit word reduction on the transport

**Target file / symbol** `src/mojoboost/distributed_transport.mojo` and the
`Collective` trait in `src/mojoboost/collective.mojo`.

**Signature**

```mojo
# added to the Collective trait
def allreduce_sum_i32(mut self, mut values: List[Int32]) raises: ...
```

**Call site** `distributed_gpu.reduce_fixed_words` would take
`List[Int32]` directly and drop the `widen_words` / `narrow_words` pair
around it.

**State flow** In place, same as the existing `allreduce_sum_int`.

**Errors** Overflow is the caller's problem and is already handled.
`check_fixed_point_headroom` proves the sum stays inside `Int32.MAX` for any
world size given a globally agreed scale, and `narrow_words` raises on
anything outside the range today. With a native Int32 reduction the
`narrow_words` check moves to being the transport's own bound check.

**Ownership** `collective.mojo` and `distributed_transport.mojo`, Task 13.
Adding a trait method is a breaking change for every `Collective`
implementation, currently `LocalCollective` and `TransportCollective`.

**Fallback** This patch is a pure optimization and is **not** required for
correctness. Without it, `reduce_fixed_words` widens to `List[Int]`, stages
64 bits per 32-bit word, and moves exactly the bytes the Float64 path moves,
while still saving two of three round trips. `staged_saving_ratio` returns
1.0 to record that honestly. With it, `native_saving_ratio` returns 2.0.

**Serialization effect** The wire message gains a 32-bit element type. That
is a transport protocol change and must bump
`TRANSPORT_PROTOCOL_VERSION`, because a rank that cannot decode a 32-bit
payload must refuse the job rather than misread it.

**Public API effect** Breaking for the `Collective` trait. Every
implementation gains a method.

**Dependency** None, but it should not be applied before a real transport
exists, because a protocol bump with no peers to disagree with buys nothing.

**Minimal later validation, UNRUN** Four-rank `LocalCollective`,
`reduce_fixed_words` over a known plane gives the same result on the Int and
the Int32 path. Then `gpu_exchange_plan(...).native_payload_bytes` equals
half `f64_path_payload_bytes`.

---

### Patch 6. Externally agreed scales on the GPU histogram builder

**Target file / symbol** `src/mojoboost/histogram_gpu.mojo`,
`GpuHistogramBuilder`. Today `stage_gradients` (near line 606) computes
`_fixed_scale(grad)` and `_fixed_scale(hess)` locally and stores them in
`self.g_scale` and `self.h_scale`, and `download_raw` (line 899) fills
`self.host_out` while `histogram_from_host` (line 907) divides by those two
scales.

**Signature** Three additions. Nothing existing changes shape.

```mojo
# 1. make the constants reachable so distributed_gpu stops mirroring them
comptime FIXED_ONE = Float64(1 << 30)      # rename of _FIXED_ONE, or an alias
def fixed_scale(values: List[Float64]) raises -> Float32   # alias of _fixed_scale

# 2. accept scales the world agreed on
def set_fixed_scales(mut self, g_scale: Float64, h_scale: Float64) raises

# 3. expose the downloaded plane so it can be reduced before conversion
def raw_words(self) raises -> List[Int32]
def set_raw_words(mut self, words: List[Int32]) raises
```

**Call site** In a distributed GPU round, between the existing
`download_raw()` and `histogram_from_host()` calls.

```mojo
var scales = agree_fixed_scales(comm, local_grad, local_hess)   # once per round
builder.set_fixed_scales(scales.grad, scales.hess)              # before staging
...
builder.download_raw()
var words = widen_words(builder.raw_words())
reduce_fixed_words(comm, words, n_features, n_bins)
builder.set_raw_words(narrow_words(words))
var hist = builder.histogram_from_host()
```

**State flow** `set_fixed_scales` must be called **before** `stage_gradients`
for that round, or `stage_gradients` will overwrite the agreed scales with
local ones and the ranks will quantize against different scales, which sums
to silent garbage. The cleanest form is for `stage_gradients` to take
optional scales and skip `_fixed_scale` when they are supplied, which makes
the ordering unrepresentable rather than documented. `set_raw_words` writes
the reduced plane back into `self.host_out` so `histogram_from_host` needs no
change at all. `round_epoch` and `has_gradients` are untouched by both new
setters.

**Errors** `set_fixed_scales` should raise on a non-finite or non-positive
scale. `set_raw_words` should raise on a length other than
`3 * n_features * n_bins`. `check_word_planes` in `distributed_gpu` already
performs exactly that check and can be called instead of restating it.
`check_fixed_point_contract(FIXED_ONE)` should be called once so a change to
`_FIXED_ONE` in this file is caught rather than silently desynchronizing the
two modules.

**Ownership** `histogram_gpu.mojo` belongs to the Apple GPU lane, and Task 20
has already wired launch gates into it.

**Fallback** Without this patch there is no distributed GPU path at all, and
`require_distributed_gpu` refuses it for four other reasons anyway. The patch
is inert on the single-node path, because a builder whose scales were never
set externally behaves exactly as it does today.

**Serialization effect** None on the model. The reduced plane is transient.

**Public API effect** Two or three new methods on `GpuHistogramBuilder`, plus
possibly renaming `_FIXED_ONE` to `FIXED_ONE`. The rename is the only
non-additive part and it can be avoided with an alias.

**Dependency** Patch 2's gate should exist first so a distributed GPU run can
be refused before it reaches a builder.

**Minimal later validation, UNRUN** On one device with a one-rank
`LocalCollective`, the download, widen, reduce, narrow, convert round trip
must produce a `Histogram` **bit-identical** to plain
`histogram_from_host()`, because a one-rank reduction is the identity. That
test needs a GPU but no second process, and it is the cheapest real check the
whole fixed-point path can get. Then, with two ranks over `LocalCollective`
and a globally agreed scale, the summed histogram must equal the CPU
histogram of the concatenated rows to within the fixed-point tolerance
`docs/DISTRIBUTED_STRATEGIES.md` section 4 states.

---

### Patch 7. Round-level scale agreement in the GPU trainer

**Target file / symbol** `src/mojoboost/train_gpu.mojo`, at the point in the
boosting loop where gradients for the round are computed and staged.

**Signature** No signature change if the trainer is given a collective; one
new optional parameter if it is not.

**Call site** Once per boosting round, before the first node of the round.

```mojo
var scales = agree_fixed_scales(comm, local_grad, local_hess)
```

**State flow** One two-element Float64 reduction per round, not per node.
`gpu_round_plan(num_leaves)` states that ratio explicitly so a caller can see
that the scale agreement is amortized over `num_leaves - 1` node reductions
and is not a per-node cost. The result feeds patch 6's `set_fixed_scales`.

**Errors** `agree_fixed_scales` raises on a non-finite gradient sum and on a
gradient list whose shape differs from the hessian list. Both are local
checks on identical shapes, so they raise on every rank or on none.

**Ownership** `train_gpu.mojo` belongs to the GPU trainer lane.

**Fallback** Not applicable. Without round-level agreement there is no
correct distributed GPU path, and this is the one thing that cannot be
deferred or approximated. A local scale per rank is not a degraded version of
a global scale, it is wrong.

**Serialization effect** None.

**Public API effect** None if the trainer already carries a collective.

**Dependency** Patch 6.

**Minimal later validation, UNRUN** Two ranks with deliberately different
local gradient magnitudes agree on one scale, and both `GpuFixedScales`
values compare equal across ranks. Then `check_fixed_point_headroom(rows,
world)` passes for the row counts the test uses.

---

### Patch 8. Bindings

**Target file / symbol** `bindings/distributed_bindings.mojo`, alongside the
existing `distributed_capability` (line 64), and its registration in
`bindings/_mojoboost.mojo` (line 373).

**Signature**

```mojo
def distributed_strategy_info() raises -> PythonObject
def distributed_gpu_status() raises -> PythonObject
```

**Call site** Registered next to the existing three.

```mojo
m.def_function[distributed_strategy_info]("distributed_strategy_info")
m.def_function[distributed_gpu_status]("distributed_gpu_status")
```

**State flow** Both are pure queries and take no arguments.
`distributed_strategy_info` returns a dict keyed by strategy name, each value
a dict of the `StrategyCapabilities` fields plus an `available` bool and a
`reason` string taken from `raise_strategy_unsupported`'s message.
`distributed_gpu_status` returns `available`, `reason` from
`distributed_gpu_unavailable_detail()`, `gates` as the list of open gate
names from `distributed_gpu_gates()` and `gate_name`, and
`device_collective` from `device_collective_name(DEVICE_COLLECTIVE_NONE)`.

**Errors** Neither raises in any build. Both exist in order to say no, which
is the same contract `distributed_capability` states for itself. A caller
must refuse on `available` and quote `reason` rather than inferring
availability from the function existing.

**Ownership** `bindings/` belongs to CONNECT_EVERYTHING Task 14.

**Fallback** If only one function is wanted, fold both records into
`distributed_capability` under new keys `strategies` and `gpu`. The Python
side reads that dict already, so no new entry point is needed at all, at the
cost of a wider record.

**Serialization effect** None on the model. The returned dicts are read by
`_dask_runtime.describe_runtime` and must stay JSON-shaped, meaning strings,
bools, ints, and lists only.

**Public API effect** Two new module-level functions on `_mojoboost`. Python
callers must reach them through `getattr` with a default, the way
`_dask_runtime` already reaches `distributed_runtime_info`, so an older
compiled extension keeps working.

**Dependency** Patch 1 is not required, because bindings can import the
modules directly.

**Minimal later validation, UNRUN** `python -c "import mojoboost;
print(mojoboost._mojoboost.distributed_gpu_status())"` prints
`available False` and a non-empty reason. The five gate names appear.

---

### Patch 9. Dask capability names and runtime record

**Target file / symbol** `python/mojoboost/dask.py`, `CAPABILITIES` (line
201), and `python/mojoboost/_dask_runtime.py`, the
`distributed_runtime_info` contract documented at lines 64 to 83.

**Signature** Three names added to the frozenset.

```python
"feature_parallel",  # features partitioned, only the winning split crosses
"voting_parallel",   # data parallel over a top-k voted feature subset
"data_parallel",     # rows sharded, histograms all-reduced
```

**Call site** A backend declares what it supports by listing names from this
set in its `capabilities`. Nothing in `dask.py` decides for it.

**State flow** `distributed_runtime_info()["capabilities"]` should be built
from `distributed_strategy_info()` in patch 8, so the Python record and the
Mojo gate cannot disagree. A runtime that reports `feature_parallel` while
`require_strategy` refuses it is a bug the two-sided derivation removes.

**Errors** `UnsupportedByBackend` is the existing exception for a requested
capability a backend does not declare, and it is the right one here. A user
asking for `tree_learner="feature"` against a backend that does not declare
`feature_parallel` must get that exception before any worker is contacted.

**Ownership** `python/mojoboost/dask.py` and `_dask_runtime.py` belong to
CONNECT_EVERYTHING Task 15.

**Fallback** Adding names to `CAPABILITIES` alone is inert. No backend
declares them, so nothing changes until a runtime does.

**Serialization effect** `CAPABILITIES` is part of the version 0 backend
protocol. Adding names is backward compatible, because a backend declares a
subset and a client checks membership. **Do not** bump
`BACKEND_PROTOCOL_VERSION` for this.

**Public API effect** Additive. `mojoboost.dask.CAPABILITIES` grows by three
entries.

**Dependency** Patch 8, if the record is to be derived rather than hardcoded.

**Minimal later validation, UNRUN** `pytest -q python/tests` still passes,
and a backend declaring `feature_parallel` with a job requesting it does not
raise `UnsupportedByBackend` while one that does not declare it does.

---

### Patch 10. Documentation rows

**Target file / symbol** `docs/LIGHTGBM_PARITY.md` section 11, and
`docs/distributed.md` section 9.

**Signature** Three rows in the parity table for `tree_learner=feature`,
`tree_learner=voting`, and distributed GPU, each marked not implemented, each
pointing at `docs/DISTRIBUTED_STRATEGIES.md`. One cross-reference line in
`docs/distributed.md` section 9, which already gates distributed GPU behind a
discrete-GPU benchmark and should name the module that now holds the
contract.

**Call site** None.

**State flow** None.

**Errors** None.

**Ownership** `docs/LIGHTGBM_PARITY.md` belongs to the parity lane and is
checked by `tools/check_parity.py` under `pixi run check-parity`. **Adding a
row may change that check's counts.** Do not edit it without the parity
lane, and confirm whether the checker treats an unimplemented row as a
failure or as a tracked gap.

**Fallback** `docs/DISTRIBUTED_STRATEGIES.md` stands alone and is complete
without these rows. The only cost of skipping this patch is that a reader of
the parity table does not learn that the two `tree_learner` values exist as
cores.

**Serialization effect** None.

**Public API effect** None.

**Dependency** None.

**Minimal later validation, UNRUN** `pixi run check-parity` passes.

---

### Patch 11. Test registration

**Target file / symbol** `pixi.toml` line 9, the `test` task, and a new
`tests/parallel/test_distributed_strategies.mojo`.

**Signature** One command appended to the chain.

```
&& mojo run -I src tests/parallel/test_distributed_strategies.mojo
```

**Call site** `pixi run test`.

**State flow** None.

**Errors** None.

**Ownership** `pixi.toml` line 9 is a single line edited by many lanes in
this shared checkout and is the most conflict-prone line in the repository.
Append at the end of the `tests/parallel/` group, never in the middle.

**Fallback** Register under `test-gpu` instead if any assertion needs a
device. Nothing in `distributed_strategies.mojo` does. Only patch 6's and
patch 7's validations need a GPU, and those belong in
`tests/test_gpu_training.mojo`.

**Serialization effect** None.

**Public API effect** None.

**Dependency** The test file has to exist. This lane was instructed not to
write one, so it does not exist.

**Minimal later validation, UNRUN** The whole point of the patch. See the
assertion list below.

---

## What a first test owes, UNRUN

This lane wrote no test. The assertions below are what one should make, in
increasing cost, and none of them has been run.

No collective needed.

1. `parse_strategy` round-trips `strategy_name` for all four codes and
   raises on anything else.
2. `FeaturePartition(13, 4)` gives counts 3, 3, 3, 4 summing to 13, and
   `owner(f)` agrees with `features(r)` for every feature.
3. `encode_candidate` then `decode_candidate` reproduces every `SplitInfo`
   field including the four-word categorical bitset and the exact Float64
   gain, for a found split, a not-found split, and a categorical split.
4. `elect_split` over hand-built candidates picks the highest gain, and on a
   tie picks the lowest feature id, matching `find_best_split`'s rule.
5. `select_top_k` returns ascending feature ids and honors `found`.
6. `elect_voted_features` is deterministic under a vote tie.
7. `strategy_unsupported_mask` returns 0 for data parallel at world 4 with
   `multi_process` False, and sets `UNSUPPORTED_NO_TRANSPORT` only when
   `multi_process` is True and `transport_ready` is False.
8. `narrow_words` raises just outside `Int32` and passes just inside.
9. `check_fixed_point_headroom(2**31 - 1, 64)` raises and a realistic row
   count passes.
10. `fixed_scale_from_total` equals `histogram_gpu._fixed_scale` on the same
    input, which is the mirror that patch 4 or patch 6 would delete.

Four-rank `LocalCollective`, no transport, no device.

11. `allgather_candidates` returns one candidate per rank in ascending rank
    order, and `elect_split_collective` returns the identical `ElectedSplit`
    on every rank.
12. `pack_selected` then `allreduce_selected` then `unpack_selected` gives
    every rank the same histogram, elected slices summed and unelected
    slices exactly zero.
13. `allreduce_votes` counts exactly, at any arrival order.
14. `check_strategy_world` raises the identical message on every rank for
    feature parallel and for voting parallel, and raises before any
    collective a refused peer would have to enter.
15. `agree_strategy` raises when one rank is given a different `top_k`.

Needs a GPU. See patch 6.

16. One-rank reduce round trip is bit-identical to `histogram_from_host`.

Needs two processes and a transport, neither of which exists.

17. Everything else. No claim in either module about multi-process behavior
    has been executed, and the modules say so in their own gate messages.

## Open questions this lane did not decide

1. Whether feature parallel should hold the full dataset on every rank, as
   LightGBM does and as `StrategyCapabilities.needs_every_row_on_every_rank`
   records, or shard rows and broadcast the row assignment. This lane built
   the former because it makes the mode a one-function-per-node seam rather
   than a third growth loop. The latter is a different design and the cores
   here would not serve it unchanged.
2. Whether voting parallel should aggregate gains rather than count votes.
   LightGBM aggregates. `allreduce_votes` counts, which is cheaper and
   coarser, and `voting_is_exact()` returns False either way. Changing it is
   a change to one function and its docstring already says so.
3. Whether a device-resident collective is worth binding at all before the
   discrete-GPU benchmark in `docs/distributed.md` section 9 exists.
   `GPU_SPEEDUP_GATE_MET` is False and the M4 measurement is 0.56x, so a
   distributed GPU path today would distribute a slowdown.
