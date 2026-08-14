# Feature-parallel, voting-parallel, and distributed GPU contracts

Status: design plus strategy cores. Not a shipped feature and not an
operational one. Nothing in this document has been run across two processes,
nothing has been run on two devices, and no parallel speedup, scaling, or
communication measurement is claimed anywhere in it.

Three separate things are described here and they are at three different
stages, so they are labeled individually rather than under one status line:

| capability | what exists | what does not |
|---|---|---|
| feature parallel | the split-election core, the feature partition, the candidate wire record, the agreement and failure protocol (`src/mojoboost/distributed_strategies.mojo`) | a grower that calls them; any run |
| voting parallel | the top-k selection, the vote reduction, the feature election, the packed histogram exchange (same file) | a grower that calls them; any run |
| distributed GPU | the global fixed-point scale agreement, the word staging and its overflow argument, the cost model, the collective seam (`src/mojoboost/distributed_gpu.mojo`) | a device-resident collective; a transport; the single-node GPU speedup that gates the work; any run |

The one mode that does run is data parallel, in `src/mojoboost/distributed.mojo`,
with every rank inside one process. `docs/distributed.md` is its design and is
the document this one extends rather than replaces.

## 0. Why these are cores and not trainers

`grow_tree_distributed` is a second copy of the growth loop, and section 11 of
`docs/distributed.md` is candid about the cost: the two loops drift, and the
distributed one silently misses whatever the single-node one gains. A third
and fourth copy, one per new mode, would multiply that cost by two.

So neither mode here is written as a loop. Each is written as the one function
a grower would call at the point where it already makes a decision:

- feature parallel replaces the per-node `find_best_split` call with
  `elect_split_collective`, which searches this rank's features and returns
  the winner every rank agrees on
- voting parallel replaces the per-node histogram all-reduce with
  `select_top_k`, `allreduce_votes`, `elect_voted_features`, and
  `allreduce_selected` over the elected features only

Everything else in a grower stays exactly what it is. That is the whole
architectural claim of this lane, and it is also why the modes cannot be
turned on yet: a seam is not a driver.

## 1. Feature parallel

### 1.1 Data model

Every rank holds the **whole** binned matrix: every row and every feature.
Features are partitioned only for the purpose of deciding who searches what.

This is LightGBM's own arrangement and it is the thing that makes the mode
cheap. The textbook version of feature parallelism partitions the columns
physically, and then every split has to be followed by a broadcast of the
row-to-child assignment, whose size grows with the row count. Keeping every
row on every rank removes that message entirely: once the winning split is
known, each rank routes its own copy of every row through it and reaches the
identical partition without being told anything.

The cost is stated plainly because it is the reason the mode is narrow:
**nothing about feature parallelism makes a dataset that does not fit on one
machine fit on two.** It parallelizes the histogram build and the split scan,
not the memory.

`FeaturePartition` is contiguous and ascending. Rank `r` owns
`[r * F // W, (r + 1) * F // W)`, the same block rule `ShardPlan.contiguous`
uses for rows. A rank may own zero features when the world is wider than the
feature count, and such a rank still takes part in every collective with a
not-found candidate, exactly as an empty shard contributes a zero histogram
in data-parallel training.

### 1.2 The algorithm

```
per node:
  build a histogram over this rank's own features only      [local]
  search those features with find_best_split                [local]
  all-gather one candidate per rank                         [collective]
  elect the winner from the gathered candidates             [local, global input]
  apply the split to this rank's own copy of the rows       [local]
  leaf values, the frontier scan, the stopping rule         [local]
```

One collective per node, carrying `world_size * CANDIDATE_WORDS` integers.
`CANDIDATE_WORDS` is 16, so a 16-rank world exchanges 256 integers per node
against the `3 * n_features * n_bins` numbers data parallel exchanges: at the
LightGBM defaults of 100 features and 255 bins that is 256 integers against
76,500 numbers.

Leaf values need no message. Every rank holds every row, so the gradient and
hessian totals of a node are computable from any feature's bins of any
histogram the rank built, and every rank computes the same ones.

### 1.3 Why every rank elects the same split

Three properties, in the order they are relied on.

**The exchange is order independent.** `allgather_candidates` writes each
rank's record into that rank's own slot of a world-sized integer vector and
reduces it with `allreduce_max_int`. Every other slot holds zero and every
record word is non-negative, so the maximum is an all-gather. Maximum is
commutative and idempotent, so the outcome depends only on what the ranks
wrote, never on arrival order, on which rank answered first, or on the shape
of the reduction tree. This is a genuinely weaker demand on a transport than
the data-parallel histogram reduction makes: requirement 2 of
`docs/distributed.md` section 5, ascending rank order, is not needed here.

**Gains cross as bits.** A gain travels as the two halves of its IEEE-754 bit
pattern, so every rank compares identical numbers rather than two roundings
of the same number. Gains are validated non-negative and finite before they
are encoded, and the bit pattern of a non-negative finite double is monotone
in its value, so the ordering is the same on the bits as on the doubles.

**The scan order reproduces the single-node one.** `elect_split` walks
candidates in ascending rank order and replaces the running best only on a
strict gain improvement. Because the partition gives rank `r` a contiguous
ascending block, ascending rank order is ascending feature order, which is
exactly the order `find_best_split` scans within one rank and exactly the rule
it applies. **Given identical histograms, feature-parallel election therefore
selects the identical split a single-node scan would, ties included.**

That last property is what a round-robin or hashed feature partition would
destroy, which is why `FeaturePartition` implements one scheme and offers no
other. There is nothing to gain from another: every feature costs the same
`n_bins` cells to scan, so contiguous blocks are already balanced.

### 1.4 The candidate record

Sixteen non-negative integers. Non-negative because the all-gather is a
maximum over slots the other ranks leave at zero, and a negative word would
reduce to another rank's zero instead of to itself.

| words | field |
|---|---|
| 0 | marker, always 1 for a rank that wrote |
| 1 | found |
| 2 | `feature + 1` |
| 3 | `bin + 1`, zero for a categorical split |
| 4 | flags: bit 0 categorical, bit 1 `default_left` |
| 5, 6 | the gain's IEEE-754 bits, low and high halves |
| 7 | `node + 1` |
| 8 to 15 | the categorical bitset, four 64-bit words as eight halves |

Every 64-bit quantity is stored as two 32-bit halves so no word can reach the
sign bit of an `Int`. The node id rides along so that two ranks electing for
different nodes are detected rather than silently merged, which is the
collective-reordering failure of `docs/distributed.md` section 5 requirement 5
observed from above the transport.

### 1.5 What feature parallel supports that data parallel does not

`strategy_capabilities` is the machine-readable form of this table.

| feature | data parallel | feature parallel | why |
|---|---|---|---|
| categorical splits | refused | supported | `search_owned_features` forwards to the one `find_best_split` the single-node and GPU growers use, so the category partition search is the same code |
| monotone constraints | refused | supported | same forward; the output bounds are a function of ancestor leaf values, which are identical on every rank |
| missing bins | refused | supported | same forward; missing routing is per feature and entirely local |
| bagging | refused | supported | the row draw is over global indices and every rank holds every row, so every rank reproduces the draw exactly. Under data parallel a shard-local bag has to be a deterministic projection of a global draw, which `distributed.mojo` refuses rather than approximates |
| feature subsampling | refused | supported | the draw depends only on the seed and the tree or node index, and `intersect_ascending` narrows it to this rank's features |
| ranking query groups | constrained | unconstrained | every row is on every rank, so no group is ever straddled and `check_group_alignment` has nothing to refuse |
| dataset larger than one machine | supported | **not supported** | the mode does not partition rows |

The first five rows are not cleverness. They are the direct consequence of
these cores forwarding to the existing split search instead of reimplementing
it, which is the difference between a seam and a second growth loop.

`extra_trees`, `max_delta_step`, and `path_smooth` are refused:
`tree.grow_tree` applies them and no feature-parallel grower exists to be
trusted with them. `search_owned_features` checks
`params.extra.needs_grower_support()` and says so.

## 2. Voting parallel

### 2.1 The algorithm

Voting parallel is data parallel with a filter, so it inherits the row
partition and every constraint that comes with it.

```
per node:
  build a local histogram over this rank's rows, all features   [local]
  rank this rank's features by local gain, keep the top k       [local]
  reduce the vote counts over features                          [collective]
  elect the globally most-voted features                        [local, global input]
  pack, reduce, and unpack only those features' cells           [collective]
  search the elected features on the reduced histogram          [local, global input]
```

Two collectives per node instead of one, and the second one carries
`n_selected * n_bins` cells instead of `n_features * n_bins`. The vote
reduction is `n_features` integers, which is negligible against either.

### 2.2 Voting parallel is not exact, and it is not LightGBM's vote

Both statements matter and they are different statements.

**Not exact.** A feature that no rank ranks highly locally can still be the
global winner. When that happens the elected set does not contain it and the
tree takes a different split from the one data parallel would take. This is
the mode, not a defect. `voting_is_exact()` returns False in one place so no
caller has to decide for itself, and there is deliberately no tolerance test
proposed anywhere in this document that would paper over it.

**Not LightGBM's.** LightGBM's voting aggregates the local gains as well as
the counts and runs a second local pass over the merged candidates. The core
here counts votes only and breaks ties by ascending feature id. It is a
different selection rule and therefore a different model, and a run of this
against LightGBM's voting mode should be expected to disagree. Matching
LightGBM's rule exactly is possible and is not attempted, because a rule that
is deterministic and stated beats a rule that is approximately LightGBM's.

The one LightGBM number that is kept is `top_k = 20`, the default, so a
configuration written against LightGBM means the same count here.

## 3. Failure semantics and rank capabilities

The two-phase discipline from `docs/distributed.md` section 7 is unchanged: a
rank records a status rather than raising, the statuses reduce with a maximum,
and every rank raises the same error naming the lowest failing rank. A run
where one rank raises while the others block in a collective it will never
call is the failure that discipline exists to prevent, and nothing in these
cores is allowed to break it.

`check_strategy_world` is the single entry point and its order is load
bearing:

1. `require_strategy`, whose inputs are the strategy, the world size, whether
   the world spans processes, and whether the build has a transport. All four
   are identical on every rank, so it raises everywhere or nowhere, and it is
   ahead of every collective so a refused rank never strands a peer.
2. `agree_status` over `strategy_statuses`, which reports a rank whose own
   partition or feature count is wrong.
3. `agree_strategy`, one reduction over the strategy code, the feature count,
   the world size, `top_k`, and the partition digest, which reports ranks that
   were configured differently from each other.

Three failures specific to these modes are detected in the election itself,
and all three are pure functions of the gathered vector, so every rank raises
together:

- **a rank that never wrote its slot.** The marker word stays zero, so a
  missing contributor is an error rather than an absent candidate. A rank with
  nothing to say writes a not-found split, which is present and not found.
- **a rank answering for a different node.** The node id in the record is
  compared against the election's, so a collective that overtook its
  predecessor is caught above the transport as well as inside it.
- **a rank proposing a split on a feature it does not own.** This is what
  makes the partition load bearing rather than advisory: without the check, a
  rank searching outside its block would double-count one feature and leave
  another unsearched, and the tree would still look plausible.

Worker loss, cancellation, deadlines, and checkpoint boundaries are the
transport's, unchanged, in `src/mojoboost/distributed_transport.mojo`. Nothing
here re-implements them and nothing here weakens them.

## 4. Distributed GPU

### 4.1 The gate, first

`docs/distributed.md` section 9 sets it: distributed GPU work waits for a
benchmark in which single-node GPU training beats single-node CPU training on
a discrete GPU. That benchmark does not exist. The only end-to-end measurement
in this repository is an Apple M4, where GPU training loses at every size
tested. `GPU_SPEEDUP_GATE_MET` is the one name that records this, and it is
evidence rather than code: it flips when a benchmark says so, not when a
feature lands.

`distributed_gpu_gates()` reports five unmet gates and
`require_distributed_gpu()` raises naming all of them. They are tracked
separately because they are independent: a socket landing does not make the
GPU faster, and a fast GPU does not write a driver.

### 4.2 The one real result: the scale has to be global

The single-node GPU histogram accumulates in fixed point. Gradients and
hessians are scaled by `2^30 / sum|values|`, rounded to `Int32`, and summed
with integer atomics, which is what makes the GPU histogram bit-deterministic
where a float atomic would not be. The scale is computed on the host from the
magnitude sum of the values being uploaded.

That has a consequence for distributing it which is easy to get wrong:

> **Two ranks that each compute their own scale produce integer words that
> cannot be added.** Word 7 on rank 0 and word 7 on rank 1 are quantities in
> different units. Summing them is meaningless, and nothing downstream would
> notice: the result is a plausible histogram of the wrong thing.

So the scale itself has to be global. `agree_fixed_scales` reduces the
magnitude sums of the gradients and the hessians, two `Float64` elements, once
per boosting round rather than once per node, and derives the scale from the
reduced total with exactly the arithmetic `_fixed_scale` uses on one node,
including the `Float32` narrowing that makes the host inverse the exact
inverse of what the device multiplied by.

Three consequences follow, and together they are the argument for exchanging
fixed-point words rather than downloading `Float64` histograms and reducing
those:

1. **The reduction is exact.** Integer addition of the scaled words has no
   rounding at all, so nothing is lost relative to the single-node GPU
   histogram, which had already quantized.
2. **The reduction is order independent.** Integer addition is associative, so
   requirement 2 of `docs/distributed.md` section 5, ascending rank order, is
   not needed on this path. The exchange is bit-identical under any reduction
   order or topology, which is a strictly weaker demand on a future transport
   than the CPU path makes.
3. **The words still fit.** With a global scale, the scaled magnitudes of
   every row on every rank sum to at most `2^30` in total, so the sum of the
   reduced words is bounded by `2^30` regardless of world size. Rounding each
   row to the nearest integer adds at most a half per row, so the realized
   bound is `2^30 + total_rows / 2`, which stays inside `Int32.MAX` for every
   row count the kernels accept. The bound is global, not per rank, which is
   exactly why it survives sharding: splitting the same rows over more ranks
   increases neither term. `check_fixed_point_headroom` states it and
   `narrow_words` verifies it on the reduced buffer rather than assuming it.

The cost is that the one `Float64` reduction of the magnitude sums does need
bit-identical delivery, requirement 1. Two ranks holding different roundings
of the magnitude sum would derive different scales and be back where they
started.

**What this does not promise.** Distributed GPU training so quantized is not
bit-identical to single-node GPU training on arbitrary data, because the
global magnitude sum is a regrouped floating point sum and the scale is
derived from it. It is bit-identical wherever that sum is exactly
representable, which is the same conditional claim `docs/distributed.md`
section 6 makes for the CPU path, for the same reason.

### 4.3 CPU staging against device collectives

The exchange sits between two calls the single-node path already makes:
`GpuHistogramBuilder.download_raw`, which copies the fixed-point planes into
pinned host memory and synchronizes, and `histogram_from_host`, which converts
them to `Float64` by dividing by the scales.

```
enqueue_leaf -> download_raw -> [widen, reduce, narrow] -> histogram_from_host
```

Staged that way, the exchange adds one upload and one reduction to a node that
was already being read back, and it needs no device API at all: every function
in `distributed_gpu.mojo` is host arithmetic over lists, which is why the file
compiles on the CPU-only Linux runners CI actually uses.

A device-resident collective would remove the download and the upload, and
would be the right answer for a device whose histograms never needed to reach
the host. Nothing in this repository implements one. `DeviceCollective` is
declared as the seam, deliberately drawn at **host-visible integer words**
rather than at a device buffer, and `UnavailableDeviceCollective` is its one
implementation: it raises with the full gate list rather than returning
plausible zeros.

A device-resident adapter cannot conform to that trait as written, and saying
so is more useful than a signature invented for it. Such an adapter takes a
device buffer and a stream, and it has to answer four questions this seam does
not ask:

1. who owns the buffer during the reduction
2. whether the caller synchronizes before handing it over, or the adapter does
3. which queue the reduction runs on relative to the kernel that filled the
   buffer
4. what a failure leaves behind on the device

Those are answers a library supplies, and binding to NCCL, RCCL, or oneCCL
here would mean shipping untested bindings to a library this build does not
have, under a contract that is otherwise fully specified. The boundary is
drawn exactly where that adapter lands, which is the same decision
`docs/DISTRIBUTED_TRANSPORT.md` section 7 makes for the socket.

### 4.4 Cost

`gpu_exchange_plan` computes it rather than asserting it.

| quantity | value at `F` features, `B` bins |
|---|---|
| cells | `F * B` |
| words | `3 * F * B` |
| reductions per node | 1, against the `Float64` path's 3 |
| device downloads per node | 1, which the single-node path already does |
| device uploads per node | 1, which it does not |
| host syncs per node | 1, which the single-node path already does |
| staged payload bytes | `3 * F * B * 8` |
| native payload bytes | `3 * F * B * 4` |
| `Float64` path payload bytes | `3 * F * B * 8` |

The staged payload is **the same size** as the `Float64` path, not smaller.
`Collective` reduces `List[Int]`, so each 32-bit word is widened to 64 bits
and fixed point gives back exactly what it saved. `staged_saving_ratio`
returns 1.0 and says so. The halving is real and unclaimed: it needs a 32-bit
reduction op on the transport, and `native_saving_ratio` returns the 2.0 that
would then be available.

What the staged path does save today is round trips, one per node against
three, because gradients, hessians, and counts are all integers here and there
is no exactness argument for keeping counts in a separate buffer the way
`allreduce_histogram` does.

### 4.5 Which modes a GPU exchange is defined for

Data parallel only, and `check_gpu_strategy` refuses the rest rather than
leaving the combination undefined.

- **feature parallel on GPU** is refused. The device builder could express a
  feature partition through `set_features`, but the mode's saving is in the
  histogram build while its cost is one full copy of the dataset per rank,
  and device memory is the resource a GPU has least of.
- **voting parallel on GPU** is refused. Its per-node feature set changes with
  the vote, and a device builder that reuploads a feature table per node is a
  different design from the one that exists.

## 5. Unsupported states, enumerated

Every one of these is an error with a message, not a silent difference.

| state | raised by |
|---|---|
| feature parallel selected | `require_strategy`, always, no driver |
| voting parallel selected | `require_strategy`, always, no driver |
| any mode over a multi-process world | `require_strategy`, because `transport_available()` is False |
| a parallel mode at world size 1 | `require_strategy` |
| the serial mode at world size above 1 | `require_strategy` |
| an unknown `tree_learner` name | `parse_strategy` |
| `extra_trees`, `max_delta_step`, `path_smooth` under feature parallel | `search_owned_features` |
| a candidate with a negative, infinite, or NaN gain | `encode_candidate` |
| a rank that contributed no candidate | `elect_split` |
| a rank answering for another node | `elect_split` |
| a rank splitting on a feature it does not own | `elect_split` |
| ranks configured with different strategies, feature counts, or `top_k` | `agree_strategy` |
| distributed GPU training, for any reason | `require_distributed_gpu` |
| a device-resident collective requested | `resolve_device_collective` |
| feature or voting parallel on the GPU | `check_gpu_strategy` |
| a reduced fixed-point word outside `Int32` | `narrow_words` |
| more global rows than the fixed-point headroom covers | `check_fixed_point_headroom` |
| a word buffer that is not three planes of one grid | `check_word_planes` |

Note what is deliberately absent: there is no code path anywhere in these two
modules that silently downgrades a mode, approximates a reduction, or falls
back from a device path to a host path without saying so.

## 6. What each mode still needs

### 6.1 Feature parallel

A grower. Concretely, the loop in `tree.grow_tree` with three changes:

1. build each node's histogram over `partition.features(rank)` only, which
   `build_histogram(features=...)` already accepts
2. replace the per-node `find_best_split` call with `search_owned_features`
   followed by `elect_split_collective`
3. leave everything else, the frontier, the subtraction trick, the leaf math,
   the row routing, exactly as it is

That grower belongs behind the histogram-source refactor
`docs/distributed.md` section 11 argues for, not in a third copy of the loop.
Doing it as a fourth copy would be the mistake this lane exists to avoid.

### 6.2 Voting parallel

A driver in the data-parallel grower, gated by a strategy code, that replaces
the one `allreduce_histogram` call per node with the four-call sequence in
section 2.1. It is a smaller change than feature parallel because it changes
what is reduced rather than who decides.

### 6.3 Distributed GPU

In order, and the order is not negotiable:

1. the discrete-GPU benchmark that closes `GPU_SPEEDUP_GATE_MET`
2. a socket `ByteEndpoint`, which `docs/DISTRIBUTED_TRANSPORT.md` section 7
   specifies, and its hermetic two-process test
3. the staged exchange wired between `download_raw` and `histogram_from_host`,
   with the builder taking the agreed global scales instead of computing local
   ones
4. only then, if the download and upload prove to dominate, a device-resident
   adapter behind `DeviceCollective`

## 7. Validation, none of which has happened

Every command below is **UNRUN**. Nothing in this lane has been compiled,
tested, benchmarked, or executed, and no claim in this document rests on a
run.

Smallest useful checks, in the order they should be attempted:

```
# UNRUN: does the strategies module compile at all
pixi run mojo build -I src src/mojoboost/distributed_strategies.mojo -o /dev/null

# UNRUN: does the GPU contract module compile on a CPU-only machine
pixi run mojo build -I src src/mojoboost/distributed_gpu.mojo -o /dev/null

# UNRUN: the focused test this lane did not write, once someone owns tests
pixi run mojo run -I src tests/parallel/test_distributed_strategies.mojo
```

The assertions a first test owes, none of which exist:

- an encode and decode round trip preserves a numerical split, a categorical
  split with a non-trivial bitset, and a not-found split
- a candidate exchange over `LocalCollective` at world sizes 1, 2, 3, and 5
  elects the same split a single-node `find_best_split` over the same
  histogram elects, including a deliberate two-feature tie
- a rank proposing a split on a feature it does not own is refused, and the
  message names the rank and the feature
- a rank answering for another node is refused
- a world wider than the feature count elects correctly, with the
  zero-feature ranks contributing not-found candidates
- `select_top_k` and `elect_voted_features` are exact against a hand-computed
  ranking, ties included
- `pack_selected` and `unpack_selected` round trip, and unpacking leaves every
  unelected feature at zero
- `fixed_scale_from_total` agrees with `histogram_gpu._fixed_scale` on the
  same input, which is the one assertion that keeps the duplicated constant
  honest
- `narrow_words` refuses a word outside `Int32` rather than truncating it
- `require_strategy` refuses feature and voting parallel, and does **not**
  refuse a `LocalCollective` world of four ranks, which is the regression that
  would break existing data-parallel training

## 8. Differences from LightGBM

- LightGBM selects a mode with `tree_learner`. mojoboost has no such
  parameter, and adding one is a decision for whoever owns the parameter
  surface. `parse_strategy` accepts LightGBM's spellings so that the mapping
  exists when it is wanted.
- LightGBM's voting aggregates local gains as well as counts and runs a second
  local pass. This counts votes and breaks ties by ascending feature id, which
  is a different rule and therefore a different model. Section 2.2.
- LightGBM's feature-parallel mode also keeps the full dataset on every
  machine, so the arrangement here matches it. What differs is that LightGBM
  ships it and this does not.
- LightGBM has no fixed-point GPU histogram, so the global-scale problem in
  section 4.2 has no LightGBM counterpart to match. It is a consequence of
  this repository's own determinism choice, which is documented in
  `histogram_gpu.mojo` and in `docs/GPU_VALIDATION.md`.
