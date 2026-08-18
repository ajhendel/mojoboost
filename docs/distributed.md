# Distributed training design

Status: design plus CPU prototype. Not a shipped feature. The prototype
runs every rank inside one process, so nothing here has been exercised over
a network, and no distributed performance number is claimed anywhere in this
document.

This document covers data-parallel training: every rank holds a horizontal
slice of the rows and a full copy of the growing model. It is the first of
the three classic parallel modes in the LightGBM family, and the only one
prototyped so far.

## 1. Scope

In scope for the prototype:

- horizontal (row) partitioning of the training data
- local histogram construction per rank
- all-reduce of histogram statistics
- globally consistent split selection
- a tree structure that is identical on every rank at every step
- deterministic failure behavior across ranks
- a transport-agnostic collective contract that tree logic depends on and
  nothing else

Explicitly out of scope for the prototype, with reasons in section 9:

- an actual network transport (MPI, gRPC, sockets)
- distributed binning
- feature-parallel and voting-parallel modes
- bagging and feature subsampling
- the quantile and L1 objectives
- early stopping and multiclass
- distributed GPU training

## 2. Why data parallel first

Three parallel modes are possible over the same histogram-based grower.

**Data parallel** partitions rows. Each rank builds a histogram over its own
rows for every feature, and one all-reduce per node produces the global
histogram. Communication per node is `n_features * n_bins` histogram cells,
independent of the row count. It is the right mode when rows are the thing
that does not fit, which is the common case.

**Feature parallel** partitions features. Each rank finds the best split
among its own features and the ranks exchange only the winning split, which
is a few bytes. But every rank then needs the row partition of the chosen
split, so each split broadcasts a row-to-child assignment whose size grows
with the row count. It only pays off when features vastly outnumber rows.

**Voting parallel** is data parallel with a filter: each rank votes for its
top-k local features, and only the globally top features have their full
histograms reduced. It reduces communication by roughly `n_features / k` at
the cost of exactness, since a feature that no rank ranks highly locally can
still be the global winner. It is a refinement to apply after data parallel
works, not a substitute for it.

Data parallel is also the only one of the three whose result can be made
exactly equal to single-node training, which makes it testable against the
existing trainer. That is the deciding argument for doing it first.

## 3. Data model

A rank owns a `DataShard`: a `BinnedMatrix` of its own rows, the targets for
those rows, and optionally their sample weights. A rank never sees another
rank's rows.

**Partitioning must be order preserving.** Rank `r` owns a contiguous block
of the global row order, and concatenating shards in ascending rank order
reproduces the original dataset row for row. This is not cosmetic. The
floating point argument in section 6 depends on the global reduction visiting
row contributions in the same order the single-node trainer does.

**Binning must be global, and the prototype does not do it.** Bin edges are
quantiles of the training data. Fitting them per shard would give each rank a
different meaning for bin 7 of feature 3, and every histogram cell reduced
across ranks would be summing unrelated quantities. The prototype therefore
takes an already-fitted `BinMapper` shared by every shard, which in practice
means binning was fit on one node before partitioning. Section 9 covers what
a real distributed binning step has to do.

## 4. The algorithm

Every rank runs the same program. Communication happens only at the marked
points, and between two communication points a rank touches only its own
rows.

```
train_distributed:
  validate locally, agree on status across ranks        [collective]
  base score: reduce (sum w*y, sum w) across ranks      [collective]
  for each boosting round:
    compute local gradients and hessians on local rows  [local]
    tree = grow_tree_distributed(...)                   [collective inside]
    if tree is a single near-zero leaf: stop            [local, but global
                                                         input, so unanimous]
    update local raw scores from the tree               [local]

grow_tree_distributed:
  root histogram:
    build a local histogram over local rows             [local]
    all-reduce it                                       [collective]
  root split = find_best_split(global root histogram)   [local, global input]
  while leaves < num_leaves:
    pick the frontier leaf with the highest gain        [local, global input]
    partition local rows by the chosen split            [local]
    decide which child is smaller from the global
      parent histogram counts                           [local, global input]
    build a local histogram for the smaller child       [local]
    all-reduce it                                       [collective]
    derive the sibling by subtraction from the global
      parent histogram                                  [local, global input]
    leaf values and child splits from global histograms [local, global input]
```

Three properties make this work.

**Every decision is a pure function of global data.** Split selection, the
frontier scan, the smaller-child choice, leaf values, and the stopping rule
all read only all-reduced histograms and the tree built from them. No rank
ever consults a local quantity when making a decision, so no rank can reach a
different conclusion. There is no leader, and no split needs to be broadcast.

**The smaller-child choice costs nothing.** Which child to build directly and
which to derive by subtraction depends on the child row counts, which are
already present exactly in the global parent histogram: sum the counts of the
split feature's bins up to and including the split bin. Counts are integers,
so this is exact, and every rank computes the same answer without an extra
message. This matters because if two ranks disagreed about which child to
build, their all-reduces would be summing different quantities.

**A rank with no rows still participates.** An empty shard contributes a zero
histogram to every all-reduce. It must not skip the call, because a collective
that one rank skips deadlocks the rest. The prototype's empty-shard test
exists to pin that.

One all-reduce per tree node is the whole communication schedule: one for the
root and one for each split, so `num_leaves` all-reduces per tree.

## 5. The collective contract

Tree logic depends on one trait, `Collective` in `collective.mojo`, and on
nothing else about how ranks talk to each other. No socket, no rank id, and
no message ever appears in `distributed.mojo`'s split or growth code except
through this trait.

```mojo
trait Collective:
    def world_size(self) -> Int
    def rank(self) -> Int
    def n_local_ranks(self) -> Int
    def local_rank(self, index: Int) -> Int
    def allreduce_sum_f64(mut self, mut buf: List[Float64]) raises
    def allreduce_sum_int(mut self, mut buf: List[Int]) raises
    def allreduce_max_int(mut self, mut buf: List[Int]) raises
    def barrier(mut self) raises
```

`n_local_ranks` and `local_rank` are what let one implementation host several
ranks in one process. A real transport returns `1` and `rank()`; the in-process
`LocalCollective` returns the whole world. The growth code loops over local
ranks and accumulates their contributions in ascending rank order before
calling the all-reduce, so the loop degenerates to a single iteration under a
real transport and the code is the same in both cases.

A conforming transport must satisfy all of the following. These are
requirements, not preferences: the correctness arguments in sections 4 and 6
fail without them.

1. **Bit-identical delivery.** Every rank receives exactly the same bytes from
   an all-reduce. Reduction schemes that leave different ranks holding
   different roundings of the same sum, which includes the usual ring
   all-reduce with per-rank rotation, are not acceptable. Reduce to a root in
   rank order and broadcast, or use a fixed reduction tree with a fixed
   traversal, or reduce in ascending rank order at every rank.
2. **Ascending rank order.** The sum is accumulated in rank order:
   `((c_0 + c_1) + c_2) + ...`. This is what makes the in-process prototype
   numerically equal to a real cluster of the same world size, and it is what
   makes an exactly representable dataset give the same answer distributed as
   on one node.
3. **Collective and ordered.** Every rank calls every collective, in the same
   order, the same number of times. The algorithm's call sequence depends only
   on global data, so this holds by construction as long as no rank takes an
   early exit that another rank does not.
4. **Fail-stop, not partial.** An all-reduce either delivers the complete
   result to every rank or fails on every rank. A transport that can deliver
   to some ranks and fail on others breaks the agreement protocol in section 7
   and cannot be used.
5. **No reordering across calls.** Collectives complete in program order. The
   histogram of one node must not overtake the histogram of the next.

Nothing in the contract mentions histograms, trees, or gradients. The three
buffer reductions plus a barrier are the entire surface a transport has to
implement, and section 10 lists the tests it has to pass.

The prototype ships one implementation, `LocalCollective(world_size)`, which
hosts every rank in the calling process. Its all-reduce is the identity: the
growth loop has already accumulated all `world_size` contributions in
ascending rank order, and there are no other processes to combine with. That
is the correct implementation of the contract for a one-process group, not a
stub, and by requirement 2 it produces the same values a conforming
multi-process transport would.

`LocalCollective` also counts calls and reduced elements, which is how the
communication cost in section 8 is tested without a network.

## 6. Determinism

Two different claims, and they need to be kept apart.

**Reproducibility, which holds unconditionally.** For a fixed dataset, fixed
partition, and fixed world size, distributed training produces a bit-identical
model on every run and on every rank. Nothing in the algorithm depends on
arrival order, timing, thread counts, or which rank finished first. This is
the property that matters operationally, and it holds for any data.

**Equality with single-node training, which is conditional.** The single-node
histogram adds row contributions one at a time in row order. The distributed
one adds them in the same order but regroups them at shard boundaries, and
floating point addition is not associative, so the two can differ in the last
bits. Where the sums are exactly representable, which covers small integers
and dyadic rationals, regrouping changes nothing and the two are bit-identical
at every world size.

Two consequences worth stating separately. A single tree grown from given
gradients is bit-identical at every world size whenever those gradients sum
exactly, which is a property the caller controls. Multi-round training is not,
because leaf values are divisions: round one is exact if the targets and the
base score are, but from round two the gradients are arbitrary doubles and the
regrouping becomes visible again. Distributed training therefore agrees with
single-node training to within accumulated rounding, and the test suite asserts
bit-identity where it is guaranteed and a tolerance where it is not, rather
than asserting a tolerance everywhere and calling it equivalence.

World size 1 is bit-identical unconditionally, on any data, because a
one-shard reduction adds a histogram to zero and changes nothing.

This is the same trade LightGBM makes, and the same one the GPU backend makes
for a different reason. It is worth stating plainly rather than quietly
shipping a `1e-9` tolerance: distributed training is reproducible, and it is
not bit-identical to single-node training on arbitrary data.

Counts are integers and reduce exactly, so every count-derived decision, which
includes `min_data_in_leaf` and the smaller-child choice, is exact at any world
size on any data.

Split selection ties are broken by scan order inside `find_best_split`, which
scans features ascending and bins ascending and keeps a candidate only on a
strict improvement. Since every rank scans the same global histogram, ties
resolve identically without a tie-break protocol.

## 7. Failure behavior

The requirement is that a failure produces the same outcome on every rank. A
run where rank 3 raises and ranks 0, 1, and 2 hang inside an all-reduce that
rank 3 will never call is the failure mode this design exists to prevent.

Validation is therefore two phase. Each rank checks its own shard locally and
records a status code instead of raising. All ranks then reduce the status
vector with `allreduce_max_int`, so every rank learns every rank's status.
If any status is nonzero, every rank raises the same error naming the
lowest-numbered failing rank and its reason. Ranks that were themselves fine
raise too, with the same message.

```
distributed training failed on rank 2: shard target length does not equal
shard row count
```

The status vector has one slot per rank, so a process hosting several ranks
fills several slots, and slots for ranks that are fine stay zero. Max is the
right reduction because status codes are nonnegative and zero means fine.

Cross-rank agreement checks ride the same mechanism. Feature count and bin
count must match across shards, which no rank can verify alone, so the ranks
reduce `[n_features, -n_features, n_bins, -n_bins]` with max and compare: if
the max of a value and the negated max of its negation disagree, the ranks
disagree, and all of them raise.

What this design does not do is recover. There is no re-execution, no
re-partition around a lost rank, and no checkpoint. A failed run fails on
every rank with the same message, and that is the entire contract. Fault
tolerance is a separate project that would need checkpointing of the model
and the raw scores, and it should not be attempted before a real transport
exists.

## 8. Communication cost

Per tree node, one histogram: `n_features * n_bins` cells of gradient sum,
hessian sum, and count.

```
bytes per node = n_features * n_bins * (8 + 8 + 8)
bytes per tree = num_leaves * that
```

At the LightGBM defaults, 100 features, 255 bins, 31 leaves, that is 612 KB
per node and 18.5 MB per tree, or 1.85 GB over 100 rounds, per rank. It does
not depend on the row count, which is exactly why the mode scales with rows.

The prototype issues three reductions per node, one per statistic array,
because keeping gradients, hessians, and counts in their own typed buffers
makes the exactness argument for counts obvious. A production transport should
pack gradient and hessian into one message and send counts as a second, or
pack all three into one buffer of doubles, since counts below 2^53 are exact
in a double. That is a factor of three fewer round trips at identical
arithmetic. It is deliberately not done in the prototype, where clarity is
worth more than round trips that do not exist yet.

The obvious next reduction is what LightGBM does: reduce-scatter the histogram
so each rank owns a feature range, find the local best split in that range,
and all-gather the `world_size` candidate splits. Communication drops from
`n_features * n_bins` per rank to roughly `n_features * n_bins / world_size`
plus a few bytes per rank. The prototype does not do this because a plain
all-reduce keeps every rank holding the complete global histogram, which is
what makes the equivalence test against single-node training possible. The
reduce-scatter version should be built only once the all-reduce version is
proven correct, and it should be tested against it.

## 9. What the prototype does not support, and why

**A network transport.** The collective contract exists so that adding one
does not touch tree logic, and the in-process implementation exists so the
algorithm can be validated first. Building the algorithm and the transport at
the same time means debugging both at once.

**Distributed binning.** Bin edges are global quantiles, so fitting them
requires either gathering a sample of every shard to one rank, or a
distributed quantile sketch such as the GK or t-digest algorithms LightGBM's
sampling approach approximates. Both are real work with their own determinism
requirements, since the edges must be bit-identical on every rank or the
shards stop agreeing about what a bin means. The prototype takes a shared
pre-fit `BinMapper` and refuses to guess.

**The quantile and L1 objectives.** Both replace leaf values with a percentile
of the residuals of the rows in the leaf, following LightGBM's
`RenewTreeOutput`. A percentile is not a sum, so it does not all-reduce. Doing
it exactly needs a distributed selection algorithm over the leaf's residuals,
and doing it approximately would silently make distributed training disagree
with single-node training in a way no tolerance test would catch. The
prototype raises `distributed training does not support the quantile or L1
objective` instead. Huber needs no renewal and is supported.

**Bagging and feature subsampling.** Both draw from a seeded RNG. Feature
subsampling would work unchanged, since the draw depends only on the tree
index and the seed and would be identical on every rank, but the single-node
grower's subsampling was in flight while this prototype was written and
matching a moving target invites silent divergence. Bagging is harder: the row
sample is drawn over global row indices, so a shard-local bag has to be a
deterministic projection of the global draw. Both are rejected rather than
approximated.

**Depth caps, monotone constraints, categorical features, and missing bins.**
All four are single-node grower features that landed while this prototype was
being written. None of them needs new communication: a depth cap and the
monotone output bounds are structural and identical on every rank, and
categorical and missing-value handling are per-node split searches over an
already-reduced histogram. They are refused here only because reimplementing
them in a second growth loop would mean matching four moving targets at once.
Interaction constraints, which are the same shape of problem, are supported,
because the allow mask is a pure function of the branch and the existing
public helpers compute it.

**Everything added to `TreeParams` after this was written.** The rejection
list is enumerated, not derived, which is the honest weakness of running a
second growth loop. A parameter added later will not be in the list, so it
will be ignored instead of refused until someone adds it. The world-size-1
equivalence test catches a change in default behavior but not the arrival of
a new opt-in parameter. That is the strongest guarantee a separate loop can
give, and it is the argument for the refactor in section 11.

**Early stopping and multiclass.** Early stopping needs the validation loss
reduced across ranks each round, which is one more collective and no new
ideas. Multiclass grows one tree per class per round, which is the same
algorithm run `n_classes` times. Both are straightforward extensions and
neither is prototyped.

**Distributed GPU training.** Not started, and still gated, though the gate's
original wording no longer describes the evidence. Single-node GPU training
exists (`train_gpu.mojo`) and on an Apple M4 it now beats the CPU trainer at
the large end: 3.58s against 6.98s at 1,000,000 rows by 50 features, and
15.30s against 25.47s on multiclass. It loses below about a million rows
(1.89 against 1.66 at 250,000), and no NVIDIA or AMD device has executed this
code at all. Those figures are `bench/results/profile_2026-08-15/RESULTS.md`,
one run on one Apple M4 laptop, taken before the 2026-08-16 changes that
retired the figures older than them and not re-measured since. The gate that remains is therefore narrower and sharper than
"the GPU has never won": it is a **discrete-GPU** measurement, on hardware
this project has never run on, at a shape a distributed job would actually
use. A network layer under a backend measured only on one integrated Apple
part would be extrapolating across both the interconnect and the device.
Until that exists, distributed work stays on the CPU.

## 10. Testing

The prototype's tests are in `tests/test_distributed.mojo` and run in the
default `pixi run test` suite, since everything is in-process and CPU only.

Equivalence, which is the whole point:

- world size 1 reproduces `grow_tree` node for node, bit for bit, on random
  gradients and hessians
- world sizes 1, 2, 3, 4, 5, 7, and 16 all produce the same tree on exactly
  representable gradients, including one that does not divide the row count
  evenly and one with more ranks than rows
- `train_distributed` at world size 1 reproduces `train` tree for tree for
  squared error and binary logistic
- at larger world sizes it agrees with `train` to 1e-9 on every prediction,
  for squared error, weighted squared error, poisson, and huber. Past round
  one the gradients are arbitrary doubles, so this is a tolerance rather
  than bit-identity, which is exactly the section 6 claim
- a model trained distributed round-trips through `save_model` and
  `load_model` unchanged, since the model structure is the ordinary one

Validation and determinism:

- empty shards change nothing and do not deadlock, forced by giving the world
  more ranks than there are rows
- repeating a run gives a bit-identical tree
- gradients of the wrong length on ranks 1 and 2 fail on every rank, with the
  lowest failing rank and the reason in the message
- a simulated remote rank's failure stops the local ranks that were fine
- ranks given different feature counts detect which value they disagree about
- quantile and L1 are refused, and the message says why
- feature subsampling, depth caps, and monotone constraints are refused
- weights that are all zero within one shard are fine as long as some other
  shard carries positive weight, and all-zero weights everywhere are refused
- the all-reduce count and the reduced element count are exactly the cost
  model in section 8

`_PeerCollective` in the test file is a second `Collective` implementation
that folds in a simulated remote rank, since a process full of healthy local
ranks can never reach the cross-rank failure branches on its own. It is also
the shape of the conformance suite a new transport has to pass: the status
agreement protocol, the configuration disagreement protocol, and the buffer
reductions. A transport that passes those and satisfies section 5's five
requirements can be substituted for `LocalCollective` without touching
`distributed.mojo`.

## 11. Relationship to the single-node grower

`grow_tree_distributed` is a separate implementation of the growth loop rather
than a modification of `grow_tree`. That is a deliberate prototype decision
with a real cost: the two loops can drift, and the distributed one silently
misses whatever the single-node one gains.

The mitigation is the world-size-1 equivalence test, which pins the two
together at every commit. If `grow_tree` gains a behavior the distributed
grower does not have, that test fails, which is the correct outcome. The
distributed grower also rejects the tree parameters it does not implement
rather than ignoring them, so an unsupported feature is an error and not a
quietly different model.

The right end state is not two loops. It is one grower that takes its
histograms from a source: a local source that builds them in process, and a
distributed source that builds a local histogram and reduces it. The growth
loop, the frontier, the subtraction trick, and the leaf math would then exist
once. That refactor touches `tree.mojo`, which is the busiest file in the
repository, so it should be done as its own change, with the equivalence tests
already in place, and coordinated with whoever owns that file at the time. It
should not be bundled into the prototype that motivates it.

## 12. Differences from LightGBM

- LightGBM's data-parallel mode reduce-scatters histograms and all-gathers
  candidate splits. This prototype all-reduces the full histogram, which is
  more communication and exactly reproduces single-node training. Section 8
  covers the trade.
- LightGBM offers feature-parallel and voting-parallel modes. This prototype
  offers neither.
- LightGBM fits bin edges from a distributed sample. This prototype requires a
  pre-fit shared `BinMapper`.
- LightGBM's parallel training is configured with `tree_learner` and a machine
  list file. This prototype has no configuration surface at all: a caller
  constructs a `Collective` and passes it. That is a prototype choice, not a
  proposed API.
- LightGBM does not promise that distributed training equals single-node
  training. This prototype promises it wherever the arithmetic is exact, and
  tests it there, because a bit-exact claim is what makes the algorithm
  falsifiable at this stage. Section 6 is precise about where the promise
  stops.
