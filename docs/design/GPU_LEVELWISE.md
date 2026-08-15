# Depth-batched (level-wise) GPU tree growth

Status (Aug 15 2026): the **growth order** is wired and exposed.
`TreeParams.grow_policy = GROW_DEPTHWISE` (Python `grow_policy="depthwise"`,
parameter string `grow_policy=depthwise`) makes every frontier grower
(`tree.grow_tree`, `tree_sparse.grow_tree_sparse`, the three loops in
`train_gpu.mojo`) split leaves in the order `growth_policy.GrowthSchedule`
plans: one depth at a time, `BUDGET_RANK` admission, ascending node id
within a level. Section 3's parameter semantics are what ships.

**Half of the launch batching now ships too**, in the resident device-search
loop (`train_gpu._device_search_resident`) and there only.
`growth_policy.plan_level` hands that loop a whole planned level, which it
commits and enqueues back to back and then searches in one launch pair, so a
level costs **one host wait instead of one per split** and two search
launches instead of two per split. At the default 31 leaves that is 5 waits
for a tree rather than 30. The row partition and the histogram build are
still per node, so of the eight launches a split costs, six remain per split
and two moved to per level; the table in section 1 is therefore reached for
synchronizations and not yet for launch groups. Section 2's step 2 (one
batched histogram pass over the active row buffer per level, using the
multi-leaf kernels in `gpu_leaf_batching.mojo`) and step 5 (one segmented
partition per level) are what remains.

No benchmark has been run on any of it. The counts above are counts. What
they would have to be read against is `pixi run bench-launch-cost`, which
prices one launch and one wait on the device in hand.

The host-side prototype this document names (`gpu_levelwise.mojo`:
`LevelFrontier`, `plan_level`, `decide_level`, `LevelCommit`, `child_sums`)
was removed the same day, unused and untested. It was a second commit path,
deriving child values from the parent histogram so a level could be
committed before its children's histograms existed. The shipped growers
commit through their own per-split bodies under `GrowthSchedule`, and that
is the shape any batching must keep: batch the histogram builds of a level
(the multi-leaf kernels in `gpu_leaf_batching.mojo`, which both policies
can feed) underneath the existing order, rather than adding a growth loop.
Sections 1, 2, 6, and 10 remain the design for that; references below to
`gpu_levelwise.*` names describe the removed prototype and are kept as the
record of what was proposed. `handoffs/consolidation_round.md` (K8) records the
disposition.

This is a proposal for a **second growth algorithm**, not a faster route to
the trees mojotrees already grows. A level-wise tree and a leaf-wise tree
built from the same data with the same parameters are different models. The
document is written so that claim stays in front of every performance
argument in it.

## 1. Why bother

`tree.grow_tree` and `train_gpu.grow_tree_gpu` grow leaf-wise, as LightGBM
does. One global priority queue over live leaves, one split committed per
iteration, one host decision per split. On the GPU path that shape has two
costs that have nothing to do with arithmetic:

1. **Launch count.** A 31-leaf tree is 31 launch groups and 31 host
   synchronizations. A 255-leaf tree is 255 of each. Every one of them is a
   point where the device drains and the host decides.
2. **Work per launch.** Each group is sized by one node's rows. Early nodes
   hold most of the dataset; late nodes hold a few hundred rows and cannot
   fill a modern GPU. The tail of a leaf-wise tree is a sequence of launches
   that are mostly launch overhead.

Depth-batched growth attacks exactly those two numbers and nothing else. The
live leaves of a tree tile its active rows exactly (this is already true, see
`gpu_active_rows.mojo`), so **one pass over the active row buffer serves an
entire level**, whatever the level's node count. That gives:

| | leaf-wise | level-wise |
| --- | --- | --- |
| launch groups per tree | `num_leaves` | `1 + effective_depth` |
| host synchronizations per tree | `num_leaves` | `1 + effective_depth` |
| rows touched per group | one node's | the whole active buffer |
| rows touched per tree | same | same |

At the default `num_leaves = 31` that is 6 groups against 31. At
`num_leaves = 255` it is 9 against 255. `levelwise_policy.leafwise_profile`
and `levelwise_policy.levelwise_profile` compute these, so the claim is a
count rather than a measurement.

What it explicitly does **not** buy is arithmetic. Per level, both modes
accumulate every active row into exactly one histogram, so the total bin
reads per level are identical. A run that is not launch bound or occupancy
bound will not move. That is the honest ceiling on the whole idea and the
first thing a benchmark should try to falsify.

## 2. The algorithm

State per level: a `LevelFrontier` (`gpu_levelwise.mojo`) holding, for every
node at one depth, its node id, row count, branch feature set, and monotone
output interval.

Per level, in order:

1. **Prefilter.** `levelwise_policy.prefilter_level` drops the nodes that
   cannot split on shape alone: past `max_depth`, or under `2` rows, or under
   `2 * min_data_in_leaf`. These are the two guards at the top of
   `tree._search` and they depend on nothing a histogram holds, so they can
   run before any device work. A node dropped here is terminal.
2. **Batched histogram build.** One launch over the union of the surviving
   nodes' row ranges, writing into per-node histogram slots. Sibling
   subtraction still applies pairwise: build only the smaller sibling of each
   pair directly and derive the larger by subtraction from the parent, which
   halves the rows read per level exactly as it does today.
3. **Batched split search.** One launch scanning every node's histogram. The
   per-node inputs the search already takes (candidate feature set, allow
   mask, monotone interval) become per-node arrays instead of restaged
   scalars.
4. **One host decision.** `gpu_levelwise.decide_level`: build the candidates,
   admit them against the leaf budget, plan the children. This is the only
   host synchronization in the level.
5. **Batched partition.** One segmented pass over the active row buffer.
   Because the level's ranges tile the buffer, the routing for a row is a
   lookup on the node that owns its slot. Each node's `expected_left` comes
   from `child_sums`, so nothing is read back.

Then the committed splits' children become the next frontier and the loop
repeats.

## 3. Parameter semantics

Every one of these is a decision the mode has to make explicitly, because
leaf-wise growth answers them by construction and level-wise growth does not.

### `num_leaves`

Committing a split turns one leaf into two, so it adds exactly one leaf. A
level offering `E` eligible splits adds `E` leaves at once, which can
overshoot the budget in a way a one-at-a-time queue never can.

`num_leaves` stays a **hard bound**. `levelwise_policy.admit_level` resolves
the overshoot in one of two ways:

- `BUDGET_RANK` (the default) admits the highest-gain prefix of the level
  that fits. The budget is met exactly whenever the tree has that many
  eligible splits to give, and the final level is partial.
- `BUDGET_WHOLE_LEVEL` refuses a level it cannot admit entirely, so every
  level is complete and the leaf count lands on the last full level, at most
  `num_leaves` and often well under it. This is the stricter reading and the
  one to use when the point of the run is that every leaf sits at one depth.

Under `BUDGET_RANK` the last level is the only place level-wise growth ranks
siblings against each other. Everywhere else, admission is unconditional.

### `max_depth`

Under leaf-wise growth `max_depth` is a side constraint that rarely binds.
Under level-wise growth it is the primary control, and `num_leaves` becomes a
depth limit in disguise: a complete level at depth `d` already holds `2**d`
leaves, so the budget caps depth at `floor(log2(num_leaves)) + 1`.
`levelwise_policy.effective_max_depth` computes whichever binds first.

The practical consequence is worth stating plainly, because it is the easiest
way to run a meaningless comparison. **A level-wise run at mojotrees's
default `num_leaves = 31` and `max_depth = -1` grows depth-5 trees with 16
full leaves and a partial fifth level.** A leaf-wise run at the same
parameters grows a tree that is much deeper on one branch and much shallower
on another. Those are different model classes, and a level-wise
configuration wants `max_depth` set deliberately rather than inherited.

### `min_data_in_leaf` and `min_sum_hessian_in_leaf`

Unchanged, and enforced in the same two places. The parent-side rejection
(`n_rows < 2` or `n_rows < 2 * min_data_in_leaf`) is mirrored in
`levelwise_policy.rows_permit_split` so a level's launch can be narrowed
before it is issued. The per-child tests, both the row minimum and the
hessian minimum, stay inside `split.find_best_split` and are untouched.

The consequence for the mode is that **a level-wise tree is not a complete
binary tree**. Nodes drop out of the frontier independently as their rows run
thin, so levels get ragged with depth. A batched launch must therefore be
sized by the surviving frontier, not by `2**depth`, and empty or ineligible
nodes must be excluded rather than launched with no work.

### Monotone constraints

Unchanged in mechanism: output intervals inherited parent to child, plus
candidate rejection inside the split search. `gpu_levelwise.plan_level`
applies the same three steps the leaf-wise growers apply, in the same order:
clamp both child values into the parent's interval, collapse both to their
midpoint if a rounding step inverted them under an active constraint, then
divide the parent's interval at that midpoint.

One difference, and it is a numeric one. The leaf-wise growers compute child
values from the children's own freshly built histograms. Level-wise growth
has to commit the level before the children exist, so it computes them from
the parent histogram partitioned by the chosen split (`child_sums`). The two
are equal in exact arithmetic, because the rows routed left are exactly the
rows in the left-going bins, but they round differently. Under the GPU's
Float32 histograms the gap is wider. The monotone guarantee itself is
unaffected, since it rests on the clamp and the midpoint, both of which
operate on whatever values are produced.

### `max_delta_step` and `path_smooth`

Honored, and honored for a reason worth recording: both of the inputs
`tree._leaf_value` needs for them are available to a level-wise grower before
the child's histogram exists. The leaf's row count is the exact integer count
`child_sums` returns, and the parent's emitted output is carried on
`LevelNode.value` rather than read back off `tree.value[parent_node]`, which
a grower planning a whole level at once cannot do.
`gpu_levelwise.child_leaf_value` applies the cap and the smoothing in
LightGBM's order, the same order `_leaf_value` uses.

### Interaction constraints

Unchanged. Both children inherit the parent's branch set extended by the
split feature, and the allow mask is derived from it exactly as today. The
only new cost is that a batched search needs one allow mask per node in the
level rather than one restaged mask per call, which is `n_level_nodes *
n_features` bytes. At 100 features and a 31-node level that is 3.1 KB.

### Categorical splits

Unchanged. A categorical split routes by 256-bit set membership and bin 0
(missing, unseen, dropped) is never a member, so those rows always route
right. The batched partition needs a per-node routing descriptor rather than
a set of kernel scalars: feature, threshold bin, missing bin, default
direction, categorical flag, and four `UInt64` set words. That is 56 bytes
per node, and `RowRouting.from_split` already builds exactly it, so the
routing rule is not duplicated to get there.

### Missing routing

Unchanged, and per node rather than per level: a node's missing bin is its
own split feature's, and different nodes at one level split on different
features. The batched partition carries `missing_bin` and `default_left` per
node for that reason. `child_sums` applies the same override when it counts,
so a planned left count and the device's actual left count agree by
construction, which is what lets the partition run without a readback.

### Feature sampling

The per-tree draw is unchanged. The per-level draw
(`sampling.select_level_features`, already present) is the one that becomes
natural here: it is keyed on depth, and under level-wise growth a depth is a
real frontier rather than a set of nodes that happen to sit at the same
distance from the root. `sampling.select_split_features` composes the tree,
level, and node draws and is the single call a level-wise grower should make.

Batching is not obstructed by any of it. Histograms are accumulated over the
**tree's** feature set (set once per tree by `builder.set_features`), and the
level and node sets only narrow the search. So one batched build serves a
whole level regardless of how the per-node draws came out.

There is a reproducibility consequence that must be stated: the per-node draw
is keyed on node id, and level-wise growth assigns node ids breadth first
while leaf-wise growth assigns them best first. **The same seed therefore
gives different per-node feature sets in the two modes.** Fits are not
comparable seed for seed across modes, and no amount of matching parameters
makes them so.

### Deterministic tie-breaking

Level-wise admission needs a total order on candidates, which leaf-wise
growth never needs because it pops one at a time. The order is:

> **gain descending, then node id ascending.**

Node ids are assigned breadth first, left child before right, in ascending
parent id order, so the order is a function of the tree alone. It does not
depend on launch scheduling, thread count, frontier container mechanics, or
which nodes finished their search first.

This is deliberately **not** the leaf-wise rule. Both shipped growers scan
their frontier with a strict `>` and keep the earliest entry, which is
frontier slot order: the left child overwrites the parent's slot and the
right child is appended, so the order is an artifact of list maintenance.
That is stable and correct for a queue popped one at a time. It is not a rule
that survives ranking a whole level, so this mode states its own.

Ties on gain are common rather than exotic. Two siblings splitting on the
same feature over identical bin totals score identically, and so do many
candidates when gradients are quantized. The node id tiebreak is what makes
those cases reproducible.

## 4. The `Tree` representation is preserved

No new node type, no new arrays, no format change. A level-wise tree is an
ordinary `tree.Tree`:

- Node ids are assigned breadth first instead of best first. Nothing
  downstream depends on the assignment order: prediction, `leaf_ordinals`,
  serialization, `model_dump`, inspection, and TreeSHAP all walk the arrays
  or the child pointers.
- The layout invariants both shipped growers produce are kept. Children are
  appended in pairs, so `left[i] == right[i] - 1` and both exceed `i`, and
  the node array stays topologically sorted from the root.
  `gpu_levelwise.check_child_ids` asserts it, because a mismatch would not
  fail loudly; the tree would still predict, it would just be a different
  tree from the one the level planned.
- Node covers are recorded from `child_sums`, which reads the parent
  histogram's exact integer counts, the same numbers the GPU leaf-wise grower
  already uses. So `has_node_counts` and exact contributions work unchanged.
- Nothing records the growth mode. A saved model is indistinguishable from a
  leaf-wise one and loads into every existing reader.

## 5. Determinism

Run to run, on fixed hardware, the mode is bit-deterministic for the same
reasons the existing GPU path is: the partition is a flag pass, an exact
integer prefix sum, and a scatter, with every destination a pure function of
position; no atomic decides placement. The batched form changes the scan into
a segmented one, which is the same argument per segment.

Across modes it is deterministic but not equal, and there are three separate
reasons, all of them load bearing:

1. Different splits get taken, because the algorithm is different.
2. Different node ids, so per-node feature draws differ.
3. Different summation order for leaf values (`child_sums` against a child
   histogram), so even an identical split gives a leaf value that differs in
   the last bits, and by more under Float32 histograms.

Any test comparing the two modes must be written against tree structure and
metric quality, never against values.

## 6. Memory growth by depth

A histogram cell is one (feature, bin) pair. On the device it is three Int32
fixed-point planes, 12 bytes; on the host it is Float64 gradient, Float64
hessian, and Int count, 24 bytes. At 100 features and 256 bins that is
300 KiB per node on the device and 600 KiB on the host.

A complete level at depth `d` holds `2**d` nodes:

| depth | nodes | device histograms | host histograms |
| --- | --- | --- | --- |
| 4 | 16 | 4.7 MiB | 9.4 MiB |
| 6 | 64 | 18.8 MiB | 37.5 MiB |
| 8 | 256 | 75 MiB | 150 MiB |
| 10 | 1024 | 300 MiB | 600 MiB |
| 12 | 4096 | 1.2 GiB | 2.4 GiB |

The bound that matters is not that table, though, it is the peak with the
leaf budget applied. While a level produces the one below it, both are
resident: the parent level is still needed for the sibling subtraction and
for `child_sums`. A level at depth `d` holds `L` nodes, which is the tree's
entire leaf count at that moment, and can commit at most
`min(L, num_leaves - L)` splits. Peak residency is therefore
`L + 2 * min(L, num_leaves - L)`, maximized at `L = num_leaves / 2`, and

> **peak residency never exceeds `1.5 * num_leaves` histograms.**

Leaf-wise growth already holds one histogram per live leaf; `tree._HistPool`
sizes its free list at `num_leaves + 1` and says so. So at the same leaf
budget, batching a whole level costs at most 1.5 times the peak the shipped
grower already pays. That is the answer to the obvious memory objection, and
it is worth stating because the objection is the first thing anyone raises.

The mode is a genuinely new memory risk in exactly one configuration: leaf
budget lifted, `max_depth` alone bounding growth. There a complete level at
depth 12 is 4096 histograms and 1.2 GiB. `LevelwiseParams.max_level_nodes`
is the control for it: a level wider than the cap is grown in several launch
groups, which trades launches back for residency and leaves the resulting
tree unchanged. `gpu_levelwise.max_level_nodes_for_bytes` derives the cap
from a byte budget.

Row-side memory does not grow with depth at all. The active row permutation
is `n_rows` Int32 plus a scratch buffer and per-block scan sums, whatever the
frontier holds, because the level's ranges tile one buffer.

## 7. Stopping behavior

Growth ends after a level for one of four reasons, reported by
`levelwise_policy.level_stop_reason` in this precedence:

| reason | meaning |
| --- | --- |
| `STOP_DRY_LEVEL` | the level offered no eligible split, so there are no children and nothing deeper can exist |
| `STOP_LEVEL_CAP` | the level had eligible splits and none were admitted, which under `BUDGET_WHOLE_LEVEL` means it did not fit and growth ends on the last complete level |
| `STOP_LEAF_BUDGET` | `num_leaves` is spent |
| `STOP_MAX_DEPTH` | the children sit at the depth limit and can offer no split of their own |

Node-level stopping is unchanged from leaf-wise growth. A node that fails a
shape rule, finds no split, or finds only a non-positive gain becomes a leaf
at its own depth and is never revisited. The strictly-positive gain bar is
the same one both leaf-wise growers set by starting their best gain at `0.0`,
so a zero-gain split is refused identically in every mode.

One asymmetry deserves naming. Under leaf-wise growth a leaf with a small
positive gain sits in the queue and may still be split much later if the
budget survives that long. Under level-wise growth it is split at its own
level or not at all, and under `BUDGET_RANK` a node that loses the final
level's ranking is terminal. This is a real behavioral difference, not an
implementation detail, and it is part of why the two modes fit differently.

## 8. Expected quality differences

Stated as expectations to be tested, not as findings. None of this has been
measured.

1. **At equal `num_leaves`, leaf-wise should fit the training data better per
   tree.** That is LightGBM's entire premise: spending the leaf budget where
   the gain is, instead of spreading it evenly, reduces training loss faster.
   Expect level-wise to need more trees, a larger budget, or greater depth to
   reach the same training loss.
2. **That does not settle generalization.** A level-wise tree is more
   regular, which makes it a weaker and lower-variance learner. XGBoost's
   default is depthwise for a reason, and on small or noisy datasets the
   regularity can win. Do not assume a direction in either sense without a
   benchmark.
3. **Leaf values differ in the last bits** even for identical splits, from
   `child_sums` versus a child histogram, and by more under Float32.
4. **Fits are not comparable seed for seed** across modes when
   `feature_fraction_bynode < 1`, because the per-node draw is keyed on node
   id and the two modes number nodes differently.
5. **Under `BUDGET_RANK` the composition of the final level is gain driven**,
   so the leaf count is exactly `num_leaves` but which nodes got the last
   splits depends on a sibling ranking that leaf-wise growth never performs.
6. **`min_gain_to_split` interacts differently.** Under leaf-wise growth it
   prunes a candidate that would have been taken next; under level-wise
   growth it can empty a whole level and end the tree several levels early.
   The same threshold is therefore a much blunter instrument in this mode.

## 9. Workloads likely to benefit, and not

Likely to benefit:

- Large row counts on a wide GPU, where the tail of a leaf-wise tree issues
  many launches too small to fill the device.
- Large `num_leaves` or deep trees, where the launch count gap is widest
  (255 leaves is 255 groups against 9).
- Many shallow trees, the XGBoost-shaped configuration of `max_depth` 4 to 8
  with a high estimator count, which is exactly where per-tree launch
  overhead is paid most often.
- Small to medium datasets on a GPU, where per-node kernel time is
  comparable to launch overhead and amortizing it across a level is the whole
  cost.

Unlikely to benefit, or likely to lose:

- The CPU backend. `tree.grow_tree` already parallelizes within a node across
  row blocks and pays no launch overhead, so there is nothing for batching to
  recover. This mode is a GPU idea.
- Small `num_leaves`, where there are few launches to save in the first
  place.
- Problems whose signal genuinely is concentrated, where leaf-wise growth's
  unbalanced trees are the point and matching its quality would take enough
  extra depth to erase the launch saving.
- Anything already bound by histogram bandwidth rather than by launches,
  which is the case the mode cannot help by construction, since per-level
  arithmetic is unchanged.

## 10. How a benchmark must be built

This is the part most likely to be got wrong, so it is specified rather than
left to the benchmark author.

**Matched tree count is not a valid comparison.** A level-wise tree at
`num_leaves = 31` and a leaf-wise tree at `num_leaves = 31` are different
models with different capacity. Reporting trees per second, or seconds for
100 trees at matched parameters, measures nothing anyone cares about and will
flatter whichever mode happens to grow the cheaper tree.

The comparison must be **time to matched quality**:

1. Fix a dataset, a validation split, and one metric. Fix the binning, the
   bagging schedule, the seeds, the thread count, and the device. Follow the
   thermal protocol in `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md` on Apple
   silicon; a level-wise run has a different duty cycle from a leaf-wise one
   and thermal state will otherwise leak into the result.
2. Tune each mode separately, under the same tuning budget and the same
   search space, over at least `num_leaves`, `max_depth`, `learning_rate`,
   and the estimator count. Level-wise needs `max_depth` in the space; a
   level-wise run inheriting a leaf-wise configuration is a strawman.
3. Report the **metric against wall clock** curve for each mode, with early
   stopping on the same validation set. The comparison is the Pareto
   relationship between the two curves, not a single number.
4. Report **time to reach each of several target metric values**, so the
   answer is not hostage to one threshold.
5. Report the mechanism metrics alongside: launch groups per tree, host
   synchronizations per tree, mean rows per launch, and peak histogram
   residency. These are what the mode changes directly, they are countable
   rather than measured, and they are what explains whichever way the timing
   goes.
6. Report a matched-parameter run too, clearly labeled as **not** a quality
   comparison. It is useful for isolating the machine effect from the
   algorithm effect, and it is misleading the moment it is quoted alone.
7. Report the negative result if the curves cross or coincide. The mode's
   claim is bounded by construction (same arithmetic, fewer launches), so a
   bandwidth-bound workload showing no gain is the expected outcome there,
   not a failure of the experiment.

## 11. What this lane implemented, and what became of it

Implemented at the time, host-side and free of device imports, in two
modules that no longer exist under those names (commit `f4651d1`):

- A levelwise policy module: `LevelwiseParams`, the budget modes,
  `LevelCandidate`, `rank_level`, `admit_level`, `level_stop_reason`, the two
  shape-rule mirrors, `level_capacity`, `full_level_depth`,
  `effective_max_depth`, and the launch profiles. The surviving parts are
  `src/mojotrees/growth_policy.mojo`, which is now the one leaf pick for
  both growth policies; the launch profiles and capacity helpers went with
  the grower below.
- A `gpu_levelwise` grower: `LevelNode`, `LevelFrontier`, `ChildSums` and
  `child_sums`, `child_leaf_value`, `CommittedSplit`, `LevelCommit`,
  `level_candidates`, `plan_level`, `decide_level`, `check_child_ids`, and
  the sizing helpers. Deleted as a second commit path nothing imported or
  tested; the header of this document says what a rebuilt version would
  sit underneath.

Not implemented, and required before any of this trains a tree:

- Batched device buffers and kernels: a multi-slot histogram output in
  `histogram_gpu.mojo`, a segmented partition in `gpu_active_rows.mojo`, and
  a multi-node search in `gpu_split_search.mojo`. All three are files this
  lane does not own.
- The grower loop itself, and its entry point in `train_gpu.mojo`.
- Tests. This lane wrote none by instruction.

Nothing here is registered in `src/mojotrees/__init__.mojo`, on purpose:
nothing should be able to reach this mode by accident. The integration
steps, in order, are recorded in `handoffs/consolidation_round.md` under
K8.

## 12. Open questions

- **Is the batched build worth it against sibling subtraction?** A fused pass
  over every child range reads each active row once and needs no subtraction;
  a pass restricted to the smaller sibling of each pair reads at most half
  the rows but gathers scattered ranges and then needs `n` subtractions. The
  second should win on bandwidth and lose on regularity. Unmeasured.
- **Should `BUDGET_WHOLE_LEVEL` be the default rather than `BUDGET_RANK`?**
  `BUDGET_RANK` keeps `num_leaves` meaningful and comparable to the leaf-wise
  mode, which is why it is the default here. `BUDGET_WHOLE_LEVEL` gives the
  cleaner algorithm and the fully uniform depth. This is a question about
  what the mode is for, and it should be settled before anything is exposed.
- **Should the mode key its per-node feature draws on something other than
  node id**, so that the two modes are comparable seed for seed? A draw keyed
  on `(depth, position within level)` would be, but it would be a third
  sampling convention in the codebase. Probably not worth it.
- **Does `min_gain_to_split` want a level-aware reading?** Ending a whole
  tree because one level came up short is a much larger effect than pruning
  one candidate, and it may need its own threshold rather than the leaf-wise
  one.
