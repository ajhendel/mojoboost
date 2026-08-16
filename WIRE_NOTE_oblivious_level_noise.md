# Wire note: the CPU oblivious level draws from the wrong stream

`lane/oblivious-noise` does not own `split.mojo` or `tree.mojo`. This is the
one-line change they need so the two backends draw the same noise, and the
reason it is not optional.

## What the device now does

`grow_policy=oblivious` keys `random_strength`'s per-candidate draw by

    (seed, tree_index, DEPTH, feature, bin)   in _OBLIVIOUS_SCORE_DOMAIN

through `tree_parameters_extra.oblivious_score_noise` /
`oblivious_score_stream`, both landed on this branch beside the existing
node-domain pair. The device's copies (`gpu_oblivious_score_stream`,
`oblivious_score_plane`) are asserted word-for-word and value-for-value equal
to them in `tests/test_gpu_oblivious_noise.mojo`.

## What the CPU still does

`split.find_best_split_shared` (split.mojo:1764) draws with

    random_score_noise(noise_stdev, noise_seed, tree_index, node, f, b)

which is the NODE domain, and `tree.grow_tree`'s oblivious loop passes
`node = frontier[order[0]].node` (tree.mojo:2060) -- the level's lowest node
id, which under this mode's numbering is `2^depth - 1`.

## Why that is a divergence and not a difference of spelling

Two independent reasons, either of which alone is disqualifying.

1. **Different domain.** `_RANDOM_SCORE_DOMAIN` and `_OBLIVIOUS_SCORE_DOMAIN`
   are different constants folded into the seed before the first mix, so the
   two streams are disjoint by construction. Even at the same site index the
   two backends would draw unrelated numbers.

2. **Different site index.** `2^depth - 1` is `0, 1, 3, 7, 15, 31` while the
   depth is `0, 1, 2, 3, 4, 5`. They agree at depths 0 and 1 and diverge at
   every depth below that -- so a depth-2 fixture is the shallowest one that
   can see this, and a depth-1 test would pass while the shipped default
   (depth 6) was wrong.

Both models would still train. Both would report success. No test in the
suite before this branch could see it.

## The change

In `split.find_best_split_shared`, the level draw becomes

    noise = oblivious_score_noise(
        noise_stdev, noise_seed, tree_index, depth, f, b
    )

`depth` is already a parameter of that function (split.mojo:1218) and
`tree.grow_tree` already passes `depth=level_depth`, so nothing new has to be
threaded and `node` stays where it is for the other callers.
`find_best_split` (split.mojo:1004), the per-node scan, does **not** change:
a node candidate belongs to a node and keeps the node domain.

`oblivious_score_noise` is already exported from `tree_parameters_extra` on
this branch; split.mojo's import list at line 207 gains one name.

## What must not be "tidied"

The mixing convention is fixed and its irregularity is load-bearing:
`tree_index` carries no `+1`; the site term, the feature and the bin all do.
A reimplementation that regularizes that desynchronizes the two backends
silently.

And the depth term is a COUNTER term, not a generator state. CatBoost gets its
per-level redraw by advancing a running generator
(`greedy_tensor_search.cpp:884`, `GenRand()` inside `CalcScores`, which
`:1199` calls inside the `curDepth` loop; the standard deviation at `:1186`
sits immediately before the loop and is per tree). We reproduce the property
and deliberately not the mechanism, because a running generator is a function
of iteration order and worker count and cannot be asserted equal across two
backends. Do not "fix" it back.
