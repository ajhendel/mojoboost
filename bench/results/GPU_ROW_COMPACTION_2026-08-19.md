# GPU row compaction, measured and declined

Date 2026-08-19, Apple M4, AC power, gbm-bench at the pinned commit, covtype
581,012 x 54 over 7 classes, 100 trees, `max_depth=8`, `num_leaves=256`,
`learning_rate=0.1`, `reg_lambda=1`. Both arms interleaved inside one process
per repeat, three repeats.

| arm | median s | min | max | accuracy |
| --- | --- | --- | --- | --- |
| mojotrees-gpu | **42.205** | 40.224 | 43.158 | 0.8804161682572739 |
| mojotrees-gpu-compact | 64.784 | 54.084 | 68.716 | 0.8804161682572739 |

**Compaction is 1.535x SLOWER and the ranges are disjoint**: the fastest
compaction repeat, 54.084 s, is slower than the slowest stock repeat, 43.158
s. This resolves rather than sitting inside the spread.

Accuracy is identical to sixteen digits, as predicted: the GPU histogram
accumulates in fixed point and an integer sum does not depend on the order of
its terms, so a permutation of the rows cannot move the model. Bit-identity
was never the question here. Speed was, and the answer is no.

## Reach was verified before the measurement, and separately

`MOJOTREES_GPU_COMPACTION_TRACE` exists precisely so a requested-but-inert arm
is distinguishable from a working one, and it earned its keep. At 80,000 x 30
over 7 classes, 10 rounds:

| arm | trace |
| --- | --- |
| off | `arm=off builds=0 scatters=0`, 70 tree records |
| on | `arm=on builds=69 scatters=17595`, 70 tree records |

So the arm ran. The 1.26x loss at that shape was the first signal and the
covtype run is the one that settles it.

## Why it lost, and what that says about the idea

**The cadence is wrong, not the idea.** 17,595 scatters over 70 trees is 251
per tree, which is one per SPLIT. CatBoost compacts once per LEVEL
(`TCalcScoreFold::SelectSmallestSplitSide`). Our GPU arm is leaf-wise at
`num_leaves = 256`, so it pays 255 physical reorders of the binned matrix per
tree where CatBoost pays 8, and each reorder moves `n_rows * n_features`
bytes. The gather it removes is real, and at depth 6 and beyond it is the
largest single term in the fit; the reorder that removes it is thirty times
more expensive than the version CatBoost ships.

This is not an argument for tuning the switch. A per-split compaction cannot
be made to pay by moving a threshold, because the reorder cost scales with
the number of splits and the gather saving does not.

## What this predicted and what it says about the prediction

`bench/results/RESUME_2026-08-19.md` records a static attribution putting 62
percent of this fit in the histogram gather's DRAM traffic and 41 percent in
depths 6, 7 and 8 alone, and it named this switch as the first thing to try.
**The attribution was probably right about where the time is and wrong that
this switch could collect it**, because it priced the gather it removes and
not the reorder it adds.

That is the FOURTH time this week arithmetic over a profile has predicted a
large win and measurement has returned nothing or worse. The rule stands and
is now cheap to state: a profile ranks where to point an experiment. It does
not predict the experiment's result, and no number derived from one is a
result until an interleaved run says so.

## Status of the switch

`MOJOTREES_GPU_ROW_COMPACTION` stays default off, and this file is the priced
cost that the decline requires. It is not deleted: the mechanism is the right
one at level cadence, and the symmetric CPU grower's level-wise fold is the
same idea at CatBoost's cadence.
