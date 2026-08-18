# First run inside a competitor's harness: NVIDIA gbm-bench, YearPredictionMSD

Date 2026-08-18. Harness NVIDIA/gbm-bench at commit `73a976b`, unmodified
apart from the three anchored edits in `bench/external/patch_gbm_bench.py`
(CUDA-only imports made optional, three arms registered, adapter imported).
No timing code, metric, dataset, or competitor parameter was touched.

**We lose this one.** LightGBM is 1.19x faster than our GPU arm and slightly
more accurate. The result is recorded here because it reproduced a loss our
own harness already knew about, in someone else's harness, which is worth
more than a win we cannot check.

## The measurement

`year` = YearPredictionMSD, 463,715 train x 90 features, 51,630 test, the
dataset's own prescribed split. 500 trees, `max_depth=8`, `num_leaves=256`,
`learning_rate=0.1`, `reg_lambda=1`, all from gbm-bench's `shared_params`.
Three repeats, four arms interleaved inside one process per repeat, `nice 0`,
AC power, no thermal warning recorded.

| arm | median s | min | max | MSE |
| --- | --- | --- | --- | --- |
| lgbm-cpu | 28.941 | 24.152 | 29.553 | 79.7905 |
| lgbm-cpu-det | 29.180 | 26.231 | 29.257 | 79.7905 |
| **mojotrees-gpu** | **34.407** | 31.228 | 36.298 | 79.9967 |
| mojotrees-cpu | 47.175 | 39.564 | 49.037 | 80.2766 |

mojotrees GPU against LightGBM stock is 0.84x, against LightGBM
deterministic 0.85x. The ranges do not overlap (LightGBM's slowest repeat,
29.553, is faster than our fastest, 31.228), so this resolves rather than
sitting inside the spread.

## What it says

**The loss is real and it is the known one.** `bench/results/COMPARISON_RUN_2026-08-16.md`
records us behind on real data at this scale. This is the same finding,
measured by a harness we did not write, on the dataset that harness chose.
1.19x here against the 1.3x recorded there is the same story, not a new one.

**Determinism is not the explanation.** `lgbm-cpu-det` costs LightGBM 0.8%
against stock, 29.180 against 28.941, well inside the repeat spread. The
concern that the stock arm handicaps us is answered: it does, by about one
percent, which is not where the 19% went.

**Our CPU arm is 1.63x behind our GPU arm** and 1.63x behind LightGBM, which
is consistent with the CPU standing recorded elsewhere.

**Accuracy is close and slightly against us.** 79.9967 against 79.7905 MSE,
0.26% worse. Worth stating beside the time rather than omitting.

## What this is not

This is one dataset, at one size, on one machine. `year` at 463,715 rows is
in the band where our own crossover records put us behind, and it says
nothing about the shapes where we lead. It is also not an Apple-silicon
claim: LightGBM here is running on the CPU because it has no Metal path, and
we still lost to it, so the "their accelerator sits idle" framing does not
rescue this number.

**Not for publication.** Per `bench/external/README.md`, the point of
external-harness runs is credibility, and publishing this one would be
accurate and unhelpful. It goes in the record, informs where to look next,
and is available to anyone who asks what our worst comparison looks like.

## Environment

Apple M4, 10 cores, 16 GB, macOS 26.5.2, AC power, no thermal warning.
Python 3.14.6, mojotrees 0.1.0a4, LightGBM 4.7.0, NumPy 2.5.2.
Raw harness output in `run_1.json`, `run_2.json`, `run_3.json`.

## Next

The open question this raises is where the crossover actually sits in the
harness's own datasets, not ours. `year` is the smallest of the regression
sets gbm-bench ships. Running the larger ones would say whether the 1.19x
narrows with scale the way our own sweeps suggest it should.
