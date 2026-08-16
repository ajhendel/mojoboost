# Float32 derivatives, re-taken under lambda_l2 = 0

2026-08-16, branch `cpu-round-1` at `f15d107`. Accuracy only, no wall clock, so
a busy box is legitimate and this was taken on one.

The Float32 default was argued from a 39/44 accuracy run taken under
`lambda_l2 = 1`. Andrew's reviewer refused to close on it, correctly: the leaf
value is `-G / (H + lambda)`, so `lambda` is a damping term sitting in exactly
the denominator where Float32 error in `H` would show. Under `lambda = 1` a
small absolute error in `H` is divided into something at least 1. Under the
now-stock `lambda = 0` it is divided into `H` itself, which near an empty leaf
is small. Any Float32 penalty should be **larger** at `lambda = 0`, not smaller,
so the earlier evidence could not settle the question in either direction.

Re-taken here: the two scenarios that carried the objection, both arms, three
whole-process repeats each, `MOJOTREES_DERIVATIVE_PRECISION` as the arm.

Both arms reach the fit. The prediction digests differ in every cell and in
both scenarios, so nothing below is a switch that failed to take effect.

## The result

`imbalanced_binary`, mojotrees CPU, `lambda = 0`:

| metric | float32 | float64 | delta |
|---|---|---|---|
| average_precision | 0.010217821 | 0.010217821 | 0 |
| auc | 0.71479686 | 0.71479686 | 0 |
| logloss | 0.032001055 | 0.032001055 | 0 |

Agreement to eight significant figures on all three metrics, with different
prediction bits underneath. **The 9.4 percent average-precision loss and the
0.006 auc loss that Float32 showed under `lambda = 1` do not reproduce at
`lambda = 0`.** They were a property of that regime, not of the precision.

`multiclass`, mojotrees CPU, `lambda = 0`:

| metric | float32 | float64 | direction |
|---|---|---|---|
| multi_logloss | 0.62112244 | 0.69716779 | float32 better by 0.076 |
| accuracy | 0.93844019 | 0.93749845 | float32 better by 0.0009 |

Float32 is ahead, by 11 percent of logloss. **I am not claiming that as a
Float32 win.** Both arms are in a badly-behaved regime here (see below), and a
difference measured inside one is weak evidence about precision. What it does
do is rule out the direction the objection feared: there is no Float32 penalty
on multiclass at `lambda = 0` either.

## Verdict on the question that was open

The Float32 derivative default **stands**, and now on evidence taken in the
regime the product actually ships. The objection was the right objection and
it is answered: the penalty it predicted would grow when the damping was
removed instead went to zero on one scenario and reversed on the other.

## What this run found that it was not looking for, and which is worse

The defaults flip to stock `lambda_l2 = 0` cost real accuracy, on both engines,
and it cost **us** more than it cost LightGBM. Same scenarios, same tier, CPU,
repeat 0, against the `lambda = 1` pre-stock record:

| scenario, metric | | lambda = 1 | lambda = 0 | ratio |
|---|---|---|---|---|
| imbalanced_binary, average_precision | mojotrees | 0.013600 | 0.010218 | 0.75x |
| | LightGBM | 0.013264 | 0.012352 | 0.93x |
| multiclass, multi_logloss (lower better) | mojotrees | 0.18763 | 0.62112 | 3.31x worse |
| | LightGBM | 0.18698 | 0.43625 | 2.33x worse |

At `lambda = 1` the two engines were level: 0.18763 against 0.18698 on
multiclass logloss, and we were slightly ahead on imbalanced_binary. At
`lambda = 0`, LightGBM's stock setting and therefore the comparator's, **we are
42 percent worse than LightGBM on multiclass logloss and 17 percent worse on
imbalanced_binary average precision.**

Accuracy on multiclass is 0.9384 against LightGBM's 0.9456, so it is not that
our trees are wrong. It is that our probabilities are, and logloss is the
metric that sees it. High accuracy with badly inflated logloss is the signature
of leaf values that are not being controlled once the `lambda` damping is gone,
and LightGBM has something in that path that we do not.

This is an accuracy defect against the one comparator, it is on the default
settings, and it was invisible until the defaults moved. It is outside the
Float32 question and it is reported here rather than held.

## Provenance

- Both arms: measured, three whole-process repeats, bit-identical across
  repeats within each arm.
- `lambda = 1` column: measured, `cpu_accuracy_2026-08-16/records_prestock.json`.
- The reading of *why* multiclass logloss inflates: **not measured**, it is a
  reading of the accuracy-versus-logloss split and it names no LightGBM file.
