# The first real-data accuracy record this project has. Pre-stock, 2026-08-16

`bench/real_data` has existed for some time and had **never been run**. This is
its first complete run, and it is the reference against which every
bit-moving change in either campaign is now gated.

Branch `cpu-round-1`, before the stock-defaults lane merged, so this is the
**pre-stock** baseline: mojotrees on its aligned-parity settings against
LightGBM pinned to match them. The post-stock run is the user-facing one and
comes after the defaults flip.

Six scenarios, both engines, CPU and GPU where the scenario supports it, three
whole-process repeats each, 44 cells. Accuracy runs do not need a quiet box and
this one was taken while both campaigns compiled, which is legitimate: nothing
here is a timing.

## Why it had never run, which is worth more than a footnote

The harness needs `python/mojotrees/_mojotrees.so`, and nothing in the run path
builds it. Without it **every mojotrees cell fails on import** and the run still
exits 0, because `run.py`'s exit code reports whether the matrix ran, not
whether the results were good — which is a deliberate and correct design, and
which also means a completely empty result set looks like a successful run.

The first attempt tonight did exactly that: 44 cells, 27 failures, every
mojotrees row a `cannot import name '_mojotrees'`, and LightGBM's rows all
green. **A reader skimming for a pass rate would have seen "18 pass" and moved
on.** `pixi run build-python` first, then the run.

## The result

**59 pass, 1 fail, 4 warn, 9 skip.**

### The differential, which is the number that matters

Every scenario, mojotrees against LightGBM on the same data, same parameters,
same bins, thresholds pre-registered in `thresholds.json`:

| scenario | metric | mojotrees | LightGBM | direction |
|---|---|---|---|---|
| dense_regression | rmse | 0.310405 | 0.313064 | **better by 0.85%** |
| dense_regression | mae | 0.24677 | 0.24812 | **better by 0.54%** |
| imbalanced_binary | auc | 0.733883 | 0.728118 | **better by 0.0058** |
| imbalanced_binary | logloss | 0.0279399 | 0.0280242 | **better by 0.30%** |
| imbalanced_binary | average_precision | 0.0135996 | 0.0132639 | **better by 0.00034** |
| sparse_highdim | auc | 0.790651 | 0.790579 | **better by 7.2e-05** |
| sparse_highdim | average_precision | 0.780139 | 0.779576 | **better by 0.00056** |
| multiclass | multi_logloss | 0.187631 | 0.186979 | worse by 0.35% |
| multiclass | accuracy | 0.957647 | 0.95782 | worse by 0.00017 |
| ranking | ndcg@10 | 0.99124 | 0.9914 | worse by 0.00016 |
| ranking | ndcg@5 | 0.982252 | 0.982431 | worse by 0.00018 |
| **categorical_missing** | **rmse** | **0.9302** | **0.878444** | **worse by 5.89%, limit 5%** |

**Eleven of twelve differential metrics pass, seven of them in our favor.** On
dense regression, imbalanced binary and sparse high-dimensional data we are
slightly *better* than LightGBM; on multiclass and ranking we are slightly
worse by margins in the fourth decimal.

### The one failure

`categorical_missing`, rmse **0.9302 against 0.878444, worse by 5.89 percent
against a 5 percent limit.** It misses by 0.89 percentage points.

This is a real accuracy gap and it is the only one. It is in the scenario that
combines categorical splits with missing values, which is the most
LightGBM-specific corner of the feature set and the likeliest place for a
semantic difference rather than a numerical one. **Nothing in this round
caused it** — the round has moved no bits that reach this path — so it is a
pre-existing gap now measured for the first time rather than a regression.

It is **not** to be fixed by loosening the threshold. `thresholds.json`'s own
preamble is explicit: "Loosening one after seeing a result is allowed and has
to be done in a commit that says so; editing one to make today's run green is
how a suite stops meaning anything."

### Warnings, and one of them is about the comparator

- **`sparse_highdim/lightgbm` produced two distinct prediction digests across
  three repeats.** LightGBM is run with `deterministic=true` and a fixed seed
  and is *supposed* to be reproducible. It is reported rather than gated
  because reproducibility of the comparator is not this project's invariant to
  hold — but it is worth knowing that our comparator is not bit-stable on this
  scenario while we are.
- **GPU against CPU agreement**: max absolute prediction difference 0.115 on
  dense regression, 0.0642 on imbalanced binary, 0.169 on multiclass, against a
  0.001 limit. The primary metric agrees in every case, and this is the
  documented near-tie divergence of the device split search. It is a warning
  rather than a failure by design, and it is the first time it has been
  quantified on real data.

## What this establishes, stated narrowly

1. **A gate now exists.** Any lane that moves bits can be run against this and
   the answer is a number rather than an argument. That was the point.
2. **Our accuracy is competitive with LightGBM's on five of six scenarios**, on
   real data, at pre-registered thresholds, and on three of them we are ahead.
3. **`categorical_missing` is the one place we are measurably worse**, by
   0.89 points more than the budget allows.
4. Nothing here is a speed statement. `verify.py` decides quality and, in its
   own words, "speed is nobody's".
