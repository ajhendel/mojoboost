# Numeric differential testing against LightGBM

`tools/lgbm_differential.py` is one script that fits both libraries on the
same fixed synthetic data with matched parameters and reports the gap per
feature. It exists for a specific set of rows in
`docs/LIGHTGBM_PARITY.md` that are `partial` for exactly one reason, which is
that the feature is implemented, wired through to Python, and covered by its
own tests, and no LightGBM comparison has ever run.

Status, in one line (2026-08-15). The harness is written and **has not been
run**. No parity row moves until a run produces numbers.

## 1. How to run it

```
pixi run build-python
pixi run -e bench python tools/lgbm_differential.py
```

The `bench` pixi environment is the one with `lightgbm`, `numpy`, and
`python`; `build-python` is what puts a loadable `mojotrees` in `python/`.
The script inserts `python/` on `sys.path` itself, so an installed wheel is
not required and the checkout is what gets measured.

```
python tools/lgbm_differential.py --list            # names and tolerances
python tools/lgbm_differential.py --case dart       # one case
python tools/lgbm_differential.py --case dart --case rf
python tools/lgbm_differential.py                   # every case
```

`--list` imports neither library, so it works on a bare interpreter and is
the cheapest way to read the tolerance table and its reasons.

Exit code zero means every case that ran stayed inside its tolerance.
Nonzero means at least one check exceeded one. A skipped case never changes
the exit code, because a skip says the comparison does not exist rather than
that it passed.

Reproducibility. One seed (`SEED = 20260815`), closed-form generators, no
file read from disk. LightGBM runs at `num_threads=1` with
`deterministic=True` and `force_row_wise=True`; mojotrees is deterministic by
construction. A run prints both library versions, the platform, and the seed
before the first case, so a result can be attached to the pair that produced
it.

## 2. What a tolerance means here

Every tolerance lives in the `TOLERANCES` table in the script, with its
number, its unit, and the sentence that justifies the number. There are four
kinds and they are not interchangeable.

**Exact.** `1e-12` relative, used where both libraries promise the same
arithmetic on the same trees. A rollback is a truncation, and a refit at
`decay_rate=1.0` is the identity on leaf values. The tolerance absorbs a
summation-order difference and nothing else. A failure here is a bug, not
drift.

**Effectively exact.** `1e-9`, used where a real floating-point effect is
expected but is bounded far below anything meaningful. Reordering a sum of a
hundred trees moves a prediction by a few ulps, and a known leaf delta of
0.25 has to show up to nine digits.

**Measured band.** Used wherever the two libraries draw random numbers from
different generators, which is every case involving dropout, bagging, or
feature sampling. mojotrees draws from counter-based splitmix64 and LightGBM
from its own sequential generators, so equal seeds select different sets by
construction and prediction-level agreement is not a thing either library
promises. The band is not a chosen number. LightGBM is fit at five seeds, the
range of its own results is taken, and mojotrees has to land inside that
range widened by half its width (floored at 2% of the midpoint, so that five
seeds landing on the same number does not create an impossible test). This is
the only honest yardstick for "the same algorithm, different draws".

**Direction of effect.** Used where the interesting claim is not a number at
all. Linear leaves must beat constant ones on piecewise-linear data, a forest
must average rather than sum, `label_gain` and the `position` column must
actually be read, and a refit at `decay_rate=0` must move toward the new
data. Each of these compares a library against itself, so it carries no slack
for cross-library divergence and only enough slack to survive a tie.

Two things are deliberately **reported and not asserted**, printed as notes
so a run records them without a verdict. The first is whether LightGBM's
`linear_lambda` reaches the intercept row of its leaf solve, the open
question in `docs/LINEAR_TREES.md`, "Parity-unverified" item 1. The second is
the sign
of the debiasing effect in the position-bias case, which is a property of the
synthetic bias in the generator rather than of either implementation.

## 3. The cases

Ten run and five skip.

| Case | What it asserts | Tolerance |
| --- | --- | --- |
| `linear_tree` | each library's linear fit beats its own constant control; the two land in one accuracy regime | 0.95 ratio, 25% relative RMSE |
| `linear_lambda` | both fits move toward their own constant fit as the penalty rises; mojotrees at `linear_lambda=1e8` **is** its constant fit | 2% per rung, 1e-6 relative |
| `dart` | mojotrees's dart RMSE is inside LightGBM's own drop-seed band; both keep every round; dropout changed both models | measured band |
| `rf` | mojotrees's rf RMSE is inside LightGBM's own bagging-seed band; both average rather than sum; both grow exactly `n_estimators` trees | measured band, 10% |
| `label_gain` | a custom gain vector is read on both sides and changes both models; NDCG@5 inside LightGBM's band; the two rankers order documents alike | measured band, rho >= 0.70 |
| `lambdarank_position_bias` | the `position` column is read on both sides and changes both models; NDCG@5 inside LightGBM's band; rank agreement | measured band, rho >= 0.60 |
| `model_edit_rollback` | each library's rollback reproduces its own one-round-shorter fit; both report the same tree count | 1e-12 relative |
| `model_edit_leaf_output` | LightGBM's rows move by the delta, mojotrees's by delta times the learning rate, and no other row moves | 1e-9 absolute |
| `model_edit_shuffle` | predictions survive the permutation on both sides; both keep every tree | 1e-9 relative |
| `model_edit_refit` | `decay_rate=1.0` is a no-op on both sides; `decay_rate=0.0` fits the new data on both sides | 1e-12 relative, strict |

## 4. The skips, and why they are the important part

A case that cannot be compared honestly skips and prints its reason. These
are the five, and none of them is a matter of effort.

**`dart_early_stopping`.** There is no shared behavior. LightGBM's
`DART::EvalAndCheckEarlyStopping` returns false unconditionally
(`src/boosting/dart.hpp`, read 2026-08-15), so `early_stopping_round` is
silently inert under `boosting='dart'` there. mojotrees does implement it,
by snapshotting the per-tree weight vector on every validation improvement
and restoring it, because popping trees off the end would recover the right
tree set and the wrong weights (`docs/DART.md` section 6). But that path has
no Python entry point, because the estimators refuse `eval_set` together with
dart by name. So one library has the feature and does not expose it here, and
the other does not have the feature. A differential would be a comparison
against a no-op.

**`tree_learner_feature`, `tree_learner_data`, `tree_learner_voting`.**
LightGBM's parallel learners need a real multi-process world, configured with
`num_machines`, a machine list, `local_listen_port`, and sockets. There is no
in-process form of it. mojotrees hosts the entire world inside one process
over `LocalCollective`.
The two configurations cannot both be constructed in one script, so "the same
`tree_learner` on both sides" is not something this harness can build. Two of
the three have a second, independent reason, listed below.

- data parallel is a different algorithm on each side. mojotrees all-reduces
  the full histogram and exactly reproduces single-node training; LightGBM
  reduce-scatters histograms, all-gathers candidates, and fits bin edges from
  a distributed sample, so LightGBM's data parallel is not equal to its own
  serial training either (`docs/distributed.md` section 12).
- voting parallel is a different vote on purpose. mojotrees counts votes and
  breaks ties by ascending feature id; LightGBM aggregates local gains and
  runs a second local pass (`docs/DISTRIBUTED_STRATEGIES.md` sections 2.2 and
  8).

And a differential could not flip the `tree_learner` row anyway. That row is
`partial` because no transport ships, not because no number was measured.

**`sparse_gpu`.** LightGBM's accelerated learners are `device_type='gpu'`
(OpenCL) and `device_type='cuda'` (NVIDIA). Neither is built into the
`lightgbm` wheels this environment installs and neither targets Apple
silicon, where mojotrees's GPU path is Metal. Beyond the platform, the sparse
GPU trainer keeps a CSC binned matrix device-resident and recovers implicit
zeros by subtraction, which LightGBM's GPU learners have no analogue of, so
even on a CUDA host the comparison would be between two different algorithms.
What that row claims today is a self-check against `train_sparse`
(`tests/test_gpu_sparse.mojo`), and a real differential would need a Linux
CUDA host with a GPU-enabled LightGBM build.

## 5. Which parity rows a green run would let us flip

Nothing here flips a row on its own. Each row below is `partial` for a list
of reasons and this harness removes exactly one of them, "no LightGBM
differential". Where the row has other reasons, they are named.

| Parity row | Case(s) | Removes | What would still hold it at `partial` |
| --- | --- | --- | --- |
| `linear_tree` | `linear_tree`, `linear_lambda` | "not compared against LightGBM numerically" | the binned-only trainers, `predict_contrib`, GPU prediction, continued training, LambdaRank, and DART with linear leaves all refuse by name; `dump_model` reports the constant fallback; `lgbm_model_io` refuses `is_linear=1` |
| `linear_lambda` | `linear_lambda` | the unverified statements in `docs/LINEAR_TREES.md` "Parity-unverified" item 1, once the intercept probe's answer is written into that document | nothing, if the probe agrees with the documented placement; a disagreement is a documentation fix, not a code one |
| DART and random forest boosting | `dart`, `rf` | "no LightGBM differential, so the fitted models are not claimed numerically equal to LightGBM's" | dense CPU single-output only from Python; sparse input, `eval_set`, a callable objective, the multiclass classifier, and the ranker all refuse both modes; no `eval_set` early stopping under dart from any entry point |
| `drop_rate` / `max_drop` / `skip_drop` / `xgboost_dart_mode` / `uniform_drop` / `drop_seed` | `dart` | the last unmeasured claim on an already-`supported` row | nothing; this row is already `supported` and the case is a regression guard for it |
| `label_gain` | `label_gain` | "no LightGBM differential covers a custom gain vector" | nothing on this row; labels above 30 stay refused on both sides by design |
| `lambdarank_position_bias_regularization` | `lambdarank_position_bias` | "no LightGBM differential exists for it, so numeric parity is not claimed" | refused together with `eval_set` |
| `Booster.rollback_one_iter` / `reset_parameter` | `model_edit_rollback` | "Not compared against LightGBM numerically" | nothing else is named on that row |
| `Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models` / `refit` | `model_edit_leaf_output`, `model_edit_shuffle`, `model_edit_refit` | "Not compared against LightGBM numerically" | a leaf edit does not recompute ancestor covers or gains, so `dump_model` reports growth-time values for those |
| `refit_decay_rate` | `model_edit_refit` | the reason it sits at `deferred` | it is reachable as `Booster.refit(decay_rate=)` and not as an estimator parameter |
| `tree_learner`, feature parallel, voting parallel | skipped | nothing | the world never leaves the process; no transport ships |
| Sparse GPU training, Sparse input on the GPU | skipped | nothing | no LightGBM counterpart exists on any platform this repository builds for; the crossover is also unmeasured |

The rule the repository already follows applies here without exception. A
row moves after a run produces numbers, and the row cites the run.

## 6. Relationship to the other LightGBM comparisons

Three comparisons already exist and this file does not duplicate them.

- `bench/compare_missing_lightgbm.py` compares missing-value routing
  decisions by reading `dump_model` on one side and a Mojo reference driver
  on the other. It is a semantics table, not a metric.
- `bench/compare_categorical_lightgbm.py` compares held-out RMSE with and
  without native categorical support, against a numerical control.
- `bench/compare_ranking.py` cross-checks the NDCG metric on LightGBM's own
  predictions, then compares independently trained rankers.

`tools/lgbm_differential.py` follows their conventions, which are closed-form
synthetic data, hyperparameters matched on both sides including the ones whose defaults
disagree (`lambda_l2` is 1.0 here and 0.0 in LightGBM, `min_child_hess` is
LightGBM's `min_sum_hessian_in_leaf`, and `enable_bundle` is off on both
because mojotrees does no bundling on these paths), and a printed note
wherever a row is reported for context rather than asserted as a contract.
It lives in `tools/` rather than `bench/` because it measures agreement
rather than speed, and because it has an exit code.
