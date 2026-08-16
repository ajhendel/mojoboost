# Wire note: the automatic learning rate now reaches Python

From `lane/auto-lr-reachable` to whoever owns `bench/real_data/*`. Nothing in
this note is an edit I made; `bench/real_data/` was not mine to touch. Route
it.

Catalog entry: A38. Everything below is written and **not compiled**: a
timing window held the box for the whole of the lane and no Mojo ran.

## What changed on your side of the boundary

`grow_policy='symmetrictree'` (spelled `oblivious` in a parameter string)
now turns CatBoost's automatic learning rate **on by default**. The rate our
trainer runs at is derived from the objective, the iteration count and the
train row count, exactly as CatBoost derives it, so the CatBoost-mode arm no
longer needs a runtime handover of `get_all_params()` to know what rate to
use. It computes the same number itself.

Under `lossguide` and `depthwise` nothing changed: the flat 0.1 default
stands, because those mirror LightGBM and LightGBM has no such feature.

## The two things that will silently break the arm if you do not do them

**1. `learning_rate` must be genuinely ABSENT from the arm's params, not set
to anything.** CatBoost's gate fires only on an UNSET rate
(`options_helper.cpp:277`, and `NotSet()` is provenance and not a value
comparison, `option.h:80-85`). We reproduce that faithfully: the estimator
now defaults `learning_rate` to `None` and "named" means `is not None`, at
any value. So the key going out as `learning_rate: 0.1` from `BASE_PARAMS`
closes the gate, the derivation never fires, the fit succeeds, and the arm
runs a pin under a heading that says it derives one. This is exactly the
`dict.update` trap you found: `MOJOTREES_CATBOOST_MODE` can override the key
and cannot delete it.

`learning_rate` belongs in `MOJOTREES_CATBOOST_MODE_UNSET`.

**2. `lambda_l2` must be absent too, for the same reason and with a
consequence.** `l2_leaf_reg` is the second of CatBoost's four gate keys
(`options_helper.cpp:280`). CatBoost's own `l2_leaf_reg = 3` leaves its gate
open because CatBoost's default machinery wrote it and did not set the
provenance flag; a `lambda_l2=3.0` typed into our arm's params closes ours,
because a caller typed it. Both readings are CatBoost's; they differ on who
supplied the value.

So `lambda_l2` also belongs in `MOJOTREES_CATBOOST_MODE_UNSET`, and the
consequence is that **until the defaults lane lands, an arm that stops naming
`lambda_l2` trains at our stock 0.0 rather than CatBoost's 3**. That trade is
yours to sequence, and there are only three honest resolutions:

- drop `lambda_l2` from the arm and accept 0.0 until the defaults lane makes
  3.0 the CatBoost-mode default, at which point the arm gets 3.0 from the
  mode default and the gate stays open. This is the one that ends correct.
- keep `lambda_l2=3.0` and accept that the automatic rate does not fire on
  the arm, and say so on the arm rather than in a comment.
- ask for `l2_leaf_reg` to be dropped from our gate condition. That is a
  named divergence from CatBoost and would have to be labeled as ours; I did
  not take it, because the standing rule is that CatBoost mode mirrors
  CatBoost.

The same applies to `leaf_estimation_iterations`, which is CatBoost's third
gate key, if the arm ever names it.

## Which side of the selfcheck problem I chose, and why

**The harness at record time, against `engine_resolved_params`. Not
`selfcheck`.**

`selfcheck.py` trains nothing and downloads nothing by design, and that is
what makes it runnable in under a second anywhere. Our resolved rate is a
function of the train row count, which selfcheck does not have and must not
acquire, because acquiring it means reading the dataset. A "static" check
would have to be a second copy of CatBoost's formula living in Python beside
the Mojo one, which is a worse defect than the one it would catch: two
transcriptions that can drift, with the gate watching the wrong one.

So the comparison is: **CatBoost's read-back against our read-back, both from
fitted models, in `worker.run_job` where both already exist.**

- CatBoost's side you already write, on every CatBoost row:
  `engine_resolved_params` at `engines.py:1275`, which carries
  `get_all_params()['learning_rate']`.
- **Our side needs no new API.** `Model.booster.learning_rate` is the rate a
  fit actually ran at, including the derived one, and it is already written
  into the model dump (`model_dump.mojo:722`) and already read back by
  `mojotrees.inspection.dump_model(model)['learning_rate']`. That is our
  `get_all_params()` and it is symmetric with CatBoost's: a read-back off a
  fitted model, not a prediction about one.

What selfcheck can still own, statically and cheaply, is the two structural
facts that make the comparison possible at all, and both are pure table
reads:

- `learning_rate` and `lambda_l2` appear in `MOJOTREES_CATBOOST_MODE_UNSET`
  and in no other table (your existing both-tables check already covers the
  collision half);
- the mojotrees CatBoost-mode arm declares `grow_policy='symmetrictree'`,
  since that is what turns the derivation on.

`CATBOOST_LEARNING_RATE_TRANSITION` should record that from this commit the
live mechanism on our arm is "derived by mojotrees, compared against
CatBoost's read-back", not "pinned" and not "handed over".

## What a passing comparison looks like

One number, checked on paper rather than by fitting. A37 records CatBoost's
own resolved rate as about **0.4273** at 100 iterations on a 20,000 by 20
shape. Our coefficients for CPU / RMSE / `use_best_model=false` /
`boost_from_average=true` (`options_helper.cpp:216-217`) give **0.427309**
for 20,000 rows and 100 iterations. Same number. If the harness comparison
disagrees, the first thing to check is not the formula.

Two ways the two can legitimately differ, and neither is a bug:

- **task type.** CatBoost's coefficient table is keyed by CPU versus GPU and
  genuinely gives a different rate for the same data. If our arm resolves to
  the GPU and the CatBoost arm ran on the CPU, the rates differ by design.
  Compare arms that resolved to the same device.
- **`boost_from_average`.** We pass what CatBoost would resolve for the loss,
  not what our trainer does, precisely so the rows match. For Logloss that is
  `false` even though we do start from the optimal constant. If CatBoost's
  side ever gets an explicit `boost_from_average`, our side needs the same
  one or the rows diverge.

## Entry point, for the record

The arm trains through `mojotrees.train(params, Dataset)`, which reaches
`train_dataset` in `bindings/_mojotrees.mojo`. That call site derives the
rate. It is one of nine that do; six others refuse an explicit
`auto_learning_rate=True` by name, and `booster_update` is among them, so an
arm that builds a Booster and calls `update()` will not get a derived rate
and will be told why.
