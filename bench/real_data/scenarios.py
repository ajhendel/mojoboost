"""The eight scenarios, and the one comparator both benchmarks run against.

A differential benchmark is only worth reading if the two libraries were
asked for the same model, and if a reader can tell from the result which
configuration produced it. This module holds both: the scenarios, and the
comparator, in one place, with the reason written next to every entry.

## The comparator

Registered as section C9 of `bench/results/PROFILE_PROTOCOL.md` on
2026-08-16, before any measurement was taken under it:

**There is exactly one comparator, LightGBM at stock defaults plus
`deterministic=true`, labelled `stock+det`.** One arm, one label. No other
LightGBM configuration is published, and speed and accuracy are always
reported together against it.

`deterministic=true` is the only deviation from pure stock that is not a
feature-space pin, and a reader will ask why it is there, so the answer
belongs next to it rather than in a commit message. **It is on because our
arm is reproducible across thread counts at no cost, so it is the setting
that makes the two sides comparable rather than one that handicaps
either.** A comparator whose repeats move is a comparator whose repeat
spread is partly its own nondeterminism, and the differential check reads
one repeat as the cell.

It does not fully succeed, and that belongs beside the claim rather than
arriving as a surprise: **in the first real-data run LightGBM produced two
distinct prediction digests across three repeats on `sparse_highdim`, with
`deterministic=true` already set and a fixed seed**, while the mojotrees
arm was bit-identical across all three. So the flag is on to give the
comparator its best chance at reproducibility, and the best chance is not
a guarantee. `thresholds.json` gates the LightGBM side of the determinism
check as non-gating for that reason.

Everything else LightGBM does is left at its own default, and
`LIGHTGBM_STOCK_DEFAULTS` records what those defaults are and where they
were read, so a result can state what ran without anyone having to open
the LightGBM source to find out.

Every deviation from stock now has to be declared in
`LIGHTGBM_DEVIATIONS_FROM_STOCK` with a reason and an exit condition, and
`selfcheck.check_params` fails if the resolved dict deviates anywhere that
dict does not name. That is the structural part: a pin can no longer be
added by editing one dict.

## What came out, and why the direction matters

The comparator used to carry six more settings, every one of them chosen
to match mojotrees or to match this harness rather than to match a user.
All six are gone:

- `min_data_in_bin = 1`, which forced LightGBM onto our old binner's
  no-minimum-population rule. mojotrees now defaults to LightGBM's 3
  (`src/mojotrees/binning.mojo`, and the `min_data_in_bin` row of
  `docs/LIGHTGBM_PARITY.md`), so both merge the same rare levels.
- `bin_construct_sample_cnt` at the training row count, which forced
  LightGBM to fit its bin edges from every row while mojotrees fit them
  from a 200000-row subsample. That is the pin *inverted*: the comparator
  was doing strictly more binning work than we were, and every binning
  ratio measured under it is wrong in our favor. mojotrees's default is
  LightGBM's 200000 now, so neither library bins rows the other does not.
  They do not draw the *same* 200000 rows (different streams, see the
  `bin_construct_sample_cnt` row of `docs/LIGHTGBM_PARITY.md`), so above
  200000 rows the two fit different edges from equally sized samples. That
  is a real difference, it is stated, and no parameter fixes it.
- `force_row_wise = true`, which chose LightGBM's histogram builder for
  it. Stock LightGBM times both on the first iterations and keeps the
  winner, which is what a user gets, so the comparator now runs against
  whichever builder LightGBM itself decides is faster. That is the
  strongest honest comparator and it retires the largest open caveat in
  `bench/results/INSTRUCTION_AUDIT.md`. The cost is that the resolved
  builder is no longer recoverable from a record; `engines._histogram_builder`
  records that as a null with the reason rather than echoing the request.

  **One open question comes with it, and it is not settled here.**
  LightGBM's own documentation for `deterministic` says "you should also
  set `force_col_wise=true` or `force_row_wise=true`", so dropping the
  forced builder drops the companion setting its determinism advice names.
  The evidence above is that the pairing did not deliver reproducibility
  anyway: the `sparse_highdim` repeats that disagreed were taken with
  `force_row_wise=true` and `deterministic=true` both set. Choosing the
  comparator's algorithm for it is the larger distortion of the two, so
  the builder pin goes and the observation is recorded in
  `comparator_block()` where a reader meets it.
- `zero_as_missing`, `min_gain_to_split` and `boost_from_average`, which
  restated LightGBM's own defaults. They are recorded in
  `LIGHTGBM_STOCK_DEFAULTS` instead of passed, which documents the same
  fact without putting three parameters in every result's parameter
  string that the comparator did not need.

All four made the comparison easier for us in at least one respect, so a
number taken before this change is not comparable with one taken after it
and both benchmarks' old figures are marked "pinned configuration,
superseded" rather than deleted.

A fifth came out on 2026-08-16, out of `BASE_PARAMS` rather than out of the
LightGBM dict, and it is the same kind of thing:

- `lambda_l2 = 1.0`, set on **both** sides. **Superseded 2026-08-16.** The
  reasoning it was set under was correct for the world it was written in
  and is kept here rather than deleted: mojotrees defaulted `lambda_l2` to
  1.0 and LightGBM defaults it to 0.0, so leaving each engine on its own
  default would have compared two different regularizers, and pinning both
  to one value was the only way to ask the two libraries for the same
  model. What changed is not that argument but its premise. mojotrees's
  default is LightGBM's 0.0 now (`TreeParams.default()` in
  `src/mojotrees/tree.mojo`), because the policy is that our defaults *are*
  LightGBM stock and an arm labelled `stock+det` carrying a non-stock
  regularizer is not stock. With the premise gone the pin is gone: both
  engines take 0.0 by saying nothing, `lambda_l2` is recorded in
  `LIGHTGBM_STOCK_DEFAULTS` instead of passed, and `tools/check_parity.py`
  fails if either default drifts back.

  **Numbers move across this change, and not slightly.** `lambda_l2` is the
  denominator of every leaf value and every split gain, `-T(G) / (H + l2)`,
  so on a node with small hessian removing it is a large multiplicative
  change to the leaf and it reorders which candidate split wins. Every
  accuracy figure taken before 2026-08-16 is on the old regularizer on both
  sides and is not comparable with one taken after; a difference across the
  boundary is this pin coming out, not a regression in anything else.
  `COMPARATOR_VERSION` is bumped for exactly that reason.

## What is still set, and what it would take to unset it

- `deterministic = true`. The comparator itself, framed above.
- `feature_pre_filter = false`. On by default, it deletes features that
  cannot satisfy `min_data_in_leaf` at Dataset construction time, which
  removes them from the matrix, from the pool `feature_fraction` samples,
  and from every feature index. That is a data change, not a training
  change, and mojotrees does not do it, so leaving it on compares two
  engines fitting different feature spaces. It comes out the day mojotrees
  implements the filter and not before; a lane is on it.
- `enable_bundle = false`, for the same reason and not for a different
  one. Exclusive feature bundling merges mutually exclusive sparse
  features before binning, which is again a change to the feature space.
  mojotrees's EFB is reachable from Python now
  (`bindings/_mojotrees.mojo` parses `enable_bundle`), so this is closer
  to coming out than it was, but it cannot come out yet: the ranking
  trainer does not apply a bundling plan at all and refuses an active
  switch by name (`efb.check_bundling_honored`), so turning bundling on
  for both engines would raise on one of the eight scenarios rather than
  compare it.
- `verbosity = -1` and a fixed `seed`. Neither changes the model or the
  work. The first keeps LightGBM's log out of the harness's stdout, which
  the backend-proof parser reads; the second makes a repeat a repeat.

Threads are matched by count, not by parameter name: LightGBM reads
`num_threads`, mojotrees reads the MOJOTREES_NUM_WORKERS environment
variable, and the runner sets both from one number before either library
is imported.

## What is NOT in the comparator

`use_quantized_grad`. C9 was first registered as stock plus
`use_quantized_grad=true`, with every mojotrees arm quantized, and that
version was withdrawn the same day and before any measurement was taken
under it. The amendment is at the top of C9. Nothing here sets it, on
either side, and the CPU quantized-gradient lane continues on its own
merits without gating publication of anything.

Where a scenario cannot be aligned, it says so in `caveats` and the caveat
is copied into every result record it produces. Silence about a known
difference is the failure mode this file exists to prevent.
"""

import copy

#: Parameters both engines get, under the names both engines accept.
#: Every one of these is LightGBM's stock default, stated rather than
#: assumed: the point of passing them is that a result's parameter string
#: says what ran, not that either engine needed telling.
#:
#: `lambda_l2` was the exception until 2026-08-16, when it was 1.0 here
#: because that was mojotrees's default. It is not in this dict any more
#: and must not come back: mojotrees defaults it to LightGBM's 0.0 now, so
#: a value here would be a pin rather than a restatement. See the module
#: docstring for what that pin was for and why its premise is gone.
BASE_PARAMS = {
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_data_in_leaf": 20,
    "min_child_hess": 1e-3,
    "lambda_l1": 0.0,
    "max_bin": 255,
    "use_missing": True,
}

#: The comparator's identity, carried by every run and every published
#: table. Bump `COMPARATOR_VERSION` when the resolved dict changes in a way
#: that makes two numbers non-comparable, which is any change to
#: `LIGHTGBM_ALIGNMENT` other than a comment, and any change to a value in
#: `BASE_PARAMS` or to which keys it holds. The second half was learned on
#: 2026-08-16: `lambda_l2` reached LightGBM through `BASE_PARAMS`, not
#: through `LIGHTGBM_ALIGNMENT`, so a rule naming only the alignment dict
#: would have let the comparator's regularizer change without a bump.
COMPARATOR_ID = "stock+det"
#: v2 as of 2026-08-16: `lambda_l2` came out of `BASE_PARAMS`, so the
#: LightGBM side moved from a pinned 1.0 to its own stock 0.0. That changes
#: every leaf value and some split choices on both engines, which is the
#: definition of non-comparable, so a v1 number and a v2 number are not the
#: same measurement.
COMPARATOR_VERSION = 2
COMPARATOR_LABEL = "LightGBM at stock defaults plus deterministic=true"
COMPARATOR_REGISTERED = (
    "bench/results/PROFILE_PROTOCOL.md section C9, as amended 2026-08-16"
)

#: Where every LightGBM default quoted in this module was read, so that a
#: reader can check the table rather than trust it.
LIGHTGBM_DEFAULTS_SOURCE = (
    "microsoft/LightGBM include/LightGBM/config.h and docs/Parameters.rst, "
    "checked at tag v4.7.0 and at master on 2026-08-16"
)

#: The minimum LightGBM this harness will run as the comparator.
#:
#: LightGBM does not reject a parameter it does not know: it logs "Unknown
#: parameter" and trains anyway, and this harness runs at verbosity -1
#: where that line never appears. A build predating `deterministic` would
#: therefore run nondeterministically and record itself as `stock+det`.
#: 4.0 is also the floor for `Dataset.feature_num_bin`, which `_bin_profile`
#: needs. Both adapters check before anything is fitted.
LIGHTGBM_MIN_VERSION = (4, 0)

#: What the comparator is: LightGBM's stock defaults, plus one switch.
#:
#: Five entries and no more. Every one of them is either the switch C9
#: names, a deviation declared in `LIGHTGBM_DEVIATIONS_FROM_STOCK` with an
#: exit condition, or a harness setting that changes neither the model nor
#: the work. `selfcheck.check_comparator` fails if this dict grows an entry
#: that is none of those, which is what stops a seventh setting from
#: arriving as a one-line edit the way the previous six did.
LIGHTGBM_ALIGNMENT = {
    "deterministic": True,
    "feature_pre_filter": False,
    "enable_bundle": False,
    "verbosity": -1,
    "seed": 190019,
}

#: The three entries above that differ from LightGBM's stock default, each
#: with the condition that removes it.
LIGHTGBM_DEVIATIONS_FROM_STOCK = {
    "deterministic": {
        "stock": False,
        "here": True,
        "why": (
            "the comparator itself, and the only deviation that is not a "
            "feature-space pin. The mojotrees arm is reproducible across "
            "thread counts at no cost, so this is the setting that makes the "
            "two sides comparable rather than one that handicaps either. It "
            "does not fully succeed: in the first real-data run LightGBM "
            "produced two distinct prediction digests across three repeats "
            "on sparse_highdim with this already set and a fixed seed, while "
            "the mojotrees arm was bit-identical across all three"
        ),
        "removed_when": (
            "never, while it is the comparator. LightGBM's documentation "
            "records that it may slow training down, which is a cost on the "
            "comparator's side and is accepted deliberately in exchange for "
            "repeats that are repeats"
        ),
    },
    "feature_pre_filter": {
        "stock": True,
        "here": False,
        "why": (
            "on by default it deletes features that cannot satisfy "
            "min_data_in_leaf at Dataset construction, removing them from "
            "the matrix, from the pool feature_fraction samples, and from "
            "every feature index. That is a data change mojotrees does not "
            "make, so leaving it on compares two engines fitting different "
            "feature spaces. This is a load-bearing pin and not a leftover "
            "from the pinned configuration that was just removed"
        ),
        "removed_when": (
            "mojotrees implements the pre-filter, which a concurrent lane is "
            "doing. Not before"
        ),
    },
    "enable_bundle": {
        "stock": True,
        "here": False,
        "why": (
            "exclusive feature bundling merges mutually exclusive sparse "
            "features before binning, which is the same kind of feature-space "
            "change"
        ),
        "removed_when": (
            "mojotrees's EFB is applied by every trainer this harness "
            "reaches. It is reachable from Python now, but the ranking "
            "trainer refuses an active bundling switch by name "
            "(efb.check_bundling_honored), so turning it on for both engines "
            "would raise on the ranking scenario rather than compare it"
        ),
    },
}

#: Entries that are in the dict but are not a deviation from anything: they
#: change what the run reports, not what it computes.
LIGHTGBM_HARNESS_SETTINGS = {
    "verbosity": (
        "keeps LightGBM's log out of the stdout stream run.py parses for "
        "backend proof. The cost is that LightGBM's own report of which "
        "histogram builder it chose is suppressed with it"
    ),
    "seed": "pins the fit so a repeat is a repeat",
}

#: LightGBM parameters this harness deliberately does NOT set, with the
#: default it will therefore use. Recorded rather than passed. Two reasons:
#: a parameter set to its own default is noise in every result's parameter
#: string, and a reader still needs to know what ran. Every value here was
#: read at LIGHTGBM_DEFAULTS_SOURCE.
#:
#: `use_quantized_grad` is in this table on purpose. C9 was first
#: registered with it turned on and that version was withdrawn the same
#: day, so a reader who has seen the first text needs to be able to
#: confirm from the run itself that the comparator is not quantized.
#: `lambda_l2` is in this table as of 2026-08-16 for the same reason as
#: `use_quantized_grad`: it used to be set, to 1.0, on both sides, and a
#: reader who has seen a run from before then needs to be able to confirm
#: from the run itself which regularizer produced the numbers in front of
#: them. Both engines take 0.0 now by saying nothing.
LIGHTGBM_STOCK_DEFAULTS = {
    "use_quantized_grad": False,
    "bin_construct_sample_cnt": 200000,
    "min_data_in_bin": 3,
    "force_row_wise": False,
    "force_col_wise": False,
    "zero_as_missing": False,
    "min_gain_to_split": 0.0,
    "boost_from_average": True,
    "lambda_l2": 0.0,
}

# ---------------------------------------------------------------------------
# The CatBoost peer arm. Everything below this line is reported BESIDE the
# comparator above and never instead of it.
# ---------------------------------------------------------------------------
#
# The headline row is `stock+det` and nothing in this section touches it.
# `LIGHTGBM_ALIGNMENT`, `BASE_PARAMS`, `COMPARATOR_ID` and
# `comparator_block()`'s existing fields are unchanged; the only addition to
# the block is a `peers` key, so a table that already prints the comparator
# prints the peer arm with it and a table that does not is unaffected.
#
# Two rows are produced, both against the same CatBoost arm:
#
#   "us in CatBoost mode vs CatBoost defaults"   engine mojotrees_catboost_mode
#   "our defaults vs CatBoost defaults"          engine mojotrees
#
# Both at matched tree count and matched learning rate, which for CatBoost is
# not a formality: CatBoost picks `learning_rate` itself from the iteration
# count and the dataset when it is not given one. Measured on this machine on
# 2026-08-16 at 20,000 rows by 20 features, CatBoost 1.2.10 resolved
# `learning_rate` to 0.5 at 2 iterations, 0.4273 at 100 and 0.06573 at 1000.
# A run that left it alone would compare two different models and call the
# difference a benchmark.

#: The peer arm's identity. Read beside `comparator_id()`, never in place of
#: it. Bump the version when `CATBOOST_ALIGNMENT` or `CATBOOST_MATCHED`
#: changes in a way that makes two CatBoost numbers non-comparable.
CATBOOST_ARM_ID = "cb-default"
CATBOOST_ARM_VERSION = 1
CATBOOST_ARM_LABEL = (
    "CatBoost at its own defaults, at matched tree count and matched "
    "learning rate"
)
CATBOOST_ARM_REGISTERED = (
    "bench/results/PROFILE_PROTOCOL.md section C9 names one comparator, "
    "LightGBM stock+det. This arm is a peer column reported beside it and "
    "is not a comparator: nothing is gated on it and no threshold is "
    "measured against it"
)

#: Where every CatBoost default in `CATBOOST_LEFT_AT_STOCK` was read.
#:
#: Not from the documentation page, and the distinction matters. CatBoost's
#: docs list `learning_rate` as "depends on the dataset and the number of
#: iterations" and `boosting_type` as "depends on the dataset size and the
#: task type", neither of which is a value a record can carry. The table
#: below is what the library itself resolved, read back through
#: `CatBoost.get_all_params()` after a two-iteration fit, which is the C++
#: side's own answer rather than a transcription of prose.
#:
#: Two consequences of reading it this way, both stated rather than hidden.
#: `get_all_params()` omits `thread_count`, so the resolved thread count is
#: recorded separately by the adapter. And a value that CatBoost derives from
#: the data is the value for *that* fit: `learning_rate` and `boosting_type`
#: are both in that class, which is why this harness pins the first and
#: records the second per run instead of asserting either here.
CATBOOST_DEFAULTS_SOURCE = (
    "catboost.CatBoost.get_all_params() on catboost 1.2.10, read 2026-08-16 "
    "after a two-iteration RMSE fit on 20,000 rows by 20 features, "
    "task_type CPU. Cross-read against catboost/core.py's Pool.quantize "
    "docstring for border_count, feature_border_type, nan_mode and "
    "sparse_features_conflict_fraction"
)

#: The minimum CatBoost this harness will run as the peer arm.
#:
#: 1.2 is the floor for `CatBoost.get_all_params()` returning the resolved
#: plist this module's defaults table was read from, and for
#: `Pool.quantize()` being a separate public step, which is what lets the
#: arm report binning apart from training at all.
CATBOOST_MIN_VERSION = (1, 2)

#: What CatBoost is passed. Three entries and no more, and the same rule
#: applies here as to `LIGHTGBM_ALIGNMENT`: every one is either declared as a
#: deviation from stock with a reason and an exit condition, or as a harness
#: setting that changes neither the model nor the work.
#:
#: The two matched parameters are deliberately NOT here. They come from
#: `BASE_PARAMS` through `CATBOOST_MATCHED`, so that "matched tree count and
#: matched learning rate" is a structural property of the translation rather
#: than two numbers copied into a second dict where they can drift.
CATBOOST_ALIGNMENT = {
    "allow_writing_files": False,
    "logging_level": "Silent",
    "random_seed": 190019,
}

#: The two shared parameters that are forced onto the CatBoost arm, and the
#: `BASE_PARAMS` key each is taken from. `selfcheck.check_catboost_arm`
#: proves the resolved dict carries the same value the other two engines get.
CATBOOST_MATCHED = {
    "iterations": "n_estimators",
    "learning_rate": "learning_rate",
}

#: Every entry in `CATBOOST_ALIGNMENT` and `CATBOOST_MATCHED` that differs
#: from CatBoost's own default, with the condition that removes it.
CATBOOST_DEVIATIONS_FROM_STOCK = {
    "iterations": {
        "stock": 1000,
        "here": "BASE_PARAMS['n_estimators']",
        "why": (
            "a comparison at different budgets is not a comparison. All "
            "three engines are asked for the same number of boosting "
            "iterations, which is what makes a wall time per iteration and "
            "a metric at a fixed budget mean the same thing on each row"
        ),
        "removed_when": (
            "never, while this is a matched-budget column. A CatBoost-at-"
            "1000-iterations figure is a different measurement and belongs "
            "in a differently labelled row"
        ),
    },
    "learning_rate": {
        "stock": "chosen by CatBoost from the iteration count and the "
                 "dataset when it is not given one",
        "here": "BASE_PARAMS['learning_rate']",
        "why": (
            "CatBoost auto-selects the learning rate, and the value it "
            "picks moves with the budget: 0.5 at 2 iterations, 0.4273 at "
            "100 and 0.06573 at 1000, measured on this machine on "
            "2026-08-16 at 20,000 rows by 20 features. Leaving it alone "
            "would compare two different models and read the difference as "
            "engine quality"
        ),
        "removed_when": (
            "never, while this is a matched-rate column. CatBoost's own "
            "auto rate is worth a separate row and it is not this one"
        ),
    },
    "allow_writing_files": {
        "stock": True,
        "here": False,
        "why": (
            "CatBoost writes a training log and a learn_error.tsv into a "
            "catboost_info directory under the working directory during "
            "fit. That is filesystem work inside the timed region that "
            "neither of the other two engines does, so leaving it on "
            "charges CatBoost for writing a log. It is a deviation rather "
            "than a harness setting because it removes work, not output"
        ),
        "removed_when": (
            "the other two engines are also measured writing a per-round "
            "log, which no scenario here asks for"
        ),
    },
}

#: Entries that change what the run reports rather than what it computes.
CATBOOST_HARNESS_SETTINGS = {
    "logging_level": (
        "keeps CatBoost's per-iteration output off the stdout stream that "
        "run.py parses for backend proof. CatBoost prints a progress table "
        "by default and that stream is read, not discarded"
    ),
    "random_seed": "pins the fit so a repeat is a repeat",
}

#: CatBoost parameters this harness deliberately does NOT set, with the value
#: CatBoost therefore resolves. Recorded rather than passed, and read at
#: CATBOOST_DEFAULTS_SOURCE.
#:
#: `boosting_type` is in this table with the value it resolved to on the
#: shape it was read on, and it is the one entry that is not a constant:
#: CatBoost chooses Plain or Ordered from the dataset size and the task, so
#: the adapter records the resolved value per run and this row is the reading
#: rather than the rule.
CATBOOST_LEFT_AT_STOCK = {
    "grow_policy": "SymmetricTree",
    "depth": 6,
    "max_leaves": 64,
    "min_data_in_leaf": 1,
    "l2_leaf_reg": 3,
    "border_count": 254,
    "feature_border_type": "GreedyLogSum",
    "nan_mode": "Min",
    "bootstrap_type": "MVS",
    "subsample": 0.8,
    "sampling_frequency": "PerTree",
    "rsm": 1,
    "random_strength": 1,
    "score_function": "Cosine",
    "leaf_estimation_method": "Newton",
    "leaf_estimation_iterations": 1,
    "leaf_estimation_backtracking": "AnyImprovement",
    "boost_from_average": True,
    "boosting_type": "Plain",
    "model_shrink_rate": 0,
    "model_shrink_mode": "Constant",
    "model_size_reg": 0.5,
    "penalties_coefficient": 1,
    "sparse_features_conflict_fraction": 0.0,
    "bayesian_matrix_reg": 0.1,
    "posterior_sampling": False,
    "random_score_type": "NormalWithModelSizeDecrease",
    "task_type": "CPU",
}

#: The differences between CatBoost's defaults and this harness's shared
#: parameters that NO parameter closes. This is the field to read before
#: quoting either CatBoost row.
#:
#: A peer column is only worth having if a reader can tell what it is a
#: column of. Every entry below is a place where "CatBoost defaults" and
#: "our defaults" are answering the same question with a different model,
#: and none of them is fixed by passing something.
CATBOOST_UNMATCHABLE = {
    "tree_shape": (
        "CatBoost grows symmetric (oblivious) trees of depth 6, where every "
        "node at a level shares one split. mojotrees now HAS a symmetric "
        "policy -- GROW_OBLIVIOUS in src/mojotrees/growth_policy.mojo, "
        "landed 2026-08-16 -- so the old claim here that it had none is "
        "withdrawn. But it is not reachable from this harness: "
        "python/mojotrees/sklearn.py validates grow_policy against "
        "_GROW_POLICIES, which carries 'leafwise' (alias 'lossguide') and "
        "'depthwise' and nothing else, and every arm in bench/real_data goes "
        "through that surface. So the CatBoost-mode mojotrees arm is STILL "
        "depthwise at depth 6 and still NOT the same tree, and the reason "
        "has changed from 'we cannot' to 'the Python layer does not expose "
        "it yet'. A symmetric tree is strictly more constrained than a "
        "depthwise one at the same depth, so this difference is expected to "
        "cost CatBoost accuracy and save it time, and neither side of that "
        "is measured here. This is the one entry in this table that is "
        "expected to STOP being unmatchable: wiring symmetrictree through "
        "_GROW_POLICIES closes it, and then this row becomes a matched "
        "parameter instead of a caveat"
    ),
    "row_sampling": (
        "CatBoost's default bootstrap_type is MVS with subsample 0.8, so it "
        "subsamples rows on every tree. LightGBM and mojotrees do not "
        "subsample at their defaults. MVS is minimum-variance sampling "
        "weighted by gradient magnitude, not uniform bagging, so mojotrees's "
        "bagging_fraction is not an emulation of it. The CatBoost arm "
        "therefore sees about 80 percent of the rows per tree and the other "
        "two see all of them. As of 2026-08-16 mojotrees HAS an MVS sampler "
        "of its own, built from CatBoost's source and off by default, so "
        "'the CatBoost-mode arm does not try' has gone from a statement "
        "about what we can do to a statement about what is wired: nothing "
        "calls the sampler from a round loop yet. When something does, this "
        "entry becomes matchable and should move out of this table"
    ),
    "split_scoring": (
        "CatBoost's default score_function is Cosine and its "
        "random_strength is 1, which adds seeded random noise to every "
        "split score. LightGBM and mojotrees score splits by gain with no "
        "perturbation. No parameter makes these the same rule"
    ),
    "leaf_population": (
        "CatBoost's min_data_in_leaf default is 1 and this harness's shared "
        "value is 20. The CatBoost arm is left at 1 because the column is "
        "'CatBoost defaults'; the CatBoost-mode mojotrees arm is set to 1 "
        "to match it, and the plain mojotrees arm stays at 20"
    ),
    "missing_values": (
        "CatBoost's nan_mode default is Min, which sends every missing "
        "numeric value to the low side. LightGBM and mojotrees learn a "
        "default direction per split. The categorical_missing scenario is "
        "where this would show, and that scenario does not run CatBoost for "
        "a separate reason: see CATBOOST_SCENARIO_SUPPORT"
    ),
    "binning_budget": (
        "CatBoost's border_count is a count of thresholds and LightGBM's "
        "max_bin is a count of bins, so CatBoost's default 254 borders and "
        "this harness's shared max_bin of 255 describe the same granularity "
        "budget, and both sides also produce the SAME NUMBER of borders: "
        "GreedyLogSum yields exactly min(border_count, distinct - 1), which "
        "is LightGBM's rule too. They differ only in where the borders sit, "
        "equal-frequency here against a recursive median split there. The "
        "earlier claim that GreedyLogSum picks under its budget (125, 113 "
        "and 101 borders on a uniform 20,000 by 20 matrix) is WITHDRAWN: "
        "those are used-split counts, because a trained model's per-feature "
        "Borders are cleared and rebuilt from the tree split set "
        "(model_build_helper.cpp:102-103, model.cpp:191-194). Read the grid "
        "off a quantized Pool, never off a fitted model. Missing values cost "
        "one bin on both sides (quantization.cpp:322-345, LightGBM "
        "bin.cpp:394-397), so that is not a divergence either -- only the "
        "side is, CatBoost putting NaN at bin 0 and both of the others last"
    ),
    "learning_rate_precision": (
        "CatBoost stores learning_rate as a 32-bit float. A matched rate of "
        "0.1 resolves to 0.10000000149011612 in get_all_params(), so the "
        "rates are matched to float32 and not exactly"
    ),
    "multiclass_tree_count": (
        "CatBoost builds one tree per iteration with a vector leaf value "
        "for MultiClass, where LightGBM and mojotrees build one tree per "
        "class per iteration. tree_count_ is 100 for a 100-iteration "
        "CatBoost multiclass fit and 700 for a seven-class LightGBM one. "
        "The matched quantity is the iteration count; the tree counts in "
        "the records are not comparable numbers on that scenario"
    ),
    "categorical_encoding": (
        "CatBoost splits categorical features on ordered target statistics "
        "(CTR), which is a different algorithm from LightGBM's and "
        "mojotrees's category-set split and uses the label to build the "
        "feature. STILL TRUE, and still the reason: no scenario in this "
        "suite runs CatBoost with categorical features. What changed on "
        "2026-08-16 is that the absence is now a BLOCKED comparison rather "
        "than an unattempted one, and the two halves have to be read apart. "
        "WHAT IS NOW COMPARED: high_cardinality_categorical puts mojotrees's "
        "category-set split against LightGBM's at five cardinalities chosen "
        "to cross both engines' capacity rules -- max_cat_to_onehot at 4, "
        "max_cat_threshold's 32-category prefix cap, min_data_per_group's "
        "100-row floor, and the 254-category ceiling that "
        "src/mojotrees/categorical.mojo has and LightGBM does not -- plus a "
        "double-centered two-column interaction that no single-column split "
        "carries. That is a real reading of the category-set algorithm at "
        "scale and it did not exist before. WHAT IS STILL NOT COMPARED: "
        "anything CatBoost does. The CTR itself, the ordered permutation it "
        "is computed over, and CTR feature COMBINATIONS -- which is the "
        "mechanism the interaction column was built for -- have no CatBoost "
        "row on that scenario or on any other, and the blocker is not this "
        "table's subject at all. It is the harness's float64 matrix: see "
        "CATBOOST_SCENARIO_SUPPORT['high_cardinality_categorical'], which is "
        "word for word the categorical_missing blocker. So the honest "
        "statement is that the scenario CatBoost's categorical algorithm "
        "would be read on now EXISTS and is unreachable, where before there "
        "was nothing to reach. This entry closes when the loader can hand "
        "every engine an integer-typed categorical block whose digest all "
        "three agree on, and not before"
    ),
}

#: Whether each scenario runs the CatBoost arm, and the exact reason when it
#: does not. A scenario that does not run it is a skip with a stated cause,
#: never an absence.
CATBOOST_SCENARIO_SUPPORT = {
    "dense_regression": None,
    "imbalanced_binary": None,
    "multiclass": None,
    "sparse_highdim": None,
    # Runs, and is the one scenario whose row count was chosen FOR this arm.
    # See ORDERED_BOOSTING_ROWS for what that row count has to be below and
    # for how little of that is verified, and CATBOOST_TIER_CAP for why the
    # arm stops at the standard tier.
    "ordered_boosting_small": None,
    "high_cardinality_categorical": (
        "the same blocker as categorical_missing, on the scenario that was "
        "built for this arm's algorithm. The harness hands every engine one "
        "float64 matrix with the categorical columns integer-coded into it, "
        "and CatBoost refuses cat_features on a floating-point array "
        "outright: \"'data' is numpy array of floating point numerical "
        "type, it means no categorical features\". Handing CatBoost a "
        "converted copy would give it different bytes from the other two "
        "engines, which is the one thing worker.py's data digest exists to "
        "prevent. Running it WITHOUT cat_features is worse than skipping "
        "it: CatBoost would then split five integer-coded columns as "
        "ordinal numbers, which is the exact error native categorical "
        "handling exists to avoid, and the row would sit in a column headed "
        "by two engines doing set splits. So this scenario has no CatBoost "
        "row, and the cost of that is higher here than anywhere else in the "
        "suite, because ordered target statistics and CTR combinations are "
        "what this shape was drawn for. It clears the day the loader can "
        "produce an integer-typed categorical block whose digest all three "
        "engines agree on -- the same day categorical_missing clears, by "
        "the same change"
    ),
    "ranking": (
        "CatBoost has no lambdarank. Its ranking losses are YetiRank, "
        "YetiRankPairwise, QueryRMSE and PairLogit, and none of them is the "
        "loss LightGBM and mojotrees are asked for here. Running one of "
        "them would put a third objective in a column headed by the other "
        "two's, so this scenario has no CatBoost row"
    ),
    "categorical_missing": (
        "the harness hands every engine one float64 matrix with the "
        "categorical columns integer-coded into it, and CatBoost refuses "
        "cat_features on a floating-point array outright: \"'data' is numpy "
        "array of floating point numerical type, it means no categorical "
        "features\". Handing CatBoost a converted copy would give it "
        "different bytes from the other two engines, which is the one thing "
        "worker.py's data digest exists to prevent. The generator also "
        "drops values inside two categorical columns, and CatBoost has no "
        "representation for a missing category: it would have to be encoded "
        "as a level, which is a modelling decision made by the harness. So "
        "this scenario has no CatBoost row until the loader can produce an "
        "integer-typed categorical block whose digest both sides agree on"
    ),
}

#: What running the CatBoost arm is expected to cost, where that is worth
#: knowing before a matrix is committed to rather than after it times out.
#:
#: A cost note is not a caveat on a number. It is a warning to whoever
#: schedules the run, and it is here rather than in a lane report so that
#: the manifest carries it.
CATBOOST_SCENARIO_COST = {
    "sparse_highdim": (
        "expected to be the expensive cell by a wide margin. CatBoost's rsm "
        "default is 1, so it considers every feature at every split, and it "
        "grows symmetric trees. The smoke tier, 3,983 rows by 2,000 "
        "features and 100 iterations, took 8.5 seconds of fit on two "
        "threads on 2026-08-16 while ingestion took 0.011. The standard "
        "tier is 100,000 rows by 50,000 features, which is 25 times each "
        "dimension, so this cell may approach or exceed run.py's 7200 "
        "second per-run timeout. That is a real property of CatBoost on "
        "high-dimensional sparse data and not a harness defect, but the "
        "matrix should be scheduled knowing it: a timed-out cell is an "
        "infrastructure failure and takes the whole run's exit code with it"
    ),
}

#: What `deterministic=true` buys the comparator, and what CatBoost has
#: instead, which is less.
#:
#: The honest form of this is the one the LightGBM entry already takes: say
#: what the setting does and does not buy, rather than claiming
#: reproducibility because a flag was set. CatBoost has no such flag at all,
#: so its like-for-like is a fixed thread count plus a fixed seed and that is
#: weaker: nothing in it is a promise by the library, only two inputs held
#: still.
CATBOOST_DETERMINISM = {
    "flag": (
        "CatBoost has no `deterministic` parameter. LightGBM does, which is "
        "why the comparator is stock+det. There is no CatBoost setting that "
        "asks the library for reproducible reductions"
    ),
    "what_is_pinned": (
        "thread_count, to the same number the other engines get, and "
        "random_seed, to the same 190019 the comparator uses. That is the "
        "whole of it"
    ),
    "status": "seeded, not guaranteed",
    "observed": (
        "checked rather than assumed, and it held everywhere it was looked "
        "at. On catboost 1.2.10, 20,000 rows by 20 features, 100 iterations "
        "at learning_rate 0.1 and random_seed 1234, the prediction digest "
        "was identical across three in-process repeats, across three "
        "separate processes, and across thread_count 1, 2, 4 and 8"
    ),
    "what_that_does_not_establish": (
        "one shape, one loss, one machine, and no dataset with missing "
        "values or categorical features. Bit-identity observed at 20,000 "
        "rows is not bit-identity at 1,000,000, where the parallel "
        "reductions are wider. The digests are recorded per repeat exactly "
        "so this is measured on every run rather than inherited from this "
        "note"
    ),
    "precedent": (
        "LightGBM produced two distinct prediction digests across three "
        "repeats on sparse_highdim with deterministic=true already set and "
        "a fixed seed. A flag being set is not reproducibility on either "
        "side, which is why both arms record a digest per repeat"
    ),
    "gating": (
        "not gating. This is a peer column: nothing in thresholds.json "
        "measures anything against it, and verify.py's differential pairs "
        "mojotrees with lightgbm and does not see these rows"
    ),
}

#: mojotrees's side of "us in CatBoost mode": every CatBoost default this
#: library has a knob for, and nothing invented for the ones it does not.
#:
#: The entries are canonical `shared_params` names, applied over
#: `BASE_PARAMS` before translation, so the CatBoost-mode arm and the plain
#: arm differ in exactly this dict and a reader can diff two records to see
#: it. What is deliberately absent is as much of the definition as what is
#: present: there is no bagging entry, because CatBoost's MVS is not uniform
#: row sampling and a bagging_fraction of 0.8 would be an imitation of the
#: number rather than of the method.
MOJOTREES_CATBOOST_MODE = {
    "grow_policy": "depthwise",
    "max_depth": 6,
    "num_leaves": 64,
    "min_data_in_leaf": 1,
    "min_child_hess": 0.0,
    "lambda_l1": 0.0,
    "lambda_l2": 3.0,
}

#: Why each entry above is what it is, carried into the record beside the
#: dict so the arm explains itself where it is read.
MOJOTREES_CATBOOST_MODE_REASONS = {
    "grow_policy": (
        "the nearest shape this harness can REACH, and not the same one. "
        "Reachable is the operative word since 2026-08-16: a symmetric "
        "policy now exists in the Mojo package but sklearn.py's "
        "_GROW_POLICIES does not expose it, so depthwise remains the "
        "closest this arm can ask for. See CATBOOST_UNMATCHABLE['tree_shape']"
    ),
    "max_depth": "CatBoost's depth default",
    "num_leaves": (
        "CatBoost's resolved max_leaves, which is 2**depth. Set so the leaf "
        "cap does not bind before the depth cap does"
    ),
    "min_data_in_leaf": "CatBoost's min_data_in_leaf default",
    "min_child_hess": (
        "CatBoost has no minimum-hessian rule, so ours is turned off rather "
        "than left at 1e-3, which would be a constraint CatBoost is not "
        "under"
    ),
    "lambda_l1": "CatBoost applies no L1 to leaf values at its defaults",
    "lambda_l2": "CatBoost's l2_leaf_reg default of 3",
}

#: The CatBoost parameters this harness refuses to be handed.
#:
#: `bin_construct_sample_cnt` at the training row count made the LightGBM
#: comparator fit its bin edges from every row while mojotrees fit them from
#: a subsample, which is strictly more binning work on the comparator's side,
#: and it was caught only after a ratio had been published. The names below
#: are the CatBoost shapes of the same defect. Refused by name here and
#: checked statically by `selfcheck.check_no_row_count_injection`, because a
#: rule that lives only in a comment survives exactly until the next call
#: site.
CATBOOST_REFUSED_PARAMS = {
    "border_count": (
        "the binning budget. Stock is 254 borders and it stays stock, for "
        "the same reason max_bin is not pinned on the LightGBM side"
    ),
    "max_bin": "a synonym for border_count in CatBoost's own API",
    "dev_max_subset_size_for_build_borders": (
        "CatBoost's own bin-construction sample cap, which is the direct "
        "counterpart of bin_construct_sample_cnt. Deriving it from the row "
        "count is the defect this list exists for"
    ),
    "dev_efb_max_buckets": "a bundling knob, and bundling is left at stock",
    "sparse_features_conflict_fraction": (
        "exclusive feature bundling. Stock is 0.0, which is off, and that "
        "matches enable_bundle=false on the LightGBM side; pinning it here "
        "would be pinning it to what it already is"
    ),
    "used_ram_limit": (
        "CatBoost changes its blocking and its quantization strategy under "
        "a memory cap, so a limit derived from the data size makes the "
        "engine do different work at different scales"
    ),
    "min_data_in_leaf": (
        "a leaf-population rule. The column is CatBoost's defaults and its "
        "default is 1. Raising it to this harness's shared 20 would make "
        "the arm neither CatBoost's defaults nor ours"
    ),
    "min_child_samples": "a synonym for min_data_in_leaf",
    "subsample": (
        "CatBoost's default bootstrap is MVS at 0.8. Pinning the fraction "
        "makes the arm something other than CatBoost's defaults, and "
        "matching it to a bagging fraction on our side would be matching a "
        "number across two different sampling methods"
    ),
    "thread_count": (
        "set from the runner's thread count by catboost_params itself, "
        "exactly as num_threads is on the LightGBM side. A second source "
        "for it is how one arm ends up on a different core count from the "
        "other two"
    ),
}

#: What each engine's timed phases actually contain, so that three engines'
#: lines can be read against each other.
#:
#: This is not bookkeeping. The end-to-end headline includes ingestion, and
#: the three libraries put ingestion and binning in different places:
#: LightGBM's `Dataset.construct()` contains both, mojotrees times a
#: transpose and a binning pass separately, and CatBoost's `Pool()` contains
#: NEITHER binning nor anything else beyond the conversion.
#:
#: The CatBoost row is the one that needed finding out, and the finding is
#: recorded here rather than folded in. `Pool(X, label=y)` leaves the pool
#: unquantized -- `Pool.is_quantized()` is False immediately after it, and
#: `Pool.quantize()` is a separate public call that flips it -- so CatBoost's
#: binning happens inside `fit` and there is no CatBoost step that
#: corresponds to `Dataset.construct()`.
#:
#: Splitting it out was tried and rejected on evidence, which is the part
#: worth keeping. Calling `Pool.quantize()` before `fit` produces a
#: DIFFERENT MODEL above a few hundred thousand rows: at 300,000 rows by 20
#: features on catboost 1.2.10, fitting from a raw pool, from a pool
#: quantized with the default seed, and from a pool quantized with this
#: harness's seed gave three distinct prediction digests and 51 against 50
#: borders on feature 0. CatBoost builds its borders from a sampled subset
#: whose draw depends on the quantization seed, which the fit path and the
#: explicit path do not share. So the CatBoost arm does not pre-quantize,
#: its `binning` is null with this reason, and its binning time is inside
#: `train` where CatBoost itself puts it. At 20,000 rows the three digests
#: agreed, which is why this had to be checked at a size where the sampling
#: bites rather than at the size that was convenient.
PHASE_SHAPE = {
    "mojotrees": {
        "ingest": "row-major to column-major transpose of the caller's array",
        "binning": "Dataset.construct(), the quantile binning pass",
        "train": "boosting rounds over an already binned matrix",
        "e2e": "ingest + binning + train",
    },
    "lightgbm": {
        "ingest": (
            "not separable. LightGBM's ingestion is inside "
            "Dataset.construct() and it has always been counted in the "
            "binning phase"
        ),
        "binning": "Dataset.construct(), which contains its ingestion",
        "train": "boosting rounds over an already constructed Dataset",
        "e2e": "binning + train",
    },
    "catboost": {
        "ingest": (
            "Pool(X, label=y): conversion of the caller's array into "
            "CatBoost's own object layout, and nothing else. The pool is "
            "not quantized when it returns"
        ),
        "binning": (
            "null. CatBoost bins inside fit, and pre-quantizing to separate "
            "it changes the model above a few hundred thousand rows"
        ),
        "train": "fit(), which contains CatBoost's quantization",
        "e2e": "ingest + train",
    },
}

#: Where each shared parameter ends up on each side, by the name it ends up
#: under. Three destinations, because the two libraries take the same
#: quantity in three different places:
#:
#:   "train"    the dict handed to the trainer, from `lightgbm_params` or
#:              `mojotrees_params`
#:   "dataset"  the dict handed to the Dataset constructor, from
#:              `dataset_params`. mojotrees takes the binning settings here
#:              and LightGBM takes them alongside the training ones
#:   "engine"   neither dict: the engine adapter reads it from BASE_PARAMS
#:              at the call site, because it is a call argument rather than
#:              a parameter (`num_boost_round`) or an estimator keyword
#:
#: This table exists so that selfcheck.check_params can cross-check every
#: shared parameter rather than the two it used to. A parameter added to
#: BASE_PARAMS without a row here fails the self-check, which is the point:
#: the failure mode being guarded against is a shared parameter that
#: silently reaches one engine and not the other.
#:
#: The routing describes the Dataset path, which is what five of the six
#: scenarios take. mojotrees's sparse path goes through the estimator
#: instead and takes max_bin and use_missing as estimator keywords, read
#: from BASE_PARAMS by the adapter: the same values through a different
#: door, and the record for that path says which door it was.
SHARED_PARAM_ROUTING = {
    #  canonical              lightgbm                            mojotrees
    "num_leaves": (("train", "num_leaves"), ("train", "num_leaves")),
    "max_depth": (("train", "max_depth"), ("train", "max_depth")),
    "learning_rate": (("train", "learning_rate"), ("train", "learning_rate")),
    "n_estimators": (("engine", "num_boost_round"), ("engine", "n_estimators")),
    "min_data_in_leaf": (
        ("train", "min_data_in_leaf"),
        ("train", "min_data_in_leaf"),
    ),
    "min_child_hess": (
        ("train", "min_sum_hessian_in_leaf"),
        ("train", "min_child_hess"),
    ),
    "lambda_l1": (("train", "lambda_l1"), ("train", "lambda_l1")),
    # No `lambda_l2` row: it left BASE_PARAMS on 2026-08-16 and neither
    # engine is handed it any more. A row here with no BASE_PARAMS key
    # would describe a route nothing travels.
    "max_bin": (("train", "max_bin"), ("dataset", "max_bin")),
    "use_missing": (("train", "use_missing"), ("dataset", "use_missing")),
}

#: Categorical hyperparameters. Identical defaults on both sides; listed so
#: that a future default change on either side shows up as a diff here
#: rather than as an unexplained quality gap.
CATEGORICAL_PARAMS = {
    "max_cat_to_onehot": 4,
    "max_cat_threshold": 32,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
    "min_data_per_group": 100,
}

#: Ranking hyperparameters, LightGBM's defaults on both sides.
RANKING_PARAMS = {
    "lambdarank_truncation_level": 30,
    "sigmoid": 1.0,
    "lambdarank_norm": True,
    "ndcg_eval_at": 10,
}

#: The cardinality ladder for `high_cardinality_categorical`, per tier, and
#: the argument for every rung. This is the longest comment in this file
#: because the number of levels is the whole design of that scenario, and a
#: reviewer's first and correct question is why these five.
#:
#: **The quantity is rows per level, not levels.** Every rule that governs a
#: categorical column here is written against a count of rows, not a count
#: of categories: `min_data_per_group` is 100 rows, `cat_smooth` is a
#: count-weighted prior on `sum_grad / (sum_hess + cat_smooth)`, LightGBM's
#: `min_data_in_bin` drop is 3 rows, and a target statistic stops being a
#: statistic when a level runs out of rows. So the three tiers hold rows per
#: level FIXED and move the level counts with the row count, which is why
#: the same five arguments hold at every tier and a smoke run is the same
#: problem at 1/50 scale rather than a different one.
#:
#: The first two rungs are the exception and are fixed across tiers, because
#: they are defined against a PARAMETER rather than against a row count:
#:
#: - **8 levels.** Above `max_cat_to_onehot` (4), so the column is searched
#:   by the prefix sort and not one-vs-rest, and below everything else. This
#:   is the CONTROL and is not claimed to stress anything: a CTR over eight
#:   levels at 100,000 rows each is a well-estimated mean and an ordered
#:   target statistic should behave here exactly like an unordered one. It
#:   earns its column by being the row that separates "ordering costs
#:   accuracy because levels are rare" from "ordering costs accuracy
#:   always". It is also, with the 64-level column, one half of the
#:   interaction described below.
#: - **64 levels.** The smallest power of two past `max_cat_threshold` (32),
#:   so this is the first cardinality at which the prefix cap BINDS and the
#:   category-set split becomes a truncated sort rather than a full one. One
#:   factor of two past it and no more, so the truncation is visible without
#:   being the only thing happening.
#:
#: The last three are pinned at 800, 40 and 4 rows per level:
#:
#: - **800 rows per level.** Comfortably above `min_data_per_group`'s 100,
#:   so every level survives the population filter and a per-level target
#:   statistic is well estimated. At the standard tier that is 1,000 levels
#:   against a 32-category prefix: the split sees three percent of the
#:   levels per side and has to choose which.
#: - **40 rows per level.** BELOW `min_data_per_group`'s 100, deliberately.
#:   Under LightGBM's rule the great majority of levels are excluded from
#:   the sort outright, and 40 samples is where `cat_smooth`'s prior
#:   actually moves the estimate rather than rounding off it. This is the
#:   rung where a category-set split and a target statistic should visibly
#:   disagree, and it is the reason to build the second one.
#: - **4 rows per level, and drawn from a power law.** The identifier
#:   column. A UNIFORM column at this ratio is the useless case -- it costs
#:   memory and exercises nothing, because no level is ever estimable -- so
#:   this one is drawn by the same inverse transform `sparse_highdim` uses
#:   on its column index, which puts an estimable head and a singleton tail
#:   in one column, the way a user id or a URL hash actually behaves.
#:
#: What that draw produces at the standard tier is MEASURED, not estimated:
#: on the 800,000 training rows of the 1,000,000-row split, nominal 200,000
#: levels, seed 1907, the column realizes 175,686 distinct levels, of which
#: 42,843 are singletons, 91,137 hold at least LightGBM's `min_data_in_bin`
#: of 3 rows, and only **306** hold `min_data_per_group`'s 100. The busiest
#: level holds 13,739 rows and it takes 167,686 levels to cover 99 percent
#: of the rows. (Read back on 2026-08-16 by drawing that one column with
#: `generators._stream`; nothing was trained and nothing was downloaded.)
#:
#: Those numbers are the scenario's whole point, so they are here rather
#: than in a lane report. **The split search on that column is effectively
#: over 306 levels on both engines**, because the population filter removes
#: the rest -- but LightGBM pays for a bin table of order 100,000 entries to
#: arrive there, since its `max_bin` is a floor on a categorical column's
#: bin count and not a ceiling, while mojotrees pays for 254 and loses
#: everything below the head (`src/mojotrees/categorical.mojo`, "Capacity").
#: An ordered target statistic pays for one numeric column of 255 bins and
#: keeps the tail's information. That three-way gap is what this scenario
#: makes visible and no other scenario in the suite can.
#:
#: **The ladder was checked to carry signal at every rung, and this is the
#: reading.** A ladder whose rare columns are pure noise is decoration, and
#: the way to find that out is not to train a model. Taken on 2026-08-16 at
#: 200,000 rows with the cardinalities divided by five -- (8, 64, 200,
#: 4,000, 40,000), which reproduces the same rows-per-level ladder exactly
#: at one fifth the rows -- by fitting a `cat_smooth`-smoothed per-level
#: target mean on the training split and scoring it on the held-out split,
#: which is a leakage-free target statistic and not a model:
#:
#:     column     rows/level   AUC from that column alone
#:     8 levels      19,956    0.7041
#:     64 levels      2,494    0.7031
#:     200 levels       798    0.6607
#:     4,000 levels      40    0.5753
#:     40,000 levels      4.5   0.5234
#:
#: All five together score 0.8165 on the held-out split; the numeric terms
#: alone score 0.6338. So the rare columns carry real and diminishing signal
#: rather than none -- the 4-rows-per-level column is still 0.52 and not
#: 0.50 -- which is the gradient the ladder was drawn for, and the baseline
#: floor in thresholds.json is set under these numbers rather than over a
#: guess. Nothing was trained and nothing was downloaded to get them.
HIGH_CARDINALITY_LEVELS = {
    # 16,000 training rows. The third rung is 160 rows per level rather than
    # 800, because 800 would put it below `max_cat_threshold` and the prefix
    # cap would stop binding; 100 levels keeps the cap binding and keeps the
    # column above `min_data_per_group`. Stated because it is the one place
    # the tiers are not the same problem at different scale.
    "smoke": (8, 64, 100, 400, 4_000),
    # 800,000 training rows: 800, 40 and 4 rows per level exactly.
    "standard": (8, 64, 1_000, 20_000, 200_000),
    # 1,600,000 training rows: the same three ratios.
    "large": (8, 64, 2_000, 40_000, 400_000),
}

#: The row count for `ordered_boosting_small`, and what it has to be below.
#:
#: **This number is an assumption and this comment is the whole of its
#: provenance.** The scenario exists so that mojotrees's ordered boosting is
#: compared against CatBoost running its OWN `Ordered` default, and CatBoost
#: resolves `boosting_type` from the data rather than taking a constant --
#: `CATBOOST_LEFT_AT_STOCK` records it as the one entry there that is a
#: reading and not a rule. It picks `Ordered` below some size on CPU and
#: `Plain` above it, so this row count must sit on the Ordered side of that
#: threshold or the scenario compares ordered boosting against plain
#: boosting and reports the difference as engine quality.
#:
#: 50,000 is what this scenario assumes, from the commonly quoted reading of
#: CatBoost's documentation that `Ordered` is chosen for small datasets.
#: **Nobody in this repository has read that out of CatBoost's source.**
#: `lane/ordered-boosting` is source-verifying it; until that lands this
#: constant is a placeholder with an argument attached, and it is a constant
#: rather than five literals so that the verified number is a one-line
#: change and every tier moves with it.
#:
#: There is evidence in this very file that the naive reading is WRONG, and
#: it belongs next to the assumption rather than in a lane report.
#: `CATBOOST_LEFT_AT_STOCK` records `boosting_type: "Plain"`, and
#: `CATBOOST_DEFAULTS_SOURCE` says that was read back from
#: `get_all_params()` after a two-iteration fit on **20,000 rows** by 20
#: features. 20,000 is well under 50,000 and CatBoost still resolved Plain.
#: Two readings of that, and this harness cannot tell them apart:
#: the threshold may be far below 50,000, or the rule may not be a row count
#: at all and may take the iteration count or the cell count as an input,
#: in which case a two-iteration fit says nothing about a 100-iteration one.
#: Either way, **a row of this scenario is only the row it claims to be if
#: the resolved `boosting_type` in that record reads `Ordered`.** The
#: CatBoost adapter already writes `get_all_params()` into every record, so
#: that is checkable per run and must be checked before the column is
#: quoted; it is repeated in the scenario's caveats, which travel into every
#: record.
ORDERED_BOOSTING_ROWS = 50_000

#: Size tiers. `smoke` exists to prove the wiring end to end in seconds and
#: is never reported as a benchmark; `standard` is the default; `large` is
#: opt-in and is the only tier that supports a scaling claim.
TIERS = ("smoke", "standard", "large")


def comparator_id():
    """The one-line identity of the comparator, for a table header, a
    parameter summary, or a CSV column."""
    return f"{COMPARATOR_ID}@v{COMPARATOR_VERSION}"


def comparator_block():
    """Everything a published table has to state about what it was measured
    against, as a dict, from the same constants the run uses.

    This exists because the alternative is a convention. Four
    comparator-configuration incidents in three days, three of them caught
    only after a number had been published, all shared one property: the
    result recorded the number and not the configuration that produced it.
    A margin against a throttled comparator, a binning ratio against a
    comparator forced to bin every row, a speculation figure that was a
    tautology over a conditioned subset, and a gain form invalid under L1.
    Every one of those numbers was real and every one was quoted for a
    question it could not answer.

    So the runner writes this into the manifest, into `records.json`, and
    into a column of every CSV row, and prints it before the first cell
    runs. A results file without it is missing a field rather than missing
    a convention.

    `reproducibility` is the field to read second. `deterministic=true` is
    the whole of the deviation from stock that is not a feature-space pin,
    and it is on because it costs the mojotrees arm nothing and gives the
    comparator its best chance at repeats that are repeats. It is not a
    guarantee, and the evidence that it is not is recorded here rather than
    left to be rediscovered.
    """
    return {
        "id": COMPARATOR_ID,
        "version": COMPARATOR_VERSION,
        "label": COMPARATOR_LABEL,
        "registered": COMPARATOR_REGISTERED,
        "one_line": comparator_id(),
        "lightgbm_passed": dict(LIGHTGBM_ALIGNMENT),
        "lightgbm_deviations_from_stock": copy.deepcopy(
            LIGHTGBM_DEVIATIONS_FROM_STOCK
        ),
        "lightgbm_harness_settings": dict(LIGHTGBM_HARNESS_SETTINGS),
        "lightgbm_left_at_stock": dict(LIGHTGBM_STOCK_DEFAULTS),
        "lightgbm_defaults_source": LIGHTGBM_DEFAULTS_SOURCE,
        "lightgbm_min_version": ".".join(str(p) for p in LIGHTGBM_MIN_VERSION),
        "reproducibility": {
            "why_deterministic": (
                "the mojotrees arm is reproducible across thread counts at "
                "no cost, so deterministic=true is the setting that makes "
                "the two sides comparable rather than one that handicaps "
                "either"
            ),
            "known_limit": (
                "it does not fully succeed. In the first real-data run "
                "LightGBM produced two distinct prediction digests across "
                "three repeats on sparse_highdim, with deterministic=true "
                "already set and a fixed seed, while the mojotrees arm was "
                "bit-identical across all three"
            ),
            "documented_companion_setting_not_used": (
                "LightGBM's documentation for deterministic says to set "
                "force_col_wise or force_row_wise as well. This comparator "
                "sets neither, because choosing the comparator's histogram "
                "algorithm for it is the larger distortion. The digests that "
                "disagreed above were taken with force_row_wise set, so the "
                "pairing was not delivering reproducibility either"
            ),
            "gating": (
                "thresholds.json gates the LightGBM side of the repeat "
                "determinism check as non-gating, so a repeat that moves is "
                "reported and not failed. The mojotrees side is gating"
            ),
        },
        "like_for_like": True,
        "like_for_like_reason": (
            "both engines fit the same feature space, from the same bins, "
            "with the same shared parameters, and neither runs quantized "
            "gradients. A speed ratio and an accuracy differential taken "
            "against this arm are comparable quantities, reported together"
        ),
        # Reported BESIDE the comparator and never instead of it. Added as
        # one key rather than as a second mechanism, so that every place
        # that already writes the comparator -- the manifest, records.json,
        # the CSV column, the console banner -- carries the peer arm with
        # it and cannot carry one without the other.
        "peers": peer_arms_block(),
        "peers_are_not_comparators": (
            "the headline row is stock+det. Nothing under `peers` gates "
            "anything, nothing in thresholds.json is measured against it, "
            "and verify.py's differential pairs mojotrees with lightgbm and "
            "does not see the peer rows"
        ),
        "phase_shape": copy.deepcopy(PHASE_SHAPE),
    }


def catboost_arm_id():
    """The one-line identity of the CatBoost peer arm."""
    return f"{CATBOOST_ARM_ID}@v{CATBOOST_ARM_VERSION}"


def catboost_arm_block():
    """Everything a published table has to state about the CatBoost column.

    Same rule as `comparator_block`, for the same reason: a results file
    cannot fail to state its configuration. The fields a reader needs first
    are `determinism`, because CatBoost has no `deterministic` flag and its
    like-for-like is therefore weaker than the comparator's, and
    `unmatchable`, because "CatBoost defaults" and "our defaults" answer the
    same question with different models in seven places no parameter closes.
    """
    return {
        "id": CATBOOST_ARM_ID,
        "version": CATBOOST_ARM_VERSION,
        "label": CATBOOST_ARM_LABEL,
        "registered": CATBOOST_ARM_REGISTERED,
        "one_line": catboost_arm_id(),
        "is_the_comparator": False,
        "catboost_passed": dict(CATBOOST_ALIGNMENT),
        "catboost_matched_from_base_params": dict(CATBOOST_MATCHED),
        "catboost_deviations_from_stock": copy.deepcopy(
            CATBOOST_DEVIATIONS_FROM_STOCK
        ),
        "catboost_harness_settings": dict(CATBOOST_HARNESS_SETTINGS),
        "catboost_left_at_stock": dict(CATBOOST_LEFT_AT_STOCK),
        "catboost_defaults_source": CATBOOST_DEFAULTS_SOURCE,
        "catboost_min_version": ".".join(str(p) for p in CATBOOST_MIN_VERSION),
        "catboost_refused_params": dict(CATBOOST_REFUSED_PARAMS),
        "determinism": copy.deepcopy(CATBOOST_DETERMINISM),
        "unmatchable": dict(CATBOOST_UNMATCHABLE),
        "scenarios_not_run": {
            name: reason
            for name, reason in CATBOOST_SCENARIO_SUPPORT.items()
            if reason
        },
        "cost_warnings": dict(CATBOOST_SCENARIO_COST),
        "rows": {
            "us in CatBoost mode vs CatBoost defaults": (
                "engine mojotrees_catboost_mode against engine catboost. "
                "The mojotrees side takes MOJOTREES_CATBOOST_MODE over "
                "BASE_PARAMS; the CatBoost side is unchanged between the "
                "two rows"
            ),
            "our defaults vs CatBoost defaults": (
                "engine mojotrees against engine catboost. The same "
                "mojotrees arm the comparator row uses, so one mojotrees "
                "record serves both the headline and this row"
            ),
        },
        "mojotrees_catboost_mode": dict(MOJOTREES_CATBOOST_MODE),
        "mojotrees_catboost_mode_reasons": dict(
            MOJOTREES_CATBOOST_MODE_REASONS
        ),
        "matched": (
            "tree count and learning rate. Both rows run the same number of "
            "boosting iterations and the same learning rate on both sides, "
            "because a comparison at different budgets is not a comparison "
            "and because CatBoost picks its own learning rate from the "
            "budget when it is not given one"
        ),
    }


def peer_arms_block():
    """Every arm reported beside the comparator, keyed by engine name.

    A dict rather than a list so that a record can be looked up by the
    engine that wrote it, and so that adding a fourth engine is one entry
    rather than a second mechanism.
    """
    return {"catboost": catboost_arm_block()}


def peer_banner():
    """The peer arms on the console, under the comparator banner.

    Short enough to read and specific enough to check, in the shape
    `run.comparator_banner` already uses. The full block goes into the
    manifest and into records.json.
    """
    block = catboost_arm_block()
    passed = " ".join(
        f"{key}={value}" for key, value in sorted(block["catboost_passed"].items())
    )
    matched = " ".join(
        f"{key}=BASE_PARAMS[{source!r}]"
        for key, source in sorted(block["catboost_matched_from_base_params"].items())
    )
    return (
        f"peer arm {block['one_line']}: {block['label']}\n"
        f"  reported beside the comparator, never instead of it. "
        f"{block['registered']}\n"
        f"  catboost gets: {passed}\n"
        f"  matched from BASE_PARAMS: {matched}\n"
        f"  everything else is CatBoost's own default "
        f"({block['catboost_defaults_source']})\n"
        f"  determinism: {block['determinism']['status']}. "
        f"{block['determinism']['flag']}\n"
        f"  observed: {block['determinism']['observed']}\n"
        f"  binning: {PHASE_SHAPE['catboost']['binning']}"
    )


def check_catboost_version(version):
    """Raise unless this CatBoost is new enough to be the peer arm.

    Called before anything is fitted, for the same reason the LightGBM
    guard is: an engine that silently ignores a parameter and trains anyway
    produces a record that names a configuration it did not run.
    """
    parts = []
    for piece in str(version).split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts) < CATBOOST_MIN_VERSION:
        want = ".".join(str(p) for p in CATBOOST_MIN_VERSION)
        raise RuntimeError(
            f"the CatBoost peer arm is {CATBOOST_ARM_LABEL} and this harness "
            f"needs CatBoost {want} or newer. This environment has {version}. "
            "Below 1.2 the resolved parameter list this arm records is not "
            "readable back through get_all_params(), so a record would state "
            "defaults nobody checked."
        )


def check_lightgbm_version(version):
    """Raise unless this LightGBM is new enough to be the comparator.

    LightGBM does not reject a parameter it does not know: it logs
    "Unknown parameter" and trains anyway, and this harness runs at
    verbosity -1 where that line never appears. An unguarded old build
    would therefore ignore `deterministic`, train nondeterministically, and
    record itself as `stock+det`. Called before anything is fitted.
    """
    parts = []
    for piece in str(version).split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts) < LIGHTGBM_MIN_VERSION:
        want = ".".join(str(p) for p in LIGHTGBM_MIN_VERSION)
        raise RuntimeError(
            f"the comparator is {COMPARATOR_LABEL} and this harness needs "
            f"LightGBM {want} or newer. This environment has {version}, "
            "which would log 'Unknown parameter' at a verbosity this harness "
            "suppresses and then train without it, producing a comparator "
            "that records itself as stock+det and was not."
        )


def _scenario(**kw):
    kw.setdefault("params", {})
    kw.setdefault("caveats", [])
    kw.setdefault("devices", ["cpu"])
    kw.setdefault("engines", ["mojotrees", "lightgbm"])
    return kw


SCENARIOS = {
    "dense_regression": _scenario(
        task="regression",
        objective="regression",
        title="Dense regression",
        dataset="year_prediction_msd",
        generator="dense_regression",
        generator_sizes={
            "smoke": {"n_rows": 5_000, "n_features": 20},
            "standard": {"n_rows": 200_000, "n_features": 50},
            "large": {"n_rows": 1_000_000, "n_features": 100},
        },
        devices=["cpu", "gpu"],
        primary_metric="rmse",
        notes=(
            "The baseline case. Dense float matrix, no missing values, no "
            "categories, squared error. If the engines disagree here they "
            "will disagree everywhere, so this scenario is the first one to "
            "read when a verdict comes back red."
        ),
    ),
    "imbalanced_binary": _scenario(
        task="binary",
        objective="binary",
        title="Imbalanced binary classification",
        dataset="bank_marketing",
        generator="imbalanced_binary",
        generator_sizes={
            "smoke": {"n_rows": 20_000, "n_features": 20, "positive_rate": 0.02},
            "standard": {"n_rows": 300_000, "n_features": 40, "positive_rate": 0.005},
            "large": {"n_rows": 2_000_000, "n_features": 40, "positive_rate": 0.002},
        },
        devices=["cpu", "gpu"],
        primary_metric="average_precision",
        notes=(
            "Half a percent positive on the synthetic variant, around "
            "eleven percent on the real one. Average precision is the "
            "primary metric because AUC saturates and accuracy is "
            "meaningless at this rate. `is_unbalance` and `scale_pos_weight` "
            "are left off on both sides: they change the objective, and an "
            "unweighted comparison is the one that isolates the trees."
        ),
        caveats=[
            "The real variant, bank_marketing, has ten categorical columns. "
            "They are passed as categorical to both engines, so this "
            "scenario overlaps with categorical_missing on real data. The "
            "synthetic variant is purely numeric and does not."
        ],
    ),
    "multiclass": _scenario(
        task="multiclass",
        objective="multiclass",
        title="Multiclass classification",
        dataset="covertype",
        generator="multiclass",
        generator_sizes={
            "smoke": {"n_rows": 10_000, "n_features": 15, "n_classes": 4},
            "standard": {"n_rows": 200_000, "n_features": 30, "n_classes": 7},
            "large": {"n_rows": 500_000, "n_features": 54, "n_classes": 7},
        },
        devices=["cpu", "gpu"],
        primary_metric="multi_logloss",
        notes=(
            "Softmax with one tree per class per round, so a round costs "
            "num_class times what a binary round costs and the model is "
            "num_class times the size. Both engines are asked for the same "
            "num_class and the same number of rounds, not the same number "
            "of trees."
        ),
    ),
    "ranking": _scenario(
        task="ranking",
        objective="lambdarank",
        title="Ranking",
        dataset="mslr_web10k",
        generator="ranking",
        generator_sizes={
            "smoke": {"n_queries": 800, "n_features": 15},
            "standard": {"n_queries": 20_000, "n_features": 30},
            "large": {"n_queries": 100_000, "n_features": 136},
        },
        params=dict(RANKING_PARAMS),
        primary_metric="ndcg@10",
        notes=(
            "LambdaRank on graded relevance, split by query on both sides. "
            "NDCG is computed by quality.py from the raw scores, never read "
            "off either engine's own metric output."
        ),
        caveats=[
            "Ranking models are the least likely to match closely. LightGBM "
            "reads its pairwise sigmoid from a lookup table where mojotrees "
            "evaluates it, so the two diverge at the first tie and the "
            "thresholds for this scenario are correspondingly loose.",
            "The real dataset is acquired manually, so a run without it "
            "falls back to the generator and says so in the record.",
        ],
    ),
    "categorical_missing": _scenario(
        task="regression",
        objective="regression",
        title="Categorical and missing data",
        dataset="adult",
        generator="categorical_missing",
        generator_sizes={
            "smoke": {"n_rows": 10_000, "n_numeric": 8, "n_categorical": 4},
            "standard": {"n_rows": 200_000, "n_numeric": 20, "n_categorical": 8},
            "large": {"n_rows": 1_000_000, "n_numeric": 40, "n_categorical": 12},
        },
        params=dict(CATEGORICAL_PARAMS),
        primary_metric="rmse",
        notes=(
            "Integer-coded categories split by category set on both sides, "
            "with no one-hot expansion, plus values missing not at random. "
            "This is the scenario where the default missing direction and "
            "the category-set search are actually under test."
        ),
        caveats=[
            "The real variant, adult, is a binary classification problem, so "
            "when it is selected the task becomes binary and the primary "
            "metric becomes auc. loaders.py sets both; the record carries "
            "whichever ran.",
            "Category codes come from the same encoding on both sides, "
            "produced once by the loader. Letting each library encode the "
            "strings itself would compare two encodings, not two trainers.",
        ],
    ),
    "high_cardinality_categorical": _scenario(
        task="binary",
        objective="binary",
        title="High-cardinality categorical",
        # No real dataset, and that is a gap rather than a preference. The
        # real reading of this shape is a click log -- Criteo or Avazu --
        # and registering one means a sources.json entry, a loader, a
        # licence check and a multi-gigabyte fetch, none of which is this
        # lane's to do and none of which can be done without downloading.
        # This is the first scenario in the suite with no dataset at all, and
        # `worker.build_data` was corrected for it: a null `dataset` now
        # produces `fallback_reason: null` plus a `no_real_variant` field,
        # rather than the string "no real dataset for this scenario", which
        # summarize.py flags as "a scenario that names a real dataset ran on
        # the generator instead". That sentence would be true-sounding and
        # false here. A scenario with no real variant is a different record
        # from one whose real variant was missing.
        dataset=None,
        generator="high_cardinality_categorical",
        generator_sizes={
            "smoke": {
                "n_rows": 20_000,
                "n_numeric": 10,
                "cardinalities": HIGH_CARDINALITY_LEVELS["smoke"],
            },
            "standard": {
                "n_rows": 1_000_000,
                "n_numeric": 10,
                "cardinalities": HIGH_CARDINALITY_LEVELS["standard"],
            },
            "large": {
                "n_rows": 2_000_000,
                "n_numeric": 10,
                "cardinalities": HIGH_CARDINALITY_LEVELS["large"],
            },
        },
        params=dict(CATEGORICAL_PARAMS),
        primary_metric="auc",
        notes=(
            "One million rows and five categorical columns whose level "
            "counts cross four different rules: max_cat_to_onehot at 4, "
            "max_cat_threshold's 32-category prefix cap, "
            "min_data_per_group's 100-row floor, and the 254-category "
            "ceiling mojotrees has and LightGBM does not. "
            "HIGH_CARDINALITY_LEVELS carries the argument for each. This is "
            "the shape ordered target statistics and CTR feature "
            "combinations exist for, and until 2026-08-16 no scenario in "
            "this suite had it, so none of that machinery could appear in a "
            "published number however well it worked. The two-column "
            "interaction is double-centered so neither marginal carries it: "
            "a tree needs two stacked set splits to reach it and a "
            "combination feature reaches it in one."
        ),
        caveats=[
            "This scenario is expected to be WORSE for mojotrees than any "
            "other in the suite, for a reason that is written down rather "
            "than discovered: src/mojotrees/categorical.mojo keeps at most "
            "max_bin - 1 = 254 categories per column and lumps the rest "
            "into bin 0, which never joins a split set, while LightGBM's "
            "max_bin is a FLOOR on a categorical column's bin count and it "
            "keeps admitting categories until it has both covered 99 "
            "percent of the rows and spent max_bin bins. On the standard "
            "tier's 200,000-level column that is 254 bins against roughly "
            "100,000. thresholds.json carries a correspondingly wide gate "
            "with an exit condition, and the gate is wide because it bounds "
            "a known structural deficit and not a tie-break.",
            "The 254-versus-100,000 gap cuts the other way on cost, and the "
            "large tier is where that is expected to hurt. LightGBM builds "
            "and clears a bin table of that size at every node of every "
            "tree; mojotrees builds 254. Nobody has timed either, so this "
            "is a derived expectation and not an observed limit: the "
            "arithmetic is 31 leaves by 100 iterations by a sort over the "
            "kept categories. run.py's per-run timeout is 7200 seconds and "
            "the standard tier is expected to fit inside it; the large "
            "tier, at twice the rows and twice the levels, is not "
            "established to. A timed-out cell is an infrastructure failure "
            "and takes the whole run's exit code, so schedule the large "
            "tier knowing that.",
            "No missing values, unlike categorical_missing. The two "
            "scenarios are meant to differ in exactly one axis, so a "
            "difference between them is attributable.",
            "CPU only, for categorical_missing's reason and not a new one: "
            "there is no accelerator path for categorical splits this "
            "harness is willing to compare.",
            "There is no CatBoost row and the cost of that is higher here "
            "than anywhere else in the suite, because CatBoost's ordered "
            "target statistic is the algorithm this shape was drawn for. "
            "See CATBOOST_SCENARIO_SUPPORT and "
            "CATBOOST_UNMATCHABLE['categorical_encoding'].",
        ],
    ),
    "ordered_boosting_small": _scenario(
        task="regression",
        objective="regression",
        title="Ordered boosting at small scale",
        dataset=None,
        # Deliberately the dense_regression recipe rather than a seventh
        # one. The axis under test is the boosting SCHEME at a chosen row
        # count; the data should be the one already understood. A distinct
        # seed keeps the two scenarios from being the same rows at a shared
        # size.
        generator="dense_regression",
        generator_sizes={
            # Derived from ORDERED_BOOSTING_ROWS rather than written out, so
            # that the verified threshold is a one-line change.
            "smoke": {
                "n_rows": ORDERED_BOOSTING_ROWS // 10,
                "n_features": 15,
                "seed": 1908,
                "noise": 0.60,
            },
            "standard": {
                "n_rows": ORDERED_BOOSTING_ROWS,
                "n_features": 30,
                "seed": 1908,
                "noise": 0.60,
            },
            # The ONLY scenario here whose large tier does not add rows. Its
            # row count is its identity: adding rows would move it across
            # the threshold it is defined by and produce a Plain-boosting
            # row under an Ordered heading. It scales in features instead.
            "large": {
                "n_rows": ORDERED_BOOSTING_ROWS,
                "n_features": 100,
                "seed": 1908,
                "noise": 0.60,
            },
        },
        primary_metric="rmse",
        notes=(
            "50,000 rows of the dense regression recipe at a high noise "
            "level, so a 100-tree model at 31 leaves genuinely overfits and "
            "a scheme that corrects prediction shift has something to "
            "correct. The row count is the point: CatBoost resolves "
            "boosting_type from the data, and this scenario is the one "
            "place in the suite where its Ordered default is supposed to be "
            "what runs. ORDERED_BOOSTING_ROWS carries what that number has "
            "to be below and how little of it is verified."
        ),
        caveats=[
            "READ THE RESOLVED boosting_type IN THE RECORD BEFORE QUOTING "
            "THE CATBOOST ROW. This scenario's row count is an assumption, "
            "not a verified threshold, and this file records evidence "
            "against the naive reading of it: CATBOOST_LEFT_AT_STOCK has "
            "boosting_type Plain, read from get_all_params() after a fit on "
            "20,000 rows, which is well under this scenario's 50,000. If a "
            "record's resolved boosting_type reads Plain, the CatBoost row "
            "is a plain-boosting row and comparing ordered boosting against "
            "it is a mislabelled comparison, not a result. See "
            "ORDERED_BOOSTING_ROWS.",
            "The catboost_lossguide row is expected NOT to be an ordered "
            "row even when the catboost row is. That arm passes "
            "grow_policy=Lossguide, and ordered boosting is a symmetric-tree "
            "mechanism in CatBoost, so the library is expected either to "
            "refuse the pair or to resolve to Plain. Neither has been "
            "checked here and this is a stated expectation, not an "
            "observation. Whichever happens is what the record says, and "
            "the harness has no way to skip one peer engine on one "
            "scenario, so the row will exist and must not be read as an "
            "ordered row without checking its resolved parameters.",
            "CPU only. CatBoost's Ordered/Plain resolution is a CPU-side "
            "default (task_type CPU, in CATBOOST_LEFT_AT_STOCK), and this "
            "row count sits below the size at which this library sends work "
            "to the accelerator at all, so a GPU row here would be a "
            "mojotrees-internal comparison at a size chosen for somebody "
            "else's threshold.",
        ],
    ),
    "sparse_highdim": _scenario(
        task="binary",
        objective="binary",
        title="High-dimensional sparse",
        dataset="rcv1_train_binary",
        generator="sparse_highdim",
        generator_sizes={
            "smoke": {"n_rows": 5_000, "n_features": 2_000, "nnz_per_row": 30},
            "standard": {"n_rows": 100_000, "n_features": 50_000, "nnz_per_row": 60},
            "large": {"n_rows": 200_000, "n_features": 500_000, "nnz_per_row": 80},
        },
        primary_metric="auc",
        notes=(
            "CSC on both sides, never densified. The interesting cost is "
            "per feature rather than per row, so this is the scenario that "
            "separates a per-feature histogram loop that skips absent "
            "entries from one that does not."
        ),
        caveats=[
            "mojotrees trains on sparse input through the estimator's CSC "
            "path, which does not take an eval_set and reports device 'cpu' "
            "whatever the device parameter says. So this scenario is CPU "
            "only on both sides and has no early stopping.",
            "The estimator bins inside fit on this path, so binning time "
            "cannot be separated from training time for mojotrees here. The "
            "record carries binning_s = null with that reason, rather than "
            "an estimate.",
        ],
    ),
}


#: A scenario that is SPECIFIED AND NOT BUILT, held outside `SCENARIOS` on
#: purpose.
#:
#: Text columns are the third shape this campaign's categorical work needs
#: and the only one whose INPUT CONTRACT does not exist yet. `lane/text-
#: features` is live and has not landed anything, so the honest state of
#: this scenario is a written design with the unknowns named, not a
#: half-registered entry that fails a self-check or, worse, runs and
#: measures something nobody specified. `selfcheck.check_pending_scenarios`
#: fails if this is ever merged into `SCENARIOS` without the three blockers
#: below being closed, which is what makes "specified but not built" a state
#: the harness enforces rather than a comment somebody has to notice.
#:
#: WHAT IT WAITS ON. Three things, and none of them is a scenario decision:
#:
#: 1. **How a text column reaches an engine at all.** Every scenario here
#:    hands all engines ONE float64 matrix, and text is not a float64
#:    column. That is the same wall `categorical_missing` and
#:    `high_cardinality_categorical` hit for integer categories -- see
#:    `CATBOOST_SCENARIO_SUPPORT` -- and text is strictly harder, because
#:    CatBoost's `text_features` takes raw strings while LightGBM has no
#:    text feature type at all and would need the tokenization done for it.
#: 2. **Who tokenizes.** If each library tokenizes, the comparison is of
#:    three tokenizers and not of three trainers, which is the defect
#:    `categorical_missing`'s second caveat already names for category
#:    codes. If the harness tokenizes once and hands over a bag-of-words
#:    matrix, then this scenario IS `sparse_highdim` with a different
#:    generator and does not need to exist. The design below assumes the
#:    harness tokenizes once and hands over token-id sequences, which is a
#:    third option neither engine takes today, and that assumption is the
#:    substance of what `lane/text-features` has to decide.
#: 3. **What `worker.py` digests.** `data_digest` hashes
#:    `np.ascontiguousarray(piece).tobytes()`. A numpy object array of
#:    Python strings has no such bytes, so the guarantee that all engines
#:    saw identical data -- the guarantee every other number in a row rests
#:    on -- does not currently extend to a text column. A digest rule for
#:    text has to be decided before a text row can be compared at all.
#:
#: Everything below the blockers is a proposal, and it is written down now
#: so that the lane that lands the contract has a target rather than a blank
#: page. Nothing here is measured and no number in it is anything but a
#: design choice.
TEXT_SCENARIO_PENDING = {
    "id": "text_features",
    "status": "specified, not built",
    "blocked_on": (
        "lane/text-features: the input contract for a text column does not "
        "exist. See the three numbered blockers above this dict"
    ),
    "spec": _scenario(
        task="binary",
        objective="binary",
        title="Text features",
        dataset=None,
        generator="text_features",
        generator_sizes={
            # Documents per tier, not rows of a matrix, because a text
            # column's cost is per token and not per row. The token budget
            # matters more than the document count and is stated with it.
            "smoke": {"n_docs": 20_000, "vocab": 5_000, "tokens_per_doc": 20},
            "standard": {"n_docs": 500_000, "vocab": 200_000, "tokens_per_doc": 40},
            "large": {"n_docs": 2_000_000, "vocab": 1_000_000, "tokens_per_doc": 40},
        },
        primary_metric="auc",
        notes=(
            "A binary target driven by a small set of signal tokens inside "
            "a Zipf vocabulary, so that most of the vocabulary is noise and "
            "the useful tokens are rare enough that a per-token statistic "
            "is a real estimate rather than a lookup. The vocabulary sizes "
            "are chosen the way HIGH_CARDINALITY_LEVELS is: a token is a "
            "category, and the rule that governs it is rows per level."
        ),
        caveats=[
            "NOT BUILT. There is no generator named 'text_features' in "
            "generators.py and no loader, and registering this dict in "
            "SCENARIOS without one fails selfcheck.check_registry.",
            "The engines are unknown, not defaulted. LightGBM has no text "
            "feature type; CatBoost has text_features and would be the only "
            "engine running its own text algorithm, which is the shape of "
            "problem CATBOOST_UNMATCHABLE exists to record; mojotrees's "
            "side does not exist yet. A support decision per engine is part "
            "of what lane/text-features has to produce, and until it does, "
            "the engines list here is a placeholder and not a decision.",
            "No threshold. thresholds.json has no entry for this scenario "
            "and must not get a speculative one: a gate written before "
            "anybody knows what the two implementations differ by is a "
            "number chosen to pass.",
        ],
    ),
}


def resolve(scenario_id, tier="standard", variant="auto"):
    """The scenario dict with its tier sizes filled in.

    `variant` is "real", "synthetic", or "auto". "auto" prefers the real
    dataset when it is present and pinned, and falls back to the generator
    otherwise; the runner records which one it got, and a fallback is never
    silent.
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; known: {', '.join(sorted(SCENARIOS))}"
        )
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    if variant not in ("real", "synthetic", "auto"):
        raise ValueError(f"unknown variant {variant!r}")

    spec = copy.deepcopy(SCENARIOS[scenario_id])
    spec["id"] = scenario_id
    spec["tier"] = tier
    spec["variant_requested"] = variant
    spec["generator_kwargs"] = spec.get("generator_sizes", {}).get(tier, {})
    return spec


def shared_params(spec, extra=None):
    """The training parameters for a scenario, in canonical names, before
    either engine's translation."""
    params = dict(BASE_PARAMS)
    params.update(spec.get("params", {}))
    params["objective"] = spec["objective"]
    if extra:
        params.update(extra)
    return params


def lightgbm_params(spec, threads, extra=None):
    """`shared_params` translated into a LightGBM parameter dict, plus the
    comparator.

    `extra` carries what the scenario cannot know, which after the C9
    change is `num_class` and nothing else. It used to carry
    `bin_construct_sample_cnt` at the training row count, injected by both
    engine adapters, which forced LightGBM to fit its bin edges from every
    row while mojotrees fit them from a 200000-row subsample. Both
    injection sites are gone. A caller that puts it back is pinning the
    comparator to a fit mojotrees is not doing, in the direction that
    flatters us, so the two binning parameters are refused by name here
    rather than merely not passed. A refusal is the only form of this rule
    that survives somebody adding a third call site.
    """
    for refused in ("bin_construct_sample_cnt", "min_data_in_bin"):
        if refused in (extra or {}):
            raise ValueError(
                f"{refused} was passed to the comparator. Both are stock in "
                f"{COMPARATOR_LABEL} and pinning either compares two "
                "different binnings: see LIGHTGBM_STOCK_DEFAULTS and C9."
            )
    # `lambda_l2` joins them as of 2026-08-16. It was pinned to 1.0 on both
    # sides for as long as mojotrees's default was 1.0, and the pin came out
    # with the default. Setting it here again puts the comparator back on a
    # non-stock regularizer under a label that says stock.
    if "lambda_l2" in (extra or {}):
        raise ValueError(
            "lambda_l2 was passed to the comparator. It is stock (0.0) in "
            f"{COMPARATOR_LABEL} and in mojotrees, and pinning it compares "
            "an arm labelled stock against a regularizer no default "
            "produces: see LIGHTGBM_STOCK_DEFAULTS and the module docstring."
        )
    shared = shared_params(spec, extra)
    params = {
        "objective": shared["objective"],
        "num_leaves": shared["num_leaves"],
        "max_depth": shared["max_depth"],
        "learning_rate": shared["learning_rate"],
        "min_data_in_leaf": shared["min_data_in_leaf"],
        "min_sum_hessian_in_leaf": shared["min_child_hess"],
        # No `lambda_l2`: stock on both sides since 2026-08-16, so it is
        # recorded in LIGHTGBM_STOCK_DEFAULTS rather than passed.
        "lambda_l1": shared["lambda_l1"],
        "max_bin": shared["max_bin"],
        "use_missing": shared["use_missing"],
        "num_threads": int(threads),
    }
    params.update(LIGHTGBM_ALIGNMENT)
    for key in CATEGORICAL_PARAMS:
        if key in shared:
            params[key] = shared[key]
    if spec["task"] == "ranking":
        params["lambdarank_truncation_level"] = shared["lambdarank_truncation_level"]
        params["sigmoid"] = shared["sigmoid"]
        params["lambdarank_norm"] = shared["lambdarank_norm"]
        params["eval_at"] = [shared["ndcg_eval_at"]]
    if spec["task"] == "multiclass":
        params["num_class"] = int(shared["num_class"])
    for key, value in (extra or {}).items():
        params[key] = value
    return params


def mojotrees_params(spec, device, extra=None):
    """`shared_params` translated into a mojotrees parameter dict for
    `mojotrees.train`.

    Thread count is deliberately absent. mojotrees takes it from
    MOJOTREES_NUM_WORKERS, which the runner sets in the environment before
    the extension is imported, and a parameter here would suggest there are
    two ways to set it.

    Nothing on this side answers the comparator's `deterministic`, and
    that is the point of the setting rather than an omission: mojotrees is
    reproducible across thread counts with no parameter and no cost, so
    there is nothing here to turn on.
    """
    shared = shared_params(spec, extra)
    params = {
        "objective": shared["objective"],
        "num_leaves": shared["num_leaves"],
        "max_depth": shared["max_depth"],
        "learning_rate": shared["learning_rate"],
        "min_data_in_leaf": shared["min_data_in_leaf"],
        "min_child_hess": shared["min_child_hess"],
        # No `lambda_l2`, for the same reason as on the LightGBM side: the
        # estimator's own default is LightGBM's 0.0 now, so passing it
        # would restate a default rather than align anything.
        "lambda_l1": shared["lambda_l1"],
        "device": device,
    }
    for key in CATEGORICAL_PARAMS:
        if key in shared:
            params[key] = shared[key]
    if spec["task"] == "ranking":
        params["lambdarank_truncation_level"] = shared["lambdarank_truncation_level"]
        params["sigmoid"] = shared["sigmoid"]
        params["lambdarank_norm"] = shared["lambdarank_norm"]
        params["ndcg_eval_at"] = shared["ndcg_eval_at"]
    if spec["task"] == "multiclass":
        params["num_class"] = int(shared["num_class"])
    for key, value in (extra or {}).items():
        if key in ("num_class", "n_classes"):
            continue
        params[key] = value
    return params


def dataset_params(spec):
    """The parameters that belong to the binning rather than the training
    run. Both libraries take these on their Dataset, and both reject them
    on `train`, so they are separated here rather than at the call site."""
    shared = shared_params(spec)
    return {"max_bin": shared["max_bin"], "use_missing": shared["use_missing"]}


#: The CatBoost loss for each task this harness runs. Ranking is absent and
#: that is the reason ranking has no CatBoost row: see
#: CATBOOST_SCENARIO_SUPPORT.
CATBOOST_LOSS = {
    "regression": "RMSE",
    "binary": "Logloss",
    "multiclass": "MultiClass",
}


#: Scenarios where the CatBoost arm runs, but only up to a tier. Distinct
#: from `CATBOOST_SCENARIO_SUPPORT`, which answers "can this arm run this
#: problem at all"; this answers "at what size does running it stop being a
#: measurement and start being a timeout".
#:
#: A cap is not a quality judgement about CatBoost and it is not a skip for
#: convenience. It exists because an arm that times out is an *infrastructure
#: failure*, and `run.py` reports infrastructure failure by refusing to give
#: the run a quality verdict at all -- so one uncapped peer cell can take the
#: exit code of a matrix whose comparator rows all succeeded. Declaring the
#: bound up front turns that into a skip with a reason, which is a result.
CATBOOST_TIER_CAP = {
    "sparse_highdim": (
        "smoke",
        "the standard tier is 100,000 rows by 50,000 features and the large "
        "tier is 200,000 by 500,000. CatBoost builds borders per feature and "
        "grows symmetric trees over the full feature set, and neither shape "
        "has been shown to complete here inside any timeout this harness "
        "sets. The smoke tier, 5,000 by 2,000, is where the CatBoost row on "
        "this scenario is a measurement rather than a timeout, so that is "
        "where it runs. This is a declared bound, not an observed limit: "
        "nobody has timed CatBoost at the standard tier to find out how far "
        "over it goes",
    ),
    "ordered_boosting_small": (
        "standard",
        "a cap on LABELLING rather than on cost, and the only one of that "
        "kind in this dict. This scenario's row count is chosen to sit on "
        "the Ordered side of CatBoost's own Ordered-versus-Plain threshold, "
        "and ORDERED_BOOSTING_ROWS records how little of that threshold is "
        "verified: the rule may not be a row count at all, and this file "
        "already carries a reading of Plain at 20,000 rows. The large tier "
        "is the one tier that changes an input the rule might take -- it "
        "holds rows at 50,000 and raises features from 30 to 100 -- so a "
        "CatBoost row there could resolve to Plain while the standard row "
        "resolves to Ordered, and the two would sit in one column under one "
        "heading. Capped at standard until the rule is read out of "
        "CatBoost's source. This is a declared bound and not an observed "
        "limit in the strongest sense: nobody has run the large tier to see "
        "which way it resolves, and the cap exists so that nobody quotes a "
        "row without having found out",
    ),
}

#: Tiers in increasing size, so a cap can be compared rather than matched.
TIER_ORDER = ("smoke", "standard", "large")


#: Every engine name that is a CatBoost-shaped peer arm, in one place.
#:
#: This exists because the same tuple was written out by hand in three files
#: -- the scenario engine lists here, the tier cap in `run.py`, and the
#: wiring check in `selfcheck.py` -- and a fourth arm added to two of the
#: three would run at an uncapped tier, or be capped and never checked, with
#: nothing failing. `CATBOOST_TIER_CAP` bounds a scenario; this bounds who
#: the bound applies to.
#:
#: `mojotrees_catboost_mode` is deliberately NOT here. It is us shaped toward
#: CatBoost, not CatBoost, so it neither takes CatBoost's tier caps nor
#: carries CatBoost's parameters. It is a peer arm by `ENGINE_ARM` and that
#: is the only list it belongs to.
CATBOOST_ENGINES = ("catboost", "catboost_lossguide")

#: The peer arms as a group, for callers that mean "anything reported beside
#: the comparator rather than as it".
PEER_ENGINES = CATBOOST_ENGINES + ("mojotrees_catboost_mode",)


def catboost_tier_ok(scenario, tier):
    """(ok, reason). Whether the CatBoost arm runs this scenario at `tier`.

    Separate from `catboost_supports` on purpose. Support is a property of
    the problem and does not vary with size; this varies only with size. A
    caller that asks one and not the other gets a wrong answer in one
    direction each, so `run.py` asks both.
    """
    scenario_id = scenario["id"] if isinstance(scenario, dict) else scenario
    capped = CATBOOST_TIER_CAP.get(scenario_id)
    if capped is None:
        return True, None
    max_tier, reason = capped
    if TIER_ORDER.index(tier) <= TIER_ORDER.index(max_tier):
        return True, None
    return False, (
        f"the CatBoost arm on {scenario_id} is capped at the {max_tier} "
        f"tier and this is {tier}: {reason}"
    )


def catboost_supports(scenario):
    """(runs, reason). Whether the CatBoost arm runs this scenario.

    Takes a scenario id or a resolved spec. A scenario with no entry at all
    is refused rather than assumed to run: an unlisted scenario is one
    nobody decided about, and defaulting it to "yes" is how an engine ends
    up in a table on a problem it was never checked against.
    """
    scenario_id = scenario["id"] if isinstance(scenario, dict) else scenario
    if scenario_id not in CATBOOST_SCENARIO_SUPPORT:
        return False, (
            f"{scenario_id} has no entry in CATBOOST_SCENARIO_SUPPORT, so "
            "nobody has decided whether the CatBoost arm can run it"
        )
    reason = CATBOOST_SCENARIO_SUPPORT[scenario_id]
    return (reason is None), reason


def catboost_params(spec, threads, extra=None):
    """`shared_params` translated into a CatBoost parameter dict.

    Only three things reach CatBoost that are not the problem statement:
    `CATBOOST_ALIGNMENT`, the thread count, and the two matched parameters
    in `CATBOOST_MATCHED`. Everything else is CatBoost's own default and is
    recorded in `CATBOOST_LEFT_AT_STOCK` rather than passed.

    The refusal list is the point of this function existing rather than the
    adapter building the dict itself. `bin_construct_sample_cnt` at the
    training row count made the LightGBM comparator do strictly more binning
    work than mojotrees did, in our favor, and it was caught only after a
    ratio had been published. `CATBOOST_REFUSED_PARAMS` is the same defect
    in CatBoost's vocabulary -- `border_count`, `max_bin`,
    `dev_max_subset_size_for_build_borders`, `used_ram_limit` and the
    leaf-population and sampling knobs -- and every one of them is refused
    by name here, because a refusal is the only form of this rule that
    survives somebody adding a third call site.

    `extra` carries what the scenario cannot know, which is `num_class` and
    nothing else.
    """
    for refused, why in CATBOOST_REFUSED_PARAMS.items():
        if refused in (extra or {}):
            raise ValueError(
                f"{refused} was passed to the CatBoost peer arm. It is stock "
                f"in {CATBOOST_ARM_LABEL}: {why}. See "
                "CATBOOST_REFUSED_PARAMS and CATBOOST_LEFT_AT_STOCK."
            )
    task = spec["task"]
    if task not in CATBOOST_LOSS:
        raise ValueError(
            f"the CatBoost peer arm has no loss for task {task!r}: "
            + str(CATBOOST_SCENARIO_SUPPORT.get(spec.get("id"), "unlisted"))
        )
    shared = shared_params(spec, extra)
    params = {
        "loss_function": CATBOOST_LOSS[task],
        # Matched, and read from BASE_PARAMS rather than restated, so the
        # three engines cannot be asked for different budgets by an edit to
        # one dict.
        "iterations": int(shared[CATBOOST_MATCHED["iterations"]]),
        "learning_rate": float(shared[CATBOOST_MATCHED["learning_rate"]]),
        # The same quantity MOJOTREES_NUM_WORKERS and num_threads carry on
        # the other two arms, set from one number by the runner.
        "thread_count": int(threads),
    }
    params.update(CATBOOST_ALIGNMENT)
    if task == "multiclass":
        params["classes_count"] = int(shared["num_class"])
    for key, value in (extra or {}).items():
        if key in ("num_class", "n_classes"):
            continue
        params[key] = value
    return params


def mojotrees_catboost_mode_params(spec, device, extra=None):
    """The "us in CatBoost mode" arm: `mojotrees_params` with
    `MOJOTREES_CATBOOST_MODE` applied over the shared defaults.

    Applied as a scenario-level override rather than as a second translator,
    so this arm and the plain one go through exactly the same code and a
    reader diffing two records sees `MOJOTREES_CATBOOST_MODE` and nothing
    else.

    What this arm is NOT: a claim that mojotrees can be made into CatBoost.
    It cannot. This is depthwise at depth 6 -- a symmetric policy now exists
    in the Mojo package but `python/mojotrees/sklearn.py` does not expose it,
    and this harness only reaches what that file validates -- and it does no
    row sampling, where CatBoost's default MVS takes 80 percent of the rows
    per tree. Both are in `CATBOOST_UNMATCHABLE` and both travel with the
    record.

    The override is applied to the resolved dict rather than to the
    scenario's `params`, and that is a correction rather than a shortcut.
    `mojotrees_params` builds an explicit dict and copies through only the
    categorical, ranking and multiclass blocks, so a scenario-level
    `grow_policy` is dropped on the floor: the first version of this
    function set it that way and produced an arm with CatBoost's depth and
    mojotrees's growth, which is neither of the two things the row claims to
    compare. `selfcheck.check_catboost_arm` caught it by diffing the two
    resolved dicts against `MOJOTREES_CATBOOST_MODE`, and that check is the
    reason this comment can be specific.
    """
    params = mojotrees_params(spec, device, extra)
    params.update(MOJOTREES_CATBOOST_MODE)
    return params


# The peer arms are added to the scenarios they can run, and a scenario that
# cannot run them says why in `catboost_arm_block()["scenarios_not_run"]`
# rather than simply not appearing. Written as a loop over
# CATBOOST_SCENARIO_SUPPORT so that a scenario added without a support
# decision fails `selfcheck.check_catboost_arm` instead of silently
# defaulting to one.
for _scenario_id, _reason in CATBOOST_SCENARIO_SUPPORT.items():
    if _reason is None:
        SCENARIOS[_scenario_id]["engines"] = list(
            SCENARIOS[_scenario_id]["engines"]
        ) + list(PEER_ENGINES)
del _scenario_id, _reason
