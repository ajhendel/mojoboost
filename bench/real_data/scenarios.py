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

Threads are matched by MEANING and not by the integer, which is a
correction made on 2026-08-17. LightGBM reads `num_threads`, which sizes a
pool its own scheduler then feeds; mojotrees reads MOJOTREES_NUM_WORKERS,
which is not a thread count at all but a fixed block count that replaces
`parallel.plan_tasks`'s rule. Exporting one integer into both therefore
moved exactly one engine off its own defaults, and the harness called the
result like-for-like. A cell at the machine's own core count now leaves
MOJOTREES_NUM_WORKERS UNSET so our planner runs, which is what a user gets
and what the peers' thread count asks of them; a cell below it pins the
count and the record says so. See `run.mojotrees_workers`.

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
import hashlib
import struct

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
#: The one seed constant every arm runs from.
#:
#: It was 190019 on the LightGBM arm and 190019 on the CatBoost arm and
#: NOTHING on the two mojotrees arms, written out twice as a literal, until
#: 2026-08-16. `mojotrees_params` passed no seed at all, on the argument in its
#: own docstring that mojotrees is reproducible across thread counts with no
#: parameter. That argument is about REPEATABILITY and it was never an argument
#: for the three arms sharing a stream.
#:
#: It cost nothing while our arms were deterministic and it started costing
#: something the day `MOJOTREES_CATBOOST_MODE` gained `bootstrap_type=MVS`,
#: whose draw is keyed on a seed (`sampling._mvs_stream`). So the constant is
#: named once here and referenced by all three translators.
#:
#: **What this does NOT align, stated because the obvious reading is wrong.**
#: `random_state` sets every seed in `sklearn.py`'s `_SEEDS` plus
#: `bootstrap_seed`, and on the scenarios this suite runs only `bootstrap_seed`
#: is live: no arm bags rows, subsamples features, runs GOSS, drops trees or
#: grows extra trees. It also does not touch the split-score noise. That stream
#: is keyed on `ExtraTreeParams.random_strength_seed`, which
#: `tree_parameters_extra.mojo::ExtraTreeParams.__init__` sets to
#: `DEFAULT_RANDOM_STRENGTH_SEED` and
#: which no Python surface can set: `bindings/_mojotrees.mojo` parses
#: `random_strength` and no seed beside it. So the CatBoost-mode arm's noise
#: runs from a native constant whatever this value is, and
#: `CATBOOST_UNMATCHABLE['split_scoring']` remains true for a second reason now:
#: not only are the two streams different by construction, ours is not even
#: reachable from the harness.
SHARED_SEED = 190019

BASE_PARAMS = {
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_data_in_leaf": 20,
    "min_child_hess": 1e-3,
    "lambda_l1": 0.0,
    # `lambda_l2` is named here, at LightGBM's stock 0.0, as of 2026-08-17.
    # It was ABSENT before that, and absence was correct for exactly as long
    # as our own default was also 0.0: both engines then resolved their own
    # stock value and landed on the same number.
    #
    # On 2026-08-17 our default moved to 1.0 (a declared divergence, see
    # `python/mojotrees/sklearn.py` `_LAMBDA_L2` and `STOCK_DIVERGENCES` in
    # `tools/check_parity.py`). From that moment an absent key stops meaning
    # "both at stock" and starts meaning "each at its own different value",
    # which would have made the lightgbm/mojotrees MIRROR PAIR quietly stop
    # being a mirror while still being labelled one. That is the same defect
    # found on 2026-08-16, in the same parameter, running the other
    # direction, and it is why this key is now explicit rather than implied.
    #
    # So this pair measures growth policy and implementation at a MATCHED
    # regularizer. Our shipped default is a separate row, and the two must
    # not be read as the same arm.
    "lambda_l2": 0.0,
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
#: v3 as of 2026-08-17, and this bump is unlike v2. `lambda_l2` was ADDED to
#: `BASE_PARAMS` at 0.0, which is the value LightGBM already resolved it to
#: from its own stock config, so **the comparator's resolved configuration is
#: unchanged and a v2 LightGBM number and a v3 LightGBM number are the same
#: measurement.** The bump is required anyway by the rule above, which names
#: "which keys it holds", and the rule is right to be that conservative: what
#: actually changed is not the comparator but the RELATIONSHIP between this
#: arm and our shipped default. Until 2026-08-17 our mojotrees arm on
#: `BASE_PARAMS` was also our shipped default. It is not any more, because the
#: shipped `lambda_l2` is 1.0 and this arm pins 0.0 to stay a mirror. Anyone
#: reading a v2 table as "this is what mojotrees ships" was right then and
#: would be wrong now, and that is exactly the confusion a version exists to
#: prevent.
COMPARATOR_VERSION = 3
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
    "seed": SHARED_SEED,
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
# Both at a matched TREE COUNT and at each engine's OWN learning rate. That
# second half changed on 2026-08-16 and it is the largest change this arm has
# ever taken, so it is stated here before anything else in the section.
#
# CatBoost derives `learning_rate` from the iteration count and the dataset
# when it is not given one. Measured on this machine on 2026-08-16 at 20,000
# rows by 20 features, CatBoost 1.2.10 resolved it to 0.5 at 2 iterations,
# **0.4273 at 100** and 0.06573 at 1000. This harness used to PIN the rate to
# BASE_PARAMS['learning_rate'] = 0.1 on the CatBoost side so that the three
# arms ran one rate; it does not any more. The measurements above stay in the
# file because they are now the EXPLANATION of the arm rather than the
# argument against it: they are why the CatBoost column trains at roughly
# 0.43 while the LightGBM comparator trains at 0.1, which a reader will
# otherwise take for a bug.
#
# The two comparisons this is choosing between, because both are defensible
# and only one of them can be the row:
#
#   like-for-like on model shape     both engines pinned to one rate, so a
#                                    metric difference is tree shape and
#                                    regularization. What this arm was.
#   as shipped                       each engine at its own resolved default
#                                    at a matched tree budget, so the row is
#                                    what a user gets out of the box. What
#                                    this arm IS, by Andrew's decision on
#                                    2026-08-16.
#
# READ THIS BEFORE QUOTING THE CATBOOST ACCURACY NUMBER. At a fixed 100-tree
# budget an engine training at 0.4273 walks much further down its loss curve
# than one training at 0.1, so CatBoost's accuracy column may improve
# substantially against the previous arm. That movement is a learning-rate-
# times-budget interaction and is NOT evidence that CatBoost's engine is more
# accurate than either arm beside it. See CATBOOST_DELIBERATE_DIVERGENCE.

#: The peer arm's identity. Read beside `comparator_id()`, never in place of
#: it. Bump the version when `CATBOOST_ALIGNMENT`, `CATBOOST_MATCHED` or
#: `CATBOOST_DELIBERATE_DIVERGENCE` changes in a way that makes two CatBoost
#: numbers non-comparable.
#:
#: The id itself changed on 2026-08-16, which is rarer than a version bump and
#: is deliberate: `cb-default@v1` pinned CatBoost's learning rate and this arm
#: does not, so the two ids compute materially different models and no reader
#: should have to notice a version digit to tell them apart. Every published
#: `cb-default@v1` number is superseded; see `CATBOOST_ARM_SUPERSEDES`.
CATBOOST_ARM_ID = "cb-shipped"
CATBOOST_ARM_VERSION = 1
CATBOOST_ARM_LABEL = "CatBoost defaults (as shipped), at a matched tree count"
CATBOOST_ARM_SUPERSEDES = (
    "cb-default@v1, which pinned learning_rate to BASE_PARAMS['learning_rate']"
    " = 0.1 on the CatBoost side. Every CatBoost number taken under that id "
    "is SUPERSEDED and not merely re-labelled: CatBoost resolves its own rate "
    "to roughly 0.43 at 100 iterations on a 20,000 by 20 shape, so the model "
    "this arm fits now is a different model. bench/results/"
    "COMPARISON_RUN_2026-08-16.md carries cb-default@v1 numbers throughout "
    "and none of its CatBoost rows survives this change"
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
    "sparse_features_conflict_fraction. The categorical entries -- "
    "one_hot_max_size, max_ctr_complexity, counter_calc_method, "
    "ctr_target_border_count, store_all_simple_ctr and has_time -- were "
    "read separately on 2026-08-16 off two-iteration fits that HAD "
    "cat_features, because a fit with no categorical column is not evidence "
    "about what CatBoost resolves for one. Three shapes and two losses, "
    "agreeing on every value"
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
#: The matched parameter is deliberately NOT here. It comes from
#: `BASE_PARAMS` through `CATBOOST_MATCHED`, so that "matched tree count" is a
#: structural property of the translation rather than a number copied into a
#: second dict where it can drift.
CATBOOST_ALIGNMENT = {
    "allow_writing_files": False,
    "logging_level": "Silent",
    "random_seed": SHARED_SEED,
}

#: The ONE shared parameter that is forced onto the CatBoost arm, and the
#: `BASE_PARAMS` key it is taken from. `selfcheck.check_catboost_arm` proves
#: the resolved dict carries the same value the other two engines get.
#:
#: `learning_rate` left this dict on 2026-08-16. It is now a declared
#: divergence rather than a match; see `CATBOOST_DELIBERATE_DIVERGENCE`.
CATBOOST_MATCHED = {
    "iterations": "n_estimators",
}

#: Where this arm DELIBERATELY does not match, with the decision and the
#: reason. The opposite of `CATBOOST_UNMATCHABLE`, which is about differences
#: no parameter can close: everything here COULD be closed by passing a value
#: and is not, on purpose.
#:
#: A separate table because the two failure modes are opposite. An
#: unmatchable difference is a fact about the engines; a deliberate divergence
#: is a choice about the row, and a choice that is not written down is
#: indistinguishable from an oversight six weeks later.
CATBOOST_DELIBERATE_DIVERGENCE = {
    "learning_rate": {
        "catboost": (
            "CatBoost's own resolution, read back per fit through "
            "get_all_params(). Roughly 0.4273 at 100 iterations on a 20,000 "
            "by 20 shape; the exact value is dataset-dependent and is "
            "recorded per run, never asserted here"
        ),
        "lightgbm": "BASE_PARAMS['learning_rate'] = 0.1, unchanged",
        "mojotrees": "BASE_PARAMS['learning_rate'] = 0.1, unchanged",
        "mojotrees_catboost_mode": (
            "CatBoost's resolved value for the same cell, taken from the "
            "read-back. This arm's whole claim is 'us in CatBoost's shape', "
            "and a hand-written 0.1 beside a CatBoost fit at 0.43 was that "
            "claim being false in the single parameter that moves a metric "
            "most"
        ),
        "decision": (
            "Andrew, 2026-08-16. Each engine runs its own shipped default at "
            "a matched tree count. The alternative -- both engines pinned to "
            "one rate, which is what this arm did until now -- is a "
            "like-for-like on model shape and is a different row, not a "
            "better version of this one"
        ),
        "measured": {
            "2_iterations": 0.5,
            "100_iterations": 0.4273,
            "1000_iterations": 0.06573,
            "shape": "20,000 rows by 20 features, catboost 1.2.10, "
                     "task_type CPU, RMSE, this machine, 2026-08-16",
        },
        "how_to_misread_it": (
            "as an engine claim. At a fixed 100-tree budget an engine "
            "training at 0.4273 converges much further than one training at "
            "0.1, so CatBoost's accuracy column can improve substantially "
            "against cb-default@v1 without a single line of CatBoost running "
            "faster or fitting better. The difference is a learning rate "
            "times a budget. A table that prints this column beside the "
            "comparator MUST carry this sentence or it is inviting the "
            "misreading"
        ),
        "how_to_read_it": (
            "as an out-of-the-box claim, which is what it is. A user who "
            "installs CatBoost and asks for 100 trees gets roughly 0.43 and "
            "a user who installs LightGBM and asks for 100 trees gets 0.1. "
            "That is a real difference between the products and this row "
            "measures it"
        ),
    },
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
    # `learning_rate` is NOT here any more, and its absence is the point of
    # cb-shipped. It is not passed, so it is not a deviation from CatBoost's
    # default: it IS CatBoost's default. The reasoning that used to live here
    # moved to CATBOOST_DELIBERATE_DIVERGENCE, with the measurements intact,
    # because a reader who sees a CatBoost column at 0.43 beside a LightGBM
    # column at 0.1 needs an explanation and not a deletion.
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
    # The categorical block, added 2026-08-16 when the arm gained a
    # categorical path and these stopped being settings nothing reached.
    # Read the same way as everything above -- get_all_params() after a
    # two-iteration fit -- but on a fit that actually HAD cat_features, at
    # 500 rows with two and with five categorical columns and at 5,000 rows
    # with five, on both Logloss and RMSE. Every value below was the same in
    # all of them.
    #
    # `max_ctr_complexity` is the entry to look at twice. CatBoost's
    # documentation gives its CPU default as 4; the library resolved 1. The
    # reading is the one recorded, and CATBOOST_UNMATCHABLE
    # ['ctr_combinations'] carries what follows from it, which is that this
    # arm builds no CTR feature combinations at its own defaults.
    "one_hot_max_size": 2,
    "max_ctr_complexity": 1,
    "counter_calc_method": "SkipTest",
    "ctr_target_border_count": 1,
    "store_all_simple_ctr": False,
    "has_time": False,
}

#: The CatBoost parameters that have NO static value, because CatBoost derives
#: them from the fit. They are absent from `CATBOOST_LEFT_AT_STOCK` on purpose:
#: a constant in that table is a claim, and a claim about a derived value is
#: false on every shape but the one it was read on.
#:
#: The only way to know one of these is to read it back off the fitted model,
#: which is what `engines.CatBoostEngine.run` does through
#: `CatBoost.get_all_params()` and records as `engine_resolved_params`. Every
#: consumer of a value in this table has to come from there or refuse.
#:
#: `boosting_type` is deliberately NOT here even though CatBoost documents it
#: as data-dependent. See `ORDERED_BOOSTING_ROWS`: the only default that
#: installs `Ordered` is inside a `TaskType == GPU` branch, so on CPU it is
#: `Plain` at every row count and it is a constant, not a reading.
CATBOOST_RESOLVED_PER_FIT = {
    "learning_rate": (
        "derived from the iteration count and the dataset. 0.5 at 2 "
        "iterations, 0.4273 at 100 and 0.06573 at 1000, on 20,000 rows by 20 "
        "features, catboost 1.2.10, this machine, 2026-08-16. Those three "
        "numbers are a demonstration that it MOVES, not a table to look a "
        "value up in: the row count and the feature count are both inputs, so "
        "a scenario with a different shape resolves a different rate. "
        "cb-shipped no longer passes it, so every cell's value comes from "
        "that cell's own read-back. See CATBOOST_DELIBERATE_DIVERGENCE"
    ),
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
        "CLOSED 2026-08-16, and recorded rather than deleted because it was "
        "the largest caveat on this comparison for the whole campaign. "
        "CatBoost grows symmetric trees of depth 6 where every node at a "
        "level shares one split; mojotrees now does too, and the "
        "CatBoost-mode arm asks for it by name. The policy landed in wave 4 "
        "and spent the intervening time unreachable -- built, tested, merged, "
        "and refused by the estimator's grow_policy validator, which listed "
        "two of the three values. A symmetric tree is strictly more "
        "constrained than a depthwise one at the same depth, so the "
        "difference this entry used to describe was expected to cost CatBoost "
        "accuracy and save it time; neither side of that was ever measured "
        "and now neither needs to be"
    ),
    "row_sampling": (
        "CLOSED 2026-08-16 ON THE SINGLE-OUTPUT DENSE SCENARIOS, AND STILL "
        "OPEN ON TWO. This line read 'SINGLE-OUTPUT DENSE CPU' until "
        "2026-08-17; the word CPU came out because the dense single-output "
        "GPU arm honors the bundle too, which the last paragraph traces. "
        "CatBoost's default bootstrap_type is MVS with "
        "subsample 0.8, so it subsamples rows on every tree, weighted by "
        "gradient magnitude rather than uniformly, and this arm used to see "
        "every row. The sampler was built from CatBoost's source and wired "
        "into boosting.train's round loop a wave earlier; what was missing "
        "was the edge from Python, and that edge now exists -- "
        "bindings/_mojotrees.mojo parses bootstrap_type into a "
        "sampling.BootstrapParams, model.fit and trainset.train_dataset "
        "forward it to boosting.train, and MOJOTREES_CATBOOST_MODE asks for "
        "it by name. dense_regression, imbalanced_binary, "
        "ordered_boosting_small and high_cardinality_categorical are matched "
        "on this axis for the first time. "
        "WHAT USED TO BE FILED AS OPEN HERE WAS WRONG IN THREE PLACES, "
        "CORRECTED 2026-08-17 AGAINST THE SOURCE. The paragraph this "
        "replaces read that neither the multiclass trainers "
        "(boosting.train_multiclass, train_multiclass_gpu) nor the sparse "
        "ones (boosting_sparse.train_sparse) call sampling.bootstrap_round, "
        "so all of them refuse an enabled bundle by name, and that 'the GPU "
        "is a third case and is refused for the same reason. train_gpu "
        "takes no bundle'. Every one of those three trainers named as "
        "incapable now draws. (1) boosting.train_multiclass draws once per "
        "round through boosting.mojo::_boost_rounds_multiclass, shared "
        "by every class's tree, and trainset.train_dataset_multiclass takes "
        "and forwards the bundle on its CPU arm. (2) "
        "boosting_sparse.train_sparse and train_multiclass_sparse both call "
        "bootstrap_round in the place the dense loops call it "
        "(boosting_sparse.mojo::train_sparse and "
        "boosting_sparse.mojo::train_multiclass_sparse) and trainset.train_dataset "
        "forwards it to the sparse arm. (3) train_gpu TAKES a bundle. MVS "
        "resolves to gpu_fused_round.ROUND_MVS_HOST_MAGNITUDES, which under "
        "AUTO routes to the host-gradient arm where sampling.bootstrap_round "
        "draws exactly and the trees still grow on the device "
        "(train_gpu.mojo::device_gradients, trainset.mojo::train_dataset, "
        "model.mojo::fit); the Bayesian draw reaches the device round "
        "through GpuObjectiveState's weight plane instead. So MVS is HONORED "
        "on a dense single-output GPU fit rather than dropped or refused, "
        "and what it costs is the device derivative kernel and not the "
        "sampler, which is why train_gpu.gpu_bootstrap_resolution exists to "
        "record which of the two arms a row came from. The observation that "
        "falsified the old sentence is a run rather than a reading. The GPU "
        "CatBoost-mode arm took three repeats on 2026-08-17 at rmse "
        "0.307693 and did not raise. "
        "WHAT IS ACTUALLY STILL OPEN, and it is two things and not three. "
        "First, the multiclass and sparse GPU trainers take no bundle "
        "(train_multiclass_gpu, train_gpu_sparse, "
        "train_multiclass_gpu_sparse), so a fit resolving to the "
        "accelerator on either shape refuses by name (model.mojo::fit_multiclass, "
        "model_sparse.mojo::fit_csc and model_sparse.mojo::fit_multiclass_csc, "
        "trainset.mojo::train_dataset_multiclass). "
        "Second, MVS on a softmax loop needs an EXPLICIT mvs_reg, because "
        "the lambda CatBoost derives reads a [dim][leaf] table that a round "
        "of K structurally different trees does not have "
        "(sampling.check_mvs_reg_is_set). MOJOTREES_CATBOOST_MODE sets no "
        "mvs_reg, so the multiclass cell still RAISES under this arm, on "
        "that ground rather than on the missing-trainer ground the old "
        "paragraph gave, and dropping the keys for multiclass is still the "
        "more faithful choice because CatBoost's own defaulting block "
        "excludes the multiclass-only losses from MVS and keeps Bayesian. "
        "sparse_highdim no longer raises on bootstrap_type at all; what "
        "stops that cell now is random_strength, which the sparse round "
        "loop computes no per-tree scale for and which the binding refuses "
        "as `not d[].is_sparse`. Both cells are skipped for this arm by "
        "MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT either way, so the "
        "scheduling decision the old paragraph demanded before the next "
        "full matrix has already been taken. NOTE that the two other "
        "places in this file which still describe those cells "
        "(MOJOTREES_CATBOOST_MODE's header comment and "
        "MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT's multiclass and "
        "sparse_highdim entries) carry the old trainer-capability reason and "
        "are owed the same correction by whoever owns them"
    ),
    "split_scoring": (
        "HALF CLOSED. CatBoost's default score_function is Cosine and this "
        "arm now asks for it, so the scoring RULE matches. What does not is "
        "random_strength's seeded noise: both engines now add it and the "
        "streams are different by construction -- CatBoost's per-document "
        "draws depend on how many queries precede a row in its thread block, "
        "reproducible only because CB_THREAD_LIMIT is a compile-time "
        "constant, and ours are keyed on (seed, tree, row). Same "
        "distribution, different numbers, and no parameter makes them the "
        "same draw"
    ),
    "leaf_population": (
        "CatBoost's min_data_in_leaf default is 1 and this harness's shared "
        "value is 20. The CatBoost arm is left at 1 because the column is "
        "'CatBoost defaults'; the CatBoost-mode mojotrees arm is set to 1 "
        "to match it, and the plain mojotrees arm stays at 20"
    ),
    "missing_values": (
        "CatBoost's nan_mode default is Min, which sends every missing "
        "NUMERIC value to the low side; LightGBM and mojotrees learn a "
        "default direction per split. On a CATEGORICAL column nan_mode does "
        "not apply at all and CatBoost has no rule to compare: it refuses "
        "the value rather than routing it. That asymmetry is why the "
        "categorical_missing scenario still has no CatBoost row, and the "
        "reason is now stated in full in "
        "CATBOOST_SCENARIO_SUPPORT['categorical_missing'] and "
        "CATBOOST_CATEGORICAL_ENCODING['missing_categories'] rather than "
        "deferred to 'a separate reason'. No scenario in this suite "
        "currently reads nan_mode against a learned direction: "
        "categorical_missing is the one that would and it does not run this "
        "arm"
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
        "CatBoost stores learning_rate as a 32-bit float, so a rate that "
        "leaves get_all_params() is a float32 widened to a double: a passed "
        "0.1 reads back as 0.10000000149011612. This mattered when the arm "
        "PINNED the rate and the claim was that the three arms ran the same "
        "number; under cb-shipped it matters for a different reason, which is "
        "that the mojotrees CatBoost-mode arm takes CatBoost's resolved rate "
        "and can only take it to float32 precision. Every comparison of the "
        "two rates therefore folds through a float32 round trip "
        "(scenarios._parity_equal) rather than asking for equality"
    ),
    "leaf_estimation_method": (
        "CatBoost resolves leaf_estimation_method to Newton and mojotrees has "
        "no such parameter. Searched on 2026-08-16 across python/mojotrees, "
        "bindings/ and docs/PARAMETER_NAMING.md: the name appears nowhere, so "
        "there is no key to set and no key to diff. What the comment at "
        "python/mojotrees/sklearn.py::_Base's docstring does say is that "
        "leaf_estimation_iterations 'keeps taking Newton steps on a leaf's "
        "own rows' through boosting._estimate_leaf_values, which is the same "
        "estimator CatBoost names -- but that is a reading of a comment about "
        "a NEIGHBOURING parameter and it is not evidence a record can carry. "
        "The honest statement is that the two engines are believed to agree "
        "here and that nothing in either resolved dict proves it. Closing "
        "this needs a leaf_estimation_method parameter on our side, or a "
        "source-level verification note in the catalog, and it has neither"
    ),
    "border_placement": (
        "CatBoost's feature_border_type resolves to GreedyLogSum, a recursive "
        "median split, and mojotrees bins equal-frequency. Neither surface "
        "has a parameter for the other's rule: feature_border_type appears "
        "nowhere in python/mojotrees, bindings/ or "
        "docs/PARAMETER_NAMING.md, searched 2026-08-16. Read this with "
        "'binning_budget' above, which is the COUNT question and is closed: "
        "the two engines produce the same number of borders and put them in "
        "different places. This entry is the placement half and it is open "
        "and unclosable by any parameter"
    ),
    "leaf_estimation_iterations": (
        "FOUND BY A RUN on 2026-08-16 and this entry is the correction of a "
        "claim this file used to make. CatBoost resolves "
        "leaf_estimation_iterations PER OBJECTIVE: 1 under RMSE and **10** "
        "under Logloss, read live off get_all_params() on dense_regression "
        "and imbalanced_binary at the smoke tier. "
        "MOJOTREES_CATBOOST_MODE_REASONS used to say the value was 'CatBoost's "
        "resolved value for the objectives this suite runs, 1 for RMSE and "
        "Logloss', which was a transcription taken on an RMSE fit and "
        "generalized to a loss it was never read on. "
        "Our side CANNOT match the Logloss value through this harness's entry "
        "point: mojotrees.train(params, Dataset) routes to "
        "trainset.train_dataset, which refuses the parameter above 1 BY NAME "
        "-- 'leaf_estimation_iterations > 1 is not implemented by a Dataset "
        "fit; it is implemented by boosting.train, boosting.train_more, "
        "boosting.train_with_valid, train_gpu.train_gpu and "
        "train_gpu.train_gpu_with_valid'. Verified by calling it, not by "
        "reading it. So on every Logloss cell CatBoost takes ten Newton steps "
        "per leaf and this arm takes one, and no parameter available here "
        "closes that. The arm keeps 1, which is right on the RMSE cells and "
        "is the only value the path accepts on the others. Closing this needs "
        "the Dataset fit to reach a round loop that estimates leaves more "
        "than once"
    ),
    "boost_from_average": (
        "FOUND BY A RUN on 2026-08-16. CatBoost resolves boost_from_average "
        "per objective too: True under RMSE and **False** under Logloss. "
        "mojotrees has no such parameter on any surface this harness reaches "
        "-- mojotrees.train forwards it into the estimator constructor, which "
        "raises '_Base.__init__() got an unexpected keyword argument "
        "boost_from_average'. Verified by calling it. LIGHTGBM_STOCK_DEFAULTS "
        "records LightGBM's own default as True, which is where the harness's "
        "previous belief that all three arms agreed came from; that belief "
        "was about LightGBM and was never checked against CatBoost on a "
        "classification loss. So on every Logloss cell CatBoost starts from "
        "zero and the other two start from the prior mean"
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
        "feature. That difference is permanent and no parameter closes it. "
        "What changed on 2026-08-16 is that it is now MEASURED rather than "
        "absent: high_cardinality_categorical runs all three arms with the "
        "same five columns declared categorical, on one canonical dataset "
        "with one digest, so the CTR and the category-set split are read "
        "against each other at five cardinalities chosen to cross both of "
        "the other engines' capacity rules -- max_cat_to_onehot at 4, "
        "max_cat_threshold's 32-category prefix cap, min_data_per_group's "
        "100-row floor, and the 254-category ceiling that "
        "src/mojotrees/categorical.mojo has and LightGBM does not. This "
        "entry stays in the unmatchable table because that is what it is: "
        "the three arms are answering the same question with three "
        "different features, and a gap between them is the algorithms and "
        "not a tuning difference. It is no longer a MISSING row, which is "
        "what it used to be. Read it with 'ctr_combinations' below, which "
        "is the half of CatBoost's categorical machinery this scenario "
        "still does not reach, and with CATBOOST_CATEGORICAL_ENCODING, "
        "which is why one digest covers two containers"
    ),
    "ctr_combinations": (
        "CatBoost's CTR feature COMBINATIONS -- a target statistic over a "
        "TUPLE of categorical values rather than over one column -- are the "
        "mechanism the double-centered two-column interaction in "
        "high_cardinality_categorical was drawn for, and at CatBoost's own "
        "defaults they DO NOT RUN. Measured, not read off the docs, and the "
        "two disagree. CatBoost documents max_ctr_complexity's default as 4 "
        "on CPU; catboost 1.2.10 resolves it to 1. Read back through "
        "get_all_params() after two-iteration fits on 2026-08-16 at 500 "
        "rows with 2 categorical columns, 500 rows with 5, and 5,000 rows "
        "with 5, on both Logloss and RMSE: 1 in every case. The readback is "
        "faithful rather than a display artifact -- passing "
        "max_ctr_complexity 2 and 4 explicitly reads back 2 and 4 -- so 1 "
        "is what CatBoost resolved, and complexity 1 means single-column "
        "CTRs only. The consequence for the record is exact: the "
        "interaction term is reachable on this arm only by stacking two "
        "splits, which is the same way LightGBM and mojotrees have to reach "
        "it, so on that term the three arms are on equal footing and the "
        "scenario's claim to read a combination feature is NOT met. Setting "
        "max_ctr_complexity would meet it and would make the column "
        "something other than CatBoost's defaults, so it is in "
        "CATBOOST_REFUSED_PARAMS instead. A combinations row is worth "
        "having and it is a differently labelled one"
    ),
}

#: How the CatBoost arm is handed a categorical column, and the argument for
#: why that is still one dataset rather than two.
#:
#: This is the field to read before quoting any CatBoost number on
#: `high_cardinality_categorical`, and it is long because the digest question
#: is the whole of whether that row means anything.
#:
#: **The problem.** `catboost/core.py:804` refuses `cat_features` on a
#: floating-point array by `dtype.kind`, not by content: "'data' is numpy
#: array of floating point numerical type, it means no categorical features,
#: but 'cat_features' parameter specifies nonzero number of categorical
#: features". No value the harness could write into a float64 array passes
#: that check, so the array has to change type. Only the categorical block
#: can -- the numeric columns are genuinely real-valued -- so the container
#: has to be mixed, and numpy has no mixed dtype. Hence a pandas frame with
#: float64 numeric columns and int64 categorical ones, columns named `f{i}`
#: positionally so `cat_features` stays a list of indices.
#:
#: **The digest decision, and it is a decision.** It is ONE dataset in two
#: encodings, and the record says which encoding each engine got. Not two
#: datasets, and not one dataset with the check quietly waived for CatBoost.
#: The argument has three steps and the third is the one that does the work:
#:
#:   1. The values are identical. Category codes are integers in
#:      `0 .. cardinality - 1`, every one of them exactly representable in
#:      float64 (the ceiling is 2**53 and the largest cardinality in this
#:      suite is 400,000), so `float64 -> int64 -> float64` is the identity
#:      on them. Checked per column per run, not argued: see
#:      `engines._catboost_categorical_frame`.
#:   2. The problem statement is identical. "These five columns are
#:      categorical" is not something CatBoost is told and the other two are
#:      not. LightGBM gets it as `categorical_feature` and mojotrees gets it
#:      as `categorical_feature` (its Python surface also accepts
#:      `categorical_features`), from the same `train["categorical_feature"]`
#:      list, on the same columns. What differs is what each engine DOES with
#:      that statement, and that difference is the comparison rather than a
#:      confound in it.
#:   3. The canonical digest is untouched and unexempted.
#:      `worker.data_digest` still hashes the float64 matrix and nothing
#:      else, every arm's record still carries the same value, and
#:      `verify.check_data_agreement` still fails the scenario if they
#:      differ. On top of that the adapter reconstructs the canonical matrix
#:      OUT of the finished frame, hashes it with the same function, and
#:      `worker.apply_encoding_report` compares the two and writes
#:      `agrees_with_canonical` into the record. So "same data in two
#:      encodings" is a per-record measurement and not a claim in a comment.
#:
#: The alternative -- a second digest for CatBoost, with a caveat -- was
#: rejected. Two digests make the check unable to fail: any bug in the
#: re-encoding produces a second digest, which is exactly what a second
#: digest is supposed to look like, and nothing distinguishes "re-encoded
#: correctly" from "corrupted". One digest plus a reconstruction that has to
#: hash back to it can fail, and a check that cannot fail is not a check.
CATBOOST_CATEGORICAL_ENCODING = {
    "form": "mixed_frame_int64_categorical",
    "container": "pandas.DataFrame, float64 numeric and int64 categorical",
    "why_not_float64": (
        "catboost/core.py:804 refuses cat_features on a floating-point "
        "ndarray by dtype.kind. Verified on catboost 1.2.10 on 2026-08-16 "
        "by a 200-row Pool construction, which raised exactly that message"
    ),
    "why_not_object_array": (
        "an object ndarray IS accepted -- dtype.kind 'O' passes the check, "
        "verified on the same 200 rows -- and is one Python object per cell. "
        "At the standard tier's 1,000,000 by 15 that is fifteen million "
        "boxed objects, which is a container this harness will not build to "
        "avoid a pandas dependency it already has"
    ),
    "digest_decision": (
        "one dataset, two encodings, one canonical digest. The float64 "
        "digest is unchanged, unexempted and still compared across all "
        "arms; the adapter additionally reconstructs the canonical matrix "
        "from the frame and hashes it with the same function, and the "
        "record carries agrees_with_canonical per part per run"
    ),
    "verified_per_run": (
        "every declared categorical column is checked finite, integral and "
        "inside int64, and bitwise equal after float64 -> int64 -> float64, "
        "before the frame is built. Then the whole frame is read back as "
        "float64 and hashed. A column that fails any check raises and the "
        "run is recorded as an error"
    ),
    "missing_categories": (
        "THE ONE THING THIS DOES NOT SOLVE, and the reason "
        "`categorical_missing` still has no CatBoost row. CatBoost has no "
        "representation for a missing category and says so: \"Invalid type "
        "for cat_feature[...]=nan : cat_features must be integer or string, "
        "real number values and NaN values should be converted to string\" "
        "(catboost 1.2.10, read 2026-08-16 off a 200-row Pool with one NaN "
        "in a declared categorical column, in both a float64 column and an "
        "object column holding np.nan). Converting it to a string or to a "
        "negative integer IS accepted -- both verified on the same 200 rows "
        "-- and both are a modelling decision, not an encoding. They are "
        "also not a NEUTRAL one, which is the part that settles it: "
        "LightGBM treats a negative or NaN category as missing and mojotrees "
        "sends it to bin 0, which `src/mojotrees/categorical.mojo` documents "
        "is never a member of a split set, so on both of those engines a "
        "missing category is structurally unsplittable. A CatBoost sentinel "
        "level is an ordinary level with an ordinary target statistic. In "
        "`categorical_missing` the missingness is drawn against the TARGET's "
        "upper tail, so that statistic is a target-correlated feature that "
        "the other two engines cannot use by construction. A row produced "
        "that way would look like a comparison and would be a handicap "
        "match, which is worse than the skip it replaced"
    ),
    "cost": (
        "a full extra copy of the matrix in the frame plus a transient copy "
        "for the reconstruction, timed as the `encode` phase and counted in "
        "e2e. It is a HARNESS-CONVERSION cost: a CatBoost user whose "
        "categories already live in an integer column pays none of it, and "
        "peak_rss_bytes on a categorical CatBoost row includes work neither "
        "of the other two arms does. Subtract it before quoting a per-"
        "iteration figure; do not subtract it before quoting an end-to-end "
        "one, because the harness really did have to do it"
    ),
    "real_variants": (
        "the two REAL datasets that carry categorical columns go through "
        "this encoder too, and one of them may be refused by it. "
        "`loaders._encode_categories` writes NaN into a categorical column "
        "for an empty or missing field, so an adult or bank_marketing "
        "column with any blank in it fails the finite check and the "
        "CatBoost run is recorded as an error naming the column. That is "
        "the intended outcome and not a defect: a blank category on a real "
        "dataset is the same problem `missing_categories` describes and it "
        "does not become a different one for being real. Neither real "
        "variant has been fetched or run, so which of them actually "
        "contains a blank is UNVERIFIED here"
    ),
    "not_warmed": (
        "the warm-up fits one round on `_tiny_like`'s purely numeric matrix "
        "and therefore never touches the categorical path, so the first "
        "cat_features Pool of a run is inside the timed `ingest` phase. "
        "Stated rather than fixed: making the warm-up categorical would need "
        "`_tiny_like` to know which columns a scenario's DATA declares, "
        "which is not something a spec carries"
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
    # Runs as of 2026-08-16, with cat_features. This entry was a skip whose
    # stated exit condition was "the day the loader can produce an
    # integer-typed categorical block whose digest all three engines agree
    # on", and that is the change that landed: the adapter re-encodes the
    # canonical float64 matrix into a mixed frame, the canonical digest is
    # unchanged and unexempted, and the re-encoding proves itself per run by
    # reconstructing the canonical matrix and hashing it back. The argument
    # is CATBOOST_CATEGORICAL_ENCODING. The old skip predicted the wrong
    # half: it said the two categorical scenarios would clear together by
    # one change, and only this one did. categorical_missing's blocker was
    # never the float64 matrix, which is why it is still below.
    "high_cardinality_categorical": None,
    "ranking": (
        "CatBoost has no lambdarank. Its ranking losses are YetiRank, "
        "YetiRankPairwise, QueryRMSE and PairLogit, and none of them is the "
        "loss LightGBM and mojotrees are asked for here. Running one of "
        "them would put a third objective in a column headed by the other "
        "two's, so this scenario has no CatBoost row"
    ),
    "categorical_missing": (
        "a missing category has no encoding that is the same data, and this "
        "is now the WHOLE reason rather than half of it. The float64-matrix "
        "half is gone: the CatBoost adapter re-encodes a categorical block "
        "into int64 columns and proves the re-encoding per run, which is "
        "what let high_cardinality_categorical start running on 2026-08-16. "
        "It does not let this one. This generator drops values inside two "
        "categorical columns, and CatBoost refuses a NaN there outright -- "
        "\"cat_features must be integer or string, real number values and "
        "NaN values should be converted to string\" -- so the harness would "
        "have to choose a level to mean absence. A string sentinel and a "
        "negative integer are both accepted by CatBoost, and both are the "
        "same trap: LightGBM treats a negative or NaN category as missing "
        "and mojotrees routes it to bin 0, which never joins a split set, "
        "so on those two engines a missing category is structurally "
        "unsplittable, while on CatBoost a sentinel level gets an ordinary "
        "target statistic. This generator drops values as a function of the "
        "TARGET's upper tail, so that statistic would be a target-"
        "correlated feature available to exactly one arm. That is not a "
        "caveat on a comparison, it is the absence of one. So this scenario "
        "has no CatBoost row and the exit condition is a real one rather "
        "than a harness change: it clears if CatBoost grows a missing-"
        "category concept, or if a variant of this scenario is drawn with "
        "missingness only in the numeric columns, which would be a "
        "different scenario and should be named as one. See "
        "CATBOOST_CATEGORICAL_ENCODING['missing_categories']"
    ),
}

#: What running the CatBoost arm is expected to cost, where that is worth
#: knowing before a matrix is committed to rather than after it times out.
#:
#: A cost note is not a caveat on a number. It is a warning to whoever
#: schedules the run, and it is here rather than in a lane report so that
#: the manifest carries it.
CATBOOST_SCENARIO_COST = {
    "high_cardinality_categorical": (
        "the newest CatBoost cell and the least characterized, so schedule "
        "it before you commit a matrix to it. Three separate costs, and not "
        "one of them is measured. (1) The `encode` phase is a full extra "
        "copy of the matrix plus a transient second one for the "
        "reconstruction hash: at the standard tier's 1,000,000 by 15 that "
        "is roughly 120 MB each, a DERIVED BOUND from the shape and not a "
        "timing. (2) CatBoost builds an ordered target statistic per "
        "categorical column per permutation, and the standard tier's fifth "
        "column realizes about 175,000 distinct levels. Whether that is "
        "cheaper or dearer than LightGBM's 100,000-entry bin table is "
        "exactly the question the scenario exists to answer and is "
        "therefore unknown before it runs. (3) The CTR store is capped by "
        "ctr_leaf_count_limit, which is in CATBOOST_REFUSED_PARAMS and "
        "resolves to 2**64-1, i.e. uncapped. Nobody has timed any of this. "
        "run.py's per-run timeout is 7200 seconds and this cell is not "
        "established to fit inside it at the standard tier, let alone the "
        "large one; a timed-out cell is an infrastructure failure and takes "
        "the whole run's exit code. Run the smoke tier first"
    ),
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
#: it. There is still no `bagging_fraction` entry, and now for a stronger
#: reason than before: `bootstrap_type="MVS"` beside `bagging_fraction` is
#: two CatBoost bootstrap types at once and is refused
#: (`boosting._check_bootstrap`).
#:
#: **READ THIS BEFORE SCHEDULING A MATRIX.** `bootstrap_type` and `subsample`
#: joined this dict on 2026-08-16.
#:
#: **CORRECTED 2026-08-17, AND THE OLD VERSION UNDERSTATED THE TRAINER'S REACH
#: WHILE STILL REACHING THE RIGHT SCHEDULING DECISION.** It said these two keys
#: are "honored on the **dense, single-output, CPU** trainer only", that
#: `multiclass` raises because "`trainset.train_dataset_multiclass` ... takes no
#: bundle" and `sparse_highdim` because `model_sparse.fit_csc` "likewise", and
#: that the two cells therefore needed "either a per-scenario exclusion of these
#: two keys or an MVS-capable multiclass and sparse trainer before the next full
#: matrix runs". `CATBOOST_UNMATCHABLE["row_sampling"]` asked for this
#: correction by name.
#:
#: What is true at head, verified in source: the bundle reaches the DENSE
#: SINGLE-OUTPUT trainer on BOTH backends (`train_gpu` takes it; MVS routes to
#: the host-gradient arm where `sampling.bootstrap_round` draws exactly and the
#: trees still grow on the device), and BOTH CPU multiclass loops and BOTH
#: sparse loops call `bootstrap_round` too. So "dense, single-output, CPU only"
#: is wrong in three directions.
#:
#: The two cells still raise, and for reasons that are NOT the missing-bundle
#: reason:
#:
#: - `multiclass`: `sampling.check_mvs_reg_is_set` refuses a DERIVED `mvs_reg`
#:   on a softmax round, because the derivation reads a one-tree
#:   `[dim][leaf]` table that a round of K structurally different trees does not
#:   have, and this arm names no `mvs_reg`. On the GPU, independently,
#:   `train_multiclass_gpu` takes no bundle.
#: - `sparse_highdim`: `random_strength`, not `bootstrap_type`. The sparse round
#:   loop computes no per-tree score scale and the binding declares that
#:   parameter honored only as `not d[].is_sparse`.
#:
#: The scheduling decision the old paragraph demanded HAS ALREADY BEEN TAKEN and
#: needs nothing before the next matrix: both cells are declared skips in
#: `MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT`, whose entries now carry these
#: reasons. A silently unsampled multiclass row labelled "CatBoost mode" remains
#: the outcome to prevent and no path to it exists.
#:
#: **DO NOT ADD boosting_type="Ordered" TO THIS DICT.** `boosting.mojo::_boost_rounds`
#: refuses ordered boosting beside any bootstrap_type, because a dropped or
#: reweighted row changes which prefix each ordered fold was fitted on. This
#: arm now carries CatBoost's row sampling and therefore forecloses ordered
#: boosting on the same arm; a mojotrees ordered-boosting row has to be its
#: own arm, without the bootstrap. That costs nothing against CatBoost's own
#: defaults: CatBoost's boosting_type resolves to Plain at every tier this
#: suite runs, because defaults_helper.h::UpdateBoostingTypeOption turns
#: Ordered OFF and its other clause is IterationCount < 500, which fires at
#: 100 estimators regardless of row count. Ordered is a GPU-task-type default
#: and this suite runs CPU. See ORDERED_BOOSTING_ROWS, which records the same
#: reading from the other side.

# ---------------------------------------------------------------------------
# NOTE ON PLACEMENT. The long `#:` comment ABOVE this line documents
# `MOJOTREES_CATBOOST_MODE`, which is defined BELOW this whole depth-wise
# section. The depth-wise block was inserted between the two when the arm was
# added and nobody noticed, so the CatBoost-mode arm's "READ THIS BEFORE
# SCHEDULING A MATRIX" warning currently reads as though it were about the dict
# that follows it. It is not. Left in place rather than moved so that this
# session's diff is about the arm and not about relocating a hundred lines of
# somebody else's comment.
# ---------------------------------------------------------------------------

#: The depth-wise arm's parameter override: XGBoost 3.4.0's resolved defaults,
#: in our names.
#:
#: **THIS DICT CHANGED ON 2026-08-17 AND USED TO SAY SOMETHING ELSE.** It was
#: `{"grow_policy": "depthwise"}` and nothing more, and its comment argued at
#: length that nothing more belonged in it: the arm existed to price a growth
#: ORDER, so it had to differ from the plain mojotrees arm in the order and in
#: nothing else, and a second changed parameter would swamp an effect that
#: small. That argument was correct for the arm it described. Andrew replaced
#: the arm, in these words: "make our depthwise params match xgboost".
#:
#: So this is no longer an ISOLATION arm and must not be read as one. It is a
#: MIRROR arm, the same kind of thing `MOJOTREES_CATBOOST_MODE` is, and it is
#: read against the `xgboost` peer column rather than against the plain
#: mojotrees arm. The standing rule it now follows: our CatBoost mode mirrors
#: CatBoost, our lossguide arm mirrors LightGBM, and this arm mirrors XGBoost.
#: What was lost by the change is written down where the deleted second arm
#: used to be, near `MOJOTREES_XGBOOST_MODE WAS HERE`; in short, no arm now
#: prices the growth order on its own.
#:
#: **THE ARM'S NAME NOW UNDERSTATES IT.** It is still `mojotrees_depthwise`,
#: because the name is the cell key, the record field, the output filename and
#: what `verify.py` and `report.py` group on, so renaming it would orphan every
#: record ever written under it. A reader who sees the name and expects one
#: changed parameter is reading a stale expectation; this dict and
#: `MOJOTREES_DEPTHWISE_CLAIMS` are the correction, and the claims string
#: travels into every record.
#:
#: **WHERE THE VALUES COME FROM.** Every one was read out of
#: `Booster.save_config()` on a live xgboost 3.4.0 fit, is asserted in
#: `XGBOOST_RESOLVED_DEFAULTS` with the path it was read from, and is re-checked
#: against every measured XGBoost fit by `check_xgboost_readback`. Nothing here
#: is transcribed from documentation. `XGBOOST_DEFAULTS_SOURCE` records the
#: read, including the fact that `XGBRegressor().get_params()` is NOT a source
#: on this version because it returns None for every tree parameter.
#:
#: JUSTIFICATION PER KEY, because a dict without one is indistinguishable from
#: whatever happened to run:
#:
#: - `grow_policy="depthwise"`. XGBoost's resolved `grow_policy`. The canonical
#:   estimator spelling (`python/mojotrees/sklearn.py::_CANONICAL_GROW_POLICIES`),
#:   reaching `GROW_DEPTHWISE` (that table's own `#:` comment). This is the
#:   one key the arm always had.
#: - `max_depth=6`. XGBoost's resolved `max_depth`. Under depthwise growth this
#:   is the bound that defines the tree, and it is enforced in `tree._search`
#:   under both policies (`growth_policy.mojo` module docstring, rule 3,
#:   "Shape rules first").
#: - `num_leaves=64`. **The load-bearing one, and it is arithmetic rather than
#:   preference.** XGBoost's resolved `max_leaves` is 0, meaning unbounded, so a
#:   depth-6 depthwise tree may hold 2**6 = 64 leaves. Our
#:   `growth_policy.mojo` (module docstring, "The three rules that make a
#:   depth-wise tree well defined") keeps `num_leaves` a HARD bound and resolves an
#:   overshooting level by admitting a gain-ranked prefix, so the shared default
#:   of 31 stops our growth around depth 5 and would have compared a 31-leaf
#:   tree against a 64-leaf one under a label claiming a shape match. 64 is the
#:   only value at which the two rules cannot disagree. See
#:   XGBOOST_UNMATCHABLE['leaf_bound'].
#: - `learning_rate=0.3`. XGBoost's resolved `eta`, which reads back as
#:   0.300000012 because XGBoost stores it as a float. A mirror arm takes the
#:   competitor's rate for the reason `mojotrees_catboost_mode` takes
#:   CatBoost's: at a matched tree count, two arms at different rates are doing
#:   different amounts of fitting. The last-figure difference is declared in
#:   XGBOOST_UNMATCHABLE['learning_rate_precision'].
#: - `lambda_l2=1.0`. XGBoost's resolved `lambda` is 1 where LightGBM's stock
#:   is 0.0. As of 2026-08-17 it is also OUR default, so this pin and our
#:   shipped value coincide at 1.0 and this mirror arm no longer has to
#:   override anything to hold. It is still passed explicitly, because a
#:   mirror that holds by coincidence is one default change away from not
#:   holding, and because `mojotrees_params` now passes `lambda_l2` from
#:   `BASE_PARAMS` at LightGBM's 0.0, which this dict must override. That
#:   override is the same route `MOJOTREES_CATBOOST_MODE` uses for its 3.0.
#: - `min_child_hess=1.0`. XGBoost's resolved `min_child_weight` is 1 and it is
#:   a SUM OF HESSIANS, which is the quantity `min_child_hess` holds. The
#:   mapping is not a guess: `min_child_weight` is an ALIAS of `min_child_hess`
#:   in our own surface (`python/mojotrees/sklearn.py::_Base` docstring and the
#:   resolution in `python/mojotrees/sklearn.py::_Base._params`). It is NOT
#:   `min_data_in_leaf`, and getting that
#:   direction wrong would be invisible on a regression fit, where the hessian
#:   is 1 per row and the two constraints coincide, and wrong by a factor of
#:   four on a logistic fit, where a hessian is at most 0.25.
#: - `min_data_in_leaf=1`. XGBoost has NO row-count rule on a leaf. Leaving the
#:   shared 20 would impose a constraint XGBoost does not have and make our
#:   trees smaller than its for a reason invisible in a parameter table. 1 is a
#:   disarming rather than a match and it is declared as one in
#:   XGBOOST_UNMATCHABLE['leaf_row_count'].
#:
#: WHAT IS ALREADY MATCHED AND SO IS ABSENT FROM THIS DICT. Each of these was
#: read back from XGBoost and checked against our default rather than assumed
#: to agree: `alpha` 0 against `lambda_l1` 0.0 (BASE_PARAMS); `min_split_loss`
#: 0 against `min_gain_to_split` 0.0; `colsample_bytree`, `colsample_bylevel`
#: and `colsample_bynode` all 1 against `feature_fraction` and
#: `feature_fraction_bynode` 1.0; `subsample` 1 against no bootstrap;
#: `max_delta_step` 0 against ours; `num_parallel_tree` 1, meaning one tree per
#: round, which is what we grow. A key that is already equal is left out on
#: purpose, so that this dict is exactly the set of DIFFERENCES.
#:
#: **`max_bin` IS NOT HERE AND THE REASON IS NOT THE ONE THAT USED TO BE
#: GIVEN.** XGBoost's resolved `max_bin` is 256 and this harness's shared value
#: is 255, and the two DO count the same quantity: that was measured, not
#: assumed, through `QuantileDMatrix.get_quantile_cut()`. The gap is one bin and
#: it is left open for two reasons, the second binding: `max_bin` is a BINNING
#: parameter and this dict is a TRAINING override, and no fit at 256 bins has
#: ever been run in this repository on either backend. The whole argument, and
#: what it would take to close it, is in XGBOOST_UNMATCHABLE['bin_budget'].
#:
#: **EVERY KEY HERE ALREADY REACHES THE TRAINER**, which is why widening this
#: dict does not need a build to be safe. All seven are carried today by
#: `MOJOTREES_CATBOOST_MODE` on the scenarios that arm runs -- it sets
#: `grow_policy`, `max_depth`, `num_leaves`, `min_data_in_leaf`,
#: `min_child_hess` and `lambda_l2` -- and `learning_rate` is passed by
#: `mojotrees_params` on every arm. So no key in this dict is new to the
#: harness; only three values are.
MOJOTREES_DEPTHWISE = {
    "grow_policy": "depthwise",
    "max_depth": 6,
    "num_leaves": 64,
    "learning_rate": 0.3,
    "lambda_l2": 1.0,
    "min_child_hess": 1.0,
    "min_data_in_leaf": 1,
}

#: What a depth-wise row may and may not claim, carried into every record so
#: the claim travels with the number rather than living in a chat message.
#:
#: REWRITTEN 2026-08-17 with the dict above. What it said was that this arm
#: changed the growth order and NOTHING else and must be read against the plain
#: mojotrees arm. That is no longer true of the arm and would now be a false
#: statement travelling inside every record, which is the worst place for one.
#:
#: The motivation that produced the arm is still recorded as a motivation and
#: not as a finding: the device path batches a level into one host wait instead
#: of one per split (`train_gpu._device_search_resident`, and
#: `train_gpu.mojo::_search_record_slots` sizes a level's worth of records for exactly this
#: mode), so a depth-wise fit is EXPECTED to pay fewer host waits per tree than
#: a leaf-wise one. Nothing has measured that. Leaf-wise growth cannot batch and
#: is not asked to, because its next pick depends on the frontier the current
#: split just changed (`growth_policy.mojo` module docstring, "What a batched
#: level is, and what it is not").
MOJOTREES_DEPTHWISE_CLAIMS = (
    "arm 'mojotrees depthwise': mojotrees with "
    f"{MOJOTREES_DEPTHWISE} applied over the shared defaults, which is "
    "XGBoost 3.4.0's resolved defaults in our parameter names. READ IT "
    "AGAINST THE XGBOOST PEER COLUMN. It is a MIRROR arm as of 2026-08-17 and "
    "it used to be an isolation arm carrying only grow_policy=depthwise; a "
    "record written before that date is not this arm and the two must not be "
    "put in one series. It is NOT the headline row, which is the plain "
    "mojotrees arm against the LightGBM comparator, and it may NOT be read as "
    "a LightGBM parity row: depthwise is XGBoost's grow_policy and LightGBM "
    "4.7.0 has no growth policy switch at all, verified three ways on "
    "2026-08-17 -- no Python file in the installed package mentions "
    "grow_policy, the compiled lib_lightgbm.dylib contains no occurrence of "
    "the strings grow_policy, depthwise or lossguide while it does contain "
    "num_leaves, max_depth, deterministic and tree_learner, and "
    "src/mojotrees/growth_policy.mojo says so in as many words. So there is "
    "no 'LightGBM with depthwise growth' arm to compare against and one must "
    "not be manufactured by pinning max_depth on the comparator: a leaf-wise "
    "tree capped at a depth is not a depth-wise tree. The tree this arm grows "
    "is a DIFFERENT TREE from every other arm's and not a faster route to the "
    "same one, so an accuracy difference against any other arm is a property "
    "of the shape and not a defect. Two further limits on reading it. The "
    "mirror is checked and not assumed: every value comes from "
    "XGBOOST_RESOLVED_DEFAULTS, read out of Booster.save_config(), and "
    "check_xgboost_readback re-reads the same paths on every measured XGBoost "
    "fit, so compare this row's resolved parameters against the xgboost row's "
    "engine_resolved_params before calling the pair like-for-like. And what "
    "no parameter can close is in XGBOOST_UNMATCHABLE, in particular the "
    "one-bin difference in the binning budget and the absence of any "
    "row-count leaf rule in XGBoost. On the categorical and ranking "
    "scenarios there is no xgboost column at all, because that arm does not "
    "run them (XGBOOST_SCENARIO_SUPPORT says why), so on those cells this row "
    "has no peer to be read against and is a speed number beside the plain "
    "mojotrees arm's, carrying XGBoost's tree shape for no comparative reason."
)

MOJOTREES_CATBOOST_MODE = {
    # CatBoost's actual CPU defaults, to the extent this surface can now
    # reach them. Four knobs became reachable on 2026-08-16 and are set here
    # for the first time: grow_policy, score_function, random_strength and
    # leaf_estimation_iterations. Two more became reachable later the same
    # day: bootstrap_type and subsample.
    "grow_policy": "symmetrictree",
    "max_depth": 6,
    "num_leaves": 64,
    "min_data_in_leaf": 1,
    "min_child_hess": 0.0,
    "lambda_l1": 0.0,
    "lambda_l2": 3.0,
    "score_function": "cosine",
    # CatBoost's default, and it reaches the trainer this harness uses as of
    # 2026-08-16. It was removed earlier the same day because the smoke pass
    # raised on it: the noise and its per-tree scale were both implemented,
    # but `_parse_params` in bindings/_mojotrees.mojo declared the scale
    # honored at exactly ONE call site (`fit`, CPU only) and this harness
    # trains through `mojotrees.train(params, Dataset)` -> `train_dataset`,
    # which inherited the False default and refused by name.
    #
    # The refusal was right and the declaration was too narrow. Two round
    # loops in the whole package compute the scale, both through
    # `boosting._round_random_score_scale`: `_boost_rounds` and
    # `train_with_valid`'s own loop. `trainset.train_dataset`'s DENSE CPU arm
    # reaches the first of them, so that arm now declares it -- as the
    # conjunction `device == CPU and not is_sparse`, because a sparse dataset
    # resolves its device to the CPU too and routes to `boosting_sparse`,
    # which computes no scale. `booster_update` declares it as well. Every
    # other entry point still refuses by name.
    #
    # It is honored on the three scenarios this arm actually runs and on no
    # others, which is what makes restoring it safe rather than hopeful:
    # MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT already skips multiclass,
    # sparse, ranking and both categorical scenarios, so no cell reaches a
    # loop without a scale. The categorical skips matter twice over here --
    # `split.find_best_split` refuses `random_strength` beside a categorical
    # feature outright, because a category set is scored inside
    # `find_best_categorical_split` and only its winner reaches the numerical
    # scan, so one candidate per categorical feature would be noised while
    # every numerical feature had every candidate noised. Turning either
    # categorical scenario back on for this arm needs that resolved first.
    "random_strength": 1.0,
    "leaf_estimation_iterations": 1,
    # No `max_bin` row, and the reason is a routing fact rather than a
    # parity one. `dataset_params` exists because `max_bin` belongs to the
    # binning and **both libraries reject it on `train`**; this dict is
    # applied over the shared *training* parameters, so a `max_bin` here
    # reaches `mojotrees.train` and raises "'max_bin' describes the data,
    # not the training run". It did: the smoke pass on 2026-08-16 killed
    # this arm on dense_regression and high_cardinality_categorical alike,
    # which under an infrastructure failure would have withheld the quality
    # verdict for every cell that did run.
    #
    # Removing it changes no value. `BASE_PARAMS["max_bin"]` is already 255
    # and already reaches the Dataset, so this arm binned at 255 before the
    # key was added and bins at 255 without it. The parity argument the key
    # carried is true and is kept in MOJOTREES_CATBOOST_MODE_REASONS, which
    # is where a claim about granularity belongs; it just was not a key this
    # dict could carry.
    # CatBoost's one_hot_max_size under our name. CatBoost resolves 2 and our
    # default is 4, so this arm and the CatBoost arm differed on the cutoff
    # that decides whether a categorical column is one-hot encoded or handed
    # to the split search. It is INERT on every scenario this arm runs, all
    # three of which are numeric, and it is set anyway: a not_reached key is a
    # divergence waiting for a scenario, and the alternative was leaving it
    # unset with a comment promising somebody would notice.
    #
    # Reachability checked rather than assumed, because the last key added to
    # this dict without a run behind it was `max_bin` and it killed a smoke
    # pass. The route is `mojotrees.train(params, Dataset)` ->
    # `basic._Config.__init__`, which forwards every remaining key as a keyword
    # into `MojoTreesRegressor.__init__`; `max_cat_to_onehot` is a declared
    # constructor argument there (python/mojotrees/sklearn.py::_Base.__init__)
    # with default 4, is validated as nonnegative, and reaches the wire at
    # `bindings/_mojotrees.mojo::_parse_cat_params`. It is NOT in `basic._DATASET_PARAMS`,
    # which is what `max_bin` was and why that one raised.
    "max_cat_to_onehot": 2,
    "bootstrap_type": "MVS",
    # NOT row bagging. Under `bootstrap_type="MVS"` this key is the MVS rate
    # and row bagging is off, which is CatBoost's own contract for the
    # parameter and is resolved in `python/mojotrees/sklearn.py`'s
    # `_resolve_bootstrap`. Passing it without `bootstrap_type` would mean
    # uniform bagging, which is a different sampler and would be an
    # imitation of the number rather than of the method.
    "subsample": 0.8,
    # A WORKAROUND FOR A PACKAGE DEFECT, and the only entry in this dict that
    # is not here to imitate CatBoost. It is set to the value that turns the
    # mechanism OFF, which is the opposite of what every other key does, so
    # read the whole of this before touching it.
    #
    # The defect. This dict names `grow_policy="symmetrictree"`, and the
    # package's mode-defaults layer answers that by resolving `ctr` to
    # `ctr` IS DELIBERATELY ABSENT FROM THIS DICT, and it was present from
    # 2026-08-17 morning until the same afternoon. The history is kept short
    # here because the defect is fixed; the long version is in the commit.
    #
    # Under `grow_policy='symmetrictree'` the PACKAGE resolves `ctr` to
    # CatBoost's own source rule, and the harness never asks for it. The
    # `train_dataset` door then refused that resolved value, because a CTR
    # column is built while the matrix is binned and a prebuilt `Dataset`
    # arriving at `train` has nowhere to put a bundle. This harness trains
    # through `Dataset` plus `train` and through nothing else, so the mode
    # default was unreachable through the only door available and EVERY
    # CatBoost-mode cell raised at fit, dense scenarios included, where there
    # is not a categorical column in the matrix. `"ctr": "off"` was carried
    # here as a workaround so the arm could run at all.
    #
    # The source fix landed the same day and the workaround came out with it.
    # `sklearn.py` now sends the caller-named flag, and the binding DECLINES an
    # unnamed mode default on an entry point that cannot honor it rather than
    # refusing it, which is the rule the `random_strength_set` flag beside it
    # already followed. `train_dataset` also passes `ctr_ok` from the dataset's
    # own bundle, so a `Dataset` that really carries one is honored and a rule
    # disagreeing with the dataset's is refused with both named. Verified
    # against the rebuilt extension: symmetric mode through `Dataset` plus
    # `train` fits, and an explicitly named rule on a bundle-less `Dataset` is
    # still refused.
    #
    # WHAT THIS MEANS FOR THIS ARM, and it is a real limit rather than a
    # formality. On the numeric scenarios this arm runs, the mode default is
    # declined and no CTR columns are built, which is the same fit `"off"`
    # produced, so removing the key moves no number. To actually EXERCISE
    # CatBoost's target statistics the rule has to reach the DATASET, as
    # `Dataset(params={'ctr': ...})`, not the training dict. That is a separate
    # change and it is what a CTR experiment on a categorical scenario needs.
    #
    # The comparison run of 2026-08-17 was taken with the workaround in place.
    # That is sound for every row it produced, because all of them are numeric
    # and an inert bundle and `"off"` are the same fit there.
}

#: Why each entry above is what it is, carried into the record beside the
#: dict so the arm explains itself where it is read.
MOJOTREES_CATBOOST_MODE_REASONS = {
    "grow_policy": (
        "CatBoost's own SymmetricTree, and no longer an approximation of it. "
        "This entry read 'depthwise, the nearest shape this harness can "
        "REACH' until 2026-08-16, when the estimator's validator was widened "
        "to carry symmetrictree through to the GROW_OBLIVIOUS the package had "
        "held unreachable since wave 4. tree_shape leaves "
        "CATBOOST_UNMATCHABLE as a result"
    ),
    "score_function": (
        "CatBoost's CPU default for EVERY grow policy -- the widely repeated "
        "Lossguide-to-NewtonL2 override is inside a TaskType == GPU branch "
        "and is unreachable on CPU. It matters HERE and would not at our "
        "stock settings: Cosine's numerator is the L2 sum and at lambda_l2 = "
        "0 its denominator collapses onto the same expression, so it "
        "degenerates to sqrt(L2). This arm runs lambda_l2 = 3"
    ),
    "random_strength": (
        "CatBoost's default. The per-split score noise was implemented on "
        "host and device for months while the function computing its "
        "per-tree scale had zero callers, so a positive value was refused; "
        "the dense CPU round loop now computes it before each tree"
    ),
    "leaf_estimation_iterations": (
        "1, which is CatBoost's resolved value under RMSE and is the ONLY "
        "value this harness's entry point accepts. This entry used to claim "
        "it was 'CatBoost's resolved value for the objectives this suite "
        "runs, 1 for RMSE and Logloss'; a run on 2026-08-16 read 10 off a "
        "live Logloss fit and that claim was a transcription from an RMSE fit "
        "generalized to a loss nobody had read it on. See "
        "CATBOOST_UNMATCHABLE['leaf_estimation_iterations'] for what our side "
        "refuses and why the arm is unmatched on every binary cell"
    ),
    "bootstrap_type": (
        "CatBoost's actual CPU default sampler, and the largest single "
        "difference this arm still carried. SetNotSpecifiedOptionsToDefaults "
        "installs MVS whenever the user set nothing, the loss is neither "
        "multiclass-only nor multi-regression, task_type is CPU and "
        "sampling_unit is Object; RMSE and Logloss satisfy all four, so "
        "every CatBoost number this suite has ever published came from an "
        "MVS fit while the two rows beside it saw every row. Reachable from "
        "Python for the first time on 2026-08-16: the sampler had been built "
        "and wired into boosting.train's round loop for a wave, and the "
        "missing edge was bindings/_mojotrees.mojo not parsing the key and "
        "model.fit not forwarding it. HONORED ON THE DENSE SINGLE-OUTPUT CPU "
        "TRAINER ONLY. The multiclass and sparse scenarios refuse it by name "
        "rather than training an unsampled model, which is correct behavior "
        "and is also a live scheduling problem; see the header comment on "
        "MOJOTREES_CATBOOST_MODE. Note that CatBoost does not run MVS for "
        "multiclass either -- its own defaulting block excludes the "
        "multiclass-only losses and falls back to the Bayesian bootstrap -- "
        "so the honest CatBoost-mode multiclass arm is not this value at all"
    ),
    "subsample": (
        "0.8, which is the value MVS installs, and NOT the 0.66 the "
        "TBootstrapConfig constructor suggests: anyone reading 0.66 out of "
        "the header is wrong by a fifth of the rows. Under "
        "bootstrap_type=MVS this key is the MVS rate and row bagging is off, "
        "which is CatBoost's own meaning for the parameter. It is not "
        "bagging_fraction: bagging_fraction IS CatBoost's Bernoulli "
        "bootstrap under mojotrees's name, so setting both would ask for two "
        "bootstrap types at once and is refused"
    ),
    "max_bin": (
        "CatBoost's border_count of 254 counts THRESHOLDS where max_bin "
        "counts BINS, so 255 here and 254 there are the same granularity "
        "budget. Both sides also produce the same NUMBER of borders: "
        "GreedyLogSum yields exactly min(border_count, distinct - 1), which "
        "is LightGBM's rule too. Only the placement differs"
    ),
    "max_cat_to_onehot": (
        "CatBoost's one_hot_max_size default of 2, which is the cardinality "
        "at or below which a categorical column is one-hot encoded instead of "
        "becoming a CTR. Ours defaults to 4, so the two arms crossed that "
        "boundary at different cardinalities. Inert on all three scenarios "
        "this arm runs, none of which has a categorical column; set anyway, "
        "because the alternative is a divergence that goes live the day a "
        "categorical scenario unparks and that nothing would have caught. "
        "CATBOOST_PARAM_MAP's required_when_scenarios is the check that this "
        "stays set once one does"
    ),
    "ctr": (
        "NOT CatBoost's value, and the one entry in the dict that turns a "
        "mechanism off rather than matching one. CatBoost's ordered target "
        "statistics stay in CATBOOST_UNMATCHABLE['categorical_encoding'] and "
        "this key does not close that gap. It is here because "
        "grow_policy=symmetrictree makes the package resolve ctr to "
        "CatBoost's source rule by itself (python/mojotrees/sklearn.py's "
        "_CATBOOST_CTR) and bindings/_mojotrees.mojo::_parse_params then refuses the "
        "value it resolved, because train_dataset passes ctr_ok=False and a "
        "CTR column has to be built while the matrix is binned. The harness "
        "trains through Dataset + train and has no other door, so every "
        "CatBoost-mode cell raised at fit until this was set -- including "
        "dense_regression, which has no categorical column. Behaviorally "
        "identical to a correctly inert bundle on the numeric scenarios this "
        "arm runs, and a workaround rather than a decision: see the comment "
        "at the key for the two package-side fixes that retire it"
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
    "learning_rate": (
        "CatBoost's own resolved rate for THIS CELL, taken from that cell's "
        "get_all_params() read-back and never from a constant. It has no "
        "entry in the dict above because it has no static value: the rate "
        "moves with the row count, the feature count and the iteration count. "
        "MOJOTREES_CATBOOST_MODE_FROM_READBACK is where it lives instead, and "
        "an arm built without a read-back REFUSES rather than falling back to "
        "BASE_PARAMS['learning_rate']"
    ),
}

#: The `MOJOTREES_CATBOOST_MODE` keys that have no static value and must come
#: from CatBoost's own read-back for the same cell, mapped to the CatBoost key
#: they are read from.
#:
#: **This is what makes the arm honest and it is also what makes it refuse.**
#: Every other entry in `MOJOTREES_CATBOOST_MODE` is a constant because
#: CatBoost resolves it to a constant, verified across three shapes and two
#: losses at `CATBOOST_DEFAULTS_SOURCE`. `learning_rate` is not one of those.
#: There is no value that could be written here that would be right on more
#: than one scenario, so the arm takes it from the fitted CatBoost model or it
#: does not run.
#:
#: **Consequence, stated rather than discovered.** `mojotrees_catboost_mode`
#: cannot be built from `scenarios.py` alone any more. It needs the CatBoost
#: cell for the same scenario, tier and variant to have run first and to have
#: left its `engine_resolved_params` where the mojotrees cell can find it. The
#: wiring for that is `catboost_readback_key`, `catboost_readback_entry` and
#: `load_catboost_readback` below, plus one edit each to `run.py` and
#: `worker.py` that this lane did not own; until those land, this arm raises
#: `CatBoostReadbackMissing` by name. It does not fall back. A fallback here
#: is exactly the hand-written belief the whole change removes.
MOJOTREES_CATBOOST_MODE_FROM_READBACK = {
    "learning_rate": "learning_rate",
}

#: Keys the CatBoost-mode arm REMOVES from the shared parameters, so that
#: mojotrees's own defaulting fires instead of receiving a value.
#:
#: **Empty today, and the reason it exists is a transition that is already
#: scheduled.** A lane is making `auto_learning_rate.mojo` reachable and ON by
#: default in CatBoost mode, with CatBoost's own gate: the automatic rate fires
#: only when `learning_rate`, `leaf_estimation_method`,
#: `leaf_estimation_iterations` and `l2_leaf_reg` are ALL unset. The moment
#: that lands, this arm has a problem that `MOJOTREES_CATBOOST_MODE` cannot
#: express, because that dict is applied with `dict.update` and update can
#: override a key but cannot delete one.
#:
#: `mojotrees_params` sets `learning_rate` unconditionally from `BASE_PARAMS`.
#: So on the day the automatic rate exists, this arm would still arrive at the
#: trainer with `learning_rate=0.1` set, CatBoost's gate would not fire, and
#: the arm would run a pinned rate under a heading that says it derives one.
#: That failure is silent: the fit succeeds and the number looks fine.
#:
#: The handover this arm uses today has the same shape from the other side --
#: it SETS `learning_rate` from CatBoost's read-back -- so the two mechanisms
#: are mutually exclusive by construction and `selfcheck` fails if a key is in
#: both tables. Read `CATBOOST_LEARNING_RATE_TRANSITION` for which one should
#: be live when.
#:
#: **Update, 2026-08-16: the half of the transition that was ours has landed
#: and this tuple is still empty, on purpose.** `lane/mode-defaults-layer`
#: built the mode-defaults layer, so `grow_policy='symmetrictree'` now supplies
#: CatBoost's `l2_leaf_reg` 3.0, its per-objective `leaf_estimation_iterations`
#: and its `random_strength` 1.0 with `SetDefault` semantics -- assigned, and
#: NOT recorded as the caller naming the key. That removes the trap the wire
#: note described: an arm that stops naming `lambda_l2` no longer falls back to
#: our stock 0.0, it gets CatBoost's 3.0 from the mode, and CatBoost's gate
#: stays open.
#:
#: What has not moved is the `learning_rate` row, and the reason is a file
#: boundary rather than a design one. `selfcheck.py::check_catboost_arm` asserts
#: `"learning_rate" in MOJOTREES_CATBOOST_MODE_FROM_READBACK`, so taking it out
#: of that dict fails the self check, and `selfcheck.py` belongs to the harness
#: lane. Moving both rows in one commit is a one-line change on each side; it
#: is not two changes that can be sequenced from here, because the assertion
#: and the tuple would disagree in between.
MOJOTREES_CATBOOST_MODE_UNSET = ()

#: How this arm gets CatBoost's learning rate, now and next, in one place.
#:
#: Written down because two mechanisms are in flight for one value and the
#: failure mode of having both half-wired is an arm that looks right and runs
#: a rate neither engine chose.
CATBOOST_LEARNING_RATE_TRANSITION = {
    "today": (
        "HANDOVER. The CatBoost cell for a scenario runs first and writes its "
        "CatBoost.get_all_params() into the run's catboost_readback.json; the "
        "CatBoost-mode cell reads that file and takes the resolved rate. "
        "MOJOTREES_CATBOOST_MODE_FROM_READBACK is the wiring and "
        "CatBoostReadbackMissing is what happens without it. The value our arm "
        "trains at is CatBoost's own number, not a number computed from a "
        "reimplementation of CatBoost's formula"
    ),
    "next": (
        "DERIVATION, once auto_learning_rate.mojo is reachable and on by "
        "default in CatBoost mode. Our arm computes the rate from the same "
        "inputs, so it no longer needs a CatBoost fit to have happened first, "
        "and the scheduling constraint between the two cells goes away"
    ),
    "the_handover_does_not_go_away_when_the_derivation_lands": (
        "and this is the whole argument for keeping it. A derived rate is our "
        "formula's output. Nothing in a record would say whether it agrees "
        "with CatBoost's, and 'CatBoost mode mirrors CatBoost exactly' is a "
        "claim that has to be measured rather than asserted. The read-back is "
        "the only artifact that carries CatBoost's actual number, so it stops "
        "being an INPUT and becomes a CHECK: catboost_parity_from_records "
        "compares our resolved learning_rate against the CatBoost row's "
        "engine_resolved_params for the same cell and fails on a mismatch. "
        "That function is already written and already called at record time"
    ),
    "how_to_switch": (
        "one dict entry moves and nothing else. Take `learning_rate` out of "
        "MOJOTREES_CATBOOST_MODE_FROM_READBACK, which makes "
        "catboost_readback_values return {} and the arm buildable without a "
        "read-back, and put it into MOJOTREES_CATBOOST_MODE_UNSET so the "
        "shared 0.1 is REMOVED rather than passed and CatBoost's gate can "
        "fire. Do not add it to MOJOTREES_CATBOOST_MODE: a value there is a "
        "pin, which is what turns the automatic rate off"
    ),
    "where_the_comparison_lives": (
        "at RECORD time, in worker.run_job, through "
        "scenarios.record_parity_block. Deliberately not in selfcheck.py, "
        "which trains nothing and downloads nothing by design and must keep "
        "running in under a second anywhere: a get_all_params() call needs a "
        "fitted model and there is no fit in that process. Deliberately not "
        "against a read-back recorded by an earlier run either, because a "
        "stale read-back from another shape is exactly the wrong-number-that-"
        "looks-right this whole entry is about. selfcheck checks the WIRING "
        "statically; the run checks the VALUE"
    ),
}

# ---------------------------------------------------------------------------
# The resolved-parameter parity map. CatBoost's key on the left, ours on the
# right, and a verdict that is checked rather than asserted.
# ---------------------------------------------------------------------------
#
# **What this replaces.** `MOJOTREES_CATBOOST_MODE` was a hand-written dict of
# what somebody believed CatBoost's defaults to be, and
# `selfcheck.check_catboost_arm` checked it against itself: it proved the
# translator did not DROP a key, which is a real thing to prove, and it could
# not tell whether any value in it was RIGHT. This table is the other half.
# Every key CatBoost resolves is classified, and the classification is
# enforced:
#
#   matched       our resolved value must equal CatBoost's resolved value
#                 after `translate`, or selfcheck FAILS.
#   unmatchable   no parameter closes it. Must name a live
#                 `CATBOOST_UNMATCHABLE` entry.
#   not_reached   we have the knob and this arm does not set it, because no
#                 scenario it runs reaches it. Must say so, and must name the
#                 scenarios that would make it reachable, so that turning one
#                 of them on for this arm fails the check instead of quietly
#                 running unmatched.
#
# Nothing may be absent. `selfcheck` walks the whole resolved CatBoost dict
# and fails on any key that is in neither this table nor
# `CATBOOST_PARAM_NOT_MAPPED`, because a key nobody classified is exactly the
# hand-written belief this table exists to remove.
#
# **`ours` names a key in the arm's RESOLVED dict**, which spans two
# containers: the training params from `mojotrees_catboost_mode_params` and
# the Dataset params from `dataset_params`. `mojotrees_catboost_mode_resolved`
# is the union and `ours_container` says which side a key comes from.
# `border_count` is the whole reason that distinction is in the table: it maps
# to `max_bin`, `max_bin` belongs to the Dataset, and putting `max_bin` into
# `MOJOTREES_CATBOOST_MODE` -- which is applied over the TRAINING dict --
# raised "'max_bin' describes the data, not the training run" and killed a
# smoke pass on 2026-08-16. It is checked here and it is not set there.
#
# **`our_default` is the fallback when the arm's dict does not carry the key**,
# with `our_default_source` saying where the default was read. The lookup
# prefers the arm's resolved value, so adding a wrong value to
# `MOJOTREES_CATBOOST_MODE` later fails this check rather than being masked by
# the declared default.
#
# **`static: False`** marks a key whose CatBoost value is derived per fit and
# cannot be diffed without a run. `learning_rate` is the only one. selfcheck
# checks its WIRING statically -- that it is sourced from the read-back on our
# side and from nothing on CatBoost's -- and `catboost_parity_from_records`
# checks its VALUE against two finished records.

#: Translation names, resolved through `_PARITY_TRANSLATORS`. Strings rather
#: than callables because `catboost_arm_block()` must stay JSON-serializable,
#: and `selfcheck` checks that it does.
CATBOOST_PARAM_MAP = {
    # --- passed by the harness -------------------------------------------
    "loss_function": {
        "ours": "objective",
        "verdict": "matched",
        "translate": "loss_to_objective",
        "note": "the problem statement, not a setting. Inverted through "
                "CATBOOST_LOSS",
    },
    "iterations": {
        "ours": "n_estimators",
        "verdict": "matched",
        "translate": "identity",
        "note": "the one matched parameter. CATBOOST_MATCHED. Note that the "
                "mojotrees adapter adds n_estimators to the translated dict "
                "itself (engines.MojoTreesEngine._run_dense), so "
                "mojotrees_catboost_mode_resolved mirrors that rather than "
                "reading it out of MOJOTREES_CATBOOST_MODE",
    },
    "classes_count": {
        "ours": "num_class",
        "verdict": "matched",
        "translate": "identity",
        "note": "multiclass only, and this arm runs no multiclass cell. "
                "Mapped anyway so that the scenario walk cannot meet an "
                "unclassified key if one is ever turned on",
    },
    # --- derived by CatBoost, taken from the read-back --------------------
    "learning_rate": {
        "ours": "learning_rate",
        "verdict": "matched",
        "translate": "identity",
        "static": False,
        "note": "NOT passed and NOT a constant. CatBoost resolves it from the "
                "iteration count and the dataset, our arm takes that resolved "
                "value out of the read-back "
                "(MOJOTREES_CATBOOST_MODE_FROM_READBACK), and the equality is "
                "checked against two finished records rather than statically. "
                "Compared through a float32 round trip because CatBoost "
                "stores it as a 32-bit float: see "
                "CATBOOST_UNMATCHABLE['learning_rate_precision']",
    },
    # --- CatBoost's resolved constants ------------------------------------
    "depth": {"ours": "max_depth", "verdict": "matched", "translate": "identity"},
    "max_leaves": {
        "ours": "num_leaves",
        "verdict": "matched",
        "translate": "identity",
        "note": "CatBoost's max_leaves resolves to 2**depth = 64 and our "
                "num_leaves is set to the same, so the leaf cap does not bind "
                "before the depth cap does",
    },
    "grow_policy": {
        "ours": "grow_policy",
        "verdict": "matched",
        "translate": "identity",
        "note": "SymmetricTree against symmetrictree. Value strings are "
                "case-insensitive on both surfaces "
                "(docs/PARAMETER_NAMING.md), so _parity_equal folds case",
    },
    "min_data_in_leaf": {
        "ours": "min_data_in_leaf",
        "verdict": "matched",
        "translate": "identity",
    },
    "l2_leaf_reg": {
        "ours": "lambda_l2",
        "verdict": "matched",
        "translate": "identity",
        "note": "docs/PARAMETER_NAMING.md maps CatBoost l2_leaf_reg to our "
                "reg_lambda, whose LightGBM spelling lambda_l2 is what this "
                "harness passes",
    },
    "border_count": {
        "ours": "max_bin",
        "ours_container": "dataset",
        "verdict": "matched",
        "translate": "borders_to_bins",
        "note": "254 borders against 255 bins is the SAME granularity budget, "
                "which is why the translation is +1 and not identity. It "
                "lives in dataset_params and must never be added to "
                "MOJOTREES_CATBOOST_MODE: that dict is applied over the "
                "training params and mojotrees.train refuses max_bin by name",
    },
    "bootstrap_type": {
        "ours": "bootstrap_type",
        "verdict": "matched",
        "translate": "identity",
    },
    "subsample": {
        "ours": "subsample",
        "verdict": "matched",
        "translate": "identity",
        "note": "under bootstrap_type=MVS this is the MVS rate on both sides, "
                "not a bagging fraction. python/mojotrees/sklearn.py "
                "_resolve_bootstrap keeps CatBoost's contract literally",
    },
    "random_strength": {
        "ours": "random_strength",
        "verdict": "matched",
        "translate": "identity",
        "note": "the VALUE matches and the draws do not. See "
                "CATBOOST_UNMATCHABLE['split_scoring']: same distribution, "
                "different streams, and no parameter makes them the same draw",
    },
    "score_function": {
        "ours": "score_function",
        "verdict": "matched",
        "translate": "identity",
    },
    "leaf_estimation_iterations": {
        "ours": "leaf_estimation_iterations",
        "verdict": "unmatchable",
        "unmatchable_key": "leaf_estimation_iterations",
        "translate": "identity",
    },
    "boosting_type": {
        "ours": "boosting_type",
        "verdict": "matched",
        "translate": "boosting_alias",
        "our_default": "gbdt",
        "our_default_source": (
            "python/mojotrees/sklearn.py _BOOSTING_ALIASES maps CatBoost's "
            "'plain' onto 'gbdt', which is the estimator's own default, so "
            "neither BASE_PARAMS nor MOJOTREES_CATBOOST_MODE sets the key"
        ),
        "note": "Plain at every tier this suite runs. ORDERED_BOOSTING_ROWS "
                "carries the source verification: the only default that "
                "installs Ordered is behind TaskType == GPU",
    },
    "rsm": {
        "ours": "feature_fraction",
        "verdict": "matched",
        "translate": "identity",
        "our_default": 1.0,
        "our_default_source": (
            "LightGBM's stock feature_fraction, which mojotrees defaults to "
            "as well. Unset on every arm in this harness"
        ),
    },
    "boost_from_average": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "boost_from_average",
    },
    "task_type": {
        "ours": "device",
        "verdict": "matched",
        "translate": "identity",
        "note": "CPU against cpu, folded case. The CatBoost adapter refuses "
                "any other device outright (engines.CatBoostEngine.load)",
    },
    "random_seed": {
        "ours": "random_state",
        "verdict": "matched",
        "translate": "identity",
        "note": "SHARED_SEED, referenced by all three translators as of "
                "2026-08-16. It was 190019 on two arms and absent on the "
                "other two until then. Read SHARED_SEED's own comment for "
                "which draws this actually reaches: on these scenarios only "
                "the MVS bootstrap, and NOT the split-score noise, whose seed "
                "has no Python edge at all",
    },
    "one_hot_max_size": {
        "ours": "max_cat_to_onehot",
        "verdict": "matched",
        "translate": "identity",
        "note": "CatBoost resolves 2 and our default is 4. Set explicitly in "
                "MOJOTREES_CATBOOST_MODE rather than left to the default, "
                "even though it is inert on all three scenarios this arm "
                "runs. required_when_scenarios below is what keeps it set: "
                "once a categorical scenario unparks for this arm, the check "
                "demands the key be named in MOJOTREES_CATBOOST_MODE rather "
                "than agreeing by coincidence",
        "required_when_scenarios": (
            "high_cardinality_categorical",
            "categorical_missing",
        ),
    },
    # --- no parameter closes it --------------------------------------------
    "feature_border_type": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "border_placement",
    },
    "nan_mode": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "missing_values",
    },
    "leaf_estimation_method": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "leaf_estimation_method",
    },
    "leaf_estimation_backtracking": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "leaf_estimation_method",
        "reason": (
            "AnyImprovement, and mojotrees has no backtracking step to "
            "configure: boosting._estimate_leaf_values takes its Newton "
            "steps unconditionally. Filed under the same entry as "
            "leaf_estimation_method because it is the same missing surface"
        ),
    },
    "sampling_frequency": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "row_sampling",
        "reason": (
            "PerTree, and there is no frequency parameter on this arm: "
            "python/mojotrees/sklearn.py _resolve_bootstrap refuses an "
            "explicit subsample_freq or bagging_freq beside a bootstrap_type. "
            "The sampler does run once per tree, from boosting.train's round "
            "loop, so the two are believed to agree -- and no value in either "
            "resolved dict proves it, which is what puts this here rather "
            "than under matched"
        ),
    },
    "random_score_type": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "split_scoring",
    },
    "max_ctr_complexity": {
        "ours": "max_ctr_complexity",
        "verdict": "unmatchable",
        "unmatchable_key": "ctr_combinations",
        "reason": (
            "CatBoost resolves 1. Our surface HAS the name and refuses every "
            "value of it: python/mojotrees/sklearn.py refuses above 1 because "
            "no grow loop drives the projection enumeration, and refuses 1 "
            "itself because the fitted CTR tables are MODEL STATE and the "
            "model format carries no section for them, so a CTR fit would "
            "write a file that loads with empty tables and scores wrong "
            "(ctr.check_ctr_model_support, serialize.check_ctr_serializable, "
            "catalog A29). A knob that refuses every value is not a knob"
        ),
    },
    "counter_calc_method": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "categorical_encoding",
    },
    "ctr_target_border_count": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "categorical_encoding",
    },
    "store_all_simple_ctr": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "categorical_encoding",
    },
    "has_time": {
        "ours": None,
        "verdict": "unmatchable",
        "unmatchable_key": "categorical_encoding",
        "reason": (
            "False, so CatBoost permutes the rows, and the ordered "
            "permutation is the whole of what an ordered target statistic is. "
            "mojotrees has no row-order parameter and no CTR to order. Filed "
            "under categorical_encoding, which is the entry that says the two "
            "engines build a different feature from a categorical column"
        ),
    },
}

#: CatBoost keys this harness deliberately does NOT diff, with the reason and,
#: where the reason depends on the value, the value it depends on.
#:
#: `holds_while` is the teeth. Every entry below is inert or harness-only AT
#: THE VALUE CATBOOST CURRENTLY RESOLVES, and several of them stop being inert
#: at another value: `model_shrink_rate` is a no-op at 0 and a per-round
#: multiplier above it. So the reason is recorded together with the value it
#: rests on, and `selfcheck` fails if the declared resolution moves off it.
#: `None` means the reason does not depend on the value.
CATBOOST_PARAM_NOT_MAPPED = {
    "thread_count": {
        "why": "set by the runner on every peer arm from one number. "
               "mojotrees takes it from MOJOTREES_NUM_WORKERS in the "
               "environment rather than as a parameter, and on a whole-machine "
               "cell that variable is deliberately unset because it is a block "
               "count and not a thread count (run.mojotrees_workers), so there "
               "is no key on our side to diff",
        "holds_while": None,
    },
    "allow_writing_files": {
        "why": "a declared deviation that removes filesystem work from inside "
               "the timed region. It moves no bit of the model and mojotrees "
               "writes no such log, so there is nothing to match",
        "holds_while": False,
    },
    "logging_level": {
        "why": "a declared harness setting that keeps CatBoost's progress "
               "table off the stdout stream run.py parses for backend proof. "
               "Reporting, not model",
        "holds_while": "Silent",
    },
    "model_shrink_rate": {
        "why": "model shrinkage, resolved OFF. mojotrees has no "
               "model-shrinkage parameter on the Python surface (catalog A14 "
               "records it as refused by name), so this arm cannot turn on "
               "something CatBoost has turned off. There is nothing to diff "
               "while the rate is 0 and there is a real gap the moment it is "
               "not, which is what holds_while is for",
        "holds_while": 0,
    },
    "model_shrink_mode": {
        "why": "the shape of a shrinkage that is off. Read with "
               "model_shrink_rate",
        "holds_while": "Constant",
    },
    "model_size_reg": {
        "why": "CatBoost penalizes model size when building CTR features from "
               "high-cardinality columns. mojotrees has no counterpart, and "
               "this arm builds no CTRs at all "
               "(CATBOOST_UNMATCHABLE['ctr_combinations']), so on the "
               "scenarios it runs the parameter has nothing to act on",
        "holds_while": 0.5,
    },
    "penalties_coefficient": {
        "why": "the multiplier on CatBoost's per-feature and per-split "
               "penalties, none of which this arm sets, so it multiplies "
               "zero. mojotrees's counterpart is the CEGB ledger and it is "
               "off on every arm here",
        "holds_while": 1,
    },
    "sparse_features_conflict_fraction": {
        "why": "exclusive feature bundling, resolved OFF, which is the same "
               "state enable_bundle=False puts the comparator in. Refused by "
               "name in CATBOOST_REFUSED_PARAMS as well, because pinning it "
               "would be pinning it to what it already is",
        "holds_while": 0.0,
    },
    "bayesian_matrix_reg": {
        "why": "a Bayesian-bootstrap parameter, and this arm's bootstrap_type "
               "resolves to MVS, so it is not read at all",
        "holds_while": 0.1,
    },
    "posterior_sampling": {
        "why": "SGLD-style posterior sampling, resolved off. mojotrees's "
               "langevin counterpart is refused by name (catalog A13), so "
               "there is no state in which the two arms could agree on a "
               "non-default value",
        "holds_while": False,
    },
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
    "learning_rate": (
        "CatBoost's own resolution, as of cb-shipped. This was PASSED until "
        "2026-08-16 and the pin came out by decision, so it is refused here "
        "rather than merely dropped: a caller putting it back through `extra` "
        "would restore cb-default's model under cb-shipped's label, which is "
        "precisely the defect CATBOOST_REFUSED_PARAMS exists to make "
        "impossible. See CATBOOST_DELIBERATE_DIVERGENCE"
    ),
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
    # The categorical knobs, refused from the day the arm gained a
    # categorical path. The temptation here is specific and worth naming:
    # CATBOOST_UNMATCHABLE['ctr_combinations'] records that CatBoost
    # resolves max_ctr_complexity to 1 at its defaults, so the interaction
    # column in high_cardinality_categorical is not read by a combination
    # feature. Setting it to 4 would make the scenario's claim true and the
    # column's heading false, which is the same defect
    # bin_construct_sample_cnt was: one arm doing more work than the label
    # says, in the direction that makes the story better.
    "max_ctr_complexity": (
        "how many categorical features a CTR may combine. Stock, and stock "
        "resolves to 1 on CPU whatever the documentation says, so this arm "
        "builds single-column CTRs only. Raising it is a differently "
        "labelled row and not this one"
    ),
    "one_hot_max_size": (
        "the cardinality below which CatBoost one-hots instead of building "
        "a CTR. Stock is 2. It is the direct counterpart of "
        "max_cat_to_onehot on the other two sides, which are also at their "
        "own defaults, and pinning it to theirs would be aligning one of "
        "the three capacity rules the scenario exists to cross"
    ),
    "simple_ctr": "the CTR definition itself. The column is CatBoost's own",
    "combinations_ctr": "a synonym problem: see max_ctr_complexity",
    "ctr_target_border_count": (
        "how finely the target is discretized before a CTR is taken. Stock "
        "is 1 and it is part of what the CTR IS"
    ),
    "counter_calc_method": (
        "whether the Counter CTR counts the test pool. Stock is SkipTest "
        "and changing it changes what the feature means"
    ),
    "ctr_leaf_count_limit": (
        "a memory cap on stored CTR values. CatBoost changes what it keeps "
        "under it, so a limit derived from the data size makes the engine "
        "do different work at different scales -- used_ram_limit's defect "
        "in the categorical vocabulary"
    ),
    "has_time": (
        "whether CatBoost treats the row order as a time order instead of "
        "permuting it. Stock is False, so it permutes, and the ordered "
        "permutation is the whole of what 'ordered target statistic' means. "
        "Pinning it True would turn the arm's headline algorithm off"
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
    "xgboost": {
        "ingest": (
            "DMatrix(X, label=y): conversion of the caller's array into "
            "XGBoost's own object layout, and nothing else. The DMatrix is "
            "not quantized when it returns, which is what makes this arm's "
            "phase split the same shape as CatBoost's rather than "
            "LightGBM's"
        ),
        "encode": (
            "null on every scenario this arm runs. XGBoost's categorical "
            "support needs a third container beside LightGBM's index list "
            "and CatBoost's mixed frame, and both categorical scenarios are "
            "refused rather than converted: see XGBOOST_SCENARIO_SUPPORT"
        ),
        "binning": (
            "null. Handed a plain DMatrix, XGBoost quantizes inside train(). "
            "QuantileDMatrix would separate it and is deliberately not used, "
            "because building one is a different ingestion path from the one "
            "a default XGBoost user takes. See "
            "XGBOOST_UNMATCHABLE['binning_phase']"
        ),
        "train": (
            "xgboost.train(), which contains XGBoost's quantization as well "
            "as its boosting rounds, so this phase is NOT comparable with "
            "mojotrees's or LightGBM's train phase on its own. Compare e2e"
        ),
        "e2e": "ingest + train",
    },
    "catboost": {
        "encode": (
            "categorical scenarios only, null everywhere else. The harness "
            "re-encoding its canonical float64 matrix into the mixed "
            "float-and-int frame CatBoost will accept cat_features over, "
            "plus the hash that proves the re-encoding is lossless. A phase "
            "neither other arm has, because neither other arm needs a "
            "different container: see CATBOOST_CATEGORICAL_ENCODING. It is "
            "a HARNESS-CONVERSION cost and not a CatBoost one -- a user "
            "whose categories already live in an integer column pays none "
            "of it -- so it is counted in e2e and named separately so it "
            "can be subtracted from a per-iteration reading"
        ),
        "ingest": (
            "Pool(X, label=y, cat_features=...): conversion of the caller's "
            "array or frame into CatBoost's own object layout, and nothing "
            "else. The pool is not quantized when it returns. On a "
            "categorical scenario this is also where CatBoost first builds "
            "its categorical hash, and the warm-up did not cover that path"
        ),
        "binning": (
            "null. CatBoost bins inside fit, and pre-quantizing to separate "
            "it changes the model above a few hundred thousand rows"
        ),
        "train": (
            "fit(), which contains CatBoost's quantization and, on a "
            "categorical scenario, the ordered target statistics it derives "
            "from the categorical block"
        ),
        "e2e": "encode + ingest + train, and encode is null on every "
               "scenario whose data declares no categorical features, "
               "which today is every one this arm runs except "
               "high_cardinality_categorical",
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
    # `lambda_l2` is routed again as of 2026-08-17. It left BASE_PARAMS on
    # 2026-08-16, when both engines defaulted it to 0.0 and handing it to
    # neither was the same fit as handing 0.0 to both. Our default moved to
    # 1.0 on 2026-08-17, so those two stopped being the same fit and the key
    # came back at LightGBM's 0.0 to keep this pair a mirror. Both engines
    # take it under the same name.
    "lambda_l2": (("train", "lambda_l2"), ("train", "lambda_l2")),
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
#: **RESOLVED 2026-08-16, and the premise above is FALSE.** The
#: source-verification came back and there is no CPU row threshold at all.
#: `boosting_options.cpp:16` constructs `BoostingType` with `Plain`, and on
#: CPU that is what a fit KEEPS at every row count. The only place `Ordered`
#: is installed as a default has `TaskType == ETaskType::GPU` in its
#: condition (`catboost_options.cpp:802-806`). The famous 50,000 is real but
#: it is a **GPU** rule at `defaults_helper.h:33-42` that turns Ordered
#: **OFF**, and it carries an iteration clause (`IterationCount < 500`) that
#: every retelling drops -- so a retelling that keeps the 50,000 and loses
#: the iteration bound describes a rule that fires on cases it should not.
#:
#: So **CatBoost on CPU never chooses ordered boosting by itself**, and the
#: harness's own contrary evidence -- `Plain` resolved on a 20,000-row fit --
#: was not a hint that the threshold was lower. It was the rule.
#:
#: What that costs this scenario: a row here is a **Plain-vs-Plain** row
#: under an Ordered heading unless `boosting_type=Ordered` is passed to the
#: CatBoost arm explicitly. Passing it makes the arm no longer "CatBoost at
#: its own defaults", which is what `CATBOOST_ARM_LABEL` promises, so this
#: needs a SECOND CatBoost arm rather than a change to the existing one.
#: That arm is not built. A variant CatBoost arm did exist until
#: 2026-08-17, `catboost_lossguide`, and Andrew removed it: it answered a
#: question about CatBoost's tree shape that nobody is asking. So the shape
#: this note points at has no live example any more, and `engines.py`
#: `variant_params` plus `run.CELL_ORDER` are what a new one would use.
#:
#: The row count is kept at 50,000 because the scenario is still worth
#: having: it is a small-data shape where ordered boosting is the mechanism
#: most likely to pay, and it is the size CatBoost's GPU rule would have
#: chosen Ordered below. It is no longer a threshold, only a size.
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
        "supersedes": CATBOOST_ARM_SUPERSEDES,
        "is_the_comparator": False,
        "catboost_passed": dict(CATBOOST_ALIGNMENT),
        "catboost_matched_from_base_params": dict(CATBOOST_MATCHED),
        "deliberate_divergence": copy.deepcopy(CATBOOST_DELIBERATE_DIVERGENCE),
        "catboost_resolved_per_fit": dict(CATBOOST_RESOLVED_PER_FIT),
        "catboost_deviations_from_stock": copy.deepcopy(
            CATBOOST_DEVIATIONS_FROM_STOCK
        ),
        "catboost_harness_settings": dict(CATBOOST_HARNESS_SETTINGS),
        "catboost_left_at_stock": dict(CATBOOST_LEFT_AT_STOCK),
        "catboost_defaults_source": CATBOOST_DEFAULTS_SOURCE,
        "catboost_min_version": ".".join(str(p) for p in CATBOOST_MIN_VERSION),
        "catboost_refused_params": dict(CATBOOST_REFUSED_PARAMS),
        "determinism": copy.deepcopy(CATBOOST_DETERMINISM),
        # How a categorical column reaches this arm, and the argument for
        # why the canonical digest still means what it says once it does.
        # In the block rather than only in the module because a published
        # table that carries a CatBoost categorical number and not this is
        # a table whose reader cannot check the one thing that makes the
        # number comparable.
        "categorical_encoding": dict(CATBOOST_CATEGORICAL_ENCODING),
        "unmatchable": dict(CATBOOST_UNMATCHABLE),
        "scenarios_not_run": {
            name: reason
            for name, reason in CATBOOST_SCENARIO_SUPPORT.items()
            if reason
        },
        # The CatBoost-mode arm skips a scenario the CatBoost arm runs, which
        # was impossible before it carried row sampling. A reader comparing
        # the two columns row by row will find rows where only one of them is
        # present, and this is where that is stated rather than inferred.
        "mojotrees_catboost_mode_scenarios_not_run": {
            name: reason
            for name, reason in (
                MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT.items()
            )
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
        "mojotrees_catboost_mode_from_readback": dict(
            MOJOTREES_CATBOOST_MODE_FROM_READBACK
        ),
        "mojotrees_catboost_mode_reasons": dict(
            MOJOTREES_CATBOOST_MODE_REASONS
        ),
        # The key-by-key diff between the two resolved dicts, per scenario,
        # in the block that already travels into the manifest, records.json
        # and the CSV comparator column. A reader of any published CatBoost
        # number can therefore see what each engine actually ran rather than
        # what the heading claims, which is the field this arm was missing.
        "parity_map": copy.deepcopy(CATBOOST_PARAM_MAP),
        "parity_not_mapped": copy.deepcopy(CATBOOST_PARAM_NOT_MAPPED),
        "resolved_parity": resolved_parity_block(),
        "matched": (
            "tree count, and nothing else by decision. Both rows run the same "
            "number of boosting iterations, because a comparison at different "
            "budgets is not a comparison. The LEARNING RATE is deliberately "
            "not matched as of 2026-08-16: CatBoost resolves its own from the "
            "budget and the dataset, roughly 0.4273 at 100 iterations against "
            "the 0.1 the LightGBM comparator and the plain mojotrees arm run. "
            "Read deliberate_divergence before quoting any accuracy number "
            "from this column"
        ),
    }


def resolved_parity_block():
    """Both resolved dicts and their key-by-key diff, per scenario.

    Wrapped defensively on purpose. This is called from `catboost_arm_block`,
    which `run.py` calls before the first cell runs and writes into every
    manifest, so an exception in here would take down a whole matrix over a
    reporting field. A scenario that cannot produce a table records the
    failure and the run continues; `selfcheck` is where a broken table is a
    failure, and it walks the same function.
    """
    block = {}
    for name in MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS:
        try:
            spec = resolve(name, "standard")
            rows = catboost_parity_rows(spec)
            block[name] = {
                "catboost_resolved_declared": catboost_resolved_declared(spec),
                "mojotrees_catboost_mode_resolved": (
                    mojotrees_catboost_mode_resolved(
                        spec, "cpu", None, _READBACK_STANDIN(spec)
                    )
                ),
                "rows": rows,
                "failures": catboost_parity_failures(rows),
                "scheduled": (
                    MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT.get(name) is None
                ),
                "note": (
                    "catboost_resolved_declared is this file's transcription "
                    "of get_all_params(), not a live read-back, and it omits "
                    "every CATBOOST_RESOLVED_PER_FIT key because no static "
                    "value for those is honest. The live values are in each "
                    "CatBoost record's engine_resolved_params; "
                    "check_catboost_readback compares the two and "
                    "catboost_parity_from_records compares the deferred keys"
                ),
            }
        except Exception as exc:  # noqa: BLE001 - a reporting field, not a fit
            block[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "note": (
                    "the parity table could not be built for this scenario. "
                    "selfcheck.check_catboost_arm fails on this; a run "
                    "records it and continues rather than losing a matrix to "
                    "a reporting field"
                ),
            }
    return block


def peer_arms_block():
    """Every arm reported beside the comparator, keyed by engine name.

    A dict rather than a list so that a record can be looked up by the
    engine that wrote it, and so that adding a fourth engine is one entry
    rather than a second mechanism.

    `xgboost` joined on 2026-08-17, and it was missing for the whole of the day
    the arm was written: every constant, refusal and support table existed and
    the block reached no manifest, no `records.json` and no CSV column, because
    this function is the only route any of them takes and it named one engine.
    That is the failure this docstring's second sentence was written to prevent,
    happening to the docstring, so it is recorded here rather than fixed
    silently.
    """
    return {"catboost": catboost_arm_block(), "xgboost": xgboost_arm_block()}


def peer_banner():
    """The peer arms on the console, under the comparator banner.

    Short enough to read and specific enough to check, in the shape
    `run.comparator_banner` already uses. The full block goes into the
    manifest and into records.json.

    The XGBoost lines are appended rather than folded into one loop over
    `peer_arms_block()`. A generic loop would have to print the fields the two
    arms share, and the fields worth reading are the ones they do not: the
    CatBoost banner's point is the learning rate CatBoost derives per fit, and
    the XGBoost banner's point is that this harness asserts XGBoost's defaults
    from a read-back and checks them live. A banner that printed neither
    because both were engine-specific would be shorter and worth nothing.
    """
    block = catboost_arm_block()
    passed = " ".join(
        f"{key}={value}" for key, value in sorted(block["catboost_passed"].items())
    )
    matched = " ".join(
        f"{key}=BASE_PARAMS[{source!r}]"
        for key, source in sorted(block["catboost_matched_from_base_params"].items())
    )
    xgb = xgboost_arm_block()
    xgb_passed = " ".join(
        f"{key}={value}" for key, value in sorted(xgb["xgboost_passed"].items())
    )
    xgb_matched = " ".join(
        f"{key}=BASE_PARAMS[{source!r}]"
        for key, source in sorted(
            xgb["xgboost_matched_from_base_params"].items()
        )
    )
    return (
        f"peer arm {block['one_line']}: {block['label']}\n"
        f"  reported beside the comparator, never instead of it. "
        f"{block['registered']}\n"
        f"  catboost gets: {passed}\n"
        f"  matched from BASE_PARAMS: {matched}\n"
        f"  NOT matched, by decision: learning_rate. CatBoost resolves its "
        f"own (about 0.4273 at 100 iterations) where the comparator runs "
        f"0.1.\n"
        f"    {block['deliberate_divergence']['learning_rate']['how_to_misread_it']}\n"
        f"  supersedes: {block['supersedes']}\n"
        f"  everything else is CatBoost's own default "
        f"({block['catboost_defaults_source']})\n"
        f"  determinism: {block['determinism']['status']}. "
        f"{block['determinism']['flag']}\n"
        f"  observed: {block['determinism']['observed']}\n"
        f"  binning: {PHASE_SHAPE['catboost']['binning']}\n"
        f"peer arm {xgb['one_line']}: {xgb['label']}\n"
        f"  reported beside the comparator, never instead of it. "
        f"{xgb['registered']}\n"
        f"  xgboost gets: {xgb_passed} plus nthread and device, set by the "
        f"adapter\n"
        f"  matched from BASE_PARAMS: {xgb_matched}\n"
        f"  NOT matched, by decision: learning_rate. XGBoost's eta resolves "
        f"to 0.300000012 where the comparator runs 0.1.\n"
        f"    {xgb['xgboost_deliberate_divergence']['learning_rate']['why']}\n"
        f"  everything else is XGBoost's own default, read back rather than "
        f"transcribed ({len(xgb['xgboost_resolved_defaults'])} values "
        f"asserted and re-checked per fit)\n"
        f"  mirrored by our {xgb['mirrored_by']} arm: {xgb['mirror_params']}\n"
        f"  determinism: {xgb['determinism']['status']}. "
        f"{xgb['determinism']['flag']}\n"
        f"  observed: {xgb['determinism']['observed']}\n"
        f"  binning: {PHASE_SHAPE['xgboost']['binning']}"
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


#: THE BAYES FLOOR, added 2026-08-17. `docs/design/ACCURACY_GAP.md` sections 1
#: and 2 established that the dense regression GENERATOR adds noise that is
#: exactly `noise * N(0,1)`, so the Bayes-optimal RMSE on that variant is
#: `noise` and no model can beat it. Almost all of every arm's RMSE is that
#: floor: at the standard tier the shipped arm is 1.7 percent behind CatBoost
#: on RMSE and 28.8 percent behind on the EXCESS error, `rmse**2 - noise**2`,
#: which is the part of the error the model is responsible for and the only
#: part a mechanism can move. Both are the same measurement through two lenses
#: and the harness reports both wherever it reports a gap between arms.
#:
#: The floor is a fact about the GENERATOR and about nothing else. A scenario
#: declares it under `bayes_floor` and the readers apply it ONLY to records
#: whose `data.data_kind` is `synthetic`; `dense_regression` also runs as UCI
#: YearPredictionMSD, whose noise nobody knows, and a record from that variant
#: gets the raw gap alone. A scenario without the field has no floor and never
#: gets one fabricated.
#:
#: TWO NUMBERS, AND THE READERS USE THE SECOND. The declaration here is the
#: POPULATION noise scale, 0.30, which is the fact about the generator. The
#: excess is measured against the noise REALIZED on the held-out rows, which
#: `generators.realized_noise_floor` reproduces bit-exactly from the record's
#: generator kwargs and split (0.298252 at the standard tier, 0.299662 at the
#: large tier, both matching the document); worker.py stamps it on the record
#: as `data.bayes_floor.realized_mse` and verify.bayes_floor_of recomputes it
#: for older records. The distinction is not pedantry: at the standard tier the
#: two MSEs differ by 0.001046, a quarter of a typical arm's excess, and
#: subtracting the population value would overstate every arm's excess by that
#: much and turn a 66 percent excess-MSE gap into a 92 percent one.
def BAYES_FLOOR_RMSE(noise, generator="dense_regression"):
    """The declaration a scenario carries when its generator's target is
    `signal + noise * N(0,1)`: metric name, the population floor in that
    metric's own units and as MSE, the generator whose realized floor the
    readers reproduce, and the variant it applies to."""
    return {
        "metric": "rmse",
        "value": float(noise),
        "mse": float(noise) ** 2,
        "generator": generator,
        "applies_to": "synthetic",
        "source": f"generators.{generator}, y = signal + noise * N(0,1)",
    }


def bayes_floor(scenario_id, data_kind, generator_kwargs=None):
    """The floor that applies to a record of `scenario_id` on `data_kind`, or
    None. None for every scenario without a declaration, for every real-data
    record, and when the record's own `generator_kwargs` carry a `noise` that
    disagrees with the declaration, because a stale declaration must fall back
    to the raw gap rather than quote an excess against the wrong floor."""
    spec = SCENARIOS.get(scenario_id) or {}
    floor = spec.get("bayes_floor")
    if not floor or data_kind != floor.get("applies_to", "synthetic"):
        return None
    noise = (generator_kwargs or {}).get("noise")
    if noise is not None and abs(float(noise) - floor["value"]) > 1e-12:
        return None
    return dict(floor)


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
        # The generator's noise is exactly 0.30 * N(0,1), so the best any
        # model can do on the SYNTHETIC variant is known and is not a
        # parameter of the comparison. See `bayes_floor` below the table for
        # what this field means, when it applies, and why the real variant
        # (UCI YearPredictionMSD) has none.
        bayes_floor=BAYES_FLOOR_RMSE(0.30),
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
            "synthetic variant is purely numeric and does not.",
            "That real variant is also the one place the CatBoost arm can "
            "meet a categorical column on a scenario nobody chose it for. "
            "It takes it through the same encoder "
            "high_cardinality_categorical uses, with the same per-run "
            "proof, and it is REFUSED with a named column if any of the ten "
            "holds a blank -- loaders._encode_categories writes NaN for "
            "one. Which of the ten does is unverified: bank_marketing has "
            "never been fetched here. The synthetic variant, which is what "
            "every run of this scenario so far has used, is purely numeric "
            "and reaches none of this.",
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
            "There is no CatBoost row and the reason is PERMANENT, not a "
            "harness limitation waiting on a change. The float64-matrix "
            "blocker that used to be quoted here is gone -- "
            "high_cardinality_categorical runs CatBoost with cat_features "
            "as of 2026-08-16 -- and what is left is that this generator "
            "drops values inside two categorical columns as a function of "
            "the target's upper tail. CatBoost has no missing-category "
            "concept, so the harness would have to invent a level, and that "
            "level would carry a target-correlated statistic that LightGBM "
            "and mojotrees cannot use: both treat a missing category as "
            "structurally unsplittable. See "
            "CATBOOST_SCENARIO_SUPPORT['categorical_missing'].",
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
            "interaction is double-centered so neither marginal carries it, "
            "and a tree needs two stacked set splits to reach it. A CTR "
            "COMBINATION would reach it in one -- and CatBoost at its own "
            "defaults does not build one, because it resolves "
            "max_ctr_complexity to 1. So the interaction makes the problem "
            "harder for all three arms equally and does not, on this "
            "scenario, separate a combination feature from a stack of "
            "splits. See CATBOOST_UNMATCHABLE['ctr_combinations']."
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
            "There IS a CatBoost row as of 2026-08-16, and this is the only "
            "scenario in the suite where CatBoost's ordered target "
            "statistic is read against a category-set split. The arm is "
            "handed the same five columns as categorical, on the same "
            "canonical dataset with the same digest, in a different "
            "container: CATBOOST_CATEGORICAL_ENCODING carries the encoding "
            "and the per-run proof that the container did not change a "
            "value. The comparison is three-way -- mojotrees, LightGBM and "
            "CatBoost -- plus mojotrees_catboost_mode beside them, and none of "
            "the four is the headline row, which is still mojotrees against "
            "LightGBM stock+det.",
            "The two-column interaction is NOT read by a CTR combination on "
            "the CatBoost row, and the scenario's own note above overstates "
            "this until it is corrected. CatBoost resolves "
            "max_ctr_complexity to 1 at its defaults -- measured, against a "
            "documented default of 4 -- so it builds single-column CTRs "
            "only and reaches the interaction the same way the other two "
            "do, by stacking two splits. See "
            "CATBOOST_UNMATCHABLE['ctr_combinations']. The interaction term "
            "is still doing its job, which is to make the scenario harder "
            "than five marginals; what it is not doing is separating a "
            "combination feature from a stack of splits.",
            "The CatBoost row pays an `encode` phase the other two arms do "
            "not: a full extra copy of the matrix, timed and counted in "
            "e2e and labelled a harness-conversion cost. peak_rss_bytes on "
            "that row is not comparable with the other two without "
            "subtracting it. CATBOOST_SCENARIO_COST warns about the rest "
            "and none of it has been timed.",
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
        # Same recipe, noise 0.60 in every tier above, so the floor is 0.60.
        # `check_bayes_floor` in selfcheck.py holds this against the
        # generator_sizes noise so the two cannot drift apart silently.
        bayes_floor=BAYES_FLOOR_RMSE(0.60),
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
            "There is no longer a leaf-wise CatBoost row here. "
            "catboost_lossguide was removed on 2026-08-17 because Andrew is "
            "not asking what CatBoost's tree shape costs. The expectation it "
            "carried is recorded because it was never checked and someone "
            "will ask again: a Lossguide CatBoost arm was expected NOT to be "
            "an ordered row even when the catboost row is, because ordered "
            "boosting is a symmetric-tree mechanism in CatBoost, so the "
            "library was expected either to refuse the pair or to resolve to "
            "Plain. That was a stated expectation and never an observation, "
            "and it is still unverified.",
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
    # `lambda_l2` joins them as of 2026-08-16, and the reason narrowed on
    # 2026-08-17. It was pinned to 1.0 on both sides for as long as
    # mojotrees's default was 1.0, and the pin came out with the default.
    # `BASE_PARAMS` now supplies LightGBM's stock 0.0 to both engines, which
    # is what keeps this a mirror pair while our own default is 1.0. What
    # this refusal still guards is a SCENARIO reaching in through `extra` to
    # pin some other value on the comparator, which would put the arm
    # labelled stock on a regularizer no default produces.
    if "lambda_l2" in (extra or {}):
        raise ValueError(
            "lambda_l2 was passed to the comparator through `extra`. It is "
            f"stock (0.0) in {COMPARATOR_LABEL}, it is supplied to both "
            "engines from BASE_PARAMS at that stock value, and mojotrees's "
            "own default is 1.0 as of 2026-08-17, so a scenario-level pin "
            "here would silently move the comparator rather than align it: "
            "see LIGHTGBM_STOCK_DEFAULTS and the module docstring."
        )
    shared = shared_params(spec, extra)
    params = {
        "objective": shared["objective"],
        "num_leaves": shared["num_leaves"],
        "max_depth": shared["max_depth"],
        "learning_rate": shared["learning_rate"],
        "min_data_in_leaf": shared["min_data_in_leaf"],
        "min_sum_hessian_in_leaf": shared["min_child_hess"],
        # `lambda_l2` is passed again since 2026-08-17. It was "stock on both
        # sides" from 2026-08-16, which made passing it redundant; our stock
        # moved to 1.0, which makes passing it the only thing keeping this
        # pair a mirror. It is LightGBM's own 0.0, so this restates
        # LightGBM's default and overrides ours.
        "lambda_l2": shared["lambda_l2"],
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
    MOJOTREES_NUM_WORKERS, which the runner decides in the environment before
    the extension is imported, and a parameter here would suggest there are
    two ways to set it. Since 2026-08-17 "decides" includes leaving it UNSET,
    which is auto mode and is what a whole-machine cell gets: see
    `run.mojotrees_workers` for why an integer copied from the peers' thread
    count was not the same instruction.

    Nothing on this side answers the comparator's `deterministic`, and
    that is the point of the setting rather than an omission: mojotrees is
    reproducible across thread counts with no parameter and no cost, so
    there is nothing here to turn on.

    **`random_state` is here as of 2026-08-16 and it used to be absent.** The
    paragraph above was read for longer than it should have been as an
    argument that this arm needed no seed at all, and it is not one: it says a
    fit REPEATS without a seed, not that three arms share a stream without one.
    They did not. LightGBM and CatBoost were both pinned to 190019 and this
    translator passed nothing, so the mojotrees arms ran from the estimator's
    own per-component defaults. `SHARED_SEED` is the one constant all three now
    reference, and its docstring records exactly which draws that does and does
    not reach.
    """
    shared = shared_params(spec, extra)
    params = {
        "objective": shared["objective"],
        "num_leaves": shared["num_leaves"],
        "max_depth": shared["max_depth"],
        "learning_rate": shared["learning_rate"],
        "min_data_in_leaf": shared["min_data_in_leaf"],
        "min_child_hess": shared["min_child_hess"],
        # scikit-learn's spelling and this library's canonical one
        # (docs/PARAMETER_NAMING.md). `MojoTreesRegressor.__init__` takes it,
        # `_resolve_seeds` fans it out to every per-component seed, and
        # `_resolve_bootstrap` reads it for `bootstrap_seed`, which is the one
        # that is live on the CatBoost-mode arm.
        "random_state": SHARED_SEED,
        # `lambda_l2` is passed again, for the same reason as on the LightGBM
        # side and with the opposite effect. Until 2026-08-17 the estimator's
        # own default WAS LightGBM's 0.0, so passing it restated a default.
        # The shipped default is 1.0 now, so passing 0.0 here OVERRIDES ours
        # to hold the mirror. This arm is therefore no longer "mojotrees at
        # its defaults"; the shipped-defaults arm is a separate row.
        "lambda_l2": shared["lambda_l2"],
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
    "high_cardinality_categorical": (
        "standard",
        "the standard tier is 1,000,000 rows with a 200,000-level column "
        "and the large tier doubles both. Nothing on this cell has been "
        "timed on any engine -- the scenario's own caveats say the large "
        "tier is not established to fit run.py's 7200-second timeout for "
        "LightGBM either -- and the CatBoost arm additionally pays an "
        "`encode` phase and builds an ordered target statistic per level "
        "per permutation. Capping at standard is a bound on an UNMEASURED "
        "cost, which is the weakest kind: it is not an observed limit and "
        "it should come off the moment somebody has a large-tier timing "
        "rather than an argument. The standard tier is the one the headline "
        "table reads and it is not capped",
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
CATBOOST_ENGINES = ("catboost",)

# ---------------------------------------------------------------------------
# The XGBoost peer arm, and the mojotrees arm shaped toward it.
# ---------------------------------------------------------------------------
#
# WHY XGBOOST IS HERE AT ALL, added 2026-08-17 on Andrew's instruction. The
# `mojotrees_depthwise` arm had nothing to be read against. `grow_policy` with
# the values `depthwise` and `lossguide` is XGBoost's parameter and LightGBM
# has no growth-policy switch at all, verified three ways and recorded in
# `MOJOTREES_DEPTHWISE_CLAIMS`. So the growth-order arm could be read against
# our own leaf-wise arm and against nothing else in the industry. XGBoost is
# the library whose DEFAULT growth order is the one that arm grows, which
# makes it the only honest external reference for it.
#
# It is a PEER, exactly as CatBoost is: reported beside the comparator, never
# instead of it. `PROFILE_PROTOCOL.md` section C9 names one comparator and
# nothing is gated on this arm.

#: The peer arm's identity. Read beside `comparator_id()`, never in place of it.
XGBOOST_ARM_ID = "xgb-shipped"
XGBOOST_ARM_VERSION = 1
XGBOOST_ARM_LABEL = "XGBoost defaults (as shipped), at a matched tree count"
XGBOOST_ARM_REGISTERED = (
    "bench/results/PROFILE_PROTOCOL.md section C9 names one comparator, "
    "LightGBM stock+det. This arm is a peer column reported beside it and is "
    "not a comparator: nothing is gated on it and no threshold is measured "
    "against it"
)

#: The minimum XGBoost this harness will run as the peer arm.
#:
#: 2.0 is the floor and the reason is that this arm's honesty depends on two
#: things that landed in 2.0. `Booster.save_config()` returning the RESOLVED
#: configuration is what lets the arm record what XGBoost actually ran rather
#: than a transcription of its documentation, which is the same standard
#: `CATBOOST_DEFAULTS_SOURCE` holds the CatBoost arm to. And `device`
#: replaced the older `gpu_id` and `predictor` spelling, so a record that says
#: which backend ran means one thing across every version this floor admits.
#: `tree_method` also defaults to `hist` from 2.0, which matters because a
#: floor that admitted `approx` or `exact` by default would put two different
#: algorithms under one label.
XGBOOST_MIN_VERSION = (2, 0)

#: Where every XGBoost default in this module was read.
#:
#: **CORRECTED 2026-08-17, later the same day.** What stood here said no
#: default was transcribed at all, because `xgboost` was not yet installed in
#: the `bench` environment and a hand-copied table would have been a set of
#: claims nobody had checked. That was the right rule under the wrong facts.
#: xgboost 3.4.0 is installed now, so the values below were read out of a live
#: fit and `XGBOOST_RESOLVED_DEFAULTS` asserts exactly the ones the
#: `mojotrees_depthwise` mirror arm depends on, with a live drift check
#: (`check_xgboost_readback`) against every fit.
#:
#: The arm still records `Booster.save_config()` per fit, and that is still the
#: authority. The table is there so the mirror arm can be checked against
#: something, which is the defect the CatBoost arm hit from the other side:
#: a mirror whose target is unasserted cannot be shown to have drifted.
#:
#: **The sklearn wrapper is NOT a source for an XGBoost default in 3.4.0, and
#: this was checked rather than assumed.** `XGBRegressor().get_params()`
#: returns `None` for every tree parameter, `objective` and `missing` excepted,
#: so the defaults live only in the C++ core and only a fitted booster can be
#: asked for them.
XGBOOST_DEFAULTS_SOURCE = (
    "xgboost.Booster.save_config() on a fitted booster, xgboost 3.4.0 in the "
    "bench environment, read 2026-08-17 from a 400-row by 6-feature "
    "throwaway regression fit at reg:squarederror through the native "
    "DMatrix plus xgboost.train path this arm uses. Cross-checked against "
    "XGBRegressor().get_params(), which in 3.4.0 returns None for every tree "
    "parameter and is therefore not a source. Every value in "
    "XGBOOST_RESOLVED_DEFAULTS carries the save_config path it was read from, "
    "and every measured fit re-reads the same paths through "
    "check_xgboost_readback, so a version that moves a default is reported "
    "rather than assumed away"
)

#: The XGBoost defaults this harness asserts, each with the `save_config()`
#: path it was read from and the value it read, as a STRING because
#: `save_config()` returns every scalar as one.
#:
#: Two rules govern this table and both are the CatBoost arm's, learned there.
#:
#: **Only what something depends on.** These are the keys the
#: `mojotrees_depthwise` mirror arm reads to build `MOJOTREES_DEPTHWISE`, plus
#: the few that say which ALGORITHM ran, because a mirror of a different
#: algorithm is not a mirror. Nothing else is asserted; the full resolved
#: configuration is in every record and is longer than any table worth
#: maintaining.
#:
#: **The paths, not the names.** A key like `learning_rate` appears twice in
#: `save_config()` under two spellings, and `max_bin` sits in
#: `tree_train_param` while `tree_method` sits in `gbtree_train_param`. A table
#: of bare names would have to guess which one a checker meant, so each entry
#: names the exact path and `check_xgboost_readback` walks it.
#:
#: WHAT THE VALUES SAY, in the two places they are surprising:
#:
#: - `tree_method` resolves to the string **"auto"**, not to "hist". The hist
#:   algorithm is what actually runs and the evidence for that is the resolved
#:   `updater`, which is `grow_quantile_histmaker`. A reader checking that this
#:   arm ran the histogram algorithm must read the updater; the tree_method
#:   string does not say so.
#: - `eta` resolves to **0.300000012**, which is float32(0.3). XGBoost stores
#:   its tree parameters as float, so the exact decimal 0.3 is not what ran.
#:   See XGBOOST_UNMATCHABLE['learning_rate_precision'].
XGBOOST_RESOLVED_DEFAULTS = {
    "grow_policy": {
        "path": "learner.gradient_booster.tree_train_param.grow_policy",
        "value": "depthwise",
    },
    "max_depth": {
        "path": "learner.gradient_booster.tree_train_param.max_depth",
        "value": "6",
    },
    "max_leaves": {
        "path": "learner.gradient_booster.tree_train_param.max_leaves",
        "value": "0",
    },
    "eta": {
        "path": "learner.gradient_booster.tree_train_param.eta",
        "value": "0.300000012",
    },
    "lambda": {
        "path": "learner.gradient_booster.tree_train_param.lambda",
        "value": "1",
    },
    "alpha": {
        "path": "learner.gradient_booster.tree_train_param.alpha",
        "value": "0",
    },
    "min_child_weight": {
        "path": "learner.gradient_booster.tree_train_param.min_child_weight",
        "value": "1",
    },
    "min_split_loss": {
        "path": "learner.gradient_booster.tree_train_param.min_split_loss",
        "value": "0",
    },
    "subsample": {
        "path": "learner.gradient_booster.tree_train_param.subsample",
        "value": "1",
    },
    "colsample_bytree": {
        "path": "learner.gradient_booster.tree_train_param.colsample_bytree",
        "value": "1",
    },
    "colsample_bylevel": {
        "path": "learner.gradient_booster.tree_train_param.colsample_bylevel",
        "value": "1",
    },
    "colsample_bynode": {
        "path": "learner.gradient_booster.tree_train_param.colsample_bynode",
        "value": "1",
    },
    "max_bin": {
        "path": "learner.gradient_booster.tree_train_param.max_bin",
        "value": "256",
    },
    "max_delta_step": {
        "path": "learner.gradient_booster.tree_train_param.max_delta_step",
        "value": "0",
    },
    "sampling_method": {
        "path": "learner.gradient_booster.tree_train_param.sampling_method",
        "value": "uniform",
    },
    "tree_method": {
        "path": "learner.gradient_booster.gbtree_train_param.tree_method",
        "value": "auto",
    },
    "updater": {
        "path": "learner.gradient_booster.gbtree_train_param.updater",
        "value": "grow_quantile_histmaker",
    },
    "num_parallel_tree": {
        "path": "learner.gradient_booster.gbtree_model_param.num_parallel_tree",
        "value": "1",
    },
    "multi_strategy": {
        "path": "learner.learner_train_param.multi_strategy",
        "value": "one_output_per_tree",
    },
    "boost_from_average": {
        "path": "learner.learner_model_param.boost_from_average",
        "value": "1",
    },
    "seed_per_iteration": {
        "path": "learner.generic_param.seed_per_iteration",
        "value": "0",
    },
}

#: Fields of `save_config()` this harness drops from the record, and why.
#:
#: One entry. `learner.generic_param.rng_state` is the Mersenne Twister state,
#: about 625 integers and 4.5 kilobytes of text, and every XGBoost record and
#: every repeat would carry it. It is fully determined by `seed`, which IS
#: recorded, so nothing is lost that a reader could use, and a 4.5-kilobyte
#: opaque field is enough to make a diff of two records unreadable, which is
#: the one thing these records exist for. The digest and the length are kept in
#: its place so that a reader can still tell two states apart if the question
#: ever comes up.
XGBOOST_READBACK_DROPPED = {
    "learner.generic_param.rng_state": (
        "the Mersenne Twister state, about 625 integers, fully determined by "
        "the recorded seed. Replaced by its sha256 and its length so two "
        "states are still distinguishable, because a 4.5-kilobyte opaque "
        "string in every record makes a diff of two records unreadable"
    ),
}

#: What XGBoost offers for reproducibility, stated honestly and separately
#: from what we would like it to offer.
#:
#: **XGBoost has no `deterministic` parameter.** LightGBM does, and that one
#: flag is the whole reason the comparator is `stock+det` rather than plain
#: stock. Andrew asked for "XGBoost plus determinism" and the honest answer is
#: that the setting does not exist to be passed. What exists is a seed and a
#: thread count, which is the same position the CatBoost arm is in, so this
#: arm gets the same treatment and the same label: seeded, not guaranteed.
#:
#: The reason a thread count matters here rather than being incidental: the
#: `hist` tree method reduces per-thread histogram partials, and a reduction
#: over a thread pool is a summation ORDER, so a fit at one thread count is
#: not bit-guaranteed against a fit at another. Pinning `nthread` removes the
#: only source of it this harness controls.
#:
#: **The absence of the flag was verified two ways on 2026-08-17**, and not
#: from memory. No installed Python file in the `xgboost` package mentions a
#: `deterministic` parameter (the only hits for the stem are a gblinear
#: shotgun docstring, an sklearn tag, and the collective and tracker modules),
#: and the string `deterministic` does not occur in the compiled
#: `libxgboost.dylib` while the parameter names `seed_per_iteration`,
#: `validate_parameters` and `debug_synchronize` all do. So the setting is not
#: merely unset here, it does not exist to be set.
#:
#: `observed` carries a real measurement as of 2026-08-17. It is the same
#: shape as the CatBoost note's and it establishes the same limited thing.
XGBOOST_DETERMINISM = {
    "flag": (
        "XGBoost has no `deterministic` parameter. LightGBM does, which is "
        "why the comparator is stock+det. There is no XGBoost setting that "
        "asks the library for reproducible reductions, so 'XGBoost plus "
        "determinism' is not a configuration that exists. Verified two ways "
        "on xgboost 3.4.0, 2026-08-17: no installed Python file in the "
        "package declares such a parameter, and the string is absent from "
        "libxgboost.dylib while seed_per_iteration, validate_parameters and "
        "debug_synchronize are all present in it"
    ),
    "what_is_pinned": (
        "nthread, to the same number the other engines get, and seed, to the "
        "same shared seed the comparator uses. That is the whole of it. "
        "seed_per_iteration is left at its resolved 0, because changing it "
        "changes which model is fitted rather than how reproducibly"
    ),
    "status": "seeded, not guaranteed",
    "observed": (
        "checked rather than assumed, and it held everywhere it was looked "
        "at. On xgboost 3.4.0, 2000 rows by 10 features, 20 boosting rounds "
        "at reg:squarederror and seed 190019, the prediction digest was "
        "identical across three in-process repeats at each of nthread 1, 2, 4 "
        "and 8, identical ACROSS those four thread counts, and identical "
        "across three separate processes"
    ),
    "what_that_does_not_establish": (
        "one shape, one loss, one machine, 2000 rows, and no dataset with "
        "missing values or a categorical column. Bit-identity at 2000 rows is "
        "not bit-identity at 800,000, where the per-thread histogram "
        "reductions are wider and a summation order has more room to differ. "
        "The digests are recorded per repeat exactly so this is measured on "
        "every run rather than inherited from this note"
    ),
    "gating": (
        "not gating. This is a peer column: nothing in thresholds.json "
        "measures anything against it, and verify.py's differential pairs "
        "mojotrees with lightgbm and does not see these rows"
    ),
}

#: What XGBoost is passed. Three entries and no more, the same rule as
#: `CATBOOST_ALIGNMENT`: each is either a declared deviation from stock with a
#: reason, or a harness setting that changes neither the model nor the work.
#:
#: `verbosity` is the harness setting. `run.py` parses the trainer's stdout
#: for backend proof, so a library writing a progress table into that stream
#: breaks the runner rather than making the arm more faithful. It moves no
#: bit. `seed` and `nthread` are the two determinism pins argued in
#: `XGBOOST_DETERMINISM`.
#:
#: The matched parameter is deliberately NOT here, for the reason
#: `CATBOOST_ALIGNMENT` gives: it comes from `BASE_PARAMS` through
#: `XGBOOST_MATCHED` so that "matched tree count" is a structural property of
#: the translation rather than a number copied into a second dict where it can
#: drift.
XGBOOST_ALIGNMENT = {
    "verbosity": 0,
    "seed": SHARED_SEED,
}

#: The ONE shared parameter forced onto the XGBoost arm, and the `BASE_PARAMS`
#: key it is taken from. `num_boost_round` is an argument to `xgboost.train`
#: rather than a member of the parameter dict, which is why the adapter reads
#: this table rather than merging it.
XGBOOST_MATCHED = {
    "num_boost_round": "n_estimators",
}

#: Where this arm deliberately does not match. The counterpart of
#: `CATBOOST_DELIBERATE_DIVERGENCE` and it holds the same single entry, for
#: the same reason.
#:
#: XGBoost's `eta` default is 0.3 against the 0.1 the comparator and the plain
#: mojotrees arm run. It is NOT pinned to 0.1, because pinning it would make
#: the arm something other than XGBoost as shipped, which is the only thing
#: this arm is for. The consequence has to be stated wherever the accuracy
#: column is read: a 100-tree budget at 0.3 and a 100-tree budget at 0.1 are
#: different amounts of fitting, so an accuracy difference between this arm
#: and the comparator is a learning-rate-times-budget interaction and is NOT
#: evidence about either engine. This is the identical trap
#: `CATBOOST_DELIBERATE_DIVERGENCE` documents, and it is worse here in one way
#: and better in another: worse because 0.3 is further from 0.1 than
#: CatBoost's resolved rate usually is, better because 0.3 is a STATIC
#: default that a read-back confirms rather than a value derived per fit.
XGBOOST_DELIBERATE_DIVERGENCE = {
    "learning_rate": {
        "xgboost": (
            "XGBoost's own default for `eta`, read back as 0.300000012 on "
            "3.4.0, which is float32(0.3). The arm passes nothing and the "
            "resolved value is recorded per fit from save_config()"
        ),
        "lightgbm": "BASE_PARAMS['learning_rate'] = 0.1, unchanged",
        "mojotrees": "BASE_PARAMS['learning_rate'] = 0.1, unchanged",
        "why": (
            "pinning it would make this arm something other than XGBoost as "
            "shipped, which is the only thing it is for. Read the accuracy "
            "column knowing that 100 trees at XGBoost's rate and 100 trees "
            "at 0.1 are different amounts of fitting"
        ),
    },
}

#: Parameters refused by name if a caller tries to pass them to this arm.
#:
#: The same defect list as `CATBOOST_REFUSED_PARAMS`, in XGBoost's
#: vocabulary, and it exists for the reason that one does: the
#: `bin_construct_sample_cnt` mistake, where a binning parameter derived from
#: the row count made a competitor do strictly more work than us in our
#: favor, and was caught only after a ratio had been published. A refusal is
#: the only form of that rule which survives somebody adding a third call
#: site.
XGBOOST_REFUSED_PARAMS = {
    "max_bin": (
        "the binning budget. It stays stock for the same reason max_bin is "
        "not pinned on the LightGBM side and border_count is not pinned on "
        "CatBoost's. XGBoost's resolved 256 and this harness's 255 do count "
        "the same quantity, which was measured rather than assumed, so the "
        "gap is one bin and it is declared rather than closed: see "
        "XGBOOST_UNMATCHABLE['bin_budget']"
    ),
    "learning_rate": "XGBoost's own default. See XGBOOST_DELIBERATE_DIVERGENCE",
    "eta": "a synonym for learning_rate in XGBoost's own vocabulary",
    "max_depth": (
        "the depth bound, which is the parameter that DEFINES this arm's "
        "tree shape. Pinning it makes the row neither XGBoost's defaults nor "
        "a comparison to them"
    ),
    "grow_policy": (
        "the growth order, and depthwise is XGBoost's default. Passing it "
        "would either restate the default or turn this arm into a lossguide "
        "row wearing the shipped-defaults label"
    ),
    "max_leaves": (
        "the leaf cap. XGBoost's default is 0, meaning unbounded, and under "
        "depthwise the tree is bounded by max_depth alone. Pinning a leaf "
        "count here would silently make this a different tree from the one "
        "XGBoost ships"
    ),
    "min_child_weight": (
        "the leaf-population rule, expressed as a hessian sum. This column "
        "is XGBoost's defaults. Raising it to this harness's shared value "
        "would make the arm neither XGBoost's defaults nor ours"
    ),
    "subsample": (
        "row sampling. Stock is 1.0, meaning off, and matching it to a "
        "bagging fraction on our side would be matching a number across two "
        "different sampling methods"
    ),
    "sampling_method": "a sampling knob, and sampling is left at stock",
    "colsample_bytree": "column sampling, left at stock",
    "colsample_bylevel": "column sampling, left at stock",
    "colsample_bynode": "column sampling, left at stock",
    "reg_lambda": "the L2 term, left at XGBoost's own default",
    "lambda": "a synonym for reg_lambda",
    "reg_alpha": "the L1 term, left at XGBoost's own default",
    "alpha": "a synonym for reg_alpha",
    "tree_method": (
        "which tree-construction algorithm runs. `hist` is the default from "
        "2.0 and XGBOOST_MIN_VERSION exists partly to guarantee it. Passing "
        "`approx` or `exact` would put a different algorithm under this "
        "arm's label"
    ),
    "device": (
        "which backend runs. Set by the adapter from the runner's device, "
        "exactly as thread_count is on the CatBoost side. A second source "
        "for it is how one arm ends up on a backend the record does not name"
    ),
    "nthread": (
        "set from the runner's thread count by xgboost_params itself, "
        "exactly as num_threads is on the LightGBM side and thread_count on "
        "CatBoost's"
    ),
    "n_jobs": "a synonym for nthread in XGBoost's sklearn API",
}

#: Differences no parameter can close, as distinct from
#: `XGBOOST_DELIBERATE_DIVERGENCE`, which is about differences that COULD be
#: closed and are not on purpose.
XGBOOST_UNMATCHABLE = {
    "determinism": (
        "XGBoost has no `deterministic` flag, so the comparator's one "
        "deviation from stock has no counterpart here and this arm cannot be "
        "given the guarantee the comparator carries. See XGBOOST_DETERMINISM"
    ),
    "bin_budget": (
        "REWRITTEN 2026-08-17, and the previous entry was wrong in the way "
        "that matters. It said XGBoost's `max_bin` was 'believed' to count "
        "bins while this harness's 255 was a border count, so the two were "
        "not the same quantity. Both halves are false. This repo's `max_bin` "
        "counts BINS and says so (binning.mojo::CATBOOST_DEFAULT_BORDER_COUNT, "
        "which distinguishes it "
        "from CatBoost's border_count precisely because that one counts "
        "thresholds). And XGBoost's counts bins too, MEASURED rather than "
        "believed: QuantileDMatrix.get_quantile_cut() returns max_bin + 1 cut "
        "values per continuous feature, the first -inf and the last the "
        "maximum plus an epsilon, so the reachable bins are exactly max_bin; "
        "at 5 distinct values and max_bin 256 it returns 6 cuts, one bin per "
        "distinct value, which is the same level rule this repo ships. "
        "\n\nSo the honest gap is ONE BIN, XGBoost's resolved 256 against "
        "this harness's shared 255, and it is NOT taken. Two reasons, and the "
        "second is the binding one. First, `max_bin` is a BINNING parameter "
        "carried by `dataset_params`, and MOJOTREES_DEPTHWISE is a TRAINING "
        "override applied to the dict `mojotrees_params` returns, so putting "
        "max_bin in it would hand a binning key to `mojotrees.train`, which "
        "does not take one; the vehicle would have to be "
        "`arm_dataset_params` or a `_dataset_params` override on the engine. "
        "Second, and this is the reason it is left alone rather than plumbed: "
        "256 is inside our ceiling (binning.mojo::MAX_BINS is 256, because "
        "a bin index is one UInt8) and `fit_bins` admits [2, 256], but NO FIT "
        "at 256 bins has ever been run on either backend in this repository, "
        "and the session that wrote this could not build to try one. An "
        "unverified bin budget on an arm that also runs GPU cells buys one "
        "bin and risks a dead cell, and a dead cell withholds the quality "
        "verdict for the whole matrix. Whoever lifts the build stop should "
        "decide this; until then the arm runs 255 and this entry is the "
        "declaration"
    ),
    "learning_rate_precision": (
        "XGBoost stores `eta` as a float and resolves 0.3 to 0.300000012, "
        "read back on 3.4.0. MOJOTREES_DEPTHWISE passes the Python float 0.3, "
        "which is the nearest double. So the two arms run learning rates that "
        "differ in about the eighth significant figure. Recorded because it "
        "is a real difference and dismissed because it is far below the "
        "spread of any measurement this harness takes; it is here so nobody "
        "spends an afternoon on it later"
    ),
    "leaf_row_count": (
        "XGBoost has NO row-count constraint on a leaf. It bounds a leaf by "
        "hessian mass alone through min_child_weight, and no parameter in the "
        "resolved configuration counts rows. `min_data_in_leaf` cannot "
        "therefore be MATCHED, only disarmed, and MOJOTREES_DEPTHWISE sets it "
        "to 1 on the argument that a leaf holding no rows is not produced by "
        "any split, so 1 and 'no rule' select the same trees. That is an "
        "argument rather than an identity and it is written here as one"
    ),
    "binning_phase": (
        "XGBoost quantizes inside `train` when handed a plain DMatrix, so "
        "this arm reports no separate binning phase, the same position the "
        "CatBoost arm is in. `QuantileDMatrix` would separate it and is "
        "deliberately not used, because constructing one is a different "
        "ingestion path from the one a default XGBoost user takes"
    ),
    "leaf_bound": (
        "XGBoost's depthwise growth is bounded by max_depth with max_leaves "
        "read back as 0, meaning unbounded, so a depth-6 tree may hold 64 "
        "leaves. Our growth policy keeps `num_leaves` a HARD bound and admits "
        "a gain-ranked prefix of the level that would overshoot "
        "(growth_policy.mojo module docstring, 'The three rules that make a "
        "depth-wise tree well defined'). The two are the same tree only when the "
        "leaf budget is at least 2**max_depth, which is why "
        "MOJOTREES_DEPTHWISE sets num_leaves to 64. It is a CAP that matches "
        "rather than the same rule: XGBoost has no leaf budget to overshoot, "
        "so on a tree that would have grown fewer than 64 leaves the two "
        "agree exactly, and 64 is where they can only agree"
    ),
    "algorithm_label": (
        "the resolved `tree_method` is the string 'auto' and not 'hist'. What "
        "says the histogram algorithm ran is the resolved `updater`, "
        "grow_quantile_histmaker. Recorded here because a reader checking "
        "that our histogram arm was compared against a histogram arm will "
        "look at tree_method first and will not find the answer there"
    ),
}

#: XGBoost's objective per task. Not asserted as a default: these are passed,
#: because an objective is the problem statement rather than a library
#: preference, exactly as `CATBOOST_LOSS` is passed.
XGBOOST_LOSS = {
    "regression": "reg:squarederror",
    "binary": "binary:logistic",
    "multiclass": "multi:softprob",
}

#: Whether the XGBoost arm runs each scenario, and why not when it does not.
#: `None` means it runs. A scenario with no entry is REFUSED rather than
#: assumed, for the reason `catboost_supports` gives: an unlisted scenario is
#: one nobody decided about.
#:
#: Ranking is refused rather than attempted. XGBoost has `rank:ndcg` and a
#: group interface, so this is a scoping decision and not a capability claim:
#: the comparator's ranking configuration took real work to match and doing
#: that again for a third library is its own task. Written as a refusal with a
#: reason so it is visibly undone rather than invisibly missing.
XGBOOST_SCENARIO_SUPPORT = {
    "dense_regression": None,
    "imbalanced_binary": None,
    "multiclass": None,
    "ordered_boosting_small": None,
    "sparse_highdim": None,
    "categorical_missing": (
        "XGBoost's categorical support needs `enable_categorical` and a "
        "pandas categorical dtype, which is a third distinct categorical "
        "container in this harness beside LightGBM's index list and "
        "CatBoost's mixed frame. Each one needs its own reconstruction proof "
        "that the container did not change a value, and that proof is the "
        "work, not the plumbing"
    ),
    "high_cardinality_categorical": (
        "same reason as categorical_missing: XGBoost needs a third "
        "categorical container and its own reconstruction proof"
    ),
    "ranking": (
        "not yet wired. XGBoost has rank:ndcg and a group interface, so this "
        "is a scoping decision rather than a capability claim"
    ),
}

#: The XGBoost engines as a group, the counterpart of `CATBOOST_ENGINES`. One
#: entry, because `xgboost` is the only engine here that IS XGBoost.
#:
#: `mojotrees_depthwise` is deliberately absent for the reason the CatBoost
#: comment gives: it is us shaped toward XGBoost, not XGBoost, so it neither
#: takes XGBoost's refusals nor belongs to the group `run.py` puts on the CPU
#: because XGBoost has no accelerator here. Our depthwise arm runs on both
#: backends and that is the point of it.
XGBOOST_ENGINES = ("xgboost",)


def xgboost_arm_id():
    """The one-line identity of the XGBoost peer arm."""
    return f"{XGBOOST_ARM_ID}@v{XGBOOST_ARM_VERSION}"


def xgboost_arm_block():
    """Everything a record needs to say what the XGBoost arm was.

    Same rule as `comparator_block` and `catboost_arm_block`: a results file
    cannot fail to state its own configuration. The two fields to read first are
    `determinism`, because XGBoost has no `deterministic` flag and its
    like-for-like is therefore weaker than the comparator's, and
    `xgboost_resolved_defaults`, because that is the table the
    `mojotrees_depthwise` mirror arm was built from and the one every fit is
    checked against.
    """
    return {
        "id": XGBOOST_ARM_ID,
        "version": XGBOOST_ARM_VERSION,
        "label": XGBOOST_ARM_LABEL,
        "registered": XGBOOST_ARM_REGISTERED,
        # The same field `catboost_arm_block` carries under the same name. It
        # was missing, and `peer_banner` reads it, so the console banner raised
        # a KeyError the first time both peers were printed.
        "one_line": xgboost_arm_id(),
        "is_the_comparator": False,
        "defaults_source": XGBOOST_DEFAULTS_SOURCE,
        # The asserted defaults, so a published table carries the values the
        # mirror arm was built from rather than requiring a reader to open
        # scenarios.py. Every measured XGBoost record also carries
        # engine_resolved_params_drift, which is this table checked live.
        "xgboost_resolved_defaults": copy.deepcopy(XGBOOST_RESOLVED_DEFAULTS),
        "xgboost_readback_dropped": dict(XGBOOST_READBACK_DROPPED),
        # The mojotrees arm this column is read against, named here so a reader
        # of the peer block does not have to know the pairing by heart.
        "mirrored_by": "mojotrees_depthwise",
        "mirror_params": dict(MOJOTREES_DEPTHWISE),
        "mirror_claims": MOJOTREES_DEPTHWISE_CLAIMS,
        "xgboost_passed": dict(XGBOOST_ALIGNMENT),
        "xgboost_matched_from_base_params": dict(XGBOOST_MATCHED),
        "xgboost_deliberate_divergence": copy.deepcopy(
            XGBOOST_DELIBERATE_DIVERGENCE
        ),
        "xgboost_refused_params": dict(XGBOOST_REFUSED_PARAMS),
        "xgboost_unmatchable": dict(XGBOOST_UNMATCHABLE),
        "xgboost_min_version": ".".join(str(p) for p in XGBOOST_MIN_VERSION),
        "determinism": copy.deepcopy(XGBOOST_DETERMINISM),
        "scenarios_not_run": {
            name: reason
            for name, reason in XGBOOST_SCENARIO_SUPPORT.items()
            if reason is not None
        },
    }


def check_xgboost_version(version):
    """Raise unless this XGBoost is new enough to be the peer arm.

    Called before anything is fitted, for the reason the other two guards
    are: an engine that silently ignores a parameter and trains anyway
    produces a record naming a configuration it did not run. Here the stakes
    are specifically `save_config()`, because this arm asserts no defaults of
    its own and a version without the read-back would produce a record with
    no resolved configuration at all.
    """
    parts = []
    for piece in str(version).split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts) < XGBOOST_MIN_VERSION:
        want = ".".join(str(p) for p in XGBOOST_MIN_VERSION)
        raise RuntimeError(
            f"the XGBoost peer arm is {XGBOOST_ARM_LABEL} and this harness "
            f"needs XGBoost {want} or newer. This environment has {version}. "
            "Below 2.0 there is no `device` parameter, `tree_method` does not "
            "default to hist, and Booster.save_config() may not return the "
            "resolved configuration this arm records instead of asserting "
            "defaults it never read"
        )


def xgboost_supports(scenario):
    """(runs, reason). Whether the XGBoost arm runs this scenario."""
    scenario_id = scenario["id"] if isinstance(scenario, dict) else scenario
    if scenario_id not in XGBOOST_SCENARIO_SUPPORT:
        return False, (
            f"{scenario_id} has no entry in XGBOOST_SCENARIO_SUPPORT, so "
            "nobody has decided whether the XGBoost arm can run it"
        )
    reason = XGBOOST_SCENARIO_SUPPORT[scenario_id]
    return (reason is None), reason


def xgboost_params(spec, threads, extra=None):
    """`shared_params` translated into an XGBoost parameter dict.

    Only three things reach XGBoost that are not the problem statement:
    `XGBOOST_ALIGNMENT`, the thread count, and the device. Everything else is
    XGBoost's own default and is recorded per fit from `save_config()` rather
    than passed or asserted.

    The matched tree count is NOT in this dict, because `num_boost_round` is
    an argument to `xgboost.train` rather than a parameter. `xgboost_rounds`
    resolves it from the same merge.

    **`n_estimators` is consumed here and never copied through, and that was a
    bug rather than a style choice.** The pass-through loop at the bottom
    copies every `extra` key onto the parameter dict, and the arm dimension
    puts the tree count into `extra` under the `BASE_PARAMS` name, so a
    frontier arm sweeping tree counts handed XGBoost a booster parameter named
    `n_estimators`. The native API takes its round count as an argument and
    IGNORES that key in silence at verbosity 0, verified on 3.4.0, so the fit
    was correct and `params_used` recorded a parameter the fit did not use.
    That is exactly the class of defect this whole module exists to prevent, so
    the key is skipped beside `num_class` and read only by `xgboost_rounds`.
    """
    for refused, why in XGBOOST_REFUSED_PARAMS.items():
        if refused in (extra or {}):
            raise ValueError(
                f"{refused} was passed to the XGBoost peer arm. It is stock "
                f"in {XGBOOST_ARM_LABEL}: {why}. See "
                "XGBOOST_REFUSED_PARAMS and XGBOOST_UNMATCHABLE."
            )
    task = spec["task"]
    if task not in XGBOOST_LOSS:
        raise ValueError(
            f"no XGBoost objective registered for task {task!r}: see "
            "XGBOOST_LOSS"
        )
    shared = shared_params(spec, extra)
    params = {
        "objective": XGBOOST_LOSS[task],
        "nthread": int(threads),
    }
    params.update(XGBOOST_ALIGNMENT)
    if task == "multiclass":
        # Named rather than left as the KeyError this used to raise. A
        # multiclass scenario carries its class count in the tier sizes and
        # not on the spec, so the adapter has to read it off the loaded data
        # and pass it, exactly as the LightGBM and CatBoost adapters do. When
        # it does not, the failure was `KeyError: 'num_class'` out of a
        # translator, which tells the reader nothing about whose job it was.
        if "num_class" not in shared:
            raise ValueError(
                "the XGBoost peer arm was asked for a multiclass fit without "
                "num_class. A scenario's class count is in its tier sizes "
                "rather than on the spec, so the ENGINE ADAPTER reads it off "
                "the loaded labels and passes it in `extra`, the way "
                "LightGBMEngine.run and CatBoostEngine.run both do. See "
                "XGBoostEngine.run"
            )
        params["num_class"] = int(shared["num_class"])
    for key, value in (extra or {}).items():
        if key in ("num_class", "n_classes", XGBOOST_MATCHED["num_boost_round"]):
            continue
        params[key] = value
    return params


def xgboost_rounds(spec, extra=None):
    """The XGBoost arm's boosting-round count, from one merge read once.

    `num_boost_round` is an argument to `xgboost.train` and not a booster
    parameter, so it cannot travel inside the dict `xgboost_params` returns and
    something has to resolve it separately. This is that something, and it
    exists rather than being three lines at the call site for the reason
    `MojoTreesEngine._n_estimators`'s docstring records: the tree count is the
    one parameter a frontier sweep moves first, and when a call site reads
    `BASE_PARAMS` directly, no caller can move it.

    It resolves through `shared_params`, which is what the CatBoost arm does
    (`catboost_params` reads `shared[CATBOOST_MATCHED["iterations"]]`) and what
    the LightGBM adapter does (`params.get("n_estimators", ...)` off the
    translated dict). The mojotrees arms read the arm override over
    `BASE_PARAMS` and skip a scenario-level `params` entry; that difference is
    theirs and is not copied here, because this arm is a competitor column and
    a competitor asked for a different budget from the comparator is not a
    comparison. Pass the SAME `extra` that went to `xgboost_params`.
    """
    return int(shared_params(spec, extra)[XGBOOST_MATCHED["num_boost_round"]])


def xgboost_config_value(resolved, path):
    """One dotted path out of a parsed `Booster.save_config()`, or None.

    None means the path is absent, which is a real answer rather than a
    failure: a version that moves a parameter into another block should be
    reported as drift and not raise inside a reporting field.
    """
    node = resolved
    for piece in path.split("."):
        if not isinstance(node, dict) or piece not in node:
            return None
        node = node[piece]
    return node


def check_xgboost_readback(resolved):
    """Where a live fit disagrees with `XGBOOST_RESOLVED_DEFAULTS`.

    Returns a list of sentences, empty when every asserted default was read
    back unchanged. The counterpart of `check_catboost_readback` and it exists
    for the same reason: the table beside it is a transcription somebody made
    on one version, and a transcription nothing reads back is a set of claims
    rather than facts.

    Recorded by the adapter rather than raised. An XGBoost upgrade that moves a
    default is a thing to find out about from a record; losing a measured cell
    to it would cost the whole matrix's quality verdict, and the drift list
    reaches the same reader either way.

    `save_config()` returns every scalar as a string, so the comparison is on
    strings and the table stores strings. Nothing here rounds or parses: a
    resolved `eta` of 0.300000012 must be read as the float32 it is, and
    comparing it as a float would need a tolerance nobody could justify.
    """
    drift = []
    for name, entry in XGBOOST_RESOLVED_DEFAULTS.items():
        got = xgboost_config_value(resolved, entry["path"])
        if got is None:
            drift.append(
                f"{name}: no value at {entry['path']} in this fit's "
                f"save_config(), where XGBOOST_RESOLVED_DEFAULTS asserts "
                f"{entry['value']!r}. Either the parameter moved in this "
                "version or the path was transcribed wrongly"
            )
        elif str(got) != entry["value"]:
            drift.append(
                f"{name}: this fit resolved {str(got)!r} at {entry['path']} "
                f"where XGBOOST_RESOLVED_DEFAULTS asserts {entry['value']!r}. "
                "MOJOTREES_DEPTHWISE mirrors these values, so read that dict "
                "before reading the two rows as like-for-like"
            )
    return drift


def xgboost_readback_for_record(resolved):
    """`save_config()` as a record should carry it, with the dropped fields
    replaced by a digest.

    See `XGBOOST_READBACK_DROPPED` for the one field this touches and why. The
    input is not mutated; every other key is passed through unchanged, so this
    is a reduction of the record's size and never of what it says.
    """
    if not isinstance(resolved, dict):
        return resolved
    out = copy.deepcopy(resolved)
    for path, why in XGBOOST_READBACK_DROPPED.items():
        pieces = path.split(".")
        node = out
        for piece in pieces[:-1]:
            if not isinstance(node, dict) or piece not in node:
                node = None
                break
            node = node[piece]
        leaf = pieces[-1]
        if isinstance(node, dict) and leaf in node:
            raw = str(node[leaf])
            node[leaf] = {
                "dropped": True,
                "reason": why,
                "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "length": len(raw),
            }
    return out


# ---------------------------------------------------------------------------
# `MOJOTREES_XGBOOST_MODE` WAS HERE AND IS GONE, 2026-08-17.
# ---------------------------------------------------------------------------
#
# It was a SECOND mojotrees arm, `mojotrees_xgboost_mode`, carrying XGBoost's
# defaults while `MOJOTREES_DEPTHWISE` stayed a one-key growth-order isolation.
# Andrew removed it the same day it was written, in these words: "make our
# depthwise params match xgboost". One arm, not two. The instruction is a
# scheduling decision as much as a design one, because an arm is a column of
# benchmark cells across every scenario, tier and backend, and this suite runs
# on one laptop under a timing lock.
#
# Where its content went. Every value it held is now in `MOJOTREES_DEPTHWISE`,
# read back off a live fit rather than transcribed; its scenario-support table
# is gone because the depthwise arm keeps its own, which is wider; its
# translator is gone because `mojotrees_depthwise_params` was already the same
# function under another name. `engines.MojoTreesXGBoostModeEngine` and the
# `verify.SUBJECT_ENGINES` entry went with it.
#
# What was LOST, stated because it was a real thing and a reader of the
# depthwise row should know it is no longer available. There is no longer any
# arm that changes the growth order and NOTHING else, so the growth order can
# no longer be priced on its own: `mojotrees_depthwise` against plain
# `mojotrees` is now a seven-parameter difference. Restoring that measurement
# means restoring an arm, and the argument for it is in git history at this
# line rather than lost.


#: The peer arms as a group, for callers that mean "anything reported beside
#: the comparator rather than as it".
#:
#: `mojotrees_depthwise` is NOT here even though it now mirrors XGBoost's
#: defaults, and the reason is the one `selfcheck.py` enforces: every member of
#: this tuple is asserted to carry an `engines.ENGINE_ARM` value beginning with
#: "peer", and a peer is a competitor library or our side of a competitor
#: PAIRING. The depthwise arm is a subject variant, it is judged by the subject
#: gates in `verify.py`, and moving it here would take it out of the backend
#: proof and the cpu-versus-gpu agreement check.
PEER_ENGINES = (
    CATBOOST_ENGINES
    + XGBOOST_ENGINES
    + ("mojotrees_catboost_mode",)
)


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
    `CATBOOST_ALIGNMENT`, the thread count, and the one matched parameter in
    `CATBOOST_MATCHED`. Everything else is CatBoost's own default and is
    recorded in `CATBOOST_LEFT_AT_STOCK` -- or, when CatBoost derives it per
    fit, in `CATBOOST_RESOLVED_PER_FIT` -- rather than passed.

    **`learning_rate` is not passed as of 2026-08-16.** It used to be, from
    `BASE_PARAMS`, and removing it is what turned `cb-default` into
    `cb-shipped`. CatBoost now resolves its own rate, which is roughly 0.4273
    at 100 iterations rather than the 0.1 the other two arms run.
    `CATBOOST_DELIBERATE_DIVERGENCE` carries the decision, the measurements
    and the sentence that stops the resulting accuracy column being read as an
    engine claim.

    **`CATBOOST_ALIGNMENT` survives the change and is worth defending here**,
    because "pass CatBoost nothing" would sweep it away with the rate and it
    must not. `allow_writing_files=False` removes a `catboost_info` directory
    write from inside the timed region, which is filesystem work neither other
    engine does; `logging_level="Silent"` keeps CatBoost's progress table off
    the stdout stream `run.py` parses for backend proof, so turning it on
    breaks the runner rather than making the arm more faithful. Neither moves
    a bit of the model. Both are declared, one as a deviation with an exit
    condition and one as a harness setting.

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
        # No `learning_rate`. See the docstring and
        # CATBOOST_DELIBERATE_DIVERGENCE: CatBoost resolves its own, and the
        # value it resolves is read back per fit rather than predicted here.
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


def mojotrees_catboost_mode_params(
    spec, device, extra=None, catboost_readback=None
):
    """The "us in CatBoost mode" arm: `mojotrees_params` with
    `MOJOTREES_CATBOOST_MODE` applied over the shared defaults.

    Applied as a scenario-level override rather than as a second translator,
    so this arm and the plain one go through exactly the same code and a
    reader diffing two records sees `MOJOTREES_CATBOOST_MODE` and nothing
    else.

    What this arm is NOT: a claim that mojotrees can be made into CatBoost.
    It cannot, but the list of reasons is shorter than it was. As of
    2026-08-16 this arm grows **symmetric** trees at depth 6, scores splits
    with **Cosine**, and adds CatBoost's **random_strength** noise -- three
    things it could not ask for a day earlier, not because the package
    lacked them but because `python/mojotrees/sklearn.py` refused the keys
    and this harness reaches only what that file validates.

    It **samples rows** as of later the same day, which this paragraph used
    to say it did not. CatBoost's default MVS takes about 80 percent of them
    per tree, weighted by gradient magnitude, and this arm now asks for it by
    name: `bootstrap_type` parses into a `sampling.BootstrapParams` at the
    binding and reaches `boosting.train` through `model.fit` and
    `trainset.train_dataset`.

    **On four of the six scenarios that run this arm.** The multiclass and
    sparse trainers call no `bootstrap_round` and refuse the bundle by name,
    so the `multiclass` and `sparse_highdim` cells raise rather than
    producing an unsampled row labelled "CatBoost mode". Read
    `CATBOOST_UNMATCHABLE["row_sampling"]` and the header comment on
    `MOJOTREES_CATBOOST_MODE` before scheduling a matrix; it travels with the
    record either way.

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

    **`catboost_readback` is required as of 2026-08-16 and the arm refuses
    without it.** Every key in `MOJOTREES_CATBOOST_MODE_FROM_READBACK` --
    today that is `learning_rate` and only that -- has no static value,
    because CatBoost derives it from the fit. The argument for refusing rather
    than falling back to `BASE_PARAMS['learning_rate']` is the whole point of
    this lane: a fallback would be a hand-written belief about CatBoost's
    resolution, silently wrong by a factor of four at 100 iterations, in the
    single parameter that moves a metric most.
    """
    params = mojotrees_params(spec, device, extra)
    params.update(MOJOTREES_CATBOOST_MODE)
    # Removals before additions, and removals at all, because `update` can
    # override a key and cannot delete one. Empty today; the key this exists
    # for is `learning_rate`, on the day mojotrees derives CatBoost's automatic
    # rate itself and needs the parameter UNSET for its own gate to fire. See
    # MOJOTREES_CATBOOST_MODE_UNSET and CATBOOST_LEARNING_RATE_TRANSITION.
    for key in MOJOTREES_CATBOOST_MODE_UNSET:
        params.pop(key, None)
    params.update(catboost_readback_values(spec, catboost_readback))
    return params


def mojotrees_depthwise_params(spec, device, extra=None):
    """The depth-wise arm: `mojotrees_params` with `MOJOTREES_DEPTHWISE`
    applied over the shared defaults.

    The same shape as `mojotrees_catboost_mode_params` and for the same
    reason its docstring gives: the override is applied to the RESOLVED dict
    rather than to the scenario's `params`, because `mojotrees_params` builds
    an explicit dict and copies through only the categorical, ranking and
    multiclass blocks, so a scenario-level `grow_policy` would be dropped on
    the floor. That defect was found once already on the CatBoost-mode arm,
    where it produced an arm with CatBoost's depth and mojotrees's growth,
    which is neither of the two things the row claims to compare.

    Simpler than the CatBoost-mode translator in the one way that matters:
    there is NO CROSS-CELL READ-BACK, and this stayed true when the arm became
    an XGBoost mirror on 2026-08-17. CatBoost derives its learning rate from
    the data, so the CatBoost-mode arm has to be handed CatBoost's resolved
    value out of a cell that already ran. XGBoost's defaults are static, so
    `MOJOTREES_DEPTHWISE` holds them directly and this function needs nothing
    from another engine. The read-back that DOES exist for this arm is a check
    rather than an input: `check_xgboost_readback` re-reads the same values off
    every measured XGBoost fit and records drift, so a version that moves a
    default is reported instead of quietly making the mirror wrong.
    """
    params = mojotrees_params(spec, device, extra)
    params.update(MOJOTREES_DEPTHWISE)
    return params


# ---------------------------------------------------------------------------
# THE TWO CORRECTNESS ARMS, added 2026-08-17.
# ---------------------------------------------------------------------------
#
# NOTE ON PLACEMENT, and it follows the precedent set by the note above
# `MOJOTREES_DEPTHWISE`. These two dicts belong up with the other override
# dicts by convention and they are down here instead, beside their own
# translators, because six lanes are live in this checkout and two of them are
# editing the region those dicts live in. One contiguous block is a diff about
# an arm; a scattered one is a diff about whoever merged last. Move it up when
# the checkout is quiet.
#
# WHY THESE ARMS EXIST, and it is not the reason the other variant arms exist.
# `mojotrees_depthwise` mirrors XGBoost and `mojotrees_catboost_mode` mirrors
# CatBoost: both are COMPARISON arms and both are read against a peer column.
# These two are read against NOTHING external. They exist so that
# `verify.check_device_agreement` has a cpu-versus-gpu pair on a configuration
# it did not previously cover, and their whole product is that comparison.
#
# THE INCIDENT. On 2026-08-17 two live wrong answers were found and fixed in
# the trainer, and NEITHER was found by this harness, because no arm here set
# either parameter to a non-default value. Both were found by reading code.
#
#   1. The oblivious (symmetric) GPU path returned a wrong answer whenever
#      `feature_fraction < 1`. `GpuLeafBatcher.feat_dev` was never set by the
#      level build, so `_batch_hist_atomic_kernel` wrote feature slice `slot`
#      while the split searcher, whose own table IS set from `tree_features`,
#      read slice `active[slot]`, and the root disagreed with its own
#      children. Fixed by one `set_shared_features(self.active)` call in
#      `histogram_gpu.stage_desc_level_plan`, whose comment records the whole
#      trace. At `feature_fraction = 1.0` the identity is correct and nothing
#      shows, which is the entire coverage story: every scenario in this file
#      runs at 1.0.
#   2. `score_function='Cosine'` was silently ignored on every LEAF-WISE GPU
#      fit. `GpuSplitSearcher.set_score_function` had no production caller, so
#      `score_function_code` stayed at its constructed `SCORE_L2`, and
#      `gpu_resident_round._launch_child_search` called `_launch_search`
#      without naming the argument and took the same default. Fixed at
#      `train_gpu.mojo::GpuSplitSearcherCache.reset_for_tree` and in
#      `gpu_resident_round`. The one arm here
#      that sets Cosine, `mojotrees_catboost_mode`, grows SYMMETRIC trees, so
#      it never went down the leaf-wise path this bug lived on.
#
# Both fixes are compiled into the extension these arms run against, so
# neither arm is expected to fail today. That is the point. A green arm here
# is a statement that the fix is still in, and a red one is the harness
# catching the regression it missed the first time.
#
# WHAT DETECTS THE DEFECT, stated exactly, because these arms are worth their
# cells only if this is true. `verify.check_device_agreement` keys on
# `(scenario, threads, engine)`, requires `record["engine"] in
# verify.SUBJECT_ENGINES`, and compares the gpu row's saved predictions
# against its own cpu twin's row by row at `max_abs_prediction_diff`. Bug 1
# corrupts the device histogram and leaves the CPU untouched, so the two rows
# diverge and the check fires. Bug 2 makes the device score L2 while the CPU
# scores Cosine, so the two grow different trees and the check fires. Neither
# arm needs a control cell at `feature_fraction = 1.0` or at `L2`, because the
# detector is the arm's own cpu twin and not another arm.
#
# THE COST, and why it is small. Neither name is in `run.py`'s default
# `--engine` list, which is `["mojotrees", "lightgbm"]`, so these arms add
# ZERO cells to every run that does not ask for them by name. When they are
# asked for, each is one scenario on two backends, and the cpu backend is an
# ORACLE cell that runs `--oracle-repeats` times, which defaults to 1. So the
# pair costs `2 * (repeats + 1)` fits and no more. `CORRECTNESS_ARM_TIER_CAP`
# below refuses the large tier, which is the only way that number could grow
# without somebody typing it.

#: The symmetric-plus-column-sampling arm's override. The configuration bug 1
#: lived in, and the smallest configuration that reaches it.
#:
#: JUSTIFICATION PER KEY, because a dict without one is indistinguishable from
#: whatever happened to run:
#:
#: - `grow_policy="symmetrictree"`. The bug was in the OBLIVIOUS level build
#:   (`histogram_gpu.stage_desc_level_plan`), which no other growth order
#:   reaches. This is the canonical spelling
#:   (`python/mojotrees/sklearn.py:_CANONICAL_GROW_POLICIES`) and it is the
#:   same value `MOJOTREES_CATBOOST_MODE` carries, so this arm is on a path
#:   that is already known to fit through `mojotrees.train(params, Dataset)`
#:   on the numeric scenario below.
#: - `max_depth=6`. **REQUIRED, not a preference.** `tree.mojo::_check_oblivious` raises
#:   "grow_policy=oblivious requires max_depth > 0" and `BASE_PARAMS` carries
#:   -1, so an arm without this key does not run at all. 6 rather than another
#:   positive number because it is what `MOJOTREES_CATBOOST_MODE` uses and is
#:   CatBoost's own default, so this arm's tree shape is one that has already
#:   been measured here rather than a new one.
#: - `num_leaves=64`. `2**max_depth`, which is what a symmetric tree at depth
#:   6 holds. The shared default of 31 would be a hard bound below the shape
#:   the depth implies. Same value and same arithmetic as
#:   `MOJOTREES_CATBOOST_MODE` and `MOJOTREES_DEPTHWISE`.
#: - `feature_fraction=0.5`. **The key this arm exists for.** See
#:   `SYMMETRIC_COLSAMPLE_FRACTION_CHOICE` for why 0.5 and not another value.
#:
#: WHAT IS DELIBERATELY ABSENT, and each absence is a device refusal this arm
#: must not trip. `score_function`, `random_strength` and `bootstrap_type` are
#: the three keys that keep `mojotrees_catboost_mode` off four scenarios, and
#: carrying any of them here would mean a red cell could be any of several
#: things. `feature_fraction_bynode` and `feature_fraction_bylevel` are absent
#: for a harder reason: `oblivious_device_supported` REFUSES both, and they
#: are not fields of a `DeviceRequest`, so no `device_policy` block can see
#: them and `train_gpu.mojo::_grow_tree_gpu_device_search` raises rather than
#: declining. An arm carrying
#: either would be an infrastructure failure, which this harness answers by
#: withholding the quality verdict for the whole matrix. The per-tree
#: `feature_fraction` is refused by nothing, which is exactly why the bug was
#: reachable.
MOJOTREES_SYMMETRIC_COLSAMPLE = {
    "grow_policy": "symmetrictree",
    "max_depth": 6,
    "num_leaves": 64,
    "feature_fraction": 0.5,
}

#: Why the fraction is 0.5, recorded because "pick a value that bites" is a
#: judgement and a judgement with no argument beside it decays into a magic
#: number.
#:
#: The failure mode decides the value. `sampling.select_tree_features` returns
#: a SORTED subset and returns the identity only at `fraction >= 1.0`, so the
#: broken read was slice `active[slot]` of a histogram written at slice
#: `slot`, and `active[slot] >= slot` always. The size of the wrong answer is
#: therefore the size of the index shift, which grows as the subset thins.
#:
#: At a fraction near 1 the subset covers nearly every column, the shift is
#: zero for most slots and one or two for the rest, and the corrupted read
#: lands on a NEIGHBORING feature's histogram, which on a correlated matrix
#: can score close enough to the right one that the metric barely moves. A
#: detector that only fires sometimes is not a detector. At 0.5 the shift
#: grows with the slot index and the last active slot reads past the end of
#: what the level wrote.
#:
#: 0.5 is also a value a user plausibly sets rather than a value chosen to
#: break something: it is a standard `colsample_bytree`, and `boosting="rf"`
#: names `feature_fraction < 1` as one of its two legal sources of per-tree
#: randomness (`sklearn.py`, the `boosting` docstring). And it makes the arm
#: CHEAPER than a full-feature one, because half the columns is half the
#: histogram work, which matters on a laptop six lanes share.
#:
#: NOT VERIFIED, and it is the honest limit on the paragraph above: nobody has
#: measured how large the divergence is at 0.5 against, say, 0.9, because the
#: bug is already fixed and reproducing it means reverting a source file this
#: lane does not own. The argument is from the index arithmetic in
#: `histogram_gpu.stage_desc_level_plan`'s comment, not from a measurement.
SYMMETRIC_COLSAMPLE_FRACTION_CHOICE = 0.5

#: What a symmetric-colsample row may and may not claim. Carried into every
#: record, so the claim travels with the number.
MOJOTREES_SYMMETRIC_COLSAMPLE_CLAIMS = (
    "arm 'mojotrees symmetric colsample': mojotrees with "
    f"{MOJOTREES_SYMMETRIC_COLSAMPLE} applied over the shared defaults. It is "
    "a CORRECTNESS arm and NOT a comparison arm, which is the first thing to "
    "know about reading it. There is no peer column for it and none is "
    "coming: no competitor in this suite grows symmetric trees under "
    "per-tree column sampling at these settings, and this arm was not built "
    "to price one. What it is for is verify.check_device_agreement, which "
    "compares this arm's gpu predictions against its own cpu twin's row by "
    "row. That pair is the whole product of the arm. On 2026-08-17 the "
    "oblivious GPU path was found returning a wrong answer whenever "
    "feature_fraction < 1, because GpuLeafBatcher.feat_dev was never set by "
    "the level build, so the histogram wrote feature slice `slot` while the "
    "split searcher read slice `active[slot]` and the root disagreed with "
    "its own children; it was found by reading code and not by this harness, "
    "because every scenario here runs feature_fraction=1.0, where the "
    "identity is correct and nothing shows. The fix is one "
    "set_shared_features call in histogram_gpu.stage_desc_level_plan and it "
    "is compiled into the extension this arm runs against, so a GREEN "
    "device_agreement row here is the intended result and means the fix is "
    "still in. TWO LIMITS ON THE ACCURACY NUMBER. First, feature_fraction "
    "0.5 is a regularizer, so this arm's metric is not comparable with any "
    "arm at 1.0 and a gap against the comparator is a property of the "
    "configuration rather than a defect; verify.check_accuracy_peer reports "
    "it because it reports every subject arm, and since 2026-08-17 it gates "
    "nothing at all. The accuracy gate is verify.check_accuracy_anchor, "
    "which compares this arm against OUR OWN recorded accuracy for it and so "
    "cannot charge a regularizer for being a regularizer. Second, the tree this arm grows is symmetric and depth-bounded "
    "at 6, which is a different tree from the plain arm's leaf-wise 31-leaf "
    "one, so the two are not a pairing either. The one comparison this arm "
    "supports is its own cpu row against its own gpu row."
)

#: The Cosine-under-leaf-wise-growth arm's override. The configuration bug 2
#: lived in.
#:
#: JUSTIFICATION PER KEY:
#:
#: - `score_function="cosine"`. The key this arm exists for. Case insensitive
#:   at the estimator (`sklearn.py`, the `score_function` docstring); spelled
#:   lower case here to match `MOJOTREES_CATBOOST_MODE`, so a reader diffing
#:   the two arms sees one spelling.
#: - `grow_policy="lossguide"`. **Set explicitly even though it is already
#:   the estimator's default** (`python/mojotrees/sklearn.py::_Base.__init__`),
#:   and that is the point of
#:   it. The bug was specific to the LEAF-WISE device path, and an arm that
#:   reached that path by inheriting a default would silently stop exercising
#:   it the day the default moves. The shipped defaults have already moved
#:   once in this direction: `mojotrees-defaults-decision` records symmetric
#:   360-tree CatBoost-mode as the shipped set with lossguide as the opt-in.
#:   An arm whose whole value is which code path it reaches must name the
#:   path.
#: - `lambda_l2=3.0`. **Load-bearing, and the arm is a much weaker detector
#:   without it.** At `reg_lambda=0` the Cosine numerator and denominator
#:   collapse onto the same expression, so the score degenerates to `sqrt` of
#:   the L2 score, and `sqrt` being strictly increasing it cannot move the
#:   argmax WITHIN one node (`sklearn.py`, the `score_function` docstring,
#:   which states exactly this and states the limit of it). Under leaf-wise
#:   growth it can still move the tree, because the frontier queue compares
#:   gains from different parents and `sqrt` does not preserve that ordering
#:   -- so an arm at 0.0 would detect the bug through split ORDER alone,
#:   which with `num_leaves` a hard bound at 31 may end at the same leaf set
#:   on many fits. 3.0 is off the degenerate point outright, so the two
#:   functionals disagree inside every node as well as across the frontier.
#:   The value is CatBoost's own `l2_leaf_reg` and the one
#:   `MOJOTREES_CATBOOST_MODE` already carries, rather than a number invented
#:   here. The cost of the key is stated in the claims string: this arm
#:   differs from the plain one in two parameters and not one, so it is not
#:   an isolation of the score function.
#:
#: WHAT IS DELIBERATELY ABSENT. `random_strength`, because it is refused on
#: the GPU by name whenever it is positive and declines to 0.0 in silence when
#: it arrives as a mode default (`run.DEVICE_PARAMETER_DIVERGENCE`), so a gpu
#: cell carrying it is either an infrastructure failure or a different
#: regularizer under the cpu cell's name -- and a device_agreement pair whose
#: two halves are not the same arm is worse than no pair. `grow_policy` stays
#: leaf-wise, so the symmetric-only keys (`max_depth`, `num_leaves`) stay at
#: the shared defaults and this arm's tree shape is the plain arm's.
MOJOTREES_COSINE_LEAFWISE = {
    "grow_policy": "lossguide",
    "score_function": "cosine",
    "lambda_l2": 3.0,
}

#: What a Cosine-leaf-wise row may and may not claim. Carried into every
#: record.
MOJOTREES_COSINE_LEAFWISE_CLAIMS = (
    "arm 'mojotrees cosine leafwise': mojotrees with "
    f"{MOJOTREES_COSINE_LEAFWISE} applied over the shared defaults. It is a "
    "CORRECTNESS arm and NOT a comparison arm. There is no peer column for "
    "it: Cosine is CatBoost's split score and CatBoost grows symmetric trees "
    "under it, so a leaf-wise Cosine fit is a configuration no competitor "
    "here ships and this arm is not a parity row for any of them. What it is "
    "for is verify.check_device_agreement, which compares this arm's gpu "
    "predictions against its own cpu twin's row by row. On 2026-08-17 "
    "score_function=Cosine was found to be silently IGNORED on every "
    "leaf-wise GPU fit: GpuSplitSearcher.set_score_function had no production "
    "caller, so score_function_code stayed at its constructed SCORE_L2, and "
    "gpu_resident_round._launch_child_search called _launch_search without "
    "naming the argument and took the same default. Both halves were needed "
    "and both are fixed (train_gpu.mojo::GpuSplitSearcherCache.reset_for_tree "
    "and gpu_resident_round). It was "
    "found by reading code and not by this harness, because the one arm here "
    "that sets Cosine, mojotrees_catboost_mode, grows SYMMETRIC trees and so "
    "never went down the leaf-wise path the bug lived on. A green "
    "device_agreement row here is the intended result and means the fix is "
    "still in. TWO LIMITS ON READING IT. First, this arm changes TWO "
    "parameters and not one: lambda_l2=3.0 is carried because at lambda_l2=0 "
    "the Cosine score degenerates to sqrt of the L2 score and cannot move "
    "the argmax within a node, which would make this a much weaker detector; "
    "so this row does not price the score function on its own and no arm "
    "here does. Second, verify.check_accuracy_peer reports it because it "
    "reports every subject arm, and since 2026-08-17 it gates nothing; a gap "
    "against the comparator on this row is a property of a configuration "
    "nobody ships rather than a defect. The gate is "
    "verify.check_accuracy_anchor, against our own recorded accuracy."
)

#: The arms whose product is a `device_agreement` verdict rather than a
#: number, in one place so a check can name the set rather than the members.
#:
#: This tuple is what `selfcheck.check_correctness_arms` walks. It exists
#: because the failure this whole block was written about is an arm silently
#: DISAPPEARING: both bugs above were live under a fully green suite, and the
#: reason was an absent configuration rather than a wrong threshold. An arm
#: deleted from `engines.ENGINES` restores that state exactly, and nothing but
#: a check that names it would notice.
CORRECTNESS_ARMS = (
    "mojotrees_symmetric_colsample",
    "mojotrees_cosine_leafwise",
)

#: The tier past which a correctness arm is a declared skip.
#:
#: **A COST BOUND, NOT A CAPABILITY BOUND, and the difference is the whole
#: entry.** Nothing about either arm fails at the large tier. What is true is
#: that the large tier is 1,000,000 rows by 100 features where the standard
#: tier is 200,000 by 50 (synthetic) or YearPredictionMSD's 515,345 by 90
#: (real), that these two arms produce a cpu-versus-gpu agreement verdict and
#: not a speed claim, and that a row-level prediction comparison is not more
#: true for having been taken on five times the rows. Six lanes share this
#: laptop under a timing lock.
#:
#: WHAT THE CAP GIVES UP, named rather than left to be discovered. The large
#: tier is where the device paths differ most from the standard tier: the
#: workload-aware AUTO strategy sends large shapes to the device split search,
#: which `thresholds.json` records as trading row-level parity for Float32
#: gain comparisons that flip near-ties. So a large-tier run of these arms
#: would test the agreement of a DIFFERENT device path from the one the
#: standard tier exercises, and it is a real measurement that this cap does
#: not take. Lift the cap for a run that is about that path, with a wall clock
#: attached; the cap is here so that nobody pays for it by accident.
#:
#: The shape is `CATBOOST_TIER_CAP`'s -- `(tier, reason)`, compared through
#: `TIER_ORDER` -- so `run.py` reads both through one comparison.
CORRECTNESS_ARM_TIER_CAP = ("standard", (
    "a correctness arm is capped at the standard tier. This arm's product is "
    "a verify.check_device_agreement verdict, which compares a gpu row's "
    "predictions against its own cpu twin's row by row, and that comparison "
    "is not more true for having been taken on 1,000,000 rows than on "
    "200,000. This is a COST bound and not a capability bound: nothing here "
    "fails at the large tier, and six lanes share this laptop under a timing "
    "lock. What the cap gives up is real and is named in "
    "scenarios.CORRECTNESS_ARM_TIER_CAP: the large tier routes to the device "
    "split search, whose row-level parity thresholds.json already records as "
    "traded away, so a large-tier run of this arm would exercise a different "
    "device path. Ask for it deliberately with a wall clock attached"
))


def correctness_arm_tier_ok(tier):
    """Whether a correctness arm may run at `tier`.

    Returns `(True, None)` or `(False, reason)`, which is
    `catboost_tier_ok`'s shape so that `run.py` reads both the same way.
    """
    cap, reason = CORRECTNESS_ARM_TIER_CAP
    if TIER_ORDER.index(tier) <= TIER_ORDER.index(cap):
        return True, None
    return False, reason


def mojotrees_symmetric_colsample_params(spec, device, extra=None):
    """The symmetric-plus-column-sampling CORRECTNESS arm.

    `mojotrees_params` with `MOJOTREES_SYMMETRIC_COLSAMPLE` applied over the
    shared defaults, by the same route and for the same reason
    `mojotrees_depthwise_params` gives: the override goes onto the RESOLVED
    dict, because `mojotrees_params` builds an explicit dict and copies
    through only the categorical, ranking and multiclass blocks, so a
    scenario-level `grow_policy` would be dropped on the floor.
    """
    params = mojotrees_params(spec, device, extra)
    params.update(MOJOTREES_SYMMETRIC_COLSAMPLE)
    return params


def mojotrees_cosine_leafwise_params(spec, device, extra=None):
    """The Cosine-under-leaf-wise-growth CORRECTNESS arm.

    Same shape and same reason as `mojotrees_symmetric_colsample_params`
    above.
    """
    params = mojotrees_params(spec, device, extra)
    params.update(MOJOTREES_COSINE_LEAFWISE)
    return params


# ---------------------------------------------------------------------------
# The read-back contract, and the parity check it makes possible.
# ---------------------------------------------------------------------------


class CatBoostReadbackMissing(RuntimeError):
    """The CatBoost-mode arm was built without CatBoost's resolved parameters.

    Raised by name rather than defaulted around. See
    `MOJOTREES_CATBOOST_MODE_FROM_READBACK` for why there is no fallback and
    `WIRE_NOTE_resolved_param_parity.md` in the worktree root for the four
    call-site edits that supply it.
    """


def catboost_readback_key(spec):
    """The cell a CatBoost read-back belongs to.

    Scenario, tier and variant, because CatBoost's derived values move with
    the DATA: the resolved learning rate is a function of the row count, the
    feature count and the iteration count, so a read-back from
    `dense_regression` at `standard` says nothing about the same scenario at
    `large` and nothing at all about a different scenario.

    The variant is the REQUESTED one (`resolve` writes `variant_requested`,
    not `variant`), which is what both cells of a pair are handed by the same
    job. It is not the variant that was loaded: `auto` resolves to real or
    synthetic at load time, and if two cells of one run resolved it
    differently they would share a key while holding different data. That has
    never been observed and nothing here would catch it.
    """
    scenario = spec["id"] if isinstance(spec, dict) else str(spec)
    tier = spec.get("tier") if isinstance(spec, dict) else None
    variant = (
        spec.get("variant_requested") or spec.get("variant")
        if isinstance(spec, dict)
        else None
    )
    return "|".join([scenario, str(tier or "standard"), str(variant or "auto")])


def catboost_readback_entry(spec, resolved, engine_version=None):
    """One cell's read-back, in the shape `load_catboost_readback` expects.

    `resolved` is `CatBoost.get_all_params()` off the fitted model, which is
    what `engines.CatBoostEngine.run` already records as
    `engine_resolved_params`. Nothing is filtered out of it here: the sidecar
    carries the library's answer verbatim and the parity map decides what is
    read from it.
    """
    return {
        "key": catboost_readback_key(spec),
        "arm": catboost_arm_id(),
        "engine_version": engine_version,
        "resolved": dict(resolved or {}),
    }


def catboost_readback_values(spec, readback):
    """The `MOJOTREES_CATBOOST_MODE` entries that come from the read-back.

    Raises `CatBoostReadbackMissing` when the read-back is absent, does not
    cover this cell, or is missing one of the keys, and the message names the
    cell and the keys rather than the type. An arm that trained on a guessed
    learning rate under a heading that says "CatBoost's shape" is the exact
    outcome this campaign exists to prevent, so the failure is loud.
    """
    wanted = dict(MOJOTREES_CATBOOST_MODE_FROM_READBACK)
    if not wanted:
        return {}
    key = catboost_readback_key(spec)
    if readback is None:
        raise CatBoostReadbackMissing(
            f"the mojotrees CatBoost-mode arm on cell {key} needs CatBoost's "
            f"resolved {', '.join(sorted(wanted))} and was given no read-back. "
            f"{CATBOOST_ARM_LABEL} does not pass a learning rate any more, so "
            "CatBoost resolves its own (roughly 0.4273 at 100 iterations "
            "against the 0.1 the other two arms run) and this arm has to take "
            "that value rather than assume one. There is deliberately no "
            "fallback: see MOJOTREES_CATBOOST_MODE_FROM_READBACK and "
            "CATBOOST_DELIBERATE_DIVERGENCE. Supplying it needs the CatBoost "
            "cell for this scenario, tier and variant to run first and to "
            "write its engine_resolved_params where this cell can read them"
        )
    entry = load_catboost_readback(readback, spec)
    resolved = entry.get("resolved") or {}
    values, missing = {}, []
    for ours, theirs in sorted(wanted.items()):
        if theirs not in resolved:
            missing.append(f"{theirs} (for our {ours})")
            continue
        values[ours] = resolved[theirs]
    if missing:
        raise CatBoostReadbackMissing(
            f"the CatBoost read-back for cell {key} does not carry "
            f"{', '.join(missing)}. get_all_params() returned "
            f"{len(resolved)} keys and none of them is what this arm needs, "
            "so either the read-back is from a different CatBoost or "
            "engine_resolved_params_source says why it failed. Read that "
            "field before changing anything here"
        )
    return values


def load_catboost_readback(source, spec):
    """One cell's read-back out of whatever the caller has.

    Accepts the entry itself, a mapping of cell key to entry, or a path to a
    JSON file holding either. A caller that has a whole run's worth of
    read-backs passes the mapping and gets the right cell rather than the
    first one, which is the failure this function exists to prevent: two
    scenarios resolve two different learning rates and nothing about the value
    itself says which cell it came from.
    """
    key = catboost_readback_key(spec)
    if isinstance(source, str):
        import json

        with open(source) as handle:
            source = json.load(handle)
    if not isinstance(source, dict):
        raise CatBoostReadbackMissing(
            f"a CatBoost read-back for cell {key} must be a mapping or a path "
            f"to one, and this is a {type(source).__name__}"
        )
    if "resolved" in source and "key" in source:
        if source["key"] != key:
            raise CatBoostReadbackMissing(
                f"this CatBoost read-back is for cell {source['key']!r} and "
                f"the arm being built is cell {key!r}. CatBoost resolves a "
                "different learning rate per shape, so a read-back from "
                "another cell is a wrong number rather than a near one"
            )
        return source
    if key in source:
        return source[key]
    raise CatBoostReadbackMissing(
        f"no CatBoost read-back for cell {key}. The mapping holds "
        f"{sorted(source)[:8]}"
    )


#: Value translations, by the name `CATBOOST_PARAM_MAP` records.
_PARITY_TRANSLATORS = {
    "identity": lambda value: value,
    # CatBoost counts THRESHOLDS and LightGBM counts BINS, so 254 there is 255
    # here and the two describe the same granularity budget.
    "borders_to_bins": lambda value: int(value) + 1,
    "boosting_alias": lambda value: {
        "plain": "gbdt", "ordered": "ordered",
    }.get(str(value).strip().lower(), str(value).strip().lower()),
    "loss_to_objective": lambda value: {
        loss: task for task, loss in CATBOOST_LOSS.items()
    }.get(str(value), str(value)),
}


def _parity_translate(name, value):
    if name not in _PARITY_TRANSLATORS:
        raise KeyError(
            f"CATBOOST_PARAM_MAP names translation {name!r}, which "
            f"_PARITY_TRANSLATORS does not define. Known: "
            + ", ".join(sorted(_PARITY_TRANSLATORS))
        )
    return _PARITY_TRANSLATORS[name](value)


def _float32(value):
    """`value` through a 32-bit float and back, or `None` if it will not go.

    CatBoost stores `learning_rate` as a float32, so a rate that leaves
    `get_all_params()` is a widened float32 and comparing it to a double for
    equality fails on a value both sides agree about. Rounding both sides
    through float32 is the comparison that means what it says.
    """
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, ValueError, TypeError):
        return None


def _parity_equal(theirs, ours):
    """Whether two resolved values are the same value.

    Three folds and no more, each of them a documented property of the two
    surfaces rather than a convenience:

    * bools compare as bools, never as 0 and 1, so `False` and `0` disagree.
    * strings compare case-folded, because `docs/PARAMETER_NAMING.md` states
      that value strings are case-insensitive on our side and CatBoost spells
      its own values in CamelCase.
    * numbers compare through a float32 round trip, because CatBoost stores
      the one number this arm reads back as a 32-bit float. `3` and `3.0`
      agree; `3` and `3.0000001` do not, since both survive the round trip
      distinctly.
    """
    if isinstance(theirs, bool) or isinstance(ours, bool):
        return isinstance(theirs, bool) and isinstance(ours, bool) and theirs == ours
    if theirs is None or ours is None:
        return theirs is None and ours is None
    if isinstance(theirs, str) and isinstance(ours, str):
        return theirs.strip().casefold() == ours.strip().casefold()
    if isinstance(theirs, (int, float)) and isinstance(ours, (int, float)):
        left, right = _float32(theirs), _float32(ours)
        return left is not None and right is not None and left == right
    return theirs == ours


#: The scenarios the parity gate is evaluated on.
#:
#: Deliberately NOT read off `MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT`. That
#: table says which cells are SCHEDULED, and the arm is parked while the
#: read-back wiring is missing, so reading the gate off it would mean the gate
#: switched itself off at the same moment the arm's parameters changed most.
#: A check that goes quiet when the thing it checks is in flux is not a check.
MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS = (
    "dense_regression",
    "imbalanced_binary",
    "ordered_boosting_small",
)


def catboost_resolved_declared(spec, threads=None, extra=None, live=None):
    """What CatBoost resolves for `spec`, live where that is known.

    Three layers, and the order is the whole point:

    1. `CATBOOST_LEFT_AT_STOCK`, this file's TRANSCRIPTION of
       `get_all_params()`, taken on one shape and one loss on 2026-08-16.
    2. everything `catboost_params` passes.
    3. `live`, a real `get_all_params()` for THIS cell, which wins.

    **Layer 3 was missing until it was caught by a run, and its absence made
    the parity table say `agree` about a key the two engines disagreed on.**
    CatBoost resolves several parameters PER OBJECTIVE, not once:
    `leaf_estimation_iterations` is 1 under RMSE and 10 under Logloss, and
    `boost_from_average` is True under RMSE and False under Logloss. Measured
    2026-08-16 on this machine, `imbalanced_binary` at the smoke tier. The
    transcription holds only the RMSE reading, so a table built from layers 1
    and 2 alone compared our arm against CatBoost-on-a-different-loss and
    reported a match. Live values win here so that the comparison is against
    the fit that actually happened.

    `CATBOOST_RESOLVED_PER_FIT` keys are ABSENT from the static layers, not
    defaulted. There is no honest static value for `learning_rate`.
    """
    declared = dict(CATBOOST_LEFT_AT_STOCK)
    declared.update(catboost_params(spec, 1 if threads is None else threads, extra))
    if threads is None:
        declared.pop("thread_count", None)
    if live:
        for key, value in live.items():
            if key in declared or key in CATBOOST_RESOLVED_PER_FIT:
                declared[key] = value
    return declared


def catboost_live_resolved(spec, catboost_readback):
    """This cell's real `get_all_params()` out of a read-back, or `{}`.

    Never raises. A parity table is a reporting field and must not be the
    thing that takes a run down, so a read-back that does not cover this cell
    yields an empty dict and the table falls back to the transcription with
    `source` saying so.
    """
    if not catboost_readback:
        return {}
    try:
        return dict((load_catboost_readback(catboost_readback, spec) or {})
                    .get("resolved") or {})
    except CatBoostReadbackMissing:
        return {}


def mojotrees_catboost_mode_resolved(
    spec,
    device="cpu",
    extra=None,
    catboost_readback=None,
    arm_params=None,
    arm_dataset_params=None,
):
    """The CatBoost-mode arm's resolved configuration, across both containers.

    The training dict from `mojotrees_catboost_mode_params`, plus the two
    Dataset parameters from `dataset_params`, plus `n_estimators`, which the
    adapter adds after translation rather than the translator emitting it
    (`engines.MojoTreesEngine._run_dense`). All three are part of what the arm
    RAN, and a parity check that saw only the first would report `max_bin` and
    the tree budget as unmatched when they are matched.

    `arm_params` and `arm_dataset_params` are the frontier arm's overrides,
    the same two dicts `engines.MojoTreesEngine` carries under those names.
    They are applied here in the same ORDER the adapter applies them --
    `_n_estimators` reads `arm_params`, `_dataset_params` layers
    `arm_dataset_params` over `dataset_params(spec)` -- because the resolved
    dict has to be what the fit ran and not what the base would have run.

    **This mirrors the adapter and is therefore a coupling, and the coupling
    has already broken once.** This docstring used to say that if `_run_dense`
    stopped adding `n_estimators` this function would go stale and the parity
    table would quietly check the wrong dict. That is exactly what happened
    when the arm dimension landed: `_run_dense` moved to `self._n_estimators()`
    and applied `arm_dataset_params`, and this function went on reading
    `BASE_PARAMS["n_estimators"]` and the base binning. Every frontier arm on
    the trees axis or the max_bin axis would have had its parity table checked
    against a configuration no fit used.

    The mitigation is no longer "both places name each other", because that is
    what was in place and it did not hold. `catboost_parity_rows` now asserts
    the tree count in this dict against the count the adapter would resolve
    from the same arm; see `_check_resolved_matches_adapter`.
    """
    resolved = dict(
        mojotrees_catboost_mode_params(
            spec, device, extra, catboost_readback=catboost_readback
        )
    )
    arm_params = dict(arm_params or {})
    resolved["n_estimators"] = int(
        arm_params.get("n_estimators", BASE_PARAMS["n_estimators"])
    )
    binning = dict(dataset_params(spec))
    binning.update(dict(arm_dataset_params or {}))
    resolved.update(binning)
    return resolved


def _check_resolved_matches_adapter(resolved, arm_params):
    """Raise when the parity dict's tree count is not the adapter's.

    The check that would have caught the break described in
    `mojotrees_catboost_mode_resolved`. It recomputes the count the way
    `engines.MojoTreesEngine._n_estimators` does -- the arm's value, else
    `BASE_PARAMS` -- and compares. It deliberately does NOT import `engines`:
    this module is imported by `engines`, and the point is to state the rule
    in one line that a reader can check against the other file, not to build
    a cycle for the sake of an assertion.

    It is a hard failure rather than a row, because a parity table that
    describes a configuration nothing ran is not a table with one bad row in
    it -- every row in it is about the wrong fit.
    """
    expected = int(
        dict(arm_params or {}).get(
            "n_estimators", BASE_PARAMS["n_estimators"]
        )
    )
    got = int(resolved.get("n_estimators", -1))
    if got != expected:
        raise ValueError(
            "the CatBoost-mode parity table resolved n_estimators=%d and the "
            "adapter would run %d; a parity table for a configuration no fit "
            "used is worse than no table (engines.MojoTreesEngine."
            "_n_estimators is the other half of this rule)" % (got, expected)
        )


def catboost_parity_rows(
    spec,
    device="cpu",
    extra=None,
    catboost_readback=None,
    arm_params=None,
    arm_dataset_params=None,
):
    """The key-by-key diff between the two resolved dicts, as a list of rows.

    One row per key CatBoost resolves, in `CATBOOST_PARAM_MAP` order, each
    carrying CatBoost's name, our name, both values and a status. This is what
    `selfcheck.check_catboost_arm` fails on and what `catboost_arm_block`
    records, so that a reader of a published number sees the same table the
    gate saw.

    Statuses:

    * `agree`           a matched key whose two values are the same value.
    * `DISAGREE`        a matched key whose values differ. A failure.
    * `deferred`        a matched key CatBoost derives per fit, so the values
                        cannot be compared without a run. `learning_rate`.
    * `unmatchable`     no parameter closes it; names a CATBOOST_UNMATCHABLE
                        entry.
    * `not_reached`     we have the knob and no cell this arm runs reaches it.
    * `UNCLASSIFIED`    a key in neither table. A failure, and the one this
                        whole structure exists to make impossible.
    """
    live = catboost_live_resolved(spec, catboost_readback)
    declared = catboost_resolved_declared(spec, None, extra, live)
    try:
        ours = mojotrees_catboost_mode_resolved(
            spec,
            device,
            extra,
            catboost_readback=catboost_readback,
            arm_params=arm_params,
            arm_dataset_params=arm_dataset_params,
        )
        ours_error = None
    except CatBoostReadbackMissing as exc:
        # The arm cannot be built without a read-back, and the parity table is
        # still worth producing: every key except `learning_rate` has a static
        # value on both sides and is checkable here. The one that is not says
        # so in its own row rather than taking the table down with it.
        ours = mojotrees_catboost_mode_resolved(
            spec,
            device,
            extra,
            catboost_readback=_READBACK_STANDIN(spec),
            arm_params=arm_params,
            arm_dataset_params=arm_dataset_params,
        )
        ours_error = str(exc)
    # Before any row is built, because a table about the wrong configuration
    # has no good rows in it. Runs on the stand-in path too: the stand-in
    # replaces the learning rate, not the tree count.
    _check_resolved_matches_adapter(ours, arm_params)

    rows = []
    missing = object()
    for theirs_key in list(CATBOOST_PARAM_MAP) + sorted(
        set(declared) - set(CATBOOST_PARAM_MAP) - set(CATBOOST_PARAM_NOT_MAPPED)
    ):
        entry = CATBOOST_PARAM_MAP.get(theirs_key)
        if entry is None:
            rows.append({
                "catboost": theirs_key,
                "catboost_value": declared.get(theirs_key),
                "mojotrees": None,
                "mojotrees_value": None,
                "status": "UNCLASSIFIED",
                "detail": (
                    f"CatBoost resolves {theirs_key} and nothing in "
                    "CATBOOST_PARAM_MAP or CATBOOST_PARAM_NOT_MAPPED says "
                    "whether this arm matches it. Classify it: a key nobody "
                    "decided about is the hand-written belief this table "
                    "removes"
                ),
            })
            continue
        if theirs_key not in declared and entry.get("static", True):
            # A mapped key the CatBoost side does not resolve on this cell:
            # `classes_count` on a regression, for instance. Not an error and
            # not a comparison either.
            continue
        ours_key = entry.get("ours")
        ours_value = ours.get(ours_key, missing) if ours_key else missing
        if ours_value is missing and "our_default" in entry:
            ours_value = entry["our_default"]
        row = {
            "catboost": theirs_key,
            "catboost_value": declared.get(theirs_key),
            "mojotrees": ours_key,
            "mojotrees_value": None if ours_value is missing else ours_value,
            "mojotrees_container": entry.get("ours_container", "train"),
            # Where the CatBoost value came from. A `declared` row is being
            # compared against a transcription taken on one shape and one
            # loss; a `live` row is being compared against the fit that
            # actually happened. The distinction is not cosmetic: see
            # catboost_resolved_declared.
            "catboost_value_source": (
                "live" if theirs_key in live else "declared"
            ),
            "status": entry["verdict"],
            "detail": entry.get("reason") or entry.get("note") or "",
        }
        if entry["verdict"] == "unmatchable":
            row["unmatchable_key"] = entry.get("unmatchable_key")
            row["detail"] = row["detail"] or CATBOOST_UNMATCHABLE.get(
                entry.get("unmatchable_key"), ""
            )
        elif entry["verdict"] == "matched":
            if not entry.get("static", True):
                row["status"] = "deferred"
                row["catboost_value"] = None
                row["detail"] = (
                    "CatBoost derives this per fit, so the two values are "
                    "compared against two finished records by "
                    "catboost_parity_from_records and not here. "
                    + row["detail"]
                )
                if ours_error:
                    row["mojotrees_value"] = None
                    row["readback_missing"] = ours_error
            elif ours_value is missing:
                row["status"] = "DISAGREE"
                row["detail"] = (
                    f"CatBoost resolves {theirs_key}="
                    f"{declared.get(theirs_key)!r} and this arm has no "
                    f"{ours_key!r} in either its training params or its "
                    "dataset params, and CATBOOST_PARAM_MAP declares no "
                    "our_default for it. Set it in MOJOTREES_CATBOOST_MODE, "
                    "declare the default it already has, or move the key out "
                    "of the matched set with a reason"
                )
            else:
                expected = _parity_translate(
                    entry.get("translate", "identity"), declared[theirs_key]
                )
                if _parity_equal(expected, ours_value):
                    row["status"] = "agree"
                else:
                    row["status"] = "DISAGREE"
                    row["detail"] = (
                        f"CatBoost resolves {theirs_key}="
                        f"{declared[theirs_key]!r} and this arm resolves "
                        f"{ours_key}={ours_value!r}"
                        + (
                            f" (CatBoost's value translates to "
                            f"{expected!r} through "
                            f"{entry.get('translate', 'identity')})"
                            if entry.get("translate", "identity") != "identity"
                            else ""
                        )
                        + ". CATBOOST_PARAM_MAP calls this key matched, so "
                        "the two arms are running different models under one "
                        "heading. Fix the arm's value, or move the key into "
                        "CATBOOST_UNMATCHABLE with a reason traced in our "
                        "code"
                    )
        rows.append(row)
    return rows


def catboost_parity_failures(rows):
    """The rows a run must not ship, as messages. Empty means the arm agrees."""
    return [
        f"{row['catboost']} -> {row['mojotrees']}: {row['detail']}"
        for row in rows
        if row["status"] in ("DISAGREE", "UNCLASSIFIED")
    ]


def catboost_parity_from_records(catboost_record, mode_record):
    """The run-time half: the deferred keys, against two finished records.

    `catboost_record` needs `engine_resolved_params`, which
    `engines.CatBoostEngine.run` already writes; `mode_record` needs
    `params.engine`. Returns a list of failure messages, empty when they
    agree. Separate from `catboost_parity_rows` because these are the keys no
    static check can see: CatBoost derives them from the data, so the only
    place they can be compared is after both cells have run.
    """
    theirs = (catboost_record or {}).get("engine_resolved_params") or {}
    ours = ((mode_record or {}).get("params") or {}).get("engine") or {}
    failures = []
    if not theirs:
        return [
            "the CatBoost record carries no engine_resolved_params, so "
            "nothing checks the values this arm took from it. "
            + str((catboost_record or {}).get("engine_resolved_params_source"))
        ]
    for ours_key, theirs_key in sorted(
        MOJOTREES_CATBOOST_MODE_FROM_READBACK.items()
    ):
        if theirs_key not in theirs:
            failures.append(
                f"CatBoost's read-back has no {theirs_key}, which this arm's "
                f"{ours_key} was supposed to come from"
            )
            continue
        if ours_key not in ours:
            failures.append(
                f"the CatBoost-mode record has no {ours_key}, so the arm was "
                f"built without CatBoost's resolved {theirs_key}="
                f"{theirs[theirs_key]!r}"
            )
            continue
        if not _parity_equal(theirs[theirs_key], ours[ours_key]):
            failures.append(
                f"{theirs_key}: CatBoost ran {theirs[theirs_key]!r} and the "
                f"CatBoost-mode arm ran {ours_key}={ours[ours_key]!r}. This "
                "is the key the whole cb-shipped change is about; a "
                "difference here means the two rows are not the comparison "
                "the heading claims"
            )
    return failures


def check_catboost_readback(resolved, spec=None, threads=None, extra=None):
    """Where a live `get_all_params()` disagrees with this file, as messages.

    The other direction of the same question. `catboost_parity_rows` asks
    whether our arm matches what we DECLARE CatBoost resolves;  this asks
    whether CatBoost still resolves what we declare. Both are needed: the
    first catches our arm drifting, the second catches a CatBoost upgrade
    moving a default under a table transcribed on 2026-08-16.

    Keys CatBoost returns that this file says nothing about are reported, not
    failed: `get_all_params()` carries more than this harness classifies and
    always has.
    """
    declared = (
        catboost_resolved_declared(spec, threads, extra) if spec is not None else
        dict(CATBOOST_LEFT_AT_STOCK)
    )
    drift = []
    for key, value in sorted(declared.items()):
        if key not in (resolved or {}):
            drift.append(
                f"this harness declares CatBoost resolves {key}={value!r} and "
                "get_all_params() does not carry the key at all"
            )
            continue
        if not _parity_equal(value, resolved[key]):
            drift.append(
                f"{key}: declared {value!r}, CatBoost resolved "
                f"{resolved[key]!r}. {CATBOOST_DEFAULTS_SOURCE} is the "
                "transcription this contradicts; re-read it rather than "
                "editing the number to match"
            )
    for key in sorted(CATBOOST_RESOLVED_PER_FIT):
        if key not in (resolved or {}):
            drift.append(
                f"CATBOOST_RESOLVED_PER_FIT names {key} as the value only a "
                "read-back can supply, and this read-back does not carry it"
            )
    return drift


#: The file a run collects CatBoost read-backs into, under the run directory.
CATBOOST_READBACK_FILE = "catboost_readback.json"


def append_catboost_readback(path, entry):
    """Add one cell's read-back to the run's sidecar, keyed by cell.

    Read-modify-write rather than append-only, because the harness runs one
    job per process and there is no shared handle. Sequential by construction
    (`run.py` runs cells one at a time and says so in its manifest), so there
    is no lock here and there must not be a parallel runner without one.
    """
    import json
    import os

    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
        except (OSError, ValueError):
            existing = {}
    existing[entry["key"]] = entry
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2, default=str)
        handle.write("\n")
    return path


def record_parity_block(
    spec, engine, params_used, dataset_params_used=None, catboost_readback=None
):
    """What every result row carries so a reader can see both engines' dicts.

    The point is stated in the negative, because that is what went wrong. A
    record used to carry what its OWN engine was passed and nothing about what
    the engine it is being compared against resolved, so a published ratio was
    readable only by someone who also held the other row and knew which
    parameters each library derives for itself. This block puts both dicts in
    front of the reader of either row.

    `catboost_resolved_declared` is this file's transcription and not a live
    read-back; the live one is `engine_resolved_params` on the CatBoost row.
    Both are here on purpose, because a difference between them is a CatBoost
    upgrade moving a default and is exactly what `check_catboost_readback`
    exists to surface.
    """
    block = {
        "engine": engine,
        "engine_resolved": dict(params_used or {}),
        "engine_dataset_resolved": dict(dataset_params_used or {}),
        "arm": catboost_arm_id(),
        "catboost_resolved_declared": None,
        "rows": None,
        "unavailable_reason": None,
    }
    runs, reason = catboost_supports(spec)
    if not runs:
        block["unavailable_reason"] = (
            f"the CatBoost arm does not run {spec.get('id')}, so there is no "
            f"CatBoost dict to put beside this row: {reason}"
        )
        return block
    try:
        block["catboost_resolved_declared"] = catboost_resolved_declared(spec)
        block["rows"] = catboost_parity_rows(
            spec, catboost_readback=catboost_readback
        )
        block["failures"] = catboost_parity_failures(block["rows"])
    except Exception as exc:  # noqa: BLE001 - a record field, not a fit
        block["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return block


def _READBACK_STANDIN(spec):
    """A read-back that satisfies the shape and carries no value.

    Used ONLY by `catboost_parity_rows` so that the static half of the table
    can be produced while the arm itself is unbuildable. The values are
    `None`, so any row that reads one is marked `deferred` and never `agree`:
    this cannot make a check pass. It exists because a parity table that
    disappears whenever the read-back is missing would have gone quiet at
    exactly the moment the parameters changed.
    """
    return {
        "key": catboost_readback_key(spec),
        "arm": catboost_arm_id(),
        "engine_version": None,
        "resolved": {
            theirs: None
            for theirs in MOJOTREES_CATBOOST_MODE_FROM_READBACK.values()
        },
    }


#: Whether each scenario runs the `mojotrees_catboost_mode` arm, and the exact
#: reason when it does not. Separate from `CATBOOST_SCENARIO_SUPPORT` because
#: the two arms are blocked by different things: that table is about what
#: CatBoost can be handed, and this one is about which of our own trainers
#: honor `MOJOTREES_CATBOOST_MODE`'s row sampling.
#:
#: **This table exists because the arm stopped being a pure parameter
#: override.** Until `bootstrap_type=MVS` was added it was seven keys of tree
#: shape and regularization, every one of which every trainer reads, so the
#: arm ran wherever CatBoost ran and no second table was needed. MVS is the
#: first key our own trainers disagree about, and the disagreement is a
#: refusal rather than a wrong number: `trainset.train_dataset`'s sparse arm
#: calls `sampling.check_bootstrap_honored` by name, and
#: `train_dataset_multiclass` takes no bootstrap bundle at all, so those two
#: cells raise instead of quietly training on every row under a label that
#: says otherwise.
#:
#: Scheduling them and letting them raise was the alternative and it is worse
#: in a specific way: a raising cell is an infrastructure failure, and this
#: harness withholds the quality verdict for the whole matrix on one of those,
#: so two cells that were never going to run would have suppressed the verdict
#: on the twenty-two that did.
#:
#: The multiclass skip is also the more faithful of the two. CatBoost does not
#: run MVS for multiclass either: its own defaulting block excludes
#: multiclass-only losses and falls back to the Bayesian bootstrap. So the
#: skip is what CatBoost does, not merely what we cannot do.
#: UNPARKED 2026-08-16, same day. The three cells below were parked for the
#: few hours between `cb-shipped` landing in this file and the read-back
#: reaching them, and the history is kept because the shape of the dependency
#: is now a permanent property of this arm rather than a transitional state.
#:
#: **The CatBoost cell for a scenario must run before the CatBoost-mode cell
#: for the same scenario.** The CatBoost arm no longer passes a learning rate,
#: CatBoost resolves its own from the budget and the dataset, and this arm
#: takes that resolved value. There is no constant that could stand in: the
#: rate moves with the row count and the feature count, so the 0.4273 measured
#: on 20,000 by 20 is not the rate for any scenario in this suite.
#:
#: `run.py` enforces the order with a sort key composed under the repeat sort,
#: and `worker.py` carries the value between the two processes through the
#: run's `catboost_readback.json`. A mode cell scheduled without its CatBoost
#: partner raises `CatBoostReadbackMissing` by name and does not fall back.
MOJOTREES_CATBOOST_MODE_PARKED = (
    "kept as a named constant because the reason it existed can recur: this "
    "arm cannot be built without CatBoost's read-back for the same cell, so "
    "any change that breaks the ordering in run.py or the sidecar in "
    "worker.py takes these three cells down again. If that happens, park them "
    "here rather than letting them raise -- a raising cell is an "
    "infrastructure failure and this harness withholds the quality verdict "
    "for the whole matrix on one of those -- and never give the arm a "
    "fallback, because a static learning rate is the defect it was rebuilt to "
    "remove"
)

#: The data sources that hand the engines a `categorical_feature`
#: declaration, by the two names a scenario can reach one under.
#:
#: **Derived from the sources rather than from the scenario table, because the
#: scenario table does not know.** A scenario declares LightGBM's categorical
#: knobs (`CATEGORICAL_PARAMS`) only if its AUTHOR expected categories, and
#: `imbalanced_binary` declares none while `loaders.load_bank_marketing`
#: returns ten categorical indices. A predicate built on the scenario's params
#: would call that cell numeric and be wrong on exactly the cell where being
#: wrong costs a scheduled fit.
#:
#: Read out of the two files on 2026-08-17: `generators.categorical_missing`
#: and `generators.high_cardinality_categorical` return a
#: `categorical_feature` key, and so do `loaders.load_adult` and
#: `loaders.load_bank_marketing`. No other generator or loader does.
#: `selfcheck.check_categorical_sources` re-derives this by introspection, so
#: a source that starts declaring categories fails the check rather than
#: silently changing what gets scheduled.
CATEGORICAL_DATA_SOURCES = {
    "generators": ("categorical_missing", "high_cardinality_categorical"),
    "datasets": ("adult", "bank_marketing"),
}


def scenario_has_categorical(spec):
    """Whether a cell for `spec` can be handed a categorical column.

    **A REFUSAL GUARD, so it answers the DECLARATION question and fails
    closed.** Over-reporting a categorical column costs a cell an accelerator
    it could have used, which is a skipped row with a reason on it.
    Under-reporting one costs a SCHEDULED fit that then raises inside the
    device policy, and this harness answers an infrastructure failure by
    withholding the quality verdict for the whole matrix -- so a mispredicting
    guard here can take down a window whose comparator rows all succeeded.
    The asymmetry is the whole design and it is why `auto` below is treated as
    both variants rather than as the one it usually resolves to.

    `variant="auto"` prefers the real dataset (`resolve`), but the choice is
    made at LOAD time and this is asked at SCHEDULING time, so either source
    may be the one that arrives. `imbalanced_binary` is the scenario where
    that matters: its generator is purely numeric and its real dataset,
    bank_marketing, has ten categorical columns.
    """
    variant = spec.get("variant_requested") or "auto"
    generator = spec.get("generator")
    dataset = spec.get("dataset")
    sources = []
    if variant in ("synthetic", "auto") and generator is not None:
        sources.append(("generators", generator))
    if variant in ("real", "auto") and dataset is not None:
        sources.append(("datasets", dataset))
    if not sources:
        # The requested variant names a source this scenario does not have --
        # `--variant real` on a synthetic-only scenario, say. That is not
        # evidence of a numeric matrix, it is no evidence at all, and the
        # asymmetry above says which way no evidence goes. The load fails
        # first in every case reachable today; this is here so that the
        # failure stays a load failure rather than becoming a scheduled fit.
        return True
    return any(
        name in CATEGORICAL_DATA_SOURCES[kind] for kind, name in sources
    )


MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT = {
    "dense_regression": None,
    "imbalanced_binary": None,
    "ordered_boosting_small": None,
    "high_cardinality_categorical": (
        "grow_policy=oblivious is implemented for numerical thresholds "
        "only, and this arm asks for symmetric trees. A level of an "
        "oblivious tree shares one split across every node on it, while a "
        "categorical feature is searched as category partitions whose order "
        "comes from one node's own statistics, so there is no single "
        "partition a level could share. The grower refuses rather than "
        "picking one node's order for the whole level. Found by the smoke "
        "pass on 2026-08-16, as the third of three failures in this arm. "
        "CatBoost does not hit this because a CTR turns the categorical "
        "into a numeric column before its split search ever sees it, which "
        "is the gap CATBOOST_UNMATCHABLE['ctr'] names: this is that gap "
        "surfacing as a scheduling decision rather than as a quality one. "
        "STILL SKIPPED after the 2026-08-16 attempt to close it by building "
        "CatBoost's CTR REPLACEMENT (catalog A36), which stopped on four "
        "traced blockers and not on effort: (1) a CTR bundle is a property "
        "of Dataset CONSTRUCTION, and this harness hands every mojotrees arm "
        "the same dataset_params(spec), which takes no engine, so the "
        "CatBoost-mode arm cannot ask for CTRs while the plain arm does not; "
        "(2) python/mojotrees/basic.py's Dataset has no ctr argument and "
        "bindings/_mojotrees.mojo parses no ctr key, so the bundle is "
        "unreachable from Python at all; (3) replacement means the raw "
        "column stops being a split feature, and BinnedMatrix.usable -- the "
        "field built for exactly that -- is read by no grower "
        "(tree.grow_tree calls sampling.select_tree_features(data.n_features, "
        "...) and never passes the pool), so there is no mechanism that "
        "removes a column from the search; (4) a second refusal sits behind "
        "the first, since this arm also carries random_strength and "
        "split.find_best_split refuses that beside any categorical feature. "
        "Re-enabling this cell needs all four closed, not the first one"
    ),
    # Both of these are already skipped for the CatBoost arm itself by
    # CATBOOST_SCENARIO_SUPPORT, so the loop below never consults these two
    # entries today. They are here because the loop subscripts this table
    # rather than using `.get`, so every scenario needs a decision and a
    # scenario added without one raises a KeyError at import instead of
    # silently defaulting to "runs". That is the same fail-closed rule
    # CATBOOST_SCENARIO_SUPPORT already states, and it caught a typo in this
    # very table on the first run.
    "ranking": (
        "already skipped for the CatBoost arm. Were it not, the ranker "
        "trainers refuse the bootstrap bundle by name too "
        "(trainset.train_dataset_ranker)"
    ),
    "categorical_missing": (
        "already skipped for the CatBoost arm, so the CatBoost-mode row has "
        "nothing to be read against"
    ),
    "multiclass": (
        "MOJOTREES_CATBOOST_MODE sets bootstrap_type=MVS and the cell raises. "
        "CORRECTED 2026-08-17, AND THE REASON IS NOT THE ONE THIS ENTRY GAVE. "
        "It read 'the multiclass trainer takes no bootstrap bundle: "
        "trainset.train_dataset_multiclass has no bootstrap argument, so the "
        "binding refuses it by name'. That is false at head and "
        "CATBOOST_UNMATCHABLE['row_sampling'] asked for this correction by "
        "name. `trainset.train_dataset_multiclass` DOES take a bundle and both "
        "of its CPU arms honor it: `boosting._boost_rounds_multiclass` and "
        "`boosting_sparse.train_multiclass_sparse` each call "
        "`sampling.bootstrap_round` once per round, shared by every class's "
        "tree. WHAT ACTUALLY REFUSES is `sampling.check_mvs_reg_is_set`: MVS "
        "with a DERIVED lambda reads `lastIterValues[dim][leaf]`, one tree with "
        "one value per output dimension per leaf, and a softmax round of K "
        "structurally different trees has no such table, so a derived reg is "
        "refused and this arm names no `mvs_reg`. On the GPU a second and "
        "independent refusal stands: `train_multiclass_gpu` takes no bundle and "
        "`model.fit_multiclass` and `trainset.train_dataset_multiclass` raise "
        "on their GPU arms rather than training unsampled. Verified in source "
        "2026-08-17. THE CONCLUSION IS UNCHANGED and dropping the keys for "
        "multiclass would still be the less faithful choice: CatBoost's own "
        "defaulting block excludes the multiclass-only losses from MVS and "
        "falls back to Bayesian, so this cell would not have been the "
        "comparison it claims to be even under a trainer that accepted it. The "
        "plain mojotrees arm and the CatBoost arm both still run this scenario; "
        "what is missing is only the CatBoost-mode row. EXIT: an explicit "
        "`mvs_reg` on this arm would open the CPU cell, and it would then be a "
        "value CatBoost does not use here, so it is a different arm rather than "
        "this one unblocked"
    ),
    "sparse_highdim": (
        "CORRECTED 2026-08-17, AND THE REFUSAL MOVED TO A DIFFERENT PARAMETER. "
        "This entry read 'MOJOTREES_CATBOOST_MODE sets bootstrap_type=MVS, and "
        "the sparse trainer refuses it by name at trainset.train_dataset's "
        "sparse arm (sampling.check_bootstrap_honored) ... No sparse round loop "
        "calls bootstrap_round', and CATBOOST_UNMATCHABLE['row_sampling'] asked "
        "for this correction by name. Both halves are false at head: "
        "`boosting_sparse.train_sparse` and "
        "`boosting_sparse.train_multiclass_sparse` both call `bootstrap_round` "
        "in the place the dense loops call it, and `trainset.train_dataset` "
        "forwards the bundle to the sparse arm. So this cell no longer raises "
        "on `bootstrap_type` at all. WHAT STOPS IT NOW is `random_strength`: "
        "the sparse round loop computes no per-tree score scale and the binding "
        "declares the parameter honored only as `not d[].is_sparse`, so a named "
        "`random_strength` is refused there. This arm names it. Verified in "
        "source 2026-08-17. The plain mojotrees arm and the CatBoost arm both "
        "still run this scenario. EXIT: a sparse round loop that computes "
        "`boosting._round_random_score_scale`"
    ),
}

#: Which scenarios the depth-wise arm runs, `None` meaning it runs. Written
#: as an explicit table with one entry per scenario, exactly like
#: `MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT`, so a scenario added later
#: without a support decision fails `selfcheck` rather than silently
#: defaulting to one.
#:
#: **Every entry is None, and that is a finding rather than a shortcut.** A
#: growth policy is a frontier ORDER and nothing else:
#: `src/mojotrees/growth_policy.mojo`'s module docstring states that the module
#: "touches no
#: histogram, no device, and no dataset", that every grower keeping a frontier
#: asks `GrowthSchedule.next_leaf` for the slot to split, and that everything
#: after the pick is the grower's own per-split body and is the same under
#: both orders. The growers that keep a frontier are named there: `tree.
#: grow_tree`, `tree_sparse.grow_tree_sparse`, and the three loops in
#: `train_gpu.mojo`. Between them those cover every cell any scenario in this
#: suite runs, on both backends.
#:
#: Checked rather than assumed on the two that looked doubtful.
#: `tree_sparse.grow_tree_sparse`'s own docstring (`tree_sparse.mojo::grow_tree_sparse`)
#: says it grows "leaf-wise by default or depth-wise under
#: `params.grow_policy == GROW_DEPTHWISE` (the same `GrowthSchedule` order the
#: dense grower follows)", so `sparse_highdim` runs. And the device sizes a
#: level's worth of records for this mode specifically at
#: `train_gpu.mojo::_search_record_slots`, `2 * params.num_leaves`, so the GPU
#: cells run.
#:
#: The ONE documented refusal is the distributed prototype, which "tracks no
#: depth and rejects the policy" (`growth_policy.mojo` module docstring). No
#: scenario in
#: this suite runs a distributed cell, so no entry here is affected, and this
#: sentence exists so that the day one does, the refusal is already written
#: down.
#:
#: Note the contrast with the CatBoost-mode arm, which is absent from four
#: scenarios. That arm carries `bootstrap_type=MVS`, `score_function=Cosine`
#: and a symmetric grower, each of which is refused somewhere. This arm
#: changes one frontier order and carries nothing that any trainer refuses.
#:
#: **STILL ALL None AFTER THE ARM BECAME AN XGBOOST MIRROR ON 2026-08-17**, and
#: the paragraphs above needed re-reading rather than re-using. The arm now
#: carries six parameters beside the growth order, so "changes one frontier
#: order and carries nothing any trainer refuses" is no longer the argument.
#: The argument is now this: `max_depth`, `num_leaves`, `learning_rate`,
#: `lambda_l2`, `min_child_hess` and `min_data_in_leaf` are ordinary tree and
#: regularization parameters, none of them appears in any device or trainer
#: refusal list, and all six are already carried by `MOJOTREES_CATBOOST_MODE`
#: on the scenarios that arm runs. What keeps THAT arm off four scenarios is
#: `bootstrap_type`, `score_function` and the symmetric grower, and this arm
#: has none of the three.
#:
#: The table was NOT narrowed to the scenarios the `xgboost` peer runs, and
#: that was a decision rather than an oversight. Narrowing it would silently
#: delete this arm's categorical and ranking cells, which are measurements we
#: have and XGBoost's absence is no reason to stop taking. What those cells
#: lack is a peer column to be read against, and `MOJOTREES_DEPTHWISE_CLAIMS`
#: says so in the record rather than the schedule saying it by omission.
MOJOTREES_DEPTHWISE_SCENARIO_SUPPORT = {
    "categorical_missing": None,
    "dense_regression": None,
    "high_cardinality_categorical": None,
    "imbalanced_binary": None,
    "multiclass": None,
    "ordered_boosting_small": None,
    "ranking": None,
    "sparse_highdim": None,
}

# The depth-wise subject variant is added to the scenarios it can run, by the
# same loop shape the peer arms use below and for the same reason: a table
# consulted by a loop is checkable, and a scenario list edited by hand is not.
for _scenario_id, _reason in MOJOTREES_DEPTHWISE_SCENARIO_SUPPORT.items():
    if _reason is None:
        SCENARIOS[_scenario_id]["engines"] = list(
            SCENARIOS[_scenario_id]["engines"]
        ) + ["mojotrees_depthwise"]
del _scenario_id, _reason


# The peer arms are added to the scenarios they can run, and a scenario that
# cannot run them says why in `catboost_arm_block()["scenarios_not_run"]`
# rather than simply not appearing. Written as a loop over
# CATBOOST_SCENARIO_SUPPORT so that a scenario added without a support
# decision fails `selfcheck.check_catboost_arm` instead of silently
# defaulting to one.
for _scenario_id, _reason in CATBOOST_SCENARIO_SUPPORT.items():
    if _reason is None:
        _engines = list(SCENARIOS[_scenario_id]["engines"]) + list(
            CATBOOST_ENGINES
        )
        if MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT[_scenario_id] is None:
            _engines.append("mojotrees_catboost_mode")
        SCENARIOS[_scenario_id]["engines"] = _engines
del _scenario_id, _reason
if "_engines" in dir():
    del _engines


# The XGBoost peer arm, wired by the same loop shape and for the same reason:
# a table consulted by a loop is checkable, and a scenario list edited by hand
# is not. Kept as a SECOND loop rather than folded into the CatBoost one
# because the two support tables are independent -- a scenario CatBoost cannot
# run is not necessarily one XGBoost cannot -- and a single loop over a union
# would have to decide what an absent entry means for the other library.
#
# ONE ARM, where this loop used to add two. The mojotrees side of the XGBoost
# pairing is `mojotrees_depthwise`, which is added to its scenarios by the
# depthwise loop ABOVE and off its own, wider support table. The two tables
# disagree on purpose and the disagreement is not a defect: the depthwise arm
# runs the categorical and ranking scenarios, where XGBoost has no column here
# (XGBOOST_SCENARIO_SUPPORT says why for each), and on those scenarios its row
# has no peer to be read against and is a speed number beside the plain
# mojotrees arm's. MOJOTREES_DEPTHWISE_CLAIMS carries that sentence into the
# record so a reader of one of those cells is told rather than left to notice.
for _scenario_id, _reason in XGBOOST_SCENARIO_SUPPORT.items():
    if _reason is None:
        SCENARIOS[_scenario_id]["engines"] = list(
            SCENARIOS[_scenario_id]["engines"]
        ) + list(XGBOOST_ENGINES)
del _scenario_id, _reason


# ---------------------------------------------------------------------------
# The two CORRECTNESS arms, wired. Their dicts, their claims and their
# translators are further up, beside each other; see the NOTE ON PLACEMENT
# there. The support tables are HERE, beside their loops, which is where every
# other support table in this file sits.
# ---------------------------------------------------------------------------
#
# THE RULE BOTH TABLES FOLLOW, and it is narrower than the one the comparison
# arms follow. A comparison arm runs every scenario it CAN run, because every
# such cell is a measurement somebody may want. A correctness arm runs only
# the scenarios where its product exists, and its product is a
# `verify.check_device_agreement` verdict, which needs BOTH a cpu row and a
# gpu row of the same arm on the same scenario at the same thread count. So a
# scenario that declares no gpu support is not a cheaper cell for these arms,
# it is a cell with nothing in it: the arm would train, record a number
# nobody asked for, and produce no verdict.
#
# That rule alone empties five of the eight scenarios. `ranking`,
# `categorical_missing`, `high_cardinality_categorical`,
# `ordered_boosting_small` and `sparse_highdim` all declare `devices: ["cpu"]`
# (`SCENARIOS`), so `run.py` skips their accelerator cell by name and no pair
# can exist. Each entry below says so in its own words rather than pointing
# here, because a support table is read one row at a time.

#: Which scenarios the symmetric-colsample correctness arm runs. `None` means
#: it runs. One entry per scenario, checked against `SCENARIOS` by
#: `selfcheck.check_correctness_arms`, so a scenario added later without a
#: decision fails the self-check rather than silently defaulting to one.
MOJOTREES_SYMMETRIC_COLSAMPLE_SCENARIO_SUPPORT = {
    "dense_regression": None,
    "imbalanced_binary": (
        "the accelerator cell would be REFUSED and only the cpu cell would "
        "run, which for a correctness arm is no cell at all. This scenario "
        "can be handed a categorical column -- its generator is numeric but "
        "its real dataset, bank_marketing, has ten categorical columns, and "
        "scenario_has_categorical fails closed and treats variant=auto as "
        "both -- and device_policy BLOCK_GROW_POLICY refuses an oblivious "
        "grower beside a categorical column, because a symmetric level "
        "commits one (feature, bin) split for the whole level while the "
        "device level search evaluates ordinal thresholds only. The cpu is "
        "no exit either: tree.mojo::_check_oblivious raises on the same pair for the same "
        "reason, so a cpu-only row would not even be an unpaired "
        "measurement, it would be an infrastructure failure. (device_policy "
        "BLOCK_GROW_POLICY's own message says the cpu grower grows the same "
        "symmetric tree with categorical splits. That message is WRONG "
        "against tree.mojo::_check_oblivious as read on 2026-08-17; the disagreement is "
        "reported and is not this table's to fix, and this entry does not "
        "depend on which of the two is right, because either way there is no "
        "gpu cell.)"
    ),
    "multiclass": (
        "OUT ON COST AND ON AN UNCHECKED PATH, not on a refusal, and this is "
        "the entry most likely to change. It is the only other scenario that "
        "declares gpu support and has no categorical column, so it is the "
        "one place this arm could be widened to. It is out for two reasons. "
        "The first is the standing cost rule for correctness arms: one "
        "scenario is enough to hold a fix in place, and a second doubles the "
        "cells on a laptop six lanes share. The second is that a softmax "
        "round grows one tree per class through its own device loop, and "
        "nobody in this repository has run an OBLIVIOUS multiclass device "
        "fit; this lane did not run one either, so scheduling the cell would "
        "be betting a matrix's exit code on an untested combination. Widen "
        "it after somebody has fitted one by hand"
    ),
    "ranking": (
        "declares devices: [cpu], so there is no accelerator cell and "
        "check_device_agreement -- the only thing this arm produces -- has "
        "no pair to compare"
    ),
    "categorical_missing": (
        "declares devices: [cpu], so there is no pair to compare. It is also "
        "a categorical matrix, which the oblivious grower refuses on the cpu "
        "at tree.mojo::_check_oblivious unless CTR columns have replaced every categorical "
        "column, and this harness cannot ask for CTRs at all: they are a "
        "property of Dataset construction and dataset_params(spec) takes no "
        "engine"
    ),
    "high_cardinality_categorical": (
        "declares devices: [cpu], so there is no pair to compare, and the "
        "same categorical refusal as categorical_missing applies on the cpu. "
        "MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT's entry for this scenario "
        "traces the four blockers that a CTR-replacement fix would have to "
        "close, and none of them has moved"
    ),
    "ordered_boosting_small": (
        "declares devices: [cpu], so there is no pair to compare. The "
        "scenario would otherwise be a candidate: it is numeric and it is "
        "small, which is what a correctness arm wants"
    ),
    "sparse_highdim": (
        "declares devices: [cpu], so there is no pair to compare. There is "
        "also no sparse oblivious grower: GrowthSchedule refuses "
        "GROW_OBLIVIOUS on the sparse path, which "
        "FRONTIER_SCENARIO_SUPPORT already records from the other side"
    ),
}

#: Which scenarios the Cosine-leaf-wise correctness arm runs. Same shape, same
#: fail-closed rule, and it is a SEPARATE table from the one above rather than
#: a shared one. The two arms are refused for different reasons on the same
#: scenarios -- one by the oblivious grower and one by the Cosine scan -- and a
#: shared table would have to state the union, which is how a reason comes to
#: be quoted for the arm it is not about.
MOJOTREES_COSINE_LEAFWISE_SCENARIO_SUPPORT = {
    "dense_regression": None,
    "imbalanced_binary": (
        "score_function=Cosine is refused beside a categorical column on "
        "BOTH backends, so this scenario would be an infrastructure failure "
        "rather than an unpaired row. split.mojo::find_best_split raises because a "
        "category set is searched and scored with the L2 gain and only that "
        "search's winner reaches the numerical scan, so the two functionals "
        "would end up inside one argmax; device_policy BLOCK_SCORE_FUNCTION "
        "refuses the same pair for the same reason, and its message says in "
        "as many words that selecting device=cpu does NOT lift it. This "
        "scenario can be handed a categorical column, because its real "
        "dataset bank_marketing has ten and scenario_has_categorical treats "
        "variant=auto as both variants"
    ),
    "multiclass": (
        "OUT ON COST AND ON AN UNCHECKED PATH, the same entry as the "
        "symmetric-colsample arm's and for the same two reasons. It is "
        "numeric and it declares gpu support, so it is where this arm would "
        "be widened first; a softmax round grows one tree per class through "
        "its own device loop and nobody here has run a Cosine multiclass "
        "device fit, and one scenario is enough to hold a fix in place"
    ),
    "ranking": (
        "declares devices: [cpu], so there is no accelerator cell and "
        "check_device_agreement -- the only thing this arm produces -- has "
        "no pair to compare"
    ),
    "categorical_missing": (
        "declares devices: [cpu], so there is no pair to compare, and Cosine "
        "is refused beside a categorical column on both backends "
        "(split.mojo::find_best_split), so the cpu row would raise rather than measure"
    ),
    "high_cardinality_categorical": (
        "declares devices: [cpu], so there is no pair to compare, and the "
        "same Cosine-beside-a-categorical refusal applies on the cpu"
    ),
    "ordered_boosting_small": (
        "declares devices: [cpu], so there is no pair to compare. Numeric "
        "and small, so it is the scenario this arm would run if it ever "
        "declared gpu support"
    ),
    "sparse_highdim": (
        "declares devices: [cpu], so there is no pair to compare. NOT "
        "checked against the sparse trainer's own refusals, because the "
        "device question settles it first and this lane does not state what "
        "it did not read"
    ),
}

# The two correctness arms are added to the scenarios they run, by the same
# loop shape every other arm uses and for the same reason: a table consulted
# by a loop is checkable, and a scenario list edited by hand is not.
for _support, _engine in (
    (MOJOTREES_SYMMETRIC_COLSAMPLE_SCENARIO_SUPPORT, "mojotrees_symmetric_colsample"),
    (MOJOTREES_COSINE_LEAFWISE_SCENARIO_SUPPORT, "mojotrees_cosine_leafwise"),
):
    for _scenario_id, _reason in _support.items():
        if _reason is None:
            SCENARIOS[_scenario_id]["engines"] = list(
                SCENARIOS[_scenario_id]["engines"]
            ) + [_engine]
del _support, _engine, _scenario_id, _reason


# ---------------------------------------------------------------------------
# THE FRONTIER BLOCK. Owned by lane/frontier-block; the arms themselves live in
# bench/real_data/frontier.py.
# ---------------------------------------------------------------------------
#
# The arms are in their own module rather than here for one reason and it is
# not size: they are a PLAN and not a matrix this harness can run. The job
# identity in `run.py` is (scenario, engine, device, threads, repeat) and has
# no arm dimension, so a frontier arm cannot be scheduled today, and putting
# unschedulable arms in the file `run.py` reads from is how a plan comes to be
# mistaken for a configuration. `frontier.WIRE_NOTES` lists what has to change,
# none of it in this lane's files.
#
# What is here is the part that belongs beside the other support tables: which
# scenarios the frontier runs on, and why the rest are absent. The pattern is
# `CATBOOST_SCENARIO_SUPPORT`'s, fail-closed for the same reason -- a scenario
# added without a decision should be a KeyError at import and not a silent
# "runs".

#: The frontier's scenario support. `None` runs; a string is the reason it does
#: not, and the reason is a property of the QUESTION rather than of the
#: trainer. A scenario is absent from the frontier because measuring it does
#: not help decide the shipped defaults, and a scenario the trainer refuses is
#: recorded in `frontier.UNREACHABLE` instead, which is a different fact.
FRONTIER_SCENARIO_SUPPORT = {
    "dense_regression": None,
    "high_cardinality_categorical": None,
    "imbalanced_binary": (
        "not in the frontier's first pass. It is a real question and it is "
        "the scenario where the float32 derivative default already cost 9.4 "
        "percent of average precision, so the precision axis has more to say "
        "here than anywhere else. It is out because the three rows Andrew "
        "ordered are the dense decision row, its real variant and one "
        "categorical row, and adding a fourth is his call with a wall clock "
        "attached rather than this lane's"
    ),
    "multiclass": (
        "out for the same scheduling reason, and it would also lose two of "
        "the four axes if it were in: the multiclass trainer takes no "
        "bootstrap bundle, so the subsample axis has no meaning on it "
        "(MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT['multiclass'])"
    ),
    "ranking": (
        "out, and the trees axis would be half a question here: CatBoost's "
        "coefficient table has no row for a ranking objective, so `T@auto` "
        "leaves the rate alone and the auto points of the axis are not auto. "
        "See frontier.UNREACHABLE['auto_rate_on_ranking']"
    ),
    "categorical_missing": (
        "out, and it is the categorical scenario that was NOT chosen rather "
        "than one nobody considered. It has no CatBoost row at all and never "
        "will, so the peer scoreboard on it would be set by LightGBM alone "
        "at every tree count, where the directive says the BETTER of the "
        "two. frontier.CATEGORICAL_ALTERNATIVE carries the argument"
    ),
    "ordered_boosting_small": (
        "out. Its row count is its identity -- it exists so that CatBoost's "
        "boosting_type resolution is the thing under test -- and the "
        "frontier varies tree counts and sampling, both of which change what "
        "that scenario is asking"
    ),
    "sparse_highdim": (
        "out. Three of the four axes are refused on the sparse path: the "
        "bootstrap by name (`sampling.check_bootstrap_honored`), the "
        "symmetric grower because `GrowthSchedule` refuses GROW_OBLIVIOUS "
        "and there is no sparse oblivious grower, and the accelerator "
        "because the CSC path reports cpu whatever is asked for. A frontier "
        "row with one live axis is not a frontier row"
    ),
}

#: Where the arms are, so that a reader who finds this table first is not left
#: looking for them.
FRONTIER_ARMS_MODULE = "bench/real_data/frontier.py"

#: The one thing about the frontier that has to be true in THIS file, because
#: it is this file's tables that a competitor row is built from.
#:
#: The frontier judges an arm against the better of CatBoost-as-shipped and
#: LightGBM stock+det AT A MATCHED TREE COUNT, and it abstains rather than
#: substituting another count. `catboost_readback_key` is (scenario, tier,
#: variant) and carries no tree count, so a run that puts CatBoost at more
#: than one tree count on one cell writes every read-back to one key and the
#: last one wins. The frontier runs CatBoost at five tree counts per cell.
#:
#: Nothing in the frontier reads it -- no frontier arm takes the read-back
#: route, because the mode-defaults layer means Base A derives its own rate
#: while carrying CatBoost's l2 of 3 -- so this is a latent collision rather
#: than a live defect. It is named here because the key is defined here, and
#: because `mojotrees_catboost_mode` DOES take the read-back route: a frontier
#: run that also scheduled that arm would hand it whichever tree count wrote
#: last.
FRONTIER_READBACK_LIMITATION = (
    "catboost_readback_key carries no n_estimators, so the frontier's five "
    "competitor tree counts on one cell collide on one key. Latent while no "
    "frontier arm reads it; live the moment a frontier run also schedules "
    "mojotrees_catboost_mode. See frontier.WIRE_NOTES["
    "'readback_key_has_no_tree_count']"
)

for _frontier_scenario in SCENARIOS:
    if _frontier_scenario not in FRONTIER_SCENARIO_SUPPORT:
        raise KeyError(
            f"{_frontier_scenario!r} has no FRONTIER_SCENARIO_SUPPORT "
            "decision. A scenario that joins this suite needs one, because "
            "the alternative is defaulting silently to 'the frontier runs "
            "here' and scheduling a row nobody costed"
        )
del _frontier_scenario
