"""The six scenarios, and the one comparator both benchmarks run against.

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

## What is still set, and what it would take to unset it

- `deterministic = true`. The comparator itself, framed above.
- `lambda_l2 = 1.0`, in `BASE_PARAMS` and therefore on both sides.
  mojotrees defaults to 1.0 and LightGBM to 0.0, so it is set explicitly
  on both or the comparison is between two different regularisers.
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
  for both engines would raise on one of the six scenarios rather than
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
#: These are LightGBM's defaults except for lambda_l2, which is mojotrees's.
BASE_PARAMS = {
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_data_in_leaf": 20,
    "min_child_hess": 1e-3,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "use_missing": True,
}

#: The comparator's identity, carried by every run and every published
#: table. Bump `COMPARATOR_VERSION` when the resolved dict changes in a way
#: that makes two numbers non-comparable, which is any change to
#: `LIGHTGBM_ALIGNMENT` other than a comment.
COMPARATOR_ID = "stock+det"
COMPARATOR_VERSION = 1
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
LIGHTGBM_STOCK_DEFAULTS = {
    "use_quantized_grad": False,
    "bin_construct_sample_cnt": 200000,
    "min_data_in_bin": 3,
    "force_row_wise": False,
    "force_col_wise": False,
    "zero_as_missing": False,
    "min_gain_to_split": 0.0,
    "boost_from_average": True,
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
    }


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
    shared = shared_params(spec, extra)
    params = {
        "objective": shared["objective"],
        "num_leaves": shared["num_leaves"],
        "max_depth": shared["max_depth"],
        "learning_rate": shared["learning_rate"],
        "min_data_in_leaf": shared["min_data_in_leaf"],
        "min_sum_hessian_in_leaf": shared["min_child_hess"],
        "lambda_l1": shared["lambda_l1"],
        "lambda_l2": shared["lambda_l2"],
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
        "lambda_l1": shared["lambda_l1"],
        "lambda_l2": shared["lambda_l2"],
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
