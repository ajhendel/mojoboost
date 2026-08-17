"""The speed and accuracy frontier: a PLAN and its arms. Nothing here measures.

    python bench/real_data/frontier.py              # the plan, as a table
    python bench/real_data/frontier.py --json p.json # the same, machine-readable
    python bench/real_data/frontier.py --skips      # only the declared skips

WHAT THIS FILE IS FOR CHANGED ON 2026-08-17, AND THE ARMS DID NOT. The plan
below, every arm in it and every skip it declares, is unchanged. What changed
is which accuracy the frontier spends.

The directive this was written to serve, and it is recorded here as it stood
because a document that quietly rewrites its own premise is worth less than
one that shows where it moved: accuracy within 1 percent relative on the
primary metric, against the BETTER of CatBoost-as-shipped and LightGBM
stock+det AT A MATCHED TREE COUNT, is a BUDGET to be spent for speed, and the
frontier is the set of configurations that spend it.

The directive now, Andrew 2026-08-17: "it should be about trading OUR accuracy
for speed. not tied to whatever lightgbm or catboost is fucking doing." So the
budget being spent is OUR OWN accuracy against our own recorded anchor, and
the peer distance is a scoreboard column that gates nothing. The old rule
failed in both directions at once: it blocked a change that cost no accuracy
at all, and it permitted a real 1.07 percent loss on any arm that happened to
be ahead of CatBoost. `verify.py`'s THE TWO ACCURACY AXES block carries the
full argument.

What survives unchanged is the shape of the question: "what is the fastest
configuration whose accuracy we are willing to pay for", answered at ONE tree
count, and the recommendation is DOCUMENTED and never applied. `report.py`
names the fastest of our arms and nothing in this repository reads that name
back into a default.

THIS FILE RUNS NOTHING. It enumerates arms, declares which of them cannot run
and why, counts cells, and estimates wall clock from timings already recorded
in `bench/results/COMPARISON_RUN_2026-08-16.md`. It takes no lock, imports no
trainer, and fits nothing.

WHAT THIS DESIGN CANNOT ANSWER, first, because a reader who takes only one
paragraph from this file should take this one
------------------------------------------------------------------------

**It is a one-axis-at-a-time sweep from two base points. It is not a grid and
it cannot see an interaction.** Every arm below differs from its base in
exactly ONE axis. So the frontier can say what `max_bin=63` costs at the
base's tree count and what `1000` trees cost at the base's `max_bin`, and it
cannot say what `max_bin=63` costs at 1000 trees. If those two axes interact
-- and coarser bins with more trees is exactly the pair where one would
expect them to, since more rounds can recover what a coarser split lost --
this design will report each effect alone and no combination of the two.

The same holds for every other pair: `max_bin` x precision, subsample x
trees, precision x subsample, and both of those against the grow policy. None
is measured. The conditional interaction check
(`INTERACTION_CHECK`) covers exactly ONE pair, on ONE scenario, on ONE
backend, and only if the one-axis results suggest it, so after it runs the
count of measured interactions is one and the count of unmeasured ones is
still every other pair.

**A one-axis sweep from a base is also blind to a better point that is two
axes away from both bases.** If the fastest inside-budget configuration is
`max_bin=63` AND `subsample=0.5` together, no arm here is that configuration
and the report will not name it. What the report names is the fastest point
AMONG THE ARMS THAT RAN, which is a weaker statement than "the fastest
configuration", and `report.py` says so in the same breath as the number.

The literal cross product of the axes Andrew named is 360 arms per scenario
and 2,160 across six scenarios, which is 25,920 cells at twelve repeats: 72
hours at ten seconds a cell and 432 at sixty. That is the arithmetic this
design exists to avoid, and avoiding it is what costs the interactions. The
trade is stated rather than hidden.

THE TWO BASES
-------------

**Base A is the new shipped default shape** and **it is CPU, on every
scenario, for two independent reasons that are properties of the trainer and
not of this harness.** MVS is refused on the accelerator by name
(`model.fit`: "bootstrap_type is not implemented on the GPU"), and a
symmetric tree carrying `score_function=Cosine` is refused by the device
grower (`train_gpu`: "grow_policy=oblivious cannot be grown on this device",
because `ExtraTreeParams.is_active()` is true under Cosine and
`oblivious_device_supported` declines on it). "The best backend that honors
it" resolves to the CPU, and it would still resolve to the CPU if either
refusal were lifted alone.

**Base B is the lossguide arm this suite has always run**, `100@0.1` at
`BASE_PARAMS`, on both backends where the scenario declares one. 100 is
LightGBM's own count, which is what makes this base the MATCHED comparison as
well as a base.

THREE BLOCKS, AND THE BLOCKS ARE THE CONSTRAINT
-----------------------------------------------

The shipped defaults are per grow policy: `symmetrictree` at 360 trees,
`lossguide` at 72. Neither is a competitor's count, because the parity
contract (89e0c96) mirrors each library's INTERNALS and not its tree budget.
So the frontier holds three kinds of mojotrees row and they answer different
questions: `defaults` is what we ship, `matched` is equal-budget against the
library it mirrors, and `axis` is everything the sweep moved. `BLOCKS` carries
the argument.

**A cross-count comparison is made impossible rather than captioned.** The
budget verdict is computed inside a tree-count group and has no fallback, the
speed ranking is built inside a tree-count group so a group with two counts
cannot exist, and an arm whose count has no competitor row abstains and is
listed as unjudged rather than ranked. A 360-tree arm therefore has no path to
a table beside a 100-tree competitor, and a reader who wanted to draw that
comparison would have to build the table themselves.

THE AUTO RATE, AND WHAT THE MODE SUPPLIES BESIDE IT
---------------------------------------------------

`T@auto` here means `auto_learning_rate=True` with `learning_rate` left
unset, so the rate is derived by CatBoost's own formula from the train row
count, the iteration count and the loss
(`src/mojotrees/auto_learning_rate.mojo`). **It is a function of T, so every
tree count gets its own rate and no constant appears anywhere in these
arms.** CatBoost's own resolution has been measured on this machine at 0.5 at
2 iterations, 0.4273 at 100 and 0.06573 at 1000, and a live read-back at
smoke shapes returned 0.331 and 0.278 at 100 iterations on two different
datasets, so the rate moves with rows and features as well as with T. An arm
that carried a rate measured at another budget would be a different fit
wearing this one's label.

**Base A runs at `lambda_l2 = 3.0` AND a derived rate, both, and the way it
gets both is the mode-defaults layer** (`params._apply_catboost_mode_defaults`
on the string surface, `_Base._params`'s `l2_default` on the Python one, which
is the surface this harness trains through). Under
`grow_policy=symmetrictree` the mode supplies `learning_rate` 0.03,
`reg_lambda` 3.0, a per-objective `leaf_estimation_iterations`,
`random_strength` 1.0 on the CPU, and `max_depth` 6 when none was given. Every
one of them is supplied with `TOption::SetDefault` semantics: the value is
assigned and the provenance flag is NOT raised, exactly as CatBoost does it
(`catboost_options.cpp:302`), so the two of them that are keys of the
automatic-learning-rate gate do not close it. The demonstrated behavior is a
symmetric fit with nothing named deriving 0.29699 and recording
`auto_lr_gate_open` while carrying `l2_leaf_reg = 3.0`.

**So the arms must leave those keys UNNAMED, and that is the whole
requirement.** A value a caller types arrives through `operator=`, raises the
flag, and pins the rate back to the constant 0.03 with
`auto_lr_skipped:l2_leaf_reg` recorded. Base A therefore names none of
`learning_rate`, `lambda_l2` or `leaf_estimation_iterations`, and it is
CatBoost's regularizer that it runs at rather than LightGBM's.

**This paragraph replaces a finding this lane reported and got wrong**, and
the correction is recorded in `RESOLVED_SINCE` rather than quietly swapped:
the first version of this file said the combination could not be built at all
and drew the sharpest consequence from it, that Cosine at `reg_lambda = 0`
collapses onto `sqrt` of the L2 score and Base A's Cosine axis therefore sat
near its degenerate point. **That does not hold.** At `reg_lambda = 3` the
score function is off the degenerate point outright, which is what
`sklearn.py` says of exactly this configuration. The reasoning was correct on
the tree it was taken from (`bda3b41`); the mode-defaults layer merged after
it at `ad6c2b6`.

WHAT IS NOT WIRED
-----------------

`WIRE_NOTES` is the list of changes outside this lane's files that scheduling
this plan needs. The harness's job identity is
`(scenario, engine, device, threads, repeat)` and has no arm dimension, so
today `run.py` cannot execute a single arm below. That is stated as a
blocker rather than worked around, because a frontier that quietly ran at
`BASE_PARAMS` and reported itself as a sweep is precisely the failure this
campaign exists to prevent.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scenarios  # noqa: E402

FRONTIER_ID = "frontier-1"
FRONTIER_VERSION = 1
FRONTIER_REGISTERED = "lane/frontier-block, 2026-08-16"

#: Repeats per cell. **Three, ruled by Andrew on 2026-08-16**, against the
#: five this lane costed and the five `COMPARISON_RUN_2026-08-16.md` ran.
#: Three is the minimum that shows a spread; twelve is the number the canary
#: refused. A cell is one whole process, so cells = arms x REPEATS and
#: nothing else, and repeats scale the wall clock linearly.
#:
#: **What three costs, stated because the choice was made against a price.**
#: This repository resolves a margin by disjoint ranges rather than by
#: medians -- Block A's headline stands because our slowest repeat beat
#: LightGBM's fastest -- and three repeats makes a range easier to overlap.
#: A 10 percent effect that five repeats would resolve may come back
#: indistinguishable. The other cut this lane priced, dropping the 1000-tree
#: arms, was DECLINED: they stay.
REPEATS = 3

#: Tree counts the competitor rows run at. Every mojotrees arm's tree count
#: must appear here or its verdict has nothing to stand on: `verify.py`
#: ABSTAINS on a missing matching-count competitor rather than comparing
#: against a different count, which would look like a result and be a
#: category error.
#:
#: **The instructed set is 100, 360 and 1000. This lane added 72 and 50, and
#: both additions are forced by the abstention rule rather than chosen.**
#:
#: `check()` enforces the rule that produced them: an arm at a tree count no
#: competitor runs at can never be judged, and an unjudged arm can never be
#: ranked or recommended, so it is a cell that costs time and answers nothing.
#: Two arms in the instructed design landed in exactly that state.
#:
#: - **72**, because that is the shipped lossguide default. Without it the
#:   single most important row in the frontier -- what we actually ship on the
#:   leaf-wise path -- would carry no verdict at all.
#: - **50**, because `50@0.2` is on the trees axis Andrew named. It is the
#:   "fewer trees at a higher rate" lever, which is a speed lever, and a speed
#:   lever with no accuracy verdict is exactly the thing this budget exists to
#:   stop anyone spending.
#:
#: They cost four competitor arms per scenario row and they are the cheapest
#: arms in the plan, since a 50-tree and a 72-tree fit are half and three
#: quarters of a 100-tree one. Drop either and the corresponding row goes
#: unjudged and unrankable; nothing else changes.
COMPETITOR_TREES = (50, 72, 100, 360, 1000)

#: The shipped defaults, PER GROW POLICY, as of Andrew's 2026-08-16 split.
#:
#: `symmetrictree` is the default policy and carries CatBoost's set at 360
#: trees with a derived rate. `lossguide` mirrors LightGBM's internals -- its
#: rate, its 31 leaves, its categorical rules -- at OUR OWN tree count of 72.
#: The tree count is the one thing lossguide does not take from LightGBM, and
#: that is the whole reason the two shipped rows cannot be read against a
#: competitor at another count.
SHIPPED_N_ESTIMATORS = {"symmetrictree": 360, "lossguide": 72}

#: Which block a row belongs to, and the blocks are the constraint.
#:
#: There are three kinds of mojotrees row here and they answer different
#: questions. A reader who merges them gets a win that is not one, so they are
#: separated by the DATA STRUCTURE and not by a caption: `block` is a field on
#: every arm, `report.py` renders one table per block per tree count, and
#: `verify.check_accuracy_peer` computes each scoreboard line against
#: competitors at that row's own count with no fallback of any kind. (That
#: function was `check_accuracy_budget` and was the accuracy GATE until
#: 2026-08-17, when the gate moved to `verify.check_accuracy_anchor`, which
#: compares an arm against our own recorded accuracy and involves no
#: competitor. The matched-tree-count rule described here is unchanged; only
#: what the number gates changed, and it now gates nothing.)
#:
#: **A cross-count comparison is therefore impossible rather than
#: discouraged.** There is no code path that puts a 360-tree arm and a
#: 100-tree competitor into one ordering: a ranking is built inside a
#: (cell, tree count) group, so a group containing two counts cannot exist,
#: and an arm whose count has no competitor row abstains and is listed as
#: unjudged rather than ranked. A missing row arriving as a quiet fallback is
#: the same defect wearing different clothes, which is why the abstention is
#: part of this rule rather than a separate one.
BLOCKS = {
    "defaults": (
        "WHAT WE SHIP. Base A at 360 under symmetrictree and Base B at 72 "
        "under lossguide, per Andrew's 2026-08-16 split. The parity contract "
        "at 89e0c96 carries both mirroring decisions: lossguide mirrors "
        "LightGBM's internals and per-parameter defaults and NOT its tree "
        "budget, symmetrictree mirrors CatBoost's and NOT its budget. So "
        "these two rows are at counts no competitor chose, and each is "
        "judged only against competitor rows at its own count. 72 is in "
        "COMPETITOR_TREES for exactly this reason"
    ),
    "matched": (
        "EQUAL BUDGET. Base B at 100@0.1 against LightGBM stock+det at 100, "
        "and Base A at 100@auto against CatBoost as shipped at 100. These "
        "are the only rows where 'faster at the same settings' is a sentence "
        "anybody may write, and they exist because the defaults rows cannot "
        "carry that sentence"
    ),
    "axis": (
        "EXPLORATORY. Every other arm: one axis moved from its base to find "
        "where the accuracy budget can be spent. An axis row is ranked "
        "against the other rows at ITS OWN tree count and nowhere else"
    ),
}

#: Which counts count as matched, per base. A matched row is one whose tree
#: count is the count the competitor it mirrors actually ran at.
MATCHED_TREES = {"A": 100, "B": 100}


# ---------------------------------------------------------------------------
# The scenario rows, in the order Andrew set them.
# ---------------------------------------------------------------------------
#
# `dense_regression --tier large` is the decision row this campaign already
# reports on, in both variants: `synthetic` is Block A of
# COMPARISON_RUN_2026-08-16.md at 799,110 train rows by 100 features, and
# `auto` resolves to UCI YearPredictionMSD at 463,715 by 90, which is Block B.
# Same scenario, two data kinds, and they are separate rows here because they
# have already been shown to behave differently: the same GPU histogram
# mechanism produced 1.04e-05 relative CPU/GPU divergence on one and 1.06 on
# the other.
#
# The categorical row is `high_cardinality_categorical --tier standard`, Block
# C, at 799,110 by 15. It is third because it is where Base A does not run at
# all, so it is the row that answers least about the shipped defaults, and it
# is the categorical scenario with BOTH competitors: `categorical_missing`
# has no CatBoost row at all and never will
# (CATBOOST_SCENARIO_SUPPORT['categorical_missing']), so a frontier run on it
# would have its budget set by LightGBM alone at every tree count. That is a
# weaker gate than the directive asks for, which is why it is the declared
# alternative rather than the choice.
SCENARIO_ROWS = (
    {
        "row": "dense_synthetic",
        "scenario": "dense_regression",
        "tier": "large",
        "variant": "synthetic",
        "devices": ("cpu", "gpu"),
        "why": (
            "the decision row. Block A of COMPARISON_RUN_2026-08-16.md, "
            "799,110 train rows by 100 features, primary metric RMSE, and the "
            "one row where mojotrees on the GPU is already faster than the "
            "comparator. If the budget buys speed anywhere it has to buy it "
            "here"
        ),
    },
    {
        "row": "dense_real",
        "scenario": "dense_regression",
        "tier": "large",
        "variant": "real",
        "devices": ("cpu", "gpu"),
        "why": (
            "real data, UCI YearPredictionMSD, 463,715 by 90. Block B, and "
            "the row we currently LOSE on: LightGBM is 1.30x faster than our "
            "best backend with the accuracy a tie at 0.03 percent. A budget "
            "worth spending should be visible here first"
        ),
    },
    {
        "row": "categorical",
        "scenario": "high_cardinality_categorical",
        "tier": "standard",
        "variant": "synthetic",
        "devices": ("cpu",),
        "why": (
            "Block C, 799,110 by 15, primary metric AUC, and the only "
            "categorical scenario with both competitors. CPU only: the "
            "scenario declares no accelerator, and Base A does not run here "
            "at all -- see UNREACHABLE['symmetric_categorical']"
        ),
    },
)

#: The categorical row that was NOT chosen, and why, so that the choice is a
#: decision on the record rather than an omission.
CATEGORICAL_ALTERNATIVE = {
    "scenario": "categorical_missing",
    "reason": (
        "no CatBoost row, permanently: a missing category has no encoding "
        "that is the same data on all three engines "
        "(CATBOOST_SCENARIO_SUPPORT['categorical_missing']). The budget "
        "would then be set by LightGBM alone at every tree count, and the "
        "directive says the BETTER of the two. Run it if the missing-value "
        "path is the question; it is not this frontier's question"
    ),
}


# ---------------------------------------------------------------------------
# The two bases.
# ---------------------------------------------------------------------------
#
# `params` are canonical `shared_params` names, applied over `BASE_PARAMS` the
# way `MOJOTREES_CATBOOST_MODE` is, so an arm's resolved dict is a diff a
# reader can take against the plain arm's. `env` carries settings that have no
# Python parameter at all and reach the trainer only through the environment.
BASES = {
    "A": {
        "id": "A",
        "label": "the shipped defaults: symmetric depth 6, 360@auto, MVS 0.8, Cosine",
        "engine": "mojotrees",
        # Both are ENUMERATED. Until 2026-08-17 the accelerator leg was
        # entirely skipped rather than absent, because a backend this base
        # cannot use is a fact a reader has to be able to see; nine arms that
        # were never written down look like nine arms nobody thought of.
        "devices": ("cpu", "gpu"),
        "device_reason": (
            "Both legs schedule since 2026-08-17. Until then this base was CPU "
            "on every scenario for two independent refusals, MVS on the GPU "
            "and symmetric trees under Cosine on the GPU, and both are gone at "
            "head: see CAPABILITIES_AT_HEAD and CAPABILITIES_CHECKED_AT for "
            "what was read and where. The GPU leg has never been RUN under "
            "this plan, only unrefused, so its first run is read for the "
            "device_agreement verdict and the backend proof before its timing"
        ),
        "device_divergence_if_cosine_lands": (
            "RESOLVED 2026-08-17 by reading params.mojo: "
            "`_apply_catboost_mode_defaults` no longer tests the device, so "
            "the mode default random_strength=1.0 is supplied on the GPU as on "
            "the CPU, and `ExtraTreeParams.device_unsupported_reason` refuses "
            "it only beside a categorical column. So a symmetric GPU fit "
            "carries the same regularizer as its CPU twin and the "
            "cpu-versus-gpu row compares two backends, which is what its label "
            "says. Kept under its old key so a reader who saw the warning "
            "finds the resolution beside it"
        ),
        "params": {
            "grow_policy": "symmetrictree",
            # Named, and it is the one mode default this base states rather
            # than inherits. `max_depth` is resolved from the VALUE and not
            # from provenance -- -1 is the absence of a bound and not a depth
            # -- and it is not one of CatBoost's four gate keys, so naming 6
            # cannot close the derivation gate under either reading. It is
            # here because a symmetric tree is bounded by its depth and by
            # nothing else, and `BASE_PARAMS` carries -1.
            "max_depth": 6,
            "num_leaves": 64,
            "min_data_in_leaf": 1,
            "min_child_hess": 0.0,
            "score_function": "cosine",
            "bootstrap_type": "MVS",
            "subsample": 0.8,
            "n_estimators": SHIPPED_N_ESTIMATORS["symmetrictree"],
            "auto_learning_rate": True,
        },
        "dataset_params": {"max_bin": 254},
        "env": {"MOJOTREES_DERIVATIVE_PRECISION": "float32"},
        "learning_rate": "auto",
        #: Supplied by the mode, not by this dict, and the distinction is the
        #: mechanism. Each of these is assigned with the provenance flag left
        #: down, so the two that are gate keys leave the derivation open.
        "mode_supplies": {
            "reg_lambda": "3.0, CatBoost's l2_leaf_reg",
            "learning_rate": (
                "0.03, the constant this base does not use, because the "
                "derivation fires and replaces it"
            ),
            "leaf_estimation_iterations": (
                "CatBoost's per-objective count with the small-run stomp; 1 "
                "under RMSE, which is what both dense rows are"
            ),
            "random_strength": (
                "1.0 on both backends since 2026-08-17 (c775959 stages and "
                "reads the plane on the oblivious device path); it read "
                "\"on the CPU only\" before that and the entry above records "
                "the resolution"
            ),
            "max_depth": "6 when unnamed; this base names it anyway",
        },
        "not_set": (
            "learning_rate, lambda_l2 and leaf_estimation_iterations, all "
            "three for one reason: a value a caller TYPES raises the "
            "provenance flag and pins the rate back to the constant 0.03, "
            "recorded as auto_lr_skipped:<key>. Left unnamed they arrive "
            "from the mode with the flag down and the gate open. This base "
            "therefore runs at CatBoost's lambda_l2 of 3.0 AND at a derived "
            "rate, which is what CatBoost itself does",
        ),
    },
    "B": {
        "id": "B",
        "label": "lossguide at LightGBM's own count, 100@0.1, the matched base",
        "engine": "mojotrees",
        "devices": ("cpu", "gpu"),
        "device_reason": (
            "both backends where the scenario declares one. Nothing in this "
            "base is refused on the accelerator: leaf-wise growth, the L2 "
            "score function, no bootstrap and float32 derivatives are the "
            "device path's own configuration"
        ),
        "params": {
            "grow_policy": "lossguide",
            # 100, which is LightGBM's count, because this base's job is the
            # MATCHED comparison: LightGBM's internals at LightGBM's budget,
            # so that "faster at the same settings" is a sentence somebody may
            # write about this row. Our shipped lossguide budget is 72 and it
            # is a point on the trees axis below, tagged as the defaults row.
            # One tree count can be an axis point, a matched base and a
            # shipped default at once; what it cannot be is compared across
            # counts.
            "n_estimators": 100,
            "learning_rate": 0.1,
        },
        "dataset_params": {"max_bin": 255},
        "env": {"MOJOTREES_DERIVATIVE_PRECISION": "float32"},
        "learning_rate": 0.1,
        "not_set": (),
    },
}


# ---------------------------------------------------------------------------
# The axes, varied ONE AT A TIME from each base.
# ---------------------------------------------------------------------------
#
# A value equal to the base's own value is the base point and is not a second
# arm. That is why Base A has three `max_bin` arms (its base is 254, which is
# CatBoost's border_count and is not one of the three) and Base B has two (its
# base IS 255).
AXES = {
    "trees": {
        "label": "trees x learning rate",
        "A": (
            (100, "auto"),
            (SHIPPED_N_ESTIMATORS["symmetrictree"], "auto"),
            (1000, "auto"),
        ),
        "B": (
            (100, 0.1),
            (SHIPPED_N_ESTIMATORS["lossguide"], 0.1),
            (50, 0.2),
            (100, "auto"),
            (SHIPPED_N_ESTIMATORS["symmetrictree"], "auto"),
            (1000, "auto"),
        ),
        "note": (
            "`auto` is CatBoost's derived rate, a function of the row count, "
            "the iteration count and the loss, so each tree count carries its "
            "own value and no constant is written down. The pinned pairs "
            "belong to Base B because a pinned rate closes the gate Base A's "
            "identity depends on. "
            "**100@0.1 is Base B's base point and 72@0.1 is a point on this "
            "axis, and the choice is deliberate in both directions.** 100 is "
            "LightGBM's own count, so a base at 100 is the configuration "
            "every other axis is varied from AND the one matched comparison "
            "available on this policy. 72 is what we ship, so it is on the "
            "axis and it is tagged as the lossguide defaults row. A tree "
            "count is allowed to be all three things at once; what is never "
            "allowed is reading one count's row against another count's "
            "competitor, which `BLOCKS` and the per-count grouping in "
            "`verify.py` and `report.py` make structurally impossible rather "
            "than discouraged"
        ),
    },
    "subsample": {
        "label": "row sampling",
        "A": ((1.0, None), (0.8, "MVS"), (0.5, "MVS")),
        "B": ((1.0, None), (0.8, "MVS"), (0.5, "MVS")),
        "note": (
            "`1.0` means no bootstrap at all rather than MVS at full rate: "
            "`bootstrap_type` is dropped, because MVS at subsample 1.0 is "
            "still a reweighting and is not the unsampled fit"
        ),
    },
    "max_bin": {
        "label": "binning granularity",
        "A": (255, 127, 63),
        "B": (255, 127, 63),
        "note": (
            "a Dataset parameter, not a training one: `dataset_params` owns "
            "it and `mojotrees_params` refuses it. Base A's base value is "
            "max_bin=254, which is 254 BINS in this repository's vocabulary "
            "(binning.mojo, and scenarios.CATBOOST_PARAM_MAP['border_count'] "
            "with translate=borders_to_bins: 254 borders is 255 bins, +1 and "
            "not identity). It is therefore ONE BIN FEWER than CatBoost's "
            "default of 254 borders and than the 255-bin axis value, not the "
            "same granularity, which this note claimed until 2026-08-17. It "
            "is not one of the three axis values, so all three are variations "
            "from it; whether the base itself should be 255 to mirror "
            "CatBoost is a plan decision left open, because 255 is an axis "
            "value and `base_arms` folds an axis value equal to the base "
            "into the base point"
        ),
    },
    "precision": {
        "label": "derivative precision",
        "A": ("float32", "float64"),
        "B": ("float32", "float64"),
        "note": (
            "reachable ONLY through the environment: "
            "`MOJOTREES_DERIVATIVE_PRECISION`, read by "
            "`histogram.derivative_precision_narrows`. There is no estimator "
            "keyword, so this axis is a per-cell environment change and not a "
            "parameter (WIRE_NOTES['per_cell_env']). float64 turns off the "
            "gathered pair buffer and the row-blocked private histograms, so "
            "it is documented slower by design: `histogram.mojo` calls it a "
            "correctness instrument and says a timing taken under it is not a "
            "timing of this package's CPU path"
        ),
    },
}


# ---------------------------------------------------------------------------
# Combinations that cannot run. A cell that cannot run is a DECLARED SKIP with
# a reason, never a hole, which is the rule `CATBOOST_SCENARIO_SUPPORT` and
# `MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT` already keep.
# ---------------------------------------------------------------------------
UNREACHABLE = {
    "base_a_on_gpu": (
        "Base A does not run on the accelerator. Two independent refusals, "
        "either of which is sufficient: `model.fit` raises on "
        "`bootstrap_type` beside a GPU fit, and `train_gpu` raises on "
        "`grow_policy=oblivious` when `ExtraTreeParams.is_active()`, which "
        "`score_function=Cosine` makes true. So every Base A arm is CPU and "
        "the frontier has no symmetric-tree accelerator row at all"
    ),
    "float64_on_gpu": (
        "`derivative_precision=float64` is refused on the accelerator by "
        "`histogram.check_device_derivative_precision`: gradients and "
        "hessians reach the device as Float32 and there is no Float64 there, "
        "so the fit would silently produce the float32 answer. The refusal "
        "covers the environment variable too, which is exactly how this axis "
        "is set, so a GPU cell with the variable left over from a CPU cell "
        "is refused rather than quietly ignored"
    ),
    "mvs_on_gpu": (
        "`bootstrap_type` is not implemented on the GPU (`model.fit`), so "
        "Base B's two MVS arms have no accelerator row. Only the "
        "`subsample=1.0` point of that axis exists on the device, and it is "
        "Base B's base point rather than a variation"
    ),
    "symmetric_categorical": (
        "`grow_policy=oblivious` is implemented for numerical thresholds "
        "only: a level of an oblivious tree shares one split across every "
        "node on it, while a categorical feature is searched as category "
        "partitions whose order comes from one node's own statistics, so "
        "there is no single partition a level could share "
        "(MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT["
        "'high_cardinality_categorical']). Separately, `score_function="
        "Cosine` refuses categorical features outright, because a category "
        "set is searched and scored with the L2 gain and only its winner "
        "reaches the numerical scan. **A third refusal joined these two with "
        "the mode-defaults merge**: the mode now supplies `random_strength = "
        "1.0` on the CPU, and `split.find_best_split` refuses a noised scan "
        "beside any SEARCHABLE categorical feature, because noising one "
        "winner per categorical column while every numerical candidate is "
        "noised is a different regularizer wearing the same name. So the "
        "whole of Base A skips the categorical row, on three unconditional "
        "refusals rather than two, and the categorical row measures Base B "
        "alone"
    ),
    "named_l2_pins_the_rate": (
        "NOT a refusal by the trainer and not a skip that removes an arm: a "
        "trap for whoever edits Base A. `lambda_l2=3` TYPED beside the "
        "symmetric mode closes CatBoost's derivation gate "
        "(options_helper.cpp:280) and pins the rate back to the constant "
        "0.03, recorded as `auto_lr_skipped:l2_leaf_reg`, while the arm's "
        "label still says `@auto`. The mode supplies the same 3.0 with the "
        "provenance flag down and the gate open, so the correct action is to "
        "name nothing. Kept in this table because the wrong version of it "
        "cost this lane a finding: see RESOLVED_SINCE"
    ),
    "mvs_on_multiclass_and_sparse": (
        "not reached by this frontier and stated so that adding a scenario "
        "does not quietly add a broken cell: `trainset."
        "train_dataset_multiclass` takes no bootstrap bundle and the sparse "
        "arm refuses it by name (`sampling.check_bootstrap_honored`), so the "
        "subsample axis has no meaning on `multiclass` or `sparse_highdim`. "
        "Neither scenario is in SCENARIO_ROWS"
    ),
    "auto_rate_on_ranking": (
        "likewise not reached and likewise stated: CatBoost's coefficient "
        "table has rows for Logloss, MultiClass and RMSE target types only, "
        "so a lambdarank fit leaves the rate alone. An explicit "
        "`auto_learning_rate=True` on `ranking` is REFUSED by name; an "
        "inherited one declines in silence, which is CatBoost's own "
        "behavior. `ranking` is not in SCENARIO_ROWS"
    ),
    "competitors_on_gpu": (
        "both competitors are CPU-only in this harness and `run.py` already "
        "declares the skips: LightGBM because a cpu-vs-gpu row would be a "
        "mojotrees-internal comparison, and CatBoost because its GPU "
        "training is a different quantization (border_count capped at 255 "
        "against 65535) and so is not the same measurement"
    ),
    "sparse_symmetric": (
        "there is no sparse oblivious grower: `tree_sparse.grow_tree_sparse` "
        "builds a `GrowthSchedule` and `GrowthSchedule.__init__` refuses "
        "`GROW_OBLIVIOUS` rather than treating it as depth-wise, because a "
        "level search is not a frontier order. No sparse scenario is in "
        "SCENARIO_ROWS, so this frontier never asks; it is declared because "
        "the axis list says 'both grow policies' and a reader should not "
        "infer that both reach everywhere"
    ),
}


#: The ONE interaction this design will look at, and only on evidence.
#:
#: Conditional by construction and NOT counted in the baseline cells below.
#: It exists because `max_bin` and trees are the pair most likely to interact
#: -- more rounds can recover accuracy a coarser split lost, which would make
#: a cheap-bin arm affordable at a tree count where the one-axis sweep says it
#: is not -- and because one 3x3 corner is affordable where a grid is not.
INTERACTION_CHECK = {
    "pair": ("max_bin", "trees"),
    "scenario_row": "dense_synthetic",
    "device": "gpu",
    "base": "B",
    "grid": {"max_bin": (255, 63), "trees": (100, 1000)},
    "already_measured": 2,
    "new_arms": 2,
    "trigger": (
        "run it only if the one-axis results suggest one, which means: the "
        "max_bin=63 arm lands inside the budget at the base tree count AND "
        "the 1000-tree arm's accuracy gain is large enough that a coarser bin "
        "could be paid for out of it. If either half is absent there is no "
        "interaction to find and the two extra arms measure nothing"
    ),
    "what_it_still_cannot_see": (
        "one pair, one scenario, one backend, one base. After it runs, the "
        "number of measured interactions in this frontier is one and the "
        "number of unmeasured ones is every other pair. It must never be "
        "reported as making this a grid"
    ),
}


# ---------------------------------------------------------------------------
# Arm construction.
# ---------------------------------------------------------------------------


def _block_of(base_id, axis, n_estimators, rate):
    """Which block this arm belongs to.

    Only an arm that differs from its base in NOTHING but the tree count can
    be a defaults or a matched row. A `max_bin=63` arm at 360 trees is not
    the shipped configuration and must not appear in the defaults block just
    because it shares its tree count: it is an axis row, and it is ranked
    against the other rows at 360 and nowhere else.

    The shipped default wins over the matched count where a base could claim
    both. Neither base can: Base A's are 360 and 100, Base B's are 72 and
    100.
    """
    if axis not in ("base", "trees"):
        return "axis"
    # The rate has to be the base's own too. Base B at 100 trees with a
    # DERIVED rate is 100 trees of something LightGBM never runs, so it is an
    # axis row and not the matched comparison, even though it shares
    # LightGBM's count.
    if rate != BASES[base_id]["learning_rate"]:
        return "axis"
    policy = "symmetrictree" if base_id == "A" else "lossguide"
    if int(n_estimators) == int(SHIPPED_N_ESTIMATORS[policy]):
        return "defaults"
    if int(n_estimators) == int(MATCHED_TREES[base_id]):
        return "matched"
    return "axis"


def _arm(row, base_id, device, axis, value, params, dataset_params, env,
         n_estimators, learning_rate, skip=None):
    # THE DEVICE IS NOT IN THE ARM ID, since 2026-08-17. It was, and that made
    # every cpu/gpu twin pairing in the harness impossible under `--arms
    # frontier`: `run._oracle_key`, `verify._oracle_cell_key`,
    # `verify.check_device_agreement`, `verify.check_backend_proof` and
    # `summarize.build_device_agreement` all pair by arm id WITHOUT the
    # device, so a cpu cell named `...baseB.cpu...` and a gpu cell named
    # `...baseB.gpu...` were two arms, no cpu cell was ever an oracle (351
    # jobs, zero oracle cells), every base cpu cell ran the full repeat count
    # and was ranked in the speed story, and every gpu row was "an
    # accelerator row with no cpu twin". The arm is the CONFIGURATION; the
    # device is the backend it ran on and lives in the job's `device` field
    # and in `run.label`, which appends it, so filenames stay unique. No
    # recorded id changes, because the frontier had never run.
    parts = [FRONTIER_ID, row["row"], f"base{base_id}", axis, str(value)]
    return {
        "block": _block_of(base_id, axis, n_estimators, learning_rate),
        "id": ".".join(parts),
        "frontier": FRONTIER_ID,
        "row": row["row"],
        "scenario": row["scenario"],
        "tier": row["tier"],
        "variant": row["variant"],
        "base": base_id,
        "engine": BASES[base_id]["engine"] if base_id in BASES else base_id,
        "device": device,
        "axis": axis,
        "axis_value": value,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "params": params,
        "dataset_params": dataset_params,
        "env": env,
        "repeats": REPEATS,
        "skip": skip,
    }


def _base_point(base):
    """The base's own (params, dataset_params, env) as three fresh dicts."""
    return (
        dict(base["params"]),
        dict(base["dataset_params"]),
        dict(base["env"]),
    )


def _apply_trees(base, params, n_estimators, rate):
    params["n_estimators"] = int(n_estimators)
    if rate == "auto":
        params["auto_learning_rate"] = True
        params.pop("learning_rate", None)
    else:
        params["auto_learning_rate"] = False
        params["learning_rate"] = float(rate)
    return params


def _apply_subsample(params, fraction, kind):
    if kind is None:
        params.pop("bootstrap_type", None)
        params.pop("subsample", None)
    else:
        params["bootstrap_type"] = kind
        params["subsample"] = float(fraction)
    return params


#: What the trainer can do, at the head this file was last checked against.
#:
#: Both were FALSE until 2026-08-17 and are flags rather than sentences so
#: that the plan can be computed under either answer. Andrew has ruled that
#: the frontier does not run until a symmetric-tree accelerator row can exist,
#: so the plan has two cell counts, what it is at this head and what it is
#: when both capabilities hold, and both are computed from this dict rather
#: than estimated. At head the two dicts now agree. Re-check them after any
#: rebase; `CAPABILITIES_CHECKED_AT` says against what.
CAPABILITIES_AT_HEAD = {
    # Both flipped to True on 2026-08-17 after reading the trainer at head,
    # not the branch list. gpu_bootstrap: `model.mojo::fit` now passes the
    # bundle to `train_gpu` by keyword (`bootstrap=bootstrap`) instead of
    # raising, and `train_gpu.mojo` draws it per round through
    # `sampling.bootstrap_round` (MVS, host-gradient arm) or the device weight
    # plane (Bayesian); the one refusal left is an explicit
    # `objective_source=device` beside MVS, which no Base A arm sets. The
    # multiclass path (`model.mojo::fit_multiclass`) still refuses, and no
    # frontier row is multiclass. gpu_oblivious_cosine:
    # `gpu_resident_round.mojo::oblivious_device_supported` and
    # `train_gpu.mojo::_check_device_search_supported` both read
    # `tree_parameters_extra.mojo::ExtraTreeParams.device_unsupported_reason`
    # rather than `is_active()`, and that function refuses `score_function`
    # and `random_strength` ONLY beside a categorical column (Base A is
    # already skipped on the categorical rows for a different reason).
    #
    # "No longer refused" is not "runs". Neither flag was verified by a fit;
    # both were verified by reading the gates, and this repository has one
    # recorded case (the same function's own comment) of a gate retired on a
    # correct reading while the device applied nothing. So the FIRST --arms
    # run must be read for Base A's GPU leg specifically: its device_agreement
    # verdict against the cpu twin, and its backend proof, before any Base A
    # GPU timing is quoted.
    "gpu_bootstrap": True,
    "gpu_oblivious_cosine": True,
}

CAPABILITIES_CHECKED_AT = (
    "2026-08-17, working tree at c775959 plus uncommitted lane work, by "
    "reading src/mojotrees: `model.mojo::fit` hands `bootstrap=` to "
    "`train_gpu` and `train_gpu.mojo` draws it (`bootstrap_round`, "
    "`refresh_bayesian_bootstrap`, `gpu_bootstrap_resolution`); "
    "`gpu_resident_round.mojo::oblivious_device_supported` reads "
    "`ExtraTreeParams.device_unsupported_reason`, which excludes "
    "score_function and random_strength by name except beside a categorical "
    "column. Not verified by a fit; see the comment on the dict. The previous "
    "entry, 1624647 on 2026-08-16, recorded both as refused"
)

#: Everything true when both merges land. Not a prediction about when.
CAPABILITIES_WHEN_MERGED = {
    "gpu_bootstrap": True,
    "gpu_oblivious_cosine": True,
}


def _skip_for(base_id, device, axis, value, row, caps=None):
    """The declared reason this arm cannot run, or None.

    Every branch names a `UNREACHABLE` key rather than writing a second copy
    of the sentence, so a reason that changes changes in one place. `caps`
    decides which of the two accelerator refusals are live, so the same
    enumeration answers both "what runs today" and "what runs once the two
    merges land" without a second copy of the arm list.
    """
    caps = CAPABILITIES_AT_HEAD if caps is None else caps
    if base_id == "A" and row["scenario"] in (
        "high_cardinality_categorical", "categorical_missing"
    ):
        # First, and unconditional: this one is not waiting on anything.
        return UNREACHABLE["symmetric_categorical"]
    if device == "gpu" and axis == "precision" and value == "float64":
        # Also unconditional. The device has no Float64 to carry a derivative
        # in, which is not a merge away.
        return UNREACHABLE["float64_on_gpu"]
    if base_id == "A" and device == "gpu":
        missing = [
            name for name in ("gpu_bootstrap", "gpu_oblivious_cosine")
            if not caps[name]
        ]
        if missing:
            return UNREACHABLE["base_a_on_gpu"]
        return None
    if device == "gpu" and axis == "subsample" and value[1] == "MVS":
        if not caps["gpu_bootstrap"]:
            return UNREACHABLE["mvs_on_gpu"]
        return None
    return None


def base_arms(row, base_id, caps=None):
    """Every arm of one base on one scenario row, base point included once.

    One axis at a time: each arm below differs from the base in exactly one
    of the four axes, and a value equal to the base's own value produces the
    base point rather than a duplicate arm.
    """
    base = BASES[base_id]
    out = []
    for device in base["devices"]:
        if device not in row["devices"]:
            # Not a frontier decision: the scenario declares its backends and
            # `run.py` already skips a device a scenario does not support.
            continue
        seen_base_point = False

        for n_estimators, rate in AXES["trees"][base_id]:
            params, dataset_params, env = _base_point(base)
            _apply_trees(base, params, n_estimators, rate)
            is_base = (
                int(n_estimators) == int(base["params"]["n_estimators"])
                and rate == base["learning_rate"]
            )
            axis = "base" if is_base else "trees"
            if is_base:
                if seen_base_point:
                    continue
                seen_base_point = True
            value = f"{n_estimators}@{rate}"
            out.append(
                _arm(
                    row, base_id, device, axis, value, params, dataset_params,
                    env, int(n_estimators), rate,
                    _skip_for(base_id, device, axis, value, row, caps),
                )
            )

        for fraction, kind in AXES["subsample"][base_id]:
            base_kind = base["params"].get("bootstrap_type")
            base_fraction = base["params"].get("subsample", 1.0)
            if kind == base_kind and float(fraction) == float(base_fraction):
                continue
            params, dataset_params, env = _base_point(base)
            _apply_subsample(params, fraction, kind)
            value = (fraction, kind)
            out.append(
                _arm(
                    row, base_id, device, "subsample",
                    f"{fraction}{'' if kind is None else '-' + kind}",
                    params, dataset_params, env,
                    int(base["params"]["n_estimators"]), base["learning_rate"],
                    _skip_for(base_id, device, "subsample", value, row, caps),
                )
            )

        for max_bin in AXES["max_bin"][base_id]:
            if int(max_bin) == int(base["dataset_params"]["max_bin"]):
                continue
            params, dataset_params, env = _base_point(base)
            dataset_params["max_bin"] = int(max_bin)
            out.append(
                _arm(
                    row, base_id, device, "max_bin", max_bin, params,
                    dataset_params, env,
                    int(base["params"]["n_estimators"]), base["learning_rate"],
                    _skip_for(base_id, device, "max_bin", max_bin, row, caps),
                )
            )

        for precision in AXES["precision"][base_id]:
            if precision == base["env"]["MOJOTREES_DERIVATIVE_PRECISION"]:
                continue
            params, dataset_params, env = _base_point(base)
            env["MOJOTREES_DERIVATIVE_PRECISION"] = precision
            out.append(
                _arm(
                    row, base_id, device, "precision", precision, params,
                    dataset_params, env,
                    int(base["params"]["n_estimators"]), base["learning_rate"],
                    _skip_for(base_id, device, "precision", precision, row, caps),
                )
            )
    return out


def competitor_arms(row):
    """CatBoost as shipped and LightGBM stock+det, at every tree count any
    mojotrees arm uses.

    They exist so that the inside-1-percent SCOREBOARD line has something to
    stand on AT A MATCHED TREE COUNT. A competitor row missing at a tree count
    is not a smaller comparison, it is no comparison:
    `verify.check_accuracy_peer` abstains rather than substituting another
    count. That comparison stopped gating anything on 2026-08-17; it is still
    reported on every run and it is still worth running these rows for.
    """
    out = []
    for engine in ("catboost", "lightgbm"):
        supported = scenarios.CATBOOST_SCENARIO_SUPPORT.get(row["scenario"])
        for trees in COMPETITOR_TREES:
            skip = None
            if engine == "catboost" and supported is not None:
                skip = supported
            if engine == "catboost":
                ok, reason = scenarios.catboost_tier_ok(
                    scenarios.resolve(row["scenario"], row["tier"], "auto"),
                    row["tier"],
                )
                if not ok and skip is None:
                    skip = reason
            out.append(
                {
                    # A competitor is the bar, not a candidate: it is never
                    # ranked and never recommended. It appears in the table
                    # for its own tree count and in no other.
                    "block": "competitor",
                    "id": f"{FRONTIER_ID}.{row['row']}.{engine}.cpu.trees.{trees}",
                    "frontier": FRONTIER_ID,
                    "row": row["row"],
                    "scenario": row["scenario"],
                    "tier": row["tier"],
                    "variant": row["variant"],
                    "base": "competitor",
                    "engine": engine,
                    "device": "cpu",
                    "axis": "trees",
                    "axis_value": trees,
                    "n_estimators": trees,
                    "learning_rate": "engine default at this tree count",
                    "params": {"n_estimators": trees},
                    "dataset_params": {},
                    "env": {},
                    "repeats": REPEATS,
                    "skip": skip,
                    "record_as_provenance": (
                        "resolved boosting_type, learning_rate and "
                        "border_count off the engine's own read-back. "
                        "PROVENANCE, not a warning: all three CatBoost rows "
                        "are Plain on the CPU. `defaults_helper.h:33-42` "
                        "hard-sets Plain when the option is unset and "
                        "(rows >= 50000 or iterations < 500), and when that "
                        "clause does not fire the option is merely NotSet, "
                        "which `boosting_options.cpp:16` constructs as Plain. "
                        "The only site that installs Ordered as a default "
                        "(`catboost_options.cpp:806`) is guarded by "
                        "TaskType == GPU and cannot fire here. A reader "
                        "should not have to derive that, which is why it is "
                        "recorded"
                    ),
                }
            )
    return out


def arms(caps=None):
    """Every arm of the frontier, runnable and skipped alike, in run order.

    Run order is the window order: one scenario row per window, dense
    synthetic first, real second, categorical third, which is `SCENARIO_ROWS`
    as written. Andrew ruled that on 2026-08-16 and it costs nothing, because
    the rows are independent and round-interleaving never spanned them.
    """
    out = []
    for row in SCENARIO_ROWS:
        for base_id in ("A", "B"):
            out.extend(base_arms(row, base_id, caps))
        out.extend(competitor_arms(row))
    return out


# ---------------------------------------------------------------------------
# Wall clock. Estimated from timings ALREADY RECORDED, never from a new one.
# ---------------------------------------------------------------------------
#
# Every number below is a median train second at 100 trees, read out of
# `bench/results/COMPARISON_RUN_2026-08-16.md`: Apple M4, 10 threads, five
# repeats, ON BATTERY, at commit 9de2ad4. That provenance is the estimate's
# main weakness and it is stated rather than buried: a battery-powered median
# is a fact about an afternoon, and this repository has measured the same
# benchmark drifting two to three times across time windows.
MEASURED_TRAIN_S_AT_100 = {
    ("dense_synthetic", "A"): 14.427,
    ("dense_synthetic", "B", "cpu"): 10.077,
    ("dense_synthetic", "B", "gpu"): 3.619,
    ("dense_synthetic", "lightgbm"): 4.749,
    ("dense_synthetic", "catboost"): 3.755,
    ("dense_real", "A"): 7.537,
    ("dense_real", "B", "cpu"): 4.975,
    ("dense_real", "B", "gpu"): 3.332,
    ("dense_real", "lightgbm"): 2.572,
    ("dense_real", "catboost"): 1.854,
    ("categorical", "B", "cpu"): 2.437,
    ("categorical", "lightgbm"): 5.621,
    ("categorical", "catboost"): 3.686,
}

#: The Base A proxy, named because it is a proxy and not a measurement of this
#: arm. `us in CatBoost's shape, CPU` in that report is symmetric depth 6 with
#: Cosine, MVS 0.8, random_strength and `lambda_l2 = 3`, at 100 trees and a
#: PINNED rate. Base A is the same shape without `random_strength` and at
#: `lambda_l2 = 0`, so the estimate inherits the shape's cost and not its
#: exact configuration.
BASE_A_PROXY = (
    "`us in CatBoost's shape, CPU` from COMPARISON_RUN_2026-08-16.md. Same "
    "grower, same score function, same bootstrap; different lambda_l2 and no "
    "random_strength. A proxy, not a measurement of Base A"
)

#: Per-cell cost that is not training: process start, extension import, data
#: build or load, binning, three prediction repeats, the digest and the
#: record. UNMEASURED. The plan prints three columns rather than one number
#: because the honest form of an unmeasured quantity is a range, and because
#: 400 cells multiply it: every ten seconds here is over an hour of run.
FIXED_PER_CELL_S = (10.0, 20.0, 40.0)

#: Cost scaling this estimate DOES apply, and the only one.
#:
#: Training time is taken as linear in the tree count, because rounds are the
#: outer loop and each one grows one tree. Nothing else is scaled: a
#: `max_bin=63` arm and a `float64` arm are both costed at their base's time,
#: which is wrong in a known direction for both (coarser bins are cheaper,
#: float64 is documented slower because it turns off the gathered pair buffer
#: and the blocked histograms). They are not scaled because the size of those
#: effects is exactly what the run exists to measure, and an estimate that
#: guessed them would be reporting its guess back as a result.
TREE_SCALING = "linear in n_estimators; nothing else is scaled"


def _train_seconds(arm):
    key_engine = arm["base"]
    if key_engine == "competitor":
        base = MEASURED_TRAIN_S_AT_100.get((arm["row"], arm["engine"]))
    elif key_engine == "A":
        base = MEASURED_TRAIN_S_AT_100.get((arm["row"], "A"))
    else:
        base = MEASURED_TRAIN_S_AT_100.get((arm["row"], "B", arm["device"]))
    if base is None:
        return None
    return base * (arm["n_estimators"] / 100.0)


def estimate(all_arms=None):
    """Cells and wall clock, per scenario row and in total.

    A cell is one whole process: one arm, one repeat. Skipped arms cost
    nothing and are counted separately, because a declared skip is a fact
    about the design and not a unit of work.
    """
    all_arms = list(all_arms if all_arms is not None else arms())
    rows = {}
    for arm in all_arms:
        bucket = rows.setdefault(
            arm["row"],
            {"arms": 0, "skipped": 0, "cells": 0, "train_s": 0.0, "unpriced": 0},
        )
        if arm["skip"]:
            bucket["skipped"] += 1
            continue
        bucket["arms"] += 1
        bucket["cells"] += REPEATS
        seconds = _train_seconds(arm)
        if seconds is None:
            bucket["unpriced"] += 1
        else:
            bucket["train_s"] += seconds * REPEATS

    total = {"arms": 0, "skipped": 0, "cells": 0, "train_s": 0.0, "unpriced": 0}
    for bucket in rows.values():
        for key in total:
            total[key] += bucket[key]

    def wall(bucket):
        return {
            f"fixed_{int(fixed)}s": bucket["train_s"] + fixed * bucket["cells"]
            for fixed in FIXED_PER_CELL_S
        }

    for bucket in rows.values():
        bucket["wall_s"] = wall(bucket)
    total["wall_s"] = wall(total)
    return {"rows": rows, "total": total}


def projected():
    """The plan under both capability dicts, computed not guessed.

    **This is the count Andrew will actually run**, because the frontier is
    sequenced not to start until a symmetric-tree accelerator row can exist.
    The arms are the same arms: only `CAPABILITIES_AT_HEAD` moves, so the
    difference between the two counts is exactly the set of cells the two
    capabilities unblock and nothing else. Since 2026-08-17 the two dicts
    agree at head and the difference is zero; the function is kept so that
    a rebase which re-refuses either capability shows up here as a count.

    Note what does NOT come back. Base A on the categorical row stays skipped
    on three unconditional refusals, and `float64` on the accelerator stays
    skipped because the device has no Float64 to carry a derivative in.
    Neither is a merge away.
    """
    head = estimate(arms(CAPABILITIES_AT_HEAD))["total"]
    merged = estimate(arms(CAPABILITIES_WHEN_MERGED))["total"]
    return {
        "at_head": {
            "arms": head["arms"], "cells": head["cells"],
            "skipped": head["skipped"],
            "wall_s": head["wall_s"],
        },
        "when_merged": {
            "arms": merged["arms"], "cells": merged["cells"],
            "skipped": merged["skipped"],
            "wall_s": merged["wall_s"],
            "unpriced": merged["unpriced"],
        },
        "unblocked_arms": merged["arms"] - head["arms"],
        "still_skipped": merged["skipped"],
        "waits_on": (
            "nothing at head, since 2026-08-17: bootstrap_type on the "
            "accelerator and score_function=Cosine under oblivious growth on "
            "the device are both unrefused (CAPABILITIES_CHECKED_AT says what "
            "was read). Until then Base A needed both and Base B's two MVS "
            "arms needed the first alone, which is why the two dicts exist"
        ),
        "unpriced_warning": (
            "Base A on the accelerator is priced by `_train_seconds` at the "
            "Base A CPU proxy, because MEASURED_TRAIN_S_AT_100 has no "
            "accelerator entry for a symmetric arm: it has never run. So that "
            "leg's wall clock is a placeholder, and probably an over-estimate, "
            "since a device symmetric grower exists to be faster than the "
            "slowest CPU arm in the plan. Corrected 2026-08-17; this used to "
            "say the arms were counted and not priced, which `_train_seconds` "
            "never did"
        ),
    }


def levers(all_arms=None):
    """What each way of making this cheaper would actually save.

    The target Andrew named is on the order of an hour or two per scenario at
    five repeats, and the plan above does not meet it on the dense synthetic
    row. **Nothing here is applied.** The design is reported at its real cost
    and these are the prices of the obvious cuts, so that a trim is his
    decision with a number attached rather than this lane's quiet edit.
    """
    all_arms = list(all_arms if all_arms is not None else arms())
    full = estimate(all_arms)["total"]["wall_s"]["fixed_20s"]

    def without(predicate):
        kept = [a for a in all_arms if not predicate(a)]
        return estimate(kept)["total"]["wall_s"]["fixed_20s"]

    thousand = without(lambda a: a["n_estimators"] == 1000)
    # Repeats scale everything linearly, the fixed per-cell cost and the
    # training seconds alike, because a repeat is a whole process.
    three = full * 3.0 / REPEATS
    return {
        "full_plan_s": full,
        "options": {
            "drop_the_1000_tree_arms": {
                "wall_s": thousand,
                "saves_fraction": 1.0 - thousand / full if full else 0.0,
                "costs": (
                    "the far end of the trees axis. 1000 trees is where a "
                    "derived rate is smallest (CatBoost resolves 0.06573 at "
                    "1000 against 0.4273 at 100 on this machine) and it is "
                    "the only point that says whether the budget can be "
                    "bought back by spending rounds. Dropping it makes the "
                    "trees axis two points wide for Base A"
                ),
            },
            "three_repeats_instead_of_five": {
                "wall_s": three,
                "saves_fraction": 1.0 - 3.0 / REPEATS,
                "costs": (
                    "three is the minimum that shows a spread and five is "
                    "what COMPARISON_RUN_2026-08-16.md ran. This repository "
                    "has resolved margins by disjoint ranges rather than by "
                    "medians, and three repeats makes a range easier to "
                    "overlap, so a 10 percent effect that five repeats would "
                    "resolve may come back indistinguishable"
                ),
            },
            "one_scenario_row_at_a_time": {
                "wall_s": None,
                "saves_fraction": None,
                "costs": (
                    "nothing at all, and it is the cut with no cost: the "
                    "three rows are independent and are already ordered. "
                    "Run dense_synthetic in one window and dense_real in the "
                    "next. The only thing it forfeits is round-interleaving "
                    "ACROSS rows, which was never available anyway, and the "
                    "protocol's own canary already treats separate windows "
                    "as separate measurements"
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# What has to change outside this lane before any of this can be scheduled.
# ---------------------------------------------------------------------------
#: **A separate lane is building all six of these.** They are recorded here
#: because the plan cannot be scheduled without them and a reader has to know
#: what "not runnable yet" consists of; they are NOT this lane's to build, and
#: this lane is not building them. When they land, the arms become
#: schedulable and the count in this file is reported again.
WIRE_NOTES = {
    "arm_dimension": {
        "file": "bench/real_data/run.py, worker.py",
        "note": (
            "the harness's job identity is (scenario, engine, device, "
            "threads, repeat) and has no arm dimension, so no arm in this "
            "file can be executed today. What it needs is small and is "
            "designed for: each arm carries `params`, `dataset_params` and "
            "`env`, and `scenarios.mojotrees_params(spec, device, extra)` "
            "already takes an `extra` dict. The record must then carry `arm` "
            "and `axis`, which `verify.py` and `report.py` read here through "
            "`record.get('arm')` with a fallback to `engine`, so both files "
            "work before and after the change"
        ),
        "owner": "not this lane",
    },
    "n_estimators_from_the_arm": {
        "file": "bench/real_data/engines.py",
        "note": (
            "`MojoTreesEngine._run_dense` and `_run_sparse` read "
            "`scenarios.BASE_PARAMS['n_estimators']` directly, and "
            "`LightGBMEngine` does the same, so the trees axis cannot move "
            "without them. This is the one place the tree count is decided "
            "and it currently cannot be overridden"
        ),
        "owner": "not this lane",
    },
    "per_cell_dataset_params": {
        "file": "bench/real_data/engines.py, scenarios.py",
        "note": (
            "`dataset_params(spec)` takes no engine and no arm, so `max_bin` "
            "is one value for the whole run. The max_bin axis needs it to "
            "take an override. Note that the axis changes the BINNING and not "
            "the data, so the canonical data digest is unaffected and the "
            "data-agreement gate stays meaningful across arms"
        ),
        "owner": "not this lane",
    },
    "per_cell_env": {
        "file": "bench/real_data/run.py, python/mojotrees/sklearn.py",
        "note": (
            "`derivative_precision` has no Python parameter at this head: "
            "`sklearn.py` has no such keyword and the only entry is "
            "`MOJOTREES_DERIVATIVE_PRECISION`, read by "
            "`histogram.derivative_precision_narrows`. **A separate lane is "
            "making it a real keyword**, which is the better fix and removes "
            "the per-cell environment question rather than answering it. "
            "Whichever way it lands, a GPU cell must not inherit a float64 "
            "setting from a CPU cell: "
            "`check_device_derivative_precision` refuses a leftover one by "
            "name, which is the behavior to keep"
        ),
        "owner": "not this lane, and a lane is on it",
    },
    "readback_key_has_no_tree_count": {
        "file": "bench/real_data/scenarios.py",
        "note": (
            "`catboost_readback_key` is (scenario, tier, variant). The "
            "frontier runs CatBoost at three tree counts on one cell, so all "
            "three would write the same key and the last would win. Harmless "
            "while nothing reads it -- no arm here takes the read-back route "
            "-- and blocking for exit (2) of "
            "UNREACHABLE['l2_3_with_auto_rate']"
        ),
        "owner": "not this lane",
    },
}

#: Findings this file made and had to withdraw, kept rather than deleted.
#:
#: A frontier that is honest about its gaps has to be honest about its
#: mistakes in the same file, and a retracted claim that leaves no trace gets
#: re-derived by the next reader from the same stale tree.
RESOLVED_SINCE = {
    "l2_3_with_auto_rate": {
        "claimed": (
            "that `lambda_l2 = 3` and a derived learning rate could not both "
            "be asked for, so Base A would run at LightGBM's 0.0; and, as the "
            "sharpest consequence, that Cosine at `reg_lambda = 0` collapses "
            "onto `sqrt` of the L2 score and Base A's Cosine axis therefore "
            "sat near its degenerate point"
        ),
        "wrong_because": (
            "the mode-defaults layer merged at `ad6c2b6`, after the "
            "`bda3b41` this lane branched from. "
            "`params._apply_catboost_mode_defaults` and, on the surface this "
            "harness actually trains through, `_Base._params`'s `l2_default` "
            "supply `reg_lambda = 3.0` under `symmetrictree` with "
            "`TOption::SetDefault` semantics: assigned, provenance flag left "
            "down. Two of the four keys CatBoost's gate reads are among the "
            "values supplied, so supplying them cannot close it. "
            "Demonstrated by the lane that landed it: a symmetric fit with "
            "nothing named derived 0.29699 and recorded `auto_lr_gate_open` "
            "while carrying `l2_leaf_reg = 3.0`"
        ),
        "now": (
            "Base A runs at `lambda_l2 = 3.0` AND a derived rate, and at "
            "`reg_lambda = 3` the Cosine score function is off its "
            "degenerate point outright. The requirement on the arms is only "
            "that they NAME none of the gate keys, which is "
            "UNREACHABLE['named_l2_pins_the_rate']"
        ),
        "how_it_happened": (
            "a branch merged into the file this lane was reasoning about "
            "while the lane held an older head. The reasoning was sound on "
            "the tree it had. The check that would have caught it is `git "
            "log --oneline -5` before re-deriving anything, which is now the "
            "first line of this lane's rebase routine"
        ),
    },
    "selfcheck_pandas": {
        "claimed": (
            "that `selfcheck.py` had one pre-existing failure, the CatBoost "
            "categorical encoder unable to import pandas, and that this was "
            "an environment gap in the repository"
        ),
        "wrong_because": (
            "it was the INTERPRETER. `pixi run -e bench python "
            "bench/real_data/selfcheck.py` passes; bare `python3` does not, "
            "because pandas lives in the bench feature. The stash test this "
            "lane ran was sound and proved what it could -- the failure was "
            "not this lane's change -- and the step not taken was checking "
            "whether the dependency existed at all under the right python"
        ),
        "now": (
            "selfcheck prints `sys.executable` beside any import-shaped "
            "failure as of `1624647`, so the next reader is told which "
            "python could not import rather than left to conclude the "
            "repository is broken. **Run it under the bench environment**"
        ),
    },
}


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def check(all_arms=None):
    """The invariants of the plan, raised rather than printed.

    Run from `main` on every invocation, so a plan that cannot be read
    correctly cannot be printed at all. Static: it imports no trainer, reads
    no results file and takes no lock.

    The first invariant is the one that matters. **Every arm's tree count must
    be a count the competitor rows run at**, because an arm at a count with no
    competitor row cannot be judged, and an unjudged arm in a frontier is a
    speed number with nothing holding it to an accuracy. This is the check
    that would have caught the lossguide-72 row going in without a 72-tree
    competitor beside it.
    """
    problems = []
    all_arms = list(all_arms if all_arms is not None else arms())
    reasons = set(UNREACHABLE.values())
    for arm in all_arms:
        if arm["skip"]:
            supported = scenarios.CATBOOST_SCENARIO_SUPPORT.get(arm["scenario"])
            known = arm["skip"] in reasons or arm["skip"] == supported
            if not known:
                problems.append(
                    f"{arm['id']}: skip reason is not one of UNREACHABLE's, "
                    "so it is a sentence written at the skip site rather than "
                    "a declared rule"
                )
            continue
        if arm["n_estimators"] not in COMPETITOR_TREES:
            problems.append(
                f"{arm['id']}: {arm['n_estimators']} trees, and no competitor "
                f"row runs at that count ({COMPETITOR_TREES}). This arm would "
                "abstain and carry no accuracy verdict at all"
            )
        if arm["block"] not in BLOCKS and arm["block"] != "competitor":
            problems.append(f"{arm['id']}: unknown block {arm['block']!r}")
    for policy, count in SHIPPED_N_ESTIMATORS.items():
        if count not in COMPETITOR_TREES:
            problems.append(
                f"the shipped {policy} default is {count} trees and no "
                "competitor row runs there, so what we ship would be the one "
                "row in the frontier with no verdict"
            )
    if problems:
        raise AssertionError("frontier plan is inconsistent:\n  " + "\n  ".join(problems))
    return True


def plan():
    all_arms = arms()
    check(all_arms)
    return {
        "frontier": FRONTIER_ID,
        "frontier_version": FRONTIER_VERSION,
        "registered": FRONTIER_REGISTERED,
        "repeats": REPEATS,
        "shipped_n_estimators": dict(SHIPPED_N_ESTIMATORS),
        "blocks": dict(BLOCKS),
        "matched_trees": dict(MATCHED_TREES),
        "competitor_trees": list(COMPETITOR_TREES),
        "scenario_rows": [dict(row) for row in SCENARIO_ROWS],
        "categorical_alternative": dict(CATEGORICAL_ALTERNATIVE),
        "bases": {k: dict(v) for k, v in BASES.items()},
        "axes": {k: dict(v) for k, v in AXES.items()},
        "unreachable": dict(UNREACHABLE),
        "interaction_check": dict(INTERACTION_CHECK),
        "wire_notes": dict(WIRE_NOTES),
        "capabilities_at_head": dict(CAPABILITIES_AT_HEAD),
        "capabilities_checked_at": CAPABILITIES_CHECKED_AT,
        "resolved_since": dict(RESOLVED_SINCE),
        "arms": all_arms,
        "estimate": estimate(all_arms),
        "when_gpu_symmetric_lands": projected(),
        "levers": levers(all_arms),
        "cannot_answer": [
            "one axis at a time from two bases is NOT a grid: no interaction "
            "between any two axes is measured, except the one conditional "
            "pair in INTERACTION_CHECK",
            "the fastest point named is the fastest AMONG THESE ARMS, which "
            "is weaker than the fastest configuration: a better point two "
            "axes away from both bases is invisible here",
            "the symmetric-tree accelerator row is UNREFUSED at head "
            "(2026-08-17, CAPABILITIES_AT_HEAD) and has never been run, so "
            "'both grow policies on both backends' is a question this plan "
            "now schedules and has not yet answered; the first run's Base A "
            "GPU leg is read for its device_agreement verdict and backend "
            "proof before its timing counts",
            "the symmetric-policy accelerator arms are priced by the Base A "
            "proxy at CPU cost (`_train_seconds`), because no symmetric GPU "
            "timing has ever been recorded here, so the wall clock for that "
            "leg is a placeholder and not an estimate",
            "the wall clock in this plan is estimated from ONE battery-powered "
            "window on one machine, and this repository has measured the "
            "same benchmark drifting two to three times between windows",
        ],
    }


def _fmt_hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m"


def render(payload, out):
    out(f"{FRONTIER_ID} v{FRONTIER_VERSION}: a plan. Nothing here has run.\n")
    shipped = payload["shipped_n_estimators"]
    out(
        f"{payload['repeats']} repeats per arm, competitor rows at "
        f"{', '.join(str(t) for t in payload['competitor_trees'])} trees.\n"
    )
    out(
        "Shipped defaults are per grow policy: symmetrictree at "
        f"{shipped['symmetrictree']} trees, lossguide at "
        f"{shipped['lossguide']}. Both are OURS and neither is a competitor's "
        "count, so each is judged only against competitor rows at its own "
        "count and is otherwise unjudged.\n"
    )

    out("\nARMS AND CELLS\n")
    header = (
        f"{'row':<18} {'arms':>5} {'skipped':>8} {'cells':>7} "
        f"{'train s':>10} {'wall @20s':>10}"
    )
    out(header + "\n")
    out("-" * len(header) + "\n")
    estimated = payload["estimate"]
    for row in payload["scenario_rows"]:
        bucket = estimated["rows"].get(row["row"])
        if not bucket:
            continue
        out(
            f"{row['row']:<18} {bucket['arms']:>5} {bucket['skipped']:>8} "
            f"{bucket['cells']:>7} {bucket['train_s']:>10.0f} "
            f"{_fmt_hms(bucket['wall_s']['fixed_20s']):>10}\n"
        )
    total = estimated["total"]
    out(
        f"{'TOTAL':<18} {total['arms']:>5} {total['skipped']:>8} "
        f"{total['cells']:>7} {total['train_s']:>10.0f} "
        f"{_fmt_hms(total['wall_s']['fixed_20s']):>10}\n"
    )
    out(
        "\nWall clock at three fixed per-cell costs, because that cost is "
        "UNMEASURED:\n"
    )
    for fixed in FIXED_PER_CELL_S:
        out(
            f"  {int(fixed):>3}s per cell -> "
            f"{_fmt_hms(total['wall_s'][f'fixed_{int(fixed)}s'])} total\n"
        )
    out(
        "  Training seconds are medians from COMPARISON_RUN_2026-08-16.md "
        "(Apple M4, 10 threads, ON BATTERY), scaled linearly in the tree "
        "count and in nothing else.\n"
    )

    out(
        "\nTHE TWO CUTS ARE RESOLVED. Andrew ruled on 2026-08-16: three "
        "repeats rather than five, taken, and it is the reason the total "
        "above is what it is; the 1000-tree arms KEPT, declined; one "
        "scenario row per window, which costs nothing because the rows are "
        "independent and round-interleaving never spanned them. Order is "
        "dense synthetic, then real, then categorical, and it is the order "
        "SCENARIO_ROWS is written in.\n"
    )
    out(
        f"  what three repeats cost: a margin here is resolved by disjoint "
        "ranges rather than by medians, and three ranges overlap more "
        "easily than five. A 10 percent effect five repeats would resolve "
        "may come back indistinguishable.\n"
    )
    out(
        f"  if the 1000-tree arms had been dropped instead: "
        f"{_fmt_hms(payload['levers']['options']['drop_the_1000_tree_arms']['wall_s'])}"
        " total. Declined, and not applied.\n"
    )

    projection = payload["when_gpu_symmetric_lands"]
    out(
        "\nWHEN THE TWO ACCELERATOR MERGES LAND, which is what this run is "
        "sequenced behind:\n"
    )
    out(
        f"  {projection['at_head']['arms']} arms / "
        f"{projection['at_head']['cells']} cells today, becoming "
        f"{projection['when_merged']['arms']} arms / "
        f"{projection['when_merged']['cells']} cells, so the two merges "
        f"unblock {projection['unblocked_arms']} arms. "
        f"{projection['still_skipped']} arms stay skipped on refusals no "
        "merge addresses.\n"
    )
    out(f"  waits on: {projection['waits_on']}\n")
    out(f"  {projection['unpriced_warning']}\n")

    out("\nDECLARED SKIPS\n")
    counts = {}
    for arm in payload["arms"]:
        if arm["skip"]:
            counts[arm["skip"]] = counts.get(arm["skip"], 0) + 1
    for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        out(f"  {count:>3} arms: {reason[:150]}\n")

    out("\nWHAT THIS DESIGN CANNOT ANSWER\n")
    for line in payload["cannot_answer"]:
        out(f"  - {line}\n")

    out("\nNOT WIRED (none of it in this lane's files)\n")
    for key, note in payload["wire_notes"].items():
        out(f"  - {key} [{note['file']}]\n")

    out(
        "\nThe interaction check is CONDITIONAL and is not in the counts "
        f"above: {payload['interaction_check']['new_arms']} extra arms on "
        f"{payload['interaction_check']['scenario_row']}/"
        f"{payload['interaction_check']['device']}, only on the trigger.\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the whole plan here")
    parser.add_argument(
        "--skips", action="store_true", help="print only the declared skips"
    )
    args = parser.parse_args(argv)

    payload = plan()
    if args.skips:
        for arm in payload["arms"]:
            if arm["skip"]:
                print(f"{arm['id']}\n    {arm['skip']}")
        return 0
    render(payload, sys.stdout.write)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
