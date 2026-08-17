"""The pair run: three mirror PAIRS and one shipped-versus-shipped class.

    python bench/real_data/pairs.py                  # the plan, as a table
    python bench/real_data/pairs.py --json plan.json # the same, machine-readable
    python bench/real_data/pairs.py --skips          # only the declared skips
    python bench/real_data/run.py --arms pairs --dry-run --device cpu gpu

THIS FILE RUNS NOTHING. It enumerates arms, declares which of them cannot run
and why, counts cells and estimates wall clock from timings already recorded
elsewhere. It imports no trainer, takes no lock and fits nothing.

WHY THIS EXISTS BESIDE `frontier.py`
------------------------------------

`frontier.py` answers "what is the fastest configuration whose accuracy we are
willing to pay for", by moving ONE axis at a time from two base points. That is
a sweep. This file answers two different questions that a sweep cannot, and it
answers them in ONE table so that a reader cannot take a row out of one and
read it as an answer to the other:

**CLASS A, the mirror pairs.** Three pairs. Each pair holds everything constant
except the thing being compared, with OUR arm wearing the competitor's resolved
defaults. A pair is a measurement of growth policy and implementation at a
matched configuration, and it is NOT a product comparison.

**CLASS B, shipped versus shipped.** Each library at its own shipped defaults,
against ours at ours. This is the product comparison, and it is the only class
in this file a reader may quote as "what you get out of the box".

THE ONE THING MOST LIKELY TO BE MISREAD, SO IT IS FIRST
-------------------------------------------------------

**The `mojotrees` arm of Class A is NOT mojotrees at its defaults, as of
2026-08-17.** Our `lambda_l2` under every non-symmetric growth policy moved from
0.0 to 1.0 that day, a declared divergence from LightGBM's stock 0.0
(`sklearn.py::_LAMBDA_L2`, `check_parity.py::STOCK_DIVERGENCES`,
`ACCURACY_BUDGET.md` section 13). `BASE_PARAMS` therefore now names
`lambda_l2: 0.0` explicitly, on BOTH sides, so that the `lightgbm`/`mojotrees`
pair stays a mirror. That pin is what makes the pair honest and it is also what
stops that arm being our product.

So the two arms differ in EXACTLY ONE PARAMETER and this file states which:

    Class A `mojotrees`              lambda_l2 = 0.0, pinned to LightGBM's stock
    Class B `shipped.lossguide`      lambda_l2 unset, resolved by us to 1.0

Everything else about the two is identical, and that is checked rather than
claimed: `BASE_PARAMS` is the package's own constructor defaults key for key
(`num_leaves` 31, `max_depth` -1, `min_data_in_leaf` 20, `min_child_hess` 1e-3,
`max_bin` 255, `use_missing` True, `n_estimators` 100, `learning_rate` 0.1,
`lambda_l1` 0.0, `grow_policy` "lossguide"), read off
`sklearn.py::MojoTreesRegressor` on 2026-08-17, and `check_shipped_is_default`
below fails if any of them drifts apart.

WHAT "OUR SHIPPED DEFAULTS" MEANS HERE, SETTLED BY A RULING AND NOT BY THIS FILE
-------------------------------------------------------------------------------

**Settled 2026-08-17 at `273504e`, "Settle 360 vs 100": the answer is that 360
never existed.** `sklearn.py::MojoTreesRegressor` defaults
`grow_policy="lossguide"` and `n_estimators=100`, `n_estimators=360` has never
been an estimator default, and CatBoost mode does not raise the tree count --
the mode supplies depth, rate, bootstrap, scoring and `l2_leaf_reg` and leaves
the budget alone. The 360 and 72 figures are the tree counts of a 2026-08-16
decision to ship symmetric growth that was recorded and never implemented, and
they leaked out of that decision into arithmetic as though they were defaults.
`docs/LIGHTGBM_PARITY.md`'s row has been corrected, `predict.mojo`'s leaf-table
budget was recomputed at 100, and `ACCURACY_GAP.md` section 8.2 had found the
same discrepancy from the other side.

So the two Class B rows are, and the second one's label matters:

    shipped.lossguide       our DEFAULT: the code's default policy, nothing set
    shipped.symmetric       our OPT-IN CatBoost mode at ITS own defaults

`symmetrictree` is an opt-in and is not the default policy. It is in Class B
anyway, because selecting it flips the whole default set to CatBoost's
(`sklearn.py` resolves `catboost_mode = grow_policy == "symmetrictree"`), so it
is a shipped configuration a user can ask for by naming one parameter, and
"shipped versus shipped" is incomplete without it.

**Both run at 100 trees**, which is the code's own `n_estimators` default and is
also every peer's count in this harness, so every Class B row is at a MATCHED
tree count against every competitor row: `verify.check_accuracy_peer` never
abstains here and no cross-count comparison is possible even by accident.
`TREE_BUDGET_DECISION` records what that leaves unmatched, which is real and is
the largest unmatched thing about the CatBoost comparison.

THE `lambda_l2` AXIS, AND WHY IT HAS FIVE POINTS AND NOT THREE
--------------------------------------------------------------

Two documents register the same experiment and they DISAGREE about its candidate
set, which was found while writing this file:

- `ACCURACY_GAP.md` section 5 R1: `{0.0, 1.0, 3.0, 10.0}`, on `dense_regression`
  at standard AND large tiers, `imbalanced_binary` and `multiclass`, three
  repeats. Decision rule registered in advance: ship the value that is best or
  tied-best on all four, and if none is, ship the best on multiclass logloss.
- `ACCURACY_BUDGET.md` section 13, exit condition: `{0, 1, 2, 3}`.

Both were registered before any data. `PROFILE_PROTOCOL.md` rule 9 says a
registered prediction is not edited afterwards, so this run carries the UNION,
`L2_AXIS` below, and either rule can then be applied to it without anybody
rewriting either document. Dropping 10.0 would make R1's rule inapplicable;
dropping 2.0 would make section 13's inapplicable. The union costs one extra arm
over the larger of the two sets and `levers()` prices it.

`1.0` appears twice on purpose and it is not redundant: once as the axis point
and once as `shipped.lossguide`, which leaves the key UNSET. If those two cells
do not produce the same prediction digest then the package's default resolution
disagrees with an explicit 1.0, which is a defect class this harness can catch
for free. `PAIR_EXPECTATIONS` registers that prediction before the run.

THE UNSET-KEY TRAP, WHICH IS A REAL DEFECT AND NOT A STYLE NOTE
---------------------------------------------------------------

`scenarios.mojotrees_params` copies `learning_rate` and `lambda_l2` out of
`BASE_PARAMS` onto EVERY mojotrees arm, so an arm that merely omits a key does
not get the package default: it gets `BASE_PARAMS`'s value. Popping the key from
an override dict does nothing.

The way to say "unset" through this harness is to pass `None`, because the
estimator's provenance test is `self.learning_rate is not None`
(`sklearn.py::_Base._learning_rate_named`, `sklearn.py::_Base._l2_named`) and
`basic.py::_Config` forwards the dict straight into the constructor as keywords.
`None` is therefore exactly the unset state and no other value is.

This matters twice over. `auto_learning_rate=True` beside a NAMED
`learning_rate` or a NAMED `lambda_l2` does not degrade quietly, it RAISES
(`sklearn.py::_Base._auto_learning_rate_knobs`), so the symmetric arm could not
have run at all without the `None`s. `frontier.py` was carrying that defect at
head and it is fixed there in the same pass, with the correction recorded at
`frontier.RESOLVED_SINCE`.

WHAT THIS CLASS PAIR CANNOT ANSWER
-----------------------------------

It is not a sweep. Nothing here varies `max_bin`, the learning rate, the
bootstrap, the derivative precision or the tree count, so it cannot say what any
of those cost. `frontier.py` is the file for that and the two are meant to be
run separately, because a table that mixed a product comparison with an axis
sweep would invite exactly the cross-reading both files are built to prevent.

The one axis that IS here, `lambda_l2`, is here because a registered decision
rule needs it and for no other reason.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import frontier  # noqa: E402
import scenarios  # noqa: E402

PAIRS_ID = "pairs-1"
PAIRS_VERSION = 1
PAIRS_REGISTERED = "lane/pair-run, 2026-08-17, before any cell was executed"

#: Repeats per cell, matching `frontier.REPEATS` and Andrew's 2026-08-16 ruling
#: rather than restating a number. Three is the minimum that shows a spread.
#: A cell is one whole process, so cells = arms x REPEATS for the accelerator
#: legs and arms x `--oracle-repeats` for a cpu leg standing beside one.
REPEATS = frontier.REPEATS

#: The tree count every row in this run uses, ours and theirs.
#:
#: 100 is `sklearn.py::MojoTreesRegressor`'s own `n_estimators` default, it is
#: LightGBM's, it is what `scenarios.BASE_PARAMS` carries, and it is the count
#: the CatBoost and XGBoost peer arms are matched to
#: (`scenarios.CATBOOST_MATCHED`, `scenarios.XGBOOST_MATCHED`). One count for
#: the whole run is what makes every peer column in it a matched-count column.
N_ESTIMATORS = scenarios.BASE_PARAMS["n_estimators"]

#: Why 100 and what it leaves unmatched, priced.
TREE_BUDGET_DECISION = (
    "Every row runs at 100 trees, and this is a ruling rather than this plan's "
    "choice. `sklearn.py::MojoTreesRegressor` defaults n_estimators to 100 "
    "under every grow policy and CatBoost mode does not raise it, so 100 is "
    "what we ship; 360 and 72 are the tree counts of an unimplemented "
    "2026-08-16 decision and were settled out at 273504e. "
    "WHAT THIS LEAVES UNMATCHED, and it is the largest unmatched thing about "
    "the CatBoost comparison rather than a rounding: CatBoost ships 1000 "
    "iterations. So `shipped.symmetric` gives a reader CatBoost's depth, rate, "
    "bootstrap, scoring and l2_leaf_reg at a TENTH of CatBoost's tree budget, "
    "and the `catboost` row beside it is CatBoost's own defaults at OUR count "
    "rather than at its own (scenarios.CATBOOST_MATCHED). That is not a defect "
    "in the mode, it is the shipped tree count, and a Class B reader has to "
    "know it before reading the accuracy column: both libraries are asked for "
    "100 trees and only one of them ships that number. "
    "WHAT IT BUYS: every row in this run is at one count, so no cross-count "
    "reading is possible and no row abstains on the peer column. "
    "WHAT A DIFFERENT ANSWER WOULD COST: putting our symmetric row at some "
    "other budget needs three competitor rows per scenario at that budget as "
    "well, because verify.check_accuracy_peer abstains without a matched "
    "count. That is 9 more cells per scenario row at 3 repeats, and a "
    "1000-iteration CatBoost fit on the large dense row is about ten times its "
    "100-iteration one."
)

#: The `lambda_l2` candidate set, as the UNION of two registered sets.
#:
#: `ACCURACY_GAP.md` section 5 R1 registers {0.0, 1.0, 3.0, 10.0};
#: `ACCURACY_BUDGET.md` section 13's exit condition registers {0, 1, 2, 3}.
#: Both were written before any data and `PROFILE_PROTOCOL.md`'s rule 9 forbids
#: editing either afterwards, so the run carries both and each rule stays
#: applicable. See the module docstring.
L2_AXIS = (0.0, 1.0, 2.0, 3.0, 10.0)

#: Where each value came from, so a reader can see which rule needs which arm.
L2_AXIS_PROVENANCE = {
    0.0: "our default until 2026-08-17, and LightGBM's stock. In both sets",
    1.0: (
        "shipped since 2026-08-17 and the only MEASURED point on our leaf-wise "
        "arm. In both sets. Also run a second time as `shipped.lossguide` with "
        "the key UNSET, which is a free check that the default resolves to it"
    ),
    2.0: "ACCURACY_BUDGET section 13 only. Unmeasured on our leaf-wise arm",
    3.0: "CatBoost's l2_leaf_reg. In both sets. Unmeasured on our leaf-wise arm",
    10.0: (
        "ACCURACY_GAP R1 only, and it is the arm that makes R1's decision rule "
        "APPLICABLE rather than an arm anybody expects to win. Without it the "
        "rule cannot be evaluated over its own candidate set"
    ),
}

#: The primary metric R1's decision rule reads on each scenario, and the lens.
#:
#: R1 says "report excess RMSE, not RMSE, on dense_regression". `report.py`
#: already prints the excess lens beside the raw one on the generator variant of
#: any scenario that declares a noise scale (`scenarios.bayes_floor`), so the
#: rule is satisfiable from the run's own output and needs no extra arm. The
#: real-data row has no floor and shows the raw gap alone, which is why the
#: `lambda_l2` axis does not run there: R1 does not name it and an excess
#: reading is not available for it.
R1_DECISION_INPUTS = {
    "scenarios_named": (
        "dense_regression at standard tier",
        "dense_regression at large tier",
        "imbalanced_binary",
        "multiclass",
    ),
    "statistic": (
        "the primary metric per scenario: excess rmse on dense_regression "
        "(report.py prints `excess over floor` beside the raw gap on the "
        "generator variant), average_precision on imbalanced_binary, "
        "multi_logloss on multiclass"
    ),
    "rule": (
        "ship the value that is best or tied-best on all four; if none is, "
        "ship the best on multiclass logloss, because that is where the "
        "recorded damage is largest"
    ),
    "device": (
        "R1 says 'mojotrees CPU only'. This run schedules BOTH backends, so "
        "the cpu leg becomes an ORACLE cell under PROFILE_PROTOCOL C10 and "
        "runs at --oracle-repeats rather than --repeats. THAT DOES NOT AFFECT "
        "R1, because R1's rule reads an accuracy and a subject arm is "
        "bit-identical across repeats at a fixed configuration "
        "(verify.check_determinism gates it), so one cpu repeat carries the "
        "same metric three would. What one repeat gives up is the cpu "
        "BIT-IDENTITY verdict, which needs two rows of a cell. Pass "
        "--oracle-repeats 2 if that verdict is wanted; it costs one extra cell "
        "per subject arm per scenario row"
    ),
}


# ---------------------------------------------------------------------------
# The two classes, and the class is the label the report renders.
# ---------------------------------------------------------------------------
#
# `block` is an existing field on every arm, written onto every job by
# `run._job` and onto every record by `worker.py`, and rendered as the `block`
# column of `report._frontier`'s tables. So the class label is carried by the
# DATA STRUCTURE and travels into records.json, not by a caption a reader has to
# find. That is the same mechanism `frontier.BLOCKS` uses and it is reused
# rather than reinvented.
#
# A cell that belongs to BOTH classes is scheduled ONCE and labelled `A+B`.
# That is not a shortcut: a competitor at its own shipped defaults, at the
# matched tree count, is literally the same fit in both classes, and scheduling
# it twice would spend cells to produce two copies of one number and would let
# the two copies disagree by noise.

CLASSES = {
    "A/mirror": (
        "CLASS A, A MIRROR PAIR. Two rows that differ in one thing: whose "
        "implementation and growth policy ran. Everything else is held, with "
        "OUR arm wearing the competitor's resolved defaults. This class is a "
        "measurement of growth policy and implementation and it is NOT a "
        "product comparison: read the Class B rows for that. In particular the "
        "`mojotrees` row here pins lambda_l2 to LightGBM's stock 0.0, which is "
        "not what we ship"
    ),
    "A+B/peer-as-shipped": (
        "CLASS A AND CLASS B, ONE CELL. A competitor at its OWN shipped "
        "defaults at the matched tree count. It is the comparator or peer half "
        "of a Class A pair and it is simultaneously that library's Class B "
        "product row, because those are the same fit. Scheduled once"
    ),
    "B/ours-default": (
        "CLASS B, OUR PRODUCT AT ITS DEFAULTS. mojotrees with nothing "
        "overridden that a user would not also get: the package's own "
        "constructor defaults, read off sklearn.py::MojoTreesRegressor, which "
        "means grow_policy lossguide, 100 trees, learning_rate 0.1 and "
        "lambda_l2 1.0. **This is the only row in the run to quote as 'what "
        "mojotrees does out of the box'.**"
    ),
    "B/ours-opt-in": (
        "CLASS B, A SHIPPED CONFIGURATION A USER MUST ASK FOR. "
        "grow_policy='symmetrictree', which is NOT the default policy and which "
        "flips the whole default set to CatBoost's when it is named "
        "(sklearn.py resolves catboost_mode from it). It belongs in a "
        "shipped-versus-shipped table because one parameter reaches it and "
        "because it is the arm the CatBoost comparison is really about, and it "
        "must NOT be read as our default. The 2026-08-16 decision to make it "
        "the default at 360 trees was recorded and never implemented; 273504e "
        "settled that"
    ),
    "B/ours-l2-axis": (
        "CLASS B, THE lambda_l2 AXIS. Our shipped lossguide configuration with "
        "lambda_l2 named at one candidate value, to settle the shipped value "
        "under the decision rule registered in ACCURACY_GAP section 5 R1 and "
        "ACCURACY_BUDGET section 13. These are not product rows and only the "
        "winner becomes one"
    ),
}

#: The three pairs, in the order a reader should meet them. Each entry names
#: the peer engine, our mirror engine, and where the mirror is ASSERTED against
#: that peer's resolved read-back rather than merely written down beside it.
#:
#: The third field is the point of this table. A mirror that holds because two
#: dicts happen to agree is one default change away from silently not holding,
#: and this run's whole Class A claim rests on it.
MIRROR_PAIRS = (
    {
        "pair": "lightgbm",
        "peer": "lightgbm",
        "ours": "mojotrees",
        "held_by": (
            "scenarios.BASE_PARAMS, passed to BOTH sides by "
            "scenarios.lightgbm_params and scenarios.mojotrees_params and "
            "routed in scenarios.SHARED_PARAM_ROUTING, which "
            "selfcheck.check_params cross-checks key by key"
        ),
        "readback": None,
        "readback_note": (
            "NO RESOLVED READ-BACK EXISTS FOR THIS PEER, and it is the only "
            "one of the three without one. CatBoost's cell records "
            "get_all_params() and XGBoost's records save_config(); the "
            "LightGBM cell records `params_used`, which is the dict this "
            "harness PASSED. So this mirror is asserted against what we asked "
            "LightGBM for and not against what LightGBM resolved. See "
            "PAIR_ASSERTION_GAPS['lightgbm_resolved_readback']"
        ),
    },
    {
        "pair": "xgboost",
        "peer": "xgboost",
        "ours": "mojotrees_depthwise",
        "held_by": (
            "scenarios.MOJOTREES_DEPTHWISE, built from "
            "scenarios.XGBOOST_RESOLVED_DEFAULTS, with "
            "selfcheck.check_correctness_arms asserting every value in the "
            "mirror against that table"
        ),
        "readback": "engine_resolved_params, from booster.save_config()",
        "readback_note": (
            "LIVE AND ASSERTED BOTH WAYS. scenarios.check_xgboost_readback "
            "re-reads the values the mirror was built from off every XGBoost "
            "fit's own save_config() and records any drift in "
            "engine_resolved_params_drift, so a version whose defaults moved "
            "shows up on the row rather than in nobody's notes"
        ),
    },
    {
        "pair": "catboost",
        "peer": "catboost",
        "ours": "mojotrees_catboost_mode",
        "held_by": (
            "scenarios.MOJOTREES_CATBOOST_MODE, plus CatBoost's RESOLVED "
            "learning rate taken per cell out of the run's "
            "catboost_readback.json. There is no fallback: "
            "scenarios.mojotrees_catboost_mode_params raises "
            "CatBoostReadbackMissing by name when the read-back is absent or "
            "is for another cell"
        ),
        "readback": "catboost_readback.json, from model.get_all_params()",
        "readback_note": (
            "LIVE, ASSERTED, AND LOAD-BEARING. This is the strongest of the "
            "three: scenarios.check_catboost_readback drift-checks the "
            "resolved dict, scenarios.catboost_parity_rows compares our "
            "resolved arm against CatBoost's resolved arm value by value, and "
            "the arm cannot be BUILT at all without the peer's cell having "
            "written its rate first (run.CELL_ORDER makes catboost run before "
            "it, run._engine_skip_reason refuses the arm outright when "
            "catboost is absent from the run)"
        ),
    },
)

#: Where a Class A mirror is NOT asserted against the competitor's resolved
#: configuration. One entry, and it is a gap rather than a defect in anything
#: that was written.
#:
#: An entry here is an OPEN item under `LANE_RULES.md` rule 4 and is not closed
#: by this run. It is declared so that a reader of a Class A table knows which
#: of the three pairs rests on a passed dict and which two rest on a read-back.
PAIR_ASSERTION_GAPS = {
    "lightgbm_resolved_readback": {
        "pair": "lightgbm",
        "what": (
            "the lightgbm/mojotrees mirror is asserted against the dict this "
            "harness PASSES to LightGBM, never against the configuration "
            "LightGBM resolved from it"
        ),
        "why_it_matters": (
            "LightGBM does not reject a parameter it does not know: it logs "
            "'Unknown parameter' and trains anyway, and this harness runs at "
            "verbosity -1 where that line never appears "
            "(scenarios.LIGHTGBM_MIN_VERSION's docstring records exactly this "
            "failure mode for `deterministic`). So a key that stopped being "
            "honored, or was renamed, would leave the mirror labelled a mirror "
            "and reading as one. The same class of hole already cost this "
            "project the force_row_wise finding, where the one fact that would "
            "have falsified a standing paragraph was suppressed by the "
            "configuration the paragraph described "
            "(PROFILE_PROTOCOL.md, the comparator rule)"
        ),
        "cost_of_the_gap": (
            "BOUNDED, and the bound is worth stating because it is why this is "
            "not a blocker. Every key in BASE_PARAMS is a long-standing "
            "LightGBM parameter under LightGBM's own spelling, and "
            "selfcheck.check_params fails if a key reaches one engine and not "
            "the other. What is unguarded is a LightGBM release that renames "
            "or retires one of them, which is a version event and not a "
            "silent drift, and LIGHTGBM_MIN_VERSION already gates the version"
        ),
        "what_would_close_it": (
            "LightGBM exposes its resolved configuration: Booster.params on a "
            "fitted booster, and the `parameters` block of "
            "Booster.dump_model(). Recording either onto the record as "
            "`engine_resolved_params`, with a drift check against BASE_PARAMS "
            "and LIGHTGBM_ALIGNMENT, would give this pair the same standing "
            "the other two have. It is a change in "
            "engines.LightGBMEngine.run, which is inside this lane's files, "
            "and it is NOT made in this pass: it changes what every LightGBM "
            "record carries, which is a records-schema change, and "
            "schema.json plus every reader of it is a separate review from "
            "shaping a run"
        ),
    },
}

#: The predictions this run registers BEFORE it is executed, per
#: `PROFILE_PROTOCOL.md` rule 9. Not edited afterwards. A refuted one is the
#: process working.
PAIR_EXPECTATIONS = {
    "default_resolves_to_1": (
        "`shipped.lossguide` (lambda_l2 UNSET) and `l2.1.0` (lambda_l2 named "
        "1.0) produce the SAME prediction digest on every scenario row and on "
        "every device. If they do not, the package's default resolution "
        "disagrees with an explicit 1.0 and that is a defect in "
        "sklearn.py::_Base._params rather than a result about lambda_l2. "
        "Verified once by hand on 2026-08-17 (the commit that moved the "
        "default records it); this makes it mechanical"
    ),
    "class_a_mojotrees_is_not_class_b": (
        "Class A `mojotrees` and Class B `shipped.lossguide` produce DIFFERENT "
        "prediction digests on every scenario row, because they differ in "
        "lambda_l2 (0.0 against 1.0) and in nothing else. If they agree, "
        "lambda_l2 is not reaching the trainer through this harness at all and "
        "every accuracy claim about it in ACCURACY_GAP section 3.4 is about "
        "something else"
    ),
    "l2_speed_is_flat": (
        "the five `l2.*` arms are indistinguishable from each other on train "
        "seconds under M0, because the `+ lambda_l2` term is unconditional in "
        "both the gain and the Newton leaf value, so no value of it adds an "
        "instruction, a pass, a kernel or an allocation "
        "(ACCURACY_BUDGET section 13). A resolved speed difference between two "
        "of them is a finding about the tree SHAPE the value produced, not "
        "about the arithmetic, and it should be read against the leaf and node "
        "counts before anything else"
    ),
}


# ---------------------------------------------------------------------------
# The scenario rows.
# ---------------------------------------------------------------------------
#
# `l2_axis` is the one field that is not uniform, and it is False on exactly
# one row. The axis exists to satisfy a registered decision rule, that rule
# names four scenario/tier combinations, and adding a fifth would be spending
# cells on a question nobody registered. The real-data row also has no Bayes
# floor, so R1's "report excess RMSE" instruction is not even available there.

SCENARIO_ROWS = (
    {
        "row": "dense_std",
        "scenario": "dense_regression",
        "tier": "standard",
        "variant": "synthetic",
        "l2_axis": True,
        "why": (
            "the first of the two tiers ACCURACY_GAP R1 names. It is also the "
            "tier the standard-tier accuracy table in that document was taken "
            "at, so the lambda_l2 arms here are the ones with recorded prior "
            "evidence to be read against"
        ),
    },
    {
        "row": "dense_large",
        "scenario": "dense_regression",
        "tier": "large",
        "variant": "synthetic",
        "l2_axis": True,
        "why": (
            "the second tier R1 names, and the decision row this campaign "
            "already reports on: 799,110 train rows by 100 features, the one "
            "row where our accelerator is already faster than the comparator"
        ),
    },
    {
        "row": "dense_real",
        "scenario": "dense_regression",
        "tier": "large",
        "variant": "real",
        "l2_axis": False,
        "why": (
            "UCI YearPredictionMSD, 463,715 by 90, and the row we currently "
            "LOSE on. It carries both classes because a product comparison "
            "that omitted the row we lose would not be one. It does NOT carry "
            "the lambda_l2 axis: R1 does not name real data, and a real-data "
            "row has no Bayes floor, so R1's own instruction to read excess "
            "rmse rather than rmse cannot be followed on it"
        ),
    },
    {
        "row": "imbalanced",
        "scenario": "imbalanced_binary",
        "tier": "standard",
        "variant": "synthetic",
        "l2_axis": True,
        "why": (
            "named by R1, and one of the two scenarios carrying the larger "
            "half of the recorded lambda_l2 damage: 0.75x average precision at "
            "0.0. The SYNTHETIC variant is pinned deliberately -- "
            "scenarios.scenario_has_categorical is True for `auto` and for "
            "`real` here, because the bank_marketing dataset has ten "
            "categorical columns, and beside a categorical column the device "
            "refuses the symmetric arm's Cosine, its random_strength and the "
            "oblivious grow policy itself"
        ),
    },
    {
        "row": "multiclass",
        "scenario": "multiclass",
        "tier": "standard",
        "variant": "synthetic",
        "l2_axis": True,
        "why": (
            "named by R1, and the TIEBREAK scenario of its decision rule: if "
            "no lambda_l2 value is best or tied-best on all four, the rule "
            "ships the best multiclass logloss. It also carries the largest "
            "recorded damage, 3.31x worse logloss at 0.0. Two arms cannot run "
            "here and both are declared: see UNREACHABLE"
        ),
    },
)


# ---------------------------------------------------------------------------
# Our two Class B configurations.
# ---------------------------------------------------------------------------

#: Our product, as the code ships it. ONE key, and the key says "unset".
#:
#: Everything else comes from `scenarios.BASE_PARAMS`, which is the package's
#: own constructor defaults key for key. `lambda_l2: None` is what makes this
#: arm our product rather than the Class A mirror: `mojotrees_params` copies
#: `BASE_PARAMS['lambda_l2']`, which is LightGBM's stock 0.0, onto every
#: mojotrees arm, and `None` is the only value that means "the caller did not
#: name it" to `sklearn.py::_Base._l2_named`.
SHIPPED_LOSSGUIDE = {
    "lambda_l2": None,
}

#: Our OPT-IN CatBoost mode at its own defaults, at the code's tree count.
#:
#: NOT our default policy. `symmetrictree` is opt in; `sklearn.py` ships
#: `grow_policy="lossguide"`, and the 2026-08-16 decision to make symmetric the
#: shipped default at 360 trees was recorded and never implemented (settled at
#: 273504e, corrected in docs/LIGHTGBM_PARITY.md). It is a Class B row anyway
#: because naming this one parameter flips the whole default set to CatBoost's,
#: so it is a shipped configuration a user can reach.
#:
#: The values are `frontier.BASES['A']['params']`'s, which is the registered
#: definition of this shape, minus its 360-tree budget and plus the two `None`s
#: that shape has always needed and never had. Three keys must be UNSET or the
#: fit RAISES rather than degrading: `learning_rate` and `lambda_l2` because
#: `auto_learning_rate=True` beside either is refused by name, and
#: `leaf_estimation_iterations` for the same reason, which is absent from
#: `BASE_PARAMS` and so needs nothing said about it here.
#:
#: Left unset, the mode-defaults layer supplies `lambda_l2` 3.0,
#: `random_strength` 1.0 and `leaf_estimation_iterations` per objective, each
#: with CatBoost's own `SetDefault` semantics, so the automatic-rate gate stays
#: open and the rate is DERIVED from the row count, the iteration count and the
#: loss. That is what makes this arm CatBoost's shape rather than an imitation
#: of one number from it.
SHIPPED_SYMMETRIC = {
    "grow_policy": "symmetrictree",
    "max_depth": 6,
    "num_leaves": 64,
    "min_data_in_leaf": 1,
    "min_child_hess": 0.0,
    "score_function": "cosine",
    "bootstrap_type": "MVS",
    "subsample": 0.8,
    "auto_learning_rate": True,
    "learning_rate": None,
    "lambda_l2": None,
}

#: Binning parameters for our Class B rows. 254 for the symmetric row, which is
#: CatBoost's `border_count`; `BASE_PARAMS`'s 255 for everything else, which is
#: LightGBM's and is also the package's own `max_bin` default.
SHIPPED_SYMMETRIC_DATASET = {"max_bin": 254}


# ---------------------------------------------------------------------------
# Declared skips. Every one of them is a refusal read off the trainer, not a
# budget decision, and every one names both the mechanism and the exit.
# ---------------------------------------------------------------------------

UNREACHABLE = {
    "scenario_declares_no_engine": (
        "the scenario's own engine list does not carry this arm's engine "
        "(`scenarios.SCENARIOS[<scenario>]['engines']`), which is the single "
        "authority on which arms a scenario has. `run.build_matrix` would emit "
        "this skip anyway; it is declared here as well so that the plan's cell "
        "count equals the matrix's rather than being an upper bound on it, and "
        "so that a reader of the plan sees the missing pair half rather than "
        "inferring it from a gap. Today it fires once: `mojotrees_catboost_mode` "
        "on multiclass, because MOJOTREES_CATBOOST_MODE sets bootstrap_type=MVS "
        "and the multiclass trainer takes no bootstrap bundle "
        "(scenarios.MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT['multiclass'] "
        "carries the full reason). Exit: whatever that table's entry names"
    ),
    "symmetric_multiclass": (
        "the symmetric shipped arm carries bootstrap_type=MVS with NO explicit "
        "mvs_reg, and `sampling.check_mvs_reg_is_set` refuses a DERIVED mvs_reg "
        "on a softmax round: CatBoost derives the lambda from "
        "`lastIterValues[dim][leaf]`, one tree with one value per output "
        "dimension per leaf, and a round of K structurally different trees has "
        "no such table. This is the same refusal "
        "scenarios.MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT declares for the "
        "CatBoost-mode arm on this scenario, and it applies here for the "
        "identical reason: the arm is the same shape under a different arm id, "
        "so the skip has to be declared here too rather than inherited from a "
        "table keyed by engine name. NOT the reason that entry USED to give, "
        "which was that the multiclass trainer takes no bootstrap bundle; it "
        "takes one and both CPU loops honor it, and both that entry and this "
        "one were corrected against the source on 2026-08-17. CatBoost does not "
        "run MVS for multiclass either -- its defaulting block excludes the "
        "multiclass-only losses and keeps Bayesian -- so this cell would not "
        "have been the comparison it claims to be even under an mvs_reg. Exit: "
        "an explicit mvs_reg, which is then a value CatBoost does not use here, "
        "so it is a different arm rather than this one unblocked"
    ),
    "symmetric_multiclass_gpu": (
        "as `symmetric_multiclass`, and a second and INDEPENDENT refusal sits "
        "behind it on the accelerator: `train_multiclass_gpu` takes no "
        "bootstrap bundle, so `model.mojo::fit_multiclass` and "
        "`trainset.mojo::train_dataset_multiclass` raise on their GPU arms "
        "rather than training unsampled (frontier.CAPABILITIES_AT_HEAD records "
        "it, and it is the one place the missing-bundle reason IS still true). "
        "Exit: both"
    ),
}


# ---------------------------------------------------------------------------
# Building the arms.
# ---------------------------------------------------------------------------


def _devices(row):
    """The devices this row's scenario declares, cpu first.

    Read off the scenario rather than written per row, so a scenario that gains
    or loses an accelerator declaration moves this plan without an edit here.
    """
    spec = scenarios.resolve(row["scenario"], row["tier"], row["variant"])
    declared = tuple(spec["devices"])
    return tuple(d for d in ("cpu", "gpu") if d in declared)


def _arm(row, arm_id, engine, device, block, params, dataset_params,
         axis=None, axis_value=None, skip=None, skip_rule=None):
    """One arm dict, in the shape `run._normalize_arm` reads.

    THE ROW IS IN THE ARM ID and it has to be, because three of the five
    scenario rows are the SAME SCENARIO at different tiers and variants.
    `run.label` is `scenario.arm.device.tThreads.rRepeat` and carries neither
    the tier nor the variant, and `report.cell_key` was `(scenario, arm, device,
    threads)` with the same two absent, so without the row in the id the
    standard-tier, large-tier and real-data dense cells share one label and one
    report cell: three different datasets averaged into one median across nine
    repeats, printed as a measurement. That is the identity defect
    `LANE_RULES.md` rule 8 names in its "missing a dimension it needs" form, and
    `frontier._arm` embeds its `row` for the same reason. `report.cell_key` is
    fixed in the same pass so the guard is in both places rather than in the
    naming convention alone.

    THE DEVICE IS NOT IN THE ARM ID, and that is load-bearing rather than
    stylistic. `run._oracle_key`, `verify._oracle_cell_key`,
    `verify.check_device_agreement`, `verify.check_backend_proof` and
    `summarize.build_device_agreement` all pair a cpu cell with its accelerator
    cell BY ARM ID WITHOUT THE DEVICE. An id carrying the device makes every
    such pairing impossible: no cpu cell is ever an oracle, every cpu cell runs
    the full repeat count and enters the speed ranking, and every accelerator
    row reports "no cpu twin". That happened, at 351 jobs and zero oracle cells,
    and `frontier._arm` carries the same note for the same reason.
    """
    return {
        "block": block,
        "id": f"{row['row']}.{arm_id}",
        # The id without its row prefix, so that the three dense rows can be
        # read as the same arm across rows without anybody parsing a string.
        "arm_base": arm_id,
        "pairs": PAIRS_ID,
        "row": row["row"],
        "scenario": row["scenario"],
        "tier": row["tier"],
        "variant": row["variant"],
        "engine": engine,
        "device": device,
        "axis": axis,
        "axis_value": axis_value,
        "n_estimators": N_ESTIMATORS,
        "params": dict(params),
        "dataset_params": dict(dataset_params),
        "env": {},
        "repeats": REPEATS,
        "skip": skip,
        # WHICH DECLARED RULE produced the skip, as a key rather than as the
        # sentence. `check` validates the key, so a skip cannot be a sentence
        # written at the skip site, and it does not have to compare prose to
        # do it. `frontier.check` does that comparison by string and it is
        # brittle in exactly the way a reworded reason exposes.
        "skip_rule": skip_rule,
    }


def _ours(row, arm_id, block, params, dataset_params=None,
          axis=None, axis_value=None, engine="mojotrees"):
    """One of our arms on every device the row declares.

    Both legs are always enumerated. The cpu leg is not a duplicate of the
    accelerator leg: `run._mark_oracle_cells` labels it an ORACLE and reduces
    its repeats, and `verify.check_device_agreement` and
    `verify.check_backend_proof` are both built on its existing. Dropping it to
    save time disarms two gates that go quiet rather than failing
    (PROFILE_PROTOCOL C10).
    """
    out = []
    for device in _devices(row):
        rule = _skip_for(arm_id, engine, row, device)
        out.append(
            _arm(
                row, arm_id, engine, device, block, params,
                dataset_params or {}, axis, axis_value,
                UNREACHABLE[rule] if rule else None, rule,
            )
        )
    return out


def _skip_for(arm_id, engine, row, device):
    """The declared refusal for this (arm, engine, row, device) as an
    `UNREACHABLE` key, or None.

    The scenario's own engine list is consulted FIRST, because it is the
    authority `run.build_matrix` reads and a plan that disagreed with it would
    over-count the run.
    """
    spec = scenarios.resolve(row["scenario"], row["tier"], row["variant"])
    if engine not in spec["engines"]:
        return "scenario_declares_no_engine"
    if arm_id == "shipped.symmetric" and row["scenario"] == "multiclass":
        if device == "cpu":
            return "symmetric_multiclass"
        return "symmetric_multiclass_gpu"
    return None


def class_a_arms(row):
    """The three mirror pairs for one scenario row.

    The peer half of each pair carries the `A+B` block, because a competitor at
    its own shipped defaults at the matched tree count IS its Class B product
    row. One cell, two classes, one number.

    Our half carries no parameter overrides at all: each of the three mojotrees
    engines is already the mirror. `mojotrees` takes `BASE_PARAMS`, which pins
    `lambda_l2` to LightGBM's stock 0.0 on both sides; `mojotrees_depthwise`
    applies `scenarios.MOJOTREES_DEPTHWISE`; `mojotrees_catboost_mode` applies
    `scenarios.MOJOTREES_CATBOOST_MODE` plus CatBoost's resolved rate. An
    override here would break the mirror, which is why there is not one.

    Arms whose engine the scenario does not declare are not written down as
    skips here. `run.build_matrix` emits a declared skip for them by name
    ("<scenario> declares no <engine> arm") off `spec['engines']`, which is the
    one authority on it, and a second copy of that decision in this file would
    be a rule with two homes.
    """
    out = []
    for pair in MIRROR_PAIRS:
        rule = _skip_for(pair["peer"], pair["peer"], row, "cpu")
        out.append(
            _arm(
                row, pair["peer"], pair["peer"], "cpu",
                "A+B/peer-as-shipped", {}, {},
                skip=UNREACHABLE[rule] if rule else None, skip_rule=rule,
            )
        )
        out.extend(
            _ours(row, pair["ours"], "A/mirror", {}, engine=pair["ours"])
        )
    return out


def class_b_arms(row):
    """Our shipped rows and the `lambda_l2` axis for one scenario row.

    The competitor half of Class B is not built here: it is the `A+B` cells
    `class_a_arms` already emitted, at the same tree count and the same
    configuration.
    """
    out = []
    out.extend(
        _ours(row, "shipped.lossguide", "B/ours-default", SHIPPED_LOSSGUIDE)
    )
    out.extend(
        _ours(
            row, "shipped.symmetric", "B/ours-opt-in", SHIPPED_SYMMETRIC,
            SHIPPED_SYMMETRIC_DATASET,
        )
    )
    if not row["l2_axis"]:
        return out
    for value in L2_AXIS:
        params = dict(SHIPPED_LOSSGUIDE)
        params["lambda_l2"] = float(value)
        out.extend(
            _ours(
                row, f"l2.{value}", "B/ours-l2-axis", params,
                axis="lambda_l2", axis_value=float(value),
            )
        )
    return out


def arms():
    """Every arm of the pair run, both classes, in reading order."""
    out = []
    for row in SCENARIO_ROWS:
        out.extend(class_a_arms(row))
        out.extend(class_b_arms(row))
    return out


# ---------------------------------------------------------------------------
# The invariants, raised rather than printed.
# ---------------------------------------------------------------------------


def check_shipped_is_default(problems):
    """`BASE_PARAMS` still equals the package's constructor defaults, except
    `lambda_l2`.

    THIS IS THE CHECK THE WHOLE FILE RESTS ON. Class B's `shipped.lossguide`
    arm claims to be our product, and the only thing making that true is that
    `BASE_PARAMS` carries the package's own defaults for every key it names. If
    a default moves on either side, that arm silently stops being our product
    while still being labelled it, which is the identical defect
    `ACCURACY_BUDGET` section 13 records for the mirror pair, running the third
    direction.

    Read from the source text rather than by importing the estimator, because
    `python/mojotrees/sklearn.py` imports the compiled extension and this file
    must run with nothing built. A signature default is a literal in that file
    and a regular expression over it is exact for the shapes involved; a
    default that becomes an expression fails this check loudly, which is the
    correct outcome, because an expression is not something a plan may assume.
    """
    import re

    path = os.path.abspath(
        os.path.join(HERE, "..", "..", "python", "mojotrees", "sklearn.py")
    )
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError as exc:
        problems.append(f"cannot read {path} to check the shipped defaults: {exc}")
        return

    def signature_default(name):
        found = re.search(
            r"^\s{8}" + re.escape(name) + r"=([^,\n]+),\s*$", text, re.MULTILINE
        )
        return None if found is None else found.group(1).strip()

    expected = {
        "num_leaves": "31",
        "max_depth": "-1",
        "min_data_in_leaf": "20",
        "min_child_hess": "1e-3",
        "max_bin": "255",
        "use_missing": "True",
        "n_estimators": "100",
        "grow_policy": '"lossguide"',
        "lambda_l1": "_LAMBDA_L1",
    }
    for name, want in expected.items():
        got = signature_default(name)
        if got != want:
            problems.append(
                f"sklearn.py::MojoTreesRegressor defaults {name} to {got!r} and "
                f"this plan was written against {want!r}. Class B's "
                "`shipped.lossguide` arm claims to be our product and it is "
                "only that while BASE_PARAMS matches these"
            )
    # The two that must be UNSET in the signature, because an unset key is the
    # only thing that reaches the mode-defaults layer and the automatic-rate
    # gate. A float here would make `_learning_rate_named` and `_l2_named`
    # unable to tell a caller's value from a default.
    for name in ("learning_rate", "lambda_l2"):
        got = signature_default(name)
        if got != "None":
            problems.append(
                f"sklearn.py::MojoTreesRegressor defaults {name} to {got!r} "
                "rather than None. This plan passes None to mean 'unset', "
                "which only works while the signature default is None"
            )
    # And the value an unset lambda_l2 resolves to, which is the divergence this
    # whole run exists to settle. Read through `verify.shipped_constant` rather
    # than with a second regular expression over the same literal: two readers of
    # one constant is how the two come to disagree, and the anchor staleness
    # mechanism already needs this one.
    import verify

    shipped = verify.shipped_constant("_LAMBDA_L2")
    if shipped is None:
        problems.append(
            "verify.shipped_constant cannot read sklearn.py::_LAMBDA_L2, so this "
            "plan cannot confirm which lambda_l2 we ship"
        )
        return
    for key in ("num_leaves", "max_depth", "min_data_in_leaf",
                "min_child_hess", "max_bin", "use_missing", "n_estimators"):
        if key not in scenarios.BASE_PARAMS:
            problems.append(f"BASE_PARAMS no longer names {key}")
    if scenarios.BASE_PARAMS.get("lambda_l2") == shipped:
        problems.append(
            f"BASE_PARAMS['lambda_l2'] is {shipped}, which is also "
            "sklearn.py::_LAMBDA_L2, so the Class A mirror arm and our shipped "
            "default are the SAME arm and Class B's shipped.lossguide row is a "
            "duplicate of it. That is a legitimate state -- it is what held "
            "before 2026-08-17 -- but this plan is built on the two being "
            "different and it must be re-read rather than run"
        )
    if shipped not in [float(v) for v in L2_AXIS]:
        problems.append(
            f"we ship lambda_l2 = {shipped} and it is not a point on L2_AXIS "
            f"{L2_AXIS}, so this run cannot tell whether the value we ship is "
            "the value the decision rule would pick"
        )


def check(all_arms=None):
    """The invariants of the plan, raised rather than printed.

    Run from `main` on every invocation and by `run.py` before any cell is
    scheduled, for the reason `run.comparator_banner` prints before the first
    cell: a plan that cannot be read correctly must not be run at all. Static.
    It imports no trainer, reads no results file and takes no lock.
    """
    problems = []
    all_arms = list(all_arms if all_arms is not None else arms())
    seen = {}
    for arm in all_arms:
        if arm["block"] not in CLASSES:
            problems.append(f"{arm['id']}: unknown class block {arm['block']!r}")
        if arm["skip"]:
            rule = arm.get("skip_rule")
            if rule not in UNREACHABLE:
                problems.append(
                    f"{arm['id']}: skip_rule {rule!r} is not a key of "
                    "UNREACHABLE, so this skip is a sentence written at the "
                    "skip site rather than a declared rule"
                )
            elif arm["skip"] != UNREACHABLE[rule]:
                problems.append(
                    f"{arm['id']}: skip text does not match UNREACHABLE"
                    f"[{rule!r}], so the row and the rule have drifted apart"
                )
            continue
        if arm["n_estimators"] != N_ESTIMATORS:
            problems.append(
                f"{arm['id']}: {arm['n_estimators']} trees against this run's "
                f"single count of {N_ESTIMATORS}. One count is what makes every "
                "peer column here a matched-count column"
            )
        key = (arm["row"], arm["id"], arm["device"])
        if key in seen:
            problems.append(f"duplicate cell {key}, scheduled twice")
        seen[key] = arm

    # Every one of our arms must have BOTH legs on a row whose scenario
    # declares an accelerator, or the oracle pairing has nothing to pair. A
    # missing leg is the failure that produced 351 jobs and zero oracle cells.
    for row in SCENARIO_ROWS:
        devices = set(_devices(row))
        if "gpu" not in devices:
            continue
        ours = {}
        for arm in all_arms:
            if arm["row"] != row["row"] or arm["skip"]:
                continue
            if arm["block"] == "A+B/peer-as-shipped":
                continue
            ours.setdefault(arm["id"], set()).add(arm["device"])
        for arm_id, legs in sorted(ours.items()):
            if legs != devices:
                problems.append(
                    f"{row['row']}/{arm_id}: legs {sorted(legs)} against the "
                    f"row's declared {sorted(devices)}. One of our arms with no "
                    "cpu leg beside its accelerator leg has no oracle cell, so "
                    "verify.check_device_agreement and "
                    "verify.check_backend_proof both go quiet on it rather than "
                    "failing"
                )

    # The mirror table and the arms must name the same engines.
    engines_in_plan = {arm["engine"] for arm in all_arms}
    for pair in MIRROR_PAIRS:
        for role in ("peer", "ours"):
            if pair[role] not in engines_in_plan:
                problems.append(
                    f"MIRROR_PAIRS names {pair[role]!r} as the {role} half of "
                    f"the {pair['pair']} pair and no arm carries that engine"
                )
    # The CatBoost-mode arm cannot be built without a CatBoost cell in the same
    # run. Checked here as well as in `run._engine_skip_reason`, because that
    # function answers per cell and this one answers for the plan.
    if "mojotrees_catboost_mode" in engines_in_plan and "catboost" not in engines_in_plan:
        problems.append(
            "the CatBoost-mode mirror is in the plan and the catboost peer is "
            "not; that arm takes CatBoost's RESOLVED learning rate for the same "
            "cell out of catboost_readback.json and raises without it"
        )
    check_shipped_is_default(problems)
    if problems:
        raise AssertionError(
            "pair plan is inconsistent:\n  " + "\n  ".join(problems)
        )
    return True


# ---------------------------------------------------------------------------
# The price.
# ---------------------------------------------------------------------------

#: Per-cell cost that is not training: process start, extension import, data
#: build or load, binning, three prediction repeats, the digest and the record.
#: UNMEASURED, and taken from `frontier.FIXED_PER_CELL_S` rather than restated.
#: Three columns rather than one number, because the honest form of an
#: unmeasured quantity is a range and because a two-hundred-cell run multiplies
#: it: every ten seconds here is over half an hour of run.
FIXED_PER_CELL_S = frontier.FIXED_PER_CELL_S

#: Recorded train seconds at 100 trees, reused from `frontier` rather than
#: copied. It covers the two dense LARGE rows and nothing else, so the standard
#: tier and both non-regression scenarios are counted and reported UNPRICED. An
#: unpriced cell is not a free cell and the plan says so rather than filling in
#: a plausible number.
MEASURED_TRAIN_S_AT_100 = frontier.MEASURED_TRAIN_S_AT_100

#: Which recorded figure stands in for which arm, and every one of these is a
#: PROXY. Labelled `estimated` on the way out, per PROFILE_PROTOCOL's
#: provenance vocabulary.
TRAIN_PROXY = {
    "mojotrees": ("B", True),
    "mojotrees_depthwise": ("B", True),
    "mojotrees_catboost_mode": ("A", False),
    "shipped.lossguide": ("B", True),
    "shipped.symmetric": ("A", False),
    "lightgbm": ("lightgbm", False),
    "catboost": ("catboost", False),
    "xgboost": (None, False),
}


def _row_key(row):
    """The `frontier.MEASURED_TRAIN_S_AT_100` row name for one of ours, or
    None. Only the two large dense rows have a recorded figure."""
    return {
        "dense_large": "dense_synthetic",
        "dense_real": "dense_real",
    }.get(row["row"])


def _train_seconds(arm):
    row_key = _row_key(next(r for r in SCENARIO_ROWS if r["row"] == arm["row"]))
    if row_key is None:
        return None
    arm_id = arm["arm_base"]
    proxy = TRAIN_PROXY.get(arm_id)
    if proxy is None and arm_id.startswith("l2."):
        proxy = TRAIN_PROXY["shipped.lossguide"]
    if proxy is None:
        return None
    base, per_device = proxy
    if base is None:
        return None
    if per_device:
        return MEASURED_TRAIN_S_AT_100.get((row_key, base, arm["device"]))
    return MEASURED_TRAIN_S_AT_100.get((row_key, base))


def estimate(all_arms=None, oracle_repeats=1):
    """Cells and wall clock, per scenario row and in total.

    A cell is one whole process: one arm, one repeat. **An ORACLE cell counts
    `oracle_repeats` and not `REPEATS`**, which is the single largest term in
    this arithmetic and the reason a naive arms-times-repeats count overstates
    the run by about a third. A skipped arm costs nothing and is counted
    separately, because a declared skip is a fact about the design rather than
    a unit of work.
    """
    all_arms = list(all_arms if all_arms is not None else arms())
    accelerated = {
        (arm["row"], arm["id"])
        for arm in all_arms
        if not arm["skip"] and arm["device"] != "cpu"
        and arm["block"] != "A+B/peer-as-shipped"
    }
    rows = {}
    for arm in all_arms:
        bucket = rows.setdefault(
            arm["row"],
            {"arms": 0, "skipped": 0, "cells": 0, "oracle_cells": 0,
             "train_s": 0.0, "unpriced": 0},
        )
        if arm["skip"]:
            bucket["skipped"] += 1
            continue
        oracle = (
            arm["device"] == "cpu"
            and (arm["row"], arm["id"]) in accelerated
        )
        repeats = int(oracle_repeats) if oracle else REPEATS
        bucket["arms"] += 1
        bucket["cells"] += repeats
        if oracle:
            bucket["oracle_cells"] += repeats
        seconds = _train_seconds(arm)
        if seconds is None:
            bucket["unpriced"] += 1
        else:
            bucket["train_s"] += seconds * repeats

    total = {"arms": 0, "skipped": 0, "cells": 0, "oracle_cells": 0,
             "train_s": 0.0, "unpriced": 0}
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


def levers(all_arms=None):
    """What each way of making this cheaper would cost. NOTHING HERE IS
    APPLIED. The plan is reported at its real price and these are the prices of
    the obvious cuts, so that a trim is Andrew's decision made against a number
    rather than this lane's made against a guess."""
    all_arms = list(all_arms if all_arms is not None else arms())
    base = estimate(all_arms)["total"]["cells"]

    def without(predicate):
        """The SIGNED change in cells from dropping these arms, so a saving
        renders negative. Returned signed rather than as a magnitude because the
        table also holds one entry that ADDS cells, and a column mixing
        magnitudes with a cost is a column a reader has to interpret."""
        kept = [a for a in all_arms if not predicate(a)]
        return estimate(kept)["total"]["cells"] - base

    return {
        "drop_dense_real": {
            "cells": without(lambda a: a["row"] == "dense_real"),
            "gives_up": (
                "the only real-data row and the only row we currently LOSE on. "
                "Both classes lose their real-data column, so the product "
                "comparison would rest on generated data alone. This is the "
                "cut this lane recommends AGAINST"
            ),
        },
        "drop_l2_10": {
            "cells": without(lambda a: a["arm_base"] == "l2.10.0"),
            "gives_up": (
                "ACCURACY_GAP R1's decision rule becomes inapplicable, because "
                "10.0 is in its registered candidate set. ACCURACY_BUDGET "
                "section 13's rule still applies. Cheapest real cut and it "
                "costs one of the two registered rules"
            ),
        },
        "drop_l2_axis_at_standard_tier": {
            "cells": without(
                lambda a: a["row"] == "dense_std" and a["axis"] == "lambda_l2"
            ),
            "gives_up": (
                "R1 names dense_regression at BOTH tiers, so its rule can no "
                "longer be evaluated on all four of the scenario/tier "
                "combinations it registers"
            ),
        },
        "drop_class_a": {
            "cells": without(lambda a: a["block"] == "A/mirror"),
            "gives_up": (
                "the three mirror pairs, which is the class that isolates "
                "growth policy and implementation from configuration. The peer "
                "cells survive because they are Class B rows too, so this cut "
                "is cheaper than it looks and it removes the only rows in the "
                "run where 'faster at the same settings' is a sentence anybody "
                "may write"
            ),
        },
        "oracle_repeats_2": {
            "cells": (
                estimate(all_arms, oracle_repeats=2)["total"]["cells"] - base
            ),
            "gives_up": (
                "nothing. This is a cost, not a saving: it BUYS "
                "verify.check_determinism's cpu bit-identity verdict for every "
                "one of our arms, which one oracle repeat cannot produce. Pass "
                "--oracle-repeats 2 when the run is about cpu bit-identity"
            ),
        },
    }


def plan(oracle_repeats=1):
    all_arms = arms()
    check(all_arms)
    return {
        "pairs": PAIRS_ID,
        "pairs_version": PAIRS_VERSION,
        "registered": PAIRS_REGISTERED,
        "repeats": REPEATS,
        "oracle_repeats": oracle_repeats,
        "n_estimators": N_ESTIMATORS,
        "tree_budget_decision": TREE_BUDGET_DECISION,
        "classes": dict(CLASSES),
        "mirror_pairs": [dict(p) for p in MIRROR_PAIRS],
        "pair_assertion_gaps": dict(PAIR_ASSERTION_GAPS),
        "expectations": dict(PAIR_EXPECTATIONS),
        "l2_axis": list(L2_AXIS),
        "l2_axis_provenance": {str(k): v for k, v in L2_AXIS_PROVENANCE.items()},
        "r1_decision_inputs": dict(R1_DECISION_INPUTS),
        "scenario_rows": [dict(r) for r in SCENARIO_ROWS],
        "unreachable": dict(UNREACHABLE),
        "arms": all_arms,
        "estimate": estimate(all_arms, oracle_repeats),
        "levers": levers(all_arms),
        "fixed_per_cell_s": list(FIXED_PER_CELL_S),
        "provenance": (
            "cell counts are EXACT arithmetic over the arm list. Wall clock is "
            "ESTIMATED: train seconds are recorded figures at 100 trees for the "
            "two large dense rows only (frontier.MEASURED_TRAIN_S_AT_100, "
            "itself from COMPARISON_RUN_2026-08-16.md) used as PROXIES across "
            "arms of the same shape, and the per-cell fixed cost is unmeasured "
            "and given as a range. No number here is measured for this plan"
        ),
    }


def _fmt_hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def render(payload, out):
    out(f"# {payload['pairs']} v{payload['pairs_version']}: "
        "three mirror pairs and shipped versus shipped\n")
    out(f"Registered {payload['registered']}. Nothing here has been run.\n")
    out(f"\nEvery row at {payload['n_estimators']} trees, "
        f"{payload['repeats']} repeats, oracle cells at "
        f"{payload['oracle_repeats']}.\n")

    total = payload["estimate"]["total"]
    out(f"\n## THE NUMBER: {total['cells']} CELLS\n")
    out(
        f"{total['arms']} arms scheduled, {total['skipped']} declared skips, "
        f"**{total['cells']} cells**, of which {total['oracle_cells']} are "
        f"oracle cells at {payload['oracle_repeats']} repeat(s). "
        f"{total['unpriced']} arms have no recorded train time and are counted "
        "but not priced.\n"
    )
    out("\n| fixed cost per cell | estimated wall clock |")
    out("| --- | --- |")
    for fixed in payload["fixed_per_cell_s"]:
        key = f"fixed_{int(fixed)}s"
        out(f"| {int(fixed)} s | {_fmt_hms(total['wall_s'][key])} |")
    out(f"\n{payload['provenance']}\n")

    out("\n## The two classes\n")
    for block, why in sorted(payload["classes"].items()):
        out(f"- **`{block}`** -- {why}\n")

    out("\n## Cells per scenario row\n")
    out("\n| row | scenario | tier | variant | arms | cells | oracle | skips |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["scenario_rows"]:
        bucket = payload["estimate"]["rows"].get(row["row"], {})
        out(
            f"| {row['row']} | {row['scenario']} | {row['tier']} | "
            f"{row['variant']} | {bucket.get('arms', 0)} | "
            f"{bucket.get('cells', 0)} | {bucket.get('oracle_cells', 0)} | "
            f"{bucket.get('skipped', 0)} |"
        )

    out("\n## The mirror pairs, and what holds each one\n")
    out("\n| pair | ours | resolved read-back | held by |")
    out("| --- | --- | --- | --- |")
    for pair in payload["mirror_pairs"]:
        out(
            f"| {pair['peer']} | {pair['ours']} | "
            f"{pair['readback'] or '**NONE**'} | {pair['held_by']} |"
        )
    for pair in payload["mirror_pairs"]:
        out(f"\n**{pair['peer']}**: {pair['readback_note']}\n")

    if payload["pair_assertion_gaps"]:
        out("\n## Where a mirror is NOT asserted against a read-back\n")
        for name, gap in sorted(payload["pair_assertion_gaps"].items()):
            out(f"\n**{name}** ({gap['pair']} pair)\n")
            out(f"- what: {gap['what']}\n")
            out(f"- why it matters: {gap['why_it_matters']}\n")
            out(f"- cost of the gap: {gap['cost_of_the_gap']}\n")
            out(f"- what would close it: {gap['what_would_close_it']}\n")

    out("\n## The lambda_l2 axis\n")
    for value in payload["l2_axis"]:
        out(f"- **{value}** -- {payload['l2_axis_provenance'][str(value)]}\n")
    out("\nThe decision rule this axis exists to satisfy:\n")
    for key, value in payload["r1_decision_inputs"].items():
        out(f"- {key}: {value}\n")

    out("\n## Registered predictions, before the data\n")
    for name, text in sorted(payload["expectations"].items()):
        out(f"- **{name}** -- {text}\n")

    out("\n## Declared skips\n")
    skips = [a for a in payload["arms"] if a["skip"]]
    if not skips:
        out("\nNone.\n")
    for arm in skips:
        out(f"\n- `{arm['id']}` on {arm['device']}: {arm['skip']}\n")

    out("\n## What a cheaper run would cost\n")
    out("\n| lever | cells | gives up |")
    out("| --- | --- | --- |")
    for name, lever in sorted(payload["levers"].items()):
        out(f"| {name} | {lever['cells']:+d} | {lever['gives_up']} |")

    out(f"\n## The tree budget\n\n{payload['tree_budget_decision']}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the plan here as JSON")
    parser.add_argument(
        "--skips", action="store_true", help="print only the declared skips"
    )
    parser.add_argument(
        "--oracle-repeats", type=int, default=1,
        help="what run.py will be given; changes the cell count",
    )
    args = parser.parse_args(argv)
    payload = plan(args.oracle_repeats)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    if args.skips:
        for arm in payload["arms"]:
            if arm["skip"]:
                print(f"{arm['id']} on {arm['device']}: {arm['skip']}")
        return 0
    render(payload, print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
