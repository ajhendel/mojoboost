#!/usr/bin/env python3
"""Which refusals does a proposed DEFAULT parameter set walk into?

    python tools/default_refusal_audit.py
    python tools/default_refusal_audit.py --json
    python tools/default_refusal_audit.py --set catboost_defaults

WHY THIS EXISTS
---------------
On 2026-08-16 Andrew decided that mojotrees's shipped defaults become
CatBoost's CPU defaults. Within an hour that proposal had collided with three
separate refusals, each of which was individually correct:

1. `score_function=Cosine` on the GPU. The per-node device search cannot score
   a Cosine ratio, and `_launch_oblivious_search` raises on anything but
   `SCORE_L2` for a real mathematical reason: a level's Cosine score is a ratio
   of two cross-leaf accumulators with one square root at the end, and summing
   per-leaf Cosine gains is not the Cosine of the level.
2. `grow_policy=symmetrictree` against `gpu_tree_tables.mojo:424`, which
   returns `TREE_RESIDENT_DEPTHWISE` for anything that is not leafwise.
3. `bootstrap_type=MVS` on multiclass and sparse, where the sparse arm refuses
   the bundle by name and the multiclass trainer takes no bundle at all, so
   even CatBoost's own Bayesian fallback has nowhere to land.

**Every one of those refusals is right. They are jointly unsatisfiable with the
proposed default set.** That is the insight this file exists to mechanize, and
it is not the same question any other tool here asks:

- `connectivity_audit.py` asks whether a module is imported.
- `refusal_consistency.py` asks whether four layers AGREE about a refusal.
- `default_argument_audit.py` asks whether a parameter is ever passed.
- `surface_parity.py` asks whether four entry surfaces answer alike.

None of them asks **"if this were the default, what would refuse?"** Changing a
default is a compatibility change against every refusal in the tree, and until
now nothing checked a proposed default set against the refusals it would hit.
The three collisions above were each found by a person, one at a time, hours
apart, after the decision was already made.

WHAT IT DOES
------------
For each parameter in a proposed default set, it finds every `raise` in the
native and binding sources whose message names that parameter, and reports them
grouped by parameter with the file, the line and the message. A reader then
decides which fire.

WHAT IT IS NOT
--------------
It does not evaluate conditions. It cannot tell you that a refusal fires only
on the sparse path, or only above a row count, because that needs the call
graph and a type checker and this is a regex over text. **Every row is a site
to read, not a prediction.** The output is deliberately a reading list.

It also cannot see a refusal that does not name its parameter in the message.
That is a real blind spot and it is the same one `refusal_consistency.py` has:
a `BLOCK_*` constant whose message says "this configuration" rather than naming
the key is invisible here. Refusals in this repository are unusually good about
naming the parameter, which is what makes the approach work at all.

HOW IT GATES WITHOUT BECOMING NOISE
-----------------------------------
`--check` is the gated mode and it does NOT fail on every row. Failing on
every row is how a gate gets disabled: this tool finds seventy-odd sites for
the current proposal and most of them are conditions the default never
reaches. A check whose honest output is "go and read seventy things" cannot
block a commit.

So `--check` fails only on a site that has NOT been acknowledged. `ACKNOWLEDGED`
is the same shape as `MONOTONE_EXEMPT` in `check_parity.py` and
`CATBOOST_UNMATCHABLE` in the harness: an entry is an argument, not a
suppression, and it records which of three things is true.

    RESOLVED   the refusal cannot fire for the default, and why
    DIVERGENCE the default resolves to something other than the proposed
               value on that path, recorded as OURS rather than as parity
    BLOCKING   it fires, it is not yet fixed, and the default cannot ship
               until it is

**A BLOCKING entry still fails the check.** That is the point: the four
collisions found by hand today were each discovered after the decision, and a
gate that lets a known-blocking collision sit silently is the same failure one
layer up. What the acknowledgement buys is that a NEW collision is
distinguishable from an old one.

Advisory in its default mode; gated in `--check`.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEARCH_DIRS = (
    os.path.join(ROOT, "src", "mojotrees"),
    os.path.join(ROOT, "bindings"),
    os.path.join(ROOT, "python", "mojotrees"),
)

#: Proposed default sets, by name. Each maps the parameter name a refusal
#: message would use to the value being proposed. The value is reported but
#: not evaluated; it is there so a reader knows what is being asked for.
DEFAULT_SETS = {
    # Andrew, 2026-08-16: mojotrees's shipped defaults become CatBoost's CPU
    # defaults, except n_estimators, where CatBoost's 1000 becomes **360**.
    #
    # It was 500 for several hours and this table carried the 500 as DATA
    # rather than as prose, so `--check` and `--removing` were reasoning about
    # a default set that was not the default. Caught by the other campaign
    # with a grep, in ninety seconds, which is the cheap part; the expensive
    # part is that nothing would ever have told either of us to look.
    #
    # **RETRACTED. An earlier version of this comment claimed the tree count
    # was silently a mode switch. It is not, on CPU, and the error is worth
    # keeping because it is a clean one.**
    #
    # `defaults_helper.h:33-42` hard-sets Plain when the option is unset and
    # `(learnSampleCount >= 50000 || IterationCount < 500)`. At T=500 that
    # iteration clause is false, so a scenario under 50,000 rows does not get
    # the hard-set. I read that as "therefore Ordered". **It only means the
    # option is left NotSet**, and NotSet is not Ordered: the constructed
    # default is `EBoostingType::Plain` (`boosting_options.cpp:16`), and the
    # only site that installs Ordered as a default is
    # `catboost_options.cpp:806`, guarded by `TaskType == ETaskType::GPU`.
    # **On CPU that site cannot fire, so the resolved value is Plain whether
    # or not the hard-set runs.**
    #
    # The mistake was treating a disjunction's failure as implying the
    # opposite assignment, when it implies only the absence of one particular
    # assignment. Verified at all three sites rather than reasoned from one.
    # The count is inert here after all; the 500-to-360 correction below still
    # matters because this table is DATA that `--check` reasons about.
    "catboost_defaults": {
        "grow_policy": "symmetrictree",
        "max_depth": 6,
        "n_estimators": 360,
        "learning_rate": "auto",
        "auto_learning_rate": True,
        "boosting_type": "plain",
        "bootstrap_type": "MVS",
        "subsample": 0.8,
        "bagging_temperature": "(Bayesian fallback)",
        "random_strength": 1.0,
        "score_function": "cosine",
        "lambda_l2": 3.0,
        "leaf_estimation_iterations": "(per objective)",
        "max_bin": 254,
        "max_cat_to_onehot": 2,
        "max_ctr_complexity": 1,
        "min_data_in_leaf": 1,
    },
}

#: Sites a human has read and ruled on, keyed by (parameter, file, message
#: prefix). The message prefix rather than a line number because line numbers
#: churn on every edit above them and a gate that fails on unrelated edits is
#: a gate that gets skipped.
#:
#: Seeded 2026-08-16 with the four collisions found by hand. Everything not
#: listed here is unreviewed, which is the state this table exists to make
#: visible.
ACKNOWLEDGED = {
    ("bootstrap_type", "src/mojotrees/train_gpu.mojo"): (
        "RESOLVED",
        "NOT a refusal, and the one entry here that exists to stop a block "
        "being written. MVS is honored on a GPU fit: device_gradients "
        "reports ROUND_MVS_HOST_MAGNITUDES (gpu_fused_round.mojo:656) and "
        "AUTO resolves it by taking the host-gradient arm, where "
        "sampling.bootstrap_round draws the sampler exactly and the trees "
        "still grow on the device. train_gpu.mojo:788 states it in its own "
        "docstring and the reporting path at :885 emits honored=yes. The "
        "raise at :3508 is a defensive guard INSIDE the device-gradient "
        "loop, for a caller arriving there directly or for the eligibility "
        "check being loosened later without this arm being taught the draw "
        "-- a configuration that never reaches it. What MVS costs is the "
        "device derivative kernel, not the accelerator. Adding a "
        "device_policy block for it would refuse a working path, which is "
        "the BLOCK_SCORE_FUNCTION mistake. "
        "SECOND SITE IN THIS FILE, covered explicitly because this key is "
        "per-file and would otherwise excuse it on an argument that does not "
        "reach it: the raise on random_strength beside "
        "bootstrap_type=Bayesian on the device-gradient arm. That one is "
        "RESOLVED for a DIFFERENT reason -- the proposed default set is "
        "bootstrap_type=MVS, not Bayesian, and MVS routes to the "
        "host-gradient arm where the scale is computed before either "
        "sampler. It would fire for a Bayesian fit that also set "
        "random_strength, which the default set does not describe. Noting "
        "both because an acknowledgement that covers less than it excuses is "
        "the same defect this tool exists to find",
    ),
    ("bootstrap_type", "src/mojotrees/model.mojo"): (
        "BLOCKING",
        "SPLIT ROW. The single-output DENSE path is fine and is not this "
        "row: model.fit passes the bundle through to train_gpu and it is "
        "honored on both backends (f9 verified the pass-through 2026-08-17; "
        "my earlier DIVERGENCE entry treated the whole file as one answer "
        "and it is two). What BLOCKS is fit_multiclass at model.mojo:580, "
        "one of four guarded raises that all read `backend == GPU_DEVICE and "
        "bootstrap.enabled()` -- the others are model_sparse.mojo:126 and "
        ":209 and trainset.mojo:1744, each tracked in its own file's row. I "
        "read all four rather than taking the list: the guard is "
        "`bootstrap.enabled()`, which asks whether the bundle is ACTIVE and "
        "not whether anyone NAMED it, so a mode-supplied MVS raises exactly "
        "as a typed one does. That is the yield rule missing on the GPU "
        "backend, and it is why this is BLOCKING rather than a divergence: "
        "there is no fallback, only a raise. **MERGE-ORDER CONSTRAINT, "
        "Andrew 2026-08-17: lane/defaults-release defaults bootstrap_type to "
        "MVS at subsample 0.8, so the day it merges a multiclass or sparse "
        "fit at the defaults that the policy sends to the accelerator raises "
        "with nothing to catch it. It does not merge until these four sites "
        "obey the yield rule on the GPU backend -- a DEFAULTED bundle yields "
        "to unsampled with the reason recorded, a NAMED one still refuses -- "
        "or those fits are routed. Adding a BLOCK_BOOTSTRAP is the WRONG "
        "fix: it would refuse the dense single-output path that honors MVS "
        "today.** And these fire under device='auto' as well as an explicit "
        "request, because there is no BLOCK_BOOTSTRAP to route them; "
        "device='auto' reaches the CPU on the shipped default only because "
        "BLOCK_SCORE_FUNCTION and BLOCK_RANDOM_STRENGTH fire first. So "
        "retiring either of those two, before the yield rule lands here, "
        "turns this from a latent raise into every multiclass and sparse "
        "GPU fit of the shipped default. Retiring a block is not a local "
        "change when another parameter's only protection was resting on it",
    ),
    ("bootstrap_type", "src/mojotrees/model_sparse.mojo"): (
        "BLOCKING",
        "the same guard twice, at :126 (train_gpu_sparse) and :209 "
        "(train_multiclass_gpu_sparse): `backend == GPU_DEVICE` and "
        "`bootstrap.enabled()`, raising because neither round loop draws a "
        "bundle, so the fit would be silently unsampled. Correct to refuse "
        "rather than train something the caller did not ask for; wrong to "
        "refuse a bundle nobody asked for. Same yield-rule build and the "
        "same merge-order constraint as the model.mojo row, which carries "
        "the full statement",
    ),
    ("bootstrap_type", "src/mojotrees/trainset.mojo"): (
        "BLOCKING",
        "TWO ARMS, TWO ANSWERS, and the more severe one has to be the "
        "verdict because a key carries one. **CPU arm: RESOLVED.** The "
        "defaulted-bundle-yields rule holds -- a bundle the MODE supplied "
        "yields on a path that cannot honor it, a bundle the user TYPED "
        "still refuses by name. Verified 2026-08-17 by fitting: "
        "MojoTreesClassifier(grow_policy='symmetrictree', ctr='off') on a "
        "three-class target succeeds with the defaulted MVS, where the "
        "reason before it predicted a raise. (ctr='off' only isolates the "
        "bootstrap question from the separate CTR refusal on that path.) "
        "**GPU arm: BLOCKING**, at :1744, which I did NOT reach with that "
        "fit because device='auto' resolved to the CPU -- so the fit that "
        "verified the CPU half is silent about this one, and I recorded "
        "RESOLVED on the strength of it. Corrected on f9's site "
        "verification, which I then read myself: the guard is `backend == "
        "GPU_DEVICE and bootstrap.enabled()`, and `enabled()` asks whether "
        "the bundle is ACTIVE, not whether anyone NAMED it, so the yield "
        "rule that holds on the CPU arm has no counterpart here. One of the "
        "four sites in that build; the model.mojo row carries the full "
        "statement and the merge-order constraint on lane/defaults-release. "
        "The old reason ALSO claimed the sparse arm as a bootstrap failure "
        "and that was wrong -- see the grow_policy row for this file",
    ),
    ("grow_policy", "src/mojotrees/trainset.mojo"): (
        "DIVERGENCE",
        "THE SPARSE PATH HAS NO SYMMETRIC GROWER. Split out of the "
        "bootstrap_type row for this file on 2026-08-17, where it had been "
        "recorded as a bootstrap failure and was not one. A sparse fit under "
        "the shipped default raises 'grow_policy=oblivious is not a frontier "
        "order and has no GrowthSchedule: a level is searched once and the "
        "winner is applied to every leaf of it' -- the bootstrap bundle never "
        "comes into it, and the defaulted MVS yields there exactly as it does "
        "on multiclass. Verified by fitting a CSR matrix under "
        "grow_policy='symmetrictree'. THE RULING (Andrew, 2026-08-17): sparse "
        "input under symmetrictree REFUSES, with the reason. It does NOT fall "
        "back to lossguide, because that would make the grow policy a "
        "function of the input CONTAINER: two fits of one configuration on "
        "one dataset, one dense and one sparse, would grow different trees "
        "and both would report success. A silent policy change is the cliff, "
        "and here the difference is the whole tree shape rather than a "
        "constant. Refusing is a divergence from CatBoost, which grows "
        "symmetric trees on sparse input; it is a divergence we can see",
    ),
    ("score_function", "src/mojotrees/gpu_split_search.mojo"): (
        "BLOCKING",
        "the oblivious level search raises on anything but L2, and earns it: "
        "a level's Cosine score is a ratio of two cross-leaf accumulators "
        "with one square root at the end, so summing per-leaf Cosine gains is "
        "not the Cosine of the level. No merge order fixes it; f9 is building "
        "oblivious-Cosine as its item (1)",
    ),
    ("grow_policy", "src/mojotrees/gpu_tree_tables.mojo"): (
        "BLOCKING",
        "the resident tree tables return TREE_RESIDENT_DEPTHWISE for anything "
        "that is not leafwise, and symmetrictree is the proposed default. "
        "Whether the oblivious device path goes through these tables at all "
        "is the first question f9's build lane answers",
    ),
    ("boosting_type", "src/mojotrees/ordered_boosting.mojo"): (
        "RESOLVED",
        "every refusal naming boosting_type in this file is about "
        "'ordered'. The proposed default is 'plain', which is what these "
        "paths already do",
    ),
    ("boosting_type", "src/mojotrees/train_gpu.mojo"): (
        "RESOLVED",
        "refuses boosting_type='ordered'; the default is 'plain'",
    ),
    ("boosting_type", "src/mojotrees/train_gpu_sparse.mojo"): (
        "RESOLVED",
        "refuses boosting_type='ordered'; the default is 'plain'",
    ),
    ("max_depth", "src/mojotrees/train_gpu.mojo"): (
        "DIVERGENCE",
        "the oblivious device path refuses max_depth above 6 by name, and the "
        "proposed default is exactly 6. So the default works and depth 7 "
        "SILENTLY LOSES THE ACCELERATOR: the fit succeeds on the CPU and "
        "nothing says the GPU declined. This is not a refusal the default "
        "trips, it is a cliff the default is standing on, and the two are "
        "different: a refusal is loud once, a cliff is quiet forever. "
        "Recorded here because this audit is the only thing that would catch "
        "a future default change walking off it",
    ),
    ("boosting_type", "src/mojotrees/boosting.mojo"): (
        "RESOLVED",
        "refuses boosting_type='ordered'; the default is 'plain'",
    ),
    # -- read 2026-08-16 by the review session, one site at a time ----------
    #
    # Two findings are worth reading before the rows, because they are not
    # "this refusal happens to fire" but "the proposed set cannot be stated".
    #
    # 1. THE SET CONTRADICTS ITSELF. params.mojo:1409-1428 refuses
    #    auto_learning_rate=true beside an explicit lambda_l2 and beside an
    #    explicit leaf_estimation_iterations, and it is right to: CatBoost
    #    gates its own derivation on l2_leaf_reg and leaf_estimation_iterations
    #    being UNSET (options_helper.cpp:279-280). The proposed set names all
    #    three. In CatBoost the auto rate would silently do nothing; here it
    #    raises. Pick the rate or pick l2=3, not both.
    # 2. CATBOOST'S HEADLINE COMBINATION IS REFUSED. The set is symmetric
    #    trees plus categorical handling (max_cat_to_onehot=2, CTRs on) plus
    #    random_strength=1. tree.mojo:1874 refuses oblivious beside any
    #    categorical feature, and split.mojo:778 refuses random_strength
    #    beside any categorical feature. Each is honest about why -- a level
    #    shares one split and a category partition's order comes from one
    #    node's own statistics -- and together they mean the default set
    #    raises on exactly the datasets CatBoost exists for.
    ("grow_policy", "src/mojotrees/tree.mojo"): (
        "BLOCKING",
        "tree.mojo:1874 refuses grow_policy=oblivious beside ANY categorical "
        "feature, and the set pairs symmetrictree with max_cat_to_onehot=2 "
        "and CTRs on, so CatBoost's own headline combination raises on the "
        "dense CPU path. What must be built is a level-shared categorical "
        "search: today the partition order comes from one node's gradient "
        "and hessian ratios inside find_best_categorical_split and there is "
        "no one order to share across a level. The other four sites in this "
        "file are fine at the default: :1859 wants max_depth > 0 and it is "
        "6, :1866 caps at OBLIVIOUS_MAX_DEPTH = 16, :1881 wants forced "
        "splits empty and they are, :1892 wants extra_trees and CEGB off "
        "and they are",
    ),
    ("grow_policy", "src/mojotrees/train_gpu.mojo"): (
        "BLOCKING",
        "train_gpu.mojo:1575 raises whenever _oblivious_route_reason is not "
        "OBLIVIOUS_OK -- the resident tables did not open, or the searcher "
        "has fewer records than oblivious_records_needed -- and it raises "
        "rather than falling back because no other GPU grower builds a "
        "symmetric tree. Same build as the gpu_tree_tables row above; f9 "
        "answers whether the oblivious device path goes through those tables "
        "at all",
    ),
    ("grow_policy", "src/mojotrees/growth_policy.mojo"): (
        "DIVERGENCE",
        "GrowthSchedule.__init__ (:645) refuses GROW_OBLIVIOUS on purpose, to "
        "stop growers that never implemented the mode -- tree_sparse, the "
        "three train_gpu loops, train_gpu_sparse -- from silently taking the "
        "depth-wise branch and growing an asymmetric tree. The dense CPU "
        "grower does not build a schedule; it runs _grow_oblivious_levels. "
        ":187 and :211 are the parser and the enum check, and both accept "
        "'symmetrictree' as an alias. Interim: symmetrictree on any path that "
        "constructs a GrowthSchedule resolves to depthwise, recorded as OUR "
        "divergence, never a raise",
    ),
    ("grow_policy", "src/mojotrees/distributed.mojo"): (
        "DIVERGENCE",
        "distributed.mojo:736 refuses any grow_policy the distributed grower "
        "does not implement, and it tracks no node depth to schedule levels "
        "by. Interim: a distributed fit resolves grow_policy to leafwise, "
        "recorded as OUR divergence",
    ),
    ("max_depth", "src/mojotrees/tree.mojo"): (
        "RESOLVED",
        "the two sites bound the oblivious depth: :1859 wants max_depth > 0 "
        "and :1866 refuses above OBLIVIOUS_MAX_DEPTH = 16. The default is 6, "
        "which is inside both, and :1859's own message names CatBoost's "
        "depth=6 as the case it expects",
    ),
    ("max_depth", "src/mojotrees/train_gpu.mojo"): (
        "BLOCKING",
        "same site as the grow_policy row for this file (:1575): the "
        "oblivious route is refused as a whole, so the depth never gets to "
        "matter. Tracked there, repeated here so a reader of max_depth alone "
        "does not conclude the GPU is fine",
    ),
    ("max_depth", "src/mojotrees/distributed.mojo"): (
        "DIVERGENCE",
        "distributed.mojo:701 refuses max_depth outright; the depth cap is "
        "not carried to the ranks. Interim: a distributed fit resolves "
        "max_depth to unset, which with leafwise above is self-consistent",
    ),
    ("n_estimators", "src/mojotrees/boosting.mojo"): (
        "RESOLVED",
        "the site is :2238, which is random_strength's per-tree scale "
        "refusal; it matched here only because its message names the round "
        "index. 360 trees cannot make a gradient standard deviation "
        "nonpositive. See the random_strength row for this file",
    ),
    ("n_estimators", "src/mojotrees/distributed.mojo"): (
        "RESOLVED",
        ":2352 fires only when a CALLBACK resets n_estimators, learning_rate "
        "or num_leaves mid-fit, because those are part of the configuration "
        "every rank agreed on. A default set is chosen before the fit starts "
        "and is not a callback reset",
    ),
    ("learning_rate", "src/mojotrees/boosting.mojo"): (
        "DIVERGENCE",
        ":2984 and :3878 refuse a learning_rate that differs from the one "
        "stored on the booster, which is what train_more and continued "
        "training compare. Under auto_learning_rate the rate is DERIVED per "
        "fit from row and feature counts, so a continued fit that re-derives "
        "on a different shape hits this. Interim: a continued fit reuses the "
        "booster's stored rate and never re-derives. :2238 is the "
        "random_strength scale, not this parameter",
    ),
    ("learning_rate", "src/mojotrees/boosting_rf.mojo"): (
        "RESOLVED",
        ":264 refuses any learning_rate other than RF_SHRINKAGE, and it is "
        "reached only in random-forest mode. The default boosting_type is "
        "'plain'",
    ),
    ("learning_rate", "src/mojotrees/distributed.mojo"): (
        "RESOLVED",
        "same :2352 callback-reset guard as the n_estimators row; a default "
        "is not a mid-fit reset",
    ),
    ("learning_rate", "src/mojotrees/split.mojo"): (
        "RESOLVED",
        ":760 guards on 'noisy and not (noise_stdev > 0.0)', which is "
        "random_strength's scale and not the learning rate; it matched "
        "because the message names both. See the random_strength row",
    ),
    ("learning_rate", "src/mojotrees/tree_parameters_extra.mojo"): (
        "RESOLVED",
        ":1913 guards on random_strength > 0 beside a zero random_score_scale; "
        "the learning rate appears in the message only. See the "
        "random_strength row for this file",
    ),
    ("auto_learning_rate", "src/mojotrees/params.mojo"): (
        "RESOLVED",
        "ALL THREE REFUSALS ARE ON THE PARAMETER-STRING SURFACE AND THE "
        "SHIPPED DEFAULT DOES NOT TRAVEL ON IT. They are in "
        "_enable_auto_learning_rate (:1854), called only from "
        "parse_params(spec: String) at :1584, and each guards on a saw_* "
        "flag that means 'this key appeared in the STRING': :1879 on "
        "saw_learning_rate, :1886 on saw_lambda_l2, :1893 on "
        "saw_leaf_estimation_iterations. They transcribe CatBoost's gating "
        "(options_helper.cpp:279, :280) correctly and they are right for the "
        "surface they are on -- a string that names both keys IS a caller "
        "naming both. The estimator does not build a string: it sends the "
        "value and its provenance as separate keys "
        "(auto_learning_rate_l2_set, leaf_estimation_iterations_set), which "
        "is how a mode-supplied value stays distinguishable from a "
        "user-named one, mirroring CatBoost's SetDefault assigning without "
        "raising IsSetFlag (option.h:27-33). VERIFIED 2026-08-17 by fitting "
        "rather than by reading: MojoTreesClassifier("
        "grow_policy='symmetrictree') at 100 and at 360 trees reports "
        "auto_learning_rate_note_ == 'auto_lr_gate_open' with mode_defaults_ "
        "== {'l2_leaf_reg': 3.0, 'learning_rate': 0.03, 'depth': 6}. The "
        "mode supplied l2_leaf_reg=3.0 and the gate stayed open, which is "
        "the whole claim. A user who NAMES either key still closes it and "
        "the note reads auto_lr_skipped:<key>. **The line numbers in the "
        "reason this replaces (:1409, :1416, :1423) had drifted and pointed "
        "at cegb_tradeoff, a closing paren and linear_lambda.** That is why "
        "this table is keyed by (parameter, file) and not by line, and it is "
        "an argument for citing the FUNCTION as well as the line, which "
        "these entries now do",
    ),
    ("lambda_l2", "src/mojotrees/params.mojo"): (
        "RESOLVED",
        "the same refusal seen from the other key, at :1886 inside "
        "_enable_auto_learning_rate, reached only from parse_params(spec: "
        "String). Recorded separately so a reader who greps lambda_l2 finds "
        "it rather than inferring it from the auto lr row",
    ),
    ("leaf_estimation_iterations", "src/mojotrees/params.mojo"): (
        "RESOLVED",
        ":1893, the third of the same string-surface trio, resolved for the "
        "same reason. Worth one extra note because this parameter's "
        "provenance is the least visible of the three: it is absent from "
        "mode_defaults_ ON PURPOSE, and sklearn.py says why -- it and "
        "random_strength resolve on the NATIVE side, where the entry point's "
        "routing is known, so a record written in Python would be a guess. "
        "What travels is leaf_estimation_iterations_set=0, which "
        "_parse_params resolves through "
        "boosting.catboost_leaf_estimation_iterations in CatBoost mode, "
        "mirroring CatBoost reading TOption::NotSet() rather than comparing "
        "against a default (option.h:80-85). A separate string-surface "
        "limitation is tracked in its own row below and is NOT this one",
    ),
    ("leaf_estimation_iterations", "PARAMETER STRING SURFACE"): (
        "DIVERGENCE",
        "params.mojo:1156 refuses leaf_estimation_iterations > 1 from a "
        "parameter string at all. Split out of the params.mojo row above on "
        "2026-08-17 because the two halves have different answers and one "
        "verdict cannot carry both. The shipped default is unaffected: it "
        "goes through the estimator, which sends the count and "
        "leaf_estimation_iterations_set as separate keys and lets the native "
        "side resolve an unset one per objective. A caller who builds a "
        "parameter string by hand gets 1 and a named refusal above it, which "
        "is honest and is a smaller surface than CatBoost's. It becomes "
        "BLOCKING the moment anything in the shipped path starts travelling "
        "as a parameter string. Not a file key, because the refusal is one "
        "site and the divergence is about a SURFACE",
    ),
    ("leaf_estimation_iterations", "src/mojotrees/boosting.mojo"): (
        "DIVERGENCE",
        ":1230 refuses it beside an objective that renews its leaves (l1, "
        "quantile, mape), whose leaf value is already the exact minimizer, "
        "and :1241 beside goss, which rescales derivatives after they are "
        "computed. :1176 names the trainers that do implement it. The set "
        "says '(per objective)', so this is the list the per-objective table "
        "must resolve to 1 on. Interim: 1 for the renewing objectives and "
        "under goss",
    ),
    ("leaf_estimation_iterations", "src/mojotrees/tree_parameters_extra.mojo"): (
        "RESOLVED",
        ":1964 refuses it beside path_smooth > 0, and path_smooth is not in "
        "the proposed set and defaults to 0",
    ),
    ("leaf_estimation_iterations", "bindings/_mojotrees.mojo"): (
        "RESOLVED",
        "the refusal fires on 'not leaf_estimation_ok'. Three of the fifteen "
        "_parse_params call sites pass True as of 2026-08-16, chosen by which "
        "trainer the entry point routes to: fit (plain fork), train_dataset "
        "(dense arms; the sparse arm has no such trainer) and booster_update "
        "(boosting.train_more). Before that date only fit passed it, which is "
        "why bench/real_data -- which trains through train_dataset -- ran "
        "every CatBoost-mode Logloss cell at one Newton step against "
        "CatBoost's ten. The earlier note here claimed every construction "
        "site passed True; that was never so",
    ),
    ("boost_from_average", "bindings/_mojotrees.mojo"): (
        "RESOLVED",
        "only false is refusable: true is what boosting._base_score has "
        "always done and is LightGBM's default (config.h:948). "
        "boost_from_average_ok is True at fit and train_dataset (dense arms) "
        "and nowhere else. It is a SEPARATE flag from leaf_estimation_ok, "
        "which it otherwise tracks, because booster_update reaches "
        "boosting.train_more: that loop reads leaf_estimation_iterations but "
        "starts from the model's stored base score and never calls "
        "boosting._base_score, so sharing one flag would have accepted a "
        "false and ignored it",
    ),
    ("boosting_type", "src/mojotrees/params.mojo"): (
        "RESOLVED",
        ":642 refuses 'dart', 'goss', 'rf' and their aliases from a parameter "
        "string because each needs a parameter bundle; the default is "
        "'plain', which needs none",
    ),
    ("boosting_type", "bindings/_mojotrees.mojo"): (
        "RESOLVED",
        "all three sites guard on an ordered bundle being enabled -- :909 on "
        "'ordered.enabled and not ordered_ok', :1209 on ordered beside "
        "dart/rf, :1218 on ordered beside linear_tree. The default is 'plain', "
        "so no ordered bundle exists",
    ),
    ("boosting_type", "python/mojotrees/sklearn.py"): (
        "RESOLVED",
        ":1791 and :2013 refuse ordered and goss beside ROW BAGGING, guarding "
        "on 'bagging_freq > 0 and bagging_fraction < 1.0'. The default leaves "
        "bagging off and carries its sampling rate as subsample under "
        "bootstrap_type=MVS, so neither guard is armed",
    ),
    ("bootstrap_type", "src/mojotrees/boosting.mojo"): (
        "DIVERGENCE",
        "MVS refuses to coexist with four things that also own the row list "
        "or the derivatives: bagging_fraction (:425, it IS the Bernoulli "
        "bootstrap under another name), goss (:432), pos/neg class bagging "
        "(:438), and a custom objective (:443, the draw would scale a "
        "callback's own derivatives). None is in the default set, so the "
        "default alone is clean. Interim: when a user sets any of the four, "
        "bootstrap_type resolves to No rather than raising, and that is a "
        "divergence to record",
    ),
    ("bootstrap_type", "src/mojotrees/sampling.mojo"): (
        "DIVERGENCE",
        ":1227 is mutual exclusion between the mvs and bayesian draws and the "
        "default picks one. :2052 and the bernoulli case are name checks that "
        "'MVS' passes. :1841 is the real one: it refuses the bundle for any "
        "entry point whose round loop never calls bootstrap_round, which is "
        "the sparse and multiclass gap already BLOCKING under trainset.mojo. "
        "Interim is that row's: No on those paths",
    ),
    ("bootstrap_type", "bindings/_mojotrees.mojo"): (
        "DIVERGENCE",
        ":1039 refuses subsample beside bootstrap_type='bayesian', since the "
        "Bayesian bootstrap reweights every row and has no fraction. The "
        "default is MVS with subsample=0.8, so it is clean -- but the set "
        "names Bayesian as the FALLBACK, and the fallback plus subsample=0.8 "
        "lands exactly here. This is why the trainset.mojo interim must be No "
        "and not Bayesian",
    ),
    ("bootstrap_type", "python/mojotrees/sklearn.py"): (
        "RESOLVED",
        ":1587 refuses bagging_fraction != 1.0 beside bootstrap_type and its "
        "own message says to use subsample for the MVS rate, which is what "
        "the default does, so bagging_fraction stays 1.0. :1596 refuses a "
        "bagging schedule and none is set. :1561 rejects 'poisson' by name. "
        ":1623 is the Bayesian-plus-subsample case covered in the bindings row",
    ),
    ("subsample", "src/mojotrees/sampling.mojo"): (
        "RESOLVED",
        ":2012 fires on 'len(kept) == 0 and n_rows > 0', which is an outcome "
        "of the draw and not a setting: it says the MVS draw kept no rows at "
        "all. If a rate of 0.8 can empty a nonempty dataset that is a sampler "
        "defect to fix in place, not a collision the default walks into",
    ),
    ("subsample", "bindings/_mojotrees.mojo"): (
        "RESOLVED",
        "same :1039 as the bootstrap_type row: it is armed only under "
        "bootstrap_type='bayesian', and the default is MVS",
    ),
    ("subsample", "python/mojotrees/sklearn.py"): (
        "RESOLVED",
        ":1623 is the Bayesian guard, :1587 the bagging_fraction guard, and "
        ":1791/:2013 the ordered-and-goss-beside-bagging guards. The default "
        "is MVS with bagging off, so none is armed. The Bayesian fallback "
        "caveat is recorded on the bootstrap_type bindings row",
    ),
    ("random_strength", "src/mojotrees/split.mojo"): (
        "BLOCKING",
        ":778 refuses random_strength beside ANY categorical feature: a "
        "categorical candidate is a category SET searched inside "
        "find_best_categorical_split, and only that search's winner reaches "
        "the loop that would add the noise, so there is nothing to perturb "
        "per candidate. The set pairs random_strength=1.0 with CTRs on and "
        "max_cat_to_onehot=2. What must be built is the noise draw inside the "
        "categorical search. :760 and :1417 are the scale guards, fine on the "
        "dense CPU path",
    ),
    ("random_strength", "src/mojotrees/gpu_split_search.mojo"): (
        "BLOCKING",
        "the same categorical hole on the device, at :5155 ('cat_n[f] >= 2'), "
        "plus :1198, which needs a nonnegative node id and so excludes any "
        "grower that cannot supply one. Same build as the split.mojo row, "
        "different backend, and f9 owns this file",
    ),
    ("random_strength", "src/mojotrees/boosting.mojo"): (
        "RESOLVED",
        ":2230 and :2238 refuse a positive random_strength beside a "
        "nonpositive per-tree scale, and the dense CPU trainer COMPUTES that "
        "scale every round at _round_random_score_scale (:2192). The raise is "
        "reachable only when random_score_scale_from_gradients returns exactly "
        "zero, which is a degenerate-gradient condition and not a setting",
    ),
    ("random_strength", "src/mojotrees/tree_parameters_extra.mojo"): (
        "RESOLVED",
        ":1892 and :1913 refuse a positive random_strength on a bundle whose "
        "scale is still zero AND whose caller did not declare that a trainer "
        "computes one per tree. The dense CPU trainer declares it; see the "
        "boosting.mojo row",
    ),
    ("random_strength", "bindings/_mojotrees.mojo"): (
        "DIVERGENCE",
        ":803 fires on 'random_strength > 0.0 and not random_strength_ok', "
        "and random_strength_ok is literally 'device == CPU_DEVICE' (:1191). "
        "So the default raises on every device='gpu' fit through this "
        "binding. :1234 and :1244 add dart/rf and linear_tree, neither in the "
        "set. Interim: random_strength resolves to 0 on the GPU, recorded as "
        "OUR divergence, until the categorical draw above lands",
    ),
    ("score_function", "bindings/_mojotrees.mojo"): (
        "RESOLVED",
        ":824 fires on 'score_function != SCORE_L2 and not "
        "score_function_ok', and every construction site in this file passes "
        "score_function_ok=True (:1193, :1431, :1539, :1786, :1859, :2055, "
        ":2101). The oblivious-Cosine hole is real but it is on the device, "
        "recorded under gpu_split_search.mojo above",
    ),
    ("max_ctr_complexity", "src/mojotrees/params.mojo"): (
        "RESOLVED",
        ":1108 guards on 'complexity != 1' and the proposed default is 1",
    ),
    ("max_ctr_complexity", "python/mojotrees/sklearn.py"): (
        "RESOLVED",
        ":1432 and :1448 both guard on the value being other than 1, and the "
        "proposed default is 1. Note this is the parameter refused because "
        "ctr_combinations.mojo is unreachable, so 1 is the only value the "
        "tree can honor today",
    ),
    ("min_data_in_leaf", "src/mojotrees/tree_parameters_extra.mojo"): (
        "RESOLVED",
        ":1494 is check_feature_pre_filter, which is about whether a "
        "parameter STRING may ask for feature_pre_filter=true; it matched "
        "only because its prose names min_data_in_leaf as the filter's "
        "threshold. :1847 is the random_strength bundle guard. Neither reads "
        "the value, and min_data_in_leaf=1 arms nothing",
    ),
}

#: Words whose presence in a raise message means it is a refusal rather than a
#: validation error. Not used to filter, only to label, because the difference
#: matters to a reader and guessing it wrong should not hide a row.
REFUSAL_WORDS = (
    "not implemented",
    "not honored",
    "not reachable",
    "not supported",
    "cannot",
    "refus",
    "does not",
    # Added 2026-08-16 after two real blockers went INVISIBLE to --check.
    # `params.mojo` refuses auto_learning_rate beside an explicit lambda_l2
    # with "contradict each other", and beside leaf_estimation_iterations
    # with "would do nothing". Both raises were still there; both stopped
    # being COUNTED, because this list decides what looks like a refusal and
    # neither phrasing was in it.
    #
    # **A shrinking blocking count is not evidence of progress when the
    # filter that produces it can lose rows to a rephrasing.** That is the
    # third time today a filter here reported its own blind spot as a result,
    # and the only reason this one surfaced is that a lane read every entry
    # by hand and noticed two it expected were missing. Nobody can run that
    # check against a list of things they do not already know.
    "contradict",
    "would do nothing",
    "would be ignored",
    "has no effect",
    "is inert",
)


def _sources():
    for directory in SEARCH_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith((".mojo", ".py")):
                yield os.path.join(directory, name)


def _raise_blocks(text):
    """(line, message) for each raise, message flattened to one line.

    Mojo's `raise Error(` takes a comma-separated list of parts that are
    concatenated, and Python's `raise ValueError(` takes implicit string
    concatenation across lines. Both flatten the same way for this purpose:
    pull every double-quoted run out of the argument list and join them.
    """
    out = []
    for match in re.finditer(r"raise\s+\w*Error\(", text):
        # Depth counting must SKIP string literals. A message containing
        # "(0, 1]" is common in this repository's range checks, and counting
        # the paren inside it ran the scan past the closing paren and merged
        # several unrelated raises plus the following docstring into one row.
        # The first version of this tool did exactly that and its output was
        # unreadable in a way that looked like the source was unreadable.
        depth, i, quote = 0, match.end() - 1, ""
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if text.startswith(quote, i):
                    i += len(quote)
                    quote = ""
                    continue
            elif text.startswith('"""', i):
                quote = '"""'
                i += 3
                continue
            elif ch == '"':
                quote = '"'
                i += 1
                continue
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[match.end() : i]
        parts = re.findall(r'"([^"]*)"', body)
        if not parts:
            continue
        line = text.count("\n", 0, match.start()) + 1
        out.append((line, " ".join(p.strip() for p in parts if p.strip())))
    return out


#: Keys of `catboost_defaults` that CatBoost mode SUPPLIES rather than the
#: caller naming, and which therefore travel with `IsSet` false.
#:
#: **This is the audit's copy of the rule the gate itself runs on, and it
#: exists because the audit could not otherwise tell the two apart.** A
#: refusal that reads whether a key was NAMED does not fire on a value the
#: mode supplied. `DEFAULT_SETS` is a flat dict of values with no provenance,
#: so the audit reads "the set names lambda_l2=3.0" where the truth is "the
#: mode supplies lambda_l2=3.0 without recording that anyone asked for it".
#:
#: **This annotates; it does not rule.** The verdicts in `ACKNOWLEDGED` are
#: human, and this flag only says which QUESTION a site is asking, so a
#: reader can see at a glance which collisions are about a value and which
#: are about provenance. It flagged 20 sites on 2026-08-17, and the ones it
#: did NOT flag are the more interesting half: a message that does not use
#: one of `PROVENANCE_WORDS` is treated as reading the value, which keeps a
#: collision visible rather than explaining it away.
#:
#: CatBoost's own mechanism is `SetDefault`, which assigns a value and never
#: raises `IsSetFlag` (`option.h:27-33`), read by `TOption::NotSet()`
#: (`option.h:80-85`). Ours is `_Base.mode_defaults_` plus the `*_set`
#: provenance keys in `python/mojotrees/sklearn.py`.
#:
#: A second copy of a rule is a liability and this one is worth its cost only
#: because it is DATA and one line long per key. It is checkable against the
#: estimator in a single fit: a fit whose `auto_learning_rate_note_` reads
#: `auto_lr_gate_open` while `mode_defaults_` carries a key proves that key
#: belongs here. Verified that way on 2026-08-17 for `lambda_l2`.
MODE_SUPPLIED = {
    "catboost_defaults": frozenset(
        {
            "lambda_l2",
            "learning_rate",
            "auto_learning_rate",
            "max_depth",
            "leaf_estimation_iterations",
            "random_strength",
            "bootstrap_type",
            "subsample",
            "score_function",
        }
    ),
}


def audit(default_set):
    params = DEFAULT_SETS[default_set]
    rows = {name: [] for name in params}
    for path in _sources():
        try:
            text = open(path, errors="ignore").read()
        except OSError:
            continue
        blocks = _raise_blocks(text)
        for name in params:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b")
            for line, message in blocks:
                if pattern.search(message):
                    rows[name].append(
                        {
                            "file": os.path.relpath(path, ROOT),
                            "line": line,
                            "message": message,
                            "looks_like_refusal": any(
                                w in message.lower() for w in REFUSAL_WORDS
                            ),
                            # True when the refusal reads whether a key was
                            # NAMED and this key is one the mode supplies, so
                            # the site cannot fire on the shipped default.
                            # It does not soften the verdict on its own: a
                            # human still rules, and this only says which
                            # question the site is asking. See MODE_SUPPLIED.
                            "provenance_gated": (
                                name
                                in MODE_SUPPLIED.get(default_set, frozenset())
                                and _reads_provenance(message)
                            ),
                        }
                    )
    return params, rows


#: A refusal that reads whether a key was NAMED rather than what its value is.
#: Deliberately narrow: these are the words our own refusals use for the
#: distinction, and a site that does not use one of them is treated as reading
#: the value, which is the answer that keeps a collision visible. Guessing
#: wide here would silence real collisions, and silence is the direction that
#: reads as good news.
PROVENANCE_WORDS = (
    "explicit",
    "explicitly",
    "named",
    "was set",
    "is set",
    "beside an explicit",
)


def _reads_provenance(message):
    low = message.lower()
    return any(w in low for w in PROVENANCE_WORDS)


#: Files that only a fit which has already been routed to the accelerator
#: reaches. A refusal here cannot fire while a policy block is sending the
#: fit to the CPU, which is the whole point of the check below.
DEVICE_FILES = (
    "src/mojotrees/train_gpu.mojo",
    "src/mojotrees/train_gpu_sparse.mojo",
    "src/mojotrees/gpu_split_search.mojo",
    "src/mojotrees/gpu_resident_round.mojo",
    "src/mojotrees/gpu_tree_tables.mojo",
    "src/mojotrees/gpu_active_rows.mojo",
)


def unmasked_by_removing(default_set, parameter):
    """What a proposed default set still walks into once one policy block is
    removed. Returns `(device_rows, unrouted)`.

    **The insight this encodes is not mine and is worth stating before the
    code.** A `BLOCK_*` in `device_policy` does two jobs at once. It refuses a
    configuration, and it is **the only thing making `auto` route to the CPU
    instead of failing.** So retiring a block because its capability finally
    landed **silently withdraws the fallback for every other reason that fit
    would still have been refused for.** The removal is correct, scheduled,
    and creates a cliff.

    The instance this was written for, now CLOSED, kept because the shape
    recurs. The proposed default set is `score_function=cosine` **and**
    `random_strength=1`. `BLOCK_SCORE_FUNCTION` fired, `auto` fell back to the
    CPU, and the fit ran. Removing that block on schedule when the device
    Cosine kernel landed would have left the policy nothing to see: `auto`
    selects the GPU on shape and the fit **raises in the grower**, because
    `ExtraTreeParams.is_active()` carries `random_strength > 0.0` while
    `device_policy` then contained zero occurrences of `random_strength`. A
    capability landing turning a working default into a raising one.

    `BLOCK_RANDOM_STRENGTH` closed it, and `BLOCK_GROW_POLICY` /
    `BLOCK_MAX_DEPTH` closed the two the same query surfaced next. **Do not
    read the paragraph above as current state** -- it is the worked example,
    and this docstring asserting a live `zero occurrences` was itself stale
    within a day of being written, which is the failure this whole tool is
    pointed at.

    What generalizes is the query, not the example: before retiring any
    block, ask which refusals the same fit still walks into. The answer
    changes every time a lane lands a capability, so it is a question to
    re-run rather than an answer to record.

    So the two halves reported here are:

    `device_rows`   refusals in device-only files, for OTHER parameters of the
                    set, which the removed block was masking.
    `unrouted`      parameters of the set that a device path refuses but that
                    `device_policy` never names, so no block can route around
                    them. **These are the dangerous ones**: a masked refusal
                    at least has a block somewhere, and an unrouted one has
                    nothing that could ever have routed it.

    Static and name-based, like the rest of this file. It reports where to
    read; it does not prove a fit reaches any of these.

    **It reads the WORKING TREE, not HEAD, and that is a hazard rather than a
    convenience.** The first time this function ran it reported
    `random_strength` as safely masked, contradicting a verified finding from
    the other campaign that `device_policy` contained zero occurrences of it.
    Both were right: the finding was true of HEAD, and by the time this ran
    there were 76 uncommitted lines adding `BLOCK_RANDOM_STRENGTH` in the
    shared checkout, written by the session that had made the finding.

    So on a shared checkout this tool can report another session's half-written
    work as landed, in the direction that says a cliff is already guarded when
    it is not. **When the answer matters, run it against a clean tree or check
    `git status` first.** The failure is silent and it looks like good news,
    which is the combination worth naming.
    """
    params, rows = audit(default_set)
    policy = _read_source("src/mojotrees/device_policy.mojo")
    device_rows, unrouted, excused = [], [], []
    for name in sorted(rows):
        if name == parameter:
            continue
        hits, skipped = [], []
        for r in rows[name]:
            if not (r["looks_like_refusal"] and r["file"] in DEVICE_FILES):
                continue
            # A site a human has read and ruled RESOLVED does not hold the
            # gate shut. **RESOLVED only** -- BLOCKING and DIVERGENCE still
            # count, because those say the refusal fires and the default is
            # not ready, which is exactly what this gate is for.
            #
            # WHY THIS IS NOT A SUPPRESSION HATCH, which is the obvious
            # objection and a fair one. Without it the gate can never go
            # green: `bootstrap_type` is a permanent UNROUTED entry because
            # the raise it matches is a defensive guard inside the
            # device-gradient loop that the configuration never reaches, and
            # a rule of the form "retire the block when this exits 0" is
            # unusable against a check that cannot reach 0. A gate that can
            # never pass gets bypassed, and then it gates nothing.
            #
            # The protections are that only RESOLVED excuses, that each
            # excused site is PRINTED with its verdict and reason rather than
            # dropped, and that an acknowledgement is an argument somebody
            # wrote and signed rather than a name on a skip list.
            verdict = ACKNOWLEDGED.get((name, r["file"]))
            if verdict is not None and verdict[0] == "RESOLVED":
                skipped.append((r, verdict))
                continue
            hits.append(r)
        if skipped:
            excused.append((name, skipped))
        if not hits:
            continue
        device_rows.append((name, hits))
        if not re.search(r"\b" + re.escape(name) + r"\b", policy):
            unrouted.append(name)
    return device_rows, unrouted, excused


def _tree_state():
    """What this answer was computed from, in one line, for the reassuring
    answers only. Reads the WORKING TREE, so an uncommitted edit in a shared
    checkout is part of the answer whether or not the reader expects it."""
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "bindings", "python"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "  (tree state unavailable; this answer was read from the working tree)"
    if not dirty:
        return f"  Read from the working tree at {head}, clean."
    # Split rather than slice a fixed offset. Porcelain's status field is two
    # characters plus a space, but the first version sliced [3:] and printed
    # "rc/mojotrees/device_policy.mojo" for a real path. A line whose entire
    # job is to be trusted cannot be one character wrong.
    files = ", ".join(sorted(line.split()[-1] for line in dirty)[:4])
    return (
        f"  Read from the working tree at {head}, with {len(dirty)} "
        f"UNCOMMITTED source file(s): {files}.\n"
        "  A reassuring answer computed over uncommitted work is the failure "
        "this line exists for."
    )


def _read_source(rel):
    try:
        return open(os.path.join(ROOT, rel), errors="ignore").read()
    except OSError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--set", default="catboost_defaults", choices=sorted(DEFAULT_SETS))
    parser.add_argument(
        "--removing",
        metavar="PARAMETER",
        help=(
            "name the parameter whose device_policy block is about to be "
            "removed; reports what the default set still walks into once the "
            "fallback that block provided is gone"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="gated mode: exit 1 on an unacknowledged or BLOCKING collision",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include raises that do not read as refusals",
    )
    args = parser.parse_args(argv)

    if args.removing:
        device_rows, unrouted, excused = unmasked_by_removing(
            args.set, args.removing
        )
        print(f"Removing the block for {args.removing!r} unmasks:")
        print()
        # Printed BEFORE the findings, not appended as a footnote. A gate that
        # quietly drops sites reads as "nothing here" whether it found nothing
        # or excused everything, and those are different answers.
        for name, skipped in excused:
            for site, verdict in skipped:
                print(
                    f"  EXCUSED  {name} at {site['file']}:{site['line']}"
                    f" -- acknowledged {verdict[0]}"
                )
                print(f"      {verdict[1][:200]}")
                print()
        for name, hits in device_rows:
            mark = "  UNROUTED" if name in unrouted else "  masked  "
            print(f"{mark} {name}")
            for h in hits[:3]:
                print(f"      {h['file']}:{h['line']}")
                print(f"        {h['message'][:130]}")
            print()
        if unrouted:
            print(
                "UNROUTED means device_policy never names the parameter, so "
                "no block\ncould route around it. After the removal, auto "
                "selects the accelerator\non shape and the fit raises in the "
                "grower.\n"
                "\n"
                "THESE ARE LEADS, NOT A WORK LIST. Read each one before "
                "writing a block\nfor it. This is a static name match: it "
                "cannot tell a refusal the\nconfiguration REACHES from a "
                "defensive raise it never arrives at, and it\ncannot see a "
                "parameter that is honored by routing to a different arm.\n"
                "On 2026-08-17 this list held three entries and one of them "
                "was a working\npath -- bootstrap_type, where MVS is honored "
                "by taking the host-gradient\narm. Writing that block would "
                "have refused a fit that trains today.\n"
                "Confirm each one against the code that would have to read "
                "it, then add\nblocks for the survivors BEFORE removing the "
                "one you named."
            )
        elif not device_rows:
            print("  nothing: the set walks into no device refusal without it")
        # Print the tree state whenever the answer is reassuring, and only
        # then. "Safely masked" and "nothing" are the two answers a reader
        # acts on by NOT looking further, so they are the two that must carry
        # what they were computed from. This function has already reported a
        # cliff as guarded because another session's 76 uncommitted lines were
        # sitting in the shared checkout; under-reporting nags until somebody
        # looks, and over-reporting sends them away satisfied.
        if not unrouted:
            print()
            print(_tree_state())
        return 1 if unrouted else 0

    params, rows = audit(args.set)

    if args.check:
        unreviewed, blocking = [], []
        for name, sites in sorted(rows.items()):
            for site in sites:
                if not site["looks_like_refusal"]:
                    continue
                verdict = ACKNOWLEDGED.get((name, site["file"]))
                if verdict is None:
                    unreviewed.append((name, site))
                elif verdict[0] == "BLOCKING":
                    blocking.append((name, site, verdict[1]))
        for name, site in unreviewed:
            print(
                f"UNREVIEWED {name} -> {site['file']}:{site['line']}\n"
                f"    {site['message'][:160]}\n"
                f"    Read it, then add ({name!r}, {site['file']!r}) to "
                "ACKNOWLEDGED as RESOLVED, DIVERGENCE or BLOCKING with the "
                "reason."
            )
        seen = set()
        for name, site, why in blocking:
            if (name, site["file"]) in seen:
                continue
            seen.add((name, site["file"]))
            print(f"BLOCKING {name} -> {site['file']}: {why}")
        total = len(unreviewed) + len(seen)
        if total == 0:
            print(f"default set '{args.set}': no unreviewed or blocking collisions")
            return 0
        print()
        print(
            f"{len(unreviewed)} unreviewed, {len(seen)} blocking. A BLOCKING "
            "entry fails on purpose: it is a collision somebody has read and "
            "not yet fixed, and the default cannot ship over it."
        )
        return 1

    if args.json:
        print(json.dumps({"set": args.set, "params": params, "sites": rows}, indent=2))
        return 0

    print(f"Refusals a default set of '{args.set}' would walk into")
    print("=" * 52)
    print()
    total = 0
    for name in sorted(rows):
        sites = rows[name]
        if not args.all:
            sites = [s for s in sites if s["looks_like_refusal"]]
        if not sites:
            continue
        print(f"  {name} = {params[name]!r}   ({len(sites)} site(s))")
        for site in sites[:6]:
            print(f"      {site['file']}:{site['line']}")
            print(f"        {site['message'][:150]}")
        if len(sites) > 6:
            print(f"      ... and {len(sites) - 6} more")
        print()
        total += len(sites)
    quiet = [n for n in sorted(rows) if not rows[n]]
    if quiet:
        print("  No raise names these at all: " + ", ".join(quiet))
        print()
    print(f"{total} site(s) to read.")
    print()
    print("Every row is a site to READ, not a prediction. This does not")
    print("evaluate conditions: it cannot tell you a refusal fires only on the")
    print("sparse path or only above a row count. A default set is compatible")
    print("with the tree when a person has read these and said so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
