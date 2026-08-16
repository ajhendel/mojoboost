"""The scikit-learn estimators: `MojoTreesRegressor`, `MojoTreesClassifier`,
`MojoTreesRanker`, and their shared base `_Base`.

Moved here from the package `__init__` in the consolidation round, the way
`lightgbm.sklearn` holds LightGBM's. Every name below, private ones
included, is bound back into the package namespace by `__init__.py`, so
`mojotrees.MojoTreesRegressor`, `from mojotrees import _SQUARED_ERROR`
(basic.py), and pickles that name `mojotrees.MojoTreesRegressor` are
unchanged. tools/api_snapshot.py reads the classes and the literals below
from this file when they are not in `__init__.py`.
"""

import array as _array
import json as _json
import os as _os
import tempfile as _tempfile
import warnings as _warnings

from . import _arrays, _compat, _eval, _validation, callback as _callback
from . import _multi_target
from . import preflight as _preflight
from ._sklearn import NotFittedError, ParamsMixin as _ParamsMixin
from ._sklearn import estimator_tags as _estimator_tags
from .basic import Booster

_mojotrees = _compat.import_extension()
_np = _arrays.np
_addr = _arrays.addr
_out_buffer = _arrays.out_buffer
_finish = _arrays.finish

_SQUARED_ERROR = 0
_BINARY_LOGISTIC = 1
_POISSON = 2
_HUBER = 3
_QUANTILE = 4
_L1 = 5
_CUSTOM = 6
_LAMBDARANK = 7
_GAMMA = 8
_TWEEDIE = 9
_MAPE = 10
_FAIR = 11
_CROSS_ENTROPY = 12

#: `objective_registry.MULTICLASS`. Softmax is negative on purpose, to stay
#: out of the single-output code space forever, and it is a *code*, not an
#: absence: `objective_code_of_name("multiclass")` returns it and
#: `_normalized_objective` in src/mojotrees/device_policy.mojo preserves it
#: while folding everything below it into `OBJECTIVE_UNSPECIFIED` (-2). The
#: two negatives are one apart and mean opposite things, so nothing in this
#: package may test an objective code for negativity, for truthiness, or
#: for `None`-ness and expect to have distinguished them.
_MULTICLASS = -1

#: LightGBM's stock `learning_rate`, and the value an unset one resolves to.
#:
#: A named constant rather than four repeated literals because the
#: constructor no longer carries it: `learning_rate` defaults to `None` so
#: that "the user did not name it" survives all the way to CatBoost's gate,
#: which fires only on an UNSET rate (`options_helper.cpp:277`). A signature
#: default of 0.1 cannot say that -- `0.1` typed by a user and `0.1` supplied
#: by the signature are the same float by the time anything can look -- and
#: `tools/check_parity.py` reads this constant instead, the way it already
#: reads `_LAMBDA_L2`.
_LEARNING_RATE = 0.1

# Defaults of the two regularization parameters, named so the constructor
# signature and the alias resolution in `_params` cannot drift apart.
_LAMBDA_L1 = 0.0
_LAMBDA_L2 = 0.0
"""LightGBM's stock `lambda_l2`.

This literal, not `TreeParams.default()`, is what a Python fit resolves an
unset `lambda_l2` from, and it is therefore what `bench/real_data` fits.
It read 1.0 until 2026-08-16, so the arm labelled `stock+det` was running a
non-stock regularizer on our side while the comparator ran LightGBM's 0.0.
Three independent literals carried the old value; `tools/check_parity.py`
now asserts every stock default across all of them.
"""

from ._fit_args import (
    _BOOSTING_TYPES,
    _DEVICES,
    _GROW_POLICIES,
    _IMPORTANCE_TYPES,
    _NO_DEVICE_PREDICT,
    _as_iteration,
    _check_eval_arguments,
    _device_name,
    _early_stopping_rounds,
    _encode_like,
    _eval_pairs,
    _metric_specs,
    _per_set,
    _primary_index,
    _store_vector,
    _unimplemented_objective_note,
    _objective_status,
    _objective_code_of_name,
    _check_objective_param,
)
from ._ranking import _check_relevance, _group_buffer, ndcg_score

#: "no value was folded in yet" for `_Base._resolve_alias`, which cannot use
#: `None` for that: `None` is exactly what an unset alias is.
_UNSET = object()

#: `grow_policy` spellings this estimator accepts and the canonical value
#: each resolves to.
#:
#: The canonical values are XGBoost's and CatBoost's words for the three
#: growths, per the `grow_policy` row of docs/PARAMETER_NAMING.md:
#: `lossguide` (best gain anywhere, LightGBM's growth and the default),
#: `depthwise` (a level at a time), and `symmetrictree` (one split per
#: level, shared by every node at that level). `leafwise`, `oblivious` and
#: `symmetric` are accepted aliases, and "oblivious" stays the word for the
#: *shape* in prose.
#:
#: **Why this table exists rather than `_fit_args._GROW_POLICIES`.** That
#: one carries the two frontier orders and nothing else, so
#: `growth_policy.GROW_OBLIVIOUS` -- which shipped in the Mojo package and
#: which `src/mojotrees/tree.mojo` grows -- could not be asked for through
#: the estimator at all. Every `bench/real_data` arm and every
#: scikit-learn user goes through this validator, so a policy that was
#: built, tested and merged was absent from the API.
#:
#: The value here is what `_params` sends to the native layer.
#: `growth_policy.parse_grow_policy` accepts all three words and is the one
#: resolver every fit entry point reaches, through `_parse_params` in
#: `bindings/_mojotrees.mojo`; `lossguide` reaches `GROW_LEAFWISE`,
#: `depthwise` reaches `GROW_DEPTHWISE`, and `symmetrictree` reaches
#: `GROW_OBLIVIOUS`. The dense CPU grower honors the last of those and the
#: sparse and GPU growers refuse it by name from `GrowthSchedule.__init__`,
#: so there is no path on which it is accepted and dropped.
_CANONICAL_GROW_POLICIES = {
    "lossguide": "lossguide",
    "leafwise": "lossguide",
    "leaf_wise": "lossguide",
    "depthwise": "depthwise",
    "depth_wise": "depthwise",
    "symmetrictree": "symmetrictree",
    "symmetric_tree": "symmetrictree",
    "symmetric": "symmetrictree",
    "oblivious": "symmetrictree",
}


def _score_function_code(value):
    """`score_function` as the integer the device policy gates on.

    Only the device decision reads this. The grower takes the name through
    its own path, and `check_score_function` in split.mojo is what refuses an
    unknown one; this function's whole job is to answer, before a backend has
    been chosen, "is this the one functional the accelerator implements".

    **It fails closed, and that is the design.** Anything that is not
    recognizably `L2` maps to `SCORE_COSINE`, which the policy blocks, rather
    than to `SCORE_L2`, which it allows. A selector this function has not been
    taught therefore keeps the fit on the CPU, where every selector is
    implemented, instead of reaching a device that computes `G^2/(H+lambda)`
    whatever it was asked for. Mapping the unknown case to L2 would be the
    silent-wrong-answer defect the block exists to close, reintroduced one
    layer up.
    """
    from .device_selection import SCORE_COSINE, SCORE_L2

    if value is None:
        return SCORE_L2
    if str(value).strip().lower() == "l2":
        return SCORE_L2
    return SCORE_COSINE


class _Base(_ParamsMixin):
    """Shared hyperparameters, mojotrees defaults (LightGBM-matched).

    `importance_type` selects what `feature_importances_` reports: "split",
    LightGBM's default, counts the nodes that split on each feature, and
    "gain" sums the gain those splits earned.

    **One canonical name per parameter, every vendor's name accepted.**
    The canonical names are the table in docs/PARAMETER_NAMING.md: one name
    per parameter, always a name that LightGBM, XGBoost, CatBoost or
    scikit-learn already uses, chosen for being the clearest of the four.
    Every other vendor's spelling of the same parameter is an accepted
    alias, so a LightGBM, XGBoost or CatBoost script runs through this
    estimator unchanged -- which is the point, because retyping a
    configuration is where two "identical" configurations quietly stop
    being identical.

    The canonical name is what this docstring and every error message use.
    It is not always the member that holds the default: LightGBM's spelling
    is kept on the wire (the native layer, the model files, and
    `tools/check_parity.py`), so `max_leaves` resolves onto `num_leaves`,
    `min_child_weight` onto `min_child_hess`, and `subsample` onto
    `bagging_fraction`. Either spelling works, and setting two spellings of
    one parameter to different non-default values raises, where LightGBM
    warns and keeps one.

    Value strings are case insensitive throughout, so CatBoost's
    `SymmetricTree`, `RMSE` and `Plain` arrive as written.

    `max_leaves` (LightGBM's `num_leaves`) bounds the leaves of a tree.
    `max_depth` (CatBoost's `depth`) bounds the depth of any leaf, counted
    in edges from the root, so `max_depth=1` gives stumps; `max_depth<=0`
    (the default, -1) means unlimited. Under the default growth a
    depth-bounded tree is still unbalanced and usually has fewer than
    `2**max_depth` leaves.

    `grow_policy` is XGBoost's and CatBoost's parameter of that name
    (LightGBM has no equivalent) and takes three values, case insensitively:

    - `"lossguide"`, the default, splits the leaf with the largest gain
      anywhere in the tree next. This is LightGBM's growth, and `leafwise`
      is accepted as an alias.
    - `"depthwise"` splits every leaf at one depth before any deeper one, so
      a tree fills level by level and is balanced. `num_leaves` stays a hard
      bound: a level that would overrun it is admitted as its highest-gain
      prefix, so at the default `num_leaves=31` and unlimited `max_depth` a
      depth-wise tree fills four levels (16 leaves) and half of a fifth. Set
      `max_depth` deliberately for depth-wise runs; a leaf-wise
      configuration is not a sensible one to inherit.
    - `"symmetrictree"` grows oblivious trees, CatBoost's default and its
      word for them: one split is chosen per level and every node at that
      level uses it, so the tree is symmetric by construction. `max_depth`
      is REQUIRED there and is the only bound on the tree's size --
      `num_leaves` does not bind, because a level splits entirely or not at
      all. `oblivious` and `symmetric` are accepted as aliases, and
      "oblivious" stays the word for the shape in prose.

    Depth-wise growth is honored on the CPU and GPU trainers alike (dense
    and sparse); the distributed prototype rejects it. Symmetric growth is
    honored by the dense CPU grower; the sparse and GPU growers refuse it by
    name rather than growing a tree that is not symmetric.

    `subsample` and `subsample_freq` (LightGBM's `bagging_fraction` and
    `bagging_freq`) are row bagging: every `subsample_freq` rounds, each row
    is kept independently with probability `subsample` and the trees of the
    following rounds are grown on that sample. `subsample=1` disables it.

    **`subsample < 1` implies `subsample_freq = 1`** unless `subsample_freq`
    was set explicitly. LightGBM leaves the frequency at 0 there, which
    makes `subsample=0.8` alone a silent no-op: no bagging happens and
    nothing says so. That is a defect and it is not copied. An explicit
    `subsample_freq=0` still disables bagging, because that is a statement
    rather than an omission.

    `bagging_seed` makes a run reproducible; the same seed and data give the
    same model on CPU and GPU alike. `random_state` (LightGBM's `seed`,
    CatBoost's `random_seed`) sets every per-component seed at once, leaving
    any seed you named yourself alone.

    `bootstrap_type` is CatBoost's parameter of that name and selects a
    different row sampler: `"MVS"` is Minimal Variance Sampling, CatBoost's
    own CPU default, which keeps large-gradient rows certainly and small ones
    with probability proportional to their gradient magnitude, weighting each
    survivor by the inverse of that probability; `"Bayesian"` keeps every row
    and reweights it by `bagging_temperature`; `"No"` is what an unbagged fit
    already is. `"Bernoulli"` is refused by name because it is row bagging
    under another name and is `subsample` with `subsample_freq` here, and
    `"Poisson"` is refused because CatBoost itself refuses it on the CPU.

    **`subsample` means the MVS rate under `bootstrap_type="MVS"`**, which is
    CatBoost's own contract, and row bagging is off there: `bootstrap_type`
    and `bagging_fraction` are two values of one CatBoost enum and cannot both
    be set. `bagging_temperature` belongs to `"Bayesian"` and is refused
    beside any other type, which is a deliberate divergence -- CatBoost
    accepts it beside MVS, never reads it, and tells the user nothing.

    A bootstrap runs on **every CPU fit whose round loop draws one**, which is
    now the dense single-output fit, the softmax fit, the sparse fit, the
    sparse softmax fit, continued training, and all of those through the
    `Dataset` API. What still cannot draw one -- the GPU on every shape, the
    rankers, custom objectives, `eval_set` fits, `linear_tree`, dart, rf, and
    the distributed prototype -- **refuses an enabled `bootstrap_type` by
    name** rather than training an unsampled model and reporting a sampled
    one. Two of those refusals are the round loop's own rather than an entry
    point's: MVS with no `mvs_reg` cannot continue an existing ensemble, and
    cannot run a softmax fit at all, because in both cases the lambda CatBoost
    derives reads a previous tree that does not exist in that shape.

    That last paragraph is about a `bootstrap_type` you **typed**. A
    `bootstrap_type` this library defaulted for you, if it ever defaults one,
    resolves quietly to `"No"` on a path that cannot honor it instead of
    raising -- a default must never turn a working `fit` into an error. The
    `bootstrap_explicit` wire key is what keeps the two apart; see
    `_BOOTSTRAP_DEFAULTS`. **Today mojotrees's own default is `None`, meaning
    `"No"`,** so nothing is being degraded and the distinction is machinery
    waiting for the default to move, not a behavior you can observe.

    `boosting_type` (LightGBM's `boosting`, XGBoost's `booster`) selects the
    training strategy: "gbdt", the default, trains every tree on
    every row, and "goss" is Gradient-based One-Side Sampling. CatBoost's
    "plain" is an alias of "gbdt". Its "ordered" trains CatBoost's ordered
    boosting, where row i's derivatives come from a model fitted on a prefix
    of a permutation that excludes it (`permutation_count`,
    `fold_len_multiplier`, `fold_permutation_block`, `has_time`,
    `ordered_seed`); it runs on a dense, single-output CPU fit without
    eval_set, a callable objective, or row sampling, and raises by name
    anywhere else. Under GOSS
    each round keeps the `top_rate` share of rows with the largest gradient
    magnitude, samples `other_rate` of the rest, and scales the sampled rows
    up to compensate. `goss_seed` makes the sample reproducible and
    `goss_warmup_rounds` overrides LightGBM's automatic
    `int(1 / learning_rate)` rounds of full-data training that precede
    sampling (-1, the default, keeps LightGBM's rule). GOSS cannot be
    combined with row bagging.

    "dart" (Dropouts meet Multiple Additive Regression Trees) drops a random
    subset of the trees already built before each round, fits the new tree
    to the residual of what is left, and rescales the dropped trees and the
    new one so the ensemble is not overshot; `drop_rate`, `max_drop`,
    `skip_drop`, `xgboost_dart_mode`, and `drop_seed` are LightGBM's
    parameters of those names. `uniform_drop` defaults to True here (LightGBM
    defaults to False); both drop rules follow LightGBM's `dart.hpp`. "rf" is random
    forest mode: every tree fits the same gradients and the model averages
    them, so `learning_rate` is ignored (trained at 1.0, as LightGBM does)
    and it needs a source of per-tree randomness, `bagging_fraction < 1` with
    `bagging_freq > 0` or `feature_fraction < 1`. Both modes train on the
    CPU, on dense input, without `eval_set` or a callable objective, and
    single-output only (a multiclass classifier or a ranker refuses them);
    the fitted model is an ordinary one for prediction, saving, and
    inspection.

    `colsample_bytree` (LightGBM's `feature_fraction`, CatBoost's `rsm`)
    samples that share of the features once per tree and `colsample_bynode`
    (LightGBM's `feature_fraction_bynode`) samples again at every node from
    the tree's own set, both without replacement and both reproducible from
    `feature_fraction_seed`. Fractions must be in (0, 1]; 1.0 (the default)
    means no subsampling. As in LightGBM, at least 2 features are selected
    whenever the data has that many.

    `n_jobs` (LightGBM's `num_threads`, CatBoost's `thread_count`) has no
    estimator-level answer here: the worker count comes from
    `MOJOTREES_NUM_WORKERS` and from the machine, and a fit is
    bit-identical at every worker count. `None`, `-1` and `0`, which all
    mean "use the machine", are what this already does and are accepted; a
    specific count raises and names the environment variable, rather than
    being accepted and dropped.

    `verbose` (LightGBM's and XGBoost's `verbosity`) is the log level.
    Nothing here writes a training log on its own, so `None`, `False` and
    any value at or below 0 are the silence this already is, and a positive
    value installs `callback.log_evaluation` with that period, which
    reports only where there is a validation set to report on.

    `early_stopping_rounds` (LightGBM's `early_stopping_round`, CatBoost's
    `od_wait`, scikit-learn's `n_iter_no_change`) sets the default for
    `fit`'s argument of the same name, which is where XGBoost and CatBoost
    take it too. A value given to `fit` wins, being the more specific
    statement.

    The CatBoost-only parameters keep CatBoost's names, and they split into
    two groups.

    **Wired.** `leaf_estimation_iterations` keeps taking Newton steps on a
    leaf's own rows after the tree's structure is fixed; above 1 it changes
    every leaf value on a plain dense, single-output fit (either device), and
    raises with the entry point's name everywhere else -- the sparse,
    multiclass, ranking and custom-objective paths, and eval_set, whose round
    loop is `custom_metric.fit_with_metrics` and not
    `boosting.train_with_valid`. It moves nothing under squared error, whose
    loss is exactly the quadratic one Newton step already minimizes.
    `boosting_type="ordered"` and its five knobs are described above.

    Left unset under `grow_policy="symmetrictree"` it resolves **per
    objective as CatBoost does** rather than to 1: `Logloss` and
    `CrossEntropy` to 10, `RMSE` and `MultiClass` to 1, read from
    `GetEstimationMethodDefaults` in
    `catboost/private/libs/options/catboost_options.cpp`. Under `lossguide`
    it stays at 1, which is LightGBM's behavior. That is the standing rule
    again, and it is the reason the parameter existing was not the same thing
    as the parameter being reachable: until 2026-08-16 only a plain
    `estimator.fit()` accepted it, while `mojotrees.train(params, Dataset)`
    -- which the benchmark suite trains through -- refused it by name, so
    every CatBoost-mode Logloss comparison had CatBoost taking ten Newton
    steps and mojotrees taking one.

    `random_strength_seed` is the seed the per-split score noise is drawn
    from. It is a seed like `bagging_seed` and follows `random_state` by the
    same rule, and it exists because until 2026-08-16 it did not: the noise
    stream ran from a native constant no matter what a caller passed for
    `random_state`, which made a `random_strength > 0` fit reproducible only
    by accident. The draw is keyed by (seed, tree, node, feature, bin) and by
    nothing else, so it is the same value at every `MOJOTREES_NUM_WORKERS`
    and on either device.

    `boost_from_average` is **LightGBM's** parameter as much as CatBoost's,
    so both halves of the standing rule apply to it, and it names behavior
    this package always had rather than adding any: `boosting._base_score`
    has seeded every fit from the objective's optimal constant since the
    beginning. `True` is therefore both LightGBM's default
    (`include/LightGBM/config.h:948`) and a no-op, and `lossguide` keeps it.
    `False` starts every row at 0.0; it is honored by the dense
    single-output CPU and GPU round loops and refused by name everywhere
    else, including continued training, whose starting point was decided by
    the original fit. Under `grow_policy="symmetrictree"` an unset value
    resolves per objective as CatBoost resolves it
    (`AdjustBoostFromAverageDefaultValue`, `options_helper.cpp:353-374`): on
    for `RMSE`, `MAE`, `Quantile` and `MAPE`, off for `Logloss`,
    `CrossEntropy` and `MultiClass`. Note that this is a place where the two
    engines genuinely disagree and an earlier reading that all three arms
    agreed came from LightGBM's default alone: on a Logloss cell CatBoost
    starts from zero and both other engines start from the prior log-odds.

    `score_function` selects the functional a split candidate is scored by:
    `"L2"` (the default) is `G**2 / (H + reg_lambda)`, which is what every
    fit here has always maximized, and `"Cosine"` is CatBoost's own default,
    a ratio rather than a sum. Both names are case insensitive. It reaches
    `split.find_best_split` through `ExtraTreeParams.score_function`, which
    `tree._search` and `tree._grow_oblivious_levels` pass in, so every
    grower in this package honors it except the distributed prototype
    (`tree_learner` other than `"serial"`), which is refused by name.

    **`"Cosine"` is not a no-op, and it is not an alias for `"L2"`
    either.** At `reg_lambda=0` its numerator and denominator collapse onto
    the same expression, so it degenerates to `sqrt` of the L2 score and,
    `sqrt` being strictly increasing, cannot move the argmax *within one
    node*. That is the only claim the derivation supports. It can still move
    the tree at `reg_lambda=0` under leaf-wise growth, which is the default,
    because the queue compares gains from different parents and `sqrt` does
    not preserve that ordering. CatBoost-mode comparisons set `reg_lambda=3`,
    which is off the degenerate point outright.
    Categorical features are refused under `"Cosine"`: a category set is
    searched and scored with the L2 gain, and only that search's winner
    reaches the numerical scan, so the two functionals would end up inside
    one argmax.

    **Not wired, and refused by name with the missing piece.**
    `max_ctr_complexity`: the CTR modules are implemented and the model file
    now carries a `ctr` section, but no binding entry point enables a CTR
    bundle, so no fit reachable from Python builds a CTR column. Not
    accepted and ignored.

    `random_strength` left that list. The claim above used to be that its
    per-tree scale "is computed by a function with no callers", and that was
    false in a way worth recording: the scale had callers, in
    `boosting._boost_rounds` and in `train_with_valid`'s own loop. What it
    lacked was a route from Python, because `_parse_params` declared
    `random_strength_ok` at exactly one call site. Three call sites declare
    it now, chosen by which round loop they reach, and every other entry
    refuses by name. See `docs/design/CATBOOST_CATALOG.md` A36 for the full
    enumeration.

    `bootstrap_type` and `bagging_temperature` left that list too: both reach
    `boosting.train`'s `bootstrap` argument on the dense single-output CPU
    fit, and refuse by entry point everywhere else. See the `bootstrap_type`
    paragraph above.

    `auto_learning_rate` (CatBoost's data-dependent rate, catalog A12/A38)
    never joined that list, because until 2026-08-16 the estimator had no
    such keyword at all and asking for it was a `TypeError`. It is a
    parameter now, and it is the first thing landed under the standing rule:
    **`grow_policy='symmetrictree'` mirrors CatBoost and turns it ON by
    default; `lossguide`, our default, mirrors LightGBM, which has no such
    feature, and leaves it off.** Nine of the fifteen `_parse_params` call
    sites derive the rate and the other six refuse an explicit request by
    name with the input they cannot supply. `learning_rate` and `lambda_l2`
    default to `None` for this: CatBoost's gate fires only on an UNSET rate
    and an unset `l2_leaf_reg`, and "unset" there is provenance rather than
    a value, so a float equal to the default cannot stand in for it.

    `use_missing` is LightGBM's parameter of the same name. With it (the
    default), `NaN` is a missing value: a feature that has any in training
    reserves a bin for them, the split search picks a default direction per
    node, and a `NaN` at predict time follows that direction. Without it,
    every `NaN` is treated as the value 0.0. `+inf` and `-inf` are never
    missing; they bin as the extreme finite values they compare as.

    `categorical_feature` is LightGBM's parameter of the same name (the
    plural `categorical_features` is accepted as an alias). It names the
    columns whose integer codes are unordered categories, and takes

    - `"auto"`, the default and LightGBM's: every pandas `category` column
      of `X`, and nothing else. On any other input that is no columns.
    - a sequence of feature names, resolved against the columns of a pandas
      DataFrame, or of column indices, or a mix of the two.
    - `None` or an empty sequence: no feature is categorical.

    Those columns are split by category set rather than by threshold, with
    no one-hot expansion, and their missing (negative or `NaN`), unseen, and
    dropped codes all route right. A pandas `category` column is encoded by
    its labels and the mapping is kept on the fitted estimator, so a
    prediction frame that orders or extends its categories differently
    still lands on the categories the model was fitted with; leaving such a
    column out of an explicit `categorical_feature` raises rather than
    feeding its codes to the numerical scan. Category codes on any other
    input must be whole numbers below 2**31, `NaN`, or negative.

    `max_cat_to_onehot`, `max_cat_threshold`, `cat_smooth`, `cat_l2`, and
    `min_data_per_group` are LightGBM's categorical hyperparameters, with
    LightGBM's defaults; they have no effect unless some feature is
    categorical.

    The remaining LightGBM tree controls keep LightGBM's names, defaults,
    and meanings, and are applied by the same Mojo code the C ABI and the
    CLI reach (`ExtraTreeParams` in src/mojotrees/tree_parameters_extra.mojo).
    Every one of them is inactive at its default, so leaving them alone
    leaves a fit bit-identical to what it was:

    - `min_gain_to_split` (alias `min_split_gain`) is the gain a split must
      clear to be taken at all.
    - `max_delta_step` caps the absolute value of a leaf's output.
    - `path_smooth` shrinks each leaf toward its parent's output in
      proportion to how few rows the leaf holds; LightGBM needs
      `min_data_in_leaf` of at least 2 for it, and so does mojotrees.
    - `extra_trees` draws one threshold per feature at random instead of
      scanning for the best one, keyed by `extra_seed` together with the
      tree index and the node id, so a fit is reproducible.
    - `monotone_penalty` (alias `monotone_constraints_penalty`) discounts
      the gain of a split near the root of a monotone branch.
      `monotone_constraints_method` is LightGBM's name for the algorithm;
      only `"basic"` is implemented, and `"intermediate"` and `"advanced"`
      raise rather than silently resolving to it.
    - `feature_contri` is one gain multiplier per feature (LightGBM's
      `feature_contri`), and `cegb_tradeoff` with `cegb_penalty_split` are
      the cost-effective gradient boosting knobs that charge a split for
      being taken. `cegb_penalty_feature_coupled` and
      `cegb_penalty_feature_lazy` are implemented too (charged against the
      per-ensemble ledger in src/mojotrees/cegb.mojo), but both are
      per-feature vectors and this estimator has no parameter carrying one
      for them yet; the Mojo API reaches them through
      `TreeParams.extra.penalties.cegb`.
    - `forced_splits` is LightGBM's forced-splits document, given as the
      JSON text or as the `dict`/`list` to serialize, rather than as
      `forcedsplits_filename`. Its raw thresholds still have to be mapped
      onto a fitted binning, which no entry point does yet, so a document
      raises with what that would take instead of training an unforced
      tree that reads as a forced one.

    `max_delta_step`, `path_smooth`, and `extra_trees` need a grower that
    passes each node's identity and row count. A backend that does not
    refuses rather than dropping them.

    `enable_bundle` is LightGBM's exclusive feature bundling: sparse
    features that are never non-zero on the same row are packed into one
    column, so the histogram loop runs over fewer of them
    (src/mojotrees/efb.mojo). It defaults to `False`, which is *not*
    LightGBM's default of true, and turning it on changes how long a fit
    takes rather than what it returns: the plan is fitted once per training
    call and dropped when the call ends, and the trees name original
    features and original bins, so a bundled fit and an unbundled one are
    the same model.

    Only the trainers that apply a plan accept the switch, and the rest
    raise rather than train an unbundled model that looks bundled: it is
    honored by dense CPU fits, by continued training, and by the sparse
    path (`fit` on a sparse matrix bundles the CSC matrix directly), and
    refused by `device="gpu"`, by a custom objective, by a custom metric or
    an eval set, and by the ranker.

    The knobs it governs are the plan-construction policy. `enable_bundle`
    and `max_conflict_rate` are LightGBM's names; the other five have no
    LightGBM counterpart and are described in `src/mojotrees/efb.mojo`:

    - `max_conflict_rate` is the fraction of rows a bundle may hold
      collisions on. Only LightGBM's own default of 0.0, which makes
      bundling exactly lossless, is accepted; above it a collision drops a
      training value where no metric reports the loss, so it raises with
      what lifting that would take.
    - `max_bundle_bins` (256) and `max_bundle_size` (0, meaning unlimited)
      cap a bundle's bins and members.
    - `max_nondefault_rate` (0.95) leaves a feature that is non-default on
      more rows than this out of any multi-member bundle.
    - `min_reduction` (0.0) is the fraction of the histogram footprint a
      plan must remove before it is applied at all; under it the fit runs
      unbundled.
    - `bundle_missing` (`False`) lets features that reserve a missing bin
      join a multi-member bundle.

    All seven are range-checked natively on every fit, whether or not the
    switch is on, so a bad value is named before any data is read.
    """

    #: Public attributes that `fit` sets and a refit clears. The model
    #: handle and the private caches are handled in `_reset_fitted`.
    _FITTED_ATTRS = (
        "n_features_in_",
        "feature_names_in_",
        "device_",
        "classes_",
        "n_classes_",
        "best_iteration_",
        "evals_result_",
        "best_score_",
        "stopped_early_",
        "n_iter_",
        "categorical_feature_",
    )

    def __init__(
        self,
        num_leaves=31,
        max_leaves=None,
        max_leaf_nodes=None,
        max_depth=-1,
        depth=None,
        grow_policy="lossguide",
        # `None`, not 0.1, and the value has not changed: an unset rate
        # resolves to `_LEARNING_RATE` at fit time. What `None` adds is the
        # one fact a float cannot carry, which is whether anybody typed it.
        # CatBoost's automatic learning rate fires only when the rate is
        # UNSET (`options_helper.cpp:277`, `TOption::NotSet()` in
        # `option.h:80-85`, an `IsSetFlag` written on assignment and never a
        # comparison against a default), and `grow_policy='symmetrictree'`
        # turns that derivation on by default. With a signature default of
        # 0.1 the gate could only have been reproduced as "the rate equals
        # the default", which would silently override a user who typed 0.1
        # and would fire for a benchmark arm that pinned it.
        learning_rate=None,
        eta=None,
        shrinkage_rate=None,
        n_estimators=100,
        num_iterations=None,
        num_boost_round=None,
        iterations=None,
        max_iter=None,
        min_data_in_leaf=20,
        min_child_samples=None,
        min_samples_leaf=None,
        # `None` for the same reason `learning_rate` is, and for one more:
        # `l2_leaf_reg` is the second of CatBoost's four gate keys
        # (`options_helper.cpp:280`) and it is the one whose CatBoost default
        # is not zero. A CatBoost-mode arm that names `lambda_l2=3.0` to
        # match CatBoost's default would, under a value test, close the gate
        # it was built to open, while CatBoost's own 3 does not close it
        # because CatBoost put it there rather than the user. `None` is the
        # only reading that gets both cases right: 3.0 supplied by a mode
        # default leaves the gate open, 3.0 typed by a caller closes it.
        # An unset value resolves to `_LAMBDA_L2` at fit time, unchanged.
        lambda_l2=None,
        lambda_l1=_LAMBDA_L1,
        reg_lambda=None,
        reg_alpha=None,
        l2_leaf_reg=None,
        l2_regularization=None,
        min_child_hess=1e-3,
        min_child_weight=None,
        min_sum_hessian_in_leaf=None,
        max_bin=255,
        max_bins=None,
        border_count=None,
        random_state=None,
        seed=None,
        random_seed=None,
        n_jobs=None,
        num_threads=None,
        nthread=None,
        thread_count=None,
        verbose=None,
        verbosity=None,
        early_stopping_rounds=None,
        early_stopping_round=None,
        od_wait=None,
        n_iter_no_change=None,
        random_strength=None,
        # CatBoost's seed for the per-split score noise
        # (`ExtraTreeParams.random_strength_seed`,
        # `tree_parameters_extra.DEFAULT_RANDOM_STRENGTH_SEED`, which is 0).
        # A real default rather than `None`, because it is a seed and
        # `_SEEDS` below is the mechanism that lets `random_state` fill it:
        # that loop overrides a seed only while it still equals its stock
        # default, so the stock default has to be a number here.
        #
        # Until this parameter existed the split-score noise ran from the
        # native constant no matter what the caller passed for
        # `random_state`, while every other draw in the fit followed it.
        random_strength_seed=0,
        bootstrap_type=None,
        bagging_temperature=None,
        leaf_estimation_iterations=None,
        # LightGBM's and CatBoost's `boost_from_average`. Three states, and
        # `None` is not "off": it is "whatever the grow policy implies",
        # which under `lossguide` and `depthwise` is LightGBM's own default
        # of True (`include/LightGBM/config.h:948`) and is what every fit
        # this package has ever made already did, and under
        # `symmetrictree` is CatBoost's PER-OBJECTIVE resolution -- True for
        # RMSE, MAE, Quantile and MAPE, False for Logloss, CrossEntropy and
        # MultiClass (`options_helper.cpp:353-374`). `True` and `False`
        # override it either way.
        #
        # The per-objective half is resolved natively, in `_parse_params`,
        # for the reason the learning rate's derivation is: the table is
        # `auto_learning_rate.catboost_boost_from_average_default` and a
        # Python copy of it would be a second table to keep true. What
        # crosses the boundary is the value plus whether anybody typed it.
        boost_from_average=None,
        # CatBoost's automatic learning rate (catalog A12/A38). Three states,
        # and `None` is not "off": it is "whatever the grow policy implies",
        # which is ON under `grow_policy='symmetrictree'` and OFF under every
        # other policy. CatBoost has no such key -- there the derivation is
        # implied by leaving the rate unset -- so naming it is an override of
        # the mode default in either direction, and `False` is how a
        # CatBoost-mode fit pins its own rate without leaving CatBoost mode.
        auto_learning_rate=None,
        score_function=None,
        max_ctr_complexity=None,
        device="cpu",
        device_type=None,
        task_type=None,
        tree_method=None,
        interaction_constraints=None,
        interaction_cst=None,
        monotone_constraints=None,
        monotonic_cst=None,
        bagging_fraction=1.0,
        subsample=None,
        bagging_freq=0,
        subsample_freq=None,
        bagging_seed=3,
        boosting="gbdt",
        boosting_type=None,
        booster=None,
        # CatBoost's ordered-boosting knobs, read only when the resolved
        # boosting strategy is "ordered" (src/mojotrees/ordered_boosting.mojo).
        # `None` is "not named" and takes the native default, so the whole
        # group is inert on every other fit and none of them appears on the
        # wire with a value this class invented.
        permutation_count=None,
        fold_len_multiplier=None,
        fold_permutation_block=None,
        has_time=None,
        ordered_seed=0,
        top_rate=0.2,
        other_rate=0.1,
        goss_seed=3,
        goss_warmup_rounds=-1,
        drop_rate=0.1,
        rate_drop=None,
        max_drop=50,
        skip_drop=0.5,
        xgboost_dart_mode=False,
        uniform_drop=True,
        drop_seed=4,
        feature_fraction=1.0,
        colsample_bytree=None,
        rsm=None,
        feature_fraction_bynode=1.0,
        colsample_bynode=None,
        max_features=None,
        feature_fraction_seed=2,
        use_missing=True,
        categorical_feature="auto",
        categorical_features=None,
        cat_features=None,
        max_cat_to_onehot=4,
        one_hot_max_size=None,
        max_cat_threshold=32,
        cat_smooth=10.0,
        cat_l2=10.0,
        min_data_per_group=100,
        min_gain_to_split=0.0,
        min_split_gain=None,
        gamma=None,
        max_delta_step=0.0,
        path_smooth=0.0,
        extra_trees=False,
        extra_seed=6,
        monotone_penalty=0.0,
        monotone_constraints_penalty=None,
        monotone_constraints_method="basic",
        feature_contri=None,
        cegb_tradeoff=1.0,
        cegb_penalty_split=0.0,
        linear_tree=False,
        linear_lambda=0.0,
        forced_splits=None,
        enable_bundle=False,
        max_conflict_rate=0.0,
        max_bundle_bins=256,
        max_bundle_size=0,
        max_nondefault_rate=0.95,
        min_reduction=0.0,
        bundle_missing=False,
        importance_type="split",
        tree_learner="serial",
        num_machines=1,
        top_k=20,
        langevin=None,
        diffusion_temperature=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
    ):
        self.num_leaves = num_leaves
        self.max_leaves = max_leaves
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.depth = depth
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.eta = eta
        self.shrinkage_rate = shrinkage_rate
        self.n_estimators = n_estimators
        self.num_iterations = num_iterations
        self.num_boost_round = num_boost_round
        self.iterations = iterations
        self.max_iter = max_iter
        self.min_data_in_leaf = min_data_in_leaf
        self.min_child_samples = min_child_samples
        self.min_samples_leaf = min_samples_leaf
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.l2_leaf_reg = l2_leaf_reg
        self.l2_regularization = l2_regularization
        self.min_child_hess = min_child_hess
        self.min_child_weight = min_child_weight
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.max_bin = max_bin
        self.max_bins = max_bins
        self.border_count = border_count
        self.random_state = random_state
        self.seed = seed
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.num_threads = num_threads
        self.nthread = nthread
        self.thread_count = thread_count
        self.verbose = verbose
        self.verbosity = verbosity
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_round = early_stopping_round
        self.od_wait = od_wait
        self.n_iter_no_change = n_iter_no_change
        self.random_strength = random_strength
        self.random_strength_seed = random_strength_seed
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.boost_from_average = boost_from_average
        self.auto_learning_rate = auto_learning_rate
        self.score_function = score_function
        self.max_ctr_complexity = max_ctr_complexity
        self.device = device
        self.device_type = device_type
        self.task_type = task_type
        self.tree_method = tree_method
        self.interaction_constraints = interaction_constraints
        self.interaction_cst = interaction_cst
        self.monotone_constraints = monotone_constraints
        self.monotonic_cst = monotonic_cst
        self.bagging_fraction = bagging_fraction
        self.subsample = subsample
        self.bagging_freq = bagging_freq
        self.subsample_freq = subsample_freq
        self.bagging_seed = bagging_seed
        self.boosting = boosting
        self.boosting_type = boosting_type
        self.booster = booster
        self.permutation_count = permutation_count
        self.fold_len_multiplier = fold_len_multiplier
        self.fold_permutation_block = fold_permutation_block
        self.has_time = has_time
        self.ordered_seed = ordered_seed
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.goss_seed = goss_seed
        self.goss_warmup_rounds = goss_warmup_rounds
        self.drop_rate = drop_rate
        self.rate_drop = rate_drop
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self.uniform_drop = uniform_drop
        self.drop_seed = drop_seed
        self.feature_fraction = feature_fraction
        self.colsample_bytree = colsample_bytree
        self.rsm = rsm
        self.feature_fraction_bynode = feature_fraction_bynode
        self.colsample_bynode = colsample_bynode
        self.max_features = max_features
        self.feature_fraction_seed = feature_fraction_seed
        self.use_missing = use_missing
        self.categorical_feature = categorical_feature
        self.categorical_features = categorical_features
        self.cat_features = cat_features
        self.max_cat_to_onehot = max_cat_to_onehot
        self.one_hot_max_size = one_hot_max_size
        self.max_cat_threshold = max_cat_threshold
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.min_data_per_group = min_data_per_group
        self.min_gain_to_split = min_gain_to_split
        self.min_split_gain = min_split_gain
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.path_smooth = path_smooth
        self.extra_trees = extra_trees
        self.extra_seed = extra_seed
        self.monotone_penalty = monotone_penalty
        self.monotone_constraints_penalty = monotone_constraints_penalty
        self.monotone_constraints_method = monotone_constraints_method
        self.feature_contri = feature_contri
        self.cegb_tradeoff = cegb_tradeoff
        self.cegb_penalty_split = cegb_penalty_split
        self.linear_tree = linear_tree
        self.linear_lambda = linear_lambda
        self.forced_splits = forced_splits
        self.enable_bundle = enable_bundle
        self.max_conflict_rate = max_conflict_rate
        self.max_bundle_bins = max_bundle_bins
        self.max_bundle_size = max_bundle_size
        self.max_nondefault_rate = max_nondefault_rate
        self.min_reduction = min_reduction
        self.bundle_missing = bundle_missing
        self.importance_type = importance_type
        self.tree_learner = tree_learner
        self.num_machines = num_machines
        self.top_k = top_k
        # CatBoost's Langevin block. Accepted so the refusal can name the
        # blocker instead of the argument; see `_check_langevin`.
        self.langevin = langevin
        self.diffusion_temperature = diffusion_temperature
        self.model_shrink_rate = model_shrink_rate
        self.model_shrink_mode = model_shrink_mode
        self._reset_fitted()

    def _interaction_buffers(self, n_features):
        """Validated float64 buffers for `interaction_constraints`: the
        flattened group features and one more offset than there are groups.
        Both must stay referenced while their addresses are in use.
        `(None, None)` when unconstrained."""
        # scikit-learn spells this `interaction_cst`.
        groups = self._resolve_alias(
            "interaction_constraints", "interaction_cst", None
        )
        if groups is None:
            return None, None
        if isinstance(groups, (str, bytes)):
            raise ValueError(
                "interaction_constraints must be a list of feature-index"
                " lists, not a string"
            )
        flat = []
        offsets = [0]
        for group in groups:
            if isinstance(group, (str, bytes)) or not hasattr(
                group, "__iter__"
            ):
                raise ValueError(
                    "each interaction constraint group must be a list of"
                    " feature indices"
                )
            members = [int(f) for f in group]
            if not members:
                raise ValueError(
                    "interaction constraint groups must not be empty"
                )
            if len(set(members)) != len(members):
                raise ValueError(
                    "an interaction constraint group repeats a feature"
                )
            for f in members:
                if not 0 <= f < n_features:
                    raise ValueError(
                        f"interaction constraint feature {f} is out of range"
                        f" for {n_features} features"
                    )
            flat.extend(members)
            offsets.append(len(flat))
        if not flat:
            return None, None
        return (
            _array.array("d", [float(f) for f in flat]),
            _array.array("d", [float(o) for o in offsets]),
        )

    def _monotone_buffer(self, n_features):
        """Validated float64 buffer for `monotone_constraints` and its address
        (the buffer must stay referenced while the address is in use);
        `(None, 0)` when unconstrained.

        One entry per feature, each exactly -1, 0, or 1. Fractional values are
        rejected here rather than truncated at the boundary, where the buffer
        is read as integers."""
        # `monotone_constraints` is unanimous across LightGBM, XGBoost and
        # CatBoost; scikit-learn spells it `monotonic_cst`.
        signs = self._resolve_alias(
            "monotone_constraints", "monotonic_cst", None
        )
        if signs is None:
            return None, 0
        if isinstance(signs, (str, bytes)):
            raise ValueError(
                "monotone_constraints must be a sequence of -1, 0, and 1"
                " values, not a string"
            )
        values = list(signs)
        if len(values) != n_features:
            raise ValueError(
                f"monotone_constraints has {len(values)} entries but X has"
                f" {n_features} features"
            )
        out = []
        for f, value in enumerate(values):
            sign = float(value)
            if sign not in (-1.0, 0.0, 1.0):
                raise ValueError(
                    f"monotone_constraints[{f}] must be -1, 0, or 1, got"
                    f" {value!r}"
                )
            out.append(sign)
        buf = _array.array("d", out)
        return buf, _addr(buf)

    def _feature_contri_buffer(self, n_features):
        """Validated float64 buffer for `feature_contri` and its address, or
        `(None, 0)` when unset. The buffer must stay referenced while the
        address is in use, the same contract `_monotone_buffer` has.

        LightGBM's `feature_contri` is one multiplier per feature applied to
        that feature's split gain, so a value below zero would flip the sign
        of a gain rather than scale it. `FeaturePenalties.check` in
        src/mojotrees/tree_parameters_extra.mojo refuses that too; the length
        is checked here because this is where `n_features` is known and the
        message can name the mismatch.
        """
        contri = self.feature_contri
        if contri is None:
            return None, 0
        if isinstance(contri, (str, bytes)):
            raise ValueError(
                "feature_contri must be a sequence of per-feature gain"
                " multipliers, not a string"
            )
        values = [float(v) for v in contri]
        if len(values) != n_features:
            raise ValueError(
                f"feature_contri has {len(values)} entries but X has"
                f" {n_features} features"
            )
        buf = _array.array("d", values)
        return buf, _addr(buf)

    def _forced_splits_text(self):
        """`forced_splits` as the document text the native parser reads, or
        `""` when unset.

        LightGBM takes this as `forcedsplits_filename`, a path. mojotrees
        refuses that name (`check_extra_option_supported` in
        src/mojotrees/tree_parameters_extra.mojo) and takes the document
        itself, so that reading a file is the caller's step and not a hidden
        one inside a fit. A `str` is passed through unchanged; a `dict` or
        `list` is serialized here, because the schema
        `parse_forced_splits` accepts is JSON and building it as Python
        objects is how a caller would rather write it:

            forced_splits={"feature": 0, "threshold": 1.5}

        Read a LightGBM file with `open(path).read()` and pass the text.
        Every error in the document is raised natively by
        `parse_forced_splits`, which names the byte it stopped at; nothing
        here inspects the schema.
        """
        forced = self.forced_splits
        if forced is None:
            return ""
        if isinstance(forced, bytes):
            return forced.decode("utf-8")
        if isinstance(forced, str):
            return forced
        if isinstance(forced, (dict, list)):
            return _json.dumps(forced)
        raise TypeError(
            "forced_splits must be the document text, or a dict or list to"
            f" serialize as JSON, not {type(forced).__name__}"
        )

    # -- categorical features ---------------------------------------------

    #: Category codes must stay representable as Int32, the way LightGBM's
    #: `static_cast<int>` requires. Kept in step with `_MAX_CATEGORY` in
    #: src/mojotrees/categorical.mojo.
    _CATEGORY_LIMIT = 1 << 31

    def _categorical_positions(self, spec, names):
        """Declared categorical features resolved to column positions.

        Entries are feature names or column indices, in any mix; names need
        a matrix that carries them, which in practice means a pandas
        DataFrame. The result is ascending and distinct, so listing a
        feature twice, by name and by index, is an error rather than a
        silent no-op.
        """
        if isinstance(spec, (str, bytes)):
            raise ValueError(
                "categorical_feature must be 'auto', None, or a sequence of "
                f"feature names or indices, got {spec!r}"
            )
        known = None if names is None else list(names)
        out = []
        for entry in spec:
            if isinstance(entry, bool):
                raise ValueError(
                    f"categorical_feature entry {entry!r} is a bool, not a "
                    "feature name or an index"
                )
            if isinstance(entry, str):
                if known is None:
                    raise ValueError(
                        f"categorical_feature names {entry!r}, but X carries "
                        "no feature names; pass column indices, or fit on a "
                        "pandas DataFrame"
                    )
                if entry not in known:
                    raise ValueError(
                        f"categorical_feature name {entry!r} is not a column "
                        f"of X; X has {known}"
                    )
                index = known.index(entry)
            else:
                try:
                    value = float(entry)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"categorical_feature entry {entry!r} is neither a "
                        "feature name nor an index"
                    ) from None
                if value != int(value):
                    raise ValueError(
                        "categorical_feature entries must be whole feature "
                        f"indices, got {entry!r}"
                    )
                index = int(value)
            if index in out:
                raise ValueError(
                    f"categorical_feature lists feature {index} twice"
                )
            out.append(index)
        out.sort()
        return out

    def _resolve_categorical(self, names, dtype_categories):
        """`(indices, encoders)` for one matrix.

        `indices` are the resolved categorical column positions and
        `encoders` maps a position to the category labels a pandas
        `category` column carries there. LightGBM's default, `"auto"`,
        means exactly those pandas columns and nothing else; `None` and an
        empty sequence mean no feature is categorical.

        A `category` column left out of an explicit list raises. LightGBM
        would quietly feed its codes to the numerical scan, which is the
        one thing a declared category must never be: an ordered number.
        """
        spec = self._resolve_alias(
            "categorical_feature", "categorical_features", "auto"
        )
        spec = self._resolve_alias(
            "categorical_feature", "cat_features", "auto", spec
        )
        if isinstance(spec, str):
            if spec != "auto":
                raise ValueError(
                    f"unknown categorical_feature {spec!r}; expected 'auto', "
                    "None, or a sequence of feature names or indices"
                )
            indices = sorted(dtype_categories)
        else:
            indices = self._categorical_positions(
                () if spec is None else spec, names
            )
            dropped = sorted(set(dtype_categories) - set(indices))
            if dropped:
                labels = [
                    i if names is None else names[i] for i in dropped
                ]
                raise ValueError(
                    f"columns {labels} have pandas categorical dtype but are "
                    "not in categorical_feature; list them, cast them to a "
                    "numeric dtype, or leave categorical_feature at 'auto'"
                )
        encoders = {
            index: dtype_categories[index]
            for index in indices
            if index in dtype_categories
        }
        return indices, encoders

    def _categorical_buffer(self, indices, n_features):
        """Float64 buffer of resolved categorical indices, or `None` when no
        feature is categorical. The buffer must stay referenced while the
        params dict that holds its address is in use."""
        for index in indices:
            if not 0 <= index < n_features:
                raise ValueError(
                    f"categorical_feature index {index} is out of range for "
                    f"{n_features} features"
                )
        if not indices:
            return None
        return _array.array("d", [float(index) for index in indices])

    def _check_category_codes(self, buf, n_rows, indices, names, name="X"):
        """Reject values a declared categorical column cannot carry.

        A code is a whole number below 2**31. `NaN` and negative values are
        missing and always allowed. Both bounds are checked here, at fit and
        at predict alike, rather than left to `bin_of`, which truncates a
        fractional code toward zero and reads an oversized one as unseen:
        either would answer a caller who encoded the column differently
        than they did at fit with a prediction instead of an error.
        """
        for index in indices:
            column = _arrays.column_view(buf, n_rows, index)
            bad = _arrays.first_bad_code(column, self._CATEGORY_LIMIT)
            if bad is None:
                continue
            label = index if names is None else repr(names[index])
            raise ValueError(
                f"categorical feature {label} of {name} holds {bad!r}, which "
                "is not a category code; codes are whole numbers below 2**31, "
                "and NaN or any negative value means missing"
            )

    def _matrix_encoders(self, X, name="X"):
        """The category tables to encode a matrix with after fitting: the
        fitted ones, never the matrix's own.

        A `category` column whose labels the model never saw cannot be
        encoded at all, and a matrix that carries no labels cannot deliver
        the ones the model was fitted on, so both raise rather than guess.
        """
        encoders = getattr(self, "_cat_encoders", None) or {}
        incoming = _arrays.frame_categories(X)
        unknown = sorted(set(incoming) - set(encoders))
        if unknown:
            raise ValueError(
                f"columns {unknown} of {name} have pandas categorical dtype, "
                f"but {type(self).__name__} holds no category mapping for "
                "them; pass their integer codes, or fit on a frame that "
                "carries the same categorical columns"
            )
        if encoders and not hasattr(X, "iloc"):
            raise ValueError(
                f"{type(self).__name__} was fitted on pandas categorical "
                f"columns {sorted(encoders)}, whose labels only a DataFrame "
                f"carries; pass {name} as a DataFrame, or fit on integer "
                "codes instead"
            )
        return encoders

    def _restore_categorical(self):
        """Recover which features are categorical from a model read back
        from disk.

        The serialized format carries the category tables, so a loaded
        model splits exactly as it did; what it cannot carry is the pandas
        label encoding the estimator applied on top, so `_cat_encoders`
        stays empty and a loaded model takes integer codes only. Pickle the
        estimator to keep the labels.
        """
        query = (
            _mojotrees.categorical_features_multiclass
            if self._multiclass
            else _mojotrees.categorical_features
        )
        self._cat_indices = [int(index) for index in query(self._model)]
        self.categorical_feature_ = list(self._cat_indices)

    def _fit_X(self, X):
        """`(buffer, n_rows, n_features, names, categorical buffer)` for a
        training matrix, with its categorical columns resolved and encoded.

        The resolved indices and category tables are recorded on the
        estimator here: prediction encodes through exactly these, so a
        prediction frame that orders or extends its categories differently
        still lands on the categories the model was fitted with.
        """
        names = _arrays.feature_names(X)
        dtype_categories = _arrays.frame_categories(X)
        indices, encoders = self._resolve_categorical(names, dtype_categories)
        Xb, n_rows, n_features, names = _arrays.check_X(X, encoders=encoders)
        cat_buf = self._categorical_buffer(indices, n_features)
        self._check_category_codes(Xb, n_rows, indices, names)
        self._cat_indices = list(indices)
        self._cat_encoders = encoders
        self.categorical_feature_ = list(indices)
        return Xb, n_rows, n_features, names, cat_buf

    def _check_fit_structure(
        self, X, y, n_rows, n_features, sample_weight=None, group=None
    ):
        """`_validation`'s structure checks for a fit call, run once the
        matrix has a shape.

        Shape ceilings and one entry per row for `y`, `sample_weight`, and
        the query counts. Every rule here is one the buffer conversions
        below would also refuse, so the checks add an earlier and named
        error, not a new rejection; the pandas index-alignment check is
        deliberately not run, because a positional `y` against a frame `X`
        is accepted here as it is in LightGBM.
        """
        _validation.check_shape(n_rows, n_features)
        _validation.check_length(y, n_rows, "y")
        _validation.check_optional_length(sample_weight, n_rows, "sample_weight")
        if group is not None:
            try:
                n_queries = len(group)
            except TypeError:
                raise TypeError(
                    "group must be a sequence of per-query row counts, got "
                    f"{type(group).__name__}"
                ) from None
            if n_queries < 1:
                raise ValueError("group must contain at least one query")
            if n_queries > n_rows:
                raise ValueError(
                    f"group has {n_queries} queries but X has only {n_rows} "
                    "rows, and every query needs at least one row"
                )

    def _refuse_alternate_boosting(self, where):
        """Raise when `boosting` is dart, rf or ordered and `where` names a
        fit path that only the plain trainer serves.

        Only `_mojotrees.fit` (dense, single-output, CPU, no eval_set, no
        callable objective) reads the `boosting` key and routes dart and rf
        to `alternate_boosting.fit_boosting`. Every other native entry point
        would train gbdt and say nothing, so those paths refuse by name here
        instead of quietly fitting a different model than the one asked for.

        `ordered` reaches exactly the same one entry point and for the same
        reason: `boosting.train` is the only trainer that grows the fold
        ladder, and `_mojotrees.fit` on the CPU is the only way this
        estimator reaches it. The native binding refuses it too
        (`_parse_params(ordered_ok=...)`), which is the check that cannot be
        forgotten; this one exists so the message names the fit path the
        user is on rather than the entry point they never typed.
        """
        boosting = self._resolve_boosting()
        if boosting in ("dart", "rf", "ordered"):
            raise ValueError(
                f"boosting={boosting!r} is not available {where}; it trains "
                "dense, single-output models on the CPU without eval_set or "
                "a callable objective"
            )

    #: `boosting_type` spellings and the strategy each names. One key
    #: carries all six values of docs/PARAMETER_NAMING.md: LightGBM's four,
    #: plus CatBoost's `Plain` (which is `gbdt` under another name) and
    #: `Ordered` (which is not implemented and is refused by name below, not
    #: rejected as unknown). XGBoost's `booster` is the same parameter and
    #: its `gbtree` is the same strategy.
    _BOOSTING_ALIASES = {
        "gbdt": "gbdt",
        "gbrt": "gbdt",
        "gbtree": "gbdt",
        "traditional": "gbdt",
        "plain": "gbdt",
        "dart": "dart",
        "goss": "goss",
        "rf": "rf",
        "random_forest": "rf",
        "ordered": "ordered",
    }

    def _resolve_boosting(self):
        """The effective boosting strategy: "gbdt", "goss", "dart" or "rf".

        `_resolve_alias` compares numerically, so the string-valued
        `boosting_type` / `boosting` / `booster` group resolves here
        instead, with the same rule: an unset alias leaves the primary
        alone, and two different non-default values raise. Values are case
        insensitive, as every value string is.

        `boosting_type` is the canonical name and `boosting` is the member
        that holds the default, which is why the primary is spelled the
        LightGBM way here and the message is not.
        """
        boosting = self.boosting
        for alias in ("boosting_type", "booster"):
            value = getattr(self, alias)
            if value is None:
                continue
            if boosting != "gbdt" and boosting != value:
                raise ValueError(
                    f"boosting_type={boosting!r} and {alias}={value!r} are "
                    "aliases with different values; set only one"
                )
            boosting = value
        resolved = None
        if isinstance(boosting, str):
            resolved = self._BOOSTING_ALIASES.get(boosting.strip().lower())
        # `ordered` is not in `_BOOSTING_TYPES` and must not be: that tuple
        # (`_fit_args.py`) is the set of *trainers* -- the four values that
        # select which round loop runs -- and ordered boosting is not a
        # fourth trainer. It is gbdt with the derivatives read off a fold
        # ladder instead of off the ensemble, so it goes over the wire as
        # `boosting="gbdt"` plus the `ordered` bundle `_params` sends beside
        # it, and `BoosterParams.ordered` is what `boosting.train` reads.
        if resolved == "ordered":
            return "ordered"
        if resolved is None or resolved not in _BOOSTING_TYPES:
            raise ValueError(
                f"unknown boosting_type {boosting!r}; expected one of "
                + ", ".join(sorted(_BOOSTING_TYPES))
                + ", plain (= gbdt), or ordered"
            )
        return resolved

    def _resolve_alias(self, primary, alias, default, folded=_UNSET):
        """The effective value of a parameter that has vendor aliases.

        scikit-learn requires `__init__` to store every argument unmodified,
        so aliases are resolved here, at fit time, rather than in the
        constructor. An unset alias is `None` and leaves the primary alone.
        LightGBM warns and keeps one value when a parameter and its alias
        disagree; mojotrees raises instead, so a typo cannot silently train
        a different model.

        `primary` is the name that holds the stock default and the value the
        native layer is sent -- LightGBM's spelling, because that is the wire
        (see the class docstring). It is not necessarily the canonical
        user-facing name: `num_leaves` is the primary and `max_leaves` is
        the canonical name resolved onto it.

        `folded` is what makes one parameter take more than one alias. A
        parameter with several vendor spellings resolves in a chain, each
        call passing the previous result:

            n = self._resolve_alias("n_estimators", "num_iterations", 100)
            n = self._resolve_alias("n_estimators", "iterations", 100, n)

        Each link keeps three literal leading arguments, which is what
        `tools/api_snapshot.py:alias_pairs` reads to derive the alias table
        from the call sites; a chain therefore stays derivable where a loop
        over a tuple of names would not. Conflicts are detected against the
        value resolved so far, so `num_iterations=200 iterations=300` raises
        rather than letting the last spelling win.
        """
        alias_value = getattr(self, alias)
        primary_value = (
            getattr(self, primary) if folded is _UNSET else folded
        )
        # A primary that is `None` is one whose signature default is `None`
        # rather than the stock value, which two of them now are:
        # `learning_rate` and `lambda_l2`. They carry `None` so that "the
        # user did not name this" is still knowable at fit time, which is
        # what CatBoost's `TOption::NotSet()` means and what its automatic
        # learning rate is gated on. Everywhere else the pair is exactly the
        # value it was before, because the fold happens here rather than at
        # every reader: `_multi_target.wire_params` and `_callback_params`
        # call this function too and neither had to learn about it.
        if primary_value is None:
            primary_value = default
        if alias_value is None:
            return primary_value
        if primary_value != default and primary_value != alias_value:
            raise ValueError(
                f"{primary}={primary_value} and {alias}={alias_value} are "
                "aliases with different values; set only one"
            )
        return alias_value

    #: The per-component seeds a global `random_state` fills, each with the
    #: stock default it keeps when neither it nor `random_state` is set.
    #: LightGBM's own defaults, and the same numbers `__init__` carries.
    _SEEDS = {
        "bagging_seed": 3,
        "feature_fraction_seed": 2,
        "extra_seed": 6,
        "goss_seed": 3,
        "drop_seed": 4,
        # `ordered_boosting.DEFAULT_ORDERED_SEED`, which is 0. CatBoost has
        # no name for this one -- its permutation comes off the global
        # generator -- so the default is the native module's and the reason
        # it is a parameter at all is that the permutation is then
        # reproducible by name.
        "ordered_seed": 0,
        # `tree_parameters_extra.DEFAULT_RANDOM_STRENGTH_SEED`, which is 0.
        # CatBoost's per-split score noise comes off its single `random_seed`
        # too, so 0 here is the same "CatBoost-inherited seeds are all 0"
        # family as `ordered_seed` and `bootstrap_seed`; the streams are kept
        # apart by their domain constants
        # (`tree_parameters_extra._RANDOM_SCORE_DOMAIN`) and not by distinct
        # seed numbers.
        #
        # **Added 2026-08-16, and its absence was a reproducibility hole
        # rather than a wrong number.** The draw was always deterministic --
        # keyed by (seed, tree, node, feature, bin) and by nothing else -- but
        # the seed was unreachable from Python, so `random_state` fanned out
        # to six draws and not to the seventh. With `random_strength`
        # reachable and CatBoost mode about to carry a positive strength by
        # default, that is the difference between a seeded fit and a fit that
        # repeats only because nothing asked it not to.
        "random_strength_seed": 0,
    }

    def _resolve_seeds(self):
        """Every per-component seed a fit runs with, after `random_state`.

        `random_state` is scikit-learn's word and the canonical one;
        LightGBM spells it `seed`, XGBoost `random_state`/`seed`, CatBoost
        `random_seed`. It seeds every draw at once, which is what a script
        that sets it is asking for, and a seed named outright wins over it
        whichever order the two appear in.

        **It sets each seed to `random_state` rather than deriving one per
        component the way LightGBM's `seed` does.** That is a divergence and
        it is stated rather than hidden. LightGBM derives its per-component
        seeds by running its own LCG over the global one; reproducing that
        would mean reimplementing that generator bit for bit, and it would
        still not reproduce LightGBM's row and feature subsets, because
        mojotrees draws with splitmix64 (src/mojotrees/rng.mojo) and a
        matching seed gives a different sample. What `random_state` buys
        here is that a mojotrees fit repeats, not that it matches LightGBM's.

        Returns the full mapping so the caller reads one dict rather than
        five aliases.
        """
        global_seed = self._resolve_alias("random_state", "seed", None)
        global_seed = self._resolve_alias(
            "random_state", "random_seed", None, global_seed
        )
        out = {}
        for name, default in self._SEEDS.items():
            value = getattr(self, name)
            if global_seed is not None and value == default:
                value = global_seed
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            out[name] = int(value)
        return out

    def _check_n_jobs(self):
        """Refuse a worker count this estimator cannot honor, accept the
        values that mean "use the machine".

        `n_jobs` is scikit-learn's word and the canonical one; LightGBM
        spells it `num_threads`, XGBoost `n_jobs`/`nthread`, CatBoost
        `thread_count`. There is no worker count on the estimator: the
        native layer takes one from `MOJOTREES_NUM_WORKERS` and from the
        machine (src/mojotrees/parallel.mojo), and a fit is bit-identical at
        every worker count by contract, so the count is a speed knob and
        nothing else.

        `None`, `-1` and `0` all mean "use the machine", which is exactly
        what happens, so they are honored by doing nothing. A specific count
        is refused by name and pointed at the variable that does reach the
        scheduler, rather than accepted and dropped: a user who asked for
        four threads and silently got sixteen has been misled about the
        thing they asked about.
        """
        jobs = self._resolve_alias("n_jobs", "num_threads", None)
        jobs = self._resolve_alias("n_jobs", "nthread", None, jobs)
        jobs = self._resolve_alias("n_jobs", "thread_count", None, jobs)
        if jobs is None or int(jobs) <= 0:
            return
        raise ValueError(
            f"n_jobs={int(jobs)} cannot be set on the estimator: the worker "
            "count comes from MOJOTREES_NUM_WORKERS and from the machine "
            "(src/mojotrees/parallel.mojo), and a fit is bit-identical at "
            f"every worker count. Set MOJOTREES_NUM_WORKERS={int(jobs)} in "
            "the environment instead. n_jobs of -1, 0 or None, which means "
            "'use the machine', is what this estimator already does and is "
            "accepted."
        )

    def _check_catboost_only(self):
        """The CatBoost-only parameters that this estimator still cannot
        honor, each accepted at the value that names what mojotrees already
        does and refused by name otherwise.

        **This function is shorter than it was, and what left it is the
        point.** `leaf_estimation_iterations` is no longer here: the binding
        now folds it onto `TreeParams.extra` and refuses it by entry point
        rather than outright (`_parse_params(leaf_estimation_ok=...)`), so a
        value above 1 reaches `boosting._estimate_leaf_values` on the dense
        single-output fits and raises with the entry point's name on the
        rest. `boosting_type='ordered'` left `_resolve_boosting` for the same
        reason. What remains here are the names whose mechanism this build
        genuinely does not run from any Python entry point, and each message
        says which piece is missing rather than that the parameter is
        unknown -- the refuse-rather-than-ignore rule
        src/mojotrees/params.mojo applies to the same names on the
        parameter-string surface.

        `None` everywhere is "not named", and nothing is checked for it.
        """
        if self.random_strength is not None:
            if float(self.random_strength) < 0.0:
                raise ValueError("random_strength must be nonnegative")
            # No longer refused. `random_score_scale_from_gradients` had no
            # caller when the old block was written; the dense CPU round
            # loops now compute the scale per tree onto their own copy of the
            # bundle (`boosting._round_random_score_scale`) before growth, so
            # the pair `random_strength > 0` with a zero scale is legitimate
            # at parameter time and `params.mojo` declares that by passing
            # `scale_computed_per_tree` on the CPU arm.
            #
            # Still refused deeper, by design rather than by omission:
            # multiclass, distributed and the device round loops do not
            # compute a scale, so a fit on any of those raises rather than
            # training a model that ignored the setting.
        if self.max_ctr_complexity is not None:
            if self.max_ctr_complexity != 1:
                raise ValueError(
                    "max_ctr_complexity above 1 is not reachable: the "
                    "projection enumeration exists "
                    "(src/mojotrees/ctr_combinations.mojo) but no grow loop "
                    "drives it, so a combination would never be built. 1 is "
                    "what CatBoost itself resolves to for any fit under 200 "
                    "iterations."
                )
            # 1 is built and reaches a design matrix, and this still refuses,
            # because the blocker moved rather than cleared. The fitted CTR
            # tables are MODEL STATE -- built from the target -- and the model
            # format has no section for them, so a CTR fit that produced a
            # model would write a file that loads with empty tables, keeps
            # every tree referencing its CTR columns, and bins them as if the
            # feature were absent. Refusing here beats producing a model that
            # loads and scores wrong.
            raise ValueError(
                "max_ctr_complexity=1 is built but no Python fit enables it: "
                "an enabled CTR bundle is refused at the trainer boundary "
                "(ctr.check_ctr_model_support) and at every model writer "
                "(serialize.check_ctr_serializable), because the fitted CTR "
                "tables are model state and the model format carries no "
                "section for them. Follow catalog A29."
            )
        # `bootstrap_type` and `bagging_temperature` used to be refused here
        # and are not any more: both travel on the wire and reach
        # `boosting.train`'s `bootstrap` argument through `_resolve_bootstrap`
        # below, `_parse_bootstrap` in bindings/_mojotrees.mojo, and
        # `model.fit`. The two spellings that stay unreachable are refused
        # there, by name, by `sampling.canonical_bootstrap_type`, which is the
        # one resolver the parameter string and the CLI also reach.

    #: The five wire keys of CatBoost's `bootstrap_type` and the value each
    #: takes when the estimator names nothing. Every key is sent on every
    #: fit, defaults included, because `_parse_bootstrap` in
    #: bindings/_mojotrees.mojo subscripts the mapping rather than testing
    #: for a key: a missing one is a KeyError at the boundary, not a default.
    #:
    #: **The three negative values are sentinels, not values.** CatBoost's
    #: ranges are `subsample` in (0, 1], `bagging_temperature` >= 0 and
    #: `mvs_reg` >= 0, so a negative number cannot be a setting and is free
    #: to mean "the user did not name this". The distinction has to survive
    #: the wire because `bagging_temperature` beside MVS is refused and a
    #: *defaulted* temperature is not a user setting
    #: (`sampling.check_mvs_bagging_temperature`, CatBoost's
    #: `TOption::IsSet`).
    #:
    #: `bootstrap_subsample` is not spelled `subsample`, and that is the
    #: whole `subsample` collision in one line: LightGBM's `subsample` is
    #: `bagging_fraction`, which already has its own wire key, and CatBoost's
    #: is the bootstrap rate. `_resolve_bootstrap` decides which of the two
    #: the user meant while both are still visible and sends the resolved
    #: number under an unambiguous name.
    #: `bootstrap_explicit` is the sixth key and is not a value of the
    #: sampler; it is **who asked**. 1 means the user wrote a
    #: `bootstrap_type` down, 0 means the value beside it is this library's
    #: own default. `_parse_bootstrap_request` in bindings/_mojotrees.mojo
    #: turns it into a `sampling.BootstrapRequest`, and the whole reason it
    #: has to cross the wire is that a defaulted MVS and a typed MVS are the
    #: same five numbers and must behave differently on a path that cannot
    #: run one: a typed request is refused by name, a default is dropped in
    #: silence. See `sampling.BootstrapRequest`.
    _BOOTSTRAP_DEFAULTS = {
        "bootstrap_type": "no",
        "bootstrap_subsample": -1.0,
        "bagging_temperature": -1.0,
        "mvs_reg": -1.0,
        # `sampling.DEFAULT_BOOTSTRAP_SEED`.
        "bootstrap_seed": 0,
        "bootstrap_explicit": 0,
    }

    def _resolve_bootstrap(self, bagging_fraction, bagging_freq):
        """CatBoost's `bootstrap_type` group as wire keys, plus the row
        bagging that survives beside it.

        Returns `(knobs, bagging_fraction, bagging_freq)`. The two bagging
        values come back changed under a real bootstrap type, and that is the
        point of the method: **`subsample` means different things under
        different `bootstrap_type` values, in CatBoost and therefore here**,
        and the choice has to be made where both the raw `subsample` and the
        raw `bagging_fraction` are still visible.

        CatBoost's `subsample` is a member of the bootstrap options. Under
        `MVS` it is the MVS rate; under `Bernoulli` it is the fraction of
        rows kept, which is the same draw mojotrees calls `bagging_fraction`.
        This estimator's `subsample` is scikit-learn's spelling of
        `bagging_fraction` and has only ever meant the second. So
        `bootstrap_type="MVS", subsample=0.8` -- which is CatBoost's own CPU
        default configuration -- names one sampler there and would build two
        here, and `boosting._check_bootstrap` refuses exactly that pair:
        `bagging_fraction` IS CatBoost's Bernoulli bootstrap under mojotrees's
        name, so MVS beside it is two bootstrap types at once.

        The resolution is CatBoost's contract read literally: under `MVS`,
        `subsample` is the MVS rate and row bagging is off. Nothing is
        silently reinterpreted, because every other reading is refused by
        name -- an explicit `bagging_fraction` below 1, an explicit
        `subsample_freq` or `bagging_freq`, and `subsample` beside Bayesian
        (which CatBoost refuses too: "bayesian bootstrap doesn't support
        'subsample' option").

        With `bootstrap_type` unset or `"No"`, every one of those parameters
        keeps the meaning it has always had and no fit moves a bit.
        """
        knobs = dict(self._BOOTSTRAP_DEFAULTS)
        # The draw is keyed on (seed, tree, row) with its own domain constant
        # (`sampling._mvs_stream`, `_MVS_DOMAIN`), so a seed is all it takes
        # to make a bootstrapped fit repeat. `random_state` is the estimator's
        # one seed and there is no `bootstrap_seed` parameter, so an unseeded
        # fit runs at the native default and a seeded one follows the seed.
        global_seed = self._resolve_alias("random_state", "seed", None)
        global_seed = self._resolve_alias(
            "random_state", "random_seed", None, global_seed
        )
        if global_seed is not None:
            if int(global_seed) < 0:
                raise ValueError("random_state must be nonnegative")
            knobs["bootstrap_seed"] = int(global_seed)

        # Who asked. Set from `self.bootstrap_type` alone and from nothing
        # else, because that is the parameter that names a sampler: a user
        # who set `subsample` or `bagging_temperature` beside no
        # `bootstrap_type` is refused by name below rather than promoted into
        # an explicit request. See `_BOOTSTRAP_DEFAULTS` for what the flag
        # buys and `sampling.BootstrapRequest` for how it is spent.
        knobs["bootstrap_explicit"] = int(self.bootstrap_type is not None)

        if self.bootstrap_type is None:
            kind = "no"
        else:
            # Values are case-insensitive and fold once, here, before the
            # native `canonical_bootstrap_type`, which takes canonical
            # lowercase (docs/PARAMETER_NAMING.md, and the same contract
            # `device_type` and `score_function` keep).
            kind = str(self.bootstrap_type).lower()

        # The two spellings that name a real CatBoost bootstrap type this
        # build does not run. Refused here rather than left to the native
        # resolver so that the message can point at the parameter that DOES
        # do the thing, which the native resolver cannot know about.
        if kind == "bernoulli":
            raise ValueError(
                "bootstrap_type='Bernoulli' is row bagging under another "
                "name; use subsample with subsample_freq."
            )
        if kind == "poisson":
            raise ValueError(
                "bootstrap_type='Poisson' is not implemented; CatBoost "
                "itself refuses it on the CPU."
            )
        if kind not in ("no", "none", "bayesian", "mvs"):
            raise ValueError(
                f"unknown bootstrap_type {self.bootstrap_type!r}; expected "
                "No, Bayesian, Bernoulli, MVS, or Poisson"
            )

        if kind in ("no", "none"):
            knobs["bootstrap_type"] = "no"
            if self.bagging_temperature is not None:
                raise ValueError(
                    "bagging_temperature belongs to bootstrap_type="
                    "'Bayesian' and is read by no other type "
                    "(sampling.bayesian_bootstrap_weights). Set "
                    "bootstrap_type='Bayesian' or remove it."
                )
            return knobs, bagging_fraction, bagging_freq

        # From here on a real sampler is configured, and row bagging is the
        # other bootstrap type. Refusing the pair is `_check_bootstrap`'s
        # rule, restated at the surface so the message can name the
        # parameters the user actually typed.
        if float(self.bagging_fraction) != 1.0:
            raise ValueError(
                f"bagging_fraction={self.bagging_fraction} cannot be set "
                f"beside bootstrap_type={self.bootstrap_type!r}: "
                "bagging_fraction IS CatBoost's Bernoulli bootstrap under "
                "mojotrees's name, so this asks for two bootstrap types at "
                "once. Use subsample for the MVS rate, or drop "
                "bootstrap_type."
            )
        if int(self.bagging_freq) != 0 or self.subsample_freq is not None:
            raise ValueError(
                "subsample_freq is a row-bagging schedule and cannot be set "
                f"beside bootstrap_type={self.bootstrap_type!r}: both MVS "
                "and the Bayesian bootstrap redraw once per tree "
                "unconditionally (CatBoost's sampling_frequency=PerTree "
                "default), so there is no frequency to set."
            )

        if kind == "mvs":
            knobs["bootstrap_type"] = "mvs"
            if self.subsample is not None:
                rate = float(self.subsample)
                if not 0.0 < rate <= 1.0:
                    raise ValueError("subsample must be in (0, 1]")
                knobs["bootstrap_subsample"] = rate
            # `bagging_temperature` beside MVS is refused natively, by
            # `sampling.check_mvs_bagging_temperature`, which carries the
            # argument and the CatBoost divergence. Emitting it and letting
            # that function refuse keeps one authority for the rule.
            if self.bagging_temperature is not None:
                knobs["bagging_temperature"] = float(self.bagging_temperature)
            # Row bagging is off under a bootstrap type: the value the alias
            # resolution folded out of `subsample` was the MVS rate.
            return knobs, 1.0, 0

        knobs["bootstrap_type"] = "bayesian"
        if self.subsample is not None:
            raise ValueError(
                "bootstrap_type='Bayesian' does not take subsample: the "
                "Bayesian bootstrap keeps every row and reweights it, so "
                "there is no fraction to set. CatBoost refuses the same "
                "pair. Use bootstrap_type='MVS' to subsample by gradient "
                "magnitude, or drop bootstrap_type for uniform row bagging."
            )
        if self.bagging_temperature is not None:
            temperature = float(self.bagging_temperature)
            if not temperature >= 0.0:
                raise ValueError("bagging_temperature must be >= 0")
            knobs["bagging_temperature"] = temperature
        return knobs, 1.0, 0

    def _check_langevin(self):
        """CatBoost's `langevin` / `diffusion_temperature` /
        `model_shrink_rate` / `model_shrink_mode`, refused by name with the
        blocker rather than accepted and dropped.

        **This is built and it is not reachable, and the two halves of that
        sentence have different reasons.** `src/mojotrees/langevin.mojo`
        holds the whole mechanism, tested: the counter-based normal draw
        (`langevin_row_noise`), the per-row injection at CatBoost's own place
        in the round (`apply_langevin_noise`, called from `DoBootstrap` on
        the line after `Bootstrap`), the leaf-sum draw, the deferred shrink
        fold (`ModelShrinkPlan.fold_into_trees`), and the coupling that makes
        `langevin=True` install a default `model_shrink_rate`. What does not
        exist is a CALLER. `BoosterParams` has no `LangevinParams` field and
        no `ModelShrinkParams` field, so `boosting.train`'s round loop never
        draws the noise and never records a shrink event, and `tree.mojo` has
        no leaf-sum call site for the second draw. Wiring it is an edit to
        `boosting.mojo` and `tree.mojo`, which is a trainer change and not a
        parameter change.

        A `False`/`0.0` is accepted, because that is exactly what an
        untouched fit already is. Anything else raises.
        """
        if self.langevin:
            raise ValueError(
                "langevin=True is not reachable: the mechanism is built and "
                "tested (src/mojotrees/langevin.mojo: apply_langevin_noise, "
                "langevin_leaf_gradient_noise, langevin_leaf_newton_noise) "
                "but no trainer calls it -- BoosterParams carries no "
                "LangevinParams, so boosting.train's round loop never draws "
                "the noise. Catalog A13."
            )
        if self.diffusion_temperature is not None:
            raise ValueError(
                "diffusion_temperature only scales langevin's noise, and "
                "langevin itself is unreachable (see the langevin refusal). "
                "Setting it alone would change nothing."
            )
        if self.model_shrink_rate is not None and float(
            self.model_shrink_rate
        ) != 0.0:
            raise ValueError(
                "model_shrink_rate is not reachable: the shrink plan is "
                "built and exact (src/mojotrees/langevin.mojo: "
                "ModelShrinkPlan.record, apply_model_shrinkage, "
                "fold_into_trees) but no round loop records an event and no "
                "fit folds the plan into the leaf values, because "
                "BoosterParams carries no ModelShrinkParams. Catalog A14."
            )
        if self.model_shrink_mode is not None:
            raise ValueError(
                "model_shrink_mode selects between two shrink schedules and "
                "neither runs; see the model_shrink_rate refusal."
            )

    #: The four wire keys of ordered boosting and the native default each
    #: takes when the estimator names nothing. The numbers are
    #: `ordered_boosting.DEFAULT_*`, restated here because the wire mapping
    #: is subscripted key by key on the native side and a missing key is a
    #: KeyError at the boundary rather than a default.
    _ORDERED_DEFAULTS = {
        "permutation_count": 1,
        "fold_len_multiplier": 2.0,
        "fold_permutation_block": 0,
    }

    def _resolve_ordered(
        self, ordered, ordered_seed, bagging_fraction, bagging_freq
    ):
        """The four wire values ordered boosting runs under, and the
        refusals for every configuration it cannot compose with.

        Returns the mapping `_params` splices into the wire dict. When
        `ordered` is False it is the defaults, unread by the native side
        because `ordered=0` sits beside them, and any knob named anyway is
        refused rather than dropped: a value that would be parsed and never
        read is the failure this whole lane exists to stop.

        **`has_time` is a block size, not a flag.** CatBoost's `has_time`
        forces `PermutationCount = 1` (`catboost_options.cpp:1043`) and makes
        `IsPermutationNeeded` false (`learn_context.cpp:38-46`), which sets
        `FoldPermutationBlockSize = learnSampleCount` -- one block, identity
        permutation. `OrderedBoostingParams.resolve_block_size` clamps a
        block larger than the row count down to the row count, so sending a
        block of `2**31 - 1` is exactly "one block" at any row count without
        this layer having to know `n_rows`, which `_params` is not given.
        An explicit `fold_permutation_block` beside it is refused rather than
        silently overridden.
        """
        named = [
            name
            for name in (
                "permutation_count",
                "fold_len_multiplier",
                "fold_permutation_block",
                "has_time",
            )
            if getattr(self, name) is not None
        ]
        if not ordered:
            if named:
                raise ValueError(
                    ", ".join(named)
                    + " configure ordered boosting and are read only when "
                    "boosting_type='ordered'; set that too, or drop the knob"
                )
            return {"ordered": 0, "ordered_seed": int(ordered_seed),
                    **self._ORDERED_DEFAULTS}
        out = dict(self._ORDERED_DEFAULTS)
        if self.permutation_count is not None:
            if int(self.permutation_count) < 1:
                raise ValueError("permutation_count must be positive")
            out["permutation_count"] = int(self.permutation_count)
        if self.fold_len_multiplier is not None:
            if not float(self.fold_len_multiplier) > 1.0:
                raise ValueError(
                    "fold_len_multiplier must be greater than 1"
                )
            out["fold_len_multiplier"] = float(self.fold_len_multiplier)
        if self.fold_permutation_block is not None:
            if int(self.fold_permutation_block) < 0:
                raise ValueError(
                    "fold_permutation_block must be nonnegative"
                )
            out["fold_permutation_block"] = int(self.fold_permutation_block)
        if self.has_time:
            if self.fold_permutation_block is not None:
                raise ValueError(
                    "has_time=True and fold_permutation_block cannot both be "
                    "set: has_time is one block over the whole learn set, "
                    "which is a block size, and honoring both would mean "
                    "choosing one of two answers"
                )
            if self.permutation_count is not None and (
                int(self.permutation_count) != 1
            ):
                raise ValueError(
                    "has_time=True forces permutation_count=1 (CatBoost's "
                    "catboost_options.cpp:1043); drop permutation_count or "
                    "set it to 1"
                )
            out["permutation_count"] = 1
            out["fold_permutation_block"] = 2**31 - 1
        # The exclusions `boosting._check_ordered` enforces, stated here so
        # the message names the estimator parameter the user set rather than
        # the native bundle they never saw. Row samplers change which prefix
        # each rung was fitted on, so the ladder stops meaning what it says.
        # `bagging_fraction` and `bagging_freq` arrive already resolved, from
        # the one `_resolve_alias` chain in `_params` that owns those two
        # aliases. Resolving them a second time here would work and would put
        # a second call site on the `subsample` / `subsample_freq` pairs,
        # which `tools/api_snapshot.py:alias_pairs` counts and reports as a
        # change to the alias table.
        if int(bagging_freq) > 0 and float(bagging_fraction) < 1.0:
            raise ValueError(
                "boosting_type='ordered' cannot be combined with row "
                "bagging: every rung's derivatives come from a model fitted "
                "on a known prefix of a known permutation, and a sampler "
                "that drops rows changes which prefix that was. Leave "
                "subsample at 1.0 or subsample_freq at 0."
            )
        out["ordered"] = 1
        out["ordered_seed"] = int(ordered_seed)
        return out

    def _resolve_grow_policy(self):
        """The canonical `grow_policy` value a fit runs under.

        One of `lossguide`, `depthwise`, `symmetrictree`
        (docs/PARAMETER_NAMING.md), resolved case insensitively from any of
        the spellings in `_CANONICAL_GROW_POLICIES`. That is the string
        `_params` sends, and `growth_policy.parse_grow_policy` on the other
        side accepts all three.
        """
        value = self.grow_policy
        if isinstance(value, str):
            resolved = _CANONICAL_GROW_POLICIES.get(value.strip().lower())
            if resolved is not None:
                return resolved
        raise ValueError(
            "grow_policy must be 'lossguide' (alias 'leafwise'), "
            "'depthwise', or 'symmetrictree' (aliases 'oblivious', "
            f"'symmetric'), got {self.grow_policy!r}"
        )

    def _auto_learning_rate_knobs(self, grow_policy, boosting):
        """The four `auto_learning_rate_*` keys a fit goes out with.

        CatBoost's automatic learning rate (catalog A12/A38,
        `src/mojotrees/auto_learning_rate.mojo`). The formula, the
        coefficient table and the half of the gate that reads the
        coefficient table all live there and none of it is copied here. What
        this method decides is the half that cannot cross the wire, which is
        **provenance**: whether the caller named a parameter, as against
        what its value came out as.

        CatBoost's gate, `UpdateLearningRate` in
        `catboost/libs/train_lib/options_helper.cpp:276-281`, is

            learningRate.NotSet() &&
            ObliviousTreeOptions->LeavesEstimationMethod.NotSet() &&
            ObliviousTreeOptions->LeavesEstimationIterations.NotSet() &&
            ObliviousTreeOptions->L2Reg.NotSet()

        and `NotSet()` is `!IsSetFlag` (`option.h:80-85`), a flag written
        when the option is assigned from the user's JSON and never a
        comparison against a default. That distinction is the whole
        difficulty of reproducing this from Python, and it is why
        `learning_rate` and `lambda_l2` default to `None` in the constructor
        rather than to their stock values: a float that equals the default
        cannot say who put it there. Two cases the value reading would get
        wrong, both of which are live in this repository today:

        - a benchmark arm that pins `learning_rate=0.1` for reproducibility
          would have the pin silently replaced under CatBoost mode, and the
          run would report a derived rate it did not use;
        - a CatBoost-mode arm that sets `lambda_l2=3.0` to match CatBoost's
          own default would close the gate that CatBoost's own 3 leaves
          open, so the derivation would never fire on the arm built to show
          it. Under `None` the two are distinguishable: 3.0 arriving from a
          mode default leaves the gate open, 3.0 named by a caller closes
          it.

        `leaf_estimation_method` has no mojotrees spelling at all (Newton
        only, catalog A6), so that third gate is permanently open for us and
        is not sent.

        The mode default. `auto_learning_rate=None` means "what the grow
        policy implies": ON under `symmetrictree`, which is CatBoost's, and
        OFF under `lossguide` and `depthwise`, which mirror LightGBM and
        LightGBM has no such feature. `True` and `False` override it either
        way. That split is the standing rule -- CatBoost mode mirrors
        CatBoost, our default mirrors LightGBM -- and this is its first
        instance.

        `required` is the difference between an explicit `True` and an
        inherited mode default, and the native side spends it on refuse
        versus decline. An explicit request an entry point cannot honor is
        refused by name, because a parameter accepted and dropped is the
        defect this whole sequence exists to remove. An inherited default it
        cannot honor falls back to the given rate in silence, because that
        is precisely what CatBoost does when its own table has no row.
        """
        asked = self.auto_learning_rate
        if asked is None:
            wanted = grow_policy == "symmetrictree"
            required = False
        else:
            wanted = bool(asked)
            required = wanted
        # Provenance, one name at a time. Each of these is `None` in the
        # constructor when it is not named, so "named" is exactly "not None"
        # and no value is compared with anything.
        rate_named = (
            self.learning_rate is not None
            or self.eta is not None
            or self.shrinkage_rate is not None
        )
        l2_named = (
            self.lambda_l2 is not None
            or self.reg_lambda is not None
            or self.l2_leaf_reg is not None
            or self.l2_regularization is not None
        )
        leaf_iters_named = self.leaf_estimation_iterations is not None
        if required:
            # The same three contradictions `params.mojo` refuses on the
            # parameter-string surface (`_enable_auto_learning_rate`), with
            # the same reasoning: CatBoost resolves each of them by silently
            # keeping its constant, and a user who asked for the derivation
            # in so many words has no way to see that they did not get it.
            # Silent under the mode default, refused under an explicit ask.
            if rate_named:
                raise ValueError(
                    "auto_learning_rate=True and an explicit learning_rate "
                    "contradict each other: the automatic rate exists to "
                    "replace an unset one. CatBoost resolves this by "
                    "silently keeping the explicit rate; give only one"
                )
            if l2_named:
                raise ValueError(
                    "auto_learning_rate=True with an explicit l2_leaf_reg "
                    "(lambda_l2, reg_lambda, l2_regularization) would do "
                    "nothing: CatBoost's derivation is gated on l2_leaf_reg "
                    "being unset (options_helper.cpp:280), so naming it "
                    "pins the rate back to the constant. Drop one of the two"
                )
            if leaf_iters_named:
                raise ValueError(
                    "auto_learning_rate=True with an explicit "
                    "leaf_estimation_iterations would do nothing: CatBoost's "
                    "derivation is gated on it being unset "
                    "(options_helper.cpp:279). Drop one of the two"
                )
            if boosting == "rf":
                raise ValueError(
                    "auto_learning_rate=True and boosting_type='rf' "
                    "contradict each other: a random forest averages its "
                    "trees and trains at a rate of 1.0 whatever it was "
                    "given, so a derived rate would be computed and thrown "
                    "away"
                )
        # A forest discards the rate, so there is nothing to derive for one
        # even when the request was only the mode default.
        enabled = wanted and not rate_named and boosting != "rf"
        return {
            "auto_learning_rate": int(enabled),
            "auto_learning_rate_required": int(enabled and required),
            "auto_learning_rate_l2_set": int(l2_named),
            "auto_learning_rate_leaf_iters_set": int(leaf_iters_named),
        }

    def _params(
        self,
        sample_weight_addr,
        device,
        ic_flat=None,
        ic_offsets=None,
        monotone_addr=0,
        categorical=None,
        contri_addr=0,
    ):
        # Every canonical name of docs/PARAMETER_NAMING.md and every other
        # vendor's spelling of it, resolved onto the LightGBM-spelled member
        # that holds the stock default and is what the native layer is sent.
        # Chained: see `_resolve_alias`.
        n_estimators = self._resolve_alias(
            "n_estimators", "num_iterations", 100
        )
        n_estimators = self._resolve_alias(
            "n_estimators", "num_boost_round", 100, n_estimators
        )
        n_estimators = self._resolve_alias(
            "n_estimators", "iterations", 100, n_estimators
        )
        n_estimators = self._resolve_alias(
            "n_estimators", "max_iter", 100, n_estimators
        )
        learning_rate_set = self._resolve_alias(
            "learning_rate", "eta", _LEARNING_RATE
        )
        learning_rate_set = self._resolve_alias(
            "learning_rate", "shrinkage_rate", _LEARNING_RATE,
            learning_rate_set,
        )
        num_leaves = self._resolve_alias("num_leaves", "max_leaves", 31)
        num_leaves = self._resolve_alias(
            "num_leaves", "max_leaf_nodes", 31, num_leaves
        )
        max_depth = self._resolve_alias("max_depth", "depth", -1)
        min_data_in_leaf = self._resolve_alias(
            "min_data_in_leaf", "min_child_samples", 20
        )
        min_data_in_leaf = self._resolve_alias(
            "min_data_in_leaf", "min_samples_leaf", 20, min_data_in_leaf
        )
        min_child_hess = self._resolve_alias(
            "min_child_hess", "min_child_weight", 1e-3
        )
        min_child_hess = self._resolve_alias(
            "min_child_hess", "min_sum_hessian_in_leaf", 1e-3, min_child_hess
        )
        lambda_l1 = self._resolve_alias("lambda_l1", "reg_alpha", _LAMBDA_L1)
        lambda_l2 = self._resolve_alias("lambda_l2", "reg_lambda", _LAMBDA_L2)
        lambda_l2 = self._resolve_alias(
            "lambda_l2", "l2_leaf_reg", _LAMBDA_L2, lambda_l2
        )
        lambda_l2 = self._resolve_alias(
            "lambda_l2", "l2_regularization", _LAMBDA_L2, lambda_l2
        )
        max_bin = self._resolve_alias("max_bin", "max_bins", 255)
        max_bin = self._resolve_alias("max_bin", "border_count", 255, max_bin)
        bagging_fraction = self._resolve_alias(
            "bagging_fraction", "subsample", 1.0
        )
        bagging_freq = self._resolve_alias(
            "bagging_freq", "subsample_freq", 0
        )
        min_gain_to_split = self._resolve_alias(
            "min_gain_to_split", "min_split_gain", 0.0
        )
        min_gain_to_split = self._resolve_alias(
            "min_gain_to_split", "gamma", 0.0, min_gain_to_split
        )
        monotone_penalty = self._resolve_alias(
            "monotone_penalty", "monotone_constraints_penalty", 0.0
        )
        drop_rate = self._resolve_alias("drop_rate", "rate_drop", 0.1)
        max_cat_to_onehot = self._resolve_alias(
            "max_cat_to_onehot", "one_hot_max_size", 4
        )
        # XGBoost's and CatBoost's spellings of the two feature-sampling
        # fractions. LightGBM accepts the XGBoost pair as aliases of
        # `feature_fraction` and `feature_fraction_bynode`, and so does this
        # estimator; `rsm` is CatBoost's per-tree share and `max_features`
        # is scikit-learn's per-split one.
        feature_fraction = self._resolve_alias(
            "feature_fraction", "colsample_bytree", 1.0
        )
        feature_fraction = self._resolve_alias(
            "feature_fraction", "rsm", 1.0, feature_fraction
        )
        feature_fraction_bynode = self._resolve_alias(
            "feature_fraction_bynode", "colsample_bynode", 1.0
        )
        feature_fraction_bynode = self._resolve_alias(
            "feature_fraction_bynode", "max_features", 1.0,
            feature_fraction_bynode,
        )
        # The seeds, the worker count, the log level, and the CatBoost-only
        # names. Each of these either sets something real or refuses by name;
        # none of them is accepted and dropped.
        seeds = self._resolve_seeds()
        self._check_n_jobs()
        self._check_catboost_only()
        self._check_langevin()
        # CatBoost's `bootstrap_type` (src/mojotrees/sampling.mojo), and the
        # one place in this method where a resolved value is UNRESOLVED
        # again. Under a real bootstrap type the number that `subsample`
        # folded onto `bagging_fraction` two blocks up was never the bagging
        # fraction: it was the MVS rate, and the pair would otherwise be two
        # bootstrap types at once. See `_resolve_bootstrap`.
        #
        # It runs before the implied-frequency fix below on purpose: with
        # `bagging_fraction` back at 1.0, that block cannot fire and turn a
        # bootstrap fit into a bagged one.
        bootstrap_knobs, bagging_fraction, bagging_freq = (
            self._resolve_bootstrap(bagging_fraction, bagging_freq)
        )
        # THE BEHAVIOR FIX. `subsample < 1` with `subsample_freq` unset is a
        # silent no-op in LightGBM: bagging never runs, and the user who
        # asked for it is told nothing. Here an unset frequency becomes 1
        # (bag every round), which is the only reading of "subsample=0.8"
        # that does what it says. An explicit `subsample_freq=0` still
        # disables bagging, because that is a statement and not an omission.
        #
        # It fires only when a fraction below 1 was actually asked for, so
        # no default configuration changes and no bits move on any fit that
        # does not name `subsample`/`bagging_fraction`.
        # `subsample_freq` is distinguishable from unset (it defaults to
        # None) and an explicit 0 there is honored. `bagging_freq` is not
        # (0 is its stock default), and an explicit `bagging_freq=0`
        # alongside a fraction below 1 is exactly the LightGBM no-op this
        # fixes, so it takes the implied 1 too.
        if (
            float(bagging_fraction) < 1.0
            and int(bagging_freq) == 0
            and self.subsample_freq is None
        ):
            bagging_freq = 1
        # Same ranges src/mojotrees/params.mojo and callback.mojo enforce,
        # so an estimator cannot construct a configuration the trainer
        # rejects (or, worse, quietly degenerates on).
        # Messages name the canonical parameter
        # (docs/PARAMETER_NAMING.md), never the member they resolved onto.
        if int(num_leaves) < 2:
            raise ValueError("max_leaves must be at least 2")
        if float(learning_rate_set) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if int(max_bin) < 2:
            raise ValueError("max_bin must be at least 2")
        if float(lambda_l1) < 0.0:
            raise ValueError("reg_alpha must be nonnegative")
        if float(lambda_l2) < 0.0:
            raise ValueError("reg_lambda must be nonnegative")
        if not 0.0 < float(bagging_fraction) <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if int(bagging_freq) < 0:
            raise ValueError("subsample_freq must be nonnegative")
        boosting = self._resolve_boosting()
        # CatBoost's ordered boosting (src/mojotrees/ordered_boosting.mojo).
        # It is not a fourth trainer, so it leaves here as `boosting="gbdt"`
        # plus its own bundle; `_parse_params` builds the
        # `OrderedBoostingParams` and refuses it on every entry point but the
        # dense single-output CPU one, which is the only trainer that grows
        # the fold ladder.
        ordered = boosting == "ordered"
        if ordered:
            boosting = "gbdt"
        ordered_knobs = self._resolve_ordered(
            ordered,
            int(seeds["ordered_seed"]),
            bagging_fraction,
            bagging_freq,
        )
        goss = boosting == "goss"
        if goss:
            top_rate = float(self.top_rate)
            other_rate = float(self.other_rate)
            if not 0.0 <= top_rate <= 1.0:
                raise ValueError("top_rate must be in [0, 1]")
            if not 0.0 <= other_rate <= 1.0:
                raise ValueError("other_rate must be in [0, 1]")
            if top_rate + other_rate > 1.0:
                raise ValueError("top_rate + other_rate must not exceed 1")
            if top_rate + other_rate <= 0.0:
                raise ValueError("top_rate + other_rate must be positive")
            if int(self.goss_seed) < 0:
                raise ValueError("goss_seed must be nonnegative")
            if int(self.goss_warmup_rounds) < -1:
                raise ValueError(
                    "goss_warmup_rounds must be -1 (automatic) or nonnegative"
                )
            # GOSS and row bagging both own the sampled row list. LightGBM
            # disables bagging under GOSS; mojotrees rejects the pair.
            if int(bagging_freq) > 0 and float(bagging_fraction) < 1.0:
                raise ValueError(
                    "boosting_type='goss' cannot be combined with row "
                    "bagging; leave subsample_freq at 0 or subsample at 1.0"
                )
        if boosting == "dart":
            if not 0.0 <= float(drop_rate) <= 1.0:
                raise ValueError("drop_rate must be in [0, 1]")
            if not 0.0 <= float(self.skip_drop) <= 1.0:
                raise ValueError("skip_drop must be in [0, 1]")
            if int(self.drop_seed) < 0:
                raise ValueError("drop_seed must be nonnegative")
        # A forest averages its trees, so LightGBM's RF ignores
        # learning_rate and trains at 1.0; the same here, whatever the
        # estimator was given, because boosting_rf refuses any other rate.
        learning_rate = 1.0 if boosting == "rf" else float(learning_rate_set)
        if not 0.0 < float(feature_fraction) <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if not 0.0 < float(feature_fraction_bynode) <= 1.0:
            raise ValueError("colsample_bynode must be in (0, 1]")
        grow_policy = self._resolve_grow_policy()
        # After `boosting` and `grow_policy` are both resolved, because the
        # mode default reads the grow policy and the forest refusal reads
        # the boosting strategy. Before the dict is built, so that a
        # contradiction raises rather than travelling.
        auto_learning_rate_knobs = self._auto_learning_rate_knobs(
            grow_policy, boosting
        )
        if int(max_cat_to_onehot) < 0:
            raise ValueError("max_cat_to_onehot must be nonnegative")
        if int(self.max_cat_threshold) < 1:
            raise ValueError("max_cat_threshold must be positive")
        if float(self.cat_smooth) < 0.0:
            raise ValueError("cat_smooth must be nonnegative")
        if float(self.cat_l2) < 0.0:
            raise ValueError("cat_l2 must be nonnegative")
        if int(self.min_data_per_group) < 1:
            raise ValueError("min_data_per_group must be positive")
        if not float(self.linear_lambda) >= 0.0:
            raise ValueError("linear_lambda must be nonnegative")
        return {
            # LightGBM's spelling on the wire, whatever the user typed: the
            # native side, the model files and `tools/check_parity.py` all
            # name these as LightGBM does, and only the user-facing layer is
            # renamed (docs/PARAMETER_NAMING.md).
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            # Sent as its canonical value; the binding parses it with the
            # same `growth_policy.parse_grow_policy` the parameter string
            # goes through, which is the one resolver every fit entry point
            # reaches.
            "grow_policy": grow_policy,
            "learning_rate": learning_rate,
            "n_estimators": int(n_estimators),
            "min_data_in_leaf": int(min_data_in_leaf),
            "lambda_l2": float(lambda_l2),
            "lambda_l1": float(lambda_l1),
            "min_child_hess": float(min_child_hess),
            "max_bin": int(max_bin),
            # int, not bool: the binding reads it as an integer.
            "use_missing": int(bool(self.use_missing)),
            "sample_weight_addr": int(sample_weight_addr),
            # The objective's scalar parameter, whichever of alpha, fair_c,
            # and tweedie_variance_power it is: one trainer slot holds it.
            "alpha": float(self._alpha_slot()),
            "device": device,
            "bagging_fraction": float(bagging_fraction),
            "bagging_freq": int(bagging_freq),
            "bagging_seed": int(seeds["bagging_seed"]),
            # int, not bool: the binding reads it as an integer.
            "goss": int(goss),
            "top_rate": float(self.top_rate),
            "other_rate": float(self.other_rate),
            "goss_seed": int(seeds["goss_seed"]),
            "goss_warmup_rounds": int(self.goss_warmup_rounds),
            # CatBoost's `bootstrap_type` group, read by `_parse_bootstrap`
            # in bindings/_mojotrees.mojo into the `BootstrapParams` that
            # `model.fit` hands to `boosting.train`. Emitted here rather than
            # refused above: the MVS and Bayesian draws are implemented
            # (src/mojotrees/sampling.mojo) and `boosting.train` has applied
            # them since the sampler landed, so a value that stopped at the
            # estimator was the one defect this whole sequence exists to
            # remove -- a parameter accepted and silently ignored. Every
            # entry point whose round loop does not call
            # `sampling.bootstrap_round` refuses an enabled bundle by name,
            # the GPU included.
            **bootstrap_knobs,
            # LightGBM's `boosting` by name; the binding routes dart and rf
            # to alternate_boosting.fit_boosting and leaves gbdt and goss on
            # the trainer they always used. The dart bundle is read only
            # when the mode is dart; ints, not bools, as everywhere here.
            "boosting": boosting,
            # CatBoost's ordered boosting (src/mojotrees/ordered_boosting.mojo),
            # read by `_parse_params` in bindings/_mojotrees.mojo into the
            # `OrderedBoostingParams` that rides on `BoosterParams.ordered`.
            # `ordered` is an int, not a bool, as everywhere here. Sent on
            # every fit, defaults included, because the native parser
            # subscripts the mapping rather than testing for a key.
            #
            # Until this group existed, `BoosterParams.ordered` was set by no
            # binding: the fold ladder was merged into `boosting.train`,
            # tested, and reachable from a hand-written Mojo call and from
            # nowhere else.
            **ordered_knobs,
            "drop_rate": float(drop_rate),
            "max_drop": int(self.max_drop),
            "skip_drop": float(self.skip_drop),
            "xgboost_dart_mode": int(bool(self.xgboost_dart_mode)),
            "uniform_drop": int(bool(self.uniform_drop)),
            "drop_seed": int(seeds["drop_seed"]),
            "feature_fraction": float(feature_fraction),
            "feature_fraction_bynode": float(feature_fraction_bynode),
            "feature_fraction_seed": int(seeds["feature_fraction_seed"]),
            "interaction_flat_addr": 0 if ic_flat is None else _addr(ic_flat),
            "interaction_flat_len": 0 if ic_flat is None else len(ic_flat),
            "interaction_offsets_addr": (
                0 if ic_offsets is None else _addr(ic_offsets)
            ),
            "interaction_offsets_len": (
                0 if ic_offsets is None else len(ic_offsets)
            ),
            "monotone_addr": int(monotone_addr),
            "categorical_addr": (
                0 if categorical is None else _addr(categorical)
            ),
            "categorical_len": 0 if categorical is None else len(categorical),
            "max_cat_to_onehot": int(max_cat_to_onehot),
            "max_cat_threshold": int(self.max_cat_threshold),
            "cat_smooth": float(self.cat_smooth),
            "cat_l2": float(self.cat_l2),
            "min_data_per_group": int(self.min_data_per_group),
            # The remaining LightGBM tree controls, read by
            # `extra_params_from_mapping` in bindings/basic_bindings.mojo
            # into the `ExtraTreeParams` that rides on `TreeParams.extra`
            # (src/mojotrees/tree_parameters_extra.mojo). Every key is sent
            # on every fit, inactive defaults included, because the parser
            # subscripts the mapping rather than testing for a key: a
            # missing one is a KeyError at the boundary, not a default.
            #
            # The ranges are not re-checked here. `ExtraTreeParams.check`
            # runs inside `tree.grow_tree`, and it is the same check the C
            # ABI and the CLI reach through params.mojo, so there is one
            # authority for what these values may be rather than a Python
            # copy of it that can drift.
            "min_gain_to_split": float(min_gain_to_split),
            # CatBoost's `leaf_estimation_iterations`: keep taking Newton
            # steps on a leaf's own rows after the structure is fixed
            # (`boosting._estimate_leaf_values`). 1, the default, is
            # LightGBM's behavior and is the early return in that function,
            # so 1 and "absent" are the same code path and the same bits.
            # Above 1 it is honored by `boosting.train`, `train_more` and
            # `train_gpu`, which `_mojotrees.fit` reaches, and refused by name
            # by every other entry point in `_parse_params` -- including the
            # eval_set one, which routes to `custom_metric.fit_with_metrics`
            # rather than to `boosting.train_with_valid` despite the name.
            "leaf_estimation_iterations": (
                1
                if self.leaf_estimation_iterations is None
                else int(self.leaf_estimation_iterations)
            ),
            # Provenance, not a value. `_parse_params` resolves an UNSET
            # count through `boosting.catboost_leaf_estimation_iterations`
            # when the fit is in CatBoost mode, and "unset" cannot be
            # recovered from the 1 above: a caller who types 1 has decided
            # something and a caller who types nothing has not. That is
            # CatBoost's own gate, which reads `TOption::NotSet()` rather
            # than comparing against a default (`option.h:80-85`).
            "leaf_estimation_iterations_set": int(
                self.leaf_estimation_iterations is not None
            ),
            # LightGBM's and CatBoost's `boost_from_average`, sent as a
            # value plus its provenance for the same reason. The value
            # itself is True when unset, which is LightGBM's default
            # (config.h:948) and is what `boosting._base_score` has always
            # done, so a fit that never names it makes the identical call it
            # made before this key existed; the native side replaces it with
            # CatBoost's per-objective answer only when the mode default
            # applies and the entry point can honor it.
            "boost_from_average": int(
                True
                if self.boost_from_average is None
                else bool(self.boost_from_average)
            ),
            "boost_from_average_set": int(
                self.boost_from_average is not None
            ),
            # Whether the two keys above may take a CatBoost per-objective
            # default. `symmetrictree` is CatBoost mode and mirrors CatBoost;
            # `lossguide` and `depthwise` mirror LightGBM, which resolves
            # neither parameter per loss. One key for both, because it is one
            # fact about the fit and two copies of it could disagree.
            "catboost_mode_defaults": int(grow_policy == "symmetrictree"),
            # CatBoost's `score_function`: which functional a split candidate
            # is scored by (`split.SCORE_L2` / `split.SCORE_COSINE`).
            # Lowercased here and nowhere else -- the native
            # `parse_score_function` takes canonical lowercase, the same
            # contract `device_type` and `bootstrap_type` keep -- so one
            # spelling of a value resolves to one code no matter which
            # surface it was typed at. `"l2"` is the default and is the
            # native default, so a fit that leaves the parameter None makes
            # the call it made before this key existed.
            #
            # An unknown name raises in `parse_score_function`, and
            # `"cosine"` raises by entry point in `_parse_params` for the
            # one trainer that cannot honor it (`distributed_train_local`,
            # i.e. tree_learner other than 'serial'). Neither is accepted
            # and dropped.
            "score_function": (
                "l2"
                if self.score_function is None
                else str(self.score_function).lower()
            ),
            # CatBoost's per-split score noise. Emitted here rather than
            # left out: the noise and its per-tree scale are both implemented
            # and the dense CPU round loops now compute the scale, so a value
            # that did not reach the binding would be accepted and silently
            # dropped -- the defect this whole sequence exists to remove. The
            # binding refuses it by entry point for the loops that do not
            # compute a scale.
            # CatBoost's automatic learning rate (catalog A12/A38). Four ints,
            # sent on every fit like everything else here, because the native
            # parser subscripts the mapping rather than testing for a key.
            #
            # The rate itself is NOT derived here: the formula and the
            # coefficient table live in src/mojotrees/auto_learning_rate.mojo
            # and the derivation needs the train row count, which `_params`
            # does not have and each fit entry point does. What crosses here
            # is the request plus the provenance the native side cannot
            # recover from a resolved float. `_parse_params` in
            # bindings/_mojotrees.mojo turns it into an
            # `AutoLearningRateParams` and calls `resolve_learning_rate`; the
            # entry points that cannot supply a row count, an iteration count
            # and a loss function refuse it by name there.
            **auto_learning_rate_knobs,
            "random_strength": float(self.random_strength or 0.0),
            # The seed that noise stream is keyed from. Through
            # `_resolve_seeds` like every other per-component seed, so
            # `random_state` reaches it by the same rule: an explicitly named
            # seed wins, an unnamed one follows the global. Inert whenever
            # `random_strength` is 0, which is the default, so this key costs
            # an unnoised fit nothing.
            "random_strength_seed": int(seeds["random_strength_seed"]),
            "max_delta_step": float(self.max_delta_step),
            "path_smooth": float(self.path_smooth),
            # int, not bool: the binding reads it as an integer.
            "extra_trees": int(bool(self.extra_trees)),
            "extra_seed": int(seeds["extra_seed"]),
            "monotone_penalty": float(monotone_penalty),
            "monotone_constraints_method": str(
                self.monotone_constraints_method
            ),
            "feature_contri_addr": int(contri_addr),
            "cegb_tradeoff": float(self.cegb_tradeoff),
            "cegb_penalty_split": float(self.cegb_penalty_split),
            # LightGBM's linear_tree / linear_lambda
            # (src/mojotrees/linear_tree.mojo): each leaf emits an affine
            # function of the numerical features on its branch, fitted after
            # growth. int, not bool: the binding reads it as an integer.
            "linear_tree": int(bool(self.linear_tree)),
            "linear_lambda": float(self.linear_lambda),
            # Both always 0. `cegb_penalty_feature_coupled` and
            # `cegb_penalty_feature_lazy` are applied by the trainer now
            # (src/mojotrees/cegb.mojo, charged against the per-ensemble
            # `CegbLedger` that `boosting.train` owns), but they are
            # per-feature vectors and this estimator has no parameter that
            # carries one for them yet; the Mojo API reaches them through
            # `TreeParams.extra.penalties.cegb`. The keys are sent on every
            # fit because the native parser subscripts the mapping rather
            # than testing for a key.
            "cegb_penalty_feature_coupled_addr": 0,
            "cegb_penalty_feature_lazy_addr": 0,
            "forced_splits": self._forced_splits_text(),
            # Exclusive feature bundling, read by
            # `efb_settings_from_mapping` in bindings/basic_bindings.mojo
            # into the `EfbSettings` that rides on `BoosterParams.bundling`
            # (src/mojotrees/efb.mojo). Sent on every fit for the same
            # reason the block above is, and range-checked in the same one
            # place: `EfbSettings.check`, which the C ABI and the CLI reach
            # through params.mojo.
            #
            # Which trainers may honor the switch is decided at the
            # boundary, in `_parse_params`, because that is where the
            # trainer about to run is known. A trainer that would ignore it
            # raises instead.
            # A knob set to None takes the native default
            # (`efb_defaults`), so LightGBM's numbers have one home.
            "enable_bundle": int(bool(self.enable_bundle)),
            **_preflight.bundling_knobs(
                max_conflict_rate=self.max_conflict_rate,
                max_bundle_bins=self.max_bundle_bins,
                max_bundle_size=self.max_bundle_size,
                max_nondefault_rate=self.max_nondefault_rate,
                min_reduction=self.min_reduction,
                bundle_missing=self.bundle_missing,
            ),
        }

    def _resolve_device(
        self,
        n_rows,
        n_features,
        n_outputs,
        objective_code=None,
        sparse=False,
        categorical=False,
        has_eval_set=False,
    ):
        """The backend a *fit* will actually run on, "cpu" or "gpu". Names
        are case-insensitive, as LightGBM treats `device_type`. Raises
        ValueError for an unknown `device` and RuntimeError when "gpu" is
        requested but unavailable or unsupported; "gpu" never falls back to
        the CPU.

        Prediction does not come through here. Where a prediction runs is
        decided by the same native policy, but from inside the prediction
        entry point, which is the only place that knows the model's bin
        count and whether the GPU predictor covers the request; see
        `_device_request`.

        Everything after `n_outputs` is what the native policy gates on
        beyond the shape, and every one of them changes an answer:
        `objective_code` blocks the GPU for a custom objective and for
        lambdarank, `sparse` routes an explicit "gpu" to the sparse GPU
        trainer and keeps "auto" on the CPU, `has_eval_set` blocks it
        because validation metrics are scored on the host, and `max_bin`
        (read off the estimator) and `categorical` and `use_missing` are
        reported. Leaving one
        undeclared does not make it false, it makes the decision
        incomplete, which the report says. `objective_code=None` is
        undeclared, which is what the multiclass classifier means: its
        trees-per-round is the fact that matters and `n_outputs` carries
        it.

        No decision is made in Python. `mojotrees.device_selection` is the
        one Python door to the native policy and holds no policy of its
        own; the direct `_mojotrees.resolve_device` call below reaches the
        same engine without the report, and is both the fallback for a
        build whose `device_selection` cannot be imported and the reason
        the callers keep their own guards (see `_gpu_unsupported`).
        """
        device = self._resolve_alias("device", "device_type", "cpu")
        device = self._resolve_alias("device", "task_type", "cpu", device)
        # XGBoost's older switch names an algorithm and a device at once.
        # Only the two values that name the histogram algorithm mojotrees
        # implements resolve; `exact` and `approx` are different split
        # searches, not spellings of `device`, and are refused by name.
        if self.tree_method is not None:
            method = str(self.tree_method).strip().lower()
            if method in ("exact", "approx"):
                raise ValueError(
                    f"tree_method={self.tree_method!r} is a different split "
                    "search, not a spelling of device: mojotrees searches a "
                    "histogram, which is XGBoost's 'hist'. Use device='cpu' "
                    "or device='gpu'."
                )
            mapped = {"hist": "cpu", "auto": "auto", "gpu_hist": "gpu"}.get(
                method
            )
            if mapped is None:
                raise ValueError(
                    f"unknown tree_method {self.tree_method!r}; expected "
                    "'hist', 'gpu_hist', or 'auto'. device='cpu' and "
                    "device='gpu' are the canonical spellings."
                )
            if device != "cpu" and device != mapped:
                raise ValueError(
                    f"device={device!r} and tree_method="
                    f"{self.tree_method!r} are aliases with different "
                    "values; set only one"
                )
            device = mapped
        if not isinstance(device, str) or device.lower() not in _DEVICES:
            raise ValueError(
                f"unknown device {device!r}; expected one of "
                + ", ".join(_DEVICES)
            )
        device = device.lower()
        try:
            from . import device_selection as _policy
        except Exception:
            _policy = None
        if _policy is not None:
            workload = _policy.Workload(
                n_rows,
                n_features,
                objective_code=objective_code,
                n_classes=n_outputs,
                max_bin=int(
                    self._resolve_alias("max_bin", "max_bins", 255)
                ),
                sparse=bool(sparse),
                categorical=bool(categorical),
                has_missing=bool(self.use_missing),
                has_eval_set=bool(has_eval_set),
                # Read through `_resolve_boosting()` rather than off
                # `self.boosting_type`, because `boosting` and `booster` are
                # aliases for the same parameter: a user who wrote
                # `booster='ordered'` would leave `self.boosting_type` None,
                # and this gate would have seen plain boosting for an ordered
                # fit. `_resolve_boosting` is where the three spellings
                # already become one word.
                ordered_boosting=(self._resolve_boosting() == "ordered"),
                # `getattr` with a default, and deliberately, because the
                # CPU campaign is landing the field that carries a non-L2
                # choice; until it does, `self.score_function` exists but no
                # value other than L2 gets past the refusal below it. Written
                # this way the gate is correct before and after that lands,
                # and it fails closed: an unrecognized spelling maps to
                # SCORE_COSINE, which the device policy blocks, rather than
                # to SCORE_L2, which it waves through.
                score_function=_score_function_code(
                    getattr(self, "score_function", None)
                ),
                # `or 0.0` because the estimator's default is None, and the
                # same expression already builds the params dict below, so the
                # gate and the fit read the parameter the same way. A value
                # this gate could not see would be the defect it exists to
                # close.
                random_strength=float(
                    getattr(self, "random_strength", None) or 0.0
                ),
            )
            # DeviceUnavailableError is a RuntimeError subclass carrying the
            # native refusal text, so it propagates as what this method has
            # always raised, with the report attached.
            return _policy.select_device(device, workload).resolved
        # The narrow contract answers on shape alone, so it cannot see
        # `boosting_type` or `score_function` any more than it sees
        # `enable_bundle` or `linear_tree`. That is not a hole this branch can
        # close: a build old enough to expose `resolve_device` and not
        # `decide_device` predates `BLOCK_ORDERED_BOOSTING` and
        # `BLOCK_SCORE_FUNCTION` entirely, so there is no native gate here to
        # reach. The trainer-side refusals are what protect this path --
        # `_check_gpu_booster_params` in train_gpu.mojo and `_refuse_unhonored`
        # in train_gpu_sparse.mojo both raise on `params.ordered.enabled`
        # whatever resolved the device.
        try:
            return _mojotrees.resolve_device(
                device, int(n_rows), int(n_features), int(n_outputs)
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from None

    def _gpu_unsupported(self, device, lead, hint=None):
        """Backstop refusal for a request no accelerator kernel covers.

        Every call site here mirrors a block in
        src/mojotrees/device_policy.mojo, which is what actually decides
        once `_resolve_device` reaches the full native contract:
        BLOCK_VALIDATION_SET, BLOCK_CUSTOM_OBJECTIVE, and
        BLOCK_RANKING_OBJECTIVE (BLOCK_SPARSE_INPUT is a prediction-side
        block now; sparse training has a device path). Against such a build these checks
        never fire, because `device` arrived already refused or already
        resolved to "cpu".

        They stay because the narrow contract exists. A build that exposes
        `resolve_device` but not `decide_device` answers on shape alone,
        so an explicit device="gpu" would otherwise reach a trainer with no
        kernel for the request and either fall back silently or fail
        somewhere less legible. `device="auto"` resolves to the CPU either
        way, which is what the native policy picks too, so only an explicit
        request is refused.
        """
        if device == "cpu":
            return
        message = f"{lead}; use device='cpu' or device='auto'"
        if hint is not None:
            message += f". {hint}"
        raise RuntimeError(message)

    def _weight_buffer(self, sample_weight, n_rows):
        """Validated weight buffer and its address (buffer must stay
        referenced while the address is in use); (None, 0) when absent.
        Weights must be finite, nonnegative, and not all zeros."""
        if sample_weight is None:
            return None, 0
        wb = _arrays.check_sample_weight(sample_weight, n_rows)
        return wb, _addr(wb)

    def _registry_objective_name(self):
        """The objective spelling `_eval.default_metric` can look up.

        The default `eval_metric` comes from a registry table keyed by
        LightGBM's objective aliases, so a vendor spelling
        (`reg:squarederror`, `RMSE`) would find nothing there and a fit with
        an `eval_set` and no `eval_metric` would fail on the spelling rather
        than on anything real. This resolves such a spelling back to a
        LightGBM alias of the same objective, derived from the estimator's
        own table rather than from a second list: the code is looked up in
        `_OBJECTIVES`, then the first spelling in `_OBJECTIVES` that the
        registry itself resolves to that code is returned.

        Estimators with no `_OBJECTIVES` (the classifier and the ranker)
        pass their spelling through; their tasks carry a task-level default
        metric, so the objective is not consulted.
        """
        resolve = getattr(self, "_resolve_objective", None)
        objective = resolve() if resolve is not None else None
        if objective is None or callable(objective):
            return objective
        key = str(objective).strip().lower()
        table = getattr(self, "_OBJECTIVES", None)
        if not table:
            return key
        if _objective_status(key) == "supported":
            return key
        code = table.get(key)
        if code is None:
            return key
        for name in sorted(table):
            if _objective_code_of_name(name) == code:
                return name
        return key

    def _alpha_slot(self):
        """The number the trainer's one objective-parameter slot carries.

        The regressor resolves it from the objective (`alpha`, `fair_c`, or
        `tweedie_variance_power`); the classifier and the ranker have no
        such parameter and pass LightGBM's default through, which their
        objectives ignore.
        """
        resolve = getattr(self, "_objective_param", None)
        if resolve is None:
            return float(getattr(self, "alpha", 0.9))
        return float(resolve())

    def _metric_objective(self, task):
        """The objective code whose inverse link the built-in metrics apply
        to the raw validation scores.

        The multiclass trainer's metrics take the softmax themselves, so the
        code is unread there; ranking and custom objectives have no link,
        which is the identity this returns for them.
        """
        if task == _eval.MULTICLASS:
            return _SQUARED_ERROR
        if task == _eval.RANKING:
            return _LAMBDARANK
        if task == _eval.BINARY:
            return _BINARY_LOGISTIC
        resolve = getattr(self, "_objective_code", None)
        return _SQUARED_ERROR if resolve is None else int(resolve())

    # -- validation sets and custom metrics -------------------------------

    def _eval_sets(
        self,
        eval_set,
        eval_names,
        n_features,
        eval_sample_weight=None,
        eval_group=None,
        encode=None,
    ):
        """Validated validation sets: the buffers to keep alive, the
        `(name, x_addr, n_rows, y_addr)` specs the binding reads, the label
        vectors the callbacks receive, the row counts, and the per-set
        weight and group buffers (None where they were not given).

        `encode` maps a set's labels through the same encoding the training
        labels went through, which is what the classifier needs and what
        makes an unseen validation label an error rather than a silent
        miscount.
        """
        pairs = list(eval_set)
        if not pairs:
            raise ValueError("eval_set must not be empty")
        if eval_names is None:
            names = [f"valid_{i}" for i in range(len(pairs))]
        else:
            names = [str(name) for name in eval_names]
            if len(names) != len(pairs):
                raise ValueError(
                    "eval_names must have one name per eval_set entry"
                )
        if len(set(names)) != len(names):
            raise ValueError("eval_names must be unique")
        set_weights = _per_set(
            eval_sample_weight, len(pairs), "eval_sample_weight"
        )
        set_groups = _per_set(eval_group, len(pairs), "eval_group")
        keep = []
        specs = []
        targets = []
        rows = []
        weights = []
        groups = []
        for name, pair, weight, group in zip(
            names, pairs, set_weights, set_groups
        ):
            try:
                X_valid, y_valid = pair
            except (TypeError, ValueError):
                raise ValueError(
                    "each eval_set entry must be an (X, y) pair"
                ) from None
            label = f"eval_set {name!r}"
            Xb, n_valid_rows, n_valid_features, valid_names = _arrays.check_X(
                X_valid, encoders=self._matrix_encoders(X_valid, label)
            )
            if n_valid_features != n_features:
                raise ValueError(
                    f"eval_set {name!r} has {n_valid_features} features, but "
                    f"X has {n_features}"
                )
            self._check_category_codes(
                Xb,
                n_valid_rows,
                getattr(self, "_cat_indices", ()),
                valid_names,
                label,
            )
            if encode is None:
                yb = _arrays.check_target(y_valid, n_valid_rows)
            else:
                yb = encode(y_valid, n_valid_rows, label)
            wb = (
                None
                if weight is None
                else _arrays.check_sample_weight(weight, n_valid_rows)
            )
            gb = None if group is None else _group_buffer(group, n_valid_rows)
            keep.append((Xb, yb, wb, gb))
            specs.append((name, _addr(Xb), n_valid_rows, _addr(yb)))
            targets.append(yb)
            rows.append(n_valid_rows)
            weights.append(wb)
            groups.append(gb)
        return keep, specs, targets, rows, weights, groups

    def _callback_params(self):
        """The resettable hyperparameters as a callback first sees them.

        Only the names in `callback.RESETTABLE` appear: `env.params` is the
        set a before-iteration callback may schedule, so listing anything
        else would invite a reset that cannot be honored. Aliases are
        resolved here, once, the way `_params` resolves them.
        """
        # `callback.RESETTABLE` is a native vocabulary and keeps LightGBM's
        # names, as every wire does; the values are resolved from the
        # canonical user-facing names the same way `_params` resolves them.
        learning_rate = self._resolve_alias(
            "learning_rate", "eta", _LEARNING_RATE
        )
        learning_rate = self._resolve_alias(
            "learning_rate", "shrinkage_rate", _LEARNING_RATE, learning_rate
        )
        num_leaves = self._resolve_alias("num_leaves", "max_leaves", 31)
        num_leaves = self._resolve_alias(
            "num_leaves", "max_leaf_nodes", 31, num_leaves
        )
        min_data_in_leaf = self._resolve_alias(
            "min_data_in_leaf", "min_child_samples", 20
        )
        min_data_in_leaf = self._resolve_alias(
            "min_data_in_leaf", "min_samples_leaf", 20, min_data_in_leaf
        )
        min_child_hess = self._resolve_alias(
            "min_child_hess", "min_child_weight", 1e-3
        )
        min_child_hess = self._resolve_alias(
            "min_child_hess", "min_sum_hessian_in_leaf", 1e-3, min_child_hess
        )
        lambda_l2 = self._resolve_alias("lambda_l2", "reg_lambda", _LAMBDA_L2)
        lambda_l2 = self._resolve_alias(
            "lambda_l2", "l2_leaf_reg", _LAMBDA_L2, lambda_l2
        )
        lambda_l2 = self._resolve_alias(
            "lambda_l2", "l2_regularization", _LAMBDA_L2, lambda_l2
        )
        feature_fraction = self._resolve_alias(
            "feature_fraction", "colsample_bytree", 1.0
        )
        feature_fraction = self._resolve_alias(
            "feature_fraction", "rsm", 1.0, feature_fraction
        )
        feature_fraction_bynode = self._resolve_alias(
            "feature_fraction_bynode", "colsample_bynode", 1.0
        )
        feature_fraction_bynode = self._resolve_alias(
            "feature_fraction_bynode", "max_features", 1.0,
            feature_fraction_bynode,
        )
        return {
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "max_depth": int(self._resolve_alias("max_depth", "depth", -1)),
            "min_data_in_leaf": int(min_data_in_leaf),
            # The environment uses LightGBM's name for this one; the
            # canonical spelling of it is `min_child_weight`.
            "min_sum_hessian_in_leaf": float(min_child_hess),
            "lambda_l1": float(
                self._resolve_alias("lambda_l1", "reg_alpha", _LAMBDA_L1)
            ),
            "lambda_l2": float(lambda_l2),
            "feature_fraction": float(feature_fraction),
            "feature_fraction_bynode": float(feature_fraction_bynode),
        }

    def _fit_with_metrics(
        self,
        Xb,
        yb,
        n_rows,
        n_features,
        params,
        device,
        objective,
        eval_set,
        eval_names,
        eval_metric,
        early_stopping_rounds,
        min_delta,
        primary_metric,
        eval_sample_weight=None,
        eval_group=None,
        task=_eval.REGRESSION,
        n_classes=0,
        encode=None,
        callbacks=None,
    ):
        """Train while metrics score the validation sets.

        `eval_metric` holds LightGBM metric names, callables, or both, and
        defaults to the metric LightGBM would score for this task and
        objective. A built-in name is evaluated by `_mojotrees.eval_metric`,
        which calls src/mojotrees/metrics.mojo, so the Python API never
        recomputes a metric the library already defines; `eval_sample_weight`
        weights those, one vector per validation set.

        A callable is called once per validation set per round as
        `metric(y_true, y_pred)`, where `y_pred` holds raw scores (log-odds
        for the binary classifier, one row-major block of `n_classes` per
        row for the softmax one), matching LightGBM's `feval`. It returns a
        float, or LightGBM's `(name, value, is_higher_better)` triple, of
        which only the value is read: the direction is declared in
        `eval_metric`. `y_pred` is a view on a buffer the trainer reuses
        every round, so read it, do not keep it.

        `task` picks the trainer and the metrics that make sense for it:
        the softmax trainer for `_eval.MULTICLASS`, the LambdaRank one for
        `_eval.RANKING`, and the single-output one otherwise.

        Sets `evals_result_`, `best_score_`, `stopped_early_`, and the
        `_metric_*` fields `_record_fit` turns into `best_iteration_` and
        `n_iter_`; see src/mojotrees/custom_metric.mojo for the
        early-stopping rules.
        """
        # Backstop; BLOCK_VALIDATION_SET is what refuses this on a build
        # whose native policy can be asked. See `_gpu_unsupported`.
        self._gpu_unsupported(
            device, "validation metrics are scored on the CPU"
        )
        specs = _metric_specs(
            eval_metric, task, self._registry_objective_name()
        )
        keep, valid_specs, targets, rows, weights, groups = self._eval_sets(
            eval_set,
            eval_names,
            n_features,
            eval_sample_weight,
            eval_group,
            encode,
        )
        primary = _primary_index(primary_metric, specs)
        callbacks = list(callbacks or ())
        # `verbose` is the canonical name for the log level (scikit-learn's
        # and CatBoost's word; LightGBM and XGBoost spell it `verbosity`).
        # Nothing in mojotrees writes a training log on its own, so a
        # positive value is honored the only way it can be: by installing
        # the callback that does. A value of 0 or below asks for the silence
        # this already is. This is the only place `verbose` acts, which is
        # also the only place there is anything to report.
        period = self._resolve_alias("verbose", "verbosity", None)
        if period is not None and not isinstance(period, bool):
            period = int(period)
        if period is True:
            period = 1
        if period and int(period) > 0:
            callbacks = callbacks + [_callback.log_evaluation(int(period))]
        # `early_stopping_rounds` given on the estimator is the default for
        # the fit-time argument of the same name, which is XGBoost's and
        # CatBoost's shape (both take it in the constructor) as well as
        # LightGBM's `early_stopping_round` parameter. An explicit fit-time
        # value wins, because it is the more specific statement.
        if not early_stopping_rounds:
            estimator_rounds = self._resolve_alias(
                "early_stopping_rounds", "early_stopping_round", None
            )
            estimator_rounds = self._resolve_alias(
                "early_stopping_rounds", "od_wait", None, estimator_rounds
            )
            estimator_rounds = self._resolve_alias(
                "early_stopping_rounds",
                "n_iter_no_change",
                None,
                estimator_rounds,
            )
            if estimator_rounds is not None:
                early_stopping_rounds = estimator_rounds
        (
            early_stopping_rounds,
            min_delta,
            first_metric_only,
            stopper,
        ) = _callback.resolve_early_stopping(
            callbacks, early_stopping_rounds, min_delta
        )
        # The per-iteration hook lives in the single-output trainer
        # (train_with_callbacks). The softmax and LambdaRank loops score
        # metrics but have no hook yet, so a callback list there is refused
        # rather than accepted and ignored.
        if callbacks and task in (_eval.MULTICLASS, _eval.RANKING):
            raise NotImplementedError(
                "callbacks are not wired into the multiclass and ranking "
                "trainers yet; they run for regression and binary "
                "classification. early_stopping_rounds= works for every task."
            )
        rounds = _early_stopping_rounds(early_stopping_rounds)
        if float(min_delta) < 0.0:
            raise ValueError("min_delta must not be negative")
        codes = [spec[4] for spec in specs]
        funcs = [spec[1] for spec in specs]
        if any(w is not None for w in weights) and any(
            code is None for code in codes
        ):
            raise ValueError(
                "eval_sample_weight weights the built-in eval metrics; a "
                "callable metric is handed unweighted predictions, so apply "
                "the weights inside it instead"
            )
        # One block of raw scores per validation row, n_classes wide for the
        # softmax trainer.
        width = n_classes if task == _eval.MULTICLASS else 1
        pred = _out_buffer(max(rows) * width)
        eval_params = [
            {
                "pred_addr": _addr(pred),
                "y_addr": _addr(targets[v]),
                "weight_addr": (
                    0 if weights[v] is None else _addr(weights[v])
                ),
                "n_rows": rows[v],
                "n_classes": int(n_classes),
                "group_addr": 0 if groups[v] is None else _addr(groups[v]),
                "n_groups": 0 if groups[v] is None else len(groups[v]),
                "ndcg_at": int(getattr(self, "ndcg_eval_at", 5)),
                "alpha": float(self._alpha_slot()),
                # The metric applies the objective's inverse link, so it
                # scores what predict() would return; see eval_metric in
                # bindings/_mojotrees.mojo.
                "objective": int(self._metric_objective(task)),
            }
            for v in range(len(rows))
        ]

        # A Python exception cannot cross the Mojo boundary as itself: it
        # arrives on the other side as a message-shaped Exception, losing
        # the type the caller wants to catch. Both callback kinds therefore
        # keep the object here and let `fit` re-raise it; see the same
        # pattern for iteration callbacks in python/mojotrees/callback.py.
        metric_failure = []

        def bridge(metric_index, valid_index):
            try:
                code = codes[metric_index]
                if code is not None:
                    return float(
                        _mojotrees.eval_metric(code, eval_params[valid_index])
                    )
                value = funcs[metric_index](
                    targets[valid_index], pred[: rows[valid_index] * width]
                )
                if isinstance(value, tuple):
                    # LightGBM's feval returns
                    # (name, value, is_higher_better).
                    value = value[1]
                return float(value)
            except BaseException as exc:
                metric_failure.append(exc)
                raise

        params["pred_addr"] = _addr(pred)
        params["valid_sets"] = valid_specs
        params["n_valid"] = len(valid_specs)
        params["metrics"] = [
            # ints, not bools: the binding reads the flags as integers.
            # first_metric_only narrows the watch to eval_metric's first
            # entry, which is what LightGBM's flag of that name does.
            (
                spec[0],
                int(spec[2]),
                int(spec[3] and (m == 0 or not first_metric_only)),
            )
            for m, spec in enumerate(specs)
        ]
        params["n_metrics"] = len(specs)
        params["primary_metric"] = primary
        params["early_stopping_rounds"] = rounds
        params["min_delta"] = float(min_delta)

        # Two small buffers carry the per-iteration traffic: the round's
        # resettable hyperparameters out and back, and the round's metric
        # values in. Both are allocated even with no callbacks so the bridge
        # never sees a null address; `has_callback` is what keeps a run
        # without callbacks from crossing the boundary at all.
        reset_buf = _out_buffer(len(_callback.RESETTABLE))
        evals_buf = _out_buffer(max(len(valid_specs) * len(specs), 1))
        runner = None
        if callbacks:
            runner = _callback.CallbackRunner(
                callbacks,
                self,
                self._callback_params(),
                [spec[0] for spec in valid_specs],
                [spec[0] for spec in specs],
                [bool(spec[2]) for spec in specs],
                reset_buf,
                evals_buf,
            )
            runner.end_iteration = int(
                self._resolve_alias("n_estimators", "num_iterations", 100)
            )
        params["callback"] = runner
        params["has_callback"] = int(runner is not None)
        params["reset_addr"] = _addr(reset_buf)
        params["evals_addr"] = _addr(evals_buf)
        try:
            if task == _eval.MULTICLASS:
                result = _mojotrees.fit_multiclass_with_metrics(
                    _addr(Xb),
                    n_rows,
                    n_features,
                    _addr(yb),
                    int(n_classes),
                    bridge,
                    params,
                )
            elif task == _eval.RANKING:
                result = _mojotrees.fit_ranker_with_metrics(
                    _addr(Xb), n_rows, n_features, _addr(yb), bridge, params
                )
            else:
                result = _mojotrees.fit_with_metrics(
                    _addr(Xb),
                    n_rows,
                    n_features,
                    _addr(yb),
                    objective,
                    bridge,
                    params,
                )
        except BaseException:
            # A callback's exception cannot cross the Mojo boundary as
            # itself, so the runner kept the object and the boundary carried
            # a control code. Re-raise the original: the caller should catch
            # its own exception type, not a message-shaped RuntimeError.
            # `_reset_fitted` already ran, so the estimator stays unfitted.
            if metric_failure:
                raise metric_failure[0] from None
            if runner is not None and runner.error is not None:
                raise runner.error from None
            raise
        self._model = result[0]
        values = result[1]
        n_rounds = int(result[2])
        self._metric_best_iteration = int(result[3])
        # The history counts the base-score-only model as round 0, so one
        # fewer round was actually trained than it holds entries for.
        self._metric_n_iter = max(n_rounds - 1, 0)
        self.best_score_ = float(result[4])
        self.stopped_early_ = bool(result[5])
        n_valid = len(valid_specs)
        n_metrics = len(specs)
        self.evals_result_ = {
            valid_specs[v][0]: {
                specs[m][0]: [
                    float(values[(r * n_valid + v) * n_metrics + m])
                    for r in range(n_rounds)
                ]
                for m in range(n_metrics)
            }
            for v in range(n_valid)
        }
        if stopper is not None and self.stopped_early_:
            # The trainer, not the callback, decided which round won, so the
            # callback reports only once that is known.
            stopper.report(
                self._metric_best_iteration,
                self.best_score_,
                specs[primary][0],
                valid_specs[0][0],
            )
        # The validation buffers had to outlive the call above.
        del keep

    # -- fitted state ----------------------------------------------------

    # -- the fitted model ------------------------------------------------
    #
    # There is one model object in this package, `mojotrees.Booster`, and an
    # estimator holds one rather than a second abstraction of its own: the
    # opaque handle the extension module returns lives in that Booster and
    # nowhere else. `_model` stays the spelling the estimator code uses for
    # the handle, so assigning a freshly trained one wraps it and reading it
    # unwraps it.

    @property
    def _model(self):
        booster = self.__dict__.get("_booster")
        return None if booster is None else booster._handle

    @_model.setter
    def _model(self, handle):
        self.__dict__["_booster"] = (
            None if handle is None else Booster._from_estimator(handle, self)
        )

    @property
    def booster_(self):
        """The fitted model, as the `Booster` the functional API returns.

        LightGBM's `booster_`. Everything a model can answer for itself is
        on it: `predict`, `eval`, `feature_importance`, `save_model`,
        `model_to_string`, `current_iteration`, `num_trees`. What it cannot
        do is continue training, because an estimator bins its own training
        matrix and does not keep the `Dataset` that `update()` would grow
        on; `mojotrees.train()` keeps one.
        """
        self._require_fitted()
        booster = self.__dict__["_booster"]
        # `fit` records the feature names after it has the model, so they
        # are copied across on the way out rather than at wrap time.
        names = getattr(self, "feature_names_in_", None)
        booster._names = None if names is None else [str(n) for n in names]
        return booster

    # -- LightGBM's fitted attributes that the model already answers -------
    #
    # These three are properties rather than entries in `_FITTED_ATTRS`
    # because their source is the model, not the fit: an estimator loaded
    # with `load()` or unpickled answers them, and nothing has to be kept
    # in step with them. `mojotrees.inspection` is the single reader; it
    # is imported inside each property so that the top-level import does
    # not pay for a submodule most callers never touch.

    @property
    def objective_(self):
        """LightGBM's `objective_`: the resolved objective's canonical
        name.

        Resolved, not echoed. It comes from the objective code the fitted
        model carries, so an estimator constructed with an alias (`mae`)
        reports the canonical spelling (`regression_l1`), a softmax
        classifier reports `multiclass`, and a callable objective reports
        `custom`.

        Until `_mojotrees.objective_code` is bound, reading this costs a
        `model_to_string()` round trip (see `inspection.objective_of`), so
        it is a fitted attribute worth reading once rather than per row.
        """
        self._require_fitted()
        from . import inspection

        return inspection.objective_of(self)

    @property
    def feature_name_(self):
        """LightGBM's `feature_name_`: the training feature names, or
        `Column_0`, `Column_1`, ... when the model carries none.

        `feature_names_in_` is scikit-learn's attribute and exists only
        when the training matrix carried names; this one always exists on a
        fitted model, which is the difference between the two.
        """
        self._require_fitted()
        from . import inspection

        return inspection.feature_name_of(self)

    @property
    def n_features_(self):
        """LightGBM's `n_features_`: the feature count the model was fitted
        on, read from the model rather than from a fit-time attribute."""
        self._require_fitted()
        from . import inspection

        return inspection.n_features_of(self)

    def _reset_fitted(self):
        """Drop everything a previous fit left behind. Called by `__init__`
        and at the top of every `fit`, so a failed refit does not leave the
        estimator claiming to hold the older model."""
        self._model = None
        self._importance_cache = None
        self._multiclass = False
        # Category state is private because prediction needs it whether or
        # not the caller ever reads `categorical_feature_`; clearing it here
        # keeps a refit from encoding new data through the old tables.
        self._cat_indices = ()
        self._cat_encoders = {}
        # What validation, if any, shaped the last fit. `_record_fit` reads
        # them, so a refit without an eval_set must not inherit them.
        self._metric_best_iteration = None
        self._metric_n_iter = None
        for name in self._FITTED_ATTRS:
            self.__dict__.pop(name, None)

    def __sklearn_is_fitted__(self):
        """scikit-learn's `check_is_fitted` hook: the model handle is the
        one true signal, not the presence of trailing-underscore
        attributes."""
        return getattr(self, "_model", None) is not None

    def _require_fitted(self):
        if getattr(self, "_model", None) is None:
            raise NotFittedError(
                f"this {type(self).__name__} is not fitted yet; call fit() "
                "with training data before using this estimator"
            )

    def _check_n_features(self, n_features):
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {type(self).__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

    def _check_feature_names(self, names, validate_features=False):
        """Compare the column names of an incoming matrix against the ones
        recorded at fit time, warning when only one side has them and
        raising when both do and they disagree, as scikit-learn does.

        `validate_features` is LightGBM's `predict` flag: it turns the
        one-sided cases from warnings into errors, so that asking for
        validation and getting it silently is not possible. A name mismatch
        raises either way, because that is a mismatch scikit-learn already
        refuses to predict through."""
        fitted = getattr(self, "feature_names_in_", None)
        if fitted is None and names is None:
            if validate_features:
                raise ValueError(
                    "validate_features=True needs feature names on both "
                    f"sides, but {type(self).__name__} was fitted without "
                    "them and X does not carry them"
                )
            return
        if fitted is None:
            message = (
                f"X has feature names, but {type(self).__name__} was fitted "
                "without feature names"
            )
            if validate_features:
                raise ValueError(message)
            _warnings.warn(message, UserWarning, stacklevel=3)
            return
        if names is None:
            message = (
                "X does not have valid feature names, but "
                f"{type(self).__name__} was fitted with feature names"
            )
            if validate_features:
                raise ValueError(message)
            _warnings.warn(message, UserWarning, stacklevel=3)
            return
        if list(names) != list(fitted):
            raise ValueError(
                "the feature names should match those passed during fit; "
                f"fitted on {list(fitted)}, got {list(names)}"
            )

    def _record_fit(self, n_features, names, device):
        """Record the fitted-state attributes every estimator shares."""
        self.n_features_in_ = n_features
        if names is not None:
            self.feature_names_in_ = _arrays.name_array(names)
        self.device_ = device
        # With validation, the best iteration is the one the primary metric
        # peaked at; the ensemble is rolled back to it whenever early
        # stopping is on, so the two agree unless it was left off.
        best = getattr(self, "_metric_best_iteration", None)
        self.best_iteration_ = (
            self._num_iterations() if best is None else best
        )
        rounds = getattr(self, "_metric_n_iter", None)
        self.n_iter_ = self._num_iterations() if rounds is None else rounds
        self._importance_cache = {
            kind: self._raw_importance(kind) for kind in _IMPORTANCE_TYPES
        }

    def _check_predict_X(self, X, validate_features=False):
        """Validate a matrix for prediction against the fitted model.

        Categorical columns are encoded through the tables `fit` recorded,
        so the code a label maps to is the one it trained as, whatever the
        incoming frame calls it."""
        self._require_fitted()
        encoders = self._matrix_encoders(X)
        Xb, n_rows, n_features, names = _arrays.check_X(X, encoders=encoders)
        self._check_n_features(n_features)
        self._check_feature_names(names, validate_features)
        self._check_category_codes(
            Xb, n_rows, getattr(self, "_cat_indices", ()), names
        )
        return Xb, n_rows

    # -- prediction options ----------------------------------------------

    def _check_predict_flags(self, raw_score, pred_leaf, pred_contrib=False):
        """Reject prediction flags that ask for different outputs.

        `raw_score` asks for scores on the link scale, `pred_leaf` for leaf
        ordinals, and `pred_contrib` for per-feature contributions; they have
        different dtypes and different shapes, so there is no result that
        satisfies more than one. LightGBM silently picks a winner; mojotrees
        raises, because the quiet winner is not discoverable from the output.

        `raw_score` with `pred_contrib` is refused for a further reason:
        contributions always explain the raw score, whatever the objective's
        link, so `raw_score=True` would read as a choice that does not exist.
        """
        asked = []
        if pred_leaf:
            asked.append("pred_leaf=True")
        if pred_contrib:
            asked.append("pred_contrib=True")
        if raw_score:
            asked.append("raw_score=True")
        if len(asked) > 1:
            raise ValueError(
                f"{' and '.join(asked)} ask for different outputs (scores, "
                "leaf ordinals, and feature contributions have different "
                "shapes); pass at most one. Contributions always explain the "
                "raw score, so raw_score=True is redundant with them."
            )

    # -- where one prediction call runs ------------------------------------
    #
    # The device is *requested* here and *decided* in Mojo. The prediction
    # entry points take the requested name in their params dict, ask
    # gpu_predict.mojo whether the GPU path covers a request of that shape
    # (only for an explicit "gpu", so the refusal is what an explicit
    # request gets), resolve through the same device.mojo policy a fit
    # resolves through, and return the backend that ran. Nothing in this
    # file decides, thresholds, or infers; adding such a thing here would
    # put a second policy beside the native one.

    def _device_request(self, device):
        """The device one prediction call asks for: "cpu", "gpu", or
        "auto".

        `device=None` means "cpu", which is the established path: the same
        backend predictions have always run on, reached the same way.
        """
        name = _device_name(device)
        return "cpu" if name is None else name

    def _batch_params(self, device, start, stop, raw_score=False):
        """The params dict the dense batch prediction entry points read:
        the requested device, the resolved half-open iteration pair, and
        the raw-score flag."""
        return {
            "device": self._device_request(device),
            "start": int(start),
            "stop": int(stop),
            "raw_score": int(bool(raw_score)),
        }

    def _refuse_device(self, device, what):
        """Refuse an explicit accelerator request for a prediction mode
        that has no device path.

        `"auto"` is not refused: it resolves to the CPU here, which is
        where it would resolve anyway. Only an explicit `"gpu"` is, because
        running it on the CPU and returning as though nothing happened is
        the one outcome an explicit request must not produce.
        """
        if self._device_request(device) == "gpu":
            raise RuntimeError(
                f"{what} is computed on the CPU; there is no accelerator "
                "kernel for it. Pass device='cpu' or device='auto' (or "
                "leave device unset), or drop the flag to predict scores "
                "on an accelerator."
            )

    def _predict_batch(
        self, entry, legacy, Xb, n_rows, params, out, pass_raw=True
    ):
        """One dense batch prediction into `out`, through `entry`.

        `entry` is a device-aware batch entry point (`predict_batch`,
        `predict_proba_batch`, `predict_leaf_batch`, ...) and `legacy` the
        one that predates it. When the build has `entry`, every call goes
        through it, `device="cpu"` included, so that there is one
        prediction path rather than a device path beside an older one, and
        the backend that ran comes back from the call. When it does not,
        `"cpu"` uses `legacy` and anything else raises: predicting on the
        CPU while reporting an accelerator is what must not happen.

        Returns the name of the backend that ran.
        """
        hook = getattr(_mojotrees, entry, None)
        if hook is not None:
            ran = hook(
                self._model,
                _addr(Xb),
                n_rows,
                self.n_features_in_,
                params,
                _addr(out),
            )
            return params["device"] if ran is None else str(ran)
        device = params["device"]
        if device != "cpu":
            raise RuntimeError(_NO_DEVICE_PREDICT % (entry, device))
        args = (
            self._model,
            _addr(Xb),
            n_rows,
            self.n_features_in_,
            params["start"],
            params["stop"],
        )
        if pass_raw:
            args += (params["raw_score"],)
        getattr(_mojotrees, legacy)(*args, _addr(out))
        return "cpu"

    def _sparse_predict_params(self, device, params):
        """A sparse prediction params dict with the requested device in it.

        Sparse prediction has no accelerator kernel, and the refusal for an
        explicit `"gpu"` is native (`_refuse_gpu_sparse` in
        bindings/_mojotrees.mojo), so that it carries the same message the
        dense path gives. A build old enough to read no device key at all
        would drop the request instead of refusing it, so on that build the
        refusal is made here.
        """
        requested = self._device_request(device)
        if requested == "cpu":
            return params
        if getattr(_mojotrees, "predict_batch", None) is None:
            raise RuntimeError(
                _NO_DEVICE_PREDICT % ("predict_batch", requested)
            )
        params["device"] = requested
        return params

    def _iteration_slice(self, start_iteration, num_iteration):
        """Resolve LightGBM's `(start_iteration, num_iteration)` pair into
        the half-open `[start, stop)` slice of boosting iterations to
        predict with, clamped to the fitted ensemble.

        The rules are LightGBM's, from `GBDT::InitPredict`: a negative start
        clamps to 0 and a start past the end clamps to an empty range at the
        end, `num_iteration=None` or any value <= 0 means every iteration
        from the start on, and a positive one is capped at what remains.
        Clamping rather than raising is what LightGBM callers depend on when
        they slice a shorter ensemble than they expected.

        `num_iteration=None` therefore predicts with `best_iteration_`
        iterations, which is LightGBM's documented default. mojotrees gets
        there structurally: early stopping truncates the ensemble at its best
        iteration, so the trees the model still holds *are* the best
        iteration and there is no later tree to exclude."""
        total = self._num_iterations()
        start = _as_iteration(start_iteration, "start_iteration")
        start = min(max(start, 0), total)
        if num_iteration is None:
            return start, total
        num = _as_iteration(num_iteration, "num_iteration")
        if num <= 0:
            return start, total
        return start, min(start + num, total)

    def _predict_contrib(self, Xb, n_rows, start, stop):
        """Exact TreeSHAP contributions from the iterations in
        `[start, stop)`.

        The shape follows LightGBM. A single-output model gives
        `(n_samples, n_features + 1)`, the last column being the expected
        value, so each row sums to that row's raw score. A multiclass model
        gives `(n_samples, n_classes * (n_features + 1))` in class-major
        blocks: columns `k * (n_features + 1)` through
        `k * (n_features + 1) + n_features` are class k's contributions and
        its expected value, and that block sums to class k's raw score.

        An empty iteration range keeps the shape: the feature columns are
        zero and the expected-value column carries the base score only when
        the range includes iteration 0, so the sum property still holds."""
        stride = self.n_features_in_ + 1
        per_row = self.n_classes_ * stride if self._multiclass else stride
        if n_rows == 0:
            if _np is not None:
                return _np.empty((0, per_row), dtype=_np.float64)
            return []
        out = _out_buffer(n_rows * per_row)
        query = (
            _mojotrees.predict_contrib_multiclass
            if self._multiclass
            else _mojotrees.predict_contrib
        )
        query(
            self._model,
            _addr(Xb),
            n_rows,
            self.n_features_in_,
            start,
            stop,
            _addr(out),
        )
        if _np is not None:
            return out.reshape(n_rows, per_row)
        return [
            [out[r * per_row + c] for c in range(per_row)]
            for r in range(n_rows)
        ]

    def _predict_leaf(self, Xb, n_rows, start, stop, device=None):
        """Leaf ordinals for every tree in `[start, stop)`.

        The shape is `(n_samples, (stop - start) * trees_per_iteration)`,
        where `trees_per_iteration` is the class count for a multiclass
        model and 1 otherwise. The extension writes float64 (the only
        element type that crosses the boundary) and the ordinals are small
        integers, so casting back is exact.

        The ordinal numbering is the model's and not the backend's: the
        device walk reports the leaf's rank among its tree's leaves in node
        order, which is what the host table indexes, so the two agree."""
        per_iteration = self.n_classes_ if self._multiclass else 1
        n_cols = (stop - start) * per_iteration
        if n_cols == 0 or n_rows == 0:
            # An empty range selects no trees, so there is nothing to ask the
            # extension for; the result keeps its shape and loses a column
            # per unselected iteration.
            if _np is not None:
                return _np.empty((n_rows, n_cols), dtype=_np.int32)
            return [[] for _ in range(n_rows)]
        out = _out_buffer(n_rows * n_cols)
        entry, legacy = (
            ("predict_leaf_multiclass_batch", "predict_leaf_multiclass")
            if self._multiclass
            else ("predict_leaf_batch", "predict_leaf")
        )
        self._predict_batch(
            entry,
            legacy,
            Xb,
            n_rows,
            self._batch_params(device, start, stop),
            out,
            pass_raw=False,
        )
        if _np is not None:
            return out.reshape(n_rows, n_cols).astype(_np.int32)
        return [
            [int(out[r * n_cols + c]) for c in range(n_cols)]
            for r in range(n_rows)
        ]

    def _check_predict_X_sparse(self, X, validate_features=False):
        """`_check_predict_X` for SciPy sparse input, as CSR."""
        self._require_fitted()
        buffers, n_rows, n_features, names = _arrays.check_X_sparse(X, "csr")
        self._check_n_features(n_features)
        self._check_feature_names(names, validate_features)
        return buffers, n_rows

    def _sparse_scores(
        self, X, raw_score, start_iteration, num_iteration, pred_leaf,
        pred_contrib, validate_features, device=None,
    ):
        """One score per row for sparse input, response scale or raw.

        The prediction options that slice or decompose the ensemble read a
        dense binned matrix, so they are refused here instead of quietly
        densifying. Plain prediction is the sparse walk: one binary search
        per node over that row's own stored entries.

        `device` travels in the params dict so that the refusal for an
        explicit `"gpu"` is the native one: there is no sparse accelerator
        prediction kernel (training has one, prediction does not).
        """
        if pred_leaf or pred_contrib:
            raise ValueError(
                "pred_leaf and pred_contrib do not take sparse input yet; "
                "densify with .toarray()"
            )
        if start_iteration != 0 or num_iteration is not None:
            raise ValueError(
                "iteration slicing does not take sparse input yet; densify "
                "with .toarray()"
            )
        buffers, n_rows = self._check_predict_X_sparse(X, validate_features)
        out = _out_buffer(n_rows)
        query = (
            _mojotrees.predict_raw_csr
            if raw_score
            else _mojotrees.predict_csr
        )
        query(
            self._model,
            self._sparse_predict_params(device, buffers.params()),
            _addr(out),
        )
        return out, n_rows

    def _sparse_fit_params(
        self, X, sample_weight, objective_code=None, n_outputs=1
    ):
        """Everything a sparse fit needs: the CSC buffers, the shape, the
        column names, the params dict with the buffers folded in, and a
        tuple to keep every referenced buffer alive across the call.

        Returns the resolved device as well. Sparse training runs on the
        CPU unless `device="gpu"` is asked for explicitly: the native sparse
        GPU trainer (`train_gpu_sparse`, reached through `fit_csc`) grows on
        the compressed matrix, never densifying, and `device="auto"`
        resolves to the CPU because that path's crossover is unmeasured.
        Both answers come from the native policy, asked with `sparse=True`;
        `objective_code` and `n_outputs` are passed so a refusal it writes
        names every reason. The device name is folded into `params`, and
        the trainer resolves it again natively, so no decision is made
        here.
        """
        buffers, n_rows, n_features, names = _arrays.check_X_sparse(X, "csc")
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        mono_buf, mono_addr = self._monotone_buffer(n_features)
        contri_buf, contri_addr = self._feature_contri_buffer(n_features)
        cat_buf = self._sparse_categorical_buffer(X, names, n_features)
        device = self._resolve_device(
            n_rows,
            n_features,
            n_outputs,
            objective_code=objective_code,
            sparse=True,
            categorical=cat_buf is not None,
        )
        params = self._params(
            w_addr, device, ic_flat, ic_offsets, mono_addr, cat_buf, contri_addr
        )
        _preflight.native_preflight(params, n_features, device)
        params.update(buffers.params())
        keep = (
            buffers,
            wb,
            ic_flat,
            ic_offsets,
            mono_buf,
            cat_buf,
            contri_buf,
        )
        return buffers, n_rows, n_features, names, params, keep, device

    def _sparse_categorical_buffer(self, X, names, n_features):
        """Categorical columns for a sparse fit.

        A SciPy matrix carries no dtypes, so only explicitly named indices
        can be categorical here; a frame's `category` dtypes have no sparse
        equivalent to read.
        """
        indices, encoders = self._resolve_categorical(names, {})
        self._cat_indices = list(indices)
        self._cat_encoders = encoders
        self.categorical_feature_ = list(indices)
        return self._categorical_buffer(indices, n_features)

    @staticmethod
    def _reject_sparse_eval_set(eval_set):
        if eval_set is not None:
            raise ValueError(
                "validation sets are not wired through the sparse path yet; "
                "the Mojo API has train_sparse_with_valid. Fit without "
                "eval_set, or densify with .toarray()."
            )

    # -- fitted attributes -----------------------------------------------

    def _num_iterations(self):
        if self._multiclass:
            return int(_mojotrees.num_iterations_multiclass(self._model))
        return int(_mojotrees.num_iterations(self._model))

    def _raw_importance(self, importance_type):
        out = _out_buffer(self.n_features_in_)
        query = (
            _mojotrees.feature_importance_multiclass
            if self._multiclass
            else _mojotrees.feature_importance
        )
        query(
            self._model,
            self.n_features_in_,
            _IMPORTANCE_TYPES[importance_type],
            _addr(out),
        )
        return _finish(out)

    @property
    def feature_importances_(self):
        """Per-feature importance of the kind `importance_type` names.

        Split counts and total gains are both computed when the model is
        fitted, so changing `importance_type` afterwards costs nothing.
        Model format v4 preserves both values across save/load and pickle;
        older model formats return zero gains because they did not store
        them.
        """
        self._require_fitted()
        importance_type = self.importance_type
        if importance_type not in _IMPORTANCE_TYPES:
            raise ValueError(
                f"unknown importance_type {importance_type!r}; expected one "
                "of " + ", ".join(sorted(_IMPORTANCE_TYPES))
            )
        cache = getattr(self, "_importance_cache", None)
        if cache is not None:
            values = cache[importance_type]
            return values.copy() if _np is not None else list(values)
        return self._raw_importance(importance_type)

    # -- pickling --------------------------------------------------------

    def _model_bytes(self):
        """The fitted model in the on-disk serialization format."""
        with _tempfile.TemporaryDirectory() as d:
            path = _os.path.join(d, "model.mbst")
            self.save(path)
            with open(path, "rb") as fh:
                return fh.read()

    def _model_from_bytes(self, blob):
        with _tempfile.TemporaryDirectory() as d:
            path = _os.path.join(d, "model.mbst")
            with open(path, "wb") as fh:
                fh.write(blob)
            if self._multiclass:
                return _mojotrees.load_multiclass(path)
            return _mojotrees.load(path)

    def __getstate__(self):
        """Pickle support. The trained model is an opaque handle owned by
        the extension module, so it travels as the bytes of the same
        versioned text format `save()` writes; everything else is ordinary
        Python state and pickles as it is."""
        state = self.__dict__.copy()
        # The handle lives inside the Booster on `_booster`, and neither a
        # Mojo handle nor the Booster's link to a training set pickles; the
        # model itself travels as text and `_model` rebuilds the Booster.
        booster = state.pop("_booster", None)
        state["_model_blob"] = (
            None if booster is None else self._model_bytes()
        )
        return state

    def __setstate__(self, state):
        state = dict(state)
        blob = state.pop("_model_blob", None)
        self.__dict__.update(state)
        self._model = None if blob is None else self._model_from_bytes(blob)


class MojoTreesRegressor(_Base):
    """Objective names follow LightGBM: "regression" (squared error;
    aliases "regression_l2", "l2", "mse", "mean_squared_error"), "huber",
    "quantile", "mae" (aliases "regression_l1", "l1",
    "mean_absolute_error"), "poisson", "gamma", "tweedie", "mape" (alias
    "mean_absolute_percentage_error"), "fair", and "cross_entropy" (alias
    "xentropy").

    Each objective's scalar parameter keeps LightGBM's name:

    - `alpha` is the quantile level for "quantile" and the transition point
      for "huber" (default 0.9).
    - `fair_c` is the fair loss's `c` (default 1.0).
    - `tweedie_variance_power` is tweedie's rho, in (1, 2) (default 1.5).

    An objective reads only its own, and setting one that belongs to a
    different objective is an error rather than a value that quietly does
    nothing.

    Link and label range, which differ by objective and are worth knowing
    before reading `predict`:

    - "poisson", "gamma", "tweedie" predict expected values through an
      exponential link and need nonnegative labels ("gamma" strictly
      positive).
    - "cross_entropy" predicts a probability through the logistic link and
      takes labels anywhere in [0, 1]. It is a regressor objective because
      its labels are soft targets rather than classes; for {0, 1} labels use
      MojoTreesClassifier.
    - "mape" and "fair" are identity-link regression losses. MAPE weights
      each row by `1 / max(1, |y|)`, so it measures relative error.

    LightGBM objectives mojotrees does not implement, deliberately:
    "cross_entropy_lambda" (a different link, not an alias of
    "cross_entropy"), "multiclassova" (one-vs-rest needs a separate
    trainer), and "rank_xendcg" (use "lambdarank" through
    MojoTreesRanker). Each is listed in docs/LIGHTGBM_PARITY.md.

    `objective` may instead be a callable `f(raw, y) -> (grad, hess)`, called
    once per boosting round with the current raw predictions and the labels.
    See `_fit_custom` for the callback contract and
    src/mojotrees/objective.mojo for how it differs from LightGBM's; the
    short version is that the trainer applies `sample_weight` for you, that
    it validates the returned arrays, and that `predict` then returns raw
    scores because the inverse link is yours to apply. `base_score` is the
    starting raw score for a custom objective (a number, or "mean" for the
    weighted label mean); built-in objectives ignore it and derive their
    own."""

    # scikit-learn before 1.6 dispatches on this; 1.6 and later read
    # __sklearn_tags__ below. Both are cheap, so both are here.
    _estimator_type = "regressor"

    # The regression spellings the compiled registry resolves, alias ->
    # code. `_objective_code` resolves through the registry
    # (`objective_code_of_name`); this literal is the frozen contract
    # tools/api_snapshot.py reads and the check that the registry still
    # resolves every one of them to the same code
    # (python/tests/test_registry_readers.py). A spelling the registry
    # gains does not become a regressor objective until it is added here.
    #: Every spelling of every regression objective, from all four
    #: vendors: LightGBM's `objective`, XGBoost's `objective`, CatBoost's
    #: `loss_function`, and scikit-learn's `HistGradientBoostingRegressor`
    #: `loss`. Names are matched after a `.lower()`, so CatBoost's `RMSE`
    #: and `MAE` arrive as written.
    #:
    #: The same table exists in `src/mojotrees/params.mojo`
    #: (`_objective_from_lower`) for the parameter-string surface. Two
    #: tables, because the two surfaces resolve through different code and
    #: neither can import the other; they are checked against each other by
    #: the vendor-dialect tests.
    _OBJECTIVES = {
        # LightGBM.
        "regression": _SQUARED_ERROR,
        "regression_l2": _SQUARED_ERROR,
        "l2": _SQUARED_ERROR,
        "mean_squared_error": _SQUARED_ERROR,
        "mse": _SQUARED_ERROR,
        # XGBoost, its pre-1.0 spelling, CatBoost, scikit-learn.
        "reg:squarederror": _SQUARED_ERROR,
        "reg:linear": _SQUARED_ERROR,
        "rmse": _SQUARED_ERROR,
        "squared_error": _SQUARED_ERROR,
        "huber": _HUBER,
        "quantile": _QUANTILE,
        "reg:quantileerror": _QUANTILE,
        "quantile_loss": _QUANTILE,
        "mae": _L1,
        "regression_l1": _L1,
        "l1": _L1,
        "mean_absolute_error": _L1,
        "reg:absoluteerror": _L1,
        "absolute_error": _L1,
        "poisson": _POISSON,
        "count:poisson": _POISSON,
        "gamma": _GAMMA,
        "reg:gamma": _GAMMA,
        "tweedie": _TWEEDIE,
        "reg:tweedie": _TWEEDIE,
        "mape": _MAPE,
        "mean_absolute_percentage_error": _MAPE,
        "reg:absolutepercentageerror": _MAPE,
        "fair": _FAIR,
        "cross_entropy": _CROSS_ENTROPY,
        "xentropy": _CROSS_ENTROPY,
        "crossentropy": _CROSS_ENTROPY,
    }

    #: objective code -> (parameter name, default). The trainer takes one
    #: scalar per objective (see src/mojotrees/boosting.mojo); these are the
    #: LightGBM names for it, and the objectives not listed here take none.
    _OBJECTIVE_PARAM = {
        _HUBER: ("alpha", 0.9),
        _QUANTILE: ("alpha", 0.9),
        _FAIR: ("fair_c", 1.0),
        _TWEEDIE: ("tweedie_variance_power", 1.5),
    }

    def __init__(
        self,
        objective="regression",
        loss_function=None,
        loss=None,
        alpha=0.9,
        fair_c=1.0,
        tweedie_variance_power=1.5,
        base_score=0.0,
        num_targets=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.objective = objective
        self.loss_function = loss_function
        self.loss = loss
        self.alpha = alpha
        self.fair_c = fair_c
        self.tweedie_variance_power = tweedie_variance_power
        self.base_score = base_score
        # CatBoost's MultiRMSE. `None` means "read it off y", which is what
        # a 2-D y already says; naming it is a CHECK on what the caller
        # believes and not a reshape instruction. See `_fit_multi_target`.
        self.num_targets = num_targets

    def _resolve_objective(self):
        """The objective as spelled, from `objective` or from either vendor
        alias. CatBoost calls the parameter `loss_function` and
        scikit-learn calls it `loss`; `objective` is the canonical name and
        the one three of the four vendors use.

        The third argument is `"regression"` and not `None` because that is
        this estimator's default for `objective`, and an alias only wins
        over a primary still sitting at its default.
        """
        objective = self._resolve_alias(
            "objective", "loss_function", "regression"
        )
        return self._resolve_alias(
            "objective", "loss", "regression", objective
        )

    def _objective_code(self):
        objective = self._resolve_objective()
        if callable(objective):
            return _CUSTOM
        code = None
        key = objective.lower() if isinstance(objective, str) else objective
        if key in self._OBJECTIVES:
            # The registry is the resolver for the spellings it knows; the
            # literal says which of its spellings are regression objectives
            # and carries the vendor names the registry has never held.
            code = _objective_code_of_name(key)
            if code is None:
                code = self._OBJECTIVES[key]
        if code is None:
            if key == "reg:pseudohubererror":
                raise ValueError(
                    "objective='reg:pseudohubererror' is XGBoost's smooth "
                    "pseudo-Huber, which is a different curve from "
                    "'huber' (a quadratic spliced to a line at alpha), not "
                    "a spelling of it; mojotrees implements 'huber'"
                )
            raise ValueError(
                f"unknown objective {objective!r}; expected one of "
                + ", ".join(sorted(self._OBJECTIVES))
                + _unimplemented_objective_note(objective)
            )
        self._objective_param(code)
        return code

    def _objective_param(self, code=None):
        """The objective's scalar parameter, validated.

        One trainer slot holds it whatever it is called. `fair_c` and
        `tweedie_variance_power` name exactly one objective each, so setting
        one away from its default for a different objective is rejected: it
        states an intention the model cannot carry out. `alpha` is lenient,
        as in LightGBM, because it is the shared default name that several
        objectives ignore and passing it alongside any objective is
        long-standing usage.
        """
        if code is None:
            code = self._objective_code()
        name, default = self._OBJECTIVE_PARAM.get(code, (None, 0.9))
        for other_name, other_default in (
            ("fair_c", 1.0),
            ("tweedie_variance_power", 1.5),
        ):
            if other_name == name:
                continue
            value = float(getattr(self, other_name, other_default))
            if value != other_default:
                raise ValueError(
                    f"{other_name}={value!r} does not apply to objective "
                    f"{self._resolve_objective()!r}"
                    + (
                        f"; it takes {name}"
                        if name is not None
                        else "; that objective takes no scalar parameter"
                    )
                )
        if name is None:
            return default
        value = float(getattr(self, name, default))
        # The trainer's range check and the trainer's sentence
        # (objective_registry.check_objective_param); no copy of the
        # bounds lives here.
        _check_objective_param(code, value)
        return value

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_metric=None,
        early_stopping_rounds=0,
        min_delta=0.0,
        primary_metric=0,
        eval_sample_weight=None,
        eval_X=None,
        eval_y=None,
        callbacks=None,
    ):
        """Fit on `X` (n_samples, n_features) and a numeric target `y`.

        `X` may contain NaN, which is the missing-value marker, but not
        infinities; `y` and `sample_weight` must be finite throughout.

        `eval_set` is a list of `(X, y)` validation pairs (or one bare pair,
        or `eval_X=`/`eval_y=`), named by `eval_names` or `valid_0`,
        `valid_1`, ... by default, weighted by `eval_sample_weight`, and
        scored every round by `eval_metric`: LightGBM metric names,
        callables, or both, defaulting to the objective's own loss (see
        `_fit_with_metrics`). With `early_stopping_rounds` above 0, training
        stops once a watched metric goes that many rounds without improving
        by more than `min_delta`, and the ensemble is truncated to the best
        round of `primary_metric` (an index or a name) on the first
        validation set.

        A 2-D `y` of shape `(n_samples, n_targets)`, or `num_targets` above
        1, takes the MultiRMSE path instead; see `_fit_multi_target` for
        what that honors and what it refuses.

        Returns self.
        """
        if _multi_target.is_multi_target(y, self.num_targets):
            return self._fit_multi_target(
                X, y, sample_weight, eval_set, eval_X, eval_y, callbacks
            )
        # A refit with a 1-D y after a multi-target one must not leave the
        # multi-output handle behind: `predict` branches on its presence and
        # would score the new fit through the old model.
        self.__dict__.pop("_multi_model", None)
        objective = self._objective_code()
        eval_set = _eval_pairs(eval_set, eval_X, eval_y)
        _check_eval_arguments(
            eval_set,
            eval_metric,
            eval_sample_weight,
            early_stopping_rounds,
            callbacks,
        )
        self._reset_fitted()
        if _arrays.is_sparse(X):
            if objective == _CUSTOM:
                raise TypeError(
                    "a Python objective callback does not take sparse input "
                    "yet; densify with .toarray() or use a built-in objective"
                )
            self._refuse_alternate_boosting("with sparse input")
            self._reject_sparse_eval_set(eval_set)
            buffers, n_rows, n_features, names, params, keep, device = (
                self._sparse_fit_params(X, sample_weight, objective)
            )
            yb = _arrays.check_target(y, n_rows)
            self._model = _mojotrees.fit_csc(_addr(yb), objective, params)
            self._record_fit(n_features, names, device)
            del keep
            return self
        Xb, n_rows, n_features, names, cat_buf = self._fit_X(X)
        self._check_fit_structure(X, y, n_rows, n_features, sample_weight)
        yb = _arrays.check_target(y, n_rows)
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        mono_buf, mono_addr = self._monotone_buffer(n_features)
        contri_buf, contri_addr = self._feature_contri_buffer(n_features)
        device = self._resolve_device(
            n_rows,
            n_features,
            1,
            objective_code=objective,
            categorical=cat_buf is not None,
            has_eval_set=eval_set is not None,
        )
        params = self._params(
            w_addr,
            device,
            ic_flat,
            ic_offsets,
            mono_addr,
            cat_buf,
            contri_addr,
        )
        # The extra tree parameters and the bundling knobs, checked
        # natively before any data is copied (the same checkers
        # `_parse_params` runs again at dispatch).
        _preflight.native_preflight(params, n_features, device)
        if eval_set is not None:
            if objective == _CUSTOM:
                raise ValueError(
                    "a Python objective callback and custom validation "
                    "metrics cannot be combined yet; the Mojo API pairs them "
                    "with train_custom_with_metrics"
                )
            self._refuse_alternate_boosting("with eval_set")
            self._fit_with_metrics(
                Xb,
                yb,
                n_rows,
                n_features,
                params,
                device,
                objective,
                eval_set,
                eval_names,
                eval_metric,
                early_stopping_rounds,
                min_delta,
                primary_metric,
                eval_sample_weight,
                callbacks=callbacks,
            )
        elif objective == _CUSTOM:
            self._refuse_alternate_boosting("with a callable objective")
            self._fit_custom(Xb, yb, n_rows, n_features, params, device)
        elif self._distributed_world() > 1:
            self._refuse_alternate_boosting("with tree_learner")
            if device != "cpu":
                raise ValueError(
                    "tree_learner other than 'serial' trains on the CPU;"
                    " set device='cpu'"
                )
            self._model = _mojotrees.distributed_train_local(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                objective,
                dict(
                    params,
                    num_machines=int(self.num_machines),
                    tree_learner=str(self.tree_learner),
                    top_k=int(self.top_k),
                ),
            )
        else:
            self._model = _mojotrees.fit(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                objective,
                params,
            )
        self._record_fit(n_features, names, device)
        return self

    def _fit_multi_target(
        self, X, y, sample_weight, eval_set, eval_X, eval_y, callbacks
    ):
        """CatBoost's `MultiRMSE`: a 2-D target, one tree per target per
        round, reached through `multi_target.fit_multi_rmse`.

        **This is the connecting edge `target_matrix.TargetMatrix` was
        written for and never got.** Its own docstring names the finding:
        every training entry point in this repository takes a single target
        column, so the wider label contract existed and nothing reached it,
        and `multi_target` and `survival` were both blocked on that one
        missing edge rather than on their own arithmetic. This closes it for
        `MultiRMSE`. `Cox` and `SurvivalAft` still have no entry point of
        their own -- the wire below carries `n_targets` columns and would
        carry theirs unchanged, but `survival.mojo` has no `fit_*` and
        nothing selects its objective codes.

        **What it is not.** One tree per target per round is not CatBoost's
        `MultiRMSE`, which grows one tree per round with a vector leaf value.
        Because the derivative has no cross-target term, what this fits is
        bit-identical to `n_targets` independent squared-error fits, and the
        shared structure is the entire modeling content of the objective. It
        is a real multi-output regression API under an honest name; see
        catalog A28 and the head of `src/mojotrees/multi_target.mojo`.

        **Not supported on a multi-target fit, and loudly rather than
        quietly:** `save`, `booster_`, `feature_importances_`, pickling, and
        `__sklearn_is_fitted__`. All of them read `self._model`, which is a
        single-output `Model` handle and stays `None` here, so each raises
        `NotFittedError` rather than answering about a model it is not
        looking at. `serialize.mojo` has no `MultiTargetBooster` section and
        widening the model format is not this edge's decision; see catalog
        A29.
        """
        if eval_set is not None or eval_X is not None or eval_y is not None:
            raise ValueError(
                "a multi-target fit takes no eval_set: the metric-path "
                "trainer is custom_metric.fit_with_metrics, which is "
                "single-output, and multi_target.train_multi_rmse has no "
                "validation loop"
            )
        if callbacks:
            raise ValueError(
                "a multi-target fit takes no callbacks: they are driven by "
                "the metric-path round loop, which is single-output"
            )
        if self.monotone_constraints is not None:
            # `catboost_reach_bindings` builds its `TreeParams` without the
            # monotone bundle, so a constraint set here would be dropped and
            # the returned model would report none. `tools/check_parity.py`
            # caught that as a silent drop, which is the defect class this
            # whole round exists to end, and the honest close is a refusal
            # rather than a forwarded constraint no trainer applies:
            # `multi_target.train_multi_rmse` has no monotone pass, so
            # passing it through would move the silence one layer down.
            raise ValueError(
                "a multi-target fit takes no monotone_constraints: "
                "multi_target.train_multi_rmse has no monotone pass, so the "
                "constraint would be accepted and never applied. Fit each "
                "target separately if you need it"
            )
        if _arrays.is_sparse(X):
            raise TypeError(
                "a multi-target fit takes a dense X: train_multi_rmse bins "
                "through binning.fit_bins and the sparse binner has no "
                "multi-output trainer behind it"
            )
        _multi_target.check_honored(self, str(self._resolve_objective()))
        self._reset_fitted()
        self.__dict__.pop("_multi_model", None)
        Xb, n_rows, n_features, names, cat_buf = self._fit_X(X)
        if cat_buf is not None:
            raise ValueError(
                "categorical features are not honored by a multi-target "
                "fit: the category split path is in the single-output "
                "grower's search only"
            )
        yb, n_targets = _multi_target.target_matrix(
            y, self.num_targets, n_rows
        )
        params = _multi_target.wire_params(self)
        wb = None
        if sample_weight is not None:
            wb = _arrays.check_sample_weight(sample_weight, n_rows)
            params["weight_addr"] = _arrays.addr(wb)
        device = _device_name(
            self.device if self.device is not None else self.device_type
        )
        if device not in (None, "cpu", "auto"):
            raise ValueError(
                "a multi-target fit runs on the CPU: train_gpu has no "
                "multi-output round loop; set device='cpu'"
            )
        self.__dict__["_multi_model"] = _mojotrees.multi_rmse_fit(
            _addr(Xb),
            n_rows,
            n_features,
            _addr(yb),
            n_targets,
            params,
        )
        del wb
        shape = _mojotrees.multi_rmse_shape(self.__dict__["_multi_model"])
        self.n_features_in_ = n_features
        if names is not None:
            self.feature_names_in_ = _arrays.name_array(names)
        self.device_ = "cpu"
        #: The target count the fit ran with, read back off the model rather
        #: than off the argument.
        self.n_targets_ = shape[0]
        self.n_iter_ = shape[1]
        self.best_iteration_ = shape[1]
        #: `len(trees)`, which is `n_iter_ * n_targets_` and is NOT the
        #: number comparable to CatBoost's `tree_count_`. Both are here
        #: because quoting the wrong one is the easy mistake.
        self.n_trees_ = shape[3]
        return self

    def _predict_multi_target(self, X, validate_features=False):
        """`(n_samples, n_targets)` predictions from a multi-target fit.

        No `raw_score`, no iteration slice, no leaf ordinals and no
        contributions: `MultiRMSE`'s link is the identity so raw and response
        are one number, and the other three have no multi-output native entry
        point. Each is refused in `predict` by name rather than ignored.
        """
        model = self.__dict__["_multi_model"]
        encoders = self._matrix_encoders(X)
        Xb, n_rows, n_features, names = _arrays.check_X(X, encoders=encoders)
        self._check_n_features(n_features)
        self._check_feature_names(names, validate_features)
        out = _out_buffer(n_rows * self.n_targets_)
        _mojotrees.multi_rmse_predict(
            model, _addr(Xb), n_rows, n_features, _addr(out)
        )
        if _arrays.have_numpy():
            import numpy as np  # noqa: PLC0415

            return np.asarray(out, dtype=np.float64).reshape(
                (n_rows, self.n_targets_)
            )
        t = self.n_targets_
        return [
            [out[r * t + k] for k in range(t)] for r in range(n_rows)
        ]

    def _distributed_world(self):
        """The world size a fit trains over: 1 for `tree_learner='serial'`
        (the single-node trainer), `num_machines` otherwise. LightGBM's
        `tree_learner` names are accepted (`serial`, `data`, `feature`,
        `voting` and the `_parallel` spellings); the world is hosted in this
        process over `LocalCollective`, which is the distributed training
        this build runs (see `_mojotrees.distributed_capability()`)."""
        learner = str(self.tree_learner)
        if learner in ("serial", "serial_tree_learner"):
            return 1
        if learner not in (
            "data", "data_parallel", "feature", "feature_parallel",
            "voting", "voting_parallel",
        ):
            raise ValueError(
                f"unknown tree_learner {self.tree_learner!r}; expected "
                "serial, data, feature, or voting"
            )
        machines = int(self.num_machines)
        if machines < 2:
            raise ValueError(
                f"tree_learner={learner!r} needs num_machines >= 2; a "
                "single machine is tree_learner='serial'"
            )
        return machines

    def _custom_base_score(self):
        """Resolve `base_score` for a custom objective into
        `(value, use_label_mean)`. The label mean is computed on the Mojo
        side rather than here: it has to match the built-in objectives'
        base score bit for bit, and only one summation order can."""
        if isinstance(self.base_score, str):
            if self.base_score != "mean":
                raise ValueError(
                    f"unknown base_score {self.base_score!r}; expected a "
                    "number or 'mean'"
                )
            return 0.0, 1
        return float(self.base_score), 0

    def _fit_custom(self, Xb, yb, n_rows, n_features, params, device):
        """Train against a Python objective callback.

        The callback is called once per boosting round with the current raw
        predictions and the labels and returns `(grad, hess)`, each of
        length n_rows. Both arguments are views on live buffers that the
        trainer reuses every round, so read them, do not keep them.

        This is the Python-callback path, whose per-round cost is measured
        in bench/bench_custom_objective.py. For a hot inner loop use the
        native Mojo interface in src/mojotrees/objective.mojo, which
        specializes on the callable and pays nothing per round.
        """
        # Backstop; BLOCK_CUSTOM_OBJECTIVE is what refuses this on a build
        # whose native policy can be asked. See `_gpu_unsupported`.
        self._gpu_unsupported(device, "custom objectives train on the CPU")
        fobj = self._resolve_objective()
        raw = _out_buffer(n_rows)
        grad = _out_buffer(n_rows)
        hess = _out_buffer(n_rows)

        def bridge():
            returned = fobj(raw, yb)
            try:
                g, h = returned
            except (TypeError, ValueError):
                raise ValueError(
                    "custom objective must return (grad, hess)"
                ) from None
            _store_vector(grad, g, n_rows, "grad")
            _store_vector(hess, h, n_rows, "hess")

        base_score, use_label_mean = self._custom_base_score()
        params["raw_addr"] = _addr(raw)
        params["grad_addr"] = _addr(grad)
        params["hess_addr"] = _addr(hess)
        params["base_score"] = base_score
        params["base_score_mean"] = use_label_mean
        self._model = _mojotrees.fit_custom(
            _addr(Xb), n_rows, n_features, _addr(yb), bridge, params
        )

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        device=None,
    ):
        """Predictions for `X`, one per row.

        `raw_score` returns scores on the link scale instead of the response
        scale. The two differ only where the objective has a link: the
        regressor's squared-error, huber, quantile, and L1 objectives predict
        on the raw scale already, and poisson returns `exp(raw)` without it.

        `start_iteration` and `num_iteration` select a slice of the boosting
        iterations, following LightGBM: `num_iteration=None` uses every
        iteration the fitted model kept, which is `best_iteration_`. See
        `_iteration_slice` for the clamping rules and where the base score
        sits.

        `pred_leaf` returns leaf ordinals instead of scores, shape
        `(n_samples, num_iteration)` and integer dtype. Column i is the leaf
        the row reaches in the tree of iteration `start_iteration + i`,
        numbered within that tree; see `MojoTreesRegressor.predict` in the
        module docstring for the numbering's guarantees.

        `validate_features` turns a missing set of feature names on either
        side into an error rather than a warning.

        `device` chooses where this one call runs, independently of the
        `device` the model was fitted with: `None` (the default) predicts
        on the CPU, `"gpu"` raises rather than falling back, and `"auto"`
        resolves through the same native policy a fit resolves through. The
        ensemble is the same object either way. Scores and leaf ordinals
        have a device path; contributions and sparse input do not, and say
        so instead of quietly running on the CPU.

        `raw_score` and `pred_leaf` cannot be combined.

        After a multi-target fit this returns `(n_samples, n_targets)` and
        takes none of the flags below; see `_predict_multi_target`."""
        if self.__dict__.get("_multi_model") is not None:
            for flag, name in (
                (raw_score, "raw_score"),
                (pred_leaf, "pred_leaf"),
                (pred_contrib, "pred_contrib"),
            ):
                if flag:
                    raise ValueError(
                        f"{name} has no multi-target path: MultiRMSE's link "
                        "is the identity and no native entry point takes a "
                        "multi-output model for leaves or contributions"
                    )
            if start_iteration or num_iteration is not None:
                raise ValueError(
                    "an iteration slice has no multi-target path: the "
                    "ensemble is indexed (round, target) and no native "
                    "entry point slices it"
                )
            self._refuse_device(device, "a multi-target model")
            return self._predict_multi_target(X, validate_features)
        self._check_predict_flags(raw_score, pred_leaf, pred_contrib)
        if _arrays.is_sparse(X):
            out, n_rows = self._sparse_scores(
                X, raw_score, start_iteration, num_iteration, pred_leaf,
                pred_contrib, validate_features, device,
            )
            return _finish(out)
        Xb, n_rows = self._check_predict_X(X, validate_features)
        start, stop = self._iteration_slice(start_iteration, num_iteration)
        if pred_leaf:
            return self._predict_leaf(Xb, n_rows, start, stop, device)
        if pred_contrib:
            self._refuse_device(device, "pred_contrib=True")
            return self._predict_contrib(Xb, n_rows, start, stop)
        out = _out_buffer(n_rows)
        self._predict_batch(
            "predict_batch",
            "predict_range",
            Xb,
            n_rows,
            self._batch_params(device, start, stop, raw_score),
            out,
        )
        return _finish(out)

    def score(self, X, y, sample_weight=None):
        """The coefficient of determination R^2 of the prediction, the
        same definition scikit-learn's regressors use: 1 minus the residual
        sum of squares over the total sum of squares, both weighted when
        `sample_weight` is given. Best is 1.0, and it can be negative."""
        pred = self.predict(X)
        n_rows = len(pred)
        target = _arrays.check_target(y, n_rows)
        weights = (
            None
            if sample_weight is None
            else _arrays.check_sample_weight(sample_weight, n_rows)
        )
        if _np is not None:
            w = 1.0 if weights is None else weights
            mean = (
                target.mean()
                if weights is None
                else float((weights * target).sum() / weights.sum())
            )
            residual = float((w * (target - pred) ** 2).sum())
            total = float((w * (target - mean) ** 2).sum())
        else:
            w = [1.0] * n_rows if weights is None else list(weights)
            mean = sum(wi * t for wi, t in zip(w, target)) / sum(w)
            residual = sum(
                wi * (t - p) ** 2 for wi, t, p in zip(w, target, pred)
            )
            total = sum(wi * (t - mean) ** 2 for wi, t in zip(w, target))
        if total == 0.0:
            # A constant target: perfect only if the residuals vanish too.
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / total

    def save(self, path):
        """Write the fitted model to `path` in mojotrees's versioned text
        format. This stores the model, including v4 feature names when they
        exist, but not the estimator's constructor hyperparameters. Pickle
        the estimator when those must travel too."""
        self._require_fitted()
        fitted_names = getattr(self, "feature_names_in_", None)
        names = [] if fitted_names is None else [str(n) for n in fitted_names]
        _mojotrees.save(self._model, str(path), names, len(names))

    @classmethod
    def load(cls, path):
        """Load a saved model into a fresh estimator.

        What comes back predicts exactly as the saved model did, and
        reports `n_features_in_` and `best_iteration_`. What does not come
        back is everything the file never held: the training device (the
        ensemble is the same either way, so there is no `device_`),
        constructor hyperparameters and estimator-only state. Use pickle
        when you want the whole estimator.
        """
        est = cls()
        est._model = _mojotrees.load(str(path))
        est.n_features_in_ = int(_mojotrees.n_features(est._model))
        est.best_iteration_ = est._num_iterations()
        # The file holds exactly the trees that survived training, so the
        # loaded model has as many iterations as it has rounds on disk.
        est.n_iter_ = est.best_iteration_
        names = list(_mojotrees.model_feature_names(str(path)))
        if names:
            est.feature_names_in_ = _arrays.name_array(names)
        est._restore_categorical()
        return est

    def __sklearn_tags__(self):
        return _estimator_tags("regressor")


class MojoTreesClassifier(_Base):
    """Binary (logistic) for 2 classes, softmax for more. Multiclass
    training is CPU-only, so `device="gpu"` raises for 3 or more classes and
    `device="auto"` resolves to the CPU.

    Labels may be of any single comparable type, as in scikit-learn: they
    are sorted, recorded on `classes_`, and encoded to the 0..n_classes-1
    the trainer needs. `predict` returns labels from `classes_`, not the
    internal codes. (mojotrees used to require the codes themselves; this
    is a deliberate widening, and passing 0..n_classes-1 still behaves
    exactly as it did.)

    The classifier takes no `objective`: custom objectives are single-output
    only, so pass yours to `MojoTreesRegressor` and apply your own link to
    its raw predictions. `objective=` is accepted here solely to raise that
    message rather than a bare TypeError.

    `class_weight` weights the classes, scikit-learn's parameter and
    LightGBM's:

    - `None`, the default, weights every row equally.
    - `"balanced"` gives class k the weight `n_samples / (n_classes *
      count_k)`, so every class contributes the same total weight. The
      counts are row counts, not weighted counts, which is scikit-learn's
      rule; a `sample_weight` you pass is then multiplied on top.
    - a dict maps a label from `classes_` to its weight. A class the dict
      does not mention keeps weight 1.0, and a key that is not one of the
      training labels is an error rather than a line with no effect.

    LightGBM's binary-only `scale_pos_weight` is `class_weight={1: w}` here,
    and its `is_unbalance` is `class_weight="balanced"` up to a constant
    factor (`balanced` keeps the mean weight at 1; `is_unbalance` leaves the
    negatives at 1 and lifts the positives). src/mojotrees/class_weight.mojo
    has both under their LightGBM names for the Mojo API.

    Weighting is not calibration. A class-weighted model's probabilities are
    probabilities under the reweighted sample, so `"balanced"` on a rare
    positive class predicts far above the base rate by design."""

    # See the note on MojoTreesRegressor._estimator_type.
    _estimator_type = "classifier"

    #: Objective spellings that name what this classifier already trains,
    #: and the class count each one asserts (`None` for a spelling that says
    #: nothing about the count). All four vendors' words: LightGBM's
    #: `objective`, XGBoost's, CatBoost's `loss_function`, scikit-learn's
    #: `loss`. Naming one of these is a statement of what the classifier
    #: does, so it is honored; naming anything else is a request for a
    #: different model, and that is what the refusal below is for.
    _CLASSIFIER_OBJECTIVES = {
        "binary": 2,
        "binary:logistic": 2,
        "binary:logitraw": 2,
        "logloss": 2,
        "log_loss": 2,
        "cross_entropy": 2,
        "crossentropy": 2,
        "xentropy": 2,
        "multiclass": None,
        "softmax": None,
        "multi:softmax": None,
        "multi:softprob": None,
        "multiclass_loss": None,
        "auto": None,
    }

    def __init__(
        self,
        objective=None,
        loss_function=None,
        loss=None,
        class_weight=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.objective = objective
        self.loss_function = loss_function
        self.loss = loss
        self.class_weight = class_weight

    def _resolve_objective(self):
        """The objective as spelled, from `objective` or from either vendor
        alias (CatBoost's `loss_function`, scikit-learn's `loss`)."""
        objective = self._resolve_alias("objective", "loss_function", None)
        return self._resolve_alias("objective", "loss", None, objective)

    def _check_objective(self, n_classes):
        """Accept an objective that names what this classifier trains for
        `n_classes` classes; refuse anything else.

        The classifier derives its task from `y`, so it needs no objective
        and had none: any value at all raised. That made a LightGBM,
        XGBoost or CatBoost classification script fail on a line that was
        telling the truth -- `objective='binary'` and
        `loss_function='Logloss'` are exactly what a two-class fit here is.
        Such a spelling is now honored, and a spelling that disagrees with
        the encoded class count is refused as the mismatch it is rather
        than as an unsupported parameter.
        """
        objective = self._resolve_objective()
        if objective is None:
            return
        if callable(objective):
            raise ValueError(
                "MojoTreesClassifier takes no callable objective; custom "
                "objectives are single-output only. Use MojoTreesRegressor "
                "with your objective and apply your own link (a sigmoid, "
                "say) to the raw predictions."
            )
        key = str(objective).strip().lower()
        if key not in self._CLASSIFIER_OBJECTIVES:
            raise ValueError(
                f"objective={objective!r} is not what MojoTreesClassifier "
                "trains; it derives the task from y, fitting binary "
                "logistic for two classes and softmax beyond that. Accepted "
                "spellings of those two: "
                + ", ".join(sorted(self._CLASSIFIER_OBJECTIVES))
                + _unimplemented_objective_note(objective)
            )
        wants = self._CLASSIFIER_OBJECTIVES[key]
        if wants is None or int(n_classes) == wants:
            return
        raise ValueError(
            f"objective={objective!r} is a two-class objective but y has "
            f"{int(n_classes)} classes, which this classifier fits with "
            "softmax; leave objective unset, or name a multiclass spelling."
        )

    def _objective_code(self, n_classes=None):
        """The native objective code this classifier trains, or None before
        there is a class count to decide it.

        Two classes is `binary_logistic`, a built-in single-output
        objective the device vocabulary routes. More than two is
        `_MULTICLASS`, which is a code and not an absence: the registry
        resolves the name `"multiclass"` to it, `_normalized_objective` in
        device_policy.mojo preserves it, and the objective gate now
        recognizes it as an objective both backends train (softmax reaches
        the device through `train_multiclass_gpu`).

        It returned None here, meaning "undeclared", which is a different
        claim and a weaker one: an undeclared request skips the objective
        gate entirely, carries `WARN_INCOMPLETE_REQUEST`, and can never
        match a crossover rule, so a multiclass fit could not be routed on
        evidence even once evidence exists. None now means only "this
        estimator has not been fitted and was not told the class count",
        which is the one case where there genuinely is no answer.

        `n_classes` is the count the current fit has just encoded, for the
        callers that ask before `n_classes_` exists; without it the fitted
        attribute answers, and before a fit there is nothing to answer
        with.
        """
        if n_classes is None:
            n_classes = getattr(self, "n_classes_", None)
        if n_classes is None:
            return None
        return _BINARY_LOGISTIC if int(n_classes) == 2 else _MULTICLASS

    def _class_weight_rows(self, codes, n_rows, classes, sample_weight):
        """`sample_weight` with `class_weight` folded in, or it unchanged
        when there is no class weighting.

        `codes` holds the encoded labels (0..n_classes-1) the trainer sees,
        so the lookup is by position and the dict form is translated through
        `classes` first.
        """
        class_weight = self.class_weight
        if class_weight is None:
            return sample_weight
        n_classes = len(classes)
        if isinstance(class_weight, str):
            if class_weight != "balanced":
                raise ValueError(
                    f"unknown class_weight {class_weight!r}; expected "
                    "'balanced', a dict, or None"
                )
            counts = [0] * n_classes
            for r in range(n_rows):
                counts[int(codes[r])] += 1
            per_class = []
            for k in range(n_classes):
                if counts[k] == 0:
                    raise ValueError(
                        f"class {classes[k]!r} has no training rows, so "
                        "class_weight='balanced' has nothing to balance"
                    )
                per_class.append(n_rows / (n_classes * counts[k]))
        elif isinstance(class_weight, dict):
            index = {label: k for k, label in enumerate(classes)}
            per_class = [1.0] * n_classes
            for label, weight in class_weight.items():
                if label not in index:
                    raise ValueError(
                        f"class_weight names {label!r}, which is not one of "
                        f"the training labels {list(classes)!r}"
                    )
                value = float(weight)
                if value < 0.0 or value != value:
                    raise ValueError(
                        "class_weight values must be finite and nonnegative"
                    )
                per_class[index[label]] = value
            if not any(per_class):
                raise ValueError(
                    "class_weight zeroes every class, leaving nothing to fit"
                )
        else:
            raise TypeError(
                "class_weight must be 'balanced', a dict, or None; got "
                f"{type(class_weight).__name__}"
            )
        rows = [per_class[int(codes[r])] for r in range(n_rows)]
        if sample_weight is None:
            return rows
        given = _arrays.check_sample_weight(sample_weight, n_rows)
        return [rows[r] * float(given[r]) for r in range(n_rows)]

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_metric=None,
        early_stopping_rounds=0,
        min_delta=0.0,
        primary_metric=0,
        eval_sample_weight=None,
        eval_X=None,
        eval_y=None,
        callbacks=None,
    ):
        """Fit on `X` (n_samples, n_features) and labels `y`.

        `X` may contain NaN, the missing-value marker, but not infinities.
        `y` needs at least 2 distinct labels, and `sample_weight` must be
        finite and nonnegative.

        The validation arguments work as they do on `MojoTreesRegressor`,
        for two classes and for many. Validation labels go through the same
        encoding the training labels did, so a label that was not in `y` is
        an error rather than a silent miscount, and the default
        `eval_metric` is `binary_logloss` or `multi_logloss` accordingly.
        Metrics receive those encoded labels and raw scores, not
        probabilities: log-odds for two classes, and one row-major block of
        `n_classes_` softmax inputs per row beyond that.

        Returns self.
        """
        eval_set = _eval_pairs(eval_set, eval_X, eval_y)
        _check_eval_arguments(
            eval_set,
            eval_metric,
            eval_sample_weight,
            early_stopping_rounds,
            callbacks,
        )
        self._reset_fitted()
        if _arrays.is_sparse(X):
            self._reject_sparse_eval_set(eval_set)
            return self._fit_sparse(X, y, sample_weight)
        Xb, n_rows, n_features, names, cat_buf = self._fit_X(X)
        self._check_fit_structure(X, y, n_rows, n_features, sample_weight)
        yb, classes = _arrays.encode_labels(y, n_rows)
        n_classes = len(classes)
        # An objective spelling is checked here rather than at the top of
        # `fit`, because whether it names this fit depends on the class
        # count and the class count comes from y.
        self._check_objective(n_classes)
        # class_weight becomes ordinary row weights before anything else
        # sees it, so the trainer has one weighting mechanism, not two.
        sample_weight = self._class_weight_rows(
            yb, n_rows, classes, sample_weight
        )
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        mono_buf, mono_addr = self._monotone_buffer(n_features)
        contri_buf, contri_addr = self._feature_contri_buffer(n_features)
        # Binary is single-output (one tree per round) and softmax is one
        # tree per class per round, so `n_outputs` is what separates them
        # and both have a GPU path. This comment used to end "only the
        # softmax ensemble is CPU-only", which stopped being true when
        # `train_multiclass_gpu` landed and stopped being invisible when
        # `crossover_rules()` gained `apple-m4-metal-dense-multiclass`:
        # `auto` now reaches the device for a multiclass fit above the row
        # floor, on the strength of a measured 1.63x.
        device = self._resolve_device(
            n_rows,
            n_features,
            1 if n_classes == 2 else n_classes,
            objective_code=self._objective_code(n_classes),
            categorical=cat_buf is not None,
            has_eval_set=eval_set is not None,
        )
        self._multiclass = n_classes > 2
        if n_classes > 2:
            self._refuse_alternate_boosting("for a multiclass classifier")
        if eval_set is not None:
            self._refuse_alternate_boosting("with eval_set")

            def encode(y_valid, n_valid_rows, label):
                return _encode_like(y_valid, n_valid_rows, classes, label)

            self._fit_with_metrics(
                Xb,
                yb,
                n_rows,
                n_features,
                self._params(
                    w_addr,
                    device,
                    ic_flat,
                    ic_offsets,
                    mono_addr,
                    cat_buf,
                    contri_addr,
                ),
                device,
                _BINARY_LOGISTIC,
                eval_set,
                eval_names,
                eval_metric,
                early_stopping_rounds,
                min_delta,
                primary_metric,
                eval_sample_weight,
                task=(
                    _eval.BINARY if n_classes == 2 else _eval.MULTICLASS
                ),
                n_classes=n_classes,
                encode=encode,
                callbacks=callbacks,
            )
        elif n_classes == 2:
            self._model = _mojotrees.fit(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                _BINARY_LOGISTIC,
                self._params(
                    w_addr,
                    device,
                    ic_flat,
                    ic_offsets,
                    mono_addr,
                    cat_buf,
                    contri_addr,
                ),
            )
        else:
            self._model = _mojotrees.fit_multiclass(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                n_classes,
                self._params(
                    w_addr,
                    device,
                    ic_flat,
                    ic_offsets,
                    mono_addr,
                    cat_buf,
                    contri_addr,
                ),
            )
        self.classes_ = (
            _np.asarray(classes) if _np is not None else list(classes)
        )
        self.n_classes_ = n_classes
        self._record_fit(n_features, names, device)
        return self

    def _fit_sparse(self, X, y, sample_weight):
        """`fit` for SciPy sparse input. Same model, same semantics; the
        matrix is never densified."""
        # The labels are encoded before the buffers are built, because
        # class_weight has to reach the weight buffer the params carry.
        yb, classes = _arrays.encode_labels(y, X.shape[0])
        n_classes = len(classes)
        self._check_objective(n_classes)
        sample_weight = self._class_weight_rows(
            yb, X.shape[0], classes, sample_weight
        )
        buffers, n_rows, n_features, names, params, keep, device = (
            self._sparse_fit_params(
                X,
                sample_weight,
                self._objective_code(n_classes),
                1 if n_classes == 2 else n_classes,
            )
        )
        if n_rows != len(yb):
            raise ValueError("X and y must have the same number of rows")
        self._refuse_alternate_boosting("with sparse input")
        self._multiclass = n_classes > 2
        if n_classes == 2:
            self._model = _mojotrees.fit_csc(
                _addr(yb), _BINARY_LOGISTIC, params
            )
        else:
            self._model = _mojotrees.fit_multiclass_csc(
                _addr(yb), n_classes, params
            )
        self.classes_ = (
            _np.asarray(classes) if _np is not None else list(classes)
        )
        self.n_classes_ = n_classes
        self._record_fit(n_features, names, device)
        del keep
        return self

    def _predict_proba_sparse(
        self, X, raw_score, start_iteration, num_iteration, pred_leaf,
        pred_contrib, validate_features, device=None,
    ):
        """`predict_proba` for sparse input. Binary goes through the shared
        single-output path; multiclass has its own row-major buffer."""
        if self.n_classes_ == 2:
            out, n_rows = self._sparse_scores(
                X, raw_score, start_iteration, num_iteration, pred_leaf,
                pred_contrib, validate_features, device,
            )
            if raw_score:
                return _finish(out)
            if _np is not None:
                return _np.column_stack([1.0 - out, out])
            return [[1.0 - p, p] for p in out]
        if pred_leaf or pred_contrib:
            raise ValueError(
                "pred_leaf and pred_contrib do not take sparse input yet; "
                "densify with .toarray()"
            )
        if start_iteration != 0 or num_iteration is not None:
            raise ValueError(
                "iteration slicing does not take sparse input yet; densify "
                "with .toarray()"
            )
        if raw_score:
            raise ValueError(
                "raw_score does not take sparse multiclass input yet; "
                "densify with .toarray()"
            )
        buffers, n_rows = self._check_predict_X_sparse(X, validate_features)
        out = _out_buffer(n_rows * self.n_classes_)
        _mojotrees.predict_proba_csr(
            self._model,
            self._sparse_predict_params(device, buffers.params()),
            _addr(out),
        )
        if _np is not None:
            return out.reshape(n_rows, self.n_classes_)
        k = self.n_classes_
        return [list(out[r * k : (r + 1) * k]) for r in range(n_rows)]

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        device=None,
    ):
        """Class probabilities, shape (n_samples, n_classes), with columns
        in `classes_` order. Rows sum to 1.

        The prediction options are LightGBM's, and so are the shapes they
        return, which are not all probability matrices:

        - `raw_score` returns scores before the inverse link, so the result
          is not a distribution and does not sum to 1. A binary classifier
          returns the log-odds of the positive class, shape `(n_samples,)`,
          because there is one score per row and not one per class; a
          multiclass classifier returns the pre-softmax scores, shape
          `(n_samples, n_classes)`.
        - `pred_leaf` returns leaf ordinals, integer dtype. A binary
          classifier is a single-output ensemble, so its shape is
          `(n_samples, num_iteration)`; a multiclass classifier grows one
          tree per class per iteration, so its shape is
          `(n_samples, num_iteration * n_classes)` with column
          `i * n_classes + k` holding class k's tree in iteration i.

        `start_iteration` and `num_iteration` slice the boosting iterations
        as in `MojoTreesRegressor.predict`; the softmax is taken over the
        sliced scores, so probabilities are those of the truncated ensemble.
        `validate_features` and the `raw_score`/`pred_leaf` exclusion are
        also as documented there. `device` chooses where this one call
        runs, as it does for `MojoTreesRegressor.predict`, and applies to
        the softmax ensemble as well as the binary one."""
        self._check_predict_flags(raw_score, pred_leaf, pred_contrib)
        if _arrays.is_sparse(X):
            return self._predict_proba_sparse(
                X, raw_score, start_iteration, num_iteration, pred_leaf,
                pred_contrib, validate_features, device,
            )
        Xb, n_rows = self._check_predict_X(X, validate_features)
        start, stop = self._iteration_slice(start_iteration, num_iteration)
        if pred_leaf:
            return self._predict_leaf(Xb, n_rows, start, stop, device)
        if pred_contrib:
            self._refuse_device(device, "pred_contrib=True")
            return self._predict_contrib(Xb, n_rows, start, stop)
        raw = int(bool(raw_score))
        params = self._batch_params(device, start, stop, raw_score)
        if self.n_classes_ == 2:
            out = _out_buffer(n_rows)
            self._predict_batch(
                "predict_batch", "predict_range", Xb, n_rows, params, out
            )
            if raw:
                # One raw score per row, as LightGBM returns for a binary
                # model: there is no second column to complement.
                return _finish(out)
            if _np is not None:
                return _np.column_stack([1.0 - out, out])
            return [[1.0 - p, p] for p in out]
        out = _out_buffer(n_rows * self.n_classes_)
        self._predict_batch(
            "predict_proba_batch",
            "predict_proba_range",
            Xb,
            n_rows,
            params,
            out,
        )
        if _np is not None:
            return out.reshape(n_rows, self.n_classes_)
        k = self.n_classes_
        return [list(out[r * k : (r + 1) * k]) for r in range(n_rows)]

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        device=None,
    ):
        """Predicted labels, drawn from `classes_`. Defined as the argmax
        of `predict_proba`, so the two can never disagree.

        `raw_score`, `pred_leaf`, and `pred_contrib` ask for something that is
        not a label, so as in LightGBM they pass `predict_proba`'s result
        straight through with the shapes documented there rather than taking
        an argmax."""
        proba = self.predict_proba(
            X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            device=device,
        )
        if raw_score or pred_leaf or pred_contrib:
            return proba
        if _np is not None:
            return self.classes_[_np.argmax(proba, axis=1)]
        indices = [max(range(len(p)), key=p.__getitem__) for p in proba]
        return [self.classes_[i] for i in indices]

    def score(self, X, y, sample_weight=None):
        """Mean accuracy on `X` against labels `y`, weighted when
        `sample_weight` is given. This is scikit-learn's classifier
        `score`."""
        pred = self.predict(X)
        n_rows = len(pred)
        weights = (
            None
            if sample_weight is None
            else _arrays.check_sample_weight(sample_weight, n_rows)
        )
        if _np is not None:
            truth = _np.asarray(y)
            if truth.shape != (n_rows,):
                raise ValueError(
                    f"y must have shape ({n_rows},), got {truth.shape}"
                )
            correct = (truth == pred).astype(_np.float64)
            if weights is None:
                return float(correct.mean())
            return float((weights * correct).sum() / weights.sum())
        truth = list(y)
        if len(truth) != n_rows:
            raise ValueError(f"y must have length {n_rows}, got {len(truth)}")
        hits = [1.0 if t == p else 0.0 for t, p in zip(truth, pred)]
        if weights is None:
            return sum(hits) / n_rows
        return sum(w * h for w, h in zip(weights, hits)) / sum(weights)

    def save(self, path):
        """Write the fitted model to `path`. As with the regressor this
        stores the model and not the estimator; in particular the original
        class labels are not part of the format."""
        self._require_fitted()
        fitted_names = getattr(self, "feature_names_in_", None)
        names = [] if fitted_names is None else [str(n) for n in fitted_names]
        if self._multiclass:
            _mojotrees.save_multiclass(
                self._model, str(path), names, len(names)
            )
        else:
            _mojotrees.save(self._model, str(path), names, len(names))

    @classmethod
    def load(cls, path):
        """Load a saved model into a fresh estimator.

        The label mapping is not in the file, so a loaded classifier
        reports `classes_` as 0..n_classes-1 and predicts those codes. That
        is exactly right for a model trained on such labels and wrong for
        one trained on, say, strings: pickle the estimator instead when the
        labels matter. As with the regressor there is no `device_`.
        """
        est = cls()
        try:
            est._model = _mojotrees.load(str(path))
            est.n_classes_ = 2
            est._multiclass = False
            est.n_features_in_ = int(_mojotrees.n_features(est._model))
        except Exception:
            est._model = _mojotrees.load_multiclass(str(path))
            est.n_classes_ = int(_mojotrees.n_classes(est._model))
            est._multiclass = True
            est.n_features_in_ = int(
                _mojotrees.n_features_multiclass(est._model)
            )
        est.classes_ = (
            _np.arange(est.n_classes_)
            if _np is not None
            else list(range(est.n_classes_))
        )
        est.best_iteration_ = est._num_iterations()
        est.n_iter_ = est.best_iteration_
        names = list(_mojotrees.model_feature_names(str(path)))
        if names:
            est.feature_names_in_ = _arrays.name_array(names)
        est._restore_categorical()
        return est

    def __sklearn_tags__(self):
        return _estimator_tags("classifier")

class MojoTreesRanker(_Base):
    """LambdaRank learning to rank, LightGBM's `objective="lambdarank"`.

    `fit(X, y, group)` needs the query structure: `y` holds graded
    relevance labels (integers in [0, 30], 0 = irrelevant) and `group` holds
    the number of rows in each query, in row order, exactly as LightGBM's
    `group` parameter does. Rows of a query must be consecutive;
    `group_from_query_ids` builds `group` from a query id column and rejects
    a query whose rows are not. `predict` returns raw scores that are only
    meaningful in the order they induce within one query, so comparing
    scores across queries means nothing.

    `lambdarank_truncation_level`, `sigmoid`, and `lambdarank_norm` are
    LightGBM's parameters of the same names. `ndcg_eval_at` is the NDCG
    cutoff this estimator's `score` reports; `ndcg_score` takes any cutoff.

    `bagging_fraction` samples whole queries rather than rows, LightGBM's
    `bagging_by_query=true` behavior: a half-sampled query would be
    normalized against a maxDCG that no served ranking ever had. See
    src/mojotrees/ranking.mojo for the objective and its documented
    differences from LightGBM.

    Because `fit` requires a third argument, this estimator does not meet
    scikit-learn's `fit(X, y)` contract and will not drop into `Pipeline`
    or `cross_val_score`; `get_params`/`set_params`/`clone` still work.
    """

    def __init__(
        self,
        lambdarank_truncation_level=30,
        sigmoid=1.0,
        lambdarank_norm=True,
        ndcg_eval_at=5,
        label_gain=None,
        lambdarank_position_bias_regularization=0.0,
        pair_sampling_rate=1.0,
        pair_sampling_seed=5,
        max_dcg_cutoff=0,
        objective=None,
        loss_function=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.objective = objective
        self.loss_function = loss_function
        self.lambdarank_truncation_level = lambdarank_truncation_level
        self.sigmoid = sigmoid
        self.lambdarank_norm = lambdarank_norm
        self.ndcg_eval_at = ndcg_eval_at
        self.label_gain = label_gain
        self.lambdarank_position_bias_regularization = (
            lambdarank_position_bias_regularization
        )
        self.pair_sampling_rate = pair_sampling_rate
        self.pair_sampling_seed = pair_sampling_seed
        self.max_dcg_cutoff = max_dcg_cutoff

    #: Objective spellings that name what this ranker already trains:
    #: LightGBM's `lambdarank` and XGBoost's `rank:ndcg`, which are the same
    #: LambdaRank-with-NDCG family. CatBoost's `YetiRank` is a different
    #: pairwise loss and is deliberately absent.
    _RANKER_OBJECTIVES = ("lambdarank", "rank:ndcg", "rank:pairwise")

    def _check_objective(self):
        """Accept an objective spelling that names LambdaRank; refuse
        anything else.

        The ranker trains one objective and took no `objective` parameter
        at all, so `MojoTreesRanker(objective="lambdarank")` -- which is
        what an LGBMRanker script says, and it is true -- raised a
        TypeError. Naming the objective this estimator does train is now
        honored and nothing else is."""
        objective = self._resolve_alias("objective", "loss_function", None)
        if objective is None:
            return
        key = str(objective).strip().lower()
        if key in self._RANKER_OBJECTIVES:
            return
        raise ValueError(
            f"objective={objective!r} is not what MojoTreesRanker trains; "
            "it fits LambdaRank, spelled "
            + ", ".join(self._RANKER_OBJECTIVES)
            + _unimplemented_objective_note(objective)
        )

    @staticmethod
    def _objective_code():
        """LambdaRank, the one objective this estimator trains. It is a
        fixed fact rather than a parameter, which is why nothing here reads
        `self`; `fit` passes it to the native device policy, which blocks
        the accelerator on it (BLOCK_RANKING_OBJECTIVE), and
        `_metric_objective` reads it for the identity link the ranking
        metrics use."""
        return _LAMBDARANK

    def _rank_params(self, params, gb):
        if int(self.lambdarank_truncation_level) < 1:
            raise ValueError("lambdarank_truncation_level must be positive")
        if float(self.sigmoid) <= 0.0:
            raise ValueError("sigmoid must be positive")
        if int(self.ndcg_eval_at) < 1:
            raise ValueError("ndcg_eval_at must be positive")
        params["lambdarank_truncation_level"] = int(
            self.lambdarank_truncation_level
        )
        params["sigmoid"] = float(self.sigmoid)
        # int, not bool: the binding reads it as an integer.
        params["lambdarank_norm"] = int(bool(self.lambdarank_norm))
        params["ndcg_eval_at"] = int(self.ndcg_eval_at)
        params["group_addr"] = _addr(gb)
        params["n_groups"] = len(gb)
        # The advanced ranking parameters (src/mojotrees/ranking_advanced.mojo).
        # Defaults route to the plain LambdaRank trainer unchanged; anything
        # else routes to the advanced loop. The gain buffer is kept on self
        # so its address stays valid for the duration of the native call.
        if self.label_gain is None:
            self._label_gain_buffer = None
            params["n_label_gain"] = 0
            params["label_gain_addr"] = 0
        else:
            gains = _arrays.f64_vector(
                self.label_gain, len(list(self.label_gain)), "label_gain"
            )
            self._label_gain_buffer = gains
            params["n_label_gain"] = int(len(gains))
            params["label_gain_addr"] = _addr(gains)
        params["lambdarank_position_bias_regularization"] = float(
            self.lambdarank_position_bias_regularization
        )
        params["pair_sampling_rate"] = float(self.pair_sampling_rate)
        params["pair_sampling_seed"] = int(self.pair_sampling_seed)
        params["max_dcg_cutoff"] = int(self.max_dcg_cutoff)
        return params

    @staticmethod
    def _position_params(params, position, n_rows):
        """LightGBM's `Dataset.position`: a per-row integer code for the
        slot each document was shown in. Returns the buffer to keep alive."""
        if position is None:
            params["n_position_rows"] = 0
            params["position_addr"] = 0
            return None
        pos = _arrays.f64_vector(position, int(n_rows), "position")
        params["n_position_rows"] = int(n_rows)
        params["position_addr"] = _addr(pos)
        return pos

    def fit(
        self,
        X,
        y,
        group=None,
        sample_weight=None,
        eval_set=None,
        eval_group=None,
        eval_names=None,
        eval_metric=None,
        early_stopping_rounds=0,
        min_delta=0.0,
        primary_metric=0,
        eval_sample_weight=None,
        eval_X=None,
        eval_y=None,
        callbacks=None,
        position=None,
    ):
        """Fit on `X` (n_samples, n_features), relevance labels `y`, and
        `group`, the row count of each query in row order. `position` is
        LightGBM's `Dataset.position`: the slot each row was shown in when
        the labels were collected, which turns on unbiased LambdaRank
        (with `lambdarank_position_bias_regularization`).

        The validation arguments work as they do on `MojoTreesRegressor`,
        with `eval_group` carrying each validation set's own query
        boundaries: a validation set is a ranking problem of its own, so it
        needs them, and the default `eval_metric` is `ndcg` at the
        estimator's `ndcg_eval_at`. `eval_sample_weight` is rejected here,
        because NDCG has no weighted definition in LightGBM to match.

        Returns self.
        """
        eval_set = _eval_pairs(eval_set, eval_X, eval_y)
        _check_eval_arguments(
            eval_set,
            eval_metric,
            eval_sample_weight,
            early_stopping_rounds,
            callbacks,
        )
        if eval_set is not None:
            if eval_group is None:
                raise ValueError(
                    "a ranker's eval_set needs eval_group: the number of "
                    "rows in each validation query, in row order"
                )
            if eval_sample_weight is not None:
                raise ValueError(
                    "eval_sample_weight is not supported for a ranker; NDCG "
                    "has no weighted definition in LightGBM to match"
                )
        elif eval_group is not None:
            raise ValueError("eval_group needs an eval_set to describe")
        self._check_objective()
        self._refuse_alternate_boosting("for a ranker")
        self._reset_fitted()
        Xb, n_rows, n_features, names, cat_buf = self._fit_X(X)
        self._check_fit_structure(
            X, y, n_rows, n_features, sample_weight, group
        )
        yb = _arrays.check_target(y, n_rows)
        _check_relevance(yb, n_rows)
        gb = _group_buffer(group, n_rows)
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        mono_buf, mono_addr = self._monotone_buffer(n_features)
        contri_buf, contri_addr = self._feature_contri_buffer(n_features)
        device = self._resolve_device(
            n_rows,
            n_features,
            1,
            objective_code=self._objective_code(),
            categorical=cat_buf is not None,
            has_eval_set=eval_set is not None,
        )
        # Backstop; BLOCK_RANKING_OBJECTIVE is what refuses this on a build
        # whose native policy can be asked. See `_gpu_unsupported`.
        self._gpu_unsupported(device, "lambdarank trains on the CPU")
        params = self._rank_params(
            self._params(
                w_addr,
                device,
                ic_flat,
                ic_offsets,
                mono_addr,
                cat_buf,
                contri_addr,
            ),
            gb,
        )
        _preflight.native_preflight(params, n_features, device)
        position_buffer = self._position_params(params, position, n_rows)
        if eval_set is not None:

            def check_grades(y_valid, n_valid_rows, label):
                valid_yb = _arrays.check_target(y_valid, n_valid_rows, label)
                _check_relevance(valid_yb, n_valid_rows)
                return valid_yb

            self._fit_with_metrics(
                Xb,
                yb,
                n_rows,
                n_features,
                params,
                device,
                _LAMBDARANK,
                eval_set,
                eval_names,
                eval_metric,
                early_stopping_rounds,
                min_delta,
                primary_metric,
                eval_group=eval_group,
                task=_eval.RANKING,
                encode=check_grades,
                callbacks=callbacks,
            )
        else:
            self._model = _mojotrees.fit_ranker(
                _addr(Xb), n_rows, n_features, _addr(yb), params
            )
        del position_buffer
        self._record_fit(n_features, names, device)
        return self

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        validate_features=False,
        device=None,
    ):
        """Raw ranking scores for `X`, one per row. Sort a query's rows by
        this score, descending, to get its ranking; the values themselves
        are not comparable between queries.

        `raw_score` is accepted for signature compatibility and changes
        nothing: lambdarank has no inverse link, so a ranker's response scale
        is its raw scale and both settings return the same scores.

        `start_iteration`, `num_iteration`, `pred_leaf`,
        `validate_features`, and `device` behave as in
        `MojoTreesRegressor.predict`; a ranker is a single-output ensemble,
        so `pred_leaf` returns shape `(n_samples, num_iteration)`.
        Lambdarank *training* is CPU-only, but a fitted ranker is an
        ordinary single-output ensemble, so predicting it is not."""
        self._check_predict_flags(raw_score, pred_leaf, pred_contrib)
        Xb, n_rows = self._check_predict_X(X, validate_features)
        start, stop = self._iteration_slice(start_iteration, num_iteration)
        if pred_leaf:
            return self._predict_leaf(Xb, n_rows, start, stop, device)
        if pred_contrib:
            self._refuse_device(device, "pred_contrib=True")
            return self._predict_contrib(Xb, n_rows, start, stop)
        out = _out_buffer(n_rows)
        self._predict_batch(
            "predict_batch",
            "predict_range",
            Xb,
            n_rows,
            self._batch_params(device, start, stop, raw_score),
            out,
        )
        return _finish(out)

    def score(self, X, y, group=None, sample_weight=None):
        """Mean NDCG@`ndcg_eval_at` of this model's ranking of `X`.
        `sample_weight` is accepted for signature compatibility and
        ignored, as a weighted NDCG has no LightGBM definition to match."""
        return ndcg_score(self.predict(X), y, group, self.ndcg_eval_at)

    def save(self, path):
        """Write the fitted model to `path` in mojotrees's versioned text
        format. Query boundaries are training data, not model state, so
        they do not travel with it."""
        self._require_fitted()
        fitted_names = getattr(self, "feature_names_in_", None)
        names = [] if fitted_names is None else [str(n) for n in fitted_names]
        _mojotrees.save(self._model, str(path), names, len(names))

    @classmethod
    def load(cls, path):
        """Load a saved ranker into a fresh estimator."""
        est = cls()
        est._model = _mojotrees.load(str(path))
        est.n_features_in_ = int(_mojotrees.n_features(est._model))
        est.best_iteration_ = est._num_iterations()
        est.n_iter_ = est.best_iteration_
        names = list(_mojotrees.model_feature_names(str(path)))
        if names:
            est.feature_names_in_ = _arrays.name_array(names)
        est._restore_categorical()
        return est
