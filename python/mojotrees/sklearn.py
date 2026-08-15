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

# Defaults of the two regularization parameters, named so the constructor
# signature and the alias resolution in `_params` cannot drift apart.
_LAMBDA_L1 = 0.0
_LAMBDA_L2 = 1.0

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


class _Base(_ParamsMixin):
    """Shared hyperparameters, mojotrees defaults (LightGBM-matched).

    `importance_type` selects what `feature_importances_` reports: "split",
    LightGBM's default, counts the nodes that split on each feature, and
    "gain" sums the gain those splits earned.

    Native LightGBM parameter names are canonical. The spellings used by
    LightGBM's scikit-learn estimators are accepted as aliases:
    `min_child_samples` for `min_data_in_leaf`, `min_child_weight` for
    `min_child_hess`, `reg_alpha`/`reg_lambda` for `lambda_l1`/`lambda_l2`,
    `subsample`/`subsample_freq` for `bagging_fraction`/`bagging_freq`, and
    `device_type` for `device`. Either spelling works; setting both members
    of a pair to different non-default values raises, where LightGBM warns
    and keeps one.

    `max_depth` bounds the depth of any leaf, counted in edges from the root,
    so `max_depth=1` gives stumps; `max_depth<=0` (the default, -1) means
    unlimited. Under the default growth a depth-bounded tree is still
    unbalanced and usually has fewer than `2**max_depth` leaves.

    `grow_policy` is XGBoost's parameter of that name (LightGBM has no
    equivalent). "leafwise", the default and LightGBM's growth, splits the
    leaf with the largest gain anywhere in the tree next; "depthwise" splits
    every leaf at one depth before any deeper one, so a tree fills level by
    level and is balanced. `num_leaves` stays a hard bound in both: a
    depth-wise level that would overrun it is admitted as its highest-gain
    prefix, so at the default `num_leaves=31` and unlimited `max_depth` a
    depth-wise tree fills four levels (16 leaves) and half of a fifth. Set
    `max_depth` deliberately for depth-wise runs; a leaf-wise configuration
    is not a sensible one to inherit. "lossguide" is accepted as XGBoost's
    alias for "leafwise". Depth-wise growth is honored on the CPU and GPU
    trainers alike (dense and sparse); the distributed prototype rejects it.

    `bagging_fraction` and `bagging_freq` are LightGBM's row bagging: every
    `bagging_freq` rounds, each row is kept independently with probability
    `bagging_fraction` and the trees of the following rounds are grown on
    that sample. `bagging_freq=0` (the default) or `bagging_fraction=1`
    disables it. `bagging_seed` makes a run reproducible; the same seed and
    data give the same model on CPU and GPU alike.

    `boosting` (LightGBM's parameter of that name, aliased `boosting_type`)
    selects the training strategy: "gbdt", the default, trains every tree on
    every row, and "goss" is Gradient-based One-Side Sampling. Under GOSS
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
    defaults to False); the non-uniform variant is refused. "rf" is random
    forest mode: every tree fits the same gradients and the model averages
    them, so `learning_rate` is ignored (trained at 1.0, as LightGBM does)
    and it needs a source of per-tree randomness, `bagging_fraction < 1` with
    `bagging_freq > 0` or `feature_fraction < 1`. Both modes train on the
    CPU, on dense input, without `eval_set` or a callable objective, and
    single-output only (a multiclass classifier or a ranker refuses them);
    the fitted model is an ordinary one for prediction, saving, and
    inspection.

    `feature_fraction` samples that share of the features once per tree and
    `feature_fraction_bynode` samples again at every node from the tree's own
    set, both without replacement and both reproducible from
    `feature_fraction_seed`. Fractions must be in (0, 1]; 1.0 (the default)
    means no subsampling. As in LightGBM, at least 2 features are selected
    whenever the data has that many.

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
        max_depth=-1,
        grow_policy="leafwise",
        learning_rate=0.1,
        n_estimators=100,
        min_data_in_leaf=20,
        min_child_samples=None,
        lambda_l2=_LAMBDA_L2,
        lambda_l1=_LAMBDA_L1,
        reg_lambda=None,
        reg_alpha=None,
        min_child_hess=1e-3,
        min_child_weight=None,
        max_bin=255,
        device="cpu",
        device_type=None,
        interaction_constraints=None,
        monotone_constraints=None,
        bagging_fraction=1.0,
        subsample=None,
        bagging_freq=0,
        subsample_freq=None,
        bagging_seed=3,
        boosting="gbdt",
        boosting_type=None,
        top_rate=0.2,
        other_rate=0.1,
        goss_seed=3,
        goss_warmup_rounds=-1,
        drop_rate=0.1,
        max_drop=50,
        skip_drop=0.5,
        xgboost_dart_mode=False,
        uniform_drop=True,
        drop_seed=4,
        feature_fraction=1.0,
        feature_fraction_bynode=1.0,
        feature_fraction_seed=2,
        use_missing=True,
        categorical_feature="auto",
        categorical_features=None,
        max_cat_to_onehot=4,
        max_cat_threshold=32,
        cat_smooth=10.0,
        cat_l2=10.0,
        min_data_per_group=100,
        min_gain_to_split=0.0,
        min_split_gain=None,
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
    ):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.min_child_samples = min_child_samples
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_hess = min_child_hess
        self.min_child_weight = min_child_weight
        self.max_bin = max_bin
        self.device = device
        self.device_type = device_type
        self.interaction_constraints = interaction_constraints
        self.monotone_constraints = monotone_constraints
        self.bagging_fraction = bagging_fraction
        self.subsample = subsample
        self.bagging_freq = bagging_freq
        self.subsample_freq = subsample_freq
        self.bagging_seed = bagging_seed
        self.boosting = boosting
        self.boosting_type = boosting_type
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.goss_seed = goss_seed
        self.goss_warmup_rounds = goss_warmup_rounds
        self.drop_rate = drop_rate
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self.uniform_drop = uniform_drop
        self.drop_seed = drop_seed
        self.feature_fraction = feature_fraction
        self.feature_fraction_bynode = feature_fraction_bynode
        self.feature_fraction_seed = feature_fraction_seed
        self.use_missing = use_missing
        self.categorical_feature = categorical_feature
        self.categorical_features = categorical_features
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.min_data_per_group = min_data_per_group
        self.min_gain_to_split = min_gain_to_split
        self.min_split_gain = min_split_gain
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
        self._reset_fitted()

    def _interaction_buffers(self, n_features):
        """Validated float64 buffers for `interaction_constraints`: the
        flattened group features and one more offset than there are groups.
        Both must stay referenced while their addresses are in use.
        `(None, None)` when unconstrained."""
        groups = self.interaction_constraints
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
        signs = self.monotone_constraints
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
        """Raise when `boosting` is dart or rf and `where` names a fit path
        that only the plain trainer serves.

        Only `_mojotrees.fit` (dense, single-output, CPU, no eval_set, no
        callable objective) reads the `boosting` key and routes dart and rf
        to `alternate_boosting.fit_boosting`. Every other native entry point
        would train gbdt and say nothing, so those paths refuse by name here
        instead of quietly fitting a different model than the one asked for.
        """
        boosting = self._resolve_boosting()
        if boosting in ("dart", "rf"):
            raise ValueError(
                f"boosting={boosting!r} is not available {where}; it trains "
                "dense, single-output models on the CPU without eval_set or "
                "a callable objective"
            )

    def _resolve_boosting(self):
        """The effective boosting strategy, "gbdt" or "goss".

        `_resolve_alias` compares numerically, so the string-valued
        `boosting` / `boosting_type` pair resolves here instead, with the
        same rule: an unset alias leaves the primary alone, and two
        different non-default values raise.
        """
        boosting = self.boosting
        if self.boosting_type is not None:
            if boosting != "gbdt" and boosting != self.boosting_type:
                raise ValueError(
                    f"boosting={boosting!r} and "
                    f"boosting_type={self.boosting_type!r} are aliases with "
                    "different values; set only one"
                )
            boosting = self.boosting_type
        if boosting not in _BOOSTING_TYPES:
            raise ValueError(
                f"unknown boosting {boosting!r}; expected one of "
                + ", ".join(sorted(_BOOSTING_TYPES))
            )
        return boosting

    def _resolve_alias(self, primary, alias, default):
        """The effective value of a parameter that has a LightGBM alias.

        scikit-learn requires `__init__` to store every argument unmodified,
        so aliases are resolved here, at fit time, rather than in the
        constructor. An unset alias is `None` and leaves the primary alone.
        LightGBM warns and keeps one value when a parameter and its alias
        disagree; mojotrees raises instead, so a typo cannot silently train
        a different model.
        """
        alias_value = getattr(self, alias)
        if alias_value is None:
            return getattr(self, primary)
        primary_value = getattr(self, primary)
        if primary_value != default and primary_value != alias_value:
            raise ValueError(
                f"{primary}={primary_value} and {alias}={alias_value} are "
                "aliases with different values; set only one"
            )
        return alias_value

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
        min_data_in_leaf = self._resolve_alias(
            "min_data_in_leaf", "min_child_samples", 20
        )
        min_child_hess = self._resolve_alias(
            "min_child_hess", "min_child_weight", 1e-3
        )
        lambda_l1 = self._resolve_alias("lambda_l1", "reg_alpha", _LAMBDA_L1)
        lambda_l2 = self._resolve_alias("lambda_l2", "reg_lambda", _LAMBDA_L2)
        bagging_fraction = self._resolve_alias(
            "bagging_fraction", "subsample", 1.0
        )
        bagging_freq = self._resolve_alias(
            "bagging_freq", "subsample_freq", 0
        )
        min_gain_to_split = self._resolve_alias(
            "min_gain_to_split", "min_split_gain", 0.0
        )
        monotone_penalty = self._resolve_alias(
            "monotone_penalty", "monotone_constraints_penalty", 0.0
        )
        # Same ranges src/mojotrees/params.mojo and callback.mojo enforce,
        # so an estimator cannot construct a configuration the trainer
        # rejects (or, worse, quietly degenerates on).
        if int(self.num_leaves) < 2:
            raise ValueError("num_leaves must be at least 2")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if int(self.max_bin) < 2:
            raise ValueError("max_bin must be at least 2")
        if float(lambda_l1) < 0.0:
            raise ValueError("lambda_l1 must be nonnegative")
        if float(lambda_l2) < 0.0:
            raise ValueError("lambda_l2 must be nonnegative")
        if not 0.0 < float(bagging_fraction) <= 1.0:
            raise ValueError("bagging_fraction must be in (0, 1]")
        if int(bagging_freq) < 0:
            raise ValueError("bagging_freq must be nonnegative")
        boosting = self._resolve_boosting()
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
                    "boosting='goss' cannot be combined with row bagging; "
                    "leave bagging_freq at 0 or bagging_fraction at 1.0"
                )
        if boosting == "dart":
            if not 0.0 <= float(self.drop_rate) <= 1.0:
                raise ValueError("drop_rate must be in [0, 1]")
            if not 0.0 <= float(self.skip_drop) <= 1.0:
                raise ValueError("skip_drop must be in [0, 1]")
            if int(self.drop_seed) < 0:
                raise ValueError("drop_seed must be nonnegative")
        # A forest averages its trees, so LightGBM's RF ignores
        # learning_rate and trains at 1.0; the same here, whatever the
        # estimator was given, because boosting_rf refuses any other rate.
        learning_rate = 1.0 if boosting == "rf" else float(self.learning_rate)
        if not 0.0 < float(self.feature_fraction) <= 1.0:
            raise ValueError("feature_fraction must be in (0, 1]")
        if not 0.0 < float(self.feature_fraction_bynode) <= 1.0:
            raise ValueError("feature_fraction_bynode must be in (0, 1]")
        grow_policy = str(self.grow_policy)
        if grow_policy not in _GROW_POLICIES:
            raise ValueError(
                "grow_policy must be 'leafwise' (alias 'lossguide') or "
                f"'depthwise', got {self.grow_policy!r}"
            )
        if int(self.max_cat_to_onehot) < 0:
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
            "num_leaves": int(self.num_leaves),
            "max_depth": int(self.max_depth),
            # Sent as its canonical name; the binding parses it with the same
            # function the parameter string goes through.
            "grow_policy": _GROW_POLICIES[grow_policy],
            "learning_rate": learning_rate,
            "n_estimators": int(self.n_estimators),
            "min_data_in_leaf": int(min_data_in_leaf),
            "lambda_l2": float(lambda_l2),
            "lambda_l1": float(lambda_l1),
            "min_child_hess": float(min_child_hess),
            "max_bin": int(self.max_bin),
            # int, not bool: the binding reads it as an integer.
            "use_missing": int(bool(self.use_missing)),
            "sample_weight_addr": int(sample_weight_addr),
            # The objective's scalar parameter, whichever of alpha, fair_c,
            # and tweedie_variance_power it is: one trainer slot holds it.
            "alpha": float(self._alpha_slot()),
            "device": device,
            "bagging_fraction": float(bagging_fraction),
            "bagging_freq": int(bagging_freq),
            "bagging_seed": int(self.bagging_seed),
            # int, not bool: the binding reads it as an integer.
            "goss": int(goss),
            "top_rate": float(self.top_rate),
            "other_rate": float(self.other_rate),
            "goss_seed": int(self.goss_seed),
            "goss_warmup_rounds": int(self.goss_warmup_rounds),
            # LightGBM's `boosting` by name; the binding routes dart and rf
            # to alternate_boosting.fit_boosting and leaves gbdt and goss on
            # the trainer they always used. The dart bundle is read only
            # when the mode is dart; ints, not bools, as everywhere here.
            "boosting": boosting,
            "drop_rate": float(self.drop_rate),
            "max_drop": int(self.max_drop),
            "skip_drop": float(self.skip_drop),
            "xgboost_dart_mode": int(bool(self.xgboost_dart_mode)),
            "uniform_drop": int(bool(self.uniform_drop)),
            "drop_seed": int(self.drop_seed),
            "feature_fraction": float(self.feature_fraction),
            "feature_fraction_bynode": float(self.feature_fraction_bynode),
            "feature_fraction_seed": int(self.feature_fraction_seed),
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
            "max_cat_to_onehot": int(self.max_cat_to_onehot),
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
            "max_delta_step": float(self.max_delta_step),
            "path_smooth": float(self.path_smooth),
            # int, not bool: the binding reads it as an integer.
            "extra_trees": int(bool(self.extra_trees)),
            "extra_seed": int(self.extra_seed),
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
            "enable_bundle": int(bool(self.enable_bundle)),
            "max_conflict_rate": float(self.max_conflict_rate),
            "max_bundle_bins": int(self.max_bundle_bins),
            "max_bundle_size": int(self.max_bundle_size),
            "max_nondefault_rate": float(self.max_nondefault_rate),
            "min_reduction": float(self.min_reduction),
            "bundle_missing": int(bool(self.bundle_missing)),
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
        lambdarank, `sparse` blocks it because there is no sparse GPU
        histogram, `has_eval_set` blocks it because validation metrics are
        scored on the host, and `max_bin` (read off the estimator) and
        `categorical` and `use_missing` are reported. Leaving one
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
                max_bin=int(self.max_bin),
                sparse=bool(sparse),
                categorical=bool(categorical),
                has_missing=bool(self.use_missing),
                has_eval_set=bool(has_eval_set),
            )
            # DeviceUnavailableError is a RuntimeError subclass carrying the
            # native refusal text, so it propagates as what this method has
            # always raised, with the report attached.
            return _policy.select_device(device, workload).resolved
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
        BLOCK_SPARSE_INPUT, BLOCK_VALIDATION_SET, BLOCK_CUSTOM_OBJECTIVE,
        and BLOCK_RANKING_OBJECTIVE. Against such a build these checks
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
        return {
            "learning_rate": float(self.learning_rate),
            "num_leaves": int(self.num_leaves),
            "max_depth": int(self.max_depth),
            "min_data_in_leaf": int(
                self._resolve_alias(
                    "min_data_in_leaf", "min_child_samples", 20
                )
            ),
            # The environment uses LightGBM's name for this one; the
            # estimator's own spelling of it is `min_child_hess`.
            "min_sum_hessian_in_leaf": float(
                self._resolve_alias("min_child_hess", "min_child_weight", 1e-3)
            ),
            "lambda_l1": float(
                self._resolve_alias("lambda_l1", "reg_alpha", _LAMBDA_L1)
            ),
            "lambda_l2": float(
                self._resolve_alias("lambda_l2", "reg_lambda", _LAMBDA_L2)
            ),
            "feature_fraction": float(self.feature_fraction),
            "feature_fraction_bynode": float(self.feature_fraction_bynode),
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
            eval_metric, task, getattr(self, "objective", None)
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
            runner.end_iteration = int(self.n_estimators)
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
        explicit `"gpu"` is the native one; there is no sparse accelerator
        kernel, the same way there is no sparse GPU histogram.
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

        Sparse training runs on the CPU. The GPU histogram kernels take a
        dense binned matrix, so rather than densify behind the caller's
        back, `device="gpu"` is refused and `device="auto"` resolves to the
        CPU, which is what it would pick anyway. Both of those answers come
        from the native policy, which is asked with `sparse=True` (that is
        its BLOCK_SPARSE_INPUT), not decided here; `objective_code` and
        `n_outputs` are passed so the refusal it writes names every reason
        the request was blocked, not just this one.
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
        # Backstop for a shape-only build; see `_gpu_unsupported`.
        self._gpu_unsupported(
            device,
            "sparse input trains on the CPU, and there is no sparse GPU "
            "kernel yet",
            "Densify with .toarray() to train on an accelerator.",
        )
        params = self._params(
            w_addr, "cpu", ic_flat, ic_offsets, mono_addr, cat_buf, contri_addr
        )
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
        return buffers, n_rows, n_features, names, params, keep

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
    _OBJECTIVES = {
        "regression": _SQUARED_ERROR,
        "regression_l2": _SQUARED_ERROR,
        "l2": _SQUARED_ERROR,
        "mean_squared_error": _SQUARED_ERROR,
        "mse": _SQUARED_ERROR,
        "huber": _HUBER,
        "quantile": _QUANTILE,
        "mae": _L1,
        "regression_l1": _L1,
        "l1": _L1,
        "mean_absolute_error": _L1,
        "poisson": _POISSON,
        "gamma": _GAMMA,
        "tweedie": _TWEEDIE,
        "mape": _MAPE,
        "mean_absolute_percentage_error": _MAPE,
        "fair": _FAIR,
        "cross_entropy": _CROSS_ENTROPY,
        "xentropy": _CROSS_ENTROPY,
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
        alpha=0.9,
        fair_c=1.0,
        tweedie_variance_power=1.5,
        base_score=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.objective = objective
        self.alpha = alpha
        self.fair_c = fair_c
        self.tweedie_variance_power = tweedie_variance_power
        self.base_score = base_score

    def _objective_code(self):
        if callable(self.objective):
            return _CUSTOM
        code = None
        if self.objective in self._OBJECTIVES:
            # The registry is the resolver; the literal says which of its
            # spellings are regression objectives.
            code = _objective_code_of_name(self.objective)
            if code is None:
                code = self._OBJECTIVES[self.objective]
        if code is None:
            raise ValueError(
                f"unknown objective {self.objective!r}; expected one of "
                + ", ".join(sorted(self._OBJECTIVES))
                + _unimplemented_objective_note(self.objective)
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
                    f"{self.objective!r}"
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

        Returns self.
        """
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
            buffers, n_rows, n_features, names, params, keep = (
                self._sparse_fit_params(X, sample_weight, objective)
            )
            yb = _arrays.check_target(y, n_rows)
            self._model = _mojotrees.fit_csc(_addr(yb), objective, params)
            self._record_fit(n_features, names, "cpu")
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
        fobj = self.objective
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

        `raw_score` and `pred_leaf` cannot be combined."""
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

    def __init__(self, objective=None, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.class_weight = class_weight

    def _objective_code(self, n_classes=None):
        """The native objective code this classifier trains, or None when
        the class count is the answer instead of a code.

        Two classes is `binary_logistic`, a built-in single-output
        objective the device vocabulary routes. Softmax is not one: it is
        its own trainer growing one tree per class per round, and what the
        native policy gates on there is `n_outputs`, which the caller
        passes. Naming a code for it would assert something the request is
        not, so it stays undeclared, which the native decision reports as
        an incomplete request rather than assuming either way.

        `n_classes` is the count the current fit has just encoded, for the
        callers that ask before `n_classes_` exists; without it the fitted
        attribute answers, and before a fit there is nothing to answer
        with.
        """
        if n_classes is None:
            n_classes = getattr(self, "n_classes_", None)
        if n_classes is None:
            return None
        return _BINARY_LOGISTIC if int(n_classes) == 2 else None

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
        if self.objective is not None:
            raise ValueError(
                "MojoTreesClassifier takes no objective; custom objectives "
                "are single-output only. Use MojoTreesRegressor with your "
                "objective and apply your own link (a sigmoid, say) to the "
                "raw predictions."
            )
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
        # class_weight becomes ordinary row weights before anything else
        # sees it, so the trainer has one weighting mechanism, not two.
        sample_weight = self._class_weight_rows(
            yb, n_rows, classes, sample_weight
        )
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        mono_buf, mono_addr = self._monotone_buffer(n_features)
        contri_buf, contri_addr = self._feature_contri_buffer(n_features)
        # Binary is single-output (one tree per round), so it has a GPU
        # path; only the softmax ensemble is CPU-only.
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
        sample_weight = self._class_weight_rows(
            yb, X.shape[0], classes, sample_weight
        )
        buffers, n_rows, n_features, names, params, keep = (
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
        self._record_fit(n_features, names, "cpu")
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
