"""scikit-learn style Python API for mojoboost.

Build the extension first: `bindings/build.sh` (produces `_mojoboost.so`
next to this file). numpy is used when available; plain Python sequences
(lists of rows) work without it.

    from mojoboost import MojoBoostRegressor
    model = MojoBoostRegressor().fit(X, y)
    pred = model.predict(X)

scikit-learn conventions
------------------------
The estimators implement `get_params`, `set_params`, `fit`, `predict`,
`predict_proba` (classifier), and `score`, record `n_features_in_`,
`feature_names_in_`, `classes_`, `feature_importances_`, and
`best_iteration_` when fitted, raise `NotFittedError` before that, and
pickle. `clone`, `Pipeline`, `GridSearchCV`, and `cross_val_score` work.
scikit-learn is optional: nothing here imports it except the
`__sklearn_tags__` hook that scikit-learn itself calls.

`check_estimator`'s full suite has not been run, so this is "scikit-learn
style" and not a compliance claim. Two known deviations:

- subclasses forward the shared hyperparameters through `**kwargs`, so
  `get_params()` lists them but `inspect.signature` does not
- `best_iteration_` is always set (see below), where LightGBM sets it only
  when early stopping ran

Validation follows LightGBM's scikit-learn wrapper, which validates with
`force_all_finite="allow-nan"`: `X` may hold NaN, mojoboost's missing-value
marker, but not infinities, and `y` and `sample_weight` must be finite.
Sparse input is rejected rather than densified silently.

`best_iteration_` is the number of boosting iterations the fitted model
kept. Validation-set early stopping is not reachable from Python yet (the
Mojo API has `train_with_valid`), so today this is `n_estimators` unless
training stopped early because the objective converged.

Estimators take `device="cpu"` (the default and the dependable backend),
`device="gpu"`, or `device="auto"`. `"gpu"` raises when no accelerator is
available or when the GPU path does not cover the workload, rather than
falling back silently; `"auto"` picks a backend for you and currently
always picks the CPU. `gpu_available()` reports whether this build can
train on an accelerator. Fitting records the backend that ran on
`device_`. See src/mojoboost/device.mojo for the full policy.

`MojoBoostRegressor(objective=f)` accepts a Python objective callback,
`f(raw, y) -> (grad, hess)`, called once per boosting round rather than per
row. It costs one Python call and two array passes per round; measure it
with bench/bench_custom_objective.py before reaching for it, and use the
native Mojo interface (src/mojoboost/objective.mojo) when the objective sits
on a hot path.

`interaction_constraints` takes LightGBM's parameter as a list of
feature-index lists, e.g. `[[0, 1], [2, 3]]`. Two features may share a
root-to-leaf path only if one group holds both, and transitively everything
else on that path. Groups may overlap. Note the LightGBM behavior this
matches: a feature in no group is never split on at all, so constraining a
few features drops the rest from the model. To leave a feature free to
interact with everything, put it in every group; a group of its own instead
isolates it. See src/mojoboost/interaction.mojo for the exact rule.
"""

import array as _array
import os as _os
import tempfile as _tempfile
import warnings as _warnings

from . import _arrays, _mojoboost
from ._sklearn import NotFittedError, ParamsMixin as _ParamsMixin
from ._sklearn import estimator_tags as _estimator_tags

_np = _arrays.np

# Keep in sync with python/pyproject.toml.
__version__ = "0.1.0"

__all__ = [
    "MojoBoostRegressor",
    "MojoBoostClassifier",
    "MojoBoostRanker",
    "NotFittedError",
    "gpu_available",
    "group_from_query_ids",
    "ndcg_score",
]

# Buffer plumbing and input validation live in _arrays; these names are the
# spellings the estimator code below has always used.
_as_column_major = _arrays.column_major
_as_f64_vector = _arrays.f64_vector
_addr = _arrays.addr
_out_buffer = _arrays.out_buffer
_finish = _arrays.finish

_IMPORTANCE_TYPES = {"split": 0, "gain": 1}

_DEVICES = ("cpu", "gpu", "auto")
_BOOSTING_TYPES = ("gbdt", "goss")

_SQUARED_ERROR = 0
_BINARY_LOGISTIC = 1
_HUBER = 3
_QUANTILE = 4
_L1 = 5
_CUSTOM = 6

# Defaults of the two regularization parameters, named so the constructor
# signature and the alias resolution in `_params` cannot drift apart.
_LAMBDA_L1 = 0.0
_LAMBDA_L2 = 1.0

# The largest relevance label a ranker accepts, the range of LightGBM's
# default label_gain (src/mojoboost/ranking.mojo).
_MAX_RELEVANCE_LABEL = 30


def _store_vector(out, values, n_rows, name):
    """Copy a custom objective's returned array into the buffer the trainer
    reads, checking the length here so the error names the offending array
    instead of surfacing as a generic shape error."""
    if _np is not None:
        arr = _np.asarray(values, dtype=_np.float64)
        if arr.shape != (n_rows,):
            raise ValueError(
                f"custom objective returned {name} with shape "
                f"{arr.shape}, expected ({n_rows},)"
            )
        out[:] = arr
        return
    if len(values) != n_rows:
        raise ValueError(
            f"custom objective returned {len(values)} {name} values, "
            f"expected {n_rows}"
        )
    for i in range(n_rows):
        out[i] = float(values[i])


def gpu_available():
    """True when this build can train on an accelerator. False on a
    CPU-only build and when `MOJOBOOST_DISABLE_GPU=1` is set."""
    return bool(_mojoboost.gpu_available())


class _Base(_ParamsMixin):
    """Shared hyperparameters, mojoboost defaults (LightGBM-matched).

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
    unlimited. Growth stays leaf-wise, so a depth-bounded tree is still
    unbalanced and usually has fewer than `2**max_depth` leaves.

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
    )

    def __init__(
        self,
        num_leaves=31,
        max_depth=-1,
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
        feature_fraction=1.0,
        feature_fraction_bynode=1.0,
        feature_fraction_seed=2,
        use_missing=True,
        importance_type="split",
    ):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
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
        self.feature_fraction = feature_fraction
        self.feature_fraction_bynode = feature_fraction_bynode
        self.feature_fraction_seed = feature_fraction_seed
        self.use_missing = use_missing
        self.importance_type = importance_type
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
        disagree; mojoboost raises instead, so a typo cannot silently train
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
        self, sample_weight_addr, device, ic_flat=None, ic_offsets=None
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
            # disables bagging under GOSS; mojoboost rejects the pair.
            if int(bagging_freq) > 0 and float(bagging_fraction) < 1.0:
                raise ValueError(
                    "boosting='goss' cannot be combined with row bagging; "
                    "leave bagging_freq at 0 or bagging_fraction at 1.0"
                )
        if not 0.0 < float(self.feature_fraction) <= 1.0:
            raise ValueError("feature_fraction must be in (0, 1]")
        if not 0.0 < float(self.feature_fraction_bynode) <= 1.0:
            raise ValueError("feature_fraction_bynode must be in (0, 1]")
        return {
            "num_leaves": int(self.num_leaves),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "n_estimators": int(self.n_estimators),
            "min_data_in_leaf": int(min_data_in_leaf),
            "lambda_l2": float(lambda_l2),
            "lambda_l1": float(lambda_l1),
            "min_child_hess": float(min_child_hess),
            "max_bin": int(self.max_bin),
            # int, not bool: the binding reads it as an integer.
            "use_missing": int(bool(self.use_missing)),
            "sample_weight_addr": int(sample_weight_addr),
            "alpha": float(getattr(self, "alpha", 0.9)),
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
        }

    def _resolve_device(self, n_rows, n_features, n_outputs):
        """The backend that will actually run, "cpu" or "gpu". Names are
        case-insensitive, as LightGBM treats `device_type`. Raises
        ValueError for an unknown `device` and RuntimeError when "gpu" is
        requested but unavailable or unsupported; "gpu" never falls back to
        the CPU."""
        device = self._resolve_alias("device", "device_type", "cpu")
        if not isinstance(device, str) or device.lower() not in _DEVICES:
            raise ValueError(
                f"unknown device {device!r}; expected one of "
                + ", ".join(_DEVICES)
            )
        try:
            return _mojoboost.resolve_device(
                device.lower(), int(n_rows), int(n_features), int(n_outputs)
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from None

    def _weight_buffer(self, sample_weight, n_rows):
        """Validated weight buffer and its address (buffer must stay
        referenced while the address is in use); (None, 0) when absent.
        Weights must be finite, nonnegative, and not all zeros."""
        if sample_weight is None:
            return None, 0
        wb = _arrays.check_sample_weight(sample_weight, n_rows)
        return wb, _addr(wb)

    # -- fitted state ----------------------------------------------------

    def _reset_fitted(self):
        """Drop everything a previous fit left behind. Called by `__init__`
        and at the top of every `fit`, so a failed refit does not leave the
        estimator claiming to hold the older model."""
        self._model = None
        self._importance_cache = None
        self._multiclass = False
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

    def _check_feature_names(self, names):
        """Compare the column names of an incoming matrix against the ones
        recorded at fit time, warning when only one side has them and
        raising when both do and they disagree, as scikit-learn does."""
        fitted = getattr(self, "feature_names_in_", None)
        if fitted is None and names is None:
            return
        if fitted is None:
            _warnings.warn(
                f"X has feature names, but {type(self).__name__} was fitted "
                "without feature names",
                UserWarning,
                stacklevel=3,
            )
            return
        if names is None:
            _warnings.warn(
                "X does not have valid feature names, but "
                f"{type(self).__name__} was fitted with feature names",
                UserWarning,
                stacklevel=3,
            )
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
        self.best_iteration_ = self._num_iterations()
        self._importance_cache = {
            kind: self._raw_importance(kind) for kind in _IMPORTANCE_TYPES
        }

    def _check_predict_X(self, X):
        """Validate a matrix for prediction against the fitted model."""
        self._require_fitted()
        Xb, n_rows, n_features, names = _arrays.check_X(X)
        self._check_n_features(n_features)
        self._check_feature_names(names)
        return Xb, n_rows

    # -- fitted attributes -----------------------------------------------

    def _num_iterations(self):
        if self._multiclass:
            return int(_mojoboost.num_iterations_multiclass(self._model))
        return int(_mojoboost.num_iterations(self._model))

    def _raw_importance(self, importance_type):
        out = _out_buffer(self.n_features_in_)
        query = (
            _mojoboost.feature_importance_multiclass
            if self._multiclass
            else _mojoboost.feature_importance
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
        fitted, so changing `importance_type` afterwards costs nothing and
        a pickled estimator keeps both. A model read back with `load()` is
        the exception: gains are not part of the serialized format, so it
        reports zero gain importance and warns.
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
        if importance_type == "gain":
            _warnings.warn(
                "split gains are not stored in the model file, so gain "
                "importance is zero for a model read back with load(); "
                "pickle the estimator instead to keep it",
                UserWarning,
                stacklevel=2,
            )
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
                return _mojoboost.load_multiclass(path)
            return _mojoboost.load(path)

    def __getstate__(self):
        """Pickle support. The trained model is an opaque handle owned by
        the extension module, so it travels as the bytes of the same
        versioned text format `save()` writes; everything else is ordinary
        Python state and pickles as it is."""
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        state["_model_blob"] = None if model is None else self._model_bytes()
        return state

    def __setstate__(self, state):
        state = dict(state)
        blob = state.pop("_model_blob", None)
        self.__dict__.update(state)
        self._model = None if blob is None else self._model_from_bytes(blob)


class MojoBoostRegressor(_Base):
    """Objective names follow LightGBM: "regression" (squared error),
    "huber", "quantile", and "mae" (alias "regression_l1"). `alpha` is the
    quantile level for "quantile" and the transition point for "huber";
    the other objectives ignore it.

    `objective` may instead be a callable `f(raw, y) -> (grad, hess)`, called
    once per boosting round with the current raw predictions and the labels.
    See `_fit_custom` for the callback contract and
    src/mojoboost/objective.mojo for how it differs from LightGBM's; the
    short version is that the trainer applies `sample_weight` for you, that
    it validates the returned arrays, and that `predict` then returns raw
    scores because the inverse link is yours to apply. `base_score` is the
    starting raw score for a custom objective (a number, or "mean" for the
    weighted label mean); built-in objectives ignore it and derive their
    own."""

    # scikit-learn before 1.6 dispatches on this; 1.6 and later read
    # __sklearn_tags__ below. Both are cheap, so both are here.
    _estimator_type = "regressor"

    _OBJECTIVES = {
        "regression": _SQUARED_ERROR,
        "huber": _HUBER,
        "quantile": _QUANTILE,
        "mae": _L1,
        "regression_l1": _L1,
    }

    def __init__(
        self, objective="regression", alpha=0.9, base_score=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.objective = objective
        self.alpha = alpha
        self.base_score = base_score

    def _objective_code(self):
        if callable(self.objective):
            return _CUSTOM
        code = self._OBJECTIVES.get(self.objective)
        if code is None:
            raise ValueError(
                f"unknown objective {self.objective!r}; expected one of "
                + ", ".join(sorted(self._OBJECTIVES))
            )
        alpha = float(self.alpha)
        if code == _HUBER and alpha <= 0.0:
            raise ValueError("huber requires alpha > 0")
        if code == _QUANTILE and not 0.0 < alpha < 1.0:
            raise ValueError("quantile requires 0 < alpha < 1")
        return code

    def fit(self, X, y, sample_weight=None):
        """Fit on `X` (n_samples, n_features) and a numeric target `y`.

        `X` may contain NaN, which is the missing-value marker, but not
        infinities; `y` and `sample_weight` must be finite throughout.
        Returns self.
        """
        objective = self._objective_code()
        self._reset_fitted()
        Xb, n_rows, n_features, names = _arrays.check_X(X)
        yb = _arrays.check_target(y, n_rows)
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        device = self._resolve_device(n_rows, n_features, 1)
        params = self._params(w_addr, device, ic_flat, ic_offsets)
        if objective == _CUSTOM:
            self._fit_custom(Xb, yb, wb, n_rows, n_features, params, device)
        else:
            self._model = _mojoboost.fit(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                objective,
                params,
            )
        self._record_fit(n_features, names, device)
        return self

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

    def _fit_custom(self, Xb, yb, wb, n_rows, n_features, params, device):
        """Train against a Python objective callback.

        The callback is called once per boosting round with the current raw
        predictions and the labels and returns `(grad, hess)`, each of
        length n_rows. Both arguments are views on live buffers that the
        trainer reuses every round, so read them, do not keep them.

        This is the Python-callback path, whose per-round cost is measured
        in bench/bench_custom_objective.py. For a hot inner loop use the
        native Mojo interface in src/mojoboost/objective.mojo, which
        specializes on the callable and pays nothing per round.
        """
        if device != "cpu":
            raise RuntimeError(
                "custom objectives train on the CPU; use device='cpu' or "
                "device='auto'"
            )
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
        self._model = _mojoboost.fit_custom(
            _addr(Xb), n_rows, n_features, _addr(yb), bridge, params
        )

    def predict(self, X):
        """Predictions for `X`, one per row."""
        Xb, n_rows = self._check_predict_X(X)
        out = _out_buffer(n_rows)
        _mojoboost.predict(
            self._model, _addr(Xb), n_rows, self.n_features_in_, _addr(out)
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
        """Write the fitted model to `path` in mojoboost's versioned text
        format. This stores the model, not the estimator: hyperparameters,
        feature names, and split gains do not travel with it. Pickle the
        estimator to keep those."""
        self._require_fitted()
        _mojoboost.save(self._model, str(path))

    @classmethod
    def load(cls, path):
        """Load a saved model into a fresh estimator.

        What comes back predicts exactly as the saved model did, and
        reports `n_features_in_` and `best_iteration_`. What does not come
        back is everything the file never held: the training device (the
        ensemble is the same either way, so there is no `device_`),
        constructor hyperparameters, feature names, and split gains. Use
        pickle when you want the whole estimator.
        """
        est = cls()
        est._model = _mojoboost.load(str(path))
        est.n_features_in_ = int(_mojoboost.n_features(est._model))
        est.best_iteration_ = est._num_iterations()
        return est

    def __sklearn_tags__(self):
        return _estimator_tags("regressor")


class MojoBoostClassifier(_Base):
    """Binary (logistic) for 2 classes, softmax for more. Multiclass
    training is CPU-only, so `device="gpu"` raises for 3 or more classes and
    `device="auto"` resolves to the CPU.

    Labels may be of any single comparable type, as in scikit-learn: they
    are sorted, recorded on `classes_`, and encoded to the 0..n_classes-1
    the trainer needs. `predict` returns labels from `classes_`, not the
    internal codes. (mojoboost used to require the codes themselves; this
    is a deliberate widening, and passing 0..n_classes-1 still behaves
    exactly as it did.)

    The classifier takes no `objective`: custom objectives are single-output
    only, so pass yours to `MojoBoostRegressor` and apply your own link to
    its raw predictions. `objective=` is accepted here solely to raise that
    message rather than a bare TypeError."""

    # See the note on MojoBoostRegressor._estimator_type.
    _estimator_type = "classifier"

    def __init__(self, objective=None, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective

    def fit(self, X, y, sample_weight=None):
        """Fit on `X` (n_samples, n_features) and labels `y`.

        `X` may contain NaN, the missing-value marker, but not infinities.
        `y` needs at least 2 distinct labels, and `sample_weight` must be
        finite and nonnegative. Returns self.
        """
        if self.objective is not None:
            raise ValueError(
                "MojoBoostClassifier takes no objective; custom objectives "
                "are single-output only. Use MojoBoostRegressor with your "
                "objective and apply your own link (a sigmoid, say) to the "
                "raw predictions."
            )
        self._reset_fitted()
        Xb, n_rows, n_features, names = _arrays.check_X(X)
        yb, classes = _arrays.encode_labels(y, n_rows)
        n_classes = len(classes)
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        # Binary is single-output (one tree per round), so it has a GPU
        # path; only the softmax ensemble is CPU-only.
        device = self._resolve_device(
            n_rows, n_features, 1 if n_classes == 2 else n_classes
        )
        self._multiclass = n_classes > 2
        if n_classes == 2:
            self._model = _mojoboost.fit(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                _BINARY_LOGISTIC,
                self._params(w_addr, device, ic_flat, ic_offsets),
            )
        else:
            self._model = _mojoboost.fit_multiclass(
                _addr(Xb),
                n_rows,
                n_features,
                _addr(yb),
                n_classes,
                self._params(w_addr, device, ic_flat, ic_offsets),
            )
        self.classes_ = (
            _np.asarray(classes) if _np is not None else list(classes)
        )
        self.n_classes_ = n_classes
        self._record_fit(n_features, names, device)
        return self

    def predict_proba(self, X):
        """Class probabilities, shape (n_samples, n_classes), with columns
        in `classes_` order. Rows sum to 1."""
        Xb, n_rows = self._check_predict_X(X)
        n_features = self.n_features_in_
        if self.n_classes_ == 2:
            out = _out_buffer(n_rows)
            _mojoboost.predict(
                self._model, _addr(Xb), n_rows, n_features, _addr(out)
            )
            if _np is not None:
                return _np.column_stack([1.0 - out, out])
            return [[1.0 - p, p] for p in out]
        out = _out_buffer(n_rows * self.n_classes_)
        _mojoboost.predict_proba(
            self._model, _addr(Xb), n_rows, n_features, _addr(out)
        )
        if _np is not None:
            return out.reshape(n_rows, self.n_classes_)
        k = self.n_classes_
        return [list(out[r * k : (r + 1) * k]) for r in range(n_rows)]

    def predict(self, X):
        """Predicted labels, drawn from `classes_`. Defined as the argmax
        of `predict_proba`, so the two can never disagree."""
        proba = self.predict_proba(X)
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
        if self._multiclass:
            _mojoboost.save_multiclass(self._model, str(path))
        else:
            _mojoboost.save(self._model, str(path))

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
            est._model = _mojoboost.load(str(path))
            est.n_classes_ = 2
            est._multiclass = False
            est.n_features_in_ = int(_mojoboost.n_features(est._model))
        except Exception:
            est._model = _mojoboost.load_multiclass(str(path))
            est.n_classes_ = int(_mojoboost.n_classes(est._model))
            est._multiclass = True
            est.n_features_in_ = int(
                _mojoboost.n_features_multiclass(est._model)
            )
        est.classes_ = (
            _np.arange(est.n_classes_)
            if _np is not None
            else list(range(est.n_classes_))
        )
        est.best_iteration_ = est._num_iterations()
        return est

    def __sklearn_tags__(self):
        return _estimator_tags("classifier")


def group_from_query_ids(query_ids):
    """LightGBM's `group` array (per-query row counts) from a per-row query
    id column.

    Ids may be anything hashable-by-equality and need not be sorted, but
    each query's rows must be consecutive: an id that reappears after a
    different one raises, because splitting a query in two would change
    every NDCG it takes part in.
    """
    ids = list(query_ids)
    if not ids:
        raise ValueError("query_ids must not be empty")
    counts = []
    seen = []
    for i, qid in enumerate(ids):
        if i and qid == ids[i - 1]:
            counts[-1] += 1
            continue
        if qid in seen:
            raise ValueError(
                f"rows of query {qid!r} are not consecutive; a query's rows "
                "must form one unbroken run"
            )
        seen.append(qid)
        counts.append(1)
    return counts


def _group_buffer(group, n_rows):
    """Validated float64 buffer of per-query row counts. Must stay
    referenced while its address is in use."""
    if group is None:
        raise ValueError(
            "a ranker needs `group`: the number of rows in each query, in "
            "row order (LightGBM's `group` parameter)"
        )
    try:
        values = list(group)
    except TypeError:
        raise ValueError("group must be a sequence of row counts") from None
    if not values:
        raise ValueError("group must contain at least one query")
    counts = []
    for value in values:
        count = float(value)
        if count != int(count):
            raise ValueError(f"group counts must be integers, got {value!r}")
        if count <= 0:
            raise ValueError(f"group counts must be positive, got {value!r}")
        counts.append(count)
    if int(sum(counts)) != n_rows:
        raise ValueError(
            f"group counts sum to {int(sum(counts))} but X has {n_rows} rows"
        )
    return _as_f64_vector(counts, len(counts), "group")


def _check_relevance(yb, n_rows):
    """Relevance labels must be integers in [0, 30], LightGBM's default
    `label_gain` range."""
    if _np is not None:
        arr = _np.asarray(yb)
        bad = (
            bool((arr < 0).any())
            or bool((arr > _MAX_RELEVANCE_LABEL).any())
            or not bool(_np.array_equal(arr, _np.floor(arr)))
        )
    else:
        bad = any(
            v < 0 or v > _MAX_RELEVANCE_LABEL or v != int(v) for v in yb
        )
    if bad:
        raise ValueError(
            "relevance labels must be integers in "
            f"[0, {_MAX_RELEVANCE_LABEL}]"
        )


def ndcg_score(scores, y, group, at=5):
    """Mean NDCG@`at` of `scores` against relevance labels `y`, averaged
    over the queries `group` describes.

    Documents are ranked within their own query and never across queries.
    A query whose labels are all 0 counts as 1.0, which is what LightGBM's
    ndcg metric does with a query that has no attainable DCG.
    """
    n_rows = len(scores)
    sb = _as_f64_vector(scores, n_rows, "scores")
    yb = _as_f64_vector(y, n_rows)
    _check_relevance(yb, n_rows)
    gb = _group_buffer(group, n_rows)
    at = int(at)
    if at < 1:
        raise ValueError("at must be positive")
    return float(
        _mojoboost.ndcg(
            _addr(sb),
            _addr(yb),
            n_rows,
            at,
            {"group_addr": _addr(gb), "n_groups": len(gb)},
        )
    )


class MojoBoostRanker(_Base):
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
    src/mojoboost/ranking.mojo for the objective and its documented
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambdarank_truncation_level = lambdarank_truncation_level
        self.sigmoid = sigmoid
        self.lambdarank_norm = lambdarank_norm
        self.ndcg_eval_at = ndcg_eval_at

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
        return params

    def fit(self, X, y, group=None, sample_weight=None):
        """Fit on `X` (n_samples, n_features), relevance labels `y`, and
        `group`, the row count of each query in row order. Returns self."""
        self._reset_fitted()
        Xb, n_rows, n_features, names = _arrays.check_X(X)
        yb = _arrays.check_target(y, n_rows)
        _check_relevance(yb, n_rows)
        gb = _group_buffer(group, n_rows)
        wb, w_addr = self._weight_buffer(sample_weight, n_rows)
        ic_flat, ic_offsets = self._interaction_buffers(n_features)
        device = self._resolve_device(n_rows, n_features, 1)
        if device != "cpu":
            raise RuntimeError(
                "lambdarank trains on the CPU; use device='cpu' or "
                "device='auto'"
            )
        params = self._rank_params(
            self._params(w_addr, device, ic_flat, ic_offsets), gb
        )
        self._model = _mojoboost.fit_ranker(
            _addr(Xb), n_rows, n_features, _addr(yb), params
        )
        self._record_fit(n_features, names, device)
        return self

    def predict(self, X):
        """Raw ranking scores for `X`, one per row. Sort a query's rows by
        this score, descending, to get its ranking; the values themselves
        are not comparable between queries."""
        Xb, n_rows = self._check_predict_X(X)
        out = _out_buffer(n_rows)
        _mojoboost.predict(
            self._model, _addr(Xb), n_rows, self.n_features_in_, _addr(out)
        )
        return _finish(out)

    def score(self, X, y, group=None, sample_weight=None):
        """Mean NDCG@`ndcg_eval_at` of this model's ranking of `X`.
        `sample_weight` is accepted for signature compatibility and
        ignored, as a weighted NDCG has no LightGBM definition to match."""
        return ndcg_score(self.predict(X), y, group, self.ndcg_eval_at)

    def save(self, path):
        """Write the fitted model to `path` in mojoboost's versioned text
        format. Query boundaries are training data, not model state, so
        they do not travel with it."""
        self._require_fitted()
        _mojoboost.save(self._model, str(path))

    @classmethod
    def load(cls, path):
        """Load a saved ranker into a fresh estimator."""
        est = cls()
        est._model = _mojoboost.load(str(path))
        est.n_features_in_ = int(_mojoboost.n_features(est._model))
        est.best_iteration_ = est._num_iterations()
        return est
