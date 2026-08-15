"""LightGBM's functional API: `Dataset`, `Booster`, and `train`.

The scikit-learn estimators in this package are one way to reach mojotrees.
This module is the other, and the one LightGBM users reach for when they
want the model object itself:

    import mojotrees as mb

    train_set = mb.Dataset(X, label=y)
    booster = mb.train({"objective": "regression", "num_leaves": 31},
                       train_set, num_boost_round=100)
    booster.predict(X_test)

`Dataset` owns the training data. Binning is the expensive part of starting
a run, so a dataset is binned once (`construct()`, or the first `train` that
uses it) and every later run on it reuses those bins. It holds the label,
weights, query groups, init scores, feature names, and the categorical
declaration; each is validated against the row count when the dataset is
built, not when training fails.

`Booster` owns a fitted model: prediction, evaluation, feature importance,
model IO, the iteration count, and continued training through `update()`.
The estimators hold one of these too, on `booster_`, so there is a single
model object in this package rather than one per API.

Differences from LightGBM, all deliberate
-----------------------------------------

- **No post-construction mutators.** LightGBM lets you `set_label`,
  `set_field`, `set_categorical_feature`, and so on after a Dataset exists.
  Bin edges are fitted from the data and from the categorical declaration,
  so changing either afterwards would leave the binned matrix describing
  data the dataset no longer holds. Every field is a constructor argument,
  and a dataset is immutable once constructed.
- **`free_raw_data` defaults to False.** LightGBM frees the raw matrix on
  construction. mojotrees evaluates a dataset by predicting through the
  model rather than by reading an internal score buffer, so `eval_train()`
  needs the raw matrix; keeping it is the default and freeing it is opt-in.
- **`reference` is checked, not required.** A validation Dataset in
  LightGBM must reference the training one to share its bin mappers.
  mojotrees predicts validation rows through the model's own mapper, so a
  reference changes nothing; it is accepted, and its binning parameters are
  checked against this one, so passing a mismatched reference reports the
  mismatch instead of being ignored.
- **`update()` does not cover ranking.** Continued training resumes from
  the raw scores the existing trees produce (see `boosting.train_more` and
  `train_multiclass_more`), which covers the single-output objectives and
  softmax multiclass. LambdaRank gradients are computed within a query from
  state the fitted ensemble does not carry, so a ranking booster says so
  rather than appending trees that would be wrong.
- **`train()` trains, it does not report.** `valid_sets` are registered on
  the returned booster for `eval_valid()`, but there is no per-round
  history and no early stopping here yet: those live on the estimators'
  `fit(eval_set=..., early_stopping_rounds=...)`.
- **`dump_model` and `trees_to_dataframe` report what mojotrees records.**
  Both are here, and both delegate to `mojotrees.inspection`, which is the
  one implementation of the schema. One column cannot be filled the way
  LightGBM fills it: `weight` is always None, because mojotrees records a
  node's training row cover (in `count`) and not the hessian sum LightGBM
  calls weight. `split_gain` is filled, and travels with the model from
  format v4 on, so only a model read from a file written before that
  reports None and says so through `has_split_gain`.
  `docs/MODEL_INSPECTION_SCHEMA.md` has the rest; `model_to_string()`
  still gives the whole model in mojotrees's versioned text format.

See docs/LIGHTGBM_PARITY.md for the full row-by-row statement.
"""

import os as _os
import tempfile as _tempfile

from . import _arrays, _eval, _mojotrees

_np = _arrays.np
_addr = _arrays.addr
_out_buffer = _arrays.out_buffer
_finish = _arrays.finish

__all__ = ["Booster", "Dataset", "train"]

#: LightGBM's aliases for the number of boosting iterations. `n_estimators`
#: is the scikit-learn spelling the estimators in this package use.
_ROUND_ALIASES = (
    "num_iterations",
    "num_iteration",
    "num_round",
    "num_rounds",
    "num_boost_round",
    "n_estimators",
)

#: Parameters that describe the data rather than the training run. They are
#: `Dataset` arguments here, and naming one in `train(params=...)` says so
#: instead of being ignored.
_DATASET_PARAMS = {
    "max_bin": "Dataset(params={'max_bin': ...})",
    "max_bins": "Dataset(params={'max_bin': ...})",
    "use_missing": "Dataset(params={'use_missing': ...})",
    "categorical_feature": "Dataset(categorical_feature=...)",
    "categorical_features": "Dataset(categorical_feature=...)",
    "feature_name": "Dataset(feature_name=...)",
    "label": "Dataset(label=...)",
    "weight": "Dataset(weight=...)",
    "group": "Dataset(group=...)",
    "init_score": "Dataset(init_score=...)",
}

#: Binning parameters a `Dataset` accepts in its `params` dict, with their
#: defaults. Everything else belongs to `train`.
_BINNING_DEFAULTS = {"max_bin": 255, "use_missing": True}

_IMPORTANCE_TYPES = {"split": 0, "gain": 1}


def _objectives():
    """The regression objective table the estimators validate against, read
    lazily so this module tracks whatever the regressor supports."""
    from . import MojoTreesRegressor

    return MojoTreesRegressor._OBJECTIVES


def _task_of(objective):
    """The task an objective name belongs to, in `_eval`'s vocabulary."""
    if callable(objective):
        raise ValueError(
            "train() takes built-in objectives only; a Python objective "
            "callback trains through MojoTreesRegressor(objective=f)"
        )
    name = str(objective)
    if name in ("binary", "binary_logloss"):
        return _eval.BINARY
    if name in ("multiclass", "softmax", "multiclassova", "multiclass_ova"):
        if name in ("multiclassova", "multiclass_ova"):
            raise ValueError(
                "objective 'multiclassova' (one-vs-all) is not implemented; "
                "use 'multiclass', which is LightGBM's softmax"
            )
        return _eval.MULTICLASS
    if name in ("lambdarank", "rank_xendcg", "rank_xendcg_norm"):
        if name != "lambdarank":
            raise ValueError(
                f"objective {name!r} is not implemented; 'lambdarank' is the "
                "ranking objective mojotrees trains"
            )
        return _eval.RANKING
    if name in _objectives():
        return _eval.REGRESSION
    raise ValueError(
        f"unknown objective {objective!r}; expected one of binary, "
        "lambdarank, multiclass, " + ", ".join(sorted(_objectives()))
    )


def _estimator_task(estimator, multiclass):
    """The task an estimator's model belongs to, which decides which metrics
    can score it. The classifier declares itself a classifier and the
    regressor a regressor, in scikit-learn's own `_estimator_type`; the
    ranker is neither, which is what identifies it."""
    if multiclass:
        return _eval.MULTICLASS
    kind = getattr(estimator, "_estimator_type", None)
    if kind == "classifier":
        return _eval.BINARY
    if kind == "regressor":
        return _eval.REGRESSION
    return _eval.RANKING


class _Config:
    """A LightGBM parameter dict as the hyperparameters the estimators
    already validate.

    Every parameter here is resolved by `_Base`, so `train()` and the
    estimators cannot drift apart on aliases, ranges, or defaults: this
    holds a `_Base` and reads the same `_params()` off it. Keys LightGBM
    spells differently from the constructor are mapped, keys that describe
    the data are rejected with the `Dataset` argument they belong to, and
    anything the constructor does not accept is reported with the list that
    it does.
    """

    def __init__(self, params, num_boost_round=None):
        from . import (
            MojoTreesClassifier,
            MojoTreesRanker,
            MojoTreesRegressor,
            _BINARY_LOGISTIC,
            _LAMBDARANK,
        )

        given = dict(params or {})
        for key in list(given):
            if key in _DATASET_PARAMS:
                raise ValueError(
                    f"{key!r} describes the data, not the training run; pass "
                    f"it as {_DATASET_PARAMS[key]}"
                )
        objective = given.pop("objective", "regression")
        self.task = _task_of(objective)
        self.objective = objective
        self.n_classes = self._class_count(given)
        self.rounds = self._rounds(given, num_boost_round)

        # The estimator of the matching task holds the configuration, so
        # every alias, range check, and default is resolved by the code the
        # estimators use rather than by a second copy of it here.
        try:
            if self.task == _eval.REGRESSION:
                self.base = MojoTreesRegressor(
                    objective=objective, n_estimators=self.rounds, **given
                )
                self.objective_code = self.base._objective_code()
            elif self.task == _eval.RANKING:
                self.base = MojoTreesRanker(
                    n_estimators=self.rounds, **given
                )
                self.objective_code = _LAMBDARANK
            else:
                self.base = MojoTreesClassifier(
                    n_estimators=self.rounds, **given
                )
                self.objective_code = _BINARY_LOGISTIC
        except TypeError as exc:
            raise ValueError(
                f"unknown or unsupported training parameter: {exc}"
            ) from None
        # The objective's scalar parameter, whichever of alpha, fair_c, and
        # tweedie_variance_power the objective reads; see
        # MojoTreesRegressor._objective_param.
        resolve_alpha = getattr(self.base, "_objective_param", None)
        self.alpha = (
            float(getattr(self.base, "alpha", 0.9))
            if resolve_alpha is None
            else float(resolve_alpha())
        )
        self.ndcg_at = int(getattr(self.base, "ndcg_eval_at", 5))

    @staticmethod
    def _class_count(given):
        num_class = given.pop("num_class", given.pop("num_classes", None))
        if num_class is None:
            return 0
        count = int(num_class)
        if count < 2:
            raise ValueError("num_class must be at least 2")
        return count

    @staticmethod
    def _rounds(given, num_boost_round):
        """The number of boosting rounds, from `num_boost_round` or from any
        of LightGBM's aliases in the parameter dict. Two that disagree raise
        rather than one silently winning."""
        named = None
        for key in _ROUND_ALIASES:
            if key not in given:
                continue
            value = int(given.pop(key))
            if named is not None and value != named:
                raise ValueError(
                    "the boosting-round parameters in params disagree; set "
                    "only one of " + ", ".join(_ROUND_ALIASES)
                )
            named = value
        if named is None:
            return 100 if num_boost_round is None else int(num_boost_round)
        if num_boost_round is not None and int(num_boost_round) != named:
            raise ValueError(
                f"num_boost_round={num_boost_round} and the boosting-round "
                f"parameter {named} in params disagree; set only one"
            )
        return named

    def check_dataset(self, dataset):
        """Reject a dataset the configured objective cannot train on."""
        if dataset.get_label() is None:
            raise ValueError("train() needs a Dataset with a label")
        if self.task == _eval.RANKING and dataset.get_group() is None:
            raise ValueError(
                "objective='lambdarank' needs a Dataset with group: the "
                "number of rows in each query, in row order"
            )
        if self.task == _eval.MULTICLASS and self.n_classes == 0:
            raise ValueError(
                "objective='multiclass' needs num_class in params"
            )
        if self.task == _eval.BINARY:
            labels = dataset.get_label()
            if _np is not None:
                arr = _np.asarray(labels)
                bad = arr[(arr != 0.0) & (arr != 1.0)]
                first = None if bad.size == 0 else bad[0]
            else:
                offenders = [v for v in labels if v not in (0.0, 1.0)]
                first = offenders[0] if offenders else None
            if first is not None:
                raise ValueError(
                    "objective='binary' needs labels in {0, 1}; got "
                    f"{first!r}"
                )

    def binding_params(self, dataset, rounds):
        """The low-level parameter dict the extension module reads, for a run
        of `rounds` boosting rounds on `dataset`.

        The buffers whose addresses this dict holds are returned with it and
        must stay referenced for the duration of the call.
        """
        n_features = dataset.num_feature()
        keep = []
        ic_flat, ic_offsets = self.base._interaction_buffers(n_features)
        mono_buf, mono_addr = self.base._monotone_buffer(n_features)
        contri_buf, contri_addr = self.base._feature_contri_buffer(n_features)
        keep.extend([ic_flat, ic_offsets, mono_buf, contri_buf])
        device = self.base._resolve_device(
            dataset.num_data(),
            n_features,
            self.n_classes if self.task == _eval.MULTICLASS else 1,
        )
        if getattr(dataset, "is_sparse", False):
            # The GPU trainer reads a dense binned matrix, so a sparse
            # dataset resolves to the CPU rather than being densified. An
            # explicit device='gpu' is a request that cannot be served and
            # is refused here, where the caller's own word for it is still
            # in hand; 'auto' would have picked the CPU anyway on anything
            # this path can run.
            if getattr(self.base, "device", "auto") == "gpu":
                raise RuntimeError(
                    "sparse input trains on the CPU; there is no sparse GPU "
                    "kernel yet. Use device='cpu' or device='auto', or "
                    "densify with .toarray() to train on the GPU."
                )
            device = "cpu"
        params = self.base._params(
            0, device, ic_flat, ic_offsets, mono_addr, None, contri_addr
        )
        params["n_estimators"] = int(rounds)
        params["objective"] = int(self.objective_code)
        params["n_classes"] = int(self.n_classes)
        if self.task == _eval.RANKING:
            # The ranker validates its own parameters here; the group it
            # writes is the dataset's, which the trainer reads from the
            # dataset itself.
            params = self.base._rank_params(params, dataset._group)
        return params, keep, device


class Dataset:
    """Training data, binned once and reused.

    `data` is a feature matrix in any layout the estimators accept, and
    every other column is optional:

    - `label`, the target. Training needs one; a dataset built without one
      can still be predicted on and describes itself.
    - `weight`, one nonnegative sample weight per row.
    - `group`, LightGBM's per-query row counts for ranking, in row order.
    - `init_score`, one raw-score offset per row to boost from instead of
      the objective's own base score. The offset is training state, not
      model state: the fitted model predicts the trees alone, so scoring
      new data means adding your offset back.
    - `feature_name`, one name per feature. A pandas frame supplies its
      column names when this is not given.
    - `categorical_feature`, the indices (or names, with a named frame) of
      integer-coded categorical features.

    `params` takes the two parameters that change the binning itself,
    `max_bin` and `use_missing`; everything else is a `train()` parameter.

    Binning happens on `construct()`, or on the first `train()` that uses
    the dataset. Until then this is validated data and nothing more.

    Sparse data
    -----------
    A SciPy sparse matrix stays sparse: it is binned as CSC and trained on
    as a `SparseBinnedMatrix`, so nothing ever allocates
    `n_rows * n_features`. The paths that have no sparse implementation say
    so rather than densifying: `device='gpu'`, `objective='lambdarank'`, and
    continued training through `Booster.update`.

    `reference` and `keep_raw`
    --------------------------
    `reference=` bins this data with **that dataset's** fitted mapper
    instead of fitting new edges. A bin index then means what it means in
    the reference, which is what a validation set needs, since it will be
    scored by a model trained on the reference, and what continued training
    requires. It is the wrong argument for a cross-validation fold, whose
    held-out rows must not have shaped the binning; `subset()` without
    `shared_binning` is that one.

    `keep_raw=True` retains the raw matrix inside the *native* dataset,
    which is what `subset()` needs, since bins cannot be refitted from
    bins. It is not `free_raw_data`, which controls the Python-side
    reference to the matrix you passed in.
    """

    def __init__(
        self,
        data,
        label=None,
        weight=None,
        group=None,
        init_score=None,
        feature_name=None,
        categorical_feature=None,
        params=None,
        reference=None,
        free_raw_data=False,
        keep_raw=False,
    ):
        self.params = self._binning_params(params)
        self.free_raw_data = bool(free_raw_data)
        self.keep_raw = bool(keep_raw)
        self._handle = None

        # Sparse input takes the sparse binner, which reads the three CSC
        # arrays directly; the dense entry point refuses it rather than
        # densifying behind the caller (see `_arrays.column_major`).
        self._sparse = _arrays.is_sparse(data)
        if self._sparse:
            buffers, n_rows, n_features, frame_names = _arrays.check_X_sparse(
                data, "csc"
            )
            self._x = buffers
        else:
            Xb, n_rows, n_features, frame_names = _arrays.check_X(data)
            self._x = Xb
        self._n_rows = n_rows
        self._n_features = n_features
        self._raw = data

        self._names = self._feature_names(
            feature_name, frame_names, n_features
        )
        self._categorical = self._categorical_indices(
            categorical_feature, self._names, n_features
        )
        self._label = (
            None if label is None else _arrays.check_target(label, n_rows)
        )
        self._weight = (
            None
            if weight is None
            else _arrays.check_sample_weight(weight, n_rows)
        )
        self._group = None if group is None else self._group_buffer(group)
        self._init_score = (
            None
            if init_score is None
            else _arrays.check_target(init_score, n_rows, "init_score")
        )
        self.reference = reference
        if reference is not None:
            self._check_reference(reference)

    # -- construction ----------------------------------------------------

    @staticmethod
    def _binning_params(params):
        """The binning parameters, defaulted. Anything else in `params`
        belongs to `train` and says so."""
        given = dict(params or {})
        out = dict(_BINNING_DEFAULTS)
        for key in list(given):
            if key in out:
                out[key] = given.pop(key)
        if given:
            raise ValueError(
                f"a Dataset takes only {', '.join(sorted(_BINNING_DEFAULTS))}"
                f" in params; {sorted(given)} belong to train()"
            )
        if int(out["max_bin"]) < 2:
            raise ValueError("max_bin must be at least 2")
        return {
            "max_bin": int(out["max_bin"]),
            "use_missing": bool(out["use_missing"]),
        }

    @staticmethod
    def _feature_names(feature_name, frame_names, n_features):
        if feature_name is None or (
            isinstance(feature_name, str) and feature_name == "auto"
        ):
            return None if frame_names is None else list(frame_names)
        names = [str(name) for name in feature_name]
        if len(names) != n_features:
            raise ValueError(
                f"feature_name has {len(names)} names but the data has "
                f"{n_features} features"
            )
        if len(set(names)) != len(names):
            raise ValueError("feature_name entries must be unique")
        return names

    @staticmethod
    def _categorical_indices(categorical_feature, names, n_features):
        """Categorical features as validated indices. Names are accepted
        when the dataset has feature names to resolve them against."""
        if categorical_feature is None or (
            isinstance(categorical_feature, str)
            and categorical_feature == "auto"
        ):
            return []
        if isinstance(categorical_feature, (str, bytes)):
            raise ValueError(
                "categorical_feature must be a sequence of feature indices "
                "or names, not a single string"
            )
        out = []
        for entry in categorical_feature:
            if isinstance(entry, str):
                if names is None:
                    raise ValueError(
                        f"categorical_feature {entry!r} is a name, but this "
                        "dataset has no feature names to resolve it against"
                    )
                if entry not in names:
                    raise ValueError(
                        f"categorical_feature {entry!r} is not one of the "
                        "feature names"
                    )
                index = names.index(entry)
            else:
                index = int(entry)
                if index != float(entry):
                    raise ValueError(
                        "categorical_feature entries must be whole feature "
                        f"indices, got {entry!r}"
                    )
            if not 0 <= index < n_features:
                raise ValueError(
                    f"categorical_feature index {index} is out of range for "
                    f"{n_features} features"
                )
            if index in out:
                raise ValueError(
                    f"categorical_feature index {index} is listed twice"
                )
            out.append(index)
        return out

    def _group_buffer(self, group):
        counts = list(group)
        if not counts:
            raise ValueError("group must contain at least one query")
        total = 0
        for value in counts:
            count = float(value)
            if count != int(count):
                raise ValueError(
                    f"group counts must be integers, got {value!r}"
                )
            if count <= 0:
                raise ValueError(
                    f"group counts must be positive, got {value!r}"
                )
            total += int(count)
        if total != self._n_rows:
            raise ValueError(
                f"group counts sum to {total} but the data has "
                f"{self._n_rows} rows"
            )
        return _arrays.f64_vector(
            [float(c) for c in counts], len(counts), "group"
        )

    def _check_reference(self, reference):
        """A reference dataset supplies this one's binning, so a mismatched
        one is refused here rather than producing bin indices that mean two
        different things.

        The binning parameters, the feature names, and the categorical
        declaration come from the reference on `construct()`, because they
        describe the columns rather than the rows. Ones passed here that
        disagree with it are reported instead of being silently overruled.
        """
        if not isinstance(reference, Dataset):
            raise TypeError("reference must be a Dataset")
        if reference.num_feature() != self._n_features:
            raise ValueError(
                f"reference has {reference.num_feature()} features but this "
                f"dataset has {self._n_features}"
            )
        if reference.params != self.params:
            raise ValueError(
                f"reference was binned with {reference.params} but this "
                f"dataset uses {self.params}; this dataset is binned by the "
                "reference's fitted mapper, so the two must agree on the "
                "binning parameters"
            )
        if self._categorical and self._categorical != reference._categorical:
            raise ValueError(
                "a dataset built from a reference takes the reference's "
                "categorical declaration, because the binning is the "
                f"reference's; it declares {reference._categorical} and this "
                f"one was given {self._categorical}"
            )
        if self._sparse:
            raise NotImplementedError(
                "reference binning takes a dense matrix: the binding reads "
                "the rows as a dense buffer. Densify with .toarray(), or "
                "build the dataset without reference= and accept that it "
                "bins itself"
            )

    def construct(self):
        """Bin the data, if it has not been binned already. Returns self.

        Three constructors live behind this, and which one runs is decided
        by what the dataset was built from rather than by a flag:
        `reference=` bins by that dataset's fitted mapper, a sparse matrix
        bins to a sparse binned matrix, and everything else takes the dense
        path. All three are the same `trainset.Dataset` afterwards.

        This is where the raw matrix is dropped when `free_raw_data=True`.
        """
        if self._handle is not None:
            return self
        if self._x is None:
            raise ValueError(
                "this Dataset's raw data was freed before it was "
                "constructed, so there is nothing left to bin"
            )
        cat = (
            None
            if not self._categorical
            else _arrays.f64_vector(
                [float(f) for f in self._categorical],
                len(self._categorical),
                "categorical_feature",
            )
        )
        params = {
            "label_addr": 0 if self._label is None else _addr(self._label),
            "weight_addr": 0 if self._weight is None else _addr(self._weight),
            "init_score_addr": (
                0 if self._init_score is None else _addr(self._init_score)
            ),
            "group_addr": 0 if self._group is None else _addr(self._group),
            "n_groups": 0 if self._group is None else len(self._group),
            "categorical_addr": 0 if cat is None else _addr(cat),
            "categorical_len": 0 if cat is None else len(cat),
            "max_bin": int(self.params["max_bin"]),
            "use_missing": int(bool(self.params["use_missing"])),
            "feature_names": [] if self._names is None else list(self._names),
            "n_names": 0 if self._names is None else len(self._names),
            "keep_raw": int(self.keep_raw),
        }
        if self.reference is not None:
            # The reference's mapper, its names, its categorical
            # declaration, and its binning parameters. Only the rows and
            # the per-row columns are this dataset's, so the keys above
            # that describe the columns are ignored on this path.
            self._handle = _mojotrees.dataset_create_reference(
                self.reference._constructed(),
                _addr(self._x),
                self._n_rows,
                params,
            )
        elif self._sparse:
            params.update(self._x.params())
            self._handle = _mojotrees.dataset_create_csc(params)
        else:
            self._handle = _mojotrees.dataset_create(
                _addr(self._x), self._n_rows, self._n_features, params
            )
        # `cat` had to outlive the call above.
        del cat
        if self.free_raw_data:
            self._x = None
            self._raw = None
        return self

    def _constructed(self):
        return self.construct()._handle

    # -- subsets and prepared tables --------------------------------------

    @staticmethod
    def _field_list(handle, field):
        """One of a constructed dataset's optional columns, read from the
        native side. Empty means the dataset has none."""
        return [float(v) for v in _mojotrees.dataset_field(handle, field)]

    @staticmethod
    def _field_buffer(handle, field):
        """`_field_list` as the float64 buffer this class keeps its columns
        in, or None. `Booster.eval` passes their addresses, so a dataset
        this process did not build has to hold real buffers rather than
        lists."""
        values = Dataset._field_list(handle, field)
        if not values:
            return None
        return _arrays.f64_vector(values, len(values), field)

    @classmethod
    def _from_handle(cls, handle):
        """A `Dataset` around a native handle this process did not bin.

        Everything it answers comes from the native side, because nothing
        else can: the shape, the binning parameters, the names, the
        categorical declaration, and the four optional columns. It holds no
        raw matrix, so it can be trained on and described but not predicted
        on; `Booster.predict` says so.
        """
        self = cls.__new__(cls)
        meta = _mojotrees.dataset_metadata(handle)
        self._handle = handle
        self._sparse = bool(meta["is_sparse"])
        self._n_rows = int(meta["num_data"])
        self._n_features = int(meta["num_feature"])
        self.params = {
            "max_bin": int(meta["max_bin"]),
            "use_missing": bool(meta["use_missing"]),
        }
        self.free_raw_data = False
        self.keep_raw = bool(meta["has_raw"])
        self.reference = None
        self._x = None
        self._raw = None
        names = [str(n) for n in _mojotrees.dataset_feature_names(handle)]
        self._names = names if names else None
        self._categorical = [
            int(f) for f in _mojotrees.dataset_categorical_features(handle)
        ]
        self._label = cls._field_buffer(handle, "label")
        self._weight = cls._field_buffer(handle, "weight")
        self._init_score = cls._field_buffer(handle, "init_score")
        self._group = cls._field_buffer(handle, "group")
        return self

    def subset(self, rows, shared_binning=False):
        """The named rows as their own `Dataset`.

        `rows` must be strictly ascending and in range. The selection and
        the binning both happen natively, so which of the two constructions
        this is stays one decision, made in `trainset.mojo`:

        - `shared_binning=False` (the default) **bins the subset over its
          own rows**, so the rows left out had no say in the edges. This is
          what a cross-validation fold or a held-out split needs.
        - `shared_binning=True` bins it by this dataset's mapper, which is
          LightGBM's `Dataset.subset`: the part is binned as the whole was,
          so a model trained on the whole can score it.

        Either way the source must have been built with `keep_raw=True`.
        Bins cannot be refitted from bins, so a dataset that dropped its raw
        matrix raises rather than returning something that looks right.

        The per-row columns come back from the native subset rather than
        being sliced again here, which is what keeps the ranking rule (a
        subset takes whole queries, never part of one) in one place.
        """
        selection = [int(r) for r in rows]
        if not selection:
            raise ValueError("a subset needs at least one row")
        buf = _arrays.i64_vector(selection, "rows")
        handle = _mojotrees.dataset_subset(
            self._constructed(),
            _addr(buf),
            len(selection),
            int(bool(shared_binning)),
        )
        # `buf` had to outlive the call above.
        del buf
        if self._raw is None or self._sparse:
            # Nothing to slice on this side: either the matrix is gone, or
            # it is sparse and row selection out of it is the native
            # subset's job, which already happened.
            return Dataset._from_handle(handle)
        group = [int(v) for v in Dataset._field_list(handle, "group")]
        out = Dataset(
            _arrays.take_rows(self._raw, selection),
            label=Dataset._field_buffer(handle, "label"),
            weight=Dataset._field_buffer(handle, "weight"),
            group=group or None,
            init_score=Dataset._field_buffer(handle, "init_score"),
            feature_name=self._names,
            categorical_feature=self._categorical or None,
            params=dict(self.params),
            free_raw_data=False,
            keep_raw=not bool(shared_binning),
        )
        # The rows were already binned natively; adopting that handle is
        # what makes this one binning rather than two.
        out._handle = handle
        return out

    def save_binned(self, path):
        """Write the binning and the columns to `path` as a prepared table.

        Binning is what a run pays for before it can start, so a table
        written here is what lets the next process, or the next machine,
        skip it. It is not a model file and cannot be loaded as one.
        Returns self.
        """
        _mojotrees.dataset_save(self._constructed(), str(path))
        return self

    @classmethod
    def load_binned(cls, path):
        """Read a prepared table written by `save_binned`.

        The result is a constructed `Dataset`: train on it, ask it for its
        fields, continue a model on it. It carries no raw matrix, because a
        prepared table is a binning rather than the values it was fitted
        from, so it cannot be `subset` and cannot be predicted on.
        """
        return cls._from_handle(_mojotrees.dataset_load(str(path)))

    # -- description -----------------------------------------------------

    def num_data(self):
        """Rows in the dataset."""
        return self._n_rows

    def num_feature(self):
        """Features in the dataset."""
        return self._n_features

    def num_bin(self):
        """Bins the binning reserved per feature. Constructs the dataset,
        since a bin count is a property of fitted bins."""
        return int(_mojotrees.dataset_num_bin(self._constructed()))

    @property
    def is_sparse(self):
        """Whether the binned matrix is sparse, which is decided by what the
        dataset was built from and never changes afterwards."""
        return self._sparse

    def nnz(self):
        """Stored entries in the binned matrix: the stored ones for a sparse
        dataset, every cell for a dense one. Constructs the dataset, since
        the count is a property of the binned matrix."""
        return int(_mojotrees.dataset_metadata(self._constructed())["nnz"])

    def metadata(self):
        """Everything scalar about the constructed dataset, in one call:
        the shape, the effective and requested bin counts, `use_missing`,
        the column and query counts, and `is_sparse`, `nnz`, and `has_raw`.

        Read from the binned dataset rather than from this object's own
        copies, so it answers for a dataset this process did not build
        (`load_binned`, `subset`) as well as for one it did.
        """
        meta = _mojotrees.dataset_metadata(self._constructed())
        return {str(key): meta[key] for key in meta}

    @property
    def feature_name(self):
        """The feature names, LightGBM's `Column_0`, `Column_1`, ... when
        the dataset was built without any."""
        if self._names is None:
            return [f"Column_{i}" for i in range(self._n_features)]
        return list(self._names)

    @property
    def categorical_feature(self):
        """The declared categorical features, as indices."""
        return list(self._categorical)

    def get_label(self):
        """The label column, or None."""
        return None if self._label is None else _finish(self._label)

    def get_weight(self):
        """The sample weights, or None."""
        return None if self._weight is None else _finish(self._weight)

    def get_group(self):
        """The per-query row counts, or None."""
        if self._group is None:
            return None
        return [int(v) for v in self._group]

    def get_init_score(self):
        """The init scores, or None."""
        return (
            None if self._init_score is None else _finish(self._init_score)
        )

    def get_data(self):
        """The raw matrix as it was passed in, or None once it was freed."""
        return self._raw

    def get_field(self, field_name):
        """LightGBM's `get_field`, for the fields mojotrees keeps."""
        getters = {
            "label": self.get_label,
            "weight": self.get_weight,
            "group": self.get_group,
            "init_score": self.get_init_score,
        }
        if field_name not in getters:
            raise ValueError(
                f"unknown field {field_name!r}; a mojotrees Dataset holds "
                + ", ".join(sorted(getters))
            )
        return getters[field_name]()

    def __repr__(self):
        state = "constructed" if self._handle is not None else "unconstructed"
        layout = "sparse" if self._sparse else "dense"
        return (
            f"Dataset({self._n_rows} rows, {self._n_features} features, "
            f"{layout}, {state})"
        )


class Booster:
    """A fitted model: predict with it, score it, save it, extend it.

    Construct one from a training set to boost on, as LightGBM does, and it
    starts with no iterations:

        booster = mojotrees.Booster({"objective": "regression"}, train_set)
        while not booster.update():
            ...

    or let `train()` do that for you. `Booster(model_file=...)` and
    `Booster(model_str=...)` read a model back; what a model file holds is
    the ensemble, its binning, its split gains, and its feature names, but
    not the training parameters, so a booster read back reports what it has
    and says so where it cannot.
    """

    def __init__(
        self, params=None, train_set=None, model_file=None, model_str=None
    ):
        self._handle = None
        self._config = None
        self._train_set = None
        self._valid_sets = []
        self._n_classes = 0
        self._task = None
        self._objective = None
        self._names = None
        self._importance_cache = None

        sources = sum(
            x is not None for x in (train_set, model_file, model_str)
        )
        if sources != 1:
            raise ValueError(
                "a Booster comes from exactly one of train_set, model_file, "
                "or model_str"
            )
        if model_file is not None or model_str is not None:
            if params:
                raise ValueError(
                    "params has no effect on a Booster read from a model; "
                    "the file holds the ensemble, not the run that made it"
                )
            self._load(model_file, model_str)
            return
        if not isinstance(train_set, Dataset):
            raise TypeError("train_set must be a mojotrees.Dataset")
        self._config = _Config(params)
        self._config.check_dataset(train_set)
        self._task = self._config.task
        self._objective = self._config.objective
        self._n_classes = (
            self._config.n_classes
            if self._config.task == _eval.MULTICLASS
            else 0
        )
        self._train_set = train_set
        self._names = train_set.feature_name
        self._build(0)

    # -- training --------------------------------------------------------

    def _build(self, rounds):
        """Train `rounds` rounds on the training set, replacing whatever
        this booster held."""
        handle = self._train_set._constructed()
        params, keep, self.device_ = self._config.binding_params(
            self._train_set, rounds
        )
        if self._task == _eval.MULTICLASS:
            self._handle = _mojotrees.train_dataset_multiclass(handle, params)
        elif self._task == _eval.RANKING:
            self._handle = _mojotrees.train_dataset_ranker(handle, params)
        else:
            self._handle = _mojotrees.train_dataset(handle, params)
        # The constraint buffers had to outlive the call above.
        del keep
        self._importance_cache = None

    def update(self, num_iteration=1):
        """Grow `num_iteration` more boosting rounds on the training set.

        Returns True when the model is finished, meaning the round produced
        nothing to add because the objective has converged, and False when
        it grew. That is LightGBM's `update()` contract, and the loop it
        supports:

            while not booster.update():
                ...

        The new trees resume from where the existing ones left off, so
        `train(..., 40)` followed by `update(60)` gives the ensemble that
        `train(..., 100)` would have (same data, same parameters). The raw
        training scores are recomputed from the model on each call, so one
        call of `update(60)` costs less than 60 calls of `update()`.
        """
        self._require_trainable()
        rounds = int(num_iteration)
        if rounds < 0:
            raise ValueError("num_iteration must not be negative")
        params, keep, _ = self._config.binding_params(
            self._train_set, rounds
        )
        grow = (
            _mojotrees.booster_update_multiclass
            if self._n_classes
            else _mojotrees.booster_update
        )
        added = int(
            grow(self._handle, self._train_set._constructed(), params)
        )
        del keep
        self._importance_cache = None
        return added == 0

    def _require_trainable(self):
        if self._train_set is None or self._config is None:
            raise ValueError(
                "this Booster has no training set to grow on: continued "
                "training needs the Dataset the model was trained on, which "
                "a model file, a pickle, and an estimator's fit() do not "
                "carry. Train through mojotrees.train() to keep it"
            )
        if self._task == _eval.RANKING:
            raise NotImplementedError(
                "continued training does not cover ranking: LambdaRank "
                "gradients are computed within a query from state the "
                "fitted ensemble does not carry, so it cannot be resumed"
            )

    # -- shape -----------------------------------------------------------

    def current_iteration(self):
        """Boosting iterations this model holds."""
        if self._n_classes:
            return int(_mojotrees.num_iterations_multiclass(self._handle))
        return int(_mojotrees.num_iterations(self._handle))

    def num_model_per_iteration(self):
        """Trees per boosting iteration: one, or one per class for a
        softmax model."""
        return self._n_classes if self._n_classes else 1

    def num_trees(self):
        """Trees in the ensemble."""
        return self.current_iteration() * self.num_model_per_iteration()

    def num_feature(self):
        """Features the model was trained on."""
        if self._n_classes:
            return int(_mojotrees.n_features_multiclass(self._handle))
        return int(_mojotrees.n_features(self._handle))

    def feature_name(self):
        """The training set's feature names, LightGBM's `Column_0`,
        `Column_1`, ... when there were none (a model file carries no
        names, so one read back always reports those)."""
        if self._names is None:
            return [f"Column_{i}" for i in range(self.num_feature())]
        return list(self._names)

    def __repr__(self):
        return (
            f"Booster({self.current_iteration()} iterations, "
            f"{self.num_feature()} features)"
        )

    # -- prediction ------------------------------------------------------

    def _predict_data(self, data):
        """The matrix to predict on, out of a `Dataset` or a bare matrix.

        A `Dataset` is a binning; predicting on it means predicting on the
        rows it was binned from, through the *model's* own mapper. So this
        needs the raw matrix, and a dataset that has none says which of the
        two ways it came to have none.
        """
        if not isinstance(data, Dataset):
            return data
        raw = data.get_data()
        if raw is None:
            raise ValueError(
                "this Dataset's raw matrix was freed or is unavailable: it "
                "was built with free_raw_data=True, or read from a prepared table, "
                "which carries a binning rather than the values it was "
                "fitted from"
            )
        return raw

    def _check_X(self, data):
        data = self._predict_data(data)
        Xb, n_rows, n_features, _ = _arrays.check_X(data)
        expected = self.num_feature()
        if n_features != expected:
            raise ValueError(
                f"X has {n_features} features, but this Booster was trained "
                f"on {expected}"
            )
        return Xb, n_rows

    def _slice(self, start_iteration, num_iteration):
        from . import _as_iteration

        total = self.current_iteration()
        start = _as_iteration(start_iteration, "start_iteration")
        start = min(max(start, 0), total)
        if num_iteration is None:
            return start, total
        num = _as_iteration(num_iteration, "num_iteration")
        if num <= 0:
            return start, total
        return start, min(start + num, total)

    def predict(
        self, data, raw_score=False, start_iteration=0, num_iteration=None
    ):
        """Predictions for `data`, which may be a matrix or a `Dataset`.

        `raw_score` returns scores on the link scale: log-odds for the
        binary objective, per-class scores before the softmax for
        multiclass, and the ensemble output itself for everything else.
        Without it a binary model returns the probability of the positive
        class, a multiclass model a row of probabilities per row, and the
        rest their response scale.

        `start_iteration` and `num_iteration` slice the ensemble the way
        LightGBM's do: out-of-range bounds clamp rather than raise, and
        `num_iteration=None` means every iteration from the start on.

        Sparse input is walked without densifying, which is what the sparse
        prediction path is for; the options it cannot serve are refused
        rather than silently densified.
        """
        matrix = self._predict_data(data)
        if _arrays.is_sparse(matrix):
            return self._predict_sparse(
                matrix, raw_score, start_iteration, num_iteration
            )
        Xb, n_rows = self._check_X(matrix)
        start, stop = self._slice(start_iteration, num_iteration)
        if self._n_classes:
            out = _out_buffer(n_rows * self._n_classes)
            _mojotrees.predict_proba_range(
                self._handle,
                _addr(Xb),
                n_rows,
                self.num_feature(),
                start,
                stop,
                int(bool(raw_score)),
                _addr(out),
            )
            if _np is not None:
                return out.reshape(n_rows, self._n_classes)
            k = self._n_classes
            return [list(out[r * k : (r + 1) * k]) for r in range(n_rows)]
        out = _out_buffer(n_rows)
        _mojotrees.predict_range(
            self._handle,
            _addr(Xb),
            n_rows,
            self.num_feature(),
            start,
            stop,
            int(bool(raw_score)),
            _addr(out),
        )
        return _finish(out)

    def _predict_sparse(self, X, raw_score, start_iteration, num_iteration):
        """Predictions for a SciPy sparse matrix, one binary search per node
        over that row's own stored entries rather than a densified row.

        Two options have no sparse implementation and are refused with the
        reason: slicing the ensemble, which reads a dense binned matrix, and
        raw scores from a softmax model, for which there is no sparse entry
        point. Densifying behind the caller would defeat the path they chose
        by passing a sparse matrix.
        """
        if start_iteration != 0 or num_iteration is not None:
            raise ValueError(
                "iteration slicing does not take sparse input yet; densify "
                "with .toarray()"
            )
        buffers, n_rows, n_features, _ = _arrays.check_X_sparse(X, "csr")
        expected = self.num_feature()
        if n_features != expected:
            raise ValueError(
                f"X has {n_features} features, but this Booster was trained "
                f"on {expected}"
            )
        params = buffers.params()
        if self._n_classes:
            if raw_score:
                raise ValueError(
                    "raw scores from a softmax model do not take sparse "
                    "input yet; densify with .toarray()"
                )
            out = _out_buffer(n_rows * self._n_classes)
            _mojotrees.predict_proba_csr(self._handle, params, _addr(out))
            # No numpy-free arm here: `check_X_sparse` above refuses sparse
            # input outright when numpy is missing, so this point is only
            # ever reached with numpy present. See
            # `test_sparse_input_is_refused_rather_than_served`.
            return out.reshape(n_rows, self._n_classes)
        out = _out_buffer(n_rows)
        query = (
            _mojotrees.predict_raw_csr
            if raw_score
            else _mojotrees.predict_csr
        )
        query(self._handle, params, _addr(out))
        # `buffers` had to outlive the call above.
        del buffers
        return _finish(out)

    # -- evaluation ------------------------------------------------------

    def add_valid(self, data, name):
        """Register a validation set for `eval_valid()`, LightGBM's
        `add_valid`."""
        if not isinstance(data, Dataset):
            raise TypeError("a validation set must be a mojotrees.Dataset")
        if data.get_label() is None:
            raise ValueError(f"validation set {name!r} needs a label")
        if data.num_feature() != self.num_feature():
            raise ValueError(
                f"validation set {name!r} has {data.num_feature()} features, "
                f"but this Booster was trained on {self.num_feature()}"
            )
        label = str(name)
        if any(label == existing for existing, _ in self._valid_sets):
            raise ValueError(f"a validation set named {label!r} is already "
                             "registered")
        self._valid_sets.append((label, data))
        return self

    def eval(self, data, name, metric=None):
        """Score `data` and return LightGBM's list of
        `(dataset_name, metric_name, value, is_higher_better)` tuples.

        `metric` names a built-in metric (see `_eval` for the names and
        their aliases) and defaults to the one LightGBM would score for
        this objective. The value comes from src/mojotrees/metrics.mojo, the
        same code the estimators' `eval_set` scores with, and it is weighted
        by the dataset's own weights when it has any.

        A Booster read back from a model file does not know its objective,
        so it needs `metric` named explicitly.
        """
        from . import _SQUARED_ERROR

        if not isinstance(data, Dataset):
            raise TypeError("eval() takes a mojotrees.Dataset")
        labels = data.get_label()
        if labels is None:
            raise ValueError(f"evaluation set {name!r} needs a label")
        if metric is None:
            if self._task is None:
                raise ValueError(
                    "this Booster was read from a model file, which does not "
                    "carry the objective, so eval() needs an explicit metric"
                )
            metric = _eval.default_metric(self._task, self._objective)
        task = self._task or _eval.REGRESSION
        canonical, code, higher = _eval.resolve(metric, task)

        width = self._n_classes if self._n_classes else 1
        pred = _out_buffer(data.num_data() * width)
        raw = self.predict(data, raw_score=True)
        if _np is not None:
            pred[:] = _np.asarray(raw, dtype=_np.float64).reshape(-1)
        else:
            flat = raw if width == 1 else [v for row in raw for v in row]
            for i, value in enumerate(flat):
                pred[i] = float(value)
        params = {
            "pred_addr": _addr(pred),
            "y_addr": _addr(data._label),
            "weight_addr": (
                0 if data._weight is None else _addr(data._weight)
            ),
            "n_rows": data.num_data(),
            "n_classes": int(self._n_classes),
            "group_addr": 0 if data._group is None else _addr(data._group),
            "n_groups": 0 if data._group is None else len(data._group),
            "ndcg_at": int(
                5 if self._config is None else self._config.ndcg_at
            ),
            "alpha": float(
                0.9 if self._config is None else self._config.alpha
            ),
            # The link the metric scores through belongs to the objective
            # (see eval_metric in bindings/_mojotrees.mojo). A Booster read
            # from a model file has no config, and its raw scores are all it
            # knows, so the identity stands in.
            "objective": int(
                _SQUARED_ERROR
                if self._config is None or task == _eval.MULTICLASS
                else self._config.objective_code
            ),
        }
        if code in (_eval.NDCG, _eval.MAP) and not params["n_groups"]:
            raise ValueError(f"{canonical} needs a Dataset with group")
        value = float(_mojotrees.eval_metric(code, params))
        return [(str(name), canonical, value, higher)]

    def eval_train(self, metric=None):
        """Score the training set, LightGBM's `eval_train`."""
        if self._train_set is None:
            raise ValueError(
                "this Booster has no training set to score: a model file, a "
                "pickle, and an estimator's fit() do not carry one"
            )
        return self.eval(self._train_set, "training", metric)

    def eval_valid(self, metric=None):
        """Score every registered validation set, LightGBM's
        `eval_valid`."""
        out = []
        for name, dataset in self._valid_sets:
            out.extend(self.eval(dataset, name, metric))
        return out

    # -- importance ------------------------------------------------------

    def feature_importance(self, importance_type="split"):
        """Per-feature importance: `split` counts the nodes that split on
        each feature and `gain` sums the gain those splits earned.

        Gains travel with the model from format v4 on, so a Booster read
        back from a file or a pickle reports the gain importance it was
        trained with. One read from a v1, v2, or v3 file reports zeros:
        those formats dropped gains, and a fitted tree cannot recompute
        them. `dump_model()["has_split_gain"]` tells that apart from a
        measured zero.
        """
        if importance_type not in _IMPORTANCE_TYPES:
            raise ValueError(
                f"unknown importance_type {importance_type!r}; expected one "
                "of " + ", ".join(sorted(_IMPORTANCE_TYPES))
            )
        cache = self._importance_cache
        if cache is None:
            cache = {
                kind: self._raw_importance(kind) for kind in _IMPORTANCE_TYPES
            }
            self._importance_cache = cache
        values = cache[importance_type]
        return values.copy() if _np is not None else list(values)

    def _raw_importance(self, importance_type):
        out = _out_buffer(self.num_feature())
        query = (
            _mojotrees.feature_importance_multiclass
            if self._n_classes
            else _mojotrees.feature_importance
        )
        query(
            self._handle,
            self.num_feature(),
            _IMPORTANCE_TYPES[importance_type],
            _addr(out),
        )
        return _finish(out)

    # -- structured inspection -------------------------------------------
    #
    # One implementation, in `mojotrees.inspection`, reached from the model
    # object the way LightGBM reaches it. The import is inside each method
    # rather than at the top of this file for two reasons: `inspection`
    # imports nothing from here but is imported *by* the estimators, and
    # `Booster` is the object it inspects, so a top-level import would put
    # a cycle between two modules that do not otherwise need each other.
    # It costs one `sys.modules` lookup per call.

    def dump_model(self, feature_names=None):
        """The model as the documented inspection schema, a plain dict.

        LightGBM's `Booster.dump_model()`. `feature_names` overrides the
        names the model carries, which is how a model read back from a file
        written before format v4 (those carry none) gets named. Every key
        is in `docs/MODEL_INSPECTION_SCHEMA.md`; the two to branch on are
        `has_split_gain` and `has_node_count`.
        """
        from . import inspection

        return inspection.dump_model(self, feature_names=feature_names)

    def trees_to_dataframe(self):
        """The ensemble as a pandas DataFrame, one row per node.

        LightGBM's `Booster.trees_to_dataframe()`, with LightGBM's column
        names and the two deliberate gaps the module docstring lists.
        pandas is not a dependency of mojotrees: `trees_to_records()`
        returns the same rows as plain dicts and needs nothing installed.
        """
        from . import inspection

        return inspection.trees_to_dataframe(self)

    def trees_to_records(self):
        """`trees_to_dataframe` without pandas: one dict per node, in the
        same column order and the same depth-first per-tree row order."""
        from . import inspection

        return inspection.trees_to_records(self)

    def get_split_value_histogram(self, feature, bins=None, as_frame=False):
        """How the ensemble's splits on one feature are distributed.

        The data behind LightGBM's `plot_split_value_histogram`, and
        nothing else: no plotting dependency is introduced by asking.
        Returns `(counts, bin_edges)`, or a pandas DataFrame with
        `as_frame=True`.
        """
        from . import inspection

        return inspection.get_split_value_histogram(
            self, feature, bins=bins, as_frame=as_frame
        )

    # -- model IO --------------------------------------------------------

    def save_model(self, filename):
        """Write the model to `filename` in mojotrees's versioned text
        format. The format is mojotrees's own and is not LightGBM's.

        The feature names travel with the model when it has any: a model
        file carries them from format v4 on, so one read back reports the
        names it was trained with rather than `Column_0`, `Column_1`, ...
        A model that never had names writes none, and its file is what it
        always was.
        """
        names = [] if self._names is None else [str(n) for n in self._names]
        if self._n_classes:
            _mojotrees.save_multiclass(
                self._handle, str(filename), names, len(names)
            )
        else:
            _mojotrees.save(self._handle, str(filename), names, len(names))
        return self

    def model_to_string(self):
        """The whole model as the text `save_model` writes."""
        with _tempfile.TemporaryDirectory() as d:
            path = _os.path.join(d, "model.mbst")
            self.save_model(path)
            with open(path, "r") as fh:
                return fh.read()

    @classmethod
    def model_from_string(cls, model_str):
        """A Booster from the text `model_to_string` produced."""
        return cls(model_str=model_str)

    def _load(self, model_file, model_str):
        if model_str is not None:
            with _tempfile.TemporaryDirectory() as d:
                path = _os.path.join(d, "model.mbst")
                with open(path, "w") as fh:
                    fh.write(model_str)
                self._load_path(path)
            return
        self._load_path(str(model_file))

    def _load_path(self, path):
        """Read a model file, single-output or multiclass. The format
        distinguishes the two, so which one it is comes from the file.

        Feature names come back too when the file carries them (v4 and
        later). A file without them leaves whatever names the caller
        already had alone, which is what keeps an older pickle's names: it
        stores them beside the model text, and an older text cannot carry
        them. With neither, `_names` stays None and `feature_name()`
        reports `Column_i`, because the model names nothing.
        """
        try:
            self._handle = _mojotrees.load(path)
            self._n_classes = 0
        except Exception:
            self._handle = _mojotrees.load_multiclass(path)
            self._n_classes = int(_mojotrees.n_classes(self._handle))
            self._task = _eval.MULTICLASS
        names = list(_mojotrees.model_feature_names(path))
        if names:
            self._names = [str(name) for name in names]

    @classmethod
    def _from_estimator(cls, handle, estimator):
        """The Booster an estimator holds for a model it just trained or
        loaded.

        Whether the handle is single-output or softmax comes from the handle
        itself rather than from the estimator, so this does not depend on
        how far through `fit` or `load` the estimator happens to be. There
        is no training set: an estimator bins the matrix it was handed and
        does not keep a `Dataset`, so the booster it holds can do everything
        except `update()`.
        """
        booster = cls.__new__(cls)
        booster._handle = handle
        booster._config = None
        booster._train_set = None
        booster._valid_sets = []
        booster._importance_cache = None
        multiclass = type(handle).__name__ == "MulticlassModel"
        booster._n_classes = (
            int(_mojotrees.n_classes(handle)) if multiclass else 0
        )
        booster._task = _estimator_task(estimator, multiclass)
        booster._objective = getattr(estimator, "objective", None)
        names = getattr(estimator, "feature_names_in_", None)
        booster._names = None if names is None else [str(n) for n in names]
        return booster

    def _copy_handle(self):
        """An independent copy of the model, split gains included."""
        if self._n_classes:
            return _mojotrees.copy_multiclass_model(self._handle)
        return _mojotrees.copy_model(self._handle)

    # -- pickling --------------------------------------------------------

    def __getstate__(self):
        """The model travels as the text `model_to_string` writes, split
        gains and feature names included (model format v4 carries both).
        The training set and the parameter object do not pickle with it: a
        Mojo handle is not picklable."""
        return {
            "model_str": None if self._handle is None
            else self.model_to_string(),
            "n_classes": self._n_classes,
            "task": self._task,
            "objective": self._objective,
            "names": None if self._names is None else list(self._names),
        }

    def __setstate__(self, state):
        self._handle = None
        self._config = None
        self._train_set = None
        self._valid_sets = []
        self._importance_cache = None
        self._n_classes = int(state.get("n_classes", 0))
        self._task = state.get("task")
        self._objective = state.get("objective")
        self._names = state.get("names")
        blob = state.get("model_str")
        if blob is not None:
            self._load(None, blob)


def train(
    params,
    train_set,
    num_boost_round=None,
    valid_sets=None,
    valid_names=None,
    init_model=None,
    *,
    feval=None,
    callbacks=None,
):
    """Train a model and return the `Booster` that holds it.

    `params` is a LightGBM parameter dict: `objective`, `num_class` for
    multiclass, and the training parameters the estimators take under their
    LightGBM names. Parameters that describe the data rather than the run
    (`max_bin`, `use_missing`, `categorical_feature`, and the columns
    themselves) belong to the `Dataset` and are rejected here rather than
    ignored.

    `num_boost_round` is the number of iterations; LightGBM's aliases in
    `params` work too, and two that disagree raise.

    `valid_sets` and `valid_names` register validation sets on the returned
    booster, so `booster.eval_valid()` scores them. There is no per-round
    history and no early stopping here yet: fit an estimator with
    `eval_set=` and `early_stopping_rounds=` for those.

    `init_model` is a `Booster` to continue training from. Its trees are
    left alone: they are copied into the returned booster, which then grows
    `num_boost_round` more rounds from where they left off. The dataset
    must be the one that model was trained on, which is checked by
    comparing the binning rather than taken on trust.

    `feval` and `callbacks` are LightGBM's per-round hooks. They are named
    in the signature only so that passing one is refused by name instead of
    as an unexpected keyword: the round loop that scores metrics and runs
    callbacks is the estimators' `fit(eval_set=...)` and `mojotrees.cv`,
    and accepting either here would let it look as though it ran.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError("train_set must be a mojotrees.Dataset")
    if feval is not None:
        raise ValueError(
            "train() does not score per-round metrics, so feval would never "
            "be called; fit an estimator with eval_set= and eval_metric=, "
            "or use mojotrees.cv(feval=...)"
        )
    if callbacks:
        raise ValueError(
            "train() has no per-round callback loop, so a callback would "
            "never be called; fit an estimator with eval_set= and "
            "callbacks=, or use mojotrees.cv(callbacks=...)"
        )
    config = _Config(params, num_boost_round)
    config.check_dataset(train_set)

    if init_model is None:
        booster = _fresh(config, train_set)
    else:
        booster = _continue(config, train_set, init_model)
    _register_valid(booster, valid_sets, valid_names)
    return booster


def _shell(config, train_set):
    """A Booster wired to a configuration and a training set, with no model
    in it yet. `Booster.__init__` trains a zero-round model, which the two
    callers below replace or extend, so it is bypassed here."""
    booster = Booster.__new__(Booster)
    booster._handle = None
    booster._config = config
    booster._train_set = train_set
    booster._valid_sets = []
    booster._n_classes = (
        config.n_classes if config.task == _eval.MULTICLASS else 0
    )
    booster._task = config.task
    booster._objective = config.objective
    booster._names = train_set.feature_name
    booster._importance_cache = None
    return booster


def _fresh(config, train_set):
    """A booster trained from scratch."""
    booster = _shell(config, train_set)
    booster._build(config.rounds)
    return booster


def _continue(config, train_set, init_model):
    """Continue training from an existing booster, leaving it untouched.

    The model is copied first, so the trees `init_model` holds are still
    its own afterwards. A copy is the handle itself and keeps everything,
    which is now also true of a save/load round trip through format v4.
    """
    if not isinstance(init_model, Booster):
        raise TypeError(
            "init_model must be a mojotrees.Booster; read a saved model with "
            "Booster(model_file=...) first"
        )
    if bool(init_model._n_classes) != (config.task == _eval.MULTICLASS):
        raise ValueError(
            "init_model and objective disagree: a softmax model can only be "
            "continued as multiclass, and a single-output model cannot"
        )
    booster = _shell(config, train_set)
    booster._handle = init_model._copy_handle()
    booster.update(config.rounds)
    return booster


def _register_valid(booster, valid_sets, valid_names):
    if valid_sets is None:
        return
    sets = list(valid_sets)
    if valid_names is None:
        names = [f"valid_{i}" for i in range(len(sets))]
    else:
        names = [str(name) for name in valid_names]
        if len(names) != len(sets):
            raise ValueError(
                "valid_names must have one name per valid_sets entry"
            )
    for name, dataset in zip(names, sets):
        booster.add_valid(dataset, name)
