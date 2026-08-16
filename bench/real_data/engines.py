"""Engine adapters. One measured run each, the same phases on every arm.

Every adapter answers the same questions in the same order, and every one is
asked only for predictions. Metrics are computed later, by quality.py, from
those predictions. An engine's own evaluation output is never read into a
result record.

Four engine names, and they are not four libraries:

- `mojotrees` and `lightgbm` are the comparator row. LightGBM at
  `stock+det` is the one comparator, and nothing in this module changes it.
- `catboost` is a peer arm, reported beside the comparator and never
  instead of it.
- `mojotrees_catboost_mode` is the mojotrees side of the second peer row.
  It differs from `mojotrees` by exactly `scenarios.MOJOTREES_CATBOOST_MODE`.

The phases, and why they are split where they are:

- `import`: loading the library. A Mojo extension pays a shared-library
  load and a dynamic-linker pass here; LightGBM pays a smaller one. It is
  measured because a user who trains one small model pays it, and it is
  separate because a user who trains a hundred does not.
- `warmup`: a deliberately tiny fit on the target device, before anything
  that counts. This is where an accelerator compiles its kernels and where
  an OpenMP runtime spins up its thread pool. Attributing that to the
  first real training run would flatter whichever engine happened to be
  second.
- `ingest`: getting the caller's array into the library's own layout, where
  that is a step the library exposes on its own. CatBoost is the only
  engine here that has one: `Pool()` converts and does not bin. LightGBM's
  ingestion is inside `Dataset.construct()` and has always been counted in
  its binning phase.
- `binning`: constructing the Dataset. Two of the three expose this as an
  explicit step. On the paths where one of them bins inside fit, the field
  is null with a reason, which is now true of every CatBoost row and of the
  mojotrees sparse path.
- `train`: boosting rounds. On the paths above it contains binning, and the
  record says so rather than leaving the two figures to be added.
- `predict_batch` and `predict_row`: throughput on the whole test matrix,
  and the latency of a single-row call. The second is not the first
  divided by the row count, and a library can be good at one and bad at
  the other.

`scenarios.PHASE_SHAPE` states, per engine, what each phase contains, so the
three end-to-end lines can be read against each other rather than assumed
comparable.

Every adapter returns the predictions themselves, so the runner can compute
metrics, hash them for the determinism check, and compare the engines row by
row.

They also return `params_used`, which is the dict that was actually handed
to the library, after the adapter has added whatever the scenario module
could not know. Rebuilding the dict for the record instead would omit
exactly those additions, and a record that understates what ran is worse
than no record, because it reads like evidence.
"""

import os
import tempfile

import numpy as np

import measure
import scenarios


#: How many per-feature bin counts a record carries in full. Every scenario
#: in this suite is far below it except sparse_highdim, whose `large` tier
#: declares 500,000 features. Chosen so that the standard tier of every
#: scenario, sparse included at 50,000, records its whole vector.
BIN_COUNT_LIST_LIMIT = 65_536


class EngineError(RuntimeError):
    """Raised when an engine cannot run a scenario at all. The runner turns
    this into a recorded skip with the reason, not a crash."""


def _tiny_like(spec):
    """A few rows shaped like the scenario, for the warmup fit. Small
    enough that the fit itself is noise next to whatever one-off cost the
    warmup is there to trigger."""
    rows, features = 64, 4
    x = np.linspace(0.0, 1.0, rows * features).reshape(rows, features)
    if spec["task"] == "multiclass":
        y = np.arange(rows, dtype=np.float64) % 3.0
        return x, y, np.full(1, rows, dtype=np.int64), 3
    if spec["task"] == "binary":
        return x, (np.arange(rows) % 2).astype(np.float64), None, 0
    if spec["task"] == "ranking":
        y = (np.arange(rows, dtype=np.float64) % 5.0)
        return x, y, np.full(rows // 8, 8, dtype=np.int64), 0
    return x, x[:, 0].copy(), None, 0


def _bin_profile(dataset, n_features):
    """The per-feature bin counts, reduced to something a record can hold.

    Both libraries expose `Dataset.feature_num_bin(i)`, under that name and
    with the same meaning: how many bins the fitted binning gave one
    feature. It is the one field in the record that lets a reader check the
    binning alignment by measurement instead of by reading scenarios.py and
    believing it, which is why it is worth the loop.

    The counts themselves are stored, and they were not. Storing only the
    shape and a digest cost a lane an hour tonight: diagnosing an accuracy
    gap, it had to reconstruct the per-feature counts by arithmetic over a
    recorded total, and the list would have answered the question by being
    read. It is a few hundred bytes on every scenario in this suite except
    the sparse one, and records are not committed.

    The reason the list was dropped is still real at one shape: the sparse
    `large` tier declares 500,000 features, and half a million integers in
    every record is a different proposition from fifty. So the list is
    stored whole up to `BIN_COUNT_LIST_LIMIT` features and truncated with a
    marker above it, and the digest is always taken over the entire vector,
    so two records with the same digest binned identically whether or not
    either list was truncated. The digest goes through measure.digest so
    there is one hashing convention in this harness rather than two.

    The loop runs after every timed phase and is not inside any of them.

    Returns a dict, or `measure.unavailable(reason)` if the library refused
    the call, which is what a 4.x LightGBM older than 4.0 would do.
    """
    try:
        counts = np.asarray(
            [int(dataset.feature_num_bin(i)) for i in range(int(n_features))],
            dtype=np.int64,
        )
    except Exception as exc:  # pragma: no cover - engine-dependent
        return measure.unavailable(
            f"feature_num_bin is not callable on this Dataset: "
            f"{type(exc).__name__}: {exc}"
        )
    if counts.size == 0:
        return measure.unavailable("the Dataset reports no features")
    profile = {
        "n_features": int(counts.size),
        "total": int(counts.sum()),
        "max": int(counts.max()),
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "sha256": measure.digest(counts),
        "counts": [int(c) for c in counts[:BIN_COUNT_LIST_LIMIT]],
    }
    if counts.size > BIN_COUNT_LIST_LIMIT:
        profile["counts_truncated"] = True
        profile["counts_truncated_reason"] = (
            f"{int(counts.size)} features; `counts` holds the first "
            f"{BIN_COUNT_LIST_LIMIT} in feature order. `sha256` is over all "
            "of them"
        )
    else:
        profile["counts_truncated"] = False
    return profile


#: mojotrees's side of the same field. It builds histograms feature-major
#: and has no parameter that selects anything else, which
#: docs/LIGHTGBM_PARITY.md records in its force_col_wise / force_row_wise
#: row. Recorded rather than left absent so that a LightGBM record's
#: builder has something to be compared against, and because the pair being
#: compared is not the same strategy on both sides: LightGBM now picks its
#: own builder by timing and mojotrees has one. What varies on this side is
#: thread dispatch, which is MOJOTREES_PARALLEL_MIN_OPS and lives in the
#: environment block.
MOJOTREES_HISTOGRAM_BUILDER = {
    "requested": "feature_major",
    "resolved": "feature_major",
    "recorded": (
        "resolved; mojotrees has one histogram construction and no "
        "parameter to select another"
    ),
}


def _histogram_builder(params):
    """Which histogram construction LightGBM was asked for, and which it
    ran.

    This belongs in the record more than most parameters do. Row-wise and
    col-wise are two different algorithms over the same bins rather than
    two tunings of one, so a training time without the builder next to it
    is not a reproducible measurement.

    Since C9 the comparator sets neither force flag, so this reports
    `auto` and an unresolved value on every LightGBM cell. That is a real
    loss of information and it is the price of the rule: forcing row-wise
    was this repository choosing the comparator's algorithm for it, and
    `bench/results/INSTRUCTION_AUDIT.md` carried the resulting row-versus-col
    caveat on every LightGBM margin in the tree. Stock LightGBM times both
    strategies on the first iterations and keeps the winner, so the
    comparator now runs whichever of its own builders it decides is faster,
    which is both what a user gets and the harder thing to beat.

    What is knowable: the request always. The resolution only when the
    request settles it, which is when either force flag is set. LightGBM
    4.7 keeps the auto choice in the tree learner's share state and exposes
    no getter for it on the Booster, on the Dataset, or in the model text,
    and the one report of it is a log line this harness silences with
    verbosity -1. So the auto case records a null and that reason rather
    than echoing back the request as though it were an answer.
    """
    row = bool(params.get("force_row_wise", False))
    col = bool(params.get("force_col_wise", False))
    if row and col:
        requested = "both_forced"
        resolved = measure.unavailable(
            "both force flags were set, which LightGBM resolves internally "
            "and does not report back. This harness sets one."
        )
    elif row:
        requested = resolved = "force_row_wise"
    elif col:
        requested = resolved = "force_col_wise"
    else:
        requested = "auto"
        resolved = measure.unavailable(
            "neither force flag was set, so LightGBM chose by timing both "
            "on the first iterations. 4.7 keeps that choice in the tree "
            "learner's share state and exposes no getter for it, and the "
            "log line that reports it is suppressed at verbosity -1."
        )
    return {
        "requested": requested,
        "resolved": resolved,
        "recorded": (
            "resolved, which the forced parameter determines"
            if isinstance(resolved, str)
            else "requested only; see resolved.unavailable_reason"
        ),
    }


#: CatBoost's side of the same field. There is nothing to resolve: CatBoost
#: has no row-wise/col-wise choice and no parameter that selects one, so the
#: question the LightGBM field answers does not arise. Recorded as answered
#: rather than as unavailable, because "unavailable" means nobody could find
#: out and this is a case where there is nothing to find out.
CATBOOST_HISTOGRAM_BUILDER = {
    "requested": "not_applicable",
    "resolved": "not_applicable",
    "recorded": (
        "CatBoost builds its histograms one way and exposes no builder "
        "choice. What varies on this side is the tree shape, which is "
        "grow_policy=SymmetricTree, and that is in the resolved parameters"
    ),
}


def _catboost_bin_profile(model, n_features):
    """CatBoost's per-feature bin counts, in the same shape `_bin_profile`
    produces for the other two engines, and from a different place.

    The other two read `Dataset.feature_num_bin(i)` off the constructed
    Dataset. CatBoost has no such call: `Pool` is not quantized after
    construction (see `scenarios.PHASE_SHAPE`), and the borders only become
    readable once a model has been fitted, through `CatBoost.get_borders()`.
    So this is read off the model after the timed fit, and the record says
    so in `source` rather than letting a reader assume the three numbers
    came from the same kind of object.

    `get_borders()` returns thresholds and the other two report bins, so one
    is added per feature. A feature CatBoost gave no borders at all still
    appears with a single bin, which is what a constant column is.
    """
    try:
        borders = model.get_borders()
    except Exception as exc:  # pragma: no cover - engine-dependent
        return measure.unavailable(
            f"CatBoost.get_borders() is not callable on this model: "
            f"{type(exc).__name__}: {exc}"
        )
    counts = np.array(
        [len(borders.get(i, ())) + 1 for i in range(int(n_features))],
        dtype=np.int64,
    )
    if counts.size == 0:
        return measure.unavailable("the model reports no features")
    profile = {
        "n_features": int(counts.size),
        "total": int(counts.sum()),
        "max": int(counts.max()),
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "sha256": measure.digest(counts),
        "counts": [int(c) for c in counts[:BIN_COUNT_LIST_LIMIT]],
        "source": (
            "CatBoost.get_borders() on the fitted model, plus one per "
            "feature to convert thresholds into bins. Not Dataset."
            "feature_num_bin, which CatBoost does not have"
        ),
    }
    if counts.size > BIN_COUNT_LIST_LIMIT:
        profile["counts_truncated"] = True
        profile["counts_truncated_reason"] = (
            f"{int(counts.size)} features; `counts` holds the first "
            f"{BIN_COUNT_LIST_LIMIT} in feature order. `sha256` is over all "
            "of them"
        )
    else:
        profile["counts_truncated"] = False
    return profile


class MojoTreesEngine:
    name = "mojotrees"

    #: The translator this arm's parameters go through. The CatBoost-mode
    #: arm below overrides exactly this and nothing else, so the two arms
    #: run the same code and a reader diffing two records sees
    #: `scenarios.MOJOTREES_CATBOOST_MODE` and nothing else.
    params_fn = staticmethod(scenarios.mojotrees_params)

    def __init__(self, threads, device="cpu"):
        self.threads = int(threads)
        self.device = device
        self.module = None
        self.version = None
        self.import_phase = None
        self.notes = []

    def load(self):
        """Import the extension, having already set the thread count.

        MOJOTREES_NUM_WORKERS is read by the Mojo side, and the runner sets
        it before this process starts. It is asserted here rather than set
        here, because setting it after an import that may already have
        cached a worker count would be a silent lie in the record.
        """
        want = str(self.threads)
        got = os.environ.get("MOJOTREES_NUM_WORKERS")
        if got != want:
            raise EngineError(
                f"MOJOTREES_NUM_WORKERS is {got!r} but this run wants {want!r}; "
                "the runner must set it before the worker process starts"
            )
        phase = measure.Phase("import")
        with phase:
            import mojotrees
        self.module = mojotrees
        self.import_phase = phase
        self.version = getattr(mojotrees, "__version__", "unknown")
        # Every record this engine writes names the comparator it will be
        # read against. worker.py copies `notes` into the record's caveats,
        # so the configuration travels with the number instead of living in
        # a README somebody has to find.
        self.notes.append(
            f"comparator {scenarios.comparator_id()}: "
            f"{scenarios.COMPARATOR_LABEL}. This arm answers nothing to "
            "LightGBM's deterministic: mojotrees is reproducible across "
            "thread counts with no parameter and no cost, which is why that "
            "is the setting the comparator carries."
        )
        if self.device in ("gpu", "auto") and not mojotrees.gpu_available():
            raise EngineError(
                "device='gpu' was requested and mojotrees.gpu_available() is "
                "False; this build or this machine has no accelerator"
            )
        return self

    def warmup(self, spec):
        x, y, group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        params = type(self).params_fn(spec, self.device, extra)
        params["n_estimators"] = 1

        def _fit():
            data = self.module.Dataset(
                x, label=y, group=group, params=scenarios.dataset_params(spec)
            )
            return self.module.train(params, data, num_boost_round=1)

        _, phase = measure.timed(_fit)
        return phase

    def _dataset(self, spec, part, params):
        return self.module.Dataset(
            part["X"],
            label=part["y"],
            group=part.get("group"),
            categorical_feature=part.get("categorical_feature"),
            params=params,
        )

    def run(self, spec, train, test, repeats=1):
        if train.get("sparse"):
            return self._run_sparse(spec, train, test, repeats)
        return self._run_dense(spec, train, test, repeats)

    def _run_dense(self, spec, train, test, repeats):
        extra = None
        if spec["task"] == "multiclass":
            extra = {"num_class": int(train.get("n_classes") or (train["y"].max() + 1))}
        params = type(self).params_fn(spec, self.device, extra)
        params["n_estimators"] = scenarios.BASE_PARAMS["n_estimators"]

        dataset_params = scenarios.dataset_params(spec)
        dataset = self._dataset(spec, train, dataset_params)
        _, binning = measure.timed(dataset.construct)

        rounds = params["n_estimators"]
        booster, training = measure.timed(
            lambda: self.module.train(params, dataset, num_boost_round=rounds)
        )

        predictions, predict_batch = measure.repeat(
            lambda: booster.predict(test["X"]), repeats
        )
        row = np.ascontiguousarray(test["X"][:1])
        _, predict_row = measure.repeat(lambda: booster.predict(row), 20, warmup=2)

        with tempfile.TemporaryDirectory() as scratch:
            size = measure.model_size(booster, scratch)

        return {
            "engine": self.name,
            "engine_version": self.version,
            "device_requested": self.device,
            "device_used": getattr(booster, "device_", None),
            "path": "dataset",
            "phases": {
                "import": self.import_phase.as_dict(),
                "binning": binning.as_dict(),
                "train": training.as_dict(),
                "predict_batch": predict_batch,
                "predict_row": predict_row,
            },
            "params_used": params,
            "dataset_params_used": dataset_params,
            "num_boost_round": rounds,
            "histogram_builder": dict(MOJOTREES_HISTOGRAM_BUILDER),
            "model": {
                "num_trees": booster.num_trees(),
                "current_iteration": booster.current_iteration(),
                "num_bin": dataset.num_bin(),
                "bins": _bin_profile(dataset, train["X"].shape[1]),
                "size": size,
            },
            "transfers": measure.unavailable(
                "mojotrees does not expose host-to-device transfer time to "
                "Python. Instrumenting it is a change to "
                "src/mojotrees/histogram_gpu.mojo and train_gpu.mojo, listed "
                "in the handoff."
            ),
            "peak_rss_bytes": measure.peak_rss_bytes(),
            "notes": list(self.notes),
            "predictions": predictions,
        }

    def _run_sparse(self, spec, train, test, repeats):
        """The CSC path, which goes through the estimator rather than
        Dataset and train.

        Two measurements are unavailable here and both are recorded as
        such: binning happens inside fit and cannot be separated, and the
        device is whatever the CSC trainer supports regardless of what was
        asked for.
        """
        if spec["task"] != "binary":
            raise EngineError(
                f"the sparse path here covers binary classification, not "
                f"{spec['task']}"
            )
        params = type(self).params_fn(spec, "cpu")
        params.pop("device", None)
        # Built once and both passed and recorded, so the record is the
        # call rather than a reconstruction of it. `objective` is dropped
        # because the classifier implies it; the record's shared block
        # still carries it. A collision raises rather than resolving
        # itself, which is what the keyword form this replaced did.
        estimator_params = {
            "n_estimators": scenarios.BASE_PARAMS["n_estimators"],
            "max_bin": scenarios.BASE_PARAMS["max_bin"],
            "use_missing": scenarios.BASE_PARAMS["use_missing"],
        }
        for key, value in params.items():
            if key == "objective":
                continue
            if key in estimator_params:
                raise EngineError(
                    f"the translated parameters and the estimator keywords "
                    f"both set {key!r}, so one of them would silently win"
                )
            estimator_params[key] = value
        estimator = self.module.MojoTreesClassifier(**estimator_params)
        _, training = measure.timed(lambda: estimator.fit(train["X"], train["y"]))

        predictions, predict_batch = measure.repeat(
            lambda: estimator.predict_proba(test["X"])[:, 1], repeats
        )
        _, predict_row = measure.repeat(
            lambda: estimator.predict_proba(test["X"][:1])[:, 1], 20, warmup=2
        )
        booster = estimator.booster_
        with tempfile.TemporaryDirectory() as scratch:
            size = measure.model_size(booster, scratch)

        self.notes.append(
            "sparse path: fit bins internally, so binning time is inside "
            "train and device is cpu whatever was requested"
        )
        return {
            "engine": self.name,
            "engine_version": self.version,
            "device_requested": self.device,
            "device_used": getattr(estimator, "device_", "cpu"),
            "path": "estimator_csc",
            "phases": {
                "import": self.import_phase.as_dict(),
                "binning": None,
                "binning_unavailable_reason": (
                    "the CSC path bins inside fit and does not expose a "
                    "separate construct step"
                ),
                "train": training.as_dict(),
                "predict_batch": predict_batch,
                "predict_row": predict_row,
            },
            "params_used": estimator_params,
            "dataset_params_used": None,
            "dataset_params_unavailable_reason": (
                "the estimator path takes the binning settings as estimator "
                "keywords, which are in params_used, rather than through a "
                "Dataset"
            ),
            "num_boost_round": int(estimator_params["n_estimators"]),
            "histogram_builder": dict(MOJOTREES_HISTOGRAM_BUILDER),
            "model": {
                "num_trees": booster.num_trees(),
                "current_iteration": booster.current_iteration(),
                "num_bin": None,
                "num_bin_unavailable_reason": (
                    "the estimator owns its Dataset internally on the CSC "
                    "path and does not hand it back, so neither num_bin nor "
                    "the per-feature bin counts can be read after fit"
                ),
                "bins": measure.unavailable(
                    "no Dataset is reachable on the estimator CSC path"
                ),
                "size": size,
            },
            "transfers": measure.unavailable("cpu-only path"),
            "peak_rss_bytes": measure.peak_rss_bytes(),
            "notes": list(self.notes),
            "predictions": predictions,
        }


class LightGBMEngine:
    name = "lightgbm"

    def __init__(self, threads, device="cpu"):
        self.threads = int(threads)
        self.device = device
        self.module = None
        self.version = None
        self.import_phase = None
        self.notes = []

    def load(self):
        if self.device != "cpu":
            raise EngineError(
                "this harness runs LightGBM on the CPU only. Its GPU builds "
                "are a compile-time option that the bench environment does "
                "not install, and comparing a mojotrees accelerator run "
                "against a LightGBM CPU run is a comparison of two "
                "different things unless it is labelled as one."
            )
        phase = measure.Phase("import")
        with phase:
            import lightgbm
        self.module = lightgbm
        self.import_phase = phase
        self.version = lightgbm.__version__
        # Before anything is fitted. LightGBM logs "Unknown parameter"
        # instead of refusing one, and this harness runs at a verbosity
        # that suppresses the line, so an unguarded old build would ignore
        # `deterministic` and record itself as stock+det anyway.
        try:
            scenarios.check_lightgbm_version(self.version)
        except RuntimeError as exc:
            raise EngineError(str(exc)) from exc
        self.notes.append(
            f"comparator {scenarios.comparator_id()}: "
            f"{scenarios.COMPARATOR_LABEL}, registered at "
            f"{scenarios.COMPARATOR_REGISTERED}"
        )
        return self

    def warmup(self, spec):
        x, y, group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        params = scenarios.lightgbm_params(spec, self.threads, extra)
        params["n_estimators"] = 1

        def _fit():
            data = self.module.Dataset(x, label=y, group=group, params=params)
            return self.module.train(params, data, num_boost_round=1)

        _, phase = measure.timed(_fit)
        return phase

    def run(self, spec, train, test, repeats=1):
        # No `bin_construct_sample_cnt` here. It used to be raised to the
        # training row count, which made the comparator fit its bin edges
        # from every row while mojotrees fit them from a 200000-row
        # subsample: strictly more binning work on the comparator's side,
        # and every binning ratio measured under it is wrong in our favor.
        # Both engines are at LightGBM's stock 200000 now.
        # `scenarios.lightgbm_params` refuses the parameter by name.
        extra = {}
        if spec["task"] == "multiclass":
            extra["num_class"] = int(
                train.get("n_classes") or (np.max(train["y"]) + 1)
            )
        params = scenarios.lightgbm_params(spec, self.threads, extra)
        rounds = scenarios.BASE_PARAMS["n_estimators"]

        dataset = self.module.Dataset(
            train["X"],
            label=train["y"],
            group=train.get("group"),
            categorical_feature=train.get("categorical_feature") or "auto",
            params=params,
            free_raw_data=False,
        )
        _, binning = measure.timed(dataset.construct)

        booster, training = measure.timed(
            lambda: self.module.train(params, dataset, num_boost_round=rounds)
        )

        predictions, predict_batch = measure.repeat(
            lambda: booster.predict(test["X"]), repeats
        )
        row = test["X"][:1]
        if not train.get("sparse"):
            row = np.ascontiguousarray(row)
        _, predict_row = measure.repeat(lambda: booster.predict(row), 20, warmup=2)

        with tempfile.TemporaryDirectory() as scratch:
            size = measure.model_size(booster, scratch)

        # LightGBM has no single "bins reserved per feature" number the way
        # mojotrees's Dataset.num_bin() does, so num_bin is recorded as the
        # largest per-feature count, which is the closest thing it has, with
        # the whole profile beside it. The two engines' counts are recorded
        # as each library reports them; whether either counts a bin held for
        # missing values is not something this harness has established, and
        # comparing the profiles is how that question gets answered rather
        # than something this comment gets to assert.
        bins = _bin_profile(dataset, train["X"].shape[1])
        num_bin = bins["max"] if isinstance(bins, dict) and "max" in bins else None
        model = {
            "num_trees": booster.num_trees(),
            "current_iteration": booster.current_iteration(),
            "num_bin": num_bin,
            "bins": bins,
            "size": size,
        }
        if num_bin is None:
            model["num_bin_unavailable_reason"] = (
                "LightGBM exposes bin counts per feature, not per Dataset, "
                "and the per-feature read failed: "
                + str(bins.get("unavailable_reason"))
            )

        return {
            "engine": self.name,
            "engine_version": self.version,
            "device_requested": self.device,
            "device_used": "cpu",
            "path": "dataset",
            "phases": {
                "import": self.import_phase.as_dict(),
                "binning": binning.as_dict(),
                "train": training.as_dict(),
                "predict_batch": predict_batch,
                "predict_row": predict_row,
            },
            "params_used": params,
            # LightGBM takes the binning settings in the same dict as the
            # training ones, so there is no second dict to record here.
            "dataset_params_used": None,
            "dataset_params_unavailable_reason": (
                "LightGBM's Dataset and train take one dict, which is "
                "params_used"
            ),
            "num_boost_round": rounds,
            "histogram_builder": _histogram_builder(params),
            "model": model,
            "transfers": measure.unavailable("cpu-only build"),
            "peak_rss_bytes": measure.peak_rss_bytes(),
            "notes": list(self.notes),
            "predictions": predictions,
        }


class MojoTreesCatBoostModeEngine(MojoTreesEngine):
    """mojotrees shaped toward CatBoost's defaults. The "us in CatBoost
    mode" row.

    Everything is inherited. The only difference from the plain arm is the
    translator, which applies `scenarios.MOJOTREES_CATBOOST_MODE` over the
    shared parameters, so the two arms run through identical code and the
    difference between two records is exactly that dict.

    It is a separate engine name rather than a flag on the existing one
    because the harness's unit of comparison is an engine name: it is what
    `verify.py` pairs on, what `report.py` groups on, and what a CSV row
    carries. Naming it `mojotrees` with a variant field would put two
    different models in one column.
    """

    name = "mojotrees_catboost_mode"
    params_fn = staticmethod(scenarios.mojotrees_catboost_mode_params)

    def load(self):
        super().load()
        self.notes.append(
            "arm 'us in CatBoost mode': mojotrees with "
            f"{scenarios.MOJOTREES_CATBOOST_MODE} applied over the shared "
            "defaults, reported against the CatBoost peer arm "
            f"{scenarios.catboost_arm_id()}. This is NOT the comparator row: "
            f"the comparator is {scenarios.comparator_id()} and the plain "
            "mojotrees arm is what is read against it. This note travels "
            "into the record, so it states what the arm IS as of "
            "2026-08-16 rather than what it was: symmetric trees at depth "
            "6, Cosine split scoring, CatBoost's random_strength noise, one "
            "leaf-estimation iteration, and the MVS bootstrap at 0.8 -- so "
            "it samples rows, which the two sentences that used to stand "
            "here denied. Both denials were true when written and became "
            "false the same day. What remains unmatchable is in "
            "scenarios.CATBOOST_UNMATCHABLE, and the arm is still not "
            "CatBoost: read that list before reading this row as a "
            "like-for-like."
        )
        return self


#: The name a re-encoded part carries in `worker.describe`'s `encoding`
#: block. One constant, so a reader grepping for it finds the encoder, the
#: record field and the scenario table entry that argues for it.
CATBOOST_CATEGORICAL_FORM = "mixed_frame_int64_categorical"


def _catboost_categorical_frame(part, cat_indices, form=CATBOOST_CATEGORICAL_FORM):
    """(frame, report). The canonical float64 matrix re-encoded so that
    CatBoost will accept `cat_features` over it, plus the proof that the
    re-encoding did not change a value.

    **Why this exists at all.** `catboost/core.py` refuses `cat_features`
    on a floating-point array by dtype and not by content:

        if (data.dtype.kind == 'f') and (cat_features is not None) \\
                and (len(cat_features) > 0):
            raise CatBoostError("'data' is numpy array of floating point
            numerical type, it means no categorical features, ...")

    read at catboost 1.2.10, core.py:804. The check is on `dtype.kind`, so
    it is not a check about the values and no value the harness could put in
    a float64 array passes it. The array has to change type, and only the
    categorical block can: the numeric columns are genuinely real-valued.
    That forces a MIXED container, which numpy does not have and pandas
    does, which is why this returns a DataFrame.

    **What is checked before anything is built.** Every declared categorical
    column must be finite and integral and inside int64, and the round trip
    `float64 -> int64 -> float64` must be BITWISE equal to the column it
    started from. All four hold for every categorical column any generator
    or loader in this harness produces, because they are integer codes; none
    of them is assumed. A column that fails any of them raises, and the
    important case is the first: a NaN fails `isfinite`, so
    `categorical_missing` cannot reach CatBoost through this function even
    if somebody flips its entry in `CATBOOST_SCENARIO_SUPPORT`. That is
    deliberate and `scenarios.CATBOOST_CATEGORICAL_ENCODING` carries the
    argument for why a missing category is not re-encodable rather than
    merely inconvenient.

    **What the report proves.** Column checks establish that each column
    survived. They do not establish that the FRAME holds the canonical
    matrix, which is a different claim and the one that matters: a mistake
    in column order, in a name, in a dtype or in the numeric passthrough
    would pass every column check. So the report also carries
    `canonical_digest_recomputed`: the canonical form reassembled out of the
    finished frame and hashed by `measure.canonical_digest`, which is the
    same function `worker.data_digest` calls and not a second one that
    looks like it. `worker.apply_encoding_report` compares it with the
    canonical digest and writes the verdict into the record. That is the
    whole digest argument, run per record instead of asserted here.

    Column names are `f{index}`, derived from `range(n_features)`. They are
    positional on purpose: `cat_features` stays a list of indices, the
    frame's column order IS the feature order, and nothing about the layout
    depends on a dict's iteration order or on a thread count.
    """
    x = part["X"]
    if hasattr(x, "tocsc"):
        raise EngineError(
            "the CatBoost arm cannot take categorical features on a sparse "
            "matrix. catboost/core.py refuses cat_features on a "
            "floating-point scipy.sparse.spmatrix by the same dtype rule it "
            "refuses a float ndarray by, and no scenario in this suite is "
            "both sparse and categorical"
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise EngineError(
            "the CatBoost arm needs pandas to hand CatBoost a mixed "
            "float-and-integer frame, because numpy has no mixed dtype and "
            "an object array of one million rows is not a container this "
            "harness will build. Install pandas in the bench environment or "
            "this scenario has no CatBoost row: " + str(exc)
        ) from exc

    n_rows, n_features = int(x.shape[0]), int(x.shape[1])
    cat = sorted({int(j) for j in cat_indices})
    for j in cat:
        if not 0 <= j < n_features:
            raise EngineError(
                f"categorical feature index {j} is outside the matrix's "
                f"{n_features} columns"
            )

    columns, checked = {}, []
    for j in range(n_features):
        column = np.ascontiguousarray(x[:, j], dtype=np.float64)
        if j not in cat:
            columns[f"f{j}"] = column
            continue
        finite = bool(np.isfinite(column).all())
        if not finite:
            n_missing = int((~np.isfinite(column)).sum())
            raise EngineError(
                f"declared categorical column {j} holds {n_missing} value(s) "
                "that are not finite, and CatBoost has no representation for "
                "a missing category. Its own message is \"cat_features must "
                "be integer or string, real number values and NaN values "
                "should be converted to string\", and converting one is a "
                "modelling decision the harness is not allowed to make "
                "silently: see scenarios.CATBOOST_CATEGORICAL_ENCODING"
                "['missing_categories']"
            )
        if not bool(np.array_equal(column, np.floor(column))):
            raise EngineError(
                f"declared categorical column {j} holds non-integral values, "
                "so it is not an integer-coded category column and CatBoost "
                "would be handed a rounded copy of it"
            )
        lo, hi = float(column.min()), float(column.max())
        if lo < np.iinfo(np.int64).min or hi > np.iinfo(np.int64).max:
            raise EngineError(
                f"declared categorical column {j} does not fit in int64 "
                f"({lo} .. {hi})"
            )
        codes = column.astype(np.int64)
        # The round trip, bitwise. `array_equal` on two float64 arrays with
        # no NaN in either is bit equality, and the `isfinite` gate above is
        # what makes that true rather than nearly true.
        if not bool(np.array_equal(codes.astype(np.float64), column)):
            raise EngineError(
                f"declared categorical column {j} does not survive "
                "float64 -> int64 -> float64, so the integer encoding is "
                "not the same data"
            )
        columns[f"f{j}"] = codes
        checked.append(j)

    frame = pd.DataFrame(columns, columns=[f"f{j}" for j in range(n_features)])
    # The canonical form REASSEMBLED from what CatBoost is about to be
    # handed: the matrix read back out of the finished frame, and the label
    # exactly as it goes to `Pool(label=...)`, which is not re-encoded at
    # all. Read out of the frame rather than out of the arrays that built it
    # so the digest is evidence about the container, which is where a
    # mistake in column order, in a name or in a dtype would live and where
    # a per-column check cannot see one.
    rebuilt = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    recomputed = measure.canonical_digest(
        rebuilt, part["y"], part.get("group")
    )
    del rebuilt

    report = {
        "form": form,
        "container": "pandas.DataFrame",
        "pandas_version": pd.__version__,
        "categorical_columns": list(checked),
        "categorical_dtype": "int64",
        "numeric_dtype": "float64",
        "canonical_digest_recomputed": recomputed,
        "proof": {
            "rows": n_rows,
            "features": n_features,
            "columns_round_tripped": len(checked),
            "round_trip_bitwise_equal": True,
            "finite": True,
            "integral": True,
            "checked": (
                "per column, in this run, before the frame was built: "
                "finite, integral, inside int64, and bitwise equal after "
                "float64 -> int64 -> float64. Then the whole frame was read "
                "back as float64, put beside the label exactly as Pool takes "
                "it, and hashed by measure.canonical_digest -- the same "
                "function worker.data_digest calls"
            ),
        },
        "why": (
            "catboost/core.py:804 refuses cat_features on a floating-point "
            "ndarray by dtype.kind, so no float64 array carries a "
            "categorical block however its values are coded. The numeric "
            "columns are real-valued and cannot become integers, so the "
            "container has to be mixed and numpy has no mixed dtype"
        ),
        "cost": (
            "a full extra copy of the matrix in the frame, plus a second "
            "transient copy for the reconstruction that is freed before the "
            "fit. peak_rss_bytes on a categorical CatBoost row therefore "
            "includes work the other two arms do not do, and it is a "
            "harness-conversion cost rather than a property of CatBoost"
        ),
    }
    return frame, report


class CatBoostEngine:
    """The CatBoost peer arm, reported beside the comparator.

    Three things about this adapter are different from the other two and all
    three are deliberate.

    **There is no separate binning phase.** `Pool(X, label=y)` is ingestion
    only: `Pool.is_quantized()` is False when it returns, and `quantize()`
    is a separate public call. Pre-quantizing to expose a binning number was
    tried and rejected, because it produces a different model above a few
    hundred thousand rows: at 300,000 rows by 20 features, a raw-pool fit, a
    default-seed quantized fit and a harness-seed quantized fit gave three
    distinct prediction digests. CatBoost draws its border-construction
    sample under the quantization seed, which the two paths do not share. So
    `binning` is null with that reason and the binning cost sits inside
    `train`, where CatBoost puts it. `scenarios.PHASE_SHAPE` states this for
    all three engines in one place so the three e2e lines can be read
    against each other.

    **Determinism is seeded, not guaranteed.** CatBoost has no
    `deterministic` flag. This arm pins `thread_count` and `random_seed` and
    that is the whole of it. `scenarios.CATBOOST_DETERMINISM` carries what
    was observed and what that does not establish, and every repeat records
    a prediction digest so the question is measured per run rather than
    inherited.

    **The resolved parameters are read back from the library.** CatBoost
    derives several of its defaults from the data -- `learning_rate` most
    importantly, which is why this arm pins it -- so the record carries
    `CatBoost.get_all_params()` from the fitted model rather than only the
    dict that was passed in. `get_all_params()` omits `thread_count`, so the
    adapter puts it back beside it.
    """

    name = "catboost"

    #: Parameters that define a CatBoost VARIANT row, merged into `extra` so
    #: they pass through `catboost_params`'s refusal list exactly like a
    #: scenario-supplied value. Empty here: the `catboost` row is CatBoost at
    #: its own defaults and that is the whole point of it. A subclass setting
    #: this is declaring a second, differently-shaped CatBoost column, and it
    #: must say so in `load()` so the note travels with the record.
    #:
    #: Merged rather than applied after the call on purpose. A dict written
    #: over the resolved params would bypass `CATBOOST_REFUSED_PARAMS`, and
    #: that list is the only thing standing between this harness and the
    #: `bin_construct_sample_cnt` defect in CatBoost's vocabulary.
    variant_params = {}

    def _extra_with_variant(self, extra):
        """`extra` plus this class's variant, or `extra` unchanged."""
        if not self.variant_params:
            return extra
        merged = dict(extra or {})
        merged.update(self.variant_params)
        return merged

    def __init__(self, threads, device="cpu"):
        self.threads = int(threads)
        self.device = device
        self.module = None
        self.version = None
        self.import_phase = None
        self.notes = []

    def load(self):
        if self.device != "cpu":
            raise EngineError(
                "this harness runs CatBoost on the CPU only. Its GPU "
                "training is a different quantization (border_count is "
                "capped at 255 on GPU against 65535 on CPU) as well as a "
                "different backend, so a CatBoost GPU row is not the same "
                "measurement as a CatBoost CPU row and would need its own "
                "label"
            )
        phase = measure.Phase("import")
        with phase:
            import catboost
        self.module = catboost
        self.import_phase = phase
        self.version = catboost.__version__
        try:
            scenarios.check_catboost_version(self.version)
        except RuntimeError as exc:
            raise EngineError(str(exc)) from exc
        self.notes.append(
            f"peer arm {scenarios.catboost_arm_id()}: "
            f"{scenarios.CATBOOST_ARM_LABEL}. Reported beside the "
            f"comparator {scenarios.comparator_id()} and never instead of "
            f"it. {scenarios.CATBOOST_ARM_REGISTERED}"
        )
        self.notes.append(
            "CatBoost determinism is "
            f"{scenarios.CATBOOST_DETERMINISM['status']}: "
            f"{scenarios.CATBOOST_DETERMINISM['flag']}. "
            f"{scenarios.CATBOOST_DETERMINISM['what_is_pinned']}."
        )
        self.notes.append(
            "CatBoost phases: " + scenarios.PHASE_SHAPE["catboost"]["e2e"]
            + ". " + scenarios.PHASE_SHAPE["catboost"]["binning"]
        )
        return self

    def _model(self, params):
        return self.module.CatBoost(params)

    def warmup(self, spec):
        x, y, _group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        params = scenarios.catboost_params(
            spec, self.threads, self._extra_with_variant(extra)
        )
        params["iterations"] = 1

        def _fit():
            pool = self.module.Pool(x, label=y)
            model = self._model(params)
            model.fit(pool)
            return model

        _, phase = measure.timed(_fit)
        return phase

    def _predict(self, model, task, matrix):
        """Predictions in the same shape the other two engines return.

        Regression is the raw value. Binary is the positive-class
        probability, which is column 1 of CatBoost's (n, 2) Probability
        output, matching LightGBM's and mojotrees's 1-D probability.
        Multiclass is the (n, k) probability matrix, which is what
        quality.multi_logloss reads. The conversion is inside the timed call
        on purpose: the other two engines return the probability from
        `predict` and pay for it there.
        """
        if task == "regression":
            return model.predict(matrix)
        probability = model.predict(matrix, prediction_type="Probability")
        if task == "binary":
            return np.asarray(probability)[:, 1]
        return np.asarray(probability)

    def run(self, spec, train, test, repeats=1):
        runs, reason = scenarios.catboost_supports(spec)
        if not runs:
            raise EngineError(
                f"the CatBoost peer arm does not run {spec['id']}: {reason}"
            )
        # The categorical block, if the scenario declares one.
        #
        # This used to be a blanket refusal, on the grounds that converting a
        # copy for CatBoost alone would break the data digest. That was the
        # right instinct and the wrong conclusion. The digest's job is to
        # prove the arms saw the same PROBLEM, and a re-encoding that is a
        # verified bijection on the values leaves the problem alone; what it
        # changes is the container. So the canonical digest is untouched and
        # unexempted, the re-encoding proves itself by reconstructing the
        # canonical matrix and hashing it back, and the record carries both
        # numbers. `_catboost_categorical_frame` is the encoder and
        # `scenarios.CATBOOST_CATEGORICAL_ENCODING` is the argument.
        #
        # The encoder refuses a column it cannot re-encode rather than
        # coercing it, which is what keeps `categorical_missing` out: a
        # missing category has no integer and no string that means absence
        # to CatBoost the way bin 0 means it to the other two.
        cat_indices = list(train.get("categorical_feature") or ())
        encoding_report = None
        encode = None
        train_matrix, test_matrix = train["X"], test["X"]
        if cat_indices:
            def _encode():
                tr, tr_report = _catboost_categorical_frame(train, cat_indices)
                te, te_report = _catboost_categorical_frame(test, cat_indices)
                return tr, te, tr_report, te_report

            # Timed, because it is real work CatBoost's row would not exist
            # without, and named separately because the other two arms do not
            # do it. `scenarios.PHASE_SHAPE` says where it sits in e2e and
            # that it is a harness-conversion cost rather than a CatBoost one,
            # so a reader can subtract it. The reconstruction digest is
            # computed inside it and is bookkeeping rather than work; it is
            # inside the timing anyway rather than being carved out, because
            # a phase that excludes part of what it did is worse than one
            # that overstates by a hash.
            (train_matrix, test_matrix, tr_report, te_report), encode = \
                measure.timed(_encode)
            encoding_report = {"train": tr_report, "test": te_report}
            self.notes.append(
                "this scenario declares categorical features and this arm "
                f"took them: {len(cat_indices)} column(s) re-encoded as "
                "int64 in a mixed pandas frame, and CatBoost given "
                "cat_features over them, so its ordered target statistic is "
                "what ran. The canonical float64 digest is unchanged and "
                "unexempted; data.train.encoding carries the reconstruction "
                "digest and the verdict. See "
                "scenarios.CATBOOST_CATEGORICAL_ENCODING."
            )

        extra = None
        if spec["task"] == "multiclass":
            extra = {
                "num_class": int(
                    train.get("n_classes") or (np.max(train["y"]) + 1)
                )
            }
        params = scenarios.catboost_params(
            spec, self.threads, self._extra_with_variant(extra)
        )

        # Ingestion, and only ingestion. See the class docstring and
        # scenarios.PHASE_SHAPE for why there is no binning phase beside it.
        # `cat_features` is passed as indices, which is why the encoder names
        # the frame's columns positionally.
        pool, ingest = measure.timed(
            lambda: self.module.Pool(
                train_matrix,
                label=train["y"],
                cat_features=(cat_indices or None),
            )
        )
        model = self._model(params)
        _, training = measure.timed(lambda: model.fit(pool))

        task = spec["task"]
        predictions, predict_batch = measure.repeat(
            lambda: self._predict(model, task, test_matrix), repeats
        )
        # One row, in whatever container the batch was predicted from. A
        # model fitted with cat_features must be predicted from a frame with
        # the same columns and the same dtypes, so slicing the float64 matrix
        # here would measure a call that raises rather than a prediction.
        if cat_indices:
            row = test_matrix.iloc[:1]
        else:
            row = test_matrix[:1]
            if not train.get("sparse"):
                row = np.ascontiguousarray(row)
        _, predict_row = measure.repeat(
            lambda: self._predict(model, task, row), 20, warmup=2
        )

        with tempfile.TemporaryDirectory() as scratch:
            size = measure.model_size(model, scratch)

        # The library's own resolved parameter list, not a restatement of
        # what was passed. CatBoost derives defaults from the data, so this
        # is the only place a record can say what actually ran.
        # `thread_count` is not in it and is put back beside it.
        try:
            resolved = dict(model.get_all_params())
            resolved["thread_count"] = int(self.threads)
            resolved_note = (
                "CatBoost.get_all_params() on the fitted model, plus "
                "thread_count, which get_all_params omits"
            )
        except Exception as exc:  # pragma: no cover - engine-dependent
            resolved = None
            resolved_note = (
                f"get_all_params() failed: {type(exc).__name__}: {exc}"
            )

        bins = _catboost_bin_profile(model, train["X"].shape[1])
        num_bin = bins["max"] if isinstance(bins, dict) and "max" in bins else None
        model_block = {
            "num_trees": int(model.tree_count_),
            "current_iteration": int(model.tree_count_),
            "num_bin": num_bin,
            "bins": bins,
            "size": size,
        }
        if num_bin is None:
            model_block["num_bin_unavailable_reason"] = (
                "CatBoost exposes borders per feature and only after a fit, "
                "and the per-feature read failed: "
                + str(bins.get("unavailable_reason"))
            )
        if spec["task"] == "multiclass":
            model_block["num_trees_note"] = (
                scenarios.CATBOOST_UNMATCHABLE["multiclass_tree_count"]
            )
        if cat_indices:
            model_block["bins_note"] = (
                "read this per-feature profile against the other two "
                "engines' with care on a categorical scenario, and probably "
                "not at all. CatBoost.get_borders() returns borders for "
                "CatBoost's FLOAT features, and a model fitted with "
                "cat_features has float features this harness never handed "
                "it: the CTR columns it derives from the categorical block. "
                "Whether get_borders' keys index the original columns, "
                "CatBoost's internal float-feature order, or the CTR "
                "features as well is NOT established here, so the vector "
                "above is recorded as read and is not claimed to be one "
                "entry per input column. The numeric columns of both "
                "categorical scenarios happen to come first, which makes a "
                "coincidental alignment likely and does not make it a fact"
            )

        return {
            "engine": self.name,
            "engine_version": self.version,
            "device_requested": self.device,
            "device_used": "cpu",
            "path": "pool",
            "phases": {
                "import": self.import_phase.as_dict(),
                # A phase the other two records do not carry, because
                # CatBoost is the only engine here whose ingestion is a
                # separate public step from its binning. LightGBM's is
                # inside Dataset.construct() and mojotrees's is its
                # transpose, which bench_train.mojo times as ingest_s.
                "ingest": ingest.as_dict(),
                # Present only on a categorical scenario, and null with a
                # reason on every other, so a reader never has to read an
                # absent field as a zero. It is the harness re-encoding the
                # canonical float64 matrix into the mixed frame CatBoost
                # will take cat_features over, plus the reconstruction hash
                # that proves it. Counted in e2e, because it is work this
                # row does not exist without; labelled a conversion cost,
                # because a CatBoost user whose categories already live in
                # an integer column pays none of it.
                "encode": encode.as_dict() if encode is not None else None,
                "encode_unavailable_reason": (
                    None if encode is not None else
                    "this scenario declares no categorical features, so "
                    "there is nothing to re-encode and the canonical "
                    "float64 matrix goes straight into the Pool"
                ),
                "binning": None,
                "binning_unavailable_reason": (
                    scenarios.PHASE_SHAPE["catboost"]["binning"]
                    + ". Pre-quantizing at 300,000 rows by 20 features gave "
                    "a different prediction digest from a raw-pool fit, and "
                    "a third one again under a different quantization seed, "
                    "so the split would change the model rather than only "
                    "the accounting"
                ),
                "train": training.as_dict(),
                "predict_batch": predict_batch,
                "predict_row": predict_row,
            },
            "params_used": params,
            "dataset_params_used": None,
            "dataset_params_unavailable_reason": (
                "CatBoost's Pool takes the data and the fit takes the "
                "parameters. The binning settings are training parameters "
                "and are in params_used, left at CatBoost's own defaults"
            ),
            "engine_resolved_params": resolved,
            "engine_resolved_params_source": resolved_note,
            "num_boost_round": int(params["iterations"]),
            "histogram_builder": dict(CATBOOST_HISTOGRAM_BUILDER),
            "model": model_block,
            # Popped by worker.run_job and folded into data.train / data.test
            # beside the canonical digest, because the two fields only mean
            # anything read together. None on a numeric scenario, where the
            # engine took the canonical form and the default block
            # worker.describe writes is already correct.
            "data_encoding": encoding_report,
            "categorical_features": list(cat_indices) or None,
            "transfers": measure.unavailable("cpu-only path"),
            "peak_rss_bytes": measure.peak_rss_bytes(),
            "notes": list(self.notes),
            "predictions": predictions,
        }


class CatBoostLossguideEngine(CatBoostEngine):
    """CatBoost grown leaf-wise instead of symmetric. The second CatBoost row.

    Everything is inherited; the only difference is `variant_params`, so the
    two CatBoost rows run through identical code and the difference between
    two records is exactly that dict.

    **What this row is for.** The `catboost` row is CatBoost at its own
    defaults, which means SymmetricTree at depth 6 -- a shape neither of the
    other two engines grows. That makes it a fair reading of CatBoost and a
    poor reading of the growth policy: any gap between it and the leaf-wise
    arms mixes CatBoost's engine with CatBoost's tree shape, and nothing in
    the record separates them. Lossguide at `max_leaves` 31 is CatBoost
    growing the same shape LightGBM and mojotrees grow by default, so the two
    CatBoost rows read together isolate the tree shape inside one engine.

    **Why 31.** It is LightGBM's `num_leaves` default and this harness's
    shared value, so this row's leaf budget matches the comparator's rather
    than matching the 64 that CatBoost's depth 6 resolves to. The intent is
    one variable, and the variable is the growth policy.

    **It is a separate engine name** for the reason
    `MojoTreesCatBoostModeEngine` gives: the harness's unit of comparison is
    an engine name -- what `verify.py` pairs on, what `report.py` groups on,
    what a CSV row carries -- so two differently-shaped CatBoost models must
    not share a column.

    **It is not a clean isolation and the record must not claim it is.**
    Verified by a 3-iteration fit on 2026-08-16: CatBoost accepts the pair
    and `get_all_params()` reads back `grow_policy=Lossguide`,
    `max_leaves=31`, with `score_function=Cosine` and `bootstrap_type=MVS`
    still stock, which is the intent. But it also reads back **`depth=6`**.
    CatBoost keeps its depth default as an active cap under Lossguide, where
    LightGBM's `max_depth` default is -1, unlimited. So this row is leaf-wise
    up to 31 leaves AND depth-capped at 6, and a leaf-wise tree that wanted a
    deeper path does not get it. Whether 31 leaves ever reaches past depth 6
    on these scenarios is not established here. The cap is left in place
    rather than overridden, because pinning `depth` would make this a row
    with three changed parameters instead of two.

    **If CatBoost refuses the combination**, that is what the row records.
    CatBoost resolves `score_function` and several sampling defaults itself
    and some of those resolutions are policy-dependent, so this arm passes
    exactly two parameters and lets the library raise if it will not accept
    them. The failure is reported as a failure. It is NOT to be papered over
    by pinning a third parameter until the fit succeeds: that would quietly
    turn this into a row comparing two things at once, which is the defect
    the row exists to remove.
    """

    name = "catboost_lossguide"
    variant_params = {"grow_policy": "Lossguide", "max_leaves": 31}

    def load(self):
        super().load()
        self.notes.append(
            "arm 'CatBoost lossguide': the CatBoost peer arm with "
            f"{self.variant_params} applied over its defaults, so it grows "
            "leaf-wise like LightGBM and mojotrees instead of its default "
            "SymmetricTree at depth 6. Read it against the 'catboost' row to "
            "separate CatBoost's engine from CatBoost's tree shape; read "
            "neither instead of the comparator "
            f"{scenarios.comparator_id()}. Everything else is stock, "
            "including score_function and the MVS bootstrap, so this row "
            "still carries every entry in scenarios.CATBOOST_UNMATCHABLE "
            "except tree_shape. tree_shape is NARROWED here, not closed: "
            "CatBoost keeps depth=6 as an active cap under Lossguide where "
            "LightGBM's max_depth default is unlimited, so this arm is "
            "leaf-wise to 31 leaves AND depth-capped at 6. Read the "
            "resolved parameters in this record rather than assuming the "
            "growth policy is the only difference left."
        )
        return self


ENGINES = {
    "mojotrees": MojoTreesEngine,
    "lightgbm": LightGBMEngine,
    # Peer arms, reported beside the comparator. Adding them here is what
    # makes worker.py able to run them without a change: it builds an engine
    # by name from this table.
    "catboost": CatBoostEngine,
    "catboost_lossguide": CatBoostLossguideEngine,
    "mojotrees_catboost_mode": MojoTreesCatBoostModeEngine,
}

#: Which arm each engine belongs to. The comparator row is LightGBM against
#: the plain mojotrees arm and nothing here changes that; the peer entries
#: exist so a reader of a record can tell a headline row from a peer row
#: without knowing the engine names by heart.
ENGINE_ARM = {
    "mojotrees": "subject",
    "lightgbm": "comparator",
    "catboost": "peer",
    "catboost_lossguide": "peer",
    "mojotrees_catboost_mode": "peer_subject",
}


def build(name, threads, device):
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(sorted(ENGINES))}")
    return ENGINES[name](threads, device)
