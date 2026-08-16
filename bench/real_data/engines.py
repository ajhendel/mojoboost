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
            "mojotrees arm is what is read against it. mojotrees has no "
            "symmetric-tree policy, so this arm is depthwise at depth 6 and "
            "is not CatBoost's tree; it also does no row sampling, where "
            "CatBoost's default MVS bootstrap takes 80 percent of the rows "
            "per tree. See scenarios.CATBOOST_UNMATCHABLE."
        )
        return self


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
        params = scenarios.catboost_params(spec, self.threads, extra)
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
        if train.get("categorical_feature"):
            # Never reached from the scenario table, and refused anyway. The
            # harness hands every engine one float64 matrix, and CatBoost
            # rejects cat_features on a floating-point array; converting a
            # copy for CatBoost alone would break the data digest that makes
            # the three records comparable in the first place.
            raise EngineError(
                "this scenario declares categorical features and the "
                "CatBoost arm cannot take them from the harness's float64 "
                "matrix: "
                + str(scenarios.CATBOOST_SCENARIO_SUPPORT.get(
                    "categorical_missing"
                ))
            )

        extra = None
        if spec["task"] == "multiclass":
            extra = {
                "num_class": int(
                    train.get("n_classes") or (np.max(train["y"]) + 1)
                )
            }
        params = scenarios.catboost_params(spec, self.threads, extra)

        # Ingestion, and only ingestion. See the class docstring and
        # scenarios.PHASE_SHAPE for why there is no binning phase beside it.
        pool, ingest = measure.timed(
            lambda: self.module.Pool(train["X"], label=train["y"])
        )
        model = self._model(params)
        _, training = measure.timed(lambda: model.fit(pool))

        task = spec["task"]
        predictions, predict_batch = measure.repeat(
            lambda: self._predict(model, task, test["X"]), repeats
        )
        row = test["X"][:1]
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
            "transfers": measure.unavailable("cpu-only path"),
            "peak_rss_bytes": measure.peak_rss_bytes(),
            "notes": list(self.notes),
            "predictions": predictions,
        }


ENGINES = {
    "mojotrees": MojoTreesEngine,
    "lightgbm": LightGBMEngine,
    # Peer arms, reported beside the comparator. Adding them here is what
    # makes worker.py able to run them without a change: it builds an engine
    # by name from this table.
    "catboost": CatBoostEngine,
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
    "mojotrees_catboost_mode": "peer_subject",
}


def build(name, threads, device):
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(sorted(ENGINES))}")
    return ENGINES[name](threads, device)
