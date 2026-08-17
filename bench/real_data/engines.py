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

import json
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


#: The arm's overrides, held as class defaults so that an engine constructed
#: directly -- selfcheck does, and so does anybody debugging one cell -- reads
#: the same empty dicts `build` would have given it. `None` rather than `{}`
#: because a shared mutable class attribute is one `.update()` away from
#: leaking one arm into every later engine in the process.
#:
#: `arm_params` are TRAINING parameters and `arm_dataset_params` are BINNING
#: ones. The split is not cosmetic: both libraries reject `max_bin` on
#: `train`, which is why `scenarios.dataset_params` exists at all, and a
#: max_bin override folded into the training dict kills the cell.
ARM_OVERRIDE_ATTRIBUTES = ("arm_params", "arm_dataset_params")


def _arm(engine, which):
    """One of an engine's arm override dicts, as a fresh dict, never None."""
    return dict(getattr(engine, which, None) or {})


class MojoTreesEngine:
    name = "mojotrees"

    arm_params = None
    arm_dataset_params = None

    #: The translator this arm's parameters go through. The CatBoost-mode
    #: arm below overrides exactly this and nothing else, so the two arms
    #: run the same code and a reader diffing two records sees
    #: `scenarios.MOJOTREES_CATBOOST_MODE` and nothing else.
    params_fn = staticmethod(scenarios.mojotrees_params)

    def __init__(self, threads, device="cpu", catboost_readback=None):
        self.threads = int(threads)
        self.device = device
        self.module = None
        self.version = None
        self.import_phase = None
        self.notes = []
        # CatBoost's resolved parameters for the cells this run has already
        # measured, or None. Unused by this arm and read by the CatBoost-mode
        # subclass, which cannot be built without it.
        self.catboost_readback = catboost_readback

    def _arm_extra(self, extra=None):
        """`extra` with the arm's TRAINING overrides folded in.

        `scenarios.mojotrees_params` copies every `extra` key onto the
        translated dict and `scenarios.shared_params` merges the same dict
        over `BASE_PARAMS`, so this is the hook the arm dimension was designed
        around rather than a new one.

        `n_estimators` is excluded here and set explicitly by `_n_estimators`,
        because the tree count is also the `num_boost_round` argument and the
        record's `num_boost_round` field, and one value reaching three places
        by three routes is how the three come apart.
        """
        merged = dict(extra or {})
        for key, value in _arm(self, "arm_params").items():
            if key == "n_estimators":
                continue
            merged[key] = value
        return merged or None

    def _n_estimators(self):
        """The arm's tree count, or the base one.

        **This is the only place the tree count is decided on this arm.** It
        used to read `scenarios.BASE_PARAMS["n_estimators"]` directly at two
        call sites, which is why a sweep over tree counts could not be
        scheduled at all: the one parameter a frontier moves first was the one
        parameter no caller could override.
        """
        return int(
            _arm(self, "arm_params").get(
                "n_estimators", scenarios.BASE_PARAMS["n_estimators"]
            )
        )

    def _dataset_params(self, spec):
        """The binning parameters, with the arm's overrides.

        The `max_bin` axis moves the BINNING and not the data, so the
        canonical data digest is unaffected and `verify.check_data_agreement`
        stays meaningful across arms at different bin budgets. That property
        is what makes this override safe to have at all.
        """
        params = scenarios.dataset_params(spec)
        params.update(_arm(self, "arm_dataset_params"))
        return params

    def _params(self, spec, extra=None, device=None):
        """The translated parameter dict for this arm.

        A method rather than the `type(self).params_fn(...)` call it replaces,
        because the CatBoost-mode subclass needs instance state -- CatBoost's
        read-back for this cell -- and a `staticmethod` cannot see it. Every
        call site goes through here so that the subclass overrides one thing.
        """
        return type(self).params_fn(spec, device or self.device, extra)

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
        """One round on a tiny matrix, through the arm's own shape.

        The arm's training and binning overrides are applied, so that what is
        warmed is the code path about to be measured: an arm at
        `grow_policy='symmetrictree'` warmed through the leaf-wise grower
        would leave the measured fit paying first-call costs that the record
        then attributes to training.

        `auto_learning_rate` is the one override dropped, and it is dropped
        rather than passed because CatBoost's derivation reads `log` of the
        iteration count and this call runs ONE round. A rate derived at one
        round is not this arm's rate, and the same code is warmed either way.
        """
        x, y, group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        extra = self._arm_extra(extra) or {}
        extra.pop("auto_learning_rate", None)
        params = self._params(spec, extra or None)
        params["n_estimators"] = 1

        def _fit():
            data = self.module.Dataset(
                x, label=y, group=group, params=self._dataset_params(spec)
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
        params = self._params(spec, self._arm_extra(extra))
        params["n_estimators"] = self._n_estimators()

        dataset_params = self._dataset_params(spec)
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
        params = self._params(spec, self._arm_extra(), device="cpu")
        params.pop("device", None)
        # Built once and both passed and recorded, so the record is the
        # call rather than a reconstruction of it. `objective` is dropped
        # because the classifier implies it; the record's shared block
        # still carries it. A collision raises rather than resolving
        # itself, which is what the keyword form this replaced did.
        estimator_params = {
            "n_estimators": self._n_estimators(),
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
        # The arm's BINNING overrides, applied last. There is no second dict
        # on this path: the estimator takes the binning settings as keywords,
        # so `max_bin` lands here beside the training ones instead of on a
        # Dataset, and `dataset_params_used` stays None with the reason it
        # already carried. The arm's TRAINING overrides are already in
        # `params` above, through `_arm_extra`.
        estimator_params.update(_arm(self, "arm_dataset_params"))
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

    arm_params = None
    arm_dataset_params = None

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
        # The arm's overrides. LightGBM takes the binning settings in the same
        # dict as the training ones, so both override dicts land here and
        # `dataset_params_used` stays None with the reason it always carried.
        extra.update(_arm(self, "arm_params"))
        extra.update(_arm(self, "arm_dataset_params"))
        params = scenarios.lightgbm_params(spec, self.threads, extra)
        # The comparator's tree count, which used to be read straight off
        # `BASE_PARAMS` here and so could not be moved by anything. `extra`
        # reaches `scenarios.shared_params`, which merges it over
        # `BASE_PARAMS`, so the count inside `params` and the count passed as
        # `num_boost_round` are one value read once rather than two constants
        # that happen to agree.
        rounds = int(
            params.get("n_estimators", scenarios.BASE_PARAMS["n_estimators"])
        )

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

    def _params(self, spec, extra=None, device=None):
        """The one override, and the reason this class needs a method at all.

        This arm's `learning_rate` is not a constant. CatBoost derives its own
        from the iteration count and the dataset, `cb-shipped` stopped pinning
        it, and "us in CatBoost's shape" is false if this side runs 0.1 while
        CatBoost runs the rate it chose. So the value comes from CatBoost's
        `get_all_params()` for the SAME cell, handed over through the run's
        `catboost_readback.json`.

        There is no fallback and there must not be one.
        `scenarios.mojotrees_catboost_mode_params` raises
        `CatBoostReadbackMissing` by name when the read-back is absent or is
        for another cell, and that exception is allowed out of here: a cell
        that trained on a guessed rate under this heading is the defect the
        whole arm was rebuilt to remove, and it would be invisible in the
        record.
        """
        return scenarios.mojotrees_catboost_mode_params(
            spec,
            device or self.device,
            extra,
            catboost_readback=self.catboost_readback,
        )

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

    read at catboost 1.2.10, catboost/core.py:804. The check is on `dtype.kind`, so
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

    arm_params = None
    arm_dataset_params = None

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
        # The arm's TRAINING overrides. `n_estimators` is the one the frontier
        # moves: `scenarios.catboost_params` reads
        # `shared[CATBOOST_MATCHED["iterations"]]`, which is `n_estimators`
        # merged over `BASE_PARAMS` by `shared_params`, so an override here
        # moves CatBoost's `iterations` and the record's `num_boost_round`
        # together. Every other key goes through
        # `scenarios.CATBOOST_REFUSED_PARAMS`, which refuses by name anything
        # that would make this arm stop being CatBoost at stock.
        #
        # The arm's BINNING overrides are REFUSED rather than translated.
        # CatBoost's counterpart of `max_bin` is `border_count`, and the two
        # are not a rename of each other: border_count counts THRESHOLDS where
        # max_bin counts BINS (scenarios.CATBOOST_PARAM_MAP["border_count"],
        # translate borders_to_bins, +1 and not identity), so a max_bin axis
        # measured by moving one library's number and not the other's is a
        # comparison of two different sweeps. Corrected 2026-08-17: this
        # comment and the message below said the default was 65535. That is
        # CatBoost's MAXIMUM (GetMaxBinCount, the CPU ceiling; 255 on GPU);
        # the CPU default is 254 borders, which is 255 bins and the same
        # granularity budget as LightGBM's max_bin=255.
        # `scenarios.CATBOOST_REFUSED_PARAMS` refuses `max_bin` at the call
        # below anyway; the message here says why rather than leaving a
        # reader with a KeyError-shaped one.
        arm_binning = _arm(self, "arm_dataset_params")
        if arm_binning:
            raise EngineError(
                "the CatBoost arm takes no per-arm binning overrides "
                f"({', '.join(sorted(arm_binning))}). CatBoost's bin budget "
                "is border_count, which counts thresholds where max_bin "
                "counts bins (CPU default 254 borders = 255 bins; 65535 is "
                "its ceiling, not its default), so moving max_bin on one arm "
                "and not the other is two sweeps rather than one axis. See "
                "scenarios.CATBOOST_REFUSED_PARAMS and "
                "scenarios.CATBOOST_LEFT_AT_STOCK"
            )
        arm_training = _arm(self, "arm_params")
        if arm_training:
            extra = dict(extra or {})
            extra.update(arm_training)
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

        # Where a LIVE fit disagrees with scenarios.CATBOOST_LEFT_AT_STOCK,
        # which is a transcription somebody made on 2026-08-16 and which
        # nothing had ever contradicted because nothing had ever read it back.
        # Recorded rather than raised: a CatBoost upgrade moving a default is
        # a thing to find out about from a record, not a reason to lose a
        # cell.
        readback_drift = (
            scenarios.check_catboost_readback(resolved, spec, self.threads)
            if resolved
            else ["get_all_params() produced nothing: " + str(resolved_note)]
        )
        # The handover. Popped by worker.run_job into the run's
        # catboost_readback.json so that the mojotrees_catboost_mode cell for
        # this same cell can take CatBoost's resolved learning rate. Not a
        # report field.
        #
        # SUPPRESSED ON A VARIANT ROW, and this is the collision the sidecar
        # made reachable. A subclass that pins parameters over CatBoost's
        # defaults inherits this whole method and would write under the same
        # cell key, because the key is (scenario, tier, variant) and not the
        # engine. Such a row's resolved dict is NOT the one the CatBoost-mode
        # arm is shaped after, so whichever of the two rows happened to run
        # last would decide what "us in CatBoost's shape" meant. The default
        # row is the only one that answers that question.
        #
        # THE ARM THAT MOTIVATED THIS GUARD IS GONE as of 2026-08-17. It was
        # `catboost_lossguide`, CatBoost pinned to grow_policy=Lossguide and
        # max_leaves=31, and Andrew removed it because it answered a question
        # about CatBoost's tree shape that he is not asking. The guard stays,
        # and deliberately: `variant_params` is still the extension point a
        # future CatBoost variant row would use, and the collision it prevents
        # is a silently wrong learning rate in the CatBoost-mode arm rather
        # than a visible failure. A guard whose only caller was deleted is
        # cheap; rediscovering this collision is not.
        readback_entry = None
        if not self.variant_params:
            readback_entry = scenarios.catboost_readback_entry(
                spec, resolved or {}, self.version
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
            # Empty list means a live fit agreed with this harness's
            # transcription of CatBoost's defaults. A non-empty one means the
            # transcription is wrong or CatBoost has moved, and either way the
            # parity table beside it is reasoning from a stale premise.
            "engine_resolved_params_drift": readback_drift,
            # Popped by worker.run_job. See the comment where it is built.
            "catboost_readback": readback_entry,
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


class MojoTreesDepthwiseEngine(MojoTreesEngine):
    """mojotrees wearing XGBoost's shipped defaults. The XGBoost mirror row.

    Everything is inherited. The only difference from the plain arm is the
    translator, which applies `scenarios.MOJOTREES_DEPTHWISE` over the shared
    parameters, so the two arms run through identical code and the difference
    between two records is exactly that dict.

    **WHAT THIS ARM IS CHANGED ON 2026-08-17, AND ITS NAME DID NOT.** It was a
    growth-ORDER isolation row: `{"grow_policy": "depthwise"}` and nothing
    else, read against the plain mojotrees arm, existing to price the growth
    order by itself. Andrew redirected it the same day, in these words: "make
    our depthwise params match xgboost". It now carries XGBoost 3.4.0's
    resolved defaults in our parameter names and is read against the `xgboost`
    peer column. The name stayed because it is the cell key, the record field,
    the output filename, and what `verify.py` and `report.py` group on, so
    renaming it would orphan every record already written under it.
    `scenarios.MOJOTREES_DEPTHWISE` and `scenarios.MOJOTREES_DEPTHWISE_CLAIMS`
    carry the correction, and the claims string travels into the record so a
    reader is told rather than left to infer the arm from its name.

    **A RECORD WRITTEN BEFORE 2026-08-17 IS A DIFFERENT ARM UNDER THIS NAME**
    and the two must not be put in one series.

    It is a separate engine name rather than a flag on the existing one for
    the reason `MojoTreesCatBoostModeEngine` gives: the harness's unit of
    comparison is an engine name, it is what `verify.py` pairs on, what
    `report.py` groups on and what a CSV row carries, so two differently
    shaped models must not share a column.

    **WHY IT IS NOT A PEER ARM.** `scenarios.PEER_ENGINES` members are
    asserted by `selfcheck.py` to carry an `ENGINE_ARM` beginning with
    "peer", and a peer is a competitor library reported beside the
    comparator. This is mojotrees, so it is a subject, and it is recorded as
    `subject_variant` rather than plain `subject` because it is NOT the
    headline row either. The headline is the plain mojotrees arm against
    `stock+det` and nothing here changes that. Being a subject is also what
    keeps it inside `verify.SUBJECT_ENGINES`, and so inside the backend proof
    and the cpu-versus-gpu agreement check, which a peer column does not get.

    **WHAT IT IS FOR, IN TWO PARTS NOW.** The first is unchanged and still
    unmeasured: the device path batches a level into one host wait instead of
    one wait per split (`train_gpu._device_search_resident`, with
    `train_gpu.mojo::_search_record_slots` sizing a level's worth of records for exactly this
    mode), while leaf-wise growth cannot batch at all because its next pick
    depends on the frontier the current split just changed
    (`growth_policy.mojo` module docstring, "What a batched level is, and
    what it is not"), so a depth-wise fit is EXPECTED to pay
    fewer host waits per tree. The second is why the arm was widened: it is our
    side of the XGBoost pairing, the same relationship
    `mojotrees_catboost_mode` has to the `catboost` column, and a competitor
    column is only interpretable beside an arm of ours wearing the same shape.

    **WHAT IS NO LONGER AVAILABLE.** The growth order can no longer be priced
    on its own, because no arm now differs from the plain one in the order
    alone. That was a real measurement and it is gone; restoring it means
    restoring an arm, which costs a column of cells in every scenario, tier and
    backend.

    **NO CROSS-CELL DEPENDENCY**, unlike the CatBoost-mode arm, which cannot be
    built until the `catboost` cell for the same cell key has written the
    learning rate CatBoost derived. XGBoost's defaults are static, so this arm
    reads nothing from another cell and can run alone. It is still ranked in
    `run.CELL_ORDER` behind the peers, and that comment says why.

    **WHAT IT MAY NOT CLAIM** is in `scenarios.MOJOTREES_DEPTHWISE_CLAIMS`,
    which travels into the record. In short: read it against the xgboost peer
    column and not against the comparator; it is not a LightGBM parity row,
    because LightGBM has no growth policy at all; the tree it grows is a
    different tree from every other arm's, so an accuracy difference is a
    property of the shape rather than a defect; and on the categorical and
    ranking scenarios there is no xgboost column at all, so those cells have no
    peer to be read against.
    """

    name = "mojotrees_depthwise"
    params_fn = staticmethod(scenarios.mojotrees_depthwise_params)

    def load(self):
        super().load()
        self.notes.append(scenarios.MOJOTREES_DEPTHWISE_CLAIMS)
        return self


class MojoTreesSymmetricColsampleEngine(MojoTreesEngine):
    """mojotrees growing SYMMETRIC trees under per-tree column sampling.

    A CORRECTNESS arm, which is a third kind of arm in this file and the
    first of it. The other two mojotrees variants are MIRRORS:
    `mojotrees_depthwise` wears XGBoost's defaults and
    `mojotrees_catboost_mode` wears CatBoost's, and each is read against a
    peer column. This one is read against NOTHING external. Its product is
    `verify.check_device_agreement`, which compares its gpu predictions
    against its own cpu twin's row by row, and the pair is the whole reason
    the arm has cells.

    **WHY IT EXISTS.** On 2026-08-17 the oblivious GPU path was found
    returning a wrong answer whenever `feature_fraction < 1`, because
    `GpuLeafBatcher.feat_dev` was never set by the level build, so the
    histogram wrote feature slice `slot` while the split searcher read slice
    `active[slot]` and the root disagreed with its own children. It was found
    by reading code. It could not have been found here, because every
    scenario in `scenarios.py` runs `feature_fraction = 1.0`, where the
    feature table is the identity and the wrong read is the right one. The
    fix is compiled into the extension this arm runs against, so this arm is
    expected to be GREEN; a green row is the statement that the fix is still
    in, and that is what a regression detector is.

    Everything is inherited except the translator, which applies
    `scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE` over the shared parameters, so
    this arm and the plain one run identical code and the difference between
    two records is exactly that dict.

    **A SEPARATE ENGINE NAME rather than a flag**, for the reason
    `MojoTreesCatBoostModeEngine` gives and for one more that is specific to
    this arm. The general reason: the harness's unit of comparison is an
    engine name, it is what `verify.py` pairs on and what a CSV row carries,
    so two differently shaped models must not share a column. The specific
    one, AS IT STOOD when this arm was written: `verify.check_device_agreement`
    keyed on `(scenario, threads, engine)` and NOT on `arm`, so two arms of
    one engine on one scenario overwrote each other in its inner dict and it
    would compare one arm's gpu predictions against a different arm's cpu
    predictions. An `--arms` cell would therefore have been wired into the
    very check this arm exists to feed, and wired wrong. That is recorded here
    because it is the reason a frontier-style arm was not used and it is not
    obvious from either file. LATER ON 2026-08-17 that check, and every other
    per-cell key in verify.py, report.py and summarize.py, moved to the arm
    (`verify._arm_of`), so an `--arms` cell now feeds it correctly too. The
    separate engine name stays, for the general reason above and because the
    engine name is still what thresholds.json, `ENGINE_ARM` and
    `SUBJECT_ENGINES` are keyed by.

    **`subject_variant`, so it stays inside `verify.SUBJECT_ENGINES`**, which
    is what gets it the backend proof and the cpu-versus-gpu agreement check
    that a peer column does not get. For this arm that is not a nice-to-have,
    it is the arm.

    **WHAT IT MAY NOT CLAIM** is in
    `scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE_CLAIMS`, which travels into the
    record. In short: there is no peer column and none is coming; the metric
    is not comparable with any arm at `feature_fraction = 1.0`, because
    column sampling is a regularizer; and the tree is symmetric at depth 6,
    which is a different tree from the plain arm's.
    """

    name = "mojotrees_symmetric_colsample"
    params_fn = staticmethod(scenarios.mojotrees_symmetric_colsample_params)

    def load(self):
        super().load()
        self.notes.append(scenarios.MOJOTREES_SYMMETRIC_COLSAMPLE_CLAIMS)
        return self


class MojoTreesCosineLeafwiseEngine(MojoTreesEngine):
    """mojotrees scoring splits with Cosine under LEAF-WISE growth.

    The second CORRECTNESS arm. Read
    `MojoTreesSymmetricColsampleEngine`'s docstring first: everything it says
    about what a correctness arm is, why it is a separate engine name rather
    than an `--arms` cell, and why it is `subject_variant`, is true of this
    one word for word and is not repeated.

    **WHY IT EXISTS.** On 2026-08-17 `score_function='Cosine'` was found to
    be silently ignored on every leaf-wise GPU fit.
    `GpuSplitSearcher.set_score_function` had no production caller anywhere
    in the package, so `score_function_code` stayed at its constructed
    `SCORE_L2`, and `gpu_resident_round._launch_child_search` called
    `_launch_search` without naming the argument and took the same default.
    Both halves were needed and both are fixed.

    **WHY THIS HARNESS DID NOT CATCH IT, which is the part worth keeping.**
    One arm here does set Cosine: `mojotrees_catboost_mode`. It also sets
    `grow_policy='symmetrictree'`, so it goes down the oblivious device path
    and never reaches the leaf-wise one the bug lived on. A parameter being
    covered by an arm is therefore not the same as the code path being
    covered, and the difference was invisible from a green suite. That is why
    this arm names `grow_policy='lossguide'` explicitly even though it is
    already the estimator's default: the arm's value is which path it
    reaches, so it must say which path.

    **WHAT IT MAY NOT CLAIM** is in
    `scenarios.MOJOTREES_COSINE_LEAFWISE_CLAIMS`. In short: no peer column,
    because no competitor here ships leaf-wise Cosine; and it moves TWO
    parameters, because `lambda_l2=3.0` is carried to get off the point where
    Cosine degenerates to `sqrt` of the L2 score, so this row does not price
    the score function on its own.
    """

    name = "mojotrees_cosine_leafwise"
    params_fn = staticmethod(scenarios.mojotrees_cosine_leafwise_params)

    def load(self):
        super().load()
        self.notes.append(scenarios.MOJOTREES_COSINE_LEAFWISE_CLAIMS)
        return self


class XGBoostEngine:
    """XGBoost at its own shipped defaults. The third peer column.

    Added 2026-08-17 on Andrew's instruction, and it exists because the
    `mojotrees_depthwise` arm had nothing external to be read against:
    `grow_policy=depthwise` is XGBoost's parameter and LightGBM has no growth
    policy at all. See `scenarios.MOJOTREES_DEPTHWISE_CLAIMS` for the
    three-way verification of that. Later the same day that arm was pointed at
    XGBoost's defaults outright, so this column is now the thing our depthwise
    row is read against rather than merely the nearest relative of it.

    Built on the same discipline as `CatBoostEngine` and deliberately not on
    the sklearn wrapper: the native `DMatrix` plus `xgboost.train` path is
    what `scenarios.xgboost_params` translates into, so exactly three things
    reach the library that are not the problem statement, and
    `XGBOOST_REFUSED_PARAMS` refuses by name anything that would make the arm
    not-stock. The wrapper is also not a source for a DEFAULT on this version:
    `XGBRegressor().get_params()` returns None for every tree parameter in
    3.4.0, checked rather than assumed.

    **THE RESOLVED CONFIGURATION IS READ BACK, AND IT IS BOTH RECORDED AND
    CHECKED.** `Booster.save_config()` after the fit is XGBoost's own answer to
    what ran, and it is the authority: it lands in
    `engine_resolved_params` on every record. Beside it,
    `scenarios.check_xgboost_readback` re-reads the handful of values the
    `mojotrees_depthwise` mirror arm was built from and records any drift in
    `engine_resolved_params_drift`, which is the CatBoost arm's structure and
    exists for the reason that one does: a mirror whose target nothing reads
    back cannot be shown to have gone wrong.

    This paragraph used to say the arm asserted NO defaults at all, on the
    grounds that `xgboost` was not installed when it was written so nothing
    could be read. That was true for a few hours and is not now. See
    `scenarios.XGBOOST_DEFAULTS_SOURCE` for the read and
    `scenarios.XGBOOST_RESOLVED_DEFAULTS` for the table.

    **CPU ONLY, and the reason differs from CatBoost's.** CatBoost is refused
    a GPU row here because its GPU training quantizes differently and so is
    not the same measurement. XGBoost is CPU-only here for a blunter reason,
    and it is worse than an absent backend: asking for one does not fail. The
    installed conda package is the CPU build (`libxgboost-3.4.0-cpu_*`) and
    the machine is Apple silicon with no CUDA device, and a fit passed
    `device="cuda"` on 3.4.0 TRAINED ANYWAY, warning "Device is changed from
    GPU to CPU as we couldn't find any available GPU on the system" and
    resolving `device` to `cpu`. Verified on 2026-08-17. So a GPU XGBoost cell
    would not have been a failed cell, it would have been a CPU measurement
    labeled GPU, which is the exact defect this harness's backend proof exists
    to catch. `load` refuses the device by name instead.

    Either way the record says which backend ran, which is what Andrew's
    directive requires: our accelerator is published beside a competitor's best
    AVAILABLE backend, and CPU is the ceiling for all three competitors here.

    **NO SEPARATE BINNING PHASE.** Handed a plain `DMatrix`, XGBoost quantizes
    inside `train`, which is the same position CatBoost is in.
    `QuantileDMatrix` would separate it and is deliberately not used, because
    building one is a different ingestion path from the one a default XGBoost
    user takes. See `scenarios.XGBOOST_UNMATCHABLE['binning_phase']`.
    """

    name = "xgboost"

    arm_params = None
    arm_dataset_params = None

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
                "this harness runs XGBoost on the CPU only, and unlike the "
                "CatBoost refusal beside it this is not a quantization "
                "argument: XGBoost's only accelerator backend is CUDA, this "
                "machine is Apple silicon, and the installed package is the "
                "CPU conda build. The refusal is here rather than left to the "
                "library because the library does NOT refuse: on 3.4.0 a fit "
                "passed device='cuda' trains on the CPU with a warning and "
                "resolves device to 'cpu', so an unguarded GPU cell would "
                "produce a CPU measurement under a GPU label. CPU is this "
                "engine's ceiling here and the record says so"
            )
        phase = measure.Phase("import")
        try:
            with phase:
                import xgboost
        except ImportError as exc:
            # Named rather than left as a bare ImportError, the same treatment
            # the GPU refusal above gets and for the same reason: a cell that
            # dies is an infrastructure failure, and this harness answers one
            # by withholding the quality verdict for the WHOLE matrix. So the
            # message has to tell whoever reads the failed run what to do,
            # not merely that a module was missing.
            #
            # INSTALLED as of 2026-08-17: the `bench` environment carries
            # xgboost 3.4.0 and every measurement and default in this arm was
            # read from it. This branch used to say the opposite, because for
            # a few hours the dependency was declared and unsolved, and it was
            # left saying so after the solve. A message that names a cause
            # which is no longer the cause is worse than a bare ImportError,
            # because whoever reads it stops looking.
            #
            # So it now says what is true and what to check, and the two most
            # likely causes are named in the order they should be checked: the
            # wrong environment, and an environment that has drifted from the
            # manifest.
            raise EngineError(
                "xgboost could not be imported. It IS declared in pixi.toml "
                "under [feature.bench.dependencies] and it WAS installed in "
                "the bench environment on 2026-08-17 at version 3.4.0, so "
                "this is an environment problem rather than a missing "
                "declaration. Check that this process is running under "
                "`pixi run -e bench`, then that the environment matches the "
                "manifest with `pixi install -e bench`. Until it imports, "
                f"this arm cannot run and no {scenarios.XGBOOST_ARM_ID} row "
                "exists. Original error: " + str(exc)
            ) from exc
        self.module = xgboost
        self.import_phase = phase
        self.version = xgboost.__version__
        try:
            scenarios.check_xgboost_version(self.version)
        except RuntimeError as exc:
            raise EngineError(str(exc)) from exc
        self.notes.append(
            f"peer arm {scenarios.xgboost_arm_id()}: "
            f"{scenarios.XGBOOST_ARM_LABEL}. Reported beside the comparator "
            f"{scenarios.comparator_id()} and never instead of it. "
            f"{scenarios.XGBOOST_ARM_REGISTERED}"
        )
        self.notes.append(
            "XGBoost determinism is "
            f"{scenarios.XGBOOST_DETERMINISM['status']}: "
            f"{scenarios.XGBOOST_DETERMINISM['flag']}. "
            f"{scenarios.XGBOOST_DETERMINISM['what_is_pinned']}. "
            f"{scenarios.XGBOOST_DETERMINISM['observed']}"
        )
        self.notes.append(
            "XGBoost defaults are NOT transcribed by this harness: "
            + scenarios.XGBOOST_DEFAULTS_SOURCE
        )
        return self

    def warmup(self, spec):
        x, y, _group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        params = scenarios.xgboost_params(spec, self.threads, extra)

        def _fit():
            dtrain = self.module.DMatrix(x, label=y)
            return self.module.train(params, dtrain, num_boost_round=1)

        _, phase = measure.timed(_fit)
        return phase

    def _predict(self, booster, task, dmatrix):
        """Predictions in the same shape the other engines return.

        Regression is the raw value, binary is the positive-class
        probability, which `binary:logistic` already returns as a 1-D array,
        and multiclass is the (n, k) matrix `multi:softprob` returns. So no
        conversion is needed and none is done, which is worth stating because
        the CatBoost adapter beside this one does convert and pays for it
        inside its timed call.
        """
        out = booster.predict(dmatrix)
        if task == "multiclass":
            return np.asarray(out)
        return np.asarray(out)

    def _model_size(self, booster):
        """Model size, computed here rather than through `measure.model_size`.

        `measure.model_size` calls `model_to_string()` and writes a
        `model.txt`. XGBoost's Booster has no `model_to_string`, and its
        `save_model` requires a `.json` or `.ubj` extension from 2.0, so both
        halves of the shared helper would record an exception string instead
        of a number. `save_raw()` is XGBoost's own serialization and is the
        honest counterpart, so this arm reports that and says which format it
        is rather than reporting two errors.
        """
        out = {"string_bytes": None, "file_bytes": None, "error": None}
        try:
            raw = booster.save_raw(raw_format="ubj")
            out["file_bytes"] = len(raw)
            out["format"] = "save_raw(raw_format='ubj')"
        except Exception as exc:  # pragma: no cover - engine-dependent
            out["error"] = f"save_raw: {type(exc).__name__}: {exc}"
        out["string_bytes_unavailable_reason"] = (
            "XGBoost's Booster has no model_to_string(); save_raw is its "
            "serialization and is reported as file_bytes"
        )
        return out

    def run(self, spec, train, test, repeats=1):
        runs, reason = scenarios.xgboost_supports(spec)
        if not runs:
            raise EngineError(
                f"the XGBoost peer arm does not run {spec['id']}: {reason}"
            )
        if train.get("categorical_feature"):
            raise EngineError(
                "the XGBoost arm is not wired for a categorical column. It "
                "needs enable_categorical and a pandas categorical dtype, "
                "which is a third categorical container in this harness "
                "beside LightGBM's index list and CatBoost's mixed frame, and "
                "each container needs its own proof that it did not change a "
                "value. See scenarios.XGBOOST_SCENARIO_SUPPORT"
            )
        arm_binning = _arm(self, "arm_dataset_params")
        if arm_binning:
            raise EngineError(
                "the XGBoost arm takes no per-arm binning overrides "
                f"({', '.join(sorted(arm_binning))}). Its bin budget is "
                "max_bin, which may not be the same quantity as this "
                "harness's bin count, so moving one and leaving the other "
                "at a different stock number is two sweeps rather than one "
                "axis. See scenarios.XGBOOST_UNMATCHABLE['bin_budget']"
            )
        # The class count, off the LOADED DATA and not off the spec.
        #
        # THIS WAS A BUG and it killed every multiclass cell of this arm. It
        # read `spec.get("n_classes")`, and a scenario carries its class count
        # inside `generator_sizes[tier]`, never at the top level of the spec,
        # so the read was always None, `extra` stayed None, and
        # `scenarios.xgboost_params` died on `KeyError: 'num_class'` for the
        # multiclass task. The two adapters beside this one both read it off
        # the training data with a fallback computed from the labels
        # (`LightGBMEngine.run`, `CatBoostEngine.run`), which is the form that
        # is right for a real dataset as well as a generated one, so this is
        # now the same line they run.
        extra = None
        if spec["task"] == "multiclass":
            extra = {
                "num_class": int(
                    train.get("n_classes") or (np.max(train["y"]) + 1)
                )
            }
        arm_training = _arm(self, "arm_params")
        if arm_training:
            extra = dict(extra or {})
            extra.update(arm_training)
        params = scenarios.xgboost_params(spec, self.threads, extra)
        # The backend, set here rather than in the translator for the reason
        # `XGBOOST_REFUSED_PARAMS['device']` gives: one source for it, so a
        # record cannot name a backend the fit did not use. `load` has already
        # refused anything but cpu.
        params["device"] = "cpu"
        # The matched tree count. `num_boost_round` is an argument to
        # `xgboost.train` rather than a member of the parameter dict, which is
        # why `XGBOOST_MATCHED` maps the argument name to the `BASE_PARAMS`
        # key instead of the translator emitting it.
        #
        # Resolved by `scenarios.xgboost_rounds` from the SAME `extra` the
        # parameter dict was built from, rather than assembled here. Two
        # reasons and the second is why it moved out of this method. It goes
        # through `shared_params`, which is the merge the CatBoost arm reads
        # its `iterations` from, so a scenario-level `params` entry moves this
        # arm's budget and the comparator's together instead of one of them;
        # asking a competitor for a different budget from the comparator is
        # not a comparison. And reading `BASE_PARAMS` at a call site is the
        # defect `MojoTreesEngine._n_estimators`'s docstring records, where the
        # one parameter a frontier sweep moves first was the one no caller
        # could override.
        rounds = scenarios.xgboost_rounds(spec, extra)

        dtrain, ingest = measure.timed(
            lambda: self.module.DMatrix(train["X"], label=train["y"])
        )
        booster, training = measure.timed(
            lambda: self.module.train(params, dtrain, num_boost_round=rounds)
        )

        task = spec["task"]
        dtest = self.module.DMatrix(test["X"])
        predictions, predict_batch = measure.repeat(
            lambda: self._predict(booster, task, dtest), repeats
        )
        row_matrix = test["X"][:1]
        if not train.get("sparse"):
            row_matrix = np.ascontiguousarray(row_matrix)
        drow = self.module.DMatrix(row_matrix)
        _, predict_row = measure.repeat(
            lambda: self._predict(booster, task, drow), 20, warmup=2
        )

        # The library's own resolved configuration. This is the ONLY place a
        # record can say what this arm ran, because this harness deliberately
        # transcribes no XGBoost default. Recorded rather than raised on
        # failure, for the reason the CatBoost read-back is: a version moving
        # a default is a thing to learn from a record, not a reason to lose a
        # cell.
        try:
            raw_resolved = json.loads(booster.save_config())
            resolved = scenarios.xgboost_readback_for_record(raw_resolved)
            resolved_note = (
                "xgboost.Booster.save_config() on the fitted booster, parsed "
                "as JSON. This is XGBoost's resolved configuration and not a "
                "restatement of what was passed. One field is replaced by its "
                "digest rather than carried: see "
                "scenarios.XGBOOST_READBACK_DROPPED"
            )
        except Exception as exc:  # pragma: no cover - engine-dependent
            raw_resolved = None
            resolved = None
            resolved_note = (
                f"save_config() failed: {type(exc).__name__}: {exc}"
            )

        # Where a LIVE fit disagrees with scenarios.XGBOOST_RESOLVED_DEFAULTS,
        # which is the table `MOJOTREES_DEPTHWISE` was built from. Checked
        # against the RAW read-back rather than the trimmed one, because the
        # trim is a reporting step and a check that runs on a reduced input is
        # checking something other than what ran.
        #
        # Recorded rather than raised, for the reason the CatBoost read-back
        # is: an XGBoost upgrade moving a default is a thing to find out about
        # from a record, not a reason to lose a measured cell and with it the
        # whole matrix's quality verdict.
        readback_drift = (
            scenarios.check_xgboost_readback(raw_resolved)
            if raw_resolved is not None
            else ["save_config() produced nothing: " + str(resolved_note)]
        )

        try:
            num_trees = int(booster.num_boosted_rounds())
        except Exception:  # pragma: no cover - engine-dependent
            num_trees = None

        return {
            "engine": self.name,
            "engine_version": self.version,
            "device_requested": self.device,
            "device_used": "cpu",
            "path": "dmatrix",
            "phases": {
                "import": self.import_phase.as_dict(),
                "ingest": ingest.as_dict(),
                "encode": None,
                "encode_unavailable_reason": (
                    "this arm does not run a categorical scenario, so there "
                    "is nothing to re-encode. See "
                    "scenarios.XGBOOST_SCENARIO_SUPPORT"
                ),
                "binning": None,
                "binning_unavailable_reason": (
                    scenarios.XGBOOST_UNMATCHABLE["binning_phase"]
                ),
                "train": training.as_dict(),
                "predict_batch": predict_batch,
                "predict_row": predict_row,
            },
            "params_used": params,
            "dataset_params_used": None,
            "dataset_params_unavailable_reason": (
                "XGBoost's DMatrix takes the data and train() takes the "
                "parameters. The binning settings are training parameters "
                "and are left at XGBoost's own defaults, so they appear in "
                "engine_resolved_params rather than in params_used"
            ),
            "engine_resolved_params": resolved,
            "engine_resolved_params_source": resolved_note,
            # Empty means a live fit resolved every value in
            # scenarios.XGBOOST_RESOLVED_DEFAULTS to what that table asserts. A
            # non-empty one means XGBoost has moved or the table was
            # transcribed wrongly, and either way the mojotrees_depthwise row
            # beside this one is mirroring something this version does not do.
            #
            # This used to be the empty list unconditionally, with a comment
            # saying a drift list is meaningless because the arm asserts no
            # defaults. That was true for the few hours before xgboost was
            # installed.
            "engine_resolved_params_drift": readback_drift,
            "num_boost_round": rounds,
            "model": {
                "num_trees": num_trees,
                "current_iteration": num_trees,
                "num_bin": None,
                "num_bin_unavailable_reason": (
                    "XGBoost exposes no per-feature bin count on the Booster. "
                    "The resolved max_bin is in engine_resolved_params, which "
                    "is a budget rather than the count actually produced"
                ),
                "size": self._model_size(booster),
            },
            "data_encoding": None,
            "categorical_features": None,
            "transfers": measure.unavailable("cpu-only path"),
            "peak_rss_bytes": measure.peak_rss_bytes(),
            "notes": list(self.notes),
            "predictions": predictions,
        }


# `MojoTreesXGBoostModeEngine` WAS HERE AND IS GONE, 2026-08-17. It was a
# second mojotrees arm, `mojotrees_xgboost_mode`, carrying XGBoost's defaults
# while `MojoTreesDepthwiseEngine` stayed a one-key growth-order isolation.
# Andrew collapsed the pair into one arm the same day ("make our depthwise
# params match xgboost"), because an arm is a column of benchmark cells in
# every scenario, tier and backend, and this suite runs on one laptop under a
# timing lock. Its whole content is now on `MojoTreesDepthwiseEngine`, which
# was already the identical class under another name;
# `scenarios.MOJOTREES_XGBOOST_MODE WAS HERE` records what the collapse cost.

ENGINES = {
    "mojotrees": MojoTreesEngine,
    # A mojotrees SUBJECT variant, not a peer: same engine, a different
    # parameter set, so it is selectable and never default. As of 2026-08-17
    # that set is XGBoost's resolved defaults and the name understates it; see
    # the class docstring.
    "mojotrees_depthwise": MojoTreesDepthwiseEngine,
    # The two CORRECTNESS arms, 2026-08-17. Subject variants like the one
    # above, and never default: `run.py`'s default --engine list is
    # ["mojotrees", "lightgbm"], so these cost nothing until asked for by
    # name. Each exists to give verify.check_device_agreement a cpu-versus-gpu
    # pair on a configuration that had none, and each covers a parameter under
    # which a live wrong answer was found by reading code on the day they were
    # added. See scenarios.CORRECTNESS_ARMS.
    "mojotrees_symmetric_colsample": MojoTreesSymmetricColsampleEngine,
    "mojotrees_cosine_leafwise": MojoTreesCosineLeafwiseEngine,
    "lightgbm": LightGBMEngine,
    # Peer arms, reported beside the comparator. Adding them here is what
    # makes worker.py able to run them without a change: it builds an engine
    # by name from this table.
    "catboost": CatBoostEngine,
    "xgboost": XGBoostEngine,
    "mojotrees_catboost_mode": MojoTreesCatBoostModeEngine,
}

#: Which arm each engine belongs to. The comparator row is LightGBM against
#: the plain mojotrees arm and nothing here changes that; the peer entries
#: exist so a reader of a record can tell a headline row from a peer row
#: without knowing the engine names by heart.
ENGINE_ARM = {
    "mojotrees": "subject",
    # Not plain "subject": it is mojotrees, so it is not a peer, and
    # it is not the headline row either. See the class docstring.
    #
    # It stayed "subject_variant" when it became the XGBoost mirror on
    # 2026-08-17, and that was a decision. "peer_subject" is what
    # `mojotrees_catboost_mode` carries and would have described the pairing
    # just as well, but it also takes an engine OUT of `verify.SUBJECT_ENGINES`
    # in the reader's mind and, more importantly, into
    # `scenarios.PEER_ENGINES`, whose members `selfcheck.py` requires to be
    # peers. This arm runs GPU cells and needs the backend proof and the
    # cpu-versus-gpu agreement check that subject arms get, so it stays a
    # subject and the pairing is recorded in the claims string instead.
    "mojotrees_depthwise": "subject_variant",
    # "subject_variant" and not a new role, deliberately. A fourth value would
    # have to be taught to every reader of this mapping, and what these two
    # arms are -- ours, not the headline -- is exactly what the existing value
    # means. What makes them CORRECTNESS arms rather than mirror arms is that
    # they have no peer column, and that is recorded where it can be acted on:
    # in scenarios.CORRECTNESS_ARMS, which selfcheck walks, and in each arm's
    # claims string, which travels into the record. Being a subject is also
    # what keeps them inside verify.SUBJECT_ENGINES, and for these two that is
    # not a side effect, it is the arm: the cpu-versus-gpu agreement check is
    # the only thing they produce.
    "mojotrees_symmetric_colsample": "subject_variant",
    "mojotrees_cosine_leafwise": "subject_variant",
    "lightgbm": "comparator",
    "catboost": "peer",
    "xgboost": "peer",
    "mojotrees_catboost_mode": "peer_subject",
}


def build(
    name, threads, device, catboost_readback=None,
    arm_params=None, arm_dataset_params=None,
):
    """An engine by name, carrying its arm's parameter overrides.

    `arm_params` and `arm_dataset_params` are the arm dimension reaching the
    engine: the first folds into the TRAINING parameters and the second into
    the BINNING ones. Both default to empty, which is the cell every matrix
    without `--arms` runs and is byte-for-byte the call this function made
    before the dimension existed.

    `catboost_readback` is the run's collected `CatBoost.get_all_params()`,
    keyed by cell, or None. Only `mojotrees_catboost_mode` reads it, and it
    refuses to build without it, so passing None here is how a caller says "no
    CatBoost cell has run in this run yet" rather than a way to opt out.
    Handed to every mojotrees-family engine because they share a constructor;
    the plain arm stores it and never looks at it.
    """
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(sorted(ENGINES))}")
    engine = ENGINES[name]
    if issubclass(engine, MojoTreesEngine):
        built = engine(threads, device, catboost_readback)
    else:
        built = engine(threads, device)
    # Set after construction rather than threaded through three constructor
    # signatures, because the three engine families do not share one and an
    # override dict is read, never validated, by any of them. The class
    # defaults are None, so an engine built any other way reads empty dicts
    # through `_arm` rather than raising an AttributeError.
    built.arm_params = dict(arm_params or {})
    built.arm_dataset_params = dict(arm_dataset_params or {})
    return built
