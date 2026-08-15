"""Engine adapters. One measured run each, same phases on both sides.

Both adapters answer the same questions in the same order, and both are
asked only for predictions. Metrics are computed later, by quality.py, from
those predictions. An engine's own evaluation output is never read into a
result record.

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
- `binning`: constructing the Dataset. Both libraries expose this as an
  explicit step, so both are measured the same way. On the paths where one
  of them bins inside fit, the field is null with a reason.
- `train`: boosting rounds only, on an already binned Dataset.
- `predict_batch` and `predict_row`: throughput on the whole test matrix,
  and the latency of a single-row call. The second is not the first
  divided by the row count, and a library can be good at one and bad at
  the other.

Both adapters return the predictions themselves, so the runner can compute
metrics, hash them for the determinism check, and compare the two engines
row by row.

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

    The counts themselves are not stored. A run with half a million
    features would put half a million integers in every record, so what is
    stored is the shape of the distribution plus a digest of the exact
    vector: two records with the same digest binned identically, and two
    that differ can be re-derived from their jobs. The digest goes through
    measure.digest so there is one hashing convention in this harness
    rather than two.

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
    return {
        "n_features": int(counts.size),
        "total": int(counts.sum()),
        "max": int(counts.max()),
        "min": int(counts.min()),
        "mean": float(counts.mean()),
        "sha256": measure.digest(counts),
    }


#: mojotrees's side of the same field. It builds histograms feature-major
#: and has no parameter that selects anything else, which
#: docs/LIGHTGBM_PARITY.md records in its force_col_wise / force_row_wise
#: row. Recorded rather than left absent so that a LightGBM record's
#: builder has something to be compared against, and because the pair being
#: compared is not the same strategy on both sides once LightGBM is forced
#: row-wise. What varies on this side is thread dispatch, which is
#: MOJOTREES_PARALLEL_MIN_OPS and lives in the environment block.
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
    is not a reproducible measurement. `LIGHTGBM_ALIGNMENT` forces
    row-wise, and a reader who does not know that cannot repeat the run.

    What is knowable: the request always. The resolution only when the
    request settles it, which is when either force flag is set. With
    neither set LightGBM times both strategies on the first iterations and
    keeps the winner in the tree learner's share state; 4.7 exposes no
    getter for it on the Booster, on the Dataset, or in the model text,
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


class MojoTreesEngine:
    name = "mojotrees"

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
        if self.device in ("gpu", "auto") and not mojotrees.gpu_available():
            raise EngineError(
                "device='gpu' was requested and mojotrees.gpu_available() is "
                "False; this build or this machine has no accelerator"
            )
        return self

    def warmup(self, spec):
        x, y, group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        params = scenarios.mojotrees_params(spec, self.device, extra)
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
        params = scenarios.mojotrees_params(spec, self.device, extra)
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
        params = scenarios.mojotrees_params(spec, "cpu")
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
        return self

    def warmup(self, spec):
        x, y, group, n_classes = _tiny_like(spec)
        extra = {"num_class": n_classes} if n_classes else None
        params = scenarios.lightgbm_params(
            spec, self.threads, dict(extra or {}, bin_construct_sample_cnt=len(y))
        )
        params["n_estimators"] = 1

        def _fit():
            data = self.module.Dataset(x, label=y, group=group, params=params)
            return self.module.train(params, data, num_boost_round=1)

        _, phase = measure.timed(_fit)
        return phase

    def run(self, spec, train, test, repeats=1):
        n_rows = train["X"].shape[0]
        extra = {"bin_construct_sample_cnt": int(n_rows)}
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


ENGINES = {"mojotrees": MojoTreesEngine, "lightgbm": LightGBMEngine}


def build(name, threads, device):
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(sorted(ENGINES))}")
    return ENGINES[name](threads, device)
