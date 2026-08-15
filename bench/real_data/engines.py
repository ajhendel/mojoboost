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

        dataset = self._dataset(spec, train, scenarios.dataset_params(spec))
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
            "model": {
                "num_trees": booster.num_trees(),
                "current_iteration": booster.current_iteration(),
                "num_bin": dataset.num_bin(),
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
        estimator = self.module.MojoTreesClassifier(
            n_estimators=scenarios.BASE_PARAMS["n_estimators"],
            max_bin=scenarios.BASE_PARAMS["max_bin"],
            use_missing=scenarios.BASE_PARAMS["use_missing"],
            **{k: v for k, v in params.items() if k != "objective"},
        )
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
            "model": {
                "num_trees": booster.num_trees(),
                "current_iteration": booster.current_iteration(),
                "num_bin": None,
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
            "model": {
                "num_trees": booster.num_trees(),
                "current_iteration": booster.current_iteration(),
                "num_bin": None,
                "size": size,
            },
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
