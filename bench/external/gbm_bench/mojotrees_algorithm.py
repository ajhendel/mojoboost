"""mojotrees adapters for NVIDIA's gbm-bench.

This file is injected into a gbm-bench checkout by
`bench/external/patch_gbm_bench.py`. It deliberately mirrors gbm-bench's own
`LgbmAlgorithm` line for line, because the whole point of running someone
else's harness is that our arm is configured the way theirs is and not the
way we would have chosen. Where a parameter has no mojotrees equivalent the
difference is recorded in `PARITY_NOTES` rather than silently dropped.

Three arms are registered.

  mojotrees-cpu   device="cpu", the oracle path
  mojotrees-gpu   device="gpu", explicit rather than "auto" so a run that
                  cannot reach the accelerator fails loudly instead of
                  quietly reporting a CPU number under a GPU label
  mojotrees-gpu-compact
                  the same GPU arm with row compaction on, so the two can be
                  interleaved inside one process against the drift
"""

import os

import numpy as np

import mojotrees

from algorithms import Algorithm, Timer, shared_params
from datasets import LearningTask


# Differences between this arm and gbm-bench's LgbmAlgorithm, for the record.
# Anything added here belongs in the writeup beside the numbers.
PARITY_NOTES = {
    "max_leaves": (
        "gbm-bench passes LightGBM's `max_leaves` alias; mojotrees spells it "
        "`num_leaves`. Same quantity, same value (256)."
    ),
    "nthread": (
        "gbm-bench passes its `-cpus` as LightGBM's `nthread`. mojotrees "
        "refuses an explicit `n_jobs` by design, because a fit is "
        "bit-identical at every worker count and the count comes from "
        "MOJOTREES_NUM_WORKERS and the machine. So the arm exports "
        "MOJOTREES_NUM_WORKERS with the same value gbm-bench gave the other "
        "libraries. Same quantity, set through the only door mojotrees "
        "offers. It is process-wide, so both mojotrees arms in an "
        "interleaved run share it, which is what we want."
    ),
    "scale_pos_weight": (
        "LightGBM's binary-only `scale_pos_weight=w` is `class_weight={1: w}` "
        "in mojotrees. Same reweighting, different spelling."
    ),
    "device": (
        "gbm-bench has no device concept for a non-CUDA accelerator, so the "
        "arm name carries it. `device='gpu'` is explicit, never 'auto', so "
        "the harness cannot report a CPU run under a GPU label."
    ),
}


class MojoTreesAlgorithm(Algorithm):
    device = "cpu"

    def configure(self, data, args):
        params = shared_params.copy()
        params.update({
            "num_leaves": 256,
            "n_estimators": args.ntrees,
            "device": self.device,
        })
        params.update(args.extra)
        return params

    @staticmethod
    def _apply_worker_count(args):
        """gbm-bench's -cpus, delivered the only way mojotrees accepts it."""
        if args.cpus:
            os.environ["MOJOTREES_NUM_WORKERS"] = str(args.cpus)

    def _estimator(self, data, params):
        params = dict(params)
        task = data.learning_task
        if task == LearningTask.REGRESSION:
            params["objective"] = "regression"
            return mojotrees.MojoTreesRegressor(**params)
        if task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary"
            # gbm-bench's scale_pos_weight, spelled the mojotrees way.
            positives = np.count_nonzero(data.y_train)
            if positives:
                params["class_weight"] = {1: len(data.y_train) / positives}
            return mojotrees.MojoTreesClassifier(**params)
        if task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "multiclass"
            return mojotrees.MojoTreesClassifier(**params)
        raise ValueError("unhandled learning task: " + str(task))

    def fit(self, data, args):
        self._apply_worker_count(args)
        params = self.configure(data, args)
        model = self._estimator(data, params)
        with Timer() as t:
            self.model = model.fit(data.X_train, data.y_train)
        return t.interval

    def test(self, data):
        if data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            return self.model.predict(data.X_test)
        if data.learning_task == LearningTask.CLASSIFICATION:
            # gbm-bench's metrics expect a positive-class probability here,
            # matching what lgb.Booster.predict returns for "binary".
            return self.model.predict_proba(data.X_test)[:, 1]
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class MojoTreesCPUAlgorithm(MojoTreesAlgorithm):
    device = "cpu"


class MojoTreesGPUAlgorithm(MojoTreesAlgorithm):
    device = "gpu"


class MojoTreesGPUCompactAlgorithm(MojoTreesAlgorithm):
    """The GPU arm with CatBoost-style physical row compaction turned on.

    A separate arm rather than a flag on the existing one, because the whole
    point is to interleave the two inside one process: this repository has
    measured the same benchmark drifting two- to threefold between time
    windows, so an on-run against an off-run from an hour earlier compares
    nothing.

    `MOJOTREES_GPU_ROW_COMPACTION` is read once per `GpuActiveRows`, which is
    constructed per fit, so setting it around the fit is enough and the arm
    that runs next does not inherit it. It is restored rather than deleted
    outright in case the caller set it deliberately.
    """

    device = "gpu"

    def fit(self, data, args):
        previous = os.environ.get("MOJOTREES_GPU_ROW_COMPACTION")
        os.environ["MOJOTREES_GPU_ROW_COMPACTION"] = "1"
        try:
            return super().fit(data, args)
        finally:
            if previous is None:
                del os.environ["MOJOTREES_GPU_ROW_COMPACTION"]
            else:
                os.environ["MOJOTREES_GPU_ROW_COMPACTION"] = previous


class LgbmCPUDeterministicAlgorithm(Algorithm):
    """LightGBM with `deterministic=true`, registered as a SEPARATE arm.

    Why this exists. gbm-bench's `lgbm-cpu` runs LightGBM stock, which is
    nondeterministic across threads: the same data and seed can give a
    different model run to run. mojotrees is reproducible by construction, so
    `mojotrees-gpu` against stock `lgbm-cpu` is not like-for-like on that
    axis, and it is not like-for-like in LightGBM's favor, because
    `deterministic=true` costs LightGBM speed.

    Why it is a new arm rather than an edit. Changing another library's
    configuration inside their harness is the exact move that would void the
    reason for using their harness at all. `lgbm-cpu` is left byte-identical
    to upstream and stays the headline number. This arm is additional
    information, run in the same process so it is interleaved with the
    others rather than compared across thermal windows.

    LightGBM's own documentation says `deterministic` should be set together
    with `force_row_wise` or `force_col_wise`, so this sets `force_row_wise`
    and nothing else. Report it under a name that says what it is.
    """

    def __init__(self):
        super(LgbmCPUDeterministicAlgorithm, self).__init__()
        from algorithms import LgbmCPUAlgorithm
        self._inner = LgbmCPUAlgorithm()

    def configure(self, data, args):
        params = self._inner.configure(data, args)
        params.update({"deterministic": True, "force_row_wise": True})
        return params

    def fit(self, data, args):
        import lightgbm as lgb
        dtrain = lgb.Dataset(data.X_train, data.y_train, free_raw_data=False)
        params = self.configure(data, args)
        with Timer() as t:
            self.model = lgb.train(params, dtrain, args.ntrees)
        self._inner.model = self.model
        return t.interval

    def test(self, data):
        return self._inner.test(data)

    def __exit__(self, exc_type, exc_value, traceback):
        return self._inner.__exit__(exc_type, exc_value, traceback)
