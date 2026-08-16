"""`device='auto'` reaches the multiclass GPU trainer: proved by the model.

WHY THIS FILE EXISTS, AND WHAT IT IS NOT. `tests/test_device_auto_crossover.
mojo` proves the *policy* answers GPU for a softmax workload: `resolve_device`
returns `GPU_DEVICE`, the decision code is `DECISION_AUTO_GPU_EVIDENCE`, the
rule cited is `apple-m4-metal-dense-multiclass`. A returned enum is not a
backend. This repository has shipped that failure more than once -- a device
test whose fixture sat below the gate it was testing, and a capability fixture
asserting a value no detection path could construct -- and
`bench/results/session3_2026-08-16/RESULTS.md` records both. The multiclass
path has its own instance of the same class of defect on record: before
2026-08-15 `trainset.train_dataset_multiclass` resolved a device and then
discarded the answer, so every multiclass GPU timing in the project was a CPU
fit wearing a GPU label, and the proof it was is that the covertype CPU and
GPU arms in `bench/real_data/results/20260815T023123Z` carry byte-identical
`predictions_sha256` while the single-output scenarios in the same run do not.

So every assertion here is on a **model**, not on a device code, and the
observable is one the host path cannot produce. `train_multiclass_gpu`
accumulates its histograms in Float32 fixed point where `train_multiclass` is
Float64 throughout, so at this scale the two backends produce leaf values that
differ in their bits. One dataset, three fits, compared bit for bit:

    auto  == gpu   ->  `auto` took the multiclass accelerator path
    auto  != cpu   ->  and that equality is not both arms agreeing

Neither half is enough alone. The first holds vacuously if the two trainers
happen to agree; the second holds if `auto` went somewhere else entirely.
Together they say `auto` ran `train_multiclass_gpu`. The second assertion is
also the exact check that would have caught the discarded-device bug the day
it shipped, which is why it is not treated as the optional half.

THE SHAPE, AND WHY IT IS THIS ONE. 300,000 rows is `AUTO_GPU_MIN_ROWS` plus a
fifth, so it is comfortably above the floor rather than sitting on it, and the
test cannot pass for the trivial reason that a small fixture resolves to the
CPU anyway. 54 features is `M4_MULTICLASS_MIN_FEATURES`, the feature count the
multiclass record was taken at and the rule's scope bound; below it no rule
covers the run, by design. Three classes rather than the record's seven,
because the rule deliberately does not bound the class count above and three
is enough to exercise one-tree-per-class-per-round at a third of the cost.
Nothing here is timed and nothing here is a benchmark.

CHECKED AGAINST VACUITY, by breaking it, on 2026-08-16. Each of the three
knobs was moved to the wrong side of its gate and the file was re-run:

- `_ROWS = AUTO_GPU_MIN_ROWS - 1`: `test_auto_actually_trains_multiclass_on_
  the_accelerator` failed on the `auto != cpu` assertion, so the row floor is
  what the passing result is about.
- `_FEATURES = M4_MULTICLASS_MIN_FEATURES - 1`: the same failure, so the
  feature scope bound is real too.
- `_CLASSES = 1`: rejected by the trainer before any comparison, which is the
  right failure and is why the class count is not a third vacuity knob.

Anyone changing the shape here should repeat that. A fixture sitting on the
wrong side of the gate it tests is this repository's most-repeated test defect.

NAMING. `test_gpu_*` and it carries no cpu-safe marker comment, so
`tools/run_tests.sh` classifies it GPU-only and a CPU-only runner never
compiles it. That is correct: this file opens a device and trains on it.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.boosting import BoosterParams
from mojotrees.device import AUTO_DEVICE, CPU_DEVICE, GPU_DEVICE
from mojotrees.device_policy import (
    AUTO_GPU_MIN_ROWS,
    M4_MULTICLASS_MIN_FEATURES,
    M4_MULTICLASS_MIN_OUTPUTS,
    build_accelerator_target,
)
from mojotrees.model import MulticlassModel
from mojotrees.tree import TreeParams
from mojotrees.trainset import Dataset, train_dataset_multiclass

from support import _uniform

# Comfortably above the floor rather than on it. Written as an expression of
# the constant so that raising `AUTO_GPU_MIN_ROWS` carries the fixture up with
# it instead of quietly leaving it underneath.
comptime _ROWS = AUTO_GPU_MIN_ROWS + AUTO_GPU_MIN_ROWS // 5

# The rule's feature scope bound, exactly. One below it and no rule covers the
# run, which is the second vacuity check in the docstring.
comptime _FEATURES = M4_MULTICLASS_MIN_FEATURES

# The rule's lower output bound, which is what "multiclass" means. Three would
# do; naming the constant means a rule that stopped applying at three classes
# would fail here rather than pass quietly on a hard-coded number.
comptime _CLASSES = M4_MULTICLASS_MIN_OUTPUTS + 1

# Small enough that three multiclass fits are quick (each grows
# `_ROUNDS * _CLASSES` trees), large enough that the trees have real structure
# to disagree about.
comptime _ROUNDS = 3
comptime _LEAVES = 8

comptime _OBSERVED_M4_TARGET = String("metal:4-metal4")


def _on_the_measured_build() -> Bool:
    """Whether this binary targets the accelerator the multiclass crossover
    rule is scoped to. The rule is Metal-on-M4 only, so on any other
    accelerator `auto` correctly keeps the CPU and there is nothing here to
    prove."""
    return build_accelerator_target() == _OBSERVED_M4_TARGET


def _features() -> List[Float64]:
    """Column-major deterministic features, the same generator
    `tests/support.mojo` uses."""
    var out = List[Float64](capacity=_ROWS * _FEATURES)
    for k in range(_ROWS * _FEATURES):
        out.append(_uniform(UInt64(k)))
    return out^


def _labels(features: List[Float64]) -> List[Float64]:
    """Class indices in `[0, _CLASSES)`, driven by a strong, distinct effect
    per class so the two backends are not deciding splits on knife-edge gain
    ties. A tie broken differently would change tree *shape*, and this file is
    about leaf value bits."""
    var y = List[Float64](capacity=_ROWS)
    for r in range(_ROWS):
        var x0 = features[0 * _ROWS + r]
        var x1 = features[1 * _ROWS + r]
        var x2 = features[2 * _ROWS + r]
        var score = 4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5)
        # Three wide, well-separated bands of a continuous score, so class
        # membership is genuinely learnable rather than noise.
        var k = 0
        if score > 0.9:
            k = 2
        elif score > -0.4:
            k = 1
        y.append(Float64(k))
    return y^


def _identical(a: MulticlassModel, b: MulticlassModel) -> Bool:
    """Whether two fitted softmax models are the same model, bit for bit.

    Class count, base scores, tree structure and leaf values, with the values
    compared as `Float64` equality, which for two finite doubles is bit
    equality. This is the observable the whole file rests on, so it is
    deliberately strict: nothing here is a tolerance."""
    if a.booster.n_classes != b.booster.n_classes:
        return False
    if len(a.booster.trees) != len(b.booster.trees):
        return False
    if len(a.booster.base_scores) != len(b.booster.base_scores):
        return False
    for i in range(len(a.booster.base_scores)):
        if a.booster.base_scores[i] != b.booster.base_scores[i]:
            return False
    for t in range(len(a.booster.trees)):
        ref ta = a.booster.trees[t]
        ref tb = b.booster.trees[t]
        if len(ta.value) != len(tb.value):
            return False
        for i in range(len(ta.value)):
            if ta.feature[i] != tb.feature[i]:
                return False
            if ta.threshold_bin[i] != tb.threshold_bin[i]:
                return False
            if ta.value[i] != tb.value[i]:
                return False
    return True


def _fit(dataset: Dataset, device: Int) raises -> MulticlassModel:
    return train_dataset_multiclass(
        dataset,
        _CLASSES,
        BoosterParams(
            _ROUNDS, 0.1, TreeParams(_LEAVES, 20, 0.0, 1e-3, 0.0)
        ),
        device=device,
    )


def test_auto_actually_trains_multiclass_on_the_accelerator() raises:
    """THE PROOF, and it is a model rather than a device code.

    `train_dataset_multiclass` is the entry point a real caller uses (it is
    what `python/mojotrees/basic.py` reaches through `_mojotrees.
    train_dataset_multiclass`) and it resolves the device itself, so nothing
    here injects a decision. The three fits differ only in the `device`
    argument.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        if not _on_the_measured_build():
            print("skipped: not the accelerator the crossover rule covers")
            return
        var features = _features()
        var dataset = Dataset(
            features, _ROWS, _FEATURES, _labels(features)
        )

        var auto = _fit(dataset, AUTO_DEVICE)
        var gpu = _fit(dataset, GPU_DEVICE)
        var cpu = _fit(dataset, CPU_DEVICE)

        # The two halves, and neither is sufficient alone. If the CPU and GPU
        # multiclass trainers ever did agree bit for bit at this scale the
        # first assertion fails and this test stops being able to prove
        # anything, which is the right way for it to break: loudly, rather
        # than by passing on a comparison that no longer separates the
        # backends. That is not hypothetical -- byte-identical multiclass
        # arms is exactly what the discarded-device bug looked like on the
        # benchmark record.
        assert_false(
            _identical(auto, cpu),
            String(
                "device='auto' produced the CPU trainer's softmax model bit"
                " for bit at ",
                _ROWS,
                " x ",
                _FEATURES,
                " over ",
                _CLASSES,
                " classes, so it did not reach the accelerator",
            ),
        )
        assert_true(
            _identical(auto, gpu),
            String(
                "device='auto' produced neither the CPU model nor the GPU"
                " model at ",
                _ROWS,
                " x ",
                _FEATURES,
                " over ",
                _CLASSES,
                " classes",
            ),
        )

        # And the fits are not degenerate: a model with no splits would
        # compare equal to anything and prove nothing. One tree per class per
        # round, round-major, is the multiclass ensemble's shape.
        assert_equal(auto.booster.n_classes, _CLASSES)
        assert_equal(len(auto.booster.trees), _ROUNDS * _CLASSES)
        var splits = 0
        for t in range(len(auto.booster.trees)):
            for i in range(len(auto.booster.trees[t].feature)):
                if auto.booster.trees[t].feature[i] >= 0:
                    splits += 1
        assert_true(
            splits > 0,
            "every tree in the auto fit is a stump, so the bit comparison"
            " above separates nothing",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
