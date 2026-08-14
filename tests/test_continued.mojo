"""Continued training: `train_more`, `init_score`, and `update_dataset`.

The contract these tests pin down is that continuing is not an approximation
of a longer run, it is the same run. Splitting 100 rounds into 40 then 60
must reproduce the 100-round ensemble tree for tree and prediction for
prediction, because `_boost_rounds` resumes from the raw scores the existing
trees produce and numbers its rounds from `round_offset`, so every seeded
decision (the bagging draw, the per-tree feature sample) draws what the
uninterrupted run would have drawn.

Bit-exactness rather than a tolerance is the right assertion for the
continued-versus-single-call comparisons. The 100-round run accumulates
`raw[r] += lr * tree.predict_row(...)` once per round; the continued run
rebuilds the same value in `Booster.predict_raw_row`, which starts at the
base score and adds `lr * trees[i].predict_row(...)` in the same tree order.
Same operands, same order, so the same float64. A tolerance here would hide
exactly the resumption bugs these tests exist to catch.

`init_score` is the one place a tolerance is used: a run that starts from an
explicit offset records a base score of 0, so recovering the plain run's
prediction means adding the offset back at the end rather than at the start,
and `(0 + a0 + a1) + base` is not required to equal `(base + a0) + a1`. The
trees themselves are still compared exactly, since both runs grow from a
bitwise identical `raw`.
"""

from std.os import remove
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojoboost.bagging import BaggingParams
from mojoboost.binning import BinnedMatrix, fit_bins
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_more,
    train_multiclass,
    train_multiclass_more,
)
from mojoboost.goss import GossParams
from mojoboost.trainset import (
    Dataset,
    train_dataset,
    train_dataset_multiclass,
    update_dataset,
    update_dataset_multiclass,
)
from mojoboost.model import Model, fit
from mojoboost.monotone import MonotoneConstraints
from mojoboost.serialize import load_model, save_model
from mojoboost.tree import TreeParams

comptime _TMP_PATH = "./.test_continued_roundtrip.tmp"

comptime _N_ROWS = 400
comptime _N_FEATURES = 6


def _make_dataset(
    mut features: List[Float64], mut target: List[Float64]
) raises:
    """Six informative features and a smooth target, deterministic. Enough
    signal that no round degenerates into a single-leaf tree, which matters:
    a skipped round would make `len(trees)` disagree with the number of
    rounds executed and the round numbering of a continued run would no
    longer line up with the uninterrupted one."""
    var state: UInt64 = 42
    for _ in range(_N_FEATURES):
        for _ in range(_N_ROWS):
            state = state * 6364136223846793005 + 1442695040888963407
            features.append(
                Float64(state >> 11) * (1.0 / 9007199254740992.0)
            )
    for r in range(_N_ROWS):
        var x0 = features[r]
        var x1 = features[_N_ROWS + r]
        var x2 = features[2 * _N_ROWS + r]
        var x3 = features[3 * _N_ROWS + r]
        target.append(
            3.0 * x0 + 2.0 * x1 * x2 - 1.5 * x3 + 0.5 * x0 * x0
        )


def _binary_target(target: List[Float64]) -> List[Float64]:
    var out = List[Float64](capacity=len(target))
    for r in range(len(target)):
        out.append(1.0 if target[r] > 1.4 else 0.0)
    return out^


def _params(n_estimators: Int, learning_rate: Float64 = 0.1) -> BoosterParams:
    return BoosterParams(
        n_estimators, learning_rate, TreeParams(8, 5, 1.0, 1e-3)
    )


def _sampled_params(n_estimators: Int) -> BoosterParams:
    """Feature subsampling on, so the per-tree feature draw (seeded by the
    absolute round index) is part of what continuation has to reproduce."""
    return BoosterParams(
        n_estimators,
        0.1,
        TreeParams(
            8,
            5,
            1.0,
            1e-3,
            0.0,
            feature_fraction=0.5,
            feature_fraction_seed=7,
        ),
    )


def _binned(features: List[Float64]) raises -> BinnedMatrix:
    var mapper = fit_bins(features, _N_ROWS, _N_FEATURES, 64)
    return mapper.transform(features, _N_ROWS)


def _assert_trees_identical(a: Booster, b: Booster) raises:
    """Node-for-node equality of two ensembles' trees."""
    assert_equal(len(a.trees), len(b.trees))
    for t in range(len(a.trees)):
        ref ta = a.trees[t]
        ref tb = b.trees[t]
        assert_equal(len(ta.feature), len(tb.feature))
        assert_equal(ta.n_leaves, tb.n_leaves)
        for i in range(len(ta.feature)):
            assert_equal(ta.feature[i], tb.feature[i])
            assert_equal(ta.threshold_bin[i], tb.threshold_bin[i])
            assert_equal(ta.left[i], tb.left[i])
            assert_equal(ta.right[i], tb.right[i])
            assert_equal(ta.missing_bin[i], tb.missing_bin[i])
            assert_true(ta.default_left[i] == tb.default_left[i])
            assert_true(ta.value[i] == tb.value[i])


def _assert_raw_identical(
    a: Booster, b: Booster, data: BinnedMatrix
) raises:
    for r in range(data.n_rows):
        assert_true(a.predict_raw_row(data, r) == b.predict_raw_row(data, r))


# -- 40 + 60 reproduces 100 --------------------------------------------------


def test_continued_rounds_match_single_call() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var full = train(data, target, SQUARED_ERROR, _params(100))
    var part = train(data, target, SQUARED_ERROR, _params(40))
    assert_equal(len(part.trees), 40)

    var added = train_more(part, data, target, _params(60))
    assert_equal(added, 60)
    assert_equal(len(part.trees), 100)
    _assert_trees_identical(full, part)
    _assert_raw_identical(full, part, data)


def test_continued_rounds_match_single_call_binary() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _binary_target(target)
    var data = _binned(features)

    var full = train(data, labels, BINARY_LOGISTIC, _params(100))
    var part = train(data, labels, BINARY_LOGISTIC, _params(40))
    var added = train_more(part, data, labels, _params(60))

    assert_equal(added, 60)
    _assert_trees_identical(full, part)
    _assert_raw_identical(full, part, data)


def test_continued_rounds_match_under_bagging() raises:
    """The bagging draw is seeded by the absolute round index, so a
    continued run reproduces the single call only if `round_offset` carries
    the rounds already grown."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)
    var bagging = BaggingParams(0.7, 1, 11)

    var full = train(
        data, target, SQUARED_ERROR, _params(100), [], 0.9, bagging
    )
    var part = train(
        data, target, SQUARED_ERROR, _params(40), [], 0.9, bagging
    )
    assert_equal(len(part.trees), 40)
    var added = train_more(
        part, data, target, _params(60), [], 0.9, bagging
    )

    assert_equal(added, 60)
    _assert_trees_identical(full, part)
    _assert_raw_identical(full, part, data)


def test_continued_rounds_match_under_feature_sampling() raises:
    """Same argument as bagging for the per-tree feature draw."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var full = train(data, target, SQUARED_ERROR, _sampled_params(100))
    var part = train(data, target, SQUARED_ERROR, _sampled_params(40))
    var added = train_more(part, data, target, _sampled_params(60))

    assert_equal(added, 60)
    _assert_trees_identical(full, part)
    _assert_raw_identical(full, part, data)


def test_continued_in_three_steps_matches_single_call() raises:
    """Resumption is not limited to one continuation."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var full = train(data, target, SQUARED_ERROR, _params(90))
    var part = train(data, target, SQUARED_ERROR, _params(30))
    _ = train_more(part, data, target, _params(30))
    _ = train_more(part, data, target, _params(30))

    assert_equal(len(part.trees), 90)
    _assert_trees_identical(full, part)
    _assert_raw_identical(full, part, data)


def test_continued_leaves_existing_trees_untouched() raises:
    """Appending must not rewrite a tree that is already in the ensemble,
    nor the base score every one of them is measured from."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(40))
    var before = part.copy()
    var base_before = part.base_score

    _ = train_more(part, data, target, _params(60))

    assert_equal(len(part.trees), 100)
    assert_true(part.base_score == base_before)
    assert_true(part.learning_rate == before.learning_rate)
    assert_equal(part.objective, before.objective)
    for t in range(len(before.trees)):
        ref old = before.trees[t]
        ref kept = part.trees[t]
        assert_equal(len(old.feature), len(kept.feature))
        for i in range(len(old.feature)):
            assert_equal(old.feature[i], kept.feature[i])
            assert_equal(old.threshold_bin[i], kept.threshold_bin[i])
            assert_true(old.value[i] == kept.value[i])


def test_continue_zero_rounds_is_a_noop() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(20))
    var added = train_more(part, data, target, _params(0))

    assert_equal(added, 0)
    assert_equal(len(part.trees), 20)


# -- rejected continuations, and their atomicity -----------------------------


def test_continued_rejects_learning_rate_change() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(20, 0.1))
    with assert_raises():
        _ = train_more(part, data, target, _params(10, 0.2))


def test_continued_rejects_monotone_change() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(20))
    var constrained = _params(10)
    var signs = List[Int]()
    signs.append(1)
    for _ in range(_N_FEATURES - 1):
        signs.append(0)
    constrained.tree.monotone = MonotoneConstraints.from_signs(
        signs, _N_FEATURES
    )
    with assert_raises():
        _ = train_more(part, data, target, constrained)


def test_continued_rejects_negative_rounds() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(20))
    with assert_raises():
        _ = train_more(part, data, target, _params(-1))


def test_continued_rejects_target_length_mismatch() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(20))
    var short = List[Float64]()
    for r in range(_N_ROWS - 1):
        short.append(target[r])
    with assert_raises():
        _ = train_more(part, data, short, _params(10))


def test_rejected_continuation_leaves_the_model_intact() raises:
    """Failure atomicity: a continuation that raises must leave the ensemble
    exactly as it was, not half-extended. `train_more` grows into a local
    list and only merges it once the loop has returned, so every rejection
    above is a no-op on the model."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var part = train(data, target, SQUARED_ERROR, _params(20))
    var before = part.copy()
    var raw_before = List[Float64]()
    for r in range(_N_ROWS):
        raw_before.append(part.predict_raw_row(data, r))

    with assert_raises():
        _ = train_more(part, data, target, _params(10, 0.2))
    with assert_raises():
        _ = train_more(part, data, target, _params(-1))

    assert_equal(len(part.trees), len(before.trees))
    assert_true(part.base_score == before.base_score)
    _assert_trees_identical(before, part)
    for r in range(_N_ROWS):
        assert_true(part.predict_raw_row(data, r) == raw_before[r])


# -- init_score --------------------------------------------------------------


def test_init_score_at_the_base_score_reproduces_a_plain_fit() raises:
    """Starting from the offset a plain run would have derived grows the
    same trees, and records a base score of 0 because the offset is the
    caller's to re-apply."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var plain = train(data, target, SQUARED_ERROR, _params(30))
    var offset = List[Float64]()
    for _ in range(_N_ROWS):
        offset.append(plain.base_score)

    var offsetted = train(
        data,
        target,
        SQUARED_ERROR,
        _params(30),
        [],
        0.9,
        BaggingParams.disabled(),
        GossParams.disabled(),
        offset,
    )

    assert_true(offsetted.base_score == 0.0)
    _assert_trees_identical(plain, offsetted)
    for r in range(0, _N_ROWS, 13):
        assert_almost_equal(
            offsetted.predict_raw_row(data, r) + plain.base_score,
            plain.predict_raw_row(data, r),
            atol=1e-9,
        )


def test_init_score_shifts_what_the_trees_have_to_learn() raises:
    """A deliberately wrong offset must change the fitted trees: if it did
    not, `init_score` would not be reaching the gradients at all."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var plain = train(data, target, SQUARED_ERROR, _params(30))
    var offset = List[Float64]()
    for _ in range(_N_ROWS):
        offset.append(plain.base_score + 5.0)
    var shifted = train(
        data,
        target,
        SQUARED_ERROR,
        _params(30),
        [],
        0.9,
        BaggingParams.disabled(),
        GossParams.disabled(),
        offset,
    )

    var differs = False
    for t in range(len(shifted.trees)):
        for i in range(len(shifted.trees[t].value)):
            if shifted.trees[t].value[i] != plain.trees[t].value[i]:
                differs = True
                break
    assert_true(differs)


def test_init_score_length_is_validated() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var data = _binned(features)

    var offset = List[Float64]()
    for _ in range(_N_ROWS - 1):
        offset.append(0.0)
    with assert_raises():
        _ = train(
            data,
            target,
            SQUARED_ERROR,
            _params(5),
            [],
            0.9,
            BaggingParams.disabled(),
            GossParams.disabled(),
            offset,
        )


# -- across a serialization boundary -----------------------------------------


def test_continuing_a_reloaded_model_matches_one_run() raises:
    """Save after 40 rounds, load, and add 60: the reloaded mapper has to
    bin the data the way the original did, or the appended trees would read
    bin indices that mean something else."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)

    var full = fit(
        features, _N_ROWS, _N_FEATURES, target, SQUARED_ERROR, _params(100), 64
    )
    var partial = fit(
        features, _N_ROWS, _N_FEATURES, target, SQUARED_ERROR, _params(40), 64
    )
    save_model(partial, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_true(loaded.mapper.matches(partial.mapper))
    assert_true(loaded.mapper.matches(full.mapper))

    var data = loaded.mapper.transform(features, _N_ROWS)
    var added = train_more(loaded.booster, data, target, _params(60))
    assert_equal(added, 60)
    assert_equal(len(loaded.booster.trees), 100)

    _assert_trees_identical(full.booster, loaded.booster)
    for r in range(0, _N_ROWS, 11):
        var row = List[Float64]()
        for f in range(_N_FEATURES):
            row.append(features[f * _N_ROWS + r])
        assert_true(loaded.predict(row) == full.predict(row))


# -- the Dataset entry point -------------------------------------------------


def test_update_dataset_matches_one_run() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)

    var ds = Dataset(
        features,
        _N_ROWS,
        _N_FEATURES,
        label=target.copy(),
        max_bin=64,
    )
    var full = train_dataset(ds, SQUARED_ERROR, _params(100))
    var partial = train_dataset(ds, SQUARED_ERROR, _params(40))
    var added = update_dataset(partial, ds, _params(60))

    assert_equal(added, 60)
    assert_equal(len(partial.booster.trees), 100)
    _assert_trees_identical(full.booster, partial.booster)


def test_update_dataset_rejects_a_differently_binned_dataset() raises:
    """A dataset binned with a different `max_bin` describes different bin
    indices, so continuing on it is refused rather than silently wrong."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)

    var ds = Dataset(
        features, _N_ROWS, _N_FEATURES, label=target.copy(), max_bin=64
    )
    var other = Dataset(
        features, _N_ROWS, _N_FEATURES, label=target.copy(), max_bin=32
    )
    assert_false(ds.mapper.matches(other.mapper))

    var model = train_dataset(ds, SQUARED_ERROR, _params(20))
    with assert_raises():
        _ = update_dataset(model, other, _params(10))
    assert_equal(len(model.booster.trees), 20)


# -- multiclass --------------------------------------------------------------


def _class_labels(target: List[Float64]) -> List[Int]:
    """Three classes cut from the regression target, so every class is well
    represented and no round degenerates."""
    var out = List[Int](capacity=len(target))
    for r in range(len(target)):
        if target[r] < 1.0:
            out.append(0)
        elif target[r] < 1.8:
            out.append(1)
        else:
            out.append(2)
    return out^


def _assert_multiclass_trees_identical(
    a: MulticlassBooster, b: MulticlassBooster
) raises:
    assert_equal(a.n_classes, b.n_classes)
    assert_equal(len(a.trees), len(b.trees))
    for t in range(len(a.trees)):
        ref ta = a.trees[t]
        ref tb = b.trees[t]
        assert_equal(len(ta.feature), len(tb.feature))
        assert_equal(ta.n_leaves, tb.n_leaves)
        for i in range(len(ta.feature)):
            assert_equal(ta.feature[i], tb.feature[i])
            assert_equal(ta.threshold_bin[i], tb.threshold_bin[i])
            assert_equal(ta.left[i], tb.left[i])
            assert_equal(ta.right[i], tb.right[i])
            assert_true(ta.value[i] == tb.value[i])


def test_multiclass_continued_rounds_match_single_call() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _class_labels(target)
    var data = _binned(features)

    var full = train_multiclass(data, labels, 3, _params(60))
    var part = train_multiclass(data, labels, 3, _params(24))
    assert_equal(len(part.trees), 24 * 3)

    var added = train_multiclass_more(part, data, labels, _params(36))
    assert_equal(added, 36)
    assert_equal(len(part.trees), 60 * 3)
    _assert_multiclass_trees_identical(full, part)

    # Base scores are the log priors of the original fit and must survive
    # continuation untouched: every existing tree is measured from them.
    for k in range(3):
        assert_true(part.base_scores[k] == full.base_scores[k])

    for r in range(0, _N_ROWS, 17):
        var bins = List[Int]()
        for f in range(_N_FEATURES):
            bins.append(data.bin_at(r, f))
        var a = full.predict_raw_bins(bins)
        var b = part.predict_raw_bins(bins)
        for k in range(3):
            assert_true(a[k] == b[k])


def test_multiclass_continued_rounds_match_under_bagging() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _class_labels(target)
    var data = _binned(features)
    var bagging = BaggingParams(0.7, 1, 5)

    var full = train_multiclass(data, labels, 3, _params(60), [], bagging)
    var part = train_multiclass(data, labels, 3, _params(24), [], bagging)
    var added = train_multiclass_more(
        part, data, labels, _params(36), [], bagging
    )

    assert_equal(added, 36)
    _assert_multiclass_trees_identical(full, part)


def test_multiclass_continued_rejects_learning_rate_change() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _class_labels(target)
    var data = _binned(features)

    var part = train_multiclass(data, labels, 3, _params(10, 0.1))
    with assert_raises():
        _ = train_multiclass_more(part, data, labels, _params(5, 0.25))
    assert_equal(len(part.trees), 30)


def test_multiclass_continued_rejects_out_of_range_label() raises:
    """A label naming a class the ensemble was never fitted with has no tree
    sequence to continue, so it is refused rather than silently dropped."""
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _class_labels(target)
    var data = _binned(features)

    var part = train_multiclass(data, labels, 3, _params(10))
    var grown = List[Int]()
    for r in range(len(labels)):
        grown.append(labels[r])
    grown[0] = 3
    with assert_raises():
        _ = train_multiclass_more(part, data, grown, _params(5))
    assert_equal(len(part.trees), 30)


def test_update_dataset_multiclass_matches_one_run() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _class_labels(target)
    var float_labels = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        float_labels.append(Float64(labels[r]))

    var ds = Dataset(
        features,
        _N_ROWS,
        _N_FEATURES,
        label=float_labels.copy(),
        max_bin=64,
    )
    var full = train_dataset_multiclass(ds, 3, _params(60))
    var partial = train_dataset_multiclass(ds, 3, _params(24))
    var added = update_dataset_multiclass(partial, ds, _params(36))

    assert_equal(added, 36)
    assert_equal(len(partial.booster.trees), 60 * 3)
    _assert_multiclass_trees_identical(full.booster, partial.booster)


def test_update_dataset_multiclass_rejects_foreign_binning() raises:
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(features, target)
    var labels = _class_labels(target)
    var float_labels = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        float_labels.append(Float64(labels[r]))

    var ds = Dataset(
        features, _N_ROWS, _N_FEATURES, label=float_labels.copy(), max_bin=64
    )
    var other = Dataset(
        features, _N_ROWS, _N_FEATURES, label=float_labels.copy(), max_bin=32
    )
    var model = train_dataset_multiclass(ds, 3, _params(10))
    with assert_raises():
        _ = update_dataset_multiclass(model, other, _params(5))
    assert_equal(len(model.booster.trees), 30)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
