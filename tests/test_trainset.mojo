"""The reusable `Dataset` and continued training.

Two claims carry this file. The first is that training on a `Dataset` is
training: a model fitted from a dataset is the model `model.fit` would have
produced from the same matrix and parameters, tree for tree, so reusing the
binning changes nothing about the answer. The second is that continued
training composes: 40 rounds followed by 60 more equal 100 rounds in one
call, bit for bit, which is what makes `Booster.update()` more than an
approximation.

The rest is the dataset's own contract: fields validated against the row
count on construction, and a model refusing to take trees from a dataset it
was not binned by.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    Dataset,
    TreeParams,
    fit,
    train_dataset,
    train_dataset_multiclass,
    train_dataset_ranker,
    update_dataset,
    update_dataset_multiclass,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _features(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major deterministic features in [0, 1)."""
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        out.append(_uniform(UInt64(k)))
    return out^


def _target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _params(n_rounds: Int) -> BoosterParams:
    var tree = TreeParams.default()
    tree.num_leaves = 8
    tree.min_data_in_leaf = 5
    return BoosterParams(n_rounds, 0.1, tree^)


def _row(
    features: List[Float64], n_rows: Int, n_features: Int, r: Int
) -> List[Float64]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def test_dataset_describes_itself() raises:
    var n_rows = 120
    var n_features = 4
    var features = _features(n_rows, n_features)
    var ds = Dataset(
        features,
        n_rows,
        n_features,
        _target(features, n_rows),
        max_bin=32,
    )
    assert_equal(ds.num_data(), n_rows)
    assert_equal(ds.num_feature(), n_features)
    assert_true(ds.num_bin() <= 32)
    assert_equal(len(ds.label), n_rows)


def test_dataset_rejects_mismatched_columns() raises:
    var n_rows = 40
    var n_features = 3
    var features = _features(n_rows, n_features)

    with assert_raises():
        # One label short.
        var short = _target(features, n_rows)
        _ = short.pop()
        _ = Dataset(features, n_rows, n_features, short^)

    with assert_raises():
        # Group counts that do not cover the rows.
        _ = Dataset(
            features,
            n_rows,
            n_features,
            _target(features, n_rows),
            group=[10, 10],
        )

    with assert_raises():
        # A categorical index off the end of the matrix.
        _ = Dataset(
            features,
            n_rows,
            n_features,
            _target(features, n_rows),
            categorical_features=[7],
        )

    with assert_raises():
        # One name per feature, or none at all.
        _ = Dataset(
            features,
            n_rows,
            n_features,
            _target(features, n_rows),
            feature_names=[String("a"), String("b")],
        )


def test_dataset_training_matches_fit() raises:
    """A dataset is a binning that has already happened, nothing more: the
    model it trains is the one `fit` trains from the same matrix."""
    var n_rows = 200
    var n_features = 4
    var features = _features(n_rows, n_features)
    var target = _target(features, n_rows)

    var direct = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, _params(15), 64
    )
    var ds = Dataset(features, n_rows, n_features, target.copy(), max_bin=64)
    var indirect = train_dataset(ds, SQUARED_ERROR, _params(15))

    assert_equal(
        len(direct.booster.trees), len(indirect.booster.trees)
    )
    assert_equal(direct.booster.base_score, indirect.booster.base_score)
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        assert_equal(direct.predict(row), indirect.predict(row))


def test_dataset_reuse_gives_the_same_model() raises:
    """Training twice on one dataset trains twice on one binning."""
    var n_rows = 150
    var n_features = 3
    var features = _features(n_rows, n_features)
    var ds = Dataset(
        features, n_rows, n_features, _target(features, n_rows), max_bin=48
    )
    var first = train_dataset(ds, SQUARED_ERROR, _params(10))
    var second = train_dataset(ds, SQUARED_ERROR, _params(10))
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        assert_equal(first.predict(row), second.predict(row))


def test_continued_training_equals_one_run() raises:
    """40 + 60 rounds equal 100 rounds, tree for tree."""
    var n_rows = 200
    var n_features = 4
    var features = _features(n_rows, n_features)
    var ds = Dataset(
        features, n_rows, n_features, _target(features, n_rows), max_bin=64
    )

    var whole = train_dataset(ds, SQUARED_ERROR, _params(100))
    var staged = train_dataset(ds, SQUARED_ERROR, _params(40))
    var added = update_dataset(staged, ds, _params(60))

    assert_equal(added, 60)
    assert_equal(len(staged.booster.trees), len(whole.booster.trees))
    assert_equal(staged.booster.base_score, whole.booster.base_score)
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        assert_equal(staged.predict(row), whole.predict(row))


def test_continued_training_from_zero_rounds() raises:
    """A model with no trees is a legitimate starting point: it is what the
    Python `Booster(params, train_set)` constructor produces."""
    var n_rows = 120
    var n_features = 3
    var features = _features(n_rows, n_features)
    var ds = Dataset(
        features, n_rows, n_features, _target(features, n_rows), max_bin=32
    )

    var empty = train_dataset(ds, SQUARED_ERROR, _params(0))
    assert_equal(len(empty.booster.trees), 0)
    _ = update_dataset(empty, ds, _params(25))

    var whole = train_dataset(ds, SQUARED_ERROR, _params(25))
    assert_equal(len(empty.booster.trees), len(whole.booster.trees))
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        assert_equal(empty.predict(row), whole.predict(row))


def test_continued_multiclass_equals_one_run() raises:
    """The softmax counterpart: 10 + 15 rounds equal 25."""
    var n_rows = 180
    var n_features = 3
    var features = _features(n_rows, n_features)
    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        labels.append(Float64(r % 3))
    var ds = Dataset(features, n_rows, n_features, labels^, max_bin=32)

    var whole = train_dataset_multiclass(ds, 3, _params(25))
    var staged = train_dataset_multiclass(ds, 3, _params(10))
    var added = update_dataset_multiclass(staged, ds, _params(15))

    assert_equal(added, 15)
    assert_equal(len(staged.booster.trees), len(whole.booster.trees))
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        var a = staged.predict_proba(row)
        var b = whole.predict_proba(row)
        for k in range(3):
            assert_equal(a[k], b[k])


def test_continued_training_needs_the_same_binning() raises:
    """A model only takes trees from the dataset it was binned by, and the
    check is on the binning rather than on the caller's word."""
    var n_rows = 120
    var n_features = 3
    var features = _features(n_rows, n_features)
    var target = _target(features, n_rows)
    var coarse = Dataset(
        features, n_rows, n_features, target.copy(), max_bin=16
    )
    var fine = Dataset(features, n_rows, n_features, target.copy(), max_bin=64)

    var model = train_dataset(coarse, SQUARED_ERROR, _params(5))
    with assert_raises():
        _ = update_dataset(model, fine, _params(5))


def test_continued_training_holds_the_shrinkage_fixed() raises:
    """One ensemble, one learning rate: a continued run cannot change it,
    because every tree in the model is shrunk by the same factor."""
    var n_rows = 100
    var n_features = 3
    var features = _features(n_rows, n_features)
    var ds = Dataset(
        features, n_rows, n_features, _target(features, n_rows), max_bin=32
    )
    var model = train_dataset(ds, SQUARED_ERROR, _params(5))

    var faster = _params(5)
    faster.learning_rate = 0.5
    with assert_raises():
        _ = update_dataset(model, ds, faster)


def test_init_score_starts_where_it_is_told() raises:
    """`init_score` is training state: boosting starts from it, the fitted
    model has a base score of 0, and continuing adds it back."""
    var n_rows = 120
    var n_features = 3
    var features = _features(n_rows, n_features)
    var target = _target(features, n_rows)
    var offsets = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        offsets.append(0.25)

    var ds = Dataset(
        features,
        n_rows,
        n_features,
        target.copy(),
        init_score=offsets^,
        max_bin=32,
    )
    var model = train_dataset(ds, SQUARED_ERROR, _params(20))
    assert_equal(model.booster.base_score, 0.0)

    var whole = train_dataset(ds, SQUARED_ERROR, _params(40))
    var staged = train_dataset(ds, SQUARED_ERROR, _params(20))
    _ = update_dataset(staged, ds, _params(20))
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        assert_equal(staged.predict(row), whole.predict(row))


def test_ranking_dataset_needs_groups() raises:
    var n_rows = 60
    var n_features = 3
    var features = _features(n_rows, n_features)
    var grades = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grades.append(Float64(r % 3))

    var ungrouped = Dataset(
        features, n_rows, n_features, grades.copy(), max_bin=16
    )
    with assert_raises():
        _ = train_dataset_ranker(ungrouped, _params(5))

    var counts = List[Int](capacity=6)
    for _ in range(6):
        counts.append(10)
    var grouped = Dataset(
        features,
        n_rows,
        n_features,
        grades.copy(),
        group=counts^,
        max_bin=16,
    )
    var model = train_dataset_ranker(grouped, _params(5))
    assert_true(len(model.booster.trees) > 0)


def test_training_needs_a_label() raises:
    var n_rows = 40
    var n_features = 3
    var features = _features(n_rows, n_features)
    var ds = Dataset(features, n_rows, n_features, max_bin=16)
    with assert_raises():
        _ = train_dataset(ds, SQUARED_ERROR, _params(5))


def test_the_two_subsets_differ_in_where_the_bins_came_from() raises:
    """`subset` refits the binning, `subset_shared_binning` reuses it.

    The difference is the whole reason both exist. `subset` fits edges over
    the rows it keeps, so the rows left out had no say in them, which is what
    a fold or a held-out split needs. `subset_shared_binning` is LightGBM's
    `Dataset.subset`, binning the part as the whole was so that a bin index
    means the same thing in both and a model trained on the whole can score
    the part.
    """
    var n_rows = 120
    var n_features = 4
    var features = _features(n_rows, n_features)
    var label = _target(features, n_rows)
    var ds = Dataset(
        features,
        n_rows,
        n_features,
        label.copy(),
        max_bin=16,
        keep_raw=True,
    )

    var rows = List[Int]()
    for r in range(30):
        rows.append(r)

    var own = ds.subset(rows)
    assert_equal(own.num_data(), len(rows))
    assert_equal(own.num_feature(), n_features)
    for i in range(len(rows)):
        assert_equal(own.label[i], label[rows[i]])
    # Quantiles over 30 rows are not the quantiles over 120, so refitting is
    # observable rather than merely claimed.
    assert_true(not own.matches_binning(ds))

    var shared = ds.subset_shared_binning(rows)
    assert_equal(shared.num_data(), len(rows))
    assert_true(shared.matches_binning(ds))
    for i in range(len(rows)):
        for f in range(n_features):
            assert_equal(
                shared.data.bins[f * shared.n_rows + i],
                ds.data.bins[f * n_rows + rows[i]],
            )


def test_subset_refuses_what_it_cannot_honor() raises:
    var n_rows = 60
    var n_features = 3
    var features = _features(n_rows, n_features)

    var dropped = Dataset(
        features, n_rows, n_features, _target(features, n_rows)
    )
    with assert_raises():
        # No raw matrix to select from, and bins cannot be refitted from
        # bins, so this raises instead of returning something plausible.
        _ = dropped.subset([0, 1, 2])

    var kept = Dataset(
        features,
        n_rows,
        n_features,
        _target(features, n_rows),
        keep_raw=True,
    )
    with assert_raises():
        _ = kept.subset([2, 1, 0])  # not strictly ascending
    with assert_raises():
        _ = kept.subset([0, n_rows])  # off the end


def test_a_ranking_subset_takes_whole_queries() raises:
    var n_rows = 60
    var n_features = 3
    var features = _features(n_rows, n_features)
    var ranked = Dataset(
        features,
        n_rows,
        n_features,
        _target(features, n_rows),
        group=[20, 20, 20],
        keep_raw=True,
    )

    var second = List[Int]()
    for r in range(20, 40):
        second.append(r)
    var whole = ranked.subset(second)
    assert_equal(len(whole.group), 1)
    assert_equal(whole.group[0], 20)

    with assert_raises():
        # Part of a query is refused rather than repaired: a query is the
        # atom the ordering is learned and scored over.
        _ = ranked.subset([20, 21, 22])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
