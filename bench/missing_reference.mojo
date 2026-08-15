"""mojotrees's half of the LightGBM missing-value comparison.

Trains the four probe datasets of `bench/compare_missing_lightgbm.py` and
prints the routing decisions as one JSON line. The datasets are closed forms
of the row index here and there alike, so both sides train on bit-identical
data without exchanging a file.

Run through the comparison driver:
    pixi run -e bench python bench/compare_missing_lightgbm.py
"""

from std.utils.numerics import inf, nan

from mojotrees.binning import fit_bins
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.model import fit
from mojotrees.tree import TreeParams

comptime NAN = nan[DType.float64]()
comptime INF = inf[DType.float64]()

comptime N_ROWS = 300
comptime N_CLEAN = 200


def _params(num_leaves: Int, min_data_in_leaf: Int) -> BoosterParams:
    # One round at learning_rate 1.0, so the printed prediction is the tree's
    # own leaf value plus the base score, as on the LightGBM side.
    return BoosterParams(
        1, 1.0, TreeParams(num_leaves, min_data_in_leaf, 1.0, 1e-3, 0.0)
    )


def _bool(x: Bool) -> String:
    return "true" if x else "false"


def _missingness_data() -> Tuple[List[Float64], List[Float64]]:
    var features = List[Float64](capacity=N_ROWS)
    var target = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        var missing = r % 3 == 0
        features.append(NAN if missing else Float64(r % 17) / 17.0)
        target.append(5.0 if missing else -5.0)
    return (features^, target^)


def _clean_data() -> Tuple[List[Float64], List[Float64]]:
    var features = List[Float64](capacity=N_CLEAN)
    var target = List[Float64](capacity=N_CLEAN)
    for i in range(N_CLEAN):
        var x = -3.0 + 6.0 * Float64(i) / Float64(N_CLEAN - 1)
        features.append(x)
        target.append(1.0 if x > 0.0 else -1.0)
    return (features^, target^)


def _infinity_data() -> Tuple[List[Float64], List[Float64]]:
    var features = List[Float64](capacity=22)
    var target = List[Float64](capacity=22)
    for i in range(20):
        features.append(Float64(i))
        target.append(0.0 if i < 10 else 1.0)
    features.append(INF)
    target.append(1.0)
    features.append(-INF)
    target.append(0.0)
    return (features^, target^)


def main() raises:
    var out = String("{")

    # 1. Missingness alone predicts the target.
    var pair = _missingness_data()
    var model = fit(
        pair[0], N_ROWS, 1, pair[1], SQUARED_ERROR, _params(4, 1), max_bins=16
    )
    var missing_bin = model.mapper.missing_bin[0]
    var nan_row: List[Float64] = [NAN]
    var zero_row: List[Float64] = [0.0]
    var half_row: List[Float64] = [0.5]
    # The root isolates the missing rows when it splits at the top ordinary
    # bin and sends missing the other way: LightGBM's threshold = 1e300.
    ref root_tree = model.booster.trees[0]
    var isolates = (
        root_tree.feature[0] == 0
        and root_tree.missing_bin[0] == missing_bin
        and root_tree.threshold_bin[0] == missing_bin - 1
        and not root_tree.default_left[0]
    )
    out += '"predictive": {'
    out += '"reserves_missing_bin": ' + _bool(missing_bin >= 0) + ", "
    out += '"isolates_missing": ' + _bool(isolates) + ", "
    out += '"default_left": ' + _bool(root_tree.default_left[0]) + ", "
    out += '"pred_nan": ' + String(model.predict(nan_row)) + ", "
    out += '"pred_zero": ' + String(model.predict(zero_row)) + ", "
    out += '"pred_half": ' + String(model.predict(half_row)) + "}, "

    # 2. No missing value in training, NaN at predict time.
    var clean = _clean_data()
    var clean_model = fit(
        clean[0], N_CLEAN, 1, clean[1], SQUARED_ERROR, _params(2, 5),
        max_bins=255,
    )
    out += '"clean": {'
    out += (
        '"reserves_missing_bin": '
        + _bool(clean_model.mapper.missing_bin[0] >= 0)
        + ", "
    )
    out += (
        '"nan_equals_zero": '
        + _bool(clean_model.predict(nan_row) == clean_model.predict(zero_row))
        + "}, "
    )

    # 3. use_missing=false.
    var off = fit(
        pair[0], N_ROWS, 1, pair[1], SQUARED_ERROR, _params(4, 1),
        max_bins=16, use_missing=False,
    )
    out += '"use_missing_false": {'
    out += (
        '"reserves_missing_bin": '
        + _bool(off.mapper.missing_bin[0] >= 0)
        + ", "
    )
    out += (
        '"nan_equals_zero": '
        + _bool(off.predict(nan_row) == off.predict(zero_row))
        + "}, "
    )

    # 4. Infinities are finite-side extremes, not missing values.
    var infinity = _infinity_data()
    var inf_mapper = fit_bins(infinity[0], 22, 1, max_bins=8)
    var clamped = True
    for i in range(len(inf_mapper.edges)):
        var e = inf_mapper.edges[i]
        if e > 1e300 or e < -1e300:
            clamped = False
    var inf_bin = inf_mapper.bin_value(0, INF)
    var neg_inf_bin = inf_mapper.bin_value(0, -INF)
    out += '"infinities": {'
    out += (
        '"reserves_missing_bin": '
        + _bool(inf_mapper.missing_bin[0] >= 0)
        + ", "
    )
    out += '"edges_clamped": ' + _bool(clamped) + ", "
    out += (
        '"inf_matches_max_finite": '
        + _bool(inf_bin == inf_mapper.bin_value(0, 19.0))
        + ", "
    )
    out += (
        '"neg_inf_matches_min_finite": '
        + _bool(neg_inf_bin == inf_mapper.bin_value(0, 0.0))
        + "}"
    )

    out += "}"
    print(out)
