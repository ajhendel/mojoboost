"""End-to-end model: raw features in, predictions out.

`fit` bins the raw training matrix with quantile binning, trains a boosted
ensemble, and returns a `Model` that carries the fitted `BinMapper` so it
can predict on raw, unseen feature values.

Both entry points take a `device` (see device.mojo). The device chooses
which trainer grows the trees; it is not a property of the fitted model,
which is the same tree ensemble either way and serializes identically.
"""

from .bagging import BaggingParams
from .binning import BinMapper, fit_bins
from .boosting import (
    Booster,
    BoosterParams,
    IterationRange,
    MulticlassBooster,
    _softmax_inplace,
    train,
    train_multiclass,
)
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device
from .goss import GossParams
from .linear_tree import (
    check_linear_tree_unconnected,
    predict_batch_raw,
    predict_ensemble_raw,
    predict_multiclass_raw,
)
from .gpu_predict import (
    predict_gpu,
    predict_proba_gpu,
    predict_raw_gpu,
    predict_raw_multiclass_gpu,
)
from .objective import GradHessFn, train_custom
from .predict import oblivious_plan, predict_oblivious_batch
from .sampling import BootstrapParams, check_bootstrap_honored
from .train_gpu import train_gpu, train_multiclass_gpu


@fieldwise_init
struct Model(Copyable, Movable, Writable):
    var mapper: BinMapper
    var booster: Booster

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Model(n_trees=", len(self.booster.trees), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def predict(self, row: List[Float64]) raises -> Float64:
        """Response-scale prediction for one raw example (length n_features)."""
        return self.booster.response(self.predict_raw(row))

    def predict_raw(self, row: List[Float64]) raises -> Float64:
        """Raw-score prediction (log-odds for BINARY_LOGISTIC).

        A model with linear leaves (`Booster.linear`, linear_tree.mojo)
        evaluates them here, on the raw row; the bins-only `Booster` methods
        see the constant fallback. Every raw-row entry point on `Model` goes
        through the same test, so a linear model predicts one way."""
        var bins = self.mapper.bin_row(row)
        if self.booster.linear.is_active():
            return predict_ensemble_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_score,
                self.booster.learning_rate,
                bins,
                row,
            )
        return self.booster.predict_raw_bins(bins)

    def n_iterations(self) -> Int:
        """Boosting iterations the fitted ensemble kept."""
        return self.booster.n_iterations()

    def predict_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> Float64:
        """Response-scale prediction from the boosting iterations in `rng`
        alone. See IterationRange for how the range is interpreted, including
        where the base score sits."""
        return self.booster.response(self.predict_raw_range(row, rng))

    def predict_raw_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> Float64:
        """Raw-score prediction from the iterations in `rng` alone."""
        var bins = self.mapper.bin_row(row)
        if self.booster.linear.is_active():
            return predict_ensemble_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_score,
                self.booster.learning_rate,
                bins,
                row,
                rng.start,
                rng.stop,
            )
        return self.booster.predict_raw_bins_range(bins, rng)

    def leaf_indices(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Int]:
        """The leaf ordinal this raw example reaches in each tree of `rng`,
        one entry per iteration. See `Tree.leaf_ordinals` for the numbering
        and its stability guarantees."""
        return self.booster.leaf_indices_bins(self.mapper.bin_row(row), rng)

    def predict_batch[
        features_origin: ImmOrigin, //
    ](
        self,
        features: Span[Float64, features_origin],
        n_rows: Int,
        rng: IterationRange,
        raw_score: Bool = False,
        device: Int = CPU_DEVICE,
    ) raises -> List[Float64]:
        """Batched prediction on a raw column-major matrix
        (`features[f * n_rows + r]`), one output per row.

        `device` follows the training-time vocabulary in device.mojo:
        CPU_DEVICE walks the trees on the host, GPU_DEVICE walks them on the
        accelerator and raises when none is available, and AUTO_DEVICE keeps
        resolving to the CPU until a measured crossover exists. Binning
        always runs on the host (bin edges are Float64; a Float32 edge
        search could move a row a whole bin), so the two devices route every
        row to the same leaf and differ only by the Float32 accumulation of
        leaf values, the same contract the GPU trainer ships."""
        var resolved = resolve_device(
            device, n_rows, self.mapper.n_features, 1
        )
        # `build_view=False`: this matrix is SCORED, never
        # histogrammed, so the row-major view would be a second full
        # copy of the bin ids that nothing ever reads. The view went
        # default-on under a memory budget in the same round, and
        # without this every `predict` would silently double its
        # footprint on a path no user thinks of as a fit.
        var data = self.mapper.transform(
            features, n_rows, build_view=False
        )
        if self.booster.linear.is_active():
            if resolved == GPU_DEVICE:
                check_linear_tree_unconnected("GPU prediction")
            var raw_copy = List[Float64](capacity=len(features))
            for i in range(len(features)):
                raw_copy.append(features[i])
            var raw_out = predict_batch_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_score,
                self.booster.learning_rate,
                data,
                raw_copy,
                rng.start,
                rng.stop,
            )
            if raw_score:
                return raw_out^
            for r in range(len(raw_out)):
                raw_out[r] = self.booster.response(raw_out[r])
            return raw_out^
        if resolved == GPU_DEVICE:
            if raw_score:
                return predict_raw_gpu(self.booster, data, rng)
            return predict_gpu(self.booster, data, rng)
        # A symmetric ensemble is evaluated without a traversal: see
        # predict.mojo. The plan is built once per call, before any fan-out,
        # and it verifies the STRUCTURE of every tree in the range rather
        # than trusting the grow policy the model was trained under. An
        # ensemble it cannot verify -- leaf-wise, depth-wise, ragged, too
        # deep, or carrying linear leaves -- leaves the plan inactive and
        # falls through to the generic walker below.
        #
        # No switch, because there is nothing to choose between: the two arms
        # reach the same leaf of the same tree and sum the same Float64
        # values in the same order, so which one runs cannot be observed in
        # an output. The argument is written out in predict.mojo.
        var plan = oblivious_plan(self.booster, rng, data.n_rows)
        if plan.active:
            return predict_oblivious_batch(
                self.booster, plan, data, rng, raw_score
            )
        # One prediction per row, over row blocks. The per-row body used to
        # live here and now lives in `Booster.predict_batch_range`, which is
        # the same gather of `bins[f * n_rows + r]` followed by the same
        # range call; rows write disjoint output slots, so the outputs are
        # bit-identical to this loop's at any block count.
        return self.booster.predict_batch_range(data, rng, raw_score)


@fieldwise_init
struct MulticlassModel(Copyable, Movable, Writable):
    var mapper: BinMapper
    var booster: MulticlassBooster

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "MulticlassModel(n_classes=",
            self.booster.n_classes,
            ", n_trees=",
            len(self.booster.trees),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def predict_proba(self, row: List[Float64]) raises -> List[Float64]:
        """Class probabilities for one raw example (length n_features)."""
        if self.booster.linear.is_active():
            var raw = self.predict_raw(row)
            _softmax_list(raw)
            return raw^
        return self.booster.predict_proba_bins(self.mapper.bin_row(row))

    def predict_raw(self, row: List[Float64]) raises -> List[Float64]:
        """Raw per-class scores before the softmax. Linear leaves
        (`MulticlassBooster.linear`) are evaluated here, on the raw row."""
        var bins = self.mapper.bin_row(row)
        if self.booster.linear.is_active():
            return predict_multiclass_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_scores,
                self.booster.learning_rate,
                self.booster.n_classes,
                bins,
                row,
            )
        return self.booster.predict_raw_bins(bins)

    def n_iterations(self) -> Int:
        """Boosting iterations the fitted ensemble kept: one iteration grows
        one tree per class."""
        return self.booster.n_iterations()

    def predict_proba_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Float64]:
        """Class probabilities from the boosting iterations in `rng` alone."""
        if self.booster.linear.is_active():
            var raw = self.predict_raw_range(row, rng)
            _softmax_list(raw)
            return raw^
        return self.booster.predict_proba_bins_range(
            self.mapper.bin_row(row), rng
        )

    def predict_raw_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Float64]:
        """Per-class raw scores from the iterations in `rng` alone."""
        var bins = self.mapper.bin_row(row)
        if self.booster.linear.is_active():
            return predict_multiclass_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_scores,
                self.booster.learning_rate,
                self.booster.n_classes,
                bins,
                row,
                rng.start,
                rng.stop,
            )
        return self.booster.predict_raw_bins_range(bins, rng)

    def leaf_indices(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Int]:
        """The leaf ordinal this raw example reaches in each tree of `rng`,
        round-major: entry `i * n_classes + k` is class k's tree in the
        range's iteration i."""
        return self.booster.leaf_indices_bins(self.mapper.bin_row(row), rng)

    def predict_batch[
        features_origin: ImmOrigin, //
    ](
        self,
        features: Span[Float64, features_origin],
        n_rows: Int,
        rng: IterationRange,
        raw_score: Bool = False,
        device: Int = CPU_DEVICE,
    ) raises -> List[Float64]:
        """Batched prediction on a raw column-major matrix
        (`features[f * n_rows + r]`), row-major `[r * n_classes + k]`
        outputs: class probabilities, or per-class raw scores with
        `raw_score`.

        `device` carries the same meaning as in `Model.predict_batch`, and
        the same contract holds: binning always runs on the host, so both
        devices route every row to the same leaves and differ only by
        Float32 accumulation (and, for probabilities, the Float32
        softmax)."""
        var resolved = resolve_device(
            device, n_rows, self.mapper.n_features, self.booster.n_classes
        )
        # `build_view=False`: this matrix is SCORED, never
        # histogrammed, so the row-major view would be a second full
        # copy of the bin ids that nothing ever reads. The view went
        # default-on under a memory budget in the same round, and
        # without this every `predict` would silently double its
        # footprint on a path no user thinks of as a fit.
        var data = self.mapper.transform(
            features, n_rows, build_view=False
        )
        var linear = self.booster.linear.is_active()
        if linear and resolved == GPU_DEVICE:
            check_linear_tree_unconnected("GPU prediction")
        if resolved == GPU_DEVICE:
            if raw_score:
                return predict_raw_multiclass_gpu(self.booster, data, rng)
            return predict_proba_gpu(self.booster, data, rng)
        if not linear:
            # The constant-leaf path, over row blocks; see
            # `MulticlassBooster.predict_batch_range`. Linear leaves keep the
            # loop below because they are evaluated on the raw row rather than
            # on the bins, which the bins-only ensemble methods cannot see.
            return self.booster.predict_batch_range(data, rng, raw_score)
        # Linear leaves only, from here down: they read the raw row as well as
        # the bins, which is why they are not in the batch entry point above.
        var n_classes = self.booster.n_classes
        var out = List[Float64](capacity=n_rows * n_classes)
        var bins = List[Int](capacity=self.mapper.n_features)
        var row = List[Float64]()
        row.resize(self.mapper.n_features, 0.0)
        for r in range(n_rows):
            bins.clear()
            for f in range(self.mapper.n_features):
                bins.append(Int(data.bins[f * n_rows + r]))
                row[f] = features[f * n_rows + r]
            var scores = predict_multiclass_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_scores,
                self.booster.learning_rate,
                n_classes,
                bins,
                row,
                rng.start,
                rng.stop,
            )
            if not raw_score:
                _softmax_list(scores)
            for k in range(n_classes):
                out.append(scores[k])
        return out^

    def predict_class(self, row: List[Float64]) raises -> Int:
        """The argmax class for one raw example."""
        var raw = self.predict_raw(row)
        var argmax = 0
        for k in range(1, len(raw)):
            if raw[k] > raw[argmax]:
                argmax = k
        return argmax


def _softmax_list(mut scores: List[Float64]):
    """In-place softmax over a per-class score list; the same arithmetic as
    `boosting._softmax_inplace` on a whole list."""
    _softmax_inplace(scores, 0, len(scores))


def fit[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Model:
    """Fit on a column-major raw feature matrix (`features[f * n_rows + r]`).
    `alpha` is the target quantile for QUANTILE and the huber transition
    point for HUBER; other objectives ignore it. `device` is CPU_DEVICE,
    GPU_DEVICE, or AUTO_DEVICE; GPU_DEVICE raises when no accelerator is
    available instead of falling back. `bagging` samples training rows per
    tree (see bagging.mojo) and draws the same rows on either device; `goss`
    is the gradient-based alternative (see goss.mojo), likewise identical on
    either device. `use_missing` is LightGBM's parameter of that name: with
    it, `NaN` feature values train and predict as missing (see binning.mojo);
    without it they are binned as 0.0. `categorical_features` lists the
    feature indices to treat as integer-coded categoricals: they are split by
    category set rather than by threshold, and missing, unseen, and dropped
    categories route right (see categorical.mojo).

    `bootstrap` is CatBoost's `bootstrap_type` (see sampling.mojo): MVS, which
    is CatBoost's real CPU default, or the Bayesian bootstrap. **It is honored
    on both backends**, which it was not until 2026-08-16: `train_gpu` took no
    bundle at all and this function refused an enabled bootstrap that resolved
    to the GPU, rather than training an unsampled model and reporting a
    sampled one. It now takes one and its round loop draws it.

    The two samplers reach the device by different routes and it is worth
    knowing which, because one of them costs a stage. The Bayesian bootstrap
    reads no gradient and drops no row, so its per-tree draw goes straight
    into the device objective state's weight plane and the device round is
    unaffected. MVS solves its keep threshold from the round's per-row
    gradient magnitudes and then drops rows, and the device round holds
    neither, so an MVS fit resolves to the host-gradient arm -- the
    derivatives are computed and sampled on the host and every tree is still
    grown on the device. That is a resolution and not a downgrade of the
    sampler: the draw is the same draw, at the same seed, on the same rows the
    CPU trainer would have used."""
    if params.linear.is_active():
        # The binned-only trainers under this entry point do not carry the
        # raw matrix; the metric path does. Refusing is the rule for a
        # parameter that would otherwise parse and do nothing.
        check_linear_tree_unconnected(
            "model.fit (use custom_metric.fit_with_metrics, which fits and"
            " predicts linear leaves)"
        )
    # The objective is part of every crossover rule in `device_policy`, and
    # this entry point holds it and used to drop it, so `device='auto'` could
    # never select the accelerator from here.
    #
    # `ordered_boosting` and `score_function` are threaded for the same reason
    # and were dropped the same way. `device_policy` carries a block for each,
    # both reading a `DeviceRequest` field, and both fields have defaults --
    # so an entry point that does not pass them hands the policy `False` and
    # `SCORE_L2` whatever the caller asked for, and the blocks cannot fire.
    # Only the Python query surface passed them, which made two gates real on
    # one entry surface and decoration on this one. The consequence was not a
    # missing refusal but a misrouted one: `device='auto'` selected the
    # accelerator on shape and then raised inside the grower, where the whole
    # point of `auto` is to choose a backend that can run the fit.
    #
    # By keyword, because `objective` is the fifth positional and these are the
    # sixth and seventh; a positional pair here binds correctly today and
    # silently rebinds the moment anyone inserts a parameter.
    var backend = resolve_device(
        device,
        n_rows,
        n_features,
        1,
        objective,
        ordered_boosting=params.ordered.enabled,
        score_function=params.tree.extra.score_function,
        random_strength=params.tree.extra.random_strength,
        derivative_precision=params.tree.extra.derivative_precision,
        grow_policy=params.tree.grow_policy,
        max_depth=params.tree.max_depth,
    )
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster: Booster
    if backend == GPU_DEVICE:
        # The bundle now crosses instead of being refused here. `train_gpu`
        # draws it per round through `sampling.bootstrap_round` (host
        # gradients) or through the device weight plane (Bayesian), and
        # refuses by name the one combination it cannot draw -- an explicit
        # `objective_source=OBJECTIVE_SOURCE_DEVICE` beside MVS. Passed by
        # keyword because it is appended last on that signature, so every
        # positional caller of `train_gpu` keeps working.
        booster = train_gpu(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
            bootstrap=bootstrap,
        )
    else:
        booster = train(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
            bootstrap=bootstrap,
        )
    return Model(mapper^, booster^)


def fit_multiclass[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> MulticlassModel:
    """Fit a softmax multiclass model on a column-major raw feature matrix
    (`features[f * n_rows + r]`), labels in 0..n_classes-1. `device` carries
    the same meaning as in `fit`: multiclass runs on either backend, growing
    one tree per class per round, so GPU_DEVICE trains on the device rather
    than raising. `bagging` draws one bag per round, shared by every class's
    tree in that round, and `goss` draws its gradient-based sample on the
    same once-per-round schedule; both draw identical rows on either device.
    `use_missing` and `categorical_features` carry the same meaning as in
    `fit`.

    `bootstrap` is CatBoost's `bootstrap_type` and is **honored on the CPU**:
    `boosting.train_multiclass` threads it into
    `boosting._boost_rounds_multiclass`, which draws once per round and shares
    the draw across every class's tree. It was refused outright here until
    that loop existed. `train_multiclass_gpu` still takes no bundle and its
    round loop never calls `sampling.bootstrap_round`, so a GPU fit is refused
    by name below rather than silently unsampled.

    The type CatBoost defaults a multiclass loss to is the **Bayesian**
    bootstrap, not MVS: `SetNotSpecifiedOptionsToDefaults` excludes the
    multiclass-only losses from the MVS default and the option keeps its
    declared Bayesian default (catalog A11 section 1, and
    `sampling.catboost_default_bootstrap_type` for the lines). MVS is accepted
    but needs an explicit `mvs_reg` (`sampling.check_mvs_reg_is_set`)."""
    if params.linear.is_active():
        check_linear_tree_unconnected(
            "model.fit_multiclass (use custom_metric.fit_multiclass_with_metrics)"
        )
    # OBJECTIVE_UNSPECIFIED, not `_MULTICLASS`, and the difference is a bug I
    # shipped and then measured. `objective_registry.MULTICLASS` is -1, a
    # registry sentinel meaning "this fit is multiclass"; `device_policy`
    # reads the same argument as a trainer objective code and -2 is its
    # "caller did not name one". So passing -1 made every multiclass GPU fit
    # raise "objective code -1 is not one the built-in trainers implement",
    # which bench/real_data caught on the first run after the patch.
    #
    # Unspecified is also the honest answer rather than merely the working
    # one: the softmax path grows one tree per class and each of those trees
    # carries a single-output objective this entry point does not know. If
    # the crossover rules should gate on multiclass as such, that is a case
    # `device_policy` needs to add, not a sentinel this caller can smuggle in.
    # `ordered_boosting` and `score_function` threaded for the reason given at
    # the single-output `fit` above: their `DeviceRequest` fields default, so
    # an entry point that drops them tells the policy the fit is plain and L2
    # whatever the caller asked, and `device='auto'` then picks the accelerator
    # on shape and raises inside the grower instead of routing to the CPU.
    # `objective` stays unpassed here, deliberately, for the reason above it.
    var backend = resolve_device(
        device,
        n_rows,
        n_features,
        n_classes,
        ordered_boosting=params.ordered.enabled,
        score_function=params.tree.extra.score_function,
        random_strength=params.tree.extra.random_strength,
        derivative_precision=params.tree.extra.derivative_precision,
        grow_policy=params.tree.grow_policy,
        max_depth=params.tree.max_depth,
    )
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster: MulticlassBooster
    if backend == GPU_DEVICE:
        # Named rather than left to `check_bootstrap_honored`, whose message
        # points at the single-output trainers; this is the same sentence
        # `fit` and `trainset.train_dataset_multiclass` use on their GPU arms.
        if bootstrap.enabled():
            raise Error(
                "bootstrap_type is not implemented on the GPU:"
                " train_multiclass_gpu takes no bootstrap bundle and its round"
                " loop never draws one, so the fit would be unsampled. This"
                " fit resolved to the GPU (device='gpu', or device='auto' on a"
                " shape the policy sends there); set device='cpu' or drop"
                " bootstrap_type"
            )
        booster = train_multiclass_gpu(
            data, labels, n_classes, params, sample_weight, bagging, goss
        )
    else:
        booster = train_multiclass(
            data,
            labels,
            n_classes,
            params,
            sample_weight,
            bagging,
            goss,
            bootstrap,
        )
    return MulticlassModel(mapper^, booster^)


def fit_custom[
    F: GradHessFn, features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Model:
    """Fit a caller-supplied objective on a column-major raw feature matrix
    (`features[f * n_rows + r]`), the `fit` counterpart of `train_custom`
    (see objective.mojo for the callback contract).

    `Model.predict` returns the raw score for a custom-objective model,
    since the framework does not know the inverse link. CPU only: there is
    no `device` argument, use `train_custom_gpu` on a pre-binned matrix for
    GPU tree growth. `use_missing` and `categorical_features` carry the same
    meaning as in `fit`.

    `bootstrap` is accepted and REFUSED when enabled. `objective.train_custom`
    takes no bundle, and the exclusion is not merely a wiring gap:
    `boosting._check_bootstrap` refuses `bootstrap_type` beside a custom
    objective outright, because a callback's derivatives are the caller's and
    a draw would rescale them without the caller's knowledge."""
    check_bootstrap_honored(bootstrap, String("model.fit_custom"))
    if params.linear.is_active():
        check_linear_tree_unconnected("model.fit_custom")
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster = train_custom(
        data, target, grad_hess, params, sample_weight, base_score
    )
    return Model(mapper^, booster^)
