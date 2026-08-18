"""Cross-validation for the functional API: `cv()` and `CVBooster`.

    import mojotrees as mb
    from mojotrees.cv import cv

    train_set = mb.Dataset(X, label=y)
    history = cv({"objective": "regression", "num_leaves": 31},
                 train_set, num_boost_round=200, nfold=5,
                 metrics=("l2", "l1"), early_stopping_rounds=20,
                 return_cvbooster=True)

    history["valid l2-mean"]   # one entry per evaluated round
    history["valid l2-stdv"]   # spread across the folds of that round
    history["cvbooster"]       # the fitted fold models

`cv()` splits the training data, trains one model per fold on that fold's
own training rows, scores the held-out rows every round, and returns the
per-round mean and standard deviation across folds. `CVBooster` is the
collection of the fitted fold models, returned on request, and it forwards
whatever you ask of it to every fold.

This module is orchestration only. It does not train, it does not bin, and
it does not compute a metric: every fold is a `Dataset`, a `Booster`, and
`Booster.eval`, so a fold model is the model `train()` would have built on
the same rows and a fold score is the number `Booster.eval` would have
reported. What lives here is the splitting, the aggregation, the callback
plumbing, and the stopping rule.

Leakage
-------
The point of cross-validation is that the held-out rows had no say in the
model that scores them, and in a gradient-boosting library the quiet way to
break that is the binning. Bin edges are fitted from data; a dataset binned
over all the rows and then sliced would carry every fold's quantiles into
every other fold.

So `cv()` never slices a constructed `Dataset`. It slices the *raw* matrix
and the raw columns (`_arrays.take_rows`, the same selection
`src/mojotrees/raw_data.mojo` performs natively), builds a fresh `Dataset`
per fold per side, and lets each one bin itself over its own rows.
`fpreproc` is called after the split, once per fold, with that fold's two
datasets and a copy of the parameters, so any preprocessing it fits is
fitted on training rows alone. The cost is one binning pass per fold rather
than one in total, which is the price of the guarantee. Only the training
side actually pays it: a `Dataset` bins on `construct()`, and the held-out
side is never constructed, because it is scored through the *model's* mapper
(`Booster.predict` reads its raw matrix) rather than through its own.

A fold is never built with `reference=`. Reference binning reuses another
dataset's fitted edges, which is right for a validation set that a model
trained on the reference will score, and is exactly the leak here: the
reference's edges were fitted over rows this fold holds out. The same
distinction is drawn on the Mojo side between `Dataset.from_raw` /
`Dataset.subset`, which fit, and `Dataset.from_reference` /
`Dataset.subset_shared_binning`, which reuse.

A fold whose training rows overlap its own held-out rows is rejected rather
than scored, including when the overlap came from `folds=` you passed in
yourself.

Parallelism
-----------
The folds run one after another, and each fold's training is what uses the
machine: binning, histogram building, and split search parallelize inside
Mojo across the workers `MOJOTREES_NUM_WORKERS` allows. So the core count a
run uses is the trainer's, whatever `nfold` is, and nothing here multiplies
it. Do not wrap `cv()` in a process or thread pool over folds without
dividing `MOJOTREES_NUM_WORKERS` by the pool size first: `nfold` trainers
each claiming every core is slower than one trainer claiming every core, not
faster.

Folds
-----
`folds` takes precedence when given, as either an iterable of
`(train_index, test_index)` pairs or a scikit-learn splitter (anything with
a `.split` method, which is called as `split(X, y, groups)`). Otherwise
`cv()` generates them:

- **ranking** (`objective='lambdarank'`) splits whole queries, never rows.
  A query is an atom: its rows go to one side together, and each fold's
  `Dataset` gets the query counts of the queries it holds. This holds for
  `folds=` too, where a query straddling the split is reported rather than
  trained on.
- **classification** (`objective='binary'` or `'multiclass'`) stratifies
  when `stratified=True`, dealing each label's rows round-robin across the
  folds so every fold sees the label distribution of the whole.
- **everything else** takes contiguous chunks of the row order, shuffled
  first when `shuffle=True`. The shuffle is `random.Random(seed)`, so the
  same `seed` gives the same folds on every machine and every run, with or
  without numpy.

Metrics
-------
`metrics` accepts what the estimators' `eval_metric` accepts: one of
LightGBM's metric names, a callable, a `(name, func, higher_is_better,
use_for_early_stopping)` tuple, a dict with those keys, or a list mixing
them. `feval` is LightGBM's separate spelling for the callable form and is
appended to whatever `metrics` holds. With neither, the metric is the one
LightGBM would score for the objective.

A callable is called once per fold per evaluated round as
`metric(y_true, y_pred)`, where `y_pred` holds raw scores, flat, one block
of `num_class` per row for a softmax model. That is the convention the
estimators use, and the return value is a float or LightGBM's
`(name, value, is_higher_better)` triple of which only the value is read.

Early stopping
--------------
`early_stopping_rounds=` (or an `early_stopping()` callback, not both) stops
the run when the *aggregate* stops improving: the rule is applied to the
across-fold mean, not to any one fold, so the folds stop together on one
consensus round. `first_metric_only=True` watches the first metric alone;
otherwise every metric that did not opt out is watched and the first one to
run out of patience stops the run. Train-set metrics are never watched.

When it fires, the returned history is truncated to the winning round and
`CVBooster.best_iteration` records it, so `cvbooster.predict(X)` predicts
through that round. Without early stopping `best_iteration` stays -1 and
prediction uses every iteration.

Where these names live
----------------------
`cv` and `CVBooster` are this module's whole public surface. Every other
name in it, `FoldModel` included, is internal and may change without
notice.

Both are meant to be re-exported at the top level, so that `mojotrees.cv(
params, train_set, ...)` is the call a LightGBM user reaches for. The
package cannot then also answer `mojotrees.cv` with this module: the
function wins the attribute, and `import mojotrees.cv as m` binds `m` to
the function rather than to the module. `from mojotrees.cv import cv,
CVBooster` keeps working either way, because that form goes through the
import system rather than through the attribute. That trade is deliberate
and is written up, with the alternative, in
handoffs/integration_06_python_api.md (deleted, recover with git log --all --diff-filter=D -- handoffs/integration_06_python_api.md).

Differences from LightGBM
-------------------------
- **`results["iterations"]`** is mojotrees's addition: the round number each
  history entry belongs to. It exists because ranking cannot report every
  round (below), so the length of a history list is not always the round
  count and a caller should not have to assume it is.
- **Ranking reports the final round only.** Continued training does not
  cover LambdaRank (see `Booster.update`), so a ranking fold is trained once
  at the full round count instead of grown a round at a time. Its history
  has one entry, `results["iterations"] == [num_boost_round]`, and
  `early_stopping_rounds=` and `callbacks=` are refused rather than accepted
  and quietly ignored. See handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md) for the native work that
  would lift this.
- **Callbacks see 4-tuples.** LightGBM's `cv` hands callbacks a 5-tuple
  carrying the standard deviation, which is what `log_evaluation`'s
  `show_stdv` formats. `mojotrees.callback` unpacks exactly four, so this
  passes `(name, metric, mean, higher)` and the deviation is in the returned
  history rather than in the log line.
- **`reset_parameter` is refused, not ignored.** Changing a hyperparameter
  between rounds needs the fold boosters to re-read it, which `Booster` has
  no way to be told. A schedule that believes it is doing something and is
  not is worse than a failed run, so `CVBooster.reset_parameter` raises.
- **`CVBooster.predict_mean`** is mojotrees's addition: the across-fold mean
  of `predict`, refused for ranking rather than guessed at. LightGBM leaves
  the combination to the caller entirely, and `predict` still does.
- **`init_model` is refused.** Continued training checks that the dataset is
  the one the model was trained on, by comparing the binning, and a fold is
  by construction binned over its own rows. So every fold would be rejected
  by the trainer, and the argument is refused here with the reason rather
  than passed on to fail per fold. See handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md).
"""

import math
import random
import warnings

from . import _arrays, _eval
from .basic import Booster, Dataset, train
from .basic import _ROUND_ALIASES, _Config

_np = _arrays.np

#: The public surface, and all of it. `FoldModel` and everything else
#: without a leading underscore is internal; see the module docstring.
__all__ = ["CVBooster", "cv"]

#: The two sides of a fold, in the order they are reported.
_TRAIN = "train"
_VALID = "valid"


# -- row slicing ---------------------------------------------------------


def _take_rows(data, rows):
    """`data` restricted to `rows`, in the order `rows` gives them.

    The layout dispatch lives in `_arrays`, next to the rest of the package's
    buffer plumbing, so that the one place that knows how to select rows out
    of a numpy array, a pandas frame, a pyarrow table, and a polars frame is
    the place that knows how to read them at all. This name stays because it
    is what the fold code reads like.
    """
    return _arrays.take_rows(data, rows, "cv() data")


def _take_column(column, rows):
    """One of a dataset's columns restricted to `rows`, or None."""
    return _arrays.take_column(column, rows)


# -- fold generation -----------------------------------------------------


class _Split:
    """One fold: which rows train, which rows are held out, and the query
    counts of each side when the objective is a ranking one."""

    def __init__(self, name, train_rows, test_rows, train_group, test_group):
        self.name = name
        self.train_rows = list(train_rows)
        self.test_rows = list(test_rows)
        self.train_group = train_group
        self.test_group = test_group

    def check(self, n_rows):
        sides = ((_TRAIN, self.train_rows), (_VALID, self.test_rows))
        for label, rows in sides:
            if not rows:
                raise ValueError(
                    f"fold {self.name!r} has no {label} rows; use fewer folds"
                )
            for row in rows:
                if not 0 <= row < n_rows:
                    raise ValueError(
                        f"fold {self.name!r} names row {row}, which is out of "
                        f"range for {n_rows} rows"
                    )
        overlap = set(self.train_rows) & set(self.test_rows)
        if overlap:
            raise ValueError(
                f"fold {self.name!r} trains on {len(overlap)} of the rows it "
                "is scored on, so its score would not be out of sample; the "
                f"first is row {min(overlap)}"
            )


def _chunk_folds(n_items, n_folds, shuffle, seed):
    """`n_items` indices dealt into `n_folds` contiguous chunks, shuffled
    first when asked. The remainder goes to the earliest folds, so the
    largest and smallest fold differ by at most one item."""
    order = list(range(n_items))
    if shuffle:
        random.Random(seed).shuffle(order)
    base, extra = divmod(n_items, n_folds)
    out = []
    start = 0
    for f in range(n_folds):
        size = base + (1 if f < extra else 0)
        out.append(sorted(order[start : start + size]))
        start += size
    return out


def _stratified_folds(labels, n_folds, shuffle, seed):
    """Rows dealt into folds label by label, so each fold carries the label
    distribution of the whole. Each label's rows are shuffled first when
    asked, and then dealt round-robin from fold 0, which is what keeps a
    rare label spread rather than pooled."""
    buckets = {}
    for row, value in enumerate(labels):
        buckets.setdefault(float(value), []).append(row)
    folds = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for key in sorted(buckets):
        members = list(buckets[key])
        if len(members) < n_folds:
            warnings.warn(
                f"label {key:g} has {len(members)} rows, fewer than "
                f"nfold={n_folds}, so some folds hold none of it",
                UserWarning,
                stacklevel=3,
            )
        if shuffle:
            rng.shuffle(members)
        for position, row in enumerate(members):
            folds[position % n_folds].append(row)
    return [sorted(fold) for fold in folds]


def _query_starts(group):
    """The first row of each query, from the per-query row counts."""
    starts = []
    total = 0
    for count in group:
        starts.append(total)
        total += int(count)
    return starts


def _query_of_row(group):
    """The query each row belongs to."""
    out = []
    for query, count in enumerate(group):
        out.extend([query] * int(count))
    return out


def _rows_of_queries(group, starts, queries):
    """The rows of a set of queries, ascending, with the matching query
    counts. Queries are kept whole and in query order, which is what a
    ranking `Dataset` means by `group`."""
    rows = []
    counts = []
    for query in sorted(queries):
        count = int(group[query])
        rows.extend(range(starts[query], starts[query] + count))
        counts.append(count)
    return rows, counts


def _group_for_rows(rows, owner, name, side):
    """The query counts of an arbitrary row list, or a `ValueError` when the
    rows do not describe whole queries.

    A query split across the two sides of a fold is the ranking form of
    leakage: the model would learn the ordering of a query it is then scored
    on. A query whose rows arrive out of order is a different mistake, and
    both are reported rather than repaired.
    """
    counts = []
    seen = set()
    current = None
    for row in rows:
        query = owner[row]
        if query != current:
            if query in seen:
                raise ValueError(
                    f"the {side} rows of fold {name!r} interleave query "
                    f"{query} with other queries; a ranking fold takes whole "
                    "queries, in row order"
                )
            seen.add(query)
            counts.append(0)
            current = query
        counts[-1] += 1
    return counts


def _check_whole_queries(split, owner, group):
    """Every query lands entirely on one side of the fold."""
    train_queries = {owner[row] for row in split.train_rows}
    test_queries = {owner[row] for row in split.test_rows}
    straddling = train_queries & test_queries
    if straddling:
        raise ValueError(
            f"fold {split.name!r} splits query {min(straddling)} across its "
            "training and held-out rows; a ranking fold takes whole queries, "
            "so the folds must be built over queries rather than rows"
        )
    missing = set(range(len(group))) - train_queries - test_queries
    if missing:
        raise ValueError(
            f"fold {split.name!r} drops query {min(missing)} entirely; every "
            "query belongs on one side of the fold"
        )


def _complement(n_items, held):
    held = set(held)
    return [i for i in range(n_items) if i not in held]


def _generated_splits(dataset, task, n_folds, stratified, shuffle, seed):
    """The folds `cv()` builds when none were given."""
    n_rows = dataset.num_data()
    group = dataset.get_group()
    if task == _eval.RANKING:
        if group is None:
            raise ValueError(
                "a ranking cv needs a Dataset with group: the number of rows "
                "in each query, in row order"
            )
        if len(group) < n_folds:
            raise ValueError(
                f"nfold={n_folds} needs at least that many queries; this "
                f"dataset has {len(group)}"
            )
        starts = _query_starts(group)
        query_folds = _chunk_folds(len(group), n_folds, shuffle, seed)
        splits = []
        for f, held in enumerate(query_folds):
            test_rows, test_group = _rows_of_queries(group, starts, held)
            train_rows, train_group = _rows_of_queries(
                group, starts, _complement(len(group), held)
            )
            splits.append(
                _Split(
                    f"fold_{f}", train_rows, test_rows, train_group, test_group
                )
            )
        return splits

    if n_rows < n_folds:
        raise ValueError(
            f"nfold={n_folds} needs at least that many rows; this dataset "
            f"has {n_rows}"
        )
    if stratified and task in (_eval.BINARY, _eval.MULTICLASS):
        row_folds = _stratified_folds(
            dataset.get_label(), n_folds, shuffle, seed
        )
    else:
        row_folds = _chunk_folds(n_rows, n_folds, shuffle, seed)
    return [
        _Split(f"fold_{f}", _complement(n_rows, held), held, None, None)
        for f, held in enumerate(row_folds)
    ]


def _given_splits(folds, dataset, task):
    """The folds the caller gave, as `_Split`s.

    `folds` is either a scikit-learn splitter, which is asked to split, or
    an iterable of `(train_index, test_index)` pairs. Ranking folds are
    sorted and turned into query counts, which is also where a query split
    across the two sides is caught.
    """
    group = dataset.get_group()
    owner = None if group is None else _query_of_row(group)
    splitter = getattr(folds, "split", None)
    if splitter is not None:
        groups = None if owner is None else list(owner)
        pairs = splitter(dataset.get_data(), dataset.get_label(), groups)
    else:
        pairs = folds
    splits = []
    for f, pair in enumerate(pairs):
        try:
            train_index, test_index = pair
        except (TypeError, ValueError):
            raise ValueError(
                "folds must be a scikit-learn splitter or an iterable of "
                f"(train_index, test_index) pairs; entry {f} is {pair!r}"
            ) from None
        train_rows = [int(i) for i in train_index]
        test_rows = [int(i) for i in test_index]
        if task == _eval.RANKING:
            if owner is None:
                raise ValueError(
                    "a ranking cv needs a Dataset with group: the number of "
                    "rows in each query, in row order"
                )
            train_rows.sort()
            test_rows.sort()
        split = _Split(f"fold_{f}", train_rows, test_rows, None, None)
        if task == _eval.RANKING:
            _check_whole_queries(split, owner, group)
            split.train_group = _group_for_rows(
                split.train_rows, owner, split.name, _TRAIN
            )
            split.test_group = _group_for_rows(
                split.test_rows, owner, split.name, _VALID
            )
        splits.append(split)
    if not splits:
        raise ValueError("folds produced no folds")
    return splits


def _fold_dataset(source, rows, group):
    """One side of one fold as its own `Dataset`.

    Built from the raw matrix and the raw columns rather than from anything
    the source dataset binned, so this dataset bins itself over its own
    rows. The binning parameters, the feature names, and the categorical
    declaration are the source's, because those describe the columns rather
    than the rows.

    `reference=` is deliberately not passed on, whether or not the source had
    one: a fold binned through someone else's fitted edges would be scored
    under quantiles its held-out rows helped choose (see the module
    docstring).
    """
    raw = source.get_data()
    if raw is None:
        raise ValueError(
            "cv() needs the raw matrix to build the folds, and this "
            "Dataset's was freed; build it with free_raw_data=False"
        )
    return Dataset(
        _take_rows(raw, rows),
        label=_take_column(source.get_label(), rows),
        weight=_take_column(source.get_weight(), rows),
        group=group,
        init_score=_take_column(source.get_init_score(), rows),
        feature_name=source._names,
        categorical_feature=source.categorical_feature or None,
        params=dict(source.params),
        free_raw_data=False,
    )


# -- the fold model adapter ----------------------------------------------


class FoldModel:
    """What `cv()` needs of one fold, and all it needs.

    `Booster` grows a round at a time through `update()` for every objective
    except ranking, where continued training is not available, so a fold is
    reached through this two-method adapter rather than through `Booster`
    directly. `_IncrementalFold` implements it by growing; `_RetrainFold`
    implements it by training once, and reports through `checkpoints` that
    the only round it can be asked about is the last one.

    Implementations expose:

    - `checkpoints(rounds)`, the round numbers this fold can be scored at,
      ascending and ending at `rounds`;
    - `advance_to(round_number)`, grow or train so the model holds that many
      iterations, returning False once the objective has stopped adding;
    - `booster`, the `Booster` that holds the model;
    - `data(side)`, the `Dataset` of `'train'` or `'valid'`.

    See handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md) for the native work that would let every task
    take the incremental path.
    """

    def checkpoints(self, rounds):
        raise NotImplementedError

    def advance_to(self, round_number):
        raise NotImplementedError

    def data(self, side):
        return self._sets[side]


class _IncrementalFold(FoldModel):
    """A fold grown one round at a time, so every round can be scored.

    The booster starts empty and `update()` adds to it, which is the loop
    `Booster.update` documents. Each call recomputes the training raw scores
    from the model, so a run of R rounds costs R rescoring passes; that is
    the price of a per-round history until the trainer carries the state
    across calls (see the handoff).
    """

    def __init__(self, dtrain, dvalid, params, init_model):
        self.booster = train(params, dtrain, 0, init_model=init_model)
        self._sets = {_TRAIN: dtrain, _VALID: dvalid}
        # The rounds an `init_model` brought are not rounds of this run, so
        # `num_boost_round` counts from wherever its trees left off.
        self._base = self.booster.current_iteration()
        self._finished = False

    def checkpoints(self, rounds):
        return list(range(1, rounds + 1))

    def grown(self):
        return self.booster.current_iteration() - self._base

    def advance_to(self, round_number):
        while not self._finished and self.grown() < round_number:
            self._finished = self.booster.update(1)
        return not self._finished


class _RetrainFold(FoldModel):
    """A fold trained in one call, for an objective `update()` cannot
    continue. It can only be scored once, at the end of the run."""

    def __init__(self, dtrain, dvalid, params, rounds, init_model):
        self._params = params
        self._rounds = rounds
        self._init_model = init_model
        self._sets = {_TRAIN: dtrain, _VALID: dvalid}
        self.booster = None

    def checkpoints(self, rounds):
        return [rounds]

    def advance_to(self, round_number):
        if self.booster is None:
            self.booster = train(
                self._params,
                self._sets[_TRAIN],
                round_number,
                init_model=self._init_model,
            )
        return False


# -- the fitted folds ----------------------------------------------------


class CVBooster:
    """The models `cv(return_cvbooster=True)` fitted, one per fold.

    It is a sequence of `Booster`s, in fold order, and it forwards anything
    else you ask of it to every one of them:

        cvbooster.num_trees()                 # [n_fold_0, n_fold_1, ...]
        cvbooster.feature_importance("gain")  # one vector per fold
        cvbooster.predict(X)                  # one prediction per fold
        cvbooster.predict_mean(X)             # their mean, where defined

    `best_iteration` is the round early stopping chose, or -1 when it did
    not run; `predict` slices to it by default, so the prediction is the one
    the reported history describes rather than the one the extra rounds
    would give.

    One run has one of these. `cv()` builds it before the first round and
    keeps it: the object a callback reads as `env.model` is the object the
    caller gets back in `results["cvbooster"]`, holding the same `Booster`
    per fold rather than a second wrapper around it. That is why
    `best_iteration` is filled in at the end of the run rather than at
    construction, and why a callback may leave state on it.
    """

    def __init__(self, boosters=None, best_iteration=-1, fold_names=None):
        self.boosters = list(boosters or ())
        self.best_iteration = int(best_iteration)
        self.fold_names = list(fold_names or ())

    def __len__(self):
        return len(self.boosters)

    def __iter__(self):
        return iter(self.boosters)

    def __getitem__(self, index):
        return self.boosters[index]

    def __repr__(self):
        return (
            f"CVBooster({len(self.boosters)} folds, "
            f"best_iteration={self.best_iteration})"
        )

    def __getattr__(self, name):
        """Whatever a `Booster` answers, answered once per fold.

        Private and dunder names are not forwarded: they are this object's
        own business, and forwarding them breaks copying, pickling, and
        `hasattr` checks in ways that are hard to see.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        boosters = self.__dict__.get("boosters")
        if not boosters:
            raise AttributeError(
                f"CVBooster has no attribute {name!r} and no folds to "
                "forward it to"
            )
        attributes = [getattr(booster, name) for booster in boosters]
        if not all(callable(attribute) for attribute in attributes):
            return attributes

        def _broadcast(*args, **kwargs):
            return [attribute(*args, **kwargs) for attribute in attributes]

        _broadcast.__name__ = name
        _broadcast.__doc__ = (
            f"{name} on every fold, as a list in fold order."
        )
        return _broadcast

    def predict(self, data, raw_score=False, start_iteration=0,
                num_iteration=None):
        """One prediction per fold, as a list in fold order.

        `num_iteration` defaults to `best_iteration` when early stopping
        chose one, and to the whole ensemble otherwise. Averaging the folds
        is left to you: the mean is right for a regression and for
        probabilities, and wrong for a ranking, so this does not guess.
        """
        if num_iteration is None and self.best_iteration > 0:
            num_iteration = self.best_iteration
        return [
            booster.predict(
                data,
                raw_score=raw_score,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
            )
            for booster in self.boosters
        ]

    def predict_mean(self, data, raw_score=False, start_iteration=0,
                     num_iteration=None):
        """The across-fold mean of `predict`, where a mean is defined.

        The bagged prediction a cross-validated model is usually wanted for:
        the same slicing rules as `predict`, averaged over the folds. It is
        the right combination for a regression, for probabilities, and for
        raw scores on a common scale.

        It is refused for a ranking model, because a LambdaRank score is
        meaningful only within one model's ordering of one query: averaging
        the folds' scores mixes scales that were never comparable, and the
        result would still look like a ranking. Take `predict()` and combine
        the orderings deliberately.
        """
        if not self.boosters:
            raise ValueError(
                "this CVBooster has no fitted folds to average; it comes "
                "from cv(return_cvbooster=True)"
            )
        for booster in self.boosters:
            if getattr(booster, "_task", None) == _eval.RANKING:
                raise ValueError(
                    "ranking scores are not comparable across folds, so "
                    "their mean is not a ranking; use CVBooster.predict() "
                    "and combine the orderings yourself"
                )
        folds = self.predict(
            data,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
        )
        if _np is not None:
            return _np.asarray(folds, dtype=_np.float64).mean(axis=0)
        n = len(folds)
        first = folds[0]
        if len(first) and hasattr(first[0], "__len__"):
            return [
                [
                    math.fsum(fold[r][c] for fold in folds) / n
                    for c in range(len(first[r]))
                ]
                for r in range(len(first))
            ]
        return [math.fsum(fold[r] for fold in folds) / n
                for r in range(len(first))]

    def reset_parameter(self, new_parameters):
        """Refused: the fold boosters have no way to be told.

        A `reset_parameter()` schedule changes hyperparameters between
        rounds, which `Booster` does not expose. Accepting the call and
        training the next round with the old values would make a schedule
        look like it ran, so it raises instead. handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md) holds
        the requirement that would make it work.
        """
        raise NotImplementedError(
            "parameters cannot be reset between cv rounds: a fold is a "
            "Booster, and Booster has no reset_parameter() for update() to "
            "honor. Drop the reset_parameter() callback, or run the schedule "
            "through an estimator's fit()"
        )


# -- metrics and aggregation ---------------------------------------------


def _metric_list(metrics, feval, task, objective):
    """Every metric to score, normalized by the estimators' own reader.

    `metrics` and `feval` are LightGBM's two spellings and they compose:
    `feval` is appended, which is what lets a run score a built-in metric
    and a custom one together. Both go through `_metric_specs`, so the forms
    they accept are the forms `eval_metric=` accepts, down to the message an
    unusable one raises.
    """
    from . import _metric_specs

    if feval is None:
        return _metric_specs(metrics, task, objective)
    extra = _metric_specs(feval, task, objective)
    if metrics is None:
        return extra
    specs = _metric_specs(metrics, task, objective) + extra
    names = [spec[0] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("eval_metric names must be unique")
    return specs


def _score(fold, side, specs):
    """Every metric on one side of one fold, in `specs` order.

    The built-in metrics go through `Booster.eval`, so a cv number is the
    number `booster.eval(...)` reports for the same rows. A callable is
    handed the raw scores, which are read once per side however many
    callables there are.
    """
    dataset = fold.data(side)
    values = []
    raw = None
    for name, func, _higher, _watch, code in specs:
        if code is not None:
            values.append(fold.booster.eval(dataset, side, name)[0][2])
            continue
        if raw is None:
            raw = _flat_raw(fold.booster.predict(dataset, raw_score=True))
        result = func(dataset.get_label(), raw)
        values.append(_metric_value(result, name))
    return values


def _flat_raw(raw):
    """Raw scores as one flat sequence, which is the layout a custom metric
    is given for every task: `n_classes` values per row for a softmax model,
    one per row otherwise."""
    if _np is not None:
        return _np.asarray(raw, dtype=_np.float64).reshape(-1)
    first = raw[0] if len(raw) else None
    if hasattr(first, "__len__"):
        return [value for row in raw for value in row]
    return list(raw)


def _metric_value(result, name):
    """The number out of a custom metric's return value.

    A float, or LightGBM's `(name, value, is_higher_better)` triple of which
    only the value is read: the direction was declared in `metrics`, because
    early stopping needs it before the first evaluation.
    """
    if isinstance(result, (tuple, list)):
        if len(result) != 3:
            raise ValueError(
                f"eval_metric {name!r} returned {len(result)} values; return "
                "a float, or (name, value, is_higher_better)"
            )
        result = result[1]
    try:
        return float(result)
    except (TypeError, ValueError):
        raise ValueError(
            f"eval_metric {name!r} returned {result!r}, which is not a number"
        ) from None


def _mean_stdv(values):
    """The across-fold mean and population standard deviation, which is the
    spread LightGBM reports (it divides by the fold count, not by one less:
    the folds are the whole population being described)."""
    n = len(values)
    mean = math.fsum(values) / n
    if n == 1:
        return mean, 0.0
    variance = math.fsum((value - mean) ** 2 for value in values) / n
    return mean, math.sqrt(max(variance, 0.0))


# -- early stopping ------------------------------------------------------


class _Consensus:
    """The early-stopping rule, applied to the across-fold mean.

    One fold improving while the others do not is noise, so the watched
    quantity is the aggregate: the folds stop together, on the round the
    mean stopped improving, which is the round the returned history is
    truncated to.
    """

    def __init__(self, rounds, min_delta, watched):
        self.rounds = rounds
        self.min_delta = float(min_delta)
        self.watched = list(watched)
        self.best_score = {key: None for key, _ in self.watched}
        self.best_index = {key: 0 for key, _ in self.watched}
        self.triggered = None

    def observe(self, index, means):
        """Record one evaluated round. Returns the winning round's index in
        the history when the run should stop, and None to continue."""
        for key, higher in self.watched:
            value = means[key]
            best = self.best_score[key]
            if best is None or (
                value > best + self.min_delta
                if higher
                else value < best - self.min_delta
            ):
                self.best_score[key] = value
                self.best_index[key] = index
                continue
            if index - self.best_index[key] >= self.rounds:
                self.triggered = key
                return self.best_index[key]
        return None


# -- the driver ----------------------------------------------------------


def cv(
    params,
    train_set,
    num_boost_round=None,
    folds=None,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics=None,
    feval=None,
    init_model=None,
    fpreproc=None,
    seed=0,
    callbacks=None,
    eval_train_metric=False,
    return_cvbooster=False,
    early_stopping_rounds=None,
    min_delta=0.0,
    first_metric_only=False,
):
    """Cross-validate a parameter set and return the per-round history.

    `params` and `train_set` are `train()`'s, and every parameter is
    validated by the same code `train()` validates it with, so a run that
    `train()` would refuse is refused here with the same message.

    The returned dict holds, for every metric:

        results["valid <metric>-mean"]   across-fold mean, per round
        results["valid <metric>-stdv"]   across-fold spread, per round

    with `train <metric>-...` alongside when `eval_train_metric=True`, and
    `results["iterations"]` giving the round number each entry belongs to.
    `return_cvbooster=True` adds `results["cvbooster"]`, the `CVBooster`
    holding the fitted fold models.

    The module docstring covers how the folds are built, what `metrics` and
    `feval` accept, how early stopping reaches a consensus across the folds,
    and why nothing here slices a dataset that was already binned.
    """
    if not isinstance(train_set, Dataset):
        raise TypeError("train_set must be a mojotrees.Dataset")

    config = _Config(params, num_boost_round)
    config.check_dataset(train_set)
    rounds = config.rounds
    if rounds < 1:
        raise ValueError("cv() needs at least one boosting round")

    n_folds = int(nfold)
    if folds is None and n_folds < 2:
        raise ValueError("nfold must be at least 2")

    # The round count is resolved once, above; the per-fold runs take it as
    # an argument, so it must not also be in the parameter dict where the
    # two would be compared and found to disagree.
    fold_params = {
        key: value
        for key, value in dict(params or {}).items()
        if key not in _ROUND_ALIASES
    }

    specs = _metric_list(metrics, feval, config.task, config.objective)
    sides = [_TRAIN, _VALID] if eval_train_metric else [_VALID]

    from . import callback as _callback

    callbacks = list(callbacks or ())
    (
        stop_rounds,
        min_delta,
        callback_first_only,
        stopper,
    ) = _callback.resolve_early_stopping(
        callbacks, early_stopping_rounds, min_delta
    )
    first_metric_only = bool(first_metric_only) or callback_first_only
    if stop_rounds is not None and int(stop_rounds) < 1:
        raise ValueError("early_stopping_rounds must be positive")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must not be negative")

    if init_model is not None:
        # `booster_update` refuses a dataset binned differently from the one
        # the model was trained on, and a fold is binned over its own rows,
        # so every fold would be refused. Saying so here beats letting the
        # trainer say it once per fold.
        raise NotImplementedError(
            "init_model does not reach the folds: continued training needs "
            "the dataset the model was trained on, and each cv fold bins "
            "itself over its own rows, so the trainer refuses every one of "
            "them. See handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md)"
        )

    incremental = config.task != _eval.RANKING
    if not incremental:
        if callbacks:
            raise NotImplementedError(
                "callbacks need a per-round history, and a ranking cv has "
                "one round it can report: continued training does not cover "
                "LambdaRank, so each fold is trained once at the full round "
                "count. See handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md)"
            )
        if stop_rounds:
            raise NotImplementedError(
                "early stopping needs a per-round history, and a ranking cv "
                "has one round it can report: continued training does not "
                "cover LambdaRank, so each fold is trained once at the full "
                "round count. See handoffs/task15_cv.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task15_cv.md)"
            )

    if folds is None:
        splits = _generated_splits(
            train_set,
            config.task,
            n_folds,
            bool(stratified),
            bool(shuffle),
            seed,
        )
    else:
        splits = _given_splits(folds, train_set, config.task)
    for split in splits:
        split.check(train_set.num_data())

    models = []
    for split in splits:
        dtrain = _fold_dataset(train_set, split.train_rows, split.train_group)
        dvalid = _fold_dataset(train_set, split.test_rows, split.test_group)
        run_params = dict(fold_params)
        if fpreproc is not None:
            # After the split and per fold, so anything it fits sees this
            # fold's training rows and nothing else.
            dtrain, dvalid, run_params = _preprocessed(
                fpreproc, dtrain, dvalid, run_params, split.name
            )
        if incremental:
            models.append(
                _IncrementalFold(dtrain, dvalid, run_params, init_model)
            )
        else:
            models.append(
                _RetrainFold(dtrain, dvalid, run_params, rounds, init_model)
            )

    # The fitted folds, owned once for the whole run: the callbacks and the
    # returned history hand out this object rather than each building their
    # own wrapper around the same Boosters. A `_RetrainFold` has no Booster
    # until it is advanced, so the list is refreshed rather than snapshotted
    # (`_sync_folds`).
    cvbooster = CVBooster(fold_names=[split.name for split in splits])
    _sync_folds(cvbooster, models)

    # (side, history key, metric spec), in the order they are reported.
    keys = [(side, f"{side} {spec[0]}", spec) for side in sides
            for spec in specs]
    results = {"iterations": []}
    for _side, key, _spec in keys:
        results[f"{key}-mean"] = []
        results[f"{key}-stdv"] = []

    consensus = None
    if stop_rounds:
        watched = [
            (key, spec[2])
            for side, key, spec in keys
            if side == _VALID and spec[3]
        ]
        if first_metric_only:
            watched = watched[:1]
        if not watched:
            raise ValueError(
                "early stopping has no metric to watch: every entry of "
                "metrics opted out with use_for_early_stopping=False"
            )
        consensus = _Consensus(int(stop_rounds), min_delta, watched)

    checkpoints = models[0].checkpoints(rounds)
    best_index = None
    for index, round_number in enumerate(checkpoints):
        if callbacks:
            stop_exception = _run_phase(
                callbacks, cvbooster, fold_params, round_number, rounds,
                before=True, env_results=[],
            )
            if stop_exception is not None:
                # Stopped before this round was grown, so the history it
                # would have produced does not exist; the run keeps what it
                # had scored, which is nothing at all on the first round.
                best_index = None if index == 0 else index - 1
                break
        for model in models:
            model.advance_to(round_number)
        _sync_folds(cvbooster, models)

        means = {}
        for side in sides:
            per_fold = [_score(model, side, specs) for model in models]
            for position, spec in enumerate(specs):
                key = f"{side} {spec[0]}"
                mean, stdv = _mean_stdv(
                    [values[position] for values in per_fold]
                )
                results[f"{key}-mean"].append(mean)
                results[f"{key}-stdv"].append(stdv)
                means[key] = mean
        results["iterations"].append(round_number)

        if callbacks:
            stop_exception = _run_phase(
                callbacks,
                cvbooster,
                fold_params,
                round_number,
                rounds,
                before=False,
                env_results=[
                    ("cv_agg", key, means[key], spec[2])
                    for _side, key, spec in keys
                ],
            )
            if stop_exception is not None:
                best_index = _exception_index(
                    stop_exception, checkpoints, index
                )
                break
        if consensus is not None:
            best_index = consensus.observe(index, means)
            if best_index is not None:
                break

    best_iteration = -1
    if best_index is not None:
        best_iteration = checkpoints[best_index]
        for key in list(results):
            results[key] = results[key][: best_index + 1]
        # The stopper reports only what it decided; a callback that raised
        # its own EarlyStopException stopped the run without it.
        if stopper is not None and consensus is not None:
            watched_key = consensus.triggered
            if watched_key is not None:
                stopper.report(
                    best_iteration,
                    consensus.best_score[watched_key],
                    watched_key,
                    "cv_agg",
                )

    _sync_folds(cvbooster, models)
    cvbooster.best_iteration = best_iteration
    if return_cvbooster:
        # Added after the truncation above, which walks every entry of
        # `results` as a history list.
        results["cvbooster"] = cvbooster
    return results


def _sync_folds(cvbooster, models):
    """Point the run's `CVBooster` at the current fold models.

    `_IncrementalFold` has its `Booster` from construction and this is a
    no-op for it. `_RetrainFold` has none until it is advanced, so the one
    `CVBooster` the run owns is refreshed at each point it becomes
    observable rather than rebuilt around a fresh list.
    """
    cvbooster.boosters = [model.booster for model in models]
    return cvbooster


def _preprocessed(fpreproc, dtrain, dvalid, run_params, name):
    """`fpreproc(dtrain, dvalid, params)` and its return value, checked.

    LightGBM's contract, with the parameter dict included so a fold can
    change what it trains with (a class weight fitted on the fold's own
    labels, say). The datasets are unconstructed here, so replacing them
    replaces the rows that will be binned rather than rows already binned.
    """
    out = fpreproc(dtrain, dvalid, run_params)
    try:
        dtrain, dvalid, run_params = out
    except (TypeError, ValueError):
        raise ValueError(
            f"fpreproc must return (dtrain, dvalid, params); for fold "
            f"{name!r} it returned {out!r}"
        ) from None
    if not isinstance(dtrain, Dataset) or not isinstance(dvalid, Dataset):
        raise TypeError(
            f"fpreproc must return two mojotrees.Datasets; for fold {name!r} "
            f"it returned {type(dtrain).__name__} and "
            f"{type(dvalid).__name__}"
        )
    return dtrain, dvalid, dict(run_params or {})


def _run_phase(callbacks, cvbooster, params, round_number, rounds, before,
               env_results):
    """One phase's callbacks for one evaluated round.

    The environment is LightGBM's, with the run's `CVBooster` in the `model`
    slot so a callback can reach the folds, and `iteration` 0-based as it is
    everywhere else in this package. It is the same object every round and
    the same object the caller is handed, so a callback that stashes state
    on it finds the state again. The before-iteration phase runs before the
    round is grown and sees no results, because nothing has been scored yet;
    the after-iteration phase sees the across-fold means.

    A callback that raises `EarlyStopException` stops the run, as it does in
    `fit`, and the exception is returned rather than raised. Anything else
    it raises propagates unchanged, with its own type and traceback.
    """
    from .callback import CallbackEnv, EarlyStopException, _sorted_phase

    env = CallbackEnv(
        model=cvbooster,
        params=dict(params),
        iteration=round_number - 1,
        begin_iteration=0,
        end_iteration=rounds,
        evaluation_result_list=list(env_results),
    )
    for cb in _sorted_phase(callbacks, before=before):
        try:
            cb(env)
        except EarlyStopException as exc:
            return exc
    return None


def _exception_index(exception, checkpoints, index):
    """The history position an `EarlyStopException` points at.

    LightGBM's callbacks carry a 0-based round; anything outside the rounds
    scored so far is a callback describing a round this run does not have,
    so the round it stopped on stands in.
    """
    best = exception.best_iteration
    if best is None:
        return index
    try:
        return checkpoints.index(int(best) + 1)
    except ValueError:
        return index
