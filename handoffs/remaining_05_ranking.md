# Task 05 handoff: advanced ranking and unbiased LambdaRank

Files this lane created, and the only ones it touched:

- `src/mojoboost/ranking_advanced.mojo`
- `docs/RANKING_ADVANCED.md`
- `handoffs/remaining_05_ranking.md` (this file)

Nothing central or shared was edited. `ranking.mojo`, the objectives, the
metrics, the trainers, `Dataset`, the bindings, the Python package, and the
tests are untouched.

**Nothing was run.** No test, no build, no benchmark, no program. Every
validation named below is marked **UNRUN** and is a request, not a report.

## 0. Ownership decision

The task allowed `ranking_advanced.mojo` to be created only if no equivalent
module existed. The repository was inspected first:

| Candidate | What it holds | Equivalent? |
| --- | --- | --- |
| `src/mojoboost/ranking.mojo` | LambdaRank, NDCG, MAP, the ranker trainers. Its docstring lists "positional/unbiased-lambdarank extensions" as *not implemented* | No |
| `src/mojoboost/custom_metric.mojo` | `train_ranker_with_metrics`; imports `ranking`'s internals, adds no ranking mathematics | No |
| `src/mojoboost/metrics.mojo` | `METRIC_NDCG` / `METRIC_MAP` codes; `eval_metric_by_code` refuses both and points at `ranking.mojo` | No |
| `src/mojoboost/objective_registry.mojo` | `TASK_RANKING`, `NEEDS_GROUPS`, `NEEDS_CUTOFF`, `GRAD_LAMBDARANK`, `INIT_ZERO` - metadata only | No |
| `python/mojoboost/cv.py` | group-safe folds, Python side, for `cv()` only | No (wrong language and wrong layer) |

No module implements position bias, a custom gain vector, `eval_at` as data,
query weighting, pair sampling, or a Mojo-side group-safe fold or partition.
`ranking_advanced.mojo` was created and is the authoritative file for those.

## 1. What is connected inside the owned file

Not scaffolding. The following call graph is live in
`ranking_advanced.mojo` today:

```
fit_ranker_advanced
  -> groups_from_counts_sanitized -> sanitize_group_counts
                                  -> ranking.groups_from_counts
  -> binning.fit_bins -> BinMapper.transform
  -> train_ranker_advanced
       -> check_query_bagging      (bagging.check_bagging + the query rule)
       -> ranking._refresh_query_bag        (whole queries, per round)
       -> advanced_lambdarank_gradients
            -> PositionBiasState.adjust_scores
            -> ranking._inverse_max_dcgs | _inverse_max_dcgs_table
            -> ranking._discounts
            -> ranking._fill_lambdas       (default configuration)
               | _fill_lambdas_general     (custom gain / pair sampling)
                   -> ranking._argsort_desc_range, ranking._pair_sigmoid
            -> update_position_bias        (Newton step, LightGBM's terms)
       -> tree.grow_tree -> boosting.Booster(LAMBDARANK, base score 0.0)
  -> model.Model
  => FittedAdvancedRanker { model, bias }

ndcg_eval / map_eval
  -> check_advanced_rank_params -> ranking.check_ranker_params
  -> ranking._argsort_desc_range, ranking._sorted_gains, ranking._discounts,
     ranking.max_dcg
  -> query_weights (LightGBM's mean-of-rows query weight)
  => RankEval { cutoffs, values, per_query, n_degenerate, ... }

query_folds -> rows_of_queries -> ranking._expand_queries
partition_queries -> check_query_partition
audit_groups, prunable_queries, pair_budget
```

The routing predicate `AdvancedRankParams.uses_baseline_lambdas()` means the
default configuration - including the unbiased one - runs
`ranking._fill_lambdas` and no arithmetic from this module at all. See
`docs/RANKING_ADVANCED.md` section 3.

## 2. What this lane does **not** claim

Read this before touching `docs/LIGHTGBM_PARITY.md`.

- mojoboost does **not** implement unbiased LambdaRank. The code exists; the
  evidence does not.
- `lambdarank_position_bias_regularization` stays `deferred` (contract line
  ~421). `Dataset.position` stays `deferred` (contract line ~295).
  `label_gain` stays `different` (line ~420). `eval_at` stays `partial`
  (line ~432).
- `ranking_advanced.mojo` is exported from nothing. `tools/check_parity.py`
  check 7 resolves *public symbols*, and a module exported by nobody
  resolves to nothing, which is the correct answer: written, not delivered.
  **Do not add it to `src/mojoboost/__init__.mojo` before patch 10's
  evidence exists**, or check 7 will start reporting stale `deferred` rows
  that are not in fact stale.

## 3. Validation this lane owes, all UNRUN

None of these were written (the task forbade writing tests). They are the
minimum before any patch below lands in a user-visible position.

| # | Check | Why it is the one that matters |
| --- | --- | --- |
| V1 | `_fill_query_lambdas_general` with `LabelGain.default()`, `pair_sampling_rate = 1.0`, `max_dcg_cutoff = 0` equals `ranking._fill_query_lambdas` **bit for bit** on a fixed query | The whole anti-duplication argument. If it fails, there are two LambdaRanks |
| V2 | `advanced_lambdarank_gradients` with `PositionMap.absent()` and default params equals `ranking.lambdarank_gradients` bit for bit | The routing predicate does what it says |
| V3 | `ndcg_eval(...).values` equals `ranking.ndcg_at_cutoffs` and `map_eval(...).values` equals `ranking.map_at_cutoffs`, unweighted, default gains | The metrics did not fork |
| V4 | One position, `reg = 0`: the learned bias after N rounds equals `learning_rate * sum(D1/(abs(D2)+1e-3))` computed by hand | The Newton step is transcribed, not invented |
| V5 | Differential against LightGBM `objective=lambdarank` + `position` + `lambdarank_position_bias_regularization` on a fixed seeded dataset, comparing per-position biases and final NDCG | **The only thing that entitles anyone to change a parity row.** Needs `bench/compare_ranking.py` extended |
| V6 | `partition_queries` + `check_query_partition` at world sizes 1..8 over a ragged query set; `world_size > n_queries` yields empty ranks and still validates | The distributed contract |
| V7 | `query_folds` fold assignment equals `python/mojoboost/cv.py`'s for the same query count and the unshuffled order | Mojo and Python folds agree |
| V8 | Pair sampling: same `(seed, iteration, query)` gives the same kept set across two calls, and the mean lambda over many seeds tracks the unsampled lambda | Determinism and the `1/rate` rescaling |

---

# READY-TO-APPLY INTEGRATION PATCHES

Ten patches. Each names its owner, its dependency, and what it changes about
serialization and the public API. Apply in the numbered order; 1 and 4 are
independent of each other, everything from 5 on depends on 1 and 2.

---

## Patch 1 - fold the advanced round into `train_ranker`

**Owner file:** `src/mojoboost/ranking.mojo`
**Target symbols:** `train_ranker`, `train_ranker_with_valid`
**Dependency:** none. Do this first; patches 2, 5, 7, 8 build on it.

**Why.** `ranking_advanced.train_ranker_advanced` is a second outer loop,
and a second outer loop is the thing this round was told to fuse. It exists
only because the position biases are per-round state `train_ranker` has
nowhere to put. Give it somewhere to put them and the second loop goes away.

**Signature change.** Add two arguments to `train_ranker`, both defaulted so
every existing call site is unchanged:

```mojo
def train_ranker(
    data: BinnedMatrix,
    labels: List[Int],
    groups: RankGroups,
    params: BoosterParams,
    rank_params: RankerParams = RankerParams.default(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    advanced: AdvancedRankParams = AdvancedRankParams.default(),   # NEW
    positions: PositionMap = PositionMap.absent(),                 # NEW
    mut bias: PositionBiasState = ...,                             # NEW, see below
) raises -> Booster:
```

**A `mut` argument cannot carry a default**, so pick one of two shapes:

- *(recommended)* keep `train_ranker` as it is and add a sibling
  `train_ranker_positioned(...) raises -> TrainedAdvancedRanker` in
  `ranking.mojo` that takes `positions` by value and returns the state,
  exactly as `ranking_advanced.train_ranker_advanced` does today. Then move
  the body of `train_ranker_advanced` into it verbatim and have
  `ranking_advanced` re-export the name. One loop, in `ranking.mojo`.
- or make `bias` a required `mut` argument on a new overload.

**Call site.** Inside the loop, replace the `_fill_lambdas(...)` call with:

```mojo
        advanced_lambdarank_gradients(
            raw, labels, groups, positions, bias, grad, hess,
            advanced, sample_weight, i, params.learning_rate,
        )
```

`inverse_max_dcg` and `discounts` are then computed inside that call and the
two hoisted locals in `train_ranker` become dead. **Do not delete the
hoisting without measuring**: it currently computes them once for the whole
run, and moving them inside makes them per round. The cheap fix is an
overload of `advanced_lambdarank_gradients` that accepts precomputed `inv`
and `discounts`; add it in `ranking_advanced.mojo`, which this lane owns.

**State flow.** `bias` is read at the top of every round (through
`adjust_scores`) and written at the bottom (through `update_position_bias`).
It is *not* touched by bagging: the Newton step sums over **every** row,
in-bag or not, because the biases describe the log and not the bag - the
same rule `boosting.mojo` already follows for the base score.

**Errors.** `train_ranker` gains: labels outside the gain vector
(`check_relevance_labels`), a position map whose length disagrees with
`n_rows` (`check_positions`), a bias state sized for a different position
count, and `check_advanced_rank_params`. All raise before the first tree.

**Fallback.** With `PositionMap.absent()` and
`AdvancedRankParams.default()`, the round is byte for byte the round
`train_ranker` runs today - that is check V2. If V2 fails, revert to the
sibling-function shape and leave `train_ranker` alone.

**Serialization effect.** None. The returned `Booster` is unchanged: trees,
base score `0.0`, `LAMBDARANK`, monotone constraints. No format change, no
version bump. The biases are not written (see patch 3).

**Public API effect.** None until patch 7.

**Validation, UNRUN:** V1, V2. Add to `tests/test_ranking.mojo`; keep the
existing ranking suite passing unchanged, which is itself the regression
test for the fallback.

---

## Patch 2 - `position` on `Dataset`

**Owner file:** `src/mojoboost/trainset.mojo`
**Target symbols:** `Dataset.__init__`, `train_dataset_ranker`
**Dependency:** patch 1.

**Why.** LightGBM's `Dataset.position` is the field that carries position
ids, and `docs/LIGHTGBM_PARITY.md` line ~295 records it as
`deferred - needs unbiased LambdaRank`. It is the same shape as `group`: a
column supplied at construction, validated against `n_rows` there, and read
back but never mutated.

**Signature change.** One new keyword argument, defaulted to empty, placed
after `group` so the existing positional order is untouched:

```mojo
    def __init__(
        out self,
        features: List[Float64],
        n_rows: Int,
        n_features: Int,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var position: List[Int] = [],          # NEW
        var init_score: List[Float64] = [],
        ...
```

and one new field, `var position: List[Int]`, beside `group`.

**Validation, in `__init__` beside the existing `group` block:**

```mojo
        if len(position) != 0:
            if len(position) != n_rows:
                raise Error("position must have one entry per row")
            if len(group) == 0:
                raise Error(
                    "position describes where a document was shown, which"
                    " only means something inside a query; supply group too"
                )
            for r in range(len(position)):
                if position[r] < 0:
                    raise Error("position ids must be nonnegative")
```

Dense-ify with `ranking_advanced.positions_from_codes`, which accepts
arbitrary integer codes and returns both the dense map and the code table;
store the `PositionMap`, or store the raw column and densify in
`train_dataset_ranker`. **Store the raw column.** A `Dataset` is
constructed once and trained on many times, and the dense id assignment is
first-appearance order over the whole column, which does not change between
runs - so densifying at train time costs one pass and keeps `Dataset` free
of a `ranking_advanced` import.

**Call site.** `train_dataset_ranker` gains:

```mojo
    var positions = PositionMap.absent()
    if len(dataset.position) != 0:
        positions = positions_from_codes(dataset.position).positions
```

and passes `positions` plus a `PositionBiasState.for_positions(positions)`
into patch 1's entry point.

**Errors.** The existing `init_score` refusal
(`"init_score is not supported for ranking"`) stays. Add nothing new at
train time; construction has already validated.

**Fallback.** An empty `position` is exactly today's behavior.

**Serialization effect.** None on the model. `Dataset` is not serialized.

**Public API effect.** `Dataset(position=)` becomes constructible from Mojo.
Not from Python until patch 7.

**Validation, UNRUN:** a `tests/test_trainset.mojo` case that a positioned
dataset trains, that `position` reads back, and that `position` without
`group` raises.

---

## Patch 3 - refuse continuation of a positioned ranker

**Owner files:** `src/mojoboost/boosting.mojo`, `python/mojoboost/basic.py`
**Target symbols:** `train_more` (Mojo), `Booster._require_trainable`
(Python, `basic.py:776`)
**Dependency:** patch 2.

**Why.** The position biases are training state that no model file carries.
Resuming a positioned run from zero biases fits the next round's trees
against a correction the earlier rounds already applied - the trees are then
double-corrected and nothing raises. `docs/RANKING_ADVANCED.md` section 2
has the argument.

**Python.** `basic.py:784` already raises `NotImplementedError` for
`_eval.RANKING`, so **nothing needs to change today**. When patch 1 of the
*task 15* handoff lands (`train_ranker_more`, which removes that branch),
the branch must be replaced rather than deleted:

```python
        if self._task == _eval.RANKING and self._train_set.get_position():
            raise NotImplementedError(
                "continued training does not cover a positioned ranker: the "
                "per-position biases are training state a model file does "
                "not carry, so resuming would fit new trees against a "
                "correction the earlier rounds already made"
            )
```

**Mojo.** If and when a `train_ranker_more` is added, it must take a
`mut PositionBiasState` or refuse a nonempty `dataset.position`. Prefer
taking the state: a caller that kept the `TrainedAdvancedRanker` this lane
returns has it.

**Errors.** `NotImplementedError` from Python, `Error` from Mojo, both
naming the reason rather than the symptom.

**Fallback.** The current unconditional refusal is strictly safer than the
conditional one; landing this patch late costs nothing.

**Serialization effect.** This patch exists *because* of a serialization
gap. Recording it: a saved unbiased-LambdaRank model is an ordinary ranking
model file, and the biases are gone. If a future format version wants to
carry them it is a new optional section and a version bump, and it should be
justified by a use case (serving does not need them - see
`docs/RANKING_ADVANCED.md` section 2).

**Public API effect.** One more `NotImplementedError` case on
`Booster.update`.

**Validation, UNRUN:** a `python/tests/` case that a positioned ranker
refuses `update()`.

---

## Patch 4 - query-aligned partitioning in `distributed.mojo`

**Owner file:** `src/mojoboost/distributed.mojo`
**Target symbol:** new `partition_rows_at`, beside `partition_rows`
(`distributed.mojo:124`)
**Dependency:** none.

**Why.** `partition_rows` cuts at `r * n_rows // W`, which lands inside a
query whenever the row count does not divide evenly. A rank holding half a
query normalizes its lambdas against the maxDCG of the half it can see, and
**no all-reduce catches it**, because the ranking objective needs no
cross-rank reduction at all. See `docs/RANKING_ADVANCED.md` section 7.

**Signature.** Same body as `partition_rows` with the boundaries supplied
rather than computed:

```mojo
def partition_rows_at(
    data: BinnedMatrix,
    target: List[Float64],
    boundaries: List[Int],
    weight: List[Float64] = [],
) raises -> List[DataShard]:
```

`boundaries` has `world_size + 1` entries, `boundaries[0] == 0`,
`boundaries[world_size] == data.n_rows`, nondecreasing. Rank r owns rows
`[boundaries[r], boundaries[r + 1])`. Equal consecutive entries mean an
empty shard, which `distributed.mojo` already supports.

**Refactor.** `partition_rows` becomes

```mojo
def partition_rows(data, target, world_size, weight = []) raises -> List[DataShard]:
    var b = List[Int](capacity=world_size + 1)
    for r in range(world_size + 1):
        b.append(r * data.n_rows // world_size)
    return partition_rows_at(data, target, b^, weight)
```

which keeps one copying loop rather than two. The existing world-size-1
equivalence test covers the refactor.

**Call site.** A ranking caller builds the boundaries from
`ranking_advanced.partition_queries`:

```mojo
    var parts = partition_queries(groups, world_size)
    check_query_partition(parts, groups)
    var b = List[Int](capacity=world_size + 1)
    b.append(0)
    for r in range(world_size):
        b.append(parts[r].row_end)
    var shards = partition_rows_at(data, target, b^, weight)
```

and each rank's own `group` array comes from `partition_group_counts`.

**Three more things the distributed ranking contract needs, none of them
code in this patch:**

1. **No gradient reduction.** Lambdas are per query and queries do not
   cross ranks, so a rank computes its own gradients complete. Only the
   histograms reduce, exactly as today.
2. **No base-score reduction.** Ranking boosts from `0.0`
   (`objective_registry.INIT_ZERO`), so `_distributed_base_score` must be
   skipped, not called with ranking targets.
3. **Metric reduction is weighted.** The mean NDCG over a partition is
   `sum(w_q * ndcg_q) / sum(w_q)` per rank, reduced as a numerator and a
   denominator - **two f64 reductions, never a mean of means**, which would
   weight ranks by their rank index rather than by their query count.
   `RankEval.total_weight` is the denominator to send.

**Errors.** `partition_rows_at` raises on boundaries that are not
nondecreasing, do not start at 0, or do not end at `n_rows` - identically on
every rank, since every rank computes the same boundaries from the same
`groups`.

**Fallback.** `partition_rows` keeps its current behavior exactly. Ranking
simply must not use it.

**Serialization effect.** None.

**Public API effect.** One new Mojo symbol; not exported to Python (the
distributed prototype is not exposed).

**Validation, UNRUN:** V6, plus the existing world-size-1 equivalence test
still passing after the refactor.

---

## Patch 5 - parameter names in `params.mojo`

**Owner file:** `src/mojoboost/params.mojo`
**Target symbol:** `_MOJO_API_ONLY` (`params.mojo:86`)
**Dependency:** patches 1, 2.

**Why.** `label_gain`, `sigmoid`, `eval_at`, and
`lambdarank_truncation_level` are already in `_MOJO_API_ONLY`. The new names
belong there too: a parameter string cannot carry a position column any more
than it can carry a group array, so these are Mojo-API-only for the same
reason and must be reported as "a real feature, asked for the wrong way"
rather than as unknown.

**Change.** Append to the `_MOJO_API_ONLY` string, keeping its line shape:

```mojo
    " lambdarank_position_bias_regularization position pair_sampling_rate"
    " pair_sampling_seed max_dcg_cutoff"
```

**Do not add these to `SUPPORTED_KEYS`.** `params_names_mojo_api_only`
already reports the right thing for anything in `_MOJO_API_ONLY`.

**Errors.** A user writing
`objective=lambdarank lambdarank_position_bias_regularization=0.01` still
hits the existing `lambdarank` refusal first (`params.mojo:176`), which is
correct and should not be softened.

**Fallback / serialization / public API effect:** none. This patch only
changes an error message's category.

**Validation, UNRUN:** a `tests/` case asserting
`params_names_mojo_api_only("lambdarank_position_bias_regularization=0.1")`
is `True`.

---

## Patch 6 - `eval_at` as a list in the registry

**Owner file:** `src/mojoboost/objective_registry.mojo`
**Target symbols:** `metric_needs` (~line 943), and the `NEEDS_CUTOFF`
documentation
**Dependency:** none, but pointless before patch 7.

**Why.** `metric_needs(METRIC_NDCG)` returns `NEEDS_GROUPS | NEEDS_CUTOFF`
(singular). `ranking_advanced.ndcg_eval` reports every position in `eval_at`
in one pass, which is what LightGBM's `eval_at` means. The flag's *meaning*
needs to widen; its *value* must not change.

**Change.** No new bit and no renumbering - `NEEDS_CUTOFF` crosses the
Python boundary as an integer. Only the comment at `objective_registry.mojo`
line ~205 changes, from "a cutoff" to:

```mojo
comptime NEEDS_CUTOFF = 8
"""One or more evaluation positions (LightGBM's `eval_at`). A caller that
supplies a single cutoff gets a single value; `ranking_advanced.ndcg_eval`
takes the whole list and returns one value per position."""
```

**Errors / fallback / serialization / public API effect:** none. Purely
documentary, which is why it is cheap and why it must not become a
renumbering.

**Validation, UNRUN:** none needed; `tools/check_parity.py` already resolves
the symbol.

---

## Patch 7 - the binding surface

**Owner file:** `bindings/_mojoboost.mojo`
**Target symbols:** `_parse_rank_params`, `_group_counts`, `fit_ranker`,
`fit_ranker_with_metrics`, `train_dataset_ranker`, `dataset_create`,
`eval_metric`
**Dependency:** patches 1, 2.

*Symbols, not line numbers.* This file is being edited by another lane in
this same round (`git status` shows it modified), and its line numbers moved
while this handoff was being written. Find the symbols by name.

**Why.** Everything above is Mojo-only until this patch. The ranking params
already cross as a dict; three more keys and one more address is the whole
change.

**New keys in the params dict**, read in `_parse_rank_params`:

```mojo
def _parse_rank_params(params: PythonObject) raises -> RankerParams:
    ...  # unchanged

def _parse_advanced_rank_params(
    params: PythonObject
) raises -> AdvancedRankParams:
    """The advanced ranking config. Every key is optional; a missing key
    takes `AdvancedRankParams.default()`'s value, so an old wrapper's dict
    still trains the LightGBM-default round."""
    var adv = AdvancedRankParams.default()
    adv.base = _parse_rank_params(params)
    adv.position_bias_regularization = Float64(
        py=params["lambdarank_position_bias_regularization"]
    )
    adv.pair_sampling_rate = Float64(py=params["pair_sampling_rate"])
    adv.pair_sampling_seed = Int(py=params["pair_sampling_seed"])
    adv.eval_at = _int_list_from_f64(
        Int(py=params["eval_at_addr"]), Int(py=params["n_eval_at"])
    )
    if len(adv.eval_at) == 0:
        adv.eval_at = default_eval_at()
    return adv^
```

`_int_list_from_f64` is the helper `_group_counts` already uses - **not**
`_int_list`, which reads an int64 buffer (SciPy's indices/indptr). Every
integer array at this boundary travels as float64, so reuse the same reader
rather than writing a second one, and have the Python side build the buffer
with `_as_f64_vector` exactly as `_group_buffer` does.

**Positions** ride exactly as `group` does - `position_addr` /
`n_position_rows`, a float64 buffer of integral codes, densified Mojo-side:

```mojo
def _position_codes(params: PythonObject) raises -> List[Int]:
    if Int(py=params["position_addr"]) == 0:
        return List[Int]()
    return _int_list_from_f64(
        Int(py=params["position_addr"]), Int(py=params["n_position_rows"])
    )
```

**Call sites.** `fit_ranker` and `train_dataset_ranker` swap
`_parse_rank_params(params)` for `_parse_advanced_rank_params(params)` and
pass `positions_from_codes(_position_codes(params)).positions` when the
codes are nonempty.

`dataset_create` reads `position_addr` into the new `Dataset` field from
patch 2, in the block that already reads `group_addr` (`var group =
List[Int]()` / `if Int(py=params["group_addr"]) != 0:`).

**Return shape.** `fit_ranker` currently returns a model handle.
`train_ranker_advanced` returns `TrainedAdvancedRanker`. Return the model
handle unchanged and add a separate `ranker_position_bias(handle)` accessor
**only if a caller needs it** - the biases are training state, and the
minimal binding does not expose them at all. Prefer not exposing them in the
first version.

**Errors.** A missing dict key raises a Python `KeyError` through the
binding, which is why every new key must be written by the Python side
unconditionally (patch 8), defaulted there rather than here.

**Fallback.** A dict with `position_addr = 0`, `pair_sampling_rate = 1.0`,
and `lambdarank_position_bias_regularization = 0.0` trains exactly today's
model.

**Serialization effect.** None. The saved model is unchanged.

**Public API effect.** None on its own; patch 8 is what users see.

**Validation, UNRUN:** `python/test_python_api.py` round trip with the new
keys present and at their defaults, asserting the fitted model is identical
to one fitted without them.

---

## Patch 8 - `MojoBoostRanker(position=..., ...)`

**Owner file:** `python/mojoboost/__init__.py`
**Target symbols:** `MojoBoostRanker.__init__`,
`MojoBoostRanker._rank_params`, `MojoBoostRanker.fit`, `_group_buffer`, and
a new module-level `_position_buffer` beside it
**Dependency:** patch 7.

*Symbols, not line numbers.* This file is also being edited by another lane
in this round; its line numbers shifted by roughly 290 lines while this
handoff was being written. Find the symbols by name.

**Constructor.** Four new keyword arguments, all defaulted to LightGBM's
values so `get_params`/`set_params`/`clone` keep working:

```python
    def __init__(
        self,
        lambdarank_truncation_level=30,
        sigmoid=1.0,
        lambdarank_norm=True,
        ndcg_eval_at=5,
        lambdarank_position_bias_regularization=0.0,   # NEW
        eval_at=None,                                  # NEW, None -> [1,2,3,4,5]
        pair_sampling_rate=1.0,                        # NEW, mojoboost extension
        pair_sampling_seed=5,                          # NEW
        **kwargs,
    ):
```

**`_rank_params`** gains the range checks beside the existing three and
writes every key unconditionally (patch 7 depends on that):

```python
        reg = float(self.lambdarank_position_bias_regularization)
        if reg < 0.0:
            raise ValueError(
                "lambdarank_position_bias_regularization must be nonnegative"
            )
        rate = float(self.pair_sampling_rate)
        if not (0.0 < rate <= 1.0):
            raise ValueError("pair_sampling_rate must be in (0, 1]")
        cutoffs = [1, 2, 3, 4, 5] if self.eval_at is None else [
            int(k) for k in self.eval_at
        ]
        if any(k < 1 for k in cutoffs):
            raise ValueError("eval_at positions must be positive")
        if int(self.ndcg_eval_at) not in cutoffs:
            raise ValueError(
                "ndcg_eval_at must appear in eval_at, so the cutoff early "
                "stopping watches is one of the cutoffs the run reports"
            )
        params["lambdarank_position_bias_regularization"] = reg
        params["pair_sampling_rate"] = rate
        params["pair_sampling_seed"] = int(self.pair_sampling_seed)
        eb = _as_f64_vector(cutoffs, len(cutoffs), "eval_at")
        params["eval_at_addr"] = _addr(eb)
        params["n_eval_at"] = len(cutoffs)
        return params, eb   # the buffer must outlive the call
```

Note the return-shape change: `_rank_params` currently returns `params`, and
the `eval_at` buffer has to stay referenced while its address is in use, the
same rule `_group_buffer` follows. Either return the
buffer alongside or stash it on `self` for the duration of `fit`.

**`fit(position=None)`**, validated by a new `_position_buffer` modeled on
`_group_buffer`:

```python
def _position_buffer(position, n_rows):
    """Validated float64 buffer of per-row position codes. Must stay
    referenced while its address is in use."""
    if position is None:
        return None
    values = list(position)
    if len(values) != n_rows:
        raise ValueError(
            f"position must have one entry per row: got {len(values)}, "
            f"X has {n_rows}"
        )
    codes = []
    for value in values:
        code = float(value)
        if code != int(code):
            raise ValueError(f"position codes must be integers, got {value!r}")
        codes.append(code)
    return _as_f64_vector(codes, len(codes), "position")
```

and in `fit`, beside the existing `gb = _group_buffer(group, n_rows)`:

```python
        pb = _position_buffer(position, n_rows)
        ...
        params["position_addr"] = 0 if pb is None else _addr(pb)
        params["n_position_rows"] = 0 if pb is None else n_rows
```

**Errors.** `position` given without `group` must raise the same message
patch 2 raises. `eval_set` + `position` is fine: a validation set has no
positions, and the metric is computed on unadjusted scores anyway.

**Fallback.** `position=None` and the four defaults reproduce today's fit
exactly - assert that in the test rather than assuming it.

**Serialization effect.** `save`/`load` are unchanged and the biases are not
saved. Document that in the `MojoBoostRanker` docstring, one sentence, next
to the existing note that "query boundaries are a property of the training
data, not of the fitted model".

**Public API effect.** Four new constructor parameters and one new `fit`
argument. `docs/LIGHTGBM_PARITY.md` line ~413 (`sigmoid`) is the row that
lists the ranker's constructor parameters; it does not change status, only
its note.

**Validation, UNRUN:** V5 through the Python path;
`python/tests/test_params.py` for the new validation messages;
`test_python_api.py` for the identical-fit fallback.

---

## Patch 9 - point the Python ranking folds at `query_folds`

**Owner file:** `python/mojoboost/cv.py`
**Target symbols:** `_chunk_folds` / `_rows_of_queries` (lines ~286-395)
**Dependency:** patches 7, 8. **Optional** - `cv.py` is correct today.

**Why.** `cv.py` already splits ranking folds on whole queries, in Python,
and `ranking_advanced.query_folds` does the same thing in Mojo. Two
implementations of one rule is one too many, and the Mojo one is the one a
Mojo caller can reach.

**Only do this if a binding for `query_folds` is added.** Without one this
patch means crossing the boundary for a list of integers, which is worse
than the duplication. If it is added, the contract to preserve exactly:

- fold `f` holds queries `[f * Q // K, (f + 1) * Q // K)` of the query order
- `shuffle` permutes queries, never rows
- both sides carry rows *and* group arrays
- a query in neither side, or in both, raises

`ranking_advanced.query_folds` implements all four. **V7 is the check that
they agree**, and it should be written *before* this patch as a pure
comparison, so that the duplication is proven redundant before it is
removed.

**Errors / fallback.** `cv.py`'s existing messages
(`_check_whole_queries`, `_group_for_rows`) are better than the Mojo ones -
they name the offending query. Keep them; call `query_folds` only for the
assignment.

**Serialization effect:** none. **Public API effect:** none;
`cv(...)` results are unchanged.

**Validation, UNRUN:** V7, then `python/tests/parallel/test_cv.py`
unchanged and still passing.

---

## Patch 10 - the parity contract, last

**Owner file:** `docs/LIGHTGBM_PARITY.md`
**Dependency:** **V5 must exist and pass.** Everything else is secondary.

**Do not touch these rows until then.** For the record, what each becomes
once the evidence exists:

| Row | Line | Today | After V5 passes |
| --- | --- | --- | --- |
| `lambdarank_position_bias_regularization` | ~421 | `deferred` - "Part of unbiased LambdaRank, which is out of v1" | `supported`, citing `src/mojoboost/ranking_advanced.mojo` and the differential bench |
| `Dataset.position` | ~295 | `deferred` - "Needs unbiased LambdaRank, task 12" | `supported`, citing `src/mojoboost/trainset.mojo` |
| `label_gain` | ~420 | `different` - fixed at `2^i - 1` | `partial`: a custom vector is accepted, with the nondecreasing and `gains[0] == 0` rules stated as intentional differences |
| `eval_at` | ~432 | `partial` - a single cutoff | `supported` if patch 8 lands with the list |
| `rank_xendcg` | ~472 | `different` | **unchanged.** This lane did not implement it, and `params.mojo:218` should keep saying so |

**`tools/check_parity.py` interactions, both of which will bite:**

- `WATCHES["Dataset.position"] = ["pymethod:Dataset.position"]`
  (`check_parity.py:479`). Check 7 resolves that name; patch 8 does not add
  a `Dataset.position` *method*, so the watch will keep resolving to nothing
  and the row will keep looking correctly deferred. When the row is upgraded,
  the watch must move out of the deferred set, or check 7 will contradict
  the new status.
- Check 5 resolves Mojo symbols against `src/mojoboost/__init__.mojo`. Any
  contract row citing a `ranking_advanced` symbol requires that symbol to be
  exported. Export at that point, and not before (section 2).

**Suggested export block for `src/mojoboost/__init__.mojo`, for when that
time comes** (not this lane's file, and deliberately not applied):

```mojo
from .ranking_advanced import (
    AdvancedRankParams,
    GroupAudit,
    LabelGain,
    PositionBiasState,
    PositionMap,
    QueryFold,
    QueryPartition,
    RankEval,
    advanced_lambdarank_gradients,
    audit_groups,
    check_advanced_rank_params,
    map_eval,
    ndcg_eval,
    partition_queries,
    positions_from_codes,
    query_folds,
    query_weights,
    train_ranker_advanced,
)
```

---

## Appendix - things deliberately not done

- **`rank_xendcg`.** Out of scope, and `params.mojo:218` reports it by name
  with a correct reason. Adding it is a separate objective, not a variant of
  this one.
- **A GPU path.** `docs/LIGHTGBM_PARITY.md` line ~478 records LambdaRank as
  CPU-only. The pair loop is `O(truncation * cnt)` per query with a data
  dependent inner bound, which is a different kernel shape from the
  histogram work, and nothing here changes that.
- **Exposing the learned biases.** They are training state and serving does
  not need them (`docs/RANKING_ADVANCED.md` section 2). Patch 7 deliberately
  does not add an accessor.
- **Writing tests.** The task forbade it. Section 3 is what should be
  written, in that order, and V1/V2/V3 are cheap enough to write before
  patch 1 lands.
