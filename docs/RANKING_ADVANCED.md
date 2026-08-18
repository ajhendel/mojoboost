# Advanced ranking: unbiased LambdaRank and the query-level contracts

`src/mojotrees/ranking.mojo` is mojotrees's LambdaRank. It is the
authoritative implementation, it is tested, it is benchmarked against
LightGBM by `bench/compare_ranking.py`, and nothing in this document
replaces any part of it.

`src/mojotrees/ranking_advanced.mojo` is the layer above it, and it holds
the ranking features that could not be expressed as a call into the existing
one. This document says what those are, what each one means precisely, what
is connected today, and - the part that matters most - what is **not
evidence** for anything.

## 0. Status, in one place

**Corrected 2026-08-18. Four of the six status claims that stood here were
wrong, and all four under-claimed: they argued that something is not
implemented when it is.** Each row below quotes what it used to say and then
says what the code says, so the correction can be checked rather than taken.

The bolded headline was the first of the four. It read:

> **Nothing in `ranking_advanced.mojo` has been run.** No test, no benchmark,
> no differential comparison against LightGBM, no build. Static inspection
> only.

`tests/test_ranking_advanced.mojo` exists and holds five tests
(`test_default_is_not_advanced_and_matches_train_ranker`,
`test_custom_label_gain_changes_the_ensemble_and_still_ranks`,
`test_metric_trainer_honors_the_advanced_parameters`,
`test_position_column_learns_a_bias_and_trains`,
`test_pair_sampling_and_cutoff_are_validated`), each of which trains an
ensemble, so the module builds and runs. What is still absent is the
*differential comparison against LightGBM*, and that is the only half of the
sentence worth keeping.

| Claim | Status |
| --- | --- |
| mojotrees implements unbiased LambdaRank | **Yes, and reported as such.** This row used to read "**No.** Not claimed, not supported, and `docs/LIGHTGBM_PARITY.md` keeps `lambdarank_position_bias_regularization` and `Dataset.position` at `deferred`", which is false in the parity document it cites: `Dataset.position` is `supported` there and `lambdarank_position_bias_regularization` is `partial`, the `partial` being that it is refused together with `eval_set` and that no LightGBM differential exists, so numeric parity is not claimed |
| The position-bias update matches LightGBM's `UpdatePositionBiasFactors` | Transcribed from it, term for term, and argued in section 2. **Still unverified against LightGBM**, and this row was right. `test_position_column_learns_a_bias_and_trains` proves the update runs, learns a per-position bias that moves off zero, and leaves a ranking NDCG above 0.9; it does not compare a single factor against LightGBM's, and nothing does |
| The generalized pair loop reduces to `ranking._fill_query_lambdas` | **Verified, in-tree.** This row used to read "**Unverified**, and the check that would verify it is listed UNRUN in `handoffs/remaining_05_ranking.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_05_ranking.md)`". `test_default_is_not_advanced_and_matches_train_ranker` is that check and it has run: on the default `AdvancedRankParams`, `train_ranker_advanced` and `train_ranker` produce the same tree count and bit-identical `predict_row` over every row. The handoff file it cited no longer exists (`ls handoffs/` lists five files and that is not one of them), so the UNRUN pointer was dangling as well as stale |
| `ndcg_eval` agrees with `ranking.ndcg_at_cutoffs` | Argued in section 5. **Still unverified**, and this row was right. `ndcg_eval` (`src/mojotrees/ranking_advanced.mojo:1487`) is named by no test in `tests/` and by no binding; the ranking tests call `ranking.ndcg` instead |
| The module is reachable from Python, the C API, or the CLI | **Yes, from Python.** This row used to read "No. It is not exported from `src/mojotrees/__init__.mojo` and no binding names it". The second half is false: `bindings/_mojotrees.mojo:370-377` imports six symbols from this module by name (`AdvancedRankParams`, `LabelGain`, `PositionMap`, `advanced_ranking_requested`, `fit_ranker_advanced`, `positions_from_codes`) and `fit_ranker` calls `fit_ranker_advanced` at :3636 whenever `advanced_ranking_requested` is true. Only the first half survives: the module is still not re-exported from `src/mojotrees/__init__.mojo`, so the Mojo package namespace does not carry it |

What the surviving `__init__.mojo` half means, and it is narrower than the
old text made it sound. `tools/check_parity.py` resolves *public symbols* to
decide whether a `deferred` row has gone stale, so a module that is exported
by nobody resolves to nothing there. That was the right answer while the
feature was written and not delivered. It is no longer the state of the
world: the Python surface reaches this file through the binding, and
`docs/LIGHTGBM_PARITY.md` already carries the rows as `supported` and
`partial` on that evidence. The remaining gap is the Mojo-side re-export and
the LightGBM differential, not reachability.

## 1. Ownership decision

The task allowed for `ranking_advanced.mojo` to be created **only if no
equivalent module already existed**. The repository was inspected before
anything was written:

- `src/mojotrees/ranking.mojo` - LambdaRank, NDCG, MAP, the ranker trainers.
  Its own docstring lists "positional/unbiased-lambdarank extensions" under
  *INTENTIONAL DIFFERENCES FROM LightGBM*, i.e. as something **that file**
  does not do. That docstring was corrected on 2026-08-18 to say so
  explicitly, because as written it read as a statement about the library and
  the library does implement position bias, here.
- `src/mojotrees/custom_metric.mojo` - `train_ranker_with_metrics`, a
  metric-callback trainer that imports `ranking`'s internals and adds no
  ranking mathematics of its own.
- `src/mojotrees/metrics.mojo` - `METRIC_NDCG` and `METRIC_MAP` are numbered
  here, but `eval_metric_by_code` explicitly refuses both and points at
  `ranking.mojo`.
- `src/mojotrees/objective_registry.mojo` - `TASK_RANKING`, `NEEDS_GROUPS`,
  `NEEDS_CUTOFF`, `GRAD_LAMBDARANK`, `INIT_ZERO`: metadata about ranking,
  no ranking implementation.
- `python/mojotrees/cv.py` - group-safe folds, in Python, for the Python
  `cv()` entry point only.

No module implements position bias, a custom gain vector, multiple
evaluation positions as data, query weighting, pair sampling, or a Mojo-side
group-safe fold or partition. `ranking_advanced.mojo` was therefore created,
and it is the authoritative file for exactly those things.

## 2. Unbiased LambdaRank

### What the problem is

Training data for ranking is usually a click log, and a click log is
positionally biased: a document shown at slot 1 is clicked more than an
equally relevant document shown at slot 8, because it was seen more. A model
fitted directly on that log learns the layout of the result page as much as
it learns relevance.

Unbiased LambdaRank adds one learned scalar per position, in score space,
absorbing the part of the score that the slot explains. The trees are then
fitted to what is left.

### The round

`ranking_advanced.advanced_lambdarank_gradients` does exactly this, in this
order:

1. `adjusted[r] = raw[r] + b[position[r]]`
2. the ordinary LambdaRank pair loop on `adjusted`
3. sample weights multiply the per-row lambdas and hessians
4. one Newton step on the biases, from those weighted lambdas:

```
D1[p] = -sum over rows at p of grad[r]  -  b[p] * reg * count[p]
D2[p] = -sum over rows at p of hess[r]  -         reg * count[p]
b[p] += learning_rate * D1[p] / (|D2[p]| + 1e-3)
```

Every term is LightGBM's. The derivatives are of the *utility*, so they are
the negated gradients of the loss. The L2 regularization
(`lambdarank_position_bias_regularization`) is scaled by the number of rows
at the position, so a slot seen a thousand times is not shrunk as hard as
one seen twice. The `1e-3` floor keeps a position with no rows from dividing
by zero, and leaves its bias where it was.

Step 2 is the load-bearing design decision: **position bias is a score
offset, so it needs no new pair loop.** `PositionBiasState.adjust_scores`
produces the offset vector and `ranking._fill_lambdas` - mojotrees's one
LambdaRank kernel - consumes it. This is also how LightGBM factors it
(`RankingObjective::GetGradients` builds `score_adjusted` and calls the
unmodified `GetGradientsForOneQuery`).

### What a model file does not hold

The biases are **training state, not model state.** They exist to make the
trees unbiased; the trees are what is served. Three consequences:

- `serialize.save_model` writes an unbiased-LambdaRank ensemble today with
  no format change and no version bump, because it is an ordinary
  single-output `Booster` carrying the `LAMBDARANK` objective code. Nothing
  in the file says position bias was used.
- `predict` must never add a bias. A served query has no position column
  yet - that is what the model is being asked to produce - so adding one
  would be adding a correction for an event that has not happened.
- **Continued training on a positioned dataset must be refused, not
  resumed.** `train_more`-style continuation restarts from zero biases,
  which would fit the next round's trees against a correction the earlier
  rounds already applied. `train_ranker_advanced` returns the state in a
  `TrainedAdvancedRanker` so that a caller who wants to continue has the
  option of carrying it forward explicitly; a caller who throws it away has
  destroyed the only record of it. The refusal is a request on
  `boosting.mojo` / `basic.py` and is written out in the handoff.

### Positions themselves

`PositionMap` is the `Dataset.position` field: one dense id per row in
`[0, n_positions)`. `positions_from_codes` densifies arbitrary integer
codes in **order of first appearance**, which is deterministic in the row
order and needs no sort, and returns the code table alongside so a learned
bias can be reported against the slot a user actually named. Unlike query
ids, positions may repeat and interleave freely: a position is a label on a
row, not a run of rows.

## 3. Label gains, truncation, and the maxDCG cutoff

`LabelGain` is LightGBM's `label_gain`. `LabelGain.default()` is the shipped
one, `2^l - 1` for labels 0..30, built by calling `ranking.label_gain` so
the two tables cannot drift.

`check_label_gain` is stricter than LightGBM in two ways, both deliberate:

- **nondecreasing.** The pair loop picks which document to push up by
  *label*, not by gain. A gain that falls as the label rises makes `dcg_gap`
  negative, which flips the sign of every lambda in the pair - the objective
  inverts and nothing raises. LightGBM only checks that the vector is long
  enough.
- **`gains[0] == 0`.** An irrelevant document contributing nothing is the
  definition DCG is built on. A nonzero entry would add a constant to every
  query's DCG that does not cancel in the NDCG ratio.

`AdvancedRankParams.max_dcg_cutoff` decouples the cutoff the lambdas are
normalized against from `lambdarank_truncation_level`. LightGBM ties them,
and the default of `0` keeps them tied, so this is an extension that is off
unless it is asked for.

`pair_budget(groups, truncation_level)` reports how many pairs the truncated
loop would visit ignoring labels, and `GroupAudit.n_pairs` reports how many
it will actually visit. The gap between the two is what label ties remove.

### The routing rule

`AdvancedRankParams.uses_baseline_lambdas()` decides which kernel runs:

| Configuration | Kernel |
| --- | --- |
| default gain, `max_dcg_cutoff = 0`, `pair_sampling_rate = 1.0` | `ranking._fill_lambdas` - this module contributes no arithmetic |
| any position bias, otherwise default | `ranking._fill_lambdas`, on offset scores |
| custom gain vector, or decoupled maxDCG cutoff, or pair sampling | `_fill_query_lambdas_general` |

`_fill_query_lambdas_general` is a generalization of
`ranking._fill_query_lambdas` over the gain table and the pair draw; every
other line is the same line. With the default table, a keep rate of 1, and
`dcg_cutoff = truncation_level` it must reproduce it bit for bit. That
equality is **unverified**: the routing rule means the case never occurs in
practice, which is what makes the second kernel safe to have but also what
makes a differential test necessary before either is exposed.

## 4. Deterministic pair sampling

An extension. LightGBM has nothing like it.

The pair loop is `O(truncation_level * cnt)` per query, so a query set with
long queries spends its round enumerating pairs. `pair_sampling_rate` in
`(0, 1]` keeps each admissible pair with that probability and rescales the
kept ones by `1 / rate`, so the expected lambda is unchanged.

The draw is counter based, like every other draw in mojotrees:

```
stream = splitmix64(splitmix64(splitmix64(seed ^ PAIR_DOMAIN) ^ iteration) ^ query)
keep(i, j) = uniform(splitmix64(stream ^ (i * GOLDEN)) + j) < rate
```

so the decision for a pair depends only on `(seed, iteration, query, i, j)`.
Nothing carries between pairs, queries, rounds, or threads, which is what
makes a round reproducible from its index alone and makes the draw
independent of bagging, ordering, and backend. The two ranks are mixed
rather than added so that pairs with the same index sum draw independently.
The default seed is `5`, LightGBM's `objective_seed`, which is the seed slot
this kind of randomness belongs in.

Two honest costs:

- the round is **not** bit-identical to an unsampled one, by construction
- with `lambdarank_norm` on, `sum_lambdas` accumulates the rescaled
  magnitudes, so the per-query `log2(1 + sum) / sum` factor sees an
  estimate of the sum rather than the sum

Tied labels are skipped **before** the draw, not after. Skipping after would
spend randomness on pairs that carry no information and would make which
pairs survive depend on the labels.

## 5. Evaluation: positions, weights, and aggregation metadata

`ndcg_eval` and `map_eval` return a `RankEval` rather than a number.

```
RankEval
  cutoffs[c], values[c]              the aggregate at each position in eval_at
  per_query[c * n_queries + q]       optional, cutoff major
  n_queries, total_weight, weighted
  n_degenerate                       queries scored by convention, not measured
  n_single_document                  queries that could not have scored anything else
```

`n_degenerate` is the number the mean hides. A query whose labels are all
zero has no attainable DCG and counts as `1.0`; a query with no relevant
document counts as `1.0` for MAP. Both conventions match LightGBM and both
are right - nothing was retrievable, so nothing was missed - but a reported
NDCG of 0.9 over a set that is a third degenerate is a different number from
0.9 over a set that is not, and no consumer of a bare mean can tell them
apart. `n_single_document` is the same problem in a milder form: a
one-document query is perfectly ranked at every cutoff, always.

`query_weights` is LightGBM's `Metadata::CalculateQueryWeights`: a query's
weight is the **mean** of its rows' weights, not their sum. The mean is
correct here and the choice is not arbitrary. A query's NDCG is already
normalized by its own maxDCG, so it is a number in `[0, 1]` that says
nothing about how many documents produced it; weighting by the sum would let
a 500-document query count a hundred times a 5-document one purely for being
longer, which is the thing per-query normalization exists to prevent.
Weighting is off by default (`weight_queries = False`), which is the
unweighted mean `ranking.ndcg_at_cutoffs` computes.

Both metrics score **raw, unadjusted scores**. A learned position bias is
not added, for the reason in section 2.

`ranking.ndcg_at_cutoffs` and `ranking.map_at_cutoffs` stay the authority
for the case they share with these - unweighted, default gains, no per-query
output - and the differential check that pins the agreement is UNRUN.

## 6. Degenerate queries, and what may be done about them

`audit_groups` counts what a query set is actually made of:

| Count | Meaning | Contributes to training | Contributes to the metric |
| --- | --- | --- | --- |
| `n_single_document` | one document | nothing (`cnt < 2`, no pairs) | `1.0`, always |
| `n_tied_label` | every label equal | nothing (every pair skipped) | a real number, unless also all-zero |
| `n_all_zero_label` | every label 0 | nothing (`inverse_max_dcg = 0`) | `1.0`, always |
| `n_pairable` / `n_pairs` | the queries and pairs that do work | everything | everything |

`GroupAudit.trains_on_nothing()` is the case worth catching early: a query
set where no query can produce a single lambda trains a model of zero trees
and reports a perfect NDCG.

Two operations, with different safety:

- **`sanitize_group_counts`** repairs a `group` array naming queries that
  hold no rows. Under `EMPTY_QUERY_DROP` this is safe in the one way that
  matters: an empty query owns no rows, so dropping it renumbers queries and
  leaves every row index, label, weight, and position exactly where it was.
  A negative count is always an error - that is a corrupt array, not an
  empty query, and the rows it claims are unaccounted for.
- **`prunable_queries`** names the queries that contribute exactly zero to
  the objective. Dropping them from **training** changes no gradient,
  because each of their rows already receives a lambda of exactly `0.0` -
  but it is not free, because `min_data_in_leaf` and every other count-based
  rule do see those rows, so pruning changes which splits are legal. Treat
  it as a cost decision, not an identity. Dropping them from **evaluation**
  is not safe under any reading: they are counted as `1.0`, so removing them
  changes the reported number. `RankEval.n_degenerate` reports them instead.

## 7. The three query-level contracts

### Bagging

`ranking._refresh_query_bag` already samples whole queries and is the one
implementation; `refresh_query_bag` here re-exports it under a name that
cannot be confused with `bagging.refresh_bag`, which samples **rows** and
would split a query.

`check_query_bagging` adds the check only ranking can make.
`bagging.sample_rows` guarantees a nonempty bag by falling back to the
single smallest draw, so `bagging_fraction = 0.001` over 20 queries does not
fail - it silently trains every round on one query. That is a configuration
mistake, not a random outcome, and it is refused rather than discovered from
a flat NDCG curve.

### Cross-validation

`query_folds` splits on **whole queries**. A query split across the two
sides of a fold is the ranking form of training on the test set: the
held-out half is normalized against a maxDCG computed from documents the
model was trained on.

Fold `f` holds queries `[f * Q // K, (f + 1) * Q // K)` of the query order -
the same contiguous chunking `python/mojotrees/cv.py::_chunk_folds` does, so
the Mojo and Python paths agree fold for fold when given the same order.
`shuffle=True` permutes the **query** indices, never the rows, with a
counter-based Fisher-Yates. Each `QueryFold` carries both sides' rows *and*
both sides' `group` arrays, because a row list without query boundaries is
not a ranking dataset.

### Distributed partitioning

This is the contract with the sharpest edge.

`distributed.partition_rows` cuts at `r * n_rows // W`, which lands inside a
query whenever the row count does not divide evenly. A rank holding half a
query computes lambdas normalized against the maxDCG of the half it can see.
The gradients are wrong on every rank that holds a fragment, and **no
all-reduce catches it**, because the ranking objective needs no cross-rank
reduction at all:

> Every lambda is computed inside one query. A query-aligned partition
> therefore makes the distributed gradient exactly the single-node gradient,
> rank by rank. That is the whole contract, and it holds only if no query is
> split.

`partition_queries` produces that partition: contiguous, order preserving,
and cut at the query boundary nearest to each even row split, with
boundaries forced nondecreasing so a run of large queries yields empty ranks
rather than a crossed partition. Concatenating the partitions in ascending
rank order reproduces the dataset row for row, the property
`distributed.partition_rows` documents and the floating point equivalence
argument rests on. `check_query_partition` re-derives all of it, so a rank
that was handed a bad partition fails identically to every other rank.

What still has to happen on the `distributed.mojo` side is one function -
a `partition_rows_at` that takes explicit boundaries instead of computing
them - plus the base-score and metric-reduction notes. Both are written out
as ready-to-apply patches in the handoff. Nothing here has run over a
network, and nothing here changes that.

## 8. What is connected, and what is not

Connected today, inside the owned file:

- `fit_ranker_advanced` → `sanitize_group_counts` → `fit_bins` →
  `train_ranker_advanced` → `advanced_lambdarank_gradients` →
  `ranking._fill_lambdas` (or the generalized kernel) →
  `update_position_bias` → `tree.grow_tree` → `boosting.Booster`
- `train_ranker_advanced` → `ranking._refresh_query_bag`, and
  `check_query_bagging` before the first round
- `ndcg_eval` / `map_eval` → `ranking._argsort_desc_range`,
  `ranking._sorted_gains`, `ranking._discounts`, `ranking.max_dcg`
- `query_folds` → `rows_of_queries` → `ranking._expand_queries`
- every entry point validates through `check_advanced_rank_params`, which
  calls `ranking.check_ranker_params`, so there is one set of rules

**This list was a ten-item handoff and four of its items have since
landed. Rechecked item by item on 2026-08-18.** It used to open "Each is a
ready-to-apply patch in `handoffs/remaining_05_ranking.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_05_ranking.md)`", and that file
does not exist any more (`ls handoffs/` lists `INDEX.md`,
`connect_22_audit.md`, `consolidation_round.md`,
`migration_20_device_policy.md`, `performance_17_thermal_energy.md`,
`remaining_14_validation_plan.md`), so the pointer is dropped rather than
repeated.

Landed:

2. `trainset.mojo` - **done.** `train_dataset_ranker_advanced` is at
   `src/mojotrees/trainset.mojo:1873` and takes a `PositionMap`;
   `basic.Dataset` carries `_position` (`python/mojotrees/basic.py:513`) and
   `_Config.binding_params` passes it through `_position_params`.
7. `bindings/_mojotrees.mojo` - **done.** `_parse_positions` reads
   `position_addr`, and all three ranker entry points parse the advanced
   rank params.
8. The Python surface - **done**, though at
   `python/mojotrees/sklearn.py:7013` rather than in `__init__.py`:
   `MojoTreesRanker.fit(position=...)`.
10. `docs/LIGHTGBM_PARITY.md` - **done**, on the evidence that now exists:
   `Dataset.position` is `supported` and
   `lambdarank_position_bias_regularization` is `partial`.

Still open, and verified still open rather than assumed:

1. `ranking.mojo` - fold `train_ranker_advanced` back into `train_ranker`.
   `ranking.mojo` still defines only `train_ranker` and
   `train_ranker_with_valid`; the advanced loop is a separate trainer.
3. `boosting.mojo` / `basic.py` - refuse continuation of a positioned
   ranker. No such refusal exists; the learned biases are training state
   that no model file holds, so a continued fit restarts them silently.
4. `distributed.mojo` - `partition_rows_at`. The symbol exists nowhere in
   `src/`; only `ranking_advanced.mojo:122` mentions the request.
5. `params.mojo` - partly. The names are in the parameter table
   (`src/mojotrees/params.mojo:182` carries `label_gain` and `eval_at`), so
   this item is narrower than it reads.
6. `objective_registry.mojo` - `eval_at` as a `NEEDS_CUTOFF` list.
   `NEEDS_CUTOFF` is a flag (`:433`, set at `:1834`), not a list.
9. `python/mojotrees/cv.py` - point the ranking folds at `query_folds`.
   `cv.py:386` still chunks queries with its own `_chunk_folds`.
