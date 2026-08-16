# ONNX export: the expressibility boundary

Companion to catalog entry A29 in [CATBOOST_CATALOG.md](CATBOOST_CATALOG.md).
That entry says what must serialize. This one says what can leave the package
as an `ai.onnx.ml` graph, what cannot, and why the second list is refused
rather than approximated.

Written 2026-08-16 against `bfd6187`. **No timings, no benchmarks.** Nothing
here is a speed claim.

## 0. The rule

An export that drops a fitted fact produces a model that loads in any runtime,
validates against the ONNX checker, and scores differently. Nothing raises.
That is the worst failure mode available to this lane, so the exporter's
contract is: **either the graph reproduces the raw score, or the export
raises and names what stopped it.** There is no lossy mode and no warning
mode.

## 1. The operator, verified

Fetched from the ONNX operator reference on 2026-08-16.

| | `ai.onnx.ml.TreeEnsembleRegressor` | `ai.onnx.ml.TreeEnsemble` |
|---|---|---|
| Opsets | ml 1, ml 3; **deprecated at ml 5** | introduced at **ml 5** |
| Split modes | `BRANCH_LEQ`, `BRANCH_LT`, `BRANCH_GTE`, `BRANCH_GT`, `BRANCH_EQ`, `BRANCH_NEQ`, `LEAF` (strings) | the same six, plus **`BRANCH_MEMBER`** (uint8 tensor) |
| Set membership | **no** | yes, `membership_values`, NaN-delimited per `BRANCH_MEMBER` node |
| Double thresholds | yes, `nodes_values_as_tensor` (ml 3) | yes, `nodes_splits` is a float64/float32/float16 tensor |
| Double leaf weights | yes, `target_weights_as_tensor` (ml 3) | yes, `leaf_weights` |
| Double base values | yes, `base_values_as_tensor` (ml 3) | n/a, folded differently |
| NaN routing | `nodes_missing_value_tracks_true` (ml 3) | `nodes_missing_value_tracks_true` |
| Aggregation | `AVERAGE`, `SUM`, `MIN`, `MAX` | same, default `SUM` |
| `post_transform` | `NONE`, `SOFTMAX`, `LOGISTIC`, `SOFTMAX_ZERO`, `PROBIT` | same |
| Output dtype | **`tensor(float)` only, in all three versions** | follows the input type constraint |

The choice made here is **ml opset 3 `TreeEnsembleRegressor`**, for both
single-output and multiclass, because it is the version every deployed runtime
reads today. Opset 5 is what unlocks categorical set splits, and section 5
says what it would take.

## 2. What converts exactly

### 2.1 The threshold

mojotrees splits on bin ids: node `i` sends a row left when
`bin(x_f) <= threshold_bin[i]` (`tree.mojo:483`). `BinMapper.bin_value`
(`binning.mojo:2365`) defines `bin(x)` as the first `b` with `x <= edges[b]`,
and `edges` is strictly increasing within a feature. So

```
bin(x) <= t   <=>   x <= edges[edge_offsets[f] + t]
```

in both directions, for every finite `x`, with no tie case: forward because
`x <= edges[bin(x)] <= edges[t]`, backward because `x <= edges[t]` makes `t`
one of the candidates the search minimizes over. **This is an equality, not an
approximation**, and it holds in `Float64`, which `nodes_values_as_tensor`
carries without narrowing.

`t` equal to the feature's edge count is the degenerate "everything
non-missing goes left" split, which exports as a threshold of `+inf`. `t`
greater than that is refused as corrupt.

### 2.2 NaN

Two cases, and the second is the one an exporter gets wrong.

- `mapper.missing_bin[f] >= 0`: NaN bins to the reserved bin, and node `i`
  routes it by `default_left[i]` whatever the threshold says.
- `mapper.missing_bin[f] == -1`: **NaN is binned as the value `0.0`**
  (`binning.mojo:2373`, LightGBM's `missing_type=None`) and then takes the
  ordinary threshold test.

In both cases the destination is a *per-node constant* the exporter can
compute: bin the NaN, ask `Tree.goes_left`, and set
`nodes_missing_value_tracks_true[i]` accordingly, with `truenodeids` bound to
the left child. So `use_missing=False` models export exactly too, which is not
obvious and is worth the two lines it costs.

### 2.3 The leaf value

`Booster.predict_raw_bins` is `base_score + sum_t learning_rate *
tree_t.value[leaf]` (`boosting.mojo:1570-1574`). **`Tree.value` does not
include the learning rate.** The exporter multiplies once, in `Float64`, which
is the same product the predictor forms, and writes it to
`target_weights_as_tensor`. `base_score` goes to `base_values_as_tensor`.

An exporter that skips the multiply produces a graph that is wrong by a factor
of `learning_rate` on every row, and there is no check anywhere in the ONNX
ecosystem that would catch it.

### 2.4 Multiclass

`MulticlassBooster` holds trees round-major, `trees[i * n_classes + k]`
(`boosting.mojo:2682`), so tree index `j` belongs to target `j % n_classes`.
That maps directly onto `target_ids` with `n_targets = n_classes`, and
`base_values` takes `base_scores`. Exact, one `TreeEnsembleRegressor`.

### 2.5 The response transform

`Booster.response` (`boosting.mojo:1530`) applies the link for
`objective_link(objective)`.

| Link | Objectives | ONNX |
|---|---|---|
| `LINK_IDENTITY` | squared error, huber, quantile, L1, MAPE, fair, lambdarank, custom | `post_transform=NONE` |
| `LINK_SIGMOID` | binary logistic, cross entropy | `post_transform=LOGISTIC` |
| `LINK_EXP` | poisson, gamma, tweedie | **no `post_transform` exists**; an explicit `Exp` node is appended |
| `LINK_SOFTMAX` | multiclass | `post_transform=SOFTMAX` |

## 3. What does NOT convert, and is refused

Each of these is a condition the exporter tests and raises on, naming the
model feature and the reason.

**R1. Categorical features of any kind.** Three independent problems, any one
of which is fatal:

1. `ai.onnx.ml` opset 3 has no set-membership mode. A mojotrees categorical
   node routes on a 256-bit set over bins (`tree.mojo:290-295`), which
   `BRANCH_EQ` cannot express for a set of size greater than one.
2. `CategoricalSpec.bin_of` **truncates toward zero** (`categorical.mojo:293`,
   matching LightGBM's `static_cast<int>`), so a raw input of `3.7` is
   category `3`. ONNX compares the float input directly. Any exact export
   needs a `Cast` to int64 in front of every categorical column, and ONNX
   leaves float-to-int casts undefined for NaN and for out-of-range values --
   which is exactly the input a categorical column gets when a value is
   missing.
3. Values below zero, NaN, values at or above `1 << 31`, and categories the
   fit did not keep all collapse to `UNKNOWN_BIN = 0`, which is never in a set
   and so always goes right (`categorical.mojo:290`). Reproducing that
   collapse needs a range guard, not a comparison.

The blast radius is not limited to the split. A model with *any* categorical
feature routes its numerical splits through the same mapper, so the refusal
is on `mapper.cats.any_categorical()`, not on the presence of a categorical
node. A model that declares a categorical feature no tree happened to split
on still exports wrong if it exports at all, because the column's binning is
not a threshold.

**R2. Linear leaves.** `LinearEnsemble` (format v5) evaluates
`intercept + sum coef * (x - center)` at the leaf (`linear_tree.mojo:537`).
`TreeEnsembleRegressor` leaves are constants. There is no ONNX tree operator
with affine leaves, and expressing it would mean emitting the tree as a leaf
*indicator* and a `MatMul`, which is a different graph with different
numerics. Refused.

**R3. The `CUSTOM` objective**, unless the caller asks for the raw score. The
link is a caller-supplied callable that the model does not hold
(`objective_registry.mojo:250`), so the graph cannot apply it. `raw_score=True`
exports the ensemble with `post_transform=NONE`, which is exactly what
`Booster.response` does for `CUSTOM` anyway, so that mode is exact.

**R4. CTR features** (catalog A5, unbuilt). A CTR replaces a category with a
statistic fitted from `y`. A single-feature CTR is a lookup and could be an
`ai.onnx.ml.LabelEncoder` feeding a `Gather`; CatBoost's *combinations* are
crosses of several categoricals resolved through a hash, and no ONNX operator
reproduces a specific hash function. The refusal must be written before the
feature lands, because the failure is silent: an exported graph that skips the
CTR feeds the tree a raw category id where it expects a target statistic.

**R5. Text and embedding features** (unbuilt). `ai.onnx` has `StringNormalizer`
and `TfIdfVectorizer`, and CatBoost's text estimators are neither: BM25 and
the naive-Bayes calcer are fitted from `y` in an ordered scheme. Refused.

**R6. Vector leaf values** (unbuilt). `TreeEnsembleRegressor` does support
`n_targets > 1` through `target_ids`, so a *future* vector-leaf model is
expressible; what is refused today is guessing at a layout that does not
exist yet.

**R7. Position bias** (`ranking_advanced.mojo:469`). Fitted per-position
offsets that the model file does not carry at all. A ranking model trained
with position bias cannot be exported correctly by anything, ONNX included,
until that state is in the file. Refused on the same principle even though
today the exporter cannot detect it -- there is nothing in the `Model` to
test. **This is the one refusal the exporter cannot enforce**, and it is
recorded here so it is not mistaken for coverage.

## 4. What the exactness claim actually is

Stated narrowly on purpose.

**Claimed.** For an accepted model, every row reaches the same leaf of every
tree in the exported graph as in `Model.predict_raw`, and each tree's
contribution is the same `Float64` value. Leaf selection is exact; the
per-tree contributions are exact.

**Not claimed.** The final score. Three reasons, and the first one is about
mojotrees rather than about ONNX.

0. **mojotrees's own predictor fuses the learning-rate multiply into the
   accumulation.** `Booster.predict_raw_bins` is
   `s += learning_rate * tree.value[leaf]`, which the compiler contracts into
   a fused multiply-add: one rounding, not two. The exported graph carries
   `learning_rate * value` as a formed product in `target_weights`, so
   summing it is a plain add. The two are not the same arithmetic, and
   **no ONNX export of this model can be bit-identical to `predict_raw`**,
   however correct it is. Measured while writing `tests/test_onnx_export.mojo`:
   a one-ULP difference on a 20-tree squared-error fit.

   The cause is pinned rather than assumed.
   `test_plan_is_bit_exact_when_the_learning_rate_is_one` fits the same data
   at `learning_rate = 1.0`, where the product is exact and fusing therefore
   cannot change anything, and the plan and the model agree bit for bit on
   every row. If leaf selection or summation order were at fault, that test
   would fail too. It passes.

Then the two ONNX-side reasons:

1. `TreeEnsembleRegressor`'s output is `tensor(float)` in every version of the
   operator, so the sum is delivered in `Float32` no matter what the
   attributes carry.
2. ONNX does not specify the summation order of `aggregate_function=SUM`. A
   runtime is free to add the trees in any order or in a tree, and floating
   point addition is not associative.

So the honest contract is "exact leaves, exact per-tree values, a fused
multiply on the mojotrees side that the graph does not have, `Float32`
delivery, and unspecified summation order". A differential test against a
runtime should compare within a stated tolerance, and any tolerance failure
is a routing bug rather than a rounding one -- which is precisely what makes
the narrow claim more useful than a broad one.

Response transforms carry no bit-level claim at all: the runtime applies them
in its own precision. `raw_score=True` is the mode with the contract.

## 5. What opset 5 would buy, and what it would not

`ai.onnx.ml.TreeEnsemble` at opset 5 has `BRANCH_MEMBER` with
`membership_values`, which expresses a category set directly. That removes
problem 1 of R1. It does **not** remove problems 2 and 3: truncation toward
zero and the `UNKNOWN_BIN` collapse are mojotrees's input semantics, not the
operator's, and they still need a guard in front of the graph. `TreeEnsemble`
also returns the input's dtype rather than `tensor(float)`, which would turn
the `Float32` half of section 4 into a real `Float64` claim.

Taking it is a later lane. It is written down here so that the categorical
refusal is understood as "not yet, and here is the exact route", not as
"impossible".

## 6. What this lane built

- `src/mojotrees/onnx_export.mojo`. All of the arithmetic above: the refusal
  predicate, the bin-to-threshold conversion, the NaN destination, the
  learning-rate multiply, the multiclass target mapping, and a
  structure-of-arrays plan emitted as the package's own text token stream.
  This is where every correctness-relevant decision lives, and it is testable
  with no dependencies.
- `python/mojotrees/onnx_export.py`. The mechanical half: read the plan, build
  a `ModelProto` with `onnx.helper`. `onnx` is an optional dependency and its
  absence raises with the install line rather than degrading.

The split is deliberate. The part that can be wrong without anyone noticing is
in Mojo, under test. The part that needs a third-party library is a
transcription with no arithmetic in it.
