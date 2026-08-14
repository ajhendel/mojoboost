# Task 21 handoff, row and feature sampling

Lane files changed
- `src/mojoboost/sampling.mojo`
- `tests/parallel/test_sampling.mojo` (new)
- `handoffs/task21_sampling.md` (this file)

Nothing central was edited. Every change below is written out for the
integration session to apply.

## 1. What this lane found before writing anything

The lane brief asks for bagging, GOSS, and feature fraction. Two of the three
already exist in the tree and are wired into every trainer.

| Asked for | State found | Owner |
| --- | --- | --- |
| `bagging_fraction`, `bagging_freq`, `bagging_seed` | implemented, integrated, tested | `src/mojoboost/bagging.mojo`, `tests/test_bagging.mojo` |
| GOSS `top_rate`/`other_rate`, amplification, warmup | implemented, integrated, tested | `src/mojoboost/goss.mojo`, `tests/test_goss.mojo` |
| `feature_fraction`, `feature_fraction_bynode` | implemented, integrated, tested | `src/mojoboost/sampling.mojo`, `tests/test_feature_sampling.mojo` |
| `colsample_bytree` / `colsample_bynode` aliases | missing, recorded as gap 3 of `docs/LIGHTGBM_PARITY.md` | this lane |
| per-level feature fraction | missing | this lane |
| `pos_bagging_fraction` / `neg_bagging_fraction` | missing, `deferred` at `docs/LIGHTGBM_PARITY.md` line 290 | this lane |
| `data_sample_strategy` value resolution | `partial` at `docs/LIGHTGBM_PARITY.md` line 271 | this lane |
| sampled row set to GPU active-row ranges | missing | this lane |

Re-implementing bagging and GOSS inside `sampling.mojo` would have produced a
second source of truth for semantics that `boosting.mojo` and `train_gpu.mojo`
already call, so this lane did not do it. `sampling.mojo` implements only the
pieces above that were missing, and this handoff maps the existing modules for
the parts that were not. That is a deliberate scope call and the reviewer
should confirm it rather than assume the brief was executed literally.

`sampling.mojo` still imports nothing from `goss.mojo` and takes only the
`DEFAULT_BAGGING_SEED` constant from `bagging.mojo`, so the three samplers stay
independent, as the existing comment in `goss.mojo` intends.

## 2. New public API in `src/mojoboost/sampling.mojo`

Per-level feature fraction

- `comptime DEFAULT_FEATURE_FRACTION_BYLEVEL = 1.0`
- `select_level_features(tree_features, fraction, seed, tree_index, depth) raises -> List[Int]`
- `select_split_features(tree_features, fraction_bylevel, fraction_bynode, seed, tree_index, depth, node) raises -> List[Int]`

Parameter names

- `sampling_param_names() -> List[String]`
- `canonical_sampling_param(name) raises -> String`
- `is_sampling_param(name) -> Bool`
- `canonical_data_sample_strategy(value) raises -> String`

Class-conditional row bagging

- `comptime DEFAULT_POS_BAGGING_FRACTION = 1.0`, `comptime DEFAULT_NEG_BAGGING_FRACTION = 1.0`
- `struct ClassBaggingParams(Copyable, Movable)` with fields `pos_fraction`, `neg_fraction`, `freq`, `seed`, plus `disabled()`, `enabled()`, `validate()`
- `has_positive_rows(labels) -> Bool`, the label half of LightGBM's enable condition
- `sample_rows_by_class(params, labels, bag_index, mut rows) raises`
- `refresh_class_bag(mut bag, params, labels, iteration) raises`

Row sets, masks, ranges, scales

- `check_row_set(rows, n_rows) raises`
- `contiguous_ranges(rows, n_rows) raises -> List[Int]`
- `ranges_row_count(ranges) raises -> Int`
- `row_mask(rows, n_rows) raises -> List[Bool]`
- `expand_row_scale(rows, scale, n_rows) raises -> List[Float64]`

## 3. LightGBM and XGBoost names, aliases, defaults

`canonical_sampling_param` is the table. Defaults are LightGBM 4.7 defaults
unless the row says otherwise.

| mojoboost name | Accepted spellings | Default | Notes |
| --- | --- | --- | --- |
| `bagging_fraction` | `sub_row`, `subsample`, `bagging` | 1.0 | already implemented in `bagging.mojo` |
| `bagging_freq` | `subsample_freq` | 0 | already implemented |
| `bagging_seed` | `bagging_fraction_seed` | 3 | already implemented |
| `pos_bagging_fraction` | `pos_sub_row`, `pos_subsample`, `pos_bagging` | 1.0 | new, `ClassBaggingParams.pos_fraction` |
| `neg_bagging_fraction` | `neg_sub_row`, `neg_subsample`, `neg_bagging` | 1.0 | new, `ClassBaggingParams.neg_fraction` |
| `feature_fraction` | `sub_feature`, `colsample_bytree` | 1.0 | alias is the parity gap this closes |
| `feature_fraction_bynode` | `sub_feature_bynode`, `colsample_bynode` | 1.0 | alias is the parity gap this closes |
| `feature_fraction_bylevel` | `colsample_bylevel` | 1.0 | XGBoost only, no LightGBM equivalent, extension not parity |
| `feature_fraction_seed` | none | 2 | already implemented |
| `top_rate` | none | 0.2 | already implemented in `goss.mojo` |
| `other_rate` | none | 0.1 | already implemented in `goss.mojo` |
| `data_sample_strategy` | none | `bagging` | values `bagging` and `goss` |

Verification. Every row of this table was diffed against a LightGBM checkout
found on this machine at `/private/tmp/lightgbm-cloc.y8ASwY/LightGBM`,
`VERSION.txt` 4.7.0.99, which is master after the 4.7.0 release rather than
the 4.7.0 tag the parity doc audits against. The sources read were the alias
map at `src/io/config_auto.cpp` lines 72 to 86 and 826 to 852 and the defaults
in `include/LightGBM/config.h`. The table matched except for two aliases this
lane had missed, `pos_bagging` and `neg_bagging`, which are now accepted and
tested. No spelling in the table is absent from LightGBM, and `bagging_freq`
correctly takes only `subsample_freq`. Two cautions. That checkout sits in a
temporary directory this lane did not create, and 4.7.0.99 is not the 4.7.0
tag, so a spelling added on master after 4.7.0 would look accepted here.
Neither of the two additions is new in that window, but the diff is worth
repeating against the tag if the parity doc's version claim has to be exact.

## 4. Central integration, exact edits

### 4.1 `src/mojoboost/tree.mojo`

Import, at the existing import block near line 70.

```mojo
from .sampling import (
    DEFAULT_FEATURE_FRACTION_SEED,
    check_feature_fractions,
    select_node_features,
    select_split_features,
    select_tree_features,
)
```

`TreeParams` gains one field. Append the constructor argument at the END of the
argument list, after `cat`, so no existing positional call site shifts.

```mojo
    var feature_fraction_bylevel: Float64
```

```mojo
        var cat: CategoricalParams = CategoricalParams.default(),
        feature_fraction_bylevel: Float64 = DEFAULT_FEATURE_FRACTION_BYLEVEL,
    ):
        ...
        self.feature_fraction_bylevel = feature_fraction_bylevel
```

Three call sites replace `select_node_features` with `select_split_features`.
Both the node id and the depth are already in scope at each one.

| Site | Line before this lane | Node argument | Depth argument |
| --- | --- | --- | --- |
| root node in `grow_tree` | 754 | `0` | `0` |
| left child in `grow_tree` | 886 | `left_node` | `child_depth` |
| right child in `grow_tree` | 904 | `right_node` | `child_depth` |

Root, replacing the call that begins at line 754.

```mojo
        select_split_features(
            tree_features,
            params.feature_fraction_bylevel,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            0,
            0,
        ),
```

Left child, replacing the call at line 886 (right child identical with
`right_node`).

```mojo
            select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                left_node,
            ),
```

`TreeParams.default()` near line 138 must set the new field to 1.0, and the
docstring near line 81 should name it. With the field at 1.0 the composed call
returns exactly what `select_node_features` returned, which is asserted by
`test_bylevel_one_leaves_the_node_draw_untouched`, so this edit alone must not
move a single existing model. Confirm that with `pixi run test` at integration
time, not before.

Also check whether the sparse grower `src/mojoboost/tree_sparse.mojo` has its
own `select_node_features` call. It is unreachable today (gap 1 of the parity
doc), but it should not drift.

### 4.2 Validation, `src/mojoboost/tree.mojo` or wherever `check_feature_fractions` is called

`check_feature_fractions` currently validates two fractions. Either add a third
call to `check_feature_fraction(params.feature_fraction_bylevel,
"feature_fraction_bylevel")` at the same place, or extend
`check_feature_fractions` to take three arguments. `select_level_features`
validates its own fraction anyway, so this is about failing before growth
starts rather than mid-tree.

### 4.3 Class-conditional bagging, `src/mojoboost/boosting.mojo`

`ClassBaggingParams` replaces the uniform bag when it is enabled. It is a
binary-classification parameter in LightGBM and should be rejected elsewhere.

Import.

```mojo
from .sampling import (
    ClassBaggingParams,
    has_positive_rows,
    refresh_class_bag,
)
```

Trainer signature, alongside the existing `bagging` and `goss` arguments at
lines 853, 914, 988, 1089, and 1378.

```mojo
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
```

Validation, next to `check_bagging(bagging)` and `_check_goss(goss, bagging)`.

```mojo
    class_bagging.validate()
    if class_bagging.enabled():
        if bagging_enabled(bagging) or goss.enabled:
            raise Error("pos/neg bagging cannot combine with bagging or goss")
        if objective != BINARY_LOGISTIC:
            raise Error("pos/neg bagging applies to binary classification only")
```

Enable gate, hoisted above the round loop. LightGBM requires a positive row to
exist before balanced bagging applies, and falls back to plain
`bagging_fraction` when none does, so the gate is both halves and the label
pass runs once.

```mojo
    var balanced = class_bagging.enabled() and has_positive_rows(labels)
```

Round loop, at each `refresh_bag(bag, bagging, n, round)` call (lines 881,
1135, 1407, and the two in `train_gpu.mojo` at 423 and its sibling).

```mojo
        if balanced:
            refresh_class_bag(bag, class_bagging, labels, round)
        else:
            refresh_bag(bag, bagging, n, round)
```

`labels` is the raw target vector the trainer already holds. Rows are positive
when the label is above zero, so 0/1 and -1/+1 encodings both work, and the
class rates apply to the raw labels rather than to any transformed score. The
else branch is not dead under an all-negative dataset, it is LightGBM's own
fallback.

The degenerate-tree guard that checks `bagging_enabled(bagging) or
goss.enabled` at lines 898, 1148, and 443 must also accept `balanced`, or a
round that produces a stump under class bagging will be read as convergence.

Semantics confirmed against LightGBM's `BalancedBaggingHelper` in
`src/boosting/bagging.hpp`. One draw per row whichever class it belongs to,
compared against that class's fraction, `label > 0` as the class test, and the
enable condition `(pos < 1.0 || neg < 1.0) && num_pos_data > 0 && freq > 0`.
mojoboost reproduces all of it apart from the RNG substitution and the
never-empty guard already documented for uniform bagging.

Nothing else changes. A class bag is an ascending row-index list exactly like a
uniform bag, so tree growth, `min_data_in_leaf`, leaf values, score updates,
and the base score all behave as they do under `bagging.mojo` today.

### 4.4 `bindings/_mojoboost.mojo`

Follow `_parse_bagging` at line 327.

```mojo
def _parse_class_bagging(params: PythonObject) raises -> ClassBaggingParams:
    return ClassBaggingParams(
        Float64(py=params["pos_bagging_fraction"]),
        Float64(py=params["neg_bagging_fraction"]),
        Int(py=params["bagging_freq"]),
        Int(py=params["bagging_seed"]),
    )
```

Pass it at the trainer call sites near line 414, next to `_parse_bagging(params)`
and `_parse_goss(params)`. Add `feature_fraction_bylevel` to the `TreeParams`
construction at line 304.

```mojo
        feature_fraction_bylevel=Float64(
            py=params["feature_fraction_bylevel"]
        ),
```

Every key read here must be present in the Python params dict, so 4.5 and this
section land together.

### 4.5 `python/mojoboost/__init__.py`

Constructor keywords, near line 754.

```python
        feature_fraction_bylevel=1.0,
        colsample_bylevel=None,
        colsample_bytree=None,
        colsample_bynode=None,
        pos_bagging_fraction=1.0,
        neg_bagging_fraction=1.0,
```

Resolution, in the block near line 1164 that already uses `_resolve_alias`.

```python
        feature_fraction = self._resolve_alias(
            "feature_fraction", "colsample_bytree", 1.0
        )
        feature_fraction_bynode = self._resolve_alias(
            "feature_fraction_bynode", "colsample_bynode", 1.0
        )
        feature_fraction_bylevel = self._resolve_alias(
            "feature_fraction_bylevel", "colsample_bylevel", 1.0
        )
```

`_resolve_alias` raises when both spellings are set to different values, which
is the behavior wanted here, so `colsample_bytree` and `feature_fraction` can
never disagree silently. Validate all three fractions in (0, 1] and both class
fractions in (0, 1] with the same `ValueError` shape the file already uses for
`bagging_fraction`, then put the six new keys into the params dict the bindings
read.

Do not add `subsample`-style aliases twice. `bagging_fraction`/`subsample` and
`bagging_freq`/`subsample_freq` are already resolved at lines 1172 and 1175.

The docstring at lines 651 and 679 lists the accepted spellings and is the
user-visible record of gap 3, so it needs the three `colsample_*` names and the
two `pos/neg_bagging_fraction` names.

### 4.6 `src/mojoboost/__init__.mojo`

Export the new symbols so `tools/check_parity.py` can see them and so Mojo
users reach them without importing a submodule.

```mojo
from .sampling import (
    ClassBaggingParams,
    canonical_sampling_param,
    contiguous_ranges,
    expand_row_scale,
    refresh_class_bag,
    row_mask,
    sample_rows_by_class,
    select_level_features,
    select_split_features,
)
```

### 4.7 `docs/LIGHTGBM_PARITY.md`

Status changes are the parity owner's call, not this lane's. What the integration
session should propose once the code is wired and the full suite is green.

- Known gap 3 becomes closed, since `colsample_bytree` and `colsample_bynode`
  are then accepted.
- `pos_bagging_fraction` / `neg_bagging_fraction` at line 290 moves off
  `deferred`.
- `data_sample_strategy` at line 271 keeps its `partial` status until the
  parameter itself is accepted by the Python layer, not only resolvable.
- A new row for `feature_fraction_bylevel` belongs in the extensions section
  rather than the LightGBM matrix, because LightGBM has no such parameter.
- `tools/check_parity.py` keeps an exception list that must match the Known
  gaps section exactly, so removing gap 3 means editing the script too.

### 4.8 CLI and C API

`cli/` and `capi/` parse parameter names of their own. Route them through
`canonical_sampling_param` so the alias table has one home. That is a cleanup,
not a blocker, and it is the reason `canonical_sampling_param` raises on an
unknown name instead of returning the input unchanged.

## 5. How sampled row sets compose with Apple Task A1 active-row ranges

Task A1 landed `src/mojoboost/gpu_active_rows.mojo` while this lane was
running. The notes below were checked against that file as it stands and not
compiled against it, so treat them as a reading of A1's contract rather than a
verified integration.

A1 keeps one device-resident permutation of the tree's rows in which every live
leaf owns a contiguous half-open `LeafRange`, seeded by
`GpuActiveRows.begin_tree(bag)`, and splits a leaf by stable in-place partition
of its own range.

1. A sampled row set is an ascending, duplicate-free `List[Int]`, and an empty
   list means every row. Uniform bagging, class bagging, and GOSS all produce
   that shape, and `begin_tree` already treats an empty list as the identity
   permutation, so sampled and unsampled training share one path.
2. A bag reaches the GPU through `begin_tree(bag)` directly. No conversion is
   needed, and a class bag from `refresh_class_bag` substitutes for a uniform
   bag with no GPU change at all, because both are the same ascending row list
   with the same meaning. `check_row_set(bag, n_rows)` is the stronger
   precondition A1's own bounds check does not make, since A1 accepts any order
   by design.
3. `contiguous_ranges` and `row_mask` are host-side forms, not A1 inputs. Their
   use is checking a sampled set against a `LeafRange` table, where
   `ranges_row_count(contiguous_ranges(bag, n_rows))` must equal
   `LeafRangeTable.total_active()` at the root and a leaf's rows must fall
   inside exactly one range of a bag that happens to be contiguous. They are
   also what the pre-A1 leaf-id filtering path in `histogram_gpu.mojo` would
   need if it is kept as a fallback.
4. Ordering composes without further work. A1's partition is stable in buffer
   order, so a root seeded from an ascending sample stays ascending inside every
   descendant range, and its own docstring records that a non-ascending root
   would still track the CPU grower because both are seeded from the same list
   in the same order. Every sampler in this lane emits ascending sets, so the
   stronger property holds.
5. GOSS amplification reaches the GPU today by a different route than
   `expand_row_scale`. `goss_round` scales the host gradient and hessian arrays
   in place before they are uploaded, so the multiplier is already inside the
   values A1's histogram kernels quantize. `expand_row_scale` is for the case
   that does not hold, a device-side objective that computes gradients on the
   GPU and never round-trips them, where the sampler cannot reach into the host
   arrays. There the dense buffer carries the multiplier on sampled
   small-gradient rows, `1.0` on kept rows, and `0.0` off the sample, so a
   kernel that multiplies by it over all rows builds exactly the sampled
   histogram whether or not it also honors the active ranges. Coordinate this
   with the `gpu_objectives_native` lane before wiring it, since only that lane
   knows whether gradients stay on device.
6. Determinism is preserved across the CPU and GPU split, because none of these
   samplers depends on iteration order. Every draw is `splitmix64` keyed by an
   explicit counter, so a row's keep decision depends only on the seed, the bag
   or round index, and the row index, never on how many rows were drawn before
   it or on thread or block layout. The two trainers therefore sample identical
   rows without exchanging state.

Counters worth adding to the GPU benchmark harness, not measured here. Rows
active per node, ranges per node, and the fraction of nodes whose active set is
a single contiguous range. The last one predicts how much index indirection the
compaction actually removes.

## 6. Determinism contract for the new draws

- The per-level stream is `splitmix64(tree_stream ^ LEVEL_DOMAIN + depth *
  GOLDEN)` where `tree_stream` is the existing per-tree stream. Level sets
  therefore move with the tree that owns them and cannot collide with the
  per-node tag space, which is `node id + 1`.
- The class bagging stream is byte for byte the stream `bagging.mojo` uses, so
  equal positive and negative fractions reproduce a uniform bag row for row.
  That equivalence is asserted in the focused test against
  `bagging.sample_rows` itself, not against a copy of it.
- Two intentional differences from LightGBM carry over from `bagging.mojo`.
  splitmix64 gives 53 bits where LightGBM's per-block LCG gives 15, so bags do
  not match LightGBM row for row at equal seeds, and a class that draws nothing
  contributes its smallest-draw row so a bag can never lose a class entirely.
  The second guard can only fire where LightGBM would have trained on a
  single-class bag.

## 7. Test evidence

Command actually run, once, under the machine-wide build lock.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_sampling.mojo
```

Result. 21 tests run, 21 passed, 0 failed, 0 skipped. The suite was run twice,
once at 20 tests and again after the LightGBM source diff in section 3 added
two aliases, the `has_positive_rows` gate, and the test that covers them. The
21-test run is the one reported here.

What the 21 cover. Per-level ordering, subset containment, determinism across
depth, tree, and seed, and the drop-in equality of `select_split_features` with
`select_node_features` at a bylevel fraction of 1.0. Fraction accounting across
all three stages, including that the counts round stage by stage rather than
once on the product, 100 to 50 to 30 to 15. The alias table, canonical names
resolving to themselves, that `bagging`, `pos_bagging`, and `neg_bagging` stay
three separate parameters, and rejection of unknown names. Class bagging
reproducing a uniform bag at equal rates, holding each class near its own rate,
staying ascending and deterministic, never emptying a present class, never
inventing an absent one, following the `bagging_freq` schedule, and the
positive-row gate that decides whether it applies at all. Row-set validation,
range and mask agreement, and GOSS amplification through `expand_row_scale`
including that the amplified weight sums back to the full row count.

Not run, and therefore not claimed. `pixi run test`, `pixi run test-python`,
the GPU suites, and `tools/check_parity.py`. No integration edit above has been
compiled. Nothing in this lane has been benchmarked.

## 8. Risks

1. The alias spellings are diffed against a LightGBM checkout, but that
   checkout is 4.7.0.99 rather than the 4.7.0 tag the parity doc cites, and it
   lives in a temporary directory. Section 3 says what to repeat and why.
2. Adding a `TreeParams` field touches a struct many call sites construct. The
   append-at-the-end instruction in 4.1 keeps positional callers working, but
   check `src/mojoboost/serialize.mojo` and the C API for anything that depends
   on `TreeParams` field order or arity before landing.
3. Class bagging is gated to binary classification by the check in 4.3. If a
   later lane wants it for the ranker or for multiclass, that is a new design
   question, not a relaxation of the check.
4. `feature_fraction_bylevel` is an XGBoost parameter in a LightGBM parity
   project. If the parity owner would rather not carry a non-LightGBM knob, the
   per-level draw can stay in `sampling.mojo` unexported and unwired, and only
   the alias table and the class bagging work land. Sections 4.1 and 4.2 are
   the only ones that would be dropped.
5. Section 5 was read off A1's `gpu_active_rows.mojo` as it stood at the end of
   this lane, and that file is itself uncommitted parallel work. If A1's
   `LeafRange` or `begin_tree` signature moves, re-read section 5 before
   trusting it. Nothing in `sampling.mojo` imports A1, so no code breaks either
   way.
6. `expand_row_scale` has no caller yet. It is the only new function in this
   lane that is speculative rather than gap-closing, and section 5 point 5 says
   which lane has to want it before it is wired.
