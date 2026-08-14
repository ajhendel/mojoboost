# Task 12 handoff: the central validation layer

Lane files (the only ones this lane wrote):

- `src/mojoboost/validation.mojo` (new)
- `python/mojoboost/_validation.py` (new)
- `docs/VALIDATION_CONTRACT.md` (new)
- `handoffs/remaining_12_validation.md` (this file)

Read and not edited: `trainset.mojo`, `raw_data.mojo`, `sparse.mojo`,
`boosting.mojo`, `params.mojo`, `callback.mojo`, `custom_metric.mojo`,
`serialize.mojo`, `binning.mojo`, `ranking.mojo`, `metrics.mojo`,
`objective.mojo`, `categorical.mojo`, `class_weight.mojo`, `tree.mojo`,
`gpu_runtime.mojo`, `lgbm_model_io.mojo`, `__init__.mojo`,
`bindings/_mojoboost.mojo`, `bindings/binding_support.mojo`,
`python/mojoboost/_arrays.py`, `python/mojoboost/basic.py`.

Nothing was committed, staged, built, run, or tested.

> **All four lane files are already tracked at HEAD, committed by other
> sessions.** `5085097` ("Expand validation and advanced training
> integration") picked up `src/mojoboost/validation.mojo`,
> `python/mojoboost/_validation.py`, and `docs/VALIDATION_CONTRACT.md`;
> `e28a24d` ("Integrate advanced training and public API work") picked up
> this handoff and the remaining `validation.mojo` delta (the `: Int`
> annotations on the `comptime` ceilings, and the `scan_column` bounds guard
> — negative-feature check, `checked_mul` for the column base offset, and a
> slice-bound check against `len(values)`). All four match what this lane
> wrote; three of them now produce **zero diff** against HEAD, and the only
> uncommitted content in the lane is the test-expectation detail added to
> **P14**, **P15**, and open question 3 below. This lane committed nothing
> itself. An integrator should not read the zero-diff files as unfinished
> work, and should not re-apply them.

> **Concurrent work is live in this tree.** While this lane was reading,
> another session extended `trainset.mojo` (reference binning,
> `subset_shared_binning`, sparse datasets, `from_binned_dense` /
> `from_binned_sparse`) and added `serialize.save_dataset` /
> `load_dataset`. `_check_labels` moved from line 51 to line 102 mid-task.
> **Every patch below is anchored on a symbol name, never on a line
> number**, for exactly that reason. Nothing in this lane's three files
> overlaps anything that session touched, and no file outside the four
> listed above was modified.

## 1. What landed

Two modules and a contract.

`src/mojoboost/validation.mojo` holds the native rules. It **imports nothing
from the package** and takes primitives rather than mojoboost structs. That
is load-bearing: `boosting`, `trainset`, `serialize`, `sparse`, `params`,
`callback`, and the bindings are all meant to call into it, and a dependency
in the other direction would close a cycle at the first adoption. A caller
unpacks its struct at the call site.

`python/mojoboost/_validation.py` holds the ecosystem-object structure rules
and deliberately holds no numeric rule at all. It never calls `np.isfinite`,
`np.isinf`, or `np.isnan`, never compares a value to a bound, and never sums
anything.

`docs/VALIDATION_CONTRACT.md` states which layer owns what, and why the line
falls where it does. Section 4 of that document lists what the validation
layer deliberately does **not** own (objective semantics, serialization
grammar, device policy, transport framing); read it before adding a check.

### 1.1 Three real defects this lane found

These are why the module is worth adopting rather than a tidying exercise.

**(a) A loaded tree can contain a cycle.** `serialize._read_trees` checks
`0 <= left[i] < n_nodes` and the same for `right[i]`. It does **not** check
`left[i] > i`. Every grower in the package appends nodes as it splits them,
and `lgbm_model_io` emits in preorder for the same reason and says so in its
own comment, so parent-before-child is a real invariant that nothing
enforces at the boundary where it can be violated. `predict_raw_row` walks
down with a `while` loop that has no visit budget and the exact-contribution
path recurses once per edge, so a file whose node 3 names node 1 as its left
child passes the reader and then hangs or overflows the stack.
`validation.check_tree_topology` closes this. **This is the highest-value
patch in the list (P9).**

**(b) A callback can return anything and training continues.**
`custom_metric.train_with_callbacks` tests for `ABORT`, then for `STOP`, and
treats everything else as `CONTINUE`. A callback returning `7`, or a bridge
returning a default on a path that fell off the end of a branch, keeps
training silently. `validation.check_control_code` closes this (P8).

**(c) `params._validate` and `callback.check_resettable` have already
drifted.** They are the same nine range checks written twice, and `_validate`
bounds `feature_fraction_bylevel` while `check_resettable` does not. A
parameter schedule can therefore set a bylevel fraction of `0.0` that the
parser would have rejected, after which the level selects no features.
`validation.check_booster_ranges` closes this (P6, P7).

### 1.2 Counts read off disk are unbounded today

`serialize._read_mapper` reads `n_edges` and passes it straight to
`List[Float64](capacity=n_edges)`. `_read_trees` does the same with
`n_trees`, `n_nodes`, and `n_codes`. The grammar is satisfied by any integer
at all, including one that sizes a list larger than the machine. The
`MAX_*` ceilings and `check_alloc` / `checked_mul` exist for this; P9 wires
them in.

## 2. Public API effect of the lane as it stands

None. Neither new module is exported: `src/mojoboost/__init__.mojo` is not
this lane's file, and `python/mojoboost/__init__.py` is not either. Until
P0 lands, `validation.mojo` is reachable only as
`from .validation import ...` from inside the package, and `_validation.py`
only as `from ._validation import ...`. That is the intended state for a
lane that must not edit other files: nothing observable changed.

## 3. Integration patches

Every patch is **READY TO APPLY** and mechanical. Each states: target file
and symbol, the signature it calls, the call site, state flow, errors,
ownership, fallback, serialization effect, public API effect, dependency,
and the minimal validation to run afterward. **All validation below is
UNRUN** — this lane ran no build, no test, and no program.

Apply in order. P0 first (nothing else compiles without the import path
being real), then P9 (the defect), then the rest in any order.

---

### P0 — export the module

- **Target file / symbol:** `src/mojoboost/__init__.mojo`, top-level imports.
- **Signature:** none; an import block.
- **Call site:** append after the `from .params import (...)` block, keeping
  the file's existing ordering convention (leaf modules before the ones that
  consume them; `validation` has no package dependencies, so it may sit
  anywhere, but placing it beside `params` keeps the parameter-facing
  symbols together).
- **Patch:**

```mojo
from .validation import (
    CONTROL_CODES,
    MAX_ALLOC_ELEMS,
    MAX_BIN_COUNT,
    MAX_CATEGORY_CODE,
    MAX_CLASSES,
    MAX_DEPTH_LIMIT,
    MAX_FEATURES,
    MAX_ITERATIONS,
    MAX_MODEL_NODES,
    MAX_MODEL_TREES,
    MAX_NNZ,
    MAX_RELEVANCE,
    MAX_ROWS,
    MAX_TREE_NODES,
    CancelToken,
    ColumnReport,
    check_alloc,
    check_ascending_rows,
    check_booster_ranges,
    check_categorical_features,
    check_category_code,
    check_class_codes,
    check_class_count,
    check_classes_present,
    check_cleanup_balanced,
    check_column_length,
    check_columns_usable,
    check_compressed,
    check_control_code,
    check_csc,
    check_csr,
    check_dataset_columns,
    check_dense_matrix,
    check_depth_budget,
    check_early_stopping_rounds,
    check_feature_index,
    check_features_finite,
    check_finite_vector,
    check_gradient_pair,
    check_group_boundaries,
    check_group_counts,
    check_iteration_range,
    check_iterations,
    check_labels_finite,
    check_loaded_tree,
    check_mapper_header,
    check_max_bin,
    check_max_depth,
    check_model_nodes,
    check_multiclass_inputs,
    check_nonnegative_scalar,
    check_num_leaves,
    check_positive_hessian_total,
    check_positive_scalar,
    check_ranking_inputs,
    check_relevance_labels,
    check_required_length,
    check_row_index,
    check_shape,
    check_sparse_values_finite,
    check_training_inputs,
    check_tree_count,
    check_tree_header,
    check_tree_topology,
    check_unit_fraction,
    check_valid_set,
    check_weights,
    checked_add,
    checked_cells,
    checked_mul,
    scan_column,
    tree_depth,
)
```

- **State flow:** none.
- **Errors:** none added.
- **Ownership:** `__init__.mojo` is unowned by this lane.
- **Fallback:** if the package prefers a narrower public surface, export only
  `CancelToken`, `ColumnReport`, and the `check_*` names, and leave the
  `MAX_*` ceilings and `checked_*` helpers module-private. Every patch below
  imports from `.validation` directly and works either way.
- **Serialization effect:** none.
- **Public API effect:** additive. `mojoboost.check_shape` and its
  neighbors become importable. No existing name is shadowed; `check_labels`
  and `check_groups` in `ranking` keep their names, and this module's
  equivalents are `check_relevance_labels` and `check_group_boundaries`
  precisely so the two do not collide in the package namespace.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** `pixi run mojo build src/mojoboost`
  or the repo's equivalent single-module compile, to confirm the import
  block resolves and no name collides.

---

### P9 — bound and topology-check what the model reader allocates

**This is the defect patch. Apply it before the cleanups.**

- **Target file / symbols:** `src/mojoboost/serialize.mojo`, `_read_mapper`,
  `_read_trees`, `_read_categorical`.
- **Signatures called:**
  - `check_mapper_header(n_features: Int, n_bins: Int, n_edges: Int) raises`
  - `check_tree_count(n_trees: Int) raises`
  - `check_alloc(n_elems: Int, what: String) raises`
  - `check_loaded_tree(feature: List[Int], left: List[Int], right: List[Int], n_nodes: Int, n_leaves: Int, n_features: Int, running_total: Int) raises -> Int`
- **Import to add:** `from .validation import (check_alloc, check_loaded_tree, check_mapper_header, check_tree_count)`
- **Call sites and patch:**

1. In `_read_mapper`, replace

```mojo
    if n_features < 1 or n_edges < 0:
        raise Error("corrupt mapper header")
```

with

```mojo
    check_mapper_header(n_features, n_bins, n_edges)
```

  This is strictly stronger: it keeps both existing conditions, adds the
  `n_bins` range (`[2, 256]`, which the `UInt8` binned matrix requires and
  which the reader does not check today), and adds the derived edge ceiling
  `n_features * (n_bins - 1)`, above which no binning could have produced
  the file. It runs **before** `List[Float64](capacity=n_edges)`.

2. In `_read_categorical`, after the existing
   `if n_flags != n_features or n_codes < 0:` check, add

```mojo
    check_alloc(n_codes, "categorical code count")
```

  before `List[Int](capacity=n_codes)`.

3. In `_read_trees`, replace

```mojo
    var n_trees = r.next_int()
    if n_trees < 0:
        raise Error("corrupt tree count")
```

with

```mojo
    var n_trees = r.next_int()
    check_tree_count(n_trees)
    var total_nodes = 0
```

  and replace the per-tree header check

```mojo
        if n_nodes < 1 or n_leaves < 1:
            raise Error("corrupt tree header")
```

with nothing at that point (it is subsumed), then replace the existing
per-node loop

```mojo
        for i in range(n_nodes):
            if feature[i] >= n_features:
                raise Error("corrupt tree: feature index out of range")
            if feature[i] >= 0 and (
                left[i] < 0
                or left[i] >= n_nodes
                or right[i] < 0
                or right[i] >= n_nodes
            ):
                raise Error("corrupt tree: child index out of range")
```

with

```mojo
        total_nodes = check_loaded_tree(
            feature, left, right, n_nodes, n_leaves, n_features, total_nodes
        )
```

  **Ordering caveat:** `check_tree_header` (called inside
  `check_loaded_tree`) must also run *before* the `capacity=n_nodes`
  allocations if the ceiling is to do its job. Two options, pick one:

  - **(i) preferred)** hoist a bare `check_tree_header(n_nodes, n_leaves)`
    to immediately after the two `r.next_int()` calls, keeping
    `check_loaded_tree` where the old loop was. `check_tree_header` is
    idempotent and cheap, so calling it twice is harmless.
  - **(ii)** accept that a hostile `n_nodes` allocates once before it is
    refused. Not recommended; the whole point is the allocation.

- **State flow:** `total_nodes` is a new local accumulator in `_read_trees`,
  threaded through the loop and discarded when the function returns. It
  crosses no struct boundary.
- **Errors:** all new failures are `Error` with the existing
  `"corrupt tree: ..."` / `"corrupt mapper header: ..."` prefixes plus the
  offending index and value. Messages that were previously bare (e.g.
  `"corrupt tree header"`) become specific, which changes their text. **Any
  test asserting on those exact strings will need its expectation updated;**
  this lane did not search the test tree, so treat that as an open item.
- **Ownership:** `serialize.mojo` is unowned by this lane.
- **Fallback:** if the topology check turns out to reject a real file
  produced before parent-before-child was universal, gate it:
  `check_tree_topology` on version >= 3 and the old bounds check on v1/v2.
  This lane found no such file and `lgbm_model_io` documents the same
  ordering, so the gate is a contingency, not a recommendation.
- **Serialization effect:** **read side only, and no format change.** No
  byte written by `save_model` changes, and every file the current writer
  produces still loads: growers append children after parents, so
  `check_tree_topology` accepts every model this package can write. What
  becomes unloadable is a file that was already unpredictable.
- **Public API effect:** `load_model` and `load_multiclass_model` gain
  failure modes for inputs that previously hung or crashed. No signature
  changes.
- **Dependency:** P0 not required (direct `from .validation import`).
- **Minimal later validation (UNRUN):** load one model saved by the current
  writer and confirm it round-trips; hand-edit one saved file to point a
  child index backwards and confirm it now raises instead of hanging.

---

### P1 — `Dataset` construction

- **Target file / symbols:** `src/mojoboost/trainset.mojo` — `Dataset.__init__`
  (the opening validation block), `_check_labels`, `_int_labels`,
  `_relevance_labels`.
- **Signatures called:**
  - `check_dataset_columns(n_rows, n_features, n_values, n_label, n_weight, n_group, n_init_score, n_feature_names) raises`
  - `check_group_counts(counts: List[Int], n_rows: Int) raises -> Int`
  - `check_categorical_features(features: List[Int], n_features: Int) raises`
  - `check_required_length(n_values: Int, n_rows: Int, name: String) raises`
  - `check_class_codes(label: List[Float64], n_classes: Int) raises -> List[Int]`
  - `check_relevance_labels(label: List[Float64]) raises -> List[Int]`
- **Import to add:** `from .validation import (check_categorical_features, check_class_codes, check_dataset_columns, check_group_counts, check_relevance_labels, check_required_length)`
- **Call site / patch:** replace the whole block from
  `if n_rows < 1:` through the `categorical_features` duplicate loop with

```mojo
        check_dataset_columns(
            n_rows,
            n_features,
            len(features),
            len(label),
            len(weight),
            len(group),
            len(init_score),
            len(feature_names),
        )
        if len(group) != 0:
            _ = check_group_counts(group, n_rows)
        check_categorical_features(categorical_features, n_features)
```

  and rewrite the three module-level helpers as one-line delegations:

```mojo
def _check_labels(label: List[Float64], n_rows: Int) raises:
    check_required_length(len(label), n_rows, "label")


def _int_labels(label: List[Float64], n_classes: Int) raises -> List[Int]:
    return check_class_codes(label, n_classes)


def _relevance_labels(label: List[Float64]) raises -> List[Int]:
    return check_relevance_labels(label)
```

  Keeping the three private names means no other call site in the file
  moves. Delete them once the callers are updated, if that is wanted.

- **State flow:** unchanged. `_int_labels` and `_relevance_labels` still
  return an owned `List[Int]` that the trainer consumes.
- **Errors:** every message gains the offending index and both counts.
  `_relevance_labels` additionally starts enforcing the `[0, MAX_RELEVANCE]`
  bound that `train_ranker` enforced two calls later; the failure moves
  earlier and names the row. Behavior for accepted inputs is identical.
- **Ownership:** `trainset.mojo` is unowned and is being **actively edited
  by another session**. Apply this patch on top of whatever that session
  leaves; it touches only the helper bodies and the constructor's opening
  block, neither of which that session's additions (reference binning,
  binned-dense constructors) appear to occupy.
- **Fallback:** if `check_dataset_columns` is too coarse for a constructor
  that has grown extra columns, call the four narrow checks
  (`check_dense_matrix`, three `check_column_length`) directly instead.
- **Serialization effect:** none directly. Note that
  `serialize.load_dataset` (added by the concurrent session) reconstructs a
  `Dataset` and so inherits whichever construction path it uses; see P10.
- **Public API effect:** none. `Dataset.__init__` keeps its signature.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** construct one dataset with a
  mismatched `weight` length and confirm the message names both counts.

---

### P2 — `RawData`

- **Target file / symbols:** `src/mojoboost/raw_data.mojo` — `RawData.dense`,
  `RawData.check_rows`.
- **Signatures called:** `check_dense_matrix(n_values, n_rows, n_features) raises`,
  `check_ascending_rows(rows: List[Int], n_rows: Int) raises`
- **Import to add:** `from .validation import check_ascending_rows, check_dense_matrix`
- **Patch:** in `dense`, replace the two `if`/`raise` pairs with
  `check_dense_matrix(len(values), n_rows, n_features)`. In `check_rows`,
  replace the body with `check_ascending_rows(rows, self.n_rows)`.
- **State flow:** unchanged.
- **Errors:** `dense` gains the overflow guard on `n_rows * n_features`,
  which today is computed as a raw product in the length comparison and can
  wrap. `check_rows` gains the "more selected rows than rows exist" case,
  which is currently only caught per-element.
- **Ownership:** `raw_data.mojo` is unowned.
- **Fallback:** none needed; both are pure substitutions.
- **Serialization effect:** none.
- **Public API effect:** `RawData.check_rows` keeps its signature and
  becomes a one-line forwarder. Message text changes.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** call `RawData.dense` with a shape
  whose product overflows and confirm it raises the ceiling message.

---

### P3 — compressed sparse structure

- **Target file / symbol:** `src/mojoboost/sparse.mojo` — `_check_compressed`.
- **Signature called:** `check_compressed(offsets, indices, n_values, n_outer, n_inner, kind, outer, inner) raises`
- **Import to add:** `from .validation import check_compressed`
- **Patch:** replace the body of `_check_compressed` with a forward. Note
  the **signature difference**: the native check takes a stored-entry *count*
  rather than the values list, so that one function serves raw `Float64`
  matrices and binned `UInt8` ones. Adapt at the seam:

```mojo
def _check_compressed(
    offsets: List[Int],
    indices: List[Int],
    values: List[Float64],
    n_outer: Int,
    n_inner: Int,
    kind: String,
    outer: String,
    inner: String,
) raises:
    check_compressed(
        offsets, indices, len(values), n_outer, n_inner, kind, outer, inner
    )
```

  Then `SparseBinnedMatrix.validate`, whose values are not `Float64`, can
  call `check_compressed` directly with its own `len(...)` and stop needing
  a parallel implementation.

- **State flow:** unchanged.
- **Errors:** every message gains the offending entry index and both index
  values. New: `nnz` is now bounded by `MAX_NNZ` and passed through
  `check_alloc` before any offset is dereferenced.
- **Ownership:** `sparse.mojo` is unowned.
- **Fallback:** keep `_check_compressed` as the private wrapper permanently
  if the `List[Float64]` signature is load-bearing at its call sites.
- **Serialization effect:** none.
- **Public API effect:** `CscMatrix.validate` / `CsrMatrix.validate` keep
  their signatures; message text changes.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** validate one canonical CSC matrix
  (must pass) and one with a duplicated row index (must name the entry).

---

### P4 — trainer input contracts

- **Target file / symbols:** `src/mojoboost/boosting.mojo` —
  `_check_sample_weight`, and the `target length must equal n_rows` /
  `init_score length must equal n_rows` / `labels length must equal n_rows`
  guards in `train`, `train_more`, `train_with_valid`, `train_multiclass`,
  `train_multiclass_more`, `train_multiclass_with_valid`.
- **Signatures called:**
  - `check_weights(weight: List[Float64], n_rows: Int) raises -> Float64`
  - `check_required_length(n_values: Int, n_rows: Int, name: String) raises`
  - `check_column_length(n_values: Int, n_rows: Int, name: String) raises`
  - `check_valid_set(train_n_features, valid_n_features, valid_n_rows, n_valid_label) raises`
  - `check_iterations(n_estimators: Int) raises`
  - `check_early_stopping_rounds(rounds: Int) raises`
- **Import to add:** `from .validation import (check_column_length, check_early_stopping_rounds, check_iterations, check_required_length, check_valid_set, check_weights)`
- **Patch:** replace `_check_sample_weight`'s body with
  `_ = check_weights(weights, n)`. Replace each length guard with the
  matching `check_required_length` / `check_column_length` call. Replace the
  valid-set triples with one `check_valid_set`. Replace
  `if params.n_estimators < 0` with `check_iterations(...)` and
  `if early_stopping_rounds <= 0` with `check_early_stopping_rounds(...)`
  where the semantics match (**check each site**: some treat `0` as
  disabling and some require positive; `check_early_stopping_rounds` treats
  `0` as disabling, so a site that requires positive keeps its own check on
  top).
- **State flow:** `check_weights` **returns** the validated total, which
  `_check_sample_weight` currently recomputes and discards, and which
  several callers sum again downstream. Threading the returned total through
  is the follow-on cleanup and is *not* part of this patch; keep the
  discard (`_ = ...`) so the patch stays behavior-preserving.
- **Errors:** messages gain the row index and both counts. `_check_objective`
  is untouched: label *domain* by objective stays where it is (see
  `docs/VALIDATION_CONTRACT.md` section 4).
- **Ownership:** `boosting.mojo` is unowned.
- **Fallback:** apply per-function; the six trainers are independent.
- **Serialization effect:** none.
- **Public API effect:** none. Message text changes.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** train once with an all-zero
  `sample_weight` and confirm the message still says the sum must be
  positive.

---

### P5 — custom-objective gradients

- **Target file / symbol:** `src/mojoboost/objective.mojo` —
  `check_custom_grad_hess`.
- **Signature called:** `check_gradient_pair(grad: List[Float64], hess: List[Float64], n_rows: Int) raises -> Float64`
- **Import to add:** `from .validation import check_gradient_pair`
- **Patch:** replace the body with `_ = check_gradient_pair(grad, hess, n_rows)`.
- **State flow:** the returned hessian total is available and currently
  discarded. **Do not add a positive-total requirement here.**
  `check_gradient_pair` deliberately does not judge the total, because a
  converged custom objective can legitimately return all-zero curvature for a
  round and the right answer is a root-only tree, not a raise. A caller whose
  next step divides by the total opts in with
  `check_positive_hessian_total(total, where)`.
- **Errors:** messages change from
  `"custom objective produced a non-finite gradient at row R"` to
  `"gradients must be finite: row R is V"`. If the `"custom objective"`
  attribution matters for the Python bridge's error text, keep the wrapper's
  own prefix by catching and re-raising, or leave this patch unapplied; it is
  the lowest-value one in the list.
- **Ownership:** `objective.mojo` is unowned.
- **Fallback:** skip. Nothing else depends on this patch.
- **Serialization effect:** none.
- **Public API effect:** `check_custom_grad_hess` is exported from
  `__init__.mojo`. Its signature is unchanged; its message text is not.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** run one custom objective that emits
  a negative hessian and confirm it still raises.

---

### P6 — parameter-string ranges

- **Target file / symbol:** `src/mojoboost/params.mojo` — `_validate`.
- **Signatures called:**
  `check_booster_ranges(n_estimators, learning_rate, num_leaves, max_depth, min_data_in_leaf, min_child_hess, lambda_l1, lambda_l2, feature_fraction, feature_fraction_bynode, feature_fraction_bylevel) raises`,
  `check_max_bin(max_bin: Int) raises`,
  `check_class_count(n_classes: Int) raises`
- **Import to add:** `from .validation import check_booster_ranges, check_class_count, check_max_bin`
- **Patch:** replace everything in `_validate` from
  `if config.booster.n_estimators < 0:` through
  `if config.max_bin < 2:` with

```mojo
    check_booster_ranges(
        config.booster.n_estimators,
        config.booster.learning_rate,
        config.booster.tree.num_leaves,
        config.booster.tree.max_depth,
        config.booster.tree.min_data_in_leaf,
        config.booster.tree.min_child_hess,
        config.booster.tree.lambda_l1,
        config.booster.tree.lambda_reg,
        config.booster.tree.feature_fraction,
        config.booster.tree.feature_fraction_bynode,
        config.booster.tree.feature_fraction_bylevel,
    )
    config.booster.tree.extra.check_scalars(
        config.booster.tree.min_data_in_leaf
    )
    check_max_bin(config.max_bin)
```

  Leave the objective/alpha block and the `num_class` block below it exactly
  as they are; only replace `if config.n_classes < 2:` with
  `check_class_count(config.n_classes)` inside the multiclass branch. The
  `extra.check_scalars` call must stay: `ExtraTreeParams` is a struct and
  this module takes primitives.

- **State flow:** none.
- **Errors:** `learning_rate`, `min_sum_hessian_in_leaf`, `lambda_l1`, and
  `lambda_l2` gain a finiteness check they did not have (a NaN
  `learning_rate` currently passes `<= 0.0`). `num_leaves`, `max_depth`,
  `num_iterations`, and `num_class` gain upper bounds. Every message gains
  the offending value.
- **Ownership:** `params.mojo` is unowned.
- **Fallback:** if the new upper bounds are unwanted at parse time, call the
  narrow checks (`check_positive_scalar`, `check_unit_fraction`,
  `check_nonnegative_scalar`) individually instead of the composite.
- **Serialization effect:** none.
- **Public API effect:** `parse_params` rejects a NaN `learning_rate` and an
  absurd `num_leaves` that it previously accepted. Both are strictly better
  refusals; note them in the changelog.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** parse `learning_rate=0.05` (passes)
  and `feature_fraction_bylevel=0` (must raise).

---

### P7 — parameter resets, and the drift fix

- **Target file / symbol:** `src/mojoboost/callback.mojo` — `check_resettable`.
- **Signature called:** same `check_booster_ranges` as P6.
- **Import to add:** `from .validation import check_booster_ranges`
- **Patch:** keep the two "not resettable" identity checks at the top
  (`n_estimators`, `feature_fraction_seed`) — those compare `before` against
  `after` and are this module's, not the validation layer's. Replace
  everything from `if after.learning_rate <= 0.0:` to the end of the
  function with the same `check_booster_ranges(...)` call as P6, reading
  from `after`.
- **State flow:** none.
- **Errors:** identical wording to P6, which is the point: a caller who sets
  `learning_rate=0` through a parameter string and one who sets it through a
  schedule now get the same sentence.
- **Ownership:** `callback.mojo` is unowned.
- **Fallback:** none needed.
- **Serialization effect:** none.
- **Public API effect:** **a behavior change worth calling out.** A callback
  that sets `feature_fraction_bylevel` to `0.0` (or to a NaN, or
  `num_leaves` past the tree ceiling) is now rejected where it previously
  passed. That is the drift being closed, and it can fail a schedule that
  currently "works" by selecting nothing at a level.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** a schedule that halves
  `learning_rate` each round still runs; one that sets
  `feature_fraction_bylevel=0.0` now raises.

---

### P8 — callback control codes

- **Target file / symbol:** `src/mojoboost/custom_metric.mojo` —
  `train_with_callbacks` and its multiclass/ranking siblings, at each of the
  four `var before = callback(...)` / `var after = callback(...)` sites.
- **Signature called:** `check_control_code(code: Int, phase: String, iteration: Int) raises -> Int`
- **Import to add:** `from .validation import check_control_code`
- **Patch:** wrap each call:

```mojo
        var before = check_control_code(
            callback(BEFORE_ITERATION, env), "before-iteration", i
        )
```

```mojo
        var after = check_control_code(
            callback(AFTER_ITERATION, env), "after-iteration", i
        )
```

  The existing `if before == ABORT:` / `if before == STOP:` branches stay
  exactly as they are. `check_control_code` returns the code, so the rest of
  the function is untouched.

- **State flow:** none.
- **Errors:** a new failure for an out-of-range code, naming the code, the
  phase, and the round. Previously silent.
- **Ownership:** `custom_metric.mojo` is unowned; `callback.mojo` is too.
- **Fallback:** none needed.
- **Serialization effect:** none.
- **Public API effect:** a Python callback that returns a non-code (an
  accidental `None` bridged to some default, a truthy object) now raises
  instead of training on. This is the intended fix and it can surface
  latent bugs in user callbacks; note it in the changelog.
- **Dependency:** **one coupling to make explicit.**
  `validation.CONTROL_CODES` is `3` and the module knows only that legal
  codes are `[0, CONTROL_CODES)`. The names live in `callback.mojo`. Add
  there, inside any function body (Mojo requires it):

```mojo
    comptime assert CONTINUE == 0 and STOP == 1 and ABORT == 2, (
        "validation.check_control_code assumes the callback control codes"
        " are 0, 1, 2"
    )
```

  `no_callback` is the natural home for it.
- **Minimal later validation (UNRUN):** a callback returning `CONTINUE`
  still trains; one returning `7` raises naming round and phase.

---

### P10 — binning entry, and the prepared-table reader

- **Target file / symbols:** `src/mojoboost/binning.mojo` — `fit_bins`,
  `BinMapper.transform`; and `src/mojoboost/serialize.mojo` —
  `load_dataset` (added by the concurrent session; this lane read only its
  signature).
- **Signatures called:** `check_dense_matrix`, `check_max_bin`,
  `check_features_finite`, `check_columns_usable`, `check_alloc`
- **Import to add:** `from .validation import (check_columns_usable, check_dense_matrix, check_features_finite, check_max_bin)`
- **Patch (binning):** replace `fit_bins`'s three opening guards with

```mojo
    check_max_bin(max_bins)
    check_dense_matrix(len(features), n_rows, n_features)
```

  Note `fit_bins` currently does **not** check `n_features >= 1`, only
  `n_rows`; `check_dense_matrix` adds it. Then, optionally and behind the
  caller's choice, add `check_features_finite(features, n_rows, n_features)`
  as the one place an infinity is refused natively rather than only in the
  Python layer, and `check_columns_usable(...)` to report a matrix on which
  no split is possible.
- **Patch (load_dataset):** every count it reads before an allocation should
  pass `check_alloc`, and its mapper header should go through
  `check_mapper_header` exactly as P9 does for `load_model`. This lane did
  not read that function's body; treat this bullet as a directed review
  rather than a literal patch.
- **State flow:** none.
- **Errors:** `fit_bins` gains the zero-feature case and, if the optional
  calls are added, the infinity and all-constant cases.
- **Ownership:** both files unowned; `serialize.mojo` is being actively
  extended.
- **Fallback:** the two optional calls are genuinely optional.
  `check_features_finite` is O(cells) and duplicates work numpy already does
  on the Python path, so gate it on a parameter or apply it only on the
  paths that do not come through `_arrays` (the C ABI, the CLI, native
  callers). **That gating decision is the one open design question this lane
  leaves.**
- **Serialization effect:** none for `fit_bins`. For `load_dataset`, read
  side only, no format change.
- **Public API effect:** `fit_bins(features, n_rows, 0)` now raises. It
  previously produced a mapper with no features.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** fit bins on a matrix with one
  all-NaN column (must succeed, that column is simply unsplittable) and on
  one where every column is constant (must raise, if
  `check_columns_usable` was adopted).

---

### P11 — ranking groups and relevance bounds

- **Target file / symbols:** `src/mojoboost/ranking.mojo` —
  `groups_from_counts`, `check_groups`, `check_labels`, `MAX_RELEVANCE_LABEL`.
- **Signatures called:** `check_group_counts(counts, n_rows) raises -> Int`,
  `check_group_boundaries(starts: List[Int], n_rows: Int) raises`,
  `MAX_RELEVANCE`
- **Import to add:** `from .validation import MAX_RELEVANCE, check_group_boundaries, check_group_counts`
- **Patch:** `groups_from_counts` does not know `n_rows` (it derives the
  total), so it keeps its loop but should accumulate through `checked_add`.
  `check_groups(groups, n_rows)` replaces its body with
  `check_group_boundaries(groups.starts, n_rows)` plus its existing
  `groups.n_rows != n_rows` check. `check_labels` keeps its body but reads
  the bound from `MAX_RELEVANCE`.
- **State flow:** unchanged.
- **Errors:** boundary messages gain the offending boundary index and both
  values.
- **Ownership:** `ranking.mojo` is unowned.
- **Fallback:** the `MAX_RELEVANCE_LABEL` unification is the important half;
  the rest is cosmetic. **`validation.MAX_RELEVANCE` is `30` and must stay
  equal to `ranking.MAX_RELEVANCE_LABEL`, because `label_gain` reads a table
  sized `MAX_RELEVANCE_LABEL + 1` and a larger label would index off its
  end.** If the two are unified, delete one; if not, add a `comptime assert`
  in `ranking` that they are equal.
- **Serialization effect:** none.
- **Public API effect:** `MAX_RELEVANCE_LABEL` stays exported at its current
  value.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** train one ranker with a valid
  `group` (passes) and one whose counts sum short (must name both totals).

---

### P12 — metric weights

- **Target file / symbol:** `src/mojoboost/metrics.mojo` — `check_metric_weight`.
- **Signature called:** `check_weights(weight: List[Float64], n_rows: Int) raises -> Float64`
- **Import to add:** `from .validation import check_weights`
- **Patch:** replace the body with `return check_weights(weight, n)`. The
  empty-vector case already returns `Float64(n)` in both, and the
  finite/nonnegative/positive-sum rules are identical, so this is a pure
  substitution.
- **State flow:** the total is returned as before, summed in the same
  front-to-back order, so no metric's floating-point result changes.
- **Errors:** messages gain the row index and the value; `"weights"` becomes
  `"sample_weight"`, which is what the caller passed. If the metric-facing
  wording matters, add a `name` parameter to `check_weights` as a follow-on;
  this lane kept the signature narrow deliberately.
- **Ownership:** `metrics.mojo` is unowned.
- **Fallback:** skip; low value, and the wording change may not be wanted.
- **Serialization effect:** none.
- **Public API effect:** `check_metric_weight` is exported. Signature
  unchanged, message text changed.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** score one weighted metric and
  confirm the value is bit-identical to before.

---

### P13 — the binding boundary

- **Target files / symbols:** `bindings/binding_support.mojo` —
  `f64_buffer`, `int_buffer_from_f64`, `write_f64_buffer`;
  `bindings/_mojoboost.mojo` — `fit`, `fit_custom`, `fit_with_metrics`, and
  the other entry points that take `n_rows` and `n_features`.
- **Signatures called:** `check_alloc(n_elems, what) raises`,
  `check_shape(n_rows, n_features) raises`, `checked_cells(n_rows, n_features) raises -> Int`
- **Import to add:** `from mojoboost.validation import check_alloc, check_shape, checked_cells` (adjust to the bindings' existing import style).
- **Patch:** in `binding_support`, replace
  `if n < 0: raise Error("buffer length must not be negative, got ", n)`
  with `check_alloc(n, "buffer length")`, keeping the null-address check
  above it. In `_mojoboost.fit` and siblings, add `check_shape(n_rows, n_features)`
  immediately after the two ints are read from `PythonObject`, before any
  buffer is materialized, and derive the matrix length with
  `checked_cells(n_rows, n_features)` rather than a bare product.
- **State flow:** none.
- **Errors:** an absurd or overflowing shape is refused at the boundary
  rather than after the Python side has already allocated. The null-address
  check is unchanged and stays first.
- **Ownership:** both binding files are unowned.
- **Fallback:** `binding_support` alone is worth doing even if the entry
  points are left alone.
- **Serialization effect:** none.
- **Public API effect:** `mojoboost._mojoboost.fit` raises `RuntimeError`
  (via the existing `Error` bridge) for a shape it previously accepted and
  then read past. No signature changes.
- **Dependency:** P0 helps if the bindings import from the package root
  rather than the module.
- **Minimal later validation (UNRUN):** one normal fit through the Python
  estimators still works.

---

### P14 — Python: delegate structure, drop the numeric domain

- **Target file / symbol:** `python/mojoboost/_arrays.py`.
- **Signatures called:** `_validation.check_shape`, `check_ndim`,
  `check_rectangular`, `check_length`, `check_optional_length`,
  `check_float64_convertible`, `check_sparse_layout`,
  `check_sparse_index_width`, `frame_column_names`, `is_sparse`.
- **Import to add:** `from . import _validation`
- **Patch, in three steps:**

  1. **Delegate structure.** `_canonical_sparse` becomes
     `_validation.check_sparse_layout`. `feature_names` becomes
     `_validation.frame_column_names`. `_is_sparse` becomes
     `_validation.is_sparse`. The `ndim` / row-count / feature-count blocks
     inside `_as_column_major_numpy` and `_as_column_major_stdlib` become
     `_validation.check_ndim` and `_validation.check_shape` (and
     `check_rectangular` for the stdlib path). The `f64_vector` conversion
     block becomes `_validation.check_float64_convertible` followed by
     `_validation.check_length`.

  2. **Remove the numeric domain.** Delete `_require_finite` and the
     `np.isinf(Xa).any()` / `np.isinf(data).any()` blocks. `check_target`
     becomes `f64_vector` alone. `check_sample_weight` drops its
     nonnegativity and all-zeros checks. Those four rules now live in
     `validation.check_labels_finite`, `validation.check_weights`, and
     `validation.check_features_finite`, and P4 and P10 are what put them on
     the path.

  3. **Keep the shims.** Leave `check_X`, `check_X_sparse`, `check_target`,
     `check_sample_weight`, `column_major`, and `encode_labels` as names with
     their current signatures. `cv.py`, `dask.py`, `basic.py`, and
     `__init__.py` all call them, and none of those files is this lane's.

- **State flow:** the buffers `_arrays` returns are unchanged in dtype,
  order, and lifetime. The caller still keeps them referenced while their
  addresses are in flight.
- **Errors:** **step 2 is the part to sequence carefully.** Removing the
  Python finiteness checks before P4/P10 land means an infinite `X` reaches
  the binner unchecked. **Apply step 2 only after P4 and P10.** Steps 1 and 3
  are safe on their own and can land first.
- **Ownership:** `_arrays.py` is unowned by this lane.
- **Fallback:** apply steps 1 and 3 only, and leave the numeric checks
  duplicated. The duplication is the thing this task exists to remove, but a
  partial patch is safe and the contract document records the intent.
- **Serialization effect:** none.
- **Public API effect:** the exception *type* is preserved
  (`ValueError`/`TypeError` for structure). After step 2, a non-finite label
  raises from the native side, which the bridge surfaces as `RuntimeError`
  rather than `ValueError`. **That is a visible change to what a caller
  catches**, and it is the one thing in this whole handoff that needs a
  decision rather than a patch. Two options: (a) accept it and document it,
  or (b) have `_arrays` catch the bridged error and re-raise as `ValueError`,
  which preserves the type at the cost of one `try` per entry point. This
  lane recommends (b).
- **Tests that assert the rules step 2 removes.**
  `python/tests/test_validation.py` (committed, 179 lines, `8913489`) pins
  exactly the Python-side numeric domain. It is **not** this lane's file and
  was not edited. Step 2 moves each of these to the native error surface, so
  whoever applies it owns moving the expectation with it. Native wording is
  given so the `match=` string can be updated in the same edit:

  | Test (line) | Asserts | Native replacement | Wording still matches? |
  | --- | --- | --- | --- |
  | `test_x_rejects_infinities` (42) | `ValueError`, `"infinite"` | `check_features_finite` — `"feature values must not be infinite (NaN is allowed ...)"` | yes |
  | `test_predict_validates_like_fit` (166) | `ValueError`, `"infinite"` | same, on the predict path | yes |
  | `test_regression_target_must_be_finite` (82) | `ValueError`, `"NaN or infinite"` | `check_labels_finite` -> `check_finite_vector` — `"label must be finite: entry 1 is nan"` | **no** |
  | `test_classifier_rejects_non_finite_numeric_labels` (100) | `ValueError`, `"NaN or infinite"` | same (this one arrives via `encode_labels`, a distinct path — check whether step 2 even reaches it) | **no** |
  | `test_sample_weight_must_be_finite` (124) | `ValueError`, `"NaN or infinite"` | `check_weights` — `"sample_weight must be finite: row 1 is nan"` | **no** |
  | `test_sample_weight_must_be_nonnegative` (129) | `ValueError`, `"nonnegative"` | `check_weights` — `"sample_weight must be nonnegative: row 1 is -1.0"` | yes |
  | `test_sample_weight_must_not_be_all_zeros` (134) | `ValueError`, `"all zeros"` | `check_weights` — `"sample_weight must have a positive sum, got 0.0; an all-zero vector drops every row from training"` | **no** |

  Four of the seven need a new `match=` string. **All seven need the
  exception type resolved first** — they all expect `ValueError`, and a
  native raise crosses the bridge as `RuntimeError` unless option (b) below
  is taken. Two neighbours must keep passing unchanged:
  `test_x_allows_nan_as_the_missing_marker` (49) — `check_features_finite`
  permits NaN, by design — and
  `test_zero_weights_are_allowed_when_some_row_survives` (139) —
  `check_weights` rejects only a zero *total*, not a zero entry.
- **Dependency:** P4 and P10 for step 2.
- **Minimal later validation (UNRUN):** fit on a frame with a NaN feature
  (must succeed, NaN is missing) and on one with an `inf` feature (must
  raise, from whichever side owns it after the patch). Then re-run
  `python/tests/test_validation.py` and reconcile the seven rows above.

---

### P15 — Python: estimator entry points

- **Target files / symbols:** `python/mojoboost/basic.py` and
  `python/mojoboost/__init__.py` — the `fit` / `predict` methods of the
  estimators, at the point where `X` has become a buffer and the per-row
  columns have not yet been validated.
- **Signatures called:**
  - `_validation.check_fit_structure(X, y, n_rows, n_features, sample_weight=None, group=None, init_score=None, name="X")` -> `(n_rows, n_features)`
  - `_validation.check_predict_structure(X, n_features, fitted_n_features, fitted_names=None, n_rows=None, name="X")` -> `int`
  - `_validation.check_param_mapping`, `check_int_param`, `check_float_param`
- **Patch:** one call each, replacing the scattered length and width guards.
- **State flow:** returns the validated shape; the estimator keeps using it
  as it does now.
- **Errors:** `check_fit_structure` adds `check_frame_index_aligned`, which
  is new and is the one Python-side mistake no native check can see: a
  misaligned pandas index produces a valid buffer of the right length
  holding the wrong labels, and no metric will flag it.
  `check_predict_structure` adds order-sensitive feature-name matching,
  which catches the same columns rearranged.
- **Ownership:** both files unowned. `__init__.py` is 160 KB and is being
  edited by other lanes; anchor on method names.
- **Fallback:** adopt `check_predict_structure` alone. It is the higher-value
  half and touches fewer call sites.
- **Serialization effect:** none.
- **Public API effect:** additive refusals, both `ValueError`. A user who has
  been fitting on a misaligned frame will start seeing an error; that is the
  point, and it should be in the changelog. **One existing expectation
  breaks on wording, not on type:**
  `python/tests/test_validation.py::test_predict_rejects_a_different_feature_count`
  (line 161) asserts `ValueError, match="expecting 4 features"`, while
  `check_feature_count_matches` raises
  `"X has 5 features, but this model was fitted on 4"`. The type is right and
  the meaning is identical. Either update the `match=` string in that test
  (not this lane's file) or reword the message in `_validation.py` (which
  *is* this lane's file) — the latter is the smaller change and the message
  is not load-bearing anywhere else.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** fit on `(DataFrame, Series)` with a
  shared index (passes) and with `y` reindexed (must raise). Then re-run
  `python/tests/test_validation.py` and settle the `match=` string above.

---

### P16 — depth budget and the cancellation seam

- **Target file / symbols:** `src/mojoboost/tree.mojo` — `grow_tree`'s
  frontier loop, at the existing
  `if params.max_depth > 0 and depth >= params.max_depth:` guard.
- **Signatures called:** `check_depth_budget(depth: Int, max_depth: Int) raises`,
  `CancelToken.check(where: String) raises`
- **Patch:** keep the existing guard, which is a *growth decision* (stop
  splitting) rather than a validation. Add `check_depth_budget(depth, params.max_depth)`
  only where an absolute ceiling is wanted, which is a design choice.
  Separately, thread an optional `CancelToken` through `grow_tree` and call
  `token.check("growing tree N, leaf L")` once per frontier iteration.
- **State flow:** the token is a new parameter with a
  `CancelToken.live()` default, so no existing call site changes.
- **Errors:** a cancelled run raises `Error` naming where it noticed, rather
  than returning a partial tree.
- **Ownership:** `tree.mojo` is unowned.
- **Fallback:** **skip this patch by default.** It is the only one that adds
  a parameter to a hot signature, and the callback protocol already covers
  cancellation between rounds. `CancelToken` exists for the case that is not
  covered (one long tree, one long prediction sweep), and adopting it is a
  product decision rather than a defect fix.
- **Serialization effect:** none.
- **Public API effect:** `grow_tree` gains a defaulted parameter.
- **Dependency:** none.
- **Minimal later validation (UNRUN):** grow one tree with a live token
  (unchanged behavior) and one with a cancelled token (raises).

---

### P17 — teardown balance

- **Target file / symbol:** `src/mojoboost/gpu_runtime.mojo` —
  `GpuSession.close`, after `self.pool.release_all()` and
  `self.residency.clear()`.
- **Signature called:** `check_cleanup_balanced(acquired: Int, released: Int, kind: String) raises`
- **Patch:** `close` already orders teardown correctly (drain, then release,
  then close) and this does not replace that. Add, if `PoolLedger` and
  `StagingRing` expose the two counters:

```mojo
        check_cleanup_balanced(
            self.pool.n_acquired, self.pool.n_released, "device pool buffer"
        )
```

  This lane did not read `PoolLedger`'s fields, so **confirm the counter
  names before applying**; if they do not exist, this patch is a request to
  add them rather than a patch.
- **State flow:** none.
- **Errors:** a leak or a double release at teardown becomes an error naming
  the count, rather than the next run's allocation failure.
- **Ownership:** `gpu_runtime.mojo` is unowned.
- **Fallback:** skip. Purely diagnostic.
- **Serialization effect:** none.
- **Public API effect:** `GpuSession.close` can now raise where it
  previously only raised from `_move_to`. It is already `raises`.
- **Dependency:** counters must exist.
- **Minimal later validation (UNRUN):** one GPU session opened and closed
  cleanly.

---

## 4. Open questions this lane leaves

1. **Who runs `check_features_finite`.** numpy already scans the matrix on
   the Python path, so running it natively as well doubles an O(cells) pass
   for the common caller. The paths that have no scan at all are the C ABI,
   the CLI, and native callers. Options: always, never, or gated on a
   `validate_input` parameter. P10 does not choose.

2. **Exception type across the bridge** (see P14's public-API note).
   Recommendation: catch and re-raise as `ValueError` so the estimators keep
   their current contract.

3. **Message-text assertions in the test tree.** Nearly every patch changes
   error wording to add the offending index and value.
   `python/tests/test_validation.py` (179 lines, `8913489`) is where that
   lands hardest: it pins the Python-side numeric domain with
   `pytest.raises(ValueError, match=...)`. This lane **read** it statically
   and did **not** run it, and did **not** edit it — it is outside this
   lane's ownership. Eight expectations are affected, enumerated with their
   native replacement wording in the tables under **P14** (seven, from the
   numeric-domain removal) and **P15** (one, the feature-count message).
   Four of the eight fail on the `match=` string alone; all seven under P14
   also depend on how open question 2 is resolved, because they expect
   `ValueError` and a native raise arrives as `RuntimeError` unless the
   bridge re-raises. Resolve question 2 first, then update the eight in a
   single pass. Other test files were not read; expect more of the same
   wherever a message is asserted.

4. **`ranking.MAX_RELEVANCE_LABEL` vs `validation.MAX_RELEVANCE`** are two
   names for `30`. Unify or assert; do not leave both free.

5. **`check_classes_present` is stricter than LightGBM**, which accepts a
   class with no rows. It is in `check_multiclass_inputs` but nothing calls
   that composite yet, so adopting P1's trainer-side equivalent is where the
   decision actually lands.
