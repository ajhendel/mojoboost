# The validation contract

Which layer checks what, what each check promises, and where a rule may not
be written twice.

## Status of this document

`src/mojotrees/validation.mojo` and `python/mojotrees/_validation.py` both
exist and hold the checks described below. What does not yet exist is the
set of call sites: the modules that own the current fragments have not been
edited, so at the time of writing every rule below is stated once in the
validation layer and still enforced from its old home as well. The call-site
edits that would remove the duplication are each mechanical and are still
owed.

Read the tables below as the contract that lands when those edits do. Where
a rule is currently enforced somewhere else, the table says so.

## 1. The two layers, and why the line falls there

There are two validation layers in mojotrees and they are divided by *what
a check needs to know*, not by which language it is written in.

**Structure** is a property of a Python object. Is this two-dimensional. Does
it have as many rows as the label claims. Are the columns named with strings.
Is the scipy matrix canonical. Can this dtype become float64 at all. None of
that survives the crossing into Mojo, because what crosses is an address and
a length, so it has to be settled on the Python side, in
`python/mojotrees/_validation.py`.

**Domain** is a property of the numbers. Finite. Nonnegative. In range.
Whole. Summing to something positive. All of it survives the crossing
intact, so it is settled once, natively, in `src/mojotrees/validation.mojo`,
next to the loops that depend on it.

The practical test: `_validation.py` never calls `np.isfinite`, `np.isinf`,
or `np.isnan`, never compares a value against a numeric bound, and never
sums anything. A check that wants to do one of those belongs natively.

### 1.1 Why not check numbers in Python as well

Because a rule implemented twice is a rule with two answers, and the Python
copy is the one that drifts. It is written against numpy's vocabulary rather
than against the loop that will read the values, so it diverges under exactly
the inputs nobody tests.

The concrete case that motivated this: the Python layer rejects a
`sample_weight` that is all zeros; the native layer rejects one whose sum is
not positive. Those are the same rule until `[1e-320, -1e-320]` arrives, at
which point one accepts and the other does not, and which error a caller sees
depends on whether they came through the estimators or through
`mojotrees.Dataset`.

### 1.2 The one deliberate overlap

Length agreement is checked on both sides. It is structure by the definition
above, and it is also the one mismatch that reads past the end of an
allocation rather than producing a bad number, so the native layer refuses to
take it on trust. `_validation.check_length` and
`validation.check_required_length` state the same rule in the same words,
and that duplication is intentional and documented here so it is not
"cleaned up" later.

## 2. What `validation.mojo` owns

The module imports nothing from the package and takes primitives, never
mojotrees structs. That is what lets `boosting`, `trainset`, `serialize`,
`sparse`, `params`, `callback`, and the bindings all call into it without
closing an import cycle. A caller unpacks its struct at the call site.

### 2.1 Shape and emptiness

| Check | Rule |
|---|---|
| `check_shape` | at least one row and one feature, both under their ceilings, product allocatable |
| `check_dense_matrix` | a column-major buffer holds exactly `n_rows * n_features` values |
| `check_column_length` | an optional per-row column is absent or has one entry per row |
| `check_required_length` | the same, with absence rejected |
| `check_row_index`, `check_feature_index` | a single index is in range |
| `check_ascending_rows` | a row selection is nonempty, in range, and strictly ascending |

Emptiness is rejected rather than accommodated. A zero-row matrix has no
quantiles to fit and a zero-feature matrix has no split to find; both reach
the histogram builders as loops that run zero times and return a model that
predicts the base score without saying why.

`check_ascending_rows` requires ascending order because a CSC column stores
its row indices ascending. A selection that reordered or repeated rows would
either violate that invariant or need a per-column sort to repair it.

### 2.2 Numeric domains

Three domains, kept apart because the rules genuinely differ.

| Domain | NaN | Infinity | Sign | Total |
|---|---|---|---|---|
| feature values | allowed, means missing | rejected | any | n/a |
| labels, scores, init scores | rejected | rejected | any | n/a |
| sample weights | rejected | rejected | nonnegative | must be positive |
| hessians | rejected | rejected | nonnegative | returned, not judged |

Collapsing these into one "must be finite" helper is what produced the
earlier drift, where a feature matrix and a label vector were checked by the
same predicate and one of the two was wrong.

- `check_features_finite` allows NaN because NaN is mojotrees's missing
  marker: the binner excludes it from the quantiles and reserves a bin for it.
  An infinity is neither a value nor a marker, cannot sit between two quantile
  edges, and is the reason `binning._avoid_inf` exists.
- `check_weights` returns the validated total, so the denominator every
  weighted statistic needs is summed once, in one accumulation order.
- `check_gradient_pair` rejects a negative hessian because a leaf value is
  `-G / (H + lambda)`; a negative hessian can drive that denominator through
  zero and produce a leaf of arbitrary magnitude from a well-behaved gradient.
  It does **not** require the hessian total to be positive: a converged
  custom objective can legitimately return all-zero curvature for a round,
  and the right answer to that is a root-only tree rather than a raise. A
  caller whose next step divides by the total says so with
  `check_positive_hessian_total`. This is deliberate asymmetry with
  `check_weights`, which does require a positive sum, because an all-zero
  weight vector is a statement about the data rather than about a round.

### 2.3 Labels, classes, and ranking groups

| Check | Rule |
|---|---|
| `check_labels_finite` | one finite label per row |
| `check_class_count` | at least 2, at most `MAX_CLASSES` |
| `check_class_codes` | whole numbers in `[0, n_classes)`, returned as `List[Int]` |
| `check_classes_present` | every class has at least one row |
| `check_relevance_labels` | whole numbers in `[0, MAX_RELEVANCE]` |
| `check_group_counts` | nonempty, every count positive, total exactly `n_rows` |
| `check_group_boundaries` | the same contract as boundaries rather than counts |

Fractional class codes are rejected rather than truncated: truncation moves a
row into a neighboring class, and a caller who wrote `1.5` meant something
this layer cannot guess.

`check_classes_present` is stricter than LightGBM, which accepts a class with
no rows. Such a class has no gradient anywhere, so its trees are all single
leaves at the base score and the model reports a class it cannot predict. The
usual cause is a label encoding that dropped a level, which is worth naming.

`check_group_counts` accumulates through `checked_add`, so a hostile array of
large counts is reported as an overflow at the entry that caused it rather
than wrapping into a total that happens to match `n_rows`.

### 2.4 Compressed sparse structure

`check_compressed` is one function for both orientations, with `check_csc`
and `check_csr` as the named wrappers. It catches every failure a malformed
producer can hand over *before any index is dereferenced*: wrong dimensions,
a wrong-length or non-monotone offset array, an offset array that does not
start at 0 or end at nnz, mismatched index and value counts, an out-of-range
index, and unsorted or duplicated indices within one outer slice.

Strictly ascending inner indices is SciPy's canonical form. It rules out
duplicates, which is what lets `lookup` be a binary search and lets a row
subset stay canonical without a re-sort.

The signature takes a stored-entry *count* rather than a values list, so the
same function serves raw `Float64` matrices and binned `UInt8` ones.

### 2.5 Categorical

| Check | Rule |
|---|---|
| `check_categorical_features` | declared indices in range and without repeats |
| `check_category_code` | a whole number in `[0, MAX_CATEGORY_CODE)` |
| `check_max_bin` | in `[2, 256]`, because bin ids are `UInt8` |

A repeated categorical index is rejected rather than deduplicated: two
entries for one feature mean the caller believes they declared something they
did not, and the second entry is the one that would be silently dropped.

`MAX_CATEGORY_CODE` is `1 << 31`, matching `categorical._MAX_CATEGORY`:
LightGBM reads codes through `static_cast<int>`, so a code at or above 2^31
cannot round-trip.

### 2.6 Allocation arithmetic

Index arithmetic in this package is `Int`, which is 64-bit and wraps on
overflow. A product that would wrap has to be caught before it is computed,
because `n_rows * n_features` for a shape a hostile caller chose can land on
a small positive number that then sizes a buffer the rest of the code indexes
past.

| Check | Rule |
|---|---|
| `check_alloc` | an element count is nonnegative and at most `MAX_ALLOC_ELEMS` |
| `checked_mul` | `a * b` by division-first, so the multiply never wraps |
| `checked_add` | `a + b` under the same ceiling |
| `checked_cells` | the named `n_rows * n_features` every dense path derives |

`MAX_ALLOC_ELEMS` is `1 << 46`. At 8 bytes an element that is 512 TiB, so no
real allocation approaches it. What it buys is that the product of any two
checked counts, and any such product plus an offset, stays comfortably inside
Int64.

### 2.7 Model, tree, and node limits

These apply to counts read off disk, before the count becomes an allocation
size. `serialize.mojo` owns whether a token is a well-formed integer. This
layer owns whether the integer is a plausible size, because the grammar is
satisfied by any integer at all, including one that sizes a list larger than
the machine.

| Check | Rule |
|---|---|
| `check_mapper_header` | edge count bounded by `n_features * (n_bins - 1)` |
| `check_tree_count` | at most `MAX_MODEL_TREES` |
| `check_tree_header` | node and leaf counts positive and mutually consistent |
| `check_model_nodes` | the running ensemble total, at most `MAX_MODEL_NODES` |
| `check_tree_topology` | the tree can be walked in bounded time (below) |
| `tree_depth` | deepest leaf, in one forward pass |

Per-tree and per-model node ceilings are both needed. Four million trees of
four nodes each exhaust memory as surely as one tree of sixteen million, and
only the running total catches the first case.

The edge ceiling is derived from the shape rather than fixed, because a
mapper fits at most `n_bins - 1` edges per feature; any larger count
describes a mapper no binning could have produced.

### 2.8 Tree topology, and the gap it closes

This is the one rule in the module that is not currently enforced anywhere.

`serialize._read_trees` bounds-checks a loaded node's children: it requires
`0 <= left[i] < n_nodes` and the same for `right[i]`. It does **not** require
`left[i] > i`.

Every grower in mojotrees appends nodes as it splits them, so a child always
sits at a higher index than its parent. `lgbm_model_io` emits in preorder for
the same reason and says so. Both `predict_raw_row` and the exact
contribution recursion rely on it: the first walks down with a `while` loop
that has no visit budget, and the second recurses once per edge.

A model file whose node 3 names node 1 as its left child passes every check
the reader performs today, and then hangs the first prediction or overflows
the stack.

`check_tree_topology` requires:

- every internal node (`feature >= 0`) has two distinct children, each in
  range and each **strictly after** the parent
- every leaf (`feature < 0`) has no children
- every node except the root is exactly one node's child
- the leaves counted match the header's `n_leaves`

`child > parent` makes the child index a strict decrease in the remaining
node count, so every walk terminates in at most `n_nodes` steps and every
recursion is at most `n_nodes` deep. That is also what makes `tree_depth` a
single forward pass with no traversal and no stack.

### 2.9 Depth, leaves, and iteration counts

| Check | Rule |
|---|---|
| `check_num_leaves` | at least 2, at most `(MAX_TREE_NODES + 1) / 2` |
| `check_max_depth` | at most `MAX_DEPTH_LIMIT`; values `<= 0` mean unlimited |
| `check_depth_budget` | the runtime guard a grower calls before descending |
| `check_iterations` | nonnegative, at most `MAX_ITERATIONS` |
| `check_iteration_range` | a half-open ensemble slice, with `end <= 0` meaning "to the end" |
| `check_early_stopping_rounds` | nonnegative; 0 disables |

`check_num_leaves`'s ceiling is the per-tree node ceiling read back through
`n_nodes = 2 * n_leaves - 1`, so a `num_leaves` that would grow a tree no
loader could read is refused before the run rather than after it.

`check_depth_budget` applies the absolute ceiling even when `max_depth <= 0`.
LightGBM's "unlimited" is unlimited in the sense of "no user bound"; from
inside this layer, an unbounded depth is what an unbounded recursion looks
like.

### 2.10 Hyperparameter ranges

`check_booster_ranges` is the exact set of data-independent range checks that
`params._validate` performs when a parameter string is parsed and that
`callback.check_resettable` performs again when a schedule rewrites a round's
parameters.

Those two copies had already drifted: `_validate` bounds
`feature_fraction_bylevel` and `check_resettable` does not, so a callback can
set a bylevel fraction of `0.0` that the parser would have rejected, and the
run then selects no features at a level.

The signature takes scalars rather than a `BoosterParams`, so the module
stays free of package imports and both callers hand over what they hold.

Objective-dependent parameters (`alpha`, `fair_c`,
`tweedie_variance_power`) are **not** here. Their legal range depends on which
objective is running, which is `boosting._check_objective`'s to know.

### 2.11 Cancellation and cleanup

| Check | Rule |
|---|---|
| `check_control_code` | a callback returned one of `0`, `1`, `2` |
| `CancelToken` | cooperative cancellation at interior points |
| `check_cleanup_balanced` | a teardown released exactly what it acquired |

The training loop in `custom_metric.train_with_callbacks` tests for `ABORT`,
then for `STOP`, and treats everything else as `CONTINUE`. That last clause
is the problem: a callback that returns `7`, or that falls off the end of a
branch and returns whatever a bridge defaulted to, keeps training and says
nothing. A control code is the one value in the callback protocol the loop
cannot sanity check from context.

`validation.CONTROL_CODES` is `3` and the module knows only that the legal
codes are `[0, CONTROL_CODES)`. The names `CONTINUE`, `STOP`, and `ABORT`
stay in `callback.mojo`, which is where they belong; the handoff asks
`callback.mojo` to assert that its three comptimes still equal `0`, `1`, `2`.

`CancelToken` exists because the callback protocol can only stop a run
between rounds. A `STOP` returned from `BEFORE_ITERATION` is not seen again
until the round it declined has finished growing, and for one long tree or a
prediction sweep over a large matrix that is the whole of the work. The token
is cooperative and single-writer: it carries a plain flag, not an atomic, is
set by the thread that owns the loop, and is read by that same loop. A token
copied into parallel workers copies the flag, which is a snapshot rather than
a channel.

`check_cleanup_balanced` does not replace a teardown. `gpu_runtime.close` and
its neighbors own the teardown and order it correctly (drain, then release,
then close). This is the assertion that the teardown finished, expressed as
an error rather than as a leak that surfaces as the next run's allocation
failure. A double release is reported separately from a leak, because a
double release is a use-after-free in waiting and reporting it as
"not balanced" would let it read as the milder of the two.

## 3. What `_validation.py` owns

| Check | Rule |
|---|---|
| `check_shape` | nonempty, under ceiling, product allocatable; returns Python `int`s |
| `check_ndim` | dimensionality, with the 1-D case told how to reshape |
| `check_rectangular` | the numpy-free path: nonempty and equal-length rows |
| `check_length`, `check_optional_length` | one entry per row |
| `check_float64_convertible` | this dtype can become a float64 buffer at all |
| `check_sparse_layout` | canonical scipy matrix in the requested layout, without mutating the caller's |
| `check_sparse_index_width` | the stored-entry count is allocatable on the far side |
| `frame_column_names` | column names when all of them are strings |
| `check_feature_names_match` | a prediction frame's columns match the fitted ones, **in order** |
| `check_frame_index_aligned` | a pandas `y` is indexed like `X` |
| `check_category_tables` | category tables are in range, nonempty, and duplicate-free |
| `check_param_mapping` | a parameter object is a mapping with string keys |
| `check_int_param`, `check_float_param` | a parameter reaches the binding as the right Python type |

Three of these deserve their reasoning stated.

**`check_feature_names_match` checks order, not just membership.** The binned
matrix is positional, so a frame whose columns are the same set in a
different order predicts confidently from the wrong features. Both sides
unnamed is fine; one side named and the other not is fine too, because
fitting on a frame and predicting on an array is reasonable.

**`check_frame_index_aligned` is the one mistake no native check can see.** A
misaligned pandas index produces a perfectly valid buffer of the right length
holding the wrong labels. No metric will flag it.

**`check_int_param` refuses `bool` first and on its own**, even though `bool`
is an `int` subclass and even though `True.is_integer()` is true on Python
3.12 and later. `True` for a count is always a mistake, and reading it as 1 is
how a caller trains a single tree and reports a bug about it. A whole-valued
`float` is accepted, because `100.0` from a grid search or a JSON round trip
means 100.

`describe_domain_owner` maps a domain name to the native check that owns it,
for building an error message that has to explain why the Python layer did
not catch something. It returns `None` for an unknown name rather than
raising: failing to build a message is worse than building a vaguer one.

## 4. What the validation layer deliberately does not own

- **Objective semantics.** That a Poisson target is nonnegative, a gamma
  target positive, a cross-entropy target in `[0, 1]`, or `alpha` in `(0, 1)`
  for quantile. `boosting._check_objective` knows which objective is running;
  duplicating those here would mean two answers to one question.
- **Serialization grammar.** That the token after `mapper` is an integer,
  that a v4 file carries a covers flag, that category codes ascend within a
  feature. `serialize.mojo` owns the format and checks it token by token.
  What the validation layer adds to the load path is only what the grammar
  cannot express: how large a count may be before it is allocated, and
  whether the resulting tree can be walked in bounded time.
- **Device and policy decisions.** `device_policy.mojo` chooses a backend.
  Nothing in the validation layer has an opinion about hardware.
- **Distributed transport framing.** `distributed_transport.mojo` has its own
  frame and session validation with its own status codes, which are a wire
  protocol rather than an input contract.

Where a check needs a value the caller has and the validation layer does not
(an objective code, a fitted mapper, a device), the caller keeps the check.
Where a check needs only numbers, it moves.

## 5. Determinism

Every failure raises with a message that names the rule, the offending index,
and the offending value, in that order. Every scan runs front to back, so the
same bad input always produces the same message, and a message can be
asserted on.

No check warns, clamps, or repairs. LightGBM clamps several parameters into
range; mojotrees rejects them, on the repository's standing rule that a
configuration which would quietly ignore half of itself is reported. A caller
who wants tolerance gets it by not calling.

## 6. Cost

The scans are O(n) in what they look at and allocate nothing, except where a
signature says it returns a list.

`check_features_finite` and `check_columns_usable` are the two that are
O(n_rows * n_features). Both are meant to run once, at binning, not per
round. Everything else is O(rows), O(nnz), or O(nodes).

`check_categorical_features` is O(k^2) in the length of the declaration,
which is the feature count at worst and is checked once per dataset.
