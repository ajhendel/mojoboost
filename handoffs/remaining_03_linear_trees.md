# Remaining parity 03: a sound linear-tree model representation

Owned files: `src/mojoboost/linear_tree.mojo`, `docs/LINEAR_TREES.md`, this
file. Nothing else was edited.

Nothing was built, run, tested, benchmarked, or committed. Every command in
this file is marked **UNRUN**.

## Ownership collision, stated first

`MOJOBOOST_CONNECT_EVERYTHING_PARALLEL_PROMPTS.txt` TASK 17 (alternate
boosting modes) also lists `src/mojoboost/linear_tree.mojo` in its ownership.
While this round ran, that lane created `boosting_dart.mojo`,
`boosting_rf.mojo`, `alternate_boosting.mojo`, and `cegb.mojo`, and committed
some of them (`860b1cf`, `dc21f03`). It did **not** create
`linear_tree.mojo`, and none of its four modules mentions linear trees
(`rg -n linear src/mojoboost/{alternate_boosting,cegb,boosting_dart,boosting_rf}.mojo`
returns nothing). So there is no duplicate implementation to fuse, and this
module is the single authoritative one.

If the TASK 17 owner is still active: this file and `linear_tree.mojo` are
the follow-up they were told to expect. Their `AlternateBoostingParams` and
this module's `LinearParams` are orthogonal (linear leaves are a leaf
representation, not a boosting mode), with exactly one interaction, patch 9
below.

Two other lanes moved under this one while it was written, and the patches
account for both:

- TASK 11 (serialization/inspection) took **format version v4** for split
  gains, a cover presence flag, and feature names. Linear leaves are
  therefore **v5**, not v4. `serialize._VERSION` is `"v4"` as of this
  writing; confirm it before applying patch 5.
- `save_model` / `save_multiclass_model` gained a `feature_names` parameter,
  and `load_model` / `load_multiclass_model` gained a `_read_feature_names`
  call. The patches below are anchored on symbols, not line numbers.

## What was found

| Question | Answer, from static inspection |
| --- | --- |
| Existing linear-tree implementation? | None. No file, no struct, no field. |
| Can `tree.Tree` store coefficients? | **No.** One `Float64` per node in `value`; every other array is routing or cover. |
| Are raw feature values available during growth? | **No.** `grow_tree` is handed a `BinnedMatrix`, which is `UInt8` bins only. `BinMapper.transform` discards the raw matrix. |
| Public rejection in place? | Yes: `tree_parameters_extra.check_extra_option_supported` (`linear_tree`, `linear_lambda`), reached from `params.mojo:636`; `lgbm_model_io.mojo:412,415` refuses `is_linear=1`, `leaf_const`, `leaf_coeff`. |
| Prior handoff position | `handoffs/task12_tree_parameters.md` "Deferred: linear trees" lists five requirements. All five are addressed below; its item 3 ("coefficient arrays on `Tree`") is answered with a sidecar instead, for the reason in `docs/LINEAR_TREES.md`. |

## What was built

`src/mojoboost/linear_tree.mojo`. It is the whole algorithm core and it sits
low in the import graph on purpose: it imports only `binning`, `categorical`,
`monotone`, and `tree`, and nothing imports it yet.

**Why it does not import `boosting`.** Patch 1 makes `boosting` import
`linear_tree` (for `LinearEnsemble`). `src/mojoboost` has no mutual imports
anywhere today, so the reverse edge would be the repository's first cycle.
The four objective codes the compatibility gate needs are therefore mirrored
as `_QUANTILE`, `_L1`, `_LAMBDARANK`, `_MAPE` with
`_objective_renews_leaves`, the same way `model_dump.mojo` mirrors
`categorical._MAX_CATEGORY`. The codes are part of a stable public numbering
(serialized in every model file, crossing the C ABI), so they do not move;
patch 11 asks for a cross-check anyway.

| Area | Symbols |
| --- | --- |
| Representation | `LinearLeaf`, `LinearTree`, `LinearEnsemble`, `linear_leaf_reduces_to_constant` |
| Parameters | `LinearParams` (`.check()`, `.disabled()`, `.is_active()`) |
| Statistics | `LeafStats`, `branch_features`, `eligible_leaf_features`, `accumulate_leaf_stats`, `select_leaf_features` |
| Solve | `LinearSolution`, `_cholesky_in_place`, `_cholesky_solve`, `solve_leaf_coefficients` |
| Fitting | `fit_leaf`, `leaf_row_lists`, `refit_linear_tree`, `refit_linear_ensemble` |
| Prediction | `predict_tree_raw`, `predict_tree_raw_at`, `predict_ensemble_raw`, `predict_multiclass_raw`, `predict_batch_raw`, `resume_raw_scores` |
| Gates | `check_objective_compatible`, `check_monotone_compatible`, `check_continuation_compatible`, `check_linear_tree_public`, `linear_tree_available` |
| Per-tree weights | `LinearEnsemble.scale`, `LinearEnsemble.scale_all` |
| Inspection | `linear_feature_use_counts`, `linear_coefficient_l1`, `linear_fit_summary`, `linear_fit_reason_name` |
| Serialization | `linear_section_text`, `read_linear_section`, `LinearSectionResult`, `linear_model_format_version`, `LINEAR_MODEL_FORMAT_VERSION = 5`, `LINEAR_SECTION_REVISION = 1` |

`docs/LINEAR_TREES.md` is the normative statement: the derivation, the
restrictions and why each one is a restriction rather than a shortcut, the
monotone box-corner condition that would lift the refusal, the wire format,
the LightGBM differences, and three parity-unverified claims.

### Connections completed inside the owned file

- `fit_leaf` -> `eligible_leaf_features` -> `accumulate_leaf_stats` ->
  `select_leaf_features` -> `solve_leaf_coefficients` is one path with no
  alternate route and no second implementation of any step.
- `refit_linear_tree` drives it from a real `Tree` plus a real
  `BinnedMatrix`, routing through `Tree.leaf_index_row` and
  `Tree.leaf_ordinals` rather than reimplementing either.
- Prediction reuses the tree's own routing (`Tree.goes_left`) and never
  copies the sidecar per tree per row; `predict_batch_raw` hoists the
  ordinal tables and the row buffers.
- `resume_raw_scores` is `predict_batch_raw` plus `init_score`, so
  continued training and prediction cannot disagree.
- Serialization reads intercepts back off `Tree.value` rather than storing
  them twice, so the two cannot drift.
- Every failure path records a `LINEAR_FIT_*` reason and
  `linear_fit_summary` aggregates them.
- Nothing was left as scaffolding. There is no `pass`, no `TODO`, no
  placeholder struct, and no function whose body is a raise standing in for
  an algorithm, except `check_linear_tree_public`, which is deliberately
  exactly that.

### Deliberate non-connections

- **`Booster` is not touched.** The sidecar has no home yet; patch 1 gives
  it one.
- **No trainer calls `refit_linear_tree`.** Patch 4.
- **`refit_linear_ensemble` is diagnostic, not training.** Every tree of a
  boosted ensemble was fitted to its own round's residuals, so refitting
  them all from one `grad`/`hess` pair reproduces the trained model only for
  a one-tree ensemble. Its docstring says so. Training builds the sidecar one
  tree at a time inside the boosting loop (patch 4).
- **`params.mojo` still refuses the parameter.** Patch 10, last.

## Ready-to-apply integration patches

Apply in order. Patches 1 through 5 are the minimum that makes a linear
model exist, save, and load. Patches 6 through 8 keep the consumers that
cannot evaluate a linear leaf from lying about one. Patch 9 pairs with
DART's weight folding and patch 11 is a compile-time guard. Patch 10 is the
only one that changes public behaviour and must be last.

---

### Patch 1 -- the sidecar field on `Booster` and `MulticlassBooster`

- **Target file / symbol**: `src/mojoboost/boosting.mojo`, `struct Booster`
  and `struct MulticlassBooster`.
- **Ownership**: boosting/trainer lane.
- **Dependency**: none. Do this first.
- **Change**: add one field and one trailing constructor parameter to each.

```mojo
from .linear_tree import LinearEnsemble          # new import

struct Booster(Copyable, Movable):
    var trees: List[Tree]
    var base_score: Float64
    var learning_rate: Float64
    var objective: Int
    var monotone: MonotoneConstraints
    var linear: LinearEnsemble                   # NEW

    def __init__(
        out self,
        var trees: List[Tree],
        base_score: Float64,
        learning_rate: Float64,
        objective: Int,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
        var linear: LinearEnsemble = LinearEnsemble.inactive(),   # NEW
    ):
        ...
        self.linear = linear^
```

  Identically for `MulticlassBooster`, whose current constructor is
  `(trees, base_scores, n_classes, learning_rate, monotone)`; append
  `var linear: LinearEnsemble = LinearEnsemble.inactive()`.

- **Signature effect**: appended with a default, so every existing positional
  caller compiles unchanged. Confirmed call sites that pass `monotone`
  positionally and would otherwise break:
  `serialize.load_model`, `serialize.load_multiclass_model`,
  `boosting.train`, `boosting.train_multiclass`, `train_gpu`,
  `boosting_sparse`, `objective.train_custom`, `custom_metric`,
  `alternate_boosting`, `distributed`. All keep working.
- **State flow**: the sidecar is built by patch 4 and read by patches 2, 3,
  6, 7, 8.
- **Errors**: none added here.
- **Fallback**: `LinearEnsemble.inactive()` is the empty sidecar and
  `is_active()` is False, so every existing predictor path is byte-identical.
- **Serialization effect**: none by itself; patch 5.
- **Public API effect**: none.
- **Later validation (UNRUN)**:
  `pixi run mojo build src/mojoboost/boosting.mojo`.

---

### Patch 2 -- linear-aware raw scores in `Booster`

- **Target file / symbol**: `src/mojoboost/boosting.mojo`,
  `Booster.predict_raw_row`, `Booster.predict_raw_bins`,
  `Booster.predict_bins_range`, `Booster.predict_raw_bins_range`, and the
  `MulticlassBooster` counterparts.
- **Ownership**: boosting/trainer lane.
- **Dependency**: patch 1.
- **Problem**: every one of them is `s += learning_rate *
  tree.predict_row(data, row)`, which reads `Tree.value` and cannot see a
  linear leaf.
- **Change**: these methods take bins only. A linear leaf needs the raw row
  as well, and the raw row is not recoverable from bins. Two options, and the
  choice belongs to that lane:

  **(a) Refuse, and route linear prediction through the new entry points.**
  Add to each bins-only method, at the top:

```mojo
    if self.linear.is_active():
        raise Error(
            "this model has linear leaves; bin-only prediction cannot"
            " evaluate them. Use linear_tree.predict_ensemble_raw, which"
            " takes the raw row alongside the bins"
        )
```

  This makes every method `raises` that is not already. It is the honest
  minimum and it is what patch 3 assumes.

  **(b) Add raw-row overloads.** Add
  `predict_raw_bins_raw(self, bins, raw_row, rng)` delegating to
  `linear_tree.predict_ensemble_raw(self.trees, self.linear,
  self.base_score, self.learning_rate, bins, raw_row, rng.start, rng.stop)`,
  and keep the bins-only methods refusing as in (a).

- **State flow**: `self.linear` in, one Float64 out.
- **Errors**: the raise above; `predict_ensemble_raw` itself does not raise.
- **Fallback**: an inactive sidecar takes the existing path untouched.
- **Serialization effect**: none.
- **Public API effect**: none until patch 10; no public path can produce an
  active sidecar before then.
- **Later validation (UNRUN)**: a focused test that a constant-leaf model
  predicts bit-identically before and after.

---

### Patch 3 -- raw values reach `Model.predict*`

- **Target file / symbol**: `src/mojoboost/model.mojo`, `Model.predict`,
  `Model.predict_raw`, `Model.predict_range`, `Model.predict_raw_range`,
  `Model.predict_batch`, and the `MulticlassModel` counterparts.
- **Ownership**: model lane.
- **Dependency**: patches 1 and 2.
- **Note**: this is the *easy* half. `Model.predict(row)` already holds the
  raw row and throws it away after `self.mapper.bin_row(row)`; keeping it is
  a one-line change:

```mojo
    def predict_raw(self, row: List[Float64]) raises -> Float64:
        if self.booster.linear.is_active():
            return linear_tree.predict_ensemble_raw(
                self.booster.trees,
                self.booster.linear,
                self.booster.base_score,
                self.booster.learning_rate,
                self.mapper.bin_row(row),
                row,
            )
        return self.booster.predict_raw_bins(self.mapper.bin_row(row))
```

  `predict_batch` likewise already holds `features` (the raw column-major
  matrix) and bins it into `data`; pass both to
  `linear_tree.predict_batch_raw(trees, linear, base_score, learning_rate,
  data, features, rng.start, rng.stop)`.
- **State flow**: raw matrix in, scores out; `predict_batch_raw` validates
  `len(raw) == n_rows * n_features` and raises with both numbers.
- **Errors**: as above.
- **Fallback**: inactive sidecar keeps the current code path.
- **Serialization effect**: none.
- **Public API effect**: none until patch 10.
- **Later validation (UNRUN)**: a focused test that `predict` and
  `predict_batch` agree on the same rows for an active sidecar.

---

### Patch 4 -- the trainer builds the sidecar

- **Target file / symbol**: `src/mojoboost/boosting.mojo`, `train`,
  `train_multiclass`, `train_more`, `train_multiclass_more`; and
  `BoosterParams`.
- **Ownership**: boosting/trainer lane.
- **Dependency**: patch 1.
- **Change**, in four parts.

  **4a. Carry the parameters.** Add to `BoosterParams`:

```mojo
    var linear: LinearParams          # default LinearParams.disabled()
```

  Appended with a default, so `BoosterParams(100, 0.1, TreeParams.default())`
  keeps working.

  **4b. Carry the raw matrix.** `train` takes a `BinnedMatrix`; a linear fit
  needs the raw one. Add a trailing parameter rather than changing the type:

```mojo
def train(
    data: BinnedMatrix,
    ...,
    raw: List[Float64] = [],          # NEW, column-major, len n_rows*n_features
) raises -> Booster:
```

  and, once per call, before the round loop:

```mojo
    if params.linear.is_active():
        params.linear.check()
        check_objective_compatible(objective)
        check_monotone_compatible(params.tree.monotone)
        if len(raw) != data.n_rows * data.n_features:
            raise Error(
                "linear_tree needs the raw feature matrix: pass `raw` as"
                " the column-major matrix `data` was binned from"
            )
```

  Rejecting all three up front, before the first histogram, matches
  `grow_tree`'s own placement of `params.extra.check`.

  **4c. Fit each tree as it is grown.** In the round loop, immediately after
  the existing `var tree = grow_tree(...)` and *before* the
  `raw[r] += learning_rate * tree.predict_row(data, r)` score update:

```mojo
    var entry = LinearTree.constant()
    if params.linear.is_active():
        entry = refit_linear_tree(
            tree, data, raw, grad, hess, params.linear, bag
        )
    linear_entries.append(entry^)
```

  where `bag` is whatever row list that round grew on (empty for the full
  set), `grad`/`hess` are the round's arrays, and `linear_entries` is a
  `List[LinearTree]` accumulated beside `trees`.

  **Critical ordering**: the score update that follows must go through the
  linear leaf, not `tree.predict_row`:

```mojo
    for r in range(data.n_rows):
        scores[r] += learning_rate * predict_tree_raw_at(
            tree, one_tree_ensemble, 0, ordinals, bins_of_row_r, raw_row_r
        )
```

  or, simpler and what is recommended, recompute the round's contribution
  with `linear_tree.predict_batch_raw` over the single new tree. Getting this
  wrong is the single most damaging way to misapply linear trees: every later
  tree would be fitted to a residual the model does not actually leave
  behind, and the error compounds silently across rounds.

  Every place `trees.pop()` is called (early stopping rollback:
  `boosting.mojo:1308,1586,1880,1894`, `boosting_sparse.mojo:399,504`,
  `custom_metric.mojo:1164`, `boosting_dart.mojo:726`) must pop
  `linear_entries` in the same statement, or the sidecar and the trees fall
  out of index alignment. `LinearEnsemble.check_against` catches that on
  load, but not before the model has already predicted wrongly in memory.

  **4d. Attach it.** At the `Booster(...)` construction:

```mojo
    var linear = LinearEnsemble.inactive()
    if params.linear.is_active():
        linear = LinearEnsemble(linear_entries^, data.n_features)
    return Booster(trees^, base_score, learning_rate, objective,
                   monotone^, linear^)
```

  **4e. Continued training.** In `train_more`, before appending anything:

```mojo
    check_continuation_compatible(
        booster.linear, len(booster.trees), params.linear, data.n_features
    )
```

  and replace the resume loop
  (`for r ... raw[r] += booster.learning_rate * booster.trees[i].predict_row(...)`)
  with

```mojo
    var scores = resume_raw_scores(
        booster.trees, booster.linear, booster.base_score,
        booster.learning_rate, data, raw_features, init_score
    )
```

  `train_more` also needs the raw matrix as a new trailing parameter, on the
  same terms as 4b.

- **State flow**: `LinearParams` in; one `LinearTree` per grown tree
  accumulated; one `LinearEnsemble` onto the `Booster`.
- **Errors**: `LinearParams.check`, `check_objective_compatible`,
  `check_monotone_compatible`, `check_continuation_compatible`, and the raw
  matrix shape check, all raising with the reason named.
- **Fallback**: `params.linear.is_active()` is False by default, so every
  branch above is skipped and the trainer is unchanged.
- **Serialization effect**: the `Booster` now carries something patch 5 must
  write.
- **Public API effect**: none until patch 10.
- **Later validation (UNRUN)**: a focused test that a two-round linear fit
  followed by `train_more` for two more rounds equals a four-round fit.

---

### Patch 5 -- the v5 `linear` section

- **Target file / symbol**: `src/mojoboost/serialize.mojo`, `_VERSION`,
  `_read_version`, `save_model`, `save_multiclass_model`, `load_model`,
  `load_multiclass_model`.
- **Ownership**: TASK 11 (serialization/inspection). **That lane is
  active**; `_VERSION` moved from `"v3"` to `"v4"` while this was written.
  Re-read it before applying.
- **Dependency**: patches 1 and 4.
- **Change**:

  **5a.** `comptime _VERSION = "v5"`, and in `_read_version` add
  `if token == "v5": return 5`.

  **5b.** In both savers, after `_write_trees(out, model.booster.trees)`:

```mojo
    out += linear_section_text(model.booster.linear)
```

  `linear_section_text` returns `""` for an inactive sidecar, so a
  constant-leaf model's bytes are unchanged and the call needs no guard.

  **5c.** In both loaders, after `_read_trees` and before the `Booster`
  construction:

```mojo
    var lin = read_linear_section(r.tokens, r.pos, trees, mapper.n_features)
    r.pos = lin.next_pos
```

  then pass `lin.linear^` as the `Booster`/`MulticlassBooster` constructor's
  new last argument. `_TokenReader.tokens` and `.pos` are module-private but
  the call site is in the same module, so no new accessor is needed. If that
  lane prefers not to expose the cursor, the alternative is a
  `_TokenReader` method `def take_linear(mut self, trees, n_features)
  raises -> LinearEnsemble` wrapping the same two lines.

  **5d.** Version history entry in the module docstring:

  > - v5: adds an optional `linear` section after the trees, holding the
  >   per-leaf coefficients of a linear-tree model (see
  >   `linear_tree.mojo` and `docs/LINEAR_TREES.md`). It is written only for
  >   a model that has them, so a v5 file for a constant-leaf model is byte
  >   for byte what v4 wrote, and a file without the section loads as a
  >   constant-leaf model. Intercepts are not stored: a linear leaf's
  >   intercept is the `Tree.value` the tree section already carries.

- **State flow**: `Booster.linear` out on save; validated
  `LinearEnsemble` in on load.
- **Errors**: `read_linear_section` raises on an unknown revision, a tree
  count or leaf count that does not match the trees, an out-of-range or
  repeated leaf ordinal, an out-of-range feature index, a non-ascending
  feature list, and a non-finite stored float. It then calls
  `LinearEnsemble.check_against` before returning.
- **Fallback**: absent section -> `LinearEnsemble.inactive()`, and v1
  through v4 files load exactly as they do now.
- **Serialization effect**: this *is* the serialization effect. The bump is
  conditional in intent (`linear_model_format_version`) but `_VERSION` is a
  single constant, so the practical rule is: v5 files that carry no `linear`
  section are readable by a v4 reader only if that reader is told to accept
  the token. If that matters, keep `_VERSION` at `"v4"` and write `"v5"`
  only when `model.booster.linear.is_active()`; that is what
  `linear_model_format_version(linear, 4)` returns and why it exists.
- **Public API effect**: `model_file_kind` and `file_kind` are unaffected;
  they read the magic and the kind token, not the section.
- **Later validation (UNRUN)**: a focused round-trip test on a two-tree
  sidecar; a focused test that a constant-leaf model's saved bytes are
  unchanged.

---

### Patch 6 -- `contrib.mojo` refuses linear leaves

- **Target file / symbol**: `src/mojoboost/contrib.mojo`,
  `predict_contrib_bins`, `predict_contrib_bins_multiclass`,
  `ContribExplainer.__init__`, `tree_expected_value`.
- **Ownership**: contrib lane.
- **Dependency**: patch 1.
- **Why**: exact TreeSHAP's efficiency property ("contributions plus the
  expected value sum to the raw score") is proved from `v(all features)`
  being *the leaf's value*. With a linear leaf, `v(all features)` is a
  function of the row, and `v(no features)` is a cover-weighted mean of
  functions. The recursion returns numbers that no longer sum to the score.
  Silently returning them would break the one property the module's own
  tests assert.
- **Change**, at the top of each entry point:

```mojo
    if booster.linear.is_active():
        raise Error(
            "feature contributions are not available for a model with"
            " linear leaves: exact attribution conditions on each leaf's"
            " constant value, and a linear leaf's value depends on the row."
            " Ignoring the coefficients would explain the model's constant"
            " fallback (the leaf at its own training centroid), not the"
            " model. See docs/LINEAR_TREES.md"
        )
```

- **State flow**: reads `booster.linear` only.
- **Fallback**: none; refusal is the correct behaviour. The message names
  what the degraded alternative would be, which
  `linear_tree.linear_leaf_reduces_to_constant` makes checkable if that lane
  later wants to offer it behind an explicit flag.
- **Serialization / public API effect**: `pred_contrib=True` raises for such
  a model instead of returning wrong numbers.
- **Later validation (UNRUN)**: a focused test that `predict_contrib` raises
  on an active sidecar.

---

### Patch 7 -- `gpu_predict.mojo` refuses linear leaves

- **Target file / symbol**: `src/mojoboost/gpu_predict.mojo`,
  `flatten_booster`, `flatten_multiclass` (and therefore `predict_gpu`,
  `predict_raw_gpu`, `predict_proba_gpu`, `predict_raw_multiclass_gpu`,
  `GpuPredictor.upload_ensemble`).
- **Ownership**: GPU lane.
- **Dependency**: patch 1.
- **Why**: the device kernels walk a flattened tree and read a leaf value
  from a Float32 plane. They have no raw feature matrix on the device and no
  per-leaf coefficient plane. Flattening an active sidecar's model would
  upload the constant fallback and predict a different model than the CPU
  does, breaking `docs/GPU_VALIDATION.md`'s CPU/GPU agreement contract.
- **Change**, in `flatten_booster` and `flatten_multiclass` (one place each,
  so every caller is covered):

```mojo
    if booster.linear.is_active():
        raise Error(
            "GPU prediction is not available for a model with linear"
            " leaves: the device kernels read one Float32 leaf value and"
            " have neither the raw feature matrix nor a coefficient plane."
            " Predict on the CPU (device=CPU_DEVICE)"
        )
```

- **State flow**: reads `booster.linear` only.
- **Fallback**: `Model.predict_batch(device=AUTO_DEVICE)` should *not*
  silently fall back to the CPU here; `resolve_device` is the place a
  documented fallback belongs, and adding one is that lane's call. Raising
  from the flatten is the safe default.
- **Serialization effect**: none. GPU-trained models stay device-independent
  and CPU-loadable; the sidecar is host data.
- **Public API effect**: `device="gpu"` raises for such a model.
- **Later validation (UNRUN)**: a focused test that `predict_gpu` raises on
  an active sidecar. Needs an accelerator.

---

### Patch 8 -- `model_dump.mojo` and `lgbm_model_io.mojo`

- **Target files / symbols**: `src/mojoboost/model_dump.mojo`,
  `struct DumpNode`, `build_dump`, `build_multiclass_dump`;
  `src/mojoboost/lgbm_model_io.mojo`, the writer.
- **Ownership**: TASK 11 (inspection) and the LightGBM-interop lane.
- **Dependency**: patch 1.
- **8a, dump.** Add three optional fields to `DumpNode`, populated for a leaf
  and empty otherwise:

```mojo
    var linear_features: List[Int]      # [] for a constant leaf
    var linear_coefficients: List[Float64]
    var linear_centers: List[Float64]
```

  and one capability flag on `ModelDump`, beside `has_split_gain` and
  `has_node_count`:

```mojo
    var has_linear_leaves: Bool         # model.booster.linear.is_active()
```

  A consumer must branch on the flag before reading `value` as the leaf's
  output. `inspection.dump_json` renders the three as JSON arrays, `null` for
  a non-finite entry as it already does elsewhere; `DUMP_FORMAT_VERSION` does
  **not** bump, because adding optional keys does not bump it per
  `docs/MODEL_INSPECTION_SCHEMA.md`. `MODEL_FORMAT_VERSION` becomes 5 with
  patch 5.

  `linear_tree.linear_feature_use_counts` and
  `linear_tree.linear_coefficient_l1` are ready to feed a
  `feature_importance(importance_type="linear_use" | "linear_l1")` if that
  lane wants one; they are data, not a ranking, and the docstrings say why.

- **8b, LightGBM export.** The writer must refuse:

```mojo
    if booster.linear.is_active():
        raise Error(
            "a model with linear leaves cannot be written as a LightGBM"
            " model: mojoboost's leaf function is centred"
            " (intercept + sum coef * (x - centre)) and LightGBM's is not,"
            " so leaf_const would have to be reinterpreted against a"
            " centroid the format does not carry. See docs/LINEAR_TREES.md"
        )
```

  The import side already refuses (`lgbm_model_io.mojo:412,415`); leave it.
  The arithmetic that would make either direction work is in
  `docs/LINEAR_TREES.md` "Interoperability"; it is a piece of work, not a
  wiring change, because it moves the missing-value rule too.

- **Fallback**: none; both refuse.
- **Later validation (UNRUN)**: a focused test that a dump of an active
  sidecar reports `has_linear_leaves: true` and carries the arrays.

---

### Patch 9 -- DART weight folding

- **Target file / symbol**: `src/mojoboost/alternate_boosting.mojo`,
  `fold_weights_into_trees`, and its callers in `_dart_rounds` /
  `train_dart` / `train_dart_more`.
- **Ownership**: TASK 17 (alternate boosting).
- **Dependency**: patches 1 and 4.
- **Why**: `fold_weights_into_trees` multiplies `Tree.value` by each tree's
  drop weight. A linear leaf's intercept is a copy of that value, so folding
  the trees without the sidecar leaves a leaf whose affine function no longer
  passes through the value its tree carries. The two halves would describe
  different models, and `linear_leaf_reduces_to_constant` would start
  returning False.
- **Change**: wherever `fold_weights_into_trees(trees, weights)` is called,
  call its other half in the same statement:

```mojo
    fold_weights_into_trees(trees, weights)
    linear.scale_all(weights)          # LinearEnsemble.scale_all
```

  `scale_all` is a no-op on an inactive sidecar and skips a weight of exactly
  1.0, matching `fold_weights_into_trees`'s own rule so an untouched tree's
  leaves come back bit-identical.

  If that lane later moves the folding onto `Tree` as `Tree.shrinkage` (its
  handoff asks for that), the paired call moves with it.

- **Errors**: `scale_all` raises when the weight vector and the sidecar have
  different lengths, the same shape of check `fold_weights_into_trees`
  already makes.
- **Serialization effect**: the scaled coefficients are what gets written.
- **Later validation (UNRUN)**: a focused test that a DART model with linear
  leaves predicts the same before and after folding.

---

### Patch 10 -- accept the parameter, last

- **Target files / symbols**:
  `src/mojoboost/tree_parameters_extra.mojo`,
  `check_extra_option_supported`; `src/mojoboost/params.mojo`;
  `bindings/_mojoboost.mojo`; `python/mojoboost/basic.py` and
  `python/mojoboost/_sklearn.py`; `capi/mojoboost_capi.mojo` and
  `capi/mojoboost.h`; `docs/LIGHTGBM_PARITY.md`;
  `src/mojoboost/linear_tree.mojo` (`LINEAR_TREE_PUBLIC`).
- **Ownership**: parameters lane, bindings lanes 06/07/14, parity lane 19.
- **Dependency**: **all of patches 1 through 9**. Do not apply this one
  early. A publicly settable `linear_tree` with any of the above missing
  produces a model that saves wrong, explains wrong, or predicts differently
  on the GPU, which is worse than a clear refusal.
- **Change**:
  1. Remove the `linear_tree` and `linear_lambda` branches from
     `check_extra_option_supported`.
  2. Map `linear_tree` -> `LinearParams.enabled` and `linear_lambda` ->
     `LinearParams.linear_lambda` in `params.mojo`, and reject the
     combination with `monotone_constraints` and with the leaf-renewing
     objectives there too, so a bad configuration is named at parameter
     parse time and again at train time.
  3. The four mojoboost-only knobs (`max_leaf_features`,
     `min_data_per_linear_feature`, `ridge_eps`, `max_linear_deviation`) are
     extensions; expose them under mojoboost-prefixed names or not at all,
     but do not give them LightGBM names.
  4. Python: `linear_tree` and `linear_lambda` are keyword params on the
     estimator and pass straight through. Python stays a thin adapter; no
     fitting, prediction, or coefficient handling in Python.
  5. C ABI: the sidecar is model state, not a new call. Existing predict
     entry points need the raw row; that is a signature change and belongs
     with patch 3's decision.
  6. Set `comptime LINEAR_TREE_PUBLIC = True` in `linear_tree.mojo` so
     `check_linear_tree_public` stops raising. **This constant is the last
     line of the whole sequence.**
  7. `docs/LIGHTGBM_PARITY.md`: move `linear_tree` and `linear_lambda` from
     `unsupported`/`deferred` to `partial`, not `supported`, with the notes
     naming the split-gain difference and the three parity-unverified claims
     in `docs/LINEAR_TREES.md`. Run `python3 tools/check_parity.py`. **UNRUN.**
- **Public API effect**: this is the entire public API effect of the
  feature.

---

---

### Patch 11 -- keep the mirrored objective codes honest

- **Target file / symbol**: `src/mojoboost/boosting.mojo`, beside
  `objective_renews_leaves`.
- **Ownership**: boosting lane.
- **Dependency**: patch 1.
- **Why**: `linear_tree.mojo` cannot import `boosting` (patch 1 makes the
  edge run the other way), so it mirrors `QUANTILE=4`, `L1=5`,
  `LAMBDARANK=7`, and `MAPE=10` locally. A silent divergence would make the
  leaf-compatibility gate refuse the wrong objective.
- **Change**: one compile-time assertion in a function body that both
  modules' constants agree, e.g. inside `train`:

```mojo
    comptime assert QUANTILE == 4 and L1 == 5 and MAPE == 10, (
        "linear_tree.mojo mirrors these codes; move both together"
    )
```

  plus the same for `ranking.LAMBDARANK == 7` wherever that module can see
  it. The alternative, and the better long-term fix, is to move the
  objective-code constants into a leaf module both can import.
- **Fallback**: none needed; this is a check, not behaviour.
- **Later validation (UNRUN)**: it is a compile-time assertion; the build is
  the check.

---

## Remaining disconnections after all eleven patches

- **Split gains stay linear-unaware.** LightGBM scores candidates under the
  per-leaf regression; mojoboost grows the constant-leaf tree and refits. The
  trees differ in shape. `accumulate_leaf_stats` and
  `solve_leaf_coefficients` are the reusable pieces; the cost is one solve
  per candidate. Documented as an intentional difference, not a gap to hide.
- **Monotone constraints stay refused.** The box-corner condition that would
  lift it is written out in `docs/LINEAR_TREES.md`; it needs bin edges at the
  leaf and per-feature box bounds on the frontier.
- **GPU training and GPU prediction stay CPU-only for linear leaves.**
- **LightGBM import and export stay refused, both directions.**
- **Sparse training** (`boosting_sparse`, `tree_sparse`) has no raw-value
  path at all; a linear fit there needs `SparseBinnedMatrix`'s raw
  counterpart, which does not exist.
- **`refit_linear_ensemble` has no production caller** and should not get
  one; it is diagnostic.

## Risks

1. **The score-update ordering in patch 4c** is the one place a mistake is
   silent and compounding. Every later tree would be fitted to the wrong
   residual.
2. **`trees.pop()` without `linear_entries.pop()`** silently misaligns the
   sidecar. Eight call sites are listed in patch 4c.
3. **`_VERSION` is contested.** It moved v3 -> v4 during this round. Confirm
   before patch 5.
4. **`predict_tree_raw`'s ordinal scan is O(nodes)** per tree per row. Batch
   prediction hoists it (`predict_tree_raw_at`); the per-row path does not,
   matching `Tree.leaf_ordinal_bins`.
5. **Nothing here has been compiled.** The module follows the repository's
   Mojo idioms (`def`, `comptime`, `ref` bindings, `List` with `capacity=`,
   `var`-transfer returns) and avoids constructs not already used in
   `src/mojoboost` (notably indexed `List.pop`, which is written out as
   `_drop_at`), but a first build will find something.

## Smallest later checks, all UNRUN

```
pixi run mojo build src/mojoboost/linear_tree.mojo          # UNRUN
```

Then, one focused test each, in this order:

1. `linear_lambda` large reproduces `Tree.value` to the last bit. **UNRUN.**
2. `linear_section_text` -> `read_linear_section` round-trips a two-tree
   sidecar exactly. **UNRUN.**
3. `check_monotone_compatible`, `check_objective_compatible`, and
   `check_continuation_compatible` each raise on the case they name.
   **UNRUN.**
4. A rank-deficient leaf (two identical columns) falls back rather than
   producing an infinity. **UNRUN.**
5. `predict_ensemble_raw` and `predict_batch_raw` agree row for row.
   **UNRUN.**
6. `python3 tools/check_parity.py` after any parity row moves. **UNRUN.**

One focused test per change. Not the full suite, and no build or benchmark
loops.
