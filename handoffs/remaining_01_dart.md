# Task 01 handoff: DART boosting core

**Status: the algorithm core is written. Nothing is public, nothing is
integrated, nothing has been run.**

This lane compiled no Mojo, wrote no test, ran no test, ran no benchmark, and
made no commit. Every claim below is about code as written, not code as
observed.

> Note: `src/mojoboost/boosting_dart.mojo` is nonetheless tracked, committed by
> a concurrent lane's sweep commit `860b1cf` ("Integrate training and
> interoperability subsystems") minutes after it was written. This lane did not
> commit it. Its content at that commit is byte-identical to what is described
> here, and the working tree matches HEAD. Being committed is not evidence that
> it compiles; see section 7.

---

## 0. Ownership conflict, and how it was resolved

**Read this first. This lane collided with CONNECT_EVERYTHING Task 17 and the
collision resolved in Task 17's favor.**

The Task 01 prompt says: *"If CONNECT_EVERYTHING Task 17 owns
boosting_dart.mojo now, do not run this as a second session; give it to that
owner as a follow-up."*

Timeline, from file mtimes in the shared checkout:

| Time | Event |
|---|---|
| 12:35 | Task 01 checks ownership. No `boosting_dart.mojo`, no `boosting_rf.mojo`, no `linear_tree.mojo`, no `cegb.mojo`, no `alternate_boosting.mojo`, no `handoffs/connect_17_*`. Task 17 has produced nothing. Task 01 proceeds. |
| 12:38 | `src/mojoboost/boosting_rf.mojo` appears. Task 17 is live. |
| 12:42 | `src/mojoboost/alternate_boosting.mojo` appears. |
| 12:43 | Task 01 writes `src/mojoboost/boosting_dart.mojo`. |
| 12:43 | Task 01 detects the conflict and stops editing that file. |

The ownership check at 12:35 was correct on the evidence available; Task 17
materialized afterwards. On detection, this lane stopped writing to the
contended file and made no further edits to it.

**The fusion already happened, in Task 17's direction.**
`alternate_boosting.mojo` imports this lane's core by name:

```mojo
from .boosting_dart import (
    DartParams,
    check_dart_supported,
    dart_begin_round,
    dart_commit_round,
    dart_normalization,
    dart_recompute_raw,
    dart_uniform_weights,
)
```

and its module docstring reproduces this lane's reasoning about per-tree
weights, the round-major multiclass layout, the refused combinations, and why
a DART ensemble cannot be truncated to its best round. So `boosting_dart.mojo`
is now a dependency of Task 17's dispatcher, not an orphan.

**Consequence for whoever picks this up: `src/mojoboost/boosting_dart.mojo`
belongs to CONNECT_EVERYTHING Task 17 from here on.** Treat this document as a
follow-up addressed to that owner. Do not open a second Task 01 session
against that file.

---

## 1. What was delivered

Three files, and nothing else:

- `src/mojoboost/boosting_dart.mojo` (729 lines) — the algorithm core.
- `docs/DART.md` — the specification: the round, the normalization table,
  determinism, multiclass, early stopping, the refusals, and what is
  unverified.
- this file.

No central trainer, no `Tree`, no `params.mojo`, no serialization, no
bindings, no Python, and no test was touched.

## 2. What the core is, precisely

It is the algorithm, not a trainer. It grows no tree, reads no label, and
owns no loop. It supplies the three things DART adds to a round that plain
GBDT does not have, expressed over `(trees, weights, raw)`:

| Function | Does |
|---|---|
| `select_drop` | Draws the round's dropped iterations from a `(seed, round)` counter stream |
| `dart_begin_round` | Subtracts the dropped iterations from cached `raw`, keeping what it subtracted |
| `dart_normalization` | The `(new_weight, dropped_scale)` pair for `k` drops |
| `dart_commit_round` | Rescales dropped weights, restores the scaled contribution, appends the new trees |
| `dart_recompute_raw` | Sums `raw` from the model, the slow way; the reference the cache must equal |
| `check_dart_supported` | Refuses GPU, ranking, GOSS by name |
| `DartParams.validate` | Range checks; refuses `uniform_drop=false` and the never-drops configuration |
| `DartBestState`, `dart_record_best`, `dart_restore_best` | Best-round weight snapshot for early stopping |
| `dart_uniform_weights`, `dart_weights_are_uniform` | Lift a GBDT ensemble into the weighted form, and detect the reverse |

Design notes worth not re-deriving:

- **Weights are per tree, not per iteration**, so the vector maps one-to-one
  onto tree storage. All `n_classes` trees of a dropped iteration share a
  scale.
- **Drops are per iteration, never per class.** Dropping one class's tree
  would tilt the softmax toward the classes whose trees survived.
- **The core is written for multiclass throughout.** `n_classes` runs through
  every entry point and the round-major layout `trees[i * n_classes + k]` is
  respected. `n_classes = 1` is the single-output case.
- **No stdlib API was assumed.** `List.insert` appears nowhere else in this
  repo, so the two ordering helpers were written with `append`, `pop`, and
  index assignment only, which is the list surface the rest of the library
  uses. `List[Bool]` is used in `levelwise_policy.mojo` and 73 other places,
  so it is safe.
- **No import cycle.** The core imports only `.binning`, `.device`, and
  `.tree`, none of which import `.boosting`. This is deliberate: the
  integration path has `boosting.mojo` importing the core, so the core must
  not reach back. This is why `check_dart_supported` takes an `is_ranking:
  Bool` rather than importing `LAMBDARANK` from `.ranking`, which imports
  `.model`, which imports `.boosting`.

## 3. Implemented vs public vs integrated vs validated

The prompt asks these be kept apart. They are:

| Layer | State |
|---|---|
| **Algorithm implemented** | Yes, as written. Every function above exists with its stated behavior. |
| **Compiles** | **Unknown.** No Mojo was run. |
| **Integrated into a trainer** | Partly, and not by this lane. `alternate_boosting.train_dart` is Task 17's; this lane has not read its loop body and makes no claim about it. |
| **Reachable from `boosting.train`** | No. |
| **Reachable from `parse_params`** | No. `boosting` is in `_MOJO_API_ONLY` (`params.mojo:86`), so `boosting=dart` raises "supported by the Mojo API only". |
| **Reachable from Python** | No. `_BOOSTING_TYPES = ("gbdt", "goss")` (`python/mojoboost/__init__.py:332`). |
| **Serializes** | Not by this lane's design. See section 4. |
| **Numerically validated** | No. Nothing was executed. |
| **LightGBM parity** | **No, and one constant is actively unverified.** See section 6. |

## 4. Serialization: my request is withdrawn in favor of Task 17's answer

This lane's analysis was that DART needs a per-tree weight vector on
`Booster`, and therefore a model format bump from v4 to v5
(`serialize.mojo:84`, `CURRENT_FORMAT_VERSION` at `serialize.mojo:89`,
`MODEL_FORMAT_VERSION = 3` at `model_dump.mojo:68`).

**Task 17 found a better answer and this lane withdraws the format-bump
request.** `alternate_boosting.fold_weights_into_trees` multiplies each tree's
node values by that tree's weight at the end of training and leaves
`Booster.learning_rate` at 1.0. That is what LightGBM's `Tree::Shrinkage`
does, it needs no format change, and `contrib`, `importance`, `inspection`,
`gpu_predict`, and `lgbm_model_io` then see a plain ensemble.

Three residual concerns this lane raises against that approach, for Task 17
to accept or dismiss. None of them is a reason to go back to a format bump:

1. **The reported learning rate becomes a lie.** After folding,
   `save_model` writes `learning_rate 1.0` (`serialize.mojo:276`) and
   `inspection.mojo:280` reports 1.0 in its JSON dump. A user who trained with
   `learning_rate=0.1` will read 1.0 back. LightGBM keeps a per-tree
   `shrinkage` field precisely so the record survives. Suggest either keeping
   the configured rate in the file as a reported-only field, or documenting
   the 1.0 explicitly in `docs/MODEL_FORMAT.md` and the parity table.
2. **Folding must happen exactly once, after the last round.** Folding per
   round would compound the rounding and would break `dart_begin_round`, which
   reads `weights[s]` to reconstruct what to subtract. The docstring says "at
   the end of training", which is right; it is worth an assertion rather than
   a comment.
3. **`_renew_leaf_values` must run before the fold.** It rewrites
   `tree.value[node]` (`boosting.mojo:627`). If any renewal ran after folding,
   the renewed leaves would lose the weight. Renewal is per round and folding
   is at the end, so the ordering is already correct; it is worth stating so a
   later edit does not invert it.

## 5. Exact patch requests

None of these were applied. Each is scoped to a file this lane does not own.
All of them are gated on the core compiling first (section 7), because there
is no point widening a public surface onto unproven code.

### 5.1 `src/mojoboost/boosting.mojo` — none, for now

This lane requests **no change**. The DART loop lives in
`alternate_boosting.train_dart` behind its own door, which is the right place
until it has been run. Folding it into `boosting.train` is Task 17's recorded
follow-up, not this lane's.

One guard is worth adding when that happens: `train_more` (`boosting.mojo:1079`)
must refuse a folded DART ensemble. It reads `booster.learning_rate` as the
rate to shrink new trees by and would read the folded `1.0`. Task 17 already
flags this and routes continuation through `train_boosting_more`.

### 5.2 `src/mojoboost/params.mojo` — none

`boosting` and `boosting_type` are already in `_MOJO_API_ONLY`
(`params.mojo:86-96`), so `boosting=dart` in a parameter string raises
"parameter 'boosting' is supported by the Mojo API only, not by parameter
strings". That message stays true after DART lands, because selecting DART
means handing the trainer a `DartParams`, exactly as selecting GOSS means
handing it a `GossParams` (`params.mojo:576-584`).

**Do not add `dart` to `SUPPORTED_KEYS`** (`params.mojo:66`). A parameter
string cannot carry a `DartParams` any more than it can carry a `GossParams`.

### 5.3 `src/mojoboost/serialize.mojo` — none

Withdrawn; see section 4.

### 5.4 `capi/mojoboost_capi.mojo` and `cli/` — none

Both drive training through `parse_params`, which cannot select DART by 5.2.
No C ABI or CLI surface changes.

### 5.5 `python/mojoboost/__init__.py` — deferred, with the exact edit

Do **not** apply until section 7 passes. When it does, the minimum is:

1. `_BOOSTING_TYPES = ("gbdt", "goss")` at line 332 gains `"dart"`.
2. `_resolve_boosting` (line 1395) returns `"dart"`; its docstring at line
   1396 says "gbdt" or "goss" and must be updated.
3. `fit` (near line 1480, where `goss = boosting == "goss"`) grows the DART
   branch and passes a `DartParams` down.
4. Six new constructor keywords with LightGBM's names and defaults:
   `drop_rate=0.1`, `max_drop=50`, `skip_drop=0.5`, `uniform_drop=True`,
   `xgboost_dart_mode=False`, `drop_seed=4`. **`uniform_drop` defaults to
   `True`, which is not LightGBM's default**; see section 6.
5. Reject the four refused combinations at the Python layer too, so the error
   arrives before binning: device, ranking, GOSS, `uniform_drop=False`.
6. `python/mojoboost/dask.py:1384-1388` resolves `boosting` for the Dask
   estimators and needs the same treatment or an explicit DART refusal.

### 5.6 Python tests — deferred, and currently correct

`python/tests/test_params.py:169` and `python/test_python_api.py:366` both
assert that `{"boosting": "dart"}` raises. **Those assertions are correct
today and this lane did not touch them.** They must be inverted in the same
change as 5.5, not before. This is the tripwire that keeps DART from
half-landing.

### 5.7 `docs/LIGHTGBM_PARITY.md` — deferred

Four rows describe DART as deferred and are accurate today: line 92 (the
deferred list), line 172 (`boosting_type`), line 330 (`boosting`), and line
366 (the six DART parameters). They change in the same commit as 5.5. When
they do, the `uniform_drop` row must record the default inversion as a
`different`, not a `supported`.

## 6. Risks

1. **The normalization constant is unverified.** `dart_normalization` uses
   `lr / (k + 1)`. Whether LightGBM multiplies the shrinkage or replaces it
   (`1 / (k + 1)`) was not checked against `dart.hpp`; this lane ran nothing
   and fetched nothing. It is isolated to one function so settling it is a
   one-line change. **No DART parity claim may be made until it is settled.**
2. **`uniform_drop` defaults opposite to LightGBM.** LightGBM defaults it
   false and mojoboost refuses false, so `DartParams.disabled()` defaults it
   true. A user porting a LightGBM config that relies on the default gets an
   error, which is the intended behavior, but it must be documented as a
   difference rather than discovered.
3. **Two selection rules are mojoboost's own**, not LightGBM's: the
   `max_drop` cap keeps the smallest draws where LightGBM truncates a shuffled
   list, and an unskipped round is forced non-empty. Both are documented in
   `docs/DART.md` section 4. Neither is bit-compatible with LightGBM.
4. **Early stopping is not connected**, and the naive connection is wrong.
   Anyone adding `train_dart_with_valid` who truncates without restoring the
   snapshot will produce the right tree set with the wrong weights, silently.
   `DartBestState` exists to prevent exactly that.
5. **The shared checkout moved under this lane.** `git status` returned two
   entirely different sets of modified files four minutes apart, so other
   lanes are committing concurrently. Every line number cited here was read
   from the working tree during this session and should be re-checked rather
   than trusted if a lane has since rewritten the file.

## 7. Smallest later checks, all UNRUN

In order. Stop at the first failure. Nothing below was executed by this lane.

1. **Does the core compile?** This is the gate on everything else.
   ```
   UNRUN:  mojo build -I src src/mojoboost/boosting_dart.mojo -o /dev/null
   ```
   The specific risks are `@always_inline` on a module-level `def`, the
   `while` shift loop in `_insert_by_key`, and the `var`-argument
   constructors on `DartDrop` and `DartBestState`.

2. **Does the cache equal the model?** The single highest-value test, and the
   one the module was shaped to make cheap. Build a small ensemble, run rounds
   through `dart_begin_round` / `dart_commit_round`, and compare the
   maintained `raw` against `dart_recompute_raw`. They must agree to within
   floating-point association.
   ```
   UNRUN:  one focused test in tests/parallel/test_dart.mojo
   ```

3. **Does `k = 0` reproduce GBDT exactly?** With `skip_drop=1.0` and
   `drop_rate` positive, every round skips, every weight is `learning_rate`,
   and the ensemble must equal what `boosting.train` produces for the same
   seed and data, tree for tree and value for value. This is the cheapest
   check that the round plumbing is not perturbing the ordinary path.

4. **Is the draw reproducible across a split run?** `select_drop(p, n, 40)`
   must return the same set whether round 40 was reached in one call or two.
   This is the property `train_more` promises for bagging and GOSS.

5. **Settle the normalization constant** against LightGBM's `dart.hpp`, then
   fix `dart_normalization` and the table in `docs/DART.md` section 3
   together. Only after this may any parity row say anything but `deferred`.

Per the repo's test budget: one focused test per change, never the full
suite, never a build-and-bench loop.
