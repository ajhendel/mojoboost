# Handoff: complete CEGB cost penalties (remaining task 04)

Owned by this lane, and the only files it touched:

- `src/mojoboost/cegb.mojo` (new)
- `docs/CEGB.md` (new)
- `handoffs/remaining_04_cegb.md` (this file)

Nothing was built, run, tested, benchmarked, or committed. Every claim below
was derived by reading source. Every command in section 9 is **UNRUN**.

Ownership note. `MOJOBOOST_CONNECT_EVERYTHING_PARALLEL_PROMPTS.txt` task 17
also names `src/mojoboost/cegb.mojo`, and task 09 owns
`tree_parameters_extra.mojo`, `split.mojo`, `tree.mojo`, `boosting.mojo`, and
`params.mojo`. At the time this lane ran, `cegb.mojo` did not exist and
`handoffs/connect_17_alternate_boosting.md` had not been written, so this lane
created it. **If task 17 becomes active, it owns this file from then on**, and
the patches in section 4 belong to task 09's owner, not to a second editor of
those files.

---

## 1. Implementations found

One fragment, in one place, plus four names that are refused.

| Where | What it holds | State |
| --- | --- | --- |
| `tree_parameters_extra.FeaturePenalties` (line 281) | `cegb_tradeoff`, `cegb_penalty_split`, `cegb_penalty_feature_coupled`, and the `feature_contri` multiplier, in one struct | live for the split cost, parsed-and-refused for coupled |
| `FeaturePenalties.penalized_gain` (line 390) | multiplier and split cost applied together, `feature_already_used` taken as a parameter | called by `split._feature_gain` |
| `split._feature_gain` (line 246) | passes `feature_already_used=True` unconditionally, because no ledger exists | live on every caller of `tree._search` |
| `ExtraTreeParams.check_scalars` (line 1152) | refuses an active coupled vector | live |
| `check_extra_option_supported` (line 1013) | refuses `cegb_penalty_feature_lazy` and `cegb_penalty_feature_coupled` by name | live |
| `params.parse_params` (lines 563, 567) | accepts `cegb_tradeoff` and `cegb_penalty_split` only | live |

No second implementation existed anywhere. There is no ledger, no per-row
state, no lazy penalty, and no feature-index recovery in the tree.
`grep -rn -i cegb` over the repository outside this lane's files returns only
the rows above plus documentation and one test file.

## 2. What this lane added, and what it fused

`src/mojoboost/cegb.mojo` is now the authoritative CEGB implementation. It
covers all four LightGBM parameters, the ledger both of the unimplemented ones
need, the feature-index recovery, the active-row accounting, the backend
refusals, and the arithmetic. `docs/CEGB.md` is the design document; the gain
formula is section 3 there and is not repeated here.

Public surface of `cegb.mojo`:

| Symbol | Purpose |
| --- | --- |
| `CegbConfig` | the four parameters, LightGBM defaults, `check(n_features)` |
| `CegbConfig.from_feature_penalties` | reads the existing fragment; deliberately does **not** read `contri` |
| `CegbLedger` | `feature_used` flags plus the (feature, row) read bitset, ensemble lifetime |
| `CegbNodeCosts` / `prepare_cegb_node` | one node's costs, computed once, O(1) per candidate |
| `cegb_split_cost` / `cegb_coupled_cost` / `cegb_lazy_cost` | the three terms, individually |
| `cegb_delta_gain` / `cegb_adjusted_gain` | the whole charge, and a gain after it |
| `cegb_commit_split` / `CegbCommit` / `cegb_stale_cached_gain` | the ledger's only writer, and the frontier refund |
| `cegb_dataset_feature` | bundle-or-identity to dataset feature id |
| `check_cegb_grower_support` / `check_cegb_device_split_search` / `check_cegb_distributed` / `check_cegb_continued_training` | the four refusals |

**Fusion status: half done, and the unfinished half is a live hazard.**
`CegbConfig.from_feature_penalties` makes `FeaturePenalties` a source rather
than a second implementation, and `cegb_adjusted_gain` reproduces
`penalized_gain`'s CEGB arithmetic exactly when the ledger is inert, so the
current split search is a special case of the new module rather than a rival
to it. But `FeaturePenalties` still owns the three `cegb_*` fields and
`penalized_gain` still applies the multiplier and the split cost in one call.
Until PATCH 1 and PATCH 2 land, **`penalized_gain` and any cegb.mojo entry
point must not both be called on one gain**, or `feature_contri` is applied
once and the split cost twice. That is the single highest-risk item in this
handoff and it is why PATCH 1 and PATCH 2 are one unit.

Nothing was quarantined and nothing was deleted. `penalized_gain` is still the
live path and stays live until PATCH 1 replaces its call site.

## 3. Call path, before and after

Before, and still true today:

```
params.parse_params  ->  TreeParams.extra.penalties (cegb_tradeoff, cegb_penalty_split)
tree.grow_tree       ->  tree._search  ->  split.find_best_split
                                        ->  split._feature_gain
                                        ->  FeaturePenalties.penalized_gain
                                            (contri x gain, minus tradeoff*split*rows,
                                             feature_already_used hardcoded True)
```

After the patches in section 4:

```
boosting.fit / fit_multiclass
  owns one CegbLedger for the whole ensemble
  -> tree.grow_tree(..., mut ledger)
       per node: prepare_cegb_node(config, ledger, n_features, node_rows, rows, features)
       -> tree._search(..., cegb=costs)
            -> split.find_best_split(..., cegb=costs)
                 -> split._feature_gain
                      g = gain * penalties.contri_of(f)        # multiplier only
                      g = costs.adjusted_gain_at(g, f, node_rows)  # every CEGB term
       on commit: cegb_commit_split(ledger, config, split.feature, rows)
                  frontier[i].split.gain += cegb_stale_cached_gain(commit, ...)
```

`cegb.mojo` is imported by nothing today. It imports `.efb` (for
`FeatureBundling` and `EFB_NONE`) and `.tree_parameters_extra` (for
`FeaturePenalties`). Neither imports `cegb`, so no cycle exists, and
`split.mojo` already imports `FeatureBundling` from `.efb`, so PATCH 1 adds no
new package edge that is not already present.

## 4. Ready-to-apply integration patches

Each patch is mechanical. Apply in order; 1 and 2 are one unit.

---

### PATCH 1 (with PATCH 2) -- `src/mojoboost/split.mojo`

**Owner**: connect-09 lane (`split.mojo`). **Depends on**: PATCH 2, same
commit.

**Target symbols**: the import block at line 93, `_feature_gain` (line 246),
`find_best_split` (line 287), and its two `_feature_gain` call sites (lines
472 and 611).

**1a. Import.** Add after the `.gain` import:

```mojo
from .cegb import CegbNodeCosts
```

**1b. Signature.** `_feature_gain` gains a trailing argument and becomes
raising (`CegbNodeCosts.adjusted_gain_at` raises):

```mojo
@always_inline
def _feature_gain(
    gain: Float64,
    feature: Int,
    extra: ExtraTreeParams,
    extra_active: Bool,
    penalize: Bool,
    sign: Int,
    depth: Int,
    node_rows: Int,
    cegb: CegbNodeCosts,
) raises -> Float64:
```

**1c. Body.** Replace the two lines

```mojo
    if penalize:
        g = extra.penalties.penalized_gain(g, feature, node_rows, True)
```

with

```mojo
    if penalize:
        g = g * extra.penalties.contri_of(feature)
    g = cegb.adjusted_gain_at(g, feature, node_rows)
```

**1d. `find_best_split` signature.** Append, after `parent_output`:

```mojo
    cegb: CegbNodeCosts = CegbNodeCosts.inactive(),
```

**1e. Fallback, so no caller regresses.** After
`var penalize = extra_active and extra.penalties.is_active()` insert:

```mojo
    # A caller that passes no prepared costs but did set the CEGB split cost
    # gets it charged anyway, from the same numbers `penalized_gain` used to
    # read. The ledger stays inert, which is exactly the "feature already
    # used" constant this scan passed before.
    var cegb_costs = cegb.copy()
    if not cegb_costs.active and extra.penalties.split_costs_active():
        cegb_costs = prepare_cegb_node(
            CegbConfig.from_feature_penalties(extra.penalties),
            CegbLedger.none(),
            hist.n_features,
            0,
        )
```

which needs `CegbConfig`, `CegbLedger`, and `prepare_cegb_node` added to the
import in 1a. Pass `cegb_costs` (not `cegb`) at both call sites.

**Why `0` for the row count**: `CegbNodeCosts` stores `split_rate` factored
from the row count precisely so that `adjusted_gain_at(g, f, node_rows)` can
charge the count the scan computes inside the loop
(`n_rows if n_rows > 0 else total_c`, line 444). The existing `total_c`
fallback keeps working unchanged. `delta_at` raises if a lazy penalty was
counted over a different row count, so the two halves of a node's cost can
never disagree about the node's size.

**State flow**: read-only. `_feature_gain` reads `cegb_costs`; nothing here
writes the ledger.

**Errors**: `_feature_gain` becomes raising. Both call sites are already
inside `find_best_split`, which is `raises`, so no other signature changes.
New error text: `CegbNodeCosts.delta_at` raises for a feature outside the
costed set, and for a row-count mismatch.

**Fallback preserved**: an inactive bundle and an inactive `CegbNodeCosts`
leave the scan on exactly the path it takes today. With
`cegb_penalty_split > 0` and no prepared costs, 1e reproduces today's number
through the same multiplication in the same order.

**Serialization effect**: none. **Public API effect**: none;
`find_best_split`'s new argument is defaulted and trailing.

**Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_tree_parameters_extra.mojo`
-- the CEGB cases at lines 199 and 219 go through `penalized_gain`, which
PATCH 2 changes, so this file is the one that proves 1+2 together.

---

### PATCH 2 (with PATCH 1) -- `src/mojoboost/tree_parameters_extra.mojo`

**Owner**: connect-09 lane. **Depends on**: nothing; must ship with PATCH 1.

**Target symbols**: `FeaturePenalties` (line 281), `check_extra_option_supported`
(line 1013), `ExtraTreeParams.check_scalars` (line 1152).

**2a. Fields.** Replace the three fields

```mojo
    var cegb_tradeoff: Float64
    var cegb_penalty_split: Float64
    var cegb_penalty_feature_coupled: List[Float64]
```

with one:

```mojo
    var cegb: CegbConfig
```

and add `from .cegb import CegbConfig` at the top. **This is the fusion**:
after it there is exactly one home for CEGB parameters.

**Import direction**: `cegb.mojo` currently imports `FeaturePenalties` from
this file, so this patch must also delete that import and delete
`CegbConfig.from_feature_penalties`, which exists only to bridge the two homes
before this patch. `docs/CEGB.md` section 11 step 1 says the same. Doing it in
the other direction (this file importing `cegb`, `cegb` importing this file)
is a cycle and will not compile.

**2b. `penalized_gain` keeps only the multiplier.** Replace the body with

```mojo
    def penalized_gain(self, gain: Float64, feature: Int) -> Float64:
        """A candidate's gain after this feature's `feature_contri`
        multiplier, and nothing else. The CEGB terms are subtracted by
        `cegb.CegbNodeCosts`, which is the only place they are charged."""
        return gain * self.contri_of(feature)
```

and delete `coupled_of`, `coupled_is_active`, and `split_costs_active`, whose
callers move to `CegbConfig.coupled_of`, `coupled_active`, and
`split_cost_active`. `is_active` becomes
`self.cegb.is_active() or <any contri != 1.0>`. `check_features` keeps its
`contri` checks and delegates the rest to `self.cegb.check(n_features)`.

**2c. Constructors.** `FeaturePenalties.cegb(tradeoff, penalty_split,
penalty_feature_coupled)` keeps its name and signature and builds a
`CegbConfig` internally, so `tests/parallel/test_tree_parameters_extra.mojo`
lines 200, 206, 213, 244, 248 keep compiling. Its callers that reach into
`p.cegb_tradeoff` / `p.cegb_penalty_split` (test lines 225 and 226) become
`p.cegb.tradeoff` / `p.cegb.penalty_split`. **That is a test edit, and this
lane must not make it**; it is listed here so whoever applies PATCH 2 makes it
in the same commit.

**2d. Refusals.** `check_extra_option_supported`'s
`cegb_penalty_feature_coupled` branch stays until PATCH 4 lands, and its
`cegb_penalty_feature_lazy` branch until PATCH 5. `check_scalars`'s
`coupled_is_active()` call becomes `self.penalties.cegb.coupled_active()`.

**State flow**: none new. **Errors**: identical messages, one owner.
**Serialization effect**: none. **Public API effect**: none through `params`;
Mojo-API callers constructing `FeaturePenalties` by field name change three
names, which is why 2c is spelled out.

**Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_tree_parameters_extra.mojo`

---

### PATCH 3 -- `src/mojoboost/tree.mojo`

**Owner**: connect-09 lane. **Depends on**: PATCH 1, PATCH 2.

**Target symbols**: `_search` (line 689), `grow_tree` (line 758), the frontier
loop (lines 900 to 1095), `_LeafState` (line 454).

**3a. `_search`** gains two trailing arguments:

```mojo
    cegb: CegbNodeCosts = CegbNodeCosts.inactive(),
    grower_applies_cegb: Bool = False,
```

and, beside the existing `grower_applies_extra` refusal at line 723, adds

```mojo
    check_cegb_grower_support(
        params.extra.penalties.cegb,
        grower_applies_cegb,
        grower_applies_cegb,
    )
```

`cegb` is forwarded to `find_best_split`. `tree_sparse.grow_tree_sparse` and
`distributed` import this same `_search` and leave both defaults, so they are
refused for the coupled and lazy penalties and unchanged for the split cost.
That is the same shape as `grower_applies_extra` and is deliberate.

**3b. `grow_tree`** gains a trailing `mut ledger: CegbLedger` argument. Every
existing caller (`boosting.mojo` lines 976, 1249, 1540) needs a ledger, which
PATCH 4 supplies; a defaulted inert one is **not** acceptable here, because a
silently inert ledger recharges every first use and produces a wrong model
rather than an error.

**3c. Per node.** Before each of the three `_search` calls (root at line 882,
children at lines 1025 and 1048), build the node's costs:

```mojo
        var costs = prepare_cegb_node(
            params.extra.penalties.cegb,
            ledger,
            data.n_features,
            len(left_rows),
            left_rows,
            <the same feature list passed to _search>,
        )
```

and pass `cegb=costs, grower_applies_cegb=True`. The row list and the feature
list are both already in hand at all three sites, which is the whole reason
this grower is the one that can do it.

**3d. On commit.** After `tree._set_split(parent_node, split, split_missing_bin)`
(line 1017), the split node's rows are `frontier[best_i].rows`, which is still
alive at that point:

```mojo
        var charged = cegb_dataset_feature(
            bundling, split.feature, split.bin, split.is_categorical
        ) if <bundling active> else split.feature
        var commit = cegb_commit_split(
            ledger,
            params.extra.penalties.cegb,
            charged,
            frontier[best_i].rows,
        )
        if commit.feature_newly_used:
            for i in range(len(frontier)):
                if i == best_i:
                    continue
                frontier[i].split.gain += cegb_stale_cached_gain(
                    commit, frontier[i].split.feature, charged
                )
```

`grow_tree` takes an unbundled `BinnedMatrix` today, so `charged` is
`split.feature` until EFB is connected; write it as the plain assignment and
leave `cegb_dataset_feature` for the lane that connects EFB, with a comment
naming it.

**Ordering matters**: commit *after* the parent's own split is recorded and
*before* the children are searched, so the children see the ledger the parent
just updated. That is also LightGBM's order.

**3e. `_LeafState.split`** must become mutable in place for 3d's refund. It is
already a `var` field of a `Movable` struct held in a `List`, so
`frontier[i].split.gain += ...` needs no struct change; confirm at build.

**State flow**: `ledger` is `mut` and flows in from `boosting`, is read by
`prepare_cegb_node`, and is written only by `cegb_commit_split`.

**Errors**: `check_cegb_grower_support` raises for the sparse, distributed,
and GPU-device callers that leave `grower_applies_cegb` False and have an
active coupled or lazy penalty. `prepare_cegb_node` raises on a row-count
mismatch and on an inert ledger with an active coupled or lazy penalty.

**Fallback preserved**: with the default `CegbConfig` (which is what every
caller has today) `prepare_cegb_node` returns the inactive costs immediately
and `cegb_commit_split` returns `CegbCommit.nothing()`, so the tree is
bit-identical.

**Serialization effect**: none directly; see PATCH 8. **Public API effect**:
`grow_tree` gains a required argument, which is a Mojo-API break for any
direct caller outside `boosting.mojo`.

**Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_tree.mojo`

---

### PATCH 4 -- `src/mojoboost/boosting.mojo`

**Owner**: connect-09 lane. **Depends on**: PATCH 3.

**Target symbols**: `_boost_rounds` (around line 938), `fit` (line 1007),
`fit_more` (line 1088), the multiclass loop (line 1500).

Create the ledger once, before the round loop, and hand the same one to every
`grow_tree` call:

```mojo
    var ledger = CegbLedger.create(
        params.tree.extra.penalties.cegb, data.n_features, data.n_rows
    )
```

For multiclass, **one ledger shared across all classes**, not one per class: a
feature computed for class 0's tree is computed for the row, and charging it
again for class 1 would charge `n_classes` times for one feature. That is a
judgment this lane is recording, not one it can test; it follows from the cost
being a property of the deployed model, and it should be checked against
LightGBM before parity is claimed.

`fit_more` calls `check_cegb_continued_training(config, resumed=True)` unless
it is handed a live ledger, which the current signature has no way to receive.
Refusing is correct until PATCH 8.

**State flow**: `ledger` lives for the ensemble, crosses every round and every
class, and is destroyed with the fit.

**Errors**: `CegbLedger.create` raises on negative dimensions or on penalty
vectors longer than the dataset. `check_cegb_continued_training` raises on a
resumed fit with an active coupled or lazy penalty.

**Fallback preserved**: `CegbLedger.create` allocates nothing for the default
configuration and returns the inert ledger, so a default fit does no extra
work at all.

**Serialization effect**: none. **Public API effect**: none through `params`.

**Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_boosting.mojo`

---

### PATCH 5 -- `src/mojoboost/distributed.mojo`

**Owner**: distributed lane. **Depends on**: PATCH 2.

In `_unsupported_mask` (line 360), beside the existing `extra_trees` and
forced-splits entries, add a `_UNSUPPORTED_CEGB_LAZY` bit set by
`params.extra.penalties.cegb.needs_row_ledger()`, and raise it in
`_raise_unsupported` with the text from `check_cegb_distributed`. Do not
refuse the split cost (every rank scores from the reduced histogram and the
exact global row count, so it already agrees) and do not refuse the coupled
cost once PATCH 3 exists (every rank commits the same split, so every rank's
flag flips at the same moment with no message). The lazy count is a sum over
shards and needs one integer allreduce per (node, feature), which
`distributed_transport` does not have.

The mask is computed identically on every rank, so it raises everywhere or
nowhere, which is that file's existing invariant.

**Public API effect**: none. **Serialization effect**: none.
**Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_distributed.mojo`

---

### PATCH 6 -- `src/mojoboost/train_gpu.mojo`

**Owner**: GPU lane. **Depends on**: PATCH 2.

`_check_device_search_supported` (line 419) already refuses the whole
`params.extra` bundle via `extra.is_active()`, so an active CEGB
configuration is refused today. Replace the generic message with a call to
`check_cegb_device_split_search(params.extra.penalties.cegb)` **before** the
generic check, so the caller is told which of the seven controls stopped them
and that the host split-search strategy is the fix. Behavior is otherwise
unchanged; this is a message improvement, not a new refusal.

The host split-search path routes through `tree._search` and therefore gets
PATCH 3's `grower_applies_cegb=False` refusal for the coupled and lazy
penalties, and keeps the split cost.

**Public API effect**: none. **Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_train_gpu.mojo`

---

### PATCH 7 -- `src/mojoboost/tree_sparse.mojo` and `boosting_sparse.mojo`

**Owner**: sparse lane. **Depends on**: PATCH 3.

`tree_sparse` imports `tree._search` (line 54) and calls it at lines 249, 419,
443 without the new arguments, so it inherits the refusal automatically and
keeps the split cost. No edit is required for correctness. It *is* worth
adding the ledger, since `grow_tree_sparse` materializes node row lists the
same way the dense grower does; that is a follow-on, not a blocker, and the
same three sites in section 4/PATCH 3 apply verbatim.

---

### PATCH 8 -- `src/mojoboost/serialize.mojo` (only if resumed training must work)

**Owner**: serialization lane. **Depends on**: PATCH 4.

Required state, and only this:

- `feature_used`: `n_features` bits. Cheap and always needed when
  `cegb_penalty_feature_coupled` is set.
- the lazy read bitset: `n_features * n_rows` bits. 12.5 MB for 100 features
  over 10 million rows, and meaningless against a *different* dataset, so it
  must be written only when the lazy penalty is configured and must refuse to
  load against a dataset with a different row count.

This bumps `CURRENT_FORMAT_VERSION` from 4 to 5. It is **training** state in a
**model** file, which is a real design question, not a mechanical addition:
the alternative is a separate resume file, which keeps the model format clean
and is this lane's recommendation. Either way,
`check_cegb_continued_training` is what has to be relaxed afterwards, and it
should be relaxed to accept a restored ledger rather than deleted.

**Public API effect**: a new format version, and `model_file_kind` /
`_read_version` gain a case. **Minimal later validation (UNRUN)**:
`pixi run mojo run tests/parallel/test_serialize.mojo`

---

### PATCH 9 -- `src/mojoboost/params.mojo`

**Owner**: connect-09 lane. **Depends on**: PATCH 4 for coupled, PATCH 4 plus
row threading for lazy. **Do not apply before then.**

- Remove the `cegb_penalty_feature_coupled` branch from
  `check_extra_option_supported` once PATCH 3 and PATCH 4 are in; remove the
  `cegb_penalty_feature_lazy` branch once node row ids are threaded.
- Both are per-feature vectors of doubles. A whitespace-separated parameter
  string cannot carry one, exactly as it cannot carry `monotone_constraints`
  or `feature_contri`. So they join `_MOJO_API_ONLY` (line 88) rather than
  `SUPPORTED_KEYS`, and `feature_contri`'s existing treatment is the
  precedent to copy.
- `cegb_tradeoff` and `cegb_penalty_split` keep their existing branches; only
  their destination changes, from
  `config.booster.tree.extra.penalties.cegb_tradeoff` to
  `config.booster.tree.extra.penalties.cegb.tradeoff`.

**Public API effect**: this is the patch that makes CEGB reachable. Nothing
before it changes what a user can set.

---

### PATCH 10 -- `src/mojoboost/__init__.mojo`

**Owner**: package lane. **Depends on**: PATCH 9.

```mojo
from .cegb import (
    CegbConfig,
    CegbLedger,
    CegbNodeCosts,
    cegb_adjusted_gain,
    cegb_commit_split,
    prepare_cegb_node,
)
```

Export **after** PATCH 9, not before. Exporting a struct no split decision
consumes would make `tools/check_parity.py` see a public symbol behind a
`deferred` row and report the row as stale, which would be a false signal.

---

### PATCH 11 -- `bindings/_mojoboost.mojo` and `python/mojoboost/basic.py`

**Owner**: bindings lane. **Depends on**: PATCH 10.

Two per-feature `List[Float64]` vectors, arriving the way `feature_contri`
and `monotone_constraints` already do. Python stays thin: validate length
against `num_feature`, convert to the array form the binding takes, and pass
through. No CEGB arithmetic in Python. `cegb_tradeoff` and
`cegb_penalty_split` are scalars and need only be routed to the new field
names.

---

### PATCH 12 -- `docs/LIGHTGBM_PARITY.md`

**Owner**: docs/parity lane (connect-19). **Depends on**: PATCH 11 plus
evidence.

Line 380's row and line 124's remaining-tree-parameters row both describe the
state before this lane. Neither should move off `deferred` on the strength of
this handoff: the implementation exists, nothing calls it, and nothing has
been run. `docs/CEGB.md` section 12 is the reachability table to copy from
when the row does move.

## 5. Remaining disconnections

1. `cegb.mojo` is imported by nothing. This is the whole of the gap.
2. `FeaturePenalties` still owns three CEGB fields (PATCH 2), so there are two
   spellings of the split cost, and calling both would double-charge it.
3. No grower carries a ledger, so `cegb_penalty_feature_coupled` is still
   correctly refused and `cegb_penalty_feature_lazy` is still correctly
   refused.
4. `cegb_dataset_feature`'s bundling branch is unexercised, because EFB is
   disabled by default and unconnected. It is written against
   `efb.FeatureBundling`'s current API (`n_bundles`, `bundle_size`,
   `member_at`, `decode_feature`, `EFB_NONE`) and will need re-reading if that
   API moves.
5. The multiclass ledger-sharing decision (PATCH 4) is recorded, not verified.

## 6. Fallbacks preserved

- Default `CegbConfig` is inactive, `CegbLedger.create` allocates nothing for
  it, and `prepare_cegb_node` returns inactive costs immediately. A default
  fit does no extra work and takes no new branch.
- `CegbNodeCosts.inactive()` is the default for every proposed new argument,
  so an unpatched caller compiles and behaves as before.
- PATCH 1's step 1e keeps today's split-cost behavior for any caller that
  passes no prepared costs, through the same multiplication in the same order.
- Every backend that cannot honor a term refuses it rather than ignoring it,
  which is `tree._search`'s existing `grower_applies_extra` contract.

## 7. Serialization and public API effects

- **Today**: none. Nothing is exported, nothing is parsed differently, no
  format version moves.
- **After PATCH 3**: `tree.grow_tree` gains a required argument (Mojo API).
- **After PATCH 9 and 10**: two new Mojo-API-only vectors, two rerouted
  scalars, six new exported symbols.
- **After PATCH 8, if taken**: `CURRENT_FORMAT_VERSION` 4 -> 5, and a model
  file that carries training state. See the recommendation there.

## 8. Risks

1. **Double-charging the split cost.** Highest risk, described in section 2.
   PATCH 1 and PATCH 2 must land together.
2. **A silently inert ledger.** An inert ledger answers "already used" and
   "already read" to every question, which charges nothing. That is the right
   answer when no coupled or lazy penalty is set and the wrong one when there
   is. `prepare_cegb_node` and `cegb_commit_split` both raise on that
   combination rather than returning a plausible zero, and PATCH 3 must not
   default `grow_tree`'s ledger argument.
3. **The cached-candidate refund rule.** `cegb_stale_cached_gain` implements
   the arithmetically consistent rule. LightGBM's has not been read. It
   changes which tree comes out. Flagged in the code and in `docs/CEGB.md`
   section 5.1.
4. **The lazy bitset's memory.** `n_features * n_rows` bits, allocated only
   when a nonzero lazy penalty is set. A user who sets it on a wide, tall
   dataset gets a large allocation with no warning. Consider a size check at
   `CegbLedger.create` before PATCH 9 exposes the parameter.
5. **Bagging changes the effective penalty.** Documented (`docs/CEGB.md`
   section 7), not corrected. A user comparing runs at two bagging fractions
   will see the regularization strength move.
6. **Ownership collision.** Task 17 also claims `cegb.mojo`. Section 0.

## 9. Smallest later validation, all UNRUN

Nothing below was run by this lane. Run them one at a time, not as a suite.

```
# after PATCH 1 + 2, the only existing test that touches CEGB arithmetic
pixi run mojo run tests/parallel/test_tree_parameters_extra.mojo

# after PATCH 3
pixi run mojo run tests/parallel/test_tree.mojo

# after PATCH 4
pixi run mojo run tests/parallel/test_boosting.mojo

# after PATCH 9, that the two vectors are still refused by the string parser
pixi run mojo run tests/parallel/test_params.mojo

# after PATCH 10, that the parity checker agrees the row moved on evidence
pixi run check-parity
```

A focused test for `cegb.mojo` itself does not exist and this lane did not
write one. The cases worth covering first, in order: the split cost against
`penalized_gain`'s current number at the same inputs; the coupled cost charged
once across two trees and not once per tree; the refund restoring a cached
gain exactly; `count_unread` under a bag that omits rows; and each of the four
refusals raising.
