# Row sampling reach

Which row sampler is honored, refused, silently ignored or raising, on each
growth policy and each backend. Read in source on 2026-08-17 at the head of
the shared checkout. **Nothing here was measured or run**; every cell is a
citation, and the cells this lane could not settle without a compiler are
listed at the end rather than guessed.

The family is the row samplers, which are uniform bagging (`bagging_fraction`,
`bagging_freq`, `bagging_seed`), class-conditional bagging
(`pos_bagging_fraction`, `neg_bagging_fraction`), GOSS (`top_rate`,
`other_rate`, `goss_seed`, `goss_warmup_rounds`) and CatBoost's
`bootstrap_type` with `subsample`, `bagging_temperature`, `mvs_reg` and
`bootstrap_seed`. Feature sampling is a different family and is not covered.

## The one structural fact that shortens the matrix

**On both backends the row sample is drawn in the round loop, above the
grower, and reaches the grower as one argument.** So no cell of the matrix can
be policy-specific unless the grower reads the bag differently per policy, and
neither grower does:

- CPU. `tree.grow_tree_leaves_profiled` builds `root_rows` from `bag` at
  tree.mojo:3391-3399, validating it with `sampling.check_row_set`, and the
  `grow_policy` fork happens **after** that at tree.mojo:3519. Leaf-wise and
  depth-wise share the loop below the fork; `_grow_oblivious_levels` is handed
  the frontier that already holds `root_rows` and partitions those rows
  (tree.mojo:2549). One row list, three policies.
- GPU. `grow_tree_gpu_profiled` calls `builder.begin_tree(bag)` on the
  host-scan arm (train_gpu.mojo:3569) and `_grow_tree_gpu_device_search` calls
  it at train_gpu.mojo:2135, in both cases before the `grow_policy` fork, and
  every arm below then works on `n_root = len(bag) if len(bag) > 0 else
  builder.n_rows`. `GpuActiveRows.begin_tree` stages the bag's row ids and
  seeds the root range with `len(bag)`
  (gpu_active_rows.mojo:7209-7231), so the rows a bag left out are simply not
  inside the root range. `grow_tree_device_oblivious` and
  `grow_tree_device_resident` both take `n_root` and state the precondition in
  their own docstrings (gpu_resident_round.mojo:3255, :2362).

That is why the interesting failures in this family are not in the growers.
They are at the two places where the sample's *shape* is assumed by something
sized from `num_leaves`, and at the surface, where a knob belonging to an
unselected sampler is accepted.

## Matrix

`policy` is `grow_policy`: leaf (lossguide), depth (depthwise), sym
(symmetrictree / oblivious). Cells are identical across the three policies
unless a row says otherwise, for the reason above.

| parameter | CPU, all three policies | GPU, all three policies |
| --- | --- | --- |
| `bagging_fraction` / `subsample` | HONORED. `boosting._boost_rounds` bagging.refresh_bag at boosting.mojo:2669; `train_with_valid` at :3333; `_boost_rounds_multiclass` at :3925. Bag reaches `grow_tree_leaves`. | HONORED. `train_gpu._train_gpu_rounds` refresh_bag at train_gpu.mojo:4471 (host-gradient arm) and :4229 (device arm); `_train_multiclass_gpu_rounds` at :5508; `_train_gpu_valid_rounds` at :6011. |
| `bagging_freq` / `subsample_freq` | HONORED. Schedule is `iteration % freq` in `bagging.refresh_bag`; the bag in force at round `i` is bag `i // freq`. | HONORED. Same function, same round index. |
| `bagging_seed` | HONORED. `bagging._stream(seed, bag_index)`. | HONORED, and identical: both backends call the same `_stream`, so round `i` grows on the same rows on either device. |
| `pos_bagging_fraction` / `neg_bagging_fraction` | HONORED on dense (`boosting._boost_rounds` :2667, `train_with_valid` :3331) and on sparse (`boosting_sparse` :454, :624). Not on the Python surface at all; `params.mojo`'s `_MOJO_API_ONLY` refuses both spellings from a parameter string by name. | Dense `train_gpu` **takes no such parameter**, so there is nothing to ignore; `train_gpu_sparse` HONORS it (train_gpu_sparse.mojo:687-690, :800-803). |
| `top_rate`, `other_rate`, `goss_seed`, `goss_warmup_rounds` | HONORED under `boosting_type='goss'`: `goss.goss_round` at boosting.mojo:2703 and :3339, `_multiclass_goss_select` at :3935. **SILENTLY IGNORED outside it** -- see the defect below. | HONORED under `boosting_type='goss'`: `goss_round` at train_gpu.mojo:4492, :6015, and `_multiclass_goss_select` at :5523. The device-gradient round is blocked for GOSS by `gpu_fused_round.round_eligibility`, so a GOSS fit always lands on the host-gradient arm and both backends sample identically. **SILENTLY IGNORED outside `boosting_type='goss'`, same defect, same surface.** |
| `bootstrap_type=MVS`, `subsample`, `mvs_reg` | HONORED, dense single-output and sparse single-output and multiclass. `sampling.bootstrap_round` at boosting.mojo:2755, :3370, :3973; boosting_sparse.mojo:484, :800. Derived `mvs_reg` refused for the softmax loops by name (`check_mvs_reg_is_set`). | HONORED on dense single-output. `device_gradients` returns False for MVS (`ROUND_MVS_HOST_MAGNITUDES`), so the fit takes the host-gradient arm and `bootstrap_round` draws it exactly while the trees still grow on the device (train_gpu.mojo:4516-4540, the call at :4528). The device-gradient arm refuses MVS by name at :4162 as a defensive raise. REFUSED BY NAME on GPU multiclass (model.mojo:596-604, trainset.mojo:1796-1804) and GPU sparse (model_sparse.mojo:127-134, :210-217). |
| `bootstrap_type=Bayesian`, `bagging_temperature` | HONORED, same call sites. `bagging_temperature` beside MVS is REFUSED BY NAME (`check_mvs_bagging_temperature`), which is a deliberate divergence from CatBoost, which accepts and ignores it. | HONORED on dense single-output, by a second mechanism: the per-tree draw goes into the objective state's weight plane through `refresh_bayesian_bootstrap` + `builder.refresh_objective_weights` (train_gpu.mojo:4243-4247) on the device arm, and through `bootstrap_round` on the host arm. Bayesian beside `random_strength > 0` routes to the host arm (`ROUND_BAYESIAN_NOISE_SCALE`) and is refused by name if it arrives on the device arm anyway (:4186-4222). |
| `bootstrap_seed` | HONORED. `sampling._mvs_stream` / `_bootstrap_stream`, each with its own domain constant so one seed can be reused across samplers without sharing a stream. | HONORED, same functions on both arms. |
| `random_state` fan-out | HONORED for `bagging_seed` and `goss_seed` through `MojoTreesRegressor._SEEDS` / `_resolve_seeds`, and for `bootstrap_seed` through `_resolve_bootstrap` (python/mojotrees/sklearn.py:2085-2094). A seed named outright wins over `random_state`; `random_state` fills only a seed still at its stock default. | Same, since the seeds are resolved at the Python surface and cross the wire as numbers. |
| `bootstrap_type` from a parameter string | REFUSED BY NAME. `params.mojo`'s `_MOJO_API_ONLY` lists `bootstrap_type`, `bagging_temperature`, `bootstrap_seed`, `bagging_fraction` and every vendor alias, and `boosting_type='goss'` is refused with the Mojo-API sentence (params.mojo:694-717). | Same. |

Two exclusive-with rows, for completeness, because they are what keeps the
matrix from having composite cells. No two of uniform bagging, class bagging,
GOSS and `bootstrap_type` can be on at once
(`boosting._check_goss`, `_check_class_bagging`, `_check_bootstrap`), and
`boosting_type='ordered'` is refused beside all four (`_check_ordered`).

## Defects

### D1. GOSS's four knobs are ignored outside `boosting_type='goss'`

SILENTLY IGNORED. The estimator decides `GossParams.enabled` from
`boosting_type` alone (python/mojotrees/sklearn.py:2918), and range-checks
`top_rate`, `other_rate`, `goss_seed` and `goss_warmup_rounds` only inside that
branch. Outside it the four values are still read onto the wire by
`_parse_goss` in bindings/_mojotrees.mojo:1635-1645, land in a disabled
`GossParams`, and are read by nothing. `top_rate=5.0` with the default
`boosting_type` is accepted, not even range-checked, and produces the model
`top_rate` was never set for.

The reason this is a defect and not a divergence to shrug at is internal
rather than comparative, because the same file already refuses this shape for
CatBoost's bootstrap knobs. `_resolve_bootstrap` refuses `bagging_temperature`
without `bootstrap_type='Bayesian'`, `mvs_reg` without `bootstrap_type='MVS'`,
and `subsample` beside `bootstrap_type='No'`, each by name, on the stated
ground that "a knob that is accepted and does nothing is a silent wrong answer
to the user who set it" (`sampling.check_mvs_bagging_temperature`). GOSS's four
knobs are the hole in that rule.

The fix is a refusal at the surface and nowhere else. It cannot be honored --
outside `boosting_type='goss'` there is no sampler for the knob to configure --
and it cannot be caught in Mojo, because a Mojo caller who builds
`GossParams(False, ...)` wrote the `False` themselves and had nothing ignored.
The exact patch is in the lane report; it must compare `goss_seed` against its
stock default of 3 rather than against the resolved seed, so that a global
`random_state` is not mistaken for a user setting of it.

### D2. The bagged device round sizes its router from `num_leaves`

RAISES, with a message that names a table instead of the parameter that sized
it wrong. train_gpu.mojo:4133-4139 constructs

    GpuTreeRouter(builder.ctx, n, 2 * params.tree.num_leaves)

and this is the one allocation that exists only on the bagged path, guarded by
`if route_all_rows and bagging_enabled(bagging)`. A symmetric tree ignores
`num_leaves` and has `2^(d+1) - 1` nodes, so at the default `num_leaves = 31`
and CatBoost's default depth 6 the router holds 62 nodes against a tree with
127, and `GpuTreeRouter.route` raises "tree has more nodes than the router was
constructed for; construct with a larger max_nodes"
(gpu_fused_round.mojo:475-479).

**This is bug 1's shape one axis over, and the fix already exists in the same
file.** `_state_max_nodes` (train_gpu.mojo:438-467) was written for precisely
this defect on the node-value table, and its docstring says the table "was
missing from that list". The router is the next item on the same list. The edit
is to call it instead of restating the arithmetic. `GpuLeafEstimator` three
lines below has the identical defect on the identical expression; it belongs to
the `leaf_estimation_iterations` family rather than to this one and is handed
off.

On reachability, `route_all_rows` is set only under an explicit
`objective_source=OBJECTIVE_SOURCE_DEVICE` or
`MOJOTREES_GPU_OBJECTIVE=device`, so no shipped default reaches it. It is
still a live wrong answer for the bench arm that sweeps that variable, which
is the arm most likely to be added next.

## Stale claims corrected

Under LANE_RULES rule 7. Each of these was read against the code on
2026-08-17 and was false at head, not merely imprecise.

1. `sampling.mojo`, `catboost_default_bootstrap_type`: "every device trainer
   in this package refuses a bootstrap by name". Corrected in place. `train_gpu`
   takes a bundle and honors it on both arms; the survivors are a list of four
   entry points, not a rule.
2. `sampling.mojo`, module docstring, where the wire paragraph named two round
   loops. There are seven. Corrected and counted.
3. `sampling.mojo`, `check_bootstrap_honored`'s message, which pointed a
   refused user at three CPU entry points, none of them the GPU or sparse or
   multiclass ones that now honor the bundle. Corrected.
4. `sampling.mojo`, `BootstrapRequest.resolve_or_defer`, which quoted a
   `train_gpu` refusal that no longer exists as its example of a trainer's
   own message. Corrected to quote one that does.
5. `sampling.mojo`, "Row sets and the GPU", which said `contiguous_ranges` and
   `row_mask` are "what a device-side active-row pass consumes". Nothing
   consumes them; the device consumes the row list unconverted. Corrected, and
   each of the three unconsumed functions now says so at its own docstring.
6. `goss.mojo`, module docstring, which read "every trainer that takes a
   `bagging` parameter takes a `goss` one too ... The trainers that take
   neither are the custom-objective ones and the ranker". Wrong in both
   directions, since two families of trainer take `bagging` and no `goss`
   (the two `boosting_rf` uniform entry points and the rankers), and the
   ranker's `bagging` samples whole queries. Corrected and enumerated.
7. `boosting.mojo`, `_tree_leaf_values`, which read "the multiclass loops do
   not call this (see `_boost_rounds_multiclass`, which takes no bootstrap
   bundle at all)". That loop takes a bundle and draws it; what it cannot do
   is derive the lambda. Conclusion kept, reason corrected.

Two more are in files this lane may not write and are handed off with exact
replacement text. They are `bench/real_data/frontier.py`'s two accelerator
capability claims, both of which are now false at head, and one comment in
`bindings/_mojotrees.mojo` saying `model.fit` refuses a GPU bootstrap.

## Thread reproducibility

Every draw in this family is a function of `(seed, index, row)` and of nothing
that task decomposition can move. Checked one at a time:

- `bagging.sample_rows`: one serial pass, `uniform(stream + r)` per row, the
  stream a *start* rather than a running state.
- `sampling.sample_rows_by_class`: same shape, plus a tie-break on the smallest
  draw value, which is order-independent because it compares values and not
  positions.
- `goss.goss_select`: one serial forward pass. The acceptance probability
  depends on how many rows have already been taken, so the WALK is ordered, but
  the walk is row order and nothing else; the random numbers are indexed by row.
  `_select_kth_desc` is a median-of-three quickselect with no random pivot, so
  its answer is a function of the input alone.
- `sampling.mvs_bootstrap_weights`: blocks of the comptime constant
  `MVS_BLOCK_SIZE = 8192`, not of a worker count -- the constant carries a
  comment saying that a block size following the worker count would make every
  weight follow it. `_mvs_threshold` uses an explicit three-way partition rather
  than `std::partition`, so its permutation, and therefore its floating-point
  addend order, is a fixed function of row order.
- `sampling.mvs_auto_lambda_from_gradients` / `_from_leaf_values`: serial
  accumulation in row order. CatBoost's own equivalent sums per-thread blocks
  and moves with `thread_count`; ours does not, and that divergence is recorded
  at both functions.
- The inputs. GOSS ranks and MVS thresholds the round's derivatives, so those
  have to be worker-count-invariant too. `boosting._fill_grad_hess` dispatches
  through `dispatch_rows_with` with an elementwise body (boosting.mojo:809), so
  every entry is a function of its own row and the buffers are identical at any
  `MOJOTREES_NUM_WORKERS`.

**No thread-reproducibility defect was found in this family.** The one thing
worth watching is stated rather than assumed. If a future lane parallelizes
`mvs_auto_lambda_from_gradients`'s sum or `goss_importance`'s pass with a
tree-shaped reduction, the lambda and the ranking both become worker-count
dependent, and the ranking one changes which rows a tree is grown on rather
than only the last bits of a value.

## Not verified without a compiler

Ranked by how much a wrong answer would cost.

1. That D2's edit compiles and that `_state_max_nodes` is in scope at
   train_gpu.mojo:4135. It is defined at :438 in the same module, so this is a
   name-resolution question only.
2. That the bagged GPU symmetric fit works once D2 is fixed. The router bound
   is one of two sizes on that path and this lane traced only the sizes; the
   `oblivious_device_supported` gate and the resident pool are separate
   conditions.
3. That Base A of `bench/real_data/frontier.py` actually runs on the
   accelerator now that both of its declared refusals are gone. The refusals
   are gone -- that part is read from source -- but "no longer refused for those
   two reasons" is not "runs".
4. Whether `top_rate` outside `boosting_type='goss'` is reached by any existing
   test, which would turn D1's fix into a test edit as well as a surface edit.
