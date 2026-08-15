# consolidation K7 - Python package split

Round: consolidation (2026-08-15). Lane owns python/mojotrees/ and
compatibility/api_snapshot.json only.

## Guard

- c66b997 Freeze the Python API snapshot: `compatibility/api_snapshot.json`
  written once by `tools/api_snapshot.py --write` on the untouched tree.
  Never regenerated afterward.
- `--check` exits nonzero on every run, before and after this lane, for one
  reason only: `PROBLEM: I8: documented environment variables that nothing
  reads: MOJOTREES_ABI_VERSION`. That name is the C header's version
  `#define`, which the environment scanner in tools/api_snapshot.py picks up
  as a documented `MOJOTREES_*` env var. It is a tools/ false positive, not
  a package problem, and tools/ is not this lane's. Pass criterion used
  here: zero difference rows and no problem other than I8. That held after
  every extraction. Coordinator: fix the scanner (exclude the C header, or
  the `_ABI_VERSION` suffix) so `--check` can be a real gate.

## Extractions (one commit each, explicit paths, snapshot green after each)

| commit  | new module                    | moved from `__init__.py`                                                                                                        |
|---------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| 087f1a7 | `mojotrees/_environment.py`   | `gpu_available`, `_OPTIONAL_DEPENDENCIES`, `_distribution_version`, `_optional_dependency_versions`, `_install_layout`, `_BUILD_INFO_FILE`, `_build_provenance`, `build_info`, `show_versions` |
| ccfe358 | `mojotrees/_fit_args.py`      | `_IMPORTANCE_TYPES`, `_DEVICES`, `_BOOSTING_TYPES`, `_NO_DEVICE_PREDICT`, `_UNIMPLEMENTED_OBJECTIVES`, `_unimplemented_objective_note`, `_GROW_POLICIES`, `_MAX_RELEVANCE_LABEL`, `_as_iteration`, `_store_vector`, `_metric_spec`, `_metric_specs`, `_primary_index`, `_eval_pairs`, `_per_set`, `_encode_like`, `_early_stopping_rounds`, `_check_eval_arguments`, `_device_name` |
| d7044e3 | `mojotrees/_ranking.py`       | `group_from_query_ids`, `_group_buffer`, `_check_relevance`, `ndcg_score`                                                       |

Every moved name is bound back into the package namespace by an explicit
`from ._x import (...)` in `__init__.py`, so `mojotrees.<name>` and the
in-package `from . import _as_iteration` (basic.py) / `from . import
_metric_specs` (cv.py) are unchanged. `build_info()["package"]` still names
the package `__init__.py` (the module reads `__version__` and `__file__`
off the package object). Checked after each step: `--check` (no rows),
`import mojotrees` smoke, and direct calls of the moved functions.
pytest is not in the default pixi env, so no python test file was run.

`__init__.py`: 4,615 -> 3,936 lines. What remains: the 305-line module
docstring, imports and `__all__`, the objective-code literals and
`_LAMBDA_L1` / `_LAMBDA_L2`, `_Base` (2,100 lines), `MojoTreesRegressor`,
`MojoTreesClassifier`, `MojoTreesRanker`, the `cv` import, and the PEP 562
lazy tail.

## Why the estimators did not move (the real blocker)

`tools/api_snapshot.py::python_block` statically parses
`python/mojotrees/__init__.py` (`PY_API`) and reads `_Base`, its
`__init__` signature, `_FITTED_ATTRS`, the `_resolve_alias` call sites, and
each estimator class (`class_named(tree, "_Base")`, gap "class is gone")
from that file's AST. Moving `_Base` and the three estimators to a
`mojotrees/sklearn.py` therefore fails `--check` with structural gaps no
extraction can avoid, and regenerating the snapshot to absorb that is
forbidden. That is also why the objective codes and `_LAMBDA_*` literals
stayed put: `module_constants(tree)` resolves `_OBJECTIVES` values and
`lambda_l1=_LAMBDA_L1` defaults from `__init__.py` literals.

Phase 2, for whoever owns tools/: point the estimator derivation at the
estimator module (e.g. `PY_ESTIMATORS = PY_PKG / "sklearn.py"`, falling
back to `PY_API` when it does not exist) and derive `module_constants` from
that file too; the emitted snapshot is unchanged by construction, so
`--check` stays green across the move. Then move `_Base` + the three
estimators + the objective-code and lambda literals to `sklearn.py` in one
commit and re-export from `__init__.py`. Splitting `_Base` across two files
to dodge the parser was considered and rejected: it lowers cohesion, which
is the opposite of the lane's purpose.

## _public_api_plan.py: kept, not deleted

`CURRENT_TOP_LEVEL == sorted(mojotrees.__all__)` is True, so the plan is
fully executed. Not deleted because it is cited as documentation by
docs/LIGHTGBM_PARITY.md:635, docs/INTEGRATION_INVENTORY.md:244,
docs/CONNECTION_AUDIT.md:325,445, docs/ECOSYSTEM_INPUTS.md:52, two
`__init__.py` comments (name collisions), dask.py:61, and the audit
CLASSIFICATION table; those files are C0's / the parity lane's, and the
audit's "paths named by documents that are not there" section would light
up. Its `NAME_COLLISIONS`, `NOT_EXPORTED`, and `LIGHTGBM_DIFFERENCES`
tables are the only written record of why each name is or is not exported.
Recommendation: keep as documentation-as-data (already classified
EXPERIMENTAL in the audit), or delete it together with those citations in
one coordinator commit.

## For the audit CLASSIFICATION table (python modules the root never reaches)

- `mojotrees.dask`: EXPERIMENTAL, intentional PEP 562 lazy submodule
  (`_LAZY_SUBMODULES`); importing it must not import dask, and it exports
  nothing at top level by decision (`_public_api_plan.NOT_EXPORTED`).
- `mojotrees._dask_runtime`: EXPERIMENTAL, reached only from `dask.py`
  inside functions, by design (backend contract).
- `mojotrees._arrow`, `mojotrees._polars`: PENDING, reached only from the
  parked `_sequence.py` (K4); connect together with it.
- `mojotrees._validation`: PENDING, nothing in the package imports it
  (structure checks that `_arrays.check_X` may absorb; connect_07's call).

## Deferred

- Estimators to `sklearn.py`: blocked on the tools/api_snapshot.py change
  above.
- The 305-line module docstring at the top of `__init__.py` was left as is;
  moving prose is not consolidation.
- Unused stdlib imports left in `__init__.py` (`_json`, `_tempfile`,
  `_array`, `_warnings`, `_operator`) were not audited for use inside the
  estimator bodies; a linter pass belongs with the estimator move.

## Phase 2 (after C0's 8e214a2 fixed both tools/ blockers)

- 2e1b26a `python/mojotrees/sklearn.py`: `_Base`, `MojoTreesRegressor`,
  `MojoTreesClassifier`, `MojoTreesRanker`, the objective-code literals
  (`_SQUARED_ERROR` .. `_CROSS_ENTROPY`) and `_LAMBDA_L1` / `_LAMBDA_L2`.
  `__init__.py` re-binds all of them at the root; `__all__` unchanged. The
  six stdlib imports only the estimators used (`array`, `json`,
  `operator`, `os`, `tempfile`, `warnings`) were dropped from
  `__init__.py` after confirming nothing reaches them through the package.
- `__init__.py`: 545 lines (from 4,615 at the start of the lane).
- `python3 tools/api_snapshot.py --check`: no `python.*` difference rows.
  The remaining rows on the tree (`versions.*` 0.1.0a2 -> 0.1.0 and
  `mojo.exports_by_module` growth_policy / levelwise_policy) are a live
  peer's uncommitted work, per C0; the snapshot was not regenerated.
- Verified: `import mojotrees`, the classes at the root and in
  `mojotrees.sklearn` are the same objects, a fresh estimator pickles and
  unpickles, `CURRENT_TOP_LEVEL == __all__` still holds.
- One observable, non-contract change to know about: the classes'
  `__module__` is now `mojotrees.sklearn`, so a pickle written from this
  build names `mojotrees.sklearn.MojoTreesRegressor` (a pickle written
  before names `mojotrees.MojoTreesRegressor`, which still resolves through
  the root binding). Snapshot does not record `__module__`; reprs use
  `__name__` only.
- `_public_api_plan.py` left in place for C0.
