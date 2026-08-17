"""The public Python surface, as data.

This module is the record of what `mojotrees/__init__.py` exports, where
each name comes from, and why. It imports nothing, touches no optional
dependency, and changes no package state: reading it cannot make
`mojotrees` behave differently, and nothing in the package imports it. It
is documentation with a shape, not a mechanism, so a test can compare it
against the real `__all__` and a reader can find the decision behind a
name without reading 4000 lines. That test is
`python/tests/test_public_api_plan.py`: every table below that states a
fact about the code is compared against the code, so a row here goes
stale as a failure rather than as a quiet lie.

Two rules it exists to keep:

- **The package import stays cheap and total.** `import mojotrees` must not
  import dask, and must not fail because dask, pandas, numpy, or
  scikit-learn is missing. Submodules with an optional dependency, or with
  a cost, are reached through a module-level `__getattr__` (PEP 562) rather
  than imported at the top of `__init__.py`. `LAZY_SUBMODULE_SNIPPET` is
  the code, and the end of `__init__.py` is where it now lives.
- **A name means one thing.** Where an export collides with a submodule of
  the same name, the collision is written down here with the decision and
  the alternative, rather than discovered later by whoever types
  `import mojotrees.cv as cv`.

History. Version 1 was a *proposal* covering `mojotrees.cv` and
`mojotrees.dask` only, written by the lane that owned those two modules
(handoffs/integration_06_python_api.md) for a later owner of `__init__.py`
to apply. Version 2 is the state after that owner applied it and finished
the surface: `cv` and `CVBooster` are exported, the `__getattr__` is in
place, and the `inspection` / `device_selection` placeholders have been
replaced by what those lanes actually shipped. See
handoffs/connect_07_python_public.md. Version 3 renames the two tables
that still said `PROPOSED_` after their contents had shipped, adds the
`lgbm_model_io` lazy submodule the integration round wired, and points
at the test that now holds all of it to the code.

Version 4 adds the `features` and `onnx_export` lazy submodules. Both went
into `__init__.py` on 2026-08-16 with the reachability lane (e71e644,
b059a6c) and neither was recorded here, so the drift this module exists to
prevent happened to this module, and it was the first pytest run of
`test_public_api_plan.py` rather than a reader that caught it. `port`,
added the following night, is eager and is in `CURRENT_TOP_LEVEL`; it has
no lazy row and is not what the lazy tables disagreed about.
"""

#: Bumped when the surface below changes, so a handoff can name the version
#: it was written against.
#:
#: 1 -- proposal (cv, CVBooster, the lazy-dask snippet)
#: 2 -- applied, plus inspection / device_selection / diagnostics
#: 3 -- PROPOSED_* tables renamed to what they are, lgbm_model_io added,
#:      enforced by python/tests/test_public_api_plan.py
#: 4 -- features and onnx_export added to the lazy tables, which the
#:      reachability lane had put in __init__.py without recording here
PLAN_VERSION = 4

#: `mojotrees.__all__` as it stands, sorted (the real one is grouped by
#: topic). A stale record shows up as a set mismatch against the real
#: `__all__`, which is the cheapest test this module supports.
CURRENT_TOP_LEVEL = (
    "Booster",
    "CVBooster",
    "CallbackEnv",
    "Dataset",
    "EarlyStopException",
    "MojoTreesClassifier",
    "MojoTreesRanker",
    "MojoTreesRegressor",
    "NotFittedError",
    "build_info",
    "callback",
    "cv",
    "dump_model",
    "early_stopping",
    "explain_device_choice",
    "get_split_value_histogram",
    "gpu_available",
    "group_from_query_ids",
    "log_evaluation",
    "ndcg_score",
    "port",
    "record_evaluation",
    "reset_parameter",
    "show_versions",
    "train",
    "trees_to_dataframe",
    "trees_to_records",
)

#: The `__all__` this module was written against before version 2, kept so
#: that "what changed" is answerable without git.
PREVIOUS_TOP_LEVEL = (
    "Booster",
    "CallbackEnv",
    "Dataset",
    "EarlyStopException",
    "MojoTreesClassifier",
    "MojoTreesRanker",
    "MojoTreesRegressor",
    "NotFittedError",
    "build_info",
    "callback",
    "early_stopping",
    "gpu_available",
    "group_from_query_ids",
    "log_evaluation",
    "ndcg_score",
    "record_evaluation",
    "reset_parameter",
    "show_versions",
    "train",
)

#: Names added to the top level in version 2, with the import that provides
#: each. `eager` is whether the name is a real attribute at the end of
#: `__init__.py`, as opposed to being resolved by `__getattr__` on first
#: access. All of them shipped; the test checks each name against
#: `__all__` and each `eager` flag against a fresh `import mojotrees`.
#:
#: `mojotrees.dask` stays a submodule and exports nothing: its three
#: estimators are contracts that cannot train (see `dask.py`), and
#: exporting them at the top level would put names that raise
#: `DistributedNotAvailable` next to names that fit a model.
TOP_LEVEL_ADDITIONS = (
    {
        "name": "cv",
        "source": "mojotrees.cv",
        "kind": "function",
        "eager": True,
        "statement": "from .cv import CVBooster, cv",
        "why": (
            "LightGBM's spelling is lgb.cv(params, train_set, ...); a "
            "cross-validation helper that can only be reached through a "
            "submodule import is a different API for no reason."
        ),
        "collides_with": "the submodule mojotrees.cv",
    },
    {
        "name": "CVBooster",
        "source": "mojotrees.cv",
        "kind": "class",
        "eager": True,
        "statement": "from .cv import CVBooster, cv",
        "why": (
            "It is what cv(return_cvbooster=True) hands back, so isinstance "
            "checks and type annotations need it where cv() is."
        ),
        "collides_with": None,
    },
    {
        "name": "port",
        "source": "mojotrees.port",
        "kind": "function",
        "eager": True,
        "statement": "from .port import port",
        "why": (
            "The estimator already ACCEPTS every LightGBM, XGBoost and "
            "CatBoost spelling, refuses contradictory duplicates, and "
            "refuses unimplemented objectives with a reason. What it could "
            "not do was tell anyone what it had done, so a user porting a "
            "script found out by trial and error. `port(params, "
            "source=...)` answers it in one call: honored, aliased to, "
            "defaulted differently, or refused with the reason. Eager "
            "because it reads only the repository's own tables and imports "
            "no optional dependency, so it costs an import nobody pays for "
            "twice."
        ),
        "collides_with": (
            "the submodule mojotrees.port, and the collision is REAL rather "
            "than theoretical: the eager export shadows the module, so "
            "`import mojotrees.port as m` binds the FUNCTION and not the "
            "module, and only `importlib.import_module` reaches the module. "
            "That cost python/test_alias_equivalence.py two wrong spellings "
            "before the right one, and both looked correct"
        ),
    },
    {
        "name": "explain_device_choice",
        "source": "mojotrees.device_selection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "'what would device=\"gpu\" do with this data' is a question "
            "asked before training, so the answer should be reachable "
            "without knowing which submodule holds it. The module is a "
            "formatter over the native policy and holds no policy of its "
            "own, so exporting one function from it does not export a "
            "second opinion."
        ),
        "collides_with": None,
    },
    {
        "name": "dump_model",
        "source": "mojotrees.inspection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "LightGBM spells it Booster.dump_model(); mojotrees has that "
            "method too (it delegates here), and the free function is what "
            "takes a model string or a fitted estimator as well."
        ),
        "collides_with": None,
    },
    {
        "name": "trees_to_dataframe",
        "source": "mojotrees.inspection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "The pandas view of the ensemble. Lazy for the same reason as "
            "the rest: pandas is imported by this call and by nothing "
            "else, so importing mojotrees must not reach it."
        ),
        "collides_with": None,
    },
    {
        "name": "trees_to_records",
        "source": "mojotrees.inspection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "The same rows without pandas installed. Exported beside "
            "trees_to_dataframe so that the pandas-free answer is as "
            "findable as the pandas one."
        ),
        "collides_with": None,
    },
    {
        "name": "get_split_value_histogram",
        "source": "mojotrees.inspection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "The data behind LightGBM's plot_split_value_histogram, which "
            "is a top-level name there. No plotting dependency comes with "
            "it here."
        ),
        "collides_with": None,
    },
)

#: Submodules the package answers for without importing them at
#: package-import time, in the order `_LAZY_SUBMODULES` lists them.
#: `optional_dependency` names the third-party import the module degrades
#: from; None means the module is pure mojotrees and is lazy only to keep
#: `import mojotrees` cheap.
#:
#: `mojotrees.dask` is the one that matters: it must be reachable as
#: `mojotrees.dask` after a plain `import mojotrees`, it must not import
#: dask to do that, and it cannot be imported from the top of
#: `__init__.py` because it subclasses estimators that are defined further
#: down that same file. `__getattr__` answers all three at once.
#:
#: `cv` is not here. It is eager (`TOP_LEVEL_ADDITIONS`), so the attribute
#: `mojotrees.cv` is the function and never reaches `__getattr__`; see
#: `NAME_COLLISIONS`.
LAZY_SUBMODULES = (
    {
        "name": "dask",
        "optional_dependency": "dask[distributed]",
        "owner": "this lane",
        "note": (
            "Imports mojotrees's three estimators at module scope because "
            "the Dask estimators subclass them. Safe under __getattr__, "
            "which runs after __init__.py has finished; unsafe as a "
            "top-of-file import."
        ),
    },
    {
        "name": "device_selection",
        "optional_dependency": None,
        "owner": "handoffs/migration_20_device_policy.md",
        "note": (
            "A formatter over the native policy, with no policy of its "
            "own. `_Base._resolve_device` imports it inside the call, so a "
            "fit reaches it whether or not anyone touched the attribute; "
            "the lazy entry is what makes `mojotrees.device_selection` "
            "resolve for a reader."
        ),
    },
    {
        "name": "diagnostics",
        "optional_dependency": None,
        "owner": (
            "handoffs/performance_15_startup.md, "
            "handoffs/connect_05_device_policy.md"
        ),
        "note": (
            "Reads the filesystem rather than importing the extension, "
            "which is what makes it usable from a cold interpreter. "
            "`build_info()` already imports it inside the call; the lazy "
            "entry is for a reader who wants describe_install() or the "
            "startup phases."
        ),
    },
    {
        "name": "features",
        "optional_dependency": "numpy (for the returned columns only)",
        "owner": "lane/reachability-rest, catalog A31",
        "note": (
            "CatBoost's generated features, text_features() and "
            "embedding_features(), over the entry points in "
            "bindings/catboost_reach_bindings.mojo. A transform, not a "
            "trainer: it hands back float64 columns to hstack onto X, and "
            "the fitted dictionary it used is not written into a model "
            "file. Nothing from it is at the top level, because a name "
            "next to train() would read as a fit-time keyword and the "
            "estimator has none."
        ),
    },
    {
        "name": "inspection",
        "optional_dependency": "pandas (for the frame output only)",
        "owner": "handoffs/migration_19_model_inspection.md",
        "note": (
            "Reaches pandas from trees_to_dataframe and "
            "get_split_value_histogram(as_frame=True) and from nowhere "
            "else, so the module itself imports nothing optional. Lazy to "
            "keep `import mojotrees` to the extension, and because "
            "`Booster` (which it inspects) would otherwise be a cycle. "
            "Four of its names are re-exported; the rest of the schema "
            "stays here."
        ),
    },
    {
        "name": "lgbm_model_io",
        "optional_dependency": None,
        "owner": "integration_C0",
        "note": (
            "LightGBM model-file conversion, over the four entry points "
            "in bindings/lgbm_bindings.mojo. Nothing is exported at the "
            "top level (NOT_EXPORTED): the conversion warns once that no "
            "file a real LightGBM build wrote has been read, so it is "
            "asked for by name rather than found next to train()."
        ),
    },
    {
        "name": "onnx_export",
        "optional_dependency": "onnx",
        "owner": "catalog A31, docs/design/MODEL_EXPORT.md",
        "note": (
            "Transcribes the export plan written by "
            "src/mojotrees/onnx_export.mojo into an ONNX ModelProto and "
            "contains no arithmetic of its own. `onnx` is in neither "
            "pixi.toml nor python/pyproject.toml, so the module must not "
            "be reached by `import mojotrees`; a missing onnx raises with "
            "the install line rather than degrading into something that "
            "looks like an export. Nothing from it is at the top level."
        ),
    },
)

#: The shape of the code at the end of `mojotrees/__init__.py`, as a string
#: so that reading this module cannot run it. Version 1 proposed it for
#: `dask` alone; what landed resolves seven submodules and five attributes,
#: and the real one carries the comments explaining each. This is the
#: mechanism, kept here so it can be read without the surrounding 4000
#: lines -- `__init__.py` is the copy that runs, and the test compares the
#: two tables below against it so this copy cannot rot.
LAZY_SUBMODULE_SNIPPET = '''
_LAZY_SUBMODULES = (
    "dask",
    "device_selection",
    "diagnostics",
    "features",
    "inspection",
    "lgbm_model_io",
    "onnx_export",
)

_LAZY_ATTRS = {
    "explain_device_choice": "device_selection",
    "dump_model": "inspection",
    "trees_to_dataframe": "inspection",
    "trees_to_records": "inspection",
    "get_split_value_histogram": "inspection",
}


def __getattr__(name):
    import importlib

    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # answered directly from here on
        return module
    origin = _LAZY_ATTRS.get(name)
    if origin is not None:
        value = getattr(
            importlib.import_module(f".{origin}", __name__), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_ATTRS)
    )
'''

#: The one name two things want. Written down with the decision, because
#: the failure mode is silent: `import mojotrees.cv as m` binds the
#: function, and `m.CVBooster` then raises AttributeError on a line that
#: looks like a module import. The decision is applied, not pending.
NAME_COLLISIONS = (
    {
        "attribute": "mojotrees.cv",
        "candidates": ("the cv() function", "the mojotrees.cv module"),
        "decision": "the function",
        "consequences": (
            "mojotrees.cv(params, train_set, ...) is the call. "
            "from mojotrees.cv import cv, CVBooster still works, because "
            "that form resolves the submodule through sys.modules rather "
            "than through the package attribute. "
            "import mojotrees.cv as m binds m to the function; use "
            "from mojotrees import cv, or import the names directly."
        ),
        "alternatives": (
            "Rename the module to mojotrees/engine.py, which is where "
            "LightGBM keeps train() and cv(), and export from there: no "
            "collision at all. It needs a lane that owns both cv.py and "
            "python/tests/parallel/test_cv.py, which imports mojotrees.cv "
            "by name. Recommended as the end state, not as part of this "
            "integration.",
            "Do not export the function, and leave callers with "
            "from mojotrees.cv import cv. No collision, and a gratuitous "
            "difference from LightGBM.",
        ),
        "precedent": (
            "mojotrees.callback is already both a submodule and an "
            "__all__ entry, and does not collide, because no function is "
            "named callback. LightGBM has lightgbm.cv the function and no "
            "lightgbm/cv.py, so it never had to make this choice."
        ),
    },
)

#: Public-looking names in the reachable submodules that are deliberately
#: NOT at the top level, so that "it is missing" is answerable.
NOT_EXPORTED = (
    {
        "name": "mojotrees.inspection.parse_model_string",
        "why": (
            "The compatibility parser behind the DELETION POINT banner in "
            "inspection.py: it rebuilds the schema from the model text for "
            "builds without the native dump, and goes away with that "
            "banner. Exporting it would make a temporary thing look "
            "permanent. The rest of inspection's __all__ is reachable as "
            "mojotrees.inspection.<name>."
        ),
    },
    {
        "name": "mojotrees.device_selection.select_device",
        "why": (
            "And Workload, DeviceReport, native_contract, and the two "
            "exception types. explain_device_choice is the question a user "
            "asks; the rest is the vocabulary an estimator and a bug "
            "report use, and it stays one import away."
        ),
    },
    {
        "name": "mojotrees.cv.FoldModel",
        "why": (
            "An internal adapter over one fold. Named without an "
            "underscore because the module docstring refers to it; it is "
            "not in cv.__all__ and is not a supported extension point."
        ),
    },
    {
        "name": "mojotrees.dask.DaskMojoTreesRegressor",
        "why": (
            "And the classifier and the ranker. fit() raises "
            "DistributedNotAvailable on every installation: no transport "
            "ships. A top-level export would read as a feature. Revisit "
            "when handoffs/task17_dask.md's checklist is complete."
        ),
    },
    {
        "name": "mojotrees.lgbm_model_io.load_lightgbm_model",
        "why": (
            "And save_lightgbm_model, convert_to_mojotrees, "
            "convert_from_mojotrees, interop_status, and unsupported_reason. "
            "The module carries EXPERIMENTAL = True and warns once that no "
            "file a real LightGBM build wrote has been read here and no "
            "file written here has been read back by LightGBM. A top-level "
            "load_lightgbm_model would read as interop that has been "
            "proven. Revisit when the status text in "
            "src/mojotrees/lgbm_model_io.mojo flips."
        ),
    },
    {
        "name": "mojotrees.dask.register_backend",
        "why": (
            "And the rest of the backend protocol (CAPABILITIES, "
            "TrainingJob, WorldPlan, PartitionMeta, ...). It is a contract "
            "between mojotrees and a transport author, not something an "
            "application calls. It stays in dask.__all__."
        ),
    },
)

#: Where mojotrees's public Python surface differs from LightGBM's on
#: purpose, for the two modules this lane owns. Each row is a difference a
#: user could hit, not an internal detail.
LIGHTGBM_DIFFERENCES = (
    {
        "area": "cv",
        "lightgbm": "results holds only the metric lists",
        "mojotrees": 'results["iterations"] gives the round of each entry',
        "why": (
            "A ranking cv reports one round, so the length of a history "
            "list is not the round count and a caller should not have to "
            "assume it is."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "every round of every objective is scored",
        "mojotrees": (
            "a ranking cv reports the final round only, and refuses "
            "callbacks and early_stopping_rounds"
        ),
        "why": (
            "Booster.update() does not cover LambdaRank, so a ranking fold "
            "is trained once at the full round count instead of grown. "
            "Refused rather than accepted and ignored. Lifting it is "
            "native work: handoffs/task15_cv.md."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "callbacks receive a 5-tuple carrying the stdev",
        "mojotrees": "callbacks receive (name, metric, mean, higher)",
        "why": (
            "mojotrees.callback unpacks four. The deviation is in the "
            "returned history instead of in the log line, so "
            "log_evaluation(show_stdv=...) has nothing to show."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "reset_parameter() schedules a hyperparameter change",
        "mojotrees": "CVBooster.reset_parameter raises NotImplementedError",
        "why": (
            "Booster has no way to be told, so the schedule would look "
            "like it ran and would not have."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "init_model continues training the folds",
        "mojotrees": "init_model is refused",
        "why": (
            "Continued training checks that the dataset is the one the "
            "model was trained on by comparing the binning, and each fold "
            "bins itself over its own rows, so every fold would be "
            "rejected by the trainer."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "lightgbm.dask trains across a cluster",
        "mojotrees": "mojotrees.dask cannot train; fit() raises",
        "why": (
            "The transport is unfinished and no backend is registered. "
            "The module is the client-side contract for one: metadata "
            "validation, the rank plan, capability negotiation, model "
            "ownership, and partition-local prediction."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "DaskLGBMClassifier.fit infers the classes from y",
        "mojotrees": "DaskMojoTreesClassifier.fit requires classes=",
        "why": (
            "The class list is a global fact. A metadata scan that missed "
            "a rare class would give two ranks different label encodings "
            "and no error."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "the ranker takes group",
        "mojotrees": "the ranker takes query_ids, one sequence per partition",
        "why": (
            "A group array cannot say whether the query at a partition's "
            "edge continues into the next partition, which is the rule "
            "that has to be checked."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "predict accepts pred_leaf / pred_contrib",
        "mojotrees": "both are refused on a Dask collection",
        "why": (
            "The output width depends on the model, and the collection's "
            "metadata is written before the model is consulted. Compute "
            "the partition and predict on it locally."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "sample weighting by class is available",
        "mojotrees": 'class_weight="balanced" is refused on a Dask fit',
        "why": (
            "It needs a global row count per class, which the client does "
            "not have and a rank cannot compute from its own rows. An "
            "explicit dict is accepted and asks the backend for the "
            "class_weight capability."
        ),
    },
)
