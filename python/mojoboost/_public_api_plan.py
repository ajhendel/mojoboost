"""The public Python surface, as data.

This module is the record of what `mojoboost/__init__.py` exports, where
each name comes from, and why. It imports nothing, touches no optional
dependency, and changes no package state: reading it cannot make
`mojoboost` behave differently, and nothing in the package imports it. It
is documentation with a shape, not a mechanism, so a test can compare it
against the real `__all__` and a reader can find the decision behind a
name without reading 4000 lines.

Two rules it exists to keep:

- **The package import stays cheap and total.** `import mojoboost` must not
  import dask, and must not fail because dask, pandas, numpy, or
  scikit-learn is missing. Submodules with an optional dependency, or with
  a cost, are reached through a module-level `__getattr__` (PEP 562) rather
  than imported at the top of `__init__.py`. `LAZY_SUBMODULE_SNIPPET` is
  the code, and the end of `__init__.py` is where it now lives.
- **A name means one thing.** Where an export collides with a submodule of
  the same name, the collision is written down here with the decision and
  the alternative, rather than discovered later by whoever types
  `import mojoboost.cv as cv`.

History. Version 1 was a *proposal* covering `mojoboost.cv` and
`mojoboost.dask` only, written by the lane that owned those two modules
(handoffs/integration_06_python_api.md) for a later owner of `__init__.py`
to apply. Version 2 is the state after that owner applied it and finished
the surface: `cv` and `CVBooster` are exported, the `__getattr__` is in
place, and the `inspection` / `device_selection` placeholders have been
replaced by what those lanes actually shipped. See
handoffs/connect_07_python_public.md.
"""

#: Bumped when the surface below changes, so a handoff can name the version
#: it was written against.
#:
#: 1 -- proposal (cv, CVBooster, the lazy-dask snippet)
#: 2 -- applied, plus inspection / device_selection / diagnostics
PLAN_VERSION = 2

#: `mojoboost.__all__` as it stands, sorted (the real one is grouped by
#: topic). A stale record shows up as a set mismatch against the real
#: `__all__`, which is the cheapest test this module supports.
CURRENT_TOP_LEVEL = (
    "Booster",
    "CVBooster",
    "CallbackEnv",
    "Dataset",
    "EarlyStopException",
    "MojoBoostClassifier",
    "MojoBoostRanker",
    "MojoBoostRegressor",
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
    "MojoBoostClassifier",
    "MojoBoostRanker",
    "MojoBoostRegressor",
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
#: access.
#:
#: `mojoboost.dask` stays a submodule and exports nothing: its three
#: estimators are contracts that cannot train (see `dask.py`), and
#: exporting them at the top level would put names that raise
#: `DistributedNotAvailable` next to names that fit a model.
PROPOSED_ADDITIONS = (
    {
        "name": "cv",
        "source": "mojoboost.cv",
        "kind": "function",
        "eager": True,
        "statement": "from .cv import CVBooster, cv",
        "why": (
            "LightGBM's spelling is lgb.cv(params, train_set, ...); a "
            "cross-validation helper that can only be reached through a "
            "submodule import is a different API for no reason."
        ),
        "collides_with": "the submodule mojoboost.cv",
    },
    {
        "name": "CVBooster",
        "source": "mojoboost.cv",
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
        "name": "explain_device_choice",
        "source": "mojoboost.device_selection",
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
        "source": "mojoboost.inspection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "LightGBM spells it Booster.dump_model(); mojoboost has that "
            "method too (it delegates here), and the free function is what "
            "takes a model string or a fitted estimator as well."
        ),
        "collides_with": None,
    },
    {
        "name": "trees_to_dataframe",
        "source": "mojoboost.inspection",
        "kind": "function",
        "eager": False,
        "statement": "resolved by __getattr__ through _LAZY_ATTRS",
        "why": (
            "The pandas view of the ensemble. Lazy for the same reason as "
            "the rest: pandas is imported by this call and by nothing "
            "else, so importing mojoboost must not reach it."
        ),
        "collides_with": None,
    },
    {
        "name": "trees_to_records",
        "source": "mojoboost.inspection",
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
        "source": "mojoboost.inspection",
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

#: Submodules the package should answer for without importing them at
#: package-import time. `optional_dependency` names the third-party import
#: the module degrades from; None means the module is pure mojoboost and is
#: lazy only to keep `import mojoboost` cheap.
#:
#: `mojoboost.dask` is the one that matters: it must be reachable as
#: `mojoboost.dask` after a plain `import mojoboost`, it must not import
#: dask to do that, and it cannot be imported from the top of
#: `__init__.py` because it subclasses estimators that are defined further
#: down that same file. `__getattr__` answers all three at once.
PROPOSED_LAZY_SUBMODULES = (
    {
        "name": "dask",
        "optional_dependency": "dask[distributed]",
        "owner": "this lane",
        "note": (
            "Imports mojoboost's three estimators at module scope because "
            "the Dask estimators subclass them. Safe under __getattr__, "
            "which runs after __init__.py has finished; unsafe as a "
            "top-of-file import."
        ),
    },
    {
        "name": "cv",
        "optional_dependency": None,
        "owner": "this lane",
        "note": (
            "Reached eagerly anyway by PROPOSED_ADDITIONS, so it needs no "
            "__getattr__ entry. Listed for completeness: after the function "
            "is exported, the attribute mojoboost.cv is the function."
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
            "keep `import mojoboost` to the extension, and because "
            "`Booster` (which it inspects) would otherwise be a cycle. "
            "Four of its names are re-exported; the rest of the schema "
            "stays here."
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
            "the lazy entry is what makes `mojoboost.device_selection` "
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
)

#: The shape of the code at the end of `mojoboost/__init__.py`, as a string
#: so that reading this module cannot run it. Version 1 proposed it for
#: `dask` alone; what landed resolves four submodules and five attributes,
#: and the real one carries the comments explaining each. This is the
#: mechanism, kept here so it can be read without the surrounding 4000
#: lines -- `__init__.py` is the copy that runs.
LAZY_SUBMODULE_SNIPPET = '''
_LAZY_SUBMODULES = ("dask", "device_selection", "diagnostics", "inspection")

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
#: the failure mode is silent: `import mojoboost.cv as m` binds the
#: function, and `m.CVBooster` then raises AttributeError on a line that
#: looks like a module import. The decision is applied, not pending.
NAME_COLLISIONS = (
    {
        "attribute": "mojoboost.cv",
        "candidates": ("the cv() function", "the mojoboost.cv module"),
        "decision": "the function",
        "consequences": (
            "mojoboost.cv(params, train_set, ...) is the call. "
            "from mojoboost.cv import cv, CVBooster still works, because "
            "that form resolves the submodule through sys.modules rather "
            "than through the package attribute. "
            "import mojoboost.cv as m binds m to the function; use "
            "from mojoboost import cv, or import the names directly."
        ),
        "alternatives": (
            "Rename the module to mojoboost/engine.py, which is where "
            "LightGBM keeps train() and cv(), and export from there: no "
            "collision at all. It needs a lane that owns both cv.py and "
            "python/tests/parallel/test_cv.py, which imports mojoboost.cv "
            "by name. Recommended as the end state, not as part of this "
            "integration.",
            "Do not export the function, and leave callers with "
            "from mojoboost.cv import cv. No collision, and a gratuitous "
            "difference from LightGBM.",
        ),
        "precedent": (
            "mojoboost.callback is already both a submodule and an "
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
        "name": "mojoboost.inspection.parse_model_string",
        "why": (
            "The compatibility parser behind the DELETION POINT banner in "
            "inspection.py: it rebuilds the schema from the model text for "
            "builds without the native dump, and goes away with that "
            "banner. Exporting it would make a temporary thing look "
            "permanent. The rest of inspection's __all__ is reachable as "
            "mojoboost.inspection.<name>."
        ),
    },
    {
        "name": "mojoboost.device_selection.select_device",
        "why": (
            "And Workload, DeviceReport, native_contract, and the two "
            "exception types. explain_device_choice is the question a user "
            "asks; the rest is the vocabulary an estimator and a bug "
            "report use, and it stays one import away."
        ),
    },
    {
        "name": "mojoboost.cv.FoldModel",
        "why": (
            "An internal adapter over one fold. Named without an "
            "underscore because the module docstring refers to it; it is "
            "not in cv.__all__ and is not a supported extension point."
        ),
    },
    {
        "name": "mojoboost.dask.DaskMojoBoostRegressor",
        "why": (
            "And the classifier and the ranker. fit() raises "
            "DistributedNotAvailable on every installation: no transport "
            "ships. A top-level export would read as a feature. Revisit "
            "when handoffs/task17_dask.md's checklist is complete."
        ),
    },
    {
        "name": "mojoboost.dask.register_backend",
        "why": (
            "And the rest of the backend protocol (CAPABILITIES, "
            "TrainingJob, WorldPlan, PartitionMeta, ...). It is a contract "
            "between mojoboost and a transport author, not something an "
            "application calls. It stays in dask.__all__."
        ),
    },
)

#: Where mojoboost's public Python surface differs from LightGBM's on
#: purpose, for the two modules this lane owns. Each row is a difference a
#: user could hit, not an internal detail.
LIGHTGBM_DIFFERENCES = (
    {
        "area": "cv",
        "lightgbm": "results holds only the metric lists",
        "mojoboost": 'results["iterations"] gives the round of each entry',
        "why": (
            "A ranking cv reports one round, so the length of a history "
            "list is not the round count and a caller should not have to "
            "assume it is."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "every round of every objective is scored",
        "mojoboost": (
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
        "mojoboost": "callbacks receive (name, metric, mean, higher)",
        "why": (
            "mojoboost.callback unpacks four. The deviation is in the "
            "returned history instead of in the log line, so "
            "log_evaluation(show_stdv=...) has nothing to show."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "reset_parameter() schedules a hyperparameter change",
        "mojoboost": "CVBooster.reset_parameter raises NotImplementedError",
        "why": (
            "Booster has no way to be told, so the schedule would look "
            "like it ran and would not have."
        ),
    },
    {
        "area": "cv",
        "lightgbm": "init_model continues training the folds",
        "mojoboost": "init_model is refused",
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
        "mojoboost": "mojoboost.dask cannot train; fit() raises",
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
        "mojoboost": "DaskMojoBoostClassifier.fit requires classes=",
        "why": (
            "The class list is a global fact. A metadata scan that missed "
            "a rare class would give two ranks different label encodings "
            "and no error."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "the ranker takes group",
        "mojoboost": "the ranker takes query_ids, one sequence per partition",
        "why": (
            "A group array cannot say whether the query at a partition's "
            "edge continues into the next partition, which is the rule "
            "that has to be checked."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "predict accepts pred_leaf / pred_contrib",
        "mojoboost": "both are refused on a Dask collection",
        "why": (
            "The output width depends on the model, and the collection's "
            "metadata is written before the model is consulted. Compute "
            "the partition and predict on it locally."
        ),
    },
    {
        "area": "dask",
        "lightgbm": "sample weighting by class is available",
        "mojoboost": 'class_weight="balanced" is refused on a Dask fit',
        "why": (
            "It needs a global row count per class, which the client does "
            "not have and a rank cannot compute from its own rows. An "
            "explicit dict is accepted and asks the backend for the "
            "class_weight capability."
        ),
    },
)
