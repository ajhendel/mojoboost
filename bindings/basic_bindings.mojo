"""Device explanation, run configuration checks, and startup diagnostics.

The questions a caller asks *around* a fit rather than during one:

- where would this run, and why (`decide_device_workload`);
- are these tree parameters ones this build can honor
  (`extra_params_check`, `extra_option_supported`, `forced_splits_check`);
- is this bundling configuration reachable (`efb_check`, `efb_defaults`);
- what does the process know about its own startup (`startup_*`).

Every one of them is answered by native code that already exists. This
module carries no policy, no threshold, no default, and no table: it
converts a Python call into the native call and the native answer back.

Argument shape. Several of these describe a workload or a parameter set
with a dozen fields, and they cross as a Python mapping the way
`_parse_params` in `_mojotrees.mojo` already reads one, rather than as a
dozen positional arguments. Eight positional arguments are proven to
register (`predict_range`); more is untested, and a mapping also lets a
field be added without every caller's call site moving.
`handoffs/performance_15_startup.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_15_startup.md)` states a six-argument cap on
`def_function`, which the eight-argument entry points already in the
module contradict; treat the number above six as unverified either way.

Flags inside a mapping are 0/1 ints, and an undeclared optional is its
documented sentinel, so the boundary converts no Python bool and carries
no optional.
"""

from std.python import Python, PythonObject

from binding_support import f64_buffer, flag, py_dict

from mojotrees.device import decide_device_report
from mojotrees.efb import (
    EfbParams,
    EfbSettings,
    check_bundling_supported,
)
from mojotrees.initialization import (
    N_STARTUP_PHASES,
    StartupTrace,
    env_warmup_level,
    startup_origin_name,
    phase_is_one_time,
    phase_name,
    phase_origin,
    warmup_level_name,
)
from mojotrees.tree_parameters_extra import (
    ExtraTreeParams,
    FeaturePenalties,
    check_extra_option_supported,
    parse_forced_splits,
    parse_monotone_method,
)


# -- device explanation --------------------------------------------------


def decide_device_workload(
    device: PythonObject, workload: PythonObject
) raises -> PythonObject:
    """Resolve a device request over a whole workload and return the
    serialized decision.

    **This is the one device decision entry point.** It is registered as
    `decide_device`, which is the name `device_selection.py` looks for.
    `handoffs/connect_05_device_policy.md (deleted, recover with git log --all --diff-filter=D -- handoffs/connect_05_device_policy.md)` section 5.1 asked for a
    ten-argument version written directly in `_mojotrees.mojo` instead;
    this form was taken over it because a workload has a dozen fields and
    sending them positionally would fix their order in two languages and
    bet on an argument count nothing in the module has tried (eight are
    proven by `predict_range`, ten were never tried). `_FullNativePolicy`
    in device_selection.py sends the mapping. Do not add the positional
    form beside this one: two entry points to one decision is how the two
    drift.

    `device` is the requested name, `"cpu"`, `"gpu"`, or `"auto"`, already
    lowercased by the caller. `workload` is a mapping with:

    | key | meaning |
    | --- | --- |
    | `n_rows`, `n_features` | the training matrix's shape |
    | `n_outputs` | trees per boosting round: 1, or the class count |
    | `n_bins` | the estimator's `max_bin`, or a nonpositive for undeclared |
    | `objective` | a trainer objective code; `-1` is the multiclass marker and IS a declaration, `-2` or below is undeclared |
    | `sparse` | the input is a sparse matrix (0/1) |
    | `categorical` | the run declares categorical features (0/1) |
    | `has_missing` | the run uses missing handling (0/1) |
    | `uses_validation` | the run has an eval set (0/1) |
    | `ordered_boosting` | `boosting_type='ordered'` (0/1) |
    | `score_function` | `split.SCORE_L2` (0) or `split.SCORE_COSINE` (1) |
    | `random_strength` | CatBoost's per-candidate split noise (float) |
    | `derivative_precision` | `DERIV_PRECISION_FLOAT32` (0) or `..._FLOAT64` (1) |
    | `grow_policy` | `GROW_LEAFWISE` (0), `GROW_DEPTHWISE` (1), `GROW_OBLIVIOUS` (2) |
    | `max_depth` | the resolved depth bound; `<= 0` means unlimited |
    | `bundling` | `enable_bundle=True` (0/1) |
    | `linear_tree` | `linear_tree=True` (0/1) |
    | `forced_splits` | the fit declares forced splits (0/1) |

    **THE LAST THREE WERE THE COMMENT BELOW COMING TRUE, AND THEY WERE ADDED
    2026-08-18.** This module used to say, of `bundling`, `linear_tree` and
    `forced_splits`, that they "are NOT sent from Python", and it was right.
    `device_policy` carried BLOCK_FEATURE_BUNDLING, BLOCK_LINEAR_TREE and
    BLOCK_FORCED_SPLITS with keyword defaults of False, so the native request
    always saw False, all three silently never matched, and a user got a raise
    out of the trainer rather than the CPU fallback the policy layer
    implements and tests.
    
    The forced-splits case was the sharpest instance of the class: its raise
    text advised "device='auto', which routes around this", and `auto` did
    not, because the field it would route on never arrived. **The error
    message advertised the disconnected path.** A user following its advice
    reproduced the error.

    **`grow_policy` AND `max_depth` WERE ADDED 2026-08-18, AND THEIR ABSENCE
    WAS A SHIPPING DEFECT RATHER THAN AN OMISSION.** `device_policy` has
    carried `BLOCK_GROW_POLICY` and `BLOCK_MAX_DEPTH` since the oblivious
    device path landed, and the ruling they implement is that `auto` falls
    back to the CPU with a message while an explicit `device="gpu"` raises.
    Neither field crossed this boundary, so `DeviceRequest` took its defaults
    of `GROW_LEAFWISE` and `0`, both blocks were unreachable from Python, and
    a user asking for `grow_policy="symmetrictree", max_depth=8,
    device="auto"` got a hard exception out of `train_gpu` instead of the CPU
    fit the policy layer implements and tests. The graceful path existed, was
    correct, and was disconnected at this call.

    The last nine are required keys, not optional ones, and that is the same
    convention every other key here follows: `device_selection.py` is the only
    sender and it always sends the whole mapping. A key read with a default
    would let a stale sender silently mean "L2, plain boosting", which is the
    answer that needs refusing.

    Returns the `key=value` lines `DeviceDecision.serialize` produces; see
    `serialize` in device_policy.mojo for the format and
    `python/mojotrees/device_selection.py` for the reader.

    It does **not** raise for a workload the GPU path refuses: that
    refusal is `blocked=true` with `message=` saying why, which is what
    lets a caller ask "what would `device='gpu'` do here" without handling
    an exception. It raises for a device name outside the vocabulary and
    for a shape with no rows or no features, which are caller errors
    rather than policy outcomes.

    NO SENTINEL FOLDING HAPPENS HERE. It used to, and that was the bug.
    This function folded `n_bins < 0` and `objective < 0` into the
    undeclared sentinels itself, one statement before calling
    `decide_device_report`, which folds them again through
    `_normalized_bins` and `_normalized_objective` in device_policy.mojo.
    Two marshallers over one wire, and they did not agree: the native one
    folds `objective < -1` and deliberately preserves `-1`, because `-1` is
    `objective_registry.MULTICLASS`, a real code and not an absent one;
    this one folded `objective < 0`, so `-1` and `-2` arrived at the engine
    as the same value and `_normalized_objective`'s documented `-1` branch
    was unreachable from Python. A Python caller could not say "this fit is
    softmax" at all: `Workload(objective="multiclass")` resolves to `-1`
    through the registry and lost it here.

    So the folding is the engine's, in one place, and this function carries
    the ints across unchanged. `-1` (multiclass) and `-2` (undeclared) are
    now distinguishable at this boundary, which is the whole point of there
    being two of them. Do not reintroduce a fold here: a second normalizer
    is how the two came to disagree.

    No device selection happens in this module. `decide_device_report` in
    device.mojo forwards to the one policy engine in device_policy.mojo,
    which detects capabilities itself.
    """
    return PythonObject(
        decide_device_report(
            String(py=device),
            Int(py=workload["n_rows"]),
            Int(py=workload["n_features"]),
            Int(py=workload["n_outputs"]),
            Int(py=workload["n_bins"]),
            Int(py=workload["objective"]),
            flag(workload["sparse"], "sparse"),
            flag(workload["categorical"], "categorical"),
            flag(workload["has_missing"], "has_missing"),
            flag(workload["uses_validation"], "uses_validation"),
            # Keyword-passed because they sit past ten positional arguments,
            # which is further than anything in this module has been carried;
            # `bundling`, `linear_tree` and `forced_splits` sit between them
            # and `uses_validation` in the native signature and are NOT sent
            # from Python, so passing these two positionally would bind them
            # to those three instead. That failure compiles and answers.
            ordered_boosting=flag(
                workload["ordered_boosting"], "ordered_boosting"
            ),
            score_function=Int(py=workload["score_function"]),
            random_strength=Float64(py=workload["random_strength"]),
            derivative_precision=Int(py=workload["derivative_precision"]),
            grow_policy=Int(py=workload["grow_policy"]),
            max_depth=Int(py=workload["max_depth"]),
            bundling=flag(workload["bundling"], "bundling"),
            linear_tree=flag(workload["linear_tree"], "linear_tree"),
            forced_splits=flag(workload["forced_splits"], "forced_splits"),
        )
    )


# -- extra tree parameters -----------------------------------------------


def _penalties(
    params: PythonObject, n_features: Int
) raises -> FeaturePenalties:
    """The per-feature costs from the params mapping.

    `feature_contri`, `cegb_penalty_feature_coupled`, and
    `cegb_penalty_feature_lazy` arrive as float64 buffer addresses with
    `n_features` entries, 0 for absent, which is the convention
    `_parse_monotone` already follows for its per-feature column. The two
    CEGB vectors land on `FeaturePenalties.cegb`, which is a
    `cegb.CegbConfig`: this struct carries the parameters and cegb.mojo
    charges them.
    """
    var out = FeaturePenalties()
    var contri_addr = Int(py=params["feature_contri_addr"])
    if contri_addr != 0:
        out.contri = f64_buffer(contri_addr, n_features)
    out.cegb.tradeoff = Float64(py=params["cegb_tradeoff"])
    out.cegb.penalty_split = Float64(py=params["cegb_penalty_split"])
    var coupled_addr = Int(py=params["cegb_penalty_feature_coupled_addr"])
    if coupled_addr != 0:
        out.cegb.penalty_feature_coupled = f64_buffer(
            coupled_addr, n_features
        )
    var lazy_addr = Int(py=params["cegb_penalty_feature_lazy_addr"])
    if lazy_addr != 0:
        out.cegb.penalty_feature_lazy = f64_buffer(lazy_addr, n_features)
    return out^


def extra_params_from_mapping(
    params: PythonObject, n_features: Int
) raises -> ExtraTreeParams:
    """The extra tree bundle from the params mapping. Reads only; every
    range check is `ExtraTreeParams.check`'s.

    Public because `_mojotrees.mojo` calls it too: `_parse_params` folds the
    result into the `TreeParams` it builds, so the bundle a fit is trained
    with and the bundle `extra_params_check` validates are parsed by the same
    function from the same keys. Two parsers over one mapping is how the
    validator and the trainer would come to disagree about what a parameter
    means, and this is a Mojo-side helper rather than a registered binding:
    it returns an `ExtraTreeParams`, which is not a `PythonObject`.
    """
    var out = ExtraTreeParams()
    out.min_gain_to_split = Float64(py=params["min_gain_to_split"])
    out.max_delta_step = Float64(py=params["max_delta_step"])
    out.path_smooth = Float64(py=params["path_smooth"])
    out.extra_trees = flag(params["extra_trees"], "extra_trees")
    out.extra_seed = Int(py=params["extra_seed"])
    out.monotone_penalty = Float64(py=params["monotone_penalty"])
    out.monotone_method = parse_monotone_method(
        String(py=params["monotone_constraints_method"])
    )
    out.penalties = _penalties(params, n_features)
    var forced = String(py=params["forced_splits"])
    if forced != "":
        out.forced = parse_forced_splits(forced)
    return out^


def extra_params_check(
    params: PythonObject, shape: PythonObject
) raises -> PythonObject:
    """Validate the extra tree parameters against the run they belong to,
    and report what honoring them would require.

    `params` carries the bundle in `src/mojotrees/tree_parameters_extra.mojo`:
    `min_gain_to_split`, `max_delta_step`, `path_smooth`, `extra_trees`
    (0/1), `extra_seed`, `monotone_penalty`,
    `monotone_constraints_method` (LightGBM's name for it),
    `cegb_tradeoff`, `cegb_penalty_split`, the three per-feature buffer
    addresses `feature_contri_addr`,
    `cegb_penalty_feature_coupled_addr`, and
    `cegb_penalty_feature_lazy_addr` (0 for absent), and
    `forced_splits`, the document's text or an empty string.

    `shape` carries `n_features`, `num_leaves`, `max_depth`, and
    `min_data_in_leaf`, because the vector lengths and the growth budget
    are checked against the run actually being fitted.

    Raises with the native message for a value out of range, for a vector
    of the wrong length, for a forced-splits document that does not fit
    the budget, and for `forcedsplits_filename`, which is refused by name
    rather than ignored. The CEGB vectors are not refused here: whether
    the coupled and lazy penalties can be charged is a property of the
    grower, and `cegb.check_cegb_grower_support` answers it at
    `tree._search`.

    Returns the four facts a caller needs to route the fit:
    `is_active` (whether anything here would change it at all),
    `needs_leaf_finish`, `needs_node_identity`, and
    `needs_grower_support` (whether only a grower that opted in can honor
    it). A caller holding `needs_grower_support` and a grower that has not
    opted in should refuse rather than train something else.
    """
    var n_features = Int(py=shape["n_features"])
    var extra = extra_params_from_mapping(params, n_features)
    extra.check(
        n_features,
        Int(py=shape["num_leaves"]),
        Int(py=shape["max_depth"]),
        Int(py=shape["min_data_in_leaf"]),
    )
    var out = py_dict()
    out["is_active"] = PythonObject(extra.is_active())
    out["needs_leaf_finish"] = PythonObject(extra.needs_leaf_finish())
    out["needs_node_identity"] = PythonObject(extra.needs_node_identity())
    out["needs_grower_support"] = PythonObject(extra.needs_grower_support())
    return out^


def extra_option_supported(name: PythonObject) raises -> PythonObject:
    """Raise for a LightGBM tree option that is real but not implemented,
    with the native message saying what it would take.

    Returns None for every other name, including one this repository has
    never heard of: "unknown parameter" is the caller's message about the
    caller's parameter string, and is not this function's to give.
    """
    check_extra_option_supported(String(py=name))
    return PythonObject(None)


def forced_splits_check(
    spec: PythonObject,
    n_features: PythonObject,
    num_leaves: PythonObject,
    max_depth: PythonObject,
) raises -> PythonObject:
    """Validate a forced-splits document and report its shape.

    Returns `n_nodes` and `depth`. This validates the document only:
    applying a forced node means mapping its raw threshold to a bin, and
    the grower is handed a `BinnedMatrix`, which carries no bin edges, so
    `extra_params_check` refuses a run that carries one. Validating is
    still worth doing early, because a forced-splits file is usually
    written long before a fit.
    """
    var forced = parse_forced_splits(String(py=spec))
    forced.check_features(Int(py=n_features))
    forced.check_budget(Int(py=num_leaves), Int(py=max_depth))
    var out = py_dict()
    out["n_nodes"] = PythonObject(forced.n_nodes())
    out["depth"] = PythonObject(forced.depth())
    return out^


# -- exclusive feature bundling ------------------------------------------


def efb_settings_from_mapping(params: PythonObject) raises -> EfbSettings:
    """The bundling switch and the knobs it governs, from the params
    mapping. Reads only; every range check is `EfbSettings.check`'s and
    every reachability check is the caller's.

    Public for the same reason `extra_params_from_mapping` is:
    `_parse_params` in `_mojotrees.mojo` calls it to fill
    `BoosterParams.bundling`, so the settings a fit is trained with and the
    settings `efb_check` validates come from one function reading one set
    of keys. It is a Mojo-side helper rather than a registered binding
    because it returns an `EfbSettings`, which is not a `PythonObject`.

    The keys are flat, not nested: `enable_bundle` (0/1),
    `max_conflict_rate`, `max_bundle_bins`, `max_bundle_size`,
    `max_nondefault_rate`, `min_reduction`, and `bundle_missing` (0/1) sit
    beside the tree parameters in the one mapping a fit already sends.
    """
    return EfbSettings(
        flag(params["enable_bundle"], "enable_bundle"),
        EfbParams(
            Float64(py=params["max_conflict_rate"]),
            Int(py=params["max_bundle_bins"]),
            Int(py=params["max_bundle_size"]),
            Float64(py=params["max_nondefault_rate"]),
            Float64(py=params["min_reduction"]),
            flag(params["bundle_missing"], "bundle_missing"),
        ),
    )


def efb_check(params: PythonObject, cpu: PythonObject) raises -> PythonObject:
    """Validate a bundling configuration against the device that would
    have to honor it.

    `params` carries the flat keys `efb_settings_from_mapping` reads, which
    are the keys a fit already sends. `cpu` is 0/1: whether the run this
    configuration belongs to would go to a CPU trainer, since only those
    apply a plan.

    The order is the native one, the order `params.mojo` checks a parameter
    string in: whether bundling is reachable on this device at all, then
    the ranges, which are checked whether or not the switch is on so that a
    bad value is named before any data is read rather than at the first
    call that happens to turn bundling on. `enable_bundle=1` on a CPU run
    is accepted; on any other device it is refused by name, because a
    silently unbundled fit is a correct model, just not the one that was
    asked for, and nothing in the metrics would show it.
    """
    var settings = efb_settings_from_mapping(params)
    check_bundling_supported(settings.enabled, flag(cpu, "cpu"))
    settings.check()
    return PythonObject(None)


def efb_defaults() raises -> PythonObject:
    """The bundling knobs' defaults, so nobody restates LightGBM's numbers
    in Python. Same keys `efb_check` reads."""
    var d = EfbParams.default()
    var out = py_dict()
    out["max_conflict_rate"] = PythonObject(d.max_conflict_rate)
    out["max_bundle_bins"] = PythonObject(d.max_bundle_bins)
    out["max_bundle_size"] = PythonObject(d.max_bundle_size)
    out["max_nondefault_rate"] = PythonObject(d.max_nondefault_rate)
    out["min_reduction"] = PythonObject(d.min_reduction)
    out["bundle_missing"] = PythonObject(d.bundle_missing)
    return out^


# -- startup diagnostics -------------------------------------------------


def startup_phase_contract() raises -> PythonObject:
    """The startup phases, in report order, as
    `[index, name, origin, one_time]` records.

    The names are the schema `python/mojotrees/diagnostics.py` parses and
    docs/STARTUP_LATENCY.md documents, so this is how that table is
    checked against the native one rather than kept in step by hand.
    `origin` is `"supplied"` for the phases that are over before any Mojo
    code runs and `"native"` for the rest; `one_time` is 0 for the phase a
    second fit pays again and 1 for the phases it does not.
    """
    var out = Python.list()
    for phase in range(N_STARTUP_PHASES):
        var record = Python.list()
        record.append(PythonObject(phase))
        record.append(PythonObject(phase_name(phase)))
        record.append(PythonObject(startup_origin_name(phase_origin(phase))))
        record.append(PythonObject(Int(phase_is_one_time(phase))))
        out.append(record^)
    return out^


def startup_environment() raises -> PythonObject:
    """What the process's environment asked of startup.

    `trace_enabled` is `MOJOTREES_STARTUP_TRACE`, and `warmup_level` is
    `MOJOTREES_GPU_WARMUP` as its name (`off`, `train`, or `all`), with an
    unset or unrecognized value reading as `off` so an unknown value never
    silently front-loads work.

    This reports the request, not a measurement. No trace is collected
    anywhere in this build: `StartupTrace` is deliberately not a
    singleton, so there is no process-wide one to report, and a
    `startup_report()` that returned an empty trace would say "every phase
    took no time" when it means "nobody measured". See the handoff for
    what an owner would have to hold for that call to exist.
    """
    var out = py_dict()
    out["trace_enabled"] = PythonObject(StartupTrace.from_env().enabled)
    out["warmup_level"] = PythonObject(warmup_level_name(env_warmup_level()))
    return out^


def native_clock_ns() raises -> PythonObject:
    """The native monotonic clock, in nanoseconds.

    The same clock a `StartupTrace` times phases with, read through an
    enabled trace so it provably is that clock and not a second one. A
    harness calls this immediately after `import mojotrees` to bound the
    extension load it cannot measure from inside.
    """
    return PythonObject(StartupTrace(True).clock())
