# Compatibility and release contract

What a mojotrees release promises, what it does not promise, and what has
to be true before one is cut.

## Status of this document

mojotrees is at version 0.1.0. There is no git tag, no published release,
and no PyPI distribution. Nothing in this document is a statement that the
project is ready for 1.0, and no rule here should be read as one. This is
the contract the project intends to keep from its first tagged release
onward, written down now so that the first release can be judged against
it rather than against a memory of what was meant.

Two things follow from that. The policy is normative for future releases
and descriptive of present intent, and where the code does not yet meet it
the gap is named in this document rather than smoothed over.

The parity statuses in [docs/LIGHTGBM_PARITY.md](LIGHTGBM_PARITY.md) are
governed by that document and by `tools/check_parity.py`. This document
does not set, raise, or lower any of them. A release gate that fails here
means a release is blocked, not that a parity row changed.

## 1. Versioning

### 1.1 The scheme

mojotrees versions are [Semantic Versioning 2.0.0](https://semver.org).
The version lives in three places that must agree, and a release where
they disagree is not a release.

| Location | Field |
|---|---|
| `pixi.toml` | `[workspace] version` |
| `python/pyproject.toml` | `[project] version` |
| `python/mojotrees/__init__.py` | `__version__` |

### 1.2 What the numbers mean after 1.0

- **Major.** A change that can break a program written against the
  previous release, in any of the surfaces enumerated in section 2.
- **Minor.** New surface. Anything a program written against the previous
  release still compiles, links, loads, and runs against.
- **Patch.** Bug fixes and internal changes with no surface change.

A change to a numeric result is a compatibility change even when no
signature moves. Section 8 states which numeric results are covered.

### 1.3 What the numbers mean before 1.0

Under SemVer, 0.x makes no compatibility promise at all. mojotrees narrows
that voluntarily. While the major version is 0:

- a breaking change may land in a minor bump, `0.N` to `0.N+1`
- a patch bump, `0.N.P` to `0.N.P+1`, never breaks a documented surface
- every break, in either direction, is listed in the release notes with
  the migration, in the format of section 3.4

The narrowing is a discipline, not a guarantee, and it is the reason the
snapshot manifest of section 11 exists before 1.0 rather than after.

### 1.4 Version numbers that move independently

Five numbers version five different things. Bumping one does not bump the
others, and a release note that changes any of them says so explicitly.

| Number | Where | Current | Bumps when |
|---|---|---|---|
| Library version | `pixi.toml`, `pyproject.toml`, `__version__` | 0.1.0 | Any release |
| C ABI version | `MOJOTREES_ABI_VERSION` in `capi/mojotrees.h` | 1 | A declaration in the header changes incompatibly |
| Model format version | `_VERSION` in `src/mojotrees/serialize.mojo` | v4 | The file format gains or changes a section |
| Dump schema version | `DUMP_FORMAT_VERSION` in `python/mojotrees/inspection.py` | 1 | A dump key is removed, retyped, or given a new meaning |
| Snapshot schema version | `schema_version` in `compatibility/api_snapshot_manifest_v1.json` | 1 | The manifest's own shape changes |

The model format version and the dump schema version are independent and
both appear in a dump, which is deliberate. `model_format_version` says
which optional facts a model of that vintage can carry at all, and
`dump_format_version` says what the dump's own keys mean.

## 2. What is public

A surface is public if it is listed here. Everything else is internal and
may change in any release, including a patch release.

**Public.**

1. The names in `mojotrees.__all__` and their documented behavior
   (`python/mojotrees/__init__.py`).
2. The public methods and attributes of `MojoTreesRegressor`,
   `MojoTreesClassifier`, `MojoTreesRanker`, `Booster`, and `Dataset`,
   excluding any name that starts with an underscore.
3. The fitted attributes of section 5.
4. The names re-exported from `src/mojotrees/__init__.mojo`.
5. The declarations in `capi/mojotrees.h`.
6. The command line interface of `cli/mojotrees`, including its exit
   statuses.
7. The LightGBM style parameter string accepted by `parse_params`, which
   is the training surface of both the C ABI and the CLI.
8. The model file format written by `save_model` and its variants.
9. The `MOJOTREES_*` environment variables of section 9.5.

**Internal, explicitly.**

- Every name beginning with `_`, in Python and in Mojo, wherever it lives.
  This includes `python/mojotrees/_arrays.py`, `_eval.py`, and `_sklearn.py`
  in their entirety, and the frozen internal helpers in
  `src/mojotrees/boosting.mojo` (`_fill_grad_hess`, `_base_score`,
  `_check_objective`, `_check_sample_weight`, `_renew_leaf_values`), which
  are frozen for the benefit of in-tree callers and not for external ones.
- Anything under `bench/`, `tools/`, `packaging/`, and `launch/`.
- Module layout inside `src/mojotrees/` and inside `python/mojotrees/`.
  Importing `mojotrees.basic` is supported because `Booster`, `Dataset`,
  and `train` are re-exported at the top level; importing any other
  submodule by path is not.
- Benchmark numbers. Every performance figure in this repository describes
  one machine on one day and is not a promise about the next release.

## 3. Deprecation

### 3.1 Periods

A public name is removed only after a deprecation period. The period is
counted in minor releases and in calendar time, and both have to elapse.

| Surface | Minimum notice | Calendar minimum |
|---|---|---|
| Python name, argument, or documented behavior | 2 minor releases | 90 days |
| Mojo export | 2 minor releases | 90 days |
| C ABI declaration | 1 major release | 180 days |
| Parameter name or alias | 2 minor releases | 90 days |
| Model format section | Never removed; see section 7.3 | n/a |
| Environment variable | 2 minor releases | 90 days |

Before 1.0 the calendar minimums are advisory rather than binding, because
the release cadence is not yet established. The minor-release counts are
binding now.

### 3.2 Mechanics

- **Python.** `DeprecationWarning` from the deprecated path, naming the
  replacement and the release the removal lands in. The warning is raised
  once per call site, never in a per-row path.
- **Mojo.** A deprecation note in the docstring and in the release notes.
  Mojo has no runtime warning channel that is free of cost in a hot loop,
  so the note is the mechanism.
- **C ABI.** The old declaration stays in the header, keeps working, and
  is marked deprecated in its comment. Removing it is a major release and
  a `MOJOTREES_ABI_VERSION` bump.
- **Parameter names.** The old name keeps working and reports the
  replacement. A parameter string containing a removed name reports the
  replacement rather than the generic unknown-key message, for as long as
  the message is useful.

### 3.3 What is exempt

A deprecation period assumes there is something to migrate to. It is
waived, with the reason stated in the release notes, when:

- the behavior was documented as unavailable or unimplemented and raised
  on every call
- keeping it would produce silently wrong numbers
- the surface never appeared in a tagged release

### 3.4 Break notes

Every breaking change carries a note with four parts: what broke, the
symptom a caller sees, the migration, and the release that removed it.
The note goes in the release notes and, where the break is in the Python
API, in the docstring of the surviving surface.

## 4. Parameters, aliases, and defaults

### 4.1 Canonical names

LightGBM's native parameter names are canonical everywhere: in the
estimator constructors, in the `params` dict of the functional API, and in
the parameter string that the C ABI and the CLI parse. Where mojotrees has
its own spelling for something LightGBM also names, LightGBM's name is the
one that is guaranteed.

### 4.2 Alias stability

The scikit-learn spellings LightGBM accepts are accepted as aliases and
are covered by this policy exactly as the canonical names are. The pairs
in force today are `min_child_samples` for `min_data_in_leaf`,
`min_child_weight` for `min_child_hess`, `reg_alpha` and `reg_lambda` for
`lambda_l1` and `lambda_l2`, `subsample` and `subsample_freq` for
`bagging_fraction` and `bagging_freq`, `device_type` for `device`,
`boosting_type` for `boosting`, and `categorical_features` for
`categorical_feature`. The callback reset path accepts a wider alias set,
listed in `_RESET_ALIASES` in `python/mojotrees/callback.py`, and it is
covered too.

Three rules hold for aliases.

1. An alias is never removed without the deprecation period of section 3.
2. An alias never diverges from its canonical name. If one is fixed, both
   are fixed, in the same release.
3. Setting both members of a pair to different non-default values raises,
   and that behavior is itself part of the contract. LightGBM warns and
   keeps one; mojotrees refuses, because a silently dropped
   hyperparameter is not recoverable from the output.

New aliases may be added in a minor release. Adding one is additive and
never changes what an existing program does.

### 4.3 Default stability

Defaults are LightGBM's, deliberately, so that a benchmark against
LightGBM compares implementations rather than configurations. A default is
frozen once it appears in a tagged release. Changing one is a breaking
change, because it changes the model a program produces without changing a
line of that program.

A default may change only in a major release, or in a minor release before
1.0, and only with all four of these in the release notes: the old value,
the new value, why, and how to pin the old behavior explicitly.

Two defaults are mojotrees's own rather than LightGBM's and are called out
because a reader may expect otherwise. `lambda_l2` defaults to 1.0, and
`min_child_hess` defaults to 1e-3. They are frozen on the same terms.

### 4.4 Unknown and unsupported keys

`parse_params` rejects an unknown key rather than ignoring it, and reports
a key that names a real LightGBM feature the parser does not cover as
unsupported rather than as unknown. Both behaviors are part of the
contract. A key moving from unsupported to accepted is additive. A key
moving the other way is breaking.

### 4.5 The unimplemented-objective register

The compiled registry (`registry_objective_unimplemented` in
bindings/objective_bindings.mojo, read by `_unimplemented_objectives()` in
`python/mojotrees/_fit_args.py`) names the LightGBM objectives that are
not implemented and says why for each. The
guarantee is that asking for one of them raises with a reason, not that
the reason's wording is stable. Implementing one and removing it from the
register is additive.

## 5. Fitted attributes

### 5.1 The set

`_FITTED_ATTRS` on `_Base` is the list of public attributes `fit` sets and
a refit clears. As of 0.1.0 it holds `n_features_in_`,
`feature_names_in_`, `device_`, `classes_`, `n_classes_`,
`best_iteration_`, `evals_result_`, `best_score_`, `stopped_early_`,
`n_iter_`, and `categorical_feature_`.

### 5.2 What is guaranteed

For every name in that set, across a minor release:

1. **Presence.** The attribute exists on a fitted estimator of a kind it
   applies to, and accessing it before `fit` raises `NotFittedError`
   rather than returning `None`.
2. **Type.** Its Python type does not change.
3. **Meaning.** Its documented meaning does not change.
4. **Reset.** A second `fit` clears it before it sets it, so no value ever
   survives from a previous fit.

Removing a name from the set, or changing any of the four, is a breaking
change. Adding a name is additive.

### 5.3 Round trips

Three ways of moving a fitted estimator exist and they carry different
amounts.

| Path | Carries | Does not carry |
|---|---|---|
| `pickle` | The whole estimator, including hyperparameters, fitted attributes, and the model's split gains and feature names | Nothing a fitted estimator holds |
| `save` and `load` | The model, its split gains, and its feature names. `n_features_in_` and `best_iteration_` are recomputed from it | Hyperparameters, `device_`, `evals_result_` |
| `Booster.model_to_string` and `model_from_string` | The model, its split gains, and its feature names | The training set, the parameter object |

That split is stable. A `load` gaining an attribute is additive. A `load`
losing one is breaking.

### 5.4 A known divergence, unresolved

`best_score_` is a scalar here, the best value of the primary metric that
`best_iteration_` was chosen by. LightGBM's `best_score_` is a nested dict
of every validation set's every metric; the equivalent grid lives in
`evals_result_`.

This is a real divergence from LightGBM's scikit-learn API and it is not
settled. Changing the type of `best_score_` to a dict would be a breaking
change under section 5.2, so it has to happen before the first tagged
release or wait for a major one. It is on the release gate of section 12
as item C4, a decision that must be made rather than one that has been
made.

## 6. Language bindings

### 6.1 Python

**Interpreter support.** `requires-python = ">=3.14"` today, and that is a
hard floor rather than a preference. The extension module is built by the
Mojo toolchain against a specific CPython, links no libpython, and carries
a `cp314-cp314` tag. There is no `abi3` build, so a wheel serves exactly
one CPython minor version.

The policy from the first release onward:

- Adding support for a CPython minor version is additive and lands in a
  minor release.
- Dropping one follows section 3.1 and, in addition, is never done while
  that version is still receiving upstream security support unless the
  Mojo toolchain has dropped it first. When the toolchain forces it, the
  release note says so and names the toolchain version that did it.
- The `requires-python` floor and the wheel tags shipped are stated in
  every release note.

**Import surface.** `import mojotrees` gives the names in `__all__`, and
`mojotrees.callback` gives the callback factories under their own module,
as LightGBM does. No other submodule is a supported import path.

`mojotrees.inspection` was the open case and is closed: `dump_model`,
`trees_to_dataframe`, `trees_to_records`, and `get_split_value_histogram`
are in `__all__`, resolved out of the submodule by `__getattr__` on first
access, so the names are public and the module that holds them is still
layout. `explain_device_choice` reaches `mojotrees.device_selection` the
same way, and `cv` and `CVBooster` are imported eagerly. Section 8.1.

Three submodules answer to `__getattr__` without exporting anything:
`mojotrees.dask`, `mojotrees.diagnostics`, and
`mojotrees.lgbm_model_io`. The attribute resolving is a convenience for a
reader, not a promise under this section, and each is experimental on its
own terms. dask's estimators raise `DistributedNotAvailable`, and the
LightGBM converter carries `EXPERIMENTAL = True` and warns on first use.
`python/mojotrees/_public_api_plan.py` records why each name stayed out,
and `python/tests/test_public_api_plan.py` holds that record to the code.
Naming them as supported, or saying in as many words that the attribute
is not a promise, is the successor to the inspection question; the
release gate carries it as item C5.

**numpy.** Optional. Every documented path works with plain Python
sequences, and this stays true. numpy is never a hard dependency.

**scikit-learn.** Optional and never imported except by the
`__sklearn_tags__` hook that scikit-learn itself calls. `check_estimator`
has not been run in full, so the package is scikit-learn style and does
not claim scikit-learn compliance. If that claim is ever made it will be
made with the suite's output next to it. The two known deviations are that
subclasses forward shared hyperparameters through `**kwargs`, so
`get_params()` lists them and `inspect.signature` does not, and that
`best_iteration_` is always set where LightGBM sets it only when early
stopping ran.

**SciPy.** Optional, and needed only to pass sparse input.

### 6.2 Mojo

**There is no Mojo ABI promise, and none is possible today.** Mojo does
not have a stable ABI, so a mojotrees built with one toolchain and a
caller built with another are not interoperable in general. Mojo
compatibility here means source compatibility: code that imported a name
from `mojotrees` and compiled against release N compiles against release
N+1 within a major version.

**Toolchain range.** `mojo >=1.0.0,<2` and `max >=26.5.0,<27` in
`pixi.toml`. Widening the range is additive. Narrowing it, including
raising the floor, is a breaking change for anyone pinned below the new
floor and is treated as one.

**Export surface.** The names re-exported from
`src/mojotrees/__init__.mojo` are the public Mojo API. A module that
exists in `src/mojotrees/` but is not re-exported is internal, whatever
its contents. Reachability by path is not an export.

**Signatures.** A trailing parameter with a default may be added to a
public function in a minor release, because it is source compatible for
every existing call. Reordering parameters, renaming one, or changing a
type is breaking.

### 6.3 C ABI

`capi/mojotrees.h` is the surface most likely to be consumed by compiled
callers that cannot be rebuilt in step with mojotrees, so it carries the
strictest rules. The header's own design rules are normative and are not
restated here.

`MOJOTREES_ABI_VERSION` is 1. It is bumped only for a change that breaks a
compiled caller. Query it with `mojotrees_abi_version()` when loading the
library dynamically, and compare it against the constant the caller was
compiled with.

**Additive, minor release, no ABI bump.**

- A new function.
- A new negative `MOJOTREES_ERROR_*` code. Callers are required to treat
  any negative return as a failure and to read the message from the error
  object, so a new code is not a break. Callers that switch exhaustively
  on the known codes and assume a default is impossible are outside the
  contract.
- A new key accepted in the parameter string.

**Breaking, major release and ABI bump.**

- Removing or renaming a function, changing its signature, or changing
  what it returns for an input it already accepted.
- Changing the numeric value of an existing status code.
- Changing an ownership or lifetime rule, including which calls clear the
  error object and how long a returned message pointer stays valid.
- Making a handle non-opaque, or exposing any mojotrees struct layout.

**Not covered.** The exact wording of an error message, and the shared
library's file name and soname, which are build outputs of `capi/build.sh`
rather than declarations in the header. Anyone loading the library
dynamically should key on `mojotrees_abi_version()` and not on a file
name.

### 6.4 Command line interface

`cli/mojotrees train` and `cli/mojotrees predict`, their flags, their CSV
input and output shapes, and their exit statuses (0 for success, 1 for a
runtime failure, 2 for a usage error) are public on the terms of section
1.2. Adding a flag is additive. Changing what an existing flag does, or
what an existing exit status means, is breaking. Text written to stderr
for humans is not covered.

## 7. Model format

### 7.1 What the format is

A versioned plain-text token stream, magic `mojotrees`, current version
`v4`. Floats travel as their IEEE-754 bit patterns rendered as decimal
`UInt64`, so a save and load round trip is bit-exact and the file has no
locale or precision pitfalls and no endianness dependence. The token after
the version distinguishes a single-output file (`objective`) from a
multiclass file (`multiclass`), so a reader determines the kind from the
file rather than from the call.

It is mojotrees's format. It is not LightGBM's text model format, and the
two are not interchangeable in either direction.

### 7.2 Backward compatibility, which is guaranteed

A release reads every model file any earlier release wrote. That is the
whole promise, and it is unconditional within a major version and across
major versions unless a release note says otherwise.

| Version | Adds | Read by current |
|---|---|---|
| v1 | Mapper edges and offsets, per-node feature, threshold, children, value | Yes |
| v2 | Missing-value routing, optional monotone section, optional categorical sections | Yes |
| v3 | Per-node covers, unconditional | Yes |
| v4 | Per-node split gains, per-node covers behind a presence flag, optional feature names | Yes |

An older file loads as what it is. A v1 or v2 file describes a model
trained without covers, so asking it for feature contributions raises
rather than guessing at them, and a v1 file routes no missing values
because the model it describes had none. A file older than v4 carries no
split gains, so the model it describes reports zero gain importance and a
dump of it reports `has_split_gain: false`: the absence is reported, never
filled in with a zero that could be mistaken for a measurement.

Re-saving an older model writes what it has and records what it lacks. v3
could not: it wrote a coverless model's zeros as if they were covers, and
its own reader refused the result. v4's presence flags are what make that
round trip lossless in the only sense available to it, and the current
reader also accepts the files that bug produced.

### 7.3 Forward compatibility, which is not guaranteed

An older release does not read a newer file. A reader that meets a version
it does not know reports the version rather than misparsing. Sections are
therefore added, never removed and never repurposed: a section name means
one thing for the life of the format.

### 7.4 Byte stability

Within one version, a model that does not use an optional section
serializes to exactly the bytes it did before that section existed. A
model with no categorical features writes no categorical section, a model
with no monotone constraints writes no monotone section, and both produce
byte-identical files to the releases that predate those features.

This is a compatibility property, not an incidental one. A change that
alters the bytes of a model that uses none of the new machinery is a
breaking change even though every file still loads.

### 7.5 What the file deliberately does not hold

Training-time knobs that only shaped which trees were grown are absent:
`num_leaves`, regularization, interaction constraints, subsampling, and
the training device. They cannot be checked against a loaded model and are
not needed to evaluate it. Two things that are also not needed to evaluate
a model are held anyway, because a consumer may need them and neither can
be recovered from a fitted tree: monotone constraints, which are a
property the trees satisfy, and split gains (v4), which are what the model
reports as gain importance and what a dump reports per node. Feature names
(v4, optional) are held for the same reason and are the model's own
labeling rather than a training knob.

Adding one of these to the format later is a format change under section
7.3, not a fix.

## 8. Inspection and numerical contracts

### 8.1 What structured inspection exists

Structured inspection landed while this document was being written, as
`python/mojotrees/inspection.py` and `src/mojotrees/inspection.mojo`,
with `dump_model`, `trees_to_dataframe`, `trees_to_records`,
`split_values`, `get_split_value_histogram`, `leaf_index_of`,
`raw_scores`, and `booster_of`. Its normative schema is
[docs/MODEL_INSPECTION_SCHEMA.md](MODEL_INSPECTION_SCHEMA.md), and that
document, not either implementation, is what a consumer reads.

It arrived with the version this policy asked for. `DUMP_FORMAT_VERSION`
is 1, it bumps only for a key removed, retyped, or given a new meaning,
and adding an optional key does not bump it, so a consumer must ignore
keys it does not know. That is the rule, and it is the schema doc's to
keep rather than this one's.

**That section 2 question is resolved.** `inspection` is a submodule with
its own `__all__`, and until 2026-08-15 nothing in it was re-exported
from `python/mojotrees/__init__.py`, which left
`mojotrees.inspection.dump_model` real, documented, and formally outside
the public surface. It closed by the first of the two ways this section
offered: `dump_model`, `trees_to_dataframe`, `trees_to_records`, and
`get_split_value_histogram` are in `mojotrees.__all__` and resolve the
submodule lazily on first access, so the four names are public under
section 2 and the module stays layout. The rest of `inspection.__all__`,
and `parse_model_string` behind the DELETION POINT banner, are reachable
as `mojotrees.inspection.<name>` and are not public;
`python/mojotrees/_public_api_plan.py` says why for each.

The rest of what a fitted model will tell you, and what each part
guarantees:

| Surface | Shape | Stability |
|---|---|---|
| `Booster.model_to_string()` | The model format text of section 7 | Versioned by the model format |
| `feature_importances_`, `Booster.feature_importance(importance_type)` | One value per feature, `"split"` or `"gain"` | Type and length stable; values change when the model does |
| `predict(pred_leaf=True)` | `(n_samples, num_iteration)`, or `(n_samples, num_iteration * n_classes)` for softmax | Shape stable; leaf numbering as in 8.2 |
| `predict(pred_contrib=True)` | `(n_samples, n_features + 1)`, or `(n_samples, n_classes * (n_features + 1))` in class-major blocks | Shape stable; sum identity as in 8.3 |
| `evals_result_` | `{valid_name: {metric_name: [round 0, round 1, ...]}}`, index 0 being the base-score-only model | Shape and indexing stable |
| `booster_` accessors | `current_iteration`, `num_trees`, `num_model_per_iteration`, `num_feature`, `feature_name` | Stable |
| `Dataset` accessors | `num_data`, `num_feature`, `num_bin`, `feature_name`, `categorical_feature`, `get_label`, `get_weight`, `get_group`, `get_init_score`, `get_data`, `get_field` | Stable |

Every one of those is covered by this policy on the terms of section 1.2.
The dump's contents are covered by its own schema document and its own
version, which is the arrangement section 1.4 describes. A JSON blob
without a version is not a contract; this one has a version.

### 8.2 Leaf identifiers

A leaf is named by its ordinal within its own tree, in `[0, num_leaves)`,
numbered in node order. The numbering is fixed once a tree is grown and
survives save, load, and pickle.

It is mojotrees's numbering and not LightGBM's, and the two agree only by
coincidence. Anything that consumes leaf ids as a categorical feature, as
LightGBM users do, is consuming a mojotrees-specific encoding and must be
refit rather than transferred.

### 8.3 Numerical guarantees that are part of the contract

These are results, and breaking one is a breaking change even though no
signature moves.

1. **Contribution sum identity.** Every `pred_contrib` row sums to that
   row's raw score, exactly. These are TreeSHAP Shapley values and the sum
   is an identity, not a normalization.
2. **Iteration slicing.** The base score belongs to iteration 0, so
   `predict(num_iteration=k)` equals the prediction of a `k`-round fit,
   and `[0, k)` plus `[k, n)` equals the whole raw score.
3. **Save and load bit exactness.** A loaded model predicts bit-identically
   to the model that was saved.
4. **Weight neutrality.** An all-ones `sample_weight` produces a
   bit-identical model to no `sample_weight` at all.
5. **Callback neutrality.** Training with no callbacks does not cross the
   Python boundary and produces a bit-identical model to a build without
   the callback bridge.
6. **Sparse and dense agreement.** A sparse fit equals the dense fit of
   the same matrix with the gaps filled with zeros. An implicit zero is
   the value 0.0 and not a missing value, matching LightGBM's default
   `zero_as_missing=false`. LightGBM's `zero_as_missing=true` is not
   implemented and no alias accepts it.
7. **Repeat-run determinism.** The same data, parameters, and seeds give a
   bit-identical model on a repeat run on the same device. Across devices
   see 8.4.

### 8.4 Determinism across devices and worker counts

CPU histogram accumulation is dispatched per feature and each feature owns
its output slice, so the worker count does not change the result.
`MOJOTREES_NUM_WORKERS` and `MOJOTREES_PARALLEL_MIN_OPS` are performance
controls and changing either does not change a model.

GPU histogram accumulation is fixed-point Int32 throughout, with the
conversion to Float64 done once at download, which is what makes a GPU run
bit-deterministic on repeat. CPU and GPU results agree to a documented
Float32 tolerance and integer counts agree exactly. They are not
bit-identical to each other, and this policy does not promise that they
will be.

Any change that would make CPU and GPU histogram reduction leave the
integer domain is a breaking change under 8.3 item 7, because it would
cost repeat-run bit-identity.

## 9. Callbacks, environment, and runtime controls

### 9.1 The callback environment

`CallbackEnv` is a `namedtuple` with fields `model`, `params`,
`iteration`, `begin_iteration`, `end_iteration`, and
`evaluation_result_list`, in that order. Because it is a tuple, positional
unpacking works and field order is therefore part of the contract.

- Adding a field is additive **only at the end**, and only in a minor
  release.
- Removing a field, renaming one, or reordering any is breaking.
- The type of a field is stable.

### 9.2 The callback protocol

A callback is a callable taking one `CallbackEnv`. Callbacks split into
before-iteration and after-iteration groups by a `before_iteration`
attribute and run in ascending `order` within each group. This is
LightGBM's contract and it does not change without a major release.

`EarlyStopException` stops training from inside a callback. Any other
exception propagates with its own type and leaves the estimator unfitted,
which is a guarantee about state and not only about the exception.

The four factories LightGBM ships are `early_stopping`,
`log_evaluation`, `record_evaluation`, and `reset_parameter`, importable
from `mojotrees` and from `mojotrees.callback`. Their signatures follow
LightGBM's.

Known limits, which are documented rather than fixed. Callbacks need an
`eval_set`, because the hook lives in the trainer that scores validation
metrics. The softmax and LambdaRank trainers refuse a callback list rather
than ignoring it. A refusal becoming an acceptance is additive; the
reverse is breaking.

### 9.3 Resettable parameters

`RESETTABLE` in `python/mojotrees/callback.py` names the hyperparameters a
before-iteration callback may change, and the tuple order is the slot
order the Mojo bridge reads. It is coupled to `RESET_SLOTS` and to
`_write_reset` and `_read_reset` in `bindings/_mojotrees.mojo`.

The order is therefore a wire format between two files in this repository,
and a change to one side alone reassigns parameters silently. Both sides
move in the same commit or neither moves. Appending a slot at the end is
additive. Reordering is breaking, and it is the kind of break that
produces wrong numbers rather than an error, so the release gate checks
the two lists against each other.

### 9.4 What a callback costs

One crossing of the Python boundary per phase per round and nothing per
row. With no callbacks the bridge does not cross the boundary at all. The
per-round rather than per-row property is part of the contract; the
measured overhead is not.

### 9.5 Environment variables

| Variable | Meaning |
|---|---|
| `MOJOTREES_NUM_WORKERS` | 1 is serial, N above 1 forces chunked dispatch and takes precedence over the threshold, 0 or unset is automatic |
| `MOJOTREES_PARALLEL_MIN_OPS` | Overrides the built-in parallel dispatch threshold |
| `MOJOTREES_DISABLE_GPU` | Makes device selection behave as though no accelerator were present |
| `MOJOTREES_GPU_HIST_STRATEGY` | Selects the GPU histogram strategy |
| `MOJOTREES_GPU_BLOCK_THREADS` | Overrides GPU launch block width |
| `MOJOTREES_GPU_ROW_TILE` | Overrides GPU row tile size |
| `MOJOTREES_AUTO_MIN_CELLS` | Threshold for automatic device selection |

The variables are public on the terms of section 2. Their **semantics**
are covered by this policy: what a value means, and the precedence of
`MOJOTREES_NUM_WORKERS` over the threshold. Their **default numeric
values** are tuning constants derived from measurements and are not
covered. A default threshold may be retuned in a patch release, and has
been, because a retune changes speed and not results (section 8.4).

`MOJOTREES_DISABLE_GPU` is the exception in one direction: it changes
which backend runs, and CPU and GPU are not bit-identical to each other,
so it changes results within the documented tolerance.

## 10. Supported platforms

### 10.0 Which document is authoritative

Two documents describe platforms and they answer different questions.
[docs/PLATFORM_MATRIX.md](PLATFORM_MATRIX.md), with its machine-readable
half `packaging/matrix/platform_matrix.toml` and the
`validate_matrix.py` that checks the two against each other, is the
authority on **installable targets**: which artifact exists for a place,
what it is called, which interpreter it targets, and what evidence backs
the claim. Where that matrix and this section disagree about an artifact,
the matrix wins and this section is the bug.

This section answers a different question, which is **what a release
promises about a platform** and what has to be true before the promise
can be made. The two vocabularies are not the same and should not be
merged. The matrix's `validated`, `tested`, `designed`, and `unsupported`
describe an artifact; the tiers below describe test evidence.

### 10.1 Tiers

| Tier | Meaning |
|---|---|
| 1 | Every push runs the full test suite here. A failure blocks the release. |
| 2 | Validated by hand before a release, on hardware the project has. A failure blocks the release. |
| 3 | Expected to work from the source's portability, never executed by this project. No claim of any kind. |

### 10.2 Where each platform sits today

| Platform | Tier | Evidence |
|---|---|---|
| linux-64 CPU | 1 | CI `test`, `python`, and `parity` jobs on `ubuntu-latest` |
| linux-aarch64 CPU | 1 | CI `test` and `python` jobs on `ubuntu-24.04-arm` |
| osx-arm64 CPU | 2 | Development machine and local suite. Deliberately not in CI: Apple-silicon runners report an accelerator at compile time but lack the Metal toolchain, so the GPU equivalence test cannot build there |
| Apple GPU, Metal | 2 | Apple M4, correctness and determinism pass, phase timings partial, profiler trace not run |
| NVIDIA GPU, CUDA | 3 | **Never executed.** No hardware, no self-hosted runner |
| AMD GPU, HIP | 3 | **Never executed.** No hardware, no self-hosted runner |
| Windows, any architecture | Unsupported | Not a `pixi.toml` platform, never built, never tested |
| linux-64 or linux-aarch64 wheels | Unsupported | No manylinux build exists |

`pixi.toml` declares `osx-arm64`, `linux-64`, and `linux-aarch64`. That
list is the outer boundary; a platform outside it is not tier 3, it is
absent.

There is one GPU source. `histogram_gpu.mojo` holds the histogram and
partition kernels, `train_gpu.mojo` drives device-resident tree growth,
and `gpu_tiling.mojo` derives launch geometry from device attributes at
runtime. There is no CUDA file, no HIP file, and no Metal file. That is a
design commitment about the source, and it is not evidence about a backend
nobody has run. See [docs/GPU_VALIDATION.md](GPU_VALIDATION.md) for what a
tier 3 row has to produce to move.

### 10.3 Distribution

One wheel has been produced locally, `packaging/build_wheel.sh` on macOS
arm64, tag `cp314-cp314`, platform `macosx_26_0_arm64`, bundling the four
MAX runtime dylibs the extension links through `@rpath`. Nothing has been
published and there is no PyPI release, so under the matrix's vocabulary
every target including that one is `designed` rather than `validated`: a
wheel that was built once on the machine that wrote it is not evidence
that the target works.

`packaging/matrix/platform_matrix.toml` is the list of targets and their
expected filenames. A release note names every artifact it ships and its
exact tag, and says plainly which platforms have no artifact.

### 10.4 Changing the table

- Promoting a platform from tier 3 to tier 2 requires the evidence the
  tier demands, recorded in the repository, not a report that it worked.
- Promoting to tier 1 requires a runner in CI running the full suite.
- Demoting a platform, or removing one from `pixi.toml`, is a breaking
  change and follows section 3.
- A platform's tier is stated in every release note. A tier that dropped
  since the last release is stated first.

## 11. The API snapshot manifest

### 11.1 What it is

`compatibility/api_snapshot_manifest_v1.json` records the public surface of
section 2 in a machine-readable form so that a diff between two releases
answers "what changed for a caller" without anybody having to remember.

**It is a proposal in its current state.** The file was written by hand
from a reading of the source and has not been generated by tooling. It
was then cross-checked mechanically against the working tree, by an `ast`
and `re` pass that imported nothing and built nothing, and every block
that pass could reach agreed with the tree; the manifest's `verification`
block lists what was checked and what was not. That is one agreement at
one moment, not a generated artifact and not a thing that re-runs. Until
the generator of section 11.3 exists and its output replaces the file,
treat it as a draft of the shape rather than as an authority on the
contents.

### 11.2 How a diff is read

| Diff | Classification |
|---|---|
| A name appears | Additive, minor |
| A name disappears | Breaking, and must have served its deprecation period |
| A default value changes | Breaking, section 4.3 |
| An argument is appended with a default | Additive, minor |
| An argument is reordered, renamed, or retyped | Breaking |
| A `CallbackEnv` field appears at the end | Additive, minor |
| A `CallbackEnv` field moves or disappears | Breaking, section 9.1 |
| `RESETTABLE` order changes | Breaking, and silently wrong, section 9.3 |
| A C ABI declaration changes | Breaking, and requires an ABI version bump |
| The model format version changes | Section 7, and the read-back matrix must be extended |
| `DUMP_FORMAT_VERSION` changes | A dump key was removed, retyped, or redefined; breaking for a consumer |
| A dump key appears | Additive, minor, and does not bump the dump version |
| A platform tier drops | Breaking, section 10.4 |

An additive diff is recorded by regenerating the manifest in the same
commit as the change. A breaking diff blocks the release until it has a
break note under section 3.4.

### 11.3 What has to be built

The manifest is only worth its storage if a tool regenerates it and CI
compares. Neither exists. `handoffs/task20_compatibility.md` carries the
specification for `tools/api_snapshot.py`, the pixi task, the CI job, and
the small number of exports the tool needs from surfaces that do not
currently expose them.

## 12. The release gate

Every item is a hard gate. A release with an unchecked item is not cut.
None of these items changes a parity status.

Items are numbered within their letter, so an item can be inserted
without renumbering a reference somewhere else in this document.

**A. Tests and contracts**

- **A1.** CI green on `ubuntu-latest` and `ubuntu-24.04-arm`, for `test`,
  `python`, and `parity`.
- **A2.** `pixi run test` green on osx-arm64 locally, the tier 2 CPU
  platform.
- **A3.** `pixi run test-gpu` green on osx-arm64 locally, the tier 2 GPU
  platform, or the GPU rows of section 10.2 demoted in the release notes.
- **A4.** `pixi run check-parity` green, with `KNOWN_UNWIRED_TESTS` still
  empty.
- **A5.** `pixi run -e pytest test-estimators` green.
- **A6.** `pixi run test-c` green, or explicitly recorded as skipped for
  want of a C compiler.
- **A7.** `python3 packaging/matrix/validate_matrix.py` green, so no row
  of `docs/PLATFORM_MATRIX.md` claims `validated` without the evidence
  file it names.

**B. Versions**

- **B1.** The three library version locations of section 1.1 agree.
- **B2.** `MOJOTREES_ABI_VERSION` bumped if and only if a header
  declaration changed incompatibly.
- **B3.** The model format `_VERSION` bumped if and only if the format
  changed, with section 7.2's read-back matrix extended and a test that
  loads a file of every earlier version.
- **B4.** `DUMP_FORMAT_VERSION` bumped if and only if a dump key was
  removed, retyped, or redefined.
- **B5.** The snapshot manifest regenerated and its diff classified under
  section 11.2.

**C. Surface**

- **C1.** `RESETTABLE` in `python/mojotrees/callback.py` checked against
  the bridge: its length equals `RESET_SLOTS` in
  `bindings/_mojotrees.mojo`, and its order matches the slot order of
  `_write_reset` and `_read_reset` there, entry by entry.
- **C2.** Every deprecation whose period has elapsed either removed with a
  break note or explicitly extended.
- **C3.** No new public name that lacks a docstring stating what it
  guarantees.
- **C4.** `best_score_` resolved for this release, either as the scalar
  this policy documents or as the LightGBM-shaped dict, with the decision
  in the release notes. Section 5.4.
- **C5.** The lazy submodules resolved for this release. `inspection` and
  `device_selection` are done: their user-facing names are in `__all__`,
  which is the first of the two ways section 8.1 offered. What is left is
  `mojotrees.dask`, `mojotrees.diagnostics`, and
  `mojotrees.lgbm_model_io`, which `__getattr__` answers for and which
  export nothing: section 6.1 either names them as supported import paths
  or says the attribute is not a promise. Sections 6.1 and 8.1.

**D. Honesty**

- **D1.** Every platform in section 10.2 carries the tier its evidence
  supports, and any row whose evidence has gone stale is demoted before
  the release rather than after.
- **D2.** The release note names every artifact shipped, with its exact
  tag, and names the platforms that get none.
- **D3.** No document in the repository claims a benchmark result, a
  backend validation, or a compliance suite pass that is not reproducible
  from the repository.
- **D4.** Any surface documented as unavailable still raises rather than
  silently doing something approximate.

**E. Not on the gate**

Version 1.0 readiness is not on this list, and passing this gate is not an
argument for it. What 1.0 would additionally require is a separate
question this document does not answer.
