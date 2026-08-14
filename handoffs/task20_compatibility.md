# Handoff, task 20, compatibility and release contract

Lane 20 of the parallel round. Nothing outside the four assigned paths was
touched, nothing was committed or staged, no build, test, bench, or
profiler was run.

## What this lane produced

| Path | What it is |
| --- | --- |
| `docs/COMPATIBILITY_POLICY.md` | The contract. Twelve sections: versioning, what is public, deprecation, parameters and defaults, fitted attributes, the three language bindings, model format, inspection and numerical guarantees, callbacks and environment, platforms, the snapshot manifest, and the release gate. |
| `docs/tutorials/feature_complete_walkthrough.md` | Fourteen sections covering validation, early stopping, callbacks, inspection, save and load, continued training, leaves and contributions, sparse, categorical, backend selection, classification and ranking, custom objectives, and a closing table of everything the walkthrough could not show. |
| `tests/parallel/api_snapshot_manifest.json` | A proposed API snapshot. Valid JSON, `schema_version` 1, `status` `"proposed"`. |
| `handoffs/task20_compatibility.md` | This file. |

`git diff --check` on the four paths is clean. All four are new files, so
they show as untracked rather than as a diff.

## Read before trusting the manifest

`tests/parallel/api_snapshot_manifest.json` was **written by hand** from a
reading of the source. No tool generated it and nothing verified it
against a built extension module. Its `about` block says so, and
`status` is `"proposed"` rather than `"generated"` for exactly that
reason.

Absence of a name from it is not evidence the name is absent from the
code. It may be evidence that the hand pass missed it. Do not treat a
diff against it as authoritative until `tools/api_snapshot.py` below
exists and its output has replaced the file wholesale.

**It was then cross-checked mechanically, and found no drift.** After the
file was written, a throwaway `ast` and `re` script parsed the working
tree and compared it block by block. It imported nothing and built
nothing. Every block it could reach agreed with the manifest: the whole
Mojo export table, module set included; `__all__`; `_FITTED_ATTRS`; all
39 shared estimator parameters **and their default values**; the
constructor, `fit`, `predict`, and `predict_proba` signatures of all
three estimators; the callback `__all__`, `CallbackEnv` field order,
`RESETTABLE`, `_RESET_ALIASES`, and `RESET_SLOTS`; the fourteen C ABI
declarations; the three library version locations; the model format and
ABI versions; and the declared platforms. The `verification` block in the
manifest lists what was checked and, more usefully, what was not.

That raises the file's standing from "written by hand" to "written by
hand and mechanically agreed with the tree once". It does not make it a
generated artifact, and nothing re-runs the check.

Its location is this round's convention rather than a considered choice.
`tests/parallel/` holds no other JSON and no test runner reads it. If the
snapshot becomes a checked artifact, `tests/api_snapshot.json` or
`docs/api_snapshot.json` is probably the better home. Moving it is free
today and expensive after the first release note cites the path. There is
no interaction with `tools/check_parity.py`, whose `KNOWN_UNWIRED_TESTS`
check only looks at `tests/*.mojo`.

## Required tooling, which does not exist

### 1. `tools/api_snapshot.py`

The generator. Model it on `tools/check_parity.py`, which already solves
the hard part of this problem twice.

**Constraints, all of them taken from `check_parity.py`.**

- Standard library only. `ast` for the Python sources, `re` for the Mojo
  and C sources.
- Builds nothing and imports nothing from `mojoboost`. It must run on a
  bare runner in seconds, because a snapshot check that needs a working
  extension module will be skipped the day it is inconvenient.
- Deterministic output. `json.dump(..., indent=2, sort_keys=True)` plus a
  trailing newline, so a regeneration with no change produces a
  byte-identical file and the diff is the signal.

**Interface.**

```
python3 tools/api_snapshot.py --check    # exit 0 if the file matches, 1 with a diff if not
python3 tools/api_snapshot.py --write    # regenerate the file in place
```

`--check` prints the classification of every difference it finds, using
the table in section 11.2 of the compatibility policy, and prints
`breaking` or `additive` next to each. Exit 1 for any difference; the
gate decides what to do about it, not the tool.

**What each block comes from.**

| Manifest block | Source | How |
| --- | --- | --- |
| `python.all` | `python/mojoboost/__init__.py` | `ast`, the `__all__` assignment. `check_parity.python_api` already locates it |
| `python.shared_estimator_parameters` | same | `ast`, `_Base.__init__` args and defaults |
| `python.estimators.*.own_parameters` | same | `ast`, each subclass `__init__` |
| `python.estimators.*.methods` | same | `ast`, public `FunctionDef` args and defaults |
| `python.fitted_attributes` | same | `ast`, `_Base._FITTED_ATTRS` |
| `python.estimators.*.objective_names` | same | `ast`, the `_OBJECTIVES` class attribute |
| `python.callbacks.*` | `python/mojoboost/callback.py` | `ast`, `__all__`, the `CallbackEnv` field list, `RESETTABLE`, `_RESET_ALIASES` |
| `python.eval_metric_names` | `python/mojoboost/_eval.py` | `ast`, `_METRICS` and `_ALIASES` keys |
| `python.functional_api` | `python/mojoboost/basic.py` | `ast`, `__all__` and the public methods of `Dataset` and `Booster` |
| `mojo.exports_by_module` | `src/mojoboost/__init__.mojo` | `re`, the `from .mod import (...)` blocks. `check_parity.mojo_exports` already parses these |
| `mojo.objective_codes` | `src/mojoboost/boosting.mojo`, `params.mojo` | `re`, the `comptime NAME = <int>` lines |
| `c_abi.*` | `capi/mojoboost.h` | `re`, the `#define` lines and the function declarations |
| `parameter_string.supported_keys` | `src/mojoboost/params.mojo` | `re`, the `SUPPORTED_KEYS` string literal |
| `model_format.version` | `src/mojoboost/serialize.mojo` | `re`, `comptime _VERSION` |
| `versions.library` | `pixi.toml`, `python/pyproject.toml`, `__init__.py` | `re` on each; **disagreement is an error, not a merge** |
| `versions.mojo_toolchain`, `max_toolchain` | `pixi.toml` | `re` on `[dependencies]` |
| `platforms.declared` | `pixi.toml` | `re` on `[workspace] platforms` |
| `platforms.tiers` tier 1 rows | `.github/workflows/ci.yml` | `re` on the runner matrix |
| `python.callbacks.reset_slots` | `bindings/_mojoboost.mojo` | `re`, `comptime RESET_SLOTS` |

**Four parsing gotchas, all of them found by writing the cross-check.**
Each one silently produces a wrong value rather than an error, which is
the worst failure mode for a tool whose whole job is detecting change.

1. **`_METRICS` in `_eval.py` is not `literal_eval`-able.** Its values are
   module-level integer constants (`L2`, `RMSE`, and so on), so
   `ast.literal_eval` on the dict raises `ValueError`. Read
   `node.value.keys` and evaluate those individually. `_ALIASES` and
   `_RESET_ALIASES` are pure literals and read fine whole.
2. **Two shared estimator defaults are named constants, not literals.**
   `_Base.__init__` has `lambda_l2=_LAMBDA_L2` and
   `lambda_l1=_LAMBDA_L1`. A naive default reader records the string
   `"_LAMBDA_L2"` and then reports drift forever, or worse, records it
   and never notices when the constant changes. Resolve module-level
   `Name` defaults against the module's own assignments before comparing.
3. **`CallbackEnv` is a `namedtuple()` call, not a class.** The field list
   is the second positional argument, `node.value.args[1]`, not anything
   reachable from a `ClassDef` walk.
4. **Mojo exports come in two spellings.** Most are
   `from .mod import (\n  a,\n  b,\n)`, but `from .gain import leaf_score`
   is a single line, and `from .binning import BinMapper, BinnedMatrix,
   bin_equal_width, fit_bins` is a single line with four names. A regex
   that only handles the parenthesized form silently drops whole modules.
   `check_parity.mojo_exports` already handles both; reuse it rather than
   writing a second parser.

**Two blocks cannot be generated and must be hand-maintained**, so the
tool has to preserve them across a `--write` rather than dropping them:
`platforms` rows above tier 1, and `numerical_contracts`. Read them from
the existing file, carry them forward unchanged, and print a reminder
that they were carried rather than derived.

### 2. Two checks the generator should also perform

Both are cheap, both catch a class of bug that produces wrong numbers
rather than an error, and both belong here rather than in a separate
tool.

1. **Reset slot agreement.** `len(RESETTABLE)` in
   `python/mojoboost/callback.py` equals `RESET_SLOTS` in
   `bindings/_mojoboost.mojo`, and the order of `RESETTABLE` matches the
   slot order of `_write_reset` and `_read_reset` entry by entry. The
   mapping to check against is `learning_rate`, `num_leaves`,
   `max_depth`, `min_data_in_leaf`, `min_sum_hessian_in_leaf` which is
   `min_child_hess` on the Mojo side, `lambda_l1`, `lambda_l2` which is
   `lambda_reg` on the Mojo side, `feature_fraction`,
   `feature_fraction_bynode`. They agree today.
2. **Version agreement.** The three library version locations of policy
   section 1.1 hold the same string.

### 3. The pixi task

`pixi.toml` is a shared file and this lane did not touch it. Add:

```toml
# The API snapshot. Standard library only and no build, like check-parity,
# so it runs on a bare runner in seconds.
api-snapshot = "python3 tools/api_snapshot.py --check"
api-snapshot-write = "python3 tools/api_snapshot.py --write"
```

### 4. The CI job

`.github/workflows/ci.yml` is a shared file and this lane did not touch
it. Add a job alongside `parity`, which it deliberately mirrors:

```yaml
  # The public API snapshot. Standard library only and no build, so it runs
  # on a bare runner and fails when a public name, default, or signature
  # changes without the manifest being regenerated.
  api-snapshot:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: API snapshot
        run: python3 tools/api_snapshot.py --check
```

## Required exports, which do not exist

Six surfaces have no machine-readable source, so the manifest carries
hand-written values for them today. Each wants a small export before the
generator can own the block.

| Need | Where it should live | Why |
| --- | --- | --- |
| CLI commands, flags, and exit statuses | `cli/mojoboost_cli.mojo`, as a `comptime` list beside the argument parser | The manifest's `cli` block is hand-written. `--help` text is prose and not a contract |
| The `MOJOBOOST_*` registry | One place. `src/mojoboost/params.mojo` or a new `env.mojo` | The seven variables are read from `parallel.mojo`, `device.mojo`, and `gpu_tiling.mojo` today. A grep finds them; a grep is not a contract, and a variable added in a fourth module would be missed silently |
| `SUPPORTED_KEYS` as one literal | `src/mojoboost/params.mojo`, already true | Keep it a single concatenated string literal. Building it from a list comprehension would put it out of reach of a regex parser that must not import Mojo |
| `_FITTED_ATTRS`, `RESETTABLE`, `_METRICS`, `_ALIASES`, `_OBJECTIVES` as module or class level assignments | Where they are now | They are `ast`-reachable today, though `_METRICS` is keys-only and two `_Base` defaults are named constants; see the parsing gotchas above. Computing any of them at import time would force the snapshot tool to import a built extension, which kills the bare-runner property |
| `RESET_SLOTS` and `_VERSION` as `comptime` scalars | Where they are now | Same reason |
| Structured model inspection | Not yet designed | When `dump_model` lands it needs a schema version of its own, per policy section 8.1. A JSON blob without a version cannot be covered by a compatibility policy |

## Drift and open questions found while reading

These were found in files this lane may not edit. None was changed.

1. **`best_score_` is unresolved, and it is on the release gate.** It is a
   scalar here and a nested dict in LightGBM's scikit-learn API. Changing
   its type after the first tagged release is a breaking change under
   policy section 5.2, so the decision belongs before that release. Gate
   item C14 records it as a decision that must be made, not one that has
   been made.
2. **A stale paragraph in `python/mojoboost/__init__.py`.** The module
   docstring at roughly lines 183 to 196 says "There are no built-in
   validation metrics in the Python API yet". The paragraph at lines 80 to
   92 documents the built-in metric names, and
   `python/mojoboost/_eval.py` implements the registry. The later
   paragraph also repeats the callable-metric contract the earlier one
   already gave. The walkthrough follows the working code and documents
   built-in names as available. Someone who owns that file should delete
   the stale half.
3. **A misnamed cross-reference in `bindings/_mojoboost.mojo`.** The
   `RESET_SLOTS` docstring says "the slot order below is the contract the
   Python side mirrors in `_RESET_SLOTS`". The Python side names it
   `RESETTABLE`; there is no `_RESET_SLOTS` in `python/mojoboost/`. The
   ordering itself agrees, entry for entry. Only the name in the comment
   is wrong.
4. **No git tag and no published artifact.** `git tag` is empty and no
   wheel has been uploaded. The policy is written to take effect at the
   first tagged release, and it says so in its first section.
5. **The walkthrough has never been executed.** It was written from the
   source, not from a transcript, and it says so at the top. Its two
   least obvious claims were checked against the code rather than against
   a docstring: `_check_predict_flags` does refuse any two of
   `raw_score`, `pred_leaf`, and `pred_contrib` together, and
   `reset_parameter` does take either a per-round list or a callable on
   the 0-based round. Reading a claim is still not running it. The obvious
   follow-up is a `docs/tutorials/feature_complete_walkthrough.py` that
   runs the same steps and a CI job that runs it, at which point the
   prose can cite a script that is known to work. That is a build lane's
   job, not a doc lane's.

## What this lane deliberately did not do

- Did not touch `docs/LIGHTGBM_PARITY.md`. No parity status was set,
  raised, or lowered, and the policy says so explicitly in its opening
  section. The release gate is additive to the parity contract and does
  not overlap it.
- Did not touch `README.md`, `pixi.toml`, `.github/workflows/ci.yml`,
  `tools/check_parity.py`, or anything under `python/`, `src/`,
  `bindings/`, `capi/`, or `cli/`. The task text for the pixi task and
  the CI job is above, ready to paste.
- Did not run a build, a test, a bench, or a profiler, and did not commit
  or stage anything.
- Did not claim 1.0 readiness. Policy section 12E states in the release
  gate itself that passing the gate is not an argument for 1.0, and that
  what 1.0 would additionally require is a question the document does not
  answer.
