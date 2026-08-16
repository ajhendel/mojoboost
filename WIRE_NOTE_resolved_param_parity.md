# LANDED: the CatBoost read-back edges

**Status: all six parts are applied on `lane/resolved-param-parity`.** This
started as a wire note, written when `engines.py`, `worker.py` and `run.py`
belonged to a session that was running from them. That session stood down and
the edits were made by this lane.

It is kept rather than deleted because it is the only document that states each
edit as a standalone change with its reasoning, which is what a reviewer wants
when reading the diff, and because two of the parts have a hazard in them that
is easy to reintroduce:

- **Part 4's sort key must stay composed under the repeat sort.** `(repeat,
  rank)`, never `(rank, repeat)`. The landed version is `run.CELL_ORDER`.
- **Part 1's `catboost_lossguide` suppression is load-bearing.** Both CatBoost
  rows share a cell key, and the variant row's resolved dict is not the shape
  the CatBoost-mode arm is built from.

What landed differs from the note in three small ways, all of them additions:
`build` dispatches on `issubclass(engine, MojoTreesEngine)` rather than on the
name; `_params` takes a `device` override so the sparse path keeps its
hard-coded `"cpu"`; and `run.build_matrix` gained a skip so that scheduling the
CatBoost-mode arm without `catboost` is a reported skip rather than a raising
cell.

Part 6 is applied too: `schema.json` no longer says the harness pins the
learning rate, and it has an `engine_resolved_params_drift` entry.
`RESULTS_TEMPLATE.md` carries the misreading warning as a block quote.

---

## Part 1. `engines.py`, `CatBoostEngine.run`: emit the read-back sidecar

Currently, immediately after `resolved` and `resolved_note` are built (the
`try` block that calls `model.get_all_params()`), add:

```python
        # Where a live fit disagrees with scenarios.CATBOOST_LEFT_AT_STOCK.
        # Recorded, not raised: a CatBoost upgrade moving a default is a thing
        # to find out about from a record, not a reason to lose a cell.
        readback_drift = (
            scenarios.check_catboost_readback(resolved, spec, self.threads)
            if resolved
            else ["get_all_params() produced nothing: " + str(resolved_note)]
        )
        readback_entry = scenarios.catboost_readback_entry(
            spec, resolved or {}, self.version
        )
```

and in the returned dict, beside the two `engine_resolved_params*` keys:

```python
            "engine_resolved_params_drift": readback_drift,
            # Popped by worker.run_job into the run's sidecar so that the
            # mojotrees_catboost_mode cell for this same scenario can read
            # CatBoost's resolved learning rate. Not a report field.
            "catboost_readback": readback_entry,
```

Do the same in `CatBoostLossguideEngine` only if you want its read-back too;
it inherits `run`, so it happens automatically and writes under the same cell
key. **That is a collision**: two CatBoost arms, one cell key. Either give
`catboost_readback_key` the engine name, or skip the write when
`self.variant_params` is non-empty. The second is one line and is what this
note recommends, because the `catboost_lossguide` row pins `grow_policy` and
`max_leaves` and its resolved dict is not the one the CatBoost-mode arm is
shaped after:

```python
        if self.variant_params:
            readback_entry = None
```

## Part 2. `engines.py`: let the CatBoost-mode arm receive it

`MojoTreesEngine._run_dense` calls `type(self).params_fn(spec, self.device, extra)`,
which cannot see instance state. Replace that one call with a method, and
override the method on the one subclass that needs more than three arguments.

In `MojoTreesEngine`:

```python
    def _params(self, spec, extra):
        """The translated parameter dict. A method rather than the
        `params_fn` call inline, because the CatBoost-mode subclass needs
        instance state (CatBoost's read-back) that a staticmethod cannot see."""
        return type(self).params_fn(spec, self.device, extra)
```

and in `_run_dense` and `_run_sparse`, replace

```python
        params = type(self).params_fn(spec, self.device, extra)
```

with

```python
        params = self._params(spec, extra)
```

In `MojoTreesEngine.__init__`, accept and store the read-back:

```python
    def __init__(self, threads, device="cpu", catboost_readback=None):
        ...
        self.catboost_readback = catboost_readback
```

In `MojoTreesCatBoostModeEngine`:

```python
    def _params(self, spec, extra):
        return scenarios.mojotrees_catboost_mode_params(
            spec, self.device, extra,
            catboost_readback=self.catboost_readback,
        )
```

and in `build`:

```python
def build(name, threads, device, catboost_readback=None):
    if name not in ENGINES:
        raise KeyError(...)
    engine = ENGINES[name]
    if name == "mojotrees_catboost_mode":
        return engine(threads, device, catboost_readback)
    return engine(threads, device)
```

`mojotrees_catboost_mode_params` raises `scenarios.CatBoostReadbackMissing`
when the read-back is absent or is for a different cell. **Do not catch it and
fall back.** The message names the cell and the keys; a fallback to
`BASE_PARAMS['learning_rate']` is the defect this whole change removes.

## Part 3. `worker.py`, `run_job`: read the sidecar, write the sidecar, record both dicts

Three edits.

Before `engine = engines.build(...)`:

```python
    # CatBoost's resolved parameters for this same cell, if a CatBoost cell
    # has already run in this run. Only the CatBoost-mode arm reads it, and it
    # refuses by name when it is absent rather than guessing a learning rate.
    catboost_readback = None
    readback_path = job.get("catboost_readback_path")
    if readback_path and os.path.exists(readback_path):
        with open(readback_path) as handle:
            catboost_readback = json.load(handle)
```

Change the build call:

```python
    engine = engines.build(
        job["engine"], job["threads"], job["device"], catboost_readback
    )
```

After `params_used = result.pop("params_used")`:

```python
    readback_entry = result.pop("catboost_readback", None)
    if readback_entry and readback_path:
        scenarios.append_catboost_readback(readback_path, readback_entry)
```

And inside `record["params"]`, one more key:

```python
            "resolved_parity": scenarios.record_parity_block(
                spec, job["engine"], params_used, dataset_params_used,
                catboost_readback,
            ),
```

That last line is requirement 3: both engines' resolved dicts and the
key-by-key verdict land in **every** result row, not only in the manifest.

## Part 4. `run.py`: give every job the sidecar path, and order the cells

In `main`, after `run_dir` is made:

```python
    readback_path = os.path.join(run_dir, scenarios.CATBOOST_READBACK_FILE)
```

In `build_matrix`, add `"catboost_readback_path": None` to each job dict, and
in `main` fill it in before the loop (it cannot be built in `build_matrix`,
which does not know `run_dir`):

```python
    for job in jobs:
        job["catboost_readback_path"] = readback_path
```

Then make the ordering explicit rather than incidental. The CatBoost cell for
a scenario must run before the CatBoost-mode cell for the same scenario in the
same round, or the mode cell refuses. Today that holds only because the engine
list happens to append `catboost` before `mojotrees_catboost_mode`. Replace

```python
    runnable.sort(key=lambda job: job["repeat"])
```

with

```python
    # Round-interleaved as before, and inside a round the CatBoost cell for a
    # scenario runs before the CatBoost-mode cell for the same scenario:
    # scenarios.mojotrees_catboost_mode_params takes CatBoost's RESOLVED
    # learning rate for that cell and refuses by name without it. Stable, so
    # every other arm keeps build order.
    _ARM_ORDER = {"catboost": 0, "catboost_lossguide": 0,
                  "mojotrees_catboost_mode": 1}
    runnable.sort(
        key=lambda job: (job["repeat"], _ARM_ORDER.get(job["engine"], 0))
    )
```

Note the consequence: a run that schedules `mojotrees_catboost_mode` **without**
`catboost` on the same scenario, tier and variant will refuse every mode cell.
That is correct behavior and it is a scheduling constraint somebody will hit
with `--engine mojotrees_catboost_mode` alone. If a friendlier failure is
wanted, `run.py` should skip the mode arm with a reason when `catboost` is not
in `args.engine`, rather than the cell raising.

## Part 5, optional: use the run-time half of the parity check

`scenarios.catboost_parity_from_records(catboost_record, mode_record)` returns
the failures on the keys no static check can see -- today only
`learning_rate`. `verify.py` is where that belongs, paired the same way its
differential pairs mojotrees with lightgbm. It was not wired here because
`verify.py` was not this lane's file either.

## Unparking

After parts 1 to 4 are in, set the three entries in
`MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT` back to `None`:

```python
    "dense_regression": None,
    "imbalanced_binary": None,
    "ordered_boosting_small": None,
```

`selfcheck.check_catboost_arm` keeps passing across that edit: the parity gate
runs off `MOJOTREES_CATBOOST_MODE_PARITY_SCENARIOS`, not off the scheduling
table, exactly so that unparking is one edit and not a re-audit.

## Part 6. `schema.json`: two stale sentences and three new fields

`bench/real_data/schema.json` is documentation and nothing validates records
against it, so this breaks nothing. It is now wrong.

`engine_resolved_params`'s description ends "This harness pins the first and
records the second per run." **The harness no longer pins the learning rate.**
Replace that sentence with:

> This harness pins NEITHER as of 2026-08-16: `cb-shipped` lets CatBoost
> resolve its own learning rate, so this field is the only place a record says
> what rate the CatBoost row actually trained at. See
> `scenarios.CATBOOST_DELIBERATE_DIVERGENCE`.

Three fields want entries once parts 1 to 3 are applied:

- `engine_resolved_params_drift` -- where a live `get_all_params()` disagrees
  with `scenarios.CATBOOST_LEFT_AT_STOCK`. Empty list is agreement; a non-empty
  list is a CatBoost upgrade having moved a default under a table transcribed
  on 2026-08-16.
- `params.resolved_parity` -- both engines' resolved dicts and the key-by-key
  verdict, on every row rather than only in the manifest.
- The `catboost_readback` sidecar (`catboost_readback.json` under the run
  directory) is not a record field; it is popped by the worker before the
  record is built.

Two other stale strings, in files this lane also did not own:

- `bench/bench_lightgbm.py:447` says `scenarios.catboost_params` passes "the
  matched learning rate". It does not any more.
- `bench/real_data/results/RESULTS_TEMPLATE.md:92` says "Both at **matched tree
  count and matched learning rate**". The template needs the divergence
  sentence instead, and it is the place a published table would inherit the
  misreading from.
