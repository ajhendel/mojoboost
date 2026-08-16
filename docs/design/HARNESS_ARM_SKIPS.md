# `ENGINE_SCENARIO_SKIP` and the third CatBoost arm: a design held back, on purpose

Status: **design note, 2026-08-16. Nothing in this note is implemented and
this lane landed nothing under `bench/`.** Both pieces were specified, both
were scoped against the current tree, and both were held out of the
comparison run rather than merged hours before the measurement window: the
third arm changes the arm set of a run whose arms are fixed, and
`ENGINE_SCENARIO_SKIP` refactors the exact skip machinery that run's
60-runs-3-skips shape rests on, machinery a peer session edited twice within
an hour of this being written. An unrun refactor of a benchmark runner, taken
just before the measurement, is risk with no upside for the table.

This note exists so that landing both after the run is mechanical. It records
where the specification lives, what it replaces, what it cannot replace, the
one constraint the specification does not state, and the reachability trace
that decides whether the third arm is real or is declared and skipped.

Companion reading: `docs/design/CATBOOST_CATALOG.md` A31 (which
`boosting_type` the comparison actually ran, verified from source) and A7
(what ordered boosting is).

---

## 1. Where the `ENGINE_SCENARIO_SKIP` specification lives

**It is not in the repository, on any branch, in any commit message, or in
any file under `bench/results/` or `docs/`.** Searched: the working tree,
every `refs/heads` tip by `git grep`, every commit message, and the whole
filesystem under both checkout roots. Zero occurrences.

It survives only as a fenced diff inside the final report of the **harness
scenario shapes lane**, which committed on `lane/harness-shapes` at
`ca89a72` and proposed the mechanism as glue it deliberately did not build.
The report is a session artifact, not a durable one, which is the whole
reason the specification was never applied and is the reason it is
transcribed in full below rather than referred to.

That is itself a finding worth keeping: a mechanism specified in a lane
report and not in a file is a mechanism that does not exist, however good the
specification is.

### 1.1 The specification, transcribed verbatim

The lane's own framing, quoted:

> Only one package-adjacent change is needed and it is not in
> `src/mojotrees/`; it is the mechanism the harness is missing. **The harness
> cannot skip one peer engine on one scenario.** `CATBOOST_SCENARIO_SUPPORT`
> is per-scenario and drives all of `PEER_ENGINES` together;
> `CATBOOST_TIER_CAP` is per-scenario and binds both CatBoost engines.

The diff:

```diff
--- a/bench/real_data/scenarios.py
+++ b/bench/real_data/scenarios.py
@@ (beside CATBOOST_TIER_CAP)
+#: (engine, scenario) pairs where one arm cannot run a scenario the OTHER
+#: peers can. CATBOOST_SCENARIO_SUPPORT answers "can the CatBoost arm run
+#: this problem" for the whole peer group and CATBOOST_TIER_CAP answers "at
+#: what size"; neither can say "this one engine, on this one scenario",
+#: which is what a grow-policy-dependent CatBoost mechanism needs. Same rule
+#: as everywhere else in this file: a skip with a stated cause, never an
+#: absence.
+ENGINE_SCENARIO_SKIP = {
+    ("catboost_lossguide", "ordered_boosting_small"): (
+        "ordered boosting is a symmetric-tree mechanism in CatBoost and this "
+        "arm passes grow_policy=Lossguide, so the library either refuses the "
+        "pair or resolves to Plain. Either way the row is not an ordered row "
+        "and this scenario's whole identity is that the CatBoost arm runs "
+        "its Ordered default. Declared, not observed: nobody has run the "
+        "combination to find out which of the two happens"
+    ),
+}
+
+
+def engine_scenario_ok(engine, scenario):
+    """(ok, reason). Whether `engine` runs `scenario` at all."""
+    scenario_id = scenario["id"] if isinstance(scenario, dict) else scenario
+    reason = ENGINE_SCENARIO_SKIP.get((engine, scenario_id))
+    return (reason is None), reason

--- a/bench/real_data/run.py
+++ b/bench/real_data/run.py
@@ build_matrix
             for engine in args.engine:
                 if engine not in spec["engines"]:
                     continue
+                ok, reason = scenarios.engine_scenario_ok(engine, spec)
+                if not ok:
+                    jobs.append(_skip(scenario_id, engine, device, args, reason))
+                    continue
                 if engine in scenarios.CATBOOST_ENGINES:

--- a/bench/real_data/selfcheck.py
+++ b/bench/real_data/selfcheck.py
@@ check_catboost_arm, after the CATBOOST_SCENARIO_COST loop
+    for (engine, name), reason in scenarios.ENGINE_SCENARIO_SKIP.items():
+        check(engine in engines.ENGINES,
+              f"ENGINE_SCENARIO_SKIP names unknown engine {engine!r}")
+        check(name in scenarios.SCENARIOS,
+              f"ENGINE_SCENARIO_SKIP names unknown scenario {name!r}")
+        check(engines.ENGINE_ARM.get(engine, "").startswith("peer"),
+              f"ENGINE_SCENARIO_SKIP excuses {engine!r}, which is not a peer "
+              "arm. A comparator or subject row is not skippable by table")
+        check(bool(reason),
+              f"ENGINE_SCENARIO_SKIP({engine}, {name}) gives no reason")
```

and the lane's own note on the last check, which is the part to keep:

> The last check is the one that matters: it makes the mechanism structurally
> unable to remove a `mojotrees` or `lightgbm` row, so it cannot become a way
> to drop a headline cell that went red.

That property is not negotiable and must survive any change to the shape
below. A skip table that can delete a comparator row is a table that will
eventually be used to delete a comparator row.

---

## 2. What it replaces, and what it cannot

### 2.1 The gap it actually closes

The harness has three places that decide whether a cell is scheduled, and
none of them is per engine per scenario:

| mechanism | asks | granularity |
|---|---|---|
| `CATBOOST_SCENARIO_SUPPORT` (`scenarios.py:852`) | can the CatBoost arm run this problem at all | per scenario, drives all of `PEER_ENGINES` together |
| `CATBOOST_TIER_CAP` (`scenarios.py:2391`) | at what size does running it stop being a measurement | per scenario, binds every engine in `CATBOOST_ENGINES` |
| the `if` ladder in `run.py:110-186` | may this engine run on this device | per engine per **device**, scenario-independent |

`ENGINE_SCENARIO_SKIP` is the missing (engine, scenario) cell of that table.
Nothing else can express "this one arm, on this one problem".

### 2.2 Which ad-hoc `run.py` skips it would absorb: **none of them, and that
is the correction to the brief**

The brief describes the mechanism as "replacing ad-hoc `if` skips scattered
through `run.py`". Read against the file, that is not what it can do. Every
skip currently in `build_matrix` is **device-conditional and
scenario-independent**:

- `run.py:120-128`, the scenario declares no such device.
- `run.py:129-138`, LightGBM on a non-CPU device.
- `run.py:149-160`, the CatBoost engines on a non-CPU device.
- `run.py:171-186`, `mojotrees_catboost_mode` on a non-CPU device.

All four sit inside `if device != "cpu":`. A table keyed by (engine,
scenario) has no device axis, so absorbing them would mean either writing a
cross product of every scenario against every engine, which is worse than the
`if`, or widening the key to (engine, scenario, device), which the
specification does not propose and which would put the four device rules in
two places at once during the migration. **Recommendation: land the table for
what it is for, leave the four device skips where they are, and say so in the
docstring** so the next reader does not re-derive this.

There is one narrower move worth taking at the same time and it is not a
migration: `run.py:149-160` and `run.py:171-186` both hard-code the reason
text inline. Those two reasons are engine facts and belong beside the engine,
not in the matrix builder. That is a separate, smaller change.

---

## 3. The constraint the specification does not state: permanent versus dated

`7e41e16` exists solely because a skip's stated reason was the dated one.
The `mojotrees_catboost_mode` GPU skip led with `BLOCK_SCORE_FUNCTION`, a
device block a lane is actively removing. A future reader who followed that
reason would have deleted the skip when the block landed, and the comparison
matrix would have silently gained twelve GPU cells for an arm whose entire
purpose is to be read against CatBoost, which is CPU-only in this harness and
has no GPU row to pair with. The fix was to reorder the sentence so the
permanent reason leads and the dated one follows.

A flat string cannot enforce that. Every entry in `ENGINE_SCENARIO_SKIP` will
be written by somebody who knows why the cell cannot run **today** and who is
under no pressure to separate that from why it must never be scheduled. So
the shape should carry the distinction rather than rely on prose discipline:

```python
ENGINE_SCENARIO_SKIP = {
    (engine, scenario_id): {
        # Required, non-empty. The reason that outlives every current block.
        # This is what a future reader acts on.
        "permanent": "...",
        # Optional. The reason it cannot run today.
        "dated": "...",
        # Required whenever `dated` is set: what removes it.
        "dated_expires_when": "...",
    },
}
```

and `engine_scenario_ok` renders one string with the permanent half first,
so the message reaching `_skip` and the manifest has exactly the shape
`7e41e16` produced by hand. The self-check gains two rules on top of the
lane's four:

- `permanent` is present and non-empty. A skip whose only reason is dated is
  not a skip, it is a bug report.
- `dated` without `dated_expires_when` fails. A dated reason that does not
  say what removes it is how a temporary skip becomes permanent by neglect.

**This diverges from the transcribed specification**, which uses a flat
string. The divergence is deliberate and is the one place this design does
not follow the spec; everything else, including the four self-check rules and
the `run.py` insertion point, is the spec unchanged.

---

## 4. The one entry the specification proposed is now void

The spec's single entry skips `catboost_lossguide` on
`ordered_boosting_small` because "this scenario's whole identity is that the
CatBoost arm runs its Ordered default". **That premise is false and was
falsified after the spec was written**, by `01b71cd`: CatBoost on the CPU
never chooses `Ordered`, at any row count. See A31 for the constants.

So on that scenario the `catboost` row is Plain, the `catboost_lossguide` row
is Plain plus Lossguide, and the lossguide row is an ordinary row that
misleads nobody. **Do not land the spec's entry.** Landing a skip whose
stated reason is untrue is worse than landing no skip.

Two things from that entry are worth keeping, both now settled rather than
guessed:

- The spec hedged, "the library either refuses the pair or resolves to
  Plain... nobody has run the combination". It is a **refusal**, verified
  from source: `catboost_options.cpp:1046-1050` wraps
  `CB_ENSURE(BoostingType == Plain, "Ordered boosting is not supported for
  nonsymmetric trees.")` in `if (GrowPolicy != SymmetricTree)`. The second
  caveat on `SCENARIOS["ordered_boosting_small"]` should be corrected to say
  so when `bench/` reopens.
- Consequently a third arm that sets `boosting_type=Ordered` must **not** set
  `grow_policy`. Inheriting CatBoost's SymmetricTree default is what makes it
  legal.

---

## 5. The third CatBoost arm

### 5.1 Shape

Match `CatBoostLossguideEngine` exactly (`engines.py:1290-1362`), which is
the precedent for a second differently-shaped CatBoost column:

```python
class CatBoostOrderedEngine(CatBoostEngine):
    name = "catboost_ordered"
    variant_params = {"boosting_type": "Ordered"}
```

plus a `load()` that appends the arm note, an entry in `ENGINES` and in
`ENGINE_ARM` as `"peer"`, and the name added to `CATBOOST_ENGINES`
(`scenarios.py:2455`), which is what makes `PEER_ENGINES`, the tier caps and
the device skip pick it up without a fourth hand-written tuple. `boosting_type`
is not in `CATBOOST_REFUSED_PARAMS`, and `variant_params` is merged into
`extra` rather than written over the resolved dict, so the refusal list still
applies to it.

### 5.2 It is a third data point, not a correction. Say that plainly.

A31 settles it: at this harness's tier CatBoost's own default is `Plain`, by
the row clause and by the iteration clause independently. So the two existing
CatBoost arms are **not** comparing against a CatBoost nobody runs; they are
CatBoost as CatBoost runs by default. The third arm adds the mode CatBoost is
famous for and that its defaults do not select on the CPU. Read against the
`catboost` row it isolates the boosting scheme inside one engine, which is
exactly the shape `catboost_lossguide` uses to isolate tree shape.

The consequence for the run that is about to happen, which holds whether or
not the arm is ever built: **no row in the comparison covers CatBoost's
ordered boosting, and this is a stated limit of the table rather than an
assumption.**

### 5.3 The cells, before and after

Counted statically from `SCENARIOS` and the support and cap tables, at
`--tier standard --device cpu` with all five engines. Not from a dry run;
this lane ran nothing.

Six scenarios admit the CatBoost arms: `dense_regression`,
`imbalanced_binary`, `multiclass`, `sparse_highdim`, `ordered_boosting_small`,
`high_cardinality_categorical`. `ranking` and `categorical_missing` do not
(`CATBOOST_SCENARIO_SUPPORT`).

BEFORE, peer arms only:

| engine | scheduled | declared skips |
|---|---|---|
| `catboost` | 5 | 1 (`sparse_highdim`, tier cap at smoke) |
| `catboost_lossguide` | 5 | 1 (same) |
| `mojotrees_catboost_mode` | 6 | 0 (deliberately not tier capped) |

AFTER, if the arm is added and **not** narrowed: `catboost_ordered` adds 5
scheduled cells and 1 tier skip, on five problems where nobody asked for an
ordered row and where CatBoost's ordered ladder is the dearest thing in the
suite.

AFTER, narrowed by `ENGINE_SCENARIO_SKIP`, which is the recommendation:
`catboost_ordered` runs **1** cell, `ordered_boosting_small`, and carries
**5** declared skips, one per remaining scenario, each with its reason in the
manifest. That is one new measured cell and five visible declarations, and it
is the shape the harness has been hardened toward twice this week: a skip
with a stated cause, never an absence.

Note that this is a second use of the mechanism, and a slightly different one
from the spec's: the spec's case is "this arm cannot run this scenario", and
this one is "this arm is not the arm to run on this scenario and the cost of
running it anyway is real". Both carry a reason and both are permanent. The
docstring should cover both causes explicitly rather than let the second
arrive as an undocumented stretch of the first.

### 5.4 Dependency on the MVS lane

None, in the direction anyone expected, and a hard one in the other.

`catboost_ordered` is a CatBoost arm; it reads nothing from
`MOJOTREES_CATBOOST_MODE` and needs no key that lane is adding. If a later
change does need those keys, it must read the dict rather than restate the
values.

The hard direction is the mojotrees counterpart, and it should be recorded
now because it is not obvious. **Our own ordered boosting is mutually
exclusive with any bootstrap**, at `src/mojotrees/boosting.mojo:2481-2485`:

    if ordered.enabled and bootstrap.enabled():
        raise Error(
            "boosting_type=ordered is exclusive with bootstrap_type: a dropped"
            " or reweighted row changes which prefix each fold was fitted on"
        )

and again for row bagging at `python/mojotrees/sklearn.py:1551-1558`. The MVS
lane is putting `bootstrap_type=MVS` and `subsample=0.8` into
`MOJOTREES_CATBOOST_MODE`. Once that lands, **`mojotrees_catboost_mode` can
never also carry `boosting_type='ordered'`**: the pair is refused, correctly,
by name. A mojotrees ordered row therefore has to be its own arm without the
bootstrap, and cannot be produced by adding one key to the CatBoost-mode
dict. Anyone who tries will get a clean refusal rather than a wrong number,
which is the design working, but they will lose an afternoon to it if this is
not written down.

---

## 6. Is our ordered boosting reachable from an arm that sets `boosting_type=Ordered`?

**Yes on the CPU, traced end to end, and refused rather than ignored on every
other path.** The trace, in the order a fit walks it:

| # | file:line | what it does |
|---|---|---|
| 1 | `python/mojotrees/sklearn.py:542, :672` | `boosting_type` is a constructor parameter and is stored |
| 2 | `python/mojotrees/sklearn.py:1176-1214` | `_resolve_boosting()` collapses `boosting_type` / `boosting` / `booster` to one canonical word; an unknown one raises |
| 3 | `python/mojotrees/sklearn.py:1500-1503` | an ordered knob set without `boosting_type='ordered'` is refused by name, not ignored |
| 4 | `python/mojotrees/sklearn.py:1551-1558` | ordered beside row bagging is refused, naming the estimator parameter the user set |
| 5 | `python/mojotrees/sklearn.py:1559-1560` | `out["ordered"] = 1` and `out["ordered_seed"]`, onto the params mapping the binding reads |
| 6 | `python/mojotrees/sklearn.py:2087` | `ordered_boosting=(self._resolve_boosting() == "ordered")` onto the `device_selection.Workload`, so the device decision can see it. Read through `_resolve_boosting` rather than off the attribute, which is what makes `booster='ordered'` visible to the gate |
| 7 | `src/mojotrees/device_policy.mojo:2315-2335` | `BLOCK_ORDERED_BOOSTING`. `device='auto'` takes the CPU; an explicit `device='gpu'` raises |
| 8 | `bindings/_mojotrees.mojo:836-844` | `OrderedBoostingParams.enable(...)` built from `params["ordered"]`, then `validate()` |
| 9 | `bindings/_mojotrees.mojo:845-847` | refused by name when `ordered.enabled and not ordered_ok` |
| 10 | `bindings/_mojotrees.mojo:985` | `ordered_ok=device == CPU_DEVICE` at `fit`, and the flag is omitted at every other entry point, so it defaults False and they refuse |
| 11 | `bindings/_mojotrees.mojo:997-1016` | ordered beside dart, rf or `linear_tree` refused by name |
| 12 | `bindings/_mojotrees.mojo:855-861` | the bundle is moved into `BoosterParams` |
| 13 | `src/mojotrees/boosting.mojo:2255, :2471` | `_boost_rounds` reads `params.ordered`, and `_check_ordered` runs whether or not it is on |
| 14 | `src/mojotrees/boosting.mojo:2481-2485` | ordered beside a bootstrap refused; see 5.4 |
| 15 | `src/mojotrees/boosting.mojo:2509+` | the rung ladder is built, and `src/mojotrees/ordered_boosting.mojo` runs the mechanism A7 describes |
| 16 | `src/mojotrees/train_gpu.mojo:1240`, `train_gpu_sparse.mojo:233` | both GPU trainers refuse `params.ordered.enabled` by name, so a caller who reaches them directly gets a refusal and not a plain-boosting model reported as ordered |
| 17 | `src/mojotrees/boosting.mojo:3122, :3638, :3959` | `check_ordered_honored` in `train_with_valid` and the multiclass trainers, same rule |

Scope of the honest claim: reachable from a **dense, single-output, CPU fit
with no validation set, no custom objective, no bootstrap and no row
bagging**. Every path outside that refuses it by name. So a future mojotrees
ordered arm is **real, not declared-and-skipped**, provided it is declared
inside that envelope, and 5.4 is the constraint that bites first.

---

## 7. Landing checklist, after the run

1. `ENGINE_SCENARIO_SKIP` and `engine_scenario_ok` in `scenarios.py` beside
   `CATBOOST_TIER_CAP`, with the permanent/dated shape from section 3.
2. The four-line insertion in `run.py:build_matrix` at the spec's insertion
   point, above the `CATBOOST_ENGINES` tier check.
3. The six self-check rules in `selfcheck.check_catboost_arm`: the lane's
   four, plus `permanent` non-empty and `dated` implying
   `dated_expires_when`.
4. `CatBoostOrderedEngine` in `engines.py`, registered in `ENGINES` and
   `ENGINE_ARM`, added to `CATBOOST_ENGINES`.
5. Five `ENGINE_SCENARIO_SKIP` entries narrowing `catboost_ordered` to
   `ordered_boosting_small`.
6. Correct the second caveat on `SCENARIOS["ordered_boosting_small"]`: the
   lossguide pair is refused by CatBoost, not silently resolved, and the
   harness now can skip one peer engine on one scenario.
7. Do NOT land the specification's own `catboost_lossguide` entry. Section 4.
8. `--dry-run` and diff the cell list against section 5.3 before anything is
   measured.
