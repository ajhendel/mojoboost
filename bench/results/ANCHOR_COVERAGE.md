# The accuracy anchor gate, and the fact that it has never gated anything

Written 2026-08-17 by a read-only audit lane. **Nothing in this document was
measured by this lane.** No build ran, no test ran, no benchmark ran, no
training ran. Every number here was read out of a file already on disk, and
every claim is marked VERIFIED (read directly, with the citation) or INFERRED
(a conclusion drawn from what was read, with the reasoning shown). Where a
question could not be answered by reading, it says so and becomes a request.

Repository state this was read at, because a citation to a line number is
worthless without it.

- branch `perf-round-2`, HEAD `9a113c8` "Five lanes land: a new plan kernel,
  rule 9 amended twice, section 14 pre-registered"
- **the working tree is DIRTY.** `git status --porcelain` reports modified
  `bench/real_data/README.md`, `pairs.py`, `scenarios.py`, `selfcheck.py` and
  `verify.py`, 473 insertions over 163 deletions. Line numbers below are
  WORKING-TREE line numbers, not `9a113c8` line numbers, and the uncommitted
  diff touches the anchor code (it adds `stale_anchor_constant` and rewrites
  `anchor_configuration`). A reader checking these citations against the commit
  will not find them where this says they are.

---

## Part 1. The timeline, and the intent

### The answer

**(a). The file was created empty this morning and the adoption step was never
performed.** It is NOT a regression. No commit has ever added an entry to
`anchors`, and therefore no commit has ever removed one.

### The evidence

`git log --all -- bench/real_data/accuracy_anchors.json` returns exactly two
commits and no others. VERIFIED.

```
abbbf98  2026-08-17 11:53:25 -0400  Symmetric trees 2.1x, predict 15x, and nine silent wrong answers fixed
9a113c8  2026-08-17 12:38:56 -0400  Five lanes land: a new plan kernel, rule 9 amended twice, section 14 pre-registered
```

`git show abbbf98 -- bench/real_data/accuracy_anchors.json` is the file's
creation and the added lines end with these two, which are the entire payload
below the `_about` array. VERIFIED.

```
+  "version": 1,
+  "anchors": {}
```

`git show 9a113c8 -- bench/real_data/accuracy_anchors.json` is 36 insertions and
1 deletion, and every one of them is inside the `_about` array. The `anchors`
key does not appear on either side of that diff. VERIFIED by reading the diff
and confirming that the only `+` lines matching `anchors` are prose lines such
as "NOTHING IS STALE TODAY BECAUSE NOTHING IS ANCHORED. `anchors` is empty and
has been since the concept landed".

So the state is original, not lost. INFERRED from the two facts above, and the
inference is tight because two commits is the whole history of the path.

### What the adopting commit claimed

`abbbf98`'s message contains this line.

> rule 3 rewritten so accuracy is gated against our own recorded absolute anchor

VERIFIED by `git log -1 --format=%B abbbf98`. The claim is false as stated in
the same commit that made it, because that commit's own diff records zero
anchors. The mechanism was rewritten. The gate was not armed.

The second commit, `9a113c8`, is the one that noticed. Its `_about` addition
says so plainly, and it is the most honest sentence in the repository on this
subject.

> NOTHING IS STALE TODAY BECAUSE NOTHING IS ANCHORED.

VERIFIED, `bench/real_data/accuracy_anchors.json`, in the `_about` array.

### One thing worth saying about the shape of the failure

The `_about` array is 76 lines of argument for a mechanism, over a payload of
`{}`. The file is internally consistent, it documents its own emptiness, and it
still leaves every reader of `LANE_RULES.md` rule 3 believing that a gate is
running. That is the same shape as rule 8's "an identity that carries a
dimension it should not, or lacks one it needs. It stays internally consistent,
so it passes", quoted from `abbbf98`'s own commit message. INFERRED, offered as
a reading of the failure rather than as a fact about the code.

---

## Part 2. The mechanism, exactly

All citations in this part are to `bench/real_data/verify.py` in the working
tree. All VERIFIED by reading.

### The key

`_anchor_key`, lines 1303 to 1336. Seven components, `|`-joined, in this order.

```
scenario | tier | data_kind | dataset | arm | device | n<trees>
```

- `scenario` from `record["scenario"]`
- `tier` from `record["tier"]`
- `data_kind` from `record["data"]["data_kind"]`, which is `"real"` or
  `"synthetic"` (`loaders.py:490` writes `"real"`; `generators.py:20` records
  that generated data is `"synthetic"`)
- `dataset` from `record["data"]["dataset"]`, which is the registry name for
  real data and `generated:<generator>` for synthetic
- `arm` from `_arm_of` (lines 983 to 987), which is `record["arm"]` falling back
  to `record["engine"]`
- `device` from `record["device_used"]` falling back to
  `record["device_requested"]`
- `n<trees>` from `_tree_count` (lines 944 to 962), which reads
  `params.num_boost_round`, then `params.engine.n_estimators`,
  `num_boost_round`, `iterations`, then gives up and returns `None`

Threads are deliberately absent and the device is deliberately present. The
reasons are argued in the docstring at 1306 to 1318 and this lane found no
fault with either argument.

### The worked example that was asked for

The cell `dense_regression / standard tier / real data / year_prediction_msd /
arm mojotrees / device gpu / 100 trees` has exactly this key.

```
dense_regression|standard|real|year_prediction_msd|mojotrees|gpu|n100
```

VERIFIED against a real record rather than derived on paper. That key is
produced by `bench/real_data/results/20260817T162834Z-lam1/records.json`, at
commit `b71b9b9`, with `primary_metric = rmse` and `quality.rmse =
9.10606505792536`. So this is the one cell in the audit where the anchor is one
adoption commit away from existing.

### The tolerance actually configured

Two sources, and the second wins where it overlaps. `check_accuracy_anchor` at
1653 to 1654 builds `dict(DEFAULT_ACCURACY_ANCHOR)` and then updates it from
`config["defaults"]["accuracy_anchor"]`.

`DEFAULT_ACCURACY_ANCHOR`, lines 1274 to 1279.

```python
{
    "max_worse_relative": 0.0025,
    "implausible_better_relative": 0.05,
    "engines_judged": list(SUBJECT_ENGINES),
    "gating": True,
}
```

`thresholds.json`, `defaults.accuracy_anchor`, carries `max_worse_relative
0.0025`, `implausible_better_relative 0.05`, `gating true` and a long
`rationale`. It does NOT carry `engines_judged`. VERIFIED by parsing the file.

So the effective policy is **0.25 percent worse is a FAIL, more than 5 percent
better is a WARN, and the judged set is the five names in `SUBJECT_ENGINES`**
(`mojotrees`, `mojotrees_depthwise`, `mojotrees_catboost_mode`,
`mojotrees_cosine_leafwise`, `mojotrees_symmetric_colsample`, at lines 723 and
following). The 0.25 percent is stated in its own docstring to be part measured
and part inferred, and the inferred part is named as such at 1250 to 1257. That
disclosure is correct and this lane has nothing to add to it.

### What the check does, case by case

Read at 1662 to 1824.

| condition | outcome | line |
|---|---|---|
| `record["engine"]` not in `engines_judged` | record ignored entirely | 1665 |
| `_tree_count` is `None` | SKIP, abstains | 1674 |
| primary metric missing or nan | SKIP, abstains | 1681 |
| **key missing from `anchors`** | **WARN**, "NO ANCHOR RECORDED", added to `uncovered` | 1688 to 1701 |
| anchor carries no `configuration` block | WARN, "ANCHOR CURRENCY UNKNOWN", does not gate | 1711 to 1719 |
| anchor's unset parameter no longer resolves the same | WARN, "STALE ANCHOR", does not gate, value is NOT compared | 1720 to 1731 |
| anchor's `primary_metric` differs from the run's | WARN, not compared | 1733 to 1743 |
| anchor entry has no `value` | WARN | 1745 to 1751 |
| worse by more than tolerance, same machine | **FAIL** | 1784 to 1798 |
| worse by more than tolerance, different machine | WARN, and the line names the mismatch | 1789 |
| better by more than `implausible_better_relative` | WARN | 1799 to 1809 |
| inside tolerance | PASS | 1811 |
| any `uncovered` at all | one run-scope NOTE naming the adoption procedure | 1813 to 1824 |

Two structural points follow from that table and both are load-bearing.

**A missing anchor is a WARN and not a FAIL.** The reasoning is given at 1636 to
1643 and it is sound on its own terms, because a harness that refuses to run a
new arm gets routed around. The consequence is that a run with zero anchors
exits green. VERIFIED from the status assignment; INFERRED as to the exit code,
because this lane did not run `verify.py` and did not trace `Verdict` to the
process exit.

**Staleness is tested before the value is used, never after.** Comment at 1702
to 1707. A stale anchor can produce neither a PASS nor a FAIL. This is right,
and it means a stale anchor and a missing anchor have identical coverage, which
is none.

### Can anything write the anchors file

The `_about` claims "No code path writes this file." **The claim is true as
written and it has one hole that the claim does not cover.** Both halves
VERIFIED.

True as written. `ACCURACY_ANCHORS_PATH` (line 1220) appears in exactly three
places in `bench/real_data`, and none of them is a write.

- `verify.py:1605`, inside `load_accuracy_anchors`, as a read default
- `selfcheck.py:2391`, as the subject of an assertion
- nowhere else, by `grep -rn ACCURACY_ANCHORS_PATH bench/real_data/*.py`

Every `open(` in `verify.py` is at lines 144, 150, 1451, 1608, 1921 and 2201.
Only 1921 and 2201 open for writing. 1921 is `propose_anchors` writing the
caller's `--propose-anchors` path. 2201 is `--json` writing the caller's report
path. Neither is `ACCURACY_ANCHORS_PATH`.

`selfcheck.py:2397` to `2405` asserts this by reading `verify.py`'s own source
text and requiring that `'open(ACCURACY_ANCHORS_PATH, "w")'` does not appear in
it, and that `open(path, "w")` does appear after `def propose_anchors`.

**The hole.** `propose_anchors` (1827 to 1924) never compares its `path`
argument against `ACCURACY_ANCHORS_PATH`. So

```
python bench/real_data/verify.py <run> --propose-anchors bench/real_data/accuracy_anchors.json
```

overwrites the live anchors file with machine-proposed values, in one command,
with no human reading any entry, and the `selfcheck` assertion above does not
trip because it is a text match on the source and the source is unchanged. That
single command is exactly the ratchet the file's `_about` says cannot happen by
accident. VERIFIED by reading `propose_anchors` end to end and finding no guard.

**REQUEST R1, for the orchestrator.** Add a refusal in `propose_anchors` when
`os.path.abspath(path) == os.path.abspath(ACCURACY_ANCHORS_PATH)`, and a
`selfcheck` assertion that the refusal exists. This is a source change, not a
measurement, so it costs no run.

### Two more mechanism gaps found by reading

**GAP 1. `shipped_at_record` does not record what it says it records.**
`anchor_configuration` (1468 to 1519) documents the field as "the value the
relevant `sklearn.py` constant held when this run was taken". It is computed at
line 1504 by `shipped_constant(constant)`, which at 1450 to 1452 opens
`python/mojotrees/sklearn.py` and reads it **at proposal time, off the working
tree**. Those are the same value only when `--propose-anchors` runs at the same
code state as the run it reads. Propose anchors from a run taken before a
default moved and the block records today's constant, after which
`anchor_staleness` compares today against today and reports "not stale" for an
anchor taken on a model nobody ships. The `_about`'s claim that "an anchor
proposed AFTER a default moves is current by construction" holds; the unstated
converse, that an anchor proposed after a default moves FROM A RUN TAKEN BEFORE
IT is also current, is false and the mechanism cannot tell. VERIFIED by reading
both functions. **REQUEST R2.** Record `shipped_at_record` from the run's own
record if the run carries it, or refuse to propose anchors when the run's
`environment.git.commit` is not HEAD.

**GAP 2. Staleness is unreachable for every arm that has ever run.**
`anchor_staleness` only fires on parameters in `followed_default`, and
`followed_default` is the subset of `STALE_ANCHOR_PARAMETERS` that the harness
passed as `None` (1498 to 1502). `STALE_ANCHOR_PARAMETERS` is exactly
`lambda_l2` and `learning_rate` (1388 to 1414). `scenarios.BASE_PARAMS` pins
`learning_rate 0.1` and `lambda_l2 0.0` and `scenarios.mojotrees_params` copies
both onto every mojotrees arm, so both resolve to a number on every record. This
lane confirmed it on the newest run. In `20260817T162834Z-lam1`, `mojotrees`
resolves `lambda_l2 = 0.0`, `mojotrees_depthwise` resolves `1.0`, and
`mojotrees_catboost_mode` resolves `3.0`, with no `None` anywhere. So an anchor
proposed from ANY run on disk would carry `followed_default = []` and could
never be judged stale. VERIFIED.

This is not a defect in isolation, and the docstring at 1482 to 1486 argues the
case correctly. It becomes a defect in combination with the fact that the two
parameters the mechanism tracks are the only two it tracks, and neither one is
detectable on any arm that has run. The mechanism IS reachable for the arms
`pairs.py` defines, because `pairs.SHIPPED_LOSSGUIDE` and
`pairs.SHIPPED_SYMMETRIC` pass `lambda_l2: None` and `learning_rate: None`
deliberately, and that is what makes them our product rather than a mirror.
VERIFIED at `bench/real_data/pairs.py:591` and `618` to `630`. **So the
staleness mechanism works for the plan that has never run and is inert for every
run that exists.**

---

## Part 3. The coverage hole, and what section 14 actually costs

### Scenarios and tiers that exist

`bench/real_data/scenarios.py`, `TIERS` at line 2778 and `SCENARIOS` at 3235.
VERIFIED by parsing the module's AST and by importing it (import only, nothing
built and nothing trained).

Tiers: `smoke`, `standard`, `large`.

| scenario | real dataset | generator | devices declared | primary metric |
|---|---|---|---|---|
| `dense_regression` | `year_prediction_msd` | `dense_regression` | cpu, gpu | rmse |
| `imbalanced_binary` | `bank_marketing` | `imbalanced_binary` | cpu, gpu | average_precision |
| `multiclass` | `covertype` | `multiclass` | cpu, gpu | multi_logloss |
| `ranking` | `mslr_web10k` | `ranking` | cpu only | ndcg@10 |
| `categorical_missing` | `adult` | `categorical_missing` | cpu only | rmse |
| `high_cardinality_categorical` | none | `high_cardinality_categorical` | cpu only | auc |
| `ordered_boosting_small` | none | `dense_regression` | cpu only | rmse |
| `sparse_highdim` | `rcv1_train_binary` | `sparse_highdim` | cpu only | auc |

`devices` defaults to `["cpu"]` when a scenario does not name it
(`_scenario`, line 3230). Six of the eight scenarios have both a real and a
synthetic variant, so the `data_kind` and `dataset` components of the key are
two distinct cells per scenario and not one.

The resolved `engines` list per scenario is amended at import time (four sites,
`scenarios.py:6432`, `6446`, `6475`, `6643`). Resolved values, VERIFIED by
import. **`mojotrees_catboost_mode` is declared on `dense_regression`,
`imbalanced_binary` and `ordered_boosting_small`, and is NOT declared on
`multiclass`.** That single fact decides most of what follows.

### What is on disk

`bench/real_data/results/` holds 90 entries. Scanned every `records.json`,
kept records with `status == "ok"` whose `engine` is in `SUBJECT_ENGINES`, and
computed `_anchor_key` by hand from the same fields `verify._anchor_key` reads.
**39 distinct anchor keys are fillable from runs already on disk.** Each line
below is `key`, then the number of distinct runs producing it, then the newest
such run, then that run's primary metric value and git commit. VERIFIED.

```
dense_regression|large|real|year_prediction_msd|mojotrees_catboost_mode|cpu|n100        1  20260816T180302Z-decision      rmse 9.21605     dcc99eee6
dense_regression|large|real|year_prediction_msd|mojotrees|cpu|n100                      7  20260816T184828Z-w-r5-cpu      rmse 9.10383     6721d2605
dense_regression|large|real|year_prediction_msd|mojotrees|gpu|n100                     16  20260816T185840Z-L-r5-gpudev   rmse 9.10607     6721d2605
dense_regression|large|synthetic|generated:dense_regression|mojotrees_catboost_mode|cpu|n100   4  20260817T113729Z-oblw0p1  rmse 0.303431   c775959cc
dense_regression|large|synthetic|generated:dense_regression|mojotrees_catboost_mode|gpu|n100   3  20260817T113729Z-oblw0p1  rmse 0.303271   c775959cc
dense_regression|large|synthetic|generated:dense_regression|mojotrees_depthwise|cpu|n100       1  20260817T110847Z-dense1mfixed  rmse 0.307403  c775959cc
dense_regression|large|synthetic|generated:dense_regression|mojotrees_depthwise|gpu|n100       1  20260817T110847Z-dense1mfixed  rmse 0.307367  c775959cc
dense_regression|large|synthetic|generated:dense_regression|mojotrees|cpu|n100                 3  20260817T110847Z-dense1mfixed  rmse 0.307782  c775959cc
dense_regression|large|synthetic|generated:dense_regression|mojotrees|gpu|n100                 3  20260817T110847Z-dense1mfixed  rmse 0.307782  c775959cc
dense_regression|smoke|real|year_prediction_msd|mojotrees_catboost_mode|cpu|n100        2  20260816T180034Z-smoke4        rmse 9.21605     9de2ad4c4
dense_regression|smoke|real|year_prediction_msd|mojotrees|cpu|n100                      2  20260816T180034Z-smoke4        rmse 9.10383     9de2ad4c4
dense_regression|smoke|real|year_prediction_msd|mojotrees|gpu|n100                      6  20260816T180034Z-smoke4        rmse 9.10607     9de2ad4c4
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_catboost_mode|cpu|n100   3  20260817T110739Z-smoke3   rmse 0.364531   c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_catboost_mode|gpu|n100   2  20260817T110739Z-smoke3   rmse 0.364531   c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_cosine_leafwise|cpu|n100      1  20260817T130902Z-covsmoke  rmse 0.363492  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_cosine_leafwise|gpu|n100      1  20260817T130902Z-covsmoke  rmse 0.363492  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_depthwise|cpu|n100            1  20260817T110739Z-smoke3    rmse 0.353745  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_depthwise|gpu|n100            1  20260817T110739Z-smoke3    rmse 0.353745  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_symmetric_colsample|cpu|n100  1  20260817T130902Z-covsmoke  rmse 0.333835  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees_symmetric_colsample|gpu|n100  1  20260817T130902Z-covsmoke  rmse 0.333835  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees|cpu|n100                      4  20260817T110739Z-smoke3    rmse 0.354845  c775959cc
dense_regression|smoke|synthetic|generated:dense_regression|mojotrees|gpu|n100                      4  20260817T110739Z-smoke3    rmse 0.354845  c775959cc
dense_regression|standard|real|year_prediction_msd|mojotrees_catboost_mode|cpu|n100     1  20260817T162834Z-lam1          rmse 9.09481     b71b9b9df
dense_regression|standard|real|year_prediction_msd|mojotrees_catboost_mode|gpu|n100     1  20260817T162834Z-lam1          rmse 9.09686     b71b9b9df
dense_regression|standard|real|year_prediction_msd|mojotrees_depthwise|cpu|n100         1  20260817T162834Z-lam1          rmse 9.10339     b71b9b9df
dense_regression|standard|real|year_prediction_msd|mojotrees_depthwise|gpu|n100         1  20260817T162834Z-lam1          rmse 9.10111     b71b9b9df
dense_regression|standard|real|year_prediction_msd|mojotrees|cpu|n100                   1  20260817T162834Z-lam1          rmse 9.10383     b71b9b9df
dense_regression|standard|real|year_prediction_msd|mojotrees|gpu|n100                   1  20260817T162834Z-lam1          rmse 9.10607     b71b9b9df
dense_regression|standard|synthetic|generated:dense_regression|mojotrees_catboost_mode|cpu|n100  6  20260817T135701Z-final  rmse 0.308262  c775959cc
dense_regression|standard|synthetic|generated:dense_regression|mojotrees_catboost_mode|gpu|n100  6  20260817T135701Z-final  rmse 0.307693  c775959cc
dense_regression|standard|synthetic|generated:dense_regression|mojotrees_depthwise|cpu|n100      4  20260817T133745Z-predfix  rmse 0.325803  c775959cc
dense_regression|standard|synthetic|generated:dense_regression|mojotrees_depthwise|gpu|n100      4  20260817T133745Z-predfix  rmse 0.324934  c775959cc
dense_regression|standard|synthetic|generated:dense_regression|mojotrees|cpu|n100                6  20260817T135701Z-final    rmse 0.310775  c775959cc
dense_regression|standard|synthetic|generated:dense_regression|mojotrees|gpu|n100                6  20260817T135701Z-final    rmse 0.310847  c775959cc
high_cardinality_categorical|smoke|synthetic|generated:high_cardinality_categorical|mojotrees|cpu|n100     2  20260816T180034Z-smoke4  auc 0.84111    9de2ad4c4
high_cardinality_categorical|standard|synthetic|generated:high_cardinality_categorical|mojotrees|cpu|n100  2  20260817T102725Z-cat1m   auc 0.841028   c775959cc
imbalanced_binary|smoke|real|bank_marketing|mojotrees|cpu|n100                          2  20260816T144520Z-agree-f64-lambda0  average_precision 0.672844  2deb1b6b6
imbalanced_binary|smoke|real|bank_marketing|mojotrees|gpu|n100                          6  20260816T144520Z-agree-f64-lambda0  average_precision 0.677738  2deb1b6b6
multiclass|smoke|real|covertype|mojotrees|gpu|n100                                      4  20260816T144339Z-gaincross-lambda0  multi_logloss 0.737755    2deb1b6b6
```

Read the shape of that list rather than the rows. **34 of the 39 are
`dense_regression`.** Every tree count is `n100`. `ranking`,
`categorical_missing`, `ordered_boosting_small` and `sparse_highdim` produce
zero fillable keys. `imbalanced_binary` produces two and `multiclass` produces
one, all at `smoke` tier on the `real` variant.

### The manifests behind the candidate runs

Requested per-run detail, from each `manifest.json` plus the ok records in it.
Only the runs that matter to Part 3 are listed; the remaining runs are all
`dense_regression` repeats of rows already above. VERIFIED.

| run id | created (UTC) | commit | tier | variant arg | scenarios with ok records | data kinds | our arms | devices |
|---|---|---|---|---|---|---|---|---|
| `20260815T014008Z` | 08-15T01:40 | `894e3aed` | standard | auto | categorical_missing, dense_regression, imbalanced_binary, ranking, sparse_highdim | real, synthetic | none usable | none |
| `20260815T014842Z` | 08-15T01:48 | `280f68e5` | standard | auto | the six above plus multiclass | real, synthetic | none usable | none |
| `20260815T023123Z` | 08-15T02:31 | `44edb081` | standard | auto | the same six | real, synthetic | none usable | none |
| `20260816T092617Z-gpu-crossgain` | 08-16T09:26 | `12ac7832` | smoke | auto | dense_regression, imbalanced_binary, multiclass | real | mojotrees | gpu |
| `20260816T093004Z-gpu-shippedgain` | 08-16T09:30 | `12ac7832` | smoke | auto | same three | real | mojotrees | gpu |
| `20260816T093347Z-gpu-subtractive` | 08-16T09:33 | `12ac7832` | smoke | auto | same three | real | mojotrees | gpu |
| `20260816T144339Z-gaincross-lambda0` | 08-16T14:43 | `2deb1b6b` | smoke | auto | dense_regression, imbalanced_binary, multiclass | real | mojotrees | gpu |
| `20260816T144510Z-agree-f32-lambda0` | 08-16T14:45 | `2deb1b6b` | smoke | auto | imbalanced_binary | real | mojotrees | cpu, gpu |
| `20260816T144520Z-agree-f64-lambda0` | 08-16T14:45 | `2deb1b6b` | smoke | auto | imbalanced_binary | real | mojotrees | cpu, gpu |
| `20260816T180034Z-smoke4` | 08-16T18:00 | `9de2ad4c` | smoke | auto | dense_regression, high_cardinality_categorical | real, synthetic | mojotrees, mojotrees_catboost_mode | cpu, gpu |
| `20260816T181826Z-categorical` | 08-16T18:18 | `dcc99eee` | standard | auto | high_cardinality_categorical | synthetic | mojotrees | cpu |
| `20260817T102725Z-cat1m` | 08-17T10:27 | `c775959c` | standard | auto | high_cardinality_categorical | synthetic | mojotrees | cpu |
| `20260817T135701Z-final` | 08-17T13:57 | `c775959c` | standard | synthetic | dense_regression | synthetic | mojotrees, mojotrees_catboost_mode | cpu, gpu |
| `20260817T162834Z-lam1` | 08-17T16:28 | `b71b9b9d` | standard | auto | dense_regression | real | mojotrees, mojotrees_catboost_mode, mojotrees_depthwise | cpu, gpu |

**Why the three Aug 15 standard-tier runs are unusable, which is the surprise
in this table.** They are the only runs on disk that cover `ranking`,
`sparse_highdim`, `categorical_missing` and `multiclass` at standard tier. Their
engine field says `mojoboost`, not `mojotrees`, because they predate the rename.
`mojoboost` is not in `SUBJECT_ENGINES`, so `check_accuracy_anchor` skips every
one of those records at line 1665 and `propose_anchors` skips them at 1848.
VERIFIED by counting engines in `20260815T023123Z/records.json`, which is 27 ok
`mojoboost` records and 18 ok `lightgbm` records. **The broadest scenario
coverage this repository has ever produced is invisible to the anchor gate.**
This lane makes no recommendation about renaming inside frozen artifacts; the
standing memory rule is that `MOJOBOOST_*` survives in frozen result artifacts,
and rewriting them to fill anchors would be adopting anchors from a model three
days and many commits old anyway.

### What section 14's veto requires

Section 14 of `docs/design/ACCURACY_BUDGET.md` (line 1982 onward) pre-registers
four cells over the shipped symmetric CatBoost-mode default's `bootstrap_type`.
D is the control at MVS 0.8, A is MVS with noise off, B is Bayesian on the
device round, C is the ship candidate at `bootstrap_type = 'No'`. Step 1 of the
decision rule, at 2034 to 2039, is the veto.

> **The anchor gate comes FIRST and it is a veto, not a contribution.** A
> candidate must be within the accuracy anchor's **0.25 percent per scenario**
> before it is a candidate at all.

And at 2050 to 2052.

> Either failing the 0.25 percent anchor on any scenario, or passing it on
> `dense_regression` while failing on `imbalanced_binary` or `multiclass`

For that to be executable, an anchor must exist for the shipped symmetric arm on
each named scenario, on the device the cells run on, at the tree count they run
at. The plan that carries those rows is `bench/real_data/pairs.py`, whose
`SCENARIO_ROWS` (line 503) are `dense_std` (dense_regression, standard,
synthetic), `dense_large` (dense_regression, large, synthetic), `dense_real`
(dense_regression, large, real), `imbalanced` (imbalanced_binary, standard,
synthetic) and `multiclass` (multiclass, standard, synthetic). `N_ESTIMATORS`
is `scenarios.BASE_PARAMS["n_estimators"]`, which is 100. The shipped symmetric
arm's id is `shipped.symmetric` and the full arm id is `<row>.shipped.symmetric`
(`pairs.py:734`). VERIFIED.

So the required keys, on the GPU leg the section's cells are about, are these
five.

```
dense_regression|standard|synthetic|generated:dense_regression|dense_std.shipped.symmetric|gpu|n100
dense_regression|large|synthetic|generated:dense_regression|dense_large.shipped.symmetric|gpu|n100
dense_regression|large|real|year_prediction_msd|dense_real.shipped.symmetric|gpu|n100
imbalanced_binary|standard|synthetic|generated:imbalanced_binary|imbalanced.shipped.symmetric|gpu|n100
multiclass|standard|synthetic|generated:multiclass|multiclass.shipped.symmetric|gpu|n100
```

INFERRED, not verified, on one point. The `dataset` component for a synthetic
`imbalanced_binary` and `multiclass` is written here as
`generated:imbalanced_binary` and `generated:multiclass`, by analogy with the
`generated:dense_regression` and `generated:high_cardinality_categorical` values
observed on disk. This lane did not find the line that formats that string and
did not run anything to confirm it. Treat the two names as probable, not
certain.

### How many of those five can be filled from disk

**Zero. Not "some are missing". None of them can be filled, and the reason is
structural rather than a gap in coverage.**

`_anchor_key` takes its `arm` component from `_arm_of`, which is
`record["arm"]` before `record["engine"]`. Every plan arm carries a plan arm id
in that field. This lane scanned all 90 result directories and counted the
records whose `arm` differs from `engine`.

```
records whose arm differs from engine: 0
```

VERIFIED. **No run on disk has ever carried a plan arm id.** Every existing
record's arm is an engine name. So every anchor key any `frontier.py` or
`pairs.py` cell will ever produce is missing by construction, no matter how
much dense_regression data accumulates, and this is true of the four `l2.<value>`
axis arms and the three mirror arms too, not only of `shipped.symmetric`.

That is the single most important finding in Part 3, and it means the cost of
executing section 14 is not "two more scenarios". It is a full plan run.

### And one of the five cannot be measured at all

**`multiclass|standard|synthetic|generated:multiclass|multiclass.shipped.symmetric|gpu|n100`
is unreachable at head, by two independent declared refusals.** VERIFIED by
reading both declarations.

First, `pairs._skip_for` at line 797 refuses `arm_id == "shipped.symmetric"` on
`scenario == "multiclass"` before anything else runs, returning
`symmetric_multiclass` on cpu and `symmetric_multiclass_gpu` on gpu. The
declared reason, quoted from `pairs.UNREACHABLE` at line 656, is that

> the symmetric shipped arm carries bootstrap_type=MVS with NO explicit
> mvs_reg, and `sampling.check_mvs_reg_is_set` refuses a DERIVED mvs_reg on a
> softmax round

with a second and independent GPU refusal at line 677, that
`train_multiclass_gpu` takes no bootstrap bundle so `model.mojo::fit_multiclass`
raises. The declared exit for the first is "an explicit mvs_reg, which is then a
value CatBoost does not use here, so it is a different arm rather than this one
unblocked".

Second, and this is the one that survives even if the cells drop MVS,
`scenarios.SCENARIOS["multiclass"]["engines"]` resolves to `['mojotrees',
'lightgbm', 'mojotrees_depthwise', 'catboost', 'xgboost']` and does not contain
`mojotrees_catboost_mode`. `run.py:552` and `run.py:574` skip any engine not in
that list. The exclusion comes from
`scenarios.MOJOTREES_CATBOOST_MODE_SCENARIO_SUPPORT["multiclass"]`, which is
keyed by ENGINE NAME and does not inspect a `bootstrap_type` override, so cells
B and C, whose whole point is that MVS is gone, are refused on multiclass for
the MVS reason anyway.

**INFERRED consequence, and the orchestrator should rule on it rather than this
lane.** Section 14 registers "failing on `multiclass`" as disqualifying for a
change whose control cell cannot run on `multiclass`. As written the veto has a
branch that can never be evaluated, and a rule with an unevaluable branch is
decided by whoever interprets it later, which is the exact thing
pre-registration exists to prevent. Two honest repairs exist. Either the section
names the scenarios the arm can actually reach and says why `multiclass` is
excluded, or `multiclass` becomes reachable first via the declared exit, an
explicit `mvs_reg`, which the refusal itself notes makes it a different arm.

### The minimum set of measurements that makes section 14's veto runnable

Stated as a request, since this lane runs nothing.

**REQUEST R3, the measurement.** One `pairs`-plan run at HEAD, on the quiet box
under the C-ops conditions, covering at minimum the `dense_std`, `dense_large`,
`dense_real` and `imbalanced` rows for the `shipped.symmetric` arm on both cpu
and gpu legs, at 100 trees. The cpu leg is not optional even though the veto is
about the gpu cells, because `_anchor_key` carries the device and
`check_device_agreement` needs the twin. That is four anchor keys on the gpu leg
plus four on the cpu leg. Then `verify.py <run> --propose-anchors <path>`, then a
human reads all eight, fills `recorded_at`, `recorded_by` and `why`, and moves
them into `accuracy_anchors.json` in a commit that says what they were adopted
for.

**REQUEST R4, the ruling that must come before R3 and costs nothing to make.**
Decide what section 14 means on `multiclass`, per the paragraph above. Running
R3 first and discovering the branch is unevaluable is how a pre-registered rule
gets renegotiated at the moment it binds.

### Four more facts that bear on adopting anchors from anything on disk

1. **Every run on disk predates HEAD's source changes.** The newest run is at
   `b71b9b9`, whose only files are `LANE_RULES.md` and `ACCURACY_BUDGET.md`. The
   commit after it, HEAD `9a113c8`, changes nine `src/mojotrees/*.mojo` files
   including `gpu_leaf_batching.mojo` at 839 lines changed, `sampling.mojo`,
   `train_gpu.mojo`, `gpu_split_search.mojo` and `histogram_gpu.mojo`. VERIFIED
   from `git show --stat`. So an anchor adopted from any existing run describes a
   model built before those changes, and `anchor_staleness` cannot see that
   because it tracks two Python-level parameters and nothing about the kernels.
   INFERRED as to whether those changes move any metric, which this lane cannot
   know and must not guess.
2. **An anchor on arm `mojotrees` does not anchor the model we ship.**
   `BASE_PARAMS` pins `lambda_l2: 0.0` (line 236) to keep the LightGBM mirror a
   mirror, while `python/mojotrees/sklearn.py:71` ships `_LAMBDA_L2 = 1.0`. The
   comment at `scenarios.py:233` to `235` says so directly, that "Our shipped
   default is a separate row, and the two must not be read as the same arm".
   VERIFIED. The shipped row is `pairs.SHIPPED_LOSSGUIDE`, which has never run.
3. **The variant arms have no accuracy gate of any kind today.** `verify.py:718`
   to `722` states that `check_differential` is deliberately NOT widened past
   the plain arm and that "Accuracy on the variant arms is gated by
   `check_accuracy_anchor` instead". `check_differential` pairs only
   `("mojotrees", "cpu")` against `("lightgbm", "cpu")` (line 604). With
   `anchors` empty, `mojotrees_depthwise`, `mojotrees_catboost_mode`,
   `mojotrees_cosine_leafwise` and `mojotrees_symmetric_colsample` have nothing
   gating their accuracy at all, and neither does any GPU row of the plain arm.
   VERIFIED. `check_device_agreement` still compares a gpu row against its cpu
   twin, which catches a backend disagreement and cannot catch a change that
   moves both legs together.
4. **`check_coverage` reports a cell as covered when the anchor gate says it
   cannot run.** `COVERING_CHECKS` includes `accuracy_anchor` (line 2049), and a
   cell counts as named if any covering check wrote a line carrying that
   scenario and arm as scope components (line 2107). The "NO ANCHOR RECORDED"
   WARN carries exactly that scope. So the run-scope line "N of N subject cells
   named by a per-cell check" can read PASS on a run where the accuracy gate
   covered nothing. VERIFIED. The docstring is explicit that the check tests for
   silence and never fails, so this is a reader trap rather than a code defect,
   and it is worth one clause in that PASS line saying that named is not gated.

---

## Part 4. The over-claims

Fifteen sites. For each one the quote, then one line on what would make it
true. Nothing here was edited; the orchestrator applies these. All quotes
VERIFIED by reading the cited line.

**1. `bench/results/LANE_RULES.md:47`**
> **The gate is OUR OWN accuracy, and it stopped being a peer comparison on
> 2026-08-17.**

Present tense about a gate with an empty reference. True when at least one
anchor is adopted, or immediately if the sentence says the mechanism replaced
the peer comparison and is not yet populated.

**2. `bench/results/LANE_RULES.md:73` to `76`**
> The anchor is recorded per arm and per scenario, it lives in a file, and it
> moves only by a deliberate act that shows up in a diff, so drift accumulates
> against a fixed point.

"Is recorded" is false for every arm and every scenario. True when anchors
exist, or if amended to "is recorded, once a run has been adopted; today the
file is empty and every arm reads UNCOVERED".

**3. `bench/results/LANE_RULES.md:298`** (rule 9, internal choices)
> It must be exact or accuracy-neutral, meaning within `device_agreement`
> tolerance and clean against the recorded anchor.

There is no recorded anchor, so the second half of the conjunction is
unevaluable and rule 9 currently reduces to the `device_agreement` tolerance
alone. True when anchors exist for the arms rule 9 is applied to, which is a
strong argument for R3.

**4. `docs/design/ACCURACY_BUDGET.md:183`**
> An internal choice has to be accuracy-neutral (exact, or within
> `device_agreement` tolerance and verified against the recorded anchor)

Same defect as site 3, in the document that carries the pricing. Same repair.

**5. `docs/design/ACCURACY_BUDGET.md:1934` to `1942`** (section 13, and this one
is the sharpest)
> **1. Every recorded accuracy anchor for a `lossguide` arm now describes a
> model we do not ship.** `bench/real_data/accuracy_anchors.json` and the
> `lossguide` rows of `bench/results/COMPARISON_RUN_2026-08-16.md` were measured
> at `lambda_l2 = 0`. They are stale, and **they must not be re-recorded by
> arithmetic.**

There are zero recorded anchors, so the count of stale ones is zero and the file
named holds none of what the sentence attributes to it. The anchors file's own
`_about` already contains the correction and names this exact passage. True if
the sentence is scoped to the run artifacts under `bench/results` and drops
`accuracy_anchors.json` from the list.

**6. `docs/design/ACCURACY_BUDGET.md:2034` to `2036`** (section 14 step 1)
> A candidate must be within the accuracy anchor's **0.25 percent per scenario**
> before it is a candidate at all.

No anchor exists for any scenario, so the veto has no reference value on any
row. True when R3 lands. A pre-registered rule whose first step cannot execute
should say so where it is registered.

**7. `docs/design/ACCURACY_BUDGET.md:2050` to `2052`**
> Either failing the 0.25 percent anchor on any scenario, or passing it on
> `dense_regression` while failing on `imbalanced_binary` or `multiclass`

Two defects rather than one. No anchor exists, and the symmetric arm cannot run
on `multiclass` at head at all (Part 3). True when the anchors exist AND the
`multiclass` branch is either made reachable or removed with its reason.

**8. `bench/real_data/README.md:139`**
> accuracy_anchors.json   the recorded accuracy the gate measures against. Read
> by verify.py, written by nothing

The description is of a file with contents. True if it reads "the recorded
accuracy the gate measures against. EMPTY today, so the gate covers nothing.
Read by verify.py, written by nothing".

**9. `docs/design/ACCURACY_GAP.md:714`**
> a re-anchor of the accuracy gate

Prices a re-anchor as a cost of `GreedyLogSum`, which presumes an anchor to
re-anchor. True once anchors exist; today the cost is an initial anchoring, and
the difference matters because one of them is a run somebody already owes.

**10. `docs/design/ACCURACY_GAP.md:731`**
> Any of these is bit-moving on every fit and must be judged against our own
> accuracy anchor.

An obligation against a reference that does not exist. Same repair as 9.

**11. `docs/design/ACCURACY_GAP.md:994`**
> re-anchored against our own accuracy gate

Same as 9 and 10, and it also conflates the anchor with the gate. Same repair.

**12. `bench/real_data/verify.py:37` to `41`** (module docstring, item 7)
> The accuracy anchor, which IS the accuracy gate, and it is self-anchored: is
> this arm within tolerance of OUR OWN recorded accuracy for the same arm on the
> same scenario, at the same tree count, on the same device.

The list at the top of the file is what a reader treats as the inventory of what
`verify.py` checks. Item 7 describes a check that abstains on every cell.
`load_accuracy_anchors`'s own docstring at 1600 to 1603 is honest about it, so
the file contradicts itself between line 40 and line 1601. True if item 7 adds
that the anchors file is empty and the check WARNs per cell.

**13. `bench/real_data/verify.py:718` to `722`**
> Accuracy on the variant arms is gated by `check_accuracy_anchor` instead,
> which compares each arm against OUR OWN recorded accuracy for that same arm
> and so cannot punish a tree shape for being a different tree shape.

This is the most consequential false claim in the codebase, because it is the
stated justification for deliberately narrowing `check_differential` to the
plain arm. The compensating gate covers nothing, so the narrowing is currently
unbalanced and four subject arms plus every gpu row are ungated on accuracy.
True when anchors exist for the variant arms. Until then the sentence should say
that the variant arms are UNGATED on accuracy today and name that as the cost of
the narrowing.

**14. `bench/results/RESUME_2026-08-17.md:187` to `189`**
> a decision rule in which the anchor's **0.25 percent per scenario is a VETO
> that comes first**, before any speed figure may be quoted. Failing it on
> `imbalanced_binary` or `multiclass` is registered now as disqualifying.

The handoff document tells the next session the veto is armed. True when R3
lands; until then the handoff should carry "and no anchor exists yet, so the
veto's first step cannot execute" as its next action.

**15. `bench/real_data/thresholds.json:55`** (`defaults.accuracy_anchor.rationale`)
> measured against a value recorded in accuracy_anchors.json for the same arm,
> scenario, tree count and device

The policy file's rationale describes a recorded value that is not there. True
when anchors exist. This one is lowest priority, since a policy rationale
describing the mechanism it configures is a defensible reading.

### Two near misses, listed so nobody edits them by mistake

`README.md:272` says the categorical row "lost the accuracy gate" and
`bench/results/COMPARISON_RUN_2026-08-16.md:19`, `202` and `345` say two
accuracy gates failed. Those refer to `check_differential` against LightGBM,
which does gate and did fail at 12.02 percent against a 10 percent limit.
**Accurate, not over-claims.** They are worth one clarifying word each only
because "the accuracy gate" is now the anchor's name in
`verify.py:869` to `871`, so the same phrase means two different checks in two
places.

### Three sites that are already honest, credited so the pattern is visible

- `bench/results/PROFILE_PROTOCOL.md:1503` to `1523`, the section headed
  "Populating the anchors, which has not happened", which states that the file
  ships with an empty object, that every subject arm reads NO ANCHOR, that the
  gate covers nothing, and that "Until step 4 the accuracy gate is unarmed".
  This is the model the other fifteen sites should be edited toward.
- `bench/real_data/accuracy_anchors.json`, the `_about` array, which states its
  own emptiness twice and corrects `ACCURACY_BUDGET.md` section 13 by name.
- `bench/real_data/verify.py:1598` to `1603`, `load_accuracy_anchors`'s
  docstring, which says a missing file "reads as 'nothing is covered yet' rather
  than as 'everything is fine'".

The pattern across all fifteen false sites is one thing rather than fifteen.
**The code and the file that hold the mechanism are honest about being unarmed,
and every document that CITES the mechanism as a constraint on a decision states
it in the present tense.** A citation is where the over-claim lives, so a repair
that only fixes the mechanism's own documentation fixes none of them.

---

## Requests, collected

Nothing below was performed by this lane.

- **R1.** `propose_anchors` must refuse to write `ACCURACY_ANCHORS_PATH`, with a
  `selfcheck` assertion for the refusal. Source change, no run.
- **R2.** `shipped_at_record` must come from the run, or `--propose-anchors`
  must refuse a run whose commit is not HEAD. Source change, no run.
- **R3.** One `pairs`-plan run at HEAD on the quiet box, `shipped.symmetric` on
  the `dense_std`, `dense_large`, `dense_real` and `imbalanced` rows, both
  device legs, 100 trees, then `--propose-anchors` and a human adoption commit.
  This is the minimum that makes section 14's veto runnable on the scenarios it
  can reach. Andrew's budget decision, not a lane's.
- **R4.** Rule on what section 14 means on `multiclass`, given that the
  symmetric arm is declared unreachable there. Free, and it must precede R3.
- **R5.** Apply the fifteen Part 4 edits. Free.
- **R6.** Consider whether `check_coverage`'s run-scope PASS should say that
  named by a check is not the same as gated by one. Free.

## What this lane could not determine

- Whether HEAD's nine changed `src/mojotrees/*.mojo` files move any accuracy
  metric. Only a run can say, and this lane runs nothing.
- The exact `dataset` string a synthetic `imbalanced_binary` or `multiclass`
  record writes. Inferred as `generated:<generator>` from the two synthetic
  datasets observed on disk, not read from the formatting site.
- Whether `verify.py` exits non-zero on WARN. The status assignment was read;
  the exit path was not traced.
- Whether the three Aug 15 `mojoboost` runs could be made usable by any means
  the campaign would accept. Out of scope, and probably moot, since anchors
  adopted from Aug 15 would describe a model many commits old.
