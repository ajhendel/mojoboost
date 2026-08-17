# Figure provenance, the three owed reconciliations

Forensic lane, 2026-08-17, read-only. Settles the three items
`RESUME_2026-08-17.md` records under "MY OWN NUMBERS ARE INCONSISTENT AND I OWE
A RECONCILIATION". Nothing was built, compiled, run or timed to produce this
file. Every finding below is either read off a file at head, read off a commit
message, or labeled as an inference.

**The headline, before the detail.** There is no arithmetic error anywhere in
the tree. Every ratio is correct for the pair of times it was computed from.
What is missing from almost every site is WHICH PAIR, and on a machine that
`PROFILE_PROTOCOL.md` A3 records as drifting two- to threefold between windows,
that omission is what turns two correct figures into an apparent contradiction.

Four different absolute baselines are in circulation for one shape, 799,110 rows
x 100 features x 100 trees, symmetric depth 6, on one M4.

| baseline | where it comes from | provenance |
| --- | --- | --- |
| 17.07 s | the 2026-08-16 comparison run, `mojotrees_catboost_mode` gpu row, `docs/design/GPU_GROWTH_ATTRIBUTION.md` table (17.0719) | artifact-backed, pre-flip |
| 20.4 s | 2026-08-17, quiet box, alternating passes of three repeats, narrow-scan arm (`gpu_split_search.oblivious_wide_scan_requested` flip comment, "20.40 s narrow") | source comment only |
| 21.97 s | 2026-08-17, conditions recorded nowhere | out-of-band brief only |
| 22.76 s | 2026-08-17, loaded box, three-cycle interleaved round robin | source comments only |

Every ratio in dispute is arithmetically exact against one of those four.

    21.97 / 12.34 = 1.7804   ->  quoted as 1.78x
    22.76 / 14.39 = 1.5817   ->  quoted as 1.58x
    22.76 / 18.06 = 1.2603   ->  quoted as 1.26x
    22.76 / 21.28 = 1.0695   ->  quoted as 1.07x
    22.76 / 10.36 = 2.1969   ->  quoted as 2.20x
    20.4  /  9.8  = 2.0816   ->  quoted as 2.08x
    17.07 / 10.36 = 1.6481   ->  quoted as 1.65x in DECLINED_OPTIMIZATIONS

So the defect is not a wrong number. It is a number without its denominator,
repeated across twenty-five sites.

---

## PART A. Every citation site

Anchored by `file.function` or by section per rule 7's CITE BY NAME clause, with
line numbers as a convenience only. Searched exhaustively across `src/`,
`docs/`, `bench/`, `tests/`, `bindings/`, `README.md`, and `git log --all` by
both `--grep` and `-S` on 1.78, 1.58, 2.08, 2.20, 21.97, 12.34, 22.76, 14.39,
10.36, 18.06, 20.40, 19.49, 21.28, `NOISE_HOIST`, `noise hoist` and
`oblivious_schedule_launches`.

### A1. Sites carrying 1.78x, 21.97 s or 12.34 s

| site | figure claimed | evidence cited |
| --- | --- | --- |
| `docs/design/SWITCH_GRID.md` section 3A, `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` row (:157) | **MEASURED, 1.78x, 21.97 s to 12.34 s at 799,110 x 100 x 100 trees, rmse identical to nine decimals** | "(2026-08-17 lane brief)" |
| `docs/design/SWITCH_GRID.md` section 4 rank 1 (:348) | "1.78x measured on the shipped symmetric default ... Largest single number on the board" | none |
| `docs/design/SWITCH_GRID.md` section 8 item 2 (:630) | "the 1.78x and the 4.5 percent are cited to the lane brief and not to an artifact" | admits the gap |
| `docs/design/OBLIVIOUS_WAIT_CENSUS.md`, "How this composes with sibling subtraction" (:285) | "1.78x, 21.97 s to 12.34 s, with a second interleaved three-cycle reading of 22.76 s to 14.39 s" | none |
| `docs/design/GROWTH_POLICY_REACH.md` switch table (:62) | "sibling subtraction, **default ON, measured 1.78x**" | none |
| `bench/results/LANE_RULES.md` rule 5 (:94) | "an exact-integer sibling subtraction worth 1.78x" | none |
| `src/mojotrees/histogram_gpu.mojo`, `GpuHistogramBuilder.enqueue_desc_level_children` docstring (:3109) | "was priced at 1.78x and then at 22.76 s to 14.39 s over three interleaved round-robin cycles" | flip comment plus SWITCH_GRID |
| `src/mojotrees/gpu_leaf_batching.mojo`, `oblivious_subtract_requested` docstring (:480) | "priced twice since (1.78x, then 22.76 s to 14.39 s interleaved)" | the flip comment below it |

Note the shape of the last two. Both name the two readings and attach the word
"interleaved" only to the second, which is the closest the tree comes to saying
that the 1.78x reading was not interleaved.

### A2. Sites carrying 1.58x

| site | figure claimed | evidence cited |
| --- | --- | --- |
| commit `abbbf98`, message body, 2026-08-17 11:53:25 -0400 | "symmetric GPU train 20.4s -> 9.8s (sibling subtraction **1.58x**, skip-last-level-build 1.26x, wide oblivious scan 1.07x)" | "All timings are standard tier on one M4" |
| `bench/results/RESUME_2026-08-17.md` (:135, :212) | poses 1.58x as the rival figure | the commit above |

**1.58x is asserted nowhere in `src/` and nowhere in `docs/`.** It exists in one
immutable commit message and in the review note that flagged it. That asymmetry
matters for the fix, because a commit message cannot be edited.

### A3. Sites carrying 2.08x

| site | figure claimed | evidence cited |
| --- | --- | --- |
| commit `abbbf98`, message body | 20.4s -> 9.8s, which is the arithmetic source of 2.08x although the message never writes the ratio | "standard tier on one M4" |
| commit `fe1f98d`, message body | "1.26x alone, **2.08x** with the subtraction and wide-scan arms beside it, **interleaved round-robin**, bit-identical to nine decimals" | none |
| `src/mojotrees/gpu_resident_round.mojo`, `OBLIVIOUS_SKIP_LAST_BUILD_VAR` docstring, MEASURED paragraph (:1015) | "**1.26x alone**, and 2.08x in combination with the sibling-subtraction and wide-scan arms" plus "interleaved round-robin against a baseline arm" | none |
| same docstring, "WHAT IT REMOVES" close (:1051) | "**It has been timed, on 2026-08-17: 1.26x alone and 2.08x with the two switches beside it, interleaved, bit-identical.**" | none |
| `src/mojotrees/gpu_resident_round.mojo`, `oblivious_skip_last_build_requested` docstring (:1129 to :1130) | records the conflict itself, "2.20x ... which is 22.76 s to 10.36 s; `OBLIVIOUS_SKIP_LAST_BUILD_VAR` says **2.08x** in combination with the same two arms" | says only a run settles it |
| `docs/design/OBLIVIOUS_WAIT_CENSUS.md` (:264) | "part of a 2.08x-to-2.20x combined arm" | quotes it as a range |
| `docs/design/SWITCH_GRID.md` section 3A, `SKIP_LAST_BUILD` row (:159) | "part of the 2.08x-to-2.20x combined arm" | range, and misattributes the 2.08x to "`oblivious_schedule_launches`'s neighbouring note" |
| `bench/results/RESUME_2026-08-17.md` (:27) | "`abbbf98` both sessions' work, 77 files: the symmetric 2.08x" | the commit |

One pointer correction while we are here. The review says the 2.08x sits in "a
note beside `oblivious_schedule_launches`". It does not.
`oblivious_schedule_launches` is defined at `gpu_resident_round.mojo:3163` and
carries no timing at all. The 2.08x lives in the `OBLIVIOUS_SKIP_LAST_BUILD_VAR`
docstring, which merely MENTIONS `oblivious_schedule_launches` at :1026. The
figure and the pointer to it drifted apart exactly as rule 7's CITE BY NAME
clause predicts.

### A4. Sites carrying 2.20x, 22.76 s, 14.39 s, 10.36 s or 18.06 s

These are the internally consistent family. Listed because each one still needs
its pair attached, not because any is wrong.

| site | figure claimed |
| --- | --- |
| `src/mojotrees/gpu_leaf_batching.mojo`, `oblivious_subtract_requested` return comment (:522 to :524) | "22.76 s to 14.39 s alone, and 10.36 s combined with the skip-last-build and wide-scan arms, which is 2.20x", three round-robin cycles, rmse 2.439382420 |
| `src/mojotrees/gpu_resident_round.mojo`, `oblivious_skip_last_build_requested` docstring and return comment (:1122, :1141 to :1142) | "22.76 s to 18.06 s alone, 1.26x, and part of the 2.20x combined arm" |
| `src/mojotrees/gpu_resident_round.mojo`, `grow_tree_device_oblivious` level loop comment (:3809) | "at 2.20x combined with the wide scan in the same round-robin" |
| `src/mojotrees/gpu_split_search.mojo`, `oblivious_wide_scan_requested` return comment (:4418 to :4421) | "20.40 s narrow against 19.49 s wide with the ranges fully disjoint, a resolved 4.5 percent. Later, on a loaded box in a three-cycle round robin, 22.76 s against 21.28 s, 1.07x, and part of the 2.20x combined arm" |
| `docs/design/SWITCH_GRID.md` section 3A rows (:157, :158, :159), section 4 ranks 1, 2 and 4 (:348, :349, :351), section 7 interaction A (:511 to :512) | the same family |
| `docs/design/OBLIVIOUS_WAIT_CENSUS.md` (:262, :286 to :287, :307) | the same family |
| `docs/design/DECLINED_OPTIMIZATIONS.md` (:284, :344 to :345) | 1.26x and the two wide-scan readings |
| `docs/design/DECLINED_OPTIMIZATIONS.md` denominator note (:68, :74) | "the symmetric arm is around **10.36 s** rather than 17.07 s (2.20x from ...)", then correctly converts at 1.65x |

`gpu_split_search.oblivious_wide_scan_requested` is the model for how all of
this should read. It gives both windows, labels one quiet and one loaded, and
refuses to collapse them. It is the only site in the tree that already satisfies
PROFILE_PROTOCOL A3.

`DECLINED_OPTIMIZATIONS.md:68` deserves a note of its own. It attaches the label
"2.20x" to the transition 17.07 s to 10.36 s, which is 1.65x, and then six lines
later correctly tells the reader to convert at 1.65. That is not an arithmetic
slip. It is the 2.20x figure, which is right against its own 22.76 baseline,
sitting next to a different baseline with no seam marked. It is the single
clearest illustration of the whole defect.

### A5. Sites carrying the noise-hoist null

Ten assertion sites, enumerated in Part C.

### A6. What is NOT in the tree

Verified absences, each one a negative result rather than a failure to look.

- **No `bench/results/` artifact and no `bench/real_data/results/` record
  contains 21.97, 22.76, 12.34, 14.39, 10.36, 18.06, 20.40, 19.49 or 21.28 as a
  training time.** All 663 rows of every `records.csv` under
  `bench/real_data/results/` were read and matched against those nine values at
  a 0.06 s tolerance. Two coincidental hits landed, both at the wrong shape and
  the wrong engine (`20260815T014842Z`, a 368,759 x 30 CPU ranking arm at
  12.3705 s, and `20260816T181134Z-decision-1m`, a CPU arm at 14.4268 s). The
  handful of hits inside `bench/results/*/*.txt` are all from the 2026-08-16
  thread-scaling and serial-kernel sweeps and none of them is an oblivious arm.
- **No commit message anywhere in `--all` mentions 21.97, 12.34, 22.76, 14.39 or
  10.36.** `git log --all --grep` on each returns nothing. The strings enter the
  tree only as file content, in `abbbf98` (SWITCH_GRID) and `9a113c8`.
- **The "2026-08-17 lane brief" is not a file in this repository.** No path
  matches `*brief*`, and the eight documents that cite it all cite it bare, with
  no path. It is an out-of-band orchestrator message. `SWITCH_GRID.md` section 8
  item 2 already says so in as many words.
- **`bench/bench_train_gpu.mojo` has a file sink and no run used it.**
  `MOJOTREES_BENCH_JSON=<path>` writes the `json_summary` record, which already
  carries `arm`, `baseline`, `speedup_x`, `delta_pct`, `noise_floor_pct`,
  `verdict`, `n_rows`, `n_features`, `n_trees`, `objective`, `seed`, `repeats`
  and the per-arm sample lists. The instrument for filing every one of these
  figures existed on the day they were taken.

---

## PART B. Items 1 and 2, settled from stored evidence

### Item 1. Sibling subtraction alone

**Status: DISTINCT WINDOWS at one identical shape, and the 1.78x reading is
UNRESOLVED as to provenance.**

Not distinct shapes. Both figures are stated at 799,110 x 100 x 100 trees,
symmetric depth 6, M4. `SWITCH_GRID.md:157` names that shape for the 1.78x
reading and `OBLIVIOUS_WAIT_CENSUS.md:285` names it for both. Nothing in the
tree offers a second shape for either.

What differs is the window and the method.

| | reading A | reading B |
| --- | --- | --- |
| baseline | 21.97 s | 22.76 s |
| subtracting arm | 12.34 s | 14.39 s |
| ratio | 1.78x | 1.58x |
| shape | 799,110 x 100 x 100 trees, sym d6, M4 | identical |
| method | **unrecorded** | three interleaved round-robin cycles |
| box regime | **unrecorded** | loaded (per `gpu_split_search` flip comment, same round robin) |
| accuracy | "rmse identical to nine decimals" | rmse 2.439382420 in every arm of every cycle |
| filed under `bench/results/` | no | no |
| citation | "2026-08-17 lane brief" | source flip comments |

**Verified.** Reading B's conditions line is complete and appears identically in
three independent places (`gpu_leaf_batching.oblivious_subtract_requested`
return comment, `histogram_gpu.enqueue_desc_level_children` docstring,
`OBLIVIOUS_WAIT_CENSUS.md`), which is a strong internal cross-check even without
an artifact. Reading A's conditions line does not exist anywhere. Its only
attribution is a brief that is not a file.

**Inferred, and flagged as inference.** Reading A is most likely the same
single-arm on/off A/B taken in an earlier and quieter window that day. Two
alternatives were tested arithmetically and both fail. If 12.34 s were secretly
a combined arm, then 22.76 / 1.58 / 1.26 = 11.4 s for subtract plus skip-last,
and 14.39 / 1.07 = 13.45 s for subtract plus wide. Neither is 12.34. So the
arithmetic does not identify 12.34 s as anything other than a subtraction-only
arm at a lower baseline. This cannot be raised above an inference from the tree.

**The resolution, and it is not "pick the bigger number".**
`PROFILE_PROTOCOL.md` A3 is dispositive and already registered. It says a result
carries its regime label, that where the effect size differs by regime you
**report both numbers and refuse to pick one**, and that anything called
indistinguishable outside a quiet window is retaken. A3 is not symmetric between
these two readings, because one of them has a regime label and the other has
none. A figure with no window and no interleaving statement cannot be preferred
over one with both, on a machine whose own protocol file records that the same
command produced opposite verdicts in two windows.

So the citable figure today is **1.58x, 22.76 s to 14.39 s, three interleaved
round-robin cycles, loaded box, 799,110 x 100 x 100 trees, symmetric depth 6,
M4, rmse 2.439382420 in every arm of every cycle.** That is also, and this
matters, exactly what commit `abbbf98` told the user. The commit message was
right and eight documents wandered off it.

**1.78x should not be deleted.** It should be demoted to what it is, an earlier
unlabeled reading, and reported beside 1.58x as A3 requires, with the range
stated as **1.58x to 1.78x** wherever one number is wanted. Every site that
prints 1.78x alone is printing the loose end of a range as though it were the
result, and one of those sites is `LANE_RULES.md` rule 5 itself.

**The single run that closes it.** One interleaved on/off pair,
`MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` 1 against 0, at 799,110 x 100, 100 trees,
`grow_policy` symmetric, `max_depth` 6, with `SKIP_LAST_BUILD` and
`OBLIVIOUS_WIDE` both held at one setting and named in the record, at least five
pairs per M0 as amended by A2, under the quiet-box precondition and the canary
on, with `MOJOTREES_BENCH_JSON` pointed at
`bench/results/oblivious_subtract_<run-id>/summary.json`. That is one command
and it retires eight citation sites.

### Item 2. The combined arm

**Status: SETTLED. Two different pairs, both ratios correct, and the real defect
is a borrowed conditions line.**

    2.08x  =  20.4 s -> 9.8 s
    2.20x  = 22.76 s -> 10.36 s

**Verified.** Commit `abbbf98`'s message body contains the 20.4 s to 9.8 s pair
in the plain text, "symmetric GPU train 20.4s -> 9.8s", and 20.4 / 9.8 = 2.0816.
That is the origin of 2.08x, and it is arithmetically exact. The 22.76 s to
10.36 s pair is in three source flip comments and three documents, and
22.76 / 10.36 = 2.1969, which is 2.20x, also exact. **Neither figure is wrong.
They are measurements of the same arm set in two different windows.** The
question as posed, "2.08x or 2.20x", has no answer because it presupposes one
pair.

**Inferred, and flagged as inference.** The 20.4 s baseline is very probably the
quiet-box window, because `gpu_split_search.oblivious_wide_scan_requested`
records "20.40 s narrow" for that day's quiet box with disjoint ranges, and a
narrow-scan arm with the other switches off is the all-off baseline. That makes
2.08x the quiet-box combined figure and 2.20x the loaded-round-robin combined
figure, which is the same regime split A3 predicts. This is circumstantial. The
commit message itself says only "standard tier on one M4" and names no window,
and 9.8 s appears in no other file in the repository.

**The actual defect, which is worse than a disagreement.** Commit `fe1f98d` and
`gpu_resident_round.OBLIVIOUS_SKIP_LAST_BUILD_VAR` both attach the words
"interleaved round-robin" to the 2.08x figure. The interleaved round robin is
the OTHER pair, the one that produced 22.76 s to 10.36 s and 2.20x. So a correct
number is carrying another run's conditions line. That is exactly the failure
`bench_train_gpu._arm_conditions`'s own docstring already names, "a pair of
figures whose conditions were written down only in a commit message", and it is
the reason this looked like a contradiction rather than a pair of windows.

**The fix is editorial, not a run.** No measurement is owed for item 2. Every
site that quotes 2.08x must name the 20.4 s to 9.8 s pair beside it and must
stop calling it interleaved unless somebody can produce that run's conditions.
Every site that quotes 2.20x must name the 22.76 s to 10.36 s pair. The two
documents that already quote it as the range "2.08x to 2.20x"
(`OBLIVIOUS_WAIT_CENSUS.md:264` and `SWITCH_GRID.md:159`) are the closest to
correct and need only the two pairs written in.

`oblivious_skip_last_build_requested`'s docstring says "only a run settles which
is meant". That was the right instinct and is not quite right. A run would give
a third window. What settles it is naming the two pairs, which the commit
message already contains.

---

## PART C. The noise hoist. The claim is VERIFIED, no artifact exists

**Status: UNRESOLVED, never filed, and under the project's own registered rule
it may not close the item.**

### What was checked, and what came back

- `grep -rn NOISE_HOIST` across the whole repository, excluding `.git`, returns
  hits in `src/mojotrees/gpu_resident_round.mojo`, four files under
  `docs/design/`, and `bench/results/RESUME_2026-08-17.md`. Under `bench/` the
  ONLY hit is the review note that raises the complaint. The review's phrasing,
  "`grep -rl NOISE_HOIST bench/` returns nothing", is a hair off at head because
  the complaint file now matches its own grep. Substantively it is correct.
  **No results artifact names the variable.**
- Zero occurrences of `NOISE_HOIST` or `noise_hoist` anywhere under
  `bench/real_data/results/`, across every `records.json`, `records.csv`,
  `manifest.json` and per-record file in all thirty-four 2026-08-17 run
  directories.
- `git log --all --grep` on `NOISE_HOIST` and on `noise hoist` returns nothing.
  No commit message in the repository has ever mentioned the arm.
- `git log --all -S` finds the token entering only through source and doc edits.
  No results file was ever added carrying it.
- The cited authority, "the 2026-08-17 lane brief", is not a file. See A6.
- `SWITCH_GRID.md` section 8 item 2 already concedes the general case, that a
  grep for these variables "across `bench/results/` and `docs/` returns nothing,
  so the 1.78x and the 4.5 percent are cited to the lane brief and not to an
  artifact".

**So the null rests entirely on one out-of-band brief, and ten sites now state
it as established.**

### The stronger objection, which does not depend on the missing file

Even if the run happened exactly as described, **`PROFILE_PROTOCOL.md` A3
forbids using it to close the item as the record currently stands.** A3 says, in
the registered text, that "a null taken in a slow or dirty window **is not
evidence of absence** and may not close a question. It may only defer it.
Anything called indistinguishable outside a quiet window is retaken before it is
used to cancel a lane."

The noise hoist's window is unrecorded. Its three sibling arms from the same day
are explicitly recorded on a **loaded box** in a three-cycle round robin
(`gpu_split_search.oblivious_wide_scan_requested`, "Later, on a loaded box").
So the nearest evidence about that day's regime points the wrong way for a null.

The consequence is precise. Three arms that day resolved FASTER and a resolved
result survives a dirty window, only its effect size moves. The fourth arm
returned indistinguishable, and that is the one verdict A3 says a dirty window
can manufacture. **The one result of the four that most needed a quiet-window
label is the one that has no label at all.**

That also disposes of the "measured null, re-measuring is optional rather than
owed" framing that four sites now carry. Under A3 it is owed.

### The ten assertion sites, and the exact replacement text

Three were asked for. There are ten. Each replacement below is a CORRECTION
under rule 7, not a suggestion.

**C1. `src/mojotrees/gpu_resident_round.mojo`, `OBLIVIOUS_NOISE_HOIST_VAR`
docstring, header paragraph.**

Currently asserts "as of 2026-08-17 it is off on a MEASURED NULL rather than on
an absent measurement. It was run that day ... and came in **indistinguishable**
in the registered M0 sense ... The standing reading is a null, not a withdrawal
and not a pending item ... re-measuring it is optional rather than owed ... The
result is not yet filed under `bench/results/`."

Replace with:

> **Off by default and an equality against "1", and the reason is UNRESOLVED
> rather than settled either way.** A 2026-08-17 lane brief reported the arm run
> that day on a fit that sets `random_strength > 0` and coming in
> indistinguishable under M0, and that report has no artifact. Nothing under
> `bench/results/` or `bench/real_data/results/` names this variable, no commit
> message mentions it, and the brief itself is not a file in this repository. The
> window is unrecorded, and `PROFILE_PROTOCOL.md` A3 says a null taken outside a
> quiet window may only DEFER a question and never close it, so even taken at
> face value that reading does not close this item. The three sibling arms of the
> same day are recorded on a loaded box, which is the regime A3 warns manufactures
> exactly this verdict. This header previously said "because nothing has measured
> it", then said the result was a measured null and re-measuring was optional. The
> standing reading is that the arm is UNMEASURED OF RECORD and the measurement is
> OWED, which under rule 4 makes it an open item rather than a closed one. What
> closes it is one interleaved A/B on a quiet box with the canary on, filed to
> `bench/results/` with `MOJOTREES_BENCH_JSON`.

**C2. Same docstring, "WHAT IT REMOVES" paragraph.**

Currently: "The count is read from the source, and the **time** was taken on
2026-08-17 and came in indistinguishable, so the count above is real and does
not convert into a clock at this shape."

Replace with:

> The count is read from the source and is verified. The **time** is not filed.
> A 2026-08-17 brief reported it indistinguishable, with no artifact and no
> window label, so whether the count converts into a clock at this shape is OPEN.
> Until that day this paragraph ended "the time is unmeasured, which is why this
> is off", which was accurate about the state of the evidence and is accurate
> again.

**C3. `src/mojotrees/gpu_resident_round.mojo`,
`oblivious_noise_hoist_requested` docstring.**

Currently: "off on a measured null rather than on an absent measurement. See
`OBLIVIOUS_NOISE_HOIST_VAR`, whose header carries the 2026-08-17 result."

Replace with:

> **Off unless `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST=1` asks for it**, and off on
> an UNFILED measurement. See `OBLIVIOUS_NOISE_HOIST_VAR`, whose header carries
> what is and is not known about the 2026-08-17 reading.

**C4. `src/mojotrees/gpu_resident_round.mojo`, `grow_tree_device_oblivious`
docstring, the noise paragraph near the four-combination discussion.**

Currently: "That one is off by default, and unlike the arm above it stayed off
on evidence rather than for want of it."

Replace with:

> That one is off by default, and unlike the arm above it has no filed
> measurement either way. See `OBLIVIOUS_NOISE_HOIST_VAR`.

While in that docstring, note a live self-contradiction thirty-four lines
earlier in the same block, which says the hoist "is off until something times
it". At head the two paragraphs of one docstring disagree about whether the arm
has been timed. Under rule 5a clause 3 that is an incomplete sweep. The
replacement above makes both paragraphs true simultaneously.

**C5. `docs/design/SWITCH_GRID.md` section 2, the flip banner.**

Currently: "The fifth arm of that group, `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST`,
did NOT flip, and the reason is that it was measured and came in
indistinguishable ... A measured null is not an unmeasured switch and its row
now says which it is."

Replace with:

> The fifth arm of that group, `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST`, did NOT
> flip, and the reason is that no measurement of it is filed. A 2026-08-17 brief
> reported it indistinguishable; nothing under `bench/results/` names the
> variable and the window is unrecorded, so under `PROFILE_PROTOCOL.md` A3 that
> reading defers the question rather than closing it. Its predicate is still
> `== "1"` and its default is still off, correctly, because a switch does not flip
> on an unfiled result. Its row says so.

**C6. `docs/design/SWITCH_GRID.md` section 3A,
`MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` row, MEASURED cell.**

Currently: "**MEASURED, AND THE RESULT IS A NULL.** It was run on 2026-08-17 and
came in **indistinguishable** ... This cell previously read "UNRESOLVED,
measured slower in a run confounded by drift, so neither result stands"; the
standing reading is a measured null rather than a withdrawn one ... Not yet filed
under `bench/results/`, and the docstring at
`gpu_resident_round.OBLIVIOUS_NOISE_HOIST_VAR` still says the time is
unmeasured, which is stale". Verdict cell currently: "**CORRECTLY OFF on a
measured null**".

Replace the MEASURED cell with:

> **UNRESOLVED, AND NOT FILED.** A 2026-08-17 lane brief reported the arm run on
> a fit that sets `random_strength > 0` and coming in indistinguishable under M0.
> No artifact backs it. `grep -r NOISE_HOIST bench/` finds no results file, no
> commit message mentions the variable, and the brief is not a file in this
> repository. The window is unrecorded, and A3 says an indistinguishable result
> outside a quiet window may only defer a question. An earlier version of this
> cell read "UNRESOLVED, measured slower in a run confounded by drift"; a middle
> version read "MEASURED, and the result is a null"; the standing reading is back
> to UNRESOLVED, on the narrower and better ground that nothing was filed. The
> count behind it is still read from source and is unaffected, six `enqueue_copy`
> drains per depth-6 tree, each between a level's child build and the next
> level's search.

Replace the verdict cell with:

> **CORRECTLY OFF, on an ABSENT filed measurement.** Off is the right state for
> an unmeasured switch under rule 2. Re-measuring is OWED rather than optional,
> on a quiet box, on a fit that sets `random_strength > 0`, filed to
> `bench/results/`.

Delete the trailing clause about the docstring still saying the time is
unmeasured. That claim is itself stale at head. See the note below.

**C7. `docs/design/SWITCH_GRID.md` section 4, rank 5.**

Currently: "**CLOSED ON A NULL, STILL OFF** ... An M0-indistinguishable result
is a null and not a pending item, so this is the one arm of the five that did not
flip and did not need to. Re-measuring is optional."

Replace with:

> **OPEN, STILL OFF, AND THE ONE ARM OF THE FIVE WHOSE MEASUREMENT IS OWED** ...
> A brief reported it indistinguishable and nothing was filed, and an
> indistinguishable result whose window is unrecorded is a deferral under A3 and
> not a null. Re-measuring is owed. A fit at `random_strength = 0` still measures
> nothing, which is how this became invisible in the first place.

**C8. `docs/design/SWITCH_GRID.md` section 8, item 6, closing sentence.**

Currently: "So rank 5 is reachable, it was measured, and it came in
indistinguishable. Nothing here is owed to another lane any longer."

Replace with:

> So rank 5 is reachable. Reachability is what this item was about and it is
> closed. Whether the hoist buys anything is a separate and still open question,
> because no measurement of it is filed. Nothing about REACH is owed to another
> lane any longer.

**C9. `docs/design/OBLIVIOUS_WAIT_CENSUS.md`, the noise-drain section.**

Currently: "**Off by default, and as of 2026-08-17 that is a MEASURED NULL
rather than an absent measurement** ... The arm was run that day ... and it came
in **indistinguishable** ... It is not filed under `bench/results/`, so the
citation is the 2026-08-17 lane brief."

Replace the opening sentence with:

> **Off by default, and as of 2026-08-17 that is an UNFILED measurement rather
> than a settled one.** This paragraph read "Off by default because the time is
> unmeasured", then read "MEASURED NULL". A lane brief that day reported the arm
> indistinguishable on a fit that sets `random_strength > 0`, and nothing under
> `bench/results/` names the variable, so the citation is a brief that is not a
> file. Its window is unrecorded and A3 will not let an unlabeled
> indistinguishable close a question.

The following paragraph, which separates the null from the count, is correct as
written and should be kept. Change only "What the null says" to "What that
reading would say, if filed".

**C10. `docs/design/DECLINED_OPTIMIZATIONS.md`, register entry A3, the Class
line.**

Currently: "**Class.** ASSERTED as to time when this row was written; **MEASURED
as of 2026-08-17, and the result is a null.** ... which is why it is the one of
the four symmetric arms whose default did not flip. The quoted docstring at
`OBLIVIOUS_NOISE_HOIST_VAR` still says the time is unmeasured and is stale on
that point."

Replace with:

> **Class.** ASSERTED as to time, and still ASSERTED. A 2026-08-17 lane brief
> reported the arm run on a fit that sets `random_strength > 0` and coming in
> indistinguishable under M0, with no artifact filed and no window recorded, so
> under rule 6 this stays an ASSERTED decline and therefore an OPEN item under
> rule 4. The count, six drains per depth-6 tree, is verified from source and is
> untouched either way.

Its "**Survives.** Partly. The decline is honest about being unmeasured" line is
correct as written and should be kept. It is, at head, the most accurate
statement about this arm anywhere in the tree.

### A stale meta-claim, found while doing this

Three documents assert that the `OBLIVIOUS_NOISE_HOIST_VAR` docstring "still
says the time is unmeasured, which is stale". At head the docstring says the
opposite twice, in its header and in WHAT IT REMOVES. So those three sentences
are themselves stale claims about a stale claim.

- `docs/design/SWITCH_GRID.md:160`
- `docs/design/OBLIVIOUS_WAIT_CENSUS.md:217`
- `docs/design/DECLINED_OPTIMIZATIONS.md:309`

Applying C1 and C2 makes all three wrong in the other direction, so each must be
deleted in the same commit rather than left. This is rule 7's VERIFY BEFORE YOU
CORRECT clause landing on the corrections themselves.

Separately, `docs/design/SWITCH_CONTRACT_REPAIRS.md` edit 9 cites the
`OBLIVIOUS_NOISE_HOIST_VAR` docstring at "lines 1029 to 1031". At head the
docstring begins at 1158. Line-number rot, exactly as rule 7 describes.

---

## PART D. Draft rule 10 for `LANE_RULES.md`

Not applied. Quoted here for the orchestrator to paste. Voice matched to rules 5
through 9.

---

**10. A SPEED FIGURE TRAVELS WITH ITS RUN, OR IT DOES NOT TRAVEL.** Added
2026-08-17, from Andrew's advisor, after three lanes independently flagged the
same three figures and a forensic pass found that **not one of them was wrong**.
That is what makes this rule necessary rather than pedantic. If the numbers had
been miscopied, the fix would be arithmetic. They were all correct, and they
still could not be reconciled, because a ratio had been separated from the pair
of times it was computed from.

Here is what the day actually produced. Four absolute baselines for ONE shape,
799,110 x 100 x 100 trees on one M4: **17.07 s** from the 2026-08-16 comparison
run, **20.4 s** from a quiet box, **21.97 s** from a window nobody wrote down,
and **22.76 s** from a loaded three-cycle round robin. Every published ratio was
exact against one of those four and no site said which. So 1.78x and 1.58x for
the same arm, 2.08x and 2.20x for the same combination, and a document that
labeled a 1.65x transition "2.20x" six lines above correctly telling the reader
to convert at 1.65. Twenty-five sites, zero arithmetic errors, nothing
checkable. On a machine that A3 records as drifting two- to threefold between
windows, **a ratio without its baseline is not a weak result, it is not a result
at all.**

So:

1. **The quotable unit is a RUN, not a ratio.** A speed figure may be published
   only with a **run id** that resolves to a directory under `bench/results/`, a
   **shape** (rows x features x trees, growth policy, max depth), an **arm set**
   with every switch that was set resolved to its value, **both absolute times**,
   the **M0 verdict**, and the **regime label** A3 requires. Six fields. A figure
   missing any of them is ASSERTED under rule 6 and is an open item under rule 4,
   however carefully it was measured.

2. **Stop transcribing. The generator emits the table and the docs cite the
   run.** `bench/bench_train_gpu.mojo` already prints a `json_summary` record
   carrying `arm`, `baseline`, `speedup_x`, `delta_pct`, `noise_floor_pct`,
   `verdict`, `n_rows`, `n_features`, `n_trees`, `objective`, `seed`, `repeats`
   and every arm's samples in the order they ran, and
   `MOJOTREES_BENCH_JSON=<path>` files it. **Every timing run sets that variable
   before it starts**, to a path under `bench/results/<run-id>/`. A prose site
   then cites the run id and quotes at most one number from it. The four figures
   this rule was written about were all taken with that sink available and none
   of them used it, which is the entire distance between a reconciliation lane
   and a `grep`.

3. **A chat brief is not an artifact.** Eight documents cite "the 2026-08-17 lane
   brief" and there is no such file in this repository. A brief is how a lane
   receives work, not how the tree records a result. **If the only witness to a
   number is a conversation, the number is unfiled.** Cite the path or downgrade
   the claim.

4. **A commit message is write-once, so it is the last place a figure may go and
   never the first.** `abbbf98` published 1.58x and 20.4 s to 9.8 s; the docs
   published 1.78x and 22.76 s to 10.36 s; both are correct and the commit can
   never be edited to say so. Quote a figure in a message only after it exists
   under `bench/results/`, and quote the run id beside it so the message stays
   checkable after the prose moves on. `bench_train_gpu._arm_conditions` already
   records that this project has discarded "a pair of figures whose conditions
   were written down only in a commit message". It has now done it twice.

5. **Copy the conditions line, not the number.** The worst site found was not a
   wrong figure. It was a right figure wearing another run's clothes: 2.08x,
   which came from a quiet-box 20.4 s to 9.8 s pair, described in two places as
   "interleaved round-robin", which is the OTHER pair. A number with a borrowed
   provenance passes every inspection and cannot be checked by anyone, which is
   the same pathology rule 5a names when it says a flip correct in the code and
   wrong everywhere a reader looks is worse than being wrong in both.

6. **Until a run resolves it, publish the RANGE and say why.** A3 already says to
   report both numbers and refuse to pick one where the effect size differs by
   regime. Rule 10 adds the enforcement: **"1.58x to 1.78x, two windows, one
   unlabeled" is publishable and "1.78x" is not.** Picking the flattering end of
   an unresolved range is not optimism, it is an unfiled claim with a decimal
   point on it.

The general form, worth carrying past benchmarks. **A number is a claim about a
procedure, and a claim whose procedure is not recorded cannot be wrong, which is
exactly why it cannot be trusted.**

---

## Addendum, written during this pass

Between this lane's first grep and its last, a peer lane edited
`gpu_leaf_batching.mojo`, `gpu_split_search.mojo` and `histogram_gpu.mojo` in
the shared checkout and applied the interim measure Part B recommends. Three
sites now quote ranges instead of single figures.

- `gpu_leaf_batching.oblivious_subtract_requested`, docstring and return comment.
  "**Quote this arm's speedup as a RANGE of 1.58x to 1.78x**", both pairs named,
  and "QUOTE THE COMBINED ARM AS A RANGE, 2.08x TO 2.20x, UNTIL A LANE
  RECONCILES IT".
- `gpu_split_search.oblivious_wide_scan_requested` return comment. "part of a
  combined arm whose multiple is recorded two ways, 2.08x and 2.20x, and is
  quoted as that RANGE here".
- `histogram_gpu.enqueue_desc_level_children` docstring. "priced twice, once at
  21.97 s to 12.34 s and once at 22.76 s to 14.39 s ... **Quote the speedup as a
  RANGE of 1.58x to 1.78x.**"

That is the right interim state and it reduces the correction count by three.
It does not supersede this file, for two reasons. **A range is honest and it is
not a resolution.** The ranges still do not carry the windows, so a reader
cannot tell that one end of the 1.58x-to-1.78x range is unlabeled and the other
is a three-cycle interleaved round robin, which is the fact A3 turns into a
preference. And the 2.08x-to-2.20x range is presented as an open question when
Part B settles it: those are two different pairs, 20.4 to 9.8 and 22.76 to
10.36, both correct, and no run is owed. **Item 2 does not need a
reconciliation. It needs its two pairs written down.**

`gpu_resident_round.mojo` was not touched and its three 2.08x sites, plus all
ten noise-hoist sites, stand exactly as tabulated above.

## Summary of status

| figure | status | resolution |
| --- | --- | --- |
| 1.78x, 21.97 s to 12.34 s | **DISTINCT WINDOWS**, same shape; provenance **UNRESOLVED** | Same 799,110 x 100 x 100 trees as the other reading. Window and interleaving unrecorded, cited only to a brief that is not a file. Publish the range 1.58x to 1.78x, or retake one interleaved on/off pair on a quiet box and file it. |
| 1.58x, 22.76 s to 14.39 s | **SETTLED**, and it is the citable figure | Three interleaved round-robin cycles, loaded box, rmse 2.439382420 in every arm of every cycle, cross-checked in three independent source sites. Also what `abbbf98` told the user. |
| 2.08x | **SETTLED as a distinct pair** | 20.4 s to 9.8 s, recorded in `abbbf98`'s message body. Quiet box, inferred not verified. Correct arithmetic; must stop being called interleaved. |
| 2.20x | **SETTLED as a distinct pair** | 22.76 s to 10.36 s, loaded three-cycle round robin. Correct arithmetic. |
| 1.26x, 1.07x, 4.5 percent | **SETTLED**, consistent everywhere | 22.76 to 18.06, 22.76 to 21.28, and 20.40 to 19.49. `gpu_split_search.oblivious_wide_scan_requested` is the model site for how to write these. |
| noise-hoist null | **UNRESOLVED, never filed** | No artifact, no commit, no file behind the brief, no window label. A3 forbids an unlabeled indistinguishable from closing the item. Ten sites downgrade to "unresolved, measurement owed". |

**Verified in this lane.** All arithmetic above. The absence of all nine timing
values from all 663 `records.csv` rows and from every `bench/results/`
subdirectory. The absence of `NOISE_HOIST` from every results artifact and every
commit message. The absence of any file behind "the 2026-08-17 lane brief". The
`20.4s -> 9.8s` pair in `abbbf98`'s message body. The existence and non-use of
`MOJOTREES_BENCH_JSON`. The self-contradiction inside
`grow_tree_device_oblivious`'s docstring. The three stale meta-claims about the
`OBLIVIOUS_NOISE_HOIST_VAR` docstring.

**Inferred, not verified.** That 20.4 s is the quiet-box baseline. That
21.97 s to 12.34 s is a subtraction-only A/B in an earlier window rather than
some other arm combination, which the arithmetic supports negatively but cannot
prove. That the noise hoist's unrecorded window was the loaded one, which its
three siblings' recorded regime suggests and nothing establishes.
