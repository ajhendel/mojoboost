# Standing rules for every lane

Both campaigns brief from this file. It is the contract a lane is held to,
not advice.

This section is identical in every lane brief and is not negotiable. Read it
before the lane-specific section below it.

## What you are

You are one lane of a CPU performance campaign on mojotrees. An orchestrator
owns the round; you own one file group. The goal of the whole project is
**speed and accuracy only**. What the code currently looks like is irrelevant
and no existing structure is owed deference. If the right answer is to delete
something and rebuild it, say so.

**Target: speed and accuracy against LightGBM stock+det, one comparison,
headline end-to-end.** One comparator, LightGBM at its own defaults plus
`deterministic=true`. Every published number is end-to-end, binning plus
training, against their Dataset construction plus train. Speed and accuracy
are always reported together, thresholds are relative to stock+det, and every
row carries its conditions line. Nothing else is published.

**Every lane is in exactly one bucket, and its brief says which.**

- **A, speed.** Makes the same result arrive sooner.
- **B, accuracy.** Moves a number in `bench/real_data` toward the comparator.
- **C, makes the comparison valid.** Instruments, harnesses, gates, layouts
  and correctness work that no user sees but without which A and B cannot be
  believed.

**Nothing else launches.** A lane that is none of these is not a lane.

## How work is classified, and what it takes to close an item

**1. Strictly less work, same result: BUILD IT.** Fewer bytes, fewer trips,
fewer dispatches, fewer allocations, identical output. No measurement gate,
largest first. The wave window *measures* these; it does not *decide* them.

**2. A trade: SHIP IT BEHIND A SWITCH.** Task counts, floors, block sizes,
group widths, layouts -- anything that moves work rather than removing it.
It is A/B'd in the window before it becomes a default.

**3. Moves bits: TAKE THE REAL-DATA GATE.** `bench/real_data`, before the
change is believed.

**The gate is OUR OWN accuracy, and it stopped being a peer comparison on
2026-08-17.** The old rule was "within 1 percent relative of the better of
CatBoost as shipped and LightGBM stock+det". It was replaced because it fails
in BOTH directions, and the second one is why it was unsafe rather than merely
awkward.

- It blocked changes that cost nothing. The leaf-wise arm improved from 1.043 s
  to 0.839 s that day on bit-identical work, and the frontier table would not
  rank the arm at all, because it sits 1.42 percent behind CatBoost.
- **It permitted real accuracy loss whenever we were ahead.** The symmetric arm
  beats CatBoost at 799k, 0.303271 against 0.303468, so under a peer-anchored
  1 percent bar it could give away about 1.07 percent of its OWN accuracy and
  still pass. A rule that lets you lose accuracy because a competitor is weak
  is not an accuracy rule.
- The bar moved when somebody else shipped. A CatBoost release could fail our
  gate with no change to our code.

So: **our own accuracy is the gate, and competitors are the scoreboard.** A
change that moves bits is judged by what it costs US, on the same arm and the
same scenario, against a recorded reference. Peer numbers are reported beside
every result, always, and gate nothing. We still want to know we are 1.42
percent behind CatBoost. That is a column, not a verdict.

**The reference is an ABSOLUTE ANCHOR, not the previous run**, and that
distinction is the whole design rather than a detail. Anchoring on the previous
run gives a ratchet: lose 0.9 percent ten times, pass every time, end up nine
percent worse with no gate ever firing. The anchor is recorded per arm and per
scenario, it lives in a file, and it moves only by a deliberate act that shows
up in a diff, so drift accumulates against a fixed point.

Two consequences worth stating because they are easy to get wrong. A
bit-identical change costs zero of our accuracy by construction, so it passes
this gate trivially and rule 5 flips its default on measurement. And the
tolerance here is a DIFFERENT quantity from the old peer-relative 1 percent,
so reusing that number without an argument for it would be a mistake.

**4. An item closes ONLY when proven zero or proven impossible**, with the
evidence recorded. **Nothing is dropped for being small.** "Under one percent"
is not a reason to skip a category-1 change; it is a reason to rank it lower
and still do it.

**5. A SWITCH IS A TEMPORARY STATE, NOT A RESTING PLACE.** Added 2026-08-17
after this rule set was audited against what it actually produced. Rule 2 says
a trade is A/B'd in the window "before it becomes a default", and nothing in
this document ever made that second half happen, so the tree accumulated
measured, proven wins that shipped to nobody. On the day this rule was written
the library carried a 4.5 percent scan rewrite (20.40 s narrow against 19.49 s
wide, quiet box, ranges fully disjoint), an exact-integer sibling subtraction
worth **1.58x to 1.78x** (22.76 s to 14.39 s on a loaded three-cycle
interleaved round robin, and 21.97 s to 12.34 s in an earlier window nobody
wrote down), and a skipped last-level build worth 1.26x (22.76 s to 18.06 s,
that same loaded round robin). All three at 799,110 rows x 100 features x 100
trees, symmetric depth 6, on one M4, all bit-identical, all default off.

**A hoisted noise copy sat beside those three and this paragraph used to count
it as a fourth measured win. It is not one.** Nothing under `bench/results/` or
`bench/real_data/results/` measures that arm, no commit message mentions it, and
the only witness is a chat brief that is not a file, so its measurement is OWED
rather than taken. Until 2026-08-17 evening this paragraph also printed "1.78x"
alone for the subtraction, which is one end of a two-window range quoted as
though it were the result. `bench/results/FIGURE_PROVENANCE.md` carries the
pairs and the windows, and rule 10 below exists because of exactly this
paragraph.

So: **when an A/B resolves faster and identity holds, the default flips in the
same session as the measurement.** The switch then inverts, surviving one round
as an escape hatch to turn the new behavior OFF, and is deleted after that. A
switch that outlives a positive measurement is a defect and is reported as one.

The reasoning is worth stating because it removes the thing that felt like
caution. **A bit-identical change cannot alter any user's output.** Not the
model, not a prediction, not a digest. Flipping its default changes the clock
and nothing else, so there is no risk to weigh, and "measured faster, identity
proven, default off" is not a conservative position, it is an unshipped one.
For a change that MOVES bits, rule 3 stands unchanged and the accuracy budget
is the gate, because that one is a real trade.

### 5a. A FLIP IS A SWEEP OF THE FILE, NOT AN ANNOTATION ON THE LINE

Added the same day, hours later, because the first four flips made under rule 5
all got this wrong. **Four out of four.** In every case the predicate was
changed from `== "1"` to `!= "0"` and the reasoning was recorded as a comment on
the `return` statement, while the DOCSTRING immediately above it went on saying
"off unless asked for" and "nothing has measured it". Five further passages in
other functions and files described the same switches from their pre-flip state,
including two that labelled the arm we no longer take as "the shipped arm",
which is the label exactly inverted.

The consequence was not hypothetical. An explorer session read that tree and
reported three shipped, measured defaults as off and unmeasured, and a peer
built a lane ranking and a four-item work plan on top of it. All of it had to be
withdrawn. **The switch was correct in the code and wrong in every place a
reader would look**, which is worse than being wrong in both, because it passes
inspection.

So a flip is not complete when the predicate changes. A flip is complete when:

1. the predicate is inverted, and the escape-hatch spelling is `!= "0"`;
2. the docstring on that same function states the new default, the date, and
   the measured figures that justified it;
3. **the variable's name is grepped across `src/` and `docs/`**, and every
   passage describing the old state is corrected in the same commit;
4. any user-facing string that tells a caller how to select an arm is re-read
   against the new default. On the day this rule was written, the wide oblivious
   scan's bin-count refusal advised the user to "unset
   `MOJOTREES_GPU_OBLIVIOUS_WIDE` to use the narrow scan", and after the flip
   unset SELECTS wide, so a user following the error message reproduced the
   error. **An error message is code, not prose, and inverting a default can
   invert its advice.**

The general form, worth carrying beyond switches: **when you change what the
default behavior is, the code is the smallest part of the change.**

**6. A DECLINE MUST CARRY A PRICE.** Also 2026-08-17, and this is the rule that
would have caught the biggest defect in the codebase. A comment that declines an
optimization must state what the decline costs, with the arithmetic, or be
labeled as ASSERTED. **An asserted decline is an open item under rule 4, not a
closed one**, however confidently it is written. Two declines were falsified on
one day: the oblivious level batch declined sibling subtraction to save about
1.1 ms of launches while paying about 125 ms of doubled traffic, an error of
roughly a hundred to one, and it framed launch count and subtraction as
exclusive when a fused subtraction costing no launch already existed thirty
lines away. Separately, the last level's histograms were built and never read
because skipping them was said to move a copy-back cost, which confused command
buffers with work. Both read as settled engineering. **When you decline
something, price it, and when you find a decline you cannot price, that is a
finding and not a footnote.**

**8. THE IDENTITY BUG: WHEN NOTHING FAILS, SUSPECT THE KEY.** Added 2026-08-17,
after a single day produced eleven instances of one defect. Every one was
internally consistent code with a wrong IDENTITY, and not one of them failed a
test. They come in two mirrored forms.

**An identity carrying a dimension it should not.** `frontier.py` put the device
inside the arm id, so a cpu cell and its gpu cell were different arms and a
351-job run produced ZERO oracle cells while every accelerator row reported "no
cpu twin". `verify.py` keyed a cell on the engine rather than the arm, so forty
arms collapsed into one verdict comparing whichever two records were written
last, at different tree counts. Four device and host tables were sized from
`num_leaves`, which does not bind under oblivious growth, so a depth-6 symmetric
tree met a table built for 31 leaves; two of those raised and two under-reserved
in silence.

**An identity missing a dimension it needs.** A role test compared an
ENGINE-name list against an arm id, matched nothing, and emitted no lines at
all, so the accuracy scoreboard simply vanished from every `--arms` run. An arm
recorded without the parameters that make it that arm is the same shape, which
is why resolved parameters travel in the record.

The rule that follows. **A gate that emits nothing is indistinguishable from a
gate that passes**, so silence is a finding and not a clean bill. When a check
is green, confirm it compared the things you meant, on the rows you meant, at
the identity you meant. `verify.py::check_coverage` is the mechanical form of
this: every subject cell must be NAMED by at least one gate, and a cell no gate
mentions is a WARN. Prefer that shape wherever a gate can be given one.

And when you find one instance, sweep for the rest, because this defect breeds.
Of the eleven, one was found by looking and ten were found by asking where else
the same identity was assembled.

**7. A STALE CLAIM IS A DEFECT, AND FINDING ONE OBLIGATES YOU TO FIX IT.**
Added 2026-08-17 (Andrew: "we need to always update if we find something
wrong"). This codebase documents itself heavily, which is a strength and is
exactly why a wrong comment is expensive: it reads as settled engineering and
gets believed instead of checked. Several of the largest wins of that day came
from falsifying a confident comment, and several near-misses came from
believing one.

So: **if you find a claim that is wrong, correcting it is part of your lane, not
a note for somebody else.** If the claim is in a file you own, fix it in the
same pass. If it is in a file another lane owns, quote the exact replacement
text in your report so the orchestrator can apply it verbatim, and say plainly
that it is a CORRECTION rather than a suggestion, so it is not triaged as an
improvement and deferred. A corrected claim records what it used to say and
when it changed, rather than quietly reading as though it always said the new
thing, because the next reader needs to know a belief moved.

This applies to measured RESULTS as much as to prose. A recorded number whose
arm never reached the code is not a small number, it is a NULL, and leaving it
filed as a result means it will be believed again. On the day this rule was
written, `MOJOTREES_CPU_LAYOUT_BY_NODE` was recorded as measuring "neutral" on
the symmetric CPU grower, and the switch had never reached that grower at all.
The conclusion drawn from it, that per-node layout is not where the symmetric
CPU cost lives, was unsupported for weeks.

**VERIFY BEFORE YOU CORRECT, because a correction applied on a stale premise
installs a new false claim.** An audit lane that day was handed four confident
corrections and found one of them already fixed in the working tree by another
lane hours earlier. Applying it would have re-asserted a condition the code no
longer has, and would have narrowed a finding that in fact still stands. A
correction is a claim like any other and takes the same standard of proof.

**CITE BY NAME, NOT BY LINE NUMBER.** This is the single most common form of rot
in this repository and it is a mechanical consequence of how we work: many lanes
edit the same files on the same day, so a line number is stale within hours
while the substance stays true. Every drifted citation found in that audit had
correct content and a wrong pointer, which is the worst combination because the
reader concludes the claim is wrong rather than the pointer. So anchor on
`file.function`, or quote the line of code you mean, and keep line numbers only
as a convenience beside a durable anchor. A document whose numbers have already
drifted should say so at the top rather than be half renumbered, because a
partly refreshed table is less trustworthy than a uniformly stale one.

The CPU path is the correctness ORACLE and the portability floor. It is not
optimized (2026-08-17 ruling). `device_agreement` and `backend_proof` in
`bench/real_data/verify.py` are built on it actually running, and on
2026-08-17 `device_agreement` caught a live noise-hash divergence in the
shipped defaults, so it earns its keep as a gate rather than as a product
claim. It is also genuinely faster than the device below roughly 150,000 rows,
and it is the only backend our CI can execute at all. What changed is where
engineering effort goes, not whether the path exists: speed lanes are GPU
lanes, and a CPU speed lane needs a reason beyond the number being improvable.

**9. TWO KINDS OF DEFAULT, AND AN EXPLICIT EXCHANGE RATE FOR ONE OF THEM.**
Added 2026-08-17, from Andrew, and it settles a confusion that produced real
mistakes in both directions on the day it was written.

**A MIRROR arm mirrors a peer's resolved defaults and never moves.** Its whole
purpose is that everything is held constant except the thing being compared, so
"our arm would score better with a different value here" is not an argument for
changing a mirror, it is an argument for a separate row. When our own default
moves, a mirror must be REPINNED to the peer's value to stay a mirror, and the
sharpest instance is `lambda_l2`, which had to be named explicitly in
`bench/real_data/scenarios.py`'s `BASE_PARAMS` the moment our shipped value
diverged, because an absent key had silently meant "both at the same 0.0" and
began meaning "each at its own different value" under a label that still said
mirror.

**A USER DEFAULT is chosen on the speed-accuracy frontier, and the exchange
rate is stated rather than felt.** One rule for every growth policy, because a
switch that is good for one is good for all unless someone can say why not.

- A change is ADOPTED if it costs at most about **1 percent of accuracy on the
  primary metric** and buys at least about **2x speed**. The multiple was 1.5x
  for a few hours on 2026-08-17 and Andrew raised it to 2x the same day. Take
  the higher number; a 1.5x that costs a real percent of accuracy is not a
  trade worth making on a library whose pitch is both.
- Accuracy is read on the **excess-error lens wherever a floor exists**, not on
  raw RMSE. On a target with a 0.30 noise floor a mechanism that halves the
  model's own error moves RMSE by under 1 percent, so a raw-RMSE budget of 1
  percent is a 50 percent model-error budget and would wave through almost
  anything. `bench/real_data/decompose.py` computes the lens.
- A change costing MORE accuracy than that needs a bigger speed multiple and a
  written price in `docs/design/ACCURACY_BUDGET.md`. There is no threshold above
  which the answer is automatically no, and none below which it is automatically
  yes.
- A **bit-identical** win is not on this frontier at all and flips immediately
  under rule 5. It costs zero accuracy by construction, so no rate applies.

**The exchange rate does NOT reach internal choices, and getting this backwards
is the most expensive mistake available here.** Corrected 2026-08-17 within
hours of the rule being written, from Andrew, because the first draft said the
rate applied everywhere and that is wrong.

**A choice a user cannot reverse must not cost accuracy.** That is the whole
rule and the reason is not subtle. An exchange rate is a bargain offered to
someone who can decline it. A user who never learns that a kernel rounds
differently, or that a bin ceiling truncates their categories, has not accepted
1 percent for 1.5x; they have simply been given a worse model by a library that
decided on their behalf and did not say so.

So internal choices are held to a different and stricter standard:

- **There is NO exchange rate for an internal choice. None, at any speed.** It
  must be exact or accuracy-neutral, meaning within `device_agreement` tolerance
  and clean against the recorded anchor. A MEASURABLE accuracy cost in a place a
  user cannot reach is a **defect regardless of what it buys**, so "it is only
  0.3 percent and it doubles throughput" is not an argument here, it is a
  description of a defect with a benchmark attached.
- **If an internal choice WOULD trade accuracy for speed, it has to be EXPOSED
  as a knob first.** Only once a user can see it and reverse it does the
  exchange rate above decide which side of it is the default. Exposure is the
  price of admission to the frontier, not an optional courtesy afterwards.
- **Anything internal that costs accuracy today is a DEFECT or a declared gap.
  It is never a priced trade.** Filing it as a priced trade launders a defect
  into a decision, and the filing is what makes it stop being looked at.

Three live cases, named so this is a test rather than a sentiment. The
**categorical bin ceiling** costs measured average precision and no user can
raise it, so it is a defect and the widening lane is its fix, not its price. The
**multiclass leaf-value mechanism at `lambda_l2 = 0`** showed 3.31x worse
logloss at unchanged accuracy, which is a leaf-VALUE signature rather than a
leaf-assignment one, and it is an open item. **Float32 gains in the device split
search near a tie** are the borderline case and must go one way or the other:
made exact, or exposed and priced. Borderline is not a resting place either.

What survives from the first draft is the observation that motivated it. A
silent internal choice IS a default, in the sense that it decides what every
user gets. The conclusion drawn from that was wrong. It does not mean such
choices join the frontier; it means they must not be on the frontier at all.

**Both directions of error are real and both happened.** Declining a free win
because a rule said "do not change defaults" cost us four shipped-but-unshipped
optimizations. Accepting a loss because we happened to be ahead of a competitor
is the failure the old peer-anchored rule permitted, which is why rule 3 anchors
on our own recorded absolute value instead. The exchange rate exists so that
neither of those is a judgment call made fresh each time.

**10. A SPEED FIGURE TRAVELS WITH ITS RUN, OR IT DOES NOT TRAVEL.** Added
2026-08-17, after three lanes independently flagged the same three figures and a
forensic pass found that **not one of them was wrong**. That is what makes this
rule necessary rather than pedantic. If the numbers had been miscopied the fix
would be arithmetic. They were all correct, they still could not be reconciled,
and the reason is that a ratio had been separated from the pair of times it was
computed from.

Here is what one day produced. Four absolute baselines for ONE shape, 799,110
rows x 100 features x 100 trees, symmetric depth 6, on one M4. **17.07 s** from
the filed run `20260817T110847Z-dense1mfixed`, **20.4 s** from a quiet box,
**21.97 s** from a window nobody wrote down, and **22.76 s** from a loaded
three-cycle interleaved round robin. Every published ratio was exact against one
of those four and no site said which. So 1.78x and 1.58x for one arm, 2.08x and
2.20x for one combination, twenty-five citation sites, zero arithmetic errors,
and nothing checkable by anybody. **Two correct figures looked like a
contradiction for a whole day because nobody wrote down which box they came
from.** On a machine that `PROFILE_PROTOCOL.md` A3 records as drifting two- to
threefold between windows, a ratio without its baseline is not a weak result, it
is not a result at all.

So:

1. **The quotable unit is a RUN, not a ratio.** A speed figure may be published
   only with a **run id** that resolves to a directory under `bench/results/` or
   `bench/real_data/results/`, a **shape** (rows x features x trees, growth
   policy, max depth), an **arm set** with every switch that was set resolved to
   its value, **both absolute times**, the **M0 verdict**, and the **regime
   label** A3 requires. Six fields. A figure missing any of them is ASSERTED
   under rule 6 and is an open item under rule 4, however carefully it was
   measured.

2. **Stop transcribing. The generator emits the table and the documents cite the
   run.** `bench/bench_train_gpu.mojo` already prints a `json_summary` record
   carrying `arm`, `baseline`, `speedup_x`, `delta_pct`, `noise_floor_pct`,
   `verdict`, `n_rows`, `n_features`, `n_trees`, `objective`, `seed` and
   `repeats` plus every arm's samples in the order they ran, and
   `MOJOTREES_BENCH_JSON=<path>` files it. **Every timing run sets that variable
   before it starts**, to a path under `bench/results/<run-id>/`. A prose site
   then cites the run id and quotes at most one number from it. All four figures
   this rule was written about were taken with that sink available and not one of
   them used it, which is the entire distance between a reconciliation lane and a
   `grep`. **A rule that asks a human to remember is weaker than one that points
   at a mechanism which already exists**, and this one already exists.

3. **A chat brief is not an artifact.** Eight documents cite "the 2026-08-17 lane
   brief" and there is no such file in this repository. A brief is how a lane
   receives work, not how the tree records a result. **If the only witness to a
   number is a conversation, the number is unfiled.** Cite the path or downgrade
   the claim.

4. **A commit message is write-once, so it is the last place a figure may go and
   never the first.** `abbbf98` published 1.58x and 20.4 s to 9.8 s, the
   documents published 1.78x and 22.76 s to 10.36 s, both are correct, and the
   commit can never be edited to say so. Quote a figure in a message only after
   it exists under `bench/results/`, and put the run id beside it so the message
   stays checkable after the prose moves on. `bench_train_gpu._arm_conditions`
   already records that this project has discarded "a pair of figures whose
   conditions were written down only in a commit message". It has now done it
   twice.

5. **Copy the conditions line, not the number.** The worst site found was not a
   wrong figure, it was a right figure wearing another run's clothes. 2.08x came
   from a quiet-box 20.4 s to 9.8 s pair and is described in two places as
   "interleaved round-robin", which is the OTHER pair. A number with a borrowed
   provenance passes every inspection and cannot be checked by anybody, which is
   the same pathology rule 5a names when it says a flip correct in the code and
   wrong everywhere a reader looks is worse than being wrong in both.

6. **Until a run resolves it, publish the RANGE and say why.** A3 already says to
   report both numbers and refuse to pick one where the effect size differs by
   regime. Rule 10 adds the enforcement. **"1.58x to 1.78x, two windows, one of
   them unlabeled" is publishable and "1.78x" is not.** Picking the flattering
   end of an unresolved range is not optimism, it is an unfiled claim with a
   decimal point on it.

One correction found while writing this rule, kept here rather than tidied away,
because it is this rule failing on its own first draft. That draft attributed the
17.07 s baseline to "the 2026-08-16 comparison run". It is not from that run. It
is the median of the five `mojotrees_catboost_mode` gpu repeats in
`20260817T110847Z-dense1mfixed`, taken 2026-08-17, and the 2026-08-16 large-tier
run carries no symmetric arm at all. **A citation with no run id gets its own
provenance wrong even when the number is right**, which is this rule in one
sentence.

The general form, worth carrying past benchmarks. **A number is a claim about a
procedure, and a claim whose procedure is not recorded cannot be wrong, which is
exactly why it cannot be trusted.**

### 11. THE CPU PATH IS NOT AN ORACLE THAT MAY NOT BE OPTIMIZED. RETIRED 2026-08-18.

The standing convention was that the CPU path exists to certify the GPU path
and is therefore never optimized. It was never written in this file, which is
part of why it outlived its premise for two days after the premise died.

**The premise was that the GPU path is the fast one.** On 2026-08-18, in
NVIDIA's gbm-bench on this M4, interleaved arms and three repeats: covtype
581,012 x 54 over 7 classes, our CPU 28.077 s against our own GPU's 40.894 s.
The CPU arm is 1.45x faster than our accelerator and it is 3.1x behind
LightGBM's 9.024 s. **The oracle is the product.** A rule that forbids
optimizing the fastest thing we ship costs the headline number and protects
nothing.

It also protected the wrong thing. The oracle's job is to certify that the
device produces the right numbers, and it does that by being A correct
implementation whose results are checked against the device's. It does not do
that by being slow, and it does not do that by being frozen. The property that
matters is that the two arms AGREE, not that either is unchanged since some
past date.

**The replacement is a classification, not a freeze.**

**Tier 1, bit-identical, land freely.** A change that provably keeps the same
Float64 additions in the same order is not an oracle question at all. It
changes which address a byte loads from, how many features share a walk, how
many fan-outs a node pays, or how wide an integer is. The gate is the digest,
not a review: `bench/bench_serial_kernel.mojo` already prints a bitwise model
digest per kernel arm and refuses to call two arms equal without it, and
`tests/test_row_major_bins.mojo` and `tests/test_grow_bin_layout.mojo` already
assert layout bit-identity through a whole grown tree. That harness exists;
use it. Rule 5 then applies unchanged, so a bit-identical measured win flips
the default in the same session.

**Tier 2, arithmetic-changing, RE-ANCHOR rather than fork.** When a change
moves CPU bits, do not keep the old kernel beside the new one. Record the
absolute accuracy anchor and the CPU/GPU agreement figure at the pinned shapes
BEFORE landing; land the change; regenerate the goldens in the same commit
with the pre and post agreement figures both in the message; and keep the OFF
SWITCH rather than the old code, because a bisection switch costs one branch
and a duplicate kernel costs a maintainer forever.

**Do not keep a slow reference implementation as a second path.** This
repository has already tried that shape and priced it twice. The four
`SERIAL_KERNEL_*` arms keep a pre-optimization scatter alive as `base` at the
cost of ninety duplicated lines that must stay correct forever while nothing
ships them. And `_env_layout_by_node` is the warning about what goes wrong: the
switch did not reach the symmetric grower until 2026-08-17, so an earlier
measurement of it recorded "neutral", and **neutral is what a switch that does
not reach the code always measures.** An unexercised reference arm does not
fail loudly, it rots into confident nulls.

### 12. A PEER'S DEFAULT MAY SET OUR DEFAULT AND MAY NEVER SET OUR CEILING.

Added 2026-08-18, from an audit of `src/mojotrees/gpu_split_search.mojo`. This
is a defect class the rule set had nothing against, which is why it shipped.

`gpu_split_search.OBLIVIOUS_MAX_LEAVES` was 64, and its own docstring said
what 64 was. It read "which is `2 ** 6` and therefore CatBoost's default depth
exactly." It was not a default. It was a HARD CEILING, and above it the device
oblivious grower refused to run and raised. **There is no citation anywhere in
this repository for CatBoost's own GPU depth LIMIT**, and there could not be,
because a default is not a limit. The ceiling turned users away at exactly the
depth every third-party benchmark harness sets, and it cost a day.

**Mirroring a peer's default is the POINT of this library. Mirroring it as a
refusal is the defect, and the two are one keystroke apart.** Defaulting
`num_leaves` to 31 because LightGBM does is the whole promise, since a user who
ports a configuration gets the model they ported. Refusing to RUN above 31
leaves because LightGBM defaults there would be a defect on the same fact. A
default says "this is what you get if you say nothing". A ceiling says "this is
what you get instead of what you asked for". The first is a service and the
second is a refusal, and **a refusal may only be sized from OUR kernel, OUR
allocation, OUR measured data, or OUR portability floor.**

The test, and it fits on one line. **If the peer moved their default tomorrow,
would this number move?** If yes, it is not a constraint. It is a coincidence
with an argument attached, and the argument was written after the number.

Three things the audit established, each of which the ceiling failed.

**A ceiling carries its arithmetic on the line, the way a crossover threshold
carries its `evidence_id`.** `device_policy.CrossoverEvidence` will not let a
performance rule exist without a run and a machine, and its constructor refuses
a rule with no `evidence_id`. A capacity refusal is held to that same standard,
because it decides more than a route does. `OBLIVIOUS_MAX_LEAVES` claimed a
threadgroup-memory bound. The shipped wide kernel's twelve shared arrays are
12,300 bytes at depth 8 against a conservative 16,384 byte budget, and the
ceiling sat at depth 6. **One subtraction would have shown it**, and nobody had
to do the subtraction because the number already had a story.

**A second justification is not corroboration, it is two chances to be wrong.**
That ceiling carried three. A memory bound with room for two more doublings. A
launch count taken from `gpu_resident_round.oblivious_launch_census`, a function
whose own docstring calls itself a frozen prediction of a schedule that does not
exist, while the built `oblivious_schedule_launches` returns 63 at depth 7 where
the refusal quoted 71. And a table-sizing claim standing in front of three
capacity tests that already measure the real allocation and decline gracefully.
**Three reasons that agree are ONE reason if they were all reverse-engineered
from the same number**, and a reader counting reasons rather than checking them
finds a ceiling more credible the more often it is wrong.

**A knee is a price, a wall is a wall, and you do not refuse at a price.** Past
64 command buffers Metal backpressures. The queue blocks, it does not drop, and
nothing fails. The shipped leaf-wise plane runs 278 command buffers a tree at 31
leaves and 2,303 at 256 and is the fastest arm we have, which is the measurement
that settles what that knee costs. A slower run is a number the user can read.
A raise is a run they do not get.

**The one-line version was already written correctly once in this repository**,
in `embedding.LdaParams.resolved_components`, and it is the model to copy. "The
cap is only the default. An explicit larger `components` is accepted here as it
is there ... It is a legal request and a bad one." **A legal request we think is
bad gets a default and a docstring. It does not get a raise.**

### 13. A REFUSAL THE USER CANNOT SEE IS NOT A POLICY, IT IS A SILENT MODEL CHANGE.

Added 2026-08-18, fixed the same day in
`python/mojotrees/sklearn.py::_warn_about_device_decision`.

`device_policy.decide_device` built a full decision for every fit, carrying the
reason, the blocks, the memory estimate and the evidence id. It was serialized
across the boundary and parsed back. Then `_resolve_device` ended `return
select_device(...).resolved` and discarded all of it, and there was no
`warnings.warn` anywhere in the device-selection path. **Thirteen distinct
blocks could move a fit to the CPU and say nothing.**

That is not cosmetic, and the reason it is not is a measurement.
`bench/results/COMPARISON_RUN_2026-08-16.md` records 51,630 of 51,630 test rows
differing between the CPU and GPU arms on real year data, median absolute
difference 0.4601, maximum 9.67, traced to the shared fixed-point Int32
histogram rather than to the split search. **The two backends do not produce the
same model. Therefore every route decision is a model decision**, and a route
decision the user cannot read is an unreported change to their numbers.

**The counter-lesson is half the rule, because the first fix was wrong in the
other direction.** Emitting every warning the decision carries produced SEVEN
per fit, six of them provenance caveats. The hardware identity came from the
build target, the capabilities are synthetic, no memory budget was reported, the
session is cold, the objective was not declared. All true, all belonging in
`explain_device_choice`, none actionable. **A library that raises seven warnings
per fit teaches people to filter its warnings, which costs more than the silence
it replaced.** The shipped emitter is a whitelist for exactly that reason.

So the rule has two halves and neither stands alone. **Warn on what the user can
act on, and put the rest in the report they can ask for.** An `auto` request
that was BLOCKED is warned, because the accelerator refused a parameter rather
than the CPU being predicted faster. An `auto` request that landed on the CPU
because no crossover rule covers the shape is the normal quiet answer and is not
warned.

### 14. A REFUSAL MUST BE REACHABLE FROM THE SURFACE THAT TRIGGERS IT, AND THE TEST IS A CALL, NOT A READ.

Added 2026-08-18, fixed the same day in `bindings/basic_bindings.mojo`.

`device_policy.BLOCK_GROW_POLICY` and `BLOCK_MAX_DEPTH` shipped implemented,
documented and tested, and were unreachable. `bindings/basic_bindings.mojo`
never sent `grow_policy` or `max_depth` across the boundary, so the native
request took its own defaults and both blocks silently never matched. A user
asking for symmetric trees at depth 8 with `device="auto"` got a hard exception
out of `train_gpu` instead of the CPU fallback the policy layer implements, and
every layer in that path was individually correct.

**Three more are unreachable today and are named here so they are not
rediscovered as new**, which this project has now done twice with the same
shape. `bundling`, `linear_tree`, and `forced_splits` are all refused by
`device_policy` and none of the three fields crosses the boundary. The sharpest
instance of the harm is the forced-splits raise, which advises the caller to use
`device='auto'`, "which routes around this". **It does not, because the field it
would route on is not sent. The error message advertises the disconnected
path**, which is rule 5a's failure mode arriving from a direction rule 5a does
not cover.

The rule. **A block is not shipped until one call from the TOP surface reaches
it.** Not a unit test of the predicate, which is what all four of these had.
Not a reading of the code that would call it. A call through the estimator, in a
test, asserting the block fired, per the correctness contract's requirement that
a gated path prove the gate opened.

And note the second way a gate can be unreachable, because it does not look like
this one at all. **A gate can be a tautology.**
`device_policy.gpu_supports_outputs` is `return n_outputs >= 1`, its docstring
explains that it is kept as the one place to reject a future workload, and its
block cannot fire for any input the vocabulary can present. That is a defensible
thing to keep and an indefensible thing to count, so it is not coverage and must
not be reported as any.

### 15. A REFUSAL'S MESSAGE IS AN API, AND EVERY "THE DEVICE CANNOT" IS A CLAIM WITH A DATE ON IT.

Added 2026-08-18. Rule 5a established that inverting a default can invert an
error message's advice. The same thing happens, from the opposite direction,
when a capability LANDS. The default does not move, the code the message
describes does, and the message keeps describing the code that used to be there.

Four found in one pass on 2026-08-18. `BLOCK_VALIDATION_SET` says validation
forces the CPU while a complete GPU eval-set trainer exists.
`BLOCK_RANKING_OBJECTIVE` says ranking trains on the CPU only while a device
ranking gradient path exists. And two Python-side comments still say the policy
blocks every non-L2 `score_function` selector, which stopped being true on
2026-08-17 when the narrowing landed and Cosine became a device capability.

The rule. **When a lane lands a capability, it greps the claims.** Every
refusal message, every block docstring, and every comment that says the device
cannot do the thing that just landed, in the same commit that lands it, under
rule 7. A stale refusal message is worse than a stale comment because a user
obeys it. It is the one piece of prose in this repository that people act on
without reading anything else.

### 16a. THE RUN BAN IS FOR SUBAGENTS. THE ORCHESTRATOR MEASURES.

Clarified by Andrew on 2026-08-18, and it replaces the reading everyone had
been operating under.

**Subagents run LOCAL TESTS ONLY.** A targeted test file, a build, a fit small
enough to prove a claim. Never the suite, never a benchmark, never a timing,
never a profile.

**The orchestrator runs the suite and the measurements**, without asking per
run. The old default of "run nothing until asked" was read as covering the
orchestrator too, and the cost of that reading is on the record: 45 commits
landed on 2026-08-18 including a default flip, a lifted depth bound, five
refusal blocks that now fire where they previously could not, POLICY_VERSION
from 8 to 10, a new refusal on rankers and a kernel rewrite in the oblivious
commit path, and **the suite did not run against a single one of them.**

Two of that day's bugs were caused by exactly that gap. A depth bound was
raised, the build was green, and the commit message said the ceiling was gone
while it still refused. And raising a constant disarmed a guard nobody knew
was load bearing, which silently corrupted every oblivious tree deeper than 7
for an hour. Both were caught by a targeted fit, which is a weaker instrument
than the suite and happened to be enough.

**Why the split is the right shape rather than a compromise.** A subagent
cannot see whether another lane is mid-measurement, so a subagent that
benchmarks is a subagent that corrupts somebody's numbers without knowing. The
orchestrator can, because it declared the window. The restriction was never
about trust or about compilers, it was about who holds the schedule.

### 16. WHEN A MEASUREMENT IS PENDING, THE BOX IS THE INSTRUMENT AND EXACTLY ONE PROCESS TOUCHES IT.

Added 2026-08-18, and it REPLACES the older blanket rule that subagents never
build. That rule was right about its consequence and wrong about its reason.
It was protecting the instrument, not the compiler, and stating it as a ban on
compiling made it look like a rule about subagents.

**The incident.** Three implementation lanes were authorized to build at once,
in the same window as a queued three-arm timing A/B that an advisor had
correctly asked to be run FIRST, because a GPU arm's ratio to the comparator
had moved 18 percent across two runs with twenty commits between them. The
compiles took the box out of any state where a timing is valid. The A/B did
not run. The regression question stayed open for the rest of the session, and
the covtype profile behind it stayed blocked, because the one thing that could
have settled both was the thing the box could no longer do.

Nobody did anything forbidden. The orchestrator changed the subagent-build
rule on an explicit instruction to let lanes fix and test things, and did not
notice that the instruction and the pending measurement were in conflict. The
conflict is invisible if you think the rule is about subagents.

**The rule, in two halves.**

Subagents may build when no measurement is pending. Building is not the
hazard and never was.

When a measurement IS pending, the box is the instrument. Exactly one process
touches it, the orchestrator DECLARES the quiet window, and nothing else
builds, fits, benchmarks or profiles inside it. That includes the
orchestrator's own convenience builds, which is the half most likely to be
forgotten, because they do not feel like someone else's work.

**Declaring the window is the orchestrator's job and it is explicit.** Say it
to every live lane. A lane that has not been told a window is open has no way
to know, and "I assumed nobody was building" is the same shape of evidence as
"it builds" and "the switch is set": it is a belief about reach, not a check.

**Why this file cares, given that it otherwise cares about correctness.** This
machine drifts two to three times across thermal windows, which is why the
interleaved-arms protocol exists at all. Interleaving defends against drift
BETWEEN windows. It does not defend against load INSIDE one, because a
compiler running through half a repeat moves that repeat and not its
neighbors, and the spread then reports a difference that is a compile. A run
contaminated that way is worse than no run, because it produces a number with
a spread attached and both are wrong.

## What you may run. This is a hard limit.

**You may run ONLY:**
- package compile checks (`pixi run build-pkg`, or a `mojo build` of one file)
- **your own test file(s), named explicitly, one at a time**:
  `bash tools/run_tests.sh cpu <your_test_name>`

  **Run it once, after your last edit, and never a second file to confirm.**
  For one or two files add `MOJOTREES_TEST_PKG=0`, which skips the package
  build and compiles only the modules your test imports. Measured on one
  49-test file, warm, interleaved, two repeats each: **0.75 s against 12.2 s**,
  and at two files 5.4 s against 16.3 s. The package build pays for itself only
  across several files, which is the suite case and not yours. It is also
  skipped automatically now when nothing under `src/` has changed since the
  last build, so a re-run of an unedited file costs about a second either way.

**You may NOT run**, under any circumstance, without exception:
- any test suite: `tools/run_tests.sh all|cpu|gpu` with no file named, or
  `pixi run test*`. **A suite counts as a compile** and invalidates the other
  campaign's measurements exactly as a benchmark would.
- `tests/test_golden_bits.mojo`. The orchestrator runs it after every merge.
- any benchmark, any profiling run, any `bench-*` task
- **any training run of any kind.** Do not train a model to see if something
  got faster. You have no clock and no quiet box; a number you produce is
  worse than no number because somebody may believe it.
- anything with a timer in it
- any `pixi run` task that builds a Python extension

The orchestrator runs every suite and takes every timing. This is not a
formality: a second orchestrator is running a GPU campaign on this same
machine, and a compile or a benchmark you start can invalidate somebody
else's measurement without either of you finding out. Two results have
already been discarded in this project for exactly that, one taken at 18.6
percent spread while agents were compiling.

**You cannot measure anything.** Therefore every number in your report is
labelled **estimated** or **derived bound**, never "measured" and never
"faster". A derived bound is arithmetic over bytes, allocations, or counts,
and it is a bound rather than a prediction. If you catch yourself writing
"this makes it about 20 percent faster", you have written a claim you have no
instrument for; write the byte or allocation arithmetic instead and let the
orchestrator measure it.

## Where you work

You have your own git worktree on your own branch off `cpu-round-1`. Work
only there.

- **Never run `git checkout <branch>` in the main checkout** at
  `/Users/andrewhendel/CascadeProjects/mojotrees`. It stays on `perf-round-2`,
  which is the GPU campaign's branch, and switching it breaks their session.
- Never `git add -A`. Live worktrees under `.claude/worktrees/` must stay out
  of the index. Add by explicit path, always.
- Do not merge your own branch anywhere. The orchestrator merges, one lane at
  a time, and runs the suites between merges.
- **Run `source tools/lane_env.sh` once, before anything else.** A worktree
  contains `pixi.toml`, so a bare `pixi run` treats it as its own project and
  installs a second complete copy of the environment into `<worktree>/.pixi`,
  about 1.1 GB, before it compiles a line. It also gives you your own empty
  Mojo compile cache, because `MODULAR_HOME` follows the environment.
  Measured 2026-08-16: 46 lane worktrees had done this, **49 GB** of
  duplicated environments, with caches of 1.1 MB to 243 MB against the main
  checkout's 8.7 GB. Every lane was starting cold and sharing nothing.
  Sourcing that file points `PATH` and `MODULAR_HOME` at the main checkout's
  environment, copies nothing, and declines with a message if your
  `pixi.toml` differs from the main one. `tools/run_tests.sh` already does it
  for you; the commands that need you to do it yourself are `pixi run
  build-pkg` and any bare `mojo build`.

## File ownership, which is the only isolation this round has

Your lane-specific section names the files you own. **You may edit those and
nothing else.**

If your change requires a change in a file you do not own: **STOP. Do not
edit it.** Report the exact change you need — file, function, signature, and
why — and the orchestrator will either make it as glue or sequence another
lane to do it. A lane that edits outside its ownership is not creating a
merge conflict, it is damaging another campaign's in-flight work, and there
is no branch to throw away to recover.

**Three symbols are GPU-visible contracts. No CPU lane changes their
signature or their semantics, and `tests/test_const_hessian_exclusions.mojo`
is off limits to every lane:**

- `boosting.round_has_constant_hessian`
- `histogram.objective_has_constant_hessian`
- `histogram.CONSTANT_HESSIAN`

The GPU trainer declares constant-hessian once per fit as builder state and
cannot withdraw the declaration mid-loop, so a change to what these mean
breaks a backend you cannot see.

## Correctness contract

- **Determinism across `MOJOTREES_NUM_WORKERS` is required.** Values must be
  identical at 1, 3 and 8 workers, and on any machine on this toolchain.
  Write a test that proves it for your change. Determinism is not negotiable
  in this round.
- **Bit-identity with *past* output is NOT required.** That is a deliberate
  relaxation for this round. If your change moves bits, that is allowed — but
  you must **say so explicitly and stop**, rather than regenerating the golden
  fixture yourself. Golden re-baselines are serialized across two campaigns
  and land one at a time with their ulp movement stated; the orchestrator
  sequences them.
- **Exact comparisons only.** `to_bits()` or integer equality. **No
  tolerances anywhere in a test.** A test that needed a tolerance did not
  establish what it claims.
- **A test for a gated or conditional path must PROVE the gate opened.**
  Assert a counter, a trace line, or a path marker. Never assume it. This
  project has already shipped a test whose six fixtures all ran below the
  gate and verified nothing, and a second one that compares two arms which
  are equal whether or not the optimization fired.
- **Any change that moves a multiply relative to an add is a numerics
  change.** FMA contraction has cost this project two results. Flag it.

## Mojo 1.0 facts that will otherwise cost you an hour

- **No partial field moves.** `__disable_del` and `fn` are gone. Use
  `.copy()`.
- `ref` is a keyword.
- **No module-level globals.** Thread a struct instead.
- Any GPU entry point needs `comptime if not has_accelerator(): raise` around
  the whole body with an `else:`. Irrelevant to a CPU lane except as a
  compile hazard if you touch one — which you should not be doing.
- `tools/run_tests.sh` selects the accelerator subset by **name** (anything
  matching `test_gpu_*` unless marked `# run_tests: cpu-safe`) as well as by
  a hand-maintained list and by content. **Do not name a CPU test
  `test_gpu_*`** or it will be silently excluded from the CPU suite.

## What your report must contain

1. What you changed and why, in mechanism terms.
2. The **derived bound** or **estimate** for what it should be worth, with
   the arithmetic shown, labelled as such.
3. Whether bits moved. If yes, exactly which values and why.
4. What you could not do because it was outside your ownership, with the
   exact change you would have needed.
5. What you are unsure of. A lane that reports no uncertainty is not being
   read as confident, it is being read as not having looked.

A lane that lands correct, tested, and moves nothing is a **null**, and a
null reported clearly is worth more than a win reported loosely. This project
removed 1,300 device copies per fit for 16 milliseconds and the honest report
of that reordered the whole plan. Do not oversell.
