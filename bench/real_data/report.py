"""The performance report. Prints numbers, decides nothing.

    python bench/real_data/report.py results/<run_id>
    python bench/real_data/report.py results/<run_id> --markdown report.md

Rules this report follows, all of them there to stop a number from
travelling further than it deserves:

- Every cell shows the median of the repeats and the min and max around it.
  A single number hides whether the machine was busy.
- A ratio between the two engines is printed only when there are at least
  as many repeats as thresholds.json requires. Below that the column reads
  "n/a", not a number with a caveat somebody will drop when they quote it.
- Runs from different machines, different builds, or different library
  versions are never put in the same table. envinfo.comparable_key decides,
  and a mixed run is split into one table per key.
- A cell whose spread across repeats is wide is marked. Wide-spread numbers
  are still printed, because hiding them would be worse, but they carry the
  mark wherever they go.
- Every table repeats the conditions underneath it: thread count, device,
  data kind, and whether the machine was on battery or thermally limited.
- Every timed phase that can be threaded prints its parallel efficiency in
  the column beside it. A thread count is what a run was asked for and
  parallel efficiency is what it got, and a table that prints only the
  first invites a one-core measurement to be read as a slow eight-core one.
- The histogram construction each engine ran is printed under the table,
  because two engines building histograms by different strategies is a
  fact about what the timings mean rather than a footnote.
- An ORACLE row is printed and is labelled. From 2026-08-17 a subject arm on
  the cpu, in a run that also ran that arm on an accelerator, is an oracle
  rather than a competitor, because it exists so verify.py can compare an
  accelerator row against its own cpu twin. Its number stays in the per-engine table,
  because a reader should be able to see it, and it is out of the frontier
  ranking and out of the headline ratio, because the GPU is the product and the
  cpu backend is no longer optimized. `verify.py`'s ORACLE CELL block is the
  rule; this file only renders it.
- **A SECONDS COLUMN THAT DOES NOT MEAN THE SAME THING ON EVERY ROW IS NEVER
  THE COLUMN A TABLE IS RANKED ON.** Added 2026-08-17 after this report ranked
  four engines on `train` while two of them bin inside their fit call and two
  do not, which put three phases against two and published it as an ordering.
  Every table that ranks now ranks on `fit s`, the whole fit, and prints
  `train s` beside it with a per-row word saying where that row's binning ran.
  Both figures are derived from the record's own phase keys and its
  `<phase>_unavailable_reason` fields, per row, so an engine that moves its
  binning moves this report by being re-run and not by being edited. See
  `FIT_PHASES`, `fit_breakdown` and `FRONTIER_RANK_FIELD`.

There is no summary line, no headline speedup, and no "x faster" anywhere
in this file. If a headline is wanted, a person writes it, having read the
distribution and named the conditions.

The one place this file names a single arm is the frontier block, and it is
the exception that proves the rule rather than a retreat from it. One tree
count makes "fastest" a well-defined question, so the answer is an ordering
this file can compute rather than a headline it would have to write. It is
still a DOCUMENTED recommendation and nothing applies it: no file in this
repository reads that name back into a default. See `_frontier` for the rules
it follows, and `bench/real_data/frontier.py` for what a one-axis sweep cannot
see.

SPEED AND ACCURACY ARE TWO AXES HERE AND NEITHER SUPPRESSES THE OTHER, from
2026-08-17. Until that date the frontier ranked only arms inside the accuracy
budget, which meant an arm outside it had no published speed at all, which
meant a 1.24x improvement on bit-identical work went unreported because the
arm was 1.7 percent behind CatBoost. Speed is now ranked for every arm in
every table. The constraint that replaced the suppression is the one rule this
file must not lose: EVERY SPEED FIGURE CARRIES ITS ACCURACY FIGURE IN THE SAME
ROW, and any ranking spanning arms of differing accuracy says so above itself
in `RANKING_CAVEAT`. A table of seconds with no metric column is a defect in
this file, not a simplification of it.
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engines  # noqa: E402
import envinfo  # noqa: E402
import quality  # noqa: E402
import verify  # noqa: E402


def load(target):
    if os.path.isdir(target):
        target = os.path.join(target, "records.json")
    with open(target) as handle:
        payload = json.load(handle)
    return payload.get("records", payload)


def thresholds():
    with open(os.path.join(HERE, "thresholds.json")) as handle:
        return json.load(handle)


def phase_value(record, name, field="elapsed_s"):
    block = (record.get("phases") or {}).get(name)
    if isinstance(block, dict) and field in block:
        return block[field]
    if isinstance(block, dict) and "measured" in block:
        values = [s[field] for s in block["measured"] if s.get(field) is not None]
        return statistics.median(values) if values else None
    return None


def _predict_speedup(record):
    """Realized parallel speedup of batch prediction: one thread over many.

    None, and therefore `n/a`, WHENEVER THE SINGLE-THREAD PHASE DID NOT PROVE
    IT GOT ONE THREAD. A ratio computed against a phase whose thread knob
    silently failed would read close to 1.0, which is the number that means
    "this arm does not scale" -- so the failure mode of the measurement and
    the finding it exists to report are the same value, and a reader could not
    tell them apart. `engines._single_thread_predict` writes the verdict on
    the row by checking the phase's own `parallel_efficiency`; this reads it
    rather than re-deriving it, so there is one definition of "verified".
    """
    phases = record.get("phases") or {}
    if phases.get("predict_batch_t1_verified") is not True:
        return None
    one = phase_value(record, "predict_batch_t1")
    many = phase_value(record, "predict_batch")
    if not one or not many:
        return None
    return one / many


#: The phases a FIT is made of, in the order a fit runs them. These and no
#: others are added up into the `fit s` column. `import` is deliberately not
#: one of them: a process loads a library once and fits many times, and it
#: has a column of its own already.
#:
#: **WHY THIS COLUMN EXISTS, and it is a reporting defect it was built to
#: end.** `train` does not mean the same thing on every row of these tables.
#: Two of the four engines expose binning as a step of their own and time it
#: separately, and two bin inside the fit call, so those two carry inside
#: their `train` figure the work their neighbours report outside it. A
#: ranking on `train` alone therefore compares three phases against two while
#: reading as though it compared like with like, and it flatters whichever
#: engines bin separately, which includes ours. On run
#: `20260817T195323Z-predict2`, medians of the repeats: our accelerator's
#: boosting rounds are 1.589 s against CatBoost's 1.777 s, and the two fits a
#: user actually waits on are 1.973 s against 1.840 s. The first ordering
#: says we win and the second says we lose by about 1.07x, and only the
#: second one is a comparison of the same quantity.
#:
#: `encode` and `ingest` are in the list because they are fit work that the
#: two engines which expose them do before boosting, and leaving them out
#: would rebuild the same asymmetry pointing the other way. They match
#: `scenarios.PHASE_SHAPE[engine]["e2e"]` for every engine in this harness.
FIT_PHASES = ("encode", "ingest", "binning", "train")

#: The suffix an adapter writes beside a null phase to say WHERE that phase's
#: cost went. Its presence is the record's own statement that the phase was
#: not skipped but folded into another one, and it is the field this file
#: reads to decide whether a fit total can be added up. See the `binning`
#: and `train` entries of `engines.py`'s module docstring, which say the
#: record "says so rather than leaving the two figures to be added".
PHASE_UNAVAILABLE_SUFFIX = "_unavailable_reason"


def fit_breakdown(record):
    """What one record's end-to-end fit is made of, read from the record.

    Returns `{"seconds", "parts", "folded", "undetermined"}`:

    - `seconds` is the sum of every fit phase this record measured, or None
      when the record cannot say what the fit cost.
    - `parts` is the list of `(phase, seconds)` that went into it.
    - `folded` is the list of `(phase, reason)` the record declares null WITH
      a reason. Those add nothing, and the reason is why: either the work
      happened inside a phase that IS counted, which is what CatBoost and
      XGBoost record for `binning`, or there was none of it to do, which is
      what every arm records for `encode` on a scenario with no categorical
      features. Both cases add zero seconds, so the total does not depend on
      telling them apart, and nothing here pretends to.
    - `undetermined` is every phase this file could not resolve either way.
      A non-empty list makes `seconds` None, because the alternative is
      guessing which side of the line a number falls on, and guessing is the
      defect this function replaces.

    **The three cases, and none of them is hardcoded per engine.** A phase
    present as a timed block is added. A phase present and null with a
    `<phase>_unavailable_reason` beside it is folded, on the record's own
    say-so. A phase present and null with NO reason is undetermined, and so
    is a missing or unreadable `train`. A phase key that is ABSENT
    ENTIRELY is a phase that engine does not have -- LightGBM and mojotrees
    write no `ingest` key at all -- and contributes nothing, which is not a
    guess about a number but a reading of which phases the adapter timed.
    """
    phases = record.get("phases") or {}
    parts, folded, undetermined = [], [], []
    for name in FIT_PHASES:
        block = phases.get(name)
        if isinstance(block, dict):
            value = phase_value(record, name)
            if value is None:
                undetermined.append(name)
            else:
                parts.append((name, value))
            continue
        reason = phases.get(name + PHASE_UNAVAILABLE_SUFFIX)
        if reason:
            folded.append((name, str(reason)))
        elif name in phases or name == "train":
            # An explicit null with nothing beside it, or a fit with no
            # boosting phase at all. Either way the record does not
            # distinguish, and the cell has to say so.
            undetermined.append(name)
    seconds = None if undetermined else sum(value for _, value in parts)
    return {
        "seconds": seconds,
        "parts": parts,
        "folded": folded,
        "undetermined": undetermined,
    }


def fit_seconds(record):
    """The end-to-end fit in seconds, or None when the record cannot say."""
    return fit_breakdown(record)["seconds"]


def fit_shape(record):
    """One line naming what this record's fit total contains, for a caption.

    Derived from the record every time rather than looked up by engine name,
    so an engine that changes where it bins changes this line by being
    re-run, with no edit here.
    """
    breakdown = fit_breakdown(record)
    if breakdown["undetermined"]:
        return (
            "not computable: this record does not say where "
            + " or ".join(f"`{name}`" for name in breakdown["undetermined"])
            + " went, so its fit total is left blank rather than guessed"
        )
    shape = " + ".join(name for name, _ in breakdown["parts"])
    for name, reason in breakdown["folded"]:
        # The recorded reason is quoted rather than summarized, and nothing
        # here asserts WHERE the phase went. Two different things produce a
        # null phase with a reason -- work that happened inside another timed
        # phase, and work there was none of to do -- and both add nothing to
        # the total, so the total is right either way and the prose does not
        # have to choose. `binning` is the one phase whose placement the
        # tables state, and `_binning_where` states it off the same field.
        shape += f"; no separate `{name}` phase, {_first_sentence(reason)}"
    return shape


def _first_sentence(text, limit=200):
    """The opening claim of a recorded reason, for a caption that has to fit
    on a line. The whole reason stays in the record, which is where a reader
    who wants the argument rather than the fact should be sent."""
    # Several recorded reasons open with the literal word `null`, which is
    # the cell's value and not its explanation, so it is dropped before the
    # first sentence is taken.
    body = str(text).strip()
    if body.lower().startswith("null"):
        body = body[4:].lstrip(". ")
    first = body.split(". ")[0].rstrip(".")
    if len(first) > limit:
        first = first[:limit].rstrip() + "..."
    return first


def cell_key(record):
    """The cell a record belongs to: scenario, TIER, DATA KIND, ARM, device,
    threads.

    The ARM field was the engine name until 2026-08-17, and on an
    `--arms` run that folded every arm of one engine into a single table
    row: forty frontier cells at different tree counts and bin counts
    rendered as one `mojotrees` row with `reps 40` and a median taken
    ACROSS ARMS, which is not a measurement of anything. `verify._arm_of`
    is the engine name on a run without `--arms`, so that shape of report
    renders exactly what it always did.

    **THE TIER AND THE DATA KIND JOINED IT LATER THE SAME DAY, and their
    absence was the same defect one level out.** A run may hold one scenario at
    two tiers, or its generator and its real dataset, in one records file:
    `frontier.py` schedules `dense_regression` as `dense_synthetic` and
    `dense_real`, and `pairs.py` schedules it at the standard tier, the large
    tier and on UCI YearPredictionMSD. With neither field in this key those
    cells shared one row, and the row printed a median across three different
    datasets over nine repeats as though it were a measurement of one thing.
    Every other keying function in the harness already carried both --
    `verify._oracle_cell_key`, `verify._budget_cell`, `verify._anchor_key` and
    `report._frontier_group` -- so this was the one reader that could average
    across them.

    It escaped notice because the two plans that span a scenario embed the row
    in the ARM ID, which made the arm field accidentally sufficient. That is a
    naming convention holding an identity together, which `LANE_RULES.md` rule 8
    says is exactly where to look. A default-matrix run resolves one tier and
    one variant, so those runs are keyed identically to before and their tables
    are unchanged.
    """
    data = record.get("data") or {}
    return (
        record["scenario"],
        record.get("tier"),
        data.get("data_kind"),
        verify._arm_of(record),
        record.get("device_used") or record.get("device_requested"),
        record["threads"],
    )


def summarise(values):
    """(median, min, max, spread) for a list of samples, or None."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    median = statistics.median(clean)
    lo, hi = min(clean), max(clean)
    spread = (hi - lo) / median if median else 0.0
    return {"median": median, "min": lo, "max": hi, "spread": spread, "n": len(clean)}


def fmt_time(summary, warn_spread):
    if not summary:
        return "n/a"
    mark = " !" if summary["spread"] > warn_spread else ""
    return f"{summary['median']:.3f} [{summary['min']:.3f}, {summary['max']:.3f}]{mark}"


def fmt_bytes(value):
    if value is None:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return str(value)


#: The columns, in order. The two parallel-efficiency columns are CPU
#: seconds over wall seconds, which is how many cores were busy on average.
#: They are here because a timing column without them is ambiguous between
#: a fast engine and a parallel one, and the ambiguity has a direction: a
#: cell at 1.00 next to a cell at 8.00 is a single-threaded measurement
#: being read as a slow one, which sends optimization work at the wrong
#: thing. Both phases carry one because they are separately parallel, and
#: an engine can be threaded in training and serial in prediction.
FIELDS = (
    # `fit s` FIRST, and ahead of `train s` on purpose. It is the only
    # seconds column in this table that means the same thing on every row:
    # everything a fit does, whichever phase each engine happens to do it in.
    # `train s` beside it is boosting rounds on the engines that bin
    # separately and boosting rounds PLUS binning on the engines that do not,
    # so two `train s` cells are not always the same quantity and the caption
    # under the table says which rows are which. A cell here reads `n/a` when
    # the record does not distinguish, never a partial sum.
    ("fit", "fit s", fit_seconds),
    ("train", "train s", lambda r: phase_value(r, "train")),
    ("cpu_ratio", "train par eff", lambda r: phase_value(r, "train", "parallel_efficiency")),
    ("binning", "bin s", lambda r: phase_value(r, "binning")),
    # TWO PREDICTION SECONDS COLUMNS, AND EACH NAMES ITS DEVICE. The single
    # column that preceded them was labeled `predict s` and was a CPU
    # prediction on EVERY row, including the rows whose `device` column said
    # `gpu`, because that column is the TRAINING device and the harness had
    # no way to ask for a device prediction. A number under `gpu` that a
    # reader had no reason to doubt was the accelerator's and was not: that
    # is the whole reason these columns are spelled this way. The label
    # carries the device, so the device column and the prediction column can
    # never disagree again.
    #
    # `predict cpu s` is the like-for-like comparison and is the one to read
    # against LightGBM, CatBoost and XGBoost, none of which can use this
    # accelerator. `predict gpu s` is OURS ONLY: a competitor row has no such
    # measurement and prints `n/a`, which `fmt_time` gives it for a missing
    # summary, and never a blank or a zero. A blank in a seconds column reads
    # as fast.
    #
    # The two are NOT interchangeable and the gpu one is not a candidate
    # default on its timing: see `engines.GPU_PREDICT_RULE`. GPU prediction
    # accumulates leaf values in Float32 where the host accumulates in
    # Float64, so it can change a user's output, and the cpu-versus-gpu
    # prediction crossover is unmeasured. Nothing here assumes the gpu column
    # is the smaller one; on the sizes this harness runs it may well not be,
    # and that is a publishable result rather than a problem.
    ("predict_batch", "predict cpu s", lambda r: phase_value(r, "predict_batch")),
    (
        "predict_batch_gpu",
        "predict gpu s",
        lambda r: phase_value(r, "predict_batch_gpu"),
    ),
    (
        "predict_batch_cpu_ratio",
        "predict cpu par eff",
        lambda r: phase_value(r, "predict_batch", "parallel_efficiency"),
    ),
    # THE SINGLE-THREAD PAIR, added 2026-08-17. `predict 1t s` is the same
    # model over the same held-out matrix with the library held to one thread,
    # and `predict speedup` is `predict 1t s` over `predict cpu s`: how much of
    # the machine the arm actually converted into wall clock.
    #
    # They are here because the seconds column alone cannot separate a slow
    # predictor from a badly scheduled one, and on run 20260817T195323Z-predict2
    # it did not: our leaf-wise arm spent 0.252 CPU seconds against LightGBM's
    # 0.409 for the same work and finished 1.6x later. Reading only the first
    # column sends optimization at the walk, which is already the cheaper of
    # the two. `engines.SINGLE_THREAD_PREDICT_RULE` holds the argument, and
    # says why this pair is a diagnostic and must never become the headline:
    # we win at one thread and lose on the machine, and the headline has to
    # keep showing the loss a user actually gets.
    (
        "predict_batch_t1",
        "predict 1t s",
        lambda r: phase_value(r, "predict_batch_t1"),
    ),
    ("predict_speedup", "predict speedup", lambda r: _predict_speedup(r)),
    ("predict_row", "row ms", lambda r: _ms(phase_value(r, "predict_row"))),
    ("warmup", "warmup s", lambda r: (r.get("warmup") or {}).get("elapsed_s")),
    ("import", "import s", lambda r: phase_value(r, "import")),
)


def _ms(value):
    return None if value is None else value * 1000.0


def role_of(record, accelerator_keys):
    """What this cell is FOR, as one word.

    `oracle` wins over everything else, because it is the label that changes
    how the row may be read. Otherwise this is `engines.ENGINE_ARM`, which
    already names the role of every engine in the comparison (subject,
    subject_variant, comparator, peer, peer_subject) and is the authority
    `selfcheck.py` asserts against. No new vocabulary is invented here. A
    reader who has seen one of those words anywhere else in the harness has
    already seen this column.
    """
    if verify.is_oracle(record, accelerator_keys):
        return verify.ORACLE_CELL_ROLE
    return engines.ENGINE_ARM.get(record.get("engine"), "unknown")


def build_cells(records, warn_spread):
    ok = [r for r in records if r.get("status") == "ok"]
    # Computed once over the whole file, because whether a cpu row is an
    # oracle is a property of the run and not of the row. Records written on
    # or after 2026-08-17 answer for themselves through `cell_role`; this is
    # what lets an older results file be read under the new labels too.
    accelerator_keys = verify.accelerator_cells(ok)

    cells = {}
    for record in ok:
        cells.setdefault(cell_key(record), []).append(record)

    out = {}
    for key, group in cells.items():
        summary = {
            name: summarise([getter(r) for r in group])
            for name, _, getter in FIELDS
        }
        summary["peak_rss"] = summarise([r.get("peak_rss_bytes") for r in group])
        first = group[0]
        out[key] = {
            "records": group,
            "engine": first["engine"],
            "repeats": len(group),
            "summary": summary,
            "role": role_of(first, accelerator_keys),
            "oracle": verify.is_oracle(first, accelerator_keys),
            "model_bytes": ((first.get("model") or {}).get("size") or {}).get("string_bytes"),
            "num_trees": (first.get("model") or {}).get("num_trees"),
            "quality": first.get("quality"),
            "primary_metric": first.get("primary_metric"),
            "data": first.get("data"),
            "caveats": first.get("caveats") or [],
            "transfers": first.get("transfers"),
            "env_key": envinfo.comparable_key(first.get("environment") or {}),
        }
    return out


def render(records, config, out):
    perf = config["performance"]["reporting"]
    warn_spread = perf["instability_warning_iqr_over_median"]
    min_repeats = perf["min_repeats_for_ratio"]
    cells = build_cells(records, warn_spread)

    groups = {}
    for key, cell in cells.items():
        groups.setdefault(json.dumps(cell["env_key"], sort_keys=True), {})[key] = cell

    if not cells:
        out("No completed runs in this results file.\n")
        return

    for env_json, group in groups.items():
        env_key = json.loads(env_json)
        out(f"## {env_key.get('cpu_model') or 'unknown cpu'} ({env_key.get('arch')})\n")
        out(
            f"mojo `{env_key.get('mojo')}`, lightgbm `{env_key.get('lightgbm')}`, "
            f"commit `{(env_key.get('git_commit') or '')[:12]}`\n"
        )
        state = ((next(iter(group.values()))["records"][0].get("environment") or {})
                 .get("machine_state") or {})
        if state.get("on_battery"):
            out("\nThis machine was on battery. Timings below are not comparable "
                "with mains-powered runs.\n")
        if state.get("thermal_pressure") and "Nominal" not in str(state["thermal_pressure"]):
            out(f"\nThermal state during the run: {state['thermal_pressure']}\n")

        # One SECTION per (scenario, tier, data kind), not per scenario, since
        # 2026-08-17 and for the reason `cell_key` records. A section prints its
        # row count, its feature count and its dataset name off ONE of its
        # cells, so a section spanning two tiers or a generator and a real
        # dataset describes itself with whichever cell sorted first and
        # misdescribes the rest. `_frontier_group` was already keyed this way;
        # this is the plain table catching up.
        for section in sorted({key[0:3] for key in group}, key=str):
            scenario, tier, kind = section
            rows = {k: v for k, v in group.items() if k[0:3] == section}
            first = next(iter(rows.values()))
            data = first["data"] or {}
            out(f"\n### {scenario} / {tier} / {kind}\n")
            out(
                f"{data.get('data_kind')} data, `{data.get('dataset')}`, "
                f"{(data.get('train') or {}).get('rows')} train rows x "
                f"{(data.get('train') or {}).get('features')} features, "
                f"primary metric {first['primary_metric']}\n"
            )
            # `role` is beside `device` and not at the end, because it
            # qualifies the device. A cpu row of one of our arms is a
            # different kind of thing depending on whether an accelerator row
            # sits beside it, and a reader scanning the two timing columns has
            # to meet that word before the numbers rather than after them.
            # The `arm` column appears only when a row's arm is not simply
            # its engine name, which is every `--arms` run and no other, so a
            # plain run's table is byte-for-byte what it was before the cell
            # key grew the arm dimension.
            show_arm = any(key[3] != cell["engine"] for key, cell in rows.items())
            out("\n| engine | " + ("arm | " if show_arm else "")
                + "device | role | threads | reps | " + " | ".join(
                label for _, label, _ in FIELDS
            ) + " | peak rss | model | metric |")
            out("| --- " * (8 + int(show_arm) + len(FIELDS)) + "|")
            for key in sorted(rows):
                cell = rows[key]
                _, _, _, arm, device, threads = key
                values = " | ".join(
                    fmt_time(cell["summary"][name], warn_spread) for name, _, _ in FIELDS
                )
                metric = cell["quality"].get(cell["primary_metric"]) if cell["quality"] else None
                peak = cell["summary"]["peak_rss"]
                shown = "n/a" if metric is None else f"{metric:.6g}"
                out(
                    f"| {cell['engine']} | " + (f"{arm} | " if show_arm else "")
                    + f"{device} | {cell['role']} | {threads} | "
                    f"{cell['repeats']} | "
                    f"{values} | {fmt_bytes(peak['median'] if peak else None)} | "
                    f"{fmt_bytes(cell['model_bytes'])} | {shown} |"
                )

            out("\nMedian across repeats, with [min, max]. A `!` marks a cell whose "
                f"spread exceeds {warn_spread:.0%} of its median.\n")
            _fit_caption(rows, out)
            _predict_device_caption(rows, out)
            _oracle_caption(rows, out)
            out(
                "The two `par eff` columns are CPU seconds over wall seconds "
                "for that phase, so they read as the average number of cores "
                "busy. A phase that ran on one core reports about 1.00 "
                "whatever the thread count column says, and two seconds "
                "columns are only comparable when the two par eff columns "
                "beside them are.\n"
            )
            _builders(rows, out)
            _bins(rows, out)
            _ratios(rows, min_repeats, out)

            caveats = sorted({c for cell in rows.values() for c in cell["caveats"]})
            if caveats:
                out("\nCaveats carried by these runs:\n")
                for caveat in caveats:
                    out(f"- {caveat}")
                out("")
            unavailable = {
                cell["transfers"].get("unavailable_reason")
                for cell in rows.values()
                if isinstance(cell.get("transfers"), dict)
                and cell["transfers"].get("unavailable_reason")
            }
            for reason in sorted(unavailable):
                out(f"\nHost-to-device transfer time was not measured: {reason}\n")


#: THE ASYMMETRY, in one string, printed under every table that has a `fit s`
#: column and above every ranking that could be read without one.
#:
#: It is a constant rather than three paragraphs written three times, for the
#: same reason `HEADLINE_LABEL` is: it is a claim about what a number means,
#: and a claim that is restated drifts.
FIT_COLUMN_CAVEAT = (
    "**`train s` is not the same quantity on every row, and `fit s` is.** "
    "An engine that exposes binning as a step of its own reports it in `bin "
    "s` and leaves it OUT of `train s`. An engine that bins inside its fit "
    "call has no `bin s` to report and its `train s` already contains that "
    "work. So comparing two `train s` cells across those two kinds of engine "
    "compares more phases against fewer, and it FLATTERS EVERY ENGINE THAT "
    "BINS SEPARATELY, WHICH INCLUDES OURS. `fit s` is every phase of the fit "
    "on every row, so it is the column two engines can be read against each "
    "other in. Neither column replaces the other: `train s` is the boosting "
    "rounds this repository's optimization work actually targets, and it is "
    "worth reading between two engines that split their phases the same way."
)


def _fit_caption(rows, out):
    """Say what `fit s` contains ON EACH ROW, read off the records.

    This is the part that has to survive a rerun. The asymmetry is not a fact
    about four engine names, it is a fact about which phases each adapter
    timed, and an adapter can move a phase. So the composition printed here
    is derived per row from the record's own phase keys and its
    `<phase>_unavailable_reason` fields, and an engine that starts or stops
    exposing a separate binning pass changes this caption by being re-run.

    A row whose record does not distinguish prints that it does not, in the
    place its number would have been.
    """
    shapes, separable, folded = {}, set(), set()
    for key, cell in sorted(rows.items()):
        name = f"{key[3]} on {key[4]}"
        for record in cell["records"]:
            shapes.setdefault(name, set()).add(fit_shape(record))
            breakdown = fit_breakdown(record)
            if "binning" in dict(breakdown["parts"]):
                separable.add(name)
            if "binning" in dict(breakdown["folded"]):
                folded.add(name)
    if not shapes:
        return
    out(FIT_COLUMN_CAVEAT + "\n")
    out("What `fit s` adds up, per row, read from the records:\n")
    for name in sorted(shapes):
        for shape in sorted(shapes[name]):
            out(f"- {name}: {shape}")
    out("")
    # A row can only be in one of the two lists, because a phase is either a
    # timed block or a null with a reason. A row in neither is one whose
    # record does not distinguish, and its line above already says so.
    separable, folded = sorted(separable - folded), sorted(folded - set(separable))
    if separable and folded:
        out(
            "So a `train s` comparison between "
            + ", ".join(f"`{name}`" for name in separable)
            + " and " + ", ".join(f"`{name}`" for name in folded)
            + " is NOT like for like, and the `fit s` column beside it is. A "
            "`train s` comparison WITHIN either of those two lists is like "
            "for like and is the sharper number of the two, because it holds "
            "the binning pass constant instead of adding it to both sides.\n"
        )


def _predict_device_caption(rows, out):
    """Say what the two prediction seconds columns are, under the table they
    are in, and say what this harness has NOT answered about them.

    Printed on every table that has them, unconditionally, because the
    failure being prevented is a reader assuming which device a prediction
    number came from and the assumption is just as available on a table with
    no accelerator row in it.

    The row-level cpu-versus-gpu gap is named here when a run recorded one,
    as a MAGNITUDE and never as a pass or a fail. There is no threshold on it
    anywhere and there should not be: the CPU backend is the correctness
    oracle available for a device path, not the definition of the right
    answer, and requiring the two to agree would forbid Float32 on the device
    and forbid GPU inference with it. What the number is for is stated in the
    caption, because a reader who meets a figure with no reason attached
    will supply one.
    """
    gaps = []
    for cell in rows.values():
        for record in cell["records"]:
            agreement = record.get("predict_device_agreement")
            if agreement and agreement.get("max_abs_diff") is not None:
                gaps.append(
                    (verify._arm_of(record), agreement["max_abs_diff"])
                )
    out(
        "`predict cpu s` and `predict gpu s` are the SAME fitted model "
        "scored on the two backends, timed separately. `predict cpu s` is "
        "the like-for-like column: it is what the competitor rows measure "
        "too, since none of the three can use this accelerator, and a "
        "competitor row reads `n/a` under `predict gpu s` because there is "
        "no such measurement to make, not because it is fast. The `device` "
        "column names the TRAINING device and says nothing about either "
        "prediction column; a `gpu`-trained row still predicts on the CPU "
        "in `predict cpu s`.\n"
    )
    if gaps:
        worst = max(gap for _, gap in gaps)
        arms = ", ".join(f"`{arm}`" for arm in sorted({arm for arm, _ in gaps}))
        out(
            "Largest row-level `|gpu - cpu|` prediction gap in this "
            f"section: {worst:.3g}, over {arms}. **This is a reported "
            "magnitude, not a pass or a fail.** The two are not expected to "
            "be bit-identical: the device walk accumulates leaf values in "
            "Float32 where the host accumulates in Float64, and both compare "
            "against the same Float64 edges, so every row reaches the same "
            "leaf and the accumulation is the only difference. A gap of "
            "roughly 1e-7 relative is exactly that accumulation and is not a "
            "defect. The number is carried anyway for two reasons. A real "
            "defect on this path, a wrong leaf index or a row read at the "
            "wrong offset, moves a prediction by 1e-1 or produces garbage, "
            "so a defect cannot hide inside an expected difference and the "
            "host is the only ground truth a device walk has. And a user who "
            "trains where there is an accelerator and scores where there is "
            "not is entitled to know the size of the difference rather than "
            "be told it does not exist.\n"
        )
    out(
        "The cpu-versus-gpu PREDICTION crossover is unmeasured, and nothing "
        "here assumes which way it goes. Launch and transfer are fixed costs "
        "that a batch has to be large enough to pay for, and prediction on "
        "these sizes is already fast on the host: an M4 scores 51,630 rows "
        "in about 0.008 s on the CPU with the symmetric arm. The "
        "accelerator losing at every size this harness runs is a legitimate "
        "result and would be worth reporting as one. No automatic device "
        "choice is wired for prediction, and `device=\"cpu\"` remains the "
        "prediction default everywhere: not because the CPU answer is the "
        "correct one, but because GPU prediction returns a slightly "
        "different number, so moving the default would change outputs that "
        "existing callers already depend on and is a person's decision "
        "rather than a benchmark's. When the crossover IS measured, record "
        "the run id beside it and replace this paragraph with it.\n"
    )


def _oracle_caption(rows, out):
    """Say what `oracle` in the role column means, under the table it means it
    in. Printed only when a row carries it, so a cpu-only run reads exactly as
    it always did."""
    oracles = sorted(
        {f"{key[3]} on {key[4]}" for key, cell in rows.items() if cell["oracle"]}
    )
    if not oracles:
        return
    out(
        "Rows marked `oracle` in the role column are "
        + ", ".join(f"`{name}`" for name in oracles)
        + ". An oracle row is one of our own arms on the cpu in a run that "
        "also ran that arm on the accelerator. It is measured and timed the "
        "same way every other row is and its number is printed here on "
        "purpose, because the cpu backend is real and a reader should be able "
        "to see it. It is NOT part of the speed story below. It is out of the "
        "frontier ranking and out of the headline ratio, because the GPU is "
        "the product and the cpu backend is kept as a correctness oracle and "
        "as the portability floor rather than optimized. It runs at "
        "`--oracle-repeats` repeats rather than `--repeats`, so its `reps` "
        "column is normally lower than the accelerator row beside it and its "
        "spread is normally narrower for that reason rather than because the "
        "machine was quieter.\n"
    )


def _builders(rows, out):
    """Which histogram construction each engine ran.

    Printed next to the timings rather than left in the records, because
    it is not a detail: row-wise and col-wise are different algorithms, and
    a training time is only reproducible if the reader knows which one
    produced it. A record that has the number and not the builder is not
    enough to repeat the run from.

    Every repeat is read rather than the first one, so a cell whose repeats
    did not all use the same builder prints two lines instead of hiding one
    of them behind the other.
    """
    lines = []
    for key in sorted(rows):
        for record in rows[key]["records"]:
            builder = record.get("histogram_builder")
            if not isinstance(builder, dict):
                continue
            resolved = builder.get("resolved")
            if isinstance(resolved, dict):
                shown = (
                    f"requested `{builder.get('requested')}`, resolved value "
                    f"not recoverable: {resolved.get('unavailable_reason')}"
                )
            else:
                shown = f"`{resolved}`"
            lines.append(f"- {key[3]} on {key[4]}: {shown}")
    if lines:
        out("\nHistogram construction:\n")
        for line in sorted(set(lines)):
            out(line)
        out("")


def _bins(rows, out):
    """The per-feature bin counts each engine ended up with.

    This is the empirical side of the binning alignment that scenarios.py
    argues for in prose. Equal totals do not prove the edges match, but
    unequal totals prove they do not, and that is the check this line
    exists to make cheap. Every repeat is read, so a binning that was not
    reproducible across repeats shows up as two lines for one engine.
    """
    lines = []
    for key in sorted(rows):
        for record in rows[key]["records"]:
            bins = (record.get("model") or {}).get("bins")
            if not isinstance(bins, dict):
                continue
            if "total" not in bins:
                lines.append(
                    f"- {key[3]} on {key[4]}: not read "
                    f"({bins.get('unavailable_reason')})"
                )
                continue
            lines.append(
                f"- {key[3]} on {key[4]}: {bins['total']} bins over "
                f"{bins['n_features']} features, per feature "
                f"min {bins['min']} / mean {bins['mean']:.1f} / "
                f"max {bins['max']}, vector sha256 {bins['sha256'][:12]}"
            )
    if lines:
        out("\nBins the two binnings produced:\n")
        for line in sorted(set(lines)):
            out(line)
        out("")


#: THE HEADLINE LABEL. One string, in one place, and it is the sentence in
#: this repository a reader could most easily be misled by, so it is written
#: out rather than assembled and it is not to be paraphrased by whoever next
#: edits the table around it.
#:
#: RE-POINTED 2026-08-17, and the meaning changed. This table used to be our
#: cpu against LightGBM's cpu, which was like-for-like on the backend and is
#: no longer the product. Andrew's ruling that day, in his words: "the entire
#: point is that WE USE THE GPU. We should be comparing us with gpu and
#: without gpu to catboost, and same for lightgbm." So the headline row is now
#: our accelerator, and the cpu row stays below it marked `oracle`.
#:
#: EVERY CLAUSE BELOW WAS CHECKED AGAINST THE CODE THAT MAKES IT TRUE rather
#: than copied from a summary, and the three refusals it names are the three
#: skip reasons this harness already prints, reused word for word in substance
#: so that a reader who sees both cannot find a difference between them:
#:
#:   LightGBM      run.py `_engine_skip_reason`, "LightGBM runs on the CPU in
#:                 this harness".
#:   CatBoost      run.py `_engine_skip_reason` and
#:                 `engines.CatBoostEngine.load`, "border_count is capped at
#:                 255 on GPU against 65535 on CPU".
#:   XGBoost       run.py `_engine_skip_reason` and
#:                 `engines.XGBoostEngine.load`, CUDA only, Apple silicon, a
#:                 CPU conda build, and a 3.4.0 fit handed device='cuda'
#:                 trains on the cpu with a warning rather than failing.
#:                 Verified 2026-08-17 and recorded in that docstring.
#:
#: What this label deliberately does NOT say is that we beat anybody's GPU.
#: Two of the three peers have a GPU trainer. Neither could be used here.
HEADLINE_LABEL = (
    "**The headline row is our accelerator against LightGBM's cpu, and that "
    "is not a like-for-like backend pairing.** It is each library's BEST "
    "AVAILABLE BACKEND ON THIS MACHINE, which is a different claim, and it is "
    "the claim this table makes. None of the three competitor arms in this "
    "harness can use this GPU. LightGBM runs on the cpu here. CatBoost's GPU "
    "training is a different quantization, border_count capped at 255 on GPU "
    "against 65535 on cpu, so a CatBoost GPU row would be a different "
    "measurement rather than a faster one, and CatBoostEngine.load refuses it "
    "by name. XGBoost's only accelerator backend is CUDA and this machine is "
    "Apple silicon, the installed package is the cpu build, and a fit handed "
    "device='cuda' on 3.4.0 trains on the cpu with a warning instead of "
    "failing, so XGBoostEngine.load refuses the device by name. Cpu is the "
    "ceiling for all three of them here. mojotrees is a GPU-first product and "
    "the accelerator is what it ships, so this is a product comparison. What "
    "it is NOT is a statement about anybody's GPU path, because two of the "
    "three peers have one and neither of those could be used on this machine."
)

#: The crossover, and why the oracle row is worth reading rather than skipping.
#:
#: These figures were handed to this lane on 2026-08-17 as medians of three
#: and were NOT taken by it, so they are recorded here as orientation and not
#: as a result of any run this file renders. At 200,000 rows by 50 features and
#: 100 trees: leaf-wise 1.046 s cpu against 1.043 s gpu, depth-wise 1.301 cpu
#: against 1.704 gpu, symmetric 1.768 cpu against 3.346 gpu. At 799,110 by 100:
#: leaf-wise 7.479 cpu against 3.659 gpu. So the accelerator only pays above a
#: few hundred thousand rows, the two leaf-wise arms are a tie at 200k, and the
#: other two arms are faster on the cpu there. A table that showed only the
#: accelerator row would hide that, which is one of the two reasons the oracle
#: row is printed rather than dropped.
CROSSOVER_NOTE = (
    "The `oracle` row below the headline is our own cpu backend against the "
    "same comparator. It is kept because it is a real number and a reader "
    "should be able to see it, and it is marked because it is not the "
    "headline. The cpu backend is maintained as a correctness oracle and as "
    "the portability floor and is no longer optimized. It is also where "
    "the crossover shows. Our accelerator does not pay at every size. Measured "
    "on 2026-08-17 at 200,000 rows by 50 features and 100 trees, the leaf-wise "
    "arm was a tie between the two backends and the depth-wise and symmetric "
    "arms were both FASTER on the cpu, while at 799,110 by 100 the leaf-wise "
    "arm was about two times faster on the accelerator. Those figures are "
    "orientation carried in this file's `CROSSOVER_NOTE`, not results of this "
    "run. Read both rows."
)


def _fit_comparability(pairs, out):
    """Say whether the `train` ratio beside the `fit` ratio is like for like,
    by CHECKING the two records rather than by asserting it.

    The mojotrees-against-LightGBM pairing this table renders is the one
    pairing in the harness where a train ratio is sound, because both engines
    expose binning as a separate step and both therefore exclude the same
    phase from `train`. That is a property of two adapters, not a law, and
    the sentence under this table used to be true only for as long as neither
    adapter moved its binning. Now the sentence is computed: if a future run
    ever pairs a row that bins separately with one that bins inside its fit,
    this prints a warning instead of the reassurance.
    """
    # AGREEMENT is the test, not separability. Two engines that both time
    # binning separately exclude the same phase from `train`; two that both
    # bin inside their fit call INCLUDE the same phase in it. Either way the
    # ratio is of one quantity. It is the mixture that is not.
    verdicts = set()
    for _threads, _device, mine, theirs in pairs:
        for cell in (mine, theirs):
            shapes = {
                "binning" in dict(fit_breakdown(r)["parts"])
                for r in cell["records"]
            }
            verdicts |= shapes
    if verdicts == {False}:
        out(
            "The `train` column of that table is like for like on this run, "
            "and it was checked rather than assumed: NEITHER engine in the "
            "pairing times a separate binning phase, so both `train` figures "
            "contain whatever binning each one does. It is a whole-fit "
            "comparison under a narrower name, and `fit (e2e)` beside it is "
            "the column to quote.\n"
        )
        return
    if verdicts == {True}:
        out(
            "The `train` column of that table IS like for like, and this run "
            "was checked rather than assumed: both engines in the pairing "
            "expose binning as a separate timed step, so both exclude the "
            "same phase from `train`. **That does not travel.** CatBoost and "
            "XGBoost bin inside their fit call, so their `train` figure "
            "contains a phase this one does not, and a ratio taken from this "
            "table must not be carried across to either of them. The `fit "
            "(e2e)` column is the one that compares against every engine in "
            "the harness.\n"
        )
        return
    out(
        "**The `train` column of that table is NOT like for like on this "
        "run.** The two engines in the pairing do not both expose binning as "
        "a separate timed step, so one side's `train` contains a phase the "
        "other side reports outside it, and the ratio compares more phases "
        "against fewer. Read the `fit (e2e)` column, which is every phase of "
        "the fit on both sides. The per-row composition is printed under the "
        "per-engine table above.\n"
    )


def _ratios(rows, min_repeats, out):
    """Our best available backend against LightGBM's, at a matched thread count.

    **Not matched on DEVICE, and that is the change.** Until 2026-08-17 this
    paired cpu with cpu and gpu with gpu, and since LightGBM never has a gpu
    row in this harness the second pairing never existed and the table was our
    cpu against theirs. That is a backend-matched comparison of a product whose
    backend is the accelerator, so it measured the thing we do not ship.

    Now the pairing is by thread count alone, our accelerator row is the
    headline, and our cpu row stays underneath marked `oracle`. `HEADLINE_LABEL`
    carries the argument for why an unmatched pairing is the honest comparison
    here rather than a flattering one, clause by clause with the code that
    makes each clause true.

    A run with no accelerator row falls back to the old shape exactly. The cpu
    row is then the measurement rather than an oracle, the pairing is
    like-for-like, and the label says so instead of claiming an accelerator
    that did not run.
    """
    # Matched on the two PLAIN arms, `mojotrees` and `lightgbm`, which is
    # what `cell_key`'s arm field yields on every run without `--arms`. On an
    # `--arms` run the frontier arms carry their own ids and never match
    # these two names, so this table does not render for them; the frontier
    # section below ranks those at a matched tree count, which is the only
    # ranking they can honestly appear in. Until 2026-08-17 the key here was
    # the engine and, on such a run, `comparator[threads]` was whichever
    # lightgbm arm was written last.
    comparator = {}
    for (scenario, _tier, _kind, arm, device, threads), cell in rows.items():
        if arm == "lightgbm" and device == "cpu":
            comparator[threads] = cell
    pairs = []
    for (scenario, _tier, _kind, arm, device, threads), cell in rows.items():
        if arm != "mojotrees":
            continue
        other = comparator.get(threads)
        if other:
            pairs.append((threads, device, cell, other))
    if not pairs:
        return

    # Oracle rows last inside a thread count, so the headline is the row a
    # reader's eye lands on first. `cell["oracle"]` sorts False before True.
    pairs.sort(key=lambda p: (p[0], p[2]["oracle"], p[1]))
    any_accelerator = any(not cell["oracle"] and device != "cpu"
                          for _t, device, cell, _o in pairs)

    # The column is `row` and not `role`, deliberately, even though one of its
    # two values also appears in the per-engine table's `role` column. `oracle`
    # means the same thing in both places. `HEADLINE` does not belong to that
    # vocabulary at all. It is a statement about which line of THIS table is
    # the published one, and putting it under `role` beside `comparator` and
    # `peer` would read as a fourth engine role, which it is not.
    # THE ACCURACY COLUMNS ARE NOT DECORATION, added 2026-08-17. This table
    # published three speed ratios and no accuracy figure anywhere near them,
    # which is the shape a speed number must never be quoted in: fewer trees,
    # fewer bins or a coarser learning rate buy any of these ratios outright.
    # A reader could take "0.72x train" out of here with nothing beside it. Now
    # both engines' primary metric sits in the same row as their ratio.
    # THE FIT RATIO LEADS AND THE TRAIN RATIO FOLLOWS, since 2026-08-17. The
    # train ratio in this particular table happens to be sound -- both sides
    # of THIS pairing expose binning separately, so both exclude the same
    # phase -- but it is sound by coincidence of which two engines are paired
    # here rather than by construction, and nothing in the table said which.
    # A reader who carried "0.63x train" from this table over to the CatBoost
    # or XGBoost row of the frontier would be carrying it into a comparison
    # where it does not hold, because those two bin inside their fit call.
    # `_fit_comparability` checks the pairing against the records instead of
    # asserting it in prose, and prints a warning under the table if a future
    # run ever pairs two rows that split their phases differently.
    out(
        "\n| threads | mojotrees on | row | lightgbm on | "
        "fit (e2e) mojotrees / lightgbm | train | binning | predict | "
        "metric | mojotrees | lightgbm | accuracy gap |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for threads, device, mine, theirs in pairs:
        cols = []
        for name in ("fit", "train", "binning", "predict_batch"):
            a, b = mine["summary"][name], theirs["summary"][name]
            if (
                a is None or b is None or not b["median"]
                or min(mine["repeats"], theirs["repeats"]) < min_repeats
            ):
                cols.append("n/a")
            else:
                cols.append(f"{a['median'] / b['median']:.2f}x")
        which = "oracle" if mine["oracle"] else "HEADLINE"
        metric = mine["primary_metric"]
        ours = (mine["quality"] or {}).get(metric)
        their_value = (theirs["quality"] or {}).get(metric)
        # The gap is computed through verify._worse_by so that its DIRECTION
        # comes from the metric rather than from this file. A hand-rolled
        # subtraction here would read backwards on every higher-is-better
        # metric and would do it silently.
        if (
            ours is None or their_value is None
            or theirs["primary_metric"] != metric
        ):
            gap = "n/a"
        else:
            worse = verify._worse_by(metric, ours, their_value, "relative")
            gap = (
                f"{abs(worse) * 100:.2f}% "
                + ("behind" if worse > 0 else "ahead")
            )
            # The excess lens beside the raw one, on the generator variant of
            # a scenario with a declared Bayes floor and nowhere else. See
            # verify.bayes_floor_of; the two cells share a data block by
            # construction of the pairing, so our record's floor is theirs.
            excess = verify.excess_worse_by(
                metric, ours, their_value,
                verify.bayes_floor_of(mine["records"][0]),
            )
            if excess is not None:
                gap += (
                    f", excess over floor {abs(excess) * 100:.1f}% "
                    + ("behind" if excess > 0 else "ahead")
                )
                root = verify.excess_root_worse_by(
                    metric, ours, their_value,
                    verify.bayes_floor_of(mine["records"][0]),
                )
                if root is not None:
                    gap += f" ({abs(root) * 100:.1f}% as excess rmse)"
        out(
            f"| {threads} | {device} | {which} | cpu | " + " | ".join(cols)
            + f" | {metric} | "
            + ("n/a" if ours is None else f"{ours:.6g}") + " | "
            + ("n/a" if their_value is None else f"{their_value:.6g}")
            + f" | {gap} |"
        )

    out("")
    _fit_comparability(pairs, out)
    out(
        "**Every speed ratio in that table carries the accuracy it was bought "
        "at, in the same row, and it is to be quoted that way or not at all.** "
        "A training time is purchasable with accuracy in either direction, so a "
        "ratio without its metric is not a result about either library. The "
        "`accuracy gap` column is this arm against LightGBM on the primary "
        "metric, relative, with the direction taken from the metric itself. "
        "Where it also reads `excess over floor`, the scenario's generator "
        "adds noise of a known scale, the raw metric is mostly that noise, "
        "and the second figure is the same gap taken on the error the model "
        "is responsible for (rmse squared minus the floor MSE), which is the "
        "figure that moves when a mechanism moves. Real-data rows have no "
        "known floor and show the raw gap alone.\n"
    )
    if any_accelerator:
        out(HEADLINE_LABEL + "\n")
        out(CROSSOVER_NOTE + "\n")
        out(
            "An `oracle` row often reads n/a across every column, and that is "
            f"by design rather than a fault. A ratio needs at least "
            f"{min_repeats} repeats on both sides and `run.py "
            "--oracle-repeats` defaults to 1, so the oracle cell usually has "
            "fewer. Its absolute timings are in the per-engine table above, "
            "which does not have that floor. Pass a higher --oracle-repeats "
            "to get the ratio as well, at the cost the argument's help text "
            "names.\n"
        )
    else:
        out(
            "**No accelerator row ran, so this table is cpu against cpu**, "
            "which is like-for-like on the backend. It is not the headline "
            "shape, because mojotrees is a GPU-first product and the "
            "comparison it publishes is our accelerator against each "
            "competitor's best available backend. Run with `--device gpu` on "
            "a machine that has one to get that table. The cpu row here is the "
            "measurement rather than an oracle, because it is the only backend "
            "that ran.\n"
        )
    out(
        f"Ratios are medians over at least {min_repeats} repeats, and read as "
        "mojotrees divided by lightgbm, so below 1.00 is mojotrees being "
        "quicker. `fit (e2e)` is the ratio of the two whole fits and is the "
        "one to quote; `train` is the diagnostic beside it, and whether it "
        "is like for like on this run is stated above the paragraph you are "
        "reading, checked against the records. They describe this machine on this day "
        "under the conditions named above. Nothing here is a claim about "
        "either library in general.\n"
    )


#: What a frontier row is ranked ON. END-TO-END FIT SECONDS, median over
#: repeats, since 2026-08-17. It was `train` until then, and the change is not
#: a preference between two reasonable columns. It is a correction.
#:
#: **The argument, and both candidates are legitimate, which is why it has to
#: be argued rather than picked.**
#:
#: Ranking on boosting rounds alone isolates the phase almost all of this
#: repository's optimization work targets, and inside one class of engine it
#: is the sharper of the two numbers, because it holds the binning pass
#: constant instead of adding the same seconds to both sides. That is a real
#: argument and the column stays in the table for it.
#:
#: It loses on one point, and the point is decisive: `train` IS NOT THE SAME
#: QUANTITY ON EVERY ROW OF THIS TABLE. LightGBM and mojotrees expose binning
#: as a separate timed step and exclude it from `train`; CatBoost and XGBoost
#: bin inside their fit call and cannot exclude it. So an ordering on `train`
#: puts three phases against two and presents the result as an ordering. A
#: table's ranking column is the one quantity every row in it has to share,
#: and `train` is not that quantity here while `fit` is.
#:
#: It also loses on the second point, which is who the table is for. A rank is
#: read as "which of these should I use", and what a user waits for is the
#: fit, not the subset of the fit this repository finds most interesting.
#: `CLAUDE.md` already says every published number is end to end, binning plus
#: training, against the comparator. This makes the frontier obey the rule the
#: rest of the harness was already written to.
#:
#: **What it costs, stated because it is a real cost.** A phase that is the
#: same work on every one of our arms now sits inside the ranked number and
#: dilutes the differences between them: our binning pass is about 0.38 s on
#: run 20260817T195323Z-predict2, which is 19 percent of the leaf-wise
#: accelerator arm's fit and moves none of the arms relative to each other.
#: That is why `train s` keeps its own column beside the rank rather than
#: being replaced by it. Read the rank to know which arm to run, and read
#: `train s` beside it to know whether a boosting change did anything.
FRONTIER_RANK_FIELD = "fit"

#: How a ranked value is read off a record, per rank field, so that changing
#: `FRONTIER_RANK_FIELD` is a one-line change and never a silent one. `fit`
#: is a sum this file computes; a phase name is a phase.
RANK_GETTERS = {
    "fit": fit_seconds,
    "train": lambda record: phase_value(record, "train"),
}

#: How the ranked column is labeled and named in prose, in one place, so that
#: a caption cannot say the table is ranked on something it is not.
RANK_LABELS = {
    "fit": ("fit s (e2e)", "median end-to-end fit seconds"),
    "train": ("train s", "median train seconds"),
}


def _binning_where(records):
    """Where a row's binning ran, as one of three words, read from the records.

    `separate` means the adapter timed a binning phase of its own, so this
    row's `train s` excludes it. `inside train` means the record declares the
    phase null WITH a reason, so this row's `train s` already contains it.
    `not stated` is everything else, including a set of repeats that disagree
    with each other, and it is printed rather than resolved: a row whose
    phases cannot be placed must not be silently placed on one side.
    """
    placements = set()
    for record in records:
        breakdown = fit_breakdown(record)
        if "binning" in dict(breakdown["parts"]):
            placements.add("separate")
        elif "binning" in dict(breakdown["folded"]):
            placements.add("inside train")
        else:
            placements.add("not stated")
    if len(placements) == 1:
        return placements.pop()
    return "not stated"


def _frontier_group(record):
    """The group a frontier row is ranked inside.

    **The tree count is in this key and that is the whole mechanism.** A
    ranking is built per group, so a group holding two tree counts cannot
    exist, so a 360-tree arm and a 100-tree competitor have no table to meet
    in. That constraint is carried by this tuple rather than by a caption
    under a table, because a caption does not stop anybody reading a row
    against the row above it.
    """
    data = record.get("data") or {}
    return (
        record["scenario"],
        record.get("tier"),
        data.get("data_kind"),
        record.get("device_used") or record.get("device_requested"),
        record["threads"],
        verify._tree_count(record),
    )


def _frontier_verdicts(records, config):
    """The two accuracy verdicts per row, from verify.py, as two maps.

    Recomputed here rather than read from a verdict file so that a report and
    a verdict cannot disagree: there is one implementation of each rule and
    this calls it.

    TWO MAPS SINCE 2026-08-17, because there are now two accuracy questions and
    they are keyed differently. The anchor gate is keyed by
    `verify._anchor_key`, which has no thread count in it; the peer scoreboard
    is keyed by scenario/arm/device/threads/trees. Collapsing them into one map
    would need one of the two keys to be wrong.
    """
    verdict = verify.Verdict()
    ok = [r for r in records if r.get("status") == "ok"]
    verify.check_accuracy_peer(ok, config, verdict)
    verify.check_accuracy_anchor(ok, config, verdict)
    peer, anchor = {}, {}
    for check in verdict.checks:
        if check["check"] == "accuracy_peer":
            peer[check["scope"]] = check
        elif check["check"] == "accuracy_anchor":
            anchor[check["scope"]] = check
    return peer, anchor


def _peer_cell(check, is_competitor):
    """The `vs best peer` cell. Never a reason to leave a row out of a rank."""
    if is_competitor:
        return "the bar"
    if check is None:
        return "no verdict"
    if check["status"] == verify.SKIP:
        return "not compared"
    worse = check.get("worse_relative")
    if worse is None:
        return "not compared"
    direction = "behind" if worse > 0 else "ahead of"
    text = f"{abs(worse) * 100:.2f}% {direction} {check.get('peer')}"
    if not check.get("inside_band"):
        text += ", outside 1%"
    # The excess lens, beside the raw one, only when verify.py found a Bayes
    # floor for this cell (generator variant of a scenario that declares one).
    # A cell without the field prints exactly what it printed before.
    excess = check.get("excess_worse_relative")
    if excess is not None:
        text += (
            f"; excess over floor {abs(excess) * 100:.1f}% "
            + ("behind" if excess > 0 else "ahead")
        )
        root = check.get("excess_root_worse_relative")
        if root is not None:
            text += f" ({abs(root) * 100:.1f}% as excess rmse)"
    return text


def _anchor_cell(check, is_competitor):
    """The `vs anchor` cell, which is the only accuracy GATE in the harness."""
    if is_competitor:
        return "not ours"
    if check is None:
        return "no verdict"
    if check["status"] == verify.SKIP:
        return "not compared"
    # A stale anchor reads differently from a missing one, because the two need
    # different actions: a missing anchor needs an adoption and a stale one
    # needs a RUN, and `NO ANCHOR` on a row that has one would send the reader
    # to the wrong remedy.
    if check.get("stale"):
        if check.get("stale_reason") == "unknown":
            return "ANCHOR CURRENCY UNKNOWN"
        parameter = check.get("stale_parameter")
        return f"STALE ANCHOR ({parameter})" if parameter else "STALE ANCHOR"
    if not check.get("anchored"):
        return "NO ANCHOR"
    worse = check.get("worse_relative") or 0.0
    where = "worse" if worse > 0 else "better"
    size = f"{abs(worse) * 100:.3f}% {where}"
    if check["status"] == verify.FAIL:
        return f"REGRESSION, {size}"
    if check["status"] == verify.WARN:
        return f"CHECK THIS, {size}"
    return f"held, {size}"


def _block_vocabulary():
    """Every `block` id any plan in this directory declares, with its meaning.

    The `block` column has been printed since the arm dimension landed and
    NOTHING has ever said what its values mean. A one-word label in a table with
    no legend is the same failure as a seconds column with no metric: the reader
    supplies a meaning and it is not necessarily the one the plan intended. This
    is why it matters here in particular: `pairs.py` uses `block` to carry which
    COMPARISON CLASS a row belongs to, and reading a Class A mirror row as a
    product row is the single most consequential misreading available in that
    table.

    Read from the plan modules rather than restated, so there is one definition
    per block. Both imports are cheap and pull in no engine library. A plan that
    grows a block gets a legend entry with no edit here.
    """
    vocabulary = {}
    for module_name in ("pairs", "frontier"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        for attribute in ("CLASSES", "BLOCKS"):
            vocabulary.update(getattr(module, attribute, {}) or {})
    vocabulary.setdefault(
        "competitor",
        "a competitor library's row, written by a plan that labels its peer "
        "rows this way rather than by class",
    )
    vocabulary.setdefault(
        "arm",
        "no block was declared for this row. It is one of our arms and nothing "
        "more specific is known about its role in the plan",
    )
    return vocabulary


def _block_legend(records, out):
    """Say what the `block` column means, above the tables that print it.

    Printed only for the blocks a run actually contains, and skipped entirely
    when every row is `arm` or `competitor`, so a default-matrix report reads
    exactly as it did.
    """
    present = {
        r.get("frontier_block") or r.get("arm_block")
        for r in records
        if r.get("status") == "ok"
    }
    present = {block for block in present if block}
    if not present:
        return
    vocabulary = _block_vocabulary()
    out("\n### What the `block` column means\n")
    out(
        "A `block` is the KIND of row, declared by the plan that scheduled it. "
        "Rows of different blocks answer different questions and are not "
        "interchangeable even when they sit in one table at one tree count. "
        "Read this before reading a rank.\n"
    )
    for block in sorted(present):
        meaning = vocabulary.get(
            block,
            "UNDECLARED. No plan module in bench/real_data declares this block, "
            "so nothing here can say what the row is for. Treat the rank as "
            "unattributed",
        )
        out(f"- **`{block}`** -- {meaning}\n")


def _pareto(rows, metric):
    """Which ranked rows are not dominated, by the definition Andrew named.

    An arm is dominated only if another arm is BOTH strictly faster AND at
    least as accurate. Nothing else counts, and in particular an arm is not
    dominated by something merely more accurate, because this is a speed table
    and a slower more accurate arm answers a different question.

    Two answers come out of this that a single ranking cannot hold at once:
    the fastest arm overall and the fastest arm that nothing beats on both
    axes. Both are true and they are frequently different rows.

    Rows with no speed or no metric are not judged and not used as
    dominators. A row that cannot be compared must not be able to eliminate
    one that can.
    """
    higher_better = quality.HIGHER_IS_BETTER.get(metric)
    out = {}
    usable = [
        row for row in rows
        if row["speed"] is not None and row["metric_value"] is not None
    ]
    for row in rows:
        if row not in usable or higher_better is None:
            out[row["arm"]] = None
            continue
        dominators = []
        for other in usable:
            if other is row or other["speed"] >= row["speed"]:
                continue
            at_least_as_accurate = (
                other["metric_value"] >= row["metric_value"] if higher_better
                else other["metric_value"] <= row["metric_value"]
            )
            if at_least_as_accurate:
                dominators.append(other["arm"])
        out[row["arm"]] = dominators
    return out


#: THE SENTENCE A READER MAY NOT MISS, printed above every frontier table.
#:
#: It is here rather than inline because it is the load-bearing caveat of the
#: whole section and it must not drift table by table. Registered 2026-08-17
#: with the separation of the two axes.
#:
#: The separation created a real hazard and this sentence is the mitigation.
#: Before it, an arm outside the accuracy budget was not ranked at all, so a
#: seconds column could not be read without its accuracy having already been
#: judged. Now every arm is ranked, which is right, and the cost is that a
#: reader can take a rank out of this table on its own. Speed is trivially
#: purchasable with accuracy -- fewer trees, fewer bins, a coarser learning
#: rate -- so a rank without the metric beside it is not a result.
RANKING_CAVEAT = (
    "**These rows are ranked on seconds alone and they do not all have the "
    "same accuracy.** Speed here is purchasable with accuracy: fewer trees, "
    "fewer bins or a coarser learning rate make any arm in this table faster. "
    "So a rank is a claim about seconds and about nothing else, and it is only "
    "readable together with the metric column beside it. Quote the two "
    "columns together or quote neither."
)

#: WHICH SECONDS THE RANK IS, printed above every frontier table beside
#: `RANKING_CAVEAT` and for the same reason: it is the sentence that decides
#: whether the ordering above it means anything, and it must not drift.
#:
#: Registered 2026-08-17, with the move of the rank from `train` to `fit`.
#: The argument for the move is on `FRONTIER_RANK_FIELD` and is not repeated
#: here; what a reader of the table needs is the consequence.
RANK_FIELD_CAVEAT = (
    "**The rank is END-TO-END FIT, and the `train s` column beside it is "
    "not rankable across these rows.** An engine that exposes binning as a "
    "step of its own keeps it out of `train s`; an engine that bins inside "
    "its fit call cannot, and its `train s` already contains that work. So "
    "RANKING ON BOOSTING ROUNDS ALONE FLATTERS EVERY ENGINE THAT BINS "
    "SEPARATELY, WHICH INCLUDES OURS, and it does so by a phase that is 19 "
    "percent of our own fit. `fit s (e2e)` is every phase of the fit on "
    "every row, which is what a user waits for and the one quantity these "
    "rows share. `train s` is kept beside it because it is the phase this "
    "repository's optimization work targets and it is the sharper number "
    "BETWEEN TWO ROWS THAT SPLIT THEIR PHASES THE SAME WAY. Which rows those "
    "are is printed under the per-engine table above, read off the records "
    "rather than assumed."
)


def _frontier(records, config, out):
    """The frontier block: every arm ranked by END-TO-END FIT SECONDS, with
    the boosting-rounds figure and both accuracy columns beside it, per tree
    count.

    THE RANK MOVED FROM `train` TO `fit` ON 2026-08-17 and the argument is on
    `FRONTIER_RANK_FIELD`. In one line: `train` is not the same quantity on
    every row of this table, because two of the four engines bin inside their
    fit call and two do not, so an ordering on it compared more phases against
    fewer and read as an ordering. Both figures are printed on every row and
    `RANK_FIELD_CAVEAT` sits above every table saying which is which.

    REBUILT 2026-08-17, AND THE MEANING CHANGED. Read this before reading a
    table from before that date against one from after, because the same
    section now answers a different question.

    **What it was.** "The accuracy budget frontier": only arms INSIDE the
    1-percent-of-CatBoost-or-LightGBM budget were ranked, everything else
    printed as `OUTSIDE, not ranked`. The budget did not label an arm, it
    suppressed it.

    **What broke.** On 2026-08-17 the leaf-wise GPU arm improved from 1.043 s
    to 0.839 s, a 1.24x from work that was bit-identical and could not have
    touched accuracy, and that improvement appeared NOWHERE in this section,
    because the arm sits outside the budget. We hid our own best result from
    ourselves. Andrew: "maybe we need to get rid of this linkage between speed
    and accuracy and just focus on them separately it is causing confusion and
    preventing us from enabling things."

    **The rule now, and it is the one rule this function exists to hold.**
    SPEED IS RANKED ALWAYS, FOR EVERY ARM. No arm is ever left out of the speed
    ranking for an accuracy reason. Accuracy is reported in its own columns
    beside the rank, never instead of it. A reader must be able to see in one
    row both that an arm is fastest and how accurate it is.

    **The constraint that survives, and it is not negotiable.** Speed is
    trivially purchasable with accuracy, so a speed figure with no accuracy
    figure beside it is not a result. Every row here carries its metric, both
    accuracy verdicts sit in the same row, and `RANKING_CAVEAT` says above
    every table that the ranking spans arms of differing accuracy. A table of
    seconds with no metric column would be a defect in this function.

    **Two accuracy columns, because there are two questions.** `vs anchor` is
    the GATE: our accuracy against our own recorded accuracy for this arm. `vs
    best peer` is the SCOREBOARD: how far from CatBoost or LightGBM, gating
    nothing. `verify.py`'s THE TWO ACCURACY AXES block is the rule; this file
    renders it.

    **Pareto, because "fastest" has two true answers.** An arm is dominated
    only if another arm is both strictly faster and at least as accurate. That
    surfaces "fastest overall" and "fastest that nothing beats on both axes" as
    two different rows, which is honest in a way a single ordering cannot be.

    **What did NOT change.**

    Ranked within one tree count only. `_frontier_group` puts the count in the
    key, so nothing here can order a 360-tree arm against a 100-tree one.

    ORACLE rows are printed and are not ranked. That exclusion is NOT an
    accuracy exclusion and it survives the separation untouched: a subject arm
    on the cpu beside an accelerator cell exists so `verify.py` can compare an
    accelerator row against its own cpu twin, the GPU is the product, and the
    cpu backend is no longer optimized. Its accuracy columns ARE filled in,
    which is new: the separation cuts both ways, and a row kept out of the
    speed story has no reason to be kept out of the accuracy one.

    COMPETITOR ROWS ARE NOW RANKED, which IS a change. They were `the bar` and
    unranked, and the reason given was that the budget is measured against them
    rather than applied to them. That is a statement about the ACCURACY axis
    and it was being used to remove them from the SPEED axis, which is the
    exact conflation this rebuild removes. Their accuracy cell still reads `the
    bar`. Their seconds are ranked with everybody else's, and the consequence
    is that "the fastest thing in this table" is frequently a competitor, which
    is true and was previously unsayable here.

    The DOCUMENTED recommendation still names one of our own arms and nothing
    applies it. No file in this repository reads that name back into a default.
    """
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return
    peer_verdicts, anchor_verdicts = _frontier_verdicts(records, config)
    accelerator_keys = verify.accelerator_cells(ok)

    groups = {}
    for record in ok:
        arm = verify._arm_of(record)
        group = _frontier_group(record)
        cell = groups.setdefault(group, {})
        cell.setdefault(arm, []).append(record)

    out("\n## The speed and accuracy frontier\n")
    out(
        "Called `the accuracy budget frontier` until 2026-08-17, when the two "
        "axes were separated. It ranked only arms inside the accuracy budget "
        "and printed the rest as `OUTSIDE, not ranked`, which hid a real 1.24x "
        "speed improvement on an arm whose accuracy the work could not have "
        "touched. Speed is now ranked for every arm and accuracy is reported "
        "beside it. Tables from before that date are not the same table.\n"
    )
    out(
        "One table per tree count, and that is the only grouping there is. A "
        "row is ranked against the rows beside it and against nothing else; "
        "there is no ordering in this section that spans two tree counts, "
        "and no arm appears in a table with a competitor it was not compared "
        "against.\n"
    )
    out(
        "Two accuracy columns, because there are two questions and one of them "
        "used to swallow the other. `vs anchor` is the GATE: this arm against "
        "OUR OWN recorded accuracy for it, from "
        "`bench/real_data/accuracy_anchors.json`. `vs best peer` is the "
        "SCOREBOARD: the distance to the better of CatBoost-as-shipped and "
        "LightGBM stock+det at this tree count, which gates nothing at all. An "
        "arm can be behind every peer and perfectly healthy, and it can be "
        "ahead of every peer and have just regressed against itself. On the "
        "generator variant of a scenario whose noise scale is known "
        "(`scenarios.bayes_floor`), the same cell also carries `excess over "
        "floor`: the gap on the error the model is responsible for, rmse "
        "squared minus the floor MSE, which is the number a mechanism moves. "
        "A 1.7 percent raw gap on dense_regression is a 28.8 percent excess "
        "gap. Real-data rows have no floor and show the raw gap alone.\n"
    )
    out(
        "One kind of row is printed and not ranked, and the reason is not "
        "accuracy. One of our own arms on the cpu, in a run that also ran it "
        "on the accelerator, reads `oracle` in the rank column, because the "
        "GPU is the product and the cpu backend is a correctness oracle and "
        "the portability floor rather than something we optimize. Its accuracy "
        "columns are filled in like everybody else's.\n"
    )
    _block_legend(records, out)

    recommendations = []
    for group in sorted(groups, key=str):
        scenario, tier, kind, device, threads, trees = group
        if trees is None:
            continue
        rows = []
        unjudged = []
        metric = None
        for arm, group_records in sorted(groups[group].items()):
            # The ranked column, through `RANK_GETTERS` so that what the rank
            # IS and what the caption SAYS it is cannot come apart.
            summary = summarise(
                [RANK_GETTERS[FRONTIER_RANK_FIELD](r) for r in group_records]
            )
            # Boosting rounds, beside the rank rather than as the rank, since
            # 2026-08-17. It is the phase this repository optimizes and it is
            # the sharper of the two numbers between rows that split their
            # phases the same way, and it is not rankable across rows that do
            # not: see `RANK_FIELD_CAVEAT`.
            train_summary = summarise(
                [phase_value(r, "train") for r in group_records]
            )
            # Inference, beside training, since 2026-08-17. Andrew asked for
            # it as its own column and the reason is that these two numbers
            # are bought at completely different rates: a model is TRAINED
            # once and PREDICTED with for as long as it is deployed, so the
            # column this table ranks on is the one that matters least in
            # production. That reason is unchanged by the rank moving to the
            # whole fit: a fit is still paid once and a prediction for as
            # long as the model is deployed.
            #
            # The measurement already existed in the phase table further up
            # and reached nobody who read only the frontier, which is how we
            # published a training win for weeks without noticing we were
            # last on inference against every competitor.
            #
            # TWO COLUMNS SINCE THE DEVICE REACHED PREDICTION, and for the
            # reason `FIELDS` states at length: this one used to be labeled
            # `predict s` and was a CPU number on every row including the
            # rows whose device column said `gpu`. The cpu figure keeps the
            # like-for-like comparison against the competitors and the gpu
            # one sits beside it, `n/a` on every row that has none.
            predict_summary = summarise(
                [phase_value(r, "predict_batch") for r in group_records]
            )
            predict_gpu_summary = summarise(
                [phase_value(r, "predict_batch_gpu") for r in group_records]
            )
            first = group_records[0]
            metric = metric or first.get("primary_metric")
            peer_scope = f"{scenario}/{arm}/{device}/t{threads}/n{trees}"
            peer_check = peer_verdicts.get(peer_scope)
            anchor_check = anchor_verdicts.get(verify._anchor_key(first))
            # `xgboost` joined the list on 2026-08-17 with the peer arm. It is
            # a competitor library, so without it an XGBoost row would be
            # labeled "arm" and then reported as having no accuracy verdict,
            # which is the treatment a mojotrees arm gets and is wrong here: a
            # competitor is not judged against our anchor and is not compared
            # to itself as a peer. `verify.DEFAULT_ACCURACY_PEER` decides which
            # competitors the scoreboard is taken from, and that is a separate
            # question from whether a row is a competitor at all.
            #
            # `arm_block` joined the chain on 2026-08-17 and it is a FIX, not
            # a widening. `frontier_block` is a field no writer in this
            # repository has ever produced; `worker.py` writes the arm's
            # declared block under the name `arm_block`. So a frontier plan
            # row that declared itself "Base A" was rendered "arm" here, and
            # the block column of a sweep table said nothing. The old name is
            # kept first so that a record which does carry it still wins.
            #
            # Whether a row is a competitor is a property of its ENGINE and is
            # read from the engine, since 2026-08-17. Until then it was read
            # from the ARM id against three literal names, which is the
            # opposite of the keying mistake fixed in verify.py the same day
            # and just as wrong: on an `--arms` run a lightgbm arm is named
            # `frontier.<row>.lightgbm.cpu.trees.100`, matches none of the
            # three, and only escaped being ranked as one of OUR arms because
            # `frontier.py` also happens to write `arm_block: competitor`.
            # `verify.SUBJECT_ENGINES` is every engine that is ours, so its
            # complement is every competitor, under any arm id.
            competitor_engine = first.get("engine") not in verify.SUBJECT_ENGINES
            block = first.get("frontier_block") or first.get("arm_block") or (
                "competitor" if competitor_engine else "arm"
            )
            is_competitor = block == "competitor" or competitor_engine
            if peer_check is not None and peer_check["status"] == verify.SKIP:
                unjudged.append((arm, "peer", peer_check["detail"]))
            if anchor_check is not None and anchor_check["status"] == verify.SKIP:
                unjudged.append((arm, "anchor", anchor_check["detail"]))
            rows.append({
                "arm": arm,
                "block": block,
                "summary": summary,
                "train_summary": train_summary,
                # What this row's fit total is made of, one entry per distinct
                # shape across its repeats. Carried on the row so the table
                # can say, per row, which cells its `train s` is comparable
                # with, without re-reading the records to find out.
                "fit_shapes": sorted({fit_shape(r) for r in group_records}),
                # WHERE THIS ROW'S BINNING IS, as one of three words, in the
                # row itself and not only in a caption. A caption does not
                # stop anybody reading one `train s` cell against the one
                # above it; a column in the same row does.
                "binning_where": _binning_where(group_records),
                "predict_summary": predict_summary,
                "predict_gpu_summary": predict_gpu_summary,
                "speed": summary["median"] if summary else None,
                "metric_value": (first.get("quality") or {}).get(
                    first.get("primary_metric")
                ),
                "oracle": verify.is_oracle(first, accelerator_keys),
                "competitor": is_competitor,
                "peer": peer_check,
                "anchor": anchor_check,
            })

        if not rows:
            continue

        # THE RANKING. Every row that is not an oracle, ordered on seconds and
        # on nothing else. No accuracy verdict is read anywhere in these three
        # lines, and that is the whole change.
        rankable = [r for r in rows if not r["oracle"] and r["speed"] is not None]
        rankable.sort(key=lambda r: r["speed"])
        for index, row in enumerate(rankable, start=1):
            row["rank"] = index
        for row in rows:
            row.setdefault("rank", "oracle" if row["oracle"] else "no timing")

        front = _pareto(rankable, metric)

        rank_column, rank_prose = RANK_LABELS[FRONTIER_RANK_FIELD]
        out(
            f"\n**{scenario} / {kind} / {device} / t{threads} / "
            f"{trees} trees**, ranked on {rank_prose}, primary "
            f"metric {metric}.\n"
        )
        out(RANKING_CAVEAT + "\n")
        out(RANK_FIELD_CAVEAT + "\n")
        out(
            f"| rank | arm | block | {rank_column} | train s | binning | "
            f"predict cpu s | predict gpu s | {metric} | vs anchor | "
            f"vs best peer | pareto |"
        )
        out("| --- " * 12 + "|")
        ordered = sorted(
            rows,
            key=lambda r: (
                r["rank"] if isinstance(r["rank"], int) else 10_000,
                r["arm"],
            ),
        )
        for row in ordered:
            value = row["metric_value"]
            dominators = front.get(row["arm"])
            # `n/a (oracle)` and not `not ranked`. The phrase "not ranked" was
            # the old accuracy suppression's label and it must not survive
            # anywhere in a table row, even attached to a different and
            # legitimate exclusion, because a reader who sees it in a row will
            # read it as the thing it used to mean.
            if row["oracle"]:
                pareto = "n/a (oracle)"
            elif row["speed"] is None:
                pareto = "n/a (no timing)"
            elif dominators is None:
                pareto = "n/a"
            elif dominators:
                pareto = f"dominated by {', '.join(sorted(set(dominators)))}"
            else:
                pareto = "frontier"
            out(
                f"| {row['rank']} | {row['arm']} | {row['block']} | "
                f"{fmt_time(row['summary'], 0.25)} | "
                f"{fmt_time(row['train_summary'], 0.25)} | "
                f"{row['binning_where']} | "
                f"{fmt_time(row['predict_summary'], 0.25)} | "
                f"{fmt_time(row['predict_gpu_summary'], 0.25)} | "
                f"{'n/a' if value is None else f'{value:.6g}'} | "
                f"{_anchor_cell(row['anchor'], row['competitor'])} | "
                f"{_peer_cell(row['peer'], row['competitor'])} | {pareto} |"
            )

        out(
            "\nThe two prediction columns are the same fitted model scored "
            "on the two backends. `predict cpu s` is the like-for-like "
            "figure and is what the competitor rows measure too, since none "
            "of the three can use this accelerator; a competitor row reads "
            "`n/a` under `predict gpu s` because there is no such "
            "measurement to make, not because it is fast. The ranking is on "
            f"{rank_prose} and reads NEITHER prediction column. The "
            "cpu-versus-gpu prediction crossover is unmeasured and the "
            "accelerator is not assumed to win at any size; no automatic "
            "device choice is wired for prediction, and `device=\"cpu\"` "
            "remains the prediction default because the gpu answer differs "
            "from it by roughly 1e-7 relative, so moving the default would "
            "change outputs existing callers depend on. That difference is "
            "recorded per run as a magnitude and gates nothing.\n"
        )

        # The composition of the ranked column, IN THIS SECTION and not only
        # under the per-engine table, because a reader who came for the
        # ranking never scrolls back up to the table that explains it. It is
        # the same per-record derivation, printed where the rank is.
        out(f"\nWhat `{rank_column}` adds up, per arm, read from the records:\n")
        for row in ordered:
            for shape in row["fit_shapes"]:
                out(f"- {row['arm']}: {shape}")
        out("")
        # TWO DIFFERENT SILENCES, and they had to be split because conflating
        # them prints a false sentence. A record can fail to place its
        # binning and still have a perfectly good fit total, which is what a
        # record with no binning phase at all looks like. Only a record whose
        # phases cannot be added has no rank.
        unplaced = [row for row in ordered if row["binning_where"] == "not stated"]
        if unplaced:
            out(
                "Rows whose records neither time a binning phase nor say "
                "where their binning ran: "
                + ", ".join(f"`{row['arm']}`" for row in sorted(
                    unplaced, key=lambda r: r["arm"]))
                + ". Their `" + rank_column + "` is the sum of the phases "
                "they did record and their rank is real; what cannot be said "
                "about them is which side of the `train s` split they belong "
                "on, so read the ranked column for them and not that one.\n"
            )
        unranked = [
            row for row in ordered
            if row["speed"] is None and not row["oracle"]
        ]
        if unranked:
            out(
                "Rows with NO RANK because their fit could not be added up: "
                + ", ".join(f"`{row['arm']}`" for row in sorted(
                    unranked, key=lambda r: r["arm"]))
                + ". A phase they declare null with no reason beside it, or a "
                "missing boosting phase, leaves the total undecidable, and a "
                "blank is the honest cell there. That is a gap in the adapter "
                "that wrote the record and is fixed there rather than here.\n"
            )

        has_oracle = any(row["oracle"] for row in rows)
        out(
            "\n`pareto` reads `frontier` when no arm in this table is both "
            "strictly faster and at least as accurate. A `dominated by` row is "
            "beaten on both axes at once and needs an argument other than "
            "speed to justify it."
            + (
                " Oracle rows are neither, because they are not in the "
                "ranking." if has_oracle else ""
            )
            + "\n"
        )
        if has_oracle:
            out(
                "`oracle` in the rank column is one of our own arms on the "
                "cpu, in a run that also ran that arm on the accelerator. It "
                "is timed and its number is above; it is out of the RANKING "
                "because the GPU is the product and the cpu backend is kept as "
                "a correctness oracle and as the portability floor rather than "
                "optimized. That is not an accuracy judgment, and since "
                "2026-08-17 its accuracy columns are filled in like every "
                "other row's. Its accelerator twin is ranked in the "
                "accelerator table for this same scenario and tree count.\n"
            )

        # THE THREE ANSWERS. They are frequently three different arms and every
        # one of them is true. Printing only one of them is how this section
        # came to hide a 1.24x.
        if rankable:
            fastest = rankable[0]
            shown = (
                "n/a" if fastest["metric_value"] is None
                else f"{fastest['metric_value']:.6g}"
            )
            # The competitor clause is conditional on there BEING a competitor
            # row in this group. A gpu table has none, because none of the
            # three peers can use this accelerator, and claiming the ranking
            # "includes the competitors" there would be a false reassurance
            # about the one comparison a reader most wants.
            has_competitor = any(r["competitor"] for r in rankable)
            # THE WORD "FASTEST" NAMES ITS QUANTITY HERE, since 2026-08-17.
            # This sentence used to read "fastest ... at 1.586 s" off a train
            # column that meant boosting rounds on some rows and boosting
            # rounds plus binning on others, which made it a claim nobody
            # could check against the row beside it. It now says end-to-end
            # fit, in the sentence, and carries the boosting figure after it
            # so the two are never separated.
            out(
                f"Fastest END TO END in this table at {trees} trees: "
                f"**{fastest['arm']}** at {fastest['speed']:.3f} s of "
                f"end-to-end fit ("
                + (
                    "boosting rounds "
                    + f"{fastest['train_summary']['median']:.3f} s, binning "
                    + fastest["binning_where"]
                    if fastest["train_summary"] else "no boosting figure"
                )
                + f"), {metric} {shown}."
                + (
                    " That includes the competitor rows, which are ranked on "
                    "speed here alongside our own arms."
                    if has_competitor else
                    " There is no competitor row in this table, so this is the "
                    "fastest of OUR arms and not the fastest arm measured on "
                    "this scenario. The competitors are in the cpu table for "
                    "the same scenario and tree count, because none of the "
                    "three can use this accelerator."
                )
                + "\n"
            )
        ours = [r for r in rankable if not r["competitor"]]
        if ours:
            best = ours[0]
            recommendations.append((group, best["arm"], best["summary"]))
            regressed = (
                best["anchor"] is not None
                and best["anchor"]["status"] in (verify.FAIL, verify.WARN)
                and best["anchor"].get("anchored")
            )
            line = (
                f"Fastest of OUR arms END TO END at {trees} trees: "
                f"**{best['arm']}** at {best['speed']:.3f} s of end-to-end "
                "fit ("
                + (
                    f"boosting rounds {best['train_summary']['median']:.3f} s, "
                    "binning " + best["binning_where"]
                    if best["train_summary"] else "no boosting figure"
                )
                + f"), {metric} "
                + ("n/a" if best["metric_value"] is None
                   else f"{best['metric_value']:.6g}")
                + f", {_peer_cell(best['peer'], False)}, anchor "
                f"{_anchor_cell(best['anchor'], False)}. That is a DOCUMENTED "
                "recommendation for the shipped defaults and nothing applies "
                "it: no file here reads this name back into a default, and it "
                "is the fastest AMONG THE ARMS THAT RAN rather than the "
                "fastest configuration, because this is a one-axis sweep and "
                "not a grid. See bench/real_data/frontier.py for what it "
                "cannot see."
            )
            if regressed:
                line += (
                    " **AND ITS ACCURACY GATE IS NOT CLEAN.** Read the anchor "
                    "column before taking this recommendation anywhere: a "
                    "fastest arm that just moved against its own recorded "
                    "accuracy is a trade somebody has to agree to, not a "
                    "result."
                )
            out(line + "\n")
        clean = [r for r in ours if not r["competitor"] and not front.get(r["arm"])]
        if clean:
            out(
                "On the Pareto frontier among our arms, meaning nothing in "
                "this table is both faster and at least as accurate: "
                + ", ".join(f"`{r['arm']}`" for r in clean)
                + ".\n"
            )
        for arm, which, why in unjudged:
            out(
                f"\nRANKED ON SPEED, NO {which.upper()} ACCURACY VERDICT: "
                f"`{arm}`. {why}. Its seconds are in the table above and its "
                "rank is real; what is missing is one of the two accuracy "
                "columns, and a missing accuracy verdict is not a reason to "
                "drop a measured time.\n"
            )

    if recommendations:
        out(
            "\nEvery recommendation above belongs to its own tree count and "
            "to no other. Two of them are not comparable with each other "
            "either, for the same reason the rows inside them are not.\n"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a records.json file or a run directory")
    parser.add_argument("--markdown", help="also write the report here")
    args = parser.parse_args(argv)

    records = load(args.results)
    config = thresholds()
    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    emit("# Real-data differential run\n")
    emit(f"Source: `{os.path.abspath(args.results)}`\n")
    render(records, config, emit)
    _frontier(records, config, emit)
    emit(
        "\nCorrectness is not decided here. Run verify.py against the same "
        "results file for that."
    )

    if args.markdown:
        with open(args.markdown, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
