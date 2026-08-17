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
    ("train", "train s", lambda r: phase_value(r, "train")),
    ("cpu_ratio", "train par eff", lambda r: phase_value(r, "train", "parallel_efficiency")),
    ("binning", "bin s", lambda r: phase_value(r, "binning")),
    ("predict_batch", "predict s", lambda r: phase_value(r, "predict_batch")),
    (
        "predict_batch_cpu_ratio",
        "predict par eff",
        lambda r: phase_value(r, "predict_batch", "parallel_efficiency"),
    ),
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
    out(
        "\n| threads | mojotrees on | row | lightgbm on | "
        "train mojotrees / lightgbm | binning | predict | "
        "metric | mojotrees | lightgbm | accuracy gap |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for threads, device, mine, theirs in pairs:
        cols = []
        for name in ("train", "binning", "predict_batch"):
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
        "quicker. They describe this machine on this day under the conditions "
        "named above. Nothing here is a claim about either library in general.\n"
    )


#: What a frontier row is ranked ON. Training seconds, median over repeats.
#:
#: Not end to end: the import, the data build and the prediction phases are
#: the same work for every arm at one tree count, so including them would
#: dilute exactly the difference the frontier is looking for. They are already
#: in the tables above for anyone who wants them.
FRONTIER_RANK_FIELD = "train"


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


def _frontier(records, config, out):
    """The frontier block: every arm ranked by speed, with both accuracy
    columns beside it, per tree count.

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
            summary = summarise(
                [phase_value(r, FRONTIER_RANK_FIELD) for r in group_records]
            )
            # Inference, beside training, since 2026-08-17. Andrew asked for
            # it as its own column and the reason is that these two numbers
            # are bought at completely different rates: a model is TRAINED
            # once and PREDICTED with for as long as it is deployed, so the
            # column this table ranks on is the one that matters least in
            # production. Ranking is left on train seconds deliberately, so
            # that this is a fact placed beside the ranking rather than a
            # silent change to what the ranking means.
            #
            # The measurement already existed in the phase table further up
            # and reached nobody who read only the frontier, which is how we
            # published a training win for weeks without noticing we were
            # last on inference against every competitor.
            predict_summary = summarise(
                [phase_value(r, "predict_batch") for r in group_records]
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
                "predict_summary": predict_summary,
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

        out(
            f"\n**{scenario} / {kind} / {device} / t{threads} / "
            f"{trees} trees**, ranked on median train seconds, primary "
            f"metric {metric}.\n"
        )
        out(RANKING_CAVEAT + "\n")
        out(
            f"| rank | arm | block | train s | predict s | {metric} | "
            "vs anchor | vs best peer | pareto |"
        )
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
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
                f"{fmt_time(row['predict_summary'], 0.25)} | "
                f"{'n/a' if value is None else f'{value:.6g}'} | "
                f"{_anchor_cell(row['anchor'], row['competitor'])} | "
                f"{_peer_cell(row['peer'], row['competitor'])} | {pareto} |"
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
            out(
                f"Fastest in this table at {trees} trees: **{fastest['arm']}** "
                f"at {fastest['speed']:.3f} s, {metric} {shown}."
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
                f"Fastest of OUR arms at {trees} trees: **{best['arm']}** at "
                f"{best['speed']:.3f} s, {metric} "
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
