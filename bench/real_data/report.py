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

There is no summary line, no headline speedup, and no "x faster" anywhere
in this file. If a headline is wanted, a person writes it, having read the
distribution and named the conditions.
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import envinfo  # noqa: E402


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
    return (
        record["scenario"],
        record["engine"],
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


def build_cells(records, warn_spread):
    cells = {}
    for record in records:
        if record.get("status") != "ok":
            continue
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
            "repeats": len(group),
            "summary": summary,
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

        for scenario in sorted({key[0] for key in group}):
            rows = {k: v for k, v in group.items() if k[0] == scenario}
            first = next(iter(rows.values()))
            data = first["data"] or {}
            out(f"\n### {scenario}\n")
            out(
                f"{data.get('data_kind')} data, `{data.get('dataset')}`, "
                f"{(data.get('train') or {}).get('rows')} train rows x "
                f"{(data.get('train') or {}).get('features')} features, "
                f"primary metric {first['primary_metric']}\n"
            )
            out("\n| engine | device | threads | reps | " + " | ".join(
                label for _, label, _ in FIELDS
            ) + " | peak rss | model | metric |")
            out("| --- " * (7 + len(FIELDS)) + "|")
            for key in sorted(rows):
                cell = rows[key]
                _, engine, device, threads = key
                values = " | ".join(
                    fmt_time(cell["summary"][name], warn_spread) for name, _, _ in FIELDS
                )
                metric = cell["quality"].get(cell["primary_metric"]) if cell["quality"] else None
                peak = cell["summary"]["peak_rss"]
                shown = "n/a" if metric is None else f"{metric:.6g}"
                out(
                    f"| {engine} | {device} | {threads} | {cell['repeats']} | "
                    f"{values} | {fmt_bytes(peak['median'] if peak else None)} | "
                    f"{fmt_bytes(cell['model_bytes'])} | {shown} |"
                )

            out("\nMedian across repeats, with [min, max]. A `!` marks a cell whose "
                f"spread exceeds {warn_spread:.0%} of its median.\n")
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
            lines.append(f"- {key[1]} on {key[2]}: {shown}")
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
                    f"- {key[1]} on {key[2]}: not read "
                    f"({bins.get('unavailable_reason')})"
                )
                continue
            lines.append(
                f"- {key[1]} on {key[2]}: {bins['total']} bins over "
                f"{bins['n_features']} features, per feature "
                f"min {bins['min']} / mean {bins['mean']:.1f} / "
                f"max {bins['max']}, vector sha256 {bins['sha256'][:12]}"
            )
    if lines:
        out("\nBins the two binnings produced:\n")
        for line in sorted(set(lines)):
            out(line)
        out("")


def _ratios(rows, min_repeats, out):
    """mojotrees against lightgbm at matched device and thread count."""
    pairs = []
    for key, cell in rows.items():
        scenario, engine, device, threads = key
        if engine != "mojotrees":
            continue
        other = rows.get((scenario, "lightgbm", device, threads))
        if other:
            pairs.append((device, threads, cell, other))
    if not pairs:
        return
    out("\n| device | threads | train mojotrees / lightgbm | binning | predict |")
    out("| --- | --- | --- | --- | --- |")
    for device, threads, mine, theirs in sorted(pairs, key=lambda p: (p[0], p[1])):
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
        out(f"| {device} | {threads} | " + " | ".join(cols) + " |")
    out(
        f"\nRatios are medians over at least {min_repeats} repeats, and read as "
        "mojotrees divided by lightgbm, so below 1.00 is mojotrees being "
        "quicker. They describe this machine on this day under the conditions "
        "named above. Nothing here is a claim about either library in general.\n"
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
