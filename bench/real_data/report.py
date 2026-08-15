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


FIELDS = (
    ("train", "train s", lambda r: phase_value(r, "train")),
    ("binning", "bin s", lambda r: phase_value(r, "binning")),
    ("predict_batch", "predict s", lambda r: phase_value(r, "predict_batch")),
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
        summary["cpu_ratio"] = summarise(
            [phase_value(r, "train", "parallel_efficiency") for r in group]
        )
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
