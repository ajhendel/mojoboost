"""Bias and variance of a stored prediction array against the noiseless signal.

    python bench/real_data/decompose.py results/<run_id> [--arm mojotrees]
        [--device gpu] [--rows-per-bin 45] [--json out.json]
    python bench/real_data/decompose.py results/<run_id> --signal signal.npy ...

Implements section 2 of docs/design/ACCURACY_GAP.md, which until 2026-08-17
existed only as arithmetic done by hand in one session. Nothing here trains,
downloads or builds anything; it reads a run directory's saved predictions and
regenerates the noiseless signal from the generator.

**The method.** With `e = prediction - signal` on the held-out rows,

- excess MSE is `mean(e**2)`, the error the model is responsible for once the
  irreducible noise is out of it (this is the document's excess, measured
  directly, and it differs from `rmse**2 - floor` by the cross term
  `2 * mean(e * noise)`, about a tenth of a percent);
- the SYSTEMATIC part on one feature is the variance of the conditional mean
  of `e` given that feature, estimated by binning the feature into about
  `rows_per_bin` rows per bin and taking the count-weighted mean of the squared
  bin means, MINUS the sampling floor `n_bins * var(e) / n`, because a
  conditional mean over many cells of a pure-noise residual is not zero and
  quoting it as bias would be wrong. Bins are equal-COUNT (quantile) bins, so
  the floor is exact per bin and no bin is empty; on a uniform feature, which
  is what `dense_regression` draws, they are also equal-width, so the numbers
  reproduce the document's;
- the total systematic part is the squared global offset `mean(e)**2` plus the
  per-feature parts computed on the CENTERED residual, so the offset is counted
  once rather than once per feature; a negative per-feature estimate (a
  feature at or below its floor) contributes zero;
- variance is the remainder, `excess - systematic`.

**Every systematic figure is a lower bound.** A conditional mean on one feature
cannot see bias that lives in an interaction of two, and the document says so
in the same words. What this tool can say is "at least this much of the excess
is bias, and here is which feature carries it".

**Resolution matters, and the first pass in the document got it wrong.** At 40
bins per feature the systematic share of our leaf-wise arm read 6.5 percent;
at 1000 bins, 38 percent. The bias on the step term `-2*(x4 > 0.7)` lives in a
window about 1/255 wide, and a coarse conditional mean averages the residual's
two opposite-signed spikes together and cancels most of it. So this tool
always computes a ladder of resolutions ending at the one requested and WARNS
when the estimate has not converged, in the terms `resolution_warning` spells
out. 40 to 50 rows per bin (1000 bins at the standard tier, 4000 at large) is
where the document found convergence.

**Regenerating the signal.** `generators.dense_regression_signal` is the
noiseless target as a function of the features, split out of the generator so
this tool can call it; the features come from the generator with the record's
own `generator_kwargs`, split with the record's own `split` block, and the
regenerated test set is checked against the record's canonical digest before
anything is computed, so a signal from the wrong data cannot be decomposed
against a prediction from the right one. Only generators listed in
`SIGNAL_FUNCTIONS` can be regenerated; for anything else, and for real data,
which has no known signal, pass `--signal file.npy` holding the noiseless
signal for the held-out rows in record order, or the tool refuses.

Nothing heavier than numpy is imported.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generators  # noqa: E402
import measure  # noqa: E402

#: How to regenerate the noiseless signal, per generator, from the held-out
#: feature matrix; and which columns carry signal, so the decomposition
#: conditions on those and reports the rest as a single "noise features" check.
SIGNAL_FUNCTIONS = {"dense_regression": generators.dense_regression_signal}
SIGNAL_COLUMNS = {"dense_regression": generators.DENSE_REGRESSION_SIGNAL_COLUMNS}

#: The document's rows-per-bin target: 1000 bins at 40,351 test rows and 4000
#: at 200,890, both about 40 to 50 rows per bin, and both converged.
DEFAULT_ROWS_PER_BIN = 45

#: The ladder of coarser resolutions computed beside the requested one, in
#: bins per feature. 40 and 200 are the document's two under-resolved passes.
LADDER_BINS = (40, 200)


def _bin_index(column, n_bins):
    """Equal-count bin of every row on `column`, 0..n_bins-1, by rank."""
    order = np.argsort(column, kind="stable")
    ranks = np.empty(len(column), dtype=np.int64)
    ranks[order] = np.arange(len(column))
    return (ranks * n_bins) // len(column)


def systematic_component(e, column, n_bins):
    """(raw, floor, net): the count-weighted mean of squared bin means of `e`
    conditioned on `column` at `n_bins` equal-count bins, the sampling floor
    `n_bins * var(e) / n`, and their difference clipped at zero."""
    n = len(e)
    bins = _bin_index(column, n_bins)
    counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
    sums = np.bincount(bins, weights=e, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    raw = float(np.sum(counts * means ** 2) / n)
    floor = float(n_bins * np.var(e) / n)
    return raw, floor, max(raw - floor, 0.0)


def decompose(predictions, signal, features, rows_per_bin=DEFAULT_ROWS_PER_BIN,
              feature_names=None, ladder=LADDER_BINS):
    """The decomposition, as a dict. `features` is the held-out feature matrix
    or the subset of signal-carrying columns; every column given is
    conditioned on. See the module docstring for the definitions."""
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    signal = np.asarray(signal, dtype=np.float64).ravel()
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    n = len(predictions)
    if len(signal) != n or features.shape[0] != n:
        raise ValueError(
            f"shape mismatch: {n} predictions, {len(signal)} signal rows, "
            f"{features.shape[0]} feature rows"
        )
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(features.shape[1])]
    n_bins = max(2, int(round(n / float(rows_per_bin))))

    e = predictions - signal
    excess = float(np.mean(e ** 2))
    offset = float(np.mean(e))
    centered = e - offset

    def at(bins):
        parts = []
        for j, name in enumerate(feature_names):
            raw, floor, net = systematic_component(centered, features[:, j], bins)
            parts.append({"feature": name, "raw": raw, "sampling_floor": floor,
                          "systematic": net})
        total = offset ** 2 + sum(p["systematic"] for p in parts)
        return {
            "n_bins": int(bins),
            "rows_per_bin": n / float(bins),
            "components": parts,
            "offset_squared": offset ** 2,
            "systematic": total,
            "systematic_share": (total / excess) if excess > 0 else None,
            "variance": excess - total,
        }

    resolutions = sorted({b for b in ladder if b < n_bins} | {n_bins})
    ladder_out = [at(b) for b in resolutions]
    final = ladder_out[-1]
    result = {
        "n_rows": int(n),
        "excess_mse": excess,
        "excess_rmse": excess ** 0.5,
        "mean_residual": offset,
        "requested_rows_per_bin": float(rows_per_bin),
        "n_bins": final["n_bins"],
        "systematic": final["systematic"],
        "systematic_share": final["systematic_share"],
        "variance": final["variance"],
        "components": final["components"],
        "resolution_ladder": [
            {"n_bins": r["n_bins"], "systematic": r["systematic"],
             "systematic_share": r["systematic_share"]}
            for r in ladder_out
        ],
        "resolution_warning": resolution_warning(ladder_out, n),
        "method": (
            "e = prediction - signal on the held-out rows; systematic per "
            "feature = count-weighted mean of squared conditional means of the "
            "centered residual at equal-count bins, minus n_bins*var(e)/n; total "
            "systematic = mean(e)**2 + sum over features; variance = excess - "
            "systematic. Systematic figures are lower bounds: a one-feature "
            "conditional mean cannot see interaction bias. ACCURACY_GAP.md s2."
        ),
    }
    return result


def resolution_warning(ladder_out, n):
    """The warning the document earned. Coarse bins UNDERSTATE bias: a wide
    conditional mean averages opposite-signed residual spikes together, so
    the 40-bin pass read 6.5 percent where 1000 bins read 38 percent. Warn
    when the coarsest resolution reads well below the finest (the estimate is
    still climbing, so the finest may itself be low), and when the finest is
    so fine that its sampling floor swamps the estimate."""
    if len(ladder_out) < 2:
        return (
            "only one resolution was computed; the document's finding is that "
            "a coarse bin count understates bias by up to a factor of six, so "
            "confirm convergence by rerunning at a finer --rows-per-bin"
        )
    coarse, prev, fine = ladder_out[0], ladder_out[-2], ladder_out[-1]
    # Thresholds are fractions of the EXCESS, not of the systematic estimate,
    # so a pure-noise residual whose systematic is a rounding error above zero
    # cannot trip them by ratio: 5 percent of the excess is well below the
    # 200-to-1000-bin move the document saw (22 percent of the excess) and
    # well above the 1000-to-4000 move it called converged (2.3 percent).
    excess = fine["systematic"] + fine["variance"]
    material = 0.05 * excess if excess > 0 else float("inf")
    notes = []
    if fine["systematic"] - coarse["systematic"] > material:
        notes.append(
            f"the systematic estimate rises from {coarse['systematic']:.6g} at "
            f"{coarse['n_bins']} bins to {fine['systematic']:.6g} at "
            f"{fine['n_bins']}: a coarse conditional mean understates bias "
            "(it averages opposite-signed residual spikes together, as on the "
            "x4 step term), so quote the finest resolution and treat every "
            "coarser figure as an artifact"
        )
    if abs(prev["systematic"] - fine["systematic"]) > material:
        notes.append(
            f"the two finest resolutions ({prev['n_bins']} and {fine['n_bins']} "
            f"bins) disagree by more than 5 percent of the excess "
            f"({prev['systematic']:.6g} against {fine['systematic']:.6g}); the "
            "estimate is NOT converged and the finest figure is a lower bound; "
            "rerun with a smaller --rows-per-bin if the rows allow it"
        )
    if fine["rows_per_bin"] < 20:
        notes.append(
            f"{fine['rows_per_bin']:.1f} rows per bin at the requested "
            "resolution; below about 20 the subtracted sampling floor is a "
            "large fraction of the raw estimate and the net figure is noisy. "
            "The document used 40 to 50"
        )
    if not notes:
        return None
    return "; ".join(notes)


def render(result, label=""):
    """A short table, as lines."""
    lines = []
    head = f"decomposition{(' of ' + label) if label else ''}: {result['n_rows']} rows"
    lines.append(head)
    lines.append(
        f"  excess MSE {result['excess_mse']:.6g} (excess RMSE "
        f"{result['excess_rmse']:.6g}), mean residual {result['mean_residual']:.3g}"
    )
    share = result["systematic_share"]
    lines.append(
        f"  systematic {result['systematic']:.6g}"
        + (f" ({share * 100:.1f}%)" if share is not None else "")
        + f", variance {result['variance']:.6g}, at {result['n_bins']} bins "
        f"per feature ({result['n_rows'] / result['n_bins']:.1f} rows per bin)"
    )
    lines.append("  | feature | raw | sampling floor | systematic |")
    lines.append("  | --- | --- | --- | --- |")
    for part in result["components"]:
        lines.append(
            f"  | {part['feature']} | {part['raw']:.6g} | "
            f"{part['sampling_floor']:.6g} | {part['systematic']:.6g} |"
        )
    lines.append("  resolution ladder (bins: systematic, share):")
    for rung in result["resolution_ladder"]:
        share = rung["systematic_share"]
        lines.append(
            f"    {rung['n_bins']}: {rung['systematic']:.6g}"
            + (f", {share * 100:.1f}%" if share is not None else "")
        )
    if result["resolution_warning"]:
        lines.append(f"  WARNING: {result['resolution_warning']}")
    return lines


# --- the run-directory CLI --------------------------------------------------


def load_records(run_dir):
    with open(os.path.join(run_dir, "records.json")) as handle:
        payload = json.load(handle)
    return payload.get("records", payload)


def load_predictions(run_dir, record):
    """The stored predictions for a record, checked against the record's own
    digest. Same location rule as `verify._load_predictions`."""
    name = record.get("predictions_path")
    if not name:
        return None
    path = os.path.join(run_dir, "predictions", name)
    if not os.path.exists(path):
        return None
    predictions = np.load(path)
    want = record.get("predictions_sha256")
    if want and measure.digest(predictions) != want:
        raise RuntimeError(
            f"{path} does not match the record's predictions_sha256; refusing "
            "to decompose predictions that are not the ones the record scored"
        )
    return predictions


_DATA_CACHE = {}


def regenerate_test_set(record):
    """(X_test, y_test, signal_test) regenerated from the record's generator
    kwargs and split, digest-checked against the record's `data.test.digest`.
    Raises when the record is not from a generator this tool knows."""
    data = record.get("data") or {}
    generator = data.get("generator")
    if data.get("data_kind") != "synthetic" or generator not in SIGNAL_FUNCTIONS:
        raise RuntimeError(
            f"cannot regenerate the signal for data_kind={data.get('data_kind')!r} "
            f"generator={generator!r}: only {sorted(SIGNAL_FUNCTIONS)} on the "
            "synthetic variant have a known noiseless signal. Pass --signal "
            "with a .npy of the held-out signal in record order"
        )
    kwargs = data.get("generator_kwargs") or {}
    split = data.get("split") or {}
    key = (generator, json.dumps(kwargs, sort_keys=True), json.dumps(split, sort_keys=True))
    if key not in _DATA_CACHE:
        full = generators.GENERATORS[generator](**kwargs)
        _, test = generators.split(
            full, split.get("train_fraction", 0.8), split.get("seed", 1900)
        )
        want = ((data.get("test") or {}).get("digest"))
        if want:
            got = measure.canonical_digest(test["X"], test["y"], test.get("group"))
            if got != want:
                raise RuntimeError(
                    "the regenerated held-out set does not match the record's "
                    f"data.test.digest ({got[:12]} against {want[:12]}); the "
                    "generator or the split has changed since this run and its "
                    "signal cannot be reconstructed"
                )
        signal = SIGNAL_FUNCTIONS[generator](test["X"])
        _DATA_CACHE[key] = (test["X"], test["y"], signal)
    return _DATA_CACHE[key]


def _select(records, args):
    chosen = []
    seen = set()
    for record in records:
        if record.get("status") != "ok" or not record.get("predictions_path"):
            continue
        arm = record.get("arm") or record.get("engine")
        device = record.get("device_used") or record.get("device_requested")
        if args.arm and arm not in args.arm:
            continue
        if args.device and device != args.device:
            continue
        if args.scenario and record.get("scenario") != args.scenario:
            continue
        # One record per cell: repeats of a deterministic trainer decompose
        # identically, and --all-repeats says otherwise.
        key = (record.get("scenario"), arm, device, record.get("threads"))
        if key in seen and not args.all_repeats:
            continue
        seen.add(key)
        chosen.append(record)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--arm", action="append", help="arm id(s) to decompose; default every arm")
    parser.add_argument("--device", help="cpu or gpu; default both")
    parser.add_argument("--scenario", help="default every scenario with a known signal")
    parser.add_argument("--rows-per-bin", type=float, default=DEFAULT_ROWS_PER_BIN)
    parser.add_argument("--all-repeats", action="store_true")
    parser.add_argument(
        "--signal", help=".npy of the noiseless held-out signal in record order, "
        "for a run whose generator this tool cannot regenerate",
    )
    parser.add_argument(
        "--all-features", action="store_true",
        help="condition on every feature, not only the signal-carrying columns",
    )
    parser.add_argument("--json", dest="json_out", help="write every result to this file")
    args = parser.parse_args(argv)

    records = load_records(args.run_dir)
    chosen = _select(records, args)
    if not chosen:
        print("no ok record with saved predictions matched", file=sys.stderr)
        return 2
    signal_override = np.load(args.signal) if args.signal else None
    results = []
    for record in chosen:
        arm = record.get("arm") or record.get("engine")
        device = record.get("device_used") or record.get("device_requested")
        label = f"{record.get('scenario')}/{arm}/{device}/t{record.get('threads')}"
        predictions = load_predictions(args.run_dir, record)
        if predictions is None:
            print(f"{label}: no predictions on disk, skipped", file=sys.stderr)
            continue
        data = record.get("data") or {}
        generator = data.get("generator")
        if signal_override is not None:
            x_test, _y, signal = None, None, signal_override
        else:
            try:
                x_test, _y, signal = regenerate_test_set(record)
            except RuntimeError as exc:
                print(f"{label}: {exc}", file=sys.stderr)
                continue
        if x_test is None:
            print(
                f"{label}: --signal given without regenerated features; the "
                "decomposition needs the feature matrix to condition on, so "
                "only the excess is reported",
                file=sys.stderr,
            )
            e = np.asarray(predictions, dtype=np.float64).ravel() - signal
            result = {"n_rows": int(len(e)), "excess_mse": float(np.mean(e ** 2)),
                      "excess_rmse": float(np.mean(e ** 2)) ** 0.5,
                      "mean_residual": float(np.mean(e))}
        else:
            columns = (
                list(range(x_test.shape[1])) if args.all_features
                else list(SIGNAL_COLUMNS.get(generator, range(x_test.shape[1])))
            )
            result = decompose(
                predictions, signal, x_test[:, columns], args.rows_per_bin,
                feature_names=[f"x{j}" for j in columns],
            )
            for line in render(result, label):
                print(line)
        result["cell"] = {
            "scenario": record.get("scenario"), "arm": arm, "device": device,
            "threads": record.get("threads"), "repeat": record.get("repeat"),
            "predictions_path": record.get("predictions_path"),
            "rmse": (record.get("quality") or {}).get("rmse"),
        }
        results.append(result)
        print()
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
