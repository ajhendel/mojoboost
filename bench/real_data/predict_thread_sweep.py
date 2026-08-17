"""Why batch prediction converts so little of this machine into wall clock.

THE QUESTION THIS FILE EXISTS TO SETTLE. Run `20260817T195323Z-predict2`
reported batch prediction at 3.30, 3.31 and 3.31 EFFECTIVE CORES on three
mojotrees arms whose wall times span 8.5x, against 8.54 for LightGBM and 8.20
for CatBoost on the same 10-core M4. Effective cores is `cpu_s / elapsed_s`,
process CPU time over wall time, so it reads as how many cores were busy on
average across the call.

Three hypotheses produce that 3.30 and they imply different fixes, different
ceilings, and in one case no fix at all. Reading the code cannot separate them:
the Mojo standard library ships here compiled, with no source for
`sync_parallelize`, so how it maps tasks onto threads is not a question this
repository can answer by grep. It is an empirical question and this is the
experiment.

    H1  CORE HETEROGENEITY, STATIC SPLIT. `parallel._blocks_for` cuts equal
        ROW counts and this chip does not have equal cores: four performance
        and six efficiency. If an E core is about 3x slower on this walk, ten
        equal blocks finish in three units while the four fast cores finished
        in one and waited, so the ceiling is 10/3 = 3.33.

    H2  A WORKER POOL SMALLER THAN THE BLOCK COUNT. If the runtime's pool
        holds P threads and P < N, some thread runs `ceil(N / P)` blocks and
        wall time is that many block-times, so effective cores is
        `N / ceil(N / P)` with no reference to core speed at all. At N = 10
        and P = 4 that is also 10/3 = 3.33.

    H3  THE PREMISE IS AN ACCOUNTING ARTIFACT. `process_time` cannot tell a
        working thread from a spinning one. OpenMP's common default is to
        busy-wait at the end of a parallel region rather than sleep, and
        LightGBM is OpenMP. If its threads spin, four of them finishing early
        and spinning while six still work reports close to ten cores busy, and
        its 8.54 is imbalance wearing balance's clothes. Under H3 our
        scheduling may be no worse than theirs and the only solid fact is wall
        time, where they take 47.8 ms and we take 76.3 ms.

## How the three are separated

By SWEEPING THE BLOCK COUNT and looking at the shape of the curve, not at any
single point. All three agree at N = 10, which is why the harness run could not
tell them apart, and they disagree almost everywhere else.

- H1 with STATIC assignment and H2 both predict the sawtooth
  `N / ceil(N / P)`: rising inside a run of N, dropping the moment N crosses a
  multiple of P, and never exceeding P. They differ only in what P is, the
  count of FAST cores under H1 and the POOL SIZE under H2, so this experiment
  measures P without having to decide which of the two named it.
- H1 with DYNAMIC assignment predicts a curve that keeps CLIMBING with N well
  past P, because extra blocks are how a fast core takes more than its equal
  share. This is the outcome under which the fix is free, since auto mode
  already asks for `physical_cores * DEFAULT_TASKS_PER_CORE` blocks.
- A HARD PLATEAU that no block count lifts says the ceiling is the pool and
  not the split, and no amount of re-blocking in `parallel.mojo` will move it.

## How H3 is separated, which is the half that is about LightGBM

The `lightgbm` arm runs the same sweep through `num_threads`, in the same
process, interleaved with ours. The tell for spin is DIVERGENCE BETWEEN THE
TWO CLOCKS: a thread count past which wall time stops improving while `cpu_s`
keeps climbing roughly linearly is CPU being consumed without work being done,
which is the definition of a busy-wait. If LightGBM's effective cores tracks
the requested thread count all the way to 10 while its wall time flattens at 6,
its 8.54 was never a balance measurement and every "CPU seconds of work"
comparison against it, including the ones in this session, has to be withdrawn.

## What this file deliberately does not do

It does not fit on a pinned dataset and it takes no scenario. The question is
the response of a scheduler to a block count, and the tree shape a real
dataset would produce changes the per-row cost without changing that response.
Synthetic data keeps the probe runnable anywhere, with no pin, no digest and
no network. The default shape matches the run whose number prompted it,
51,630 rows scored against 100 trees over 90 features, so the effective-cores
figures here are directly comparable to that table.

It also measures PREDICTION ONLY, on an already fitted model. Nothing here
times a fit, and the fit that produces the model is outside every clock.

## Usage

    pixi run -e bench python bench/real_data/predict_thread_sweep.py
    pixi run -e bench python bench/real_data/predict_thread_sweep.py \
        --rows 51630 --features 90 --trees 100 --repeats 5

Interleaved, like every other comparison in this repository: within a repeat
each point runs once in a fixed order, so a thermal or power regime that moves
across the window moves across every point rather than under one of them.
See `bench/bench_train_gpu.mojo` on why an arm timing from a run containing
another arm is a derived quantity of that run, and `bench/results/
PROFILE_PROTOCOL.md` for the quiet-box precondition this inherits.
"""

import argparse
import json
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import envinfo  # noqa: E402
import measure  # noqa: E402


#: The block counts swept, and they are not evenly spaced on purpose. The
#: sawtooth `N / ceil(N / P)` is diagnostic exactly at the points where N
#: crosses a multiple of a plausible P, so the ladder is dense at 1 through 12
#: where P is likely to sit, and then jumps to 20 and 40 to ask whether MORE
#: blocks than cores buy anything. 40 is auto mode's own answer on this
#: machine, `physical_cores * apple_cpu_policy.DEFAULT_TASKS_PER_CORE`.
DEFAULT_POINTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 40)

#: Sentinel for the auto-mode point, where `MOJOTREES_NUM_WORKERS` is UNSET
#: rather than set to a number. It is a point of its own because it is the
#: SHIPPED path, and since 2026-08-17 it is also what the differential harness
#: schedules for a whole-machine cell (`run.mojotrees_workers`). A sweep that
#: only ever set the variable would never measure what a user gets.
AUTO = "auto"


def _fit_mojotrees(module, x, y, trees, seed):
    """One fitted model, outside every clock in this file."""
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": trees,
        "max_bin": 255,
        "seed": seed,
        "verbosity": -1,
    }
    dataset = module.Dataset(x, label=y, params=params)
    dataset.construct()
    return module.train(params, dataset, num_boost_round=trees)


def _fit_lightgbm(module, x, y, trees, seed, threads):
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "max_bin": 255,
        "seed": seed,
        "deterministic": True,
        "num_threads": threads,
        "verbosity": -1,
    }
    dataset = module.Dataset(x, label=y, params=params)
    dataset.construct()
    return module.train(params, dataset, num_boost_round=trees)


def _time_once(call):
    """One timed call, in the same two clocks the differential harness uses.

    `measure.Phase` and not a local pair of counters, so that an effective-cores
    figure printed here and one printed in `records.json` are the same
    quantity computed by the same code. A probe that measured the same thing a
    slightly different way would be answering a slightly different question.
    """
    _, phase = measure.timed(call)
    return phase.as_dict()


def _mojotrees_call(booster, x, workers):
    """The predict call for one point, with its block count already in place.

    The variable is moved around the call rather than inside it. `plan_tasks`
    reads the environment on every call by design, which its own docstring
    states, and `predict.predict_batch` reaches it through `dispatch_rows`
    rather than a `DispatchSettings` snapshot, so a point set here is a point
    the next call sees. IF THAT EVER STOPS BEING TRUE THIS SWEEP GOES FLAT,
    every point reporting the same effective cores, which is a shape no
    hypothesis above predicts and should be read as a broken probe rather than
    as a finding.
    """
    previous = os.environ.get("MOJOTREES_NUM_WORKERS")
    if workers == AUTO:
        os.environ.pop("MOJOTREES_NUM_WORKERS", None)
    else:
        os.environ["MOJOTREES_NUM_WORKERS"] = str(workers)
    try:
        return _time_once(lambda: booster.predict(x, device="cpu"))
    finally:
        if previous is None:
            os.environ.pop("MOJOTREES_NUM_WORKERS", None)
        else:
            os.environ["MOJOTREES_NUM_WORKERS"] = previous


def _lightgbm_call(booster, x, threads):
    """`num_threads` reaches `Booster.predict` through its `**kwargs`, which
    `basic.py` forwards verbatim as `pred_parameter`, so this is LightGBM's own
    prediction parameter rather than a pool resized behind its back."""
    if threads == AUTO:
        return _time_once(lambda: booster.predict(x))
    return _time_once(lambda: booster.predict(x, num_threads=threads))


def sweep(args):
    cpu = envinfo._cpu()
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.rows, args.features))
    y = (
        x[:, 0] * 2.0
        + x[:, 1] * x[:, 2]
        + rng.standard_normal(args.rows) * 0.1
    )
    x_test = np.ascontiguousarray(
        rng.standard_normal((args.test_rows, args.features))
    )

    import mojotrees

    arms = {}
    arms["mojotrees"] = (
        _fit_mojotrees(mojotrees, x, y, args.trees, args.seed),
        _mojotrees_call,
    )
    versions = {"mojotrees": getattr(mojotrees, "__version__", "unknown")}
    if not args.no_lightgbm:
        import lightgbm

        arms["lightgbm"] = (
            _fit_lightgbm(
                lightgbm, x, y, args.trees, args.seed, args.fit_threads
            ),
            _lightgbm_call,
        )
        versions["lightgbm"] = lightgbm.__version__

    points = list(args.points)
    if not args.no_auto:
        points.append(AUTO)

    # INTERLEAVED, and the loop order is what makes it so: repeat on the
    # outside, then arm, then point. Every point of every arm runs once before
    # any of them runs twice, so a window that drifts drifts across the whole
    # sweep. The samples are kept per point rather than reduced here, because
    # a median computed before the spread is known is how a drifting window
    # gets published as a measurement.
    samples = {(arm, point): [] for arm in arms for point in points}
    for _ in range(args.warmup):
        for arm, (booster, call) in arms.items():
            for point in points:
                call(booster, x_test, point)
    for _ in range(args.repeats):
        for arm, (booster, call) in arms.items():
            for point in points:
                samples[(arm, point)].append(call(booster, x_test, point))

    return {
        "machine": cpu,
        "versions": versions,
        "shape": {
            "train_rows": args.rows,
            "test_rows": args.test_rows,
            "features": args.features,
            "trees": args.trees,
        },
        "repeats": args.repeats,
        "warmup": args.warmup,
        "points": [str(p) for p in points],
        "arms": sorted(arms),
        "env": {
            name: os.environ.get(name)
            for name in (
                "MOJOTREES_NUM_WORKERS",
                "MOJOTREES_CPU_TASKS_PER_CORE",
                "MOJOTREES_CPU_CORE_POOL",
                "MOJOTREES_RAW_PREDICT",
                "OMP_NUM_THREADS",
                "OMP_WAIT_POLICY",
            )
        },
        "samples": {
            f"{arm}|{point}": rows for (arm, point), rows in samples.items()
        },
    }


def _median(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return statistics.median(values) if values else None


def _sawtooth(n, p):
    """`N / ceil(N / P)`, the effective cores a STATIC split of N blocks over
    P workers can reach. The shared prediction of H1-with-static-assignment
    and H2; they disagree about what P names, not about this shape."""
    return n / math.ceil(n / p)


def classify(result):
    """Which hypothesis the mojotrees curve matches, with every residual shown.

    A VERDICT WITH ITS ARITHMETIC BESIDE IT, not a label. The three curves are
    far apart at the points this ladder was chosen to include, so a fit that is
    close for one and not the others is worth stating; a fit that is close for
    two is worth stating as well, and this returns the residuals for every
    candidate P rather than only the winner so that case is visible instead of
    being resolved by whichever comparison ran last.
    """
    observed = {}
    for point in result["points"]:
        if point == AUTO:
            continue
        rows = result["samples"].get(f"mojotrees|{point}") or []
        eff = _median(rows, "parallel_efficiency")
        if eff is not None:
            observed[int(point)] = eff
    if len(observed) < 4:
        return {"verdict": "not enough points", "candidates": []}

    cores = result["machine"].get("physical_cores") or 1
    candidates = []
    for p in range(1, cores + 1):
        residual = max(
            abs(eff - _sawtooth(n, p)) for n, eff in observed.items()
        )
        candidates.append({"p": p, "max_abs_residual": residual})
    candidates.sort(key=lambda c: c["max_abs_residual"])
    best = candidates[0]

    # THE DISCRIMINATOR IS THE DIPS AND NOT THE HEIGHT. A static split of N
    # blocks over P workers is NON-MONOTONE by construction: the moment N
    # crosses a multiple of P some worker picks up an extra block and wall time
    # steps up, so effective cores falls back. `N/ceil(N/P)` at P=4 goes
    # 4.0 at N=4 and 2.5 at N=5. Dynamic assignment has no such cliff, because
    # the extra block goes to whichever worker is free. So a curve with
    # repeated drops is static whatever its height, and a curve with none is
    # not, and neither statement needs to know P first. Height alone was the
    # first version of this test and it was wrong: it compared against the P
    # the sawtooth fit had already chosen, so a genuinely dynamic curve that
    # fit best at a large P could never clear its own threshold.
    ladder = sorted(observed)
    dips = sum(
        1
        for a, b in zip(ladder, ladder[1:])
        if observed[b] < observed[a] - 0.25
    )
    ceiling = max(observed.values())
    floor = min(observed.values())
    climbs = dips == 0 and ceiling > floor * 1.25

    if ceiling <= floor * 1.15:
        # THE PROBE'S OWN FAILURE MODE, checked before any hypothesis is
        # fitted. Every point reporting the same effective cores is a shape
        # none of H1, H2 or H3 predicts, and the likeliest cause is that
        # `MOJOTREES_NUM_WORKERS` is no longer read per call, so every point
        # ran the same geometry. Reported as a broken instrument rather than
        # as a finding about scheduling. See `_mojotrees_call`.
        verdict = (
            f"FLAT, {floor:.2f} to {ceiling:.2f} across every block count. No "
            "hypothesis predicts this. The likeliest cause is that the block "
            "count is not reaching the predictor at all, which would make "
            "every point one measurement repeated; check that "
            "predict.predict_batch still dispatches through dispatch_rows "
            "rather than a DispatchSettings snapshot before reading anything "
            "into this"
        )
    elif climbs:
        verdict = (
            f"CLIMBS, NO DIPS, saturating near {ceiling:.1f} effective cores. "
            "Extra blocks are being converted into parallelism and no block "
            "count steps backwards, which a static equal split cannot do, so "
            "assignment is dynamic and the fix is block count. Auto mode "
            "already asks for physical_cores * DEFAULT_TASKS_PER_CORE blocks, "
            "so the harness change of 2026-08-17 is the whole fix"
        )
    elif dips >= 1 and best["max_abs_residual"] <= 0.35:
        verdict = (
            f"SAWTOOTH, P={best['p']}. Effective cores follows N/ceil(N/P) and "
            f"never exceeds {best['p']}, so the split is static and the "
            f"ceiling is {best['p']} workers, not {cores} cores. Whether P is "
            "the fast-core count (H1) or the runtime pool size (H2) is not "
            "decided by this fit; both predict this curve"
        )
    else:
        verdict = (
            f"NEITHER. {dips} drop(s) in the curve and the closest sawtooth is "
            f"P={best['p']} at {best['max_abs_residual']:.2f} max residual, "
            "which is too loose to claim. Read the table rather than this line"
        )
    return {
        "verdict": verdict,
        "best_p": best["p"],
        "climbs": climbs,
        "dips": dips,
        "ceiling": ceiling,
        "candidates": candidates[:4],
        "observed": observed,
    }


def spin_check(result):
    """Whether an arm's two clocks diverge, which is what a busy-wait looks
    like from outside.

    Wall time that stops improving while `cpu_s` keeps climbing is CPU being
    consumed without work being done. This reports the point past which wall
    time is within `--spin-tolerance` of its best, and what `cpu_s` did after
    it, for every arm including ours. It is REPORTED AND NOT JUDGED: a small
    divergence is ordinary thread-pool overhead, and the number that matters
    is whether it is small.
    """
    out = {}
    for arm in result["arms"]:
        curve = []
        for point in result["points"]:
            if point == AUTO:
                continue
            rows = result["samples"].get(f"{arm}|{point}") or []
            wall = _median(rows, "elapsed_s")
            cpu = _median(rows, "cpu_s")
            if wall is not None and cpu is not None:
                curve.append((int(point), wall, cpu))
        if len(curve) < 3:
            continue
        curve.sort()
        best_wall = min(w for _, w, _ in curve)
        knee = next(
            (n for n, w, _ in curve if w <= best_wall * 1.05), curve[-1][0]
        )
        after = [(n, c) for n, _, c in curve if n >= knee]
        out[arm] = {
            "wall_knee_at": knee,
            "best_wall_s": best_wall,
            "cpu_s_at_knee": after[0][1] if after else None,
            "cpu_s_at_max_point": after[-1][1] if after else None,
            "cpu_growth_after_knee": (
                None
                if not after or not after[0][1]
                else after[-1][1] / after[0][1]
            ),
        }
    return out


def render(result, out=print):
    machine = result["machine"]
    out("# Batch prediction against block count\n")
    out(
        f"{machine.get('model') or machine.get('arch')}, "
        f"{machine.get('physical_cores')} physical cores "
        f"({machine.get('performance_cores')} performance, "
        f"{machine.get('efficiency_cores')} efficiency)"
    )
    shape = result["shape"]
    out(
        f"\n{shape['test_rows']} held-out rows scored against "
        f"{shape['trees']} trees over {shape['features']} features, "
        f"{result['repeats']} interleaved repeats after "
        f"{result['warmup']} warmup passes.\n"
    )
    out("| arm | blocks | wall s | cpu s | effective cores | vs 1 block |")
    out("| --- | --- | --- | --- | --- | --- |")
    for arm in result["arms"]:
        base = _median(result["samples"].get(f"{arm}|1") or [], "elapsed_s")
        for point in result["points"]:
            rows = result["samples"].get(f"{arm}|{point}") or []
            wall = _median(rows, "elapsed_s")
            cpu = _median(rows, "cpu_s")
            eff = _median(rows, "parallel_efficiency")
            speedup = "n/a" if not (base and wall) else f"{base / wall:.2f}x"
            out(
                f"| {arm} | {point} | "
                + ("n/a" if wall is None else f"{wall:.4f}")
                + " | "
                + ("n/a" if cpu is None else f"{cpu:.4f}")
                + " | "
                + ("n/a" if eff is None else f"{eff:.2f}")
                + f" | {speedup} |"
            )
    verdict = classify(result)
    out(f"\n**Curve:** {verdict['verdict']}\n")
    if verdict.get("candidates"):
        out("Sawtooth fits, closest first (max absolute residual):\n")
        for candidate in verdict["candidates"]:
            out(f"- P={candidate['p']}: {candidate['max_abs_residual']:.3f}")
    out("\n**Two-clock divergence**, which is what a busy-wait looks like:\n")
    for arm, block in spin_check(result).items():
        growth = block["cpu_growth_after_knee"]
        out(
            f"- {arm}: wall stops improving at {block['wall_knee_at']} "
            f"({block['best_wall_s']:.4f} s), and cpu_s "
            + (
                "could not be compared past it"
                if growth is None
                else f"then grows {growth:.2f}x to the widest point"
            )
        )
    out(
        "\nA large growth figure beside a flat wall time is CPU spent without "
        "work done, and it means that arm's effective-cores number is not a "
        "balance measurement. See H3 in this file's docstring."
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=463715)
    parser.add_argument("--test-rows", type=int, default=51630)
    parser.add_argument("--features", type=int, default=90)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="whole-sweep passes discarded before measuring; one is enough to "
             "pay first-touch and any lazy pool creation on every point",
    )
    parser.add_argument("--seed", type=int, default=190019)
    parser.add_argument(
        "--fit-threads", type=int, default=0,
        help="threads for the LightGBM FIT, which is outside every clock "
             "here; 0 is its own default",
    )
    parser.add_argument(
        "--points", type=int, nargs="*", default=list(DEFAULT_POINTS),
        help="block counts to sweep",
    )
    parser.add_argument("--no-auto", action="store_true")
    parser.add_argument("--no-lightgbm", action="store_true")
    parser.add_argument("--json", default="", help="write the raw samples here")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = sweep(args)
    render(result)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"\nraw samples: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
