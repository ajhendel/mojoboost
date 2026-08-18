# Running someone else's benchmark harness

Written 2026-08-18.

Every performance number this project has published came from a harness this
project wrote. That is the weakest position a benchmark claim can be in,
because the first question a skeptical reader asks is whether the harness was
shaped around the result. This directory answers that question by handing the
measurement to a harness written by a competitor.

Nothing here has been run yet.

## What is here

| Path | What it does |
|---|---|
| `patch_gbm_bench.py` | Makes NVIDIA's `gbm-bench` import on a machine with no CUDA and registers two mojotrees arms. Anchored edits, idempotent, fails loudly if upstream moves |
| `gbm_bench/mojotrees_algorithm.py` | The arm itself, mirroring gbm-bench's `LgbmAlgorithm` parameter for parameter, with every spelling difference recorded in `PARITY_NOTES` |
| `run_gbm_bench.sh` | Clone, pin, patch, record the box, run, write JSON into `bench/results/` |
| `record_environment.sh` | Chip, cores, memory, OS, thermal state, power source, library versions |

The patch touches three things in `algorithms.py`: it makes the CUDA-only
imports optional, adds two names to the factory, and imports the adapter. It
changes no timing code, no metric, no dataset, and no parameter belonging to
another library. That restraint is the entire value of the exercise, so a
future edit that relaxes it forfeits the credibility this was built to buy.

## The arena question, which matters more than the code

An honest reading of our own records says we lose to hypertuned CUDA
backends on NVIDIA hardware. XGBoost and CatBoost have spent a decade on
theirs; our CUDA path has no validation record at all. Running gbm-bench on
an NVIDIA box and publishing the result would be paying for credibility with
a loss.

So the arm we can win in someone else's harness is Apple silicon, where all
three competitors run on the CPU because none of them ships a GPU path for
Metal (see the citations in the repository README). The comparison there is
our accelerator against their CPU, and it is a fair fight rather than a
rigged one, because that is the machine the user is actually sitting at.

**Publication rule.** Apple silicon numbers are publishable once they meet
the conditions below. NVIDIA and AMD numbers are not publishable from this
harness until those backends have a validation record and are competitive.
Portability is an architecture claim and belongs in prose. Speed is a
measurement and belongs in a table. Running them together in one table
invites a fight we currently lose and teaches a reader nothing.

## Determinism, and why it is a third arm

gbm-bench's `lgbm-cpu` runs LightGBM stock, which is nondeterministic across
threads. mojotrees is reproducible by construction, so `mojotrees-gpu`
against stock `lgbm-cpu` is not like-for-like on that axis, and the
difference does not favor us in the way it might look: `deterministic=true`
costs LightGBM speed, so the stock arm is LightGBM at its fastest.

Setting `deterministic` on their arm would be the exact move that voids the
reason for using their harness. `-extra` cannot do it either, because
`runme.py` applies one dict to every arm in the run, so it would reach
mojotrees and CatBoost as well.

So `lgbm-cpu` is left byte-identical to upstream and stays the headline
comparison, and `lgbm-cpu-det` is registered as an additional arm that runs
LightGBM with `deterministic=true` and `force_row_wise=true`, the pairing
LightGBM's own documentation asks for. Adding an arm is not the same act as
changing one. Both run in the same process, so all three are interleaved
rather than compared across thermal windows.

Report all three. The honest sentence is that we beat or lose to stock
LightGBM by X and to reproducible LightGBM by Y, and that we are reproducible
in both cases.

## Conditions on any number that leaves this machine

These are the same rules the internal harness already follows, and they apply
harder here because these numbers are meant for strangers.

1. **Interleave the arms inside one process.** This repository has measured
   the same benchmark drifting two- to threefold between time windows.
   Sequential arms in separate runs are not comparable, whoever wrote the
   harness. `runme.py` takes a comma-separated `-algorithm` list and runs
   them in one process, which is what makes this workable; repeat the whole
   invocation and report the spread.
2. **Report accuracy beside every time.** gbm-bench already computes it. A
   speed number without the metric next to it is the thing we criticize
   other people for.
3. **Ship the box record.** `record_environment.sh` output goes with the
   numbers, including thermal state and power source. It was a laptop.
4. **Pin the harness commit.** `run_gbm_bench.sh` writes it into the env
   file. A gbm-bench number is only reproducible against the gbm-bench that
   produced it.
5. **Name the parity gaps.** `PARITY_NOTES` in the adapter is the list. It
   goes in the writeup, not just in the source.

## Running it

    bench/external/run_gbm_bench.sh year 500 mojotrees-gpu,lgbm-cpu,cat-cpu

`year` is the smallest useful dataset here and the right one to shake the
plumbing out on. `airline` and `bosch` are tens of gigabytes; check free
space first. The harness downloads into `bench/external/.gbm-datasets` unless
`GBM_BENCH_DATA` says otherwise, and both scratch directories are ignored.

## The other two suites

`catboost/benchmarks` and Microsoft's `lightgbm-benchmark` are the other two
harnesses people cite. Neither is wired up here. gbm-bench went first
because its algorithm interface is two abstract methods and its parameter
handling is shared across arms, which makes the adapter small and makes the
"we changed nothing of theirs" claim easy for a reader to verify.
