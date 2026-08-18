"""Evidence, emitted by the trainer, of which backend a run actually used.

Why this module exists
----------------------
`device_used` in a record is a label, not evidence. It is
`Booster.device_`, which `python/mojotrees/basic.py` sets from the Python
side's own policy resolution, and whose own comment says the trainer
resolves the device again natively. A native fallback, or a dispatch that
resolves the device and then ignores the answer, is invisible in that
field: the label still says gpu.

That is not hypothetical. `trainset.train_dataset_multiclass` resolved the
device, discarded the answer, and called the CPU trainer unconditionally,
while `model.fit_multiclass` dispatched on the same answer and reached the
GPU. Every multiclass record written through the `Dataset` path in that
window carried `device_used="gpu"` over a fit that ran entirely on the CPU,
and the run's quality gate, self-check, LightGBM differential,
device-agreement check and repeat-determinism check all passed, because
every one of them is consistent with a CPU fit wearing a GPU label. What
caught it was a human noticing that covertype's CPU and GPU records shared
a `predictions_sha256`.

So a record needs a field that only a device run can produce, and it has to
come from the trainer rather than from a second Python-side resolution of
the same policy.

What is reachable, and what is not
----------------------------------
The strongest evidence the trainer emits today is the phase profile
(`src/mojotrees/phase_profile.mojo`). It is printed by the trainer itself,
on stdout, from four call sites that are structurally tied to a backend:

    boosting.train              -> phase_profile begin label=train scope=fit
    train_gpu.train_gpu         -> label=train_gpu scope=fit
    tree.grow_tree              -> label=grow_tree scope=tree
    train_gpu.grow_tree_gpu     -> label=grow_tree_gpu scope=tree

Multiclass matters here because multiclass is where the bug was: the
softmax loop grows one tree per class per round through the public
`grow_tree` on the CPU and the public `grow_tree_gpu` on the device, so a
multiclass fit is labelled `grow_tree` or `grow_tree_gpu` and the two
cannot be confused. A CPU trainer cannot print `train_gpu` or
`grow_tree_gpu` under any resolution of any policy, because those strings
are literals inside the GPU trainer's own source.

Beside the labels the report carries two counters that a host fit leaves at
zero by construction: `PROF_TRANSFER` calls (a device histogram or frontier
download) and `syncs` (a host wait on a device queue). Those are counts of
work only a device path performs, which is why they are recorded separately
from the label rather than folded into it.

What this proves, exactly, and what it does not
-----------------------------------------------
A `grow_tree_gpu` label with non-zero transfers and host synchronizations
proves that the GPU trainer's code path ran and that it moved bytes off a
device and waited for them. It does not separately attest which physical
device, and it is not a driver-level kernel count: this harness cannot
observe a driver. It is evidence about the code path, which is the thing
that was wrong.

The cost of collecting it
-------------------------
`MOJOTREES_PHASE_PROFILE=async` adds two `perf_counter_ns()` reads per
charge and no fences, and the module's own docstring is explicit that async
mode perturbs nothing about the schedule. The arithmetic, so a reader does
not have to take that on faith: a default 31-leaf tree charges on the order
of a few hundred phases, so a hundred-round fit performs of order 10^4 to
10^5 clock reads, which at tens of nanoseconds each is under a millisecond
against fits this harness measures in seconds. That is far inside the two-
to three-fold drift `bench/results/PROFILE_PROTOCOL.md` documents between
time windows. It is not zero, and the record says the profile was on so a
reader can see that it was.

The host-span axis added on 2026-08-18 does not change that order. On a
device-resident fit it brackets about four launch groups per growth step plus
a dozen per tree, so a 31-leaf tree charges of order 130 host spans and a
hundred-round fit performs of order 10^4 more clock reads, which lands in the
same sub-millisecond band as the arithmetic above. It adds no fence either;
every bracket on that axis closes over a host call the schedule already makes,
which is the property `phase_profile`'s host-span section is written around.

`fenced` mode is never used here. It inserts real waits and changes the
schedule, and a timing taken under it would be a different measurement.

What could not be built, and where it would go
----------------------------------------------
The proof that would cost nothing at all is the device identity.
`GpuHistogramBuilder` already reads `ctx.api()` and `ctx.arch_name()` when
it opens and keeps them in `device_api` and `device_arch`
(`src/mojotrees/histogram_gpu.mojo`), and a host fit never opens a context
at all, so a non-empty API string is free positive evidence. It is not
reachable from Python, and exposing it is not a bindings change:

- the builder is constructed and dropped inside `train_gpu.mojo`, which
  returns a `Booster` and nothing else;
- `trainset.train_dataset` and `trainset.train_dataset_multiclass`, which
  are what the Python `train(params, dataset, ...)` binding calls, keep the
  `Booster` and discard everything the fit knew about its backend;
- `Booster`, `MulticlassBooster` (`boosting.mojo`) and `Model`
  (`model.mojo`) have no field to carry it.

Surfacing it therefore needs a witness value threaded out of `train_gpu`
through `trainset` and onto the returned model, in four files this lane
does not own. The same is true of an always-on launch tally: a counter
field on `GpuHistogramBuilder` is cheap and needs no module-level global,
but it would have exactly the same problem, which is that nothing can read
it after the fit returns. A tally nobody can read is not evidence.

Until that plumbing exists, the phase profile is the cheapest thing the
trainer actually emits, and this module records it rather than inventing a
field the trainer does not produce.
"""

import os

#: The trainer's own instrument. `async` is on-with-no-fences; see the cost
#: note above and the mode list in `src/mojotrees/phase_profile.mojo`.
ENV_VAR = "MOJOTREES_PHASE_PROFILE"
MODE = "async"

#: Report labels only the GPU trainer's source can print. `train_gpu` is the
#: single-output fit loop; `grow_tree_gpu` is the per-tree entry the softmax
#: multiclass loop calls once per class per round. Their host twins are
#: `train` and `grow_tree`, which are recorded too but prove the opposite.
DEVICE_LABELS = frozenset({"train_gpu", "grow_tree_gpu"})
HOST_LABELS = frozenset({"train", "grow_tree"})

#: Phases a host fit leaves at zero by construction. `transfer` is a device
#: download plus the wait inside it; `convert` is dequantizing downloaded
#: fixed-point histogram planes, which exists only because the planes came
#: off a device.
DEVICE_ONLY_PHASES = ("transfer", "convert")

_PREFIX = "phase_profile"


def enable():
    """Turn the trainer's profile on for this process and return what was
    there before, for `restore`.

    Written through `os.environ` rather than passed to a subprocess,
    because the caller wants the instrument on for one call and off for the
    rest of the process: the warm-up fit and the prediction passes go
    through the same module and would otherwise contribute blocks that are
    not the measured run's.
    """
    previous = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = MODE
    return previous


def restore(previous):
    if previous is None:
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = previous


def pending(requested):
    """The `backend_proof` block a worker writes before anything parses it.

    The worker turns the instrument on and the trainer prints to the
    process's stdout, which the worker cannot read back reliably: the write
    comes from compiled Mojo through file descriptor 1, not through
    `sys.stdout`. `run.py` already captures the worker's output, so the
    evidence is filled in there. A record that was never run under `run.py`
    keeps this block and says so, rather than claiming an absence of proof
    that nobody looked for.
    """
    if not requested:
        return {
            "source": ENV_VAR,
            "requested": False,
            "unavailable_reason": (
                "the run asked for no backend proof (--no-backend-proof), so "
                "nothing recorded which backend the trainer used"
            ),
        }
    return {
        "source": ENV_VAR,
        "requested": True,
        "mode": MODE,
        "unavailable_reason": (
            "the trainer printed its profile to this process's stdout; "
            "run.py parses it into this field"
        ),
    }


def _empty(requested):
    return {
        "source": ENV_VAR,
        "requested": requested,
        "mode": MODE,
        "blocks": 0,
        "labels": [],
        "modes_seen": [],
        "trees": 0,
        "nodes": 0,
        "transfer_calls": 0,
        "convert_calls": 0,
        "host_sync_calls": 0,
        "syncs": 0,
        "dispatches": 0,
        "unavailable_reason": None,
    }


def parse(text, requested=True):
    """The `backend_proof` block for one run, from its captured stdout.

    The report's format is documented in `PhaseProfile.report`: every line
    begins with the literal `phase_profile`, the second word is the record
    kind, and everything after that is positional and space separated. Only
    three kinds are read here. `begin` carries the label and the mode;
    `phase <name> all ...` carries the per-phase totals, whose fourth,
    fifth and sixth fields after the name are calls, dispatches and syncs;
    `totals` carries the whole-report dispatch and sync counts as
    `key=value`.

    Blocks are summed rather than kept separately. A single-output fit
    prints one fit-scope block; a multiclass fit prints one tree-scope
    block per class per round, because the softmax loop calls the public
    per-tree entry point. Summing makes those two shapes read the same way,
    and the label set says which happened.
    """
    proof = _empty(requested)
    if not requested:
        return pending(False)

    labels, modes = set(), set()
    for line in text.splitlines():
        if not line.startswith(_PREFIX):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        kind = fields[1]
        if kind == "begin":
            proof["blocks"] += 1
            for field in fields[2:]:
                key, _, value = field.partition("=")
                if key == "label" and value:
                    labels.add(value)
                elif key == "mode" and value:
                    modes.add(value)
                elif key in ("trees", "nodes"):
                    proof[key] += _int(value)
        elif kind == "phase" and len(fields) >= 6 and fields[3] == "all":
            # phase_profile phase <name> all <calls> <dispatches> <syncs> ...
            name = fields[2]
            calls = _int(fields[4])
            if name == "transfer":
                proof["transfer_calls"] += calls
            elif name == "convert":
                proof["convert_calls"] += calls
            elif name == "host_sync":
                proof["host_sync_calls"] += calls
        elif kind == "totals":
            for field in fields[2:]:
                key, _, value = field.partition("=")
                if key in ("dispatches", "syncs"):
                    proof[key] += _int(value)

    proof["labels"] = sorted(labels)
    proof["modes_seen"] = sorted(modes)
    if proof["blocks"] == 0:
        proof["unavailable_reason"] = (
            "the instrument was requested but the trainer printed no "
            "phase_profile block; either the run never reached a trainer or "
            f"{ENV_VAR} did not take"
        )
    return proof


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def device_evidence(proof):
    """`(bool, reason)`: whether this record carries trainer-emitted proof
    that a device backend ran, and the sentence a verdict should print.

    Three independent things can carry it and any one is enough, because
    each is emitted by a different part of the GPU trainer: a report label
    only the GPU trainer's source contains, a non-zero count of a phase
    only a device path performs, or a non-zero count of host waits on a
    device queue. Absence of all three is not proof that the CPU ran; it is
    proof that nothing established which backend did, which is the state
    this lane exists to stop shipping.
    """
    if not isinstance(proof, dict):
        return False, "the record carries no backend_proof field at all"
    if proof.get("unavailable_reason"):
        return False, proof["unavailable_reason"]

    labels = [str(name) for name in proof.get("labels") or []]
    device_labels = sorted(set(labels) & DEVICE_LABELS)
    transfers = _int(proof.get("transfer_calls"))
    converts = _int(proof.get("convert_calls"))
    syncs = _int(proof.get("syncs"))

    reasons = []
    if device_labels:
        reasons.append("trainer label " + ", ".join(device_labels))
    if transfers:
        reasons.append(f"{transfers} device transfers")
    if converts:
        reasons.append(f"{converts} device histogram conversions")
    if syncs:
        reasons.append(f"{syncs} host synchronizations")
    if reasons:
        return True, "; ".join(reasons)

    host_labels = sorted(set(labels) & HOST_LABELS)
    if host_labels:
        return False, (
            "the trainer emitted only host labels ("
            + ", ".join(host_labels)
            + ") and no transfer or synchronization"
        )
    return False, (
        "the profile recorded no device label, no transfer, and no host "
        "synchronization"
    )


def csv_cell(proof):
    """One spreadsheet cell, beside `device_used`, so the label and the
    evidence for it can be compared without opening the JSON."""
    if not isinstance(proof, dict):
        return None
    if proof.get("unavailable_reason"):
        return "unavailable"
    labels = "+".join(str(name) for name in proof.get("labels") or []) or "-"
    return (
        f"{labels} transfers={_int(proof.get('transfer_calls'))} "
        f"syncs={_int(proof.get('syncs'))}"
    )
