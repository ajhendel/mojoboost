"""Training device selection: the public vocabulary.

One vocabulary for choosing where training runs, shared by the Mojo API,
the CPython bindings, and the Python estimators:

- `cpu`: the dependable path. Float64 throughout, every objective, every
  entry point.
- `gpu`: device-resident tree growth (`train_gpu` in train_gpu.mojo). It
  raises when no accelerator is present, or when the workload is outside
  what the GPU path covers. It never silently falls back to the CPU.
- `auto`: the GPU when the complete GPU path covers the workload and
  evidence selects it, the CPU otherwise.

**This module decides nothing.** It is the compatibility facade over
`device_policy.mojo`, which is the one authoritative implementation of
what the GPU path supports, what a training session is estimated to
allocate, how detected hardware capabilities are read, what evidence a
GPU selection rests on, and what an explicit `gpu` request is refused
for. Everything below is a thin wrapper that forwards to it.

The wrappers are wrappers and not re-exports so the symbols this module
has always exported keep being defined here, whatever an importer's view
of a re-exported name is.

Callers that want more than "cpu or gpu" have two ways up from a bare
device code, and neither one is a different policy:

- `resolve_device_full` is the same raising answer over the *whole*
  workload rather than only its shape, so the objective, bin-limit,
  sparse, validation, and memory gates all apply. A trainer that knows
  what it is about to train should call it instead of `resolve_device`.
- `decide_device_report` returns the serialized `DeviceDecision`: the
  blocking reasons, the warnings, the memory estimate, the transfer routes
  in effect, the session state, the policy version, and the evidence
  identifier, none of which a device code can carry. It never raises for a
  refusal, so "what would `device='gpu'` do here" is answerable without a
  try. It is also the seam the bindings, the C API, and the CLI bind,
  because it needs no struct constructed across the boundary.

A Mojo caller that would rather hold the decision than parse it should
still go straight to `decide_device` in device_policy.mojo.

Why `auto` is the CPU today, what `MOJOTREES_AUTO_MIN_CELLS` and
`MOJOTREES_DISABLE_GPU` do, and why accelerator availability is a
property of the build rather than of the running machine are all
documented in device_policy.mojo, where the rules that implement them
live.

LightGBM difference: LightGBM spells this `device_type` and takes `cpu`,
`gpu`, or `cuda`, with no `auto`. mojotrees has a single portable GPU
backend rather than separate OpenCL and CUDA ones, so the value is `gpu`
for every accelerator, and `auto` is an addition.
"""

from .device_policy import (
    AUTO_DEVICE as _AUTO_DEVICE,
    AUTO_MIN_CELLS as _AUTO_MIN_CELLS,
    BINS_UNSPECIFIED as _BINS_UNSPECIFIED,
    CPU_DEVICE as _CPU_DEVICE,
    GPU_DEVICE as _GPU_DEVICE,
    NO_DEVICE as _NO_DEVICE,
    OBJECTIVE_UNSPECIFIED as _OBJECTIVE_UNSPECIFIED,
    decide_device_report as _decide_device_report,
    decide_device_report_reported as _decide_device_report_reported,
    device_name as _device_name,
    env_auto_min_cells as _env_auto_min_cells,
    gpu_available as _gpu_available,
    gpu_supports_outputs,
    parse_device as _parse_device,
    resolve_device as _resolve_device,
    resolve_device_full as _resolve_device_full,
)

comptime CPU_DEVICE = _CPU_DEVICE
comptime GPU_DEVICE = _GPU_DEVICE
comptime AUTO_DEVICE = _AUTO_DEVICE

# `selected_device` when a request was refused: an explicit `gpu` that
# cannot run selects nothing rather than falling back to the CPU.
comptime NO_DEVICE = _NO_DEVICE

# "The caller did not declare this." Re-exported because a caller of
# `resolve_device_full` that knows some of its workload and not the rest
# needs to be able to say which, and inventing a bin count or an objective
# to fill the gap is how a gate silently starts admitting the wrong thing.
comptime BINS_UNSPECIFIED = _BINS_UNSPECIFIED
comptime OBJECTIVE_UNSPECIFIED = _OBJECTIVE_UNSPECIFIED

# Cells (n_rows * n_features) at or above which `auto` chooses the GPU.
# Negative disables the heuristic, which is the default: see
# device_policy.mojo for why there is no measured crossover to ship.
comptime AUTO_MIN_CELLS = _AUTO_MIN_CELLS


def gpu_available() -> Bool:
    """True when training can run on an accelerator: one was present when
    this build was compiled and `MOJOTREES_DISABLE_GPU=1` is not set."""
    return _gpu_available()


def env_auto_min_cells() -> Int:
    """The `auto` size threshold in cells. Unset, negative, or unparsable
    means disabled, in which case `auto` never selects the GPU on size
    alone."""
    return _env_auto_min_cells()


def parse_device(name: String) raises -> Int:
    """Device code for a public device name ("cpu", "gpu", or "auto").

    Names are canonical lowercase here. The Python wrapper lowercases what
    the user passes before calling in, which is how LightGBM treats
    `device_type`."""
    return _parse_device(name)


def device_name(device: Int) raises -> String:
    """Public device name for a device code."""
    return _device_name(device)


def gpu_supports(n_outputs: Int) -> Bool:
    """Whether the complete GPU training path covers this workload.
    `n_outputs` is 1 for single-output training and the class count for
    multiclass.

    The former name of `gpu_supports_outputs` in device_policy.mojo, kept
    so callers written against it keep compiling. New code should use the
    policy module's name, which says which of a workload's several
    dimensions it is answering for."""
    return gpu_supports_outputs(n_outputs)


def resolve_device(
    device: Int, n_rows: Int, n_features: Int, n_outputs: Int = 1
) raises -> Int:
    """Resolve a requested device to the backend that will actually run:
    CPU_DEVICE or GPU_DEVICE, never AUTO_DEVICE.

    CPU_DEVICE resolves to itself. GPU_DEVICE raises when no accelerator is
    available or when the GPU path does not cover the workload, rather than
    falling back. AUTO_DEVICE chooses the GPU only when it is available,
    covers the workload, and evidence or the size heuristic selects it.

    The shape is all this signature carries, so the gates that need an
    objective, a bin count, or the input flags are skipped here and the
    decision records that they were. Callers that know the whole workload
    should call `resolve_device_full` below, which is the same engine asked
    the whole question.
    """
    return _resolve_device(device, n_rows, n_features, n_outputs)


def resolve_device_full(
    device: Int,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    n_bins: Int = BINS_UNSPECIFIED,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    sparse: Bool = False,
    categorical: Bool = False,
    has_missing: Bool = False,
    uses_validation: Bool = False,
) raises -> Int:
    """Resolve a fully described workload to CPU_DEVICE or GPU_DEVICE,
    raising the refusal when it cannot run.

    What a trainer that knows its own objective, bin count, and input flags
    should call instead of `resolve_device`. Same engine, whole question: the
    objective gate, the bin-limit gate, the sparse and validation blocks, and
    the memory gate all apply, so an explicit `device='gpu'` that would fail
    deeper in is refused here instead.

    Fields left at their `_UNSPECIFIED` defaults are treated as undeclared,
    which skips the gates that need them and marks the decision incomplete.
    That is deliberately different from passing a made-up value.
    """
    return _resolve_device_full(
        device,
        n_rows,
        n_features,
        n_outputs,
        n_bins,
        objective,
        sparse,
        categorical,
        has_missing,
        uses_validation,
    )


def decide_device_report(
    device: String,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    n_bins: Int = BINS_UNSPECIFIED,
    objective: Int = OBJECTIVE_UNSPECIFIED,
    sparse: Bool = False,
    categorical: Bool = False,
    has_missing: Bool = False,
    uses_validation: Bool = False,
) raises -> String:
    """The whole decision as `key=value` lines, over a flat boundary.

    The seam a CPython binding, a C API, or a CLI binds: plain scalars in,
    the serialized decision out, no struct to construct. It never raises for
    a workload it refuses; the refusal is `blocked=true` with `message=`
    saying why, so "what would device='gpu' do here" is answerable without a
    try. See `serialize` in device_policy.mojo for the format.
    """
    return _decide_device_report(
        device,
        n_rows,
        n_features,
        n_outputs,
        n_bins,
        objective,
        sparse,
        categorical,
        has_missing,
        uses_validation,
    )


def decide_device_report_reported(
    device: String,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int,
    n_bins: Int,
    objective: Int,
    sparse: Bool,
    categorical: Bool,
    has_missing: Bool,
    uses_validation: Bool,
    reported_api: String,
    reported_arch: String,
    core_count: Int,
    max_threads_per_block: Int,
    max_shared_memory_per_block: Int,
    memory_budget_bytes: Int = 0,
    context_open: Bool = True,
    kernels_ready: Bool = False,
    warmup_level: Int = 0,
) raises -> String:
    """`decide_device_report` for a caller holding an open `DeviceContext`.

    The same engine over capabilities that were read rather than guessed,
    which is what makes the memory gate and the unified-memory warning mean
    anything. The decision carries `profile_source=reported`, so a report
    from here can be told from one that fell back.
    """
    return _decide_device_report_reported(
        device,
        n_rows,
        n_features,
        n_outputs,
        n_bins,
        objective,
        sparse,
        categorical,
        has_missing,
        uses_validation,
        reported_api,
        reported_arch,
        core_count,
        max_threads_per_block,
        max_shared_memory_per_block,
        memory_budget_bytes,
        context_open,
        kernels_ready,
        warmup_level,
    )
