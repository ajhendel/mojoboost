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
of a re-exported name is. Callers that want more than "cpu or gpu" should
skip this module: build a `DeviceRequest`, call `decide_device`, and read
the `DeviceDecision` it returns, which carries the blocking reasons, the
warnings, the memory estimate, the policy version, and the evidence
identifier that a bare device code cannot.

Why `auto` is the CPU today, what `MOJOBOOST_AUTO_MIN_CELLS` and
`MOJOBOOST_DISABLE_GPU` do, and why accelerator availability is a
property of the build rather than of the running machine are all
documented in device_policy.mojo, where the rules that implement them
live.

LightGBM difference: LightGBM spells this `device_type` and takes `cpu`,
`gpu`, or `cuda`, with no `auto`. mojoboost has a single portable GPU
backend rather than separate OpenCL and CUDA ones, so the value is `gpu`
for every accelerator, and `auto` is an addition.
"""

from .device_policy import (
    AUTO_DEVICE as _AUTO_DEVICE,
    AUTO_MIN_CELLS as _AUTO_MIN_CELLS,
    CPU_DEVICE as _CPU_DEVICE,
    GPU_DEVICE as _GPU_DEVICE,
    device_name as _device_name,
    env_auto_min_cells as _env_auto_min_cells,
    gpu_available as _gpu_available,
    gpu_supports_outputs,
    parse_device as _parse_device,
    resolve_device as _resolve_device,
)

comptime CPU_DEVICE = _CPU_DEVICE
comptime GPU_DEVICE = _GPU_DEVICE
comptime AUTO_DEVICE = _AUTO_DEVICE

# Cells (n_rows * n_features) at or above which `auto` chooses the GPU.
# Negative disables the heuristic, which is the default: see
# device_policy.mojo for why there is no measured crossover to ship.
comptime AUTO_MIN_CELLS = _AUTO_MIN_CELLS


def gpu_available() -> Bool:
    """True when training can run on an accelerator: one was present when
    this build was compiled and `MOJOBOOST_DISABLE_GPU=1` is not set."""
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
    should call `decide_device` in device_policy.mojo instead.
    """
    return _resolve_device(device, n_rows, n_features, n_outputs)
