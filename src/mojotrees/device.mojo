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

**This module defines nothing.** It is the compatibility facade over
`device_policy.mojo`, which is the one authoritative implementation of
what the GPU path supports, what a training session is estimated to
allocate, how detected hardware capabilities are read, what evidence a
GPU selection rests on, and what an explicit `gpu` request is refused
for. Every name below is re-exported from it unchanged: same constant,
same function, same signature, same defaults. A module that imports from
here and a module that imports from `device_policy` hold the same object,
so there is one table of these facts and not two (the consolidation round
of 2026-08 collapsed the earlier forwarding wrappers into these
re-exports; the former `gpu_supports` alias of `gpu_supports_outputs`,
which nothing called, went with them).

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

Where `auto` chooses the GPU (the measured Apple M4 rules), where it
still resolves to the CPU, what `MOJOTREES_AUTO_MIN_CELLS` and
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
    AUTO_DEVICE,
    AUTO_MIN_CELLS,
    BINS_UNSPECIFIED,
    CPU_DEVICE,
    GPU_DEVICE,
    NO_DEVICE,
    OBJECTIVE_UNSPECIFIED,
    decide_device_report,
    decide_device_report_reported,
    device_name,
    env_auto_min_cells,
    gpu_available,
    gpu_supports_outputs,
    parse_device,
    resolve_device,
    resolve_device_full,
)

# What each re-exported name is, for a reader who lands here first. The
# rules themselves, and the rest of the policy surface (`decide_device`,
# `DeviceRequest`, `DeviceCapabilities`, `DeviceDecision`, the block and
# warning codes), live in device_policy.mojo.
#
# CPU_DEVICE, GPU_DEVICE, AUTO_DEVICE
#     The device codes `parse_device` returns for "cpu", "gpu", "auto".
# NO_DEVICE
#     `selected_device` when a request was refused: an explicit `gpu` that
#     cannot run selects nothing rather than falling back to the CPU.
# BINS_UNSPECIFIED, OBJECTIVE_UNSPECIFIED
#     "The caller did not declare this." A caller of `resolve_device_full`
#     that knows some of its workload and not the rest needs to be able to
#     say which; inventing a bin count or an objective to fill the gap is
#     how a gate silently starts admitting the wrong thing.
# AUTO_MIN_CELLS
#     Cells (n_rows * n_features) at or above which `auto` chooses the GPU
#     through the environment override. Negative disables that override,
#     which is the default; the shipped crossover rules in device_policy.mojo
#     decide without it.
# gpu_available
#     True when training can run on an accelerator: one was present when
#     this build was compiled and `MOJOTREES_DISABLE_GPU=1` is not set.
# env_auto_min_cells
#     The `auto` size threshold in cells. Unset, negative, or unparsable
#     means disabled.
# parse_device, device_name
#     Device code for a public device name and back. Names are canonical
#     lowercase here; the Python wrapper lowercases what the user passes,
#     which is how LightGBM treats `device_type`.
# gpu_supports_outputs
#     Whether the complete GPU training path covers a workload with this
#     many outputs per round (1 for single-output, the class count for
#     multiclass).
# resolve_device
#     Requested device and shape in, CPU_DEVICE or GPU_DEVICE out, never
#     AUTO_DEVICE; raises for a `gpu` request that cannot run.
# resolve_device_full
#     The same answer over the whole workload (objective, bins, sparse,
#     validation, memory gates all apply).
# decide_device_report, decide_device_report_reported
#     The serialized decision over a flat boundary; the latter for a caller
#     holding an open `DeviceContext` that has read the device.
