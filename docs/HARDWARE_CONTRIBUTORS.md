# Contributing hardware results

Contributor protocol version 1.0.0. Record schema
[`hardware/templates/result.schema.json`](../hardware/templates/result.schema.json),
version 1.0.0. Capture scripts
[`hardware/capture/`](../hardware/capture/), version 1.0.0.

mojoboost has one GPU source for Metal, CUDA, and HIP. No CUDA file, no HIP
file, no Metal file, and no vendor branch anywhere in it. That is a design
commitment, and it is worth exactly as much as the evidence that the one source
is correct on every backend it claims to target.

The evidence today is one Apple M4. No NVIDIA device and no AMD device has ever
executed this code, on a desk or in CI, and no Apple chip other than that M4 has
either. This document is how somebody outside the project changes that.

You do not need to be a Mojo programmer. You need a machine with a GPU, an hour,
and a willingness to paste output you did not edit.

## What a first record buys

In rough order of how much it is worth, and none of it is rhetorical: every row
below is currently empty.

| Device | What a first record settles |
|---|---|
| Any NVIDIA board | Whether the shared GPU source builds, computes the right answer, and repeats bit-identically on CUDA. Determinism is the interesting one: fixed-point Int32 accumulation is what buys it, and a backend whose atomics or scheduling differ from Metal's is exactly where it would break. |
| Any AMD board | The same three questions on HIP, plus which of consumer RDNA and datacenter CDNA you ran, because a result from one says nothing about the other. |
| Apple M1, M2, M3, M5 | Whether the tiling policy's fallbacks are right on a chip that answers a different set of device attributes. Metal refuses six of eleven queries on the recorded M4; nobody knows what the others refuse. |
| Apple M4 Pro, Max, Ultra | Whether the policy scales with GPU core count on Metal. Same generation as the recorded M4, so core count is the only variable. |
| A machine where it does not work | The most under-supplied evidence there is. See [failures are results](#failures-and-unsupported-are-results). |

## The three paths

Pick by how much time you have. Each one is complete in itself, and a shorter
one is not a lesser contribution.

| Path | Time | What you produce |
|---|---|---|
| **Correctness only** | around 30 minutes, most of it the build | Does the code build, see the device, and agree with the CPU. This is the bar for a first record on any device, and it is the one we most want. |
| **Full device record** | two to three hours | Correctness, determinism, the device attribute header, and a phase-timing sweep across several shapes. This is what fills a row in the matrix. |
| **Profiled record** | half a day | The above plus a real profiler trace. Rare, and the only thing that can justify a device-specific change to the code. |

Apple contributors who intend to produce timings anyone will quote have a fourth
option, which is the full benchmark protocol below. It takes an afternoon and it
has an idle gate, cooldowns, and matched-thread comparisons against LightGBM and
XGBoost.

## The procedures, which live elsewhere

This document is the paperwork. The procedures are authoritative and are not
repeated here, because a copy would drift.

| For | Read | It gives you |
|---|---|---|
| Any device, correctness through profiling | [`docs/GPU_VALIDATION.md`](GPU_VALIDATION.md) | The six validation steps, the exact commands, the phase table, and the profiler metric names per vendor |
| Apple silicon, quotable timings and energy | [`docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`](APPLE_GPU_BENCHMARK_PROTOCOL.md) | The eight workloads, the idle gate, cooldown rules, thread matching, and what makes a number quotable |
| The prose record a maintainer will append | [`packaging/matrix/accelerators/TEMPLATE_apple.md`](../packaging/matrix/accelerators/TEMPLATE_apple.md), [`TEMPLATE_nvidia.md`](../packaging/matrix/accelerators/TEMPLATE_nvidia.md), [`TEMPLATE_amd.md`](../packaging/matrix/accelerators/TEMPLATE_amd.md) | The vendor-specific things worth checking while the hardware is in front of you |
| Where a result ends up | [`docs/PLATFORM_MATRIX.md`](PLATFORM_MATRIX.md), [`packaging/matrix/accelerators/index.toml`](../packaging/matrix/accelerators/index.toml) | The status vocabulary and the rule that a status cannot move ahead of its evidence |

## Steps

**1. Get the code.** A clean tree at a commit you can name.

```sh
git clone https://github.com/mojoboost-ml/mojoboost && cd mojoboost
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
git rev-parse HEAD
```

**2. Capture the environment.** One script, one vendor, read-only.

```sh
sh hardware/capture/capture_nvidia.sh > nvidia-capture.txt 2>&1
```

`capture_apple.sh` and `capture_amd.sh` are the other two. They install nothing,
write nothing outside the file you redirect them to, need no root, contact no
network, and upload nothing. `--commands-only` prints the full command list
without running any of it, if you would rather read before you run. They exit 0
even when they find no device, because that is a result.

**3. Run the procedure.** From `docs/GPU_VALIDATION.md`, in its order.
Correctness gates everything after it, so a failure at correctness means you
stop and report the failure rather than continuing to timings.

**4. Fill in the record.** Copy the template for your vendor from
[`hardware/templates/`](../hardware/templates/) and fill it from output you
watched print. Every field that no command answered stays `null`. The standard
commands are already listed under `runs` at status `not-run`, so most of the
work is replacing that word with what happened.

**5. Keep the raw output.** Redirect it to files as you go, or paste your
terminal scrollback into one. Do not summarize it, do not round a number, do not
retype a figure from memory.

**6. Submit.** Open the
[hardware result issue](https://github.com/mojoboost-ml/mojoboost/issues/new?template=hardware_result.yml),
paste the record, and attach the raw output files by dragging them into the
issue. If you would rather write prose than JSON, the older
[accelerator validation report](https://github.com/mojoboost-ml/mojoboost/issues/new?template=hardware_validation.yml)
form is still there and still accepted; the machine-readable form exists because
it makes conflicting results comparable, not because prose is unwelcome.

## Metadata every record must carry

Not a wish list. A record missing any of these cannot be compared against
another record, cannot be reproduced, and will be sent back with a request for
the missing field.

- **Device.** Exact board, as the vendor tool prints it, plus the architecture:
  `gfx1100`, compute capability 8.9, `4-metal4`. The marketing name does not
  transfer between boards; the architecture does.
- **Host.** CPU model, core count, memory, OS name and version, OS build or
  distribution release, kernel, architecture. mojoboost bins on the CPU and
  stages every transfer through host memory, so the host is part of every
  timing.
- **Driver.** Version, from `nvidia-smi`, `rocm-smi`, or the macOS build for
  Metal. On AMD, the ROCm version from `/opt/rocm/.info/version` as well.
- **Mojo and MAX versions.** From `pixi run mojo --version` and
  `pixi list --environment default | grep -Ei '^(mojo|max)'`, in the session that
  produced the results.
- **Commit.** The full 40-character hash from `git rev-parse HEAD`, plus whether
  the tree was dirty. Not a branch name, not "latest main". A timing without a
  commit describes nothing anybody can return to.
- **The exact commands**, including every environment variable set on the line.
- **Machine kind.** Laptop, desktop, or cloud instance, with the instance type
  if it was rented. A shared cloud host and a desk machine are different
  measurement conditions and a reader cannot tell from the numbers.

The capture script prints most of this for you. Two gaps it cannot close, both
on purpose. It reads a Mojo version only if `mojo` is already on your `PATH`, and
it never reads the MAX version at all, because getting either reliably means
running `pixi run`, which solves and installs the environment on first use, and a
capture script must not install anything. Take both from your own
`pixi run mojo --version` and `pixi list` in the validation session. And it does
not know whether the machine is a laptop, a desk, or a rented instance; only you
do, and it changes how every timing should be read.

## A skipped test is not a pass

The GPU suites print `skipped: no accelerator` and pass on a CPU-only machine.
That is correct behavior and it is why CI stays green without a GPU. It also
means the single most common way to file a worthless record is to run
`pixi run test-gpu`, see a green exit code, and report a pass.

Read the output. On a machine that has a GPU, `skipped` means the build did not
see the device, and nothing at all has been tested. That is a finding worth
reporting in its own right, and the schema keeps it separate: `skipped` is its
own status, it requires the printed line as its reason, and it never becomes
`pass`. When a maintainer transfers your record into the matrix, a step you
marked `skipped` becomes `not-run` there, never `pass`.

The usual causes, in the order they are usually true: on ROCm, the user is not in
the render or video group, or the installed ROCm does not list the card as a
supported target; on macOS, the Metal compiler is not installed; anywhere, the
build ran on a machine different from the one you think it did.

## Quality goes next to timing

Inherited from `docs/GPU_VALIDATION.md` and unchanged here: **a throughput
number without a loss number next to it is not a result.** Both trainers print
training MSE and the record has a field for it, next to the timings from the
same fit rather than in a separate section.

The schema enforces this. A measurement with status `pass` must carry a quality
metric, a value, and the name of the raw file it came from. This is the one rule
most likely to feel like bureaucracy while you are filling in a form, and it is
the rule that makes the difference between a benchmark and a demo.

CPU and GPU losses are expected to differ, because the device accumulates
gradients in Float32. What matters is that the difference is inside the
documented tolerance, not that it is zero.

## Raw output is the evidence

A record without attached raw output is a claim. Claims are not accepted, from
anybody, including maintainers.

- Paste or attach what the terminal printed, unedited.
- Truncation is fine when output runs to megabytes. Say `truncated: true` and
  what was cut.
- Redaction is fine. Absolute paths carry usernames. Say what you redacted.
- One repetition is a number, not a measurement. Say how many you ran.
- If the machine was busy, throttled, on battery, or shared, say so and mark the
  measurement not quotable. A record full of unquotable numbers is still useful
  evidence, and an unmarked bad number is worse than no number.

## Failures and unsupported are results

The most valuable submission this project could receive right now is somebody
reporting that mojoboost does not build on their card, with the exact error and
the ROCm or driver version that produced it. Nobody has filed one, and the reason
is that failure feels like a non-result. It is not.

File the record when:

- the build fails on the target, with the error;
- the installed ROCm does not list the card as a supported target;
- `xcrun --find metal` fails, so the Mac has a GPU and no way to compile for it;
- the GPU suites skip on a machine with a working device;
- a test fails, with the failing assertion;
- the device is one MAX does not support at all.

`unsupported` is a status the schema accepts and the matrix has a word for.
Fill in the environment block, set the verdict, write down what refused and with
what message, and stop. The rest of the record staying `not-run` is correct.

One thing that turns a useful failure record into a misleading one: working
around the refusal and then reporting the run as a validation. `HSA_OVERRIDE_GFX_VERSION`
is the common case. An overridden target is a different device from the one the
record names. Record the override in `findings.workarounds_used`, name the target
that actually ran, and the result stays honest and stays publishable.

## License and attribution

Contributed records are committed to this repository and redistributed with it,
so a grant is required before anything can be accepted.

By submitting a record you license it, its raw output, and the numbers in it for
inclusion in mojoboost under the **Apache License 2.0**, the repository's
license, with attribution to the account you submitted from. The issue form
carries this as a required checkbox and the record carries it as the `license`
block.

- **Attribution is default and automatic.** You are credited in this document
  and in the record itself. Set `attribution_text` if you want a different name
  than your handle.
- **Employer clearance is your call and we ask once.** If the hardware or the
  time belonged to somebody else, confirm you may publish before you file. The
  record has a field for it so the question does not come up in the thread.
- **We do not ask for anything we do not need.** No hostname, no username, no
  serial number, no UUID, no company name. The capture scripts avoid all of them
  on purpose and their headers say which.
- **You keep your copyright.** The grant is a license, not an assignment.
- **Withdrawal.** Ask and the record is removed from the repository, along with
  any status it supported. It cannot be removed from the git history or from
  clones other people already have, and no license claim survives the removal.

## After you submit

A maintainer reviews the record against the raw output, asks about anything the
two disagree on, and then either accepts it, accepts it with notes, or asks for
more. Acceptance commits the record and its raw output to
[`hardware/results/`](../hardware/results/), appends a prose record to
`docs/GPU_VALIDATION.md`, and moves the matching row in
`packaging/matrix/accelerators/index.toml`. The full review procedure, including
what happens when two records disagree, is in
[`handoffs/release_11_hardware_network.md`](../handoffs/release_11_hardware_network.md).

Two things that will not happen to your result:

- **It will not be generalized.** A record about an L4 becomes a statement about
  an L4. Not about Ada, not about NVIDIA, not about CUDA. The matrix has one row
  per device for this reason.
- **It will not be quoted past what it supports.** A correctness-only record
  moves the correctness column and nothing else. A timing from a busy laptop
  stays in the record file and out of the README.

If a maintainer disagrees with your reading of your own output, the disagreement
goes in the record rather than being resolved by one side. Your numbers are your
numbers.

## Contributors

Nobody yet. This section lists every person who has filed an accepted hardware
record, and it is empty because no outside device has ever run this code.

| Contributor | Device | Record | Date |
|---|---|---|---|
| | | | |

The Apple M4 result already in `docs/GPU_VALIDATION.md` is the maintainer's
development machine and is not an outside contribution.
