# hardware/

Everything an outside contributor needs to send a trustworthy hardware result,
and the place accepted results are kept.

The procedures live elsewhere and are not repeated here. This directory holds
the paperwork around them.

| You want to | Read |
|---|---|
| Contribute a result, start to finish | [`docs/HARDWARE_CONTRIBUTORS.md`](../docs/HARDWARE_CONTRIBUTORS.md) |
| Know what to run on a CUDA or HIP device | [`docs/GPU_VALIDATION.md`](../docs/GPU_VALIDATION.md) |
| Know what to run on Apple silicon for quotable timings | [`docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`](../docs/APPLE_GPU_BENCHMARK_PROTOCOL.md) |
| Write the prose record a maintainer appends to the record of truth | [`packaging/matrix/accelerators/`](../packaging/matrix/accelerators/) |

## Layout

```
hardware/
  capture/
    capture_apple.sh      environment capture, Apple silicon and Metal
    capture_nvidia.sh     environment capture, NVIDIA and CUDA
    capture_amd.sh        environment capture, AMD and ROCm
  templates/
    result.schema.json    what a record is, field by field
    result_apple.json     empty record, Apple
    result_nvidia.json    empty record, NVIDIA
    result_amd.json       empty record, AMD
  results/                accepted records and their raw output
  run_validation.sh       the whole procedure, one paste, any vendor
  Dockerfile              optional hermetic runner, never run on a GPU
```

## The capture scripts

One per vendor, self-contained, POSIX `sh`, no arguments needed:

```sh
sh hardware/capture/capture_nvidia.sh > nvidia-capture.txt 2>&1
sh hardware/capture/capture_nvidia.sh --commands-only   # audit, runs nothing
```

They run read-only informational commands and print each command above its
output. They install nothing, write nothing, need no root, touch no network, and
upload nothing. `--commands-only` prints the full command list without executing
any of it, so you can read what a script would do before letting it do it.

Three properties worth knowing before you run one:

- **They never call `pixi run`.** That solves and installs the environment on
  first use, which is a mutation. Mojo and MAX versions are read only if those
  tools are already on `PATH`; otherwise take them from your own
  `pixi run mojo --version` in the validation session.
- **They exit 0 even when they find nothing.** No driver, no device, no Metal
  compiler: all complete results, and all worth filing. `unsupported` is a
  status the schema accepts and the matrix has a word for.
- **They avoid the identifying fields on purpose.** No hostname, no username, no
  wholesale environment dump, no GPU UUID or serial, no git remote URL. Each
  script's header says exactly what it skips and why. Absolute paths in tool
  output can still carry your username, so read the file before attaching it.

## `Dockerfile`, and what it is not

Optional and secondary. The image **assembles and has never seen a GPU**, which
are two claims and both are in the file's own header along with what was checked.
What is verified is the non-GPU half. The image builds on arm64 against the
default CUDA base, runs as a non-root user, puts pixi on `PATH`, can write to a
bind-mounted checkout, and carries `run_validation.sh` far enough to read the
mounted tree's commit and dirty state. What is unverified is everything that
involves a device, plus the ROCm base, which has not been built at all.

The bare path is the recommended one. `run_validation.sh` needs no root and
installs nothing but pixi's own environment, so on a machine whose driver
userspace is already in place the container buys nothing and adds a failure mode.
Reach for it only when the userspace is missing, which in practice means a rented
box with a bare OS image and no ROCm or CUDA libraries, where installing them on
the host is the one step in the procedure that wants root.

The consequence that matters for a result. A failure inside the container is not
automatically a mojotrees failure, because GPU passthrough is its own failure
mode and this file has no evidence behind it. Re-run on the bare host before
reporting. If the host and the container disagree, that disagreement is the
finding and is worth filing on its own.

## The record templates

`result.schema.json` is the definition. The three vendor files are empty
instances of it, with the standard commands already listed under `runs` at
status `not-run`, so filling one in is mostly a matter of replacing that word.

The templates carry `"is_template": true` and are deliberately not valid
records: `record_id`, `commit`, and the contributor block are empty, and a
validator run against an untouched template is expected to fail with the list of
things still to fill in. Set `is_template` to `false` when you fill one in.

Rules the schema enforces rather than merely requests:

- A `run` with status `skipped` must carry the line the suite printed. A GPU
  suite that says `skipped: no accelerator` on a machine with a GPU is not a
  pass; it means the build did not see the device.
- A `run` with status `fail` must carry the failing assertion, verbatim.
- A `measurement` with status `pass` must carry a `quality` metric and value,
  and must name the raw output file it came from. A throughput number without a
  loss number next to it is not a result.
- Every record must name at least one raw output file. A record without raw
  output is a claim, not evidence.

Nothing has been validated against this schema. It is a specification, and no
checker for it exists in the repository yet.

## `results/`

Where an accepted record and its raw output are committed, by a maintainer,
after review. Contributors do not open pull requests against this directory;
they file the issue form and the maintainer commits what was reviewed. See
[`results/README.md`](results/README.md) for the naming and the rules that keep
two disagreeing records both on disk.

`docs/GPU_VALIDATION.md` remains the record of truth and
`packaging/matrix/accelerators/index.toml` remains its index. A file in
`results/` is the underlying evidence, never a replacement for either.
