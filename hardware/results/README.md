# hardware/results/

Accepted hardware records and the raw output behind them.

Empty. No outside result has been submitted, reviewed, or accepted, and no file
here means exactly that.

## What lands here

Two things per accepted submission, committed by a maintainer after review:

```
hardware/results/<record_id>.json          the record, as reviewed
hardware/results/raw/<record_id>/*.txt     the terminal output it came from
```

`<record_id>` is `<vendor>-<device-slug>-<YYYY-MM-DD>-<handle>`, for example
`nvidia-l4-2026-08-20-octocat`. A second record from the same person, device, and
day gets a `-2` suffix. Ids are never reused and never renamed, because
`docs/GPU_VALIDATION.md`, `packaging/matrix/accelerators/index.toml`, and the
`conflicts` field of other records all point at them by name.

Contributors do not open pull requests here. File the
[hardware result issue form](../../.github/ISSUE_TEMPLATE/hardware_result.yml)
with the record and its attachments; a maintainer commits what was reviewed, so
that what is in the repository is what somebody read.

## Rules

**Nothing here is edited after it is committed.** Not to fix a number, not to
tidy a field, not to reconcile it with a later run. A record is a description of
one session on one machine, and that session does not change. A mistake found
later is corrected by a new record whose `notes` says what it supersedes, and by
a maintainer note in the review block of the original.

**Raw output is not summarized.** The files under `raw/` are what the terminal
printed. Truncation is allowed when output runs to megabytes, and the record says
`truncated: true` and what was cut.

**Two records that disagree both stay.** Neither is deleted, neither is edited to
agree, and neither is quietly preferred. Each names the other in its `conflicts`
array. Which of the two the matrix believes, and what happens to the index row
while a conflict is open, is decided by the procedure in
[`handoffs/release_11_hardware_network.md`](../../handoffs/release_11_hardware_network.md);
the short version is that a conflict pushes the affected step back to the weaker
status until a third run settles it.

**A record here is evidence, not a status.** Status lives in
`packaging/matrix/accelerators/index.toml`, prose lives in
`docs/GPU_VALIDATION.md`, and both are maintained by hand from what is here.
A file appearing in this directory changes neither on its own.

**A record about one device is about one device.** Not the architecture, not the
generation, not the vendor. An M4 Pro says nothing about an M4 Max, gfx1100 says
nothing about gfx942, and one L4 says nothing about NVIDIA.
