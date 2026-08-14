# Handoff: outside hardware validation network (release task 11)

The paperwork that lets somebody who is not the maintainer send in hardware
evidence, and the procedure for turning what they send into a status change.

Nothing here executes. No script in this lane has been run, no record has been
submitted, no issue form has been rendered by GitHub, and no schema has been
validated against anything. This is a specification and a set of forms.

## Files this lane owns

| File | State |
|---|---|
| `docs/HARDWARE_CONTRIBUTORS.md` | New. The contributor protocol, version 1.0.0, plus the contributors table, which is empty. |
| `hardware/README.md` | New. Index of the directory. |
| `hardware/templates/result.schema.json` | New. JSON Schema draft 2020-12. What a record is, field by field. |
| `hardware/templates/result_apple.json` | New. Empty record, Apple and Metal. |
| `hardware/templates/result_nvidia.json` | New. Empty record, NVIDIA and CUDA. |
| `hardware/templates/result_amd.json` | New. Empty record, AMD and HIP. |
| `hardware/capture/capture_apple.sh` | New. Read-only environment capture, POSIX `sh`. |
| `hardware/capture/capture_nvidia.sh` | New. Same, NVIDIA. |
| `hardware/capture/capture_amd.sh` | New. Same, AMD. |
| `hardware/results/README.md` | New. Where accepted records land, and the rules that keep two disagreeing records both on disk. |
| `.github/ISSUE_TEMPLATE/hardware_result.yml` | New. Machine-readable submission form. |
| `launch/HARDWARE_VALIDATION_REQUEST.txt` | New. Recruitment post, four versions. |
| `handoffs/release_11_hardware_network.md` | This file. |

Nothing else was touched. Not `docs/GPU_VALIDATION.md`, not
`docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`, not `docs/PLATFORM_MATRIX.md`, not
`.github/ISSUE_TEMPLATE/hardware_validation.yml`, not the workflows, not
`packaging/matrix/`, not `CONTRIBUTING.md`, not `README.md`, not `pixi.toml`, and
no source, test, or benchmark file.

The existing `hardware_validation.yml` issue form stays exactly as it is. The new
form is an addition beside it, not a replacement: prose reports are still
accepted, and both documents say so.

## Design decisions worth knowing before changing any of it

**The protocol references, it does not restate.** `docs/GPU_VALIDATION.md` is the
procedure for every vendor and `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md` is the one
for quotable Apple timings. `docs/HARDWARE_CONTRIBUTORS.md` says who, what
metadata, what license, and what happens next, and points at those two for what
to type. A copy of a command list in a second document is a copy that drifts.

**The capture scripts never call `pixi run`.** It solves and installs the
environment on first use, which is a mutation, and these scripts must not mutate
anything. Mojo and MAX versions are therefore not captured automatically; the
protocol asks the contributor for them and says why. This is the one piece of
required metadata the scripts cannot supply, and it is a deliberate trade.

**The scripts exit 0 when they find nothing.** A missing driver, an absent Metal
compiler, and a card the runtime refuses are all complete results. Every script
ends by telling the contributor to file the record with `unsupported` and stop.

**The scripts collect a narrow fixed list.** No hostname, no username, no
wholesale environment dump, no GPU UUID or serial, no git remote URL, which is
the field in this area that has been known to carry an access token. On macOS
they read `sysctl hw.model` rather than `system_profiler SPHardwareDataType`,
because that report carries the serial number and the hardware UUID and the
record needs neither. Each header states its own exclusions. Adding a command to
one of these scripts means checking it against that list.

**Skipped is a first-class status, everywhere.** In the schema it is its own
value, it requires the printed line as its reason, and it can never be written as
`pass`. The issue form makes it a dropdown option next to pass and fail rather
than something a contributor has to volunteer. This is the single most likely way
for a worthless record to enter the repository, so it is guarded three times.

**Quality is structurally next to timing.** `measurements[].quality` sits inside
the same object as `measurements[].timings_s`, and the schema requires a metric
and a value on any measurement whose status is `pass`. The rule is inherited from
`docs/GPU_VALIDATION.md`; what this lane adds is that a record violating it does
not validate.

**The templates are deliberately not valid records.** They carry
`"is_template": true`, and `record_id`, `commit`, and the contributor block are
empty, so validating an untouched template fails with the list of things left to
fill in. That failure is the point.

## Review procedure

The maintainer's side. Assume the submission arrived as an issue with the record
in one field and raw output attached.

**1. Triage.** Confirm the required metadata is present: exact device,
architecture, host, driver, Mojo and MAX versions, full commit hash, machine
kind, and the exact commands. Anything missing goes back as `needs-info` with the
specific field named. Do not fill a field in on the contributor's behalf, and do
not accept "latest main" as a commit.

**2. Reconcile the record against the raw output.** This is the review. Every
non-null number in the record must appear in the attached output. Check at
minimum the commit, the tool versions, every `runs[].status`, one timing per
shape, and every quality value. A number in the record that is not in the output
is the one thing that stops a review outright: ask where it came from before
anything else happens.

**3. Search the raw output for `skipped`.** If the record claims correctness
`pass` and the output contains `skipped: no accelerator`, the record is wrong and
goes back. On a machine with a GPU that line means the build did not see the
device. This check is mechanical and worth doing every time.

**4. Check quality against timing.** Any measurement with a timing and no quality
metric from the same fit is returned. No exceptions, including for maintainers.

**5. Check the license and attribution.** The consent checkbox on the form and
the `license.grant` field in the record must both be present. Confirm the
attribution name. If the contributor named an employer or the hardware sounds
institutional, confirm `license.employer_cleared` was answered rather than left
null.

**6. Decide.**

| Outcome | When |
|---|---|
| `accepted` | The record and the output agree, and the metadata is complete. |
| `accepted-with-notes` | Both true, but something limits how the numbers may be read: a busy machine, one repetition, a dirty tree, a workaround. The limitation goes in `review.notes` and into the prose record, not only into the thread. |
| `needs-info` | Something is missing or does not reconcile. Name the field. |
| `rejected` | Numbers that are not in the output and cannot be produced, no license grant, or no identifiable contributor. Three reasons, and no others. A failing result is never rejected; a failing result is the point. |

**7. Commit what was reviewed.** The record to
`hardware/results/<record_id>.json` with `review` filled in, the raw output to
`hardware/results/raw/<record_id>/`. Contributors do not open pull requests
against that directory, so what is in the repository is what somebody read.

**8. Append the prose record** to the record section of
`docs/GPU_VALIDATION.md`, in the format of the matching template in
`packaging/matrix/accelerators/`. Name the contributor and the record id in the
heading area so the prose and the evidence are one click apart.

**9. Move the index row** in `packaging/matrix/accelerators/index.toml`, using the
mapping below, and set `record` on that row. If no row exists for the device, add
one; a device nobody anticipated is normal and its absence is not a judgment.

**10. Run the checker.** `python3 packaging/matrix/validate_matrix.py`. It fails
if a status moved ahead of its evidence, which is the automated defense against
this drifting into marketing, and it is the reason steps 8 and 9 happen in that
order.

**11. Credit and close.** Add the row to the contributors table in
`docs/HARDWARE_CONTRIBUTORS.md`, reply on the issue with links to what changed,
and say plainly which claims the record does and does not support.

### Mapping a record verdict onto an index row

`verdict` in the record uses the four step names from
`packaging/matrix/accelerators/index.toml` on purpose, so the transfer is
mechanical rather than interpretive.

| Record verdict | Index step value | Note |
|---|---|---|
| `pass` | `pass` | |
| `partial` | `partial` | The record says which part. |
| `skipped` | `not-run` | Never `pass`. A suite that declined to test anything tested nothing. |
| `not-run` | `not-run` | |
| `unsupported` | see below | |
| `fail` | see below | |

The index vocabulary has no word for a step that ran and failed, and this lane
cannot add one, because `index.toml` belongs to another lane. Until it has one,
the interim rule:

- A **refusal** (the runtime or the support list declined the device, the Metal
  compiler is absent, MAX does not support the card) sets the row `status` to
  `unsupported` with the four steps left at `not-run` and the refusal quoted in
  the row's `notes`.
- A **failure** (it built, it ran, it computed the wrong answer or diverged
  between repeats) leaves the steps at `not-run`, leaves `status` at `not-run`,
  and requires the row's `notes` to name the record id and state the failure in
  one sentence. A silent `not-run` row over a known failure is the worst outcome
  this procedure can produce, and the note is the only thing preventing it.

Adding `fail` to the `[steps]` vocabulary in `index.toml` is the clean fix and is
listed under central integration below.

## How conflicting results coexist

Two accepted records on the same device disagreeing about an outcome. Not two
records with different timings; timings differ because conditions differ, and the
conditions block is where that is explained.

**Neither record is deleted, edited, or preferred.** Each names the other in its
`conflicts` array. Both stay in `hardware/results/`, both stay in
`docs/GPU_VALIDATION.md`, and the prose keeps them as separate records under
separate headings rather than merging them into one summary.

**The index row takes the weaker of the two.** A step one record calls `pass` and
another calls `fail` is not `pass`. The row's `notes` names both record ids and
states the disagreement in one sentence, so a reader of the metadata alone learns
that the question is open. A row that averages a conflict away is worse than a
row that says `not-run`.

**First check whether it is a conflict at all.** Two records that differ in
driver version, ROCm version, OS build, commit, or an override such as
`HSA_OVERRIDE_GFX_VERSION` are not describing the same configuration. That is not
a conflict; it is two results, and it is usually the most informative thing in
either of them. Where the difference is a different device wearing the same name,
split the index row rather than reconciling the records.

**Resolution is a third run, not an argument.** A conflict closes when a third
independent record on the same configuration agrees with one of the two, or when
somebody identifies the difference that made them differ. Either way, all three
records stay, and the conflict and its resolution are written into the prose
record. Nothing is quietly cleaned up later.

## How results feed the matrix without overstating generality

The existing chain is `hardware/results/` for evidence,
`docs/GPU_VALIDATION.md` as the record of truth,
`packaging/matrix/accelerators/index.toml` as its index, and the accelerator
summary in `docs/PLATFORM_MATRIX.md` as the three-line view. This lane adds the
first link and changes nothing downstream of it.

The guards, all of which already exist and none of which this lane may weaken:

- **One row per device, and a record is about one device.** An L4 record moves
  the L4 row. Not Ada, not NVIDIA, not CUDA. `index.toml` already splits consumer
  from datacenter parts and Apple tiers from each other for this reason, and the
  `nvidia-rtx-consumer` row carries an explicit instruction to split itself the
  moment a second board is run.
- **`validated` requires all four steps `pass`**, and `validate_matrix.py`
  enforces it. A correctness-only record moves one column. It does not make a
  device validated, and the contributor is told this before they start so the
  distinction does not read as an insult afterwards.
- **A status cannot move ahead of its evidence.** Any status other than `not-run`
  requires a non-empty `record`, checked mechanically.
- **A quotable number is a separate question from an accepted record.** A
  measurement marked `quotable: false` is accepted, committed, and published, and
  stays out of `README.md` and out of any comparison table. The Apple protocol's
  rules for what is quotable apply unchanged; this lane adds the field, not a new
  standard.
- **No aggregation into a vendor claim.** Three NVIDIA records are three device
  rows. There is no procedure in this repository for combining them into a
  sentence about NVIDIA, and the absence is deliberate.
- **The one-GPU-source claim is a claim about the code, not about the
  hardware.** It stays true whatever the records say. What the records change is
  what may be said about behavior and performance on a backend, which today is
  nothing outside one M4.

## Central integration required

None of this is done. Each item is an edit to a file this lane may not touch.

1. **`CONTRIBUTING.md`** ends by pointing at the hardware validation issue
   template. Add `docs/HARDWARE_CONTRIBUTORS.md` beside it as the starting point,
   and mention `hardware/capture/`.
2. **`.github/ISSUE_TEMPLATE/config.yml`** has contact links for the parity
   contract and the validation procedure. Add one for the contributor protocol,
   which is the document somebody deciding whether to spend an hour should see
   first.
3. **`README.md`** has no pointer to any of this. One line in whatever section
   lists documents, plus, if there is a contributing or help-wanted section, the
   specific ask: correctness on any NVIDIA or AMD board, and any Apple chip that
   is not an M4.
4. **`docs/GPU_VALIDATION.md`** could note in its recording section that outside
   records arrive through the issue form and land in `hardware/results/` first.
   Optional, and it is the record of truth either way.
5. **`packaging/matrix/accelerators/index.toml`** needs `fail` added to the
   `[steps]` vocabulary, and the `[vocabulary]` block may want a word for a
   device with a known failing record. Until then the interim rule above applies.
6. **A GitHub label.** The new form applies `hardware-validation`, the label the
   existing form uses, so nothing breaks if no new label exists. If the two
   should be distinguishable in search, create `hardware-result` and add it to
   the new form's `labels` list.
7. **A record checker.** `hardware/templates/result.schema.json` has no
   validator. A standard-library Python script in the shape of
   `packaging/matrix/validate_matrix.py`, run on `hardware/results/*.json`, would
   turn the schema from documentation into a gate, and would let a contributor
   check their own record before filing. It is the single highest-value follow-up
   in this lane, and it is deliberately out of scope here, because this lane was
   not permitted to run anything and shipping an unrun checker is worse than
   shipping none.
8. **Executable bits.** The three capture scripts have the executable bit set and
   a `#!/bin/sh` line, but every instruction in the documentation invokes them as
   `sh hardware/capture/capture_*.sh`, which works either way and is what a
   cautious contributor will type after reading the script.

## Open items and risks

- **Nothing has been executed.** No capture script has run on any machine,
  including the M4 in front of the author. The commands in them are read-only by
  construction and were chosen from the procedures already in the repository, but
  their exact output on a real machine is unknown, and the first contributor is
  also the first person to run them. Expect a version 1.0.1.
- **The issue form has never been rendered.** GitHub validates form YAML on push
  and rejects a malformed one. If it is rejected, the fields most likely at fault
  are the `render:` values and the dropdown blocks.
- **The schema has validated nothing.** The conditional rules, on `skipped`
  requiring a reason and `pass` requiring a quality metric, are written against
  draft 2020-12 and have not been exercised by any validator.
- **Two forms may confuse contributors.** The new one asks for JSON, the existing
  one asks for prose. Both documents say prose is still accepted, but if
  submissions stall, collapsing to one form is the first thing to try.
- **The protocol asks for a lot.** A required 40-character commit hash, required
  raw output, and four attestation checkboxes are friction, and friction costs
  submissions. The judgment made here is that a small number of records that can
  be reviewed beats a larger number that cannot. If nobody submits anything for a
  while, the honest response is to lower the bar for the correctness-only path
  deliberately and in writing, not to quietly accept records that do not meet it.
- **`hardware/results/` is empty and reads as neglect.** It reads correctly: no
  outside device has ever run this code. The directory README says so in its
  second line so that an empty directory cannot be mistaken for a missing one.
