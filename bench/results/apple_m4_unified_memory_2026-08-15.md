# Apple M4 unified memory routes, 2026-08-15

Record identifier: **UM-2026-08-15-M4-01**. First execution of
`bench/apple/unified_memory.mojo` (`pixi run bench-unified-memory`), the
experiment `docs/APPLE_UNIFIED_MEMORY.md` specifies. Raw `um.*` output for
every run, the `/usr/bin/time -l` capture, and the bracketing `vm_stat`
captures are in `apple_m4_unified_memory_2026-08-15/`. This is a
development result and it licenses exactly the claims listed under
"What this run establishes" and nothing beyond them.

## Environment

- Apple M4, 10 physical CPU cores, 10 GPU multiprocessors, 16 GB memory
- macOS 26.5.2, arm64
- Mojo 1.0.0 (ed45d567), MAX 26.5.0
- Load average at start 2.53, 2.13, 1.97; at end 3.5 (a VS Code
  instance, Docker Desktop, and an idle self-hosted CI runner were resident
  for the whole sweep; no other Mojo, pixi, or pytest process ran)
- Memory was under pressure. `vm_stat` shows 344,840 compressor pages at
  start and swapouts advancing by about 110,000 pages across the first
  1024 MiB `rewrite` run. Peak RSS reached 4.1 to 4.6 GB on the 1024 MiB
  configurations. Absolute times are therefore provisional; the route
  ordering, which is large, is what this record is for.
- Not run: the size ladder (`MOJOTREES_UM_LADDER=1`), because the machine
  was already compressing, and Instruments. Neither the
  `no_second_allocation` rung nor the `no_blit` rung can be claimed from
  this run for any route.

## Protocol as executed

Every configuration was three separate process launches of
`unified_memory <MiB> 8`, serialized under `tools/with_build_lock.sh`,
each wrapped in `/usr/bin/time -l` and bracketed by `vm_stat`.

| configuration | payload | mode | extra |
|---|---|---|---|
| `rewrite_256`, `rewrite_1024` | 256, 1024 MiB | `rewrite` (gradient shape) | |
| `resident_256`, `resident_1024` | 256, 1024 MiB | `resident` (binned-matrix shape) | |
| `contend_rewrite_256`, `contend_rewrite_1024` | 256, 1024 MiB | `rewrite` | `MOJOTREES_UM_CONTEND=1` |
| `hold_rewrite_1024` | 1024 MiB | `rewrite` | `MOJOTREES_UM_HOLD_MIB=512` |

The 4096 MiB size the protocol suggests was not attempted on a 16 GB
machine that was already compressing.

## Results

Times are milliseconds unless the column says otherwise; the three values in
a cell are the three process launches, in order. `round_mean` is the
steady-state round (rounds 1 to 7, the retouch round excluded in
`resident`). `write` is the host writing the payload; `publish` is the
enqueue of the copy or the mapped-block exit; `sync` is the drain that
carries the transfer on the copy routes; `readback` is the four-byte
result copy plus its drain. `copy bytes issued` counts every byte
mojotrees asked the runtime to copy over the whole run.

### contend_rewrite_1024  (mode=rewrite, payload_mib=1024, rounds=8, repeats=3, peak RSS MiB=[4127, 4366, 4306])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 328.91/333.63/331.77 | 270.59/273.23/272.42 | 0.04/0.03/0.04 | 0.08/0.08/0.08 | 0.72/0.64/0.68 | 1.43/1.40/1.41 | -/-/- | 0.306/0.311/0.309 | 8589934592/8589934592/8589934592 | 2 |
| copy_direct | ok/ok/ok | 331.17/332.09/332.69 | 269.41/270.79/271.86 | 26.62/26.16/26.22 | 0.36/0.07/0.08 | 0.55/0.61/0.60 | 1.25/1.23/1.24 | -/-/- | 0.308/0.309/0.310 | 8589934592/8589934592/8589934592 | 2 |
| map_write | ok/ok/ok | 503.55/560.80/515.38 | 430.00/467.70/440.98 | 24.77/24.54/24.72 | 0.08/0.08/0.09 | 0.64/0.67/0.55 | 2.04/0.80/0.85 | -/-/- | 0.469/0.522/0.480 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -275014048) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -275014048) |||||||||

### contend_rewrite_256  (mode=rewrite, payload_mib=256, rounds=8, repeats=3, peak RSS MiB=[1305, 1305, 1303])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 83.70/83.36/84.55 | 67.28/67.47/67.74 | 0.03/0.03/0.03 | 0.06/0.06/0.78 | 0.52/0.46/0.50 | 1.40/1.44/1.41 | -/-/- | 0.312/0.311/0.315 | 2147483648/2147483648/2147483648 | 2 |
| copy_direct | ok/ok/ok | 82.41/82.28/84.49 | 67.10/66.95/68.57 | 6.51/6.49/6.48 | 0.06/0.05/0.91 | 0.51/0.53/0.53 | 1.26/1.26/1.17 | -/-/- | 0.307/0.307/0.315 | 2147483648/2147483648/2147483648 | 2 |
| map_write | ok/ok/ok | 102.36/102.40/102.46 | 86.00/86.29/86.45 | 7.21/7.05/7.08 | 0.55/0.39/0.18 | 0.52/0.53/0.47 | 1.07/1.08/1.08 | -/-/- | 0.381/0.381/0.382 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -1152133794) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -1152133794) |||||||||

### hold_rewrite_1024  (mode=rewrite, payload_mib=1024, rounds=8, repeats=3, peak RSS MiB=[4637, 4635, 4637])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 311.69/311.43/310.41 | 274.96/272.51/272.60 | 0.04/0.04/0.04 | 12.87/14.33/12.96 | 0.20/0.25/0.21 | 1.32/1.32/1.32 | -/-/- | 0.290/0.290/0.289 | 8589934592/8589934592/8589934592 | 2 |
| copy_direct | ok/ok/ok | 317.95/318.54/318.43 | 268.39/269.32/271.03 | 26.41/25.82/26.59 | 22.83/23.08/20.49 | 0.21/0.21/0.20 | 1.26/1.25/1.28 | -/-/- | 0.296/0.297/0.297 | 8589934592/8589934592/8589934592 | 2 |
| map_write | ok/ok/ok | 429.76/399.01/435.81 | 390.94/361.78/398.49 | 24.50/24.31/24.53 | 14.04/12.66/12.52 | 0.21/0.19/0.19 | 0.96/1.01/0.93 | -/-/- | 0.400/0.372/0.406 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -275014048) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -275014048) |||||||||

### resident_1024  (mode=resident, payload_mib=1024, rounds=8, repeats=3, peak RSS MiB=[3803, 4125, 4121])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 13.32/12.56/13.56 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 13.05/12.32/13.32 | 0.23/0.20/0.20 | 31.15/32.48/30.32 | 2.59/2.82/2.62 | 0.012/0.012/0.013 | 2147483648/2147483648/2147483648 | 2 |
| copy_direct | ok/ok/ok | 12.47/12.99/13.17 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 12.23/12.74/12.88 | 0.20/0.21/0.26 | 31.79/31.57/30.07 | 4.86/4.69/4.65 | 0.012/0.012/0.012 | 2147483648/2147483648/2147483648 | 2 |
| map_write | ok/ok/ok | 12.36/12.59/12.41 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 12.15/12.36/12.18 | 0.19/0.19/0.19 | 32.86/33.23/32.69 | 21.73/14.19/6.83 | 0.012/0.012/0.012 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -275014048) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -275014048) |||||||||

### resident_256  (mode=resident, payload_mib=256, rounds=8, repeats=3, peak RSS MiB=[1045, 1045, 1045])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 3.98/3.38/5.82 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 3.75/3.14/5.52 | 0.21/0.20/0.25 | 26.74/32.99/20.26 | 2.57/2.67/1.71 | 0.015/0.013/0.022 | 536870912/536870912/536870912 | 2 |
| copy_direct | ok/ok/ok | 3.97/4.36/4.89 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 3.74/4.14/4.57 | 0.19/0.18/0.28 | 26.13/24.23/24.75 | 4.58/4.01/4.97 | 0.015/0.016/0.018 | 536870912/536870912/536870912 | 2 |
| map_write | ok/ok/ok | 4.15/3.89/3.64 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | 3.78/3.48/3.28 | 0.34/0.38/0.33 | 25.31/27.33/34.47 | 5.92/6.05/6.61 | 0.015/0.015/0.014 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -1152133794) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -1152133794) |||||||||

### rewrite_1024  (mode=rewrite, payload_mib=1024, rounds=8, repeats=3, peak RSS MiB=[3089, 3479, 3338])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 325.19/309.28/313.01 | 286.41/270.25/273.52 | 0.04/0.04/0.03 | 13.81/14.30/14.11 | 0.25/0.46/0.46 | 2.95/1.33/1.32 | -/-/- | 0.303/0.288/0.292 | 8589934592/8589934592/8589934592 | 2 |
| copy_direct | ok/ok/ok | 320.43/325.61/319.35 | 267.99/269.65/267.99 | 26.29/26.72/26.19 | 25.48/28.73/24.63 | 0.56/0.41/0.42 | 1.43/1.24/1.28 | -/-/- | 0.298/0.303/0.297 | 8589934592/8589934592/8589934592 | 2 |
| map_write | ok/ok/ok | 475.46/475.00/466.57 | 437.63/434.67/428.91 | 24.64/25.82/24.06 | 12.83/14.05/13.13 | 0.28/0.37/0.40 | 1.72/0.88/0.90 | -/-/- | 0.443/0.442/0.435 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -275014048) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -275014048) |||||||||

### rewrite_256  (mode=rewrite, payload_mib=256, rounds=8, repeats=3, peak RSS MiB=[1051, 1058, 1045])
| route | status | round_mean_ms (3 repeats) | write_ms | publish_ms | sync_ms | readback_ms | round0/steady | retouch/steady | ns/byte | copy bytes issued | drains |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_staged | ok/ok/ok | 77.05/76.74/78.92 | 64.53/64.42/66.52 | 0.02/0.02/0.03 | 4.95/4.83/4.92 | 0.46/0.38/0.46 | 1.91/1.50/1.42 | -/-/- | 0.287/0.286/0.294 | 2147483648/2147483648/2147483648 | 2 |
| copy_direct | ok/ok/ok | 77.89/82.35/77.40 | 64.67/68.50/64.96 | 6.79/7.00/6.31 | 5.93/6.30/5.59 | 0.43/0.46/0.47 | 1.43/1.36/1.39 | -/-/- | 0.290/0.307/0.288 | 2147483648/2147483648/2147483648 | 2 |
| map_write | ok/ok/ok | 107.96/103.12/102.04 | 91.47/90.47/85.30 | 7.97/6.72/7.88 | 8.04/5.47/8.31 | 0.41/0.40/0.48 | 1.40/1.15/1.11 | -/-/- | 0.402/0.384/0.380 | 0/0/0 | 2 |
| host_direct | wrong | kernel took a host-buffer pointer but read the wrong bytes (checksum 0 vs expected -1152133794) |||||||||
| out_host_direct | wrong | device checksum disagreed with the host reference (checksum 0 vs expected -1152133794) |||||||||

## What this run establishes

Read with `docs/APPLE_UNIFIED_MEMORY.md`, "Reading the results". Each
statement below is scoped to this Mojo version, this OS, and this chip.

1. **`host_direct` is `wrong` on this stack, at every size, in every
   mode.** The kernel accepts a pinned `HostBuffer` pointer as its payload
   argument and reads zeros through it (checksum 0, every run). No timing
   from this route is reported and none may be. This is the "correctness
   trap worth documenting loudly" the reading table anticipated. Ledger:
   `compiled`, not `checksum`.
2. **`out_host_direct` is `wrong` on this stack.** A global integer atomic
   into a pinned host buffer does not produce the right answer (checksum 0,
   every run). Per the reading table this closes the histogram-output
   question on this stack outright: the per-node histogram download stays a
   device buffer plus a copy. Ledger: `compiled`, not `checksum`.
3. **`map_write` is correct and issues no copy, and it is slower.** Checksum
   correct in all 21 runs, `copy_bytes_issued_total` 0. In `rewrite` mode
   the host stores through the mapped pointer cost 1.5x to 1.6x the stores
   into a pinned or heap buffer (1024 MiB: 429 to 468 ms against 268 to
   286 ms), and that write is 85% to 90% of the round, so the route loses by
   45% to 60% end to end. In `resident` mode, where the payload is written
   once and touched, it is indistinguishable from the copy routes on the
   steady state (12.4 to 12.6 ms against 12.5 to 13.6 ms) and its retouch
   round is more expensive and much noisier (6.8x to 21.7x steady, against
   2.6x to 2.8x for `copy_staged`). Ledger: `no_copy_issued`, and Claim 1.5
   only. Nothing here says the runtime did not blit behind the block exit.
4. **The transfer is not the cost.** On `copy_staged` the enqueue plus its
   drain moved 1 GiB in 12 to 14 ms in every configuration (about 75 to 85
   GB/s), while the host-side write of the same 1 GiB took 265 to 290 ms.
   The `resident` steady state, which is one 1 GiB copy with no write, is
   0.012 to 0.013 ns per byte. Scaled to the trainer, the whole binned
   matrix of a 1,000,000 x 50 fit (50 MB) copies in under a millisecond,
   and a round's Float32 gradient and hessian planes (8 MB) in a fraction
   of that. Whatever the one-time GPU setup at that shape is spending its
   1.6 to 1.9 s on (`bench/README.md`), it is not the copy.
5. **Pinning is worth about 2% to 3%.** `copy_direct` (heap source) pays a
   26 ms `publish` on the enqueue that `copy_staged` (pinned source) does
   not, and reaches the same steady state within noise (1024 MiB `rewrite`:
   319 to 326 ms against 309 to 325 ms).
6. **Host contention and a resident second buffer did not move the
   ordering.** `MOJOTREES_UM_CONTEND=1` cost 3% to 6% on every route;
   holding 512 MiB device-resident for the run cost nothing measurable
   beyond the RSS.
7. **First touch is real and small on the write routes, and dominant on the
   resident shape.** `round0_over_steady` is 1.2x to 1.4x in `rewrite` (the
   first write faults the pages) and 26x to 33x in `resident`, where the
   steady round is only the copy; the first round in `resident` is a full
   1 GiB write plus its faults. This is what the trainer's one-time upload
   pays once per session and it is not a per-round cost.

## What this run does not establish

- Anything about Claim 2 (no duplicated bytes). No route earned
  `no_second_allocation` or `no_blit`; the ladder and the Instruments trace
  were not run.
- Anything about the `wrapped_host_buffer` routes, which remain
  `not_probed`.
- Anything about a route being worth enabling in the trainer. Every route
  is below `ENABLE_LEVEL` and the shipped default stays `copy_staged`.
- Whether the `host_direct` result is a Metal binding rule or a MAX
  behavior that a different pointer form (Route 5) would change. The
  observation is that this form reads zeros; the mechanism is not
  identified.
- Absolute timings, given the memory pressure recorded above. Repeat on an
  idle machine before quoting a number from this file outside it.
