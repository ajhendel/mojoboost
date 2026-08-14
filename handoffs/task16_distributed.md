# Task 16 handoff: real distributed transport

Lane: transport and session state machine for a multi-process, multi-host
backend.

This lane did not commit or stage anything. It should be noted anyway that
`src/mojoboost/distributed_transport.mojo` and
`tests/parallel/test_distributed_transport.mojo` are already in commit
`b5a9afc "Add parallel accelerator and compatibility foundations"`, which a
concurrent lane in this round created with a whole-tree add while these files
were on disk. Their content there is identical to what is described below.
`docs/DISTRIBUTED_TRANSPORT.md` and this handoff are untracked working-tree
files. Whoever assembles the round should be aware that the transport landed in
someone else's commit message rather than its own.

## Files this lane owns

| path | lines | state |
| --- | --- | --- |
| `src/mojoboost/distributed_transport.mojo` | 1844 | new |
| `tests/parallel/test_distributed_transport.mojo` | 1053 | new |
| `docs/DISTRIBUTED_TRANSPORT.md` | new | new |
| `handoffs/task16_distributed.md` | this file | new |

No existing file was touched. `distributed.mojo`, `collective.mojo`,
`docs/distributed.md`, `pixi.toml`, and the CI workflows are unchanged.

`tests/parallel/` and `handoffs/` did not exist and were created by this lane.

## Test result

```
pixi run mojo run -I src tests/parallel/test_distributed_transport.mojo
Summary [ 0.241 ] 39 tests run: 39 passed , 0 failed , 0 skipped
```

The focused test needed three compile-fix iterations before it ran; the run
above is the final state. `git diff --check` is clean on the owned paths.

No other test, build, benchmark, or Pixi task was run. In particular the full
suite was not run, so whether these files affect anything else is unverified
beyond the fact that no existing file was edited.

## What is real and what is not

**Real, and covered by the 39 tests.** Frame encode and decode with a checksum
over header and payload, `Float64` and zigzag `Int` payload codecs, the session
state machine (epoch and sequence numbering, sticky failure, cancellation,
membership, deadlines on a manual clock), the handshake and its schema digest
over the machine list, machine list parsing and config validation, the
gather-at-root and broadcast-back collective as both pure step functions and a
blocking `Collective` driver, worker loss, the histogram cost contract, split
agreement digests, and checkpoint restart compatibility.

**Not real.** There is no socket. Mojo's standard library exposes no socket
module at the pinned version and this repository has no FFI precedent, so the
`ByteEndpoint` boundary is drawn exactly where a libc TCP adapter lands and the
adapter is not written. `MemoryEndpoint` is an in-memory fake, named as one.

**Therefore.** Nothing here has moved a byte between two processes or two
hosts. Do not describe this lane's output as distributed training over a
network in any changelog, README line, or release note.

## Integration edits, exact

### 1. Register the test in `pixi.toml` (required)

CI runs `pixi run test` and nothing else, so the new test is not executed until
this lands. Append one command to the `test` task on line 9, immediately after
the existing `tests/test_distributed.mojo` entry:

find:

```
&& mojo run -I src tests/test_distributed.mojo && mojo run -I src tests/test_device.mojo
```

replace with:

```
&& mojo run -I src tests/test_distributed.mojo && mojo run -I src tests/parallel/test_distributed_transport.mojo && mojo run -I src tests/test_device.mojo
```

Nothing else in `pixi.toml` changes. The test is CPU only, in process, and runs
in 0.24 s, so it belongs in the default suite rather than behind a feature.

### 2. Cross-reference from `docs/distributed.md` (recommended)

Two edits, both additive.

In section 1, under "Explicitly out of scope for the prototype", change:

```
- an actual network transport (MPI, gRPC, sockets)
```

to:

```
- an actual network transport (MPI, gRPC, sockets); the framing, session, and
  collective protocol a socket transport would sit under are in
  docs/DISTRIBUTED_TRANSPORT.md, which is explicit that no socket exists
```

At the end of section 5, after the paragraph about `LocalCollective` counting
calls and reduced elements, add:

```
A second implementation, `TransportCollective` in
`src/mojoboost/distributed_transport.mojo`, satisfies the same contract over a
byte-endpoint boundary and is tested against in-memory endpoints. It has never
run over a network. See docs/DISTRIBUTED_TRANSPORT.md.
```

### 3. `README.md` (optional, and only with the caveat)

If the transport is mentioned at all, it has to carry the limit in the same
sentence. Suggested wording, to be placed wherever distributed training is
already described:

```
The transport layer underneath it (framing, session ordering, timeouts,
worker loss, checkpoint metadata) is implemented and tested in process; no
socket transport exists and nothing has run across hosts.
```

### 4. `docs/LIGHTGBM_PARITY.md` and `tools/check_parity.py`

No edit needed. This lane adds no LightGBM-facing behavior, no parameter, and
no default. `pixi run check-parity` was not run because nothing this lane
touches is in its scope.

### 5. No edit needed anywhere else

`distributed.mojo` and `collective.mojo` are unchanged and need no change.
`TransportCollective` conforms to `Collective`, so `train_distributed` and
`grow_tree_distributed` accept one today. The test suite demonstrates this by
running `agree_equal_ints`, the real configuration agreement protocol from
`collective.mojo`, over the transport rather than over a mock.

## Conflicts to expect with other lanes in this round

- **`pixi.toml` line 9.** Every lane that adds a test file edits the same
  single-line `test` task. Expect a textual conflict there and resolve it by
  keeping every added command, not by taking one side.
- **`tests/parallel/`.** Created by this lane. If another lane also creates it,
  the directory merges cleanly and only the file list differs.
- **`docs/distributed.md`.** This lane does not edit it. The two edits in
  section 2 above are proposed text for whoever owns that file, and should be
  applied by them rather than by merging this handoff.
- **`collective.mojo` status codes.** This lane deliberately did not extend
  `STATUS_*`. Transport statuses are a separate `TRANSPORT_*` set in the new
  module, because one describes bad training input and the other describes a
  broken session, and merging them would make "the peer went away"
  indistinguishable from "this shard's weights are negative". If another lane
  wants a single code space, that is a follow-up design decision, not a merge
  resolution.

## Next steps, in the order they should be done

1. **Write the TCP `ByteEndpoint`.** `external_call` bindings to libc:
   `socket`, `setsockopt` (`SO_REUSEADDR`, `SO_RCVTIMEO`, `SO_SNDTIMEO`,
   `TCP_NODELAY`), `bind`, `listen`, `accept`, `connect`, `send`, `recv`,
   `close`, plus a hand-built `sockaddr_in`. Loops around `send` and `recv` for
   short writes and short reads, `EINTR` retried. About 150 lines, and the only
   new thing to test. `docs/DISTRIBUTED_TRANSPORT.md` section 7 has the full
   list.
2. **Write the hermetic two-process test.** Binds `127.0.0.1` only, allocates
   its own ports, cleans up every process it starts. The assertions it has to
   make are enumerated in section 7 of the design doc: bit-identical results at
   world sizes 1, 2, 3; repeat-run bit identity; agreement with
   `LocalCollective`; a killed rank failing every survivor with the peer-loss
   message and no hang; a rewritten machine list refused at the handshake; a
   silent peer tripping the deadline. Until this passes, no statement about
   multi-process operation belongs in the repository.
3. **Only then** wire `TransportCollective` into a CLI or Python entry point.
   Doing it earlier gives users a way to reach code that has never talked to
   another process.
4. **Consider packing the histogram into one reduction.** Three typed buffers
   per node is three round trips at identical arithmetic; packing is a factor
   of three. Worth doing once round trips exist and cost something, not before.
5. **Replace the topology when the world gets large.** Gather and broadcast
   puts a linear-in-world-size load on the root. The replacement has to keep
   bit-identical delivery, which rules out a ring with per-rank rotation as
   written. Section 8 of the design doc has the comparison.

## Known gaps and honest weaknesses

- **No socket, so no end-to-end proof.** Everything above `ByteEndpoint` is
  tested; `ByteEndpoint` itself has one implementation and it is a fake.
- **A timeout is recorded as peer loss.** When a read trips the deadline, the
  driver marks the peer lost and the session records `TRANSPORT_PEER_LOST`
  while the raised error still says timeout. That is the intended fail-stop
  behavior but the two codes disagree, and a future reader will notice.
- **No authentication or encryption.** Any process that can reach the root's
  port and speak the protocol can inject contributions, and histogram
  statistics derived from training data would cross the network in the clear.
  This gates a real transport to a trusted network. Design doc section 9.
- **The deadline defaults are guesses.** 30 s to connect and 300 s per
  collective are plausible numbers, not measured ones, and no run has ever
  tested whether they are right.
- **`MAX_PAYLOAD_BYTES` is a guess too.** 64 MiB bounds a corrupted length
  field; it is not derived from any real histogram size.
- **The checksum is FNV-1a 32.** Adequate against truncation and bit flips,
  useless against anything deliberate.
- **Split agreement is unreachable in a correct run.** The split is a pure
  function of the all-reduced histogram, so ranks cannot disagree unless
  something else already broke. It is a cheap assertion for a real transport,
  where a silently corrupted histogram would otherwise produce two different
  trees and no error at all. It is not load bearing today.
- **Checkpointing is metadata only.** No model is written and no run is
  resumed. What exists makes a wrong restart an error; it does not make a
  right one possible.
