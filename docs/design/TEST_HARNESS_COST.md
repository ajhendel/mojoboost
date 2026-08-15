# What a test run actually costs

Status (Aug 15 2026), measured on `Mojo 1.0.0 (ed45d567)`, macOS arm64, M4.

## The claim, and the answer

`handoffs/consolidation_round.md` records this, from the session that shipped
the sparse GPU path.

> Known: `tests/test_gpu_sparse.mojo` takes about twenty minutes under
> `TestSuite` on the M4 although the same functions called directly run in
> well under a second each (22 ms against 18 s for one of them); the harness
> runs host code far slower than a compiled call.

The claim is wrong, and the factor of a thousand in it is a unit error.

`TestSuite` does not slow host code down at all. In one binary, the same
compute-heavy function costs **231.9 ms called directly, 220.4 ms called
through a runtime function pointer, and 221.2 ms called by
`TestSuite.discover_tests[__functions_in_module()]().run()`**. Every test body
is invoked exactly once. Per test, the harness adds about **0.2 microseconds**,
which is mostly the cost of printing the `PASS` line.

The number `TestSuite` prints in brackets is **milliseconds**. A test that
sleeps for exactly one second prints `PASS [ 1000.598 ]`. Read as seconds it
looks a thousand times worse than it is, and a thousand is exactly the factor
the handoff reports. A direct call at 22 ms and a harness line reading
`[ 18.xxx ]` are not a 1000x gap, they are agreement.

Where the wall clock actually goes is `mojo` compiling. Taking the file the
counter-evidence came from, `tests/test_tree_parameters_extra.mojo`, and
running it cold, the wall clock is **3.10 s** and the 47 tests inside it take
**0.061 ms**. That is 0.002 percent of the run. The other 99.998 percent is
compilation. CI reports 17 s for the same file on a slower shared runner, and
the same split applies.

## The measurements

Every number below comes from a reproducer built and run in a scratch
directory, not from the suite. The source of each is at the end of this file.

### 1. Dispatch is free

`repro_heavy.mojo` holds eight tests whose bodies each run a 20 million
iteration integer loop, seeded from the clock so the optimizer can neither
fold the loop to a constant nor hoist it out of a repeated call, and printing
the result so it cannot be deleted as dead. `main` calls the eight bodies
three ways and times each phase inside the process, so compile time is shared
and cancels.

| Path | Run 1 | Run 2 |
| --- | --- | --- |
| direct calls from `main` | 231.9 ms | 251.7 ms |
| through a `List[def () raises thin -> None]` | 220.4 ms | 254.6 ms |
| `TestSuite.discover_tests[...]().run()` | 221.2 ms | 244.1 ms |

The spread is run to run noise on a shared laptop. There is no gap. The
harness reported `Summary [ 221.136 ]` against a `perf_counter_ns` measurement
of 221.168 ms for the same phase, so its own accounting is honest as well.

Each body prints one marker line. The run prints **24** of them, eight per
phase, so the harness runs each test once and does not repeat it.

The first attempt at this reproducer measured 99 microseconds for all eight
bodies, because the loop masked its accumulator with `0x7FFF_FFFF` and the
assertion was `r != -1`. The optimizer proved the assertion and deleted the
work. A direct-call baseline that has been optimized away is the other way to
manufacture a 1000x gap, and it is worth ruling out before believing one.

### 2. JIT is not the problem either

The suite runs files with `mojo run`, which compiles and executes rather than
producing a binary. Codegen quality is the same.

| | direct | indirect | `TestSuite` | process wall |
| --- | --- | --- | --- | --- |
| `mojo run` (cold) | 245.0 ms | 242.7 ms | 242.6 ms | 1.73 s |
| `mojo run` (warm) | 242.9 ms | 243.5 ms | 245.2 ms | 1.08 s |
| `mojo build` then exec | 251.7 ms | 254.6 ms | 244.1 ms | 0.75 s |

### 3. Per test overhead is a printed line

`repro_many.mojo` holds 500 trivial tests. Three consecutive runs of the built
binary.

| | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| `TestSuite` phase, all 500 | 101 us | 85 us | 79 us |
| harness `Summary` | 0.009 ms | 0.007 ms | 0.006 ms |

About 0.2 microseconds per test of harness overhead, and the bodies themselves
are optimized to nothing. Compiling that file took 11.5 s, so a test costs
roughly 23 ms to compile and about 0.2 microseconds to dispatch. That ratio is
the whole finding.

### 4. The bracketed number is milliseconds

```
    PASS [ 1000.598 ] test_one_second      # sleep(1.0)
    PASS [  105.007 ] test_tenth_second    # sleep(0.1)
--------
Summary [ 1105.605 ] 2 tests run
wall_ns 1105690000 = wall_ms 1105
```

### 5. A real file, cold and warm

`tests/test_tree_parameters_extra.mojo` copied into scratch with a timing
`main`, run with `mojo run -I build`.

The `tests` column is the timed `TestSuite` phase measured around the call,
which for this file the harness reports as `Summary [ 0.061 ]`.

| | wall | tests |
| --- | --- | --- |
| cold | 3.10 s | 0.114 ms |
| warm | 0.57 s | 0.111 ms |
| warm again | 0.57 s | 0.109 ms |

An import-only file with the same import block and a single trivial test costs
1.60 s cold and 0.34 s warm, so roughly 1.6 s of the cold cost is fixed per
file and the rest is proportional to the code in it.

### 6. Optimization level is not a lever

Same file, four cold runs at four levels, each on a fresh copy so the compile
cache cannot serve it.

| level | wall | tests |
| --- | --- | --- |
| `-O3` (default) | 3.09 s | 0.061 ms |
| `-O2` | 3.08 s | 0.065 ms |
| `-O1` | 3.03 s | 0.058 ms |
| `-O0` | 4.68 s | 0.637 ms |

`-O0` is worse on both axes. It compiles 51 percent slower and runs ten times
slower. Elaboration and monomorphization dominate the compile, and turning the
optimizer off leaves more code for the rest of the pipeline to carry. Do not
retry this.

### 7. The compile cache is the one lever left

Mojo keeps a content addressed on disk compile cache at
`$MODULAR_HOME/cache/.mojo_cache/mojo/transform/<version-hash>-production/`.
Under pixi, `MODULAR_HOME` is `.pixi/envs/default/share/max`. In this checkout
it is 1.0 GB across 1723 entries.

It is worth 3.10 s cold against 0.57 s warm on the 47 test file, a factor of
5.4, and 1.60 s against 0.34 s on the import-only file, a factor of 4.7.

CI throws it away on every run. `prefix-dev/setup-pixi` with `cache: true`
keys its cache on the lockfile and saves the environment right after
installing it, before any test has compiled anything, so the cache directory
it stores is empty of test artifacts. Every CI suite run is therefore fully
cold.

## What to change

1. **Persist the compile cache in CI.** This is the only untaken lever with a
   large number attached to it. In `.github/workflows/ci.yml`, after the pixi
   setup step and before `pixi run test-cpu`, add a cache step over
   `.pixi/envs/default/share/max/cache/.mojo_cache`, keyed on the Mojo version
   plus a hash of `src/**` and `tests/**` with a prefix-only restore key so a
   partial hit still serves the unchanged modules. Bound in this checkout at
   1.0 GB, which is within the GitHub per repository budget but wants a
   `restore-keys` fallback rather than an exact match. Nobody owns `.github`
   in the current split, so this is written down rather than applied.

2. **Already done, keep it.** One `mojo precompile` of the package instead of
   sixty in-process rebuilds, and files run concurrently rather than chained
   with `&&`. Both are compile-side, which is the side that matters.

3. **Not worth doing.** Optimization level, per section 6. Anything aimed at
   `TestSuite` dispatch, per section 1.

4. **The remaining structural cost is file count.** About 1.6 s of cold
   compile is fixed per test file, independent of what is in it, and roughly
   23 ms per test on top. Sixty files at 8 way concurrency puts a floor of
   something like 12 s on a cold suite that no runner change can move. Fewer
   and larger test files would move it. That is a change to `tests/`, which
   this note does not make.

## `tests/test_gpu_sparse.mojo` and the twenty minutes

Untested here, because that file is out of budget on this machine, but section
1 rules out the stated cause. `TestSuite` cannot turn a set of functions that
run in well under a second into twenty minutes of host dispatch. The remaining
candidates are that the twenty minutes is compile time, most likely GPU kernel
elaboration and Metal compilation for the sparse kernels, or that the tests
genuinely do that much device work.

One run now discriminates between them, because `tools/run_tests.sh` prints
both numbers.

```
tools/with_build_lock.sh bash tools/run_tests.sh gpu test_gpu_sparse
```

If the line reads something like `(1200s wall, 40ms in tests)` it is compile
and the cache in change 1 is the fix. If it reads `(1200s wall, 1150000ms in
tests)` the device work is real and belongs to the GPU lane.

## The harness change this note justifies

`tools/run_tests.sh` now parses the `Summary [ N ]` line out of each file's log
and prints it beside the wall clock.

```
  ok   test_gpu_split_policy (0s wall, 0.010ms in tests)
```

That line is from a real run after the change. The whole command took 9.29 s,
essentially all of it the one `mojo precompile` of the package, the file
itself compiled and ran in under a second against a warm cache, and the tests
in it took ten microseconds. Anyone reading that output can no longer mistake
compile time for test time, which is the mistake this note exists to correct.

## Reproducers

Unit check, section 4.

```mojo
from std.testing import TestSuite
from std.time import sleep, perf_counter_ns


def test_one_second() raises:
    sleep(1.0)


def test_tenth_second() raises:
    sleep(0.1)


def main() raises:
    var t0 = perf_counter_ns()
    TestSuite.discover_tests[__functions_in_module()]().run()
    var t1 = perf_counter_ns()
    print("wall_ns", t1 - t0)
```

Dispatch check, sections 1 and 2, shown with two of the eight tests. Note the
three things that keep the workload real, an opaque seed from the clock, a
printed result, and an assertion the optimizer cannot discharge.

```mojo
from std.testing import TestSuite, assert_true
from std.time import perf_counter_ns

comptime ITERS = 20_000_000


def _seed() -> Int:
    return Int(perf_counter_ns() & 0xFF)


def _work(seed: Int) -> Int:
    var acc = seed
    for i in range(ITERS):
        acc = (acc * 1103515245 + 12345) & 0x7FFF_FFFF
        acc ^= i & 0xFF
    return acc


def _one() raises:
    var r = _work(_seed())
    print("MARK", r)
    assert_true(r != -1)


def test_a() raises:
    _one()


def test_b() raises:
    _one()


def main() raises:
    var t0 = perf_counter_ns()
    test_a()
    test_b()
    var t1 = perf_counter_ns()
    print("PHASE direct_ns", t1 - t0)

    # A thin function pointer, which is what discovery hands the runner.
    # `def () raises -> None` is the fat closure type and will not accept a
    # plain function; `thin` is the spelling that does.  Iterating a
    # `List` of these yields a reference that has no `__call__`, so index
    # into it and bind a local first.
    var fns: List[def () raises thin -> None] = [test_a, test_b]
    var t2 = perf_counter_ns()
    for i in range(len(fns)):
        var f = fns[i]
        f()
    var t3 = perf_counter_ns()
    print("PHASE indirect_ns", t3 - t2)

    var t4 = perf_counter_ns()
    TestSuite.discover_tests[__functions_in_module()]().run()
    var t5 = perf_counter_ns()
    print("PHASE testsuite_ns", t5 - t4)
```

Cold and warm compile, sections 5 through 7, is any real test file copied to a
fresh path with `main` replaced by the timed `main` above, run as
`mojo run -I build <copy>.mojo`. Cold means a copy whose contents the cache
has not seen, which a one word edit to the docstring is enough to produce.
Never clear `$MODULAR_HOME/cache` to force a cold measurement in a shared
checkout, since every other session pays for it.
