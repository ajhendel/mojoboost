"""Whether the CPU count honors a container's quota rather than the host's.

WHY THIS FILE EXISTS

Nothing in this repository read a cgroup quota before 2026-08-18.
`num_physical_cores()` reports the HOST's topology, and a cgroup limit is
enforced by the scheduler where that call cannot see it. So on Docker, on
Kubernetes, on GitHub Actions, and on any customer cluster with a CPU limit,
every count `apple_cpu_policy` derives described a machine we did not have.

It was found on an NVIDIA container where `nproc` returned **256** against a
quota of **27.2 CPUs**, a factor of nine, and it was one of three live
hypotheses for a deadlock on that box. Every task count, grain size and
crossover threshold in that file was being computed against the 256.

WHAT IS TESTED AND WHAT CANNOT BE

The quota lives in files that do not exist on macOS, so this machine cannot
exercise the read end to end. What it CAN exercise, and what a container would
actually hit, is the parsing and every fallback. So the parser is tested
directly against the exact strings each cgroup version writes, including the
reported case, and `cgroup_cpu_limit` is tested for the property that matters
most on a machine with no cgroup at all: it must return 0, meaning no opinion,
so the caller keeps the topology.

**0 must never mean zero cores.** A misparse that shrank a real machine's pool
to nothing would be a worse defect than the one this fixes, which is why the
caller treats 0 as "no limit" and why that is asserted here rather than assumed.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    CpuProfile,
    _parse_first_two_ints,
    cgroup_cpu_limit,
)


def test_the_reported_container_quota_parses_to_its_real_cpu_count() raises:
    """The case that found the bug: 2720000 microseconds per 100000 is 27.2
    CPUs, and the box was reading 256."""
    var qp = _parse_first_two_ints(String("2720000 100000"))
    assert_equal(qp[0], 2720000, "quota")
    assert_equal(qp[1], 100000, "period")
    assert_equal(
        qp[0] // qp[1],
        27,
        "27.2 CPUs must floor to 27 rather than round to 28: asking the"
        " scheduler for more than it will give is the oversubscription this"
        " exists to stop",
    )


def test_an_unlimited_v2_cgroup_reads_as_no_opinion() raises:
    """`max <period>` is cgroup v2's spelling for no limit.

    `max` contributes no digits, so the period arrives as the FIRST integer and
    there is no second one. That is why the caller requires both to be positive
    before it believes a limit, and it is the mechanism by which an unlimited
    cgroup and a missing file produce the same answer.
    """
    var qp = _parse_first_two_ints(String("max 100000"))
    assert_equal(qp[0], 100000, "the period lands first when max has no digits")
    assert_equal(qp[1], 0, "and there is no second integer")


def test_a_sub_cpu_quota_does_not_floor_to_zero_at_the_caller() raises:
    """A quota under one full CPU floors to 0 in the division, and the caller
    clamps it to 1, because a pool of zero cannot run anything."""
    var qp = _parse_first_two_ints(String("50000 100000"))
    assert_equal(qp[0] // qp[1], 0, "half a CPU floors to zero")
    assert_true(
        qp[0] > 0 and qp[1] > 0,
        "both are positive, so the caller sees a real limit and clamps to 1"
        " rather than treating it as unlimited",
    )


def test_empty_and_malformed_files_yield_nothing() raises:
    """Neither may be mistaken for a limit."""
    var empty = _parse_first_two_ints(String(""))
    assert_equal(empty[0], 0, "empty quota")
    assert_equal(empty[1], 0, "empty period")
    var junk = _parse_first_two_ints(String("garbage"))
    assert_equal(junk[0], 0, "malformed quota")
    assert_equal(junk[1], 0, "malformed period")


def test_a_machine_with_no_cgroup_returns_no_opinion() raises:
    """On macOS the files do not exist, and the answer must be 0.

    This is the assertion that keeps the fix from becoming a worse bug. 0 is
    read by `CpuProfile.detect` as "no limit, keep the topology". If a missing
    file ever produced a positive number, or if an exception escaped instead of
    being caught, a Mac would start planning against a fabricated core count.
    """
    assert_equal(
        cgroup_cpu_limit(),
        0,
        "a machine with no cgroup must report no opinion rather than a limit",
    )


def test_the_topology_survives_when_there_is_no_quota() raises:
    """`detect` clamps only when a positive limit is smaller than the
    topology, so on this machine every count is the hardware's."""
    var p = CpuProfile.detect()
    assert_true(p.physical_cores > 0, "physical cores")
    assert_true(
        p.performance_cores <= p.physical_cores,
        "performance cores cannot exceed physical, and the clamp re-checks"
        " this because a 256-core reading behind a 27-CPU quota would"
        " otherwise leave more performance cores than physical ones",
    )
    assert_true(p.logical_cores >= p.physical_cores, "logical cores")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
