"""The host row-range table refuses to be read while a plane that does not
maintain it owns the partitioning.

# run_tests: cpu-safe -- host bookkeeping only, opens no device.

What is being guarded
---------------------
`GpuActiveRows` keeps a host-side `LeafRangeTable` mapping node id to the
window of the active-row buffer that node owns. Two planes partition that
buffer and only one of them maintains the table. `GpuActiveRows.partition`
ends every split with `LeafRangeTable.split`; `enqueue_partition_desc`, the
descriptor-driven partition the device-resident round uses, routes rows from a
step descriptor a device kernel wrote and updates nothing on the host.

The failure that produced this file was not a crash. A device-gradient round
read the unmaintained table to advance its raw scores, found node 0 owning
every row, and added the root's value to every row instead of each row's own
leaf value. Tree 0 came out bit-identical; every tree after it diverged;
nothing raised. The reason nothing raised is the point of
`test_a_stale_table_passes_every_structural_check` below: a stale table is
*well formed*. Its windows are in bounds, they are pairwise disjoint, and they
cover the active prefix exactly. There is no invariant to check that catches
it, so the invalidity has to be recorded when it is created rather than
detected when it is read.

How this file avoids being a test that verifies nothing
-------------------------------------------------------
This repository has already shipped a resident-plane test whose six fixtures
all ran below the gate that enables the plane, so it compared the fallback
against itself and asserted nothing about the thing it named. A poison test
has the same shape of hazard in a sharper form: "assert the read raises" passes
just as well when the arming silently did nothing and the read raised for an
unrelated reason (a bad node id, an empty table, a typo in the fixture).

So every negative assertion here is paired with two things:

1. **A positive control.** The same read, on the same table, at the same node,
   before the poison is armed and again after it is cleared -- asserted to
   succeed *and* to return the right window. A refusal that refuses everything
   would fail those.
2. **An observable that can only be true if the poisoned path ran.**
   `LeafRangeTable.desc_partitions` counts armings since the last
   `reset_root`. It feeds no decision; it exists so that these tests assert
   the arming happened rather than infer it from a raise. Every test that
   expects a refusal asserts the counter first.

What this file does NOT prove, stated plainly
---------------------------------------------
That `GpuActiveRows.enqueue_partition_desc` calls the arming function. That
call needs a `DeviceContext`, this file opens no device, and the naming rule in
`tools/run_tests.sh` puts a file not named `test_gpu_*` in the CPU-only set,
where a device test would either fail or compile a kernel a CPU-only runner
cannot build. Two things stand in for the test that cannot be written here:

- The wiring is load-bearing rather than advisory.
  `begin_descriptor_partition` returns the validated row bound that
  `enqueue_partition_desc` derives its launch geometry from, so an edit that
  drops the arming does not compile. That is a construction argument, not a
  measurement, and it is labeled as one.
- The two bound refusals asserted in
  `test_a_refused_bound_arms_nothing` are now defined in exactly one place,
  inside `begin_descriptor_partition`. Any accelerator test that calls
  `enqueue_partition_desc` with a bound of zero and sees "a descriptor
  partition needs a positive row bound" has thereby executed the arming
  function; that check belongs in a `test_gpu_*` file and is noted here so
  whoever adds it knows it is available for free.
"""

from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.gpu_active_rows import LeafRangeTable

# The sentence fragment every staleness refusal carries. Asserted rather than
# assumed so that a test expecting the poison cannot be satisfied by a
# bounds error or an empty-table error that happens to raise at the same call.
comptime _STALE = "device-resident partition owns the active-row windows"


def _grown_tree(n_rows: Int, n_active: Int) raises -> LeafRangeTable:
    """A table carrying a small finished tree, built the way the shipping
    partition builds it: root, then two splits, giving five nodes of which
    three are live.

        node 0  [0, n_active)  -> split at 12
        node 1  [0, 12)        -> split at 5
        node 2  [12, n_active)  live
        node 3  [0, 5)          live
        node 4  [5, 12)         live

    Written out because the tests below assert these exact numbers after a
    replay, and a fixture whose expected values are recomputed from the code
    under test asserts nothing.
    """
    var table = LeafRangeTable(n_rows)
    table.reset_root(n_active)
    _ = table.split(0, 1, 2, 12)
    _ = table.split(1, 3, 4, 5)
    return table^


def _assert_grown_tree_windows(table: LeafRangeTable) raises:
    """The five windows `_grown_tree` describes, read through the public
    accessors. This is the positive control: it is what a *correct* read of a
    published table returns, and it has to succeed wherever it is called."""
    assert_equal(table.n_nodes(), 5)
    assert_true(table.get(0).is_empty())
    assert_true(table.get(1).is_empty())
    assert_equal(table.get(2).begin, 12)
    assert_equal(table.get(2).end, 40)
    assert_equal(table.get(3).begin, 0)
    assert_equal(table.get(3).end, 5)
    assert_equal(table.get(4).begin, 5)
    assert_equal(table.get(4).end, 12)
    assert_equal(table.total_active(), 40)
    table.check_invariants()


def test_a_stale_table_passes_every_structural_check() raises:
    """The reason the poison is recorded rather than detected.

    This is the exact table a device-owned tree used to leave behind: the root
    seeded by `begin_tree` and not one window moved, while on the device the
    tree grew five nodes. Every check the struct can make on it passes, and a
    reader gets a window that is in bounds, non-empty, and completely wrong
    about which leaf owns those rows -- which is how the root's value came to
    be added to every row's raw score.
    """
    var stale = LeafRangeTable(100)
    stale.reset_root(40)
    # Nothing here is malformed. That is the finding.
    stale.check_invariants()
    assert_equal(stale.n_nodes(), 1)
    assert_equal(stale.total_active(), 40)
    var window = stale.get(0)
    assert_equal(window.begin, 0)
    assert_equal(window.end, 40)
    assert_equal(window.count(), 40)
    # And that is the whole tree, as far as any host reader can tell: the
    # four nodes the device grew are not absent-and-flagged, they are simply
    # not there, so a loop over `n_nodes()` visits the root and stops.
    with assert_raises():
        _ = stale.get(1)


def test_arming_is_observable_and_not_a_no_op() raises:
    """The counter moves, the state flips, and the bound comes back.

    Asserted before any refusal is asserted anywhere else in this file,
    because every other test in it reads as a tautology if the arming is a
    no-op.
    """
    var table = LeafRangeTable(100)
    table.reset_root(40)
    assert_false(table.is_resident_owned())
    assert_equal(table.desc_partitions, 0)

    # The bound is returned, not swallowed: this is the value
    # `enqueue_partition_desc` sizes its grid from, which is what makes the
    # arming impossible to delete without breaking the compile.
    assert_equal(table.begin_descriptor_partition(40), 40)
    assert_true(table.is_resident_owned())
    assert_equal(table.desc_partitions, 1)

    # Re-arming an already-poisoned table is a no-op apart from the counter.
    # Every step of a resident tree calls it, dead steps included.
    assert_equal(table.begin_descriptor_partition(40), 40)
    assert_true(table.is_resident_owned())
    assert_equal(table.desc_partitions, 2)


def test_every_window_accessor_refuses_while_poisoned() raises:
    """Reads that return a window, or a count of windows, or a verdict about
    windows, all refuse -- and each refusal is the staleness refusal, not some
    other error the same call could have raised."""
    var table = _grown_tree(100, 40)
    # Positive control: the table is readable, and correct, before the poison.
    _assert_grown_tree_windows(table)

    _ = table.begin_descriptor_partition(40)
    assert_equal(table.desc_partitions, 1)

    with assert_raises(contains=_STALE):
        _ = table.get(0)
    with assert_raises(contains=_STALE):
        _ = table.get(3)
    with assert_raises(contains=_STALE):
        _ = table.n_nodes()
    with assert_raises(contains=_STALE):
        _ = table.total_active()
    with assert_raises(contains=_STALE):
        table.check_invariants()
    # An out-of-range node while poisoned reports the staleness, not the
    # range: the caller's real problem is that it is reading this table at
    # all, and telling it "no such leaf" would send it looking in the wrong
    # place.
    with assert_raises(contains=_STALE):
        _ = table.get(99)


def test_the_replay_writer_is_not_refused() raises:
    """`split` has to work while poisoned, because it is the call the replay
    makes to lift the poison. A refusal that refused it too would make the
    table permanently unreadable, which is a different bug and one this test
    would rather find here than in a fit."""
    var table = LeafRangeTable(100)
    table.reset_root(40)
    _ = table.begin_descriptor_partition(40)

    # The device's commit log, replayed: two commits, in commit order.
    var left0 = table.split(0, 1, 2, 12)
    assert_equal(left0.begin, 0)
    assert_equal(left0.end, 12)
    var left1 = table.split(1, 3, 4, 5)
    assert_equal(left1.begin, 0)
    assert_equal(left1.end, 5)
    # Still poisoned: writing windows is not the same as declaring the table
    # published, and only `end_descriptor_partition` does the second.
    assert_true(table.is_resident_owned())
    with assert_raises(contains=_STALE):
        _ = table.get(3)

    # `split`'s own refusals still hold on the replay, which is what makes a
    # replay of a log that does not describe a real growth an error rather
    # than a silent rewrite.
    with assert_raises():
        _ = table.split(2, 3, 5, 1)


def test_publish_clears_the_poison_and_the_windows_are_right() raises:
    """The full cycle: readable, poisoned, replayed, readable again -- with
    the same window assertions on both ends, so a poison that never lifted and
    a poison that lifted onto garbage both fail."""
    var table = _grown_tree(100, 40)
    _assert_grown_tree_windows(table)

    # A second tree: reseed the root, then let the resident plane take over.
    table.reset_root(40)
    assert_equal(table.desc_partitions, 0)
    _ = table.begin_descriptor_partition(40)
    _ = table.begin_descriptor_partition(40)
    assert_equal(table.desc_partitions, 2)
    with assert_raises(contains=_STALE):
        _ = table.get(0)

    # The replay, then the publish. Five nodes is `snap.next_node` for a tree
    # of two commits: one root plus two children per commit.
    _ = table.split(0, 1, 2, 12)
    _ = table.split(1, 3, 4, 5)
    table.end_descriptor_partition(5)

    assert_false(table.is_resident_owned())
    _assert_grown_tree_windows(table)


def test_publish_refuses_a_replay_that_left_the_tail_of_the_tree_out() raises:
    """A commit log that came home truncated -- `_pick_and_commit_kernel`
    stops appending once the log is full -- leaves the last nodes with no
    window at all. Un-poisoning then would be a lie, so it raises instead."""
    var table = LeafRangeTable(100)
    table.reset_root(40)
    _ = table.begin_descriptor_partition(40)
    # One commit replayed out of the two the tree actually made.
    _ = table.split(0, 1, 2, 12)
    with assert_raises(contains="the tree ended with 5"):
        table.end_descriptor_partition(5)
    # Refused, and therefore still poisoned. A failed publish must not leave
    # the table readable, or the refusal would only have delayed the wrong
    # read by one call.
    assert_true(table.is_resident_owned())
    with assert_raises(contains=_STALE):
        _ = table.get(2)

    # Positive control on the same fixture: replay the missing commit and the
    # same publish succeeds. Without this, the test above would pass against
    # an `end_descriptor_partition` that rejected everything.
    _ = table.split(1, 3, 4, 5)
    table.end_descriptor_partition(5)
    assert_false(table.is_resident_owned())
    _assert_grown_tree_windows(table)


def test_a_hole_in_the_middle_of_the_log_is_invisible_here() raises:
    """The limit of what the table can check, asserted rather than assumed.

    An earlier draft of `end_descriptor_partition` claimed the invariant
    caught a commit missing from the middle of the log. It does not, and this
    test is the proof, so that nobody removes the check that does catch it on
    the strength of a docstring.

    Replaying commits (0 -> 1, 2) and (2 -> 5, 6) while the commit that would
    have split node 1 into 3 and 4 goes missing produces a table that is
    entirely well formed: `_grow_to` back-fills ids 3 and 4 as empty so the
    node count is right, and node 1 simply stays a live leaf owning the
    twelve rows its children should have owned, so coverage is right too. The
    publish therefore succeeds on a table that describes a different tree
    from the one the device grew.

    What sees it is one level up: `gpu_resident_round._publish_row_ranges`
    checks `2 * len(commit_order) + 1 == next_node` before replaying
    anything, because every commit creates exactly two nodes. Asserted here as
    arithmetic, since this file opens no device and cannot run that function:
    two logged commits account for five nodes, and the tree ended with seven.
    """
    var table = LeafRangeTable(100)
    table.reset_root(40)
    _ = table.begin_descriptor_partition(40)
    _ = table.split(0, 1, 2, 12)
    _ = table.split(2, 5, 6, 20)
    assert_equal(len(table.ranges), 7)

    # Well formed, and wrong. Both halves matter.
    table.end_descriptor_partition(7)
    assert_false(table.is_resident_owned())
    table.check_invariants()
    assert_equal(table.total_active(), 40)
    # Node 1 looks like a live leaf holding twelve rows; on the device it is
    # an internal node and those rows belong to 3 and 4.
    assert_equal(table.get(1).count(), 12)
    assert_true(table.get(3).is_empty())
    assert_true(table.get(4).is_empty())

    # The upstream identity, which is what rejects this log before a single
    # window is written.
    var logged_commits = 2
    var tree_nodes = 7
    assert_true(2 * logged_commits + 1 != tree_nodes)
    # And it accepts the complete log for the same tree: three commits.
    assert_equal(2 * 3 + 1, tree_nodes)


def test_a_refused_bound_arms_nothing() raises:
    """The two bound refusals live in the arming function, so that there is no
    version of the descriptor partition that validates its bound and forgets
    to arm. A refused bound launches nothing, so it invalidates nothing, and
    the table stays readable."""
    var table = _grown_tree(100, 40)
    with assert_raises(
        contains="a descriptor partition needs a positive row bound"
    ):
        _ = table.begin_descriptor_partition(0)
    with assert_raises(contains="the row bound exceeds the row buffer"):
        _ = table.begin_descriptor_partition(101)
    assert_false(table.is_resident_owned())
    assert_equal(table.desc_partitions, 0)
    _assert_grown_tree_windows(table)


def test_reset_root_makes_the_table_current_again() raises:
    """A new tree re-establishes the one fact the poison denies, so it clears
    it.

    `GpuActiveRows.begin_tree` reseeds the device row buffer in the same call,
    after which node 0 genuinely owns `[0, n_active)` and no other node
    exists. Keeping the poison across that would refuse the shipping partition
    plane -- which is entirely correct after a `begin_tree` -- for the rest of
    the session. The window in which a stale read does damage runs from the
    last descriptor partition to the next `begin_tree`, and
    `update_raw_device` lives inside it by contract.
    """
    var table = _grown_tree(100, 40)
    _ = table.begin_descriptor_partition(40)
    assert_true(table.is_resident_owned())
    with assert_raises(contains=_STALE):
        _ = table.get(0)

    table.reset_root(25)
    assert_false(table.is_resident_owned())
    assert_equal(table.desc_partitions, 0)
    assert_equal(table.n_nodes(), 1)
    assert_equal(table.get(0).begin, 0)
    assert_equal(table.get(0).end, 25)
    assert_equal(table.total_active(), 25)
    table.check_invariants()


def test_a_one_leaf_tree_publishes_with_no_commits() raises:
    """A resident tree that never commits still calls the descriptor
    partition on its dead steps, so it still arms, and its publish has an
    empty commit log. One node, the root, and the invariant holds."""
    var table = LeafRangeTable(64)
    table.reset_root(64)
    _ = table.begin_descriptor_partition(64)
    assert_equal(table.desc_partitions, 1)
    with assert_raises(contains=_STALE):
        _ = table.get(0)
    table.end_descriptor_partition(1)
    assert_false(table.is_resident_owned())
    assert_equal(table.n_nodes(), 1)
    assert_equal(table.get(0).count(), 64)


def test_publish_refuses_a_nonsense_node_count() raises:
    var table = LeafRangeTable(64)
    table.reset_root(64)
    _ = table.begin_descriptor_partition(64)
    with assert_raises(contains="a finished tree holds at least the root"):
        table.end_descriptor_partition(0)
    assert_true(table.is_resident_owned())


def test_an_empty_split_replays_and_publishes() raises:
    """A device-owned split can send every row one way; both children still
    get a window and one of them is empty at the other's edge. The replay has
    to reproduce that, because an empty leaf is a first-class state here and
    the publish checks coverage."""
    var table = LeafRangeTable(32)
    table.reset_root(32)
    _ = table.begin_descriptor_partition(32)
    _ = table.split(0, 1, 2, 32)
    _ = table.split(1, 3, 4, 0)
    table.end_descriptor_partition(5)
    assert_false(table.is_resident_owned())
    assert_equal(table.get(2).count(), 0)
    assert_equal(table.get(3).count(), 0)
    assert_equal(table.get(4).count(), 32)
    assert_equal(table.total_active(), 32)
    table.check_invariants()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
