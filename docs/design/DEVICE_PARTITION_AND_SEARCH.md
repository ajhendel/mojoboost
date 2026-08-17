# The device split search and the device row partition: what they cost

Written 2026-08-17 by the lane that owns `src/mojotrees/gpu_split_search.mojo`
and `src/mojotrees/gpu_active_rows.mojo`. Nothing in it was built or run by
this lane; every number is either read out of a committed results file, counted
statically off the source, or derived from committed numbers by arithmetic that
is shown. Derived numbers are labelled ESTIMATE and the derivation is written
out so it can be checked or discarded.

The reason the file exists: this lane was asked to attribute the cost of two
stages before optimizing them, and the attribution turned out to contradict the
plan that motivated the lane. That is worth a file rather than a paragraph in a
report.

## 1. The instrument, and the one thing it cannot see

`bench/results/profile_2026-08-15/phase_gpu_1m.txt` and its `_fenced` twin are
the only stage-level attribution this repository has. Shape 1,000,000 x 50, 100
trees, 6,100 nodes, 3,100 splits. Provenance: the **host-driven** device plane,
one readback per split, which is what `syncs=3100` in the `transfer` row says.
The device-resident plane had not shipped, so this is not the shape the default
takes today. It is still the only phase decomposition in the tree.

`split_search` records **`syncs = 0` in both modes**. So the split search is
never fenced, in either mode, and the nanoseconds charged to it are host enqueue
time only. Its device time drains at the next `transfer`. Any reading of "split
search is 2.4 percent of the round" as a statement about device work is
therefore wrong, and the results file that quotes 2.4 percent does not say so.

## 2. Attribution, per split, at 1,000,000 x 50

Measured, straight off the two files:

| per split | fenced | async |
|---|---|---|
| histogram | 779.7 us | 20.0 us (enqueue only) |
| partition | 288.3 us | 44.0 us (enqueue only) |
| split_search | 37.8 us | 34.5 us (**never fenced, enqueue only**) |
| transfer | 445.5 us | 998.1 us |
| wall | 4.905 s | 3.518 s |

ESTIMATE, from those five measured numbers and nothing else. Write `f` for the
cost of one added fence. In fenced mode the drain before each readback holds
only the search, because the histogram and the partition each fenced already;
in async mode it holds the histogram, the search and the previous partition. So

    async transfer - fenced transfer = hist_dev + part_dev
    998.1 - 445.5 = 552.6

    fenced histogram = hist_dev + f = 779.7
    fenced partition = part_dev  + f = 288.3
    => 779.7 + 288.3 - 2f = 552.6  =>  f = 257.7 us

    hist_dev  = 522.0 us per split
    part_dev  =  30.6 us per split
    readback + search_dev = 445.5 us per split

Over a 100-tree fit, against the 3.518 s async wall:

| stage | device time per fit | share |
|---|---|---|
| histogram | **1.618 s** | 46.0% |
| partition | **0.092 s** | 2.6% |
| split search | **under 0.08 s** | under 2.3% |
| readback wait | ~1.30 s | ~37% |

The split-search bound is the interesting one. `readback + search_dev` is
445.5 us and the readback alone was measured independently at **606 us median**
by `docs/METAL_TIMELINE.md` and registered at ~458 us in
`PROFILE_PROTOCOL.md`. Two instruments therefore leave the split search
somewhere between zero and about 25 microseconds per split. It is not where the
time is, and no rearrangement of the scan can be worth more than about 2
percent of that round.

The `f = 257.7 us` figure is a check on the derivation rather than an output:
fencing added 6,100 syncs and 1.387 s of wall, which is 227 us each. Two
independent routes to within 12 percent.

## 3. What the two stages own, in launches

Counted statically off `gpu_resident_round`'s own census, which this lane
re-counted rather than trusting. Default arms, 31 leaves, so 30 growth steps:

    per tree, once                                    7
    per growth step x30                               9
    the one round trip                                1
                                                    278

Of those 278, this lane's two stages own **122**:

| stage | per tree | share of 278 |
|---|---|---|
| split search (scan + reduce) | 62 | 22.3% |
| partition (flag scan + scatter) | 60 | 21.6% |

So the lane is 2.6 percent plus under 2.3 percent of device time and **44
percent of the command buffers**, on a queue measured flat at 6 to 7
microseconds per enqueue through 64 and 14 to 17 beyond it (`session3`), which
this plane is past for most of every tree. Launch count is the lever this lane
holds; kernel shape is not.

## 4. Row compaction: it exists, it is order-preserving, and the arithmetic
predicts it loses

**It exists.** `GpuActiveRows.set_row_compaction`, three kernels
(`_compact_build_kernel`, `_compact_scatter_kernel`,
`_compact_copy_back_kernel`), reached from `train_gpu` at two call sites through
`MOJOTREES_GPU_ROW_COMPACTION`, default off. Wired to the descriptor path as
well as the host path, so the shipped device-resident plane can reach it.
Preconditions: quantized gradients on (they are, by default), blocked layout
off, packed bin layout off.

**It is order-preserving, and this is verifiable by reading rather than by
running.** `_compact_scatter_kernel` consumes the *same* `offsets` and
`block_sums` that `_scatter_kernel` consumes, written by the same flag pass,
and computes `dst` with the same expressions. The two are one function of one
input evaluated twice. `_scatter_kernel`'s destination rule is a stable
partition -- left rows keep their relative order ascending from `begin`, right
rows ascending from `begin + carry`, both ranks monotone in `j` -- so the
compacted plane holds, at position `j`, exactly the row `rows[j]` names, in the
order `rows` names them. The histogram then reads `cbins[f * n_rows + j]`
against an identity index, which is by definition the byte
`bins[f * n_rows + rows[j]]` the un-compacted launch would have read at the
same step of the same loop. **Bit-identical by construction, not by an argument
about the order of a sum.** No summation order changes anywhere, so the
reproducibility guarantee is untouched.

**And the arithmetic says it loses.** This is the finding that matters, because
it contradicts the module docstring's own optimism. That docstring prices one
split: `4 * L * (nf + 8)` bytes moved against a sparse column read, and
concludes the crossover is "around depth two or three". What it never does is
multiply by how many times each stage touches a row over a whole tree, and
those two counts are in the profile:

    partition row-touches per tree   P = 639,494,863 / 100 = 6.395 M
    histogram row-touches per tree   H = 230,601,070 / 100 = 2.306 M
    P / H = 2.77

`P` exceeds `H` by 2.77x because sibling subtraction means only the smaller
child's histogram is accumulated while the partition rewrites the whole parent
window. Compaction pays `nf` bytes per *partition* touch and saves gather
penalty on `nf` bytes per *histogram* touch, so the ratio is against it before
any bandwidth number is chosen.

ESTIMATE, at 1,000,000 x 50, per tree:

    compaction moves   4 * (nf + 8) * P = 232 * 6.395 M = 1.484 GB
      (scatter reads and writes the window; the copy-back reads and writes it
       again; the compact copy-back is NOT folded, unlike the row copy-back)
    plus one rebuild    2 * (nf + 8) * n_rows = 0.116 GB
    total               1.600 GB per tree, 160 GB per 100-tree fit

    at 100 GB/s achieved device-local bandwidth: 16.0 ms per tree, 1.60 s per fit

Against that, what it removes. `session3`'s histogram decomposition measured
the **gather at 56.7 percent** of the histogram kernel, resolved. Applying that
to the 1.618 s of section 2 gives **0.92 s per fit** of gather (MEASURED
fraction x ESTIMATED base, so treat it as an estimate). And compaction does not
remove all of it: it converts a scattered gather into a coalesced one, so the
credit is some fraction of 0.92 s, not 0.92 s.

**So the registered prediction is that row compaction as built is roughly 1.7x
to 2.5x underwater on the leaf-wise plane at this shape.** Registered before
the run, per `PROFILE_PROTOCOL.md`, so that a run which contradicts it is
evidence and not a rewrite.

The A/B should still be taken. It is one pair of processes, the prediction is
falsifiable, and this repository has had four registered predictions refuted in
three days -- including one that priced thirteen copies at 0.64 s and measured
0.016. An arithmetic prediction is not a measurement.

### Two changes that would move the prediction

1. **Remove the compact copy-back.** It is half of the `4 *` above. The row
   buffer cannot ping-pong, because a partition rewrites one window and the
   rest of the array lives in the other buffer -- but the compacted planes can,
   because **a node's plane parity is its depth parity, exactly**. Every
   partition writes the window into the alternate plane, and a child's window
   is written by its parent's partition, so `plane(node) = plane(parent) XOR 1
   = depth(node) mod 2`. The histogram would select `cbins` or `cbins_alt` by
   the node's depth, and the copy-back would not exist. Cost: the descriptor
   needs one depth word (`gpu_tree_tables` writes it), and
   `enqueue_range_histogram` needs the depth from its host callers. That is two
   files this lane may not write, which is why it is written down here.
   It also removes one command buffer per split under compaction.

2. **Read the flag pass out of the compacted plane.** Built by this lane,
   `MOJOTREES_GPU_COMPACT_FLAG_READ`, default off, see section 6. It does not
   change the byte count above but it removes the partition's own scattered
   gather, which the compaction arithmetic never credited itself with.

## 5. The scan tail: 64 is right, and the reduce is the bubble

**Is `WIDE_SCAN_THREADS = 64` right for this hardware?** Yes, by arithmetic,
and the arithmetic is recorded at the constant itself. The wide scan splits
`n_bins` chunkwise across the threadgroup, so at the 256-bin ceiling 64 threads
walk four bins each while the surrounding block work is fixed at three chunk-sum
reductions, three exclusive prefixes and three maxima over the full width.
Widening to 128 halves a four-step walk and pays a wider reduction for it.
Narrowing to 32 is on the do-not-retry list: a 32-wide scan block was built,
measured inside the noise on the oblivious plane and reverted, and
`block.prefix_sum` constrains block size to a multiple of the repository's
`WARP_GRANULARITY` of 64 in any case. Mojo 1.0 has no warp-level primitives at
all -- only `block` -- so there is no sub-block cooperation to reach for, and
both the Modular docs page and the docs MCP server are wrong about this.

**The bigger finding about the scan is that the shipped leaf-wise default does
not use it.** `wide_scan_requested()` is `MOJOTREES_GPU_SPLIT_WIDE == "1"`,
default off, so `_launch_search` takes the `_scan_slot_kernel` branch at
`block_dim=1`: one lane per (leaf, feature), 200 single-lane threadgroups at
the reference shape. That is the same shape the oblivious lane replaced today
and measured at 4.5 percent, resolved and bit-identical. The leaf-wise arm is
built, tested for bit-identity in `tests/test_gpu_split_search.mojo`,
reachable from the resident loop (`_launch_child_search` passes
`searcher.wide_scan`), and **has never been run**. Zero code is needed to price
it. Given section 2's bound of under 0.08 s of search device time per fit, the
honest expectation is under 2 percent, and the reason to take it anyway is that
it costs one pair of processes.

**Is the final reduce a bubble? Yes.** `_reduce_slots_block_kernel` is launched
at `grid_dim = n_records`, and on the leaf-wise resident plane `n_records` is
**two**. One command buffer carries two threadgroups reducing `widest_slots`
records each, which is one launch in nine per growth step for well under a
percent of the step's arithmetic. The fix is not the width: it is folding the
cross-slot reduction into the kernel that consumes it, which is
`gpu_tree_tables`'s record filing -- the very next command buffer, already one
threadgroup, already holding the two record slots. Counted: **30 command
buffers per tree, about 3,000 per 100-tree fit**, the same size as the
copy-back fold that shipped and took the tree from 308 to 278. Not built here
because the consuming kernel is in another file.

Fusing the scan and the reduce *inside this file* was considered and rejected on
arithmetic. Cross-block reduction needs either a grid-wide barrier, which Metal
does not offer and Mojo 1.0 exposes no primitive for, or a device atomic over a
packed key, which cannot carry a Float32 gain and a tie-breaking slot ordinal in
32 bits without quantizing the gain and breaking bit-identity. The remaining
shape -- one threadgroup per record looping over every feature -- fuses the two
launches but confines the work to `n_records = 2` threadgroups, i.e. two of ten
cores, in exchange for one enqueue. That is a worse trade than the launch is
worth.

## 6. What this lane changed

Both changes are in `src/mojotrees/gpu_active_rows.mojo`.

**The compacted flag read**, `MOJOTREES_GPU_COMPACT_FLAG_READ=1` /
`set_compact_flag_read(True)`, **default off**, and inert unless
`row_compaction_live()` is also true. `compact_flag_read_live()` is the single
predicate. The flag pass reads `cbins[f * n_rows + begin + j]` instead of
`bins[f * n_rows + rows[begin + j]]`; under the compaction invariant that is
the same byte, so every flag, prefix, packed offset, block sum and permutation
is unchanged value for value, and no summation order moves. What it removes is
the module docstring's own "one random-access read of the partition", plus the
permutation load that feeds it: two loads per row, one of them a scattered
byte, become one contiguous byte.

**An early return for scatter blocks that own nothing**, unswitched, in
`_scatter_kernel`, `_scatter_prim_kernel` and `_compact_scatter_kernel`. It is
argued rather than gated because it provably cannot change a written value: the
blocks it returns store nothing in either branch, and the head scan over
`block_sums` only reads. Block 0 is exempt in the two kernels that store
`total`, since `carry` is only correct after the scan. The condition is
block-uniform, so the whole threadgroup leaves together and no barrier or
collective is reached by part of a block -- the same early exit
`_flag_scan_kernel` has taken since the tiled grid landed.

It matters because `enqueue_partition_desc` cannot size its grid to a window
the host has not read back, so it launches a grid covering the whole active
prefix. At 800,000 rows, 256 threads and the default cap of 256, that is 241
blocks with a chunk of 3,328; a 62,000-row window leaves 222 of them owning
nothing, and before this every one of those ran a full prefix sum over 241 block
sums to discover it. ESTIMATE of the win: 2 to 5 us per split, 6 to 15 ms per
100-tree fit, which is **below this repository's noise floor**. It is included
because it is free and because it is the precondition for ever raising
`partition_block_cap`, whose cost is quadratic in the block count precisely
because of that scan.

## 7. The commands

Interleaving inside one process is only available for arms the harness names,
and neither of these is named, so both are alternating-process pairs on the
`PROFILE_PROTOCOL.md` quiet-box precondition: no lane, build, agent or compile
running, `uptime` and top processes recorded before the first pair and after the
last, at least five repeats each, and the canary either side.

Row compaction, the section 4 prediction, at the shape section 2 attributes:

    MOJOTREES_GPU_ROW_COMPACTION=1 pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device
    MOJOTREES_GPU_ROW_COMPACTION=0 pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device

Compaction plus the compacted flag read, which is the arm the two switches are
separate for. Take it only after the pair above, so the reorder is priced before
the reorder-plus-flag-read:

    MOJOTREES_GPU_ROW_COMPACTION=1 MOJOTREES_GPU_COMPACT_FLAG_READ=1 \
      pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device

Verify the arm engaged rather than assuming the variable was spelled right.
`MOJOTREES_GPU_COMPACTION_TRACE=1` writes one record per tree saying whether the
arm was on and how many launches it had issued; an off arm reports zero for the
whole fit.

The leaf-wise wide scan, section 5:

    MOJOTREES_GPU_SPLIT_WIDE=1 pixi run -e bench bench-train-gpu 799110 100 reg 5 gpu-device
    MOJOTREES_GPU_SPLIT_WIDE=0 pixi run -e bench bench-train-gpu 799110 100 reg 5 gpu-device

The scatter early return is unswitched, so it has no pair. What it needs instead
is the correctness check, which is `device_agreement` plus
`tests/test_gpu_partition_launches.mojo`, whose whole job is to assert the two
partition arms produce the identical permutation element for element.
