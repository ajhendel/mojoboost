# Algorithm 25: GPU-optimized binned data layout and compression

Status: designed and implemented in isolation. Nothing central was edited, no
threshold was added, and no existing path changed behavior.

Files added by this lane:

- `src/mojoboost/gpu_bin_packing.mojo` (bit primitives)
- `src/mojoboost/gpu_binned_layout.mojo` (plans, policy, cost model)
- `bench/apple/bin_layout_plan.json` (the measurement plan)
- `handoffs/algorithm_25_binned_layout.md` (this file)

Nothing was executed. No Mojo, pixi, Python, test, build, benchmark, or
profiler run was performed by this lane, so no line below is a measurement.
Everything here is static reasoning against the source, and every quantity
that would need a device to establish is called out as unmeasured.

Not edited: `binning.mojo`, `histogram_gpu.mojo`, `train_gpu.mojo`,
`gpu_active_rows.mojo`, `gpu_predict.mojo`, the sparse modules, Python,
bindings, tests, docs, packaging, workflows.

## The finding that should drive the integration order

The current histogram kernel moves, per (row, feature):

| bytes | what | why |
| --- | --- | --- |
| 1 | the bin | `bins[f * n_rows + r]` |
| 4 | the active-row index | `rows[begin + j]` |
| 4 | the gradient | `grad[r]` |
| 4 | the hessian | `hess[r]` |

`grid.x` is the feature, so the last three are re-read once per feature. The
bin is **one byte of thirteen**. Compressing it to four bits removes at best
3.8% of that traffic and adds a decode to every element.

Blocking removes the other twelve. A threadgroup that owns `G` features
loads the row index, gradient, and hessian once and accumulates `G` partial
histograms from them, so the per-(row, feature) cost becomes `w/8 + 12/G`
bytes: 4 at `G = 4` and 3.5 if those four features also pack to 4 bits,
against 13 today.

So the honest ordering is **blocking first, compression second**, and
compression's strongest argument is not bandwidth at all:

> A feature stored at width `w` can only produce bins `0 .. 2^w - 1`, so its
> shared-memory partial histogram needs `2^w` slots and not `n_bins`. At
> `n_bins = 256` an 8-bit feature costs 3072 shared bytes and a 4-bit feature
> costs 192, so a 4-bit block can be **sixteen times wider** under the same
> threadgroup budget.

Blocking is bounded by shared memory; compression is what raises the bound.
That is the composition worth building, and `max_block_for_shared` is where
it is computed. It is also a prediction, not a result: see Q4 in the plan.

## The layout family, as one construction

Every layout is the same thing with different parameters. Features are
partitioned into **blocks**; a block is stored **row-major inside itself**,
at one common bit width, as a single packed stream; blocks follow one
another. One formula addresses all of them:

```
index(f, r) = r * G[b] + lane(f)        b = block_of[f]
base(f)     = block_offset[b]
width(f)    = block_width[b]
```

`G = 1` is feature-major. `G = n_features` with one block is row-major.
Anything between is feature-blocked. Width 8 is uncompressed.

### Diagrams

```
FEATURE-MAJOR (G = 1)                       today's buffer at width 8
  feature 0                feature 1
  +----------------------+----------------------+
  | r0 r1 r2 ... rN-1    | r0 r1 r2 ... rN-1    |
  +----------------------+----------------------+
  one node reads one column, gathered at its rows
  no waste under colsample: an inactive column is never touched


ROW-MAJOR (one block, G = n_features)
  row 0                    row 1
  +----------------------+----------------------+
  | f0 f1 f2 ... fF-1    | f0 f1 f2 ... fF-1    |
  +----------------------+----------------------+
  one node reads count contiguous runs of F*w/8 bytes
  what gpu_predict already wants; needs F*n_bins*12 shared bytes to train


FEATURE-BLOCKED (G = 4)
  block 0 (f0..f3)                       block 1 (f4..f7)
  +---------+---------+-----   ----+     +---------+-----
  |r0:f0f1f2f3|r1:f0f1f2f3| ...    |     |r0:f4f5f6f7| ...
  +---------+---------+-----   ----+     +---------+-----
  row side read once per block instead of once per feature
  wastes (G - active)/G of bin bytes under colsample


PACKED, width 4, inside one block of 4 features, one row
  byte 0           byte 1
  +-------+-------+-------+-------+
  |  f1   |  f0   |  f3   |  f2   |    low nibble first
  +-------+-------+-------+-------+

PACKED, width 5, inside one block, first three elements
  bit    0    5    10   15
         |----|----|----|
  byte 0 [ e0 ][e1 ]e1'|      e1 straddles bytes 0 and 1
  byte 1      |[ e1'][ e2 ]|  e2 straddles bytes 1 and 2
```

### Byte formulas

```
G(b)        = block_size[b]
elems(b)    = n_rows * G(b)
data(b)     = ceil(elems(b) * block_width[b] / 8)
pad(b)      = 0 if block_width[b] == 8 else 1
stream(b)   = data(b) + pad(b)

offset(0)   = 0
offset(b+1) = align_up(offset(b) + stream(b), BLOCK_ALIGN_BYTES)
total       = offset(last) + stream(last)

index(f, r) = r * G(block_of[f]) + lane(f)
byte(f, r)  = block_offset[block_of[f]] + (index(f, r) * w) >> 3
shift(f, r) = (index(f, r) * w) & 7
value       = (buf[byte] | (buf[byte+1] << 8)) >> shift & ((1 << w) - 1)
```

## Decode invariants

These are functions in the source, not comments, so they can be run:

| invariant | where | what it guarantees |
| --- | --- | --- |
| straddle | `assert_straddle_invariant` | `shift + w <= 15`, so an element spans at most two bytes and the decode is a fixed, branch-free instruction sequence |
| window in bounds | `assert_window_inside` | the last element's window ends inside `packed_stream_bytes`, which is what the tail pad is for |
| byte identity | `assert_byte_identity_invariant` | at width 8, element `i` is byte `i`, shift 0, no straddle, no pad |
| footprint disjointness | `streams_disjoint`, `BinLayoutPlan.check_blocks_disjoint` | two blocks never share a writable byte |
| round trip | `column_roundtrips`, `matches_dense` | unpack(pack(x)) == x for every cell |
| passthrough addresses | `BinLayoutPlan.check_passthrough_offsets` | a width-8 feature-major plan really does compute `f * n_rows + r` |

Three of them deserve their reasons stated.

**Endianness.** The two-byte window is composed arithmetically,
`buf[b] | (buf[b+1] << 8)`, never by reinterpreting memory as a 16-bit word.
So the representation is byte-order independent. A `UInt32` bit stream, the
obvious alternative, would bake the host's endianness into the device buffer
and would need 32-bit loads at arbitrary byte offsets, which is not portably
expressible across Metal, CUDA, and HIP. That is why the byte-pair form was
chosen over the word form, and it is the one place where the slower-looking
option is the correct one.

**The width-8 specialization is load-bearing, not an optimization.** At
width 8 the shift is zero, the mask is the identity, and the second window
byte contributes nothing. `unpack_value` and `pack_value` both branch on it
and take the one-byte path. That is what makes the uncompressed layout the
*same code* as the packed one rather than a parallel implementation, and it
is what lets a width-8 stream need no pad, which in turn is what lets the
baseline plan be `BinnedMatrix.bins` itself with `block_offset[f] ==
f * n_rows`. Without the specialization the baseline would carry pads, would
not be the existing buffer, and would be charged a packing pass it does not
perform, tilting every comparison toward the candidate.

**The writable footprint is one byte wider than the data, below width 8.**
`pack_value` is a read-modify-write of both window bytes, so the last element
of a packed stream *writes* the byte past its data. Two packed streams laid
back to back therefore share a byte even though their data does not overlap,
and packing them concurrently would lose updates. `BinLayoutPlan` pads every
block, not only the last, for exactly this reason. Alignment alone does not
supply the separation: a block whose data is a multiple of the alignment ends
flush against the next.

## Bin ids are never renumbered

This is the semantic contract, and it is what keeps the change to a
reformatting of bytes:

- `BinnedMatrix.missing_bin[f]` keeps its value; a missing row is still found
  by the same equality test. A feature's width is derived to *cover* its
  missing bin (`declared_bins_from_mapper` uses `missing_bin[f] + 1` as a
  lower bound), never to exclude it.
- Categorical bins keep their positions in the 256-bit `CatBitset`, so
  `RowRouting.categorical` and `Tree.goes_left` are untouched.
  `check_categorical_widths` refuses a width too narrow for a feature's
  highest category bin.
- `threshold_bin` keeps its meaning, so trees, model dumps, serialization,
  and LightGBM parity are unaffected.
- `n_bins` still sizes every histogram. Width is a *storage* property.

Any scheme that remapped ids to shrink a width would silently invalidate all
four. There is no such scheme in the source and no parameter that enables
one.

## Which bin count a width is derived from

Two different counts exist and they are not interchangeable:

- `declared_bins_from_mapper(mapper)` — capacity, from the fitted
  `BinMapper`. A numerical feature with `k` edges can return `0..k`, plus a
  missing bin at `k+1`; a categorical one can return `0..n_categories`.
  **Safe for every matrix the mapper produces**, including prediction
  batches binned later.
- `observed_bins_from_matrix(data)` — one past the largest id present in one
  matrix. Narrower, and valid **for that matrix only**. A feature whose
  rarest bin is absent from the training rows gets a width that a validation
  set can overflow.

Prefer the mapper form. `check_plan_covers_matrix` is the guard when the
observed form is used anyway, and `pack_binned_matrix` range-checks every
value regardless, so the failure is a raised error and never a truncated id.

## The cost model, and what it refuses to do

`layout_node_cost` returns counts, not a time:

| unit | meaning |
| --- | --- |
| `bin_sectors` | memory sectors the bin reads touch |
| `row_sectors` | sectors the row index, gradient, and hessian touch, once per *block* |
| `decode_ops` | bit-extraction ALU ops, zero at width 8 |
| `launches` | one per touched block |
| `shared_bytes` | widest touched block's threadgroup allocation (an occupancy input, never folded into a time) |

The sector model is two regimes and nothing between:

```
touched = min(count * ceil(bytes_per_access / S), ceil(span_bytes / S))
```

Node rows stay ascending, because `gpu_active_rows.mojo` partitions stably,
so a node sits somewhere on this curve rather than randomly on it. The
crossover is at `count = span_bytes / S`. At 1e6 rows, width 8, and a
32-byte sector that is **about depth 5**: above it the streaming term binds
and compression pays in full; below it every read is its own sector whatever
the width and compression buys nothing on this axis. A benchmark that
measures only root nodes concludes compression works; one that measures only
leaves concludes it does nothing. Both would be wrong, which is why
`node_depths` is swept in the plan.

`decide_layout` combines these into nanoseconds **only** when the caller
supplies `MeasuredLayoutCosts` with every field positive, including
`sector_bytes`. Otherwise it returns `LAYOUT_UNDECIDED`. There is no default
threshold, no width heuristic that fires on its own, and no automatic
fallback. `SECTOR_BYTES_UNKNOWN` propagates: an unmeasured sector makes
`sectors_touched` return 0 and `is_measured()` false, so it cannot silently
become a constant.

Deliberately not modelled, and each is a gap a policy must not paper over:

- **L2 residency across blocks.** The row side is charged once per touched
  block, exact when `rows`/`grad`/`hess` exceed L2 and an overestimate when
  they do not. This is the largest unknown in the model and Q1 in the plan.
- **The partition kernel.** `_flag_scan_kernel` and `_scatter_kernel` read
  one feature's bins per split. Feature-major gives them a contiguous
  column; a blocked layout gives them a stride-`G*w/8` gather. Blocking
  could win the histogram and lose the partition. Not priced at all (Q7).
- **Sibling subtraction.** Only one child is built per split, which changes
  the depth mix and therefore which regime dominates a tree.
- **The prediction path.** `gpu_predict` walks trees per row and prefers the
  opposite layout. A training-side verdict does not cover it (Q8).

## Integration seam

Nothing below has been done; this is the shape the integration takes.

**Where the layout attaches.** `GpuHistogramBuilder.__init__` currently does:

```
self.bins_dev = ctx.enqueue_create_buffer[DType.uint8](n_rows * n_features)
ctx.enqueue_copy(dst_buf=self.bins_dev, src_ptr=data.bins.unsafe_ptr())
```

The packed form is the same two lines against a plan:

```
var plan = <chosen BinLayoutPlan>
if plan.is_passthrough():
    # unchanged: upload data.bins directly, no packing pass
else:
    var packed = pack_binned_matrix(data, plan)
    self.bins_dev = ctx.enqueue_create_buffer[DType.uint8](plan.bytes())
    ctx.enqueue_copy(dst_buf=self.bins_dev, src_ptr=packed.bytes.unsafe_ptr())
```

plus the plan's block tables as small Int32 device buffers
(`block_of`, `lane_of`, `block_offset`, `block_width`, `block_size`), or as
kernel parameters when the kernel is specialized on one block.

**Where the kernels change.** Exactly one expression, in three places:

```
old:  var bin = Int(bins[unsafe_offset = f * nr + r])
new:  var bin = decode(bins, base, r * G + lane, w)
```

- `gpu_active_rows._range_hist_atomic_kernel` (line ~455)
- `gpu_active_rows._range_hist_partial_kernel` (line ~537)
- `gpu_active_rows._scatter_kernel` / `_flag_scan_kernel` (the partition)

and, if the prediction path adopts a layout, `gpu_predict._predict_kernel`.
Everything downstream is untouched: the fixed-point Int32 accumulation, the
shared-memory partials, the tiled reduction, the stable partition, and the
range bookkeeping all operate on bin *ids*, which are preserved.

**The kernel should be specialized on the width, not branch on it.** A block
is width-homogeneous by construction, so the width is uniform across a
threadgroup and a `comptime` specialization per width compiles the decode
away entirely at width 8 and to a fixed sequence below it. A runtime branch
would be uniform too, but the specialization is what makes the fallback
provably free.

**Staging order.** Each step is independently testable and independently
revertible:

1. Land the modules unused (this lane). Nothing changes.
2. Add a packed *read* path to the histogram kernel, selectable by an env
   var, defaulting to the passthrough plan. Assert bit-identical histograms
   against the baseline (Q6). No policy, no threshold.
3. Measure `bench/apple/bin_layout_plan.json`'s units and answer Q1–Q3.
4. Only then add a multi-feature (blocked) histogram kernel, which is a
   genuinely new kernel and the largest piece of work here.
5. Only then, and only if the measurements support it, let a policy select
   anything other than passthrough.

Steps 1 and 2 are safe under any measurement outcome. Step 4 is where the
predicted win lives and also where the effort is.

## Portability risks

| risk | why it is a risk | how it is handled here |
| --- | --- | --- |
| Unaligned multi-byte loads | Metal cannot portably load a `UInt32` at an arbitrary byte offset | every access is a byte load; the window is composed arithmetically |
| Endianness | a word bit stream encodes host byte order into a device buffer | the byte-pair window is byte-order independent by construction |
| Shift by full width | `x << 8` on a `UInt8` is undefined | all arithmetic is done at 16-bit-or-wider (`Int` on the host, `UInt32` in a kernel) and the shift is at most 7 |
| Shared memory ceiling | Metal threadgroups are far smaller than CUDA's | `max_block_for_shared` derives the block cap from `MAX_SHARED_MEMORY_PER_BLOCK`, which `gpu_tiling.query_device_caps` already queries with a conservative fallback |
| Int32 byte offsets | a packed buffer is indexed by Int32 in kernels | `LAYOUT_BYTES_OVERFLOW` checked at plan construction, against the same `Int32.MAX` bound `histogram_gpu.MAX_ROWS` uses |
| Warp divergence on width | a per-feature width would diverge inside a threadgroup | blocks are width-homogeneous, so the width is threadgroup-uniform |
| Sector size assumed | a wrong `S` silently mis-prices everything | `SECTOR_BYTES_UNKNOWN` is the default and forces `LAYOUT_UNDECIDED` |
| Plan derived from the wrong matrix | an observed width can be too narrow for a later batch | `declared_bins_from_mapper` is the safe source; `check_plan_covers_matrix` and a range check in `pack_binned_matrix` catch the unsafe one |

## Future profiler questions

Full text with the reasoning is in `bench/apple/bin_layout_plan.json`. In
short:

- **Q1** Does the row side stay L2-resident across a node's features? This
  decides whether blocking's 4x is real or illusory, and it is the single
  question that most changes what should be built.
- **Q2** What is measured bin-read efficiency versus node depth, per layout?
  This tests the two-regime shape of `sectors_touched` itself, not its
  coefficients.
- **Q3** Is the 4-bit decode really 4 ALU ops in the emitted kernel, or does
  the compiler fuse it?
- **Q4** At what block width does occupancy actually fall off? The
  arithmetic shared-memory bound is an upper bound on the useful one.
- **Q5** On Apple unified memory, is per-byte upload cost distinguishable
  from zero? If it is zero, compression's transfer argument disappears on
  the target device and only the shared-memory argument survives.
- **Q6** Are packed histograms bit-identical to the baseline's? They must
  be; any difference is an addressing bug, not precision.
- **Q7** What does the partition kernel cost under a blocked layout?
- **Q8** Is a second, prediction-shaped resident copy of the matrix worth
  its memory?

## Known risks in this lane's own work

- The modules are unbuilt and untested. No Mojo compiler ran. Syntax and
  type errors are possible and a build is the first thing the next lane
  should do.
- `check_blocks_disjoint` is quadratic in the block count. Bounded by
  `n_features`, run once, and meant for verification, but it is not free on
  a very wide matrix.
- `observed_bins_from_matrix` is a full host pass over the matrix. It is the
  fallback path, not the recommended one, and it is why the mapper form
  exists.
- `pack_binned_matrix` is serial. Its inner loop is per-block and the blocks
  are provably disjoint, so it parallelizes across blocks with
  `dispatch_features`-style scheduling, but this lane did not add that:
  packing cost is a measured quantity (`ns_per_pack_write`) and optimizing
  it before measuring would be the same mistake the module is written to
  avoid.
- The blocked histogram kernel does not exist. Everything about blocking in
  this document is a design and a cost model, not a running path.
