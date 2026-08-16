"""Boundary helpers shared by the mojotrees extension modules.

`bindings/_mojotrees.mojo` is the CPython entry point; the modules beside
it (`objective_bindings`, `dataset_bindings`, `inspection_bindings`,
`distributed_bindings`, `basic_bindings`) hold one capability each and are
registered from there. This module holds what all of them need to cross
the boundary and nothing else: no policy, no algorithm, no table.

Conventions, all of them the ones `_mojotrees.mojo` already established:

- Bulk numeric data crosses as a raw buffer address (an integer) plus its
  length. The Python caller keeps the buffer alive for the call; nothing
  here retains it past the copy.
- Flags cross as 0/1 ints, so the boundary converts no Python bool.
- Sequences of strings cross as the sequence plus its length, because a
  `PythonObject` length is one more call that can fail.
- Every read validates before it dereferences: a null address, a negative
  length, an out-of-range index, and a flag that is neither 0 nor 1 are
  all refused here rather than in the implementation, where the caller's
  argument is no longer in hand.

Nothing in this module hands a raw pointer or a device buffer to Python.
The only direction an address travels is in.
"""

from std.memory import unsafe_memcpy
from std.python import Python, PythonObject

from mojotrees.sparse import CscMatrix, CsrMatrix


# -- building Python values ----------------------------------------------


def py_dict() raises -> PythonObject:
    """An empty Python dict.

    Built through `builtins` rather than a constructor on `Python`: this
    module is compiled against whatever Mojo the repository pins, and
    `builtins.dict` exists in every CPython it can be built against.
    """
    return Python.import_module("builtins").dict()


def py_int_list(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(values[i]))
    return out^


def py_f64_list(values: List[Float64]) raises -> PythonObject:
    """A float64 list. The values cross as C doubles, so the payload holds
    the model's own bits and not a decimal rendering of them."""
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(values[i]))
    return out^


def py_str_list(values: List[String]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(values[i]))
    return out^


def py_pair(first: PythonObject, second: PythonObject) raises -> PythonObject:
    """A two-element list. Records cross as lists rather than tuples
    because a list is what every other sequence at this boundary already
    is; a Python consumer that wants tuples builds them once on arrival."""
    var out = Python.list()
    out.append(first)
    out.append(second)
    return out^


# -- reading Python values -----------------------------------------------


def f64_buffer(addr: Int, n: Int) raises -> List[Float64]:
    """Copy a float64 buffer (NumPy's X, y, weights) into a Mojo list.

    One bulk copy, not an element-by-element append. The caller's buffer and
    the list hold the same bytes in the same order, so there is nothing per
    element to decide: `unsafe_uninit_length` skips the zero fill that
    `resize` would do, and the copy that follows writes every one of those
    bytes.

    The copy is a small share of an ingest. For a C-ordered array the
    NumPy-side `asfortranarray` transpose costs far more, and it belongs on
    the NumPy side, which does it blocked; handing the trainer a row-major
    buffer instead would only move the same work into the binner's
    per-column gather, strided and cold. Where the copy is worth avoiding
    is space, not time; `_f64_view` in `_mojotrees.mojo` borrows the
    feature matrix for that reason.
    """
    if addr == 0:
        raise Error("null buffer address")
    if n < 0:
        raise Error("buffer length must not be negative, got ", n)
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    if n == 0:
        return List[Float64]()
    var out = List[Float64](unsafe_uninit_length=n)
    unsafe_memcpy(dest=out.unsafe_ptr(), src=p, count=n)
    return out^


def f64_view(addr: Int, n: Int) raises -> Span[Float64, ImmUntrackedOrigin]:
    """Borrow a float64 buffer (NumPy's X) instead of copying it.

    The feature matrix is the one input where the copy is worth avoiding,
    and not for the reason it looks like: `f64_buffer` moves it at memory
    speed, a low single-digit percentage of an ingest. What the copy costs
    is *space*. It doubles the resident footprint of the matrix for as long
    as binning runs, so a 5,000,000 x 100 fit holds 4 GB of NumPy plus 4 GB
    of Mojo, and on a machine that can afford one of those but not both the
    difference is not a percentage.

    Borrowing is sound where the matrix is read, never written, and is dead
    early: `fit_bins` and `BinMapper.transform` are the only things that look
    at it, and after transform the trainer works on the binned `UInt8`
    matrix. It stays alive throughout because the Python wrapper holds the
    array it took the address of (see `_arrays.column_major`, whose contract
    is exactly that) for the whole call.

    The origin is untracked because the owner is on the other side of the
    boundary and Mojo cannot see it. That is the same contract `f64_buffer`
    already relies on for its source pointer; the difference is only how
    long it has to hold, which is the length of one call either way.

    Not every input can do this. A buffer that outlives the call must be
    copied, so the validation sets, which `RawValidSet` owns, still take
    `f64_buffer`. `dataset_create` and `dataset_create_reference` used to be
    on that list and no longer are: `Dataset` bins the matrix where it lies
    and retains it only under `keep_raw`, which it copies for itself.
    """
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=addr)
    return Span[Float64, ImmUntrackedOrigin](unsafe_ptr=p, length=n)


def f64_view_mut(
    addr: Int, n: Int
) raises -> Span[Float64, MutUntrackedOrigin]:
    """`f64_view` for a buffer the caller wants written.

    The one thing this library writes into a caller's memory that is not an
    output vector: the column-major feature buffer `_arrays.column_major`
    allocates and then hands to the binner. The wrapper allocates it because
    the wrapper has to return it and keep it alive; the transpose that fills
    it belongs on this side because it is `n_rows * n_features` doubles of
    work and NumPy would do it on one thread.
    """
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    return Span[Float64, MutUntrackedOrigin](unsafe_ptr=p, length=n)


def int_buffer_from_f64(addr: Int, n: Int) raises -> List[Int]:
    """Copy a float64 buffer holding whole numbers into an int list. The
    Python layer normalizes integer columns to float64 before they cross,
    which is why an int-valued column arrives this way."""
    if addr == 0:
        raise Error("null buffer address")
    if n < 0:
        raise Error("buffer length must not be negative, got ", n)
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(p.unsafe_load(i)))
    return out^


def write_f64_buffer(
    values: List[Float64], out_addr: Int, capacity: Int
) raises:
    """Write a float64 list into a buffer the caller preallocated.

    `capacity` is what the caller says the buffer holds, and a list longer
    than that is refused rather than truncated: a short write here is a
    memory error in the caller's process, so the length disagreement has to
    be fatal at the boundary.
    """
    if out_addr == 0:
        raise Error("null output buffer address")
    if capacity < len(values):
        raise Error(
            "output buffer holds ",
            capacity,
            " values but ",
            len(values),
            " were produced",
        )
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=out_addr
    )
    for i in range(len(values)):
        out.unsafe_store(i, values[i])


def int_buffer(addr: Int, n: Int) raises -> List[Int]:
    """Copy an int64 buffer (SciPy's `indices` and `indptr`) into a Mojo
    list. Same bulk copy as `f64_buffer`, for the same reason: int64 in, Int
    out, identical bytes."""
    if addr == 0:
        raise Error("null buffer address")
    if n < 0:
        raise Error("buffer length must not be negative, got ", n)
    var p = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=addr)
    if n == 0:
        return List[Int]()
    var out = List[Int](unsafe_uninit_length=n)
    unsafe_memcpy(dest=out.unsafe_ptr(), src=p, count=n)
    return out^


def csc_from_params(params: PythonObject) raises -> CscMatrix:
    """Rebuild a CSC matrix from the buffer addresses in a params mapping.

    The six keys `_arrays.SparseBuffers.params()` emits:
    `sparse_data_addr`, `sparse_indices_addr`, `sparse_indptr_addr`,
    `sparse_nnz`, `n_rows`, and `n_features`. SciPy's arrays are
    normalized to float64 data and int64 indices on the Python side; the
    matrix itself is validated by the sparse binner, which stays the only
    place that knows what a well-formed one is.
    """
    var n_rows = Int(py=params["n_rows"])
    var n_features = Int(py=params["n_features"])
    var nnz = nonnegative(params["sparse_nnz"], "sparse_nnz")
    return CscMatrix(
        int_buffer(Int(py=params["sparse_indices_addr"]), nnz),
        f64_buffer(Int(py=params["sparse_data_addr"]), nnz),
        int_buffer(Int(py=params["sparse_indptr_addr"]), n_features + 1),
        n_rows,
        n_features,
    )


def csr_from_params(params: PythonObject) raises -> CsrMatrix:
    """Rebuild a CSR matrix from the same six keys, whose `indptr` has one
    entry per row rather than per feature."""
    var n_rows = Int(py=params["n_rows"])
    var n_features = Int(py=params["n_features"])
    var nnz = nonnegative(params["sparse_nnz"], "sparse_nnz")
    return CsrMatrix(
        int_buffer(Int(py=params["sparse_indices_addr"]), nnz),
        f64_buffer(Int(py=params["sparse_data_addr"]), nnz),
        int_buffer(Int(py=params["sparse_indptr_addr"]), n_rows + 1),
        n_rows,
        n_features,
    )


def str_sequence(seq: PythonObject, n: Int) raises -> List[String]:
    """A Python sequence of `n` strings as a Mojo list. `n` travels
    alongside the sequence, as every other sequence at this boundary
    does."""
    if n < 0:
        raise Error("sequence length must not be negative, got ", n)
    var out = List[String](capacity=n)
    for i in range(n):
        out.append(String(py=seq[i]))
    return out^


def flag(value: PythonObject, name: String) raises -> Bool:
    """A 0/1 int as a Bool.

    Flags cross as ints so the boundary converts no Python bool, and a
    value that is neither is refused rather than read as truthiness: a
    caller that passed 2 meant something this side cannot guess.
    """
    var raw = Int(py=value)
    if raw == 0:
        return False
    if raw == 1:
        return True
    raise Error("'", name, "' must be 0 or 1, got ", raw)


def nonnegative(value: PythonObject, name: String) raises -> Int:
    var raw = Int(py=value)
    if raw < 0:
        raise Error("'", name, "' must not be negative, got ", raw)
    return raw


def index_within(value: PythonObject, count: Int, name: String) raises -> Int:
    """An index checked against the number of things it indexes."""
    var raw = Int(py=value)
    if raw < 0 or raw >= count:
        raise Error(
            "'",
            name,
            "' must be in [0, ",
            count,
            "), got ",
            raw,
        )
    return raw
