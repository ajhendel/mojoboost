"""Boundary helpers shared by the mojoboost extension modules.

`bindings/_mojoboost.mojo` is the CPython entry point; the modules beside
it (`objective_bindings`, `dataset_bindings`, `inspection_bindings`,
`distributed_bindings`, `basic_bindings`) hold one capability each and are
registered from there. This module holds what all of them need to cross
the boundary and nothing else: no policy, no algorithm, no table.

Conventions, all of them the ones `_mojoboost.mojo` already established:

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

from std.python import Python, PythonObject


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
    """Copy a float64 buffer into a Mojo list.

    The same read `_f64_list` in `_mojoboost.mojo` does, and the intended
    single home for it: see `handoffs/connect_14_bindings.md` for the patch
    that retires that copy.
    """
    if addr == 0:
        raise Error("null buffer address")
    if n < 0:
        raise Error("buffer length must not be negative, got ", n)
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


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
