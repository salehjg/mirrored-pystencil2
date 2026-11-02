"""Execution tests for the RVV platform that the generic test suite does not cover.

Most RVV numerics are exercised by the target-parametrized suites (test_vectorization,
test_type_cast, test_reduction, test_functions, ...), which pick up `Target.RISCV_RVV`
automatically on RVV hardware. This file adds:

- Philox RNG parity against the scalar `GenericCPU` kernel. (`test_rng.py` compares
  against `randomgen`, which is usually not installed on RISC-V boards, and skips.)
- The float -> int truncation corner case, which only `_rtz_` conversions get right.

Skipped unless the host exposes `Target.RISCV_RVV`.
"""

import numpy as np
import pytest

import pystencils as ps
from pystencils import Target, Assignment, TypedSymbol, DynamicType, fields
from pystencils.sympyextensions.typed_sympy import tcast

pytestmark = pytest.mark.skipif(
    Target.RISCV_RVV not in Target.available_vector_cpu_targets(),
    reason="requires RVV hardware",
)


def make_config(target: Target, dtype: str, index_dtype: str = "int64"):
    cfg = ps.CreateKernelConfig(target=target, default_dtype=dtype, index_dtype=index_dtype)
    if target.is_vector_cpu():
        cfg.cpu.vectorize.enable = True
        cfg.cpu.vectorize.assume_inner_stride_one = True
    return cfg


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("offsets", [(0, 0), (12, 41)])
def test_philox_matches_generic_cpu(dtype, offsets):
    rng = ps.random.Philox("philox", dtype, seed=0x7EADBEEF, offsets=offsets)
    q = rng.vector_size
    f = fields(f"f({q}): {dtype}[2D]", layout="fzyx")
    c = TypedSymbol("c", DynamicType.INDEX_TYPE)
    rx, rasm = rng.get_random_vector(c)
    asms = [rasm] + [Assignment(f(i), rx[i]) for i in range(q)]

    #   Counters have the width of the results, as in tests/kernelcreation/test_rng.py
    index_dtype = f"int{np.dtype(dtype).itemsize * 8}"

    #   Odd extent: exercises the vectorized block loop *and* the scalar remainder loop
    shape = (13, 21, q)
    layout = ps.field.layout_string_to_tuple("fzyx", 3)
    results = {}
    for target in (Target.GenericCPU, Target.RISCV_RVV):
        kernel = ps.create_kernel(asms, make_config(target, dtype, index_dtype)).compile()
        arr = ps.field.create_numpy_array_with_layout(shape, layout, dtype=dtype)
        kernel(f=arr, c=7)
        results[target] = arr

    np.testing.assert_array_equal(results[Target.RISCV_RVV], results[Target.GenericCPU])


@pytest.mark.parametrize("src, dst", [("float64", "int64"), ("float64", "int32"), ("float32", "int32")])
def test_float_to_int_truncates(src, dst):
    #   C casts truncate: (int)(-1.5) == -1. A plain `vfcvt_x_f` would round to -2.
    inp = fields(f"inp: {src}[1D]")
    outp = fields(f"outp: {dst}[1D]")
    max_width = max(np.dtype(src).itemsize, np.dtype(dst).itemsize) * 8
    kernel = ps.create_kernel(
        [Assignment(outp.center(), tcast(inp.center(), dst))],
        make_config(Target.RISCV_RVV, f"float{max_width}"),
    ).compile()

    values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -7.9, 7.9, -1.0, 3.0, 1e6], dtype=src)
    out = np.zeros_like(values, dtype=dst)
    kernel(inp=values, outp=out)
    np.testing.assert_array_equal(out, values.astype(dst))
