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
import sympy as sp
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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vf_forms_and_fma_match_numpy(dtype):
    """Scalar (.vf) operands, reversed forms and fused multiply-adds compute what they should.

    One output per form: sympy would merge the terms of a combined expression."""
    a, b, c = fields(f"a, b, c: {dtype}[1D]")
    x, y, z = sp.symbols("x y z")
    forms = {
        "vfrsub_vf": (3 - x, lambda x, y, z: 3 - x),
        "vfmsac_vv": (x * y - z, lambda x, y, z: x * y - z),
        "vfnmsac_vv": (z - x * y, lambda x, y, z: z - x * y),
        "vfmacc_vv": (x * y + z, lambda x, y, z: x * y + z),
        "vfmacc_vf": (2 * x + z, lambda x, y, z: 2 * x + z),
        "vfmsac_vf": (2 * x - z, lambda x, y, z: 2 * x - z),
        "vfadd_vf": (x * sp.Rational(1, 7) + 3, lambda x, y, z: x / 7 + 3),
    }
    outs = fields(", ".join(f"d{i}" for i in range(len(forms))) + f": {dtype}[1D]")
    asms = [Assignment(x, a.center()), Assignment(y, b.center()), Assignment(z, c.center())]
    asms += [Assignment(d.center(), expr) for d, (expr, _) in zip(outs, forms.values())]
    kernel = ps.create_kernel(asms, make_config(Target.RISCV_RVV, dtype))
    code = kernel.get_c_code()
    for op in forms:
        assert f"__riscv_{op}_" in code

    rng = np.random.default_rng(5)
    av, bv, cv = (rng.random(203).astype(dtype) + 0.5 for _ in range(3))
    results = {f"d{i}": np.zeros_like(av) for i in range(len(forms))}
    kernel.compile()(a=av, b=bv, c=cv, **results)
    #   FMAs round once, numpy twice: allow a few ulps of the operands (x y - z may nearly cancel)
    tol = 16 * np.finfo(dtype).eps
    for i, (_, ref) in enumerate(forms.values()):
        np.testing.assert_allclose(results[f"d{i}"], ref(av, bv, cv), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prefetch_does_not_change_results(dtype):
    """Software prefetches are hints: the kernel output must be bit-identical."""
    f, g = fields(f"f, g: {dtype}[2D]")
    asms = [Assignment(g.center(), f[1, 0] + 2 * f[-1, 0] - f[0, 1] * f[0, -1])]
    rng = np.random.default_rng(3)
    src = rng.random((37, 29)).astype(dtype)
    results = []
    for distance in (None, 128):
        cfg = make_config(Target.RISCV_RVV, dtype)
        cfg.cpu.vectorize.prefetch_distance = distance
        kernel = ps.create_kernel(asms, cfg)
        assert ("prefetch_read" in kernel.get_c_code()) == (distance is not None)
        dst = np.zeros_like(src)
        kernel.compile()(f=src, g=dst)
        results.append(dst)
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_stencil_row_pointers_match_generic_cpu(dtype):
    #   Every access of the vector loop goes through a row base pointer (HoistRowPointers):
    #   neighbours at +-1 along all axes, the innermost one included, with an extent that also
    #   runs the scalar remainder loop. The scalar GenericCPU kernel is the reference.
    src, dst = fields(f"src, dst: {dtype}[3D]")
    offsets = [(0, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 1), (-1, 1, -1)]
    rhs = sum((k + 1) * src[off] for k, off in enumerate(offsets))
    asms = [Assignment(dst.center(), rhs)]
    rng = np.random.default_rng(42)
    src_arr = rng.random((7, 9, 23)).astype(dtype)
    results = {}
    for target in (Target.GenericCPU, Target.RISCV_RVV):
        kernel = ps.create_kernel(asms, make_config(target, dtype)).compile()
        dst_arr = np.zeros_like(src_arr)
        kernel(src=src_arr, dst=dst_arr)
        results[target] = dst_arr
    rtol = 1e-6 if dtype == "float32" else 1e-14
    np.testing.assert_allclose(results[Target.RISCV_RVV], results[Target.GenericCPU], rtol=rtol)
    assert np.count_nonzero(results[Target.RISCV_RVV]) == 5 * 7 * 21
