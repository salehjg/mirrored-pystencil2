"""Code generation tests for the RVV platform (VLS, VLEN = 256).

These only generate code and inspect it, so they run on any host. The last test
additionally compiles the generated code with a RISC-V compiler, if one is available
(the native `g++` on a riscv64 host, or the `riscv64-linux-gnu-g++` cross compiler),
which catches misspelled intrinsics without having to run anything.
"""

import platform
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

import pystencils as ps
from pystencils import Target, Assignment, TypedSymbol, DynamicType, fields
from pystencils.backend.exceptions import MaterializationError
from pystencils.backend.platforms.rvv import _rvv_type
from pystencils.sympyextensions import ReductionOp, reduction_assignment
from pystencils.sympyextensions.typed_sympy import tcast
from pystencils.types import PsVectorType
from pystencils.types.quick import Fp, SInt, UInt


def rvv_config(default_dtype="float64", **kwargs) -> ps.CreateKernelConfig:
    """Vectorized RVV config; lanes default to one full 256-bit register (like AVX)."""
    cfg = ps.CreateKernelConfig(target=Target.RISCV_RVV, default_dtype=default_dtype, **kwargs)
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.assume_inner_stride_one = True
    return cfg


def rvv_code(asms, cfg=None) -> str:
    return ps.create_kernel(asms, cfg or rvv_config()).get_c_code()


#   -------------------------------------------------------------------------------------
#   Type mapping: the analogue of `__m256d` / `__m128` on AVX
#   -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stype, lanes, expected",
    [
        (Fp(64), 4, "vfloat64m1_t"),  # 256 bit -> one register
        (Fp(32), 8, "vfloat32m1_t"),
        (Fp(32), 4, "vfloat32mf2_t"),  # 128 bit -> half a register
        (SInt(64), 4, "vint64m1_t"),
        (SInt(32), 8, "vint32m1_t"),
        (SInt(32), 4, "vint32mf2_t"),
        (SInt(16), 16, "vint16m1_t"),
    ],
)
def test_vector_types(stype, lanes, expected):
    assert _rvv_type(PsVectorType(stype, lanes)).c_string() == expected


@pytest.mark.parametrize(
    "stype, lanes",
    [
        (Fp(64), 2),  # 128 bit of f64 would be `vfloat64mf2_t`, which does not exist
        (Fp(64), 8),  # 512 bit: wider than a register (no LMUL > 1, as on AVX)
        (Fp(32), 16),
        (UInt(32), 8),  # unsigned: not supported (as on x86)
        (Fp(16), 16),
    ],
)
def test_unsupported_vector_types(stype, lanes):
    with pytest.raises(MaterializationError):
        _rvv_type(PsVectorType(stype, lanes))


def test_default_lanes_and_header():
    f, g = fields("f, g: float64[2D]")
    ker = ps.create_kernel([Assignment(g.center(), f.center())], rvv_config())
    assert '"pystencils_runtime/rvv.hpp"' in ker.required_headers
    code = ker.get_c_code()
    #   4 x f64 per iteration; every intrinsic gets the literal vector length
    assert "__riscv_vle64_v_f64m1(" in code and "4uLL)" in code
    assert "__riscv_vse64_v_f64m1(" in code
    assert "+= 4LL" in code


#   -------------------------------------------------------------------------------------
#   Arithmetic, constants, math functions
#   -------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, sfx", [("float64", "f64m1"), ("float32", "f32m1")])
def test_float_arithmetic(dtype, sfx):
    a, b, c = fields(f"a, b, c: {dtype}[2D]")
    code = rvv_code(
        [Assignment(c.center(), (a.center() + 2 * b.center()) / (a.center() - b[1, 0]))],
        rvv_config(dtype),
    )
    for op in ("vfadd_vv", "vfmul_vv", "vfdiv_vv", "vfmv_v_f"):
        assert f"__riscv_{op}_{sfx}(" in code


@pytest.mark.parametrize("dtype, sfx", [("int64", "i64m1"), ("int32", "i32m1")])
def test_int_arithmetic(dtype, sfx):
    a, b, c = fields(f"a, b, c: {dtype}[2D]")
    code = rvv_code(
        [Assignment(c.center(), a.center() * b.center() + 3)],
        rvv_config(dtype, index_dtype=dtype),
    )
    assert f"__riscv_vmul_vv_{sfx}(" in code
    assert f"__riscv_vadd_vv_{sfx}(" in code
    assert f"__riscv_vmv_v_x_{sfx}(" in code


def test_counter_offsets_use_set_helper():
    #   The vectorized inner loop counter is `ctr + [0, 1, 2, 3]`; that non-uniform
    #   constant goes through the `set_*` helper (the AVX analogue is `_mm256_set_*`).
    f = fields("f: float64[1D]")
    x = ps.DEFAULTS.spatial_counters[0]
    code = rvv_code([Assignment(f.center(), tcast(x, "float64"))])
    assert "pystencils::runtime::rvv::set_i64m1(0LL, 1LL, 2LL, 3LL)" in code
    assert "__riscv_vfcvt_f_x_v_f64m1(" in code


@pytest.mark.parametrize("func, intrin", [(sp.sqrt, "vfsqrt_v"), (sp.Abs, "vfabs_v")])
def test_unary_float_math(func, intrin):
    f, g = fields("f, g: float64[1D]")
    code = rvv_code([Assignment(g.center(), func(f.center()))])
    assert f"__riscv_{intrin}_f64m1(" in code


@pytest.mark.parametrize("func, name", [(sp.Min, "min"), (sp.Max, "max")])
@pytest.mark.parametrize("dtype, sfx, prefix", [("float64", "f64m1", "vf"), ("int32", "i32m1", "v")])
def test_min_max(func, name, dtype, sfx, prefix):
    a, b, c = fields(f"a, b, c: {dtype}[1D]")
    code = rvv_code(
        [Assignment(c.center(), func(a.center(), b.center()))],
        rvv_config(dtype, index_dtype="int32" if dtype == "int32" else "int64"),
    )
    assert f"__riscv_{prefix}{name}_vv_{sfx}(" in code


def test_integer_abs():
    #   No integer abs instruction on RVV: |x| = max(x, -x)
    f, g = fields("f, g: int64[1D]")
    code = rvv_code([Assignment(g.center(), sp.Abs(f.center()))], rvv_config("int64"))
    assert "__riscv_vneg_v_i64m1(" in code
    assert "__riscv_vmax_vv_i64m1(" in code


@pytest.mark.parametrize("func", [sp.floor, sp.exp])
def test_unsupported_math(func):
    f, g = fields("f, g: float64[1D]")
    with pytest.raises(MaterializationError):
        rvv_code([Assignment(g.center(), func(f.center()))])


#   -------------------------------------------------------------------------------------
#   Casts: at most a factor of two in element width, one intrinsic each
#   -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, dst, intrin",
    [
        ("float32", "float64", "vfwcvt_f_f_v_f64m1"),
        ("float64", "float32", "vfncvt_f_f_w_f32mf2"),
        ("int32", "float64", "vfwcvt_f_x_v_f64m1"),
        ("float64", "int32", "vfncvt_rtz_x_f_w_i32mf2"),
        ("int64", "float64", "vfcvt_f_x_v_f64m1"),
        ("float64", "int64", "vfcvt_rtz_x_f_v_i64m1"),
        ("int32", "int64", "vsext_vf2_i64m1"),
        ("int64", "int32", "vncvt_x_x_w_i32mf2"),
        ("int32", "float32", "vfcvt_f_x_v_f32m1"),
        ("float32", "int32", "vfcvt_rtz_x_f_v_i32m1"),
        ("int64", "float32", "vfncvt_f_x_w_f32mf2"),
        ("float32", "int64", "vfwcvt_rtz_x_f_v_i64m1"),
    ],
)
def test_casts(src, dst, intrin):
    inp = fields(f"inp: {src}[1D]")
    outp = fields(f"outp: {dst}[1D]")
    #   Same setup as tests/kernelcreation/test_type_cast.py: the wider type sets the lanes
    max_width = max(np.dtype(src).itemsize, np.dtype(dst).itemsize) * 8
    code = rvv_code(
        [Assignment(outp.center(), tcast(inp.center(), dst))],
        rvv_config(f"float{max_width}"),
    )
    assert f"__riscv_{intrin}(" in code


#   -------------------------------------------------------------------------------------
#   Memory access
#   -------------------------------------------------------------------------------------


def test_strided_load_store():
    f = fields("f: float64[2D]", layout="c")
    cfg = rvv_config()
    cfg.cpu.vectorize.assume_inner_stride_one = False
    code = rvv_code([Assignment(f.center(), 2 * f.center())], cfg)
    #   Strides are passed in bytes
    assert "__riscv_vlse64_v_f64m1(" in code
    assert "__riscv_vsse64_v_f64m1(" in code
    assert "8LL * _stride_f_1" in code


#   -------------------------------------------------------------------------------------
#   Reductions
#   -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, helper",
    [
        (ReductionOp.Add, "add"),
        (ReductionOp.Sub, "add"),  # accumulates the negated terms, like x86
        (ReductionOp.Mul, "mul"),
        (ReductionOp.Min, "min"),
        (ReductionOp.Max, "max"),
    ],
)
@pytest.mark.parametrize("dtype, sfx", [("float64", "f64m1"), ("float32", "f32m1")])
def test_reductions(op, helper, dtype, sfx):
    w = TypedSymbol("w", dtype)
    x = fields(f"x: {dtype}[1D]")
    code = rvv_code([reduction_assignment(w, op, x.center())], rvv_config(dtype))
    assert f"pystencils::runtime::rvv::horizontal_{helper}_{sfx}(" in code


#   -------------------------------------------------------------------------------------
#   Random number generation
#   -------------------------------------------------------------------------------------


def philox_asms(dtype: str):
    rng = ps.random.Philox("philox", dtype, seed=0x7EADBEEF, offsets=(0, 0))
    q = rng.vector_size
    f = fields(f"f({q}): {dtype}[2D]", layout="fzyx")
    c = TypedSymbol("c", DynamicType.INDEX_TYPE)
    rx, rasm = rng.get_random_vector(c)
    return [rasm] + [Assignment(f(i), rx[i]) for i in range(q)]


@pytest.mark.parametrize(
    "dtype, func, tuple_type, vget",
    [
        ("float32", "philox_float32", "vfloat32m1x4_t", "__riscv_vget_v_f32m1x4_f32m1"),
        ("float64", "philox_float64", "vfloat64m1x2_t", "__riscv_vget_v_f64m1x2_f64m1"),
    ],
)
def test_philox(dtype, func, tuple_type, vget):
    width = np.dtype(dtype).itemsize * 8
    #   Counters must have the width of the results (as in tests/kernelcreation/test_rng.py)
    code = rvv_code(philox_asms(dtype), rvv_config(dtype, index_dtype=f"int{width}"))
    assert f"pystencils::runtime::rvv::random::{func}(" in code
    assert tuple_type in code
    assert f"{vget}(" in code


#   -------------------------------------------------------------------------------------
#   Compile the generated code with a RISC-V compiler (if available)
#   -------------------------------------------------------------------------------------


def _riscv_cxx() -> str | None:
    if platform.machine() == "riscv64":
        return shutil.which("g++")
    return shutil.which("riscv64-linux-gnu-g++")


def _compile_kernels():
    """A handful of kernels covering every code path of the RVV selector."""
    a, b, c = fields("a, b, c: float64[2D]")
    s32 = fields("s32: float32[1D]")
    i32 = fields("i32: int32[1D]")
    d64 = fields("d64: float64[1D]")
    w = TypedSymbol("w", "float64")
    strided = fields("strided: float64[2D]", layout="c")
    strided_cfg = rvv_config()
    strided_cfg.cpu.vectorize.assume_inner_stride_one = False
    x = ps.DEFAULTS.spatial_counters[0]

    return {
        "arith": ([Assignment(c.center(), sp.sqrt(sp.Abs(a.center())) * 2 + b[1, 0] / a[0, 1])], rvv_config()),
        "minmax": ([Assignment(c.center(), sp.Max(a.center(), sp.Min(b.center(), 1.5)))], rvv_config()),
        "casts": (
            [Assignment(d64.center(), tcast(s32.center(), "float64") + tcast(i32.center(), "float64")),
             Assignment(i32.center(), tcast(d64.center(), "int32"))],
            rvv_config(),
        ),
        "counter": ([Assignment(d64.center(), tcast(x, "float64") * 0.5)], rvv_config()),
        "reduction": ([reduction_assignment(w, ReductionOp.Max, d64.center())], rvv_config()),
        "strided": ([Assignment(strided.center(), 3 * strided.center())], strided_cfg),
        "philox32": (philox_asms("float32"), rvv_config("float32", index_dtype="int32")),
        "philox64": (philox_asms("float64"), rvv_config("float64", index_dtype="int64")),
    }


@pytest.mark.parametrize("name", list(_compile_kernels().keys()))
def test_generated_code_compiles(name):
    cxx = _riscv_cxx()
    if cxx is None:
        pytest.skip("no RISC-V C++ compiler available")

    from pystencils.include import get_pystencils_include_path

    asms, cfg = _compile_kernels()[name]
    ker = ps.create_kernel(asms, cfg)
    source = "\n".join(
        [f"#include {h}" for h in sorted(ker.required_headers)]
        + ["#define RESTRICT __restrict__", ker.get_c_code()]
    )

    cmd = [
        cxx, "-std=c++17", "-fsyntax-only", "-march=rv64gcv_zvl256b",
        f"-I{get_pystencils_include_path()}", "-x", "c++", "-",
    ]
    result = subprocess.run(cmd, input=source, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stderr}\n--- source ---\n{source}"
