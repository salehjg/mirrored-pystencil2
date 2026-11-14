"""Code generation tests for the RVV platform (VLS, VLEN = 256).

These only generate code and inspect it, so they run on any host. The last test
additionally compiles the generated code with a RISC-V compiler, if one is available
(the native `g++` on a riscv64 host, or the `riscv64-linux-gnu-g++` cross compiler),
which catches misspelled intrinsics without having to run anything.
"""

import platform
import re
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
    #   a + 2 b and a - b[1, 0] (sympy: a + (-1) b) are multiply-adds with a scalar factor; the
    #   division becomes a multiplication with the reciprocal 1 / (...) (vfrdiv: scalar / vector)
    for op in ("vfmacc_vf", "vfrdiv_vf", "vfmul_vv"):
        assert f"__riscv_{op}_{sfx}(" in code


def _vector_loop_ops(code: str, sfx: str) -> list[str]:
    """The RVV operations (loads and stores excluded) of the first, i.e. the vectorized, loop."""
    start = code.index("for(")
    body = code[start : code.index("for(", start + 4)]
    return [op for op in re.findall(rf"__riscv_(\w+?)_{sfx}\(", body) if not op.startswith(("vle", "vse"))]


def _on_symbols(expr_fn, dtype="float64"):
    """d = expr(x, y, z) with x, y, z loaded from fields first, as lbmpy's subexpressions are."""
    a, b, c, d = fields(f"a, b, c, d: {dtype}[1D]")
    x, y, z = sp.symbols("x y z")
    return [
        Assignment(x, a.center()),
        Assignment(y, b.center()),
        Assignment(z, c.center()),
        Assignment(d.center(), expr_fn(x, y, z)),
    ]


@pytest.mark.parametrize(
    "expr, ops",
    [
        (lambda x, y, z: x + 3, ["vfadd_vf"]),
        (lambda x, y, z: x * 3, ["vfmul_vf"]),
        (lambda x, y, z: 3 - x, ["vfrsub_vf"]),
        (lambda x, y, z: x - y, ["vfsub_vv"]),  # two vectors stay .vv
    ],
)
def test_scalar_operands_use_vf_forms(expr, ops):
    assert _vector_loop_ops(rvv_code(_on_symbols(expr)), "f64m1") == ops


@pytest.mark.parametrize(
    "expr, op",
    [
        (lambda x, y, z: x * y + z, "vfmacc_vv"),
        (lambda x, y, z: x * y - z, "vfmsac_vv"),
        (lambda x, y, z: z - x * y, "vfnmsac_vv"),
        (lambda x, y, z: 2 * x + z, "vfmacc_vf"),
        (lambda x, y, z: 2 * x - z, "vfmsac_vf"),
    ],
)
def test_fused_multiply_add(expr, op):
    assert _vector_loop_ops(rvv_code(_on_symbols(expr)), "f64m1") == [op]


def test_scalar_addend_is_not_fused():
    #   An FMA would need the scalar 3 splatted into a vector register: product, then .vf addition
    ops = _vector_loop_ops(rvv_code(_on_symbols(lambda x, y, z: x * y + 3)), "f64m1")
    assert sorted(ops) == ["vfadd_vf", "vfmul_vv"]


@pytest.mark.parametrize("dtype, ctype, sfx", [("float64", "double", "f64m1"), ("float32", "float", "f32m1")])
def test_hoisted_constants_are_scalar_operands(dtype, ctype, sfx):
    #   Rational constants are hoisted out of the loop as vector declarations. Each one gets a
    #   scalar twin, which the loop passes to .vf operations instead of holding the constant in
    #   a vector register.
    a, b = fields(f"a, b: {dtype}[1D]")
    code = rvv_code(
        [Assignment(b.center(), a.center() * sp.Rational(1, 6) + sp.Rational(1, 7))], rvv_config(dtype)
    )
    twins = re.findall(rf"const {ctype} (\w+_s) = ", code)
    assert len(twins) == 2
    ops = _vector_loop_ops(code, sfx)
    assert ops and all(op.endswith("_vf") for op in ops)
    loop = code[code.index("for(") :]
    assert all(t in loop for t in twins)
    #   constant twins are hidden from the optimizer, so GCC keeps them in FP registers instead
    #   of reloading them from the constant pool in every iteration
    for t in twins:
        assert re.search(rf"const {ctype} {t} = pystencils::runtime::rvv::in_register\(", code)


def test_stencil_accesses_use_row_pointers():
    #   Every contiguous access of the vector loop goes through its own row base pointer, declared
    #   before the loop (`_data_src_row[ctr]`), so that no access adds a stencil offset such as
    #   x - 1 to a shared pointer register in every iteration. The remainder loop is unchanged.
    src, dst = fields("src, dst: float64[2D]")
    code = rvv_code([Assignment(dst.center(), src[0, -1] + src[0, 1] + 2 * src[1, 0])])
    m = re.search(r"for\(int64_t (ctr_\d)__1 = ", code)
    ctr, start = m.group(1), m.start()
    vloop = code[start : code.index("for(", start + 4)]
    accesses = re.findall(r"__riscv_v[ls]e64_v_f64m1\(&(\w+)\[([^\]]*)\]", vloop)
    assert len(accesses) == 4
    assert all(re.fullmatch(r"_data_(src|dst)_row(__\d+)?", row) and idx == ctr for row, idx in accesses)
    rows = re.findall(r"(_data_\w+_row\w*) = &_data_(src|dst)\[([^\]]*)\];", code)
    assert sorted(f for _, f, _ in rows) == ["dst", "src", "src", "src"]
    assert all(ctr not in idx and code.index(r) < start for r, _, idx in rows)
    #   the x - 1 / x + 1 offsets moved into the row pointers
    assert sorted(idx.split(" + ")[-1] for _, f, idx in rows if f == "src" and idx.endswith("LL")) == ["-1LL", "1LL"]


def test_reassigned_vector_symbol_keeps_its_register():
    #   A reduction accumulator starts as a broadcast of 0 but changes in the loop: it must not
    #   be replaced by a scalar twin (that would add 0 to each load instead of accumulating)
    f = fields("f: float64[1D]")
    w = TypedSymbol("w", "float64")
    code = rvv_code([reduction_assignment(w, ReductionOp.Add, f.center())])
    assert re.search(r"(\w+) = __riscv_vfadd_vv_f64m1\(\1, ", code)


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
#   Software prefetch (`cpu.vectorize.prefetch_distance`)
#   -------------------------------------------------------------------------------------


def _prefetches(code: str) -> list[tuple[str, str]]:
    import re

    return re.findall(r"pystencils::runtime::rvv::prefetch_(read|write)\(([^;]*)\);", code)


def test_prefetch_off_by_default():
    f, g = fields("f, g: float64[2D]")
    assert not _prefetches(rvv_code([Assignment(g.center(), f[1, 0] + f[-1, 0])]))


def test_prefetch_one_per_distinct_access():
    f, g = fields("f, g: float32[2D]")
    cfg = rvv_config("float32")
    cfg.cpu.vectorize.prefetch_distance = 128
    #   f[1, 0] is read twice: one prefetch; g is written: a write prefetch
    code = rvv_code([Assignment(g.center(), f[1, 0] * f[1, 0] + f[0, 1])], cfg)
    pf = _prefetches(code)
    assert sorted(kind for kind, _ in pf) == ["read", "read", "write"]
    #   128 bytes ahead = 32 floats, and only inside the vector loop (not the remainder loop)
    assert all(addr.endswith(" + 32LL]") for _, addr in pf)
    vec_loop = code[code.index("ctr_1__1 += 8LL") :]
    assert vec_loop.index("prefetch_read") < vec_loop.index("__riscv_vle32_v_f32m1")
    assert code.count("prefetch_") == 3


def test_prefetch_read_and_written_address_gets_write_prefetch():
    f = fields("f: float64[2D]")
    cfg = rvv_config()
    cfg.cpu.vectorize.prefetch_distance = 64
    pf = _prefetches(rvv_code([Assignment(f.center(), 2 * f.center())], cfg))
    assert [kind for kind, _ in pf] == ["write"]
    assert pf[0][1].endswith(" + 8LL]")


def test_prefetch_needs_vector_platform():
    f, g = fields("f, g: float64[2D]")
    cfg = ps.CreateKernelConfig(target=Target.X86_AVX)
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.assume_inner_stride_one = True
    cfg.cpu.vectorize.prefetch_distance = 64
    with pytest.raises(MaterializationError):
        ps.create_kernel([Assignment(g.center(), f.center())], cfg)


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
    prefetch_cfg = rvv_config("float32")
    prefetch_cfg.cpu.vectorize.prefetch_distance = 128

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
        "prefetch": ([Assignment(s32.center(), 2 * s32[1])], prefetch_cfg),
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
