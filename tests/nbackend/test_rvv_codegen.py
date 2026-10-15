"""WP2 codegen tests for the RVV VLS backend.

These generate C code for ``Target.RISCV_RVV`` and assert on the emitted intrinsic
names / types. They exercise the *entire* driver (loop strategy, materialization,
intrinsic selection, lowering) but stop before compilation, so they run on any host
with no RISC-V toolchain.
"""

import re
import pytest
import sympy as sp

import pystencils as ps
from pystencils import (
    Target,
    CreateKernelConfig,
    create_kernel,
    fields,
    Assignment,
    tcast,
)
from pystencils.backend.exceptions import MaterializationError
from pystencils.sympyextensions import ReductionOp, reduction_assignment


def rvv_c_code(
    assignments,
    *,
    vlen,
    lanes,
    default_dtype="float64",
    index_dtype="int64",
    mode="fixed",
    check_vlen=True,
    assume_inner_stride_one=True,
):
    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = default_dtype
    cfg.index_dtype = index_dtype
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.lanes = lanes
    cfg.cpu.vectorize.mode = mode
    cfg.cpu.vectorize.assume_inner_stride_one = assume_inner_stride_one
    cfg.cpu.rvv.vlen = vlen
    cfg.cpu.rvv.check_vlen = check_vlen
    return create_kernel(assignments, cfg).get_c_code()


#   ---- LMUL derivation (design decision D2) ------------------------------------


@pytest.mark.parametrize(
    "dtype, lanes, vlen, expected_lmul",
    [
        ("float64", 4, 256, "m1"),
        ("float64", 8, 256, "m2"),
        ("float64", 2, 128, "m1"),
        ("float32", 8, 256, "m1"),
        ("float32", 16, 256, "m2"),
        ("float32", 4, 256, "mf2"),
        ("int64", 4, 256, "m1"),
        ("int32", 8, 256, "m1"),
    ],
)
def test_lmul_type_derivation(dtype, lanes, vlen, expected_lmul):
    #   Two distinct source fields so the add is not folded into a scalar multiply.
    a, b, g = fields(f"a, b, g: {dtype}[2D]", layout="fzyx")
    code = rvv_c_code(
        [Assignment(g.center(), a.center() + b.center())],
        vlen=vlen,
        lanes=lanes,
        default_dtype=dtype,
        index_dtype="int64" if "64" in dtype else "int32",
    )
    char = {"float64": "f64", "float32": "f32", "int64": "i64", "int32": "i32"}[dtype]
    w = int(re.search(r"\d+", dtype).group())
    #   The LMUL suffix is encoded in the intrinsic names.
    add = "vfadd" if "float" in dtype else "vadd"
    assert f"__riscv_{add}_vv_{char}{expected_lmul}" in code
    assert f"__riscv_vle{w}_v_{char}{expected_lmul}" in code


def test_illegal_lmul_raises():
    #   f64 x 8 at VLEN=512 -> LMUL = 8*64/512 = 1 (legal). Use x2 at vlen=256 -> mf... illegal for f64.
    f, g = fields("f, g: float64[2D]", layout="fzyx")
    with pytest.raises(MaterializationError):
        #   2 lanes * 64 bit / 512 = 1/4 -> vfloat64mf4_t does not exist
        rvv_c_code(
            [Assignment(g.center(), f.center() + f.center())],
            vlen=512,
            lanes=2,
            default_dtype="float64",
        )


#   ---- Core arithmetic + memory ------------------------------------------------


def test_core_arith_and_memory_f64():
    src, dst = fields("src(2), dst(4): float64[2D]", layout="fzyx")
    x = sp.symbols("x_:4")
    update = [
        Assignment(x[0], src[0, 0](0) + src[0, 0](1)),
        Assignment(x[1], src[0, 0](0) - src[0, 0](1)),
        Assignment(x[2], src[0, 0](0) * src[0, 0](1)),
        Assignment(x[3], src[0, 0](0) / src[0, 0](1)),
        Assignment(dst.center(0), x[0]),
        Assignment(dst.center(1), x[1]),
        Assignment(dst.center(2), x[2]),
        Assignment(dst.center(3), x[3]),
    ]
    code = rvv_c_code(update, vlen=256, lanes=4)

    #   NOTE: sympy canonicalizes `a - b` -> `a + (-1)*b` (so the `-1` broadcast folds to
    #   vfmul_vf under D14) and `a / b` -> `a * (1/b)` (so the reciprocal folds to
    #   vfrdiv_vf). Subtraction/division correctness is covered by the on-board numeric
    #   tests (WP4/WP5).
    for stem in (
        "__riscv_vfadd_vv_f64m1",  # x0 = src0 + src1
        "__riscv_vfmul_vv_f64m1",  # x2 = src0 * src1
        "__riscv_vfrdiv_vf_f64m1",  # x3 = src0 * (1/src1), reciprocal folded (D14)
        "__riscv_vle64_v_f64m1",
        "__riscv_vse64_v_f64m1",
    ):
        assert stem in code, f"missing {stem}"

    #   VLS mode passes the literal lane count as the trailing vl argument
    assert re.search(r"__riscv_vfadd_vv_f64m1\([^;]*,\s*4uLL\)", code)


def test_vlen_guard_emitted_and_suppressible():
    f, g = fields("f, g: float64[2D]", layout="fzyx")
    asm = [Assignment(g.center(), 2 * f.center())]

    code = rvv_c_code(asm, vlen=256, lanes=4)
    assert "pystencils::runtime::rvv::require_vlen(256" in code

    code_off = rvv_c_code(asm, vlen=256, lanes=4, check_vlen=False)
    assert "require_vlen" not in code_off


def test_broadcast_folds_to_scalar_op():
    #   D14: a broadcast scalar operand folds into the .vf scalar-operand form, so NO
    #   `vfmv_v_f` broadcast instruction remains (the scalar is read from the f-register).
    f, g = fields("f, g: float64[2D]", layout="fzyx")
    code = rvv_c_code([Assignment(g.center(), 2.0 * f.center())], vlen=256, lanes=4)
    assert "__riscv_vfmul_vf_f64m1" in code
    assert "__riscv_vfmv_v_f" not in code


def test_broadcast_reverse_forms():
    #   D14 reverse forms for the non-commutative cases (scalar on the left).
    f, g = fields("f, g: float64[2D]", layout="fzyx")
    code = rvv_c_code([Assignment(g.center(), (3.0 - f.center()))], vlen=256, lanes=4)
    assert "__riscv_vfrsub_vf_f64m1" in code
    code = rvv_c_code([Assignment(g.center(), 1.0 / f.center())], vlen=256, lanes=4)
    assert "__riscv_vfrdiv_vf_f64m1" in code
    assert "__riscv_vfmv_v_f" not in code


#   ---- Counter / vid progression (design decision D5) --------------------------


def test_counter_uses_vid():
    from pystencils import DEFAULTS

    f = fields("f(1): int64[1D]", layout="fzyx")
    code = rvv_c_code(
        [Assignment(f(0), DEFAULTS.spatial_counters[0])],
        vlen=256,
        lanes=4,
        default_dtype="int64",
        index_dtype="int64",
    )
    assert "__riscv_vid_v_u64m1" in code


#   ---- Strided access (design decision D6) -------------------------------------


def test_strided_load_uses_vlse():
    from pystencils import DEFAULTS, FieldType

    f = fields("f: float64[2D]", layout="fzyx")
    g = fields("g: float64[2D]", layout="fzyx", field_type=FieldType.CUSTOM)
    i, j = DEFAULTS.spatial_counters[:2]
    update = [
        Assignment(
            f(),
            g.absolute_access((3 * i + 0, j), ())
            + g.absolute_access((3 * i + 1, j), ()),
        )
    ]
    code = rvv_c_code(update, vlen=256, lanes=4)
    assert "__riscv_vlse64_v_f64m1" in code


#   ---- Reductions (design decision D7) -----------------------------------------


@pytest.mark.parametrize(
    "op, redop",
    [
        (ReductionOp.Add, "vfredosum"),
        (ReductionOp.Min, "vfredmin"),
        (ReductionOp.Max, "vfredmax"),
    ],
)
def test_float_reduction_intrinsics(op, redop):
    x = ps.fields("x: float64[1D]")
    w = ps.TypedSymbol("w", "float64")
    red = reduction_assignment(w, op, x.center())

    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = "float64"
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.lanes = 4
    cfg.cpu.rvv.vlen = 256
    code = create_kernel([red], cfg).get_c_code()

    assert f"__riscv_{redop}_vs_f64m1_f64m1" in code
    assert "__riscv_vfmv_s_f_f64m1" in code
    assert "__riscv_vfmv_f_s_f64m1_f64" in code


def test_reduction_accumulator_not_broadcast_folded():
    """Regression: the reduction accumulator is declared as a broadcast of its identity
    but is *reassigned* each iteration, so D14 must NOT fold its uses into a `.vf` form
    (which would drop the accumulation). The in-loop update must stay `.vv`."""
    x = ps.fields("x: float64[1D]")
    w = ps.TypedSymbol("w", "float64")
    red = reduction_assignment(w, ReductionOp.Add, x.center())
    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = "float64"
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.lanes = 4
    cfg.cpu.rvv.vlen = 256
    code = create_kernel([red], cfg).get_c_code()
    #   accumulator update stays vector-vector; no vfadd_vf-with-identity fold
    assert "__riscv_vfadd_vv_f64m1" in code


#   ---- VLA / stripmining (design decision D9) ----------------------------------


def test_vla_loop_shape():
    a, b, g = fields("a, b, g: float64[2D]", layout="c")
    code = rvv_c_code(
        [Assignment(g.center(), a.center() + b.center())],
        vlen=256,
        lanes=None,
        mode="vla",
    )
    #   stripmine loop: vsetvl + advance by the granted count, no peel/remainder loop
    assert "__riscv_vsetvl_e64m1" in code
    assert "_rvv_vl_idx" in code
    assert "+= _rvv_vl_idx" in code
    #   every vectorized op takes the granted vl, not a literal
    assert "__riscv_vfadd_vv_f64m1(" in code
    assert "_rvv_vl)" in code
    #   the granted-count symbol must NOT leak into the kernel signature
    signature = code.split("{", 1)[0]
    assert "_rvv_vl" not in signature


def test_vla_lmul_from_config():
    #   vlen floor 256, lmul 2, f64 -> l_nom = 256*2/64 = 8 -> m2
    f, g = fields("f, g: float64[2D]", layout="c")
    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = "float64"
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.mode = "vla"
    cfg.cpu.vectorize.assume_inner_stride_one = True
    cfg.cpu.rvv.vlen = 256
    cfg.cpu.rvv.lmul = 2
    a, b = fields("a, b: float64[2D]", layout="c")
    code = create_kernel(
        [Assignment(g.center(), a.center() + b.center())], cfg
    ).get_c_code()
    assert "__riscv_vsetvl_e64m2" in code
    assert "__riscv_vfadd_vv_f64m2" in code
    assert "__riscv_vle64_v_f64m2" in code


def test_stripmine_lanes_lowering():
    """PsVecStripmineLanes lowers to vsetvl (RVV) and min(r, l_nom) (GenericCpu)."""
    from pystencils.backend.kernelcreation import KernelCreationContext
    from pystencils.backend.platforms.rvv import RvvCpu
    from pystencils.backend.platforms.generic_cpu import GenericCpu
    from pystencils.backend.functions import PsVecStripmineLanes
    from pystencils.backend.ast.expressions import PsExpression
    from pystencils.backend.constants import PsConstant
    from pystencils.types import PsUnsignedIntegerType
    from pystencils.backend.emission import CAstPrinter

    ctx = KernelCreationContext()
    size_t = PsUnsignedIntegerType(64)
    r = ctx.get_new_symbol("r", size_t)
    call = PsVecStripmineLanes(64, 1, 4)(PsExpression.make(r))

    rvv = RvvCpu(ctx, vlen=256, check_vlen=False)
    rvv_code = CAstPrinter()(rvv.select_function(call.clone()))
    assert "__riscv_vsetvl_e64m1" in rvv_code

    generic = GenericCpu(ctx)
    generic_code = CAstPrinter()(generic.select_function(call.clone()))
    #   min(r, 4): ternary or a comparison against the nominal lane count
    assert "4" in generic_code and ("?" in generic_code or "r" in generic_code)


def test_vla_reduction_raises():
    x = ps.fields("x: float64[1D]")
    w = ps.TypedSymbol("w", "float64")
    red = reduction_assignment(w, ReductionOp.Add, x.center())
    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = "float64"
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.mode = "vla"
    cfg.cpu.rvv.vlen = 256
    with pytest.raises(MaterializationError):
        create_kernel([red], cfg).get_c_code()


#   ---- RNG / tuple types (design decision D13) ---------------------------------


def test_philox_f32_rng_codegen():
    dtype = "float32"
    rng = ps.random.Philox("philox", dtype, seed=0x7EADBEEF, offsets=(0, 0))
    q = rng.vector_size
    f = fields(f"f({q}): {dtype}[2D]", layout="fzyx")
    c = ps.TypedSymbol("c", ps.DynamicType.INDEX_TYPE)
    rx, rasm = rng.get_random_vector(c)
    asms = [rasm] + [Assignment(f.center(i), rx[i]) for i in range(q)]

    #   lanes=4 @ vlen=128 -> u32 counters at LMUL=1 (== VLMAX(e32m1))
    code = rvv_c_code(
        asms, vlen=128, lanes=4, default_dtype=dtype, index_dtype="int32"
    )
    assert "pystencils::runtime::rvv::random::philox_float32" in code
    assert "__riscv_vget_v_f32m1x4_f32m1" in code
    assert "vfloat32m1x4_t" in code


def test_philox_f64_rng_deferred():
    dtype = "float64"
    rng = ps.random.Philox("philox", dtype, seed=0x7EADBEEF, offsets=(0, 0))
    q = rng.vector_size
    f = fields(f"f({q}): {dtype}[2D]", layout="fzyx")
    c = ps.TypedSymbol("c", ps.DynamicType.INDEX_TYPE)
    rx, rasm = rng.get_random_vector(c)
    asms = [rasm] + [Assignment(f.center(i), rx[i]) for i in range(q)]
    with pytest.raises(MaterializationError):
        rvv_c_code(asms, vlen=256, lanes=4, default_dtype=dtype, index_dtype="int64")


def test_rng_wrong_lmul_raises():
    dtype = "float32"
    rng = ps.random.Philox("philox", dtype, seed=1, offsets=(0, 0))
    q = rng.vector_size
    f = fields(f"f({q}): {dtype}[2D]", layout="fzyx")
    c = ps.TypedSymbol("c", ps.DynamicType.INDEX_TYPE)
    rx, rasm = rng.get_random_vector(c)
    asms = [rasm] + [Assignment(f.center(i), rx[i]) for i in range(q)]
    #   lanes=4 @ vlen=256 -> u32 counters at LMUL=1/2, not 1 -> must raise
    with pytest.raises(MaterializationError):
        rvv_c_code(asms, vlen=256, lanes=4, default_dtype=dtype, index_dtype="int32")


#   ---- Casts (design decision D11) ---------------------------------------------


@pytest.mark.parametrize(
    "src, dst, lanes, expected",
    [
        #   same-width float<->int: must use the round-toward-zero (_rtz_) variant
        ("float64", "int64", 4, "__riscv_vfcvt_rtz_x_f_v_i64m1"),
        ("float32", "int32", 8, "__riscv_vfcvt_rtz_x_f_v_i32m1"),
        ("int64", "float64", 4, "__riscv_vfcvt_f_x_v_f64m1"),
        ("int32", "float32", 8, "__riscv_vfcvt_f_x_v_f32m1"),
        #   float widening / narrowing
        ("float32", "float64", 4, "__riscv_vfwcvt_f_f_v_f64m1"),
        ("float64", "float32", 8, "__riscv_vfncvt_f_f_w_f32m1"),
        #   integer widening (single op) / narrowing
        ("int32", "int64", 4, "__riscv_vsext_vf2_i64m1"),
        ("int64", "int32", 8, "__riscv_vncvt_x_x_w_i32m1"),
    ],
)
def test_cast_intrinsics(src, dst, lanes, expected):
    fsrc = fields(f"fsrc: {src}[2D]", layout="fzyx")
    fdst = fields(f"fdst: {dst}[2D]", layout="fzyx")
    #   default_dtype chosen so that the derived lanes give the intended LMUL
    default = src if "float" in src else dst
    code = rvv_c_code(
        [Assignment(fdst.center(), tcast(fsrc.center(), dst))],
        vlen=256,
        lanes=lanes,
        default_dtype=default,
        index_dtype="int64",
    )
    assert expected in code, f"missing {expected}"


#   ---- Integer completeness ops ------------------------------------------------


@pytest.mark.parametrize(
    "builder, expected",
    [
        (lambda a, b: a + b, "__riscv_vadd_vv_i64m1"),
        (lambda a, b: a * b, "__riscv_vmul_vv_i64m1"),
    ],
)
def test_integer_ops(builder, expected):
    #   Bitwise/shift/rem intrinsics are implemented in the platform but are awkward to
    #   express through sympy field accesses (`&` becomes a boolean And); they are
    #   exercised directly at the IR level below.
    a, b, g = fields("a, b, g: int64[2D]", layout="fzyx")
    code = rvv_c_code(
        [Assignment(g.center(), builder(a.center(), b.center()))],
        vlen=256,
        lanes=4,
        default_dtype="int64",
        index_dtype="int64",
    )
    assert expected in code


@pytest.mark.parametrize(
    "op_cls, expected",
    [
        ("PsIntDiv", "__riscv_vdiv_vv_i64m1"),
        ("PsRem", "__riscv_vrem_vv_i64m1"),
        ("PsBitwiseAnd", "__riscv_vand_vv_i64m1"),
        ("PsBitwiseOr", "__riscv_vor_vv_i64m1"),
        ("PsBitwiseXor", "__riscv_vxor_vv_i64m1"),
        ("PsLeftShift", "__riscv_vsll_vv_i64m1"),
        ("PsRightShift", "__riscv_vsra_vv_i64m1"),
    ],
)
def test_integer_bitwise_shift_ir_level(op_cls, expected):
    """Exercise div/rem/bitwise/shift intrinsic selection directly at the IR level."""
    from pystencils.backend.kernelcreation import KernelCreationContext
    from pystencils.backend.platforms.rvv import SelectIntrinsicsRvv, RvvSelectionContext
    from pystencils.backend.ast.expressions import PsExpression, PsSymbolExpr
    from pystencils.backend import ast as _ast_mod
    import pystencils.backend.ast.expressions as exprs
    from pystencils.types import PsVectorType, PsSignedIntegerType
    from pystencils.backend.emission import CAstPrinter

    ctx = KernelCreationContext()
    sel = SelectIntrinsicsRvv(ctx, vlen=256, check_vlen=False)
    sc = RvvSelectionContext(sel)

    vtype = PsVectorType(PsSignedIntegerType(64), 4)
    a = ctx.get_new_symbol("a", sel._vtype_intrin(vtype))
    b = ctx.get_new_symbol("b", sel._vtype_intrin(vtype))
    lhs = PsSymbolExpr(a)
    rhs = PsSymbolExpr(b)

    OpCls = getattr(exprs, op_cls)
    node = OpCls(lhs, rhs)
    node.dtype = vtype

    result = sel.op_intrinsic(node, [lhs, rhs], sc)
    code = CAstPrinter()(result)
    assert expected in code, f"got: {code}"
