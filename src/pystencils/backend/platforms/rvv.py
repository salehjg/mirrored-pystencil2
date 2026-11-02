"""RISC-V Vector extension (RVV 1.0) platform -- vector-length-specific, VLEN = 256.

This backend deliberately follows the assumptions of the x86/AVX backend (`x86.py`)
instead of exploiting RVV's scalable programming model:

- The hardware vector register width is a codegen-time constant, `RVV_VLEN` = 256 bits
  (the SpacemiT K1). Like AVX, a vector type is therefore determined by its width in
  bits: a full 256-bit vector is one register (LMUL ``m1``, AVX: ``__m256*``), a
  128-bit vector is half a register (LMUL ``mf2``, AVX: ``__m128*``). Wider vectors
  are rejected, just like on AVX; there is no register grouping (``m2``, ``m4``, ...).
- The loop structure is the generic one (peel + vectorized block loop + scalar
  remainder). Every RVV intrinsic takes an explicit active length ``vl``; we always pass
  the literal lane count. At VLEN = 256 the lane count of every legal vector type equals
  its VLMAX, so this is exactly "one full (fractional) register per operation".
- Only floating-point and signed integer element types are supported (as on x86).
- What x86 implements with helpers in its runtime headers (``_mm256_set_*`` and the
  horizontal reductions), we implement with helpers in ``pystencils_runtime/rvv.hpp``.

Running the generated code on hardware with VLEN < 256 is NOT supported (operations
would silently process fewer lanes). ``rvv.hpp`` refuses to compile unless the compiler
is told VLEN >= 256 (``-march=rv64gcv_zvl256b``), which is the flag the JIT passes.
"""

from __future__ import annotations
from typing import Sequence, cast

from ..ast.expressions import (
    PsExpression,
    PsAddressOf,
    PsMemAcc,
    PsUnOp,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsConstantExpr,
    PsCast,
    PsCall,
    PsSubscript,
)
from ..ast.vector import PsVecMemAcc, PsVecBroadcast, PsVecHorizontal
from ...sympyextensions import ReductionOp
from ...types import (
    PsCustomType,
    PsVectorType,
    PsPointerType,
    PsScalarType,
    PsType,
    PsVoidType,
    PsShortArrayType,
    deconstify,
)
from ...types.quick import Fp, SInt, UInt
from ..constants import PsConstant
from ..exceptions import MaterializationError
from ..functions import CFunction, PsMathFunction, MathFunctions
from ..kernelcreation import KernelCreationContext, Typifier
from ..transformations import SelectIntrinsics, EliminateConstants
from ..transformations.select_intrinsics import SelectionContext
from .generic_cpu import GenericVectorCpu


RVV_VLEN = 256
"""Hardware vector register width (VLEN) in bits assumed by the code generator.

Hardcoded on purpose: fixing VLEN turns RVV into a fixed-width SIMD ISA like AVX,
so none of the vector-length-agnostic machinery is needed. 256 matches the SpacemiT K1.
"""

#   C types of the scalar arguments of RVV intrinsics on rv64
_SIZE_T = UInt(64)  # the active vector length `vl`
_PTRDIFF_T = SInt(64)  # the byte stride of strided loads/stores

_RUNTIME_NS = "pystencils::runtime::rvv"


#   -------------------------------------------------------------------------------------
#   Type naming -- the RVV analogue of `X86VectorArch.intrin_{prefix,suffix,type}`
#   -------------------------------------------------------------------------------------


def _lmul(vtype: PsVectorType) -> str:
    """Register-group multiplier (LMUL) of a vector type, as used in intrinsic names.

    With a fixed VLEN this is just ``width / VLEN``. We only allow the two widths the
    AVX backend allows (256 and 128 bits). Fractional LMUL additionally requires
    ``SEW <= LMUL * ELEN`` (ELEN = 64), so e.g. ``vfloat64mf2_t`` does not exist.
    """
    match vtype.width:
        case w if w == RVV_VLEN:
            return "m1"
        case w if w == RVV_VLEN // 2 and vtype.scalar_type.width <= 32:
            return "mf2"
        case _:
            raise MaterializationError(
                f"RVV (VLEN={RVV_VLEN}) does not support vector type {vtype}: "
                f"vectors must be {RVV_VLEN} bits wide, or {RVV_VLEN // 2} bits "
                "wide with elements of at most 32 bits."
            )


def _elem(stype: PsScalarType) -> tuple[str, int]:
    """Element kind (``float``/``int``) and width; raises for unsupported scalar types."""
    match stype:
        case Fp(32) | Fp(64):
            return "float", stype.width
        case SInt(width):
            return "int", width
        case _:
            raise MaterializationError(f"RVV backend does not support scalar type {stype}")


def _sfx(vtype: PsVectorType) -> str:
    """Type suffix of intrinsic names, e.g. ``f64m1`` or ``i32mf2``."""
    kind, width = _elem(vtype.scalar_type)
    return f"{kind[0]}{width}{_lmul(vtype)}"


def _rvv_type(vtype: PsVectorType, tuple_size: int | None = None) -> PsCustomType:
    """C type of a vector (``vfloat64m1_t``) or of a vector tuple (``vfloat32m1x4_t``)."""
    kind, width = _elem(vtype.scalar_type)
    tuple_part = "" if tuple_size is None else f"x{tuple_size}"
    return PsCustomType(f"v{kind}{width}{_lmul(vtype)}{tuple_part}_t")


class RvvCpu(GenericVectorCpu):
    """Platform modelling the RISC-V Vector extension (RVV 1.0) at a fixed VLEN of 256 bits.

    Intrinsics reference: https://github.com/riscv-non-isa/rvv-intrinsic-doc
    """

    def __init__(self, ctx: KernelCreationContext):
        super().__init__(ctx)

    @property
    def required_headers(self) -> set[str]:
        return super().required_headers | {
            '"pystencils_runtime/rvv.hpp"',
        }

    def get_intrinsic_selector(
        self, use_builtin_convertvector: bool = False
    ) -> SelectIntrinsics:
        if use_builtin_convertvector:
            #   RVV vector types are sizeless; GCC/Clang vector extensions can't hold them.
            raise MaterializationError(
                "`__builtin_convertvector` is not supported on the RVV platform."
            )
        return SelectIntrinsicsRvv(self._ctx)


class SelectIntrinsicsRvv(SelectIntrinsics):
    def __init__(self, ctx: KernelCreationContext):
        super().__init__(ctx)

        typify = Typifier(ctx)
        elim_constants = EliminateConstants(ctx)

        def type_fold(expr):
            return elim_constants(typify(expr))

        self._type_fold = type_fold

    #   ---------------------------------------------------------------------------------
    #   Helpers
    #   ---------------------------------------------------------------------------------

    @staticmethod
    def _vl(vtype: PsVectorType) -> PsExpression:
        """Active vector length passed to each intrinsic: always all lanes (VLS)."""
        return PsConstantExpr(PsConstant(vtype.vector_entries, _SIZE_T))

    @staticmethod
    def _call(
        name: str, arg_types: Sequence[PsType], ret_type: PsType, *args: PsExpression
    ) -> PsExpression:
        """Build an intrinsic call and give it its return type (as `x86.py` does)."""
        func = CFunction(name, tuple(arg_types), ret_type)
        call = func(*args)
        call.dtype = ret_type
        return call

    def _broadcast(self, vtype: PsVectorType, scalar: PsExpression) -> PsExpression:
        """Splat a scalar into all lanes (AVX: ``_mm256_set1_*``)."""
        stem = "vfmv_v_f" if vtype.scalar_type.is_float() else "vmv_v_x"
        return self._call(
            f"__riscv_{stem}_{_sfx(vtype)}",
            (vtype.scalar_type, _SIZE_T),
            _rvv_type(vtype),
            scalar,
            self._vl(vtype),
        )

    #   ---------------------------------------------------------------------------------
    #   Types and constants
    #   ---------------------------------------------------------------------------------

    def type_intrinsic(
        self, vector_type: PsVectorType | PsShortArrayType, sc: SelectionContext
    ) -> PsCustomType:
        match vector_type:
            case PsVectorType():
                return _rvv_type(vector_type)

            case PsShortArrayType(btype, num_elements) if isinstance(
                btype, PsVectorType
            ):
                #   x86 wraps short arrays of vectors in structs; RVV vector types are
                #   sizeless and cannot be struct members, but RVV has native tuple types.
                return _rvv_type(btype, num_elements)

            case _:
                raise MaterializationError(
                    f"Cannot select intrinsic for non-vectorial short array type {vector_type}"
                )

    def constant_intrinsic(self, c: PsConstant, sc: SelectionContext) -> PsExpression:
        vtype = c.dtype
        assert isinstance(vtype, PsVectorType)
        stype = vtype.scalar_type
        values = list(c.value)

        #   Uniform constants (by far the most common: coefficients, weights) are a plain
        #   broadcast. Compare bit patterns so that e.g. 0.0 and -0.0 stay distinct.
        if all(v.tobytes() == values[0].tobytes() for v in values):
            return self._broadcast(vtype, PsConstantExpr(PsConstant(values[0], stype)))

        #   Anything else (e.g. the lane offsets [0, 1, ..., L-1] of a vectorized loop
        #   counter) goes through the `set_*` runtime helper, the analogue of AVX's
        #   `_mm256_set_*`. RVV has no such intrinsic, the helper loads from a stack array.
        args = [PsConstantExpr(PsConstant(v, stype)) for v in values]
        return self._call(
            f"{_RUNTIME_NS}::set_{_sfx(vtype)}",
            (stype,) * len(args),
            _rvv_type(vtype),
            *args,
        )

    #   ---------------------------------------------------------------------------------
    #   Arithmetic, broadcasts, casts, horizontal reductions
    #   ---------------------------------------------------------------------------------

    def op_intrinsic(
        self,
        expr: PsUnOp | PsBinOp,
        operands: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        if isinstance(expr, PsVecHorizontal):
            return self._horizontal(expr, operands)

        vtype = cast(PsVectorType, expr.get_dtype())
        is_float = vtype.scalar_type.is_float()

        match expr:
            case PsVecBroadcast():
                return self._broadcast(vtype, operands[0])
            case PsCast():
                return self._cast(expr, operands[0])
            case PsAdd():
                stem = "vfadd_vv" if is_float else "vadd_vv"
            case PsSub():
                stem = "vfsub_vv" if is_float else "vsub_vv"
            case PsMul():
                stem = "vfmul_vv" if is_float else "vmul_vv"
            case PsDiv():
                stem = "vfdiv_vv" if is_float else "vdiv_vv"
            case _:
                raise MaterializationError(
                    f"Unable to select RVV operation intrinsic for {type(expr)}"
                )

        rtype = _rvv_type(vtype)
        return self._call(
            f"__riscv_{stem}_{_sfx(vtype)}",
            (rtype, rtype, _SIZE_T),
            rtype,
            *operands,
            self._vl(vtype),
        )

    def _horizontal(
        self, expr: PsVecHorizontal, operands: Sequence[PsExpression]
    ) -> PsExpression:
        """Fold a vector into a scalar, e.g. ``horizontal_add_f64m1(dst, v) = dst + sum(v)``.

        Same scheme as x86's ``_mm256_horizontal_*`` helpers. Like there, a subtraction
        reduction accumulates with ``+`` (the accumulator already holds the negated terms).
        """
        vtype = cast(PsVectorType, expr.vector_operand.get_dtype())
        stype = vtype.scalar_type
        if not stype.is_float():
            raise MaterializationError(
                f"RVV horizontal reductions are only available for floating-point types, not {stype}"
            )

        op = ReductionOp.Add if expr.reduction_op == ReductionOp.Sub else expr.reduction_op
        return self._call(
            f"{_RUNTIME_NS}::horizontal_{op.name.lower()}_{_sfx(vtype)}",
            (stype, _rvv_type(vtype)),
            stype,
            *operands,
        )

    #   Conversion intrinsics, keyed by (source is float, target is float).
    #   Legal vector widths are 128 and 256 bits at the same lane count, so a cast
    #   changes the element width by at most a factor of two: one intrinsic suffices.
    #   Float -> int always uses the `rtz` (round towards zero) variant: C truncation.
    _SAME_WIDTH_CVT = {
        (True, False): "vfcvt_rtz_x_f_v",
        (False, True): "vfcvt_f_x_v",
    }
    _WIDENING_CVT = {
        (True, True): "vfwcvt_f_f_v",
        (False, False): "vsext_vf2",
        (False, True): "vfwcvt_f_x_v",
        (True, False): "vfwcvt_rtz_x_f_v",
    }
    _NARROWING_CVT = {
        (True, True): "vfncvt_f_f_w",
        (False, False): "vncvt_x_x_w",
        (False, True): "vfncvt_f_x_w",
        (True, False): "vfncvt_rtz_x_f_w",
    }

    def _cast(self, expr: PsCast, operand: PsExpression) -> PsExpression:
        target = cast(PsVectorType, expr.target_type)
        source = cast(PsVectorType, expr.operand.get_dtype())
        assert target.vector_entries == source.vector_entries

        sw, tw = source.scalar_type.width, target.scalar_type.width
        key = (source.scalar_type.is_float(), target.scalar_type.is_float())

        if tw == sw:
            table = self._SAME_WIDTH_CVT
        elif tw == 2 * sw:
            table = self._WIDENING_CVT
        elif sw == 2 * tw:
            table = self._NARROWING_CVT
        else:
            table = {}

        if key not in table:
            raise MaterializationError(
                f"Unable to select RVV intrinsic for type conversion from {source} to {target}.\n"
                f"    at: {expr}"
            )

        return self._call(
            f"__riscv_{table[key]}_{_sfx(target)}",
            (_rvv_type(source), _SIZE_T),
            _rvv_type(target),
            operand,
            self._vl(target),
        )

    #   ---------------------------------------------------------------------------------
    #   Math functions
    #   ---------------------------------------------------------------------------------

    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        assert isinstance(expr.function, PsMathFunction)

        vtype = cast(PsVectorType, expr.get_dtype())
        rtype = _rvv_type(vtype)
        sfx = _sfx(vtype)
        vl = self._vl(vtype)
        is_float = vtype.scalar_type.is_float()

        match expr.function.func:
            case MathFunctions.Sqrt if is_float:
                return self._call(f"__riscv_vfsqrt_v_{sfx}", (rtype, _SIZE_T), rtype, *operands, vl)

            case MathFunctions.Abs if is_float:
                return self._call(f"__riscv_vfabs_v_{sfx}", (rtype, _SIZE_T), rtype, *operands, vl)

            case MathFunctions.Abs:
                #   RVV has no integer abs instruction: |x| = max(x, -x)
                (x,) = operands
                neg_x = self._call(f"__riscv_vneg_v_{sfx}", (rtype, _SIZE_T), rtype, x.clone(), vl.clone())
                return self._call(f"__riscv_vmax_vv_{sfx}", (rtype, rtype, _SIZE_T), rtype, x, neg_x, vl)

            case MathFunctions.Min | MathFunctions.Max:
                stem = expr.function.func.function_name  # "min" / "max"
                stem = f"vf{stem}_vv" if is_float else f"v{stem}_vv"
                return self._call(f"__riscv_{stem}_{sfx}", (rtype, rtype, _SIZE_T), rtype, *operands, vl)

            case _:
                #   Transcendentals have no RVV instructions (x86 needs SVML for them);
                #   floor/ceil have no direct RVV counterpart either.
                raise MaterializationError(
                    f"RVV does not support {expr.function.func.function_name} on type {vtype}."
                )

    #   ---------------------------------------------------------------------------------
    #   Memory access
    #   ---------------------------------------------------------------------------------

    def _byte_stride(self, acc: PsVecMemAcc) -> PsExpression:
        """RVV strided loads/stores take their stride in bytes, the IR stores elements."""
        assert acc.stride is not None
        itemsize = cast(PsVectorType, acc.dtype).scalar_type.itemsize
        return self._type_fold(PsConstantExpr(PsConstant(itemsize)) * acc.stride.clone())

    def vector_load(self, acc: PsVecMemAcc, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)
        stype = vtype.scalar_type
        ptr_type = PsPointerType(stype, const=True)
        addr = PsAddressOf(PsMemAcc(acc.pointer, acc.offset))
        name = f"vle{stype.width}_v_{_sfx(vtype)}"

        if acc.stride is None:
            return self._call(
                f"__riscv_{name}", (ptr_type, _SIZE_T), _rvv_type(vtype), addr, self._vl(vtype)
            )
        else:
            #   Native strided load (x86 needs a gather here).
            return self._call(
                f"__riscv_vlse{stype.width}_v_{_sfx(vtype)}",
                (ptr_type, _PTRDIFF_T, _SIZE_T),
                _rvv_type(vtype),
                addr,
                self._byte_stride(acc),
                self._vl(vtype),
            )

    def vector_store(
        self, acc: PsVecMemAcc, arg: PsExpression, sc: SelectionContext
    ) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)
        stype = vtype.scalar_type
        ptr_type = PsPointerType(stype)
        addr = PsAddressOf(PsMemAcc(acc.pointer, acc.offset))
        rtype = _rvv_type(vtype)

        if acc.stride is None:
            return self._call(
                f"__riscv_vse{stype.width}_v_{_sfx(vtype)}",
                (ptr_type, rtype, _SIZE_T),
                PsVoidType(),
                addr,
                arg,
                self._vl(vtype),
            )
        else:
            #   Native strided store (x86 needs an AVX-512 scatter here).
            return self._call(
                f"__riscv_vsse{stype.width}_v_{_sfx(vtype)}",
                (ptr_type, _PTRDIFF_T, rtype, _SIZE_T),
                PsVoidType(),
                addr,
                self._byte_stride(acc),
                arg,
                self._vl(vtype),
            )

    #   ---------------------------------------------------------------------------------
    #   Random number generation (results are vector tuples)
    #   ---------------------------------------------------------------------------------

    def rng_engine_intrinsic(
        self, expr: PsCall, args: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        #   -> pystencils::runtime::rvv::random::philox_float{32,64}(ctr..., key...)
        return self._common_rng_engine_intrinsic(expr, args, sc, namespace="rvv")

    def subscript_intrinsic(
        self,
        subscript: PsSubscript,
        arr: PsExpression,
        indices: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        #   Element i of a vector tuple: __riscv_vget_v_f32m1x4_f32m1(arr, i).
        #   The index must be a compile-time constant, which RNG result accesses are.
        arr_type = deconstify(subscript.array.get_dtype())
        assert isinstance(arr_type, PsShortArrayType)
        btype = cast(PsVectorType, arr_type.base_type)
        n = arr_type.num_elements

        return self._call(
            f"__riscv_vget_v_{_sfx(btype)}x{n}_{_sfx(btype)}",
            (_rvv_type(btype, n), _SIZE_T),
            _rvv_type(btype),
            arr,
            indices[0],
        )
