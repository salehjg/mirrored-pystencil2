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

from ..ast import PsAstNode
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
    PsNeg,
    PsConstantExpr,
    PsSymbolExpr,
    PsCast,
    PsCall,
    PsSubscript,
)
from ..ast.vector import PsVecMemAcc, PsVecBroadcast, PsVecHorizontal
from ..ast.structural import PsBlock, PsDeclaration, PsAssignment, PsStructuralNode
from ..memory import PsSymbol
from ...sympyextensions import ReductionOp
from ...types import (
    PsCustomType,
    PsVectorType,
    PsPointerType,
    PsScalarType,
    PsType,
    PsVoidType,
    PsShortArrayType,
    constify,
    deconstify,
)
from ...types.quick import Fp, SInt, UInt
from ..constants import PsConstant
from ..exceptions import MaterializationError
from ..functions import CFunction, PsMathFunction, MathFunctions
from ..kernelcreation import KernelCreationContext, Typifier
from ..transformations import SelectIntrinsics, EliminateConstants, InsertPrefetches, HoistRowPointers
from ..transformations.select_intrinsics import SelectionContext
from ..ast.analysis import collect_undefined_symbols
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

    def prepare_vector_loops(self, kernel_ast: PsBlock) -> PsBlock:
        """One row base pointer per contiguous access of the innermost loops (`HoistRowPointers`).

        RISC-V has no register + register addressing and RVV loads/stores take no immediate
        offset, so without this GCC adds stencil offsets such as ``x - 1`` to a shared pointer
        in every iteration and spills the extra addresses.
        """
        return cast(PsBlock, HoistRowPointers(self._ctx)(kernel_ast))

    def insert_prefetches(self, kernel_ast: PsBlock, distance_bytes: int) -> PsBlock:
        """Zicbop software prefetches (`prefetch.r` / `prefetch.w`) in the innermost vector loops.

        The calls go to ``pystencils::runtime::rvv::prefetch_{read,write}`` (rvv.hpp). Their
        encodings are RISC-V HINTs, so the kernels stay correct, and the prefetches become
        no-ops, on cores without Zicbop.
        """

        def make_prefetch(addr: PsExpression, elem_type: PsType, write: bool) -> PsExpression:
            name = "pystencils::runtime::rvv::prefetch_" + ("write" if write else "read")
            func = CFunction(name, (PsPointerType(elem_type),), PsVoidType())
            call = func(addr)
            call.dtype = PsVoidType()
            return call

        return cast(PsBlock, InsertPrefetches(self._ctx, distance_bytes, make_prefetch)(kernel_ast))


class SelectIntrinsicsRvv(SelectIntrinsics):
    """RVV intrinsic selection.

    Beyond the one-to-one mapping of IR operations to intrinsics, two rewrites keep the
    vector loop short:

    - Scalar operands: an operand that is the same in every lane (a uniform constant, a
      broadcast, or a vector symbol declared as one of those, such as the hoisted constants
      of a kernel) is passed as a scalar to the ``.vf`` form of the operation
      (``vfadd_vf``, ``vfrsub_vf``, ...), instead of occupying a vector register. Every such
      vector declaration gets a scalar twin declared right before it, unless the symbol is
      assigned again later (e.g. a reduction accumulator).
    - Fused multiply-add: ``a * b + c``, ``a * b - c`` and ``c - a * b`` become ``vfmacc``,
      ``vfmsac`` and ``vfnmsac``. RVV intrinsics are opaque to the compiler, so it cannot
      contract them itself as it does for scalar code and x86 vector code by default.

    Scalar twins whose value is a compile-time constant are passed through
    ``pystencils::runtime::rvv::in_register`` (rvv.hpp). Otherwise GCC reloads each of them
    from the constant pool in every loop iteration (an address computation plus a load per
    use) instead of keeping it in one of the floating-point registers, which a vector loop
    leaves idle.
    """

    def __init__(self, ctx: KernelCreationContext):
        super().__init__(ctx)

        typify = Typifier(ctx)
        elim_constants = EliminateConstants(ctx)

        def type_fold(expr):
            return elim_constants(typify(expr))

        self._type_fold = type_fold
        #   Uniform floating-point vector symbols -> their scalar twins
        self._uniform: dict[PsSymbol, PsSymbol] = dict()
        #   id(IR expression) -> (expression, is uniform); holding the expression keeps the id unique
        self._is_uniform_memo: dict[int, tuple[PsExpression, bool]] = dict()
        #   Symbols assigned anywhere besides their declaration: never uniform
        self._reassigned: set[PsSymbol] = set()

    def __call__(self, node: PsBlock) -> PsBlock:
        def collect(n: PsAstNode) -> None:
            if isinstance(n, PsAssignment) and not isinstance(n, PsDeclaration) and isinstance(n.lhs, PsSymbolExpr):
                self._reassigned.add(n.lhs.symbol)
            for c in n.children:
                collect(c)

        collect(node)
        return super().__call__(node)

    def visit(self, node: PsAstNode, sc: SelectionContext) -> PsAstNode:
        if not isinstance(node, PsBlock):
            return super().visit(node, sc)

        statements: list[PsStructuralNode] = []
        for stmt in node.statements:
            if isinstance(stmt, PsDeclaration) and isinstance(stmt.lhs, PsSymbolExpr):
                vtype = deconstify(stmt.lhs.get_dtype())
                if (
                    isinstance(vtype, PsVectorType)
                    and vtype.scalar_type.is_float()
                    and stmt.lhs.symbol not in self._reassigned
                ):
                    scalar = self._uniform_scalar(stmt.rhs)
                    if scalar is not None:
                        vsymb = stmt.lhs.symbol
                        stype = vtype.scalar_type
                        ssymb = self._ctx.get_new_symbol(f"{vsymb.name}_s", constify(stype))
                        self._uniform[vsymb] = ssymb
                        if not collect_undefined_symbols(scalar):
                            scalar = self._in_register(scalar, stype)
                        statements.append(PsDeclaration(PsSymbolExpr(ssymb), scalar))
            statements.append(cast(PsStructuralNode, self.visit(stmt, sc)))
        node.statements = statements
        return node

    def _is_uniform(self, expr: PsExpression) -> bool:
        """Whether all lanes of the IR vector expression `expr` are provably equal (memoized: the
        selector asks again for every enclosing operation)."""
        memo = self._is_uniform_memo.get(id(expr))
        if memo is not None:
            return memo[1]
        match expr:
            case PsConstantExpr(c) if isinstance(c.dtype, PsVectorType):
                values = list(c.value)
                uniform = all(v.tobytes() == values[0].tobytes() for v in values)
            case PsVecBroadcast():
                uniform = True
            case PsSymbolExpr(symb):
                uniform = symb in self._uniform
            case PsAdd() | PsSub() | PsMul() | PsDiv():
                uniform = self._is_uniform(expr.operand1) and self._is_uniform(expr.operand2)
            case PsNeg():
                uniform = self._is_uniform(expr.operand)
            case _:
                uniform = False
        self._is_uniform_memo[id(expr)] = (expr, uniform)
        return uniform

    def _uniform_scalar(self, expr: PsExpression) -> PsExpression | None:
        """The scalar value of every lane of the IR vector expression `expr`, if all lanes are equal."""
        if not self._is_uniform(expr):
            return None
        result: PsExpression | None = None
        match expr:
            case PsConstantExpr(c) if isinstance(c.dtype, PsVectorType):
                values = list(c.value)
                if all(v.tobytes() == values[0].tobytes() for v in values):
                    result = PsConstantExpr(PsConstant(values[0], c.dtype.scalar_type))
            case PsVecBroadcast():
                result = expr.operand.clone()
            case PsSymbolExpr(symb) if symb in self._uniform:
                result = PsSymbolExpr(self._uniform[symb])
            case PsAdd() | PsSub() | PsMul() | PsDiv():
                s1 = self._uniform_scalar(expr.operand1)
                s2 = self._uniform_scalar(expr.operand2)
                if s1 is not None and s2 is not None:
                    result = type(expr)(s1, s2)
            case PsNeg():
                s = self._uniform_scalar(expr.operand)
                if s is not None:
                    result = PsNeg(s)
        return None if result is None else self._type_fold(result)

    #   ---------------------------------------------------------------------------------
    #   Helpers
    #   ---------------------------------------------------------------------------------

    @staticmethod
    def _in_register(scalar: PsExpression, stype: PsScalarType) -> PsExpression:
        """``pystencils::runtime::rvv::in_register(scalar)``: the same value, opaque to the optimizer."""
        func = CFunction("pystencils::runtime::rvv::in_register", (stype,), stype)
        call = func(scalar)
        call.dtype = constify(stype)
        return call

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
            case PsAdd() | PsSub() | PsMul() | PsDiv() if is_float and (
                (special := self._float_binop(expr, operands, vtype)) is not None
            ):
                return special
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

    #   .vf forms (scalar operand) and fused multiply-add, see the class docstring
    _VF_STEMS = {PsAdd: "vfadd_vf", PsSub: "vfsub_vf", PsMul: "vfmul_vf", PsDiv: "vfdiv_vf"}
    _VF_STEMS_REVERSED = {PsAdd: "vfadd_vf", PsSub: "vfrsub_vf", PsMul: "vfmul_vf", PsDiv: "vfrdiv_vf"}

    def _float_binop(
        self, expr: PsBinOp, operands: Sequence[PsExpression], vtype: PsVectorType
    ) -> PsExpression | None:
        s1 = self._uniform_scalar(expr.operand1)
        s2 = self._uniform_scalar(expr.operand2)
        if s1 is not None and s2 is not None:
            return None  # uniform as a whole: computed once, outside the loop

        if isinstance(expr, PsAdd | PsSub):
            fused = self._fma(expr, operands, s1, s2, vtype)
            if fused is not None:
                return fused

        if s2 is not None:
            return self._vf(self._VF_STEMS[type(expr)], vtype, operands[0], s2)
        if s1 is not None:
            return self._vf(self._VF_STEMS_REVERSED[type(expr)], vtype, operands[1], s1)
        return None

    def _vf(self, stem: str, vtype: PsVectorType, vec: PsExpression, scalar: PsExpression) -> PsExpression:
        rtype = _rvv_type(vtype)
        return self._call(
            f"__riscv_{stem}_{_sfx(vtype)}", (rtype, vtype.scalar_type, _SIZE_T), rtype, vec, scalar, self._vl(vtype)
        )

    @staticmethod
    def _as_product(lowered: PsExpression) -> tuple[str, PsExpression, PsExpression] | None:
        """If `lowered` is a ``vfmul`` intrinsic call: its form ('vv' or 'vf') and its two factors,
        a scalar factor (form 'vf') first."""
        if isinstance(lowered, PsCall) and isinstance(lowered.function, CFunction):
            name = lowered.function.name
            if name.startswith("__riscv_vfmul_vv_"):
                return "vv", lowered.args[0], lowered.args[1]
            if name.startswith("__riscv_vfmul_vf_"):
                return "vf", lowered.args[1], lowered.args[0]
        return None

    def _fma(
        self,
        expr: PsAdd | PsSub,
        operands: Sequence[PsExpression],
        s1: PsExpression | None,
        s2: PsExpression | None,
        vtype: PsVectorType,
    ) -> PsExpression | None:
        """``a * b + c`` -> vfmacc(c, a, b), ``a * b - c`` -> vfmsac(c, a, b), ``c - a * b`` -> vfnmsac(c, a, b).

        ``a * b - c`` also arrives as ``(-1) * c + a * b``, which is how sympy writes it."""
        rtype = _rvv_type(vtype)

        def fused(stem: str, form: str, acc: PsExpression, x: PsExpression, y: PsExpression) -> PsExpression:
            x_type = vtype.scalar_type if form == "vf" else rtype
            return self._call(
                f"__riscv_{stem}_{form}_{_sfx(vtype)}", (rtype, x_type, rtype, _SIZE_T), rtype, acc, x, y, self._vl(vtype)
            )

        if isinstance(expr, PsAdd) and s1 is None and s2 is None:
            for neg_idx in (0, 1):
                negated = self._as_negation(operands[neg_idx])
                product = self._as_product(operands[1 - neg_idx])
                if negated is not None and product is not None:
                    return fused("vfmsac", product[0], negated, product[1], product[2])

        for prod_idx in (0, 1):
            if (s1, s2)[1 - prod_idx] is not None:
                continue  # a uniform addend stays a .vf addition
            product = self._as_product(operands[prod_idx])
            if product is None:
                continue
            form, x, y = product
            if isinstance(expr, PsAdd):
                stem = "vfmacc"
            else:
                stem = "vfmsac" if prod_idx == 0 else "vfnmsac"
            return fused(stem, form, operands[1 - prod_idx], x, y)
        return None

    @staticmethod
    def _as_negation(lowered: PsExpression) -> PsExpression | None:
        """If `lowered` is ``vfmul_vf(v, -1)``: v."""
        if (
            isinstance(lowered, PsCall)
            and isinstance(lowered.function, CFunction)
            and lowered.function.name.startswith("__riscv_vfmul_vf_")
            and isinstance(lowered.args[1], PsConstantExpr)
            and lowered.args[1].constant.value == -1
        ):
            return lowered.args[0]
        return None

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
