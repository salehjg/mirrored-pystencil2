from __future__ import annotations

from typing import cast, Sequence, Callable
from fractions import Fraction
from functools import reduce

import numpy as np

from ...types import (
    PsType,
    PsVectorType,
    PsCustomType,
    PsScalarType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
    PsPointerType,
    PsShortArrayType,
    PsVoidType,
    constify,
)
from ..exceptions import MaterializationError

from ..constants import PsConstant
from ..functions import CFunction, PsMathFunction, MathFunctions, PsVecStripmineLanes
from .generic_cpu import GenericVectorCpu
from ..kernelcreation import KernelCreationContext, Typifier
from ..transformations.select_intrinsics import SelectIntrinsics, SelectionContext
from ..ast.astnode import PsAstNode
from ..ast.expressions import (
    PsExpression,
    PsSymbolExpr,
    PsConstantExpr,
    PsAddressOf,
    PsMemAcc,
    PsUnOp,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsIntDiv,
    PsRem,
    PsNeg,
    PsCast,
    PsCall,
    PsSubscript,
    PsBitwiseAnd,
    PsBitwiseOr,
    PsBitwiseXor,
    PsLeftShift,
    PsRightShift,
)
from ..ast.structural import (
    PsStatement,
    PsBlock,
    PsLoop,
    PsDeclaration,
    PsAssignment,
)
from ..ast.vector import PsVecMemAcc, PsVecBroadcast, PsVecHorizontal, ReductionOp
from ..ast.iteration import dfs_preorder
from ..memory import PsSymbol

from ...codegen.config import VectorizationMode


#   Maximum element width supported by the vector unit (ELEN). RVV mandates >= 32;
#   all shipping application-class cores (incl. the SpacemiT K1) implement 64.
_ELEN = 64

_LEGAL_LMUL = frozenset(
    Fraction(n, d)
    for n, d in [(1, 8), (1, 4), (1, 2), (1, 1), (2, 1), (4, 1), (8, 1)]
)

#   size_t / ptrdiff_t as they appear in the RVV intrinsic signatures on rv64
_SIZE_T = PsUnsignedIntegerType(64)
_PTRDIFF_T = PsSignedIntegerType(64)


class RvvCpu(GenericVectorCpu):
    """Platform modeling the RISC-V Vector extension (RVV 1.0).

    A *scalable* vector platform, structurally analogous to `SveCpu`, but threading a
    ``size_t vl`` count argument through every intrinsic instead of an ``svbool_t``
    predicate. Two code generation modes are supported (see `VectorizationMode`):

    - ``FIXED`` (VLS): the count is the compile-time constant lane count.
    - ``STRIPMINE`` (VLA): the count is the stripmine loop's ``vl`` symbol.

    Args:
        ctx: The kernel creation context
        vlen: Assumed (VLS) or minimum (VLA) hardware ``VLEN`` in bits
        mode: Code generation mode
        check_vlen: Emit the D8 runtime guard at kernel entry
    """

    def __init__(
        self,
        ctx: KernelCreationContext,
        vlen: int,
        mode: VectorizationMode = VectorizationMode.FIXED,
        check_vlen: bool = True,
    ):
        super().__init__(ctx)
        if vlen is None:
            raise MaterializationError(
                "The RVV backend requires the assumed/minimum VLEN to be set "
                "(cpu.rvv.vlen)."
            )
        self._vlen = vlen
        self._mode = mode
        self._check_vlen = check_vlen

    @property
    def required_headers(self) -> set[str]:
        headers = {'"pystencils_runtime/rvv.hpp"'}
        return super().required_headers | headers

    def get_intrinsic_selector(
        self, use_builtin_convertvector: bool = False
    ) -> SelectIntrinsics:
        return SelectIntrinsicsRvv(
            self._ctx,
            vlen=self._vlen,
            mode=self._mode,
            check_vlen=self._check_vlen,
            use_builtin_convertvector=use_builtin_convertvector,
        )

    def select_function(self, call: PsCall) -> PsExpression:
        func = call.function
        if isinstance(func, PsVecStripmineLanes):
            #   Lower the VLA stripmine step to `vsetvl` for the reference config.
            lmul_suffix = (
                f"m{func.lmul}" if func.lmul >= 1 else f"mf{int(1 / func.lmul)}"
            )
            vsetvl = CFunction(
                f"__riscv_vsetvl_e{func.sew}{lmul_suffix}", (_SIZE_T,), _SIZE_T
            )
            return vsetvl(call.args[0])
        return super().select_function(call)


class RvvSelectionContext(SelectionContext):
    """Selection context for the RVV backend.

    In VLA mode this holds a stack of active-lane (``vl``) expressions pushed when
    entering a stripmined loop body (wired up in WP6). In VLS mode it is currently a
    thin wrapper, kept so that a per-``(SEW,LMUL)`` symbol cache could be introduced
    later without touching call sites.
    """

    def __init__(self, visitor: SelectIntrinsicsRvv):
        super().__init__(visitor)
        self._active_lanes: list[PsExpression] = []

    def push_active_lanes(self, vl: PsExpression) -> None:
        self._active_lanes.append(vl)

    def pop_active_lanes(self) -> None:
        self._active_lanes.pop()

    @property
    def active_lanes(self) -> PsExpression | None:
        return self._active_lanes[-1] if self._active_lanes else None


class SelectIntrinsicsRvv(SelectIntrinsics):
    """Lower IR vector types and operations to RVV-1.0 intrinsics."""

    def __init__(
        self,
        ctx: KernelCreationContext,
        vlen: int,
        mode: VectorizationMode = VectorizationMode.FIXED,
        check_vlen: bool = True,
        use_builtin_convertvector: bool = False,
    ):
        super().__init__(ctx, use_builtin_convertvector=use_builtin_convertvector)
        self._vlen = vlen
        self._mode = mode
        self._check_vlen = check_vlen
        self._typify = Typifier(ctx)
        self._used_vector = False
        #   D14: symbols declared as broadcasts of a scalar leaf, mapped to that scalar.
        #   Uses of these are folded into `.vf`/`.vx` scalar-operand op forms, freeing the
        #   vector register the broadcast would otherwise pin for the kernel's lifetime.
        self._broadcast_scalars: dict[PsSymbol, PsExpression] = {}

    def __call__(self, node: PsBlock) -> PsBlock:
        sc = RvvSelectionContext(self)
        self._collect_broadcast_scalars(node)
        node = cast(PsBlock, self.visit(node, sc))
        #   Only guard kernels that actually contain vector operations; a scalar kernel
        #   compiled for the RVV target needs no VLEN precondition.
        if self._check_vlen and self._used_vector:
            node.statements = [self._make_vlen_guard()] + node.statements
        return node

    def visit(self, node: PsAstNode, sc: SelectionContext) -> PsAstNode:
        #   VLA: when entering a stripmine loop, push its granted-count symbol so that
        #   every intrinsic emitted inside the loop body takes `vl`. Operations that were
        #   hoisted *out* of the loop (e.g. broadcasts of loop-invariant scalars) are
        #   visited with an empty stack and correctly fall back to `vsetvlmax`.
        if self._mode == VectorizationMode.STRIPMINE and isinstance(node, PsLoop):
            vl_symb = self._stripmine_vl_symbol(node)
            if vl_symb is not None:
                rvv_sc = cast(RvvSelectionContext, sc)
                rvv_sc.push_active_lanes(PsExpression.make(vl_symb))
                try:
                    return super().visit(node, sc)
                finally:
                    rvv_sc.pop_active_lanes()
        return super().visit(node, sc)

    @staticmethod
    def _stripmine_vl_symbol(loop: PsLoop) -> PsSymbol | None:
        """If ``loop`` is a stripmine loop, return the ``vl`` symbol it assigns.

        Detected structurally: the loop body assigns a symbol from a
        `PsVecStripmineLanes` call (``vl = vsetvl(remaining)``). The symbol is declared
        outside the loop (see `AxisExpansion.stripmine`), so it appears here as an
        assignment, not a declaration.
        """
        for stmt in loop.body.statements:
            match stmt:
                case (
                    PsAssignment(PsSymbolExpr(symb), PsCall(PsVecStripmineLanes(), _))
                    | PsDeclaration(PsSymbolExpr(symb), PsCall(PsVecStripmineLanes(), _))
                ):
                    return symb
        return None

    #   -------------------------------------------------------------------------
    #   Broadcast folding into .vf / .vx scalar-operand forms (design decision D14)
    #   -------------------------------------------------------------------------

    def _collect_broadcast_scalars(self, node: PsAstNode) -> None:
        """Record symbols declared as an invariant broadcast of a scalar leaf.

        Only leaf scalars (a scalar symbol or constant) are recorded; uses of such a
        symbol as a direct operand of ``+ - * /`` are folded into a scalar-operand
        intrinsic, and the (then dead) broadcast declaration is elided by the C compiler.

        A symbol that is **reassigned** anywhere (e.g. a reduction accumulator / modulo
        variable, whose *initial* value is a broadcast of the identity) must NOT be
        folded — its later value is a genuine vector, not the constant it was seeded with.
        """
        reassigned: set[PsSymbol] = set()
        for n in dfs_preorder(node):
            match n:
                case PsAssignment(PsSymbolExpr(symb), _):
                    reassigned.add(symb)

        for n in dfs_preorder(node):
            match n:
                case PsDeclaration(PsSymbolExpr(symb), PsVecBroadcast(_, operand)) if (
                    isinstance(operand, (PsSymbolExpr, PsConstantExpr))
                    and symb not in reassigned
                ):
                    self._broadcast_scalars[symb] = operand
                case PsDeclaration(PsSymbolExpr(symb), PsConstantExpr(c)) if (
                    isinstance(c.get_dtype(), PsVectorType) and symb not in reassigned
                ):
                    #   EliminateConstants hoists the equilibrium constants as all-equal
                    #   constant-vector declarations (not PsVecBroadcast); fold those too.
                    values = np.asarray(c.value).reshape(-1)
                    if len(values) > 0 and bool(np.all(values == values[0])):
                        scalar_t = cast(PsVectorType, c.get_dtype()).scalar_type
                        self._broadcast_scalars[symb] = PsExpression.make(
                            PsConstant(values[0], scalar_t)
                        )

    def _broadcast_scalar_of(self, operand: PsExpression) -> PsExpression | None:
        """If ``operand`` is (or references) a broadcast of a scalar leaf, return the
        scalar expression; otherwise ``None``."""
        match operand:
            case PsVecBroadcast(_, inner) if isinstance(
                inner, (PsSymbolExpr, PsConstantExpr)
            ):
                return inner
            case PsSymbolExpr(symb) if symb in self._broadcast_scalars:
                return self._broadcast_scalars[symb]
            case PsConstantExpr(c) if isinstance(c.get_dtype(), PsVectorType):
                #   An all-equal constant vector (e.g. the equilibrium constants
                #   3.0/4.5/1.5) is a broadcast; fold it into the scalar-operand form.
                values = np.asarray(c.value).reshape(-1)
                if len(values) > 0 and bool(np.all(values == values[0])):
                    scalar_t = cast(PsVectorType, c.get_dtype()).scalar_type
                    return PsExpression.make(PsConstant(values[0], scalar_t))
                return None
            case _:
                return None

    def visit_expr(self, expr: PsExpression, sc: SelectionContext) -> PsExpression:
        #   D14: fold `op(vec, broadcast(s))` (and the reverse) into a `.vf`/`.vx` form.
        if isinstance(expr, (PsAdd, PsMul, PsSub, PsDiv)) and isinstance(
            expr.get_dtype(), PsVectorType
        ):
            o1, o2 = expr.operand1, expr.operand2
            s1 = self._broadcast_scalar_of(o1)
            s2 = self._broadcast_scalar_of(o2)
            if s2 is not None and s1 is None:
                folded = self._fold_scalar_op(expr, o1, s2, reverse=False, sc=sc)
                if folded is not None:
                    return folded
            elif s1 is not None and s2 is None:
                folded = self._fold_scalar_op(expr, o2, s1, reverse=True, sc=sc)
                if folded is not None:
                    return folded
        return super().visit_expr(expr, sc)

    def _fold_scalar_op(
        self,
        expr: PsBinOp,
        vec_operand: PsExpression,
        scalar: PsExpression,
        reverse: bool,
        sc: SelectionContext,
    ) -> PsExpression | None:
        vtype = cast(PsVectorType, expr.get_dtype())
        sctype = vtype.scalar_type
        is_float = isinstance(sctype, PsIeeeFloatType)
        suffix = self._type_suffix(vtype)
        op = "vf" if is_float else "vx"

        stem: str | None
        match expr:
            case PsAdd():
                stem = f"vfadd_{op}_{suffix}" if is_float else f"vadd_{op}_{suffix}"
            case PsMul():
                stem = f"vfmul_{op}_{suffix}" if is_float else f"vmul_{op}_{suffix}"
            case PsSub() if not reverse:
                stem = f"vfsub_{op}_{suffix}" if is_float else f"vsub_{op}_{suffix}"
            case PsSub():  # reverse: scalar - vector
                stem = f"vfrsub_{op}_{suffix}" if is_float else f"vrsub_{op}_{suffix}"
            case PsDiv() if is_float and not reverse:
                stem = f"vfdiv_vf_{suffix}"
            case PsDiv() if is_float:  # reverse: scalar / vector
                stem = f"vfrdiv_vf_{suffix}"
            case _:
                #   No integer reverse-divide (and forward int div fold not worthwhile);
                #   fall back to the materialized-broadcast .vv path.
                stem = None

        if stem is None:
            return None

        vec = self.visit_expr(vec_operand, sc)
        vtype_intrin = self._vtype_intrin(vtype)
        func = CFunction(
            f"__riscv_{stem}", (vtype_intrin, sctype, _SIZE_T), vtype_intrin
        )
        return func(vec, scalar.clone(), self._vl(vtype, sc))

    def _make_vlen_guard(self) -> PsStatement:
        guard = CFunction(
            "pystencils::runtime::rvv::require_vlen", (_SIZE_T,), PsVoidType()
        )
        return PsStatement(
            guard(PsExpression.make(PsConstant(self._vlen, _SIZE_T)))
        )

    #   -------------------------------------------------------------------------
    #   LMUL / type derivation (design decision D2)
    #   -------------------------------------------------------------------------

    def _lmul(self, vtype: PsVectorType) -> Fraction:
        w = vtype.scalar_type.width
        lanes = vtype.vector_entries
        lmul = Fraction(lanes * w, self._vlen)

        if lmul not in _LEGAL_LMUL:
            legal_lanes = sorted(
                int(m * self._vlen / w)
                for m in _LEGAL_LMUL
                if (m * self._vlen) % w == 0 and Fraction(w) / m <= _ELEN
            )
            raise MaterializationError(
                f"Illegal RVV register group: {lanes} lanes of {w}-bit elements at "
                f"VLEN={self._vlen} implies LMUL={lmul}, which is not in "
                f"{{1/8,1/4,1/2,1,2,4,8}}. Legal lane counts for this (dtype, VLEN): "
                f"{legal_lanes}."
            )

        #   Type-existence constraint: fractional LMUL types only exist while
        #   SEW <= LMUL * ELEN, i.e. w / LMUL <= ELEN (e.g. vfloat64mf2_t does not exist).
        if Fraction(w) / lmul > _ELEN:
            raise MaterializationError(
                f"RVV vector type for {w}-bit elements at LMUL={lmul} does not exist "
                f"(requires SEW <= LMUL*ELEN, i.e. w/LMUL <= {_ELEN})."
            )

        return lmul

    def _lmul_suffix(self, vtype: PsVectorType) -> str:
        lmul = self._lmul(vtype)
        if lmul >= 1:
            return f"m{int(lmul)}"
        else:
            return f"mf{int(1 / lmul)}"

    @staticmethod
    def _type_char(sctype: PsScalarType) -> str:
        match sctype:
            case PsIeeeFloatType():
                return "f"
            case PsUnsignedIntegerType():
                return "u"
            case PsSignedIntegerType():
                return "i"
            case _:
                raise MaterializationError(
                    f"Invalid scalar type for RVV vector operation: {sctype}"
                )

    def _type_suffix(self, vtype: PsVectorType) -> str:
        """Intrinsic name suffix for a vector type, e.g. ``f64m1``, ``i32m2``, ``u32mf2``."""
        sctype = vtype.scalar_type
        return f"{self._type_char(sctype)}{sctype.width}{self._lmul_suffix(vtype)}"

    def _vtype_intrin(self, vtype: PsVectorType) -> PsCustomType:
        sctype = vtype.scalar_type
        match sctype:
            case PsIeeeFloatType():
                stem = f"vfloat{sctype.width}"
            case PsUnsignedIntegerType():
                stem = f"vuint{sctype.width}"
            case PsSignedIntegerType():
                stem = f"vint{sctype.width}"
            case _:
                raise MaterializationError(
                    f"Don't know RVV intrinsic type for vector type {vtype}."
                )
        self._used_vector = True
        return PsCustomType(f"{stem}{self._lmul_suffix(vtype)}_t")

    def _m1_vtype(self, vtype: PsVectorType) -> PsVectorType:
        """The LMUL=1 sibling of a vector type (used for reduction results)."""
        m1_lanes = self._vlen // vtype.scalar_type.width
        return PsVectorType(vtype.scalar_type, m1_lanes)

    def type_intrinsic(
        self, vector_type: PsVectorType | PsShortArrayType, sc: SelectionContext
    ) -> PsCustomType:
        match vector_type:
            case PsVectorType():
                return self._vtype_intrin(vector_type)
            case PsShortArrayType():
                return self._tuple_type_intrin(vector_type)

    def _tuple_type_intrin(self, array_type: PsShortArrayType) -> PsCustomType:
        btype = array_type.base_type
        if not isinstance(btype, PsVectorType):
            raise MaterializationError(
                f"Cannot select intrinsic for non-vectorial short array type {array_type}"
            )
        if array_type.num_elements not in (2, 3, 4, 5, 6, 7, 8):
            raise MaterializationError(
                f"Cannot select intrinsic for tuple type {array_type} (num_elements must be 2..8)"
            )
        base_stem = self._vtype_intrin(btype).c_string()[:-2]  # strip trailing "_t"
        return PsCustomType(f"{base_stem}x{array_type.num_elements}_t")

    #   -------------------------------------------------------------------------
    #   Active-length (vl) operand (design decision D3)
    #   -------------------------------------------------------------------------

    def _vl(self, vtype: PsVectorType, sc: SelectionContext) -> PsExpression:
        """The ``vl`` count operand for an operation on ``vtype``."""
        if self._mode == VectorizationMode.STRIPMINE:
            active = cast(RvvSelectionContext, sc).active_lanes
            if active is not None:
                return active.clone()
            #   Outside the stripmined body (accumulator init / horizontal reduce):
            #   operate at VLMAX for the reference config.
            return self._vsetvlmax(vtype)
        #   VLS: the compile-time constant lane count.
        return PsExpression.make(PsConstant(vtype.vector_entries, _SIZE_T))

    def _vsetvlmax(self, vtype: PsVectorType) -> PsExpression:
        suffix = f"e{vtype.scalar_type.width}{self._lmul_suffix(vtype)}"
        f = CFunction(f"__riscv_vsetvlmax_{suffix}", (), _SIZE_T)
        return f()

    #   -------------------------------------------------------------------------
    #   Constants (design decision D5)
    #   -------------------------------------------------------------------------

    def constant_intrinsic(self, c: PsConstant, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, c.get_dtype())
        sctype = vtype.scalar_type
        values = [v for v in np.asarray(c.value).reshape(-1)]

        prog = _arithmetic_progression(values)

        if prog is not None:
            a, d = prog
            if d == 0:
                #   All-equal constant vector == broadcast of the scalar `a`.
                return self._broadcast(vtype)(
                    PsExpression.make(PsConstant(a, sctype)), self._vl(vtype, sc)
                )
            if isinstance(sctype, PsIntegerType):
                return self._vid_progression(vtype, int(a), int(d), sc)

        #   Fallback: arbitrary per-lane constants via a vslide1down chain.
        if self._mode == VectorizationMode.STRIPMINE:
            raise MaterializationError(
                "VLA mode cannot express arbitrary (non-affine) per-lane constant "
                f"vectors length-agnostically: {c}"
            )
        return self._slide_chain(vtype, values, sc)

    def _vid_progression(
        self, vtype: PsVectorType, a: int, d: int, sc: SelectionContext
    ) -> PsExpression:
        """Construct the integer progression ``a + i*d`` using ``vid.v``."""
        sctype = cast(PsIntegerType, vtype.scalar_type)
        w = sctype.width
        lanes = vtype.vector_entries

        u_vtype = PsVectorType(PsUnsignedIntegerType(w), lanes)
        u_intrin = self._vtype_intrin(u_vtype)
        vid = CFunction(
            f"__riscv_vid_v_{self._type_suffix(u_vtype)}", (_SIZE_T,), u_intrin
        )(self._vl(vtype, sc))

        vtype_intrin = self._vtype_intrin(vtype)
        if isinstance(sctype, PsSignedIntegerType):
            result = self._reinterpret(u_vtype, vtype, vid)
        else:
            result = vid

        suffix = self._type_suffix(vtype)
        if d != 1:
            result = CFunction(
                f"__riscv_vmul_vx_{suffix}",
                (vtype_intrin, sctype, _SIZE_T),
                vtype_intrin,
            )(result, PsExpression.make(PsConstant(d, sctype)), self._vl(vtype, sc))
        if a != 0:
            result = CFunction(
                f"__riscv_vadd_vx_{suffix}",
                (vtype_intrin, sctype, _SIZE_T),
                vtype_intrin,
            )(result, PsExpression.make(PsConstant(a, sctype)), self._vl(vtype, sc))

        return result

    def _reinterpret(
        self, src: PsVectorType, dst: PsVectorType, expr: PsExpression
    ) -> PsExpression:
        f = CFunction(
            f"__riscv_vreinterpret_v_{self._type_suffix(src)}_{self._type_suffix(dst)}",
            (self._vtype_intrin(src),),
            self._vtype_intrin(dst),
        )
        return f(expr)

    def _slide_chain(
        self, vtype: PsVectorType, values: Sequence, sc: SelectionContext
    ) -> PsExpression:
        sctype = vtype.scalar_type
        vtype_intrin = self._vtype_intrin(vtype)
        suffix = self._type_suffix(vtype)

        undef = CFunction(f"__riscv_vundefined_{suffix}", (), vtype_intrin)()

        if isinstance(sctype, PsIeeeFloatType):
            slide = CFunction(
                f"__riscv_vfslide1down_vf_{suffix}",
                (vtype_intrin, sctype, _SIZE_T),
                vtype_intrin,
            )
        else:
            slide = CFunction(
                f"__riscv_vslide1down_vx_{suffix}",
                (vtype_intrin, sctype, _SIZE_T),
                vtype_intrin,
            )

        def push(acc: PsExpression, val) -> PsExpression:
            return slide(
                acc, PsExpression.make(PsConstant(val, sctype)), self._vl(vtype, sc)
            )

        return reduce(push, values, undef)

    #   -------------------------------------------------------------------------
    #   Operations
    #   -------------------------------------------------------------------------

    def op_intrinsic(
        self,
        expr: PsUnOp | PsBinOp,
        operands: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        if isinstance(expr, PsVecHorizontal):
            vtype = cast(PsVectorType, expr.vector_operand.get_dtype())
            #   operands == [scalar_operand, vector_operand]
            return self._hreduce(expr, vtype, operands, sc)

        vtype = cast(PsVectorType, expr.get_dtype())

        if isinstance(expr, PsVecBroadcast):
            return self._broadcast(vtype)(*operands, self._vl(vtype, sc))

        if isinstance(expr, PsCast):
            return self._cast_intrinsic(expr, operands[0], sc)

        intrin = self._op_intrin(expr, vtype)
        return intrin(*operands, self._vl(vtype, sc))

    def _broadcast(self, vtype: PsVectorType) -> Callable[..., PsExpression]:
        sctype = vtype.scalar_type
        vtype_intrin = self._vtype_intrin(vtype)
        suffix = self._type_suffix(vtype)
        if isinstance(sctype, PsIeeeFloatType):
            name = f"__riscv_vfmv_v_f_{suffix}"
        else:
            name = f"__riscv_vmv_v_x_{suffix}"
        return CFunction(name, (sctype, _SIZE_T), vtype_intrin)

    def _op_intrin(
        self, op: PsUnOp | PsBinOp, vtype: PsVectorType
    ) -> CFunction:
        sctype = vtype.scalar_type
        is_float = isinstance(sctype, PsIeeeFloatType)
        is_unsigned = isinstance(sctype, PsUnsignedIntegerType)
        suffix = self._type_suffix(vtype)
        vtype_intrin = self._vtype_intrin(vtype)

        nargs = 1 if isinstance(op, PsUnOp) else 2
        atypes: tuple[PsType, ...] = (vtype_intrin,) * nargs + (_SIZE_T,)

        match op:
            case PsAdd():
                stem = f"vfadd_vv_{suffix}" if is_float else f"vadd_vv_{suffix}"
            case PsSub():
                stem = f"vfsub_vv_{suffix}" if is_float else f"vsub_vv_{suffix}"
            case PsMul():
                stem = f"vfmul_vv_{suffix}" if is_float else f"vmul_vv_{suffix}"
            case PsDiv():
                if is_float:
                    stem = f"vfdiv_vv_{suffix}"
                elif is_unsigned:
                    stem = f"vdivu_vv_{suffix}"
                else:
                    stem = f"vdiv_vv_{suffix}"
            case PsIntDiv():
                stem = f"vdivu_vv_{suffix}" if is_unsigned else f"vdiv_vv_{suffix}"
            case PsRem():
                stem = f"vremu_vv_{suffix}" if is_unsigned else f"vrem_vv_{suffix}"
            case PsBitwiseAnd():
                stem = f"vand_vv_{suffix}"
            case PsBitwiseOr():
                stem = f"vor_vv_{suffix}"
            case PsBitwiseXor():
                stem = f"vxor_vv_{suffix}"
            case PsLeftShift():
                stem = f"vsll_vv_{suffix}"
            case PsRightShift():
                #   logical shift for unsigned, arithmetic for signed
                stem = f"vsrl_vv_{suffix}" if is_unsigned else f"vsra_vv_{suffix}"
            case PsNeg():
                stem = f"vfneg_v_{suffix}" if is_float else f"vneg_v_{suffix}"
            case _:
                raise MaterializationError(
                    f"Unable to select RVV operation intrinsic for {type(op).__name__} "
                    f"on {vtype}"
                )

        return CFunction(f"__riscv_{stem}", atypes, vtype_intrin)

    #   -------------------------------------------------------------------------
    #   Casts (design decision D11)
    #   -------------------------------------------------------------------------

    def _cast_intrinsic(
        self, expr: PsCast, operand: PsExpression, sc: SelectionContext
    ) -> PsExpression:
        target = cast(PsVectorType, expr.get_dtype())
        src = expr.operand.get_dtype()

        if not (
            isinstance(target, PsVectorType)
            and isinstance(src, PsVectorType)
            and target.vector_entries == src.vector_entries
        ):
            raise MaterializationError(
                f"Unable to select RVV intrinsic for vector cast from {src} to {target}"
            )

        st, tt = src.scalar_type, target.scalar_type
        sw, tw = st.width, tt.width
        src_float = isinstance(st, PsIeeeFloatType)
        tgt_float = isinstance(tt, PsIeeeFloatType)
        tgt_uint = isinstance(tt, PsUnsignedIntegerType)
        src_uint = isinstance(st, PsUnsignedIntegerType)
        tgt_suffix = self._type_suffix(target)

        def conv(name: str) -> PsExpression:
            f = CFunction(
                name, (self._vtype_intrin(src), _SIZE_T), self._vtype_intrin(target)
            )
            return f(operand, self._vl(target, sc))

        #   -- same width --------------------------------------------------------
        if sw == tw:
            if src_float and not tgt_float:
                #   float -> int: C truncates, so use the round-toward-zero variant.
                xu = "xu" if tgt_uint else "x"
                return conv(f"__riscv_vfcvt_rtz_{xu}_f_v_{tgt_suffix}")
            if not src_float and tgt_float:
                xu = "xu" if src_uint else "x"
                return conv(f"__riscv_vfcvt_f_{xu}_v_{tgt_suffix}")
            if src_float and tgt_float:
                return operand  # same-width float->float is a no-op
            #   int -> int, same width, differing signedness == reinterpret
            return self._reinterpret(src, target, operand)

        #   -- float <-> float, width-changing ----------------------------------
        if src_float and tgt_float:
            if tw > sw:
                return conv(f"__riscv_vfwcvt_f_f_v_{tgt_suffix}")
            return conv(f"__riscv_vfncvt_f_f_w_{tgt_suffix}")

        #   -- int <-> int, width-changing --------------------------------------
        if not src_float and not tgt_float:
            if tw > sw:
                factor = tw // sw
                if factor not in (2, 4, 8):
                    raise MaterializationError(
                        f"Unsupported integer widening factor {factor} ({src}->{target})"
                    )
                op = "vzext" if src_uint else "vsext"
                return conv(f"__riscv_{op}_vf{factor}_{tgt_suffix}")
            return self._int_narrow(src, target, operand, sc)

        #   -- float <-> int, width-changing ------------------------------------
        if src_float and not tgt_float:
            xu = "xu" if tgt_uint else "x"
            if tw > sw:
                return conv(f"__riscv_vfwcvt_rtz_{xu}_f_v_{tgt_suffix}")
            return conv(f"__riscv_vfncvt_rtz_{xu}_f_w_{tgt_suffix}")
        else:  # int -> float, width-changing
            xu = "xu" if src_uint else "x"
            if tw > sw:
                return conv(f"__riscv_vfwcvt_f_{xu}_v_{tgt_suffix}")
            return conv(f"__riscv_vfncvt_f_{xu}_w_{tgt_suffix}")

    def _int_narrow(
        self,
        src: PsVectorType,
        target: PsVectorType,
        operand: PsExpression,
        sc: SelectionContext,
    ) -> PsExpression:
        """Narrow an integer vector by repeated 2x ``vncvt`` steps."""
        lanes = target.vector_entries
        int_cls = PsUnsignedIntegerType if target.is_uint() else PsSignedIntegerType

        cur = operand
        cur_vtype = src
        cur_w = src.scalar_type.width
        tgt_w = target.scalar_type.width

        while cur_w > tgt_w:
            next_w = cur_w // 2
            next_vtype = PsVectorType(int_cls(next_w), lanes)
            name = f"__riscv_vncvt_x_x_w_{self._type_suffix(next_vtype)}"
            f = CFunction(
                name,
                (self._vtype_intrin(cur_vtype), _SIZE_T),
                self._vtype_intrin(next_vtype),
            )
            cur = f(cur, self._vl(next_vtype, sc))
            cur_vtype = next_vtype
            cur_w = next_w

        return cur

    def _hreduce(
        self,
        expr: PsVecHorizontal,
        vtype: PsVectorType,
        operands: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        if self._mode == VectorizationMode.STRIPMINE:
            raise MaterializationError(
                "VLA-mode vectorized reductions are not yet supported: the tail-undisturbed "
                "accumulator commit (design decision D7) is deferred. Use FIXED (VLS) mode "
                "for reduction kernels."
            )

        scalar_operand, vector_operand = operands
        sctype = vtype.scalar_type
        is_float = isinstance(sctype, PsIeeeFloatType)
        is_unsigned = isinstance(sctype, PsUnsignedIntegerType)
        w = sctype.width

        reduction_op = expr.reduction_op
        actual_op = (
            ReductionOp.Add if reduction_op == ReductionOp.Sub else reduction_op
        )

        m1_vtype = self._m1_vtype(vtype)
        m1_intrin = self._vtype_intrin(m1_vtype)
        in_suffix = self._type_suffix(vtype)
        m1_suffix = self._type_suffix(m1_vtype)
        vtype_intrin = self._vtype_intrin(vtype)

        scalar_combine: Callable[[PsExpression, PsExpression], PsExpression]
        init_scalar: PsExpression
        match actual_op:
            case ReductionOp.Add:
                redop = "vfredosum" if is_float else "vredsum"
                init_scalar = PsExpression.make(PsConstant(0, sctype))
                scalar_combine = PsAdd
            case ReductionOp.Min:
                if is_float:
                    redop = "vfredmin"
                    init_scalar = self._pos_infinity(sctype)
                else:
                    redop = "vredminu" if is_unsigned else "vredmin"
                    init_scalar = PsExpression.make(
                        PsConstant(_int_max(sctype), sctype)
                    )
                scalar_combine = PsMathFunction(MathFunctions.Min)
            case ReductionOp.Max:
                if is_float:
                    redop = "vfredmax"
                    init_scalar = -self._pos_infinity(sctype)
                else:
                    redop = "vredmaxu" if is_unsigned else "vredmax"
                    init_scalar = PsExpression.make(
                        PsConstant(_int_min(sctype), sctype)
                    )
                scalar_combine = PsMathFunction(MathFunctions.Max)
            case ReductionOp.Mul:
                raise MaterializationError(
                    "Horizontal multiplication is not available on RVV"
                )
            case _:
                raise MaterializationError(
                    f"Unsupported reduction op {reduction_op}"
                )

        #   Build the scalar-in-vector accumulator init at LMUL=1.
        if is_float:
            init_func = CFunction(
                f"__riscv_vfmv_s_f_{m1_suffix}", (sctype, _SIZE_T), m1_intrin
            )
        else:
            init_func = CFunction(
                f"__riscv_vmv_s_x_{m1_suffix}", (sctype, _SIZE_T), m1_intrin
            )
        init = init_func(init_scalar, self._vl(vtype, sc))

        #   The reduction itself: vs2 == input vector, vs1 == init (element 0), -> m1.
        reduce_func = CFunction(
            f"__riscv_{redop}_vs_{in_suffix}_{m1_suffix}",
            (vtype_intrin, m1_intrin, _SIZE_T),
            m1_intrin,
        )
        reduced = reduce_func(vector_operand, init, self._vl(vtype, sc))

        #   Extract lane 0 to a scalar.
        if is_float:
            extract = CFunction(
                f"__riscv_vfmv_f_s_{m1_suffix}_{self._type_char(sctype)}{w}",
                (m1_intrin,),
                sctype,
            )
        else:
            extract = CFunction(
                f"__riscv_vmv_x_s_{m1_suffix}_{self._type_char(sctype)}{w}",
                (m1_intrin,),
                sctype,
            )
        extracted = extract(reduced)

        return self._typify(scalar_combine(scalar_operand, extracted))

    def _pos_infinity(self, sctype: PsScalarType) -> PsExpression:
        return CFunction(
            f"std::numeric_limits< {sctype.c_string()} >::infinity", (), sctype
        )()

    #   -------------------------------------------------------------------------
    #   Math functions
    #   -------------------------------------------------------------------------

    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        vtype = cast(PsVectorType, expr.get_dtype())
        func = expr.function
        assert isinstance(func, PsMathFunction)

        sctype = vtype.scalar_type
        is_float = isinstance(sctype, PsIeeeFloatType)
        suffix = self._type_suffix(vtype)
        vtype_intrin = self._vtype_intrin(vtype)

        nargs = func.arg_count
        atypes: tuple[PsType, ...] = (vtype_intrin,) * nargs + (_SIZE_T,)

        match func.func:
            case MathFunctions.Abs if is_float:
                stem = f"vfabs_v_{suffix}"
            case MathFunctions.Sqrt if is_float:
                stem = f"vfsqrt_v_{suffix}"
            case MathFunctions.Min if is_float:
                stem = f"vfmin_vv_{suffix}"
            case MathFunctions.Max if is_float:
                stem = f"vfmax_vv_{suffix}"
            case MathFunctions.Min if vtype.is_int():
                stem = (
                    f"vminu_vv_{suffix}" if vtype.is_uint() else f"vmin_vv_{suffix}"
                )
            case MathFunctions.Max if vtype.is_int():
                stem = (
                    f"vmaxu_vv_{suffix}" if vtype.is_uint() else f"vmax_vv_{suffix}"
                )
            case _:
                raise MaterializationError(
                    f"Unable to select RVV intrinsic for math function {func} on {vtype}"
                )

        return CFunction(f"__riscv_{stem}", atypes, vtype_intrin)(
            *operands, self._vl(vtype, sc)
        )

    #   -------------------------------------------------------------------------
    #   Memory access (design decision D6)
    #   -------------------------------------------------------------------------

    def vector_load(self, acc: PsVecMemAcc, sc: SelectionContext) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)
        sctype = vtype.scalar_type
        w = sctype.width
        suffix = self._type_suffix(vtype)
        vtype_intrin = self._vtype_intrin(vtype)
        ptr_type = PsPointerType(constify(sctype))

        addr = self._typify(PsAddressOf(PsMemAcc(acc.pointer, acc.offset)))

        if acc.stride is not None:
            bstride = self._byte_stride(acc.stride, w)
            ld = CFunction(
                f"__riscv_vlse{w}_v_{suffix}",
                (ptr_type, _PTRDIFF_T, _SIZE_T),
                vtype_intrin,
            )
            return ld(addr, bstride, self._vl(vtype, sc))
        else:
            ld = CFunction(
                f"__riscv_vle{w}_v_{suffix}", (ptr_type, _SIZE_T), vtype_intrin
            )
            return ld(addr, self._vl(vtype, sc))

    def vector_store(
        self, acc: PsVecMemAcc, arg: PsExpression, sc: SelectionContext
    ) -> PsExpression:
        vtype = cast(PsVectorType, acc.dtype)
        sctype = vtype.scalar_type
        w = sctype.width
        suffix = self._type_suffix(vtype)
        vtype_intrin = self._vtype_intrin(vtype)
        ptr_type = PsPointerType(sctype)

        addr = self._typify(PsAddressOf(PsMemAcc(acc.pointer, acc.offset)))

        if acc.stride is not None:
            bstride = self._byte_stride(acc.stride, w)
            st = CFunction(
                f"__riscv_vsse{w}_v_{suffix}",
                (ptr_type, _PTRDIFF_T, vtype_intrin, _SIZE_T),
                PsVoidType(),
            )
            return st(addr, bstride, arg, self._vl(vtype, sc))
        else:
            st = CFunction(
                f"__riscv_vse{w}_v_{suffix}",
                (ptr_type, vtype_intrin, _SIZE_T),
                PsVoidType(),
            )
            return st(addr, arg, self._vl(vtype, sc))

    def _byte_stride(self, elem_stride: PsExpression, width: int) -> PsExpression:
        itemsize = width // 8
        return self._typify(
            PsCast(
                _PTRDIFF_T,
                elem_stride.clone() * PsExpression.make(PsConstant(itemsize)),
            )
        )

    #   -------------------------------------------------------------------------
    #   RNG / tuple subscript (design decision D13) -- implemented in WP3
    #   -------------------------------------------------------------------------

    def rng_engine_intrinsic(
        self, expr: PsCall, args: Sequence[PsExpression], sc: SelectionContext
    ) -> PsExpression:
        from ..functions import PsRngEngineFunction

        assert isinstance(expr.function, PsRngEngineFunction)
        spec = expr.function.rng_spec

        value_vtype = spec.short_array_type.base_type
        ctr_vtype = spec.int_arg_type
        if not isinstance(value_vtype, PsVectorType) or not isinstance(
            ctr_vtype, PsVectorType
        ):
            raise MaterializationError(
                f"Cannot select intrinsic RNG engine {spec.rng_name}: expected vectorized spec."
            )

        if self._mode == VectorizationMode.STRIPMINE:
            raise MaterializationError(
                "VLA-mode vectorized RNG is not yet supported (design decision D13); "
                "threading `vl` through the Philox helpers is deferred."
            )

        if value_vtype.scalar_type.width != 32:
            raise MaterializationError(
                "The RVV backend currently only supports 32-bit (float32) Philox RNG; "
                f"got value type {value_vtype}. f64 RNG is deferred (D13)."
            )

        #   The hand-written Philox kernels run at LMUL=1 / VLMAX(e32m1); the kernel's
        #   counter lane count must therefore equal VLEN/32 exactly.
        if self._lmul(ctr_vtype) != Fraction(1):
            raise MaterializationError(
                f"RVV Philox RNG requires LMUL=1 counter vectors (lanes == VLEN/32 = "
                f"{self._vlen // 32} at VLEN={self._vlen}); got {ctr_vtype.vector_entries} "
                f"lanes ({ctr_vtype})."
            )

        return self._common_rng_engine_intrinsic(expr, args, sc, namespace="rvv")

    def subscript_intrinsic(
        self,
        subscript: PsSubscript,
        arr: PsExpression,
        indices: Sequence[PsExpression],
        sc: SelectionContext,
    ) -> PsExpression:
        array_type = subscript.array.get_dtype()
        if not isinstance(array_type, PsShortArrayType):
            raise MaterializationError(
                f"Cannot materialize subscript on vector array type {array_type}"
            )
        btype = cast(PsVectorType, array_type.base_type)
        tuple_type = self._tuple_type_intrin(array_type)
        base_suffix = self._type_suffix(btype)
        vget = CFunction(
            f"__riscv_vget_v_{base_suffix}x{array_type.num_elements}_{base_suffix}",
            (tuple_type, PsUnsignedIntegerType(64)),
            self._vtype_intrin(btype),
        )
        return vget(arr, indices[0])


#   -------------------------------------------------------------------------
#   Helpers
#   -------------------------------------------------------------------------


def _arithmetic_progression(values: Sequence) -> tuple | None:
    """If ``values`` form an arithmetic progression ``a + i*d``, return ``(a, d)``.

    Otherwise return ``None``. Works for numpy scalar element types.
    """
    n = len(values)
    if n == 0:
        return None
    a = values[0]
    if n == 1:
        return (a, type(a)(0) if hasattr(a, "__sub__") else 0)
    d = values[1] - values[0]
    for i in range(2, n):
        if values[i] - values[i - 1] != d:
            return None
    return (a, d)


def _int_max(sctype: PsIntegerType):
    return np.iinfo(sctype.numpy_dtype).max


def _int_min(sctype: PsIntegerType):
    return np.iinfo(sctype.numpy_dtype).min
