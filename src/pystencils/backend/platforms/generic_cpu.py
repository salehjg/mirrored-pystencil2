from abc import ABC, abstractmethod
import numpy as np

from ..ast.expressions import PsCall, PsMemAcc, PsConstantExpr

from ..functions import (
    CFunction,
    MathFunctions,
    PsIrFunction,
    PsMathFunction,
    PsConstantFunction,
    ConstantFunctions,
    PsRngEngineFunction,
)
from ..reduction_op_mapping import reduction_op_to_expr
from ...sympyextensions import ReductionOp
from ...types import PsIntegerType, PsIeeeFloatType

from .platform import Platform
from ..exceptions import MaterializationError

from ..constants import PsConstant
from ..ast.structural import (
    PsStructuralNode,
    PsAssignment,
)
from ..ast.expressions import PsExpression
from ..kernelcreation import Typifier
from ..transformations import SelectIntrinsics


class GenericCpu(Platform):
    """Generic CPU platform.

    The `GenericCPU` platform models the following execution environment:

     - Generic multicore CPU architecture
     - Iteration space represented by a loop nest, kernels are executed as a whole
     - C standard library math functions available (``#include <math.h>`` or ``#include <cmath>``)
    """

    @property
    def required_headers(self) -> set[str]:
        return {"<cmath>", "<limits>", '"pystencils_runtime/generic_cpu.hpp"'}

    def resolve_reduction(
        self,
        ptr_expr: PsExpression,
        symbol_expr: PsExpression,
        reduction_op: ReductionOp,
    ) -> PsStructuralNode:

        ptr_access = PsMemAcc(
            ptr_expr, PsConstantExpr(PsConstant(0, self._ctx.index_dtype))
        )

        # inspired by OpenMP: local reduction variable (negative sign) is added at the end
        actual_op = ReductionOp.Add if reduction_op is ReductionOp.Sub else reduction_op

        # create binop and potentially select corresponding function for e.g. min or max
        potential_call = reduction_op_to_expr(actual_op, ptr_access, symbol_expr)
        typify = Typifier(self._ctx)
        potential_call = typify(potential_call)

        # if rhs contains a function call, resolve it for the current platform
        rhs: PsExpression
        if isinstance(potential_call, PsCall):
            rhs = self.select_function(potential_call)
        else:
            rhs = potential_call

        return PsAssignment(ptr_access, rhs)

    def select_function(self, call: PsCall) -> PsExpression:
        call_func = call.function
        assert isinstance(call_func, PsIrFunction)

        dtype = call.get_dtype()
        arg_types = (dtype,) * call.function.arg_count

        expr: PsExpression | None = None

        if isinstance(call_func, PsMathFunction | PsConstantFunction):
            func = call_func.func

            if isinstance(dtype, PsIeeeFloatType):
                if dtype.width in (32, 64):
                    match func:
                        case (
                            MathFunctions.Exp
                            | MathFunctions.Log
                            | MathFunctions.Sin
                            | MathFunctions.Cos
                            | MathFunctions.Tan
                            | MathFunctions.Sinh
                            | MathFunctions.Cosh
                            | MathFunctions.Tanh
                            | MathFunctions.ASin
                            | MathFunctions.ACos
                            | MathFunctions.ATan
                            | MathFunctions.ATan2
                            | MathFunctions.Pow
                            | MathFunctions.Sqrt
                            | MathFunctions.Floor
                            | MathFunctions.Ceil
                        ):
                            call.function = CFunction(
                                func.function_name, arg_types, dtype
                            )
                            expr = call
                        case MathFunctions.Abs | MathFunctions.Min | MathFunctions.Max:
                            call.function = CFunction(
                                "f" + func.function_name, arg_types, dtype
                            )
                            expr = call

                match func:
                    case ConstantFunctions.Pi:
                        assert dtype.numpy_dtype is not None
                        expr = PsExpression.make(
                            PsConstant(dtype.numpy_dtype.type(np.pi), dtype)
                        )

                    case ConstantFunctions.E:
                        assert dtype.numpy_dtype is not None
                        expr = PsExpression.make(
                            PsConstant(dtype.numpy_dtype.type(np.e), dtype)
                        )

                    case ConstantFunctions.PosInfinity | ConstantFunctions.NegInfinity:
                        call.function = CFunction(
                            f"std::numeric_limits< {dtype.c_string()} >::infinity",
                            [],
                            dtype,
                        )
                        if func == ConstantFunctions.NegInfinity:
                            expr = -call
                        else:
                            expr = call

            elif isinstance(dtype, PsIntegerType):
                expr = self._select_integer_function(call)

        elif isinstance(call.function, PsRngEngineFunction):
            spec = call.function.rng_spec
            ctr_type = spec.int_arg_type
            atypes = (spec.int_arg_type,) * call.function.arg_count
            rng_func = CFunction(
                f"pystencils::runtime::random::{spec.rng_name}< {ctr_type.c_string()} >",
                atypes,
                spec.short_array_type,
            )

            expr = rng_func(*call.args)

        if expr is not None:
            if expr.dtype is None:
                typify = Typifier(self._ctx)
                expr = typify(expr)
            return expr
        else:
            raise MaterializationError(
                f"No implementation available for function {call_func} on data type {dtype}"
            )


class GenericVectorCpu(GenericCpu, ABC):
    """Base class for CPU platforms with vectorization support through intrinsics."""

    @abstractmethod
    def get_intrinsic_selector(
        self, use_builtin_convertvector: bool = False
    ) -> SelectIntrinsics:
        """Return an instance of a subclass of `SelectIntrinsics`
        to perform vector intrinsic selection for this platform."""
