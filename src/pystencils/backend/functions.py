from __future__ import annotations
from typing import Sequence, TYPE_CHECKING, cast
from abc import ABC
from enum import Enum, IntEnum
from dataclasses import dataclass

from ..sympyextensions import ReductionOp
from ..sympyextensions.random import RngState, Philox
from ..types import (
    PsType,
    PsScalarType,
    PsNumericType,
    PsTypeError,
    PsIeeeFloatType,
    PsIntegerType,
    PsShortArrayType,
    PsVectorType,
)
from .exceptions import PsInternalCompilerError

if TYPE_CHECKING:
    from .ast.expressions import PsExpression, PsCall


class PsFunction(ABC):
    """Base class for functions occuring in the IR"""

    __match_args__ = ("name", "arg_count")

    def __init__(self, name: str, num_args: int):
        self._name = name
        self._num_args = num_args

    @property
    def name(self) -> str:
        """Name of this function."""
        return self._name

    @property
    def arg_count(self) -> int:
        "Number of arguments this function takes"
        return self._num_args

    def __call__(self, *args: PsExpression) -> PsCall:
        from .ast.expressions import PsCall

        return PsCall(self, args)


class PsIrFunction(PsFunction):
    """Base class for IR functions that must be lowered to target-specific implementations."""


class MathFunctions(Enum):
    """Mathematical functions supported by the backend.

    Each platform has to materialize these functions to a concrete implementation.
    """

    Exp = ("exp", 1)
    Log = ("log", 1)
    Sin = ("sin", 1)
    Cos = ("cos", 1)
    Tan = ("tan", 1)
    Sinh = ("sinh", 1)
    Cosh = ("cosh", 1)
    Tanh = ("tanh", 1)
    ASin = ("asin", 1)
    ACos = ("acos", 1)
    ATan = ("atan", 1)
    Sqrt = ("sqrt", 1)

    Abs = ("abs", 1)
    Floor = ("floor", 1)
    Ceil = ("ceil", 1)

    Min = ("min", 2)
    Max = ("max", 2)

    Pow = ("pow", 2)
    ATan2 = ("atan2", 2)

    def __init__(self, func_name, num_args):
        self.function_name = func_name
        self.num_args = num_args

    def __str__(self) -> str:
        return self.function_name

    def __repr__(self) -> str:
        return f"MathFunctions.{self.name}"


class PsMathFunction(PsIrFunction):
    """Homogeneously typed mathematical functions."""

    __match_args__ = ("func",)

    def __init__(self, func: MathFunctions) -> None:
        super().__init__(func.function_name, func.num_args)
        self._func = func

    @property
    def func(self) -> MathFunctions:
        return self._func

    def __str__(self) -> str:
        return f"{self._func.function_name}"

    def __repr__(self) -> str:
        return f"PsMathFunction({repr(self._func)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsMathFunction):
            return False

        return self._func == other._func

    def __hash__(self) -> int:
        return hash(self._func)


class PsReductionWriteBack(PsIrFunction):
    """Function representing a reduction kernel's write-back step supported by the backend.

    Each platform has to materialize this function to a concrete implementation.
    """

    def __init__(self, reduction_op: ReductionOp) -> None:
        super().__init__("WriteBackToPtr", 2)
        self._reduction_op = reduction_op

    @property
    def reduction_op(self) -> ReductionOp:
        return self._reduction_op

    def __str__(self) -> str:
        return f"{super().name}"

    def __repr__(self) -> str:
        return f"PsReductionWriteBack({repr(self._reduction_op)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsReductionWriteBack):
            return False

        return self._reduction_op == other._reduction_op

    def __hash__(self) -> int:
        return hash(self._reduction_op)


class ConstantFunctions(Enum):
    """Numerical constant functions.

    Each platform has to materialize these functions to a concrete implementation.
    """

    Pi = "pi"
    E = "e"
    PosInfinity = "pos_infinity"
    NegInfinity = "neg_infinity"

    def __init__(self, func_name):
        self.function_name = func_name

    def __str__(self) -> str:
        return self.function_name

    def __repr__(self) -> str:
        return f"ConstantFunctions.{self.name}"


class PsConstantFunction(PsIrFunction):
    """Data-type-specific numerical constants.

    Represents numerical constants which need to be exactly represented,
    e.g. transcendental numbers and non-finite constants.

    Functions of this class are treated the same as `PsConstant` instances
    by most transforms.
    In particular, they are subject to the same contextual typing rules,
    and will be broadcast by the vectorizer.
    """

    __match_args__ = ("func",)

    def __init__(
        self, func: ConstantFunctions, dtype: PsNumericType | None = None
    ) -> None:
        super().__init__(func.function_name, 0)
        self._func = func
        self._set_dtype(dtype)

    @property
    def func(self) -> ConstantFunctions:
        return self._func

    @property
    def dtype(self) -> PsNumericType | None:
        """This constant function's data type, or ``None`` if it is untyped."""
        return self._dtype

    @dtype.setter
    def dtype(self, t: PsNumericType):
        self._set_dtype(t)

    def get_dtype(self) -> PsNumericType:
        """Retrieve this constant function's data type, throwing an exception if it is untyped."""
        if self._dtype is None:
            raise PsInternalCompilerError(
                "Data type of constant  function was not set."
            )
        return self._dtype

    def __str__(self) -> str:
        return f"{self._func.function_name}"

    def __repr__(self) -> str:
        return f"PsConstantFunction({repr(self._func)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsConstantFunction):
            return False

        return (self._func, self._dtype) == (other._func, other._dtype)

    def __hash__(self) -> int:
        return hash((self._func, self._dtype))

    def _set_dtype(self, dtype: PsNumericType | None):
        if dtype is not None:
            match self._func:
                case (
                    ConstantFunctions.Pi
                    | ConstantFunctions.E
                    | ConstantFunctions.PosInfinity
                    | ConstantFunctions.NegInfinity
                ) if not dtype.is_float():
                    raise PsTypeError(
                        f"Invalid type for {self.func.function_name}: {dtype}"
                    )

        self._dtype = dtype


class GpuFpIntrinsics(Enum):
    """GPU floating point intrinsics."""

    dividef = ("dividef", 2)
    """Fast approximate division"""

    SqrtRn = ("sqrt_rn", 1)
    """Fast square root in round-to-nearest-even mode"""

    RSqrtRn = ("rsqrt_rn", 1)
    """Fast reciprocal square root in round-to-nearest-even mode"""

    def __init__(self, func_name, num_args):
        self.function_name = func_name
        self.num_args = num_args

    def __str__(self) -> str:
        return f"{self.function_name}"

    def __repr__(self) -> str:
        return f"GpuFpIntrinsics.{self.name}"


class PsGpuIntrinsicFunction(PsIrFunction):
    """GPU floating point intrinsics"""

    __match_args__ = ("func",)

    def __init__(self, intrin: GpuFpIntrinsics) -> None:
        super().__init__(intrin.function_name, intrin.num_args)
        self._intrin = intrin

    @property
    def func(self) -> GpuFpIntrinsics:
        return self._intrin

    def __str__(self) -> str:
        return f"{self._intrin.function_name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsGpuIntrinsicFunction):
            return False

        return self._intrin == other._intrin

    def __hash__(self) -> int:
        return hash(self._intrin)


class GpuGridDimension(IntEnum):
    X = 0
    Y = 1
    Z = 2


class GpuGridScope(Enum):
    threadIdx = "threadIdx"
    blockIdx = "blockIdx"
    blockDim = "blockDim"
    gridDim = "gridDim"


class PsGpuIndexingFunction(PsIrFunction):
    """Gpu block, thread, and grid indexing functions.

    Calls to IR GPU indexing functions will always typify to the context's index data type.
    Platforms must insert appropriate type casts when materializing.
    """

    __match_args__ = ("scope", "dimension")

    def __init__(self, scope: GpuGridScope, dimension: GpuGridDimension):
        func_name = f"gpu.{scope.name}.{dimension.name}"
        super().__init__(func_name, 0)

        self._scope = scope
        self._dimension = dimension

    @property
    def scope(self) -> GpuGridScope:
        return self._scope

    @property
    def dimension(self) -> GpuGridDimension:
        return self._dimension

    def __str__(self):
        return self._name

    def __eq__(self, other: object):
        if not isinstance(other, PsGpuIndexingFunction):
            return False

        return (self._scope, self._dimension) == (other._scope, other._dimension)

    def __hash__(self):
        return hash(type(self), self._scope, self._dimension)


@dataclass(frozen=True)
class RngSpec:
    """Random number generator specifications for `PsRngEngineFunction`."""

    rng_name: str
    short_array_type: PsShortArrayType
    num_ctrs: int
    num_keys: int
    int_arg_type: PsNumericType

    @staticmethod
    def philox(fptype: PsIeeeFloatType, ctr_type: PsIntegerType) -> RngSpec:
        """Philox W=32, N=4 RNG engine returning four float32-values"""
        if fptype.width not in (32, 64):
            raise ValueError(f"Data type {fptype} not supported in Philox RNG")

        return RngSpec(
            f"philox_{fptype}",
            Philox._get_short_array_type(fptype),
            4,
            2,
            ctr_type,
        )

    def vectorize(self, num_lanes: int) -> RngSpec:
        """Transform this RNG specification to a vectorized version."""
        return RngSpec(
            self.rng_name,
            PsShortArrayType(
                PsVectorType(
                    cast(PsScalarType, self.short_array_type.base_type), num_lanes
                ),
                self.short_array_type.num_elements,
            ),
            self.num_ctrs,
            self.num_keys,
            PsVectorType(cast(PsIntegerType, self.int_arg_type), num_lanes),
        )


class PsRngEngineFunction(PsIrFunction):
    """IR function that represents the invocation of a random number generation engine.

    This is the IR representation of the symbolic random number generators
    implemented in `pystencils.sympyextensions.random`.
    Each symbolic RNG invocation is mapped onto an RNG engine function
    through `get_for_rng`.

    The characteristics of an RNG engine are defined by its `RngSpec`.
    Each engine is a function with the signature

    .. code-block::

        engine(ctr_0, ..., ctr_n, key_0, ..., key_m) -> ShortArray< value_type, K >

    which takes n+1 *counter* arguments, and m+1 *key* arguments.
    For scalar RNGs, all arguments have the same integer data type
    (the ``int_arg_type`` parameter of the `RngSpec`);
    for vector RNGs (with ``l`` vector lanes), the ``ctr`` arguments are ``l``-vectors of ``int_arg_type``,
    while the key arguments remain scalar.

    The engine function returns a k-array of random values of type ``value_type``
    as an instance of `PsShortArrayType`.
    The ``value_type`` may be a scalar (for scalar RNGs) or a SIMD vector type (for vector RNGs).

    RNG engine functions must be mapped to platform-specific implementations according to their `RngSpec`.
    This must happen either in `SelectFunctions` (for scalar RNGs) or `SelectIntrinsics` (for vector RNGs).

    Args:
        rng_spec: Specification defining the RNG's properties
    """

    __match_args__ = ("rng_spec",)

    @staticmethod
    def get_for_rng(state: RngState, ctr_type: PsIntegerType) -> PsRngEngineFunction:
        """Retrieve the function to be invoked for the given symbolic RNG."""
        match state:
            case Philox.PhiloxState():
                spec = RngSpec.philox(state.dtype, ctr_type)
            case _:
                raise ValueError(f"Unexpected RNG type: {type(state)}")

        return PsRngEngineFunction(spec)

    def __init__(self, rng_spec: RngSpec):
        self._rng_spec = rng_spec
        super().__init__(rng_spec.rng_name, rng_spec.num_ctrs + rng_spec.num_keys)

    @property
    def rng_spec(self) -> RngSpec:
        return self._rng_spec

    def __str__(self) -> str:
        return f"{self._rng_spec.rng_name}"

    def __repr__(self) -> str:
        return f"PsRngEngineFunction({repr(self._rng_spec)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsRngEngineFunction):
            return False

        return self._rng_spec == other._rng_spec

    def __hash__(self) -> int:
        return hash(self._rng_spec)


class CFunction(PsFunction):
    """A concrete C function.

    Instances of this class represent a C function by its name, parameter types, and return type.

    Args:
        name: Function name
        param_types: Types of the function parameters
        return_type: The function's return type
    """

    __match_args__ = ("name", "parameter_types", "return_type")

    @staticmethod
    def parse(obj) -> CFunction:
        """Parse the signature of a Python callable object to obtain a CFunction object.

        The callable must be fully annotated with type-like objects convertible by `create_type`.
        """
        import inspect
        from pystencils.types import create_type

        if not inspect.isfunction(obj):
            raise PsInternalCompilerError(f"Cannot parse object {obj} as a function")

        func_sig = inspect.signature(obj)
        func_name = obj.__name__
        arg_types = [
            create_type(param.annotation) for param in func_sig.parameters.values()
        ]
        ret_type = create_type(func_sig.return_annotation)

        return CFunction(func_name, arg_types, ret_type)

    def __init__(self, name: str, param_types: Sequence[PsType], return_type: PsType):
        super().__init__(name, len(param_types))

        self._param_types = tuple(param_types)
        self._return_type = return_type

    @property
    def parameter_types(self) -> tuple[PsType, ...]:
        return self._param_types

    @property
    def return_type(self) -> PsType:
        return self._return_type

    def __str__(self) -> str:
        param_types = ", ".join(str(t) for t in self._param_types)
        return f"{self._return_type} {self._name}({param_types})"

    def __repr__(self) -> str:
        return f"CFunction({self._name}, {self._param_types}, {self._return_type})"
