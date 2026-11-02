from __future__ import annotations
from typing import Callable, overload, cast, Sequence
from dataclasses import dataclass

from ..kernelcreation import KernelCreationContext, AstFactory
from ..ast.axes import (
    PsAxesCube,
    PsIterationAxis,
    PsLoopAxis,
    PsAxisRange,
    PsSimdAxis,
    PsParallelLoopAxis,
    PsGpuIndexingAxis,
    PsGpuBlockAxis,
    PsGpuThreadAxis,
    PsGpuBlockXThreadAxis,
    GpuGridDimension,
)
from ..ast.structural import PsStructuralNode, PsBlock, PsDeclaration
from ..ast.expressions import PsExpression
from ..constants import PsConstant
from ..functions import (
    PsMathFunction,
    MathFunctions,
    PsGpuIndexingFunction,
    GpuGridScope,
)

from .eliminate_constants import TypifyAndFold
from .canonical_clone import CanonicalClone
from .rewrite import substitute_symbols
from ..exceptions import PsInternalCompilerError


@dataclass
class ExpansionFunc:
    name: str
    func: Callable[[PsAxesCube], PsStructuralNode]

    def __call__(self, cube: PsAxesCube) -> PsStructuralNode:
        return self.func(cube)


@dataclass
class StrategyNode:
    func: ExpansionFunc
    tail: list[StrategyNode]


class AxisExpansionStrategy:
    def __init__(self, strategy: tuple[StrategyNode, ...]):
        self._strategy = strategy

    def __call__(self, ast: PsBlock | PsAxesCube) -> PsBlock:
        if isinstance(ast, PsAxesCube):
            ast = PsBlock([ast])

        return self.visit(ast, list(self._strategy))

    @overload
    def visit(self, node: PsBlock, strategy: list[StrategyNode]) -> PsBlock: ...

    @overload
    def visit(
        self, node: PsStructuralNode, strategy: list[StrategyNode]
    ) -> PsStructuralNode: ...

    def visit(
        self, node: PsStructuralNode, strategy: list[StrategyNode]
    ) -> PsStructuralNode:
        match node:
            case PsAxesCube() if not strategy:
                raise PsInternalCompilerError(
                    "Cube cannot be expanded: expansion strategy is exhausted.\n"
                    f"    at:\n{node}"
                )
            case PsAxesCube():
                strategy_node = strategy[0]
                del strategy[0]
                replacement = strategy_node.func(node)
                substrategy = strategy_node.tail.copy()
                replacement = self.visit(replacement, substrategy)

                #   During handling of newly created cubes,
                #   the substrategy list must be fully exhausted
                if substrategy:
                    raise PsInternalCompilerError(
                        "Substrategy was not exhausted, but there are no more cubes left to expand.\n"
                        f"    Unmatched expansions: {', '.join(s.func.name for s in substrategy)}"
                    )

                return replacement

            case other:
                other.children = [
                    (self.visit(c, strategy) if isinstance(c, PsStructuralNode) else c)
                    for c in other.children
                ]
                return other


StrategyDescr = Sequence["ExpansionFunc | StrategyDescr"]


class AxisExpansion:

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        self._factory = AstFactory(ctx)
        self._type_fold = TypifyAndFold(ctx)
        self._canon_clone = CanonicalClone(ctx)

    def create_strategy(self, funcs: StrategyDescr) -> AxisExpansionStrategy:
        def _substrategy(sublist: StrategyDescr) -> StrategyNode:
            if not isinstance(sublist[0], ExpansionFunc):
                raise ValueError(
                    "First entry of a strategy must be an expansion function"
                )
            return StrategyNode(sublist[0], _recurse(sublist[1:]))

        def _recurse(substrategy: StrategyDescr) -> list[StrategyNode]:
            if not substrategy:
                return []
            if isinstance(substrategy[0], ExpansionFunc):
                return [StrategyNode(substrategy[0], _recurse(substrategy[1:]))]
            else:
                if not all(isinstance(s, list) for s in substrategy):
                    raise ValueError("Invalid strategy.")
                return [_substrategy(cast(StrategyDescr, s)) for s in substrategy]

        cnode = _substrategy(funcs)
        return AxisExpansionStrategy((cnode,))

    def loop(self, coordinate: int | None = None) -> ExpansionFunc:
        """Expand one dimension fully as a loop."""
        return self._loop_impl("loop", coordinate, PsLoopAxis)

    def parallel_loop(
        self,
        coordinate: int | None = None,
        *,
        num_threads: int | None = None,
        schedule: str | None = None,
        collapse: int | None = None,
    ) -> ExpansionFunc:
        """Expand one dimension fully as a parallel loop.

        Args:
            coordinate: Which dimension to expand. If `None`, the first dimension is used
            num_threads: OpenMP ``num_threads`` clause
            schedule: OpenMP ``schedule`` clause
            collapse: OpenMP ``collapse`` clause
        """

        def make_parallel_loop(
            xrange: PsAxisRange, body: PsBlock
        ) -> PsParallelLoopAxis:
            return PsParallelLoopAxis(
                xrange,
                body,
                num_threads=num_threads,
                schedule=schedule,
                collapse=collapse,
            )

        return self._loop_impl("parallel_loop", coordinate, make_parallel_loop)

    def _loop_impl(
        self,
        func_name: str,
        coordinate: int | None,
        loop_axis_ctor: Callable[[PsAxisRange, PsBlock], PsIterationAxis],
    ) -> ExpansionFunc:
        coordinate = 0 if coordinate is None else coordinate

        def make_loop(
            cube: PsAxesCube,
        ) -> PsStructuralNode:
            loop_range = cube.ranges[coordinate]

            remaining_ranges = cube.ranges.copy()
            del remaining_ranges[coordinate]

            if remaining_ranges:
                new_cube = PsAxesCube(remaining_ranges, cube.body)
                body = PsBlock([new_cube])
            else:
                body = cube.body

            loop_axis = loop_axis_ctor(loop_range, body)
            return loop_axis

        return ExpansionFunc(f"{func_name}({coordinate})", make_loop)

    def block_loop(
        self,
        block_size: int | PsExpression,
        coordinate: int | None = None,
        *,
        assume_divisible: bool = False,
    ) -> ExpansionFunc:
        """Introduce a block loop with given block size in the given dimension.

        Args:
            block_size: Block size of the block loop
            coordinate: Which dimension to block; if `None`, the first dimension is used
            assume_divisible: If `True`, assume that the iteration count is divisible
                by the block size and optimize accordingly.
        """

        return self._block_loop_impl(
            "block_loop", block_size, coordinate, assume_divisible, PsLoopAxis
        )

    def parallel_block_loop(
        self,
        block_size: int | PsExpression,
        coordinate: int | None = None,
        *,
        assume_divisible: bool = False,
        num_threads: int | None = None,
        schedule: str | None = None,
        collapse: int | None = None,
    ) -> ExpansionFunc:
        """Introduce a parallel block loop with given block size in the given dimension.

        Args:
            block_size: Block size of the block loop
            coordinate: Which dimension to block; if `None`, the first dimension is used
            assume_divisible: If `True`, assume that the iteration count is divisible
                by the block size and optimize accordingly.
            num_threads: OpenMP ``num_threads`` clause
            schedule: OpenMP ``schedule`` clause
            collapse: OpenMP ``collapse`` clause
        """

        def make_parallel_loop(
            xrange: PsAxisRange, body: PsBlock
        ) -> PsParallelLoopAxis:
            return PsParallelLoopAxis(
                xrange,
                body,
                num_threads=num_threads,
                schedule=schedule,
                collapse=collapse,
            )

        return self._block_loop_impl(
            "parallel_block_loop",
            block_size,
            coordinate,
            assume_divisible,
            make_parallel_loop,
        )

    def _block_loop_impl(
        self,
        func_name: str,
        block_size: int | PsExpression,
        coordinate: int | None,
        assume_divisible: bool,
        loop_axis_ctor: Callable[[PsAxisRange, PsBlock], PsIterationAxis],
    ) -> ExpansionFunc:
        coordinate = 0 if coordinate is None else coordinate

        def make_block_loop(
            cube: PsAxesCube,
        ) -> PsStructuralNode:
            my_range = cube.ranges[coordinate]

            block_size_expr: PsExpression
            if not isinstance(block_size, PsExpression):
                block_size_expr = self._factory.parse_index(block_size)
            else:
                block_size_expr = block_size

            blocked_ctr_symb = self._ctx.duplicate_symbol(my_range.counter.symbol)
            blocked_step = self._type_fold(my_range.step.clone() * block_size_expr)
            blocked_range = PsAxisRange(
                PsExpression.make(blocked_ctr_symb),
                my_range.start,
                my_range.stop,
                blocked_step,
            )

            my_range.start = PsExpression.make(blocked_ctr_symb)
            block_stop = (
                PsExpression.make(blocked_ctr_symb)
                + my_range.step.clone() * block_size_expr.clone()
            )

            if assume_divisible:
                my_range.stop = self._type_fold(block_stop)
            else:
                min_ = PsMathFunction(MathFunctions.Min)
                my_range.stop = self._type_fold(min_(my_range.stop.clone(), block_stop))

            blocked_loop = loop_axis_ctor(blocked_range, PsBlock([cube]))
            return blocked_loop

        return ExpansionFunc(
            f"{func_name}({block_size}, {coordinate})", make_block_loop
        )

    def peel_for_divisibility(
        self, k: int, coordinate: int | None = None
    ) -> ExpansionFunc:
        """Peel off the minimal number of iterations from the back of one dimension
        such that the number of iterations in the bulk part is divisible by ``k``."""
        coordinate = 0 if coordinate is None else coordinate

        def do_peel(
            cube: PsAxesCube,
        ) -> PsStructuralNode:
            my_range = cube.ranges[coordinate]
            kc = PsConstant(k)

            #   Compute starting index of remainder axis and declare it as a symbol
            icount = self._iteration_count(my_range)
            icount_bulk = (icount / PsExpression.make(kc)) * PsExpression.make(kc)
            rem_start = my_range.start.clone() + icount_bulk * my_range.step.clone()
            rem_start_symb = self._ctx.get_new_symbol(
                my_range.counter.symbol.name + "__rem_start",
                my_range.counter.symbol.dtype,
            )
            rem_start_decl = self._type_fold(
                PsDeclaration(PsExpression.make(rem_start_symb), rem_start)
            )

            #   Construct remainer cube and its body
            rem_ctr_symb = self._ctx.duplicate_symbol(my_range.counter.symbol)
            rem_ranges = cube.ranges.copy()
            rem_ranges[coordinate] = PsAxisRange(
                PsExpression.make(rem_ctr_symb),
                PsExpression.make(rem_start_symb),
                my_range.stop,
                my_range.step.clone(),
            )
            rem_body = self._canon_clone(cube.body)
            substitute_symbols(
                rem_body, {my_range.counter.symbol: PsExpression.make(rem_ctr_symb)}
            )
            rem_cube = PsAxesCube(rem_ranges, rem_body)

            #   Construct bulk cube and its body by modifying the original cube in-place
            my_range.stop = PsExpression.make(rem_start_symb)
            block = PsBlock([rem_start_decl, cube, rem_cube])

            return block

        return ExpansionFunc(f"peel_for_divisibility({k}, {coordinate})", do_peel)

    def simd(self, lanes: int) -> ExpansionFunc:
        """Apply to a cube with only one coordinate to convert it to a vectorized block"""

        def apply_simd(cube: PsAxesCube) -> PsStructuralNode:
            if not len(cube.ranges) == 1:
                raise PsInternalCompilerError(
                    "Cannot apply `simd` expansion to a cube with more than one axis"
                )

            return PsSimdAxis(lanes, cube.ranges[0], cube.body)

        return ExpansionFunc(f"simd({lanes})", apply_simd)

    def _iteration_count(self, rang: PsAxisRange) -> PsExpression:
        extent = rang.stop - rang.start
        one = PsConstant(1)
        return self._type_fold(
            (extent + (rang.step - PsExpression.make(one))) / rang.step
        )

    def gpu_block(
        self, dim: str | GpuGridDimension, coordinate: int | None = None
    ) -> ExpansionFunc:
        """Map one cube coordinate onto the GPU block index in the given grid dimension.

        Args:
            dim: GPU grid coordinate, ``"x"``, ``"y"`` or ``"z"``.
        """
        return self._gpu_indexing_impl(
            f"gpu_block({dim}, {coordinate})", dim, coordinate, PsGpuBlockAxis
        )

    def gpu_thread(
        self, dim: str | GpuGridDimension, coordinate: int | None = None
    ) -> ExpansionFunc:
        """Map one cube coordinate onto the GPU thread index in the given grid dimension.

        Args:
            dim: GPU grid coordinate, ``"x"``, ``"y"`` or ``"z"``.
        """
        return self._gpu_indexing_impl(
            f"gpu_thread({dim}, {coordinate})", dim, coordinate, PsGpuThreadAxis
        )

    def gpu_block_x_thread(
        self, dim: str | GpuGridDimension, coordinate: int | None = None
    ) -> ExpansionFunc:
        """Map one cube coordinate onto the product of GPU block and thread index
        in the given grid dimension.

        Args:
            dim: GPU grid coordinate, ``"x"``, ``"y"`` or ``"z"``.
        """
        return self._gpu_indexing_impl(
            f"gpu_block_x_thread({dim}, {coordinate})",
            dim,
            coordinate,
            PsGpuBlockXThreadAxis,
        )

    def _gpu_indexing_impl(
        self,
        func_name: str,
        dim: str | GpuGridDimension,
        coordinate: int | None,
        axis_type: type[PsGpuIndexingAxis],
    ):
        gpu_dim = (
            dim if isinstance(dim, GpuGridDimension) else GpuGridDimension[dim.upper()]
        )
        coordinate = 0 if coordinate is None else coordinate

        def make_block_axis(cube: PsAxesCube) -> PsStructuralNode:
            axis_range = cube.ranges[coordinate]

            remaining_ranges = cube.ranges.copy()
            del remaining_ranges[coordinate]

            if remaining_ranges:
                new_cube = PsAxesCube(remaining_ranges, cube.body)
                body = PsBlock([new_cube])
            else:
                body = cube.body

            loop_axis = axis_type(gpu_dim, axis_range, body)
            return loop_axis

        return ExpansionFunc(func_name, make_block_axis)

    def gridstrided_loop(
        self,
        dim: str | GpuGridDimension,
        coordinate: int | None = None,
    ):
        """Introduce a grid-strided loop in the given dimension.

        Args:
            dim: GPU grid coordinate, ``"x"``, ``"y"`` or ``"z"``.
            coordinate: Which dimension to block; if `None`, the first dimension is used
        """

        return self._gridstrided_loop_impl("gridstrided_loop", dim, coordinate)

    def _gridstrided_loop_impl(
        self,
        func_name: str,
        dim: str | GpuGridDimension,
        coordinate: int | None,
    ) -> ExpansionFunc:
        gpu_dim = (
            dim if isinstance(dim, GpuGridDimension) else GpuGridDimension[dim.upper()]
        )
        coordinate = 0 if coordinate is None else coordinate

        def make_gridstrided_loop(
            cube: PsAxesCube,
        ) -> PsStructuralNode:
            my_range = cube.ranges[coordinate]

            gridstride = self._type_fold(
                PsGpuIndexingFunction(GpuGridScope.gridDim, gpu_dim)()
                * PsGpuIndexingFunction(GpuGridScope.blockDim, gpu_dim)()
            )

            # create new axis range with duplicated ctr symbol, adapt stride to grid stride
            blocked_ctr_symb = self._ctx.duplicate_symbol(my_range.counter.symbol)
            blocked_step = self._type_fold(my_range.step.clone() * gridstride)
            blocked_range = PsAxisRange(
                PsExpression.make(blocked_ctr_symb),
                my_range.start,
                my_range.stop,
                blocked_step,
            )

            # offset axr start with ctr of grid-strided loop, stop & step remain the same
            my_range.start = PsExpression.make(blocked_ctr_symb)

            return PsLoopAxis(blocked_range, PsBlock([cube]))

        return ExpansionFunc(f"{func_name}({coordinate})", make_gridstrided_loop)
