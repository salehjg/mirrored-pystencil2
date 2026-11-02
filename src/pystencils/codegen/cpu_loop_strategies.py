from __future__ import annotations
from typing import cast
import sympy as sp

from .target import Target
from .config import CpuOptions
from .errors import CodegenError

from ..types import PsScalarType

from ..backend.kernelcreation import KernelCreationContext, AstFactory
from ..backend.kernelcreation import (
    IterationSpace,
    SparseIterationSpace,
    FullIterationSpace,
)
from ..backend.ast.structural import PsBlock
from ..backend.ast.axes import PsAxesCube
from ..backend.constants import PsConstant
from ..backend.transformations import (
    CanonicalizeSymbols,
    AxisExpansion,
    HoistIterationInvariantDeclarations,
)


class DefaultCpuLoopStrategies:
    def __init__(
        self, ctx: KernelCreationContext, target: Target, cpu_options: CpuOptions
    ):
        self._ctx = ctx
        self._target = target
        self._cpu_options = cpu_options

        self._check_cpu_features()

        self._factory = AstFactory(ctx)

    def _check_cpu_features(self) -> None:
        if self._cpu_options.use_cacheline_zeroing:
            raise NotImplementedError("CL-zeroing not implemented yet")

    def create_axes(self, body: PsBlock, ispace: IterationSpace) -> PsBlock:
        kernel_ast: PsBlock
        match ispace:
            case FullIterationSpace():
                kernel_ast = self._dense_ispace_axes(body, ispace)
            case SparseIterationSpace():
                kernel_ast = self._sparse_ispace_axes(body, ispace)
            case _:
                assert False, "Invalid ispace type"

        return kernel_ast

    def _dense_ispace_axes(self, body: PsBlock, ispace: FullIterationSpace) -> PsBlock:
        vec_options = self._cpu_options.vectorize
        enable_vec = vec_options.get_option("enable")

        vec_lanes: int | None = vec_options.get_option("lanes")

        if enable_vec:
            self._apply_vectorization_assumptions()

            if vec_lanes is None:
                vec_lanes = self._target.default_vector_lanes(
                    cast(PsScalarType, self._ctx.default_dtype)
                )

        omp_kwargs = self._get_parallel_loop_kwargs()

        cube = self._factory.cube_from_ispace(ispace, body)

        canonicalize = CanonicalizeSymbols(self._ctx, True)
        cube = cast(PsAxesCube, canonicalize(cube))

        ae = AxisExpansion(self._ctx)
        rank = ispace.rank

        block_loops, parallelize = self._block_loop_expansions(ispace, ae)

        if rank == 1 and enable_vec:
            assert vec_lanes is not None
            strategy = ae.create_strategy(
                block_loops
                + [
                    ae.peel_for_divisibility(vec_lanes),
                    [
                        (
                            ae.parallel_block_loop(
                                vec_lanes, assume_divisible=True, **omp_kwargs
                            )
                            if parallelize
                            else ae.block_loop(vec_lanes, assume_divisible=True)
                        ),
                        ae.simd(vec_lanes),
                    ],
                    [ae.loop()],
                ]
            )
        elif enable_vec:
            assert vec_lanes is not None
            strategy = ae.create_strategy(
                block_loops
                + [ae.parallel_loop(**omp_kwargs) if parallelize else ae.loop()]
                + [ae.loop() for _ in range(rank - 2)]
                + [
                    ae.peel_for_divisibility(vec_lanes),
                    [
                        ae.block_loop(vec_lanes, assume_divisible=True),
                        ae.simd(vec_lanes),
                    ],
                    [ae.loop()],
                ]
            )
        else:
            strategy = ae.create_strategy(
                block_loops
                + [ae.parallel_loop(**omp_kwargs) if parallelize else ae.loop()]
                + [ae.loop() for _ in range(rank - 1)]
            )

        kernel_ast: PsBlock = strategy(cube)

        hoist_invariants = HoistIterationInvariantDeclarations(self._ctx)
        kernel_ast = cast(PsBlock, hoist_invariants(kernel_ast))

        return kernel_ast

    def _apply_vectorization_assumptions(self) -> None:
        vec_options = self._cpu_options.vectorize

        if not self._target.is_vector_cpu():
            raise CodegenError(
                "Cannot vectorize: selected target is no vector CPU target."
            )

        ispace = self._ctx.get_full_iteration_space()

        inner_loop_coord = ispace.loop_order[-1]

        #   Apply stride (TODO: and alignment) assumptions
        assume_unit_stride: bool = vec_options.get_option("assume_inner_stride_one")

        if assume_unit_stride:
            for field in self._ctx.fields:
                buf = self._ctx.get_buffer(field)
                inner_stride = buf.strides[inner_loop_coord]
                if isinstance(inner_stride, PsConstant):
                    if inner_stride.value != 1:
                        raise CodegenError(
                            f"Unable to apply assumption 'assume_inner_stride_one': "
                            f"Field {field} has fixed stride {inner_stride} "
                            f"set in the inner coordinate {inner_loop_coord}."
                        )
                else:
                    buf.strides[inner_loop_coord] = PsConstant(1, buf.index_type)
                    #   TODO: Communicate assumption to runtime system via a precondition

    def _block_loop_expansions(
        self, ispace: FullIterationSpace, ae: AxisExpansion
    ) -> tuple[list, bool]:
        """Generate expansion rules for CPU block loops.

        Return:
            (Possibly empty) list of expansion rules,
            and a boolean indicating whether remaining inner loops should be parallelized
        """

        parallelize: bool = self._cpu_options.openmp.get_option("enable")
        blocking_spec: tuple[int | sp.Expr | None, ...] = self._cpu_options.get_option(
            "loop_blocking"
        )

        if blocking_spec is None:
            return [], parallelize

        if len(blocking_spec) != ispace.rank:
            raise CodegenError(
                f"Invalid value for `cpu.loop_blocking`: `{blocking_spec}`. "
                "Must specify block size, or `None`, for each dimension."
            )

        expansions = []
        omp_kwargs = self._get_parallel_loop_kwargs()

        for axis_idx, dim_idx in enumerate(ispace.loop_order):
            if (bs := blocking_spec[dim_idx]) is not None:
                bs_expr = self._factory.parse_index(bs)
                expansions.append(
                    ae.parallel_block_loop(bs_expr, axis_idx, **omp_kwargs)
                    if parallelize
                    else ae.block_loop(bs_expr, axis_idx)
                )
                parallelize = False

        return expansions, parallelize

    def _sparse_ispace_axes(
        self, body: PsBlock, ispace: SparseIterationSpace
    ) -> PsBlock:
        omp_options = self._cpu_options.openmp
        enable_omp: bool = omp_options.get_option("enable")

        body.statements = (
            ispace.get_spatial_counter_declarations(self._ctx) + body.statements
        )
        cube = self._factory.cube_from_ispace(ispace, body)

        canonicalize = CanonicalizeSymbols(self._ctx, True)
        cube = cast(PsAxesCube, canonicalize(cube))

        ae = AxisExpansion(self._ctx)
        strategy = ae.create_strategy(
            [
                (
                    ae.parallel_loop(**self._get_parallel_loop_kwargs())
                    if enable_omp
                    else ae.loop()
                )
            ]
        )

        kernel_ast: PsBlock = strategy(cube)

        hoist_invariants = HoistIterationInvariantDeclarations(self._ctx)
        kernel_ast = cast(PsBlock, hoist_invariants(kernel_ast))

        return kernel_ast

    def _get_parallel_loop_kwargs(self) -> dict:
        omp_options = self._cpu_options.openmp
        enable_omp: bool = omp_options.get_option("enable")
        if enable_omp:
            return dict(
                num_threads=omp_options.get_option("num_threads"),
                schedule=omp_options.get_option("schedule"),
                collapse=omp_options.get_option("collapse"),
            )
        else:
            return dict()
