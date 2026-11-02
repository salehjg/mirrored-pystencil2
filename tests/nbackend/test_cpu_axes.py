from typing import cast
import pytest

import pystencils as ps
import numpy as np


from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.platforms import (
    GenericCpu,
    GenericVectorCpu,
    X86VectorCpu,
    X86VectorArch,
    NeonCpu,
    SveCpu,
    RvvCpu,
)
from pystencils.backend.ast.structural import PsBlock
from pystencils.backend.transformations import (
    AxisExpansion,
    MaterializeAxes,
    ReductionsToMemory,
    SelectFunctions,
    LowerToC,
)
from pystencils.backend.transformations.axis_expansion import AxisExpansionStrategy

from pystencils.codegen.driver import KernelFactory
from pystencils.jit.cpu import CpuJit


def _make_kernel(
    ctx: KernelCreationContext,
    target: ps.Target,
    body: PsBlock,
    ae_strategy: AxisExpansionStrategy,
    cpujit: CpuJit,
):
    factory = AstFactory(ctx)
    cube = factory.cube_from_ispace(ctx.get_full_iteration_space(), body)

    match target:
        case ps.Target.X86_SSE:
            platform = X86VectorCpu(ctx, X86VectorArch.SSE)
        case ps.Target.X86_AVX:
            platform = X86VectorCpu(ctx, X86VectorArch.AVX)
        case ps.Target.X86_AVX512 | ps.Target.X86_AVX512_FP16:
            platform = X86VectorCpu(ctx, X86VectorArch.AVX512)
        case ps.Target.ARM_NEON | ps.Target.ARM_NEON_FP16:
            platform = NeonCpu(ctx)
        case ps.Target.ARM_SVE:
            platform = SveCpu(ctx)
        case ps.Target.RISCV_RVV:
            platform = RvvCpu(ctx)
        case _:
            platform = GenericCpu(ctx)

    loops = ae_strategy(cube)

    materialize_axes = MaterializeAxes(ctx)
    ast = materialize_axes(loops)

    r_to_mem = ReductionsToMemory(ctx, ctx.reduction_data.values())
    ast = r_to_mem(ast)

    lower = LowerToC(ctx)
    ast = cast(PsBlock, lower(ast))

    if isinstance(platform, GenericVectorCpu):
        select_intrin = platform.get_intrinsic_selector()
        ast = cast(PsBlock, select_intrin(ast))

    select_functions = SelectFunctions(platform)
    ast = cast(PsBlock, select_functions(ast))

    kfactory = KernelFactory(ctx)
    kernel = kfactory.create_generic_kernel(
        platform,
        ast,
        "vector_kernel",
        target,
        cpujit,
    )

    return kernel


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("target", [ps.Target.GenericCPU])
def test_omp_with_reduction(rank: int, target, cpujit):
    f = ps.fields(f"f: [{rank}D]")
    w = ps.TypedSymbol("w", ps.DynamicType.NUMERIC_TYPE)

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.MaxReductionAssignment(w, f()))])

    ae = AxisExpansion(ctx)
    strategy = ae.create_strategy(
        [ae.parallel_loop()] + [ae.loop() for _ in range(rank - 1)]
    )
    kernel = _make_kernel(ctx, target, body, strategy, cpujit)
    kfunc = kernel.compile()

    rng = np.random.default_rng(seed=514)
    f_arr = rng.random((49,) * rank)
    w_arr = np.zeros((1,))
    expected = np.max(f_arr)

    kfunc(f=f_arr, w=w_arr)
    np.testing.assert_allclose(w_arr, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("target", ps.Target.available_vector_cpu_targets())
@pytest.mark.parametrize(
    "remainder_loop", [False, True], ids=["no_remainder", "with_remainder"]
)
def test_simd_with_reduction(
    rank: int, target: ps.Target, remainder_loop: bool, cpujit
):
    rng = np.random.default_rng(seed=514)
    L = 51 if remainder_loop else 48
    f_arr = rng.random((L,) * rank)  # x-size divisible by eight

    f = ps.Field.create_from_numpy_array("f", f_arr)
    w = ps.TypedSymbol("w", ps.DynamicType.NUMERIC_TYPE)

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.MaxReductionAssignment(w, f()))])

    ae = AxisExpansion(ctx)

    if target == ps.Target.ARM_SVE:
        lanes = 2
    else:
        lanes = target.default_vector_lanes(
            cast(ps.types.PsScalarType, ctx.default_dtype)
        )

    if remainder_loop:
        strategy = ae.create_strategy(
            [ae.loop() for _ in range(rank - 1)]
            + [
                ae.peel_for_divisibility(lanes),
                [ae.block_loop(lanes, assume_divisible=True), ae.simd(lanes)],
                [ae.loop()],
            ]
        )
    else:
        strategy = ae.create_strategy(
            [ae.loop() for _ in range(rank - 1)]
            + [ae.block_loop(lanes, assume_divisible=True), ae.simd(lanes)]
        )

    kernel = _make_kernel(ctx, target, body, strategy, cpujit)
    kfunc = kernel.compile()

    w_arr = np.zeros((1,))
    expected = np.max(f_arr)

    kfunc(f=f_arr, w=w_arr)
    np.testing.assert_allclose(w_arr, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("target", ps.Target.available_vector_cpu_targets())
def test_omp_simd_with_reduction(rank: int, target: ps.Target, cpujit):
    rng = np.random.default_rng(seed=514)
    f_arr = rng.random((48,) * rank)  # x-size divisible by eight

    f = ps.Field.create_from_numpy_array("f", f_arr)
    w = ps.TypedSymbol("w", ps.DynamicType.NUMERIC_TYPE)

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, f)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(ps.MaxReductionAssignment(w, f()))])

    ae = AxisExpansion(ctx)

    if target == ps.Target.ARM_SVE:
        lanes = 2
    else:
        lanes = target.default_vector_lanes(
            cast(ps.types.PsScalarType, ctx.default_dtype)
        )

    if rank == 1:
        strategy = ae.create_strategy(
            [
                ae.parallel_block_loop(lanes, assume_divisible=True),
                ae.simd(lanes),
            ]
        )
    else:
        strategy = ae.create_strategy(
            [ae.parallel_loop()]
            + [ae.loop() for _ in range(rank - 2)]
            + [ae.block_loop(lanes, assume_divisible=True), ae.simd(lanes)]
        )
    kernel = _make_kernel(ctx, target, body, strategy, cpujit)
    kfunc = kernel.compile()

    w_arr = np.zeros((1,))
    expected = np.max(f_arr)

    kfunc(f=f_arr, w=w_arr)
    np.testing.assert_allclose(w_arr, expected)
