from typing import cast

import pytest
import sympy as sp
import numpy as np
from dataclasses import dataclass
from itertools import chain
from functools import partial
from typing import Callable
import re

from pystencils import DEFAULTS
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.platforms import (
    GenericVectorCpu,
    X86VectorArch,
    X86VectorCpu,
    NeonCpu,
    SveCpu,
    RvvCpu,
)
from pystencils.backend.ast.structural import PsBlock
from pystencils.backend.transformations import AxisExpansion, MaterializeAxes, LowerToC
from pystencils.backend.constants import PsConstant
from pystencils.codegen.driver import KernelFactory
from pystencils.jit import CpuJit
from pystencils.jit.cpu import CompilerInfo
from pystencils import Target, fields, Assignment, Field, FieldType
from pystencils.field import create_numpy_array_with_layout
from pystencils.types.quick import SInt, Fp


@dataclass
class VectorTestSetup:
    target: Target
    platform_factory: Callable[[KernelCreationContext], GenericVectorCpu]
    lanes: int
    type_width: int

    @property
    def floating_type(self):
        return Fp(self.type_width)

    @property
    def integer_type(self):
        return SInt(self.type_width)

    @property
    def name(self) -> str:
        return f"({self.target.name} | {self.type_width}bit x {self.lanes})"


def get_setups(target: Target) -> list[VectorTestSetup]:
    match target:
        case Target.X86_SSE:
            sse_platform = partial(X86VectorCpu, vector_arch=X86VectorArch.SSE)
            return [
                VectorTestSetup(target, sse_platform, 4, 32),
                VectorTestSetup(target, sse_platform, 2, 64),
            ]

        case Target.X86_AVX:
            avx_platform = partial(X86VectorCpu, vector_arch=X86VectorArch.AVX)
            return [
                VectorTestSetup(target, avx_platform, 4, 32),
                VectorTestSetup(target, avx_platform, 8, 32),
                VectorTestSetup(target, avx_platform, 2, 64),
                VectorTestSetup(target, avx_platform, 4, 64),
            ]

        case Target.X86_AVX512:
            avx512_platform = partial(X86VectorCpu, vector_arch=X86VectorArch.AVX512)
            return [
                VectorTestSetup(target, avx512_platform, 4, 32),
                VectorTestSetup(target, avx512_platform, 8, 32),
                VectorTestSetup(target, avx512_platform, 16, 32),
                VectorTestSetup(target, avx512_platform, 2, 64),
                VectorTestSetup(target, avx512_platform, 4, 64),
                VectorTestSetup(target, avx512_platform, 8, 64),
            ]

        case Target.X86_AVX512_FP16:
            avx512_platform = partial(
                X86VectorCpu, vector_arch=X86VectorArch.AVX512_FP16
            )
            return [
                VectorTestSetup(target, avx512_platform, 8, 16),
                VectorTestSetup(target, avx512_platform, 16, 16),
                VectorTestSetup(target, avx512_platform, 32, 16),
            ]

        case Target.ARM_NEON | Target.ARM_NEON_FP16:
            enable_fp16 = Target._FP16 in target
            neon_platform = partial(NeonCpu, enable_fp16=enable_fp16)
            return (
                [
                    VectorTestSetup(target, neon_platform, 4, 16),
                    VectorTestSetup(target, neon_platform, 8, 16),
                ]
                if enable_fp16
                else []
            ) + [
                VectorTestSetup(target, neon_platform, 2, 32),
                VectorTestSetup(target, neon_platform, 4, 32),
                VectorTestSetup(target, neon_platform, 2, 64),
            ]
        case Target.ARM_SVE:
            return [
                VectorTestSetup(target, SveCpu, 8, 16),
                VectorTestSetup(target, SveCpu, 4, 32),
                VectorTestSetup(target, SveCpu, 2, 64),
            ]
        case Target.RISCV_RVV:
            #   Board profile (SpacemiT K1, VLEN=256): m1 and m2 configs for f32/f64.
            rvv256 = partial(RvvCpu, vlen=256)
            return [
                VectorTestSetup(target, rvv256, 4, 64),   # f64/i64 -> m1
                VectorTestSetup(target, rvv256, 8, 32),   # f32/i32 -> m1
                VectorTestSetup(target, rvv256, 8, 64),   # f64/i64 -> m2
                VectorTestSetup(target, rvv256, 16, 32),  # f32/i32 -> m2
            ]
        case _:
            return []


TEST_SETUPS: list[VectorTestSetup] = list(
    chain.from_iterable(get_setups(t) for t in Target.available_vector_cpu_targets())
)

TEST_IDS = [t.name for t in TEST_SETUPS]


@pytest.fixture(params=TEST_SETUPS, ids=TEST_IDS)
def vectorization_setup(request) -> VectorTestSetup:
    return request.param


def create_vector_kernel(
    assignments: list[Assignment],
    field: Field,
    setup: VectorTestSetup,
    ghost_layers: int = 0,
    assume_unit_stride: bool = True,
):
    ctx = KernelCreationContext(
        default_dtype=setup.floating_type, index_dtype=setup.integer_type
    )
    platform = setup.platform_factory(ctx)

    factory = AstFactory(ctx)

    ispace = FullIterationSpace.create_with_ghost_layers(ctx, ghost_layers, field)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(asm) for asm in assignments])

    if assume_unit_stride:
        for field in ctx.fields:
            #   Set inner strides to one to ensure packed memory access
            buf = ctx.get_buffer(field)
            buf.strides[0] = PsConstant(1, ctx.index_dtype)

    cube = factory.cube_from_ispace(ispace, body)

    lanes = setup.lanes

    ae = AxisExpansion(ctx)
    strategy = ae.create_strategy(
        [ae.loop() for _ in range(ispace.rank - 1)]
        + [
            ae.peel_for_divisibility(lanes),
            [ae.block_loop(lanes, assume_divisible=True), ae.simd(lanes)],
            [ae.loop()],
        ]
    )

    ast: PsBlock = strategy(cube)

    materialize_axes = MaterializeAxes(ctx)
    ast = materialize_axes(ast)

    select_intrin = platform.get_intrinsic_selector()
    ast = cast(PsBlock, select_intrin(ast))

    lower = LowerToC(ctx)
    ast = cast(PsBlock, lower(ast))

    cinfo = CompilerInfo.get_default(target=setup.target)

    kfactory = KernelFactory(ctx)
    kernel = kfactory.create_generic_kernel(
        platform,
        ast,
        "vector_kernel",
        Target.CPU,
        CpuJit(cinfo),
    )

    return kernel


@pytest.mark.parametrize("ghost_layers", [0, 2], ids=["0gls", "2gls"])
def test_update_kernel(vectorization_setup: VectorTestSetup, ghost_layers: int):
    setup = vectorization_setup
    src, dst = fields(f"src(2), dst(4): {setup.floating_type}[2D]", layout="fzyx")

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

    kernel = create_vector_kernel(update, src, setup, ghost_layers).compile()

    shape = (23, 17)

    rgen = np.random.default_rng(seed=1648)
    src_arr = create_numpy_array_with_layout(
        shape + (2,), layout=(2, 1, 0), dtype=setup.floating_type.numpy_dtype
    )

    np_ftype = setup.floating_type.numpy_dtype
    if np_ftype == np.float16:
        rnumbers = rgen.random(shape + (2,), dtype=np.float32)
        src_arr[:] = rnumbers.astype(np_ftype)
    else:
        rgen.random(dtype=np_ftype, out=src_arr)

    dst_arr = create_numpy_array_with_layout(
        shape + (4,), layout=(2, 1, 0), dtype=setup.floating_type.numpy_dtype
    )
    dst_arr[:] = 0.0

    check_arr = np.zeros_like(dst_arr)
    check_arr[:, :, 0] = src_arr[:, :, 0] + src_arr[:, :, 1]
    check_arr[:, :, 1] = src_arr[:, :, 0] - src_arr[:, :, 1]
    check_arr[:, :, 2] = src_arr[:, :, 0] * src_arr[:, :, 1]
    check_arr[:, :, 3] = src_arr[:, :, 0] / src_arr[:, :, 1]

    kernel(src=src_arr, dst=dst_arr)

    resolution = np.finfo(setup.floating_type.numpy_dtype).resolution
    gls = ghost_layers

    np.testing.assert_allclose(
        dst_arr[gls:-gls, gls:-gls, :],
        check_arr[gls:-gls, gls:-gls, :],
        rtol=resolution,
    )

    if gls != 0:
        for i in range(gls):
            np.testing.assert_equal(dst_arr[i, :, :], 0.0)
            np.testing.assert_equal(dst_arr[-i, :, :], 0.0)
            np.testing.assert_equal(dst_arr[:, i, :], 0.0)
            np.testing.assert_equal(dst_arr[:, -i, :], 0.0)


def test_trailing_iterations(vectorization_setup: VectorTestSetup):
    setup = vectorization_setup
    f = fields(f"f(1): {setup.floating_type}[1D]", layout="fzyx")

    update = [Assignment(f(0), 2 * f(0))]

    kernel = create_vector_kernel(update, f, setup).compile()

    for trailing_iters in range(setup.lanes):
        shape = (setup.lanes * 12 + trailing_iters, 1)
        f_arr = create_numpy_array_with_layout(
            shape, layout=(1, 0), dtype=setup.floating_type.numpy_dtype
        )

        f_arr[:] = 1.0

        kernel(f=f_arr)

        np.testing.assert_equal(f_arr, 2.0)


def test_only_trailing_iterations(vectorization_setup: VectorTestSetup):
    setup = vectorization_setup
    f = fields(f"f(1): {setup.floating_type}[1D]", layout="fzyx")

    update = [Assignment(f(0), 2 * f(0))]

    kernel = create_vector_kernel(update, f, setup).compile()

    for trailing_iters in range(1, setup.lanes):
        shape = (trailing_iters, 1)
        f_arr = create_numpy_array_with_layout(
            shape, layout=(1, 0), dtype=setup.floating_type.numpy_dtype
        )

        f_arr[:] = 1.0

        kernel(f=f_arr)

        np.testing.assert_equal(f_arr, 2.0)


def test_set(vectorization_setup: VectorTestSetup):
    setup = vectorization_setup
    f = fields(f"f(1): {setup.integer_type}[1D]", layout="fzyx")

    update = [Assignment(f(0), DEFAULTS.spatial_counters[0])]

    kernel = create_vector_kernel(update, f, setup).compile()

    shape = (23, 1)
    f_arr = create_numpy_array_with_layout(
        shape, layout=(1, 0), dtype=setup.integer_type.numpy_dtype
    )

    f_arr[:] = 42

    kernel(f=f_arr)

    reference = np.array(range(shape[0])).reshape(shape)
    np.testing.assert_equal(f_arr, reference)


TEST_SETUPS_FOR_STRIDED_LOAD = [
    s
    for s in TEST_SETUPS
    if s.target not in (Target.X86_SSE, Target.ARM_NEON, Target.ARM_NEON_FP16)
]

TEST_SETUPS_FOR_STRIDED_LOAD = [
    s
    for s in TEST_SETUPS_FOR_STRIDED_LOAD
    if not (s.target == Target.ARM_SVE and s.type_width == 16)
]


@pytest.mark.parametrize(
    "vectorization_setup",
    TEST_SETUPS_FOR_STRIDED_LOAD,
    ids=[t.name for t in TEST_SETUPS_FOR_STRIDED_LOAD],
)
@pytest.mark.parametrize("int_or_float", ["int", "float"])
def test_strided_load(vectorization_setup: VectorTestSetup, int_or_float):
    setup = vectorization_setup

    match int_or_float:
        case "int":
            dtype = setup.integer_type
        case "float":
            dtype = setup.floating_type
        case _:
            assert False

    f = fields(f"f: {dtype}[2D]", layout="fzyx")
    g = fields(f"g: {dtype}[2D]", layout="fzyx", field_type=FieldType.CUSTOM)

    i, j = DEFAULTS.spatial_counters[:2]
    update = [
        Assignment(
            f(),
            g.absolute_access((3 * i + 0, j), ())
            + g.absolute_access((3 * i + 1, j), ())
            + g.absolute_access((3 * i + 2, j), ()),
        )
    ]

    kernel = create_vector_kernel(update, f, setup).compile()

    f_shape = (21, 17)
    f_arr = create_numpy_array_with_layout(
        f_shape, layout=(1, 0), dtype=dtype.numpy_dtype
    )
    g_shape = (3 * f_shape[0], f_shape[1])
    g_arr = create_numpy_array_with_layout(
        g_shape, layout=(1, 0), dtype=dtype.numpy_dtype
    )

    g_arr[:] = np.arange(np.prod(g_shape), dtype=dtype.numpy_dtype).reshape(g_shape)

    kernel(f=f_arr, g=g_arr)

    f_reference = g_arr[::3, :] + g_arr[1::3, :] + g_arr[2::3, :]
    np.testing.assert_allclose(f_arr, f_reference)


TEST_SETUPS_FOR_STRIDED_STORE = get_setups(Target.X86_AVX512)

if Target.ARM_SVE in Target.available_vector_cpu_targets():
    TEST_SETUPS_FOR_STRIDED_STORE += [
        VectorTestSetup(Target.ARM_SVE, SveCpu, 4, 32),
        VectorTestSetup(Target.ARM_SVE, SveCpu, 2, 64),
    ]

if Target.RISCV_RVV in Target.available_vector_cpu_targets():
    TEST_SETUPS_FOR_STRIDED_STORE += get_setups(Target.RISCV_RVV)


@pytest.mark.parametrize(
    "vectorization_setup",
    TEST_SETUPS_FOR_STRIDED_STORE,
    ids=[t.name for t in TEST_SETUPS_FOR_STRIDED_STORE],
)
@pytest.mark.parametrize("int_or_float", ["int", "float"])
def test_strided_store(vectorization_setup: VectorTestSetup, int_or_float):
    setup = vectorization_setup

    match int_or_float:
        case "int":
            dtype = setup.integer_type
        case "float":
            dtype = setup.floating_type
        case _:
            assert False

    f = fields(f"f: {dtype}[2D]", layout="fzyx")
    g = fields(f"g: {dtype}[2D]", layout="fzyx", field_type=FieldType.CUSTOM)

    i, j = DEFAULTS.spatial_counters[:2]
    update = [
        Assignment(
            g.absolute_access((3 * i + 0, j), ()),
            1 + f(),
        ),
        Assignment(
            g.absolute_access((3 * i + 1, j), ()),
            2 + f(),
        ),
        Assignment(
            g.absolute_access((3 * i + 2, j), ()),
            3 + f(),
        ),
    ]

    kernel = create_vector_kernel(update, f, setup)

    if setup.target == Target.X86_AVX512:
        bits = setup.type_width * setup.lanes
        prefix = "_mm" if bits == 128 else f"_mm{bits}"

        if int_or_float == "int":
            suffix = f"epi{setup.type_width}"
        else:
            match setup.type_width:
                case 32:
                    suffix = "ps"
                case 64:
                    suffix = "pd"
                case _:
                    assert False, "unexpected width"

        scatter_pattern = (
            f"{prefix}_i{setup.type_width}scatter_{suffix}"
            + r"\(.*,.*,.*,\s*"
            + str(setup.type_width // 8)
            + r"\);"
        )

        assert len(re.findall(scatter_pattern, kernel.get_c_code())) == 3

    if setup.target in Target.available_vector_cpu_targets():
        #   We don't have AVX512 CPUs on the CI runners
        kernel = kernel.compile()

        f_shape = (21, 17)
        f_arr = create_numpy_array_with_layout(
            f_shape, layout=(1, 0), dtype=dtype.numpy_dtype
        )
        g_shape = (3 * f_shape[0], f_shape[1])
        g_arr = create_numpy_array_with_layout(
            g_shape, layout=(1, 0), dtype=dtype.numpy_dtype
        )

        f_arr[:] = np.arange(np.prod(f_shape), dtype=dtype.numpy_dtype).reshape(f_shape)

        kernel(f=f_arr, g=g_arr)

        g_reference = np.zeros_like(g_arr)
        g_reference[::3, :] = 1.0 + f_arr
        g_reference[1::3, :] = 2.0 + f_arr
        g_reference[2::3, :] = 3.0 + f_arr

        np.testing.assert_allclose(g_arr, g_reference)
