import pytest
import sympy as sp
import numpy as np

from pystencils import (
    fields,
    Field,
    AssignmentCollection,
    Target,
    CreateKernelConfig,
    create_type,
    TypedSymbol,
    DynamicType,
)
from pystencils.assignment import assignment_from_stencil
from pystencils.codegen.properties import FieldBasePtr
from pystencils.types import PsPointerType

from pystencils import create_kernel, Kernel
from pystencils.backend.emission import emit_code
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import PsLoop


def inspect_kernel(kernel: Kernel, gen_config: CreateKernelConfig):
    code = emit_code(kernel)

    px = (
        "pd"
        if gen_config.get_option("default_dtype") == create_type("float64")
        else "ps"
    )

    match gen_config.target:
        case Target.X86_SSE:
            assert f"_mm_loadu_{px}" in code
            assert f"_mm_storeu_{px}" in code
        case Target.X86_AVX:
            assert f"_mm256_loadu_{px}" in code
            assert f"_mm256_storeu_{px}" in code
        case Target.X86_AVX512:
            assert f"_mm512_loadu_{px}" in code
            assert f"_mm512_storeu_{px}" in code
        case Target.RISCV_RVV:
            bits = 64 if px == "pd" else 32
            assert f"__riscv_vle{bits}_v_f{bits}m1" in code
            assert f"__riscv_vse{bits}_v_f{bits}m1" in code


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filter_kernel(gen_config, xp, dtype):
    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    src, dst = fields("src, dst: [2D]")
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    cfg = gen_config.copy()
    cfg.default_dtype = dtype

    if cfg.get_target().is_cpu():
        cfg.cpu.openmp.enable = True

    ker = create_kernel(asms, cfg)

    for param in ker.parameters:
        if pprop := param.get_properties(FieldBasePtr):
            field: Field = pprop.pop().field
            assert isinstance(param.dtype, PsPointerType)
            if field == src:
                assert param.dtype.base_type.const
            else:
                assert not param.dtype.base_type.const

    inspect_kernel(ker, cfg)

    kfunc = ker.compile()

    src_arr = xp.ones((42, 31), dtype=dtype)
    dst_arr = xp.zeros_like(src_arr)

    kfunc(src=src_arr, dst=dst_arr, weight=2.0)

    expected = xp.zeros_like(src_arr)
    expected[1:-1, 1:-1] = 18.0

    xp.testing.assert_allclose(dst_arr, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filter_kernel_fixedsize(gen_config, xp, dtype):
    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    src_arr = xp.ones((42, 31), dtype=dtype)
    dst_arr = xp.zeros_like(src_arr)

    src = Field.create_from_numpy_array("src", xp.asarray(src_arr))
    dst = Field.create_from_numpy_array("dst", xp.asarray(dst_arr))

    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    cfg = gen_config.copy()
    cfg.default_dtype = dtype

    if cfg.get_target().is_cpu():
        cfg.cpu.openmp.enable = True

    ker = create_kernel(asms, cfg)

    for param in ker.parameters:
        if pprop := param.get_properties(FieldBasePtr):
            field: Field = pprop.pop().field
            assert isinstance(param.dtype, PsPointerType)
            if field == src:
                assert param.dtype.base_type.const
            else:
                assert not param.dtype.base_type.const

    inspect_kernel(ker, cfg)

    kfunc = ker.compile()

    kfunc(src=src_arr, dst=dst_arr, weight=2.0)

    expected = xp.zeros_like(src_arr)
    expected[1:-1, 1:-1] = 18.0

    xp.testing.assert_allclose(dst_arr, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "target", [t for t in Target.available_targets() if t.is_cpu()]
)
@pytest.mark.parametrize("openmp", [False, True])
@pytest.mark.parametrize("block_size", [2, 8, 64])
def test_cpu_loop_blocking(gen_config, dtype, openmp, block_size):
    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    src, dst = fields("src, dst: [2D]")
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    cfg = gen_config.copy()
    cfg.default_dtype = dtype
    cfg.cpu.openmp.enable = openmp

    k = TypedSymbol("k", DynamicType.INDEX_TYPE)
    cfg.cpu.loop_blocking = (k, 4)

    ker = create_kernel(asms, cfg)

    loops = list(filter(lambda n: isinstance(n, PsLoop), dfs_preorder(ker.body)))
    assert len(loops) == 5 if gen_config.target.is_vector_cpu() else 4

    kfunc = ker.compile()

    src_arr = np.ones((44, 31), dtype=dtype)
    dst_arr = np.zeros_like(src_arr)

    kfunc(src=src_arr, dst=dst_arr, weight=2.0, k=block_size)

    expected = np.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    np.testing.assert_allclose(dst_arr, expected)
