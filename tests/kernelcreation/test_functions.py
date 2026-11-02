import sympy as sp
import numpy as np
import pytest
from dataclasses import replace
from itertools import product

from pystencils import (
    fields,
    create_kernel,
    CreateKernelConfig,
    Target,
    Assignment,
    Field,
)
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.expressions import PsCall


def constant(name, dtype):
    return {
        "pi": (sp.pi, dtype(np.pi)),
        "e": (sp.E, dtype(np.e)),
        "infinity": (sp.core.numbers.Infinity(), dtype(np.inf)),
        "neg_infinity": (sp.core.numbers.NegativeInfinity(), -dtype(np.inf)),
    }[name]


def unary_function(name, xp):
    return {
        "exp": (sp.exp, xp.exp),
        "log": (sp.log, xp.log),
        "sin": (sp.sin, xp.sin),
        "cos": (sp.cos, xp.cos),
        "tan": (sp.tan, xp.tan),
        "sinh": (sp.sinh, xp.sinh),
        "cosh": (sp.cosh, xp.cosh),
        "tanh": (sp.tanh, xp.tanh),
        "asin": (sp.asin, xp.arcsin),
        "acos": (sp.acos, xp.arccos),
        "atan": (sp.atan, xp.arctan),
        "sqrt": (sp.sqrt, xp.sqrt),
        "abs": (sp.Abs, xp.abs),
        "floor": (sp.floor, xp.floor),
        "ceil": (sp.ceiling, xp.ceil),
    }[name]


def binary_function(name, xp):
    return {
        "min": (sp.Min, xp.fmin),
        "max": (sp.Max, xp.fmax),
        "pow": (sp.Pow, xp.power),
        "atan2": (sp.atan2, xp.arctan2),
    }[name]


AVAIL_TARGETS = Target.available_targets()
TARGETS_WITHOUT_VECTOR_CPUS = [t for t in AVAIL_TARGETS if not t.is_vector_cpu()]


@pytest.fixture
def function_domain(function_name, dtype):
    eps = 1e-6
    rng = np.random.default_rng()

    match function_name:
        case (
            "exp"
            | "sin"
            | "cos"
            | "sinh"
            | "cosh"
            | "tanh"
            | "atan"
            | "abs"
            | "floor"
            | "ceil"
        ):
            return np.concatenate(
                [
                    [0.0, -1.0, 1.0],
                    rng.uniform(-0.1, 0.1, 5),
                    rng.uniform(-5.0, 5.0, 8),
                    rng.uniform(-10, 10, 8),
                ]
            ).astype(dtype)
        case "tan":
            return np.concatenate(
                [
                    [0.0, -1.0, 1.0],
                    rng.uniform(-np.pi / 2.0 + eps, np.pi / 2.0 - eps, 13),
                ]
            ).astype(dtype)
        case "asin" | "acos":
            return np.concatenate(
                [
                    [0.0, 1.0, -1.0],
                    rng.uniform(-1.0, 1.0, 13),
                ]
            ).astype(dtype)
        case "log" | "sqrt":
            return np.concatenate(
                [
                    [1.0],
                    rng.uniform(eps, 0.1, 7),
                    rng.uniform(eps, 1.0, 8),
                    rng.uniform(eps, 1e6, 8),
                ]
            ).astype(dtype)
        case "min" | "max" | "atan2":
            return np.concatenate(
                [
                    rng.uniform(-0.1, 0.1, 8),
                    rng.uniform(-5.0, 5.0, 8),
                    rng.uniform(-10, 10, 8),
                ]
            ).astype(dtype), np.concatenate(
                [
                    rng.uniform(-0.1, 0.1, 8),
                    rng.uniform(-5.0, 5.0, 8),
                    rng.uniform(-10, 10, 8),
                ]
            ).astype(
                dtype
            )
        case "pow":
            return np.concatenate(
                [
                    [0.0, 1.0, 1.0],
                    rng.uniform(-1.0, 1.0, 8),
                    rng.uniform(0.0, 5.0, 8),
                ]
            ).astype(dtype), np.concatenate(
                [
                    [1.0, 0.0, 2.0],
                    np.arange(2.0, 10.0, 1.0),
                    rng.uniform(-2.0, 2.0, 8),
                ]
            ).astype(
                dtype
            )
        case _:
            assert False, "I don't know the domain of that function"


@pytest.mark.parametrize(
    "function_name, target",
    list(
        product(
            (
                "exp",
                "log",
                "sin",
                "cos",
                "tan",
                "sinh",
                "cosh",
                "tanh",
                "asin",
                "acos",
                "atan",
            ),
            TARGETS_WITHOUT_VECTOR_CPUS,
        )
    )
    + list(
        product(
            ["floor", "ceil"],
            [
                t
                for t in AVAIL_TARGETS
                if not (
                    Target._AVX512 in t
                    or Target._NEON in t
                    or t == Target.ARM_SVE
                    or t == Target.RISCV_RVV
                )
            ],
        )
    )
    + list(product(["sqrt", "abs"], AVAIL_TARGETS)),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_unary_functions(gen_config, xp, function_name, dtype, function_domain):
    sp_func, xp_func = unary_function(function_name, xp)
    resolution = np.finfo(dtype).resolution

    #   Array size should be larger than eight, such that vectorized kernels don't just run their remainder loop
    inp = xp.array(function_domain)
    outp = xp.zeros_like(inp)

    reference = xp_func(inp)

    inp_field = Field.create_from_numpy_array("inp", xp.asarray(inp))
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center()))]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    xp.testing.assert_allclose(outp, reference, rtol=resolution)


@pytest.mark.parametrize(
    "function_name,target",
    list(product(["min", "max"], AVAIL_TARGETS))
    + list(product(["pow", "atan2"], TARGETS_WITHOUT_VECTOR_CPUS)),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_binary_functions(gen_config, xp, function_name, dtype, function_domain):
    sp_func, xp_func = binary_function(function_name, xp)
    resolution: dtype = np.finfo(dtype).resolution

    inp = xp.array(function_domain[0])
    inp2 = xp.array(function_domain[1])
    outp = xp.zeros_like(inp)

    reference = xp_func(inp, inp2)

    inp_field = Field.create_from_numpy_array("inp", xp.asarray(inp))
    inp2_field = Field.create_from_numpy_array("inp2", xp.asarray(inp2))
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [
        Assignment(
            outp_field.center(), sp_func(inp_field.center(), inp2_field.center())
        )
    ]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, inp2=inp2, outp=outp)

    xp.testing.assert_allclose(outp, reference, rtol=resolution)


@pytest.mark.parametrize(
    "dtype, target",
    list(product([np.int32], AVAIL_TARGETS))
    + list(
        product(
            [np.int64],
            [t for t in AVAIL_TARGETS if t not in (Target.X86_SSE, Target.X86_AVX)],
        )
    ),
)
def test_integer_abs(gen_config, xp, dtype):
    sp_func, xp_func = unary_function("abs", xp)

    smallest = np.iinfo(dtype).min
    largest = np.iinfo(dtype).max

    inp = xp.array([-1, 0, 1, 3, -5, -312, smallest + 1, largest], dtype=dtype)

    outp = xp.zeros_like(inp)
    reference = xp_func(inp)

    inp_field = Field.create_from_numpy_array("inp", xp.asarray(inp))
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center()))]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    xp.testing.assert_array_equal(outp, reference)


@pytest.mark.parametrize("function_name", ("min", "max"))
@pytest.mark.parametrize(
    "dtype, target",
    list(product([np.int32], AVAIL_TARGETS))
    + list(
        product(
            [np.int64],
            [
                t
                for t in AVAIL_TARGETS
                if t
                not in (
                    Target.X86_SSE,
                    Target.X86_AVX,
                    Target.ARM_NEON,
                    Target.ARM_NEON_FP16,
                )
            ],
        )
    ),
)
def test_integer_binary_functions(gen_config, xp, function_name, dtype):
    sp_func, xp_func = binary_function(function_name, xp)

    smallest = np.iinfo(dtype).min
    largest = np.iinfo(dtype).max

    inp1 = xp.array([-1, 0, 1, 3, -5, -312, smallest + 1, largest], dtype=dtype)
    inp2 = xp.array([3, -5, 1, 12, 1, 11, smallest + 42, largest - 3], dtype=dtype)

    outp = xp.zeros_like(inp1)
    reference = xp_func(inp1, inp2)

    inp_field = Field.create_from_numpy_array("inp1", xp.asarray(inp1))
    inp2_field = Field.create_from_numpy_array("inp2", xp.asarray(inp2))
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [
        Assignment(
            outp_field.center(), sp_func(inp_field.center(), inp2_field.center())
        )
    ]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp1=inp1, inp2=inp2, outp=outp)

    xp.testing.assert_array_equal(outp, reference)


@pytest.mark.parametrize("c_name", ["pi", "e", "infinity", "neg_infinity"])
@pytest.mark.parametrize(
    "target, dtype",
    list(product(AVAIL_TARGETS, [np.float32, np.float64]))
    + [(t, np.float16) for t in AVAIL_TARGETS if t.is_gpu()],
)
def test_constants(c_name, dtype, gen_config, xp):
    c_sp, c_np = constant(c_name, dtype)

    outp = xp.zeros(
        (17,), dtype=dtype
    )  # 17 entries to run both vectorized and remainder loops
    reference = xp.zeros_like(outp)
    reference[:] = c_np

    outp_field = Field.create_from_numpy_array("outp", xp.asarray(outp))
    asm = Assignment(outp_field(0), c_sp)

    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asm, gen_config)
    kfunc = kernel.compile()
    kfunc(outp=outp)

    xp.testing.assert_array_equal(outp, reference)


@pytest.mark.parametrize("a", [sp.Symbol("a"), fields("a:  float64[2d]").center])
def test_avoid_pow(a):
    x = fields("x:  float64[2d]")

    up = Assignment(x.center_vector[0], 2 * a**2 / 3)
    func = create_kernel(up)

    powers = list(
        dfs_preorder(
            func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name
        )
    )
    assert not powers


@pytest.mark.xfail(reason="fast_div not available yet")
def test_avoid_pow_fast_div():
    x = fields("x:  float64[2d]")
    a = fields("a:  float64[2d]").center

    up = Assignment(x.center_vector[0], fast_division(1, (a**2)))
    func = create_kernel(up, config=CreateKernelConfig(target=Target.GPU))

    powers = list(
        dfs_preorder(
            func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name
        )
    )
    assert not powers


def test_avoid_pow_move_constants():
    # At the end of the kernel creation the function move_constants_before_loop will be called
    # This function additionally contains substitutions for symbols with the same value
    # Thus it simplifies the equations again
    x = fields("x:  float64[2d]")
    a, b, c = sp.symbols("a, b, c")

    up = [
        Assignment(a, 0.0),
        Assignment(b, 0.0),
        Assignment(c, 0.0),
        Assignment(
            x.center_vector[0],
            a**2 / 18 - a * b / 6 - a / 18 + b**2 / 18 + b / 18 - c**2 / 36,
        ),
    ]
    func = create_kernel(up)

    powers = list(
        dfs_preorder(
            func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name
        )
    )
    assert not powers
