import numpy as np
import pytest
from itertools import product
from dataclasses import replace

from pystencils import (
    create_kernel,
    Target,
    Assignment,
    Field,
)
from pystencils.sympyextensions.typed_sympy import tcast


AVAILABLE_TARGETS = Target.available_targets()


TYPE_CAST_COMBINATIONS = list(
    product(
        [
            t
            for t in AVAILABLE_TARGETS
            if Target._X86 in t and Target._AVX512 not in t and Target._SSE not in t
        ],
        [np.int32, np.float32, np.float64],
        [np.int32, np.float32, np.float64],
    )
) + list(
    product(
        [t for t in AVAILABLE_TARGETS if not t.is_vector_cpu() or Target._AVX512 in t],
        [np.int32, np.int64, np.float32, np.float64],
        [np.int32, np.int64, np.float32, np.float64],
    )
)

if Target.ARM_NEON in AVAILABLE_TARGETS:
    TYPE_CAST_COMBINATIONS += [
        (Target.ARM_NEON, from_type, to_type)
        for from_type, to_type in [
            (np.int32, np.float32),
            (np.int64, np.float64),
            (np.uint32, np.float32),
            (np.uint64, np.float64),
            (np.float32, np.uint32),
            (np.float64, np.uint64),
            (np.float32, np.int32),
            (np.float64, np.int64),
            (np.float32, np.float64),
            (np.float64, np.float32),
        ]
    ]

if Target.ARM_NEON_FP16 in AVAILABLE_TARGETS:
    TYPE_CAST_COMBINATIONS += [
        (Target.ARM_NEON_FP16, from_type, to_type)
        for from_type, to_type in [
            (np.int16, np.float16),
            (np.uint16, np.float16),
            (np.float16, np.uint16),
            (np.float16, np.int16),
            (np.float16, np.float32),
            (np.float32, np.float16),
            (np.float32, np.float64),
        ]
    ]

if Target.RISCV_RVV in AVAILABLE_TARGETS:
    #   Signed integers and floats of 32/64 bit; the wider type sets the lane count, so
    #   every conversion changes the element width by at most a factor of two.
    TYPE_CAST_COMBINATIONS += [
        (Target.RISCV_RVV, from_type, to_type)
        for from_type, to_type in product(
            [np.int32, np.int64, np.float32, np.float64], repeat=2
        )
    ]

if Target.ARM_SVE in AVAILABLE_TARGETS:
    TYPE_CAST_COMBINATIONS += [
        (Target.ARM_SVE, from_type, to_type)
        for from_type, to_type in product(
            [
                np.int16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ],
            repeat=2,
        )
        if from_type != to_type
        and np.dtype(from_type).itemsize == np.dtype(to_type).itemsize
    ]


@pytest.mark.parametrize("target, from_type, to_type", TYPE_CAST_COMBINATIONS)
def test_type_cast(gen_config, xp, from_type, to_type):
    if np.issubdtype(from_type, np.floating):
        if np.issubdtype(to_type, np.unsignedinteger):
            inp = xp.array([1.25, 0, 1.5, 3, 5, 312, 42, 6.625, 9], dtype=from_type)
        else:
            inp = xp.array(
                [-1.25, -0, 1.5, 3, -5, -312, 42, 6.625, -9], dtype=from_type
            )
    elif np.issubdtype(from_type, np.signedinteger):
        inp = xp.array([-1, 0, 1, 3, -5, -312, 42, 6, -9], dtype=from_type)
    else:
        inp = xp.array([15, 0, 1, 3, 5, 312, 42, 6, 9], dtype=from_type)

    outp = xp.ndarray.astype(xp.zeros_like(inp), to_type)
    truncated = xp.ndarray.astype(inp, to_type)
    rounded = xp.ndarray.astype(xp.round(inp), to_type)

    inp_field = Field.create_from_numpy_array("inp", xp.asarray(inp))
    outp_field = Field.create_from_numpy_array("outp", xp.asarray(outp))

    asms = [Assignment(outp_field.center(), tcast(inp_field.center(), to_type))]

    max_width = max(np.dtype(from_type).itemsize, np.dtype(to_type).itemsize) * 8
    gen_config = replace(gen_config, default_dtype=f"float{max_width}")

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    if np.issubdtype(from_type, np.floating) and not np.issubdtype(
        to_type, np.floating
    ):
        # rounding mode depends on platform
        try:
            xp.testing.assert_array_equal(outp, truncated)
        except AssertionError:
            xp.testing.assert_array_equal(outp, rounded)
    else:
        xp.testing.assert_array_equal(outp, truncated)
