"""On-hardware numeric execution tests for the RVV backend.

These compile and *run* RVV kernels, so they are skipped unless the current machine
actually exposes ``Target.RISCV_RVV`` (i.e. they run on the Banana Pi F3 board, and
under a riscv64 QEMU/container in CI, but are collected-and-skipped on x86 dev hosts).
"""

import os
import sys
import subprocess
import textwrap

import numpy as np
import pytest

import pystencils as ps
from pystencils import (
    Target,
    CreateKernelConfig,
    create_kernel,
    fields,
    Assignment,
    tcast,
)
from pystencils.sympyextensions import ReductionOp, reduction_assignment

pytestmark = pytest.mark.skipif(
    Target.RISCV_RVV not in Target.available_vector_cpu_targets(),
    reason="requires RVV hardware",
)


def rvv_kernel(
    assignments,
    *,
    lanes=None,
    vlen=256,
    default_dtype="float64",
    index_dtype="int64",
    mode="fixed",
):
    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = default_dtype
    cfg.index_dtype = index_dtype
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.mode = mode
    cfg.cpu.vectorize.assume_inner_stride_one = True
    cfg.cpu.rvv.vlen = vlen
    if mode == "fixed":
        cfg.cpu.vectorize.lanes = lanes
    return create_kernel(assignments, cfg).compile()


@pytest.mark.parametrize(
    "mode, lanes", [("fixed", 4), ("fixed", 8), ("vla", None)]
)
def test_core_arithmetic(mode, lanes):
    a, b, dst = fields("a, b, dst: float64[2D]", layout="c")
    k = rvv_kernel(
        [
            Assignment(
                dst.center(),
                (a.center() + b.center()) * (a.center() - b.center())
                / (b.center() + 1.0),
            )
        ],
        lanes=lanes,
        mode=mode,
    )
    rng = np.random.default_rng(0)
    a_arr = np.ascontiguousarray(rng.random((13, 37)))
    b_arr = np.ascontiguousarray(rng.random((13, 37)) + 0.5)
    dst_arr = np.zeros((13, 37))
    k(a=a_arr, b=b_arr, dst=dst_arr)
    ref = (a_arr + b_arr) * (a_arr - b_arr) / (b_arr + 1.0)
    np.testing.assert_allclose(dst_arr, ref, rtol=1e-13)


@pytest.mark.parametrize("inner", [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 64, 100])
def test_vla_tail_correctness(inner):
    """VLA must be correct for any inner extent -- the tail is absorbed by the final
    short vsetvl, with no separate remainder loop."""
    f, g = fields("f, g: float64[2D]", layout="c")
    k = rvv_kernel(
        [Assignment(g.center(), 3.0 * f.center() + 1.0)], mode="vla", vlen=128
    )
    rng = np.random.default_rng(inner)
    f_arr = np.ascontiguousarray(rng.random((3, inner)))
    g_arr = np.zeros((3, inner))
    k(f=f_arr, g=g_arr)
    np.testing.assert_allclose(g_arr, 3.0 * f_arr + 1.0, rtol=1e-13)


def test_cast_truncation_rtz():
    """C `(int)` casts truncate toward zero; RVV must use vfcvt_rtz, not the
    round-to-nearest vfcvt. -1.5 -> -1 (rtz), not -2 (round)."""
    src = fields("src: float64[2D]", layout="c")
    dst = fields("dst: int64[2D]", layout="c")
    k = rvv_kernel(
        [Assignment(dst.center(), tcast(src.center(), "int64"))],
        lanes=4,
        default_dtype="float64",
    )
    vals = np.array(
        [-1.5, 2.5, -2.5, 3.9, -3.9, 0.5, -0.5, 7.2, -7.2, 0.0, 1.0, -1.0, 100.9, -100.9],
        dtype=np.float64,
    )
    src_arr = np.ascontiguousarray(np.tile(vals, (5, 1)))
    dst_arr = np.zeros_like(src_arr, dtype=np.int64)
    k(src=src_arr, dst=dst_arr)
    np.testing.assert_array_equal(dst_arr, np.trunc(src_arr).astype(np.int64))


def test_float_widen_narrow_roundtrip():
    src = fields("src: float32[2D]", layout="c")
    dst = fields("dst: float32[2D]", layout="c")
    #   f32 -> f64 (widen) -> arithmetic -> f32 (narrow)
    wide = tcast(src.center(), "float64")
    k = rvv_kernel(
        [Assignment(dst.center(), tcast(wide * 2.0 + 1.0, "float32"))],
        lanes=8,
        default_dtype="float32",
        index_dtype="int32",
    )
    rng = np.random.default_rng(1)
    src_arr = np.ascontiguousarray(rng.random((11, 29)).astype(np.float32))
    dst_arr = np.zeros_like(src_arr)
    k(src=src_arr, dst=dst_arr)
    ref = (src_arr.astype(np.float64) * 2.0 + 1.0).astype(np.float32)
    np.testing.assert_allclose(dst_arr, ref, rtol=1e-6)


def test_int_widen_narrow():
    src = fields("src: int32[2D]", layout="c")
    dst = fields("dst: int32[2D]", layout="c")
    #   i32 -> i64 (widen) -> *3 -> i32 (narrow)
    k = rvv_kernel(
        [Assignment(dst.center(), tcast(tcast(src.center(), "int64") * 3, "int32"))],
        lanes=8,
        default_dtype="int32",
        index_dtype="int32",
    )
    src_arr = np.ascontiguousarray(
        np.arange(11 * 29, dtype=np.int32).reshape(11, 29) - 100
    )
    dst_arr = np.zeros_like(src_arr)
    k(src=src_arr, dst=dst_arr)
    np.testing.assert_array_equal(dst_arr, (src_arr.astype(np.int64) * 3).astype(np.int32))


def test_broadcast_lbm_like():
    """Broadcast scalar coefficients (LBM equilibrium-style)."""
    src = fields("src: float64[2D]", layout="c")
    dst = fields("dst: float64[2D]", layout="c")
    w0, c1, c2 = 0.4444444444, 3.0, 1.5
    k = rvv_kernel(
        [Assignment(dst.center(), w0 * src.center() * c1 - c2 / (src.center() + 2.0))],
        lanes=4,
    )
    rng = np.random.default_rng(2)
    src_arr = np.ascontiguousarray(rng.random((9, 31)))
    dst_arr = np.zeros_like(src_arr)
    k(src=src_arr, dst=dst_arr)
    ref = w0 * src_arr * c1 - c2 / (src_arr + 2.0)
    np.testing.assert_allclose(dst_arr, ref, rtol=1e-13)


_GUARD_TRIP_PROG = textwrap.dedent(
    """
    import numpy as np
    from pystencils import Target, CreateKernelConfig, create_kernel, fields, Assignment
    f, g = fields("f, g: float64[2D]", layout="c")
    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.lanes = {lanes}
    cfg.cpu.vectorize.assume_inner_stride_one = True
    cfg.cpu.rvv.vlen = {vlen}
    k = create_kernel([Assignment(g.center(), 2 * f.center())], cfg).compile()
    a = np.ones((4, 32)); b = np.zeros((4, 32))
    k(f=a, g=b)
    assert np.allclose(b, 2.0)
    print("KERNEL_RAN_OK")
    """
)


def _run_guard_prog(vlen, lanes):
    env = dict(os.environ)
    #   Ensure the child sees the same pystencils source checkout.
    src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src")
    )
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", _GUARD_TRIP_PROG.format(vlen=vlen, lanes=lanes)],
        capture_output=True,
        env=env,
    )


def test_vlen_guard_trips_on_narrow_hardware():
    """D8: a kernel built assuming VLEN=512 must ABORT on the 256-bit board rather
    than silently corrupt data; a kernel built for VLEN=256 must run fine."""
    #   vlen=512, lanes=8 -> f64 LMUL=1; guard require_vlen(512) must trip on VLEN=256.
    trip = _run_guard_prog(vlen=512, lanes=8)
    assert b"KERNEL_RAN_OK" not in trip.stdout, (
        "kernel built for VLEN=512 must not run to completion on a 256-bit board"
    )
    assert trip.returncode != 0
    assert b"VLEN" in trip.stderr  # our fprintf diagnostic

    #   vlen=256, lanes=4 -> matches the board; must run cleanly.
    ok = _run_guard_prog(vlen=256, lanes=4)
    assert ok.returncode == 0, ok.stderr.decode(errors="replace")
    assert b"KERNEL_RAN_OK" in ok.stdout


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "op", [ReductionOp.Add, ReductionOp.Sub, ReductionOp.Min, ReductionOp.Max]
)
def test_reduction(dtype, op):
    lanes = 4 if dtype == "float64" else 8
    x = ps.fields(f"x: {dtype}[2D]")
    w = ps.TypedSymbol("w", dtype)
    red = reduction_assignment(w, op, x.center())

    cfg = CreateKernelConfig()
    cfg.target = Target.RISCV_RVV
    cfg.default_dtype = dtype
    cfg.cpu.vectorize.enable = True
    cfg.cpu.vectorize.lanes = lanes
    cfg.cpu.rvv.vlen = 256
    k = create_kernel([red], cfg).compile()

    rng = np.random.default_rng(3)
    arr = np.ascontiguousarray(
        (rng.random((16, 20)) * 10 - 5).astype(dtype)
    )
    init = 5.0
    acc = np.full((1,), init, dtype=dtype)
    k(x=arr, w=acc)

    match op:
        case ReductionOp.Add:
            ref = init + np.sum(arr, dtype=np.float64)
        case ReductionOp.Sub:
            ref = init - np.sum(arr, dtype=np.float64)
        case ReductionOp.Min:
            ref = min(init, float(np.min(arr)))
        case ReductionOp.Max:
            ref = max(init, float(np.max(arr)))

    rtol = 1e-6 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(acc[0], ref, rtol=rtol)
