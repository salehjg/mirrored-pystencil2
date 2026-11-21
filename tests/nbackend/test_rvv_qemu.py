"""Execution tests for the RVV platform under QEMU user-mode emulation, on any host.

The kernels are generated for each RVV target (VLEN 256 and 128) and LMUL 1 and 2, cross-compiled
with `riscv64-linux-gnu-g++ -static`, and run under `qemu-riscv64 -cpu rv64,v=true,vlen=<N>`. Their
outputs are compared with the same kernels generated for `Target.GenericCPU` and run through the
JIT on the host. This checks what the code generation tests cannot: that every lane of every
vector is computed and written exactly once, at each VLEN and LMUL. Code for VLEN 128 is also run
on a VLEN-256 machine (it must stay correct there), and code for VLEN 256 must refuse to compile
for VLEN 128.

Skipped without `qemu-riscv64` or the cross compiler, and on riscv64 hosts (where
test_rvv_execution.py runs kernels natively).
"""

import platform
import shutil
import subprocess

import numpy as np
import pytest
import sympy as sp

import pystencils as ps
from pystencils import Target, Assignment, TypedSymbol
from pystencils.include import get_pystencils_include_path
from pystencils.sympyextensions import ReductionOp, reduction_assignment
from pystencils.sympyextensions.typed_sympy import tcast

CXX = shutil.which("riscv64-linux-gnu-g++")
QEMU = shutil.which("qemu-riscv64")
pytestmark = pytest.mark.skipif(
    CXX is None or QEMU is None or platform.machine() == "riscv64",
    reason="needs qemu-riscv64 and riscv64-linux-gnu-g++ on a non-RISC-V host",
)

SHAPE = (5, 23)  # odd inner extent: the scalar remainder loop runs at every lane count
SENTINEL = -7.0  # initial value of output fields: cells a kernel must not write keep it


def _kernels():
    """name -> (assignments, default dtype)"""
    a, b = ps.fields(f"a, b: float64[{SHAPE[0]}, {SHAPE[1]}]")
    s, t = ps.fields(f"s, t: float32[{SHAPE[0]}, {SHAPE[1]}]")
    w = TypedSymbol("w", "float64")
    v = TypedSymbol("v", "float32")
    x = ps.DEFAULTS.spatial_counters[1]
    return {
        "stencil64": ([Assignment(b.center(), 2 * a.center() + a[1, 0] - a[0, -1] * a[-1, 0])], "float64"),
        "stencil32": ([Assignment(t.center(), sp.sqrt(sp.Abs(s.center())) + 3 * s[0, 1])], "float32"),
        "cast": ([Assignment(b.center(), tcast(s.center(), "float64") * 0.5 + a.center())], "float64"),
        "counter": ([Assignment(b.center(), tcast(x, "float64") * 0.25)], "float64"),
        "sum64": ([reduction_assignment(w, ReductionOp.Add, a.center())], "float64"),
        "max32": ([reduction_assignment(v, ReductionOp.Max, s.center())], "float32"),
    }


def _inputs() -> dict[str, np.ndarray]:
    """Initial values of every kernel parameter, by parameter name."""
    rng = np.random.default_rng(7)
    return {
        "_data_a": rng.uniform(-2, 2, SHAPE),
        "_data_s": rng.uniform(-2, 2, SHAPE).astype(np.float32),
        "_data_b": np.full(SHAPE, SENTINEL),
        "_data_t": np.full(SHAPE, SENTINEL, dtype=np.float32),
        "w": np.array([1.5]),
        "v": np.array([-1e30], dtype=np.float32),
    }


def _config(target: Target, dtype: str, lmul: int | None, name: str) -> ps.CreateKernelConfig:
    cfg = ps.CreateKernelConfig(target=target, default_dtype=dtype, function_name=name)
    if target != Target.GenericCPU:
        cfg.cpu.vectorize.enable = True
        cfg.cpu.vectorize.assume_inner_stride_one = True
        cfg.cpu.vectorize.rvv_lmul = lmul
    return cfg


def _host_reference() -> dict[str, dict[str, np.ndarray]]:
    """Outputs of every kernel, generated for GenericCPU and run on the host."""
    results = {}
    for name, (asms, dtype) in _kernels().items():
        func = ps.create_kernel(asms, _config(Target.GenericCPU, dtype, None, name)).compile()
        arrays = {p.name: _inputs()[p.name].copy() for p in func.kernel.parameters}
        func(**{(n[6:] if n.startswith("_data_") else n): arr for n, arr in arrays.items()})
        results[name] = arrays
    return results


def _c_array(name: str, arr: np.ndarray) -> str:
    ctype = "double" if arr.dtype == np.float64 else "float"
    values = ", ".join(float(x).hex() for x in arr.ravel())
    return f"static {ctype} {name}[{arr.size}] = {{{values}}};"


def _program(target: Target, lmul: int | None) -> str:
    """One C++ program running every kernel once and printing all its parameters in hex."""
    headers: set[str] = set()
    kernels: list[str] = []
    main: list[str] = []
    for name, (asms, dtype) in _kernels().items():
        ker = ps.create_kernel(asms, _config(target, dtype, lmul, name))
        headers |= ker.required_headers
        kernels.append(ker.get_c_code())
        params = [p.name for p in ker.parameters]
        main += [_c_array(f"{name}{p}", _inputs()[p]) for p in params]
        main.append(f"{name}({', '.join(name + p for p in params)});")
        for p in params:
            n = _inputs()[p].size
            main.append(
                f'for (int i = 0; i < {n}; ++i) printf("{name} {p} %d %a\\n", i, (double) {name}{p}[i]);'
            )
    return "\n".join(
        [f"#include {h}" for h in sorted(headers)]
        + ["#include <cstdio>", "#define RESTRICT __restrict__"]
        + kernels
        + ["int main()", "{"] + main + ["return 0;", "}"]
    )


def _compile(source: str, vlen: int, exe) -> subprocess.CompletedProcess:
    cmd = [
        CXX, "-std=c++17", "-O2", "-static", f"-march=rv64gcv_zvl{vlen}b",
        f"-I{get_pystencils_include_path()}", "-x", "c++", "-", "-o", str(exe),
    ]
    return subprocess.run(cmd, input=source, capture_output=True, text=True)


def _parse(stdout: str) -> dict[tuple[str, str], dict[int, float]]:
    values: dict[tuple[str, str], dict[int, float]] = {}
    for line in stdout.splitlines():
        name, param, idx, val = line.split()
        values.setdefault((name, param), {})[int(idx)] = float.fromhex(val)
    return values


@pytest.mark.parametrize(
    "target, lmul, machine_vlen",
    [
        (Target.RISCV_RVV256, None, 256),
        (Target.RISCV_RVV128, None, 128),
        (Target.RISCV_RVV128, 2, 128),
        (Target.RISCV_RVV256, 2, 256),
        (Target.RISCV_RVV128, None, 256),  # VLEN-128 code on a VLEN-256 machine
    ],
)
def test_kernels_under_qemu(tmp_path, target, lmul, machine_vlen):
    exe = tmp_path / "kernels"
    built = _compile(_program(target, lmul), target.rvv_vlen(), exe)
    assert built.returncode == 0, built.stderr
    run = subprocess.run(
        [QEMU, "-cpu", f"rv64,v=true,vlen={machine_vlen}", str(exe)], capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stderr
    got = _parse(run.stdout)

    for name, arrays in _host_reference().items():
        for param, expected in arrays.items():
            values = got[(name, param)]
            actual = np.array([values[i] for i in range(expected.size)], dtype=expected.dtype).reshape(expected.shape)
            rtol = 1e-12 if expected.dtype == np.float64 else 1e-5
            #   Same cells written (the sentinel survives elsewhere), same values up to rounding
            #   (fused multiply-adds and the order of reduction sums differ from the host)
            np.testing.assert_array_equal(actual == SENTINEL, expected == SENTINEL, err_msg=f"{name} {param}")
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-12, err_msg=f"{name} {param}")


def test_vlen256_code_refuses_vlen128_flags(tmp_path):
    asms, dtype = _kernels()["stencil64"]
    ker = ps.create_kernel(asms, _config(Target.RISCV_RVV256, dtype, None, "stencil64"))
    source = "\n".join([f"#include {h}" for h in sorted(ker.required_headers)] + ["#define RESTRICT __restrict__", ker.get_c_code()])
    built = _compile(source + "\nint main() { return 0; }\n", 128, tmp_path / "x")
    assert built.returncode != 0
    assert "need -march=rv64gcv_zvl256b" in built.stderr


def test_vlen256_code_is_wrong_on_a_vlen128_machine(tmp_path):
    """Why the VLEN must match: code for VLEN 256, compiled with matching flags, silently computes
    only part of each vector on VLEN-128 hardware (each intrinsic gets fewer lanes than it asks
    for, while the loop still steps by the full count). This is the control for the test above."""
    exe = tmp_path / "kernels"
    assert _compile(_program(Target.RISCV_RVV256, None), 256, exe).returncode == 0
    run = subprocess.run([QEMU, "-cpu", "rv64,v=true,vlen=128", str(exe)], capture_output=True, text=True, timeout=120)
    got = _parse(run.stdout)
    expected = _host_reference()["stencil64"]["_data_b"]
    actual = np.array([got[("stencil64", "_data_b")][i] for i in range(expected.size)]).reshape(expected.shape)
    assert not np.allclose(actual, expected)
