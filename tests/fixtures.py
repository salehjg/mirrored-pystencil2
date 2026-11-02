"""Fixtures for the pystencils test suite

This module provides a number of fixtures used by the pystencils test suite.
Use these fixtures wherever applicable to extend the code surface area covered
by your tests:

- All tests that should work for every target should use the `target` fixture
- All tests that should work with the highest optimization level for every target
  should use the `gen_config` fixture
- Use the `xp` fixture to access the correct array module (numpy or cupy) depending
  on the target

"""

from types import ModuleType

import pytest

import pystencils as ps
from pystencils.jit import CpuJit, NoJit, JitBase
from pystencils.jit.cpu import CompilerInfo
from pystencils.jit.error import JitError
from pystencils.jit.sycl import SYCLIcpxInfo, SYCLJit

AVAILABLE_TARGETS = ps.Target.available_targets()
TARGET_IDS = [t.name for t in AVAILABLE_TARGETS]


@pytest.fixture(params=AVAILABLE_TARGETS, ids=TARGET_IDS)
def target(request) -> ps.Target:
    """Provides all code generation targets available on the current hardware"""
    return request.param


@pytest.fixture
def compiler_info(target: ps.Target) -> CompilerInfo:
    return CompilerInfo.get_default(target=target)


@pytest.fixture
def cpujit(compiler_info, tmp_path) -> CpuJit:
    return CpuJit(compiler_info=compiler_info, objcache=tmp_path, emit_warnings=True)


@pytest.fixture
def sycl_jit(tmp_path) -> SYCLJit | NoJit:
    cinfo = SYCLIcpxInfo(optlevel="3")
    cinfo.extra_cxxflags = ["-Wall", "-Wconversion", "-Werror", "-fp-model=precise"]
    try:
        return SYCLJit(compiler_info=cinfo, objcache=tmp_path, emit_warnings=True)
    except JitError:
        return NoJit()


@pytest.fixture
def jit(target: ps.Target, compiler_info, tmp_path) -> JitBase:
    if target.is_cpu():
        #   Set target in compiler info such that `-march` is set accordingly
        return CpuJit(
            compiler_info=compiler_info,
            objcache=tmp_path,
            emit_warnings=True
        )

    elif target == ps.Target.SYCL:
        cinfo = SYCLIcpxInfo(optlevel="3")
        cinfo.extra_cxxflags = [
            "-Wall",
            "-Wconversion",
            "-Werror",
            "-fp-model=precise"
        ]
        try:
            return SYCLJit(compiler_info=cinfo,
                           objcache=tmp_path,
                           emit_warnings=True)
        except JitError:
            return NoJit()
    else:
        return NoJit()


@pytest.fixture
def gen_config(request: pytest.FixtureRequest, target: ps.Target, jit: JitBase):
    """Default codegen configuration for the current target.

    For GPU targets, set default indexing options.
    For vector-CPU targets, set default vectorization config.
    """

    gen_config = ps.CreateKernelConfig(target=target)

    if target.is_cpu() or target == ps.Target.SYCL:
        gen_config.jit = jit

    if target.is_vector_cpu():
        gen_config.cpu.vectorize.enable = True
        gen_config.cpu.vectorize.assume_inner_stride_one = True

        if target == ps.Target.ARM_SVE:
            gen_config.cpu.vectorize.lanes = 2

    return gen_config


@pytest.fixture()
def xp(target: ps.Target) -> ModuleType:
    """Primary array module for the current target.

    Returns:
        `cupy` if `target.is_gpu()`, and `numpy` otherwise
    """
    if target.is_gpu():
        import cupy as xp

        return xp
    elif target == ps.Target.SYCL:
        import tests.dpctl_compat as dpt

        return dpt
    else:
        import numpy as np

        return np
