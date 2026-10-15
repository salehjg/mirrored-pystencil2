"""WP1 tests for the RISC-V RVV target: enum algebra, ISA-string detection,
config surface, and JIT ``-march`` flag emission.

These are pure codegen/configuration tests and run on any host (no RISC-V toolchain
required).
"""

import pytest

from pystencils import Target
from pystencils.codegen import VectorizationMode, RvvOptions
from pystencils.codegen.config import CreateKernelConfig
from pystencils.codegen.errors import CodegenError
from pystencils.codegen.target import _riscv_isa_string_has_rvv
from pystencils.jit.cpu.compiler_info import (
    GccInfo,
    ClangInfo,
    _synthesize_riscv_march,
)


#   ---- Enum algebra ------------------------------------------------------------


def test_rvv_target_flags():
    t = Target.RISCV_RVV
    assert t.is_cpu()
    assert t.is_vector_cpu()
    assert not t.is_gpu()
    assert not t.is_automatic()
    #   RVV is distinct from every other vector target
    assert t not in Target.ARM_SVE
    assert Target._RISCV in t and Target._RVV in t
    #   It must not accidentally overlap with SVE / x86 component flags
    assert Target._SVE not in t
    assert Target._X86 not in t


def test_rvv_no_default_lanes():
    from pystencils.types import PsIeeeFloatType

    with pytest.raises(CodegenError):
        Target.RISCV_RVV.default_vector_lanes(PsIeeeFloatType(64))


#   ---- ISA string detection ----------------------------------------------------


@pytest.mark.parametrize(
    "isa, expected",
    [
        ("rv64imafdcv_zicsr_zvl256b", True),
        ("rv64gcv", True),
        ("rv64imafdcv", True),
        #   The real SpacemiT K1 string (VLEN=256, no Zvl256b token)
        (
            "rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_"
            "zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_"
            "zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt",
            True,
        ),
        #   RVV 0.7 (T-Head) must NOT be detected as RVV 1.0
        ("rv64gc_xtheadvector", False),
        #   Vector *embedded* (Zve*) subsets without full V must not match
        ("rv64imafdc_zve64d_zve32f", False),
        ("rv64gc", False),
        ("rv64imafdc_zicsr_zvl128b", False),
        (None, False),
        ("", False),
        ("garbage", False),
    ],
)
def test_riscv_isa_detection(isa, expected):
    assert _riscv_isa_string_has_rvv(isa) is expected


#   ---- march synthesis ---------------------------------------------------------


@pytest.mark.parametrize(
    "isa, expected",
    [
        ("rv64imafdcv_zicsr_zvl256b", "rv64imafdcv_zicsr_zvl256b"),
        ("rv64gc_xtheadvector_zba", "rv64gc_zba"),
        ("  rv64GCV_Zba  ", "rv64gcv_zba"),
        #   supervisor/hypervisor extensions are dropped (QEMU advertises svadu, which
        #   GCC 14 rejects); z* vector extensions are kept
        ("rv64imafdcv_zve64d_sstc_svadu_svpbmt", "rv64imafdcv_zve64d"),
        (None, None),
        ("garbage", None),
    ],
)
def test_synthesize_riscv_march(isa, expected):
    assert _synthesize_riscv_march(isa) == expected


#   ---- JIT flag emission -------------------------------------------------------


def test_rvv_compiler_flags_gcc():
    info = GccInfo(target=Target.RISCV_RVV, openmp=False)
    flags = info.cxxflags()
    assert "-march=rv64gcv" in flags
    assert "-mrvv-vector-bits=zvl" not in flags


def test_rvv_compiler_flags_vls_with_vlen():
    info = GccInfo(
        target=Target.RISCV_RVV, rvv_vlen=256, rvv_fixed_length=True, openmp=False
    )
    flags = info.cxxflags()
    assert "-march=rv64gcv_zvl256b" in flags
    assert "-mrvv-vector-bits=zvl" in flags


def test_rvv_compiler_flags_vla_no_vector_bits():
    #   VLA mode: floor VLEN goes into -march, but never -mrvv-vector-bits
    info = GccInfo(
        target=Target.RISCV_RVV, rvv_vlen=128, rvv_fixed_length=False, openmp=False
    )
    flags = info.cxxflags()
    assert "-march=rv64gcv_zvl128b" in flags
    assert "-mrvv-vector-bits=zvl" not in flags


def test_clang_rvv_flags():
    info = ClangInfo(target=Target.RISCV_RVV, rvv_vlen=256, openmp=False)
    flags = info.cxxflags()
    assert "-march=rv64gcv_zvl256b" in flags


#   ---- Config surface ----------------------------------------------------------


def test_vectorization_mode_default_and_parsing():
    cfg = CreateKernelConfig()
    assert cfg.cpu.vectorize.get_option("mode") == VectorizationMode.FIXED

    cfg.cpu.vectorize.mode = "vla"
    assert cfg.cpu.vectorize.get_option("mode") == VectorizationMode.STRIPMINE

    cfg.cpu.vectorize.mode = "fixed"
    assert cfg.cpu.vectorize.get_option("mode") == VectorizationMode.FIXED

    cfg.cpu.vectorize.mode = VectorizationMode.STRIPMINE
    assert cfg.cpu.vectorize.get_option("mode") == VectorizationMode.STRIPMINE

    with pytest.raises(ValueError):
        cfg.cpu.vectorize.mode = "nonsense"


def test_rvv_options_category():
    cfg = CreateKernelConfig()
    assert isinstance(cfg.cpu.rvv, RvvOptions)
    assert cfg.cpu.rvv.get_option("check_vlen") is True
    assert cfg.cpu.rvv.get_option("lmul") == 1

    cfg.cpu.rvv.vlen = 256
    assert cfg.cpu.rvv.get_option("vlen") == 256
