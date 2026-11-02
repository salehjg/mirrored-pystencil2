"""Target, detection and JIT-flag tests for `Target.RISCV_RVV` (VLS, VLEN = 256).

Pure Python -- these run on any host.
"""

import pytest

from pystencils import Target
from pystencils.codegen.target import _riscv_has_rvv
from pystencils.jit.cpu import GccInfo, ClangInfo
from pystencils.jit.cpu import compiler_info
from pystencils.types.quick import Fp, SInt

#   /proc/cpuinfo `isa` line of the SpacemiT K1 (Banana Pi F3). Note: no `zvl256b` token,
#   the kernel does not advertise VLEN -- which is one reason VLEN is hardcoded.
K1_ISA = (
    "rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin"
    "_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt"
    "_sscofpmf_sstc_svinval_svnapot_svpbmt"
)


def test_target_flags():
    t = Target.RISCV_RVV
    assert t.is_cpu() and t.is_vector_cpu()
    assert not t.is_gpu()
    assert Target._RVV in t and Target._RISCV in t
    assert Target._X86 not in t and Target._ARM not in t


@pytest.mark.parametrize(
    "isa, expected",
    [
        (K1_ISA, True),
        ("rv64gcv", True),
        ("rv64imafdcv_zvl256b", True),
        ("rv64imafdc_zve64d", False),  # embedded subset only, no full `v`
        ("rv64gc_xtheadvector", False),  # RVV 0.7 draft, not supported
        ("rv64gc", False),
    ],
)
def test_isa_detection(isa, expected):
    cpuinfo = f"processor\t: 0\nhart\t\t: 0\nisa\t\t: {isa}\nmmu\t\t: sv39\n"
    assert _riscv_has_rvv(cpuinfo) == expected


def test_isa_detection_without_isa_line():
    assert not _riscv_has_rvv("processor\t: 0\n")


@pytest.mark.parametrize(
    "dtype, lanes", [(Fp(64), 4), (Fp(32), 8), (SInt(64), 4), (SInt(32), 8), (SInt(16), 16)]
)
def test_default_lanes_like_avx(dtype, lanes):
    #   VLEN = 256 is hardcoded: one full register, exactly like AVX
    assert Target.RISCV_RVV.default_vector_lanes(dtype) == lanes
    assert Target.X86_AVX.default_vector_lanes(dtype) == lanes


@pytest.mark.parametrize("info_cls", [GccInfo, ClangInfo])
def test_jit_flags(info_cls):
    flags = info_cls(target=Target.RISCV_RVV).cxxflags()
    assert "-march=rv64gcv_zvl256b" in flags
    #   -mrvv-vector-bits would make the compiler assume VLEN == 256 exactly; we only
    #   assume VLEN >= 256.
    assert not any(f.startswith("-mrvv-vector-bits") for f in flags)


def test_current_cpu_march_on_riscv(monkeypatch):
    import platform

    monkeypatch.setattr(platform, "machine", lambda: "riscv64")

    monkeypatch.setattr(Target, "available_vector_cpu_targets", lambda: (Target.RISCV_RVV,))
    assert compiler_info._current_cpu_march() == ["-march=rv64gcv_zvl256b"]

    monkeypatch.setattr(Target, "available_vector_cpu_targets", lambda: ())
    assert compiler_info._current_cpu_march() == ["-march=rv64gc"]


def test_current_cpu_march_elsewhere(monkeypatch):
    import platform

    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert compiler_info._current_cpu_march() == ["-march=native"]
