from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ...codegen.target import Target


def _synthesize_riscv_march(isa: str | None) -> str | None:
    """Build a GCC/Clang ``-march`` string from a RISC-V ``/proc/cpuinfo`` ISA string.

    GCC does not accept ``-march=native`` on RISC-V, so we reconstruct the flag from
    the reported ISA string. Two classes of multi-letter tokens are dropped, since they
    do not affect the instructions the compiler emits for a compute kernel and are not
    universally recognized by every toolchain version:

    - vendor-specific extensions (prefix ``x``, e.g. ``xtheadvector``); and
    - supervisor / hypervisor extensions (prefix ``s`` or ``h``, e.g. ``svadu``,
      ``svpbmt``, ``sstc``) -- these control the MMU/privileged behaviour. Notably
      QEMU advertises ``svadu`` in its emulated ISA string, which GCC 14 rejects.

    Examples:
        >>> _synthesize_riscv_march("rv64imafdcv_zicsr_zvl256b")
        'rv64imafdcv_zicsr_zvl256b'
        >>> _synthesize_riscv_march("rv64gc_xtheadvector_zba")
        'rv64gc_zba'
        >>> _synthesize_riscv_march("rv64imafdcv_zve64d_sstc_svadu")
        'rv64imafdcv_zve64d'
        >>> _synthesize_riscv_march(None) is None
        True
    """
    if not isa:
        return None

    tokens = isa.strip().lower().split("_")
    if not tokens or not tokens[0].startswith(("rv32", "rv64", "rv128")):
        return None

    kept = [tokens[0]] + [
        t for t in tokens[1:] if t and not t.startswith(("x", "s", "h"))
    ]
    return "_".join(kept)


@dataclass
class CompilerInfo(ABC):
    """Base class for compiler infos."""

    openmp: bool = True
    """Enable/disable OpenMP compilation"""

    optlevel: str | None = "fast"
    """Compiler optimization level"""

    cxx_standard: str = "c++17"
    """C++ language standard to be compiled with"""

    target: Target = Target.CurrentCPU
    """Hardware target to compile for.

    The value of ``target`` is used to set the ``-march`` compiler
    option (or equivalent).
    `Target.CurrentCPU` translates to ``-march=native`` on x86/ARM, and to an
    ``-march=`` string synthesized from ``/proc/cpuinfo`` on RISC-V (where
    ``-march=native`` is not accepted by GCC).
    """

    rvv_vlen: int | None = None
    """Assumed/minimum ``VLEN`` (bits) for a `Target.RISCV_RVV` build.

    When set, ``_zvl{vlen}b`` is appended to the ``-march`` string. In fixed-length
    (VLS) mode (see `rvv_fixed_length`) this additionally enables
    ``-mrvv-vector-bits=zvl``, letting the compiler treat the scalable vector types as
    fixed-size and optimize more aggressively.
    """

    rvv_fixed_length: bool = False
    """If `True` and building for `Target.RISCV_RVV` with `rvv_vlen` set, pass
    ``-mrvv-vector-bits=zvl`` (VLS mode only; must not be used for VLA kernels)."""

    extra_cxxflags: list[str] = field(default_factory=list)

    @abstractmethod
    def cxx(self) -> str:
        """Path to the executable of this compiler"""

    @abstractmethod
    def cxxflags(self) -> list[str]:
        """Compiler flags affecting C++ compilation"""

    @abstractmethod
    def linker_flags(self) -> list[str]:
        """Flags affecting linkage of the extension module"""

    @abstractmethod
    def include_flags(self, include_dirs: Sequence[str]) -> list[str]:
        """Convert a list of include directories into corresponding compiler flags"""

    @abstractmethod
    def restrict_qualifier(self) -> str:
        """*restrict* memory qualifier recognized by this compiler"""

    @staticmethod
    def get_default(**kwargs) -> CompilerInfo:
        """Create a default compiler info object for the current runtime environment.

        Args:
            kwargs: Are forwarded to the constructor of the selected `CompilerInfo` subclass.
        """
        return CompilerInfo.get_available_compilers()[0](**kwargs)

    @staticmethod
    def get_available_compilers() -> Sequence[type[CompilerInfo]]:
        def _test_compiler(cmd):
            try:
                if subprocess.run(cmd, capture_output=True).returncode == 0:
                    return True
            except FileNotFoundError:
                pass
            return False

        import platform

        sysname = platform.system()

        compilers: list[type[CompilerInfo]] = []

        match sysname.lower():
            case "linux":
                if _test_compiler(["g++", "--version"]):
                    compilers.append(GccInfo)
                if _test_compiler(["clang++", "--version"]):
                    compilers.append(ClangInfo)
            case "darwin":
                if _test_compiler(["clang++", "--version"]):
                    compilers.append(AppleClangInfo)
            case _:
                raise RuntimeError(
                    f"Cannot determine compiler information for platform {sysname}"
                )

        return tuple(compilers)


class _GnuLikeCliCompiler(CompilerInfo):
    def cxxflags(self) -> list[str]:
        flags = ["-DNDEBUG", f"-std={self.cxx_standard}", "-fPIC"]

        if self.optlevel is not None:
            flags.append(f"-O{self.optlevel}")

        if self.openmp:
            flags.append("-fopenmp")

        match self.target:
            case Target.CurrentCPU:
                flags += self._current_cpu_march_flags()
            case Target.X86_SSE:
                flags += ["-march=x86-64-v2"]
            case Target.X86_AVX:
                flags += ["-march=x86-64-v3"]
            case Target.X86_AVX512:
                flags += ["-march=x86-64-v4"]
            case Target.X86_AVX512_FP16:
                flags += ["-march=x86-64-v4", "-mavx512fp16"]
            case Target.ARM_NEON:
                flags += ["-march=armv8-a"]
            case Target.ARM_NEON_FP16:
                flags += ["-march=armv8.2-a+fp16"]
            case Target.ARM_SVE:
                flags += ["-march=armv8.2-a+sve"]
            case Target.RISCV_RVV:
                flags += self._riscv_rvv_march_flags()

        return flags + self.extra_cxxflags

    def _current_cpu_march_flags(self) -> list[str]:
        """``-march`` flags for `Target.CurrentCPU`.

        On x86/ARM this is ``-march=native``. On RISC-V, GCC rejects
        ``-march=native`` (`ISA string must begin with rv32 or rv64`), so we synthesize
        an explicit ``-march=`` string from ``/proc/cpuinfo``, dropping vendor-specific
        (``x*``) extension tokens the compiler may not recognize.
        """
        import platform

        if platform.machine() in ("riscv64", "riscv"):
            from ...codegen.target import _query_riscv_isa_string

            march = _synthesize_riscv_march(_query_riscv_isa_string())
            if march is not None:
                return [f"-march={march}"]
            #   Fall back to a conservative RVV-capable baseline
            return ["-march=rv64gcv"]

        return ["-march=native"]

    def _riscv_rvv_march_flags(self) -> list[str]:
        """``-march`` (and vector-bits) flags for an explicit `Target.RISCV_RVV` build."""
        march = "rv64gcv"
        if self.rvv_vlen is not None:
            march += f"_zvl{self.rvv_vlen}b"
        flags = [f"-march={march}"]
        if self.rvv_fixed_length and self.rvv_vlen is not None:
            flags.append("-mrvv-vector-bits=zvl")
        return flags

    def linker_flags(self) -> list[str]:
        return ["-shared"]

    def include_flags(self, include_dirs: Sequence[str]) -> list[str]:
        return [f"-I{d}" for d in include_dirs]

    def restrict_qualifier(self) -> str:
        return "__restrict__"


class GccInfo(_GnuLikeCliCompiler):
    """Compiler info for the GNU Compiler Collection C++ compiler (``g++``)."""

    def cxx(self) -> str:
        return "g++"


@dataclass
class ClangInfo(_GnuLikeCliCompiler):
    """Compiler info for the LLVM C++ compiler (``clang``)."""

    def cxx(self) -> str:
        return "clang++"

    def cxxflags(self):
        flags = super().cxxflags()

        if self.optlevel == "fast":
            #   clang deprecates -Ofast
            flags.remove("-Ofast")
            flags += ["-O3", "-ffast-math", "-fhonor-infinities"]

        return flags


@dataclass
class AppleClangInfo(ClangInfo):
    """Compiler info for the Apple Clang compiler."""

    def cxxflags(self) -> list[str]:
        flags = super().cxxflags()

        if self.openmp:
            #   AppleClang requires the `-Xclang -fopenmp` in exactly that order for OpenMP to work
            flags.remove("-fopenmp")
            flags += ["-Xclang", "-fopenmp"]

        return flags

    def linker_flags(self):
        ldflags = super().linker_flags()

        #   Link against libpython
        import sysconfig

        libpython_file = Path(sysconfig.get_config_var("LIBRARY")).with_suffix(".dylib")
        libpython_dir = Path(sysconfig.get_config_var("LIBDIR"))
        libpython = libpython_dir / libpython_file

        ldflags += [str(libpython)]

        #   Find an appropriate OpenMP dylib to link against
        omp_candidates = [
            Path("/opt/local/lib/libomp/libomp.dylib"),
            Path("/usr/local/lib/libomp.dylib"),
            Path("/opt/homebrew/lib/libomp.dylib"),
        ]
        for omplib in omp_candidates:
            if omplib.exists():
                ldflags.append(str(omplib))
                break

        return ldflags
