from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ...codegen.target import Target

#   The RVV backend hardcodes VLEN = 256 (SpacemiT K1). `zvl256b` tells the compiler
#   the same thing (VLEN >= 256), and is what `pystencils_runtime/rvv.hpp` checks for.
_RISCV_RVV_MARCH = "-march=rv64gcv_zvl256b"


def _current_cpu_march() -> list[str]:
    """``-march`` flag for `Target.CurrentCPU`.

    GCC (at least up to 14) rejects ``-march=native`` on RISC-V, so on riscv64 hosts
    we resolve the current CPU ourselves: the RVV flags if the V extension is present,
    plain ``rv64gc`` otherwise. All other hosts use ``-march=native``.
    """
    import platform

    if platform.machine() == "riscv64":
        if Target.RISCV_RVV in Target.available_vector_cpu_targets():
            return [_RISCV_RVV_MARCH]
        return ["-march=rv64gc"]

    return ["-march=native"]


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
    `Target.CurrentCPU` translates to ``-march=native``
    (on RISC-V hosts, to an explicit ``-march`` string; see `_current_cpu_march`).
    """

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
                flags += _current_cpu_march()
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
                flags += [_RISCV_RVV_MARCH]

        return flags + self.extra_cxxflags

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
