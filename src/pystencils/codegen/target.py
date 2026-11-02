from __future__ import annotations

from enum import Flag, auto
from functools import cache
from warnings import warn

from ..types import PsScalarType
from .errors import CodegenError


class Target(Flag):
    """
    The Target enumeration represents all possible targets that can be used for code generation.
    """

    #   ------------------ Component Flags - Do Not Use Directly! -------------------------------------------

    _CPU = auto()

    _VECTOR = auto()

    _X86 = auto()
    _SSE = auto()
    _AVX = auto()
    _AVX512 = auto()
    _VL = auto()
    _FP16 = auto()

    _ARM = auto()
    _NEON = auto()
    _SVE = auto()

    _RISCV = auto()
    _RVV = auto()

    _GPU = auto()

    _CUDA = auto()
    _HIP = auto()

    _SYCL = auto()

    _AUTOMATIC = auto()

    #   ------------------ Actual Targets -------------------------------------------------------------------

    CurrentCPU = _CPU | _AUTOMATIC
    """
    Auto-best CPU target.

    `CurrentCPU` causes the code generator to automatically select a CPU target according to CPUs found
    on the current machine and runtime environment.
    """

    GenericCPU = _CPU
    """Generic CPU target.

    Generate the kernel for a generic multicore CPU architecture. This opens up all architecture-independent
    optimizations including OpenMP, but no vectorization.
    """

    CPU = GenericCPU
    """Alias for backward-compatibility"""

    X86_SSE = _CPU | _VECTOR | _X86 | _SSE
    """x86 architecture with SSE vector extensions (architecture level x86-64-v2)."""

    X86_AVX = _CPU | _VECTOR | _X86 | _AVX
    """x86 architecture with AVX vector extensions (architecture level x86-64-v3)."""

    X86_AVX512 = _CPU | _VECTOR | _X86 | _AVX512
    """x86 architecture with AVX512 vector extensions (architecture level x86-64-v4)."""

    X86_AVX512_FP16 = _CPU | _VECTOR | _X86 | _AVX512 | _FP16
    """x86 architecture with AVX512 vector extensions (architecture level x86-64-v4) and fp16-support."""

    ARM_NEON = _CPU | _VECTOR | _ARM | _NEON
    """64-bit ARM architecture with NEON vector instructions"""

    ARM_NEON_FP16 = _CPU | _VECTOR | _ARM | _NEON | _FP16
    """64-bit ARM architecture with NEON vector instructions and half-precision support"""

    ARM_SVE = _CPU | _VECTOR | _ARM | _SVE
    """ARM architecture with SVE vector extensions"""

    RISCV_RVV = _CPU | _VECTOR | _RISCV | _RVV
    """RISC-V architecture with the ratified V vector extension (RVV 1.0).

    Treated like a fixed-width target (cf. `X86_AVX`): the code generator assumes a
    hardware vector register width of exactly ``VLEN = 256`` bits (SpacemiT K1) and
    emits vector-length-specific (VLS) code. Kernels require hardware with VLEN >= 256.
    """

    CurrentGPU = _GPU | _AUTOMATIC
    """Auto-best GPU target.

    `CurrentGPU` causes the code generator to automatically select a GPU target according to GPU devices
    found on the current machine and runtime environment.
    """

    CUDA = _GPU | _CUDA
    """Generic CUDA GPU target.

    Generate a CUDA kernel for a generic Nvidia GPU.
    """

    HIP = _GPU | _HIP
    """Generic HIP GPU target.

    Generate a HIP kernel for generic AMD or NVidia GPUs.
    """

    GPU = CurrentGPU
    """Alias for `Target.CurrentGPU`, for backward compatibility."""

    SYCL = _SYCL
    """SYCL kernel target.

    Generate a function to be called within a SYCL parallel command.

    ..  note::
        The SYCL target is experimental and not thoroughly tested yet.
    """

    def is_automatic(self) -> bool:
        """Determine if this target is a proxy target that is automatically resolved
        according to the runtime environment."""
        return Target._AUTOMATIC in self

    def is_cpu(self) -> bool:
        """Determine if this target is a CPU target."""
        return Target._CPU in self

    def is_vector_cpu(self) -> bool:
        """Determine if this target is a vector CPU target."""
        return self.is_cpu() and Target._VECTOR in self

    def is_gpu(self) -> bool:
        """Determine if this target is a GPU target.

        This refers to targets for the CUDA and HIP family of platforms.
        `Target.SYCL` is *not* a GPU target.
        """
        return Target._GPU in self

    def default_vector_lanes(self, dtype: PsScalarType) -> int:
        if not self.is_vector_cpu():
            raise ValueError("This target is no vector CPU target.")

        assert dtype.itemsize is not None

        match self:
            case Target.X86_SSE | Target.ARM_NEON | Target.ARM_NEON_FP16:
                return 128 // (dtype.itemsize * 8)
            case Target.X86_AVX | Target.RISCV_RVV:
                #   RVV: VLEN is hardcoded to 256 bits, so lanes == one full register (LMUL=1)
                return 256 // (dtype.itemsize * 8)
            case Target.X86_AVX512 | Target.X86_AVX512_FP16:
                return 512 // (dtype.itemsize * 8)
            case Target.ARM_SVE:
                raise CodegenError(
                    "Number of lanes for scalable vectors of the SVE architecture must be specified manually"
                )
            case _:
                raise NotImplementedError(
                    f"No default number of lanes known for {dtype} on {self}"
                )

    @staticmethod
    def auto_cpu() -> Target:
        """Return the most capable vector CPU target available on the current machine."""
        avail_targets = _available_vector_targets()
        if avail_targets:
            return avail_targets[-1]
        else:
            return Target.GenericCPU

    @staticmethod
    def auto_gpu() -> Target:
        """Return the GPU target available in the current runtime environment.

        Raises:
            RuntimeError: If `cupy` is not installed and therefore no GPU runtime is available.
        """
        try:
            import cupy

            if cupy.cuda.runtime.is_hip:
                return Target.HIP
            else:
                return Target.CUDA
        except ImportError:
            raise RuntimeError("Cannot infer GPU target since cupy is not installed.")

    @staticmethod
    def available_targets() -> tuple[Target, ...]:
        """List available"""
        targets = [Target.GenericCPU]
        try:
            import cupy  # noqa: F401

            targets.append(Target.auto_gpu())
        except ImportError:
            pass
        try:
            import dpctl  # noqa: F401

            targets.append(Target.SYCL)
        except ImportError:
            pass

        targets += Target.available_vector_cpu_targets()
        return tuple(targets)

    @staticmethod
    def available_vector_cpu_targets() -> tuple[Target, ...]:
        """Returns a list of available vector CPU targets, ordered from least to most capable."""
        return _available_vector_targets()


@cache
def _available_vector_targets() -> tuple[Target, ...]:
    """Returns available vector targets, sorted from leat to most capable."""

    targets: list[Target] = []

    import platform

    machine_spec = platform.machine()

    if machine_spec in ["x86_64", "x86", "AMD64", "i386"]:
        try:
            from cpuinfo import get_cpu_info
        except ImportError:
            warn(
                "Unable to determine available x86 vector CPU targets for this system: "
                "py-cpuinfo is not available.",
                UserWarning,
            )
            return ()

        flags = set(get_cpu_info()["flags"])

        if {"sse", "sse2", "ssse3", "sse4_1", "sse4_2"} < flags:
            targets.append(Target.X86_SSE)

        if {"avx", "avx2"} < flags:
            targets.append(Target.X86_AVX)

        if {"avx512f"} < flags:
            targets.append(Target.X86_AVX512)

        if {"avx512_fp16"} < flags:
            targets.append(Target.X86_AVX512_FP16)
    elif machine_spec in ["arm64", "ARM64", "aarch64"]:
        targets.append(Target.ARM_NEON)

        if platform.system() == "Darwin":
            if _darwin_query_cpu_feature(b"hw.optional.arm.FEAT_FP16"):
                targets.append(Target.ARM_NEON_FP16)

        elif platform.system() == "Linux":
            try:
                from cpuinfo import get_cpu_info

                flags = set(get_cpu_info()["flags"])

                if "fphp" in flags:
                    targets.append(Target.ARM_NEON_FP16)

                if "sve" in flags:
                    targets.append(Target.ARM_SVE)
            except ImportError:
                warn(
                    "Unable to determine available ARM vector CPU features for this system: "
                    "py-cpuinfo is not available.",
                    UserWarning,
                )
                return ()
        else:
            warn(
                f"Unable to determine ARM architecture features on system {platform.system()}"
            )
    elif machine_spec == "riscv64":
        if _riscv_has_rvv():
            targets.append(Target.RISCV_RVV)
    else:
        warn(
            "Unable to determine available vector CPU targets for this system: "
            f"unknown platform {platform.machine()}.",
            UserWarning,
        )

    return tuple(targets)


def _riscv_has_rvv(cpuinfo: str | None = None) -> bool:
    """Check whether this RISC-V machine implements the V extension (RVV 1.0).

    Linux reports the ISA string in ``/proc/cpuinfo`` (``isa : rv64imafdcv_zicsr_...``).
    RVV 1.0 shows up as the single-letter extension ``v`` in the leading part before
    the first ``_``. Multi-letter tokens such as ``_zve64d`` or ``_xtheadvector``
    (the RVV 0.7 draft) must not count, hence we only look at that leading part.

    Args:
        cpuinfo: Text of ``/proc/cpuinfo``; read from the system if `None` (testing hook).
    """
    if cpuinfo is None:
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
        except OSError:
            return False

    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "isa":
            base = value.strip().lower().split("_")[0]  # e.g. "rv64imafdcv"
            return base.startswith("rv") and "v" in base[4:]  # skip "rv64"/"rv32"

    return False


def _darwin_query_cpu_feature(feature: bytes) -> bool:
    """See https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics"""  # noqa: E501
    from ctypes import CDLL, c_int, c_size_t, sizeof, byref

    libc = CDLL("/usr/lib/libc.dylib")
    value = c_int(0)
    size = c_size_t(sizeof(value))
    status = libc.sysctlbyname(feature, byref(value), byref(size), None, 0)
    if status == 0:
        return value.value == 1
    else:
        raise RuntimeError(
            f"Error while retrieving CPU features. `sysctlbyname` returned {status}"
        )
