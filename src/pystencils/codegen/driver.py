from __future__ import annotations
from typing import (
    cast,
    Sequence,
    Callable,
    TYPE_CHECKING,
    Protocol,
    TypeAlias,
)
from dataclasses import replace
from abc import abstractmethod
from warnings import warn

from sympy.codegen.ast import AssignmentBase

from ..field import Field
from ..simp import AssignmentCollection
from ..types import PsIntegerType, PsScalarType
from ..flow.flowgraph import FlowgraphNode
from ..flow import Flowgraph, tie as tie_flowgraph

from ..backend.memory import PsSymbol
from ..backend.ast import PsAstNode
from ..backend.ast.analysis import collect_required_headers, collect_undefined_symbols
from ..backend.ast.expressions import PsExpression
from ..backend.ast.structural import PsBlock
from ..backend.kernelcreation import (
    KernelCreationContext,
    KernelAnalysis,
    FreezeExpressions,
    Typifier,
    FreezeFlowgraph,
    AstFactory,
)
from ..backend.kernelcreation.iteration_space import (
    create_sparse_iteration_space,
    create_full_iteration_space,
    infer_ghost_layers,
    IterationSpace,
)
from ..backend.platforms import (
    Platform,
    GenericCpu,
    GenericGpu,
    GenericVectorCpu,
)

from ..backend.transformations import (
    CanonicalizeSymbols,
    EliminateConstants,
    HoistIterationInvariantDeclarations,
    LowerToC,
    MaterializeAxes,
    ReductionsToMemory,
    SelectFunctions,
)

from .config import (
    _AUTO_TYPE,
    AUTO,
    CreateKernelConfig,
    GhostLayerSpec,
    GpuIndexingScheme,
    GpuOptions,
    IterationSliceSpec,
)
from .cpu_loop_strategies import DefaultCpuLoopStrategies
from .functions import Lambda
from .gpu_indexing import GpuIndexing, GpuIndexMappingStrategy, GpuLaunchConfiguration
from .kernel import GpuKernel, Kernel
from .parameters import Parameter
from .properties import FieldBasePtr, PsSymbolProperty
from .sycl_indexing import SyclIndexing
from .target import Target

if TYPE_CHECKING:
    from ..jit import JitBase


SymbolicKernel: TypeAlias = (
    AssignmentCollection
    | Sequence[AssignmentBase]
    | AssignmentBase
    | FlowgraphNode
    | Flowgraph
)
"""Various forms of symbolic kernels that can be consumed by the code generator."""


class ProvidesLaunchConfigFactory(Protocol):

    @abstractmethod
    def get_launch_config_factory(self) -> Callable[[], GpuLaunchConfiguration]: ...


def create_kernel(
    assignments: SymbolicKernel,
    config: CreateKernelConfig | None = None,
    **kwargs,
) -> Kernel:
    """Create a kernel function from a set of assignments.

    Args:
        assignments: The kernel's sequence of assignments, expressed using SymPy
        config: The configuration for the kernel translator
        kwargs: If ``config`` is not set, it is created from the keyword arguments;
            if it is set, its option will be overridden by any keyword arguments.

    Returns:
        The numerical kernel in pystencil's internal representation, ready to be
        exported or compiled
    """

    if not config:
        config = CreateKernelConfig()

    if kwargs:
        config = replace(config, **kwargs)

    driver = DefaultKernelCreationDriver(config)
    return driver(assignments)


def get_driver(cfg: CreateKernelConfig) -> DefaultKernelCreationDriver:
    """Create a code generation driver object from the given configuration.

    Args:
        cfg: Configuration for the code generator
    """
    return DefaultKernelCreationDriver(cfg)


class AxesFactory(Protocol):
    def create_axes(self, body: PsBlock, ispace: IterationSpace) -> PsBlock: ...


class DefaultKernelCreationDriver:
    """Drives the default kernel creation sequence.

    Args:
        cfg: Configuration for the code generator
    """

    def __init__(self, cfg: CreateKernelConfig):
        self._cfg = cfg

        #   Data Type Options
        idx_dtype: PsIntegerType = cfg.get_option("index_dtype")
        default_dtype: PsScalarType = cfg.get_option("default_dtype")

        #   Iteration Space Options
        num_ispace_options_set = (
            int(cfg.is_option_set("ghost_layers"))
            + int(cfg.is_option_set("iteration_slice"))
            + int(cfg.is_option_set("index_field"))
        )

        if num_ispace_options_set > 1:
            raise ValueError(
                "At most one of the options 'ghost_layers' 'iteration_slice' and 'index_field' may be set."
            )

        self._ghost_layers: GhostLayerSpec | None = cfg.get_option("ghost_layers")
        self._iteration_slice: IterationSliceSpec | None = cfg.get_option(
            "iteration_slice"
        )
        self._index_field: Field | None = cfg.get_option("index_field")

        if num_ispace_options_set == 0:
            self._ghost_layers = AUTO

        #   Create the context
        self._ctx = KernelCreationContext(
            default_dtype=default_dtype,
            index_dtype=idx_dtype,
        )

        self._target = cfg.get_target()
        self._gpu_indexing: ProvidesLaunchConfigFactory | None = (
            self._get_gpu_indexing()
        )
        self._platform = self._get_platform()
        self._factory = AstFactory(self._ctx)

    def __call__(
        self,
        symbolic_kernel: SymbolicKernel,
    ) -> Kernel:
        kernel_body = self.parse_kernel_body(symbolic_kernel)
        kernel_ast = self._materialize_iteration_space(kernel_body)
        kernel_ast = self._general_optimize(kernel_ast)
        kernel_ast = self._lowering(kernel_ast)
        return self._finalize(kernel_ast)

    def parse_kernel_body(
        self,
        symbolic_kernel: SymbolicKernel,
    ):
        if isinstance(symbolic_kernel, FlowgraphNode | Flowgraph):
            kernel_body = self.parse_flowgraph(symbolic_kernel)
        else:
            kernel_body = self.parse_assignment_collection(symbolic_kernel)

        return kernel_body

    def parse_assignment_collection(
        self,
        assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    ) -> PsBlock:
        if isinstance(assignments, AssignmentBase):
            assignments = [assignments]

        if not isinstance(assignments, AssignmentCollection):
            assignments = AssignmentCollection(assignments)  # type: ignore

        _ = _parse_simplification_hints(assignments)

        analysis = KernelAnalysis(
            self._ctx,
            not self._cfg.skip_independence_check,
            not self._cfg.allow_double_writes,
        )
        analysis(assignments)

        self._create_iteration_space(assignments)

        freeze = FreezeExpressions(self._ctx)
        kernel_body = freeze(assignments)

        typify = Typifier(self._ctx)
        kernel_body = typify(kernel_body)

        return kernel_body

    def parse_flowgraph(self, flowgraph: FlowgraphNode | Flowgraph):
        if not isinstance(flowgraph, Flowgraph):
            flowgraph = tie_flowgraph(flowgraph)

        fields = flowgraph.fields_read | flowgraph.fields_written
        for f in fields:
            const = f not in flowgraph.fields_written
            self._ctx.add_field(f, const=const)

        self._create_iteration_space(flowgraph)

        freeze = FreezeFlowgraph(self._ctx)
        kernel_body = freeze(flowgraph)

        return kernel_body

    def _create_iteration_space(
        self, symbolic_kernel: AssignmentCollection | Flowgraph
    ):
        if self._index_field is not None:
            ispace = create_sparse_iteration_space(
                self._ctx, index_field=self._cfg.index_field
            )
        else:
            gls: GhostLayerSpec | None
            if self._ghost_layers == AUTO:
                gls = infer_ghost_layers(symbolic_kernel)
            else:
                assert not isinstance(self._ghost_layers, _AUTO_TYPE)
                gls = self._ghost_layers

            ispace = create_full_iteration_space(
                self._ctx,
                ghost_layers=gls,
                iteration_slice=self._iteration_slice,
            )

        self._ctx.set_iteration_space(ispace)

    def _materialize_iteration_space(self, kernel_body: PsBlock) -> PsBlock:
        match self._platform:
            case GenericCpu() | GenericGpu():
                axes_factory = self._get_axes_factory()

                kernel_ast = axes_factory.create_axes(
                    kernel_body, self._ctx.get_iteration_space()
                )

                materialize_axes = MaterializeAxes(self._ctx)
                kernel_ast = materialize_axes(kernel_ast)

                r_to_mem = ReductionsToMemory(
                    self._ctx, self._ctx.reduction_data.values()
                )
                kernel_ast = r_to_mem(kernel_ast)

                return kernel_ast
            case _:
                return self._platform.materialize_iteration_space(
                    kernel_body, self._ctx.get_iteration_space()
                )

    def _general_optimize(self, kernel_ast: PsBlock) -> PsBlock:
        canonicalize = CanonicalizeSymbols(self._ctx, True)
        kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

        elim_constants = EliminateConstants(self._ctx, extract_constant_exprs=True)
        kernel_ast = cast(PsBlock, elim_constants(kernel_ast))

        hoist_invariants = HoistIterationInvariantDeclarations(self._ctx)
        kernel_ast = cast(PsBlock, hoist_invariants(kernel_ast))

        return kernel_ast

    def _lowering(self, kernel_ast: PsBlock) -> PsBlock:
        if isinstance(self._platform, GenericVectorCpu):
            select_intrin = self._platform.get_intrinsic_selector()
            kernel_ast = cast(PsBlock, select_intrin(kernel_ast))

        lower_to_c = LowerToC(self._ctx)
        kernel_ast = cast(PsBlock, lower_to_c(kernel_ast))

        select_functions = SelectFunctions(self._platform)
        kernel_ast = cast(PsBlock, select_functions(kernel_ast))

        return kernel_ast

    def _finalize(self, kernel_ast: PsBlock) -> Kernel:
        #   Late canonicalization pass: Canonicalize new symbols introduced by LowerToC

        canonicalize = CanonicalizeSymbols(self._ctx, True)
        kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

        kernel_factory = KernelFactory(self._ctx)

        range_kernel = self._target == Target.SYCL and self._cfg.sycl.get_option(
            "automatic_block_size"
        )
        nd_range_kernel = self._target == Target.SYCL and not self._cfg.sycl.get_option(
            "automatic_block_size"
        )

        if self._target.is_cpu() or range_kernel:
            return kernel_factory.create_generic_kernel(
                self._platform,
                kernel_ast,
                self._cfg.get_option("function_name"),
                self._target,
                self._cfg.get_jit(),
            )
        elif self._target.is_gpu() or nd_range_kernel:
            assert self._gpu_indexing is not None

            return kernel_factory.create_gpu_kernel(
                self._platform,
                kernel_ast,
                self._cfg.get_option("function_name"),
                self._target,
                self._cfg.get_jit(),
                self._gpu_indexing.get_launch_config_factory(),
            )
        else:
            assert False, "unexpected target"

    def _get_gpu_indexing(self) -> ProvidesLaunchConfigFactory | None:
        nd_range_kernel = self._target == Target.SYCL and not self._cfg.sycl.get_option(
            "automatic_block_size"
        )
        if not self._target.is_gpu() and not nd_range_kernel:
            return None

        idx_scheme: GpuIndexingScheme = self._cfg.gpu.get_option("indexing_scheme")
        manual_launch_grid: bool = self._cfg.gpu.get_option("manual_launch_grid")
        assume_warp_aligned_block_size: bool = self._cfg.gpu.get_option(
            "assume_warp_aligned_block_size"
        )
        warp_size: int | None = self._cfg.gpu.get_option("warp_size")

        if warp_size is None:
            warp_size = GpuOptions.default_warp_size(self._target)

        if warp_size is None and assume_warp_aligned_block_size:
            warn(
                "GPU warp size is unknown - ignoring assumption `assume_warp_aligned_block_size`."
            )

        if nd_range_kernel:
            return SyclIndexing(
                self._ctx,
                idx_scheme,
                warp_size,
                manual_launch_grid,
                assume_warp_aligned_block_size,
            )

        return GpuIndexing(
            self._ctx,
            self._target,
            idx_scheme,
            warp_size,
            manual_launch_grid,
            assume_warp_aligned_block_size,
        )

    def _get_platform(self) -> Platform:
        if Target._CPU in self._target:
            if Target._X86 in self._target:
                from ..backend.platforms.x86 import X86VectorArch, X86VectorCpu

                arch: X86VectorArch

                if Target._SSE in self._target:
                    arch = X86VectorArch.SSE
                elif Target._AVX in self._target:
                    arch = X86VectorArch.AVX
                elif Target._AVX512 in self._target:
                    if Target._FP16 in self._target:
                        arch = X86VectorArch.AVX512_FP16
                    else:
                        arch = X86VectorArch.AVX512
                else:
                    assert False, "unreachable code"

                return X86VectorCpu(self._ctx, arch)
            elif Target._NEON in self._target:
                from ..backend.platforms.neon import NeonCpu

                return NeonCpu(self._ctx, enable_fp16=Target._FP16 in self._target)
            elif self._target == Target.ARM_SVE:
                from ..backend.platforms.sve import SveCpu

                return SveCpu(self._ctx)
            elif Target._RVV in self._target:
                from ..backend.platforms.rvv import RvvCpu
                from .config import VectorizationMode

                vlen: int | None = self._cfg.cpu.rvv.get_option("vlen")
                mode: VectorizationMode = self._cfg.cpu.vectorize.get_option("mode")
                check_vlen: bool = self._cfg.cpu.rvv.get_option("check_vlen")
                lanes: int | None = self._cfg.cpu.vectorize.get_option("lanes")

                if vlen is None:
                    #   Derive a sensible default: in FIXED mode with an explicit lane
                    #   count, pick the VLEN that yields LMUL=1 for the default dtype
                    #   (the recommended register-pressure sweet spot, plan section 6).
                    #   Otherwise fall back to the RVV minimum VLEN of 128 bits.
                    if mode == VectorizationMode.FIXED and lanes is not None:
                        vlen = lanes * self._ctx.default_dtype.width
                    else:
                        vlen = 128

                return RvvCpu(
                    self._ctx, vlen=vlen, mode=mode, check_vlen=check_vlen
                )
            elif self._target == Target.GenericCPU:
                return GenericCpu(self._ctx)
            else:
                raise NotImplementedError(
                    f"No platform is currently available for CPU target {self._target}"
                )

        elif self._target.is_gpu():
            assume_warp_aligned_block_size: bool = self._cfg.gpu.get_option(
                "assume_warp_aligned_block_size"
            )
            warp_size: int | None = self._cfg.gpu.get_option("warp_size")

            GpuPlatform: type
            match self._target:
                case Target.CUDA:
                    from ..backend.platforms import CudaPlatform as GpuPlatform
                case Target.HIP:
                    from ..backend.platforms import HipPlatform as GpuPlatform
                case _:
                    assert False, f"unexpected GPU target: {self._target}"

            return GpuPlatform(
                self._ctx,
                assume_warp_aligned_block_size=assume_warp_aligned_block_size,
                warp_size=warp_size,
            )

        elif self._target == Target.SYCL:
            from ..backend.platforms import SyclPlatform

            auto_block_size: bool = self._cfg.sycl.get_option("automatic_block_size")

            return SyclPlatform(
                self._ctx,
                automatic_block_size=auto_block_size,
            )

        raise NotImplementedError(
            f"Code generation for target {self._target} not implemented"
        )

    def _get_axes_factory(self) -> AxesFactory:
        match self._platform:
            case GenericCpu():
                return DefaultCpuLoopStrategies(self._ctx, self._target, self._cfg.cpu)
            case GenericGpu():
                return GpuIndexMappingStrategy(self._ctx, self._cfg.gpu)
            case _:
                raise NotImplementedError(
                    f"No axis builder available for platform of type {type(self._platform)}"
                )


class KernelFactory:
    """Factory for wrapping up backend and IR objects into exportable kernels and function objects."""

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def create_lambda(self, expr: PsExpression) -> Lambda:
        """Create a Lambda from an expression."""
        params = self._get_function_params(expr)
        return Lambda(expr, params)

    def create_generic_kernel(
        self,
        platform: Platform,
        body: PsBlock,
        function_name: str,
        target_spec: Target,
        jit: JitBase,
    ) -> Kernel:
        """Create a kernel for a generic target"""
        params = self._get_function_params(body)
        req_headers = self._get_headers(platform, body)

        kfunc = Kernel(body, target_spec, function_name, params, req_headers, jit)
        kfunc.metadata.update(self._ctx.metadata)
        return kfunc

    def create_gpu_kernel(
        self,
        platform: Platform,
        body: PsBlock,
        function_name: str,
        target_spec: Target,
        jit: JitBase,
        launch_config_factory: Callable[[], GpuLaunchConfiguration],
    ) -> GpuKernel:
        """Create a kernel for a GPU target"""
        params = self._get_function_params(body)
        req_headers = self._get_headers(platform, body)

        kfunc = GpuKernel(
            body,
            target_spec,
            function_name,
            params,
            req_headers,
            jit,
            launch_config_factory,
        )
        kfunc.metadata.update(self._ctx.metadata)
        return kfunc

    def _symbol_to_param(self, symbol: PsSymbol):
        from pystencils.backend.memory import BackendPrivateProperty, BufferBasePtr

        props: set[PsSymbolProperty] = set()
        for prop in symbol.properties:
            match prop:
                case BufferBasePtr(buf):
                    field = self._ctx.find_field(buf.name)
                    props.add(FieldBasePtr(field))
                case BackendPrivateProperty():
                    pass
                case _:
                    props.add(prop)

        return Parameter(symbol.name, symbol.get_dtype(), props)

    def _get_function_params(self, ast: PsAstNode) -> list[Parameter]:
        symbols = collect_undefined_symbols(ast)
        params: list[Parameter] = [self._symbol_to_param(s) for s in symbols]
        params.sort(key=lambda p: p.name)
        return params

    def _get_headers(self, platform: Platform, body: PsBlock) -> set[str]:
        req_headers = collect_required_headers(body)
        req_headers |= platform.required_headers
        req_headers |= self._ctx.required_headers
        return req_headers


def create_staggered_kernel(
    assignments, target: Target = Target.CPU, gpu_exclusive_conditions=False, **kwargs
):
    raise NotImplementedError(
        "Staggered kernels are not yet implemented for pystencils 2.0"
    )


#   Internals


def _parse_simplification_hints(ac: AssignmentCollection):
    if "split_groups" in ac.simplification_hints:
        raise NotImplementedError(
            "Loop splitting was requested, but is not implemented yet"
        )
