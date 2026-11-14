"""Software prefetching for vectorized CPU loops."""

from __future__ import annotations

from typing import Callable

from ..kernelcreation import KernelCreationContext, Typifier
from ..ast import PsAstNode
from ..ast.structural import PsBlock, PsLoop, PsDeclaration, PsAssignment, PsStatement
from ..ast.expressions import PsExpression, PsConstantExpr, PsMemAcc, PsAddressOf
from ..ast.vector import PsVecMemAcc
from ..ast.analysis import collect_undefined_symbols
from ..constants import PsConstant
from ...types import PsVectorType, PsPointerType, PsType


class InsertPrefetches:
    """Insert software prefetches into innermost vectorized loops.

    For every innermost loop whose body contains contiguous vector memory accesses, one
    prefetch per distinct access address (base pointer and offset) is inserted right before the
    first statement that accesses memory: a write prefetch if the address is written in the
    loop body, a read prefetch otherwise. The prefetched address lies ``distance_bytes`` bytes
    ahead of the accessed one, i.e. it is the address the access will touch a few iterations
    later.

    Accesses whose offset depends on a symbol declared inside the loop body, at or after the
    insertion point, are skipped; so are strided accesses.

    Why: hardware stream prefetchers track a limited number of concurrent streams. Kernels with
    many streams (D3Q19 LBM: 19 read and 23 write streams) exceed that on small in-order cores
    such as the SpacemiT X60, and every cache miss then costs a full memory latency.

    Args:
        ctx: The kernel creation context
        distance_bytes: Prefetch distance in bytes
        make_prefetch: Platform hook ``(address, element type, write) -> call``; builds the
            prefetch call for an (untyped) address expression
    """

    def __init__(
        self,
        ctx: KernelCreationContext,
        distance_bytes: int,
        make_prefetch: Callable[[PsExpression, PsType, bool], PsExpression],
    ):
        self._ctx = ctx
        self._distance = int(distance_bytes)
        self._make_prefetch = make_prefetch
        self._typify = Typifier(ctx)

    def __call__(self, node: PsAstNode) -> PsAstNode:
        self._visit(node)
        return node

    def _visit(self, node: PsAstNode) -> None:
        if isinstance(node, PsLoop) and not self._contains_loop(node.body):
            self._process(node.body)
        else:
            for c in node.children:
                self._visit(c)

    @staticmethod
    def _contains_loop(node: PsAstNode) -> bool:
        return any(isinstance(c, PsLoop) or InsertPrefetches._contains_loop(c) for c in node.children)

    @staticmethod
    def _accesses(node: PsAstNode, written: bool = False) -> list[tuple[PsVecMemAcc, bool]]:
        """All vector memory accesses in `node`, with a flag telling whether they are written."""
        match node:
            case PsAssignment(lhs, rhs) if isinstance(lhs, PsVecMemAcc):
                return [(lhs, True)] + InsertPrefetches._accesses(lhs.offset) + InsertPrefetches._accesses(rhs)
            case PsVecMemAcc():
                return [(node, written)] + [a for c in node.children for a in InsertPrefetches._accesses(c)]
            case _:
                return [a for c in node.children for a in InsertPrefetches._accesses(c)]

    def _process(self, body: PsBlock) -> None:
        stmts = list(body.statements)
        found: list[list] = []   # [access, written]
        first: int | None = None
        for i, stmt in enumerate(stmts):
            accs = self._accesses(stmt)
            if accs and first is None:
                first = i
            for acc, written in accs:
                if acc.stride is not None:
                    continue
                for entry in found:
                    if entry[0].pointer.structurally_equal(acc.pointer) and entry[0].offset.structurally_equal(
                        acc.offset
                    ):
                        entry[1] = entry[1] or written
                        break
                else:
                    found.append([acc, written])

        if first is None:
            return

        declared_later = {s.lhs.symbol for s in stmts[first:] if isinstance(s, PsDeclaration)}

        prefetches: list[PsStatement] = []
        for acc, written in found:
            if collect_undefined_symbols(acc.offset) & declared_later:
                continue
            vtype = acc.get_vector_type()
            assert isinstance(vtype, PsVectorType)
            elems = max(1, self._distance // vtype.scalar_type.itemsize)
            idx_type = acc.offset.get_dtype()
            offset = acc.offset.clone() + PsConstantExpr(PsConstant(elems, idx_type))
            addr = PsAddressOf(PsMemAcc(acc.pointer.clone(), offset))
            #   element type as seen through the base pointer (keeps its const qualifier)
            ptr_type = acc.pointer.get_dtype()
            elem_type = ptr_type.base_type if isinstance(ptr_type, PsPointerType) else vtype.scalar_type
            prefetches.append(self._typify(PsStatement(self._make_prefetch(addr, elem_type, written))))

        body.statements = stmts[:first] + prefetches + stmts[first:]
