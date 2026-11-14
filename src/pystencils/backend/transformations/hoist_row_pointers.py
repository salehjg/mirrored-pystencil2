"""Row base pointers for the contiguous accesses of innermost vector loops."""

from __future__ import annotations

from ..kernelcreation import KernelCreationContext, Typifier
from ..ast import PsAstNode
from ..ast.structural import PsBlock, PsLoop, PsDeclaration
from ..ast.expressions import PsExpression, PsSymbolExpr, PsAdd, PsMemAcc, PsAddressOf
from ..ast.vector import PsVecMemAcc
from ..ast.analysis import collect_undefined_symbols
from ..memory import PsSymbol


class HoistRowPointers:
    """Give every contiguous vector access of an innermost loop its own row base pointer.

    An access ``ptr[ctr + rest]``, where ``ctr`` is the loop counter (or a symbol declared in
    the loop body as a copy of it) and ``rest`` does not change inside the loop, becomes
    ``row[ctr]``, with ``row = &ptr[rest]`` declared right before the loop. Accesses with the
    same pointer and the same ``rest`` share one row pointer.

    Why: on targets without register + register addressing, such as RISC-V, each access
    needs a pointer register that advances with the loop. Compilers build these from the
    address expressions and may share one register between addresses that differ by a
    constant, e.g. the neighbours at ``x - 1`` and ``x + 1`` of a stencil, adding the
    constant in every iteration. For RVV loads and stores, which take no immediate offset,
    GCC then emits an extra ``addi`` per access and spills the extra addresses (D3Q19 AA
    odd step: 383 instead of 335 instructions per vector iteration). With one opaque base
    pointer per row, every pointer register points at its access.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        self._typify = Typifier(ctx)

    def __call__(self, node: PsAstNode) -> PsAstNode:
        self._visit(node)
        return node

    def _visit(self, node: PsAstNode) -> None:
        if isinstance(node, PsBlock):
            statements = []
            for stmt in node.statements:
                if isinstance(stmt, PsLoop) and not self._contains_loop(stmt.body):
                    statements += self._process(stmt)
                else:
                    self._visit(stmt)
                statements.append(stmt)
            node.statements = statements
        else:
            for c in node.children:
                self._visit(c)

    @staticmethod
    def _contains_loop(node: PsAstNode) -> bool:
        return any(isinstance(c, PsLoop) or HoistRowPointers._contains_loop(c) for c in node.children)

    @staticmethod
    def _vector_accesses(node: PsAstNode) -> list[PsVecMemAcc]:
        found = [node] if isinstance(node, PsVecMemAcc) else []
        return found + [a for c in node.children for a in HoistRowPointers._vector_accesses(c)]

    @staticmethod
    def _summands(expr: PsExpression) -> list[PsExpression]:
        if isinstance(expr, PsAdd):
            return HoistRowPointers._summands(expr.operand1) + HoistRowPointers._summands(expr.operand2)
        return [expr]

    def _process(self, loop: PsLoop) -> list[PsDeclaration]:
        """Rewrite the accesses in `loop`; returns the row pointer declarations to put before it."""
        counter = loop.counter.symbol
        body = loop.body
        declared_in_body: set[PsSymbol] = {counter}
        counters: set[PsSymbol] = {counter}
        for stmt in body.statements:
            if isinstance(stmt, PsDeclaration) and isinstance(stmt.lhs, PsSymbolExpr):
                declared_in_body.add(stmt.lhs.symbol)
                if isinstance(stmt.rhs, PsSymbolExpr) and stmt.rhs.symbol == counter:
                    counters.add(stmt.lhs.symbol)

        rows: list[tuple[PsExpression, PsExpression, PsSymbol]] = []   # (pointer, rest, row pointer)
        decls: list[PsDeclaration] = []
        for acc in self._vector_accesses(body):
            if acc.stride is not None:
                continue
            terms = self._summands(acc.offset)
            ctr_terms = [
                i for i, t in enumerate(terms) if isinstance(t, PsSymbolExpr) and t.symbol in counters
            ]
            if len(ctr_terms) != 1:
                continue
            ctr_term = terms[ctr_terms[0]]
            rest_terms = [t for i, t in enumerate(terms) if i != ctr_terms[0]]
            if not rest_terms or any(collect_undefined_symbols(t) & declared_in_body for t in rest_terms):
                continue
            if collect_undefined_symbols(acc.pointer) & declared_in_body:
                continue

            rest = rest_terms[0].clone()
            for t in rest_terms[1:]:
                rest = PsAdd(rest, t.clone())
            for ptr, r, row in rows:
                if ptr.structurally_equal(acc.pointer) and r.structurally_equal(rest):
                    break
            else:
                name = acc.pointer.symbol.name if isinstance(acc.pointer, PsSymbolExpr) else "ptr"
                row = self._ctx.get_new_symbol(f"{name}_row")
                decl = PsDeclaration(PsSymbolExpr(row), PsAddressOf(PsMemAcc(acc.pointer.clone(), rest.clone())))
                decls.append(self._typify(decl))
                rows.append((acc.pointer, rest, row))
            row_expr = PsSymbolExpr(row)
            row_expr.dtype = row.get_dtype()
            acc.pointer = row_expr
            acc.offset = ctr_term.clone()
        return decls
