"""
This module contains various transformation and optimization passes that can be
executed on the backend AST.

Transformations
===============

Canonicalization
----------------

.. autoclass:: CanonicalizeSymbols
    :members: __call__

AST Cloning
-----------

.. autoclass:: CanonicalClone
    :members: __call__

Simplifying Transformations
---------------------------

.. autoclass:: EliminateConstants
    :members: __call__

.. autoclass:: TypifyAndFold
    :members:

.. autoclass:: EliminateBranches
    :members: __call__


Code Rewriting
--------------

.. autofunction:: substitute_symbols

.. autofunction:: collapse_blocks

Code Motion
-----------

.. autoclass:: HoistIterationInvariantDeclarations
    :members: __call__

Axis and Loop Transformations
-----------------------------

.. autoclass:: AxisExpansion
    :members:

.. autoclass:: MaterializeAxes
    :members:

.. autoclass:: ReshapeLoops
    :members:

.. autoclass:: InsertPragmasAtLoops
    :members:

.. autoclass:: AddOpenMP
    :members:

Vectorization
-------------

.. autoclass:: VectorizationAxis
    :members:

.. autoclass:: VectorizationContext
    :members:

.. autoclass:: AstVectorizer
    :members:

Code Lowering and Materialization
---------------------------------

.. autoclass:: ReductionsToMemory

.. autoclass:: LowerToC
    :members: __call__

.. autoclass:: SelectFunctions
    :members: __call__

.. autoclass:: SelectIntrinsics
    :members:

"""

from .canonicalize_symbols import CanonicalizeSymbols
from .canonical_clone import CanonicalClone
from .rewrite import substitute_symbols, collapse_blocks
from .eliminate_constants import EliminateConstants, TypifyAndFold
from .eliminate_branches import EliminateBranches
from .hoist_iteration_invariant_decls import HoistIterationInvariantDeclarations
from .reshape_loops import ReshapeLoops
from .add_pragmas import InsertPragmasAtLoops, LoopPragma, AddOpenMP
from .ast_vectorizer import VectorizationAxis, VectorizationContext, AstVectorizer
from .axis_expansion import AxisExpansion
from .materialize_axes import MaterializeAxes
from .reductions_to_memory import ReductionsToMemory
from .lower_to_c import LowerToC
from .select_functions import SelectFunctions
from .select_intrinsics import SelectIntrinsics
from .insert_prefetches import InsertPrefetches

__all__ = [
    "CanonicalizeSymbols",
    "CanonicalClone",
    "substitute_symbols",
    "collapse_blocks",
    "EliminateConstants",
    "TypifyAndFold",
    "EliminateBranches",
    "HoistIterationInvariantDeclarations",
    "ReshapeLoops",
    "InsertPragmasAtLoops",
    "LoopPragma",
    "AddOpenMP",
    "VectorizationAxis",
    "VectorizationContext",
    "AstVectorizer",
    "AxisExpansion",
    "MaterializeAxes",
    "ReductionsToMemory",
    "LowerToC",
    "SelectFunctions",
    "SelectIntrinsics",
    "InsertPrefetches",
]
