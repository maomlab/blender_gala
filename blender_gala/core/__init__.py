"""Core layer: data adapters, selection language and geometry primitives.

Nothing in this package depends on Gala's higher-level features, and the
science-facing modules (:mod:`selection`, :mod:`chemistry`, :mod:`units`, and
the pure-geometry half of :mod:`geometry`) import ``bpy`` only optionally.
"""

from __future__ import annotations

from . import chemistry, collections, entity, exceptions, geometry, mn, selection, units
from .entity import AtomStructure
from .exceptions import (
    AmbiguousSelectionError,
    EmptySelectionError,
    GalaError,
    MolecularNodesUnavailable,
    SelectionError,
    SelectionSyntaxError,
    StructureError,
)
from .selection import Selection, compile_selection, select, select_indices

__all__ = [
    "AmbiguousSelectionError",
    "AtomStructure",
    "EmptySelectionError",
    "GalaError",
    "MolecularNodesUnavailable",
    "Selection",
    "SelectionError",
    "SelectionSyntaxError",
    "StructureError",
    "chemistry",
    "collections",
    "compile_selection",
    "entity",
    "exceptions",
    "geometry",
    "mn",
    "select",
    "select_indices",
    "selection",
    "units",
]
