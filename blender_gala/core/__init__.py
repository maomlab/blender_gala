"""Core layer: data adapters, selection language and geometry primitives.

Nothing in this package depends on Gala's higher-level features, and the
science-facing modules (:mod:`selection`, :mod:`chemistry`, :mod:`units`, and
the pure-geometry half of :mod:`geometry`) import ``bpy`` only optionally.
"""

from __future__ import annotations

from . import (
    attributes,
    chemistry,
    collections,
    entity,
    exceptions,
    geometry,
    interactive,
    mn,
    mn_compat,
    registration,
    selection,
    units,
    viewport,
)
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
from .interactive import (
    alias_combine,
    create_alias,
    delete_alias,
    describe_viewport_selection,
    expand_viewport_selection,
    list_aliases,
    select_alias,
    set_viewport_selection,
    style_alias,
    viewport_selection,
)
from .selection import (
    LEVELS,
    Selection,
    compile_selection,
    describe_selection,
    expand_selection,
    select,
    select_indices,
)

__all__ = [
    "LEVELS",
    "AmbiguousSelectionError",
    "AtomStructure",
    "EmptySelectionError",
    "GalaError",
    "MolecularNodesUnavailable",
    "Selection",
    "SelectionError",
    "SelectionSyntaxError",
    "StructureError",
    "alias_combine",
    "attributes",
    "chemistry",
    "collections",
    "compile_selection",
    "create_alias",
    "delete_alias",
    "describe_selection",
    "describe_viewport_selection",
    "entity",
    "exceptions",
    "expand_selection",
    "expand_viewport_selection",
    "geometry",
    "interactive",
    "list_aliases",
    "mn",
    "mn_compat",
    "registration",
    "select",
    "select_alias",
    "select_indices",
    "selection",
    "set_viewport_selection",
    "style_alias",
    "units",
    "viewport",
    "viewport_selection",
]
