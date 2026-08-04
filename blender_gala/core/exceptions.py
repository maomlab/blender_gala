"""Exception types raised by Blender Gala."""

from __future__ import annotations

__all__ = [
    "AmbiguousSelectionError",
    "EmptySelectionError",
    "GalaError",
    "MolecularNodesUnavailable",
    "SelectionError",
    "SelectionSyntaxError",
    "StructureError",
]


class GalaError(Exception):
    """Base class for every error raised by Blender Gala."""


class MolecularNodesUnavailable(GalaError):
    """Molecular Nodes is required for this operation but is not importable."""

    def __init__(self, detail: str = "") -> None:
        message = (
            "Molecular Nodes could not be imported. Blender Gala's molecule-aware "
            "features require it.\n"
            "Install it from Edit > Preferences > Get Extensions, search for "
            "'Molecular Nodes', then restart Blender."
        )
        if detail:
            message = f"{message}\n\nUnderlying import error: {detail}"
        super().__init__(message)


class StructureError(GalaError):
    """The supplied object could not be interpreted as a molecular structure."""


class SelectionError(GalaError):
    """Base class for selection failures."""


class SelectionSyntaxError(SelectionError):
    """A selection string could not be parsed.

    Carries the original string and the character offset so the message can
    point at the offending token.
    """

    def __init__(self, message: str, selection: str = "", position: int = -1) -> None:
        self.selection = selection
        self.position = position
        if selection and position >= 0:
            caret = " " * position + "^"
            message = f"{message}\n    {selection}\n    {caret}"
        super().__init__(message)


class EmptySelectionError(SelectionError):
    """A selection matched no atoms where at least one was required."""


class AmbiguousSelectionError(SelectionError):
    """A selection matched several atoms where exactly one was required."""
