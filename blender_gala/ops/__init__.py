"""Blender operators wrapping the Python API."""

from __future__ import annotations

from . import operators
from .operators import active_structure, selected_atom_indices

__all__ = [
    "active_structure",
    "operators",
    "register",
    "selected_atom_indices",
    "unregister",
]


def register() -> None:
    """Register every operator."""
    operators.register()


def unregister() -> None:
    """Unregister every operator."""
    operators.unregister()
