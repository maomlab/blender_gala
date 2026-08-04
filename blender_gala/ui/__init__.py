"""Blender user interface: scene properties and sidebar panels."""

from __future__ import annotations

from . import panels, properties

__all__ = ["panels", "properties", "register", "unregister"]


def register() -> None:
    """Register the property group and panels."""
    properties.register()
    panels.register()


def unregister() -> None:
    """Unregister the panels and property group."""
    panels.unregister()
    properties.unregister()
