"""Blender user interface: scene properties, sidebar panels and File menus."""

from __future__ import annotations

from . import menus, panels, properties

__all__ = ["menus", "panels", "properties", "register", "unregister"]


def register() -> None:
    """Register the property group, panels and File menu entries."""
    properties.register()
    panels.register()
    menus.register()


def unregister() -> None:
    """Unregister them again, in the reverse order."""
    menus.unregister()
    panels.unregister()
    properties.unregister()
