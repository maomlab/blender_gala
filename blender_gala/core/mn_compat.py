"""A guard around one Molecular Nodes panel function.

Molecular Nodes draws, under its Styles list, the selection that limits the
active style. It finds the Named Attribute node feeding the style's
``Selection`` socket and then looks the attribute name up in
``entity.selections`` — the manager that turns MDAnalysis selection strings
into boolean attributes as a trajectory plays.

Only trajectories have one. A ``Molecule`` does not, so as soon as any style
on a molecule is limited to a named attribute, drawing that panel raises
``AttributeError`` and the Styles panel comes up empty until a different style
is made active. Molecular Nodes reaches the same state through its own
``Molecule.add_style(style, selection="…")``, so this is an upstream bug
rather than something Gala causes uniquely — but Gala's stored selections
(:mod:`blender_gala.core.attributes`) are built on exactly that mechanism, so
leaving it alone would mean shipping a button that visibly breaks somebody
else's panel.

The patch is deliberately small: the original is called first and only a
failure falls back to a plain label, so trajectories keep the full editable
selection string they have today. It is removed again on unregister, and it
does nothing at all when Molecular Nodes is absent or has been fixed.
"""

from __future__ import annotations

from typing import Any

from . import mn as mn_bridge

__all__ = ["install", "remove"]

_original: Any = None
_patched: Any = None


def _panel_module() -> Any:
    """Molecular Nodes' styles panel module, or ``None``."""
    module = mn_bridge.get_mn()
    if module is None:
        return None
    panel = getattr(getattr(module, "ui", None), "panel", None)
    return panel if hasattr(panel, "panel_selection_node") else None


def install() -> bool:
    """Wrap the panel function. Returns whether anything was patched."""
    global _original, _patched

    if _original is not None:  # already installed
        return True
    panel = _panel_module()
    if panel is None:
        return False

    original = panel.panel_selection_node

    def panel_selection_node(layout: Any, node: Any, entity: Any) -> None:
        try:
            original(layout, node, entity)
            return
        except (AttributeError, KeyError, TypeError):
            pass

        # Fall back to naming the selection. Read-only: the mask came from a
        # viewport pick or a selection string, not from anything this panel
        # could meaningfully edit in place.
        name = _selection_attribute(node)
        if name:
            row = layout.row()
            row.label(text="Selection:")
            row.label(text=name, icon="RESTRICT_SELECT_OFF")

    _original = original
    _patched = panel_selection_node
    panel.panel_selection_node = panel_selection_node
    return True


def remove() -> None:
    """Put Molecular Nodes' own function back, if this module replaced it."""
    global _original, _patched

    if _original is None:
        return
    panel = _panel_module()
    # Only restore if nothing else has replaced it since; overwriting a third
    # party's patch would be the same discourtesy this module is avoiding.
    if panel is not None and getattr(panel, "panel_selection_node", None) is _patched:
        panel.panel_selection_node = _original
    _original = None
    _patched = None


def _selection_attribute(node: Any) -> str | None:
    """The attribute name limiting a style node, if one does."""
    socket = getattr(node, "inputs", {}).get("Selection")
    links = getattr(socket, "links", ()) if socket is not None else ()
    if not links:
        return None
    source = links[0].from_socket.node
    if "NamedAttribute" not in getattr(source, "bl_idname", ""):
        return None
    inputs: Any = getattr(source, "inputs", None)
    return str(inputs[0].default_value) if inputs else None
