"""Entries in Blender's ``File > Import`` and ``File > Export`` menus.

The sidebar panel is where someone already using Gala reaches for a session.
The File menu is where someone who has just been handed a ``.pse`` looks
first, because that is where every other format in Blender lives — so the
same two operators are registered in both places.

Appending to a menu is not like registering a class: Blender keeps a list of
draw functions per menu, and removing one that is not there raises. Both
directions are therefore written to tolerate being called twice, for the same
reason :mod:`blender_gala.core.registration` is (SPECIFICATION D-2a): having
the extension installed *and* a checkout on ``sys.path`` is the normal
developer setup, and the two copies must not break each other's teardown.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = ["draw_export", "draw_import", "register", "unregister"]

#: What the entry is called in the menu. The extension in the label is the
#: convention every other Blender importer follows.
_LABEL = "PyMOL Session (.pse)"


def draw_import(self: Any, context: Any) -> None:
    """Draw the import entry. Appended to ``TOPBAR_MT_file_import``."""
    self.layout.operator("gala.load_pymol_session", text=_LABEL, icon="IMPORT")


def draw_export(self: Any, context: Any) -> None:
    """Draw the export entry. Appended to ``TOPBAR_MT_file_export``."""
    self.layout.operator("gala.save_pymol_session", text=_LABEL, icon="EXPORT")


#: The levels offered in the Edit Mode Select menu, and what each is called
#: there. The sidebar has the same four behind an enum; this is for the person
#: who has just box-selected some atoms and has the Select menu open.
_LEVELS = (
    ("residue", "Expand to Residue"),
    ("chain", "Expand to Chain"),
    ("fragment", "Expand to Fragment"),
)


def draw_select(self: Any, context: Any) -> None:
    """Draw the expand entries. Appended to ``VIEW3D_MT_select_edit_mesh``."""
    layout = self.layout
    layout.separator()
    for level, label in _LEVELS:
        layout.operator(
            "gala.expand_selection", text=label, icon="SELECT_EXTEND"
        ).level = level


#: The menus to extend, and the function each takes.
_ENTRIES = (
    ("TOPBAR_MT_file_import", draw_import),
    ("TOPBAR_MT_file_export", draw_export),
    ("VIEW3D_MT_select_edit_mesh", draw_select),
)


def register() -> None:
    """Add Gala's entries to the File menus, without adding them twice."""
    if bpy is None:  # pragma: no cover
        return
    for menu_name, function in _ENTRIES:
        menu = getattr(bpy.types, menu_name, None)
        if menu is None:  # pragma: no cover - both exist in every Blender
            continue
        if function not in _drawn_by(menu):
            menu.append(function)


def unregister() -> None:
    """Remove them again, and do not raise if they are already gone."""
    if bpy is None:  # pragma: no cover
        return
    for menu_name, function in _ENTRIES:
        menu = getattr(bpy.types, menu_name, None)
        if menu is None:  # pragma: no cover
            continue
        if function in _drawn_by(menu):
            menu.remove(function)


def _drawn_by(menu: Any) -> list[Any]:
    """The draw functions currently appended to ``menu``.

    Blender exposes them through ``_dyn_ui_initialize``, which is also what
    ``append`` and ``remove`` operate on. It is private, so a missing one is
    treated as "nothing appended" rather than allowed to raise.
    """
    initialise = getattr(menu, "_dyn_ui_initialize", None)
    if initialise is None:  # pragma: no cover - present in every Blender
        return []
    try:
        return list(initialise())
    except Exception:  # pragma: no cover
        return []
