"""Named boolean attributes on a molecule's mesh — Gala's *aliases*.

A named selection is nothing more than a boolean attribute on the ``POINT``
domain, one value per atom. That is deliberately the same representation
three other things already use:

``Molecular Nodes``
    ``Molecule.add_style(style, selection="pocket")`` wires a Named Attribute
    node into the style's ``Selection`` socket, so an alias can be styled
    without Gala touching the node tree at all.
``PyMOL``
    :func:`blender_gala.pymol.save.save_session` writes named boolean
    attributes out as PyMOL selections, so an alias survives the round trip.
``Geometry nodes``
    anything else the user wires up by hand.

Attribute names are therefore left *bare* — ``pocket``, not ``gala_pocket`` —
because those consumers expect the name the user chose. What distinguishes an
alias from the dozen booleans Molecular Nodes stores (``is_solvent``,
``is_backbone`` …) is an explicit registry kept on the object, which is what
:func:`registered` reads.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

__all__ = [
    "REGISTRY_KEY",
    "delete_boolean",
    "list_booleans",
    "read_boolean",
    "register",
    "registered",
    "safe_name",
    "unregister",
    "write_boolean",
]

#: Object custom property holding the names Gala considers user aliases.
REGISTRY_KEY = "gala_selections"


def _mesh(obj: Any) -> Any:
    data = getattr(obj, "data", None)
    return data if getattr(data, "attributes", None) is not None else None


def _bmesh_layer(obj: Any, name: str, create: bool) -> tuple[Any, Any] | None:
    """Return ``(bmesh, layer)`` when ``obj`` is in Edit Mode, else ``None``.

    Mesh attributes are unreadable and unwritable while the mesh is open in
    Edit Mode — the values live in the BMesh until the mode is toggled. Rather
    than force a mode switch under the user, both directions go through the
    BMesh when that is where the data is.
    """
    if getattr(obj, "mode", "OBJECT") != "EDIT":
        return None
    import bmesh

    mesh = bmesh.from_edit_mesh(obj.data)
    layer = mesh.verts.layers.bool.get(name)
    if layer is None:
        if not create:
            return None
        layer = mesh.verts.layers.bool.new(name)
    return mesh, layer


def read_boolean(obj: Any, name: str, n_atoms: int | None = None) -> np.ndarray | None:
    """Read a boolean point attribute off a molecule's mesh.

    Parameters
    ----------
    obj : bpy.types.Object
        The molecule object.
    name : str
        Attribute name.
    n_atoms : int, optional
        Expected length. A mismatch returns ``None`` rather than a mask that
        would silently address the wrong atoms.

    Returns
    -------
    numpy.ndarray or None
        Boolean mask, or ``None`` if there is no such attribute.
    """
    found = _bmesh_layer(obj, name, create=False)
    if found is not None:
        mesh, layer = found
        values = np.array([vertex[layer] for vertex in mesh.verts], dtype=bool)
        return None if n_atoms is not None and len(values) != n_atoms else values

    mesh_data = _mesh(obj)
    attribute = mesh_data.attributes.get(name) if mesh_data else None
    if attribute is None or getattr(attribute, "data_type", "") != "BOOLEAN":
        return None
    length = len(attribute.data)
    if n_atoms is not None and length != n_atoms:
        return None
    values = np.zeros(length, dtype=bool)
    attribute.data.foreach_get("value", values)
    return values


def write_boolean(obj: Any, name: str, mask: np.ndarray) -> None:
    """Store a boolean mask as a named point attribute.

    Works in Edit Mode as well as Object Mode, so an alias can be created
    without leaving the mode the selection was made in.

    Raises
    ------
    ValueError
        If ``mask`` is not one value per vertex.
    """
    mask = np.asarray(mask, dtype=bool)

    found = _bmesh_layer(obj, name, create=True)
    if found is not None:
        import bmesh

        mesh, layer = found
        if len(mask) != len(mesh.verts):
            raise ValueError(
                f"mask has {len(mask)} values but the mesh has {len(mesh.verts)} vertices"
            )
        mesh.verts.ensure_lookup_table()
        for vertex, value in zip(mesh.verts, mask, strict=True):
            vertex[layer] = bool(value)
        bmesh.update_edit_mesh(obj.data)
        return

    mesh_data = _mesh(obj)
    if mesh_data is None:
        raise ValueError(f"object {getattr(obj, 'name', obj)!r} has no mesh")
    if len(mask) != len(mesh_data.vertices):
        raise ValueError(
            f"mask has {len(mask)} values but the mesh has "
            f"{len(mesh_data.vertices)} vertices"
        )

    attribute = mesh_data.attributes.get(name)
    if attribute is not None and (
        attribute.data_type != "BOOLEAN" or attribute.domain != "POINT"
    ):
        # A float attribute of the same name would take the write silently and
        # then never match anything.
        mesh_data.attributes.remove(attribute)
        attribute = None
    if attribute is None:
        attribute = mesh_data.attributes.new(name, "BOOLEAN", "POINT")
    attribute.data.foreach_set("value", mask)
    mesh_data.update()


def delete_boolean(obj: Any, name: str) -> bool:
    """Remove a named attribute. Returns whether there was one to remove."""
    found = _bmesh_layer(obj, name, create=False)
    if found is not None:
        import bmesh

        mesh, layer = found
        mesh.verts.layers.bool.remove(layer)
        bmesh.update_edit_mesh(obj.data)
        return True

    mesh_data = _mesh(obj)
    attribute = mesh_data.attributes.get(name) if mesh_data else None
    if attribute is None:
        return False
    mesh_data.attributes.remove(attribute)
    return True


def list_booleans(obj: Any) -> list[str]:
    """Every boolean point attribute on the mesh, Gala's or not.

    Blender's own bookkeeping attributes — ``.select_vert`` and friends — are
    left out; they are internal and start with a dot by convention.
    """
    mesh_data = _mesh(obj)
    if mesh_data is None:
        return []
    return [
        attribute.name
        for attribute in mesh_data.attributes
        if attribute.data_type == "BOOLEAN"
        and attribute.domain == "POINT"
        and not attribute.name.startswith(".")
    ]


# ---------------------------------------------------------------------------
# The alias registry
# ---------------------------------------------------------------------------


def registered(obj: Any) -> list[str]:
    """The alias names on this object, in the order they were created.

    Names whose attribute has since been deleted — by hand in the Object Data
    panel, say — are dropped, so the list never advertises an alias that would
    style nothing.
    """
    if obj is None:
        return []
    stored = obj.get(REGISTRY_KEY, ())
    present = set(list_booleans(obj))
    return [str(name) for name in stored if str(name) in present]


def register(obj: Any, name: str) -> None:
    """Record ``name`` as an alias, keeping the order and avoiding duplicates."""
    names = [str(existing) for existing in obj.get(REGISTRY_KEY, ())]
    if name not in names:
        names.append(name)
    obj[REGISTRY_KEY] = names


def unregister(obj: Any, name: str) -> None:
    """Forget ``name``, leaving the attribute itself alone."""
    names = [str(existing) for existing in obj.get(REGISTRY_KEY, ())]
    obj[REGISTRY_KEY] = [existing for existing in names if existing != name]


_UNSAFE = re.compile(r"[^0-9A-Za-z_]+")


def safe_name(name: str) -> str:
    """Turn user input into a usable attribute name.

    Geometry nodes will read any name at all, but one with a space or a dot in
    it is painful to type into a Named Attribute node, and a leading dot marks
    an attribute as internal and hides it from the UI.
    """
    cleaned = _UNSAFE.sub("_", name.strip()).strip("_")
    if not cleaned:
        return "selection"
    if cleaned[0].isdigit():
        cleaned = f"sel_{cleaned}"
    return cleaned
