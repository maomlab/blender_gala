"""Named boolean attributes on a molecule's mesh — Gala's *aliases*.

A named selection is nothing more than a boolean attribute on the ``POINT``
domain, one value per atom. That is deliberately the same representation
several other things already use:

``The selection language``
    a bare word that is not a keyword is looked up here, so ``pocket around 4``
    means what it says (:func:`named_selections`).
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

import hashlib
import re
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .viewport import is_removed, require_object

__all__ = [
    "REGISTRY_KEY",
    "attribute_conflict",
    "delete_boolean",
    "list_booleans",
    "named_selections",
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
    """The mesh datablock behind ``obj``, or ``None`` when there is none.

    A deleted object counts as having none. Reading it is what raises
    ``ReferenceError``, and the reading side of this module answers "no
    attributes" rather than refusing: a selection string that never names a
    stored selection has no business failing because the mesh went away. The
    writing side below is where a deleted object is an error, because there is
    nowhere for the value to go.
    """
    if is_removed(obj):
        return None
    data = getattr(obj, "data", None)
    return data if getattr(data, "attributes", None) is not None else None


def _flush(bmesh: Any, obj: Any) -> None:
    """Push BMesh edits back to the mesh without claiming a rebuild.

    ``update_edit_mesh`` defaults to ``destructive=True``, which announces that
    geometry has been added or removed. Storing attribute values changes
    neither, and saying otherwise makes Blender tear down and rebuild the edit
    mesh — needless work on a structure with tens of thousands of atoms, and
    enough to leave what a geometry nodes modifier is drawing in Edit Mode out
    of step with the atoms underneath it.
    """
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


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
        Boolean mask, or ``None`` if there is no such attribute — including
        when the object holding it has been deleted, since a mask that no
        longer exists and one that never did read the same way here.
    """
    if is_removed(obj):
        return None

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


def attribute_conflict(obj: Any, name: str) -> str | None:
    """Say why a mask cannot be stored under ``name``, or ``None`` if it can.

    Attribute names are bare, so the ones a user reaches for when naming a
    selection are exactly the ones Molecular Nodes has already taken for the
    structure's own per-atom data: ``res_id``, ``b_factor``, ``atomic_number``,
    ``Color``. Writing a mask over one of those would replace the residue
    numbering with a boolean and there is nothing to undo it with, so the
    collision is refused and the user picks another name.

    Parameters
    ----------
    obj : bpy.types.Object
        The molecule object.
    name : str
        The attribute name a mask is about to be written under.

    Returns
    -------
    str or None
        A message naming what is in the way, or ``None`` when the name is free
        or already holds a boolean point attribute — rewriting one of those is
        how a stored selection is updated.
    """
    mesh_data = _mesh(obj)
    attribute = mesh_data.attributes.get(name) if mesh_data else None
    if attribute is None:
        return None
    data_type = getattr(attribute, "data_type", "")
    domain = getattr(attribute, "domain", "")
    if data_type == "BOOLEAN" and domain == "POINT":
        return None
    return (
        f"{name!r} is already a {data_type} attribute on the {domain.lower()} "
        f"domain of {getattr(obj, 'name', obj)!r} — storing a selection under "
        "that name would destroy it. Choose another name."
    )


def write_boolean(obj: Any, name: str, mask: np.ndarray) -> None:
    """Store a boolean mask as a named point attribute.

    Works in Edit Mode as well as Object Mode, so an alias can be created
    without leaving the mode the selection was made in.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted, so there is no mesh to store the mask on.
    ValueError
        If ``mask`` is not one value per vertex, or ``name`` already belongs to
        an attribute this could not write to without destroying it — see
        :func:`attribute_conflict`.
    """
    # Before anything else: every message below names the object, and asking a
    # deleted one for its name is what raises.
    require_object(obj)

    mask = np.asarray(mask, dtype=bool)

    # Checked before the mode is looked at: the mesh carries the same
    # attributes either way, and the Edit Mode path writes to a BMesh layer of
    # its own, so leaving the check to that path would let the same name be
    # refused in one mode and taken in the other.
    conflict = attribute_conflict(obj, name)
    if conflict is not None:
        raise ValueError(conflict)

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
        _flush(bmesh, obj)
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
    if attribute is None:
        attribute = mesh_data.attributes.new(name, "BOOLEAN", "POINT")
    attribute.data.foreach_set("value", mask)
    mesh_data.update()


def delete_boolean(obj: Any, name: str) -> bool:
    """Remove a named attribute. Returns whether there was one to remove.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted. Removing an attribute from an object that
        is gone is not the same as finding nothing to remove.
    """
    require_object(obj)

    found = _bmesh_layer(obj, name, create=False)
    if found is not None:
        import bmesh

        mesh, layer = found
        mesh.verts.layers.bool.remove(layer)
        _flush(bmesh, obj)
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
    style nothing. An object that has itself been deleted has no aliases to
    list; callers for whom that is an error say so themselves, as
    :meth:`blender_gala.core.entity.AtomStructure.alias_names` does.
    """
    if obj is None or is_removed(obj):
        return []
    stored = obj.get(REGISTRY_KEY, ())
    present = set(list_booleans(obj))
    return [str(name) for name in stored if str(name) in present]


def register(obj: Any, name: str) -> None:
    """Record ``name`` as an alias, keeping the order and avoiding duplicates.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted, so the registry has nowhere to live.
    """
    require_object(obj)
    names = [str(existing) for existing in obj.get(REGISTRY_KEY, ())]
    if name not in names:
        names.append(name)
    obj[REGISTRY_KEY] = names


def unregister(obj: Any, name: str) -> None:
    """Forget ``name``, leaving the attribute itself alone.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted, so the registry cannot be rewritten.
    """
    require_object(obj)
    names = [str(existing) for existing in obj.get(REGISTRY_KEY, ())]
    obj[REGISTRY_KEY] = [existing for existing in names if existing != name]


# ---------------------------------------------------------------------------
# Aliases as selection-language names
# ---------------------------------------------------------------------------


class _BooleanAttributes(Mapping):
    """The mesh's boolean attributes, read one at a time as they are asked for.

    Backs the names a selection string may refer to. Reading every attribute
    up front would be wasteful — ``pocket around 4`` needs one of them — and in
    Edit Mode a read walks the BMesh vertex by vertex, so each mask is fetched
    only when it is named, and then kept for the life of this mapping.

    Registered aliases come first, but the plain boolean attributes are here
    too: a selection imported from a PyMOL session, or one wired up by hand in
    the geometry node editor, is just as referenceable as one Gala stored.
    """

    def __init__(self, obj: Any, n_atoms: int | None = None) -> None:
        self._obj = obj
        self._n_atoms = n_atoms
        self._masks: dict[str, np.ndarray | None] = {}

    def _names(self) -> list[str]:
        aliases = registered(self._obj)
        return aliases + [n for n in list_booleans(self._obj) if n not in aliases]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names())

    def __len__(self) -> int:
        return len(self._names())

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._masks:
            self._masks[key] = read_boolean(self._obj, key, self._n_atoms)
        mask = self._masks[key]
        if mask is None:
            raise KeyError(key)
        return mask


def named_selections(target: Any, n_atoms: int | None = None) -> Mapping[str, Any]:
    """The named selections ``target`` can resolve, as a lazy ``{name: mask}``.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, AtomStructure, or AtomArray
        Whatever the caller had. Anything without a mesh behind it — a bare
        biotite ``AtomArray``, say — has no names, and gets an empty mapping.
    n_atoms : int, optional
        Expected mask length. Attributes of another length are left out rather
        than returned to address the wrong atoms.

    Returns
    -------
    Mapping
        Names in creation order, masks read on demand. Empty when the object
        the names would have come from has been deleted: the chemistry a
        selection is mostly made of does not depend on the mesh, and taking
        ``protein`` down with ``pocket`` would make the answer depend on
        whether anything happened to have filled a cache earlier.
    """
    obj = _target_object(target)
    if obj is None and _mesh(target) is not None:
        obj = target
    return _BooleanAttributes(obj, n_atoms) if obj is not None else {}


def _target_object(target: Any) -> Any:
    """The Blender object ``target`` carries, or ``None``.

    Not a plain ``getattr``: Molecular Nodes' ``Molecule.object`` raises
    ``LinkedObjectError`` once the object is gone and a deleted object raises
    ``ReferenceError`` for any attribute at all, neither of which is an
    ``AttributeError``, so the default of a ``getattr`` would never apply.
    """
    if is_removed(target):
        return None
    try:
        obj = target.object
    except Exception:
        return None
    return None if is_removed(obj) else obj


#: Anything that is not a word character. ``\W`` rather than ``[^0-9A-Za-z_]``
#: because it is Unicode-aware: a name written in Cyrillic or Chinese is made
#: of word characters and survives, where the ASCII class dropped every letter
#: of it and left nothing to name the selection with.
_UNSAFE = re.compile(r"\W+")


def safe_name(name: str) -> str:
    """Turn user input into a usable attribute name.

    Geometry nodes will read any name at all, but one with a space or a dot in
    it is painful to type into a Named Attribute node, and a leading dot marks
    an attribute as internal and hides it from the UI.
    """
    cleaned = _UNSAFE.sub("_", name.strip()).strip("_")
    if not cleaned:
        # Nothing of what was typed is a word character. Every such name would
        # otherwise become "selection", so storing "!!!" and then "@@@" would
        # silently replace the first with the second; the digest keeps names
        # that differ apart, and is itself already safe, so cleaning the result
        # again leaves it alone.
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        return f"selection_{digest}"
    if cleaned[0].isdigit():
        cleaned = f"sel_{cleaned}"
    return cleaned
