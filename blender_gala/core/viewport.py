"""The bridge between Blender's viewport selection and an atom mask.

Molecular Nodes builds a molecule so that vertex *i* is atom *i*
(:mod:`blender_gala.core.entity`), which means the vertices a user has
selected in Edit Mode already *are* a boolean atom mask — Blender's box,
circle and lasso select are a working atom picker, and nothing here has to
reimplement them.

What is missing is only the translation, in both directions: what is selected
becomes a mask (and from there a PyMOL selection string), and any mask can be
pushed back out to the viewport so a typed selection can be seen.

Both directions work in Edit Mode and Object Mode. The distinction matters:
while a mesh is open in Edit Mode its selection flags live in the BMesh and
the copies on ``mesh.vertices`` are stale, so reading them there would return
whatever was selected when the mode was last toggled.

This is also where Gala asks whether a Blender object is still *there*
(:func:`is_removed`, :func:`require_object`), since deleting the molecule is
the state every other module has to have an answer for.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from .exceptions import StructureError

__all__ = [
    "is_removed",
    "is_selectable",
    "object_mode",
    "read_selection",
    "require_object",
    "selected_indices",
    "write_selection",
]


def is_removed(obj: Any) -> bool:
    """Whether ``obj`` is a Blender datablock that has since been deleted.

    Deleting an object — one press of X in the outliner, an undo of an import,
    a scene reload — leaves every Python reference to it pointing at a
    ``StructRNA`` that Blender has unlinked. *Any* attribute access on it then
    raises ``ReferenceError``, which is not an ``AttributeError``, so no
    ``getattr`` default applies and no ordinary liveness test works: the only
    way to ask is to touch it and see.

    Parameters
    ----------
    obj : bpy.types.Object or None
        Whatever the caller is holding. ``None``, a biotite array or anything
        else that was never a Blender datablock has not been deleted, so the
        answer for those is ``False``.

    Returns
    -------
    bool
    """
    if obj is None:
        return False
    try:
        # The access is the test; the value is of no interest.
        _ = obj.name
    except ReferenceError:
        return True
    except Exception:
        # Not a Blender datablock at all, or one whose name is unreadable for
        # some other reason. Either way nothing was deleted out from under us.
        pass
    return False


def require_object(obj: Any, name: str | None = None) -> Any:
    """Return ``obj``, refusing when it is a Blender object that is gone.

    Parameters
    ----------
    obj : bpy.types.Object or None
        The object an operation is about to use. ``None`` is passed through —
        having no object at all is a state several callers handle themselves,
        and only a *deleted* one is an error.
    name : str, optional
        What the object was called, remembered from before it was deleted;
        reading ``obj.name`` is exactly what raises once it is gone.

    Returns
    -------
    bpy.types.Object or None
        ``obj`` unchanged.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted.
    """
    if not is_removed(obj):
        return obj
    label = repr(name) if name else "this structure's Blender object"
    raise StructureError(
        f"the Blender object {label} has been deleted, so there is no geometry, "
        "viewport selection or stored selection left to read. Re-import the "
        "structure; the chemistry already read from it — atom count, "
        "coordinates, labels — is still answerable."
    )


@contextlib.contextmanager
def object_mode(obj: Any) -> Iterator[None]:
    """Run a block in Object Mode, and put the user back where they were.

    Some of what Gala calls cannot run from Edit Mode at all. Appending a
    Molecular Nodes style is the one that matters here: it reaches for
    ``bpy.ops.wm.append`` to pull the node group out of the add-on's asset
    file, and that operator's poll fails outside Object Mode — so styling a
    selection would fail precisely when it is most useful, straight after
    picking the atoms.

    Leaving and re-entering Edit Mode preserves the vertex selection, so the
    round trip is invisible apart from the viewport blinking.
    """
    # Reading the mode of a deleted object raises Blender's own ReferenceError,
    # and this is a context manager: the caller would be left with a traceback
    # from the `with` line rather than a message about the missing molecule.
    require_object(obj)
    # Annotated because `obj` is untyped: Blender's own `mode_set` takes a
    # literal, and whatever an object reports as its mode is one at runtime.
    mode: Any = getattr(obj, "mode", "OBJECT")
    if obj is None or mode == "OBJECT":
        yield
        return

    import bpy

    view_layer = bpy.context.view_layer
    if view_layer is None:  # pragma: no cover - a real context always has one
        # No view layer, so nothing can be made active and no mode can be set.
        # Whatever the caller is doing will fail on its own terms, which is a
        # better error than one about a missing attribute here.
        yield
        return

    previous = view_layer.objects.active
    view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        yield
    finally:
        view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode=mode)
        if previous is not None:
            view_layer.objects.active = previous


def is_selectable(obj: Any) -> bool:
    """Whether ``obj`` is a mesh whose vertices Gala can read a selection from.

    A deleted object is not selectable rather than an error: this is a
    predicate, and the functions that go on to *use* the object say so
    themselves through :func:`require_object`.
    """
    if obj is None or is_removed(obj):
        return False
    return getattr(obj, "type", None) == "MESH"


def read_selection(obj: Any, n_atoms: int | None = None) -> np.ndarray:
    """Return the selected vertices of ``obj`` as a boolean atom mask.

    Parameters
    ----------
    obj : bpy.types.Object
        A mesh object, usually a molecule imported by Molecular Nodes.
    n_atoms : int, optional
        Expected length. When the mesh no longer has one vertex per atom — an
        edit that added or removed geometry — the selection is refused rather
        than returned, since it would address the wrong atoms.

    Returns
    -------
    numpy.ndarray
        Boolean mask, empty when there is nothing to read.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted. An all-false mask would say that nothing
        is selected, which is a different — and wrong — answer from "the
        object this selection was on is gone".

        Or if the mesh has a vertex count other than ``n_atoms``. The mask read
        off the mesh is correct about the *mesh*; what cannot be done is to say
        which atoms it names, and "nothing is selected" is a valid-looking
        answer that is false — the sibling paths here already refuse a mask of
        the wrong length rather than pad it.
    """
    require_object(obj)
    if not is_selectable(obj):
        return np.zeros(0 if n_atoms is None else n_atoms, dtype=bool)

    if getattr(obj, "mode", "OBJECT") == "EDIT":
        import bmesh

        mesh = bmesh.from_edit_mesh(obj.data)
        flags = np.array([vertex.select for vertex in mesh.verts], dtype=bool)
    else:
        vertices = obj.data.vertices
        flags = np.zeros(len(vertices), dtype=bool)
        vertices.foreach_get("select", flags)

    if n_atoms is not None and len(flags) != n_atoms:
        raise StructureError(
            f"the mesh of {getattr(obj, 'name', obj)!r} has {len(flags)} vertices "
            f"but the structure has {n_atoms} atoms, so what is selected in the "
            "viewport cannot be read as atoms. Undo the edit that added or "
            "removed vertices, or re-import the structure."
        )
    return flags


def selected_indices(obj: Any) -> np.ndarray:
    """The 0-based indices of the selected vertices."""
    return np.flatnonzero(read_selection(obj))


def write_selection(obj: Any, mask: np.ndarray) -> None:
    """Select exactly the vertices in ``mask``, deselecting the rest.

    Raises
    ------
    StructureError
        If ``obj`` has been deleted.
    ValueError
        If ``obj`` has no mesh, or ``mask`` is not one value per vertex.
    """
    require_object(obj)
    if not is_selectable(obj):
        raise ValueError(f"object {getattr(obj, 'name', obj)!r} is not a mesh")

    mask = np.asarray(mask, dtype=bool)

    if getattr(obj, "mode", "OBJECT") == "EDIT":
        import bmesh

        # Named apart from the mesh datablock below: the same name for both
        # would give the two branches one variable of two unrelated types.
        edit_mesh = bmesh.from_edit_mesh(obj.data)
        if len(mask) != len(edit_mesh.verts):
            raise ValueError(
                f"mask has {len(mask)} values but the mesh has "
                f"{len(edit_mesh.verts)} vertices"
            )
        for vertex, value in zip(edit_mesh.verts, mask, strict=True):
            vertex.select = bool(value)
        # Without the flush, edges and faces between newly selected vertices
        # stay unselected and the mesh redraws inconsistently.
        edit_mesh.select_flush_mode()
        # Not `destructive`: selecting changes no geometry, and saying it does
        # makes Blender rebuild the edit mesh under the modifier for nothing.
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return

    mesh = getattr(obj, "data", None)
    vertices = getattr(mesh, "vertices", None)
    if mesh is None or vertices is None:
        raise ValueError(f"object {getattr(obj, 'name', obj)!r} has no mesh")
    if len(mask) != len(vertices):
        raise ValueError(
            f"mask has {len(mask)} values but the mesh has {len(vertices)} vertices"
        )

    vertices.foreach_set("select", mask)

    # Edges and faces carry their own selection flags, and entering Edit Mode
    # flushes them *down* onto the vertices: leaving a fully selected bond
    # behind would re-select both its atoms the moment the user pressed Tab,
    # which silently undoes the selection that was just written. Setting them
    # from the vertices — a bond is selected exactly when both its atoms are —
    # is the same rule Blender itself applies, so the flush becomes a no-op.
    edges = mesh.edges
    if len(edges):
        flat = np.empty(len(edges) * 2, dtype=np.int32)
        edges.foreach_get("vertices", flat)
        pairs = flat.reshape(-1, 2)
        edges.foreach_set("select", mask[pairs[:, 0]] & mask[pairs[:, 1]])

    # Molecules have no faces, so this loop normally does not run at all.
    polygons = mesh.polygons
    if len(polygons):
        polygons.foreach_set(
            "select", [bool(mask[list(face.vertices)].all()) for face in polygons]
        )

    mesh.update()
