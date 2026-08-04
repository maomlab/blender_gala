"""Scene organisation for everything Gala creates.

Gala puts every object it creates under a top-level ``Gala`` collection with
one child per category (SPECIFICATION D-17). That single decision buys a lot:
a user can hide all measurements with one checkbox, a render can exclude
labels via a view layer, and ``clear()`` operations never have to guess which
objects were Gala's.

Objects additionally carry a ``gala_type`` custom property so individual
objects remain identifiable after a user has moved them.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "INTERACTIONS",
    "LABELS",
    "LIGHTING",
    "MEASUREMENTS",
    "ROOT",
    "clear",
    "get_collection",
    "iter_tagged",
    "link_object",
    "tag",
]

ROOT = "Gala"
INTERACTIONS = "Gala Interactions"
MEASUREMENTS = "Gala Measurements"
LABELS = "Gala Labels"
LIGHTING = "Gala Lighting"

_CATEGORIES = (INTERACTIONS, MEASUREMENTS, LABELS, LIGHTING)


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def get_collection(name: str = ROOT, scene: Any = None) -> Any:
    """Return the named Gala collection, creating it and its parent if needed.

    Parameters
    ----------
    name : str, optional
        One of the module-level category constants, or any name. Category
        collections are parented to :data:`ROOT`; :data:`ROOT` is parented to
        the scene.
    scene : bpy.types.Scene, optional
        Scene to create the collection in. Defaults to the active scene.

    Returns
    -------
    bpy.types.Collection
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    collection = bpy_mod.data.collections.get(name)
    if collection is None:
        collection = bpy_mod.data.collections.new(name)

    parent = scene.collection if name == ROOT else get_collection(ROOT, scene)

    if collection.name not in parent.children:
        # Unlink from anywhere else first so re-running never nests twice.
        for other in bpy_mod.data.collections:
            if collection.name in other.children:
                other.children.unlink(collection)
        if (
            collection.name in scene.collection.children
            and parent is not scene.collection
        ):
            scene.collection.children.unlink(collection)
        parent.children.link(collection)

    return collection


def link_object(obj: Any, category: str = ROOT, scene: Any = None) -> Any:
    """Link ``obj`` into a Gala collection and unlink it from all others.

    Parameters
    ----------
    obj : bpy.types.Object
        Object to place.
    category : str, optional
        Target collection name.
    scene : bpy.types.Scene, optional
        Scene to work in.

    Returns
    -------
    bpy.types.Object
        The same object, for chaining.
    """
    collection = get_collection(category, scene)
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def tag(obj: Any, gala_type: str, **properties: Any) -> Any:
    """Mark ``obj`` as Gala-created and record what kind of thing it is.

    Parameters
    ----------
    obj : bpy.types.Object
        Object to tag.
    gala_type : str
        Category such as ``"hbond"``, ``"distance"``, ``"label"``, ``"key_light"``.
    **properties
        Extra custom properties to store, e.g. the measured value.

    Returns
    -------
    bpy.types.Object
        The same object, for chaining.
    """
    obj["gala"] = True
    obj["gala_type"] = gala_type
    for key, value in properties.items():
        obj[key] = value
    return obj


def iter_tagged(gala_type: str | None = None, scene: Any = None) -> list[Any]:
    """Return every Gala-tagged object, optionally filtered by type.

    Parameters
    ----------
    gala_type : str, optional
        Restrict to a single ``gala_type``.
    scene : bpy.types.Scene, optional
        Scene to search. Defaults to the active scene.

    Returns
    -------
    list of bpy.types.Object
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    found = []
    for obj in scene.objects:
        if not obj.get("gala"):
            continue
        if gala_type is not None and obj.get("gala_type") != gala_type:
            continue
        found.append(obj)
    return found


def clear(
    category: str | None = None, gala_type: str | None = None, scene: Any = None
) -> int:
    """Delete Gala-created objects.

    Parameters
    ----------
    category : str, optional
        Limit the deletion to one collection, e.g. :data:`MEASUREMENTS`.
    gala_type : str, optional
        Limit the deletion to one ``gala_type``.
    scene : bpy.types.Scene, optional
        Scene to work in.

    Returns
    -------
    int
        Number of objects removed.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    targets = []
    if category is not None:
        collection = bpy_mod.data.collections.get(category)
        if collection is None:
            return 0
        candidates = list(collection.objects)
    else:
        candidates = [
            obj
            for name in (*_CATEGORIES, ROOT)
            for obj in getattr(bpy_mod.data.collections.get(name), "objects", [])
        ]

    for obj in candidates:
        if gala_type is not None and obj.get("gala_type") != gala_type:
            continue
        targets.append(obj)

    removed = 0
    for obj in targets:
        data = obj.data
        bpy_mod.data.objects.remove(obj, do_unlink=True)
        removed += 1
        # Orphaned data blocks otherwise accumulate across re-runs.
        if data is not None and getattr(data, "users", 1) == 0:
            for library in (
                bpy_mod.data.curves,
                bpy_mod.data.meshes,
                bpy_mod.data.lights,
            ):
                try:
                    library.remove(data)
                except (TypeError, ReferenceError):
                    continue
                else:
                    break
    return removed
