"""Set an object's origin to its molecular geometry.

Blender rotates and scales about the object origin. A molecule imported at its
crystallographic coordinates can have an origin hundreds of ångström away from
any atom, so orbiting the view or animating a turntable swings the molecule
through a huge arc instead of spinning it in place. Fixing the origin is the
single most useful thing to do to a freshly imported structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import chemistry
from ..core.entity import AtomStructure

try:  # pragma: no cover
    import bpy
    from mathutils import Vector
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]
    Vector = None  # type: ignore[assignment,misc]

__all__ = ["ORIGIN_METHODS", "geometry_centre", "set_origin_to_geometry"]

ORIGIN_METHODS = ("centroid", "mass", "bounds")


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def geometry_centre(
    positions: np.ndarray,
    method: str = "centroid",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a centre for a point cloud.

    Parameters
    ----------
    positions : numpy.ndarray
        Shape ``(n, 3)``.
    method : {"centroid", "mass", "bounds"}, optional
        ``"centroid"`` is the unweighted mean — the usual choice, and what
        PyMOL's ``origin`` does. ``"mass"`` weights by atomic mass, which
        differs meaningfully only for structures with heavy atoms.
        ``"bounds"`` uses the bounding-box centre, which is what you want when
        framing an elongated molecule such as DNA.
    weights : numpy.ndarray, optional
        Per-point weights for ``method="mass"``.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``positions`` is empty, or mass weighting was
        requested without usable weights.
    """
    if method not in ORIGIN_METHODS:
        raise ValueError(f"method must be one of {ORIGIN_METHODS}, got {method!r}")

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("positions must be a non-empty (n, 3) array")

    if method == "centroid":
        return points.mean(axis=0)
    if method == "bounds":
        return 0.5 * (points.min(axis=0) + points.max(axis=0))

    if weights is None:
        raise ValueError("method='mass' requires per-atom weights")
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("mass weights sum to zero")
    return (points * w[:, None]).sum(axis=0) / total


def _atom_masses(structure: AtomStructure) -> np.ndarray | None:
    array = structure.array
    stored = getattr(array, "mass", None)
    if stored is not None:
        masses = np.asarray(stored, dtype=float)
        if masses.sum() > 0:
            return masses

    elements = getattr(array, "element", None)
    if elements is None:
        return None
    lookup = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "P": 30.974,
        "S": 32.06,
        "SE": 78.97,
        "F": 18.998,
        "CL": 35.45,
        "BR": 79.904,
        "I": 126.90,
        "NA": 22.990,
        "MG": 24.305,
        "K": 39.098,
        "CA": 40.078,
        "MN": 54.938,
        "FE": 55.845,
        "CO": 58.933,
        "NI": 58.693,
        "CU": 63.546,
        "ZN": 65.38,
    }
    symbols = chemistry.normalise_element(np.asarray(elements))
    return np.array([lookup.get(s, 12.011) for s in symbols], dtype=float)


def set_origin_to_geometry(
    target: Any,
    method: str = "centroid",
    selection: str = "all",
    move_to_world_origin: bool = False,
) -> np.ndarray:
    """Move an object's origin onto its molecular geometry, in place.

    The geometry does not move in world space: the mesh data is shifted by the
    negated centre and the object transform is shifted by the same amount in
    the opposite direction.

    Parameters
    ----------
    target : AtomStructure, molecularnodes.Molecule, or bpy.types.Object
        The object whose origin to change.
    method : {"centroid", "mass", "bounds"}, optional
        See :func:`geometry_centre`.
    selection : str, optional
        Restrict the centre calculation to a subset, e.g. ``"chain A"`` to
        pivot a complex about one chain, or ``"not solvent"`` to ignore waters.
    move_to_world_origin : bool, optional
        Additionally place the object at the world origin. This is what you
        want before setting up a camera and light rig, and is why
        :func:`~blender_gala.scene.setup.publication_setup` passes ``True``.

    Returns
    -------
    numpy.ndarray
        The centre that was used, in world-space Blender units.

    Raises
    ------
    StructureError
        If ``target`` has no Blender object to move.
    """
    bpy_mod = _require_bpy()

    if (
        bpy is not None
        and isinstance(target, bpy.types.Object)
        and target.type != "MESH"
    ):
        obj = target
        structure = None
    else:
        structure = AtomStructure.from_any(target)
        obj = structure.object

    if obj is None:
        from ..core.exceptions import StructureError

        raise StructureError(
            "set_origin_to_geometry needs a Blender object; the structure was "
            "loaded without one."
        )

    if structure is not None:
        mask = structure.select(selection)
        if not mask.any():
            from ..core.exceptions import EmptySelectionError

            raise EmptySelectionError(
                f"selection {selection!r} matched no atoms in {obj.name!r}"
            )
        local = structure.local_positions()[mask]
        weights = None
        if method == "mass":
            masses = _atom_masses(structure)
            weights = masses[mask] if masses is not None else None
    else:  # pragma: no cover - non-molecular objects
        local = _mesh_positions(obj)
        weights = None

    centre_local = geometry_centre(local, method=method, weights=weights)

    _shift_mesh(obj, -centre_local)
    offset = obj.matrix_world.to_3x3() @ Vector(centre_local.tolist())
    obj.matrix_world.translation = obj.matrix_world.translation + offset

    if move_to_world_origin:
        obj.matrix_world.translation = Vector((0.0, 0.0, 0.0))

    if obj.data is not None:
        obj.data.update_tag()
    bpy_mod.context.view_layer.update()
    return np.array(obj.matrix_world.translation)


def _mesh_positions(obj: Any) -> np.ndarray:
    vertices = obj.data.vertices
    flat = np.empty(len(vertices) * 3, dtype=np.float32)
    vertices.foreach_get("co", flat)
    return flat.reshape(-1, 3).astype(float)


def _shift_mesh(obj: Any, delta: np.ndarray) -> None:
    """Translate every vertex of ``obj`` by ``delta`` in local space."""
    vertices = obj.data.vertices
    count = len(vertices)
    if count == 0:
        return
    flat = np.empty(count * 3, dtype=np.float32)
    vertices.foreach_get("co", flat)
    flat = flat.reshape(-1, 3) + np.asarray(delta, dtype=np.float32)
    vertices.foreach_set("co", flat.ravel())
