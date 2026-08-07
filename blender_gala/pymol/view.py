"""PyMOL's camera, and Blender's.

The two describe a camera almost identically — both look down their local
``-Z`` with ``+Y`` up — so the conversion is mostly bookkeeping about where
the numbers live and what units they are in.

The one thing worth being sure of is which way the stored 3x3 goes, since a
transposed rotation still produces a valid-looking camera pointing somewhere
mirrored. It was settled by experiment rather than from the documentation:
PyMOL was given an identity view and asked to render landmarks at ``+X``,
``+Y`` and ``+Z``, then the same with a 90 degree rotation about ``Y``. The
matrix whose **columns are the camera axes in world space** is the one that
predicts where they landed.

So, with ``M`` that matrix, ``o`` the origin of rotation and ``p`` the
position field:

* a world point maps to camera space as ``M.T @ (world - o) + p``;
* the camera therefore sits at ``o - M @ p``;
* and ``M`` is exactly the rotation part of a Blender camera's world matrix.

Ångström in, Blender units out: the position and clipping planes are scaled,
the rotation is not.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import units
from ..scene import camera as gala_camera
from .session import PymolView

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "camera_matrix",
    "camera_to_view",
    "view_from_matrix",
    "view_to_camera",
]


def camera_matrix(view: PymolView) -> np.ndarray:
    """Return the camera's world matrix, in ångström.

    Parameters
    ----------
    view : PymolView
        A view read from a session.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)``. Rotation columns are the camera's right, up and backward
        axes; the translation is the camera position in ångström.
    """
    rotation = np.asarray(view.rotation, dtype=float).reshape(3, 3)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(view.origin, dtype=float) - rotation @ np.asarray(
        view.position, dtype=float
    )
    return matrix


def view_from_matrix(
    matrix: Any,
    origin: Any = None,
    field_of_view: float = 20.0,
    orthoscopic: bool = False,
    near: float | None = None,
    far: float | None = None,
) -> PymolView:
    """Build a :class:`PymolView` from a camera world matrix in ångström.

    Parameters
    ----------
    matrix : array_like
        ``(4, 4)`` camera world matrix, translation in ångström.
    origin : array_like, optional
        Origin of rotation in ångström — what PyMOL turns around, and what
        ``near``/``far`` are measured relative to. Defaults to the point
        directly in front of the camera at the distance implied by the
        matrix, which for a camera that was framed on a molecule is the
        molecule.
    field_of_view : float, optional
        Vertical field of view in degrees.
    orthoscopic : bool, optional
        Whether the projection is orthographic.
    near, far : float, optional
        Clipping planes as distances from the camera, in ångström. Default to
        a symmetric bracket around the origin of rotation.

    Returns
    -------
    PymolView
    """
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    rotation = _orthonormal(matrix[:3, :3])
    position_world = matrix[:3, 3]

    if origin is None:
        # Nothing better to turn around than whatever the camera is looking
        # at; use its own distance from the world origin as the depth.
        distance = float(np.linalg.norm(position_world)) or 100.0
        centre = position_world - rotation[:, 2] * distance
    else:
        centre = np.asarray(origin, dtype=float).reshape(3)

    position = rotation.T @ (centre - position_world)
    distance = abs(float(position[2])) or 100.0
    return PymolView(
        rotation=rotation,
        position=position,
        origin=centre,
        near=float(distance * 0.5 if near is None else near),
        far=float(distance * 1.5 if far is None else far),
        field_of_view=float(field_of_view),
        orthoscopic=bool(orthoscopic),
    )


def view_to_camera(
    view: PymolView,
    camera: Any = None,
    scale: float = units.DEFAULT_WORLD_SCALE,
    scene: Any = None,
) -> Any:
    """Point a Blender camera the way PyMOL was pointing.

    Parameters
    ----------
    view : PymolView
        The view from a session.
    camera : bpy.types.Object, optional
        Camera to move. Created if omitted, and made the scene camera.
    scale : float, optional
        Blender units per ångström.
    scene : bpy.types.Scene, optional

    Returns
    -------
    bpy.types.Object
        The camera.
    """
    module = _require_bpy()
    scene = scene or module.context.scene

    if camera is None:
        camera = scene.camera
    if camera is None:
        data = module.data.cameras.new("Gala PyMOL Camera")
        camera = module.data.objects.new("Gala PyMOL Camera", data)
        scene.collection.objects.link(camera)
        scene.camera = camera

    matrix = camera_matrix(view)
    matrix[:3, 3] *= scale
    camera.matrix_world = _as_blender_matrix(matrix)

    data = camera.data
    # PyMOL quotes a *vertical* field of view, so the sensor has to be fit
    # vertically or the same number means a different framing.
    data.sensor_fit = "VERTICAL"
    if view.orthoscopic:
        data.type = "ORTHO"
        data.ortho_scale = (
            2.0 * view.distance * math.tan(math.radians(view.field_of_view) / 2.0)
        ) * scale
    else:
        data.type = "PERSP"
        data.angle = math.radians(view.field_of_view)

    # Blender will not accept a zero or negative near plane; PyMOL happily
    # stores one when the camera is inside the molecule.
    data.clip_start = max(view.near * scale, 1e-5)
    data.clip_end = max(view.far * scale, data.clip_start * 1.001)
    return camera


def camera_to_view(
    camera: Any = None,
    origin: Any = None,
    scale: float = units.DEFAULT_WORLD_SCALE,
    scene: Any = None,
) -> PymolView:
    """Read a Blender camera as a PyMOL view.

    Parameters
    ----------
    camera : bpy.types.Object, optional
        Defaults to the scene camera.
    origin : array_like, optional
        Origin of rotation in **ångström**; usually the centre of the subject.
    scale : float, optional
        Blender units per ångström.
    scene : bpy.types.Scene, optional

    Returns
    -------
    PymolView

    Raises
    ------
    ValueError
        If there is no camera to read.
    """
    module = _require_bpy()
    scene = scene or module.context.scene
    camera = camera or scene.camera
    if camera is None:
        raise ValueError("the scene has no camera to convert")

    matrix = np.array(camera.matrix_world).reshape(4, 4)
    matrix[:3, 3] /= scale

    data = camera.data
    orthoscopic = getattr(data, "type", "PERSP") == "ORTHO"
    if orthoscopic:
        # Recover the angle that frames the same height at the subject, so
        # switching PyMOL back to perspective keeps the composition.
        distance = (
            float(
                np.linalg.norm(
                    matrix[:3, 3]
                    - np.asarray(
                        origin if origin is not None else np.zeros(3), dtype=float
                    )
                )
            )
            or 100.0
        )
        # An orthographic frame is the same size at every depth, so the angle
        # that would frame the same height at the subject is what PyMOL needs
        # if it is switched back to perspective.
        half = gala_camera.visible_height(0.0, camera, scene) / scale / 2.0
        field_of_view = math.degrees(2.0 * math.atan2(half, distance))
    else:
        field_of_view = math.degrees(gala_camera.vertical_field_of_view(camera, scene))

    return view_from_matrix(
        matrix,
        origin=origin,
        field_of_view=field_of_view,
        orthoscopic=orthoscopic,
        near=data.clip_start / scale,
        far=data.clip_end / scale,
    )


def _orthonormal(rotation: np.ndarray) -> np.ndarray:
    """Strip any scale a camera object picked up, keeping the orientation."""
    columns = []
    for i in range(3):
        column = np.asarray(rotation[:, i], dtype=float)
        norm = float(np.linalg.norm(column))
        columns.append(column / norm if norm else np.eye(3)[i])
    return np.stack(columns, axis=1)


def _as_blender_matrix(matrix: np.ndarray) -> Any:
    from mathutils import Matrix

    return Matrix([[float(v) for v in row] for row in matrix])


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy
