"""Camera creation and framing.

Molecular Nodes' ``Camera`` helper adjusts an existing camera. Gala adds the
part that matters when you have just imported a structure and there is either
no camera or one pointing at nothing: put a camera somewhere sensible and frame
the molecule so it fills the frame with a predictable margin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

try:  # pragma: no cover
    import bpy
    from mathutils import Vector
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]
    Vector = None  # type: ignore[assignment,misc]

__all__ = ["VIEWPOINTS", "ensure_camera", "frame_target", "orbit"]

#: Named viewpoints, as ``(azimuth, elevation)`` in degrees. Azimuth is
#: measured about ``+Z`` from ``-Y``, matching :mod:`blender_gala.scene.lighting`.
VIEWPOINTS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 89.9),
    "bottom": (0.0, -89.9),
    "iso": (35.0, 20.0),
    "default": (35.0, 20.0),
}


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def ensure_camera(scene: Any = None, lens: float = 85.0) -> Any:
    """Return the scene camera, creating one if there is none.

    Parameters
    ----------
    scene : bpy.types.Scene, optional
        Scene to work in.
    lens : float, optional
        Focal length in mm for a newly created camera. 85 mm is a mild
        telephoto: it flattens perspective, which keeps a molecule's proportions
        honest, and is the usual choice for product-style renders.

    Returns
    -------
    bpy.types.Object
        The camera object.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    if scene.camera is not None:
        return scene.camera

    existing = next((o for o in scene.objects if o.type == "CAMERA"), None)
    if existing is not None:
        scene.camera = existing
        return existing

    data = bpy_mod.data.cameras.new("GALA Camera")
    data.lens = lens
    data.clip_start = 0.001
    data.clip_end = 1000.0
    obj = bpy_mod.data.objects.new("GALA Camera", data)
    scene.collection.objects.link(obj)
    scene.camera = obj
    return obj


def _fit_distance(radius: float, camera_data: Any, scene: Any, margin: float) -> float:
    """Distance at which a sphere of ``radius`` fits the frame."""
    render = scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y

    sensor_fit = camera_data.sensor_fit
    if sensor_fit == "VERTICAL":
        sensor = camera_data.sensor_height
        extent = height
    elif sensor_fit == "HORIZONTAL":
        sensor = camera_data.sensor_width
        extent = width
    else:  # AUTO: the larger image dimension uses the sensor width
        sensor = camera_data.sensor_width
        extent = max(width, height)

    # Half-angle of the *narrower* of the two field-of-view axes, so the
    # molecule fits in both directions rather than being cropped in one.
    aspect = min(width, height) / extent if extent else 1.0
    half_fov = math.atan((sensor * aspect) / (2.0 * camera_data.lens))
    half_fov = max(half_fov, 1e-3)
    return float(radius * margin / math.sin(half_fov))


def frame_target(
    target: Any = None,
    viewpoint: str | Sequence[float] = "iso",
    margin: float = 1.15,
    camera: Any = None,
    scene: Any = None,
) -> Any:
    """Point the camera at a molecule and back off until it fits.

    Parameters
    ----------
    target : AtomStructure, Molecule, bpy.types.Object, or None, optional
        What to frame. ``None`` frames every visible mesh.
    viewpoint : str or sequence of float, optional
        A key of :data:`VIEWPOINTS`, or an ``(azimuth, elevation)`` pair in
        degrees.
    margin : float, optional
        Extra room around the molecule. ``1.0`` is a tight fit; the default
        leaves a little air so atoms do not touch the frame edge.
    camera : bpy.types.Object, optional
        Camera to move. Defaults to the scene camera, created if absent.
    scene : bpy.types.Scene, optional
        Scene to work in.

    Returns
    -------
    bpy.types.Object
        The camera object.

    Raises
    ------
    ValueError
        If ``viewpoint`` is neither a known name nor a pair of angles.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    camera = camera or ensure_camera(scene)

    if isinstance(viewpoint, str):
        if viewpoint not in VIEWPOINTS:
            raise ValueError(
                f"unknown viewpoint {viewpoint!r}; choose from {sorted(VIEWPOINTS)} "
                "or pass an (azimuth, elevation) pair"
            )
        azimuth, elevation = VIEWPOINTS[viewpoint]
    else:
        angles = list(viewpoint)
        if len(angles) != 2:
            raise ValueError(
                f"viewpoint must be a name or an (azimuth, elevation) pair, "
                f"got {viewpoint!r}"
            )
        azimuth, elevation = float(angles[0]), float(angles[1])

    centre, radius = _target_bounds(target, scene)
    distance = _fit_distance(radius, camera.data, scene, margin)

    az = math.radians(azimuth)
    el = math.radians(elevation)
    horizontal = distance * math.cos(el)
    offset = np.array(
        [
            horizontal * math.sin(az),
            -horizontal * math.cos(az),
            distance * math.sin(el),
        ]
    )

    camera.location = tuple(float(v) for v in (centre + offset))
    direction = Vector((-offset).tolist())
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    camera.data.clip_start = max(1e-4, distance - radius * 4.0)
    camera.data.clip_end = distance + radius * 8.0

    bpy_mod.context.view_layer.update()
    return camera


def _target_bounds(target: Any, scene: Any) -> tuple[np.ndarray, float]:
    from .lighting import _subject_bounds

    return _subject_bounds(target, scene)


def orbit(
    frames: int = 120,
    target: Any = None,
    camera: Any = None,
    scene: Any = None,
) -> Any:
    """Animate a full turntable orbit around a target.

    The camera is parented to an empty at the target centre and the empty is
    rotated, so the framing computed by :func:`frame_target` is preserved for
    every frame.

    Parameters
    ----------
    frames : int, optional
        Length of one full revolution.
    target : AtomStructure, Molecule, or bpy.types.Object, optional
        Centre of the orbit.
    camera : bpy.types.Object, optional
        Camera to orbit.
    scene : bpy.types.Scene, optional
        Scene to work in.

    Returns
    -------
    bpy.types.Object
        The pivot empty, keyframed over ``frames``.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    camera = camera or ensure_camera(scene)

    centre, _ = _target_bounds(target, scene)

    pivot = bpy_mod.data.objects.get("GALA Camera Pivot")
    if pivot is None:
        pivot = bpy_mod.data.objects.new("GALA Camera Pivot", None)
        pivot.empty_display_type = "PLAIN_AXES"
        scene.collection.objects.link(pivot)

    pivot.location = tuple(float(v) for v in centre)
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    camera.parent = pivot
    camera.matrix_parent_inverse = pivot.matrix_world.inverted()

    scene.frame_start = 1
    scene.frame_end = frames
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.keyframe_insert("rotation_euler", frame=1)
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert("rotation_euler", frame=frames + 1)

    for fcurve in _action_fcurves(pivot.animation_data.action):
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

    return pivot


def _action_fcurves(action: Any) -> list[Any]:
    """Return an action's F-curves on both the legacy and slotted APIs.

    Blender 4.4 introduced slotted actions and Blender 5 removed
    ``Action.fcurves`` outright, moving the curves into
    ``layers > strips > channelbags``.
    """
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)

    curves: list[Any] = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                curves.extend(channelbag.fcurves)
    return curves
