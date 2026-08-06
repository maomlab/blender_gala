"""Camera creation and framing.

Molecular Nodes' ``Camera`` helper adjusts an existing camera. Gala adds the
part that matters when you have just imported a structure and there is either
no camera or one pointing at nothing: put a camera somewhere sensible and frame
the molecule so it fills the frame with a predictable margin.

"Fills the frame" means the atoms as they project from the chosen viewpoint.
Molecules are not spherical, and fitting the sphere around one — the usual
shortcut, and what this did first — leaves it filling under half the frame.
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


def _half_fov_tangents(camera_data: Any, scene: Any) -> tuple[float, float]:
    """Return ``(horizontal, vertical)`` half-field-of-view tangents.

    Both axes, rather than the narrower one, because a molecule is not round:
    which axis it runs out of room in depends on how it is turned.
    """
    render = scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y
    aspect = (width / height) if height else 1.0
    lens = max(camera_data.lens, 1e-6)

    sensor_fit = camera_data.sensor_fit
    if sensor_fit == "VERTICAL" or (sensor_fit == "AUTO" and height > width):
        sensor = (
            camera_data.sensor_height
            if sensor_fit == "VERTICAL"
            else camera_data.sensor_width
        )
        tan_v = sensor / (2.0 * lens)
        tan_h = tan_v * aspect
    else:  # HORIZONTAL, or AUTO where the width is the larger dimension
        tan_h = camera_data.sensor_width / (2.0 * lens)
        tan_v = tan_h / aspect if aspect else tan_h

    return max(tan_h, 1e-6), max(tan_v, 1e-6)


def _fit_distance(radius: float, camera_data: Any, scene: Any, margin: float) -> float:
    """Distance at which a sphere of ``radius`` fits the frame.

    The fallback for targets whose individual points are not available. A
    sphere is tangent to the frustum rather than spanning it, hence ``sin``.
    """
    tan_h, tan_v = _half_fov_tangents(camera_data, scene)
    half_fov = math.atan(min(tan_h, tan_v))
    return float(radius * margin / math.sin(max(half_fov, 1e-3)))


def _fit_to_points(
    points: np.ndarray,
    centre: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    camera_data: Any,
    scene: Any,
    margin: float,
) -> tuple[np.ndarray, float]:
    """Where to aim, and from how far, to fit every point in the frame.

    Fitting the bounding *sphere* is what a camera helper usually does, and it
    wastes most of the frame: the sphere circumscribes the molecule, so the
    silhouette only touches it where the single most distant atom is. On the
    vignette structures that left the molecule filling under half the width.

    This projects the atoms onto the camera's own axes instead and solves for
    the distance at which the extreme one lands on the frame edge. A point at
    depth ``z`` beyond the aim point is inside the horizontal field of view
    while ``|x| <= (d + z) * tan(h)``, so ``d = max(|x| / tan(h) - z)`` over
    every point and both axes. Perspective is in the ``- z``: atoms nearer the
    camera need more room than atoms behind it.

    The aim point is the middle of the silhouette rather than the centroid.
    They are the same for a symmetrical molecule and far apart for one with a
    long tail on one side, where aiming at the centroid leaves a band of empty
    frame down the other — and, since the fit is to the *worst* offset from the
    aim point, a wider shot than the molecule needs.

    Returns
    -------
    tuple of (numpy.ndarray, float)
        The point to look at, and the distance to look from.
    """
    right, up, forward = basis
    tan_h, tan_v = _half_fov_tangents(camera_data, scene)

    relative = np.asarray(points, dtype=float) - centre
    x = relative @ right
    y = relative @ up
    depth = relative @ forward

    # Recentre across the view axes only: shifting the aim point sideways does
    # not change how deep anything is, so `depth` still applies.
    x_middle = 0.5 * (float(x.max()) + float(x.min()))
    y_middle = 0.5 * (float(y.max()) + float(y.min()))
    aim = centre + right * x_middle + up * y_middle

    half_width = np.abs(x - x_middle) * margin
    half_height = np.abs(y - y_middle) * margin
    distance = float(
        (np.maximum(half_width / tan_h, half_height / tan_v) - depth).max()
    )
    return aim, distance


def frame_target(
    target: Any = None,
    viewpoint: str | Sequence[float] = "iso",
    margin: float = 1.15,
    selection: str | None = None,
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
        Extra room around the molecule, as a multiple of its projected extent.
        ``1.0`` puts the outermost atom exactly on the frame edge; the default
        leaves a little air so atoms do not touch it.
    selection : str, optional
        Frame only these atoms, which is how you close in on a binding site
        without hiding the rest of the protein. Everything else stays in the
        scene and is simply allowed out of frame.
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
    EmptySelectionError
        If ``selection`` matches no atoms.
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

    # The viewing direction does not depend on how far away the camera ends up,
    # so it is fixed first and the distance solved for along it.
    az = math.radians(azimuth)
    el = math.radians(elevation)
    unit = np.array(
        [
            math.cos(el) * math.sin(az),
            -math.cos(el) * math.cos(az),
            math.sin(el),
        ]
    )

    # The camera's own axes, taken from the orientation it will be given, so
    # the projection below is the one the render will use.
    orientation = Vector((-unit).tolist()).to_track_quat("-Z", "Y")
    basis = (
        np.array(orientation @ Vector((1.0, 0.0, 0.0))),  # right
        np.array(orientation @ Vector((0.0, 1.0, 0.0))),  # up
        np.array(orientation @ Vector((0.0, 0.0, -1.0))),  # forward
    )

    points = _subject_points(target, scene, selection)
    if points is None or len(points) == 0:
        distance = _fit_distance(radius, camera.data, scene, margin)
    else:
        # Recomputed from the points actually being framed, which is not the
        # whole molecule when a selection narrows it.
        centre = points.mean(axis=0)
        radius = max(float(np.linalg.norm(points - centre, axis=1).max()), 1e-4)
        centre, distance = _fit_to_points(
            points, centre, basis, camera.data, scene, margin
        )
        # Framing the silhouette can put the camera inside a molecule seen
        # end-on, where the extent across the view is small and the extent
        # along it is not.
        distance = max(distance, radius * 1.25)

    offset = unit * distance
    camera.location = tuple(float(v) for v in (centre + offset))
    camera.rotation_euler = orientation.to_euler()

    camera.data.clip_start = max(1e-4, distance - radius * 4.0)
    camera.data.clip_end = distance + radius * 8.0

    bpy_mod.context.view_layer.update()
    return camera


def _target_bounds(target: Any, scene: Any) -> tuple[np.ndarray, float]:
    from .lighting import _subject_bounds

    return _subject_bounds(target, scene)


def _subject_points(
    target: Any, scene: Any, selection: str | None = None
) -> np.ndarray | None:
    """World-space points to fit in the frame, or ``None`` if there are none.

    Atom positions when the target is a molecule, and bounding-box corners
    otherwise: a box is still a much closer fit than the sphere around it.
    """
    from ..core.entity import AtomStructure
    from ..core.exceptions import EmptySelectionError

    if target is not None:
        try:
            structure = AtomStructure.from_any(target)
        except Exception:
            structure = None
        if structure is not None and structure.n_atoms and structure.object is not None:
            positions = structure.world_positions()
            if selection is None:
                return positions
            mask = structure.select(selection)
            if not mask.any():
                raise EmptySelectionError(
                    f"cannot frame {selection!r}: it matches no atoms"
                )
            return positions[mask]

    if selection is not None:
        raise TypeError("framing a selection needs a molecule to select from")

    if bpy is None:
        return None

    if isinstance(target, bpy.types.Object):
        objects = [target]
    elif target is None:
        objects = [
            obj
            for obj in scene.objects
            if obj.type == "MESH" and not obj.get("gala") and obj.visible_get()
        ]
    else:
        return None

    corners = [
        np.array(obj.matrix_world @ Vector(corner))
        for obj in objects
        for corner in obj.bound_box
    ]
    return np.asarray(corners) if corners else None


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
    """Return an action's F-curves.

    Blender 5 removed ``Action.fcurves``, moving the curves into
    ``layers > strips > channelbags``, so they have to be gathered rather than
    read off the action.
    """
    curves: list[Any] = []
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                curves.extend(channelbag.fcurves)
    return curves
