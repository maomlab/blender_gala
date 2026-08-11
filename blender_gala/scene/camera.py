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

__all__ = [
    "VIEWPOINTS",
    "ensure_camera",
    "frame_target",
    "orbit",
    "vertical_field_of_view",
    "visible_height",
]

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


def vertical_field_of_view(camera: Any = None, scene: Any = None) -> float:
    """The camera's vertical field of view in radians, as actually rendered.

    Not ``camera.data.angle_y``, which is the angle the sensor *height*
    subtends and is the rendered field of view only when the sensor is fit
    vertically. Blender's default fit is ``AUTO``, where the sensor width
    spans the larger image dimension instead.

    Returns
    -------
    float
        Radians.

    Raises
    ------
    ValueError
        If there is no camera.
    """
    module = _require_bpy()
    scene = scene or module.context.scene
    camera = camera or scene.camera
    if camera is None:
        raise ValueError("the scene has no camera")
    return float(2.0 * math.atan(_half_fov_tangents(camera.data, scene)[1]))


def visible_height(distance: float, camera: Any = None, scene: Any = None) -> float:
    """How tall the frame is, in Blender units, ``distance`` from the camera.

    What a figure needs to size anything by how big it will *look* rather than
    how big it is: text at a fixed size in ångström is legible on a whole
    protein and covers the frame in a close-up, because the two are the same
    number of ångström and a very different number of pixels.

    Parameters
    ----------
    distance : float
        Distance in front of the camera, in Blender units. Ignored by an
        orthographic camera, whose frame is the same size at every depth.
    camera : bpy.types.Object, optional
        Defaults to the scene camera.
    scene : bpy.types.Scene, optional

    Returns
    -------
    float
        Frame height in Blender units.

    Raises
    ------
    ValueError
        If there is no camera.
    """
    module = _require_bpy()
    scene = scene or module.context.scene
    camera = camera or scene.camera
    if camera is None:
        raise ValueError("the scene has no camera")

    data = camera.data
    if getattr(data, "type", "PERSP") == "ORTHO":
        render = scene.render
        width = render.resolution_x * render.pixel_aspect_x
        height = render.resolution_y * render.pixel_aspect_y
        fit = getattr(data, "sensor_fit", "AUTO")
        if fit == "VERTICAL" or (fit == "AUTO" and height > width):
            return float(data.ortho_scale)
        return float(data.ortho_scale * (height / width if width else 1.0))

    return float(2.0 * abs(distance) * _half_fov_tangents(data, scene)[1])


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
        If ``viewpoint`` is neither a known name nor a pair of angles, or if
        ``margin`` is not positive.
    EmptySelectionError
        If ``selection`` matches no atoms.
    GalaError
        If the subject has non-finite coordinates.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    camera = camera or ensure_camera(scene)

    # The fit divides by the margin, so zero puts the camera at the aim point —
    # inside the molecule — and a negative one turns the fit inside out, where
    # asking for more room gives less of it.
    if not margin > 0:
        raise ValueError(f"margin must be positive, got {margin}")

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
        # along it is not. What decides that is the subject's depth *along the
        # view*, not its overall radius: two molecules side by side have a
        # bounding sphere far larger than anything the camera could end up
        # inside, and backing off by its radius leaves a figure that does not
        # fill its frame.
        depth = float(np.ptp(points @ basis[2]))
        distance = max(distance, depth * 0.75)

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


def _placed(points: np.ndarray) -> np.ndarray:
    """Return the points that are real positions, refusing if none are.

    A single ``nan`` makes the min and max of the whole set ``nan``, which
    becomes a camera transform of ``nan`` and a render that is black for no
    stated reason; ``inf`` arrives at the same place, since ``inf - inf`` is
    ``nan``. Dropping those points rather than refusing them is what the
    selection language and :meth:`AtomStructure.bounding_sphere` already do
    with a missing coordinate, and for the same reason: a structure read from
    one state of a multi-state session carries ``nan`` for the atoms that
    state does not contain, and framing the atoms it *does* contain is the
    useful answer.
    """
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        from ..core.exceptions import GalaError

        raise GalaError(
            f"none of the {len(points)} points to frame has a finite position, "
            "so there is no extent to fit. Check the structure for nan or "
            "infinite atom coordinates."
        )
    return finite


def _subject_points(
    target: Any, scene: Any, selection: str | None = None
) -> np.ndarray | None:
    """World-space points to fit in the frame, or ``None`` if there are none.

    Atom positions when the target is a molecule; the vertices of every
    visible mesh when there is no target, which is what makes framing a scene
    of several molecules as tight as framing one.

    Raises
    ------
    GalaError
        If any point is not finite. See :func:`_placed`.
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
                return _placed(positions)
            mask = structure.select(selection)
            if not mask.any():
                raise EmptySelectionError(
                    f"cannot frame {selection!r}: it matches no atoms"
                )
            return _placed(positions[mask])

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

    gathered: list[np.ndarray] = [
        points
        for points in (_object_points(obj) for obj in objects)
        if points is not None and len(points)
    ]
    return _placed(np.vstack(gathered)) if gathered else None


#: Vertices above this and an object is sampled rather than read whole. Framing
#: is a min and a max; a hundred thousand points do not decide it any better
#: than ten thousand do.
_POINT_LIMIT = 10000


def _object_points(obj: Any) -> np.ndarray | None:
    """World-space points describing one object's extent.

    Its vertices where it has them, its bounding box otherwise. The corners of
    a box around a globular molecule stick out well past the molecule — far
    enough that framing on them leaves a visibly loose figure — so they are
    the fallback, not the rule.

    The vertices are read from the object *as evaluated*, which is the only
    reading that survives geometry nodes. A modifier that builds a membrane
    from a grid, or instances a subunit onto the transforms of a biological
    assembly, leaves the original mesh holding a handful of points — in the
    assembly's case, all of them at the origin. Framing on those points aims
    the camera at the middle of a shell 130 Å across and renders nothing at
    all, which is a hard failure to read backwards from a black image.
    """
    evaluated = _evaluated_points(obj)
    if evaluated is not None:
        return evaluated

    mesh = getattr(obj, "data", None)
    vertices = getattr(mesh, "vertices", None)
    if vertices is None or len(vertices) == 0:
        return np.asarray(
            [np.array(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
        )

    return _mesh_points(mesh, np.array(obj.matrix_world).reshape(4, 4), _POINT_LIMIT)


#: The 26 directions of a 3x3x3 neighbourhood, normalised. Used to reduce an
#: instanced mesh to the few vertices that can possibly decide its extent.
_SUPPORT_DIRECTIONS = np.array(
    [
        (x, y, z)
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
        for z in (-1.0, 0.0, 1.0)
        if (x, y, z) != (0.0, 0.0, 0.0)
    ]
)
_SUPPORT_DIRECTIONS /= np.linalg.norm(_SUPPORT_DIRECTIONS, axis=1, keepdims=True)


def _local_vertices(mesh: Any) -> np.ndarray | None:
    """Vertex coordinates of ``mesh`` in its own space, or ``None`` if it has none."""
    vertices = getattr(mesh, "vertices", None)
    if vertices is None or len(vertices) == 0:
        return None
    flat = np.empty(len(vertices) * 3, dtype=np.float32)
    vertices.foreach_get("co", flat)
    return flat.reshape(-1, 3).astype(float)


def _support_points(local: np.ndarray) -> np.ndarray:
    """The handful of vertices that bound ``local`` from every direction.

    Thinning a mesh by taking every *n*-th vertex is the wrong thinning for a
    bounding box: it drops the extremes along with everything else, and the
    frame it decides is too tight — which on a capsid means the shell being
    cropped by the edge of the picture. The vertex furthest along each of 26
    directions cannot be dropped, and 26 of them describe a globular subunit
    closely enough that the box around them is the box around the mesh.
    """
    if len(local) <= len(_SUPPORT_DIRECTIONS):
        return local
    extremes = (local @ _SUPPORT_DIRECTIONS.T).argmax(axis=0)
    return local[np.unique(extremes)]


def _transform(local: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Carry local-space points through a 4x4 matrix."""
    homogeneous = np.hstack([local, np.ones((local.shape[0], 1))])
    return (homogeneous @ matrix.T)[:, :3]


def _mesh_points(mesh: Any, matrix: np.ndarray, limit: int) -> np.ndarray | None:
    """World-space vertices of ``mesh``, thinned to at most ``limit`` of them."""
    local = _local_vertices(mesh)
    if local is None:
        return None
    if limit and len(local) > limit:
        local = local[:: len(local) // limit + 1]
    return _transform(local, matrix)


def _evaluated_points(obj: Any) -> np.ndarray | None:
    """World-space points of what ``obj`` actually draws, instances included.

    Returns ``None`` when the depsgraph is unavailable or the evaluated object
    turns out to hold nothing, so the caller can fall back to reading the mesh
    as it is stored.
    """
    if bpy is None:  # pragma: no cover - only reachable outside Blender
        return None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except (AttributeError, RuntimeError):  # pragma: no cover - no depsgraph
        return None
    if depsgraph is None:  # pragma: no cover - a context with no scene in it
        return None

    # Instances are read through their source mesh rather than one at a time: a
    # capsid is sixty copies of one surface and a crowded scene is thousands of
    # copies of a handful, so the vertices are reduced to support points once
    # per distinct mesh and only those are carried through each matrix.
    cache: dict[int, np.ndarray] = {}
    gathered: list[np.ndarray] = []
    for instance in depsgraph.object_instances:
        if not (
            instance.is_instance
            and instance.parent is not None
            and instance.parent.original == obj
        ):
            continue
        mesh = getattr(instance.object, "data", None)
        if mesh is None:
            continue
        key = mesh.as_pointer()
        if key not in cache:
            local = _local_vertices(mesh)
            cache[key] = np.empty((0, 3)) if local is None else _support_points(local)
        local = cache[key]
        if len(local):
            gathered.append(
                _transform(local, np.array(instance.matrix_world).reshape(4, 4))
            )

    # Whatever the object draws in its own right, which for a modifier that
    # only instances is nothing, and for one that builds a mesh is everything.
    own = _mesh_points(
        getattr(obj.evaluated_get(depsgraph), "data", None),
        np.array(obj.matrix_world).reshape(4, 4),
        _POINT_LIMIT,
    )
    if own is not None:
        gathered.append(own)

    return np.vstack(gathered) if gathered else None


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

    Raises
    ------
    ValueError
        If ``frames`` is not positive.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    # Before anything is created: an orbit of no frames puts both keyframes on
    # the same frame and collapses the scene range to 0..0, which looks like a
    # scene that was never set up rather than an argument that was refused.
    if frames < 1:
        raise ValueError(f"frames must be positive, got {frames}")

    camera = camera or ensure_camera(scene)

    centre, _ = _target_bounds(target, scene)

    pivot = bpy_mod.data.objects.get("GALA Camera Pivot")
    if pivot is None:
        pivot = bpy_mod.data.objects.new("GALA Camera Pivot", None)
        pivot.empty_display_type = "PLAIN_AXES"
        scene.collection.objects.link(pivot)

    pivot.location = tuple(float(v) for v in centre)
    pivot.rotation_euler = (0.0, 0.0, 0.0)

    # `matrix_world` is a depsgraph result, not a property of the assignment
    # above: read without flushing it, the pivot is still at the world origin
    # and the parent inverse comes back as the identity, which displaces the
    # camera by the whole target centre. The displacement then appears at
    # whatever the *next* update happens to be, long after this call.
    scene.view_layers[0].update()

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
