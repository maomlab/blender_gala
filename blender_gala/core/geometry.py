"""Geometry construction: dashes, arcs, curve objects and text objects.

Gala draws measurements and interactions as **real 3D geometry** rather than
viewport overlays (SPECIFICATION D-15), so a hydrogen bond receives light,
casts shadows, appears in cryptomatte and depth-sorts against the molecule like
anything else in the scene.

The maths lives in pure-numpy functions at the top of this module so it can be
tested without Blender; the ``bpy`` layer underneath only turns point lists
into data blocks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import collections as gala_collections

try:  # pragma: no cover
    import bpy
    from mathutils import Vector
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]
    Vector = None  # type: ignore[assignment,misc]

__all__ = [
    "arc_points",
    "billboard",
    "clear_of_occluders",
    "dash_segments",
    "dihedral_arc_points",
    "make_card",
    "make_curve",
    "make_line",
    "make_text",
]


# ---------------------------------------------------------------------------
# Pure geometry
# ---------------------------------------------------------------------------


def dash_segments(
    start: Any,
    end: Any,
    dash_length: float = 0.15,
    gap_length: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a segment into evenly distributed dashes.

    The dash length is adjusted so that a whole number of dashes spans the
    segment with a dash at each end. Truncating instead would leave a ragged
    stub against an atom, which reads as a rendering error in a figure.

    Parameters
    ----------
    start, end : sequence of float
        Segment endpoints, in the same units as ``dash_length``.
    dash_length : float, optional
        Desired dash length. Must be positive.
    gap_length : float, optional
        Desired gap between dashes. Zero produces a solid line.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        Sub-segment endpoints. A degenerate input segment yields an empty list.

    Raises
    ------
    ValueError
        If ``dash_length`` is not positive or ``gap_length`` is negative.
    """
    if dash_length <= 0:
        raise ValueError(f"dash_length must be positive, got {dash_length}")
    if gap_length < 0:
        raise ValueError(f"gap_length must not be negative, got {gap_length}")

    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    span = p1 - p0
    length = float(np.linalg.norm(span))
    if length <= 1e-12:
        return []
    if gap_length == 0.0:
        return [(p0, p1)]

    direction = span / length
    period = dash_length + gap_length
    n_dashes = max(1, round((length + gap_length) / period))
    if n_dashes == 1:
        return [(p0, p1)]

    actual_dash = (length - (n_dashes - 1) * gap_length) / n_dashes
    if actual_dash <= 0:
        # Requested gaps do not fit; fall back to a solid line rather than
        # emitting zero-length dashes.
        return [(p0, p1)]

    segments = []
    for i in range(n_dashes):
        offset = i * (actual_dash + gap_length)
        segments.append(
            (p0 + direction * offset, p0 + direction * (offset + actual_dash))
        )
    return segments


def arc_points(
    centre: Any,
    point_a: Any,
    point_b: Any,
    radius: float | None = None,
    resolution: int = 24,
) -> np.ndarray:
    """Sample the arc from ``point_a`` to ``point_b`` about ``centre``.

    Used to draw the angle between two rays of a measurement.

    Parameters
    ----------
    centre : sequence of float
        Arc centre (the vertex atom of an angle).
    point_a, point_b : sequence of float
        Points defining the two rays. Only their directions matter.
    radius : float, optional
        Arc radius. Defaults to 40 % of the shorter ray, which keeps the arc
        clear of both atoms.
    resolution : int, optional
        Number of sampled points. Minimum 2.

    Returns
    -------
    numpy.ndarray
        Shape ``(resolution, 3)``. Empty ``(0, 3)`` if the rays are degenerate
        or exactly collinear, where an arc is undefined.
    """
    c = np.asarray(centre, dtype=float)
    va = np.asarray(point_a, dtype=float) - c
    vb = np.asarray(point_b, dtype=float) - c
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return np.zeros((0, 3))

    ua = va / na
    ub = vb / nb
    cos_angle = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    angle = float(np.arccos(cos_angle))

    axis = np.cross(ua, ub)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return np.zeros((0, 3))
    axis = axis / axis_norm

    if radius is None:
        radius = 0.4 * min(na, nb)

    steps = max(2, int(resolution))
    thetas = np.linspace(0.0, angle, steps)
    # Rodrigues' rotation of ua about axis.
    cos_t = np.cos(thetas)[:, None]
    sin_t = np.sin(thetas)[:, None]
    rotated = (
        ua[None, :] * cos_t
        + np.cross(axis, ua)[None, :] * sin_t
        + axis[None, :] * float(np.dot(axis, ua)) * (1.0 - cos_t)
    )
    return c + rotated * radius


def dihedral_arc_points(
    point_a: Any,
    point_b: Any,
    point_c: Any,
    point_d: Any,
    resolution: int = 24,
) -> np.ndarray:
    """Sample the arc representing the dihedral A-B-C-D.

    The arc is drawn about the midpoint of the B-C bond, sweeping from the
    projection of A to the projection of D onto the plane normal to B-C. That
    is the standard Newman-projection reading of a torsion.

    Parameters
    ----------
    point_a, point_b, point_c, point_d : sequence of float
        The four atoms of the dihedral.
    resolution : int, optional
        Number of sampled points.

    Returns
    -------
    numpy.ndarray
        Shape ``(resolution, 3)``, or ``(0, 3)`` if the dihedral is degenerate.
    """
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)
    d = np.asarray(point_d, dtype=float)

    axis = c - b
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return np.zeros((0, 3))
    axis = axis / axis_norm
    centre = 0.5 * (b + c)

    # Components of the outer bonds perpendicular to the central axis.
    va = (a - b) - np.dot(a - b, axis) * axis
    vd = (d - c) - np.dot(d - c, axis) * axis
    if np.linalg.norm(va) <= 1e-12 or np.linalg.norm(vd) <= 1e-12:
        return np.zeros((0, 3))

    radius = 0.35 * min(float(np.linalg.norm(a - b)), float(np.linalg.norm(d - c)))
    return arc_points(
        centre, centre + va, centre + vd, radius=radius, resolution=resolution
    )


# ---------------------------------------------------------------------------
# Blender object construction
# ---------------------------------------------------------------------------


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def make_curve(
    name: str,
    polylines: Any,
    radius: float = 0.01,
    material: Any = None,
    collection: str = gala_collections.MEASUREMENTS,
    gala_type: str = "line",
    resolution: int = 6,
    **properties: Any,
) -> Any:
    """Create a curve object from one or more polylines.

    A curve with a non-zero ``bevel_depth`` renders as a tube in both EEVEE and
    Cycles and needs no mesh geometry, which keeps dashed bonds cheap even when
    there are hundreds of them.

    Parameters
    ----------
    name : str
        Object and data-block name.
    polylines : sequence of sequence of point
        Each polyline is a sequence of ``(x, y, z)`` points in world space,
        Blender units.
    radius : float, optional
        Tube radius in Blender units.
    material : bpy.types.Material, optional
        Material to assign.
    collection : str, optional
        Gala collection to link into.
    gala_type : str, optional
        Value for the ``gala_type`` custom property.
    resolution : int, optional
        Bevel resolution; 6 is smooth enough at figure scale.
    **properties
        Extra custom properties recorded on the object.

    Returns
    -------
    bpy.types.Object

    Raises
    ------
    ValueError
        If ``polylines`` contains no usable points.
    """
    bpy_mod = _require_bpy()

    usable = [np.asarray(line, dtype=float) for line in polylines]
    usable = [line for line in usable if line.ndim == 2 and line.shape[0] >= 2]
    if not usable:
        raise ValueError(f"{name!r}: no polyline with at least two points")

    curve = bpy_mod.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = resolution
    curve.fill_mode = "FULL"
    curve.use_fill_caps = True

    for line in usable:
        spline = curve.splines.new("POLY")
        spline.points.add(len(line) - 1)
        flat = np.hstack([line, np.ones((line.shape[0], 1))]).ravel()
        spline.points.foreach_set("co", flat)

    obj = bpy_mod.data.objects.new(name, curve)
    if material is not None:
        curve.materials.append(material)

    gala_collections.link_object(obj, collection)
    gala_collections.tag(obj, gala_type, **properties)
    return obj


def make_line(
    name: str,
    start: Any,
    end: Any,
    style: str = "dashed",
    radius: float = 0.01,
    dash_length: float = 0.15,
    gap_length: float = 0.1,
    material: Any = None,
    collection: str = gala_collections.MEASUREMENTS,
    gala_type: str = "line",
    **properties: Any,
) -> Any:
    """Create a solid or dashed line between two world-space points.

    Parameters
    ----------
    name : str
        Object name.
    start, end : sequence of float
        Endpoints in world space, Blender units.
    style : {"dashed", "solid"}, optional
        Dashed is the convention for non-covalent interactions.
    radius : float, optional
        Tube radius in Blender units.
    dash_length, gap_length : float, optional
        Dash pattern in Blender units. Ignored when ``style="solid"``.
    material : bpy.types.Material, optional
        Material to assign.
    collection : str, optional
        Gala collection to link into.
    gala_type : str, optional
        Value for the ``gala_type`` custom property.
    **properties
        Extra custom properties recorded on the object.

    Returns
    -------
    bpy.types.Object

    Raises
    ------
    ValueError
        If ``style`` is unknown or the endpoints coincide.
    """
    if style not in ("dashed", "solid"):
        raise ValueError(f"style must be 'dashed' or 'solid', got {style!r}")

    if style == "solid":
        polylines = [[np.asarray(start, dtype=float), np.asarray(end, dtype=float)]]
    else:
        polylines = [
            list(seg) for seg in dash_segments(start, end, dash_length, gap_length)
        ]

    if not polylines:
        raise ValueError(f"{name!r}: start and end coincide, nothing to draw")

    return make_curve(
        name,
        polylines,
        radius=radius,
        material=material,
        collection=collection,
        gala_type=gala_type,
        **properties,
    )


def clear_of_occluders(
    position: np.ndarray,
    clearance: float,
    extent: float = 0.0,
    scene: Any = None,
) -> tuple[np.ndarray, float]:
    """Move a label towards the camera until nothing is in front of it.

    A label anchored on an atom inside a protein is behind that protein from
    almost every angle, and a fixed offset does not help: the direction that
    clears the geometry depends on where the camera is. So the ray from the
    camera to the anchor is cast, and if anything is hit on the way the label
    moves to ``clearance`` in front of it.

    View dependent, deliberately. For a still it does what the eye wants; for
    an orbit, pass ``avoid_occlusion=False`` and place the labels yourself,
    since no single position is in front from every frame.

    Parameters
    ----------
    position : numpy.ndarray
        Where the label would go.
    clearance : float
        How far in front of the occluder to sit, in Blender units.
    extent : float, optional
        Half the size of the label. A label is a card, not a point, so its
        corners are tested as well as its centre — testing the centre alone
        leaves a label whose middle happens to see through a gap with its
        edges still buried.
    scene : bpy.types.Scene, optional
        Scene to cast in. Defaults to the active one.

    Returns
    -------
    tuple of (numpy.ndarray, float)
        Where to put the label, and what to multiply its size by. Moving a
        label along the ray it is already on leaves it at the same place on
        screen, but nearer the camera, so it is drawn bigger — the factor
        cancels that out, and the move becomes invisible except for the
        occluder it escaped. Unchanged and 1.0 when there is no camera or
        nothing in the way.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    camera = getattr(scene, "camera", None)
    if camera is None:
        return position, 1.0

    origin = np.array(camera.matrix_world.translation, dtype=float)
    delta = np.asarray(position, dtype=float) - origin
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return position, 1.0
    direction = delta / distance

    # The label's own plane, to spread the samples over: it faces the camera,
    # so its axes are the camera's.
    right = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)

    # A three by three grid over the card, not just its corners: a ribbon
    # crossing the middle of a label passes between four corner rays and
    # leaves the label buried with its corners in clear air.
    samples = [np.asarray(position, dtype=float)]
    if extent > 0.0:
        samples += [
            position + right * sx * extent + up * sy * extent
            for sx in (-1.0, 0.0, 1.0)
            for sy in (-1.0, 0.0, 1.0)
            if (sx, sy) != (0.0, 0.0)
        ]

    depsgraph = bpy_mod.context.evaluated_depsgraph_get()
    blocked_at = distance
    for sample in samples:
        towards = np.asarray(sample, dtype=float) - origin
        reach = float(np.linalg.norm(towards))
        if reach <= 1e-9:
            continue
        hit, location, *_ = scene.ray_cast(
            depsgraph,
            origin=origin,
            direction=towards / reach,
            distance=reach,
        )
        if hit:
            blocked_at = min(
                blocked_at,
                float(np.linalg.norm(np.array(location, dtype=float) - origin)),
            )
    moved_to = max(blocked_at - clearance, 1e-4)
    if moved_to >= distance:
        # Nothing between the camera and the label that it is not already in
        # front of.
        return position, 1.0
    return origin + direction * moved_to, moved_to / distance


def make_text(
    name: str,
    text: str,
    location: Any,
    size: float = 0.1,
    material: Any = None,
    align_x: str = "CENTER",
    align_y: str = "CENTER",
    extrude: float = 0.0,
    collection: str = gala_collections.LABELS,
    gala_type: str = "label",
    **properties: Any,
) -> Any:
    """Create a 3D text object at a world-space location.

    Parameters
    ----------
    name : str
        Object name.
    text : str
        Body text. Newlines are honoured.
    location : sequence of float
        World-space position in Blender units.
    size : float, optional
        Font size in Blender units.
    material : bpy.types.Material, optional
        Material to assign; use an emissive one so labels stay readable
        regardless of the lighting.
    align_x, align_y : str, optional
        Blender text alignment enums.
    extrude : float, optional
        Extrusion depth; ``0`` keeps the text flat.
    collection : str, optional
        Gala collection to link into.
    gala_type : str, optional
        Value for the ``gala_type`` custom property.
    **properties
        Extra custom properties recorded on the object.

    Returns
    -------
    bpy.types.Object
    """
    bpy_mod = _require_bpy()

    data = bpy_mod.data.curves.new(name, type="FONT")
    data.body = text
    data.size = size
    data.align_x = align_x
    data.align_y = align_y
    data.extrude = extrude

    obj = bpy_mod.data.objects.new(name, data)
    obj.location = tuple(float(v) for v in location)
    if material is not None:
        data.materials.append(material)

    gala_collections.link_object(obj, collection)
    gala_collections.tag(obj, gala_type, text=text, **properties)
    return obj


def billboard(obj: Any, camera: Any = None) -> Any:
    """Make ``obj`` always face the camera, squarely.

    Copies the camera's rotation rather than baking one in, so the label keeps
    facing the camera through an orbit animation.

    ``COPY_ROTATION`` rather than ``TRACK_TO``, which is the obvious choice and
    is wrong: tracking aims each label's +Z at the camera and then rolls it to
    put its +Y as near *world* +Y as it can. Every label is in a different
    place, so every label aims along a slightly different line and takes a
    different roll — a page of text each tipped a few degrees from the next,
    and none of them level with the frame unless the camera happens to be
    upright. Copying the rotation gives every label the camera's own axes, so
    they are square to the frame and parallel to each other.

    Parameters
    ----------
    obj : bpy.types.Object
        Object to constrain.
    camera : bpy.types.Object, optional
        Target camera. Defaults to the scene camera; a no-op if there is none.

    Returns
    -------
    bpy.types.Object
        The same object, for chaining.
    """
    bpy_mod = _require_bpy()
    camera = camera or bpy_mod.context.scene.camera
    if camera is None:
        return obj

    # TRACK_TO as well as COPY_ROTATION: a scene built by an earlier version
    # carries the tracking constraint, and the two would fight.
    for existing in list(obj.constraints):
        if existing.type in {"TRACK_TO", "COPY_ROTATION"}:
            obj.constraints.remove(existing)

    constraint = obj.constraints.new("COPY_ROTATION")
    constraint.target = camera
    return obj


def rounded_rectangle(
    half_width: float,
    half_height: float,
    corner: float = 0.35,
    segments: int = 6,
) -> list[tuple[float, float, float]]:
    """Return the outline of a rectangle with rounded corners, anticlockwise.

    Parameters
    ----------
    half_width, half_height : float
        Half extents in local X and Y.
    corner : float
        Corner radius as a fraction of the shorter half extent. ``0`` is a
        plain rectangle and ``1`` a pill — a full semicircle at each end.
    segments : int
        Straight segments per corner arc.

    Returns
    -------
    list of tuple
        Vertices in local space, at z = 0.
    """
    radius = max(0.0, min(corner, 1.0)) * min(half_width, half_height)
    if radius <= 0.0:
        return [
            (-half_width, -half_height, 0.0),
            (half_width, -half_height, 0.0),
            (half_width, half_height, 0.0),
            (-half_width, half_height, 0.0),
        ]

    inner_x = half_width - radius
    inner_y = half_height - radius
    points: list[tuple[float, float, float]] = []
    for centre_x, centre_y, start in (
        (inner_x, -inner_y, -math.pi / 2),
        (inner_x, inner_y, 0.0),
        (-inner_x, inner_y, math.pi / 2),
        (-inner_x, -inner_y, math.pi),
    ):
        for step in range(segments + 1):
            angle = start + (math.pi / 2) * (step / segments)
            points.append(
                (
                    centre_x + radius * math.cos(angle),
                    centre_y + radius * math.sin(angle),
                    0.0,
                )
            )
    return points


#: Backing for a number sitting over a molecule — an interaction's distance or
#: a measurement's value. Cool, dark and see-through, against the residue
#: cards' neutral near-black, and used with a pill shape: different tint and
#: different outline, so which kind of label you are looking at needs no
#: thought. White text on a pale molecule is otherwise unreadable.
LABEL_CARD_COLOUR = (0.02, 0.07, 0.12, 0.62)


def make_card(
    text_object: Any,
    colour: tuple[float, float, float, float],
    padding: float = 0.35,
    corner: float = 0.35,
    collection: str = gala_collections.LABELS,
    gala_type: str = "label_card",
    material_name: str = "GALA Label Card",
) -> Any | None:
    """Create a translucent rounded plane sized to sit behind a text object.

    Parameters
    ----------
    text_object : bpy.types.Object
        The ``FONT`` object to back. The card is parented to it, so it inherits
        the billboard constraint and stays behind the glyphs from any angle.
    colour : tuple of float
        Card RGBA. The alpha is what makes it a tint over the scene rather
        than a hole in it.
    padding : float
        Extra size around the text, as a fraction of its dimensions.
    corner : float
        Corner rounding, ``0`` square through ``1`` for a pill. See
        :func:`rounded_rectangle`.
    collection : str
        Gala collection to link into — the label's own, so that clearing that
        category takes its cards with it.
    gala_type : str
        Tag to apply.
    material_name : str
        Name for the card material.

    Returns
    -------
    bpy.types.Object or None
        ``None`` when the text has no dimensions to measure yet.
    """
    bpy_mod = _require_bpy()
    from ..scene import materials as gala_materials

    # Text dimensions are only valid once the depsgraph has evaluated the font.
    bpy_mod.context.view_layer.update()
    width, height, _ = text_object.dimensions
    if width <= 0 or height <= 0:
        return None

    half_w = width * (0.5 + padding)
    half_h = height * (0.5 + padding * 2.0)
    points = rounded_rectangle(half_w, half_h, corner)

    mesh = bpy_mod.data.meshes.new(f"{text_object.name} card")
    mesh.from_pydata(points, [], [list(range(len(points)))])
    mesh.update()

    card = bpy_mod.data.objects.new(f"{text_object.name} card", mesh)
    # Behind the text along the label's local -Z.
    card.location = (0.0, 0.0, -max(width, height) * 0.01)
    card.parent = text_object

    material = gala_materials.build_material(
        gala_materials.MATERIAL_PRESETS["label"].with_(
            base_color=colour,
            emission_strength=0.0,
            alpha=colour[3],
            use_attribute_color=False,
        ),
        name=material_name,
    )
    mesh.materials.append(material)

    gala_collections.link_object(card, collection)
    gala_collections.tag(card, gala_type)
    return card
