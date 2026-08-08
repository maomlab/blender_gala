"""Drawing measurements into the scene.

A distance is a dashed line with the value at its midpoint. An angle adds an
arc between its two rays. A dihedral draws the three bonds plus the arc that a
Newman projection would show, which is the only representation that makes the
sign of a torsion legible.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core import collections as gala_collections
from ..core import geometry
from ..core.entity import AtomStructure
from ..scene import camera as gala_camera
from ..scene import materials as gala_materials
from .measurements import Measurement

__all__ = ["clear_measurements", "draw_measurement"]

_COUNTER: dict[str, int] = {}


def draw_measurement(
    measurement: Measurement,
    target: Any = None,
    colour: tuple[float, float, float] = (1.0, 0.85, 0.2),
    radius: float = 0.1,
    dash_length: float = 0.4,
    gap_length: float = 0.25,
    style: str = "dashed",
    label: bool = True,
    label_template: str | None = None,
    label_size: float | None = 1.5,
    label_offset: float | Sequence[float] = 0.5,
    label_card: bool = True,
    label_avoid_occlusion: bool = True,
    arc: bool = True,
    scale: float | None = None,
) -> list[Any]:
    """Draw a measurement and return the objects created.

    Distances are in **ångström**; they are converted using the structure's
    world scale.

    Parameters
    ----------
    measurement : Measurement
        The measurement to draw.
    target : AtomStructure, Molecule, or bpy.types.Object, optional
        Used to read the world scale.
    colour : tuple[float, float, float], optional
        Line and label colour.
    radius : float, optional
        Tube radius in ångström.
    dash_length, gap_length : float, optional
        Dash pattern in ångström.
    style : {"dashed", "solid"}, optional
        Line style.
    label : bool, optional
        Place a text label with the value.
    label_template : str, optional
        ``str.format`` template with ``value``, ``text``, ``unit`` and ``kind``
        available. Defaults to the formatted value, e.g. ``"2.85 A"``.
    label_size : float or None, optional
        Text size in ångström. ``None`` sizes it to the *frame* instead —
        :data:`LABEL_FRAME_FRACTION` of the visible height at the label's
        depth — so a value reads the same whether the camera is on a whole
        ribosome or two ångström from a hydrogen bond. A fixed size in
        ångström cannot do both: it is legible on one and covers the frame on
        the other.
    label_card : bool, optional
        Put a translucent pill behind the value, so white text stays readable
        over a pale molecule.
    label_offset : float or sequence of float, optional
        Offset from the line, in ångström. A scalar lifts the label along
        ``+Z``; a 3-sequence offsets in all axes, which is how a value is
        moved *across the frame* rather than up it when two measurements
        would otherwise label the same patch of screen.
    label_avoid_occlusion : bool, optional
        Move the value towards the camera until nothing is in front of it. The
        atoms worth measuring between are usually inside the molecule, and a
        label at the midpoint is then behind the geometry that surrounds them.
        View dependent: turn it off for an orbit.
    arc : bool, optional
        Draw the arc for angles and dihedrals.
    scale : float, optional
        Explicit ångström-to-Blender-unit scale.

    Returns
    -------
    list[bpy.types.Object]
    """
    structure = AtomStructure.from_any(target) if target is not None else None
    if scale is None:
        scale = structure.world_scale if structure is not None else 0.01

    index = _COUNTER.get(measurement.kind, 0)
    _COUNTER[measurement.kind] = index + 1
    name = f"GALA {measurement.kind} {index:03d}"

    material = gala_materials.build_material(
        gala_materials.MATERIAL_PRESETS["measurement"].with_(
            base_color=(*colour, 1.0), emission_color=(*colour, 1.0)
        ),
        name="GALA Measurement",
    )

    points = np.asarray(measurement.points, dtype=float)
    created: list[Any] = []

    line_kwargs = {
        "style": style,
        "radius": radius * scale,
        "dash_length": dash_length * scale,
        "gap_length": gap_length * scale,
        "material": material,
        "collection": gala_collections.MEASUREMENTS,
        "gala_type": f"measurement_{measurement.kind}",
        "value": measurement.value,
        "unit": measurement.unit,
    }

    for segment, (start, end) in enumerate(itertools.pairwise(points)):
        created.append(
            geometry.make_line(f"{name} seg{segment}", start, end, **line_kwargs)
        )

    if created:
        # The measured points, on the first segment only. A measurement is
        # drawn as one object per segment plus a label, so the geometry alone
        # no longer says which points were picked — and anything that wants to
        # rebuild the measurement later (exporting the scene to PyMOL, for
        # one) needs exactly that. Blender units, like everything on an object.
        created[0]["gala_points"] = [float(v) for v in points.reshape(-1)]

    if arc and measurement.kind == "angle":
        arc_points = geometry.arc_points(points[1], points[0], points[2])
        if len(arc_points) >= 2:
            created.append(
                geometry.make_curve(
                    f"{name} arc",
                    [arc_points],
                    radius=radius * scale * 0.7,
                    material=material,
                    collection=gala_collections.MEASUREMENTS,
                    gala_type="measurement_arc",
                )
            )
    elif arc and measurement.kind == "dihedral":
        arc_points = geometry.dihedral_arc_points(*points)
        if len(arc_points) >= 2:
            created.append(
                geometry.make_curve(
                    f"{name} arc",
                    [arc_points],
                    radius=radius * scale * 0.7,
                    material=material,
                    collection=gala_collections.MEASUREMENTS,
                    gala_type="measurement_arc",
                )
            )

    if label:
        text = (
            measurement.text
            if label_template is None
            else label_template.format(
                value=measurement.value,
                text=measurement.text,
                unit=measurement.unit,
                kind=measurement.kind,
            )
        )
        if isinstance(label_offset, (int, float)):
            offset_vector = np.array([0.0, 0.0, float(label_offset)]) * scale
        else:
            offset_vector = np.asarray(label_offset, dtype=float) * scale
        anchor = _label_anchor(measurement, points) + offset_vector
        if label_size is None:
            label_size = _frame_relative_size(anchor, scale)
        magnification = 1.0
        if label_avoid_occlusion:
            anchor, magnification = geometry.clear_of_occluders(
                anchor,
                float(np.linalg.norm(offset_vector)) or 2.0 * scale,
                extent=0.7 * label_size * scale * len(text) ** 0.5,
            )
        text_object = geometry.make_text(
            f"{name} label",
            text,
            anchor,
            size=label_size * scale * magnification,
            # Tinted like the line it belongs to. Two measurements in one
            # figure are told apart by colour, and a value in the default
            # white belongs to neither of them.
            material=gala_materials.build_material(
                gala_materials.MATERIAL_PRESETS["label"].with_(
                    base_color=(*colour, 1.0), emission_color=(*colour, 1.0)
                ),
                name="GALA Measurement Label "
                + "".join(f"{round(channel * 255):02x}" for channel in colour),
            ),
            collection=gala_collections.MEASUREMENTS,
            gala_type=f"measurement_label_{measurement.kind}",
            value=measurement.value,
        )
        geometry.billboard(text_object)
        created.append(text_object)

        if label_card:
            card = geometry.make_card(
                text_object,
                geometry.LABEL_CARD_COLOUR,
                padding=0.3,
                corner=1.0,
                collection=gala_collections.MEASUREMENTS,
                gala_type=f"measurement_label_{measurement.kind}",
                material_name="GALA Measurement Label Card",
            )
            if card is not None:
                created.append(card)

    return created


#: How much of the frame's height a frame-relative label should occupy. Small
#: enough not to sit on top of the structure, big enough to read at the sizes
#: a figure is printed and viewed at.
LABEL_FRAME_FRACTION = 0.035


def _frame_relative_size(anchor: Any, scale: float) -> float:
    """Text size in ångström that fills the same share of the frame anywhere.

    Measured at the label's own depth, so a value near the camera and one
    further back come out the same size on screen rather than the same size
    in ångström.
    """
    try:
        import bpy
    except ImportError:  # pragma: no cover - Blender only
        return 1.5

    scene = getattr(bpy.context, "scene", None)
    camera = scene.camera if scene is not None else None
    if camera is None:
        # Nothing to be relative to; the fixed default is as good a guess as
        # any and is what the caller would have got anyway.
        return 1.5

    matrix = np.array(camera.matrix_world).reshape(4, 4)
    towards = np.asarray(anchor, dtype=float) - matrix[:3, 3]
    # Depth along the view axis rather than straight-line distance: that is
    # what the projection divides by.
    depth = abs(float(np.dot(towards, -matrix[:3, 2])))
    height = gala_camera.visible_height(depth, camera, scene)
    return float(height * LABEL_FRAME_FRACTION / scale)


def _label_anchor(measurement: Measurement, points: np.ndarray) -> np.ndarray:
    """Where the value label sits for each measurement kind."""
    if measurement.kind == "distance":
        return points.mean(axis=0)
    if measurement.kind == "angle":
        # Just outside the arc, along the bisector of the two rays.
        vertex = points[1]
        rays = [
            (p - vertex) / max(np.linalg.norm(p - vertex), 1e-12)
            for p in (points[0], points[2])
        ]
        bisector = rays[0] + rays[1]
        norm = float(np.linalg.norm(bisector))
        if norm < 1e-9:
            return vertex
        span = min(
            float(np.linalg.norm(points[0] - vertex)),
            float(np.linalg.norm(points[2] - vertex)),
        )
        return vertex + bisector / norm * span * 0.55
    # Dihedral: the midpoint of the central bond.
    return 0.5 * (points[1] + points[2])


def clear_measurements(kind: str | None = None) -> int:
    """Remove drawn measurements.

    Parameters
    ----------
    kind : str, optional
        Remove only ``"distance"``, ``"angle"`` or ``"dihedral"``. ``None``
        removes every measurement.

    Returns
    -------
    int
        Number of objects removed.
    """
    if kind is None:
        _COUNTER.clear()
        return gala_collections.clear(gala_collections.MEASUREMENTS)
    removed = gala_collections.clear(
        gala_collections.MEASUREMENTS, gala_type=f"measurement_{kind}"
    )
    removed += gala_collections.clear(
        gala_collections.MEASUREMENTS, gala_type=f"measurement_label_{kind}"
    )
    _COUNTER.pop(kind, None)
    return removed
