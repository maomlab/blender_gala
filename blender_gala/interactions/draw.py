"""Turning detected interactions into scene geometry.

Each interaction becomes a dashed curve object (SPECIFICATION D-15), styled by
kind with colours that follow the conventions structural biologists already
read: yellow for hydrogen bonds, cyan for generic polar contacts, orange for
salt bridges, green for pi-stacking.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import collections as gala_collections
from ..core import geometry, units
from ..core.entity import AtomStructure
from ..scene import materials as gala_materials
from .detect import Interaction

__all__ = [
    "INTERACTION_STYLES",
    "InteractionStyle",
    "clear_interactions",
    "draw_interactions",
]


@dataclass(frozen=True)
class InteractionStyle:
    """How one kind of interaction is drawn.

    Distances are in **ångström** and converted at the Blender boundary.

    Attributes
    ----------
    colour : tuple[float, float, float]
        Line colour.
    radius : float
        Tube radius in ångström.
    dash_length : float
        Dash length in ångström.
    gap_length : float
        Gap between dashes in ångström. ``0`` gives a solid line.
    style : {"dashed", "solid"}
        Line style. Covalent-like interactions (metal coordination) read
        better solid; everything else is conventionally dashed.
    label : bool
        Whether to place a distance label at the midpoint by default.
    """

    colour: tuple[float, float, float] = (1.0, 0.9, 0.2)
    radius: float = 0.12
    dash_length: float = 0.4
    gap_length: float = 0.25
    style: str = "dashed"
    label: bool = False


INTERACTION_STYLES: dict[str, InteractionStyle] = {
    "hbond": InteractionStyle(colour=(1.0, 0.88, 0.2)),
    "polar": InteractionStyle(colour=(0.3, 0.9, 0.95)),
    "salt_bridge": InteractionStyle(colour=(1.0, 0.45, 0.1), radius=0.15),
    "hydrophobic": InteractionStyle(
        colour=(0.65, 0.65, 0.65), radius=0.08, dash_length=0.25, gap_length=0.35
    ),
    "pi_stacking": InteractionStyle(colour=(0.25, 0.85, 0.4), radius=0.14),
    "cation_pi": InteractionStyle(colour=(1.0, 0.6, 0.15), radius=0.14),
    "halogen": InteractionStyle(colour=(0.7, 0.4, 1.0)),
    "metal": InteractionStyle(colour=(0.6, 0.72, 0.95), style="solid", radius=0.1),
    "contact": InteractionStyle(colour=(0.9, 0.9, 0.9)),
}


def _style_material(kind: str, style: InteractionStyle) -> Any:
    """Get or build the emissive material for one interaction kind."""
    colour = (*style.colour, 1.0)
    spec = gala_materials.MATERIAL_PRESETS["interaction"].with_(
        base_color=colour, emission_color=colour
    )
    return gala_materials.build_material(spec, name=f"GALA Interaction {kind.title()}")


def _is_drawable(point_a: np.ndarray, point_b: np.ndarray) -> bool:
    """Whether a line can be built between two endpoints.

    Coincident endpoints have no direction to lay dashes along, and a
    non-finite one reaches Blender as a curve of ``nan`` control points, which
    it accepts and then draws as nothing at all.
    """
    if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
        return False
    return float(np.linalg.norm(point_b - point_a)) > 1e-12


def draw_interactions(
    interactions: Iterable[Interaction],
    target: Any = None,
    styles: dict[str, InteractionStyle] | None = None,
    label: bool | None = None,
    label_card: bool = True,
    label_template: str = "{distance:.1f}",
    label_size: float = 1.2,
    name_prefix: str = "GALA",
    scale: float | None = None,
) -> list[Any]:
    """Draw interactions as dashed lines, one object per interaction.

    Parameters
    ----------
    interactions : iterable of Interaction
        What to draw, typically the result of
        :func:`~blender_gala.interactions.detect.find_interactions`.
    target : AtomStructure, Molecule, or bpy.types.Object, optional
        Used only to read the world scale, so ångström-denominated style
        values convert correctly. Defaults to Molecular Nodes' 0.01.
    styles : dict[str, InteractionStyle], optional
        Override styling per kind. Missing kinds fall back to
        :data:`INTERACTION_STYLES`.
    label : bool, optional
        Force distance labels on or off. ``None`` uses each style's default.
    label_card : bool, optional
        Put a translucent pill behind each distance label. On by default: the
        text is white, and white on a pale molecule is unreadable.
    label_template : str, optional
        ``str.format`` template with ``distance``, ``angle``, ``kind`` and
        ``label`` available.
    label_size : float, optional
        Label text size in ångström.
    name_prefix : str, optional
        Prefix for created object names.
    scale : float, optional
        Explicit ångström-to-Blender-unit scale, overriding ``target``.

    Returns
    -------
    list[bpy.types.Object]
        Every object created, lines and labels together.

    Warns
    -----
    UserWarning
        If an interaction has nowhere to draw a line — endpoints that coincide
        or are not finite. Duplicate ATOM records and unmerged altlocs put two
        atoms of different residues at the same coordinates, and
        :func:`~blender_gala.interactions.detect.atom_contacts` reports the
        pair. Those are skipped: half a figure and a traceback is worse than
        the rest of the figure and a warning.
    """
    if scale is None:
        scale = (
            AtomStructure.from_any(target).world_scale
            if target is not None
            else units.DEFAULT_WORLD_SCALE
        )

    merged = dict(INTERACTION_STYLES)
    if styles:
        merged.update(styles)

    created: list[Any] = []
    counters: dict[str, int] = {}
    undrawable: list[str] = []

    for interaction in interactions:
        point_a = np.asarray(interaction.point_a, dtype=float)
        point_b = np.asarray(interaction.point_b, dtype=float)
        if not _is_drawable(point_a, point_b):
            undrawable.append(interaction.label or str(interaction))
            continue

        style = merged.get(interaction.kind, INTERACTION_STYLES["contact"])
        index = counters.get(interaction.kind, 0)
        counters[interaction.kind] = index + 1
        name = f"{name_prefix} {interaction.kind} {index:03d}"

        line = geometry.make_line(
            name,
            point_a,
            point_b,
            style=style.style,
            radius=style.radius * scale,
            dash_length=style.dash_length * scale,
            gap_length=style.gap_length * scale,
            material=_style_material(interaction.kind, style),
            collection=gala_collections.INTERACTIONS,
            gala_type=f"interaction_{interaction.kind}",
            interaction_kind=interaction.kind,
            distance=interaction.distance,
            description=interaction.label,
        )
        created.append(line)

        wants_label = style.label if label is None else label
        if not wants_label:
            continue

        text = label_template.format(
            distance=interaction.distance,
            angle=interaction.angle if interaction.angle is not None else float("nan"),
            kind=interaction.kind,
            label=interaction.label,
        )
        midpoint = 0.5 * (point_a + point_b)
        text_object = geometry.make_text(
            f"{name} label",
            text,
            midpoint,
            size=label_size * scale,
            material=gala_materials.get_material("label"),
            collection=gala_collections.INTERACTIONS,
            gala_type=f"interaction_label_{interaction.kind}",
        )
        geometry.billboard(text_object)
        created.append(text_object)

        # A pill behind the number. White text on a pale protein is unreadable,
        # and a distance is the one label that has to be legible or it is just
        # decoration. Pill-shaped and cooler than the residue cards, so the two
        # kinds of label are told apart at a glance rather than read.
        if label_card:
            card = geometry.make_card(
                text_object,
                geometry.LABEL_CARD_COLOUR,
                padding=0.3,
                corner=1.0,
                collection=gala_collections.INTERACTIONS,
                gala_type=f"interaction_label_{interaction.kind}",
                material_name="GALA Interaction Label Card",
            )
            if card is not None:
                created.append(card)

    if undrawable:
        listed = ", ".join(undrawable[:3])
        if len(undrawable) > 3:
            listed += f", and {len(undrawable) - 3} more"
        warnings.warn(
            f"{len(undrawable)} interaction(s) have no line to draw — their "
            f"endpoints coincide or are not finite — and were skipped: "
            f"{listed}. Two atoms at identical coordinates usually means "
            "duplicate ATOM records or unmerged altlocs.",
            stacklevel=2,
        )

    return created


def clear_interactions(kind: str | None = None) -> int:
    """Remove drawn interactions.

    Parameters
    ----------
    kind : str, optional
        Remove only one kind, e.g. ``"hbond"``. ``None`` removes all of them.

    Returns
    -------
    int
        Number of objects removed.
    """
    if kind is None:
        return gala_collections.clear(gala_collections.INTERACTIONS)
    removed = gala_collections.clear(
        gala_collections.INTERACTIONS, gala_type=f"interaction_{kind}"
    )
    removed += gala_collections.clear(
        gala_collections.INTERACTIONS, gala_type=f"interaction_label_{kind}"
    )
    return removed
