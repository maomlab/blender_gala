"""Labelling atoms, residues and interactions.

Two mechanisms, because they solve different problems (SPECIFICATION D-16):

* :func:`label` creates a real 3D ``FONT`` object — an "in-scene card". It
  occludes correctly behind the molecule, appears in cryptomatte, and can be
  moved by hand when a label lands somewhere awkward.
* :func:`label_hud` registers a Molecular Nodes annotation, drawn as a
  resolution-independent 2D overlay. Right for a title, a scale note, or
  anything that should never be hidden by geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core import collections as gala_collections
from ..core import geometry
from ..core import mn as mn_bridge
from ..core.entity import AtomStructure
from ..core.exceptions import EmptySelectionError
from ..scene import materials as gala_materials

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "DEFAULT_TEMPLATE",
    "clear_labels",
    "label",
    "label_atoms",
    "label_hud",
    "label_residues",
]

#: One-letter code plus residue number, e.g. ``"R45"``. Compact enough that a
#: binding site can be labelled without the labels colliding.
DEFAULT_TEMPLATE = "{one}{resi}"


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def label(
    target: Any,
    selection: Any = "all",
    text: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    level: str = "residue",
    anchor: str = "centroid",
    style: str = "text",
    size: float = 2.0,
    offset: Sequence[float] | float = 2.0,
    avoid_occlusion: bool = True,
    colour: tuple[float, float, float] = (0.95, 0.95, 0.95),
    card_colour: tuple[float, float, float, float] = (0.05, 0.05, 0.08, 0.75),
    billboard: bool = True,
    extrude: float = 0.0,
    scale: float | None = None,
) -> list[Any]:
    """Label the atoms or residues matched by a selection.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to label.
    selection : str or array, optional
        What to label.
    text : str, optional
        Literal text for a single label. Overrides ``template`` and produces
        exactly one label at the selection's anchor point.
    template : str, optional
        ``str.format`` template evaluated per label. Fields: ``chain``,
        ``resi``, ``resn``, ``one``, ``name``, ``elem``, ``b``, ``q``,
        ``index``.
    level : {"residue", "atom", "selection"}, optional
        ``"residue"`` places one label per residue — the usual choice for a
        binding site. ``"atom"`` labels every matched atom. ``"selection"``
        places a single label for the whole selection.
    anchor : {"centroid", "first", "ca"}, optional
        Where a residue's label sits: at the residue centroid, at its first
        matched atom, or at its C-alpha.
    style : {"text", "card"}, optional
        ``"card"`` adds a translucent backing plane so the label stays legible
        over a busy molecular surface.
    size : float, optional
        Text size in ångström.
    avoid_occlusion : bool, optional
        Move each label towards the camera until nothing is in front of it.
        A label on an atom inside a protein is otherwise hidden by that
        protein, and which way clears it depends on where the camera is.
        View dependent: turn it off for an orbit.
    offset : sequence of float or float, optional
        Offset from the anchor in ångström. A scalar lifts the label along
        ``+Z``; a 3-sequence offsets in all axes.
    colour : tuple[float, float, float], optional
        Text colour.
    card_colour : tuple[float, float, float, float], optional
        Backing card RGBA, used when ``style="card"``.
    billboard : bool, optional
        Keep labels facing the camera.
    extrude : float, optional
        Text extrusion in ångström; a little depth catches the key light.
    scale : float, optional
        Explicit ångström-to-Blender-unit scale.

    Returns
    -------
    list[bpy.types.Object]
        The label objects created, cards included.

    Raises
    ------
    ValueError
        If ``level``, ``anchor`` or ``style`` is unknown.
    EmptySelectionError
        If the selection matched no atoms.
    """
    if level not in ("residue", "atom", "selection"):
        raise ValueError(
            f"level must be 'residue', 'atom' or 'selection', got {level!r}"
        )
    if anchor not in ("centroid", "first", "ca"):
        raise ValueError(f"anchor must be 'centroid', 'first' or 'ca', got {anchor!r}")
    if style not in ("text", "card"):
        raise ValueError(f"style must be 'text' or 'card', got {style!r}")

    structure = AtomStructure.from_any(target)
    if scale is None:
        scale = structure.world_scale

    indices = structure.indices(selection)
    if indices.size == 0:
        raise EmptySelectionError(f"selection matched no atoms in {structure.name!r}")

    positions = structure.world_positions()
    if isinstance(offset, (int, float)):
        offset_vector = np.array([0.0, 0.0, float(offset)]) * scale
    else:
        offset_vector = np.asarray(offset, dtype=float) * scale

    entries = _label_entries(
        structure, indices, positions, text, template, level, anchor
    )

    text_material = gala_materials.build_material(
        gala_materials.MATERIAL_PRESETS["label"].with_(
            base_color=(*colour, 1.0), emission_color=(*colour, 1.0)
        ),
        name="GALA Label Text",
    )

    clearance = float(np.linalg.norm(offset_vector)) or 2.0 * scale

    created: list[Any] = []
    for body, position in entries:
        placed = np.asarray(position) + offset_vector
        magnification = 1.0
        if avoid_occlusion:
            placed, magnification = geometry.clear_of_occluders(
                placed, clearance, extent=0.7 * size * scale * len(body) ** 0.5
            )
        obj = geometry.make_text(
            f"GALA Label {body}",
            body,
            placed,
            size=size * scale * magnification,
            material=text_material,
            extrude=extrude * scale,
            collection=gala_collections.LABELS,
            gala_type="label",
        )
        # Where the label points, as opposed to where it ended up: the offset
        # and any occlusion avoidance have already moved it off its atom, so
        # the object's own location no longer says what it is labelling.
        # Exporting the scene needs the anchor, not the resting place.
        obj["gala_anchor"] = [float(v) for v in position]
        # Which molecule it labels, as opposed to which molecule it is nearest.
        # Two superposed conformations — the ordinary comparison figure — leave
        # no signal in the scene at all: identical coordinates, identical
        # transforms, and an anchor exactly zero from both. Writing the name
        # down here is the only thing that can tell them apart later, and
        # `save_session` reads it in preference to guessing from the geometry.
        if structure.object is not None:
            obj["gala_molecule"] = structure.object.name
        created.append(obj)

        if style == "card":
            card = geometry.make_card(obj, card_colour, padding=0.35, corner=0.35)
            if card is not None:
                created.append(card)

        if billboard:
            geometry.billboard(obj)

    return created


def _label_entries(
    structure: AtomStructure,
    indices: np.ndarray,
    positions: np.ndarray,
    text: str | None,
    template: str,
    level: str,
    anchor: str,
) -> list[tuple[str, np.ndarray]]:
    """Resolve a selection into ``(text, world position)`` pairs."""
    if level == "selection":
        body = (
            text
            if text is not None
            else structure.atom_label(int(indices[0]), template)
        )
        return [(body, positions[indices].mean(axis=0))]

    if level == "atom":
        return [
            (
                text if text is not None else structure.atom_label(int(i), template),
                positions[i],
            )
            for i in indices
        ]

    residues = structure.context.residue_key
    atom_names = structure.context.upper("atom_name")

    entries: list[tuple[str, np.ndarray]] = []
    for residue in dict.fromkeys(residues[indices]):  # preserve first-seen order
        member = indices[residues[indices] == residue]
        if member.size == 0:
            continue

        if anchor == "centroid":
            position = positions[member].mean(axis=0)
        elif anchor == "ca":
            alpha = member[atom_names[member] == "CA"]
            position = (
                positions[alpha[0]] if alpha.size else positions[member].mean(axis=0)
            )
        else:
            position = positions[member[0]]

        body = (
            text if text is not None else structure.atom_label(int(member[0]), template)
        )
        entries.append((body, position))
    return entries


def label_residues(
    target: Any,
    selection: Any = "all",
    template: str = "{resn}{resi}",
    **kwargs: Any,
) -> list[Any]:
    """Label one text object per residue. Shorthand for ``label(level="residue")``.

    Returns
    -------
    list[bpy.types.Object]
    """
    return label(target, selection, template=template, level="residue", **kwargs)


def label_atoms(
    target: Any,
    selection: Any = "all",
    template: str = "{name}",
    **kwargs: Any,
) -> list[Any]:
    """Label every matched atom. Shorthand for ``label(level="atom")``.

    Returns
    -------
    list[bpy.types.Object]
    """
    return label(target, selection, template=template, level="atom", **kwargs)


def label_hud(
    target: Any,
    text: str,
    location: tuple[float, float] = (0.05, 0.95),
    size: int = 24,
    colour: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Any:
    """Add a 2D overlay label through Molecular Nodes' annotation system.

    Drawn in screen space and composited over the render, so it is never
    occluded and never changes size with the camera. Use for titles and
    captions; use :func:`label` for anything that must sit *in* the scene.

    Parameters
    ----------
    target : Molecule or AtomStructure
        The molecule whose annotation manager receives the label.
    text : str
        Label text.
    location : tuple[float, float], optional
        Normalised viewport coordinates, ``(0, 0)`` bottom-left.
    size : int, optional
        Text size in pixels.
    colour : tuple[float, float, float, float], optional
        Text RGBA.

    Returns
    -------
    The Molecular Nodes annotation interface, whose properties can be edited
    afterwards.

    Raises
    ------
    MolecularNodesUnavailable
        If Molecular Nodes is not installed.
    TypeError
        If the target has no annotation manager.
    """
    mn_bridge.require_mn()

    molecule = target
    if isinstance(target, AtomStructure):
        molecule = target.molecule
    if molecule is not None and mn_bridge.is_molecule(molecule):
        # A molecule, but from a build older than the 4.5 Gala requires: the
        # annotation manager is what sets that floor. Nothing stops an old
        # Molecular Nodes being pinned under a supported Blender, so say which
        # version is wanted rather than fail obscurely.
        if not hasattr(molecule, "annotations"):
            installed = mn_bridge.version()
            named = (
                "Molecular Nodes " + ".".join(str(part) for part in installed)
                if installed
                else "This Molecular Nodes build"
            )
            raise TypeError(
                f"{named} has no annotation manager, so it cannot draw a 2D "
                "overlay. The manager arrived in Molecular Nodes 4.5, which "
                "needs Blender 5.1 or newer. Use label() for an in-scene 3D "
                "label instead."
            )
    else:
        raise TypeError(
            "label_hud needs a Molecular Nodes entity with an annotation "
            "manager. Pass the Molecule returned by mn.Molecule.load(), or use "
            "label() for an in-scene 3D label."
        )

    annotation = molecule.annotations.add_label_2d(text=text, location=location)
    annotation.text_size = size
    annotation.text_color = colour
    return annotation


def clear_labels() -> int:
    """Remove every 3D label Gala created.

    Returns
    -------
    int
        Number of objects removed.
    """
    return gala_collections.clear(gala_collections.LABELS)
