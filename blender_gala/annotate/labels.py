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


def _clear_of_occluders(
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
            placed, magnification = _clear_of_occluders(
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
