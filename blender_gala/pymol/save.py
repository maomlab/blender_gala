"""Writing a Blender scene out as a PyMOL session.

The reverse of :mod:`blender_gala.pymol.load`, and lossier in the direction
you would expect: Blender can express things PyMOL has no word for. What
survives is what a structural biologist would want back — the molecules where
they are on screen, in the representations they are shown in, with their
colours, the collections they are grouped by, the measurements, and the
camera.

Coordinates are written in **world** space: an object dragged across the
scene, or placed by :func:`~blender_gala.pymol.load.load_session` from a
PyMOL matrix, arrives in PyMOL where it looks in Blender, rather than back
where its file put it.

Colours go across per atom. Where a colour is exactly one of PyMOL's own it is
written as that colour's index, so a chain painted ``skyblue`` in a session
that made the round trip comes back as ``skyblue`` rather than as an anonymous
copy of it; everything else is defined in the session itself.

What does not go: materials, lighting, node trees, and any geometry that is
not a molecule. Those are the parts of a Blender scene PyMOL cannot hold, and
a figure that depends on them is a figure to render in Blender.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..color import colormaps
from ..core import chemistry, units
from ..core import mn as mn_bridge
from ..core.entity import AtomStructure
from .palette import COUNT as PALETTE_COUNT
from .palette import index_for_name, rgb_for_index
from .session import (
    REPS,
    PymolMeasurement,
    PymolMolecule,
    PymolSelection,
    PymolSession,
    write_session,
)
from .view import camera_to_view

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = ["SavedSession", "save_session", "scene_to_session"]

#: Which PyMOL representations each Molecular Nodes style becomes. Keyed by
#: the tail of MN's generated interface class name — ``DynamicStyleInterface_
#: Cartoon``, or ``DynamicStyleInterface_Ball and Stick`` — reduced to letters
#: so the spaces in that last one do not matter. Ball and stick is two
#: representations in PyMOL rather than one.
REP_MAP = {
    "cartoon": ("cartoon",),
    "ribbon": ("ribbon",),
    "surface": ("surface",),
    "sticks": ("sticks",),
    "spheres": ("spheres",),
    "ballandstick": ("sticks", "spheres"),
}

#: Bond orders, from biotite's numbering to PyMOL's.
_BOND_ORDER = {0: 1, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4, 6: 4, 7: 4}

#: How many points each kind of measurement is between. PyMOL's objects hold
#: pairs, triples and quadruples and have no shape for anything else.
_MEASURED_BETWEEN = {"distance": 2, "angle": 3, "dihedral": 4}

#: The collection Molecular Nodes puts every molecule it imports into. It says
#: where an object came from rather than what it was grouped with.
_MN_COLLECTION = "MolecularNodes"

#: How far from an atom, in ångström, a label's anchor may sit and still be a
#: label on that atom. A label is drawn at an atom or at a residue centroid,
#: so the atom it belongs to is always within about a bond length of it.
#: Anything further away is a label on something that is not being written, and
#: attaching it anyway is how a label drawn 100 Å away ends up on atom 71.
_ANCHOR_TOLERANCE = 2.0

#: The custom property a label carries naming the molecule it was drawn from.
#: Where the geometry is ambiguous — two superposed copies — this is the only
#: thing that can tell them apart.
_LABEL_MOLECULE = "gala_molecule"


@dataclass
class SavedSession:
    """What :func:`save_session` wrote.

    Attributes
    ----------
    path : str
        The file written.
    session : PymolSession
        The session that was written, for inspection.
    skipped : list of str
        Anything in the scene that PyMOL cannot hold.
    """

    path: str
    session: PymolSession
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A readable block, for a vignette or the UI to print."""
        lines = [f"Wrote {self.path}"]
        for molecule in self.session.molecules:
            lines.append(f"  {molecule.summary()}")
        if self.session.measurements:
            lines.append(
                f"  {sum(len(m.points) for m in self.session.measurements)} "
                "measurement(s)"
            )
        if self.session.colors:
            lines.append(f"  {len(self.session.colors)} session colour(s) defined")
        for note in self.skipped:
            lines.append(f"  not written: {note}")
        return "\n".join(lines)


def save_session(
    path: str,
    molecules: Sequence[Any] | None = None,
    measurements: bool = True,
    labels: bool = True,
    camera: bool = True,
    colors: bool = True,
    styles: bool = True,
    selections: Sequence[str] = (),
    scale: float | None = None,
) -> SavedSession:
    """Write the scene as a PyMOL session.

    Parameters
    ----------
    path : str
        Where to write. A ``.gz`` suffix compresses it.
    molecules : sequence, optional
        Molecules to write. Defaults to every Molecular Nodes molecule in the
        scene.
    measurements : bool, optional
        Include Gala measurements as PyMOL distance, angle and dihedral
        objects.
    labels : bool, optional
        Attach Gala labels to the atoms they point at, as PyMOL atom labels.
    camera : bool, optional
        Write the scene camera as the session's view.
    colors : bool, optional
        Carry the per-atom ``Color`` attribute over.
    styles : bool, optional
        Turn Molecular Nodes styles into PyMOL representations.
    selections : sequence of str, optional
        Names of boolean mesh attributes to write as named selections.
    scale : float, optional
        Blender units per ångström for the measurements, labels and camera.
        Defaults to the scale the molecules being written were built at, which
        is the scale their coordinates are read back at: a measurement written
        against any other one is drawn across a different distance than it
        measures.

    Returns
    -------
    SavedSession
    """
    result = scene_to_session(
        molecules=molecules,
        measurements=measurements,
        labels=labels,
        camera=camera,
        colors=colors,
        styles=styles,
        selections=selections,
        scale=scale,
        path=path,
    )
    write_session(result.session, path)
    return result


def scene_to_session(
    molecules: Sequence[Any] | None = None,
    measurements: bool = True,
    labels: bool = True,
    camera: bool = True,
    colors: bool = True,
    styles: bool = True,
    selections: Sequence[str] = (),
    scale: float | None = None,
    path: str = "",
) -> SavedSession:
    """Build a :class:`PymolSession` from the scene without writing it.

    Takes and returns the same things as :func:`save_session`; useful for
    inspecting or editing what would be written.
    """
    _require_bpy()

    session = PymolSession(source=path or "Blender scene")
    result = SavedSession(path=path, session=session)
    palette = _Palette(session)

    entities = _molecules(molecules, result)
    scale = _scene_scale(entities) if scale is None else scale

    for entity in entities:
        structure = AtomStructure.from_any(entity)
        molecule = _build_molecule(
            structure, entity, palette, colors, styles, scale, result
        )
        if molecule is not None:
            session.molecules.append(molecule)
            _add_selections(session, structure, entity, selections)
            _add_group(session, entity, molecule)

    # The molecules the scene holds that this session does not. An annotation
    # sitting on one of them belongs to it, not to whatever is nearest among
    # the molecules that happen to be written.
    left_out = _left_out(entities, result)

    if labels:
        _add_labels(session, scale, result, left_out)

    if measurements:
        session.measurements.extend(_measurements(session, scale, result, left_out))

    if camera:
        session.view = _camera_view(session, scale)

    return result


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------


def _scene_scale(entities: Sequence[Any]) -> float:
    """The scale the scene is at: the one its molecules were built at.

    Atom coordinates are written by dividing world positions by each
    molecule's *own* world scale, so a default taken from anywhere else puts
    the measurements, the labels and the camera at a scale the atoms are not
    at — at half Molecular Nodes' scale a measurement comes out across twice
    the separation it was drawn between.
    """
    for entity in entities:
        obj = _entity_object(entity)
        if obj is not None:
            return units.world_scale_of(obj)
    return units.DEFAULT_WORLD_SCALE


def _molecules(given: Sequence[Any] | None, result: SavedSession) -> list[Any]:
    """The molecules to write: those given, or every one in the scene."""
    if given is not None:
        return list(given)

    mn_bridge.require_mn()
    scene: Any = getattr(bpy.context, "scene", None)
    # MNSession is attached to the Scene by Molecular Nodes' registration, so
    # it is absent whenever MN is installed but not registered.
    mn_session = getattr(scene, "MNSession", None)
    if mn_session is None:  # pragma: no cover - MN not registered
        return []

    found = []
    for entity in mn_session.entities.values():
        obj = _entity_object(entity)
        if obj is None or getattr(entity, "array", None) is None:
            continue
        if obj.name not in scene.objects:
            continue
        found.append(entity)
    return found


def _entity_object(entity: Any) -> Any:
    """The Blender object behind a Molecular Nodes entity, or ``None``.

    Deleting a molecule's object leaves its entry in the Molecular Nodes
    session behind, and asking that entry for its object raises rather than
    returning nothing. Exporting a scene should not care that something was
    deleted earlier in it.
    """
    try:
        return entity.object
    except Exception:
        return None


def _build_molecule(
    structure: AtomStructure,
    entity: Any,
    palette: _Palette,
    colors: bool,
    styles: bool,
    scale: float,
    result: SavedSession,
) -> PymolMolecule | None:
    """Turn one molecule into the PyMOL object it corresponds to."""
    array = structure.array
    n_atoms = structure.n_atoms
    if not n_atoms:
        return None

    own = structure.world_scale
    if own != scale:
        # The atoms are read back at the scale they were built at, whatever
        # the session is being written at, so say when the two differ: the
        # measurements beside them are at the session's scale and will no
        # longer touch the atoms they measure.
        result.skipped.append(
            f"{_object_name(entity)}: its coordinates are read at the "
            f"{own:g} Blender units per ångström it was built at, not the "
            f"{scale:g} the rest of the session is written at"
        )
    coord = structure.world_positions() / own

    def annotation(name: str, default: Any, dtype: Any) -> np.ndarray:
        values = getattr(array, name, None)
        if values is None:
            return np.full(n_atoms, default, dtype=dtype)
        return np.asarray(values).astype(dtype)

    element = np.array(
        [str(e).strip().upper() for e in annotation("element", "C", "U2")], dtype="U2"
    )

    color_index = np.full(n_atoms, index_for_name("grey80") or 0, dtype=int)
    if colors:
        rgba = _read_colors(entity, n_atoms)
        if rgba is not None:
            color_index = palette.indices(rgba)

    reps = np.zeros(n_atoms, dtype=np.int64)
    if styles:
        reps = _reps_for(entity, n_atoms, result)

    return PymolMolecule(
        name=_object_name(entity),
        coord=coord.reshape(1, n_atoms, 3),
        chain_id=annotation("chain_id", "A", "U4"),
        res_id=annotation("res_id", 1, int),
        ins_code=annotation("ins_code", "", "U1"),
        res_name=annotation("res_name", "UNK", "U5"),
        atom_name=annotation("atom_name", "C", "U6"),
        element=element,
        alt_id=np.full(n_atoms, "", dtype="U4"),
        segi=np.full(n_atoms, "", dtype="U4"),
        b_factor=annotation("b_factor", 0.0, float),
        occupancy=annotation("occupancy", 1.0, float),
        charge=annotation("charge", 0, float),
        vdw=np.array([chemistry.vdw_radius(e) for e in element], dtype=float),
        hetero=annotation("hetero", False, bool),
        label=np.full(n_atoms, "", dtype=object),
        reps=reps,
        color_index=color_index,
        bonds=_bonds(array, n_atoms),
        ss=_secondary_structure(entity, array, n_atoms),
        visible=not entity.object.hide_render,
    )


def _secondary_structure(entity: Any, array: Any, n_atoms: int) -> np.ndarray:
    """PyMOL's per-atom ``ss`` letters, from Molecular Nodes' codes.

    Without this a cartoon exported to PyMOL is drawn entirely as loops:
    PyMOL assigns secondary structure when it *loads a structure*, and a
    session it opens is taken at its word.
    """
    codes = getattr(array, "sec_struct", None)
    if codes is None:
        mesh = getattr(entity.object, "data", None)
        attribute = getattr(mesh, "attributes", {}).get("sec_struct") if mesh else None
        if attribute is None or len(attribute.data) != n_atoms:
            return np.zeros(0, dtype="U1")
        values = np.zeros(n_atoms, dtype=int)
        attribute.data.foreach_get("value", values)
        codes = values

    codes = np.asarray(codes, dtype=int)
    if len(codes) != n_atoms:
        return np.zeros(0, dtype="U1")
    letters = np.full(n_atoms, "", dtype="U1")
    for letter, code in (("H", 1), ("S", 2), ("L", 3)):
        letters[codes == code] = letter
    return letters


def _object_name(entity: Any) -> str:
    """A PyMOL-safe object name: no spaces, since selections are parsed."""
    name = str(getattr(entity.object, "name", "molecule"))
    return name.replace(" ", "_")


def _read_colors(entity: Any, n_atoms: int) -> np.ndarray | None:
    """Read the mesh ``Color`` attribute as ``(n, 3)`` of PyMOL colour values.

    Blender's colour attributes are linear and PyMOL's values are display
    values, so this is where the two meet. Without the conversion every
    exported colour arrives in PyMOL noticeably darker than it looks in
    Blender, and none of them match the built-in colour they came from.
    """
    mesh = getattr(entity.object, "data", None)
    attribute = getattr(mesh, "attributes", {}).get("Color") if mesh else None
    if attribute is None or len(attribute.data) != n_atoms:
        return None
    flat = np.zeros(n_atoms * 4, dtype=np.float32)
    attribute.data.foreach_get("color", flat)
    linear = flat.reshape(n_atoms, 4)[:, :3].astype(float)
    return np.asarray(colormaps.linear_to_srgb(linear), dtype=float)


def _reps_for(entity: Any, n_atoms: int, result: SavedSession) -> np.ndarray:
    """Build the per-atom representation mask from the styles on an object."""
    reps = np.zeros(n_atoms, dtype=np.int64)
    for style in getattr(entity, "styles", []):
        label = type(style).__name__.rsplit("_", 1)[-1]
        name = "".join(c for c in label.lower() if c.isalnum())
        mapped = REP_MAP.get(name)
        if mapped is None:
            result.skipped.append(f"{entity.object.name}: {label} style")
            continue
        mask = _style_selection(entity, style, n_atoms, label, result)
        for rep in mapped:
            reps[mask] |= 1 << REPS.index(rep)
    return reps


def _style_selection(
    entity: Any, style: Any, n_atoms: int, label: str, result: SavedSession
) -> np.ndarray:
    """Which atoms a style applies to.

    A style limited to a selection is driven by a boolean attribute wired into
    its ``Selection`` socket, so reading it back means walking the node tree —
    Molecular Nodes' internal business, and liable to move. When the socket is
    linked to something that cannot be read the style is taken to cover every
    atom, which is right for the unlimited case and over-reports otherwise; it
    is said out loud rather than assumed, because a stick representation that
    quietly spreads to the whole protein is very visible in PyMOL.
    """
    everything = np.ones(n_atoms, dtype=bool)
    try:
        for node in getattr(style, "_nodes", []):
            for socket in getattr(node, "inputs", []):
                if socket.name.lower() != "selection" or not socket.is_linked:
                    continue
                for link in socket.links:
                    attribute = _named_attribute(link.from_node)
                    mask = (
                        _read_boolean(entity, attribute, n_atoms) if attribute else None
                    )
                    if mask is not None:
                        return mask
                result.skipped.append(
                    f"{entity.object.name}: the selection limiting its {label} "
                    "style, which is written over every atom instead"
                )
                return everything
    except Exception:  # pragma: no cover - depends on MN internals
        return everything
    return everything


def _named_attribute(node: Any) -> str | None:
    """The attribute name a Named Attribute node reads, if that is what it is."""
    if "NamedAttribute" not in getattr(node, "bl_idname", ""):
        return None
    for socket in getattr(node, "inputs", []):
        if socket.name.lower() == "name":
            return str(getattr(socket, "default_value", "")) or None
    return None


def _read_boolean(entity: Any, name: str, n_atoms: int) -> np.ndarray | None:
    """Read a boolean point attribute off the mesh."""
    mesh = getattr(entity.object, "data", None)
    attribute = getattr(mesh, "attributes", {}).get(name) if mesh else None
    if attribute is None or len(attribute.data) != n_atoms:
        return None
    values = np.zeros(n_atoms, dtype=bool)
    attribute.data.foreach_get("value", values)
    return values


def _bonds(array: Any, n_atoms: int) -> np.ndarray:
    """Bonds as ``(n, 3)`` of ``atom1, atom2, order``."""
    bonds = getattr(array, "bonds", None)
    if bonds is None:
        return np.zeros((0, 3), dtype=int)
    try:
        table = np.asarray(bonds.as_array(), dtype=int)
    except Exception:  # pragma: no cover - depends on biotite
        return np.zeros((0, 3), dtype=int)
    if not len(table):
        return np.zeros((0, 3), dtype=int)
    orders = np.array([_BOND_ORDER.get(int(t), 1) for t in table[:, 2]], dtype=int)
    keep = (table[:, 0] < n_atoms) & (table[:, 1] < n_atoms)
    return np.column_stack([table[keep, 0], table[keep, 1], orders[keep]])


def _add_selections(
    session: PymolSession,
    structure: AtomStructure,
    entity: Any,
    names: Sequence[str],
) -> None:
    """Write named boolean attributes out as PyMOL selections."""
    for name in names:
        mask = _read_boolean(entity, name, structure.n_atoms)
        if mask is None or not mask.any():
            continue
        existing = next((s for s in session.selections if s.name == name), None)
        if existing is None:
            existing = PymolSelection(name=name, members={})
            session.selections.append(existing)
        existing.members[_object_name(entity)] = np.flatnonzero(mask)


# ---------------------------------------------------------------------------
# Measurements and the camera
# ---------------------------------------------------------------------------


def _measurements(
    session: PymolSession,
    scale: float,
    result: SavedSession,
    left_out: Sequence[tuple[str, np.ndarray]] = (),
) -> list[PymolMeasurement]:
    """Collect Gala measurements from the scene, grouped by kind.

    Restricting the export to some of the scene's molecules restricts the
    measurements with it: a distance to an atom that is not in the file is
    drawn in PyMOL as a distance to nothing, and it is the *other* molecule's
    measurement in the first place. Which one a measurement belongs to is
    decided per endpoint, by whichever molecule is nearest — and unlike a
    label there is no absolute distance to test against, because a distance
    between two centroids is legitimately drawn between no atoms at all.
    """
    from ..core import collections as gala_collections

    by_kind: dict[str, list[np.ndarray]] = {}
    for obj in gala_collections.iter_tagged():
        points = obj.get("gala_points")
        if points is None:
            continue
        kind = str(obj.get("gala_type", "")).replace("measurement_", "")
        needed = _MEASURED_BETWEEN.get(kind)
        if needed is None:
            continue
        flat = np.asarray(list(points), dtype=float).reshape(-1)
        if len(flat) != needed * 3:
            # A distance between five points is not a distance PyMOL has a
            # shape for, and the writer used to meet it as a reshape it could
            # not do — naming no object and leaving no file. Gala can reach
            # this itself: a session whose distance carries five points loads,
            # and the scene it makes is then unsaveable.
            result.skipped.append(
                f"{obj.name} ({kind} of {len(flat)} coordinates, not the "
                f"{needed * 3} PyMOL's is between)"
            )
            continue

        points = flat.reshape(needed, 3) / scale
        elsewhere = _measured_elsewhere(session, left_out, points)
        if elsewhere is not None:
            result.skipped.append(
                f"{obj.name}: it measures to {elsewhere}, which is not being written"
            )
            continue

        by_kind.setdefault(kind, []).append(points)

    return [
        PymolMeasurement(name=f"gala_{kind}s", kind=kind, points=np.stack(groups))
        for kind, groups in sorted(by_kind.items())
    ]


def _measured_elsewhere(
    session: PymolSession,
    left_out: Sequence[tuple[str, np.ndarray]],
    points: np.ndarray,
) -> str | None:
    """The molecule not being written that one end of a measurement is on."""
    if not left_out:
        return None
    for point in points:
        here = min(
            (_nearest(molecule.coord[0], point)[0] for molecule in session.molecules),
            default=float("inf"),
        )
        name = _nearer_elsewhere(left_out, point, here)
        if name is not None:
            return name
    return None


def _add_group(session: PymolSession, entity: Any, molecule: PymolMolecule) -> None:
    """Record the collection a molecule sits in as its PyMOL group.

    A session loaded into Blender puts each group's molecules in a collection
    of that name, so this is what makes the grouping survive a round trip.
    The collections Molecular Nodes and Gala organise their own objects with
    are not groups anybody asked for; see :func:`_is_group`.
    """
    obj = _entity_object(entity)
    collection = next(
        (c for c in getattr(obj, "users_collection", ()) if _is_group(c)), None
    )
    if collection is None:
        return

    molecule.group = collection.name
    # Every collection above it is a group too, or the session names a parent
    # it does not contain.
    while collection is not None and collection.name not in session.groups:
        parent = _parent_collection(collection)
        session.groups[collection.name] = (
            parent.name if parent is not None and _is_group(parent) else ""
        )
        collection = parent if parent is not None and _is_group(parent) else None


def _is_group(collection: Any) -> bool:
    """Whether a collection is one PyMOL should be told about.

    The scene's master collection is not: it is not in ``bpy.data`` at all,
    which is what distinguishes it. Neither are Molecular Nodes' own import
    collection nor Gala's, which say where an object came from rather than
    what it was grouped with — writing them would invent a ``MolecularNodes``
    group in every session exported from Blender.
    """
    from ..core import collections as gala_collections

    name = str(getattr(collection, "name", ""))
    if not name or bpy.data.collections.get(name) is not collection:
        return False
    if name.startswith(".") or name == _MN_COLLECTION:
        return False
    root = gala_collections.ROOT
    return name != root and not name.startswith(f"{root} ")


def _parent_collection(collection: Any) -> Any:
    """The collection this one is nested inside, if any."""
    for candidate in bpy.data.collections:
        if collection.name in candidate.children:
            return candidate
    return None


def _left_out(
    written: Sequence[Any], result: SavedSession
) -> list[tuple[str, np.ndarray]]:
    """The scene's molecules that this session is not being given, in ångström.

    ``save_session(path, molecules=[a])`` writes one molecule out of a scene
    that holds several. Every annotation in that scene is still there to be
    collected, and deciding who owns one by asking which of the *written*
    molecules is nearest answers a question nobody asked: a label drawn on the
    molecule that was left out is nearest to it, not to anything here.
    """
    try:
        everything = _molecules(None, result)
    except Exception:  # pragma: no cover - no MN session to read
        return []

    kept = {obj.name for obj in (_entity_object(entity) for entity in written) if obj}
    left = []
    for entity in everything:
        obj = _entity_object(entity)
        if obj is None or obj.name in kept:
            continue
        try:
            structure = AtomStructure.from_any(entity)
            coord = structure.world_positions() / structure.world_scale
        except Exception:  # pragma: no cover - unreadable molecule
            continue
        if len(coord):
            left.append((obj.name, coord))
    return left


def _nearest(coord: np.ndarray, point: np.ndarray) -> tuple[float, int]:
    """Distance to the nearest of ``coord`` and its index, ``inf`` if none is.

    A structure with no placed atoms — every coordinate ``nan``, as a PyMOL
    state with absent atoms is stored — has no nearest anything, and asking
    numpy for one raises rather than saying so.
    """
    distances = np.linalg.norm(coord - point, axis=1)
    if not np.isfinite(distances).any():
        return float("inf"), -1
    index = int(np.nanargmin(distances))
    return float(distances[index]), index


def _nearer_elsewhere(
    left_out: Sequence[tuple[str, np.ndarray]], point: np.ndarray, distance: float
) -> str | None:
    """The molecule not being written that is closer to ``point``, if any.

    A tie is not enough to disqualify an annotation: two superposed copies are
    the same place, and the copy being written is as good an owner as the one
    that is not.
    """
    for name, coord in left_out:
        if _nearest(coord, point)[0] < distance:
            return name
    return None


def _add_labels(
    session: PymolSession,
    scale: float,
    result: SavedSession,
    left_out: Sequence[tuple[str, np.ndarray]] = (),
) -> None:
    """Attach each Gala label to the atom it was drawn on.

    PyMOL hangs a label off an atom, where Gala's is a text object standing in
    space near one. The anchor recorded when the label was drawn says which
    point it belongs to; the nearest atom to that point is the atom PyMOL
    should carry it — but only if there really is an atom there. Nearest with
    no bound on the distance turns "this label belongs to a molecule you are
    not writing" into "atom 71 of this one", which is a wrong figure rather
    than a missing label, so a label with no atom near it is skipped and said
    out loud like every other per-item failure.

    Where the label recorded which molecule it was drawn from, that is used
    instead of the geometry: two superposed copies are equally near every
    anchor, so the geometry cannot tell them apart at all.
    """
    from ..core import collections as gala_collections

    if not session.molecules:
        return

    for obj in gala_collections.iter_tagged("label"):
        anchor = obj.get("gala_anchor")
        text = _label_text(obj)
        if anchor is None or not text:
            continue
        point = np.asarray(list(anchor), dtype=float) / scale

        drawn_from = obj.get(_LABEL_MOLECULE)
        candidates = list(session.molecules)
        if drawn_from is not None:
            named = str(drawn_from).replace(" ", "_")
            candidates = [m for m in session.molecules if m.name == named]
            if not candidates:
                result.skipped.append(
                    f"{obj.name}: it labels {drawn_from}, which is not being written"
                )
                continue

        best: tuple[float, PymolMolecule, int] | None = None
        tied = False
        for molecule in candidates:
            distance, index = _nearest(molecule.coord[0], point)
            if best is None or distance < best[0]:
                best = (distance, molecule, index)
                tied = False
            elif distance == best[0]:
                tied = True

        if best is None or not np.isfinite(best[0]):
            continue
        distance, molecule, index = best

        if drawn_from is None:
            elsewhere = _nearer_elsewhere(left_out, point, distance)
            if elsewhere is not None:
                result.skipped.append(
                    f"{obj.name}: it was drawn on {elsewhere}, which is not "
                    "being written"
                )
                continue
            if tied:
                # Two molecules exactly as near as each other is the superposed
                # comparison figure, and whichever is written first is not an
                # answer — it is a coin toss recorded as fact in the session.
                result.skipped.append(
                    f"{obj.name}: it sits the same distance from more than one "
                    "molecule, so which one it labels cannot be told from the "
                    "scene"
                )
                continue

        if distance > _ANCHOR_TOLERANCE:
            result.skipped.append(
                f"{obj.name}: the nearest atom to it is {distance:.1f} A away, "
                f"in {molecule.name}, which is further than a label is ever "
                "drawn from what it labels"
            )
            continue

        molecule.label[index] = text
        molecule.reps[index] |= 1 << REPS.index("labels")


def _label_text(obj: Any) -> str:
    """The text a Gala label object shows."""
    data = getattr(obj, "data", None)
    body = getattr(data, "body", None)
    if body:
        return str(body)
    name = str(getattr(obj, "name", ""))
    prefix = "GALA Label "
    return name[len(prefix) :] if name.startswith(prefix) else ""


def _camera_view(session: PymolSession, scale: float) -> Any:
    """The scene camera as a PyMOL view, turning about what is in frame."""
    origin = None
    if session.molecules:
        centres = [molecule.coord[0] for molecule in session.molecules]
        stacked = np.vstack(centres)
        finite = stacked[np.isfinite(stacked).all(axis=1)]
        if len(finite):
            origin = finite.mean(axis=0)
    try:
        return camera_to_view(origin=origin, scale=scale)
    except ValueError:
        return session.view


class _Palette:
    """Maps RGB to PyMOL colour indices, defining new ones as needed."""

    def __init__(self, session: PymolSession) -> None:
        self.session = session
        self._cache: dict[tuple[int, int, int], int] = {}
        self._next = PALETTE_COUNT

    def indices(self, rgba: np.ndarray) -> np.ndarray:
        return np.array([self.index(rgb) for rgb in rgba], dtype=int)

    def index(self, rgb: Any) -> int:
        """The index for one colour, reusing a built-in when it matches."""
        key = _as_bytes(rgb)
        if key in self._cache:
            return self._cache[key]

        index = _builtin_index(key)
        if index is None:
            index = self._next
            self._next += 1
            self.session.colors[index] = (
                key[0] / 255.0,
                key[1] / 255.0,
                key[2] / 255.0,
            )
        self._cache[key] = index
        return index


def _as_bytes(rgb: Any) -> tuple[int, int, int]:
    """One colour as three 0-255 channels, which is how the table is keyed."""
    red, green, blue = (min(255, max(0, round(float(c) * 255))) for c in rgb[:3])
    return red, green, blue


_BUILTIN_BY_RGB: dict[tuple[int, int, int], int] | None = None


def _builtin_index(key: tuple[int, int, int]) -> int | None:
    """The index of a built-in colour that is exactly this one, if any."""
    global _BUILTIN_BY_RGB
    if _BUILTIN_BY_RGB is None:
        table: dict[tuple[int, int, int], int] = {}
        for index in range(PALETTE_COUNT):
            rgb = rgb_for_index(index)
            if rgb is None:  # pragma: no cover - the table is dense
                continue
            # Lower indices are the named colours people recognise, so the
            # first one to claim a key keeps it.
            table.setdefault(_as_bytes(rgb), index)
        _BUILTIN_BY_RGB = table
    return _BUILTIN_BY_RGB.get(key)


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy
