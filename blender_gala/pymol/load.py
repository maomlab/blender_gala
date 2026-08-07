"""Building a Blender scene out of a PyMOL session.

What comes across, and what it becomes:

============================  =========================================
PyMOL                         Blender
============================  =========================================
molecular object              a Molecular Nodes molecule
cartoon, ribbon, surface,     the matching Molecular Nodes style, applied
sticks, spheres               to the atoms that were shown in it
lines, nonbonded              sticks and spheres, thinner — Molecular
                              Nodes has no line representation
per-atom colour               the ``Color`` attribute, so every style
                              shows the colours the session had
atom labels                   Gala label objects
distance, angle, dihedral     Gala measurements, drawn
named selection               a boolean attribute of the same name
group                         a collection
the view                      the scene camera
============================  =========================================

Everything else — maps, meshes, CGOs, ramps — is listed in
:attr:`LoadedSession.skipped` rather than silently dropped.

The molecules are handed to Molecular Nodes as a structure file rather than
built vertex by vertex, so that they arrive with the node tree, attributes and
styles MN would have given them had you imported them yourself.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..color import coloring, colormaps
from ..core import mn as mn_bridge
from ..core import units
from ..measure.draw import draw_measurement
from ..measure.measurements import Measurement
from .session import PymolMolecule, PymolSession, read_session
from .view import view_to_camera

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = ["LoadedSession", "load_session"]

#: How each PyMOL representation is built in Molecular Nodes. ``None`` means
#: there is no equivalent and the representation is reported instead.
STYLE_MAP = {
    "cartoon": ("StyleCartoon", {}),
    "ribbon": ("StyleRibbon", {}),
    "surface": ("StyleSurface", {}),
    "sticks": ("StyleSticks", {}),
    "spheres": ("StyleSpheres", {}),
    # PyMOL's lines and nonbonded markers are wireframe-thin; the nearest
    # honest equivalent is a very thin stick and a very small sphere.
    "lines": ("StyleSticks", {"radius": 0.05}),
    "nonbonded": ("StyleSpheres", {"radius": 0.15}),
    "nb_spheres": ("StyleSpheres", {"radius": 0.25}),
}

#: Representations that are drawn some other way, so they are not "skipped".
_HANDLED_ELSEWHERE = frozenset({"labels", "dashes", "angles", "dihedrals"})


@dataclass
class LoadedSession:
    """What :func:`load_session` created.

    Attributes
    ----------
    molecules : dict
        The Molecular Nodes molecule per PyMOL object name.
    styles : dict
        The PyMOL representations applied, per object name.
    session : PymolSession
        The session as read, for anything the scene does not carry.
    measurements : list of Measurement
        Measurements recreated and drawn.
    skipped : list of str
        Things that did not come across, each with a reason.
    """

    session: PymolSession
    molecules: dict[str, Any] = field(default_factory=dict)
    styles: dict[str, list[str]] = field(default_factory=dict)
    measurements: list[Measurement] = field(default_factory=list)
    labels: list[Any] = field(default_factory=list)
    camera: Any = None
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A readable block, for a vignette or the UI to print."""
        lines = [f"Loaded {self.session.source or 'session'}"]
        for name, molecule in self.molecules.items():
            styles = ", ".join(self.styles.get(name, []))
            lines.append(
                f"  {name}: {len(molecule.object.data.vertices)} atoms"
                + (f" [{styles}]" if styles else "")
            )
        if self.measurements:
            lines.append(f"  {len(self.measurements)} measurement(s) drawn")
        if self.labels:
            lines.append(f"  {len(self.labels)} label(s)")
        for note in self.skipped:
            lines.append(f"  not converted: {note}")
        return "\n".join(lines)


def load_session(
    path: str,
    state: int = 0,
    styles: bool = True,
    colors: bool = True,
    camera: bool = True,
    measurements: bool = True,
    labels: bool = True,
    selections: bool = True,
    groups: bool = True,
    scale: float | None = None,
) -> LoadedSession:
    """Open a PyMOL session in Blender.

    Parameters
    ----------
    path : str
        A ``.pse``, or a :class:`~blender_gala.pymol.session.PymolSession`
        that has already been read.
    state : int, optional
        Which state to build for multi-state objects.
    styles : bool, optional
        Apply Molecular Nodes styles matching the representations that were
        shown.
    colors : bool, optional
        Carry the per-atom colours over.
    camera : bool, optional
        Point the scene camera the way PyMOL was pointing.
    measurements, labels : bool, optional
        Recreate distance/angle/dihedral objects and atom labels.
    selections : bool, optional
        Store each named selection as a boolean attribute on the molecule, so
        it can be used as a style selection or in a node tree.
    groups : bool, optional
        Recreate PyMOL groups as collections.
    scale : float, optional
        Blender units per ångström. Defaults to Molecular Nodes' 0.01.

    Returns
    -------
    LoadedSession

    Raises
    ------
    MolecularNodesUnavailable
        If Molecular Nodes is not installed; it is what builds the molecules.
    PymolSessionError
        If the file cannot be read faithfully.
    """
    module = _require_bpy()
    mn = mn_bridge.require_mn()

    session = path if isinstance(path, PymolSession) else read_session(path)
    scale = units.DEFAULT_WORLD_SCALE if scale is None else scale
    result = LoadedSession(session=session)

    for name, kind in session.unsupported:
        result.skipped.append(f"{name} ({kind})")

    for molecule in session.molecules:
        entity = _build_molecule(mn, molecule, state, result)
        if entity is None:
            continue
        result.molecules[molecule.name] = entity

        if colors:
            _apply_colors(session, molecule, entity)
        if selections:
            _apply_selections(session, molecule, entity)
        if styles:
            _apply_styles(mn, molecule, entity, result)
        if molecule.matrix is not None:
            _apply_transform(module, entity, molecule.matrix, scale)
        entity.object.hide_render = not molecule.visible
        entity.object.hide_viewport = not molecule.visible

    if groups:
        _apply_groups(module, session, result)
    if camera:
        result.camera = view_to_camera(session.view, scale=scale)
    if measurements:
        _draw_measurements(session, result, scale)
    if labels:
        _draw_labels(session, result, scale)

    return result


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------


def _build_molecule(
    mn: Any, molecule: PymolMolecule, state: int, result: LoadedSession
) -> Any:
    """Hand one object to Molecular Nodes as a structure file."""
    try:
        array = molecule.to_atom_array(state if molecule.n_states > 1 else 0)
    except Exception as exc:
        result.skipped.append(f"{molecule.name} ({exc})")
        return None

    finite = np.isfinite(array.coord).all(axis=1)
    if not finite.any():
        result.skipped.append(f"{molecule.name} (no coordinates in state {state})")
        return None
    if not finite.all():
        # An atom with no position in this state cannot be given one; keeping
        # it would put it at the origin and drag every bond towards it.
        array = array[finite]

    with tempfile.TemporaryDirectory() as workdir:
        for suffix, writer in ((".cif", _write_cif), (".pdb", _write_pdb)):
            target = os.path.join(workdir, _safe_name(molecule.name) + suffix)
            try:
                writer(array, target)
            except Exception:  # pragma: no cover - depends on biotite build
                continue
            try:
                with warnings.catch_warnings():
                    # biotite warns that the CIF it just read has no B-factor
                    # or occupancy column, which is true and is handled a few
                    # lines below by putting the session's own values back.
                    # Left alone it fires once per column per object.
                    warnings.filterwarnings(
                        "ignore", message=".*(B_iso_or_equiv|occupancy).*"
                    )
                    # remove_solvent would break the 1:1 correspondence every
                    # colour and selection below depends on.
                    entity = mn.Molecule.load(
                        target, name=molecule.name, remove_solvent=False
                    )
            except Exception:  # pragma: no cover - depends on MN
                continue
            if _atom_count(entity) == len(array):
                _restore_annotations(entity, array)
                _restore_secondary_structure(entity, molecule, finite)
                return entity
            result.skipped.append(
                f"{molecule.name} styles/colours "
                f"(Molecular Nodes read {_atom_count(entity)} of {len(array)} atoms)"
            )
            return entity

    result.skipped.append(f"{molecule.name} (could not be written for import)")
    return None


#: Per-atom values that do not survive every interchange format, and the
#: Blender attribute type each is stored as.
_RESTORED = {"b_factor": "FLOAT", "occupancy": "FLOAT", "charge": "FLOAT"}


def _restore_annotations(entity: Any, array: Any) -> None:
    """Put back the per-atom values the interchange file dropped.

    biotite's CIF writer emits no ``B_iso_or_equiv`` and no ``occupancy``
    column, so a structure that goes out as CIF comes back with ``nan`` in
    both — and a b-factor of ``nan`` quietly breaks ``b > 70`` selections and
    :func:`~blender_gala.color.color_by_bfactor` alike. The session has the
    real values, so they are written straight onto the mesh and onto the atom
    array Molecular Nodes kept.
    """
    for name, atype in _RESTORED.items():
        values = getattr(array, name, None)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        with contextlib.suppress(Exception):  # depends on MN
            entity.store_named_attribute(values, name=name, atype=atype)
        target = getattr(entity, "array", None)
        if target is not None and hasattr(target, name):
            with contextlib.suppress(Exception):  # depends on biotite
                setattr(target, name, values)


#: Molecular Nodes' ``sec_struct`` codes, and the PyMOL letters that map to
#: them. 0 means "not protein"; 3 is loop.
SEC_STRUCT_CODES = {"H": 1, "S": 2, "L": 3}


def _restore_secondary_structure(
    entity: Any, molecule: PymolMolecule, keep: np.ndarray
) -> None:
    """Carry the session's secondary structure onto the imported molecule.

    Molecular Nodes draws a cartoon from its ``sec_struct`` attribute, which it
    fills from a PDB's HELIX and SHEET records or, failing those, by computing
    one. Neither happens here: the structure goes across as CIF, so without
    this every helix in a session arrives as a loop. The session already knows
    what PyMOL was drawing, which is the right answer even when it disagrees
    with a fresh assignment.
    """
    if len(molecule.ss) != len(keep):
        return
    letters = molecule.ss[keep]
    codes = np.zeros(len(letters), dtype=int)
    for letter, code in SEC_STRUCT_CODES.items():
        codes[letters == letter] = code
    # Anything in a polymer that PyMOL left blank is loop rather than "not
    # protein", or the cartoon breaks wherever the assignment stopped.
    residues = molecule.res_name[keep]
    from ..core import chemistry

    polymer = np.array(
        [str(name).strip().upper() in chemistry.AMINO_ACIDS for name in residues]
    )
    codes[(codes == 0) & polymer] = SEC_STRUCT_CODES["L"]

    try:
        entity.store_named_attribute(codes, name="sec_struct", atype="INT")
    except Exception:  # pragma: no cover - depends on MN
        return
    target = getattr(entity, "array", None)
    if target is not None:
        with contextlib.suppress(Exception):  # depends on biotite
            target.set_annotation("sec_struct", codes)


def _atom_count(entity: Any) -> int:
    """How many atoms the imported object actually has.

    Counted on the mesh rather than from ``entity.array``, which is an
    ``AtomArrayStack`` for a multi-model file and whose length is then the
    number of models. The vertices are what every colour, mask and attribute
    below is indexed against.
    """
    data = getattr(entity.object, "data", None)
    return len(data.vertices) if data is not None else 0


def _write_cif(array: Any, path: str) -> None:
    from biotite.structure.io.pdbx import CIFFile, set_structure

    handle = CIFFile()
    set_structure(handle, array)
    handle.write(path)


def _write_pdb(array: Any, path: str) -> None:
    from biotite.structure.io.pdb import PDBFile

    handle = PDBFile()
    handle.set_structure(array)
    handle.write(path)


def _safe_name(name: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "_" for c in name]
    return "".join(keep) or "object"


def _apply_colors(session: PymolSession, molecule: PymolMolecule, entity: Any) -> None:
    """Write the session's per-atom colours onto the mesh.

    PyMOL's colour values are display values — what ``set_color`` takes and
    what the viewer draws — while Blender's colour attributes are linear. Put
    across unconverted they come out washed out, so the conversion happens
    here, at the boundary, as it does for every other colour Gala writes.
    """
    colors = session.atom_colors(molecule)
    if len(colors) != _atom_count(entity):
        return
    colors[:, :3] = colormaps.srgb_to_linear(colors[:, :3])
    coloring.write_colors(entity, colors)


def _apply_selections(
    session: PymolSession, molecule: PymolMolecule, entity: Any
) -> None:
    """Store each named selection as a boolean attribute."""
    for selection in session.selections:
        indices = selection.members.get(molecule.name)
        if indices is None:
            continue
        mask = np.zeros(_atom_count(entity), dtype=bool)
        inside = indices[(indices >= 0) & (indices < len(mask))]
        mask[inside] = True
        _store_mask(entity, _attribute_name(selection.name), mask)


def _apply_styles(
    mn: Any, molecule: PymolMolecule, entity: Any, result: LoadedSession
) -> None:
    """Add one Molecular Nodes style per representation that was shown."""
    for rep in molecule.reps_present():
        if rep in _HANDLED_ELSEWHERE:
            continue
        mapped = STYLE_MAP.get(rep)
        if mapped is None:
            result.skipped.append(f"{molecule.name}: {rep} representation")
            continue

        mask = molecule.rep_mask(rep)
        if len(mask) != _atom_count(entity) or not mask.any():
            continue

        style_name, options = mapped
        style_class = getattr(mn, style_name, None)
        if style_class is None:  # pragma: no cover - depends on MN
            result.skipped.append(f"{molecule.name}: {rep} ({style_name} missing)")
            continue

        selection = None
        if not mask.all():
            selection = _attribute_name(f"pymol {rep}")
            _store_mask(entity, selection, mask)

        # color=None leaves the Color attribute alone: the session's colours
        # are already there, and MN's own colour generator would overwrite it.
        entity.add_style(style_class(**options), color=None, selection=selection)
        result.styles.setdefault(molecule.name, []).append(rep)


def _store_mask(entity: Any, name: str, mask: np.ndarray) -> None:
    entity.store_named_attribute(mask, name=name, atype="BOOLEAN", domain="POINT")


def _attribute_name(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name.strip().lower())


def _apply_transform(
    module: Any, entity: Any, matrix: np.ndarray, scale: float
) -> None:
    """Place an object that PyMOL had moved away from its coordinates."""
    from mathutils import Matrix

    placed = np.array(matrix, dtype=float).copy()
    placed[:3, 3] *= scale
    entity.object.matrix_world = (
        Matrix([[float(v) for v in row] for row in placed]) @ entity.object.matrix_world
    )


def _apply_groups(module: Any, session: PymolSession, result: LoadedSession) -> None:
    """Recreate PyMOL groups as collections."""
    wanted = {
        molecule.group
        for molecule in session.molecules
        if molecule.group and molecule.name in result.molecules
    }
    for name in sorted(wanted | set(session.groups)):
        if not name:
            continue
        collection = module.data.collections.get(name)
        if collection is None:
            collection = module.data.collections.new(name)
            module.context.scene.collection.children.link(collection)
        for molecule in session.molecules:
            if molecule.group != name:
                continue
            entity = result.molecules.get(molecule.name)
            if entity is None:
                continue
            obj = entity.object
            for existing in list(obj.users_collection):
                existing.objects.unlink(obj)
            collection.objects.link(obj)


# ---------------------------------------------------------------------------
# Measurements and labels
# ---------------------------------------------------------------------------


def _draw_measurements(
    session: PymolSession, result: LoadedSession, scale: float
) -> None:
    """Recreate distance, angle and dihedral objects as Gala measurements."""
    for entry in session.measurements:
        target = result.molecules.get(entry.group) or next(
            iter(result.molecules.values()), None
        )
        for points, value in zip(entry.points, entry.values, strict=True):
            measurement = Measurement(
                kind=entry.kind,
                value=float(value),
                unit="A" if entry.kind == "distance" else "deg",
                atoms=(),
                points=np.asarray(points, dtype=float) * scale,
                labels=(),
            )
            measurement.objects.extend(
                draw_measurement(
                    measurement,
                    target=target,
                    label_avoid_occlusion=False,
                    scale=scale,
                )
            )
            result.measurements.append(measurement)


def _draw_labels(session: PymolSession, result: LoadedSession, scale: float) -> None:
    """Recreate PyMOL's atom labels as Gala label objects.

    PyMOL labels carry their own text — often something the user typed rather
    than anything derivable from the atom — so each is placed literally, one
    call per labelled atom.
    """
    from ..annotate.labels import label

    for molecule in session.molecules:
        entity = result.molecules.get(molecule.name)
        if entity is None or not len(molecule.label):
            continue
        n_atoms = _atom_count(entity)
        for index, text in enumerate(molecule.label):
            text = str(text).strip()
            if not text or index >= n_atoms:
                continue
            mask = np.zeros(n_atoms, dtype=bool)
            mask[index] = True
            result.labels.extend(
                label(
                    entity,
                    mask,
                    text=text,
                    level="selection",
                    avoid_occlusion=False,
                    scale=scale,
                )
            )


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy
