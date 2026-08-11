"""Optional PLIP backend.

PLIP (the Protein-Ligand Interaction Profiler) is the reference tool for
ligand-binding-site interaction analysis, but it depends on OpenBabel and
cannot be assumed inside Blender's interpreter (SPECIFICATION D-18). Gala's
native detectors use PLIP's published criteria; this module lets a user who
*does* have PLIP available feed its output straight into Gala's drawing layer,
so the figure matches the analysis in their paper exactly.

Usage
-----
Run PLIP in an environment that has it, save the result, and import it here::

    from blender_gala.interactions import plip
    interactions = plip.from_pdb("complex.pdb", ligand="STI:A:1")
    gala.draw_interactions(interactions, target=mol)
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ..core.entity import AtomStructure
from .detect import Interaction

__all__ = ["PLIP_KIND_MAP", "available", "from_pdb", "from_plip_site"]

#: PLIP interaction-set attribute -> Gala interaction kind.
PLIP_KIND_MAP: dict[str, str] = {
    "hbonds_ldon": "hbond",
    "hbonds_pdon": "hbond",
    "hydrophobic_contacts": "hydrophobic",
    "saltbridge_lneg": "salt_bridge",
    "saltbridge_pneg": "salt_bridge",
    "pistacking": "pi_stacking",
    "pication_laro": "cation_pi",
    "pication_paro": "cation_pi",
    "halogen_bonds": "halogen",
    "metal_complexes": "metal",
    "water_bridges": "polar",
}


def available() -> bool:
    """Return ``True`` when PLIP can be imported in this interpreter."""
    try:
        importlib.import_module("plip.structure.preparation")
    except Exception:
        return False
    return True


def _require_plip() -> Any:
    try:
        return importlib.import_module("plip.structure.preparation")
    except Exception as exc:
        raise ImportError(
            "PLIP is not importable in this interpreter. Blender Gala's own "
            "detectors (blender_gala.find_interactions) use the same geometric "
            "criteria and need no extra install; use this module only when you "
            "want PLIP's exact output.\n"
            f"Underlying import error: {exc}"
        ) from exc


def from_pdb(
    filepath: str,
    target: Any = None,
    ligand: str | None = None,
) -> list[Interaction]:
    """Run PLIP over a PDB file and convert the result.

    Parameters
    ----------
    filepath : str
        Path to the complex, as PDB.
    target : AtomStructure, Molecule, or bpy.types.Object, optional
        The structure already loaded in Blender. Supplying it lets Gala map
        PLIP's atom serial numbers onto the loaded atoms, so the drawn lines
        land on the right atoms even if PLIP reordered or protonated the file.
    ligand : str, optional
        Restrict to one binding site, in PLIP's ``"RESNAME:CHAIN:RESNUM"``
        form. ``None`` returns every site.

    Returns
    -------
    list[Interaction]

    Raises
    ------
    ImportError
        If PLIP is not installed.
    KeyError
        If ``ligand`` names a site PLIP did not find.
    """
    preparation = _require_plip()

    complex_ = preparation.PDBComplex()
    complex_.load_pdb(filepath)
    complex_.analyze()

    sites = complex_.interaction_sets
    if ligand is not None:
        if ligand not in sites:
            raise KeyError(
                f"PLIP found no binding site {ligand!r}; available: {sorted(sites)}"
            )
        selected = {ligand: sites[ligand]}
    else:
        selected = sites

    found: list[Interaction] = []
    for site in selected.values():
        found.extend(from_plip_site(site, target))
    return found


def from_plip_site(site: Any, target: Any = None) -> list[Interaction]:
    """Convert one PLIP ``PLInteraction`` object into Gala interactions.

    Parameters
    ----------
    site : plip.basic.interactions.PLInteraction
        A single binding site's interaction set.
    target : AtomStructure, Molecule, or bpy.types.Object, optional
        Loaded structure used to resolve atom serial numbers to scene
        positions. Without it, PLIP's own coordinates are used, which are in
        ångström and will need scaling before they can be drawn.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target) if target is not None else None
    serial_to_index = _serial_lookup(structure) if structure is not None else {}
    positions = structure.world_positions() if structure is not None else None
    scale = structure.world_scale if structure is not None else 1.0

    found: list[Interaction] = []
    for attribute, kind in PLIP_KIND_MAP.items():
        for record in getattr(site, attribute, ()) or ():
            converted = _convert(
                record, kind, serial_to_index, positions, scale, structure
            )
            if converted is not None:
                found.append(converted)
    return found


def _serial_lookup(structure: AtomStructure) -> dict[int, int]:
    """Map PDB atom serial numbers to atom indices in the loaded structure."""
    serials = getattr(structure.array, "atom_id", None)
    if serials is None:
        return {}
    return {int(serial): index for index, serial in enumerate(np.asarray(serials))}


def _coords_of(
    record: Any, names: tuple[str, ...], exclude: str | None = None
) -> tuple[np.ndarray | None, str | None]:
    """Coordinates from the first of ``names`` the record carries, and which one."""
    for name in names:
        if name == exclude:
            continue
        value = getattr(record, name, None)
        if value is None:
            continue
        coords = getattr(value, "coords", value)
        try:
            array = np.asarray(coords, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.shape == (3,) and np.isfinite(array).all():
            return array, name
    return None, None


def _serial_of(
    record: Any, names: tuple[str, ...], exclude: str | None = None
) -> tuple[int | None, str | None]:
    """Atom serial from the first of ``names`` the record carries, and which one."""
    for name in names:
        if name == exclude:
            continue
        value = getattr(record, name, None)
        if value is None:
            continue
        serial = getattr(value, "idx", None)
        if serial is None:
            serial = getattr(value, "atom_orig_idx", None)
        if serial is None:
            continue
        try:
            return int(serial), name
        except (TypeError, ValueError):
            # PLIP's own records always carry an integer index; something
            # else's do not, and one field being unreadable is no reason to
            # give up on the record.
            continue
    return None, None


def _number(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` if it is not one."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


# The two sides overlap on 'a' and 'd': PLIP names the acceptor of a hydrogen
# bond `a` and the *aromatic* ring of a pi-cation `a`, and `d` is a donor in
# one record type and a distance-bearing partner in another. Which side each
# resolves to therefore depends on what the record carries, and the field one
# side used is excluded from the other's search — otherwise a record with only
# one of them puts both ends of the interaction on the same atom, and the
# figure gets a line of zero length.
_SIDE_A_FIELDS = ("d", "donor", "a", "bsatom", "atom", "protisdon", "restype")
_SIDE_B_FIELDS = ("a", "acceptor", "ligatom", "d", "metal", "charge")


def _convert(
    record: Any,
    kind: str,
    serial_to_index: dict[int, int],
    positions: np.ndarray | None,
    scale: float,
    structure: AtomStructure | None,
) -> Interaction | None:
    """Best-effort conversion of one PLIP record.

    PLIP's record types are namedtuples whose fields differ per interaction,
    so this reads defensively and skips anything it cannot place. Nothing a
    record carries is trusted to be the type its name suggests: a field that
    does not read as a number is treated as absent, because one unreadable
    value is no reason to drop a whole binding site's worth of interactions.
    """
    serial_a, field_a = _serial_of(record, _SIDE_A_FIELDS)
    serial_b, _ = _serial_of(record, _SIDE_B_FIELDS, exclude=field_a)

    index_a = serial_to_index.get(serial_a) if serial_a is not None else None
    index_b = serial_to_index.get(serial_b) if serial_b is not None else None

    if positions is not None and index_a is not None and index_b is not None:
        point_a = positions[index_a]
        point_b = positions[index_b]
    else:
        raw_a, coord_field_a = _coords_of(record, _SIDE_A_FIELDS)
        raw_b, _ = _coords_of(record, _SIDE_B_FIELDS, exclude=coord_field_a)
        if raw_a is None or raw_b is None:
            return None
        point_a = raw_a * scale
        point_b = raw_b * scale

    distance = _number(getattr(record, "distance", None))
    if distance is None:
        distance = _number(getattr(record, "distance_ah", None)) or _number(
            getattr(record, "distance_aw", None)
        )
    if distance is None:
        distance = float(np.linalg.norm((point_a - point_b) / max(scale, 1e-12)))

    angle = _number(getattr(record, "angle", None))

    if structure is not None and index_a is not None and index_b is not None:
        label = (
            f"{structure.atom_label(index_a, '{chain}/{resn}{resi}/{name}')} - "
            f"{structure.atom_label(index_b, '{chain}/{resn}{resi}/{name}')}"
        )
    else:
        label = f"{getattr(record, 'restype', '?')}{getattr(record, 'resnr', '')}"

    return Interaction(
        kind=kind,
        atoms_a=(index_a,) if index_a is not None else (),
        atoms_b=(index_b,) if index_b is not None else (),
        point_a=np.asarray(point_a, dtype=float),
        point_b=np.asarray(point_b, dtype=float),
        distance=distance,
        angle=angle,
        label=label,
    )
