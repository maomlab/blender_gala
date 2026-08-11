"""Geometric detection of non-covalent interactions.

Implemented natively with numpy and scipy rather than by shelling out to PLIP
(SPECIFICATION D-18): PLIP depends on OpenBabel, which cannot be assumed inside
Blender's interpreter. The geometric criteria follow PLIP's published defaults
so results are comparable, and :mod:`blender_gala.interactions.plip` can ingest
a real PLIP result when a user does have it installed.

Everything in this module operates on :class:`~blender_gala.core.entity.AtomStructure`
and returns plain dataclasses, so the science is testable without Blender.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import chemistry
from ..core.entity import AtomStructure
from ..core.exceptions import GalaError

__all__ = [
    "DEFAULT_CRITERIA",
    "INTERACTION_KINDS",
    "Interaction",
    "InteractionCriteria",
    "atom_contacts",
    "cation_pi",
    "find_interactions",
    "halogen_bonds",
    "hydrogen_bonds",
    "hydrophobic_contacts",
    "metal_coordination",
    "pi_stacking",
    "polar_contacts",
    "salt_bridges",
]

INTERACTION_KINDS = (
    "hbond",
    "polar",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "cation_pi",
    "halogen",
    "metal",
)


@dataclass(frozen=True)
class Interaction:
    """One detected interaction.

    Attributes
    ----------
    kind : str
        One of :data:`INTERACTION_KINDS`.
    atoms_a, atoms_b : tuple[int, ...]
        Atom indices on each side. Usually one atom each; ring-based
        interactions carry every ring atom, and charged groups every atom of
        the group.
    point_a, point_b : numpy.ndarray
        World-space endpoints in Blender units — the atom positions, or the
        ring/charge-group centroids.
    distance : float
        Distance between the endpoints, in ångström.
    angle : float or None
        The defining angle in degrees, where the criterion has one.
    label : str
        Human-readable description, e.g. ``"ASP102/OD2 - HIS57/ND1"``.
    """

    kind: str
    atoms_a: tuple[int, ...]
    atoms_b: tuple[int, ...]
    point_a: np.ndarray
    point_b: np.ndarray
    distance: float
    angle: float | None = None
    label: str = ""

    def __str__(self) -> str:
        angle = f", {self.angle:.0f} deg" if self.angle is not None else ""
        return f"{self.kind}: {self.label} ({self.distance:.2f} A{angle})"


@dataclass(frozen=True)
class InteractionCriteria:
    """Geometric cutoffs, in ångström and degrees.

    Defaults follow PLIP (SPECIFICATION §6.2). Every value is a keyword
    argument of the detection functions, so a user who wants PyMOL's more
    permissive polar-contact cutoff can pass ``polar_max=3.6`` without editing
    anything.

    A value that is not a distance or an angle — negative, or ``nan`` — is
    never satisfied, so a mistyped criterion tightens a detector to nothing
    rather than loosening it. The alternative is worse: a sign typo that
    reports *more* interactions looks like a discovery.

    Attributes
    ----------
    hbond_h_acceptor_max : float
        Maximum H to acceptor distance.
    hbond_donor_acceptor_max : float
        Maximum donor to acceptor distance.
    hbond_angle_min : float
        Minimum donor-H-acceptor angle.
    polar_max : float
        Maximum heavy-atom donor to acceptor distance when no hydrogens are
        present.
    polar_min : float
        Minimum such distance; below this the atoms are almost certainly bonded.
    salt_bridge_max : float
        Maximum distance between charged-group centroids.
    hydrophobic_max : float
        Maximum carbon to carbon distance.
    hydrophobic_min : float
        Minimum such distance.
    pi_stack_max : float
        Maximum ring-centroid separation.
    pi_stack_parallel_angle : float
        Maximum inter-plane angle for parallel (sandwich) stacking.
    pi_stack_t_angle_min : float
        Minimum inter-plane angle for T-shaped stacking.
    pi_stack_offset_max : float
        Maximum lateral offset between ring centroids.
    cation_pi_max : float
        Maximum cation to ring-centroid distance.
    cation_pi_offset_max : float
        Maximum lateral offset of the cation from the ring axis.
    halogen_max : float
        Maximum halogen to acceptor distance.
    halogen_donor_angle : tuple[float, float]
        Acceptable C-X to acceptor angle range.
    metal_max : float
        Maximum metal to coordinating-atom distance.
    """

    hbond_h_acceptor_max: float = 2.5
    hbond_donor_acceptor_max: float = 3.5
    hbond_angle_min: float = 130.0
    polar_max: float = 3.5
    polar_min: float = 2.2
    salt_bridge_max: float = 5.5
    hydrophobic_max: float = 4.0
    hydrophobic_min: float = 2.8
    pi_stack_max: float = 5.5
    pi_stack_parallel_angle: float = 30.0
    pi_stack_t_angle_min: float = 60.0
    pi_stack_offset_max: float = 2.0
    cation_pi_max: float = 6.0
    cation_pi_offset_max: float = 2.0
    halogen_max: float = 4.0
    halogen_donor_angle: tuple[float, float] = (140.0, 180.0)
    metal_max: float = 3.0


DEFAULT_CRITERIA = InteractionCriteria()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passes(value: float, minimum: float = -np.inf, maximum: float = np.inf) -> bool:
    """Whether ``value`` satisfies a criterion's bounds.

    Written as a positive test so that a ``nan`` on either side fails it. The
    negated form a criterion check falls into naturally — ``if value > maximum:
    continue`` — accepts what it cannot judge, because every comparison
    against ``nan`` is False, and a mistyped criterion then loosens the
    detector silently.
    """
    return bool(value >= minimum) and bool(value <= maximum)


def _require_finite(coord: np.ndarray, *groups: np.ndarray) -> None:
    """Refuse coordinates that are not numbers, before scipy does it opaquely.

    An atom absent from the state a structure was loaded for carries ``nan``
    coordinates, so this is reachable with nothing wrong with the file.
    cKDTree reports it as "data must be finite", which names neither the
    add-on nor the atom it came from.
    """
    for indices in groups:
        broken = indices[~np.isfinite(coord[indices]).all(axis=1)]
        if broken.size:
            others = (
                f", and so do {broken.size - 1} more of the atoms being measured"
                if broken.size > 1
                else ""
            )
            raise GalaError(
                f"atom {int(broken[0])} has a coordinate that is not a number"
                f"{others}; no distance can be measured from it. An atom "
                "missing from the state a structure was loaded for carries nan "
                "coordinates."
            )


def _pairs_within(
    coord: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    cutoff: float,
) -> list[tuple[int, int]]:
    """Return index pairs (a, b) closer than ``cutoff`` ångström.

    Raises
    ------
    GalaError
        If any atom being measured has a non-finite coordinate.
    """
    if indices_a.size == 0 or indices_b.size == 0:
        return []

    # A cutoff that is not a positive distance matches nothing. cKDTree takes
    # the absolute value of a negative radius, so without this a sign typo
    # searches *wider* than the default while the numpy fallback below — which
    # compares `distances <= cutoff` — finds nothing, and the two paths
    # disagree about the same structure. nan is the same mistake seen from the
    # other side: every comparison against it is False.
    if np.isnan(cutoff) or cutoff <= 0.0:
        return []

    _require_finite(coord, indices_a, indices_b)

    try:
        from scipy.spatial import cKDTree

        tree_b = cKDTree(coord[indices_b])
        neighbours = tree_b.query_ball_point(coord[indices_a], r=cutoff)
        return [
            (int(indices_a[i]), int(indices_b[j]))
            for i, group in enumerate(neighbours)
            for j in group
        ]
    except ImportError:  # pragma: no cover - scipy ships with Blender + MN
        deltas = coord[indices_a][:, None, :] - coord[indices_b][None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        rows, cols = np.nonzero(distances <= cutoff)
        return [
            (int(indices_a[r]), int(indices_b[c]))
            for r, c in zip(rows, cols, strict=False)
        ]


def _angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle a-vertex-b in degrees."""
    va = a - vertex
    vb = b - vertex
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _residue_ids(structure: AtomStructure) -> np.ndarray:
    return structure.context.residue_key


def _elements(structure: AtomStructure) -> np.ndarray:
    from ..core.selection import _elements as element_array

    return element_array(structure.context)


def _describe_pair(structure: AtomStructure, i: int, j: int) -> str:
    return (
        f"{structure.atom_label(i, '{chain}/{resn}{resi}/{name}')} - "
        f"{structure.atom_label(j, '{chain}/{resn}{resi}/{name}')}"
    )


def _make(
    structure: AtomStructure,
    kind: str,
    i: int,
    j: int,
    positions: np.ndarray,
    coord: np.ndarray,
    angle: float | None = None,
) -> Interaction:
    return Interaction(
        kind=kind,
        atoms_a=(int(i),),
        atoms_b=(int(j),),
        point_a=positions[i],
        point_b=positions[j],
        distance=float(np.linalg.norm(coord[i] - coord[j])),
        angle=angle,
        label=_describe_pair(structure, i, j),
    )


def _bonded_hydrogens(
    structure: AtomStructure, coord: np.ndarray
) -> dict[int, list[int]]:
    """Map each heavy atom to the hydrogens covalently bound to it.

    Hydrogen positions rather than an explicit bond table: a hydrogen is
    assigned to its nearest heavy atom within 1.3 Å, which is unambiguous
    because no two heavy atoms are that close to the same hydrogen.
    """
    elements = _elements(structure)
    hydrogens = np.flatnonzero(np.isin(elements, list(chemistry.HYDROGEN_ELEMENTS)))
    heavy = np.flatnonzero(~np.isin(elements, list(chemistry.HYDROGEN_ELEMENTS)))
    mapping: dict[int, list[int]] = {}
    if hydrogens.size == 0 or heavy.size == 0:
        return mapping

    for h, parent in _pairs_within(coord, hydrogens, heavy, 1.3):
        distance = float(np.linalg.norm(coord[h] - coord[parent]))
        current = mapping.setdefault(parent, [])
        current.append(h)
        # Keep only the closest parent per hydrogen.
        for other, children in mapping.items():
            if other == parent or h not in children:
                continue
            if float(np.linalg.norm(coord[h] - coord[other])) <= distance:
                current.remove(h)
            else:
                children.remove(h)
    return mapping


def _ring_atoms(structure: AtomStructure) -> list[tuple[int, ...]]:
    """Return aromatic rings, including perceived ligand rings."""
    from .perception import aromatic_rings

    return aromatic_rings(structure)


def _ring_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the ``(centroid, unit normal)`` of a ring, or ``None`` if it has none.

    A ring whose atoms are coincident, collinear or non-finite has no plane,
    and the normal an SVD hands back for one is an arbitrary member of a
    perpendicular family — every angle measured against it is meaningless. A
    ring that has no plane is not a ring, so callers drop it.
    """
    from .perception import _plane_of

    return _plane_of(points)


def _charged_groups(structure: AtomStructure, positive: bool) -> list[tuple[int, ...]]:
    """Return formally charged groups as tuples of atom indices."""
    from .perception import charged_groups

    return charged_groups(structure, positive)


def _apolar_carbons(structure: AtomStructure, coord: np.ndarray) -> np.ndarray:
    """Carbons with no polar heavy atom bonded to them."""
    elements = _elements(structure)
    carbons = np.flatnonzero(elements == "C")
    polar = np.flatnonzero(np.isin(elements, list(chemistry.POLAR_ELEMENTS)))
    if carbons.size == 0 or polar.size == 0:
        return carbons
    bonded = {a for a, _ in _pairs_within(coord, carbons, polar, 1.8)}
    return np.array([c for c in carbons if int(c) not in bonded], dtype=int)


def _resolve(structure: AtomStructure, selection: Any, name: str) -> np.ndarray:
    mask = structure.select(selection)
    return np.flatnonzero(mask)


def _filter_pairs(
    structure: AtomStructure,
    pairs: Iterable[tuple[int, int]],
    exclude_same_residue: bool,
    exclude_bonded: bool = True,
    coord: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    residues = _residue_ids(structure)
    out = []
    seen = set()
    for i, j in pairs:
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        if exclude_same_residue and residues[i] == residues[j]:
            continue
        if (
            exclude_bonded
            and coord is not None
            and float(np.linalg.norm(coord[i] - coord[j])) < 1.8
        ):
            continue
        seen.add(key)
        out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def hydrogen_bonds(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
    exclude_same_residue: bool = True,
) -> list[Interaction]:
    """Detect hydrogen bonds using explicit hydrogens.

    Requires hydrogens in the structure. Crystal structures usually have none,
    in which case :func:`polar_contacts` is the right function — it applies the
    heavy-atom criterion that PyMOL's ``polar_contacts`` uses.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to analyse.
    selection_a, selection_b : str or array, optional
        The two sides of the interaction. Pass the same selection twice for
        internal hydrogen bonds.
    criteria : InteractionCriteria, optional
        Geometric cutoffs.
    exclude_same_residue : bool, optional
        Skip pairs within one residue.

    Returns
    -------
    list[Interaction]
        Hydrogen bonds, with ``angle`` set to the D-H-A angle.
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()
    elements = _elements(structure)

    donors_h = _bonded_hydrogens(structure, coord)
    if not donors_h:
        return []

    side_a = _resolve(structure, selection_a, "selection_a")
    side_b = _resolve(structure, selection_b, "selection_b")

    donor_mask = np.isin(elements, list(chemistry.DONOR_ELEMENTS))
    acceptor_mask = np.isin(elements, list(chemistry.ACCEPTOR_ELEMENTS))

    found: list[Interaction] = []
    seen: set[tuple[int, int]] = set()

    for donors, acceptors in (
        (side_a[donor_mask[side_a]], side_b[acceptor_mask[side_b]]),
        (side_b[donor_mask[side_b]], side_a[acceptor_mask[side_a]]),
    ):
        pairs = _pairs_within(
            coord, donors, acceptors, criteria.hbond_donor_acceptor_max
        )
        for donor, acceptor in _filter_pairs(
            structure, pairs, exclude_same_residue, coord=coord
        ):
            hydrogens = donors_h.get(donor, [])
            best: tuple[float, int] | None = None
            for h in hydrogens:
                h_a = float(np.linalg.norm(coord[h] - coord[acceptor]))
                if not _passes(h_a, maximum=criteria.hbond_h_acceptor_max):
                    continue
                angle = _angle_between(coord[donor], coord[h], coord[acceptor])
                if not _passes(angle, minimum=criteria.hbond_angle_min):
                    continue
                if best is None or angle > best[0]:
                    best = (angle, h)
            if best is None:
                continue
            key = (min(donor, acceptor), max(donor, acceptor))
            if key in seen:
                continue
            seen.add(key)
            found.append(
                _make(structure, "hbond", donor, acceptor, positions, coord, best[0])
            )
    return found


def polar_contacts(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
    exclude_same_residue: bool = True,
) -> list[Interaction]:
    """Detect heavy-atom polar contacts, the no-hydrogens case.

    This is the criterion behind PyMOL's ``polar_contacts``: any N/O/S/F pair
    within ~3.5 Å that is not covalently bonded. It over-reports compared with
    a true hydrogen-bond calculation — the geometry alone cannot tell a donor
    from an acceptor — which is exactly the trade-off crystallographers accept
    when their model has no hydrogens.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to analyse.
    selection_a, selection_b : str or array, optional
        The two sides of the interaction.
    criteria : InteractionCriteria, optional
        Geometric cutoffs.
    exclude_same_residue : bool, optional
        Skip pairs within one residue.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()
    elements = _elements(structure)
    polar = np.isin(elements, list(chemistry.ACCEPTOR_ELEMENTS))

    side_a = _resolve(structure, selection_a, "selection_a")
    side_b = _resolve(structure, selection_b, "selection_b")
    pairs = _pairs_within(
        coord, side_a[polar[side_a]], side_b[polar[side_b]], criteria.polar_max
    )

    found = []
    for i, j in _filter_pairs(structure, pairs, exclude_same_residue, coord=coord):
        distance = float(np.linalg.norm(coord[i] - coord[j]))
        if not _passes(distance, minimum=criteria.polar_min):
            continue
        found.append(_make(structure, "polar", i, j, positions, coord))
    return found


def salt_bridges(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
) -> list[Interaction]:
    """Detect salt bridges between oppositely charged groups.

    Measured centroid-to-centroid rather than atom-to-atom, because a
    carboxylate or a guanidinium delocalises its charge over the whole group.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()

    side_a = {int(i) for i in _resolve(structure, selection_a, "selection_a")}
    side_b = {int(i) for i in _resolve(structure, selection_b, "selection_b")}

    positives = _charged_groups(structure, positive=True)
    negatives = _charged_groups(structure, positive=False)
    residues = _residue_ids(structure)

    found = []
    for pos in positives:
        for neg in negatives:
            if set(pos) & set(neg):
                continue
            # A residue cannot form a salt bridge with itself; without this an
            # amino acid's own alpha-amino and alpha-carboxy groups pair up.
            if residues[pos[0]] == residues[neg[0]]:
                continue
            if not (
                (set(pos) & side_a and set(neg) & side_b)
                or (set(pos) & side_b and set(neg) & side_a)
            ):
                continue
            centre_p = coord[list(pos)].mean(axis=0)
            centre_n = coord[list(neg)].mean(axis=0)
            distance = float(np.linalg.norm(centre_p - centre_n))
            if not _passes(distance, maximum=criteria.salt_bridge_max):
                continue
            found.append(
                Interaction(
                    kind="salt_bridge",
                    atoms_a=pos,
                    atoms_b=neg,
                    point_a=positions[list(pos)].mean(axis=0),
                    point_b=positions[list(neg)].mean(axis=0),
                    distance=distance,
                    label=_describe_pair(structure, pos[0], neg[0]),
                )
            )
    return found


def hydrophobic_contacts(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
    exclude_same_residue: bool = True,
) -> list[Interaction]:
    """Detect apolar carbon-carbon contacts.

    Only carbons with no polar heavy neighbour count, so the C-alpha of a
    charged residue does not register as hydrophobic.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()

    apolar = {int(i) for i in _apolar_carbons(structure, coord)}
    side_a = np.array(
        [i for i in _resolve(structure, selection_a, "a") if int(i) in apolar],
        dtype=int,
    )
    side_b = np.array(
        [i for i in _resolve(structure, selection_b, "b") if int(i) in apolar],
        dtype=int,
    )

    pairs = _pairs_within(coord, side_a, side_b, criteria.hydrophobic_max)
    found = []
    for i, j in _filter_pairs(structure, pairs, exclude_same_residue, coord=coord):
        distance = float(np.linalg.norm(coord[i] - coord[j]))
        if not _passes(distance, minimum=criteria.hydrophobic_min):
            continue
        found.append(_make(structure, "hydrophobic", i, j, positions, coord))
    return found


def pi_stacking(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
) -> list[Interaction]:
    """Detect parallel and T-shaped aromatic stacking.

    Both geometries are reported; ``angle`` carries the inter-plane angle, so
    a caller can separate sandwich from edge-to-face stacking.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()

    side_a = {int(i) for i in _resolve(structure, selection_a, "a")}
    side_b = {int(i) for i in _resolve(structure, selection_b, "b")}
    rings = _ring_atoms(structure)

    # Rings with no plane are dropped here rather than reported with an
    # arbitrary normal: coincident atoms would otherwise stack confidently at
    # 0 degrees, which is the most convincing wrong answer this module can
    # give.
    planes = []
    for ring in rings:
        plane = _ring_plane(coord[list(ring)])
        if plane is not None:
            planes.append((plane, ring))

    found = []
    for index, ((centre_i, normal_i), ring_i) in enumerate(planes):
        for (centre_j, normal_j), ring_j in planes[index + 1 :]:
            if not (
                (set(ring_i) & side_a and set(ring_j) & side_b)
                or (set(ring_i) & side_b and set(ring_j) & side_a)
            ):
                continue
            if set(ring_i) & set(ring_j):
                continue

            offset_vector = centre_j - centre_i
            distance = float(np.linalg.norm(offset_vector))
            if not _passes(distance, maximum=criteria.pi_stack_max):
                continue

            angle = float(
                np.degrees(
                    np.arccos(np.clip(abs(np.dot(normal_i, normal_j)), 0.0, 1.0))
                )
            )
            # Lateral offset: how far the second centroid sits from the first
            # ring's axis.
            offset = float(
                np.linalg.norm(
                    offset_vector - np.dot(offset_vector, normal_i) * normal_i
                )
            )

            parallel = _passes(angle, maximum=criteria.pi_stack_parallel_angle)
            t_shaped = _passes(angle, minimum=criteria.pi_stack_t_angle_min)
            if not (parallel or t_shaped):
                continue
            if parallel and not _passes(offset, maximum=criteria.pi_stack_offset_max):
                continue

            found.append(
                Interaction(
                    kind="pi_stacking",
                    atoms_a=ring_i,
                    atoms_b=ring_j,
                    point_a=positions[list(ring_i)].mean(axis=0),
                    point_b=positions[list(ring_j)].mean(axis=0),
                    distance=distance,
                    angle=angle,
                    label=_describe_pair(structure, ring_i[0], ring_j[0]),
                )
            )
    return found


def cation_pi(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
) -> list[Interaction]:
    """Detect cation-pi interactions between a positive charge and a ring.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()

    side_a = {int(i) for i in _resolve(structure, selection_a, "a")}
    side_b = {int(i) for i in _resolve(structure, selection_b, "b")}

    rings = _ring_atoms(structure)
    cations = _charged_groups(structure, positive=True)

    found = []
    for ring in rings:
        plane = _ring_plane(coord[list(ring)])
        if plane is None:
            continue
        centre, normal = plane
        for group in cations:
            if set(ring) & set(group):
                continue
            if not (
                (set(ring) & side_a and set(group) & side_b)
                or (set(ring) & side_b and set(group) & side_a)
            ):
                continue
            cation_centre = coord[list(group)].mean(axis=0)
            offset_vector = cation_centre - centre
            distance = float(np.linalg.norm(offset_vector))
            if not _passes(distance, maximum=criteria.cation_pi_max):
                continue
            offset = float(
                np.linalg.norm(offset_vector - np.dot(offset_vector, normal) * normal)
            )
            if not _passes(offset, maximum=criteria.cation_pi_offset_max):
                continue
            found.append(
                Interaction(
                    kind="cation_pi",
                    atoms_a=ring,
                    atoms_b=group,
                    point_a=positions[list(ring)].mean(axis=0),
                    point_b=positions[list(group)].mean(axis=0),
                    distance=distance,
                    label=_describe_pair(structure, ring[0], group[0]),
                )
            )
    return found


def halogen_bonds(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
) -> list[Interaction]:
    """Detect halogen bonds (the sigma-hole interaction of a C-X group).

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()
    elements = _elements(structure)

    halogen_mask = np.isin(elements, list(chemistry.HALOGENS))
    acceptor_mask = np.isin(elements, list(chemistry.ACCEPTOR_ELEMENTS))
    carbons = np.flatnonzero(elements == "C")

    side_a = _resolve(structure, selection_a, "a")
    side_b = _resolve(structure, selection_b, "b")

    found = []
    seen: set[tuple[int, int]] = set()
    for halogens, acceptors in (
        (side_a[halogen_mask[side_a]], side_b[acceptor_mask[side_b]]),
        (side_b[halogen_mask[side_b]], side_a[acceptor_mask[side_a]]),
    ):
        for x, acceptor in _pairs_within(
            coord, halogens, acceptors, criteria.halogen_max
        ):
            key = (min(x, acceptor), max(x, acceptor))
            if key in seen:
                continue
            attached = _pairs_within(coord, np.array([x]), carbons, 2.2)
            if not attached:
                continue
            carbon = attached[0][1]
            angle = _angle_between(coord[carbon], coord[x], coord[acceptor])
            low, high = criteria.halogen_donor_angle
            if not _passes(angle, minimum=low, maximum=high):
                continue
            seen.add(key)
            found.append(
                _make(structure, "halogen", x, acceptor, positions, coord, angle)
            )
    return found


def metal_coordination(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
) -> list[Interaction]:
    """Detect metal coordination bonds.

    Returns
    -------
    list[Interaction]
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()
    elements = _elements(structure)

    metals = np.flatnonzero(np.isin(elements, list(chemistry.METALS)))
    coordinating = np.isin(elements, list(chemistry.ACCEPTOR_ELEMENTS))

    side_a = _resolve(structure, selection_a, "a")
    side_b = _resolve(structure, selection_b, "b")
    allowed = {int(i) for i in side_a} | {int(i) for i in side_b}

    metals = np.array([m for m in metals if int(m) in allowed], dtype=int)
    partners = np.array(
        [i for i in np.flatnonzero(coordinating) if int(i) in allowed], dtype=int
    )

    found = []
    for metal, partner in _pairs_within(coord, metals, partners, criteria.metal_max):
        found.append(_make(structure, "metal", metal, partner, positions, coord))
    return found


def atom_contacts(
    target: Any,
    selection_a: Any,
    selection_b: Any,
    cutoff: float = 4.0,
    minimum: float = 0.0,
    exclude_same_residue: bool = True,
) -> list[Interaction]:
    """Find every atom pair between two selections within a distance cutoff.

    The escape hatch for custom criteria the named detectors do not cover.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to analyse.
    selection_a, selection_b : str or array
        The two sides.
    cutoff : float, optional
        Maximum distance in ångström.
    minimum : float, optional
        Minimum distance in ångström.
    exclude_same_residue : bool, optional
        Skip pairs within one residue.

    Returns
    -------
    list[Interaction]
        Interactions of kind ``"contact"``.
    """
    structure = AtomStructure.from_any(target)
    coord = structure.coord
    positions = structure.world_positions()

    side_a = _resolve(structure, selection_a, "a")
    side_b = _resolve(structure, selection_b, "b")
    pairs = _pairs_within(coord, side_a, side_b, cutoff)

    found = []
    for i, j in _filter_pairs(
        structure, pairs, exclude_same_residue, exclude_bonded=False
    ):
        distance = float(np.linalg.norm(coord[i] - coord[j]))
        if not _passes(distance, minimum=minimum):
            continue
        found.append(_make(structure, "contact", i, j, positions, coord))
    return found


_DETECTORS: dict[str, Callable[..., list[Interaction]]] = {
    "hbond": hydrogen_bonds,
    "polar": polar_contacts,
    "salt_bridge": salt_bridges,
    "hydrophobic": hydrophobic_contacts,
    "pi_stacking": pi_stacking,
    "cation_pi": cation_pi,
    "halogen": halogen_bonds,
    "metal": metal_coordination,
}


def find_interactions(
    target: Any,
    selection_a: Any = "all",
    selection_b: Any = "all",
    kinds: Sequence[str] | str = ("hbond", "polar", "salt_bridge"),
    criteria: InteractionCriteria = DEFAULT_CRITERIA,
    exclude_same_residue: bool = True,
) -> list[Interaction]:
    """Run several detectors and return the combined result.

    Interactions are always found *between two selections* (SPECIFICATION
    D-19), matching how one thinks about them: "what does this ligand touch".

    ``"hbond"`` falls back to :func:`polar_contacts` when the structure has no
    hydrogens, because asking for hydrogen bonds and silently getting nothing
    is the single most confusing outcome for a user working from a crystal
    structure.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to analyse.
    selection_a, selection_b : str or array, optional
        The two sides of the interaction.
    kinds : sequence of str or str, optional
        Any of :data:`INTERACTION_KINDS`, or ``"all"``.
    criteria : InteractionCriteria, optional
        Geometric cutoffs.
    exclude_same_residue : bool, optional
        Skip pairs within one residue.

    Returns
    -------
    list[Interaction]
        Sorted by kind, then distance.

    Raises
    ------
    ValueError
        If a requested kind is unknown.
    """
    structure = AtomStructure.from_any(target)

    if kinds == "all":
        requested: Sequence[str] = INTERACTION_KINDS
    elif isinstance(kinds, str):
        requested = (kinds,)
    else:
        requested = tuple(kinds)

    unknown = [k for k in requested if k not in _DETECTORS]
    if unknown:
        raise ValueError(
            f"unknown interaction kind(s) {unknown}; choose from {list(INTERACTION_KINDS)}"
        )

    has_hydrogens = bool(
        np.isin(_elements(structure), list(chemistry.HYDROGEN_ELEMENTS)).any()
    )

    results: list[Interaction] = []
    for kind in requested:
        detector = _DETECTORS[kind]
        if kind == "hbond" and not has_hydrogens:
            if "polar" in requested:
                continue  # polar_contacts is already being run
            detector = polar_contacts

        if detector in (
            salt_bridges,
            pi_stacking,
            cation_pi,
            halogen_bonds,
            metal_coordination,
        ):
            results.extend(detector(structure, selection_a, selection_b, criteria))
        else:
            results.extend(
                detector(
                    structure,
                    selection_a,
                    selection_b,
                    criteria,
                    exclude_same_residue=exclude_same_residue,
                )
            )

    results.sort(key=lambda item: (item.kind, item.distance))
    return results
