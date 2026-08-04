"""Chemical perception: bonds, rings and charged groups.

Standard residues are handled by lookup tables in
:mod:`blender_gala.core.chemistry` — exact and instant. Ligands are the hard
case: a table cannot know that a novel inhibitor has a thiazole ring or a
carboxylate, and the ligand is usually the thing a figure is *about*. So for
anything outside the tables, Gala perceives the chemistry from geometry.

A note on partial charges: Molecular Nodes stores force-field partial charges
in the ``charge`` annotation, where every backbone carbonyl carbon reads around
+0.6. Treating those as formal charges makes a salt-bridge detector fire on
every residue in the protein, so this module ignores them entirely and derives
formal charge from connectivity instead.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from ..core import chemistry
from ..core.entity import AtomStructure

__all__ = [
    "COVALENT_RADII",
    "aromatic_rings",
    "bond_graph",
    "charged_groups",
    "find_rings",
]

#: Covalent radii in ångström (Cordero et al. 2008), used for bond perception.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "D": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "SE": 1.20,
    "B": 0.84,
    "SI": 1.11,
    "NA": 1.66,
    "MG": 1.41,
    "K": 2.03,
    "CA": 1.76,
    "MN": 1.61,
    "FE": 1.52,
    "CO": 1.50,
    "NI": 1.24,
    "CU": 1.32,
    "ZN": 1.22,
}
_DEFAULT_RADIUS = 0.77

#: Multiplier on the sum of covalent radii. 1.25 is the usual tolerance: loose
#: enough for a long C-S bond, tight enough to exclude a close contact.
_BOND_TOLERANCE = 1.25

#: Metals are excluded from bond perception. Coordination distances overlap
#: with covalent ones, so including them fuses a whole binding site into one
#: connected component and ruins ring perception.
_NON_BONDING = frozenset(chemistry.METALS) - {"AL", "SI"}


def _elements(structure: AtomStructure) -> np.ndarray:
    from ..core.selection import _elements as element_array

    return element_array(structure.context)


def bond_graph(
    structure: AtomStructure,
    include_hydrogen: bool = False,
    subset: np.ndarray | None = None,
) -> dict[int, list[int]]:
    """Perceive covalent bonds from interatomic distances.

    Two atoms are bonded when their separation is below
    ``1.25 * (r_i + r_j)`` using covalent radii. Metals are excluded, because
    their coordination distances overlap with covalent bond lengths and would
    merge a whole metal site into one component.

    Parameters
    ----------
    structure : AtomStructure
        Structure to analyse.
    include_hydrogen : bool, optional
        Include hydrogens in the graph. Off by default: ring and charge
        perception work on the heavy-atom skeleton.
    subset : numpy.ndarray, optional
        Atom indices to restrict the graph to.

    Returns
    -------
    dict[int, list[int]]
        Adjacency list. Atoms with no bonds are absent.
    """
    coord = structure.coord
    elements = _elements(structure)

    candidates = np.arange(structure.n_atoms) if subset is None else np.asarray(subset)
    keep = ~np.isin(elements[candidates], list(_NON_BONDING))
    if not include_hydrogen:
        keep &= ~np.isin(elements[candidates], list(chemistry.HYDROGEN_ELEMENTS))
    candidates = candidates[keep]

    graph: dict[int, list[int]] = {}
    if candidates.size < 2:
        return graph

    radii = np.array(
        [COVALENT_RADII.get(e, _DEFAULT_RADIUS) for e in elements[candidates]]
    )
    max_bond = _BOND_TOLERANCE * 2.0 * float(radii.max())

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coord[candidates])
        pairs = tree.query_pairs(r=max_bond, output_type="ndarray")
    except ImportError:  # pragma: no cover
        deltas = coord[candidates][:, None, :] - coord[candidates][None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        rows, cols = np.nonzero(np.triu(distances <= max_bond, k=1))
        pairs = np.column_stack([rows, cols])

    for local_i, local_j in pairs:
        limit = _BOND_TOLERANCE * (radii[local_i] + radii[local_j])
        i, j = int(candidates[local_i]), int(candidates[local_j])
        if float(np.linalg.norm(coord[i] - coord[j])) > limit:
            continue
        graph.setdefault(i, []).append(j)
        graph.setdefault(j, []).append(i)

    return graph


def _shortest_cycle_through_edge(
    graph: dict[int, list[int]], start: int, end: int, max_length: int
) -> tuple[int, ...] | None:
    """Shortest path from ``start`` to ``end`` avoiding the direct edge."""
    queue: deque[tuple[int, tuple[int, ...]]] = deque([(start, (start,))])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if len(path) >= max_length:
            continue
        for neighbour in graph.get(node, ()):
            if neighbour == end:
                if len(path) >= 3:  # a ring needs at least three atoms
                    return (*path, end)
                continue
            if neighbour in visited or neighbour == start:
                continue
            visited.add(neighbour)
            queue.append((neighbour, (*path, neighbour)))
    return None


def find_rings(
    structure: AtomStructure,
    sizes: Sequence[int] = (5, 6),
    subset: np.ndarray | None = None,
    graph: dict[int, list[int]] | None = None,
) -> list[tuple[int, ...]]:
    """Find small rings by shortest-cycle search over the bond graph.

    Every bond is temporarily cut and the shortest remaining path between its
    two atoms is found; a path of the right length closes a ring. That
    reproduces the smallest set of smallest rings for the fused and unfused
    5-and-6-membered systems that make up essentially all drug-like ligands,
    without needing a full SSSR implementation.

    Parameters
    ----------
    structure : AtomStructure
        Structure to analyse.
    sizes : sequence of int, optional
        Ring sizes to look for.
    subset : numpy.ndarray, optional
        Restrict to these atom indices.
    graph : dict[int, list[int]], optional
        A pre-computed bond graph.

    Returns
    -------
    list[tuple[int, ...]]
        Each ring as a tuple of atom indices, deduplicated.
    """
    graph = graph if graph is not None else bond_graph(structure, subset=subset)
    max_size = max(sizes) if sizes else 6

    rings: dict[frozenset[int], tuple[int, ...]] = {}
    for node, neighbours in graph.items():
        for neighbour in neighbours:
            if neighbour <= node:
                continue
            cycle = _shortest_cycle_through_edge(graph, node, neighbour, max_size)
            if cycle is None or len(cycle) not in sizes:
                continue
            rings.setdefault(frozenset(cycle), tuple(cycle))
    return list(rings.values())


def _is_planar(points: np.ndarray, tolerance: float = 0.25) -> bool:
    """Whether every point lies within ``tolerance`` ångström of a best-fit plane."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2]
    return bool(np.abs((points - centroid) @ normal).max() <= tolerance)


def aromatic_rings(structure: AtomStructure) -> list[tuple[int, ...]]:
    """Return every aromatic ring, from tables for polymers and geometry for ligands.

    Standard residues use :data:`blender_gala.core.chemistry.AROMATIC_RINGS`,
    which is exact. Everything else is perceived: rings of 5 or 6 atoms, all of
    them C/N/O/S, and planar to within 0.25 A. Planarity is a good proxy for
    aromaticity here because a saturated ring of the same size puckers well
    beyond that.

    Parameters
    ----------
    structure : AtomStructure
        Structure to analyse.

    Returns
    -------
    list[tuple[int, ...]]
        Rings as tuples of atom indices.
    """
    coord = structure.coord
    res_names = structure.context.upper("res_name")
    atom_names = structure.context.upper("atom_name")
    residues = structure.context.residue_key
    elements = _elements(structure)

    rings: list[tuple[int, ...]] = []
    tabled = np.zeros(structure.n_atoms, dtype=bool)

    for residue in np.unique(residues):
        member = np.flatnonzero(residues == residue)
        if member.size == 0:
            continue
        table = chemistry.AROMATIC_RINGS.get(res_names[member[0]])
        if table is None:
            continue
        tabled[member] = True
        for ring_names in table:
            members = tuple(int(i) for i in member if atom_names[i] in ring_names)
            if len(members) >= len(ring_names) - 1:
                rings.append(members)

    # Geometric perception for everything the tables did not cover.
    remaining = np.flatnonzero(~tabled & np.isin(elements, ["C", "N", "O", "S"]))
    if remaining.size >= 5:
        for ring in find_rings(structure, sizes=(5, 6), subset=remaining):
            if _is_planar(coord[list(ring)]):
                rings.append(ring)

    return rings


def _neighbour_elements(
    graph: dict[int, list[int]], elements: np.ndarray, atom: int
) -> list[str]:
    return [elements[n] for n in graph.get(atom, ())]


def charged_groups(structure: AtomStructure, positive: bool) -> list[tuple[int, ...]]:
    """Find formally charged groups.

    Standard residues come from
    :data:`blender_gala.core.chemistry.POSITIVE_GROUPS` and
    :data:`~blender_gala.core.chemistry.NEGATIVE_GROUPS`. Ligand groups are
    perceived from connectivity:

    * negative — carboxylate (C bonded to exactly two O), phosphate and
      sulfate (P or S bonded to three or more O);
    * positive — guanidinium and amidinium (C bonded to two or three N in a
      planar arrangement), and quaternary nitrogen (N with four heavy
      neighbours).

    Parameters
    ----------
    structure : AtomStructure
        Structure to analyse.
    positive : bool
        Which sign to look for.

    Returns
    -------
    list[tuple[int, ...]]
        Each group as a tuple of atom indices.
    """
    table = chemistry.POSITIVE_GROUPS if positive else chemistry.NEGATIVE_GROUPS
    res_names = structure.context.upper("res_name")
    atom_names = structure.context.upper("atom_name")
    residues = structure.context.residue_key
    elements = _elements(structure)

    groups: list[tuple[int, ...]] = []
    tabled = np.zeros(structure.n_atoms, dtype=bool)
    known_residues = (
        chemistry.AMINO_ACIDS | chemistry.NUCLEOTIDES | chemistry.SOLVENT_NAMES
    )

    for residue in np.unique(residues):
        member = np.flatnonzero(residues == residue)
        if member.size == 0:
            continue
        res_name = res_names[member[0]]
        if res_name in known_residues:
            tabled[member] = True
            names = table.get(res_name)
            if names is not None:
                members = tuple(int(i) for i in member if atom_names[i] in names)
                if members:
                    groups.append(members)

    remaining = np.flatnonzero(~tabled)
    if remaining.size == 0:
        return groups

    graph = bond_graph(structure, subset=remaining)
    for raw_index in remaining:
        atom = int(raw_index)
        element = elements[atom]
        neighbours = graph.get(atom, [])
        neighbour_elements = _neighbour_elements(graph, elements, atom)

        if positive:
            n_nitrogen = neighbour_elements.count("N")
            if element == "C" and n_nitrogen >= 2:
                # Guanidinium or amidinium: charge is delocalised over C and Ns.
                groups.append(
                    (atom, *(int(n) for n in neighbours if elements[n] == "N"))
                )
            elif element == "N" and len(neighbours) >= 4:
                groups.append((atom,))
        else:
            n_oxygen = neighbour_elements.count("O")
            if element == "C" and n_oxygen == 2:
                # Carboxylate: both oxygens terminal (no further heavy bonds).
                oxygens = [int(n) for n in neighbours if elements[n] == "O"]
                if all(len(graph.get(o, [])) == 1 for o in oxygens):
                    groups.append((atom, *oxygens))
            elif element in ("P", "S") and n_oxygen >= 3:
                groups.append(
                    (atom, *(int(n) for n in neighbours if elements[n] == "O"))
                )

    return groups
