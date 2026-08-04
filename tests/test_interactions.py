"""Tests for interaction detection and chemical perception.

The fixture ``site.pdb`` was built with one clean example of each interaction
at exactly known geometry (see ``tests/data/make_fixtures.py``), so these tests
assert distances and angles rather than just counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from blender_gala.interactions import detect, perception


def labels(structure, group):
    return sorted(structure.atom_label(i, "{resn}{resi}/{name}") for i in group)


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


def test_bond_graph_finds_the_ligand_ring(site):
    graph = perception.bond_graph(site, subset=site.indices("resn LIG"))
    ring = site.indices("resn LIG and name C1+C2+C3+C4+C5+C6")
    # Every ring atom has exactly two ring neighbours.
    for atom in ring:
        assert len(graph.get(int(atom), [])) == 2


def test_bond_graph_excludes_metals(site):
    graph = perception.bond_graph(site)
    zinc = int(site.one_index("resn ZN"))
    assert zinc not in graph


def test_aromatic_rings_found_by_table_and_by_geometry(site):
    rings = perception.aromatic_rings(site)
    named = {frozenset(labels(site, ring)) for ring in rings}

    phe = frozenset(f"PHE4/{name}" for name in ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"))
    lig = frozenset(f"LIG1/C{i}" for i in range(1, 7))

    assert phe in named, "the PHE ring should come from the residue table"
    assert lig in named, "the ligand ring should be perceived geometrically"


def test_ligand_ring_is_planar(site):
    ring = site.indices("resn LIG and name C1+C2+C3+C4+C5+C6")
    assert perception._is_planar(site.coord[ring])


def test_charged_groups_use_the_residue_table(site):
    positive = perception.charged_groups(site, positive=True)
    negative = perception.charged_groups(site, positive=False)

    positive_labels = [labels(site, group) for group in positive]
    negative_labels = [labels(site, group) for group in negative]

    assert ["LYS3/NZ"] in positive_labels
    assert sorted(["ARG6/NE", "ARG6/CZ", "ARG6/NH1", "ARG6/NH2"]) in positive_labels
    assert sorted(["ASP2/CG", "ASP2/OD1", "ASP2/OD2"]) in negative_labels


def test_partial_charges_do_not_create_charged_groups(site):
    """Molecular Nodes stores partial charges; treating them as formal charges
    would make every backbone carbonyl a charged group."""
    groups = perception.charged_groups(site, positive=True)
    flat = {label for group in groups for label in labels(site, group)}
    assert not any(label.endswith("/C") for label in flat)
    assert not any(label.endswith("/O") for label in flat)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def test_hydrogen_bond_geometry(site):
    bonds = detect.hydrogen_bonds(site)
    assert len(bonds) == 1
    bond = bonds[0]
    assert bond.kind == "hbond"
    assert bond.distance == pytest.approx(2.80, abs=1e-3)
    assert bond.angle == pytest.approx(180.0, abs=1e-3)
    assert "SER1/OG" in bond.label
    assert "ASP2/OD1" in bond.label


def test_polar_contacts_include_the_hydrogen_bond(site):
    contacts = detect.polar_contacts(site)
    distances = {round(c.distance, 2) for c in contacts}
    assert 2.80 in distances
    assert all(c.distance <= detect.DEFAULT_CRITERIA.polar_max for c in contacts)


def test_polar_contacts_exclude_bonded_atoms(site):
    for contact in detect.polar_contacts(site):
        assert contact.distance >= detect.DEFAULT_CRITERIA.polar_min


def test_salt_bridge_is_found_once(site):
    bridges = detect.salt_bridges(site)
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge.distance == pytest.approx(4.40, abs=0.05)
    assert "LYS3" in bridge.label
    assert "ASP2" in bridge.label


def test_salt_bridges_are_not_intra_residue(site):
    residues = site.context.residue_key
    for bridge in detect.salt_bridges(site):
        assert residues[bridge.atoms_a[0]] != residues[bridge.atoms_b[0]]


def test_pi_stacking_geometry(site):
    stacks = detect.pi_stacking(site)
    assert len(stacks) == 1
    stack = stacks[0]
    assert stack.distance == pytest.approx(3.80, abs=1e-3)
    assert stack.angle == pytest.approx(0.0, abs=1.0)


def test_cation_pi_geometry(site):
    found = detect.cation_pi(site)
    assert len(found) == 1
    assert found[0].distance == pytest.approx(5.02, abs=0.05)


def test_halogen_bond_geometry(site):
    found = detect.halogen_bonds(site)
    assert len(found) == 1
    bond = found[0]
    assert bond.distance == pytest.approx(3.30, abs=1e-3)
    assert bond.angle == pytest.approx(180.0, abs=1.0)
    assert "CL1" in bond.label


def test_metal_coordination(site):
    found = detect.metal_coordination(site)
    assert len(found) == 2
    assert {round(f.distance, 2) for f in found} == {2.10, 2.23}


def test_hydrophobic_contacts_are_apolar_only(site):
    from blender_gala.core.selection import _elements

    elements = _elements(site.context)
    found = detect.hydrophobic_contacts(site)
    assert found
    for contact in found:
        assert elements[contact.atoms_a[0]] == "C"
        assert elements[contact.atoms_b[0]] == "C"


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def test_tighter_criteria_find_less(site):
    loose = detect.polar_contacts(site)
    tight = detect.polar_contacts(
        site, criteria=detect.InteractionCriteria(polar_max=2.9)
    )
    assert len(tight) < len(loose)
    assert len(tight) == 1


def test_hydrogen_bond_angle_cutoff_is_respected(site):
    strict = detect.hydrogen_bonds(
        site, criteria=detect.InteractionCriteria(hbond_angle_min=179.9)
    )
    assert len(strict) == 1
    impossible = detect.hydrogen_bonds(
        site, criteria=detect.InteractionCriteria(hbond_h_acceptor_max=0.5)
    )
    assert impossible == []


# ---------------------------------------------------------------------------
# find_interactions
# ---------------------------------------------------------------------------


def test_find_interactions_between_two_selections(site):
    found = detect.find_interactions(site, "resn LIG", "protein", kinds="all")
    kinds = {item.kind for item in found}
    assert "pi_stacking" in kinds
    assert "halogen" in kinds
    # Everything reported must genuinely bridge the two selections.
    ligand = {int(i) for i in site.indices("resn LIG")}
    for item in found:
        sides = (set(item.atoms_a), set(item.atoms_b))
        assert any(side & ligand for side in sides)
        assert any(not (side & ligand) for side in sides)


def test_find_interactions_sorted_by_kind_then_distance(site):
    found = detect.find_interactions(site, kinds="all")
    keys = [(item.kind, round(item.distance, 6)) for item in found]
    assert keys == sorted(keys)


def test_find_interactions_rejects_unknown_kinds(site):
    with pytest.raises(ValueError, match="unknown interaction kind"):
        detect.find_interactions(site, kinds=["not_a_kind"])


def test_hbond_falls_back_to_polar_without_hydrogens(site_array):
    """Asking for hydrogen bonds in a structure with no hydrogens should not
    silently return nothing."""
    from blender_gala.core.entity import AtomStructure

    stripped = site_array[site_array.element != "H"]
    structure = AtomStructure(array=stripped)

    found = detect.find_interactions(structure, kinds=["hbond"])
    assert found, "expected a fallback to heavy-atom polar contacts"
    assert {item.kind for item in found} == {"polar"}


def test_hbond_uses_hydrogens_when_present(site):
    found = detect.find_interactions(site, kinds=["hbond"])
    assert {item.kind for item in found} == {"hbond"}


def test_atom_contacts_is_a_plain_distance_search(site):
    contacts = detect.atom_contacts(site, "resn LIG", "protein", cutoff=4.0)
    assert contacts
    assert all(c.distance <= 4.0 for c in contacts)
    assert all(c.kind == "contact" for c in contacts)


def test_interaction_repr_is_readable(site):
    bond = detect.hydrogen_bonds(site)[0]
    text = str(bond)
    assert "hbond" in text
    assert "2.80 A" in text
    assert "180 deg" in text


def test_interaction_points_are_in_blender_units(site):
    bond = detect.hydrogen_bonds(site)[0]
    separation = np.linalg.norm(bond.point_a - bond.point_b)
    assert separation == pytest.approx(bond.distance * site.world_scale, rel=1e-6)
