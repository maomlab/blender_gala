"""Tests for units, chemistry, the pure geometry helpers and the adapter."""

from __future__ import annotations

import numpy as np
import pytest

from blender_gala.core import chemistry, units
from blender_gala.core.entity import AtomStructure
from blender_gala.core.exceptions import (
    AmbiguousSelectionError,
    EmptySelectionError,
    StructureError,
)
from blender_gala.core.geometry import arc_points, dash_segments, dihedral_arc_points

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_default_world_scale():
    assert units.DEFAULT_WORLD_SCALE == 0.01
    assert units.world_scale_of(None) == 0.01


def test_round_trip_conversion():
    assert units.bu_to_angstrom(units.angstrom_to_bu(3.5)) == pytest.approx(3.5)


def test_conversion_preserves_scalars_and_arrays():
    assert isinstance(units.angstrom_to_bu(1.0), float)
    result = units.angstrom_to_bu([1.0, 2.0])
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [0.01, 0.02]


def test_world_scale_read_from_object():
    class FakeObject(dict):
        pass

    obj = FakeObject(world_scale=0.1)
    assert units.world_scale_of(obj) == 0.1


def test_world_scale_ignores_nonsense():
    class FakeObject(dict):
        pass

    assert units.world_scale_of(FakeObject(world_scale="banana")) == 0.01
    assert units.world_scale_of(FakeObject(world_scale=-1.0)) == 0.01


def test_zero_scale_rejected():
    with pytest.raises(ValueError, match="positive"):
        units.bu_to_angstrom(1.0, scale=0.0)


# ---------------------------------------------------------------------------
# Chemistry tables
# ---------------------------------------------------------------------------


def test_canonical_amino_acids_are_a_subset():
    assert chemistry.CANONICAL_AMINO_ACIDS <= chemistry.AMINO_ACIDS
    assert len(chemistry.CANONICAL_AMINO_ACIDS) == 20


def test_vdw_radius_falls_back_for_unknown_elements():
    assert chemistry.vdw_radius("C") == pytest.approx(1.70)
    assert chemistry.vdw_radius("c") == pytest.approx(1.70)
    assert chemistry.vdw_radius("Xx") == pytest.approx(1.70)


def test_normalise_element():
    result = chemistry.normalise_element(np.array([" c ", "Cl", "n"]))
    assert result.tolist() == ["C", "CL", "N"]


def test_aromatic_ring_tables_are_plausible():
    for residue, rings in chemistry.AROMATIC_RINGS.items():
        for ring in rings:
            assert 5 <= len(ring) <= 6, residue
            assert len(set(ring)) == len(ring), residue


# ---------------------------------------------------------------------------
# Dashes
# ---------------------------------------------------------------------------


def test_dash_segments_span_the_whole_line():
    segments = dash_segments([0, 0, 0], [10, 0, 0], dash_length=1.0, gap_length=0.5)
    assert len(segments) > 1
    assert segments[0][0].tolist() == [0.0, 0.0, 0.0]
    assert segments[-1][1] == pytest.approx([10.0, 0.0, 0.0])


def test_dash_segments_are_evenly_spaced():
    segments = dash_segments([0, 0, 0], [10, 0, 0], dash_length=1.0, gap_length=0.5)
    lengths = [np.linalg.norm(end - start) for start, end in segments]
    assert np.allclose(lengths, lengths[0])


def test_zero_gap_gives_a_solid_line():
    segments = dash_segments([0, 0, 0], [5, 0, 0], dash_length=1.0, gap_length=0.0)
    assert len(segments) == 1


def test_degenerate_segment_gives_nothing():
    assert dash_segments([1, 1, 1], [1, 1, 1]) == []


def test_short_line_gives_one_dash():
    segments = dash_segments([0, 0, 0], [0.1, 0, 0], dash_length=1.0, gap_length=1.0)
    assert len(segments) == 1


@pytest.mark.parametrize(
    ("dash", "gap"),
    [(0.0, 0.1), (-1.0, 0.1), (1.0, -0.1)],
)
def test_invalid_dash_parameters_raise(dash, gap):
    with pytest.raises(ValueError):
        dash_segments([0, 0, 0], [1, 0, 0], dash_length=dash, gap_length=gap)


# ---------------------------------------------------------------------------
# Arcs
# ---------------------------------------------------------------------------


def test_arc_starts_and_ends_on_the_rays():
    points = arc_points([0, 0, 0], [1, 0, 0], [0, 1, 0], radius=1.0, resolution=17)
    assert points.shape == (17, 3)
    assert points[0] == pytest.approx([1.0, 0.0, 0.0])
    assert points[-1] == pytest.approx([0.0, 1.0, 0.0])


def test_arc_points_lie_on_the_circle():
    points = arc_points([1, 2, 3], [2, 2, 3], [1, 3, 3], radius=0.5)
    radii = np.linalg.norm(points - np.array([1, 2, 3]), axis=1)
    assert np.allclose(radii, 0.5)


def test_collinear_rays_have_no_arc():
    assert arc_points([0, 0, 0], [1, 0, 0], [2, 0, 0]).shape == (0, 3)


def test_degenerate_arc_is_empty():
    assert arc_points([0, 0, 0], [0, 0, 0], [1, 0, 0]).shape == (0, 3)


def test_dihedral_arc_is_centred_on_the_central_bond():
    # A 90-degree torsion: A along +X, D along +Y, about a central bond on Z.
    points = dihedral_arc_points([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1])
    assert points.shape[0] >= 2

    midpoint = np.array([0.0, 0.0, 0.5])
    radii = np.linalg.norm(points - midpoint, axis=1)
    assert np.allclose(radii, radii[0])
    # The arc lies in the plane normal to the central bond.
    assert np.allclose(points[:, 2], 0.5)


def test_dihedral_arc_of_a_planar_torsion_is_empty():
    """A 0-degree torsion has no swept angle, so there is no arc to draw."""
    points = dihedral_arc_points([1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1])
    assert points.shape == (0, 3)


def test_dihedral_arc_of_degenerate_input_is_empty():
    zero = dihedral_arc_points([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    assert zero.shape == (0, 3)


# ---------------------------------------------------------------------------
# AtomStructure
# ---------------------------------------------------------------------------


def test_from_any_accepts_an_atom_array(site_array):
    structure = AtomStructure.from_any(site_array)
    assert structure.n_atoms == len(site_array)
    assert structure.object is None


def test_from_any_is_idempotent(site):
    assert AtomStructure.from_any(site) is site


def test_from_any_rejects_nonsense():
    with pytest.raises(StructureError, match="cannot interpret"):
        AtomStructure.from_any(object())


def test_positions_fall_back_to_scaled_coordinates(site):
    expected = site.coord * units.DEFAULT_WORLD_SCALE
    assert np.allclose(site.world_positions(), expected)


def test_bounding_sphere_contains_every_atom(site):
    centre, radius = site.bounding_sphere()
    distances = np.linalg.norm(site.world_positions() - centre, axis=1)
    assert distances.max() <= radius + 1e-9
    assert radius > 0


def test_one_index_requires_exactly_one_atom(site):
    index = site.one_index("resi 1 and name OG")
    assert site.atom_label(index, "{resn}{resi}/{name}") == "SER1/OG"


def test_ambiguous_selection_is_rejected(site):
    with pytest.raises(AmbiguousSelectionError, match="matched"):
        site.one_index("name CA")


def test_empty_selection_is_rejected(site):
    with pytest.raises(EmptySelectionError, match="no atoms"):
        site.one_index("resn XXX")


def test_reduce_policies_resolve_ambiguity(site):
    first = site.one_index("name CA", reduce="first")
    last = site.one_index("name CA", reduce="last")
    assert first < last


def test_reduce_closest_uses_the_reference(site):
    reference = site.world_point(site.one_index("resi 1 and name OG"))
    index = site.one_index("name CA", reduce="closest", reference=reference)
    assert site.atom_label(index, "{resi}") == "1"


def test_centroid_is_not_a_single_index(site):
    with pytest.raises(ValueError, match="one_point"):
        site.one_index("protein", reduce="centroid")


def test_one_point_centroid_averages(site):
    indices = site.indices("resn LIG")
    expected = site.world_positions()[indices].mean(axis=0)
    assert np.allclose(site.one_point("resn LIG", reduce="centroid"), expected)


def test_unknown_reduce_policy_raises(site):
    with pytest.raises(ValueError, match="unknown reduce policy"):
        site.one_index("name CA", reduce="magic")


def test_atom_fields_are_complete(site):
    fields = site.atom_fields(site.one_index("resi 1 and name OG"))
    assert fields["resn"] == "SER"
    assert fields["one"] == "S"
    assert fields["name"] == "OG"
    assert fields["chain"] == "A"
    assert fields["elem"] == "O"


def test_atom_label_uses_the_template(site):
    index = site.one_index("resi 2 and name OD1")
    assert site.atom_label(index, "{chain}/{resn}{resi}/{name}") == "A/ASP2/OD1"


def test_len_and_repr(site):
    assert len(site) == site.n_atoms
    assert "AtomStructure" in repr(site)
