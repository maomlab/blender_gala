"""Tests for distance, angle and dihedral measurement.

``geometry.pdb`` holds four atoms placed so the distance is exactly 1.500 A,
the angle exactly 90 degrees and the torsion exactly -90 degrees.
"""

from __future__ import annotations

import pytest

from blender_gala.core.exceptions import AmbiguousSelectionError, EmptySelectionError
from blender_gala.measure import measurements


def test_distance_is_exact(geometry_structure):
    result = measurements.distance(geometry_structure, "name C1", "name C2")
    assert result.value == pytest.approx(1.500, abs=1e-4)
    assert result.unit == "A"
    assert result.kind == "distance"


def test_angle_is_exact(geometry_structure):
    result = measurements.angle(geometry_structure, "name C1", "name C2", "name C3")
    assert result.value == pytest.approx(90.0, abs=1e-4)
    assert result.unit == "deg"


def test_dihedral_is_exact_and_signed(geometry_structure):
    result = measurements.dihedral(
        geometry_structure, "name C1", "name C2", "name C3", "name C4"
    )
    assert result.value == pytest.approx(-90.0, abs=1e-4)
    assert result.unit == "deg"


def test_dihedral_sign_flips_when_reversed(geometry_structure):
    forward = measurements.dihedral(
        geometry_structure, "name C1", "name C2", "name C3", "name C4"
    )
    backward = measurements.dihedral(
        geometry_structure, "name C4", "name C3", "name C2", "name C1"
    )
    assert forward.value == pytest.approx(backward.value, abs=1e-4)


def test_measure_dispatches_on_argument_count(geometry_structure):
    assert (
        measurements.measure(geometry_structure, "name C1", "name C2").kind
        == "distance"
    )
    assert (
        measurements.measure(geometry_structure, "name C1", "name C2", "name C3").kind
        == "angle"
    )
    assert (
        measurements.measure(
            geometry_structure, "name C1", "name C2", "name C3", "name C4"
        ).kind
        == "dihedral"
    )


def test_measure_rejects_wrong_argument_counts(geometry_structure):
    with pytest.raises(ValueError, match="2 \\(distance\\)"):
        measurements.measure(geometry_structure, "name C1")
    with pytest.raises(ValueError):
        measurements.measure(
            geometry_structure, "name C1", "name C2", "name C3", "name C4", "name C1"
        )


def test_ambiguous_selection_is_rejected(site):
    with pytest.raises(AmbiguousSelectionError):
        measurements.distance(site, "name CA", "name CB")


def test_empty_selection_is_rejected(site):
    with pytest.raises(EmptySelectionError):
        measurements.distance(site, "resn XXX", "name CA", reduce="first")


def test_reduce_policy_resolves_ambiguity(site):
    result = measurements.distance(site, "name CA", "name CB", reduce="first")
    assert result.value > 0


def test_per_selection_reduce_policies(site):
    result = measurements.distance(
        site,
        "resn LIG and name C1+C2+C3+C4+C5+C6",
        "resi 1 and name OG",
        reduce=["centroid", "single"],
    )
    assert result.value > 0
    assert result.atoms[0] == -1  # a centroid has no single atom index
    assert "centroid of 6 atoms" in result.labels[0]


def test_wrong_number_of_reduce_policies_raises(site):
    with pytest.raises(ValueError, match="reduce policies"):
        measurements.distance(site, "name CA", "name CB", reduce=["first"])


def test_measurement_on_the_real_binding_site(site):
    """The SER OG to ASP OD1 hydrogen bond was planted at exactly 2.80 A."""
    result = measurements.distance(site, "resi 1 and name OG", "resi 2 and name OD1")
    assert result.value == pytest.approx(2.80, abs=1e-3)
    assert result.labels == ("A/SER1/OG", "A/ASP2/OD1")


def test_measurement_text_and_float(geometry_structure):
    result = measurements.distance(geometry_structure, "name C1", "name C2")
    assert result.text == "1.50 A"
    assert float(result) == pytest.approx(1.5, abs=1e-4)
    assert "distance" in str(result)


def test_angle_of_coincident_atoms_raises(site):
    with pytest.raises(ValueError, match="undefined"):
        measurements.angle(
            site, "resi 1 and name OG", "resi 1 and name OG", "resi 2 and name OD1"
        )


def test_dihedral_of_degenerate_atoms_raises(geometry_structure):
    with pytest.raises(ValueError, match="undefined"):
        measurements.dihedral(
            geometry_structure, "name C1", "name C2", "name C3", "name C3"
        )


def test_points_are_recorded_in_blender_units(geometry_structure):
    import numpy as np

    result = measurements.distance(geometry_structure, "name C1", "name C2")
    separation = np.linalg.norm(result.points[0] - result.points[1])
    assert separation == pytest.approx(
        result.value * geometry_structure.world_scale, rel=1e-6
    )
