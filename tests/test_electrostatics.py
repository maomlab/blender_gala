"""Tests for reading APBS grids and painting them onto a structure.

The grid half runs anywhere: an OpenDX file is a text header and a list of
numbers, and interpolating it is arithmetic. Only the parts that touch a
Blender scene are marked.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_mn

# A three-by-three-by-three grid whose value is its x index, so anything read
# out of it can be checked by hand.
DX_TEXT = """# Data from a test
# POTENTIAL (kT/e)
object 1 class gridpositions counts 3 3 3
origin 1.000000e+00 2.000000e+00 3.000000e+00
delta 2.000000e+00 0.000000e+00 0.000000e+00
delta 0.000000e+00 2.000000e+00 0.000000e+00
delta 0.000000e+00 0.000000e+00 2.000000e+00
object 2 class gridconnections counts 3 3 3
object 3 class array type double rank 0 items 27 data follows
{values}
attribute "dep" string "positions"
object "regular positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
"""


def write_dx(path, values=None, text=DX_TEXT):
    """Write a grid file and return its path."""
    if values is None:
        values = np.repeat(np.arange(3.0), 9)
    body = "\n".join(
        " ".join(f"{v:.6e}" for v in values[i : i + 3])
        for i in range(0, len(values), 3)
    )
    path.write_text(text.format(values=body))
    return str(path)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_read_dx_reads_the_header_and_the_values(tmp_path):
    from blender_gala.electrostatics import read_dx

    grid = read_dx(write_dx(tmp_path / "pot.dx"))

    assert grid.shape == (3, 3, 3)
    assert np.allclose(grid.origin, [1.0, 2.0, 3.0])
    assert np.allclose(grid.spacing, [2.0, 2.0, 2.0])
    assert grid.unit == "kT/e"
    # z varies fastest, so the first nine values are the x = 0 plane.
    assert np.allclose(grid.values[0], 0.0)
    assert np.allclose(grid.values[2], 2.0)


def test_read_dx_reads_a_gzipped_grid(tmp_path):
    import gzip

    from blender_gala.electrostatics import read_dx

    plain = write_dx(tmp_path / "pot.dx")
    packed = tmp_path / "pot.dx.gz"
    with open(plain, "rb") as source, gzip.open(packed, "wb") as target:
        target.write(source.read())

    assert read_dx(str(packed)).shape == (3, 3, 3)


def test_read_dx_rejects_a_truncated_file(tmp_path):
    from blender_gala.electrostatics import read_dx

    path = write_dx(tmp_path / "short.dx", values=np.arange(9.0))
    with pytest.raises(ValueError, match="promised 27"):
        read_dx(path)


def test_read_dx_rejects_a_skewed_grid(tmp_path):
    """A non-axis-aligned lattice would need a different interpolation."""
    from blender_gala.electrostatics import read_dx

    skewed = DX_TEXT.replace(
        "delta 0.000000e+00 2.000000e+00 0.000000e+00",
        "delta 5.000000e-01 2.000000e+00 0.000000e+00",
    )
    path = write_dx(tmp_path / "skew.dx", text=skewed)
    with pytest.raises(ValueError, match="not aligned"):
        read_dx(path)


def test_read_dx_rejects_something_else_entirely(tmp_path):
    from blender_gala.electrostatics import read_dx

    path = tmp_path / "notes.txt"
    path.write_text("this is not a grid\n")
    with pytest.raises(ValueError, match="not an OpenDX grid"):
        read_dx(str(path))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_returns_the_node_value_at_a_node(tmp_path):
    from blender_gala.electrostatics import read_dx

    grid = read_dx(write_dx(tmp_path / "pot.dx"))
    node = grid.origin + np.array([2.0, 0.0, 4.0])  # index (1, 0, 2)

    assert grid.sample(node[None, :])[0] == pytest.approx(1.0)


def test_sample_interpolates_between_nodes(tmp_path):
    from blender_gala.electrostatics import read_dx

    grid = read_dx(write_dx(tmp_path / "pot.dx"))
    between = grid.origin + np.array([1.0, 1.0, 1.0])  # halfway along x

    assert grid.sample(between[None, :])[0] == pytest.approx(0.5)


def test_sample_outside_the_box_clamps_or_says_nothing(tmp_path):
    from blender_gala.electrostatics import read_dx

    grid = read_dx(write_dx(tmp_path / "pot.dx"))
    outside = grid.origin - 100.0

    assert grid.sample(outside[None, :])[0] == pytest.approx(0.0)
    assert np.isnan(grid.sample(outside[None, :], outside="nan")[0])
    with pytest.raises(ValueError, match="clamp"):
        grid.sample(outside[None, :], outside="sideways")


def test_sample_checks_the_shape_of_its_input(tmp_path):
    from blender_gala.electrostatics import read_dx

    grid = read_dx(write_dx(tmp_path / "pot.dx"))
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        grid.sample(np.zeros((4, 2)))


def test_bounds_and_summary_describe_the_box(tmp_path):
    from blender_gala.electrostatics import read_dx

    grid = read_dx(write_dx(tmp_path / "pot.dx"))
    low, high = grid.bounds

    assert np.allclose(low, [1.0, 2.0, 3.0])
    assert np.allclose(high, [5.0, 6.0, 7.0])
    assert "3x3x3" in grid.summary()


# ---------------------------------------------------------------------------
# Running APBS
# ---------------------------------------------------------------------------


def test_find_executable_says_how_to_get_it(monkeypatch):
    from blender_gala.electrostatics import find_executable
    from blender_gala.electrostatics.apbs import ApbsUnavailable

    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("GALA_APBS", raising=False)

    with pytest.raises(ApbsUnavailable) as caught:
        find_executable("apbs")
    assert "pip install apbs-binary" in str(caught.value)
    assert "GALA_APBS" in str(caught.value)


def test_the_input_file_gets_gala_s_settings():
    """PDB2PQR sizes the grid; everything else is Gala's to set."""
    from blender_gala.electrostatics.apbs import _tune_input

    generated = "\n".join(
        [
            "elec",
            "    mg-auto",
            "    mol 1",
            "    lpbe",
            "    pdie 2.0000",
            "    sdie 78.5400",
            "    temp 298.15",
            "end",
        ]
    )
    tuned = _tune_input(
        generated,
        {
            "pdie": 4.0,
            "sdie": 80.0,
            "temperature": 310.0,
            "solver": "npbe",
            "ionic_strength": 0.15,
        },
    )

    assert "pdie 4.0000" in tuned
    assert "sdie 80.0000" in tuned
    assert "temp 310.00" in tuned
    assert "npbe" in tuned and "lpbe" not in tuned
    assert tuned.count("ion charge") == 2
    assert "conc 0.150" in tuned
    # The grid PDB2PQR worked out is left exactly as it was.
    assert "mg-auto" in tuned


def test_no_salt_means_no_ion_lines():
    from blender_gala.electrostatics.apbs import _tune_input

    tuned = _tune_input(
        "elec\n    mol 1\n    lpbe\nend",
        {
            "pdie": 2.0,
            "sdie": 78.54,
            "temperature": 298.15,
            "solver": "lpbe",
            "ionic_strength": 0.0,
        },
    )
    assert "ion charge" not in tuned


# ---------------------------------------------------------------------------
# Painting it onto a structure
# ---------------------------------------------------------------------------


def ramp_grid(structure, low=-10.0, high=10.0):
    """A grid over ``structure`` whose potential rises along x."""
    from blender_gala.electrostatics import PotentialGrid

    coordinates = np.asarray(structure.array.coord, dtype=float)
    origin = coordinates.min(axis=0) - 15.0
    span = coordinates.max(axis=0) + 15.0 - origin
    counts = np.array([32, 32, 32])
    spacing = span / (counts - 1)

    ramp = np.linspace(low, high, counts[0])
    values = np.repeat(ramp[:, None, None], counts[1], axis=1)
    values = np.repeat(values, counts[2], axis=2)
    return PotentialGrid(values=values, origin=origin, spacing=spacing)


def test_potential_at_atoms_follows_the_field(site):
    """A field that rises along x has to come back rising along x."""
    from blender_gala.electrostatics import potential_at_atoms

    grid = ramp_grid(site)
    values = potential_at_atoms(site, grid)
    x = np.asarray(site.array.coord)[:, 0]

    seen = np.isfinite(values)
    assert seen.any(), "no atom of a 30-atom site is solvent accessible"

    # Spearman rather than Pearson: the sample point sits a radius and a probe
    # outside the atom, so the relationship is monotone rather than exact.
    def ranks(values):
        return np.argsort(np.argsort(values))

    assert np.corrcoef(ranks(x[seen]), ranks(values[seen]))[0, 1] > 0.9


def test_buried_atoms_have_no_surface_value(site):
    """An atom with no accessible surface gets nan, not the interior field."""
    from blender_gala.electrostatics import potential_at_atoms

    values = potential_at_atoms(site, ramp_grid(site), probe=6.0)
    # A six-ångström probe cannot reach much of a packed site.
    assert np.isnan(values).any()


def test_sampling_ignores_where_blender_moved_the_object(site):
    """The map is in the deposited frame; the object's transform is not."""
    from blender_gala.electrostatics import potential_at_atoms

    grid = ramp_grid(site)
    before = potential_at_atoms(site, grid)

    moved = site.array.copy()
    structure = type(site)(array=moved)
    after = potential_at_atoms(structure, grid)

    assert np.allclose(before, after, equal_nan=True)


@requires_mn
def test_electrostatic_surface_colours_and_makes_it_translucent(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.electrostatics import electrostatic_surface

    structure = AtomStructure.from_any(site_molecule)
    surface = electrostatic_surface(
        site_molecule, grid=ramp_grid(structure), ramp=5.0, alpha=0.5
    )

    assert surface.styles >= 1, "the surface style got no material"
    assert surface.material.name == "GALA Electrostatic Surface"
    assert surface.colors.n_colored > 0

    principled = next(
        node
        for node in surface.material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    assert principled.inputs["Alpha"].default_value == pytest.approx(0.5)

    # Low x is negative potential, which has to come out red rather than blue.
    x = np.asarray(structure.array.coord)[:, 0]
    lowest = int(np.argmin(np.where(np.isfinite(surface.potential), x, np.inf)))
    red, _, blue = surface.colors.colors[lowest][:3]
    assert red > blue, "negative potential should be red"
