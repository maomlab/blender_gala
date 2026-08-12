"""What happens when the molecule is edited after Gala has met it.

Gala's central assumption, stated in ``core/entity.py``, is that the Blender
object's vertex *i* is atom *i* of the biotite array. Every other test in the
suite loads a molecule and leaves it alone. A user does not: they tab into Edit
Mode, delete a few atoms, scale the object to fit a layout, duplicate it for a
comparison figure, or step a trajectory.

The distinction that matters here is not "does it work" but **silently wrong
versus refused**. A measurement that raises is a nuisance; a measurement that
returns 100 Å for a 1.41 Å bond, or a colour painted onto the wrong atoms, goes
into a figure and then into a paper. ``tests/ROBUSTNESS.md`` §1 puts refusal
above guessing for exactly this reason.

See also ``tests/ROBUSTNESS.md`` §8.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_bpy, requires_mn

pytestmark = [pytest.mark.bpy, pytest.mark.mn, requires_bpy, requires_mn]

#: The C1-C2 distance of the two atoms used throughout, in angstrom. It is a
#: property of the molecule and must survive anything done to the object.
BOND = 1.414213


def edit(obj):
    """Put ``obj`` into Edit Mode and return its BMesh."""
    import bmesh
    import bpy

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    mesh = bmesh.from_edit_mesh(obj.data)
    mesh.verts.ensure_lookup_table()
    return mesh


def leave_edit():
    import bpy

    bpy.ops.object.mode_set(mode="OBJECT")


def delete_vertices(obj, count: int) -> None:
    """Delete the first ``count`` vertices, as a user pruning atoms would."""
    import bmesh

    mesh = edit(obj)
    bmesh.ops.delete(mesh, geom=mesh.verts[:count], context="VERTS")
    leave_edit()


# ---------------------------------------------------------------------------
# The vertex-to-atom correspondence
# ---------------------------------------------------------------------------


def test_a_restored_vertex_count_is_not_taken_as_proof(site_molecule):
    """Deleting five vertices and adding five back leaves the count matching
    and the correspondence destroyed.

    The guard in ``local_positions`` compares counts, and a count match is not
    evidence: every atom now reads a different vertex. Measured before this was
    caught, a 1.41 A bond reported as 100 A, with nothing anywhere refusing.
    Molecular Nodes writes ``res_id`` per point and a new BMesh vertex gets
    zero, so there is a cheap way to tell.
    """
    import blender_gala as gala

    obj = site_molecule.object
    mesh = edit(obj)
    for vertex in mesh.verts[:5]:
        mesh.verts.remove(vertex)
    for _ in range(5):
        mesh.verts.new((10.0, 10.0, 10.0))
    leave_edit()

    assert len(obj.data.vertices) == gala.AtomStructure.from_any(obj).n_atoms

    try:
        measured = gala.distance(obj, "index 1", "index 2").value
    except gala.GalaError:
        return  # refusing is the other acceptable answer
    assert measured == pytest.approx(BOND, abs=1e-3)


def test_a_viewport_selection_of_something_is_not_reported_as_nothing(site_molecule):
    """With the counts out of step the mask cannot be read — but "none" is a
    valid-looking answer that is false, and the advice that follows it ("select
    some atoms in Edit Mode first") is advice the user has already taken."""
    import blender_gala as gala

    obj = site_molecule.object
    delete_vertices(obj, 10)

    mesh = edit(obj)
    for vertex in mesh.verts[:20]:
        vertex.select = True
    leave_edit()

    try:
        described = gala.describe_viewport_selection(obj)
    except gala.GalaError:
        return  # refusing is the other acceptable answer
    assert described != "none"


def test_colour_is_written_to_the_atoms_that_were_selected(site_molecule):
    """`write_colors` checks its array against the *mesh*, so an atom-indexed
    colour array is written straight onto vertex indices when the two happen to
    be the same length."""
    import blender_gala as gala

    obj = site_molecule.object
    mesh = edit(obj)
    for vertex in mesh.verts[:5]:
        mesh.verts.remove(vertex)
    for _ in range(5):
        mesh.verts.new((10.0, 10.0, 10.0))
    leave_edit()

    expected = np.flatnonzero(gala.select(site_molecule, "chain A and resi 1"))
    try:
        gala.color_by_selection(obj, {"chain A and resi 1": "#ff0000"})
    except gala.GalaError:
        return  # refusing is the other acceptable answer

    colours = gala.read_colors(obj)
    reddened = np.flatnonzero(colours[:, 0] > 0.5)
    assert set(reddened) <= set(expected)


def test_reading_colours_gives_one_row_per_atom(site_molecule):
    """Documented as ``(n_atoms, 4)``. After an edit it returned one row per
    *vertex*, which a caller then indexes with atom indices."""
    import blender_gala as gala

    obj = site_molecule.object
    n_atoms = gala.AtomStructure.from_any(obj).n_atoms
    delete_vertices(obj, 10)

    try:
        colours = gala.read_colors(obj)
    except gala.GalaError:
        return  # refusing is the other acceptable answer
    assert colours.shape[0] == n_atoms


def test_colouring_a_drifted_mesh_says_what_is_wrong(site_molecule):
    """`write_boolean` already sets the standard here — ``mask has 72 values
    but the mesh has 62 vertices``. The colour path escapes as raw numpy,
    naming neither the molecule nor the edit that caused it."""
    import blender_gala as gala

    obj = site_molecule.object
    delete_vertices(obj, 10)

    with pytest.raises(gala.GalaError, match="vert"):
        gala.color_by_bfactor(obj)


# ---------------------------------------------------------------------------
# The object's transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scale", "label"),
    [
        ((2.0, 2.0, 2.0), "uniform"),
        ((3.0, 1.0, 1.0), "non-uniform"),
        # A mirror leaves a distance alone; it is the dihedral that flips.
        ((-1.0, 1.0, 1.0), "mirrored"),
    ],
)
def test_a_measurement_is_in_angstrom_whatever_the_object_scale(
    site_molecule, scale, label
):
    """A bond length is a property of the molecule, not of how the object is
    displayed. `_angstrom` divides world Blender units by the world scale,
    which knows nothing about ``matrix_world`` — so scaling an object to fit a
    layout doubled the reported bond, and interaction detection, which measures
    on the array, went on disagreeing with it about the same molecule.
    """
    import bpy

    import blender_gala as gala

    obj = site_molecule.object
    pristine = gala.distance(obj, "index 1", "index 2").value
    assert pristine == pytest.approx(BOND, abs=1e-4)

    obj.scale = scale
    bpy.context.view_layer.update()

    assert gala.distance(obj, "index 1", "index 2").value == pytest.approx(
        BOND, abs=1e-3
    ), label


def test_a_mirrored_object_does_not_flip_the_chirality(site_molecule):
    """A dihedral read out of the display rather than out of the chemistry
    changes sign when the object is mirrored — and a torsion's sign is the
    whole point of reporting one."""
    import bpy

    import blender_gala as gala

    obj = site_molecule.object
    selections = ("index 1", "index 2", "index 3", "index 5")
    upright = gala.dihedral(obj, *selections).value

    obj.scale = (-1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

    assert gala.dihedral(obj, *selections).value == pytest.approx(upright, abs=1e-3)


# ---------------------------------------------------------------------------
# Several models
# ---------------------------------------------------------------------------


def test_a_frame_places_the_atoms_of_the_frame_it_names(clean_scene):
    """`frame=` is honoured by the chemistry and ignored by the geometry: the
    base mesh is always model 0, and the vertex count matches, so
    `local_positions` reads it and everything *drawn* lands on the wrong model.

    The same split `AtomStructure` had between `.coord` and `.context.coord`,
    reappearing between `.coord` and `.world_positions()`.
    """
    import os

    from blender_gala.core import mn
    from blender_gala.core.entity import AtomStructure
    from tests.conftest import DATA_DIR

    module = mn.require_mn()
    module.register()
    molecule = module.Molecule.load(os.path.join(DATA_DIR, "ensemble.pdb"))

    structure = AtomStructure.from_any(molecule.object, frame=2)

    assert np.allclose(
        structure.world_positions()[1],
        structure.coord[1] * structure.world_scale,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Moving the origin
# ---------------------------------------------------------------------------


def test_moving_the_origin_does_not_displace_what_is_drawn(site_molecule):
    """`set_origin_to_geometry` shifts the mesh one way and the object
    transform the other, leaving world space unchanged — but the biotite array
    is never shifted, so the moment a later edit makes `local_positions` fall
    back to ``coord * world_scale`` every world position is off by exactly the
    origin offset. Measured at 4.4 A, which is a figure with the labels in the
    wrong place rather than a build that failed.
    """
    import blender_gala as gala

    obj = site_molecule.object
    gala.set_origin_to_geometry(obj)
    delete_vertices(obj, 1)

    structure = gala.AtomStructure.from_any(obj)
    drawn = structure.world_positions()[10]
    actual = np.array(obj.matrix_world @ obj.data.vertices[9].co)

    assert np.allclose(drawn, actual, atol=1e-6)


# ---------------------------------------------------------------------------
# What holds, pinned so a fix cannot break it
# ---------------------------------------------------------------------------


def test_a_style_does_not_break_the_correspondence(site_molecule):
    """The guard's docstring blames a style modifier for breaking the 1:1
    mapping. It does not: the base mesh keeps its vertices and geometry nodes
    builds beside it. Pinned so the real trigger — an edit-mode geometry
    change — stays the one the guard is about."""
    import blender_gala as gala

    obj = site_molecule.object
    n_atoms = gala.AtomStructure.from_any(obj).n_atoms

    site_molecule.add_style("cartoon")
    site_molecule.add_style("ball_and_stick")

    assert len(obj.data.vertices) == n_atoms
    assert gala.distance(obj, "index 1", "index 2").value == pytest.approx(
        BOND, abs=1e-4
    )


def test_renaming_the_object_changes_nothing(site_molecule):
    """Resolution is by identity, not by name."""
    import blender_gala as gala

    obj = site_molecule.object
    obj.name = "renamed for a figure"
    obj.data.name = "renamed mesh"

    assert gala.distance(obj, "index 1", "index 2").value == pytest.approx(
        BOND, abs=1e-4
    )


def test_deleting_an_attribute_molecular_nodes_owns_changes_nothing(site_molecule):
    """The chemistry comes from the biotite array, so a mesh attribute the user
    removed by hand cannot make a selection wrong."""
    import blender_gala as gala

    obj = site_molecule.object
    for name in ("res_id", "b_factor"):
        attribute = obj.data.attributes.get(name)
        if attribute is not None:
            obj.data.attributes.remove(attribute)

    assert int(gala.select(obj, "resi 1").sum()) == 17


def test_a_separated_mesh_is_not_mistaken_for_the_molecule(site_molecule):
    """Separating part of a mesh makes a new object that Molecular Nodes does
    not track, and Gala has to say so rather than read the original's atoms."""
    import bmesh
    import bpy

    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.exceptions import StructureError

    obj = site_molecule.object
    mesh = edit(obj)
    for vertex in mesh.verts[:10]:
        vertex.select = True
    bpy.ops.mesh.separate(type="SELECTED")
    leave_edit()
    bmesh.update_edit_mesh(obj.data) if obj.mode == "EDIT" else None

    separated = [o for o in bpy.data.objects if o is not obj and o.type == "MESH"]
    assert separated, "separate produced no object"

    with pytest.raises(StructureError, match="Molecular Nodes"):
        AtomStructure.from_any(separated[0])
