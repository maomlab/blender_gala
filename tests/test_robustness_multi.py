"""More than one molecule in the scene at once.

Every other Blender test in the suite uses exactly one. A real figure rarely
does: a protein and its ligand imported separately, two conformations
superposed for comparison, a copy made with Shift+D to show a mutation beside
the wild type. The failures here are not crashes — they are figures that come
out wrong, because an operation on one molecule reached another, or did not
reach it when it should have.

See also ``tests/ROBUSTNESS.md`` §8.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_bpy, requires_mn

pytestmark = [pytest.mark.bpy, pytest.mark.mn, requires_bpy, requires_mn]


def closest_approach(first, second) -> float:
    """Smallest distance between the vertices of two objects, in Blender units.

    The quantity a figure is actually about: how near the ligand sits to the
    pocket. Anything that moves one molecule and not the other changes it.
    """
    import numpy as np

    def world(obj):
        local = np.array([vertex.co for vertex in obj.data.vertices])
        matrix = np.array(obj.matrix_world).reshape(4, 4)
        return (np.hstack([local, np.ones((len(local), 1))]) @ matrix.T)[:, :3]

    a, b = world(first), world(second)
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())


@pytest.fixture
def registered(clean_scene):
    """The add-on registered for the duration of one test."""
    import blender_gala

    blender_gala.register()
    yield clean_scene
    blender_gala.unregister()


@pytest.fixture
def duplicated(site_molecule):
    """One molecule and a Shift+D copy of it, which nothing tracks.

    The style matters: the ``Set Color`` node Gala mutes only exists once one
    has been applied, so without it the copy shares nothing worth sharing.
    """
    import bpy

    site_molecule.add_style("cartoon")
    obj = site_molecule.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.duplicate()
    copy = bpy.context.view_layer.objects.active
    assert copy is not obj
    return site_molecule, copy


# ---------------------------------------------------------------------------
# One molecule moving without the others
# ---------------------------------------------------------------------------


def test_setting_up_one_molecule_keeps_the_others_in_register(
    site_molecule, plddt_molecule
):
    """`publication_setup` moves its target to the world origin by default, and
    moves nothing else. Two structures that were in contact are then not.

    Measured on a protein and its separately imported ligand: the closest
    approach went from 2.28 A to 3.52 A, every contact geometry in the figure
    became false, and `warnings` was empty. `set_origin_to_geometry`'s docstring
    promises the geometry does not move in world space, which is true of the
    origin shift and untrue of the move that follows it.
    """
    import blender_gala as gala

    before = closest_approach(site_molecule.object, plddt_molecule.object)

    gala.publication_setup(site_molecule, preset="draft", use_gpu=False)

    after = closest_approach(site_molecule.object, plddt_molecule.object)
    assert after == pytest.approx(before, abs=1e-6)


def test_moving_one_origin_keeps_the_others_in_register(site_molecule, plddt_molecule):
    """The same defect at its source, with nothing else in the way."""
    import blender_gala as gala

    before = closest_approach(site_molecule.object, plddt_molecule.object)

    gala.set_origin_to_geometry(site_molecule.object, move_to_world_origin=True)

    after = closest_approach(site_molecule.object, plddt_molecule.object)
    assert after == pytest.approx(before, abs=1e-6)


# ---------------------------------------------------------------------------
# A copy that is not a molecule
# ---------------------------------------------------------------------------


def test_an_untracked_copy_does_not_retarget_an_operator(registered, duplicated):
    """Shift+D is how a user makes "a second copy for comparison". The copy is
    a mesh that looks like a molecule and that Molecular Nodes does not track.

    Resolving it raises, the operator layer swallows that, and the fallback
    "the only molecule in the session" then applies — so every button pressed
    with the copy selected acts on the *original*. Colours, aliases, labels and
    measurements all land on the object the user was not looking at.
    """
    import bpy

    from blender_gala.ops.operators import active_structure

    _, copy = duplicated
    bpy.context.view_layer.objects.active = copy

    assert active_structure(bpy.context) is None


def test_colouring_one_copy_does_not_recolour_the_other(duplicated):
    """A duplicate shares its node tree with the original, and muting the
    `Set Color` node is how Gala stops Molecular Nodes overwriting what it
    wrote. Muting it on one object therefore changes what the other renders —
    its colour source switches from the generator to its own stale attribute.
    """
    import blender_gala as gala

    original, copy = duplicated
    original = original.object

    def muted(obj):
        return [
            node.mute
            for modifier in obj.modifiers
            for node in getattr(modifier.node_group, "nodes", ())
            if getattr(node, "node_tree", None) is not None
            and node.node_tree.name == "Set Color"
        ]

    gala.color_by_bfactor(original)

    # The probe has to be able to see the node at all, or the assertion below
    # passes by finding nothing rather than by finding it unmuted.
    assert muted(original), "no Set Color node found on the original"
    assert not any(muted(copy))


# ---------------------------------------------------------------------------
# Writing a session for some of what is in the scene
# ---------------------------------------------------------------------------


def test_saving_one_molecule_does_not_adopt_another_ones_annotations(
    site_molecule, plddt_molecule, tmp_path
):
    """A label is written by finding the nearest atom among the molecules being
    saved, at any distance — so restricting the export to one molecule moves
    the other's label onto it, 100 A from where it was drawn, with nothing in
    `skipped`. Measurements ignore the molecule filter altogether.
    """
    import blender_gala as gala

    gala.label(plddt_molecule, "resi 1 and name CA")
    path = str(tmp_path / "one.pse")

    gala.save_session(path, molecules=[site_molecule])

    written = gala.read_session(path)
    assert len(written.molecules) == 1
    assert not any(text for text in written.molecules[0].label)


def test_a_label_on_the_second_copy_is_recorded_on_the_second_copy(
    clean_scene, tmp_path
):
    """Two conformations superposed is the canonical two-molecule figure. The
    nearest-atom search replaces only on a strict `<`, so an exact tie always
    resolves to whichever molecule was written first — and the exported session
    labels the wrong one, invisibly until someone moves them apart in PyMOL.
    """
    import os

    import blender_gala as gala
    from blender_gala.core import mn
    from tests.conftest import DATA_DIR

    module = mn.require_mn()
    module.register()
    path = os.path.join(DATA_DIR, "site.pdb")
    first = module.Molecule.load(path)
    second = module.Molecule.load(path)

    gala.label(second, "resi 1 and name CA")
    out = str(tmp_path / "pair.pse")
    gala.save_session(out, molecules=[first, second])

    written = gala.read_session(out)
    labelled = [
        molecule.name
        for molecule in written.molecules
        if any(text for text in molecule.label)
    ]
    assert labelled == [written.molecules[1].name]


# ---------------------------------------------------------------------------
# Setting up a scene that has no single subject
# ---------------------------------------------------------------------------


def test_setting_up_without_a_subject_says_what_it_skipped(
    site_molecule, plddt_molecule
):
    """With two molecules loaded and neither active, nothing resolves — so
    materials and the origin are skipped. The report records the origin as
    skipped and says nothing in `warnings`, which is the only field the
    operator surfaces, so the user is told nothing at all.
    """
    import blender_gala as gala

    report = gala.publication_setup(None, preset="draft", use_gpu=False)

    assert report.warnings


# ---------------------------------------------------------------------------
# What holds, pinned so a fix cannot break it
# ---------------------------------------------------------------------------


def test_aliases_do_not_cross_between_molecules(site_molecule, plddt_molecule):
    """Stored selections live on the mesh, so each molecule has its own
    namespace — and a name stored on one is unknown to the other, with an error
    that lists what *is* stored rather than guessing."""
    import blender_gala as gala
    from blender_gala.core.exceptions import SelectionSyntaxError

    gala.create_alias(site_molecule, "pocket", "resi 1")
    gala.create_alias(plddt_molecule, "pocket", "resi 2")

    assert int(gala.select(site_molecule, "%pocket").sum()) == 17
    assert int(gala.select(plddt_molecule, "%pocket").sum()) == 5

    gala.create_alias(site_molecule, "only_here", "resi 2")
    with pytest.raises(SelectionSyntaxError, match="only_here"):
        gala.select(plddt_molecule, "only_here")


def test_colouring_one_molecule_leaves_the_other_alone(site_molecule, plddt_molecule):
    """Two separately imported files get separate node trees, so the mute that
    makes a colour visible on one must not reach the other."""
    import blender_gala as gala

    before = gala.read_colors(site_molecule.object).copy()

    gala.color_by_plddt(plddt_molecule)

    assert np.allclose(gala.read_colors(site_molecule.object), before)


def test_a_session_keeps_several_molecules_apart(
    site_molecule, plddt_molecule, tmp_path
):
    """Names and atom counts have to survive the round trip separately, or a
    two-molecule figure comes back as one molecule with the other's atoms."""
    import blender_gala as gala

    path = str(tmp_path / "both.pse")
    gala.save_session(path)

    written = gala.read_session(path)
    counts = sorted(molecule.n_atoms for molecule in written.molecules)

    assert counts == [40, 72]
