"""What happens to Gala when structures come and go.

Every other Blender test in the suite loads exactly one molecule, once, and
leaves it alone until the test ends. Real use is not like that: a user imports
a structure, imports a second to compare it against, deletes the first, reloads
it, and presses a button in the sidebar at every point in between. Molecular
Nodes' session outlives the objects it tracks, so "the molecule is gone" is a
state Gala meets routinely and has to have an answer for.

The contract, the same one ``tests/ROBUSTNESS.md`` §1 sets out: a molecule that
has been deleted is *reported*, not tripped over. `LinkedObjectError` and
`ReferenceError` are what Blender and databpy raise when an object is gone;
neither is a `GalaError`, so neither reaches the user as anything but a
traceback in a console they are not watching.

See also ``tests/ROBUSTNESS.md`` §8.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_bpy, requires_mn

pytestmark = [pytest.mark.bpy, pytest.mark.mn, requires_bpy, requires_mn]


def remove(obj) -> None:
    """Delete a Blender object the way pressing X in the outliner does."""
    import bpy

    bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# A molecule that is gone, while another is not
# ---------------------------------------------------------------------------


def test_deleting_one_molecule_leaves_the_other_usable(site_molecule, plddt_molecule):
    """Molecular Nodes' session keeps an entity for the deleted object, and
    resolving *any* object walks that session. The dead entry must not decide
    what happens to a live one.

    Both orders are asserted because the session is a dict: deleting the
    later-loaded molecule leaves the earlier one resolvable while deleting the
    earlier one poisons the later, so the same user action has opposite
    outcomes depending on import order.
    """
    from blender_gala.core.entity import AtomStructure

    survivor = plddt_molecule.object
    expected = AtomStructure.from_any(survivor).n_atoms

    remove(site_molecule.object)

    assert AtomStructure.from_any(survivor).n_atoms == expected


def test_deleting_the_later_molecule_leaves_the_earlier_usable(
    site_molecule, plddt_molecule
):
    """The order that happens to work today. Asserted so that a fix for the
    other order cannot quietly break this one."""
    from blender_gala.core.entity import AtomStructure

    survivor = site_molecule.object
    expected = AtomStructure.from_any(survivor).n_atoms

    remove(plddt_molecule.object)

    assert AtomStructure.from_any(survivor).n_atoms == expected


def test_the_public_api_still_works_beside_a_deleted_molecule(
    site_molecule, plddt_molecule
):
    """The shape a user's script takes: two structures, one no longer wanted."""
    import blender_gala as gala

    survivor = plddt_molecule.object
    remove(site_molecule.object)

    assert int(gala.select(survivor, "protein").sum()) > 0
    assert gala.find_interactions(survivor, "all", "all", kinds=("polar",)) is not None


# ---------------------------------------------------------------------------
# A structure whose object has been deleted underneath it
# ---------------------------------------------------------------------------


@pytest.fixture
def stale(site_molecule):
    """An :class:`AtomStructure` whose Blender object no longer exists."""
    from blender_gala.core.entity import AtomStructure

    structure = AtomStructure.from_any(site_molecule.object)
    remove(site_molecule.object)
    return structure


def test_a_stale_structure_can_still_be_printed(stale):
    """`repr` is the sharp edge: every other error message here interpolates
    `self.name`, so a name that raises turns one failure into two."""
    assert "AtomStructure" in repr(stale)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.name,
        lambda s: s.world_positions(),
        lambda s: s.local_positions(),
        lambda s: s.bounding_sphere(),
        lambda s: s.alias_names(),
        lambda s: s.viewport_selection(),
        lambda s: s.store_alias("pocket", "resi 1"),
    ],
    ids=[
        "name",
        "world_positions",
        "local_positions",
        "bounding_sphere",
        "alias_names",
        "viewport_selection",
        "store_alias",
    ],
)
def test_a_stale_structure_reports_rather_than_tripping_over_itself(stale, call):
    """Anything that needs the Blender object has to say the object is gone.

    `ReferenceError: StructRNA of type Object has been removed` is Blender
    telling Python; it is not a `GalaError`, so the operator layer cannot turn
    it into a message and the user gets a traceback.
    """
    from blender_gala.core.exceptions import GalaError

    with pytest.raises(GalaError):
        call(stale)


def test_the_chemistry_of_a_stale_structure_survives(stale):
    """The atom array was read at load time and does not depend on the object,
    so the chemistry is still answerable — and answering is more useful than
    refusing."""
    assert stale.n_atoms == 72
    assert stale.coord.shape == (72, 3)
    assert stale.atom_label(0)


def test_a_stale_structure_does_not_depend_on_what_was_cached(twins):
    """Two scripts differing only in an earlier, unrelated call must not
    disagree about whether selecting works.

    The selection context caches the molecule's stored selections, and its
    freshness check cannot tell that the object behind them has gone — so a
    structure that happened to be used before the deletion kept answering
    while an untouched one raised. Both paths are run here and compared
    against each other, rather than each being allowed its own answer.
    """
    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.exceptions import GalaError

    def answer(molecule, warm: bool):
        structure = AtomStructure.from_any(molecule.object)
        if warm:
            # Any call that populates the context cache will do; storing an
            # alias is the one a user is most likely to have made.
            structure.store_alias("pocket", "resi 1")
        remove(molecule.object)
        try:
            return int(structure.select("protein").sum())
        except GalaError as exc:
            return type(exc).__name__

    cold, warm = twins

    assert answer(cold, warm=False) == answer(warm, warm=True)


# ---------------------------------------------------------------------------
# Two copies of one file
# ---------------------------------------------------------------------------


@pytest.fixture
def twins(clean_scene):
    """The same file loaded twice — two objects, two entities, one structure."""
    import os

    from blender_gala.core import mn
    from tests.conftest import DATA_DIR

    module = mn.require_mn()
    module.register()
    path = os.path.join(DATA_DIR, "site.pdb")
    return module.Molecule.load(path), module.Molecule.load(path)


def test_each_copy_resolves_to_its_own_entity(twins):
    """Blender names the second object `site.001`; the session holds both.
    Resolving by object identity has to pick the right one, or every later
    call reads the first copy's atoms at the second copy's position."""
    from blender_gala.core.entity import AtomStructure

    first, second = twins

    assert AtomStructure.from_any(first.object).molecule is first
    assert AtomStructure.from_any(second.object).molecule is second


def test_an_alias_does_not_leak_between_two_copies(twins):
    """Separate mesh datablocks, so a name stored on one is unknown to the
    other — including to the selection language."""
    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.exceptions import SelectionSyntaxError

    first, second = twins
    one = AtomStructure.from_any(first.object)
    other = AtomStructure.from_any(second.object)

    one.store_alias("pocket", "resi 1")

    assert one.alias_names() == ["pocket"]
    assert other.alias_names() == []
    with pytest.raises(SelectionSyntaxError):
        other.select("pocket")


# ---------------------------------------------------------------------------
# Deleting and reloading the same file
# ---------------------------------------------------------------------------


def test_a_reloaded_molecule_does_not_inherit_the_old_alias(clean_scene):
    """A stored selection is a mesh attribute, so it goes with the mesh. The
    reloaded structure must not answer to a name it never stored — a stale
    `pocket` quietly meaning different atoms is worse than not existing.
    """
    import os

    from blender_gala.core import mn
    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.exceptions import SelectionSyntaxError
    from tests.conftest import DATA_DIR

    module = mn.require_mn()
    module.register()
    path = os.path.join(DATA_DIR, "site.pdb")

    first = module.Molecule.load(path)
    AtomStructure.from_any(first.object).store_alias("pocket", "resi 1")
    remove(first.object)

    reloaded = AtomStructure.from_any(module.Molecule.load(path).object)

    assert reloaded.alias_names() == []
    with pytest.raises(SelectionSyntaxError, match="pocket"):
        reloaded.select("pocket around 4")


# ---------------------------------------------------------------------------
# Passing a Blender object where a structure is expected
# ---------------------------------------------------------------------------


def test_select_takes_a_blender_object_like_everything_else(site_molecule):
    """`distance`, `label` and `find_interactions` all accept the object,
    because they resolve it through `from_any`. `select` does not, and fails
    with `TypeError: object of type 'Object' has no len()` — an inconsistency
    in the public API rather than a documented refusal."""
    import blender_gala as gala

    from_object = gala.select(site_molecule.object, "protein")
    from_molecule = gala.select(site_molecule, "protein")

    assert np.array_equal(from_object, from_molecule)


# ---------------------------------------------------------------------------
# Drawn geometry outliving its molecule
# ---------------------------------------------------------------------------


def test_clearing_still_works_after_the_molecule_is_gone(site_molecule):
    """Measurements, labels and interaction lines are tagged by collection and
    kind rather than by the molecule they came from, so deleting the molecule
    orphans them in the scene — but the clearing contract has to hold, or the
    user is left with geometry nothing can remove."""
    import blender_gala as gala

    gala.distance(site_molecule, "resi 1 and name OG", "resi 2 and name OD1", draw=True)
    gala.label(site_molecule, "resi 1 and name CA")

    remove(site_molecule.object)

    assert gala.clear_measurements() > 0
    assert gala.clear_labels() > 0
