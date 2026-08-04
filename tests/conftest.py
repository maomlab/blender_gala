"""Shared fixtures.

The science layer works on a bare biotite ``AtomArray``, so most tests need
neither Blender nor Molecular Nodes. Tests that do are marked ``bpy`` or ``mn``
and skip cleanly elsewhere.
"""

from __future__ import annotations

import contextlib
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

try:
    import bpy  # noqa: F401

    HAS_BPY = True
except ImportError:
    HAS_BPY = False

try:
    import biotite.structure.io.pdb  # noqa: F401

    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False


def _has_mn() -> bool:
    if not HAS_BPY:
        return False
    from blender_gala.core import mn

    return mn.available()


HAS_MN = _has_mn()

requires_bpy = pytest.mark.skipif(not HAS_BPY, reason="needs a Blender interpreter")
requires_biotite = pytest.mark.skipif(not HAS_BIOTITE, reason="needs biotite")
requires_mn = pytest.mark.skipif(not HAS_MN, reason="needs Molecular Nodes")


def pytest_configure(config):
    config.addinivalue_line("markers", "bpy: needs a Blender interpreter")
    config.addinivalue_line("markers", "mn: needs Molecular Nodes")


# ---------------------------------------------------------------------------
# Structure fixtures
# ---------------------------------------------------------------------------


def load_array(name: str):
    """Load a fixture PDB into a biotite ``AtomArray``, no Blender involved.

    ``b_factor`` and friends are optional in biotite and are dropped unless
    asked for; the pLDDT and B-factor colouring tests need them.
    """
    import biotite.structure.io.pdb as pdb

    path = os.path.join(DATA_DIR, name)
    return pdb.PDBFile.read(path).get_structure(
        model=1, extra_fields=["b_factor", "occupancy", "charge", "atom_id"]
    )


@pytest.fixture(scope="session")
def site_array():
    """The synthetic binding site, as a bare ``AtomArray``."""
    if not HAS_BIOTITE:
        pytest.skip("needs biotite")
    return load_array("site.pdb")


@pytest.fixture(scope="session")
def plddt_array():
    """The AlphaFold-style model with pLDDT in the B-factor column."""
    if not HAS_BIOTITE:
        pytest.skip("needs biotite")
    return load_array("plddt.pdb")


@pytest.fixture(scope="session")
def geometry_array():
    """Four atoms with an exactly known distance, angle and dihedral."""
    if not HAS_BIOTITE:
        pytest.skip("needs biotite")
    return load_array("geometry.pdb")


@pytest.fixture
def site(site_array):
    """The binding site as an :class:`AtomStructure`, with no Blender object."""
    from blender_gala.core.entity import AtomStructure

    return AtomStructure(array=site_array)


@pytest.fixture
def geometry_structure(geometry_array):
    from blender_gala.core.entity import AtomStructure

    return AtomStructure(array=geometry_array)


@pytest.fixture
def plddt_structure(plddt_array):
    from blender_gala.core.entity import AtomStructure

    return AtomStructure(array=plddt_array)


# ---------------------------------------------------------------------------
# Blender fixtures
# ---------------------------------------------------------------------------


def purge_scene() -> None:
    """Empty the current scene without touching Blender's preferences.

    Deliberately *not* ``bpy.ops.wm.read_factory_settings``: that disables
    every extension, and Blender then syncs the extension wheel directory to
    match, uninstalling Molecular Nodes' Python dependencies for the rest of
    the session. Every later test in the run would then fail on a missing
    biotite. Purging the data-blocks by hand is both safer and faster.
    """
    import bpy

    scene = bpy.context.scene

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for library in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
        bpy.data.node_groups,
        bpy.data.worlds,
        bpy.data.images,
    ):
        for block in list(library):
            if block.users == 0:
                library.remove(block)

    scene.camera = None
    scene.world = None
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None
    scene.use_nodes = False

    view_layer = scene.view_layers[0]
    for attr in (
        "use_pass_z",
        "use_pass_normal",
        "use_pass_mist",
        "use_pass_cryptomatte_object",
        "use_pass_cryptomatte_material",
        "use_pass_cryptomatte_asset",
    ):
        if hasattr(view_layer, attr):
            setattr(view_layer, attr, False)

    # Molecular Nodes keeps a session dictionary of entities that outlives the
    # objects they wrap; stale entries would leak between tests.
    from blender_gala.core import mn as mn_bridge

    module = mn_bridge.get_mn()
    if module is not None:
        with contextlib.suppress(Exception):
            module.session.get_session().entities.clear()


@pytest.fixture
def clean_scene():
    """An empty scene, before and after each test."""
    if not HAS_BPY:
        pytest.skip("needs a Blender interpreter")
    import bpy

    purge_scene()
    yield bpy.context.scene
    purge_scene()


@pytest.fixture
def site_molecule(clean_scene):
    """The binding site loaded through Molecular Nodes, with a real object."""
    if not HAS_MN:
        pytest.skip("needs Molecular Nodes")
    from blender_gala.core import mn

    module = mn.require_mn()
    module.register()
    return module.Molecule.load(os.path.join(DATA_DIR, "site.pdb"))


@pytest.fixture
def plddt_molecule(clean_scene):
    if not HAS_MN:
        pytest.skip("needs Molecular Nodes")
    from blender_gala.core import mn

    module = mn.require_mn()
    module.register()
    return module.Molecule.load(os.path.join(DATA_DIR, "plddt.pdb"))
