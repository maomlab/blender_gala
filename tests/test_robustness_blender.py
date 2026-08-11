"""Hostile-input tests for the parts of Gala that need a running Blender.

The companion to ``tests/test_robustness.py``, which covers everything that is
just data. Here the input is a scene: a structure with no atoms in it, a
sequence of calls in an order nobody anticipated, an argument that fails
validation halfway through a rebuild. See ``tests/ROBUSTNESS.md``.

Two contracts recur, and neither is about the happy path:

``a call that fails changes nothing``
    validation that runs *after* the old state has been torn down leaves a
    scene that is neither the old one nor the new one, and the exception says
    nothing about that;
``a call that succeeds twice is the same as once``
    every entry point here is something a user runs again after adjusting a
    setting, so duplicated lights, stacked node groups and orphaned cameras are
    the failure mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_biotite, requires_bpy, requires_mn

pytestmark = [pytest.mark.bpy, requires_bpy, requires_biotite]


def build(n: int, coord=None):
    """A bare structure of ``n`` carbons, with no Blender object behind it."""
    import biotite.structure as struc

    from blender_gala.core.entity import AtomStructure

    array = struc.AtomArray(n)
    if coord is None:
        coord = [[float(i) * 3.0, 0.0, 0.0] for i in range(n)]
    array.coord = np.asarray(coord, dtype=np.float32).reshape(n, 3)
    array.chain_id = np.array(["A"] * n)
    array.res_id = np.arange(1, n + 1)
    array.res_name = np.array(["ALA"] * n)
    array.atom_name = np.array(["CA"] * n)
    array.element = np.array(["C"] * n)
    return AtomStructure(array=array)


def node_tree(scene):
    """The compositing tree, under whichever name this Blender keeps it."""
    return getattr(scene, "compositing_node_group", None) or scene.node_tree


# ---------------------------------------------------------------------------
# Structures with nothing, or almost nothing, in them
# ---------------------------------------------------------------------------


def test_a_structure_with_no_atoms_still_produces_a_scene(clean_scene):
    """`bounding_sphere` cannot describe an empty structure, so the scene layer
    has to notice before it asks."""
    from blender_gala.scene import camera, lighting

    empty = build(0)

    assert camera.frame_target(empty, scene=clean_scene) is not None
    assert lighting.three_point_lighting(empty, scene=clean_scene) is not None


def test_framing_a_single_atom_leaves_it_between_the_clip_planes(clean_scene):
    """A one-atom subject has a radius of 1e-4 BU, which is what every distance
    below is derived from. Clipping it out gives an empty render and nothing
    to explain it."""
    from blender_gala.scene import camera

    one = build(1)
    obj = camera.frame_target(one, scene=clean_scene)

    centre, _ = one.bounding_sphere()
    distance = float(np.linalg.norm(np.array(obj.matrix_world.translation) - centre))

    assert distance > 0.0
    assert obj.data.clip_start < distance < obj.data.clip_end


def test_every_light_is_lit_for_a_single_atom(clean_scene):
    from blender_gala.scene import lighting

    lighting.three_point_lighting(build(1), scene=clean_scene)
    energies = [obj.data.energy for obj in clean_scene.objects if obj.type == "LIGHT"]

    assert energies
    assert all(np.isfinite(energy) and energy > 0.0 for energy in energies)


# ---------------------------------------------------------------------------
# Doing it twice
# ---------------------------------------------------------------------------


def test_lighting_a_scene_again_replaces_the_rig_rather_than_adding_one(clean_scene):
    from blender_gala.scene import lighting

    structure = build(20)
    for _ in range(3):
        lighting.three_point_lighting(structure, scene=clean_scene)

    lights = [obj.name for obj in clean_scene.objects if obj.type == "LIGHT"]

    assert sorted(lights) == ["GALA Fill", "GALA Key", "GALA Rim"]


def test_framing_again_reuses_the_camera(clean_scene):
    from blender_gala.scene import camera

    structure = build(20)
    for _ in range(3):
        camera.frame_target(structure, scene=clean_scene)

    cameras = [obj.name for obj in clean_scene.objects if obj.type == "CAMERA"]

    assert len(cameras) == 1


def test_setting_up_the_compositor_again_reuses_the_tree(clean_scene):
    """Node groups are the datablock Blender is happiest to accumulate."""
    import bpy

    from blender_gala.scene import compositing

    first = compositing.setup_compositor(scene=clean_scene)
    groups = len(bpy.data.node_groups)

    for _ in range(3):
        assert compositing.setup_compositor(scene=clean_scene) is first

    assert len(bpy.data.node_groups) == groups


# ---------------------------------------------------------------------------
# Arguments that are refused, cleanly
# ---------------------------------------------------------------------------


def test_an_unknown_viewpoint_lists_the_ones_that_exist(clean_scene):
    from blender_gala.scene import camera

    with pytest.raises(ValueError, match="unknown viewpoint"):
        camera.frame_target(build(4), viewpoint="sideways", scene=clean_scene)


# ---------------------------------------------------------------------------
# Confirmed gaps
# ---------------------------------------------------------------------------


def test_orbiting_keeps_the_framing_it_was_given(clean_scene):
    """`orbit` documents that the framing `frame_target` computed is preserved
    for every frame of the turntable."""
    from blender_gala.scene import camera

    structure = build(20)  # centred well away from the world origin
    obj = camera.frame_target(structure, scene=clean_scene)
    before = np.array(obj.matrix_world.translation).copy()

    camera.orbit(60, target=structure, scene=clean_scene)
    clean_scene.view_layers[0].update()

    assert np.allclose(np.array(obj.matrix_world.translation), before)


@pytest.mark.parametrize("frames", [0, -10])
def test_an_orbit_of_no_frames_is_refused(clean_scene, frames):
    from blender_gala.scene import camera

    structure = build(20)
    camera.frame_target(structure, scene=clean_scene)

    with pytest.raises(ValueError):
        camera.orbit(frames, target=structure, scene=clean_scene)


def test_a_rejected_depth_cue_leaves_the_compositor_alone(clean_scene):
    from blender_gala.scene import compositing

    compositing.setup_compositor(scene=clean_scene)
    before = sorted(node.name for node in node_tree(clean_scene).nodes)

    with pytest.raises(ValueError, match="far > near"):
        compositing.depth_cue(100.0, 10.0, scene=clean_scene)

    tree = node_tree(clean_scene)
    assert sorted(node.name for node in tree.nodes) == before
    output = next(node for node in tree.nodes if node.name == "Group Output")
    assert any(socket.is_linked for socket in output.inputs)


def test_a_rejected_lighting_spec_leaves_the_rig_alone(clean_scene):
    from blender_gala.scene import lighting

    structure = build(20)
    lighting.three_point_lighting(structure, scene=clean_scene)
    before = sorted(obj.name for obj in clean_scene.objects if obj.type == "LIGHT")

    specs = [
        lighting.LightSpec("Key", 0.0, 0.0, 1.0, 1.0),
        lighting.LightSpec("Bad", 0.0, 0.0, 1.0, 1.0, colour=(1.0, 0.0)),
    ]
    with pytest.raises(ValueError):
        lighting.three_point_lighting(structure, specs=specs, scene=clean_scene)

    after = sorted(obj.name for obj in clean_scene.objects if obj.type == "LIGHT")
    assert after == before


def test_the_pass_output_survives_a_later_render_setup(clean_scene, tmp_path):
    from blender_gala.scene import compositing, render

    compositing.set_exr_output(str(tmp_path / "passes.exr"), scene=clean_scene)
    render.setup_render("draft", use_gpu=False, transparent=True, scene=clean_scene)

    assert clean_scene.render.image_settings.file_format == "OPEN_EXR_MULTILAYER"


def test_a_file_that_is_not_an_image_is_not_an_hdri(clean_scene, tmp_path):
    from blender_gala.scene import lighting

    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("not an image\n")

    with pytest.raises((ValueError, OSError)):
        lighting.hdri_lighting(str(not_an_image), scene=clean_scene)


@pytest.mark.parametrize("kind", ["empty", "camera", "light"])
def test_setting_the_origin_of_something_that_is_not_a_molecule_is_reported(
    clean_scene, kind
):
    import bpy

    from blender_gala.core.exceptions import GalaError
    from blender_gala.scene import origin

    data = {
        "empty": None,
        "camera": bpy.data.cameras.new("c"),
        "light": bpy.data.lights.new("l", type="AREA"),
    }[kind]
    obj = bpy.data.objects.new(kind, data)
    clean_scene.collection.objects.link(obj)

    with pytest.raises(GalaError):
        origin.set_origin_to_geometry(obj)


@pytest.mark.parametrize("preset", [None, 3, ["figure"]])
def test_a_preset_that_is_not_a_name_is_reported(preset):
    from blender_gala.scene import presets

    with pytest.raises(ValueError):
        presets.get_preset(preset)


def test_an_unknown_material_lists_the_ones_that_exist(clean_scene):
    import bpy

    from blender_gala.scene import materials

    obj = bpy.data.objects.new("m", bpy.data.meshes.new("m"))
    clean_scene.collection.objects.link(obj)

    with pytest.raises(ValueError, match="unknown material"):
        materials.assign_material(obj, "no such material")


def test_a_broken_coordinate_does_not_produce_a_broken_camera(clean_scene):
    from blender_gala.scene import camera, lighting

    broken = build(3, [[0.0, 0, 0], [np.nan, 0, 0], [1.0, 0, 0]])

    obj = camera.frame_target(broken, scene=clean_scene)
    lighting.three_point_lighting(broken, scene=clean_scene)

    assert np.isfinite(np.array(obj.matrix_world.translation)).all()
    assert np.isfinite(obj.data.clip_end)
    assert all(
        np.isfinite(light.data.energy)
        for light in clean_scene.objects
        if light.type == "LIGHT"
    )


# ---------------------------------------------------------------------------
# Loading a session that is unusual but not corrupt
# ---------------------------------------------------------------------------


def pymol_molecule(n=4, name="m"):
    """A minimal molecular object, spaced 3 A apart along +X."""
    from blender_gala.pymol.session import PymolMolecule

    return PymolMolecule(
        name=name,
        coord=np.array([[[float(i) * 3.0, 0.0, 0.0] for i in range(n)]], dtype=float),
        chain_id=np.array(["A"] * n, dtype="U4"),
        res_id=np.arange(1, n + 1),
        ins_code=np.array([""] * n, dtype="U1"),
        res_name=np.array(["ALA"] * n, dtype="U5"),
        atom_name=np.array(["CA"] * n, dtype="U6"),
        element=np.array(["C"] * n, dtype="U2"),
        alt_id=np.array([""] * n, dtype="U4"),
        segi=np.array([""] * n, dtype="U4"),
        b_factor=np.zeros(n),
        occupancy=np.ones(n),
        charge=np.zeros(n),
        vdw=np.full(n, 1.7),
        hetero=np.zeros(n, dtype=bool),
        label=np.array([""] * n, dtype=object),
        reps=np.full(n, 1 << 5, dtype=np.int64),  # cartoon
        color_index=np.zeros(n, dtype=int),
        bonds=np.zeros((0, 3), dtype=int),
        ss=np.array([""] * n, dtype="U1"),
    )


def written_session(session, path):
    """Write a session out and return the path, so the reader is in the loop."""
    from blender_gala.pymol.session import write_session

    write_session(session, str(path))
    return str(path)


def drawn_extent(scene):
    """How far apart the drawn measurement's endpoints are, in Blender units."""
    spans = [
        float(np.ptp(np.asarray(obj["gala_points"]).reshape(-1, 3)[:, 0]))
        for obj in scene.objects
        if obj.get("gala_points") is not None
    ]
    return max(spans, default=0.0)


def atom_span(loaded):
    """How far apart the loaded atoms are, in Blender units."""
    mesh = next(iter(loaded.molecules.values())).object.data
    xs = np.array([vertex.co[0] for vertex in mesh.vertices])
    return float(xs.max() - xs.min())


@requires_mn
def test_a_selection_named_like_an_ordinary_word_loads(clean_scene, tmp_path):
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolSelection, PymolSession

    session = PymolSession(
        molecules=[pymol_molecule()],
        selections=[PymolSelection("pocket", {"m": np.array([0, 1])})],
    )

    result = load.load_session(written_session(session, tmp_path / "s.pse"))

    assert result.skipped == []


@requires_mn
def test_measurements_are_drawn_at_the_scale_the_atoms_are(clean_scene, tmp_path):
    """At the default scale, a 9 A measurement spans the same distance as the
    9 A of atoms it was measured between."""
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolMeasurement, PymolSession

    session = PymolSession(
        molecules=[pymol_molecule()],
        measurements=[
            PymolMeasurement("d", "distance", np.array([[[0.0, 0, 0], [9.0, 0, 0]]]))
        ],
    )

    result = load.load_session(written_session(session, tmp_path / "s.pse"))

    assert drawn_extent(clean_scene) == pytest.approx(atom_span(result), rel=1e-3)


@requires_mn
def test_a_non_default_scale_keeps_measurements_on_the_atoms(clean_scene, tmp_path):
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolMeasurement, PymolSession

    session = PymolSession(
        molecules=[pymol_molecule()],
        measurements=[
            PymolMeasurement("d", "distance", np.array([[[0.0, 0, 0], [9.0, 0, 0]]]))
        ],
    )

    result = load.load_session(
        written_session(session, tmp_path / "s.pse"), scale=0.005
    )

    assert drawn_extent(clean_scene) == pytest.approx(atom_span(result), rel=1e-3)


@requires_mn
def test_a_measurement_of_zero_length_does_not_abort_the_load(clean_scene, tmp_path):
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolMeasurement, PymolSession

    session = PymolSession(
        molecules=[pymol_molecule()],
        measurements=[
            PymolMeasurement("d", "distance", np.array([[[1.0, 2, 3], [1.0, 2, 3]]]))
        ],
    )

    result = load.load_session(written_session(session, tmp_path / "s.pse"))

    assert result.molecules  # the molecules still arrived
    assert result.skipped  # and the measurement was reported, not raised


@requires_mn
@pytest.mark.parametrize("name", ["position", "sharp_face"])
def test_a_selection_named_after_a_mesh_attribute_does_not_abort_the_load(
    clean_scene, tmp_path, name
):
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolSelection, PymolSession

    session = PymolSession(
        molecules=[pymol_molecule()],
        selections=[PymolSelection(name, {"m": np.array([0, 1])})],
    )

    load.load_session(written_session(session, tmp_path / "s.pse"))


@requires_mn
def test_a_group_hierarchy_survives_the_load(clean_scene, tmp_path):
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolSession

    session = PymolSession(
        molecules=[pymol_molecule(name="child")],
        groups={"grp": "parent", "parent": ""},
    )

    load.load_session(written_session(session, tmp_path / "s.pse"))

    parent = next(
        c for c in clean_scene.collection.children_recursive if c.name == "parent"
    )
    assert [child.name for child in parent.children] == ["grp"]


@requires_mn
def test_a_state_that_is_not_there_is_reported(clean_scene, tmp_path):
    from blender_gala.pymol import load
    from blender_gala.pymol.session import PymolSession

    session = PymolSession(molecules=[pymol_molecule()])  # one state

    result = load.load_session(written_session(session, tmp_path / "s.pse"), state=99)

    assert result.skipped


# ---------------------------------------------------------------------------
# Operators run in a state the panel did not anticipate
# ---------------------------------------------------------------------------


@pytest.fixture
def registered(clean_scene):
    """The add-on registered for the duration of one test."""
    import blender_gala

    blender_gala.register()
    yield clean_scene
    blender_gala.unregister()


def has_traceback(exc):
    """Whether an operator failure was a reported error or a raw traceback."""
    return "Traceback" in str(exc)


@requires_mn
@pytest.mark.parametrize("template", ["{bogus}", "{", "{resi!z}"])
def test_a_label_template_that_cannot_be_formatted_is_reported(
    registered, site_molecule, template
):
    """Every one of these is a plausible typo in a panel field."""
    import bpy

    registered.gala.selection_text = "resi 1 and name CA"
    registered.gala.label_template = template

    with pytest.raises(RuntimeError) as info:
        bpy.ops.gala.label()

    assert not has_traceback(info.value)


@requires_mn
def test_a_positional_label_template_is_reported(registered, site_molecule):
    import bpy

    registered.gala.selection_text = "resi 1 and name CA"
    registered.gala.label_template = "{0}"

    with pytest.raises(RuntimeError) as info:
        bpy.ops.gala.label()

    assert not has_traceback(info.value)


@requires_mn
def test_an_operator_survives_the_molecule_being_deleted(registered, site_molecule):
    """One press of X in the outliner, and then any button in the sidebar."""
    import bpy

    bpy.data.objects.remove(site_molecule.object, do_unlink=True)

    with pytest.raises(RuntimeError) as info:
        bpy.ops.gala.expand_selection()

    assert not has_traceback(info.value)


@requires_mn
@pytest.mark.parametrize("name", ["res_id", "b_factor"])
def test_storing_a_selection_does_not_overwrite_the_structures_own_data(
    registered, site_molecule, name
):
    import bpy

    before = site_molecule.object.data.attributes[name].data_type
    registered.gala.alias_name = name
    registered.gala.selection_text = "resi 1"

    bpy.ops.gala.create_alias()

    assert site_molecule.object.data.attributes[name].data_type == before


@requires_mn
def test_deleting_a_selection_that_is_not_there_is_reported(registered, site_molecule):
    import bpy

    assert bpy.ops.gala.delete_alias(alias="ghost") == {"CANCELLED"}


@requires_mn
def test_the_publication_setup_operator_uses_the_active_molecule(
    registered, site_molecule
):
    import bpy

    site_molecule.object.location = (5.0, 2.0, 1.0)
    bpy.context.view_layer.objects.active = site_molecule.object

    bpy.ops.gala.publication_setup()

    assert list(site_molecule.object.data.materials)
