"""Tests for the parts that create scene objects: drawing, labels, operators."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from tests.conftest import requires_bpy, requires_mn

pytestmark = [pytest.mark.bpy, requires_bpy]


# ---------------------------------------------------------------------------
# Collections and tagging
# ---------------------------------------------------------------------------


def test_collections_are_nested_under_a_single_root(clean_scene):
    import bpy

    from blender_gala.core import collections as gala_collections

    interactions = gala_collections.get_collection(
        gala_collections.INTERACTIONS, clean_scene
    )
    root = bpy.data.collections[gala_collections.ROOT]

    assert interactions.name in root.children
    assert root.name in clean_scene.collection.children


def test_get_collection_is_idempotent(clean_scene):
    from blender_gala.core import collections as gala_collections

    first = gala_collections.get_collection(gala_collections.LABELS, clean_scene)
    second = gala_collections.get_collection(gala_collections.LABELS, clean_scene)
    assert first is second

    root = gala_collections.get_collection(gala_collections.ROOT, clean_scene)
    assert len(root.children) == 1


def test_tag_and_iterate(clean_scene):
    import bpy

    from blender_gala.core import collections as gala_collections

    obj = bpy.data.objects.new("thing", None)
    gala_collections.link_object(obj, gala_collections.LABELS, clean_scene)
    gala_collections.tag(obj, "label", value=3.0)

    assert obj["gala"] is True
    assert obj["gala_type"] == "label"
    assert obj["value"] == 3.0
    assert gala_collections.iter_tagged("label", clean_scene) == [obj]
    assert gala_collections.iter_tagged("other", clean_scene) == []


def test_clear_removes_only_gala_objects(clean_scene):
    import bpy

    from blender_gala.core import collections as gala_collections

    keeper = bpy.data.objects.new("keeper", None)
    clean_scene.collection.objects.link(keeper)

    victim = bpy.data.objects.new("victim", None)
    gala_collections.link_object(victim, gala_collections.LABELS, clean_scene)
    gala_collections.tag(victim, "label")

    assert gala_collections.clear(scene=clean_scene) == 1
    assert "keeper" in bpy.data.objects


# ---------------------------------------------------------------------------
# Curve and text construction
# ---------------------------------------------------------------------------


def test_make_line_dashed_creates_multiple_splines(clean_scene):
    from blender_gala.core import geometry

    obj = geometry.make_line(
        "dashes", (0, 0, 0), (1, 0, 0), dash_length=0.1, gap_length=0.05
    )
    assert obj.type == "CURVE"
    assert len(obj.data.splines) > 1
    assert obj.data.bevel_depth > 0


def test_make_line_solid_creates_one_spline(clean_scene):
    from blender_gala.core import geometry

    obj = geometry.make_line("solid", (0, 0, 0), (1, 0, 0), style="solid")
    assert len(obj.data.splines) == 1


def test_make_line_rejects_bad_input(clean_scene):
    from blender_gala.core import geometry

    with pytest.raises(ValueError, match="style"):
        geometry.make_line("bad", (0, 0, 0), (1, 0, 0), style="wiggly")
    with pytest.raises(ValueError, match="coincide"):
        geometry.make_line("bad", (0, 0, 0), (0, 0, 0))


def test_make_curve_rejects_degenerate_polylines(clean_scene):
    from blender_gala.core import geometry

    with pytest.raises(ValueError, match="two points"):
        geometry.make_curve("bad", [[(0, 0, 0)]])


def test_make_text(clean_scene):
    from blender_gala.core import geometry

    obj = geometry.make_text("label", "HIS57", (1, 2, 3), size=0.5)
    assert obj.type == "FONT"
    assert obj.data.body == "HIS57"
    assert obj.data.size == pytest.approx(0.5)
    assert tuple(obj.location) == pytest.approx((1.0, 2.0, 3.0))


def test_billboard_adds_one_constraint(clean_scene):
    import bpy

    from blender_gala.core import geometry
    from blender_gala.scene import camera

    camera.ensure_camera(clean_scene)
    obj = geometry.make_text("label", "X", (0, 0, 0))

    geometry.billboard(obj)
    geometry.billboard(obj)
    assert len([c for c in obj.constraints if c.type == "TRACK_TO"]) == 1
    assert obj.constraints[0].target is bpy.context.scene.camera


def test_billboard_without_a_camera_is_a_no_op(clean_scene):
    from blender_gala.core import geometry

    obj = geometry.make_text("label", "X", (0, 0, 0))
    assert geometry.billboard(obj) is obj
    assert len(obj.constraints) == 0


# ---------------------------------------------------------------------------
# Drawing interactions
# ---------------------------------------------------------------------------


@requires_mn
def test_draw_interactions_creates_one_object_each(site_molecule):
    import bpy

    from blender_gala.core import collections as gala_collections
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect, draw

    structure = AtomStructure.from_any(site_molecule)
    found = detect.find_interactions(structure, kinds=["hbond", "salt_bridge"])
    assert found

    created = draw.draw_interactions(found, target=structure)
    assert len(created) == len(found)

    collection = bpy.data.collections[gala_collections.INTERACTIONS]
    assert len(collection.objects) == len(found)
    for obj in created:
        assert obj.get("gala_type", "").startswith("interaction_")
        assert obj.data.materials


@requires_mn
def test_drawn_lines_land_on_the_atoms(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect, draw

    structure = AtomStructure.from_any(site_molecule)
    bond = detect.hydrogen_bonds(structure)[0]
    (obj,) = draw.draw_interactions([bond], target=structure)

    points = np.array([p.co[:3] for spline in obj.data.splines for p in spline.points])
    assert np.linalg.norm(points - bond.point_a, axis=1).min() < 1e-5
    assert np.linalg.norm(points - bond.point_b, axis=1).min() < 1e-5


@requires_mn
def test_interaction_labels_are_optional(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect, draw

    structure = AtomStructure.from_any(site_molecule)
    found = detect.hydrogen_bonds(structure)

    assert len(draw.draw_interactions(found, target=structure, label=False)) == 1
    assert len(draw.draw_interactions(found, target=structure, label=True)) == 2


@requires_mn
def test_clear_interactions_by_kind(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect, draw

    structure = AtomStructure.from_any(site_molecule)
    draw.draw_interactions(
        detect.find_interactions(structure, kinds=["hbond", "metal"]), target=structure
    )

    assert draw.clear_interactions("metal") == 2
    assert draw.clear_interactions() == 1


# ---------------------------------------------------------------------------
# Drawing measurements
# ---------------------------------------------------------------------------


@requires_mn
def test_distance_draws_a_line_and_a_label(site_molecule):
    from blender_gala.measure import draw, measurements

    draw.clear_measurements()
    result = measurements.distance(
        site_molecule, "resi 1 and name OG", "resi 2 and name OD1", draw=True
    )
    assert len(result.objects) == 2
    kinds = {obj.get("gala_type") for obj in result.objects}
    assert kinds == {"measurement_distance", "measurement_label_distance"}

    label = next(o for o in result.objects if o.type == "FONT")
    assert label.data.body == "2.80 A"


@requires_mn
def test_angle_draws_two_rays_and_an_arc(site_molecule):
    from blender_gala.measure import draw, measurements

    draw.clear_measurements()
    result = measurements.angle(
        site_molecule,
        "resi 1 and name CB",
        "resi 1 and name OG",
        "resi 2 and name OD1",
        draw=True,
    )
    kinds = [obj.get("gala_type") for obj in result.objects]
    assert kinds.count("measurement_angle") == 2
    assert "measurement_arc" in kinds


@requires_mn
def test_dihedral_draws_three_bonds_and_an_arc(site_molecule):
    from blender_gala.measure import draw, measurements

    draw.clear_measurements()
    result = measurements.dihedral(
        site_molecule,
        "resi 2 and name N",
        "resi 2 and name CA",
        "resi 2 and name CB",
        "resi 2 and name CG",
        draw=True,
    )
    kinds = [obj.get("gala_type") for obj in result.objects]
    assert kinds.count("measurement_dihedral") == 3
    assert "measurement_arc" in kinds


@requires_mn
def test_clear_measurements(site_molecule):
    from blender_gala.measure import draw, measurements

    draw.clear_measurements()
    measurements.distance(
        site_molecule, "resi 1 and name OG", "resi 2 and name OD1", draw=True
    )
    assert draw.clear_measurements() == 2


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@requires_mn
def test_label_one_per_residue(site_molecule):
    from blender_gala.annotate import labels

    created = labels.label(site_molecule, "chain A and resi 1+2", level="residue")
    assert len(created) == 2
    bodies = {obj.data.body for obj in created}
    assert bodies == {"S1", "D2"}


@requires_mn
def test_label_one_per_atom(site_molecule):
    from blender_gala.annotate import labels

    created = labels.label_atoms(site_molecule, "resi 1 and name OG+CB")
    assert {obj.data.body for obj in created} == {"OG", "CB"}


@requires_mn
def test_label_whole_selection(site_molecule):
    from blender_gala.annotate import labels

    created = labels.label(
        site_molecule, "resn LIG", level="selection", text="inhibitor"
    )
    assert len(created) == 1
    assert created[0].data.body == "inhibitor"


@requires_mn
def test_label_card_adds_a_backing_plane(site_molecule):
    from blender_gala.annotate import labels

    created = labels.label(site_molecule, "resi 1", style="card")
    types = {obj.get("gala_type") for obj in created}
    assert types == {"label", "label_card"}
    card = next(o for o in created if o.get("gala_type") == "label_card")
    assert card.parent is not None


@requires_mn
def test_label_template_fields(site_molecule):
    from blender_gala.annotate import labels

    created = labels.label(
        site_molecule, "resi 2", template="{chain}/{resn}{resi}", level="residue"
    )
    assert created[0].data.body == "A/ASP2"


@requires_mn
def test_label_validation(site_molecule):
    from blender_gala.annotate import labels
    from blender_gala.core.exceptions import EmptySelectionError

    with pytest.raises(ValueError, match="level"):
        labels.label(site_molecule, "resi 1", level="molecule")
    with pytest.raises(ValueError, match="style"):
        labels.label(site_molecule, "resi 1", style="neon")
    with pytest.raises(EmptySelectionError):
        labels.label(site_molecule, "resn XXX")


@requires_mn
def test_label_hud_registers_an_annotation(site_molecule):
    from blender_gala.annotate import labels

    annotation = labels.label_hud(site_molecule, "Figure 1", size=32)
    assert annotation.text == "Figure 1"
    assert annotation.text_size == 32


@requires_mn
def test_label_hud_says_why_when_annotations_are_missing(site_molecule):
    """Molecular Nodes older than 4.5 is out of support, not silently degraded.

    Nothing stops someone pinning an old Molecular Nodes under a new Blender,
    and the message they get should name the version that has what they want
    rather than blame them for passing the wrong object.
    """
    from blender_gala.annotate import labels

    with contextlib.suppress(AttributeError):
        del site_molecule.annotations  # a build that predates the manager

    with pytest.raises(TypeError, match=r"4\.5"):
        labels.label_hud(site_molecule, "Figure 1")


@requires_mn
def test_clear_labels(site_molecule):
    from blender_gala.annotate import labels

    labels.label(site_molecule, "chain A and resi 1+2")
    assert labels.clear_labels() == 2


# ---------------------------------------------------------------------------
# Writing colours
# ---------------------------------------------------------------------------


@requires_mn
def test_write_and_read_colours(plddt_molecule):
    from blender_gala.color import coloring

    n_atoms = len(plddt_molecule.object.data.vertices)
    result = coloring.color_by_plddt(plddt_molecule)
    assert result.n_colored == n_atoms

    stored = coloring.read_colors(plddt_molecule)
    assert np.allclose(stored, result.colors, atol=1e-6)


@requires_mn
def test_colours_are_written_to_the_attribute_molecular_nodes_reads(plddt_molecule):
    from blender_gala.color import coloring

    coloring.color_by_plddt(plddt_molecule)
    attribute = plddt_molecule.object.data.color_attributes.get("Color")
    assert attribute is not None
    assert attribute.domain == "POINT"
    assert attribute.data_type == "FLOAT_COLOR"


@requires_mn
def test_colours_survive_the_molecular_nodes_tree(plddt_molecule):
    """The mesh attribute is not what renders; the evaluated geometry is.

    Molecular Nodes' `Set Color` node stores a generated colour over the mesh
    attribute on the way to the style, so colouring used to be correct on the
    mesh and invisible in the render — a whole figure of flat pink.
    """
    import bpy

    from blender_gala.color import coloring

    plddt_molecule.add_style("cartoon")
    coloring.color_by_plddt(plddt_molecule, mode="banded")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = plddt_molecule.object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        attribute = mesh.color_attributes.get("Color")
        assert attribute is not None, "the styled geometry carries no colour"
        buffer = np.empty(len(attribute.data) * 4, dtype=np.float32)
        attribute.data.foreach_get("color", buffer)
        distinct = np.unique(np.round(buffer.reshape(-1, 4), 2), axis=0)
    finally:
        evaluated.to_mesh_clear()

    # The fixture sweeps all four confidence bands, so a single colour means
    # the tree overwrote them.
    assert len(distinct) > 1, f"every atom rendered the same colour: {distinct}"


@requires_mn
def test_selection_limits_which_atoms_are_recoloured(plddt_molecule):
    from blender_gala.color import coloring

    coloring.color_by_selection(plddt_molecule, {"all": "#000000"})
    coloring.color_by_plddt(plddt_molecule, selection="resi 1")

    stored = coloring.read_colors(plddt_molecule)
    assert not np.allclose(stored[0, :3], 0.0)
    assert np.allclose(stored[-1, :3], 0.0)


def test_write_colors_needs_an_object(site):
    from blender_gala.color import coloring
    from blender_gala.core.exceptions import StructureError

    with pytest.raises(StructureError, match="Blender object"):
        coloring.write_colors(site, np.ones((site.n_atoms, 4)))


@requires_mn
def test_write_colors_validates_shape(plddt_molecule):
    from blender_gala.color import coloring

    with pytest.raises(ValueError, match="shape"):
        coloring.write_colors(plddt_molecule, np.ones((10, 2)))
    with pytest.raises(ValueError, match="expected"):
        coloring.write_colors(plddt_molecule, np.ones((3, 4)))


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_registration_round_trip(clean_scene):
    import bpy

    import blender_gala

    blender_gala.register()
    try:
        assert hasattr(bpy.types.Scene, "gala")
        assert hasattr(bpy.ops.gala, "publication_setup")
        assert hasattr(bpy.ops.gala, "find_interactions")
    finally:
        blender_gala.unregister()
    assert not hasattr(bpy.types.Scene, "gala")


@pytest.fixture
def registered(clean_scene):
    import blender_gala

    blender_gala.register()
    yield clean_scene
    blender_gala.unregister()


@requires_mn
def test_operators_run_end_to_end(registered, site_molecule):
    import bpy

    site_molecule.add_style("ball_and_stick")
    bpy.context.view_layer.objects.active = site_molecule.object

    props = bpy.context.scene.gala
    props.preset = "draft"
    props.use_gpu = False
    props.selection_a = "resn LIG"
    props.selection_b = "protein"
    props.interaction_kinds = {"pi_stacking", "halogen"}

    assert bpy.ops.gala.publication_setup() == {"FINISHED"}
    assert bpy.ops.gala.find_interactions() == {"FINISHED"}

    props.measure_selection = "resi 1 and name OG; resi 2 and name OD1"
    assert bpy.ops.gala.measure() == {"FINISHED"}

    props.label_selection = "resn LIG"
    props.label_level = "selection"
    assert bpy.ops.gala.label() == {"FINISHED"}

    props.color_mode = "bfactor"
    assert bpy.ops.gala.color() == {"FINISHED"}

    assert bpy.ops.gala.setup_compositor() == {"FINISHED"}
    assert bpy.ops.gala.clear_all() == {"FINISHED"}


def test_operator_without_a_molecule_reports_cleanly(registered):
    """Blender turns an operator ERROR report into a RuntimeError for callers,
    so the message must name the actual problem."""
    import bpy

    for operator in (bpy.ops.gala.find_interactions, bpy.ops.gala.label):
        with pytest.raises(RuntimeError, match="Molecular Nodes"):
            operator()


@requires_mn
def test_measure_operator_uses_selected_vertices(registered, site_molecule):
    import bpy

    from blender_gala.ops.operators import selected_atom_indices

    obj = site_molecule.object
    bpy.context.view_layer.objects.active = obj

    vertices = obj.data.vertices
    flags = np.zeros(len(vertices), dtype=bool)
    flags[[0, 1]] = True
    vertices.foreach_set("select", flags)

    assert selected_atom_indices(bpy.context) == [0, 1]

    bpy.context.scene.gala.measure_selection = ""
    assert bpy.ops.gala.measure() == {"FINISHED"}


@requires_mn
def test_measure_operator_rejects_a_bad_atom_count(registered, site_molecule):
    import bpy

    obj = site_molecule.object
    bpy.context.view_layer.objects.active = obj

    flags = np.zeros(len(obj.data.vertices), dtype=bool)
    flags[0] = True
    obj.data.vertices.foreach_set("select", flags)

    bpy.context.scene.gala.measure_selection = ""
    with pytest.raises(RuntimeError, match="Select 2, 3 or 4 atoms"):
        bpy.ops.gala.measure()


def test_panels_declare_the_gala_category(registered):
    from blender_gala.ui import panels

    for panel in panels.classes:
        assert panel.bl_category == "Gala"
        assert panel.bl_space_type == "VIEW_3D"


def test_public_api_is_importable():
    import blender_gala

    for name in blender_gala.__all__:
        assert hasattr(blender_gala, name), name


# ---------------------------------------------------------------------------
# Regressions found by running the vignettes
# ---------------------------------------------------------------------------


def test_render_is_a_function_not_the_module():
    """`gala.scene.render` is a module, so the top-level name must be the
    function or every `gala.render(path)` call fails."""
    import blender_gala

    assert callable(blender_gala.render)
    assert callable(blender_gala.scene.render_image)


@requires_mn
def test_select_accepts_a_molecule(site_molecule):
    """Passing a Molecule straight to select() used to find no annotations and
    silently return an all-false mask."""
    import blender_gala as gala
    from blender_gala.core.entity import AtomStructure

    structure = AtomStructure.from_any(site_molecule)
    expected = int(structure.select("resn LIG").sum())
    assert expected > 0

    assert int(gala.select(site_molecule, "resn LIG").sum()) == expected
    assert int(gala.select(structure, "resn LIG").sum()) == expected
    assert int(gala.select(structure.array, "resn LIG").sum()) == expected
    assert gala.select_indices(site_molecule, "resn LIG").size == expected


@requires_mn
def test_select_on_a_molecule_reads_numeric_annotations(plddt_molecule):
    import blender_gala as gala

    assert int(gala.select(plddt_molecule, "b > 70").sum()) > 0
    assert int(gala.select(plddt_molecule, "b < 50").sum()) > 0


@requires_mn
def test_material_reads_colour_through_the_instancer(site_molecule):
    """Ball-and-stick renders atoms as instances, where a plain GEOMETRY
    attribute lookup returns black."""
    from blender_gala.scene import materials

    site_molecule.add_style("ball_and_stick")
    material = materials.build_material("protein", name="GALA Test Colour")

    groups = [
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeGroup"
        and getattr(node.node_tree, "name", "").startswith("MN Color Input")
    ]
    assert groups, "expected Molecular Nodes' colour-input group to be used"

    principled = next(
        n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    assert principled.inputs["Base Color"].is_linked


def test_rig_lights_are_hidden_from_camera(clean_scene):
    """An area light is an emitting surface, and the rim light sits directly
    behind the subject."""
    from blender_gala.scene import lighting

    rig = lighting.three_point_lighting(scene=clean_scene)
    lights = [child for child in rig.children if child.type == "LIGHT"]
    assert lights
    assert all(not light.visible_camera for light in lights)

    rig = lighting.three_point_lighting(visible_to_camera=True, scene=clean_scene)
    assert all(light.visible_camera for light in rig.children if light.type == "LIGHT")


def test_light_power_is_not_blinding(clean_scene):
    """1000 W at three radii blows out every pixel; the calibrated value does
    not."""
    from blender_gala.scene import lighting

    rig = lighting.three_point_lighting(scene=clean_scene)
    key = next(c for c in rig.children if c.get("gala_type") == "key_light")
    radius = rig["subject_radius"]
    # Irradiance at the subject, W/m^2, for a light at three radii.
    irradiance = key.data.energy / (4 * np.pi * (3 * radius) ** 2)
    assert 0.5 < irradiance < 5.0, irradiance


def test_orbit_keyframes_are_linear(clean_scene):
    """Blender 5 removed Action.fcurves in favour of slotted actions."""
    import bpy

    from blender_gala.scene import camera

    cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
    clean_scene.collection.objects.link(cube)

    pivot = camera.orbit(frames=60, target=cube, scene=clean_scene)
    curves = camera._action_fcurves(pivot.animation_data.action)

    assert curves, "expected keyframes on the pivot"
    for fcurve in curves:
        for keyframe in fcurve.keyframe_points:
            assert keyframe.interpolation == "LINEAR"


def test_registration_survives_a_second_copy(clean_scene):
    """A developer often has the extension installed *and* a checkout on the
    path. Registering the second copy makes Blender unregister the first's
    classes behind its back, and the first's unregister then used to raise
    'missing bl_rna attribute ... (may not be registered)' at shutdown.
    """
    from blender_gala.core.registration import (
        is_registered,
        register_classes,
        unregister_classes,
    )
    from blender_gala.ui import panels

    register_classes(panels.classes)
    assert all(is_registered(cls) for cls in panels.classes)

    # Registering again must not leave duplicates or raise.
    register_classes(panels.classes)
    assert unregister_classes(panels.classes) == len(panels.classes)

    # Unregistering what is already gone is a no-op rather than an error.
    assert unregister_classes(panels.classes) == 0


def test_is_registered_reports_this_class_not_its_base(clean_scene):
    """``bl_rna`` has to be looked for in the class's own ``__dict__``.

    Every Blender base class has a ``bl_rna``, so an inherited lookup answers
    True for every class, registered or not. That made both guards in this
    module dead code: `register_classes` called `unregister_class` on classes
    that were not registered, which segfaults Blender 4.2.
    """
    from blender_gala.core.registration import (
        is_registered,
        register_classes,
        unregister_classes,
    )
    from blender_gala.ui import panels

    cls = panels.classes[0]
    unregister_classes([cls])  # whatever ran before, start from unregistered

    assert not is_registered(cls)
    register_classes([cls])
    assert is_registered(cls)
    unregister_classes([cls])
    assert not is_registered(cls)


def test_register_unregister_is_repeatable(clean_scene):
    import bpy

    import blender_gala

    for _ in range(3):
        blender_gala.register()
        assert hasattr(bpy.types.Scene, "gala")
        blender_gala.unregister()
        assert not hasattr(bpy.types.Scene, "gala")
