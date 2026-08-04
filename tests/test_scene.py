"""Tests for the scene-setup layer. These need a running Blender."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_bpy, requires_mn

pytestmark = [pytest.mark.bpy, requires_bpy]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_are_well_formed():
    from blender_gala.scene import presets

    for name, preset in presets.PRESETS.items():
        assert preset.name == name
        assert preset.resolution[0] > 0 and preset.resolution[1] > 0
        assert preset.samples > 0
        assert preset.description


def test_get_preset_accepts_names_and_objects():
    from blender_gala.scene import presets

    figure = presets.get_preset("figure")
    assert presets.get_preset(figure) is figure
    assert presets.get_preset("FIGURE") is figure


def test_unknown_preset_raises():
    from blender_gala.scene import presets

    with pytest.raises(ValueError, match="unknown render preset"):
        presets.get_preset("enormous")


def test_preset_scaling():
    from blender_gala.scene import presets

    half = presets.PRESETS["figure"].scaled(0.5)
    assert half.resolution == (1000, 1000)
    assert half.samples == presets.PRESETS["figure"].samples


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------


def test_setup_render_applies_the_preset(clean_scene):
    from blender_gala.scene import presets, render

    render.setup_render("draft", use_gpu=False, scene=clean_scene)
    preset = presets.PRESETS["draft"]

    assert clean_scene.render.engine == "CYCLES"
    assert clean_scene.render.resolution_x == preset.resolution[0]
    assert clean_scene.cycles.samples == preset.samples
    assert clean_scene.cycles.use_adaptive_sampling
    assert clean_scene.render.film_transparent


def test_transparent_film_writes_rgba(clean_scene):
    from blender_gala.scene import render

    render.set_transparent(True, scene=clean_scene)
    assert clean_scene.render.film_transparent
    assert clean_scene.render.image_settings.color_mode == "RGBA"

    render.set_transparent(False, scene=clean_scene)
    assert not clean_scene.render.film_transparent


def test_resolution_validation(clean_scene):
    from blender_gala.scene import render

    render.set_resolution(640, 480, scene=clean_scene)
    assert (clean_scene.render.resolution_x, clean_scene.render.resolution_y) == (
        640,
        480,
    )

    with pytest.raises(ValueError, match="positive"):
        render.set_resolution(0, 100, scene=clean_scene)


def test_unknown_engine_raises(clean_scene):
    from blender_gala.scene import render

    with pytest.raises(ValueError, match="CYCLES"):
        render.setup_render(engine="RENDERMAN", scene=clean_scene)


def test_unknown_view_transform_raises(clean_scene):
    from blender_gala.scene import render

    with pytest.raises(ValueError, match="unknown view transform"):
        render.setup_color_management("Sepia", scene=clean_scene)


def test_gpu_report_is_always_returned(clean_scene):
    from blender_gala.scene import render

    report = render.setup_render("draft", use_gpu=False, scene=clean_scene)
    assert report.message
    assert not report.enabled
    assert clean_scene.cycles.device == "CPU"


# ---------------------------------------------------------------------------
# Origin
# ---------------------------------------------------------------------------


def test_geometry_centre_methods():
    from blender_gala.scene import origin

    points = np.array([[0.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
    assert origin.geometry_centre(points, "centroid")[0] == pytest.approx(4.0)
    assert origin.geometry_centre(points, "bounds")[0] == pytest.approx(5.0)
    weighted = origin.geometry_centre(points, "mass", weights=np.array([1.0, 1.0, 8.0]))
    assert weighted[0] == pytest.approx(8.2)


def test_geometry_centre_validation():
    from blender_gala.scene import origin

    with pytest.raises(ValueError, match="method"):
        origin.geometry_centre(np.zeros((3, 3)), "middle")
    with pytest.raises(ValueError, match="non-empty"):
        origin.geometry_centre(np.zeros((0, 3)), "centroid")
    with pytest.raises(ValueError, match="weights"):
        origin.geometry_centre(np.zeros((3, 3)), "mass")


@requires_mn
def test_set_origin_keeps_geometry_in_place(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.scene import origin

    structure = AtomStructure.from_any(site_molecule)
    before = structure.world_positions().copy()

    origin.set_origin_to_geometry(structure, method="centroid")
    after = structure.world_positions()

    assert np.allclose(before, after, atol=1e-6), "atoms must not move in world space"


@requires_mn
def test_set_origin_moves_the_origin_onto_the_molecule(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.scene import origin

    structure = AtomStructure.from_any(site_molecule)
    expected = structure.world_positions().mean(axis=0)

    origin.set_origin_to_geometry(structure, method="centroid")
    assert np.allclose(
        np.array(site_molecule.object.matrix_world.translation), expected
    )


@requires_mn
def test_set_origin_to_world_origin(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.scene import origin

    structure = AtomStructure.from_any(site_molecule)
    origin.set_origin_to_geometry(structure, move_to_world_origin=True)

    assert np.allclose(np.array(site_molecule.object.matrix_world.translation), 0.0)
    assert np.allclose(structure.world_positions().mean(axis=0), 0.0, atol=1e-6)


@requires_mn
def test_set_origin_honours_a_selection(site_molecule):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.scene import origin

    structure = AtomStructure.from_any(site_molecule)
    ligand_centre = structure.world_positions()[structure.indices("resn LIG")].mean(
        axis=0
    )

    origin.set_origin_to_geometry(structure, selection="resn LIG")
    assert np.allclose(
        np.array(site_molecule.object.matrix_world.translation), ligand_centre
    )


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------


def test_three_point_rig_structure(clean_scene):
    import bpy

    from blender_gala.scene import lighting

    cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
    clean_scene.collection.objects.link(cube)

    rig = lighting.three_point_lighting(scene=clean_scene)
    assert rig.name == lighting.RIG_NAME
    assert rig.get("gala_type") == "light_rig"

    lights = [child for child in rig.children if child.type == "LIGHT"]
    assert len(lights) == 3
    assert {light.get("gala_type") for light in lights} == {
        "key_light",
        "fill_light",
        "rim_light",
    }


def test_light_power_scales_with_subject_size(clean_scene):
    from blender_gala.scene.lighting import THREE_POINT, _make_area_light

    small = _make_area_light(THREE_POINT[0], 1.0, 1.0, 3.0, 1.0, clean_scene)
    large = _make_area_light(THREE_POINT[0], 2.0, 1.0, 3.0, 1.0, clean_scene)
    # Power goes as distance squared, and distance goes as radius.
    assert large.data.energy == pytest.approx(small.data.energy * 4.0)


def test_rebuilding_the_rig_does_not_duplicate(clean_scene):
    from blender_gala.core import collections as gala_collections
    from blender_gala.scene import lighting

    lighting.three_point_lighting(scene=clean_scene)
    lighting.three_point_lighting(scene=clean_scene)

    collection = lighting.bpy.data.collections.get(gala_collections.LIGHTING)
    assert len(collection.objects) == 4  # rig plus three lights


def test_clear_lighting(clean_scene):
    from blender_gala.scene import lighting

    lighting.three_point_lighting(scene=clean_scene)
    assert lighting.clear_lighting(scene=clean_scene) == 4


def test_builtin_hdris_are_available():
    from blender_gala.scene import lighting

    available = lighting.list_hdris()
    assert "studio" in available
    assert available["studio"].endswith(".exr")


def test_hdri_lighting_builds_a_world(clean_scene):
    from blender_gala.scene import lighting

    world = lighting.hdri_lighting("studio", strength=0.5, scene=clean_scene)
    assert world.use_nodes
    node_types = {node.bl_idname for node in world.node_tree.nodes}
    assert "ShaderNodeTexEnvironment" in node_types
    assert "ShaderNodeMapping" in node_types
    background = next(
        n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"
    )
    assert background.inputs["Strength"].default_value == pytest.approx(0.5)


def test_unknown_hdri_raises(clean_scene):
    from blender_gala.scene import lighting

    with pytest.raises(FileNotFoundError, match="not found"):
        lighting.hdri_lighting("nonexistent", scene=clean_scene)


def test_invalid_lighting_arguments(clean_scene):
    from blender_gala.scene import lighting

    with pytest.raises(ValueError, match="backend"):
        lighting.three_point_lighting(backend="blender_internal", scene=clean_scene)
    with pytest.raises(ValueError, match="distance"):
        lighting.three_point_lighting(distance=0.0, scene=clean_scene)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def test_every_material_preset_builds():
    from blender_gala.scene import materials

    for name in materials.MATERIAL_PRESETS:
        material = materials.build_material(name)
        assert material.use_nodes
        principled = next(
            n
            for n in material.node_tree.nodes
            if n.bl_idname == "ShaderNodeBsdfPrincipled"
        )
        assert principled.outputs["BSDF"].is_linked


def test_material_spec_values_reach_the_shader():
    from blender_gala.scene import materials

    spec = materials.MATERIAL_PRESETS["protein"].with_(roughness=0.9, metallic=0.3)
    material = materials.build_material(spec, name="GALA Test Rough")
    principled = next(
        n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    assert principled.inputs["Roughness"].default_value == pytest.approx(0.9)
    assert principled.inputs["Metallic"].default_value == pytest.approx(0.3)


def test_attribute_colour_is_wired_in():
    """Base Color must be driven by the per-atom colour, whether that comes
    from Molecular Nodes' colour group or from a plain attribute lookup."""
    from blender_gala.scene import materials

    material = materials.build_material("protein", name="GALA Test Attribute")
    principled = next(
        n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    assert principled.inputs["Base Color"].is_linked

    source = principled.inputs["Base Color"].links[0].from_node
    if source.bl_idname == "ShaderNodeAttribute":
        assert source.attribute_name == "Color"
    else:
        assert source.bl_idname == "ShaderNodeGroup"
        assert "Color" in source.node_tree.name


def test_ambient_occlusion_is_opt_in():
    from blender_gala.scene import materials

    plain = materials.build_material("protein", name="GALA Test No AO")
    assert not any(
        n.bl_idname == "ShaderNodeAmbientOcclusion" for n in plain.node_tree.nodes
    )

    spec = materials.MATERIAL_PRESETS["protein"].with_(ao_strength=0.5)
    with_ao = materials.build_material(spec, name="GALA Test AO")
    ao = next(
        n
        for n in with_ao.node_tree.nodes
        if n.bl_idname == "ShaderNodeAmbientOcclusion"
    )
    assert ao.outputs["Color"].is_linked


def test_unknown_material_preset_raises():
    from blender_gala.scene import materials

    with pytest.raises(ValueError, match="unknown material preset"):
        materials.build_material("unobtainium")


def test_get_material_is_cached():
    from blender_gala.scene import materials

    assert materials.get_material("ligand") is materials.get_material("ligand")


def test_unknown_material_scheme_raises(clean_scene):
    from blender_gala.scene import materials

    with pytest.raises(ValueError, match="unknown material scheme"):
        materials.assign_materials(None, scheme="neon")


@requires_mn
def test_assign_materials_maps_styles(site_molecule):
    from blender_gala.scene import materials

    site_molecule.add_style("cartoon")
    site_molecule.add_style("ball_and_stick")

    assigned = materials.assign_materials(site_molecule, scheme="chemistry")
    assert assigned.get("cartoon") == "protein"
    assert assigned.get("ball_and_stick") == "ligand"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_ensure_camera_creates_one(clean_scene):
    from blender_gala.scene import camera

    assert clean_scene.camera is None
    obj = camera.ensure_camera(clean_scene)
    assert obj.type == "CAMERA"
    assert clean_scene.camera is obj
    assert camera.ensure_camera(clean_scene) is obj


def test_unknown_viewpoint_raises(clean_scene):
    from blender_gala.scene import camera

    with pytest.raises(ValueError, match="unknown viewpoint"):
        camera.frame_target(viewpoint="dutch angle", scene=clean_scene)

    with pytest.raises(ValueError, match="azimuth"):
        camera.frame_target(viewpoint=(1.0, 2.0, 3.0), scene=clean_scene)


@requires_mn
def test_frame_target_fits_the_molecule(site_molecule):
    import bpy

    from blender_gala.core.entity import AtomStructure
    from blender_gala.scene import camera, render

    render.set_resolution(800, 800)
    structure = AtomStructure.from_any(site_molecule)
    centre, radius = structure.bounding_sphere()

    obj = camera.frame_target(structure, viewpoint="front", margin=1.2)
    distance = np.linalg.norm(np.array(obj.location) - centre)
    assert distance > radius, "the camera must sit outside the molecule"

    # The molecule centre must project inside the frame.
    from bpy_extras.object_utils import world_to_camera_view

    projected = world_to_camera_view(
        bpy.context.scene,
        obj,
        bpy.context.scene.camera.matrix_world.inverted()
        @ bpy.context.scene.camera.matrix_world
        @ __import__("mathutils").Vector(centre),
    )
    assert 0.0 <= projected.x <= 1.0
    assert 0.0 <= projected.y <= 1.0


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def test_enable_passes(clean_scene):
    from blender_gala.scene import compositing

    view_layer = clean_scene.view_layers[0]
    enabled = compositing.enable_passes(view_layer=view_layer)

    assert view_layer.use_pass_z
    assert view_layer.use_pass_cryptomatte_object
    assert view_layer.use_pass_cryptomatte_material
    assert view_layer.pass_cryptomatte_depth == 6
    assert "cryptomatte_object" in enabled


def test_cryptomatte_layers_reflect_the_passes(clean_scene):
    from blender_gala.scene import compositing

    view_layer = clean_scene.view_layers[0]
    compositing.enable_passes(cryptomatte=True, view_layer=view_layer)
    assert compositing.cryptomatte_layers(view_layer) == [
        "CryptoObject",
        "CryptoMaterial",
        "CryptoAsset",
    ]

    compositing.enable_passes(cryptomatte=False, view_layer=view_layer)
    assert compositing.cryptomatte_layers(view_layer) == []


def test_compositor_connects_render_layers_to_output(clean_scene):
    from blender_gala.scene import compositing

    tree = compositing.setup_compositor(scene=clean_scene)
    output = compositing._output_node(tree)
    assert output.inputs[0].is_linked

    render_layers = next(
        n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"
    )
    assert render_layers.outputs["Image"].is_linked


def test_compositor_is_idempotent(clean_scene):
    from blender_gala.scene import compositing

    tree = compositing.setup_compositor(scene=clean_scene)
    first = len(tree.nodes)
    compositing.setup_compositor(scene=clean_scene)
    assert len(tree.nodes) == first


def test_compositor_adds_cryptomatte_nodes(clean_scene):
    from blender_gala.scene import compositing

    tree = compositing.setup_compositor(cryptomatte=True, scene=clean_scene)
    crypto = [n for n in tree.nodes if n.bl_idname == "CompositorNodeCryptomatteV2"]
    assert len(crypto) == 3
    # Deliberately unconnected to the output: matting the beauty pass would
    # defeat the point of shipping mattes.
    for node in crypto:
        assert not node.outputs[0].is_linked


def test_depth_cue_range_validation(clean_scene):
    from blender_gala.scene import compositing

    with pytest.raises(ValueError, match="far > near"):
        compositing.setup_compositor(depth_cue_range=(100.0, 10.0), scene=clean_scene)


def test_clear_compositor(clean_scene):
    from blender_gala.scene import compositing

    compositing.setup_compositor(scene=clean_scene)
    assert compositing.clear_compositor(scene=clean_scene) > 0
    assert compositing.clear_compositor(scene=clean_scene) == 0


def test_file_output_node(clean_scene, tmp_path):
    from blender_gala.scene import compositing

    compositing.setup_compositor(scene=clean_scene)
    node = compositing.add_file_output(str(tmp_path), scene=clean_scene)
    assert node is not None
    assert node.format.file_format == "OPEN_EXR_MULTILAYER"
    assert len(node.inputs) >= 2


def test_set_exr_output(clean_scene, tmp_path):
    from blender_gala.scene import compositing

    path = compositing.set_exr_output(str(tmp_path / "passes.exr"), scene=clean_scene)
    assert clean_scene.render.image_settings.file_format == "OPEN_EXR_MULTILAYER"
    assert path.endswith("passes.exr")


def test_depth_of_field_needs_a_camera(clean_scene):
    from blender_gala.scene import compositing

    with pytest.raises(RuntimeError, match="no camera"):
        compositing.depth_of_field(scene=clean_scene)


def test_depth_of_field_configures_the_camera(clean_scene):
    from blender_gala.scene import camera, compositing

    camera.ensure_camera(clean_scene)
    data = compositing.depth_of_field(fstop=4.0, focus_distance=2.0, scene=clean_scene)
    assert data.dof.use_dof
    assert data.dof.aperture_fstop == pytest.approx(4.0)

    compositing.depth_of_field(enable=False, scene=clean_scene)
    assert not data.dof.use_dof


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------


@requires_mn
def test_publication_setup_end_to_end(site_molecule):
    import bpy

    from blender_gala.scene import setup

    site_molecule.add_style("cartoon")
    report = setup.publication_setup(
        site_molecule, preset="draft", use_gpu=False, scene=bpy.context.scene
    )

    scene = bpy.context.scene
    assert scene.render.engine == "CYCLES"
    assert scene.render.film_transparent
    assert scene.view_layers[0].use_pass_cryptomatte_object
    assert scene.camera is not None
    assert bpy.data.objects.get("GALA Light Rig") is not None
    assert report.origin == "centroid"
    assert "resolution" in str(report)


def test_publication_setup_without_a_molecule(clean_scene):
    from blender_gala.scene import setup

    report = setup.publication_setup(
        None, preset="draft", use_gpu=False, scene=clean_scene
    )
    assert clean_scene.render.engine == "CYCLES"
    assert "skipped" in report.origin


def test_publication_setup_rejects_unknown_lighting(clean_scene):
    from blender_gala.scene import setup

    with pytest.raises(ValueError, match="lighting_style"):
        setup.publication_setup(None, lighting_style="candles", scene=clean_scene)
