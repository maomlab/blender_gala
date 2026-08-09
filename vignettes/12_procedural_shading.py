"""Vignette 12 — shader nodes on top of a Gala material.

`build_material` gives you a Principled BSDF with sensible numbers in it. It is
a *node tree*, though, and the shader editor is where the rest of Blender's
texturing lives — so a Gala material is a starting point rather than a fixed
list of options.

Three additions here, each of which does something a molecular figure actually
wants:

* **Pointiness** darkens the crevices and lifts the ridges. Cycles computes it
  from the curvature of the mesh, so a cleft in a molecular surface comes out
  shaded like a cleft instead of like a flat patch of the same colour. This is
  the trick behind every molecular illustration that looks carved rather than
  drawn, and it is nearly free — no extra rays, unlike ambient occlusion.
* **A noise texture through a Bump node** gives the surface a grain finer than
  its own geometry. Nothing at this scale is smooth, and a perfectly smooth
  surface under a raking light reads as plastic.
* **A Fresnel term into emission** puts a light edge around the silhouette, so
  the molecule separates from the page without an outline drawn over it.

None of the three touches colour, in the sense that matters: they multiply and
add to whatever is already in Base Color, so `color_by_bfactor` or a chain
rainbow underneath still means what its legend says.

Hen egg-white lysozyme (1LYZ) is the subject, for its cleft — the deepest
groove on any protein everybody already knows the shape of.

    blender --background --python vignettes/12_procedural_shading.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import QUALITY, heading, load_structure, render, save_blend, setup

mn, gala = setup()


from blender_gala.core.entity import AtomStructure
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material

# Warm bone, so the shading is what the eye reads rather than the hue.
IVORY = "#c9bda6"

# How deep the crevice darkening goes, and how fine the grain is. Pointiness
# comes out of Cycles centred on 0.5, with a narrow spread, so the range that
# maps onto "crevice" to "ridge" is narrow too.
CREVICE_LOW, CREVICE_HIGH = 0.46, 0.53
CREVICE_DEPTH = 0.55
GRAIN_SCALE = 45.0
GRAIN_HEIGHT = 0.22
RIM_STRENGTH = 0.6


# ---------------------------------------------------------------------------
heading("1. A surface to shade")
# ---------------------------------------------------------------------------
molecule = load_structure("1lyz")
molecule.add_style("surface", color=None)
gala.color_by_selection(molecule, {"all": IVORY})

structure = AtomStructure.from_any(molecule)
print(f"  {structure.n_atoms} atoms, {len(set(structure.array.res_id))} residues")


# ---------------------------------------------------------------------------
heading("2. The material Gala builds")
# ---------------------------------------------------------------------------
plain = build_material(
    MATERIAL_PRESETS["protein"].with_(roughness=0.55, subsurface_weight=0.04),
    name="Lysozyme Plain",
)
print(f"  {plain.name}: {len(plain.node_tree.nodes)} nodes")
for node in plain.node_tree.nodes:
    print(f"      {node.bl_idname}")


# ---------------------------------------------------------------------------
heading("3. The same material, with three things added")
# ---------------------------------------------------------------------------
textured = build_material(
    MATERIAL_PRESETS["protein"].with_(roughness=0.55, subsurface_weight=0.04),
    name="Lysozyme Textured",
)
tree = textured.node_tree
principled = next(
    node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
)

# Whatever is currently driving the colour — the per-atom Color attribute, in
# this case, because that is how Molecular Nodes styles carry colour. The
# additions go *between* it and the shader rather than replacing it.
base_socket = principled.inputs["Base Color"]
upstream = base_socket.links[0].from_socket if base_socket.links else None

geometry = tree.nodes.new("ShaderNodeNewGeometry")
geometry.location = (-900, -260)

# --- crevices ---------------------------------------------------------------
# Pointiness is 0.5 on a flat surface, below it in a concavity and above it on
# a ridge. Map Range pulls that narrow band out to the full 0-1, and the ramp
# turns it into something to multiply the colour by: dark in the grooves,
# white on the ridges, so the ridges are left exactly as they were.
curvature = tree.nodes.new("ShaderNodeMapRange")
curvature.label = "Crevice depth"
curvature.location = (-700, -260)
curvature.inputs["From Min"].default_value = CREVICE_LOW
curvature.inputs["From Max"].default_value = CREVICE_HIGH
curvature.inputs["To Min"].default_value = 1.0 - CREVICE_DEPTH
curvature.inputs["To Max"].default_value = 1.0
curvature.clamp = True
tree.links.new(geometry.outputs["Pointiness"], curvature.inputs["Value"])

cavity = tree.nodes.new("ShaderNodeMix")
cavity.label = "Darken the crevices"
cavity.data_type = "RGBA"
cavity.blend_type = "MULTIPLY"
cavity.location = (-450, -80)
cavity.inputs["Factor"].default_value = 1.0
cavity_colours = [socket for socket in cavity.inputs if socket.type == "RGBA"]
if upstream is not None:
    tree.links.new(upstream, cavity_colours[0])
else:  # pragma: no cover - only when the style carries no Color attribute
    cavity_colours[0].default_value = principled.inputs["Base Color"].default_value
tree.links.new(curvature.outputs["Result"], cavity_colours[1])
tree.links.new(
    next(socket for socket in cavity.outputs if socket.type == "RGBA"), base_socket
)

# --- grain ------------------------------------------------------------------
# Object coordinates rather than generated ones, so the grain does not stretch
# when the molecule's bounding box changes — which it does the moment a style
# is switched or a chain hidden.
coordinates = tree.nodes.new("ShaderNodeTexCoord")
coordinates.location = (-900, 240)

noise = tree.nodes.new("ShaderNodeTexNoise")
noise.label = "Grain"
noise.location = (-700, 240)
noise.inputs["Scale"].default_value = GRAIN_SCALE
noise.inputs["Detail"].default_value = 6.0
noise.inputs["Roughness"].default_value = 0.55
tree.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])

bump = tree.nodes.new("ShaderNodeBump")
bump.label = "Grain relief"
bump.location = (-450, 240)
bump.inputs["Strength"].default_value = GRAIN_HEIGHT
bump.inputs["Distance"].default_value = 0.004
tree.links.new(noise.outputs["Factor"], bump.inputs["Height"])
tree.links.new(bump.outputs["Normal"], principled.inputs["Normal"])

# --- rim --------------------------------------------------------------------
# Layer Weight's Facing output is 0 head-on and 1 at a grazing angle, which is
# the silhouette. Driving emission with it rather than mixing in a white
# colour keeps the edge alive in shadow, where an unlit rim would vanish.
facing = tree.nodes.new("ShaderNodeLayerWeight")
facing.label = "Silhouette"
facing.location = (-700, 480)
facing.inputs["Blend"].default_value = 0.32

rim = tree.nodes.new("ShaderNodeMath")
rim.label = "Rim strength"
rim.operation = "POWER"
rim.location = (-450, 480)
rim.inputs[1].default_value = 3.0
tree.links.new(facing.outputs["Facing"], rim.inputs[0])

rim_level = tree.nodes.new("ShaderNodeMath")
rim_level.operation = "MULTIPLY"
rim_level.location = (-250, 480)
rim_level.inputs[1].default_value = RIM_STRENGTH
tree.links.new(rim.outputs["Value"], rim_level.inputs[0])
tree.links.new(rim_level.outputs["Value"], principled.inputs["Emission Strength"])
principled.inputs["Emission Color"].default_value = (0.72, 0.83, 1.0, 1.0)

print(f"  {textured.name}: {len(tree.nodes)} nodes")
print(f"      crevices : Pointiness {CREVICE_LOW}-{CREVICE_HIGH} -> multiply")
print(f"      grain    : noise at scale {GRAIN_SCALE:.0f} -> bump")
print(f"      rim      : Fresnel^3 x {RIM_STRENGTH} -> emission")


# ---------------------------------------------------------------------------
heading("4. Light it so the relief has something to catch")
# ---------------------------------------------------------------------------
# A raking key, low and to one side. Grain and curvature are both about the
# angle between the surface and the light: light it flat from the camera and
# neither of them exists.
gala.setup_render(preset=QUALITY, transparent=True)
gala.scene.render.setup_color_management()
gala.set_origin_to_geometry(molecule, method="centroid", move_to_world_origin=True)
gala.three_point_lighting(molecule, energy=0.8, softness=0.85, rotation=-38.0)

# Looking into the cleft: lysozyme's substrate groove runs across the front of
# the molecule in the orientation it is deposited in.
gala.frame_target(molecule, viewpoint=(-25.0, 12.0), margin=1.12)


# ---------------------------------------------------------------------------
heading("5. Render both, from the same scene")
# ---------------------------------------------------------------------------
# One camera, one light rig, one molecule: the only difference between the two
# images is which material is on the style node, which is the only honest way
# to show what a shader does.
gala.assign_material(molecule, plain, style="surface")
render(gala, "12_procedural_plain")

gala.assign_material(molecule, textured, style="surface")
render(gala, "12_procedural_shading")


# ---------------------------------------------------------------------------
heading("6. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# Both materials are in the file. Open the shading workspace, put `Lysozyme
# Textured` in the editor and drag the Map Range handles: the crevice band is
# the one number worth tuning per molecule, because how pointy a surface is
# depends on the probe radius it was built with.
save_blend("12_procedural_shading")
