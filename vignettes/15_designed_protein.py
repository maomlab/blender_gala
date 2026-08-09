"""Vignette 15 — a designed protein, lit like a press image.

Every other vignette here is a figure: transparent background, honest colour
management, nothing in the picture that is not evidence. This one is the other
job. When a design lab announces a protein, the picture that runs with it is
lit — dark set, coloured rims, a haze for the light to travel through, a bloom
on the highlights. Ian Haydon's images for the Institute for Protein Design are
the reference; so is every structure that has ever appeared on a journal cover.

That look is not a betrayal of the data. It is a different question being asked
of the same coordinates: *what is this thing*, rather than *what does this
measurement show*. Blender is unusually good at it, and none of what follows is
special to it — coloured area lights, a world volume, depth of field and a
Glare node are the same tools any product shot uses.

Top7 (1QYS) is the subject. Kuhlman et al. designed it in 2003 with a fold
that had never been observed in nature, and then solved it and found the design
had been right to 1.2 A. It is the beginning of everything that de novo protein
design has become, and it is 93 residues, which is small enough to light like
an object rather than a landscape.

    Kuhlman, B. et al. Design of a novel globular protein fold with atomic-level
    accuracy. Science 302, 1364-1368 (2003). https://doi.org/10.1126/science.1089427

    blender --background --python vignettes/15_designed_protein.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import QUALITY, heading, load_structure, render, save_blend, setup

mn, gala = setup()

import bpy

from blender_gala.core.entity import AtomStructure
from blender_gala.scene.lighting import LightSpec

# The set. A very dark blue rather than black: black clips, and a background
# with a little colour in it gives the haze something to be.
BACKDROP = (0.008, 0.012, 0.022)
HAZE_DENSITY = 0.06

# Rim colours. Two, opposed, and complementary, which is the whole trick — the
# silhouette is drawn twice, in two colours, from behind.
MAGENTA = (1.0, 0.22, 0.55)
CYAN = (0.18, 0.72, 1.0)

# The protein itself, ice-blue with warmth in the sheet so the two halves of
# the fold separate.
HELIX = "#63c6ff"
SHEET = "#ffd67a"
LOOP = "#5f7f9c"


# ---------------------------------------------------------------------------
heading("1. Top7")
# ---------------------------------------------------------------------------
molecule = load_structure("1qys")
molecule.add_style("cartoon", color=None)

structure = AtomStructure.from_any(molecule)
print(f"  {structure.n_atoms} atoms, {len(set(structure.array.res_id))} residues")

# Secondary structure is an attribute Molecular Nodes writes on import, so the
# two elements of the fold can be coloured apart without naming a residue. The
# codes are integers rather than DSSP letters: 1 helix, 2 sheet, 3 loop. Top7
# is one long helix pair across a five-strand sheet, and colouring the two
# apart is the fastest way to say that.
gala.color_by_selection(molecule, {"all": LOOP, "ss 1": HELIX, "ss 2": SHEET})

gala.setup_render(preset=QUALITY, transparent=False)
gala.scene.render.setup_color_management()
gala.set_origin_to_geometry(molecule, method="centroid", move_to_world_origin=True)
_, radius = structure.bounding_sphere()


# ---------------------------------------------------------------------------
heading("2. Three of it, at three distances")
# ---------------------------------------------------------------------------
# One protein on a black field is a specimen. Three, with two of them behind
# the plane of focus, is a scene — the out-of-focus copies give the light
# something to fall on and the frame a sense of depth, and they cost nothing
# because they share the mesh.
scene = bpy.context.scene
companions = []
for index, (offset, turn, scale) in enumerate(
    (
        ((-1.4, 3.0, -1.0), 1.1, 0.85),
        ((1.5, 4.2, 0.9), -2.3, 1.05),
    )
):
    copy = molecule.object.copy()
    copy.data = molecule.object.data
    copy.name = f"1qys companion {index}"
    copy.location = tuple(component * radius for component in offset)
    copy.rotation_euler = (0.4 * index, turn, 0.9 * index)
    copy.scale = (scale, scale, scale)
    scene.collection.objects.link(copy)
    companions.append(copy)
print(f"  {len(companions)} companions, sharing one mesh and one material")


# ---------------------------------------------------------------------------
heading("3. A glass body rather than a painted ribbon")
# ---------------------------------------------------------------------------
# `build_glass_subsurface` is Gala's other material: a subsurface body under a
# glass shell rather than one Principled BSDF. Light entering the ribbon
# scatters inside it instead of passing through, which is what makes a thin
# ribbon look like it is made of something.
from blender_gala.scene.materials import build_glass_subsurface

body = build_glass_subsurface(
    name="Designed Protein",
    mix=0.35,
    color_mix=0.85,
    subsurface_scale=6.0,
    glass_roughness=0.12,
    glass_ior=1.45,
)
gala.assign_material(molecule, body, style="cartoon")


# ---------------------------------------------------------------------------
heading("4. Light it from behind, in two colours")
# ---------------------------------------------------------------------------
# Gala's rig is a set of `LightSpec`s in spherical coordinates about the
# subject, so it takes a different set as easily as its own. This one is two
# hard coloured rims well behind the molecule and a single soft key barely
# above the camera — the opposite balance from the figure rig, where the key
# does the work and the rim only separates.
gala.three_point_lighting(
    molecule,
    energy=1.6,
    specs=(
        LightSpec(
            "Key",
            azimuth=25.0,
            elevation=18.0,
            power=0.30,
            size=3.0,
            distance=3.2,
            colour=(1.0, 0.95, 0.92),
        ),
        LightSpec(
            "Rim Magenta",
            azimuth=-140.0,
            elevation=28.0,
            power=1.6,
            size=0.6,
            distance=2.6,
            colour=MAGENTA,
        ),
        LightSpec(
            "Rim Cyan",
            azimuth=145.0,
            elevation=-12.0,
            power=1.4,
            size=0.6,
            distance=2.6,
            colour=CYAN,
        ),
    ),
)
rig = bpy.data.objects["GALA Light Rig"]
for light in rig.children:
    print(
        f"  {light.name:18s} {light.data.energy:8.2f} W  {tuple(round(c, 2) for c in light.data.color)}"
    )


# ---------------------------------------------------------------------------
heading("5. Something for the light to travel through")
# ---------------------------------------------------------------------------
# A world volume: the whole scene is inside a very thin fog, so the rim lights
# leave a visible cone rather than only landing on the protein. This is the
# single most expensive line in the vignette and the one that does the most —
# it is what separates "lit on black" from "lit in a room".
world = scene.world or bpy.data.worlds.new("GALA Designed Protein")
scene.world = world
tree = world.node_tree
tree.nodes.clear()

output = tree.nodes.new("ShaderNodeOutputWorld")
output.location = (300, 0)

background = tree.nodes.new("ShaderNodeBackground")
background.location = (0, 120)
background.inputs["Color"].default_value = (*BACKDROP, 1.0)
background.inputs["Strength"].default_value = 1.0
tree.links.new(background.outputs["Background"], output.inputs["Surface"])

haze = tree.nodes.new("ShaderNodeVolumeScatter")
haze.location = (0, -140)
haze.inputs["Color"].default_value = (0.55, 0.68, 1.0, 1.0)
haze.inputs["Density"].default_value = HAZE_DENSITY
haze.inputs["Anisotropy"].default_value = 0.45
tree.links.new(haze.outputs["Volume"], output.inputs["Volume"])

# Volume bounces are their own budget in Cycles and default to nearly none.
scene.cycles.volume_bounces = 2
scene.cycles.volume_step_rate = 2.0
print(f"  haze density {HAZE_DENSITY}, anisotropy 0.45 (forward-scattering)")


# ---------------------------------------------------------------------------
heading("6. Frame it as a portrait")
# ---------------------------------------------------------------------------
# Longer lens and a wider margin than a figure would use: the subject is meant
# to sit *in* the frame rather than fill it, and a long lens keeps the
# companions behind it from being thrown out to the edges by perspective.
camera = gala.frame_target(molecule, viewpoint=(24.0, 12.0), margin=1.65)
camera.data.lens = 110.0
gala.frame_target(molecule, viewpoint=(24.0, 12.0), margin=1.65)
gala.depth_of_field(target=molecule, fstop=2.8)

# 3:2, which is what a press image is, and what a square figure is not.
from blender_gala.scene.presets import get_preset

preset = get_preset(QUALITY)
scene.render.resolution_x = preset.resolution[0]
scene.render.resolution_y = int(preset.resolution[0] * 0.68)


# ---------------------------------------------------------------------------
heading("7. Bloom, in the compositor")
# ---------------------------------------------------------------------------
# The Glare node spreads the brightest pixels into the ones around them, which
# is what a real lens does and what makes a rim light read as bright rather
# than merely light-coloured. It happens after the render, so the strength is
# a slider rather than another hour of sampling.
compositor = bpy.data.node_groups.new("GALA Bloom", "CompositorNodeTree")
compositor.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
scene.compositing_node_group = compositor

layers = compositor.nodes.new("CompositorNodeRLayers")
layers.scene = scene
layers.location = (-400, 0)

glare = compositor.nodes.new("CompositorNodeGlare")
glare.location = (-100, 0)
glare.inputs["Type"].default_value = "Bloom"
glare.inputs["Quality"].default_value = "High"
glare.inputs["Threshold"].default_value = 0.6
glare.inputs["Strength"].default_value = 0.35
glare.inputs["Size"].default_value = 0.6
compositor.links.new(layers.outputs["Image"], glare.inputs["Image"])

group_output = compositor.nodes.new("NodeGroupOutput")
group_output.location = (200, 0)
compositor.links.new(glare.outputs["Image"], group_output.inputs[0])
print(f"  {compositor.name}: {[node.bl_idname for node in compositor.nodes]}")


# ---------------------------------------------------------------------------
heading("8. Render")
# ---------------------------------------------------------------------------
render(gala, "15_designed_protein")
print("\n  Opaque, on purpose: the background is part of the picture.")


# ---------------------------------------------------------------------------
heading("9. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# Three knobs worth turning first: the rim colours on the two lamps, the haze
# density in the world, and the Glare strength in the compositor. Between them
# they are most of the difference between one lab's press image and another's.
save_blend("15_designed_protein")
