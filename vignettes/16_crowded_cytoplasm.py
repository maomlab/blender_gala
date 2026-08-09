"""Vignette 16 — a crowded cytoplasm, after David Goodsell.

A protein drawn on its own is a lie of omission. Inside a cell it is shoulder to
shoulder with everything else: *E. coli* cytoplasm is around 300 mg/mL of
macromolecule, a fifth to a third of its volume, and nothing in it is ever more
than a molecule or two away from something. Goodsell has been drawing that fact
since the 1990s — flat colour, hard outlines, everything at one scale, no
lighting to speak of, because the argument is about *how much of it there is*
and lighting would only get in the way.

Which makes it an interesting thing to build in a path tracer, because almost
every default has to be turned off. What is left is:

* **Geometry nodes** placing a hundred-odd copies of four proteins;
* **Freestyle**, Blender's line renderer, drawing an ink outline on every
  silhouette — the single feature that makes this style read as illustration
  rather than as a render;
* **An orthographic camera**, so a molecule at the back of the slab is the same
  size as one at the front and the picture can be measured with a ruler;
* flat matte materials and one soft light, so that colour means identity and
  nothing else.

The packing is not decorative. Positions are rejection-sampled with each
species' own radius, and the vignette reports what volume fraction it reached,
against the measured 20-30% of real cytoplasm.

The cast is four *E. coli* proteins, in roughly their order of abundance:
elongation factor Tu (1EFC), triosephosphate isomerase (1TRE), adenylate kinase
(4AKE) and the GroEL chaperonin (1GRL). The ribosomes that ought to be here as
well are 2.5 million daltons each and would have made this a vignette about
waiting for a download.

    blender --background --python vignettes/16_crowded_cytoplasm.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, save_blend, setup

mn, gala = setup()

import bpy

from blender_gala.core.entity import AtomStructure
from blender_gala.core.units import world_scale_of
from blender_gala.scene.lighting import LightSpec
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material
from blender_gala.scene.presets import get_preset

# The slab, in angstrom: wide and tall across the picture, shallow into it.
# Deep enough that molecules overlap and occlude each other, shallow enough
# that the picture is a section rather than a fog.
SLAB = np.array([760.0, 200.0, 470.0])

# How close two molecules may come, as a fraction of the bounding sphere. Not
# 1.0, and the reason is the whole difficulty of packing a cell: a protein's
# bounding sphere is several times the protein. Pack bounding spheres until
# they jam and the result is about 5% protein by volume — a quarter of what a
# cell manages, because real molecules interlock, a lobe of one sitting in the
# groove of the next. Purpose-built packers (cellPACK, and Goodsell's own)
# work from the shape. Shrinking the exclusion sphere is the cheap stand-in
# for that, and it is why neighbours here touch rather than merely approach.
EXCLUSION = 0.62

# What fraction of the slab to fill with protein, and how hard to try before
# giving up on one molecule. 10% is what this sampler reaches comfortably; a
# cell reaches two to three times that, for the reason above.
TARGET_OCCUPANCY = 0.10
ATTEMPTS_PER_MOLECULE = 600

# Volume per non-hydrogen atom in a folded protein, in cubic angstrom. Close
# enough for an occupancy figure, which is all it is used for.
VOLUME_PER_ATOM = 10.0

# How much of the slab's width the frame covers. Below 1 so the crowd is cut
# off by the edge of the picture instead of ending inside it.
CROP = 0.84

# Goodsell's palette is flat and unlit, so the colours have to do all of the
# separating. Warm against cool, on cream.
PAPER = (0.93, 0.89, 0.80, 1.0)
SPECIES = (
    # code, name, colour, relative abundance
    ("1efc", "EF-Tu", "#d97a3c", 8.0),
    ("1tre", "triosephosphate isomerase", "#7d9445", 5.0),
    ("4ake", "adenylate kinase", "#b5544e", 5.0),
    ("1grl", "GroEL", "#3f7f8c", 1.0),
)


# ---------------------------------------------------------------------------
heading("1. Four proteins, each one flat colour")
# ---------------------------------------------------------------------------
# Surfaces rather than cartoons: at this scale a ribbon is a scribble, and the
# thing being shown is how much room each molecule takes up.
#
# `quality=1` is much coarser than the default, and the number matters more
# than it looks. Freestyle builds a view map over every triangle it can see,
# and there are a hundred and twenty molecules here: at quality 2 this scene
# peaks at 23 GB and is killed on a 16 GB CI runner, and at quality 1 it peaks
# at 6.6 GB and renders in a third of the time. Across a hundred molecules a
# few hundred pixels wide the two are indistinguishable — the outline is doing
# the work, not the tessellation.
scene = bpy.context.scene
crowd_collection = bpy.data.collections.new("GALA Crowd Species")
scene.collection.children.link(crowd_collection)

species = []
for index, (code, name, colour, abundance) in enumerate(SPECIES):
    molecule = load_structure(code)
    molecule.add_style(mn.StyleSurface(quality=1), color=None)
    gala.color_by_selection(molecule, {"all": colour})
    gala.assign_material(
        molecule,
        build_material(
            MATERIAL_PRESETS["protein"].with_(
                roughness=1.0,
                specular=0.05,
                subsurface_weight=0.0,
                ao_strength=0.25,
                ao_distance=0.15,
            ),
            name=f"Crowd {name}",
        ),
        style="surface",
    )

    structure = AtomStructure.from_any(molecule)
    gala.set_origin_to_geometry(molecule, method="centroid", move_to_world_origin=True)
    _, radius = structure.bounding_sphere()
    scale = world_scale_of(molecule.object)

    # Out of the scene's own collection and into the source collection, and
    # hidden from the render: these four are the things being instanced, not
    # things in the picture. Collection Info reads them anyway.
    #
    # Renamed with their index first, because Collection Info hands its
    # children over in *alphabetical* order and `Instance Index` counts
    # through that order. Left as PDB codes, 1GRL sorts second and every
    # GroEL in the picture comes out as a triosephosphate isomerase.
    molecule.object.name = f"Crowd {index} {code}"
    for collection in list(molecule.object.users_collection):
        collection.objects.unlink(molecule.object)
    crowd_collection.objects.link(molecule.object)
    molecule.object.hide_render = True

    species.append(
        {
            "name": name,
            "object": molecule.object,
            "radius": radius / scale,
            "volume": structure.n_atoms * VOLUME_PER_ATOM,
            "abundance": abundance,
        }
    )
    print(
        f"  {name:28s} {structure.n_atoms:6d} atoms, "
        f"radius {radius / scale:5.1f} A, volume {structure.n_atoms * VOLUME_PER_ATOM / 1000:6.1f} nm^3"
    )

SCALE = world_scale_of(species[0]["object"])


# ---------------------------------------------------------------------------
heading("2. Pack the slab, and count what that comes to")
# ---------------------------------------------------------------------------
generator = np.random.default_rng(20240808)

slab_volume = float(np.prod(SLAB))
wanted_volume = TARGET_OCCUPANCY * slab_volume

# How many of each. The abundances fix the *ratio*; the occupancy target fixes
# the total. One scale factor solves both, because
# sum(count_i * volume_i) = wanted_volume with count_i proportional to
# abundance_i is a single equation in a single unknown.
weights = np.array([entry["abundance"] for entry in species])
weights = weights / weights.sum()
volumes = np.array([entry["volume"] for entry in species])
factor = wanted_volume / float((weights * volumes).sum())
wanted_counts = np.maximum(1, np.round(weights * factor).astype(int))

# Biggest first. Drawn at random instead, GroEL loses every time: it is a
# twentieth of the population and needs eight times the room, so by the time
# its turn comes there is nowhere in the slab left to put it, and a cytoplasm
# with no chaperonins in it is the one thing everyone would notice.
queue = [
    index
    for index in sorted(range(len(species)), key=lambda i: -species[i]["radius"])
    for _ in range(int(wanted_counts[index]))
]

placed_positions: list[np.ndarray] = []
placed_radii: list[float] = []
placed_species: list[int] = []
filled = 0.0
refused = 0

for index in queue:
    radius = species[index]["radius"] * EXCLUSION
    for _ in range(ATTEMPTS_PER_MOLECULE):
        candidate = (generator.random(3) - 0.5) * (SLAB - 2.0 * radius)
        if placed_positions:
            gaps = np.linalg.norm(np.array(placed_positions) - candidate, axis=1)
            if np.any(gaps < np.array(placed_radii) + radius):
                continue
        placed_positions.append(candidate)
        placed_radii.append(radius)
        placed_species.append(index)
        filled += species[index]["volume"]
        break
    else:
        refused += 1

counts = {
    entry["name"]: placed_species.count(number) for number, entry in enumerate(species)
}
print(f"  {len(placed_positions)} molecules placed: {counts}")
if refused:
    print(f"  {refused} found nowhere to go in {ATTEMPTS_PER_MOLECULE} tries")
print(
    f"  volume fraction {filled / slab_volume * 100:.1f}%, "
    f"against 20-30% measured in E. coli"
)
print(f"  slab {SLAB[0]:.0f} x {SLAB[1]:.0f} x {SLAB[2]:.0f} A")


# ---------------------------------------------------------------------------
heading("3. One point per molecule, carrying which one it is")
# ---------------------------------------------------------------------------
points = bpy.data.meshes.new("Crowd Points")
points.from_pydata([tuple(p * SCALE) for p in placed_positions], [], [])
points.update()

kind = points.attributes.new("species", "INT", "POINT")
kind.data.foreach_set("value", placed_species)

# A random orientation each. Nothing in a cell is aligned with anything.
turn = points.attributes.new("euler", "FLOAT_VECTOR", "POINT")
turn.data.foreach_set(
    "vector",
    (generator.random((len(placed_positions), 3)) * 2 * math.pi).ravel().tolist(),
)

crowd = bpy.data.objects.new("Cytoplasm", points)
scene.collection.objects.link(crowd)


# ---------------------------------------------------------------------------
heading("4. Collection Info, and one instance picked per point")
# ---------------------------------------------------------------------------
# Instance on Points can take a whole collection as its instance input and
# choose between the children per point — which is exactly "four species, one
# per position". `Pick Instance` is the switch that turns the collection from
# one clump into a menu.
tree = bpy.data.node_groups.new("GALA Cytoplasm", "GeometryNodeTree")
tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

group_input = tree.nodes.new("NodeGroupInput")
group_input.location = (-600, 0)
group_output = tree.nodes.new("NodeGroupOutput")
group_output.location = (400, 0)

sources = tree.nodes.new("GeometryNodeCollectionInfo")
sources.label = "The four species"
sources.inputs["Collection"].default_value = crowd_collection
sources.inputs["Separate Children"].default_value = True
sources.inputs["Reset Children"].default_value = True
sources.location = (-350, -180)

which = tree.nodes.new("GeometryNodeInputNamedAttribute")
which.data_type = "INT"
which.inputs["Name"].default_value = "species"
which.location = (-350, -360)

euler = tree.nodes.new("GeometryNodeInputNamedAttribute")
euler.data_type = "FLOAT_VECTOR"
euler.inputs["Name"].default_value = "euler"
euler.location = (-350, 180)

rotation = tree.nodes.new("FunctionNodeEulerToRotation")
rotation.location = (-120, 180)
tree.links.new(euler.outputs["Attribute"], rotation.inputs["Euler"])

instancer = tree.nodes.new("GeometryNodeInstanceOnPoints")
instancer.location = (120, 0)
instancer.inputs["Pick Instance"].default_value = True
tree.links.new(group_input.outputs[0], instancer.inputs["Points"])
tree.links.new(sources.outputs["Instances"], instancer.inputs["Instance"])
tree.links.new(which.outputs["Attribute"], instancer.inputs["Instance Index"])
tree.links.new(rotation.outputs["Rotation"], instancer.inputs["Rotation"])
tree.links.new(instancer.outputs["Instances"], group_output.inputs[0])

modifier = crowd.modifiers.new("GALA Cytoplasm", "NODES")
modifier.node_group = tree
bpy.context.view_layer.update()
print(f"  node group: {tree.name}, {len(tree.nodes)} nodes")


# ---------------------------------------------------------------------------
heading("5. Flat light, flat film, and an orthographic camera")
# ---------------------------------------------------------------------------
# Perspective would make the near molecules bigger, and the whole point of the
# picture is that they are all the same size. An orthographic camera has no
# vanishing point and no field of view, so its framing is one number: how wide
# a slice of world the frame covers.
gala.setup_render(preset=QUALITY, transparent=False)
gala.scene.render.setup_color_management()

preset = get_preset(QUALITY)
scene.render.resolution_x = preset.resolution[0]
scene.render.resolution_y = int(preset.resolution[0] * SLAB[2] / SLAB[0])

camera = gala.frame_target(crowd, viewpoint="front", margin=1.0)
camera.data.type = "ORTHO"
# Cropped inside the slab rather than fitted to it, so the crowd runs off all
# four edges. A cell has no edges, and a picture of one with a margin of empty
# paper around it says the opposite of what it is for.
camera.data.ortho_scale = float(SLAB[0] * CROP * SCALE)
camera.data.clip_start = 0.01
camera.data.clip_end = float(
    np.linalg.norm(np.array(camera.location)) + SLAB[1] * SCALE
)

# One big soft light almost square on, and no rim: shadows here would be read
# as colour, and colour means species.
gala.three_point_lighting(
    crowd,
    energy=0.9,
    softness=4.0,
    specs=(
        LightSpec(
            "Flat", azimuth=12.0, elevation=22.0, power=1.0, size=6.0, distance=4.0
        ),
    ),
)

world = scene.world
world.node_tree.nodes.clear()
world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
paper = world.node_tree.nodes.new("ShaderNodeBackground")
paper.inputs["Color"].default_value = PAPER
paper.inputs["Strength"].default_value = 0.6
world.node_tree.links.new(paper.outputs["Background"], world_output.inputs["Surface"])
print(
    f"  orthographic, {SLAB[0]:.0f} A across, {scene.render.resolution_x} x {scene.render.resolution_y}"
)


# ---------------------------------------------------------------------------
heading("6. Freestyle, for the ink")
# ---------------------------------------------------------------------------
# Freestyle is a separate renderer that runs after Cycles and draws lines
# rather than pixels. It finds the silhouettes in the scene — the edges where a
# surface turns away from the camera — and strokes them. That single outline is
# most of what the eye reads as "illustration".
scene.render.use_freestyle = True
scene.render.line_thickness_mode = "ABSOLUTE"
scene.render.line_thickness = 1.0

view_layer = scene.view_layers[0]
view_layer.use_freestyle = True
settings = view_layer.freestyle_settings
for existing in list(settings.linesets):
    settings.linesets.remove(existing)

lineset = settings.linesets.new("GALA Outline")
lineset.select_silhouette = True
lineset.select_border = True
lineset.select_crease = False
lineset.select_edge_mark = False

style = lineset.linestyle
style.color = (0.10, 0.09, 0.08)
style.thickness = 1.4
style.thickness_position = "CENTER"
print(f"  freestyle: silhouette + border at {style.thickness} px")


# ---------------------------------------------------------------------------
heading("7. Render")
# ---------------------------------------------------------------------------
render(gala, "16_crowded_cytoplasm")
print("\n  Every molecule the same size, every colour a species, and no space.")


# ---------------------------------------------------------------------------
heading("8. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# Add a species by putting another molecule in the `GALA Crowd Species`
# collection and giving some points its index — the node group does not need
# to change. Turn Freestyle off in the render properties to see how much of
# the style was the outline.
save_blend("16_crowded_cytoplasm")
