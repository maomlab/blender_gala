"""Vignette 10 — putting a membrane protein back in its membrane.

A crystal structure of a membrane protein arrives with the membrane missing.
Every figure of one puts it back, usually as two grey lines drawn in Illustrator
afterwards, because a bilayer is a thousand copies of one molecule and nobody
models a thousand of anything by hand.

A thousand copies of one molecule is what geometry nodes is for. This one:

* asks the structure where its membrane was — the hydrophobic belt is a
  measurable thing, not a guess, and biotite's solvent accessibility plus
  Gala's selections put a number on its thickness;
* scatters points across two leaflets and instances a lipid onto each;
* opens a hole for the protein with a Geometry Proximity node, so the gap is
  the protein's own footprint rather than a circle that approximates it.

Bacteriorhodopsin (1C3W) is the structure: seven transmembrane helices around
a retinal, and the light-driven proton pump that made purple membrane famous.

The lipid is schematic — a head and two tails, the shape everyone draws — and
deliberately so. It stands for the bilayer the way a cartoon ribbon stands for
a backbone.

    blender --background --python vignettes/10_membrane.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, save_blend, setup

mn, gala = setup()

import bmesh
import bpy
import mathutils

from blender_gala.core.entity import AtomStructure
from blender_gala.core.units import world_scale_of

# The patch of membrane to build, in angstrom, and how much room to leave the
# protein inside it. The clearance is measured from the protein's own surface.
#
# A strip rather than a square: seen from just above the bilayer plane, a
# square patch puts a hundred angstrom of foreground lipid between the camera
# and the protein and the figure becomes a picture of the near edge. Long
# across the frame and shallow into it reads as the cross-section every
# membrane figure is drawn as.
PATCH_ACROSS = 260.0
PATCH_DEEP = 70.0
CLEARANCE = 3.0

# Lipid geometry, in angstrom. A phosphatidylcholine head group is about 8 A
# across and its tails run 16-18 A, which is what sets a bilayer's thickness.
HEAD_RADIUS = 2.9
TAIL_LENGTH = 16.0
TAIL_RADIUS = 1.2
TAIL_SPACING = 2.0

# How closely to pack them. Roughly 65 A^2 per lipid is the measured area per
# molecule in a fluid PC bilayer, so ~8 A between neighbours.
LIPID_SPACING = 7.5

PROTEIN = "#4c8f9c"
RETINAL = "#e0507a"
HEAD = "#e8d5a8"
TAIL = "#c9b78e"


# ---------------------------------------------------------------------------
heading("1. The protein, stood up in its membrane's frame")
# ---------------------------------------------------------------------------
# A crystal frame is arbitrary, and a bilayer built in the wrong one cuts the
# protein at an angle. Seven roughly parallel helices have one long axis
# between them, and that axis is the membrane normal: the first principal
# component of the alpha carbons, pointed at +Z.
protein = load_structure("1c3w", fallback="site.pdb")
# Cartoon for the fold, and a translucent surface over it for the volume: a
# ribbon alone reads as thin and gets lost among the lipids, and a surface
# alone hides the seven helices that are the reason to draw this protein.
protein.add_style("cartoon", selection="is_peptide", color=None)
protein.add_style("surface", selection="is_peptide", color=None)
protein.add_style("ball_and_stick", selection="is_hetero", color=None)

structure = AtomStructure.from_any(protein)
SCALE = world_scale_of(protein.object)

alpha_carbons = structure.array.coord[structure.array.atom_name == "CA"]
centre = alpha_carbons.mean(axis=0)
_, _, components = np.linalg.svd(alpha_carbons - centre)
normal = components[0]

# Any rotation taking `normal` to +Z will do; the one built from a track
# quaternion keeps the molecule from being rolled arbitrarily about it.
rotation = np.array(
    mathutils.Vector(normal.tolist()).to_track_quat("Z", "Y").to_matrix().transposed()
)
matrix = mathutils.Matrix.Identity(4)
for row in range(3):
    for column in range(3):
        matrix[row][column] = float(rotation[row][column])
matrix.translation = mathutils.Vector((-rotation @ centre * SCALE).tolist())
protein.object.matrix_world = matrix @ protein.object.matrix_world
bpy.context.view_layer.update()

print(f"  membrane normal in the crystal frame: {normal.round(3)}")
print(f"  {len(alpha_carbons)} residues, now standing on +Z")


# ---------------------------------------------------------------------------
heading("2. Where the membrane was: the hydrophobic belt")
# ---------------------------------------------------------------------------
# The part of a membrane protein that sits in the bilayer is the part whose
# *exposed* surface is greasy. Buried carbon says nothing — every protein core
# is hydrophobic — so this is solvent accessibility first, then the carbon
# fraction of what is left, slab by slab up the membrane normal.
import biotite.structure as struc

polymer = structure.array[~structure.array.hetero]
accessibility = struc.sasa(polymer, vdw_radii="Single")
accessibility = np.nan_to_num(accessibility)

# World positions, in angstrom, so the slabs are in the frame just built.
heights = structure.world_positions()[~structure.array.hetero][:, 2] / SCALE
greasy = np.isin(polymer.element, ("C", "S"))

edges = np.arange(math.floor(heights.min()), math.ceil(heights.max()) + 2.0, 2.0)
slab = np.digitize(heights, edges) - 1
exposed_area = np.bincount(slab, weights=accessibility, minlength=len(edges))
greasy_area = np.bincount(slab, weights=accessibility * greasy, minlength=len(edges))
with np.errstate(invalid="ignore", divide="ignore"):
    fraction = np.where(exposed_area > 0, greasy_area / exposed_area, 0.0)

# The core is the run of slabs whose exposed surface is mostly carbon. 0.75 is
# well clear of the ~0.55 a soluble protein's surface sits at and well below
# the ~0.9 of the belt itself, so the boundary does not move if it shifts.
in_core = fraction > 0.75
if in_core.any():
    inside = np.flatnonzero(in_core)
    core_min, core_max = float(edges[inside.min()]), float(edges[inside.max()] + 2.0)
else:  # pragma: no cover - only when the fallback fixture stands in
    core_min, core_max = -15.0, 15.0

half_thickness = 0.5 * (core_max - core_min)
mid_plane = 0.5 * (core_max + core_min)
print(f"  hydrophobic core spans z = {core_min:.1f} to {core_max:.1f} A")
print(f"  thickness {core_max - core_min:.1f} A, against ~30 A for a fluid bilayer")
for row in range(len(edges) - 1):
    if exposed_area[row] > 0 and row % 2 == 0:
        bar = "#" * int(fraction[row] * 40)
        mark = "  <-- core" if in_core[row] else ""
        print(f"      z {edges[row]:+6.1f} {fraction[row]:4.2f} {bar}{mark}")


# ---------------------------------------------------------------------------
heading("3. One lipid, built out of primitives")
# ---------------------------------------------------------------------------
# Head at the origin, tails hanging down -Z, which makes the upper leaflet the
# unrotated one. Two material slots so the head group can be a different
# colour from the tails without splitting it into two objects.
mesh = bpy.data.meshes.new("Lipid")
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=HEAD_RADIUS * SCALE)
head_faces = len(bm.faces)

for side in (-1.0, 1.0):
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=10,
        radius1=TAIL_RADIUS * SCALE,
        radius2=TAIL_RADIUS * 0.55 * SCALE,
        depth=TAIL_LENGTH * SCALE,
        matrix=mathutils.Matrix.Translation(
            (side * TAIL_SPACING * SCALE, 0.0, -(TAIL_LENGTH * 0.5 + 1.0) * SCALE)
        ),
    )

bm.faces.ensure_lookup_table()
for position_in_mesh, face in enumerate(bm.faces):
    face.material_index = 0 if position_in_mesh < head_faces else 1
bm.to_mesh(mesh)
bm.free()
mesh.shade_smooth()

lipid = bpy.data.objects.new("Lipid", mesh)
bpy.context.scene.collection.objects.link(lipid)
lipid.hide_render = True
print(f"  lipid: {len(mesh.vertices)} vertices, {len(mesh.polygons)} faces")


# ---------------------------------------------------------------------------
heading("4. A hidden surface to open the hole against")
# ---------------------------------------------------------------------------
# Geometry Proximity measures the distance to a geometry, and the geometry
# that describes where the protein *is* is its molecular surface — not the
# cartoon, which is a ribbon threaded through the middle of it and would leave
# lipids sitting inside the helices.
footprint = load_structure("1c3w", fallback="site.pdb")
footprint.add_style("surface", selection="is_peptide", color=None)
footprint.object.matrix_world = protein.object.matrix_world.copy()
footprint.object.hide_render = True
bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
heading("5. The node tree that builds the bilayer")
# ---------------------------------------------------------------------------
tree = bpy.data.node_groups.new("GALA Bilayer", "GeometryNodeTree")
tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
tree.nodes.new("NodeGroupInput").location = (-1200, 300)
group_output = tree.nodes.new("NodeGroupOutput")
group_output.location = (900, 0)

lipid_source = tree.nodes.new("GeometryNodeObjectInfo")
lipid_source.label = "Lipid"
lipid_source.inputs["Object"].default_value = lipid
lipid_source.inputs["As Instance"].default_value = True
lipid_source.location = (-400, -320)

# The protein, as geometry rather than as an instance: Proximity has to
# measure against the surface itself.
protein_source = tree.nodes.new("GeometryNodeObjectInfo")
protein_source.label = "Protein footprint"
protein_source.inputs["Object"].default_value = footprint.object
protein_source.location = (-1200, -520)

position = tree.nodes.new("GeometryNodeInputPosition")
position.location = (-1200, -680)

proximity = tree.nodes.new("GeometryNodeProximity")
proximity.label = "Distance to the protein"
proximity.location = (-940, -560)
tree.links.new(protein_source.outputs["Geometry"], proximity.inputs["Geometry"])
tree.links.new(position.outputs["Position"], proximity.inputs["Sample Position"])

clear_of_protein = tree.nodes.new("FunctionNodeCompare")
clear_of_protein.label = "Outside the protein"
clear_of_protein.operation = "GREATER_THAN"
clear_of_protein.inputs[1].default_value = CLEARANCE * SCALE
clear_of_protein.location = (-700, -560)
tree.links.new(proximity.outputs["Distance"], clear_of_protein.inputs[0])

# Random orientation and size, keyed on the point index so each lipid keeps
# its own values from frame to frame rather than shimmering.
index = tree.nodes.new("GeometryNodeInputIndex")
index.location = (-1200, -140)

wobble = tree.nodes.new("FunctionNodeRandomValue")
wobble.label = "Orientation"
wobble.data_type = "FLOAT_VECTOR"
wobble.location = (-940, -140)
wobble.inputs[0].default_value = (-0.22, -0.22, -math.pi)
wobble.inputs[1].default_value = (0.22, 0.22, math.pi)
tree.links.new(index.outputs["Index"], wobble.inputs["ID"])

to_rotation = tree.nodes.new("FunctionNodeEulerToRotation")
to_rotation.location = (-700, -140)
tree.links.new(wobble.outputs["Value"], to_rotation.inputs["Euler"])

ripple = tree.nodes.new("FunctionNodeRandomValue")
ripple.label = "Leaflet ripple"
ripple.data_type = "FLOAT_VECTOR"
ripple.location = (-940, 120)
ripple.inputs[0].default_value = (0.0, 0.0, -1.8 * SCALE)
ripple.inputs[1].default_value = (0.0, 0.0, 1.8 * SCALE)
ripple.inputs["Seed"].default_value = 21
tree.links.new(index.outputs["Index"], ripple.inputs["ID"])

size = tree.nodes.new("FunctionNodeRandomValue")
size.label = "Size"
size.data_type = "FLOAT"
size.location = (-940, -20)
# Min and Max are always the first two sockets: the node shows only the pair
# belonging to the data type it is set to.
size.inputs[0].default_value = 0.86
size.inputs[1].default_value = 1.14
size.inputs["Seed"].default_value = 7
tree.links.new(index.outputs["Index"], size.inputs["ID"])

join = tree.nodes.new("GeometryNodeJoinGeometry")
join.location = (700, 0)
tree.links.new(join.outputs["Geometry"], group_output.inputs[0])


def leaflet(name: str, height: float, flip: bool, seed: int, y: float) -> None:
    """Scatter one leaflet's worth of lipids at ``height`` angstrom."""
    grid = tree.nodes.new("GeometryNodeMeshGrid")
    grid.label = f"{name} leaflet"
    grid.inputs["Size X"].default_value = PATCH_ACROSS * SCALE
    grid.inputs["Size Y"].default_value = PATCH_DEEP * SCALE
    grid.inputs["Vertices X"].default_value = 2
    grid.inputs["Vertices Y"].default_value = 2
    grid.location = (-400, y + 160)

    # Pushed away from the camera, so the strip runs mostly behind the
    # protein. Lipids in front of it are lipids in the way of it.
    lift = tree.nodes.new("GeometryNodeTransform")
    lift.location = (-200, y + 160)
    lift.inputs["Translation"].default_value = (
        0.0,
        (PATCH_DEEP * 0.5 - 16.0) * SCALE,
        height * SCALE,
    )
    tree.links.new(grid.outputs["Mesh"], lift.inputs["Geometry"])

    # Poisson disk, so the lipids pack at a spacing instead of clumping the way
    # uniform random points do. A bilayer that clumps reads as noise.
    scatter = tree.nodes.new("GeometryNodeDistributePointsOnFaces")
    scatter.distribute_method = "POISSON"
    scatter.label = f"{name} lipids"
    scatter.inputs["Distance Min"].default_value = LIPID_SPACING * SCALE
    scatter.inputs["Density Max"].default_value = 4.0 / (LIPID_SPACING * SCALE) ** 2
    scatter.inputs["Seed"].default_value = seed
    scatter.location = (0, y + 160)
    tree.links.new(lift.outputs["Geometry"], scatter.inputs["Mesh"])

    # A leaflet whose head groups all sit at exactly one height reads as a
    # machined part. A bilayer is a fluid; a couple of angstrom of slop is what
    # makes it look like one.
    ripples = tree.nodes.new("GeometryNodeSetPosition")
    ripples.label = f"{name} ripple"
    ripples.location = (130, y + 160)
    tree.links.new(scatter.outputs["Points"], ripples.inputs["Geometry"])
    tree.links.new(ripple.outputs["Value"], ripples.inputs["Offset"])

    instancer = tree.nodes.new("GeometryNodeInstanceOnPoints")
    instancer.label = f"{name} leaflet"
    instancer.location = (260, y)
    tree.links.new(ripples.outputs["Geometry"], instancer.inputs["Points"])
    tree.links.new(clear_of_protein.outputs["Result"], instancer.inputs["Selection"])
    tree.links.new(lipid_source.outputs["Geometry"], instancer.inputs["Instance"])
    tree.links.new(to_rotation.outputs["Rotation"], instancer.inputs["Rotation"])
    tree.links.new(size.outputs["Value"], instancer.inputs["Scale"])

    # The lower leaflet is the upper one turned over, which is what a bilayer
    # is: two sheets of lipid meeting tail to tail.
    turn = tree.nodes.new("GeometryNodeRotateInstances")
    turn.label = f"{name} leaflet facing"
    turn.location = (480, y)
    turn.inputs["Rotation"].default_value = (math.pi if flip else 0.0, 0.0, 0.0)
    tree.links.new(instancer.outputs["Instances"], turn.inputs["Instances"])
    tree.links.new(turn.outputs["Instances"], join.inputs["Geometry"])


# Head groups sit a head's radius clear of the hydrophobic core, which is what
# the core being hydrophobic means.
leaflet("Upper", mid_plane + half_thickness + HEAD_RADIUS, False, 3, 220.0)
leaflet("Lower", mid_plane - half_thickness - HEAD_RADIUS, True, 11, -220.0)

membrane = bpy.data.objects.new("Membrane", bpy.data.meshes.new("Membrane"))
bpy.context.scene.collection.objects.link(membrane)
modifier = membrane.modifiers.new("GALA Bilayer", "NODES")
modifier.node_group = tree
bpy.context.view_layer.update()

lipids = sum(
    1
    for instance in bpy.context.evaluated_depsgraph_get().object_instances
    if instance.is_instance
    and instance.parent is not None
    and instance.parent.original == membrane
)
print(f"  node group: {tree.name}, {len(tree.nodes)} nodes")
print(
    f"  {lipids} lipids instanced across a "
    f"{PATCH_ACROSS:.0f} x {PATCH_DEEP:.0f} A strip, with a hole in it"
)


# ---------------------------------------------------------------------------
heading("6. Colour and materials")
# ---------------------------------------------------------------------------
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material

# 1C3W was solved with lipids and a squalene still bound to the protein. They
# are the same molecules the bilayer around them stands for, so they are
# coloured as lipids rather than as protein — the modelled membrane and the
# crystallographic one meeting at the surface of the same molecule.
gala.color_by_selection(
    protein,
    {"all": PROTEIN, "resn LI1+SQU": HEAD, "resn RET": RETINAL},
)

gala.assign_material(
    protein,
    build_material(
        MATERIAL_PRESETS["protein"].with_(roughness=0.4, subsurface_weight=0.12),
        name="Membrane Protein",
    ),
    style="cartoon",
)
gala.assign_material(
    protein,
    build_material(
        MATERIAL_PRESETS["glass"].with_(alpha=0.22, roughness=0.12),
        name="Membrane Protein Envelope",
    ),
    style="surface",
)
gala.assign_material(
    protein,
    build_material(MATERIAL_PRESETS["ligand"], name="Retinal"),
    style="ball_and_stick",
)

# The lipid's two slots. Waxy and a little translucent: a bilayer is a fluid,
# and lipids that read as hard plastic beads make it look like a solid.
head_material = build_material(
    MATERIAL_PRESETS["lipid"].with_(
        use_attribute_color=False,
        base_color=(*gala.color.hex_to_rgb(HEAD), 1.0),
        roughness=0.5,
        subsurface_weight=0.2,
    ),
    name="Lipid Head",
)
tail_material = build_material(
    MATERIAL_PRESETS["lipid"].with_(
        use_attribute_color=False,
        base_color=(*gala.color.hex_to_rgb(TAIL), 1.0),
        roughness=0.7,
        subsurface_weight=0.25,
        alpha=0.92,
    ),
    name="Lipid Tail",
)
mesh.materials.append(head_material)
mesh.materials.append(tail_material)


# ---------------------------------------------------------------------------
heading("7. Frame it the way a membrane is drawn")
# ---------------------------------------------------------------------------
# Edge on, from just below the plane of the bilayer, so the two leaflets read
# as two leaflets and the protein is seen spanning them. Framing on the
# protein rather than the membrane: the membrane runs off both sides of the
# picture, which is the point of it.
gala.setup_render(preset=QUALITY, transparent=True)
gala.scene.render.setup_color_management()
gala.three_point_lighting(protein, energy=0.9, softness=1.4)
camera = gala.frame_target(protein, viewpoint=(22.0, 20.0), margin=1.9)
gala.depth_of_field(target=protein, fstop=5.6)
print(f"  camera at {tuple(round(v, 3) for v in camera.location)}")


# ---------------------------------------------------------------------------
heading("8. Render")
# ---------------------------------------------------------------------------
render(gala, "10_membrane")


# ---------------------------------------------------------------------------
heading("9. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# Everything worth changing is a socket in the node group: the patch size, the
# spacing, the clearance around the protein, the seed. Swap the lipid object
# for a real one imported from a force field and nothing else has to change.
save_blend("10_membrane")
