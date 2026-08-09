"""Vignette 9 — one subunit, sixty times: a capsid built by instancing.

A virus capsid is the same protein over and over. The PDB deposits one copy of
it and a list of the rotations that put the rest where they belong, and
Molecular Nodes will apply that list for you: ``add_style(..., assembly=True)``
and there is the shell.

This builds it the long way instead, with a geometry node tree of its own —
because once the instancing is yours, you can decide *which* copies to draw.
Sixty subunits with a cap lifted off is a picture of a container; sixty
subunits is a picture of a ball.

Satellite tobacco mosaic virus (1A34) is the smallest icosahedral virus there
is: a T=1 shell of sixty coat proteins about 170 A across, each one clamped
onto a piece of its own genome. Take the near cap off and the RNA is still
there, sixty fragments of it, holding the shape of a sphere with nothing
underneath.

    blender --background --python vignettes/09_capsid_assembly.py
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
import mathutils

from blender_gala.core.entity import AtomStructure
from blender_gala.core.units import world_scale_of
from blender_gala.scene.camera import VIEWPOINTS

# Which assembly. The PDB numbers them; "1" is the biological one, and for a
# virus that is the whole capsid.
ASSEMBLY = "1"

# Where the camera stands, and how much of the shell to lift off in front of
# it. The cut is a cone about the viewing direction rather than a plane: a
# plane through a sphere of subunits slices some of them in half, and a subunit
# is an object, not a volume to be cut.
VIEWPOINT = "iso"
CUT_ANGLE = 52.0

# Cool shell, warm genome. The protein is the container and reads as one
# material; the RNA is what the picture is about once the cap is off.
SHELL = "#86a3b8"
GENOME = "#f0a33c"


# ---------------------------------------------------------------------------
heading("1. The protomer, twice: once as protein, once as RNA")
# ---------------------------------------------------------------------------
# Two objects rather than one with two styles, because the two are instanced
# differently further down — the shell loses its near cap and the genome does
# not. An object is the unit geometry nodes instances, so anything that gets a
# different selection needs to be one.
#
# Neither is re-centred. The assembly transforms are expressed in the
# deposited crystal frame and rotate about its origin, so moving the atoms off
# it would scatter the sixty copies rather than close them into a shell.
shell = load_structure("1a34")
shell.add_style("surface", selection="is_peptide", color=None)

genome = load_structure("1a34")
genome.add_style("ball_and_stick", selection="is_nucleic", color=None)

gala.color_by_selection(shell, {"all": SHELL})
gala.color_by_selection(genome, {"all": GENOME})

structure = AtomStructure.from_any(shell)
print(f"  protomer : {structure.n_atoms} atoms")
print(f"  protein  : {gala.select(shell, 'protein').sum()} atoms in chain A")
print(f"  RNA      : {gala.select(genome, 'nucleic').sum()} atoms in chains B and C")


# ---------------------------------------------------------------------------
heading("2. The sixty transforms deposited with it")
# ---------------------------------------------------------------------------
# `assemblies()` is Molecular Nodes reading the assembly records out of the
# file. Each entry is a 4x4 matrix and the chains it applies to; for an
# icosahedral capsid every one of them is a pure rotation about the origin.
assemblies = shell.assemblies()
print(f"  assemblies in the file: { {k: len(v) for k, v in assemblies.items()} }")

transforms = [
    np.asarray(entry["matrix"], dtype=float) for entry in assemblies[ASSEMBLY]
]
print(f"  assembly {ASSEMBLY}: {len(transforms)} copies of the protomer")

rotations = np.array([matrix[:3, :3] for matrix in transforms])
determinants = np.linalg.det(rotations)
print(
    f"  every transform a proper rotation: "
    f"{bool(np.allclose(determinants, 1.0))} "
    f"(determinants {determinants.min():.3f} to {determinants.max():.3f})"
)


# ---------------------------------------------------------------------------
heading("3. A point per copy, carrying its rotation")
# ---------------------------------------------------------------------------
# Instance on Points needs points, so the transforms become a mesh of sixty
# vertices: the translation is the vertex position, and the rotation rides
# along as a vertex attribute. Rotations first, because the geometry nodes
# rotation socket wants Euler angles and a matrix is not one.
SCALE = world_scale_of(shell.object)
positions = np.array([matrix[:3, 3] * SCALE for matrix in transforms])
eulers = np.array(
    [
        list(mathutils.Matrix([list(row) for row in rotation]).to_euler())
        for rotation in rotations
    ]
)

# Where each copy ends up, which is what decides whether it is in the cap being
# lifted off. The protomer's centroid carried through its own transform.
centroid = structure.array.coord.mean(axis=0)
centres = np.array([rotation @ centroid for rotation in rotations]) * SCALE + positions
capsid_centre = centres.mean(axis=0)

radii = np.linalg.norm(centres - capsid_centre, axis=1)
print(
    f"  subunit centres lie {radii.min() / SCALE:.1f} to "
    f"{radii.max() / SCALE:.1f} A from the middle — a shell, not a ball"
)

# The direction the camera will look from, computed the same way
# `frame_target` computes it, so the hole is where the camera is.
azimuth, elevation = (math.radians(angle) for angle in VIEWPOINTS[VIEWPOINT])
towards_camera = np.array(
    [
        math.cos(elevation) * math.sin(azimuth),
        -math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
    ]
)
outward = (centres - capsid_centre) / np.linalg.norm(
    centres - capsid_centre, axis=1, keepdims=True
)
keep = np.degrees(np.arccos(np.clip(outward @ towards_camera, -1.0, 1.0))) > CUT_ANGLE
print(f"  keeping {int(keep.sum())} of {len(transforms)} subunits in the shell")

points = bpy.data.meshes.new("Capsid Points")
points.from_pydata([tuple(p) for p in positions], [], [])
points.update()

euler_attribute = points.attributes.new("euler", "FLOAT_VECTOR", "POINT")
euler_attribute.data.foreach_set("vector", eulers.ravel().tolist())
keep_attribute = points.attributes.new("keep", "BOOLEAN", "POINT")
keep_attribute.data.foreach_set("value", keep.tolist())

capsid = bpy.data.objects.new("STMV Capsid", points)
bpy.context.scene.collection.objects.link(capsid)


# ---------------------------------------------------------------------------
heading("4. The node tree that puts one thing in sixty places")
# ---------------------------------------------------------------------------
# Ten nodes, of which two do the work. Object Info reads a molecule *as it is
# drawn* — the styled surface Molecular Nodes' own modifier produced, not the
# atom positions underneath — and hands it over as an instance, which is why
# sixty copies of a 2268-atom surface cost one surface and sixty matrices.
# Instance on Points puts one at each vertex.
tree = bpy.data.node_groups.new("GALA Capsid Assembly", "GeometryNodeTree")
tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

group_input = tree.nodes.new("NodeGroupInput")
group_input.location = (-600, 0)
group_output = tree.nodes.new("NodeGroupOutput")
group_output.location = (600, 0)

euler_input = tree.nodes.new("GeometryNodeInputNamedAttribute")
euler_input.data_type = "FLOAT_VECTOR"
euler_input.inputs["Name"].default_value = "euler"
euler_input.location = (-600, -220)

to_rotation = tree.nodes.new("FunctionNodeEulerToRotation")
to_rotation.location = (-380, -220)
tree.links.new(euler_input.outputs["Attribute"], to_rotation.inputs["Euler"])

keep_input = tree.nodes.new("GeometryNodeInputNamedAttribute")
keep_input.data_type = "BOOLEAN"
keep_input.inputs["Name"].default_value = "keep"
keep_input.location = (-600, -400)

join = tree.nodes.new("GeometryNodeJoinGeometry")
join.location = (380, 0)
tree.links.new(join.outputs["Geometry"], group_output.inputs[0])


def instance_branch(name: str, source, y: float, selection=None):
    """Wire one Object Info into one Instance on Points, and join it in."""
    info = tree.nodes.new("GeometryNodeObjectInfo")
    info.name = f"GALA {name} Source"
    info.label = name
    info.inputs["Object"].default_value = source.object
    info.inputs["As Instance"].default_value = True
    info.location = (-180, y - 180)

    instancer = tree.nodes.new("GeometryNodeInstanceOnPoints")
    instancer.name = f"GALA {name} Instances"
    instancer.label = name
    instancer.location = (140, y)
    tree.links.new(group_input.outputs[0], instancer.inputs["Points"])
    tree.links.new(info.outputs["Geometry"], instancer.inputs["Instance"])
    tree.links.new(to_rotation.outputs["Rotation"], instancer.inputs["Rotation"])
    if selection is not None:
        tree.links.new(selection, instancer.inputs["Selection"])
    tree.links.new(instancer.outputs["Instances"], join.inputs["Geometry"])
    return instancer


# The shell is drawn only where `keep` is true, so the cap in front of the
# camera is simply never instanced. The genome has no selection wired at all,
# which is what leaves it complete inside the opening.
instance_branch("Shell", shell, 160.0, selection=keep_input.outputs["Attribute"])
instance_branch("Genome", genome, -160.0)

modifier = capsid.modifiers.new("GALA Assembly", "NODES")
modifier.node_group = tree

# The two protomers are the source geometry, not part of the picture: leaving
# them visible would put an extra, unrotated copy of each in the middle of the
# shell. Object Info reads them regardless of whether they render.
for molecule in (shell, genome):
    molecule.object.hide_render = True

print(f"  node group : {tree.name} with {len(tree.nodes)} nodes")
for node in sorted(tree.nodes, key=lambda n: n.location[0]):
    print(f"      {node.bl_idname:38s} {node.label or node.name}")


# ---------------------------------------------------------------------------
heading("5. Gala frames and lights what geometry nodes drew")
# ---------------------------------------------------------------------------
# The mesh on disk is sixty vertices, all of them within a few angstroms of
# the origin — a lattice of translations for a shell that has none. What the
# camera has to fit is the 170 A of instanced surface those points stand for,
# so the framing and the light rig read the object as evaluated, instances
# included, rather than as stored.
gala.setup_render(preset=QUALITY, transparent=True)
gala.scene.render.setup_color_management()

rig = gala.three_point_lighting(capsid, energy=1.0, softness=1.3)
camera = gala.frame_target(capsid, viewpoint=VIEWPOINT, margin=1.06)

print(f"  camera at {tuple(round(v, 3) for v in camera.location)}")
print(f"  light rig : {len(rig.children)} lamps sized to the capsid, not the points")

# A fourth lamp, low and in front, aimed into the hole. Three-point lighting is
# built for a solid object lit from outside; a container that has been opened
# has an inside, and nothing in the standard rig reaches it.
inside = bpy.data.lights.new("GALA Interior Fill", "AREA")
inside.energy = float(rig.children[0].data.energy) * 0.5
inside.size = float(rig.children[0].data.size) * 1.5
interior = bpy.data.objects.new("GALA Interior Fill", inside)
bpy.context.scene.collection.objects.link(interior)
distance = float(np.linalg.norm(np.array(camera.location) - capsid_centre))
interior.location = tuple(capsid_centre + towards_camera * distance * 0.55)
interior.rotation_euler = (
    mathutils.Vector((-towards_camera).tolist()).to_track_quat("-Z", "Y").to_euler()
)
interior.parent = rig


# ---------------------------------------------------------------------------
heading("6. Materials")
# ---------------------------------------------------------------------------
# Assigned to the two source molecules, because that is where the geometry
# comes from: an instance carries the material of the thing being instanced.
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material

gala.assign_material(
    shell,
    build_material(
        MATERIAL_PRESETS["protein"].with_(
            roughness=0.55, ao_strength=0.4, ao_distance=0.08
        ),
        name="Capsid Shell",
    ),
    style="surface",
)
gala.assign_material(
    genome,
    build_material(
        MATERIAL_PRESETS["nucleic"].with_(roughness=0.35, subsurface_weight=0.15),
        name="Capsid Genome",
    ),
    style="ball_and_stick",
)


# ---------------------------------------------------------------------------
heading("7. Render")
# ---------------------------------------------------------------------------
render(gala, "09_capsid_assembly")
print("\n  Sixty subunits, one surface mesh, and a hole where the camera is.")


# ---------------------------------------------------------------------------
heading("8. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# The interesting knob is `keep`: it is an ordinary boolean attribute on an
# ordinary mesh, so anything that can write one — a Set Selection node, a
# proximity test, a driver on the frame number — decides how much of the shell
# is there. Animate it and the capsid opens.
save_blend("09_capsid_assembly")
