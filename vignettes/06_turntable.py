"""Vignette 6 — a turntable animation.

Orbiting a molecule is the standard way to show a structure in a talk. The
camera is parented to a pivot at the molecule's centre, so the framing computed
once holds for every frame.

    blender --background --python vignettes/06_turntable.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import QUALITY, heading, load_structure, save_blend, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. Set up the scene")
# ---------------------------------------------------------------------------
# Triosephosphate isomerase, the textbook obligate dimer. 1AKE, which the
# other vignettes use, has two chains in its asymmetric unit but no interface
# worth the name — they come within 3.35 A of each other at exactly one point,
# so there would be a single contact to draw across it.
mol = load_structure("1tim")

from blender_gala.core.entity import AtomStructure

structure = AtomStructure.from_any(mol)

# Whichever two chains this structure has, rather than a hard-coded A and B:
# with no network the fallback fixture has one chain, and this then draws both
# treatments on it instead of styling nothing and rendering an empty frame.
chain_ids = sorted(set(structure.context.upper("chain_id")))
CHAIN_A, CHAIN_B = (f"chain {name}" for name in (chain_ids * 2)[:2])
print(f"  chains: {chain_ids}")

# One chain as a surface for shape, the other as ball-and-stick for detail. A
# style's `selection` names a boolean attribute, so each chain is selected here
# and stored under a name geometry nodes can read. `color=None` keeps Molecular
# Nodes from wiring a colour node that would paint over the scheme below.
for name, selection in (("chain_a", CHAIN_A), ("chain_b", CHAIN_B)):
    attribute = mol.object.data.attributes.new(name, "BOOLEAN", "POINT")
    attribute.data.foreach_set(
        "value", gala.select(mol, selection).astype(bool).tolist()
    )

mol.add_style("surface", selection="chain_a", color=None)
mol.add_style("ball_and_stick", selection="chain_b", color=None)

# A neutral cool surface behind warm, bright sticks: the eye goes to the chain
# drawn atom by atom, and the surface reads as the thing it packs against.
# Heteroatoms stay CPK, since that is what the contacts land on.
COOL, WARM = "#8fa3b3", "#ffb03a"
gala.color_by_selection(
    mol,
    {
        "all": COOL,
        f"({CHAIN_B}) and elem C": WARM,
        f"({CHAIN_B}) and elem N": "#3050f8",
        f"({CHAIN_B}) and elem O": "#ff2010",
        f"({CHAIN_B}) and elem S": "#ffd030",
    },
)

gala.publication_setup(
    mol,
    preset=QUALITY,
    lighting_style="three_point",
    material_scheme="chemistry",
    origin_method="centroid",
    move_to_world_origin=True,  # the orbit pivots on the molecule
    viewpoint="iso",
)


# ---------------------------------------------------------------------------
heading("2. What the chains hold each other with")
# ---------------------------------------------------------------------------
# Across the interface, not within a chain. 1AKE is a crystal structure with no
# hydrogens, so `hbond` finds nothing and `polar` is the right criterion — the
# heavy-atom donor/acceptor test crystallographers use, and what PyMOL calls
# polar_contacts. Asking for both says which one answered.
interface = gala.find_interactions(mol, CHAIN_A, CHAIN_B, kinds=["hbond", "polar"])
kinds: dict[str, int] = {}
for contact in interface:
    kinds[contact.kind] = kinds.get(contact.kind, 0) + 1
print(f"  {len(interface)} contacts across the interface: {kinds or 'none'}")

# No distance labels: on something that turns, a dozen numbers orbiting with it
# is noise rather than information.
gala.draw_interactions(interface, target=mol, label=False)


# ---------------------------------------------------------------------------
heading("3. Build the orbit")
# ---------------------------------------------------------------------------
FRAMES = 120
pivot = gala.orbit(frames=FRAMES, target=mol)

import bpy

scene = bpy.context.scene
print(f"  pivot        : {pivot.name} at {tuple(round(v, 4) for v in pivot.location)}")
print(f"  camera parent: {scene.camera.parent.name}")
print(f"  frame range  : {scene.frame_start} to {scene.frame_end}")

# Linear interpolation, so the rotation is at constant speed rather than
# easing in and out at the loop point.
# Blender 5 moved F-curves into slotted actions; Gala has a shim for both.
from blender_gala.scene.camera import _action_fcurves

action = pivot.animation_data.action
interpolations = {
    keyframe.interpolation
    for fcurve in _action_fcurves(action)
    for keyframe in fcurve.keyframe_points
}
print(f"  interpolation: {interpolations}")


# ---------------------------------------------------------------------------
heading("4. The lighting rig does not orbit with the camera")
# ---------------------------------------------------------------------------
# Lights stay put, so the molecule turns through the light rather than being
# lit identically from every angle — which is what makes the shape read.
rig = bpy.data.objects.get("GALA Light Rig")
print(f"  light rig parent: {rig.parent}")
print("  Rotate the rig itself to re-light the whole scene at once:")
rig.rotation_euler.z = math.radians(20)


# ---------------------------------------------------------------------------
heading("5. Check a few frames")
# ---------------------------------------------------------------------------
# Where the camera is at a few points around the turn, which is what there is
# to check about an orbit. No still is rendered: the figure this vignette
# contributes is the animation below, and a frozen frame of a turntable was
# only ever a stand-in for it.
for frame in (1, FRAMES // 4, FRAMES // 2):
    scene.frame_set(frame)
    location = tuple(round(v, 3) for v in scene.camera.matrix_world.translation)
    print(f"  frame {frame:3d}: camera at {location}")

scene.frame_set(1)


# ---------------------------------------------------------------------------
# The whole turn, when asked for it
# ---------------------------------------------------------------------------
# Off unless GALA_TURNTABLE_DIR names somewhere to put the frames, because CI
# runs every vignette on every push and a hundred-odd Cycles frames is not what
# a smoke test is for. `make vignettes-turntable` sets it and builds the animation.
#
# The orbit puts 0 degrees on frame 1 and 360 on frame FRAMES + 1, so the
# whole of 1..FRAMES comes back round to where it started without repeating a
# frame. STEP thins that out when a smaller file matters more than smoothness.
frames_dir = os.environ.get("GALA_TURNTABLE_DIR")
if frames_dir:
    heading("6. Render the turn")
    # Every second frame: 60 of them at 25 fps is a two and a half second
    # turn, which is motion rather than a slideshow, and half the file of
    # rendering all 120.
    STEP = int(os.environ.get("GALA_TURNTABLE_STEP", "2"))
    SIZE = int(os.environ.get("GALA_TURNTABLE_SIZE", "480"))
    scene.render.resolution_x = scene.render.resolution_y = SIZE

    os.makedirs(frames_dir, exist_ok=True)
    wanted = range(1, FRAMES + 1, STEP)
    for index, frame in enumerate(wanted):
        scene.frame_set(frame)
        gala.render(os.path.join(frames_dir, f"{index:03d}.png"))
    print(f"  {len(wanted)} frames at {SIZE}x{SIZE} in {frames_dir}")


# ---------------------------------------------------------------------------
heading("7. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# The orbit is keyframed on the pivot rather than baked into the frames, so
# this is the file to open when you want the turn to be slower, longer, or
# about a different axis. Press space in the viewport to watch it.
save_blend("06_turntable")
