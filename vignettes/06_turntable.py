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

from _common import IMAGE_DIR, QUALITY, heading, load_structure, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. Set up the scene")
# ---------------------------------------------------------------------------
mol = load_structure("1ake")
mol.add_style("cartoon")

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
heading("2. Build the orbit")
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
heading("3. The lighting rig does not orbit with the camera")
# ---------------------------------------------------------------------------
# Lights stay put, so the molecule turns through the light rather than being
# lit identically from every angle — which is what makes the shape read.
rig = bpy.data.objects.get("GALA Light Rig")
print(f"  light rig parent: {rig.parent}")
print("  Rotate the rig itself to re-light the whole scene at once:")
rig.rotation_euler.z = math.radians(20)


# ---------------------------------------------------------------------------
heading("4. Check a few frames")
# ---------------------------------------------------------------------------
# Rendering 120 frames in CI would be wasteful, so sample a few stills.
os.makedirs(IMAGE_DIR, exist_ok=True)
for frame in (1, FRAMES // 4, FRAMES // 2):
    scene.frame_set(frame)
    location = tuple(round(v, 3) for v in scene.camera.matrix_world.translation)
    print(f"  frame {frame:3d}: camera at {location}")

scene.frame_set(1)
render(gala, "06_turntable")

print("\n  To render the whole animation:")
print("      gala.scene.render('frames/turntable_', animation=True)")
