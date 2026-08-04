"""Vignette 1 — from a freshly imported structure to a publication figure.

The shortest path through Objective 1: load a molecule, style it, and let
``publication_setup`` do everything else.

    blender --background --python vignettes/01_publication_figure.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import QUALITY, heading, load_structure, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. Load and style the molecule")
# ---------------------------------------------------------------------------
# Molecular Nodes owns import and styling. Gala never duplicates that.
mol = load_structure("1ake")
mol.add_style("cartoon", selection="polymer")
mol.add_style("ball_and_stick", selection="not polymer")

print(f"  chains  : {sorted(set(mol.array.chain_id))}")
print(
    f"  residues: {len(set(zip(mol.array.chain_id, mol.array.res_id, strict=False)))}"
)


# ---------------------------------------------------------------------------
heading("2. One call for the whole scene")
# ---------------------------------------------------------------------------
# Order matters inside publication_setup: the origin is fixed first, so the
# lighting rig and camera — both sized from the bounding sphere — are computed
# against final coordinates.
report = gala.publication_setup(
    mol,
    preset=QUALITY,
    lighting_style="three_point",
    material_scheme="chemistry",
    origin_method="centroid",
    viewpoint="iso",
    transparent=True,
    cryptomatte=True,
)
print(report)


# ---------------------------------------------------------------------------
heading("3. What that produced")
# ---------------------------------------------------------------------------
import bpy

scene = bpy.context.scene
print(f"  engine        : {scene.render.engine}")
print(f"  resolution    : {scene.render.resolution_x} x {scene.render.resolution_y}")
print(f"  samples       : {scene.cycles.samples}")
print(f"  transparent   : {scene.render.film_transparent}")
print(f"  view transform: {scene.view_settings.view_transform}")
print(f"  camera        : {scene.camera.name if scene.camera else None}")

rig = bpy.data.objects.get("GALA Light Rig")
if rig is not None:
    print(f"  light rig     : {rig.name} with {len(rig.children)} lights")
    for light in rig.children:
        print(
            f"      {light.name:12s} {light.data.energy:8.1f} W  size {light.data.size:.3f}"
        )


# ---------------------------------------------------------------------------
heading("4. Adjusting individual steps")
# ---------------------------------------------------------------------------
# Every step is separately callable, so nothing about the one-call form is
# a dead end.

# Softer, warmer light from a different angle.
gala.three_point_lighting(mol, energy=1.2, softness=1.6, rotation=30.0)

# A matte protein with ambient occlusion in the crevices.
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material

matte = build_material(
    MATERIAL_PRESETS["protein"].with_(roughness=0.65, ao_strength=0.35),
    name="Vignette Protein",
)
gala.assign_material(mol, matte, style="cartoon")

# Frame it head-on instead.
gala.frame_target(mol, viewpoint="front", margin=1.2)


# ---------------------------------------------------------------------------
heading("5. Render")
# ---------------------------------------------------------------------------
render(gala, "01_publication_figure")
print("\n  The background is alpha, so this drops onto any page colour.")
