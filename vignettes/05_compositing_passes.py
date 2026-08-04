"""Vignette 5 — cryptomatte, Z depth and compositing.

Setting these up before rendering is what lets you change your mind afterwards:
brighten just the ligand, or pull it forward with depth of field, without
re-rendering.

    blender --background --python vignettes/05_compositing_passes.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import DATA_DIR, IMAGE_DIR, QUALITY, heading, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. A scene to composite")
# ---------------------------------------------------------------------------
mol = mn.Molecule.load(os.path.join(DATA_DIR, "site.pdb"))
mol.add_style("ball_and_stick")

gala.publication_setup(
    mol,
    preset=QUALITY,
    lighting_style="three_point",
    viewpoint="iso",
    cryptomatte=False,  # done explicitly below, to show the pieces
)


# ---------------------------------------------------------------------------
heading("2. Enable the passes")
# ---------------------------------------------------------------------------
import bpy

view_layer = bpy.context.scene.view_layers[0]
enabled = gala.enable_passes(
    cryptomatte=True,
    depth=True,
    normal=True,
    cryptomatte_levels=6,  # enough for the soft edges of a molecular surface
    view_layer=view_layer,
)
print(f"  enabled: {', '.join(enabled)}")
print(f"  cryptomatte layers: {gala.scene.compositing.cryptomatte_layers(view_layer)}")


# ---------------------------------------------------------------------------
heading("3. Build the compositing chain")
# ---------------------------------------------------------------------------
exr_dir = os.path.join(IMAGE_DIR, "passes")
tree = gala.setup_compositor(
    denoise=True,
    cryptomatte=True,
    exposure=0.0,
    contrast=0.0,
    file_output=exr_dir,
)

gala_nodes = [n for n in tree.nodes if n.name.startswith("GALA ")]
print(f"  {len(gala_nodes)} Gala nodes:")
for node in sorted(gala_nodes, key=lambda n: n.name):
    print(f"      {node.name}")

# Re-running rewires rather than duplicating.
before = len(tree.nodes)
gala.setup_compositor(cryptomatte=True, file_output=exr_dir)
print(f"\n  idempotent: {before} nodes before, {len(tree.nodes)} after re-running")

print("\n  The Cryptomatte nodes are deliberately left unconnected: linking one")
print("  to the output would matte the beauty pass, which is the opposite of")
print("  the point. They are there to pick with, and the mattes are written")
print("  into the multilayer EXR.")


# ---------------------------------------------------------------------------
heading("4. Depth of field")
# ---------------------------------------------------------------------------
# Physical camera DOF: accurate, costs samples, focus follows the object.
camera_data = gala.depth_of_field(mol, fstop=2.8)
print(f"  camera DOF on, f/{camera_data.dof.aperture_fstop}")
print(f"  focused on: {camera_data.dof.focus_object.name}")
print("\n  One angstrom is 0.01 Blender units, so depth of field at molecular")
print("  scale is very shallow. Check the result rather than trusting the number.")


# ---------------------------------------------------------------------------
heading("5. Depth cueing")
# ---------------------------------------------------------------------------
# The classic way of keeping a crowded site readable: fade with depth.
gala.depth_cue(near=0.0, far=40.0)  # angstrom from the camera
print("  depth cue over 0-40 A")

# That rebuilt the compositor, so re-attach the EXR output.
gala.setup_compositor(
    cryptomatte=True, depth_cue_range=(0.0, 40.0), file_output=exr_dir
)


# ---------------------------------------------------------------------------
heading("6. Render, keeping every pass")
# ---------------------------------------------------------------------------
render(gala, "05_compositing_passes")
print(f"\n  multilayer EXR passes written under {exr_dir}")
print("  Those carry Image, Depth, Normal and every cryptomatte layer — the")
print("  format Nuke, Fusion, Krita and Blender's own compositor read.")

# The alternative: make the render output itself a multilayer EXR, which
# captures every enabled pass with no File Output node at all.
#
#     gala.scene.set_exr_output("render/figure.exr")
#     gala.render()
