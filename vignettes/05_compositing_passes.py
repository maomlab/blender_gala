"""Vignette 5 — compositing: one render, several figures.

A talk needs the same picture of haemoglobin three times: once plain, once with
the alpha subunits carrying the argument, once with the beta subunits. Rendering
it three times is three chances for the lighting, the framing or the colours to
drift apart, and three times the wait.

Cryptomatte is how you avoid that. Rendering *once* with the passes on writes,
alongside the picture, a per-pixel record of which material each pixel came
from; the compositor can then re-select any of them afterwards. This vignette
renders haemoglobin once and cuts four figures out of that single render, three
of them without touching Cycles again.

    blender --background --python vignettes/05_compositing_passes.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import IMAGE_DIR, QUALITY, heading, load_structure, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. A scene worth compositing")
# ---------------------------------------------------------------------------
# 2HHB is deoxyhaemoglobin: two alpha subunits, two beta subunits and four
# hemes. Which pair the figure is about depends on the slide, which is exactly
# the situation cryptomatte is for.
mol = load_structure("2hhb")

CHAINS = {"A": "alpha 1", "B": "beta 1", "C": "alpha 2", "D": "beta 2"}

# Two blues for the alpha subunits, two warms for the beta ones: related within
# a pair, opposed across it, so the tetramer reads as 2+2 before anything is
# highlighted at all.
COLOURS = {
    "A": "#4e9fd1",
    "C": "#8ecae6",
    "B": "#e2794a",
    "D": "#f2a978",
    "heme": "#c0392b",
}


# ---------------------------------------------------------------------------
heading("2. One material per chain — what cryptomatte separates by")
# ---------------------------------------------------------------------------
# Molecular Nodes puts a whole structure in a single object, so the *object*
# cryptomatte layer cannot tell one chain from another: they are all the same
# object. The material layer can, provided each chain is drawn by its own style
# with its own material. That is the one preparation step the payoff depends on.
#
# Styles select by named attribute rather than by Gala's selection language, so
# each chain's mask is computed once with `select` and stored on the mesh for
# the style to read.
import databpy

materials = {}
for chain, label in CHAINS.items():
    mask = gala.select(mol, f"chain {chain}")
    attribute = f"chain_{chain}"
    mol.store_named_attribute(
        mask,
        name=attribute,
        atype=databpy.AttributeTypes.BOOLEAN,
        domain=databpy.AttributeDomains.POINT,
    )
    material = gala.scene.materials.build_material("protein", name=f"GALA {label}")
    mol.add_style("cartoon", selection=attribute, material=material)
    materials[chain] = material.name
    print(f"  chain {chain} ({label}): {int(mask.sum())} atoms -> {material.name}")

# The hemes get a material of their own too, so "just the hemes" is also a matte
# you can pull later. The fixture used offline has no HEM, so fall back to
# whatever it calls its ligand.
heme = gala.select(mol, "resn HEM")
HEME = "resn HEM"
if not heme.any():
    HEME = "ligand"
    heme = gala.select(mol, HEME)
mol.store_named_attribute(
    heme,
    name="heme",
    atype=databpy.AttributeTypes.BOOLEAN,
    domain=databpy.AttributeDomains.POINT,
)
heme_material = gala.scene.materials.build_material("ligand", name="GALA heme")
mol.add_style("ball_and_stick", selection="heme", material=heme_material)
print(f"  hemes ({HEME}): {int(heme.sum())} atoms -> {heme_material.name}")

result = gala.color_by_selection(
    mol,
    {
        **{f"chain {chain}": COLOURS[chain] for chain in CHAINS},
        HEME: COLOURS["heme"],
    },
)
print(f"  coloured {result.n_colored} atoms")

# material_scheme=None because the materials are already assigned, one per
# chain, and the chemistry scheme would put a single protein material back
# across all of them — which would collapse the four mattes into one.
gala.publication_setup(
    mol,
    preset=QUALITY,
    lighting_style="three_point",
    viewpoint="iso",
    material_scheme=None,
    cryptomatte=False,  # done explicitly below, to show the pieces
)


# ---------------------------------------------------------------------------
heading("3. Enable the passes")
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
heading("4. Build the compositing chain")
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
heading("5. Render once, keeping every pass")
# ---------------------------------------------------------------------------
render(gala, "05_compositing_beauty")

exr = os.path.join(exr_dir, "gala.exr")
print(f"\n  passes written to {exr} ({os.path.getsize(exr) / 1e6:.1f} MB)")
print("  Image, Depth, Normal and every cryptomatte layer, in the multilayer")
print("  EXR that Nuke, Fusion, Krita and Blender's own compositor read.")

# The alternative: make the render output itself a multilayer EXR, which
# captures every enabled pass with no File Output node at all.
#
#     gala.scene.set_exr_output("render/figure.exr")
#     gala.render()


# ---------------------------------------------------------------------------
heading("6. Cut three more figures out of it, without rendering again")
# ---------------------------------------------------------------------------
# The scene this composites in has no molecule, no lights and no camera in it.
# Every pixel comes out of the EXR, which is the whole claim: past this point
# Cycles has nothing left to do, and each figure below costs a fraction of a
# second rather than another full render.
composite = bpy.data.scenes.new("GALA Composite")
composite.render.resolution_x = bpy.context.scene.render.resolution_x
composite.render.resolution_y = bpy.context.scene.render.resolution_y
composite.render.film_transparent = True
# Cycles at one sample rather than the default engine: there is nothing in the
# scene to trace, and Cycles is the engine that renders the same way on a
# headless machine as on a workstation. Standard view transform and RGBA to
# match what the beauty pass was written with.
composite.render.engine = "CYCLES"
composite.cycles.samples = 1
composite.view_settings.view_transform = "Standard"
gala.scene.render.set_image_format(
    composite.render.image_settings, "PNG", color_mode="RGBA"
)

FIGURES = {
    "05_compositing_passes": (["A", "C"], "the alpha subunits"),
    "05_compositing_beta": (["B", "D"], "the beta subunits"),
    "05_compositing_heme": (["heme"], "the four hemes"),
}

for name, (keep, description) in FIGURES.items():
    mattes = [heme_material.name if k == "heme" else materials[k] for k in keep]
    gala.highlight_matte(mattes, layer="material", source=exr, scene=composite)
    render(gala, name, scene=composite)
    print(f"      {description}: {', '.join(mattes)}")

print("\n  Same render, same lighting, same framing — only the emphasis moves.")
print("  A Cryptomatte node's picker writes the same names when you click a")
print("  chain in the render, so this is scriptable and clickable alike.")


# ---------------------------------------------------------------------------
heading("7. Depth of field")
# ---------------------------------------------------------------------------
# Physical camera DOF: accurate, costs samples, focus follows the object.
camera_data = gala.depth_of_field(mol, selection=HEME, fstop=2.8)
print(f"  camera DOF on, f/{camera_data.dof.aperture_fstop}")
print(f"  focused on: {camera_data.dof.focus_object.name}")
print("\n  One angstrom is 0.01 Blender units, so depth of field at molecular")
print("  scale is very shallow. Check the result rather than trusting the number.")
gala.depth_of_field(mol, enable=False)


# ---------------------------------------------------------------------------
heading("8. Depth cueing")
# ---------------------------------------------------------------------------
# The classic way of keeping a crowded scene readable: fade with depth. Unlike
# the highlights above this one does need the Z pass at render time, so it is
# the one variant here that is rendered rather than recomposited.
#
# The range is measured in angstrom *from the camera*, so it has to bracket
# where the molecule actually sits — a few hundred angstrom away at this
# framing. A range that stops short of it fades the whole frame into the
# background colour, which looks like a broken render rather than a cue.
import numpy as np

from blender_gala.core import units
from blender_gala.core.entity import AtomStructure

centre, radius = AtomStructure.from_any(mol).bounding_sphere()
distance = float(np.linalg.norm(np.asarray(bpy.context.scene.camera.location) - centre))
near = units.bu_to_angstrom(distance - radius)
far = units.bu_to_angstrom(distance + radius)

gala.depth_cue(near=near, far=far)
print(f"  depth cue over {near:.0f}-{far:.0f} A from the camera")
print("  Fading towards the world colour fills the background with it too, so")
print("  a depth-cued figure is opaque where an untouched one is transparent.")
render(gala, "05_compositing_depth_cue")
