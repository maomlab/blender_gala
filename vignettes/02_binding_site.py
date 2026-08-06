"""Vignette 2 — a ligand binding site, analysed and annotated.

The full Objective 2 workflow: find every interaction between a ligand and its
pocket, draw them, and label the residues that make them.

    blender --background --python vignettes/02_binding_site.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. Load a ligand complex")
# ---------------------------------------------------------------------------
# 1STP is streptavidin with biotin bound: a small, well-behaved site.
mol = load_structure("1stp")

n_ligand = gala.select(mol, "ligand").sum()
if n_ligand == 0:
    # The fallback fixture names its ligand LIG.
    LIGAND = "resn LIG"
    n_ligand = gala.select(mol, LIGAND).sum()
else:
    LIGAND = "ligand"

POCKET = f"byres (protein within 4.5 of ({LIGAND}))"
print(f"  ligand atoms  : {n_ligand}")
print(f"  pocket atoms  : {gala.select(mol, POCKET).sum()}")
from blender_gala.core.entity import AtomStructure

structure = AtomStructure.from_any(mol)
pocket_residues = set(structure.context.residue_key[structure.select(POCKET)])
print(f"  pocket residues: {len(pocket_residues)}")


# ---------------------------------------------------------------------------
heading("2. Style: cartoon protein, ball-and-stick ligand and pocket")
# ---------------------------------------------------------------------------
mol.add_style("cartoon", selection="is_peptide")
mol.add_style("ball_and_stick", selection="is_hetero")

# A cool neutral protein and a warm ligand, so the eye goes to the ligand
# without anything having to point at it. Complementary hues an octave apart in
# brightness do that work by themselves; Molecular Nodes' default gives protein
# and ligand the same pink and leaves them competing.
#
# Ordered general to specific, because later entries win where selections
# overlap: everything cool, then the ligand's carbons warm, then its
# heteroatoms back to the CPK colours a chemist reads without a legend.
SLATE = "#93a6b8"
AMBER = "#ffaf3a"

gala.color_by_selection(
    mol,
    {
        "all": SLATE,
        f"({LIGAND}) and elem C": AMBER,
        f"({LIGAND}) and elem N": "#3050f8",
        f"({LIGAND}) and elem O": "#ff2010",
        f"({LIGAND}) and elem S": "#ffd030",
    },
)

gala.publication_setup(
    mol,
    preset=QUALITY,
    lighting_style="both",
    hdri="studio",
    material_scheme="chemistry",
    viewpoint="iso",
)


# ---------------------------------------------------------------------------
heading("3. What does the ligand touch?")
# ---------------------------------------------------------------------------
# Interactions are always found between two selections, which is how one
# actually thinks about them.
contacts = gala.find_interactions(
    mol,
    LIGAND,
    "protein",
    kinds=["hbond", "polar", "salt_bridge", "hydrophobic", "pi_stacking", "cation_pi"],
)

by_kind: dict[str, int] = {}
for contact in contacts:
    by_kind[contact.kind] = by_kind.get(contact.kind, 0) + 1

print(f"  {len(contacts)} interactions")
for kind, count in sorted(by_kind.items()):
    print(f"      {kind:14s} {count}")

print("\n  The closest few:")
for contact in sorted(contacts, key=lambda c: c.distance)[:6]:
    print(f"      {contact}")


# ---------------------------------------------------------------------------
heading("4. Draw them")
# ---------------------------------------------------------------------------
# Real curve objects, not overlays: they light, shadow and occlude correctly.
#
# Distances are quoted for the polar contacts only. It is what a figure legend
# would report, and twenty numbers stacked over one binding site is not a
# figure anybody can read.
POLAR_KINDS = {"hbond", "polar", "salt_bridge"}

drawn = gala.draw_interactions(
    [c for c in contacts if c.kind not in POLAR_KINDS],
    target=mol,
    label=False,
)
drawn += gala.draw_interactions(
    [c for c in contacts if c.kind in POLAR_KINDS],
    target=mol,
    label=True,
    label_template="{distance:.1f}",
)
print(f"  created {len(drawn)} objects in the Gala Interactions collection")

# Restyle one kind. Distances are in angstrom throughout.
from blender_gala import InteractionStyle

gala.clear_interactions("hbond")
gala.draw_interactions(
    [c for c in contacts if c.kind == "hbond"],
    target=mol,
    styles={
        "hbond": InteractionStyle(
            colour=(1.0, 0.95, 0.35), radius=0.14, dash_length=0.45, gap_length=0.3
        )
    },
    label=True,
)


# ---------------------------------------------------------------------------
heading("5. Label the pocket")
# ---------------------------------------------------------------------------
# Only the residues in closest contact, and sized for the close-up below
# rather than for the whole protein. Naming all sixteen pocket residues at this
# range stacks the cards on top of each other and on the site they describe.
CLOSEST = f"byres (protein within 3.4 of ({LIGAND}))"

labels = gala.label(
    mol,
    CLOSEST,
    template="{one}{resi}",
    level="residue",
    anchor="ca",
    style="card",  # a translucent plane keeps text legible over the surface
    size=0.9,
    offset=2.0,
    billboard=True,
)
print(f"  labelled {len([o for o in labels if o.type == 'FONT'])} residues")

# A caption that can never be occluded, drawn in screen space.
gala.label_hud(mol, "Ligand binding site", location=(0.04, 0.95), size=28)


# ---------------------------------------------------------------------------
heading("6. Frame the site rather than the whole protein")
# ---------------------------------------------------------------------------
# Frame the pocket, not the protein: the rest stays in the scene and is simply
# allowed out of shot. Framing `mol` here — which this used to do, despite the
# heading — put the whole 917-atom protein in the frame and left the site a
# knot in the middle of it.

# Look into the pocket rather than from a fixed compass point. The direction
# out of the protein through the ligand is where the site opens, so that is
# where the camera belongs — derived from this structure rather than
# hand-tuned, so it still points at the site for a different complex.
import math

positions = structure.world_positions()
outward = positions[structure.select(LIGAND)].mean(axis=0) - positions[
    structure.select("protein")
].mean(axis=0)
outward /= np.linalg.norm(outward)
viewpoint = (
    math.degrees(math.atan2(outward[0], -outward[1])),
    math.degrees(math.asin(float(np.clip(outward[2], -1.0, 1.0)))),
)
print(f"  looking in along ({viewpoint[0]:.0f} deg, {viewpoint[1]:.0f} deg)")

# Framed on the ligand with room around it, rather than on the pocket. A
# `byres` selection pulls in whole residues, so "the pocket" reaches as far as
# the backbone of anything with a side chain pointing in — a much wider shot
# than the site itself. The margin is what sets how much context comes with
# it, and it has to clear the labels, which sit outside the atoms they name.
gala.frame_target(mol, selection=LIGAND, viewpoint=viewpoint, margin=2.6)
# Focus on the ligand, not on the molecule's origin: that sits at the protein
# centroid, several angstrom behind the site, and at this aperture that is the
# difference between a sharp ligand and a blurred one.
gala.depth_of_field(mol, selection=LIGAND, fstop=11.0)


# ---------------------------------------------------------------------------
heading("7. Render")
# ---------------------------------------------------------------------------
render(gala, "02_binding_site")
