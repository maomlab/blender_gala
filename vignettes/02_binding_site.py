"""Vignette 2 — a ligand binding site, analysed and annotated.

The full Objective 2 workflow: find every interaction between a ligand and its
pocket, draw them, and label the residues that make them.

    blender --background --python vignettes/02_binding_site.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
drawn = gala.draw_interactions(
    contacts,
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
labels = gala.label(
    mol,
    POCKET,
    template="{one}{resi}",
    level="residue",
    anchor="ca",
    style="card",  # a translucent plane keeps text legible over the surface
    size=2.0,
    offset=3.0,
    billboard=True,
)
print(f"  labelled {len([o for o in labels if o.type == 'FONT'])} residues")

# A caption that can never be occluded, drawn in screen space.
gala.label_hud(mol, "Ligand binding site", location=(0.04, 0.95), size=28)


# ---------------------------------------------------------------------------
heading("6. Frame the site rather than the whole protein")
# ---------------------------------------------------------------------------
# The margin only has to clear the labels and dashes drawn around the
# molecule; the framing itself is now tight to the atoms.
gala.frame_target(mol, viewpoint="iso", margin=1.15)
gala.depth_of_field(mol, fstop=4.0)


# ---------------------------------------------------------------------------
heading("7. Render")
# ---------------------------------------------------------------------------
render(gala, "02_binding_site")
