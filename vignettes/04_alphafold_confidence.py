"""Vignette 4 — colouring an AlphaFold model by pLDDT confidence.

The primary colouring test case from the objectives: reproduce the AlphaFold
database's confidence bands, then use them to decide what to show.

    blender --background --python vignettes/04_alphafold_confidence.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_alphafold, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. Load a predicted model")
# ---------------------------------------------------------------------------
# Human p53 as AlphaFold predicts it: a confidently folded DNA-binding core
# between two long disordered arms. That shape is what makes the confidence
# bands worth looking at — a prediction is rarely uniformly good, and the
# colouring is how you see which parts to believe.
#
# AlphaFold writes pLDDT into the B-factor column, which is why colouring by
# B-factor and colouring by confidence are the same operation here.
mol = load_alphafold("P04637")
mol.add_style("cartoon")

plddt = np.asarray(mol.array.b_factor)
print(f"  {len(plddt)} atoms, pLDDT {plddt.min():.0f} to {plddt.max():.0f}")
for label, low, high in [
    ("very high (>90) ", 90, 100),
    ("confident (70-90)", 70, 90),
    ("low (50-70)      ", 50, 70),
    ("very low (<50)   ", 0, 50),
]:
    share = ((plddt > low) & (plddt <= high)).mean()
    print(f"      {label} {share:5.1%} of atoms")


# ---------------------------------------------------------------------------
heading("2. Colour by confidence")
# ---------------------------------------------------------------------------
result = gala.color_by_plddt(mol, mode="banded")
print(f"  coloured {result.n_colored} atoms over {result.vmin:.0f}-{result.vmax:.0f}")

print("\n  Legend:")
for label, rgb in result.legend:
    srgb = gala.color.linear_to_srgb(np.asarray(rgb))
    hexcode = "#" + "".join(f"{round(c * 255):02x}" for c in srgb)
    print(f"      {hexcode}  {label}")

# The 0-1 and 0-100 conventions are both handled; ColabFold writes 0-1.
print("\n  Colours are written to the mesh 'Color' attribute, which is the one")
print("  Molecular Nodes styles already read, so every style picks them up.")
print("  Writing it also mutes MN's 'Set Color' node, which would otherwise")
print("  store a generated colour over these on the way to the style.")


# ---------------------------------------------------------------------------
heading("3. Banded or continuous")
# ---------------------------------------------------------------------------
banded = gala.color_by_plddt(mol, mode="banded", write=False)
smooth = gala.color_by_plddt(mol, mode="continuous", write=False)
print(
    f"  banded and continuous differ: {not np.allclose(banded.colors, smooth.colors)}"
)
print("  'banded' matches the AlphaFold database viewer exactly;")
print("  'continuous' reads better on a molecular surface.")

gala.color_by_plddt(mol, mode="banded")


# ---------------------------------------------------------------------------
heading("4. Show only what is worth trusting")
# ---------------------------------------------------------------------------
confident = gala.select(mol, "b > 70")
print(f"  {confident.sum()} of {len(confident)} atoms are confident or better")
print(f"  very low (< 50): {gala.select(mol, 'b < 50').sum()} atoms")

# Selections and colouring compose: style only the confident part.
#
# A style's `selection` names a boolean attribute on the mesh — Molecular Nodes
# reads it inside geometry nodes, where Gala's parser is not running, so the
# selection string has to be evaluated here and stored under a name. Passing
# "b > 70" straight to add_style warns and draws nothing.
attribute = mol.object.data.attributes.new("confident", "BOOLEAN", "POINT")
attribute.data.foreach_set("value", confident.astype(bool).tolist())
mol.add_style("ribbon", selection="confident")


# ---------------------------------------------------------------------------
heading("5. The same machinery for any per-residue data")
# ---------------------------------------------------------------------------
# A conservation score, a per-residue energy, a deep mutational scan.
conservation = {res_id: (res_id % 5) / 4.0 for res_id in range(1, 9)}
scores = gala.color_by_attribute(
    mol, conservation, cmap="viridis", vmin=0.0, vmax=1.0, write=False
)
print(f"  mapped {len(conservation)} residues over a fixed 0-1 range,")
print("  so several structures stay comparable to each other.")

# Put the confidence colours back for the render.
gala.color_by_plddt(mol)


# ---------------------------------------------------------------------------
heading("6. Render")
# ---------------------------------------------------------------------------
gala.publication_setup(
    mol,
    preset=QUALITY,
    lighting_style="three_point",
    material_scheme="chemistry",
    viewpoint="front",
)
gala.label_hud(mol, "AlphaFold confidence (pLDDT)", location=(0.04, 0.95), size=26)
render(gala, "04_alphafold_confidence")
