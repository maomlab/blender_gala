"""Vignette 3 — measuring distances, angles and dihedrals.

Gala follows PyMOL's measurement wizard: each pick must be unambiguous, and
the value is drawn into the scene as real geometry.

    blender --background --python vignettes/03_measurements.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import DATA_DIR, QUALITY, heading, render, setup

mn, gala = setup()

# ---------------------------------------------------------------------------
heading("1. A structure with known geometry")
# ---------------------------------------------------------------------------
# The synthetic binding site: every distance and angle in it was placed
# deliberately, so the numbers below can be checked by eye.
mol = mn.Molecule.load(os.path.join(DATA_DIR, "site.pdb"))
mol.add_style("ball_and_stick")

gala.publication_setup(
    mol, preset=QUALITY, lighting_style="three_point", viewpoint="front"
)


# ---------------------------------------------------------------------------
heading("2. A distance")
# ---------------------------------------------------------------------------
# SER1 OG donates a hydrogen bond to ASP2 OD1, placed at exactly 2.80 A.
distance = gala.distance(
    mol,
    "resi 1 and name OG",
    "resi 2 and name OD1",
    draw=True,
    colour=(1.0, 0.85, 0.2),
)
print(f"  {distance}")
print(f"  value as a float: {float(distance):.3f} A")
print(f"  drew {len(distance.objects)} objects")


# ---------------------------------------------------------------------------
heading("3. An angle")
# ---------------------------------------------------------------------------
# Three picks: the middle one is the vertex. An arc is drawn between the rays.
angle = gala.angle(
    mol,
    "resi 1 and name CB",
    "resi 1 and name OG",
    "resi 2 and name OD1",
    draw=True,
    colour=(0.35, 0.85, 1.0),
)
print(f"  {angle}")


# ---------------------------------------------------------------------------
heading("4. A dihedral")
# ---------------------------------------------------------------------------
# Four picks, signed by the IUPAC convention, drawn as the arc a Newman
# projection would show.
dihedral = gala.dihedral(
    mol,
    "resi 2 and name N",
    "resi 2 and name CA",
    "resi 2 and name CB",
    "resi 2 and name CG",
    draw=True,
    colour=(1.0, 0.5, 0.9),
)
print(f"  {dihedral}  (chi-1 of ASP2)")


# ---------------------------------------------------------------------------
heading("5. Ambiguity is an error, not a guess")
# ---------------------------------------------------------------------------
try:
    gala.distance(mol, "name CA", "name CB")
except gala.AmbiguousSelectionError as exc:
    print(f"  refused, correctly:\n      {exc}")

# When the ambiguity is intentional, say so.
from_centroid = gala.distance(
    mol,
    "resn LIG and name C1+C2+C3+C4+C5+C6",  # a whole aromatic ring
    "resi 1 and name OG",
    reduce=["centroid", "single"],
    draw=True,
)
print(f"\n  ring centroid to SER1 OG: {from_centroid.text}")
print(f"      from: {from_centroid.labels[0]}")


# ---------------------------------------------------------------------------
heading("6. measure() dispatches on how many atoms you give it")
# ---------------------------------------------------------------------------
for selections in (
    ("resi 1 and name OG", "resi 2 and name OD1"),
    ("resi 1 and name CB", "resi 1 and name OG", "resi 2 and name OD1"),
):
    result = gala.measure_atoms(mol, *selections)
    print(f"  {len(selections)} atoms -> {result.kind:9s} = {result.text}")


# ---------------------------------------------------------------------------
heading("7. Render")
# ---------------------------------------------------------------------------
gala.frame_target(mol, viewpoint="iso", margin=1.4)
render(gala, "03_measurements")
