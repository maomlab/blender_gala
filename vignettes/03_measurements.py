"""Vignette 3 — measuring, on a distance that means something.

The intracellular end of transmembrane helix 6 swings away from helix 2 when a
GPCR activates, opening the cavity a transducer couples into. Aranda-García et
al. turn that into a number: the distance between two alpha carbons, one on
TM2 and one on TM6, with thresholds sorting a structure into closed,
intermediate or open.

    Large scale investigation of GPCR molecular dynamics data uncovers
    allosteric sites and lateral gateways. Nat Commun 16, 2020 (2025).
    https://doi.org/10.1038/s41467-025-57034-y

So this measures exactly that, on one receptor caught in both states.

    blender --background --python vignettes/03_measurements.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, setup

mn, gala = setup()

from blender_gala.core.entity import AtomStructure

# The descriptor, in GPCRdb generic numbering: TM2 2x46 to TM6 6x37, alpha
# carbon to alpha carbon. For the adenosine A2A receptor those are L48 and
# L235 (GPCRdb, aa2ar_human).
TM2 = "chain A and resi 48 and name CA"
TM6 = "chain A and resi 235 and name CA"

# The transmembrane bundle. Two things are deliberately left out: 5UIG's BRIL
# fusion, numbered from 1001, which is a crystallography artefact; and helix 8,
# which lies flat along the membrane and, seen from the cytoplasm, sprawls
# across the frame in front of the measurement.
RECEPTOR = "chain A and protein and resi 1-291"

# One receptor in two positions rather than two receptors: cool for the
# antagonist-bound state, warm for the transducer-bound one.
COOL_HEX, WARM_HEX = "#93a6b8", "#ffab3d"
COOL = (0.45, 0.62, 0.82)
WARM = (1.0, 0.671, 0.239)

# Straight on, in the membrane plane, which with the frame built below puts
# the helices vertical and the TM6-TM2 gap across the frame. Casting rays at
# the measured span from a grid of angles says a 15 degree tilt would leave a
# little more of it visible; not enough to lean the receptor over for.
VIEWPOINT = (0.0, 0.0)

# Class A thresholds, the paper's Table 3.
INTERMEDIATE_FROM = 13.60
OPEN_FROM = 16.36


def state(distance: float) -> str:
    """Sort a TM2-TM6 distance into the paper's three states."""
    if distance < INTERMEDIATE_FROM:
        return "closed"
    if distance < OPEN_FROM:
        return "intermediate"
    return "open"


# ---------------------------------------------------------------------------
heading("1. The same receptor, caught closed and open")
# ---------------------------------------------------------------------------
# 5UIG is A2A with an antagonist bound; 6GDG is A2A coupled to mini-Gs. One
# protein, two ends of the same motion.
closed = load_structure("5UIG")
opened = load_structure("6GDG")


def show_receptor(molecule):
    """Style chain A only: 6GDG carries a transducer, and this is about TM6."""
    mask = gala.select(molecule, RECEPTOR)
    attribute = molecule.object.data.attributes.new("receptor", "BOOLEAN", "POINT")
    attribute.data.foreach_set("value", mask.astype(bool).tolist())
    # A slim cartoon on purpose. At the default ribbon width the measured
    # span runs inside the helices it connects and the figure shows a dashed
    # line vanishing into a ribbon; narrowing it opens the gaps between the
    # helices without giving up the fold.
    molecule.add_style(
        mn.StyleCartoon(
            peptide_width=1.3, peptide_thickness=0.35, peptide_arrows=False, quality=3
        ),
        selection="receptor",
        color=None,
    )
    return AtomStructure.from_any(molecule)


closed_structure = show_receptor(closed)
opened_structure = show_receptor(opened)


# ---------------------------------------------------------------------------
heading("2. Superpose, so the difference is the receptor's")
# ---------------------------------------------------------------------------
# Two crystals are two arbitrary orientations. Skip this and the helices differ
# by however the structures happened to be deposited, which is not the thing
# being measured.


def matched_alpha_carbons(mobile: AtomStructure, target: AtomStructure):
    """Alpha-carbon coordinates for the residues the two structures share."""
    coordinates = []
    for structure in (mobile, target):
        mask = structure.select(f"{RECEPTOR} and name CA")
        residues = np.asarray(structure.context.annotation("res_id"))[mask]
        coordinates.append(
            dict(zip(residues.tolist(), structure.world_positions()[mask], strict=True))
        )
    shared = sorted(set(coordinates[0]) & set(coordinates[1]))
    return (
        np.array([coordinates[0][res] for res in shared]),
        np.array([coordinates[1][res] for res in shared]),
    )


def superpose(molecule, mobile_points, target_points):
    """Kabsch: rotate and shift ``molecule`` onto the target points."""
    import bpy
    import mathutils

    mobile_centre = mobile_points.mean(axis=0)
    target_centre = target_points.mean(axis=0)
    correlation = (mobile_points - mobile_centre).T @ (target_points - target_centre)
    left, _, right = np.linalg.svd(correlation)
    # Guard against a reflection, which would superpose a mirror image.
    sign = float(np.sign(np.linalg.det(right.T @ left.T)))
    rotation = right.T @ np.diag([1.0, 1.0, sign]) @ left.T
    shift = target_centre - rotation @ mobile_centre

    matrix = mathutils.Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            matrix[row][column] = float(rotation[row][column])
    matrix.translation = mathutils.Vector(shift.tolist())
    molecule.object.matrix_world = matrix @ molecule.object.matrix_world
    bpy.context.view_layer.update()

    moved = (rotation @ mobile_points.T).T + shift
    return float(np.sqrt(((moved - target_points) ** 2).sum(axis=1).mean()))


mobile, target = matched_alpha_carbons(opened_structure, closed_structure)
rmsd = superpose(opened, mobile, target) * 100
print(f"  {len(mobile)} shared alpha carbons, {rmsd:.2f} A RMSD after superposing")


def atom_position(structure: AtomStructure, selection: str) -> np.ndarray:
    """World position of the one atom ``selection`` matches."""
    return structure.world_positions()[structure.select(selection)][0]


def stand_upright(molecules, reference: AtomStructure) -> None:
    """Rotate everything into the orientation the paper draws the receptor in.

    Figure 2A of Aranda-García et al. is a side view in the membrane plane:
    the transmembrane helices run up the frame, TM6 on the left and TM2 on the
    right, so the gap between them opens across the page. A crystal frame is
    arbitrary, and seen down the bundle axis a 5 A displacement projects to
    almost nothing, so the frame is built rather than guessed -- membrane
    normal to ``+Z``, TM6-to-TM2 to ``+X``, which the camera then looks at
    head on from ``-Y``.
    """
    import bpy
    import mathutils

    points = reference.world_positions()[reference.select(f"{RECEPTOR} and name CA")]
    centre = points.mean(axis=0)
    # Longest axis of a seven-helix bundle is the bundle axis.
    _, _, components = np.linalg.svd(points - centre)
    up = components[0]
    # Point it extracellular-side up, so the intracellular ends being measured
    # sit at the bottom of the frame.
    intracellular = atom_position(reference, TM2) + atom_position(reference, TM6)
    if np.dot(intracellular / 2 - centre, up) > 0:
        up = -up
    # Only the part of TM6-to-TM2 that lies in the membrane plane: the two
    # alpha carbons are at slightly different depths, and tilting the bundle
    # to square that up would be tilting it for nothing.
    across = atom_position(reference, TM2) - atom_position(reference, TM6)
    across = across - np.dot(across, up) * up
    across /= np.linalg.norm(across)
    rotation = np.array([across, np.cross(up, across), up])

    matrix = mathutils.Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            matrix[row][column] = float(rotation[row][column])
    matrix.translation = mathutils.Vector((-rotation @ centre).tolist())
    for molecule in molecules:
        molecule.object.matrix_world = matrix @ molecule.object.matrix_world
    bpy.context.view_layer.update()


stand_upright((closed, opened), closed_structure)


# ---------------------------------------------------------------------------
heading("3. The measurement")
# ---------------------------------------------------------------------------
values = {}
measurements = {}
for name, molecule in (("antagonist (5UIG)", closed), ("mini-Gs (6GDG)", opened)):
    measurement = gala.distance(molecule, TM2, TM6)
    values[name] = measurement.value
    measurements[name] = measurement
    print(
        f"  {name:20} TM2-TM6 = {measurement.value:5.2f} A  ->  "
        f"{state(measurement.value)}"
    )

print(f"\n  closed below {INTERMEDIATE_FROM} A, open at or above {OPEN_FROM} A")
print(
    f"  TM6 swings out by "
    f"{values['mini-Gs (6GDG)'] - values['antagonist (5UIG)']:.2f} A"
)


# ---------------------------------------------------------------------------
heading("4. Ambiguity is an error, not a guess")
# ---------------------------------------------------------------------------
# A selection matching more than one atom has no single distance, and picking
# one of them is how a figure ends up quietly wrong.
try:
    gala.distance(closed, "chain A and name CA", TM6)
except Exception as exc:
    print(f"  refused, correctly:\n      {exc}")

whole_residue = gala.distance(closed, "chain A and resi 48", TM6, reduce="centroid")
print("\n  Say how to reduce, and it will — residue 48 centroid to the TM6 CA:")
print(f"      {whole_residue.value:.2f} A")


# ---------------------------------------------------------------------------
heading("5. measure() dispatches on how many atoms it is given")
# ---------------------------------------------------------------------------
# Two picks is a distance, three an angle, four a dihedral: the scripting
# equivalent of clicking atoms in PyMOL's wizard.
for label, selections in (
    ("2 atoms -> distance", (TM2, TM6)),
    ("3 atoms -> angle", (TM2, "chain A and resi 232 and name CA", TM6)),
    (
        "4 atoms -> dihedral",
        (
            "chain A and resi 234 and name CA",
            "chain A and resi 235 and name N",
            TM6,
            "chain A and resi 235 and name C",
        ),
    ),
):
    value = gala.measure_atoms(closed, *selections)
    print(f"  {label:22} {value.value:8.2f} {value.unit}")


# ---------------------------------------------------------------------------
heading("6. Render")
# ---------------------------------------------------------------------------
gala.color_by_selection(closed, {"all": COOL_HEX})
gala.color_by_selection(opened, {"all": WARM_HEX})

gala.publication_setup(
    closed,
    preset=QUALITY,
    lighting_style="three_point",
    material_scheme="chemistry",
    viewpoint=VIEWPOINT,
    # Leave the origins alone. Recentring moves one molecule and not the
    # other, which would undo the superposition the whole figure rests on --
    # and strand the drawn measurements where the molecule used to be.
    origin_method=None,
)
# Frame on the receptor. Framing everything visible would include 6GDG's
# transducer, which is loaded, unstyled, and would push the receptor into a
# corner of an otherwise empty figure.
gala.frame_target(closed, viewpoint=VIEWPOINT, margin=1.12, selection=RECEPTOR)

# Draw last, once there is a camera. Both values want the same patch of screen
# -- the two lines share their TM2 end and run along the same axis -- so they
# are pushed apart along the camera's own up axis, which only exists after the
# camera has been aimed. Drawing before this point put the labels inside the
# helix bundle, where a still shows nothing at all.
import bpy
import mathutils

up = np.array(
    bpy.context.scene.camera.matrix_world.to_3x3() @ mathutils.Vector((0, 1, 0))
)
# Both lines start at the same end of TM2 and run along the same axis, so the
# open one is drawn as a thick ruler and the closed one as a thin line along
# its inner span: the 4.9 A the warm ruler carries on past the cool line is
# the activation the whole figure is about.
for name, molecule, colour, offset, thickness in (
    ("mini-Gs (6GDG)", opened, WARM, 9.0, 0.24),
    ("antagonist (5UIG)", closed, COOL, -9.0, 0.12),
):
    gala.measure.draw_measurement(
        measurements[name],
        target=molecule,
        colour=colour,
        # The verdict travels with the number: 12.70 A on its own says nothing
        # about whether the receptor is open.
        label_template="{text}  " + state(values[name]),
        label_size=2.0,
        label_offset=up * offset,
        radius=thickness,
    )

render(gala, "03_measurements")
