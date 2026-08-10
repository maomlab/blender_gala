"""Vignette 17 — the picture, built out of its own screen.

After Elena Manferdini. For *Building the Picture* she took Mies van der Rohe's
860-880 Lake Shore Drive, traced the facade into a digital drawing of its grid,
and then multiplied the grid, wove the lines and infused colour and line weight
until the orthogonal structure came back as ornament. The work is hung to be
read twice — up close it is a field of marks, and from across the room it is a
building — and in the *Madame Architect* interview she says why she keeps
making it: public work "reflects and reveals our society, adding meaning to our
cities and uniqueness to our communities."

    https://www.madamearchitect.org/interviews/2024/10/22/elena-manferdini

That is a claim about what art is for, and it is a narrow one. Not to decorate
the structure, and not to explain it. To *re-print* it — through a medium with
its own grain and its own colour — until a thing that was only functional
starts addressing somebody. The structure is not ornamented. It is rebuilt out
of ornament.

Structural biology is full of pictures that refuse this on principle, and the
other sixteen vignettes here are those pictures. This one takes the offer.

**The facade is already in the coordinates.** A class A GPCR is seven
transmembrane helices standing in a membrane: a colonnade with a floor slab
top and bottom. So the grid this traces is not borrowed from a building — it is
measured off the protein, and then multiplied across the sheet the way she
multiplies Mies.

**The protein is the one that makes the medium possible.** Four visual
pigments, which are the whole apparatus of seeing a colour:

* rhodopsin, the rod pigment (1F88 — Palczewski's structure, the first GPCR
  anyone solved), lmax 498 nm. Achromatic. It carries form and brightness and
  no colour at all, which is exactly the job of the **black key plate**;
* the three cone opsins — OPN1SW, OPN1MW, OPN1LW at lmax 420, 534 and 564 nm —
  which are the **colour separations**. None of the three has an experimental
  structure, so all three come from AlphaFold, and they are close enough
  relatives of rhodopsin that they superpose onto it to about 1 A.

Every colour anybody has ever seen — including every colour in Manferdini's
print — is three numbers off three of those pigments, sampled on a mosaic of
discrete receptors and reassembled at a distance. Which is a halftone. The
retina *is* the medium this picture is printed in.

And the fact that makes it art rather than illustration: **the "red" cone peaks
at 564 nm, which is yellow-green.** There is no red pigment in the human eye.
Red is what gets constructed out of the difference between two proteins whose
absorption maxima are 30 nm and a handful of side chains apart. So the L plate
here is screened at 0 degrees — the angle a printer gives the yellow plate,
because yellow is the ink the eye is least able to catch — and inked in coral.
The plate is yellow, the ink is red, the sensation is red, and none of the
three is the same statement.

So the whole sheet is printed rather than lit. Four plates, four inks, the
standard screen angles (0, 15, 45, 75 degrees, black on 45 where the eye is
worst at seeing a screen), and dot area driven by depth. Nine bays over three
floors, each taking a different combination of the four plates and each of them
a hair out of register — because registration is what a printer and a
structural biologist both call the same operation, and because the slip is
where the picture stops being a diagram.

    Palczewski, K. et al. Crystal structure of rhodopsin: a G protein-coupled
    receptor. Science 289, 739-745 (2000). https://doi.org/10.1126/science.289.5480.739

    Bowmaker, J. K. & Dartnall, H. J. A. Visual pigments of rods and cones in a
    human retina. J. Physiol. 298, 501-511 (1980).
    https://doi.org/10.1113/jphysiol.1980.sp013097

    blender --background --python vignettes/17_halftone_facade.py
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (
    QUALITY,
    heading,
    load_alphafold,
    load_structure,
    render,
    save_blend,
    setup,
)

mn, gala = setup()

import bpy
import mathutils
import numpy as np

from blender_gala.color.colormaps import hex_to_rgb
from blender_gala.core.entity import AtomStructure
from blender_gala.scene.camera import ensure_camera
from blender_gala.scene.presets import get_preset

# --- the paper --------------------------------------------------------------
# A cream sheet, and the two greys that sit on it before any pigment does: the
# coarse dot field that covers the whole page, and the pale flood coat that
# lifts a rectangle out of it and gives the plates somewhere to land.
PAPER = "#f2ece1"
FIELD_INK = "#2b2b31"
FLOOD_INK = "#c6c2cf"

# 1358 x 2013 is the sheet the reference print is on. A figure is square here
# and a press image is 3:2; a print is neither, and the proportion is the first
# thing that says which of the three this is.
SHEET_RATIO = 1358 / 2013

# --- the four plates --------------------------------------------------------
# (accession, label, lmax in nm, ink, screen angle in degrees).
#
# The inks are the print's, not the pigments'. A pigment's lmax is a number
# about absorption and says nothing about what the pigment looks like or what
# seeing through it feels like, and pretending otherwise — 564 nm rendered as
# yellow-green, which is what it is — would be the one dishonest thing in this
# picture. The angles are a printer's standard set.
PLATES = (
    ("1f88", "rhodopsin", 498, "#1b1b21", 45.0),
    ("P03999", "OPN1SW", 420, "#2f3f78", 15.0),
    ("P04001", "OPN1MW", 534, "#2f93ad", 75.0),
    ("P04000", "OPN1LW", 564, "#dd564a", 0.0),
)
#: One letter each, for the map of which bay took which ink.
PLATE_CODES = ("K", "S", "M", "L")
#: pLDDT below which a predicted residue does not go on the plate. 70 is the
#: bottom of AlphaFold's own "confident" band, so this is their line, not one
#: chosen to make the picture tidy.
PLDDT_FLOOR = 70.0

# --- the screen -------------------------------------------------------------
#: Dot cells across the height of the sheet. Coarse on purpose — a screen you
#: cannot see is a photograph, and a screen you can see is a print — but not so
#: coarse that a bundle is a dozen dots wide, because then it is only a screen.
#: At 210 an impression is about 35 cells across — three or four to a helix,
#: which is a halftone of a protein rather than a mosaic of one.
RULING = 210.0
#: The background field is a good deal coarser, so the two screens read apart
#: at a glance and the block sits in front of the sheet rather than in it.
FIELD_RULING = 52.0
#: How opaque one ink is where its dot is solid. Under 1 so a plate laid over
#: another mixes with it instead of erasing it, which is what overprinting is.
INK_OPACITY = 0.80

#: Dot area as a fraction of its cell, near the camera and far from it. Nearer
#: is denser, which is PyMOL's depth cue restated as a quantity of ink. Light,
#: because a plate here is one pass of four and a pass that covers the paper
#: leaves nothing for the next one to say.
TONE_NEAR, TONE_FAR = 0.58, 0.07
#: How much the silhouette thickens the dot. A ribbon seen edge-on gets more
#: ink than one seen flat, so the form draws its own outline.
FACING_LIFT = 0.34

# --- the layout -------------------------------------------------------------
#: Floors down the sheet and bays across each one.
BANDS, COLUMNS = 3, 3
#: How much of the sheet the impressions cover, across and down. The rest is
#: margin, and the margin is not empty — it is where the dot field is left
#: showing, which is what makes the block read as printed onto something.
BLOCK = (0.78, 0.74)
#: How far a plate slips from register, as a fraction of the sheet width.
MISREGISTRATION = 0.014
#: Depth between successive plates in a stack, in Blender units. Small — they
#: are meant to be one impression, not four objects.
PLATE_PITCH = 0.04

#: Which plates each bay gets, one entry per bay. Not every bay takes every
#: ink: her facade is a grid of cells under different treatments, and a sheet
#: where all four plates hit all nine bays is one flat colour made of four.
#: Index 0 is the key, and it is in most of them because a key plate that skips
#: is a picture that has lost its drawing.
TREATMENTS = (
    (0, 3),
    (0, 1, 2),
    (2,),
    (0, 2, 3),
    (0, 1),
    (1, 2, 3),
    (0,),
    (0, 2),
    (0, 1, 3),
)

random.seed(17)


# ---------------------------------------------------------------------------
heading("1. Four pigments")
# ---------------------------------------------------------------------------
molecules, structures = [], []
for accession, label, lmax, ink, angle in PLATES:
    experimental = len(accession) == 4
    molecule = load_structure(accession) if experimental else load_alphafold(accession)
    structure = AtomStructure.from_any(molecule)

    # 1F88 is a dimer in the asymmetric unit, and a facade traced off two
    # copies of a colonnade leaning against each other is a facade of nothing.
    monomer = "chain A" if structure.count("chain A") else "all"
    if not experimental:
        # AlphaFold writes pLDDT into the B-factor column, and an opsin model's
        # termini come out under 50 — long strings the predictor is telling you
        # it does not know the shape of. Printed, they are ink wandering off
        # the sheet. Only what the model stands behind goes on the plate, which
        # is the same judgement `color_by_plddt` makes, used as a mask rather
        # than as a colour.
        confident = f"({monomer}) and b > {PLDDT_FLOOR}"
        if structure.count(confident) > 200:
            monomer = confident

    # Through Gala's own machinery for it: store the plate as a named selection
    # and hang the style on that, so what is off the plate is never drawn
    # rather than drawn and then hidden.
    gala.create_alias(molecule, "plate", monomer)
    gala.style_alias(molecule, "plate", style="cartoon", color=None)

    molecules.append(molecule)
    structures.append(structure)
    print(
        f"  {label:10s} lmax {lmax} nm   ink {ink}   screen {angle:4.0f} deg   "
        f"{structure.count(monomer)} of {structure.n_atoms} atoms on the plate"
    )

reference = structures[0]
CHAIN = "chain A" if reference.count("chain A") else "all"
print(f"  tracing the grid off {CHAIN} of {PLATES[0][1]}")


# ---------------------------------------------------------------------------
heading("2. Register the plates")
# ---------------------------------------------------------------------------
# Registration, in printing, is getting the four plates to describe the same
# picture before any of them is inked. It is the same word and the same
# operation here, and it is worth doing properly: rhodopsin and the cone opsins
# are about 40% identical and their residue numbers do not line up, so pairing
# them by number — which is what the two-conformations-of-one-protein vignettes
# here can get away with — misregisters the plates by 20 A and turns the whole
# sheet into four different pictures. Biotite's `superimpose_homologs` aligns
# the sequences first and rejects the outliers, and Gala's Kabsch then puts the
# Blender object where the alignment says it goes.


def matched_alpha_carbons(mobile: AtomStructure, target: AtomStructure):
    """Alpha carbons paired by sequence alignment, in world coordinates."""
    from biotite.structure import superimpose_homologs

    frames = []
    for structure, selection in (
        (mobile, "name CA"),
        (target, f"{CHAIN} and name CA"),
    ):
        mask = structure.select(selection)
        carbons = structure.array[mask].copy()
        # The array carries the deposited coordinates; the object carries the
        # frame the last step left it in. Only the second one is the picture.
        carbons.coord = structure.world_positions()[mask]
        frames.append(carbons)

    _, _, target_anchors, mobile_anchors = superimpose_homologs(frames[1], frames[0])
    return frames[0].coord[mobile_anchors], frames[1].coord[target_anchors]


def superpose(molecule, mobile_points, target_points) -> float:
    """Kabsch: rotate and shift ``molecule`` onto the target points."""
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


for molecule, structure, spec in zip(
    molecules[1:], structures[1:], PLATES[1:], strict=True
):
    try:
        mobile, target = matched_alpha_carbons(structure, reference)
    except Exception as exc:  # pragma: no cover - the fixture fallback
        print(
            f"  {spec[1]:10s} cannot be aligned ({exc.__class__.__name__}); left as is"
        )
        continue
    if len(mobile) < 20:  # pragma: no cover - the fixture fallback
        print(f"  {spec[1]:10s} only {len(mobile)} anchors; left in its own frame")
        continue
    rmsd = superpose(molecule, mobile, target) * 100
    print(f"  {spec[1]:10s} {len(mobile):4d} anchors, {rmsd:5.2f} A RMSD in register")


# ---------------------------------------------------------------------------
heading("3. Stand the colonnade up, and turn it broadside")
# ---------------------------------------------------------------------------
# A crystal frame is arbitrary and a predicted one is worse. What makes seven
# helices read as columns is that they are vertical, so the frame is built:
# the bundle's principal axis — which for a transmembrane bundle *is* the
# membrane normal — rotated onto +Z, and the whole registered set carried with
# it by the same matrix, so nothing that step 2 achieved is undone.


def helix_runs(structure: AtomStructure, selection: str, minimum: int = 12):
    """Contiguous runs of helical residues, as arrays of alpha-carbon positions."""
    mask = structure.select(selection)
    residues = np.asarray(structure.context.annotation("res_id"))[mask]
    points = structure.world_positions()[mask]
    runs, start = [], 0
    for index in range(1, len(residues) + 1):
        if index == len(residues) or residues[index] != residues[index - 1] + 1:
            if index - start >= minimum:
                runs.append(points[start:index])
            start = index
    return runs


helix_mask = reference.select(f"{CHAIN} and name CA and ss 1")
helix_points = reference.world_positions()[helix_mask]
if len(helix_points) < 20:  # pragma: no cover - only on the fixture fallback
    helix_points = reference.world_positions()

centre = helix_points.mean(axis=0)
_, _, axes = np.linalg.svd(helix_points - centre)
normal = axes[0] / np.linalg.norm(axes[0])
if normal[2] < 0:
    normal = -normal

upright = mathutils.Vector(normal.tolist()).rotation_difference(
    mathutils.Vector((0.0, 0.0, 1.0))
)
frame = mathutils.Matrix.Translation(
    -(
        mathutils.Matrix.Rotation(upright.angle, 4, upright.axis)
        @ mathutils.Vector(centre.tolist())
    )
) @ mathutils.Matrix.Rotation(upright.angle, 4, upright.axis)

for molecule in molecules:
    molecule.object.matrix_world = frame @ molecule.object.matrix_world
bpy.context.view_layer.update()

# Re-read, because everything after this is measured in the frame just built.
structures = [AtomStructure.from_any(molecule) for molecule in molecules]
reference = structures[0]
print(f"  bundle axis {tuple(round(float(v), 3) for v in normal)} rotated onto +Z")

# Upright is half of it. Seen down the wrong side, a seven-helix bundle is a
# ring seen end-on and the columns pile up on each other; seen broadside they
# spread across the frame. Which side that is comes from the helices as well:
# the principal axis of their centres *in the picture plane*, turned onto the
# horizontal. This is the difference between an elevation and a corner view,
# and no facade was ever drawn from the corner.
runs = helix_runs(reference, f"{CHAIN} and name CA and ss 1")
runs.sort(key=len, reverse=True)
plan = np.array([run[:, :2].mean(axis=0) for run in runs[:7]])
if len(plan) > 2:
    _, _, ground = np.linalg.svd(plan - plan.mean(axis=0))
    broadside = -math.atan2(float(ground[0][1]), float(ground[0][0]))
else:  # pragma: no cover - the fixture fallback
    broadside = 0.0

elevation = mathutils.Matrix.Rotation(broadside, 4, "Z")
for molecule in molecules:
    molecule.object.matrix_world = elevation @ molecule.object.matrix_world
bpy.context.view_layer.update()
structures = [AtomStructure.from_any(molecule) for molecule in molecules]
reference = structures[0]
print(f"  turned {math.degrees(broadside):.1f} deg about +Z onto the elevation")


# ---------------------------------------------------------------------------
heading("4. Trace the grid off the protein")
# ---------------------------------------------------------------------------
# Manferdini's first move is tracing: the facade becomes a drawing of the grid
# before it becomes anything else. Here the drawing is measured. Each run of
# helical alpha carbons is a column — its centre and its width taken from where
# its atoms actually are — and the two ends of the bundle are the floor slabs,
# which for a membrane protein is not a metaphor: they are where the bilayer is.
runs = helix_runs(reference, f"{CHAIN} and name CA and ss 1")
# Seven transmembrane helices, longest first; rhodopsin also has helix 8 lying
# along the membrane, which is short and is not a column.
runs.sort(key=len, reverse=True)
runs = runs[:7]
columns = sorted(
    (float(run[:, 0].mean()), max(float(run[:, 0].std()), 0.02)) for run in runs
)
if not columns:  # pragma: no cover - only on the fixture fallback
    # Nothing helical to trace, which happens when the PDB is unreachable and
    # the committed test fixture stands in. Seven bays laid out evenly across
    # whatever is there, so the sheet still has a grid and the picture still
    # composes — but said out loud, because a measured grid and an assumed one
    # are not the same claim.
    reach = float(np.ptp(reference.world_positions()[:, 0])) / 2
    columns = [(reach * (index - 3) / 3.5, reach * 0.09) for index in range(7)]
    print("  no secondary structure to trace; seven bays assumed instead")
print(f"  {len(columns)} columns traced, {len(runs)} helical runs kept")
for position, width in columns:
    print(f"      x {position * 100:7.1f} A   width {width * 200:5.1f} A")

# The slabs. Where the helical carbons start and stop along the bundle axis,
# which is the hydrophobic span the bilayer sits on.
spans = (
    np.concatenate([run[:, 2] for run in runs])
    if runs
    else reference.world_positions()[:, 2]
)
slab_low, slab_high = (float(v) for v in np.percentile(spans, (6.0, 94.0)))
print(
    f"  slabs at z {slab_low * 100:.1f} and {slab_high * 100:.1f} A "
    f"({(slab_high - slab_low) * 100:.0f} A apart)"
)

# The monomer's own size, not the crystal's: the second copy of the dimer is
# not on the plate, so it must not set the scale of the sheet either. Its
# height and width are taken separately, because what has to fit a cell is the
# standing bundle rather than a sphere around it.
plate_points = reference.world_positions()[reference.select(CHAIN)]
bundle_width = float(np.ptp(plate_points[:, 0]))
bundle_height = float(np.ptp(plate_points[:, 2]))
motif_centre = sum(position for position, _ in columns) / len(columns)
period = (columns[-1][0] - columns[0][0]) if len(columns) > 1 else bundle_width
print(f"  bundle {bundle_width * 100:.0f} A across, {bundle_height * 100:.0f} A tall")


# ---------------------------------------------------------------------------
heading("5. The sheet, and where the plates fall on it")
# ---------------------------------------------------------------------------
# Every dimension below is a multiple of the protein, so the composition is the
# same composition on the real structure and on the fixture CI falls back to
# when the PDB is unreachable.
sheet_height = bundle_height * BANDS / BLOCK[1] * 1.18
sheet_width = sheet_height * SHEET_RATIO
block_width, block_height = sheet_width * BLOCK[0], sheet_height * BLOCK[1]
cell_width, cell_height = block_width / COLUMNS, block_height / BANDS

# One scale for every impression: a facade whose bays are different sizes is a
# collage. What varies from cell to cell is which inks it took, not how big it
# is — the same discipline the grid itself is under.
impression = cell_height * 1.08 / bundle_height
print(
    f"  sheet {sheet_width * 100:.0f} x {sheet_height * 100:.0f} A, "
    f"block {BLOCK[0]:.0%} x {BLOCK[1]:.0%}, {COLUMNS} bays by {BANDS} floors"
)
print(f"  each impression at {impression:.2f}x, bay {cell_width * 100:.0f} A wide")

# Her bands do not stack flush; each one slides against the one below it, and
# that slide is most of what stops a grid from being a spreadsheet.
band_shift = (-0.055, 0.042, -0.018)
cells = []
for band in range(BANDS):
    z = ((BANDS - 1) / 2 - band) * cell_height
    for column in range(COLUMNS):
        x = (column - (COLUMNS - 1) / 2) * cell_width + band_shift[band] * sheet_width
        cells.append(
            (
                x,
                z + random.uniform(-0.02, 0.02) * cell_height,
                math.radians(random.uniform(-1.6, 1.6)),  # a hair off square
                TREATMENTS[(band * COLUMNS + column) % len(TREATMENTS)],
            )
        )

for band in range(BANDS):
    row = cells[band * COLUMNS : (band + 1) * COLUMNS]
    treatment = "  ".join(
        "".join(PLATE_CODES[index] for index in sorted(cell[3])).ljust(4)
        for cell in row
    )
    print(f"      floor {band + 1}: {treatment}")
print(f"  {len(cells)} bays, {sum(len(cell[3]) for cell in cells)} impressions")


# ---------------------------------------------------------------------------
heading("6. The screen")
# ---------------------------------------------------------------------------
# A halftone, built where a halftone belongs: in *window* coordinates, so the
# dot grid is fixed to the sheet and not to any object on it. Every plate is
# screened by the same ruling through the same grid, rotated to its own angle —
# which is the whole reason four-colour printing works and the whole reason it
# produces rosettes where the angles cross.
#
# The shader is Emission mixed with Transparent, and nothing else. A Mix Shader
# is a linear blend, so `mix(fac, Transparent, Emission)` is exactly source-over
# compositing: ink laid on what is behind it at opacity `fac`. No lights, no
# bounces, no shading — a printed sheet is not lit, it is covered.
#
# `aspect` below is set with the frame in step 7 and read when a material is
# built in step 8, because how wide a window-space dot has to be to come out
# round is a fact about the sheet rather than about the ink.
aspect = 1.0


def build_screen(
    name: str,
    ink: str,
    angle: float,
    ruling: float,
    opacity: float,
    tone: float | None = None,
    depth: tuple[float, float] | None = None,
):
    """A halftone ink: flat colour, screened at ``angle``, dot area from tone.

    Parameters
    ----------
    name, ink : str
        Material name, and the ink as a hex colour.
    angle : float
        Screen angle in degrees.
    ruling : float
        Dot cells across the height of the frame.
    opacity : float
        How opaque the ink is inside a dot.
    tone : float, optional
        A flat tone, for the background elements.
    depth : tuple of float, optional
        ``(near, far)`` view depths to map onto ``TONE_NEAR``-``TONE_FAR``.
        Given instead of ``tone`` for anything with form in it.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (900, 0)

    # --- the grid -----------------------------------------------------------
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-1100, 300)

    centred = tree.nodes.new("ShaderNodeVectorMath")
    centred.label = "To sheet centre"
    centred.operation = "SUBTRACT"
    centred.location = (-900, 300)
    centred.inputs[1].default_value = (0.5, 0.5, 0.0)
    tree.links.new(coordinates.outputs["Window"], centred.inputs[0])

    # Square the cell up: a unit of window x is `resolution` times as many
    # pixels as a unit of window y, and a dot has to come out round.
    square = tree.nodes.new("ShaderNodeVectorMath")
    square.label = "Round the dot"
    square.operation = "MULTIPLY"
    square.location = (-700, 300)
    square.inputs[1].default_value = (aspect, 1.0, 0.0)
    tree.links.new(centred.outputs["Vector"], square.inputs[0])

    turn = tree.nodes.new("ShaderNodeVectorRotate")
    turn.label = f"Screen angle {angle:.0f} deg"
    turn.rotation_type = "Z_AXIS"
    turn.location = (-500, 300)
    turn.inputs["Center"].default_value = (0.0, 0.0, 0.0)
    turn.inputs["Angle"].default_value = math.radians(angle)
    tree.links.new(square.outputs["Vector"], turn.inputs["Vector"])

    lines = tree.nodes.new("ShaderNodeVectorMath")
    lines.label = f"{ruling:.0f} lines"
    lines.operation = "SCALE"
    lines.location = (-300, 300)
    lines.inputs["Scale"].default_value = ruling
    tree.links.new(turn.outputs["Vector"], lines.inputs[0])

    cell = tree.nodes.new("ShaderNodeVectorMath")
    cell.label = "One cell"
    cell.operation = "FRACTION"
    cell.location = (-100, 300)
    tree.links.new(lines.outputs["Vector"], cell.inputs[0])

    offset = tree.nodes.new("ShaderNodeVectorMath")
    offset.label = "Distance from the dot"
    offset.operation = "DISTANCE"
    offset.location = (100, 300)
    offset.inputs[1].default_value = (0.5, 0.5, 0.0)
    tree.links.new(cell.outputs["Vector"], offset.inputs[0])

    # --- the tone -----------------------------------------------------------
    if depth is None:
        level = tree.nodes.new("ShaderNodeValue")
        level.label = "Flat tone"
        level.location = (100, -60)
        level.outputs[0].default_value = float(tone if tone is not None else 0.5)
        tone_socket = level.outputs[0]
    else:
        camera = tree.nodes.new("ShaderNodeCameraData")
        camera.location = (-1100, -160)

        cue = tree.nodes.new("ShaderNodeMapRange")
        cue.label = "Depth to ink"
        cue.location = (-700, -160)
        cue.clamp = True
        cue.inputs[1].default_value = depth[0]
        cue.inputs[2].default_value = depth[1]
        cue.inputs[3].default_value = TONE_NEAR
        cue.inputs[4].default_value = TONE_FAR
        tree.links.new(camera.outputs["View Z Depth"], cue.inputs[0])

        facing = tree.nodes.new("ShaderNodeLayerWeight")
        facing.label = "Silhouette"
        facing.location = (-700, -420)
        facing.inputs["Blend"].default_value = 0.42

        edge = tree.nodes.new("ShaderNodeMath")
        edge.operation = "POWER"
        edge.location = (-500, -420)
        edge.inputs[1].default_value = 2.0
        tree.links.new(facing.outputs["Facing"], edge.inputs[0])

        lift = tree.nodes.new("ShaderNodeMath")
        lift.operation = "MULTIPLY"
        lift.location = (-300, -420)
        lift.inputs[1].default_value = FACING_LIFT
        tree.links.new(edge.outputs["Value"], lift.inputs[0])

        combined = tree.nodes.new("ShaderNodeMath")
        combined.label = "Tone"
        combined.operation = "ADD"
        combined.location = (-100, -160)
        combined.use_clamp = True
        tree.links.new(cue.outputs["Result"], combined.inputs[0])
        tree.links.new(lift.outputs["Value"], combined.inputs[1])
        tone_socket = combined.outputs["Value"]

    # --- dot area proportional to tone --------------------------------------
    # A dot covering fraction t of a unit cell has radius sqrt(t/pi). Getting
    # this right is the difference between a screen and a polka dot: it is what
    # makes a mid tone read as a mid tone rather than as a pattern.
    area = tree.nodes.new("ShaderNodeMath")
    area.label = "sqrt(tone)"
    area.operation = "SQRT"
    area.location = (100, -160)
    tree.links.new(tone_socket, area.inputs[0])

    radius = tree.nodes.new("ShaderNodeMath")
    radius.label = "Dot radius"
    radius.operation = "MULTIPLY"
    radius.location = (300, -160)
    radius.inputs[1].default_value = 1.0 / math.sqrt(math.pi)
    tree.links.new(area.outputs["Value"], radius.inputs[0])

    # The dot's edge, softened by about half a cell's worth of pixel so it does
    # not alias into a stipple at the resolutions these are viewed at.
    softness = 0.5 / max(ruling * 0.06, 1.0)
    inner = tree.nodes.new("ShaderNodeMath")
    inner.operation = "SUBTRACT"
    inner.location = (500, -260)
    inner.inputs[1].default_value = softness
    tree.links.new(radius.outputs["Value"], inner.inputs[0])

    outer = tree.nodes.new("ShaderNodeMath")
    outer.operation = "ADD"
    outer.location = (500, -60)
    outer.inputs[1].default_value = softness
    tree.links.new(radius.outputs["Value"], outer.inputs[0])

    dot = tree.nodes.new("ShaderNodeMapRange")
    dot.label = "Inside the dot"
    dot.location = (700, 150)
    dot.clamp = True
    tree.links.new(offset.outputs["Value"], dot.inputs[0])
    tree.links.new(outer.outputs["Value"], dot.inputs[1])
    tree.links.new(inner.outputs["Value"], dot.inputs[2])
    dot.inputs[3].default_value = 0.0
    dot.inputs[4].default_value = 1.0

    covered = tree.nodes.new("ShaderNodeMath")
    covered.label = "Ink opacity"
    covered.operation = "MULTIPLY"
    covered.location = (900, 150)
    covered.inputs[1].default_value = opacity
    tree.links.new(dot.outputs["Result"], covered.inputs[0])

    # --- the ink ------------------------------------------------------------
    clear = tree.nodes.new("ShaderNodeBsdfTransparent")
    clear.location = (900, 380)

    pigment = tree.nodes.new("ShaderNodeEmission")
    pigment.location = (900, 260)
    pigment.inputs["Color"].default_value = (*hex_to_rgb(ink), 1.0)
    pigment.inputs["Strength"].default_value = 1.0

    press = tree.nodes.new("ShaderNodeMixShader")
    press.location = (1150, 300)
    tree.links.new(covered.outputs["Value"], press.inputs[0])
    tree.links.new(clear.outputs["BSDF"], press.inputs[1])
    tree.links.new(pigment.outputs["Emission"], press.inputs[2])
    output.location = (1400, 300)
    tree.links.new(press.outputs["Shader"], output.inputs["Surface"])
    return material


print(
    f"  {RULING:.0f}-line screen at "
    + ", ".join(f"{angle:.0f} deg" for *_, angle in PLATES)
    + f"; ink {INK_OPACITY:.0%} opaque, tone {TONE_FAR:.0%}-{TONE_NEAR:.0%} by depth"
)


# ---------------------------------------------------------------------------
heading("7. The camera as a copy stand")
# ---------------------------------------------------------------------------
# Orthographic and square on, because a print has no perspective in it. The
# camera is placed rather than fitted: `frame_target` solves for a subject, and
# what is being framed here is the sheet.
gala.setup_render(preset=QUALITY, transparent=False)
gala.scene.render.setup_color_management()

scene = bpy.context.scene
preset = get_preset(QUALITY)
scene.render.resolution_x = preset.resolution[0]
scene.render.resolution_y = round(preset.resolution[0] / SHEET_RATIO)
aspect = scene.render.resolution_x / scene.render.resolution_y

camera = ensure_camera(scene)
camera.data.type = "ORTHO"
# Horizontal, so `ortho_scale` is the width of the sheet rather than whichever
# of the two image dimensions happens to be larger.
camera.data.sensor_fit = "HORIZONTAL"
camera.data.ortho_scale = sheet_width
stand = sheet_width * 4.0
camera.location = (0.0, -stand, 0.0)
camera.rotation_euler = (math.pi / 2, 0.0, 0.0)
camera.data.clip_start = 0.001
camera.data.clip_end = stand * 3.0
print(
    f"  orthographic, {sheet_width * 100:.0f} A across, "
    f"{scene.render.resolution_x} x {scene.render.resolution_y}"
)

# Where the plate stack sits in depth, which is what the screen's tone is
# mapped over. Deliberately narrow — about one impression deep — so the ramp is
# spent across the thickness of a single bundle and the near face of a ribbon
# takes visibly more ink than its far face. Widen it and every plate comes out
# a flat silhouette.
plate_depth = bundle_width * impression * 0.6
plate_near = stand - PLATE_PITCH * len(PLATES) - plate_depth
plate_far = stand + plate_depth
print(f"  ink ramp over view depth {plate_near:.3f} to {plate_far:.3f} BU")


# ---------------------------------------------------------------------------
heading("8. Ink the plates and set them down")
# ---------------------------------------------------------------------------
inks = [
    build_screen(
        f"GALA Screen {label}",
        ink,
        angle,
        RULING,
        INK_OPACITY,
        depth=(plate_near, plate_far),
    )
    for _, label, _, ink, angle in PLATES
]
for molecule, ink in zip(molecules, inks, strict=True):
    gala.assign_material(molecule, ink, style="cartoon")

bases = [molecule.object.matrix_world.copy() for molecule in molecules]
used = [0] * len(PLATES)
for cell_index, (x, z, tilt, treatment) in enumerate(cells):
    for plate_index in treatment:
        molecule = molecules[plate_index]
        # Out of register by a hair, differently in every cell. In a print this
        # is a fault; in her work — and here — it is the thing that says four
        # separate passes were made over the same paper.
        slip = (
            random.uniform(-1.0, 1.0) * MISREGISTRATION * sheet_width,
            random.uniform(-1.0, 1.0) * MISREGISTRATION * sheet_width,
        )
        placement = (
            mathutils.Matrix.Translation(
                (x + slip[0], -plate_index * PLATE_PITCH, z + slip[1])
            )
            @ mathutils.Matrix.Rotation(tilt, 4, "Y")
            @ mathutils.Matrix.Scale(impression, 4)
        )
        if used[plate_index] == 0:
            obj = molecule.object
        else:
            # Linked: one mesh and one node tree per pigment, however many
            # impressions of it the sheet asked for. The ink is on the pigment,
            # so every copy of a plate is the same plate — which is what a
            # plate is.
            obj = molecule.object.copy()
            obj.data = molecule.object.data
            obj.name = f"{PLATES[plate_index][1]} impression {cell_index}"
            scene.collection.objects.link(obj)
        obj.matrix_world = placement @ bases[plate_index]
        used[plate_index] += 1
for plate_index, count in enumerate(used):
    if count == 0:  # pragma: no cover - only if TREATMENTS drops a plate
        molecules[plate_index].object.hide_render = True
print(
    "  "
    + ", ".join(
        f"{PLATES[index][1]} x{count}" for index, count in enumerate(used) if count
    )
)
print(f"  {sum(used)} impressions from {len(molecules)} meshes")


# ---------------------------------------------------------------------------
heading("9. Multiply the grid")
# ---------------------------------------------------------------------------
# The other half of her operation. The seven columns traced in step 4 are a
# motif the width of one receptor; repeated across the sheet at their own
# period they become a facade, and the slabs — the bilayer, once per band —
# weave the horizontals through them. Behind the plates, in grey, screened at
# 0 degrees so the field reads as a different pass from any of the pigments.


def printed_quads(name: str, rectangles, y: float, material) -> object:
    """One flat mesh of camera-facing rectangles, carrying one ink.

    Everything on this sheet that is not a molecule is a rectangle, and a
    rectangle seen through an orthographic camera square-on needs no thickness.
    One mesh for all of them, so the facade is one object to move.
    """
    vertices, faces = [], []
    for x0, x1, z0, z1 in rectangles:
        base = len(vertices)
        vertices += [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        faces.append((base, base + 1, base + 2, base + 3))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    scene.collection.objects.link(obj)
    return obj


# The motif, at the size the impressions are printed at, so a bar in the facade
# is the same width as the helix it was traced from and the two register.
bay = period * impression
reach = block_width / 2 + cell_width * 0.15
repeats = math.ceil(reach / bay) + 1 if bay > 0 else 1

bars = []
for repeat in range(-repeats, repeats + 1):
    for position, width in columns:
        x = (position - motif_centre) * impression + repeat * bay
        if abs(x) > reach:
            continue
        # Narrow. A bar as wide as the helix it came from closes the gaps
        # between its neighbours and the seven columns become one grey field,
        # which is a texture rather than a facade.
        half = max(width * impression * 0.34, cell_width * 0.006)
        bars.append((x - half, x + half, -block_height / 2, block_height / 2))
verticals = len(bars)

# The slabs, once per floor. For a membrane protein this is not a metaphor and
# not a drawn line: it is where the bilayer is, measured in step 4, and it does
# in this facade exactly what a floor plate does in Mies's.
rule = cell_height * 0.028
for band in range(BANDS):
    z = ((BANDS - 1) / 2 - band) * cell_height
    for edge in (slab_low, slab_high):
        line = z + edge * impression
        bars.append((-reach, reach, line - rule, line + rule))

facade = printed_quads(
    "GALA Facade",
    bars,
    stand * 0.06,
    build_screen("GALA Screen Facade", FIELD_INK, 0.0, RULING, 0.55, tone=0.34),
)
print(
    f"  {verticals} columns from {2 * repeats + 1} repeats of a "
    f"{period * 100:.0f} A motif, printed at {bay * 100:.0f} A"
)
print(
    f"  {len(bars) - verticals} slabs, {(slab_high - slab_low) * 100:.0f} A apart, "
    "once per floor"
)

# And one pass over the top: the same columns again, thinner, in front of
# everything. This is the weave — a grid that is only ever behind the picture
# is a background, and hers is not a background.
printed_quads(
    "GALA Facade Overprint",
    [
        (x0 + (x1 - x0) * 0.34, x1 - (x1 - x0) * 0.34, z0, z1)
        for x0, x1, z0, z1 in bars[:verticals]
    ],
    -stand * 0.04,
    build_screen("GALA Screen Overprint", FIELD_INK, 90.0, RULING, 0.30, tone=0.38),
)

# And under it, a flat tint in some of the bays. This is the last of her three
# moves — infusing colour and line weight, once the grid has been multiplied
# and woven — and it is what stops nine bays of one facade from being nine of
# the same picture. The tints are drawn from the plates' own inks, so nothing
# on this sheet is a colour that is not one of the four pigments or the paper.
TINTS = (
    None,
    (PLATES[2][3], 0.34),
    None,
    (PLATES[3][3], 0.24),
    None,
    (PLATES[1][3], 0.30),
    (FLOOD_INK, 0.95),
    None,
    (PLATES[2][3], 0.26),
)
fields: dict[tuple[str, float], list[tuple[float, float, float, float]]] = {}
for band in range(BANDS):
    for column in range(COLUMNS):
        index = band * COLUMNS + column
        spec = TINTS[index % len(TINTS)]
        if spec is None:
            continue
        x = (column - (COLUMNS - 1) / 2) * cell_width + band_shift[band] * sheet_width
        z = ((BANDS - 1) / 2 - band) * cell_height
        fields.setdefault(spec, []).append(
            (
                x - cell_width * 0.46,
                x + cell_width * 0.46,
                z - cell_height * 0.46,
                z + cell_height * 0.46,
            )
        )

for order, (spec, rectangles) in enumerate(fields.items()):
    printed_quads(
        f"GALA Tint {order}",
        rectangles,
        stand * 0.12,
        build_screen(
            f"GALA Screen Tint {order}", spec[0], 0.0, RULING, 0.70, tone=spec[1]
        ),
    )
print(
    f"  {sum(len(rects) for rects in fields.values())} bays tinted, {len(fields)} tints"
)


# ---------------------------------------------------------------------------
heading("10. The paper")
# ---------------------------------------------------------------------------
# Two passes before anything else went down: a coarse dot field over the whole
# sheet, and a pale flood coat over the middle of it that veils the field
# without covering it. Both are printing operations rather than lighting ones,
# which is the point — nothing in this scene is lit.
world = scene.world or bpy.data.worlds.new("GALA Halftone")
scene.world = world
world.use_nodes = True
tree = world.node_tree
tree.nodes.clear()
world_output = tree.nodes.new("ShaderNodeOutputWorld")
world_output.location = (300, 0)
paper = tree.nodes.new("ShaderNodeBackground")
paper.location = (0, 0)
paper.inputs["Color"].default_value = (*hex_to_rgb(PAPER), 1.0)
paper.inputs["Strength"].default_value = 1.0
tree.links.new(paper.outputs["Background"], world_output.inputs["Surface"])

printed_quads(
    "GALA Dot Field",
    [(-sheet_width, sheet_width, -sheet_height, sheet_height)],
    stand * 0.5,
    build_screen("GALA Screen Field", FIELD_INK, 0.0, FIELD_RULING, 0.9, tone=0.34),
)
flood = (
    -block_width / 2 - cell_width * 0.22,
    block_width / 2 + cell_width * 0.22,
    -block_height / 2 - cell_height * 0.20,
    block_height / 2 + cell_height * 0.20,
)
printed_quads(
    "GALA Flood Coat",
    [flood],
    stand * 0.28,
    # Tone above 1 is a solid: the dot has grown past the corners of its cell
    # and there is no cell left to see. Printers call that pass a flood coat,
    # and it is the same node graph as every other ink here with one number
    # turned up, which is the argument this whole vignette is making.
    build_screen("GALA Screen Flood", FLOOD_INK, 0.0, FIELD_RULING, 0.62, tone=2.0),
)
print(
    f"  paper {PAPER}, field at 34% on a {FIELD_RULING:.0f}-line screen, "
    f"flood coat solid over {(flood[1] - flood[0]) / sheet_width:.0%} of the width"
)


# ---------------------------------------------------------------------------
heading("11. Print it")
# ---------------------------------------------------------------------------
# Cycles, with nothing for it to do but resolve transparency: no lights, no
# bounces off anything, every surface either clear or emitting a flat colour.
# The one setting that matters is how many clear surfaces a ray may pass
# through before Cycles gives up, and a stack of four screened ribbons in front
# of a screened facade in front of two screened fields is a great many.
scene.cycles.transparent_max_bounces = 256
scene.cycles.max_bounces = 1
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.render.film_transparent = False
print(
    f"  {scene.cycles.samples} samples, {scene.cycles.transparent_max_bounces} transparent bounces"
)

render(gala, "17_halftone_facade")
print(
    "\n  Read it twice: a field of dots at arm's length, four pigments across the room."
)


# ---------------------------------------------------------------------------
heading("12. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# The four screen materials are the whole picture and they are four copies of
# one node graph. `Screen angle` on the Vector Rotate is what makes the plates
# separate; `Dot radius` and the ruling on the Vector Math above it are the
# medium's grain; `Ink opacity` is how wet the press is. Turn any of them and
# the same coordinates come back as a different print.
save_blend("17_halftone_facade")
