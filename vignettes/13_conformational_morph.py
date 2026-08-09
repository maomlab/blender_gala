"""Vignette 13 — a conformational change, animated with shape keys.

Adenylate kinase is the textbook induced fit. Apo, its LID and NMP-binding
domains stand open away from the CORE; with substrate bound they fold down over
it and close the active site. The PDB has both ends of that motion — 4AKE open,
1AKE closed — and nothing in between, because there is nothing in between to
crystallise.

Blender has a tool for exactly this shape: a **shape key**. Give the mesh a
second set of vertex positions and a slider between them, and every frame is an
interpolation. Because shape keys are evaluated *before* modifiers, Molecular
Nodes then rebuilds the cartoon from the interpolated coordinates on every
frame — the ribbon is re-derived, not deformed, so the secondary structure
stays a ribbon all the way through.

What Gala contributes is the part that decides whether the animation means
anything: superposing the two structures on the CORE domain alone. Superpose on
everything and the CORE swims while the LID stays put, which is the same motion
seen from a moving chair. Superpose on the CORE and what is left moving is what
actually moves.

The linear path between two endpoints is *not* the reaction coordinate — the
real trajectory bends, and no morph should be read as one. It is a way to see
which parts move and how far, which two still pictures side by side never quite
show.

    blender --background --python vignettes/13_conformational_morph.py

`make vignettes-morph` renders the animation.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, save_blend, setup

mn, gala = setup()

import bpy
import mathutils

from blender_gala.core.entity import AtomStructure
from blender_gala.core.units import world_scale_of

# The three domains, in the residue ranges the adenylate kinase literature
# uses (Henzler-Wildman et al., Nature 450, 838 (2007); Beckstein et al.,
# J Mol Biol 394, 160 (2009)).
CORE = "chain A and resi 1-29+60-121+160-214"
NMP = "chain A and resi 30-59"
LID = "chain A and resi 122-159"

# Warm for the two lids that move, cool for the CORE they move against.
CORE_COLOUR = "#7d93a6"
NMP_COLOUR = "#f0a03c"
LID_COLOUR = "#e2585f"

# The distance the closure is usually quoted by: one alpha carbon on the LID,
# one on the NMP-binding domain, closing across the cleft.
LID_ATOM = "chain A and resi 142 and name CA"
NMP_ATOM = "chain A and resi 55 and name CA"

# 25 fps: one second open, one closing, one closed, one opening again, so the
# animation loops without a jump.
FPS = 25
CLOSED_FRAME = 50
END_FRAME = 100


# ---------------------------------------------------------------------------
heading("1. Both ends of the motion")
# ---------------------------------------------------------------------------
opened = load_structure("4ake")
closed = load_structure("1ake")

open_structure = AtomStructure.from_any(opened)
closed_structure = AtomStructure.from_any(closed)
SCALE = world_scale_of(opened.object)
print(f"  4AKE open  : {open_structure.n_atoms} atoms")
print(f"  1AKE closed: {closed_structure.n_atoms} atoms")

# One monomer. Both entries hold two copies of the chain in the asymmetric
# unit, and superposing on chain A's CORE leaves chain B related to it by
# whatever the two crystals happened to do — a rigid-body swing that is
# crystallography, not induced fit. A style's selection names a boolean
# attribute, so chain A is written as one.
chain_a = opened.object.data.attributes.new("chain_a", "BOOLEAN", "POINT")
chain_a.data.foreach_set(
    "value", gala.select(opened, "chain A and protein").astype(bool).tolist()
)

# Only the open one is drawn. The closed one is a bag of coordinates to aim at,
# and is never styled or rendered.
opened.add_style("cartoon", selection="chain_a", color=None)
gala.color_by_selection(opened, {CORE: CORE_COLOUR, NMP: NMP_COLOUR, LID: LID_COLOUR})


# ---------------------------------------------------------------------------
heading("2. The same atom in both structures")


# ---------------------------------------------------------------------------
# Matched by chain, residue number and atom name rather than by index: the two
# entries differ in their waters, their hetero atoms and a couple of side
# chains that were not modelled, so index *i* is not the same atom in both.
def atom_keys(structure: AtomStructure) -> dict[tuple[str, int, str], int]:
    array = structure.array
    return {
        (str(chain), int(residue), str(atom)): index
        for index, (chain, residue, atom) in enumerate(
            zip(array.chain_id, array.res_id, array.atom_name, strict=True)
        )
    }


open_index = atom_keys(open_structure)
closed_index = atom_keys(closed_structure)
shared = sorted(key for key in set(open_index) & set(closed_index) if key[0] == "A")
print(f"  {len(shared)} chain A atoms in both, of {open_structure.n_atoms} in 4AKE")


# ---------------------------------------------------------------------------
heading("3. Superpose on the CORE, so the CORE is what holds still")
# ---------------------------------------------------------------------------
core_mask = open_structure.select(f"({CORE}) and name CA")
core_residues = {
    int(residue)
    for residue in np.asarray(open_structure.context.annotation("res_id"))[core_mask]
}
core_pairs = [key for key in shared if key[2] == "CA" and key[1] in core_residues]
mobile = np.array([closed_structure.array.coord[closed_index[k]] for k in core_pairs])
target = np.array([open_structure.array.coord[open_index[k]] for k in core_pairs])
print(f"  {len(core_pairs)} CORE alpha carbons to superpose on")


def kabsch(mobile_points: np.ndarray, target_points: np.ndarray):
    """Rotation and shift putting ``mobile_points`` onto ``target_points``."""
    mobile_centre = mobile_points.mean(axis=0)
    target_centre = target_points.mean(axis=0)
    correlation = (mobile_points - mobile_centre).T @ (target_points - target_centre)
    left, _, right = np.linalg.svd(correlation)
    sign = float(np.sign(np.linalg.det(right.T @ left.T)))
    rotation = right.T @ np.diag([1.0, 1.0, sign]) @ left.T
    return rotation, target_centre - rotation @ mobile_centre


rotation, shift = kabsch(mobile, target)
moved = (rotation @ mobile.T).T + shift
print(
    f"  CORE RMSD after superposing: {np.sqrt(((moved - target) ** 2).sum(1).mean()):.2f} A"
)

# Every closed-state coordinate, brought into the open state's frame.
closed_in_frame = (rotation @ closed_structure.array.coord.T).T + shift

# How far each domain travels once the CORE is held still — which is the whole
# argument for superposing this way rather than on everything.
for name, selection in (("CORE", CORE), ("NMP", NMP), ("LID", LID)):
    mask = open_structure.select(f"({selection}) and name CA")
    residues = {
        int(residue)
        for residue in np.asarray(open_structure.context.annotation("res_id"))[mask]
    }
    keys = [k for k in shared if k[2] == "CA" and k[1] in residues]
    start = np.array([open_structure.array.coord[open_index[k]] for k in keys])
    finish = np.array([closed_in_frame[closed_index[k]] for k in keys])
    travel = np.linalg.norm(finish - start, axis=1)
    print(
        f"  {name:5s}: {len(keys):3d} residues, "
        f"mean {travel.mean():5.2f} A, furthest {travel.max():5.2f} A"
    )


# ---------------------------------------------------------------------------
heading("4. Turn the molecule so the motion happens across the frame")
# ---------------------------------------------------------------------------
# A crystal frame has nothing to do with what moves in it, and a domain closing
# straight towards the camera closes by a few pixels. So the frame is built out
# of the motion: the LID's mean displacement laid across the picture, the
# CORE's long axis stood up the picture, and the camera left looking squarely
# down what is left.
lid_mask = open_structure.select(f"({LID}) and name CA")
lid_residues = {
    int(residue)
    for residue in np.asarray(open_structure.context.annotation("res_id"))[lid_mask]
}
lid_keys = [key for key in shared if key[2] == "CA" and key[1] in lid_residues]
displacement = np.array(
    [
        closed_in_frame[closed_index[k]] - open_structure.array.coord[open_index[k]]
        for k in lid_keys
    ]
).mean(axis=0)
across = displacement / np.linalg.norm(displacement)

core_points = np.array([open_structure.array.coord[open_index[k]] for k in core_pairs])
_, _, components = np.linalg.svd(core_points - core_points.mean(axis=0))
up = components[0] - np.dot(components[0], across) * across
up /= np.linalg.norm(up)

frame_axes = np.array([across, np.cross(up, across), up])
chain_centre = np.array(
    [open_structure.array.coord[open_index[k]] for k in shared]
).mean(axis=0)

stand = mathutils.Matrix.Identity(4)
for row in range(3):
    for column in range(3):
        stand[row][column] = float(frame_axes[row][column])
stand.translation = mathutils.Vector((-frame_axes @ chain_centre * SCALE).tolist())
opened.object.matrix_world = stand @ opened.object.matrix_world
bpy.context.view_layer.update()
print(f"  the LID travels {np.linalg.norm(displacement):.1f} A, now left to right")


# ---------------------------------------------------------------------------
heading("5. A shape key, and a slider between the two")
# ---------------------------------------------------------------------------
# The basis key is the mesh as it stands, which is the open structure. The
# second key holds the closed positions for every atom the two share; atoms
# that exist only in 4AKE keep their basis position and simply do not move,
# which is invisible in a cartoon because a cartoon is drawn from the backbone
# and every backbone atom is shared.
obj = opened.object
obj.shape_key_add(name="Open", from_mix=False)
closing = obj.shape_key_add(name="Closed", from_mix=False)

for key in shared:
    closing.data[open_index[key]].co = closed_in_frame[closed_index[key]] * SCALE

# Bezier interpolation, which is Blender's default: the motion eases in and out
# rather than starting and stopping at full speed. A protein does not, but an
# animation that does reads as a jump cut.
for frame, value in ((1, 0.0), (CLOSED_FRAME, 1.0), (END_FRAME, 0.0)):
    closing.value = value
    closing.keyframe_insert("value", frame=frame)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = END_FRAME
scene.render.fps = FPS
scene.frame_set(1)
print(f"  shape keys: {[k.name for k in obj.data.shape_keys.key_blocks]}")
print(f"  {scene.frame_start}-{scene.frame_end} at {FPS} fps")


# ---------------------------------------------------------------------------
heading("6. What the cleft does on the way")
# ---------------------------------------------------------------------------
# Measured on the interpolation itself rather than on the rendered mesh: the
# mesh at frame *n* is the styled cartoon, and the number worth quoting is
# between two alpha carbons.
lid_atom = np.flatnonzero(open_structure.select(LID_ATOM))[0]
nmp_atom = np.flatnonzero(open_structure.select(NMP_ATOM))[0]
lid_key = next(k for k, v in open_index.items() if v == lid_atom)
nmp_key = next(k for k, v in open_index.items() if v == nmp_atom)


def cleft_at(fraction: float) -> float:
    """LID-to-NMP alpha carbon distance at ``fraction`` of the way closed."""

    def position(key):
        start = open_structure.array.coord[open_index[key]]
        finish = closed_in_frame[closed_index[key]]
        return start + fraction * (finish - start)

    return float(np.linalg.norm(position(lid_key) - position(nmp_key)))


for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
    bar = "=" * int(cleft_at(fraction))
    print(f"  {fraction * 100:5.0f}% closed   {cleft_at(fraction):5.2f} A  {bar}")


# ---------------------------------------------------------------------------
heading("7. One camera for the whole animation")
# ---------------------------------------------------------------------------
# Framed head on in the frame just built, and on the open state, which is the
# larger of the two, so nothing leaves the picture as it closes. The mesh
# itself is never moved — only the object's matrix — because shifting the
# basis vertices would shift them out from under the shape key.
gala.setup_render(preset=QUALITY, transparent=True)
gala.scene.render.setup_color_management()
gala.three_point_lighting(opened, energy=1.05, softness=1.3)
gala.frame_target(opened, viewpoint=(0.0, 8.0), margin=1.1, selection="chain A")
gala.assign_materials(opened, scheme="chemistry")

# The closed structure has done its job.
closed.object.hide_render = True
closed.object.hide_viewport = True


# ---------------------------------------------------------------------------
heading("8. The two ends, from the same camera")
# ---------------------------------------------------------------------------
scene.frame_set(1)
render(gala, "13_morph_open")
scene.frame_set(CLOSED_FRAME)
render(gala, "13_morph_closed")
scene.frame_set(1)


# ---------------------------------------------------------------------------
# The whole motion, when asked for it
# ---------------------------------------------------------------------------
# Off unless GALA_MORPH_DIR names somewhere to put the frames, for the same
# reason the turntable is: CI runs every vignette on every push, and a hundred
# Cycles frames is not a smoke test. `make vignettes-morph` sets it.
frames_dir = os.environ.get("GALA_MORPH_DIR")
if frames_dir:
    heading("9. Render the motion")
    STEP = int(os.environ.get("GALA_MORPH_STEP", "2"))
    SIZE = int(os.environ.get("GALA_MORPH_SIZE", "480"))
    scene.render.resolution_x = scene.render.resolution_y = SIZE

    os.makedirs(frames_dir, exist_ok=True)
    wanted = range(1, END_FRAME + 1, STEP)
    for index, frame in enumerate(wanted):
        scene.frame_set(frame)
        gala.render(os.path.join(frames_dir, f"{index:03d}.png"))
    print(f"  {len(wanted)} frames at {SIZE}x{SIZE} in {frames_dir}")
    scene.frame_set(1)


# ---------------------------------------------------------------------------
heading("10. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# The slider is in the Object Data properties, under Shape Keys. Scrub the
# timeline, or drag `Closed` by hand and watch the cartoon rebuild — which is
# the thing worth seeing, because it is what tells you the ribbon is being
# recomputed rather than stretched.
save_blend("13_conformational_morph")
