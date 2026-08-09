"""Vignette 14 — a camera move and a focus pull.

A talk usually needs the same structure twice: once wide, so the audience can
see what protein it is, and once close, so they can see the thing being argued
about. Cutting between two stills makes them re-find themselves; moving between
them does not.

The move is two camera poses and an interpolation, and Gala computes both poses
for you — `frame_target` already knows how to fit a molecule, and it takes a
`selection`, so "frame the whole kinase" and "frame the drug in its pocket" are
the same call twice. What this vignette does is record where the camera ended
up each time and keyframe between them.

Three things are animated, because a push-in that only moves is a zoom rather
than a shot:

* the camera's **position**, sampled along an arc about the molecule rather
  than interpolated straight from one pose to the other;
* the **aim and the focus**, both handed to a target that slides from the
  middle of the protein to the middle of the drug, so neither is interpolated
  and neither can drift off the subject;
* the **aperture**, opening from f/8 to f/4 as the shot closes in, so the
  wide frame is sharp front to back and the close-up throws the rest of the
  protein into a wash behind the ligand. f/4 rather than something wider
  because the lens is a 200 mm: depth of field falls off with focal length as
  well as with aperture, and at f/2 this shot has none at all.

The subject is the Abl kinase domain with imatinib bound (1IEP) — the drug
that made kinase inhibitors a category, caught in the inactive conformation it
selects for.

    blender --background --python vignettes/14_focus_pull.py

`make vignettes-focus-pull` renders the move.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, save_blend, setup

mn, gala = setup()

import bpy

from blender_gala.core.entity import AtomStructure

# Imatinib's PDB chemical component. One copy per chain in 1IEP; the shot
# closes in on the one in chain A.
LIGAND = "resn STI and chain A"
PROTEIN = "chain A and protein"

PROTEIN_COLOUR = "#8497a8"
POCKET_COLOUR = "#cfd9e2"
LIGAND_COLOUR = "#ffb347"

# One second wide, two seconds moving, one second held on the ligand.
FPS = 25
MOVE_FROM = 25
MOVE_TO = 75
END_FRAME = 100

WIDE_FSTOP = 8.0
CLOSE_FSTOP = 4.0


# ---------------------------------------------------------------------------
heading("1. The molecule, styled for both ends of the shot")
# ---------------------------------------------------------------------------
# The close-up has to survive being looked at from 20 A away, so the residues
# lining the pocket are drawn atom by atom. Everything else is cartoon, which
# is what the wide shot needs and what keeps the close-up's background from
# turning into a thicket of sticks.
molecule = load_structure("1iep")

pocket = f"byres ({PROTEIN} within 4.5 of ({LIGAND}))"
for name, selection in (
    ("chain_a", PROTEIN),
    ("pocket", pocket),
    ("ligand", LIGAND),
):
    attribute = molecule.object.data.attributes.new(name, "BOOLEAN", "POINT")
    attribute.data.foreach_set(
        "value", gala.select(molecule, selection).astype(bool).tolist()
    )

molecule.add_style("cartoon", selection="chain_a", color=None)
molecule.add_style("ball_and_stick", selection="pocket", color=None)
molecule.add_style("ball_and_stick", selection="ligand", color=None)

gala.color_by_selection(
    molecule,
    {
        "all": PROTEIN_COLOUR,
        pocket: POCKET_COLOUR,
        LIGAND: LIGAND_COLOUR,
        f"({LIGAND}) and elem N": "#4f6fd8",
        f"({LIGAND}) and elem O": "#e0402a",
    },
)

structure = AtomStructure.from_any(molecule)
print(f"  {structure.n_atoms} atoms; imatinib is {gala.select(molecule, LIGAND).sum()}")
print(
    f"  pocket: {len(set(np.asarray(structure.context.annotation('res_id'))[structure.select(pocket)]))} residues within 4.5 A"
)


# ---------------------------------------------------------------------------
heading("2. Two poses, both computed rather than placed")
# ---------------------------------------------------------------------------
gala.publication_setup(
    molecule,
    preset=QUALITY,
    lighting_style="three_point",
    material_scheme="chemistry",
    origin_method="centroid",
    viewpoint="iso",
    cryptomatte=False,
)

scene = bpy.context.scene
camera = scene.camera

# One focal length for the whole shot, set before either pose is computed,
# because a move that also changes lens is a zoom and reads as one. A long one:
# at 85 mm the close-up stands so near the pocket that half the domain is
# between the camera and the drug, and at f/4 that half is a wash of blurred
# ribbon across the frame. 200 mm buys the same framing from three times the
# distance, which puts most of that foreground outside the cone entirely.
camera.data.lens = 200.0


def pose(viewpoint, margin: float, selection: str | None = None):
    """Aim the camera, and remember where that put it.

    The distance is measured to the middle of whatever was framed rather than
    to the origin: the ligand sits several angstrom off the molecule's
    centroid, and at a wide aperture several angstrom is the difference between
    a sharp drug and a sharp piece of protein behind it.
    """
    gala.frame_target(molecule, viewpoint=viewpoint, margin=margin, selection=selection)
    subject = structure.world_positions()
    if selection is not None:
        subject = subject[structure.select(selection)]
    distance = float(np.linalg.norm(np.array(camera.location) - subject.mean(axis=0)))
    return (
        camera.location.copy(),
        camera.rotation_euler.copy(),
        distance,
        (camera.data.clip_start, camera.data.clip_end),
    )


# The wide shot: the whole kinase domain, seen from an angle that keeps the
# cleft between the two lobes open rather than edge on. Framed on chain A
# rather than on everything, because 1IEP holds two copies of the domain and
# only one of them is being drawn — frame both and the shot is mostly a gap
# where the other one would be.
wide = pose((28.0, 14.0), margin=1.18, selection=PROTEIN)

# The close-up: the same molecule, framed on 37 atoms of drug.
#
# The angle is the whole shot. Imatinib is a long molecule threaded through the
# cleft between the two lobes, and its *shape* is the argument — it is what
# makes the drug specific for the inactive conformation. Seen from anywhere
# near the wide angle it is end on, and a 20 A molecule pointing at the camera
# projects to an orange knot. Swung round to the left it lies across the frame
# at full length, with the N-lobe above it and the activation loop below, which
# is the geometry worth pushing in on. The low elevation is the other half of
# it: from higher up the helices of the N-lobe cross in front of the end of the
# molecule, and from down here they sit clear above it. Two more things come
# free with the swing — fewer atoms sit in the cone between camera and pocket,
# and 73 degrees of arc gives the move real parallax rather than a zoom.
close = pose((-45.0, 10.0), margin=1.6, selection=LIGAND)

print(f"  wide : {tuple(round(v, 3) for v in wide[0])}, {wide[2]:.3f} units out")
print(f"  close: {tuple(round(v, 3) for v in close[0])}, {close[2]:.3f} units out")
print(
    f"  the camera travels {np.linalg.norm(np.array(wide[0]) - np.array(close[0])):.3f} units"
)

# `frame_target` sets the near and far clipping planes to bracket the molecule
# *from the pose it just computed*, which is right for a still and wrong for
# every frame of a move except the last one. Left alone, the wide shot renders
# empty: the far plane is the close-up's, a couple of units out, and the wide
# camera stands more than twice that far back — the whole molecule is behind
# it. The range has to cover both ends.
camera.data.clip_start = min(wide[3][0], close[3][0])
camera.data.clip_end = max(wide[3][1], close[3][1])
print(
    f"  clipping {camera.data.clip_start:.3f} to {camera.data.clip_end:.3f}, "
    "covering both poses"
)


# ---------------------------------------------------------------------------
heading("3. Hand the aim to a target, so it is never interpolated")
# ---------------------------------------------------------------------------
# Keyframing the camera's rotation at the two ends and letting Blender fill in
# between is the obvious thing to do, and on a swing this wide it does not
# work. Position interpolates along the chord between the poses, which cuts
# well inside the arc; orientation interpolates separately as three Euler
# angles. Neither knows about the other, so half way through the move the
# camera is somewhere the rotation was never computed for, pointing at nothing
# in particular — the molecule swings out of frame and comes back.
#
# A Track To constraint removes the question. The camera's rotation is not
# animated at all: it is derived, every frame, from where the camera is and
# where the target is. There is no interpolation left to go wrong.
target = bpy.data.objects.new("GALA Shot Target", None)
target.empty_display_size = 0.05
scene.collection.objects.link(target)

aim = camera.constraints.new("TRACK_TO")
aim.target = target
aim.track_axis = "TRACK_NEGATIVE_Z"
aim.up_axis = "UP_Y"

# And the same target takes the focus, which is the other half of the problem:
# a keyframed focus distance is only right if the camera is looking where the
# distance was measured to. Focused on an object, it cannot drift.
camera.data.dof.use_dof = True
camera.data.dof.focus_object = target

protein_centre = structure.world_positions()[structure.select(PROTEIN)].mean(axis=0)
ligand_centre = structure.world_positions()[structure.select(LIGAND)].mean(axis=0)
print(
    f"  the target moves {np.linalg.norm(ligand_centre - protein_centre) * 100:.1f} A"
)
print("  camera rotation: not animated; the constraint derives it every frame")


# ---------------------------------------------------------------------------
heading("4. A path along the arc rather than across it")
# ---------------------------------------------------------------------------
# With the aim taken care of, what is left is where the camera stands. Two
# keys still cut the corner, so the move is sampled: the viewing *direction* is
# slerped around the sphere, the distance is interpolated geometrically —
# halving twice reads as an even approach where subtracting a constant twice
# does not — and the target slides from the middle of the protein to the middle
# of the drug. Every sample is therefore a pose that would have framed
# correctly on its own.
start_offset = np.array(wide[0]) - protein_centre
finish_offset = np.array(close[0]) - ligand_centre
start_distance = float(np.linalg.norm(start_offset))
finish_distance = float(np.linalg.norm(finish_offset))
start_direction = start_offset / start_distance
finish_direction = finish_offset / finish_distance

swing = float(np.arccos(np.clip(start_direction @ finish_direction, -1.0, 1.0)))
print(f"  the camera swings {np.degrees(swing):.1f} degrees about the molecule")


def along_the_arc(fraction: float):
    """Camera position and target at ``fraction`` of the way through the move."""
    if swing < 1e-6:  # pragma: no cover - the two poses coincide
        direction = start_direction
    else:
        direction = (
            np.sin((1.0 - fraction) * swing) * start_direction
            + np.sin(fraction * swing) * finish_direction
        ) / np.sin(swing)
    distance = start_distance * (finish_distance / start_distance) ** fraction
    centre = protein_centre + fraction * (ligand_centre - protein_centre)
    return centre + direction * distance, centre


# The easing is baked into *which* fractions are sampled rather than left to
# the F-curve handles. Smoothstep accelerates and decelerates once across the
# whole move; handles on twelve keys would ease into and out of every one of
# them, which is a stutter rather than a move.
def smoothstep(fraction: float) -> float:
    return fraction * fraction * (3.0 - 2.0 * fraction)


SAMPLES = 12
keys = [(1, 0.0), (END_FRAME, 1.0)]
keys += [
    (
        round(MOVE_FROM + (MOVE_TO - MOVE_FROM) * index / (SAMPLES - 1)),
        smoothstep(index / (SAMPLES - 1)),
    )
    for index in range(SAMPLES)
]

clearance = []
for frame, fraction in sorted(keys):
    position, centre = along_the_arc(fraction)
    camera.location = tuple(float(v) for v in position)
    target.location = tuple(float(v) for v in centre)
    camera.data.dof.aperture_fstop = WIDE_FSTOP + fraction * (CLOSE_FSTOP - WIDE_FSTOP)
    camera.keyframe_insert("location", frame=frame)
    target.keyframe_insert("location", frame=frame)
    camera.data.dof.keyframe_insert("aperture_fstop", frame=frame)
    clearance.append(float(np.linalg.norm(position - protein_centre)))

scene.frame_start = 1
scene.frame_end = END_FRAME
scene.render.fps = FPS

# The path has to stay outside the molecule the whole way, not just at the two
# ends — which the chord between the poses would not have.
_, protein_radius = structure.bounding_sphere()
print(f"  {len(keys)} keys; the camera stays {min(clearance):.3f} units out at closest")
print(f"  the molecule's radius is {protein_radius:.3f} units")
if min(clearance) <= protein_radius:
    raise SystemExit("the camera path enters the molecule; move the poses")

# Linear between the samples, because the samples already carry the shape of
# the move. Bezier handles here would smooth a curve that is not meant to be
# smoothed and let the path bulge off the arc between keys.
from blender_gala.scene.camera import _action_fcurves

for animated in (camera, camera.data, target):
    action = animated.animation_data.action
    for curve in _action_fcurves(action):
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"
        curve.update()


# ---------------------------------------------------------------------------
heading("5. Both ends of the shot")
# ---------------------------------------------------------------------------
scene.frame_set(1)
render(gala, "14_focus_wide")
scene.frame_set(END_FRAME)
render(gala, "14_focus_pull")
scene.frame_set(1)


# ---------------------------------------------------------------------------
# The move itself, when asked for it
# ---------------------------------------------------------------------------
frames_dir = os.environ.get("GALA_FOCUS_DIR")
if frames_dir:
    heading("6. Render the move")
    STEP = int(os.environ.get("GALA_FOCUS_STEP", "2"))
    SIZE = int(os.environ.get("GALA_FOCUS_SIZE", "480"))
    scene.render.resolution_x = scene.render.resolution_y = SIZE

    # Frames are intermediates, and stay lossless PNG: `make_animation`
    # re-encodes them into an animated WebP, and compressing lossily twice is
    # visibly worse than doing it once. Set explicitly rather than inherited,
    # because a still rendered earlier in the script leaves the scene in WebP
    # and these would then be WebP bytes under a .png name.
    gala.scene.render.set_image_format(
        scene.render.image_settings, "PNG", color_mode="RGBA"
    )

    os.makedirs(frames_dir, exist_ok=True)
    wanted = range(1, END_FRAME + 1, STEP)
    for index, frame in enumerate(wanted):
        scene.frame_set(frame)
        gala.render(os.path.join(frames_dir, f"{index:03d}.png"))
    print(f"  {len(wanted)} frames at {SIZE}x{SIZE} in {frames_dir}")
    scene.frame_set(1)


# ---------------------------------------------------------------------------
heading("7. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# The move is in two places: the keys on the camera and its target, and the
# Track To constraint that turns the two into a shot. Move the target and the
# camera re-aims itself; drag the first and last keys to change when the move
# starts and stops. To re-time it without re-deriving the path, scale the keys
# in the graph editor — they carry the easing between them, so stretching them
# stretches the whole move rather than flattening its ends.
save_blend("14_focus_pull")
