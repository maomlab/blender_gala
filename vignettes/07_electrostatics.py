"""Vignette 7 — electrostatics, solved rather than eyeballed.

Barnase cuts RNA; barstar stops it by binding to its active site with one of
the tightest protein-protein affinities known. The reason is visible as soon
as the potential is on the surface: barnase's active site is a patch of
positive, barstar's binding face is a patch of negative, and the two are
shaped like each other.

    Schreiber & Fersht. Rapid, electrostatically assisted association of
    proteins. Nat Struct Biol 3, 427-431 (1996).

This is what the PyMOL APBS plugin does, in Blender: PDB2PQR assigns charges
and radii, APBS solves the Poisson-Boltzmann equation on a grid around the
molecule, and the potential is painted onto a translucent molecular surface.

Needs `apbs` and `pdb2pqr`. Both install with pip::

    pip install apbs-binary pdb2pqr

    blender --background --python vignettes/07_electrostatics.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import QUALITY, heading, load_structure, render, setup

mn, gala = setup()

import bpy

from blender_gala.core.entity import AtomStructure
from blender_gala.electrostatics.apbs import ApbsUnavailable

#: 1BRS holds three copies of the complex; A/D is one of them.
BARNASE = "A"
BARSTAR = "D"

#: kT/e. Where the ramp saturates. The conventional figure quotes 5; these
#: two are small, mostly-neutral proteins whose interesting patches sit around
#: 2-4, and at 5 the whole surface comes out pale. The number is a display
#: choice and belongs in the caption, which is why it is printed below.
RAMP = 3.0

#: Ångström. Two atoms this close, one from each partner, are interface.
CONTACT = 4.5

WORKDIR = os.environ.get("GALA_APBS_DIR") or tempfile.mkdtemp(prefix="gala-apbs-")


# ---------------------------------------------------------------------------
heading("1. Two proteins that are shaped like each other's charge")
# ---------------------------------------------------------------------------
# Each partner is solved on its own. The potential barstar approaches is the
# one barnase makes by itself, not the one the finished complex settles into,
# and it is the approach that the association rate is about.
complex_structure = AtomStructure.from_any(load_structure("1brs"))
array = complex_structure.array


def partner(chain: str, name: str):
    """Write one chain out and load it back as a molecule of its own."""
    from biotite.structure.io.pdb import PDBFile

    subset = array[(array.chain_id == chain) & (array.element != "H")]
    path = os.path.join(WORKDIR, f"{name}.pdb")
    pdb = PDBFile()
    pdb.set_structure(subset)
    pdb.write(path)

    molecule = mn.Molecule.load(path)
    molecule.object.name = name
    return molecule, subset


# The whole complex was only ever a source of coordinates.
bpy.data.objects.remove(complex_structure.object, do_unlink=True)

barnase, barnase_atoms = partner(BARNASE, "barnase")
barstar, barstar_atoms = partner(BARSTAR, "barstar")
print(f"  barnase: {barnase_atoms.array_length()} atoms, chain {BARNASE}")
print(f"  barstar: {barstar_atoms.array_length()} atoms, chain {BARSTAR}")


# ---------------------------------------------------------------------------
heading("2. PDB2PQR and APBS")
# ---------------------------------------------------------------------------
# Everything about the calculation is a keyword: the force field the charges
# come from, the salt the solvent has in it, the two dielectrics. They are
# arguments rather than a dialog box, which is what makes the figure something
# a reader can reproduce.
try:
    runs = {
        "barnase": gala.run_apbs(
            barnase,
            workdir=os.path.join(WORKDIR, "barnase"),
            forcefield="AMBER",
            ionic_strength=0.15,
        ),
        "barstar": gala.run_apbs(
            barstar,
            workdir=os.path.join(WORKDIR, "barstar"),
            forcefield="AMBER",
            ionic_strength=0.15,
        ),
    }
except ApbsUnavailable as exc:
    raise SystemExit(f"\n{exc}\n") from exc

for name, run in runs.items():
    print(f"  {name}")
    print("    " + run.summary().replace("\n", "\n    "))


# ---------------------------------------------------------------------------
heading("3. The potential, on the surface")
# ---------------------------------------------------------------------------
# Translucent, so the partner behind shows through, and so the far side of
# each surface reads as a surface rather than as a silhouette.
surfaces = {
    name: gala.electrostatic_surface(
        molecule,
        grid=runs[name].grid,
        ramp=RAMP,
        alpha=0.68,
        quality=4,
        # Matte rather than the preset's wet look: a glossy translucent
        # surface reflects enough of the key light to bleach the ramp.
        material_options={"roughness": 0.45, "specular": 0.25},
    )
    for name, molecule in (("barnase", barnase), ("barstar", barstar))
}
for name, surface in surfaces.items():
    print(f"  {name}")
    print("    " + surface.summary().replace("\n", "\n    "))


# ---------------------------------------------------------------------------
heading("4. The interface is not the average of the surface")
# ---------------------------------------------------------------------------
# The picture says complementary; this says by how much. Each partner's
# interface is the atoms within contact range of the other, and the number is
# the mean potential there against the mean over the rest of its surface.


def interface_mask(atoms, other) -> np.ndarray:
    """Atoms of ``atoms`` within :data:`CONTACT` of any atom of ``other``."""
    from biotite.structure import CellList

    neighbours = CellList(other, cell_size=CONTACT).get_atoms(atoms.coord, CONTACT)
    return (neighbours >= 0).any(axis=1)


for name, atoms, other in (
    ("barnase", barnase_atoms, barstar_atoms),
    ("barstar", barstar_atoms, barnase_atoms),
):
    potential = surfaces[name].potential
    at_interface = interface_mask(atoms, other) & np.isfinite(potential)
    elsewhere = ~interface_mask(atoms, other) & np.isfinite(potential)
    print(
        f"  {name:8} interface {potential[at_interface].mean():+6.2f} kT/e "
        f"over {int(at_interface.sum()):3d} atoms, "
        f"rest of surface {potential[elsewhere].mean():+6.2f} kT/e"
    )


# ---------------------------------------------------------------------------
heading("5. Render")
# ---------------------------------------------------------------------------
# Look at the complex across the interface: the axis between the two centres
# of mass runs across the frame, so both faces are in view and the join
# between them is in the middle of it.


def viewpoint_across(axis: np.ndarray) -> tuple[float, float]:
    """Azimuth and elevation of a view direction perpendicular to ``axis``."""
    up = np.array([0.0, 0.0, 1.0])
    direction = np.cross(axis, up)
    if np.linalg.norm(direction) < 1e-6:  # axis is vertical; any view will do
        direction = np.array([0.0, -1.0, 0.0])
    direction /= np.linalg.norm(direction)
    azimuth = np.degrees(np.arctan2(direction[0], -direction[1]))
    elevation = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    return float(azimuth), float(elevation)


axis = barstar_atoms.coord.mean(axis=0) - barnase_atoms.coord.mean(axis=0)
viewpoint = viewpoint_across(axis)
print(f"  looking across the interface from ({viewpoint[0]:.0f}, {viewpoint[1]:.0f})")


def open_the_book(molecules, axis: np.ndarray, gap: float) -> None:
    """Swing both partners open so each shows the face the other binds.

    Docked, each interface is behind the partner covering it and the figure
    shows two surfaces touching. Rotating each partner ninety degrees about
    the view's up axis turns both interfaces towards the camera, which is what
    makes the complementarity — a positive patch the shape of a negative one —
    something the reader can see rather than take on trust.
    """
    import mathutils

    normal = axis / np.linalg.norm(axis)
    view = np.cross(normal, [0.0, 0.0, 1.0])
    view /= np.linalg.norm(view)
    up = np.cross(view, normal)
    # Slide along the horizontal of the frame rather than along the interface
    # normal: the normal lies in the frame but tilts out of horizontal, and
    # the two would separate diagonally across a mostly empty picture.
    sideways = np.cross([0.0, 0.0, 1.0], view)
    sideways /= np.linalg.norm(sideways)

    # Levelled along the frame's vertical, which is world +Z: the camera sits
    # in the horizontal plane and keeps its head up, so that is the direction
    # "higher in the picture" means.
    vertical = np.array([0.0, 0.0, 1.0])
    heights = [
        float(
            np.dot(AtomStructure.from_any(m).world_positions().mean(axis=0), vertical)
        )
        for m in molecules
    ]
    level = sum(heights) / len(heights)

    for molecule, turn, slide, height in zip(
        molecules, (90.0, -90.0), (-1.0, 1.0), heights, strict=True
    ):
        obj = molecule.object
        centre = np.array(
            AtomStructure.from_any(molecule).world_positions().mean(axis=0)
        )
        rotation = mathutils.Matrix.Rotation(
            np.radians(turn), 4, mathutils.Vector(up.tolist())
        )
        pivot = mathutils.Matrix.Translation(mathutils.Vector(centre.tolist()))
        # Sideways to separate them, and up or down to put both faces at the
        # same height: they were docked at an angle, and an open book that
        # runs diagonally across the frame wastes two corners of it.
        shift = sideways * slide * gap * scale + vertical * (level - height)
        offset = mathutils.Matrix.Translation(mathutils.Vector(shift.tolist()))
        obj.matrix_world = (
            offset @ pivot @ rotation @ pivot.inverted() @ obj.matrix_world
        )
    bpy.context.view_layer.update()


# Blender units, not ångström: the molecules are already in the scene.
scale = AtomStructure.from_any(barnase).world_scale
radius = max(
    np.linalg.norm(atoms.coord - atoms.coord.mean(axis=0), axis=1).max()
    for atoms in (barnase_atoms, barstar_atoms)
)
open_the_book((barnase, barstar), axis, gap=radius * 0.3)

gala.publication_setup(
    barnase,
    preset=QUALITY,
    lighting_style="three_point",
    # The materials are the ones the surfaces were given; a scheme here would
    # overwrite the translucency that is the point of the figure.
    material_scheme=None,
    viewpoint=viewpoint,
    # Both partners have to keep the coordinates they were docked in.
    origin_method=None,
)
# One label each, carrying the net charge PDB2PQR assigned: the patches are
# local, the charges are not, and the figure is about both.
for name, molecule in (("barnase", barnase), ("barstar", barstar)):
    gala.label(
        molecule,
        text=f"{name}  {runs[name].net_charge:+.0f} e",
        level="selection",
        style="card",
        size=3.2,
        offset=(0.0, 0.0, 26.0),
        avoid_occlusion=False,
    )

# Framing measures atoms, and the surface stands a probe radius and a bit
# further out than the outermost of them, so the margin has to cover the skin
# as well as the labels.
gala.frame_target(margin=1.2, viewpoint=viewpoint)
render(gala, "07_electrostatics")
