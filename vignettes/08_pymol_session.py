"""Vignette 8 — PyMOL sessions, in and out.

Most structural biologists already have a folder of ``.pse`` files. This is
how one becomes a Blender scene, and how a Blender scene becomes one back:

    blender --background --python vignettes/08_pymol_session.py

Nothing here needs PyMOL installed. A session is a pickled tree of plain
lists, so Gala reads and writes the format directly — which it has to, since
Blender's interpreter has no PyMOL in it.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import DATA_DIR, QUALITY, heading, load_structure, render, setup

mn, gala = setup()

import bpy

WORKDIR = tempfile.mkdtemp(prefix="gala-pymol-")


# ---------------------------------------------------------------------------
heading("1. Read a session PyMOL wrote")
# ---------------------------------------------------------------------------
# This one is committed with the tests and was written by PyMOL 3.1.8. Reading
# it needs neither PyMOL nor Blender: `read_session` returns plain arrays, so
# what a session contains can be inspected from any interpreter.
example = os.path.join(DATA_DIR, "session.pse")
session = gala.read_session(example)
print(session.summary())

site = session.find("site")
print(f"\n  representations shown: {', '.join(site.reps_present())}")
print(f"  atoms in the cartoon  : {int(site.rep_mask('cartoon').sum())}")
print(f"  distinct colours      : {len(set(site.color_index.tolist()))}")
print(
    f"  camera                : {session.view.distance:.0f} A away, "
    f"{session.view.field_of_view:.0f} degree lens, "
    f"{'orthographic' if session.view.orthoscopic else 'perspective'}"
)

# A .pse is a pickle, and a pickle can name anything it likes for the reader
# to import. Gala's reader refuses every global except the handful a genuine
# session contains, so opening one that came by email cannot run code.
print("\n  The reader allows these globals and no others:")
for module_name, attribute in sorted(gala.pymol.session.ALLOWED_GLOBALS):
    print(f"      {module_name}.{attribute}")


# ---------------------------------------------------------------------------
heading("2. Build a scene worth exporting")
# ---------------------------------------------------------------------------
# Adenylate kinase, closed around its inhibitor: a cartoon for the fold, the
# ligand in sticks, and the residues that touch it picked out.
mol = load_structure("1ake")
mol.add_style("cartoon", color=None)
# `is_hetero` is one of the boolean attributes Molecular Nodes writes on
# import, and a style limited to a named attribute is one Gala can read back
# when it exports — so the sticks stay on the ligand rather than spreading to
# the whole protein.
mol.add_style("ball_and_stick", selection="is_hetero", color=None)

gala.color_by_selection(
    mol,
    {
        "polymer": "#9fb4c7",
        "byres (polymer within 4.5 of ligand)": "#ffab3d",
        "ligand": "#e4572e",
    },
)

contacts = gala.find_interactions(mol, "ligand", "protein", kinds="all")
print(f"  {len(contacts)} interactions found")

gala.publication_setup(mol, preset=QUALITY, viewpoint="iso", material_scheme=None)


# ---------------------------------------------------------------------------
heading("3. Write it out as a session")
# ---------------------------------------------------------------------------
# Coordinates go out in world space, so a molecule moved in Blender arrives in
# PyMOL where it looks here rather than back where its file put it. Colours go
# per atom: the ones that are exactly a PyMOL colour are written as that
# colour, and the rest are defined in the session itself.
exported = os.path.join(WORKDIR, "from_blender.pse")
written = gala.save_session(exported)
print(written.summary())
print(f"\n  {os.path.getsize(exported) / 1024:.0f} kB, and PyMOL opens it directly.")


# ---------------------------------------------------------------------------
heading("4. Read it back, and check what survived")
# ---------------------------------------------------------------------------
# The scene is cleared first, so what follows is built entirely from the file.
before = gala.AtomStructure.from_any(mol)
before_atoms = before.n_atoms
before_positions = before.world_positions()
before_colors = np.array(gala.read_colors(mol))

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

result = gala.load_session(exported)
print(result.summary())

restored = next(iter(result.molecules.values()))
after = gala.AtomStructure.from_any(restored)
after_colors = np.array(gala.read_colors(restored))

drift = np.abs(after.world_positions() - before_positions).max()
print(f"\n  atoms      : {before_atoms} out, {after.n_atoms} back")
print(f"  positions  : largest drift {drift / after.world_scale:.4f} A")
print(f"  colours    : {np.abs(after_colors - before_colors).max():.4f} largest change")
print(f"  b-factors  : {np.isfinite(after.array.b_factor).all()} all finite")

# What a session cannot hold is worth being explicit about: materials,
# lighting, node trees, and any geometry that is not a molecule. Those are the
# parts of the scene that made you open Blender in the first place.
print("\n  Not written to the session: materials, lighting, compositing.")
print("  Those are what Blender is for; the session carries the science.")


# ---------------------------------------------------------------------------
heading("5. Render the scene that came out of the file")
# ---------------------------------------------------------------------------
# `load_session` set the camera from the session's view, so the framing below
# is the one that was exported rather than a fresh one.
gala.three_point_lighting(restored)
gala.setup_render(preset=QUALITY, transparent=True)
render(gala, "08_pymol_session")
