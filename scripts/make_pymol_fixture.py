"""Regenerate ``tests/data/session.pse`` with a real PyMOL.

The session reader is written against what PyMOL actually writes, so the
fixture it is tested against has to come from PyMOL rather than from Gala's
own writer — a round trip through one's own code proves nothing about the
format. Run this only when the fixture needs to change:

    pymol -cq scripts/make_pymol_fixture.py

It builds a small session out of ``tests/data/site.pdb`` that exercises every
part of the format Gala reads: two objects, one of them moved and grouped,
several representations, a per-atom colour and a session-defined one, a
label, a named selection, and one measurement of each kind.
"""

from __future__ import annotations

import os

from pymol import cmd  # type: ignore[import-not-found]


def _repo_root() -> str:
    """Find the checkout. PyMOL's ``__file__`` points into its own tree."""
    for start in (os.environ.get("GALA_ROOT"), os.getcwd()):
        if not start:
            continue
        current = os.path.abspath(start)
        while True:
            if os.path.exists(
                os.path.join(current, "blender_gala", "blender_manifest.toml")
            ):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    raise SystemExit("run this from the blender_gala checkout, or set GALA_ROOT")


ROOT = _repo_root()
DATA = os.path.join(ROOT, "tests", "data")
TARGET = os.path.join(DATA, "session.pse")


def main() -> None:
    # The text form is the default and the only one Gala reads; setting it
    # explicitly means the fixture does not change if a future PyMOL flips
    # the default.
    cmd.set("pse_binary_dump", 0)

    cmd.load(os.path.join(DATA, "site.pdb"), "site")
    cmd.hide("everything")
    cmd.show("cartoon", "polymer")
    cmd.show("sticks", "organic")
    cmd.show("spheres", "inorganic")
    cmd.color("skyblue", "polymer")
    cmd.color("orange", "organic")
    cmd.set_color("gala_teal", [0.1, 0.7, 0.65])
    cmd.color("gala_teal", "resi 1")
    cmd.label("resi 1 and name CA", '"first CA"')

    # A second object, surfaced, moved off its coordinates and in a group, so
    # the reader's object transform and grouping are covered.
    cmd.create("shell", "site and polymer")
    cmd.hide("everything", "shell")
    cmd.show("surface", "shell")
    cmd.color("grey70", "shell")
    cmd.set("transparency", 0.5, "shell")
    cmd.translate([12.0, 0.0, 0.0], object="shell", camera=0)
    cmd.group("assembly", "shell")

    cmd.select("pocket", "site and resi 1-3")

    names = [a.index for a in cmd.get_model("site and name CA").atom[:4]]
    if len(names) >= 4:
        pick = [f"site and index {i}" for i in names]
        cmd.distance("d1", pick[0], pick[1])
        cmd.angle("a1", pick[0], pick[1], pick[2])
        cmd.dihedral("t1", pick[0], pick[1], pick[2], pick[3])

    # A view with a rotation in it, so a transposed matrix would be visible.
    cmd.set_view(
        (
            0.8,
            0.0,
            0.6,
            0.0,
            1.0,
            0.0,
            -0.6,
            0.0,
            0.8,
            0.0,
            0.0,
            -80.0,
            0.0,
            0.0,
            0.0,
            40.0,
            120.0,
            -20.0,
        )
    )
    cmd.bg_color("white")

    cmd.save(TARGET)
    print(f"wrote {TARGET} ({os.path.getsize(TARGET)} bytes)")
    print(f"  objects: {cmd.get_names('all')}")


main()
