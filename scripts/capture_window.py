"""Capture a whole Blender window with a Gala scene in it.

    python3 scripts/capture_window.py

Writes ``docs/images/ui/window.webp``: the 3D View in camera view showing a
styled molecule with interactions and a measurement drawn, and the Gala tab
open in the sidebar. :mod:`scripts.make_hero` composes that into the front
page's hero image.

Like :mod:`scripts.capture_ui`, this cannot run in ``--background``:
``screen.screenshot`` reads the drawn framebuffer, so a real window has to
open. Expect one to appear and close while this runs. The script is both the
driver and, when Blender launches it with a ``--`` argument, the thing that
runs inside.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from capture_ui import (  # noqa: E402  (path has to be set up first)
    WINDOW,
    _redraw,
    _sidebar_region,
    _view3d_areas,
    empty_file,
    find_blender,
)

OUT_PATH = os.path.join(REPO_ROOT, "docs", "images", "ui", "window.webp")

#: The structure in the shot. Adenylate kinase: two domains, a ligand in the
#: cleft between them, and small enough to draw interactively.
PDB_CODE = "1ake"

#: The shot comes back at the display's device resolution, which on a retina
#: screen is 3200 px of PNG for a picture the hero uses at 1560. Stored at
#: twice what it is drawn at, which is sharp on any display and a third of the
#: bytes.
MAX_WIDTH = 2200

#: Cool protein, warm ligand — the same pairing the vignettes use.
PROTEIN_COLOUR = "#93a6b8"
LIGAND_COLOUR = "#ffab3d"


# ---------------------------------------------------------------------------
# Inside Blender
# ---------------------------------------------------------------------------


def build_scene():
    """Load, style and light a molecule, and draw some Gala output on it."""
    import bpy

    bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.molecularnodes")

    sys.path.insert(0, REPO_ROOT)
    from blender_gala.ui import panels

    # Scene Setup open, the rest collapsed: the shape of the add-on in one
    # picture. `bl_options` is only honoured by the first draw of a session,
    # which is why this happens before `register`.
    for panel in panels.classes:
        panel.bl_options = (
            set() if panel.bl_idname == "GALA_PT_scene" else {"DEFAULT_CLOSED"}
        )

    import blender_gala as gala
    from blender_gala.core import mn as mn_bridge

    # Molecular Nodes is an extension, so it is `bl_ext.blender_org...` rather
    # than a plain import; the bridge knows where to find it.
    mn = mn_bridge.require_mn()
    mn.register()
    gala.register()
    bpy.context.preferences.system.use_region_overlap = False

    # The factory startup cube is a room the camera sits inside, and its walls
    # are what the viewport would otherwise be showing.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    try:
        molecule = mn.Molecule.fetch(PDB_CODE)
    except Exception as exc:  # pragma: no cover - depends on the network
        print(f"  could not fetch {PDB_CODE} ({exc}); using the bundled fixture")
        molecule = mn.Molecule.load(
            os.path.join(REPO_ROOT, "tests", "data", "site.pdb")
        )
    molecule.add_style("cartoon", selection="is_peptide")
    molecule.add_style("ball_and_stick", selection="is_hetero")
    gala.color_by_selection(
        molecule, {"protein": PROTEIN_COLOUR, "not protein": LIGAND_COLOUR}
    )

    gala.publication_setup(
        molecule,
        preset="draft",
        lighting_style="three_point",
        material_scheme="chemistry",
        viewpoint="iso",
        transparent=False,
    )

    # Something of Gala's own in the viewport, not just a lit molecule.
    contacts = gala.find_interactions(molecule, "not protein", "protein")
    gala.draw_interactions(contacts[:12], target=molecule)
    gala.distance(
        molecule,
        "chain A and resi 30 and name CA",
        "chain A and resi 150 and name CA",
        draw=True,
        label_size=3.0,
    )

    # Active but not selected: the outliner, the properties editor and the
    # viewport header are all about the molecule, without the orange selection
    # outline tracing every ribbon in the shot.
    bpy.context.view_layer.objects.active = molecule.object
    return molecule


def capture(path: str) -> None:
    """Set the window up and screenshot it, then quit."""
    import bpy

    try:
        build_scene()
    except Exception:
        # Without this the window stays open with no timers registered, and
        # the driver waits on a Blender that is never going to finish.
        import traceback

        traceback.print_exc()
        bpy.app.timers.register(bpy.ops.wm.quit_blender, first_interval=0.1)
        return

    def show_sidebar():
        for area in _view3d_areas():
            area.spaces[0].show_region_ui = True

    def dress_viewport():
        _sidebar_region().active_panel_category = "Gala"
        for area in _view3d_areas():
            space = area.spaces[0]
            # Material preview rather than Cycles: with the scene's own lights
            # and world it is close to what the render gives, and it draws fast
            # enough that a screenshot catches a finished frame instead of a
            # half-converged one.
            space.shading.type = "MATERIAL"
            space.shading.use_scene_lights = True
            space.shading.use_scene_world = True
            space.region_3d.view_perspective = "CAMERA"
            space.overlay.show_floor = False
            space.overlay.show_axis_x = False
            space.overlay.show_axis_y = False
            space.overlay.show_cursor = False
            space.overlay.show_object_origins = False
            # The light rig is an empty, drawn as three axis lines the size of
            # the molecule, and the rig is parented to a pivot that draws a
            # line to each light. All of it crosses the shot.
            space.overlay.show_relationship_lines = False
            space.overlay.show_extras = False

            # Fit the camera frame to the region: unzoomed it is a small
            # rectangle with most of the viewport around it.
            window = bpy.context.window_manager.windows[0]
            region = next(r for r in area.regions if r.type == "WINDOW")
            with bpy.context.temp_override(
                window=window, area=area, region=region, space_data=space
            ):
                bpy.ops.view3d.view_center_camera()

    def take():
        _redraw()
        bpy.ops.screen.screenshot(filepath=path)
        bpy.ops.wm.quit_blender()

    steps = [show_sidebar, dress_viewport, take]

    def advance():
        steps.pop(0)()
        _redraw()
        return 0.4 if steps else None

    bpy.app.timers.register(advance, first_interval=1.5)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch Blender, take the shot, and report what came back."""
    import tempfile

    from PIL import Image

    blender = find_blender()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    with tempfile.TemporaryDirectory(prefix="gala-window-") as work_dir:
        blend = empty_file(blender, work_dir)
        result = subprocess.run(
            [
                blender,
                "--factory-startup",
                "--window-geometry",
                "0",
                "0",
                str(WINDOW[0]),
                str(WINDOW[1]),
                blend,
                "--python",
                os.path.abspath(__file__),
                "--",
                OUT_PATH,
            ],
            capture_output=True,
            text=True,
        )

    if not os.path.exists(OUT_PATH):
        sys.stdout.write(result.stdout[-3000:])
        sys.stderr.write(result.stderr[-3000:])
        raise SystemExit("Blender produced no screenshot")

    image = Image.open(OUT_PATH)
    if image.width > MAX_WIDTH:
        image = image.resize(
            (MAX_WIDTH, round(MAX_WIDTH * image.height / image.width)), Image.LANCZOS
        )
        image.save(OUT_PATH, optimize=True)
    print(
        f"  wrote {os.path.relpath(OUT_PATH, REPO_ROOT)} ({image.width}x{image.height})"
    )


if __name__ == "__main__":
    if "--" in sys.argv:
        capture(sys.argv[sys.argv.index("--") + 1])
    else:
        main()
