"""Capture the Gala sidebar panels for the documentation.

    python3 scripts/capture_ui.py            # every shot
    python3 scripts/capture_ui.py measure    # one shot

Images land in ``docs/images/ui``.

Unlike the vignettes, these cannot be made in ``--background``: the only way to
get a picture of a panel is ``screen.screenshot``, which reads the drawn
framebuffer, so a real Blender window has to open. Expect windows to appear and
close while this runs.

The script is both halves of that dance. Run under the system Python it is the
driver: it launches Blender once per shot and crops what comes back. Run under
Blender — which the driver does, passing the shot name after ``--`` — it sets
up the sidebar, screenshots the window, and reports where the sidebar was.

One launch per shot, rather than one launch cycling through them, because
whether a panel is expanded is not writable from Python. ``bl_options`` decides
it at registration, and only the first draw of a fresh session honours it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(REPO_ROOT, "docs", "images", "ui")

#: Big enough that no shot has to scroll, small enough for a 1728x1117 desktop.
WINDOW = (1600, 1040)

#: Blender draws the sidebar's category tabs 1.4 UI units wide. The tab column
#: is chrome, not panel, so per-panel shots trim it off; the overview keeps it
#: to show which tab these panels live under.
TAB_UNITS = 1.4
UI_UNIT = 20

#: Pixels of region border to ignore when looking for the end of the panels,
#: and of flat background to leave below them.
MARGIN = 4
PADDING = 8

#: Pixels in a row that may differ from an empty one before it counts as drawn on.
NOISE = 4


@dataclass(frozen=True)
class Shot:
    """One screenshot: which panels are open, and what to call the file.

    Attributes
    ----------
    name : str
        File stem, and the name used to request a single shot.
    expand : tuple of str
        ``bl_idname`` of every panel to draw expanded.
    props : dict
        Values to set on ``scene.gala`` first, when a filled-in field shows
        more than an empty one.
    tabs : bool
        Keep the category tab column.
    others : bool
        Draw the rest of the panels too, collapsed. Off for the close-ups, so
        each one is about its own panel — and because Scene Setup with its
        sub-panels open is already taller than the window.
    """

    name: str
    expand: tuple[str, ...]
    props: dict[str, object] = field(default_factory=dict)
    tabs: bool = False
    others: bool = False


SCENE_PANELS = (
    "GALA_PT_scene",
    "GALA_PT_origin",
    "GALA_PT_lighting",
    "GALA_PT_compositing",
)

SHOTS = (
    # The whole tab: Scene Setup open, everything else collapsed, so the shape
    # of the add-on is visible in one picture.
    Shot("sidebar", expand=("GALA_PT_scene",), tabs=True, others=True),
    Shot("scene-setup", expand=SCENE_PANELS),
    Shot("interactions", expand=("GALA_PT_interactions",)),
    Shot(
        "measure",
        expand=("GALA_PT_measure",),
        # The ';' form is the half of this panel a screenshot can show; the
        # other half is an Edit Mode selection.
        props={"measure_selection": "resi 15 and name CA; resi 90 and name CA"},
    ),
    Shot("label", expand=("GALA_PT_label",)),
    Shot("colour", expand=("GALA_PT_color",)),
    Shot("colour-csv", expand=("GALA_PT_color",), props={"color_mode": "csv"}),
    Shot("pymol", expand=("GALA_PT_pymol",)),
)

SHOTS_BY_NAME = {shot.name: shot for shot in SHOTS}


# ---------------------------------------------------------------------------
# Inside Blender
# ---------------------------------------------------------------------------


def capture(shot: Shot, raw_path: str, meta_path: str) -> None:
    """Open the Gala sidebar, screenshot the window, and record its geometry.

    Parameters
    ----------
    shot : Shot
        The configuration to draw.
    raw_path : str
        Where to write the full-window screenshot.
    meta_path : str
        Where to write the sidebar rectangle the driver should crop to.
    """
    import bpy

    # Molecular Nodes is a soft dependency, and the Scene Setup panel says so
    # in red when it is missing. Enable it so the shots show the normal case.
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.molecularnodes")
    except Exception as exc:  # pragma: no cover - depends on the local install
        print(f"  Molecular Nodes not enabled: {exc}")

    sys.path.insert(0, REPO_ROOT)
    from blender_gala.ui import panels

    for panel in panels.classes:
        if panel.bl_idname in shot.expand:
            panel.bl_options = set()
        elif shot.others:
            panel.bl_options = {"DEFAULT_CLOSED"}
        else:
            panel.poll = classmethod(lambda cls, context: False)

    import blender_gala

    blender_gala.register()

    # Overlapping regions make the sidebar translucent, so an empty panel shows
    # the viewport through it. Off, the background is flat and croppable.
    preferences = bpy.context.preferences
    preferences.system.use_region_overlap = False

    for key, value in shot.props.items():
        setattr(bpy.context.scene.gala, key, value)

    def open_sidebar():
        for area in _view3d_areas():
            area.spaces[0].show_region_ui = True

    def select_tab():
        # Only after the sidebar has drawn once does it know its categories,
        # so this cannot be folded into the step above.
        _sidebar_region().active_panel_category = "Gala"

    def take():
        region = _sidebar_region()
        _redraw()
        bpy.ops.screen.screenshot(filepath=raw_path)
        with open(meta_path, "w") as handle:
            json.dump(
                {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                    "tab_width": round(
                        TAB_UNITS
                        * UI_UNIT
                        * preferences.view.ui_scale
                        * preferences.system.pixel_size
                    ),
                    "category": region.active_panel_category,
                },
                handle,
            )
        bpy.ops.wm.quit_blender()

    # `screen.screenshot` reads the framebuffer, which still holds the previous
    # frame until Blender redraws — so each change has to be drawn before the
    # next step looks at it, and certainly before the picture is taken.
    steps = [open_sidebar, select_tab, take]

    def advance():
        steps.pop(0)()
        _redraw()
        return 0.3 if steps else None

    bpy.app.timers.register(advance, first_interval=1.0)


def _redraw() -> None:
    """Draw and swap the window, so the framebuffer matches the current state.

    Twice, because the window is double buffered: one draw-and-swap leaves the
    frame we want on screen and the stale one in the buffer the screenshot
    reads back.
    """
    import bpy

    window = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=window, screen=window.screen):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)


def _view3d_areas():
    """Yield every 3D View area in the first window."""
    import bpy

    for area in bpy.context.window_manager.windows[0].screen.areas:
        if area.type == "VIEW_3D":
            yield area


def _sidebar_region():
    """Return the 3D View's sidebar region."""
    for area in _view3d_areas():
        for region in area.regions:
            if region.type == "UI":
                return region
    raise RuntimeError("no 3D View sidebar")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def find_blender() -> str:
    """Locate the Blender executable, the same way the Makefile does."""
    from shutil import which

    candidates = [
        os.environ.get("BLENDER"),
        which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        os.path.expanduser("~/blender/blender"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise SystemExit("Blender not found. Set BLENDER=/path/to/blender")


def crop(raw_path: str, meta_path: str, out_path: str, keep_tabs: bool) -> None:
    """Cut the sidebar out of a full-window screenshot and trim it to fit.

    Parameters
    ----------
    raw_path : str
        The full-window screenshot.
    meta_path : str
        The geometry written by :func:`capture`.
    out_path : str
        Where to write the cropped image.
    keep_tabs : bool
        Keep the category tab column on the right.
    """
    import numpy
    from PIL import Image

    with open(meta_path) as handle:
        meta = json.load(handle)
    if meta["category"] != "Gala":
        raise SystemExit(
            f"{out_path}: the sidebar was on the {meta['category']!r} tab, not Gala"
        )

    # Blender reports region geometry in device pixels measured from the bottom
    # left, and the screenshot is those same device pixels indexed from the top
    # left — so the only conversion needed is flipping y.
    image = Image.open(raw_path).convert("RGB")
    sidebar = image.crop(
        (
            meta["x"],
            image.height - (meta["y"] + meta["height"]),
            meta["x"] + meta["width"],
            image.height - meta["y"],
        )
    )

    panels_width = max(1, sidebar.width - meta["tab_width"])
    pixels = numpy.asarray(sidebar).astype(int)

    # Cut where the panels stop, so a short panel gets a short image. The tab
    # column is measured separately and the taller of the two wins, otherwise
    # the shot that keeps the tabs would slice through the last tab label.
    last = _content_bottom(pixels[:, :panels_width])
    if keep_tabs:
        last = max(last, _content_bottom(pixels[:, panels_width:]))
    else:
        sidebar = sidebar.crop((0, 0, panels_width, sidebar.height))
    if not last:
        raise SystemExit(f"{out_path}: the sidebar came out blank")
    if last >= sidebar.height - 2 * MARGIN:
        print(f"  warning: {os.path.basename(out_path)} may be cut off at the bottom")
    sidebar = sidebar.crop((0, 0, sidebar.width, min(sidebar.height, last + PADDING)))

    sidebar.save(out_path)
    print(
        f"  wrote {os.path.relpath(out_path, REPO_ROOT)}"
        f" ({sidebar.width}x{sidebar.height})"
    )


def empty_file(blender: str, work_dir: str) -> str:
    """Save a factory-default .blend, and return its path.

    Started with no file, Blender puts the splash screen over the left edge of
    the sidebar; started with one, it does not. Dismissing the splash from
    inside the session is not an option — the only thing that clears it is
    re-reading the startup file, which crashes Blender when a timer does it.

    Parameters
    ----------
    blender : str
        The Blender executable.
    work_dir : str
        Where to put the file.

    Returns
    -------
    str
        Path to the saved file.
    """
    path = os.path.join(work_dir, "startup.blend")
    subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-expr",
            f"import bpy; bpy.ops.wm.save_as_mainfile(filepath={path!r})",
        ],
        capture_output=True,
        check=True,
    )
    return path


def _content_bottom(pixels) -> int:
    """Return the last row of ``pixels`` that has something drawn on it.

    Rows are compared against the bottom row rather than against a sampled
    background colour, because "empty" is not one flat colour: an empty row
    still carries the region's border down its edges, and the tab column has a
    background of its own. Anything that runs the full height cancels out.

    Parameters
    ----------
    pixels : numpy.ndarray
        A height x width x 3 slice of the sidebar.

    Returns
    -------
    int
        Row index, or 0 if nothing was drawn.
    """
    import numpy

    # Inset on every side: the region's border runs along the bottom edge too,
    # so the reference row has to come from just above it. A few pixels still
    # differ where that border rounds off at the corners, hence the threshold.
    inset = pixels[MARGIN:-MARGIN, MARGIN:-MARGIN]
    differing = (numpy.abs(inset - inset[-1]) > 2).any(axis=2).sum(axis=1)
    filled = numpy.flatnonzero(differing > NOISE)
    return int(filled[-1]) + MARGIN if len(filled) else 0


def run(shot: Shot, blender: str, work_dir: str, blend: str) -> None:
    """Launch Blender for one shot and crop the result."""
    raw_path = os.path.join(work_dir, f"{shot.name}.raw.png")
    meta_path = os.path.join(work_dir, f"{shot.name}.json")
    for stale in (raw_path, meta_path):
        if os.path.exists(stale):
            os.remove(stale)

    command = [
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
        shot.name,
        work_dir,
    ]
    print(f"=== {shot.name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if not os.path.exists(meta_path):
        sys.stdout.write(result.stdout[-2000:])
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit(f"{shot.name}: Blender produced no screenshot")

    crop(raw_path, meta_path, os.path.join(IMAGE_DIR, f"{shot.name}.png"), shot.tabs)


def main(argv: list[str]) -> None:
    """Capture every requested shot."""
    names = argv or list(SHOTS_BY_NAME)
    unknown = [name for name in names if name not in SHOTS_BY_NAME]
    if unknown:
        raise SystemExit(
            f"unknown shot(s): {', '.join(unknown)}; have {', '.join(SHOTS_BY_NAME)}"
        )

    blender = find_blender()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gala-ui-") as work_dir:
        blend = empty_file(blender, work_dir)
        for name in names:
            run(SHOTS_BY_NAME[name], blender, work_dir, blend)


if __name__ == "__main__":
    if "--" in sys.argv:
        # Launched by the driver, inside Blender.
        shot_name, work = sys.argv[sys.argv.index("--") + 1 :]
        capture(
            SHOTS_BY_NAME[shot_name],
            os.path.join(work, f"{shot_name}.raw.png"),
            os.path.join(work, f"{shot_name}.json"),
        )
    else:
        main(sys.argv[1:])
