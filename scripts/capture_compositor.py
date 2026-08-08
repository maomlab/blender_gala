"""Capture the compositor node graphs for the documentation.

    python3 scripts/capture_compositor.py            # every shot
    python3 scripts/capture_compositor.py highlight  # one shot

Images land in ``docs/images/compositor``.

Like :mod:`scripts.capture_ui`, and for the same reason: a picture of a node
editor can only come from ``screen.screenshot_area``, which reads the drawn
framebuffer, so a real Blender window has to open. Expect windows to appear and
close while this runs.

The script is both halves of that dance. Run under the system Python it is the
driver: it launches Blender once per shot, then crops and shrinks what comes
back. Run under Blender — which the driver does, passing the shot name after
``--`` — it builds the graph, maximises the node editor, and reports where on
screen the nodes ended up so the driver can crop to them.

The ``highlight`` shot composites from the multilayer EXR that vignette 5
writes, so the Cryptomatte node shows a real picked material rather than an
empty matte list. Run ``make vignettes`` first; without the EXR the shot falls
back to the live render layers, which draws the same graph with nothing picked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(REPO_ROOT, "docs", "images", "compositor")
PASSES_DIR = os.path.join(REPO_ROOT, "docs", "images", "passes")
EXR = os.path.join(PASSES_DIR, "gala.exr")

#: Big enough to hold either graph without scrolling.
WINDOW = (1600, 1040)

#: Widest the finished image is allowed to be, in pixels of the documentation.
MAX_WIDTH = 1400

#: Canvas to leave around the nodes, in screenshot pixels.
PADDING = 32

#: Canvas edge to ignore when looking for the nodes. Hiding the toolbar leaves
#: a small arrow to bring it back, and the region draws a border down its own
#: edges; both are canvas furniture rather than graph, and both would otherwise
#: hold the crop open to the full width.
EDGE = 48


@dataclass(frozen=True)
class Shot:
    """One screenshot: which graph to build, and what to call the file."""

    name: str
    description: str


SHOTS = (
    Shot("chain", "what setup_compositor() builds"),
    Shot("highlight", "one chain kept, the rest knocked back"),
)

SHOTS_BY_NAME = {shot.name: shot for shot in SHOTS}


# ---------------------------------------------------------------------------
# Inside Blender
# ---------------------------------------------------------------------------


def build(shot: Shot) -> None:
    """Build the graph this shot is about, in the current scene."""
    sys.path.insert(0, REPO_ROOT)
    import blender_gala as gala

    gala.register()

    if shot.name == "chain":
        gala.setup_compositor(denoise=True, cryptomatte=True, file_output=PASSES_DIR)
        return

    # The highlight graph on its own, rather than on top of the chain above:
    # the point of the picture is the four nodes that turn a matte into a
    # knocked-back frame, and the picker nodes beside them only crowd it.
    source = EXR if os.path.exists(EXR) else None
    if source is None:
        print(f"  no {EXR}; run `make vignettes` for a shot with a real matte")
    gala.highlight_matte("GALA beta 1", source=source)


def capture(shot: Shot, raw_path: str, meta_path: str) -> None:
    """Draw the node editor full-window, screenshot it, and record the nodes' box.

    Parameters
    ----------
    shot : Shot
        The graph to build.
    raw_path : str
        Where to write the screenshot.
    meta_path : str
        Where to write the rectangle the driver should crop to.
    """
    import bpy

    build(shot)

    def show_graph():
        area = _biggest_area()
        area.ui_type = "CompositorNodeTree"
        space = area.spaces[0]
        # The N panel and the asset shelf are chrome that says nothing about
        # the graph, and between them they eat a third of the window. The
        # breadcrumb is drawn over the canvas, where it would defeat the crop.
        space.show_region_ui = False
        space.overlay.show_context_path = False
        for attr in ("show_region_asset_shelf", "show_region_toolbar"):
            if hasattr(space, attr):
                setattr(space, attr, False)

        # Painting the grid dots out in the background colour leaves a canvas
        # that is genuinely one flat colour, which is what lets the driver find
        # where the nodes stop.
        theme = bpy.context.preferences.themes[0].node_editor
        theme.grid = theme.space.back

    def maximize():
        area = _biggest_area()
        with _override(area=area):
            bpy.ops.screen.screen_full_area()

    def frame():
        area = _biggest_area()
        region = next(r for r in area.regions if r.type == "WINDOW")
        with _override(area=area, region=region, space_data=area.spaces[0]):
            bpy.ops.node.view_all()

    def take():
        area = _biggest_area()
        region = next(r for r in area.regions if r.type == "WINDOW")
        with _override(area=area):
            bpy.ops.screen.screenshot_area(filepath=raw_path)
        # Blender measures regions from the bottom left of the window and the
        # screenshot is indexed from the top left of the area, so the driver is
        # handed the canvas rectangle already converted.
        with open(meta_path, "w") as handle:
            json.dump(
                {
                    "left": region.x - area.x,
                    "top": area.height - (region.y - area.y) - region.height,
                    "width": region.width,
                    "height": region.height,
                },
                handle,
            )
        bpy.ops.wm.quit_blender()

    # `screenshot_area` reads the framebuffer, which still holds the previous
    # frame until Blender redraws — so each change has to be drawn before the
    # next step looks at it, and certainly before the picture is taken.
    steps = [show_graph, maximize, frame, take]

    def advance():
        steps.pop(0)()
        _redraw()
        return 0.3 if steps else None

    bpy.app.timers.register(advance, first_interval=1.0)


def _override(**parts):
    """Context override carrying the window and screen every operator needs."""
    import bpy

    window = bpy.context.window_manager.windows[0]
    return bpy.context.temp_override(window=window, screen=window.screen, **parts)


def _biggest_area():
    """Return the largest area of the first window — the 3D View, at startup."""
    import bpy

    window = bpy.context.window_manager.windows[0]
    return max(window.screen.areas, key=lambda area: area.width * area.height)


def _redraw() -> None:
    """Draw and swap the window, so the framebuffer matches the current state.

    Twice, because the window is double buffered: one draw-and-swap leaves the
    frame we want on screen and the stale one in the buffer the screenshot
    reads back.
    """
    import bpy

    with _override():
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def crop(raw_path: str, meta_path: str, out_path: str) -> None:
    """Cut the nodes out of an area screenshot and shrink them to fit the page.

    Two steps: the canvas rectangle, which drops the header, and then the nodes
    within it, since ``view_all`` fits a graph of one shape into a region of
    another and leaves the difference as empty canvas.

    Parameters
    ----------
    raw_path : str
        The area screenshot.
    meta_path : str
        The canvas rectangle written by :func:`capture`.
    out_path : str
        Where to write the finished image.
    """
    import numpy
    from PIL import Image

    with open(meta_path) as handle:
        box = json.load(handle)

    image = Image.open(raw_path).convert("RGB")
    canvas = image.crop(
        (
            box["left"],
            box["top"],
            box["left"] + box["width"],
            box["top"] + box["height"],
        )
    )

    # With the grid painted out, the canvas is one flat colour and anything
    # that differs from its corner is a node or a link.
    pixels = numpy.asarray(canvas).astype(int)[EDGE:-EDGE, EDGE:-EDGE]
    drawn = (numpy.abs(pixels - pixels[0, 0]) > 2).any(axis=2)
    rows = numpy.flatnonzero(drawn.any(axis=1))
    columns = numpy.flatnonzero(drawn.any(axis=0))
    if not len(rows) or not len(columns):
        raise SystemExit(f"{out_path}: the node editor came out empty")

    graph = canvas.crop(
        (
            max(0, int(columns[0]) + EDGE - PADDING),
            max(0, int(rows[0]) + EDGE - PADDING),
            min(canvas.width, int(columns[-1]) + EDGE + PADDING),
            min(canvas.height, int(rows[-1]) + EDGE + PADDING),
        )
    )

    # The screenshot is in device pixels, which on a HiDPI display is twice
    # what a documentation page needs.
    if graph.width > MAX_WIDTH:
        height = round(graph.height * MAX_WIDTH / graph.width)
        graph = graph.resize((MAX_WIDTH, height), Image.LANCZOS)

    graph.save(out_path)
    print(
        f"  wrote {os.path.relpath(out_path, REPO_ROOT)} ({graph.width}x{graph.height})"
    )


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
    print(f"=== {shot.name}: {shot.description}")
    result = subprocess.run(command, capture_output=True, text=True)
    if not os.path.exists(meta_path):
        sys.stdout.write(result.stdout[-2000:])
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit(f"{shot.name}: Blender produced no screenshot")

    crop(raw_path, meta_path, os.path.join(IMAGE_DIR, f"{shot.name}.png"))


def main(argv: list[str]) -> None:
    """Capture every requested shot."""
    from capture_ui import empty_file, find_blender

    names = argv or list(SHOTS_BY_NAME)
    unknown = [name for name in names if name not in SHOTS_BY_NAME]
    if unknown:
        raise SystemExit(
            f"unknown shot(s): {', '.join(unknown)}; have {', '.join(SHOTS_BY_NAME)}"
        )

    blender = find_blender()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gala-compositor-") as work_dir:
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
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        main(sys.argv[1:])
