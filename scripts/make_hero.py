"""Compose the front page's hero image.

    python3 scripts/make_hero.py

Puts the Blender window shot from :mod:`scripts.capture_window` on the left,
the vignette renders down the right as a showcase of what the add-on does, and
an arrow from the window to each one. Writes ``docs/images/hero.png``.

Run under the system Python rather than Blender's: Blender bundles numpy but
not Pillow, and this is image plumbing rather than anything to do with bpy.
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(REPO_ROOT, "docs", "images")
WINDOW_SHOT = os.path.join(IMAGE_DIR, "ui", "window.png")
OUT_PATH = os.path.join(IMAGE_DIR, "hero.png")

#: What each vignette shows, in the order they run. The renders are the
#: showcase; the text says what part of the add-on made them.
CAPABILITIES = [
    (
        "01_publication_figure.png",
        "Publication scenes",
        "Render preset, three-point rig, materials and camera, in one call",
    ),
    (
        "02_binding_site.png",
        "Interactions",
        "Hydrogen bonds, salt bridges and stacking, found and drawn",
    ),
    (
        "03_measurements.png",
        "Measurements",
        "Distances, angles and dihedrals, with the value on the figure",
    ),
    (
        "04_alphafold_confidence.png",
        "Colour by data",
        "AlphaFold confidence, B-factors, or a column of your own",
    ),
    (
        "05_compositing_passes.png",
        "Compositing passes",
        "Cryptomatte and depth, depth of field, depth cueing",
    ),
    (
        "06_turntable.webp",
        "Animation",
        "Turntables and orbits, rendered frame by frame",
    ),
]

# --- Geometry, in pixels of the finished image ----------------------------
MARGIN = 56
SHOT_WIDTH = 1560
GUTTER = 150
CARD_WIDTH = 940
CARD_HEIGHT = 168
CARD_GAP = 24
THUMB = 144
RADIUS = 18

# --- Colour ----------------------------------------------------------------
BACKGROUND = (14, 19, 24)
PANEL = (24, 32, 41)
PANEL_EDGE = (44, 58, 71)
ACCENT = (38, 166, 154)
TITLE = (236, 242, 245)
BODY = (150, 168, 180)

#: Where to look for a text face, most preferred first. Falls back to Pillow's
#: bitmap default, which is ugly but always there.
FONT_CANDIDATES = {
    "bold": (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ),
    "regular": (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ),
}


def load_font(weight: str, size: int):
    """Return a font of ``size`` pixels, or Pillow's default."""
    for path in FONT_CANDIDATES[weight]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def rounded_shadow(size: tuple[int, int], radius: int, spread: int) -> Image.Image:
    """A soft dark rectangle to sit behind a panel."""
    from PIL import ImageFilter

    width, height = size
    shadow = Image.new("RGBA", (width + 4 * spread, height + 4 * spread), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (2 * spread, 2 * spread, 2 * spread + width, 2 * spread + height),
        radius=radius,
        fill=(0, 0, 0, 160),
    )
    return shadow.filter(ImageFilter.GaussianBlur(spread))


def rounded(image: Image.Image, radius: int) -> Image.Image:
    """Round the corners of ``image``."""
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius=radius, fill=255)
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


def thumbnail(name: str, box: int) -> Image.Image:
    """Load a vignette render and fit it into a ``box`` square.

    The renders have transparent backgrounds and a lot of air around the
    molecule, so this trims to what was actually drawn before scaling: at
    thumbnail size the difference between filling the box and filling half of
    it is the difference between reading and not.
    """
    image = Image.open(os.path.join(IMAGE_DIR, name))
    image.seek(0)  # An animation is represented by its first frame.
    image = image.convert("RGBA")

    bounds = image.getchannel("A").getbbox()
    if bounds is not None:
        image = image.crop(bounds)
    image = trim_border(image)

    image.thumbnail((box, box), Image.LANCZOS)
    canvas = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    canvas.paste(image, ((box - image.width) // 2, (box - image.height) // 2), image)
    return canvas


def wrapped(draw, text: str, font, limit: int) -> list[str]:
    """Break ``text`` into lines no wider than ``limit`` pixels."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > limit:
            lines.append(line)
            line = word
        else:
            line = candidate
    return [*lines, line] if line else lines


def trim_border(image: Image.Image, tolerance: int = 10) -> Image.Image:
    """Crop a uniform border, if the image has one.

    Transparent renders are trimmed by their alpha, but the compositing
    vignette is a lit scene on an opaque background, and at 144 pixels a wide
    flat margin is most of the thumbnail.
    """
    from PIL import ImageChops

    background = Image.new("RGBA", image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image.convert("RGB"), background.convert("RGB"))
    bounds = difference.convert("L").point(lambda v: 255 * (v > tolerance)).getbbox()
    return image.crop(bounds) if bounds else image


def arrow(draw: ImageDraw.ImageDraw, start, end, colour, width: int = 5) -> None:
    """Draw a curved arrow from ``start`` to ``end``.

    A quadratic Bézier, with its control point pushed out horizontally, so six
    arrows leaving the same window fan out instead of crossing.
    """
    (x0, y0), (x2, y2) = start, end
    control = ((x0 + x2) / 2 + (x2 - x0) * 0.18, y0)

    points = []
    for step in range(41):
        t = step / 40
        points.append(
            (
                (1 - t) ** 2 * x0 + 2 * (1 - t) * t * control[0] + t**2 * x2,
                (1 - t) ** 2 * y0 + 2 * (1 - t) * t * control[1] + t**2 * y2,
            )
        )
    draw.line(points, fill=colour, width=width, joint="curve")

    # Head, aimed along the last segment of the curve.
    (tail_x, tail_y), (tip_x, tip_y) = points[-6], points[-1]
    angle = math.atan2(tip_y - tail_y, tip_x - tail_x)
    length, spread = 26, 0.42
    draw.polygon(
        [
            (tip_x, tip_y),
            (
                tip_x - length * math.cos(angle - spread),
                tip_y - length * math.sin(angle - spread),
            ),
            (
                tip_x - length * math.cos(angle + spread),
                tip_y - length * math.sin(angle + spread),
            ),
        ],
        fill=colour,
    )


def main() -> None:
    if not os.path.exists(WINDOW_SHOT):
        raise SystemExit(
            f"{os.path.relpath(WINDOW_SHOT, REPO_ROOT)} is missing; "
            "run `make window-shot` first"
        )

    shot = Image.open(WINDOW_SHOT).convert("RGB")
    shot = shot.resize(
        (SHOT_WIDTH, round(SHOT_WIDTH * shot.height / shot.width)), Image.LANCZOS
    )

    column_height = len(CAPABILITIES) * CARD_HEIGHT + (len(CAPABILITIES) - 1) * CARD_GAP
    width = MARGIN * 2 + SHOT_WIDTH + GUTTER + CARD_WIDTH
    height = MARGIN * 2 + max(column_height, shot.height)

    canvas = Image.new("RGBA", (width, height), (*BACKGROUND, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font("bold", 34)
    body_font = load_font("regular", 26)
    caption_font = load_font("regular", 24)

    # --- The window, on the left -----------------------------------------
    shot_x, shot_y = MARGIN, (height - shot.height) // 2
    shadow = rounded_shadow(shot.size, RADIUS, 18)
    canvas.alpha_composite(shadow, (shot_x - 36, shot_y - 36))
    canvas.alpha_composite(rounded(shot, RADIUS), (shot_x, shot_y))
    draw.rounded_rectangle(
        (shot_x, shot_y, shot_x + shot.width, shot_y + shot.height),
        radius=RADIUS,
        outline=PANEL_EDGE,
        width=2,
    )
    draw.text(
        (shot_x + 6, shot_y + shot.height + 16),
        "The Gala tab in Blender's 3D View sidebar, on a scene it set up itself",
        font=caption_font,
        fill=BODY,
    )

    # --- The capabilities, down the right --------------------------------
    card_x = MARGIN + SHOT_WIDTH + GUTTER
    top = (height - column_height) // 2
    for index, (name, heading, blurb) in enumerate(CAPABILITIES):
        card_y = top + index * (CARD_HEIGHT + CARD_GAP)
        draw.rounded_rectangle(
            (card_x, card_y, card_x + CARD_WIDTH, card_y + CARD_HEIGHT),
            radius=RADIUS,
            fill=PANEL,
            outline=PANEL_EDGE,
            width=2,
        )

        pad = (CARD_HEIGHT - THUMB) // 2
        canvas.alpha_composite(thumbnail(name, THUMB), (card_x + pad, card_y + pad))

        text_x = card_x + pad + THUMB + 28
        text_width = card_x + CARD_WIDTH - pad - text_x
        lines = wrapped(draw, blurb, body_font, text_width)
        # Two lines of blurb sit lower than one, so the pair is centred on the
        # card rather than hanging off the bottom of it.
        top_of_text = card_y + (CARD_HEIGHT - (44 + 34 * len(lines))) // 2
        draw.text((text_x, top_of_text), heading, font=title_font, fill=TITLE)
        for number, line in enumerate(lines):
            draw.text(
                (text_x, top_of_text + 50 + 34 * number),
                line,
                font=body_font,
                fill=BODY,
            )

        # One arrow per capability, all leaving the window's right edge.
        span = shot.height * 0.62
        start_y = (
            shot_y + shot.height / 2 - span / 2 + span * index / (len(CAPABILITIES) - 1)
        )
        arrow(
            draw,
            (shot_x + shot.width + 10, start_y),
            (card_x - 16, card_y + CARD_HEIGHT / 2),
            (*ACCENT, 235),
        )

    canvas.convert("RGB").save(OUT_PATH, optimize=True)
    print(
        f"  wrote {os.path.relpath(OUT_PATH, REPO_ROOT)} ({width}x{height}, "
        f"{os.path.getsize(OUT_PATH) / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
