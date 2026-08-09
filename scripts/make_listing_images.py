"""Compose the preview images for the extensions.blender.org listing.

The platform renders previews as 16:9 thumbnails (1920x1080 and 640x360), and
every figure this project produces is square or nearly so — a vignette render
is 960x960, the hero is 1.93:1. Uploaded as they are, each one is letterboxed
by the platform with whatever background it feels like, and the transparent
ones composite onto it unpredictably.

So this fits each figure onto a 16:9 canvas here, where the background is a
decision rather than an accident, and writes them where they can be uploaded
without further thought.

    python scripts/make_listing_images.py        # or: make listing

Nothing renders: this only rearranges images the vignettes already produced,
so it is cheap to re-run after any of them changes.
"""

from __future__ import annotations

import os

from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(HERE, "docs", "images")
OUT_DIR = os.path.join(IMAGE_DIR, "listing")

#: PNG on purpose, and the only PNG this project still writes: everything in
#: `docs/images` is WebP, but the extensions platform takes PNG and JPEG and
#: these are uploaded rather than served. They are build output, so they are
#: not committed either.
#:
#: The thumbnail the platform renders. Matching it exactly means nothing is
#: scaled twice.
SIZE = (1920, 1080)

#: Blender's own editor grey. A figure with a transparent background lands on
#: this rather than on whatever the page happens to be.
BACKGROUND = (24, 24, 24)

#: Source figure, and the caption it illustrates on the listing. Ordered: the
#: first is what someone sees before clicking anything.
PREVIEWS = (
    ("hero.webp", "01-overview"),
    ("01_publication_figure.webp", "02-publication-figure"),
    ("02_binding_site.webp", "03-binding-site"),
    ("03_measurements.webp", "04-measurements"),
    ("07_electrostatics.webp", "05-electrostatics"),
    ("08_pymol_session.webp", "06-pymol-session"),
)

#: Fraction of the canvas the figure fills. Short of 1 so nothing touches the
#: edge, which is what makes a thumbnail look cropped rather than composed.
FILL = 0.94


def compose(source: str, name: str) -> str | None:
    """Fit one figure onto the 16:9 canvas and write it out."""
    path = os.path.join(IMAGE_DIR, source)
    if not os.path.exists(path):
        print(f"  skipped {source}: not rendered yet")
        return None

    figure = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", SIZE, (*BACKGROUND, 255))

    scale = min(SIZE[0] * FILL / figure.width, SIZE[1] * FILL / figure.height)
    fitted = figure.resize(
        (max(1, round(figure.width * scale)), max(1, round(figure.height * scale))),
        Image.LANCZOS,
    )
    canvas.alpha_composite(
        fitted,
        ((SIZE[0] - fitted.width) // 2, (SIZE[1] - fitted.height) // 2),
    )

    target = os.path.join(OUT_DIR, f"{name}.png")
    canvas.convert("RGB").save(target, optimize=True)
    print(
        f"  {os.path.basename(target):<26} {SIZE[0]}x{SIZE[1]}  "
        f"{round(os.path.getsize(target) / 1024)} kB  from {source}"
    )
    return target


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    written = [compose(source, name) for source, name in PREVIEWS]
    print(f"\n{len([w for w in written if w])} preview(s) in {OUT_DIR}")
    print("Upload them on the extension's page: Manage > Edit > Preview images.")


if __name__ == "__main__":
    main()
