"""Assemble rendered frames into a looping animation.

    python3 scripts/make_animation.py frames/ docs/images/06_turntable.webp

Run under the system Python rather than Blender's: Blender bundles numpy but
not Pillow, and this is image plumbing rather than anything to do with bpy.

WebP rather than GIF. GIF gives 256 colours and one bit of alpha, so a smooth
render has to be quantised against a shared palette and its antialiased edge
has to fall one way or the other; WebP keeps the full range and a real alpha
channel, at a fraction of the size. That fraction is what pays for animating
every frame of the orbit rather than every third.
"""

from __future__ import annotations

import os
import re
import sys

from PIL import Image

#: Frames are numbered, `000.png` upwards. Matching only those means a stray
#: image in the same directory — a contact sheet, a thumbnail — cannot quietly
#: become an extra frame of the animation.
FRAME_NAME = re.compile(r"^\d+\.png$")

#: Milliseconds per frame. 40 is 25 fps, where a turn stops reading as a
#: sequence of positions and starts reading as motion.
DELAY_MS = int(os.environ.get("GALA_ANIM_DELAY", "40"))

#: WebP quality for the colour channels.
QUALITY = int(os.environ.get("GALA_ANIM_QUALITY", "72"))

#: Quality for the alpha channel, which is where the bytes actually go. WebP
#: stores alpha losslessly by default, and on a translucent surface turning
#: against nothing that is most of the file: 120 frames came to 9.6 MB, and
#: dropping colour quality from 80 to 45 took off barely a megabyte. Making
#: alpha lossy halves it, with no halo at the edges that survives a look at
#: 4x against black.
ALPHA_QUALITY = int(os.environ.get("GALA_ANIM_ALPHA_QUALITY", "50"))

#: Encoder effort, 0 to 6.
METHOD = int(os.environ.get("GALA_ANIM_METHOD", "4"))


def load_frames(directory: str) -> list[Image.Image]:
    """Return the numbered PNG frames in ``directory``, in order."""
    names = sorted(name for name in os.listdir(directory) if FRAME_NAME.match(name))
    if not names:
        raise SystemExit(f"no numbered PNG frames in {directory}")
    return [Image.open(os.path.join(directory, name)).convert("RGBA") for name in names]


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: make_animation.py <frames dir> <output.webp>")
    directory, output = argv

    frames = load_frames(directory)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=DELAY_MS,
        loop=0,
        quality=QUALITY,
        alpha_quality=ALPHA_QUALITY,
        # Encoder effort, 0 fast to 6 smallest. 6 is the obvious choice for
        # something built once, and it is not: on 120 frames of a translucent
        # surface it runs for over ten minutes for a few per cent. 4 encodes in
        # under one.
        method=METHOD,
    )

    width, height = frames[0].size
    size = os.path.getsize(output)
    seconds = len(frames) * DELAY_MS / 1000
    print(
        f"  wrote {output} — {len(frames)} frames, {width}x{height}, "
        f"{1000 / DELAY_MS:.0f} fps, {seconds:.1f} s loop, {size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
