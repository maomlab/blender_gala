"""Render presets aimed at journal figures.

Blender has no notion of DPI, so presets are expressed in pixels
(SPECIFICATION D-7). To convert: *pixels = inches x dpi*. A single-column
figure in most journals is 3.3 in wide, a double-column one 6.7 in; at the
300 dpi that publishers ask for those are 990 px and 2010 px. ``figure``
therefore defaults to 2000 px, which covers both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["PRESETS", "RenderPreset", "get_preset"]


@dataclass(frozen=True)
class RenderPreset:
    """A named bundle of render settings.

    Attributes
    ----------
    name : str
        Preset identifier.
    resolution : tuple[int, int]
        Output size in pixels.
    samples : int
        Cycles path-tracing samples per pixel.
    preview_samples : int
        Viewport samples.
    adaptive_threshold : float
        Cycles adaptive sampling noise threshold. Lower is cleaner and slower.
    max_bounces : int
        Total light bounce limit.
    transparent : bool
        Whether the film is rendered with an alpha background.
    description : str
        Human-readable summary shown in the UI.
    """

    name: str
    resolution: tuple[int, int]
    samples: int
    preview_samples: int = 32
    adaptive_threshold: float = 0.01
    max_bounces: int = 12
    transparent: bool = True
    description: str = ""

    def scaled(self, factor: float) -> RenderPreset:
        """Return a copy with the resolution scaled by ``factor``."""
        width, height = self.resolution
        return replace(
            self,
            resolution=(max(1, int(width * factor)), max(1, int(height * factor))),
        )


PRESETS: dict[str, RenderPreset] = {
    "draft": RenderPreset(
        name="draft",
        resolution=(960, 960),
        samples=64,
        preview_samples=16,
        adaptive_threshold=0.05,
        max_bounces=6,
        description="Fast look-development render; noisy but shows the composition.",
    ),
    "figure": RenderPreset(
        name="figure",
        resolution=(2000, 2000),
        samples=512,
        preview_samples=32,
        adaptive_threshold=0.01,
        max_bounces=12,
        description="Journal figure at 300 dpi up to 6.7 inches wide.",
    ),
    "print": RenderPreset(
        name="print",
        resolution=(4000, 4000),
        samples=1024,
        preview_samples=32,
        adaptive_threshold=0.005,
        max_bounces=16,
        description="Full-page print quality; slow.",
    ),
    "poster": RenderPreset(
        name="poster",
        resolution=(6000, 6000),
        samples=1024,
        preview_samples=32,
        adaptive_threshold=0.005,
        max_bounces=16,
        description="Conference poster; very slow, very large.",
    ),
    "presentation": RenderPreset(
        name="presentation",
        resolution=(1920, 1080),
        samples=256,
        preview_samples=32,
        adaptive_threshold=0.02,
        max_bounces=8,
        description="16:9 slide-ready still.",
    ),
}


def get_preset(preset: str | RenderPreset) -> RenderPreset:
    """Look up a preset by name.

    Parameters
    ----------
    preset : str or RenderPreset
        A preset name or an already-constructed preset (returned unchanged, so
        callers can accept either).

    Returns
    -------
    RenderPreset

    Raises
    ------
    ValueError
        If ``preset`` is not a string, or is not a known preset name.
    """
    if isinstance(preset, RenderPreset):
        return preset
    # Checked before `.lower()` rather than after: `None`, `3` and a list of
    # names are all plausible things to arrive here — `publication_setup`
    # passes this argument straight through — and each of them otherwise gives
    # an AttributeError about `lower` instead of the error documented above.
    if not isinstance(preset, str):
        raise ValueError(
            f"render preset must be a name or a RenderPreset, got {preset!r}; "
            f"choose from {sorted(PRESETS)}"
        )
    try:
        return PRESETS[preset.lower()]
    except KeyError:
        raise ValueError(
            f"unknown render preset {preset!r}; choose from {sorted(PRESETS)}"
        ) from None
