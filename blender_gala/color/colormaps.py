"""Colormaps, implemented natively.

matplotlib is not available inside Blender by default and is far too heavy to
vendor for the sake of a lookup table, so the maps used in structural biology
figures are sampled here from their reference control points and linearly
interpolated (SPECIFICATION §6.4).

All colours are handled as **sRGB** on the way in and converted to **linear**
on the way out, because Blender colour attributes are linear. Skipping that
conversion is why hand-written colouring scripts so often come out washed out.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ALPHAFOLD_BANDS",
    "COLORMAPS",
    "hex_to_rgb",
    "linear_to_srgb",
    "list_colormaps",
    "sample",
    "srgb_to_linear",
]

# Control points sampled evenly from the reference maps. Nine stops reproduces
# each map to within a couple of 8-bit levels once interpolated, which is well
# below what is visible in a figure.
COLORMAPS: dict[str, tuple[str, ...]] = {
    "viridis": (
        "#440154",
        "#472d7b",
        "#3b528b",
        "#2c728e",
        "#21918c",
        "#28ae80",
        "#5ec962",
        "#addc30",
        "#fde725",
    ),
    "plasma": (
        "#0d0887",
        "#4c02a1",
        "#7e03a8",
        "#a92395",
        "#cc4778",
        "#e66c5c",
        "#f89540",
        "#fdc328",
        "#f0f921",
    ),
    "magma": (
        "#000004",
        "#1c1044",
        "#4f127b",
        "#812581",
        "#b5367a",
        "#e55964",
        "#fb8761",
        "#fec287",
        "#fcfdbf",
    ),
    "inferno": (
        "#000004",
        "#1f0c48",
        "#550f6d",
        "#88226a",
        "#ba3655",
        "#e35933",
        "#f98c0a",
        "#f9c932",
        "#fcffa4",
    ),
    "cividis": (
        "#00224e",
        "#123570",
        "#3b496c",
        "#575d6d",
        "#707173",
        "#8a8678",
        "#a59c74",
        "#c3b369",
        "#e1cc55",
    ),
    "coolwarm": (
        "#3b4cc0",
        "#6788ee",
        "#9abbff",
        "#c9d7f0",
        "#edd1c2",
        "#f7a789",
        "#e26952",
        "#b40426",
        "#b40426",
    ),
    "bwr": (
        "#0000ff",
        "#4040ff",
        "#8080ff",
        "#c0c0ff",
        "#ffffff",
        "#ffc0c0",
        "#ff8080",
        "#ff4040",
        "#ff0000",
    ),
    "rdylbu": (
        "#a50026",
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#ffffbf",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695",
    ),
    "spectral": (
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#ffffbf",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",
    ),
    "turbo": (
        "#30123b",
        "#4145ab",
        "#4675ed",
        "#39a2fc",
        "#1bcfd4",
        "#62fc6b",
        "#d2e935",
        "#fea223",
        "#c92903",
    ),
    "grey": ("#000000", "#ffffff"),
    "rainbow": (
        "#0000ff",
        "#00a0ff",
        "#00ffff",
        "#00ff80",
        "#00ff00",
        "#a0ff00",
        "#ffff00",
        "#ff8000",
        "#ff0000",
    ),
    # The AlphaFold DB confidence palette, low to high.
    "alphafold": ("#ff7d45", "#ffdb13", "#65cbf3", "#0053d6"),
}

#: AlphaFold DB confidence bands as ``(lower pLDDT bound, sRGB hex, name)``,
#: ordered from most to least confident.
ALPHAFOLD_BANDS: tuple[tuple[float, str, str], ...] = (
    (90.0, "#0053d6", "Very high (pLDDT > 90)"),
    (70.0, "#65cbf3", "Confident (90 > pLDDT > 70)"),
    (50.0, "#ffdb13", "Low (70 > pLDDT > 50)"),
    (0.0, "#ff7d45", "Very low (pLDDT < 50)"),
)


#: The only characters a hex colour is made of.
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Convert sRGB channel values in ``[0, 1]`` to linear.

    Parameters
    ----------
    values : numpy.ndarray
        sRGB values.

    Returns
    -------
    numpy.ndarray
        Linear values, same shape.
    """
    array = np.asarray(values, dtype=float)
    return np.where(
        array <= 0.04045,
        array / 12.92,
        ((array + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Convert linear channel values in ``[0, 1]`` to sRGB.

    Returns
    -------
    numpy.ndarray
    """
    array = np.asarray(values, dtype=float)
    return np.where(
        array <= 0.0031308,
        array * 12.92,
        1.055 * np.power(np.clip(array, 0.0, None), 1.0 / 2.4) - 0.055,
    )


def hex_to_rgb(colour: str, linear: bool = True) -> np.ndarray:
    """Convert a hex colour string to RGB.

    Parameters
    ----------
    colour : str
        ``"#RRGGBB"`` or ``"RRGGBB"``, case-insensitive.
    linear : bool, optional
        Convert from sRGB to linear. Leave ``True`` for anything written into
        Blender.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)``.

    Raises
    ------
    ValueError
        If the string is not a six-digit hex colour.
    """
    text = colour.lstrip("#").strip()
    if len(text) != 6:
        raise ValueError(f"expected a six-digit hex colour, got {colour!r}")
    # `int(..., 16)` is more generous than a colour is: it takes a sign, and it
    # takes any character Unicode calls a digit, so "+12345" and fullwidth
    # digits would be read as colours and a sign would make a channel negative.
    if not set(text) <= _HEX_DIGITS:
        raise ValueError(f"{colour!r} is not a valid hex colour")
    channels = np.array(
        [int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float
    )
    return srgb_to_linear(channels) if linear else channels


def _stops(name: str) -> np.ndarray:
    try:
        colours = COLORMAPS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown colormap {name!r}; choose from {sorted(COLORMAPS)}"
        ) from None
    return np.array([hex_to_rgb(c) for c in colours])


def sample(name: str, values: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Sample a colormap at normalised positions.

    Parameters
    ----------
    name : str
        A key of :data:`COLORMAPS`.
    values : array_like
        Positions in ``[0, 1]``. Values outside the range are clamped, which
        is the right behaviour for data with outliers: they saturate rather
        than wrapping to the far end of the map.
    reverse : bool, optional
        Reverse the map.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 3)`` of **linear** RGB.

    Raises
    ------
    ValueError
        If the colormap is unknown.
    """
    stops = _stops(name)
    if reverse:
        stops = stops[::-1]

    positions = np.clip(np.asarray(values, dtype=float).ravel(), 0.0, 1.0)
    anchors = np.linspace(0.0, 1.0, len(stops))
    return np.stack(
        [np.interp(positions, anchors, stops[:, channel]) for channel in range(3)],
        axis=1,
    )


def list_colormaps() -> list[str]:
    """Return the available colormap names."""
    return sorted(COLORMAPS)
