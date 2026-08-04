"""Data-driven colouring, with AlphaFold pLDDT as the primary case."""

from __future__ import annotations

from . import coloring, colormaps
from .coloring import (
    ColorResult,
    color_by_attribute,
    color_by_bfactor,
    color_by_plddt,
    color_by_selection,
    color_from_csv,
    plddt_legend,
    read_colors,
    write_colors,
)
from .colormaps import (
    ALPHAFOLD_BANDS,
    COLORMAPS,
    hex_to_rgb,
    linear_to_srgb,
    list_colormaps,
    sample,
    srgb_to_linear,
)

__all__ = [
    "ALPHAFOLD_BANDS",
    "COLORMAPS",
    "ColorResult",
    "color_by_attribute",
    "color_by_bfactor",
    "color_by_plddt",
    "color_by_selection",
    "color_from_csv",
    "coloring",
    "colormaps",
    "hex_to_rgb",
    "linear_to_srgb",
    "list_colormaps",
    "plddt_legend",
    "read_colors",
    "sample",
    "srgb_to_linear",
    "write_colors",
]
