"""Labelling: in-scene 3D text cards and 2D compositing overlays."""

from __future__ import annotations

from . import labels
from .labels import (
    DEFAULT_TEMPLATE,
    clear_labels,
    label,
    label_atoms,
    label_hud,
    label_residues,
)

__all__ = [
    "DEFAULT_TEMPLATE",
    "clear_labels",
    "label",
    "label_atoms",
    "label_hud",
    "label_residues",
    "labels",
]
