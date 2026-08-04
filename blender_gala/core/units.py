"""Unit conversion between ångström and Blender units.

Molecular Nodes imports structures at a world scale of ``0.01`` — one ångström
becomes 0.01 Blender units, so a 100 Å protein is 1 unit across and sits
comfortably in Blender's default clipping range and light falloff.

Every Blender Gala public function takes and returns **ångström**. Conversion
happens only where a value crosses into or out of Blender (SPECIFICATION §3.2).
The scale is read from the object when possible rather than hard-coded, so a
molecule imported at a non-default scale still measures correctly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_WORLD_SCALE",
    "angstrom_to_bu",
    "bu_to_angstrom",
    "world_scale_of",
]

#: Molecular Nodes' default import scale: 1 Å -> 0.01 Blender units.
DEFAULT_WORLD_SCALE = 0.01

#: Custom properties that Molecular Nodes (or Gala) may store the scale under.
_SCALE_KEYS = ("world_scale", "mn_world_scale", "gala_world_scale")


def world_scale_of(obj: Any = None) -> float:
    """Return the ångström-to-Blender-unit scale factor for ``obj``.

    Parameters
    ----------
    obj : bpy.types.Object or None, optional
        Object to inspect for a stored world scale. ``None`` returns the
        default.

    Returns
    -------
    float
        Blender units per ångström. ``0.01`` unless the object records
        otherwise.
    """
    if obj is None:
        return DEFAULT_WORLD_SCALE

    for key in _SCALE_KEYS:
        try:
            value = obj[key]
        except (KeyError, TypeError):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return value

    return DEFAULT_WORLD_SCALE


def angstrom_to_bu(value: Any, scale: float = DEFAULT_WORLD_SCALE) -> Any:
    """Convert ångström to Blender units.

    Parameters
    ----------
    value : float or array_like
        Distance(s) in ångström.
    scale : float, optional
        Blender units per ångström.

    Returns
    -------
    float or numpy.ndarray
        The converted value, preserving scalar-ness.
    """
    if np.isscalar(value):
        return float(value) * scale  # type: ignore[arg-type]
    return np.asarray(value, dtype=float) * scale


def bu_to_angstrom(value: Any, scale: float = DEFAULT_WORLD_SCALE) -> Any:
    """Convert Blender units to ångström.

    Parameters
    ----------
    value : float or array_like
        Distance(s) in Blender units.
    scale : float, optional
        Blender units per ångström.

    Returns
    -------
    float or numpy.ndarray
        The converted value, preserving scalar-ness.

    Raises
    ------
    ValueError
        If ``scale`` is not positive.
    """
    if scale <= 0.0:
        raise ValueError(f"world scale must be positive, got {scale}")
    if np.isscalar(value):
        return float(value) / scale  # type: ignore[arg-type]
    return np.asarray(value, dtype=float) / scale
