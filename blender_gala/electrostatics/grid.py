"""OpenDX scalar grids: reading them, and reading values out of them.

APBS writes its electrostatic potential as an OpenDX file — a header giving
the grid's origin and spacing, then one number per point with *z* varying
fastest. This module turns that into a :class:`PotentialGrid` and interpolates
values out of it at arbitrary coordinates.

Nothing here imports ``bpy``: a potential map is chemistry, not scene, and
being able to read and check one outside Blender is what makes it testable.

Coordinates are in **ångström**, in the frame of the structure APBS was given
— which is the frame of the deposited coordinates, not of anything Blender has
since done to the object. Values are in **kT/e**, the unit APBS writes and the
one every published ramp is quoted in.
"""

from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["PotentialGrid", "read_dx"]

#: Lines that matter, and what they carry.
_COUNTS = re.compile(r"^object\s+\d+\s+class\s+gridpositions\s+counts\s+(.+)$")
_ORIGIN = re.compile(r"^origin\s+(.+)$")
_DELTA = re.compile(r"^delta\s+(.+)$")
_ITEMS = re.compile(r"^object\s+\d+\s+class\s+array.*?items\s+(\d+)")


@dataclass(frozen=True)
class PotentialGrid:
    """A scalar field on a regular axis-aligned grid.

    Parameters
    ----------
    values : numpy.ndarray
        Shape ``(nx, ny, nz)``.
    origin : numpy.ndarray
        Coordinates of ``values[0, 0, 0]``, in ångström.
    spacing : numpy.ndarray
        Grid step along each axis, in ångström.
    unit : str, optional
        What the values are in. APBS writes kT/e.
    source : str, optional
        Where it came from, for the report a vignette or the UI prints.

    Attributes
    ----------
    shape : tuple[int, int, int]
        Points along each axis.
    """

    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    unit: str = "kT/e"
    source: str | None = field(default=None, compare=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)  # type: ignore[return-value]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lowest and highest coordinate the grid covers, in ångström."""
        extent = (np.asarray(self.shape) - 1) * self.spacing
        return self.origin, self.origin + extent

    def sample(self, points: Any, outside: str = "clamp") -> np.ndarray:
        """Interpolate the field at ``points``.

        Trilinear, because that is what the field is: APBS solves on this grid
        and everything between the nodes is interpolation whoever does it.

        Parameters
        ----------
        points : array_like
            Shape ``(n, 3)`` in ångström, in the structure's frame.
        outside : {"clamp", "nan"}, optional
            What to do with points beyond the grid. ``"clamp"`` returns the
            value at the nearest face, which is where the potential is
            smallest and smoothest anyway; ``"nan"`` marks them, which is how
            you find out the box was too small.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``.

        Raises
        ------
        ValueError
            If ``outside`` is not one of the two, or ``points`` is not ``(n, 3)``.
        """
        if outside not in ("clamp", "nan"):
            raise ValueError(f"outside must be 'clamp' or 'nan', got {outside!r}")

        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {coordinates.shape}")

        fractional = (coordinates - self.origin) / self.spacing
        counts = np.asarray(self.shape)
        beyond = (fractional < 0).any(axis=1) | (fractional > counts - 1).any(axis=1)

        clamped = np.clip(fractional, 0.0, counts - 1.0)
        lower = np.floor(clamped).astype(int)
        # The last node has no cell above it, so a point exactly on the far
        # face has to interpolate within the cell below it instead.
        lower = np.minimum(lower, counts - 2)
        weight = clamped - lower

        result = np.zeros(len(coordinates))
        for corner in range(8):
            offset = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
            share = np.prod(np.where(offset == 1, weight, 1.0 - weight), axis=1)
            index = lower + offset
            result += share * self.values[index[:, 0], index[:, 1], index[:, 2]]

        if outside == "nan":
            result[beyond] = np.nan
        return result

    def summary(self) -> str:
        """A one-block description, for a vignette or the UI to print."""
        low, high = self.bounds
        return "\n".join(
            [
                f"{self.source or 'potential grid'}",
                f"  grid    : {'x'.join(str(n) for n in self.shape)} points, "
                f"{self.spacing[0]:.2f} A spacing",
                f"  box     : {low[0]:.1f} {low[1]:.1f} {low[2]:.1f} to "
                f"{high[0]:.1f} {high[1]:.1f} {high[2]:.1f} A",
                f"  values  : {self.values.min():.2f} to {self.values.max():.2f} "
                f"{self.unit}",
            ]
        )


def read_dx(path: str) -> PotentialGrid:
    """Read an OpenDX scalar grid, as written by APBS.

    Parameters
    ----------
    path : str
        An ``.dx`` file, or a gzipped one.

    Returns
    -------
    PotentialGrid

    Raises
    ------
    ValueError
        If the header is missing, incomplete, describes a grid whose axes are
        not aligned with the coordinate axes, or promises more values than the
        file contains.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        counts: np.ndarray | None = None
        origin: np.ndarray | None = None
        deltas: list[np.ndarray] = []
        items: int | None = None

        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            match = _COUNTS.match(stripped)
            if match:
                counts = np.array([int(n) for n in match.group(1).split()])
                continue
            match = _ORIGIN.match(stripped)
            if match:
                origin = np.array([float(v) for v in match.group(1).split()])
                continue
            match = _DELTA.match(stripped)
            if match:
                deltas.append(np.array([float(v) for v in match.group(1).split()]))
                continue
            match = _ITEMS.match(stripped)
            if match:
                items = int(match.group(1))
                break

        if counts is None or origin is None or items is None or len(deltas) != 3:
            raise ValueError(f"{path}: not an OpenDX grid, or the header is truncated")

        matrix = np.array(deltas)
        off_axis = matrix - np.diag(np.diag(matrix))
        if np.abs(off_axis).max() > 1e-9:
            # A general lattice would need the inverse of the delta matrix in
            # `sample`, and APBS has never written one. Refusing beats
            # interpolating in the wrong basis and looking plausible.
            raise ValueError(
                f"{path}: the grid axes are not aligned with the coordinate axes, "
                "which this reader does not support"
            )
        spacing = np.diag(matrix)

        # The data block is the rest of the file, whitespace separated, up to
        # `items` values; the trailer after it is not numeric.
        values = np.empty(items, dtype=float)
        filled = 0
        for line in handle:
            fields = line.split()
            if not fields or not _looks_numeric(fields[0]):
                break
            take = min(len(fields), items - filled)
            values[filled : filled + take] = [float(v) for v in fields[:take]]
            filled += take
            if filled == items:
                break

    if filled != items:
        raise ValueError(
            f"{path}: header promised {items} values, the file has {filled}"
        )
    expected = int(np.prod(counts))
    if items != expected:
        raise ValueError(
            f"{path}: {items} values for a {'x'.join(str(n) for n in counts)} grid"
        )

    # OpenDX writes z fastest, which is C order over (nx, ny, nz).
    return PotentialGrid(
        values=values.reshape(tuple(int(n) for n in counts)),
        origin=origin,
        spacing=spacing,
        source=os.path.basename(path),
    )


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
