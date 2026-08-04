"""Distances, angles and dihedrals.

Modelled on PyMOL's measurement wizard, where a user picks atoms one at a time
and each pick must be unambiguous. Gala keeps that guarantee — a selection that
matches several atoms raises rather than silently picking one — while staying
scriptable through the ``reduce`` policy (SPECIFICATION §6.3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.entity import AtomStructure, ReducePolicy

__all__ = [
    "Measurement",
    "angle",
    "dihedral",
    "distance",
    "measure",
]


@dataclass(frozen=True)
class Measurement:
    """The result of one measurement.

    Attributes
    ----------
    kind : {"distance", "angle", "dihedral"}
        What was measured.
    value : float
        Ångström for a distance, degrees for an angle or dihedral.
    unit : str
        ``"A"`` or ``"deg"``.
    atoms : tuple[int, ...]
        The atom indices involved, in order. Empty entries appear where a
        centroid was used instead of a single atom.
    points : numpy.ndarray
        Shape ``(n, 3)``: the world-space points, in Blender units.
    labels : tuple[str, ...]
        Human-readable name of each picked atom.
    objects : list
        Blender objects created when ``draw=True``.
    """

    kind: str
    value: float
    unit: str
    atoms: tuple[int, ...]
    points: np.ndarray
    labels: tuple[str, ...] = ()
    objects: list = field(default_factory=list)

    @property
    def text(self) -> str:
        """A formatted value, e.g. ``"2.85 A"`` or ``"109.5 deg"``."""
        if self.unit == "A":
            return f"{self.value:.2f} A"
        return f"{self.value:.1f} deg"

    def __str__(self) -> str:
        return f"{self.kind}: {' - '.join(self.labels)} = {self.text}"

    def __float__(self) -> float:
        return float(self.value)


def _resolve_points(
    structure: AtomStructure,
    selections: Sequence[Any],
    reduce: ReducePolicy | Sequence[ReducePolicy],
) -> tuple[tuple[int, ...], np.ndarray, tuple[str, ...]]:
    """Resolve each selection to one world point, index and label."""
    policies = [reduce] * len(selections) if isinstance(reduce, str) else list(reduce)
    if len(policies) != len(selections):
        raise ValueError(
            f"expected {len(selections)} reduce policies, got {len(policies)}"
        )

    indices: list[int] = []
    points: list[np.ndarray] = []
    labels: list[str] = []

    for selection, policy in zip(selections, policies, strict=True):
        if policy == "centroid":
            points.append(structure.one_point(selection, reduce="centroid"))
            indices.append(-1)
            count = structure.count(selection)
            labels.append(f"centroid of {count} atoms")
        else:
            index = structure.one_index(selection, reduce=policy)
            indices.append(index)
            points.append(structure.world_point(index))
            labels.append(structure.atom_label(index, "{chain}/{resn}{resi}/{name}"))

    return tuple(indices), np.asarray(points, dtype=float), tuple(labels)


def _angstrom(structure: AtomStructure, points: np.ndarray) -> np.ndarray:
    """Convert world-space Blender-unit points to ångström for measurement."""
    return points / structure.world_scale


def distance(
    target: Any,
    selection_a: Any,
    selection_b: Any,
    reduce: ReducePolicy | Sequence[ReducePolicy] = "single",
    draw: bool = False,
    **draw_kwargs: Any,
) -> Measurement:
    """Measure the distance between two atoms.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to measure in.
    selection_a, selection_b : str or array
        Selections identifying the two atoms, e.g.
        ``"chain A and resi 10 and name CA"``.
    reduce : str or sequence of str, optional
        How to resolve a selection matching several atoms. ``"single"``
        (default) raises; ``"centroid"``, ``"first"``, ``"last"`` and
        ``"closest"`` resolve it. Pass a sequence to use a different policy per
        selection.
    draw : bool, optional
        Create a dashed line and a value label in the scene.
    **draw_kwargs
        Forwarded to :func:`blender_gala.measure.draw.draw_measurement`.

    Returns
    -------
    Measurement
        ``value`` is in ångström.

    Raises
    ------
    EmptySelectionError
        If a selection matched no atoms.
    AmbiguousSelectionError
        If a selection matched several atoms under ``reduce="single"``.
    """
    structure = AtomStructure.from_any(target)
    atoms, points, labels = _resolve_points(
        structure, (selection_a, selection_b), reduce
    )
    angstrom = _angstrom(structure, points)
    value = float(np.linalg.norm(angstrom[0] - angstrom[1]))

    result = Measurement("distance", value, "A", atoms, points, labels)
    return _maybe_draw(result, structure, draw, draw_kwargs)


def angle(
    target: Any,
    selection_a: Any,
    selection_b: Any,
    selection_c: Any,
    reduce: ReducePolicy | Sequence[ReducePolicy] = "single",
    draw: bool = False,
    **draw_kwargs: Any,
) -> Measurement:
    """Measure the angle A-B-C, with B at the vertex.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to measure in.
    selection_a, selection_b, selection_c : str or array
        The three atoms; ``selection_b`` is the vertex.
    reduce : str or sequence of str, optional
        See :func:`distance`.
    draw : bool, optional
        Create the two rays, an arc and a value label.
    **draw_kwargs
        Forwarded to the drawing function.

    Returns
    -------
    Measurement
        ``value`` is in degrees, 0 to 180.

    Raises
    ------
    ValueError
        If two of the points coincide, leaving the angle undefined.
    """
    structure = AtomStructure.from_any(target)
    atoms, points, labels = _resolve_points(
        structure, (selection_a, selection_b, selection_c), reduce
    )
    angstrom = _angstrom(structure, points)

    v1 = angstrom[0] - angstrom[1]
    v2 = angstrom[2] - angstrom[1]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        raise ValueError(
            "the angle is undefined because two of the picked atoms coincide"
        )

    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    value = float(np.degrees(np.arccos(cosine)))

    result = Measurement("angle", value, "deg", atoms, points, labels)
    return _maybe_draw(result, structure, draw, draw_kwargs)


def dihedral(
    target: Any,
    selection_a: Any,
    selection_b: Any,
    selection_c: Any,
    selection_d: Any,
    reduce: ReducePolicy | Sequence[ReducePolicy] = "single",
    draw: bool = False,
    **draw_kwargs: Any,
) -> Measurement:
    """Measure the dihedral (torsion) A-B-C-D.

    Uses the IUPAC sign convention: looking along B towards C, a clockwise
    rotation from A to D is positive. This matches PyMOL, and matches how phi
    and psi are reported in a Ramachandran plot.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to measure in.
    selection_a, selection_b, selection_c, selection_d : str or array
        The four atoms in order.
    reduce : str or sequence of str, optional
        See :func:`distance`.
    draw : bool, optional
        Create the bonds, an arc and a value label.
    **draw_kwargs
        Forwarded to the drawing function.

    Returns
    -------
    Measurement
        ``value`` is in degrees, -180 to 180.

    Raises
    ------
    ValueError
        If the four atoms are collinear, leaving the torsion undefined.
    """
    structure = AtomStructure.from_any(target)
    atoms, points, labels = _resolve_points(
        structure, (selection_a, selection_b, selection_c, selection_d), reduce
    )
    angstrom = _angstrom(structure, points)

    b0 = angstrom[0] - angstrom[1]
    b1 = angstrom[2] - angstrom[1]
    b2 = angstrom[3] - angstrom[2]

    norm = float(np.linalg.norm(b1))
    if norm < 1e-9:
        raise ValueError("the dihedral is undefined: the central atoms coincide")
    b1_unit = b1 / norm

    # Project the outer bonds onto the plane perpendicular to the central bond.
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    if np.linalg.norm(v) < 1e-9 or np.linalg.norm(w) < 1e-9:
        raise ValueError("the dihedral is undefined: the four atoms are collinear")

    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_unit, v), w))
    value = float(np.degrees(np.arctan2(y, x)))

    result = Measurement("dihedral", value, "deg", atoms, points, labels)
    return _maybe_draw(result, structure, draw, draw_kwargs)


def measure(
    target: Any,
    *selections: Any,
    reduce: ReducePolicy | Sequence[ReducePolicy] = "single",
    draw: bool = False,
    **draw_kwargs: Any,
) -> Measurement:
    """Measure a distance, angle or dihedral depending on how many atoms are given.

    The scripting equivalent of clicking atoms in PyMOL's wizard: two picks is
    a distance, three an angle, four a dihedral.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to measure in.
    *selections
        Two, three or four selections.
    reduce : str or sequence of str, optional
        See :func:`distance`.
    draw : bool, optional
        Draw the measurement.
    **draw_kwargs
        Forwarded to the drawing function.

    Returns
    -------
    Measurement

    Raises
    ------
    ValueError
        If the number of selections is not 2, 3 or 4.
    """
    dispatch: dict[int, Callable[..., Measurement]] = {
        2: distance,
        3: angle,
        4: dihedral,
    }
    function = dispatch.get(len(selections))
    if function is None:
        raise ValueError(
            f"measure() takes 2 (distance), 3 (angle) or 4 (dihedral) selections, "
            f"got {len(selections)}"
        )
    return function(target, *selections, reduce=reduce, draw=draw, **draw_kwargs)


def _maybe_draw(
    result: Measurement,
    structure: AtomStructure,
    draw: bool,
    draw_kwargs: dict[str, Any],
) -> Measurement:
    if not draw:
        return result
    from .draw import draw_measurement

    result.objects.extend(draw_measurement(result, structure, **draw_kwargs))
    return result
