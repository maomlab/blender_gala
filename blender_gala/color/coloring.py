"""Colouring molecules from data.

Colours are written to the mesh ``Color`` attribute (SPECIFICATION D-20) — the
same attribute Molecular Nodes' styles already read — so a recoloured molecule
renders correctly in every style with no node-graph surgery.

The primary test case from the objectives is AlphaFold pLDDT confidence, which
:func:`color_by_plddt` reproduces using the official AlphaFold DB bands.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.entity import AtomStructure
from ..core.exceptions import EmptySelectionError, StructureError
from . import colormaps

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "ColorResult",
    "color_by_attribute",
    "color_by_bfactor",
    "color_by_plddt",
    "color_by_selection",
    "color_from_csv",
    "plddt_legend",
    "read_colors",
    "write_colors",
]

_COLOR_ATTRIBUTE = "Color"


@dataclass
class ColorResult:
    """What a colouring call did.

    Attributes
    ----------
    colors : numpy.ndarray
        Shape ``(n_atoms, 4)`` of linear RGBA that was written.
    n_colored : int
        How many atoms received a colour.
    vmin, vmax : float
        The value range that was mapped, for building a legend.
    legend : list[tuple[str, tuple[float, float, float]]]
        Category or band labels with their colours, when the scheme has them.
    """

    colors: np.ndarray
    n_colored: int
    vmin: float = 0.0
    vmax: float = 1.0
    legend: list[tuple[str, tuple[float, float, float]]] | None = None


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def write_colors(
    target: Any,
    colors: np.ndarray,
    mask: np.ndarray | None = None,
) -> int:
    """Write per-atom colours to the mesh ``Color`` attribute.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to colour.
    colors : numpy.ndarray
        Shape ``(n, 3)`` or ``(n, 4)`` of **linear** RGB(A). ``n`` is either
        the number of atoms, or the number of atoms selected by ``mask``.
    mask : numpy.ndarray, optional
        Boolean mask of atoms to write. Unmasked atoms keep their colour.

    Returns
    -------
    int
        Number of atoms written.

    Raises
    ------
    StructureError
        If the target has no Blender mesh to write to.
    ValueError
        If the array shape does not match the atom count.
    """
    bpy_mod = _require_bpy()
    structure = AtomStructure.from_any(target)
    obj = structure.object
    if obj is None or getattr(obj, "data", None) is None:
        raise StructureError(
            "write_colors needs a Blender object; the structure was loaded without one."
        )

    mesh = obj.data
    n_points = len(mesh.vertices)

    rgba = np.asarray(colors, dtype=float)
    if rgba.ndim != 2 or rgba.shape[1] not in (3, 4):
        raise ValueError(f"colors must have shape (n, 3) or (n, 4), got {rgba.shape}")
    if rgba.shape[1] == 3:
        rgba = np.hstack([rgba, np.ones((rgba.shape[0], 1))])

    attribute = mesh.color_attributes.get(_COLOR_ATTRIBUTE)
    if attribute is None:
        attribute = mesh.color_attributes.new(
            name=_COLOR_ATTRIBUTE, type="FLOAT_COLOR", domain="POINT"
        )

    flat = np.empty(n_points * 4, dtype=np.float32)
    attribute.data.foreach_get("color", flat)
    current = flat.reshape(-1, 4)

    if mask is None:
        if rgba.shape[0] != n_points:
            raise ValueError(
                f"expected {n_points} colours for {n_points} atoms, got {rgba.shape[0]}"
            )
        current[:] = rgba
        written = n_points
    else:
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if rgba.shape[0] == n_points:
            current[indices] = rgba[indices]
        elif rgba.shape[0] == indices.size:
            current[indices] = rgba
        else:
            raise ValueError(
                f"expected {indices.size} or {n_points} colours, got {rgba.shape[0]}"
            )
        written = int(indices.size)

    attribute.data.foreach_set("color", current.astype(np.float32).ravel())
    mesh.update()
    obj.update_tag()
    bpy_mod.context.view_layer.update()
    return written


def read_colors(target: Any) -> np.ndarray:
    """Read the current per-atom colours.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_atoms, 4)`` of linear RGBA. All white if the attribute does
        not exist yet.
    """
    structure = AtomStructure.from_any(target)
    obj = structure.object
    if obj is None:
        raise StructureError("read_colors needs a Blender object")

    mesh = obj.data
    attribute = mesh.color_attributes.get(_COLOR_ATTRIBUTE)
    n_points = len(mesh.vertices)
    if attribute is None:
        return np.ones((n_points, 4), dtype=float)

    flat = np.empty(n_points * 4, dtype=np.float32)
    attribute.data.foreach_get("color", flat)
    return flat.reshape(-1, 4).astype(float)


# ---------------------------------------------------------------------------
# AlphaFold confidence
# ---------------------------------------------------------------------------


def _plddt_values(structure: AtomStructure) -> np.ndarray:
    """Read pLDDT, normalising the 0-1 and 0-100 conventions.

    AlphaFold DB writes pLDDT into the B-factor column on a 0-100 scale, but
    ColabFold and several downstream tools write 0-1. Guessing from the maximum
    is reliable because a real 0-100 pLDDT essentially always exceeds 1.
    """
    values = getattr(structure.array, "b_factor", None)
    if values is None:
        raise StructureError(
            "the structure has no B-factor column, so there is no pLDDT to read"
        )
    plddt = np.asarray(values, dtype=float)
    if plddt.size and np.nanmax(plddt) <= 1.0:
        plddt = plddt * 100.0
    return plddt


def color_by_plddt(
    target: Any,
    selection: Any = "all",
    mode: str = "banded",
    write: bool = True,
) -> ColorResult:
    """Colour a model by AlphaFold pLDDT confidence.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        The predicted model. pLDDT is read from the B-factor column and
        auto-scaled if it is on a 0-1 scale.
    selection : str or array, optional
        Restrict colouring to part of the model.
    mode : {"banded", "continuous"}, optional
        ``"banded"`` reproduces the AlphaFold DB viewer exactly: four flat
        confidence bands. ``"continuous"`` interpolates between the band
        colours, which reads better on a surface but is no longer directly
        comparable to the AFDB.
    write : bool, optional
        Write the colours to the mesh. ``False`` computes them only, which is
        how the tests check the mapping without a Blender object.

    Returns
    -------
    ColorResult
        ``legend`` carries the four band labels and colours.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    StructureError
        If the structure has no B-factor column.
    """
    if mode not in ("banded", "continuous"):
        raise ValueError(f"mode must be 'banded' or 'continuous', got {mode!r}")

    structure = AtomStructure.from_any(target)
    plddt = _plddt_values(structure)
    mask = structure.select(selection)

    rgb = np.zeros((structure.n_atoms, 3), dtype=float)
    if mode == "banded":
        # Assign from least to most confident so higher bands overwrite.
        for lower, hex_colour, _ in sorted(colormaps.ALPHAFOLD_BANDS):
            rgb[plddt >= lower] = colormaps.hex_to_rgb(hex_colour)
    else:
        normalised = np.clip(plddt / 100.0, 0.0, 1.0)
        rgb = colormaps.sample("alphafold", normalised)

    rgba = np.hstack([rgb, np.ones((rgb.shape[0], 1))])
    legend = [
        (name, tuple(colormaps.hex_to_rgb(hex_colour)))
        for _, hex_colour, name in colormaps.ALPHAFOLD_BANDS
    ]

    written = 0
    if write:
        written = write_colors(structure, rgba, mask=mask)

    return ColorResult(
        colors=rgba,
        n_colored=written or int(mask.sum()),
        vmin=0.0,
        vmax=100.0,
        legend=legend,
    )


def plddt_legend() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the AlphaFold confidence legend as ``(label, linear RGB)`` pairs."""
    return [
        (name, tuple(colormaps.hex_to_rgb(hex_colour)))
        for _, hex_colour, name in colormaps.ALPHAFOLD_BANDS
    ]


# ---------------------------------------------------------------------------
# Generic data colouring
# ---------------------------------------------------------------------------


def _expand_values(
    structure: AtomStructure, values: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Expand any accepted ``values`` form into a per-atom array plus a mask.

    Accepts a per-atom array, a per-residue mapping keyed by ``res_id`` or by
    ``(chain_id, res_id)``, or a callable taking the atom index.
    """
    n_atoms = structure.n_atoms
    per_atom = np.full(n_atoms, np.nan, dtype=float)

    if callable(values):
        for index in range(n_atoms):
            per_atom[index] = float(values(index))
        return per_atom, ~np.isnan(per_atom)

    if isinstance(values, Mapping):
        res_ids = np.asarray(getattr(structure.array, "res_id", np.zeros(n_atoms)))
        chains = structure.context.upper("chain_id")
        sample_key = next(iter(values), None)
        keyed_by_pair = isinstance(sample_key, tuple)
        for index in range(n_atoms):
            key = (
                (chains[index], int(res_ids[index]))
                if keyed_by_pair
                else int(res_ids[index])
            )
            if key in values:
                per_atom[index] = float(values[key])
        return per_atom, ~np.isnan(per_atom)

    array = np.asarray(values, dtype=float).ravel()
    if array.size == n_atoms:
        return array, np.isfinite(array)

    raise ValueError(
        f"values has length {array.size} but the structure has {n_atoms} atoms. "
        "Pass a per-atom array, a {res_id: value} mapping, a "
        "{(chain, res_id): value} mapping, or a callable."
    )


def color_by_attribute(
    target: Any,
    values: Any,
    selection: Any = "all",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    reverse: bool = False,
    missing: tuple[float, float, float] = (0.6, 0.6, 0.6),
    write: bool = True,
) -> ColorResult:
    """Colour a molecule by an arbitrary per-atom or per-residue quantity.

    This is the general form of Objective 2's data-annotation colouring:
    conservation scores, B-factors, per-residue energies, experimental
    occupancies, or anything else a user has computed.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to colour.
    values : array_like, Mapping, or callable
        A per-atom array, a ``{res_id: value}`` mapping, a
        ``{(chain, res_id): value}`` mapping, or ``callable(atom_index)``.
    selection : str or array, optional
        Restrict colouring to part of the structure.
    cmap : str, optional
        A key of :data:`blender_gala.color.colormaps.COLORMAPS`.
    vmin, vmax : float, optional
        Value range to map. Defaults to the data's own min and max. Set them
        explicitly when comparing several structures, otherwise each gets its
        own scale and the colours are not comparable.
    reverse : bool, optional
        Reverse the colormap.
    missing : tuple[float, float, float], optional
        Linear RGB for atoms with no value, e.g. residues absent from a
        per-residue mapping.
    write : bool, optional
        Write the colours to the mesh.

    Returns
    -------
    ColorResult

    Raises
    ------
    ValueError
        If ``values`` cannot be matched to the structure, or the colormap is
        unknown.
    """
    structure = AtomStructure.from_any(target)
    per_atom, has_value = _expand_values(structure, values)
    mask = structure.select(selection) & has_value

    if not mask.any():
        raise EmptySelectionError(
            "no atom has both a value and a place in the selection"
        )

    data = per_atom[mask]
    low = float(np.nanmin(data)) if vmin is None else float(vmin)
    high = float(np.nanmax(data)) if vmax is None else float(vmax)
    span = high - low
    if span <= 0:
        # A constant field is not an error; map it to the middle of the ramp.
        normalised = np.full(structure.n_atoms, 0.5)
    else:
        normalised = np.clip((per_atom - low) / span, 0.0, 1.0)
    normalised = np.nan_to_num(normalised, nan=0.5)

    rgb = colormaps.sample(cmap, normalised, reverse=reverse)
    rgb[~has_value] = np.asarray(missing, dtype=float)
    rgba = np.hstack([rgb, np.ones((rgb.shape[0], 1))])

    written = write_colors(structure, rgba, mask=mask) if write else 0
    return ColorResult(
        colors=rgba, n_colored=written or int(mask.sum()), vmin=low, vmax=high
    )


def color_by_bfactor(
    target: Any,
    selection: Any = "all",
    cmap: str = "coolwarm",
    **kwargs: Any,
) -> ColorResult:
    """Colour by crystallographic B-factor. Shorthand for :func:`color_by_attribute`.

    Returns
    -------
    ColorResult
    """
    structure = AtomStructure.from_any(target)
    values = getattr(structure.array, "b_factor", None)
    if values is None:
        raise StructureError("the structure has no B-factor column")
    return color_by_attribute(
        structure, np.asarray(values, dtype=float), selection, cmap=cmap, **kwargs
    )


def color_by_selection(
    target: Any,
    scheme: Mapping[str, Any],
    default: Any = "#cccccc",
    write: bool = True,
) -> ColorResult:
    """Colour categorically, one colour per selection.

    Later entries win where selections overlap, so order the mapping from
    general to specific — background chain first, highlighted site last.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to colour.
    scheme : Mapping[str, str or sequence]
        Selection string to colour, as a hex string or an RGB(A) sequence in
        **sRGB**.
    default : str or sequence, optional
        Colour for atoms matched by nothing.
    write : bool, optional
        Write the colours to the mesh.

    Returns
    -------
    ColorResult
        ``legend`` carries one entry per selection.
    """
    structure = AtomStructure.from_any(target)

    rgb = np.tile(_as_linear_rgb(default), (structure.n_atoms, 1))
    covered = np.zeros(structure.n_atoms, dtype=bool)
    legend: list[tuple[str, tuple[float, float, float]]] = []

    for selection, colour in scheme.items():
        mask = structure.select(selection)
        linear = _as_linear_rgb(colour)
        rgb[mask] = linear
        covered |= mask
        legend.append((str(selection), tuple(linear)))

    rgba = np.hstack([rgb, np.ones((rgb.shape[0], 1))])
    written = write_colors(structure, rgba) if write else 0
    return ColorResult(
        colors=rgba, n_colored=written or int(covered.sum()), legend=legend
    )


def _as_linear_rgb(colour: Any) -> np.ndarray:
    if isinstance(colour, str):
        return colormaps.hex_to_rgb(colour)
    array = np.asarray(colour, dtype=float)[:3]
    return colormaps.srgb_to_linear(array)


def color_from_csv(
    target: Any,
    filepath: str,
    value_column: str,
    res_id_column: str = "res_id",
    chain_column: str | None = None,
    **kwargs: Any,
) -> ColorResult:
    """Colour from a CSV of per-residue values.

    The common real-world case: a conservation score, a per-residue ddG, or a
    deep-mutational-scanning summary produced somewhere else entirely.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to colour.
    filepath : str
        CSV file with a header row.
    value_column : str
        Column holding the numeric value.
    res_id_column : str, optional
        Column holding the residue number.
    chain_column : str, optional
        Column holding the chain id. Supply it whenever the structure has more
        than one chain, or residue numbers will collide between them.
    **kwargs
        Forwarded to :func:`color_by_attribute`.

    Returns
    -------
    ColorResult

    Raises
    ------
    KeyError
        If a named column is missing from the file.
    """
    values: dict[Any, float] = {}
    with open(filepath, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in (value_column, res_id_column, chain_column)
            if column is not None and column not in (reader.fieldnames or [])
        ]
        if missing:
            raise KeyError(
                f"{filepath} has no column(s) {missing}; found {reader.fieldnames}"
            )
        for row in reader:
            raw = row[value_column]
            if raw is None or raw.strip() == "":
                continue
            res_id = int(float(row[res_id_column]))
            key: Any = (
                (row[chain_column].strip().upper(), res_id) if chain_column else res_id
            )
            values[key] = float(raw)

    return color_by_attribute(target, values, **kwargs)
