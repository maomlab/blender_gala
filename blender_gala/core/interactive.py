"""Interactive selection: the viewport, selection levels and named aliases.

The functions here are the ones the sidebar panel and the Python API share.
Each takes whatever the caller has — a Molecular Nodes ``Molecule``, a Blender
object, an :class:`~blender_gala.core.entity.AtomStructure` — and does one
thing, so that the operators in :mod:`blender_gala.ops.operators` stay three
lines long (SPECIFICATION D-21).

The workflow they add up to is PyMOL's, split across the two applications
that are good at each half::

    box-select some atoms in Edit Mode      # Blender already does this well
    gala.expand_viewport_selection(mol, "residue")
    gala.describe_viewport_selection(mol)   # 'chain A and resi 45-47'
    gala.create_alias(mol, "pocket")
    gala.style_alias(mol, "pocket", style="ball_and_stick")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import attributes
from .entity import AtomStructure
from .exceptions import StructureError
from .selection import LEVELS, Selection

__all__ = [
    "alias_combine",
    "create_alias",
    "delete_alias",
    "describe_viewport_selection",
    "expand_viewport_selection",
    "list_aliases",
    "select_alias",
    "set_viewport_selection",
    "style_alias",
    "viewport_selection",
]


def viewport_selection(target: Any) -> np.ndarray:
    """Return the atoms selected in the viewport, as a boolean mask.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        The molecule to read from.

    Returns
    -------
    numpy.ndarray
        Boolean mask, all false when nothing is selected.
    """
    return AtomStructure.from_any(target).viewport_selection()


def set_viewport_selection(
    target: Any, selection: str | Selection | np.ndarray
) -> np.ndarray:
    """Select ``selection`` in the viewport, deselecting everything else.

    The inverse of :func:`describe_viewport_selection`, and the quickest way
    to see what a typed selection actually covers before committing it to a
    figure.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        The molecule to select in.
    selection : str, Selection, or numpy.ndarray
        A PyMOL-style selection string, or a mask.

    Returns
    -------
    numpy.ndarray
        The mask that was applied.
    """
    return AtomStructure.from_any(target).set_viewport_selection(selection)


def expand_viewport_selection(target: Any, level: str = "residue") -> np.ndarray:
    """Grow the viewport selection to whole residues, chains or fragments.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        The molecule to expand in.
    level : {"atom", "residue", "chain", "fragment", "object"}, optional
        How far to grow. Levels compose: expanding twice at ``"chain"`` after
        ``"residue"`` grows the residues to their chains.

    Returns
    -------
    numpy.ndarray
        The expanded mask, which has also been applied to the viewport.

    Raises
    ------
    ValueError
        If ``level`` is not one of :data:`~blender_gala.core.selection.LEVELS`.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown selection level {level!r}; expected one of {LEVELS}")
    structure = AtomStructure.from_any(target)
    expanded = structure.expand(structure.viewport_selection(), level)
    structure.set_viewport_selection(expanded)
    return expanded


def describe_viewport_selection(target: Any) -> str:
    """Render the viewport selection as a PyMOL selection string.

    Returns
    -------
    str
        A selection string that evaluates back to the selected atoms, or
        ``"none"`` when nothing is selected.
    """
    structure = AtomStructure.from_any(target)
    return structure.describe(structure.viewport_selection())


# ---------------------------------------------------------------------------
# Named selections
# ---------------------------------------------------------------------------


def create_alias(
    target: Any, name: str, selection: str | Selection | np.ndarray | None = None
) -> str:
    """Store a selection under a name, for styling and for PyMOL.

    The alias becomes a boolean attribute on the mesh, which is what
    Molecular Nodes reads when a style is limited to a selection and what
    :func:`blender_gala.save_session` writes out as a PyMOL selection.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        The molecule to store the alias on.
    name : str
        What to call it. Characters an attribute name cannot carry are
        replaced, so ``"binding site"`` is stored as ``binding_site``.
    selection : str, Selection, or numpy.ndarray, optional
        What to store. Defaults to whatever is selected in the viewport.

    Returns
    -------
    str
        The name the alias was actually stored under.

    Raises
    ------
    ValueError
        If the selection is empty — an alias matching nothing would style
        nothing, and is more likely a mistake than an intention.
    """
    structure = AtomStructure.from_any(target)
    mask = (
        structure.viewport_selection()
        if selection is None
        else structure.select(selection)
    )
    if not mask.any():
        raise ValueError(
            f"nothing to store as {name!r}: the selection is empty. Select some "
            "atoms in Edit Mode first, or pass a selection."
        )
    stored = attributes.safe_name(name)
    structure.store_alias(stored, mask)
    return stored


def list_aliases(target: Any) -> dict[str, np.ndarray]:
    """Every alias stored on ``target``, as ``{name: mask}``."""
    return AtomStructure.from_any(target).aliases()


def select_alias(target: Any, name: str) -> np.ndarray:
    """Select a stored alias in the viewport.

    Returns
    -------
    numpy.ndarray
        The mask that was applied.
    """
    structure = AtomStructure.from_any(target)
    return structure.set_viewport_selection(structure.alias(name))


def delete_alias(target: Any, name: str) -> bool:
    """Remove a stored alias. Returns whether there was one to remove.

    Any style already limited to it keeps pointing at a name that no longer
    resolves, and will show nothing — so this deliberately does not go looking
    through the node tree to tidy up after itself.
    """
    return AtomStructure.from_any(target).delete_alias(name)


def style_alias(
    target: Any,
    name: str,
    style: str = "ball_and_stick",
    color: str | None = "common",
    material: Any = None,
) -> Any:
    """Add a Molecular Nodes style limited to a stored alias.

    Molecular Nodes takes the attribute name straight through: ``add_style``
    wires a Named Attribute node into the style's ``Selection`` socket, so the
    new style covers exactly the atoms in the alias and the existing styles
    are left alone.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        The molecule to style. Must be tracked by Molecular Nodes.
    name : str
        An alias created by :func:`create_alias`.
    style : str, optional
        A Molecular Nodes style name, such as ``"spheres"``, ``"cartoon"`` or
        ``"ball_and_stick"``.
    color : str or None, optional
        The colouring scheme to apply to the new style branch.
    material : bpy.types.Material or str, optional
        Material for the new style.

    Returns
    -------
    molecularnodes.Molecule
        The molecule, for chaining.

    Raises
    ------
    StructureError
        If there is no such alias, or the molecule did not come from
        Molecular Nodes.
    """
    structure = AtomStructure.from_any(target)
    if attributes.read_boolean(structure.object, name, structure.n_atoms) is None:
        available = ", ".join(structure.alias_names()) or "none"
        raise StructureError(
            f"no selection named {name!r} to style. Stored selections: {available}."
        )

    molecule = structure.molecule
    if molecule is None or not hasattr(molecule, "add_style"):
        raise StructureError(
            "styling a selection needs the Molecular Nodes entity behind this "
            "object. Re-import the structure with Molecular Nodes."
        )

    kwargs: dict[str, Any] = {"style": style, "selection": name}
    if color is not None:
        kwargs["color"] = color
    if material is not None:
        kwargs["material"] = material
    molecule.add_style(**kwargs)
    return molecule


def alias_combine(
    target: Any,
    name: str,
    selection: str | Selection | np.ndarray | None = None,
    mode: str = "union",
) -> np.ndarray:
    """Combine a stored alias with another selection, in place.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        The molecule holding the alias.
    name : str
        The alias to update.
    selection : str, Selection, or numpy.ndarray, optional
        The other operand. Defaults to the viewport selection.
    mode : {"union", "intersect", "subtract"}, optional
        How to combine them.

    Returns
    -------
    numpy.ndarray
        The new mask, which has been written back to the alias.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three.
    KeyError
        If there is no alias of that name.
    """
    modes = ("union", "intersect", "subtract")
    if mode not in modes:
        raise ValueError(f"unknown mode {mode!r}; expected one of {modes}")

    structure = AtomStructure.from_any(target)
    current = structure.alias(name)
    other = (
        structure.viewport_selection()
        if selection is None
        else structure.select(selection)
    )

    if mode == "union":
        combined = current | other
    elif mode == "intersect":
        combined = current & other
    else:
        combined = current & ~other

    structure.store_alias(name, combined)
    return combined
