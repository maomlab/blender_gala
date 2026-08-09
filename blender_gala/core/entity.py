"""The :class:`AtomStructure` adapter.

Every selection, measurement and interaction routine in Gala takes an
``AtomStructure`` (SPECIFICATION D-4). It pairs the *chemistry* — a biotite
``AtomArray``, where element, atom name, residue name and chain id are readable
as strings — with the *geometry* — a Blender object whose vertex *i* is atom
*i*.

Constructing one from a bare ``AtomArray`` (no Blender object) is fully
supported, which is what makes the science layer testable outside Blender.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import attributes as gala_attributes
from . import mn as mn_bridge
from . import units, viewport
from .exceptions import (
    AmbiguousSelectionError,
    EmptySelectionError,
    StructureError,
)
from .selection import (
    Selection,
    SelectionContext,
    describe_selection,
    expand_selection,
    select,
    select_indices,
)

try:  # pragma: no cover - exercised implicitly by the Blender test run
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = ["AtomStructure", "ReducePolicy"]

#: How a selection matching several atoms is reduced to a single point.
ReducePolicy = str
_REDUCE_POLICIES = ("single", "centroid", "first", "last", "closest")


@dataclass
class AtomStructure:
    """A molecular structure Gala can select from, measure and draw onto.

    Parameters
    ----------
    array : biotite.structure.AtomArray
        Per-atom chemistry. An ``AtomArrayStack`` is reduced to ``frame``.
    object : bpy.types.Object, optional
        The Blender object whose vertices correspond 1:1 with ``array``.
        ``None`` for headless use.
    molecule : molecularnodes.Molecule, optional
        The originating Molecular Nodes entity, when there was one.
    frame : int, optional
        Model index used when ``array`` was a stack.

    Attributes
    ----------
    name : str
        A human-readable label, taken from the Blender object when present.
    """

    array: Any
    object: Any = None
    molecule: Any = None
    frame: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_any(cls, source: Any, frame: int = 0) -> AtomStructure:
        """Build an :class:`AtomStructure` from whatever the caller had.

        Parameters
        ----------
        source : AtomStructure, molecularnodes.Molecule, bpy.types.Object, or AtomArray
            The thing to adapt. A Blender object is resolved back to its
            Molecular Nodes entity through the MN session.
        frame : int, optional
            Model index for multi-model structures.

        Returns
        -------
        AtomStructure

        Raises
        ------
        StructureError
            If ``source`` cannot be interpreted as a molecular structure.
        """
        if isinstance(source, AtomStructure):
            return source

        if mn_bridge.is_molecule(source):
            return cls._from_molecule(source, frame)

        if bpy is not None and isinstance(source, bpy.types.Object):
            return cls._from_object(source, frame)

        if hasattr(source, "coord") and hasattr(source, "element"):
            return cls(array=_single_model(source, frame), frame=frame)

        raise StructureError(
            f"cannot interpret {type(source).__name__} as a molecular structure. "
            "Pass a Molecular Nodes Molecule, a Blender object created by "
            "Molecular Nodes, or a biotite AtomArray."
        )

    @classmethod
    def _from_molecule(cls, molecule: Any, frame: int) -> AtomStructure:
        array = getattr(molecule, "array", None)
        if array is None:
            raise StructureError("Molecular Nodes entity has no atom array")
        return cls(
            array=_single_model(array, frame),
            object=getattr(molecule, "object", None),
            molecule=molecule,
            frame=frame,
        )

    @classmethod
    def _from_object(cls, obj: Any, frame: int) -> AtomStructure:
        molecule = _molecule_for_object(obj)
        if molecule is not None:
            structure = cls._from_molecule(molecule, frame)
            structure.object = obj
            return structure
        raise StructureError(
            f"object {obj.name!r} is not tracked by Molecular Nodes, so its "
            "chemistry is unavailable. Re-import the structure with Molecular "
            "Nodes, or pass the biotite AtomArray directly."
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A readable name for messages and generated object names."""
        if self.object is not None:
            return str(self.object.name)
        return "structure"

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return len(self.array)

    @property
    def coord(self) -> np.ndarray:
        """Atom coordinates in ångström, shape ``(n_atoms, 3)``."""
        coord = np.asarray(self.array.coord, dtype=float)
        if coord.ndim == 3:
            coord = coord[self.frame]
        return coord

    @property
    def world_scale(self) -> float:
        """Blender units per ångström for this structure."""
        return units.world_scale_of(self.object)

    @property
    def context(self) -> SelectionContext:
        """A cached :class:`SelectionContext` for this structure.

        It carries the structure's named selections, so every selection this
        structure evaluates — a colour, an interaction side, a label — can name
        one that was stored from the viewport.
        """
        cached = getattr(self, "_context", None)
        if cached is None or cached.array is not self.array:
            cached = SelectionContext(
                self.array,
                named=gala_attributes.named_selections(self.object, self.n_atoms),
            )
            self._context = cached
        return cached

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def local_positions(self) -> np.ndarray:
        """Atom positions in the object's local space, in Blender units.

        Falls back to ``coord * world_scale`` when there is no Blender object,
        which keeps headless code paths identical to in-Blender ones.
        """
        if self.object is None or getattr(self.object, "data", None) is None:
            return self.coord * self.world_scale

        mesh = self.object.data
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or len(vertices) == 0:
            return self.coord * self.world_scale

        flat = np.empty(len(vertices) * 3, dtype=np.float32)
        vertices.foreach_get("co", flat)
        positions = flat.reshape(-1, 3).astype(float)

        if positions.shape[0] != self.n_atoms:
            # A style modifier or an edit-mode change broke the 1:1 mapping;
            # the atom array remains authoritative.
            return self.coord * self.world_scale
        return positions

    def world_positions(self) -> np.ndarray:
        """Atom positions in world space, in Blender units.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_atoms, 3)``.
        """
        local = self.local_positions()
        if self.object is None:
            return local
        matrix = np.array(self.object.matrix_world).reshape(4, 4)
        homogeneous = np.hstack([local, np.ones((local.shape[0], 1))])
        return (homogeneous @ matrix.T)[:, :3]

    def world_point(self, index: int) -> np.ndarray:
        """World-space position of a single atom, in Blender units."""
        return self.world_positions()[index]

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Return the ``(centre, radius)`` of the atoms in world space.

        Returns
        -------
        tuple of (numpy.ndarray, float)
            Centre in Blender units and radius in Blender units. A structure
            with a single atom gets a small non-zero radius so that callers
            sizing lights or cameras never divide by zero.
        """
        positions = self.world_positions()
        centre = positions.mean(axis=0)
        radius = float(np.linalg.norm(positions - centre, axis=1).max())
        return centre, max(radius, 1e-4)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self, selection: str | Selection | np.ndarray) -> np.ndarray:
        """Return a boolean mask for ``selection``."""
        return select(self.array, selection, self.context)

    def indices(self, selection: str | Selection | np.ndarray) -> np.ndarray:
        """Return the 0-based atom indices matched by ``selection``."""
        return select_indices(self.array, selection, self.context)

    def count(self, selection: str | Selection | np.ndarray) -> int:
        """Return how many atoms ``selection`` matches."""
        return int(self.select(selection).sum())

    def expand(
        self, selection: str | Selection | np.ndarray, level: str = "residue"
    ) -> np.ndarray:
        """Grow ``selection`` to whole residues, chains or bonded fragments.

        See :func:`blender_gala.core.selection.expand_selection`.
        """
        return expand_selection(self.array, selection, level, self.context)

    def describe(self, selection: str | Selection | np.ndarray) -> str:
        """Render ``selection`` as a PyMOL selection string.

        See :func:`blender_gala.core.selection.describe_selection`.
        """
        return describe_selection(self.array, selection, self.context)

    # ------------------------------------------------------------------
    # Viewport selection
    # ------------------------------------------------------------------
    def viewport_selection(self) -> np.ndarray:
        """Return the atoms selected in the viewport as a boolean mask.

        Reads Edit Mode's vertex selection when the object is in Edit Mode,
        and the stored flags otherwise. An all-false mask comes back when
        there is no Blender object behind this structure.
        """
        return viewport.read_selection(self.object, self.n_atoms)

    def set_viewport_selection(
        self, selection: str | Selection | np.ndarray
    ) -> np.ndarray:
        """Select ``selection`` in the viewport, deselecting everything else.

        Returns
        -------
        numpy.ndarray
            The mask that was applied.

        Raises
        ------
        StructureError
            If this structure has no Blender object.
        """
        mask = self.select(selection)
        if self.object is None:
            raise StructureError(
                "this structure has no Blender object, so there is no viewport "
                "selection to set"
            )
        viewport.write_selection(self.object, mask)
        return mask

    # ------------------------------------------------------------------
    # Named selections
    # ------------------------------------------------------------------
    def alias_names(self) -> list[str]:
        """The names of the aliases stored on this structure's object."""
        return gala_attributes.registered(self.object)

    def aliases(self) -> dict[str, np.ndarray]:
        """Every stored alias, as ``{name: mask}``."""
        return {
            name: mask
            for name in self.alias_names()
            if (mask := gala_attributes.read_boolean(self.object, name, self.n_atoms))
            is not None
        }

    def alias(self, name: str) -> np.ndarray:
        """Return one alias as a mask.

        Raises
        ------
        KeyError
            If there is no alias of that name.
        """
        mask = gala_attributes.read_boolean(self.object, name, self.n_atoms)
        if mask is None:
            raise KeyError(f"no selection named {name!r} on {self.name!r}")
        return mask

    def store_alias(
        self, name: str, selection: str | Selection | np.ndarray
    ) -> np.ndarray:
        """Store ``selection`` as a named boolean attribute on the mesh.

        The attribute is what Molecular Nodes reads when a style is limited to
        a selection, and what :func:`blender_gala.save_session` writes out as
        a PyMOL selection.

        Returns
        -------
        numpy.ndarray
            The stored mask.

        Raises
        ------
        StructureError
            If this structure has no Blender object.
        """
        mask = self.select(selection)
        if self.object is None:
            raise StructureError(
                "this structure has no Blender object, so a named selection has "
                "nowhere to live"
            )
        gala_attributes.write_boolean(self.object, name, mask)
        gala_attributes.register(self.object, name)
        self._forget_names()
        return mask

    def delete_alias(self, name: str) -> bool:
        """Remove a stored alias. Returns whether there was one."""
        if self.object is None:
            return False
        gala_attributes.unregister(self.object, name)
        deleted = gala_attributes.delete_boolean(self.object, name)
        self._forget_names()
        return deleted

    def _forget_names(self) -> None:
        """Re-read the stored selections after one of them changed.

        The cached context holds masks it has already been asked for, so a
        selection naming the alias just written would otherwise see the old
        one. The geometry caches beside it are untouched — nothing about the
        structure moved.
        """
        cached = getattr(self, "_context", None)
        if cached is not None:
            cached.named = gala_attributes.named_selections(self.object, self.n_atoms)

    def one_index(
        self,
        selection: str | Selection | np.ndarray,
        reduce: ReducePolicy = "single",
        reference: np.ndarray | None = None,
    ) -> int:
        """Reduce ``selection`` to exactly one atom index.

        Parameters
        ----------
        selection : str, Selection, or numpy.ndarray
            The selection to reduce.
        reduce : {"single", "centroid", "first", "last", "closest"}, optional
            ``"single"`` (the default) requires the selection to match exactly
            one atom, which is what the PyMOL measurement wizard enforces when
            a user clicks atoms. The others resolve ambiguity explicitly.
            ``"centroid"`` is not valid here — use :meth:`one_point`.
        reference : numpy.ndarray, optional
            World-space point used by ``reduce="closest"``.

        Returns
        -------
        int
            A single atom index.

        Raises
        ------
        EmptySelectionError
            If nothing matched.
        AmbiguousSelectionError
            If several atoms matched under ``reduce="single"``.
        ValueError
            If ``reduce`` is not a known policy, or is ``"centroid"``.
        """
        if reduce not in _REDUCE_POLICIES:
            raise ValueError(
                f"unknown reduce policy {reduce!r}; expected one of {_REDUCE_POLICIES}"
            )
        if reduce == "centroid":
            raise ValueError(
                "reduce='centroid' has no single atom index; use one_point() instead"
            )

        indices = self.indices(selection)
        if indices.size == 0:
            raise EmptySelectionError(
                f"selection {_describe(selection)} matched no atoms"
            )
        if reduce == "single":
            if indices.size > 1:
                raise AmbiguousSelectionError(
                    f"selection {_describe(selection)} matched {indices.size} atoms "
                    "but exactly one was required. Narrow the selection, or pass "
                    "reduce='first'/'closest'/'centroid'."
                )
            return int(indices[0])
        if reduce == "first":
            return int(indices[0])
        if reduce == "last":
            return int(indices[-1])

        if reference is None:
            raise ValueError("reduce='closest' requires a reference point")
        positions = self.world_positions()[indices]
        distances = np.linalg.norm(
            positions - np.asarray(reference, dtype=float), axis=1
        )
        return int(indices[int(np.argmin(distances))])

    def one_point(
        self,
        selection: str | Selection | np.ndarray,
        reduce: ReducePolicy = "single",
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reduce ``selection`` to a single world-space point (Blender units).

        Unlike :meth:`one_index` this accepts ``reduce="centroid"``, which
        returns the mean position of the selected atoms — how one measures to
        a ring centre or a whole ligand.
        """
        if reduce == "centroid":
            indices = self.indices(selection)
            if indices.size == 0:
                raise EmptySelectionError(
                    f"selection {_describe(selection)} matched no atoms"
                )
            return self.world_positions()[indices].mean(axis=0)
        return self.world_point(self.one_index(selection, reduce, reference))

    # ------------------------------------------------------------------
    # Atom description
    # ------------------------------------------------------------------
    def atom_label(self, index: int, template: str = "{resn}{resi}/{name}") -> str:
        """Render a label for one atom.

        Parameters
        ----------
        index : int
            Atom index.
        template : str, optional
            A ``str.format`` template. Available fields: ``chain``, ``resi``,
            ``resn``, ``name``, ``elem``, ``b``, ``q``, ``index``, ``one``
            (one-letter residue code).

        Returns
        -------
        str
            The formatted label.
        """
        return template.format(**self.atom_fields(index))

    def atom_fields(self, index: int) -> dict[str, Any]:
        """Return the label fields for one atom. See :meth:`atom_label`."""

        def value(name: str, default: Any = "") -> Any:
            data = getattr(self.array, name, None)
            if data is None:
                return default
            return data[index]

        res_name = str(value("res_name", "")).strip().upper()
        return {
            "chain": str(value("chain_id", "")).strip(),
            "resi": value("res_id", 0),
            "resn": res_name,
            "one": _ONE_LETTER.get(res_name, "X"),
            "name": str(value("atom_name", "")).strip(),
            "elem": str(value("element", "")).strip(),
            "b": float(value("b_factor", 0.0) or 0.0),
            "q": float(value("occupancy", 0.0) or 0.0),
            "index": int(index),
        }

    def __len__(self) -> int:
        return self.n_atoms

    def __repr__(self) -> str:
        return f"AtomStructure(name={self.name!r}, n_atoms={self.n_atoms})"


_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


def _describe(selection: Any) -> str:
    if isinstance(selection, str):
        return repr(selection)
    if isinstance(selection, Selection):
        return repr(selection.text)
    return "<mask>"


def _single_model(array: Any, frame: int) -> Any:
    """Reduce an ``AtomArrayStack`` to one ``AtomArray``."""
    coord = getattr(array, "coord", None)
    if coord is not None and np.asarray(coord).ndim == 3:
        n_models = np.asarray(coord).shape[0]
        if not -n_models <= frame < n_models:
            raise StructureError(
                f"frame {frame} is out of range for a {n_models}-model structure"
            )
        return array[frame]
    return array


def _molecule_for_object(obj: Any) -> Any:
    """Look an object up in the Molecular Nodes session."""
    module = mn_bridge.get_mn()
    if module is None:
        return None
    try:
        session = module.session.get_session()
    except Exception:  # pragma: no cover - session unavailable outside Blender
        return None
    for entity in getattr(session, "entities", {}).values():
        if getattr(entity, "object", None) is obj:
            return entity
    return None
