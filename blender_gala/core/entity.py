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

#: Integer point attributes Molecular Nodes writes onto every vertex, which the
#: biotite array carries too. Comparing them is how the vertex/atom mapping is
#: checked: ``atom_id`` is unique per atom, so it catches a reordering as well
#: as an addition or a deletion, and a vertex made in Edit Mode gets zero.
_IDENTITY_ATTRIBUTES = ("atom_id", "res_id")

#: How far two vertices may disagree about the mesh's offset and still count as
#: the same rigid shift, in Blender units. Vertex positions are stored as
#: float32, so even an untouched mesh disagrees with the array in the last bits;
#: 1e-5 BU is 1e-3 Å at Molecular Nodes' default scale, finer than any structure
#: is deposited at.
_OFFSET_TOLERANCE = 1e-5


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

    def __post_init__(self) -> None:
        """Reduce a stack to the one model this structure is about.

        The reduction used to happen in :meth:`from_any` alone, so a directly
        constructed structure over an ``AtomArrayStack`` reported the *model*
        count as its atom count and read one model through :attr:`coord` and
        another through :attr:`context`. The constructor is public, so it has
        to keep the same promise the factory does.
        """
        reduced = _single_model(self.array, self.frame)
        # The stack itself is kept: once the reduction has happened `frame` is
        # only a record of where the atoms came from, and `at_frame` needs the
        # other models to be able to honour a request for one of them.
        self._models = self.array if reduced is not self.array else None
        self.array = reduced
        # Read now, while the object is certainly there: once it is deleted
        # `object.name` is precisely what raises, and a message about the
        # deletion that cannot say *which* object went is not much of a
        # message. Refreshed whenever `name` is read successfully.
        self._object_name = _name_of(self.object)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_any(cls, source: Any, frame: int | None = None) -> AtomStructure:
        """Build an :class:`AtomStructure` from whatever the caller had.

        Parameters
        ----------
        source : AtomStructure, molecularnodes.Molecule, bpy.types.Object, or AtomArray
            The thing to adapt. A Blender object is resolved back to its
            Molecular Nodes entity through the MN session.
        frame : int, optional
            Model index for multi-model structures. ``None``, the default,
            means *whichever model this already is*: a structure that arrives
            already built comes back untouched, and anything else is read at
            model 0. Naming a frame is a request, and one that cannot be met —
            a structure with no other model to read — is refused rather than
            ignored.

        Returns
        -------
        AtomStructure

        Raises
        ------
        StructureError
            If ``source`` cannot be interpreted as a molecular structure, or
            ``frame`` is not one of its models.
        """
        if isinstance(source, AtomStructure):
            return source if frame is None else source.at_frame(frame)

        index = 0 if frame is None else frame

        if mn_bridge.is_molecule(source):
            return cls._from_molecule(source, index)

        if bpy is not None and isinstance(source, bpy.types.Object):
            return cls._from_object(source, index)

        if hasattr(source, "coord") and hasattr(source, "element"):
            return cls(array=source, frame=index)

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
        try:
            obj = molecule.object
        except AttributeError:
            # Not something that carries an object at all; headless, and the
            # chemistry alone is a complete structure.
            obj = None
        except Exception as exc:
            # `LinkedObjectError`: the entity is still in the session but the
            # object it wrapped has been deleted. Reading it as "no object"
            # would put every later measurement at the default world scale,
            # which is a quiet guess about a molecule that is not there.
            raise StructureError(
                "the Blender object of this Molecular Nodes entity has been "
                "deleted, so its geometry cannot be read. Re-import the "
                "structure, or pass its atom array directly."
            ) from exc
        return cls(array=array, object=obj, molecule=molecule, frame=frame)

    @classmethod
    def _from_object(cls, obj: Any, frame: int) -> AtomStructure:
        molecule = _molecule_for_object(obj)
        if molecule is not None:
            structure = cls._from_molecule(molecule, frame)
            structure.object = obj
            structure._object_name = _name_of(obj)
            return structure
        raise StructureError(
            f"object {obj.name!r} is not tracked by Molecular Nodes, so its "
            "chemistry is unavailable. Re-import the structure with Molecular "
            "Nodes, or pass the biotite AtomArray directly."
        )

    def at_frame(self, frame: int) -> AtomStructure:
        """Return this structure read at another model index.

        Parameters
        ----------
        frame : int
            Model index.

        Returns
        -------
        AtomStructure
            ``self`` when it is already at ``frame``, otherwise a new structure
            over the same models, the same Blender object and the same entity.

        Raises
        ------
        StructureError
            If there is no such model, or none to choose from at all.
        """
        if frame == self.frame:
            return self
        models = self._models
        if models is None:
            # A structure built from a molecule keeps the entity, whose array
            # may still hold every model even though this one does not.
            models = getattr(self.molecule, "array", None)
        coord = getattr(models, "coord", None)
        if coord is None or np.asarray(coord).ndim != 3:
            raise StructureError(
                f"{self._label!r} was built from a single model, so frame {frame} "
                "cannot be read from it"
            )
        return AtomStructure(
            array=models, object=self.object, molecule=self.molecule, frame=frame
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A readable name for messages and generated object names.

        Raises
        ------
        StructureError
            If the Blender object behind this structure has been deleted.
            Reading its name is one of the things that needs it to be there,
            and Blender's own ``ReferenceError`` is not a :class:`GalaError`,
            so nothing above could turn it into a message.
        """
        if self.object is None:
            return "structure"
        viewport.require_object(self.object, self._object_name)
        self._object_name = str(self.object.name)
        return self._object_name

    @property
    def _label(self) -> str:
        """A name for error messages, which must not fail while producing one.

        Every message in this module interpolates the structure's name, so a
        name that raises would turn one failure into two — and :meth:`__repr__`
        would stop working exactly when a user is trying to find out what they
        are holding.
        """
        try:
            return self.name
        except StructureError:
            return f"{self._object_name or 'structure'} (deleted)"

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
        """Blender units per ångström for this structure.

        Raises
        ------
        StructureError
            If the Blender object has been deleted. The default scale would be
            a guess about an object that is no longer there, and everything
            measured with it would be silently in the wrong units.
        """
        viewport.require_object(self.object, self._object_name)
        return units.world_scale_of(self.object)

    @property
    def context(self) -> SelectionContext:
        """A cached :class:`SelectionContext` for this structure.

        It carries the structure's named selections, so every selection this
        structure evaluates — a colour, an interaction side, a label — can name
        one that was stored from the viewport.

        The freshness check is on the *array*, which cannot tell that the
        object behind the stored selections has been deleted — so the two are
        kept from disagreeing at the other end instead: a deleted object simply
        has no names (:func:`.attributes.named_selections`), and a name that
        was cached before the deletion reads back as absent. Either way
        ``select("protein")`` still answers and ``select("pocket")`` is an
        unknown name, whether or not something happened to warm the cache
        first. Selecting is mostly chemistry, and the chemistry did not go
        anywhere.
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
    def geometry_drift(self) -> str | None:
        """Say how the mesh stopped being this structure's atoms, or ``None``.

        The whole adapter rests on vertex *i* being atom *i*, and a user in Edit
        Mode can break that in a way no count notices: delete five vertices, add
        five back, and the mesh is the right length while every atom reads a
        different vertex. Measured that way, a 1.41 Å bond reported as 100 Å and
        a colour landed on seven vertices for a selection of two atoms — both
        reporting success.

        So the check is on identity rather than on length. Molecular Nodes
        writes ``atom_id`` and ``res_id`` as point attributes, the biotite array
        carries the same two, and a vertex created in Edit Mode gets zero — so
        comparing them catches a deletion, an addition *and* a reordering for
        the cost of one attribute read, with no per-atom Python.

        Returns
        -------
        str or None
            A message naming the structure and what no longer lines up, ready
            to be raised or reported. ``None`` when the mesh still holds one
            vertex per atom in order — and when there is no mesh at all, since
            a structure with only chemistry has nothing to disagree with.

        Raises
        ------
        StructureError
            If the Blender object has been deleted.
        """
        viewport.require_object(self.object, self._object_name)
        mesh = getattr(self.object, "data", None) if self.object is not None else None
        vertices = getattr(mesh, "vertices", None)
        if vertices is None:
            return None

        n_points = len(vertices)
        if n_points != self.n_atoms:
            return (
                f"{self._label!r} has {self.n_atoms} atoms but its mesh has "
                f"{n_points} vertices, so vertex i is no longer atom i. Undo the "
                "edit that added or removed vertices, or re-import the structure."
            )

        for name in _IDENTITY_ATTRIBUTES:
            stored = _point_ints(mesh, name)
            expected = getattr(self.array, name, None)
            if stored is None or expected is None:
                continue
            expected = np.asarray(expected)
            if stored.shape != expected.shape:
                # Edit Mode: the values live in the BMesh and the datablock's
                # copy is empty, exactly as `mesh.vertices` is stale there.
                continue
            wrong = int(np.count_nonzero(stored != expected))
            if wrong:
                return (
                    f"{self._label!r} has one vertex per atom, but they are no "
                    f"longer the same atoms: {wrong} of its {n_points} vertices "
                    f"carry a different {name} from the atom they stand for. "
                    "Deleting vertices and adding others back restores the count "
                    "without restoring the correspondence."
                )
        return None

    def require_atom_geometry(self) -> None:
        """Refuse unless the mesh still holds one vertex per atom, in order.

        For the callers that address the mesh *by atom index* — colouring is the
        one whose mistakes reach a figure — and for which guessing is worse than
        failing. See :meth:`geometry_drift`.

        Raises
        ------
        StructureError
            If the mesh no longer corresponds to the atoms, or the Blender
            object has been deleted.
        """
        drift = self.geometry_drift()
        if drift is not None:
            raise StructureError(drift)

    def local_positions(self) -> np.ndarray:
        """Atom positions in the object's local space, in Blender units.

        The mesh is read when — and only when — its vertices still *are* this
        structure's atoms (:meth:`geometry_drift`) and hold the model this
        structure is about. Otherwise the atom array is authoritative and the
        positions come from ``coord * world_scale``, corrected for any rigid
        shift between the array's frame and the mesh's (:meth:`_mesh_offset`).
        That fallback is also the headless path, which keeps code outside
        Blender identical to code inside it.

        Raises
        ------
        StructureError
            If the Blender object has been deleted. Never having had one and
            having lost one are different states: the first has a documented
            answer, the second has coordinates nobody can place in the scene.
        """
        viewport.require_object(self.object, self._object_name)
        mesh = getattr(self.object, "data", None) if self.object is not None else None
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or len(vertices) == 0:
            return self._array_positions(None)

        # `frame=` reaches the chemistry and not the geometry: Molecular Nodes
        # builds the base mesh from the first model and animates the rest
        # through geometry nodes, so reading the mesh for any other frame draws
        # model 0 while `.coord` reports the one that was asked for.
        if not self._mesh_holds_frame() or self.geometry_drift() is not None:
            return self._array_positions(mesh)

        flat = np.empty(len(vertices) * 3, dtype=np.float32)
        vertices.foreach_get("co", flat)
        return flat.reshape(-1, 3).astype(float)

    def _array_positions(self, mesh: Any) -> np.ndarray:
        """The atom array's own coordinates, in the object's local frame."""
        positions = self.coord * self.world_scale
        offset = None if mesh is None else self._mesh_offset(mesh)
        return positions if offset is None else positions + offset

    def _mesh_holds_frame(self) -> bool:
        """Whether the base mesh holds the model this structure is about.

        Only model 0 is ever in the mesh, so any other frame has to be read out
        of the array — the same split :attr:`coord` and :attr:`context` had one
        round ago, reappearing one level out between :attr:`coord` and
        :meth:`world_positions`.
        """
        coord = self._all_model_coords()
        return True if coord is None else self.frame % coord.shape[0] == 0

    def _all_model_coords(self) -> np.ndarray | None:
        """Every model's coordinates, or ``None`` when there is only one."""
        models = self._models
        if models is None:
            # A structure built from a molecule keeps the entity, whose array
            # may still hold every model even though this one does not.
            models = getattr(self.molecule, "array", None)
        coord = getattr(models, "coord", None)
        if coord is None:
            return None
        coord = np.asarray(coord)
        return coord if coord.ndim == 3 else None

    def _mesh_offset(self, mesh: Any) -> np.ndarray | None:
        """The rigid shift from the atom array's frame to the mesh's, or ``None``.

        :func:`blender_gala.scene.origin.set_origin_to_geometry` moves an origin
        by shifting every vertex one way and the object transform the other,
        which leaves world space untouched — but nothing shifts the biotite
        array, so the moment the mesh can no longer be read directly the
        fallback lands a whole origin offset away from what is drawn. Measured
        at 4.4 Å, which is a figure with the labels in the wrong place rather
        than a build that failed.

        The shift is recovered rather than looked up, because the object records
        no such number. It is read off the vertices that can still be named —
        ``atom_id`` is unique per atom, and a vertex added in Edit Mode has zero,
        which is no atom — and taken as the median, so that a handful of atoms
        dragged about in Edit Mode do not veto the correction for all the rest.
        A mesh where most vertices disagree has been deformed rather than moved,
        and there is no single frame to correct into.

        Returns
        -------
        numpy.ndarray or None
            Shape ``(3,)`` in Blender units, or ``None`` when the mesh cannot
            say.
        """
        stored = _point_ints(mesh, "atom_id")
        expected = getattr(self.array, "atom_id", None)
        if stored is None or expected is None or stored.size == 0:
            return None
        expected = np.asarray(expected)
        if expected.size == 0 or np.unique(expected).size != expected.size:
            # Without ids that are unique, no vertex can be tied to one atom.
            return None

        order = np.argsort(expected, kind="stable")
        ordered = expected[order]
        slot = np.clip(np.searchsorted(ordered, stored), 0, ordered.size - 1)
        found = ordered[slot] == stored
        if not found.any():
            return None
        atoms = order[slot[found]]

        vertices = mesh.vertices
        flat = np.empty(len(vertices) * 3, dtype=np.float32)
        vertices.foreach_get("co", flat)
        if flat.size != stored.size * 3:
            # Edit Mode, where the attribute and the vertices are read from
            # different copies of the mesh and need not describe each other.
            return None

        reference = self._mesh_model_coord()[atoms] * self.world_scale
        deltas = flat.reshape(-1, 3).astype(float)[found] - reference
        offset = np.median(deltas, axis=0)
        agreed = np.abs(deltas - offset).max(axis=1) <= _OFFSET_TOLERANCE
        return None if agreed.mean() <= 0.5 else offset

    def _mesh_model_coord(self) -> np.ndarray:
        """The coordinates the mesh was built from — model 0, in ångström."""
        models = self._all_model_coords()
        return self.coord if models is None else np.asarray(models[0], dtype=float)

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

        Raises
        ------
        StructureError
            If no atom of the structure has a position.

        Notes
        -----
        Atoms whose coordinates are not finite are left out rather than making
        the whole answer ``nan``. A structure read from one state of a
        multi-state session carries ``nan`` for the atoms that state does not
        contain, and the bounds of the atoms it *does* contain are the useful
        answer — the same reading the selection language takes of a missing
        coordinate.
        """
        positions = self.world_positions()
        if positions.size:
            positions = positions[np.isfinite(positions).all(axis=1)]
        if positions.size == 0:
            # Framing, lighting and orbiting all start from this, and a
            # structure with nothing placed has no centre to offer them; numpy
            # would say so only as a warning about the mean of an empty slice.
            raise StructureError(f"{self._label!r} has no placed atoms to bound")
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
        self,
        selection: str | Selection | np.ndarray,
        level: str = "residue",
        distance: float = 0.0,
    ) -> np.ndarray:
        """Grow ``selection`` by ``distance`` ångström, then to whole residues.

        See :func:`blender_gala.core.selection.expand_selection`.
        """
        return expand_selection(
            self.array, selection, level, distance, context=self.context
        )

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

        Raises
        ------
        StructureError
            If the Blender object has been deleted, or its vertices are no
            longer this structure's atoms. What the user has selected is real
            either way, and reporting it as "nothing selected" is a false
            answer that the advice which follows it — *select some atoms in
            Edit Mode first* — makes worse.
        """
        self.require_atom_geometry()
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
        """The names of the aliases stored on this structure's object.

        Raises
        ------
        StructureError
            If the Blender object has been deleted. The aliases went with the
            mesh, and an empty list would say there never were any.
        """
        viewport.require_object(self.object, self._object_name)
        return gala_attributes.registered(self.object)

    def aliases(self) -> dict[str, np.ndarray]:
        """Every stored alias, as ``{name: mask}``. See :meth:`alias_names`."""
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
        StructureError
            If the Blender object has been deleted.
        KeyError
            If there is no alias of that name.
        """
        viewport.require_object(self.object, self._object_name)
        mask = gala_attributes.read_boolean(self.object, name, self.n_atoms)
        if mask is None:
            raise KeyError(f"no selection named {name!r} on {self._label!r}")
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
            If this structure has no Blender object, or had one that has since
            been deleted.
        """
        mask = self.select(selection)
        if self.object is None:
            raise StructureError(
                "this structure has no Blender object, so a named selection has "
                "nowhere to live"
            )
        viewport.require_object(self.object, self._object_name)
        gala_attributes.write_boolean(self.object, name, mask)
        gala_attributes.register(self.object, name)
        self._forget_names()
        return mask

    def delete_alias(self, name: str) -> bool:
        """Remove a stored alias. Returns whether there was one.

        Raises
        ------
        StructureError
            If the Blender object has been deleted, which is not the same
            answer as "there was no such alias".
        """
        if self.object is None:
            return False
        viewport.require_object(self.object, self._object_name)
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
        return f"AtomStructure(name={self._label!r}, n_atoms={self.n_atoms})"


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


def _point_ints(mesh: Any, name: str) -> np.ndarray | None:
    """Read an integer point attribute off a mesh, or ``None`` if there is none.

    ``None`` covers every way of not having the data — no mesh, no such
    attribute, or one of another type or domain that happens to share the name —
    because each of them means the same thing to the caller: this attribute
    cannot say whether the vertices are still the atoms.
    """
    attributes = getattr(mesh, "attributes", None)
    attribute = attributes.get(name) if attributes is not None else None
    if attribute is None:
        return None
    if getattr(attribute, "domain", "") != "POINT":
        return None
    if getattr(attribute, "data_type", "") != "INT":
        return None
    values = np.empty(len(attribute.data), dtype=np.int32)
    attribute.data.foreach_get("value", values)
    return values


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
    """Look an object up in the Molecular Nodes session.

    The session outlives the objects it tracks: deleting a molecule leaves an
    entity behind whose ``object`` property raises ``LinkedObjectError``. That
    is not an ``AttributeError``, so a ``getattr`` default never applies and
    the walk used to die on the first dead entry it passed — whichever *live*
    object had been asked for, and with the outcome depending on the order the
    molecules happened to be imported in.
    """
    module = mn_bridge.get_mn()
    if module is None:
        return None
    try:
        session = module.session.get_session()
    except Exception:  # pragma: no cover - session unavailable outside Blender
        return None
    for entity in getattr(session, "entities", {}).values():
        if _entity_object(entity) is obj:
            return entity
    return None


def _entity_object(entity: Any) -> Any:
    """The Blender object of a session entry, or ``None`` if it is gone.

    ``Molecule.object`` raises ``LinkedObjectError`` once the object it wraps
    has been deleted — see :func:`_molecule_for_object`. Every exception is
    caught rather than that one class alone, because the class lives in
    ``databpy`` and importing it here to name it would make a soft dependency
    a hard one.
    """
    try:
        return entity.object
    except Exception:
        return None


def _name_of(obj: Any) -> str | None:
    """The name of a Blender object, or ``None`` when it has none to give.

    Read while the object is known to be there, so that a message written
    after it was deleted can still say which object went: reading ``.name`` is
    exactly what raises once it has.
    """
    if obj is None:
        return None
    try:
        return str(obj.name)
    except Exception:
        return None
