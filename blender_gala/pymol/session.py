"""Reading and writing PyMOL session files.

A ``.pse`` is a pickled ``dict``. That is the whole format: PyMOL builds a
tree of plain lists describing every named object and pickles it. So a session
can be read without PyMOL, which is the point — Blender's interpreter has no
PyMOL in it, and asking a user to install one to open their own figure would
defeat the exercise.

Nothing here imports ``bpy``: a session is data, and being able to read one
outside Blender is what makes it testable. :mod:`blender_gala.pymol.load` and
:mod:`blender_gala.pymol.save` are the halves that touch a scene.

Unpickling is arbitrary code execution, and a ``.pse`` is a file people email
each other. :class:`_Unpickler` therefore refuses every global except the
handful a genuine session contains — see :data:`ALLOWED_GLOBALS`.

Coordinates are **ångström** in the frame PyMOL held them in. Colours are
indices into :mod:`blender_gala.pymol.palette` plus whatever the session
defines itself.

The layout below was read off sessions written by PyMOL 3.1.8 at every
``pse_export_version`` from 1.7 to 3.0; where those disagree the difference is
noted at the point it matters.
"""

from __future__ import annotations

import codecs
import gzip
import io
import os
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.exceptions import GalaError
from . import palette

__all__ = [
    "ALLOWED_GLOBALS",
    "OBJECT_TYPES",
    "REPS",
    "PymolMeasurement",
    "PymolMolecule",
    "PymolSelection",
    "PymolSession",
    "PymolSessionError",
    "PymolView",
    "read_session",
    "write_session",
]


class PymolSessionError(GalaError):
    """A session file could not be read, or could not be read faithfully."""


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

#: Globals a real session contains. Anything else is refused rather than
#: imported: this is a pickle, and importing what it names is how a session
#: file would run code. ``Session_Storage`` is an attribute bag PyMOL keeps
#: its Python-side state in; ``_codecs.encode`` appears wherever a Python 2
#: string was pickled.
ALLOWED_GLOBALS = frozenset(
    {
        ("pymol", "Session_Storage"),
        ("_codecs", "encode"),
        ("copy_reg", "_reconstructor"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)

#: PyMOL's object type codes, as stored in a name record.
OBJECT_TYPES = {
    1: "molecule",
    2: "map",
    3: "mesh",
    4: "measurement",
    5: "callback",
    6: "cgo",
    7: "surface",
    8: "gadget",
    9: "slice",
    11: "alignment",
    12: "group",
    13: "volume",
}

#: Representations, in bit order. Verified by showing each one in PyMOL and
#: reading back the mask it wrote.
REPS = (
    "sticks",
    "spheres",
    "surface",
    "labels",
    "nb_spheres",
    "cartoon",
    "ribbon",
    "lines",
    "mesh",
    "dots",
    "dashes",
    "nonbonded",
    "cell",
    "cgo",
    "callback",
    "extent",
    "slice",
    "angles",
    "dihedrals",
    "ellipsoids",
    "volume",
)

#: Settings Gala reads, by PyMOL's numeric setting index.
SETTINGS = {
    6: "bg_rgb",
    23: "orthoscopic",
    138: "transparency",
    152: "field_of_view",
    155: "sphere_scale",
    172: "sphere_transparency",
    279: "cartoon_transparency",
    453: "label_size",
}

# --- name record ----------------------------------------------------------
_REC_NAME, _REC_EXEC, _REC_VISIBLE, _REC_REPON, _REC_TYPE, _REC_DATA, _REC_GROUP = (
    range(7)
)
_EXEC_OBJECT, _EXEC_SELECTION = 0, 1

# --- ObjectMolecule payload ----------------------------------------------
_MOL_HEADER, _MOL_NCSET, _MOL_NBOND, _MOL_NATOM = 0, 1, 2, 3
_MOL_CSETS, _MOL_CSTMPL, _MOL_BONDS, _MOL_ATOMS = 4, 5, 6, 7
_MOL_DISCRETE = 8

# --- object header --------------------------------------------------------
_HDR_COLOR, _HDR_VISREP, _HDR_TTT_FLAG, _HDR_SETTINGS = 2, 3, 7, 8
_HDR_TTT = 11

# --- coordinate set -------------------------------------------------------
_CS_NINDEX, _CS_COORD, _CS_IDX_TO_ATM = 0, 2, 3

# --- atom info ------------------------------------------------------------
# Field order is identical in every export version from 1.7 to 3.0; later
# PyMOLs append rather than reorder, so reading a prefix is safe.
_AI_RESV, _AI_CHAIN, _AI_ALT, _AI_RESI, _AI_SEGI = 0, 1, 2, 3, 4
_AI_RESN, _AI_NAME, _AI_ELEM, _AI_TEXT_TYPE, _AI_LABEL, _AI_SS = 5, 6, 7, 8, 9, 10
_AI_B, _AI_Q, _AI_VDW, _AI_PARTIAL_CHARGE, _AI_FORMAL_CHARGE = 14, 15, 16, 17, 18
_AI_HETATM, _AI_VISREP, _AI_COLOR, _AI_ID = 19, 20, 21, 22
_AI_MIN_FIELDS = 23

#: How many fields :func:`write_session` emits per atom. PyMOL reads a fixed
#: count for the file version it is told, and reads it without checking the
#: list is that long — a short row segfaults it rather than raising. 49 is
#: what PyMOL 3 writes and expects.
_AI_FIELDS = 49
_AI_PROTONS, _AI_UNIQUE_ID, _AI_RANK = 31, 32, 36
_AI_FLAGS, _AI_BONDED, _AI_CHEMFLAG, _AI_GEOM, _AI_VALENCE = 24, 25, 26, 27, 28
_AI_ELEC_RADIUS = 35
_AI_ANISOU = 41
_AI_CUSTOM = 47

#: Classification bits in the ``flags`` field. PyMOL's ``polymer``,
#: ``solvent``, ``organic`` and ``inorganic`` selection macros read these
#: rather than re-deriving them, so an object exported with zero flags is in
#: none of those classes: ``show cartoon`` finds no polymer and draws nothing.
#: The exact patterns are what PyMOL itself writes, read back off a session it
#: saved for a structure containing all four classes.
FLAG_POLYMER = 0x08000040
FLAG_SOLVENT = 0x12000000
FLAG_ORGANIC = 0x22000000
FLAG_INORGANIC = 0x42000000

#: Set on the one atom per residue a cartoon is traced through — ``CA`` for
#: protein, ``C4'`` for nucleic acid.
FLAG_GUIDE = 0x80000000

#: The atoms a cartoon follows, by the residue class they belong to.
GUIDE_ATOMS = {"CA", "C4'"}

#: Element symbol to the PyMOL colour named after it, for atoms whose colour
#: index says "by element" rather than naming a colour.
_ELEMENT_COLOURS = {
    "H": "hydrogen",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "S": "sulfur",
    "P": "phosphorus",
    "F": "fluorine",
    "CL": "chlorine",
    "BR": "bromine",
    "I": "iodine",
    "SE": "selenium",
    "FE": "iron",
    "ZN": "zinc",
    "MG": "magnesium",
    "CA": "calcium",
    "MN": "manganese",
    "CU": "copper",
    "NA": "sodium",
    "K": "potassium",
}

_FALLBACK_COLOUR = (0.8, 0.8, 0.8)


class _Bag:
    """Stand-in for ``pymol.Session_Storage``, which is an attribute bag."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Unpickler(pickle.Unpickler):
    """An unpickler that will not import anything a session should not name."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise PymolSessionError(
                f"session refers to {module}.{name}, which Blender Gala will not "
                "import. A .pse is a pickle, so loading one can run whatever it "
                "names; Gala allows only the handful of globals a genuine "
                "session contains. If this file is trustworthy and you need it, "
                "open it in PyMOL and re-save."
            )
        if (module, name) == ("_codecs", "encode"):
            return codecs.encode
        if module.startswith("numpy"):
            import numpy

            return getattr(numpy, name, None) or _resolve_numpy(module, name)
        return _Bag


def _resolve_numpy(module: str, name: str) -> Any:
    import importlib

    return getattr(importlib.import_module(module), name)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PymolView:
    """PyMOL's camera, as the 25 numbers a session stores.

    Attributes
    ----------
    rotation : numpy.ndarray
        ``(3, 3)``, world to camera.
    position : numpy.ndarray
        Camera position in camera space; ``position[2]`` is the negative
        distance to the origin of rotation.
    origin : numpy.ndarray
        Origin of rotation, in world space (ångström).
    near, far : float
        Clipping planes, as distances from the camera.
    field_of_view : float
        Vertical field of view in degrees.
    orthoscopic : bool
        Whether PyMOL was drawing an orthographic projection.
    """

    rotation: np.ndarray
    position: np.ndarray
    origin: np.ndarray
    near: float
    far: float
    field_of_view: float
    orthoscopic: bool

    @property
    def distance(self) -> float:
        """Distance from the camera to the origin of rotation, in ångström."""
        return abs(float(self.position[2]))

    @classmethod
    def from_list(cls, values: Any) -> PymolView:
        """Build from the 25 floats a session stores."""
        numbers = [float(v) for v in values]
        if len(numbers) < 25:
            raise PymolSessionError(f"view has {len(numbers)} numbers, expected 25")
        matrix = np.array(numbers[:16], dtype=float).reshape(4, 4)
        ortho = numbers[24]
        return cls(
            rotation=matrix[:3, :3],
            position=np.array(numbers[16:19], dtype=float),
            origin=np.array(numbers[19:22], dtype=float),
            near=numbers[22],
            far=numbers[23],
            # Positive means orthoscopic, negative perspective; the magnitude
            # is the field of view either way.
            field_of_view=abs(ortho),
            orthoscopic=ortho > 0,
        )

    def to_list(self) -> list[float]:
        """Return the 25 floats a session stores."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        sign = 1.0 if self.orthoscopic else -1.0
        return (
            [float(v) for v in matrix.reshape(-1)]
            + [float(v) for v in self.position]
            + [float(v) for v in self.origin]
            + [float(self.near), float(self.far), sign * float(self.field_of_view)]
        )

    @classmethod
    def default(cls) -> PymolView:
        """A view looking down ``-Z`` from 100 Å, PyMOL's own starting point."""
        return cls(
            rotation=np.eye(3),
            position=np.array([0.0, 0.0, -100.0]),
            origin=np.zeros(3),
            near=60.0,
            far=140.0,
            field_of_view=20.0,
            orthoscopic=False,
        )


@dataclass
class PymolMolecule:
    """One molecular object out of a session.

    Chemistry is held as parallel arrays rather than a biotite ``AtomArray``
    so that the reader works wherever numpy does; :meth:`to_atom_array` builds
    the biotite object when biotite is present.

    Attributes
    ----------
    coord : numpy.ndarray
        ``(n_states, n_atoms, 3)`` in ångström. Atoms missing from a state
        are ``nan``.
    reps : numpy.ndarray
        Per-atom bitmask over :data:`REPS`.
    color_index : numpy.ndarray
        Per-atom PyMOL colour index; resolve with
        :meth:`PymolSession.color`.
    ss : numpy.ndarray
        Per-atom secondary structure, as PyMOL's ``H``, ``S`` or empty. It is
        what the session was *drawing*, which is not always what a fresh
        assignment would give.
    matrix : numpy.ndarray or None
        A 4x4 object transform, when the object had been moved independently
        of its coordinates.
    """

    name: str
    coord: np.ndarray
    chain_id: np.ndarray
    res_id: np.ndarray
    ins_code: np.ndarray
    res_name: np.ndarray
    atom_name: np.ndarray
    element: np.ndarray
    alt_id: np.ndarray
    segi: np.ndarray
    b_factor: np.ndarray
    occupancy: np.ndarray
    charge: np.ndarray
    vdw: np.ndarray
    hetero: np.ndarray
    label: np.ndarray
    reps: np.ndarray
    color_index: np.ndarray
    bonds: np.ndarray
    ss: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="U1"))
    visible: bool = True
    group: str = ""
    color: int = -1
    matrix: np.ndarray | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.element)

    @property
    def n_states(self) -> int:
        return int(self.coord.shape[0])

    def reps_present(self) -> list[str]:
        """Names of the representations any atom of this object is shown in."""
        if not self.n_atoms:
            return []
        combined = int(np.bitwise_or.reduce(self.reps.astype(np.int64)))
        return [name for bit, name in enumerate(REPS) if combined >> bit & 1]

    def rep_mask(self, rep: str) -> np.ndarray:
        """Boolean mask of the atoms shown in ``rep``."""
        if rep not in REPS:
            raise ValueError(f"unknown representation {rep!r}; expected one of {REPS}")
        bit = REPS.index(rep)
        return (self.reps.astype(np.int64) >> bit & 1).astype(bool)

    def to_atom_array(self, state: int = 0) -> Any:
        """Build a biotite ``AtomArray`` for one state.

        Raises
        ------
        PymolSessionError
            If biotite is not importable, or ``state`` is out of range.
        """
        try:
            from biotite.structure import AtomArray, BondList
        except ImportError as exc:  # pragma: no cover - biotite ships with MN
            raise PymolSessionError(
                "building an AtomArray needs biotite, which comes with "
                "Molecular Nodes. The session itself has been read; use its "
                "arrays directly if you are working without biotite."
            ) from exc

        if not -self.n_states <= state < self.n_states:
            raise PymolSessionError(
                f"state {state} is out of range for {self.name!r}, which has "
                f"{self.n_states}"
            )

        array = AtomArray(self.n_atoms)
        array.coord = self.coord[state].astype(np.float32)
        array.chain_id = self.chain_id
        array.res_id = self.res_id
        array.ins_code = self.ins_code
        array.res_name = self.res_name
        array.atom_name = self.atom_name
        array.element = self.element
        array.hetero = self.hetero
        array.b_factor = self.b_factor.astype(np.float32)
        array.occupancy = self.occupancy.astype(np.float32)
        array.charge = self.charge.astype(int)
        if len(self.bonds):
            array.bonds = BondList(self.n_atoms, self.bonds.astype(np.uint32))
        else:
            array.bonds = BondList(self.n_atoms)
        return array

    def summary(self) -> str:
        """A one-line description, for the report a load prints."""
        reps = ", ".join(self.reps_present()) or "nothing shown"
        states = f", {self.n_states} states" if self.n_states > 1 else ""
        return f"{self.name}: {self.n_atoms} atoms{states} [{reps}]"


@dataclass
class PymolMeasurement:
    """A distance, angle or dihedral object.

    PyMOL stores the *points* it drew between rather than the atoms it
    measured, so that is what comes back. ``kind`` is inferred from how many
    points each measurement has.
    """

    name: str
    kind: str
    points: np.ndarray
    visible: bool = True
    group: str = ""
    color: int = -1

    @property
    def values(self) -> list[float]:
        """Recompute each measurement: ångström, or degrees for angles."""
        return [_measure(group) for group in self.points]


@dataclass
class PymolSelection:
    """A named selection: which atoms of which objects were in it."""

    name: str
    members: dict[str, np.ndarray]
    visible: bool = False

    @property
    def n_atoms(self) -> int:
        return int(sum(len(v) for v in self.members.values()))


@dataclass
class PymolSession:
    """Everything Gala understands out of one ``.pse``.

    Attributes
    ----------
    unsupported : list of tuple
        ``(name, kind)`` for objects that were present and not converted —
        maps, meshes, CGOs. They are listed rather than silently dropped so a
        caller can say what did not come across.
    """

    molecules: list[PymolMolecule] = field(default_factory=list)
    measurements: list[PymolMeasurement] = field(default_factory=list)
    selections: list[PymolSelection] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)
    view: PymolView = field(default_factory=PymolView.default)
    colors: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    source: str | None = field(default=None, compare=False)
    unsupported: list[tuple[str, str]] = field(default_factory=list)

    def color(self, index: int, element: str = "") -> tuple[float, float, float]:
        """Resolve a PyMOL colour index to RGB.

        Session-local colours win over the built-in table, because that is the
        order PyMOL resolves them in. A negative index means "by element",
        which is resolved from ``element`` when one is given.
        """
        if index in self.colors:
            return self.colors[index]
        rgb = palette.rgb_for_index(index)
        if rgb is not None:
            return rgb
        if element:
            named = _ELEMENT_COLOURS.get(element.strip().upper())
            if named is not None:
                by_name = palette.index_for_name(named)
                if by_name is not None:
                    resolved = palette.rgb_for_index(by_name)
                    if resolved is not None:
                        return resolved
        return _FALLBACK_COLOUR

    def atom_colors(self, molecule: PymolMolecule) -> np.ndarray:
        """Resolve one molecule's per-atom colours to an ``(n, 4)`` RGBA array."""
        colors = np.ones((molecule.n_atoms, 4), dtype=float)
        cache: dict[tuple[int, str], tuple[float, float, float]] = {}
        for i, (index, element) in enumerate(
            zip(molecule.color_index, molecule.element, strict=True)
        ):
            key = (int(index), str(element))
            if key not in cache:
                cache[key] = self.color(int(index), str(element))
            colors[i, :3] = cache[key]
        return colors

    @property
    def background(self) -> tuple[float, float, float] | None:
        """The background colour, if the session set one."""
        value = self.settings.get("bg_rgb")
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return self.color(int(value))
        return tuple(float(v) for v in value)[:3]  # type: ignore[return-value]

    def find(self, name: str) -> PymolMolecule | None:
        """Return the molecule called ``name``, if there is one."""
        for molecule in self.molecules:
            if molecule.name == name:
                return molecule
        return None

    def summary(self) -> str:
        """A readable block, for a vignette or the UI to print."""
        lines = [f"{self.source or 'PyMOL session'}"]
        for molecule in self.molecules:
            lines.append(f"  {molecule.summary()}")
        for measurement in self.measurements:
            lines.append(
                f"  {measurement.name}: {len(measurement.points)} {measurement.kind}(s)"
            )
        if self.selections:
            lines.append(
                "  selections: "
                + ", ".join(f"{s.name} ({s.n_atoms})" for s in self.selections)
            )
        for name, kind in self.unsupported:
            lines.append(f"  {name}: {kind}, not converted")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_session(path: str) -> PymolSession:
    """Read a PyMOL session file.

    Parameters
    ----------
    path : str
        A ``.pse`` or ``.psw``, optionally gzipped.

    Returns
    -------
    PymolSession

    Raises
    ------
    PymolSessionError
        If the file is not a session, refers to globals Gala will not import,
        or was written with ``pse_binary_dump`` on, which stores raw C structs
        this reader deliberately does not attempt to decode.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    try:
        # latin1 so that sessions written by PyMOL 1.7's Python 2 pickler,
        # whose strings are bytes, decode rather than raise.
        data = _Unpickler(io.BytesIO(raw), encoding="latin1").load()
    except PymolSessionError:
        raise
    except Exception as exc:
        raise PymolSessionError(
            f"{path}: not a readable PyMOL session ({exc})"
        ) from exc

    if not isinstance(data, dict) or "names" not in data:
        raise PymolSessionError(
            f"{path}: unpickled to {type(data).__name__}, not a session dictionary"
        )

    session = PymolSession(
        version=int(data.get("version", 0) or 0),
        source=os.path.basename(path),
    )
    session.colors = _read_colors(data.get("colors"))
    session.settings = _read_settings(data.get("settings"))
    if data.get("view"):
        session.view = PymolView.from_list(data["view"])

    for record in data["names"]:
        if not record or len(record) <= _REC_DATA:
            continue
        _read_record(session, record)

    return session


def _read_record(session: PymolSession, record: Any) -> None:
    """Convert one name record into whatever it describes."""
    name = str(record[_REC_NAME])
    visible = bool(record[_REC_VISIBLE])
    group = str(record[_REC_GROUP]) if len(record) > _REC_GROUP else ""

    if record[_REC_EXEC] == _EXEC_SELECTION:
        session.selections.append(
            PymolSelection(
                name=name, members=_read_selection(record[_REC_DATA]), visible=visible
            )
        )
        return

    kind = OBJECT_TYPES.get(int(record[_REC_TYPE]), "unknown")
    payload = record[_REC_DATA]

    if kind == "group":
        session.groups[name] = group
        return
    if kind == "molecule":
        session.molecules.append(_read_molecule(name, payload, visible, group))
        return
    if kind == "measurement":
        session.measurements.extend(_read_measurements(name, payload, visible, group))
        return
    session.unsupported.append((name, kind))


def _read_colors(entries: Any) -> dict[int, tuple[float, float, float]]:
    """Read the colours a session defined for itself."""
    colors: dict[int, tuple[float, float, float]] = {}
    for entry in entries or ():
        try:
            index = int(entry[1])
            rgb = tuple(float(c) for c in entry[2])[:3]
        except (IndexError, TypeError, ValueError):
            continue
        if len(rgb) == 3:
            colors[index] = rgb  # type: ignore[assignment]
    return colors


def _read_settings(entries: Any) -> dict[str, Any]:
    """Read the global settings Gala knows what to do with."""
    settings: dict[str, Any] = {}
    for entry in entries or ():
        try:
            index = int(entry[0])
            value = entry[2]
        except (IndexError, TypeError, ValueError):
            continue
        name = SETTINGS.get(index)
        if name is not None:
            settings[name] = value
    return settings


def _read_selection(payload: Any) -> dict[str, np.ndarray]:
    """Read a selection's membership: ``{object name: atom indices}``."""
    members: dict[str, np.ndarray] = {}
    for entry in payload or ():
        try:
            name = str(entry[0])
            indices = np.asarray(entry[1], dtype=int)
        except (IndexError, TypeError, ValueError):
            continue
        members[name] = indices
    return members


def _read_molecule(name: str, payload: Any, visible: bool, group: str) -> PymolMolecule:
    """Read an ObjectMolecule."""
    if not isinstance(payload, (list, tuple)) or len(payload) <= _MOL_ATOMS:
        raise PymolSessionError(f"{name}: molecule record is truncated")

    atoms = payload[_MOL_ATOMS]
    _refuse_binary(name, atoms)
    _refuse_binary(name, payload[_MOL_BONDS])

    n_atoms = int(payload[_MOL_NATOM])
    if not isinstance(atoms, (list, tuple)) or len(atoms) < n_atoms:
        raise PymolSessionError(
            f"{name}: header says {n_atoms} atoms, the record has "
            f"{len(atoms) if hasattr(atoms, '__len__') else 'none'}"
        )
    atoms = list(atoms[:n_atoms])
    for atom in atoms:
        if len(atom) < _AI_MIN_FIELDS:
            raise PymolSessionError(
                f"{name}: an atom has {len(atom)} fields, fewer than the "
                f"{_AI_MIN_FIELDS} every PyMOL since 1.7 has written"
            )

    element = np.array(
        [str(a[_AI_ELEM]).strip().capitalize() for a in atoms], dtype="U2"
    )
    res_id, ins_code = _read_residue_numbers(atoms)

    molecule = PymolMolecule(
        name=name,
        coord=_read_coordsets(name, payload, n_atoms),
        chain_id=np.array([str(a[_AI_CHAIN]).strip() for a in atoms], dtype="U4"),
        res_id=res_id,
        ins_code=ins_code,
        res_name=np.array(
            [str(a[_AI_RESN]).strip().upper() for a in atoms], dtype="U5"
        ),
        atom_name=np.array([str(a[_AI_NAME]).strip() for a in atoms], dtype="U6"),
        element=element,
        alt_id=np.array([str(a[_AI_ALT]).strip() for a in atoms], dtype="U4"),
        segi=np.array([str(a[_AI_SEGI]).strip() for a in atoms], dtype="U4"),
        b_factor=np.array([_number(a[_AI_B]) for a in atoms], dtype=float),
        occupancy=np.array([_number(a[_AI_Q]) for a in atoms], dtype=float),
        charge=np.array([_number(a[_AI_FORMAL_CHARGE]) for a in atoms], dtype=float),
        vdw=np.array([_number(a[_AI_VDW]) for a in atoms], dtype=float),
        hetero=np.array([bool(a[_AI_HETATM]) for a in atoms], dtype=bool),
        label=np.array([str(a[_AI_LABEL]) for a in atoms], dtype=object),
        reps=np.array([_read_visrep(a[_AI_VISREP]) for a in atoms], dtype=np.int64),
        color_index=np.array([int(a[_AI_COLOR]) for a in atoms], dtype=int),
        bonds=_read_bonds(payload[_MOL_BONDS]),
        ss=np.array([str(a[_AI_SS]).strip().upper()[:1] for a in atoms], dtype="U1"),
        visible=visible,
        group=group,
    )

    header = payload[_MOL_HEADER]
    if isinstance(header, (list, tuple)) and len(header) > _HDR_TTT:
        molecule.color = int(header[_HDR_COLOR])
        molecule.settings = _read_object_settings(header[_HDR_SETTINGS])
        if header[_HDR_TTT_FLAG] and header[_HDR_TTT]:
            molecule.matrix = _read_ttt(header[_HDR_TTT])
    return molecule


def _read_residue_numbers(atoms: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Split residue identifiers into number and insertion code.

    ``resv`` carries the number. The string field beside it is empty in modern
    sessions, holds the whole identifier in 1.7-era ones, and carries the
    insertion code when there is one — so the letters are taken from it and
    the digits ignored in favour of ``resv``.
    """
    res_id = np.array([int(_number(a[_AI_RESV])) for a in atoms], dtype=int)
    codes = []
    for atom in atoms:
        text = str(atom[_AI_RESI]).strip()
        letters = "".join(c for c in text if c.isalpha())
        codes.append(letters[-1:] if letters else "")
    return res_id, np.array(codes, dtype="U1")


def _read_visrep(value: Any) -> int:
    """Normalise a visRep to a bitmask.

    Export version 1.7 writes a list with one flag per representation; 1.8 and
    later write the mask directly.
    """
    if isinstance(value, (list, tuple)):
        mask = 0
        for bit, flag in enumerate(value):
            if flag:
                mask |= 1 << bit
        return mask
    return int(value)


def _read_coordsets(name: str, payload: Any, n_atoms: int) -> np.ndarray:
    """Read every state's coordinates into ``(n_states, n_atoms, 3)``."""
    csets = payload[_MOL_CSETS] or []
    states = []
    for cset in csets:
        coords = np.full((n_atoms, 3), np.nan)
        if not cset:
            states.append(coords)
            continue
        _refuse_binary(name, cset[_CS_COORD])
        n_index = int(cset[_CS_NINDEX])
        flat = np.asarray(cset[_CS_COORD], dtype=float)
        if flat.size < n_index * 3:
            raise PymolSessionError(
                f"{name}: a state promised {n_index} positions and has {flat.size // 3}"
            )
        positions = flat[: n_index * 3].reshape(n_index, 3)
        mapping = cset[_CS_IDX_TO_ATM]
        if mapping is None:
            coords[:n_index] = positions
        else:
            index = np.asarray(mapping, dtype=int)[:n_index]
            inside = (index >= 0) & (index < n_atoms)
            coords[index[inside]] = positions[inside]
        states.append(coords)
    if not states:
        states.append(np.full((n_atoms, 3), np.nan))
    return np.stack(states)


def _read_bonds(entries: Any) -> np.ndarray:
    """Read bonds as ``(n, 3)`` of ``atom1, atom2, order``."""
    bonds = []
    for entry in entries or ():
        try:
            bonds.append((int(entry[0]), int(entry[1]), int(entry[2])))
        except (IndexError, TypeError, ValueError):
            continue
    if not bonds:
        return np.zeros((0, 3), dtype=int)
    return np.array(bonds, dtype=int)


def _read_object_settings(entries: Any) -> dict[str, Any]:
    """Read the per-object settings Gala knows about."""
    settings: dict[str, Any] = {}
    for entry in entries or ():
        try:
            name = SETTINGS.get(int(entry[0]))
        except (IndexError, TypeError, ValueError):
            continue
        if name is not None:
            settings[name] = entry[2]
    return settings


def _read_ttt(values: Any) -> np.ndarray:
    """Read PyMOL's TTT matrix as a plain 4x4 homogeneous transform.

    A TTT applies a pre-translation, then the rotation, then a
    post-translation: ``x' = R (x + pre) + post``, where ``pre`` is the last
    row and ``post`` the last column. Folding the pre-translation in gives an
    ordinary matrix that composes the usual way.
    """
    raw = np.array([float(v) for v in values], dtype=float).reshape(4, 4)
    rotation = raw[:3, :3]
    post = raw[:3, 3]
    pre = raw[3, :3]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = rotation @ pre + post
    return matrix


#: Where each kind of measurement lives in a distance set, how many points
#: PyMOL stores per measurement, and how many of those are the measured atoms.
#: Angles and dihedrals carry two trailing entries that are markers rather
#: than positions — ``(1, 1, 0)`` and ``(1, 1, 1)`` respectively, then zeros.
_MEASURE_SLOTS = {
    "distance": (0, 1, 2, 2, None),
    "angle": (3, 4, 5, 3, (1.0, 1.0, 0.0)),
    "dihedral": (5, 6, 6, 4, (1.0, 1.0, 1.0)),
}


def _read_measurements(
    name: str, payload: Any, visible: bool, group: str
) -> list[PymolMeasurement]:
    """Read a measurement object.

    One object holds a separate array for distances, angles and dihedrals, so
    it can in principle contain more than one kind; each becomes its own
    :class:`PymolMeasurement`.
    """
    if not isinstance(payload, (list, tuple)) or len(payload) < 3:
        return []

    found: dict[str, list[np.ndarray]] = {}
    for dset in payload[2] or ():
        if not dset:
            continue
        for kind, (count_at, coord_at, stride, used, _) in _MEASURE_SLOTS.items():
            if len(dset) <= coord_at or not dset[coord_at]:
                continue
            _refuse_binary(name, dset[coord_at])
            n_index = int(dset[count_at] or 0)
            flat = np.asarray(dset[coord_at], dtype=float)
            if n_index < stride or flat.size < n_index * 3:
                continue
            points = flat[: n_index * 3].reshape(n_index, 3)
            found.setdefault(kind, []).extend(
                points[i : i + used] for i in range(0, n_index, stride)
            )

    return [
        PymolMeasurement(
            name=name,
            kind=kind,
            points=np.stack(groups),
            visible=visible,
            group=group,
        )
        for kind, groups in found.items()
        if groups
    ]


def _refuse_binary(name: str, value: Any) -> None:
    """Reject the ``pse_binary_dump`` form rather than misread it."""
    binary = isinstance(value, (bytes, bytearray))
    if not binary and isinstance(value, (list, tuple)) and len(value) >= 2:
        binary = isinstance(value[1], (bytes, bytearray))
    if binary:
        raise PymolSessionError(
            f"{name}: this session was written with pse_binary_dump on, which "
            "stores raw C structs whose layout changes between PyMOL builds. "
            "Gala will not guess at it. In PyMOL:\n"
            "    set pse_binary_dump, 0\n"
            "    save session.pse\n"
            "and read that instead."
        )


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _measure(points: np.ndarray) -> float:
    """Distance, angle or dihedral, depending on how many points there are."""
    points = np.asarray(points, dtype=float)
    if len(points) == 2:
        return float(np.linalg.norm(points[1] - points[0]))
    if len(points) == 3:
        first = points[0] - points[1]
        second = points[2] - points[1]
        cosine = np.dot(first, second) / (
            np.linalg.norm(first) * np.linalg.norm(second)
        )
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if len(points) == 4:
        b0 = points[0] - points[1]
        b1 = points[2] - points[1]
        b2 = points[3] - points[2]
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))
    return float("nan")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_session(session: PymolSession, path: str) -> str:
    """Write a session PyMOL can open.

    Parameters
    ----------
    session : PymolSession
        What to write. Molecules, measurements, selections, groups, the view
        and any session-local colours are all emitted.
    path : str
        Destination. A ``.gz`` suffix compresses it.

    Returns
    -------
    str
        ``path``.
    """
    names: list[Any] = [None]
    for molecule in session.molecules:
        names.append(_write_molecule(molecule))
    for measurement in session.measurements:
        names.append(_write_measurement(measurement))
    for name, parent in session.groups.items():
        names.append(_write_group(name, parent))
    for selection in session.selections:
        names.append(_write_selection(selection))

    data = {
        # 3.0's file version. PyMOL reads anything at or below its own, and
        # every field written here has been present since well before it.
        "version": session.version or 3000000,
        "names": names,
        "colors": [
            [f"gala_color_{index}", index, list(rgb), 1, 0, [0.0, 0.0, 0.0], 0]
            for index, rgb in sorted(session.colors.items())
        ],
        "color_ext": [],
        "view": session.view.to_list(),
        "view_dict": {},
        "settings": _write_settings(session.settings),
        "unique_settings": [],
        "selector_secrets": [],
        "editor": [],
        "main": [0, 0],
        # An empty movie: no frames, no scenes, and the 25-float view matrix
        # it interpolates through left at zero.
        "movie": [0, 0, [0.0] * 25, 0, None, None, None],
        "moviescenes": [[], []],
        "cache": [],
        # PyMOL reads these without checking they are there, so a session
        # missing them segfaults it rather than reporting anything. Each is
        # the empty value PyMOL itself writes; `wizard` is a nested pickle of
        # an empty list, which is what an absent wizard looks like.
        "cachemgr": {"images": []},
        "timeline": [[]],
        "wizard": pickle.dumps([], protocol=2),
        "session": {},
    }

    payload = pickle.dumps(data, protocol=2)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as handle:
            handle.write(payload)
    else:
        with open(path, "wb") as handle:
            handle.write(payload)
    return path


def _write_settings(settings: dict[str, Any]) -> list[Any]:
    """Turn named settings back into PyMOL's ``[index, type, value]`` triples."""
    by_name = {name: index for index, name in SETTINGS.items()}
    entries = []
    for name, value in settings.items():
        index = by_name.get(name)
        if index is None:
            continue
        if isinstance(value, (list, tuple)):
            kind = 5
            value = list(value)
        elif isinstance(value, float):
            kind = 3
        else:
            kind = 2
            value = int(value)
        entries.append([index, kind, value])
    return sorted(entries)


def _object_header(
    name: str,
    kind: int,
    color: int,
    matrix: Any = None,
    settings: dict[str, Any] | None = None,
) -> list[Any]:
    """The 14-element header every object carries."""
    ttt = [0.0] * 16
    flag = 0
    if matrix is not None:
        ttt = _write_ttt(matrix)
        flag = 1
    return [
        kind,
        name,
        int(color),
        0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0,
        flag,
        _write_settings(settings) if settings else None,
        1,
        0,
        ttt,
        0,
        None,
    ]


def _write_ttt(matrix: Any) -> list[float]:
    """Turn an ordinary 4x4 back into PyMOL's TTT form.

    :func:`_read_ttt` folded the pre-translation into the post-translation, so
    writing it back is a matter of leaving the pre-translation at zero.
    """
    raw = np.asarray(matrix, dtype=float).reshape(4, 4)
    ttt = np.zeros((4, 4))
    ttt[:3, :3] = raw[:3, :3]
    ttt[:3, 3] = raw[:3, 3]
    ttt[3, 3] = 1.0
    return [float(v) for v in ttt.reshape(-1)]


def _write_molecule(molecule: PymolMolecule) -> list[Any]:
    """Write one ObjectMolecule name record."""
    n_atoms = molecule.n_atoms
    bonded = np.zeros(n_atoms, dtype=bool)
    if len(molecule.bonds):
        involved = molecule.bonds[:, :2].reshape(-1)
        bonded[involved[(involved >= 0) & (involved < n_atoms)]] = True

    atoms = []
    for i in range(n_atoms):
        atom: list[Any] = [0] * _AI_FIELDS
        atom[_AI_ANISOU : _AI_ANISOU + 6] = [0.0] * 6
        atom[_AI_ELEC_RADIUS] = 0.0
        atom[_AI_CUSTOM] = ""
        atom[_AI_FIELDS - 1] = None
        atom[_AI_RESV] = int(molecule.res_id[i])
        atom[_AI_CHAIN] = str(molecule.chain_id[i])
        atom[_AI_ALT] = str(molecule.alt_id[i])
        atom[_AI_RESI] = str(molecule.ins_code[i])
        atom[_AI_SEGI] = str(molecule.segi[i])
        atom[_AI_RESN] = str(molecule.res_name[i])
        atom[_AI_NAME] = str(molecule.atom_name[i])
        atom[_AI_ELEM] = str(molecule.element[i])
        atom[_AI_TEXT_TYPE] = ""
        atom[_AI_LABEL] = str(molecule.label[i]) if len(molecule.label) else ""
        atom[_AI_SS] = str(molecule.ss[i]) if len(molecule.ss) == n_atoms else ""
        atom[_AI_B] = float(molecule.b_factor[i])
        atom[_AI_Q] = float(molecule.occupancy[i])
        atom[_AI_VDW] = float(molecule.vdw[i])
        atom[_AI_PARTIAL_CHARGE] = 0.0
        atom[_AI_FORMAL_CHARGE] = int(molecule.charge[i])
        atom[_AI_HETATM] = int(bool(molecule.hetero[i]))
        atom[_AI_VISREP] = int(molecule.reps[i])
        atom[_AI_COLOR] = int(molecule.color_index[i])
        atom[_AI_ID] = i + 1
        atom[_AI_FLAGS] = _atom_flags(
            str(molecule.res_name[i]),
            str(molecule.atom_name[i]),
            bool(molecule.hetero[i]),
        )
        atom[_AI_BONDED] = int(bonded[i])
        atom[_AI_PROTONS] = _protons(str(molecule.element[i]))
        atom[_AI_UNIQUE_ID] = i + 1
        atom[_AI_RANK] = i
        atoms.append(atom)

    csets = []
    for state in np.asarray(molecule.coord, dtype=float):
        # An atom with no position in this state is absent from it rather than
        # at the origin, which is what the index maps are for: PyMOL stores
        # only the atoms present and the mapping back to atom numbers.
        present = np.flatnonzero(np.isfinite(state).all(axis=1))
        atm_to_idx = np.full(n_atoms, -1, dtype=int)
        atm_to_idx[present] = np.arange(len(present))
        csets.append(
            [
                len(present),
                n_atoms,
                [float(v) for v in state[present].reshape(-1)],
                [int(i) for i in present],
                [int(i) for i in atm_to_idx],
                "",
                [None],
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        )

    bonds = [
        [int(a), int(b), int(order), i, 0, 0, 0]
        for i, (a, b, order) in enumerate(molecule.bonds)
    ]

    payload = [
        _object_header(
            molecule.name, 1, molecule.color, molecule.matrix, molecule.settings
        ),
        len(csets),
        len(bonds),
        n_atoms,
        csets,
        None,
        bonds,
        atoms,
        0,
        0,
        None,
        0,
        len(bonds),
        n_atoms + 1,
        None,
        None,
    ]
    return [
        molecule.name,
        _EXEC_OBJECT,
        int(bool(molecule.visible)),
        None,
        1,
        payload,
        molecule.group,
    ]


def _write_measurement(measurement: PymolMeasurement) -> list[Any]:
    """Write a distance/angle/dihedral object.

    Every measurement of one kind goes into a single array, which is how
    PyMOL stores them. Angles and dihedrals need their trailing marker
    entries too, or PyMOL reads a shorter array than the count promises.
    """
    slot = _MEASURE_SLOTS.get(measurement.kind)
    if slot is None:
        raise ValueError(
            f"cannot write a {measurement.kind!r} measurement; expected one of "
            f"{tuple(_MEASURE_SLOTS)}"
        )
    count_at, coord_at, stride, used, marker = slot

    rows: list[np.ndarray] = []
    for group in np.asarray(measurement.points, dtype=float).reshape(-1, used, 3):
        rows.extend(group)
        if marker is not None:
            rows.append(np.array(marker, dtype=float))
            rows.extend(np.zeros((stride - used - 1, 3)))

    dset: list[Any] = [0, None, None, 0, None, 0, None, None, None, None]
    dset[count_at] = len(rows)
    dset[coord_at] = [float(v) for v in np.stack(rows).reshape(-1)]
    payload = [
        _object_header(measurement.name, 4, measurement.color),
        1,
        [dset],
        0,
    ]
    return [
        measurement.name,
        _EXEC_OBJECT,
        int(bool(measurement.visible)),
        None,
        4,
        payload,
        measurement.group,
    ]


def _write_group(name: str, parent: str) -> list[Any]:
    """Write a group object."""
    return [
        name,
        _EXEC_OBJECT,
        1,
        None,
        12,
        [_object_header(name, 12, 0), 0, [None]],
        parent,
    ]


def _write_selection(selection: PymolSelection) -> list[Any]:
    """Write a named selection."""
    payload = [
        [name, [int(i) for i in indices], [1] * len(indices)]
        for name, indices in selection.members.items()
    ]
    return [
        selection.name,
        _EXEC_SELECTION,
        int(bool(selection.visible)),
        None,
        -1,
        payload,
        "",
    ]


_PROTONS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}


def _protons(element: str) -> int:
    return _PROTONS.get(element.strip().upper(), 0)


def _atom_flags(res_name: str, atom_name: str, hetero: bool) -> int:
    """Classify an atom the way PyMOL's selection macros expect.

    Derived from the residue name rather than the hetero flag, because that is
    what PyMOL does and because a modified residue written as HETATM is still
    polymer as far as a cartoon is concerned.
    """
    from ..core import chemistry

    name = res_name.strip().upper()
    if name in chemistry.AMINO_ACIDS or name in chemistry.NUCLEOTIDES:
        flags = FLAG_POLYMER
        if atom_name.strip().upper() in GUIDE_ATOMS:
            flags |= FLAG_GUIDE
        return flags
    if name in chemistry.SOLVENT_NAMES:
        return FLAG_SOLVENT
    if name in chemistry.MONOATOMIC_IONS:
        return FLAG_INORGANIC
    return FLAG_ORGANIC
