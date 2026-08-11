"""Running PDB2PQR and APBS, and getting a potential map back.

The pipeline is the one the `PyMOL APBS plugin
<https://pymolwiki.org/index.php/Apbsplugin>`_ runs, and the one the `APBS
documentation <https://www.poissonboltzmann.org>`_ describes:

1. **PDB2PQR** turns a PDB into a PQR — the same atoms with a charge and a
   radius on each, hydrogens added, ionisable groups assigned a protonation
   state. It also writes an APBS input file whose grid is sized to the
   molecule.
2. **APBS** solves the Poisson-Boltzmann equation on that grid and writes the
   potential as OpenDX.
3. :func:`blender_gala.electrostatics.grid.read_dx` reads it back.

Neither program is bundled. Gala shells out to whatever is on ``PATH``, or to
what ``GALA_APBS`` and ``GALA_PDB2PQR`` point at — the same arrangement as the
PyMOL plugin, and for the same reason: APBS is a substantial piece of
computational chemistry with its own release cycle, and a visualization
add-on has no business vendoring it.

Both are pip-installable, which is the least invasive way to get them::

    pip install apbs-binary pdb2pqr
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.exceptions import GalaError
from .grid import PotentialGrid, read_dx

__all__ = [
    "ApbsResult",
    "ApbsUnavailable",
    "find_executable",
    "run_apbs",
    "write_pdb",
]

#: Environment variables that override executable discovery, per program.
_ENV_VARS = {"apbs": "GALA_APBS", "pdb2pqr": "GALA_PDB2PQR"}

#: Other names the same program goes by.
_ALIASES = {"pdb2pqr": ("pdb2pqr", "pdb2pqr30", "pdb2pqr.exe")}

#: Fields in the shortest PQR atom line: record name, serial, atom name,
#: residue name, residue number, x, y, z, charge, radius. A chain identifier
#: makes eleven.
_PQR_MINIMUM_FIELDS = 10

#: What the PDB format's numbering columns hold: five digits of atom serial,
#: four of residue number. Past them biotite wraps the number back to zero and
#: says so in a ``UserWarning`` — which goes to a console nobody is watching,
#: while the file it wrote describes a different molecule.
_PDB_MAX_ATOM_ID = 99_999
_PDB_MAX_RES_ID = 9_999

#: What the same two columns hold in hybrid-36, the PDB's own extension for
#: exactly this: decimal until the field is full, then base 36 with a letter in
#: the leading column. biotite writes it and reads it back, so it is lossless
#: here; whether the *next* program in a pipeline reads it is a separate
#: question, and PDB2PQR does not.
_HYBRID36_MAX_ATOM_ID = 87_440_031
_HYBRID36_MAX_RES_ID = 2_436_111

#: Seconds. How long APBS or PDB2PQR may run before Gala stops waiting.
#: Generous, because a large solute genuinely takes many minutes; it is there
#: for the run that has hung, which would otherwise hold Blender's main thread
#: forever with no way to cancel it.
DEFAULT_TIMEOUT = 3600.0


class ApbsUnavailable(GalaError):
    """APBS or PDB2PQR is required for this operation but was not found."""

    def __init__(self, program: str) -> None:
        variable = _ENV_VARS.get(program, "")
        super().__init__(
            f"{program} was not found on PATH.\n"
            "Blender Gala shells out to APBS rather than bundling it. Both it "
            "and PDB2PQR install with pip:\n"
            "    pip install apbs-binary pdb2pqr\n"
            f"then either put them on PATH or set {variable} to the executable."
        )
        self.program = program


@dataclass
class ApbsResult:
    """What a run produced, files included so a failure can be looked at.

    Attributes
    ----------
    grid : PotentialGrid
        The potential, in kT/e.
    pqr : str
        The charged, protonated structure PDB2PQR wrote.
    dx : str
        The OpenDX map APBS wrote.
    input_file : str
        The APBS input file that was run.
    workdir : str
        Directory holding all of it.
    net_charge : float
        Sum of the PQR charges. A protein with a wildly non-integral net
        charge usually means PDB2PQR patched something it should not have.
    """

    grid: PotentialGrid
    pqr: str
    dx: str
    input_file: str
    workdir: str
    net_charge: float

    def summary(self) -> str:
        """A readable block, for a vignette or the UI to print."""
        return "\n".join(
            [
                "APBS electrostatics",
                f"  net charge : {self.net_charge:+.2f} e",
                self.grid.summary(),
                f"  files      : {self.workdir}",
            ]
        )


def find_executable(program: str, override: str | None = None) -> str:
    """Locate ``apbs`` or ``pdb2pqr``.

    Parameters
    ----------
    program : {"apbs", "pdb2pqr"}
        Which program to find.
    override : str, optional
        An explicit path, which wins over everything else.

    Returns
    -------
    str
        Path to the executable.

    Raises
    ------
    ApbsUnavailable
        If it is nowhere to be found.
    """
    candidates = [override, os.environ.get(_ENV_VARS.get(program, ""))]
    candidates += list(_ALIASES.get(program, (program,)))

    for candidate in candidates:
        if not candidate:
            continue
        # `os.access(..., X_OK)` is true of a directory, and a directory that
        # happens to be called `apbs` would then be handed to subprocess and
        # come back as a bare PermissionError from three modules down.
        if (
            os.path.isabs(candidate)
            and os.path.isfile(candidate)
            and os.access(candidate, os.X_OK)
        ):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise ApbsUnavailable(program)


def _environment(executable: str) -> dict[str, str]:
    """Environment for running ``executable``, with a bundled ``lib`` added.

    The pip-installed APBS is a binary in ``<package>/bin`` with its shared
    libraries in ``<package>/lib``, and nothing on the system knows to look
    there. Adding a sibling ``lib`` to the loader path costs nothing when
    there is not one, and is the difference between working and a dyld error
    when there is.
    """
    env = os.environ.copy()
    lib = os.path.join(os.path.dirname(os.path.dirname(executable)), "lib")
    if os.path.isdir(lib):
        key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        env[key] = os.pathsep.join([lib, env[key]] if env.get(key) else [lib])
    return env


def _run(
    command: list[str],
    workdir: str,
    log: str,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run a command in ``workdir``, keeping its output next to its results.

    Output is written straight to the log file rather than through a pipe:
    both programs are chatty, and one that decides to write hundreds of
    megabytes would otherwise be held in memory in full — and written to disk
    in full afterwards — before anything could look at it. Only the tail is
    read back, and only when the run failed.
    """
    path = os.path.join(workdir, log)
    with open(path, "w") as handle:
        try:
            result = subprocess.run(
                command,
                cwd=workdir,
                env=_environment(command[0]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GalaError(
                f"{os.path.basename(command[0])} did not finish within "
                f"{timeout:g} s and was stopped; Blender would otherwise wait "
                f"on it forever. What it wrote before then is in {path}. Pass "
                "a larger timeout= to run_apbs if the solute really is that big."
            ) from exc
        except OSError as exc:
            raise GalaError(f"{command[0]} could not be run: {exc}") from exc

    if result.returncode != 0:
        with open(path) as handle:
            tail = "".join(deque(handle, maxlen=15)).rstrip()
        raise GalaError(
            f"{os.path.basename(command[0])} failed (exit {result.returncode}). "
            f"Full output in {path}:\n{tail}"
        )
    return result


def _numbering(array: Any) -> tuple[np.ndarray, np.ndarray]:
    """The atom serials and residue numbers a write would put in the file.

    biotite numbers the atoms ``1..n`` when the array carries no ``atom_id``,
    so the limit for one of those is the atom count.
    """
    serials = getattr(array, "atom_id", None)
    if serials is None:
        serials = np.arange(1, len(array) + 1)
    res_id = getattr(array, "res_id", None)
    if res_id is None:
        res_id = np.ones(len(array), dtype=int)
    return np.asarray(serials), np.asarray(res_id)


def _beyond_pdb_numbering(array: Any) -> str:
    """Describe how ``array`` overflows the PDB's numbering columns, or ``""``.

    Returns
    -------
    str
        A phrase naming each field that does not fit and the value that does
        not fit in it, ready to open a sentence. Empty when the plain format
        holds the whole structure, which it does for all but the largest.
    """
    serials, res_id = _numbering(array)
    over = []
    if serials.size and int(serials.max()) > _PDB_MAX_ATOM_ID:
        over.append(
            f"atom serial {int(serials.max()):,} does not fit the PDB format's "
            f"five-digit field, which stops at {_PDB_MAX_ATOM_ID:,}"
        )
    if res_id.size and int(res_id.max()) > _PDB_MAX_RES_ID:
        over.append(
            f"residue number {int(res_id.max()):,} does not fit the PDB "
            f"format's four-digit field, which stops at {_PDB_MAX_RES_ID:,}"
        )
    return ", and ".join(over)


def _needs_hybrid36(array: Any) -> bool:
    """Whether writing ``array`` losslessly needs hybrid-36 numbering.

    Raises
    ------
    GalaError
        If the numbering fits neither notation — it runs past hybrid-36 as
        well, or it includes a number below one, which hybrid-36 cannot
        encode at all.
    """
    overflow = _beyond_pdb_numbering(array)
    if not overflow:
        return False

    serials, res_id = _numbering(array)
    smallest = min(int(serials.min()), int(res_id.min()))
    if smallest < 1:
        raise GalaError(
            f"this structure cannot be written as PDB: {overflow}, and the "
            f"hybrid-36 notation that would carry it cannot encode {smallest}, "
            "since it has no digits left for a sign. Renumber the atoms and "
            "residues from 1, or write out a subset."
        )
    if (
        int(serials.max()) > _HYBRID36_MAX_ATOM_ID
        or int(res_id.max()) > _HYBRID36_MAX_RES_ID
    ):
        raise GalaError(
            f"this structure cannot be written as PDB: {overflow}, and it runs "
            f"past hybrid-36 numbering too ({_HYBRID36_MAX_ATOM_ID:,} atoms, "
            f"{_HYBRID36_MAX_RES_ID:,} residues). Write it as mmCIF, or write "
            "out a subset."
        )
    return True


def write_pdb(target: Any, path: str) -> str:
    """Write a structure out as PDB, which is what PDB2PQR reads.

    Waters and other solvent are kept: PDB2PQR is told to drop them, and it
    knows better than a selection string here would.

    Numbering past what the format's columns hold — more than 99,999 atoms or
    a residue numbered past 9,999 — is written in hybrid-36, the PDB's own
    extension for it, which biotite reads back exactly. The alternative
    biotite offers is to *wrap* the number and warn, which produces a file
    describing a molecule with twelve atoms per residue and no way for a
    reader to tell. A structure neither notation can hold is refused.

    Note that hybrid-36 is not universally read — :func:`run_apbs` refuses a
    structure that would need it, because PDB2PQR is one of the programs that
    does not.

    Parameters
    ----------
    target : AtomStructure, Molecule, bpy.types.Object, or str
        A structure, or a path to a PDB file, which is returned unchanged.
    path : str
        Where to write.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    GalaError
        If the structure cannot be represented in the PDB format: a chain
        identifier of more than one character, a coordinate needing more than
        four digits before the decimal point, or numbering beyond hybrid-36.
    """
    if isinstance(target, str):
        return target

    from biotite.structure import BadStructureError
    from biotite.structure.io.pdb import PDBFile

    from ..core.entity import AtomStructure

    structure = AtomStructure.from_any(target)
    hybrid36 = _needs_hybrid36(structure.array)

    pdb = PDBFile()
    try:
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            pdb.set_structure(structure.array, hybrid36=hybrid36)
    except BadStructureError as exc:
        # Multi-character chain identifiers (ordinary in an mmCIF
        # `auth_asym_id`) and assembly frames far from the origin both land
        # here, and biotite's exception is not a GalaError — so without this
        # it reaches the user as a console traceback rather than a report.
        raise GalaError(
            f"this structure cannot be written as PDB: {exc}. mmCIF carries "
            "what the PDB format's fixed columns cannot; a structure that "
            "needs it has to be renamed, moved to the origin, or subset "
            "before APBS can be run on it."
        ) from exc

    # A number that will not fit is a warning in biotite and a wrapped value in
    # the file, which is the one outcome this function must never produce. The
    # width check above should have caught it; this is the check that the
    # widths are still the ones biotite is using.
    for warning in raised:
        if "wrapped" in str(warning.message):
            raise GalaError(
                f"this structure cannot be written as PDB: {warning.message}. "
                "A wrapped number would describe a different molecule."
            )
        warnings.warn_explicit(
            warning.message, warning.category, warning.filename, warning.lineno
        )

    pdb.write(path)
    return path


def _require_plain_pdb_numbering(target: Any) -> None:
    """Refuse a structure PDB2PQR cannot be handed as a PDB file.

    PDB2PQR reads the numbering columns as plain integers, so the hybrid-36
    notation :func:`write_pdb` falls back to is not an answer here — and
    neither is letting biotite wrap the numbers, which is what produced this
    check: a 126,000-atom array wrote 63,000 residues as 9,999 distinct
    numbers, PDB2PQR read about twelve atoms per residue, and APBS returned a
    potential map for a chemically impossible molecule without complaint.

    A path is not inspected: it is whatever file the caller already has.
    """
    if isinstance(target, str):
        return

    from ..core.entity import AtomStructure

    overflow = _beyond_pdb_numbering(AtomStructure.from_any(target).array)
    if overflow:
        raise GalaError(
            f"this structure is too large to solve through PDB2PQR: {overflow}. "
            "PDB2PQR reads the PDB format, and a structure past those fields "
            "can only be written by wrapping the numbers — which silently "
            "renumbers the molecule — or in hybrid-36, which PDB2PQR does not "
            "read. Run APBS on a subset: the chain, or the site being "
            "rendered, is what a figure needs and is what the solver can hold."
        )


def _net_charge(pqr_path: str) -> float:
    """Sum the charge column of a PQR file.

    A PQR line is a PDB line with charge and radius in place of occupancy and
    B-factor, written whitespace-separated rather than column-aligned: record
    name, serial, atom name, residue name, an optional chain, residue number,
    *x*, *y*, *z*, charge, radius. Only a line carrying all of that, whose
    last five fields all read as numbers, is an atom — anything shorter is
    something else that happens to begin with ``ATOM`` and has no charge to
    add. A charge that is not finite is not a charge either: a ``nan`` here
    would become the net charge, which is the one number that says whether
    PDB2PQR protonated the structure sensibly.
    """
    total = 0.0
    with open(pqr_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if len(fields) < _PQR_MINIMUM_FIELDS:
                continue
            with contextlib.suppress(ValueError):
                # x, y, z, charge, radius: the coordinates are read only to
                # confirm the line really has the layout it is being read as.
                numbers = [float(value) for value in fields[-5:]]
                if np.isfinite(numbers).all():
                    total += numbers[3]
    return total


def _tune_input(text: str, options: dict[str, Any]) -> str:
    """Apply Gala's settings to the input file PDB2PQR generated.

    PDB2PQR sizes the grid for the molecule, which is the fiddly part and the
    part worth keeping. Everything else — solver, dielectrics, temperature,
    salt — is a keyword swap, so the file is edited rather than rewritten and
    what APBS actually ran stays readable next to its output.
    """
    replacements = {
        "pdie": f"{options['pdie']:.4f}",
        "sdie": f"{options['sdie']:.4f}",
        "temp": f"{options['temperature']:.2f}",
    }
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        key = stripped.split()[0] if stripped else ""
        indent = line[: len(line) - len(line.lstrip())]

        if key in replacements:
            lines.append(f"{indent}{key} {replacements[key]}")
            continue
        if key in ("lpbe", "npbe"):
            lines.append(f"{indent}{options['solver']}")
            continue
        lines.append(line)

        # Mobile salt, as a symmetric monovalent pair — the 0.15 M of a
        # physiological buffer unless asked otherwise. It goes after `mol`,
        # which is where APBS's own examples put it.
        if key == "mol" and stripped != "mol pqr" and options["ionic_strength"] > 0:
            for charge in (1, -1):
                lines.append(
                    f"{indent}ion charge {charge:+d} "
                    f"conc {options['ionic_strength']:.3f} radius 2.0"
                )
    return "\n".join(lines) + "\n"


def _dx_path(input_text: str, workdir: str) -> str:
    """Where the input file tells APBS to write the potential."""
    match = re.search(r"^\s*write\s+pot\s+dx\s+(\S+)", input_text, re.MULTILINE)
    if match is None:  # pragma: no cover - PDB2PQR always writes one
        raise GalaError("the APBS input file does not write a potential map")
    return os.path.join(workdir, f"{match.group(1)}.dx")


def run_apbs(
    target: Any,
    workdir: str | None = None,
    forcefield: str = "AMBER",
    ph: float | None = None,
    ionic_strength: float = 0.15,
    pdie: float = 2.0,
    sdie: float = 78.54,
    temperature: float = 298.15,
    solver: str = "lpbe",
    drop_water: bool = True,
    apbs_path: str | None = None,
    pdb2pqr_path: str | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> ApbsResult:
    """Solve the Poisson-Boltzmann equation for a structure.

    Parameters
    ----------
    target : AtomStructure, Molecule, bpy.types.Object, or str
        The structure, or a path to a PDB file.
    workdir : str, optional
        Where to put the PQR, the input file, the map and the logs. Defaults
        to a new temporary directory, which is *not* cleaned up: a run takes
        long enough that throwing the results away is rarely what you want.
    forcefield : str, optional
        PDB2PQR force field: ``AMBER``, ``PARSE``, ``CHARMM``, ``TYL06``,
        ``PEOEPB`` or ``SWANSON``. AMBER is the usual choice for proteins.
    ph : float, optional
        Assign protonation states for this pH with PROPKA. ``None`` leaves
        PDB2PQR's defaults, which is much faster.
    ionic_strength : float, optional
        Monovalent salt concentration in mol/L. ``0`` runs without salt.
    pdie, sdie : float, optional
        Solute and solvent dielectric constants.
    temperature : float, optional
        Kelvin.
    solver : {"lpbe", "npbe"}, optional
        Linearised or full non-linear Poisson-Boltzmann. Linear is the default
        and is what most figures use; the non-linear solver matters for highly
        charged solutes such as nucleic acids.
    drop_water : bool, optional
        Have PDB2PQR discard waters, which is nearly always what you want:
        crystallographic waters are not part of the continuum solvent model.
    apbs_path, pdb2pqr_path : str, optional
        Explicit executables, overriding ``PATH`` and the environment.
    timeout : float, optional
        Seconds either program may run for before it is stopped. ``None``
        waits indefinitely, which from inside Blender means a hung solver
        holds the whole interface with nothing to press.

    Returns
    -------
    ApbsResult

    Raises
    ------
    ApbsUnavailable
        If either program is missing.
    GalaError
        If either program fails, times out, or writes nothing to read; or if
        the structure has more atoms or residues than the PDB format PDB2PQR
        reads can number, since the only ways to write it are a wrapped —
        silently renumbered — file and one PDB2PQR cannot parse.
    ValueError
        If ``solver`` is not one of the two.
    """
    if solver not in ("lpbe", "npbe"):
        raise ValueError(f"solver must be 'lpbe' or 'npbe', got {solver!r}")

    # Before looking for the programs: a structure the pipeline cannot carry is
    # a fact about the argument, and saying so is more use than reporting that
    # APBS is missing to someone whose structure could not have been solved.
    _require_plain_pdb_numbering(target)

    pdb2pqr = find_executable("pdb2pqr", pdb2pqr_path)
    apbs = find_executable("apbs", apbs_path)

    workdir = workdir or tempfile.mkdtemp(prefix="gala-apbs-")
    os.makedirs(workdir, exist_ok=True)

    pdb_path = write_pdb(target, os.path.join(workdir, "structure.pdb"))
    pqr_path = os.path.join(workdir, "structure.pqr")
    input_path = os.path.join(workdir, "apbs.in")

    command = [pdb2pqr, f"--ff={forcefield}", f"--apbs-input={input_path}"]
    if ph is not None:
        command += ["--titration-state-method=propka", f"--with-ph={ph}"]
    if drop_water:
        command.append("--drop-water")
    command += [os.path.abspath(pdb_path), pqr_path]
    _run(command, workdir, "pdb2pqr.log", timeout=timeout)

    if not os.path.exists(pqr_path):  # pragma: no cover - defensive
        raise GalaError(f"PDB2PQR wrote no PQR file; see {workdir}/pdb2pqr.log")

    with open(input_path) as handle:
        text = handle.read()
    text = _tune_input(
        text,
        {
            "pdie": pdie,
            "sdie": sdie,
            "temperature": temperature,
            "solver": solver,
            "ionic_strength": ionic_strength,
        },
    )
    with open(input_path, "w") as handle:
        handle.write(text)

    _run([apbs, os.path.basename(input_path)], workdir, "apbs.log", timeout=timeout)

    dx_path = _dx_path(text, workdir)
    if not os.path.exists(dx_path):  # pragma: no cover - defensive
        raise GalaError(f"APBS wrote no potential map; see {workdir}/apbs.log")

    grid = read_dx(dx_path)
    return ApbsResult(
        grid=PotentialGrid(
            values=grid.values,
            origin=grid.origin,
            spacing=grid.spacing,
            unit=grid.unit,
            source=f"APBS {solver}, {forcefield} charges",
        ),
        pqr=pqr_path,
        dx=dx_path,
        input_file=input_path,
        workdir=workdir,
        net_charge=float(np.round(_net_charge(pqr_path), 4)),
    )
