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
        if os.path.isabs(candidate) and os.access(candidate, os.X_OK):
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


def _run(command: list[str], workdir: str, log: str) -> subprocess.CompletedProcess:
    """Run a command in ``workdir``, keeping its output next to its results."""
    result = subprocess.run(
        command,
        cwd=workdir,
        env=_environment(command[0]),
        capture_output=True,
        text=True,
    )
    with open(os.path.join(workdir, log), "w") as handle:
        handle.write(result.stdout)
        handle.write(result.stderr)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        raise GalaError(
            f"{os.path.basename(command[0])} failed (exit {result.returncode}). "
            f"Full output in {os.path.join(workdir, log)}:\n{tail}"
        )
    return result


def write_pdb(target: Any, path: str) -> str:
    """Write a structure out as PDB, which is what PDB2PQR reads.

    Waters and other solvent are kept: PDB2PQR is told to drop them, and it
    knows better than a selection string here would.

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
    """
    if isinstance(target, str):
        return target

    from biotite.structure.io.pdb import PDBFile

    from ..core.entity import AtomStructure

    structure = AtomStructure.from_any(target)
    pdb = PDBFile()
    pdb.set_structure(structure.array)
    pdb.write(path)
    return path


def _net_charge(pqr_path: str) -> float:
    """Sum the charge column of a PQR file."""
    total = 0.0
    with open(pqr_path) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                if len(fields) >= 2:
                    # A PQR line is a PDB line with charge and radius in place
                    # of occupancy and B-factor; anything else in the file is
                    # not an atom and has no charge to add.
                    with contextlib.suppress(ValueError):
                        total += float(fields[-2])
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

    Returns
    -------
    ApbsResult

    Raises
    ------
    ApbsUnavailable
        If either program is missing.
    GalaError
        If either program fails, or writes nothing to read.
    ValueError
        If ``solver`` is not one of the two.
    """
    if solver not in ("lpbe", "npbe"):
        raise ValueError(f"solver must be 'lpbe' or 'npbe', got {solver!r}")

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
    _run(command, workdir, "pdb2pqr.log")

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

    _run([apbs, os.path.basename(input_path)], workdir, "apbs.log")

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
