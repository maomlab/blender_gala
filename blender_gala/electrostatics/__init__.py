"""Electrostatics: APBS potential maps, and surfaces coloured by them.

`APBS <https://www.poissonboltzmann.org>`_ solves the Poisson-Boltzmann
equation for a structure and writes the potential as a grid. This subpackage
runs it, reads the grid, and paints it onto a molecular surface the way the
PyMOL APBS plugin does.

    >>> import blender_gala as gala                    # doctest: +SKIP
    >>> surface = gala.electrostatic_surface(mol)      # doctest: +SKIP
    >>> print(surface.summary())                       # doctest: +SKIP

APBS and PDB2PQR are external programs; Gala shells out to them and does not
bundle either. See :mod:`blender_gala.electrostatics.apbs`.
"""

from __future__ import annotations

from . import apbs, grid, surface
from .apbs import ApbsResult, ApbsUnavailable, find_executable, run_apbs
from .grid import PotentialGrid, read_dx
from .surface import (
    ElectrostaticSurface,
    color_by_potential,
    electrostatic_surface,
    potential_at_atoms,
)

__all__ = [
    "ApbsResult",
    "ApbsUnavailable",
    "ElectrostaticSurface",
    "PotentialGrid",
    "apbs",
    "color_by_potential",
    "electrostatic_surface",
    "find_executable",
    "grid",
    "potential_at_atoms",
    "read_dx",
    "run_apbs",
    "surface",
]
