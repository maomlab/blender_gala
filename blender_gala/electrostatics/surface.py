"""Painting a potential map onto a molecular surface.

The picture this makes is the one the PyMOL APBS plugin makes: a translucent
molecular surface, red where the potential is negative and blue where it is
positive, saturating at a ramp quoted in kT/e, with whatever is inside the
surface visible through it.

Two details are worth knowing about, because they are the difference between
a figure that means something and one that only looks like it does.

**Where the potential is read.** The surface sits about a probe radius outside
the atoms it is built from, and the potential there is not the potential at
the atom's centre — inside the solute the field is enormous and the sign is
the atom's own charge, which would paint every carbonyl red and every amide
blue regardless of what the molecule as a whole is doing. So the value is
sampled a probe radius out along each atom's outward direction, which is
where the surface actually is.

**Where the colours land.** Molecular Nodes builds the surface in geometry
nodes and colours it from the mesh's ``Color`` attribute, one atom per point.
Set ``color_source`` to ``"Nearest"`` and each patch of surface takes the
colour of the atom nearest it, which is the same correspondence PyMOL's
``ramp_new`` + ``set surface_color`` arrangement produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..color import coloring
from ..color.coloring import ColorResult
from ..core import mn as mn_bridge
from ..core.entity import AtomStructure
from ..scene import materials as gala_materials
from .apbs import run_apbs
from .grid import PotentialGrid

__all__ = [
    "ElectrostaticSurface",
    "color_by_potential",
    "electrostatic_surface",
    "potential_at_atoms",
]

#: Ångström. The radius of a water molecule, and so how far outside the atoms
#: the molecular surface runs.
PROBE_RADIUS = 1.4

#: Ångström. Atoms within this of each other define "outward" for one another.
NEIGHBOURHOOD = 8.0

#: Ångström. Van der Waals radii, which is how far the surface is from the
#: atom's centre before the probe is added. Bondi's values for the elements a
#: protein or a ligand is made of; anything else gets carbon's.
VDW_RADII = {
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "ZN": 1.39,
    "MG": 1.73,
    "NA": 2.27,
    "K": 2.75,
    "CA": 2.31,
    "FE": 2.00,
}
_DEFAULT_RADIUS = 1.70

#: kT/e. Where the colour ramp saturates. Every APBS figure in the literature
#: quotes one, and ±5 is the conventional choice.
RAMP = 5.0


@dataclass
class ElectrostaticSurface:
    """What :func:`electrostatic_surface` built.

    Attributes
    ----------
    grid : PotentialGrid
        The map that was painted on.
    potential : numpy.ndarray
        Per-atom potential at the surface, in kT/e.
    colors : ColorResult
        The colours written to the mesh.
    ramp : float
        Where the ramp saturates, in kT/e.
    material : bpy.types.Material
        The translucent material assigned to the surface style.
    styles : int
        Number of style nodes the material was assigned to.
    """

    grid: PotentialGrid
    potential: np.ndarray
    colors: ColorResult
    ramp: float
    material: Any = None
    styles: int = 0

    def summary(self) -> str:
        """A readable block, for a vignette or the UI to print."""
        finite = self.potential[np.isfinite(self.potential)]
        return "\n".join(
            [
                "Electrostatic surface",
                f"  ramp     : -{self.ramp:g} to +{self.ramp:g} {self.grid.unit} "
                "(red to blue)",
                f"  surface  : {finite.min():+.2f} to {finite.max():+.2f} "
                f"{self.grid.unit}, mean {finite.mean():+.2f}",
                f"  beyond   : {(np.abs(finite) > self.ramp).mean() * 100:.1f}% of "
                "atoms saturate the ramp",
            ]
        )


def _fibonacci_sphere(count: int) -> np.ndarray:
    """``count`` unit vectors spread evenly over the sphere.

    The golden-angle spiral, which is what Shrake and Rupley's construction
    wants: points with no clustering at the poles and no preferred axis.
    """
    index = np.arange(count) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


def _accessible_points(
    coordinates: np.ndarray,
    radii: np.ndarray,
    array: Any,
    probe: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Points on each atom's solvent-accessible sphere, and which are free.

    This is Shrake and Rupley's construction, the one every SASA routine
    uses: put ``count`` points on the sphere of radius *r + probe* around each
    atom and keep the ones no other atom's sphere covers. Those points are
    where a water molecule's centre could sit, which is as close to "where the
    surface is" as a per-atom scheme gets.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Points, shape ``(n_atoms, count, 3)``, and a boolean mask of the same
        first two dimensions saying which are accessible.
    """
    sphere = _fibonacci_sphere(count)
    extended = radii + probe
    points = coordinates[:, None, :] + extended[:, None, None] * sphere[None, :, :]
    free = np.ones(points.shape[:2], dtype=bool)

    try:
        from biotite.structure import CellList
    except ImportError:  # pragma: no cover - biotite ships with Molecular Nodes
        return points, free

    reach = float(extended.max())
    neighbours = CellList(array, cell_size=reach).get_atoms(
        points.reshape(-1, 3), reach
    )
    if neighbours.ndim == 1:  # pragma: no cover - a single neighbour column
        neighbours = neighbours[:, None]

    valid = neighbours >= 0
    indices = np.where(valid, neighbours, 0)
    flat = points.reshape(-1, 1, 3)
    distance = np.linalg.norm(flat - coordinates[indices], axis=2)

    # A point is covered when it lies inside another atom's own sphere. Its
    # parent atom is excluded: the point is exactly on that sphere, and
    # floating point decides which side.
    owner = np.repeat(np.arange(len(coordinates)), count)[:, None]
    covered = valid & (indices != owner) & (distance < extended[indices] - 1e-6)
    return points, ~covered.any(axis=1).reshape(points.shape[:2])


def potential_at_atoms(
    target: Any,
    grid: PotentialGrid,
    probe: float = PROBE_RADIUS,
    outside: str = "clamp",
    points: int = 32,
) -> np.ndarray:
    """Read the potential where each atom's piece of surface would be.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        The structure the map was computed for.
    grid : PotentialGrid
        The map.
    probe : float, optional
        Probe radius in ångström. The sample point is this far outside the
        atom's *van der Waals sphere*, which is where the molecular surface
        runs. ``0`` reads on the van der Waals surface itself; passing a
        negative value large enough to cancel the radius reads at the atom's
        centre, which is inside the solute and is almost never what a surface
        figure wants — the field there is dominated by the atom's own partial
        charge and saturates any sensible ramp.
    outside : {"clamp", "nan"}, optional
        What to do with atoms whose sample point lies outside the grid.
    points : int, optional
        Points per atom on the solvent-accessible sphere. The value is the
        mean over the accessible ones; a buried atom, whose points are all
        covered by its neighbours, gets ``nan``. It has no surface to colour,
        and a number read from inside the solute would be the interior field —
        hundreds of kT/e, and the sign of the atom's own partial charge.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_atoms,)``, in the grid's units, ``nan`` where there is no
        surface.

    Notes
    -----
    Coordinates come from the structure's own array, in ångström, not from the
    Blender mesh: APBS worked in the deposited frame, and the object may since
    have been moved, recentred or scaled.
    """
    structure = AtomStructure.from_any(target)
    coordinates = np.asarray(structure.array.coord, dtype=float)
    elements = [str(e).upper() for e in structure.array.element]
    radii = np.array([VDW_RADII.get(e, _DEFAULT_RADIUS) for e in elements])

    sample_points, accessible = _accessible_points(
        coordinates, radii, structure.array, probe, points
    )
    sampled = grid.sample(sample_points.reshape(-1, 3), outside=outside)
    sampled = sampled.reshape(accessible.shape)

    exposed = accessible.sum(axis=1)
    totals = np.where(accessible, sampled, 0.0).sum(axis=1)
    return np.divide(
        totals,
        exposed,
        out=np.full(len(coordinates), np.nan),
        where=exposed > 0,
    )


def color_by_potential(
    target: Any,
    grid: PotentialGrid | None = None,
    ramp: float = RAMP,
    probe: float = PROBE_RADIUS,
    cmap: str = "bwr",
    write: bool = True,
    **apbs_options: Any,
) -> tuple[ColorResult, PotentialGrid, np.ndarray]:
    """Colour a molecule by electrostatic potential.

    Parameters
    ----------
    target : AtomStructure, Molecule, or bpy.types.Object
        Structure to colour.
    grid : PotentialGrid, optional
        A map to use. ``None`` runs APBS on the structure first, which needs
        ``apbs`` and ``pdb2pqr``.
    ramp : float, optional
        Saturation point of the colour ramp, in kT/e. Symmetric about zero,
        because the sign is the whole point: red at ``-ramp``, white at 0,
        blue at ``+ramp``.
    probe : float, optional
        Sampling distance outside each atom, in ångström.
    cmap : str, optional
        Colormap name. The default is reversed so that negative is red, which
        is the convention every APBS figure follows.
    write : bool, optional
        Write the colours to the mesh.
    **apbs_options
        Forwarded to :func:`blender_gala.electrostatics.apbs.run_apbs` when
        ``grid`` is ``None``.

    Returns
    -------
    tuple
        ``(ColorResult, PotentialGrid, potential)``, where ``potential`` is
        the per-atom value in kT/e.
    """
    structure = AtomStructure.from_any(target)
    if grid is None:
        grid = run_apbs(structure, **apbs_options).grid

    potential = potential_at_atoms(structure, grid, probe=probe)
    colours = coloring.color_by_attribute(
        structure,
        potential,
        cmap=cmap,
        vmin=-abs(ramp),
        vmax=abs(ramp),
        reverse=True,
        write=write,
    )
    return colours, grid, potential


def electrostatic_surface(
    target: Any,
    grid: PotentialGrid | None = None,
    ramp: float = RAMP,
    alpha: float = 0.55,
    probe: float = PROBE_RADIUS,
    quality: int = 3,
    add_style: bool = True,
    cmap: str = "bwr",
    material_options: dict[str, Any] | None = None,
    **apbs_options: Any,
) -> ElectrostaticSurface:
    """Build the translucent, potential-coloured surface of a molecule.

    The APBS plugin's picture, in Blender: solve, sample the map where the
    surface will be, colour by it, and put a surface style in front of
    whatever else the molecule is styled as.

    Parameters
    ----------
    target : Molecule, AtomStructure, or bpy.types.Object
        The molecule.
    grid : PotentialGrid, optional
        A map to use; ``None`` runs APBS.
    ramp : float, optional
        Saturation point of the colour ramp, in kT/e.
    alpha : float, optional
        Surface opacity. ``1`` is solid; the default lets a cartoon inside
        show through, which is the point of a translucent surface.
    probe : float, optional
        Probe radius, in ångström. Used both for the surface Molecular Nodes
        builds and for where the potential is read, because they are the same
        physical quantity.
    quality : int, optional
        Surface mesh resolution, passed to Molecular Nodes.
    add_style : bool, optional
        Add a surface style. Turn it off if the molecule already has one you
        have set up yourself.
    cmap : str, optional
        Colormap name.
    material_options : dict, optional
        Overrides for the surface material, as fields of
        :class:`~blender_gala.scene.materials.GalaMaterialSpec` — ``roughness``
        is the one worth reaching for, since a glossy translucent surface
        carries enough highlight to wash the ramp out.
    **apbs_options
        Forwarded to :func:`blender_gala.electrostatics.apbs.run_apbs`.

    Returns
    -------
    ElectrostaticSurface
    """
    structure = AtomStructure.from_any(target)
    colours, grid, potential = color_by_potential(
        structure, grid=grid, ramp=ramp, probe=probe, cmap=cmap, **apbs_options
    )

    # What is needed from it is `add_style`, so that is what is checked for:
    # `is_molecule` duck-types on `array` and `object`, which an AtomStructure
    # also has, and the UI hands one of those in.
    molecule = structure.molecule
    if molecule is None and hasattr(target, "add_style"):
        molecule = target
    if add_style and molecule is not None:
        mn = mn_bridge.require_mn()
        molecule.add_style(
            mn.StyleSurface(
                quality=quality,
                probe_size=probe,
                # Nearest atom rather than the residue's alpha carbon: the
                # potential varies across a residue, and reading it per
                # residue throws that away.
                color_source="Nearest",
                color_blur=2,
                shade_smooth=True,
            ),
            color=None,
        )

    material = gala_materials.build_material(
        gala_materials.MATERIAL_PRESETS["surface"].with_(
            alpha=alpha, **(material_options or {})
        ),
        name="GALA Electrostatic Surface",
    )
    styles = gala_materials.assign_material(
        molecule if molecule is not None else structure, material, style="surface"
    )

    return ElectrostaticSurface(
        grid=grid,
        potential=potential,
        colors=colours,
        ramp=abs(ramp),
        material=material,
        styles=styles,
    )
