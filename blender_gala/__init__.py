"""Blender Gala — structural biology visualization tools for Blender.

Gala complements `Molecular Nodes <https://bradyajohnston.github.io/MolecularNodes>`_
with the day-to-day tasks that sit either side of it: turning a freshly
imported molecule into a publication-ready scene, and measuring and annotating
what is in that scene.

Quick start
-----------
::

    import molecularnodes as mn
    import blender_gala as gala

    mol = mn.Molecule.fetch("1ake").add_style("cartoon")

    # Objective 1: a publication-ready scene in one call.
    gala.publication_setup(mol, preset="figure")

    # Objective 2: find, measure and label what matters.
    contacts = gala.find_interactions(mol, "ligand", "protein", kinds="all")
    gala.draw_interactions(contacts, target=mol)
    gala.distance(mol, "resi 15 and name CA", "resi 90 and name CA", draw=True)
    gala.label(mol, "byres (protein within 4 of ligand)")
    gala.color_by_plddt(mol)

    # APBS electrostatics, painted onto a translucent surface.
    gala.electrostatic_surface(mol, ramp=5.0)

    # Open a PyMOL session, or write the scene back out as one.
    gala.load_session("figure.pse")
    gala.save_session("from_blender.pse")

Selections use PyMOL syntax throughout, so ``"byres (protein within 4 of
ligand)"`` means what you would expect.

See Also
--------
SPECIFICATION.md : the design decisions behind these choices.
"""

from __future__ import annotations

import os
import sys
import tomllib


def _drop_stale_submodules() -> None:
    """Forget submodules left behind by a previous version of this add-on.

    Installing a new version into a *running* Blender re-executes this module
    but reloads nothing below it, so every ``blender_gala.*`` submodule stays
    in ``sys.modules`` as the old version left it. The imports below then
    resolve against that old package and fail on the first name the new
    version added — ``cannot import name 'enable_caustics' from
    '...blender_gala.scene'`` — which reads like a broken download and is
    really a stale cache that a restart would have cleared.

    Dropping them here means the imports that follow load from the files that
    were just installed. On a first import there is nothing to drop and this
    does nothing.
    """
    prefix = __name__ + "."
    for name in [name for name in sys.modules if name.startswith(prefix)]:
        del sys.modules[name]


_drop_stale_submodules()

# --- subpackages -----------------------------------------------------------
from . import annotate, color, core, electrostatics, interactions, measure, pymol, scene
from .annotate import (
    clear_labels,
    label,
    label_atoms,
    label_hud,
    label_residues,
)
from .color import (
    color_by_attribute,
    color_by_bfactor,
    color_by_plddt,
    color_by_selection,
    color_from_csv,
    plddt_legend,
    read_colors,
    write_colors,
)

# --- public API ------------------------------------------------------------
from .core import (
    AmbiguousSelectionError,
    AtomStructure,
    EmptySelectionError,
    GalaError,
    MolecularNodesUnavailable,
    Selection,
    SelectionSyntaxError,
    StructureError,
    alias_combine,
    compile_selection,
    create_alias,
    delete_alias,
    describe_selection,
    describe_viewport_selection,
    expand_selection,
    expand_viewport_selection,
    list_aliases,
    select,
    select_alias,
    select_indices,
    set_viewport_selection,
    style_alias,
    viewport_selection,
)
from .electrostatics import (
    PotentialGrid,
    color_by_potential,
    electrostatic_surface,
    potential_at_atoms,
    read_dx,
    run_apbs,
)
from .interactions import (
    INTERACTION_KINDS,
    Interaction,
    InteractionCriteria,
    InteractionStyle,
    atom_contacts,
    cation_pi,
    clear_interactions,
    draw_interactions,
    find_interactions,
    halogen_bonds,
    hydrogen_bonds,
    hydrophobic_contacts,
    metal_coordination,
    pi_stacking,
    polar_contacts,
    salt_bridges,
)
from .measure import (
    Measurement,
    angle,
    clear_measurements,
    dihedral,
    distance,
)
from .measure import measure as measure_atoms
from .pymol import (
    PymolSession,
    PymolSessionError,
    load_session,
    read_session,
    save_session,
    write_session,
)
from .scene import (
    assign_material,
    assign_materials,
    depth_cue,
    depth_of_field,
    enable_caustics,
    enable_passes,
    frame_target,
    hdri_lighting,
    highlight_matte,
    orbit,
    publication_setup,
    set_origin_to_geometry,
    setup_compositor,
    setup_render,
    three_point_lighting,
)
from .scene.render import render


def _read_version() -> str:
    """Read the version from ``blender_manifest.toml``, the single source of truth."""
    manifest = os.path.join(os.path.dirname(__file__), "blender_manifest.toml")
    try:
        with open(manifest, "rb") as handle:
            return str(tomllib.load(handle)["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):  # pragma: no cover
        return "0.0.0"


__version__ = _read_version()

__all__ = [
    "INTERACTION_KINDS",
    "AmbiguousSelectionError",
    "AtomStructure",
    "EmptySelectionError",
    "GalaError",
    "Interaction",
    "InteractionCriteria",
    "InteractionStyle",
    "Measurement",
    "MolecularNodesUnavailable",
    "PotentialGrid",
    "PymolSession",
    "PymolSessionError",
    "Selection",
    "SelectionSyntaxError",
    "StructureError",
    "__version__",
    "alias_combine",
    "angle",
    "annotate",
    "assign_material",
    "assign_materials",
    "atom_contacts",
    "cation_pi",
    "clear_interactions",
    "clear_labels",
    "clear_measurements",
    "color",
    "color_by_attribute",
    "color_by_bfactor",
    "color_by_plddt",
    "color_by_potential",
    "color_by_selection",
    "color_from_csv",
    "compile_selection",
    "core",
    "create_alias",
    "delete_alias",
    "depth_cue",
    "depth_of_field",
    "describe_selection",
    "describe_viewport_selection",
    "dihedral",
    "distance",
    "draw_interactions",
    "electrostatic_surface",
    "electrostatics",
    "enable_caustics",
    "enable_passes",
    "expand_selection",
    "expand_viewport_selection",
    "find_interactions",
    "frame_target",
    "halogen_bonds",
    "hdri_lighting",
    "highlight_matte",
    "hydrogen_bonds",
    "hydrophobic_contacts",
    "interactions",
    "label",
    "label_atoms",
    "label_hud",
    "label_residues",
    "list_aliases",
    "load_session",
    "measure",
    "measure_atoms",
    "metal_coordination",
    "orbit",
    "pi_stacking",
    "plddt_legend",
    "polar_contacts",
    "potential_at_atoms",
    "publication_setup",
    "pymol",
    "read_colors",
    "read_dx",
    "read_session",
    "register",
    "render",
    "run_apbs",
    "salt_bridges",
    "save_session",
    "scene",
    "select",
    "select_alias",
    "select_indices",
    "set_origin_to_geometry",
    "set_viewport_selection",
    "setup_compositor",
    "setup_render",
    "style_alias",
    "three_point_lighting",
    "unregister",
    "viewport_selection",
    "write_colors",
    "write_session",
]


# ---------------------------------------------------------------------------
# Blender add-on registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the add-on with Blender.

    Called automatically when Blender enables the extension. Safe to call from
    a script.
    """
    from . import ops, ui
    from .core import mn_compat

    ops.register()
    ui.register()
    mn_compat.install()


def unregister() -> None:
    """Unregister the add-on."""
    from . import ops, ui
    from .core import mn_compat

    mn_compat.remove()
    ui.unregister()
    ops.unregister()
