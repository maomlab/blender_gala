"""Detection and representation of non-covalent interactions.

Implements the first half of Objective 2: find hydrogen bonds, polar contacts,
salt bridges, hydrophobic contacts, pi-stacking, cation-pi, halogen bonds and
metal coordination between two selections, then draw them as dashed lines with
sensible, overridable defaults.
"""

from __future__ import annotations

from . import detect, draw, plip
from .detect import (
    DEFAULT_CRITERIA,
    INTERACTION_KINDS,
    Interaction,
    InteractionCriteria,
    atom_contacts,
    cation_pi,
    find_interactions,
    halogen_bonds,
    hydrogen_bonds,
    hydrophobic_contacts,
    metal_coordination,
    pi_stacking,
    polar_contacts,
    salt_bridges,
)
from .draw import (
    INTERACTION_STYLES,
    InteractionStyle,
    clear_interactions,
    draw_interactions,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "INTERACTION_KINDS",
    "INTERACTION_STYLES",
    "Interaction",
    "InteractionCriteria",
    "InteractionStyle",
    "atom_contacts",
    "cation_pi",
    "clear_interactions",
    "detect",
    "draw",
    "draw_interactions",
    "find_interactions",
    "halogen_bonds",
    "hydrogen_bonds",
    "hydrophobic_contacts",
    "metal_coordination",
    "pi_stacking",
    "plip",
    "polar_contacts",
    "salt_bridges",
]
