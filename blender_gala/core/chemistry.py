"""Chemical reference data and element-level predicates.

Kept free of :mod:`bpy` and of :mod:`biotite` imports at module scope so it can
be used from any interpreter. Everything here operates on plain numpy arrays of
strings.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ACCEPTOR_ELEMENTS",
    "AMINO_ACIDS",
    "AROMATIC_RINGS",
    "CANONICAL_AMINO_ACIDS",
    "DONOR_ELEMENTS",
    "HALOGENS",
    "HYDROGEN_ELEMENTS",
    "METALS",
    "MONOATOMIC_IONS",
    "NEGATIVE_GROUPS",
    "NUCLEIC_BACKBONE_ATOMS",
    "NUCLEOTIDES",
    "POLAR_ELEMENTS",
    "POSITIVE_GROUPS",
    "PROTEIN_BACKBONE_ATOMS",
    "SOLVENT_NAMES",
    "VDW_RADII",
    "normalise_element",
    "vdw_radius",
]

#: Three-letter codes accepted as amino acids, including common modified and
#: protonation-state variants produced by MD and docking tools.
CANONICAL_AMINO_ACIDS = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]
)

AMINO_ACIDS = CANONICAL_AMINO_ACIDS | frozenset(
    [
        "MSE",
        "SEC",
        "PYL",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
        "CYX",
        "CYM",
        "ASH",
        "GLH",
        "LYN",
        "ACE",
        "NME",
        "NMA",
        "ABA",
        "ORN",
        "DAL",
        "DAR",
        "DSG",
        "DAS",
        "DCY",
        "DGN",
        "DGL",
        "DHI",
        "DIL",
        "DLE",
        "DLY",
        "DPN",
        "DPR",
        "DSN",
        "DTH",
        "DTR",
        "DTY",
        "DVA",
        "PCA",
        "SEP",
        "TPO",
        "PTR",
        "MLY",
        "M3L",
        "CSO",
        "KCX",
        "LLP",
        "CME",
        "NEP",
    ]
)

NUCLEOTIDES = frozenset(
    [
        "A",
        "C",
        "G",
        "U",
        "T",
        "DA",
        "DC",
        "DG",
        "DT",
        "DU",
        "RA",
        "RC",
        "RG",
        "RU",
        "ADE",
        "CYT",
        "GUA",
        "THY",
        "URA",
        "A5",
        "A3",
        "C5",
        "C3",
        "G5",
        "G3",
        "U5",
        "U3",
        "DA5",
        "DA3",
        "DC5",
        "DC3",
        "DG5",
        "DG3",
        "DT5",
        "DT3",
    ]
)

SOLVENT_NAMES = frozenset(
    [
        "HOH",
        "WAT",
        "TIP",
        "TIP3",
        "TIP4",
        "TIP5",
        "T3P",
        "SOL",
        "H2O",
        "DOD",
        "D2O",
        "OH2",
        "SPC",
    ]
)

MONOATOMIC_IONS = frozenset(
    [
        "NA",
        "K",
        "LI",
        "RB",
        "CS",
        "MG",
        "CA",
        "ZN",
        "FE",
        "FE2",
        "FE3",
        "MN",
        "CU",
        "CU1",
        "CU2",
        "CO",
        "NI",
        "CD",
        "HG",
        "BA",
        "SR",
        "AL",
        "CL",
        "BR",
        "IOD",
        "F",
        "IUM",
        "NAG_ION",
        "PB",
        "PT",
        "AU",
        "AG",
        "AS",
        "SE4",
        "SO4_ION",
    ]
)

METALS = frozenset(
    [
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "BE",
        "MG",
        "CA",
        "SR",
        "BA",
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "CD",
        "LA",
        "HF",
        "TA",
        "W",
        "RE",
        "OS",
        "IR",
        "PT",
        "AU",
        "HG",
        "AL",
        "GA",
        "IN",
        "TL",
        "SN",
        "PB",
        "BI",
    ]
)

PROTEIN_BACKBONE_ATOMS = frozenset(["N", "CA", "C", "O", "OXT"])

NUCLEIC_BACKBONE_ATOMS = frozenset(
    [
        "P",
        "OP1",
        "OP2",
        "OP3",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
    ]
)

HYDROGEN_ELEMENTS = frozenset({"H", "D"})

#: Elements that can donate a hydrogen bond when carrying a hydrogen.
DONOR_ELEMENTS = frozenset({"N", "O", "S"})

#: Elements that can accept a hydrogen bond.
ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S", "F"})

POLAR_ELEMENTS = frozenset({"N", "O", "S", "P", "F"})

HALOGENS = frozenset({"F", "CL", "BR", "I"})

#: Aromatic ring atom names keyed by residue name, used for pi-stacking and
#: cation-pi detection. Values are tuples of ring atom-name tuples.
AROMATIC_RINGS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("CG", "CD1", "NE1", "CE2", "CD2"),
    ),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HID": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HIE": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HIP": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    # Nucleobases
    "DA": (("N9", "C8", "N7", "C5", "C4"), ("N1", "C2", "N3", "C4", "C5", "C6")),
    "DG": (("N9", "C8", "N7", "C5", "C4"), ("N1", "C2", "N3", "C4", "C5", "C6")),
    "DC": (("N1", "C2", "N3", "C4", "C5", "C6"),),
    "DT": (("N1", "C2", "N3", "C4", "C5", "C6"),),
    "A": (("N9", "C8", "N7", "C5", "C4"), ("N1", "C2", "N3", "C4", "C5", "C6")),
    "G": (("N9", "C8", "N7", "C5", "C4"), ("N1", "C2", "N3", "C4", "C5", "C6")),
    "C": (("N1", "C2", "N3", "C4", "C5", "C6"),),
    "U": (("N1", "C2", "N3", "C4", "C5", "C6"),),
}

#: Positively charged groups: residue name -> atom names forming the group.
POSITIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "LYS": ("NZ",),
    "ARG": ("NH1", "NH2", "NE", "CZ"),
    "HIS": ("ND1", "NE2"),
    "HIP": ("ND1", "NE2"),
    "HSP": ("ND1", "NE2"),
}

#: Negatively charged groups: residue name -> atom names forming the group.
NEGATIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2", "CG"),
    "GLU": ("OE1", "OE2", "CD"),
}

#: Van der Waals radii in ångström (Bondi 1964, with later additions).
VDW_RADII: dict[str, float] = {
    "H": 1.10,
    "D": 1.10,
    "HE": 1.40,
    "LI": 1.81,
    "BE": 1.53,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "NE": 1.54,
    "NA": 2.27,
    "MG": 1.73,
    "AL": 1.84,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "AR": 1.88,
    "K": 2.75,
    "CA": 2.31,
    "MN": 2.05,
    "FE": 2.04,
    "CO": 2.00,
    "NI": 1.97,
    "CU": 1.96,
    "ZN": 2.01,
    "SE": 1.90,
    "BR": 1.85,
    "I": 1.98,
}

_DEFAULT_VDW = 1.70


def normalise_element(values: np.ndarray) -> np.ndarray:
    """Upper-case and strip an array of element symbols.

    Parameters
    ----------
    values : numpy.ndarray
        Array of element symbol strings.

    Returns
    -------
    numpy.ndarray
        Upper-cased, whitespace-stripped copy.
    """
    return np.char.upper(np.char.strip(values.astype(str)))


def vdw_radius(element: str) -> float:
    """Return the van der Waals radius of ``element`` in ångström.

    Unknown elements fall back to the carbon radius, which keeps contact
    heuristics conservative rather than silently failing.
    """
    return VDW_RADII.get(element.strip().upper(), _DEFAULT_VDW)
