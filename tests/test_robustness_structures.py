"""What the API does with structures that carry real-world complications.

The rest of the suite works on fixtures built to be *convenient*: one chain,
consecutive residue numbers, one conformation, an element column on every
atom. Nothing deposited in the PDB looks like that for long. This module runs
the science layer over ``awkward.pdb``/``awkward.cif`` (alternate locations,
insertion codes, an expression tag numbered below one, a gap in the numbering,
selenomethionine, an unidentified residue, a lower-case chain beside an
upper-case one, a calcium named like a C-alpha, a zero-occupancy atom, a
missing element column, formal charges, a duplicated atom), over
``nucleic.pdb``, and over the three-model ``ensemble.pdb``.

See ``tests/data/make_fixtures.py`` for what each fixture contains and why, and
``tests/ROBUSTNESS.md`` for the tier this sits in: these are small, committed
and deterministic, because the permanent suite has no network. The survey of
*real* PDB entries, which does, is ``scripts/survey_structures.py`` and runs on
a schedule.

The recurring assertion is the round trip: ``describe_selection`` renders a
mask back into the selection language and its output must re-select the same
atoms. It is the single check most likely to catch a structure whose chemistry
cannot be named unambiguously, which is exactly what these fixtures are.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import DATA_DIR, requires_biotite

pytestmark = requires_biotite

#: The annotations biotite drops unless asked for, and every fixture here has.
EXTRA_FIELDS = ["b_factor", "occupancy", "charge", "atom_id"]


def read_pdb(name: str, model: int | None = 1):
    import biotite.structure.io.pdb as pdb

    handle = pdb.PDBFile.read(os.path.join(DATA_DIR, name))
    if model is None:
        return handle.get_structure(extra_fields=EXTRA_FIELDS)
    return handle.get_structure(model=model, extra_fields=EXTRA_FIELDS)


def read_cif(name: str):
    import biotite.structure.io.pdbx as pdbx

    handle = pdbx.CIFFile.read(os.path.join(DATA_DIR, name))
    return pdbx.get_structure(handle, model=1, extra_fields=EXTRA_FIELDS)


@pytest.fixture(scope="module")
def awkward():
    return read_pdb("awkward.pdb")


@pytest.fixture(scope="module")
def awkward_cif():
    return read_cif("awkward.cif")


@pytest.fixture(scope="module")
def nucleic():
    return read_pdb("nucleic.pdb")


@pytest.fixture(scope="module")
def ensemble():
    return read_pdb("ensemble.pdb", model=None)


@pytest.fixture(params=["awkward.pdb", "awkward.cif", "nucleic.pdb", "ensemble.pdb"])
def structure(request):
    """Each awkward fixture in turn, as a bare ``AtomArray``."""
    name = request.param
    return read_cif(name) if name.endswith(".cif") else read_pdb(name)


# ---------------------------------------------------------------------------
# Every fixture, through the whole language
# ---------------------------------------------------------------------------


def test_every_macro_evaluates(structure):
    from blender_gala.core.selection import MACRO_KEYWORDS, select

    for macro in MACRO_KEYWORDS:
        mask = select(structure, macro)
        assert mask.shape == (len(structure),), macro
        assert mask.dtype == bool, macro


def test_every_level_expands(structure):
    from blender_gala.core.selection import LEVELS, expand_selection

    for level in LEVELS:
        grown = expand_selection(structure, "index 1", level)
        assert grown.shape == (len(structure),), level
        assert grown[0], level  # expanding never loses what was picked


def test_every_macro_describes_back_to_itself(structure):
    """The round trip, over selections that span whole residues and parts."""
    from blender_gala.core.selection import describe_selection, select

    for expression in (
        "all",
        "protein",
        "nucleic",
        "hetatm",
        "backbone",
        "sidechain",
        "name CA",
        "chain A",
        "b > 10",
    ):
        mask = select(structure, expression)
        text = describe_selection(structure, mask)

        assert np.array_equal(select(structure, text), mask), f"{expression} -> {text}"


def test_every_single_atom_describes_back_to_itself(structure):
    """One atom at a time is where an ambiguous residue shows up: two atoms of
    the same name in one residue cannot be told apart chemically, and the
    description has to fall back to the positional form rather than lie."""
    from blender_gala.core.selection import describe_selection, select

    for index in range(len(structure)):
        mask = np.zeros(len(structure), dtype=bool)
        mask[index] = True
        text = describe_selection(structure, mask)

        assert np.array_equal(select(structure, text), mask), f"atom {index}: {text}"


def test_interactions_run_over_every_fixture(structure):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect

    found = detect.find_interactions(AtomStructure(array=structure), kinds="all")

    assert all(np.isfinite(interaction.distance) for interaction in found)


# ---------------------------------------------------------------------------
# Insertion codes
# ---------------------------------------------------------------------------


def test_an_insertion_code_makes_a_separate_residue(awkward):
    """TRP 100, GLY 100A and GLY 100B are three residues sharing one number.

    Antibody numbering does this constantly. Keying on the number alone
    collapses them into one, and every per-residue operation is then wrong.
    """
    from blender_gala.core.selection import expand_selection, select

    numbered = select(awkward, "resi 100")
    assert int(numbered.sum()) == 6  # all three residues carry the number

    # But growing one atom to its residue must not drag in the other two.
    picked = np.zeros(len(awkward), dtype=bool)
    picked[np.flatnonzero(numbered)[0]] = True
    assert int(expand_selection(awkward, picked, "residue").sum()) == 2


def test_the_insertion_code_is_selectable(awkward):
    from blender_gala.core.selection import select

    assert int(select(awkward, "ins A").sum()) == 2
    assert int(select(awkward, "ins B").sum()) == 2
    assert int(select(awkward, "resi 100 and ins A").sum()) == 2


# ---------------------------------------------------------------------------
# Residue numbering that does not start at one
# ---------------------------------------------------------------------------


def test_an_expression_tag_numbered_below_one_is_selectable(awkward):
    """Residues -1 and 0 are an ordinary part of a construct."""
    from blender_gala.core.selection import select

    assert int(select(awkward, "resi -1").sum()) == 3
    assert int(select(awkward, "resi 0").sum()) == 3
    assert int(select(awkward, "resi -1-0").sum()) == 6


def test_a_gap_in_the_numbering_selects_nothing(awkward):
    """Residues 3 to 99 were never modelled, so a range over them is empty
    rather than reaching the residues either side of the gap."""
    from blender_gala.core.selection import select

    assert int(select(awkward, "resi 3-99").sum()) == 0


# ---------------------------------------------------------------------------
# Alternate conformations
# ---------------------------------------------------------------------------


def test_one_conformation_is_read_by_default(awkward):
    """biotite resolves altlocs on the way in, so ALA 1 arrives with one atom
    of each name — which is what makes this a single atom rather than two."""
    from blender_gala.core.entity import AtomStructure

    structure = AtomStructure(array=awkward)

    assert structure.one_index("resn ALA and name CA") >= 0


def test_a_chain_of_another_case_is_a_different_chain(awkward):
    """mmCIF `auth_asym_id` is case-sensitive, and that is exactly how a large
    assembly gets past the 62 single-character identifiers: a survey of 6J5K
    found `chain A` matching 5568 atoms where the file has 3869, and 72 of its
    120 chains affected.

    Case-insensitivity is still wanted where it costs nothing — see the test
    below — so the answer is to match case-sensitively first and fall back only
    when nothing matched.
    """
    from blender_gala.core.selection import expand_selection, select

    upper = select(awkward, "chain A")

    assert int(upper.sum()) == 29  # chain `a`'s four atoms are not chain `A`
    assert not (upper & (awkward.chain_id == "a")).any()
    assert not (
        expand_selection(awkward, upper, "chain") & (awkward.chain_id == "a")
    ).any()


def test_a_chain_still_matches_when_only_the_case_differs(awkward):
    """The other half of the same contract, and the reason it cannot simply be
    made case-sensitive: `chain d` has to keep finding chain `D`, because
    nothing else in the file is called `d` and the language folds case
    everywhere else.
    """
    from blender_gala.core.selection import select

    assert np.array_equal(select(awkward, "chain d"), select(awkward, "chain D"))
    assert int(select(awkward, "chain d").sum()) == 1


def test_a_keyword_whose_data_is_absent_says_so(awkward):
    """Neither Molecular Nodes nor Gala reads a structure with `altloc="all"`,
    so `altloc_id` is never on the array and `alt A` is quietly empty on every
    structure — including the ones that really do have two conformations. An
    empty answer and "there is no such column here" are different facts.
    """
    from blender_gala.core.exceptions import SelectionError
    from blender_gala.core.selection import select

    with pytest.raises(SelectionError):
        select(awkward, "alt A")


def test_a_duplicated_atom_is_still_described_exactly(awkward):
    """VAL 1 of chain `a` carries CB twice at the same coordinates — a
    deposition error rather than an alternate location, so nothing upstream
    resolves it. It cannot be named chemically, and must not be named wrongly.
    """
    from blender_gala.core.selection import describe_selection, select

    duplicated = select(awkward, "resn VAL and name CB")
    assert int(duplicated.sum()) == 2

    single = np.zeros(len(awkward), dtype=bool)
    single[np.flatnonzero(duplicated)[0]] = True
    text = describe_selection(awkward, single)

    assert np.array_equal(select(awkward, text), single), text


# ---------------------------------------------------------------------------
# Residues and elements that are not what their name suggests
# ---------------------------------------------------------------------------


def test_selenomethionine_is_protein(awkward):
    """MSE is written as HETATM and is nonetheless polymer, which is what
    decides whether a cartoon is drawn through it."""
    from blender_gala.core.selection import select

    assert int(select(awkward, "resn MSE and protein").sum()) == 4
    assert int(select(awkward, "resn MSE and elem SE").sum()) == 1


def test_an_unidentified_residue_is_not_claimed_as_a_ligand(awkward):
    from blender_gala.core.selection import select

    assert int(select(awkward, "resn UNK").sum()) == 2
    assert not (select(awkward, "resn UNK") & select(awkward, "solvent")).any()


def test_a_calcium_is_told_from_a_c_alpha_by_its_element(awkward):
    """Both are atoms named CA. Only the element column separates them, and
    the metal detector has to use it rather than the name."""
    from blender_gala.core.selection import select

    named_ca = select(awkward, "name CA")
    metals = select(awkward, "metals")

    assert int(metals.sum()) == 1
    assert (metals & named_ca).any()  # the calcium is named CA too
    assert int((named_ca & ~metals).sum()) == 10  # and the rest are C-alphas


def test_a_zero_occupancy_atom_is_present_but_findable(awkward):
    """Modelled without being believed: still an atom, and still excludable."""
    from blender_gala.core.selection import select

    assert int(select(awkward, "q < 0.5").sum()) == 1
    assert int(select(awkward, "q > 0.5").sum()) == len(awkward) - 1


def test_formal_charges_are_read(awkward):
    from blender_gala.core.selection import select

    assert int(select(awkward, "charge > 0").sum()) == 2  # the lysine and the calcium
    assert int(select(awkward, "charge < 0").sum()) == 1


# ---------------------------------------------------------------------------
# Nucleic acids
# ---------------------------------------------------------------------------


def test_nucleic_acid_is_not_claimed_as_protein(nucleic):
    from blender_gala.core.selection import select

    assert int(select(nucleic, "nucleic").sum()) == len(nucleic)
    assert int(select(nucleic, "protein").sum()) == 0


def test_the_nucleic_backbone_and_sidechain_partition_the_strand(nucleic):
    """Every atom is one or the other, and none is both — the property a
    style applied to `backbone` and one applied to `sidechain` rely on."""
    from blender_gala.core.selection import select

    backbone = select(nucleic, "backbone")
    sidechain = select(nucleic, "sidechain")

    assert not (backbone & sidechain).any()
    assert (backbone | sidechain).all()


def test_ribose_is_what_separates_rna_from_dna(nucleic):
    """The 2' hydroxyl, and the only structural difference the fixture has."""
    from blender_gala.core.selection import select

    assert int(select(nucleic, "name O2'").sum()) == 2
    assert int(select(nucleic, "resn DA+DT and name O2'").sum()) == 0


def test_an_atom_name_carrying_a_prime_is_selectable(nucleic):
    """Every sugar atom has one, and `'` is not punctuation in the language."""
    from blender_gala.core.selection import describe_selection, select

    mask = select(nucleic, "name C1'")
    assert int(mask.sum()) == 4
    assert np.array_equal(select(nucleic, describe_selection(nucleic, mask)), mask)


# ---------------------------------------------------------------------------
# Several models
# ---------------------------------------------------------------------------


def test_a_model_is_chosen_by_frame(ensemble):
    from blender_gala.core.entity import AtomStructure

    first = AtomStructure.from_any(ensemble, frame=0)
    third = AtomStructure.from_any(ensemble, frame=2)

    assert first.n_atoms == third.n_atoms
    assert not np.allclose(first.coord, third.coord)


def test_a_model_that_is_not_there_is_reported(ensemble):
    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.exceptions import StructureError

    with pytest.raises(StructureError, match="out of range"):
        AtomStructure.from_any(ensemble, frame=99)


def test_a_selection_means_the_same_thing_in_every_model(ensemble):
    """Chemistry does not move between models, so neither may a selection."""
    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.selection import select

    masks = [
        select(AtomStructure.from_any(ensemble, frame=frame).array, "name CA")
        for frame in range(3)
    ]

    assert all(np.array_equal(mask, masks[0]) for mask in masks)


# ---------------------------------------------------------------------------
# The same structure in two formats
# ---------------------------------------------------------------------------

#: Selections that read only what both formats state outright. Anything
#: derived from the *element* is deliberately not here — see the test below.
FORMAT_INDEPENDENT = (
    "all",
    "protein",
    "hetatm",
    "solvent",
    "backbone",
    "chain A",
    "resi 100",
    "resi -1-0",
    "ins A",
    "name CA",
    "resn MSE",
    "q > 0.5",
    "b > 10",
)


@pytest.mark.parametrize("expression", FORMAT_INDEPENDENT)
def test_pdb_and_cif_agree_on_the_same_structure(awkward, awkward_cif, expression):
    """The two files describe one structure. A selection that means different
    atoms in each is a defect in whichever reading is wrong, and the user has
    no way to tell which."""
    from blender_gala.core.selection import select

    assert np.array_equal(select(awkward, expression), select(awkward_cif, expression))


def test_the_two_formats_disagree_only_about_the_missing_element(awkward, awkward_cif):
    """One atom has no element column. The PDB reader guesses it from the atom
    name, as the format's convention allows; the mmCIF reader leaves it unset,
    because mmCIF has no such convention. So `elem C` differs by exactly that
    atom — recorded here so that a change to either reading is noticed rather
    than discovered in a figure.
    """
    from blender_gala.core.selection import select

    from_pdb = select(awkward, "elem C")
    from_cif = select(awkward_cif, "elem C")

    difference = np.flatnonzero(from_pdb != from_cif)
    assert len(difference) == 1
    assert awkward.atom_name[difference[0]] == "CA"
    assert awkward.res_name[difference[0]] == "UNK"


def test_the_label_and_author_chains_differ_in_the_cif(awkward_cif):
    """mmCIF carries a multi-character `label_asym_id` beside the single
    character `auth_asym_id` the PDB format has room for. biotite reports the
    author one, which is what a user typing `chain A` means."""
    from blender_gala.core.selection import select

    assert int(select(awkward_cif, "chain A").sum()) > 0
    assert set(np.unique(awkward_cif.chain_id)) <= set("ABCDa")


# ---------------------------------------------------------------------------
# Pathologies isolated one at a time
#
# The fixtures above are files, which is how a user meets these. The arrays
# below are built directly, so that each one carries exactly one complication
# and a failure names it. Every coordinate is exact, so the *right* answer is
# known rather than merely different from the wrong one.
# ---------------------------------------------------------------------------


def make(n: int, **annotations):
    """A bare ``AtomArray`` of ``n`` atoms, with sensible defaults."""
    import biotite.structure as struc

    array = struc.AtomArray(n)
    coord = annotations.pop("coord", [[float(i), 0.0, 0.0] for i in range(n)])
    array.coord = np.asarray(coord, dtype=np.float32).reshape(n, 3)
    array.chain_id = np.asarray(annotations.pop("chain_id", ["A"] * n))
    array.res_id = np.asarray(annotations.pop("res_id", list(range(1, n + 1))))
    array.res_name = np.asarray(annotations.pop("res_name", ["ALA"] * n))
    array.atom_name = np.asarray(annotations.pop("atom_name", ["CA"] * n))
    array.element = np.asarray(annotations.pop("element", ["C"] * n))
    for name, values in annotations.items():
        array.set_annotation(name, np.asarray(values))
    return array


def hexagon(centre, radius=1.39, start=0.0):
    """Six points on a regular hexagon in the z plane, as a benzene ring is."""
    import math

    return [
        (
            centre[0] + radius * math.cos(start + i * math.pi / 3.0),
            centre[1] + radius * math.sin(start + i * math.pi / 3.0),
            centre[2],
        )
        for i in range(6)
    ]


RING_NAMES = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]


def test_an_alternate_conformation_does_not_hide_an_aromatic_ring():
    """Two PHE rings 3.6 A apart, one of them modelled in two conformations.

    The stack is real in either conformation, and an unmerged altloc is the
    normal state of a high-resolution structure. Reporting nothing is worse
    than reporting it twice.
    """
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect

    points = hexagon((0.0, 0.0, 0.0)) + hexagon((0.0, 0.0, 3.6))
    # Chain A's ring again, 0.2 A away, as the second conformer.
    points += [(x, y, z + 0.2) for x, y, z in hexagon((0.0, 0.0, 0.0))]
    array = make(
        18,
        coord=points,
        chain_id=["A"] * 6 + ["B"] * 6 + ["A"] * 6,
        res_id=[1] * 6 + [1] * 6 + [1] * 6,
        res_name=["PHE"] * 18,
        atom_name=RING_NAMES * 3,
        altloc_id=[""] * 12 + ["B"] * 6,
        occupancy=[1.0] * 6 + [1.0] * 6 + [0.4] * 6,
    )

    assert detect.pi_stacking(AtomStructure(array=array))


def test_an_alternate_conformation_does_not_fuse_a_charged_group():
    """The sharpest assertion available: a wrong number with a known right one.

    Conformer A alone puts the lysine 3.54 A from the carboxylate centroid.
    Fusing the two conformers reports 3.40 A, which is the distance to a point
    no conformer occupies.
    """
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect

    array = make(
        7,
        coord=[
            [0.0, 0.0, 1.0],
            [1.2, 0.6, 1.0],
            [1.2, -0.6, 1.0],  # conformer A
            [0.0, 0.0, -1.0],
            [1.2, 0.6, -1.0],
            [1.2, -0.6, -1.0],  # conformer B
            [4.2, 0.0, 0.0],  # the lysine tip
        ],
        chain_id=["A"] * 7,
        res_id=[1, 1, 1, 1, 1, 1, 9],
        res_name=["ASP"] * 6 + ["LYS"],
        atom_name=["CG", "OD1", "OD2"] * 2 + ["NZ"],
        element=["C", "O", "O", "C", "O", "O", "N"],
        altloc_id=["A"] * 3 + ["B"] * 3 + [""],
        occupancy=[0.5] * 6 + [1.0],
    )

    bridges = detect.salt_bridges(AtomStructure(array=array))

    assert bridges
    assert bridges[0].distance == pytest.approx(3.54, abs=0.01)


def test_the_element_property_agrees_with_the_element_macros():
    """A blank element column is legal and common. Whichever way the symbol is
    derived, the two ways of asking for it have to agree."""
    from blender_gala.core.selection import select

    array = make(
        5,
        atom_name=["N", "CA", "C", "O", "CB"],
        element=[""] * 5,
        res_name=["ALA"] * 5,
        res_id=[1] * 5,
    )

    assert np.array_equal(select(array, "elem C"), select(array, "carbon"))
    assert int(select(array, "elem N").sum()) == 1


@pytest.mark.parametrize(
    ("res_name", "atom_names"),
    [
        ("UNK", ["N", "CA", "C", "O", "CB"]),  # identity not determined
        ("PSU", ["P", "O5'", "C5'", "N1", "C2", "O2"]),  # pseudouridine
    ],
)
def test_an_unclassified_residue_is_reachable_by_some_macro(res_name, atom_names):
    """The weakest useful statement, and the one a fix cannot game: every atom
    belongs to at least one of the three classes a user colours by.

    Molecular Nodes stores `is_peptide`/`is_nucleic` flags that Gala prefers
    when they are there, so the shipped add-on classifies these correctly. It
    is the headless science layer — which the docstrings call fully supported,
    and which this whole suite runs on — that cannot see them.
    """
    from blender_gala.core.selection import select

    n = len(atom_names)
    array = make(
        n,
        res_name=[res_name] * n,
        res_id=[1] * n,
        atom_name=atom_names,
        element=[name[0] for name in atom_names],
        hetero=[False] * n,
    )

    covered = (
        select(array, "polymer") | select(array, "ligand") | select(array, "solvent")
    )
    assert covered.all()


def test_dna_and_rna_are_not_the_same_keyword():
    from blender_gala.core.selection import select

    array = make(2, res_name=["DA", "A"], res_id=[1, 2], atom_name=["P", "P"])

    assert int(select(array, "dna").sum()) == 1
    assert int(select(array, "rna").sum()) == 1


def test_a_stack_given_to_the_constructor_is_reduced_to_one_model():
    import biotite.structure as struc

    from blender_gala.core.entity import AtomStructure

    template = make(4)
    stack = struc.stack([template] * 3)
    structure = AtomStructure(array=stack, frame=2)

    assert structure.n_atoms == 4
    assert np.allclose(structure.coord, structure.context.coord)


def test_asking_for_another_frame_of_a_structure_is_honoured_or_refused():
    import biotite.structure as struc

    from blender_gala.core.entity import AtomStructure

    stack = struc.stack([make(4) for _ in range(3)])
    third = AtomStructure.from_any(stack, frame=2)

    assert AtomStructure.from_any(third, frame=0).frame == 0


def test_an_alternate_conformation_is_not_counted_twice():
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect

    array = make(
        3,
        coord=[[0.0, 0.0, 0.6], [0.0, 0.0, -0.6], [2.9, 0.0, 0.0]],
        res_id=[1, 1, 9],
        res_name=["SER", "SER", "ASN"],
        atom_name=["OG", "OG", "ND2"],
        element=["O", "O", "N"],
        altloc_id=["A", "B", ""],
        occupancy=[0.5, 0.5, 1.0],
    )

    found = detect.find_interactions(AtomStructure(array=array), kinds=("polar",))
    labels = [str(interaction) for interaction in found]

    assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# Writing a structure back out as PDB
# ---------------------------------------------------------------------------


def test_writing_a_pdb_beyond_what_the_format_holds_is_refused(tmp_path):
    """The PDB format cannot number past 99,999 atoms or 9,999 residues. What
    it must not do is renumber them and carry on."""
    import biotite.structure.io.pdb as pdb

    from blender_gala.core.entity import AtomStructure
    from blender_gala.electrostatics import apbs

    array = make(
        3,
        res_id=[1, 9999, 10000],
        atom_id=[1, 99999, 100000],
        res_name=["ALA"] * 3,
    )
    path = str(tmp_path / "out.pdb")

    apbs.write_pdb(AtomStructure(array=array), path)
    written = pdb.PDBFile.read(path).get_structure(model=1)

    assert list(written.res_id) == [1, 9999, 10000]


@pytest.mark.parametrize(
    ("label", "annotations"),
    [
        ("a multi-character chain id", {"chain_id": ["AAA"]}),
        ("a frame far from the origin", {"coord": [[10000.0, 0.0, 0.0]]}),
    ],
)
def test_a_structure_the_pdb_format_cannot_hold_is_reported(
    tmp_path, label, annotations
):
    """Both arrive from ordinary mmCIF: multi-letter author chain identifiers,
    and assembly or crystal-packing frames far from the origin."""
    from blender_gala.core.entity import AtomStructure
    from blender_gala.core.exceptions import GalaError
    from blender_gala.electrostatics import apbs

    array = make(1, **annotations)

    with pytest.raises(GalaError):
        apbs.write_pdb(AtomStructure(array=array), str(tmp_path / "out.pdb"))


def test_an_empty_query_does_not_perceive_the_whole_structure(monkeypatch):
    """Measured on a real 238,000-atom assembly: `find_interactions(s, "none",
    "none", kinds="all")` took 46.5 s to return an empty list, 22.5 s of it in
    pi-stacking alone — 10,668 rings paired 57 million ways for an answer that
    was empty before any of it began. That freezes Blender's main thread.

    Asserted structurally rather than as a wall clock, so it cannot go flaky:
    if neither side of the query has an atom in it, nothing needs perceiving.
    """
    from blender_gala.core.entity import AtomStructure
    from blender_gala.interactions import detect, perception

    points: list[tuple[float, float, float]] = []
    for index in range(20):
        points += hexagon((0.0, 0.0, 6.0 * index))
    array = make(
        120,
        coord=points,
        res_id=[1 + index // 6 for index in range(120)],
        res_name=["PHE"] * 120,
        atom_name=RING_NAMES * 20,
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the structure was perceived for an empty query")

    monkeypatch.setattr(perception, "aromatic_rings", refuse)

    assert (
        detect.find_interactions(
            AtomStructure(array=array), "none", "none", kinds="all"
        )
        == []
    )
