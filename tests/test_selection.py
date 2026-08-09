"""Tests for the PyMOL-compatible selection language."""

from __future__ import annotations

import numpy as np
import pytest

from blender_gala.core.exceptions import SelectionSyntaxError
from blender_gala.core.selection import (
    LEVELS,
    MACRO_KEYWORDS,
    PROPERTY_KEYWORDS,
    Selection,
    SelectionContext,
    compile_selection,
    describe_selection,
    expand_selection,
    select,
    select_indices,
)


class Fake:
    """A minimal stand-in for an ``AtomArray``, so the parser needs no biotite."""

    def __init__(self, **annotations):
        for key, value in annotations.items():
            setattr(self, key, np.asarray(value))
        self._n = len(self.element)

    def __len__(self):
        return self._n


@pytest.fixture
def array():
    return Fake(
        chain_id=["A", "A", "A", "A", "A", "B", "B", "B"],
        res_id=[1, 1, 1, 1, 2, 10, 10, 999],
        res_name=["ALA", "ALA", "ALA", "ALA", "PHE", "LYS", "LYS", "HOH"],
        atom_name=["N", "CA", "C", "CB", "CA", "CA", "NZ", "O"],
        element=["N", "C", "C", "C", "C", "C", "N", "O"],
        b_factor=[92.0, 92.0, 88.0, 60.0, 45.0, 30.0, 30.0, 20.0],
        occupancy=[1.0] * 8,
        hetero=[False] * 7 + [True],
        coord=[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1, 1, 0],
            [5, 0, 0],
            [10, 0, 0],
            [11, 0, 0],
            [20, 0, 0],
        ],
    )


def count(array, expression):
    return int(select(array, expression).sum())


def count_in(array, expression, context):
    return int(select(array, expression, context).sum())


# ---------------------------------------------------------------------------
# Property selectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("all", 8),
        ("none", 0),
        ("chain A", 5),
        ("chain B", 3),
        ("chain A+B", 8),
        ("chain A,B", 8),
        ("resi 1", 4),
        ("resi 1-2", 5),
        ("resi 1+10", 6),
        ("resi 10-", 3),
        ("resi :2", 5),
        ("resi -2", 0),  # a bare minus is a negative residue number, as in PyMOL
        ("resn ALA", 4),
        ("resn ALA+PHE", 5),
        ("name CA", 3),
        ("name CA+CB", 4),
        ("elem N", 2),
        ("index 1", 1),
        ("index 1-3", 3),
    ],
)
def test_property_selectors(array, expression, expected):
    assert count(array, expression) == expected


def test_selectors_are_case_insensitive(array):
    assert count(array, "chain a") == count(array, "chain A") == 5
    assert count(array, "RESN ala") == 4
    assert count(array, "Name ca") == 3


def test_wildcards(array):
    # C* matches CA, C, CB, CA, CA.
    assert count(array, "name C*") == 5
    assert count(array, "name C?") == 4
    assert count(array, "resn A*") == 4


def test_every_property_keyword_parses(array):
    for keyword in PROPERTY_KEYWORDS:
        # A syntactically valid use of each keyword must at least parse.
        compile_selection(f"{keyword} 1")


# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("b > 70", 3),
        ("b >= 88", 3),
        ("b < 50", 4),
        ("b <= 45", 4),
        ("b = 92", 2),
        ("b != 92", 6),
        ("q >= 1", 8),
    ],
)
def test_numeric_comparison(array, expression, expected):
    assert count(array, expression) == expected


def test_string_equality_operators(array):
    assert count(array, "resn = ALA") == 4
    assert count(array, "resn != ALA") == 4


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("protein", 7),
        ("polymer", 7),
        ("water", 1),
        ("solvent", 1),
        ("hetatm", 1),
        ("backbone", 5),  # N, CA, C of ALA1 plus the two other CA atoms
        ("sidechain", 2),  # ALA1 CB and LYS10 NZ
        ("carbon", 5),
        ("polar", 3),
        ("hydro", 0),
        ("ligand", 0),
    ],
)
def test_macros(array, expression, expected):
    assert count(array, expression) == expected


def test_every_macro_evaluates(array):
    for macro in MACRO_KEYWORDS:
        result = select(array, macro)
        assert result.shape == (len(array),)
        assert result.dtype == bool


# ---------------------------------------------------------------------------
# Boolean logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("chain A and name CA", 2),
        ("chain A or chain B", 8),
        ("not protein", 1),
        ("chain A and not backbone", 1),
        ("(chain A) and (resi 1)", 4),
        ("chain A & name CA", 2),
        ("chain A | chain B", 8),
        ("!protein", 1),
        ("not (water or hydro)", 7),
        ("chain A and (name CA or name CB)", 3),
    ],
)
def test_boolean_logic(array, expression, expected):
    assert count(array, expression) == expected


def test_and_binds_tighter_than_or(array):
    # "chain B and name NZ" is 1 atom; or-ing with resi 2 gives 2.
    assert count(array, "resi 2 or chain B and name NZ") == 2


# ---------------------------------------------------------------------------
# Spatial and expansion operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("all within 2 of resi 2", 1),  # only the atom itself
        ("protein within 3 of resi 1", 5),  # 3.0 A is inclusive
        ("resi 1 around 5", 1),
        ("chain A expand 4", 5),
        ("byres (name CB)", 4),
        ("bychain (name NZ)", 3),
        ("first chain A", 1),
        ("last chain A", 1),
    ],
)
def test_spatial_operators(array, expression, expected):
    assert count(array, expression) == expected


def test_around_excludes_the_source(array):
    around = select(array, "resi 1 around 5")
    source = select(array, "resi 1")
    assert not (around & source).any()


def test_within_includes_the_source(array):
    within = select(array, "all within 5 of resi 1")
    assert within[select(array, "resi 1")].all()


def test_byres_expands_to_whole_residues(array):
    mask = select(array, "byres (name CB)")
    assert count(array, "resi 1 and chain A") == int(mask.sum())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "chain",  # missing value
        "resi abc",  # non-integer
        "bogus X",  # unknown keyword
        "(chain A",  # unbalanced parenthesis
        "chain A and",  # dangling operator
        "chain A)",  # stray close
        "all within of resi 1",  # missing distance
        "all within 3 resi 1",  # missing 'of'
        "b >",  # missing comparison value
    ],
)
def test_syntax_errors(array, expression):
    with pytest.raises(SelectionSyntaxError):
        select(array, expression)


def test_syntax_error_points_at_the_problem(array):
    with pytest.raises(SelectionSyntaxError) as info:
        select(array, "chain A and bogus B")
    message = str(info.value)
    assert "bogus" in message
    assert "^" in message  # the caret line locating the token


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_select_indices_returns_positions(array):
    indices = select_indices(array, "name CA")
    assert indices.tolist() == [1, 4, 5]


def test_masks_pass_through(array):
    mask = np.zeros(len(array), dtype=bool)
    mask[[0, 3]] = True
    assert select(array, mask).tolist() == mask.tolist()


def test_index_arrays_pass_through(array):
    result = select(array, np.array([0, 3]))
    assert result.tolist() == [True, False, False, True, False, False, False, False]


def test_wrong_length_mask_raises(array):
    with pytest.raises(ValueError, match="length"):
        select(array, np.zeros(3, dtype=bool))


def test_bad_selection_type_raises(array):
    with pytest.raises(TypeError):
        select(array, 3.5)


def test_compiled_selections_are_cached():
    assert compile_selection("chain A") is compile_selection("chain A")


def test_selection_is_reusable_across_arrays(array):
    selection = Selection("name CA")
    other = Fake(
        chain_id=["Z"],
        res_id=[1],
        res_name=["GLY"],
        atom_name=["CA"],
        element=["C"],
        b_factor=[1.0],
        occupancy=[1.0],
        hetero=[False],
        coord=[[0, 0, 0]],
    )
    assert int(selection.evaluate(array).sum()) == 3
    assert int(selection.evaluate(other).sum()) == 1


def test_repr_is_useful():
    assert repr(Selection("chain A")) == "Selection('chain A')"


# ---------------------------------------------------------------------------
# Named selections
# ---------------------------------------------------------------------------


@pytest.fixture
def named(array):
    """A context carrying one stored selection: the two atoms of chain B's LYS."""
    pocket = select(array, "resn LYS")
    return SelectionContext(array, named={"pocket": pocket})


def test_a_name_resolves_to_its_stored_mask(array, named):
    assert select(array, "pocket", named).tolist() == select(array, "resn LYS").tolist()


def test_a_name_is_a_selection_like_any_other(array, named):
    """The point of the feature: a stored pick composes with the language."""
    assert count_in(array, "not pocket", named) == 6
    assert count_in(array, "pocket and name NZ", named) == 1
    assert count_in(array, "byres (name NZ)", named) == count_in(array, "pocket", named)
    # The residue 10 A away, found from the stored selection alone.
    assert count_in(array, "resn HOH within 10 of pocket", named) == 1


def test_a_name_is_matched_regardless_of_case(array, named):
    assert count_in(array, "POCKET", named) == 2


def test_a_keyword_wins_over_a_name_of_its_own(array):
    """A stored selection called `ligand` does not quietly redefine the macro."""
    ctx = SelectionContext(array, named={"ligand": select(array, "chain A")})
    assert count_in(array, "ligand", ctx) == 0  # the macro: no ligand here
    assert count_in(array, "%ligand", ctx) == 5  # the stored selection


def test_an_unknown_name_says_what_is_stored(array, named):
    with pytest.raises(SelectionSyntaxError) as info:
        select(array, "pockte around 4", named)
    message = str(info.value)
    assert "pockte" in message
    assert "pocket" in message  # what was stored, offered as the alternative
    assert "^" in message


def test_an_unknown_name_without_any_stored_says_so(array):
    with pytest.raises(SelectionSyntaxError, match="no stored selections"):
        select(array, "pocket")


def test_a_percent_needs_a_name(array):
    with pytest.raises(SelectionSyntaxError, match="needs the name"):
        select(array, "%")


def test_a_stored_mask_of_the_wrong_length_is_not_used(array):
    """It would address the wrong atoms; better to say the name is unknown."""
    ctx = SelectionContext(array, named={"pocket": np.ones(3, dtype=bool)})
    with pytest.raises(SelectionSyntaxError, match="pocket"):
        select(array, "pocket", ctx)


def test_a_name_is_resolved_per_structure_not_per_string(array, named):
    """Compiling is cached across structures, so the name cannot bind at parse
    time — the same string means different atoms on different molecules."""
    selection = compile_selection("pocket")
    other = SelectionContext(array, named={"pocket": select(array, "chain A")})
    assert int(selection.evaluate(array, named).sum()) == 2
    assert int(selection.evaluate(array, other).sum()) == 5


def test_expand_and_describe_take_names_too(array, named):
    grown = expand_selection(array, "pocket", "chain", named)
    assert grown.tolist() == select(array, "chain B").tolist()
    assert describe_selection(array, "pocket", named) == "chain B and resi 10"


# ---------------------------------------------------------------------------
# Against a real structure
# ---------------------------------------------------------------------------


def test_selection_on_real_structure(site_array):
    assert int(select(site_array, "protein").sum()) > 0
    assert int(select(site_array, "resn LIG").sum()) == 9
    assert int(select(site_array, "chain B").sum()) == 9
    assert int(select(site_array, "resn LIG and elem CL").sum()) == 1
    # The ligand ring sits 3.8 A above the PHE ring, so a 4 A shell catches it.
    near = select(site_array, "byres (protein within 4.0 of resn LIG)")
    assert int(near.sum()) > 0


# ---------------------------------------------------------------------------
# Selection levels
# ---------------------------------------------------------------------------


def test_expand_grows_to_whole_residues(array):
    picked = select(array, "index 2")
    assert int(picked.sum()) == 1
    assert int(expand_selection(array, picked, "residue").sum()) == 4


def test_expand_grows_to_whole_chains(array):
    assert int(expand_selection(array, "index 2", "chain").sum()) == 5


def test_expand_at_atom_level_changes_nothing(array):
    picked = select(array, "name CA")
    assert np.array_equal(expand_selection(array, picked, "atom"), picked)


def test_expand_at_object_level_takes_everything(array):
    assert expand_selection(array, "index 1", "object").all()


def test_expand_of_nothing_stays_nothing(array):
    for level in LEVELS:
        assert not expand_selection(array, "none", level).any()


def test_expand_levels_compose(array):
    once = expand_selection(array, "index 2", "residue")
    twice = expand_selection(array, once, "chain")
    assert int(twice.sum()) == 5


def test_expand_rejects_an_unknown_level(array):
    with pytest.raises(ValueError, match="unknown selection level"):
        expand_selection(array, "all", "molecule")


def test_fragment_falls_back_to_chain_without_bonds(array):
    # `array` carries no bond list, so the honest answer for "everything
    # bonded to this" is the chain it belongs to.
    assert np.array_equal(
        expand_selection(array, "index 2", "fragment"),
        expand_selection(array, "index 2", "chain"),
    )


def test_fragment_follows_bonds_when_there_are_any(array):
    # Two atoms of chain A bonded to each other, and nothing else.
    array.bonds = _Bonds([[0, 1, 1]])
    fragment = expand_selection(array, "index 1", "fragment")
    assert list(np.flatnonzero(fragment)) == [0, 1]


class _Bonds:
    """The part of a biotite ``BondList`` the fragment level uses."""

    def __init__(self, table):
        self._table = np.asarray(table, dtype=int)

    def as_array(self):
        return self._table


def test_bymol_is_the_fragment_level_in_the_language(array):
    array.bonds = _Bonds([[0, 1, 1]])
    assert list(np.flatnonzero(select(array, "bymol index 1"))) == [0, 1]


# ---------------------------------------------------------------------------
# Describing a mask
# ---------------------------------------------------------------------------


def _round_trips(array, selection) -> str:
    """Assert the description of a selection selects the same atoms again."""
    mask = select(array, selection)
    text = describe_selection(array, mask)
    assert np.array_equal(select(array, text), mask), f"{selection!r} -> {text!r}"
    return text


def test_describe_the_extremes(array):
    assert describe_selection(array, "all") == "all"
    assert describe_selection(array, "none") == "none"


def test_describe_a_whole_chain(array):
    assert _round_trips(array, "chain A") == "chain A"


@pytest.fixture
def peptides():
    """Three chains of five residues, two atoms each — room for real ranges."""
    chains, res_ids, names = [], [], []
    for chain in "ABC":
        for res_id in range(1, 6):
            chains += [chain, chain]
            res_ids += [res_id, res_id]
            names += ["CA", "CB"]
    n = len(chains)
    return Fake(
        chain_id=chains,
        res_id=res_ids,
        res_name=["GLY"] * n,
        atom_name=names,
        element=["C"] * n,
        b_factor=[1.0] * n,
        occupancy=[1.0] * n,
        hetero=[False] * n,
        coord=[[0, 0, 0]] * n,
    )


def test_describe_several_whole_chains_as_one_clause(peptides):
    assert _round_trips(peptides, "chain B or chain C") == "chain B+C"


def test_describe_whole_residues_as_a_range(peptides):
    assert _round_trips(peptides, "chain A and resi 2-4") == "chain A and resi 2-4"


def test_describe_groups_partial_residues_by_atom_name(peptides):
    # One clause for the lot, not one clause per residue.
    assert (
        _round_trips(peptides, "chain A and name CA")
        == "chain A and resi 1-5 and name CA"
    )


def test_describe_keeps_whole_and_partial_residues_apart(peptides):
    text = _round_trips(peptides, "(chain A and resi 1) or (chain A and name CA)")
    assert text == "(chain A and resi 1) or (chain A and resi 2-5 and name CA)"


def test_describe_spans_chains(array):
    text = _round_trips(array, "chain A and resi 1 or chain B and resi 999")
    assert " or " in text


def test_describe_is_verified_and_falls_back_to_index(array):
    # Blank chain identifiers cannot be written as `chain X`, so the
    # positional form is used instead — and it is still exact.
    blank = Fake(
        chain_id=["", ""],
        res_id=[1, 1],
        res_name=["ALA", "ALA"],
        atom_name=["N", "CA"],
        element=["N", "C"],
        b_factor=[1.0, 1.0],
        occupancy=[1.0, 1.0],
        hetero=[False, False],
        coord=[[0, 0, 0], [1, 0, 0]],
    )
    assert describe_selection(blank, np.array([True, False])) == "index 1"


def test_describe_compacts_runs_but_not_negative_ones():
    array = Fake(
        chain_id=["A"] * 5,
        res_id=[-2, -1, 3, 4, 5],
        res_name=["GLY"] * 5,
        atom_name=["CA"] * 5,
        element=["C"] * 5,
        b_factor=[1.0] * 5,
        occupancy=[1.0] * 5,
        hetero=[False] * 5,
        coord=[[0, 0, 0]] * 5,
    )
    text = describe_selection(array, "resi 3-5")
    assert text == "chain A and resi 3-5"
    # A run that starts below zero is ambiguous with PyMOL's negative residue
    # numbering, so it is written out rather than collapsed.
    assert describe_selection(array, "resi -2+-1") == "chain A and resi -2+-1"


def test_describe_round_trips_on_a_real_structure(site_array):
    for selection in (
        "all",
        "protein",
        "chain B",
        "name CA",
        "backbone",
        "sidechain",
        "byres (protein within 4.0 of resn LIG)",
        "b > 0",
        "index 1+5+9-12",
    ):
        _round_trips(site_array, selection)
