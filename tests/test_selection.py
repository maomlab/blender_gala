"""Tests for the PyMOL-compatible selection language."""

from __future__ import annotations

import numpy as np
import pytest

from blender_gala.core.exceptions import SelectionSyntaxError
from blender_gala.core.selection import (
    MACRO_KEYWORDS,
    PROPERTY_KEYWORDS,
    Selection,
    compile_selection,
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
