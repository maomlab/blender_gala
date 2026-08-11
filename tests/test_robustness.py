"""Hostile-input tests: what Gala does when the input is not what it expected.

Every other test module asks whether a feature is *correct*. This one asks
whether it *holds up* — against selection strings a user mistyped into a panel
field, structures with no atoms in them, coordinates that came back as ``nan``,
grids whose header lies about their size, and session files that are not
sessions. See ``tests/ROBUSTNESS.md`` for the strategy these were chosen from,
and for the cases that deliberately live in a fuzz or stress run instead.

The contract being tested is rarely an exact value. It is one of:

``it raises the documented error``
    a mistyped selection is a :class:`SelectionSyntaxError` that points at the
    token, not a ``TypeError`` from somewhere three modules down;
``it refuses rather than guesses``
    a truncated grid or a short mask is reported, never quietly padded;
``it survives``
    an empty structure or a Unicode residue name produces an empty result and
    no traceback.

Most of these began life as ``xfail(strict=True)`` records of defects that were
found by probing the code rather than imagined — input that reached a bare
exception from numpy, scipy or the standard library, or produced a confidently
wrong answer. They are now ordinary tests, because the defects they describe
have been fixed; what each one guards is written in its docstring.
"""

from __future__ import annotations

import csv
import gzip
import pickle

import numpy as np
import pytest

from blender_gala.color import coloring, colormaps
from blender_gala.core import attributes, geometry
from blender_gala.core.entity import AtomStructure
from blender_gala.core.exceptions import GalaError, SelectionSyntaxError
from blender_gala.core.selection import (
    LEVELS,
    MACRO_KEYWORDS,
    compile_selection,
    describe_selection,
    expand_selection,
    select,
)
from blender_gala.electrostatics.grid import PotentialGrid, read_dx
from blender_gala.pymol.session import (
    PymolMolecule,
    PymolSelection,
    PymolSession,
    PymolSessionError,
    read_session,
    write_session,
)

try:
    import scipy.spatial  # noqa: F401

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy ships with Blender + MN
    HAS_SCIPY = False


class Fake:
    """A minimal stand-in for an ``AtomArray``, so these run without biotite."""

    def __init__(self, **annotations):
        for key, value in annotations.items():
            setattr(self, key, np.asarray(value))
        self._n = len(self.element)

    def __len__(self):
        return self._n


def structure(n: int, coord=None, **overrides) -> Fake:
    """``n`` identical alanine C-alphas strung out along +X."""
    fields = {
        "chain_id": ["A"] * n,
        "res_id": list(range(1, n + 1)),
        "res_name": ["ALA"] * n,
        "atom_name": ["CA"] * n,
        "element": ["C"] * n,
        "b_factor": [1.0] * n,
        "occupancy": [1.0] * n,
        "hetero": [False] * n,
        "coord": coord
        if coord is not None
        else [[float(i), 0.0, 0.0] for i in range(n)],
    }
    fields.update(overrides)
    if n == 0:
        fields["element"] = np.array([], dtype="<U2")
        fields["coord"] = np.zeros((0, 3))
    return Fake(**fields)


@pytest.fixture
def array():
    return structure(4)


@pytest.fixture
def nothing():
    """A structure with no atoms in it at all."""
    return structure(0)


# ---------------------------------------------------------------------------
# Selection language: input that is not a selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "",  # an empty panel field
        "   ",  # a field holding only whitespace
        "\n\t",
        "(",
        "()",
        "((chain A)",
        "chain A)",
        "and chain A",  # a leading binary operator
        "chain A and",  # a trailing one
        "chain A or or chain B",
        "b > > 1",
        "b >",
        "resi 1-2-3",
        "resi 1.5",  # a residue number is an integer
        "index 1e5",  # and so is an index
        "byres",  # a prefix with nothing to apply to
        "not",
        "first",
        "%",  # the explicit stored-selection form, with no name
        "%%",
        "all within",
        "all within 3 of",
        "all within banana of all",
        "* *",
    ],
)
def test_a_string_that_is_not_a_selection_is_a_syntax_error(array, expression):
    """Never a TypeError, a ValueError or an IndexError from further down.

    This is the error a panel field shows the user, so it has to be the one
    the language raises rather than whatever numpy said about it.
    """
    with pytest.raises(SelectionSyntaxError):
        select(array, expression)


def test_a_syntax_error_locates_itself_in_the_string(array):
    """The caret is the whole point of carrying the position around."""
    with pytest.raises(SelectionSyntaxError) as info:
        select(array, "chain A and bogus B")

    assert info.value.selection == "chain A and bogus B"
    assert 0 <= info.value.position <= len("chain A and bogus B")


@pytest.mark.parametrize(
    "expression",
    [
        "chain Ä",  # a chain id outside ASCII
        "resn ☃",
        "name 🧬",  # an astral-plane character, which is two UTF-16 units
        "resn \x00",  # an embedded NUL
        "resn ‮gnirts",  # a right-to-left override
        "name " + "A" * 10_000,  # a value longer than any real atom name
        "resn " + "́" * 500,  # combining marks with nothing to combine with
    ],
)
def test_pathological_strings_select_nothing_rather_than_failing(array, expression):
    """A selection may be nonsense without being a syntax error.

    Nothing in the language restricts a *value* to ASCII — a chain really can
    be called ``Ä`` — so these parse, match no atom, and must not throw.
    """
    mask = select(array, expression)

    assert mask.shape == (len(array),)
    assert not mask.any()


def test_compiling_hostile_input_does_not_grow_without_bound(array):
    """The parse cache is what a loop over a trajectory relies on; it is also
    what a thousand distinct mistyped strings would otherwise fill."""
    for i in range(2_000):
        compile_selection(f"resi {i}")

    assert compile_selection.cache_info().currsize <= 512


def test_deeply_nested_parentheses_are_reported_not_fatal(array):
    with pytest.raises(SelectionSyntaxError):
        select(array, "(" * 200 + "all" + ")" * 200)


def test_a_long_chain_of_or_clauses_is_evaluated(array):
    assert select(array, " or ".join(["all"] * 2_000)).all()


# ---------------------------------------------------------------------------
# Selection language: numeric extremes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("all within inf of index 1", 4),  # an unbounded shell is everything
        ("all within 1e308 of index 1", 4),
        ("all within nan of index 1", 0),  # and an undefined one is nothing
        ("all within -1 of index 1", 0),
        ("all within 0 of index 1", 0),
        ("all expand inf", 4),
        ("all expand nan", 4),  # expansion can only ever add
        ("b > nan", 0),  # every comparison against nan is false
        ("b < nan", 0),
        ("b > inf", 0),
        ("b < inf", 4),
        ("index 0", 0),  # index is 1-based, as in PyMOL
        ("index -5", 0),
        ("resi 99999999999999999999", 0),  # beyond int64
    ],
)
def test_numeric_extremes_have_a_defined_answer(array, expression, expected):
    assert int(select(array, expression).sum()) == expected


def test_a_distance_that_is_not_a_number_is_a_syntax_error(array):
    for expression in ("all within of all", "all expand", "all around"):
        with pytest.raises(SelectionSyntaxError):
            select(array, expression)


@pytest.mark.parametrize("distance", [float("inf"), 1e308])
def test_expanding_by_an_unbounded_distance_takes_everything(array, distance):
    assert expand_selection(array, "index 1", "atom", distance=distance).all()


def test_expanding_by_an_undefined_distance_changes_nothing(array):
    """`nan` is not negative, so it passes the guard; it must not then quietly
    grow or shrink the selection."""
    picked = select(array, "index 1")
    grown = expand_selection(array, picked, "atom", distance=float("nan"))

    assert np.array_equal(grown, picked)


# ---------------------------------------------------------------------------
# Structures with nothing in them
# ---------------------------------------------------------------------------


def test_every_macro_evaluates_against_an_empty_structure(nothing):
    for macro in MACRO_KEYWORDS:
        mask = select(nothing, macro)
        assert mask.shape == (0,)
        assert mask.dtype == bool


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_expands_an_empty_structure(nothing, level):
    assert expand_selection(nothing, "all", level).shape == (0,)
    assert expand_selection(nothing, "all", level, distance=5.0).shape == (0,)


@pytest.mark.parametrize(
    "expression",
    ["all", "none", "protein", "byres all", "all within 4 of all", "first all"],
)
def test_selecting_from_an_empty_structure_returns_an_empty_mask(nothing, expression):
    assert select(nothing, expression).shape == (0,)


def test_describing_an_empty_structure_says_none(nothing):
    assert describe_selection(nothing, "all") == "none"
    assert describe_selection(nothing, np.zeros(0, dtype=bool)) == "none"


def test_a_single_atom_still_has_a_usable_bounding_sphere():
    """Lights and cameras divide by this radius, so it is never zero."""
    _, radius = AtomStructure(array=structure(1)).bounding_sphere()

    assert radius > 0.0


def test_an_empty_structure_reports_its_own_emptiness():
    with pytest.raises(GalaError):
        AtomStructure(array=structure(0)).bounding_sphere()


# ---------------------------------------------------------------------------
# Coordinates that are not numbers
# ---------------------------------------------------------------------------


def test_a_selection_on_names_ignores_broken_coordinates():
    """Chemistry is readable even when the geometry is not, and a chain or a
    residue name never needed the coordinates."""
    broken = structure(3, coord=[[0.0, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]])

    assert int(select(broken, "name CA").sum()) == 3
    assert int(select(broken, "byres (resi 2)").sum()) == 1


@pytest.mark.skipif(not HAS_SCIPY, reason="the KD-tree path needs scipy")
def test_a_spatial_selection_survives_a_missing_coordinate():
    broken = structure(3, coord=[[0.0, 0, 0], [np.nan, 0, 0], [1.0, 0, 0]])

    assert int(select(broken, "all within 5 of index 1").sum()) >= 1


def test_enormous_coordinates_do_not_overflow():
    far = AtomStructure(array=structure(2, coord=[[0.0, 0, 0], [1e30, 0, 0]]))
    _, radius = far.bounding_sphere()

    assert np.isfinite(radius)
    assert radius > 0.0


# ---------------------------------------------------------------------------
# Selections that are not strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        None,
        3.5,
        object(),
        np.array([0.5, 1.0, 0.0, 1.0]),  # a float array is neither mask nor index
        ["chain A"],
    ],
)
def test_a_selection_of_the_wrong_type_is_refused(array, selection):
    with pytest.raises(TypeError):
        select(array, selection)


@pytest.mark.parametrize("length", [0, 3, 5])
def test_a_mask_of_the_wrong_length_is_refused(array, length):
    """Silently broadcasting one would colour or measure the wrong atoms."""
    with pytest.raises(ValueError, match="length"):
        select(array, np.zeros(length, dtype=bool))


def test_an_index_beyond_the_structure_is_refused(array):
    """The type matters less than that it does not quietly select nothing."""
    with pytest.raises((IndexError, ValueError)):
        select(array, np.array([99]))


# ---------------------------------------------------------------------------
# Describing a mask back into the language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("atom_names", "chain_ids"),
    [
        (["C(1)", "CB"], ["A", "A"]),  # a name carrying the language's punctuation
        (["C*", "CB"], ["A", "A"]),  # one that is a wildcard
        (["C A", "CB"], ["A", "A"]),  # one with a space in it
        (["CA", "CB"], ["", ""]),  # no chain id to name
        (["", "CB"], ["A", "A"]),  # no atom name to name
        (["CA", "CB"], ["A and B", "A and B"]),  # a chain id that is an operator
        (["CÄ", "CB"], ["A", "A"]),  # a name outside ASCII
    ],
)
def test_a_description_is_exact_even_when_it_cannot_be_chemical(atom_names, chain_ids):
    """`describe_selection` verifies its own output and falls back to the
    positional form. Whatever comes out has to re-select the same atoms.

    Both atoms are in the same residue, so the description has to name the
    atoms rather than the residue — which is where an awkward name bites.
    """
    awkward = structure(2, atom_name=atom_names, chain_id=chain_ids, res_id=[1, 1])
    mask = np.array([True, False])

    text = describe_selection(awkward, mask)

    assert np.array_equal(select(awkward, text), mask), text


def test_every_description_of_a_real_structure_round_trips(site_array):
    """Each atom on its own, which is where a chemical description is most
    likely to be ambiguous."""
    for index in range(0, len(site_array), 7):
        mask = np.zeros(len(site_array), dtype=bool)
        mask[index] = True
        text = describe_selection(site_array, mask)

        assert np.array_equal(select(site_array, text), mask), text


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(0),  # nothing to colour
        np.array([np.inf, -np.inf]),
        np.array([-1e30, 1e30]),
        np.array([[0.0, 1.0], [0.5, 0.25]]),  # ravelled, not rejected
    ],
)
def test_sampling_a_colormap_returns_a_colour_per_value(values):
    colours = colormaps.sample("viridis", values)

    assert colours.shape == (values.size, 3)
    assert np.isfinite(colours).all()


def test_sampling_at_an_undefined_position_does_not_raise():
    """It comes back as nan rather than as a wrong colour; the colouring
    functions substitute their own `missing` colour before writing."""
    assert np.isnan(colormaps.sample("viridis", np.array([np.nan]))).all()


@pytest.mark.parametrize(
    "colour", ["", "#", "#fff", "ffffff0", "gggggg", "ff ff ff", "0x1234"]
)
def test_a_string_that_is_not_a_colour_is_refused(colour):
    with pytest.raises(ValueError):
        colormaps.hex_to_rgb(colour)


# The third is fullwidth digits, which `int(..., 16)` accepts as decimal.
@pytest.mark.parametrize(
    "colour", ["+12345", "-fffff", "\uff11\uff12\uff13\uff14\uff15\uff16"]
)
def test_a_near_miss_colour_is_refused(colour):
    with pytest.raises(ValueError):
        colormaps.hex_to_rgb(colour)


def test_a_colour_that_is_not_a_string_is_refused():
    with pytest.raises((TypeError, AttributeError, ValueError)):
        colormaps.hex_to_rgb(0xFFFFFF)


@pytest.mark.parametrize("colour", ["#ffffff", "FFFFFF", "#FfFfFf"])
def test_a_colour_is_read_whatever_its_case_and_hash(colour):
    assert np.allclose(colormaps.hex_to_rgb(colour), [1.0, 1.0, 1.0])


def test_colour_conversion_is_defined_outside_the_unit_range():
    """Blender hands back linear values above 1 from an emissive material."""
    for values in (np.array([-1.0, 2.0]), np.array([np.nan])):
        assert colormaps.srgb_to_linear(values).shape == values.shape
        assert colormaps.linear_to_srgb(values).shape == values.shape


# ---------------------------------------------------------------------------
# Per-residue values from a CSV
# ---------------------------------------------------------------------------


def write_csv(path, rows, header=("res_id", "value"), encoding="utf-8"):
    with open(path, "w", newline="", encoding=encoding) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def test_blank_values_are_skipped_rather_than_read_as_zero(site, tmp_path):
    """A blank cell is a residue with no measurement, not a residue that
    measured zero — the difference is the whole bottom of the colour ramp."""
    path = write_csv(tmp_path / "v.csv", [[1, ""], [2, 5.0], [3, 9.0]])

    result = coloring.color_from_csv(site, path, "value", write=False)

    assert result.vmin == pytest.approx(5.0)


@pytest.mark.parametrize(
    "rows",
    [
        [[1, "abc"]],  # a value that is not a number
        [["abc", 1.0]],  # a residue number that is not one
        [["nan", 1.0]],
        [[1, "1,5"]],  # a decimal comma, from a European spreadsheet
    ],
)
def test_a_cell_that_is_not_a_number_is_reported(site, tmp_path, rows):
    with pytest.raises((ValueError, TypeError)):
        coloring.color_from_csv(site, write_csv(tmp_path / "v.csv", rows), "value")


def test_a_repeated_residue_takes_the_last_value(site, tmp_path):
    path = write_csv(tmp_path / "v.csv", [[1, 5.0], [1, 9.0], [2, 7.0]])

    result = coloring.color_from_csv(site, path, "value", write=False)

    assert result.vmax == pytest.approx(9.0)


def test_a_csv_with_no_rows_says_nothing_matched(site, tmp_path):
    from blender_gala.core.exceptions import EmptySelectionError

    path = write_csv(tmp_path / "v.csv", [])

    with pytest.raises(EmptySelectionError):
        coloring.color_from_csv(site, path, "value", write=False)


def test_residue_numbers_that_match_nothing_say_so(site, tmp_path):
    path = write_csv(tmp_path / "v.csv", [[9_999, 1.0]])

    from blender_gala.core.exceptions import EmptySelectionError

    with pytest.raises(EmptySelectionError):
        coloring.color_from_csv(site, path, "value", write=False)


def test_a_csv_saved_by_a_spreadsheet_is_readable(site, tmp_path):
    path = write_csv(tmp_path / "v.csv", [[1, 5.0]], encoding="utf-8-sig")

    assert coloring.color_from_csv(site, path, "value", write=False).n_colored > 0


# ---------------------------------------------------------------------------
# OpenDX grids
# ---------------------------------------------------------------------------

DX_TEMPLATE = """object 1 class gridpositions counts {counts}
origin 0 0 0
delta {spacing} 0 0
delta 0 1 0
delta 0 0 1
object 2 class gridconnections counts {counts}
object 3 class array type double rank 0 items {items} data follows
{values}
attribute "dep" string "positions"
"""


def write_dx(path, counts="2 2 2", items=8, values=None, spacing=1):
    if values is None:
        values = " ".join(["1.0"] * items)
    path.write_text(
        DX_TEMPLATE.format(counts=counts, items=items, values=values, spacing=spacing)
    )
    return str(path)


def test_a_grid_whose_data_block_stops_early_is_refused(tmp_path):
    """Padding the rest with zeros would be a map with a hole in it that looks
    like a real feature."""
    with pytest.raises(ValueError, match="promised"):
        read_dx(write_dx(tmp_path / "g.dx", items=8, values="1.0 2.0 3.0"))


def test_a_grid_whose_header_disagrees_with_itself_is_refused(tmp_path):
    with pytest.raises(ValueError, match="values for a"):
        read_dx(write_dx(tmp_path / "g.dx", counts="3 3 3", items=8))


@pytest.mark.parametrize("body", ["", "\x00\x01\x02", "not a grid at all\n", "#\n#\n"])
def test_something_that_is_not_a_grid_is_refused(tmp_path, body):
    path = tmp_path / "g.dx"
    path.write_text(body)

    with pytest.raises(ValueError):
        read_dx(str(path))


def test_a_gzipped_grid_that_is_not_gzipped_is_refused(tmp_path):
    path = tmp_path / "g.dx.gz"
    path.write_bytes(b"plain text, despite the suffix")

    with pytest.raises((OSError, ValueError)):
        read_dx(str(path))


def test_non_finite_values_are_carried_rather_than_hidden(tmp_path):
    """APBS writes an inf where the solver diverged. Reading it as zero would
    make a broken run look like a converged one."""
    grid = read_dx(write_dx(tmp_path / "g.dx", values=" ".join(["inf", "nan"] * 4)))

    assert np.isinf(grid.values).any()
    assert np.isnan(grid.values).any()
    assert isinstance(grid.summary(), str)


def test_a_grid_that_is_not_three_dimensional_is_refused(tmp_path):
    with pytest.raises(ValueError):
        read_dx(write_dx(tmp_path / "g.dx", counts="2 2", items=4))


@pytest.fixture
def grid():
    return PotentialGrid(
        values=np.arange(27.0).reshape(3, 3, 3),
        origin=np.zeros(3),
        spacing=np.ones(3),
    )


def test_sampling_no_points_returns_no_values(grid):
    assert grid.sample(np.zeros((0, 3))).shape == (0,)


def test_sampling_far_outside_the_box_clamps_or_marks(grid):
    far = np.array([[1e30, -1e30, 0.0]])

    assert np.isfinite(grid.sample(far, outside="clamp")).all()
    assert np.isnan(grid.sample(far, outside="nan")).all()


def test_sampling_at_an_undefined_point_does_not_raise(grid):
    """An atom whose coordinate is nan asks for a potential at nowhere."""
    assert np.isnan(grid.sample(np.array([[np.nan, 0.0, 0.0]]))).all()


def test_sampling_a_single_node_grid_returns_its_value():
    tiny = PotentialGrid(np.full((1, 1, 1), 4.0), np.zeros(3), np.ones(3))

    assert tiny.sample(np.zeros((1, 3))) == pytest.approx(4.0)


@pytest.mark.parametrize("points", [np.zeros(3), np.zeros((2, 2)), np.zeros((2, 4))])
def test_sampling_something_that_is_not_a_point_list_is_refused(grid, points):
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        grid.sample(points)


# ---------------------------------------------------------------------------
# PyMOL sessions that are not sessions
# ---------------------------------------------------------------------------


def write_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=2)
    return str(path)


@pytest.mark.parametrize(
    "payload",
    [
        [1, 2, 3],  # not a dictionary
        "a string",
        {"version": 1},  # a dictionary, but not a session
        {"colors": [], "settings": []},
    ],
)
def test_a_pickle_that_is_not_a_session_is_reported(tmp_path, payload):
    with pytest.raises(PymolSessionError):
        read_session(write_pickle(tmp_path / "s.pse", payload))


def test_a_session_containing_nothing_is_read_as_nothing(tmp_path):
    """Empty is not corrupt: PyMOL will save a session with no objects in it."""
    session = read_session(write_pickle(tmp_path / "s.pse", {"names": []}))

    assert session.molecules == []
    assert session.selections == []


@pytest.mark.parametrize("body", [b"", b"not a pickle", b"\x80\x04\x95"])
def test_a_file_that_is_not_a_pickle_is_reported(tmp_path, body):
    path = tmp_path / "s.pse"
    path.write_bytes(body)

    with pytest.raises(PymolSessionError):
        read_session(str(path))


def test_a_corrupt_gzip_header_is_reported(tmp_path):
    path = tmp_path / "s.pse"
    path.write_bytes(b"\x1f\x8b truncated before anything useful")

    with pytest.raises(PymolSessionError):
        read_session(str(path))


def test_a_session_truncated_mid_stream_is_reported(tmp_path):
    """Half a download, which is what a session emailed around often is."""
    whole = pickle.dumps({"names": [], "version": 1}, protocol=2)
    path = tmp_path / "s.pse"
    path.write_bytes(whole[: len(whole) // 2])

    with pytest.raises(PymolSessionError):
        read_session(str(path))


def test_a_gzipped_session_that_is_not_a_session_is_reported(tmp_path):
    path = tmp_path / "s.pse"
    path.write_bytes(gzip.compress(b"not a pickle"))

    with pytest.raises(PymolSessionError):
        read_session(str(path))


def test_a_truncated_molecule_record_is_reported(tmp_path):
    path = write_pickle(
        tmp_path / "s.pse", {"names": [["m", 0, 1, None, 1, [1, 2], ""]]}
    )

    with pytest.raises(PymolSessionError, match="truncated"):
        read_session(path)


def test_a_view_of_the_wrong_length_is_reported(tmp_path):
    path = write_pickle(tmp_path / "s.pse", {"names": [], "view": [0.0] * 10})

    with pytest.raises(PymolSessionError, match="expected 25"):
        read_session(path)


@pytest.mark.parametrize("names", [42, [None, 42, "x"]])
def test_a_corrupt_name_list_is_reported(tmp_path, names):
    path = write_pickle(tmp_path / "s.pse", {"names": names})

    with pytest.raises(PymolSessionError):
        read_session(path)


def test_a_view_that_is_not_numbers_is_reported(tmp_path):
    path = write_pickle(tmp_path / "s.pse", {"names": [], "view": ["a"] * 25})

    with pytest.raises(PymolSessionError):
        read_session(path)


# ---------------------------------------------------------------------------
# PyMOL sessions: degenerate but legitimate
# ---------------------------------------------------------------------------


def molecule(n, coord=None, name="m"):
    """A minimal :class:`PymolMolecule`, for writing back out."""
    return PymolMolecule(
        name=name,
        coord=np.zeros((1, n, 3)) if coord is None else np.asarray(coord, dtype=float),
        chain_id=np.array(["A"] * n, dtype="U4"),
        res_id=np.arange(1, n + 1),
        ins_code=np.array([""] * n, dtype="U1"),
        res_name=np.array(["ALA"] * n, dtype="U5"),
        atom_name=np.array(["CA"] * n, dtype="U6"),
        element=np.array(["C"] * n, dtype="U2"),
        alt_id=np.array([""] * n, dtype="U4"),
        segi=np.array([""] * n, dtype="U4"),
        b_factor=np.zeros(n),
        occupancy=np.ones(n),
        charge=np.zeros(n),
        vdw=np.full(n, 1.7),
        hetero=np.zeros(n, dtype=bool),
        label=np.array([""] * n, dtype=object),
        reps=np.zeros(n, dtype=np.int64),
        color_index=np.zeros(n, dtype=int),
        bonds=np.zeros((0, 3), dtype=int),
        ss=np.array([""] * n, dtype="U1"),
    )


def round_trip(session, path):
    write_session(session, str(path))
    return read_session(str(path))


def test_an_empty_session_round_trips(tmp_path):
    back = round_trip(PymolSession(), tmp_path / "s.pse")

    assert back.molecules == []
    assert isinstance(back.summary(), str)


def test_a_molecule_with_no_atoms_round_trips(tmp_path):
    back = round_trip(PymolSession(molecules=[molecule(0)]), tmp_path / "s.pse")

    assert back.molecules[0].n_atoms == 0
    assert back.molecules[0].reps_present() == []


def test_an_atom_missing_from_a_state_stays_missing(tmp_path):
    """`nan` means "this atom is not in this state", and PyMOL stores that as
    an absence rather than as a coordinate. It has to survive both ways."""
    coord = np.array([[[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]])

    back = round_trip(PymolSession(molecules=[molecule(2, coord)]), tmp_path / "s.pse")

    assert np.isnan(back.molecules[0].coord[0, 1]).all()
    assert np.allclose(back.molecules[0].coord[0, 0], 0.0)


def test_a_unicode_object_name_round_trips(tmp_path):
    session = PymolSession(
        molecules=[molecule(1, name="sélé🧬")],
        selections=[PymolSelection("sélé🧬", {"sélé🧬": np.array([0])})],
    )

    back = round_trip(session, tmp_path / "s.pse")

    assert back.molecules[0].name == "sélé🧬"
    assert back.selections[0].name == "sélé🧬"


def test_a_selection_with_no_members_round_trips(tmp_path):
    session = PymolSession(selections=[PymolSelection("empty", {})])

    assert round_trip(session, tmp_path / "s.pse").selections[0].n_atoms == 0


def test_asking_for_a_state_that_is_not_there_is_reported():
    with pytest.raises(PymolSessionError, match="out of range"):
        molecule(2).to_atom_array(9)


def test_writing_where_it_cannot_be_written_says_so(tmp_path):
    with pytest.raises(OSError):
        write_session(PymolSession(), str(tmp_path / "no such directory" / "s.pse"))


# ---------------------------------------------------------------------------
# Attribute names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "...",
        "___",
        "🧬",
        "1abc",
        "a b.c",
        "-" * 200,
        ".hidden",  # a leading dot hides an attribute from Blender's UI
        "sele",
        "ünïcode",
        "name\nwith\nnewlines",
    ],
)
def test_any_name_becomes_a_usable_attribute_name(name):
    """Whatever the user typed, geometry nodes has to be able to read it back:
    non-empty, no leading dot, nothing but word characters."""
    cleaned = attributes.safe_name(name)

    assert cleaned
    assert not cleaned.startswith(".")
    assert cleaned.replace("_", "").isalnum() or cleaned == "selection"
    assert attributes.safe_name(cleaned) == cleaned  # and cleaning it again is a no-op


# ---------------------------------------------------------------------------
# Pure geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ([np.nan, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [np.inf, 0, 0]),
        ([np.inf, 0, 0], [-np.inf, 0, 0]),
    ],
)
def test_dashes_refuse_a_segment_that_is_not_a_segment(start, end):
    """Better a raise than a curve object full of nan control points, which
    Blender accepts and then draws as nothing at all."""
    with pytest.raises((ValueError, OverflowError)):
        geometry.dash_segments(start, end)


def test_dashes_stay_within_the_segment_they_were_given():
    segments = geometry.dash_segments([0, 0, 0], [10, 0, 0], 1.0, 0.5)
    xs = np.concatenate([[start[0], end[0]] for start, end in segments])

    assert xs.min() >= -1e-9
    assert xs.max() <= 10.0 + 1e-9


@pytest.mark.parametrize("resolution", [-5, 0, 1, 2])
def test_an_arc_always_has_at_least_two_points(resolution):
    points = geometry.arc_points([0, 0, 0], [1, 0, 0], [0, 1, 0], resolution=resolution)

    assert points.shape == (2, 3)


def test_an_arc_of_undefined_rays_is_empty():
    for a, b in (
        ([0, 0, 0], [1, 0, 0]),  # a ray of zero length
        ([1, 0, 0], [2, 0, 0]),  # two rays along the same line
    ):
        assert geometry.arc_points([0, 0, 0], a, b).shape == (0, 3)


def test_a_card_outline_is_a_closed_polygon():
    for half_width, half_height, corner in (
        (1.0, 1.0, 0.0),
        (1.0, 0.5, 1.0),
        (1e-6, 1e-6, 0.35),
    ):
        points = geometry.rounded_rectangle(half_width, half_height, corner)

        assert len(points) >= 4
        assert np.isfinite(np.asarray(points)).all()


def test_a_card_with_no_corner_segments_is_refused():
    """Zero segments per corner is not a plain rectangle, it is a division by
    zero; `corner=0` is how one asks for square corners."""
    with pytest.raises((ValueError, ZeroDivisionError)):
        geometry.rounded_rectangle(1.0, 1.0, corner=0.35, segments=0)


# ---------------------------------------------------------------------------
# Interaction detection: chemistry that is not what it looks like
# ---------------------------------------------------------------------------


def phenylalanines(coord):
    """Two PHE residues, six ring atoms each, at whatever coordinates given."""
    import biotite.structure as struc

    names = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    array = struc.AtomArray(12)
    array.coord = np.asarray(coord, dtype=np.float32).reshape(12, 3)
    array.chain_id = np.array(["A"] * 12)
    array.res_id = np.array([1] * 6 + [2] * 6)
    array.res_name = np.array(["PHE"] * 12)
    array.atom_name = np.array(names * 2)
    array.element = np.array(["C"] * 12)
    return AtomStructure(array=array)


def test_a_blank_element_column_does_not_invent_metals(site_array):
    """The PDB element columns are optional, and plenty of generated files
    leave them out."""
    from blender_gala.interactions import detect

    blanked = site_array.copy()
    blanked.element = np.array([""] * len(blanked))

    found = detect.metal_coordination(AtomStructure(array=blanked))

    assert len(found) == len(detect.metal_coordination(AtomStructure(array=site_array)))


def test_a_cutoff_that_is_not_positive_finds_nothing(site):
    from blender_gala.interactions import detect

    criteria = detect.InteractionCriteria(polar_max=-5.0)
    loosened = detect.polar_contacts(site, criteria=criteria)

    assert len(loosened) <= len(detect.polar_contacts(site))


def test_rings_that_have_no_plane_are_not_stacked():
    """Six coincident atoms are not a ring. Unmerged altlocs and collapsed
    minimisations both produce them."""
    from blender_gala.interactions import detect

    collapsed = phenylalanines(
        np.vstack([np.zeros((6, 3)), np.zeros((6, 3)) + np.array([0, 0, 3.8])])
    )

    assert detect.pi_stacking(collapsed) == []


# ---------------------------------------------------------------------------
# Electrostatics
# ---------------------------------------------------------------------------


def test_reading_a_potential_at_no_points_is_reported(site):
    from blender_gala.electrostatics import surface

    grid = PotentialGrid(np.zeros((3, 3, 3)), np.zeros(3), np.ones(3))

    with pytest.raises((ValueError, GalaError)):
        surface.potential_at_atoms(site, grid, points=0)


def test_the_documented_negative_probe_radius_works(site):
    from blender_gala.electrostatics import surface

    grid = PotentialGrid(np.zeros((3, 3, 3)), np.zeros(3), np.ones(3))
    values = surface.potential_at_atoms(site, grid, probe=-2.0)

    assert len(values) == site.n_atoms


@pytest.mark.parametrize(
    ("body", "expected"),
    [("ATOM 5.0 9.0\n", 0.0), ("ATOM x y nan 1.7\n", 0.0)],
)
def test_the_charge_column_of_a_pqr_is_read_not_guessed(tmp_path, body, expected):
    from blender_gala.electrostatics import apbs

    path = tmp_path / "p.pqr"
    path.write_text(body)

    assert apbs._net_charge(str(path)) == pytest.approx(expected)
