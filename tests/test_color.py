"""Tests for colormaps and data-driven colouring."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from blender_gala.color import coloring, colormaps

# ---------------------------------------------------------------------------
# Colour space
# ---------------------------------------------------------------------------


def test_srgb_linear_round_trip():
    values = np.linspace(0.0, 1.0, 21)
    assert np.allclose(
        colormaps.linear_to_srgb(colormaps.srgb_to_linear(values)), values
    )


def test_srgb_endpoints_are_fixed():
    assert colormaps.srgb_to_linear(np.array([0.0, 1.0])).tolist() == [0.0, 1.0]


def test_mid_grey_is_darker_in_linear():
    """0.5 sRGB is about 0.21 linear; getting this wrong washes out every figure."""
    assert float(colormaps.srgb_to_linear(np.array([0.5]))[0]) == pytest.approx(
        0.2140, abs=1e-3
    )


def test_hex_to_rgb():
    assert colormaps.hex_to_rgb("#000000").tolist() == [0.0, 0.0, 0.0]
    assert colormaps.hex_to_rgb("ffffff").tolist() == [1.0, 1.0, 1.0]
    red = colormaps.hex_to_rgb("#ff0000")
    assert red[0] == pytest.approx(1.0)
    assert red[1] == 0.0


def test_hex_to_rgb_can_skip_the_linear_conversion():
    assert colormaps.hex_to_rgb("#808080", linear=False)[0] == pytest.approx(
        128 / 255, abs=1e-6
    )


@pytest.mark.parametrize("bad", ["#12345", "not a colour", "#gggggg"])
def test_bad_hex_raises(bad):
    with pytest.raises(ValueError):
        colormaps.hex_to_rgb(bad)


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------


def test_every_colormap_samples_cleanly():
    positions = np.linspace(0, 1, 32)
    for name in colormaps.list_colormaps():
        sampled = colormaps.sample(name, positions)
        assert sampled.shape == (32, 3)
        assert np.isfinite(sampled).all()
        assert (sampled >= 0.0).all() and (sampled <= 1.0).all()


def test_sample_endpoints_match_the_stops():
    first = colormaps.sample("viridis", [0.0])[0]
    last = colormaps.sample("viridis", [1.0])[0]
    assert np.allclose(first, colormaps.hex_to_rgb("#440154"))
    assert np.allclose(last, colormaps.hex_to_rgb("#fde725"))


def test_sample_clamps_out_of_range_values():
    below = colormaps.sample("viridis", [-5.0])[0]
    above = colormaps.sample("viridis", [5.0])[0]
    assert np.allclose(below, colormaps.sample("viridis", [0.0])[0])
    assert np.allclose(above, colormaps.sample("viridis", [1.0])[0])


def test_reverse_flips_the_map():
    forward = colormaps.sample("viridis", [0.0])[0]
    reversed_ = colormaps.sample("viridis", [1.0], reverse=True)[0]
    assert np.allclose(forward, reversed_)


def test_unknown_colormap_raises():
    with pytest.raises(ValueError, match="unknown colormap"):
        colormaps.sample("chartreuse", [0.5])


# ---------------------------------------------------------------------------
# AlphaFold pLDDT
# ---------------------------------------------------------------------------


def test_plddt_bands_match_the_alphafold_database(plddt_structure):
    result = coloring.color_by_plddt(plddt_structure, write=False)

    b_factors = np.asarray(plddt_structure.array.b_factor)
    expected = {
        95.0: "#0053d6",
        92.0: "#0053d6",  # very high
        85.0: "#65cbf3",
        75.0: "#65cbf3",  # confident
        65.0: "#ffdb13",
        55.0: "#ffdb13",  # low
        45.0: "#ff7d45",
        30.0: "#ff7d45",  # very low
    }
    for value, hex_colour in expected.items():
        index = int(np.flatnonzero(b_factors == value)[0])
        assert np.allclose(
            result.colors[index, :3], colormaps.hex_to_rgb(hex_colour)
        ), f"pLDDT {value}"


def test_plddt_band_boundaries_are_inclusive_below(plddt_structure):
    """A pLDDT of exactly 90 is 'very high', matching the AFDB convention."""
    array = plddt_structure.array.copy()
    array.b_factor = np.full(len(array), 90.0)
    from blender_gala.core.entity import AtomStructure

    result = coloring.color_by_plddt(AtomStructure(array=array), write=False)
    assert np.allclose(result.colors[0, :3], colormaps.hex_to_rgb("#0053d6"))


def test_plddt_autoscales_a_zero_to_one_column(plddt_structure):
    array = plddt_structure.array.copy()
    array.b_factor = np.asarray(array.b_factor) / 100.0
    from blender_gala.core.entity import AtomStructure

    scaled = coloring.color_by_plddt(AtomStructure(array=array), write=False)
    original = coloring.color_by_plddt(plddt_structure, write=False)
    assert np.allclose(scaled.colors, original.colors)


def test_plddt_continuous_mode_differs_from_banded(plddt_structure):
    banded = coloring.color_by_plddt(plddt_structure, mode="banded", write=False)
    smooth = coloring.color_by_plddt(plddt_structure, mode="continuous", write=False)
    assert not np.allclose(banded.colors, smooth.colors)


def test_plddt_rejects_unknown_mode(plddt_structure):
    with pytest.raises(ValueError, match="banded"):
        coloring.color_by_plddt(plddt_structure, mode="stripes", write=False)


def test_plddt_legend_has_four_bands():
    legend = coloring.plddt_legend()
    assert len(legend) == 4
    assert "Very high" in legend[0][0]
    assert "Very low" in legend[-1][0]


# ---------------------------------------------------------------------------
# Generic colouring
# ---------------------------------------------------------------------------


def test_color_by_per_atom_array(site):
    values = np.arange(site.n_atoms, dtype=float)
    result = coloring.color_by_attribute(site, values, write=False)
    assert result.vmin == 0.0
    assert result.vmax == float(site.n_atoms - 1)
    assert not np.allclose(result.colors[0, :3], result.colors[-1, :3])


def test_color_by_residue_mapping(site):
    result = coloring.color_by_attribute(
        site, {1: 0.0, 2: 1.0}, missing=(0.5, 0.5, 0.5), write=False
    )
    ser = site.indices("chain A and resi 1")
    asp = site.indices("chain A and resi 2")
    phe = site.indices("resi 4")

    assert np.allclose(result.colors[ser[0], :3], result.colors[ser[-1], :3])
    assert not np.allclose(result.colors[ser[0], :3], result.colors[asp[0], :3])
    # Residue 4 has no entry in the mapping, so it gets the 'missing' colour.
    assert np.allclose(result.colors[phe[0], :3], [0.5, 0.5, 0.5])


def test_residue_only_mapping_collides_across_chains(site):
    """Why ``chain_column`` matters: residue 1 exists in four chains here."""
    result = coloring.color_by_attribute(
        site, {1: 0.0}, missing=(0.5, 0.5, 0.5), write=False
    )
    ser = site.indices("chain A and resi 1")[0]
    lig = site.indices("resn LIG")[0]
    assert np.allclose(result.colors[ser, :3], result.colors[lig, :3])


def test_color_by_chain_and_residue_mapping(site):
    result = coloring.color_by_attribute(
        site, {("A", 1): 0.0, ("A", 2): 1.0}, missing=(0.5, 0.5, 0.5), write=False
    )
    ser = site.indices("chain A and resi 1")[0]
    lig = site.indices("resn LIG")[0]
    assert result.n_colored > 0
    # A chain-qualified key does not leak onto the ligand's residue 1.
    assert np.allclose(result.colors[lig, :3], [0.5, 0.5, 0.5])
    assert not np.allclose(result.colors[ser, :3], [0.5, 0.5, 0.5])


def test_color_by_callable(site):
    result = coloring.color_by_attribute(site, lambda i: float(i), write=False)
    assert result.vmax == float(site.n_atoms - 1)


def test_explicit_range_makes_structures_comparable(site):
    values = np.arange(site.n_atoms, dtype=float)
    result = coloring.color_by_attribute(
        site, values, vmin=0.0, vmax=1000.0, write=False
    )
    assert result.vmin == 0.0
    assert result.vmax == 1000.0
    # Everything is near the bottom of the ramp when the range is much wider.
    assert np.allclose(result.colors[0, :3], result.colors[-1, :3], atol=0.1)


def test_constant_values_do_not_divide_by_zero(site):
    result = coloring.color_by_attribute(site, np.ones(site.n_atoms), write=False)
    assert np.isfinite(result.colors).all()


def test_wrong_length_values_raise(site):
    with pytest.raises(ValueError, match="atoms"):
        coloring.color_by_attribute(site, np.zeros(3), write=False)


def test_color_by_selection_is_categorical(site):
    result = coloring.color_by_selection(
        site, {"protein": "#3366cc", "resn LIG": "#cc6633"}, write=False
    )
    protein = site.indices("protein")[0]
    ligand = site.indices("resn LIG")[0]
    assert np.allclose(result.colors[protein, :3], colormaps.hex_to_rgb("#3366cc"))
    assert np.allclose(result.colors[ligand, :3], colormaps.hex_to_rgb("#cc6633"))
    assert len(result.legend) == 2


def test_color_by_selection_later_entries_win(site):
    result = coloring.color_by_selection(
        site, {"all": "#000000", "resn LIG": "#ffffff"}, write=False
    )
    ligand = site.indices("resn LIG")[0]
    assert np.allclose(result.colors[ligand, :3], [1.0, 1.0, 1.0])


def test_color_by_bfactor(site):
    result = coloring.color_by_bfactor(site, write=False)
    assert result.vmin < result.vmax


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


@pytest.fixture
def scores_csv(tmp_path):
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chain", "res_id", "conservation"])
        writer.writerow(["A", 1, 0.1])
        writer.writerow(["A", 2, 0.9])
        writer.writerow(["A", 3, 0.5])
    return str(path)


def test_color_from_csv(site, scores_csv):
    result = coloring.color_from_csv(
        site,
        scores_csv,
        value_column="conservation",
        chain_column="chain",
        write=False,
    )
    assert result.vmin == pytest.approx(0.1)
    assert result.vmax == pytest.approx(0.9)


def test_color_from_csv_reports_missing_columns(site, scores_csv):
    with pytest.raises(KeyError, match="no column"):
        coloring.color_from_csv(
            site, scores_csv, value_column="nonexistent", write=False
        )


def test_color_from_csv_without_chain_column(site, scores_csv):
    result = coloring.color_from_csv(
        site, scores_csv, value_column="conservation", write=False
    )
    assert result.n_colored > 0
