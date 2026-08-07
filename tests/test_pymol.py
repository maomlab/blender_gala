"""Tests for reading and writing PyMOL sessions.

``tests/data/session.pse`` was written by a real PyMOL — see
``scripts/make_pymol_fixture.py``. That matters: a fixture produced by Gala's
own writer would only prove the writer and the reader agree with each other,
not that either agrees with PyMOL. Everything here except the two marked
groups runs under a plain interpreter, because a session is data.
"""

from __future__ import annotations

import gzip
import os
import pickle

import numpy as np
import pytest

from blender_gala.pymol import palette
from blender_gala.pymol import view as view_module
from blender_gala.pymol.session import (
    REPS,
    PymolMeasurement,
    PymolMolecule,
    PymolSelection,
    PymolSession,
    PymolSessionError,
    PymolView,
    read_session,
    write_session,
)
from tests.conftest import DATA_DIR, requires_bpy, requires_mn

FIXTURE = os.path.join(DATA_DIR, "session.pse")


@pytest.fixture(scope="module")
def session():
    """The committed PyMOL session."""
    return read_session(FIXTURE)


# ---------------------------------------------------------------------------
# Reading what PyMOL wrote
# ---------------------------------------------------------------------------


def test_reads_every_object(session):
    assert [molecule.name for molecule in session.molecules] == ["site", "shell"]
    assert session.find("site").n_atoms == 73
    assert session.find("shell").n_atoms == 61
    assert session.find("nothing") is None


def test_reads_chemistry(session):
    site = session.find("site")
    assert site.chain_id[0] == "A"
    assert site.res_id[0] == 1
    assert site.res_name[0] == "SER"
    assert site.atom_name[0] == "N"
    assert site.element[0] == "N"
    assert site.b_factor[0] == pytest.approx(30.0)
    assert site.occupancy[0] == pytest.approx(1.0)
    # vdW radii come from PyMOL rather than from a table of our own.
    assert site.vdw[0] == pytest.approx(1.55, abs=1e-3)
    assert site.hetero.any()


def test_reads_coordinates_in_angstrom(session):
    site = session.find("site")
    assert site.coord.shape == (1, 73, 3)
    assert np.isfinite(site.coord).all()
    assert site.coord[0, 0] == pytest.approx([-2.5, 1.2, 0.0], abs=1e-4)


def test_reads_bonds(session):
    bonds = session.find("site").bonds
    assert bonds.shape[1] == 3
    assert len(bonds) == 79
    assert bonds[:, :2].max() < 73


def test_reads_representations(session):
    site = session.find("site")
    assert set(site.reps_present()) == {"cartoon", "sticks", "spheres", "labels"}
    assert session.find("shell").reps_present() == ["surface"]

    cartoon = site.rep_mask("cartoon")
    assert cartoon.sum() == 61  # the polymer, and only the polymer
    assert not cartoon[site.res_name == "HOH"].any()


def test_rep_mask_rejects_an_unknown_representation(session):
    with pytest.raises(ValueError, match="unknown representation"):
        session.find("site").rep_mask("wireframe")


def test_reads_secondary_structure(session):
    site = session.find("site")
    assert len(site.ss) == site.n_atoms
    assert set(site.ss.tolist()) <= {"", "H", "S", "L"}


def test_reads_labels(session):
    labels = [text for text in session.find("site").label if text]
    assert labels == ["first CA"]


def test_reads_named_selections(session):
    pocket = next(s for s in session.selections if s.name == "pocket")
    assert pocket.n_atoms == 36
    assert set(pocket.members) == {"site"}
    assert pocket.members["site"].max() < 73


def test_reads_groups_and_membership(session):
    assert "assembly" in session.groups
    assert session.find("shell").group == "assembly"
    assert session.find("site").group == ""


def test_reads_object_transform_and_settings(session):
    shell = session.find("shell")
    assert shell.matrix is not None
    # The fixture translates it 12 A along +X and leaves the rotation alone.
    assert shell.matrix[:3, 3] == pytest.approx([12.0, 0.0, 0.0], abs=1e-3)
    assert shell.matrix[:3, :3] == pytest.approx(np.eye(3), abs=1e-6)
    assert shell.settings["transparency"] == pytest.approx(0.5, abs=1e-6)
    assert session.find("site").matrix is None


def test_reads_measurements_of_each_kind(session):
    by_name = {m.name: m for m in session.measurements}
    assert by_name["d1"].kind == "distance"
    assert by_name["a1"].kind == "angle"
    assert by_name["t1"].kind == "dihedral"
    assert by_name["d1"].points.shape == (1, 2, 3)
    assert by_name["a1"].points.shape == (1, 3, 3)
    assert by_name["t1"].points.shape == (1, 4, 3)


def test_recomputes_measurement_values(session):
    by_name = {m.name: m for m in session.measurements}
    distance = by_name["d1"]
    expected = np.linalg.norm(distance.points[0][1] - distance.points[0][0])
    assert distance.values[0] == pytest.approx(expected)
    assert 0.0 <= by_name["a1"].values[0] <= 180.0
    assert -180.0 <= by_name["t1"].values[0] <= 180.0


def test_reads_the_view(session):
    assert not session.view.orthoscopic
    assert session.view.field_of_view == pytest.approx(20.0)
    assert session.view.distance == pytest.approx(80.0)
    assert session.view.rotation[0] == pytest.approx([0.8, 0.0, 0.6], abs=1e-6)
    assert session.view.near == pytest.approx(40.0)
    assert session.view.far == pytest.approx(120.0)


def test_summary_names_what_was_read(session):
    text = session.summary()
    assert "site: 73 atoms" in text
    assert "cartoon" in text
    assert "pocket" in text


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


def test_resolves_builtin_colours(session):
    site = session.find("site")
    # The polymer is skyblue, except residue 1 which the fixture recolours.
    polymer = site.rep_mask("cartoon") & (site.res_id != 1)
    index = int(site.color_index[polymer][0])
    assert palette.name_for_index(index) == "skyblue"
    assert session.color(index) == pytest.approx((0.2, 0.502, 0.8), abs=0.01)


def test_resolves_a_session_defined_colour(session):
    # The fixture defines gala_teal, which is numbered past the built-in table.
    assert session.colors
    index, rgb = next(iter(session.colors.items()))
    assert index >= palette.COUNT
    assert session.color(index) == pytest.approx(rgb)


def test_falls_back_to_the_element_colour(session):
    # A negative index means "by element"; carbon is PyMOL's green.
    carbon = session.color(-4, element="C")
    assert carbon == pytest.approx(
        palette.rgb_for_index(palette.index_for_name("carbon"))
    )
    assert session.color(-4, element="Xx") == pytest.approx((0.8, 0.8, 0.8))


def test_atom_colors_are_one_row_per_atom(session):
    site = session.find("site")
    colors = session.atom_colors(site)
    assert colors.shape == (site.n_atoms, 4)
    assert (colors[:, 3] == 1.0).all()
    assert len(np.unique(colors, axis=0)) > 1


def test_palette_lookups():
    assert palette.name_for_index(0) == "white"
    assert palette.rgb_for_index(0) == pytest.approx((1.0, 1.0, 1.0))
    assert palette.index_for_name("SkyBlue") == palette.index_for_name("skyblue")
    assert palette.index_for_name("no such colour") is None
    assert palette.rgb_for_index(-1) is None
    assert palette.rgb_for_index(palette.COUNT) is None
    assert palette.name_for_index(-4) == "atomic"


# ---------------------------------------------------------------------------
# Refusing what cannot be read faithfully
# ---------------------------------------------------------------------------


def test_refuses_a_pickle_that_names_anything_else(tmp_path):
    """A .pse is a pickle, so the reader must not import what it is told to."""
    path = tmp_path / "hostile.pse"
    path.write_bytes(pickle.dumps({"names": [None], "evil": os.system}))
    with pytest.raises(PymolSessionError, match="will not import"):
        read_session(str(path))


def test_refuses_a_binary_dump(tmp_path):
    """pse_binary_dump stores C structs whose layout is not ours to guess."""
    molecule = [
        [1, "x", 0, 0, [0.0] * 3, [0.0] * 3, 0, 0, None, 1, 0, [0.0] * 16, 0, None],
        1,
        0,
        1,
        [[1, 1, b"\x00\x00\x00\x00", [0], [0], "", [None]]],
        None,
        [181, b"\x00"],
        [181, b"\x00"],
    ]
    data = {
        "version": 3000000,
        "names": [None, ["x", 0, 1, None, 1, molecule, ""]],
        "view": PymolView.default().to_list(),
    }
    path = tmp_path / "binary.pse"
    path.write_bytes(pickle.dumps(data, protocol=2))
    with pytest.raises(PymolSessionError, match="pse_binary_dump"):
        read_session(str(path))


def test_refuses_something_that_is_not_a_session(tmp_path):
    path = tmp_path / "not.pse"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(PymolSessionError, match="not a session dictionary"):
        read_session(str(path))


def test_refuses_an_unreadable_file(tmp_path):
    path = tmp_path / "junk.pse"
    path.write_bytes(b"this is not a pickle at all")
    with pytest.raises(PymolSessionError, match="not a readable PyMOL session"):
        read_session(str(path))


def test_reads_a_gzipped_session(tmp_path):
    path = tmp_path / "session.pse.gz"
    with open(FIXTURE, "rb") as handle, gzip.open(path, "wb") as out:
        out.write(handle.read())
    assert read_session(str(path)).find("site").n_atoms == 73


def test_view_rejects_a_short_vector():
    with pytest.raises(PymolSessionError, match="expected 25"):
        PymolView.from_list([0.0] * 18)


# ---------------------------------------------------------------------------
# Writing, and the round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_everything(session, tmp_path):
    path = str(tmp_path / "out.pse")
    write_session(session, path)
    again = read_session(path)

    assert [m.name for m in again.molecules] == [m.name for m in session.molecules]
    for before, after in zip(session.molecules, again.molecules, strict=True):
        assert after.n_atoms == before.n_atoms
        assert after.coord == pytest.approx(before.coord, abs=1e-4)
        assert list(after.res_name) == list(before.res_name)
        assert list(after.atom_name) == list(before.atom_name)
        assert list(after.element) == list(before.element)
        assert list(after.chain_id) == list(before.chain_id)
        assert after.b_factor == pytest.approx(before.b_factor)
        assert list(after.color_index) == list(before.color_index)
        assert list(after.reps) == list(before.reps)
        assert list(after.ss) == list(before.ss)
        assert list(after.label) == list(before.label)
        assert after.bonds.tolist() == before.bonds.tolist()

    assert again.view.to_list() == pytest.approx(session.view.to_list())
    assert again.groups == session.groups
    assert {s.name: s.n_atoms for s in again.selections} == {
        s.name: s.n_atoms for s in session.selections
    }
    assert {m.name: m.kind for m in again.measurements} == {
        m.name: m.kind for m in session.measurements
    }


def test_round_trip_preserves_an_object_transform(session, tmp_path):
    path = str(tmp_path / "out.pse")
    write_session(session, path)
    shell = read_session(path).find("shell")
    assert shell.matrix == pytest.approx(session.find("shell").matrix, abs=1e-5)


def test_round_trip_preserves_measurement_values(session, tmp_path):
    path = str(tmp_path / "out.pse")
    write_session(session, path)
    before = {m.name: m.values for m in session.measurements}
    after = {m.name: m.values for m in read_session(path).measurements}
    for name, values in before.items():
        assert after[name] == pytest.approx(values, abs=1e-4)


def test_writes_a_gzipped_session(session, tmp_path):
    path = str(tmp_path / "out.pse.gz")
    write_session(session, path)
    with open(path, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"
    assert read_session(path).find("site").n_atoms == 73


def test_written_atoms_carry_pymol_class_flags(session, tmp_path):
    """Without these, `show cartoon` in PyMOL finds no polymer."""
    from blender_gala.pymol.session import FLAG_GUIDE, FLAG_POLYMER, _atom_flags

    assert _atom_flags("SER", "N", False) == FLAG_POLYMER
    assert _atom_flags("SER", "CA", False) == FLAG_POLYMER | FLAG_GUIDE
    assert _atom_flags("HOH", "O", True) != FLAG_POLYMER
    assert _atom_flags("LIG", "C1", True) != _atom_flags("ZN", "ZN", True)


def test_writes_a_state_an_atom_is_missing_from(tmp_path):
    """An atom absent from a state must not be written at the origin."""
    molecule = _minimal_molecule()
    molecule.coord = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [np.nan] * 3]]
    )
    session = PymolSession(molecules=[molecule])
    path = str(tmp_path / "states.pse")
    write_session(session, path)

    back = read_session(path).molecules[0]
    assert back.n_states == 2
    assert np.isnan(back.coord[1, 1]).all()
    assert back.coord[1, 0] == pytest.approx([2.0, 0.0, 0.0])


def test_writing_an_unknown_measurement_kind_is_an_error(tmp_path):
    session = PymolSession(
        measurements=[
            PymolMeasurement(name="m", kind="volume", points=np.zeros((1, 2, 3)))
        ]
    )
    with pytest.raises(ValueError, match="cannot write a 'volume'"):
        write_session(session, str(tmp_path / "bad.pse"))


def test_writes_a_selection(tmp_path):
    session = PymolSession(
        molecules=[_minimal_molecule()],
        selections=[PymolSelection(name="picked", members={"m": np.array([1])})],
    )
    path = str(tmp_path / "sele.pse")
    write_session(session, path)
    picked = read_session(path).selections[0]
    assert picked.name == "picked"
    assert list(picked.members["m"]) == [1]


def _minimal_molecule() -> PymolMolecule:
    """Two carbons, enough to write and read back."""
    n = 2
    return PymolMolecule(
        name="m",
        coord=np.zeros((1, n, 3)),
        chain_id=np.array(["A"] * n),
        res_id=np.array([1, 2]),
        ins_code=np.array([""] * n),
        res_name=np.array(["ALA"] * n),
        atom_name=np.array(["CA", "CB"]),
        element=np.array(["C"] * n),
        alt_id=np.array([""] * n),
        segi=np.array([""] * n),
        b_factor=np.zeros(n),
        occupancy=np.ones(n),
        charge=np.zeros(n),
        vdw=np.full(n, 1.7),
        hetero=np.zeros(n, dtype=bool),
        label=np.array(["", ""], dtype=object),
        reps=np.zeros(n, dtype=np.int64),
        color_index=np.zeros(n, dtype=int),
        bonds=np.array([[0, 1, 1]]),
    )


# ---------------------------------------------------------------------------
# The camera
# ---------------------------------------------------------------------------


def test_identity_view_puts_the_camera_on_plus_z():
    """The convention PyMOL was asked about directly, kept as a test.

    Landmarks at +X, +Y and +Z were rendered under an identity view and under
    a 90 degree rotation about Y; only a matrix whose *columns* are the camera
    axes in world space predicts where they landed.
    """
    view = PymolView(
        rotation=np.eye(3),
        position=np.array([0.0, 0.0, -80.0]),
        origin=np.zeros(3),
        near=40.0,
        far=120.0,
        field_of_view=20.0,
        orthoscopic=False,
    )
    matrix = view_module.camera_matrix(view)
    assert matrix[:3, 3] == pytest.approx([0.0, 0.0, 80.0])
    assert matrix[:3, :3] == pytest.approx(np.eye(3))


def test_a_y_rotation_puts_the_camera_on_plus_x():
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    view = PymolView(
        rotation=rotation,
        position=np.array([0.0, 0.0, -120.0]),
        origin=np.zeros(3),
        near=20.0,
        far=220.0,
        field_of_view=20.0,
        orthoscopic=False,
    )
    assert view_module.camera_matrix(view)[:3, 3] == pytest.approx([120.0, 0.0, 0.0])


def test_camera_matrix_round_trips_through_view_from_matrix(session):
    matrix = view_module.camera_matrix(session.view)
    again = view_module.view_from_matrix(
        matrix,
        origin=session.view.origin,
        field_of_view=session.view.field_of_view,
        orthoscopic=session.view.orthoscopic,
        near=session.view.near,
        far=session.view.far,
    )
    # PyMOL stores the view as single precision, so the camera position comes
    # back a couple of parts in ten million out.
    assert again.to_list() == pytest.approx(session.view.to_list(), abs=1e-4)


def test_view_round_trips_through_its_own_numbers(session):
    assert PymolView.from_list(session.view.to_list()).to_list() == pytest.approx(
        session.view.to_list()
    )


def test_orthoscopic_is_the_sign_of_the_last_number():
    view = PymolView.default()
    view.orthoscopic = True
    assert view.to_list()[24] > 0
    view.orthoscopic = False
    assert view.to_list()[24] < 0
    assert PymolView.from_list(view.to_list()).orthoscopic is False


@requires_bpy
def test_camera_round_trip_keeps_the_framing(clean_scene):
    """The vertical field of view depends on the sensor fit, not on angle_y.

    Blender's default fit is AUTO, where the sensor width spans the larger
    image dimension; taking `angle_y` for the vertical field of view there
    makes the camera a third too tight and crops the figure.
    """
    import math

    import bpy

    from blender_gala.pymol.view import camera_to_view, view_to_camera

    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = 512

    data = bpy.data.cameras.new("test")
    data.lens = 85.0
    data.sensor_fit = "AUTO"
    camera = bpy.data.objects.new("test", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (0.0, -5.0, 0.0)

    view = camera_to_view(origin=[0.0, 0.0, 0.0])
    # On a square render under AUTO fit, that is the horizontal angle.
    assert view.field_of_view == pytest.approx(math.degrees(data.angle_x), abs=1e-3)

    view_to_camera(view)
    again = camera_to_view(origin=[0.0, 0.0, 0.0])
    assert again.field_of_view == pytest.approx(view.field_of_view, abs=1e-3)
    assert again.position == pytest.approx(view.position, abs=1e-3)


# ---------------------------------------------------------------------------
# In Blender
# ---------------------------------------------------------------------------


@requires_mn
def test_load_session_builds_the_scene(clean_scene):
    from blender_gala.pymol.load import load_session

    result = load_session(FIXTURE)

    assert set(result.molecules) == {"site", "shell"}
    site = result.molecules["site"]
    assert len(site.object.data.vertices) == 73
    assert set(result.styles["site"]) >= {"cartoon", "sticks"}
    assert result.camera is not None
    # One of each kind of measurement, and the session's one label.
    assert {m.kind for m in result.measurements} == {
        "distance",
        "angle",
        "dihedral",
    }
    assert len(result.labels) >= 1


@requires_mn
def test_load_session_carries_colours_and_selections(clean_scene):
    from blender_gala.pymol.load import load_session

    result = load_session(FIXTURE)
    mesh = result.molecules["site"].object.data

    colors = np.zeros(len(mesh.vertices) * 4, dtype=np.float32)
    mesh.attributes["Color"].data.foreach_get("color", colors)
    colors = colors.reshape(-1, 4)
    assert len(np.unique(np.round(colors, 3), axis=0)) > 1

    # skyblue, which is what the fixture paints the polymer — but Blender's
    # colour attributes are linear and PyMOL's values are display values, so
    # what lands on the mesh is the converted one.
    from blender_gala.color.colormaps import srgb_to_linear

    skyblue = srgb_to_linear(np.array([0.2, 0.502, 0.8]))
    assert any(row[:3] == pytest.approx(skyblue, abs=0.01) for row in colors)

    assert "pocket" in [a.name for a in mesh.attributes]
    picked = np.zeros(len(mesh.vertices), dtype=bool)
    mesh.attributes["pocket"].data.foreach_get("value", picked)
    assert picked.sum() == 36


@requires_mn
def test_load_session_restores_b_factors(clean_scene):
    """The CIF handed to Molecular Nodes has no B_iso column to read."""
    from blender_gala.pymol.load import load_session

    result = load_session(FIXTURE)
    mesh = result.molecules["site"].object.data
    values = np.zeros(len(mesh.vertices), dtype=np.float32)
    mesh.attributes["b_factor"].data.foreach_get("value", values)
    assert np.isfinite(values).all()
    assert values.max() > 0.0


@requires_mn
def test_load_session_places_a_moved_object(clean_scene):
    from blender_gala.pymol.load import load_session

    result = load_session(FIXTURE)
    shell = result.molecules["shell"].object
    # 12 A along +X at Molecular Nodes' world scale.
    assert shell.matrix_world.translation.x == pytest.approx(0.12, abs=1e-4)


@requires_mn
def test_save_session_writes_what_is_in_the_scene(clean_scene, tmp_path):
    from blender_gala.pymol.load import load_session
    from blender_gala.pymol.save import save_session

    load_session(FIXTURE)
    path = str(tmp_path / "out.pse")
    result = save_session(path, selections=["pocket"])

    assert os.path.exists(path)
    names = {molecule.name for molecule in result.session.molecules}
    assert names == {"site", "shell"}

    written = read_session(path)
    site = written.find("site")
    assert site.n_atoms == 73
    assert "cartoon" in site.reps_present()
    assert site.b_factor.max() > 0.0
    # Secondary structure has to survive, or PyMOL draws every helix as a loop.
    assert set(site.ss.tolist()) <= {"", "H", "S", "L"}
    assert [s.name for s in written.selections] == ["pocket"]


@requires_mn
def test_save_session_keeps_colours_recognisable(clean_scene, tmp_path):
    from blender_gala.pymol.load import load_session
    from blender_gala.pymol.save import save_session

    load_session(FIXTURE)
    path = str(tmp_path / "out.pse")
    save_session(path)

    written = read_session(path)
    site = written.find("site")
    names = {palette.name_for_index(int(i)) for i in site.color_index}
    # A colour that is exactly one of PyMOL's own is written as that colour,
    # rather than as an anonymous copy of it.
    assert "skyblue" in names


@requires_mn
def test_save_session_exports_measurements(clean_scene, tmp_path):
    from blender_gala.pymol.load import load_session
    from blender_gala.pymol.save import save_session

    loaded = load_session(FIXTURE)
    path = str(tmp_path / "out.pse")
    save_session(path)

    written = read_session(path)
    kinds = {m.kind for m in written.measurements}
    assert kinds == {m.kind for m in loaded.measurements}
    values = sorted(v for m in written.measurements for v in m.values)
    expected = sorted(m.value for m in loaded.measurements)
    assert values == pytest.approx(expected, abs=1e-3)


@requires_mn
def test_reps_map_to_styles_and_back(clean_scene, tmp_path):
    """Every representation Gala claims to handle survives both directions."""
    from blender_gala.pymol.load import STYLE_MAP
    from blender_gala.pymol.save import REP_MAP

    for rep in STYLE_MAP:
        assert rep in REPS
    for reps in REP_MAP.values():
        for rep in reps:
            assert rep in REPS


@requires_mn
def test_colours_survive_the_colour_space_round_trip(clean_scene, tmp_path):
    """Linear on the mesh, display values in the session, and back again.

    Getting this wrong is not subtle: every exported colour lands in PyMOL
    noticeably darker than it looks in Blender, and none of them match the
    built-in colour they came from.
    """
    from blender_gala.pymol.load import load_session
    from blender_gala.pymol.save import save_session

    before = read_session(FIXTURE)
    load_session(FIXTURE)
    path = str(tmp_path / "out.pse")
    save_session(path)

    after = read_session(path)
    for name in ("site", "shell"):
        original = before.atom_colors(before.find(name))
        again = after.atom_colors(after.find(name))
        assert again == pytest.approx(original, abs=1.5 / 255)
