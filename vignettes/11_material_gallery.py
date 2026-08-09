"""Vignette 11 — one fold, nine ways.

Gala ships a table of materials, and `publication_setup` picks from it for you.
This is what the table is: the same protein nine times, differing in nothing
but what it is made of. Same camera, same lights, same coordinates — so every
difference in the picture is the material and nothing else.

They are grouped, because "material" covers three questions that are not the
same question:

* **shading** — the Principled BSDF's own lobes. Matte is the figure default;
  wax adds subsurface scattering; metal swaps diffuse reflection for metallic.
* **optics** — what a diffuse-plus-specular model cannot do at all. Light going
  *through* the surface, light *made* by it, and light interfering with itself
  in a film a few hundred nanometres thick.
* **texture** — the surface varying from place to place. A BSDF with numbers
  in it can be a convincing *class* of surface and never a particular one: no
  combination of roughness and base colour produces rust in patches or the
  grain of a plank.

Every size on the sheet comes from one decision: the fold is lit and framed as
though it were an object you could pick up, so twenty-five angstrom of
ubiquitin stands in for about fifteen centimetres of something in your hand.
That is what decides how big its materials should be — and it is what splits
the last band in two.

A photograph has a real-world size, which Poly Haven publishes: the rust here
is 2.2 metres of wall, and clothing a hand-sized object with it means using
seven per cent of the image. It works, and the softness is the honest cost.
The marble and the oak are **procedural** instead: a distorted three-
dimensional field evaluated in object space, which has no tile and no
resolution, so the vein spacing is a number in millimetres. The surface does
not carry the texture, it cuts through it — the protein is not wrapped in
marble, it is carved out of a block of it.

Ubiquitin (1UBQ) is the subject: a beta-grasp fold small enough to read at a
ninth of the frame, with a helix, a sheet and a long tail, so there is a curved
surface, a flat one and an edge for each material to be judged on.

The one photographed texture is downloaded on first run into `build/textures`
and cached. Nothing is committed, and a run with no network draws the eight
materials that need no download and says what it could not reach.

    blender --background --python vignettes/11_material_gallery.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from _common import (
    QUALITY,
    heading,
    load_structure,
    load_texture,
    render,
    save_blend,
    setup,
)

mn, gala = setup()

import bpy

from blender_gala.core.entity import AtomStructure
from blender_gala.core.geometry import make_text
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material
from blender_gala.scene.presets import get_preset

# One colour under all of them, so nothing but the material changes between
# cells. Anything that sets its own colour — the clay, and every texture —
# overrides it, and saying so is half the point of showing them side by side.
HUE = "#6f9bc4"

# Space between cells, as a fraction of the size of what is drawn in one. Small
# enough that the molecules are the picture and the gaps are just gaps.
GAP_ACROSS = 0.22
GAP_DOWN = 0.07


# ---------------------------------------------------------------------------
heading("1. One molecule, centred, and measured as drawn")
# ---------------------------------------------------------------------------
molecule = load_structure("1ubq")
molecule.add_style("cartoon", color=None)
gala.color_by_selection(molecule, {"all": HUE})

gala.set_origin_to_geometry(molecule, method="centroid", move_to_world_origin=True)
structure = AtomStructure.from_any(molecule)
_, radius = structure.bounding_sphere()

# The bounding sphere is the wrong size to lay a grid out on. It is the radius
# of a ball containing every atom, and a cartoon seen head on fills well under
# that: the ribbon is a thread through the middle of the atoms, the fold is
# flatter than it is wide, and nothing at all is drawn in the corners. Spacing
# the cells by it leaves a third of the picture as air and shrinks every
# molecule in it.
#
# What matters is the extent of the geometry *as evaluated*, in the two
# directions the front view puts on screen — X across, Z up. That is the same
# reading `frame_target` takes, so a grid built on it and a camera fitted to it
# agree.
from blender_gala.scene.camera import _object_points

drawn = _object_points(molecule.object)
half_width = 0.5 * float(np.ptp(drawn[:, 0]))
half_height = 0.5 * float(np.ptp(drawn[:, 2]))
print(f"  {structure.n_atoms} atoms, bounding radius {radius:.3f} Blender units")
print(
    f"  drawn: {half_width * 2:.3f} wide by {half_height * 2:.3f} tall, "
    f"against a {radius * 2:.3f} bounding sphere"
)


# ---------------------------------------------------------------------------
heading("2. Shading: the Principled BSDF's own lobes")
# ---------------------------------------------------------------------------
# Each of these is `MATERIAL_PRESETS[...]` with a couple of numbers moved. The
# dataclass is frozen and `with_()` returns a copy, so a spec can be derived
# from a preset without the preset changing underneath anyone else.
SHADING = (
    (
        "matte",
        "default shading",
        MATERIAL_PRESETS["protein"],
    ),
    (
        "wax",
        "subsurface scattering",
        # `subsurface_scale` is the whole difference between this cell and the
        # one to its left. Blender's default is 0.005 units, which is 5 mm in a
        # scene built at human scale and half an angstrom in one built at
        # Molecular Nodes'. Light that only penetrates half an angstrom does
        # not visibly penetrate anything: at the default, a subsurface weight
        # of 1.0 renders identically to matte. 0.25 puts the scattering
        # distance at a few angstrom, which is the thickness of the ribbon it
        # has to cross.
        MATERIAL_PRESETS["protein"].with_(
            roughness=0.28,
            subsurface_weight=1.0,
            subsurface_radius=(1.0, 0.42, 0.22),
            subsurface_scale=0.25,
        ),
    ),
    (
        "metal",
        "metallic reflection",
        MATERIAL_PRESETS["metal"].with_(roughness=0.28),
    ),
)


# ---------------------------------------------------------------------------
heading("3. Optics: what a diffuse-plus-specular model cannot do")
# ---------------------------------------------------------------------------
# Frosted rather than polished. Clear glass on a molecule is a puzzle: the
# ribbon refracts the ribbon behind it, the silhouette dissolves, and what
# survives is a shape nobody can name. Roughening the transmission scatters
# the light on its way through, which keeps the outline and still says
# *transparent*.
OPTICS = [
    (
        "frosted glass",
        "rough transmission",
        MATERIAL_PRESETS["glass_surface"].with_(color_mix=0.25, roughness=0.32),
    ),
    (
        "emission",
        "self-illumination",
        # Bright enough to be *emissive* rather than pale: under a shared light
        # rig a gently glowing material is indistinguishable from a light-
        # coloured one, and the difference only shows when the surface is
        # brighter than anything falling on it. Not so bright that it stops
        # being a protein, though — past about three it clips to white across
        # the whole ribbon and the helices and sheet go with it. This sits far
        # enough over white to spill onto the backdrop and to bloom, and no
        # further.
        MATERIAL_PRESETS["protein"].with_(
            emission_strength=2.3,
            emission_color=(0.16, 0.62, 1.0, 1.0),
            base_color=(0.05, 0.16, 0.28, 1.0),
            use_attribute_color=False,
            roughness=0.5,
        ),
    ),
]


def iridescent_material():
    """A thin film over metal, which is where beetle shells get their colour.

    Not a texture and not a tint: `Thin Film Thickness` puts a film a few
    hundred nanometres thick over the surface, and Cycles works out the
    interference between light reflected off the top of it and light reflected
    off the bottom. The colour that comes back therefore depends on the angle
    the surface is seen at, so it sweeps across a curved ribbon on its own.
    Nothing in a base-colour-and-roughness model reproduces it.
    """
    material = build_material(
        MATERIAL_PRESETS["metal"].with_(roughness=0.22, use_attribute_color=False),
        name="Gallery iridescent",
    )
    principled = next(
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    principled.inputs["Base Color"].default_value = (0.55, 0.58, 0.62, 1.0)
    # Nanometres. Around a quarter to a half of a wavelength of visible light
    # is where the interference lands in the visible band at all.
    principled.inputs["Thin Film Thickness"].default_value = 380.0
    # 1.7 rather than 2.4: the higher the film's index, the more of the
    # spectrum it sweeps through and the closer it gets to an oil slick.
    # Lower keeps it to the pearl end, where the colour shifts without
    # announcing itself.
    principled.inputs["Thin Film IOR"].default_value = 1.7
    return material


# ---------------------------------------------------------------------------
heading("4. Texture: photographs of a real surface")
# ---------------------------------------------------------------------------
# A Principled BSDF with numbers in it can be a convincing *class* of surface
# and never a particular one: real iron rusts in patches, real moss grows in
# clumps, and no combination of roughness and base colour produces either.
# That takes an image, and Poly Haven gives away thousands of them under CC0 —
# each a set of maps photographed off one real surface.
#
# Two things decide whether one is worth putting on a molecule, and neither is
# how good the photograph is.
#
# The first is **scale** — how many times the image tiles across a Blender
# unit, which is a hundred angstrom. A protein thirty across at scale ten
# shows three tiles per cell, which is a flat wash of the image's average
# colour. Twenty to thirty puts six or more across it, and the pattern becomes
# something the eye can read.
#
# The second is **local contrast**, and it rules out most of the catalogue.
# Polished marble is a beautiful texture and a hopeless one here: its veins are
# metres apart on a wall, so a ribbon two angstrom wide crosses a uniform patch
# of it and comes out flat cream whatever the scale. What survives being cut
# into ribbons is anything whose pattern repeats over a few millimetres —
# corrosion, moss, grain.
#
# `load_texture` fetches them into build/ and caches them. Anything it cannot
# reach is dropped rather than faked, so a run with no network draws the two
# bands above and says what is missing.
# One photograph, kept to show the route. Poly Haven publishes the real-world
# size of every texture, and that number is the whole argument of this band:
# `rust_coarse_01` is 2.2 metres of wall. Asking it to clothe an object you
# could hold means using seven per cent of it, and seven per cent of a 2k image
# is a hundred and forty pixels across the molecule. It works, and it is as
# sharp as it is ever going to be.
#
# (name, asset, descriptor, tile size in mm, bump strength)
TEXTURES = (("corroded iron", "rust_coarse_01", "photographed", 2200.0, 2.2),)

textures = []
for label, asset, note, tile_mm, relief in TEXTURES:
    # 2k rather than 1k: at this magnification the crop is small enough that
    # the extra pixels are the difference between grain and mush.
    files = load_texture(asset, resolution="2k")
    if files.get("Diffuse"):
        textures.append((label, note, tile_mm, relief, files))
    else:
        print(f"  no {label}: leaving it off the sheet")


# How big the protein is pretending to be. Every scale below is worked out
# from this: the sheet is lit and framed as though the fold were an object you
# could pick up and turn over, so its materials should be the size they would
# be on one. Twenty-five angstrom of ubiquitin stands in for fifteen
# centimetres of something in your hand.
HAND_MM = 150.0


def per_mm() -> float:
    """Blender units per millimetre of the object as held."""
    return (2.0 * half_width) / HAND_MM


def tiles_across(tile_mm: float) -> float:
    """Mapping scale for a photographed texture of a given real-world size."""
    return (HAND_MM / tile_mm) / (2.0 * half_width)


def feature_scale(spacing_mm: float) -> float:
    """Mapping scale that puts a procedural feature every ``spacing_mm``."""
    return 1.0 / (spacing_mm * per_mm())


def procedural_field(
    tree,
    label: str,
    stretch,
    vein_mm: float,
    wander_mm: float,
    distortion: float,
    detail: float,
):
    """A distorted 3D band field in object space, and the socket carrying it.

    This is the part a photograph cannot do. The field is evaluated in the
    *object's own three-dimensional space*, so it does not lie on the surface —
    the surface cuts through it. A vein does not stop at a silhouette and
    resume somewhere unrelated on the far side; it continues through the body
    of the molecule and comes out where the geometry says it should, which is
    what carving something out of a block looks like.

    It also has no resolution and no tile. A photograph of marble is 4.3 metres
    of quarry face at 2048 pixels, and asking it for a vein every four
    centimetres means magnifying it thirty times. A field is evaluated per
    shading point, so the vein spacing is simply a number in millimetres.
    """
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-1500, 0)

    # A BANDS wave already varies along one axis only, so it is *already*
    # infinitely long in the other two — the anisotropy here is not for the
    # bands. It is for the noise that displaces them: compressing the
    # coordinate across the grain makes the wander change slowly along a vein,
    # so veins run a long way before they bend. Scaling the band axis instead
    # multiplies the frequency, which is how the first attempt at oak turned
    # thirty grain lines into three hundred and came out flat tan.
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.label = f"{label} block"
    mapping.location = (-1300, 0)
    mapping.inputs["Scale"].default_value = stretch
    tree.links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])

    # The wander. Bands on their own are a barcode; displacing the coordinate
    # with a noise field before the bands are evaluated is what makes a vein
    # meander.
    drift = tree.nodes.new("ShaderNodeTexNoise")
    drift.label = f"{label} wander"
    drift.location = (-1080, -260)
    drift.inputs["Scale"].default_value = feature_scale(vein_mm * 2.2)
    drift.inputs["Detail"].default_value = 6.0
    drift.inputs["Roughness"].default_value = 0.55
    tree.links.new(mapping.outputs["Vector"], drift.inputs["Vector"])

    push = tree.nodes.new("ShaderNodeVectorMath")
    push.operation = "SCALE"
    push.location = (-880, -260)
    # In millimetres, like every other distance here, and that matters more
    # than it looks: this displacement is *added to the coordinate*, so a value
    # in raw Blender units is being compared against the quarter of a unit the
    # whole molecule occupies. Asking for 0.34 — larger than the object — did
    # not make the bands meander, it dissolved them. Then asking for a fifth of
    # the spacing made them stop meandering at all, and oak came out as
    # corduroy. About the spacing itself is where grain looks like grain.
    push.inputs["Scale"].default_value = wander_mm * per_mm()
    tree.links.new(drift.outputs["Color"], push.inputs[0])

    displaced = tree.nodes.new("ShaderNodeVectorMath")
    displaced.operation = "ADD"
    displaced.location = (-700, -120)
    tree.links.new(mapping.outputs["Vector"], displaced.inputs[0])
    tree.links.new(push.outputs["Vector"], displaced.inputs[1])

    bands = tree.nodes.new("ShaderNodeTexWave")
    bands.label = f"{label} veins"
    bands.wave_type = "BANDS"
    bands.bands_direction = "X"
    bands.location = (-500, -120)
    bands.inputs["Scale"].default_value = feature_scale(vein_mm)
    bands.inputs["Distortion"].default_value = distortion
    bands.inputs["Detail"].default_value = detail
    bands.inputs["Detail Scale"].default_value = 1.0
    tree.links.new(displaced.outputs["Vector"], bands.inputs["Vector"])
    return bands.outputs["Fac"]


def carved_material(
    label: str,
    stretch,
    vein_mm: float,
    wander_mm: float,
    stops,
    roughness_range,
    bump: float,
    grain_mm: float,
    tooth: float = 0.12,
    distortion: float = 0.6,
    detail: float = 1.0,
):
    """A Gala material whose colour, roughness and relief come from one field.

    All three from the *same* field, which is the difference between a
    material and a picture with a bump map bolted on: where the vein is, the
    stone is a different colour, takes light differently and stands slightly
    proud. A second, much finer noise adds the microscopic tooth that catches
    the key light.
    """
    material = build_material(
        MATERIAL_PRESETS["protein"].with_(use_attribute_color=False),
        name=f"Gallery {label}",
    )
    tree = material.node_tree
    principled = next(
        node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )

    field = procedural_field(
        tree, label, stretch, vein_mm, wander_mm, distortion, detail
    )

    colour = tree.nodes.new("ShaderNodeValToRGB")
    colour.label = f"{label} colour"
    colour.location = (-260, 120)
    ramp = colour.color_ramp
    ramp.elements[0].position, ramp.elements[0].color = stops[0]
    ramp.elements[1].position, ramp.elements[1].color = stops[1]
    for position, value in stops[2:]:
        element = ramp.elements.new(position)
        element.color = value
    tree.links.new(field, colour.inputs["Fac"])
    tree.links.new(colour.outputs["Color"], principled.inputs["Base Color"])

    rough = tree.nodes.new("ShaderNodeMapRange")
    rough.label = f"{label} roughness"
    rough.location = (-260, -120)
    rough.inputs["To Min"].default_value = roughness_range[0]
    rough.inputs["To Max"].default_value = roughness_range[1]
    tree.links.new(field, rough.inputs["Value"])
    tree.links.new(rough.outputs["Result"], principled.inputs["Roughness"])

    # The veins, in relief. Small: a vein standing a millimetre proud of a
    # marble surface would be a fault, not a figure.
    vein_bump = tree.nodes.new("ShaderNodeBump")
    vein_bump.label = f"{label} vein relief"
    vein_bump.location = (-60, -320)
    vein_bump.inputs["Strength"].default_value = bump
    vein_bump.inputs["Distance"].default_value = 0.4 * per_mm()
    tree.links.new(field, vein_bump.inputs["Height"])

    # Coarse enough to be seen. A procedural field costs nothing to evaluate at
    # any frequency, which makes it easy to ask for detail finer than a pixel —
    # and detail finer than a pixel is not detail, it is noise: it shimmers, it
    # does not read as surface, and it defeats PNG so thoroughly that it turned
    # a 1.5 MB figure into a 7 MB one. Keep the tooth a few millimetres across
    # and it lands on tens of pixels instead of fractions of one.
    speckle = tree.nodes.new("ShaderNodeTexNoise")
    speckle.label = f"{label} tooth"
    speckle.location = (-500, -520)
    speckle.inputs["Scale"].default_value = feature_scale(grain_mm)
    speckle.inputs["Detail"].default_value = 5.0
    speckle.inputs["Roughness"].default_value = 0.5

    micro = tree.nodes.new("ShaderNodeBump")
    micro.label = f"{label} tooth relief"
    micro.location = (-260, -520)
    micro.inputs["Strength"].default_value = tooth
    micro.inputs["Distance"].default_value = 0.08 * per_mm()
    tree.links.new(speckle.outputs["Fac"], micro.inputs["Height"])
    # Chained, so the fine relief perturbs the normal the veins already gave.
    tree.links.new(micro.outputs["Normal"], vein_bump.inputs["Normal"])
    tree.links.new(vein_bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def marble_material():
    """Pale stone, veined every four centimetres of the object as held."""
    return carved_material(
        "marble",
        stretch=(1.0, 0.32, 0.45),
        vein_mm=46.0,
        wander_mm=52.0,
        # Four stops, and the point of them is the *narrow* one. A two-stop
        # ramp gives a smooth gradient from white to grey, which is a blush
        # rather than a vein. Holding white until 0.46, dropping to dark over
        # six hundredths and coming back is what makes a line.
        stops=(
            (0.0, (0.90, 0.89, 0.87, 1.0)),
            (1.0, (0.88, 0.87, 0.85, 1.0)),
            (0.46, (0.87, 0.86, 0.84, 1.0)),
            (0.52, (0.24, 0.26, 0.29, 1.0)),
            (0.60, (0.85, 0.84, 0.82, 1.0)),
        ),
        roughness_range=(0.18, 0.34),
        bump=0.28,
        grain_mm=4.0,
        tooth=0.10,
        distortion=2.2,
        detail=1.0,
    )


def oak_material():
    """Grain every four millimetres, running the length of the block."""
    return carved_material(
        "oiled oak",
        # Elongated along the grain, but only by half — compress it much
        # harder and the drift noise stops varying along a line, so every band
        # shifts as a rigid unit and stays exactly parallel to its neighbours.
        # That is corduroy, not oak. The grain has to wander *along* itself.
        stretch=(1.0, 0.5, 0.5),
        vein_mm=9.0,
        wander_mm=7.0,
        # Early wood against late wood. The first attempt put three browns a
        # few percent apart in here and the grain vanished under the tooth —
        # the bands were there, they just had nothing to say.
        stops=(
            (0.0, (0.60, 0.38, 0.18, 1.0)),
            (1.0, (0.16, 0.08, 0.03, 1.0)),
            (0.5, (0.36, 0.20, 0.09, 1.0)),
        ),
        roughness_range=(0.22, 0.48),
        bump=0.34,
        grain_mm=6.0,
        tooth=0.08,
        distortion=0.9,
        detail=2.0,
    )


def textured_material(label: str, files: dict, scale: float, relief: float = 1.0):
    """A Gala material with a photographed surface wired into it.

    Two problems have to be solved to put an image on a molecule. The first is
    that a cartoon has no UV map — nothing ever unwrapped it, and there is no
    sensible way to. **Box projection** solves it without one: the image is
    projected down all three axes at once and blended where they meet, so any
    geometry takes the texture whatever shape it is.

    The second is scale. The coordinates come from `Object` rather than
    `Generated`, so the texture is pinned to the molecule's own space and does
    not stretch when a style changes its bounding box, and the mapping scale
    says how many times the image tiles across one Blender unit — a hundred
    angstrom. Around ten puts a few tiles across a small protein, which is what
    makes marble read as marble rather than as a grey wash.
    """
    material = build_material(
        MATERIAL_PRESETS["protein"].with_(
            use_attribute_color=False, subsurface_weight=0.0
        ),
        name=f"Gallery {label}",
    )
    tree = material.node_tree
    principled = next(
        node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )

    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-1100, 0)
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.location = (-900, 0)
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    tree.links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])

    def image(path: str, colorspace: str, height: float):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(path, check_existing=True)
        node.image.colorspace_settings.name = colorspace
        node.projection = "BOX"
        node.projection_blend = 0.25
        node.location = (-650, height)
        tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    # Colour is a photograph and belongs in sRGB; roughness and normals are
    # measurements that happen to be stored as pictures, and a colour transform
    # applied to either is a wrong answer rather than a different look.
    colour = image(files["Diffuse"], "sRGB", 300.0)
    tree.links.new(colour.outputs["Color"], principled.inputs["Base Color"])

    if files.get("Rough"):
        rough = image(files["Rough"], "Non-Color", 0.0)
        tree.links.new(rough.outputs["Color"], principled.inputs["Roughness"])

    if files.get("nor_gl"):
        bumps = image(files["nor_gl"], "Non-Color", -300.0)
        normals = tree.nodes.new("ShaderNodeNormalMap")
        normals.location = (-380, -300)
        # Above 1, because a normal map photographed off a wall is calibrated
        # for a wall: at the size these tile to here, its relief is a fraction
        # of a degree of surface tilt and the light does not visibly catch on
        # it. Pushing it is the difference between a photograph projected onto
        # a protein and a surface that the key light rakes across.
        normals.inputs["Strength"].default_value = relief
        tree.links.new(bumps.outputs["Color"], normals.inputs["Color"])
        tree.links.new(normals.outputs["Normal"], principled.inputs["Normal"])

    return material


# The sheet, as three bands of three. Grouping them is the point: a subsurface
# wax and a photograph of moss are both "materials" and are not the same kind
# of answer to the same question, and a contact sheet that mixes them without
# saying so reads as a bag of effects.
BANDS = [
    (
        "shading",
        [
            (name, note, build_material(spec, name=f"Gallery {name}"))
            for name, note, spec in SHADING
        ],
    ),
    (
        "optics",
        [
            (name, note, build_material(spec, name=f"Gallery {name}"))
            for name, note, spec in OPTICS
        ]
        + [("iridescent", "thin-film interference", iridescent_material())],
    ),
    (
        "texture",
        [
            (label, note, textured_material(label, files, tiles_across(tile), relief))
            for label, note, tile, relief, files in textures
        ]
        + [
            ("oiled oak", "procedural, in 3D", oak_material()),
            ("marble", "procedural, in 3D", marble_material()),
        ],
    ),
]
BANDS = [(name, cells) for name, cells in BANDS if cells]

#: What each band is about, in three words, under its heading.
BAND_NOTES = {
    "shading": "surface response",
    "optics": "light transport",
    "texture": "spatial variation",
}
COLUMNS = max(len(cells) for _, cells in BANDS)
print(f"  {sum(len(cells) for _, cells in BANDS)} cells in {len(BANDS)} bands")
for name, cells in BANDS:
    print(f"      {name:9s} {', '.join(cell[0] for cell in cells)}")


# ---------------------------------------------------------------------------
heading("5. One copy each, every one with its own node group")
# ---------------------------------------------------------------------------
# `object.copy()` gives a new object that still points at the *same* geometry
# node group, so setting a material on one would set it on all six. Copying
# the group is what makes the copies independent; the mesh underneath stays
# shared, because that is 602 atoms nobody needs six of.
label_material = build_material(
    MATERIAL_PRESETS["protein"].with_(
        use_attribute_color=False,
        base_color=(0.88, 0.89, 0.91, 1.0),
        emission_strength=0.8,
        emission_color=(0.88, 0.89, 0.91, 1.0),
        roughness=1.0,
    ),
    name="Gallery Label",
)

# A second, dimmer grey for everything that is not a heading. Size alone did
# not separate the descriptor from the name it sits under — at the same white,
# a line of 60% type still reads as a second label rather than as a gloss on
# the first. Dropping its value does what shrinking it could not.
subdued_material = build_material(
    MATERIAL_PRESETS["protein"].with_(
        use_attribute_color=False,
        base_color=(0.50, 0.54, 0.59, 1.0),
        emission_strength=0.55,
        emission_color=(0.50, 0.54, 0.59, 1.0),
        roughness=1.0,
    ),
    name="Gallery Label Subdued",
)

CAPTION_SIZE = half_height * 0.30
SUBTITLE_SIZE = CAPTION_SIZE * 0.62
BAND_SIZE = CAPTION_SIZE * 1.15
BAND_NOTE_SIZE = BAND_SIZE * 0.52

# One vertical rhythm, used everywhere above the molecule: the gap from the
# band heading down to its note is the same as the gap from that note down to
# the top of the protein. Two different gaps there read as an accident even
# when neither is wrong, and a heading floating a long way above its own row
# stops belonging to it.
STEP = BAND_NOTE_SIZE * 1.45

BAND_NOTE_RISE = half_height + STEP
BAND_RISE = BAND_NOTE_RISE + STEP

# Below the molecule the names sit close to what they name — a caption with a
# gap above it belongs to nothing in particular.
CAPTION_DROP = half_height + CAPTION_SIZE * 0.62
SUBTITLE_DROP = CAPTION_DROP + CAPTION_SIZE * 0.78

CELL_TOP = BAND_RISE + BAND_SIZE * 0.85
CELL_BOTTOM = -(SUBTITLE_DROP + SUBTITLE_SIZE * 1.1)
CELL_MIDDLE = 0.5 * (CELL_TOP + CELL_BOTTOM)

COLUMN_PITCH = 2.0 * half_width * (1.0 + GAP_ACROSS)
ROW_PITCH = (CELL_TOP - CELL_BOTTOM) * (1.0 + GAP_DOWN)
print(f"  cells {COLUMN_PITCH:.3f} apart across, {ROW_PITCH:.3f} down")


extent: list[tuple[float, float, float]] = []


def caption_at(
    name: str,
    text: str,
    x: float,
    z: float,
    size: float,
    align="CENTER",
    material=None,
    italic: bool = False,
):
    """One line of type, lying in the plane the front view looks at.

    Italics are a shear on the glyphs rather than an italic font, because
    Blender ships exactly one typeface and an italic cut would have to be found
    on the machine — which works on the one it was written on and not in CI.
    A slant is what an oblique is anyway.
    """
    obj = make_text(
        f"GALA Gallery {name}",
        text,
        (x, -half_width, z),
        size=size,
        material=material or label_material,
        align_x=align,
    )
    if italic:
        obj.data.shear = 0.28
    obj.rotation_euler = (math.pi / 2, 0.0, 0.0)

    # However wide the words turn out to be. A caption is laid out by the font,
    # not by the grid, and "thin-film interference" is wider than the molecule
    # it belongs to — so the frame is told about the text as well, and a long
    # name pushes the framing out rather than being quietly cropped in half.
    bpy.context.view_layer.update()
    width = float(obj.dimensions.x)
    left = obj.location.x if align == "LEFT" else obj.location.x - width / 2.0
    for edge in (left, left + width):
        extent.append((edge, 0.0, z))
        extent.append((edge, 0.0, z - size * 0.4))
    return obj


first = True
for row, (band, cells) in enumerate(BANDS):
    middle = (len(cells) - 1) / 2
    band_z = (0.5 * (len(BANDS) - 1) - row) * ROW_PITCH - CELL_MIDDLE

    for column, (name, note, material) in enumerate(cells):
        if first:
            obj = molecule.object
            first = False
        else:
            obj = molecule.object.copy()
            obj.data = molecule.object.data
            obj.name = f"1ubq {name}"
            for modifier in obj.modifiers:
                if modifier.type == "NODES" and modifier.node_group is not None:
                    modifier.node_group = modifier.node_group.copy()
            bpy.context.scene.collection.objects.link(obj)

        # The cell's *content* is what sits on the grid, not the molecule: two
        # lines of caption hang below it and a band label sits above, so
        # placing the molecules on a regular pitch would leave the sheet
        # hanging off centre.
        obj.location = ((column - middle) * COLUMN_PITCH, 0.0, band_z)
        gala.assign_material(obj, material)

        caption_at(
            f"Name {name}",
            name,
            obj.location.x,
            obj.location.z - CAPTION_DROP,
            CAPTION_SIZE,
        )
        caption_at(
            f"Note {name}",
            note,
            obj.location.x,
            obj.location.z - SUBTITLE_DROP,
            SUBTITLE_SIZE,
        )

        # What the frame has to hold for this cell. Stated here rather than
        # left to the camera, because framing deliberately ignores Gala's own
        # annotations — a long material name is not a reason to push the
        # molecules further away — and a caption cropped off the bottom edge is
        # exactly what that rule costs when the annotation *is* the figure.
        for corner in (-1.0, 1.0):
            for height in (CELL_TOP, CELL_BOTTOM):
                extent.append(
                    (obj.location.x + corner * half_width, 0.0, obj.location.z + height)
                )

        print(f"  {band:9s} {name:15s} {note}")

    # The band's heading, over the leftmost cell of its row and left-aligned
    # with it, so the three groups read as three groups.
    left = -middle * COLUMN_PITCH - half_width
    caption_at(
        f"Band {band}",
        band.upper(),
        left,
        band_z + BAND_RISE,
        BAND_SIZE,
        align="LEFT",
    )
    caption_at(
        f"Band note {band}",
        BAND_NOTES[band],
        left,
        band_z + BAND_NOTE_RISE,
        BAND_NOTE_SIZE,
        align="LEFT",
        material=subdued_material,
        italic=True,
    )

layout = bpy.data.meshes.new("Gallery Extent")
layout.from_pydata(extent, [], [])
layout.update()
extent_object = bpy.data.objects.new("GALA Gallery Extent", layout)
extent_object["gala"] = "extent"
extent_object.hide_render = True
bpy.context.scene.collection.objects.link(extent_object)


# ---------------------------------------------------------------------------
heading("6. Light it for materials, not for a figure")
# ---------------------------------------------------------------------------
# Metal and glass are mostly made of what they reflect, and a three-lamp rig
# in an empty world gives them three white rectangles and a void. An HDRI is
# what puts something in the reflection; the rig on top of it keeps the shapes
# reading, which an environment alone does not do. `lighting_style="both"` in
# `publication_setup` is this pair.
gala.setup_render(preset=QUALITY, transparent=True)
gala.scene.render.setup_color_management()
gala.hdri_lighting(hdri="studio", strength=0.9)
gala.three_point_lighting(None, energy=0.5, softness=1.6)

# And something to reflect. Every other vignette here is happy on alpha,
# because a figure drops onto a page; a material sheet is not, because metal is
# mostly a picture of its surroundings and glass is entirely a picture of what
# is behind it — over nothing, both are black. The film is still transparent;
# this wall simply fills the frame, so the alpha comes out opaque anyway and
# the background is a thing in the scene rather than a render setting.
backdrop_mesh = bpy.data.meshes.new("Gallery Backdrop")
span = COLUMN_PITCH * COLUMNS * 4.0
backdrop_mesh.from_pydata(
    [
        (-span, half_width * 30.0, -span),
        (span, half_width * 30.0, -span),
        (span, half_width * 30.0, span),
        (-span, half_width * 30.0, span),
    ],
    [],
    [(0, 1, 2, 3)],
)
backdrop_mesh.update()
backdrop = bpy.data.objects.new("GALA Gallery Backdrop", backdrop_mesh)
backdrop_mesh.materials.append(
    build_material(
        MATERIAL_PRESETS["protein"].with_(
            use_attribute_color=False,
            base_color=(0.16, 0.17, 0.19, 1.0),
            roughness=0.95,
            specular=0.15,
        ),
        name="Gallery Backdrop",
    )
)
bpy.context.scene.collection.objects.link(backdrop)
# Tagged as Gala furniture, which is what keeps a wall four times the width of
# the picture out of any framing that fits "everything visible".
backdrop["gala"] = "backdrop"

# Transmission needs bounces to cross a shell twice and come out the far side.
# The default six is enough for opaque work and turns glass black.
scene = bpy.context.scene
scene.cycles.transmission_bounces = max(scene.cycles.transmission_bounces, 12)
scene.cycles.max_bounces = max(scene.cycles.max_bounces, 16)

# And more samples than the draft preset hands a single opaque molecule. Three
# of these nine are expensive to sample — rough transmission, an emitter and a
# bumped surface — and at 64 the figure comes out grainy enough that the noise
# is the finest detail in it. That is visible, and it is also why the PNG was
# five times the size of every other figure in the documentation: per-pixel
# noise is exactly what a lossless format cannot compress.
scene.cycles.samples = max(scene.cycles.samples, 192)
scene.cycles.adaptive_threshold = min(scene.cycles.adaptive_threshold, 0.02)


# A Glare node, thresholded above white. Only pixels brighter than anything
# the lights could produce bloom, which on this sheet is the emissive cell and
# nothing else — so it costs the other eight nothing and gives that one the
# halo that says *this surface is a light source*. Done after the render, so
# the strength is a slider rather than another hour of sampling.
compositor = bpy.data.node_groups.new("GALA Gallery Bloom", "CompositorNodeTree")
compositor.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
scene.compositing_node_group = compositor

layers = compositor.nodes.new("CompositorNodeRLayers")
layers.scene = scene
layers.location = (-400, 0)

glare = compositor.nodes.new("CompositorNodeGlare")
glare.location = (-100, 0)
glare.inputs["Type"].default_value = "Bloom"
glare.inputs["Quality"].default_value = "High"
glare.inputs["Threshold"].default_value = 1.0
glare.inputs["Strength"].default_value = 0.4
glare.inputs["Size"].default_value = 0.5
compositor.links.new(layers.outputs["Image"], glare.inputs["Image"])

bloom_output = compositor.nodes.new("NodeGroupOutput")
bloom_output.location = (200, 0)
compositor.links.new(glare.outputs["Image"], bloom_output.inputs[0])
print("  bloom above white, so only the emissive cell has a halo")


# ---------------------------------------------------------------------------
heading("7. Let the sheet decide the shape of the frame")
# ---------------------------------------------------------------------------
# Gala's presets are square, because a figure of one molecule is. A contact
# sheet is whatever shape its grid came out, and guessing at that is what
# leaves a band of empty backdrop down both sides — the camera fits the
# narrower dimension and everything in the frame shrinks to pay for the
# margin it did not need.
#
# So the resolution follows the layout: the aspect of the extent that was just
# built, at the preset's width. Three by two of a fold that is taller than it
# is wide, each cell carrying a caption, comes out very nearly square.
preset = get_preset(QUALITY)
sheet = np.asarray(extent)
sheet_width = float(np.ptp(sheet[:, 0]))
sheet_height = float(np.ptp(sheet[:, 2]))
scene.render.resolution_x = preset.resolution[0]
scene.render.resolution_y = round(preset.resolution[0] * sheet_height / sheet_width)
print(f"  the sheet is {sheet_width:.3f} by {sheet_height:.3f} Blender units")

# Framed on the layout rather than on the molecules, so the captions are
# inside the picture. The lens is set between the two calls because the focal
# length decides the field of view and therefore the distance: framing, then
# changing the lens, is framing something else.
camera = gala.frame_target(extent_object, viewpoint="front", margin=1.07)
camera.data.lens = 135.0
gala.frame_target(extent_object, viewpoint="front", margin=1.07)
print(f"  {scene.render.resolution_x} x {scene.render.resolution_y}, 135 mm lens")


# ---------------------------------------------------------------------------
heading("8. Render")
# ---------------------------------------------------------------------------
render(gala, "11_material_gallery")


# ---------------------------------------------------------------------------
# The same sheet, big enough to look at the materials
# ---------------------------------------------------------------------------
# A material is a thing you judge close up, and at the width the documentation
# uses each cell is about three hundred pixels — enough to tell nine materials
# apart and not enough to see what any of them is doing. This renders the sheet
# again at the width a printed figure would use, so the grain of the oak and
# the veins in the marble are there to be looked at.
#
# Off unless asked for, because it is four times the pixels and four times the
# samples of the one CI needs.
if os.environ.get("GALA_GALLERY_DETAIL"):
    heading("9. The same sheet, at figure resolution")
    detail = get_preset("figure")
    scene.render.resolution_x = detail.resolution[0]
    scene.render.resolution_y = round(detail.resolution[0] * sheet_height / sheet_width)
    scene.cycles.samples = detail.samples
    scene.cycles.adaptive_threshold = detail.adaptive_threshold
    # WebP: the same sheet as lossless PNG is 25 MB, which is not a thing to
    # put in a documentation repository for the sake of a figure nobody
    # measures anything off.
    gala.scene.render.set_image_format(scene.render.image_settings, "WEBP", quality=92)
    print(f"  {scene.render.resolution_x} x {scene.render.resolution_y}")
    render(gala, "11_material_gallery_detail", extension="webp")


# ---------------------------------------------------------------------------
heading("10. Save the scene, to open in Blender")
# ---------------------------------------------------------------------------
# Six materials in one file, each on its own object: open it, pick the one
# that suits the molecule you are actually publishing, and copy its node tree
# across. Or keep going — every one of these is an ordinary Principled BSDF,
# and the shader editor is where the seventh treatment comes from.
save_blend("11_material_gallery")
