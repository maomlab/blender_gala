"""Scene-level settings, stored on ``bpy.types.Scene.gala``.

A ``PropertyGroup`` rather than module globals so settings survive save and
reload, and so several scenes in one file can be configured differently.
"""

from __future__ import annotations

import contextlib

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from ..color import colormaps
from ..core.registration import register_classes, unregister_classes
from ..interactions.detect import INTERACTION_KINDS
from ..scene import lighting, materials, presets

__all__ = ["GalaSceneProperties", "classes"]


def _preset_items(self, context):
    return [
        (name, name.title(), preset.description)
        for name, preset in presets.PRESETS.items()
    ]


def _hdri_items(self, context):
    return [
        (name, name.title(), f"Blender's built-in {name} environment")
        for name in lighting.STUDIO_HDRIS
    ]


def _colormap_items(self, context):
    return [(name, name, f"{name} colormap") for name in colormaps.list_colormaps()]


def _material_scheme_items(self, context):
    return [
        (name, name.title(), f"{name} material scheme")
        for name in materials.MATERIAL_SCHEMES
    ]


class GalaSceneProperties(PropertyGroup):
    """Blender Gala settings for one scene."""

    # -- scene setup ----------------------------------------------------
    preset: EnumProperty(
        name="Preset",
        description="Resolution and sampling preset",
        items=_preset_items,
        default=1,  # "figure"
    )
    engine: EnumProperty(
        name="Engine",
        description="Render engine",
        items=[
            ("CYCLES", "Cycles", "Path tracing; best quality for figures"),
            ("EEVEE", "EEVEE", "Rasterised; much faster, less accurate"),
        ],
        default="CYCLES",
    )
    transparent: BoolProperty(
        name="Transparent Background",
        description="Render the background as alpha so the figure drops onto any page",
        default=True,
    )
    use_gpu: BoolProperty(
        name="Use GPU",
        description="Render on the GPU when a supported backend is available",
        default=True,
    )
    view_transform: EnumProperty(
        name="View Transform",
        description=(
            "Standard preserves the colours you chose. AgX looks more "
            "cinematic but shifts hue and desaturates"
        ),
        items=[
            ("Standard", "Standard", "Faithful colour; recommended for figures"),
            ("AgX", "AgX", "Filmic highlight rolloff"),
            ("Khronos PBR Neutral", "PBR Neutral", "Neutral tone mapping"),
            ("Filmic", "Filmic", "Legacy filmic transform"),
        ],
        default="Standard",
    )

    # -- lighting -------------------------------------------------------
    lighting_style: EnumProperty(
        name="Lighting",
        items=[
            ("three_point", "Three Point", "Controllable studio key, fill and rim"),
            ("hdri", "HDRI", "Environment lighting from an image"),
            ("both", "Both", "Three-point rig over a dim HDRI fill"),
            ("none", "None", "Leave the lighting alone"),
        ],
        default="three_point",
    )
    hdri: EnumProperty(name="HDRI", items=_hdri_items, default=5)  # "studio"
    light_energy: FloatProperty(
        name="Energy",
        description="Overall brightness multiplier for the three-point rig",
        default=1.0,
        min=0.0,
        soft_max=5.0,
    )
    light_softness: FloatProperty(
        name="Softness",
        description="Above 1 gives softer shadows, below 1 crisper ones",
        default=1.0,
        min=0.05,
        soft_max=4.0,
    )
    light_rotation: FloatProperty(
        name="Rotation",
        description="Rotate the whole rig about the vertical axis, in degrees",
        default=0.0,
        soft_min=-180.0,
        soft_max=180.0,
    )

    # -- materials and origin -------------------------------------------
    material_scheme: EnumProperty(
        name="Materials", items=_material_scheme_items, default=0
    )
    origin_method: EnumProperty(
        name="Origin",
        items=[
            ("centroid", "Centroid", "Unweighted mean of the atom positions"),
            ("mass", "Centre of Mass", "Mass-weighted mean"),
            ("bounds", "Bounding Box", "Centre of the bounding box"),
            ("none", "Leave Alone", "Do not change the origin"),
        ],
        default="centroid",
    )
    move_to_world_origin: BoolProperty(name="Move to World Origin", default=True)
    viewpoint: EnumProperty(
        name="Viewpoint",
        items=[
            ("iso", "Isometric", "Three-quarter view"),
            ("front", "Front", ""),
            ("back", "Back", ""),
            ("left", "Left", ""),
            ("right", "Right", ""),
            ("top", "Top", ""),
            ("bottom", "Bottom", ""),
        ],
        default="iso",
    )
    frame_camera: BoolProperty(name="Frame Camera", default=True)

    # -- compositing ----------------------------------------------------
    cryptomatte: BoolProperty(
        name="Cryptomatte",
        description=(
            "Enable cryptomatte and depth passes so objects can be re-selected "
            "in the compositor after rendering"
        ),
        default=True,
    )
    depth_of_field: BoolProperty(name="Depth of Field", default=False)
    dof_fstop: FloatProperty(name="F-Stop", default=2.8, min=0.1, soft_max=22.0)
    depth_cue: BoolProperty(name="Depth Cue", default=False)
    depth_cue_near: FloatProperty(name="Near", default=0.0, unit="NONE")
    depth_cue_far: FloatProperty(name="Far", default=100.0)
    exr_directory: StringProperty(
        name="EXR Directory", subtype="DIR_PATH", default="//passes"
    )

    # -- interactions ---------------------------------------------------
    selection_a: StringProperty(
        name="Selection A",
        description="PyMOL-style selection, e.g. 'ligand' or 'chain A and resi 45-60'",
        default="ligand",
    )
    selection_b: StringProperty(
        name="Selection B",
        description="The other side of the interaction",
        default="protein",
    )
    interaction_kinds: EnumProperty(
        name="Interactions",
        description="Which interaction types to look for",
        items=[
            (kind, kind.replace("_", " ").title(), f"Detect {kind} interactions")
            for kind in INTERACTION_KINDS
        ],
        options={"ENUM_FLAG"},
        default={"hbond", "polar", "salt_bridge"},
    )
    interaction_labels: BoolProperty(
        name="Distance Labels",
        description="Place a distance label on each interaction",
        default=False,
    )
    exclude_same_residue: BoolProperty(name="Exclude Same Residue", default=True)

    # -- measurement ----------------------------------------------------
    measure_selection: StringProperty(
        name="Atoms",
        description=(
            "Two to four selections separated by ';', each matching one atom. "
            "Leave empty to measure the atoms selected in Edit Mode"
        ),
        default="",
    )
    measure_draw: BoolProperty(name="Draw", default=True)
    measure_colour: FloatVectorProperty(
        name="Colour",
        subtype="COLOR",
        size=3,
        default=(1.0, 0.85, 0.2),
        min=0.0,
        max=1.0,
    )

    # -- labels ---------------------------------------------------------
    label_selection: StringProperty(name="Selection", default="ligand")
    label_template: StringProperty(
        name="Template",
        description="Fields: {chain} {resi} {resn} {one} {name} {elem} {b} {q}",
        default="{one}{resi}",
    )
    label_level: EnumProperty(
        name="Level",
        items=[
            ("residue", "Per Residue", "One label per residue"),
            ("atom", "Per Atom", "One label per atom"),
            ("selection", "Whole Selection", "A single label"),
        ],
        default="residue",
    )
    label_style: EnumProperty(
        name="Style",
        items=[
            ("text", "Text", "Plain 3D text"),
            ("card", "Card", "Text on a translucent backing plane"),
        ],
        default="text",
    )
    label_size: FloatProperty(name="Size", default=2.0, min=0.01, soft_max=20.0)
    label_billboard: BoolProperty(name="Face Camera", default=True)

    # -- colour ---------------------------------------------------------
    color_mode: EnumProperty(
        name="Colour By",
        items=[
            ("plddt", "AlphaFold pLDDT", "Confidence bands from the B-factor column"),
            ("bfactor", "B-factor", "Crystallographic B-factor"),
            ("csv", "CSV File", "Per-residue values from a CSV file"),
        ],
        default="plddt",
    )
    plddt_mode: EnumProperty(
        name="Mode",
        items=[
            ("banded", "Banded", "Four flat bands, as in the AlphaFold database"),
            ("continuous", "Continuous", "Smooth ramp between the band colours"),
        ],
        default="banded",
    )
    colormap: EnumProperty(name="Colormap", items=_colormap_items, default=0)
    color_selection: StringProperty(name="Selection", default="all")
    csv_path: StringProperty(name="CSV", subtype="FILE_PATH", default="")
    csv_value_column: StringProperty(name="Value Column", default="value")
    csv_resid_column: StringProperty(name="Residue Column", default="res_id")
    csv_chain_column: StringProperty(name="Chain Column", default="")

    # -- electrostatics ---------------------------------------------------
    apbs_map: StringProperty(
        name="Map",
        subtype="FILE_PATH",
        default="",
        description="An OpenDX potential map. Leave empty to run APBS",
    )
    apbs_forcefield: EnumProperty(
        name="Force Field",
        items=[
            ("AMBER", "AMBER", "AMBER charges and radii"),
            ("PARSE", "PARSE", "PARSE, tuned for solvation energies"),
            ("CHARMM", "CHARMM", "CHARMM charges and radii"),
            ("PEOEPB", "PEOEPB", "PEOE_PB charges"),
            ("SWANSON", "SWANSON", "Swanson et al. charges"),
            ("TYL06", "TYL06", "Tan, Yang and Luo charges"),
        ],
        default="AMBER",
    )
    apbs_solver: EnumProperty(
        name="Solver",
        items=[
            ("lpbe", "Linearised", "Linearised Poisson-Boltzmann; the usual choice"),
            ("npbe", "Non-linear", "Full equation; for strongly charged solutes"),
        ],
        default="lpbe",
    )
    apbs_ionic_strength: FloatProperty(
        name="Salt (M)",
        default=0.15,
        min=0.0,
        soft_max=1.0,
        description="Monovalent salt concentration in the solvent",
    )
    apbs_ramp: FloatProperty(
        name="Ramp (kT/e)",
        default=5.0,
        min=0.1,
        soft_max=25.0,
        description="Where the colour ramp saturates, either side of zero",
    )
    apbs_alpha: FloatProperty(
        name="Opacity",
        default=0.6,
        min=0.05,
        max=1.0,
        description="Surface opacity; below 1 lets what is inside show through",
    )


classes = (GalaSceneProperties,)


def register() -> None:
    """Register the property group and attach it to Scene."""
    register_classes(classes)
    bpy.types.Scene.gala = bpy.props.PointerProperty(type=GalaSceneProperties)


def unregister() -> None:
    """Detach and unregister the property group."""
    if hasattr(bpy.types.Scene, "gala"):
        # Another copy of the add-on may already have removed it.
        with contextlib.suppress(AttributeError, RuntimeError):
            del bpy.types.Scene.gala
    unregister_classes(classes)
