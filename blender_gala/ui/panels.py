"""Sidebar panels, under a ``Gala`` tab in the 3D View.

Grouped to match the two objectives: scene setup on top, then the measuring
and annotation tools.
"""

from __future__ import annotations

from typing import Any

from bpy.types import Panel, UIList

from ..core import attributes as gala_attributes
from ..core import mn as mn_bridge
from ..core.registration import register_classes, unregister_classes
from ..ops import operators

__all__ = ["classes"]

_CATEGORY = "Gala"


class _GalaPanel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY

    @classmethod
    def poll(cls, context: Any) -> bool:
        return context.scene is not None


class GALA_PT_scene(_GalaPanel):
    """Top-level panel: the one-click setup plus its ingredients."""

    bl_idname = "GALA_PT_scene"
    bl_label = "Scene Setup"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        if not mn_bridge.available():
            box = layout.box()
            box.alert = True
            box.label(text="Molecular Nodes not found", icon="ERROR")
            box.label(text="Scene tools work; molecule tools need it.")

        column = layout.column(align=True)
        column.scale_y = 1.4
        column.operator("gala.publication_setup", icon="SHADERFX")

        layout.separator()
        column = layout.column(align=True)
        column.prop(props, "preset")
        column.prop(props, "engine")
        column.prop(props, "view_transform")

        row = layout.row(align=True)
        row.prop(props, "transparent", toggle=True)
        row.prop(props, "use_gpu", toggle=True)

        layout.operator("gala.setup_render", icon="RESTRICT_RENDER_OFF")


class GALA_PT_origin(_GalaPanel):
    bl_idname = "GALA_PT_origin"
    bl_parent_id = "GALA_PT_scene"
    bl_label = "Origin and Camera"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "origin_method")
        column.prop(props, "move_to_world_origin")
        column.operator("gala.set_origin", icon="OBJECT_ORIGIN")

        layout.separator()
        column = layout.column(align=True)
        column.prop(props, "viewpoint")
        column.prop(props, "frame_camera")


class GALA_PT_lighting(_GalaPanel):
    bl_idname = "GALA_PT_lighting"
    bl_parent_id = "GALA_PT_scene"
    bl_label = "Lighting and Materials"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "lighting_style")
        if props.lighting_style in ("hdri", "both"):
            column.prop(props, "hdri")
        if props.lighting_style in ("three_point", "both"):
            column.prop(props, "light_energy")
            column.prop(props, "light_softness")
            column.prop(props, "light_rotation")
        column.operator("gala.lighting", icon="LIGHT_AREA")

        layout.separator()
        column = layout.column(align=True)
        column.prop(props, "material_scheme")
        column.operator("gala.assign_materials", icon="MATERIAL")


class GALA_PT_compositing(_GalaPanel):
    bl_idname = "GALA_PT_compositing"
    bl_parent_id = "GALA_PT_scene"
    bl_label = "Passes and Compositing"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "cryptomatte")
        column.prop(props, "exr_directory")

        layout.separator()
        column = layout.column(align=True)
        column.prop(props, "depth_of_field")
        sub = column.row()
        sub.enabled = props.depth_of_field
        sub.prop(props, "dof_fstop")

        column = layout.column(align=True)
        column.prop(props, "depth_cue")
        sub = column.row(align=True)
        sub.enabled = props.depth_cue
        sub.prop(props, "depth_cue_near")
        sub.prop(props, "depth_cue_far")

        layout.operator("gala.setup_compositor", icon="NODE_COMPOSITING")
        layout.operator("gala.render", icon="RENDER_STILL")


class GALA_UL_selections(UIList):
    """The stored selections on the active molecule.

    Drawn over ``mesh.attributes`` rather than over a list of Gala's own: a
    stored selection *is* a boolean attribute, so listing the attributes
    directly means the panel can never disagree with what the mesh carries.
    Everything that is not one of Gala's is filtered out — Molecular Nodes
    keeps a dozen booleans of its own on the same mesh.
    """

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_prop, index
    ):
        row = layout.row(align=True)
        row.label(text=item.name, icon="RESTRICT_SELECT_OFF")

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        stored = set(gala_attributes.registered(context.active_object))
        query = self.filter_name.lower()

        flags = [0] * len(items)
        for index, item in enumerate(items):
            if item.name not in stored:
                continue
            if query and (query in item.name.lower()) is self.use_filter_invert:
                continue
            flags[index] |= self.bitflag_filter_item
        return flags, []


class GALA_PT_selection(_GalaPanel):
    """Pick atoms in Edit Mode, then grow, read out or name what you picked."""

    bl_idname = "GALA_PT_selection"
    bl_label = "Selection"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala
        obj = context.active_object

        if obj is None or obj.mode != "EDIT":
            layout.label(text="Tab into Edit Mode to pick atoms", icon="INFO")

        column = layout.column(align=True)
        column.label(text="Expand to:")
        column.row(align=True).prop(props, "selection_level", expand=True)
        expand = column.operator("gala.expand_selection", icon="SELECT_EXTEND")
        expand.level = props.selection_level

        layout.separator()
        column = layout.column(align=True)
        column.label(text="Syntax:")
        column.prop(props, "selection_text", text="")
        row = column.row(align=True)
        row.operator("gala.selection_to_text", icon="FILE_REFRESH")
        row.operator("gala.copy_selection_text", text="", icon="COPYDOWN")
        row.operator("gala.text_to_selection", icon="RESTRICT_SELECT_OFF")


class GALA_PT_named_selections(_GalaPanel):
    """The stored selections, and what to do with them."""

    bl_idname = "GALA_PT_named_selections"
    bl_parent_id = "GALA_PT_selection"
    bl_label = "Stored Selections"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala
        obj = context.active_object
        mesh = obj.data if obj is not None and obj.type == "MESH" else None

        row = layout.row(align=True)
        row.prop(props, "alias_name", text="")
        row.operator("gala.create_alias", text="", icon="ADD").source = "viewport"
        row.operator("gala.create_alias", text="", icon="SYNTAX_ON").source = "text"

        if mesh is None:
            layout.label(text="Select a molecule", icon="INFO")
            return

        layout.template_list(
            "GALA_UL_selections",
            "",
            mesh,
            "attributes",
            obj,
            "gala_selection_index",
            rows=3,
        )

        column = layout.column(align=True)
        column.enabled = bool(gala_attributes.registered(obj))

        # Written out rather than looped: the icon argument is a literal enum,
        # so a variable carrying the name is not a valid icon as far as the
        # type checker is concerned — and a literal at the call site is what
        # the test that verifies every icon exists looks for.
        row = column.row(align=True)
        row.operator("gala.select_alias", icon="RESTRICT_SELECT_OFF")
        row.operator("gala.alias_boolean", text="", icon="SELECT_EXTEND").mode = "union"
        row.operator(
            "gala.alias_boolean", text="", icon="SELECT_INTERSECT"
        ).mode = "intersect"
        row.operator(
            "gala.alias_boolean", text="", icon="SELECT_SUBTRACT"
        ).mode = "subtract"
        row.operator("gala.delete_alias", text="", icon="X")

        column.separator()
        row = column.row(align=True)
        row.prop(props, "alias_style", text="")
        row.prop(props, "alias_color", text="")
        column.operator("gala.style_alias", icon="SHADING_RENDERED")

        # A stored selection is a word in the selection language, which is not
        # something the panel would otherwise say anywhere.
        active = operators.alias_of_object(obj)
        if active:
            layout.label(text=f'Usable in selections: "{active}"', icon="SYNTAX_ON")


class GALA_PT_interactions(_GalaPanel):
    bl_idname = "GALA_PT_interactions"
    bl_label = "Interactions"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "selection_a")
        column.prop(props, "selection_b")

        layout.label(text="Types:")
        grid = layout.grid_flow(columns=2, even_columns=True, align=True)
        grid.prop(props, "interaction_kinds", expand=True)

        column = layout.column(align=True)
        column.prop(props, "exclude_same_residue")
        column.prop(props, "interaction_labels")

        row = layout.row(align=True)
        row.operator("gala.find_interactions", icon="LINKED")
        row.operator("gala.clear_interactions", text="", icon="TRASH")


class GALA_PT_measure(_GalaPanel):
    bl_idname = "GALA_PT_measure"
    bl_label = "Measure"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        layout.label(text="Select 2-4 atoms in Edit Mode,", icon="INFO")
        layout.label(text="or type selections below.")

        column = layout.column(align=True)
        column.prop(props, "measure_selection")
        column.prop(props, "measure_draw")
        column.prop(props, "measure_colour")

        row = layout.row(align=True)
        row.operator("gala.measure", icon="DRIVER_DISTANCE")
        row.operator("gala.clear_measurements", text="", icon="TRASH")


class GALA_PT_label(_GalaPanel):
    bl_idname = "GALA_PT_label"
    bl_label = "Label"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "label_selection")
        column.prop(props, "label_template")
        column.prop(props, "label_level")
        column.prop(props, "label_style")
        column.prop(props, "label_size")
        column.prop(props, "label_billboard")

        row = layout.row(align=True)
        row.operator("gala.label", icon="FONT_DATA")
        row.operator("gala.clear_labels", text="", icon="TRASH")


class GALA_PT_color(_GalaPanel):
    bl_idname = "GALA_PT_color"
    bl_label = "Colour"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "color_mode")
        column.prop(props, "color_selection")

        if props.color_mode == "plddt":
            column.prop(props, "plddt_mode")
        else:
            column.prop(props, "colormap")

        if props.color_mode == "csv":
            box = layout.box()
            box.prop(props, "csv_path")
            box.prop(props, "csv_value_column")
            box.prop(props, "csv_resid_column")
            box.prop(props, "csv_chain_column")

        layout.operator("gala.color", icon="COLOR")


class GALA_PT_electrostatics(_GalaPanel):
    bl_idname = "GALA_PT_electrostatics"
    bl_label = "Electrostatics"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gala

        column = layout.column(align=True)
        column.prop(props, "apbs_map")
        if not props.apbs_map:
            column.prop(props, "apbs_forcefield")
            column.prop(props, "apbs_solver")
            column.prop(props, "apbs_ionic_strength")

        column = layout.column(align=True)
        column.prop(props, "apbs_ramp")
        column.prop(props, "apbs_alpha")

        layout.operator("gala.electrostatic_surface", icon="OUTLINER_OB_FORCE_FIELD")


class GALA_PT_pymol(_GalaPanel):
    """Sessions in and out. Both open a file browser with their own options."""

    bl_idname = "GALA_PT_pymol"
    bl_label = "PyMOL Session"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)
        column.operator("gala.load_pymol_session", icon="IMPORT")
        column.operator("gala.save_pymol_session", icon="EXPORT")


class GALA_PT_cleanup(_GalaPanel):
    bl_idname = "GALA_PT_cleanup"
    bl_label = "Clean Up"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.operator("gala.clear_all", icon="TRASH")


classes = (
    GALA_PT_scene,
    GALA_PT_origin,
    GALA_PT_lighting,
    GALA_PT_compositing,
    GALA_UL_selections,
    GALA_PT_selection,
    GALA_PT_named_selections,
    GALA_PT_interactions,
    GALA_PT_measure,
    GALA_PT_label,
    GALA_PT_color,
    GALA_PT_electrostatics,
    GALA_PT_pymol,
    GALA_PT_cleanup,
)


def register() -> None:
    """Register every panel."""
    register_classes(classes)


def unregister() -> None:
    """Unregister every panel."""
    unregister_classes(classes)
