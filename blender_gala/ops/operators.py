"""Blender operators.

Each operator does three things and no more: find the active structure,
validate the inputs, and call one function from the Python API
(SPECIFICATION D-21). Keeping the logic out of the operators is what makes the
behaviour testable without simulating UI events.
"""

from __future__ import annotations

import os
from typing import Any

import bpy
import numpy as np
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..annotate import labels as gala_labels
from ..color import coloring
from ..core import collections as gala_collections
from ..core import mn as mn_bridge
from ..core.entity import AtomStructure
from ..core.exceptions import GalaError
from ..core.registration import register_classes, unregister_classes
from ..electrostatics import grid as gala_grid
from ..electrostatics import surface as electrostatics
from ..interactions import detect
from ..interactions import draw as interaction_draw
from ..measure import draw as measure_draw
from ..measure import measurements
from ..scene import compositing, lighting, materials, origin, render, setup

__all__ = ["active_structure", "classes", "selected_atom_indices"]


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def active_structure(context: Any) -> AtomStructure | None:
    """Return the active object as an :class:`AtomStructure`, if possible.

    Falls back to the only Molecular Nodes molecule in the session when
    nothing suitable is active — the common case where a user has just
    imported one structure and then clicked in the sidebar.
    """
    obj = context.active_object
    if obj is not None and obj.type == "MESH":
        try:
            return AtomStructure.from_any(obj)
        except Exception:
            pass

    module = mn_bridge.get_mn()
    if module is None:
        return None
    try:
        molecules = list(module.session.get_session().molecules.values())
    except Exception:
        return None

    # The session outlives the objects it tracks, so entries whose object has
    # been deleted have to be filtered out or the panel acts on a ghost.
    live = [m for m in molecules if _object_is_alive(getattr(m, "object", None))]
    if len(live) == 1:
        return AtomStructure.from_any(live[0])
    return None


def _object_is_alive(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        return obj.name in bpy.data.objects
    except ReferenceError:
        return False


def selected_atom_indices(context: Any) -> list[int]:
    """Return the indices of the atoms the user has selected.

    Works in both Edit Mode (via bmesh) and Object Mode (via the stored
    selection flags), so the measurement operator behaves like PyMOL's wizard:
    click atoms, then measure.
    """
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return []

    if obj.mode == "EDIT":
        import bmesh

        mesh = bmesh.from_edit_mesh(obj.data)
        return [vertex.index for vertex in mesh.verts if vertex.select]

    vertices = obj.data.vertices
    flags = np.empty(len(vertices), dtype=bool)
    vertices.foreach_get("select", flags)
    return [int(i) for i in np.flatnonzero(flags)]


def _report_error(operator: Operator, exc: Exception) -> set[str]:
    operator.report({"ERROR"}, str(exc))
    return {"CANCELLED"}


class _GalaOperator(Operator):
    """Base class: needs a structure, reports Gala errors cleanly."""

    bl_options = {"REGISTER", "UNDO"}

    requires_structure = True

    @classmethod
    def poll(cls, context: Any) -> bool:
        return context.scene is not None

    def execute(self, context: Any) -> set[str]:
        structure = active_structure(context) if self.requires_structure else None
        if self.requires_structure and structure is None:
            self.report(
                {"ERROR"},
                "Select a molecule imported with Molecular Nodes first.",
            )
            return {"CANCELLED"}
        try:
            return self.run(context, structure)
        except GalaError as exc:
            return _report_error(self, exc)
        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            FileNotFoundError,
        ) as exc:
            return _report_error(self, exc)

    def run(self, context: Any, structure: AtomStructure | None) -> set[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Objective 1: scene setup
# ---------------------------------------------------------------------------


class GALA_OT_publication_setup(_GalaOperator):
    """Set up the whole scene for a publication-quality still"""

    bl_idname = "gala.publication_setup"
    bl_label = "Publication Setup"
    requires_structure = False

    def run(self, context, structure):
        props = context.scene.gala
        report = setup.publication_setup(
            structure,
            preset=props.preset,
            engine=props.engine,
            transparent=props.transparent,
            use_gpu=props.use_gpu,
            lighting_style=props.lighting_style,
            hdri=props.hdri,
            light_energy=props.light_energy,
            material_scheme=props.material_scheme,
            origin_method=None
            if props.origin_method == "none"
            else props.origin_method,
            move_to_world_origin=props.move_to_world_origin,
            frame_camera=props.frame_camera,
            viewpoint=props.viewpoint,
            cryptomatte=props.cryptomatte,
            depth_of_field=props.depth_of_field,
            view_transform=props.view_transform,
            scene=context.scene,
        )
        for warning in report.warnings:
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, f"{report.preset} preset. {report.gpu}")
        return {"FINISHED"}


class GALA_OT_setup_render(_GalaOperator):
    """Apply the render preset, colour management and GPU settings"""

    bl_idname = "gala.setup_render"
    bl_label = "Apply Render Settings"
    requires_structure = False

    def run(self, context, structure):
        props = context.scene.gala
        gpu = render.setup_render(
            preset=props.preset,
            engine=props.engine,
            transparent=props.transparent,
            use_gpu=props.use_gpu,
            scene=context.scene,
        )
        render.setup_color_management(
            view_transform=props.view_transform, scene=context.scene
        )
        self.report({"INFO"}, gpu.message)
        return {"FINISHED"}


class GALA_OT_lighting(_GalaOperator):
    """Build the lighting rig"""

    bl_idname = "gala.lighting"
    bl_label = "Build Lighting"
    requires_structure = False

    def run(self, context, structure):
        props = context.scene.gala
        if props.lighting_style in ("three_point", "both"):
            lighting.three_point_lighting(
                structure,
                energy=props.light_energy,
                softness=props.light_softness,
                rotation=props.light_rotation,
                scene=context.scene,
            )
        if props.lighting_style in ("hdri", "both"):
            lighting.hdri_lighting(
                props.hdri,
                strength=0.3 if props.lighting_style == "both" else 1.0,
                visible_to_camera=not props.transparent,
                scene=context.scene,
            )
        if props.lighting_style == "none":
            lighting.clear_lighting(scene=context.scene)
        return {"FINISHED"}


class GALA_OT_assign_materials(_GalaOperator):
    """Assign Gala materials to the molecule's styles"""

    bl_idname = "gala.assign_materials"
    bl_label = "Assign Materials"

    def run(self, context, structure):
        assigned = materials.assign_materials(
            structure.molecule or structure.object,
            scheme=context.scene.gala.material_scheme,
        )
        if assigned:
            summary = ", ".join(f"{k} -> {v}" for k, v in assigned.items())
            self.report({"INFO"}, summary)
        else:
            self.report(
                {"WARNING"},
                "No Molecular Nodes styles found; assigned a material slot instead.",
            )
        return {"FINISHED"}


class GALA_OT_set_origin(_GalaOperator):
    """Move the object origin onto the molecule's geometry"""

    bl_idname = "gala.set_origin"
    bl_label = "Origin to Geometry"

    def run(self, context, structure):
        props = context.scene.gala
        if props.origin_method == "none":
            self.report({"INFO"}, "Origin method is set to 'Leave Alone'.")
            return {"CANCELLED"}
        origin.set_origin_to_geometry(
            structure,
            method=props.origin_method,
            move_to_world_origin=props.move_to_world_origin,
        )
        return {"FINISHED"}


class GALA_OT_setup_compositor(_GalaOperator):
    """Enable render passes and build the compositing chain"""

    bl_idname = "gala.setup_compositor"
    bl_label = "Set Up Compositor"
    requires_structure = False

    def run(self, context, structure):
        props = context.scene.gala
        depth_range = (
            (props.depth_cue_near, props.depth_cue_far) if props.depth_cue else None
        )
        compositing.setup_compositor(
            cryptomatte=props.cryptomatte,
            dof=False,
            depth_cue_range=depth_range,
            file_output=bpy.path.abspath(props.exr_directory) or None,
            scene=context.scene,
        )
        if props.depth_of_field:
            compositing.depth_of_field(
                structure, fstop=props.dof_fstop, scene=context.scene
            )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Objective 2: interactions
# ---------------------------------------------------------------------------


class GALA_OT_find_interactions(_GalaOperator):
    """Detect interactions between two selections and draw them"""

    bl_idname = "gala.find_interactions"
    bl_label = "Find Interactions"

    def run(self, context, structure):
        props = context.scene.gala
        kinds = tuple(props.interaction_kinds)
        if not kinds:
            self.report({"ERROR"}, "Select at least one interaction type.")
            return {"CANCELLED"}

        found = detect.find_interactions(
            structure,
            props.selection_a,
            props.selection_b,
            kinds=kinds,
            exclude_same_residue=props.exclude_same_residue,
        )
        if not found:
            self.report({"WARNING"}, "No interactions matched those criteria.")
            return {"CANCELLED"}

        interaction_draw.draw_interactions(
            found, target=structure, label=props.interaction_labels
        )
        counts: dict[str, int] = {}
        for item in found:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        self.report(
            {"INFO"},
            "Drew " + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())),
        )
        return {"FINISHED"}


class GALA_OT_clear_interactions(_GalaOperator):
    """Remove every drawn interaction"""

    bl_idname = "gala.clear_interactions"
    bl_label = "Clear Interactions"
    requires_structure = False

    def run(self, context, structure):
        removed = interaction_draw.clear_interactions()
        self.report({"INFO"}, f"Removed {removed} objects.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Objective 2: measurement
# ---------------------------------------------------------------------------


class GALA_OT_measure(_GalaOperator):
    """Measure a distance, angle or dihedral

    Uses the atoms selected in Edit Mode, or the selection strings typed in
    the panel.
    """

    bl_idname = "gala.measure"
    bl_label = "Measure"

    def run(self, context, structure):
        props = context.scene.gala

        text = props.measure_selection.strip()
        if text:
            selections: list[Any] = [
                part.strip() for part in text.split(";") if part.strip()
            ]
        else:
            indices = selected_atom_indices(context)
            if not 2 <= len(indices) <= 4:
                self.report(
                    {"ERROR"},
                    f"Select 2, 3 or 4 atoms (got {len(indices)}), or type "
                    "selection strings separated by ';'.",
                )
                return {"CANCELLED"}
            selections = [np.array([i]) for i in indices]

        if not 2 <= len(selections) <= 4:
            self.report(
                {"ERROR"},
                f"Give 2 (distance), 3 (angle) or 4 (dihedral) selections, "
                f"got {len(selections)}.",
            )
            return {"CANCELLED"}

        result = measurements.measure(
            structure,
            *selections,
            draw=props.measure_draw,
            colour=tuple(props.measure_colour),
        )
        self.report({"INFO"}, str(result))
        return {"FINISHED"}


class GALA_OT_clear_measurements(_GalaOperator):
    """Remove every drawn measurement"""

    bl_idname = "gala.clear_measurements"
    bl_label = "Clear Measurements"
    requires_structure = False

    def run(self, context, structure):
        removed = measure_draw.clear_measurements()
        self.report({"INFO"}, f"Removed {removed} objects.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Objective 2: labels
# ---------------------------------------------------------------------------


class GALA_OT_label(_GalaOperator):
    """Label the selected atoms or residues"""

    bl_idname = "gala.label"
    bl_label = "Add Labels"

    def run(self, context, structure):
        props = context.scene.gala
        created = gala_labels.label(
            structure,
            props.label_selection,
            template=props.label_template,
            level=props.label_level,
            style=props.label_style,
            size=props.label_size,
            billboard=props.label_billboard,
        )
        self.report({"INFO"}, f"Created {len(created)} label objects.")
        return {"FINISHED"}


class GALA_OT_clear_labels(_GalaOperator):
    """Remove every 3D label"""

    bl_idname = "gala.clear_labels"
    bl_label = "Clear Labels"
    requires_structure = False

    def run(self, context, structure):
        removed = gala_labels.clear_labels()
        self.report({"INFO"}, f"Removed {removed} objects.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Objective 2: colouring
# ---------------------------------------------------------------------------


class GALA_OT_color(_GalaOperator):
    """Colour the molecule from data"""

    bl_idname = "gala.color"
    bl_label = "Apply Colours"

    def run(self, context, structure):
        props = context.scene.gala

        if props.color_mode == "plddt":
            result = coloring.color_by_plddt(
                structure, props.color_selection, mode=props.plddt_mode
            )
        elif props.color_mode == "bfactor":
            result = coloring.color_by_bfactor(
                structure, props.color_selection, cmap=props.colormap
            )
        else:
            path = bpy.path.abspath(props.csv_path)
            if not path:
                self.report({"ERROR"}, "Choose a CSV file first.")
                return {"CANCELLED"}
            result = coloring.color_from_csv(
                structure,
                path,
                value_column=props.csv_value_column,
                res_id_column=props.csv_resid_column,
                chain_column=props.csv_chain_column or None,
                selection=props.color_selection,
                cmap=props.colormap,
            )

        self.report(
            {"INFO"},
            f"Coloured {result.n_colored} atoms "
            f"({result.vmin:.3g} to {result.vmax:.3g}).",
        )
        return {"FINISHED"}


class GALA_OT_electrostatic_surface(_GalaOperator):
    """Solve the Poisson-Boltzmann equation and paint it on the surface"""

    bl_idname = "gala.electrostatic_surface"
    bl_label = "Electrostatic Surface"

    def run(self, context, structure):
        props = context.scene.gala

        path = bpy.path.abspath(props.apbs_map) if props.apbs_map else ""
        grid = gala_grid.read_dx(path) if path else None

        # APBS is seconds on a small protein and minutes on a large one, and
        # it is a subprocess either way: the wait cursor is the only signal
        # the window can give while it runs.
        context.window.cursor_set("WAIT")
        try:
            result = electrostatics.electrostatic_surface(
                structure,
                grid=grid,
                ramp=props.apbs_ramp,
                alpha=props.apbs_alpha,
                forcefield=props.apbs_forcefield,
                solver=props.apbs_solver,
                ionic_strength=props.apbs_ionic_strength,
            )
        finally:
            context.window.cursor_set("DEFAULT")

        surface_values = result.potential[np.isfinite(result.potential)]
        self.report(
            {"INFO"},
            f"Surface potential {surface_values.min():+.1f} to "
            f"{surface_values.max():+.1f} kT/e over {surface_values.size} atoms; "
            f"ramp +/-{result.ramp:g}.",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# PyMOL sessions
# ---------------------------------------------------------------------------


class GALA_OT_load_pymol_session(_GalaOperator, ImportHelper):
    """Open a PyMOL session: molecules, representations, colours and the view"""

    bl_idname = "gala.load_pymol_session"
    bl_label = "Load PyMOL Session"
    requires_structure = False

    filename_ext = ".pse"
    filter_glob: StringProperty(
        default="*.pse;*.psw;*.pse.gz", options={"HIDDEN"}, maxlen=255
    )
    state: IntProperty(
        name="State",
        description="Which state to build for multi-state objects",
        default=1,
        min=1,
    )
    styles: BoolProperty(
        name="Representations",
        description="Apply the Molecular Nodes style matching each representation",
        default=True,
    )
    colors: BoolProperty(
        name="Colours",
        description="Carry the per-atom colours over",
        default=True,
    )
    camera: BoolProperty(
        name="Camera",
        description="Point the scene camera the way PyMOL was pointing",
        default=True,
    )
    annotations: BoolProperty(
        name="Measurements and labels",
        description="Recreate distance, angle and dihedral objects, and labels",
        default=True,
    )

    def run(self, context, structure):
        from ..pymol import load as pymol_load

        context.window.cursor_set("WAIT")
        try:
            result = pymol_load.load_session(
                self.filepath,
                # The dialog counts states from 1, the way PyMOL does.
                state=max(0, self.state - 1),
                styles=self.styles,
                colors=self.colors,
                camera=self.camera,
                measurements=self.annotations,
                labels=self.annotations,
            )
        finally:
            context.window.cursor_set("DEFAULT")

        atoms = sum(len(m.object.data.vertices) for m in result.molecules.values())
        message = f"Loaded {len(result.molecules)} object(s), {atoms} atoms."
        if result.skipped:
            message += f" Not converted: {'; '.join(result.skipped[:3])}"
        self.report({"WARNING"} if result.skipped else {"INFO"}, message)
        return {"FINISHED"}


class GALA_OT_save_pymol_session(_GalaOperator, ExportHelper):
    """Write the scene as a PyMOL session"""

    bl_idname = "gala.save_pymol_session"
    bl_label = "Save PyMOL Session"
    requires_structure = False

    filename_ext = ".pse"
    filter_glob: StringProperty(
        default="*.pse;*.pse.gz", options={"HIDDEN"}, maxlen=255
    )
    colors: BoolProperty(
        name="Colours",
        description="Carry the per-atom Color attribute over",
        default=True,
    )
    styles: BoolProperty(
        name="Representations",
        description="Turn Molecular Nodes styles into PyMOL representations",
        default=True,
    )
    camera: BoolProperty(
        name="Camera",
        description="Write the scene camera as the session's view",
        default=True,
    )
    annotations: BoolProperty(
        name="Measurements and labels",
        description="Include Gala measurements and labels",
        default=True,
    )

    def run(self, context, structure):
        from ..pymol import save as pymol_save

        result = pymol_save.save_session(
            self.filepath,
            colors=self.colors,
            styles=self.styles,
            camera=self.camera,
            measurements=self.annotations,
            labels=self.annotations,
        )
        if not result.session.molecules:
            self.report(
                {"WARNING"},
                "Nothing to write: no Molecular Nodes molecules in the scene.",
            )
            return {"CANCELLED"}

        atoms = sum(m.n_atoms for m in result.session.molecules)
        self.report(
            {"INFO"},
            f"Wrote {len(result.session.molecules)} object(s), {atoms} atoms to "
            f"{os.path.basename(result.path)}.",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


class GALA_OT_clear_all(_GalaOperator):
    """Remove everything Blender Gala has added to the scene"""

    bl_idname = "gala.clear_all"
    bl_label = "Clear All Gala Objects"
    requires_structure = False

    def run(self, context, structure):
        removed = gala_collections.clear(scene=context.scene)
        removed += compositing.clear_compositor(scene=context.scene)
        self.report({"INFO"}, f"Removed {removed} objects and nodes.")
        return {"FINISHED"}


class GALA_OT_render(_GalaOperator):
    """Render a still with the current settings"""

    bl_idname = "gala.render"
    bl_label = "Render Still"
    requires_structure = False

    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="//gala_render.png")

    def run(self, context, structure):
        path = render.render(bpy.path.abspath(self.filepath), scene=context.scene)
        self.report({"INFO"}, f"Rendered to {path}")
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


classes = (
    GALA_OT_publication_setup,
    GALA_OT_setup_render,
    GALA_OT_lighting,
    GALA_OT_assign_materials,
    GALA_OT_set_origin,
    GALA_OT_setup_compositor,
    GALA_OT_find_interactions,
    GALA_OT_clear_interactions,
    GALA_OT_measure,
    GALA_OT_clear_measurements,
    GALA_OT_label,
    GALA_OT_clear_labels,
    GALA_OT_color,
    GALA_OT_electrostatic_surface,
    GALA_OT_load_pymol_session,
    GALA_OT_save_pymol_session,
    GALA_OT_clear_all,
    GALA_OT_render,
)


def register() -> None:
    """Register every operator."""
    register_classes(classes)


def unregister() -> None:
    """Unregister every operator."""
    unregister_classes(classes)
