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
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..annotate import labels as gala_labels
from ..color import coloring
from ..core import attributes as gala_attributes
from ..core import collections as gala_collections
from ..core import interactive as gala_interactive
from ..core import mn as mn_bridge
from ..core import viewport
from ..core.entity import AtomStructure
from ..core.exceptions import GalaError
from ..core.registration import register_classes, unregister_classes
from ..core.selection import MACRO_KEYWORDS, PROPERTY_KEYWORDS
from ..electrostatics import grid as gala_grid
from ..electrostatics import surface as electrostatics
from ..interactions import detect
from ..interactions import draw as interaction_draw
from ..measure import draw as measure_draw
from ..measure import measurements
from ..scene import compositing, lighting, materials, origin, render, setup

__all__ = [
    "STYLE_ITEMS",
    "active_alias",
    "active_structure",
    "alias_of_object",
    "classes",
    "selected_atom_indices",
    "selection_word",
]

#: The Molecular Nodes styles worth offering for a named selection. Molecular
#: Nodes accepts more names than these, but the rest are aliases of the same
#: node (``vdw`` for ``spheres``) or are for density maps rather than atoms.
STYLE_ITEMS = (
    ("ball_and_stick", "Ball and Stick", "Spheres joined by bonds"),
    ("spheres", "Spheres", "Space-filling"),
    ("sticks", "Sticks", "Bonds only"),
    ("cartoon", "Cartoon", "Secondary-structure ribbon"),
    ("ribbon", "Ribbon", "Backbone trace"),
    ("surface", "Surface", "Molecular surface"),
)

#: Selection levels, as an operator and panel enum.
LEVEL_ITEMS = (
    ("atom", "Atom", "Just the atoms picked"),
    ("residue", "Residue", "Every atom of each residue touched"),
    ("chain", "Chain", "Every atom of each chain touched"),
    ("fragment", "Fragment", "Everything bonded to what was picked"),
)


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
    live = [m for m in molecules if _object_is_alive(_molecule_object(m))]
    if len(live) != 1:
        return None
    try:
        return AtomStructure.from_any(live[0])
    except Exception:
        # Every operator resolves the structure, including the ones that go on
        # without one, so a molecule that cannot be read must leave them with
        # nothing rather than with an exception.
        return None


def _molecule_object(molecule: Any) -> Any:
    """The Blender object behind a Molecular Nodes molecule, or ``None``.

    ``Molecule.object`` is a property that raises ``LinkedObjectError`` once
    the object it wraps has been deleted, and that is not an ``AttributeError``
    — so the default of a ``getattr`` never applies and the exception has to be
    caught here, or every button throws a traceback after one press of X in the
    outliner.
    """
    try:
        return molecule.object
    except Exception:
        return None


def _object_is_alive(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        return obj.name in bpy.data.objects
    except ReferenceError:
        return False


def selection_word(name: str) -> str:
    """How a stored selection has to be written to be read back as one.

    A name that is also a word in the selection language — ``protein``, ``all``,
    ``b`` — parses as that word, so the only way to reach the stored selection
    is PyMOL's explicit ``%name`` form. Saying so is the panel's job: the name
    was accepted, and the user has no way of knowing which words are taken.
    """
    if name in MACRO_KEYWORDS or name in PROPERTY_KEYWORDS:
        return f"%{name}"
    return name


def selected_atom_indices(
    context: Any, structure: AtomStructure | None = None
) -> list[int]:
    """Return the indices of the atoms the user has selected.

    Works in both Edit Mode (via bmesh) and Object Mode (via the stored
    selection flags), so the measurement operator behaves like PyMOL's wizard:
    click atoms, then measure.

    Parameters
    ----------
    context : bpy.types.Context
        The context to read the active object from.
    structure : AtomStructure, optional
        The structure the caller is about to act on. The picking is then read
        from *its* object rather than from whatever happens to be active, which
        need not be the same thing: :func:`active_structure` falls back to the
        only molecule in the session, so a selection made on an unrelated mesh
        would otherwise be applied to the molecule's atoms.

    Raises
    ------
    ValueError
        If an index lands past the end of the structure, which is what a mesh
        edited to a different vertex count leaves behind.
    """
    obj = context.active_object if structure is None else structure.object
    if obj is None or obj.type != "MESH":
        return []

    if obj.mode == "EDIT":
        import bmesh

        mesh = bmesh.from_edit_mesh(obj.data)
        indices = [vertex.index for vertex in mesh.verts if vertex.select]
    else:
        vertices = obj.data.vertices
        flags = np.empty(len(vertices), dtype=bool)
        vertices.foreach_get("select", flags)
        indices = [int(i) for i in np.flatnonzero(flags)]

    if indices and structure is not None and max(indices) >= structure.n_atoms:
        raise ValueError(
            f"{obj.name!r} has vertices past atom {structure.n_atoms - 1}, the "
            "last of the structure: the mesh and the atoms are out of step. "
            "Undo the edit that changed the vertex count."
        )
    return indices


def active_alias(structure: AtomStructure | None) -> str | None:
    """The named selection highlighted in the panel's list, if any."""
    return alias_of_object(getattr(structure, "object", None))


def alias_of_object(obj: Any) -> str | None:
    """The named selection highlighted in the panel's list for ``obj``.

    The list is drawn over ``mesh.attributes`` — the attributes *are* the
    selections, so there is no second copy of the names to keep in step — and
    the active index is an index into that. An index that has drifted onto
    something which is not one of Gala's selections falls back to the first
    one, which is friendlier than doing nothing.
    """
    if obj is None:
        return None
    names = gala_attributes.registered(obj)
    if not names:
        return None

    mesh_attributes = getattr(getattr(obj, "data", None), "attributes", None)
    index = getattr(obj, "gala_selection_index", 0)
    if mesh_attributes is not None and 0 <= index < len(mesh_attributes):
        candidate = mesh_attributes[index].name
        if candidate in names:
            return str(candidate)
    return names[0]


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
        try:
            # Resolved whether or not it is required, and inside the try: the
            # scene-wide operators still want the molecule when there is one —
            # framing, origin and materials are all settings the same panel
            # offers — and resolving it is itself something that can fail once
            # the object behind a molecule has been deleted.
            structure = active_structure(context)
            if self.requires_structure and structure is None:
                self.report(
                    {"ERROR"},
                    "Select a molecule imported with Molecular Nodes first.",
                )
                return {"CANCELLED"}
            return self.run(context, structure)
        except GalaError as exc:
            return _report_error(self, exc)
        except (
            ValueError,
            TypeError,
            KeyError,
            # A mask read from a mesh whose vertex count has drifted from the
            # structure, or a label template with a positional field in it.
            IndexError,
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
        # Moving the origin rewrites the mesh vertices and shifts the object
        # transform to compensate. In Edit Mode the vertices are a stale copy
        # of what the user is looking at, so the write would be thrown away and
        # only the transform would survive: the molecule jumps by its own
        # centroid the moment the mode is toggled back.
        with viewport.object_mode(structure.object):
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
# Objective 2: interactive selection
# ---------------------------------------------------------------------------


def _selection_summary(structure: AtomStructure, mask: np.ndarray) -> str:
    """ "142 atoms in 8 residues", the way PyMOL counts what you picked."""
    atoms = int(mask.sum())
    if not atoms:
        return "nothing selected"
    keys = structure.context.residue_key
    residues = len(np.unique(keys[mask]))
    return (
        f"{atoms} atom{'s' * (atoms != 1)} in {residues} residue{'s' * (residues != 1)}"
    )


class GALA_OT_expand_selection(_GalaOperator):
    """Grow the selected atoms through space, then to whole residues or chains

    Select atoms in Edit Mode however you like — click, box, circle or
    lasso — then expand. Expanding again grows the result further, so
    residue then chain works the way PyMOL's selection levels do.

    With a distance, everything within that many angstrom comes in first and
    the level completes the residues it clipped: pick a ligand, expand by 6 at
    the residue level, and you have its binding site.
    """

    bl_idname = "gala.expand_selection"
    bl_label = "Expand Selection"

    level: EnumProperty(name="Level", items=LEVEL_ITEMS, default="residue")
    distance: FloatProperty(
        name="Expand By",
        description="Radius in angstrom to grow by first. Zero grows by level alone",
        default=0.0,
        min=0.0,
        soft_max=15.0,
    )

    def run(self, context, structure):
        before = structure.viewport_selection()
        if not before.any():
            self.report(
                {"ERROR"},
                "Nothing selected. Enter Edit Mode and select some atoms first.",
            )
            return {"CANCELLED"}

        after = gala_interactive.expand_viewport_selection(
            structure, self.level, self.distance
        )
        grew_by = f" within {self.distance:g} A" if self.distance else ""
        self.report(
            {"INFO"},
            f"{self.level.title()}{grew_by}: {int(before.sum())} -> "
            f"{_selection_summary(structure, after)}.",
        )
        return {"FINISHED"}


class GALA_OT_selection_to_text(_GalaOperator):
    """Write the selected atoms into the box below as a PyMOL selection"""

    bl_idname = "gala.selection_to_text"
    bl_label = "From Selection"

    def run(self, context, structure):
        mask = structure.viewport_selection()
        text = structure.describe(mask)
        context.scene.gala.selection_text = text
        if not mask.any():
            self.report({"WARNING"}, "Nothing is selected.")
            return {"FINISHED"}
        self.report({"INFO"}, f"{_selection_summary(structure, mask)}: {text}")
        return {"FINISHED"}


class GALA_OT_copy_selection_text(_GalaOperator):
    """Copy the selection string to the clipboard"""

    bl_idname = "gala.copy_selection_text"
    bl_label = "Copy"
    requires_structure = False

    def run(self, context, structure):
        text = context.scene.gala.selection_text.strip()
        if not text:
            self.report({"WARNING"}, "Nothing to copy.")
            return {"CANCELLED"}
        context.window_manager.clipboard = text
        self.report({"INFO"}, f"Copied {text!r}.")
        return {"FINISHED"}


class GALA_OT_text_to_selection(_GalaOperator):
    """Select the atoms the selection string matches

    The quickest way to see what a selection actually covers before using it
    in a figure.
    """

    bl_idname = "gala.text_to_selection"
    bl_label = "Select"

    def run(self, context, structure):
        text = context.scene.gala.selection_text.strip()
        if not text:
            self.report({"ERROR"}, "Type a selection first.")
            return {"CANCELLED"}

        mask = structure.set_viewport_selection(text)
        if structure.object.mode != "EDIT":
            self.report(
                {"INFO"},
                f"{_selection_summary(structure, mask)}. Enter Edit Mode to see it.",
            )
        else:
            self.report({"INFO"}, _selection_summary(structure, mask))
        return {"FINISHED"}


class GALA_OT_create_alias(_GalaOperator):
    """Store the selection under a name, so it can be styled and exported

    The name becomes a boolean attribute on the mesh, which is what
    Molecular Nodes reads when a style is limited to a selection and what a
    saved PyMOL session carries as a named selection.
    """

    bl_idname = "gala.create_alias"
    bl_label = "Store Selection"

    source: EnumProperty(
        name="From",
        items=(
            ("viewport", "Selected Atoms", "Whatever is selected in the viewport"),
            ("text", "Selection String", "The selection typed in the panel"),
        ),
        default="viewport",
    )

    def run(self, context, structure):
        props = context.scene.gala
        name = props.alias_name.strip()
        if not name:
            self.report({"ERROR"}, "Give the selection a name first.")
            return {"CANCELLED"}

        # The name box is where a user types 'res_id' without knowing that the
        # residue numbering is stored under that name, so the collision is
        # caught before anything is written. A warning rather than an error:
        # nothing has gone wrong, the name box still holds what was typed, and
        # the next thing to do is edit it.
        wanted = gala_attributes.safe_name(name)
        conflict = gala_attributes.attribute_conflict(structure.object, wanted)
        if conflict is not None:
            self.report({"WARNING"}, conflict)
            return {"CANCELLED"}
        replacing = wanted in gala_attributes.registered(structure.object)

        selection: Any = None
        if self.source == "text":
            selection = props.selection_text.strip()
            if not selection:
                self.report({"ERROR"}, "The selection string is empty.")
                return {"CANCELLED"}

        stored = gala_interactive.create_alias(structure, name, selection)
        mask = structure.alias(stored)

        # Point the list at what was just made, and offer the next name rather
        # than leaving the last one to be overwritten by accident.
        names = gala_attributes.registered(structure.object)
        mesh_attributes = structure.object.data.attributes
        for index, attribute in enumerate(mesh_attributes):
            if attribute.name == stored:
                structure.object.gala_selection_index = index
                break
        props.alias_name = _next_alias_name(names)

        # Storing over a name that was already in the list is a replacement,
        # and the count in the report is the only clue that the selection the
        # user had under that name is gone.
        self.report(
            {"INFO"},
            f"{'Replaced' if replacing else 'Stored'} {stored!r}: "
            f"{_selection_summary(structure, mask)}.",
        )
        return {"FINISHED"}


def _next_alias_name(existing: list[str]) -> str:
    """``sele``, then ``sele_1`` … — PyMOL's habit of never reusing a name."""
    if "sele" not in existing:
        return "sele"
    index = 1
    while f"sele_{index}" in existing:
        index += 1
    return f"sele_{index}"


class _AliasOperator(_GalaOperator):
    """Base for the operators that act on one stored selection.

    The property is ``alias`` rather than ``name`` because ``Operator.name``
    is already Blender's own read-only label for the operator.
    """

    alias: StringProperty(
        name="Name",
        description="The stored selection to act on. Empty means the active one",
        default="",
    )

    def resolve(self, structure: AtomStructure) -> str | None:
        return self.alias.strip() or active_alias(structure)


class GALA_OT_select_alias(_AliasOperator):
    """Select the atoms of the stored selection"""

    bl_idname = "gala.select_alias"
    bl_label = "Select"

    def run(self, context, structure):
        name = self.resolve(structure)
        if name is None:
            self.report({"ERROR"}, "There are no stored selections.")
            return {"CANCELLED"}
        mask = gala_interactive.select_alias(structure, name)
        self.report({"INFO"}, f"{name}: {_selection_summary(structure, mask)}.")
        return {"FINISHED"}


class GALA_OT_alias_boolean(_AliasOperator):
    """Combine the stored selection with what is selected in the viewport"""

    bl_idname = "gala.alias_boolean"
    bl_label = "Combine Selection"

    mode: EnumProperty(
        name="Mode",
        items=(
            ("union", "Add", "Add the selected atoms to the stored selection"),
            ("intersect", "Intersect", "Keep only atoms in both"),
            ("subtract", "Subtract", "Remove the selected atoms from it"),
        ),
        default="union",
    )

    def run(self, context, structure):
        name = self.resolve(structure)
        if name is None:
            self.report({"ERROR"}, "There are no stored selections.")
            return {"CANCELLED"}
        combined = gala_interactive.alias_combine(structure, name, mode=self.mode)
        self.report({"INFO"}, f"{name}: {_selection_summary(structure, combined)}.")
        return {"FINISHED"}


class GALA_OT_delete_alias(_AliasOperator):
    """Forget the stored selection

    Any style limited to it keeps pointing at the name and will show nothing,
    so remove the style too if you no longer want it.
    """

    bl_idname = "gala.delete_alias"
    bl_label = "Delete Selection"

    def run(self, context, structure):
        name = self.resolve(structure)
        if name is None:
            self.report({"ERROR"}, "There are no stored selections.")
            return {"CANCELLED"}
        if not gala_interactive.delete_alias(structure, name):
            self.report({"WARNING"}, f"There is no stored selection named {name!r}.")
            return {"CANCELLED"}
        structure.object.gala_selection_index = 0
        self.report({"INFO"}, f"Removed {name!r}.")
        return {"FINISHED"}


class GALA_OT_style_alias(_AliasOperator):
    """Add a Molecular Nodes style covering only the stored selection

    The existing styles are left alone: this adds a branch to the node tree
    limited to the selection, which is how a pocket gets sticks while the rest
    of the protein stays a cartoon.
    """

    bl_idname = "gala.style_alias"
    bl_label = "Apply Style"

    def run(self, context, structure):
        name = self.resolve(structure)
        if name is None:
            self.report({"ERROR"}, "There are no stored selections.")
            return {"CANCELLED"}

        props = context.scene.gala
        gala_interactive.style_alias(
            structure,
            name,
            style=props.alias_style,
            color=props.alias_color if props.alias_color != "none" else None,
        )
        self.report(
            {"INFO"},
            f"Added a {props.alias_style.replace('_', ' ')} style on {name!r}.",
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
            indices = selected_atom_indices(context, structure)
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
    lighting: EnumProperty(
        name="Lighting",
        description=(
            "Light the molecules once they are built. A session carries no "
            "lighting of its own, so without this the scene opens unlit"
        ),
        items=(
            ("three_point", "Three Point", "A studio rig sized to the molecule"),
            ("hdri", "HDRI", "Softer, more natural environment light"),
            ("both", "Both", "An HDRI as fill under the rig"),
            ("none", "None", "Leave the scene unlit"),
        ),
        default="three_point",
    )
    materials: BoolProperty(
        name="Materials",
        description=(
            "Assign Gala's materials to each style. They take their colour "
            "from the mesh, so the session's colours are kept"
        ),
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
                lighting=self.lighting,
                materials="chemistry" if self.materials else None,
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
    GALA_OT_expand_selection,
    GALA_OT_selection_to_text,
    GALA_OT_copy_selection_text,
    GALA_OT_text_to_selection,
    GALA_OT_create_alias,
    GALA_OT_select_alias,
    GALA_OT_alias_boolean,
    GALA_OT_delete_alias,
    GALA_OT_style_alias,
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
