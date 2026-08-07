"""Scene setup: render settings, lighting, materials, camera and compositing.

Implements Objective 1. :func:`publication_setup` does everything at once;
every other function here is a single, separately usable step.
"""

from __future__ import annotations

from . import camera, compositing, lighting, materials, origin, presets, render, setup
from .camera import VIEWPOINTS, ensure_camera, frame_target, orbit
from .compositing import (
    add_file_output,
    clear_compositor,
    depth_cue,
    depth_of_field,
    enable_passes,
    set_exr_output,
    setup_compositor,
)
from .lighting import (
    STUDIO_HDRIS,
    THREE_POINT,
    LightSpec,
    clear_lighting,
    hdri_lighting,
    list_hdris,
    three_point_lighting,
)
from .materials import (
    MATERIAL_PRESETS,
    MATERIAL_SCHEMES,
    GalaMaterialSpec,
    assign_material,
    assign_materials,
    build_material,
    get_material,
)
from .origin import geometry_centre, set_origin_to_geometry
from .presets import PRESETS, RenderPreset, get_preset
from .render import (
    CausticsReport,
    GPUReport,
    enable_caustics,
    enable_gpu,
    set_resolution,
    set_transparent,
    setup_color_management,
    setup_render,
)
from .render import render as render_image
from .setup import SetupReport, publication_setup

__all__ = [
    "MATERIAL_PRESETS",
    "MATERIAL_SCHEMES",
    "PRESETS",
    "STUDIO_HDRIS",
    "THREE_POINT",
    "VIEWPOINTS",
    "CausticsReport",
    "GPUReport",
    "GalaMaterialSpec",
    "LightSpec",
    "RenderPreset",
    "SetupReport",
    "add_file_output",
    "assign_material",
    "assign_materials",
    "build_material",
    "camera",
    "clear_compositor",
    "clear_lighting",
    "compositing",
    "depth_cue",
    "depth_of_field",
    "enable_caustics",
    "enable_gpu",
    "enable_passes",
    "ensure_camera",
    "frame_target",
    "geometry_centre",
    "get_material",
    "get_preset",
    "hdri_lighting",
    "lighting",
    "list_hdris",
    "materials",
    "orbit",
    "origin",
    "presets",
    "publication_setup",
    "render",
    "render_image",
    "set_exr_output",
    "set_origin_to_geometry",
    "set_resolution",
    "set_transparent",
    "setup",
    "setup_color_management",
    "setup_compositor",
    "setup_render",
    "three_point_lighting",
]
