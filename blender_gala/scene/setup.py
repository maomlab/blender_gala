"""The one-call publication setup.

``publication_setup`` is the function Objective 1 asks for: after loading a
molecule, get from "default cube scene with a protein in it" to "render this
and put it in a paper" in one call — while every step underneath stays
separately callable for people who only want part of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.entity import AtomStructure
from . import camera as gala_camera
from . import compositing, lighting, materials, origin, render
from .presets import RenderPreset, get_preset

__all__ = ["SetupReport", "publication_setup"]


@dataclass
class SetupReport:
    """What :func:`publication_setup` did.

    Returned rather than printed so the UI, a notebook and a script can each
    present it appropriately.

    Attributes
    ----------
    preset : str
        Render preset that was applied.
    resolution : tuple[int, int]
        Output resolution in pixels.
    gpu : str
        Outcome of GPU detection.
    lighting : str
        Lighting backend that was used.
    materials : dict[str, str]
        Style name to material preset.
    origin : str
        Origin method applied, or why it was skipped.
    passes : list[str]
        Render passes enabled.
    warnings : list[str]
        Anything the user should know about.
    """

    preset: str = ""
    resolution: tuple[int, int] = (0, 0)
    gpu: str = ""
    lighting: str = ""
    materials: dict[str, str] = field(default_factory=dict)
    origin: str = ""
    passes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Blender Gala publication setup ({self.preset})",
            f"  resolution : {self.resolution[0]} x {self.resolution[1]}",
            f"  gpu        : {self.gpu}",
            f"  lighting   : {self.lighting}",
            f"  origin     : {self.origin}",
            f"  materials  : {', '.join(f'{k}->{v}' for k, v in self.materials.items()) or 'default'}",
            f"  passes     : {', '.join(self.passes) or 'none'}",
        ]
        lines.extend(f"  warning    : {w}" for w in self.warnings)
        return "\n".join(lines)


def _no_subject_reason(target: Any, structure: AtomStructure | None) -> str:
    """Why there is no molecule to build the scene around, in the user's terms.

    "Skipped" on its own is not actionable: from the panel the user chose
    nothing and was told nothing, and the two ways of arriving here — nothing
    to work from, and something that turned out not to have a molecule behind
    it — need different things done about them.
    """
    if structure is not None:
        return "the structure given has no Blender object"
    if target is not None:
        return "the target could not be read as a structure"

    loaded = _molecules_in_scene()
    if loaded > 1:
        return (
            f"{loaded} molecules are loaded and none is active. Pass one, or "
            "select it in the viewport first"
        )
    if loaded == 1:
        return "pass the molecule, or select it in the viewport first"
    return "no molecule is loaded"


def _molecules_in_scene() -> int:
    """How many molecules Molecular Nodes is tracking, for the message above."""
    from ..core import mn as mn_bridge

    module = mn_bridge.get_mn()
    if module is None:
        return 0
    try:
        return len(module.session.get_session().molecules)
    except Exception:  # pragma: no cover - depends on MN being registered
        return 0


def publication_setup(
    target: Any = None,
    preset: str | RenderPreset = "figure",
    engine: str = "CYCLES",
    transparent: bool = True,
    use_gpu: bool = True,
    lighting_style: str = "three_point",
    hdri: str = "studio",
    light_energy: float = 1.0,
    material_scheme: str | None = "chemistry",
    origin_method: str | None = "centroid",
    move_to_world_origin: bool = True,
    frame_camera: bool = True,
    viewpoint: str = "iso",
    cryptomatte: bool = True,
    depth_of_field: bool = False,
    view_transform: str = "Standard",
    scene: Any = None,
) -> SetupReport:
    """Configure the whole scene for a publication-quality still.

    The order matters and is deliberate: the origin is fixed first so that the
    lighting and camera, which are both sized from the molecule's bounding
    sphere, are computed against final coordinates.

    Parameters
    ----------
    target : AtomStructure, Molecule, bpy.types.Object, or None, optional
        The molecule the scene is being built around. ``None`` still configures
        the render, colour management and compositor, and lights the scene from
        the bounding box of everything visible.
    preset : str or RenderPreset, optional
        See :mod:`blender_gala.scene.presets`.
    engine : {"CYCLES", "EEVEE"}, optional
        Render engine.
    transparent : bool, optional
        Transparent film, so the figure drops onto any page background.
    use_gpu : bool, optional
        Attempt GPU rendering.
    lighting_style : {"three_point", "hdri", "both", "none"}, optional
        ``"three_point"`` gives controllable, figure-friendly light.
        ``"hdri"`` gives softer, more natural light. ``"both"`` uses an HDRI at
        low strength as fill under the rig, which is the most forgiving setup
        for glossy or metallic materials.
    hdri : str, optional
        HDRI name or path, used by the ``hdri`` and ``both`` styles.
    light_energy : float, optional
        Brightness multiplier for the three-point rig.
    material_scheme : str or None, optional
        Material scheme, or ``None`` to leave materials alone.
    origin_method : str or None, optional
        Origin method, or ``None`` to leave the origin alone.
    move_to_world_origin : bool, optional
        Move the molecule to the world origin after fixing its origin.
    frame_camera : bool, optional
        Create and aim a camera at the molecule.
    viewpoint : str, optional
        Camera viewpoint; see :data:`blender_gala.scene.camera.VIEWPOINTS`.
    cryptomatte : bool, optional
        Enable cryptomatte and depth passes and build the compositor chain.
    depth_of_field : bool, optional
        Enable camera depth of field focused on the molecule.
    view_transform : str, optional
        Colour management view transform. See SPECIFICATION D-8 for why this
        defaults to ``"Standard"``.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    SetupReport
        A summary; ``print(report)`` gives a readable block.

    Raises
    ------
    ValueError
        If ``lighting_style`` is unknown.
    """
    import bpy

    scene = scene or bpy.context.scene
    config = get_preset(preset)
    report = SetupReport(preset=config.name, resolution=config.resolution)

    if lighting_style not in ("three_point", "hdri", "both", "none"):
        raise ValueError(
            "lighting_style must be 'three_point', 'hdri', 'both' or 'none', "
            f"got {lighting_style!r}"
        )

    structure = None
    if target is not None:
        try:
            structure = AtomStructure.from_any(target)
        except Exception as exc:
            report.warnings.append(f"could not read the target as a structure: {exc}")

    # 1. Origin first: everything sized from the bounding sphere depends on it.
    if (
        origin_method is not None
        and structure is not None
        and structure.object is not None
    ):
        try:
            origin.set_origin_to_geometry(
                structure,
                method=origin_method,
                move_to_world_origin=move_to_world_origin,
            )
            report.origin = origin_method
        except Exception as exc:
            report.origin = "skipped"
            report.warnings.append(f"origin was not changed: {exc}")
    elif origin_method is None:
        report.origin = "not requested"
    else:
        report.origin = "skipped (no object)"
        # `origin` records this, and the operator surfaces `warnings` and
        # nothing else — so a scene with two molecules and neither of them
        # active was set up with no materials, no origin and not a word about
        # either. What was skipped and why has to reach the user.
        report.warnings.append(
            "no molecule to work from, so the origin was left where it was: "
            + _no_subject_reason(target, structure)
        )

    # 2. Render engine, sampling, GPU and colour management.
    gpu = render.setup_render(
        preset=config,
        engine=engine,
        transparent=transparent,
        use_gpu=use_gpu,
        scene=scene,
    )
    report.gpu = gpu.message
    render.setup_color_management(view_transform=view_transform, scene=scene)

    # 3. Lighting.
    if lighting_style in ("three_point", "both"):
        lighting.three_point_lighting(
            structure or target, energy=light_energy, scene=scene
        )
    if lighting_style in ("hdri", "both"):
        try:
            lighting.hdri_lighting(
                hdri,
                strength=0.3 if lighting_style == "both" else 1.0,
                visible_to_camera=not transparent,
                scene=scene,
            )
        except FileNotFoundError as exc:
            report.warnings.append(str(exc))
    report.lighting = lighting_style

    # 4. Materials.
    if material_scheme is not None and structure is not None:
        try:
            report.materials = materials.assign_materials(
                structure.molecule or structure.object, scheme=material_scheme
            )
        except Exception as exc:
            report.warnings.append(f"materials were not assigned: {exc}")
    elif material_scheme is not None:
        report.warnings.append(
            f"no molecule to assign the {material_scheme!r} materials to: "
            + _no_subject_reason(target, structure)
        )

    # 5. Camera.
    if frame_camera:
        gala_camera.frame_target(structure or target, viewpoint=viewpoint, scene=scene)

    # 6. Passes and compositing.
    if cryptomatte:
        report.passes = compositing.enable_passes(
            view_layer=compositing.scene_view_layer(scene)
        )
        compositing.setup_compositor(cryptomatte=True, scene=scene)

    if depth_of_field:
        try:
            compositing.depth_of_field(structure or target, scene=scene)
        except RuntimeError as exc:
            report.warnings.append(str(exc))

    return report
