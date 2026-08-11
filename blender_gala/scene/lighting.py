"""Lighting rigs: three-point studio lighting and HDRI environments.

Gala ships its own three-point rig rather than driving Blender's bundled
``Tri-lighting`` add-on (SPECIFICATION D-9): that add-on is not enabled by
default, its operator needs a 3D View context so it cannot be scripted
headlessly, and it leaves behind three unrelated lights with no handle for
later adjustment. Gala's rig is parented to one empty, so rotating that empty
re-lights the whole scene.

Light power scales with the square of the molecule's radius, so the same rig
looks identical on a 20-residue peptide and on a ribosome.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import collections as gala_collections
from ..core.entity import AtomStructure

try:  # pragma: no cover
    import bpy
    from mathutils import Vector
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]
    Vector = None  # type: ignore[assignment,misc]

__all__ = [
    "STUDIO_HDRIS",
    "THREE_POINT",
    "LightSpec",
    "clear_lighting",
    "hdri_lighting",
    "list_hdris",
    "three_point_lighting",
]

RIG_NAME = "GALA Light Rig"


@dataclass(frozen=True)
class LightSpec:
    """One light of a rig, positioned in spherical coordinates.

    Attributes
    ----------
    name : str
        Suffix used for the light object's name.
    azimuth : float
        Horizontal angle in degrees, measured from the camera axis.
    elevation : float
        Vertical angle in degrees above the horizon.
    power : float
        Power relative to the key light.
    size : float
        Area light size as a multiple of the subject radius. Larger means
        softer shadows.
    distance : float
        Distance from the subject as a multiple of the subject radius.
    colour : tuple[float, float, float]
        Light colour. Slightly cool fill and warm key is a standard portrait
        trick that also helps molecular surfaces read as three-dimensional.
    """

    name: str
    azimuth: float
    elevation: float
    power: float
    size: float
    distance: float = 3.0
    colour: tuple[float, float, float] = (1.0, 1.0, 1.0)


#: The default three-point rig (SPECIFICATION §5.4).
THREE_POINT: tuple[LightSpec, ...] = (
    LightSpec(
        "Key",
        azimuth=45.0,
        elevation=30.0,
        power=1.0,
        size=1.5,
        colour=(1.0, 0.98, 0.95),
    ),
    LightSpec(
        "Fill",
        azimuth=-60.0,
        elevation=5.0,
        power=0.35,
        size=2.5,
        colour=(0.95, 0.97, 1.0),
    ),
    LightSpec(
        "Rim",
        azimuth=170.0,
        elevation=25.0,
        power=0.7,
        size=1.0,
        colour=(1.0, 1.0, 1.0),
    ),
)

#: Watts for the key light when the subject radius is 1 Blender unit (100 A)
#: and the light sits at the default distance of three radii.
#:
#: Calibrated by rendering the test structure across a range of values and
#: measuring clipping: 1000 W blows out almost every pixel, 40 W clips about a
#: third, and 12 W lands a matte protein around 0.45 mean luminance with no
#: clipping — bright enough to read, with headroom for a glossy ligand.
_BASE_POWER = 12.0

#: The largest power a light can hold: ``Light.energy`` is a 32-bit float, so a
#: value computed above this arrives as ``inf``. A 1e30 A coordinate is enough
#: to get there, and an infinitely bright light renders as a white frame.
_MAX_ENERGY = float(np.finfo(np.float32).max)

STUDIO_HDRIS = (
    "studio",
    "courtyard",
    "interior",
    "city",
    "forest",
    "night",
    "sunrise",
    "sunset",
)


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def _subject_bounds(target: Any, scene: Any) -> tuple[np.ndarray, float]:
    """Return the ``(centre, radius)`` in world Blender units for ``target``.

    Everything the scene layer sizes comes from these two numbers: where the
    camera stands, where its clip planes go, how much power each light gets.
    One non-finite coordinate makes both of them ``nan`` — and ``inf`` does
    too, since ``inf - inf`` is ``nan`` — which propagates into a camera
    transform of ``nan`` and lights of ``nan`` watts. That renders black, and
    a black frame says nothing about the atom that caused it, so the bounds
    are refused here instead of being passed on.
    """
    centre, radius = _bounds_of(target, scene)
    if not (np.isfinite(centre).all() and math.isfinite(radius)):
        from ..core.exceptions import GalaError

        raise GalaError(
            "the subject has non-finite coordinates: its bounds came out as "
            f"centre {tuple(float(v) for v in centre)}, radius {radius}. There "
            "is no camera distance or light power to derive from that; check "
            "the structure for nan or infinite atom positions."
        )
    return centre, radius


def _bounds_of(target: Any, scene: Any) -> tuple[np.ndarray, float]:
    """The unchecked ``(centre, radius)``; see :func:`_subject_bounds`."""
    if target is None:
        return _scene_bounds(scene)
    try:
        structure = AtomStructure.from_any(target)
    except Exception:
        structure = None
    if structure is not None and structure.n_atoms:
        return structure.bounding_sphere()
    if bpy is not None and isinstance(target, bpy.types.Object):
        return _object_bounds([target])
    return _scene_bounds(scene)


def _object_bounds(objects: Sequence[Any]) -> tuple[np.ndarray, float]:
    """Centre and radius of a set of objects, as they are actually drawn.

    Points come from the evaluated objects, so a light rig scaled from these
    is scaled to what geometry nodes built rather than to the mesh it was
    built from. The bounding-box corners are the fallback for anything that
    evaluates to no vertices at all.
    """
    from .camera import _object_points

    gathered: list[np.ndarray] = []
    for obj in objects:
        points = _object_points(obj)
        if points is None or not len(points):
            matrix = obj.matrix_world
            points = np.asarray([np.array(matrix @ Vector(c)) for c in obj.bound_box])
        gathered.append(np.asarray(points))
    if not gathered:
        return np.zeros(3), 1.0
    points = np.vstack(gathered)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = float(np.linalg.norm(points - centre, axis=1).max())
    return centre, max(radius, 1e-3)


def _scene_bounds(scene: Any) -> tuple[np.ndarray, float]:
    meshes = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and not obj.get("gala") and obj.visible_get()
    ]
    if not meshes:
        return np.zeros(3), 1.0
    return _object_bounds(meshes)


def _as_float(value: Any) -> float | None:
    """``value`` as a float, or ``None`` if it is not a number at all."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: Every numeric field of a :class:`LightSpec`, all of which reach ``bpy``.
_SPEC_NUMBERS = ("azimuth", "elevation", "power", "size", "distance")


def _validate_rig(
    specs: Sequence[LightSpec],
    radius: float,
    energy: float,
    distance: float,
    softness: float,
) -> None:
    """Refuse a rig that could only fail partway through building itself.

    :func:`three_point_lighting` clears the old rig before it builds the new
    one, and a :class:`LightSpec` is otherwise only validated by ``bpy`` at the
    moment it is applied — so a colour of the wrong length, or a size that
    works out as ``nan``, leaves the scene holding whichever lights happened to
    come first and no way to tell that from a rig that was never built. The
    whole rig is therefore checked here, before anything is removed.

    Raises
    ------
    ValueError
        If a spec, or the power it works out at, cannot be applied.
    """
    for label, value in (("energy", energy), ("softness", softness)):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value}")

    for index, spec in enumerate(specs):
        where = f"light {getattr(spec, 'name', index)!r}"

        for field in _SPEC_NUMBERS:
            raw = getattr(spec, field, None)
            number = _as_float(raw)
            if number is None or not math.isfinite(number):
                raise ValueError(
                    f"{where} has a {field} of {raw!r}; it must be a finite number"
                )

        try:
            colour = [float(channel) for channel in spec.colour]
        except TypeError:
            colour = []
        if len(colour) != 3 or not all(math.isfinite(c) for c in colour):
            raise ValueError(
                f"{where} has a colour of {spec.colour!r}; it must be three "
                "finite numbers, as (red, green, blue)"
            )

        watts = _BASE_POWER * float(spec.power) * energy * (radius * distance) ** 2
        if not math.isfinite(watts) or abs(watts) > _MAX_ENERGY:
            raise ValueError(
                f"{where} works out at {watts} W around a subject of radius "
                f"{radius}, which is more than a light can hold; the subject is "
                "too large, or its coordinates are not in angstrom"
            )


def three_point_lighting(
    target: Any = None,
    energy: float = 1.0,
    distance: float = 3.0,
    softness: float = 1.0,
    specs: Sequence[LightSpec] = THREE_POINT,
    rotation: float = 0.0,
    visible_to_camera: bool = False,
    backend: str = "gala",
    scene: Any = None,
) -> Any:
    """Build a three-point studio rig around a subject.

    Parameters
    ----------
    target : AtomStructure, Molecule, bpy.types.Object, or None, optional
        Subject to light. ``None`` uses the bounding box of every visible mesh
        in the scene, which is the right behaviour when several molecules make
        up one figure.
    energy : float, optional
        Overall brightness multiplier. Power is derived from the subject radius
        so this is a taste knob, not a calibration knob.
    distance : float, optional
        Light distance as a multiple of the subject radius.
    softness : float, optional
        Multiplier on light size. Above 1 gives softer, more diffuse shadows;
        below 1 gives crisper ones that emphasise surface detail.
    specs : sequence of LightSpec, optional
        Override the rig layout entirely.
    rotation : float, optional
        Rotate the whole rig about the vertical axis, in degrees.
    visible_to_camera : bool, optional
        Whether the lights themselves appear in the render. Off by default:
        an area light is an emitting surface, and the rim light sits directly
        behind the subject, so leaving it visible puts a white disk across the
        background of every figure.
    backend : {"gala", "tri_lighting"}, optional
        ``"tri_lighting"`` delegates to Blender's bundled add-on if it is
        enabled, falling back to the Gala rig with a warning if it is not.
    scene : bpy.types.Scene, optional
        Scene to build in.

    Returns
    -------
    bpy.types.Object
        The rig empty. Its children are the lights; rotate or scale it to
        adjust the whole rig at once.

    Raises
    ------
    ValueError
        If ``backend`` is unknown, ``distance`` is not positive, or a spec
        cannot be applied.
    GalaError
        If the subject has non-finite coordinates.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    if backend not in ("gala", "tri_lighting"):
        raise ValueError(f"backend must be 'gala' or 'tri_lighting', got {backend!r}")
    # Not `distance <= 0`, which `nan` passes: every comparison against it is
    # False, so the one value that breaks every light gets through untouched.
    if not distance > 0:
        raise ValueError(f"distance must be positive, got {distance}")

    if backend == "tri_lighting":
        delegated = _try_tri_lighting(target, energy, scene)
        if delegated is not None:
            return delegated
        warnings.warn(
            "The Tri-lighting add-on is not enabled; using Gala's rig instead. "
            "Enable it under Edit > Preferences > Add-ons to use that backend.",
            stacklevel=2,
        )

    centre, radius = _subject_bounds(target, scene)
    _validate_rig(specs, radius, energy, distance, softness)

    clear_lighting(scene=scene)

    rig = bpy_mod.data.objects.new(RIG_NAME, None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = radius
    rig.location = tuple(float(v) for v in centre)
    rig.rotation_euler = (0.0, 0.0, math.radians(rotation))
    gala_collections.link_object(rig, gala_collections.LIGHTING, scene)
    gala_collections.tag(rig, "light_rig", subject_radius=radius)

    for spec in specs:
        light = _make_area_light(spec, radius, energy, distance, softness, scene)
        light.visible_camera = visible_to_camera
        light.parent = rig
        light.matrix_parent_inverse = rig.matrix_world.inverted()

    bpy_mod.context.view_layer.update()
    return rig


def _make_area_light(
    spec: LightSpec,
    radius: float,
    energy: float,
    distance: float,
    softness: float,
    scene: Any,
) -> Any:
    bpy_mod = _require_bpy()

    data = bpy_mod.data.lights.new(f"GALA {spec.name}", type="AREA")
    data.shape = "DISK"
    data.size = max(radius * spec.size * softness, 1e-4)
    data.color = spec.colour
    # Inverse-square falloff means power must scale with distance squared to
    # hold illumination constant as the subject size changes.
    light_distance = radius * distance
    data.energy = _BASE_POWER * spec.power * energy * (light_distance**2)

    obj = bpy_mod.data.objects.new(f"GALA {spec.name}", data)
    obj.location = _spherical(spec.azimuth, spec.elevation, light_distance)
    obj.rotation_euler = _aim_at_origin(obj.location)

    gala_collections.link_object(obj, gala_collections.LIGHTING, scene)
    gala_collections.tag(obj, f"{spec.name.lower()}_light")
    return obj


def _spherical(
    azimuth: float, elevation: float, distance: float
) -> tuple[float, float, float]:
    """Convert rig-relative spherical angles to a Cartesian offset.

    Azimuth is measured about ``+Z`` starting from ``-Y`` (Blender's default
    camera direction), so ``azimuth=0`` sits between the viewer and the
    subject.
    """
    az = math.radians(azimuth)
    el = math.radians(elevation)
    horizontal = distance * math.cos(el)
    return (
        horizontal * math.sin(az),
        -horizontal * math.cos(az),
        distance * math.sin(el),
    )


def _aim_at_origin(location: Sequence[float]) -> tuple[float, float, float]:
    """Euler rotation that points a light's ``-Z`` axis back at the rig centre."""
    direction = -Vector(tuple(float(v) for v in location))
    euler = direction.to_track_quat("-Z", "Y").to_euler()
    return (float(euler[0]), float(euler[1]), float(euler[2]))


def _try_tri_lighting(target: Any, energy: float, scene: Any) -> Any | None:
    bpy_mod = _require_bpy()
    op = getattr(bpy_mod.ops.object, "tri_lighting", None)
    if op is None:
        return None
    try:
        if not op.poll():
            return None
        op(height=2.0, distance=4.0, energy=_BASE_POWER * energy)
    except (RuntimeError, AttributeError):
        return None
    return scene.objects.get("Key") or None


def list_hdris() -> dict[str, str]:
    """Return the built-in world HDRIs as ``{name: filepath}``.

    Gala ships no HDRI files of its own (SPECIFICATION D-10); these come with
    Blender.

    Returns
    -------
    dict[str, str]
        Mapping of short name to absolute ``.exr`` path. Empty if the data
        files cannot be located.
    """
    bpy_mod = _require_bpy()
    try:
        directory = bpy_mod.utils.system_resource(
            "DATAFILES", path="studiolights/world"
        )
    except Exception:  # pragma: no cover
        return {}
    if not directory or not os.path.isdir(directory):
        return {}
    return {
        os.path.splitext(name)[0]: os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith((".exr", ".hdr"))
    }


def _load_image(path: str) -> Any:
    """Load an image datablock, or raise if the file is not one.

    Reading ``size`` is what forces the decode; an image whose dimensions come
    back as zero is the one Blender made to stand in for a file it could not
    read. The datablock is dropped again so a mistyped path does not leave a
    broken image behind for the next call to find with ``check_existing``.

    Raises
    ------
    ValueError
        If ``path`` holds no readable image.
    """
    bpy_mod = _require_bpy()

    image = bpy_mod.data.images.load(path, check_existing=True)
    if all(image.size):
        return image

    if image.users == 0:
        bpy_mod.data.images.remove(image)
    raise ValueError(
        f"{path} is not an image Blender can read; an HDRI must be an .exr or "
        ".hdr file. Blender's own reason is on stderr."
    )


def hdri_lighting(
    hdri: str = "studio",
    strength: float = 1.0,
    rotation: float = 0.0,
    visible_to_camera: bool = False,
    scene: Any = None,
) -> Any:
    """Light the scene with an environment texture.

    Parameters
    ----------
    hdri : str, optional
        A name from :data:`STUDIO_HDRIS`, or a path to an ``.exr``/``.hdr``
        file.
    strength : float, optional
        Environment brightness.
    rotation : float, optional
        Rotate the environment about the vertical axis, in degrees. This is
        how you move a highlight without moving the camera.
    visible_to_camera : bool, optional
        Whether the environment shows up in the background. ``False`` keeps a
        transparent film transparent while still lighting the molecule, which
        is almost always what a figure needs.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    bpy.types.World
        The configured world.

    Raises
    ------
    FileNotFoundError
        If ``hdri`` is neither a known name nor an existing file.
    ValueError
        If the file exists but is not an image Blender can read.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    builtin = list_hdris()
    if hdri in builtin:
        path = builtin[hdri]
    elif os.path.isfile(hdri):
        path = hdri
    else:
        raise FileNotFoundError(
            f"HDRI {hdri!r} not found. Use a file path, or one of "
            f"{sorted(builtin) or list(STUDIO_HDRIS)}."
        )

    # Loaded before the world is touched, because this is where a wrong path
    # that happens to exist is found out. `images.load` does not raise on a
    # file it cannot decode — it reports `unknown file-format` to stderr and
    # hands back an image of no size, which lights the scene with nothing at
    # all while every return value says the call succeeded.
    image = _load_image(path)

    world = scene.world
    if world is None:
        world = bpy_mod.data.worlds.new("GALA World")
        scene.world = world
    world.use_nodes = True

    tree = world.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputWorld")
    background = tree.nodes.new("ShaderNodeBackground")
    environment = tree.nodes.new("ShaderNodeTexEnvironment")
    mapping = tree.nodes.new("ShaderNodeMapping")
    coords = tree.nodes.new("ShaderNodeTexCoord")

    output.location = (600, 0)
    background.location = (400, 0)
    environment.location = (150, 0)
    mapping.location = (-100, 0)
    coords.location = (-300, 0)

    environment.image = image
    background.inputs["Strength"].default_value = strength
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation))

    tree.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    world.cycles_visibility.camera = visible_to_camera

    world["gala_hdri"] = os.path.basename(path)
    return world


def clear_lighting(scene: Any = None) -> int:
    """Remove Gala's lights and rig empty.

    Parameters
    ----------
    scene : bpy.types.Scene, optional
        Scene to clean.

    Returns
    -------
    int
        Number of objects removed.
    """
    return gala_collections.clear(gala_collections.LIGHTING, scene=scene)
