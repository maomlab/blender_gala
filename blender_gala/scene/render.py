"""Render engine, GPU, denoising and colour management.

Implements the render half of Objective 1. Each function is independently
callable so a user can, for example, keep their own sampling settings but adopt
Gala's colour management.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from .presets import RenderPreset, get_preset

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "VIEW_TRANSFORMS",
    "CausticsReport",
    "GPUReport",
    "enable_caustics",
    "enable_gpu",
    "render",
    "set_resolution",
    "set_transparent",
    "setup_color_management",
    "setup_render",
]

#: Backends tried in order of preference. OPTIX beats CUDA on NVIDIA hardware,
#: METAL is the only option on Apple silicon, HIP on AMD, ONEAPI on Intel.
_GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")

#: View transforms Blender ships with, in the order the UI lists them.
VIEW_TRANSFORMS = (
    "Standard",
    "Khronos PBR Neutral",
    "AgX",
    "Filmic",
    "Filmic Log",
    "False Color",
    "Raw",
)


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


@dataclass
class GPUReport:
    """What :func:`enable_gpu` actually managed to turn on.

    Molecular Nodes' equivalent silently falls back to CPU; Gala returns this
    so the UI and the logs can say *which* backend and devices are in use
    (SPECIFICATION §5.1).

    Attributes
    ----------
    enabled : bool
        Whether Cycles will render on the GPU.
    backend : str
        The compute device type, e.g. ``"METAL"``. Empty when CPU-only.
    devices : list[str]
        Names of the activated devices.
    message : str
        Human-readable summary.
    """

    enabled: bool = False
    backend: str = ""
    devices: list[str] = field(default_factory=list)
    message: str = ""

    def __str__(self) -> str:
        return self.message


def enable_gpu(use_cpu_too: bool = False, scene: Any = None) -> GPUReport:
    """Enable GPU rendering for Cycles, preferring the fastest backend.

    Parameters
    ----------
    use_cpu_too : bool, optional
        Also use CPU cores alongside the GPU. Usually a small win on machines
        with many cores and a slow GPU, and a small loss otherwise, so it is
        off by default.
    scene : bpy.types.Scene, optional
        Scene to configure. Defaults to the active scene.

    Returns
    -------
    GPUReport
        What was enabled. ``enabled=False`` means the render will use the CPU;
        a warning is emitted but no exception is raised, because falling back
        to CPU is a degraded result, not a failure.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    addon = bpy_mod.context.preferences.addons.get("cycles")
    if addon is None:
        scene.cycles.device = "CPU"
        return GPUReport(message="The Cycles add-on is disabled; rendering on CPU.")

    preferences = addon.preferences

    for backend in _GPU_BACKENDS:
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue  # This build was not compiled with that backend.

        try:
            preferences.refresh_devices()
        except Exception:  # pragma: no cover - driver dependent
            continue

        devices = [d for d in preferences.devices if d.type != "CPU"]
        if not devices:
            continue

        for device in preferences.devices:
            device.use = device.type != "CPU" or use_cpu_too

        scene.cycles.device = "GPU"
        names = [d.name for d in devices]
        return GPUReport(
            enabled=True,
            backend=backend,
            devices=names,
            message=f"Cycles is using {backend} on {', '.join(names)}.",
        )

    scene.cycles.device = "CPU"
    message = (
        "No supported GPU backend was found; Cycles will render on the CPU. "
        "Check System preferences > Cycles Render Devices."
    )
    warnings.warn(message, stacklevel=2)
    return GPUReport(message=message)


def setup_render(
    preset: str | RenderPreset = "figure",
    engine: str = "CYCLES",
    transparent: bool | None = None,
    use_gpu: bool = True,
    denoise: bool = True,
    scene: Any = None,
) -> GPUReport:
    """Configure the render engine for publication output.

    Parameters
    ----------
    preset : str or RenderPreset, optional
        One of ``"draft"``, ``"figure"``, ``"print"``, ``"poster"``,
        ``"presentation"``, or a :class:`~blender_gala.scene.presets.RenderPreset`.
    engine : {"CYCLES", "EEVEE"}, optional
        Cycles is the default because ambient occlusion, accurate shadows and
        subsurface scattering are what make molecular surfaces read as solid.
    transparent : bool, optional
        Override the preset's transparent-film setting.
    use_gpu : bool, optional
        Attempt GPU rendering.
    denoise : bool, optional
        Enable OpenImageDenoise for both render and viewport.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    GPUReport
        The GPU result, so callers can surface it. A CPU-only report is
        returned when ``use_gpu`` is ``False`` or the engine is not Cycles.

    Raises
    ------
    ValueError
        If ``engine`` is not recognised.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    config = get_preset(preset)

    engine_key = engine.upper()
    if engine_key == "CYCLES":
        scene.render.engine = "CYCLES"
    elif engine_key in ("EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        # The EEVEE identifier changed in Blender 5.0.
        for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = candidate
                break
            except TypeError:
                continue
    else:
        raise ValueError(f"engine must be 'CYCLES' or 'EEVEE', got {engine!r}")

    set_resolution(*config.resolution, scene=scene)
    set_transparent(
        config.transparent if transparent is None else transparent, scene=scene
    )

    report = GPUReport(message="GPU rendering was not requested.")
    if scene.render.engine == "CYCLES":
        cycles = scene.cycles
        cycles.samples = config.samples
        cycles.preview_samples = config.preview_samples
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = config.adaptive_threshold
        cycles.max_bounces = config.max_bounces
        cycles.transparent_max_bounces = max(8, config.max_bounces)
        cycles.transmission_bounces = max(8, config.max_bounces)
        # A light tree makes many-light scenes (our three-point rig plus an
        # HDRI) sample far more efficiently.
        if hasattr(cycles, "use_light_tree"):
            cycles.use_light_tree = True
        # Caustics off: molecular surfaces with transmission otherwise produce
        # fireflies that no practical sample count clears.
        cycles.caustics_reflective = False
        cycles.caustics_refractive = False

        _setup_denoising(scene, denoise)

        if use_gpu:
            report = enable_gpu(scene=scene)
        else:
            cycles.device = "CPU"
            report = GPUReport(message="GPU rendering was disabled by the caller.")
    else:
        eevee = scene.eevee
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = config.samples
        if hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = True

    setup_color_management(scene=scene)
    return report


def _setup_denoising(scene: Any, denoise: bool) -> None:
    cycles = scene.cycles
    cycles.use_denoising = denoise
    cycles.use_preview_denoising = denoise
    if not denoise:
        return
    for attr, value in (
        ("denoiser", "OPENIMAGEDENOISE"),
        ("preview_denoiser", "OPENIMAGEDENOISE"),
        ("denoising_input_passes", "RGB_ALBEDO_NORMAL"),
        ("preview_denoising_input_passes", "RGB_ALBEDO_NORMAL"),
        ("denoising_prefilter", "ACCURATE"),
        ("denoising_use_gpu", True),
    ):
        try:
            setattr(cycles, attr, value)
        except (AttributeError, TypeError):
            # Availability varies with Blender version and build options;
            # a missing denoiser knob is not worth failing the setup over.
            continue


def setup_color_management(
    view_transform: str = "Standard",
    look: str = "None",
    exposure: float = 0.0,
    gamma: float = 1.0,
    scene: Any = None,
) -> None:
    """Configure colour management for faithful figure colours.

    ``Standard`` is the default rather than Blender's ``AgX``
    (SPECIFICATION D-8): molecular figures use categorical colours — a chain
    rainbow, an AlphaFold confidence band — and a tone mapper that desaturates
    and shifts hue makes those colours no longer mean what the legend says.

    Parameters
    ----------
    view_transform : str, optional
        One of :data:`VIEW_TRANSFORMS`. Use ``"AgX"`` for cinematic renders
        where highlight rolloff matters more than colour fidelity.
    look : str, optional
        Contrast look, normally ``"None"``.
    exposure : float, optional
        Exposure in stops.
    gamma : float, optional
        Display gamma.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Raises
    ------
    ValueError
        If ``view_transform`` is not available in this Blender build.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    settings = scene.view_settings

    if view_transform not in VIEW_TRANSFORMS:
        raise ValueError(
            f"unknown view transform {view_transform!r}; "
            f"expected one of {VIEW_TRANSFORMS}"
        )
    try:
        settings.view_transform = view_transform
    except TypeError:
        # Background Blender sometimes exposes only a stub OpenColorIO enum.
        # The name was validated above, so this is an environment limitation
        # rather than caller error.
        warnings.warn(
            f"view transform {view_transform!r} is not available in this "
            "Blender session; leaving colour management unchanged.",
            stacklevel=2,
        )
        return

    try:
        settings.look = look
    except TypeError:
        settings.look = "None"

    settings.exposure = exposure
    settings.gamma = gamma
    scene.display_settings.display_device = "sRGB"
    scene.sequencer_colorspace_settings.name = "sRGB"


#: File format -> Blender 5 media type. Blender 5 filters the ``file_format``
#: enum by ``media_type``, so the media type has to be set first or assigning
#: the format silently fails.
_MEDIA_TYPES = {
    "OPEN_EXR_MULTILAYER": "MULTI_LAYER_IMAGE",
    "FFMPEG": "VIDEO",
}


def set_image_format(settings: Any, file_format: str, **options: Any) -> None:
    """Set an ``ImageFormatSettings`` block, tolerating version differences.

    Parameters
    ----------
    settings : bpy.types.ImageFormatSettings
        The block to configure, from ``scene.render.image_settings`` or from a
        File Output node.
    file_format : str
        A Blender file-format identifier, e.g. ``"PNG"``.
    **options
        Further attributes such as ``color_mode`` or ``exr_codec``. Options
        that this Blender build does not support are skipped rather than
        raising, since availability varies by version and build flags.
    """
    media_type = _MEDIA_TYPES.get(file_format, "IMAGE")
    for attr, value in (("media_type", media_type), ("file_format", file_format)):
        try:
            setattr(settings, attr, value)
        except (AttributeError, TypeError):
            continue

    for attr, value in options.items():
        try:
            setattr(settings, attr, value)
        except (AttributeError, TypeError):
            continue


#: Formats that already carry alpha and are therefore left alone by
#: :func:`set_transparent`. A multilayer EXR carries every compositing pass as
#: well, so replacing one with a PNG for the sake of an alpha channel it
#: already has throws away the cryptomatte and depth the render was set up for.
_ALPHA_FORMATS = ("OPEN_EXR", "OPEN_EXR_MULTILAYER")


def set_transparent(transparent: bool = True, scene: Any = None) -> None:
    """Enable or disable the transparent film.

    Also switches the output to RGBA PNG, because a transparent render written
    as RGB silently loses the alpha channel — a common and confusing failure.
    A format that already has an alpha channel keeps it, and only its colour
    mode is corrected.

    Parameters
    ----------
    transparent : bool, optional
        Whether the world background is rendered as alpha.
    scene : bpy.types.Scene, optional
        Scene to configure.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    scene.render.film_transparent = transparent

    if not transparent:
        return

    settings = scene.render.image_settings
    if settings.file_format in _ALPHA_FORMATS:
        set_image_format(settings, settings.file_format, color_mode="RGBA")
    else:
        set_image_format(
            settings,
            "PNG",
            color_mode="RGBA",
            color_depth="16",
            compression=15,
        )


def set_resolution(
    width: int, height: int, percentage: int = 100, scene: Any = None
) -> None:
    """Set the output resolution in pixels.

    Parameters
    ----------
    width, height : int
        Output size in pixels. Must be positive.
    percentage : int, optional
        Render scale; useful for quick previews at ``50``.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Raises
    ------
    ValueError
        If a dimension is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, got {width}x{height}")
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = int(percentage)


def render(
    filepath: str | None = None, animation: bool = False, scene: Any = None
) -> str:
    """Render the current scene.

    Parameters
    ----------
    filepath : str, optional
        Output path. Blender treats it as a stem: the extension comes from the
        output format and an animation adds the frame number, so what is
        written is not always what is asked for.
    animation : bool, optional
        Render the frame range instead of a single still.
    scene : bpy.types.Scene, optional
        Scene to render.

    Returns
    -------
    str
        The path that was written to, as Blender resolved it — the first
        frame's when rendering an animation. Asking for ``shot.jpg`` while the
        output format is PNG writes, and returns, ``shot.png``.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    if filepath is not None:
        scene.render.filepath = str(filepath)
    bpy_mod.ops.render.render(
        write_still=not animation, animation=animation, scene=scene.name
    )
    # `filepath` is the stem Blender starts from, not the file it wrote: it
    # appends the format's own extension and, for an animation, the frame
    # number. Returning the stem gives a caller that opens the result a
    # FileNotFoundError, so the resolved name is returned instead.
    frame = scene.frame_start if animation else scene.frame_current
    return str(scene.render.frame_path(frame=frame))


@dataclass
class CausticsReport:
    """What :func:`enable_caustics` switched on.

    Attributes
    ----------
    casters : int
        Objects marked as casting caustics.
    receivers : int
        Objects marked as receiving them.
    lights : int
        Lights allowed to produce them.
    """

    casters: int = 0
    receivers: int = 0
    lights: int = 0

    def __str__(self) -> str:
        return (
            f"Caustics: {self.casters} caster(s), {self.receivers} receiver(s), "
            f"{self.lights} light(s)"
        )


def enable_caustics(
    casters: Any = (),
    receivers: Any = (),
    lights: Any = None,
    transmission_bounces: int = 24,
    scene: Any = None,
) -> CausticsReport:
    """Let refracted light focus onto something, and keep it sharp.

    Cycles will not show a caustic by default, and three separate things are
    in the way. The renderer's caustic paths are off; the glossy filter blurs
    what does get through, which is what stops fireflies and also what turns a
    caustic into a smudge; and the shortcut that makes them tractable —
    manifold next event estimation — only runs between objects that have been
    told they are the caster and the receiver.

    This turns on all three. Nothing else in Gala needs it, and it is not part
    of :func:`~blender_gala.scene.setup.publication_setup`, because sharp
    caustics cost samples and most figures do not want to pay for them.

    Parameters
    ----------
    casters : object or sequence of object
        The refracting objects — a glass molecular surface, typically.
    receivers : object or sequence of object
        What the light lands on: the cartoon inside the surface, a floor, the
        binding partner.
    lights : object or sequence of object, optional
        Lights allowed to cast caustics. ``None`` means every light in the
        scene, which is usually what you want and is what makes a three-point
        rig produce three of them.
    transmission_bounces : int, optional
        Light entering a closed surface and leaving it again is two bounces,
        and it has to survive enough of them to reach the camera.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    CausticsReport
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    cycles = getattr(scene, "cycles", None)
    if cycles is None:  # pragma: no cover - Cycles is always present
        raise RuntimeError("Cycles is not available in this build")

    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    # The glossy filter is a blur, and a blurred caustic is just a bright
    # patch. Zero is what the Cycles manual calls for when caustics matter.
    cycles.blur_glossy = 0.0
    cycles.transmission_bounces = max(cycles.transmission_bounces, transmission_bounces)
    cycles.max_bounces = max(cycles.max_bounces, transmission_bounces + 4)

    report = CausticsReport()
    for obj in _as_sequence(casters):
        obj.cycles.is_caustics_caster = True
        report.casters += 1
    for obj in _as_sequence(receivers):
        obj.cycles.is_caustics_receiver = True
        report.receivers += 1

    if lights is None:
        chosen = [obj for obj in scene.objects if obj.type == "LIGHT"]
    else:
        chosen = list(_as_sequence(lights))
    for light in chosen:
        data = getattr(light, "data", light)
        data.cycles.is_caustics_light = True
        report.lights += 1

    return report


def _as_sequence(value: Any) -> list[Any]:
    """One object, a molecule, or a sequence of either, as a list of objects."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    return [getattr(item, "object", item) for item in items]
