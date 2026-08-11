"""Render passes and compositing: cryptomatte, Z depth, depth of field, depth cue.

The point of setting up cryptomatte before rendering is that it lets you change
your mind *afterwards* — brighten just the ligand, or knock back one chain —
without re-rendering a 40-minute image. Gala therefore enables the passes and
writes them to a multilayer EXR rather than baking any of it into the beauty
pass (SPECIFICATION D-14).

Blender 5 moved the scene compositor into a reusable node group
(``scene.compositing_node_group``), replaced several ``CompositorNode*`` types
with the unified ``ShaderNode*`` ones, and swapped the File Output node's
``base_path``/``layer_slots`` for ``directory``/``file_name`` and a
``file_output_items`` collection. Only that generation is handled: the minimum
supported Blender is 5.1.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

from ..core.entity import AtomStructure

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "add_file_output",
    "clear_compositor",
    "cryptomatte_layers",
    "depth_cue",
    "depth_of_field",
    "enable_passes",
    "highlight_matte",
    "set_exr_output",
    "setup_compositor",
]

_GALA_PREFIX = "GALA "
_RENDER_LAYERS = "GALA Render Layers"

#: Nodes :func:`highlight_matte` owns. Prefixed so re-running replaces them
#: rather than stacking another knock-back on top of the last one.
_HIGHLIGHT_PREFIX = "GALA Highlight"

#: What the ``layer`` argument of :func:`highlight_matte` names.
_CRYPTO_LAYERS = {
    "object": "CryptoObject",
    "material": "CryptoMaterial",
    "asset": "CryptoAsset",
}


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def enable_passes(
    cryptomatte: bool = True,
    depth: bool = True,
    normal: bool = True,
    mist: bool = False,
    ambient_occlusion: bool = False,
    cryptomatte_levels: int = 6,
    view_layer: Any = None,
) -> list[str]:
    """Turn on the render passes Gala's compositing needs.

    Parameters
    ----------
    cryptomatte : bool, optional
        Enable object, material and asset cryptomatte. This is what lets a
        compositor re-select "the ligand" in a finished render.
    depth : bool, optional
        Enable the Z pass, needed for depth of field and depth cueing.
    normal : bool, optional
        Enable the normal pass; also improves denoising.
    mist : bool, optional
        Enable the mist pass, an alternative normalised depth.
    ambient_occlusion : bool, optional
        Enable the AO pass for compositing-time occlusion.
    cryptomatte_levels : int, optional
        Cryptomatte ranks per pixel. 6 handles the semi-transparent edges of a
        molecular surface; lower values produce fringing there.
    view_layer : bpy.types.ViewLayer, optional
        View layer to configure. Defaults to the active one.

    Returns
    -------
    list[str]
        Names of the passes that were enabled.
    """
    bpy_mod = _require_bpy()
    view_layer = view_layer or bpy_mod.context.view_layer

    enabled: list[str] = []
    settings = {
        "use_pass_combined": True,
        "use_pass_z": depth,
        "use_pass_normal": normal,
        "use_pass_mist": mist,
        "use_pass_ambient_occlusion": ambient_occlusion,
        "use_pass_cryptomatte_object": cryptomatte,
        "use_pass_cryptomatte_material": cryptomatte,
        "use_pass_cryptomatte_asset": cryptomatte,
    }
    for attr, value in settings.items():
        if not hasattr(view_layer, attr):
            continue
        setattr(view_layer, attr, value)
        if value:
            enabled.append(attr.replace("use_pass_", ""))

    if cryptomatte and hasattr(view_layer, "pass_cryptomatte_depth"):
        view_layer.pass_cryptomatte_depth = cryptomatte_levels
    if cryptomatte and hasattr(view_layer, "use_pass_cryptomatte_accurate"):
        view_layer.use_pass_cryptomatte_accurate = True

    return enabled


def cryptomatte_layers(view_layer: Any = None) -> list[str]:
    """Return the cryptomatte layer names currently enabled.

    Returns
    -------
    list[str]
        Some of ``"CryptoObject"``, ``"CryptoMaterial"``, ``"CryptoAsset"``.
    """
    bpy_mod = _require_bpy()
    view_layer = view_layer or bpy_mod.context.view_layer
    names = []
    for attr, name in (
        ("use_pass_cryptomatte_object", "CryptoObject"),
        ("use_pass_cryptomatte_material", "CryptoMaterial"),
        ("use_pass_cryptomatte_asset", "CryptoAsset"),
    ):
        if getattr(view_layer, attr, False):
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Node tree plumbing
# ---------------------------------------------------------------------------


def _compositor_tree(scene: Any, create: bool = True) -> Any:
    """Return the scene's compositor node tree, creating it if needed."""
    bpy_mod = _require_bpy()

    tree = scene.compositing_node_group
    if tree is None and create:
        tree = bpy_mod.data.node_groups.new("GALA Compositing", "CompositorNodeTree")
        tree.interface.new_socket(
            "Image", in_out="INPUT", socket_type="NodeSocketColor"
        )
        tree.interface.new_socket(
            "Image", in_out="OUTPUT", socket_type="NodeSocketColor"
        )
        tree.nodes.new("NodeGroupOutput")
        scene.compositing_node_group = tree
    return tree


def _output_node(tree: Any) -> Any:
    """Return the node that terminates the compositor chain."""
    for node in tree.nodes:
        if node.bl_idname == "NodeGroupOutput":
            return node
    return tree.nodes.new("NodeGroupOutput")


def _render_layers_node(tree: Any, scene: Any) -> Any:
    for node in tree.nodes:
        if node.bl_idname == "CompositorNodeRLayers":
            return node
    node = tree.nodes.new("CompositorNodeRLayers")
    node.name = _RENDER_LAYERS
    node.scene = scene
    node.location = (-600, 0)
    return node


def _remove_gala_nodes(tree: Any) -> int:
    removed = 0
    for node in list(tree.nodes):
        if (
            node.name.startswith(_GALA_PREFIX)
            and node.bl_idname != "CompositorNodeRLayers"
        ):
            tree.nodes.remove(node)
            removed += 1
    return removed


def _new(
    tree: Any,
    bl_idname: str,
    name: str,
    location: tuple[float, float],
) -> Any | None:
    """Create a node, or warn and return ``None`` if this build has no such type.

    Every type used here exists in 5.1 and 5.2, so the ``None`` is insurance
    against a future release retiring one — the callers each skip their step
    rather than raising in the middle of building a tree.
    """
    try:
        node = tree.nodes.new(bl_idname)
    except RuntimeError:
        warnings.warn(
            f"this Blender build has no {bl_idname}; skipping the {name} step",
            stacklevel=3,
        )
        return None

    node.name = f"{_GALA_PREFIX}{name}"
    node.label = name
    node.location = location
    return node


def _typed_sockets(sockets: Any, socket_type: str) -> list[Any]:
    """Return sockets of one data type.

    ``ShaderNodeMix`` exposes several identically named sockets — one per data
    type — so they can only be told apart by ``.type``.
    """
    return [s for s in sockets if s.type == socket_type]


def _mix_sockets(node: Any) -> tuple[Any, Any, Any, Any]:
    """Return ``(factor, a, b, result)`` of a ``ShaderNodeMix``."""
    factor = _typed_sockets(node.inputs, "VALUE")[0]
    colours = _typed_sockets(node.inputs, "RGBA")
    result = _typed_sockets(node.outputs, "RGBA")[0]
    return factor, colours[0], colours[1], result


# ---------------------------------------------------------------------------
# Compositor setup
# ---------------------------------------------------------------------------


def setup_compositor(
    denoise: bool = True,
    cryptomatte: bool = True,
    dof: bool = False,
    dof_fstop: float = 2.8,
    depth_cue_range: tuple[float, float] | None = None,
    exposure: float = 0.0,
    contrast: float = 0.0,
    file_output: str | None = None,
    scene: Any = None,
) -> Any:
    """Build Gala's compositing chain. Idempotent.

    Every node Gala creates is named with a ``GALA`` prefix and removed before
    rebuilding, so calling this repeatedly rewires rather than accumulating
    duplicates. Nodes the user added themselves are left alone.

    Parameters
    ----------
    denoise : bool, optional
        Insert a compositing Denoise node after the render layers. Useful even
        with Cycles' own denoiser when compositing several passes.
    cryptomatte : bool, optional
        Enable cryptomatte passes and add a Cryptomatte node per layer, ready
        for picking. The nodes are deliberately left unconnected — connecting
        one would matte the beauty pass.
    dof : bool, optional
        Add a Z-driven Defocus node. Prefer real camera depth of field
        (:func:`depth_of_field`) when you can afford the samples; this is the
        cheap, tweakable-after-the-fact version.
    dof_fstop : float, optional
        Defocus f-stop. Lower is blurrier.
    depth_cue_range : tuple[float, float], optional
        ``(near, far)`` in ångström. Fades the image towards the background
        with depth, the classic way of keeping a crowded binding site readable.
    exposure : float, optional
        Exposure adjustment in stops.
    contrast : float, optional
        Contrast adjustment, -100 to 100.
    file_output : str, optional
        Directory for a multilayer EXR File Output node carrying Image, Depth
        and every cryptomatte layer.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    bpy.types.NodeTree
        The compositor node tree.

    Raises
    ------
    ValueError
        If ``depth_cue_range`` is not a usable ``(near, far)``.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    # Every argument is checked before the first of them is acted on. The tree
    # is rebuilt by removing what Gala owns and making it again, so a range
    # rejected halfway through leaves the scene with an output nothing feeds:
    # not the old chain, not the new one, and unrenderable either way.
    if depth_cue_range is not None:
        depth_cue_range = _checked_depth_range(depth_cue_range)

    enable_passes(
        cryptomatte=cryptomatte, depth=True, view_layer=scene_view_layer(scene)
    )
    scene.render.use_lock_interface = True

    tree = _compositor_tree(scene)
    _remove_gala_nodes(tree)

    render_layers = _render_layers_node(tree, scene)
    output = _output_node(tree)
    output.location = (900, 0)

    image = render_layers.outputs["Image"]
    depth = render_layers.outputs.get("Depth")
    x = -300

    if denoise:
        node = _new(tree, "CompositorNodeDenoise", "Denoise", (x, 0))
        if node is not None:
            tree.links.new(image, node.inputs["Image"])
            normal = render_layers.outputs.get("Normal")
            if normal is not None and "Normal" in node.inputs:
                tree.links.new(normal, node.inputs["Normal"])
            albedo = render_layers.outputs.get("Diffuse Color")
            if albedo is not None and "Albedo" in node.inputs:
                tree.links.new(albedo, node.inputs["Albedo"])
            image = node.outputs["Image"]
            x += 220

    if dof and depth is not None:
        node = _new(tree, "CompositorNodeDefocus", "Defocus", (x, 0))
        if node is not None:
            node.use_zbuffer = True
            node.f_stop = dof_fstop
            node.blur_max = 32.0
            tree.links.new(image, node.inputs["Image"])
            tree.links.new(depth, node.inputs["Z"])
            image = node.outputs["Image"]
            x += 220

    if depth_cue_range is not None and depth is not None:
        image = _build_depth_cue(tree, image, depth, depth_cue_range, scene, x)
        x += 440

    if exposure:
        node = _new(tree, "CompositorNodeExposure", "Exposure", (x, 0))
        if node is not None:
            node.inputs["Exposure"].default_value = exposure
            tree.links.new(image, node.inputs["Image"])
            image = node.outputs["Image"]
            x += 220

    if contrast:
        node = _new(tree, "CompositorNodeBrightContrast", "Contrast", (x, 0))
        if node is not None:
            node.inputs["Contrast"].default_value = contrast
            tree.links.new(image, node.inputs["Image"])
            image = node.outputs["Image"]
            x += 220

    tree.links.new(image, output.inputs[0])

    if cryptomatte:
        _add_cryptomatte_nodes(tree, render_layers, scene)

    if file_output:
        add_file_output(file_output, scene=scene, tree=tree)

    return tree


def scene_view_layer(scene: Any) -> Any:
    """Return the scene's first view layer.

    A helper rather than an inline expression because ``bpy.context.view_layer``
    is unavailable in some background contexts.
    """
    return scene.view_layers[0]


def _checked_depth_range(depth_range: tuple[float, float]) -> tuple[float, float]:
    """Return ``(near, far)`` as usable floats, or raise saying why not.

    Separate from :func:`_build_depth_cue` so that :func:`setup_compositor` can
    refuse the range before it starts pulling the tree apart.

    Raises
    ------
    ValueError
        If the range is not a pair, is not finite, or does not increase.
    """
    try:
        near, far = (float(value) for value in depth_range)
    except (TypeError, ValueError):
        raise ValueError(
            f"depth_cue_range must be a (near, far) pair of numbers, "
            f"got {depth_range!r}"
        ) from None

    # `far <= near` on its own lets `nan` through, since it compares False
    # against everything, and a Map Range of `nan` fades the whole frame away.
    if not math.isfinite(near) or not math.isfinite(far):
        raise ValueError(f"depth_cue_range must be finite, got {depth_range}")
    if far <= near:
        raise ValueError(f"depth_cue_range needs far > near, got {depth_range}")
    return near, far


def _build_depth_cue(
    tree: Any,
    image: Any,
    depth: Any,
    depth_range: tuple[float, float],
    scene: Any,
    x: float,
) -> Any:
    """Fade the image towards the background with increasing depth."""
    from ..core import units

    near, far = depth_range
    scale = units.DEFAULT_WORLD_SCALE
    map_range = _new(tree, "ShaderNodeMapRange", "Depth Cue Range", (x, -250))
    mix = _new(tree, "ShaderNodeMix", "Depth Cue", (x + 220, 0))
    if map_range is None or mix is None:
        return image

    map_range.inputs["From Min"].default_value = near * scale
    map_range.inputs["From Max"].default_value = far * scale
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True
    tree.links.new(depth, map_range.inputs["Value"])
    factor_out = map_range.outputs["Result"]

    mix.data_type = "RGBA"
    mix.blend_type = "MIX"

    factor_in, colour_a, colour_b, result = _mix_sockets(mix)
    background = tuple(scene.world.color[:3]) if scene.world else (0.05, 0.05, 0.05)
    if factor_out is not None:
        tree.links.new(factor_out, factor_in)
    tree.links.new(image, colour_a)
    colour_b.default_value = (*background, 1.0)
    return result


def _add_cryptomatte_nodes(tree: Any, render_layers: Any, scene: Any) -> list[Any]:
    nodes = []
    for index, layer in enumerate(cryptomatte_layers(scene_view_layer(scene))):
        node = _new(
            tree, "CompositorNodeCryptomatteV2", layer, (-300, -400 - index * 260)
        )
        if node is None:
            continue
        node.source = "RENDER"
        node.scene = scene
        _set_crypto_layer(node, layer, scene)
        tree.links.new(render_layers.outputs["Image"], node.inputs["Image"])
        nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Using the mattes
# ---------------------------------------------------------------------------


def _set_crypto_layer(node: Any, layer: str, scene: Any = None) -> str | None:
    """Point a Cryptomatte node at the ``CryptoObject``/``Material``/``Asset`` layer.

    The identifiers are qualified by the name of the view layer that produced
    them — ``ViewLayer.CryptoMaterial``, and whatever the user renamed it to if
    they did. That enum is generated at draw time from the node's source, so it
    cannot be read back through ``bl_rna`` (which reports the unqualified
    names); the only way to find out what it accepts is to try. Callers may
    pass a fully qualified name to skip the guessing.

    Returns
    -------
    str or None
        The identifier that was accepted, or ``None`` if none were.
    """
    candidates = [layer]
    if scene is not None:
        candidates += [f"{view_layer.name}.{layer}" for view_layer in scene.view_layers]
    candidates.append(f"ViewLayer.{layer}")

    for candidate in dict.fromkeys(candidates):
        try:
            node.layer_name = candidate
        except TypeError:
            continue
        return candidate

    # An unrendered scene has no cryptomatte layers to offer yet, so the node
    # keeps its default — the object layer — until there is a render result.
    return None


def _highlight_base(tree: Any, output: Any, doomed: list[Any]) -> Any | None:
    """Return the socket a rebuilt highlight should knock back.

    That is whatever currently feeds the output — except when the output is fed
    by an earlier highlight, in which case it is the image that highlight was
    given, recovered from the untouched side of its mix. Sockets belonging to
    nodes that are about to be removed are no use to anyone, so they come back
    as ``None`` and the caller falls back to the render layers.
    """
    links = output.inputs[0].links
    if not links:
        return None

    socket = links[0].from_socket
    node = links[0].from_node
    if node.name.startswith(_HIGHLIGHT_PREFIX):
        colours = _typed_sockets(node.inputs, "RGBA")
        inner = colours[1].links if len(colours) > 1 else ()
        if not inner:
            return None
        socket, node = inner[0].from_socket, inner[0].from_node

    return None if node in doomed else socket


def highlight_matte(
    matte: str | list[str] | tuple[str, ...],
    layer: str = "material",
    source: str | None = None,
    dim: float = 0.75,
    desaturate: float = 0.9,
    scene: Any = None,
) -> Any:
    """Knock everything back except one cryptomatte selection.

    This is what the passes were for: the chain, ligand or subunit named by
    ``matte`` keeps its colour and brightness, and the rest of the frame is
    darkened and drained of colour so it reads as context. Nothing about the
    3D scene changes, so the same render can be re-cut for as many figures or
    slides as the talk needs.

    With ``source`` it does that without rendering again at all: the image and
    its mattes are read from a multilayer EXR, so a scene with no molecule in
    it is enough to produce the picture.

    Parameters
    ----------
    matte : str or sequence of str
        What to keep. Object names for ``layer="object"``, material names for
        ``layer="material"`` — a Cryptomatte node's "pick" in the UI writes the
        same strings.
    layer : {"material", "object", "asset"}, optional
        Which cryptomatte layer to match ``matte`` against. Material is the
        useful one for molecules: Molecular Nodes puts a whole structure in one
        object, so per-chain mattes come from giving each chain's style its own
        material.
    source : str, optional
        Path to a multilayer EXR to composite from, such as the one
        :func:`set_exr_output` wrote. When omitted the current scene's render
        layers are used, and the highlight applies to the next render.
    dim : float, optional
        How far to darken everything else, ``0`` unchanged to ``1`` black.
    desaturate : float, optional
        How much colour to drain from everything else, ``0`` unchanged to
        ``1`` fully grey.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    bpy.types.NodeTree
        The compositor node tree.

    Raises
    ------
    ValueError
        If ``matte`` is empty, ``layer`` is not a cryptomatte layer, or ``dim``
        or ``desaturate`` is outside 0-1.
    FileNotFoundError
        If ``source`` does not exist.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

    names = [matte] if isinstance(matte, str) else list(matte)
    names = [name for name in names if name]
    if not names:
        raise ValueError("highlight_matte needs at least one matte name")
    if layer not in _CRYPTO_LAYERS:
        raise ValueError(
            f"unknown cryptomatte layer {layer!r}; choose from {sorted(_CRYPTO_LAYERS)}"
        )
    for label, value in (("dim", dim), ("desaturate", desaturate)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1, got {value}")
    if source is not None and not os.path.exists(source):
        raise FileNotFoundError(f"no such EXR to composite from: {source}")

    tree = _compositor_tree(scene)
    output = _output_node(tree)

    doomed = [node for node in tree.nodes if node.name.startswith(_HIGHLIGHT_PREFIX)]
    base = _highlight_base(tree, output, doomed)
    for node in doomed:
        tree.nodes.remove(node)

    # Laid out to the right of, and above, whatever `setup_compositor` built,
    # so a highlight added on top of the standard chain reads as a stage rather
    # than as nodes dropped on the ones already there.
    image = None
    if source is not None:
        node = _new(tree, "CompositorNodeImage", "Highlight Source", (-1000, 300))
        if node is None:  # pragma: no cover - the node type exists in 5.1+
            return tree
        image = bpy_mod.data.images.load(source, check_existing=True)
        node.image = image
        # A multilayer EXR's Combined pass; the socket is named for the pass
        # rather than "Image", and is the first one either way.
        base = node.outputs.get("Image") or node.outputs[0]
    elif base is None:
        enable_passes(cryptomatte=True, view_layer=scene_view_layer(scene))
        base = _render_layers_node(tree, scene).outputs["Image"]

    crypto = _new(tree, "CompositorNodeCryptomatteV2", "Highlight Matte", (300, -150))
    darken = _new(tree, "CompositorNodeExposure", "Highlight Dim", (300, 250))
    grey = _new(tree, "CompositorNodeHueSat", "Highlight Desaturate", (550, 250))
    mix = _new(tree, "ShaderNodeMix", "Highlight Mix", (850, 250))
    if crypto is None or darken is None or grey is None or mix is None:
        return tree
    output.location = (max(output.location[0], 1200), output.location[1])

    if image is not None:
        crypto.source = "IMAGE"
        crypto.image = image
    else:
        crypto.source = "RENDER"
        crypto.scene = scene
    _set_crypto_layer(crypto, _CRYPTO_LAYERS[layer], scene)
    crypto.matte_id = ", ".join(names)
    tree.links.new(base, crypto.inputs["Image"])

    # Stops rather than a multiplier, because that is what the node takes:
    # dim=0.75 keeps a quarter of the light, which is two stops down.
    darken.inputs["Exposure"].default_value = (
        math.log2(1.0 - dim) if dim < 1.0 else -12.0
    )
    tree.links.new(base, darken.inputs["Image"])

    grey.inputs["Saturation"].default_value = 1.0 - desaturate
    tree.links.new(darken.outputs["Image"], grey.inputs["Image"])

    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    factor, knocked_back, kept, result = _mix_sockets(mix)
    tree.links.new(crypto.outputs["Matte"], factor)
    tree.links.new(grey.outputs["Image"], knocked_back)
    tree.links.new(base, kept)
    tree.links.new(result, output.inputs[0])

    return tree


def add_file_output(
    directory: str,
    basename: str = "gala",
    scene: Any = None,
    tree: Any = None,
) -> Any | None:
    """Add a multilayer EXR File Output node carrying image, depth and mattes.

    Multilayer EXR is the interchange format that Nuke, Fusion, Krita and
    Blender's own compositor read cryptomatte from; a PNG cannot carry it.

    Parameters
    ----------
    directory : str
        Output directory. Created if missing.
    basename : str, optional
        File name stem.
    scene : bpy.types.Scene, optional
        Scene to configure.
    tree : bpy.types.NodeTree, optional
        Compositor tree; looked up from the scene when omitted.

    Returns
    -------
    bpy.types.Node or None
        The File Output node, or ``None`` if this build lacks the node type.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    tree = tree if tree is not None else _compositor_tree(scene)

    os.makedirs(directory, exist_ok=True)

    node = _new(tree, "CompositorNodeOutputFile", "EXR Output", (900, -400))
    if node is None:
        return None

    node.directory = directory
    node.file_name = basename
    slots = node.file_output_items

    from .render import set_image_format

    set_image_format(
        node.format, "OPEN_EXR_MULTILAYER", color_depth="32", exr_codec="ZIP"
    )

    render_layers = _render_layers_node(tree, scene)
    slots.clear()

    wanted = ["Image", "Depth", "Normal"]
    wanted.extend(
        socket.name
        for socket in render_layers.outputs
        if socket.name.startswith("Crypto")
    )

    linked = 0
    for label in wanted:
        socket = render_layers.outputs.get(label)
        if socket is None:
            continue
        slots.new("RGBA", label)
        tree.links.new(socket, node.inputs[linked])
        linked += 1

    return node


def set_exr_output(filepath: str, scene: Any = None) -> str:
    """Configure the render output as a 32-bit multilayer EXR.

    Saving the render result this way captures every enabled pass, including
    cryptomatte, without needing a File Output node.

    Parameters
    ----------
    filepath : str
        Output path.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    str
        The configured output path.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    from .render import set_image_format

    set_image_format(
        scene.render.image_settings,
        "OPEN_EXR_MULTILAYER",
        color_depth="32",
        color_mode="RGBA",
        exr_codec="ZIP",
    )
    scene.render.filepath = str(filepath)
    return scene.render.filepath


def clear_compositor(scene: Any = None) -> int:
    """Remove every node Gala added to the compositor.

    Returns
    -------
    int
        Number of nodes removed.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    tree = _compositor_tree(scene, create=False)
    if tree is None:
        return 0
    return _remove_gala_nodes(tree)


# ---------------------------------------------------------------------------
# Camera depth of field
# ---------------------------------------------------------------------------


#: An empty parked on whatever the camera should focus on. An object rather
#: than a distance so the focus keeps up with an orbit or an animation.
_FOCUS_NAME = "GALA Focus"


def _focus_empty(target: Any, selection: str, scene: Any) -> Any:
    """Return an empty at the middle of ``selection``, creating it if needed."""
    bpy_mod = _require_bpy()

    from ..core import collections as gala_collections
    from ..core.exceptions import EmptySelectionError

    structure = AtomStructure.from_any(target)
    mask = structure.select(selection)
    if not mask.any():
        raise EmptySelectionError(f"cannot focus on {selection!r}: it matches no atoms")
    point = structure.world_positions()[mask].mean(axis=0)

    obj = bpy_mod.data.objects.get(_FOCUS_NAME)
    if obj is None:
        obj = bpy_mod.data.objects.new(_FOCUS_NAME, None)
        obj.empty_display_type = "SPHERE"
        obj.empty_display_size = 0.02
        gala_collections.link_object(obj, gala_collections.ROOT, scene)
        gala_collections.tag(obj, "focus")
    obj.location = tuple(float(value) for value in point)
    return obj


def depth_of_field(
    target: Any = None,
    fstop: float = 2.8,
    selection: str | None = None,
    focus_distance: float | None = None,
    enable: bool = True,
    camera: Any = None,
    scene: Any = None,
) -> Any:
    """Set physical camera depth of field, focused on a molecule.

    This is the accurate version: Cycles traces a real aperture, so out-of-focus
    highlights behave correctly and there is no edge bleeding. It costs samples,
    which is why :func:`setup_compositor` also offers a Z-based Defocus.

    Parameters
    ----------
    target : AtomStructure, Molecule, bpy.types.Object, or None, optional
        What to focus on. When it is an object, Blender tracks it, so the focus
        follows it through an animation. ``None`` with an explicit
        ``focus_distance`` sets a fixed focus.
    fstop : float, optional
        Aperture. Lower is a shallower depth of field. At molecular scale
        (1 Å = 0.01 units) the depth of field is very shallow, so values around
        2.8-8 are usually a starting point rather than a final answer.
    selection : str, optional
        Focus on the middle of these atoms rather than on the target's origin.
        Without it the focus lands on the object origin, which for a molecule
        whose origin has been moved to its centroid is the middle of the whole
        protein — so a ligand a few ångström in front of that is exactly what
        the shallow depth of field blurs.
    focus_distance : float, optional
        Explicit focus distance in Blender units. Ignored when ``target`` is an
        object.
    enable : bool, optional
        Set ``False`` to switch depth of field off.
    camera : bpy.types.Object, optional
        Camera to configure. Defaults to the scene camera.
    scene : bpy.types.Scene, optional
        Scene to work in.

    Returns
    -------
    bpy.types.Camera
        The camera data block that was configured.

    Raises
    ------
    RuntimeError
        If there is no camera to configure.
    ValueError
        If ``fstop`` is not positive.
    EmptySelectionError
        If ``selection`` matches no atoms.
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    camera = camera or scene.camera
    if camera is None:
        raise RuntimeError(
            "the scene has no camera; add one before setting depth of field"
        )

    data = camera.data
    if not enable:
        data.dof.use_dof = False
        return data

    # A camera aperture of zero is not a very shallow depth of field, it is
    # none at all — the opposite of what "lower is blurrier" promises, and a
    # plausible way to spell "as blurry as possible".
    if not fstop > 0:
        raise ValueError(f"fstop must be positive, got {fstop}")

    # Resolved before the camera is touched, so that a selection matching no
    # atoms leaves the camera as it was rather than switching depth of field on
    # and focusing it wherever it was last pointed.
    focus_object = None
    if target is not None:
        if selection is not None:
            focus_object = _focus_empty(target, selection, scene)
        elif bpy is not None and isinstance(target, bpy.types.Object):
            focus_object = target
        else:
            structure = AtomStructure.from_any(target)
            focus_object = structure.object

    data.dof.use_dof = True
    data.dof.aperture_fstop = fstop

    if focus_object is not None:
        data.dof.focus_object = focus_object
    elif focus_distance is not None:
        data.dof.focus_object = None
        data.dof.focus_distance = focus_distance

    return data


def depth_cue(
    near: float,
    far: float,
    scene: Any = None,
) -> Any:
    """Enable depth cueing over an ångström depth range.

    A convenience wrapper that rebuilds the compositor with
    ``depth_cue_range`` set.

    Parameters
    ----------
    near, far : float
        Depth range in ångström measured from the camera. Geometry at ``near``
        is untouched; geometry at ``far`` fades fully into the background.
    scene : bpy.types.Scene, optional
        Scene to configure.

    Returns
    -------
    bpy.types.NodeTree

    Raises
    ------
    ValueError
        If the range is not finite and increasing. The compositor is left as it
        was.
    """
    return setup_compositor(depth_cue_range=(near, far), scene=scene)
