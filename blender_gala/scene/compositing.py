"""Render passes and compositing: cryptomatte, Z depth, depth of field, depth cue.

The point of setting up cryptomatte before rendering is that it lets you change
your mind *afterwards* — brighten just the ligand, or knock back one chain —
without re-rendering a 40-minute image. Gala therefore enables the passes and
writes them to a multilayer EXR rather than baking any of it into the beauty
pass (SPECIFICATION D-14).

Blender 5.x moved the scene compositor into a reusable node group
(``scene.compositing_node_group``) while 4.x uses ``scene.node_tree``. Both are
handled here so the same call works on either.
"""

from __future__ import annotations

import contextlib
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
    "set_exr_output",
    "setup_compositor",
]

_GALA_PREFIX = "GALA "
_RENDER_LAYERS = "GALA Render Layers"


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def _is_blender_5() -> bool:
    bpy_mod = _require_bpy()
    return bpy_mod.app.version[0] >= 5


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

    if _is_blender_5():
        tree = scene.compositing_node_group
        if tree is None and create:
            tree = bpy_mod.data.node_groups.new(
                "GALA Compositing", "CompositorNodeTree"
            )
            tree.interface.new_socket(
                "Image", in_out="INPUT", socket_type="NodeSocketColor"
            )
            tree.interface.new_socket(
                "Image", in_out="OUTPUT", socket_type="NodeSocketColor"
            )
            tree.nodes.new("NodeGroupOutput")
            scene.compositing_node_group = tree
        return tree

    if create:
        scene.use_nodes = True
    return scene.node_tree


def _output_node(tree: Any) -> Any:
    """Return the node that terminates the compositor chain."""
    if _is_blender_5():
        for node in tree.nodes:
            if node.bl_idname == "NodeGroupOutput":
                return node
        return tree.nodes.new("NodeGroupOutput")

    for node in tree.nodes:
        if node.bl_idname == "CompositorNodeComposite":
            return node
    return tree.nodes.new("CompositorNodeComposite")


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
    bl_idname: str | tuple[str, ...],
    name: str,
    location: tuple[float, float],
) -> Any | None:
    """Create a node, trying each candidate type in turn.

    Blender 5.0 rewrote the compositor and replaced several ``CompositorNode*``
    types with the unified ``ShaderNode*`` ones, so most nodes need a candidate
    list to work on both 4.2 LTS and 5.x.

    Returns
    -------
    bpy.types.Node or None
        ``None``, with a warning, if no candidate exists in this build.
    """
    candidates = (bl_idname,) if isinstance(bl_idname, str) else bl_idname
    for candidate in candidates:
        try:
            node = tree.nodes.new(candidate)
        except RuntimeError:
            continue
        node.name = f"{_GALA_PREFIX}{name}"
        node.label = name
        node.location = location
        return node

    warnings.warn(
        f"this Blender build has none of {candidates}; skipping the {name} step",
        stacklevel=3,
    )
    return None


def _output_named(node: Any, *names: str) -> Any | None:
    for name in names:
        socket = node.outputs.get(name)
        if socket is not None:
            return socket
    return None


def _typed_sockets(sockets: Any, socket_type: str) -> list[Any]:
    """Return sockets of one data type.

    ``ShaderNodeMix`` exposes several identically named sockets — one per data
    type — so they can only be told apart by ``.type``.
    """
    return [s for s in sockets if s.type == socket_type]


def _mix_sockets(node: Any) -> tuple[Any, Any, Any, Any]:
    """Return ``(factor, a, b, result)`` for either mix node generation."""
    if node.bl_idname == "CompositorNodeMixRGB":
        colours = _typed_sockets(node.inputs, "RGBA")
        return node.inputs["Fac"], colours[0], colours[1], node.outputs[0]
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
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene

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
    depth = _output_named(render_layers, "Depth", "Z")
    x = -300

    if denoise:
        node = _new(tree, "CompositorNodeDenoise", "Denoise", (x, 0))
        if node is not None:
            tree.links.new(image, node.inputs["Image"])
            normal = _output_named(render_layers, "Normal")
            if normal is not None and "Normal" in node.inputs:
                tree.links.new(normal, node.inputs["Normal"])
            albedo = _output_named(render_layers, "DiffCol", "Diffuse Color")
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
    if far <= near:
        raise ValueError(f"depth_cue_range needs far > near, got {depth_range}")

    scale = units.DEFAULT_WORLD_SCALE
    map_range = _new(
        tree,
        ("CompositorNodeMapRange", "ShaderNodeMapRange"),
        "Depth Cue Range",
        (x, -250),
    )
    mix = _new(
        tree, ("CompositorNodeMixRGB", "ShaderNodeMix"), "Depth Cue", (x + 220, 0)
    )
    if map_range is None or mix is None:
        return image

    map_range.inputs["From Min"].default_value = near * scale
    map_range.inputs["From Max"].default_value = far * scale
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True
    tree.links.new(depth, map_range.inputs["Value"])
    factor_out = _output_named(map_range, "Result", "Value")

    if getattr(mix, "bl_idname", "") == "ShaderNodeMix":
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
        # The layer_name enum is only populated once the passes exist in a
        # render result; until then the node defaults to the object layer.
        with contextlib.suppress(TypeError):
            node.layer_name = layer
        tree.links.new(render_layers.outputs["Image"], node.inputs["Image"])
        nodes.append(node)
    return nodes


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

    # Blender 5.x replaced base_path/layer_slots with directory/file_name and a
    # file_output_items collection.
    if hasattr(node, "file_output_items"):
        node.directory = directory
        node.file_name = basename
        slots = _SlotAdapter(node.file_output_items, modern=True)
    else:
        node.base_path = os.path.join(directory, basename)
        slots = _SlotAdapter(node.layer_slots, modern=False)

    from .render import set_image_format

    set_image_format(
        node.format, "OPEN_EXR_MULTILAYER", color_depth="32", exr_codec="ZIP"
    )

    render_layers = _render_layers_node(tree, scene)
    slots.clear()

    wanted = [
        ("Image", ("Image",)),
        ("Depth", ("Depth", "Z")),
        ("Normal", ("Normal",)),
    ]
    wanted.extend(
        (socket.name, (socket.name,))
        for socket in render_layers.outputs
        if socket.name.startswith("Crypto")
    )

    linked = 0
    for label, candidates in wanted:
        socket = _output_named(render_layers, *candidates)
        if socket is None:
            continue
        slots.new(label)
        tree.links.new(socket, node.inputs[linked])
        linked += 1

    return node


class _SlotAdapter:
    """Uniform ``new``/``clear`` over both File Output slot APIs."""

    def __init__(self, collection: Any, modern: bool) -> None:
        self._collection = collection
        self._modern = modern

    def clear(self) -> None:
        self._collection.clear()

    def new(self, name: str) -> Any:
        if self._modern:
            return self._collection.new("RGBA", name)
        return self._collection.new(name)


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


def depth_of_field(
    target: Any = None,
    fstop: float = 2.8,
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
    """
    bpy_mod = _require_bpy()
    scene = scene or bpy_mod.context.scene
    camera = camera or scene.camera
    if camera is None:
        raise RuntimeError(
            "the scene has no camera; add one before setting depth of field"
        )

    data = camera.data
    data.dof.use_dof = enable
    if not enable:
        return data

    data.dof.aperture_fstop = fstop

    focus_object = None
    if target is not None:
        if bpy is not None and isinstance(target, bpy.types.Object):
            focus_object = target
        else:
            structure = AtomStructure.from_any(target)
            focus_object = structure.object

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
    """
    return setup_compositor(depth_cue_range=(near, far), scene=scene)
