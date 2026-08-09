"""Material presets for molecular representations.

One parameterised Principled BSDF builder plus a table of chemistry-oriented
presets (SPECIFICATION D-11). A dataclass rather than a fixed list of materials
means a user can start from ``protein`` and nudge one number without rebuilding
a node tree by hand.

Ambient occlusion is handled at material level (SPECIFICATION D-12): Cycles has
no per-material AO switch, so when ``ao_strength`` is non-zero the builder
multiplies an :class:`ShaderNodeAmbientOcclusion` into the base colour. That
darkens the crevices between atoms — the look that makes a space-filling model
read as solid rather than as a pile of flat circles.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any

from ..core import mn as mn_bridge

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = [
    "MATERIAL_PRESETS",
    "MATERIAL_SCHEMES",
    "GalaMaterialSpec",
    "assign_material",
    "assign_materials",
    "build_glass_subsurface",
    "build_material",
    "get_material",
]

_PREFIX = "GALA "


@dataclass(frozen=True)
class GalaMaterialSpec:
    """Parameters of a Gala material.

    All values map onto Principled BSDF inputs except ``ao_strength`` and
    ``ao_distance``, which drive the ambient-occlusion node.

    Attributes
    ----------
    base_color : tuple[float, float, float, float]
        RGBA base colour. Molecular styles normally drive colour from the mesh
        ``Color`` attribute, so this is only the fallback for objects that do
        not.
    use_attribute_color : bool
        Read the per-atom ``Color`` attribute into Base Color. Leave ``True``
        for anything driven by Molecular Nodes; ``False`` for decorations such
        as measurement dashes.
    color_mix : float
        How much of that attribute colour to keep, mixed towards
        ``base_color``. ``1`` is the colour as written. Below 1 matters for
        transmissive materials: light crossing coloured glass is tinted on the
        way in and again on the way out, so a saturated ramp turns the inside
        of the surface into a dark gemstone.
    roughness : float
        0 is a mirror, 1 is fully diffuse.
    metallic : float
        Metallic weight.
    ior : float
        Index of refraction.
    specular : float
        Specular IOR level.
    subsurface_weight : float
        Subsurface scattering weight. A little makes protein surfaces look less
        like plastic.
    subsurface_radius : tuple[float, float, float]
        Per-channel scattering radius, as a ratio between the three channels.
        Blender multiplies it by ``subsurface_scale``, so this alone does not
        set a distance.
    subsurface_scale : float
        What that radius is multiplied by, in Blender units — the distance
        light actually travels inside the surface. Blender's default is 0.005,
        which is 5 mm in a scene built to human scale and half an ångström in
        one built to Molecular Nodes'. Subsurface scattering at that distance
        is invisible on a molecule, so anything that wants the effect to show
        has to raise it: a cartoon ribbon is about 0.02 units thick.
    coat_weight : float
        Clear-coat weight; useful to make a ligand glossier than its protein.
    sheen_weight : float
        Sheen weight, for a soft velvet edge.
    emission_strength : float
        Emission strength. Non-zero makes the material self-lit.
    emission_color : tuple[float, float, float, float]
        Emission colour.
    alpha : float
        Opacity. Below 1 switches the material to alpha blending.
    transmission_weight : float
        Transmission weight. Above 0 the surface refracts what is behind it
        rather than blending with it, which is what makes caustics possible
        and what makes the render cost real.
    thin_wall : bool
        Treat the surface as a film with no thickness. A molecular surface is
        a shell around a protein, not a solid lump of glass, and rendering it
        as one gives the light a long dense path through the middle that comes
        out dark and slow.
    ao_strength : float
        Ambient occlusion mix, 0 to 1. Off by default.
    ao_distance : float
        Ambient occlusion sampling distance in Blender units.
    shadow : bool
        Whether the material casts shadows.
    description : str
        Human-readable summary shown in the UI.
    """

    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    use_attribute_color: bool = True
    color_mix: float = 1.0
    roughness: float = 0.45
    metallic: float = 0.0
    ior: float = 1.45
    specular: float = 0.5
    subsurface_weight: float = 0.0
    subsurface_radius: tuple[float, float, float] = (0.005, 0.005, 0.005)
    subsurface_scale: float = 0.005
    coat_weight: float = 0.0
    sheen_weight: float = 0.0
    emission_strength: float = 0.0
    emission_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    alpha: float = 1.0
    transmission_weight: float = 0.0
    thin_wall: bool = False
    ao_strength: float = 0.0
    ao_distance: float = 0.05
    shadow: bool = True
    description: str = ""

    def with_(self, **changes: Any) -> GalaMaterialSpec:
        """Return a copy with ``changes`` applied.

        Examples
        --------
        >>> MATERIAL_PRESETS["protein"].with_(roughness=0.7)  # doctest: +SKIP
        """
        return replace(self, **changes)


MATERIAL_PRESETS: dict[str, GalaMaterialSpec] = {
    "protein": GalaMaterialSpec(
        roughness=0.45,
        subsurface_weight=0.05,
        subsurface_radius=(0.005, 0.004, 0.003),
        description="Matte, slightly waxy. Reads well at cartoon and surface scale.",
    ),
    "ligand": GalaMaterialSpec(
        roughness=0.25,
        coat_weight=0.3,
        specular=0.6,
        description="Glossier than the protein so the ligand separates from it.",
    ),
    "nucleic": GalaMaterialSpec(
        roughness=0.6,
        specular=0.35,
        description="Matte and low-specular; keeps long backbones from glaring.",
    ),
    "surface": GalaMaterialSpec(
        roughness=0.15,
        alpha=0.55,
        specular=0.6,
        description=(
            "Translucent molecular surface. Uses alpha blending rather than "
            "transmission: it renders far faster and avoids caustic fireflies."
        ),
    ),
    "glass_surface": GalaMaterialSpec(
        base_color=(0.94, 0.95, 0.97, 1.0),
        color_mix=0.35,
        roughness=0.03,
        ior=1.45,
        transmission_weight=1.0,
        thin_wall=True,
        specular=0.5,
        description=(
            "A molecular surface as thin glass: it refracts what is inside it "
            "and can focus light onto it. Needs Cycles, transmission bounces "
            "and, for the caustics to survive, `enable_caustics`."
        ),
    ),
    "metal": GalaMaterialSpec(
        metallic=1.0,
        roughness=0.2,
        description="Metal ions and metal centres.",
    ),
    "lipid": GalaMaterialSpec(
        roughness=0.7,
        subsurface_weight=0.1,
        description="Membranes and detergent belts.",
    ),
    "glass": GalaMaterialSpec(
        roughness=0.05,
        alpha=0.25,
        ior=1.45,
        description="Barely-there context surface behind a highlighted site.",
    ),
    "measurement": GalaMaterialSpec(
        base_color=(1.0, 0.85, 0.2, 1.0),
        use_attribute_color=False,
        emission_strength=1.0,
        emission_color=(1.0, 0.85, 0.2, 1.0),
        roughness=1.0,
        shadow=False,
        description="Unlit yellow for measurement dashes; readable at any exposure.",
    ),
    "interaction": GalaMaterialSpec(
        base_color=(0.35, 0.75, 1.0, 1.0),
        use_attribute_color=False,
        emission_strength=1.0,
        emission_color=(0.35, 0.75, 1.0, 1.0),
        roughness=1.0,
        shadow=False,
        description="Unlit blue for hydrogen bonds and polar contacts.",
    ),
    "label": GalaMaterialSpec(
        base_color=(0.05, 0.05, 0.05, 1.0),
        use_attribute_color=False,
        emission_strength=1.0,
        emission_color=(0.95, 0.95, 0.95, 1.0),
        roughness=1.0,
        shadow=False,
        description="Unlit near-white for 3D text labels.",
    ),
}

#: Style-name fragment -> material preset. Used by :func:`assign_materials`.
MATERIAL_SCHEMES: dict[str, dict[str, str]] = {
    "chemistry": {
        "cartoon": "protein",
        "ribbon": "protein",
        "surface": "surface",
        "spheres": "protein",
        "ball_and_stick": "ligand",
        "sticks": "ligand",
        "preset": "protein",
        "default": "protein",
    },
    "matte": dict.fromkeys(
        [
            "cartoon",
            "ribbon",
            "surface",
            "spheres",
            "ball_and_stick",
            "sticks",
            "default",
        ],
        "nucleic",
    ),
    "glossy": dict.fromkeys(
        [
            "cartoon",
            "ribbon",
            "surface",
            "spheres",
            "ball_and_stick",
            "sticks",
            "default",
        ],
        "ligand",
    ),
}


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def _set_input(node: Any, names: tuple[str, ...], value: Any) -> bool:
    """Set the first matching input socket, tolerating renames across versions."""
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


_MN_COLOR_GROUP = "MN Color Input"


def _mn_colour_group() -> Any:
    """Return Molecular Nodes' colour-input node group, appending it if needed.

    Reading the per-atom ``Color`` attribute is not a one-node job: a style
    such as ball-and-stick renders atoms as *instanced* spheres, where a
    ``GEOMETRY`` attribute lookup returns black and only an ``INSTANCER``
    lookup works, while bonds and cartoon meshes are the other way round.
    Molecular Nodes already solves this in a reusable group, so Gala uses that
    group rather than reimplementing it — and stays correct if MN improves it.
    """
    bpy_mod = _require_bpy()

    group = bpy_mod.data.node_groups.get(_MN_COLOR_GROUP)
    if group is not None:
        return group

    module = mn_bridge.get_mn()
    if module is None:
        return None
    try:
        module.nodes.material.add_all_materials()
    except Exception:  # pragma: no cover - depends on the MN asset file
        return None
    return bpy_mod.data.node_groups.get(_MN_COLOR_GROUP)


def _colour_input(tree: Any, location: tuple[float, float]) -> tuple[Any, Any]:
    """Add the per-atom colour source to ``tree``.

    Returns
    -------
    tuple
        ``(colour socket, alpha socket)``. The alpha socket is ``None`` when
        falling back to a plain attribute lookup.
    """
    group = _mn_colour_group()
    if group is not None:
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = group
        node.location = location
        return node.outputs["Color"], node.outputs.get("Alpha")

    # Molecular Nodes is not installed, so there is no instancing to worry
    # about: a plain geometry attribute is right for an ordinary mesh.
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Color"
    attribute.attribute_type = "GEOMETRY"
    attribute.location = location
    return attribute.outputs["Color"], None


def build_material(
    spec: GalaMaterialSpec | str = "protein", name: str | None = None
) -> Any:
    """Construct (or rebuild) a material from a spec.

    Parameters
    ----------
    spec : GalaMaterialSpec or str
        A spec, or the name of a preset in :data:`MATERIAL_PRESETS`.
    name : str, optional
        Material name. Defaults to ``"GALA <preset>"``. An existing material
        with the same name is rebuilt in place, so every object already using
        it picks up the change.

    Returns
    -------
    bpy.types.Material

    Raises
    ------
    ValueError
        If ``spec`` names an unknown preset.
    """
    bpy_mod = _require_bpy()

    if isinstance(spec, str):
        if spec not in MATERIAL_PRESETS:
            raise ValueError(
                f"unknown material preset {spec!r}; "
                f"choose from {sorted(MATERIAL_PRESETS)}"
            )
        name = name or f"{_PREFIX}{spec.replace('_', ' ').title()}"
        spec = MATERIAL_PRESETS[spec]
    name = name or f"{_PREFIX}Material"

    material = bpy_mod.data.materials.get(name)
    if material is None:
        material = bpy_mod.data.materials.new(name)
    material.use_nodes = True

    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output.location = (600, 0)
    principled.location = (250, 0)
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    _set_input(principled, ("Base Color",), spec.base_color)
    _set_input(principled, ("Roughness",), spec.roughness)
    _set_input(principled, ("Metallic",), spec.metallic)
    _set_input(principled, ("IOR",), spec.ior)
    _set_input(principled, ("Specular IOR Level", "Specular"), spec.specular)
    _set_input(principled, ("Subsurface Weight", "Subsurface"), spec.subsurface_weight)
    _set_input(principled, ("Subsurface Radius",), spec.subsurface_radius)
    _set_input(principled, ("Subsurface Scale",), spec.subsurface_scale)
    _set_input(principled, ("Coat Weight", "Clearcoat"), spec.coat_weight)
    _set_input(principled, ("Sheen Weight", "Sheen"), spec.sheen_weight)
    _set_input(principled, ("Emission Strength",), spec.emission_strength)
    _set_input(principled, ("Emission Color", "Emission"), spec.emission_color)
    _set_input(principled, ("Alpha",), spec.alpha)
    _set_input(
        principled, ("Transmission Weight", "Transmission"), spec.transmission_weight
    )
    _set_input(principled, ("Thin Wall", "Thin Film"), spec.thin_wall)

    colour_socket = None
    alpha_socket = None
    if spec.use_attribute_color:
        colour_socket, alpha_socket = _colour_input(tree, (-350, 100))

    if colour_socket is not None and spec.color_mix < 1.0:
        tint = tree.nodes.new("ShaderNodeMix")
        tint.data_type = "RGBA"
        tint.location = (-180, 100)
        tint.inputs["Factor"].default_value = spec.color_mix
        inputs = [socket for socket in tint.inputs if socket.type == "RGBA"]
        inputs[0].default_value = spec.base_color
        tree.links.new(colour_socket, inputs[1])
        colour_socket = next(s for s in tint.outputs if s.type == "RGBA")

    if spec.ao_strength > 0.0:
        ao = tree.nodes.new("ShaderNodeAmbientOcclusion")
        ao.location = (-350, -150)
        ao.samples = 16
        ao.only_local = True
        _set_input(ao, ("Distance",), spec.ao_distance)

        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.location = (-50, 0)
        mix.inputs["Factor"].default_value = spec.ao_strength
        # ShaderNodeMix exposes several same-named sockets; index by type.
        colour_inputs = [s for s in mix.inputs if s.type == "RGBA"]
        if colour_socket is not None:
            tree.links.new(colour_socket, colour_inputs[0])
        else:
            colour_inputs[0].default_value = spec.base_color
        tree.links.new(ao.outputs["Color"], colour_inputs[1])
        colour_socket = next(s for s in mix.outputs if s.type == "RGBA")

    if colour_socket is not None:
        tree.links.new(colour_socket, principled.inputs["Base Color"])
    if alpha_socket is not None and spec.alpha >= 1.0:
        # Only let the attribute drive alpha when the spec is not deliberately
        # making the material translucent.
        tree.links.new(alpha_socket, principled.inputs["Alpha"])

    if spec.alpha < 1.0:
        for attr, value in (
            ("blend_method", "BLEND"),
            ("surface_render_method", "BLENDED"),
        ):
            try:
                setattr(material, attr, value)
            except (AttributeError, TypeError):
                continue

    if not spec.shadow:
        # The attribute name for "do not cast shadows" moved between EEVEE
        # generations, so try each and ignore the ones this build lacks.
        for attr, flag in (("shadow_method", "NONE"), ("is_shadow_catcher", False)):
            try:
                setattr(material, attr, flag)
            except (AttributeError, TypeError):
                continue
        with contextlib.suppress(AttributeError):
            material.cycles.is_caustics_light = False

    material["gala_preset"] = getattr(spec, "description", "")
    return material


def build_glass_subsurface(
    name: str = "GALA Glass Subsurface",
    mix: float = 0.4,
    color: Any = None,
    color_mix: float = 1.0,
    subsurface_scale: float = 20.0,
    subsurface_radius: tuple[float, float, float] = (0.1, 0.2, 0.1),
    glass_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    glass_roughness: float = 0.2,
    glass_ior: float = 0.2,
    distribution: str = "BECKMANN",
) -> Any:
    """A subsurface-scattering body under a glass shell, mixed.

    A different shape of material from the Principled ones in
    :data:`MATERIAL_PRESETS`, and built by hand for that reason: two shaders
    into a Mix, rather than one node with a transmission weight. What it buys
    over Principled glass is a body — light entering the shell scatters inside
    it instead of passing straight through, which on a molecular surface reads
    as depth rather than as a dark window.

    Parameters
    ----------
    name : str, optional
        Material name. An existing material of this name is rebuilt in place.
    mix : float, optional
        Weight of the glass shader, 0 to 1. The remainder is subsurface.
    color : sequence of float, optional
        Fixed RGBA for the subsurface body. ``None`` drives it from the mesh
        ``Color`` attribute, which is what carries a potential ramp or any
        other per-atom colouring.
    color_mix : float, optional
        How much of that attribute colour to keep, mixed towards white.
    subsurface_scale : float, optional
        Multiplier on the scattering radius. Blender units, so it is relative
        to the 0.01 scale Molecular Nodes gives a molecule: a radius larger
        than the molecule makes it glow rather than scatter.
    subsurface_radius : tuple[float, float, float], optional
        Per-channel scattering distance before scaling.
    glass_color : tuple[float, float, float, float], optional
        Tint of the shell itself.
    glass_roughness : float, optional
        0 is a polished shell; a little roughness keeps what is underneath
        legible through it.
    glass_ior : float, optional
        Index of refraction. Below 1 is not a mistake in a shell: it bends
        light the other way, as a bubble in water does rather than a bead of
        glass in air.
    distribution : str, optional
        Microfacet distribution for the glass, ``"BECKMANN"`` or ``"GGX"``.

    Returns
    -------
    bpy.types.Material
    """
    bpy_mod = _require_bpy()

    material = bpy_mod.data.materials.get(name)
    if material is None:
        material = bpy_mod.data.materials.new(name)
    material.use_nodes = True

    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)

    mixer = tree.nodes.new("ShaderNodeMixShader")
    mixer.location = (350, 0)
    mixer.inputs["Fac"].default_value = mix
    tree.links.new(mixer.outputs["Shader"], output.inputs["Surface"])

    subsurface = tree.nodes.new("ShaderNodeSubsurfaceScattering")
    subsurface.location = (100, 150)
    subsurface.falloff = "BURLEY"
    _set_input(subsurface, ("Scale",), subsurface_scale)
    _set_input(subsurface, ("Radius",), subsurface_radius)
    tree.links.new(subsurface.outputs["BSSRDF"], mixer.inputs[1])

    glass = tree.nodes.new("ShaderNodeBsdfGlass")
    glass.location = (100, -150)
    glass.distribution = distribution
    _set_input(glass, ("Color",), glass_color)
    _set_input(glass, ("Roughness",), glass_roughness)
    _set_input(glass, ("IOR",), glass_ior)
    tree.links.new(glass.outputs["BSDF"], mixer.inputs[2])

    if color is not None:
        _set_input(subsurface, ("Color",), color)
    else:
        source, _ = _colour_input(tree, (-400, 200))
        if color_mix < 1.0:
            tint = tree.nodes.new("ShaderNodeMix")
            tint.data_type = "RGBA"
            tint.location = (-180, 200)
            tint.inputs["Factor"].default_value = color_mix
            colours = [socket for socket in tint.inputs if socket.type == "RGBA"]
            colours[0].default_value = (1.0, 1.0, 1.0, 1.0)
            tree.links.new(source, colours[1])
            source = next(s for s in tint.outputs if s.type == "RGBA")
        tree.links.new(source, subsurface.inputs["Color"])

    return material


def get_material(preset: str = "protein", rebuild: bool = False) -> Any:
    """Return a preset material, building it on first use.

    Parameters
    ----------
    preset : str, optional
        A key of :data:`MATERIAL_PRESETS`.
    rebuild : bool, optional
        Rebuild the node tree even if the material already exists. Use after
        editing a preset.

    Returns
    -------
    bpy.types.Material
    """
    bpy_mod = _require_bpy()
    name = f"{_PREFIX}{preset.replace('_', ' ').title()}"
    existing = bpy_mod.data.materials.get(name)
    if existing is not None and not rebuild:
        return existing
    return build_material(preset, name)


def _style_nodes(target: Any) -> list[Any]:
    """Return the Molecular Nodes style nodes driving ``target``."""
    tree = None
    if mn_bridge.is_molecule(target):
        tree = getattr(target, "tree", None)
    elif bpy is not None and isinstance(target, bpy.types.Object):
        for modifier in target.modifiers:
            node_group = getattr(modifier, "node_group", None)
            if modifier.type == "NODES" and node_group is not None:
                tree = node_group
                break
    if tree is None:
        return []
    return [
        node
        for node in tree.nodes
        if node.bl_idname == "GeometryNodeGroup" and "Material" in node.inputs
    ]


def _style_key(node: Any) -> str:
    """Normalise a style node's name into a scheme lookup key."""
    raw = (getattr(node.node_tree, "name", "") or node.name).lower()
    for prefix in ("style ", "mn_", "mn "):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    return raw.strip().replace(" ", "_").replace("-", "_")


def assign_material(target: Any, material: Any, style: str | int | None = None) -> int:
    """Assign one material to a molecule's styles or to a plain object.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        What to assign to.
    material : bpy.types.Material or str
        A material, or the name of a Gala preset.
    style : str or int, optional
        Limit the assignment to one style: an index into the molecule's style
        nodes, or a substring of the style name such as ``"surface"``.
        ``None`` assigns to every style.

    Returns
    -------
    int
        Number of style nodes (or material slots) that were changed.
    """
    bpy_mod = _require_bpy()

    resolved = material
    if isinstance(material, str):
        resolved = (
            get_material(material)
            if material in MATERIAL_PRESETS
            else bpy_mod.data.materials[material]
        )

    obj = getattr(target, "object", target)
    nodes = _style_nodes(target)

    if not nodes:
        # Not a Molecular Nodes molecule: fall back to object material slots.
        if obj is None or not hasattr(obj, "data") or obj.data is None:
            return 0
        if obj.data.materials:
            obj.data.materials[0] = resolved
        else:
            obj.data.materials.append(resolved)
        return 1

    changed = 0
    for index, node in enumerate(nodes):
        if isinstance(style, int) and index != style:
            continue
        if isinstance(style, str) and style.lower() not in _style_key(node):
            continue
        socket = node.inputs["Material"]
        for link in list(socket.links):
            node.id_data.links.remove(link)
        socket.default_value = resolved
        changed += 1
    return changed


def assign_materials(target: Any, scheme: str = "chemistry") -> dict[str, str]:
    """Assign a different material to each of a molecule's styles.

    Different molecule classes want different surface qualities: a ligand that
    is glossier than its protein separates visually without needing an outline
    or a colour change. Gala infers the class from the style, because a
    cartoon is almost always polymer and a ball-and-stick almost always is not.

    Parameters
    ----------
    target : Molecule, bpy.types.Object, or AtomStructure
        What to assign to.
    scheme : str, optional
        A key of :data:`MATERIAL_SCHEMES`.

    Returns
    -------
    dict[str, str]
        Mapping of style name to the preset that was assigned. Empty if the
        target has no Molecular Nodes styles.

    Raises
    ------
    ValueError
        If ``scheme`` is unknown.
    """
    if scheme not in MATERIAL_SCHEMES:
        raise ValueError(
            f"unknown material scheme {scheme!r}; choose from {sorted(MATERIAL_SCHEMES)}"
        )
    mapping = MATERIAL_SCHEMES[scheme]

    nodes = _style_nodes(target)
    if not nodes:
        assign_material(target, get_material("protein"))
        return {}

    assigned: dict[str, str] = {}
    for node in nodes:
        key = _style_key(node)
        preset = next(
            (value for fragment, value in mapping.items() if fragment in key),
            mapping.get("default", "protein"),
        )
        socket = node.inputs["Material"]
        for link in list(socket.links):
            node.id_data.links.remove(link)
        socket.default_value = get_material(preset)
        assigned[key] = preset
    return assigned
