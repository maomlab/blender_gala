# Publication scenes

## One call

```python
report = gala.publication_setup(mol, preset="figure")
```

Everything below is a step inside that call, and each is separately usable.

## Render settings

```python
report = gala.setup_render(
    preset="figure",
    engine="CYCLES",
    transparent=True,
    use_gpu=True,
    denoise=True,
)
print(report)   # "Cycles is using OPTIX on NVIDIA RTX 4090."
```

What it configures:

- **Sampling** from the preset, with adaptive sampling and a noise threshold
  matched to the intended output size.
- **Denoising** with OpenImageDenoise, using albedo and normal, in both render
  and viewport.
- **Light tree** on, so a three-point rig plus an HDRI samples efficiently.
- **Caustics off.** Translucent molecular surfaces otherwise produce fireflies
  that no practical sample count clears.
- **Transparent film**, with the output set to 16-bit RGBA PNG. A transparent
  render saved as RGB silently loses its alpha, which is a confusing way to
  lose an afternoon.

### GPU detection

```python
report = gala.scene.enable_gpu()
if not report.enabled:
    print(report.message)   # says why, rather than silently using the CPU
```

Backends are tried in order: OPTIX, CUDA, HIP, METAL, oneAPI. The report names
the backend and the devices, and the sidebar shows it after a setup run.

### Colour management

```python
gala.scene.setup_color_management(view_transform="Standard")
```

`Standard` is the default rather than Blender's `AgX`. Molecular figures use
categorical colour — a chain rainbow, an AlphaFold confidence band, a
highlighted mutation — and a tone mapper that desaturates and shifts hue makes
those colours stop meaning what the legend says. `AgX` is available when
highlight rolloff matters more than fidelity:

```python
gala.scene.setup_color_management(view_transform="AgX", exposure=0.5)
```

## Origin

```python
gala.set_origin_to_geometry(mol, method="centroid", move_to_world_origin=True)
```

A structure imported at its crystallographic coordinates can have its origin
hundreds of ångström from any atom, so orbiting swings it through a huge arc
instead of spinning it in place. This moves the origin without moving the
geometry in world space.

| Method | Uses |
| --- | --- |
| `centroid` | Unweighted mean of atom positions; the usual choice |
| `mass` | Mass-weighted; differs meaningfully only with heavy atoms |
| `bounds` | Bounding-box centre; better for elongated molecules such as DNA |

A selection restricts the calculation, which is how you pivot a complex about
one chain or ignore a long disordered tail:

```python
gala.set_origin_to_geometry(mol, selection="chain A and not solvent")
```

## Lighting

### Three-point studio rig

```python
rig = gala.three_point_lighting(mol, energy=1.0, softness=1.0, rotation=0.0)
```

Key, fill and rim area lights parented to one empty:

| Light | Azimuth | Elevation | Relative power | Size |
| --- | --- | --- | --- | --- |
| Key | +45° | +30° | 1.0 | 1.5 × radius |
| Fill | −60° | +5° | 0.35 | 2.5 × radius |
| Rim | 170° | +25° | 0.7 | 1.0 × radius |

Power scales with the square of the molecule's radius, so the rig looks the
same on a 20-residue peptide and on a ribosome. Because everything is parented
to `GALA Light Rig`, rotating that empty re-lights the whole scene:

```python
import math, bpy
bpy.data.objects["GALA Light Rig"].rotation_euler.z = math.radians(45)
```

`softness` above 1 gives softer shadows; below 1 gives crisper ones that
emphasise surface detail. The layout itself can be replaced:

```python
from blender_gala.scene.lighting import LightSpec

gala.three_point_lighting(mol, specs=[
    LightSpec("Key",  azimuth=30,   elevation=45, power=1.0, size=2.0),
    LightSpec("Fill", azimuth=-90,  elevation=0,  power=0.5, size=3.0),
])
```

### HDRI

```python
gala.scene.hdri_lighting("studio", strength=1.0, rotation=45, visible_to_camera=False)
```

The HDRIs are the ones Blender already ships — `studio`, `courtyard`,
`interior`, `city`, `forest`, `night`, `sunrise`, `sunset` — so Gala adds no
multi-megabyte files. Any `.exr` or `.hdr` path works too.

`visible_to_camera=False` keeps a transparent film transparent while still
lighting the molecule, which is usually what a figure needs. `rotation` moves a
highlight without moving the camera.

```python
gala.scene.list_hdris()   # {"studio": "/path/to/studio.exr", ...}
```

Combining both gives the most forgiving setup for glossy or metallic
materials — a dim HDRI as ambient fill under a controllable rig:

```python
gala.publication_setup(mol, lighting_style="both", hdri="courtyard")
```

## Materials

```python
assigned = gala.assign_materials(mol, scheme="chemistry")
# {"cartoon": "protein", "ball_and_stick": "ligand"}
```

Different molecule classes want different surface qualities. A ligand that is
glossier than its protein separates visually without needing an outline or a
colour change. Gala infers the class from the style, because a cartoon is
almost always polymer and a ball-and-stick almost always is not.

| Preset | Intent |
| --- | --- |
| `protein` | Matte, faintly subsurface-scattering; reads well at cartoon scale |
| `ligand` | Glossier, with a clear coat, so it pops |
| `nucleic` | Matte and low-specular; long backbones stop glaring |
| `surface` | Translucent, alpha-blended (faster than transmission, no caustics) |
| `metal` | Metallic, for ions and metal centres |
| `lipid` | Membranes and detergent belts |
| `measurement`, `interaction`, `label` | Unlit; readable at any exposure |

Presets are dataclasses, so adjust rather than rebuild:

```python
from blender_gala.scene.materials import MATERIAL_PRESETS, build_material

spec = MATERIAL_PRESETS["protein"].with_(roughness=0.7, ao_strength=0.4)
matte = build_material(spec, name="My Protein")
gala.assign_material(mol, matte, style="cartoon")
```

### Ambient occlusion

Cycles has no per-material AO switch, so `ao_strength` inserts an ambient
occlusion node multiplied into the base colour. That darkens the crevices
between atoms — the look that makes a space-filling model read as solid rather
than as a pile of flat circles. It is off by default:

```python
spec = MATERIAL_PRESETS["protein"].with_(ao_strength=0.5, ao_distance=0.05)
```

## Camera

```python
gala.frame_target(mol, viewpoint="iso", margin=1.15)
```

Creates a camera if there is none (85 mm, a mild telephoto that keeps a
molecule's proportions honest) and backs it off until the molecule fits with
the given margin. Viewpoints: `front`, `back`, `left`, `right`, `top`,
`bottom`, `iso`, or an `(azimuth, elevation)` pair in degrees.

The fit is to the atoms as they project from that viewpoint, not to the sphere
around them, so `margin=1.0` puts the outermost atom on the frame edge and
`1.15` leaves 15% air. Fitting the bounding sphere — the usual shortcut — backs
the camera off far enough for the *most distant single atom* in any direction,
which on a typical protein wastes about half the frame.

The camera also aims at the middle of that silhouette rather than at the
centroid. The two coincide for a symmetrical molecule and separate for one with
a long tail on one side — an AlphaFold model with disordered arms, say — where
aiming at the centroid leaves a band of empty frame down the other side. One
atom well away from the rest still costs you the whole frame either way,
because it has to be in it: frame a selection instead if you would rather it
were not.

A turntable:

```python
gala.orbit(frames=120, target=mol)
gala.render("turntable.mp4", animation=True)
```

The camera is parented to a pivot at the molecule's centre, so the framing
computed by `frame_target` holds for every frame.

## Rendering

```python
gala.render("figure.png")
gala.render("frames/", animation=True)
```
