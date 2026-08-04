# Getting started

## Install

Blender Gala needs **Blender 4.2 LTS or newer**, and
[Molecular Nodes](https://extensions.blender.org/add-ons/molecularnodes/) for
anything that touches a molecule.

1. *Edit → Preferences → Get Extensions*, search for **Molecular Nodes**,
   install it.
2. Download `blender_gala-<version>.zip` from the
   [releases page](https://github.com/blender-gala/blender_gala/releases).
3. *Edit → Preferences → Add-ons → ▾ → Install from Disk…*, choose the zip.
4. Restart Blender.

Press `N` in the 3D View and you should see a **Gala** tab.

!!! note "No bundled dependencies"

    Gala ships no Python wheels. It uses the numpy that comes with Blender and
    the biotite and scipy that come with Molecular Nodes. Two extensions each
    putting their own copy of biotite on `sys.path` is a real and confusing
    failure mode, and this avoids it.

## Your first figure

Open the Scripting workspace and run:

```python
import molecularnodes as mn
import blender_gala as gala

# Molecular Nodes loads and styles the molecule.
mol = mn.Molecule.fetch("1ake")
mol.add_style("cartoon")

# Gala sets up everything needed to render it well.
report = gala.publication_setup(mol, preset="figure")
print(report)
```

`publication_setup` prints a summary of what it changed:

```
Blender Gala publication setup (figure)
  resolution : 2000 x 2000
  gpu        : Cycles is using METAL on Apple M2 Max (GPU - 30 cores).
  lighting   : three_point
  origin     : centroid
  materials  : cartoon->protein
  passes     : combined, z, normal, cryptomatte_object, cryptomatte_material, cryptomatte_asset
```

Then render as usual (`F12`), or from Python:

```python
gala.render("figure.png")
```

The background is transparent and the output is 16-bit RGBA PNG, so the figure
drops onto any page colour.

## What just happened

`publication_setup` is a sequence of steps, each separately callable. In order:

1. **Origin** moved onto the molecule and the molecule moved to the world
   origin, so everything sized from its bounding sphere is computed against
   final coordinates.
2. **Render engine** set to Cycles with the preset's sampling, GPU detection,
   denoising and a transparent film.
3. **Colour management** set to `Standard`, which preserves the colours you
   chose (see [Design decisions](../design.md)).
4. **Lighting**: a three-point rig sized from the molecule.
5. **Materials** assigned per style — matte protein, glossier ligand.
6. **Camera** created and framed on the molecule.
7. **Passes** enabled (cryptomatte, Z, normal) and the compositor wired up.

Any of those can be run on its own:

```python
gala.setup_render(preset="print")
gala.three_point_lighting(mol, energy=1.4, softness=1.5)
gala.scene.hdri_lighting("courtyard", strength=0.8)
gala.assign_materials(mol, scheme="glossy")
gala.set_origin_to_geometry(mol, method="mass")
gala.frame_target(mol, viewpoint="front")
```

And any of them can be skipped:

```python
gala.publication_setup(
    mol,
    preset="draft",
    lighting_style="hdri",
    material_scheme=None,   # keep the materials you already set up
    origin_method=None,     # keep your origin
    frame_camera=False,     # keep your camera
)
```

## Presets

| Preset | Resolution | Samples | For |
| --- | --- | --- | --- |
| `draft` | 960 × 960 | 64 | Composition checks; noisy and fast |
| `figure` | 2000 × 2000 | 512 | A journal figure at 300 dpi up to 6.7 in |
| `print` | 4000 × 4000 | 1024 | Full-page print |
| `poster` | 6000 × 6000 | 1024 | Conference posters |
| `presentation` | 1920 × 1080 | 256 | 16:9 slides |

Blender has no DPI setting, so presets are in pixels. To convert: *pixels =
inches × dpi*. A single-column figure at 3.3 in and 300 dpi is 990 px; a
double-column one at 6.7 in is 2010 px.

Presets are dataclasses, so you can start from one:

```python
from blender_gala.scene.presets import PRESETS
custom = PRESETS["figure"].scaled(0.5)      # 1000 x 1000, same sampling
gala.setup_render(preset=custom)
```

## Doing the same thing without Python

The **Gala** sidebar tab has the same features:

- **Scene Setup** — the preset, engine and one-click *Publication Setup*
- **Origin and Camera**, **Lighting and Materials**, **Passes and Compositing**
- **Interactions**, **Measure**, **Label**, **Colour**
- **Clean Up** — remove everything Gala added

Everything Gala creates lives in a `Gala` collection with per-category
children, so it can be hidden, excluded from a view layer, or deleted as a
unit.

## Next

- [Selection language](selections.md) — how to say what you mean
- [Interactions](interactions.md) — find and draw contacts
- [Colouring by data](colouring.md) — AlphaFold confidence and your own values
