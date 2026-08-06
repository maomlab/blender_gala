# Blender Gala

**Structural biology visualization tools for Blender.**

![The Gala sidebar open in Blender, with arrows to the figures it produces:
publication scenes, interactions, measurements, colouring by data, compositing
passes and animation](images/hero.png)

[Molecular Nodes][mn] brought molecules into Blender and gave them Blender's
rendering. Blender Gala adds the day-to-day work that sits either side of that:
getting from a freshly imported structure to a publication-ready image, and
measuring and annotating what is in the scene.

```python
import molecularnodes as mn
import blender_gala as gala

mol = mn.Molecule.fetch("1ake").add_style("cartoon")

gala.publication_setup(mol, preset="figure")
gala.find_interactions(mol, "ligand", "protein")
gala.distance(mol, "resi 15 and name CA", "resi 90 and name CA", draw=True)
gala.color_by_plddt(mol)
```

Everything is also available from a **Gala** tab in the 3D View sidebar.

## Where to start

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](guide/getting-started.md)**

    Install it, load a structure, render a figure.

-   :material-magnify: **[Selection language](guide/selections.md)**

    PyMOL syntax: `byres (protein within 4 of ligand)`.

-   :material-camera: **[Publication scenes](guide/scenes.md)**

    Render settings, lighting, materials, camera.

-   :material-vector-line: **[Interactions](guide/interactions.md)**

    Hydrogen bonds, salt bridges, stacking — found and drawn.

-   :material-ruler: **[Measurement and labels](guide/measurement.md)**

    Distances, angles, dihedrals, and text that stays readable.

-   :material-palette: **[Colouring by data](guide/colouring.md)**

    AlphaFold confidence, B-factors, your own CSV.

</div>

## Design

Gala is opinionated in a few places, and [Design decisions](design.md) says why
— why the default view transform is `Standard` rather than `AgX`, why
measurements are real geometry rather than viewport overlays, why no Python
wheels are bundled.

## Requirements

| | |
| --- | --- |
| Blender | **5.1 or newer** |
| Molecular Nodes | **4.5 or newer**, for anything molecule-aware |
| Python dependencies | none bundled |

Those are minimums for the whole feature set, not a starting point to negotiate
from: Gala is written against the Molecular Nodes 4.5 API, and Molecular Nodes
4.5 requires Blender 5.1. The extension manifest declares the same floor, so
Blender will not install Gala where it could not work fully.

Older pairings are not partially supported. Blender 4.2 can only install
Molecular Nodes 4.4, which has no annotation manager and so cannot draw 2D
overlay labels at all.

[mn]: https://bradyajohnston.github.io/MolecularNodes
