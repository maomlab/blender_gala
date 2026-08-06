# Blender Gala

**Structural biology visualization tools for Blender.**

![The Gala sidebar open in Blender, with arrows to the figures it produces:
publication scenes, interactions, measurements, colouring by data, compositing
passes and animation](docs/images/hero.png)

[Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes) brought
molecules into Blender and gave them Blender's rendering. Blender Gala adds the
day-to-day work that sits either side of that: getting from a freshly imported
structure to a publication-ready image, and measuring and annotating what is in
the scene — the things PyMOL and Mol* do without being asked.

```python
import molecularnodes as mn
import blender_gala as gala

mol = mn.Molecule.fetch("1ake").add_style("cartoon")

gala.publication_setup(mol, preset="figure")      # camera, lights, materials, passes
gala.find_interactions(mol, "ligand", "protein")  # H-bonds, salt bridges, stacking
gala.distance(mol, "resi 15 and name CA", "resi 90 and name CA", draw=True)
gala.label(mol, "byres (protein within 4 of ligand)")
gala.color_by_plddt(mol)
```

Everything is also available from a **Gala** tab in the 3D View sidebar, so
none of it requires writing Python.

---

## What it does

### Publication-ready scenes, in one call

`publication_setup()` takes a molecule from "default scene with a protein in
it" to "render this and put it in a paper":

- **Cycles** configured for quality per unit time — adaptive sampling,
  OpenImageDenoise, light tree, caustics off — with **GPU auto-detection**
  across OPTIX, CUDA, HIP, METAL and oneAPI that *tells you what it picked*
  instead of quietly falling back to CPU.
- **Transparent background** with the output format set to RGBA, so the alpha
  channel actually survives being written to disk.
- **Colour management that keeps your colours.** The default view transform is
  `Standard`, not `AgX` — a chain rainbow or an AlphaFold confidence band has
  to still mean what the legend says.
- **Origin on the molecule**, so orbiting spins the structure in place rather
  than swinging it through a crystallographic arc.
- **Three-point studio lighting** whose power scales with the molecule's
  radius, so the same rig looks right on a peptide and on a ribosome — or
  **HDRI** environments taken from the ones Blender already ships.
- **Chemistry-aware materials**: a matte, faintly subsurface-scattering protein
  under a glossier ligand, so the ligand separates without an outline. Optional
  material-level ambient occlusion darkens the crevices between atoms.
- **Cryptomatte and Z passes** written to a multilayer EXR, so you can brighten
  just the ligand *after* a 40-minute render instead of starting over.
  Depth of field and PyMOL-style depth cueing are one argument each.

### Measuring and annotating

- **Interactions.** Hydrogen bonds, polar contacts, salt bridges, hydrophobic
  contacts, π-stacking, cation-π, halogen bonds and metal coordination, using
  PLIP's published geometric criteria. Ligand rings and charged groups are
  perceived from connectivity, so a novel inhibitor works as well as a standard
  residue. PLIP itself is supported as an optional backend.
- **Real geometry, not overlays.** Every interaction and measurement is a curve
  object, so it receives light, casts shadows, occludes correctly and appears
  in cryptomatte.
- **Measurements.** Distances, angles and dihedrals with the same guarantee as
  PyMOL's measurement wizard — a selection that matches more than one atom is
  an error, not a coin flip — plus explicit `reduce=` policies when you want a
  centroid.
- **Labels.** In-scene 3D text (optionally on a translucent card, optionally
  billboarded) *and* 2D compositing overlays via Molecular Nodes' annotation
  system. They solve different problems, so both are here.
- **Electrostatics.** [APBS](https://www.poissonboltzmann.org) run through
  PDB2PQR and painted onto a translucent molecular surface, the way the PyMOL
  APBS plugin does it — sampled where the surface actually is rather than at
  the atom centres, with buried atoms left uncoloured. Reads an existing
  OpenDX map too.

- **Data-driven colour.** AlphaFold pLDDT with the official confidence bands,
  B-factors, or any per-residue value from an array, a dict or a CSV, through
  colormaps implemented natively (no matplotlib inside Blender).

### PyMOL selection syntax, throughout

```python
gala.select(mol, "byres (protein within 4 of ligand)")
gala.select(mol, "chain A and resi 45-60 and not backbone")
gala.select(mol, "resn HIS and name ND1+NE2 and b > 70")
```

`chain`, `resi`, `resn`, `name`, `elem`, `index`, `b`, `q`, the macros
(`protein`, `nucleic`, `backbone`, `sidechain`, `water`, `ligand`, `ions`,
`aromatic`, `donors`, `acceptors`, …), `and`/`or`/`not`, `within N of`,
`around`, `byres`, `bychain`, `expand`, wildcards, ranges and numeric
comparisons all work as you would expect.

---

## Installation

**Requirements:** Blender 5.1 or newer, and
[Molecular Nodes](https://extensions.blender.org/add-ons/molecularnodes/) 4.5
or newer. Gala is written against the Molecular Nodes 4.5 API, and 4.5 itself
requires Blender 5.1; the extension manifest declares the same floor.

1. Install Molecular Nodes from *Edit → Preferences → Get Extensions*.
2. Download `blender_gala-<version>.zip` from the
   [releases page](https://github.com/maomlab/blender_gala/releases).
3. *Edit → Preferences → Add-ons → ▾ → Install from Disk…* and pick the zip.
4. Restart Blender. A **Gala** tab appears in the 3D View sidebar (`N`).

Gala bundles **no** Python dependencies. It uses the numpy that ships with
Blender and the biotite/scipy that come with Molecular Nodes, which avoids two
extensions putting different versions of the same library on `sys.path`.

Building from source:

```bash
git clone https://github.com/maomlab/blender_gala
cd blender_gala
make build     # -> dist/blender_gala-0.1.0.zip
make install   # build and install into your Blender
```

---

## A worked example

```python
import molecularnodes as mn
import blender_gala as gala

# 1. Load a ligand complex and style it.
mol = mn.Molecule.fetch("1stp")
mol.add_style("cartoon", selection="polymer")
mol.add_style("ball_and_stick", selection="not polymer")

# 2. A publication-ready scene.
report = gala.publication_setup(
    mol,
    preset="figure",           # 2000 x 2000, 512 spp
    lighting_style="three_point",
    material_scheme="chemistry",
    viewpoint="iso",
)
print(report)                  # what it did, including which GPU it found

# 3. What does the ligand touch?
contacts = gala.find_interactions(
    mol, "ligand", "protein", kinds=["hbond", "polar", "hydrophobic", "pi_stacking"]
)
for contact in contacts:
    print(contact)             # hbond: A/ASN23/ND2 - B/BTN1/O11 (2.91 A, 158 deg)

gala.draw_interactions(contacts, target=mol, label=True)

# 4. Label the binding site and measure the key contact.
gala.label(mol, "byres (protein within 4 of ligand)", style="card")
gala.distance(mol, "resn BTN and name O11", "resi 23 and name ND2", draw=True)

# 5. Render, with every pass saved for later compositing.
gala.setup_compositor(cryptomatte=True, file_output="passes/")
gala.render("figure.png")
```

---

## Documentation

- [Getting started](https://maomlab.github.io/blender_gala/guide/getting-started.html)
- [Selection language reference](https://maomlab.github.io/blender_gala/guide/selections.html)
- [Publication scenes](https://maomlab.github.io/blender_gala/guide/scenes.html)
- [Interactions and measurement](https://maomlab.github.io/blender_gala/guide/interactions.html)
- [Colouring by data](https://maomlab.github.io/blender_gala/guide/colouring.html)
- [API reference](https://maomlab.github.io/blender_gala/api/index.html)

Runnable end-to-end examples live in [`vignettes/`](vignettes/); CI executes
every one of them, so they cannot drift from the code.

`SPECIFICATION.md` records the design decisions and the reasoning behind them —
why `Standard` rather than `AgX`, why real geometry rather than overlays, why
no vendored wheels.

---

## Development

```bash
make dev        # linting and docs toolchain, in your system Python
make dev-deps   # pytest, into .blender-deps for Blender's Python
make test       # the suite, inside Blender
make check      # lint + typecheck + test, exactly what CI runs
make docs-serve # documentation with live reload
```

The test suite runs inside Blender because that is the only place `bpy` exists.
Most of it does not need Blender objects, though: the selection language,
interaction geometry, measurement maths and colour mapping all work on a bare
biotite `AtomArray`, and are tested against synthetic structures built with
exactly known geometry — a 2.80 Å hydrogen bond is 2.80 Å because it was placed
there.

CI runs the suite on Blender 5.1 — the oldest supported — and 5.2 LTS, since
compositor node names have moved between releases and both paths need to keep
working.

---

## Status

Version 0.1.0. The API is settling but not yet frozen; pin a version if you are
scripting figures for a paper.

## Licence

GPL-3.0-or-later, matching Blender and Molecular Nodes.

## Acknowledgements

Built on [Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes) by
Brady Johnston. Interaction criteria follow those published for
[PLIP](https://github.com/pharmai/plip). The selection language follows
[PyMOL](https://pymol.org). The confidence palette is AlphaFold DB's.
