# Blender Gala

**Structural biology visualization tools for Blender.**

[![Documentation](https://img.shields.io/badge/docs-maomlab.github.io%2Fblender__gala-0b7285?logo=materialformkdocs&logoColor=white)](https://maomlab.github.io/blender_gala/)
[![CI](https://github.com/maomlab/blender_gala/actions/workflows/ci.yml/badge.svg)](https://github.com/maomlab/blender_gala/actions/workflows/ci.yml)
[![Blender 5.1+](https://img.shields.io/badge/Blender-5.1%2B-ea7600?logo=blender&logoColor=white)](https://www.blender.org)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

Guides, the full API reference and every vignette are on the
**[documentation site](https://maomlab.github.io/blender_gala/)** —
start with [Getting started](https://maomlab.github.io/blender_gala/guide/getting-started.html).

![The Gala sidebar open in Blender, with arrows to the figures it produces:
publication scenes, interactions, measurements, colouring by data, compositing
passes and animation](docs/images/hero.webp)

[Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes) brought
molecules into Blender and gave them Blender's rendering. Blender Gala adds the
day-to-day work that sits either side of that: getting from a freshly imported
structure to a publication-ready image, and measuring and annotating what is in
the scene — the things PyMOL and Mol* do without being asked.

```python
from bl_ext.user_default import blender_gala as gala
from bl_ext.blender_org import molecularnodes as mn

mol = mn.Molecule.fetch("1ake").add_style("cartoon")

gala.publication_setup(mol, preset="figure")      # camera, lights, materials, passes
gala.find_interactions(mol, "ligand", "protein")  # H-bonds, salt bridges, stacking
gala.distance(mol, "resi 15 and name CA", "resi 90 and name CA", draw=True)
gala.label(mol, "byres (protein within 4 of ligand)")
gala.color_by_plddt(mol)
gala.load_session("figure.pse")                   # or open a PyMOL session
```

Everything is also available from a **Gala** tab in the 3D View sidebar, so
none of it requires writing Python. Sessions are additionally in **File ▸
Import** and **File ▸ Export**, where the rest of Blender's formats live.

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
  `highlight_matte()` cuts the figure again straight from that EXR — one chain
  kept, the rest darkened and drained of colour — in a scene with no molecule
  in it, so a talk that needs the same picture three times with a different
  subunit emphasised each time costs one render rather than three. Depth of
  field and PyMOL-style depth cueing are one argument each.

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

- **PyMOL sessions, both ways.** Open a `.pse` in Blender — molecules,
  representations, per-atom colours, secondary structure, selections,
  measurements and the camera — and write the scene back out as one. The
  format is parsed directly, so no PyMOL install is needed, and the reader
  refuses to import anything a pickle asks it to.

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
make build     # -> dist/blender_gala-0.2.0.zip
make install   # build and install into your Blender
```

---

## A worked example

```python
# Extensions import as `bl_ext.<repository>.<id>`, where the repository is
# where you installed it: `user_default` for a zip, `blender_org` for the
# extensions platform.
from bl_ext.user_default import blender_gala as gala
from bl_ext.blender_org import molecularnodes as mn

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

**<https://maomlab.github.io/blender_gala/>**

- [Getting started](https://maomlab.github.io/blender_gala/guide/getting-started.html)
- [Selection language reference](https://maomlab.github.io/blender_gala/guide/selections.html)
- [Publication scenes](https://maomlab.github.io/blender_gala/guide/scenes.html)
- [Interactions](https://maomlab.github.io/blender_gala/guide/interactions.html)
  and [measurement](https://maomlab.github.io/blender_gala/guide/measurement.html)
- [Colouring by data](https://maomlab.github.io/blender_gala/guide/colouring.html)
- [Electrostatics](https://maomlab.github.io/blender_gala/guide/electrostatics.html)
- [PyMOL sessions](https://maomlab.github.io/blender_gala/guide/pymol.html)
- [Compositing and passes](https://maomlab.github.io/blender_gala/guide/compositing.html)
- [The Blender UI](https://maomlab.github.io/blender_gala/guide/ui.html)
- [Vignettes](https://maomlab.github.io/blender_gala/vignettes.html)
- [API reference](https://maomlab.github.io/blender_gala/api/index.html)

Runnable end-to-end examples live in [`vignettes/`](vignettes/); CI executes
every one of them, so they cannot drift from the code. Eight cover Gala's own
work, from a publication figure to a PyMOL round trip; eight more put it in a
scene with the rest of Blender — a virus capsid and a lipid bilayer built with
geometry nodes, a material contact sheet and procedural surface shading, a
conformational morph and a camera move, and two pictures made to be looked at
rather than measured.

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

Version 0.2.0. The API is settling but not yet frozen; pin a version if you are
scripting figures for a paper.

## Licence

[GPL-3.0-or-later](LICENSE), matching Blender and Molecular Nodes.

Blender is GPL, and the Blender Foundation's position is that a Python add-on
using `bpy` is a derivative work that has to be distributed under a
GPL-compatible licence. It is also what
[extensions.blender.org](https://extensions.blender.org) requires of an
add-on: its upload check looks for `SPDX:GPL-3.0-or-later` specifically, so
this is the licence Gala can be published under there. See
[Extension Licenses](https://docs.blender.org/manual/en/latest/advanced/extensions/licenses.html).

## Acknowledgements

Built on [Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes) by
Brady Johnston. Interaction criteria follow those published for
[PLIP](https://github.com/pharmai/plip). The selection language follows
[PyMOL](https://pymol.org). The confidence palette is AlphaFold DB's.

Electrostatics are computed by [APBS](https://www.poissonboltzmann.org),
which solves the Poisson-Boltzmann equation, with
[PDB2PQR](https://www.poissonboltzmann.org) assigning the charges and radii it
needs. Gala shells out to both rather than reimplementing either: the numbers
on a potential map should be the ones the field has agreed on.

- Jurrus, E. *et al.* Improvements to the APBS biomolecular solvation software
  suite. *Protein Sci.* **27**, 112–128 (2018).
  [doi:10.1002/pro.3280](https://doi.org/10.1002/pro.3280)
- Dolinsky, T. J., Nielsen, J. E., McCammon, J. A. & Baker, N. A. PDB2PQR: an
  automated pipeline for the setup of Poisson–Boltzmann electrostatics
  calculations. *Nucleic Acids Res.* **32**, W665–W667 (2004).
  [doi:10.1093/nar/gkh381](https://doi.org/10.1093/nar/gkh381)

## How to cite

There is no paper for Blender Gala. Cite the software itself:

> O'Meara, M. *Blender Gala: structural biology visualization tools for
> Blender.* https://github.com/maomlab/blender_gala

with the version you used. GitHub's **Cite this repository** button fills that
in from [`CITATION.cff`](CITATION.cff), which is generated from the extension
manifest — so it cannot name a version that was never released.

A figure made with Gala is made with several other people's work, and the
citation that matters most is usually not this one. Cite alongside it:

- **[Blender](https://www.blender.org)** — the renderer.
- **[Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes)** —
  every molecule in the scene was imported and styled by it.
- **[APBS and PDB2PQR](https://www.poissonboltzmann.org)** — if the figure
  shows electrostatics, using the two references above.
- **[PLIP](https://github.com/pharmai/plip)** — if you report interactions
  found by `find_interactions`, whose criteria are theirs.

And cite the structures themselves: a PDB entry has an accession and a paper,
and a figure of one owes both.
