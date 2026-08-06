# Vignettes

Runnable end-to-end examples. Each is a standalone script in
[`vignettes/`](https://github.com/maomlab/blender_gala/tree/main/vignettes)
that CI executes on every push, so none of them can drift from the code.

Run one:

```bash
blender --background --python vignettes/01_publication_figure.py
```

or all of them:

```bash
make vignettes
```

They write their output into `docs/images/`.

## 1. A publication figure

![01_publication_figure](images/01_publication_figure.png)

`01_publication_figure.py` — load a structure, style it, and go from the
default scene to a transparent-background 300 dpi figure in one
`publication_setup` call. Shows the render preset, the lighting rig and the
report it returns.

## 2. A binding site

![02_binding_site](images/02_binding_site.png)

`02_binding_site.py` — find every interaction between a ligand and its pocket,
draw them as dashed lines, quote the polar distances, and label the closest
residues on translucent cards. A cool neutral protein against warm ligand
carbons, so the eye goes where the figure is about, and the contacting residues
as ball-and-stick at half the ligand's radii and coloured by element, so the
dashes land on atoms you can identify without competing with the ligand. The camera looks in along the direction the
pocket opens, computed from the structure, and frames the site rather than the
protein. The full Objective 2 workflow.

## 3. Measuring

![03_measurements](images/03_measurements.png)

`03_measurements.py` — the distance that decides whether a GPCR is activated.
Aranda-García et al.
([Nat Commun 16, 2020](https://doi.org/10.1038/s41467-025-57034-y)) define the
state of a class A receptor by the distance between two alpha carbons, one on
TM2 and one on TM6, and give the thresholds that sort a structure into closed,
intermediate or open. The vignette measures exactly that on the adenosine A2A
receptor in both states — 5UIG with an antagonist bound, 6GDG coupled to
mini-Gs — superposing them with Kabsch onto a common frame first, so what is
left is the receptor's own motion: 12.70 Å closed, 17.59 Å open, TM6 out by
4.9 Å. Drawn in the paper's own orientation — a side view in the membrane
plane, helices vertical, TM6 on the left and TM2 on the right — built from the
bundle's principal axis rather than from whichever way the crystals point. It also shows what happens when a selection is ambiguous, and how
`measure` dispatches on two, three or four atoms to give a distance, an angle
or a dihedral.

## 4. AlphaFold confidence

![04_alphafold_confidence](images/04_alphafold_confidence.png)

`04_alphafold_confidence.py` — fetch human p53 from the AlphaFold database,
colour it by pLDDT with the official confidence bands, print the legend, and
render only the parts worth trusting. A real prediction rather than a fixture,
because the point of the bands is that confidence is uneven: p53's DNA-binding
core comes out dark blue and its disordered arms orange.

## 5. Compositing passes

![05_compositing_passes](images/05_compositing_passes.png)

`05_compositing_passes.py` — enable cryptomatte and Z, render to a multilayer
EXR, and set up depth of field and depth cueing so the figure can be adjusted
after the render.

## 6. A turntable

![06_turntable](images/06_turntable.webp)

`06_turntable.py` — an orbiting animation with framing that holds for every
frame. The camera is parented to a pivot at the molecule's centre and the
lights are not, so the structure turns *through* the light rather than being lit
identically from every angle, which is what makes its shape read.

The animation above is the turn itself — 60 frames at 25 fps, in WebP rather
than GIF so that it keeps a real alpha channel and the full colour range at a
fraction of the size. It is built by `make turntable`, which is not part of
`make vignettes`: a hundred-odd Cycles frames is more than a smoke test on
every push should be doing.

## 7. Electrostatics

![07_electrostatics](images/07_electrostatics.png)

`07_electrostatics.py` — barnase and barstar, each solved on its own with
PDB2PQR and APBS, then opened out like a book so both binding faces are in
view. Barnase's is a patch of positive, barstar's is a patch of negative, and
the two are shaped like each other — which is why they associate as fast as
they do. The vignette also puts a number on it: the mean potential over each
partner's interface against the mean over the rest of its surface. Needs
`apbs` and `pdb2pqr`; `pip install apbs-binary pdb2pqr`, or `make apbs`.
