# Design decisions

Gala is opinionated in a handful of places. The full record lives in
[`SPECIFICATION.md`](https://github.com/blender-gala/blender_gala/blob/main/SPECIFICATION.md);
this page covers the decisions a user is most likely to notice.

## `Standard`, not `AgX`

Blender's default view transform is `AgX`, which rolls off highlights
attractively and desaturates and shifts hue while doing it. That is right for
photography and wrong for a scientific figure: molecular figures use
*categorical* colour — a chain rainbow, an AlphaFold confidence band, a
highlighted mutation — and a tone mapper that changes those hues makes them
stop meaning what the legend says.

Gala defaults to `Standard` with no look. `AgX` remains one argument away when
highlight rolloff matters more than fidelity.

## Real geometry, not viewport overlays

Molecular Nodes has an annotation framework that draws with the GPU module and
composites a 2D image. It is excellent for HUD text, and Gala uses it for
exactly that (`label_hud`).

But an overlay cannot receive light, cast a shadow, appear in cryptomatte, or
depth-sort against the molecule. A hydrogen bond in a figure should behave like
every other object in the scene, so Gala draws interactions and measurements as
**curve objects** with a bevel — dashed or solid, materialised, renderable.

## Cryptomatte set up, not baked in

`setup_compositor` adds a Cryptomatte node per layer and writes a multilayer
EXR, but deliberately leaves those nodes unconnected to the output. Connecting
one would matte the beauty pass, which is the opposite of the point. The mattes
exist so that *after* a long render you can still isolate the ligand.

## No bundled Python wheels

Vendoring dependencies would duplicate Molecular Nodes' ~200 MB of wheels and
risk version skew — two extensions putting different versions of biotite on
`sys.path` is a real and very confusing failure mode.

Gala uses the numpy Blender ships and the biotite, scipy and databpy that come
with Molecular Nodes. Scene-level features work without Molecular Nodes at all;
molecule-aware features raise an error that names the install steps.

## PyMOL selection syntax

Structural biologists already think in PyMOL selections. Requiring a new syntax
is the main friction in Blender-based workflows, so Gala implements a real
parser for the PyMOL language rather than inventing something.

## Interaction criteria from PLIP, implemented natively

PLIP is the reference tool, but it depends on OpenBabel, which cannot be
assumed inside Blender's interpreter. Gala implements PLIP's published
geometric criteria directly with numpy and scipy, so the results are
comparable, and accepts real PLIP output when a user has it.

For ligands, rings and charged groups are perceived from connectivity rather
than looked up in a table — the ligand is usually what the figure is about, and
no table knows about a novel inhibitor.

### Partial charges are not formal charges

Molecular Nodes stores force-field partial charges in the `charge` annotation,
where a backbone carbonyl carbon reads about +0.6. An early version of the
salt-bridge detector treated those as formal charges and reported a salt bridge
on essentially every residue. Charge is now derived from connectivity.

## Ambiguous measurements are errors

A selection matching several atoms is rejected by default rather than silently
resolved. Measuring to *some* CA produces a number that looks right and is not,
and there is no way to notice from the output. Explicit `reduce=` policies
handle the cases where the ambiguity is intended.

## Light power scales with molecular size

The three-point rig derives light power from the square of the subject radius,
so the same call produces the same look on a 20-residue peptide and on a
ribosome. Without that, every new structure means re-tuning three lights.

## Everything lives in a `Gala` collection

One top-level collection with a child per category, and a `gala_type` custom
property on each object. That makes the output hideable, excludable from a view
layer, and safely removable — `clear_all` never has to guess what was Gala's.

## Ångström at the boundary

Every public function takes and returns ångström. Molecular Nodes imports at a
world scale of 0.01, and Gala reads that scale from the object rather than
hard-coding it, converting only where a value crosses into Blender. A 0.15 Å
dash radius is 0.15 Å whatever the scale.

## Vignettes are executed by CI

A tutorial that has drifted from the code is worse than no tutorial. Every
vignette is a runnable script that CI executes; a broken example fails the
build.
