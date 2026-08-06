# Blender Gala — Specification

Living record of design decisions and requirements. Derived from `OBJECTIVES.md`.

---

## 1. Scope and positioning

Blender Gala ("Gala") is a Blender extension that complements
[Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes) (MN). MN owns
*import, geometry-node styles and molecular data*. Gala owns *the day-to-day
structural-biology tasks that sit either side of that*: getting from a freshly
imported molecule to a publication-ready image, and measuring/annotating what is
in the scene.

**Non-goals.** Gala does not re-implement molecule loading, trajectory handling,
geometry-node styles or MN's material library. Where MN already provides a
primitive (`mn.Canvas`, `mn.scene.Cycles`, `mn.Material`, the annotation
framework) Gala builds on it rather than around it.

### 1.1 Naming

The upstream project is **Molecular Nodes** (`molecularnodes`), not
"MoleculeNodes" as written in `OBJECTIVES.md`. Gala uses the correct name
throughout.

---

## 2. Target environment

| Item | Decision |
| --- | --- |
| Blender | 5.1 minimum — the floor for Molecular Nodes 4.5, which Gala is written against — developed and tested against 5.2 LTS |
| Python | Blender's bundled interpreter (3.13 on Blender 5.2) |
| Molecular Nodes | ≥ 4.5, optional at import time, required at run time for molecule-aware features |
| Packaging | Blender **extension** (`blender_manifest.toml`), installable as a zip |
| Third-party deps | None bundled. `numpy`/`scipy`/`biotite`/`databpy`/`MDAnalysis` are already on `sys.path` when MN is installed; `numpy` ships with Blender. Gala ships **no wheels**. |

**D-1. No vendored wheels.** Bundling wheels would duplicate MN's ~200 MB of
dependencies and risk version skew: two extensions putting different `biotite`
versions on `sys.path` is a real failure mode. Gala instead declares MN as a
soft dependency and degrades gracefully (see D-2).

**D-2. Soft dependency on Molecular Nodes.** Gala imports MN lazily through
`blender_gala.core.mn`, which resolves in order:
`bl_ext.blender_org.molecularnodes` → `bl_ext.user_default.molecularnodes` →
`molecularnodes`. Scene-level features (render, lighting, materials,
compositing) work without MN. Molecule-aware features raise a clear
`MolecularNodesUnavailable` error naming the install steps.

**D-2a. Registration is tolerant of a second copy.** Having the extension
installed *and* a checkout on `sys.path` is the normal developer setup.
Registering the second copy makes Blender unregister the first's classes behind
its back, and the first's `unregister` then raises `missing bl_rna attribute`
at shutdown. `core.registration` drops a stale registration before registering
and never raises while tearing down, so the two copies coexist.

**D-3. Dual importability.** Every intra-package import is relative, so the
package works both as `bl_ext.user_default.blender_gala` (installed extension)
and as a plain `blender_gala` on `sys.path` (tests, scripting).

---

## 3. Data model

### 3.1 How Gala reads a molecule

MN stores an `AtomArray`/`AtomArrayStack` on the Python `Molecule` object and a
1:1 index-aligned Blender mesh — vertex *i* is atom *i*. Gala relies on this
alignment and nothing else:

* **Chemistry** (element, `atom_name`, `res_id`, `res_name`, `chain_id`,
  `b_factor`, hetero flags) comes from the biotite `AtomArray`. String
  annotations are readable there; on the Blender mesh they are integer-encoded
  and lossy.
* **Geometry** comes from the Blender object's `position` attribute mapped
  through `obj.matrix_world`. This is authoritative: it survives MN's centring,
  user transforms and the world-scale factor.

**D-4. `AtomStructure` adapter.** `blender_gala.core.entity.AtomStructure`
wraps `(bpy.types.Object, AtomArray)` and is the single input type for every
selection, measurement and interaction routine. It can be constructed from an
MN `Molecule`, from a bare `bpy.types.Object` created by MN, or from a plain
`AtomArray` (headless, no Blender object — used heavily in tests).

### 3.2 Units

MN uses a world scale of 0.01: 1 Å = 0.01 Blender units. Gala never hard-codes
this. `core.units` reads the scale from the object when possible and exposes
`angstrom_to_bu()` / `bu_to_angstrom()`. **All Gala public API distances are in
ångström**; conversion happens at the Blender boundary only.

---

## 4. Selection language

**D-5. PyMOL-compatible selection strings.** Structural biologists already know
PyMOL syntax; requiring them to learn a new one is the main friction in
Blender-based workflows. Gala implements a tokenizer + recursive-descent parser
producing a boolean mask over an `AtomArray`.

Grammar (implemented in `core/selection.py`):

```
selection := or_expr
or_expr   := and_expr (("or" | "|") and_expr)*
and_expr  := not_expr (("and" | "&") not_expr)*
not_expr  := ("not" | "!") not_expr | modifier
modifier  := primary (("within" NUM "of" | "around" NUM | "byres" | "expand" NUM) ...)*
primary   := "(" selection ")" | keyword_sel | macro | identifier_macro
```

| Category | Tokens |
| --- | --- |
| Property | `chain`, `resi`/`resid`/`residue`, `resn`/`resname`, `name`, `elem`/`element`, `index`, `id`, `b`, `q`, `segi`, `alt`, `ss` |
| Macro | `all`, `none`, `protein`/`polymer`, `nucleic`, `backbone`, `sidechain`, `water`/`solvent`, `hetatm`/`hetero`, `ligand`, `ions`, `metals`, `hydro`, `donors`, `acceptors`, `polar`, `carbon` |
| Operator | `and`, `or`, `not`, `within N of S`, `around N`, `byres S`, `expand N`, `first`, `last` |
| Value forms | lists `A+B+C`, ranges `10-20`, open ranges `10-`, wildcards `CA*`, negative resi `\-5` |
| Numeric | `b > 70`, `b < 50`, `q = 1.0` (`>`, `<`, `>=`, `<=`, `=`, `!=`) |

`+` and `,` are value separators inside a property's argument only, never
boolean operators, so `chain A+B` is one selector rather than two.

Comparisons are case-insensitive for chain/residue/atom names, matching PyMOL.
`within` uses a `scipy.spatial.cKDTree` when available and a vectorised numpy
fallback otherwise.

**D-6. Selections are pure functions of an `AtomArray`.** No Blender state is
touched, so the parser is unit-testable outside Blender and reusable for both
molecules and trajectories.

**D-6a. `select()` accepts anything structure-like.** Users naturally pass
the `Molecule` they already have. Requiring `.array` meant a `Molecule`
found no annotations and returned an all-false mask — wrong answers, no
error. `core.selection.as_atom_array` unwraps a `Molecule`, an
`AtomStructure` or an `AtomArrayStack` before evaluation.

---

## 5. Objective 1 — publication-ready scenes

`blender_gala.scene.publication_setup(...)` is the single entry point; each
sub-step is independently callable and independently tested.

### 5.1 Render settings (`scene/render.py`)

* Engine Cycles; `samples` default 128 (preview) / 1024 (final) via preset.
* GPU: probes `OPTIX → CUDA → HIP → METAL → ONEAPI`, activates all non-CPU
  devices, falls back to CPU with a warning. (MN has `enable_optimal_gpu`; Gala
  reimplements because MN's version silently prefers the *first* backend that
  does not raise and does not report what it chose. Gala returns a
  `GPUReport` naming backend and devices, which the UI surfaces.)
* Denoising: `OPENIMAGEDENOISE`, `use_denoising=True`, GPU denoise when
  available, plus viewport denoising (`use_preview_denoising`).
* Adaptive sampling on, `adaptive_threshold=0.01`, `use_light_tree=True`.
* Transparent film: `render.film_transparent = True`; PNG RGBA output.
* Resolution: preset-driven, default 2000 × 2000 at 100 % (see D-7).

**D-7. Presets are journal-figure oriented.** `"draft"` (960², 64 spp),
`"figure"` (2000², 512 spp — a 300 dpi single-column figure at ~6.7 in),
`"print"` (4000², 1024 spp), `"poster"` (6000², 1024 spp). Resolution in
*pixels*, since Blender has no DPI concept; the docs give the inch/dpi maths.

### 5.2 Colour management (`scene/render.py`)

**D-8. Default view transform is `Standard`, not `AgX`/`Filmic`.** Molecular
figures use categorical colours that must survive round-tripping to a journal;
AgX desaturates and shifts hue. `Standard` + `look="None"` + sRGB display keeps
the colour a user picked. `AgX` is offered as an opt-in for cinematic renders.
Sequencer colour space `sRGB`, `exposure=0`, `gamma=1`.

### 5.3 Origin (`scene/origin.py`)

`set_origin_to_geometry(obj, method=...)` with `centroid` (unweighted mean of
atom positions), `mass` (mass-weighted), `bounds` (bounding-box centre). Moves
the object's origin without moving the geometry in world space (offsets
`matrix_world.translation` and shifts mesh data by the inverse), so downstream
rotations pivot on the molecule. Optionally also moves the object to the world
origin (`move_to_world_origin=True`).

### 5.4 Lighting (`scene/lighting.py`)

**D-9. Ship a native three-point rig rather than depending on the bundled
`Tri-lighting` add-on.** The add-on is not enabled by default, its operator is
context-sensitive (requires an active object and a 3D View), and it exposes no
handle for later editing. Gala's `three_point_lighting()` builds key/fill/rim
area lights parented to a single `GALA_LightRig` empty, sized and positioned
from the target's bounding sphere, with documented defaults:

| Light | Angle (azimuth, elevation) | Relative power | Size |
| --- | --- | --- | --- |
| Key | +45°, +30° | 1.0 | 1.5 × radius |
| Fill | −60°, +5° | 0.35 | 2.5 × radius |
| Rim | 170°, +25° | 0.7 | 1.0 × radius |

Power scales with radius² so the rig looks identical for a peptide and a
ribosome. The empty can be rotated to re-light the whole scene at once. If the
`Tri-lighting` add-on *is* enabled and `backend="tri_lighting"` is requested,
Gala delegates to it.

**D-9a. Base power is calibrated, not guessed.** The key light is 12 W at a
subject radius of 1 Blender unit and a distance of three radii. The first
implementation used 1000 W by analogy with Blender's default lamp and blew
out 99 % of every pixel; the value was then chosen by rendering the test
structure across a range and measuring clipping. A regression test asserts
the resulting irradiance stays between 0.5 and 5 W/m².

**D-9b. Rig lights are hidden from the camera.** An area light is an
emitting surface, and the rim light sits almost directly behind the subject,
so leaving it camera-visible puts a white disk across the background of
every figure. `visible_to_camera=True` restores it.

`hdri_lighting()` sets the world to an Environment Texture. **D-10. HDRIs come
from Blender's own `datafiles/studiolights/world` directory** (`forest.exr`,
`city.exr`, `interior.exr`, `night.exr`, `studio.exr`, `sunrise.exr`,
`courtyard.exr`) so Gala ships no multi-megabyte binaries. A user path may be
supplied. Strength, rotation and "visible to camera" (transparent film still
lit) are exposed.

### 5.5 Materials (`scene/materials.py`)

**D-11. One parameterised Principled BSDF builder, several chemistry presets.**
Rather than a fixed material list, `GalaMaterialSpec` is a dataclass
(`roughness`, `metallic`, `ior`, `subsurface_weight`, `subsurface_radius`,
`coat_weight`, `emission`, `alpha`, `ao_strength`, `ao_distance`) and
`build_material(spec, name)` constructs the node tree. Presets:

| Preset | Intent | Key parameters |
| --- | --- | --- |
| `protein` | matte, slightly waxy; reads well at cartoon scale | roughness 0.45, SSS 0.05, radius 0.5 Å |
| `ligand` | glossier so it pops against protein | roughness 0.25, coat 0.3 |
| `nucleic` | matte, low specular | roughness 0.6 |
| `surface` | translucent molecular surface | alpha 0.55, roughness 0.15, transmission-free (alpha blend renders faster and avoids caustics) |
| `metal` | ions/metal centres | metallic 1.0, roughness 0.2 |
| `lipid` | membranes | roughness 0.7, SSS 0.1 |
| `measurement` | dashes/labels; unlit so they read at any exposure | emission 1.0, shadow off |

**D-12. Ambient occlusion is a material-level effect.** Cycles has no AO pass
knob per material, so when `ao_strength > 0` the builder inserts a
`ShaderNodeAmbientOcclusion` multiplied into Base Color. This darkens crevices
between atoms — the standard "molecular AO" look — without a compositing step,
and is disabled by default (0.0) so it is opt-in.

**D-11a. Per-atom colour is read through Molecular Nodes' `MN Color Input`
group.** Reading the `Color` attribute is not a one-node job: a ball-and-stick
style renders atoms as *instanced* spheres, where a `GEOMETRY` attribute
lookup returns black and only an `INSTANCER` lookup works — while bonds and
cartoon meshes are the other way round. MN already solves this in a reusable
node group, so Gala instantiates that group rather than reimplementing it,
and stays correct if MN improves it. Without MN, a plain `GEOMETRY` attribute
is used, which is right for an ordinary mesh.

`assign_materials(molecule, scheme)` maps chemistry classes to materials by
selecting MN style nodes and setting their `Material` socket, falling back to
object material slots for non-MN meshes.

### 5.6 Compositing (`scene/compositing.py`)

**D-13. Passes first, node tree second.** `enable_passes()` turns on
`use_pass_cryptomatte_object`, `_material`, `_asset` (levels 6), `use_pass_z`,
`use_pass_mist`, `use_pass_normal`, `use_pass_combined`.

`setup_compositor()` builds a **`GALA Compositor` node group** containing, in
order: optional Denoise, optional depth-of-field via the Z pass (`Defocus`),
exposure/contrast, and an alpha-preserving output. It is inserted between Render
Layers and the output, and is idempotent — re-running rewires rather than
duplicating. It lives in `scene.compositing_node_group`, the Blender 5 form.

**D-13a. Blender 5 renamed half the compositor, and only Blender 5 is
supported.** Blender 5.0 rewrote the compositor: `CompositorNodeMapRange`,
`CompositorNodeMixRGB` and `CompositorNodeMath` are gone in favour of the
unified `ShaderNode*` types, `CompositorNodeOutputFile` swapped
`base_path`/`layer_slots` for `directory`/`file_name`/`file_output_items`, and
the Render Layers sockets became `Depth` and `Diffuse Color` rather than `Z`
and `DiffCol`. Gala used to accept either generation. Since the minimum is
Blender 5.1 (D-1) it targets the new names only — 5.1 and 5.2 were checked to
be identical on every one of them — and CI runs the suite on both so a future
rename cannot rot silently.

**D-14. Cryptomatte is set up for *downstream* use, not baked in.** Gala adds a
`CryptomatteV2` node per requested layer and a `File Output` node writing a
multi-layer EXR containing Combined + Z + Cryptomatte. That is the format
Nuke/Fusion/Krita/Blender's own compositor need to re-select an object after the
render. Baking a matte into the beauty pass would defeat the purpose.

`focus_on(target)` sets camera DOF (`use_dof`, `focus_object`, `aperture_fstop`)
and `depth_of_field(near, far)` drives the compositor Defocus/Z-mix so a chosen
ångström depth slab stays sharp.

---

## 6. Objective 2 — measurement and annotation

### 6.1 Representation strategy

**D-15. Measurements and interactions are real 3D geometry, not viewport
overlays.** MN's annotation framework draws with the GPU module and composites a
2D image. That is ideal for HUD-style text but it cannot receive light, cast
shadows, appear in cryptomatte, or be depth-sorted against the molecule. Gala
therefore creates **curve objects** (poly splines with `bevel_depth`) for every
line it draws — dashed or solid — so a hydrogen bond behaves like any other
scene object.

**D-16. Text labels get both treatments.** `label_3d()` creates a real `FONT`
object ("in-scene card", optionally with a rounded backing plane — actually
rounded, since D-16 claimed it long before the geometry did — and a
`COPY_ROTATION` billboard constraint — not `TRACK_TO`, which aims each label at
the camera and then rolls it towards world +Y, so labels in different places
come out tipped by different amounts). `label_hud()` registers an MN annotation so
the label is a resolution-independent 2D overlay for compositing. The objective
asked for both; they solve different problems.

Values — an interaction's distance, a measurement's number — get a backing too,
as a pill rather than a rounded rectangle and in a cooler tint. White text over
a pale molecule is unreadable without one, and the different outline says which
kind of label it is without anyone having to read it.

**D-17. Everything Gala creates lives under a `Gala` collection**, sub-divided
into `Gala/Interactions`, `Gala/Measurements`, `Gala/Labels`, `Gala/Lighting`.
Objects carry a `gala_type` custom property. This makes the output selectable,
hideable and removable as a unit, and makes `clear_*()` operations safe.

### 6.2 Interaction detection (`interactions/`)

**D-18. Geometry-based detectors implemented natively, PLIP optional.** PLIP is
a heavy dependency (openbabel) that cannot be assumed inside Blender. Gala
implements the interaction geometry directly with numpy/scipy, using
PLIP-derived criteria, and *additionally* accepts a PLIP result object when the
user has PLIP installed (`interactions.plip`).

Default criteria (all user-overridable, ångström and degrees):

| Interaction | Criteria |
| --- | --- |
| Hydrogen bond (with H) | D–H···A distance H···A ≤ 2.5, D···A ≤ 3.5, angle D–H···A ≥ 130° |
| Polar contact (no H) | D···A ≤ 3.5 between N/O/S/F donor–acceptor pairs, excluding same-residue backbone |
| Salt bridge | charged-group centroid distance ≤ 5.5 (Asp/Glu carboxylate, Lys NZ, Arg guanidinium, His ND1/NE2, ligand formal charges) |
| Hydrophobic | C···C ≤ 4.0, both atoms apolar (no polar neighbour) |
| π-stacking | ring-centroid distance ≤ 5.5, inter-plane angle ≤ 30° (parallel) or 60–90° with offset ≤ 2.0 (T-shaped) |
| Cation–π | ring centroid to cation ≤ 6.0, offset ≤ 2.0 |
| Halogen bond | X···A ≤ 4.0, C–X···A 140–180°, X···A–Y 90–180° |
| Metal coordination | metal···(N/O/S) ≤ 3.0 |

**D-18a. Ligand chemistry is perceived, not tabled.** A residue-name table
cannot know that a novel inhibitor has a thiazole ring or a carboxylate, and
the ligand is usually what the figure is about. `interactions/perception.py`
builds a distance-based bond graph (metals excluded, since their coordination
distances overlap covalent ones and would fuse a whole site into one
component), finds 5- and 6-membered planar rings by shortest-cycle search,
and derives formal charge from connectivity: carboxylate, phosphate and
sulfate as negative; guanidinium, amidinium and quaternary nitrogen as
positive.

**D-18b. Partial charges are never treated as formal charges.** Molecular
Nodes populates the `charge` annotation with force-field partial charges,
where every backbone carbonyl carbon reads about +0.6. An early version used
`|charge| > 0.5` as a fallback and reported a salt bridge on essentially
every residue in the protein. That fallback is gone.

Detection returns a list of `Interaction` dataclasses
(`kind`, `atom_i`, `atom_j`, `distance`, `angle`, `point_i`, `point_j`,
`label`), which is a plain data structure — testable without Blender, and the
input to drawing.

**D-19. Interactions are found between two selections.** `find_interactions(structure,
sel_a, sel_b, kinds=...)` mirrors PyMOL's mental model (`polar_contacts` between
a ligand and everything around it). Self-interactions within one selection are
supported by passing the same selection twice, with an
`exclude_same_residue` flag on by default.

### 6.3 Measurement (`measure/measurements.py`, `measure/draw.py`)

`distance(structure, sel_a, sel_b)`, `angle(a, b, c)`, `dihedral(a, b, c, d)`.
Each selection must resolve to exactly one atom, or is reduced by an explicit
`reduce=` policy (`"single"` (default, raises on ambiguity), `"centroid"`,
`"first"`, `"closest"`). This mirrors PyMOL's measurement wizard where the user
picks atoms one at a time, while remaining scriptable.

Each returns a `Measurement` dataclass with the value **and** the world-space
points, and `draw=True` creates the dashed line / angle arc / dihedral wedge
plus a value label. Angles draw as an arc between the two rays; dihedrals draw
the two half-planes and the arc between them.

### 6.4 Data-driven colouring (`color/`)

**D-20. Colours are written to the mesh `Color` attribute** (`FLOAT_COLOR`,
point domain) — the attribute MN's styles already read — so recolouring works
with every MN style.

One piece of node surgery is unavoidable, and was missing until it was caught
rendering a whole vignette flat pink. The styles *read* that attribute, but MN's
tree *writes* it: importing a molecule with a style wires a `Set Color` node
that stores a generated colour (`Color Common` over a random per-entity colour,
or `Color pLDDT`) over the mesh's own, between the mesh and the style. Writing
the attribute is therefore invisible on its own. `write_colors` mutes that node,
rather than deleting it, so MN's colouring is one click away in the geometry
nodes editor.

* `color_by_plddt(molecule)` — reads pLDDT from `b_factor`, applies the official
  AlphaFold DB bands: ≥90 `#0053D6` (very high), 70–90 `#65CBF3` (confident),
  50–70 `#FFDB13` (low), <50 `#FF7D45` (very low). `mode="banded"` (default,
  matches the AFDB viewer) or `mode="continuous"` (smooth ramp between band
  colours). Auto-detects 0–1 vs 0–100 pLDDT scaling.
* `color_by_attribute(molecule, values, ...)` — generic: values may be a
  per-atom array, a `{res_id: value}` mapping, a `(chain, res_id) -> value`
  mapping, or a column of a CSV. Normalised by `vmin`/`vmax` and mapped through
  a colormap.
* `color_by_selection(molecule, {"chain A": "#FF0000", ...})` — categorical.
* Colormaps are implemented natively (no matplotlib dependency at run time):
  `viridis`, `plasma`, `magma`, `inferno`, `cividis`, `coolwarm`, `RdYlBu`,
  `bwr`, plus `alphafold`. Sampled from the reference control points and
  linearly interpolated; values are written in **linear** space (Blender colour
  attributes are linear, hex input is sRGB — conversion is applied).

---

## 7. Blender UI

**D-21. The Python API is the product; the UI is a thin shell.** Every operator
does nothing but validate context and call one API function, so behaviour is
testable without simulating UI events. Panels live in the 3D View sidebar under
a **Gala** tab: *Scene Setup*, *Materials & Lighting*, *Interactions*,
*Measure*, *Label*, *Colour*. Settings are stored in a
`bpy.types.Scene.gala` `PropertyGroup` so they survive save/load.

---

## 8. Engineering

| Concern | Decision |
| --- | --- |
| Tests | `pytest`, executed **inside Blender** (`blender --background --python tests/run_tests.py`). `bpy`-free modules are additionally importable under system Python. |
| Test deps | Installed into `.blender-deps/` by `make dev-deps` and injected via `sys.path`; nothing is written into the Blender install or MN's wheel directory. |
| Fixtures | Three synthetic structures committed under `tests/data/`, generated by `make_fixtures.py` with exactly known geometry so tests assert real numbers. No network access in tests. |
| Scene reset | Tests purge data-blocks by hand. `bpy.ops.wm.read_factory_settings` disables every extension, and Blender then syncs the extension wheel directory to match — uninstalling Molecular Nodes' Python dependencies for the rest of the session, so every later test fails on a missing biotite. |
| Lint/format | `ruff` (lint + format), line length 88. |
| Types | `mypy` on `blender_gala/`, with `bpy`/`biotite`/`scipy` treated as untyped third-party. `fake-bpy-module` supplies `bpy` stubs in CI. |
| CI | GitHub Actions: lint + typecheck job on system Python; test job downloading Blender 5.1 and 5.2 LTS and running the headless suite on both. |
| Build | `make build` produces `dist/blender_gala-<version>.zip` via `blender --command extension build`. |
| Versioning | SemVer, single source of truth in `blender_manifest.toml` — Blender requires a literal version there, so everything else derives from it: `blender_gala.__version__` reads it at run time, `make help` greps it, and `[tool.hatch.version]` reads it for the wheel rather than `pyproject.toml` restating it. |
| Docs | MkDocs Material with `use_directory_urls: false`, so the built site is browsable straight off disk — clean URLs point at directories, which only a web server resolves. `scripts/check_links.py` fails the build on any internal link that does not resolve. API reference generated with `mkdocstrings`; vignettes are executable Python scripts that are *run* in CI and produce the images embedded in the docs. |

**D-22. Vignettes are executable and rendered by CI.** A tutorial that has
drifted from the code is worse than no tutorial. Each vignette in `vignettes/`
is a runnable script; `make vignettes` executes them headlessly and writes
images into `docs/`, so a broken example fails the build.

---

## 9. Public API surface

```python
import blender_gala as gala

# Objective 1
gala.publication_setup(mol, preset="figure", lighting="three_point",
                       materials="chemistry", transparent=True)
gala.scene.setup_render(preset="figure")
gala.scene.three_point_lighting(mol, energy=1.0)
gala.scene.hdri_lighting("studio", strength=1.0)
gala.scene.assign_materials(mol, "chemistry")
gala.scene.set_origin_to_geometry(mol, method="centroid")
gala.scene.setup_compositor(cryptomatte=True, depth_of_field=True)
gala.render("figure.png")

# Objective 2
gala.find_interactions(mol, "ligand", "protein within 5 of ligand")
gala.hydrogen_bonds(mol, "chain A", "chain B", draw=True)
gala.distance(mol, "chain A and resi 10 and name CA",
                   "chain A and resi 20 and name CA", draw=True)
gala.angle(mol, sel_a, sel_b, sel_c, draw=True)
gala.dihedral(mol, sel_a, sel_b, sel_c, sel_d, draw=True)
gala.label(mol, "resi 45", text="{resn}{resi}", style="card")
gala.color_by_plddt(mol)
gala.color_by_attribute(mol, values, cmap="viridis")
gala.select(mol, "byres (protein within 4 of ligand)")
```

---

## 10. Open questions

* **Trajectory support.** The selection engine works on any `AtomArray`, so
  per-frame measurements are feasible. Deferred until the static case is solid;
  the `AtomStructure` adapter is designed to accept a frame index.
* **Ensembles / CellPack.** Out of scope for v0.1.
* **Interaction updating on frame change.** v0.1 draws at the current frame. A
  depsgraph handler that re-runs detection per frame is a v0.2 candidate.
* **Framing on a sub-selection.** `frame_target` uses the whole target's
  bounding sphere, so one distant ion loosens the framing of everything else.
  Accepting a selection there is a small, obvious improvement.
* **Legend rendering.** `color_by_plddt` returns a legend as data; drawing it
  into the compositor as a colour bar is not yet implemented.
