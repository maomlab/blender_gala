# The Blender UI

Everything Gala does is available from a **Gala** tab in the 3D View sidebar
(`N`). The panels are a thin shell: each operator validates its context and
calls exactly one Python API function, so nothing is available in the UI that
is not scriptable, and vice versa.

## Panels

### Scene Setup

The one-click **Publication Setup** button, plus the preset, engine and view
transform it will use. After a run, the status bar reports which GPU backend
was found.

Sub-panels:

- **Origin and Camera** — origin method, whether to move to the world origin,
  and the camera viewpoint.
- **Lighting and Materials** — three-point or HDRI, energy, softness, rig
  rotation, and the material scheme.
- **Passes and Compositing** — cryptomatte, the EXR output directory, depth of
  field, depth cue, and a *Render Still* button.

### Interactions

Two selection fields and a grid of interaction types. *Find Interactions*
detects and draws them; the trash button clears them.

### Measure

Select 2–4 atoms in Edit Mode and press *Measure*: Gala picks distance, angle
or dihedral from how many you selected. Or type selection strings separated by
`;` to do the same without leaving Object Mode.

### Label

Selection, template, level, style and size. The template accepts `{chain}`,
`{resi}`, `{resn}`, `{one}`, `{name}`, `{elem}`, `{b}` and `{q}`.

### Colour

AlphaFold pLDDT, B-factor, or a CSV of per-residue values, with a colormap
picker.

### Clean Up

*Clear All Gala Objects* removes everything Gala added — interactions,
measurements, labels, lights and compositor nodes — and leaves your molecule
and your own objects alone.

## How Gala organises the scene

Everything created goes into a `Gala` collection with one child per category:

```
Gala
├── Gala Interactions
├── Gala Measurements
├── Gala Labels
└── Gala Lighting
```

So a whole category can be hidden with one checkbox, excluded from a view
layer, or deleted as a unit. Objects also carry a `gala_type` custom property,
so they stay identifiable after you have moved them.

## Settings persistence

Panel settings live in a `PropertyGroup` on the scene (`bpy.context.scene.gala`),
so they survive save and reload, and two scenes in one file can be configured
differently.

```python
import bpy
props = bpy.context.scene.gala
props.preset = "print"
props.selection_a = "resn STI"
bpy.ops.gala.find_interactions()
```
