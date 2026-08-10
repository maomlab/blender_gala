# API reference

Everything on this page is importable from the top-level package. Installed
as an extension, that package is `bl_ext.<repository>.blender_gala`, the
repository being wherever you installed it from — see
[Getting started](../guide/getting-started.md):

```python
from bl_ext.user_default import blender_gala as gala
```

## Scene setup

| Function | Purpose |
| --- | --- |
| [`publication_setup`][blender_gala.scene.setup.publication_setup] | Configure the whole scene in one call |
| [`setup_render`][blender_gala.scene.render.setup_render] | Engine, sampling, GPU, transparency |
| [`three_point_lighting`][blender_gala.scene.lighting.three_point_lighting] | Studio key, fill and rim |
| [`hdri_lighting`][blender_gala.scene.lighting.hdri_lighting] | Environment lighting |
| [`assign_materials`][blender_gala.scene.materials.assign_materials] | Chemistry-aware materials |
| [`set_origin_to_geometry`][blender_gala.scene.origin.set_origin_to_geometry] | Origin onto the molecule |
| [`frame_target`][blender_gala.scene.camera.frame_target] | Create and aim a camera |
| [`orbit`][blender_gala.scene.camera.orbit] | Turntable animation |
| [`enable_passes`][blender_gala.scene.compositing.enable_passes] | Cryptomatte, Z, normal |
| [`setup_compositor`][blender_gala.scene.compositing.setup_compositor] | Build the compositing chain |
| [`depth_of_field`][blender_gala.scene.compositing.depth_of_field] | Camera depth of field |
| [`depth_cue`][blender_gala.scene.compositing.depth_cue] | Fade with depth |

## Interactions

| Function | Purpose |
| --- | --- |
| [`find_interactions`][blender_gala.interactions.detect.find_interactions] | Run several detectors |
| [`hydrogen_bonds`][blender_gala.interactions.detect.hydrogen_bonds] | D–H···A with explicit hydrogens |
| [`polar_contacts`][blender_gala.interactions.detect.polar_contacts] | Heavy-atom polar contacts |
| [`salt_bridges`][blender_gala.interactions.detect.salt_bridges] | Charged-group pairs |
| [`hydrophobic_contacts`][blender_gala.interactions.detect.hydrophobic_contacts] | Apolar carbon contacts |
| [`pi_stacking`][blender_gala.interactions.detect.pi_stacking] | Aromatic stacking |
| [`cation_pi`][blender_gala.interactions.detect.cation_pi] | Cation–π |
| [`halogen_bonds`][blender_gala.interactions.detect.halogen_bonds] | Halogen bonds |
| [`metal_coordination`][blender_gala.interactions.detect.metal_coordination] | Metal coordination |
| [`atom_contacts`][blender_gala.interactions.detect.atom_contacts] | Any pair within a cutoff |
| [`draw_interactions`][blender_gala.interactions.draw.draw_interactions] | Draw them as curves |

## Measurement and annotation

| Function | Purpose |
| --- | --- |
| [`distance`][blender_gala.measure.measurements.distance] | Between two atoms |
| [`angle`][blender_gala.measure.measurements.angle] | A–B–C |
| [`dihedral`][blender_gala.measure.measurements.dihedral] | A–B–C–D, signed |
| [`label`][blender_gala.annotate.labels.label] | In-scene 3D text |
| [`label_hud`][blender_gala.annotate.labels.label_hud] | 2D compositing overlay |

## Colour

| Function | Purpose |
| --- | --- |
| [`color_by_plddt`][blender_gala.color.coloring.color_by_plddt] | AlphaFold confidence bands |
| [`color_by_attribute`][blender_gala.color.coloring.color_by_attribute] | Any per-atom or per-residue value |
| [`color_by_bfactor`][blender_gala.color.coloring.color_by_bfactor] | Crystallographic B-factor |
| [`color_by_selection`][blender_gala.color.coloring.color_by_selection] | Categorical |
| [`color_from_csv`][blender_gala.color.coloring.color_from_csv] | Per-residue values from a file |

## Selection

| Function | Purpose |
| --- | --- |
| [`select`][blender_gala.core.selection.select] | Boolean mask for a selection |
| [`select_indices`][blender_gala.core.selection.select_indices] | Matched atom indices |
| [`compile_selection`][blender_gala.core.selection.compile_selection] | Parse once, reuse |
