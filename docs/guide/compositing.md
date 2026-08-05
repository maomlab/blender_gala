# Compositing and passes

The point of setting up cryptomatte *before* rendering is that it lets you
change your mind afterwards — brighten just the ligand, knock back one chain,
pull a matte for a figure inset — without re-rendering a 40-minute image.

## Passes

```python
gala.enable_passes(
    cryptomatte=True,   # object, material and asset
    depth=True,         # Z
    normal=True,        # also improves denoising
    mist=False,
    cryptomatte_levels=6,
)
```

Six cryptomatte levels handles the semi-transparent edges of a molecular
surface; fewer produces fringing there.

## The compositor chain

```python
gala.setup_compositor(
    denoise=True,
    cryptomatte=True,
    dof=False,
    depth_cue_range=None,
    exposure=0.0,
    contrast=0.0,
    file_output="passes/",
)
```

Every node Gala adds is named with a `GALA` prefix and removed before
rebuilding, so calling this repeatedly rewires rather than accumulating
duplicates. Nodes you added yourself are left alone.

Blender 5 moved the scene compositor into a reusable node group and replaced
several `CompositorNode*` types with unified `ShaderNode*` ones. Gala targets
that generation only — the minimum supported Blender is 5.1 — and CI runs the
suite on the oldest supported release and the current one, so a later rename
shows up as a failure rather than a silently skipped step.

### Cryptomatte nodes are deliberately unconnected

`setup_compositor` adds a `CryptomatteV2` node per layer, ready to pick with,
but does not link one to the output. Connecting one would matte the beauty
pass, which defeats the point of shipping mattes.

## Writing the passes

```python
gala.setup_compositor(cryptomatte=True, file_output="passes/")
```

adds a File Output node writing a **32-bit multilayer EXR** with Image, Depth,
Normal and every cryptomatte layer. That is the format Nuke, Fusion, Krita and
Blender's own compositor read cryptomatte from; a PNG cannot carry it.

Alternatively, make the render output itself a multilayer EXR, which captures
every enabled pass without needing a File Output node:

```python
gala.scene.set_exr_output("render/figure.exr")
gala.render()
```

## Depth of field

Two ways, with different trade-offs.

**Physical camera DOF** traces a real aperture, so out-of-focus highlights
behave correctly and there is no edge bleeding. It costs samples:

```python
gala.depth_of_field(mol, fstop=2.8)
```

Focusing on an object rather than a distance means the focus follows it through
an animation.

!!! note "F-stops at molecular scale"

    One ångström is 0.01 Blender units, so the depth of field is very shallow.
    Values around f/2.8–8 are a starting point, not an answer — check the
    result rather than trusting the number.

**Compositor defocus** uses the Z pass, is far cheaper, and can be adjusted
after the render:

```python
gala.setup_compositor(dof=True, dof_fstop=4.0)
```

## Depth cueing

Fading the image towards the background with depth is the classic way of
keeping a crowded binding site readable — PyMOL's `depth_cue`:

```python
gala.depth_cue(near=0.0, far=60.0)     # angstrom from the camera
```

Geometry at `near` is untouched; geometry at `far` fades fully into the
background.

## Cleaning up

```python
gala.scene.clear_compositor()
```
