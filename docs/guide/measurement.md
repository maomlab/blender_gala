# Measurement and labels

## Distances, angles, dihedrals

```python
d = gala.distance(mol, "resi 10 and name CA", "resi 20 and name CA", draw=True)
a = gala.angle(mol, "resi 1 and name CB", "resi 1 and name OG", "resi 2 and name OD1")
t = gala.dihedral(mol, "resi 5 and name C", "resi 6 and name N",
                       "resi 6 and name CA", "resi 6 and name C")

print(d)          # distance: A/ALA10/CA - A/GLY20/CA = 12.43 A
print(float(d))   # 12.43
print(d.text)     # "12.43 A"
```

Distances are in ångström, angles and dihedrals in degrees. Dihedrals use the
IUPAC sign convention, so a phi or psi value matches what a Ramachandran plot
would show.

`measure()` dispatches on how many selections it is given — two is a distance,
three an angle, four a dihedral — which is the scripting equivalent of clicking
atoms in PyMOL's wizard:

```python
gala.measure_atoms(mol, sel_a, sel_b)                    # distance
gala.measure_atoms(mol, sel_a, sel_b, sel_c)             # angle
gala.measure_atoms(mol, sel_a, sel_b, sel_c, sel_d)      # dihedral
```

## Ambiguity is an error

Each selection must resolve to exactly one atom, exactly as clicking an atom in
PyMOL does:

```python
>>> gala.distance(mol, "name CA", "name CB")
AmbiguousSelectionError: selection 'name CA' matched 214 atoms but exactly one
was required. Narrow the selection, or pass reduce='first'/'closest'/'centroid'.
```

Silently measuring to *some* CA would produce a number that looks right and
is not. When the ambiguity is intentional, resolve it explicitly:

| `reduce` | Behaviour |
| --- | --- |
| `"single"` | Default; raises if more than one atom matches |
| `"centroid"` | Mean position of the matched atoms |
| `"first"` / `"last"` | The first or last matched atom |
| `"closest"` | Nearest to a supplied reference point |

Policies can differ per selection — measuring from a ring centre to one atom:

```python
gala.distance(
    mol,
    "resn TYR and resi 45 and name CG+CD1+CD2+CE1+CE2+CZ",
    "resn LIG and name N1",
    reduce=["centroid", "single"],
)
```

## Drawing

`draw=True` creates the geometry:

- **Distance** — a dashed line with the value at its midpoint.
- **Angle** — the two rays plus an arc, labelled along the bisector.
- **Dihedral** — the three bonds plus the arc a Newman projection would show,
  which is the only representation that makes the sign legible.

```python
gala.distance(
    mol, sel_a, sel_b,
    draw=True,
    colour=(0.2, 0.9, 0.4),
    radius=0.12,          # angstrom
    dash_length=0.4,
    gap_length=0.25,
    label_template="{value:.2f} A",
    label_size=1.5,               # angstrom; None sizes it to the frame
)
gala.clear_measurements()          # or clear_measurements("angle")
```

### Labels that match the zoom

`label_size` is in ångström, which is the right unit when the camera is on the
whole molecule and the wrong one when it is two ångström from a hydrogen bond:
the same value is legible in one and covers the frame in the other.

`label_size=None` sizes the text to the **frame** instead — a fixed share of
the visible height at the label's own depth — so it reads the same whatever
the camera is doing. It is what
[`load_session`](pymol.md) uses, because a PyMOL session's view is as often a
close-up of a contact as it is a view of a whole complex.

## From the viewport

The **Measure** panel works the way the PyMOL wizard does: select 2–4 atoms in
Edit Mode on the molecule, then press *Measure*. Gala picks distance, angle or
dihedral from how many you selected. Typing selection strings separated by `;`
into the panel field does the same thing without leaving Object Mode.

## Labels

### In-scene 3D labels

```python
gala.label(mol, "byres (protein within 4 of ligand)")
```

Creates a real text object per residue. It occludes correctly behind the
molecule, appears in cryptomatte, and can be nudged by hand when one lands
somewhere awkward.

```python
gala.label(
    mol,
    "resn LIG",
    template="{resn}{resi}",   # {chain} {resi} {resn} {one} {name} {elem} {b} {q}
    level="residue",           # "residue" | "atom" | "selection"
    anchor="ca",               # "centroid" | "first" | "ca"
    style="card",              # a translucent backing plane
    size=2.0,                  # angstrom
    offset=2.5,                # lifted above the anchor
    billboard=True,            # faces the camera, square to the frame
)
```

Shorthands:

```python
gala.label_residues(mol, "chain A and resi 45-50", template="{resn}{resi}")
gala.label_atoms(mol, "resn ZN", template="{elem}")
```

`style="card"` puts the text on a translucent plane, which is what keeps a
label legible over a busy molecular surface. The card is parented to the text
and billboards with it.

Labels are moved towards the camera until nothing is in front of them, and
scaled down by as much as they moved so that the shift does not enlarge them.
A label anchored on a residue inside a protein is otherwise behind that
protein from almost every angle, and no fixed offset fixes it: the direction
that clears the ribbon depends on where the camera is. Which is also why
labelling comes *after* framing — pass `avoid_occlusion=False` for an orbit,
where no one position is in front for every frame.

Distances and measured values get a translucent pill behind them for the same
reason, in a cooler tint and a different outline from the residue cards, so the
two kinds of label are distinguishable at a glance. Pass `label_card=False` to
`draw_interactions` or `draw_measurement` for bare text.

Billboarding copies the camera's rotation rather than tracking it, so every
label is level with the frame and parallel to every other. Tracking would aim
each label at the camera and then roll it towards world +Y, which tips labels
in different places by different amounts.

### 2D overlay labels

```python
gala.label_hud(mol, "Figure 1: the biotin site", location=(0.05, 0.95), size=28)
```

Drawn in screen space through Molecular Nodes' annotation system and composited
over the render, so it is never occluded and never changes size with the
camera. Right for titles and captions; use `label()` for anything that must sit
*in* the scene.

This is the feature that sets the floor in
[Requirements](../index.md#requirements): it goes through the annotation
manager added in Molecular Nodes 4.5, so on anything older it raises and says
which version it wants.

```python
gala.clear_labels()
```

## Why real geometry

Molecular Nodes' annotation framework draws with the GPU module and composites
a 2D image. That is ideal for HUD text but such an overlay cannot receive
light, cast shadows, appear in cryptomatte, or be depth-sorted against the
molecule. Since a hydrogen bond in a figure should behave like any other object
in the scene, Gala draws it as one — and offers the overlay path separately,
for the cases where being un-occludable is the point.
