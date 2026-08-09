# Interactions

## Finding them

Interactions are always found **between two selections**, which matches how one
thinks about them: *what does this ligand touch?*

```python
contacts = gala.find_interactions(
    mol,
    "ligand",                       # side A
    "protein",                      # side B
    kinds=["hbond", "polar", "salt_bridge", "hydrophobic", "pi_stacking"],
)

for contact in contacts:
    print(contact)
# hbond: A/ASN23/ND2 - B/BTN1/O11 (2.91 A, 158 deg)
# pi_stacking: A/TRP79/CD2 - B/BTN1/C4 (3.74 A, 12 deg)
```

`kinds="all"` runs every detector. Results are sorted by kind, then distance.

For internal contacts, pass the same selection twice — `exclude_same_residue`
is on by default so a residue does not report contacts with itself:

```python
gala.find_interactions(mol, "chain A", "chain A", kinds="hbond")
```

Either side can be a [stored selection](selections.md#named-selections), which
is the quickest route from a pick in the viewport to what it touches: box-select
some atoms, name them in the **Gala ▸ Stored Selections** panel, then type that
name into Selection A.

```python
gala.create_alias(mol, "pocket")
gala.find_interactions(mol, "pocket", "not pocket", kinds="all")
```

## What is detected

| Kind | Criterion (defaults) |
| --- | --- |
| `hbond` | D–H···A with H···A ≤ 2.5 Å, D···A ≤ 3.5 Å, angle ≥ 130° |
| `polar` | N/O/S/F pair at 2.2–3.5 Å, not covalently bonded |
| `salt_bridge` | Charged-group centroids ≤ 5.5 Å |
| `hydrophobic` | Apolar C···C at 2.8–4.0 Å |
| `pi_stacking` | Ring centroids ≤ 5.5 Å; parallel ≤ 30° with offset ≤ 2 Å, or T-shaped ≥ 60° |
| `cation_pi` | Cation to ring centroid ≤ 6 Å, lateral offset ≤ 2 Å |
| `halogen` | X···A ≤ 4 Å with C–X···A between 140° and 180° |
| `metal` | Metal to N/O/S ≤ 3 Å |

The criteria follow [PLIP](https://github.com/pharmai/plip), so results are
comparable with what that reports.

### Hydrogen bonds without hydrogens

Crystal structures usually have no hydrogens, and a hydrogen-bond calculation
that needs them would return nothing. Asking for `hbond` on such a structure
therefore falls back to `polar_contacts`, the heavy-atom criterion behind
PyMOL's `polar_contacts`:

```python
gala.find_interactions(mol, kinds=["hbond"])
# On a structure with hydrogens: kind == "hbond", with D-H...A angles.
# On one without:                kind == "polar", distance only.
```

The fallback over-reports compared with a true hydrogen-bond calculation —
geometry alone cannot tell a donor from an acceptor — which is exactly the
trade-off crystallographers already accept.

### Ligands

A residue-name table cannot know that a novel inhibitor has a thiazole ring or
a carboxylate, and the ligand is usually what a figure is *about*. So anything
outside the standard-residue tables is perceived from connectivity:

- **Rings** — 5- and 6-membered cycles of C/N/O/S, planar to within 0.25 Å.
- **Negative groups** — carboxylates (C with two terminal O), phosphates and
  sulfates (P or S with three or more O).
- **Positive groups** — guanidinium and amidinium (C bonded to two or three N),
  quaternary nitrogen.

```python
from blender_gala.interactions import perception

perception.aromatic_rings(structure)          # tables for polymers, geometry for ligands
perception.charged_groups(structure, positive=True)
perception.bond_graph(structure)              # distance-based, metals excluded
```

!!! warning "Partial charges are not formal charges"

    Molecular Nodes stores force-field partial charges in the `charge`
    annotation, where a backbone carbonyl carbon reads about +0.6. Treating
    those as formal charges makes a salt-bridge detector fire on every residue
    in the protein. Gala ignores them and derives charge from connectivity.

## Adjusting the criteria

```python
from blender_gala import InteractionCriteria

loose = InteractionCriteria(polar_max=3.8, hbond_angle_min=120.0)
gala.find_interactions(mol, "ligand", "protein", criteria=loose)
```

Every cutoff in the table above is a field of `InteractionCriteria`.

## Custom contacts

When none of the named detectors fits, ask for atom pairs directly:

```python
contacts = gala.atom_contacts(
    mol,
    "resn ZN",
    "protein and (elem N or elem O or elem S)",
    cutoff=2.6,
)
```

## Drawing them

```python
gala.draw_interactions(contacts, target=mol, label=True)
```

Each interaction becomes a **curve object**, not a viewport overlay. It
receives light, casts shadows, occludes correctly against the molecule, and
appears in cryptomatte — a viewport overlay can do none of those.

Default styling follows the conventions people already read:

| Kind | Colour | Style |
| --- | --- | --- |
| `hbond` | Yellow | Dashed |
| `polar` | Cyan | Dashed |
| `salt_bridge` | Orange | Dashed, thicker |
| `hydrophobic` | Grey | Finely dashed |
| `pi_stacking` | Green | Dashed |
| `cation_pi` | Amber | Dashed |
| `halogen` | Purple | Dashed |
| `metal` | Blue-grey | Solid |

Restyle per kind:

```python
from blender_gala import InteractionStyle

gala.draw_interactions(contacts, target=mol, styles={
    "hbond": InteractionStyle(
        colour=(1.0, 0.2, 0.4),
        radius=0.15,        # angstrom
        dash_length=0.5,
        gap_length=0.3,
    ),
})
```

All distances are in ångström and converted at the Blender boundary, so a
0.15 Å dash radius is 0.15 Å regardless of the world scale.

Labels take a template with `distance`, `angle`, `kind` and `label`:

```python
gala.draw_interactions(
    contacts, target=mol, label=True, label_template="{distance:.2f} A"
)
```

## Clearing

```python
gala.clear_interactions()          # everything
gala.clear_interactions("hbond")   # one kind
```

## PLIP as a backend

If PLIP is installed in an interpreter you can reach, its output can be fed
straight into the drawing layer, so the figure matches the analysis in the
paper exactly:

```python
from blender_gala.interactions import plip

if plip.available():
    contacts = plip.from_pdb("complex.pdb", target=mol, ligand="STI:A:1")
    gala.draw_interactions(contacts, target=mol)
```

Passing `target` lets Gala map PLIP's atom serial numbers onto the loaded
structure, so the lines land on the right atoms even if PLIP protonated or
reordered the file.
