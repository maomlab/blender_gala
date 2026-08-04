# Colouring by data

Colours are written to the mesh `Color` attribute — the attribute Molecular
Nodes' styles already read — so a recoloured molecule renders correctly in
every style with no node-graph surgery.

## AlphaFold confidence

```python
gala.color_by_plddt(mol)
```

pLDDT is read from the B-factor column, where AlphaFold puts it, and mapped to
the official AlphaFold DB bands:

| pLDDT | Colour | Meaning |
| --- | --- | --- |
| ≥ 90 | `#0053D6` | Very high |
| 70–90 | `#65CBF3` | Confident |
| 50–70 | `#FFDB13` | Low |
| < 50 | `#FF7D45` | Very low |

The 0–1 and 0–100 conventions are both handled: ColabFold and several
downstream tools write 0–1, and the scale is detected from the data.

```python
gala.color_by_plddt(mol, mode="banded")      # matches the AFDB viewer exactly
gala.color_by_plddt(mol, mode="continuous")  # smooth ramp; reads better on a surface
gala.color_by_plddt(mol, selection="chain A")
```

`mode="continuous"` looks better on a molecular surface but is no longer
directly comparable with the AlphaFold database, so `banded` is the default.

A legend for the figure caption:

```python
for label, rgb in gala.plddt_legend():
    print(label, rgb)
# Very high (pLDDT > 90) (0.0, 0.087, 0.672)
```

Hiding the parts you should not trust is one line:

```python
gala.color_by_plddt(mol)
mol.add_style("cartoon", selection="b > 70")
```

## Any per-residue value

The general case: conservation scores, per-residue energies, deep mutational
scanning summaries — anything computed elsewhere.

```python
import numpy as np

gala.color_by_attribute(mol, np.load("conservation.npy"), cmap="viridis")
gala.color_by_attribute(mol, {45: 0.9, 46: 0.2, 47: 0.5}, cmap="coolwarm")
gala.color_by_attribute(mol, {("A", 45): 0.9, ("B", 45): 0.1})
gala.color_by_attribute(mol, lambda i: some_function(i))
```

Accepted forms: a per-atom array, a `{res_id: value}` mapping, a
`{(chain, res_id): value}` mapping, or a callable taking an atom index.

!!! tip "Use the chain-qualified form for multi-chain structures"

    Residue numbering restarts per chain, so `{45: 0.9}` colours residue 45 in
    *every* chain — including a ligand or an ion that happens to be residue 45.
    `{("A", 45): 0.9}` does not.

### Comparable colours across structures

By default the range is the data's own min and max, so two structures get two
different scales and their colours cannot be compared. Fix the range when that
matters:

```python
for model in models:
    gala.color_by_attribute(model, values[model], vmin=0.0, vmax=1.0)
```

Residues with no value get `missing`, grey by default, which keeps "no data"
visually distinct from "a low value".

## B-factors

```python
gala.color_by_bfactor(mol, cmap="coolwarm")
gala.color_by_bfactor(mol, "chain A", cmap="plasma", vmin=10, vmax=60)
```

## Categorically

```python
gala.color_by_selection(mol, {
    "chain A": "#4477AA",
    "chain B": "#CC6677",
    "ligand": "#DDCC77",
    "byres (protein within 4 of ligand)": "#EE8866",
})
```

Later entries win where selections overlap, so order from general to specific:
background chain first, highlighted site last.

## From a CSV

```python
gala.color_from_csv(
    mol,
    "conservation.csv",
    value_column="score",
    res_id_column="position",
    chain_column="chain",
    cmap="viridis",
)
```

## Colormaps

`viridis`, `plasma`, `magma`, `inferno`, `cividis`, `turbo`, `coolwarm`, `bwr`,
`rdylbu`, `spectral`, `rainbow`, `grey`, `alphafold`. `reverse=True` flips any
of them.

```python
gala.color.list_colormaps()
```

They are implemented natively — matplotlib is not available inside Blender and
is far too heavy to vendor for a lookup table.

!!! note "sRGB in, linear out"

    Blender colour attributes are linear; hex colours and colormap definitions
    are sRGB. Gala converts. Skipping that conversion is why hand-written
    colouring scripts so often come out washed out — mid-grey `#808080` is
    0.216 linear, not 0.5.

## Reading and writing directly

```python
current = gala.read_colors(mol)         # (n_atoms, 4) linear RGBA
current[:, 0] *= 1.2
gala.write_colors(mol, current)

mask = gala.select(mol, "chain A")
gala.write_colors(mol, colours, mask=mask)
```

Every colouring function also accepts `write=False`, which computes the colours
and returns them without touching the mesh — useful for building a legend, or
for testing.
