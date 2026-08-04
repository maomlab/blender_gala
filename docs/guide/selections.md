# Selection language

Gala speaks PyMOL selection syntax. Every function that takes a selection —
interaction detection, measurement, labelling, colouring, origin — takes the
same strings.

```python
gala.select(mol, "byres (protein within 4 of ligand)")
```

Selections are parsed once and evaluated against a biotite `AtomArray`, so they
work on any structure and never touch Blender state.

## Properties

| Keyword | Aliases | Matches | Example |
| --- | --- | --- | --- |
| `chain` | `c`, `segi` | Chain identifier | `chain A` |
| `resi` | `resid`, `residue`, `i` | Residue number | `resi 45-60` |
| `resn` | `resname`, `r` | Residue name | `resn HIS` |
| `name` | `n`, `atom` | Atom name | `name CA` |
| `elem` | `element`, `e` | Element symbol | `elem N` |
| `index` | `idx` | 1-based atom position | `index 1-100` |
| `id` | | PDB atom serial number | `id 250` |
| `b` | `bfactor` | B-factor or pLDDT | `b > 70` |
| `q` | `occupancy` | Occupancy | `q = 1.0` |
| `charge` | | Partial charge | `charge < -0.5` |
| `ss` | | Secondary structure code | `ss 1` |

Names are matched case-insensitively, so `chain a` and `chain A` agree.

### Value forms

```python
"chain A+B+C"        # a list; commas work too
"resi 10-20"         # an inclusive range
"resi 10-"           # from 10 upwards
"resi :20"           # up to 20
"resi -5"            # residue number minus five, as in PyMOL
"name CA+CB+CG"      # several atom names
"name C*"            # wildcard: CA, CB, CG, C ...
"name C?"            # single-character wildcard
```

### Numeric comparison

`>`, `<`, `>=`, `<=`, `=`, `!=` work on numeric properties:

```python
"b > 90"                  # AlphaFold: very high confidence
"b < 50"                  # very low confidence
"q != 1.0"                # partial occupancy
"resn = HIS"              # string equality
"resn != HOH"             # string inequality
```

## Macros

| Macro | Aliases | Matches |
| --- | --- | --- |
| `all` / `none` | | Everything / nothing |
| `protein` | `peptide` | Amino acid residues, including modified ones |
| `nucleic` | `dna`, `rna` | Nucleotides |
| `polymer` | | Protein or nucleic |
| `backbone` | `bb` | Protein N/CA/C/O, nucleic phosphate backbone |
| `sidechain` | `sc` | Polymer, not backbone, not hydrogen |
| `water` | `solvent` | HOH, WAT, TIP3, SOL … |
| `hetatm` | `hetero` | HETATM records |
| `ligand` | `organic` | Hetero, not water, not an ion |
| `ions` | `ion` | Monoatomic ions |
| `metals` | | Metal elements |
| `hydro` | `hydrogen` | Hydrogen and deuterium |
| `donors` | | N, O, S — potential hydrogen-bond donors |
| `acceptors` | | N, O, S, F |
| `polar` | | N, O, S, P, F |
| `carbon` | | Carbon |
| `aromatic` | | Aromatic ring atoms of standard residues |
| `ca` | `alpha` | Alpha carbons |

## Operators

### Boolean

```python
"chain A and name CA"
"protein or nucleic"
"not water"
"chain A and not (backbone or hydro)"
```

`&`, `|` and `!` are accepted as `and`, `or` and `not`. `and` binds more
tightly than `or`, so `a or b and c` means `a or (b and c)` — parenthesise when
in doubt.

### Spatial

```python
"protein within 4 of ligand"    # protein atoms no more than 4 A from a ligand atom
"ligand around 5"               # everything within 5 A of the ligand, ligand excluded
"chain A expand 3"              # chain A plus everything within 3 A of it
"protein beyond 10 of ligand"   # protein atoms further than 10 A away
```

Distances are in **ångström**. `within` includes the source selection;
`around` excludes it — the same distinction PyMOL draws.

### Expansion

```python
"byres (protein within 4 of ligand)"   # whole residues, not clipped side chains
"bychain (resi 45)"                    # every chain containing residue 45
"first chain A"                        # the first matched atom
"last protein"                         # the last matched atom
```

`byres` is what you almost always want when selecting a binding site: without
it you get whichever atoms happened to fall inside the sphere, and a rendered
side chain sliced in half.

## Recipes

```python
# The binding site around a ligand
"byres (protein within 4.5 of ligand)"

# A catalytic triad
"chain A and resi 57+102+195"

# Confident parts of an AlphaFold model
"b > 70"

# Everything except waters and hydrogens
"not (water or hydro)"

# A pocket lining, excluding backbone
"byres (protein within 5 of resn STI) and sidechain"

# Polar atoms of one chain facing another
"chain A and polar and (chain B around 4)"

# Just the ring of a specific tyrosine
"resi 45 and resn TYR and name CG+CD1+CD2+CE1+CE2+CZ"
```

## Errors

A bad selection tells you where it went wrong:

```python
>>> gala.select(mol, "chain A and bogus B")
SelectionSyntaxError: unknown selection keyword 'bogus'
    chain A and bogus B
                ^
```

## Other forms

Anywhere a selection string is accepted, so is a boolean mask or an array of
atom indices:

```python
import numpy as np

mask = gala.select(mol, "protein")
gala.label(mol, mask)                 # a mask
gala.label(mol, np.array([0, 1, 2]))  # atom indices
```

Precompiling helps when the same selection is applied repeatedly:

```python
site = gala.compile_selection("byres (protein within 4 of ligand)")
for frame in range(100):
    ...
    mask = site.evaluate(structure.array)
```
