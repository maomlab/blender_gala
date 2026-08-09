# Selection language

Gala speaks PyMOL selection syntax. Every function that takes a selection —
interaction detection, measurement, labelling, colouring, origin — takes the
same strings.

```python
gala.select(mol, "byres (protein within 4 of ligand)")
```

Selections are parsed once and evaluated against a biotite `AtomArray`, so they
work on any structure and never touch Blender state — the one exception being a
[stored selection](#using-a-name-in-a-selection), whose name lives on the
molecule.

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
"bymol (resi 45)"                      # everything bonded to residue 45
"first chain A"                        # the first matched atom
"last protein"                         # the last matched atom
```

`byres` is what you almost always want when selecting a binding site: without
it you get whichever atoms happened to fall inside the sphere, and a rendered
side chain sliced in half.

`bymol` (also `byfrag`) follows the bond graph rather than the residue or
chain annotations, which is how you grab a whole ligand or one strand of a
complex. It needs a structure that arrived with bonds; without them it falls
back to the chain.

## Picking atoms in the viewport

Selections do not have to be typed. Because Molecular Nodes builds a molecule
so that vertex *i* is atom *i*, the atoms you select in Blender's Edit Mode
**are** a selection — box select, circle select and lasso all work on atoms
already. Gala adds the three things that were missing: a selection *level*, a
readout in selection syntax, and a way to name what you picked.

Tab into Edit Mode, select some atoms, then open the **Gala ▸ Selection**
panel — or reach the same expansions from **Select ▸ Expand to Residue** while
you are already in the Select menu.

```python
mask = gala.viewport_selection(mol)                  # what is picked, as a mask
mask = gala.expand_viewport_selection(mol, "residue")  # grow it, and apply it
text = gala.describe_viewport_selection(mol)         # 'chain A and resi 45-47'

gala.set_viewport_selection(mol, "byres (ligand expand 4)")  # the other way
```

### Levels

| Level | Grows a picked atom to |
| --- | --- |
| `atom` | itself — no change |
| `residue` | every atom of its residue |
| `chain` | every atom of its chain |
| `fragment` | everything bonded to it |
| `object` | the whole structure |

Levels compose, so expanding to `residue` and then to `chain` grows the
residues to their chains, the way PyMOL's selection modes behave.

### Reading a selection back

`describe_selection` turns a mask into a selection string, and verifies it: the
string is re-evaluated against the structure and only returned if it selects
exactly the same atoms. Structures where the chemical description would be
ambiguous — repeated residue numbers under insertion codes, blank chain
identifiers — fall back to the positional `index 3+7-10` form, which is always
exact.

```python
>>> gala.describe_selection(mol, gala.select(mol, "name CA"))
'chain A and resi 1-7 and name CA'
```

## Named selections

A named selection — PyMOL's `sele` — is stored as a boolean attribute on the
mesh, which is the same thing three other tools already read:

```python
gala.create_alias(mol, "pocket")                 # names what is picked
gala.create_alias(mol, "core", "b > 70")         # or names a selection string
gala.list_aliases(mol)                           # {'pocket': mask, 'core': mask}
gala.select_alias(mol, "pocket")                 # select it again later
gala.alias_combine(mol, "pocket", mode="union")  # add the current pick to it
gala.delete_alias(mol, "core")
```

### Using a name in a selection

A stored name is a word in the language, the same way it is in PyMOL. Anywhere
Gala takes a selection — an interaction side, a colour, a label, a measurement,
the panel's text fields — the name of a stored selection will do:

```python
gala.create_alias(mol, "pocket")                        # name what you picked

gala.find_interactions(mol, "pocket", "not pocket")     # what does it touch?
gala.select(mol, "byres (protein within 4 of pocket)")  # what lines it?
gala.color_by_selection(mol, "pocket", "orange")
gala.label(mol, "pocket and name CA")
```

The name has to reach the molecule to mean anything, since that is where it is
stored — `gala.select(mol, "pocket")` works, `gala.select(mol.array, "pocket")`
cannot. Case does not matter, as it does not anywhere else in the language.

Keywords are matched first, so a selection called `ligand` does not quietly
redefine the macro of that name. Prefix it with `%` to reach it anyway:

```python
gala.select(mol, "%ligand")     # the stored selection
gala.select(mol, "ligand")      # the macro: hetero, not water, not an ion
```

Any boolean attribute on the mesh answers to its name, not only the ones Gala
stored — so selections that arrived with a PyMOL session, and ones built by
hand in the node editor, are usable too. A name that matches nothing says what
the molecule does have:

```python
>>> gala.select(mol, "pockte")
SelectionSyntaxError: unknown selection keyword 'pockte'. Stored selections: pocket, core.
    pockte
    ^
```

Because the attribute name is what Molecular Nodes reads, a named selection can
be styled on its own — which is how a pocket gets sticks while the rest of the
protein stays a cartoon:

```python
mol.add_style("cartoon")
gala.style_alias(mol, "pocket", style="ball_and_stick")
```

That adds a style branch limited to the selection and leaves the existing
styles alone. The equivalent in Molecular Nodes' own API is
`mol.add_style("ball_and_stick", selection="pocket")`, and in the geometry node
editor it is a Named Attribute node wired into the style's `Selection` socket.

Named selections also survive the round trip to PyMOL:

```python
gala.save_session("figure.pse", selections=["pocket"])
```

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
>>> gala.select(mol, "chain A and resi abc")
SelectionSyntaxError: expected an integer or range, got 'abc'
    chain A and resi abc
                     ^
```

A word that is neither a keyword nor a stored selection is reported when the
selection meets the molecule, since only the molecule knows which names exist:

```python
>>> gala.select(mol, "chain A and pockte")
SelectionSyntaxError: unknown selection keyword 'pockte'. Stored selections: pocket.
    chain A and pockte
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
