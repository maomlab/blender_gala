# Electrostatics

Gala runs [APBS](https://www.poissonboltzmann.org) the way the [PyMOL APBS
plugin](https://pymolwiki.org/index.php/Apbsplugin) does — PDB2PQR assigns
charges and radii, APBS solves the Poisson–Boltzmann equation on a grid — and
paints the result onto a translucent molecular surface.

```python
import blender_gala as gala

surface = gala.electrostatic_surface(mol, ramp=5.0, alpha=0.6)
print(surface.summary())
```

That single call does four things: writes the structure out as PDB, runs
PDB2PQR and APBS on it, reads the potential back, and colours a Molecular
Nodes surface style by it — red where the potential is negative, white at
zero, blue where it is positive.

![Barnase and barstar, opened out, coloured by potential](../images/07_electrostatics.png)

## Getting APBS

APBS and PDB2PQR are separate programs and Gala does not bundle either; it
shells out to whatever is on `PATH`. Both install with pip:

```console
$ pip install apbs-binary pdb2pqr
```

If they are somewhere else, point `GALA_APBS` and `GALA_PDB2PQR` at them, or
pass `apbs_path=` and `pdb2pqr_path=`. In this repository, `make apbs` puts
both in `.venv` and `make vignettes` finds them there.

Without them you get one error that says exactly this, rather than a traceback
from a missing file.

## What the calculation is

Every setting that changes the answer is an argument, not a dialog box:

```python
run = gala.run_apbs(
    mol,
    forcefield="AMBER",      # PDB2PQR's charges and radii
    ph=7.4,                  # optional; assigns protonation states with PROPKA
    ionic_strength=0.15,     # mol/L of monovalent salt
    pdie=2.0, sdie=78.54,    # solute and solvent dielectrics
    temperature=298.15,      # K
    solver="lpbe",           # or "npbe" for the non-linear equation
)
print(run.net_charge, run.grid.shape, run.workdir)
```

The grid itself is sized by PDB2PQR, which knows how to fit a `mg-auto` box
around a molecule; Gala edits the rest of the input file it generates rather
than writing one from scratch, so what APBS actually ran is readable next to
its output in `run.workdir`. Nothing there is deleted — a run takes long
enough that throwing away the PQR and the logs is rarely what you want.

An already-computed map skips all of it:

```python
grid = gala.read_dx("mymap.dx")          # or .dx.gz
gala.electrostatic_surface(mol, grid=grid)
```

## Where the potential is read

The potential at an atom's centre is not the potential at the surface. Inside
the solute the field is dominated by that atom's own partial charge — hundreds
of kT/e, and every carbonyl would come out red regardless of what the molecule
as a whole is doing.

So Gala reads it where the surface is. Each atom gets a set of points on its
solvent-accessible sphere — radius *r*<sub>vdW</sub> + probe, the same
construction Shrake and Rupley use for SASA — the points covered by
neighbouring atoms are discarded, and the value is the mean over what is left.
An atom whose points are all covered is buried, has no surface to colour, and
gets `nan` rather than a number read from inside the protein.

```python
values = gala.potential_at_atoms(mol, grid, probe=1.4, points=32)
```

Molecular Nodes then does the rest: the surface style is set to take its
colour from the nearest atom, so each patch of surface carries the value that
belongs to it.

## The ramp

`ramp` is where the colour saturates, in kT/e, symmetric about zero because
the sign is the whole point. ±5 is the conventional choice and the one the
PyMOL plugin opens with; small, mostly-neutral proteins often want ±3, and a
nucleic acid or a strongly charged complex wants more. It is a display choice,
so quote it in the caption — `surface.summary()` prints it along with the
range actually present and how much of the surface saturates:

```
Electrostatic surface
  ramp     : -3 to +3 kT/e (red to blue)
  surface  : -14.27 to +14.24 kT/e, mean +0.67
  beyond   : 6.3% of atoms saturate the ramp
```

## Transparency

`alpha` sets the surface's opacity, and the material uses alpha blending
rather than transmission: it renders far faster and avoids the caustic
fireflies a glass-like surface produces. Anything inside — a cartoon, a
ligand, the partner it binds — shows through.

A translucent surface picks up specular highlights that can bleach the ramp;
`material_options={"roughness": 0.45, "specular": 0.25}` takes the shine off.

## What this is not

- **Not a substitute for reading the APBS documentation.** The defaults here
  are reasonable for a soluble protein at physiological salt. A membrane
  protein, a nucleic acid or anything with an unusual metal centre deserves an
  input file you have looked at.
- **Not per-vertex.** The potential is read per atom, in the way described
  above, and the surface interpolates between atoms. It is the same
  correspondence PyMOL's `ramp_new` and `set surface_color` produce.
- **Not a free energy.** A potential map says where the field is, not what
  binding will cost.
