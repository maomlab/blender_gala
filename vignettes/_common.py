"""Shared setup for the vignettes.

Every vignette is a standalone script executed by CI (SPECIFICATION D-22), so
this module keeps them short and gives them all the same behaviour:

* find the repository regardless of where Blender was launched from;
* fetch a real structure from the PDB when there is network, and fall back to
  the committed synthetic fixture when there is not, so the scripts always run;
* render at a quality set by ``GALA_VIGNETTE_QUALITY`` — ``draft`` by default
  so CI stays fast, ``figure`` when generating the documentation images.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(REPO_ROOT, "docs", "images")
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: Set GALA_VIGNETTE_QUALITY=figure for publication-resolution output.
QUALITY = os.environ.get("GALA_VIGNETTE_QUALITY", "draft")


def heading(text: str) -> None:
    """Print a section header, so the CI log reads as a narrative."""
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def clear_scene() -> None:
    """Empty the scene Blender started with.

    Blender's startup file contains a 2-unit Cube — 200 A at Molecular Nodes'
    world scale — which dwarfs any molecule and fills the frame. A vignette
    owns its scene, so it starts by clearing it.
    """
    import bpy

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def setup(clear: bool = True) -> tuple:
    """Register Molecular Nodes and Blender Gala, and return both modules.

    Parameters
    ----------
    clear : bool, optional
        Empty the startup scene first.

    Returns
    -------
    tuple
        ``(molecularnodes, blender_gala)``.
    """
    from blender_gala.core import mn as mn_bridge

    module = mn_bridge.require_mn()
    module.register()

    import blender_gala as gala

    gala.register()
    if clear:
        clear_scene()
    return module, gala


def load_structure(pdb_code: str, fallback: str = "site.pdb"):
    """Fetch ``pdb_code`` from the PDB, falling back to a bundled fixture.

    Parameters
    ----------
    pdb_code : str
        Four-character PDB identifier.
    fallback : str, optional
        File in ``tests/data`` to use when the fetch fails.

    Returns
    -------
    molecularnodes.Molecule
    """
    from blender_gala.core import mn as mn_bridge

    module = mn_bridge.require_mn()

    try:
        molecule = module.Molecule.fetch(pdb_code)
    except Exception as exc:
        path = os.path.join(DATA_DIR, fallback)
        print(
            f"  could not fetch {pdb_code} ({exc.__class__.__name__}); using {fallback}"
        )
        molecule = module.Molecule.load(path)
    else:
        from blender_gala.core.entity import AtomStructure

        print(f"  loaded {pdb_code}: {AtomStructure.from_any(molecule).n_atoms} atoms")
    return molecule


def render(gala, name: str) -> str:
    """Render the current scene into ``docs/images``.

    Parameters
    ----------
    gala : module
        The imported ``blender_gala`` module.
    name : str
        File name stem.

    Returns
    -------
    str
        The path written.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, f"{name}.png")
    gala.render(path)
    print(f"  wrote {path}")
    return path
