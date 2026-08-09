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
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(REPO_ROOT, "docs", "images")
DATA_DIR = os.path.join(REPO_ROOT, "tests", "data")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: Set GALA_VIGNETTE_QUALITY=figure for publication-resolution output.
QUALITY = os.environ.get("GALA_VIGNETTE_QUALITY", "draft")

#: Where the vignettes save the scenes they built. Generated output, so it
#: lives with the rest of it under build/; set GALA_VIGNETTE_BLEND_DIR to move
#: it somewhere else.
BLEND_DIR = os.environ.get(
    "GALA_VIGNETTE_BLEND_DIR", os.path.join(REPO_ROOT, "build", "vignettes")
)

#: Downloaded Poly Haven textures. Generated, gitignored, and cached: they are
#: a few hundred kilobytes each and there is no reason to fetch one twice.
TEXTURE_DIR = os.environ.get(
    "GALA_TEXTURE_DIR", os.path.join(REPO_ROOT, "build", "textures")
)

#: Poly Haven's public catalogue. Everything on it is CC0, so a vignette may
#: fetch and use one without a licence to satisfy; the asset names are printed
#: anyway, because saying where a picture's materials came from costs nothing.
POLYHAVEN_API = "https://api.polyhaven.com"

#: Poly Haven is behind Cloudflare, which answers Python's default
#: ``Python-urllib/3.x`` with a 403. Any honest identifier gets through, and an
#: honest one is better manners than pretending to be a browser.
_USER_AGENT = "blender-gala-vignettes (+https://github.com/maomlab/blender_gala)"

#: A render this uniform has nothing in it. Fractions of the 0-1 pixel range.
_FLAT_TOLERANCE = 0.01


def _die_on_unhandled_exception() -> None:
    """Make an unhandled exception stop the run with a non-zero exit code.

    Blender prints the traceback from a ``--python`` script and then exits 0
    regardless, so a vignette that dies half way through is indistinguishable
    from one that finished — to CI, to ``make vignettes``, and to whoever is
    reading the log. Only an explicit exit code gets out of Blender, so that is
    what the hook does.
    """

    def hook(exc_type, value, tb):
        traceback.print_exception(exc_type, value, tb)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.excepthook = hook


_die_on_unhandled_exception()


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


def load_alphafold(accession: str, fallback: str = "plddt.pdb"):
    """Fetch a model from the AlphaFold database, falling back to a fixture.

    The AlphaFold database is keyed by UniProt accession rather than by PDB ID,
    and it is what puts pLDDT in the B-factor column — the whole point of the
    colouring this demonstrates. The synthetic fixture has the same B-factors
    but none of the shape, so it stands in only when there is no network.

    Parameters
    ----------
    accession : str
        UniProt accession, e.g. ``"P04637"``.
    fallback : str, optional
        File in ``tests/data`` to use when the fetch fails.

    Returns
    -------
    molecularnodes.Molecule
    """
    from blender_gala.core import mn as mn_bridge

    module = mn_bridge.require_mn()

    try:
        molecule = module.Molecule.fetch(accession, database="alphafold")
    except Exception as exc:
        path = os.path.join(DATA_DIR, fallback)
        print(
            f"  could not fetch AlphaFold {accession} "
            f"({exc.__class__.__name__}); using {fallback}"
        )
        molecule = module.Molecule.load(path)
    else:
        from blender_gala.core.entity import AtomStructure

        print(
            f"  loaded AlphaFold {accession}: "
            f"{AtomStructure.from_any(molecule).n_atoms} atoms"
        )
    return molecule


def load_texture(
    name: str,
    maps: tuple[str, ...] = ("Diffuse", "Rough", "nor_gl"),
    resolution: str = "1k",
    file_format: str = "jpg",
) -> dict[str, str]:
    """Fetch a Poly Haven texture, cached, and return the files it downloaded.

    The same bargain :func:`load_structure` makes: fetch it when there is
    network, and when there is not, say so and hand back nothing so the caller
    can carry on without it. A vignette that dies because a texture site is
    down is a vignette that fails CI for a reason that has nothing to do with
    the code being tested.

    Nothing is committed. The files land in ``build/textures`` alongside the
    rest of the generated output, and each is downloaded once.

    Parameters
    ----------
    name : str
        Poly Haven asset slug, e.g. ``"marble_01"``.
    maps : tuple of str, optional
        Which maps to fetch, named as the API names them: ``"Diffuse"``,
        ``"Rough"``, ``"nor_gl"``, ``"AO"``, ``"Displacement"``.
    resolution : str, optional
        ``"1k"`` through ``"8k"``. 1k is plenty on a molecule a few hundred
        pixels across, and is a tenth the download of 4k.
    file_format : str, optional
        ``"jpg"``, ``"png"`` or ``"exr"``.

    Returns
    -------
    dict[str, str]
        Map name to file path, or an empty dict if the texture could not be
        fetched.
    """
    import json
    import urllib.request

    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    wanted = {
        entry: os.path.join(TEXTURE_DIR, f"{name}_{entry}_{resolution}.{file_format}")
        for entry in maps
    }
    if all(os.path.exists(path) for path in wanted.values()):
        return wanted

    try:
        listing = json.loads(fetch(f"{POLYHAVEN_API}/files/{name}"))

        # Not every asset offers every map — a texture photographed in several
        # colourways has `col_1` and `col_2` rather than a single `Diffuse` —
        # so take what is there and say what was not, rather than failing the
        # whole texture over one missing file.
        missing = [entry for entry in wanted if entry not in listing]
        for entry in missing:
            del wanted[entry]
        if missing:
            print(f"  {name} has no {', '.join(missing)}; using {sorted(wanted)}")

        os.makedirs(TEXTURE_DIR, exist_ok=True)
        for entry, path in wanted.items():
            if os.path.exists(path):
                continue
            payload = fetch(listing[entry][resolution][file_format]["url"])
            # Through a temporary name, so an interrupted download cannot leave
            # a half-written file behind for the next run to trust and load.
            partial = f"{path}.partial"
            with open(partial, "wb") as handle:
                handle.write(payload)
            os.replace(partial, path)
    except Exception as exc:
        print(f"  could not fetch the {name} texture ({exc.__class__.__name__})")
        return {}

    size = sum(os.path.getsize(path) for path in wanted.values())
    print(f"  {name}: {', '.join(wanted)} at {resolution} ({size / 1e6:.1f} MB)")
    return wanted


def _mn_session_pickles() -> bool:
    """Report whether Molecular Nodes' session can be written beside the .blend.

    MN stashes its Python-side session — the parsed structures behind the
    objects — in a ``.blend.MNSession`` pickle, from a ``save_post`` handler.
    Some structures hold a reader object that pickle refuses, and a structure
    read from a file on disk is one of them.

    That has to be found out *before* saving rather than caught during it.
    Blender reports an exception raised inside a handler through
    ``sys.excepthook``, which is the hook :func:`_die_on_unhandled_exception`
    installed to stop a vignette dead — so by the time the failure is visible
    the process is already on its way out, and no ``try`` around the save runs.
    """
    import bpy

    # Read the session off the scene rather than importing it: MN is installed
    # as an extension, so the module it is imported under is not the name it
    # has on PyPI, and the property is the same object either way.
    session = getattr(bpy.context.scene, "MNSession", None)
    if session is None:  # pragma: no cover - MN is a soft dependency
        return True

    import pickle

    try:
        pickle.dumps(session)
    except Exception as exc:
        print(f"  Molecular Nodes cannot stash its session here ({exc}).")
        return False
    return True


def save_blend(name: str) -> str:
    """Save everything the vignette built as a .blend, and say how to open it.

    A vignette that only leaves a PNG behind is a demonstration; one that also
    leaves the scene is a starting point. The file holds what the script set up
    — molecule, styles, materials, lights, camera, compositor — so the next
    move is to open it and turn the knobs, which is the half of the work a
    script is the wrong tool for.

    Saving is deliberately the last thing each vignette does: it makes the
    saved path the one relative paths in the file resolve against, and it is
    the state at the end of the script that is worth reopening.

    Parameters
    ----------
    name : str
        File name stem, matching the vignette's own.

    Returns
    -------
    str
        The path written.
    """
    import bpy

    os.makedirs(BLEND_DIR, exist_ok=True)
    path = os.path.join(BLEND_DIR, f"{name}.blend")

    # Save without the session rather than not at all: everything Blender
    # itself needs — objects, node trees, materials, the compositor — is in the
    # .blend, and MN rebuilds what it can from that. It only skips its stash
    # when reopening if there is no file to find, so a stale one goes too.
    handlers = bpy.app.handlers.save_post
    suppressed = (
        []
        if _mn_session_pickles()
        else [
            handler
            for handler in list(handlers)
            if "molecularnodes" in getattr(handler, "__module__", "")
        ]
    )
    for handler in suppressed:
        handlers.remove(handler)

    try:
        # Compressed: these carry a whole structure's geometry, and the
        # difference is several fold for a file nothing reads sequentially.
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    finally:
        for handler in suppressed:
            handlers.append(handler)

    if suppressed:
        stale = f"{path}.MNSession"
        if os.path.exists(stale):
            os.remove(stale)
        print("  Saved without it; the scene itself is all in the .blend.")

    # The relative form only if it is genuinely shorter, which it is not when
    # GALA_VIGNETTE_BLEND_DIR points somewhere outside the repository.
    shown = os.path.relpath(path, REPO_ROOT)
    if shown.startswith(os.pardir):
        shown = path

    print(f"\n  scene saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"  open it with:  blender {shown}")
    return path


def render(gala, name: str, scene=None, extension: str = "png") -> str:
    """Render a scene into ``docs/images``.

    Parameters
    ----------
    gala : module
        The imported ``blender_gala`` module.
    name : str
        File name stem.
    scene : bpy.types.Scene, optional
        Scene to render. Defaults to the active one.
    extension : str, optional
        File extension, which has to agree with the scene's image format. PNG
        is lossless and right for a figure; a large detail render is several
        times smaller as WebP and no worse to look at.

    Returns
    -------
    str
        The path written.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, f"{name}.{extension}")
    if scene is None:
        gala.render(path)
    else:
        gala.scene.render.render(path, scene=scene)
    check_not_blank(path)
    print(f"  wrote {path}")
    return path


def check_not_blank(path: str) -> None:
    """Raise if the rendered image is a single flat colour.

    The ways a vignette produces an empty figure are all silent: a style whose
    selection names an attribute that does not exist draws nothing but only
    warns, and a depth cue whose range misses the molecule fades the whole
    frame to the background colour. Neither stops the script, so without this
    the vignette "passes" and ships a blank figure.

    Parameters
    ----------
    path : str
        The rendered image.

    Raises
    ------
    ValueError
        If every pixel is the same colour.
    """
    import bpy
    import numpy as np

    image = bpy.data.images.load(path)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(image)

    spread = float(np.ptp(pixels.reshape(-1, 4), axis=0).max())
    if spread < _FLAT_TOLERANCE:
        raise ValueError(
            f"{os.path.basename(path)} is one flat colour (spread {spread:.4f}): "
            "nothing rendered. Check that every style selection names an "
            "attribute that exists, and that the camera and depth cue bracket "
            "the molecule."
        )
