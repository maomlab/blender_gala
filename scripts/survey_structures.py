"""Run Gala's public API over real deposited structures from the PDB.

Every fixture in ``tests/`` is synthetic: a handful of atoms written by hand so
that the assertion can be read off the file. That is the right shape for a unit
test and it is a poor model of the archive. Real entries carry alternate
conformations, insertion codes, twenty NMR models, residues numbered from
``-73``, chain identifiers that differ from each other only in case, modified
nucleotides that are neither a nucleotide nor a ligand, and — in the entries
that no longer fit the PDB format at all — a quarter of a million atoms whose
``auth_*`` identifiers disagree with their ``label_*`` ones.

This script points the public API at those entries and reports what breaks. It
is the standing form of ``tests/ROBUSTNESS.md`` tier 3: it needs the network,
so it is not part of the suite and is never run in CI, but it is deterministic
enough to be run on a schedule and to fail loudly when something regresses.

    blender --background --python scripts/survey_structures.py
    blender --background --python scripts/survey_structures.py -- --format cif
    blender --background --python scripts/survey_structures.py -- 4HHB 1EHZ -v

Blender is used only for its interpreter: this is the science layer, so it runs
against a biotite ``AtomArray`` wrapped in an ``AtomStructure`` and never builds
a Blender object. ``Molecule.load`` is minutes of work for the large entries and
proves nothing extra about the selection language.

There are, however, *two* ways an ``AtomArray`` reaches Gala, and they do not
agree. Molecular Nodes' readers attach ``is_peptide``/``is_nucleic``/
``is_solvent`` boolean annotations that Gala's macros prefer when they are
there (``selection._flag_or``); a caller who reads the file with biotite
directly — the documented headless path in ``core/entity.py`` — gets Gala's own
residue-name tables instead. ``--reader both`` (the default) surveys each
structure twice and reports where the two answers differ, because a macro that
means different things in and out of Blender is a defect whichever answer is
the right one.

Exit status is non-zero if any check failed, so this can be scheduled.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import os
import shutil
import sys
import time
import traceback
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_gala.color.coloring import color_by_bfactor  # noqa: E402
from blender_gala.core.entity import AtomStructure  # noqa: E402
from blender_gala.core.exceptions import GalaError  # noqa: E402
from blender_gala.electrostatics.apbs import write_pdb  # noqa: E402
from blender_gala.electrostatics.grid import PotentialGrid  # noqa: E402
from blender_gala.electrostatics.surface import potential_at_atoms  # noqa: E402
from blender_gala.interactions.detect import find_interactions  # noqa: E402
from blender_gala.measure.measurements import angle, distance  # noqa: E402

# ---------------------------------------------------------------------------
# What to survey, and why
# ---------------------------------------------------------------------------

#: PDB id -> why this entry is in the list. Every claim here was checked
#: against the deposited file rather than taken from the entry's reputation;
#: several obvious candidates turned out not to contain the feature they are
#: famous for, and one (2N9S) is no longer in the archive at all.
CURATED: dict[str, str] = {
    "1CRN": "control: 46 residues, one chain, no hetero groups, no waters",
    "1UBQ": "control: protein plus ordered waters, the shape of most entries",
    "1EJG": "alternate conformations A/B/C at 0.54 A, with riding hydrogens",
    "3NIR": "alternate conformations A-D at 0.48 A; waters numbered into the 3000s",
    "12E8": "Fab: insertion codes A/B/C, four chains, 80 zero-occupancy atoms",
    "1IGT": "intact IgG: insertion codes A-K, N-glycans (NAG/BMA/MAN/FUC/GAL), 6 chains",
    "1D3Z": "NMR ensemble, 10 models, hydrogens present",
    "1G03": "NMR ensemble, 20 models — the multi-model path at a different depth",
    "4HHB": "four chains, four HEM cofactors, four FE, waters",
    "1AKE": "two chains, the AP5 bisubstrate ligand, alternate conformations",
    "1BNA": "B-DNA dodecamer: nucleic acid, primed atom names (C1', O5')",
    "1EHZ": "tRNA: 11 modified bases (PSU, 1MA, 7MG, H2U, YYG...), MG and MN",
    "6VXX": "SARS-CoV-2 spike: 18 chains, 66 NAG glycans, 23k atoms",
    "1A8O": "selenomethionine: MSE in the polymer, small enough to check by eye",
    "1U14": "selenomethionine plus a residue numbered 0 (an expression-tag remnant)",
    "1FIP": "UNK residues in the coordinates, next to a real polymer",
    "1KX5": "nucleosome: DNA numbered -73..+73, eight histone chains",
    "6J5K": "mmCIF only: ATP synthase tetramer, 153k atoms, 120 auth chains,"
    " 36 of which differ from another only in case",
    "4V6X": "mmCIF only: human 80S ribosome, 238k atoms, auth ids disagree with"
    " label ids on every single atom",
}

#: Entries the PDB no longer distributes in the legacy format. Asking for
#: ``pdb`` here is not a failure of Gala's, so the survey says so and moves on.
CIF_ONLY = frozenset({"6J5K", "4V6X"})

#: Refuse a download larger than this. 3J3Q and its friends are hundreds of
#: megabytes; nothing in the curated list is close, and the cap is here so that
#: adding an id on the command line cannot fill a disk by accident.
DOWNLOAD_CAP = 120 * 1024 * 1024

#: Above this many atoms, ``find_interactions(kinds="all")`` is not attempted
#: between two large selections — several of its detectors are quadratic in the
#: number of atoms matched (``tests/ROBUSTNESS.md`` §5, "Still open"). The
#: survey narrows the two sides instead and says that it did.
INTERACTION_ATOM_LIMIT = 60_000

#: A check slower than this is reported even when it succeeds. Nothing in the
#: public API should take a minute on a structure a user can open.
SLOW_SECONDS = 20.0


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

#: Ordered worst first, which is also the order the summary prints them in.
FINDING_LEVELS = ("fail", "warn", "note")


@dataclass
class Finding:
    """One observation about one structure.

    Parameters
    ----------
    level : {"fail", "warn", "note"}
        ``fail`` means a check did not hold — an exception that is not a
        ``GalaError``, a describe round trip that lost atoms, a write that
        dropped a chain. Only ``fail`` affects the exit status.
    check : str
        The check that produced it.
    message : str
        What was observed, verbatim where an exception was involved.
    """

    level: str
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():4}] {self.check}: {self.message}"


@dataclass
class Report:
    """Every finding for one structure in one format under one reader."""

    pdb_id: str
    fmt: str
    reader: str
    n_atoms: int = 0
    findings: list[Finding] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.pdb_id}.{self.fmt} [{self.reader}]"

    def add(self, level: str, check: str, message: str) -> None:
        self.findings.append(Finding(level, check, message))

    @property
    def failed(self) -> bool:
        return any(f.level == "fail" for f in self.findings)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch(pdb_id: str, fmt: str, cache: str) -> str:
    """Download one entry, or return the copy already in ``cache``.

    Parameters
    ----------
    pdb_id : str
        Four-character PDB id.
    fmt : {"pdb", "cif"}
        Which distribution to fetch.
    cache : str
        Directory to keep the decompressed files in. Reused across runs; the
        RCSB is asked for nothing it has already given us.

    Returns
    -------
    str
        Path to the decompressed file.

    Raises
    ------
    RuntimeError
        If the download exceeds :data:`DOWNLOAD_CAP`.
    """
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{pdb_id.upper()}.{fmt}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{fmt}.gz"
    buffer = io.BytesIO()
    with urllib.request.urlopen(url, timeout=180) as response:
        total = 0
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > DOWNLOAD_CAP:
                raise RuntimeError(
                    f"{url} is larger than the {DOWNLOAD_CAP // 1024 // 1024} MB cap"
                )
            buffer.write(chunk)
    buffer.seek(0)
    # Decompress into a temporary name so an interrupted run cannot leave a
    # half-written file behind that the next run would happily read.
    partial = path + ".part"
    with gzip.open(buffer) as source, open(partial, "wb") as target:
        shutil.copyfileobj(source, target)
    os.replace(partial, path)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_biotite(path: str, fmt: str) -> Any:
    """Read with biotite alone: the documented headless path.

    The extra fields are the ones Molecular Nodes asks for, so the only
    difference from :func:`read_mn` is the boolean annotations MN computes on
    top — which is exactly the difference this survey wants to isolate.
    """
    if fmt == "pdb":
        from biotite.structure.io import pdb

        file = pdb.PDBFile.read(path)
        return pdb.get_structure(
            file,
            model=1,
            extra_fields=["b_factor", "occupancy", "charge", "atom_id"],
            include_bonds=True,
        )

    from biotite.structure import connect_via_residue_names
    from biotite.structure.io import pdbx

    file = pdbx.CIFFile.read(path)
    array = pdbx.get_structure(
        file, model=1, extra_fields=["b_factor", "occupancy", "atom_id"]
    )
    if array.bonds is None:
        array.bonds = connect_via_residue_names(array, inter_residue=True)
    return array


def read_mn(path: str, fmt: str) -> Any:
    """Read through Molecular Nodes' own reader, minus the Blender object.

    ``Molecule.load`` builds a mesh, a node tree and a session entry, all of
    which take minutes on a ribosome and none of which the selection language
    touches. The reader classes underneath it are what produce the array, and
    they are what attach ``is_peptide`` and friends, so instantiating one is
    the faithful and cheap way to see what a user in Blender actually selects
    from.
    """
    from pathlib import Path

    if fmt == "pdb":
        from bl_ext.blender_org.molecularnodes.entities.molecule.pdb import PDBReader

        reader: Any = PDBReader(Path(path))
    else:
        from bl_ext.blender_org.molecularnodes.entities.molecule.pdbx import PDBXReader

        reader = PDBXReader(Path(path))

    array = reader.array
    if getattr(array, "stack_depth", None) is not None and array.coord.ndim == 3:
        # MN keeps every model; the survey works one frame at a time, as
        # AtomStructure.from_any would.
        array = array[0]
    return array


READERS = {"biotite": read_biotite, "mn": read_mn}


# ---------------------------------------------------------------------------
# The selection battery
# ---------------------------------------------------------------------------


def battery(structure: AtomStructure) -> list[str]:
    """Build the selection strings to try against ``structure``.

    Most are fixed; a few name a chain or a residue range and have to be taken
    from the structure in hand, since ``chain A`` on an entry whose chains are
    called ``Aa`` and ``AA`` tests nothing.
    """
    array = structure.array
    chains = [c for c in np.unique(np.asarray(array.chain_id).astype(str)) if c.strip()]
    first_chain = chains[0] if chains else "A"
    res_ids = np.unique(np.asarray(array.res_id))
    low = int(res_ids[0])
    high = int(res_ids[min(len(res_ids) - 1, 9)])

    return [
        "all",
        "protein",
        "nucleic",
        "ligand",
        "solvent",
        "metals",
        "ions",
        "backbone",
        "sidechain",
        "hydro",
        "aromatic",
        "byres (protein within 4 of ligand)",
        f"chain {first_chain}",
        f"resi {low}-{high}" if low >= 0 else f"resi {low}:{high}",
        "name CA",
        "b > 50",
        "q < 1",
        "alt A",
        "ins A",
        "elem FE+ZN+MG+MN",
        "index 1-25",
        "resn HOH",
        "byres (all within 5 of metals)",
        "not polymer and not solvent",
    ]


def masks_to_describe(
    structure: AtomStructure, results: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Masks worth putting through the describe/select round trip.

    The battery's own results are the interesting ones — they are the shapes a
    user actually produces — plus a few pathological ones a viewport pick makes
    easily: a single atom, a scattered subset that belongs to no whole residue,
    and everything-but-one-atom.
    """
    n = structure.n_atoms
    rng = np.random.default_rng(20260811)
    chosen = {
        name: mask
        for name, mask in results.items()
        if mask is not None and mask.any() and not mask.all()
    }

    if n:
        single = np.zeros(n, dtype=bool)
        single[n // 2] = True
        chosen["<one atom>"] = single

        all_but_one = np.ones(n, dtype=bool)
        all_but_one[0] = False
        chosen["<all but one>"] = all_but_one

        # Deliberately scattered: no residue is complete, which is the case
        # `_describe_by_chain` has to write out atom name by atom name. Capped
        # so that a quarter-million-atom entry does not turn one check into
        # several minutes of string building.
        scatter = np.zeros(n, dtype=bool)
        scatter[rng.choice(n, size=min(400, n), replace=False)] = True
        chosen["<scattered>"] = scatter

    return chosen


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def timed(report: Report, name: str):
    """Context-manager-ish helper: returns a closure that records elapsed time."""
    start = time.perf_counter()

    def done() -> float:
        elapsed = time.perf_counter() - start
        report.timings[name] = elapsed
        if elapsed > SLOW_SECONDS:
            report.add("warn", name, f"took {elapsed:.1f}s")
        return elapsed

    return done


def check_selections(
    structure: AtomStructure, report: Report, expectations: dict[str, bool]
) -> dict[str, np.ndarray]:
    """Evaluate the battery, recording counts and anything that escaped.

    ``expectations`` maps a selection to whether the *file* says it should
    match something, worked out independently of Gala — from biotite's own
    ``filter_*`` functions and from the raw annotation arrays. A selection that
    matches nothing where the file plainly has the feature is the quietest
    possible failure, so it is reported as one.
    """
    done = timed(report, "selections")
    results: dict[str, np.ndarray] = {}
    for text in battery(structure):
        try:
            mask = structure.select(text)
        except GalaError as exc:
            report.add("warn", "selections", f"{text!r} -> {type(exc).__name__}: {exc}")
            continue
        except Exception as exc:
            report.add(
                "fail",
                "selections",
                f"{text!r} raised {type(exc).__name__}: {exc}",
            )
            continue
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            report.add("fail", "selections", f"{text!r} returned {type(mask).__name__}")
            continue
        if mask.shape != (structure.n_atoms,):
            report.add("fail", "selections", f"{text!r} returned shape {mask.shape}")
            continue
        results[text] = mask
        report.counts[text] = int(mask.sum())

        expected = expectations.get(text)
        if expected is True and not mask.any():
            report.add(
                "fail",
                "selections",
                f"{text!r} matched no atoms, but the file contains that feature",
            )
        elif expected is False and mask.any():
            report.add(
                "warn",
                "selections",
                f"{text!r} matched {int(mask.sum())} atoms, but the file has none",
            )
    done()
    return results


def check_round_trip(
    structure: AtomStructure, report: Report, masks: dict[str, np.ndarray]
) -> None:
    """``select(describe(mask)) == mask``, the self-verifying property.

    ``describe_selection`` re-evaluates its own output before returning it and
    falls back to ``index ...`` when the chemical form does not reproduce the
    mask, so a failure here is a failure of that verification, not of the
    prettifier — which makes it the most valuable single check in this script.
    """
    done = timed(report, "describe_round_trip")
    for name, mask in masks.items():
        try:
            text = structure.describe(mask)
        except Exception as exc:
            report.add(
                "fail",
                "describe_round_trip",
                f"describe({name}) raised {type(exc).__name__}: {exc}",
            )
            continue
        try:
            back = structure.select(text)
        except Exception as exc:
            report.add(
                "fail",
                "describe_round_trip",
                f"describe({name}) -> {text[:120]!r} which does not parse: "
                f"{type(exc).__name__}: {exc}",
            )
            continue
        if not np.array_equal(back, mask):
            lost = int((mask & ~back).sum())
            gained = int((back & ~mask).sum())
            report.add(
                "fail",
                "describe_round_trip",
                f"describe({name}) -> {text[:120]!r} re-selects "
                f"{int(back.sum())} of {int(mask.sum())} atoms "
                f"({lost} lost, {gained} gained)",
            )
    done()


def check_expand(structure: AtomStructure, report: Report) -> None:
    """Every level, plus the two properties expansion has to have.

    Growing never shrinks, and expanding twice at one level is expanding once.
    Both are cheap to state and neither is checked anywhere in the permanent
    suite against a structure with more than one chain in it.
    """
    from blender_gala.core.selection import LEVELS as EXPAND_LEVELS

    done = timed(report, "expand")
    seed = structure.select("name CA") if structure.n_atoms else None
    if seed is None or not seed.any():
        seed = structure.select("index 1-50")
    if not seed.any():
        done()
        return

    for level in EXPAND_LEVELS:
        try:
            grown = structure.expand(seed, level=level)
        except Exception as exc:
            report.add(
                "fail", "expand", f"level={level!r} raised {type(exc).__name__}: {exc}"
            )
            continue
        if not (grown | seed == grown).all():
            report.add("fail", "expand", f"level={level!r} dropped atoms it was given")
        twice = structure.expand(grown, level=level)
        if not np.array_equal(twice, grown):
            report.add(
                "fail",
                "expand",
                f"level={level!r} is not idempotent: "
                f"{int(grown.sum())} then {int(twice.sum())}",
            )
        report.counts[f"expand {level}"] = int(grown.sum())

    near = structure.expand(seed, level="atom", distance=4.0)
    far = structure.expand(seed, level="atom", distance=8.0)
    if int(near.sum()) > int(far.sum()):
        report.add(
            "fail",
            "expand",
            f"distance=4 matched {int(near.sum())} atoms but distance=8 "
            f"matched {int(far.sum())}",
        )
    report.counts["expand 4A"] = int(near.sum())
    done()


def check_interactions(structure: AtomStructure, report: Report) -> None:
    """``find_interactions(kinds="all")`` between two real selections."""
    done = timed(report, "interactions")
    side_a, side_b = "ligand", "protein or nucleic"
    if not structure.select(side_a).any():
        side_a = "metals"
    if not structure.select(side_a).any():
        side_a = "solvent"
    if structure.n_atoms > INTERACTION_ATOM_LIMIT:
        # Narrow both sides rather than skipping: the point is to exercise the
        # detectors on real chemistry, not to measure how quadratic they are.
        first_chain = str(np.asarray(structure.array.chain_id)[0])
        side_a = f"({side_a}) and chain {first_chain}"
        side_b = f"({side_b}) and chain {first_chain}"
        report.add(
            "note",
            "interactions",
            f"{structure.n_atoms} atoms: restricted to chain {first_chain}",
        )
    try:
        contacts = find_interactions(structure, side_a, side_b, kinds="all")
    except GalaError as exc:
        report.add("warn", "interactions", f"{type(exc).__name__}: {exc}")
        done()
        return
    except Exception as exc:
        report.add(
            "fail",
            "interactions",
            f"{side_a!r} vs {side_b!r} raised {type(exc).__name__}: {exc}",
        )
        done()
        return
    report.counts["interactions"] = len(contacts)
    bad = [c for c in contacts if not np.isfinite(getattr(c, "distance", np.nan))]
    if bad:
        report.add(
            "fail", "interactions", f"{len(bad)} contacts have a non-finite distance"
        )
    done()


def check_empty_interactions(structure: AtomStructure, report: Report) -> None:
    """An answer that must be empty should be cheap to reach.

    ``find_interactions("none", "none")`` cannot find anything, so whatever it
    costs is work done before the selections were consulted. Timing it is the
    cleanest way to see a detector that perceives the whole structure and only
    then filters — which is how a 50-second wait for zero contacts happens.
    """
    done = timed(report, "interactions_empty")
    try:
        contacts = find_interactions(structure, "none", "none", kinds="all")
    except Exception as exc:
        report.add("fail", "interactions_empty", f"{type(exc).__name__}: {exc}")
        done()
        return
    if contacts:
        report.add(
            "fail",
            "interactions_empty",
            f"'none' vs 'none' found {len(contacts)} contacts",
        )
    elapsed = done()
    if elapsed > 1.0:
        report.add(
            "warn",
            "interactions_empty",
            f"{elapsed:.1f}s to find nothing between two empty selections on "
            f"{structure.n_atoms} atoms",
        )


def check_chain_identity(structure: AtomStructure, report: Report) -> None:
    """``chain X`` must match exactly the atoms whose chain id is ``X``.

    The selection language upper-cases string annotations before comparing,
    which is right for residue and atom names — nobody writes ``resn hoh``
    meaning something different from ``resn HOH`` — and wrong for a chain
    identifier, because mmCIF's ``auth_asym_id`` is case-sensitive and the
    large assemblies use that to get past 62 chains.
    """
    done = timed(report, "chain_identity")
    chains = np.asarray(structure.array.chain_id).astype(str)
    unique = [c for c in np.unique(chains) if c.strip()]
    # Only the ids that could collide are worth a selection each; on a
    # 120-chain entry, testing every one is a hundred KD-tree-free mask builds
    # and is still cheap, but the report only wants the ones that go wrong.
    for chain in unique:
        try:
            mask = structure.select(f"chain {chain}")
        except Exception as exc:
            report.add(
                "fail", "chain_identity", f"'chain {chain}' {type(exc).__name__}: {exc}"
            )
            continue
        expected = chains == chain
        if not np.array_equal(mask, expected):
            others = sorted(set(chains[mask & ~expected].tolist()))
            report.add(
                "fail",
                "chain_identity",
                f"'chain {chain}' matched {int(mask.sum())} atoms where the file "
                f"has {int(expected.sum())}; it also matched chain(s) {others}",
            )
    done()


def check_macro_agreement(structure: AtomStructure, report: Report) -> None:
    """Compare Gala's polymer macros with biotite's own residue classification.

    A residue biotite calls a nucleotide and Gala does not has to land
    somewhere, and where it lands is ``ligand`` — so a modified base in a tRNA
    is reported as a small molecule bound to it. The disagreement is reported
    with the residue names in it, because the names are the fix.
    """
    import biotite.structure as struc

    done = timed(report, "macro_agreement")
    res_name = np.asarray(structure.array.res_name).astype(str)
    with _quiet():
        oracles = {
            "protein": struc.filter_amino_acids(structure.array),
            "nucleic": struc.filter_nucleotides(structure.array),
            "solvent": struc.filter_solvent(structure.array),
        }
    for macro, oracle in oracles.items():
        mine = structure.select(macro)
        missed = sorted(set(res_name[oracle & ~mine].tolist()))
        extra = sorted(set(res_name[mine & ~oracle].tolist()))
        if missed:
            elsewhere = sorted(
                set(res_name[oracle & ~mine & structure.select("ligand")].tolist())
            )
            report.add(
                "warn",
                "macro_agreement",
                f"{macro!r} misses {int((oracle & ~mine).sum())} atoms biotite "
                f"classifies as {macro}: {missed}"
                + (f"; {elsewhere} come back as 'ligand'" if elsewhere else ""),
            )
        if extra:
            report.add(
                "note",
                "macro_agreement",
                f"{macro!r} adds {extra} that biotite does not classify as {macro}",
            )
    done()


def check_measure(structure: AtomStructure, report: Report) -> None:
    """``distance`` and ``angle`` on three atoms that are really there."""
    done = timed(report, "measure")
    indices = np.flatnonzero(structure.select("name CA"))
    if indices.size < 3:
        indices = np.arange(min(3, structure.n_atoms))
    if indices.size < 3:
        done()
        return
    picks = [f"index {int(i) + 1}" for i in indices[:3]]
    try:
        d = distance(structure, picks[0], picks[1], draw=False)
        a = angle(structure, picks[0], picks[1], picks[2], draw=False)
    except Exception as exc:
        report.add("fail", "measure", f"{type(exc).__name__}: {exc}")
        done()
        return
    if not np.isfinite(float(d)) or not np.isfinite(float(a)):
        report.add("fail", "measure", f"distance={float(d)} angle={float(a)}")
    report.counts["distance x100"] = int(float(d) * 100)
    done()


def check_color(structure: AtomStructure, report: Report) -> None:
    """``color_by_bfactor`` with ``write=False``: no mesh, just the mapping."""
    done = timed(report, "color")
    try:
        result = color_by_bfactor(structure, write=False)
    except GalaError as exc:
        report.add("warn", "color", f"{type(exc).__name__}: {exc}")
        done()
        return
    except Exception as exc:
        report.add("fail", "color", f"{type(exc).__name__}: {exc}")
        done()
        return
    colors = np.asarray(result.colors)
    if colors.shape != (structure.n_atoms, 4):
        report.add("fail", "color", f"returned colours of shape {colors.shape}")
    elif not np.isfinite(colors).all():
        report.add("fail", "color", "returned non-finite colour channels")
    report.counts["coloured"] = int(result.n_colored)
    done()


def check_potential(structure: AtomStructure, report: Report) -> None:
    """``potential_at_atoms`` against a synthetic grid over the structure.

    The grid is a linear ramp in *x*, so the answer is checkable: a value read
    outside the grid would fall outside the ramp's own range.
    """
    done = timed(report, "potential")
    coord = structure.coord
    finite = coord[np.isfinite(coord).all(axis=1)]
    if finite.size == 0:
        done()
        return
    low = finite.min(axis=0) - 5.0
    high = finite.max(axis=0) + 5.0
    shape = (16, 16, 16)
    spacing = (high - low) / (np.asarray(shape) - 1)
    ramp = np.linspace(-10.0, 10.0, shape[0])
    values = np.broadcast_to(ramp[:, None, None], shape).copy()
    grid = PotentialGrid(values=values, origin=low, spacing=spacing, source="survey")

    try:
        sampled = potential_at_atoms(structure, grid, points=16)
    except GalaError as exc:
        report.add("warn", "potential", f"{type(exc).__name__}: {exc}")
        done()
        return
    except Exception as exc:
        report.add("fail", "potential", f"{type(exc).__name__}: {exc}")
        done()
        return
    if sampled.shape != (structure.n_atoms,):
        report.add("fail", "potential", f"returned shape {sampled.shape}")
    real = sampled[np.isfinite(sampled)]
    if real.size and (real.min() < -10.001 or real.max() > 10.001):
        report.add(
            "fail",
            "potential",
            f"sampled [{real.min():.3f}, {real.max():.3f}] outside the grid's "
            "own [-10, 10]",
        )
    report.counts["potential valued"] = int(real.size)
    done()


def check_write_pdb(structure: AtomStructure, report: Report, out_dir: str) -> None:
    """Round-trip the structure through ``apbs.write_pdb``.

    That function is how a structure reaches PDB2PQR, and it writes the legacy
    format unconditionally. Anything the format cannot hold — a two-character
    chain id, a hundred-thousandth atom, a residue number past 9999 — is lost
    or mangled here, and the caller is told nothing. So the check is to read
    the file back and compare it with what went in.
    """
    done = timed(report, "write_pdb")
    path = os.path.join(out_dir, f"{report.pdb_id}_{report.fmt}_{report.reader}.pdb")
    try:
        write_pdb(structure, path)
    except GalaError as exc:
        report.add("warn", "write_pdb", f"{type(exc).__name__}: {exc}")
        done()
        return
    except Exception as exc:
        report.add("fail", "write_pdb", f"{type(exc).__name__}: {exc}")
        done()
        return

    from biotite.structure.io import pdb

    try:
        back = pdb.get_structure(pdb.PDBFile.read(path), model=1)
    except Exception as exc:
        report.add(
            "fail",
            "write_pdb",
            f"the file it wrote cannot be read back: {type(exc).__name__}: {exc}",
        )
        done()
        return

    if len(back) != structure.n_atoms:
        report.add(
            "fail",
            "write_pdb",
            f"wrote {len(back)} atoms for a {structure.n_atoms}-atom structure",
        )
    before = set(np.unique(np.asarray(structure.array.chain_id).astype(str)).tolist())
    after = set(np.unique(np.asarray(back.chain_id).astype(str)).tolist())
    if before != after:
        report.add(
            "fail",
            "write_pdb",
            f"chain ids changed: {sorted(before)[:12]} -> {sorted(after)[:12]}",
        )
    before_res = set(np.unique(np.asarray(structure.array.res_id)).tolist())
    after_res = set(np.unique(np.asarray(back.res_id)).tolist())
    if not before_res <= after_res:
        missing = sorted(before_res - after_res)
        report.add(
            "fail",
            "write_pdb",
            f"{len(missing)} residue numbers did not survive, e.g. {missing[:8]}",
        )
    with contextlib.suppress(OSError):
        os.remove(path)
    done()


# ---------------------------------------------------------------------------
# Expectations, computed from the file rather than from Gala
# ---------------------------------------------------------------------------


def expectations(array: Any, path: str, fmt: str) -> dict[str, bool]:
    """What the *file* says each selection should find.

    Deliberately computed with biotite's own filters and with the raw
    annotation arrays, so that a disagreement is evidence about Gala rather
    than Gala agreeing with itself.
    """
    import biotite.structure as struc

    out: dict[str, bool] = {}
    res_name = np.asarray(array.res_name).astype(str)

    with _quiet():
        out["protein"] = bool(struc.filter_amino_acids(array).any())
        out["nucleic"] = bool(struc.filter_nucleotides(array).any())
        out["solvent"] = bool(struc.filter_solvent(array).any())

    element = np.asarray(getattr(array, "element", np.array([]))).astype(str)
    out["metals"] = bool(
        np.isin(
            np.char.upper(element), ["FE", "ZN", "MG", "MN", "CA", "CU", "NI"]
        ).any()
    )
    out["elem FE+ZN+MG+MN"] = bool(
        np.isin(np.char.upper(element), ["FE", "ZN", "MG", "MN"]).any()
    )
    out["resn HOH"] = bool((np.char.upper(res_name) == "HOH").any())

    ins = getattr(array, "ins_code", None)
    if ins is not None:
        out["ins A"] = bool((np.asarray(ins).astype(str) == "A").any())

    # The altloc question cannot be asked of the array Gala was handed, because
    # every reader in use here has already collapsed the alternates. It is a
    # property of the file, so read the file again and ask it.
    out["alt A"] = _file_has_altloc(path, fmt)

    occupancy = getattr(array, "occupancy", None)
    if occupancy is not None:
        out["q < 1"] = bool((np.asarray(occupancy) < 1.0).any())

    b_factor = getattr(array, "b_factor", None)
    if b_factor is not None:
        out["b > 50"] = bool((np.asarray(b_factor) > 50.0).any())

    return out


def _file_has_altloc(path: str, fmt: str) -> bool:
    """Does the deposited file carry an alternate location indicator of ``A``?"""
    if fmt == "pdb":
        with open(path, errors="replace") as handle:
            return any(
                line.startswith(("ATOM  ", "HETATM")) and line[16:17] == "A"
                for line in handle
            )
    from biotite.structure.io import pdbx

    block = pdbx.CIFFile.read(path).block
    column = block["atom_site"].get("label_alt_id")
    if column is None:
        return False
    return bool((column.as_array(str) == "A").any())


@contextlib.contextmanager
def _quiet():
    """Silence biotite's warnings about residues and elements it cannot place.

    They are about the *file*, not about Gala, and on a ribosome there are
    thousands of them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def survey_one(pdb_id: str, fmt: str, reader: str, cache: str, out_dir: str) -> Report:
    """Run every check against one entry, in one format, through one reader."""
    report = Report(pdb_id=pdb_id, fmt=fmt, reader=reader)

    try:
        path = fetch(pdb_id, fmt, cache)
    except Exception as exc:
        report.add("fail", "fetch", f"{type(exc).__name__}: {exc}")
        return report

    done = timed(report, "read")
    try:
        array = READERS[reader](path, fmt)
    except Exception as exc:
        report.add("fail", "read", f"{type(exc).__name__}: {exc}")
        done()
        return report
    done()

    done = timed(report, "from_any")
    try:
        structure = AtomStructure.from_any(array)
    except Exception as exc:
        report.add("fail", "from_any", f"{type(exc).__name__}: {exc}")
        done()
        return report
    done()
    report.n_atoms = structure.n_atoms

    expected = expectations(array, path, fmt)
    results = check_selections(structure, report, expected)
    check_round_trip(structure, report, masks_to_describe(structure, results))
    check_expand(structure, report)
    check_chain_identity(structure, report)
    check_macro_agreement(structure, report)
    check_interactions(structure, report)
    check_empty_interactions(structure, report)
    check_measure(structure, report)
    check_color(structure, report)
    check_potential(structure, report)
    check_write_pdb(structure, report, out_dir)
    return report


def compare_formats(reports: list[Report]) -> list[Finding]:
    """Report where ``.pdb`` and ``.cif`` of one entry disagree.

    An entry is one structure; a selection that means one thing in one
    distribution and another in the other is a defect regardless of which
    distribution is right.
    """
    findings: list[Finding] = []
    by_key: dict[tuple[str, str], dict[str, Report]] = {}
    for report in reports:
        by_key.setdefault((report.pdb_id, report.reader), {})[report.fmt] = report

    for (pdb_id, reader), formats in sorted(by_key.items()):
        if len(formats) < 2:
            continue
        a, b = formats["pdb"], formats["cif"]
        if a.failed or b.failed or not a.n_atoms or not b.n_atoms:
            continue
        if a.n_atoms != b.n_atoms:
            findings.append(
                Finding(
                    "warn",
                    f"{pdb_id} [{reader}] pdb-vs-cif",
                    f"{a.n_atoms} atoms in .pdb, {b.n_atoms} in .cif",
                )
            )
        for key in sorted(set(a.counts) & set(b.counts)):
            if key.startswith(("expand", "interactions", "distance", "potential")):
                continue
            if a.counts[key] != b.counts[key]:
                findings.append(
                    Finding(
                        "warn",
                        f"{pdb_id} [{reader}] pdb-vs-cif",
                        f"{key!r}: {a.counts[key]} atoms in .pdb, "
                        f"{b.counts[key]} in .cif",
                    )
                )
    return findings


def compare_readers(reports: list[Report]) -> list[Finding]:
    """Report where the biotite path and the Molecular Nodes path disagree."""
    findings: list[Finding] = []
    by_key: dict[tuple[str, str], dict[str, Report]] = {}
    for report in reports:
        by_key.setdefault((report.pdb_id, report.fmt), {})[report.reader] = report

    for (pdb_id, fmt), readers in sorted(by_key.items()):
        if len(readers) < 2:
            continue
        a, b = readers["biotite"], readers["mn"]
        if not a.n_atoms or not b.n_atoms or a.n_atoms != b.n_atoms:
            continue
        for key in sorted(set(a.counts) & set(b.counts)):
            if key.startswith(("expand", "interactions", "distance", "potential")):
                continue
            if a.counts[key] != b.counts[key]:
                findings.append(
                    Finding(
                        "warn",
                        f"{pdb_id}.{fmt} biotite-vs-mn",
                        f"{key!r}: {a.counts[key]} atoms read with biotite, "
                        f"{b.counts[key]} read through Molecular Nodes",
                    )
                )
    return findings


def print_report(report: Report, verbose: bool) -> None:
    """One block per structure: what it is, what it cost, what it found."""
    why = CURATED.get(report.pdb_id, "")
    print(f"\n--- {report.label}  ({report.n_atoms} atoms)")
    if why:
        print(f"    why: {why}")
    if verbose and report.counts:
        for key, value in report.counts.items():
            print(f"    {key:38} {value}")
    if verbose and report.timings:
        slow = sorted(report.timings.items(), key=lambda kv: -kv[1])[:4]
        print("    slowest: " + ", ".join(f"{k}={v:.2f}s" for k, v in slow))
    if not report.findings:
        print("    ok")
        return
    for level in FINDING_LEVELS:
        for finding in report.findings:
            if finding.level == level:
                print(f"    {finding}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="survey_structures.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ids",
        nargs="*",
        metavar="PDBID",
        help="entries to survey; defaults to the curated list",
    )
    parser.add_argument(
        "--format",
        choices=("pdb", "cif", "both"),
        default="both",
        help="which distribution to read (default: both, and compare them)",
    )
    parser.add_argument(
        "--reader",
        choices=("biotite", "mn", "both"),
        default="both",
        help="biotite alone, Molecular Nodes' reader, or both and compare",
    )
    parser.add_argument(
        "--cache",
        default=os.path.join(REPO_ROOT, "build", "pdb-cache"),
        help="directory to keep downloaded structures in",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ids = [i.upper() for i in args.ids] or list(CURATED)
    formats = ("pdb", "cif") if args.format == "both" else (args.format,)
    readers = ("biotite", "mn") if args.reader == "both" else (args.reader,)

    if "mn" in readers:
        try:
            import bl_ext.blender_org.molecularnodes  # noqa: F401
        except ImportError:
            print("Molecular Nodes is not available; surveying with biotite only")
            readers = tuple(r for r in readers if r != "mn") or ("biotite",)

    out_dir = os.path.join(args.cache, "written")
    os.makedirs(out_dir, exist_ok=True)

    print(f"surveying {len(ids)} entries: {', '.join(ids)}")
    print(f"formats: {', '.join(formats)}   readers: {', '.join(readers)}")
    print(f"cache: {args.cache}")

    reports: list[Report] = []
    for pdb_id in ids:
        for fmt in formats:
            if fmt == "pdb" and pdb_id in CIF_ONLY:
                print(f"\n--- {pdb_id}.pdb  (skipped: distributed as mmCIF only)")
                continue
            for reader in readers:
                try:
                    report = survey_one(pdb_id, fmt, reader, args.cache, out_dir)
                except Exception:
                    report = Report(pdb_id=pdb_id, fmt=fmt, reader=reader)
                    report.add("fail", "survey", traceback.format_exc(limit=6))
                reports.append(report)
                print_report(report, args.verbose)

    cross = compare_formats(reports) + compare_readers(reports)
    if cross:
        print("\n=== cross-checks ===")
        for finding in cross:
            print(f"    {finding}")

    print("\n=== summary ===")
    failed = [r for r in reports if r.failed]
    counted = dict.fromkeys(FINDING_LEVELS, 0)
    for report in reports:
        for finding in report.findings:
            counted[finding.level] += 1
    print(
        f"{len(reports)} structure/format/reader combinations, "
        f"{len(failed)} with at least one failure"
    )
    print(
        "findings: "
        + ", ".join(f"{counted[level]} {level}" for level in FINDING_LEVELS)
        + f", {len(cross)} cross-check"
    )
    for report in failed:
        for finding in report.findings:
            if finding.level == "fail":
                print(f"    {report.label}: {finding.check}: {finding.message}")
    return 1 if failed else 0


if __name__ == "__main__":
    # Blender ignores a script's return value, so the status has to be set
    # explicitly for this to be worth scheduling.
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
