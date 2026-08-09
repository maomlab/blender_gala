"""A PyMOL-compatible atom selection language.

Structural biologists already think in PyMOL selection syntax, so Gala speaks
it (SPECIFICATION D-5). A selection string is tokenised, parsed into a small
expression tree, and evaluated against a biotite ``AtomArray`` to produce a
boolean mask. Nothing here touches Blender, which makes the whole language
unit-testable and reusable for trajectories (SPECIFICATION D-6).

Examples
--------
>>> select(array, "chain A and resi 10-20 and name CA")   # doctest: +SKIP
>>> select(array, "byres (protein within 4 of ligand)")   # doctest: +SKIP
>>> select(array, "not (solvent or hydro) and b > 70")    # doctest: +SKIP
"""

from __future__ import annotations

import fnmatch
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from . import chemistry
from .exceptions import SelectionSyntaxError

__all__ = [
    "LEVELS",
    "MACRO_KEYWORDS",
    "PROPERTY_KEYWORDS",
    "Selection",
    "SelectionContext",
    "compile_selection",
    "describe_selection",
    "expand_selection",
    "select",
    "select_indices",
]

#: Selection levels, from finest to coarsest. These are what PyMOL's selection
#: mode buttons switch between: a click gives you an atom, and the level grows
#: it to the residue, chain or bonded fragment that atom belongs to.
LEVELS = ("atom", "residue", "chain", "fragment", "object")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<lparen>\()
    | (?P<rparen>\))
    | (?P<compare>>=|<=|!=|<>|==|=|>|<)
    | (?P<and>&&|&)
    | (?P<or>\|\||\|)
    | (?P<not>!)
    | (?P<word>[^\s()<>=!&|]+)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    pos: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    length = len(text)
    while pos < length:
        match = _TOKEN_RE.match(text, pos)
        if match is None:  # pragma: no cover - the word rule is a catch-all
            raise SelectionSyntaxError("unexpected character", text, pos)
        kind = match.lastgroup
        assert kind is not None
        value = match.group()
        if kind != "ws":
            tokens.append(_Token(kind, value, pos))
        pos = match.end()
    return tokens


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------


@dataclass
class SelectionContext:
    """Per-array cache used while evaluating a selection.

    Building the KD-tree or the residue-grouping arrays is far more expensive
    than the mask arithmetic around them, so they are computed once per
    evaluation and shared by every node in the expression tree.
    """

    array: Any
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_atoms(self) -> int:
        return len(self.array)

    def annotation(self, name: str) -> np.ndarray | None:
        """Return an annotation array from the structure, or ``None``."""
        key = f"annot:{name}"
        if key in self._cache:
            return self._cache[key]
        value = getattr(self.array, name, None)
        if value is not None:
            value = np.asarray(value)
        self._cache[key] = value
        return value

    def upper(self, name: str) -> np.ndarray:
        """Return an annotation as an upper-cased, stripped string array."""
        key = f"upper:{name}"
        if key not in self._cache:
            raw = self.annotation(name)
            if raw is None:
                self._cache[key] = np.full(self.n_atoms, "", dtype="<U8")
            else:
                self._cache[key] = np.char.upper(np.char.strip(raw.astype(str)))
        return self._cache[key]

    @property
    def coord(self) -> np.ndarray:
        """Atom coordinates in ångström, shape ``(n_atoms, 3)``."""
        if "coord" not in self._cache:
            coord = np.asarray(self.array.coord, dtype=float)
            if coord.ndim == 3:  # AtomArrayStack: use the first model
                coord = coord[0]
            self._cache["coord"] = coord
        return self._cache["coord"]

    @property
    def residue_key(self) -> np.ndarray:
        """Integer id that is constant within a residue and unique across them."""
        if "residue_key" not in self._cache:
            chain = self.upper("chain_id")
            res_id = self.annotation("res_id")
            ins = self.annotation("ins_code")
            if res_id is None:
                res_id = np.zeros(self.n_atoms, dtype=int)
            parts = [chain, np.asarray(res_id).astype(str)]
            if ins is not None:
                parts.append(np.asarray(ins).astype(str))
            joined = np.array(["\x00".join(t) for t in zip(*parts, strict=False)])
            _, inverse = np.unique(joined, return_inverse=True)
            self._cache["residue_key"] = inverse
        return self._cache["residue_key"]

    @property
    def chain_key(self) -> np.ndarray:
        """Integer id that is constant within a chain."""
        if "chain_key" not in self._cache:
            _, inverse = np.unique(self.upper("chain_id"), return_inverse=True)
            self._cache["chain_key"] = inverse
        return self._cache["chain_key"]

    @property
    def fragment_key(self) -> np.ndarray:
        """Integer id that is constant within a bonded fragment.

        This is the level PyMOL calls a *molecule*: the connected component of
        the bond graph, so picking one atom of a ligand can grow to the whole
        ligand without going through its residue. Structures that arrived
        without a bond list fall back to the chain, which is the closest
        honest answer for a polymer.
        """
        if "fragment_key" not in self._cache:
            self._cache["fragment_key"] = self._connected_components()
        return self._cache["fragment_key"]

    def _connected_components(self) -> np.ndarray:
        edges = self._bond_edges()
        if edges.size == 0:
            return self.chain_key

        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:  # pragma: no cover - scipy ships with Blender + MN
            return _components_union_find(self.n_atoms, edges)

        n = self.n_atoms
        data = np.ones(len(edges), dtype=np.int8)
        graph = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        return np.asarray(labels)

    def _bond_edges(self) -> np.ndarray:
        """Bonds as an ``(n, 2)`` array of atom indices, or an empty array."""
        bonds = getattr(self.array, "bonds", None)
        if bonds is None:
            return np.zeros((0, 2), dtype=int)
        try:
            table = np.asarray(bonds.as_array(), dtype=int)
        except Exception:  # pragma: no cover - depends on the biotite version
            return np.zeros((0, 2), dtype=int)
        if table.size == 0:
            return np.zeros((0, 2), dtype=int)
        edges = table[:, :2]
        # A bond list written for a larger array — a stack sliced down to one
        # model, say — would index out of range and crash the graph build.
        valid = (edges >= 0).all(axis=1) & (edges < self.n_atoms).all(axis=1)
        return edges[valid]

    def neighbours_within(self, mask: np.ndarray, cutoff: float) -> np.ndarray:
        """Return atoms within ``cutoff`` ångström of any atom in ``mask``."""
        if not mask.any() or cutoff <= 0:
            return np.zeros(self.n_atoms, dtype=bool)
        coord = self.coord
        source = coord[mask]
        try:
            from scipy.spatial import cKDTree
        except ImportError:  # pragma: no cover - scipy ships with Blender + MN
            deltas = coord[:, None, :] - source[None, :, :]
            distances = np.einsum("ijk,ijk->ij", deltas, deltas)
            return np.any(distances <= cutoff * cutoff, axis=1)

        tree = self._cache.get("kdtree")
        if tree is None:
            tree = cKDTree(coord)
            self._cache["kdtree"] = tree
        hits = tree.query_ball_point(source, r=cutoff)
        out = np.zeros(self.n_atoms, dtype=bool)
        for group in hits:
            if group:
                out[np.asarray(group, dtype=int)] = True
        return out

    def expand_to_residues(self, mask: np.ndarray) -> np.ndarray:
        """Grow ``mask`` to cover every atom of each touched residue."""
        if not mask.any():
            return mask
        keys = self.residue_key
        return np.isin(keys, np.unique(keys[mask]))

    def expand_to_chains(self, mask: np.ndarray) -> np.ndarray:
        """Grow ``mask`` to cover every atom of each touched chain."""
        if not mask.any():
            return mask
        keys = self.chain_key
        return np.isin(keys, np.unique(keys[mask]))

    def expand_to_fragments(self, mask: np.ndarray) -> np.ndarray:
        """Grow ``mask`` to cover every atom bonded to each touched atom."""
        if not mask.any():
            return mask
        keys = self.fragment_key
        return np.isin(keys, np.unique(keys[mask]))


def _components_union_find(n_atoms: int, edges: np.ndarray) -> np.ndarray:
    """Connected components without scipy. Only reached if scipy is missing."""
    parent = np.arange(n_atoms)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b in edges:
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a

    roots = np.array([find(i) for i in range(n_atoms)])
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


class _Node(ABC):
    @abstractmethod
    def evaluate(self, ctx: SelectionContext) -> np.ndarray: ...


@dataclass
class _Constant(_Node):
    value: bool

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return np.full(ctx.n_atoms, self.value, dtype=bool)


@dataclass
class _And(_Node):
    left: _Node
    right: _Node

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return self.left.evaluate(ctx) & self.right.evaluate(ctx)


@dataclass
class _Or(_Node):
    left: _Node
    right: _Node

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return self.left.evaluate(ctx) | self.right.evaluate(ctx)


@dataclass
class _Not(_Node):
    operand: _Node

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return ~self.operand.evaluate(ctx)


@dataclass
class _Macro(_Node):
    name: str

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return MACRO_KEYWORDS[self.name](ctx)


@dataclass
class _StringMatch(_Node):
    """Match a string annotation against a ``+``/``,``-separated value list."""

    annotation: str
    patterns: tuple[str, ...]

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        values = ctx.upper(self.annotation)
        out = np.zeros(ctx.n_atoms, dtype=bool)
        for pattern in self.patterns:
            if any(ch in pattern for ch in "*?["):
                regex = re.compile(fnmatch.translate(pattern))
                out |= np.array([bool(regex.match(v)) for v in values], dtype=bool)
            else:
                out |= values == pattern
        return out


@dataclass
class _IntRanges(_Node):
    """Match an integer annotation against values and inclusive ranges."""

    annotation: str
    singles: tuple[int, ...]
    ranges: tuple[tuple[int | None, int | None], ...]

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        values = ctx.annotation(self.annotation)
        if values is None:
            return np.zeros(ctx.n_atoms, dtype=bool)
        values = np.asarray(values)
        out = np.zeros(ctx.n_atoms, dtype=bool)
        if self.singles:
            out |= np.isin(values, np.asarray(self.singles))
        for low, high in self.ranges:
            part = np.ones(ctx.n_atoms, dtype=bool)
            if low is not None:
                part &= values >= low
            if high is not None:
                part &= values <= high
            out |= part
        return out


@dataclass
class _IndexRanges(_Node):
    """Match by positional index. ``index`` is 1-based, as in PyMOL."""

    singles: tuple[int, ...]
    ranges: tuple[tuple[int | None, int | None], ...]
    one_based: bool = True

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        offset = 1 if self.one_based else 0
        values = np.arange(ctx.n_atoms) + offset
        out = np.zeros(ctx.n_atoms, dtype=bool)
        if self.singles:
            out |= np.isin(values, np.asarray(self.singles))
        for low, high in self.ranges:
            part = np.ones(ctx.n_atoms, dtype=bool)
            if low is not None:
                part &= values >= low
            if high is not None:
                part &= values <= high
            out |= part
        return out


_COMPARATORS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "=": lambda a, b: a == b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<>": lambda a, b: a != b,
}


@dataclass
class _NumericCompare(_Node):
    annotation: str
    operator: str
    value: float

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        values = ctx.annotation(self.annotation)
        if values is None:
            return np.zeros(ctx.n_atoms, dtype=bool)
        return _COMPARATORS[self.operator](np.asarray(values, dtype=float), self.value)


@dataclass
class _Within(_Node):
    """``A within N of B`` — atoms of A no further than N ångström from B."""

    subject: _Node
    reference: _Node
    cutoff: float
    exclude_self: bool = False

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        reference = self.reference.evaluate(ctx)
        near = ctx.neighbours_within(reference, self.cutoff)
        if self.exclude_self:
            near &= ~reference
        return self.subject.evaluate(ctx) & near


@dataclass
class _Expand(_Node):
    """``A expand N`` — A plus everything within N ångström of A."""

    operand: _Node
    cutoff: float

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        mask = self.operand.evaluate(ctx)
        return mask | ctx.neighbours_within(mask, self.cutoff)


@dataclass
class _ByResidue(_Node):
    operand: _Node

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return ctx.expand_to_residues(self.operand.evaluate(ctx))


@dataclass
class _ByChain(_Node):
    operand: _Node

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return ctx.expand_to_chains(self.operand.evaluate(ctx))


@dataclass
class _ByFragment(_Node):
    operand: _Node

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        return ctx.expand_to_fragments(self.operand.evaluate(ctx))


@dataclass
class _Edge(_Node):
    """``first``/``last`` — reduce a selection to its first or last atom."""

    operand: _Node
    last: bool = False

    def evaluate(self, ctx: SelectionContext) -> np.ndarray:
        mask = self.operand.evaluate(ctx)
        indices = np.flatnonzero(mask)
        out = np.zeros(ctx.n_atoms, dtype=bool)
        if indices.size:
            out[indices[-1] if self.last else indices[0]] = True
        return out


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def _res_names(ctx: SelectionContext) -> np.ndarray:
    return ctx.upper("res_name")


def _atom_names(ctx: SelectionContext) -> np.ndarray:
    return ctx.upper("atom_name")


def _elements(ctx: SelectionContext) -> np.ndarray:
    elements = ctx.upper("element")
    if not np.any(elements != ""):
        # Some PDB files omit the element column; fall back to the leading
        # alphabetic run of the atom name, which is right for the vast
        # majority of PDB-style names.
        names = _atom_names(ctx)
        elements = np.array(
            [re.sub(r"[^A-Z].*$", "", n)[:2] or "C" for n in names], dtype=object
        ).astype(str)
    return elements


def _flag_or(ctx: SelectionContext, name: str, fallback: np.ndarray) -> np.ndarray:
    """Prefer a boolean annotation stored by Molecular Nodes, else compute it."""
    stored = ctx.annotation(name)
    if stored is not None and stored.dtype == bool:
        return np.asarray(stored, dtype=bool)
    return fallback


def _macro_protein(ctx: SelectionContext) -> np.ndarray:
    return _flag_or(
        ctx, "is_peptide", np.isin(_res_names(ctx), list(chemistry.AMINO_ACIDS))
    )


def _macro_nucleic(ctx: SelectionContext) -> np.ndarray:
    return _flag_or(
        ctx, "is_nucleic", np.isin(_res_names(ctx), list(chemistry.NUCLEOTIDES))
    )


def _macro_polymer(ctx: SelectionContext) -> np.ndarray:
    return _macro_protein(ctx) | _macro_nucleic(ctx)


def _macro_solvent(ctx: SelectionContext) -> np.ndarray:
    return _flag_or(
        ctx, "is_solvent", np.isin(_res_names(ctx), list(chemistry.SOLVENT_NAMES))
    )


def _macro_hetero(ctx: SelectionContext) -> np.ndarray:
    stored = ctx.annotation("hetero")
    if stored is not None and stored.dtype == bool:
        return np.asarray(stored, dtype=bool)
    return _flag_or(ctx, "is_hetero", ~_macro_polymer(ctx))


def _macro_backbone(ctx: SelectionContext) -> np.ndarray:
    names = _atom_names(ctx)
    protein = _macro_protein(ctx) & np.isin(
        names, list(chemistry.PROTEIN_BACKBONE_ATOMS)
    )
    nucleic = _macro_nucleic(ctx) & np.isin(
        names, list(chemistry.NUCLEIC_BACKBONE_ATOMS)
    )
    return protein | nucleic


def _macro_sidechain(ctx: SelectionContext) -> np.ndarray:
    return _macro_polymer(ctx) & ~_macro_backbone(ctx) & ~_macro_hydrogen(ctx)


def _macro_hydrogen(ctx: SelectionContext) -> np.ndarray:
    return np.isin(_elements(ctx), list(chemistry.HYDROGEN_ELEMENTS))


def _macro_ions(ctx: SelectionContext) -> np.ndarray:
    return np.isin(_res_names(ctx), list(chemistry.MONOATOMIC_IONS)) & ~_macro_polymer(
        ctx
    )


def _macro_metals(ctx: SelectionContext) -> np.ndarray:
    return np.isin(_elements(ctx), list(chemistry.METALS)) & ~_macro_polymer(ctx)


def _macro_ligand(ctx: SelectionContext) -> np.ndarray:
    return (
        _macro_hetero(ctx)
        & ~_macro_solvent(ctx)
        & ~_macro_ions(ctx)
        & ~_macro_polymer(ctx)
    )


def _macro_donors(ctx: SelectionContext) -> np.ndarray:
    return np.isin(_elements(ctx), list(chemistry.DONOR_ELEMENTS))


def _macro_acceptors(ctx: SelectionContext) -> np.ndarray:
    return np.isin(_elements(ctx), list(chemistry.ACCEPTOR_ELEMENTS))


def _macro_polar(ctx: SelectionContext) -> np.ndarray:
    return np.isin(_elements(ctx), list(chemistry.POLAR_ELEMENTS))


def _macro_carbon(ctx: SelectionContext) -> np.ndarray:
    return _elements(ctx) == "C"


def _macro_aromatic(ctx: SelectionContext) -> np.ndarray:
    res_names = _res_names(ctx)
    atom_names = _atom_names(ctx)
    out = np.zeros(ctx.n_atoms, dtype=bool)
    for res, rings in chemistry.AROMATIC_RINGS.items():
        in_res = res_names == res
        if not in_res.any():
            continue
        for ring in rings:
            out |= in_res & np.isin(atom_names, list(ring))
    return out


def _macro_alpha_carbon(ctx: SelectionContext) -> np.ndarray:
    return _flag_or(
        ctx, "is_alpha_carbon", _macro_protein(ctx) & (_atom_names(ctx) == "CA")
    )


MACRO_KEYWORDS: dict[str, Callable[[SelectionContext], np.ndarray]] = {
    "all": lambda ctx: np.ones(ctx.n_atoms, dtype=bool),
    "none": lambda ctx: np.zeros(ctx.n_atoms, dtype=bool),
    "protein": _macro_protein,
    "peptide": _macro_protein,
    "polymer": _macro_polymer,
    "nucleic": _macro_nucleic,
    "dna": _macro_nucleic,
    "rna": _macro_nucleic,
    "backbone": _macro_backbone,
    "bb": _macro_backbone,
    "sidechain": _macro_sidechain,
    "sc": _macro_sidechain,
    "water": _macro_solvent,
    "solvent": _macro_solvent,
    "hetatm": _macro_hetero,
    "hetero": _macro_hetero,
    "ligand": _macro_ligand,
    "organic": _macro_ligand,
    "ions": _macro_ions,
    "ion": _macro_ions,
    "metals": _macro_metals,
    "hydro": _macro_hydrogen,
    "hydrogen": _macro_hydrogen,
    "donors": _macro_donors,
    "acceptors": _macro_acceptors,
    "polar": _macro_polar,
    "carbon": _macro_carbon,
    "aromatic": _macro_aromatic,
    "ca": _macro_alpha_carbon,
    "alpha": _macro_alpha_carbon,
}

#: Property keyword -> (annotation name, value kind).
PROPERTY_KEYWORDS: dict[str, tuple[str, str]] = {
    "chain": ("chain_id", "str"),
    "c": ("chain_id", "str"),
    "segi": ("chain_id", "str"),
    "segment": ("chain_id", "str"),
    "resi": ("res_id", "int"),
    "resid": ("res_id", "int"),
    "residue": ("res_id", "int"),
    "i": ("res_id", "int"),
    "resn": ("res_name", "str"),
    "resname": ("res_name", "str"),
    "r": ("res_name", "str"),
    "name": ("atom_name", "str"),
    "n": ("atom_name", "str"),
    "atom": ("atom_name", "str"),
    "elem": ("element", "str"),
    "element": ("element", "str"),
    "e": ("element", "str"),
    "alt": ("altloc_id", "str"),
    "ins": ("ins_code", "str"),
    "ss": ("sec_struct", "int"),
    "b": ("b_factor", "float"),
    "bfactor": ("b_factor", "float"),
    "q": ("occupancy", "float"),
    "occupancy": ("occupancy", "float"),
    "charge": ("charge", "float"),
    "id": ("atom_id", "int"),
    "index": ("__index__", "int"),
    "idx": ("__index__", "int"),
    "rank": ("__rank__", "int"),
}

_UNARY_PREFIXES = {
    "not",
    "byres",
    "byresidue",
    "bychain",
    "byobject",
    "byfrag",
    "byfragment",
    "bymol",
    "bymolecule",
    "first",
    "last",
}
_POSTFIX_OPERATORS = {"within", "around", "expand", "gap", "near_to", "beyond"}


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^(?P<low>-?\d+)?(?P<sep>-|:)(?P<high>-?\d+)?$")


def _split_values(raw: str) -> list[str]:
    parts = [p for chunk in raw.split("+") for p in chunk.split(",")]
    return [p for p in parts if p != ""]


def _parse_int_values(
    raw: str, token: _Token, text: str
) -> tuple[tuple[int, ...], tuple[tuple[int | None, int | None], ...]]:
    singles: list[int] = []
    ranges: list[tuple[int | None, int | None]] = []
    for part in _split_values(raw):
        cleaned = part.replace("\\", "")
        if re.fullmatch(r"-?\d+", cleaned):
            singles.append(int(cleaned))
            continue
        match = _RANGE_RE.match(cleaned)
        if match and (match.group("low") or match.group("high")):
            low = match.group("low")
            high = match.group("high")
            ranges.append(
                (int(low) if low is not None else None, int(high) if high else None)
            )
            continue
        raise SelectionSyntaxError(
            f"expected an integer or range, got {part!r}", text, token.pos
        )
    return tuple(singles), tuple(ranges)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    # -- token helpers ---------------------------------------------------
    def peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> _Token:
        token = self.peek()
        if token is None:
            raise SelectionSyntaxError(
                "unexpected end of selection", self.text, len(self.text)
            )
        self.pos += 1
        return token

    def at_word(self, *words: str) -> bool:
        token = self.peek()
        return (
            token is not None and token.kind == "word" and token.text.lower() in words
        )

    def expect_word(self, word: str) -> _Token:
        token = self.peek()
        if token is None or token.kind != "word" or token.text.lower() != word:
            pos = token.pos if token else len(self.text)
            raise SelectionSyntaxError(f"expected {word!r}", self.text, pos)
        return self.next()

    # -- grammar ---------------------------------------------------------
    def parse(self) -> _Node:
        node = self.parse_or()
        remaining = self.peek()
        if remaining is not None:
            raise SelectionSyntaxError(
                f"unexpected token {remaining.text!r}", self.text, remaining.pos
            )
        return node

    def parse_or(self) -> _Node:
        node = self.parse_and()
        while True:
            token = self.peek()
            if token is None:
                break
            if token.kind == "or" or (
                token.kind == "word" and token.text.lower() in ("or", "||")
            ):
                self.next()
                node = _Or(node, self.parse_and())
            else:
                break
        return node

    def parse_and(self) -> _Node:
        node = self.parse_postfix()
        while True:
            token = self.peek()
            if token is None:
                break
            if token.kind == "and" or (
                token.kind == "word" and token.text.lower() == "and"
            ):
                self.next()
                node = _And(node, self.parse_postfix())
            else:
                break
        return node

    def parse_postfix(self) -> _Node:
        node = self.parse_unary()
        while self.peek() is not None and self.peek().kind == "word":  # type: ignore[union-attr]
            word = self.peek().text.lower()  # type: ignore[union-attr]
            if word not in _POSTFIX_OPERATORS:
                break
            token = self.next()
            cutoff = self._parse_number(token)
            if word in ("within", "near_to", "beyond"):
                self.expect_word("of")
                reference = self.parse_unary()
                if word == "beyond":
                    node = _And(node, _Not(_Within(_Constant(True), reference, cutoff)))
                else:
                    node = _Within(node, reference, cutoff)
            elif word == "around" or word == "gap":
                node = _Within(_Constant(True), node, cutoff, exclude_self=True)
            else:  # expand
                node = _Expand(node, cutoff)
        return node

    def parse_unary(self) -> _Node:
        token = self.peek()
        if token is None:
            raise SelectionSyntaxError(
                "unexpected end of selection", self.text, len(self.text)
            )
        if token.kind == "not":
            self.next()
            return _Not(self.parse_unary())
        if token.kind == "word":
            word = token.text.lower()
            if word in _UNARY_PREFIXES:
                self.next()
                operand = self.parse_unary()
                if word == "not":
                    return _Not(operand)
                if word in ("byres", "byresidue"):
                    return _ByResidue(operand)
                if word in ("bychain", "byobject"):
                    return _ByChain(operand)
                if word in ("byfrag", "byfragment", "bymol", "bymolecule"):
                    return _ByFragment(operand)
                return _Edge(operand, last=(word == "last"))
        return self.parse_primary()

    def parse_primary(self) -> _Node:
        token = self.next()
        if token.kind == "lparen":
            node = self.parse_or()
            closing = self.peek()
            if closing is None or closing.kind != "rparen":
                pos = closing.pos if closing else len(self.text)
                raise SelectionSyntaxError("expected ')'", self.text, pos)
            self.next()
            return node
        if token.kind != "word":
            raise SelectionSyntaxError(
                f"unexpected token {token.text!r}", self.text, token.pos
            )

        word = token.text.lower()
        if word == "*":
            return _Constant(True)
        if word in MACRO_KEYWORDS:
            return _Macro(word)
        if word in PROPERTY_KEYWORDS:
            return self._parse_property(word, token)
        raise SelectionSyntaxError(
            f"unknown selection keyword {token.text!r}", self.text, token.pos
        )

    # -- property selectors ----------------------------------------------
    def _parse_number(self, origin: _Token) -> float:
        token = self.peek()
        if token is None or token.kind != "word":
            pos = token.pos if token else len(self.text)
            raise SelectionSyntaxError(
                f"{origin.text!r} needs a distance in angstrom", self.text, pos
            )
        self.next()
        try:
            return float(token.text)
        except ValueError as exc:
            raise SelectionSyntaxError(
                f"{origin.text!r} needs a number, got {token.text!r}",
                self.text,
                token.pos,
            ) from exc

    def _parse_property(self, word: str, token: _Token) -> _Node:
        annotation, kind = PROPERTY_KEYWORDS[word]

        comparison = self.peek()
        if comparison is not None and comparison.kind == "compare":
            self.next()
            value_token = self.next()
            if value_token.kind != "word":
                raise SelectionSyntaxError(
                    "expected a value after a comparison operator",
                    self.text,
                    value_token.pos,
                )
            operator = comparison.text
            if kind == "str":
                if operator in ("=", "=="):
                    return _StringMatch(
                        annotation,
                        tuple(v.upper() for v in _split_values(value_token.text)),
                    )
                if operator in ("!=", "<>"):
                    return _Not(
                        _StringMatch(
                            annotation,
                            tuple(v.upper() for v in _split_values(value_token.text)),
                        )
                    )
                raise SelectionSyntaxError(
                    f"{word!r} does not support the {operator!r} operator",
                    self.text,
                    comparison.pos,
                )
            try:
                number = float(value_token.text)
            except ValueError as exc:
                raise SelectionSyntaxError(
                    f"expected a number, got {value_token.text!r}",
                    self.text,
                    value_token.pos,
                ) from exc
            return _NumericCompare(annotation, operator, number)

        peeked = self.peek()
        if peeked is None or peeked.kind != "word":
            pos = peeked.pos if peeked else len(self.text)
            raise SelectionSyntaxError(f"{word!r} needs a value", self.text, pos)
        value_token = self.next()

        if kind == "str":
            return _StringMatch(
                annotation, tuple(v.upper() for v in _split_values(value_token.text))
            )
        if kind == "float":
            try:
                number = float(value_token.text)
            except ValueError as exc:
                raise SelectionSyntaxError(
                    f"expected a number, got {value_token.text!r}",
                    self.text,
                    value_token.pos,
                ) from exc
            return _NumericCompare(annotation, "=", number)

        singles, ranges = _parse_int_values(value_token.text, value_token, self.text)
        if annotation in ("__index__", "__rank__"):
            return _IndexRanges(singles, ranges, one_based=(annotation == "__index__"))
        return _IntRanges(annotation, singles, ranges)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Selection:
    """A parsed, reusable atom selection.

    Parsing is done once; the resulting expression tree can be evaluated
    against any structure, which matters when the same selection is applied
    across the frames of a trajectory.

    Parameters
    ----------
    text : str
        A PyMOL-style selection string.

    Attributes
    ----------
    text : str
        The original selection string.
    """

    __slots__ = ("_root", "text")

    def __init__(self, text: str) -> None:
        self.text = text
        self._root = _Parser(text).parse()

    def evaluate(
        self, array: Any, context: SelectionContext | None = None
    ) -> np.ndarray:
        """Evaluate the selection and return a boolean mask.

        Parameters
        ----------
        array : biotite.structure.AtomArray
            Structure to select from.
        context : SelectionContext, optional
            Reuse an existing context to share cached KD-trees between several
            selections over the same structure.

        Returns
        -------
        numpy.ndarray
            Boolean mask of length ``len(array)``.
        """
        ctx = context if context is not None else SelectionContext(array)
        return np.asarray(self._root.evaluate(ctx), dtype=bool)

    def __repr__(self) -> str:
        return f"Selection({self.text!r})"


@lru_cache(maxsize=512)
def compile_selection(text: str) -> Selection:
    """Parse ``text`` into a reusable :class:`Selection`, with caching.

    Parameters
    ----------
    text : str
        A PyMOL-style selection string.

    Returns
    -------
    Selection
        The parsed selection.

    Raises
    ------
    SelectionSyntaxError
        If the string cannot be parsed.
    """
    return Selection(text)


def _coerce(
    selection: str | Selection | np.ndarray | Any,
) -> Selection | np.ndarray:
    if isinstance(selection, Selection):
        return selection
    if isinstance(selection, str):
        return compile_selection(selection)
    mask = np.asarray(selection)
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.integer):
        return mask
    raise TypeError(
        "selection must be a string, Selection, boolean mask or integer index array"
    )


def as_atom_array(target: Any) -> Any:
    """Return the ``AtomArray`` behind whatever the caller passed.

    Accepts a bare ``AtomArray``, an ``AtomArrayStack`` (reduced to its first
    model), or anything carrying one on an ``.array`` attribute — a Molecular
    Nodes ``Molecule`` or a Gala ``AtomStructure``. Without this, passing a
    ``Molecule`` straight to ``select`` would find no annotations and silently
    return an all-false mask.
    """
    array = getattr(target, "array", target)
    coord = getattr(array, "coord", None)
    if coord is not None and np.asarray(coord).ndim == 3:
        array = array[0]
    return array


def select(
    target: Any,
    selection: str | Selection | np.ndarray,
    context: SelectionContext | None = None,
) -> np.ndarray:
    """Return a boolean mask for ``selection`` over ``array``.

    Parameters
    ----------
    target : AtomArray, AtomArrayStack, Molecule, or AtomStructure
        Structure to select from.
    selection : str, Selection, or numpy.ndarray
        A selection string, a pre-compiled selection, a boolean mask, or an
        array of atom indices. Masks and index arrays are passed through, which
        lets callers accept "a selection" without caring which form it took.
    context : SelectionContext, optional
        Shared evaluation cache.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``len(array)``.
    """
    array = as_atom_array(target)
    coerced = _coerce(selection)
    n_atoms = len(array)
    if isinstance(coerced, Selection):
        return coerced.evaluate(array, context)
    if coerced.dtype == bool:
        if coerced.shape != (n_atoms,):
            raise ValueError(
                f"boolean mask has length {coerced.shape[0]}, expected {n_atoms}"
            )
        return coerced
    mask = np.zeros(n_atoms, dtype=bool)
    mask[coerced] = True
    return mask


def select_indices(
    target: Any,
    selection: str | Selection | np.ndarray,
    context: SelectionContext | None = None,
) -> np.ndarray:
    """Return the integer atom indices matched by ``selection``.

    Parameters
    ----------
    target : AtomArray, AtomArrayStack, Molecule, or AtomStructure
        Structure to select from.
    selection : str, Selection, or numpy.ndarray
        See :func:`select`.
    context : SelectionContext, optional
        Shared evaluation cache.

    Returns
    -------
    numpy.ndarray
        Sorted array of 0-based atom indices.
    """
    return np.flatnonzero(select(target, selection, context))


def expand_selection(
    target: Any,
    selection: str | Selection | np.ndarray,
    level: str = "residue",
    context: SelectionContext | None = None,
) -> np.ndarray:
    """Grow a selection to whole residues, chains or bonded fragments.

    This is PyMOL's selection level applied after the fact: pick a few atoms
    in the viewport, then expand to the residues they belong to. Levels
    compose, so expanding a residue-level mask to ``"chain"`` grows it again.

    Parameters
    ----------
    target : AtomArray, AtomArrayStack, Molecule, or AtomStructure
        Structure to select from.
    selection : str, Selection, or numpy.ndarray
        What to expand. See :func:`select`.
    level : {"atom", "residue", "chain", "fragment", "object"}, optional
        ``"atom"`` returns the mask unchanged, ``"fragment"`` grows to the
        connected component of the bond graph, and ``"object"`` to everything.
    context : SelectionContext, optional
        Shared evaluation cache.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``len(array)``.

    Raises
    ------
    ValueError
        If ``level`` is not one of :data:`LEVELS`.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown selection level {level!r}; expected one of {LEVELS}")

    array = as_atom_array(target)
    ctx = context if context is not None else SelectionContext(array)
    mask = select(array, selection, ctx)

    if level == "atom" or not mask.any():
        return mask
    if level == "residue":
        return ctx.expand_to_residues(mask)
    if level == "chain":
        return ctx.expand_to_chains(mask)
    if level == "fragment":
        return ctx.expand_to_fragments(mask)
    return np.ones(ctx.n_atoms, dtype=bool)


def describe_selection(
    target: Any,
    selection: str | Selection | np.ndarray,
    context: SelectionContext | None = None,
) -> str:
    """Render a selection as a PyMOL selection string.

    The inverse of :func:`select`, and the reason a viewport pick can be
    pasted into any other Gala call or into PyMOL itself. The result is
    *verified* before it is returned — it is re-evaluated against the
    structure and only used if it reproduces the mask exactly. Structures
    where the chemical description is ambiguous (repeated residue numbers
    under insertion codes, blank chain identifiers) therefore fall back to the
    positional ``index 3+7-10`` form, which is always exact.

    Parameters
    ----------
    target : AtomArray, AtomArrayStack, Molecule, or AtomStructure
        Structure the selection refers to.
    selection : str, Selection, or numpy.ndarray
        Usually a boolean mask. See :func:`select`.
    context : SelectionContext, optional
        Shared evaluation cache.

    Returns
    -------
    str
        A selection string that evaluates back to the same atoms.

    Examples
    --------
    >>> describe_selection(array, mask)   # doctest: +SKIP
    'chain A and resi 45-47'
    """
    array = as_atom_array(target)
    ctx = context if context is not None else SelectionContext(array)
    mask = select(array, selection, ctx)

    if not mask.any():
        return "none"
    if mask.all():
        return "all"

    text = _describe_by_chain(ctx, mask)
    if text is not None:
        try:
            if np.array_equal(select(array, text, ctx), mask):
                return text
        except SelectionSyntaxError:
            # A residue or atom name carrying a character the language uses as
            # punctuation. Nothing to salvage; the positional form is exact.
            pass
    return _describe_by_index(mask)


def _describe_by_chain(ctx: SelectionContext, mask: np.ndarray) -> str | None:
    """Describe a mask chain by chain, or ``None`` if it cannot be."""
    chains = ctx.upper("chain_id")
    raw_res_ids = ctx.annotation("res_id")
    if raw_res_ids is None:
        return None
    res_ids = np.asarray(raw_res_ids)
    names = ctx.upper("atom_name")
    residue_key = ctx.residue_key

    clauses: list[str] = []
    complete: list[str] = []
    for chain in np.unique(chains[mask]):
        if not chain:
            return None  # a blank chain id cannot be written as `chain X`
        in_chain = chains == chain
        picked = mask & in_chain
        prefix = f"chain {chain}"

        if np.array_equal(picked, in_chain):
            complete.append(str(chain))
            continue

        whole: list[int] = []
        # Partly selected residues are grouped by *which* atoms of them were
        # picked, so that a backbone trace comes out as one
        # `chain A and resi 1-7 and name CA` rather than one clause per
        # residue. Insertion order keeps the output deterministic.
        partial: dict[tuple[str, ...], list[int]] = {}

        for key in np.unique(residue_key[picked]):
            in_residue = residue_key == key
            if not (in_residue & ~picked).any():
                whole.append(int(key))
                continue
            atom_names = tuple(np.unique(names[in_residue & picked]))
            if any(not name for name in atom_names):
                return None  # an unnamed atom cannot be written as `name X`
            partial.setdefault(atom_names, []).append(int(key))

        if whole:
            covered = np.isin(residue_key, whole)
            clauses.append(f"{prefix} and resi {_compact_ints(res_ids[covered])}")

        for atom_names, keys in partial.items():
            numbers = []
            for key in keys:
                residue_number = np.unique(res_ids[residue_key == key])
                if residue_number.size != 1:  # pragma: no cover - residue_key
                    return None  # includes res_id, so this cannot happen
                numbers.append(int(residue_number[0]))
            clauses.append(
                f"{prefix} and resi {_compact_ints(np.array(numbers))} "
                f"and name {'+'.join(atom_names)}"
            )

    # Chains taken whole collapse into one `chain B+C+D` rather than a clause
    # each, and lead, because that is the coarsest statement being made.
    if complete:
        clauses.insert(0, f"chain {'+'.join(complete)}")

    if not clauses:  # pragma: no cover - an empty mask returns earlier
        return None
    if len(clauses) == 1:
        return clauses[0]

    # `and` binds tighter than `or`, so the parentheses are for the reader
    # rather than the parser.
    return " or ".join(
        clause if clause.startswith("(") or " and " not in clause else f"({clause})"
        for clause in clauses
    )


def _describe_by_index(mask: np.ndarray) -> str:
    """Describe a mask positionally. Always exact, never enlightening."""
    return f"index {_compact_ints(np.flatnonzero(mask) + 1)}"


def _compact_ints(values: np.ndarray) -> str:
    """Render integers as a PyMOL value list, collapsing runs into ranges."""
    unique = np.unique(np.asarray(values, dtype=int))
    parts: list[str] = []
    start = previous = int(unique[0])
    for value in unique[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue
        parts.append(_range_text(start, previous))
        start = previous = value
    parts.append(_range_text(start, previous))
    return "+".join(parts)


def _range_text(low: int, high: int) -> str:
    if low == high:
        return str(low)
    if low < 0:
        # `resi -5--3` is ambiguous with PyMOL's negative residue numbers, so
        # a run that starts below zero is written out one number at a time.
        return "+".join(str(value) for value in range(low, high + 1))
    return f"{low}-{high}"
