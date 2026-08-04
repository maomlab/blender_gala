"""Bridge to Molecular Nodes.

Molecular Nodes is a *soft* dependency (SPECIFICATION D-2). It is imported
lazily and through several candidate module paths, because the module name
depends on how it was installed:

``bl_ext.blender_org.molecularnodes``
    Installed from the Blender extensions platform (the usual case).
``bl_ext.user_default.molecularnodes``
    Installed from a local zip file.
``molecularnodes``
    On ``sys.path`` directly, e.g. a development checkout or the PyPI package
    used with the ``bpy`` module.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from .exceptions import MolecularNodesUnavailable

__all__ = [
    "MODULE_CANDIDATES",
    "available",
    "get_mn",
    "is_molecule",
    "molecule_class",
    "require_mn",
    "version",
]

MODULE_CANDIDATES = (
    "bl_ext.blender_org.molecularnodes",
    "bl_ext.user_default.molecularnodes",
    "molecularnodes",
)

_module: ModuleType | None = None
_import_error: str = ""


def get_mn() -> ModuleType | None:
    """Return the Molecular Nodes module, or ``None`` if it is not installed.

    The result is cached after the first successful import.

    Returns
    -------
    ModuleType or None
        The imported ``molecularnodes`` module.
    """
    global _module, _import_error
    if _module is not None:
        return _module

    errors = []
    for name in MODULE_CANDIDATES:
        try:
            _module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on install
            errors.append(f"{name}: {exc}")
        else:
            return _module

    _import_error = "; ".join(errors)
    return None


def require_mn() -> ModuleType:
    """Return the Molecular Nodes module or raise.

    Returns
    -------
    ModuleType
        The imported ``molecularnodes`` module.

    Raises
    ------
    MolecularNodesUnavailable
        If Molecular Nodes cannot be imported.
    """
    module = get_mn()
    if module is None:
        raise MolecularNodesUnavailable(_import_error)
    return module


def available() -> bool:
    """Return ``True`` when Molecular Nodes can be imported."""
    return get_mn() is not None


def version() -> tuple[int, ...] | None:
    """Return the installed Molecular Nodes version as a tuple, if known."""
    module = get_mn()
    if module is None:
        return None
    raw = getattr(module, "__version__", None)
    if raw is None:
        return None
    parts = []
    for chunk in str(raw).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def molecule_class() -> type | None:
    """Return the Molecular Nodes ``Molecule`` class, if importable."""
    module = get_mn()
    if module is None:
        return None
    return getattr(module, "Molecule", None)


def is_molecule(obj: Any) -> bool:
    """Return ``True`` when ``obj`` is a Molecular Nodes ``Molecule`` instance.

    Falls back to duck typing when Molecular Nodes cannot be imported, so that
    test doubles are still recognised.
    """
    cls = molecule_class()
    if cls is not None and isinstance(obj, cls):
        return True
    return hasattr(obj, "array") and hasattr(obj, "object")
