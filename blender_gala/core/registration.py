"""Tolerant registration of Blender classes.

Blender's ``register_class``/``unregister_class`` assume one copy of an add-on
and a clean pairing of calls. Neither holds in practice:

* A developer often has the extension installed *and* a checkout on
  ``sys.path``. Registering the second copy makes Blender unregister the
  first's classes behind its back — "has been registered before, unregistering
  previous" — so the first copy's ``unregister`` then raises
  ``missing bl_rna attribute ... (may not be registered)`` and Blender reports
  a broken add-on at shutdown.
* If registration fails part-way, the classes that did register are left
  behind, and the next attempt fails on them.

So Gala registers defensively: drop any stale same-named class first, and
never raise while tearing down.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from typing import Any

try:  # pragma: no cover
    import bpy
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore[assignment,misc]

__all__ = ["is_registered", "register_classes", "unregister_classes"]


def _require_bpy() -> Any:
    if bpy is None:  # pragma: no cover
        raise RuntimeError("this function must be called from inside Blender")
    return bpy


def is_registered(cls: type) -> bool:
    """Return whether ``cls`` is currently registered with Blender.

    A registered class has an ``bl_rna`` attribute of its own; an unregistered
    one inherits nothing useful, which is what ``unregister_class`` complains
    about.
    """
    return getattr(cls, "bl_rna", None) is not None


def register_classes(classes: Sequence[type]) -> list[type]:
    """Register ``classes``, replacing any stale registration first.

    Parameters
    ----------
    classes : sequence of type
        Blender classes, in dependency order.

    Returns
    -------
    list[type]
        The classes that are registered afterwards.
    """
    bpy_mod = _require_bpy()

    registered: list[type] = []
    for cls in classes:
        if is_registered(cls):
            # A previous registration of this exact class object is still
            # live: re-registering would leave a duplicate behind.
            with contextlib.suppress(RuntimeError, ValueError):
                bpy_mod.utils.unregister_class(cls)
        bpy_mod.utils.register_class(cls)
        registered.append(cls)
    return registered


def unregister_classes(classes: Iterable[type]) -> int:
    """Unregister ``classes`` in reverse order, skipping anything already gone.

    Parameters
    ----------
    classes : iterable of type
        Blender classes, in registration order.

    Returns
    -------
    int
        How many classes were actually unregistered.
    """
    bpy_mod = _require_bpy()

    removed = 0
    for cls in reversed(list(classes)):
        if not is_registered(cls):
            # Another copy of the add-on displaced it, or registration never
            # got this far. Either way there is nothing to undo.
            continue
        try:
            bpy_mod.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            continue
        removed += 1
    return removed
