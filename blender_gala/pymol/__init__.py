"""PyMOL interoperability: reading and writing ``.pse`` sessions.

A session is a pickled tree of plain lists, so Gala reads and writes one
directly rather than shelling out to PyMOL — which matters because Blender's
interpreter has no PyMOL in it.

    >>> import blender_gala as gala                        # doctest: +SKIP
    >>> result = gala.load_session("figure.pse")           # doctest: +SKIP
    >>> print(result.summary())                            # doctest: +SKIP
    >>> gala.save_session("from_blender.pse")              # doctest: +SKIP

:mod:`~blender_gala.pymol.session` is the format layer and needs nothing but
numpy; :mod:`~blender_gala.pymol.load` and :mod:`~blender_gala.pymol.save`
are the halves that build or read a Blender scene.
"""

from __future__ import annotations

from . import load, palette, save, session, view
from .load import LoadedSession, load_session
from .save import SavedSession, save_session, scene_to_session
from .session import (
    REPS,
    PymolMeasurement,
    PymolMolecule,
    PymolSelection,
    PymolSession,
    PymolSessionError,
    PymolView,
    read_session,
    write_session,
)
from .view import camera_to_view, view_to_camera

__all__ = [
    "REPS",
    "LoadedSession",
    "PymolMeasurement",
    "PymolMolecule",
    "PymolSelection",
    "PymolSession",
    "PymolSessionError",
    "PymolView",
    "SavedSession",
    "camera_to_view",
    "load",
    "load_session",
    "palette",
    "read_session",
    "save",
    "save_session",
    "scene_to_session",
    "session",
    "view",
    "view_to_camera",
    "write_session",
]
