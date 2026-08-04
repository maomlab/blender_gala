"""Run the test suite inside Blender.

Usage::

    blender --background --python tests/run_tests.py -- [pytest args]

Everything after ``--`` is forwarded to pytest, so
``... -- -k selection -x`` works as usual.

Test dependencies live in ``.blender-deps`` (see ``make dev-deps``) rather than
in Blender's own site-packages, so nothing is written into the Blender install
or into Molecular Nodes' wheel directory.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS_DIR = os.path.join(REPO_ROOT, ".blender-deps")


def main() -> int:
    for path in (DEPS_DIR, REPO_ROOT):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    try:
        import pytest
    except ImportError:
        sys.stderr.write(
            "pytest is not available to Blender's interpreter.\n"
            "Run 'make dev-deps' first; it installs pytest into "
            f"{DEPS_DIR} without touching the Blender installation.\n"
        )
        return 2

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not any(not arg.startswith("-") for arg in argv):
        argv.append(os.path.join(REPO_ROOT, "tests"))

    return int(pytest.main(argv))


if __name__ == "__main__":
    # Blender ignores a script's exit code, so exit explicitly to give CI a
    # non-zero status when tests fail.
    sys.exit(main())
