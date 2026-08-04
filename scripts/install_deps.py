"""Install test dependencies for Blender's bundled interpreter.

Run through Blender so the packages match its Python version::

    blender --background --python scripts/install_deps.py

Packages go into ``.blender-deps`` at the repository root, which
``tests/run_tests.py`` prepends to ``sys.path``. Nothing is written into the
Blender installation or into Molecular Nodes' wheel directory, so this cannot
break an existing setup and is undone by deleting one folder.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS_DIR = os.path.join(REPO_ROOT, ".blender-deps")

PACKAGES = ["pytest>=8.0", "pytest-cov>=5.0"]


def python_executable() -> str:
    """Locate Blender's Python binary.

    ``sys.executable`` is the Blender binary, not the interpreter, so the
    binary has to be found under ``sys.exec_prefix``.
    """
    patterns = (
        os.path.join(sys.exec_prefix, "bin", "python*"),
        os.path.join(sys.exec_prefix, "Scripts", "python.exe"),
        os.path.join(sys.exec_prefix, "python.exe"),
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    raise RuntimeError(
        f"could not find Blender's Python interpreter under {sys.exec_prefix}"
    )


def main() -> int:
    executable = python_executable()
    os.makedirs(DEPS_DIR, exist_ok=True)

    print(f"Installing into {DEPS_DIR} using {executable}")
    command = [
        executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        DEPS_DIR,
        *PACKAGES,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("pip failed; see the output above.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
