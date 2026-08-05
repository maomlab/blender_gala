"""Check that every internal link in the built site resolves to a real file.

A broken internal link is invisible until someone clicks it, and the failure
mode that prompted this check — MkDocs' clean URLs pointing at directories,
which only a web server resolves — looks fine in ``mkdocs serve`` and breaks
the moment the site is opened from disk. So the check runs against the built
output, the way a reader sees it.

    python scripts/check_links.py [site]
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from urllib.parse import unquote, urldefrag

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: href/src values in the built HTML. Deliberately a regex rather than an HTML
#: parser: the built pages are machine-generated and well-formed, and this
#: keeps the check dependency-free.
_LINK_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:", "javascript:", "//")

#: A 404 page is served from any depth, so its links must be absolute paths
#: rooted at the deployed site. Those correctly do not resolve on disk, so
#: checking them would only ever produce noise.
_SKIP_FILES = frozenset({"404.html"})


def internal_targets(html: str) -> list[str]:
    """Return the internal link targets in one page."""
    targets = []
    for raw in _LINK_RE.findall(html):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value.lower().startswith(_EXTERNAL_PREFIXES):
            continue
        target, _ = urldefrag(value)
        if target:
            targets.append(unquote(target))
    return targets


def check(site_dir: str) -> dict[str, list[str]]:
    """Return ``{page: [broken targets]}`` for every page under ``site_dir``."""
    broken: dict[str, list[str]] = defaultdict(list)

    for root, _, files in os.walk(site_dir):
        for name in files:
            if not name.endswith(".html") or name in _SKIP_FILES:
                continue
            page = os.path.join(root, name)
            with open(page, encoding="utf-8", errors="replace") as handle:
                html = handle.read()

            for target in internal_targets(html):
                if target.startswith("/"):
                    resolved = os.path.join(site_dir, target.lstrip("/"))
                else:
                    resolved = os.path.normpath(os.path.join(root, target))

                if os.path.isfile(resolved):
                    continue
                # A directory only resolves under a web server, which is
                # exactly the failure this script exists to catch.
                if os.path.isdir(resolved):
                    broken[page].append(f"{target}  (a directory, not a page)")
                else:
                    broken[page].append(target)

    return broken


def main() -> int:
    site_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "site")
    if not os.path.isdir(site_dir):
        print(f"{site_dir} does not exist; run 'make docs' first.", file=sys.stderr)
        return 2

    broken = check(site_dir)
    pages = sum(
        1
        for _, _, files in os.walk(site_dir)
        for name in files
        if name.endswith(".html") and name not in _SKIP_FILES
    )

    if not broken:
        print(f"All internal links resolve across {pages} pages in {site_dir}.")
        return 0

    total = sum(len(targets) for targets in broken.values())
    print(f"{total} broken internal link(s):", file=sys.stderr)
    for page, targets in sorted(broken.items()):
        print(f"\n  {os.path.relpath(page, site_dir)}", file=sys.stderr)
        for target in sorted(set(targets)):
            print(f"      {target}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
