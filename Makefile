# Blender Gala — development tasks.
#
# Most targets run inside Blender's own interpreter, because that is the only
# place `bpy` and Molecular Nodes exist. Set BLENDER if yours is elsewhere:
#
#     make test BLENDER=/opt/blender/blender

SHELL := /bin/bash
.DEFAULT_GOAL := help

PACKAGE   := blender_gala
DIST      := dist
DEPS_DIR  := .blender-deps
TURNTABLE_DIR := build/turntable

# Read by hand rather than with tomllib: this runs on every invocation of make,
# including `make help`, and tomllib needs Python 3.11. Reading it with an
# interpreter that turns out to be older printed a traceback before every
# target and left the version blank.
VERSION := $(shell sed -n 's/^version *= *"\(.*\)"/\1/p' $(PACKAGE)/blender_manifest.toml)

VENV        := .venv
VENV_PYTHON := $(VENV)/bin/python

# The toolchain that runs outside Blender: ruff, mypy, mkdocs, and the helper
# scripts. `make venv` puts it in $(VENV), and once that exists every target
# finds it without being told. Otherwise it is whatever `python3` is, which
# still works if that interpreter has the toolchain. Override for a one-off:
#
#     make docs PYTHON=python3.13
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)

# Commands that are run by name rather than as modules — ruff, mypy — should
# come from the same place. Prepending is enough: with no venv this changes
# nothing, and it does not disturb a contributor whose ruff is a standalone
# binary rather than a Python package.
ifneq ($(wildcard $(VENV)/bin),)
  export PATH := $(abspath $(VENV)/bin):$(PATH)
endif

# The interpreter that *creates* the venv has to be 3.11 or newer, and the
# default `python3` often is not. Prefer it when it qualifies, so the venv
# matches the interpreter the rest of the machine uses.
BASE_PYTHON ?= $(shell for p in python3 python3.13 python3.12 python3.11; do \
	  command -v $$p >/dev/null 2>&1 \
	    && $$p -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
	    && { echo $$p; break; }; \
	done)

# Locate Blender: an explicit BLENDER wins, then PATH, then the usual
# per-platform install locations.
BLENDER ?= $(shell command -v blender 2>/dev/null)
ifeq ($(BLENDER),)
  BLENDER := $(firstword $(wildcard \
    /Applications/Blender.app/Contents/MacOS/Blender \
    /usr/bin/blender \
    /usr/local/bin/blender \
    $(HOME)/blender/blender))
endif

BLENDER_RUN := $(BLENDER) --background --factory-startup
PYTEST_ARGS ?=

.PHONY: help
help: ## Show this help
	@echo "Blender Gala $(VERSION)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Blender: $(if $(BLENDER),$(BLENDER),NOT FOUND — set BLENDER=/path/to/blender)"
	@echo "Python : $(PYTHON) ($(shell $(PYTHON) -V 2>&1 || echo NOT FOUND))"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.PHONY: check-blender
check-blender:
	@if [ -z "$(BLENDER)" ]; then \
	  echo "Blender not found. Set BLENDER=/path/to/blender"; exit 1; \
	fi

.PHONY: check-python
check-python:
	@$(PYTHON) -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>/dev/null || { \
	  echo "$(PYTHON) is $$($(PYTHON) -V 2>&1); this project needs 3.11 or newer."; \
	  echo "Point PYTHON at a newer one, e.g. PYTHON=python3.13"; exit 1; \
	}

# Homebrew and system Pythons are usually "externally managed" and refuse a
# plain pip install, and a conda base is rarely where you want a project's
# toolchain either. A venv sidesteps both, and needs no tooling beyond the
# stdlib. Safe to re-run: that is how the toolchain gets updated.
.PHONY: venv
venv: ## Create .venv and install the toolchain into it (recommended)
	@if [ -z "$(BASE_PYTHON)" ]; then \
	  echo "No Python 3.11 or newer found on PATH."; \
	  echo "Install one, or set BASE_PYTHON=/path/to/python3.13"; exit 1; \
	fi
	@echo "Creating $(VENV) with $(BASE_PYTHON) ($$($(BASE_PYTHON) -V 2>&1))"
	$(BASE_PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --quiet --upgrade pip
	$(VENV_PYTHON) -m pip install --upgrade -e ".[dev,docs]"
	@echo
	@echo "Done. make docs, lint, typecheck and ui-shots will now use $(VENV_PYTHON)."

# The wheel drops the APBS binary inside the package rather than in bin/, and
# it needs its own lib/ on the loader path, so this writes a wrapper instead of
# symlinking. `GALA_APBS` would do as well; a wrapper means `apbs` also works
# from an activated shell.
.PHONY: apbs
apbs: ## Install APBS and PDB2PQR into .venv, for the electrostatics vignette
	$(VENV_PYTHON) -m pip install --upgrade -e ".[apbs]"
	@bin=$$($(VENV_PYTHON) -c "import apbs_binary, os; print(os.path.dirname(apbs_binary.__file__))"); \
	  printf '#!/bin/sh\nexport DYLD_LIBRARY_PATH="%s/lib:$$DYLD_LIBRARY_PATH"\nexport LD_LIBRARY_PATH="%s/lib:$$LD_LIBRARY_PATH"\nexec "%s/bin/apbs" "$$@"\n' "$$bin" "$$bin" "$$bin" > $(VENV)/bin/apbs; \
	  chmod +x $(VENV)/bin/apbs
	@echo "  apbs   -> $$($(VENV)/bin/apbs --version 2>&1 | head -1)"
	@echo "  pdb2pqr installed; make vignettes will find both."

.PHONY: dev-deps
dev-deps: check-blender ## Install pytest into .blender-deps for Blender's Python
	$(BLENDER_RUN) --python scripts/install_deps.py

.PHONY: dev
dev: check-python ## Install the toolchain into PYTHON instead of a venv
	$(PYTHON) -m pip install --upgrade -e ".[dev,docs]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Check formatting and lint rules
	ruff check $(PACKAGE) tests vignettes scripts
	ruff format --check $(PACKAGE) tests vignettes scripts

.PHONY: format
format: ## Reformat and autofix
	ruff format $(PACKAGE) tests vignettes scripts
	ruff check --fix $(PACKAGE) tests vignettes scripts

.PHONY: typecheck
typecheck: ## Static type check
	mypy $(PACKAGE)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test: check-blender ## Run the test suite inside Blender
	$(BLENDER) --background --python tests/run_tests.py -- $(PYTEST_ARGS)

.PHONY: test-fast
test-fast: check-blender ## Run only the tests that need no Blender objects
	$(BLENDER) --background --python tests/run_tests.py -- -m "not bpy" $(PYTEST_ARGS)

.PHONY: coverage
coverage: check-blender ## Run the suite with a coverage report
	$(BLENDER) --background --python tests/run_tests.py -- \
	  --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html $(PYTEST_ARGS)

.PHONY: fixtures
fixtures: ## Regenerate the synthetic test structures
	$(PYTHON) tests/data/make_fixtures.py

.PHONY: check
check: lint typecheck test ## Everything CI runs

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

# Run mkdocs as a module of $(PYTHON) rather than as whatever `mkdocs` is on
# PATH, so it is the same interpreter `make dev` installed it into.
.PHONY: check-mkdocs
check-mkdocs:
	@$(PYTHON) -c "import mkdocs" 2>/dev/null || { \
	  echo "mkdocs is not installed for $(PYTHON). Run: make dev"; \
	  echo "(or point PYTHON at the interpreter that has it)"; exit 1; \
	}

.PHONY: docs
docs: check-mkdocs ## Build the documentation site into site/ and verify its links
	$(PYTHON) -m mkdocs build --strict
	$(PYTHON) scripts/check_links.py

.PHONY: docs-links
docs-links: ## Check that every internal link in site/ resolves
	$(PYTHON) scripts/check_links.py

.PHONY: docs-serve
docs-serve: check-mkdocs ## Serve the documentation with live reload
	$(PYTHON) -m mkdocs serve

# Only the numbered scripts: `_common.py` is the shared helper module, and
# running it as a vignette renders nothing and proves nothing.
.PHONY: vignettes
vignettes: check-blender ## Run every vignette and render its images
	@for script in vignettes/[0-9]*.py; do \
	  echo "=== $$script"; \
	  PATH="$(CURDIR)/$(VENV)/bin:$$PATH" $(BLENDER) --background --python "$$script" \
	    || exit 1; \
	done

# Renders the orbit frame by frame and assembles them. Kept out of
# `vignettes` because it is a hundred-odd Cycles frames, which is not what a
# smoke test on every push is for.
.PHONY: turntable
turntable: check-blender ## Render the turntable animation into docs/images
	@rm -rf $(TURNTABLE_DIR) && mkdir -p $(TURNTABLE_DIR)
	GALA_TURNTABLE_DIR=$(TURNTABLE_DIR) $(BLENDER) --background \
	  --python vignettes/06_turntable.py
	$(PYTHON) scripts/make_animation.py $(TURNTABLE_DIR) \
	  docs/images/06_turntable.webp
	@echo "  frames kept in $(TURNTABLE_DIR) for re-encoding; make clean removes them"

.PHONY: ui-shots
ui-shots: check-blender ## Recapture the sidebar screenshots in docs/images/ui
	BLENDER="$(BLENDER)" $(PYTHON) scripts/capture_ui.py

# Opens a window for the same reason `ui-shots` does. The highlight graph reads
# its matte out of the EXR vignette 5 writes, so run `make vignettes` first for
# a shot with a real material picked in it.
.PHONY: compositor-shots
compositor-shots: check-blender ## Recapture the node graphs in docs/images/compositor
	BLENDER="$(BLENDER)" $(PYTHON) scripts/capture_compositor.py

# Opens a Blender window, like `ui-shots`, and for the same reason: a
# screenshot is a picture of the framebuffer, which `--background` never fills.
.PHONY: window-shot
window-shot: check-blender ## Recapture the whole-window shot the hero is built from
	BLENDER="$(BLENDER)" $(PYTHON) scripts/capture_window.py

# Composes what `window-shot` and `vignettes` produced; renders nothing itself,
# so it is cheap to rerun after editing the captions.
.PHONY: hero
hero: ## Compose the front page hero into docs/images/hero.png
	$(PYTHON) scripts/make_hero.py

# The extensions platform renders previews as 16:9; every figure here is
# square or nearly so, so they are fitted onto that canvas rather than left
# for the platform to letterbox however it likes.
.PHONY: listing
listing: ## Compose the extensions.blender.org preview images
	$(PYTHON) scripts/make_listing_images.py

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

.PHONY: build
build: check-blender ## Build the installable extension zip into dist/
	@mkdir -p $(DIST)
	$(BLENDER) --command extension build \
	  --source-dir $(PACKAGE) --output-dir $(DIST)
	@ls -lh $(DIST)

.PHONY: validate
validate: check-blender ## Validate the extension manifest
	$(BLENDER) --command extension validate $(PACKAGE)

.PHONY: install
install: build ## Install the built extension into your Blender
	$(BLENDER_RUN) --python-expr \
	  "import bpy,glob; bpy.ops.extensions.package_install_files( \
	    filepath=sorted(glob.glob('$(DIST)/*.zip'))[-1], repo='user_default', \
	    enable_on_install=True)"

# build/ holds the turntable frames and the .blend scenes the vignettes save;
# docs/images/passes holds the multilayer EXRs vignette 5 writes beside the
# figures. All generated, all gitignored, and tens of megabytes between them,
# so they go with the rest of the build output rather than lingering.
.PHONY: clean
clean: ## Remove build output and caches
	rm -rf $(DIST) build site htmlcov .coverage .pytest_cache .mypy_cache \
	  .ruff_cache docs/images/passes
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean ## Also remove the venv and the Blender test dependencies
	rm -rf $(DEPS_DIR) $(VENV)
