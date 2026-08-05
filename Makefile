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

# Read by hand rather than with tomllib: this runs on every invocation of make,
# including `make help`, and tomllib needs Python 3.11. Reading it with an
# interpreter that turns out to be older printed a traceback before every
# target and left the version blank.
VERSION := $(shell sed -n 's/^version *= *"\(.*\)"/\1/p' $(PACKAGE)/blender_manifest.toml)

# The toolchain that runs outside Blender: ruff, mypy, mkdocs, and the helper
# scripts. Needs 3.11 or newer, per `requires-python`. Override it when the
# default `python3` is older or is a conda base you would rather not install
# into:
#
#     make docs PYTHON=python3.13
PYTHON ?= python3

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

.PHONY: dev-deps
dev-deps: check-blender ## Install pytest into .blender-deps for Blender's Python
	$(BLENDER_RUN) --python scripts/install_deps.py

.PHONY: dev
dev: check-python ## Install the linting and docs toolchain into PYTHON
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
	  $(BLENDER) --background --python "$$script" || exit 1; \
	done

.PHONY: ui-shots
ui-shots: check-blender ## Recapture the sidebar screenshots in docs/images/ui
	BLENDER="$(BLENDER)" $(PYTHON) scripts/capture_ui.py

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

# docs/images/passes holds the multilayer EXRs vignette 5 writes beside the
# figures. Generated, gitignored, and several megabytes, so it goes with the
# rest of the build output rather than lingering as untracked files.
.PHONY: clean
clean: ## Remove build output and caches
	rm -rf $(DIST) site htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache \
	  docs/images/passes
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean ## Also remove the Blender test dependencies
	rm -rf $(DEPS_DIR)
