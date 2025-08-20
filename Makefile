# Makefile targets for Sphinx documentation (all targets prefixed with 'docs-')

.PHONY: docs-html docs-clean docs-live docs-env docs-publish \
        docs-html-internal docs-html-ga docs-html-ea docs-html-draft \
        docs-live-internal docs-live-ga docs-live-ea docs-live-draft \
        docs-publish-internal docs-publish-ga docs-publish-ea docs-publish-draft

# Usage:
#   make docs-html DOCS_ENV=internal   # Build docs for internal use
#   make docs-html DOCS_ENV=ga         # Build docs for GA
#   make docs-html                     # Build docs with no special tag
#   make docs-live DOCS_ENV=draft      # Live server with draft tag
#   make docs-publish DOCS_ENV=ga      # Production build (fails on warnings)

DOCS_ENV ?=

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = $(CURDIR)/.venv-docs/Scripts/python.exe
    VENV_ACTIVATE = .venv-docs\Scripts\activate
    VENV_ACTIVATE_PS = .venv-docs\Scripts\Activate.ps1
    RM_CMD = if exist docs\_build rmdir /s /q docs\_build
else
    VENV_PYTHON = $(CURDIR)/.venv-docs/bin/python
    VENV_ACTIVATE = source .venv-docs/bin/activate
    RM_CMD = cd docs && rm -rf _build
endif

# Pass DOCS_ENV to sphinx-build if set

# Makefile targets for Sphinx documentation (all targets prefixed with 'docs-')

.PHONY: docs-html docs-clean docs-live docs-env


docs-html:
	@echo "Building HTML documentation..."
	cd docs && $(VENV_PYTHON) -m sphinx -b html $(if $(DOCS_ENV),-t $(DOCS_ENV)) . _build/html

docs-publish:
	@echo "Building HTML documentation for publication (fail on warnings)..."
	cd docs && $(VENV_PYTHON) -m sphinx --fail-on-warning --builder html $(if $(DOCS_ENV),-t $(DOCS_ENV)) . _build/html

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live:
	@echo "Starting live-reload server (sphinx-autobuild)..."
	cd docs && $(VENV_PYTHON) -m sphinx_autobuild $(if $(DOCS_ENV),-t $(DOCS_ENV)) . _build/html

docs-env:
	@echo "Setting up docs virtual environment with uv..."
	uv venv .venv-docs
	uv pip install -r requirements-docs.txt --python .venv-docs
	@echo "\nTo activate the docs environment, run:"
ifeq ($(OS),Windows_NT)
	@echo "  For Command Prompt: $(VENV_ACTIVATE)"
	@echo "  For PowerShell: $(VENV_ACTIVATE_PS)"
else
	@echo "  $(VENV_ACTIVATE)"
endif

# HTML build shortcuts

docs-html-internal:
	$(MAKE) docs-html DOCS_ENV=internal

docs-html-ga:
	$(MAKE) docs-html DOCS_ENV=ga

docs-html-ea:
	$(MAKE) docs-html DOCS_ENV=ea

docs-html-draft:
	$(MAKE) docs-html DOCS_ENV=draft

# Publish build shortcuts

docs-publish-internal:
	$(MAKE) docs-publish DOCS_ENV=internal

docs-publish-ga:
	$(MAKE) docs-publish DOCS_ENV=ga

docs-publish-ea:
	$(MAKE) docs-publish DOCS_ENV=ea

docs-publish-draft:
	$(MAKE) docs-publish DOCS_ENV=draft

# Live server shortcuts

docs-live-internal:
	$(MAKE) docs-live DOCS_ENV=internal

docs-live-ga:
	$(MAKE) docs-live DOCS_ENV=ga

docs-live-ea:
	$(MAKE) docs-live DOCS_ENV=ea

docs-live-draft:
	$(MAKE) docs-live DOCS_ENV=draft 
