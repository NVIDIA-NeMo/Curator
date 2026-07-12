# Makefile targets for NeMo Curator's Fern documentation.
# All targets are prefixed with `docs-` (the bare `docs` target boots the dev server).
#
# Usage:
#   make docs                          # local dev server at http://localhost:3000
#   make docs-check                    # validate Fern config + MDX
#   make docs-login                    # guided dashboard sign-in + fern login
#   make docs-generate-library         # regenerate the Python API reference (libraries:)
#   make docs-substitute               # substitute {{ variables }} in MDX (CI runs this automatically)
#   make docs-substitute DOCS_VERSION=26.02
#   make docs-preview                  # build a shared preview URL (needs DOCS_FERN_TOKEN)
#   make docs-publish                  # trigger the Publish Fern Docs workflow on origin/main
#   make docs-clean                    # remove generated product-docs/
#
# See fern/README.md for the full authoring guide.

.PHONY: docs docs-check docs-login docs-generate-library docs-substitute \
        docs-preview docs-publish docs-clean docs-help

# Override FERN to pin a CLI version, e.g. `FERN=npx -y fern-api@4.43.1 make docs-check`.
FERN ?= npx -y fern-api@latest
FERN_DIR := fern

# The bleeding-edge train. Override on the command line to substitute a different snapshot.
DOCS_VERSION ?= 26.04

docs:
	@echo "Starting Fern dev server (http://localhost:3000)..."
	cd $(FERN_DIR) && $(FERN) docs dev

docs-check:
	@echo "Validating Fern config and MDX..."
	cd $(FERN_DIR) && $(FERN) check

docs-login:
	@echo "Step 1/2: open https://dashboard.buildwithfern.com and sign in to the NVIDIA org."
	@echo "          (Skipping this makes 'fern docs md generate' return HTTP 403.)"
	@echo "Step 2/2: authenticating the Fern CLI..."
	$(FERN) login

docs-generate-library:
	@echo "Generating Python API reference under $(FERN_DIR)/product-docs/ (requires FERN_TOKEN)..."
	cd $(FERN_DIR) && $(FERN) docs md generate

docs-substitute:
	@echo "Substituting {{ variables }} in versions/v$(DOCS_VERSION) MDX..."
	python $(FERN_DIR)/substitute_variables.py versions/v$(DOCS_VERSION) --version $(DOCS_VERSION)

docs-preview:
	@if [ -z "$$DOCS_FERN_TOKEN" ]; then \
	    echo "DOCS_FERN_TOKEN is not set. Issue a token via 'fern token' on a privileged NVIDIA Fern dashboard account."; \
	    exit 1; \
	fi
	@echo "Building a shared Fern preview URL..."
	cd $(FERN_DIR) && FERN_TOKEN=$$DOCS_FERN_TOKEN $(FERN) generate --docs --preview

docs-publish:
	@echo "Triggering the 'Publish Fern Docs' workflow on origin/main..."
	gh workflow run publish-fern-docs.yml --ref main

docs-clean:
	@echo "Removing generated product-docs/..."
	rm -rf $(FERN_DIR)/product-docs

docs-help:
	@echo "NeMo Curator Fern docs — make targets:"
	@echo "  docs                    local dev server (http://localhost:3000)"
	@echo "  docs-check              validate Fern config + MDX"
	@echo "  docs-login              guided dashboard sign-in + fern login"
	@echo "  docs-generate-library   regenerate Python API reference (libraries:)"
	@echo "  docs-substitute         run substitute_variables.py on a version tree"
	@echo "                          (override version: make docs-substitute DOCS_VERSION=26.02)"
	@echo "  docs-preview            build a shared preview URL (needs DOCS_FERN_TOKEN)"
	@echo "  docs-publish            trigger the Publish Fern Docs workflow on origin/main"
	@echo "  docs-clean              remove generated product-docs/"
