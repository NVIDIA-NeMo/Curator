# NeMo Curator Documentation (Fern)

This directory contains the NeMo Curator documentation built with [Fern](https://buildwithfern.com/).

## Directory Structure

```
fern/
├── fern.config.json     # Fern configuration
├── docs.yml             # Site config and version list
├── versions/            # Version-specific navigation + page trees
│   ├── v25.09.yml
│   ├── v26.02.yml
│   ├── latest.yml       # Symlink to v26.02.yml (same nav; /latest/ URLs in Fern)
│   ├── v25.09/pages/
│   └── v26.02/pages/
├── substitute_variables.py  # CI: substitute {{ variables }} in MDX before generate
├── assets/              # Images and static files
└── README.md            # This file
```

### Versions and URLs

`docs.yml` defines multiple versions. Each **`display-name`** pairs the **NeMo calendar train** (e.g. `26.02`) with the **git release tag** (e.g. `v1.1.0`) so the version picker matches PyPI/GitHub. Align these with `CHANGELOG.md` and `nemo_curator/package_info.py` when you ship.

- **Latest** — same nav as 26.02 (`versions/latest.yml` → `v26.02.yml`); URLs under **`/latest/...`**
- **26.02** — current train; default ordering depends on list position in `docs.yml`
- **25.09** — prior train under **`/v25.09/...`**

When you ship a new release, update MDX under `versions/vXX.YY/pages/`, add a `vXX.YY.yml`, repoint `latest.yml` → `vXX.YY.yml`, and refresh the **calendar + `vX.Y.Z` strings** in each version’s `display-name` in `docs.yml`.

## Local Development

### Prerequisites

- Node.js 20+
- Python 3.10+
- npm (for Fern CLI)

### Setup

```bash
# Install Fern CLI
npm install -g fern-api

# Navigate to fern directory
cd fern

# Validate configuration
fern check

# Start local development server
fern docs dev
```

### Build

```bash
# Generate static documentation
fern generate --docs
```

## Variable substitution (CI)

Before `fern generate`, CI runs `substitute_variables.py` so `{{ container_version }}` and other tokens in MDX are replaced. To run locally (from repo root):

```bash
python fern/substitute_variables.py versions/v25.09 --version 25.09
python fern/substitute_variables.py versions/v26.02 --version 26.02
```

Variables are defined in `substitute_variables.py` (e.g. `{{ product_name }}`, `{{ github_repo }}`).

## CI/CD

Documentation is managed via three GitHub Actions workflows:

| Workflow | Trigger | Purpose |
|---|---|---|
| `fern-docs-ci.yml` | PR touching `fern/**` | Validates autodocs generation |
| `fern-docs-preview.yml` | PR touching `fern/**` | Builds a preview site and posts a :herb: comment on the PR |
| `publish-fern-docs.yml` | `docs/v*` tag push or manual dispatch | Publishes to production at [docs.nvidia.com/nemo/curator](https://docs.nvidia.com/nemo/curator) |

### Previewing

Every PR that touches `fern/**` automatically gets a Fern preview URL posted as a PR comment. The preview updates on each push to the same branch (stable URL per branch name). No action needed — just open a PR.

### Publishing to Production

Push a docs tag:

```bash
git tag docs/v1.1.0
git push origin docs/v1.1.0
```

Or trigger manually from the **Actions** tab → **Publish Fern Docs** → **Run workflow**.

### Required Secrets

- `DOCS_FERN_TOKEN`: Fern API token (org-level secret in the nemo org)

## Migration from Sphinx

This documentation was migrated from Sphinx MyST format. See [RFC-FERN-MIGRATION.md](../docs/RFC-FERN-MIGRATION.md) for historical notes. One-off conversion tooling lives outside this repo.

## Contributing

1. Make changes to MDX files in `versions/v26.02/pages/` (latest version)
2. Run `fern check` to validate
3. Test locally with `fern docs dev`
4. Submit PR — a preview URL will be posted automatically as a PR comment

### Adding New Pages

1. Create MDX file in `versions/v26.02/pages/` (or appropriate version)
2. Add frontmatter with `title` and `description`
3. Add page to `versions/v26.02.yml` navigation
4. Run `fern check` to validate

### Frontmatter Format

```yaml
---
title: Page Title
description: "Brief description for SEO"
---
```

## Resources

- [Fern Documentation](https://buildwithfern.com/learn/docs/getting-started/overview)
- [Fern Components](https://buildwithfern.com/learn/docs/writing-content/components/overview)
- [NeMo Curator Source](https://github.com/NVIDIA-NeMo/Curator)
