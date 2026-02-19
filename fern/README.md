# NeMo Curator Documentation (Fern)

This directory contains the NeMo Curator documentation built with [Fern](https://buildwithfern.com/).

## Directory Structure

```
fern/
├── fern.config.json     # Fern configuration
├── docs.yml             # Navigation and settings
├── versions/            # Version configurations
├── pages/               # MDX documentation pages
│   ├── index.mdx        # Home page
│   ├── about/           # About section
│   ├── get-started/     # Quickstart guides
│   ├── curate-text/     # Text curation docs
│   ├── curate-images/   # Image curation docs
│   ├── curate-video/    # Video curation docs
│   ├── curate-audio/    # Audio curation docs
│   ├── admin/           # Setup & deployment
│   ├── reference/       # Reference docs
│   └── api-reference/   # API documentation
├── assets/              # Images and static files
├── scripts/             # Build and conversion scripts
└── README.md            # This file
```

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

## Scripts

### Variable Substitution

The `scripts/substitute_variables.py` script replaces template variables with their values:

```bash
python scripts/substitute_variables.py pages
```

Variables are defined in the script and include:
- `{{ container_version }}` - Docker container version
- `{{ product_name }}` - Product name (NeMo Curator)
- `{{ github_repo }}` - GitHub repository URL

### Check Unconverted Syntax

Verify all MyST syntax has been converted to Fern format:

```bash
bash scripts/check_unconverted.sh pages
```

### Convert MyST to Fern

Convert a MyST markdown file to Fern MDX:

```bash
python scripts/convert_myst_to_fern.py input.md > output.mdx
```

## CI/CD

Documentation is automatically built and deployed via GitHub Actions:

- **Pull Requests**: Preview deployments generated
- **Main branch**: Production deployment

Required secrets:
- `FERN_TOKEN`: API token from Fern dashboard

## Migration from Sphinx

This documentation was migrated from Sphinx MyST format. See:
- [RFC-FERN-MIGRATION.md](../docs/RFC-FERN-MIGRATION.md) - Migration RFC
- [scripts/convert_myst_to_fern.py](scripts/convert_myst_to_fern.py) - Conversion script

## Contributing

1. Make changes to MDX files in `pages/`
2. Run `fern check` to validate
3. Test locally with `fern docs dev`
4. Submit PR for review

### Adding New Pages

1. Create MDX file in appropriate directory
2. Add frontmatter with `title` and `description`
3. Add page to `docs.yml` navigation
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
