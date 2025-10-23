# LLM.txt Output Extension

Sphinx extension to generate `llm.txt` files for every documentation page in a standardized format optimized for Large Language Model consumption.

## What is llm.txt?

The `llm.txt` format is a proposed standard designed to help LLMs like ChatGPT and Claude better understand and index website content. Each file contains:

- Document title and summary
- Clean overview text
- Key sections with descriptions
- Related resources/links
- Metadata

This extension generates:

1. **Individual `.llm.txt` files**: One file for each documentation page (e.g., `index.llm.txt`, `getting-started/quickstart.llm.txt`)
2. **Aggregated `llm-full.txt`**: A single file containing all documentation with a table of contents for complete site indexing

## Features

- **Simple text format**: Plain markdown files that LLMs can easily parse
- **Clean content extraction**: Removes MyST directive artifacts, toctrees, and navigation elements
- **Structured sections**: Organized with headings, summaries, and key sections
- **Related links**: Automatically extracts internal links for context
- **Metadata support**: Includes frontmatter metadata and document classification
- **Content gating integration**: Respects content gating rules from the content_gating extension
- **Configurable**: Control content length, sections included, and more

## Installation

The extension is included in the `_extensions` directory. Enable it in your `conf.py`:

```python
extensions = [
    # ... other extensions
    '_extensions.llm_txt_output',
]
```

## Configuration

### Minimal Configuration (Recommended)

```python
# conf.py
llm_txt_settings = {
    'enabled': True,  # All other settings use defaults
}
```

### Full Configuration Options

```python
# conf.py
llm_txt_settings = {
    'enabled': True,                    # Enable/disable generation
    'exclude_patterns': [               # Patterns to exclude
        '_build',
        '_templates',
        '_static',
        'apidocs'
    ],
    'verbose': True,                    # Verbose logging
    'base_url': 'https://docs.example.com/latest',  # Base URL for absolute links
    'max_content_length': 5000,         # Max chars in overview (0 = no limit)
    'summary_sentences': 2,             # Number of sentences for summary
    'include_metadata': True,           # Include metadata section
    'include_headings': True,           # Include key sections from headings
    'include_related_links': True,      # Include internal links
    'max_related_links': 10,            # Max related links to include
    'card_handling': 'simple',          # 'simple' or 'smart' for grid cards
    'clean_myst_artifacts': True,       # Remove MyST directive artifacts
    'generate_full_file': True,         # Generate llm-full.txt with all docs
}
```

### Important: Configure base_url for Proper Link Handling

To comply with the [llms.txt specification](https://llmstxt.org/), you **must** set the `base_url` option to your documentation's canonical URL. This ensures all internal links are converted to absolute URLs that LLMs can access:

```python
llm_txt_settings = {
    'base_url': 'https://docs.nvidia.com/nemo/evaluator/latest',
}
```

Without `base_url`, links will remain relative (e.g., `../quickstart.html`) instead of absolute URLs (e.g., `https://docs.nvidia.com/nemo/evaluator/latest/quickstart.html`).

## Output Files

The extension generates two types of output:

### Individual Page Files

For a file `docs/get-started/install.md`, the extension generates `_build/html/get-started/install.llm.txt`

### Aggregated Full Site File

The extension also generates `_build/html/llm-full.txt` containing all documentation in a single file with:

- **Header**: Project name and version
- **Table of Contents**: Complete list of all documentation pages with titles
- **Document Separators**: Clear HTML comment markers between documents (e.g., `<!-- Document 1/103: index -->`)
- **Sorted Order**: Documents alphabetically sorted by path for consistency

This aggregated file is ideal for:

- Training LLMs on your complete documentation
- Feeding entire documentation into context windows
- Creating embeddings of your documentation corpus
- Providing a single download for offline LLM consumption

To disable generation of the full file:

```python
llm_txt_settings = {
    'generate_full_file': False,
}
```

## Output Format

For a file `docs/get-started/install.md`, the extension generates `_build/html/get-started/install.llm.txt`:

```markdown
# Installation Guide

> Learn how to install NeMo Evaluator on your system using pip, Docker, or from source.

## Overview

NeMo Evaluator is a Python package for evaluating large language models. You can install it using pip for a quick setup, or use Docker for a containerized environment...

## Key Sections

- **Prerequisites**: Required software and dependencies before installation
- **Installation Methods**: Different ways to install the package
- **Verify Installation**: Steps to confirm successful installation
- **Troubleshooting**: Common installation issues and solutions

## Related Resources

- [Quickstart Guide](https://docs.nvidia.com/nemo/evaluator/latest/quickstart.html)
- [Configuration Guide](https://docs.nvidia.com/nemo/evaluator/latest/configuration.html)

## Metadata

- Document Type: guide
- Categories: getting-started
- Last Updated: 2025-10-02
```

**Note**: Links are converted to absolute URLs when `base_url` is configured, ensuring compatibility with the [llms.txt specification](https://llmstxt.org/).

## MyST Markdown Handling

The extension intelligently handles MyST markdown directives:

### What Gets Cleaned

- **Toctrees**: Hidden navigation removed
- **Directive markers**: `:::`, `:::{grid}`, etc. removed
- **Directive options**: `:hidden:`, `:caption:`, `:link:`, etc. removed
- **Icons**: `{octicon}` references removed
- **HTML tags**: `<br />`, `<div>`, `<hr>` removed
- **Escaped characters**: `\\` backslashes cleaned
- **Code fences**: Language indicators cleaned

### What Gets Preserved and Enhanced

- **Grid Cards** (when `card_handling: 'smart'`): Converted to clean markdown lists with proper links
- **Badges**: Converted to parentheses (e.g., `{bdg-secondary}`cli`` → `(cli)`)
- **Headings**: All heading structure maintained
- **Links**: Internal and external links extracted and converted to absolute URLs

### Smart Card Handling

When `card_handling` is set to `'smart'`, the extension parses MyST markdown source files directly to extract `{grid-item-card}` directives before Sphinx processes them. This approach is more reliable than trying to parse the complex doctree structure.

**Example MyST Input:**
```markdown
:::{grid-item-card} {octicon}`rocket;1.5em` NeMo Evaluator Launcher
:link: nemo-evaluator-launcher/index
:link-type: doc

**Start here** - Unified CLI and Python API for running evaluations
+++
{bdg-secondary}`CLI`
:::
```

**Generated llm.txt Output:**
```markdown
## Available Options

- **[NeMo Evaluator Launcher](https://docs.nvidia.com/nemo/evaluator/latest/nemo-evaluator-launcher/index.html)**
  Start here - Unified CLI and Python API for running evaluations
```

**What Gets Cleaned:**
- Octicon references: `{octicon}`icon`` removed from titles
- Badge syntax: `{bdg-secondary}`cli`` removed from descriptions
- Card footers: `+++` separator and everything after it
- Template variables: `{{ product_name_short }}` replaced with "NeMo Evaluator"
- Directive options: `:link:`, `:link-type:`, etc. removed

**How It Works:**
1. Reads the raw `.md` source file
2. Uses regex to find `:::{grid-item-card}...:::` blocks
3. Extracts title (first line), link (from `:link:` option), and description (card body)
4. Converts internal links to absolute URLs using `base_url`
5. Keeps external links (GitHub, etc.) unchanged
6. Formats as clean markdown list with proper links

## Content Gating Integration

This extension automatically respects content gating rules:

- Documents excluded by the `content_gating` extension are not processed
- Respects Sphinx's `exclude_patterns` configuration
- Provides debug logging when content gating rules are applied

## Example Usage

### Build Documentation with llm.txt Files

```bash
# Build HTML documentation (llm.txt files generated automatically)
make html

# Check generated files
ls _build/html/*.llm.txt
ls _build/html/**/*.llm.txt
```

### Access llm.txt Files

After building:

- Root page: `_build/html/index.llm.txt`
- Regular pages: `_build/html/path/to/page.llm.txt`
- Directory indexes: `_build/html/directory/index.llm.txt`

## Troubleshooting

### No llm.txt files generated

Check that the extension is enabled:

```python
llm_txt_settings = {'enabled': True}
```

### Files missing content

Increase content length limit:

```python
llm_txt_settings = {'max_content_length': 10000}
```

### Too many MyST artifacts

Enable artifact cleaning:

```python
llm_txt_settings = {'clean_myst_artifacts': True}
```

### Content gated documents included

The extension respects `exclude_patterns`. Add patterns to exclude:

```python
llm_txt_settings = {
    'exclude_patterns': ['_build', '_templates', '_static', 'apidocs', 'internal/*']
}
```

## Dependencies

- Sphinx >= 4.0
- docutils >= 0.16
- PyYAML (optional, for frontmatter extraction)

## License

Same as the parent project.

