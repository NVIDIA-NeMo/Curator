---
orphan: true
---

# Documentation Development

- [Documentation Development](#documentation-development)
  - [Set Up the Documentation Environment](#set-up-the-documentation-environment)
  - [Build the Documentation](#build-the-documentation)
    - [Build Variants](#build-variants)
  - [Live Building](#live-building)
  - [Conditional Content for Different Build Types](#conditional-content-for-different-build-types)
    - [1. File-Level Exclusion (Recommended for Entire Sections)](#1-file-level-exclusion-recommended-for-entire-sections)
    - [2. Grid Card Conditional Rendering](#2-grid-card-conditional-rendering)
    - [3. Toctree Conditional Rendering](#3-toctree-conditional-rendering)
    - [Best Practices](#best-practices)
    - [Testing Conditional Content](#testing-conditional-content)
  - [Run Doctests (if present)](#run-doctests-if-present)
  - [Example: How to Write Doctests in Documentation](#example-how-to-write-doctests-in-documentation)
  - [MyST Substitutions in Code Blocks](#myst-substitutions-in-code-blocks)
    - [Configuration](#configuration)
    - [Usage](#usage)
      - [Basic MyST Substitutions in Text](#basic-myst-substitutions-in-text)
      - [MyST Substitutions in Code Blocks](#myst-substitutions-in-code-blocks-1)
    - [Template Language Protection](#template-language-protection)
      - [Protected Languages](#protected-languages)
      - [Pattern Protection](#pattern-protection)
    - [Mixed Usage Examples](#mixed-usage-examples)
      - [YAML with Mixed Syntax](#yaml-with-mixed-syntax)
      - [Ansible with Mixed Syntax](#ansible-with-mixed-syntax)
      - [Benefits](#benefits)


## Set Up the Documentation Environment

Before building or serving the documentation, set up the docs environment using the Makefile:

```sh
make docs-env
source .venv-docs/bin/activate
```

This will create a virtual environment in `.venv-docs` and install all required dependencies for building the documentation.

## Build the Documentation

To build the NeMo Curator documentation, run:

```sh
make docs-html
```

* The resulting HTML files are generated in a `_build/html` folder under the project `docs/` folder.
* The generated Python API docs are placed in `apidocs` under the `docs/` folder.

### Build Variants

The documentation supports different build variants:

- `make docs-html` - Default build (includes all content)
- `make docs-html-ga` - GA (General Availability) build (excludes EA-only content)
- `make docs-html-ea` - EA (Early Access) build (includes all content)

## Live Building

To serve the documentation with live updates as you edit, run:

```sh
make docs-live
```

Open a web browser and go to `http://localhost:8000` (or the port shown in your terminal) to view the output.

## Conditional Content for Different Build Types

The documentation system supports three ways to conditionally include/exclude content based on build tags (e.g., GA vs EA builds). All methods use the unified `:only:` syntax.

### 1. File-Level Exclusion (Recommended for Entire Sections)

Use frontmatter to exclude entire documents from specific builds:

```yaml
---
only: not ga
---

# This entire document will be excluded from GA builds
```

**Supported conditions:**
- `only: not ga` - Exclude from GA builds (EA-only content)
- `only: ga` - Include only in GA builds  
- `only: not ea` - Exclude from EA builds
- `only: internal` - Include only in internal builds

**Directory inheritance:** If a parent directory's `index.md` has an `only` condition, all child documents inherit it automatically.

### 2. Grid Card Conditional Rendering

Hide specific grid cards from certain builds:

```markdown
:::{grid-item-card} Video Curation Features
:link: video-overview  
:link-type: ref
:only: not ga
Content for EA-only features
+++
{bdg-secondary}`early-access`
:::
```

### 3. Toctree Conditional Rendering

Control navigation entries conditionally:

```markdown
# Global toctree condition (hides entire section)
::::{toctree}
:hidden:
:caption: Early Access Features
:only: not ga

ea-feature1.md
ea-feature2.md  
::::

# Inline entry conditions (hides individual entries)
::::{toctree}
:hidden:
:caption: Documentation

standard-doc.md
ea-only-doc.md :only: not ga
another-standard-doc.md
::::
```

### Best Practices

- **Use file-level exclusion** for entire documentation sections (better performance, no warnings)
- **Use grid/toctree conditions** for fine-grained control within shared documents
- **Be consistent** with condition syntax across all methods
- **Test both build variants** to ensure content appears/disappears correctly

### Testing Conditional Content

```bash
# Test default build (includes all content)
make docs-html

# Test GA build (excludes EA-only content)  
make docs-html-ga

# Verify content is properly hidden/shown in each build
```

## Run Doctests (if present)

Sphinx is configured to support running doctests in both Python docstrings and in Markdown code blocks with the `{doctest}` directive. However, as of now, there are **no real doctest examples** in the codebase—only the example below in this README. If you add doctest examples, you can run them manually with:

```sh
source .venv-docs/bin/activate
cd docs
sphinx-build -b doctest . _build/doctest
```

There is currently **no Makefile target** for running doctests; you must use the above command directly.

## Example: How to Write Doctests in Documentation

Any code in triple backtick blocks with the `{doctest}` directive will be tested if you add real examples. The format follows Python's doctest module syntax, where `>>>` indicates Python input and the following line shows the expected output. Here's an example:

```python
def add(x: int, y: int) -> int:
    """
    Adds two integers together.

    Args:
        x (int): The first integer to add.
        y (int): The second integer to add.

    Returns:
        int: The sum of x and y.

    Examples:
    ```{doctest}
    >>> add(1, 2)
    3
    ```

    """
    return x + y
```

## MyST Substitutions in Code Blocks

The documentation uses a custom Sphinx extension (`myst_codeblock_substitutions`) that enables MyST substitutions to work inside standard code blocks. This allows you to maintain consistent variables (like version numbers, URLs, product names) across your documentation while preserving template syntax in YAML and other template languages.

### Configuration

The extension is configured in `docs/conf.py`:

```python
# Add the extension
extensions = [
    # ... other extensions
    "myst_codeblock_substitutions",  # Our custom MyST substitutions in code blocks
]

# Define reusable variables
myst_substitutions = {
    "product_name": "NeMo Curator",
    "product_name_short": "Curator", 
    "company": "NVIDIA",
    "version": release,  # Uses the release variable from conf.py
    "current_year": "2025",
    "github_repo": "https://github.com/NVIDIA/NeMo-Curator",
    "docs_url": "https://docs.nvidia.com/nemo-curator",
    "support_email": "nemo-curator-support@nvidia.com",
    "min_python_version": "3.8",
    "recommended_cuda": "12.0+",
}
```

### Usage

#### Basic MyST Substitutions in Text
Use `{{ variable }}` syntax in regular markdown text:

```markdown
Welcome to {{ product_name }} version {{ version }}!

{{ product_name_short }} is developed by {{ company }}.
For support, contact {{ support_email }}.
```

#### MyST Substitutions in Code Blocks

The extension enables substitutions in standard code blocks:

```bash
# Install {{ product_name }} version {{ version }}
helm install my-release oci://nvcr.io/nvidia/nemo-curator --version {{ version }}
kubectl get pods -n {{ product_name_short }}-namespace
docker run --rm nvcr.io/nvidia/nemo-curator:{{ version }}
pip install nemo-curator=={{ version }}
```

### Template Language Protection

The extension intelligently protects template languages from unwanted substitutions:

#### Protected Languages

These languages are treated carefully to preserve their native `{{ }}` syntax:
- `yaml`, `yml` (Kubernetes, Docker Compose)
- `helm`, `gotmpl`, `go-template` (Helm charts)
- `jinja`, `jinja2`, `j2` (Ansible, Python templates)
- `ansible` (Ansible playbooks)
- `handlebars`, `hbs`, `mustache` (JavaScript templates)
- `django`, `twig`, `liquid`, `smarty` (Web framework templates)

#### Pattern Protection

The extension automatically detects and preserves common template patterns:
- `{{ .Values.something }}` (Helm values)
- `{{ ansible_variable }}` (Ansible variables)
- `{{ item.property }}` (Template loops)
- `{{- variable }}` (Whitespace control)
- `{{ range ... }}`, `{{ if ... }}` (Control structures)

### Mixed Usage Examples

#### YAML with Mixed Syntax

```yaml
# values.yaml - MyST substitutions work alongside Helm templates
image:
  repository: nvcr.io/nvidia/nemo-curator
  tag: {{ .Values.image.tag | default "latest" }}        # ← Helm template (preserved)

# Documentation URLs using MyST substitutions  
downloads:
  releaseUrl: "https://github.com/NVIDIA/NeMo-Curator/releases/download/v{{ version }}/nemo-curator.tar.gz"  # ← MyST substitution
  docsUrl: "{{ docs_url }}"                              # ← MyST substitution
  supportEmail: "{{ support_email }}"                    # ← MyST substitution

service:
  name: {{ include "nemo-curator.fullname" . }}          # ← Helm template (preserved)
  
env:
  - name: CURATOR_VERSION
    value: "{{ .Chart.AppVersion }}"                     # ← Helm template (preserved)
  - name: DOCS_VERSION  
    value: "{{ version }}"                               # ← MyST substitution
```

#### Ansible with Mixed Syntax  

```yaml
# MyST substitutions for documentation
- name: "Install {{ product_name }} version {{ version }}"     # ← MyST substitution
  shell: |
    wget {{ github_repo }}/releases/download/v{{ version }}/nemo-curator.tar.gz  # ← MyST substitution
    
  # Ansible templates preserved
  when: "{{ ansible_distribution }} == 'Ubuntu'"              # ← Ansible template (preserved)
  notify: "{{ handlers.restart_service }}"                    # ← Ansible template (preserved)
```

#### Benefits

1. **Single Source of Truth**: Update version numbers, URLs, and product names in one place (`conf.py`)
2. **Template Safety**: Won't break existing Helm charts, Ansible playbooks, or other templates
3. **Intelligent Processing**: Only processes simple variable names, preserves complex template syntax
4. **Cross-Format Support**: Works in bash, python, dockerfile, and other code blocks
5. **Maintainability**: Reduces copy-paste errors and keeps documentation in sync with releases

The extension automatically handles the complexity of mixed template syntax, so you can focus on writing great documentation without worrying about breaking existing templates.