# Content Gating Extension

A comprehensive Sphinx extension for conditional content rendering based on release stage tags.

## Features

### Multi-Level Content Gating
- **Document Level**: Filter entire documents via frontmatter
- **Toctree Level**: Conditional toctrees with global and per-entry filtering  
- **Directive Level**: Conditional grid cards and other directives
- **Content Level**: Conditional text sections and arbitrary content blocks

### Supported Tags
- `ga` - General Availability (production ready)
- `ea` - Early Access (limited availability)
- `internal` - Internal only
- Custom tags as needed

## Usage

### 1. Document-Level Filtering

Add to document frontmatter:
```yaml
---
only: not ga
---
```

This excludes the entire document when building with the `ga` tag.

### 2. Toctree Filtering

**Global condition (entire toctree):**
```rst
::::{toctree}
:only: not ga  
:hidden:
:caption: Early Access Features

ea-feature-1.md
ea-feature-2.md
::::
```

**Per-entry conditions:**
```rst
::::{toctree}
:hidden:
:caption: Mixed Content

stable-feature.md
new-feature.md :only: ea
experimental.md :only: internal
::::
```

### 3. Grid Card Filtering

```rst
:::{grid-item-card} EA Feature
:only: ea

This card only shows in EA builds.
:::
```

### 4. Content Section Filtering

```rst
:::{conditional}
:only: not ga

## Early Access Feature

This entire section only appears in non-GA builds.

- Feature details
- Usage examples
- Configuration options
:::
```

Or for inline content:

```rst
:::{conditional}
:only: internal

**Internal Note**: This information is only visible in internal builds.
:::
```

## Building with Tags

```bash
# GA build (production)
sphinx-build -t ga docs/ _build/ga/

# EA build (early access)  
sphinx-build -t ea docs/ _build/ea/

# Internal build (all content)
sphinx-build -t internal docs/ _build/internal/

# Default build (no special tags)
sphinx-build docs/ _build/
```

## Condition Syntax

- `ga` - Include only if `ga` tag present
- `not ga` - Include only if `ga` tag NOT present  
- `ea` - Include only if `ea` tag present
- `not ea` - Include only if `ea` tag NOT present
- `internal` - Include only if `internal` tag present
- `not internal` - Include only if `internal` tag NOT present

## Directory Inheritance

Documents inherit `only` conditions from parent directory `index.md` files:

```
feature-x/
├── index.md          # only: ea
├── tutorial.md       # inherits "only: ea"
└── reference.md      # inherits "only: ea" 
```

## Configuration

Add to `conf.py`:
```python
extensions = [
    # ... other extensions
    'content_gating',
]
```

## Module Structure

- `__init__.py` - Main extension setup
- `condition_evaluator.py` - Shared condition evaluation logic
- `document_filter.py` - Document-level filtering
- `conditional_directives.py` - Directive-level filtering (toctree, grid cards) 