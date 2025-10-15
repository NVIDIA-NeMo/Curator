# Rich Metadata Extension - Usage Guide

Complete guide for using the rich metadata extension to inject SEO-optimized metadata into your documentation.

## Quick Start

### 1. Enable the Extension

The extension is already enabled in `docs/conf.py`:

```python
extensions = [
    # ... other extensions
    "rich_metadata",  # SEO metadata injection from frontmatter
]
```

### 2. Add Frontmatter to Your Pages

Add frontmatter at the top of any markdown file:

```markdown
---
description: "Brief summary for SEO and social sharing"
tags: ["keyword1", "keyword2", "keyword3"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
cascade:
  product:
    name: "NeMo Curator"
    version: "25.09"
---

# Your Page Title

Your content goes here...
```

### 3. Build and Verify

Build your documentation:

```bash
cd docs/
make clean html
```

Verify metadata was injected:

```bash
python _extensions/rich_metadata/verify_metadata.py _build/html/index.html
```

## Complete Example

### Example Page: `docs/tutorials/my-tutorial.md`

```markdown
---
description: "Learn how to process text data with NeMo Curator in this hands-on tutorial"
tags: ["text-curation", "tutorial", "data-processing", "filtering", "python-api"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
cascade:
  product:
    name: "NeMo Curator"
    version: "25.09"
---

(my-tutorial)=

# My First Text Curation Tutorial

This tutorial shows you how to...

## Prerequisites

- Python 3.10+
- NeMo Curator installed

## Steps

1. First step...
2. Second step...
```

### Generated HTML Head

After building, the HTML will contain:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>My First Text Curation Tutorial</title>
  
  <!-- Standard Meta Tags -->
  <meta name="description" content="Learn how to process text data with NeMo Curator in this hands-on tutorial">
  <meta name="keywords" content="text-curation, tutorial, data-processing, filtering, python-api">
  <meta name="audience" content="Data Scientists, Machine Learning Engineers">
  <meta name="content-type-category" content="tutorial">
  <meta name="difficulty" content="beginner">
  <meta name="modality" content="text-only">
  <meta name="product-name" content="NeMo Curator">
  <meta name="product-version" content="25.09">
  
  <!-- Open Graph -->
  <meta property="og:type" content="article">
  <meta property="og:title" content="My First Text Curation Tutorial">
  <meta property="og:description" content="Learn how to process text data with NeMo Curator in this hands-on tutorial">
  <meta property="og:url" content="https://docs.nvidia.com/nemo-curator/tutorials/my-tutorial.html">
  
  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="My First Text Curation Tutorial">
  <meta name="twitter:description" content="Learn how to process text data with NeMo Curator in this hands-on tutorial">
  
  <!-- JSON-LD Structured Data -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "HowTo",
    "headline": "My First Text Curation Tutorial",
    "name": "My First Text Curation Tutorial",
    "description": "Learn how to process text data with NeMo Curator in this hands-on tutorial",
    "keywords": ["text-curation", "tutorial", "data-processing", "filtering", "python-api"],
    "proficiencyLevel": "Beginner",
    "audience": {
      "@type": "Audience",
      "audienceType": ["Data Scientists", "Machine Learning Engineers"]
    },
    "url": "https://docs.nvidia.com/nemo-curator/tutorials/my-tutorial.html",
    "publisher": {
      "@type": "Organization",
      "name": "NVIDIA Corporation",
      "url": "https://www.nvidia.com"
    },
    "about": {
      "@type": "SoftwareApplication",
      "name": "NeMo Curator",
      "applicationCategory": "Data Curation Software",
      "operatingSystem": "Linux",
      "softwareVersion": "25.09"
    }
  }
  </script>
  
  <!-- Rest of head content -->
</head>
<body>
  <!-- Page content -->
</body>
</html>
```

## Frontmatter Field Reference

### Required Fields

None! All fields are optional. However, for best SEO results, include at least:

- `description`: Brief page summary (1-2 sentences)
- `tags`: 3-8 relevant keywords

### Recommended Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `description` | string | Brief page summary for SEO | `"Step-by-step guide to text curation"` |
| `tags` | list | Keywords (3-8 recommended) | `["text", "tutorial", "filtering"]` |
| `difficulty` | string | Content difficulty level | `"beginner"`, `"intermediate"`, `"advanced"`, `"reference"` |
| `content_type` | string | Type of content | `"tutorial"`, `"concept"`, `"reference"`, `"troubleshooting"`, `"example"` |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `personas` | list | Target audiences | `["data-scientist-focused", "mle-focused"]` |
| `modality` | string | Content modality | `"text-only"`, `"image-only"`, `"video-only"`, `"multimodal"`, `"universal"` |
| `cascade.product.name` | string | Product name | `"NeMo Curator"` |
| `cascade.product.version` | string | Product version | `"25.09"` |

### Personas

Available persona values:

- `data-scientist-focused` → "Data Scientists"
- `mle-focused` → "Machine Learning Engineers"
- `admin-focused` → "Cluster Administrators"
- `devops-focused` → "DevOps Professionals"

### Content Types

| Value | Maps to Schema.org | Best For |
|-------|-------------------|----------|
| `tutorial` | `HowTo` | Step-by-step guides |
| `troubleshooting` | `HowTo` | Problem-solving guides |
| `concept` | `Article` | Explanatory content |
| `reference` | `TechArticle` | API docs, specifications |
| `example` | `HowTo` | Code examples |

## Verification

### Method 1: Using the Verification Script

```bash
# Verify a single page
python _extensions/rich_metadata/verify_metadata.py _build/html/index.html

# Verify multiple pages
python _extensions/rich_metadata/verify_metadata.py \
  _build/html/index.html \
  _build/html/get-started/text.html \
  _build/html/about/index.html

# Verbose output
python _extensions/rich_metadata/verify_metadata.py -v _build/html/index.html
```

Example output:

```
================================================================================
Verifying: index.html
================================================================================

✅ Standard Meta Tags:
   • description: Overview of NeMo Curator...
   • keywords: overview, platform, multimodal, enterprise, getting-started
   • audience: Data Scientists, Machine Learning Engineers, Cluster Administrators...
   • difficulty: beginner
   • content-type-category: concept
   • modality: universal

✅ Open Graph Tags:
   • og:type: article
   • og:title: NeMo Curator Documentation
   • og:description: Overview of NeMo Curator...
   • og:url: https://docs.nvidia.com/nemo-curator/index.html

✅ Twitter Card Tags:
   • twitter:card: summary
   • twitter:title: NeMo Curator Documentation
   • twitter:description: Overview of NeMo Curator...

✅ JSON-LD Structured Data:
   • @type: Article
   • headline: NeMo Curator Documentation
   • description: Overview of NeMo Curator, an open-source platform for scalable data...
   • keywords: overview, platform, multimodal, enterprise, getting-started
   • audience: Data Scientists, Machine Learning Engineers, Cluster Administrators...
   • proficiency: Beginner

✅ Rich metadata extension is working!
```

### Method 2: Manual Inspection

```bash
# View the HTML head
head -n 100 _build/html/index.html | grep -A 2 'meta'

# Search for JSON-LD
grep -A 20 'application/ld+json' _build/html/index.html

# Check specific meta tags
grep 'meta name="description"' _build/html/index.html
```

### Method 3: Browser DevTools

1. Build and serve the docs:
   ```bash
   make html
   python -m http.server 8000 -d _build/html
   ```

2. Open http://localhost:8000 in your browser

3. Open DevTools (F12) → Elements tab → `<head>` section

4. Look for the meta tags and JSON-LD script

## Testing Social Sharing

### Test Open Graph (Facebook/LinkedIn)

1. Use the Facebook Sharing Debugger:
   https://developers.facebook.com/tools/debug/

2. Enter your page URL

3. Verify the preview shows your description, title, etc.

### Test Twitter Cards

1. Use the Twitter Card Validator:
   https://cards-dev.twitter.com/validator

2. Enter your page URL

3. Verify the card preview

### Test Structured Data

1. Use Google's Rich Results Test:
   https://search.google.com/test/rich-results

2. Enter your page URL or paste the HTML

3. Verify the structured data is recognized

## Troubleshooting

### No Metadata in Output

**Problem**: Built HTML has no meta tags

**Solutions**:
1. Check extension is enabled in `conf.py`
2. Verify frontmatter is valid YAML
3. Build with verbose output: `sphinx-build -v docs/ _build/html`
4. Check build logs for errors

### Metadata Not Visible in Browser

**Problem**: Metadata is in HTML source but not rendering

**Solutions**:
1. Meta tags don't render visibly - check page source (Ctrl+U)
2. Verify you're looking at `<head>` not `<body>`
3. Use verification script: `python verify_metadata.py <file>`

### Template Not Rendering `{{ rich_metadata }}`

**Problem**: Template doesn't support custom metadata

**Solutions**:
1. The extension includes `templates/layout.html` which should work automatically
2. Verify the template was added: check build logs for "Rich metadata templates added"
3. Customize the template if needed (see Advanced Usage)
4. Check theme documentation for metadata injection points

### JSON-LD Not Valid

**Problem**: Structured data fails validation

**Solutions**:
1. Use verification script to check JSON-LD
2. Validate with: https://validator.schema.org/
3. Check for quote escaping in description field
4. Report issues if data structure is incorrect

## Advanced Usage

### Custom Template Integration

The extension includes a template override at `_extensions/rich_metadata/templates/layout.html` which is automatically added to Sphinx's template search path.

To customize the template further, edit `_extensions/rich_metadata/templates/layout.html`:

```html
{% extends "!layout.html" %}

{% block extrahead %}
  {{ super() }}
  
  {# Rich metadata from frontmatter #}
  {% if rich_metadata %}
    {{ rich_metadata|safe }}
  {% endif %}
  
  {# Additional custom meta tags #}
  <meta name="custom-tag" content="custom-value">
{% endblock %}
```

### Programmatic Access

Access metadata in templates:

```html
{% if page_metadata %}
  <div class="page-info">
    Difficulty: {{ page_metadata.difficulty }}
    Content Type: {{ page_metadata.content_type }}
  </div>
{% endif %}
```

### Conditional Metadata

Only include metadata for certain pages:

```python
# In conf.py
def skip_metadata(app, pagename, templatename, context, doctree):
    """Skip metadata for certain pages."""
    if pagename.startswith("_"):
        context["rich_metadata"] = ""

def setup(app):
    app.connect("html-page-context", skip_metadata, priority=1000)
```

## Best Practices

### Description Field

- **Length**: 150-160 characters optimal
- **Content**: Action-oriented, clear value proposition
- **Avoid**: Generic phrases, duplicate content
- **Example**: ✅ "Learn how to filter and deduplicate text datasets for LLM training"
- **Bad**: ❌ "This page has information about text processing"

### Tags Field

- **Count**: 3-8 tags per page
- **Style**: Use lowercase, hyphen-separated
- **Relevance**: Choose specific, relevant terms
- **Example**: ✅ `["text-curation", "quality-filtering", "deduplication", "llm-training"]`
- **Bad**: ❌ `["docs", "page", "information"]` (too generic)

### Personas

- Include 1-3 relevant personas
- Choose based on actual content
- Don't include all personas on every page

### Difficulty

- `beginner`: No prior knowledge required
- `intermediate`: Some familiarity expected
- `advanced`: Deep technical knowledge required
- `reference`: API docs, specifications

### Content Type

- `tutorial`: Step-by-step instructions
- `concept`: Explanations and theory
- `reference`: API docs, specifications
- `troubleshooting`: Problem-solving guides
- `example`: Code samples and demos

## Migration Guide

### Adding Metadata to Existing Pages

Use this workflow to add metadata to all pages:

1. **Audit existing pages**:
   ```bash
   find docs/ -name "*.md" -type f | wc -l
   ```

2. **Generate metadata** (see `docs-frontmatter` rule):
   Use AI assistant to analyze and generate appropriate frontmatter

3. **Verify metadata**:
   ```bash
   make html
   python _extensions/rich_metadata/verify_metadata.py _build/html/**/*.html
   ```

4. **Test social sharing**:
   Use Facebook/Twitter validators

## Performance

### Build Time Impact

- **Minimal**: < 1% build time increase
- **Parallel Safe**: Extension supports parallel builds
- **No External Calls**: All processing is local

### HTML Size Impact

- **Per Page**: ~1-3 KB additional HTML
- **Typical**: ~1.5 KB (meta tags + JSON-LD)
- **Gzipped**: ~500 bytes

## Support

### Debugging

Enable debug logging:

```python
# In conf.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **No frontmatter**: Pages without frontmatter won't have metadata
2. **Invalid YAML**: Check YAML syntax with a validator
3. **Theme incompatibility**: Use template override
4. **Build caching**: Run `make clean html` to rebuild

### Getting Help

1. Check build logs for errors
2. Run verification script
3. Review README.md and USAGE.md
4. Check Sphinx and MyST parser documentation

## Examples

See these pages for reference:

- `docs/about/index.md` - Concept page with full metadata
- `docs/get-started/text.md` - Tutorial with rich metadata
- `docs/index.md` - Homepage with universal modality

## License

Copyright (c) 2025, NVIDIA CORPORATION. Licensed under Apache 2.0.

