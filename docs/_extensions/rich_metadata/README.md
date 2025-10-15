# Rich Metadata Extension

Sphinx extension that injects SEO-optimized metadata into HTML `<head>` based on page frontmatter.

## Features

- **Standard Meta Tags**: Description, keywords, author
- **Open Graph**: For social media sharing (Facebook, LinkedIn, etc.)
- **Twitter Cards**: Enhanced Twitter previews
- **JSON-LD Structured Data**: Machine-readable content for search engines
- **Custom Metadata**: Product information, difficulty, content type, modality

## Installation

The extension is automatically loaded when present in `_extensions/`. Add it to `conf.py`:

```python
extensions = [
    # ... other extensions
    "rich_metadata",
]
```

## Supported Frontmatter Fields

The extension recognizes these frontmatter fields:

```yaml
---
description: "Brief summary for SEO and social sharing"
tags: ["keyword1", "keyword2", "keyword3"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"  # beginner, intermediate, advanced, reference
content_type: "tutorial"  # tutorial, concept, reference, troubleshooting, example
modality: "text-only"  # text-only, image-only, video-only, multimodal, universal
cascade:
  product:
    name: "NeMo Curator"
    version: "25.09"
---
```

## Generated Metadata

### Standard Meta Tags

```html
<meta name="description" content="...">
<meta name="keywords" content="...">
<meta name="audience" content="...">
<meta name="difficulty" content="...">
<meta name="content-type-category" content="...">
<meta name="modality" content="...">
<meta name="product-name" content="...">
<meta name="product-version" content="...">
```

### Open Graph Tags

```html
<meta property="og:type" content="article">
<meta property="og:title" content="...">
<meta property="og:description" content="...">
<meta property="og:url" content="...">
```

### Twitter Card Tags

```html
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="...">
<meta name="twitter:description" content="...">
```

### JSON-LD Structured Data

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "...",
  "description": "...",
  "keywords": [...],
  "audience": {
    "@type": "Audience",
    "audienceType": [...]
  },
  "publisher": {
    "@type": "Organization",
    "name": "NVIDIA Corporation"
  }
}
</script>
```

## How It Works

1. **Frontmatter Extraction**: Reads frontmatter from each markdown file using MyST parser metadata
2. **Tag Generation**: Converts frontmatter fields into appropriate meta tags and structured data
3. **Context Injection**: Adds generated HTML to Sphinx's page context
4. **Template Rendering**: Metadata is automatically included in `<head>` via `{{ metatags }}` or `{{ rich_metadata }}`

## Customization

### Template Override

A template override is included at `templates/layout.html` which ensures metadata is injected into the `<head>` section. The extension automatically adds its template directory to Sphinx's template search path.

If you need to customize the template further, edit `templates/layout.html`:

```html
{# templates/layout.html #}
{% extends "!layout.html" %}

{% block extrahead %}
  {{ super() }}
  {{ rich_metadata|safe }}
{% endblock %}
```

### Custom Audience Mapping

Edit the `audience_map` in `__init__.py` to customize persona labels:

```python
audience_map = {
    "data-scientist-focused": "Data Scientists",
    "custom-persona": "Custom Audience Label",
}
```

### Custom Content Type Mapping

Edit the `type_mapping` in `build_json_ld()` to map content types to schema.org types:

```python
type_mapping = {
    "tutorial": "HowTo",
    "custom-type": "Article",
}
```

## Testing

To verify the extension is working:

1. Build the docs: `make html`
2. Open any page's HTML source
3. Look for `<meta>` tags and `<script type="application/ld+json">` in the `<head>`

Example verification:

```bash
# Build docs
cd docs/
make html

# Check generated metadata
grep -A 5 'application/ld+json' _build/html/index.html
```

## Debugging

Enable debug logging in `conf.py`:

```python
# Enable debug logging for the extension
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check the build output for messages like:

```
Rich metadata added for page: index
```

## Compatibility

- **Sphinx**: 4.0+
- **Python**: 3.8+
- **MyST Parser**: Required for frontmatter extraction
- **Themes**: Works with any Sphinx theme that renders `{{ metatags }}` or supports `{% block extrahead %}`

## SEO Benefits

This extension improves SEO by providing:

1. **Search Engine Discovery**: Meta descriptions and keywords help search engines understand content
2. **Social Sharing**: Open Graph and Twitter Cards create rich previews
3. **Structured Data**: JSON-LD helps search engines display rich snippets
4. **Audience Targeting**: Persona and difficulty metadata helps users find relevant content
5. **Product Context**: Version and product metadata helps users find documentation for specific releases

## Future Enhancements

Possible future additions:

- [ ] Article publish/modified dates
- [ ] Article author information
- [ ] Breadcrumb structured data
- [ ] Video/image structured data for media-heavy pages
- [ ] FAQ structured data for Q&A content
- [ ] Custom schema.org types based on modality
- [ ] Language/locale metadata
- [ ] Canonical URL handling

## License

Copyright (c) 2025, NVIDIA CORPORATION. Licensed under Apache 2.0.

