"""
Rich Metadata Extension for Sphinx

Injects SEO-optimized metadata into HTML <head> based on frontmatter.
Supports Open Graph, Twitter Cards, JSON-LD structured data, and standard meta tags.

Frontmatter fields supported:
- description: Page description for meta tags
- tags: Keywords for SEO
- personas: Target audience information
- difficulty: Content difficulty level
- content_type: Type of content (tutorial, concept, reference, etc.)
- modality: Content modality (text-only, image-only, video-only, multimodal, universal)
- cascade.product.name: Product name
- cascade.product.version: Product version
"""

import json
import os
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.util import logging

# Import YAML for frontmatter parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = logging.getLogger(__name__)


def extract_frontmatter(env, docname: str) -> dict[str, Any]:  # noqa: ANN001
    """
    Extract frontmatter from markdown source file.

    Uses the same approach as the json_output extension (proven to work).

    Args:
        env: Sphinx build environment
        docname: Document name

    Returns:
        Dictionary of frontmatter fields
    """
    metadata = {}

    if not YAML_AVAILABLE:
        logger.warning("PyYAML not available, frontmatter cannot be parsed")
        return metadata

    try:
        # Get source path - this returns the full absolute path
        source_path = env.doc2path(docname)

        # Only process markdown files
        if source_path and str(source_path).endswith(".md"):
            with open(source_path, encoding="utf-8") as f:
                content = f.read()

            # Check for valid frontmatter format (YAML between --- markers)
            if content.startswith("---"):
                end_marker = content.find("\n---\n", 3)
                if end_marker != -1:
                    frontmatter_text = content[3:end_marker]
                    metadata = yaml.safe_load(frontmatter_text) or {}
                    logger.debug(f"rich_metadata: Extracted {len(metadata)} fields from {docname}")

    except yaml.YAMLError as e:
        logger.warning(f"rich_metadata: YAML parsing error in {docname}: {e}")
    except OSError as e:
        logger.warning(f"rich_metadata: File error for {docname}: {e}")

    return metadata


def build_page_title(_metadata: dict[str, Any], context: dict[str, Any], _pagename: str) -> str | None:
    """
    Build enhanced page title in format: Page Title: Section - Site | NVIDIA

    Args:
        metadata: Page frontmatter metadata
        context: Sphinx page context
        pagename: Document name

    Returns:
        Enhanced title string or None to keep default
    """
    # Get base title
    page_title = context.get("title", "")
    if not page_title:
        return None

    # Build title components
    components = [page_title]

    # Add parent section if available (from breadcrumbs or parents)
    if context.get("parents"):
        # Get the immediate parent (last item in parents list)
        parent = context["parents"][-1]
        if "title" in parent and parent["title"] != page_title:
            components.append(parent["title"])

    # Add site name (from Sphinx's project config)
    site_name = context.get("docstitle", "")
    if site_name and site_name != page_title:
        components.append(site_name)

    # Build final title with format depending on available components
    if len(components) == 1:
        # Just page: "Page Title | NVIDIA"
        return f"{components[0]} | NVIDIA"
    elif len(components) == 2:  # noqa: PLR2004
        # Page and site: "Page Title - Site | NVIDIA"
        return f"{components[0]} - {components[1]} | NVIDIA"
    else:
        # Page, section, and site: "Page Title: Section - Site | NVIDIA"
        return f"{components[0]}: {components[1]} - {components[2]} | NVIDIA"


def _add_basic_fields(metadata: dict[str, Any]) -> list[str]:
    """Add basic SEO meta tags."""
    basic_tags = []
    if "description" in metadata:
        description = metadata["description"]
        basic_tags.append(f'<meta name="description" content="{description}">')

    if "tags" in metadata:
        keywords = metadata["tags"]
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
            basic_tags.append(f'<meta name="keywords" content="{keywords_str}">')
    return basic_tags


def _add_opengraph_fields(metadata: dict[str, Any], context: dict[str, Any]) -> list[str]:
    """Add Open Graph meta tags."""
    og_tags = []
    if "description" in metadata:
        og_tags.append(f'<meta property="og:description" content="{metadata["description"]}">')

    og_tags.append('<meta property="og:type" content="article">')

    enhanced_title = context.get("pagetitle") or context.get("title", "")
    if enhanced_title:
        og_tags.append(f'<meta property="og:title" content="{enhanced_title}">')

    if "pageurl" in context:
        url = context["pageurl"]
        og_tags.append(f'<meta property="og:url" content="{url}">')

    return og_tags


def _add_twitter_fields(metadata: dict[str, Any], context: dict[str, Any]) -> list[str]:
    """Add Twitter Card meta tags."""
    twitter_tags = []
    if "description" in metadata:
        twitter_tags.append(f'<meta name="twitter:description" content="{metadata["description"]}">')

    enhanced_title = context.get("pagetitle") or context.get("title", "")
    if enhanced_title:
        twitter_tags.append(f'<meta name="twitter:title" content="{enhanced_title}">')

    twitter_tags.append('<meta name="twitter:card" content="summary">')
    return twitter_tags


def _add_personas_tag(metadata: dict[str, Any]) -> list[str]:
    """Add personas/audience metadata tag."""
    if "personas" not in metadata:
        return []

    personas = metadata["personas"]
    if not isinstance(personas, list):
        return []

    audience_map = {
        "data-scientist-focused": "Data Scientists",
        "mle-focused": "Machine Learning Engineers",
        "admin-focused": "Cluster Administrators",
        "devops-focused": "DevOps Professionals",
    }
    audiences = [audience_map.get(p, p) for p in personas]
    audience_str = ", ".join(audiences)
    return [f'<meta name="audience" content="{audience_str}">']


def _add_product_tags(metadata: dict[str, Any]) -> list[str]:
    """Add product metadata tags from cascade."""
    if "cascade" not in metadata:
        return []

    cascade = metadata["cascade"]
    if not isinstance(cascade, dict) or "product" not in cascade:
        return []

    product = cascade["product"]
    if not isinstance(product, dict):
        return []

    tags = []
    if product.get("name"):
        tags.append(f'<meta name="product-name" content="{product["name"]}">')
    if product.get("version"):
        tags.append(f'<meta name="product-version" content="{product["version"]}">')
    return tags


def _add_custom_fields(metadata: dict[str, Any]) -> list[str]:
    """Add custom NVIDIA/content metadata tags."""
    custom_tags = []

    # Add personas/audience tags
    custom_tags.extend(_add_personas_tag(metadata))

    # Add simple content metadata fields
    if "content_type" in metadata:
        custom_tags.append(f'<meta name="content-type-category" content="{metadata["content_type"]}">')

    if "difficulty" in metadata:
        custom_tags.append(f'<meta name="difficulty" content="{metadata["difficulty"]}">')

    if "modality" in metadata:
        custom_tags.append(f'<meta name="modality" content="{metadata["modality"]}">')

    # Add product information from cascade
    custom_tags.extend(_add_product_tags(metadata))

    return custom_tags


def build_meta_tags(metadata: dict[str, Any], context: dict[str, Any]) -> dict[str, list[str]]:
    """
    Build HTML meta tags from frontmatter metadata, organized by category.

    Args:
        metadata: Frontmatter metadata dictionary
        context: Sphinx HTML context

    Returns:
        Dictionary with categorized meta tag lists
    """
    return {
        "basic": _add_basic_fields(metadata),
        "opengraph": _add_opengraph_fields(metadata, context),
        "twitter": _add_twitter_fields(metadata, context),
        "custom": _add_custom_fields(metadata)
    }


def build_json_ld(metadata: dict[str, Any], context: dict[str, Any]) -> str | None:  # noqa: C901, PLR0912
    """
    Build JSON-LD structured data for SEO.

    Args:
        metadata: Frontmatter metadata dictionary
        context: Sphinx HTML context

    Returns:
        JSON-LD script tag string or None
    """
    # Base structure
    structured_data = {
        "@context": "https://schema.org",
        "@type": "TechArticle",
    }

    # Add title
    if "title" in context:
        structured_data["headline"] = context["title"]
        structured_data["name"] = context["title"]

    # Add description
    if "description" in metadata:
        structured_data["description"] = metadata["description"]

    # Add keywords
    if "tags" in metadata and isinstance(metadata["tags"], list):
        structured_data["keywords"] = metadata["tags"]

    # Add content type mapping
    if "content_type" in metadata:
        content_type = metadata["content_type"]
        type_mapping = {
            "tutorial": "HowTo",
            "troubleshooting": "HowTo",
            "concept": "Article",
            "reference": "TechArticle",
            "example": "HowTo",
        }
        if content_type in type_mapping:
            structured_data["@type"] = type_mapping[content_type]

    # Add difficulty as proficiency level
    if "difficulty" in metadata:
        difficulty_map = {
            "beginner": "Beginner",
            "intermediate": "Intermediate",
            "advanced": "Expert",
            "reference": "Expert",
        }
        if metadata["difficulty"] in difficulty_map:
            structured_data["proficiencyLevel"] = difficulty_map[metadata["difficulty"]]

    # Add audience
    if "personas" in metadata and isinstance(metadata["personas"], list):
        audience_map = {
            "data-scientist-focused": "Data Scientists",
            "mle-focused": "Machine Learning Engineers",
            "admin-focused": "System Administrators",
            "devops-focused": "DevOps Engineers",
        }
        audiences = [audience_map.get(p, p) for p in metadata["personas"]]
        structured_data["audience"] = {
            "@type": "Audience",
            "audienceType": audiences,
        }

    # Add URL
    if "pageurl" in context:
        structured_data["url"] = context["pageurl"]

    # Add publisher (NVIDIA)
    structured_data["publisher"] = {
        "@type": "Organization",
        "name": "NVIDIA Corporation",
        "url": "https://www.nvidia.com",
    }

    # Add product information
    if "cascade" in metadata:
        cascade = metadata["cascade"]
        if isinstance(cascade, dict) and "product" in cascade:
            product = cascade["product"]
            if isinstance(product, dict) and product.get("name"):
                structured_data["about"] = {
                    "@type": "SoftwareApplication",
                    "name": product.get("name", ""),
                    "applicationCategory": "Data Curation Software",
                    "operatingSystem": "Linux",
                }
                if product.get("version"):
                    structured_data["about"]["softwareVersion"] = product["version"]

    # Generate JSON-LD script tag
    json_str = json.dumps(structured_data, indent=2)
    return f'<script type="application/ld+json">\n{json_str}\n</script>'


def add_metadata_to_context(
    app: Sphinx,
    pagename: str,
    _templatename: str,
    context: dict[str, Any],
    doctree: nodes.document,
) -> None:
    """
    Add rich metadata to the HTML page context.

    This function is called for each page during the HTML build process.
    It extracts frontmatter and injects SEO metadata into the page context.
    """
    # Skip generated pages (genindex, py-modindex, search, etc.)
    # These don't have doctrees and don't need custom metadata
    if doctree is None:
        logger.debug(f"rich_metadata: Skipping generated page {pagename} (no doctree)")
        return

    # Extract frontmatter metadata from source file
    env = app.builder.env
    metadata = extract_frontmatter(env, pagename)

    if not metadata:
        return

    # Build enhanced page title
    enhanced_title = build_page_title(metadata, context, pagename)
    if enhanced_title:
        # Set pagetitle which is used by our layout template
        # Template will use this for <title> tag to prevent theme from appending docstitle
        context["pagetitle"] = enhanced_title

    # Build meta tags (returns dict with categorized tags)
    meta_tags = build_meta_tags(metadata, context)

    # Build JSON-LD structured data
    json_ld = build_json_ld(metadata, context)

    # Organize metadata with clear grouping
    metadata_sections = []

    # Basic SEO meta tags
    if meta_tags.get("basic"):
        metadata_sections.append("<!-- SEO Meta Tags -->")
        metadata_sections.extend(meta_tags["basic"])

    # Open Graph tags
    if meta_tags.get("opengraph"):
        metadata_sections.append("\n    <!-- Open Graph / Facebook -->")
        metadata_sections.extend(meta_tags["opengraph"])

    # Twitter Card tags
    if meta_tags.get("twitter"):
        metadata_sections.append("\n    <!-- Twitter -->")
        metadata_sections.extend(meta_tags["twitter"])

    # Custom NVIDIA/content metadata
    if meta_tags.get("custom"):
        metadata_sections.append("\n    <!-- Content Metadata -->")
        metadata_sections.extend(meta_tags["custom"])

    # JSON-LD structured data
    if json_ld:
        metadata_sections.append("\n    <!-- Structured Data (JSON-LD) -->")
        metadata_sections.append(json_ld)

    # Combine all sections
    metadata_html = "\n    ".join(metadata_sections)

    # Add to context for template injection
    if "metatags" not in context:
        context["metatags"] = ""
    context["metatags"] = f"{context['metatags']}\n    {metadata_html}"

    logger.debug(f"rich_metadata: Added {len(meta_tags)} meta tags + JSON-LD for {pagename}")


def add_template_path(_app: Sphinx, config: Config) -> None:
    """
    Add template path during config initialization.

    This ensures our layout.html override is found, which injects
    the metatags context variable into the extrahead block.
    """
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(extension_dir, "templates")

    if os.path.exists(templates_path):
        # Ensure templates_path is a list
        if not isinstance(config.templates_path, list):
            config.templates_path = list(config.templates_path) if config.templates_path else []

        # Add our template path if not already present
        if templates_path not in config.templates_path:
            config.templates_path.append(templates_path)
            logger.info(f"Rich metadata templates added: {templates_path}")


def setup(app: Sphinx) -> dict[str, Any]:
    """
    Setup function for the rich metadata extension.

    This extension injects SEO metadata into the HTML <head> by:
    1. Modifying the page context's metatags variable
    2. Providing a layout.html override that renders metatags in extrahead block
    """
    # Add our templates directory to Sphinx's template search path
    app.connect("config-inited", add_template_path)

    # Connect to the html-page-context event
    # This is called after the context is created but before the template is rendered
    app.connect("html-page-context", add_metadata_to_context)

    logger.info("Rich metadata extension initialized")

    return {
        "version": "1.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

