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

logger = logging.getLogger(__name__)


def extract_frontmatter(doctree: nodes.document) -> dict[str, Any]:
    """
    Extract frontmatter metadata from a document tree.
    
    Args:
        doctree: The document tree to extract metadata from
        
    Returns:
        Dictionary of frontmatter fields
    """
    metadata = {}
    
    # Check if the document has docinfo (frontmatter)
    if hasattr(doctree, "settings") and hasattr(doctree.settings, "env"):
        env = doctree.settings.env
        docname = env.docname
        
        # MyST parser stores frontmatter in env.metadata
        if hasattr(env, "metadata") and docname in env.metadata:
            metadata = env.metadata[docname]
    
    return metadata


def build_meta_tags(metadata: dict[str, Any], context: dict[str, Any]) -> list[str]:
    """
    Build HTML meta tags from frontmatter metadata.
    
    Args:
        metadata: Frontmatter metadata dictionary
        context: Sphinx HTML context
        
    Returns:
        List of HTML meta tag strings
    """
    tags = []
    
    # Standard meta tags
    if "description" in metadata:
        description = metadata["description"]
        tags.append(f'<meta name="description" content="{description}">')
        
        # Open Graph
        tags.append(f'<meta property="og:description" content="{description}">')
        
        # Twitter Card
        tags.append(f'<meta name="twitter:description" content="{description}">')
    
    # Keywords from tags
    if "tags" in metadata:
        keywords = metadata["tags"]
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
            tags.append(f'<meta name="keywords" content="{keywords_str}">')
    
    # Author/personas
    if "personas" in metadata:
        personas = metadata["personas"]
        if isinstance(personas, list):
            # Map personas to readable audience descriptions
            audience_map = {
                "data-scientist-focused": "Data Scientists",
                "mle-focused": "Machine Learning Engineers",
                "admin-focused": "Cluster Administrators",
                "devops-focused": "DevOps Professionals",
            }
            audiences = [audience_map.get(p, p) for p in personas]
            audience_str = ", ".join(audiences)
            tags.append(f'<meta name="audience" content="{audience_str}">')
    
    # Content type and difficulty
    if "content_type" in metadata:
        tags.append(f'<meta name="content-type-category" content="{metadata["content_type"]}">')
    
    if "difficulty" in metadata:
        tags.append(f'<meta name="difficulty" content="{metadata["difficulty"]}">')
    
    # Modality
    if "modality" in metadata:
        tags.append(f'<meta name="modality" content="{metadata["modality"]}">')
    
    # Open Graph type
    tags.append('<meta property="og:type" content="article">')
    
    # Open Graph title (from page title)
    if "title" in context:
        title = context["title"]
        tags.append(f'<meta property="og:title" content="{title}">')
        tags.append(f'<meta name="twitter:title" content="{title}">')
        tags.append(f'<meta name="twitter:card" content="summary">')
    
    # Open Graph URL (from page URL)
    if "pageurl" in context:
        url = context["pageurl"]
        tags.append(f'<meta property="og:url" content="{url}">')
    
    # Product information from cascade
    if "cascade" in metadata:
        cascade = metadata["cascade"]
        if isinstance(cascade, dict) and "product" in cascade:
            product = cascade["product"]
            if isinstance(product, dict):
                if "name" in product and product["name"]:
                    tags.append(f'<meta name="product-name" content="{product["name"]}">')
                if "version" in product and product["version"]:
                    tags.append(f'<meta name="product-version" content="{product["version"]}">')
    
    return tags


def build_json_ld(metadata: dict[str, Any], context: dict[str, Any]) -> str | None:
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
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document,
) -> None:
    """
    Add rich metadata to the HTML page context.
    
    This function is called for each page during the HTML build process.
    It extracts frontmatter and injects SEO metadata into the page context.
    """
    if doctree is None:
        return
    
    # Extract frontmatter metadata
    metadata = extract_frontmatter(doctree)
    
    if not metadata:
        return
    
    # Build meta tags
    meta_tags = build_meta_tags(metadata, context)
    
    # Build JSON-LD structured data
    json_ld = build_json_ld(metadata, context)
    
    # Combine all metadata HTML
    metadata_html = "\n    ".join(meta_tags)
    if json_ld:
        metadata_html = f"{metadata_html}\n    {json_ld}"
    
    # Add to context for template injection
    # This will be available in templates as {{ rich_metadata }}
    context["rich_metadata"] = metadata_html
    
    # Also add to metatags if the theme supports it
    if "metatags" not in context:
        context["metatags"] = ""
    context["metatags"] = f"{context['metatags']}\n    {metadata_html}"
    
    logger.debug(f"Rich metadata added for page: {pagename}")


def add_template_path(_app: Sphinx, config: Config) -> None:
    """Add template path during config initialization."""
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

