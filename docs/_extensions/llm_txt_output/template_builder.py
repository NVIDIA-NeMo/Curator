"""Template building for llm.txt format."""

import re
from datetime import datetime, timezone
from typing import Any

from sphinx.util import logging

from .content_extractor import break_into_paragraphs

logger = logging.getLogger(__name__)


def build_llm_txt_content(content_data: dict[str, Any]) -> str:  # noqa: C901, PLR0912, PLR0915
    """Build llm.txt content from extracted data."""
    lines = []

    # Title (H1)  # noqa: ERA001
    title = content_data.get("title", "Untitled")
    lines.append(f"# {title}")
    lines.append("")

    # Summary (blockquote)  # noqa: ERA001
    summary = content_data.get("summary", "")
    if summary:
        lines.append(f"> {summary}")
        lines.append("")

    # Overview section (with paragraph breaking for readability)
    overview = content_data.get("overview", "")
    if overview:
        lines.append("## Overview")
        lines.append("")
        # Break long overview into readable paragraphs
        overview = break_into_paragraphs(overview, max_paragraph_length=400)
        lines.append(overview)
        lines.append("")

    # Grid Cards (for index pages with card layouts)
    grid_cards = content_data.get("grid_cards", [])
    if grid_cards:
        lines.append("## Available Options")
        lines.append("")
        for card in grid_cards:
            card_title = card.get("title", "")
            card_desc = card.get("description", "")
            card_url = card.get("url", "")

            if card_url:
                lines.append(f"- **[{card_title}]({card_url})**")
            else:
                lines.append(f"- **{card_title}**")

            if card_desc:
                lines.append(f"  {card_desc}")
        lines.append("")

    # Key Sections (from headings)
    headings = content_data.get("headings", [])
    if headings:
        lines.append("## Key Sections")
        lines.append("")
        for heading in headings:
            heading_text = heading.get("text", "")
            preview = heading.get("preview", "")
            if preview:
                lines.append(f"- **{heading_text}**: {preview}")
            else:
                lines.append(f"- **{heading_text}**")
        lines.append("")

    # Related Resources (links)
    links = content_data.get("links", [])
    if links:
        lines.append("## Related Resources")
        lines.append("")
        for link in links:
            link_text = link.get("text", "")
            link_url = link.get("url", "")
            lines.append(f"- [{link_text}]({link_url})")
        lines.append("")

    # Metadata section
    metadata = content_data.get("metadata", {})
    if metadata:
        metadata_lines = _build_metadata_section(metadata)
        if metadata_lines:
            lines.extend(metadata_lines)

    # Join lines and normalize spacing
    content = "\n".join(lines)
    return _normalize_spacing(content)


def _normalize_spacing(content: str) -> str:
    """
    Normalize spacing in final output.

    Removes excessive blank lines while preserving document structure.
    Ensures max 2 consecutive blank lines anywhere in the document.

    Args:
        content: Text content to normalize

    Returns:
        Content with normalized spacing
    """
    # Replace 3+ consecutive newlines with just 2
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Clean up spacing around section markers
    content = re.sub(r"\n{2,}(---)\n{2,}", r"\n\n\1\n\n", content)

    # Ensure proper spacing before headings (should have blank line before)
    content = re.sub(r"\n(#{1,6}\s)", r"\n\n\1", content)

    # But remove triple newlines that might have been created
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


def _build_metadata_section(metadata: dict[str, Any]) -> list[str]:
    """Build metadata section from metadata dict."""
    lines = []
    lines.append("## Metadata")
    lines.append("")

    # Common metadata fields
    metadata_fields = {
        "description": "Description",
        "doc_type": "Document Type",
        "content_type": "Content Type",
        "difficulty": "Difficulty",
        "tags": "Tags",
        "categories": "Categories",
        "personas": "Target Audience",
        "author": "Author",
    }

    has_content = False
    for key, label in metadata_fields.items():
        if metadata.get(key):
            value = metadata[key]
            # Format lists
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"- **{label}**: {value}")
            has_content = True

    # Add last updated timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines.append(f"- **Last Updated**: {timestamp}")
    lines.append("")

    # Only return if we have actual metadata
    return lines if has_content else []
