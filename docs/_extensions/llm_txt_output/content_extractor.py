"""Content extraction functions for llm.txt generation."""

import re
from typing import Any
from urllib.parse import urljoin

from docutils import nodes
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Constants for content extraction thresholds
MIN_TITLE_LENGTH = 3
MIN_TEXT_LENGTH = 20
MAX_TITLE_LENGTH = 60
MAX_DESCRIPTION_LENGTH = 200  # For grid cards
MAX_DESCRIPTION_TRUNCATE = 197  # Leave room for "..."
MAX_KEY_SECTION_LENGTH = 350  # For key sections (longer for navigation)
MAX_KEY_SECTION_TRUNCATE = 347  # Leave room for "..."


def _clean_card_title(title: str) -> str:
    """
    Clean title text by removing octicon references and other artifacts.

    Args:
        title: Raw title text

    Returns:
        Cleaned title text
    """
    # Remove octicon with various formats
    title = re.sub(r"\{octicon\}[^{}`]+", "", title)  # {octicon}icon
    title = re.sub(r"`[^`]*octicon[^`]*`", "", title)  # `octicon...`
    title = re.sub(r"\{octicon\}`[^`]+`", "", title)  # {octicon}`icon`
    title = re.sub(r"octicon`[^`]+`", "", title)  # octicon`icon`
    # Clean remaining backticks
    title = re.sub(r"`+", "", title)
    return title.strip()


def _truncate_description(description: str, max_length: int = MAX_DESCRIPTION_LENGTH) -> str:
    """
    Truncate description text to specified length with ellipsis.

    Args:
        description: Description text to truncate
        max_length: Maximum length (default: MAX_DESCRIPTION_LENGTH)

    Returns:
        Truncated description with "..." if needed
    """
    if not description:
        return ""

    # Clean up description
    description = re.sub(r"\s+", " ", description).strip()

    if len(description) > max_length:
        truncate_at = max_length - 3  # Leave room for "..."
        return description[:truncate_at] + "..."

    return description


def _resolve_url(url: str, base_url: str) -> str:
    """
    Properly resolve relative URLs to absolute URLs using urllib.parse.urljoin.

    Args:
        url: Relative or absolute URL
        base_url: Base URL for resolution

    Returns:
        Absolute URL
    """
    if not url or not base_url:
        return url

    # Handle anchor-only links
    if url.startswith("#"):
        return url

    # Handle external links
    if url.startswith(("http://", "https://", "ftp://", "mailto:")):
        return url

    # Ensure base_url ends with / for proper urljoin behavior
    if not base_url.endswith("/"):
        base_url = base_url + "/"

    # Use urljoin for proper path resolution
    return urljoin(base_url, url)


def extract_document_content(env: BuildEnvironment, docname: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Extract content from document for llm.txt format."""
    try:
        doctree = env.get_doctree(docname)

        # Get title
        title = _extract_title(env, docname)

        # Check for grid cards first by parsing raw markdown (before MyST processing)
        base_url = settings.get("base_url", "")
        card_handling = settings.get("card_handling", "simple")
        grid_cards = []
        if card_handling == "smart":
            # Try extracting from raw markdown first
            grid_cards = _extract_grid_cards_from_markdown(env, docname, base_url)
            if not grid_cards:
                # Fallback to doctree extraction
                grid_cards = _extract_grid_cards(env, doctree, base_url)

        # Extract raw text content
        raw_text = _extract_text_content(doctree)

        # Clean text for LLM consumption
        if settings.get("clean_myst_artifacts", True):
            raw_text = clean_text_for_llm_txt(raw_text)

        # Extract summary
        summary = _extract_summary(doctree, settings.get("summary_sentences", 2))

        # Extract overview (first N chars of content)
        max_length = settings.get("max_content_length", 5000)
        overview = _extract_overview(raw_text, max_length)

        # Extract headings for key sections
        headings = []
        if settings.get("include_headings", True):
            headings = _extract_headings(doctree)

        # Extract related links
        links = []
        if settings.get("include_related_links", True):
            links = _extract_internal_links(doctree, settings.get("max_related_links", 10), base_url)

        # Get metadata
        metadata = {}
        if settings.get("include_metadata", True):
            metadata = _extract_metadata(env, docname)

        return {  # noqa: TRY300
            "title": title,
            "summary": summary,
            "overview": overview,
            "headings": headings,
            "grid_cards": grid_cards,
            "links": links,
            "metadata": metadata,
        }

    except Exception:
        logger.exception(f"Error extracting content from {docname}")
        return {
            "title": docname,
            "summary": "",
            "overview": "",
            "headings": [],
            "grid_cards": [],
            "links": [],
            "metadata": {},
        }


def _extract_title(env: BuildEnvironment, docname: str) -> str:
    """Extract document title."""
    if docname in env.titles:
        return env.titles[docname].astext().strip()
    return docname.split("/")[-1].replace("-", " ").replace("_", " ").title()


def _extract_text_content(doctree: nodes.document) -> str:
    """Extract plain text content from document tree."""
    text_parts = []

    for node in doctree.traverse():
        # Skip certain node types that aren't content
        if isinstance(node, (nodes.target, nodes.substitution_definition)):
            continue

        # Skip toctree and other directive content
        if hasattr(node, "tagname") and node.tagname in ["toctree", "index", "meta"]:
            continue

        # Extract text from text nodes
        if isinstance(node, nodes.Text):
            text = node.astext().strip()
            if text and not text.startswith("¶"):  # Skip permalink symbols
                text_parts.append(text)

    # Join and clean up the text
    full_text = " ".join(text_parts)
    full_text = re.sub(r"\s+", " ", full_text)

    return full_text.strip()


def clean_text_for_llm_txt(text: str) -> str:
    """Clean text content for llm.txt format (remove MyST artifacts)."""
    if not text:
        return ""

    # Remove SVG content
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove HTML line breaks (common in tables and grid cards)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Remove escaped backslashes (MyST artifacts)
    text = re.sub(r"\s*\\\\\s*", " ", text)

    # Remove other common HTML tags that might appear
    text = re.sub(r"<hr\s*/?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<div[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "", text, flags=re.IGNORECASE)

    # Remove octicon references
    text = re.sub(r"\{octicon\}[^{}`]+", "", text)
    text = re.sub(r"octicon`[^`]+`", "", text)

    # Remove empty directive blocks
    text = re.sub(r"^\s*```\{[^}]+\}\s*```\s*$", "", text, flags=re.MULTILINE)

    # Remove toctree artifacts
    text = re.sub(r"^\s*:caption:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:hidden:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:glob:\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:maxdepth:\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove MyST directive markers
    text = re.sub(r"^\s*:::\{[^}]+\}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:::\s*$", "", text, flags=re.MULTILINE)

    # Remove directive options
    text = re.sub(r"^\s*:ref-type:\s*\w+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:link-type:\s*\w+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:color:\s*\w+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:class:\s*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:link:\s*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:gutter:\s*.*$", "", text, flags=re.MULTILINE)

    # Convert badges to parentheses (preserve meaningful info)
    text = re.sub(r"\{bdg-\w+\}`([^`]+)`", r"(\1)", text)

    # Clean up code block language indicators
    text = re.sub(r"```(\w+)\s*\n", "```\n", text)

    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Multiple line breaks -> double
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs -> single space
    text = re.sub(r" +", " ", text)  # Clean up any remaining multiple spaces

    # Remove lines that are just punctuation or symbols
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and re.search(r"[a-zA-Z0-9]", stripped):
            # Remove standalone punctuation at start/end
            stripped = re.sub(r"^[^\w\s]+\s*", "", stripped)
            stripped = re.sub(r"\s*[^\w\s]+$", "", stripped)
            if stripped:
                cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    return text.strip()


def break_into_paragraphs(text: str, max_paragraph_length: int = 400) -> str:
    """
    Break long text blocks into readable paragraphs.

    Args:
        text: Text to break into paragraphs
        max_paragraph_length: Target maximum length for each paragraph

    Returns:
        Text with paragraph breaks inserted at sentence boundaries
    """
    if not text or len(text) < max_paragraph_length:
        return text

    # Preserve existing paragraph breaks
    existing_paragraphs = re.split(r"\n\n+", text)

    result = []
    for para in existing_paragraphs:
        # If paragraph is short enough, keep as-is
        if len(para) <= max_paragraph_length:
            result.append(para)
            continue

        # Split long paragraphs at sentence boundaries
        # Improved regex that avoids splitting on common abbreviations
        # Matches sentence endings followed by whitespace and capital letter or digit
        # But not after common abbreviations like Dr., Mr., U.S., etc.
        sentences = re.split(
            r"(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|i\.e|e\.g|Fig|Vol|No|Ph\.D|U\.S|U\.K))"
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            para,
        )

        current_block = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max length and we have content
            if current_length + sentence_length > max_paragraph_length and current_block:
                result.append(" ".join(current_block))
                current_block = [sentence]
                current_length = sentence_length
            else:
                current_block.append(sentence)
                current_length += sentence_length + 1  # +1 for space

        # Add remaining sentences
        if current_block:
            result.append(" ".join(current_block))

    return "\n\n".join(result)


def _extract_summary(doctree: nodes.document, num_sentences: int = 2) -> str:
    """Extract summary from first substantial paragraph."""
    min_length = 50

    # Try to find the first substantial paragraph
    for node in doctree.traverse(nodes.paragraph):
        text = node.astext().strip()
        if text and len(text) > min_length:
            # Clean the text
            text = re.sub(r"\s+", " ", text)

            # Extract first N sentences
            sentences = re.split(r"[.!?]+\s+", text)
            summary_sentences = sentences[:num_sentences]
            summary = ". ".join(summary_sentences)

            # Add period if missing
            if summary and not summary.endswith((".", "!", "?")):
                summary += "."

            return summary

    return ""


def _extract_overview(text: str, max_length: int) -> str:
    """Extract overview from text with length limit."""
    if not text:
        return ""

    if max_length > 0 and len(text) > max_length:
        # Try to cut at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        if last_period > max_length * 0.8:  # If we found a period in last 20%
            return truncated[: last_period + 1]
        else:
            return truncated + "..."

    return text


def _extract_headings(doctree: nodes.document) -> list[dict[str, str]]:
    """Extract headings from document tree."""
    headings = []

    # Extract headings from section nodes
    for node in doctree.traverse(nodes.section):
        title_node = node.next_node(nodes.title)
        if title_node:
            title_text = title_node.astext().strip()
            if title_text:
                # Determine heading level
                level = 1
                parent = node.parent
                while parent and isinstance(parent, nodes.section):
                    level += 1
                    parent = parent.parent

                # Try to get a preview of the section content
                preview = _extract_section_preview(node, max_chars=MAX_KEY_SECTION_LENGTH)

                headings.append({"text": title_text, "level": level, "preview": preview})

    # Remove duplicates while preserving order
    seen = set()
    unique_headings = []
    for heading in headings:
        heading_key = (heading["text"], heading["level"])
        if heading_key not in seen:
            seen.add(heading_key)
            unique_headings.append(heading)

    return unique_headings


def _resolve_reference_target(env: BuildEnvironment, target: str) -> str | None:
    """
    Resolve a MyST reference target to its actual document path.

    Args:
        env: Sphinx build environment
        target: Reference target (e.g., 'gs-text')

    Returns:
        Document path (e.g., 'get-started/text') or None if not found
    """
    try:
        # Check if target exists in Sphinx's standard domain labels
        if hasattr(env, "domaindata"):
            std_domain = env.domaindata.get("std", {})
            labels = std_domain.get("labels", {})

            # Labels map: target -> (docname, labelid, title)
            if target in labels:
                docname, _labelid, _title = labels[target]
                return docname

            # Also check anonlabels for anonymous references
            anonlabels = std_domain.get("anonlabels", {})
            if target in anonlabels:
                docname, _labelid = anonlabels[target]
                return docname

    except (AttributeError, KeyError, TypeError) as e:
        logger.debug(f"Error resolving reference target '{target}': {e}")

    return None


def _extract_grid_cards_from_markdown(env: BuildEnvironment, docname: str, base_url: str = "") -> list[dict[str, str]]:
    """Extract grid-item-card directives directly from raw markdown source."""
    cards = []

    try:
        # Get the source file path
        source_path = env.doc2path(docname)
        if not source_path or not source_path.exists():
            logger.debug(f"Source file not found for {docname}")
            return []

        # Read raw markdown
        with open(source_path, encoding="utf-8") as f:
            content = f.read()

        # Look for grid-item-card directives using regex
        # Pattern: :::{grid-item-card} [icon] Title
        #          :link: target
        #          :link-type: doc
        #
        #          Description text
        #          :::

        card_pattern = r":::\{grid-item-card\}([^\n]*)\n(.*?):::"
        matches = re.finditer(card_pattern, content, re.DOTALL)

        for match in matches:
            title_line = match.group(1).strip()
            card_body = match.group(2).strip()

            # Extract title and clean using helper function
            title = _clean_card_title(title_line)

            # Extract link target
            link_match = re.search(r":link:\s*([^\n]+)", card_body)
            link = link_match.group(1).strip() if link_match else ""

            # Check link type to see if it's a reference
            link_type_match = re.search(r":link-type:\s*([^\n]+)", card_body)
            link_type = link_type_match.group(1).strip() if link_type_match else ""

            # Extract description (text after the directive options)
            # Remove directive options first
            desc_text = re.sub(r":[a-z-]+:.*?(?=\n|$)", "", card_body, flags=re.MULTILINE)

            # Remove card footer (the +++ separator and anything after it)
            if "+++" in desc_text:
                desc_text = desc_text.split("+++")[0]

            description = desc_text.strip()

            # Clean up description - remove template variables and format better
            if description:
                description = re.sub(
                    r"\{\{[^}]+\}\}", "NeMo Evaluator", description
                )  # Replace {{ product_name_short }}
                description = re.sub(r"\{bdg-[^}]+\}", "", description)  # Remove badge directives
                description = re.sub(r"`+", "", description)  # Remove backticks
                description = re.sub(r"^\*\*([^*]+)\*\*\s*-?\s*", r"\1 - ", description)  # Handle **Bold** - text
                description = _truncate_description(description)

            # Convert link to absolute URL only if it's an internal link
            if link and not link.startswith(("http://", "https://", "ftp://", "mailto:")):
                # If link-type is 'ref', resolve the reference target to actual document path
                if link_type == "ref":
                    resolved_doc = _resolve_reference_target(env, link)
                    if resolved_doc:
                        link = resolved_doc
                    else:
                        logger.debug(f"Could not resolve reference target: {link}")

                # Add .html extension if not present
                if not link.endswith((".html", ".htm")):
                    link = link + ".html"

                # Use helper function for proper URL resolution
                link = _resolve_url(link, base_url)

            if title and link:
                cards.append({"title": title, "description": description, "url": link})

    except (OSError, AttributeError, ValueError) as e:
        logger.warning(f"Error extracting grid cards from markdown for {docname}: {e}")

    return cards


def _extract_grid_cards(env: BuildEnvironment, doctree: nodes.document, base_url: str = "") -> list[dict[str, str]]:
    """
    Extract grid-item-card elements by finding reference patterns.

    Since MyST grid cards create complex nested structures, we use a simpler
    approach: find all internal reference links and their surrounding context.
    """
    cards = []
    seen_urls = set()

    # Collect all paragraphs with their text
    all_paragraphs = {}
    for para in doctree.traverse(nodes.paragraph):
        para_text = para.astext().strip()
        if para_text:
            all_paragraphs[id(para)] = para_text

    # Find all internal reference links and pending cross-references
    all_refs = list(doctree.traverse(nodes.reference))

    # Also check for pending_xref nodes (MyST grid cards use these)
    try:
        from sphinx.addnodes import pending_xref

        pending_refs = list(doctree.traverse(pending_xref))
    except ImportError:
        pending_refs = []

    # Process regular references
    for ref in all_refs:
        if not hasattr(ref, "attributes"):
            continue

        attrs = ref.attributes
        if "refuri" not in attrs:
            continue

        link = attrs["refuri"]

        # Skip external links
        if link.startswith(("http://", "https://", "ftp://", "mailto:", "#")):
            continue

        # Skip if we've already seen this URL
        if link in seen_urls:
            continue
        seen_urls.add(link)

        # Get title from link text and clean using helper function
        title = _clean_card_title(ref.astext().strip())

        if not title or len(title) < MIN_TITLE_LENGTH:
            continue

        # Convert to absolute URL using helper function
        link = _resolve_url(link, base_url)

        # Try to find a description in the same or nearby paragraph
        description = ""
        parent = ref.parent

        # Check if parent paragraph has more text
        if isinstance(parent, nodes.paragraph):
            parent_text = _clean_card_title(parent.astext().strip())

            # If parent has more text than just the title, use it as description
            if len(parent_text) > len(title) + 10:
                # Remove the title part
                description = parent_text.replace(title, "").strip()
                # Clean up common separators
                description = re.sub(r"^[\s\-:•]+", "", description)
                description = re.sub(r"[\s\-:•]+$", "", description)

        # Look for description in next sibling paragraph
        if not description and parent and hasattr(parent, "parent"):
            try:
                parent_node = parent.parent
                if hasattr(parent_node, "children"):
                    parent_idx = list(parent_node.children).index(parent)
                    if parent_idx + 1 < len(parent_node.children):
                        next_node = parent_node.children[parent_idx + 1]
                        if isinstance(next_node, nodes.paragraph):
                            desc_text = _clean_card_title(next_node.astext().strip())
                            if desc_text and desc_text != title:
                                description = desc_text
            except (ValueError, IndexError, AttributeError):
                pass

        # Truncate long descriptions using helper function
        description = _truncate_description(description)

        cards.append({"title": title, "description": description, "url": link})

    # Process pending_xref nodes (MyST grid card links)
    for pxref in pending_refs:
        if not hasattr(pxref, "attributes"):
            continue

        attrs = pxref.attributes
        # pending_xref uses 'reftarget' for the link destination
        if "reftarget" in attrs:
            link = attrs["reftarget"]

            # Skip if already seen
            if link in seen_urls:
                continue
            seen_urls.add(link)

            # Get title from the text content and clean using helper function
            title = _clean_card_title(pxref.astext().strip())

            if not title or len(title) < MIN_TITLE_LENGTH:
                continue

            # Resolve reference target if needed
            reftype = attrs.get("reftype", "")
            if reftype in {"ref", ""}:
                resolved_doc = _resolve_reference_target(env, link)
                if resolved_doc:
                    link = resolved_doc

            # Add .html extension if not present
            if not link.endswith((".html", ".htm")):
                link = link + ".html"

            # Use helper function for proper URL resolution
            link = _resolve_url(link, base_url)

            # Try to find description
            description = ""
            parent = pxref.parent
            if isinstance(parent, nodes.paragraph):
                parent_text = _clean_card_title(parent.astext().strip())

                if len(parent_text) > len(title) + 10:
                    description = parent_text.replace(title, "").strip()
                    description = re.sub(r"^[\s\-:•]+", "", description)
                    description = re.sub(r"[\s\-:•]+$", "", description)

            # Truncate description using helper function
            description = _truncate_description(description)

            cards.append({"title": title, "description": description, "url": link})

    return cards


def _extract_card_info(card_node: nodes.container, base_url: str = "") -> dict[str, str] | None:
    """Extract title, description, and link from a grid card."""
    card_title = ""
    card_description = ""
    card_link = ""

    # Get all text from the card first and clean using helper function
    full_text = _clean_card_title(card_node.astext().strip())
    logger.debug(f"Card full text: {full_text[:100]}")

    # Extract link first - this is most reliable
    for node in card_node.traverse(nodes.reference):
        if hasattr(node, "attributes"):
            attrs = node.attributes
            # Get link text as potential title and clean using helper function
            link_text = _clean_card_title(node.astext().strip())
            if link_text and not card_title:
                card_title = link_text

            # Get URL
            if "refuri" in attrs:
                link = attrs["refuri"]
                # Skip external links
                if not link.startswith(("http://", "https://", "ftp://", "mailto:")):
                    card_link = _resolve_url(link, base_url)
                    logger.debug(f"Found card link: {card_link}")
                    break

    # Try to extract title from strong/emphasis nodes if not found
    if not card_title:
        for node in card_node.traverse():
            if isinstance(node, (nodes.strong, nodes.emphasis)):
                text = _clean_card_title(node.astext().strip())
                if text and len(text) > MIN_TITLE_LENGTH:
                    card_title = text
                    break

    # Fallback: use first line of text
    if not card_title and full_text:
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        if lines:
            card_title = lines[0]
            # If title is too long, it might be the description, take first few words
            if len(card_title) > MAX_TITLE_LENGTH:
                words = card_title.split()[:5]
                card_title = " ".join(words)

    # Extract description (all paragraphs)
    paragraphs = []
    for node in card_node.traverse(nodes.paragraph):
        text = _clean_card_title(node.astext().strip())

        # Don't include text that's part of the title
        if text and text != card_title and not card_title.startswith(text):
            paragraphs.append(text)

    if paragraphs:
        card_description = " ".join(paragraphs)
        # Truncate using helper function
        card_description = _truncate_description(card_description)

    logger.debug(
        f"Card extraction result - Title: {card_title}, Link: {card_link}, Desc: {card_description[:50] if card_description else 'None'}"
    )

    # Only return if we have at least a title or link
    if card_title or card_link:
        if not card_title:
            card_title = "Link"  # Fallback
        return {"title": card_title, "description": card_description, "url": card_link}

    return None


def _extract_section_preview(section_node: nodes.section, max_chars: int = 150) -> str:
    """Extract a preview of section content."""
    # Get first paragraph under this section
    for node in section_node.traverse(nodes.paragraph):
        text = node.astext().strip()
        if text and len(text) > MIN_TEXT_LENGTH:
            # Clean and truncate
            text = re.sub(r"\s+", " ", text)
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            return text

    return ""


def _extract_internal_links(doctree: nodes.document, max_links: int = 10, base_url: str = "") -> list[dict[str, str]]:
    """Extract internal links from document and convert to absolute URLs."""
    links = []
    seen_urls = set()

    for node in doctree.traverse(nodes.reference):
        if len(links) >= max_links:
            break

        link_text = node.astext().strip()
        if not link_text:
            continue

        link_url = ""
        if hasattr(node, "attributes"):
            attrs = node.attributes
            if "refuri" in attrs:
                link_url = attrs["refuri"]
                # Skip external links
                if link_url.startswith(("http://", "https://", "ftp://", "mailto:")):
                    continue
            elif "refid" in attrs:
                link_url = "#" + attrs["refid"]
            elif "reftarget" in attrs:
                link_url = attrs["reftarget"]

        # Convert relative URLs to absolute URLs using helper function
        if link_url and not link_url.startswith(("http://", "https://", "#")):
            link_url = _resolve_url(link_url, base_url)

        # Add if we have valid text and URL, and haven't seen it before
        if link_text and link_url and link_url not in seen_urls:
            links.append({"text": link_text, "url": link_url})
            seen_urls.add(link_url)

    return links


def _extract_metadata(env: BuildEnvironment, docname: str) -> dict[str, Any]:
    """Extract metadata from document."""
    metadata = {}

    try:
        # Get metadata from Sphinx environment
        if hasattr(env, "metadata") and docname in env.metadata:
            metadata.update(env.metadata[docname])

        # Try to extract frontmatter if it's a markdown file
        source_path = env.doc2path(docname)
        if source_path and str(source_path).endswith(".md"):
            frontmatter = _extract_frontmatter(str(source_path))
            if frontmatter:
                metadata.update(frontmatter)

    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not extract metadata from {docname}: {e}")

    return metadata


def _extract_frontmatter(file_path: str) -> dict[str, Any] | None:
    """Extract YAML frontmatter from markdown files."""
    try:
        import yaml

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Normalize line endings to handle Windows CRLF
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        if content.startswith("---"):
            end_marker = content.find("\n---\n", 3)
            if end_marker != -1:
                frontmatter_text = content[3:end_marker]
                return yaml.safe_load(frontmatter_text)

    except ImportError:
        logger.debug("PyYAML not available, skipping frontmatter extraction")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Could not extract frontmatter from {file_path}: {e}")
    except Exception as e:  # noqa: BLE001
        # Catch any other errors (including YAML parsing errors)
        # We use broad exception handling here because yaml.YAMLError
        # is not available if PyYAML is not installed
        logger.debug(f"Could not extract frontmatter from {file_path}: {e}")
    return None
