"""Pure-function HTML → interleaved-rows extractor.

This is the algorithm that turns one WARC response record (HTML bytes +
metadata) into a list of dict rows shaped like the InterleavedBatch
schema.  Kept as a pure function so it can be unit-tested independently
and called from outside Curator.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser, Node

# Tags whose entire subtree is dropped at extract time.
# Aligned with OBELICS §3.2 + magic-html's MANUALLY_CLEANED list.
DROP_TAGS: frozenset[str] = frozenset({
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "aside", "header", "form", "button",
    "select", "input", "textarea", "object", "embed",
})

# Tags that mark a block boundary — flush text before/after.
BLOCK_TAGS: frozenset[str] = frozenset({
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "article", "section", "blockquote", "tr", "td", "th",
    "figure", "figcaption", "main", "details", "summary", "pre",
})

# Image-URL substring blocklist applied at extract time.
URL_SUBSTRING_BLOCKLIST: tuple[str, ...] = (
    "logo", "icon", "button", "plugin", "widget",
    "avatar", "spacer", "1x1", "blank",
)

# Image-URL extension allowlist (anything else, e.g. svg/gif, dropped).
URL_EXT_ALLOWLIST: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


def _should_keep_image_url(url: str) -> bool:
    """Cheap pre-filter on the URL string itself; no network."""
    if not url or url.startswith(("data:", "javascript:", "#")):
        return False
    lower = url.lower()
    if any(bad in lower for bad in URL_SUBSTRING_BLOCKLIST):
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    path = parsed.path.lower()
    if not any(path.endswith(ext) for ext in URL_EXT_ALLOWLIST):
        # Allow extension-less URLs only if path looks like a real path.
        if "." in path.split("/")[-1]:
            return False
    return True


def _make_text_row(sample_id: str, position: int, text: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "position": position,
        "modality": "text",
        "content_type": "text/plain",
        "text_content": text,
        "binary_content": None,
        "source_ref": None,
        "materialize_error": None,
    }


def _make_image_row(
    sample_id: str, position: int, image_url: str, alt: str | None
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "position": position,
        "modality": "image",
        "content_type": None,  # filled by downloader later
        "text_content": alt,
        "binary_content": None,
        "source_ref": json.dumps({"url": image_url, "alt": alt}),
        "materialize_error": None,
    }


def _make_metadata_row(
    sample_id: str,
    source_url: str,
    warc_id: str,
    *,
    warc_file: str | None = None,
    warc_segment: str | None = None,
    snapshot: str | None = None,
    extractor: str | None = None,
) -> dict[str, Any]:
    """Build the doc's metadata row.  ``source_ref`` JSON carries lineage
    fields (Option A in the lineage design)."""
    lineage: dict[str, Any] = {
        "source_url": source_url,
        "warc_id": warc_id,
    }
    if warc_file is not None:
        lineage["warc_file"] = warc_file
    if warc_segment is not None:
        lineage["warc_segment"] = warc_segment
    if snapshot is not None:
        lineage["snapshot"] = snapshot
    if extractor is not None:
        lineage["extractor"] = extractor
    return {
        "sample_id": sample_id,
        "position": -1,
        "modality": "metadata",
        "content_type": "application/json",
        "text_content": None,
        "binary_content": None,
        "source_ref": json.dumps(lineage),
        "materialize_error": None,
    }


def warc_html_to_interleaved_rows(
    warc_record: dict[str, Any],
    *,
    min_text_chars: int = 1,
) -> list[dict[str, Any]]:
    """Convert one WARC response record into InterleavedBatch-shaped rows.

    Args:
        warc_record: dict with keys ``{url, warc_id, source_id, content}``
            (the shape Curator's :class:`CommonCrawlWarcIterator` yields).
        min_text_chars: drop text runs shorter than this many chars.

    Returns:
        List of row dicts in InterleavedBatch schema.  Always starts with
        a single metadata row at ``position=-1``.  If the page has no
        usable content the list contains only the metadata row.
    """
    sample_id = warc_record["warc_id"]
    source_url = warc_record["url"]
    html_bytes = warc_record["content"]

    rows: list[dict[str, Any]] = [
        _make_metadata_row(
            sample_id, source_url, sample_id,
            warc_file=warc_record.get("warc_file"),
            warc_segment=warc_record.get("warc_segment"),
            snapshot=warc_record.get("snapshot"),
            extractor=warc_record.get("extractor"),
        )
    ]

    if not html_bytes:
        return rows

    try:
        # selectolax wraps modest_html (C library) — much lighter than
        # bs4's Python wrapper layer and more lenient than lxml's HTML
        # parser on real-world malformed CC HTML.  Tree is built in
        # memory but allocations live C-side, so per-page peak is
        # ~5-10× smaller than bs4+lxml.
        html_str = (
            html_bytes if isinstance(html_bytes, str)
            else html_bytes.decode("utf-8", errors="replace")
        )
        tree = HTMLParser(html_str)
    except Exception:  # noqa: BLE001  (parser errors are common in CC)
        return rows

    root = tree.body or tree.root
    if root is None:
        return rows

    text_buffer: list[str] = []
    position = 0

    def flush_text() -> None:
        nonlocal position
        if not text_buffer:
            return
        text = " ".join(text_buffer).strip()
        text_buffer.clear()
        if len(text) < min_text_chars:
            return
        rows.append(_make_text_row(sample_id, position, text))
        position += 1

    # Iterative DOM walk over selectolax nodes.  Many real CC pages have
    # DOM depth > 1000 (deeply nested <div>s, malformed HTML), so we
    # avoid Python recursion via an explicit ``(node, action)`` stack
    # where ``action`` is ENTER (process the node, push its children)
    # or EXIT (run post-children logic, e.g. flush block boundaries).
    #
    # selectolax represents text nodes via tag == ``"-text"``; child
    # iteration with ``include_text=True`` yields these alongside
    # element children so we can emit text in document order.
    ENTER, EXIT = 0, 1
    stack: list[tuple[Node, int]] = [(root, ENTER)]
    try:
        while stack:
            node, action = stack.pop()
            tag = (node.tag or "").lower() if node.tag else ""

            if action == EXIT:
                if tag in BLOCK_TAGS:
                    flush_text()
                continue

            # action == ENTER
            if tag == "-text":
                txt = node.text(strip=False) or ""
                txt = txt.strip()
                if txt:
                    text_buffer.append(txt)
                continue

            if tag in DROP_TAGS:
                # Skip the entire subtree.
                continue

            if tag == "img":
                flush_text()
                attrs = node.attributes
                raw_src = (attrs.get("src") or attrs.get("data-src") or "").strip()
                if not raw_src:
                    continue
                try:
                    abs_url = urljoin(source_url, raw_src)
                except ValueError:
                    continue
                if not _should_keep_image_url(abs_url):
                    continue
                alt = (attrs.get("alt") or "").strip() or None
                rows.append(_make_image_row(sample_id, position, abs_url, alt))
                position += 1
                continue

            is_block = tag in BLOCK_TAGS
            if is_block:
                flush_text()
            # Push EXIT before children so it runs AFTER they're done.
            stack.append((node, EXIT))
            # Push children in REVERSE so they pop in DOM order.
            # iter(include_text=True) yields text nodes (tag="-text")
            # alongside element children, preserving document order.
            for child in reversed(list(node.iter(include_text=True))):
                stack.append((child, ENTER))
    except Exception:  # noqa: BLE001
        # Any unexpected parser / encoding failure: keep whatever we
        # already accumulated (metadata + any prior content rows) rather
        # than losing the whole document.
        return rows
    flush_text()

    return rows
