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

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

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
        soup = BeautifulSoup(html_bytes, "lxml")
    except Exception:  # noqa: BLE001  (parser errors are common in CC)
        return rows

    for tag_name in DROP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()

    body = soup.body or soup
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

    # Iterative DOM walk.  Many real web pages have DOM depth > 1000
    # (deeply nested <div>s, malformed HTML).  A recursive walk would
    # blow Python's call stack; we use an explicit stack of
    # ``(element, action)`` pairs where ``action`` is either ``ENTER``
    # (process the node and push its children) or ``EXIT`` (run any
    # post-children logic, e.g. flush block boundaries).
    ENTER, EXIT = 0, 1
    stack: list[tuple[Any, int]] = [(body, ENTER)]
    try:
        while stack:
            elem, action = stack.pop()

            if action == EXIT:
                if isinstance(elem, Tag) and elem.name in BLOCK_TAGS:
                    flush_text()
                continue

            # action == ENTER
            if isinstance(elem, NavigableString):
                text = str(elem)
                if text.strip():
                    text_buffer.append(text.strip())
                continue

            if not isinstance(elem, Tag):
                continue

            if elem.name == "img":
                flush_text()
                raw_src = elem.get("src") or elem.get("data-src") or ""
                raw_src = raw_src.strip()
                if not raw_src:
                    continue
                try:
                    abs_url = urljoin(source_url, raw_src)
                except ValueError:
                    continue
                if not _should_keep_image_url(abs_url):
                    continue
                alt = (elem.get("alt") or "").strip() or None
                rows.append(_make_image_row(sample_id, position, abs_url, alt))
                position += 1
                continue

            is_block = elem.name in BLOCK_TAGS
            if is_block:
                flush_text()
            # Push EXIT before children so it runs AFTER they're done.
            stack.append((elem, EXIT))
            # Push children in REVERSE so they pop in DOM order.
            for child in reversed(list(elem.children)):
                stack.append((child, ENTER))
    except Exception:  # noqa: BLE001
        # Any unexpected parser / encoding failure: keep whatever we
        # already accumulated (metadata + any prior content rows) rather
        # than losing the whole document.
        return rows
    flush_text()

    return rows
