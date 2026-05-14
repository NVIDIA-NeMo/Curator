"""magic-html based extractor variants.

`magic_html` strips nav / footer / boilerplate up front and returns
cleaned main-content HTML.  We then run the same DOM walker over that
cleaned HTML so image positions are still preserved in DOM order.

Two variants are exposed:

* ``warc_html_to_interleaved_rows_magic_html`` — clean-only.
* ``warc_html_to_interleaved_rows_hybrid``     — magic-html first; if it
  produces no content rows (a known failure mode on diverse CC data),
  fall back to the naive walker on the raw HTML.
"""
from __future__ import annotations

import threading
from typing import Any

from nemo_curator.stages.nemotron_cc_mm.extractor import warc_html_to_interleaved_rows

_extractor_singleton = None
_extractor_lock = threading.Lock()


def _get_magic_html_extractor():
    """Per-process cached ``GeneralExtractor`` — constructor is non-trivial."""
    global _extractor_singleton
    if _extractor_singleton is None:
        with _extractor_lock:
            if _extractor_singleton is None:
                from magic_html import GeneralExtractor
                _extractor_singleton = GeneralExtractor()
    return _extractor_singleton


def warc_html_to_interleaved_rows_magic_html(
    warc_record: dict[str, Any],
    *,
    min_text_chars: int = 1,
) -> list[dict[str, Any]]:
    """Run magic-html for main-content extraction, then DOM-walk the cleaned HTML.

    Returns the same row shape as :func:`warc_html_to_interleaved_rows`.
    On any magic-html error, falls through to the walker with empty HTML
    (the walker still emits the mandatory metadata row).
    """
    html_bytes = warc_record["content"]
    cleaned_bytes: bytes = b""
    if html_bytes:
        try:
            html_str = html_bytes.decode("utf-8", errors="replace")
            extractor = _get_magic_html_extractor()
            out = extractor.extract(html=html_str, base_url=warc_record["url"])
            cleaned_html = out.get("html") or ""
            cleaned_bytes = cleaned_html.encode("utf-8")
        except Exception:  # noqa: BLE001
            cleaned_bytes = b""

    cleaned_record = {**warc_record, "content": cleaned_bytes}
    return warc_html_to_interleaved_rows(
        cleaned_record, min_text_chars=min_text_chars
    )


def warc_html_to_interleaved_rows_hybrid(
    warc_record: dict[str, Any],
    *,
    min_text_chars: int = 1,
) -> list[dict[str, Any]]:
    """magic-html first; fall back to naive walker if the page comes back empty.

    ``rows[0]`` is always the metadata row.  If that's the only row,
    magic-html produced no usable content — try the raw-HTML walker
    before declaring the document empty.
    """
    rows = warc_html_to_interleaved_rows_magic_html(
        warc_record, min_text_chars=min_text_chars
    )
    if len(rows) <= 1:
        return warc_html_to_interleaved_rows(
            warc_record, min_text_chars=min_text_chars
        )
    return rows
