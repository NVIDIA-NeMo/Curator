"""magic-html based extractor variants.

`magic_html` strips nav / footer / boilerplate up front and returns
cleaned main-content HTML.  We then run the same DOM walker over that
cleaned HTML so image positions are still preserved in DOM order.

Two variants are exposed:

* ``warc_html_to_interleaved_rows_magic_html`` — magic-html only.
* ``warc_html_to_interleaved_rows_magic_traf`` — 2-tier fallback:
      magic-html → trafilatura(output_format='html').  Records where
      both tiers return empty yield only a metadata row (no naive
      walker on raw HTML).  rows[0]['source_ref'] JSON records the
      actual tier used under ``extractor``: one of
      ``{"magic_html", "trafilatura", "empty"}``.
"""
from __future__ import annotations

import threading
from typing import Any

from nemo_curator.stages.nemotron_cc_mm.extractor import warc_html_to_interleaved_rows

# --- magic-html singleton (per worker process) ---
_extractor_singleton = None
_extractor_lock = threading.Lock()

# --- trafilatura singleton (per worker process) ---
_trafilatura_extract = None
_trafilatura_settings = None
_trafilatura_lock = threading.Lock()


def _get_magic_html_extractor():
    """Per-process cached ``GeneralExtractor`` — constructor is non-trivial.

    NOTE: ImportError here means ``magic-html`` (and its quirky deps:
    ``py_asciimath``, ``readability-lxml``, ``tldextract``, ``faust-cchardet``)
    is not installed.  We let ImportError propagate so the caller can fail
    loudly — silently falling back would mask a misconfigured environment.
    """
    global _extractor_singleton
    if _extractor_singleton is None:
        with _extractor_lock:
            if _extractor_singleton is None:
                from magic_html import GeneralExtractor  # noqa: PLC0415
                _extractor_singleton = GeneralExtractor()
    return _extractor_singleton


def _get_trafilatura():
    """Per-process cached trafilatura ``extract`` + ``Extractor`` settings.

    ``output_format='html'`` is critical — returns cleaned HTML (with
    ``<img>`` tags preserved) instead of plain text, so the downstream
    DOM walker can still build image rows in DOM order.
    """
    global _trafilatura_extract, _trafilatura_settings
    if _trafilatura_extract is None:
        with _trafilatura_lock:
            if _trafilatura_extract is None:
                from trafilatura import extract
                from trafilatura.settings import Extractor
                _trafilatura_settings = Extractor(
                    output_format="html", comments=False,
                )
                _trafilatura_extract = extract
    return _trafilatura_extract, _trafilatura_settings


def _walk_with_extractor_tag(
    warc_record: dict[str, Any],
    cleaned_html_bytes: bytes,
    extractor_tag: str,
    *,
    min_text_chars: int,
) -> list[dict[str, Any]]:
    """Run the naive DOM walker on ``cleaned_html_bytes`` and tag the
    resulting metadata row's ``source_ref`` JSON with the actual tier
    name (``magic_html`` / ``trafilatura`` / ``naive``)."""
    record = {
        **warc_record,
        "content": cleaned_html_bytes,
        "extractor": extractor_tag,
    }
    return warc_html_to_interleaved_rows(record, min_text_chars=min_text_chars)


def _magic_html_clean(html_bytes: bytes, base_url: str) -> bytes:
    """Run magic-html and return cleaned HTML bytes (or empty on failure).

    ImportError (magic-html not installed) is NOT swallowed — we let it
    propagate so a misconfigured env fails loudly instead of silently
    masquerading 100% of records as magic-html failures.
    """
    if not html_bytes:
        return b""
    try:
        html_str = html_bytes.decode("utf-8", errors="replace")
        extractor = _get_magic_html_extractor()
        out = extractor.extract(html=html_str, base_url=base_url)
        cleaned = out.get("html") or ""
        return cleaned.encode("utf-8")
    except ImportError:
        raise
    except Exception:  # noqa: BLE001
        return b""


def _trafilatura_clean(html_bytes: bytes) -> bytes:
    """Run trafilatura with ``output_format='html'``. Returns cleaned
    HTML bytes (preserving ``<img>``) or empty bytes on failure/empty result.
    """
    if not html_bytes:
        return b""
    try:
        extract, settings = _get_trafilatura()
        html_str = html_bytes.decode("utf-8", errors="replace")
        cleaned = extract(html_str, options=settings)
        if cleaned:
            return cleaned.encode("utf-8")
        return b""
    except Exception:  # noqa: BLE001
        return b""


def warc_html_to_interleaved_rows_magic_html(
    warc_record: dict[str, Any],
    *,
    min_text_chars: int = 1,
) -> list[dict[str, Any]]:
    """Run magic-html for main-content extraction, then DOM-walk the cleaned HTML.

    Metadata row's ``source_ref`` JSON gets ``extractor="magic_html"``.
    """
    cleaned_bytes = _magic_html_clean(warc_record["content"], warc_record["url"])
    return _walk_with_extractor_tag(
        warc_record, cleaned_bytes, "magic_html",
        min_text_chars=min_text_chars,
    )


def warc_html_to_interleaved_rows_magic_traf(
    warc_record: dict[str, Any],
    *,
    min_text_chars: int = 1,
) -> list[dict[str, Any]]:
    """2-tier extraction (NO naive fallback): magic-html → trafilatura.

    Same as ``warc_html_to_interleaved_rows_hybrid`` except records where
    both magic-html and trafilatura return empty content yield ONLY the
    metadata row — we do NOT fall through to the naive walker on raw HTML.

    Use this variant to isolate whether the naive-walker-on-raw-HTML
    Tier-3 path is the perf bottleneck.

    rows[0]['source_ref'] JSON ``extractor``: ``{"magic_html", "trafilatura", "empty"}``.
    """
    # Tier 1: magic-html
    cleaned = _magic_html_clean(warc_record["content"], warc_record["url"])
    if cleaned:
        rows = _walk_with_extractor_tag(
            warc_record, cleaned, "magic_html",
            min_text_chars=min_text_chars,
        )
        if len(rows) > 1:
            return rows

    # Tier 2: trafilatura (HTML mode)
    cleaned = _trafilatura_clean(warc_record["content"])
    if cleaned:
        rows = _walk_with_extractor_tag(
            warc_record, cleaned, "trafilatura",
            min_text_chars=min_text_chars,
        )
        if len(rows) > 1:
            return rows

    # Both tiers empty — emit just the metadata row, no Tier-3 fallback.
    return _walk_with_extractor_tag(
        warc_record, b"", "empty",
        min_text_chars=min_text_chars,
    )
