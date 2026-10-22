# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import pyarrow as pa
from markdown_it import MarkdownIt
from markdown_it.common.utils import unescapeAll
from markdown_it.helpers import parseLinkDestination, parseLinkLabel, parseLinkTitle
from markdown_it.rules_inline import StateInline

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


def _resolve_url(raw: str | None, page_url: str | None) -> tuple[str | None, str]:
    if raw is None or not raw.strip():
        return None, "missing_url"
    raw = raw.strip()
    try:
        parts = urlsplit(raw)
        if parts.scheme and parts.scheme.lower() not in ("http", "https"):
            return None, "unsupported_scheme"
        if not parts.scheme:
            base = urlsplit(page_url or "")
            if base.scheme.lower() not in ("http", "https") or not base.hostname:
                return None, "missing_page_base"
        resolved = raw if parts.scheme else urljoin(page_url, raw)
        parts = urlsplit(resolved)
        if not parts.hostname or any(c.isspace() or c < " " for c in resolved):
            return None, "invalid_url"
        # Validate malformed ports too. Keep query order, case, escapes and fragments.
        _ = parts.port
        return resolved, "absolute" if urlsplit(raw).scheme else "resolved_relative"
    except ValueError:
        return None, "invalid_url"


# Use text/escape rules for labels; locate image markers ourselves so code and
# escaped examples remain visible as mentions.
LABEL_PARSER = MarkdownIt("zero").enable("escape")
DEFINITION = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)\]:[ \t]*", re.MULTILINE)
SPACE = re.compile(r"\s*")
BBCODE_IMAGE = re.compile(r"\[img\]([^\[\]]*)\[/img\]", re.IGNORECASE)
TAG = re.compile(r"<(img|source)\b(?:[^\"'<>]|\"[^\"]*\"|'[^']*')*>", re.IGNORECASE)
ATTR = re.compile(r"""([^\s=<>/]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([“”][^“”]*[“”]|[‘’][^‘’]*[‘’])|([^\s>]+))""")  # noqa: RUF001 - recognize typographic quotes without repairing URLs
SINGLE = {
    "src",
    "data-src",
    "data-original",
    "data-lazy-src",
    "data-original-src",
    "data-image",
    "data-url",
    "data-high-res-src",
    "data-thumb-src",
}
SETS = {"srcset", "data-srcset"}


class ImageTags(HTMLParser):
    """Tolerant fallback with offsets into the original, unmodified source."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
        self.tags = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("img", "source"):
            line, column = self.getpos()
            self.tags.append((tag, self.get_starttag_text(), self.line_starts[line - 1] + column))

    handle_startendtag = handle_starttag


def srcset_candidates(value: str) -> Iterator[tuple[str, str]]:
    """Tokenize URL/descriptor candidates; preserve commas inside URL tokens.

    Keeps candidates even with invalid descriptors; it does not
    implement browser source selection or descriptor validity rules.
    """
    i = 0
    while i < len(value):
        while i < len(value) and (value[i].isspace() or value[i] == ","):
            i += 1
        start = i
        while i < len(value) and not value[i].isspace():
            i += 1
        url = value[start:i]
        if not url:
            return
        if url.endswith(","):
            yield url.rstrip(","), ""
            continue
        start = i
        depth = 0
        while i < len(value):
            if value[i] == "," and depth == 0:
                break
            depth += (value[i] == "(") - (value[i] == ")" and depth > 0)
            i += 1
        yield url, value[start:i].strip()
        i += 1


def extract_image_references(text: str | None, page_url: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # noqa: C901, PLR0912, PLR0915
    """Extract supported image mentions, including code/escaped syntax, without I/O.

    Returns ordered occurrences and uncovered-syntax diagnostics. Offsets are
    half-open Unicode codepoints into text. HTML offsets cover attributes;
    srcset alternatives share that span and retain source_attribute_raw.
    Ambiguous/unsupported destinations retain raw values with null URLs.
    This is not exhaustive discovery of CSS, JavaScript or arbitrary bare URLs.
    """
    text = text or ""
    rows, covered = [], set()

    def add(  # noqa: PLR0913
        raw: str,
        kind: str,
        start: int,
        end: int,
        *,
        descriptor: str | None = None,
        review_status: str | None = None,
        decoded: str | None = None,
        source_attribute_raw: str | None = None,
    ) -> None:
        if decoded is None:
            decoded = unescapeAll(raw) if kind.startswith("markdown") else html.unescape(raw)
        url, status = _resolve_url(decoded, page_url)
        candidate_url = None
        if review_status:
            if review_status == "ambiguous_typographic_quotes":
                candidate_url, _ = _resolve_url(html.unescape(raw[1:-1]), page_url)
            elif review_status == "ambiguous_media_source":
                candidate_url = url
            url, status = None, review_status
        elif "<" in decoded or ">" in decoded:
            url, status = None, "invalid_destination_markup"
        rows.append(
            {
                "raw": raw,
                "url": url,
                "status": status,
                "kind": kind,
                "start": start,
                "end": end,
                "descriptor": descriptor,
                "candidate_url": candidate_url,
            }
        )
        if source_attribute_raw is not None:
            rows[-1]["source_attribute_raw"] = source_attribute_raw

    definitions = {}

    def normalize(value: str) -> str:
        return " ".join(unescapeAll(value).split()).casefold()

    for match in DEFINITION.finditer(text):
        dest = parseLinkDestination(text, match.end(), len(text))
        if dest.ok:
            raw = text[match.end() : dest.pos]
            definitions.setdefault(normalize(match[1]), raw[1:-1] if raw.startswith("<") else raw)

    state = StateInline(text, LABEL_PARSER, {}, [])
    marker_starts = {m.start() for m in re.finditer(r"!\[", text)}
    unresolved = 0
    for start in sorted(marker_starts):
        label_end = parseLinkLabel(state, start + 1)
        if label_end < 0:
            continue
        pos = label_end + 1
        if text[pos : pos + 1] == "(":
            dest_start = SPACE.match(text, pos + 1).end()
            dest = parseLinkDestination(text, dest_start, len(text))
            if not dest.ok and text[dest_start : dest_start + 1] != ")":
                continue
            dest_end = dest.pos if dest.ok else dest_start
            end = SPACE.match(text, dest_end).end()
            if end > dest_end and text[end : end + 1] != ")":
                title = parseLinkTitle(text, end, len(text))
                if not title.ok:
                    continue
                end = SPACE.match(text, title.pos).end()
            if text[end : end + 1] != ")":
                continue
            raw = text[dest_start:dest_end]
            add(raw[1:-1] if raw.startswith("<") else raw, "markdown_inline", start, end + 1)
        else:
            label = text[start + 2 : label_end]
            if text[pos : pos + 1] == "[":
                ref_end = parseLinkLabel(state, pos)
                if ref_end >= 0:
                    label = text[pos + 1 : ref_end] or label
                    pos = ref_end + 1
            label = normalize(label)
            if label not in definitions:
                unresolved += 1
                continue
            add(definitions[label], "markdown_reference", start, pos)
        covered.add(start)
    attribute_counts = Counter()
    bbcode_starts = set()
    for match in BBCODE_IMAGE.finditer(text):
        # BBCode destinations are literal: do not apply Markdown/HTML unescaping.
        add(match[1], "bbcode_img", *match.span(), decoded=match[1])
        bbcode_starts.add(match.start())
    tags = [(m[1].lower(), m[0], m.start()) for m in TAG.finditer(text)]
    matched_starts = {start for _, _, start in tags}
    tag_starts = {m.start() for m in re.finditer(r"<(?:img|source)\b", text, re.IGNORECASE)}
    fallback_starts = set()
    fallback_errors = 0
    if tag_starts - matched_starts:
        parser = ImageTags(text)
        try:
            parser.feed(text)
            parser.close()
        except (AssertionError, ValueError):
            # Preserve successful mentions and flag the source row for retry.
            fallback_errors = 1
        for tag in parser.tags:
            if tag[2] not in matched_starts:
                tags.append(tag)
                fallback_starts.add(tag[2])
                matched_starts.add(tag[2])
    for tag_name, markup, tag_start in tags:
        first_row = len(rows)
        attributes = list(ATTR.finditer(markup))
        declared_type = next(
            (
                next(v for v in a.groups()[1:] if v is not None).strip().lower()
                for a in attributes
                if a[1].lower() == "type"
            ),
            "",
        )
        for attr in attributes:
            name = attr[1].lower()
            value = next(v for v in attr.groups()[1:] if v is not None)
            attribute_counts[name] += 1
            kind = "html_" + tag_name + "." + name
            review_status = None
            if tag_name == "source" and declared_type.startswith(("video/", "audio/")):
                review_status = "excluded_non_image_media"
            elif attr[4] is not None:
                review_status = "ambiguous_typographic_quotes"
            elif tag_name == "source" and name in SINGLE and not declared_type.startswith("image/"):
                review_status = "ambiguous_media_source"
            if name in SINGLE:
                add(value, kind, tag_start + attr.start(), tag_start + attr.end(), review_status=review_status)
            elif name in SETS:
                if review_status:
                    add(value, kind, tag_start + attr.start(), tag_start + attr.end(), review_status=review_status)
                    continue
                for url, descriptor in srcset_candidates(html.unescape(value)):
                    # Tokenization follows one HTML decode; retain the original
                    # attribute and apply the same URL checks as other sources.
                    add(
                        url,
                        kind,
                        tag_start + attr.start(),
                        tag_start + attr.end(),
                        descriptor=descriptor,
                        decoded=url,
                        source_attribute_raw=value,
                    )
        if tag_start in fallback_starts:
            for row in rows[first_row:]:
                row["extraction_method"] = "html_parser_fallback"
                row["source_tag_start"] = tag_start
    rows.sort(key=lambda row: (row["start"], row["end"]))
    return rows, {
        "uncovered_markers": len(marker_starts - covered),
        "unresolved_reference_usages": unresolved,
        "uncovered_html_tags": len(tag_starts - matched_starts),
        "html_fallback_errors": fallback_errors,
        "uncovered_bbcode_images": sum(
            m.start() not in bbcode_starts for m in re.finditer(r"\[img\]", text, re.IGNORECASE)
        ),
        "review_candidates": sum(r["status"].startswith(("ambiguous_", "invalid_destination_")) for r in rows),
        "html_attribute_counts": attribute_counts,
    }


_REFERENCE_TYPE = pa.list_(
    pa.struct(
        [
            *(
                (name, pa.string())
                for name in (
                    "raw",
                    "url",
                    "status",
                    "kind",
                    "descriptor",
                    "candidate_url",
                    "source_attribute_raw",
                    "extraction_method",
                )
            ),
            *((name, pa.int64()) for name in ("start", "end", "source_tag_start")),
        ]
    )
)
_DIAGNOSTICS_TYPE = pa.struct(
    [
        *(
            (name, pa.int64())
            for name in (
                "uncovered_markers",
                "unresolved_reference_usages",
                "uncovered_html_tags",
                "html_fallback_errors",
                "uncovered_bbcode_images",
                "review_candidates",
            )
        ),
        ("html_attribute_counts", pa.map_(pa.string(), pa.int64())),
    ]
)


@dataclass
class ImageReferenceExtractor(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Attach image mentions to Markdown/HTML documents without dropping rows.

    Adds ``image_references`` (list of structs) and
    ``image_reference_diagnostics`` (struct), compatible with ParquetWriter.
    All input columns are retained, including caller-supplied source file/row
    IDs. The stage does not infer physical Parquet offsets or deduplicate URLs.
    Use separate adapters for structured JSON/interleaved input.

    Example::

        pipeline.add_stage(ParquetReader(file_paths=input_paths))
        pipeline.add_stage(ImageReferenceExtractor(text_field="text", url_field="url"))
        pipeline.add_stage(ParquetWriter(path=output_path))

    Requires the ``text_cpu`` extra. Relative URLs use the page URL; HTML base
    elements are not interpreted. Empty/null text yields an empty reference
    list. Diagnostics flag recognized but uncovered syntax, not exhaustive recall.
    """

    text_field: str = "text"
    url_field: str = "url"
    name: str = "image_reference_extractor"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, self.url_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_references", "image_reference_diagnostics"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        table = batch.to_pyarrow()
        if set(self.outputs()[1]) & set(table.column_names):
            msg = "Image reference output columns already exist; refusing to overwrite them"
            raise ValueError(msg)
        references, diagnostics = [], []
        for text, url in zip(table[self.text_field].to_pylist(), table[self.url_field].to_pylist(), strict=True):
            if (text is not None and not isinstance(text, str)) or (url is not None and not isinstance(url, str)):
                msg = "ImageReferenceExtractor expects string or null text and page URLs"
                raise TypeError(msg)
            rows, flags = extract_image_references(text, url)
            references.append(rows)
            diagnostics.append(flags)
        table = table.append_column("image_references", pa.array(references, type=_REFERENCE_TYPE))
        table = table.append_column("image_reference_diagnostics", pa.array(diagnostics, type=_DIAGNOSTICS_TYPE))
        return replace(batch, data=table)
