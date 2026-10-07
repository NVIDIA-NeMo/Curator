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

"""A validated view of step 4's `semantic_diff`, so critics can cite evidence by span ID."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

DEFAULT_MAX_EVIDENCE_CHARS = 240
_ID_PREFIX = {"SHARED": "S", "A_ONLY": "A", "B_ONLY": "B"}

# Fixed schema so every batch has the same column type however many spans it holds.
PACKET_ARROW_TYPE = pa.struct(
    [
        ("status", pa.string()),
        ("truncated", pa.bool_()),
        ("span_counts", pa.struct([(kind, pa.int64()) for kind in _ID_PREFIX])),
        (
            "spans",
            pa.list_(
                pa.struct(
                    [
                        *(
                            (name, pa.string())
                            for name in ("span_id", "kind", "a_text", "b_text", "text", "context_text")
                        ),
                        *(
                            (name, pa.int64())
                            for name in (
                                "a_start_char",
                                "a_end_char",
                                "b_start_char",
                                "b_end_char",
                                "start_char",
                                "end_char",
                            )
                        ),
                    ]
                )
            ),
        ),
    ]
)


@dataclass(frozen=True)
class Excerpt:
    """A passage of one document, located by character offsets into the original text."""

    side: str
    start_char: int
    end_char: int
    text: str

    def as_evidence(self) -> dict[str, Any]:
        return {"side": self.side, "start_char": self.start_char, "end_char": self.end_char, "quote": self.text}


@dataclass(frozen=True)
class Span:
    """One S###/A###/B### span. SHARED spans have both excerpts; A_ONLY/B_ONLY spans have one."""

    span_id: str
    kind: str
    a: Excerpt | None
    b: Excerpt | None
    context_text: str = ""  # Padded reading window around an A_ONLY/B_ONLY difference.

    def excerpt(self, side: str) -> Excerpt | None:
        return self.a if side == "A" else self.b


@dataclass(frozen=True)
class SpanPacket:
    status: str
    truncated: bool
    span_counts: dict[str, int]
    spans: dict[str, Span]

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE" and not self.truncated

    @classmethod
    def from_record(
        cls, record: dict[str, Any], *, max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS
    ) -> SpanPacket:
        """Validate a pair record's `semantic_diff` against its original `text_a`/`text_b`.

        Raises only ValueError, so callers can keep a bad row instead of aborting the batch.
        """
        try:
            return cls._parse(record, max_evidence_chars)
        except (KeyError, TypeError, AttributeError) as error:
            msg = f"malformed semantic_diff: {error!r}"
            raise ValueError(msg) from error

    @classmethod
    def _parse(cls, record: dict[str, Any], max_evidence_chars: int) -> SpanPacket:
        documents = {"A": record["text_a"], "B": record["text_b"]}
        if not all(isinstance(text, str) for text in documents.values()):
            msg = "text_a and text_b must be strings"
            raise ValueError(msg)
        packet = record["semantic_diff"]
        if not isinstance(packet, dict):
            msg = "missing semantic_diff"
            raise ValueError(msg)
        if packet["status"] not in {"COMPLETE", "INCOMPLETE_LIMIT"}:
            msg = f"invalid packet status {packet['status']!r}"
            raise ValueError(msg)
        flags = [record["truncated"], *(packet[key] for key in ("truncated", "truncated_a", "truncated_b"))]
        if not all(type(flag) is bool for flag in flags):
            msg = "truncation flags must be booleans"
            raise ValueError(msg)
        if not flags[0] == flags[1] == (flags[2] or flags[3]):
            msg = "inconsistent truncation flags"
            raise ValueError(msg)

        spans: dict[str, Span] = {}
        for raw in packet["spans"]:
            span = _build_span(raw, documents, max_evidence_chars)
            if span.span_id in spans:
                msg = f"duplicate span ID {span.span_id}"
                raise ValueError(msg)
            spans[span.span_id] = span

        counts = Counter(span.kind for span in spans.values())
        declared = {kind: int(packet["span_counts"][kind]) for kind in _ID_PREFIX}
        if packet["status"] == "COMPLETE" and declared != {kind: counts[kind] for kind in _ID_PREFIX}:
            msg = "COMPLETE packet is missing spans"
            raise ValueError(msg)
        return cls(packet["status"], bool(packet["truncated"]), declared, spans)

    def for_prompt(self) -> dict[str, Any]:
        """The packet in the fixed shape of `PACKET_ARROW_TYPE`, for Jinja prompts.

        SHARED spans fill the `a_*`/`b_*` fields; A_ONLY/B_ONLY spans fill `text` and the offsets
        of their core difference. Fields that don't apply to a span are "" or None.
        """
        spans = []
        for span in self.spans.values():
            shared = span.kind == "SHARED"
            unique = None if shared else (span.a or span.b)
            spans.append(
                {
                    "span_id": span.span_id,
                    "kind": span.kind,
                    "a_text": span.a.text if shared else "",
                    "b_text": span.b.text if shared else "",
                    "text": "" if shared else unique.text,
                    "context_text": span.context_text,
                    "a_start_char": span.a.start_char if shared else None,
                    "a_end_char": span.a.end_char if shared else None,
                    "b_start_char": span.b.start_char if shared else None,
                    "b_end_char": span.b.end_char if shared else None,
                    "start_char": None if shared else unique.start_char,
                    "end_char": None if shared else unique.end_char,
                }
            )
        return {"status": self.status, "truncated": self.truncated, "span_counts": self.span_counts, "spans": spans}


def _offset(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    # Arrow and pandas round trips turn nullable integers into floats.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"missing or non-integer {key} in span {raw.get('span_id')}"
    raise ValueError(msg)


def _excerpt(document: str, side: str, raw: dict[str, Any], prefix: str) -> Excerpt:
    start, end = _offset(raw, prefix + "start_char"), _offset(raw, prefix + "end_char")
    text = raw.get(prefix + "text")
    if not (0 <= start < end <= len(document) and document[start:end] == text):
        msg = f"offsets do not match the original text for span {raw.get('span_id')}"
        raise ValueError(msg)
    return Excerpt(side, start, end, text)


def _build_span(raw: dict[str, Any], documents: dict[str, str], max_evidence_chars: int) -> Span:
    span_id, kind = raw["span_id"], raw["kind"]
    if kind not in _ID_PREFIX or not re.fullmatch(rf"{_ID_PREFIX[kind]}\d{{3}}", str(span_id)):
        msg = f"invalid span ID/kind: {span_id!r}/{kind!r}"
        raise ValueError(msg)

    if kind == "SHARED":
        a, b = (_excerpt(documents[side], side, raw, side.lower() + "_") for side in ("A", "B"))
        span = Span(span_id, kind, a, b)
    else:
        side = kind[0]
        window = _excerpt(documents[side], side, raw, "")
        start, end = _offset(raw, "delta_start_char"), _offset(raw, "delta_end_char")
        if not window.start_char <= start < end <= window.end_char:
            msg = f"invalid delta boundaries for span {span_id}"
            raise ValueError(msg)
        delta = Excerpt(side, start, end, documents[side][start:end])
        span = Span(span_id, kind, delta if side == "A" else None, delta if side == "B" else None, window.text)

    if any(len(excerpt.text) > max_evidence_chars for excerpt in (span.a, span.b) if excerpt):
        msg = f"oversized span {span_id}"
        raise ValueError(msg)
    return span
