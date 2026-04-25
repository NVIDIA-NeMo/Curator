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

"""SegmentationStage -- splits documents into translatable segments.

Supports two modes:

* **coarse** -- line-level splitting with code-block awareness.
* **fine** -- sentence-level splitting via spaCy with exact-structure preservation.

Multi-field and wildcard-path support allows translating nested structures
such as ``messages.*.content`` without manual flattening.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.translation.utils.field_paths import (
    extract_nested_fields,
    is_wildcard_path,
    normalize_text_field,
    parse_structured_value,
)
from nemo_curator.tasks.document import DocumentBatch

# ---------------------------------------------------------------------------
# spaCy language model registry
# ---------------------------------------------------------------------------

SPACY_LANG_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",      # English
    "de": "de_core_news_sm",     # German
    "fr": "fr_core_news_sm",     # French
    "es": "es_core_news_sm",     # Spanish
    "it": "it_core_news_sm",     # Italian
    "pt": "pt_core_news_sm",     # Portuguese
    "nl": "nl_core_news_sm",     # Dutch
    "pl": "pl_core_news_sm",     # Polish
    "ru": "ru_core_news_sm",     # Russian
    "zh": "zh_core_web_sm",      # Chinese
    "ja": "ja_core_news_sm",     # Japanese
    # Languages without dedicated models use multilingual fallback
    "hi": "xx_sent_ud_sm",       # Hindi
    "ar": "xx_sent_ud_sm",       # Arabic
    "ko": "xx_sent_ud_sm",       # Korean
}
SPACY_FALLBACK_MODEL: str = "xx_sent_ud_sm"

# Cache for loaded spaCy models. Variants with a custom ``max_length`` are
# cached separately so callers do not mutate shared model instances.
_nlp_cache: dict[tuple[str, int | None], Any] = {}

def _resolve_spacy_model_name(src_lang: str = "en") -> str:
    """Resolve the spaCy model name for the given language."""
    return SPACY_LANG_MODELS.get(src_lang, SPACY_FALLBACK_MODEL)


def _get_spacy_nlp(src_lang: str = "en", *, max_length: int | None = None) -> Any:
    """Lazy-load a spaCy model for the given source language.

    Args:
        src_lang: ISO 639-1 language code (e.g. ``'en'``, ``'de'``, ``'hi'``).
        max_length: Optional override for ``nlp.max_length`` on the cached
            instance created for this call.

    Returns:
        A loaded spaCy ``Language`` model appropriate for *src_lang*.
    """
    model_name = _resolve_spacy_model_name(src_lang)
    cache_key = (model_name, max_length)

    if cache_key not in _nlp_cache:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for segmentation_mode='fine'. "
                "Install the optional translation_segmentation extra "
                "(for example, `uv sync --extra translation_segmentation`)."
            ) from exc

        try:
            nlp = spacy.load(model_name)
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found for lang '{src_lang}', "
                f"using fallback '{SPACY_FALLBACK_MODEL}'"
            )
            model_name = SPACY_FALLBACK_MODEL
            nlp = spacy.load(model_name)
            cache_key = (model_name, max_length)

        if max_length is not None:
            nlp.max_length = max_length
        _nlp_cache[cache_key] = nlp
        logger.info(
            "spaCy model '{}' loaded for lang '{}'{}",
            model_name,
            src_lang,
            f" with max_length: {nlp.max_length}" if max_length is not None else "",
        )

    return _nlp_cache[cache_key]


# ---------------------------------------------------------------------------
# Sentence splitting with structure preservation
# ---------------------------------------------------------------------------

def split_into_sentences_with_structure(text: str, src_lang: str = "en") -> list[tuple[str, str]]:
    """Split *text* using spaCy, then apply custom regex patterns while preserving exact structure.

    Returns a list of ``(sentence_text, separator_after)`` tuples such that
    ``"".join(t + s for t, s in result)`` reconstructs the original *text*.

    Args:
        text: The text to split into sentences.
        src_lang: ISO 639-1 language code for loading the appropriate spaCy model.
    """
    nlp = _get_spacy_nlp(src_lang)
    if len(text) > nlp.max_length:
        nlp = _get_spacy_nlp(src_lang, max_length=max(10_000_000, len(text) + 1))

    # Custom regex pattern for special characters that should be treated as separators
    special_separator_pattern = (
        r"(\#{2,}|\_{2,}|\…{2,}|\%{2,}|\+{2,}|\.{2,}|\-{3,}|\*{2,}|\~{2,}|\={2,}|\!{2,}"
        r"|\n|\t|\‣|\⁃|\⁌|\⁍|\●|\○|\•|\·|\◘|\◦|\⦾|\⦿|\|)"
    )

    doc = nlp(text)
    spacy_sentences = list(doc.sents)

    spacy_units_with_separators: list[tuple[str, str]] = []

    # Check if there's text before the first spaCy sentence
    if spacy_sentences:
        first_sent_start = spacy_sentences[0].start_char
        if first_sent_start > 0:
            leading_text = text[:first_sent_start]
            spacy_units_with_separators.append(("", leading_text))

    for i, sent in enumerate(spacy_sentences):
        sent_start = sent.start_char
        sent_end = sent.end_char

        # Calculate the separator after this sentence
        if i < len(spacy_sentences) - 1:
            next_sent_start = spacy_sentences[i + 1].start_char
            separator = text[sent_end:next_sent_start]
        else:
            separator = text[sent_end:]

        sentence_text = text[sent_start:sent_end]
        spacy_units_with_separators.append((sentence_text, separator))

    # Process each spaCy unit for special separators
    all_text_units: list[tuple[str, str]] = []

    for sent_text, sent_separator in spacy_units_with_separators:
        special_matches = list(re.finditer(special_separator_pattern, sent_text))

        if not special_matches:
            # Strip leading/trailing spaces from text unit and add to separator
            stripped_text = sent_text.strip()
            leading_spaces = sent_text[: len(sent_text) - len(sent_text.lstrip())]
            trailing_spaces = sent_text[len(sent_text.rstrip()) :]

            new_separator = trailing_spaces + sent_separator

            if leading_spaces and stripped_text:
                all_text_units.append(("", leading_spaces))

            all_text_units.append((stripped_text, new_separator))
        else:
            # Split this sentence by special separators
            last_end = 0
            for match in special_matches:
                split_start = match.start()
                split_end = match.end()

                text_unit = sent_text[last_end:split_start]
                separator = sent_text[split_start:split_end]

                stripped_text_unit = text_unit.strip()
                leading_spaces = text_unit[: len(text_unit) - len(text_unit.lstrip())]
                trailing_spaces = text_unit[len(text_unit.rstrip()) :]

                new_separator = trailing_spaces + separator

                if leading_spaces and stripped_text_unit:
                    all_text_units.append(("", leading_spaces))

                all_text_units.append((stripped_text_unit, new_separator))
                last_end = split_end

            # Handle remaining text after last special separator
            if last_end < len(sent_text):
                remaining = sent_text[last_end:]
                stripped_remaining = remaining.strip()
                leading_spaces = remaining[: len(remaining) - len(remaining.lstrip())]
                trailing_spaces = remaining[len(remaining.rstrip()) :]

                new_sent_separator = trailing_spaces + sent_separator

                if leading_spaces and stripped_remaining:
                    all_text_units.append(("", leading_spaces))

                all_text_units.append((stripped_remaining, new_sent_separator))
            else:
                if sent_separator:
                    all_text_units.append(("", sent_separator))

    # Verify reconstruction
    reconstructed = "".join(text_unit + separator for text_unit, separator in all_text_units)
    if text != reconstructed:
        logger.warning("Structure mismatch in sentence splitting, falling back to single unit")
        return [(text, "")]

    return all_text_units


def is_line_translatable_content(line: str) -> bool:
    """Determine whether *line* contains translatable content.

    Returns ``False`` for lines that have no alphabetic characters or that
    look like XML/HTML tags (e.g. ``<tag>``). Structured JSON blobs are also
    treated as non-translatable so tool payloads and machine-readable content
    are preserved verbatim.
    """
    stripped_line = line.strip()
    if not any(char.isalpha() for char in stripped_line):
        return False
    if stripped_line.startswith("<") and stripped_line.endswith(">"):
        return False
    if (
        (stripped_line.startswith("{") and stripped_line.endswith("}"))
        or (stripped_line.startswith("[") and stripped_line.endswith("]"))
    ):
        try:
            parsed = json.loads(stripped_line)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return False
    return True


# ---------------------------------------------------------------------------
# SegmentationStage
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class SegmentationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split documents into translatable segments.

    Each input row is *exploded* into N output rows (one per translatable
    segment).  Reconstruction metadata is stored as a JSON string in the
    ``_seg_metadata`` column so that :class:`ReassemblyStage` can later
    collapse the rows back into whole documents.

    Attributes:
        text_field: Name of the input column containing source text, **or** a
            wildcard dot-path (e.g. ``"messages.*.content"``), **or** a list
            of such paths for multi-field translation.
        source_lang: ISO 639-1 code used for spaCy model selection.
        mode: ``"coarse"`` (line-level) or ``"fine"`` (sentence-level).
        skipme_field: If set, rows where ``df[skipme_field] != 0`` are passed
            through without segmentation (preserved with empty segments).
    """

    name: str = "SegmentationStage"
    source_lang: str
    text_field: str | list[str] = "text"
    mode: str = "coarse"
    min_segment_chars: int = 0
    skipme_field: str | None = None

    def __post_init__(self) -> None:
        self.source_lang = self.source_lang.strip()
        if not self.source_lang:
            raise ValueError("SegmentationStage requires a non-empty 'source_lang'")

    def inputs(self) -> tuple[list[str], list[str]]:
        # For simple (non-wildcard) single fields, declare the column dependency.
        # For wildcard / multi-field cases the actual column may hold structured
        # data, so we only declare the root column names.
        paths = normalize_text_field(self.text_field)
        root_cols = list({p.split(".")[0] for p in paths})
        return ["data"], root_cols

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_seg_segments", "_seg_metadata", "_seg_doc_id"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Segment each document into translatable units.

        For each row in ``batch.data``:

        1. If ``skipme_field`` is set and the row is flagged, pass through
           with an empty segment.
        2. Resolve ``text_field`` -- may be a plain column, a wildcard path
           into structured data, or a list of paths (multi-field).
        3. Apply coarse or fine segmentation to each extracted text.
        4. Explode: one output row per segment.
        """
        df = batch.to_pandas()
        field_paths = normalize_text_field(self.text_field)

        all_rows: list[dict[str, Any]] = []

        total_docs = len(df)
        total_segments = 0
        docs_with_zero_segments = 0

        for doc_idx, (_row_idx, row) in enumerate(df.iterrows()):
            doc_rows, doc_segment_count = self._segment_document(
                row=row,
                field_paths=field_paths,
                doc_idx=doc_idx,
            )
            all_rows.extend(doc_rows)

            if doc_segment_count == 0:
                docs_with_zero_segments += 1
            else:
                total_segments += doc_segment_count

        avg_segs = total_segments / max(total_docs - docs_with_zero_segments, 1)
        logger.info(
            f"SegmentationStage: {total_docs} documents | "
            f"{total_segments} segments created | "
            f"{docs_with_zero_segments} documents with zero translatable segments | "
            f"avg segments/doc (excl. zero): {avg_segs:.2f}"
        )

        out_df = pd.DataFrame(all_rows)
        # Reset index so downstream stages get a clean 0-based index
        out_df.reset_index(drop=True, inplace=True)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=out_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _segment_document(
        self,
        *,
        row: pd.Series,
        field_paths: list[str],
        doc_idx: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Segment a single source document and emit exploded output rows."""
        original_cols = row.to_dict()

        skipped_row = self._build_skip_output_row(original_cols, doc_idx)
        if skipped_row is not None:
            return [skipped_row], 0

        segments, field_metadatas = self._collect_document_segments(row, field_paths)
        metadata_json = self._build_metadata_json(field_metadatas)
        return self._build_output_rows(original_cols, segments, metadata_json, doc_idx), len(segments)

    def _build_skip_output_row(
        self,
        original_cols: dict[str, Any],
        doc_idx: int,
    ) -> dict[str, Any] | None:
        """Return a passthrough row when ``skipme_field`` marks the document."""
        if self.skipme_field is None or self.skipme_field not in original_cols:
            return None

        skipme_val = original_cols[self.skipme_field]
        if skipme_val == 0 or skipme_val is None:
            return None

        out_row = dict(original_cols)
        out_row["_seg_segments"] = ""
        out_row["_seg_metadata"] = json.dumps({"mode": "skip"}, ensure_ascii=False)
        out_row["_seg_doc_id"] = doc_idx
        return out_row

    def _collect_document_segments(
        self,
        row: pd.Series,
        field_paths: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Collect translated segments and metadata for all requested field paths."""
        all_segments: list[str] = []
        field_metadatas: list[dict[str, Any]] = []

        for field_path in field_paths:
            texts = self._extract_texts(row, field_path)
            for text in texts:
                segments, metadata = self._segment_text(text, field_path)
                field_metadatas.append(metadata)
                all_segments.extend(segments)

        return all_segments, field_metadatas

    def _segment_text(self, text: str, field_path: str) -> tuple[list[str], dict[str, Any]]:
        """Segment one extracted text value and attach its field path."""
        if self.min_segment_chars > 0 and len(text) < self.min_segment_chars:
            return [text], {
                "mode": "passthrough",
                "field_path": field_path,
                "original_text": text,
            }

        segment_fn = self._segment_fine if self.mode == "fine" else self._segment_coarse
        segments, meta_json = segment_fn(text)
        metadata = json.loads(meta_json)
        metadata["field_path"] = field_path
        return segments, metadata

    @staticmethod
    def _build_metadata_json(field_metadatas: list[dict[str, Any]]) -> str:
        """Serialize the per-field metadata envelope for one source document."""
        return json.dumps({"field_metadatas": field_metadatas}, ensure_ascii=False)

    @staticmethod
    def _build_output_rows(
        original_cols: dict[str, Any],
        segments: list[str],
        metadata_json: str,
        doc_idx: int,
    ) -> list[dict[str, Any]]:
        """Create exploded output rows for the segmented document."""
        row_segments = segments or [""]
        return [
            {
                **original_cols,
                "_seg_segments": segment,
                "_seg_metadata": metadata_json,
                "_seg_doc_id": doc_idx,
            }
            for segment in row_segments
        ]

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_texts(row: pd.Series, field_path: str) -> list[str]:
        """Extract translatable text(s) from a row given a *field_path*.

        If *field_path* is a simple column name (no wildcard), the column
        value is returned directly.  If it is a wildcard dot-path, the root
        column is parsed as structured data (dict or JSON string) and
        :func:`extract_nested_fields` is used to pull matching string values.

        Args:
            row: A single DataFrame row.
            field_path: A plain column name or a wildcard dot-path.

        Returns:
            A list of string texts extracted from the row.
        """
        if not is_wildcard_path(field_path) and "." not in field_path:
            # Simple flat column -- original behaviour.
            val = row.get(field_path, "")
            if isinstance(val, str):
                return [val] if val else []
            return [str(val)] if val else []

        # Wildcard / nested path -- the root key is the first path component.
        root_key = field_path.split(".")[0]
        raw_value = row.get(root_key)
        if raw_value is None:
            return []

        record = parse_structured_value(raw_value)
        if record is None:
            # Not structured data; fall back to treating root column as plain text.
            if isinstance(raw_value, str) and raw_value:
                return [raw_value]
            return []

        # The parsed record is the cell's value (a dict).  field_path
        # starts with root_key, so we wrap the record under that key so
        # extract_nested_fields can traverse from the top.
        return extract_nested_fields({root_key: record}, field_path)

    # ------------------------------------------------------------------
    # Coarse segmentation
    # ------------------------------------------------------------------

    def _segment_coarse(self, text: str) -> tuple[list[str], str]:
        """Line-level segmentation with code-block awareness.

        Returns:
            A tuple of (segments, metadata_json) where *segments* is a list of
            translatable stripped lines and *metadata_json* is the JSON-serialised
            reconstruction template.
        """
        lines = text.split("\n")
        template: list[str | None] = []
        leading_spaces_list: list[str] = []
        segments: list[str] = []
        original_stripped_lines: list[str] = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                template.append(line)
                continue

            if in_code_block or not is_line_translatable_content(line):
                template.append(line)
            else:
                num_leading = len(line) - len(line.lstrip())
                leading = line[:num_leading]
                stripped = line[num_leading:]

                template.append(None)
                segments.append(stripped)
                leading_spaces_list.append(leading)
                original_stripped_lines.append(stripped)

        metadata = {
            "mode": "coarse",
            "template": template,
            "leading_spaces": leading_spaces_list,
            "original_stripped_lines": original_stripped_lines,
        }
        return segments, json.dumps(metadata, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Fine segmentation
    # ------------------------------------------------------------------

    def _segment_fine(self, text: str) -> tuple[list[str], str]:
        """Sentence-level segmentation via spaCy with exact-structure preservation.

        Returns:
            A tuple of (segments, metadata_json) where *segments* is a list of
            translatable sentence-like units and *metadata_json* is the
            JSON-serialised reconstruction data.
        """
        sentence_units = split_into_sentences_with_structure(text, self.source_lang)

        units: list[dict[str, Any]] = []
        segments: list[str] = []

        for text_unit, separator in sentence_units:
            if text_unit.strip() and is_line_translatable_content(text_unit):
                units.append({"translatable": True, "original": text_unit, "separator": separator})
                segments.append(text_unit)
            else:
                units.append({"translatable": False, "original": text_unit, "separator": separator})

        metadata = {
            "mode": "fine",
            "units": units,
        }
        return segments, json.dumps(metadata, ensure_ascii=False)
