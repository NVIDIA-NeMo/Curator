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

"""ReassemblyStage -- stitches translated segments back into whole documents.

Reverses the explosion performed by
:class:`~nemo_curator.stages.text.translation.segmentation.SegmentationStage`
using the reconstruction metadata stored in ``_seg_metadata``.

Supports both flat text columns and nested/wildcard field paths (Gap 1.1 / 1.2).
When the segmentation metadata contains ``field_metadatas`` with wildcard paths,
:func:`~nemo_curator.stages.text.translation.field_utils.set_nested_fields` is
used to inject translated text back into structured data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.translation.field_utils import (
    is_wildcard_path,
    parse_structured_value,
    set_nested_fields,
)
from nemo_curator.tasks.document import DocumentBatch

# Internal columns written by SegmentationStage / TranslateStage
_INTERNAL_COLUMNS = {
    "_seg_segments",
    "_seg_metadata",
    "_seg_doc_id",
    "_translated",
    "_translation_time",
    "_translation_error",
}


@dataclass
class ReassemblyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Collapse translated segment rows back into one row per original document.

    Groups rows by ``_seg_doc_id``, reads the ``_seg_metadata`` JSON to
    determine the reconstruction strategy (coarse or fine), and substitutes
    translated segments into the stored template.

    When segments originated from wildcard / nested field paths, the
    reassembled translations are written back into the structured column
    via :func:`set_nested_fields`.

    Attributes:
        text_field: Name of the column that held the original source text,
            **or** a wildcard dot-path / list matching what was passed to
            :class:`SegmentationStage`.
        output_field: Name of the column to write the reassembled translation
            into.  For flat (non-wildcard) fields the reassembled string is
            stored here.  For wildcard paths the structured column is updated
            in-place and a copy with translations injected is stored in
            ``output_field``.
    """

    name: str = "ReassemblyStage"
    text_field: str = "text"
    output_field: str = "translated_text"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_translated", "_seg_metadata", "_seg_doc_id"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_field, "translation_time", "translation_errors"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Reassemble translated segments into full documents.

        1. Group rows by ``_seg_doc_id``.
        2. For each group, parse ``_seg_metadata`` from the first row.
        3. Dispatch to the appropriate reassembly logic (legacy single-field,
           multi-field with optional wildcard paths, or skip-mode passthrough).
        4. Produce one output row per original document.
        5. Drop internal ``_seg_*`` / ``_translated`` columns.
        """
        df = batch.to_pandas()

        result_rows: list[dict[str, Any]] = []

        for doc_id, group in df.groupby("_seg_doc_id", sort=True):
            group = group.sort_index()  # preserve original segment order

            metadata_json = group.iloc[0]["_seg_metadata"]
            metadata: dict[str, Any] = json.loads(metadata_json)
            translated_segments: list[str] = group["_translated"].tolist()

            # Aggregate per-segment timing and error columns before dropping.
            # Sum timing across all segments; concatenate non-empty errors.
            if "_translation_time" in group.columns:
                total_time = group["_translation_time"].sum()
            else:
                total_time = 0.0
            if "_translation_error" in group.columns:
                non_empty_errors = [
                    str(e) for e in group["_translation_error"] if e and str(e).strip()
                ]
                combined_errors = "; ".join(non_empty_errors)
            else:
                combined_errors = ""

            # Build output row from the first row of the group (carries all
            # original columns) plus the reassembled translation.
            first_row = group.iloc[0].to_dict()
            out_row = {k: v for k, v in first_row.items() if k not in _INTERNAL_COLUMNS}

            # Store aggregated timing/error as public (non-underscore) columns
            out_row["translation_time"] = total_time
            out_row["translation_errors"] = combined_errors

            # -------------------------------------------------------
            # Dispatch based on metadata shape
            # -------------------------------------------------------
            if metadata.get("mode") == "skip":
                # Gap 9.2: skipme passthrough -- no translation was done.
                out_row[self.output_field] = ""
                result_rows.append(out_row)
                continue

            if "field_metadatas" in metadata:
                # New multi-field / wildcard-aware format produced by the
                # updated SegmentationStage.
                self._reassemble_multi_field(metadata, translated_segments, out_row)
            else:
                # Legacy single-field format (backwards compatibility with
                # metadata produced before the Gap 1.1/1.2 changes).
                mode = metadata.get("mode", "coarse")
                if mode == "fine":
                    reassembled = self._reassemble_fine(metadata, translated_segments)
                else:
                    reassembled = self._reassemble_coarse(metadata, translated_segments)
                out_row[self.output_field] = reassembled

            result_rows.append(out_row)

        out_df = pd.DataFrame(result_rows)
        out_df.reset_index(drop=True, inplace=True)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=out_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    # ------------------------------------------------------------------
    # Multi-field reassembly (Gap 1.1 / 1.2)
    # ------------------------------------------------------------------

    def _reassemble_multi_field(
        self,
        metadata: dict[str, Any],
        translated_segments: list[str],
        out_row: dict[str, Any],
    ) -> None:
        """Reassemble translations from one or more field paths.

        Each entry in ``metadata["field_metadatas"]`` describes the
        segmentation of one text blob (one field-path x one extracted string).
        The translated segments are consumed in order across all entries.

        For flat (non-wildcard) field paths, the reassembled text is written
        to ``self.output_field``.  For wildcard paths the structured column
        is updated in-place using :func:`set_nested_fields`.

        Args:
            metadata: The combined metadata envelope.
            translated_segments: All translated segments for this document,
                in the same order they were emitted during segmentation.
            out_row: The output row dict (mutated in place).
        """
        field_metadatas: list[dict[str, Any]] = metadata["field_metadatas"]

        seg_offset = 0
        # Group reassembled texts by field_path for wildcard injection.
        reassembled_by_path: dict[str, list[str]] = {}

        for fm in field_metadatas:
            mode = fm.get("mode", "coarse")
            field_path = fm.get("field_path", self.text_field)

            # Determine how many segments this entry consumed.
            n_segments = self._count_segments_in_meta(fm)
            entry_segments = translated_segments[seg_offset : seg_offset + n_segments]
            seg_offset += n_segments

            if mode == "fine":
                reassembled = self._reassemble_fine(fm, entry_segments)
            elif mode == "coarse":
                reassembled = self._reassemble_coarse(fm, entry_segments)
            else:
                # Unknown mode -- concatenate segments as fallback.
                reassembled = " ".join(entry_segments)

            reassembled_by_path.setdefault(field_path, []).append(reassembled)

        if seg_offset != len(translated_segments):
            logger.warning(
                f"Multi-field reassembly: consumed {seg_offset} segments but "
                f"received {len(translated_segments)}"
            )

        # Write results back into out_row.
        for field_path, texts in reassembled_by_path.items():
            if is_wildcard_path(field_path) or "." in field_path:
                # Structured column -- inject translated texts back into the
                # nested structure stored in the root column.
                root_key = field_path.split(".")[0]
                raw_value = out_row.get(root_key)
                record = parse_structured_value(raw_value)
                if record is not None:
                    updated = set_nested_fields({root_key: record}, field_path, texts)
                    # Store the updated structure in output_field.
                    # If the original value was a JSON string, serialize back.
                    if isinstance(raw_value, str):
                        out_row[self.output_field] = json.dumps(
                            updated[root_key], ensure_ascii=False
                        )
                    else:
                        out_row[self.output_field] = updated[root_key]
                else:
                    # Fallback: store concatenated texts.
                    out_row[self.output_field] = "\n\n".join(texts)
            else:
                # Flat column -- simple reassembled string.
                out_row[self.output_field] = texts[0] if len(texts) == 1 else "\n\n".join(texts)

    @staticmethod
    def _count_segments_in_meta(fm: dict[str, Any]) -> int:
        """Count how many translatable segments a single field-metadata entry expects.

        Args:
            fm: A single field metadata dict (with ``mode``, ``template`` or
                ``units``, etc.).

        Returns:
            The number of translatable segments for this entry.
        """
        mode = fm.get("mode", "coarse")
        if mode == "coarse":
            template = fm.get("template", [])
            return sum(1 for entry in template if entry is None)
        elif mode == "fine":
            units = fm.get("units", [])
            return sum(1 for u in units if u.get("translatable", False))
        return 0

    # ------------------------------------------------------------------
    # Coarse reassembly
    # ------------------------------------------------------------------

    @staticmethod
    def _reassemble_coarse(metadata: dict[str, Any], translated_segments: list[str]) -> str:
        """Reconstruct a document from coarse-mode metadata.

        Iterates over the ``template`` list.  For each ``None`` entry the next
        translated segment is substituted (with leading whitespace restored).
        All entries are joined with ``"\\n"``.
        """
        template: list[str | None] = metadata["template"]
        leading_spaces: list[str] = metadata["leading_spaces"]

        trans_idx = 0
        output_lines: list[str] = []

        for entry in template:
            if entry is None:
                if trans_idx < len(translated_segments):
                    leading = leading_spaces[trans_idx] if trans_idx < len(leading_spaces) else ""
                    output_lines.append(leading + translated_segments[trans_idx])
                    trans_idx += 1
                else:
                    logger.warning("Coarse reassembly: ran out of translated segments")
                    output_lines.append("")
            else:
                output_lines.append(entry)

        if trans_idx != len(translated_segments):
            logger.warning(
                f"Coarse reassembly: expected {trans_idx} segments consumed but "
                f"received {len(translated_segments)}"
            )

        return "\n".join(output_lines)

    # ------------------------------------------------------------------
    # Fine reassembly
    # ------------------------------------------------------------------

    @staticmethod
    def _reassemble_fine(metadata: dict[str, Any], translated_segments: list[str]) -> str:
        """Reconstruct a document from fine-mode metadata.

        Iterates over the ``units`` list.  For translatable units, the next
        translated segment plus its stored separator is emitted.  For
        non-translatable units, ``original + separator`` is emitted verbatim.
        """
        units: list[dict[str, Any]] = metadata["units"]

        trans_idx = 0
        parts: list[str] = []

        for unit in units:
            if unit["translatable"]:
                if trans_idx < len(translated_segments):
                    parts.append(translated_segments[trans_idx] + unit["separator"])
                    trans_idx += 1
                else:
                    logger.warning(
                        f"Fine reassembly: ran out of translated segments at unit '{unit['original']}'"
                    )
                    parts.append(unit["original"] + unit["separator"])
            else:
                parts.append(unit["original"] + unit["separator"])

        if trans_idx != len(translated_segments):
            logger.warning(
                f"Fine reassembly: expected {trans_idx} segments consumed but "
                f"received {len(translated_segments)}"
            )

        return "".join(parts)
