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

"""Reassemble translated segments back into document rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.translation.utils.field_paths import (
    is_wildcard_path,
    parse_structured_value,
    set_nested_fields,
)
from nemo_curator.tasks.document import DocumentBatch

_INTERNAL_COLUMNS = {
    "_seg_segments",
    "_seg_metadata",
    "_seg_doc_id",
    "_translated",
    "_translation_time",
    "_translation_error",
}

_FAITH_SCORE_COLUMNS = {
    "faith_fluency": "Fluency",
    "faith_accuracy": "Accuracy",
    "faith_idiomaticity": "Idiomaticity",
    "faith_terminology": "Terminology",
    "faith_handling_of_format": "Handling_of_Format",
}


@dataclass
class ReassemblyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Collapse segment rows back into one row per document."""

    name: str = "ReassemblyStage"
    text_field: str = "text"
    output_field: str = "translated_text"
    replace_source_fields: bool = False
    emit_metadata_helpers: bool = False
    aggregate_faith_scores: bool = False

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_translated", "_seg_metadata", "_seg_doc_id"]

    def outputs(self) -> tuple[list[str], list[str]]:
        out_cols = [self.output_field, "translation_time", "translation_errors"]
        if self.emit_metadata_helpers:
            out_cols.extend(["_translation_map", "_segmented_translation_map"])
        if self.aggregate_faith_scores:
            out_cols.extend(
                [
                    "faith_fluency",
                    "faith_accuracy",
                    "faith_idiomaticity",
                    "faith_terminology",
                    "faith_handling_of_format",
                    "faith_avg",
                    "faith_parse_failed",
                    "faith_segment_scores",
                ]
            )
        return ["data"], out_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Reassemble translated segments into full documents."""
        df = batch.to_pandas()

        result_rows: list[dict[str, Any]] = []

        for _doc_id, doc_group in df.groupby("_seg_doc_id", sort=True):
            sorted_group = doc_group.sort_index()  # preserve original segment order
            result_rows.append(self._build_reassembled_row(sorted_group))

        out_df = pd.DataFrame(result_rows)
        out_df = out_df.reset_index(drop=True)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=out_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _build_reassembled_row(self, group: pd.DataFrame) -> dict[str, Any]:
        """Build one output document row from its segment group."""
        metadata: dict[str, Any] = json.loads(group.iloc[0]["_seg_metadata"])
        translated_segments: list[str] = group["_translated"].tolist()
        out_row = self._base_output_row(group)

        if metadata.get("mode") == "skip":
            return self._build_skip_row(out_row, group)

        translation_map, segmented_map = self._build_translation_maps(
            metadata=metadata,
            translated_segments=translated_segments,
            out_row=out_row,
        )

        if self.emit_metadata_helpers:
            out_row["_translation_map"] = json.dumps(translation_map, ensure_ascii=False)
            out_row["_segmented_translation_map"] = json.dumps(segmented_map, ensure_ascii=False)
        if self.aggregate_faith_scores:
            self._write_aggregated_faith_scores(out_row, group)
        return out_row

    def _base_output_row(self, group: pd.DataFrame) -> dict[str, Any]:
        """Create the common output row fields for one document group."""
        first_row = group.iloc[0].to_dict()
        out_row = {k: v for k, v in first_row.items() if k not in _INTERNAL_COLUMNS}
        out_row["translation_time"] = group["_translation_time"].sum() if "_translation_time" in group.columns else 0.0
        if "_translation_error" in group.columns:
            errors = [str(e) for e in group["_translation_error"] if e and str(e).strip()]
            out_row["translation_errors"] = "; ".join(errors)
        else:
            out_row["translation_errors"] = ""
        return out_row

    def _build_skip_row(self, out_row: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
        """Build passthrough output for rows marked as skipped."""
        out_row[self.output_field] = ""
        if self.emit_metadata_helpers:
            out_row["_translation_map"] = json.dumps({}, ensure_ascii=False)
            out_row["_segmented_translation_map"] = json.dumps({}, ensure_ascii=False)
        if self.aggregate_faith_scores:
            self._write_aggregated_faith_scores(out_row, group)
        return out_row

    def _build_translation_maps(
        self,
        *,
        metadata: dict[str, Any],
        translated_segments: list[str],
        out_row: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reassemble translated text and return metadata helper maps."""
        if "field_metadatas" in metadata:
            return self._reassemble_multi_field(metadata, translated_segments, out_row)
        return self._reassemble_single_field(metadata, translated_segments, out_row)

    def _reassemble_single_field(
        self,
        metadata: dict[str, Any],
        translated_segments: list[str],
        out_row: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reassemble a single translated field."""
        mode = metadata.get("mode", "coarse")
        reassembled = (
            self._reassemble_fine(metadata, translated_segments)
            if mode == "fine"
            else self._reassemble_coarse(metadata, translated_segments)
        )
        out_row[self.output_field] = reassembled
        field_path = self.text_field
        field_key = self._leaf_field_key(field_path)
        if self.replace_source_fields and not (is_wildcard_path(field_path) or "." in field_path):
            out_row[field_path] = reassembled
        return {field_key: reassembled}, {field_key: self._build_segment_pairs(metadata, translated_segments)}

    def _reassemble_multi_field(
        self,
        metadata: dict[str, Any],
        translated_segments: list[str],
        out_row: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reassemble one or more translated field paths."""
        reassembled_by_path, translation_map, segmented_map, seg_offset = self._collect_multi_field_outputs(
            metadata,
            translated_segments,
        )
        self._warn_for_unconsumed_segments(seg_offset, translated_segments)
        self._write_multi_field_payload(out_row, reassembled_by_path, translation_map)
        return translation_map, segmented_map

    def _collect_multi_field_outputs(
        self,
        metadata: dict[str, Any],
        translated_segments: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, Any], int]:
        """Collect per-field reassembled text and helper maps."""
        field_metadatas: list[dict[str, Any]] = metadata["field_metadatas"]

        seg_offset = 0
        reassembled_by_path: dict[str, list[str]] = {}
        translation_map: dict[str, Any] = {}
        segmented_map: dict[str, Any] = {}

        for fm in field_metadatas:
            mode = fm.get("mode", "coarse")
            field_path = fm.get("field_path", self.text_field)
            field_key = self._leaf_field_key(field_path)
            wildcard_path = is_wildcard_path(field_path)

            n_segments = self._count_segments_in_meta(fm)
            entry_segments = translated_segments[seg_offset : seg_offset + n_segments]
            seg_offset += n_segments

            if mode == "passthrough":
                reassembled = entry_segments[0] if entry_segments else ""
            elif mode == "fine":
                reassembled = self._reassemble_fine(fm, entry_segments)
            elif mode == "coarse":
                reassembled = self._reassemble_coarse(fm, entry_segments)
            else:
                reassembled = " ".join(entry_segments)

            reassembled_by_path.setdefault(field_path, []).append(reassembled)
            current_pairs = self._build_segment_pairs(fm, entry_segments)
            if wildcard_path:
                translation_map.setdefault(field_key, []).append(reassembled)
                segmented_map.setdefault(field_key, []).extend(current_pairs)
            else:
                translation_map[field_key] = reassembled
                segmented_map[field_key] = current_pairs

        return reassembled_by_path, translation_map, segmented_map, seg_offset

    @staticmethod
    def _warn_for_unconsumed_segments(seg_offset: int, translated_segments: list[str]) -> None:
        """Log when multi-field metadata did not consume all translated segments."""
        remaining_segments = translated_segments[seg_offset:]
        has_nonempty_remaining = any(str(seg).strip() for seg in remaining_segments)
        if seg_offset != len(translated_segments) and has_nonempty_remaining:
            msg = f"Multi-field reassembly: consumed {seg_offset} segments but received {len(translated_segments)}"
            logger.warning(msg)

    def _write_multi_field_payload(
        self,
        out_row: dict[str, Any],
        reassembled_by_path: dict[str, list[str]],
        translation_map: dict[str, Any],
    ) -> None:
        """Write reassembled multi-field output payload into ``out_row``."""
        output_payload: Any = None
        for field_path, texts in reassembled_by_path.items():
            output_payload = self._write_one_field_payload(out_row, field_path, texts)

        if not reassembled_by_path:
            out_row[self.output_field] = ""
        elif len(reassembled_by_path) == 1 and output_payload is not None:
            out_row[self.output_field] = output_payload
        else:
            out_row[self.output_field] = translation_map

    def _write_one_field_payload(
        self,
        out_row: dict[str, Any],
        field_path: str,
        texts: list[str],
    ) -> object:
        """Write one reassembled field and return its output payload value."""
        if is_wildcard_path(field_path) or "." in field_path:
            return self._write_nested_field_payload(out_row, field_path, texts)
        reassembled_text = texts[0] if len(texts) == 1 else "\n\n".join(texts)
        if self.replace_source_fields:
            out_row[field_path] = reassembled_text
        return reassembled_text

    def _write_nested_field_payload(
        self,
        out_row: dict[str, Any],
        field_path: str,
        texts: list[str],
    ) -> object:
        """Write a nested or wildcard field payload."""
        root_key = field_path.split(".")[0]
        raw_value = out_row.get(root_key)
        record = parse_structured_value(raw_value)
        if record is None:
            return "\n\n".join(texts)

        updated = set_nested_fields({root_key: record}, field_path, texts)
        updated_value = (
            json.dumps(updated[root_key], ensure_ascii=False) if isinstance(raw_value, str) else updated[root_key]
        )
        if self.replace_source_fields:
            out_row[root_key] = updated_value
        return updated_value

    @staticmethod
    def _count_segments_in_meta(fm: dict[str, Any]) -> int:
        """Count the translatable segments expected by one field entry."""
        mode = fm.get("mode", "coarse")
        if mode == "passthrough":
            return 1
        elif mode == "coarse":
            template = fm.get("template", [])
            return sum(1 for entry in template if entry is None)
        elif mode == "fine":
            units = fm.get("units", [])
            return sum(1 for u in units if u.get("translatable", False))
        return 0

    @staticmethod
    def _leaf_field_key(field_path: str) -> str:
        """Return the metadata key for *field_path*."""
        return field_path.split(".")[-1]

    @classmethod
    def _write_aggregated_faith_scores(
        cls,
        out_row: dict[str, Any],
        group: pd.DataFrame,
    ) -> None:
        """Aggregate segment-level FAITH scores into one document-level record."""
        if not set(_FAITH_SCORE_COLUMNS).issubset(group.columns):
            out_row["faith_fluency"] = 0.0
            out_row["faith_accuracy"] = 0.0
            out_row["faith_idiomaticity"] = 0.0
            out_row["faith_terminology"] = 0.0
            out_row["faith_handling_of_format"] = 0.0
            out_row["faith_avg"] = 0.0
            out_row["faith_parse_failed"] = False
            out_row["faith_segment_scores"] = "[]"
            return

        segment_scores: list[dict[str, float]] = []
        for _, row in group.iterrows():
            segment_scores.append(
                {
                    faith_key: float(row.get(column_name, 0.0) or 0.0)
                    for column_name, faith_key in _FAITH_SCORE_COLUMNS.items()
                }
            )

        averaged_scores = cls._average_faith_scores(segment_scores)
        out_row["faith_fluency"] = averaged_scores["Fluency"]
        out_row["faith_accuracy"] = averaged_scores["Accuracy"]
        out_row["faith_idiomaticity"] = averaged_scores["Idiomaticity"]
        out_row["faith_terminology"] = averaged_scores["Terminology"]
        out_row["faith_handling_of_format"] = averaged_scores["Handling_of_Format"]
        out_row["faith_avg"] = cls._compute_faith_avg(averaged_scores)
        out_row["faith_parse_failed"] = bool(group.get("faith_parse_failed", pd.Series(dtype=bool)).any())
        out_row["faith_segment_scores"] = json.dumps(segment_scores, ensure_ascii=False)

    @staticmethod
    def _average_faith_scores(segment_scores: list[dict[str, float]]) -> dict[str, float]:
        """Average FAITH scores across segments, ignoring zero-valued dimensions."""
        if not segment_scores:
            return dict.fromkeys(_FAITH_SCORE_COLUMNS.values(), 0.0)

        averaged: dict[str, float] = {}
        for faith_key in _FAITH_SCORE_COLUMNS.values():
            values = [score.get(faith_key, 0.0) for score in segment_scores if score.get(faith_key, 0.0) > 0]
            averaged[faith_key] = round(sum(values) / len(values), 2) if values else 0.0
        return averaged

    @staticmethod
    def _compute_faith_avg(scores: dict[str, float]) -> float:
        """Compute ``faith_avg`` as the mean of non-zero FAITH dimensions."""
        values = [float(scores.get(key, 0.0)) for key in _FAITH_SCORE_COLUMNS.values()]
        non_zero = [value for value in values if value > 0]
        if not non_zero:
            return 0.0
        return float(sum(non_zero) / len(non_zero))

    @staticmethod
    def _build_segment_pairs(metadata: dict[str, Any], translated_segments: list[str]) -> list[dict[str, str]]:
        """Build ``[{src, tgt}, ...]`` pairs for one field entry."""
        mode = metadata.get("mode", "coarse")
        if mode == "passthrough":
            original_text = metadata.get("original_text", "")
            translated_text = translated_segments[0] if translated_segments else ""
            return [{"src": original_text, "tgt": translated_text}]
        if mode == "coarse":
            original_lines = metadata.get("original_stripped_lines", [])
            return [{"src": src, "tgt": tgt} for src, tgt in zip(original_lines, translated_segments, strict=False)]
        if mode == "fine":
            pairs: list[dict[str, str]] = []
            units = metadata.get("units", [])
            trans_idx = 0
            for unit in units:
                if not unit.get("translatable", False):
                    continue
                tgt = translated_segments[trans_idx] if trans_idx < len(translated_segments) else ""
                pairs.append({"src": unit.get("original", ""), "tgt": tgt})
                trans_idx += 1
            return pairs
        return []

    @staticmethod
    def _reassemble_coarse(metadata: dict[str, Any], translated_segments: list[str]) -> str:
        """Reconstruct a document from coarse-mode metadata."""
        template: list[str | None] = metadata["template"]
        leading_spaces: list[str] = metadata["leading_spaces"]

        expected_segments = sum(1 for entry in template if entry is None)
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

        if expected_segments != len(translated_segments):
            logger.warning(
                "Coarse reassembly: segment count mismatch: metadata expected {} segments but pipeline processed {}",
                expected_segments,
                len(translated_segments),
            )

        return "\n".join(output_lines)

    @staticmethod
    def _reassemble_fine(metadata: dict[str, Any], translated_segments: list[str]) -> str:
        """Reconstruct a document from fine-mode metadata."""
        units: list[dict[str, Any]] = metadata["units"]

        expected_segments = sum(1 for u in units if u.get("translatable", False))
        trans_idx = 0
        parts: list[str] = []

        for unit in units:
            if unit["translatable"]:
                if trans_idx < len(translated_segments):
                    parts.append(translated_segments[trans_idx] + unit["separator"])
                    trans_idx += 1
                else:
                    logger.warning(
                        "Fine reassembly: ran out of translated segments at unit {!r}",
                        unit["original"],
                    )
                    parts.append(unit["original"] + unit["separator"])
            else:
                parts.append(unit["original"] + unit["separator"])

        if expected_segments != len(translated_segments):
            logger.warning(
                "Fine reassembly: segment count mismatch: metadata expected {} segments but pipeline processed {}",
                expected_segments,
                len(translated_segments),
            )

        return "".join(parts)
