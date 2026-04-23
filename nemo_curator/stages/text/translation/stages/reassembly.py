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
from nemo_curator.stages.text.translation.utils.field_paths import (
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


@dataclass
class ReassemblyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Collapse segment rows back into one row per document."""

    name: str = "ReassemblyStage"
    text_field: str = "text"
    output_field: str = "translated_text"
    replace_source_fields: bool = False
    emit_metadata_helpers: bool = False
    emit_faith_helpers: bool = False

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_translated", "_seg_metadata", "_seg_doc_id"]

    def outputs(self) -> tuple[list[str], list[str]]:
        out_cols = [self.output_field, "translation_time", "translation_errors"]
        if self.emit_metadata_helpers:
            out_cols.extend(["_translation_map", "_segmented_translation_map"])
        if self.emit_faith_helpers:
            out_cols.extend(["_faith_source_text", "_faith_translated_text"])
        return ["data"], out_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Reassemble translated segments into full documents."""
        df = batch.to_pandas()

        result_rows: list[dict[str, Any]] = []

        for doc_id, group in df.groupby("_seg_doc_id", sort=True):
            group = group.sort_index()  # preserve original segment order

            metadata_json = group.iloc[0]["_seg_metadata"]
            metadata: dict[str, Any] = json.loads(metadata_json)
            translated_segments: list[str] = group["_translated"].tolist()

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

            first_row = group.iloc[0].to_dict()
            out_row = {k: v for k, v in first_row.items() if k not in _INTERNAL_COLUMNS}

            out_row["translation_time"] = total_time
            out_row["translation_errors"] = combined_errors

            if metadata.get("mode") == "skip":
                out_row[self.output_field] = ""
                if self.emit_metadata_helpers:
                    out_row["_translation_map"] = json.dumps({}, ensure_ascii=False)
                    out_row["_segmented_translation_map"] = json.dumps({}, ensure_ascii=False)
                result_rows.append(out_row)
                continue

            if "field_metadatas" in metadata:
                translation_map, segmented_map, faith_source_text, faith_translated_text = self._reassemble_multi_field(
                    metadata,
                    translated_segments,
                    out_row,
                )
            else:
                mode = metadata.get("mode", "coarse")
                if mode == "fine":
                    reassembled = self._reassemble_fine(metadata, translated_segments)
                else:
                    reassembled = self._reassemble_coarse(metadata, translated_segments)
                out_row[self.output_field] = reassembled
                field_path = self.text_field
                field_key = self._leaf_field_key(field_path)
                translation_map = {field_key: reassembled}
                segmented_map = {
                    field_key: self._build_segment_pairs(metadata, translated_segments)
                }
                faith_source_text = self._serialize_faith_entries(
                    [{"field_path": field_path, "text": self._reassemble_original_text(metadata)}]
                )
                faith_translated_text = self._serialize_faith_entries(
                    [{"field_path": field_path, "text": reassembled}]
                )
                if self.replace_source_fields and not (is_wildcard_path(field_path) or "." in field_path):
                    out_row[field_path] = reassembled

            if self.emit_metadata_helpers:
                out_row["_translation_map"] = json.dumps(translation_map, ensure_ascii=False)
                out_row["_segmented_translation_map"] = json.dumps(segmented_map, ensure_ascii=False)
            if self.emit_faith_helpers:
                out_row["_faith_source_text"] = faith_source_text
                out_row["_faith_translated_text"] = faith_translated_text

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

    def _reassemble_multi_field(
        self,
        metadata: dict[str, Any],
        translated_segments: list[str],
        out_row: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], str, str]:
        """Reassemble one or more translated field paths."""
        field_metadatas: list[dict[str, Any]] = metadata["field_metadatas"]

        seg_offset = 0
        reassembled_by_path: dict[str, list[str]] = {}
        translation_map: dict[str, Any] = {}
        segmented_map: dict[str, Any] = {}
        faith_source_entries: list[dict[str, str]] = []
        faith_translated_entries: list[dict[str, str]] = []

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

            faith_source_entries.append(
                {"field_path": field_path, "text": self._reassemble_original_text(fm)}
            )
            faith_translated_entries.append(
                {"field_path": field_path, "text": reassembled}
            )

            reassembled_by_path.setdefault(field_path, []).append(reassembled)
            current_pairs = self._build_segment_pairs(fm, entry_segments)
            if wildcard_path:
                translation_map.setdefault(field_key, []).append(reassembled)
                segmented_map.setdefault(field_key, []).extend(current_pairs)
            else:
                translation_map[field_key] = reassembled
                segmented_map[field_key] = current_pairs

        remaining_segments = translated_segments[seg_offset:]
        has_nonempty_remaining = any(str(seg).strip() for seg in remaining_segments)
        if seg_offset != len(translated_segments) and has_nonempty_remaining:
            logger.warning(
                f"Multi-field reassembly: consumed {seg_offset} segments but "
                f"received {len(translated_segments)}"
            )

        output_payload: Any = None

        for field_path, texts in reassembled_by_path.items():
            if is_wildcard_path(field_path) or "." in field_path:
                root_key = field_path.split(".")[0]
                raw_value = out_row.get(root_key)
                record = parse_structured_value(raw_value)
                if record is not None:
                    updated = set_nested_fields({root_key: record}, field_path, texts)
                    updated_value = (
                        json.dumps(updated[root_key], ensure_ascii=False)
                        if isinstance(raw_value, str)
                        else updated[root_key]
                    )
                    if self.replace_source_fields:
                        out_row[root_key] = updated_value
                    output_payload = updated_value
                else:
                    output_payload = "\n\n".join(texts)
            else:
                reassembled_text = texts[0] if len(texts) == 1 else "\n\n".join(texts)
                if self.replace_source_fields:
                    out_row[field_path] = reassembled_text
                output_payload = reassembled_text

        if len(reassembled_by_path) == 1 and output_payload is not None:
            out_row[self.output_field] = output_payload
        else:
            out_row[self.output_field] = translation_map

        return (
            translation_map,
            segmented_map,
            self._serialize_faith_entries(faith_source_entries),
            self._serialize_faith_entries(faith_translated_entries),
        )

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
            return [
                {"src": src, "tgt": tgt}
                for src, tgt in zip(original_lines, translated_segments)
            ]
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

    @classmethod
    def _reassemble_original_text(cls, metadata: dict[str, Any]) -> str:
        """Reconstruct the original source text for one field entry."""
        mode = metadata.get("mode", "coarse")
        if mode == "passthrough":
            return str(metadata.get("original_text", ""))
        if mode == "coarse":
            return cls._reassemble_coarse(
                metadata,
                [str(text) for text in metadata.get("original_stripped_lines", [])],
            )
        if mode == "fine":
            units = metadata.get("units", [])
            original_segments = [
                str(unit.get("original", ""))
                for unit in units
                if unit.get("translatable", False)
            ]
            return cls._reassemble_fine(metadata, original_segments)
        return ""

    @staticmethod
    def _serialize_faith_entries(entries: list[dict[str, str]]) -> str:
        """Serialize reassembled field texts into a FAITH-friendly string."""
        if not entries:
            return ""
        if len(entries) == 1:
            return entries[0]["text"]
        return json.dumps(entries, ensure_ascii=False)

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
                "Coarse reassembly: segment count mismatch: metadata expected "
                "{} segments but pipeline processed {}",
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
                "Fine reassembly: segment count mismatch: metadata expected "
                "{} segments but pipeline processed {}",
                expected_segments,
                len(translated_segments),
            )

        return "".join(parts)
