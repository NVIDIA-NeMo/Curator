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

"""Stage for shaping translation output columns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.translation.utils.metadata import (
    build_translation_metadata,
    reconstruct_messages_with_translation,
)
from nemo_curator.tasks import DocumentBatch


@dataclass(kw_only=True)
class FormatTranslationOutputStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Apply the requested translation output format."""

    name: str = "FormatTranslationOutputStage"
    target_lang: str
    output_mode: str = "replaced"
    output_field: str = "translated_text"
    reconstruct_messages: bool = False
    messages_field: str = "messages"
    messages_content_field: str = "content"

    def __post_init__(self) -> None:
        self.target_lang = self.target_lang.strip()
        if not self.target_lang:
            raise ValueError(
                "FormatTranslationOutputStage requires a non-empty 'target_lang'"
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        out_cols: list[str] = []
        if self.output_mode in ("raw", "both"):
            out_cols.append("translation_metadata")
        if self.output_mode in ("replaced", "both"):
            out_cols.append(self.output_field)
        if self.reconstruct_messages:
            out_cols.append("translated_messages")
        return ["data"], out_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Apply output formatting to the batch."""
        df = batch.to_pandas().copy()
        if df.empty:
            return batch

        if self.output_mode in ("raw", "both"):
            self._build_metadata_column(df)

        if self.output_mode == "raw" and self.output_field in df.columns:
            df = df.drop(columns=[self.output_field])

        if self.reconstruct_messages and self.messages_field in df.columns:
            self._build_translated_messages(df)

        columns_to_drop = [
            col
            for col in (
                "_translation_map",
                "_segmented_translation_map",
                "_faith_source_text",
                "_faith_translated_text",
            )
            if col in df.columns
        ]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _build_metadata_column(self, df: pd.DataFrame) -> None:
        """Construct the ``translation_metadata`` JSON column."""
        metadata_values: list[str] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            translated_text = row.get(self.output_field, "")
            translation_map = self._parse_optional_json_object(row.get("_translation_map"))
            segmented_translation_map = self._parse_optional_json_object(
                row.get("_segmented_translation_map")
            )

            metadata_values.append(
                build_translation_metadata(
                    target_lang=self.target_lang,
                    translated_text=translated_text,
                    translation_map=translation_map,
                    segmented_translation_map=segmented_translation_map,
                )
            )

        df["translation_metadata"] = metadata_values

    def _build_translated_messages(self, df: pd.DataFrame) -> None:
        """Construct the ``translated_messages`` column from original messages."""
        translated_msgs: list[str] = []
        for idx in range(len(df)):
            raw_messages = df.iloc[idx].get(self.messages_field)
            translated_text = df.iloc[idx].get(self.output_field, "")

            if raw_messages is None:
                translated_msgs.append("[]")
                continue

            if isinstance(raw_messages, str):
                try:
                    messages_list = json.loads(raw_messages)
                except (json.JSONDecodeError, TypeError):
                    translated_msgs.append("[]")
                    continue
            elif isinstance(raw_messages, list):
                messages_list = raw_messages
            else:
                translated_msgs.append("[]")
                continue

            reconstructed = reconstruct_messages_with_translation(
                original_messages=messages_list,
                translated_text=translated_text,
                field_path=self.messages_content_field,
            )
            translated_msgs.append(json.dumps(reconstructed, ensure_ascii=False))

        df["translated_messages"] = translated_msgs

    @staticmethod
    def _parse_optional_json_object(value: Any) -> dict[str, Any] | None:
        """Parse helper JSON emitted by ReassemblyStage when present."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return None
            if isinstance(parsed, dict):
                return parsed
        return None


__all__ = ["FormatTranslationOutputStage"]
