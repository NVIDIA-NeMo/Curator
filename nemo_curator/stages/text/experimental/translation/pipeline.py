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

"""Experimental translation pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.experimental.translation.evaluation.faith import (
    FaithEvalFilter,
    FaithThresholdFilterStage,
)
from nemo_curator.stages.text.experimental.translation.stages import (
    FormatTranslationOutputStage,
    MergeFaithScoresStage,
    ReassemblyStage,
    RestoreSkippedRowsStage,
    SegmentationStage,
    SegmentTranslationStage,
    SkipExistingTranslationsStage,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig

_VALID_OUTPUT_MODES = {"replaced", "raw", "both"}


@dataclass(kw_only=True)
class TranslationStage(CompositeStage[DocumentBatch, DocumentBatch]):
    """Experimental composite stage for translation and optional quality scoring."""

    name: str = "TranslationStage"

    source_lang: str
    target_lang: str
    text_field: str | list[str] = "text"
    output_field: str = "translated_text"
    segmentation_mode: str = "coarse"
    min_segment_chars: int = 0

    client: AsyncLLMClient | None = None
    model_name: str = ""
    generation_config: GenerationConfig | None = None

    backend_type: str = "llm"
    backend_config: dict = field(default_factory=dict)

    enable_faith_eval: bool = False
    faith_threshold: float = 2.5
    faith_model_name: str = ""
    filter_enabled: bool = True

    output_mode: str = "replaced"
    merge_scores: bool = False
    reconstruct_messages: bool = False
    messages_field: str = "messages"
    messages_content_field: str = "content"
    skip_translated: bool = False
    translation_column: str = "translated_text"

    def __post_init__(self) -> None:
        self.source_lang = self.source_lang.strip()
        self.target_lang = self.target_lang.strip()
        self.model_name = self.model_name.strip()
        self.faith_model_name = self.faith_model_name.strip()

        self._validate_languages()
        self._validate_output_mode()
        self._validate_translation_backend()
        self._validate_faith_config()
        self._validate_score_merging()

        super().__init__()
        self.stages = self._build_stages()

    def _validate_languages(self) -> None:
        """Validate source and target language codes."""
        if not self.source_lang:
            msg = "TranslationStage requires a non-empty 'source_lang'"
            raise ValueError(msg)
        if not self.target_lang:
            msg = "TranslationStage requires a non-empty 'target_lang'"
            raise ValueError(msg)

    def _validate_output_mode(self) -> None:
        """Validate requested output mode."""
        if self.output_mode not in _VALID_OUTPUT_MODES:
            msg = f"Invalid output_mode '{self.output_mode}'. Must be one of: {sorted(_VALID_OUTPUT_MODES)}"
            raise ValueError(msg)

    def _validate_translation_backend(self) -> None:
        """Validate backend-specific translation requirements."""
        if self.backend_type == "llm":
            if self.client is None:
                msg = "TranslationStage with backend_type='llm' requires a non-None 'client' (AsyncLLMClient)"
                raise ValueError(msg)
            if not self.model_name:
                msg = "TranslationStage with backend_type='llm' requires a non-empty 'model_name'"
                raise ValueError(msg)

    def _validate_faith_config(self) -> None:
        """Validate optional FAITH scoring configuration."""
        if self.enable_faith_eval:
            if self.client is None:
                if self.backend_type == "llm":
                    msg = "TranslationStage with enable_faith_eval=True requires a non-None 'client' (AsyncLLMClient)"
                    raise ValueError(msg)
                msg = (
                    "TranslationStage with enable_faith_eval=True and "
                    f"backend_type={self.backend_type!r} requires a separate "
                    "AsyncLLMClient for FAITH scoring"
                )
                raise ValueError(msg)

            faith_model = self.faith_model_name or self.model_name
            if not faith_model:
                msg = (
                    "TranslationStage with enable_faith_eval=True requires "
                    "'faith_model_name' or 'model_name' to be set for FAITH scoring"
                )
                raise ValueError(msg)

    def _validate_score_merging(self) -> None:
        """Validate score-merging options."""
        if self.merge_scores and self.output_mode == "replaced":
            msg = (
                "merge_scores=True requires output_mode in {'raw','both'}. "
                "Got output_mode='replaced'. Set output_mode='both' explicitly."
            )
            raise ValueError(msg)

        if self.merge_scores and not self.enable_faith_eval:
            logger.warning("merge_scores=True but enable_faith_eval=False; score merging will be skipped")

    def _build_stages(self) -> list[ProcessingStage]:
        """Construct the ordered list of sub-stages."""
        stages: list[ProcessingStage] = []

        if self.skip_translated:
            stages.append(
                SkipExistingTranslationsStage(
                    translation_column=self.translation_column,
                )
            )

        stages.append(
            SegmentationStage(
                text_field=self.text_field,
                source_lang=self.source_lang,
                mode=self.segmentation_mode,
                min_segment_chars=self.min_segment_chars,
            )
        )
        stages.append(
            SegmentTranslationStage(
                client=self.client,
                model_name=self.model_name,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                backend_type=self.backend_type,
                backend_config=self.backend_config,
                generation_config=self.generation_config,
            )
        )

        if self.enable_faith_eval:
            faith_model = self.faith_model_name or self.model_name
            stages.append(
                FaithEvalFilter(
                    client=self.client,
                    model_name=faith_model,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    source_text_field="_seg_segments",
                    translated_text_field="_translated",
                    threshold=self.faith_threshold,
                    filter_enabled=False,
                )
            )

        stages.append(
            ReassemblyStage(
                text_field=self.text_field,
                output_field=self.output_field,
                replace_source_fields=self.output_mode in ("replaced", "both"),
                emit_metadata_helpers=self.output_mode in ("raw", "both"),
                aggregate_faith_scores=self.enable_faith_eval,
            )
        )

        if self.enable_faith_eval and self.filter_enabled:
            stages.append(FaithThresholdFilterStage(threshold=self.faith_threshold))

        if self.skip_translated:
            stages.append(RestoreSkippedRowsStage())

        needs_formatting = self.output_mode != "replaced" or self.reconstruct_messages
        if needs_formatting:
            stages.append(
                FormatTranslationOutputStage(
                    output_mode=self.output_mode,
                    target_lang=self.target_lang,
                    output_field=self.output_field,
                    reconstruct_messages=self.reconstruct_messages,
                    messages_field=self.messages_field,
                    messages_content_field=self.messages_content_field,
                )
            )

        if self.enable_faith_eval and self.merge_scores and self.output_mode in ("raw", "both"):
            stages.append(MergeFaithScoresStage())

        return stages

    def decompose(self) -> list[ProcessingStage]:
        """Return the ordered sub-stages for pipeline execution."""
        return self.stages
