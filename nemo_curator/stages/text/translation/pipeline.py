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

"""Translation pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.translation.evaluation.faith import (
    FaithEvalFilter,
    FaithThresholdFilterStage,
)
from nemo_curator.stages.text.translation.stages import (
    FormatTranslationOutputStage,
    MergeFaithScoresStage,
    SegmentTranslationStage,
    ReassemblyStage,
    SegmentationStage,
    RestoreSkippedRowsStage,
    SkipExistingTranslationsStage,
)
from nemo_curator.tasks import DocumentBatch

_VALID_OUTPUT_MODES = {"replaced", "raw", "both"}


def _needs_structured_faith_helpers(text_field: str | list[str]) -> bool:
    """Return whether FAITH needs flattened helper columns."""
    if isinstance(text_field, list):
        return True
    return "*" in text_field or "." in text_field


@dataclass(kw_only=True)
class TranslationStage(CompositeStage[DocumentBatch, DocumentBatch]):
    """Composite stage for document translation and optional quality scoring."""

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
    segment_level: bool = False
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

        if not self.source_lang:
            raise ValueError("TranslationStage requires a non-empty 'source_lang'")
        if not self.target_lang:
            raise ValueError("TranslationStage requires a non-empty 'target_lang'")

        if self.output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"Invalid output_mode '{self.output_mode}'. "
                f"Must be one of: {sorted(_VALID_OUTPUT_MODES)}"
            )

        if self.backend_type == "llm":
            if self.client is None:
                raise ValueError(
                    "TranslationStage with backend_type='llm' requires a non-None "
                    "'client' (AsyncLLMClient)"
                )
            if not self.model_name:
                raise ValueError(
                    "TranslationStage with backend_type='llm' requires a non-empty "
                    "'model_name'"
                )

        if self.enable_faith_eval:
            if self.client is None:
                if self.backend_type == "llm":
                    raise ValueError(
                        "TranslationStage with enable_faith_eval=True requires a non-None "
                        "'client' (AsyncLLMClient)"
                    )
                raise ValueError(
                    "TranslationStage with enable_faith_eval=True and "
                    f"backend_type={self.backend_type!r} requires a separate "
                    "AsyncLLMClient for FAITH scoring"
                )

            faith_model = self.faith_model_name or self.model_name
            if not faith_model:
                raise ValueError(
                    "TranslationStage with enable_faith_eval=True requires "
                    "'faith_model_name' or 'model_name' to be set for FAITH scoring"
                )

        if self.merge_scores and self.output_mode == "replaced":
            raise ValueError(
                "merge_scores=True requires output_mode in {'raw','both'}. "
                "Got output_mode='replaced'. Set output_mode='both' explicitly."
            )

        if self.merge_scores and not self.enable_faith_eval:
            logger.warning(
                "merge_scores=True but enable_faith_eval=False; "
                "score merging will be skipped"
            )

        super().__init__()
        self.stages = self._build_stages()

    def _build_stages(self) -> list[ProcessingStage]:
        """Construct the ordered list of sub-stages."""
        stages: list[ProcessingStage] = []
        faith_helper_needed = (
            self.enable_faith_eval
            and not self.segment_level
            and _needs_structured_faith_helpers(self.text_field)
        )
        use_segment_level_faith = self.enable_faith_eval and self.segment_level
        use_document_level_faith = self.enable_faith_eval and not self.segment_level

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

        if use_segment_level_faith:
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
                emit_faith_helpers=faith_helper_needed,
                aggregate_faith_scores=use_segment_level_faith,
            )
        )

        if use_segment_level_faith and self.filter_enabled:
            stages.append(FaithThresholdFilterStage(threshold=self.faith_threshold))

        if self.skip_translated:
            stages.append(RestoreSkippedRowsStage())

        if use_document_level_faith:
            faith_model = self.faith_model_name or self.model_name
            faith_source_field = (
                "_faith_source_text" if faith_helper_needed else self.text_field
            )
            faith_translated_field = (
                "_faith_translated_text" if faith_helper_needed else self.output_field
            )

            stages.append(
                FaithEvalFilter(
                    client=self.client,
                    model_name=faith_model,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    source_text_field=faith_source_field,
                    translated_text_field=faith_translated_field,
                    threshold=self.faith_threshold,
                    filter_enabled=self.filter_enabled,
                ),
            )

        needs_formatting = (
            self.output_mode != "replaced"
            or self.reconstruct_messages
            or faith_helper_needed
        )
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
