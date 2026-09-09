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

"""Curator stage for a prebuilt Indic Canary TensorRT-LLM engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.models.asr.indic_canary import IndicCanaryTRTLLMASR
from nemo_curator.stages.audio.inference.asr.stage import ASRStage, _set_note
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.models.asr.base import ASRResult
    from nemo_curator.tasks import AudioTask

_ADAPTER_TARGET = "nemo_curator.models.asr.indic_canary.IndicCanaryTRTLLMASR"


@dataclass
class InferenceIndicCanaryStage(ASRStage):
    """Transcribe Indic audio with an existing Canary TensorRT-LLM engine."""

    adapter_target: str = field(default=_ADAPTER_TARGET, init=False, repr=False)
    engine_dir: str = ""
    name: str = "IndicCanary_inference"
    num_beams: int = 4
    max_new_tokens: int = 374
    pnc: bool = False
    max_duration_sec: float = 40.0
    min_duration_sec: float = 0.5
    kv_cache_free_gpu_memory_fraction: float = 0.2
    cross_kv_cache_fraction: float = 0.2
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    skip_me_key: str = "_skipme"
    keep_waveform: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    model_id: str = field(default="", init=False, repr=False)
    audio_filepath_key: str = field(default="", init=False, repr=False)
    target_sample_rate: int = field(default=16_000, init=False, repr=False)
    default_language: str | None = field(default=None, init=False, repr=False)
    supported_language_codes: list[str] | None = field(default=None, init=False, repr=False)
    extras_key: str | None = field(default=None, init=False, repr=False)
    unsupported_language_marks_skip: bool = field(default=True, init=False, repr=False)
    unsupported_language_skip_reason: str | None = field(
        default="lang_not_supported:{stage_name}",
        init=False,
        repr=False,
    )
    preserve_existing_skip: bool = field(default=True, init=False, repr=False)
    missing_language_is_unsupported: bool = field(default=False, init=False, repr=False)
    skip_if_output_exists: bool = field(default=False, init=False, repr=False)
    fail_on_audio_error: bool = field(default=True, init=False, repr=False)
    prefetch_fail_on_error: bool = field(default=True, init=False, repr=False)
    adapter_kwargs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_id = self.engine_dir
        self.adapter_kwargs = {
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "pnc": self.pnc,
            "max_duration_sec": self.max_duration_sec,
            "min_duration_sec": self.min_duration_sec,
            "kv_cache_free_gpu_memory_fraction": self.kv_cache_free_gpu_memory_fraction,
            "cross_kv_cache_fraction": self.cross_kv_cache_fraction,
        }
        super().__post_init__()

    def outputs(self) -> tuple[list[str], list[str]]:
        """Match the integration-stage output declaration."""
        return [], [self.pred_text_key, self.language_key]

    def _create_adapter(self) -> IndicCanaryTRTLLMASR:
        """Construct Canary through its reference-compatible ``engine_dir`` API."""
        return IndicCanaryTRTLLMASR(engine_dir=self.engine_dir, **self.adapter_kwargs)

    def assemble(
        self,
        tasks: list[AudioTask],
        items: list[dict[str, Any]],
        results: list[ASRResult],
    ) -> list[AudioTask]:
        """Write Canary-specific unsupported-language and truncation metadata."""
        previous_skip_values = [task.data.get(self.skip_me_key) for task in tasks]
        super().assemble(tasks, items, results)

        for task, item, result, previous_skip in zip(
            tasks,
            items,
            results,
            previous_skip_values,
            strict=True,
        ):
            if result.extras.get("language_unsupported"):
                language = str(item.get("language_code") or "").strip().lower()
                _set_note(
                    task.data,
                    self.name,
                    f"skipped (unsupported language: {language})",
                    self.notes_key,
                )
                _set_note(
                    task.data,
                    self.pred_text_key,
                    f"lang_not_supported:{language}",
                    self.notes_key,
                )
                if previous_skip:
                    task.data[self.skip_me_key] = previous_skip
                else:
                    task.data[self.skip_me_key] = f"lang_not_supported:{self.name}"
            duration = float(result.extras.get("audio_duration_sec") or 0.0)
            if duration >= self.max_duration_sec:
                _set_note(
                    task.data,
                    self.name,
                    f"audio {duration:.2f}s exceeds {self.max_duration_sec:.0f}s encoder window; "
                    "transcription truncated (split with VAD before inference)",
                    self.notes_key,
                )
        return tasks


__all__ = ["IndicCanaryTRTLLMASR", "InferenceIndicCanaryStage"]
