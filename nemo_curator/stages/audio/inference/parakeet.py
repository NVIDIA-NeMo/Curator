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

"""NVIDIA Parakeet-TDT inference stage for in-memory audio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.resources import Resources

PARAKEET_TDT_0_6B_V3_LANGS: frozenset[str] = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)

_ADAPTER_TARGET = "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter"


@dataclass
class InferenceParakeetStage(ASRStage):
    """Transcribe Parakeet-supported languages without temporary WAV files."""

    adapter_target: str = field(default=_ADAPTER_TARGET, init=False, repr=False)
    model_id: str = "nvidia/parakeet-tdt-0.6b-v3"
    name: str = "Parakeet_inference"
    supported_langs: frozenset[str] | set[str] | None = None
    # Kept as a named compatibility option for integration-pipeline configs;
    # this adapter-backed port deliberately supports Curator's NeMo runtime.
    backend: Literal["nemo"] = "nemo"
    tensorrt_engine_dir: str | None = None
    chunking_mode: Literal["engine", "none"] = "engine"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    source_lang_key: str = "source_lang"
    keep_waveform: bool = False
    skip_if_output_exists: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128

    audio_filepath_key: str = field(default="", init=False, repr=False)
    target_sample_rate: int = field(default=16_000, init=False, repr=False)
    default_language: str | None = field(default=None, init=False, repr=False)
    supported_language_codes: list[str] = field(default_factory=list, init=False, repr=False)
    extras_key: str | None = field(default=None, init=False, repr=False)
    skip_me_key: str = field(default="_skipme", init=False, repr=False)
    unsupported_language_marks_skip: bool = field(default=False, init=False, repr=False)
    unsupported_language_skip_reason: str | None = field(default=None, init=False, repr=False)
    preserve_existing_skip: bool = field(default=False, init=False, repr=False)
    missing_language_is_unsupported: bool = field(default=True, init=False, repr=False)
    fail_on_audio_error: bool = field(default=True, init=False, repr=False)
    prefetch_fail_on_error: bool = field(default=True, init=False, repr=False)
    adapter_kwargs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend != "nemo":
            msg = (
                "InferenceParakeetStage currently supports backend='nemo' only; "
                "the TensorRT encoder runtime is not installed in Curator's standard audio environment"
            )
            raise ValueError(msg)
        if self.tensorrt_engine_dir is not None:
            msg = "tensorrt_engine_dir is only valid with backend='tensorrt'"
            raise ValueError(msg)
        if self.chunking_mode not in {"engine", "none"}:
            msg = f"Unsupported Parakeet chunking mode: {self.chunking_mode!r}"
            raise ValueError(msg)
        accepted_languages = self.supported_langs or PARAKEET_TDT_0_6B_V3_LANGS
        self.supported_language_codes = sorted(accepted_languages)
        self.adapter_kwargs = {
            "empty_audio_marks_skip": False,
            "use_cuda_graph_decoder": False,
        }
        super().__post_init__()

    def outputs(self) -> tuple[list[str], list[str]]:
        """Match the integration-stage output declaration."""
        return [], [self.pred_text_key, self.language_key]


__all__ = ["PARAKEET_TDT_0_6B_V3_LANGS", "InferenceParakeetStage", "NeMoASRAdapter"]
