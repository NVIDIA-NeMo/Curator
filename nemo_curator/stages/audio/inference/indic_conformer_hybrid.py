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

"""Curator stage for AI4Bharat IndicConformer hybrid ASR checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nemo_curator.models.indic_conformer_hybrid import (
    INDIC_CONFORMER_HYBRID_LANGS,
    IndicConformerHybridASR,
)
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.resources import Resources

_ADAPTER_TARGET = "nemo_curator.models.indic_conformer_hybrid.IndicConformerHybridASR"


@dataclass
class InferenceIndicConformerHybridStage(ASRStage):
    """Transcribe in-memory audio with an IndicConformer hybrid checkpoint.

    This is the current Curator adapter-backed equivalent of the stage used by
    ``examples/audio/qwen_omni_inprocess/run_pipeline.py`` on the integration
    branch. The public model and task-routing options intentionally match that
    stage while model lifecycle remains in ``IndicConformerHybridASR``.
    """

    adapter_target: str = field(default=_ADAPTER_TARGET, init=False, repr=False)
    model_id: str = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
    name: str = "IndicConformerHybrid_inference"
    decode_mode: Literal["ctc", "rnnt"] = "rnnt"
    # Kept as a named compatibility option for integration-pipeline configs;
    # this adapter-backed port deliberately supports Curator's NeMo runtime.
    backend: Literal["nemo"] = "nemo"
    tensorrt_engine_dir: str | None = None
    rnnt_precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    keep_waveform: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128

    audio_filepath_key: str = field(default="", init=False, repr=False)
    target_sample_rate: int = field(default=16_000, init=False, repr=False)
    default_language: str | None = field(default=None, init=False, repr=False)
    supported_language_codes: list[str] = field(
        default_factory=lambda: sorted(INDIC_CONFORMER_HYBRID_LANGS),
        init=False,
        repr=False,
    )
    extras_key: str | None = field(default=None, init=False, repr=False)
    skip_me_key: str = field(default="_skipme", init=False, repr=False)
    unsupported_language_marks_skip: bool = field(default=False, init=False, repr=False)
    unsupported_language_skip_reason: str | None = field(default=None, init=False, repr=False)
    preserve_existing_skip: bool = field(default=False, init=False, repr=False)
    missing_language_is_unsupported: bool = field(default=True, init=False, repr=False)
    skip_if_output_exists: bool = field(default=False, init=False, repr=False)
    fail_on_audio_error: bool = field(default=True, init=False, repr=False)
    prefetch_fail_on_error: bool = field(default=True, init=False, repr=False)
    adapter_kwargs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend != "nemo":
            msg = (
                "InferenceIndicConformerHybridStage currently supports backend='nemo' only; "
                "the TensorRT encoder runtime is not installed in Curator's standard audio environment"
            )
            raise ValueError(msg)
        if self.tensorrt_engine_dir is not None:
            msg = "tensorrt_engine_dir is only valid with backend='tensorrt'"
            raise ValueError(msg)
        if self.decode_mode not in {"ctc", "rnnt"}:
            msg = f"Unsupported IndicConformer decode mode: {self.decode_mode!r}"
            raise ValueError(msg)
        if self.rnnt_precision not in {"fp32", "fp16", "bf16"}:
            msg = f"Unsupported IndicConformer RNNT precision: {self.rnnt_precision!r}"
            raise ValueError(msg)
        self.adapter_kwargs = {
            "decode_mode": self.decode_mode,
            "rnnt_precision": self.rnnt_precision,
            "empty_audio_marks_skip": False,
        }
        super().__post_init__()

    def outputs(self) -> tuple[list[str], list[str]]:
        """Match the integration-stage output declaration."""
        return [], [self.pred_text_key, self.language_key]


__all__ = [
    "INDIC_CONFORMER_HYBRID_LANGS",
    "IndicConformerHybridASR",
    "InferenceIndicConformerHybridStage",
]
