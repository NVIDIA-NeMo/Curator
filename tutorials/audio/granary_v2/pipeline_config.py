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

"""Build the production Granary-v2 stage contract without importing model code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ModelName = Literal[
    "qwen_omni",
    "qwen_asr",
    "parakeet_v3",
    "whisper",
    "parakeet_riva",
    "indic_monolingual",
    "none",
]

_ASR_STAGE = "nemo_curator.stages.audio.inference.asr.stage.ASRStage"
_QWEN_ADAPTERS = {
    "qwen_omni": "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter",
    "qwen_asr": "nemo_curator.models.asr.qwen_asr.QwenASRAdapter",
}


@dataclass(frozen=True)
class GranaryV2PipelineSettings:
    """Resolved settings for one Granary-v2 pipeline run."""

    input_config: str
    output_dir: str
    language: str
    primary_model: ModelName
    recovery_model: ModelName
    regex_yaml: str
    hallucination_phrases: str
    qwen_omni_model_id: str
    qwen_asr_model_id: str
    parakeet_v3_model_id: str
    parakeet_riva_model_id: str
    indic_model_id: str
    whisper_model_size_or_path: str = "large-v3"
    sed_checkpoint: str | None = None
    reader_concurrency: int = 2
    primary_batch_size: int = 64
    recovery_batch_size: int = 128

    def __post_init__(self) -> None:
        if self.primary_model == "none":
            msg = "Granary-v2 requires a primary ASR model"
            raise ValueError(msg)
        if self.primary_batch_size <= 0 or self.recovery_batch_size <= 0:
            msg = "ASR batch sizes must be positive"
            raise ValueError(msg)


def _resources(*, gpus: float = 0.0, gpu_memory_gb: float | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "_target_": "nemo_curator.stages.resources.Resources",
        "cpus": 1.0,
        "gpus": gpus,
    }
    if gpu_memory_gb is not None:
        config["gpu_memory_gb"] = gpu_memory_gb
    return config


def _nemo_model_args(model_id: str) -> dict[str, str]:
    if model_id.endswith(".nemo"):
        return {"model_path": model_id}
    return {"model_name": model_id}


def _model_stage(
    model: ModelName,
    *,
    role: Literal["primary", "recovery"],
    settings: GranaryV2PipelineSettings,
) -> dict[str, Any]:
    pred_text_key = "primary_model_prediction" if role == "primary" else "fallback_model_prediction"
    keep_waveform = role == "primary" and settings.recovery_model != "none"
    batch_size = settings.primary_batch_size if role == "primary" else settings.recovery_batch_size
    name = f"{model}_{role}"

    if model in _QWEN_ADAPTERS:
        model_id = settings.qwen_omni_model_id if model == "qwen_omni" else settings.qwen_asr_model_id
        adapter_kwargs: dict[str, Any]
        if model == "qwen_omni":
            adapter_kwargs = {
                "prompt_text": "Transcribe the audio.",
                "max_output_tokens": 256,
                "max_model_len": 4096,
                "max_num_seqs": 128,
                "gpu_memory_utilization": 0.85,
                "temperature": 0.0,
                "top_k": 1,
            }
        else:
            adapter_kwargs = {
                "max_new_tokens": 4096,
                "max_inference_batch_size": batch_size,
                "gpu_memory_utilization": 0.95,
            }
        return {
            "_target_": _ASR_STAGE,
            "name": name,
            "adapter_target": _QWEN_ADAPTERS[model],
            "model_id": model_id,
            "audio_filepath_key": "audio_filepath",
            "source_lang_key": "source_lang",
            "default_language": settings.language,
            "pred_text_key": pred_text_key,
            "primary_model_value": model if role == "primary" else None,
            "batch_size": batch_size,
            "resources": _resources(gpus=2.0 if model == "qwen_omni" else 1.0),
            "adapter_kwargs": adapter_kwargs,
        }

    if model == "whisper":
        return {
            "_target_": ("nemo_curator.stages.audio.inference.faster_whisper.InferenceFasterWhisperStage"),
            "name": name,
            "model_size_or_path": settings.whisper_model_size_or_path,
            "source_lang_key": "source_lang",
            "waveform_key": "waveform",
            "sample_rate_key": "sampling_rate",
            "pred_text_key": pred_text_key,
            "keep_waveform": keep_waveform,
            "batch_size": batch_size,
            "resources": _resources(gpus=1.0),
        }

    if model == "indic_monolingual":
        model_id = settings.indic_model_id.format(language=settings.language)
        return {
            "_target_": (
                "nemo_curator.stages.audio.inference.indic_conformer_hybrid.InferenceIndicConformerHybridStage"
            ),
            "name": name,
            "model_id": model_id,
            "source_lang_key": "source_lang",
            "waveform_key": "waveform",
            "sample_rate_key": "sampling_rate",
            "pred_text_key": pred_text_key,
            "keep_waveform": keep_waveform,
            "batch_size": batch_size,
            "resources": _resources(gpus=1.0),
        }

    if model in {"parakeet_v3", "parakeet_riva"}:
        model_id = settings.parakeet_v3_model_id if model == "parakeet_v3" else settings.parakeet_riva_model_id
        return {
            "_target_": ("nemo_curator.stages.audio.inference.asr.asr_nemo.InferenceAsrNemoStage"),
            "name": name,
            **_nemo_model_args(model_id),
            "filepath_key": "audio_filepath",
            "pred_text_key": pred_text_key,
            "batch_size": batch_size,
            "resources": _resources(gpus=1.0),
        }

    msg = f"Unsupported ASR model stage: {model}"
    raise ValueError(msg)


def build_stage_configs(settings: GranaryV2PipelineSettings) -> list[dict[str, Any]]:
    """Return the ordered, end-to-end Granary-v2 stage configuration.

    The returned dictionaries are Hydra-compatible. Model modules are named by
    string so this configuration PR remains source-mergeable before any of the
    individual stage PRs. Runtime activation requires those target PRs.
    """
    stages: list[dict[str, Any]] = [
        {
            "_target_": "nemo_curator.stages.audio.io.nemo_speech_reader.NeMoSpeechAudioReader",
            "yaml_path": settings.input_config,
            "output_dir": settings.output_dir,
            "read_concurrency": settings.reader_concurrency,
        },
        {
            "_target_": ("nemo_curator.stages.audio.text_filtering.initialize_fields.InitializeFieldsStage"),
            "default_source_lang": settings.language,
            "pipeline_notes": {
                "primary_model": settings.primary_model,
                "recovery_model": settings.recovery_model,
            },
        },
    ]

    if settings.sed_checkpoint:
        stages.extend(
            [
                {
                    "_target_": "nemo_curator.stages.audio.inference.sed.SEDInferenceStage",
                    "checkpoint_path": settings.sed_checkpoint,
                    "waveform_key": "waveform",
                    "sample_rate_key": "sampling_rate",
                    "resources": _resources(gpu_memory_gb=4.0),
                },
                {
                    "_target_": ("nemo_curator.stages.audio.postprocessing.sed_postprocessing.SEDPostprocessingStage"),
                },
            ]
        )

    stages.extend(
        [
            _model_stage(settings.primary_model, role="primary", settings=settings),
            {
                "_target_": (
                    "nemo_curator.stages.audio.text_filtering.whisper_hallucination.WhisperHallucinationStage"
                ),
                "name": "WhisperHallucination_primary",
                "common_hall_file": settings.hallucination_phrases,
                "language_key": "source_lang",
                "text_key": "primary_model_prediction",
            },
        ]
    )

    if settings.recovery_model != "none":
        stages.extend(
            [
                _model_stage(settings.recovery_model, role="recovery", settings=settings),
                {
                    "_target_": (
                        "nemo_curator.stages.audio.text_filtering.whisper_hallucination.WhisperHallucinationStage"
                    ),
                    "name": "WhisperHallucination_recovery",
                    "common_hall_file": settings.hallucination_phrases,
                    "language_key": "source_lang",
                    "text_key": "fallback_model_prediction",
                    "overwrite": True,
                    "recovery_value": "Recovered:ASR",
                },
            ]
        )

    stages.extend(
        [
            {
                "_target_": (
                    "nemo_curator.stages.audio.text_filtering.select_best_prediction.SelectBestPredictionStage"
                ),
                "primary_text_key": "primary_model_prediction",
                "fallback_text_key": "fallback_model_prediction",
                "output_key": "best_prediction",
                "reference_text_key": "granary_v1_prediction",
            },
            {
                "_target_": ("nemo_curator.stages.audio.text_filtering.regex_substitution.RegexSubstitutionStage"),
                "regex_params_yaml": settings.regex_yaml,
                "text_key": "best_prediction",
                "output_text_key": "text",
            },
            {
                "_target_": ("nemo_curator.stages.audio.text_filtering.abbreviation_concat.AbbreviationConcatStage"),
                "text_key": "text",
                "output_text_key": "text",
            },
            {
                "_target_": ("nemo_curator.stages.audio.alm.sharded_manifest_writer.ShardedManifestWriterStage"),
                "output_dir": settings.output_dir,
            },
        ]
    )
    return stages
