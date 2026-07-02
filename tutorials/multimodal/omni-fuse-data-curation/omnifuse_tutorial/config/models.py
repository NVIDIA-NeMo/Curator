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

"""Configuration dataclasses for the standalone Omni-Fuse Curator pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Modality = Literal["text", "image", "video", "audio", "point-cloud"]
SNSDirection = Literal["forward", "backward", "bidirectional"]
ExpertName = Literal["text-based", "fusion", "e2e"]


@dataclass
class DataPoolConfig:
    name: str
    modality: Modality
    root_dir: Path
    mapping_file: str = "pair_mapping.jsonl"
    n_samples: int | None = None
    shuffle: bool = False
    max_file_size_mb: int | None = None
    max_video_frames: int | None = None
    max_audio_duration_seconds: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DataPoolConfig:
        name = value.get("name", value.get("title"))
        modality = value.get("modality", value.get("data_modality"))
        root_dir = value.get("root_dir", value.get("data_root_dir"))
        if name is None:
            raise ValueError(f"Data pool missing name/title: {value}")
        if modality is None:
            raise ValueError(f"Data pool missing modality/data_modality: {value}")
        if root_dir is None:
            raise ValueError(f"Data pool missing root_dir/data_root_dir: {value}")
        return cls(
            name=str(name),
            modality=modality,
            root_dir=Path(root_dir),
            mapping_file=str(value.get("mapping_file", "pair_mapping.jsonl")),
            n_samples=value.get("n_samples"),
            shuffle=bool(value.get("shuffle", False)),
            max_file_size_mb=value.get("max_file_size_mb"),
            max_video_frames=value.get("max_video_frames"),
            max_audio_duration_seconds=value.get("max_audio_duration_seconds"),
        )


@dataclass
class SNSConfig:
    enabled: bool = True
    backend: Literal["auto", "hybrid", "local", "api"] = "auto"
    direction: SNSDirection = "bidirectional"
    mi_ratio: float = 0.95
    mi_eps: float = 0.05
    tau_forward_text: float = 0.30
    tau_forward_image: float = 0.30
    tau_forward_video: float = 0.20
    tau_forward_audio: float = 0.25
    tau_backward: float = 0.35
    grid_size: int = 5
    max_patches: int = 4
    max_video_segments: int = 5
    max_audio_segments: int = 5
    min_segment_duration: float = 2.0
    bbox_padding_px: int = 0
    reinject: bool = False
    sns_output_dir: Path | None = None
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    cg_detr_checkpoint: Path = Path("model_files/best.ckpt")
    amdetr_repo_id: str = "lighthouse-emnlp2024/AM-DETR"
    require_forward_models: bool = True
    use_ann_components: bool = True
    nvidia_model: str = "nvidia/omni-embed-nemotron-3b"

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> SNSConfig:
        value = value or {}
        sns_output_dir = value.get("sns_output_dir")
        cg_detr_checkpoint = value.get("cg_detr_checkpoint")
        cfg = cls(**{key: item for key, item in value.items() if key not in {"sns_output_dir", "cg_detr_checkpoint"}})
        if sns_output_dir:
            cfg.sns_output_dir = Path(sns_output_dir)
        if cg_detr_checkpoint:
            cfg.cg_detr_checkpoint = Path(cg_detr_checkpoint)
        return cfg


@dataclass
class EEEConfig:
    experts: list[ExpertName] = field(default_factory=lambda: ["text-based", "fusion", "e2e"])
    backend: Literal["hybrid", "local", "api"] = "hybrid"
    embedding_dim: int = 2048
    batch_size: int = 32
    text_prompt_base: str = "Describe this in detail."
    text_prompt_prefix: str = "Focus specifically on the aspects highlighted in this annotation."
    nvidia_api_key: str | None = None
    nvidia_api_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_text_describer_model: str = "nvidia/nemotron-nano-12b-v2-vl"
    nvidia_image_describer_model: str = "nvidia/nemotron-nano-12b-v2-vl"
    nvidia_video_describer_model: str = "nvidia/nemotron-nano-12b-v2-vl"
    nvidia_audio_describer_model: str = "google/gemma-3n-e4b-it"
    nvidia_embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    nvidia_multimodal_model: str = "nvidia/omni-embed-nemotron-3b"

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> EEEConfig:
        value = dict(value or {})
        if "text_expert_backend" in value and "backend" not in value:
            value["backend"] = value.pop("text_expert_backend")
        cfg = cls(**value)
        if not cfg.experts:
            raise ValueError("eee.experts cannot be empty")
        if cfg.backend not in {"hybrid", "local", "api"}:
            raise ValueError(f"Unsupported eee.backend: {cfg.backend}")
        return cfg


@dataclass
class ProjectionConfig:
    enabled: bool = True
    backend: Literal["auto", "linear", "torch"] = "auto"
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    contrastive_loss_weight: float = 0.99
    scale_loss_weight: float = 0.0001
    bias_loss_weight: float = 0.01
    contrastive_temperature: float = 0.07
    num_layers: int = 3
    hidden_layer_size: int = 512
    dropout: float = 0.1
    output_embeddings: bool = True
    save_weights_path: Path | None = None
    eval_recall_k: int = 10
    verbose: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> ProjectionConfig:
        value = dict(value or {})
        save_weights_path = value.get("save_weights_path")
        cfg = cls(**{key: item for key, item in value.items() if key != "save_weights_path"})
        if save_weights_path:
            cfg.save_weights_path = Path(save_weights_path)
        return cfg


@dataclass
class DatablendConfig:
    query: str
    top_k: int | None = None
    blend_fraction: float | None = None
    include_metadata: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> DatablendConfig:
        if not value or not value.get("query"):
            raise ValueError("datablend.query is required")
        return cls(
            query=str(value["query"]),
            top_k=value.get("top_k"),
            blend_fraction=value.get("blend_fraction"),
            include_metadata=bool(value.get("include_metadata", True)),
        )


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "float32"
    offline_mode: bool = False
    cache_dir: Path | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> RuntimeConfig:
        value = value or {}
        cache_dir = value.get("cache_dir")
        return cls(
            device=value.get("device", "auto"),
            dtype=value.get("dtype", "float32"),
            offline_mode=bool(value.get("offline_mode", False)),
            cache_dir=Path(cache_dir) if cache_dir else None,
        )


@dataclass
class ExperimentConfig:
    experiment_id: str
    output_dir: Path
    data_pools: list[DataPoolConfig]
    sns: SNSConfig
    eee: EEEConfig
    projection: ProjectionConfig
    datablend: DatablendConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    description: str = ""
    embedsim_config_name: str = ""
    reranking_enabled: bool = True
    random_shuffle: bool = False
    strict_data_validation: bool = False
    downstream_eval: dict[str, Any] = field(default_factory=dict)
    log_wandb: bool = False
    log_local: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        value = dict(value)
        data_pool_values = value.get("data_pools", value.get("data_pools_config", []))
        data_pools = [DataPoolConfig.from_dict(item) for item in data_pool_values]
        if not data_pools:
            raise ValueError("data_pools cannot be empty")
        random_shuffle = bool(value.get("random_shuffle", False))
        if random_shuffle:
            for pool in data_pools:
                pool.shuffle = True
        experiment_id = str(value.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ValueError("experiment_id is required")
        eee_value = _eee_value_from_experiment(value)
        projection_value = value.get("projection", value.get("awn"))
        datablend_value = (
            value.get("datablend")
            or _datablend_from_downstream(value.get("downstream_eval"))
            or {"query": "Describe the media content in detail"}
        )
        return cls(
            experiment_id=experiment_id,
            description=str(value.get("description", "")),
            output_dir=Path(value.get("output_dir", value.get("experiment_dir", "outputs"))),
            data_pools=data_pools,
            sns=SNSConfig.from_dict(value.get("sns")),
            eee=EEEConfig.from_dict(eee_value),
            projection=ProjectionConfig.from_dict(projection_value),
            datablend=DatablendConfig.from_dict(datablend_value),
            runtime=RuntimeConfig.from_dict(value.get("runtime")),
            embedsim_config_name=str(value.get("embedsim_config_name", "")),
            reranking_enabled=bool(value.get("reranking_enabled", True)),
            random_shuffle=random_shuffle,
            strict_data_validation=bool(value.get("strict_data_validation", False)),
            downstream_eval=dict(value.get("downstream_eval") or {}),
            log_wandb=bool(value.get("log_wandb", False)),
            log_local=bool(value.get("log_local", True)),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return _stringify_paths(value)


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_or_stringify(key, item) for key, item in value.items()}
    return value


def _redact_or_stringify(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in ("api_key", "token", "secret", "password")) and value:
        return "***REDACTED***"
    return _stringify_paths(value)


def _eee_value_from_experiment(value: dict[str, Any]) -> dict[str, Any] | None:
    eee = dict(value.get("eee") or {})
    if "experts" in value and "experts" not in eee:
        eee["experts"] = [item.value if hasattr(item, "value") else str(item) for item in value["experts"]]
    if "text_expert_backend" in value and "backend" not in eee:
        backend = value["text_expert_backend"]
        eee["backend"] = backend.value if hasattr(backend, "value") else str(backend)
    for key in (
        "text_prompt_base",
        "text_prompt_prefix",
        "nvidia_api_key",
        "nvidia_api_base_url",
        "nvidia_text_describer_model",
        "nvidia_image_describer_model",
        "nvidia_video_describer_model",
        "nvidia_audio_describer_model",
        "nvidia_embedding_model",
    ):
        if key in value and key not in eee:
            eee[key] = value[key]
    return eee or None


def _datablend_from_downstream(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    query = value.get("query")
    if not query:
        return None
    result: dict[str, Any] = {"query": query}
    if value.get("train_count") is not None:
        result["top_k"] = value["train_count"]
    if value.get("blend_fraction") is not None:
        result["blend_fraction"] = value["blend_fraction"]
    return result
