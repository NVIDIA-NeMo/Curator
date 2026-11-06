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

"""Experiment configuration dataclasses and YAML/JSON loading."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
EXPERTS = ("text-based", "fusion", "e2e")


def languagebind_root() -> Path:
    return (
        Path(os.environ.get("LANGUAGEBIND_ROOT") or TUTORIAL_DIR / "third_party" / "LanguageBind")
        .expanduser()
        .resolve()
    )


def _at_least_one(section: str, config: Any, *names: str) -> None:
    for name in names:
        if getattr(config, name) < 1:
            raise ValueError(f"{section}.{name} must be at least 1")


@dataclass
class DataPoolConfig:
    name: str
    modality: Literal["text", "image", "video", "audio", "point-cloud"]
    root_dir: Path
    mapping_file: str = "pair_mapping.jsonl"
    n_samples: int | None = None
    shuffle: bool = False
    max_file_size_mb: int | None = None


@dataclass
class SNSConfig:
    enabled: bool = True
    continue_on_error: bool = True
    direction: Literal["forward", "backward", "bidirectional"] = "bidirectional"
    mi_ratio: float = 0.95
    mi_eps: float = 0.05
    tau_forward_text: float = 0.30
    tau_forward_image: float = 0.30
    tau_forward_video: float = 0.20
    tau_forward_audio: float = 0.25
    tau_backward: float = 0.35
    max_video_segments: int = 5
    max_audio_segments: int = 5
    min_segment_duration: float = 2.0
    bbox_padding_px: int = 0
    sns_output_dir: Path | None = None
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    cg_detr_checkpoint: Path = Path("model_files/best.ckpt")
    amdetr_repo_id: str = "lighthouse-emnlp2024/AM-DETR"
    require_forward_models: bool = True
    use_ann_components: bool = True
    nvidia_model: str = "nvidia/omni-embed-nemotron-3b"
    embedding_batch_size: int = 16

    def __post_init__(self) -> None:
        _at_least_one("sns", self, "embedding_batch_size")


@dataclass
class EEEConfig:
    experts: list[str] = field(default_factory=lambda: list(EXPERTS))
    continue_on_error: bool = True
    embedding_dim: int = 2048
    nvidia_api_key: str | None = None
    nvidia_api_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_image_describer_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    nvidia_video_describer_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    nvidia_audio_describer_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    nvidia_embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    nvidia_multimodal_model: str = "nvidia/omni-embed-nemotron-3b"

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("eee.experts cannot be empty")
        if unknown := sorted(set(self.experts) - set(EXPERTS)):
            raise ValueError(f"Unsupported EEE experts: {unknown}")

    @property
    def api_key(self) -> str | None:
        return self.nvidia_api_key or os.environ.get("NV_BUILD_API_KEY") or os.environ.get("NVIDIA_API_KEY")


@dataclass
class ProjectionConfig:
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_gpus: int = 1
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
    save_weights_path: Path | None = None
    eval_recall_k: int = 10
    verbose: bool = False
    log_every_n_epochs: int = 100

    def __post_init__(self) -> None:
        if self.num_gpus < 0:
            raise ValueError("projection.num_gpus cannot be negative")
        if self.device == "cuda" and self.num_gpus == 0:
            raise ValueError("projection.device=cuda requires projection.num_gpus to be at least 1")
        _at_least_one("projection", self, "num_epochs", "batch_size", "log_every_n_epochs")


@dataclass
class DatablendConfig:
    query: str
    top_k: int | None = None
    blend_fraction: float | None = None
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("datablend.query is required")


@dataclass
class RuntimeConfig:
    device: str = "auto"


@dataclass
class ParallelismConfig:
    """Settings for the optional sharded SNS -> EEE streaming pipeline."""

    enabled: bool = False
    records_per_shard: int = 25
    sns_workers: int = 1
    eee_workers: int = 1
    sns_gpus_per_worker: float = 1.0
    eee_gpus_per_worker: float = 1.0
    autoscale_interval_s: int = 10
    logging_interval_s: int = 60

    def __post_init__(self) -> None:
        names = ("records_per_shard", "sns_workers", "eee_workers", "autoscale_interval_s", "logging_interval_s")
        _at_least_one("parallelism", self, *names)
        if self.sns_gpus_per_worker <= 0 or self.eee_gpus_per_worker <= 0:
            raise ValueError("parallelism GPU allocations must be greater than 0")

    @property
    def required_gpus(self) -> float:
        return self.sns_workers * self.sns_gpus_per_worker + self.eee_workers * self.eee_gpus_per_worker


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
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        if unknown := sorted(set(value) - {item.name for item in fields(cls)}):
            raise ValueError(f"Unknown top-level config keys: {unknown}")
        pools = [_section(DataPoolConfig, item, "root_dir") for item in value.get("data_pools") or []]
        if not pools:
            raise ValueError("data_pools cannot be empty")
        experiment_id = str(value.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ValueError("experiment_id is required")
        datablend = value.get("datablend") or {"query": "Describe the media content in detail"}
        return cls(
            experiment_id=experiment_id,
            output_dir=Path(value.get("output_dir", "outputs")),
            data_pools=pools,
            sns=_section(SNSConfig, value.get("sns"), "sns_output_dir", "cg_detr_checkpoint"),
            eee=_section(EEEConfig, value.get("eee")),
            projection=_section(ProjectionConfig, value.get("projection"), "save_weights_path"),
            datablend=_section(DatablendConfig, datablend),
            runtime=_section(RuntimeConfig, value.get("runtime")),
            parallelism=_section(ParallelismConfig, value.get("parallelism")),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _section(cls: type, value: dict[str, Any] | None, *path_keys: str) -> Any:
    """Strict section: unknown keys raise TypeError; empty path values keep the default."""
    value = dict(value or {})
    paths = {key: value.pop(key) for key in path_keys if key in value}
    return cls(**value, **{key: Path(item) for key, item in paths.items() if item})


def _jsonable(value: Any, key: str = "") -> Any:
    if value and any(marker in key.lower() for marker in ("api_key", "token", "secret", "password")):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {name: _jsonable(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML/JSON config; `.env` files in the cwd and config directory fill unset variables."""
    path = Path(path)
    for env_file in (Path.cwd() / ".env", path.parent / ".env"):
        if env_file.exists():
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                key, sep, item = line.partition("=")
                if sep and key.strip() and not line.startswith("#"):
                    os.environ.setdefault(key.strip(), item.strip().strip('"').strip("'"))
    suffix = path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError(f"Unsupported config suffix: {path.suffix}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    config = ExperimentConfig.from_dict(data)

    def resolve(item: Path | None) -> Path | None:
        return item if item is None or item.is_absolute() else (path.parent / item).resolve()

    config.output_dir = resolve(config.output_dir)
    config.sns.sns_output_dir = resolve(config.sns.sns_output_dir)
    config.sns.cg_detr_checkpoint = resolve(config.sns.cg_detr_checkpoint)
    config.projection.save_weights_path = resolve(config.projection.save_weights_path)
    for pool in config.data_pools:
        pool.root_dir = resolve(pool.root_dir)
    return config
