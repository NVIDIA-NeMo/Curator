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

"""Shared helpers for the Omni-Fuse data curation tutorial scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from omnifuse_tutorial.compat.curator import (
    make_curator_pipeline,
    make_document_batch,
    make_empty_task,
    records_from_task,
)
from omnifuse_tutorial.config.loader import load_config
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import read_jsonl, write_json, write_jsonl
from omnifuse_tutorial.data.loader import load_all_pools
from omnifuse_tutorial.eee.results import EmbeddingBundle
from omnifuse_tutorial.projection.trainer import ProjectionResult
from omnifuse_tutorial.stages import (
    DatablendRankingStage,
    EEEEmbeddingStage,
    PairManifestReaderStage,
    ProjectionTrainingStage,
    SNSStage,
)


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the Omni-Fuse tutorial YAML config")
    return parser


def load_tutorial_config(config_path: str | Path) -> ExperimentConfig:
    config = load_config(config_path)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.resolved.json", config.to_dict())
    return config


def run_curator_step(name: str, stages: list[Any], initial_task: Any) -> Any:
    """Run one numbered tutorial step as a real NeMo Curator Pipeline."""

    pipeline = make_curator_pipeline(
        name=name,
        description=f"Omni-Fuse tutorial step: {name}",
        stages=stages,
    )
    tasks = pipeline.run(initial_tasks=[initial_task])
    if not tasks:
        raise RuntimeError(f"Curator pipeline {name} produced no output tasks")
    return tasks[-1]


def run_reader(config: ExperimentConfig) -> Any:
    return run_curator_step(
        name=f"{config.experiment_id}-0-read-pairs",
        stages=[PairManifestReaderStage(config=config)],
        initial_task=make_empty_task(),
    )


def run_sns(config: ExperimentConfig) -> Any:
    task = run_curator_step(
        name=f"{config.experiment_id}-1-sns",
        stages=[SNSStage(config=config)],
        initial_task=run_reader(config),
    )
    records_path = config.run_dir / "sns" / "records.jsonl"
    write_jsonl(records_path, records_from_task(task))
    metadata = dict(getattr(task, "_metadata", {}) or {})
    metadata["sns_records_path"] = str(records_path)
    task._metadata = metadata
    return task


def load_sns_task(config: ExperimentConfig) -> Any:
    records_path = config.run_dir / "sns" / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing {records_path}. Run 1_sns.py first.")
    records = read_jsonl(records_path)
    return make_document_batch(
        task_id=f"{config.experiment_id}_sns",
        dataset_name=config.experiment_id,
        records=records,
        metadata={
            "experiment_id": config.experiment_id,
            "sns_records_path": str(records_path),
            "sns_manifest_path": str(config.run_dir / "sns" / "manifest.jsonl"),
        },
    )


def run_eee(config: ExperimentConfig) -> Any:
    task = run_curator_step(
        name=f"{config.experiment_id}-2-embed",
        stages=[EEEEmbeddingStage(config=config)],
        initial_task=load_sns_task(config),
    )
    records_path = config.run_dir / "embeddings" / "records.jsonl"
    write_jsonl(records_path, records_from_task(task))
    metadata = dict(getattr(task, "_metadata", {}) or {})
    metadata["embedding_records_path"] = str(records_path)
    task._metadata = metadata
    return task


def load_embedding_task(config: ExperimentConfig) -> Any:
    records_path = config.run_dir / "embeddings" / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing {records_path}. Run 2_embed.py first.")
    records = read_jsonl(records_path)
    bundle = load_embedding_bundle(config, records)
    return make_document_batch(
        task_id=f"{config.experiment_id}_embeddings",
        dataset_name=config.experiment_id,
        records=records,
        metadata={
            "experiment_id": config.experiment_id,
            "embedding_bundle": bundle,
            "embedding_records_path": str(records_path),
            "embedding_metadata_path": str(config.run_dir / "embeddings" / "metadata.json"),
        },
    )


def load_embedding_bundle(config: ExperimentConfig, records: list[dict[str, Any]] | None = None) -> EmbeddingBundle:
    metadata_path = config.run_dir / "embeddings" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run 2_embed.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if records is None:
        records = read_jsonl(config.run_dir / "embeddings" / "records.jsonl")
    embeddings: dict[str, list[list[float]]] = {}
    for expert in metadata["experts"]:
        safe_name = expert.replace("-", "_")
        path = config.run_dir / "embeddings" / f"{safe_name}_interleaved.npy"
        embeddings[expert] = np.load(path).astype(float).tolist()
    return EmbeddingBundle(
        pair_ids=list(metadata["pair_ids"]),
        modalities=list(metadata["modalities"]),
        records=records,
        experts=list(metadata["experts"]),
        embeddings=embeddings,
    )


def run_projection(config: ExperimentConfig) -> Any:
    task = run_curator_step(
        name=f"{config.experiment_id}-3-project",
        stages=[ProjectionTrainingStage(config=config)],
        initial_task=load_embedding_task(config),
    )
    records_path = config.run_dir / "projection" / "records.jsonl"
    write_jsonl(records_path, records_from_task(task))
    metadata = dict(getattr(task, "_metadata", {}) or {})
    metadata["projection_records_path"] = str(records_path)
    task._metadata = metadata
    return task


def load_projection_task(config: ExperimentConfig) -> Any:
    records_path = config.run_dir / "projection" / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing {records_path}. Run 3_project.py first.")
    records = read_jsonl(records_path)
    projection = load_projection_result(config)
    return make_document_batch(
        task_id=f"{config.experiment_id}_projection",
        dataset_name=config.experiment_id,
        records=records,
        metadata={
            "experiment_id": config.experiment_id,
            "projection_result": projection,
            "projected_embeddings_path": str(config.run_dir / "projection" / "projected_embeddings.npy"),
            "annotation_embeddings_path": str(config.run_dir / "projection" / "annotation_embeddings.npy"),
            "projection_model_path": str(config.run_dir / "projection" / "model.json"),
            "projection_loss_path": str(config.run_dir / "projection" / "loss_history.json"),
            "projection_metrics_path": str(config.run_dir / "projection" / "metrics.json"),
        },
    )


def load_projection_result(config: ExperimentConfig) -> ProjectionResult:
    output_dir = config.run_dir / "projection"
    model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
    loss_payload = json.loads((output_dir / "loss_history.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    projected = np.load(output_dir / "projected_embeddings.npy").astype(float).tolist()
    annotations = np.load(output_dir / "annotation_embeddings.npy").astype(float).tolist()
    expert_weights = model.get("expert_weights")
    if not isinstance(expert_weights, dict):
        experts = model.get("experts") or []
        expert_weights = {expert: 1.0 / len(experts) for expert in experts} if experts else {}
    return ProjectionResult(
        projected_raw=projected,
        annotation_embeddings=annotations,
        expert_weights={str(key): float(value) for key, value in expert_weights.items()},
        loss_history=[float(value) for value in loss_payload.get("loss", [])],
        recall_at_10={str(key): float(value) for key, value in metrics_payload.get("recall_at_10", {}).items()},
        model=model,
    )


def run_datablend(config: ExperimentConfig) -> Any:
    return run_curator_step(
        name=f"{config.experiment_id}-4-datablend",
        stages=[DatablendRankingStage(config=config)],
        initial_task=load_projection_task(config),
    )


def validate_inputs(config: ExperimentConfig) -> dict[str, Any]:
    records = load_all_pools(config.data_pools)
    missing: list[str] = []
    effective_sns_backend = config.eee.backend if config.sns.backend == "auto" else config.sns.backend
    if config.eee.backend in {"hybrid", "api"} or effective_sns_backend in {"hybrid", "api"}:
        if not (config.eee.nvidia_api_key or os.environ.get("NV_BUILD_API_KEY") or os.environ.get("NVIDIA_API_KEY")):
            missing.append("NV_BUILD_API_KEY")
    if config.eee.backend in {"hybrid", "local"} and "fusion" in config.eee.experts:
        languagebind_value = os.environ.get("LANGUAGEBIND_ROOT")
        languagebind_root = Path(languagebind_value).expanduser() if languagebind_value else None
        default_languagebind_root = Path(__file__).resolve().parent / "third_party" / "LanguageBind"
        if (languagebind_root is None or not languagebind_root.exists()) and not default_languagebind_root.exists():
            missing.append("LANGUAGEBIND_ROOT")
    uses_video_forward = any(pool.modality == "video" for pool in config.data_pools) and config.sns.direction in {
        "forward",
        "bidirectional",
    }
    if (
        effective_sns_backend in {"hybrid", "local"}
        and uses_video_forward
        and not config.sns.cg_detr_checkpoint.exists()
    ):
        missing.append(str(config.sns.cg_detr_checkpoint))
    if missing:
        raise RuntimeError("Missing required API keys or local assets: " + ", ".join(missing))
    return {
        "experiment_id": config.experiment_id,
        "run_dir": str(config.run_dir),
        "records": len(records),
        "modalities": sorted({record["modality"] for record in records}),
        "sns_backend": config.sns.backend,
        "eee_backend": config.eee.backend,
        "experts": config.eee.experts,
    }


def print_outputs(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
