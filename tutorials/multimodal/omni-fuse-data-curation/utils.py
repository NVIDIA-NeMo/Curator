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

"""Shared helpers for the numbered Omni-Fuse tutorial scripts: each step is one NeMo Curator pipeline."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.tasks import EmptyTask

from omnifuse_tutorial.config import ExperimentConfig, languagebind_root, load_config
from omnifuse_tutorial.data import (
    load_records,
    make_batch,
    read_jsonl,
    task_records,
    write_json,
    write_jsonl,
)
from omnifuse_tutorial.stages import (
    DatablendRankingStage,
    EEEEmbeddingStage,
    PairManifestReaderStage,
    ProjectionTrainingStage,
    SNSStage,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def main(description: str, step: Callable[[ExperimentConfig], dict[str, Any]]) -> None:
    """Parse --config, record config.resolved.json in the run directory, run the step, and print its outputs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the Omni-Fuse tutorial YAML config")
    config = load_config(parser.parse_args().config)
    write_json(config.run_dir / "config.resolved.json", config.to_dict())
    print(json.dumps(step(config), indent=2, sort_keys=True))


def validate_inputs(config: ExperimentConfig) -> dict[str, Any]:
    records = load_records(config.data_pools)
    missing = [] if config.eee.api_key else ["NV_BUILD_API_KEY"]
    if "fusion" in config.eee.experts and not languagebind_root().exists():
        missing.append("LANGUAGEBIND_ROOT")
    video_forward = config.sns.direction in {"forward", "bidirectional"} and any(
        pool.modality == "video" for pool in config.data_pools
    )
    if video_forward and not config.sns.cg_detr_checkpoint.exists():
        missing.append(str(config.sns.cg_detr_checkpoint))
    if missing:
        raise RuntimeError("Missing required API keys or local assets: " + ", ".join(missing))
    parallel = config.parallelism
    return {
        "experiment_id": config.experiment_id,
        "run_dir": str(config.run_dir),
        "records": len(records),
        "modalities": sorted({record["modality"] for record in records}),
        "sns_embedding_batch_size": config.sns.embedding_batch_size,
        "experts": config.eee.experts,
        "parallelism": {
            "enabled": parallel.enabled,
            "records_per_shard": parallel.records_per_shard,
            "sns_workers": parallel.sns_workers,
            "eee_workers": parallel.eee_workers,
            "required_gpus": parallel.required_gpus,
        },
    }


def run_sns(config: ExperimentConfig) -> dict[str, Any]:
    task = _run(config, "1-sns", [PairManifestReaderStage(config), SNSStage(config)], EmptyTask())
    records = write_jsonl(config.run_dir / "sns" / "records.jsonl", task_records(task))
    return {**_outputs(config, task, "sns_manifest_path"), "sns_records_path": str(records)}


def run_eee(config: ExperimentConfig) -> dict[str, Any]:
    task = _run(config, "2-embed", [EEEEmbeddingStage(config)], _load(config, "sns", "1_sns.py"))
    records = write_jsonl(config.run_dir / "embeddings" / "records.jsonl", task_records(task))
    return {**_outputs(config, task, "embedding_metadata_path"), "embedding_records_path": str(records)}


def run_projection(config: ExperimentConfig) -> dict[str, Any]:
    task = _run(config, "3-project", [ProjectionTrainingStage(config)], _load(config, "embeddings", "2_embed.py"))
    records = write_jsonl(config.run_dir / "projection" / "records.jsonl", task_records(task))
    keys = ("projection_model_path", "projection_metrics_path", "projected_embeddings_path")
    return {**_outputs(config, task, *keys), "projection_records_path": str(records)}


def run_datablend(config: ExperimentConfig) -> dict[str, Any]:
    task = _run(config, "4-datablend", [DatablendRankingStage(config)], _load(config, "projection", "3_project.py"))
    return _outputs(config, task, "datablend_ranked_path", "datablend_topk_path", "datablend_size")


def _run(config: ExperimentConfig, step: str, stages: list[Any], initial_task: Any) -> Any:
    name = f"{config.experiment_id}-{step}"
    tasks = Pipeline(name=name, description=f"Omni-Fuse tutorial step: {name}", stages=stages).run(
        initial_tasks=[initial_task]
    )
    if not tasks:
        raise RuntimeError(f"Curator pipeline {name} produced no output tasks")
    return tasks[-1]


def _load(config: ExperimentConfig, step: str, script: str) -> Any:
    path = config.run_dir / step / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {script} first.")
    return make_batch(config.experiment_id, read_jsonl(path), {"experiment_id": config.experiment_id})


def _outputs(config: ExperimentConfig, task: Any, *keys: str) -> dict[str, Any]:
    return {"run_dir": str(config.run_dir), **{key: task._metadata.get(key) for key in keys}}
