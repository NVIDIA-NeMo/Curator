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

"""Sharded, multi-GPU SNS -> EEE pipeline for the Omni-Fuse tutorial."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.resources import Resources

from omnifuse_tutorial.compat.curator import make_curator_pipeline, make_document_batch, records_from_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import ensure_dir, read_jsonl, write_json, write_jsonl
from omnifuse_tutorial.data.loader import load_all_pools
from omnifuse_tutorial.stages import EEEEmbeddingStage, SNSStage


def create_record_shards(
    config: ExperimentConfig,
    records: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Split ordered input records into deterministic contiguous Curator tasks."""

    records = list(records) if records is not None else load_all_pools(config.data_pools)
    if not records:
        raise ValueError("Parallel SNS/EEE requires at least one input record")
    records_per_shard = config.parallelism.records_per_shard
    shard_count = math.ceil(len(records) / records_per_shard)
    tasks: list[Any] = []
    for shard_index, start in enumerate(range(0, len(records), records_per_shard)):
        stop = min(start + records_per_shard, len(records))
        tasks.append(
            make_document_batch(
                task_id=f"{config.experiment_id}_shard_{shard_index:05d}",
                dataset_name=config.experiment_id,
                records=records[start:stop],
                metadata={
                    "experiment_id": config.experiment_id,
                    "parallel_pipeline": True,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "record_start": start,
                    "record_stop": stop,
                },
            )
        )
    return tasks


def build_parallel_stages(config: ExperimentConfig) -> list[Any]:
    """Build fixed GPU stages so SNS and EEE stay resident on separate GPUs."""

    parallel = config.parallelism
    return [
        SNSStage(
            config=config,
            resources=Resources(cpus=1.0, gpus=parallel.sns_gpus_per_worker),
            sharded_outputs=True,
            worker_count=parallel.sns_workers,
            slots_per_actor=1,
        ),
        EEEEmbeddingStage(
            config=config,
            resources=Resources(cpus=1.0, gpus=parallel.eee_gpus_per_worker),
            sharded_outputs=True,
            worker_count=parallel.eee_workers,
            slots_per_actor=1,
        ),
    ]


def validate_parallel_gpu_capacity(config: ExperimentConfig, available_gpus: int | None = None) -> None:
    """Fail before scheduling if the local tutorial machine lacks enough GPUs."""

    if available_gpus is None:
        import torch

        available_gpus = torch.cuda.device_count()
    required_gpus = config.parallelism.required_gpus
    if available_gpus + 1e-9 < required_gpus:
        raise RuntimeError(
            f"Parallel SNS/EEE requests {required_gpus:g} GPUs but only {available_gpus} are visible. "
            "Reduce parallelism worker/GPU counts or expose more GPUs."
        )


def run_parallel_sns_eee(
    config: ExperimentConfig,
    records: list[dict[str, Any]] | None = None,
    executor: Any | None = None,
) -> dict[str, Any]:
    """Stream record shards through SNS and EEE, then merge canonical outputs."""

    if not config.parallelism.enabled:
        raise ValueError("Set parallelism.enabled: true to run the parallel SNS/EEE tutorial step")
    if executor is None:
        validate_parallel_gpu_capacity(config)
        executor = XennaExecutor(
            config={
                "execution_mode": "streaming",
                "autoscale_interval_s": config.parallelism.autoscale_interval_s,
                "logging_interval": config.parallelism.logging_interval_s,
            }
        )

    initial_tasks = create_record_shards(config, records)
    pipeline = make_curator_pipeline(
        name=f"{config.experiment_id}-1-2-parallel-sns-eee",
        description="Sharded pipeline-parallel Omni-Fuse SNS and EEE tutorial steps",
        stages=build_parallel_stages(config),
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    output_tasks = pipeline.run(executor=executor, initial_tasks=initial_tasks)
    if not output_tasks:
        raise RuntimeError("Parallel SNS/EEE pipeline produced no output tasks")

    summary = merge_parallel_outputs(config, output_tasks)
    completed_at = datetime.now(timezone.utc)
    summary.update(
        {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    summary_path = write_json(config.run_dir / "parallelism" / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary


def merge_parallel_outputs(config: ExperimentConfig, output_tasks: list[Any]) -> dict[str, Any]:
    """Merge shard artifacts in original record order for Steps 3 and 4."""

    tasks = _ordered_tasks(output_tasks)
    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    embedding_errors: list[dict[str, Any]] = []

    for task in tasks:
        metadata = _task_metadata(task)
        shard_records = records_from_task(task)
        shard_manifest = read_jsonl(_required_path(metadata, "sns_manifest_path"))
        if len(shard_manifest) != len(shard_records):
            raise ValueError(
                f"Shard {metadata['shard_index']} has {len(shard_records)} records but "
                f"{len(shard_manifest)} SNS manifest rows"
            )
        records.extend(shard_records)
        manifest_rows.extend(shard_manifest)
        errors_path = _required_path(metadata, "embedding_errors_path")
        errors = json.loads(errors_path.read_text(encoding="utf-8"))
        if not isinstance(errors, list):
            raise TypeError(f"Embedding errors must be a list: {errors_path}")
        embedding_errors.extend(errors)

    sns_manifest_path = write_jsonl(config.run_dir / "sns" / "manifest.jsonl", manifest_rows)
    sns_records_path = write_jsonl(config.run_dir / "sns" / "records.jsonl", records)
    embedding_records_path = write_jsonl(config.run_dir / "embeddings" / "records.jsonl", records)
    embedding_errors_path = write_json(config.run_dir / "embeddings" / "errors.json", embedding_errors)

    for expert in config.eee.experts:
        _merge_expert_arrays(config, tasks, expert, len(records))

    embedding_metadata_path = write_json(
        config.run_dir / "embeddings" / "metadata.json",
        {
            "pair_ids": [record["pair_id"] for record in records],
            "modalities": [record["modality"] for record in records],
            "experts": list(config.eee.experts),
            "embedding_dim": config.eee.embedding_dim,
            "error_count": len(embedding_errors),
            "errors_path": str(embedding_errors_path),
            "parallel_shards": len(tasks),
        },
    )
    return {
        "mode": "pipeline_parallel",
        "record_count": len(records),
        "shard_count": len(tasks),
        "records_per_shard": config.parallelism.records_per_shard,
        "sns_workers": config.parallelism.sns_workers,
        "eee_workers": config.parallelism.eee_workers,
        "required_gpus": config.parallelism.required_gpus,
        "sns_manifest_path": str(sns_manifest_path),
        "sns_records_path": str(sns_records_path),
        "embedding_records_path": str(embedding_records_path),
        "embedding_metadata_path": str(embedding_metadata_path),
        "embedding_errors_path": str(embedding_errors_path),
        "embedding_error_count": len(embedding_errors),
    }


def _ordered_tasks(tasks: list[Any]) -> list[Any]:
    indexed: list[tuple[int, Any]] = []
    for task in tasks:
        metadata = _task_metadata(task)
        shard_index = metadata.get("shard_index")
        if not isinstance(shard_index, int):
            raise ValueError("Parallel output tasks require integer shard_index metadata")
        indexed.append((shard_index, task))
    indexed.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indices))):
        raise ValueError(f"Parallel output shards must be unique and contiguous, got {indices}")
    expected_count = {_task_metadata(task).get("shard_count") for _, task in indexed}
    if expected_count != {len(indexed)}:
        raise ValueError(f"Parallel output shard_count metadata does not match returned tasks: {expected_count}")
    return [task for _, task in indexed]


def _merge_expert_arrays(config: ExperimentConfig, tasks: list[Any], expert: str, record_count: int) -> None:
    safe_name = expert.replace("-", "_")
    chunks: list[np.ndarray] = []
    for task in tasks:
        metadata = _task_metadata(task)
        shard_dir = _required_path(metadata, "embedding_shard_dir")
        chunk_path = shard_dir / f"{safe_name}_interleaved.npy"
        chunk = np.load(chunk_path, allow_pickle=False)
        expected_rows = 2 * len(records_from_task(task))
        if chunk.shape != (expected_rows, config.eee.embedding_dim):
            raise ValueError(
                f"Unexpected {expert} embedding shape for shard {metadata['shard_index']}: "
                f"{chunk.shape}, expected {(expected_rows, config.eee.embedding_dim)}"
            )
        chunks.append(chunk)
    merged = np.concatenate(chunks, axis=0)
    if merged.shape != (2 * record_count, config.eee.embedding_dim):
        raise ValueError(f"Unexpected merged {expert} embedding shape: {merged.shape}")
    output_dir = ensure_dir(config.run_dir / "embeddings")
    _write_array(output_dir / f"{safe_name}_interleaved.npy", merged)
    _write_array(output_dir / f"{safe_name}_raw.npy", merged[0::2])
    _write_array(output_dir / f"{safe_name}_annotation.npy", merged[1::2])


def _write_array(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _task_metadata(task: Any) -> dict[str, Any]:
    return dict(getattr(task, "_metadata", {}) or {})


def _required_path(metadata: dict[str, Any], key: str) -> Path:
    value = metadata.get(key)
    if not value:
        raise ValueError(f"Parallel output task is missing {key} metadata")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path
