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

"""Sharded multi-GPU SNS -> EEE: record shards stream through resident SNS and EEE workers, then outputs merge."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources

from omnifuse_tutorial.data import (
    expert_file,
    load_records,
    make_batch,
    read_jsonl,
    task_records,
    write_json,
    write_jsonl,
    write_npy,
)
from omnifuse_tutorial.stages import EEEEmbeddingStage, SNSStage

if TYPE_CHECKING:
    from omnifuse_tutorial.config import ExperimentConfig


def run_parallel_sns_eee(config: ExperimentConfig) -> dict[str, Any]:
    """Steps 1 and 2 at once, on GPUs reserved per stage; writes the same canonical files as 1_sns.py and 2_embed.py."""
    import torch

    parallel = config.parallelism
    if not parallel.enabled:
        raise ValueError("Set parallelism.enabled: true to run the parallel SNS/EEE tutorial step")
    if torch.cuda.device_count() + 1e-9 < parallel.required_gpus:
        raise RuntimeError(
            f"Parallel SNS/EEE requests {parallel.required_gpus:g} GPUs but only {torch.cuda.device_count()} are visible. "
            "Reduce parallelism worker/GPU counts or expose more GPUs."
        )
    records, size = load_records(config.data_pools), parallel.records_per_shard
    if not records:
        raise ValueError("Parallel SNS/EEE requires at least one input record")
    shards = [
        make_batch(
            config.experiment_id,
            records[start : start + size],
            {
                "experiment_id": config.experiment_id,
                "parallel_pipeline": True,
                "shard_index": index,
                "shard_count": math.ceil(len(records) / size),
                "record_start": start,
                "record_stop": min(start + size, len(records)),
            },
        )
        for index, start in enumerate(range(0, len(records), size))
    ]
    stages = [
        SNSStage(
            config, Resources(cpus=1.0, gpus=parallel.sns_gpus_per_worker), sharded=True, workers=parallel.sns_workers
        ),
        EEEEmbeddingStage(
            config, Resources(cpus=1.0, gpus=parallel.eee_gpus_per_worker), sharded=True, workers=parallel.eee_workers
        ),
    ]
    executor = XennaExecutor(
        config={
            "execution_mode": "streaming",
            "autoscale_interval_s": parallel.autoscale_interval_s,
            "logging_interval": parallel.logging_interval_s,
        }
    )
    started_at, started = datetime.now(timezone.utc), time.perf_counter()
    pipeline = Pipeline(
        name=f"{config.experiment_id}-1-2-parallel-sns-eee",
        description="Sharded pipeline-parallel Omni-Fuse SNS and EEE tutorial steps",
        stages=stages,
    )
    tasks = pipeline.run(executor=executor, initial_tasks=shards)
    if not tasks:
        raise RuntimeError("Parallel SNS/EEE pipeline produced no output tasks")
    summary = {
        **_merge(config, tasks),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {**summary, "summary_path": str(write_json(config.run_dir / "parallelism" / "summary.json", summary))}


def _merge(config: ExperimentConfig, tasks: list[Any]) -> dict[str, Any]:
    """Validate every shard, then write merged outputs in original record order."""
    for task in tasks:
        if not isinstance(task._metadata.get("shard_index"), int):
            raise ValueError("Parallel output tasks require integer shard_index metadata")
    tasks = sorted(tasks, key=lambda task: task._metadata["shard_index"])
    if (indices := [task._metadata["shard_index"] for task in tasks]) != list(range(len(tasks))):
        raise ValueError(f"Parallel output shards must be unique and contiguous, got {indices}")
    if (counts := {task._metadata.get("shard_count") for task in tasks}) != {len(tasks)}:
        raise ValueError(f"Parallel output shard_count metadata does not match returned tasks: {counts}")
    records, manifest, errors, dim = [], [], [], config.eee.embedding_dim
    for task in tasks:
        shard_records, shard_manifest = task_records(task), read_jsonl(_required_path(task, "sns_manifest_path"))
        if len(shard_manifest) != len(shard_records):
            raise ValueError(
                f"Shard {task._metadata['shard_index']} has {len(shard_records)} records but {len(shard_manifest)} SNS manifest rows"
            )
        errors_path = _required_path(task, "embedding_errors_path")
        shard_errors = json.loads(errors_path.read_text(encoding="utf-8"))
        if not isinstance(shard_errors, list):
            raise TypeError(f"Embedding errors must be a list: {errors_path}")
        records += shard_records
        manifest += shard_manifest
        errors += shard_errors
    arrays = {}
    for expert in config.eee.experts:
        chunks = []
        for task in tasks:
            chunk = np.load(
                expert_file(_required_path(task, "embeddings_shard_dir"), expert, "interleaved"), allow_pickle=False
            )
            expected = (2 * len(task.data), dim)
            if chunk.shape != expected:
                shard = task._metadata["shard_index"]
                raise ValueError(
                    f"Unexpected {expert} embedding shape for shard {shard}: {chunk.shape}, expected {expected}"
                )
            chunks.append(chunk)
        arrays[expert] = np.concatenate(chunks)
    embeddings = config.run_dir / "embeddings"
    paths = {
        "sns_manifest_path": write_jsonl(config.run_dir / "sns" / "manifest.jsonl", manifest),
        "sns_records_path": write_jsonl(config.run_dir / "sns" / "records.jsonl", records),
        "embedding_records_path": write_jsonl(embeddings / "records.jsonl", records),
        "embedding_errors_path": write_json(embeddings / "errors.json", errors),
    }
    for expert, merged in arrays.items():
        for kind, part in (("interleaved", merged), ("raw", merged[0::2]), ("annotation", merged[1::2])):
            write_npy(expert_file(embeddings, expert, kind), part)
    metadata = {
        "pair_ids": [record["pair_id"] for record in records],
        "modalities": [record["modality"] for record in records],
        "experts": list(config.eee.experts),
        "embedding_dim": dim,
        "error_count": len(errors),
        "errors_path": str(paths["embedding_errors_path"]),
        "parallel_shards": len(tasks),
    }
    paths["embedding_metadata_path"] = write_json(embeddings / "metadata.json", metadata)
    parallel = config.parallelism
    return {
        "mode": "pipeline_parallel",
        "record_count": len(records),
        "shard_count": len(tasks),
        "records_per_shard": parallel.records_per_shard,
        "sns_workers": parallel.sns_workers,
        "eee_workers": parallel.eee_workers,
        "required_gpus": parallel.required_gpus,
        "embedding_error_count": len(errors),
        **{key: str(path) for key, path in paths.items()},
    }


def _required_path(task: Any, key: str) -> Path:
    if not task._metadata.get(key):
        raise ValueError(f"Parallel output task is missing {key} metadata")
    path = Path(task._metadata[key])
    if not path.exists():
        raise FileNotFoundError(path)
    return path
