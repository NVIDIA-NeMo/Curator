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
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources

from omnifuse_tutorial.data import (
    expert_file,
    read_jsonl,
    task_records,
    write_json,
    write_jsonl,
    write_npy,
)
from omnifuse_tutorial.stages import EEEEmbeddingStage, PairManifestReaderStage, SNSStage

if TYPE_CHECKING:
    from omnifuse_tutorial.config import ExperimentConfig


def run_parallel_sns_eee(config: ExperimentConfig) -> dict[str, Any]:
    """Steps 1 and 2 at once, on GPUs reserved per stage; writes the same canonical files as 1_sns.py and 2_embed.py."""
    parallel = config.parallelism
    if not parallel.enabled:
        raise ValueError("Set parallelism.enabled: true to run the parallel SNS/EEE tutorial step")
    stages = [
        PairManifestReaderStage(config, sharded=True),
        SNSStage(
            config, Resources(cpus=1.0, gpus=parallel.sns_gpus_per_worker), sharded=True, workers=parallel.sns_workers
        ),
        EEEEmbeddingStage(
            config, Resources(cpus=1.0, gpus=parallel.eee_gpus_per_worker), sharded=True, workers=parallel.eee_workers
        ),
    ]
    started_at, started = datetime.now(UTC), time.perf_counter()
    pipeline = Pipeline(
        name=f"{config.experiment_id}-1-2-parallel-sns-eee",
        description="Sharded pipeline-parallel Omni-Fuse SNS and EEE tutorial steps",
        stages=stages,
    )
    tasks = pipeline.run(executor=RayDataExecutor()) or []
    summary = {
        **_merge(config, tasks),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {**summary, "summary_path": str(write_json(config.run_dir / "parallelism" / "summary.json", summary))}


def _merge(config: ExperimentConfig, tasks: list[Any]) -> dict[str, Any]:
    """Merge completed shards into the canonical SNS and EEE output files."""
    expected = tasks[0]._metadata.get("shard_count", len(tasks)) if tasks else 0
    indices = [task._metadata.get("shard_index") for task in tasks]
    if not tasks or len(indices) != expected or set(indices) != set(range(expected)):
        raise ValueError(f"Parallel output shards must be unique and complete, got {indices}")
    records, manifest, errors = [], [], []
    for task in tasks:
        records += task_records(task)
        manifest += read_jsonl(Path(task._metadata["sns_manifest_path"]))
        errors += json.loads(Path(task._metadata["embedding_errors_path"]).read_text(encoding="utf-8"))
    embeddings = config.run_dir / "embeddings"
    paths = {
        "sns_manifest_path": write_jsonl(config.run_dir / "sns" / "manifest.jsonl", manifest),
        "sns_records_path": write_jsonl(config.run_dir / "sns" / "records.jsonl", records),
        "embedding_records_path": write_jsonl(embeddings / "records.jsonl", records),
        "embedding_errors_path": write_json(embeddings / "errors.json", errors),
    }
    for expert in config.eee.experts:
        merged = np.concatenate(
            [
                np.load(expert_file(Path(task._metadata["embeddings_shard_dir"]), expert, "interleaved"))
                for task in tasks
            ]
        )
        for kind, part in (("interleaved", merged), ("raw", merged[0::2]), ("annotation", merged[1::2])):
            write_npy(expert_file(embeddings, expert, kind), part)
    metadata = {
        "pair_ids": [record["pair_id"] for record in records],
        "modalities": [record["modality"] for record in records],
        "experts": list(config.eee.experts),
        "embedding_dim": config.eee.embedding_dim,
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
        "embedding_error_count": len(errors),
        **{key: str(path) for key, path in paths.items()},
    }
