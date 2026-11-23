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

"""Steps 1 & 2 (parallel): Stream sharded SNS and EEE across GPUs, then merge shard outputs."""

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
from utils import parse_config, print_result

if TYPE_CHECKING:
    from omnifuse_tutorial.config import ExperimentConfig


def main() -> None:
    config = parse_config(__doc__)
    parallel = config.parallelism
    if not parallel.enabled:
        raise ValueError("Set parallelism.enabled: true to run the parallel SNS/EEE tutorial step")

    # Fan out input records into shards and overlap SNS and EEE on separate GPU worker pools.
    pipeline = Pipeline(
        name=f"{config.experiment_id}-1-2-parallel-sns-eee",
        description="Omni-Fuse steps 1 & 2: sharded pipeline-parallel SNS and EEE",
        stages=[
            PairManifestReaderStage(config, sharded=True),
            SNSStage(
                config,
                resources=Resources(cpus=1.0, gpus=parallel.sns_gpus_per_worker),
                sharded=True,
                workers=parallel.sns_workers,
            ),
            EEEEmbeddingStage(
                config,
                resources=Resources(cpus=1.0, gpus=parallel.eee_gpus_per_worker),
                sharded=True,
                workers=parallel.eee_workers,
            ),
        ],
    )

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    tasks = pipeline.run(executor=RayDataExecutor()) or []

    # Merge completed shard artifacts into the same sns/ and embeddings/ layout used by Step 3.
    summary = {
        **_merge(config, tasks),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = write_json(config.run_dir / "parallelism" / "summary.json", summary)
    print_result({**summary, "summary_path": str(summary_path)})


def _merge(config: ExperimentConfig, tasks: list[Any]) -> dict[str, Any]:
    """Merge completed shards into the canonical `sns/` and `embeddings/` output files."""
    expected = tasks[0]._metadata.get("shard_count", len(tasks)) if tasks else 0
    indices = [task._metadata.get("shard_index") for task in tasks]
    if not tasks or len(indices) != expected or set(indices) != set(range(expected)):
        raise ValueError(f"Parallel output shards must be unique and complete, got {indices}")

    # Concatenate records, SNS manifests, and EEE errors in the same shard completion order.
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        records += task_records(task)
        manifest += read_jsonl(Path(task._metadata["sns_manifest_path"]))
        errors += json.loads(Path(task._metadata["embedding_errors_path"]).read_text(encoding="utf-8"))

    embeddings_dir = config.run_dir / "embeddings"
    paths = {
        "sns_manifest_path": write_jsonl(config.run_dir / "sns" / "manifest.jsonl", manifest),
        "sns_records_path": write_jsonl(config.run_dir / "sns" / "records.jsonl", records),
        "embedding_records_path": write_jsonl(embeddings_dir / "records.jsonl", records),
        "embedding_errors_path": write_json(embeddings_dir / "errors.json", errors),
    }

    # Concatenate each expert's interleaved (raw, annotation) embedding rows across shards.
    for expert in config.eee.experts:
        shard_arrays = [
            np.load(expert_file(Path(task._metadata["embeddings_shard_dir"]), expert, "interleaved")) for task in tasks
        ]
        merged = np.concatenate(shard_arrays)
        write_npy(expert_file(embeddings_dir, expert, "interleaved"), merged)
        write_npy(expert_file(embeddings_dir, expert, "raw"), merged[0::2])
        write_npy(expert_file(embeddings_dir, expert, "annotation"), merged[1::2])

    metadata = {
        "pair_ids": [record["pair_id"] for record in records],
        "modalities": [record["modality"] for record in records],
        "experts": list(config.eee.experts),
        "embedding_dim": config.eee.embedding_dim,
        "error_count": len(errors),
        "errors_path": str(paths["embedding_errors_path"]),
        "parallel_shards": len(tasks),
    }
    paths["embedding_metadata_path"] = write_json(embeddings_dir / "metadata.json", metadata)

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


if __name__ == "__main__":
    main()
