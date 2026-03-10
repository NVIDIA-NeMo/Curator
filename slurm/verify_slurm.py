# This is just for testing. Will be removed once the PR is merged.

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Minimal pipeline to verify that SlurmRayClient + XennaExecutor work correctly.

Stages (all CPU-only, no model downloads):
    1. TaskCreationStage  — generates dummy text tasks
    2. WordCountStage     — adds a ``word_count`` column
    3. NodeReporterStage  — records which Ray node processed each task

Run inside a SLURM allocation::

    sbatch slurm/submit_verify.sh
"""

import json
import os
import random
import socket
import sys
import time
from dataclasses import field

import pandas as pd
import ray

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task, _EmptyTask

SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "NeMo Curator scales data curation across many nodes",
    "SLURM manages workload scheduling on HPC clusters",
    "Ray distributes Python workloads transparently",
    "GPU acceleration dramatically speeds up deep learning",
    "Data quality is critical for training large language models",
    "Distributed systems require careful coordination",
    "Multimodal AI combines text, image, and audio understanding",
]


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #


class SampleTask(Task[pd.DataFrame]):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


class TaskCreationStage(ProcessingStage[_EmptyTask, SampleTask]):
    name: str = "TaskCreationStage"

    def __init__(self, num_sentences_per_task: int = 4, num_tasks: int = 50):
        self.num_sentences_per_task = num_sentences_per_task
        self.num_tasks = num_tasks

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence"]

    def process(self, _: _EmptyTask) -> list[SampleTask]:
        tasks = []
        for i in range(self.num_tasks):
            sentences = random.choices(SAMPLE_SENTENCES, k=self.num_sentences_per_task)  # noqa: S311
            tasks.append(
                SampleTask(
                    data=pd.DataFrame({"sentence": sentences}),
                    task_id=f"task_{i}",
                    dataset_name="verify",
                )
            )
        return tasks


class WordCountStage(ProcessingStage[SampleTask, SampleTask]):
    name: str = "WordCountStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count"]

    def process(self, task: SampleTask) -> SampleTask:
        task.data["word_count"] = task.data["sentence"].str.split().str.len()
        return task


class NodeReporterStage(ProcessingStage[SampleTask, SampleTask]):
    """Stamps each task with the hostname of the Ray worker that processed it."""

    name: str = "NodeReporterStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count", "processed_by"]

    def process(self, task: SampleTask) -> SampleTask:
        time.sleep(0.5)
        hostname = socket.gethostname()
        task.data["processed_by"] = hostname
        return task


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def collect_cluster_info() -> dict:
    """Connect to Ray briefly and snapshot cluster resources."""
    ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
    try:
        nodes = ray.nodes()
        alive = [n for n in nodes if n.get("Alive")]
        resources = ray.cluster_resources()
        return {
            "ray_address": os.environ.get("RAY_ADDRESS"),
            "num_alive_nodes": len(alive),
            "total_cpus": resources.get("CPU", 0),
            "total_gpus": resources.get("GPU", 0),
            "node_details": [
                {
                    "node_id": n["NodeID"][:12],
                    "hostname": n.get("NodeManagerHostname", "?"),
                    "alive": n.get("Alive", False),
                    "cpus": n.get("Resources", {}).get("CPU", 0),
                    "gpus": n.get("Resources", {}).get("GPU", 0),
                }
                for n in alive
            ],
        }
    finally:
        ray.shutdown()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    num_tasks = 200
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    output_dir = os.environ.get(
        "VERIFY_OUTPUT_DIR",
        os.path.join(os.path.dirname(__file__), "outputs"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Determine a label for this run from SLURM env
    num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", "?")
    gpus_per_node = os.environ.get("SLURM_GPUS_ON_NODE", "0")
    run_label = f"{num_nodes}n_{gpus_per_node}g"
    result_file = os.path.join(output_dir, f"result_{run_label}_{slurm_job_id}.json")

    report: dict = {
        "status": "FAILED",
        "config": {
            "slurm_job_id": slurm_job_id,
            "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", "?"),
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "cpus_per_node": os.environ.get("SLURM_CPUS_ON_NODE", "?"),
        },
    }

    try:
        print("=" * 60)
        print(f"  SLURM Verify: {num_nodes} node(s), {gpus_per_node} GPU(s)/node")
        print(f"  Job: {slurm_job_id}")
        print("=" * 60)

        with SlurmRayClient() as _client:
            # Snapshot cluster resources
            cluster_info = collect_cluster_info()
            report["cluster"] = cluster_info
            print("\nCluster info:")
            print(f"  Alive nodes : {cluster_info['num_alive_nodes']}")
            print(f"  Total CPUs  : {cluster_info['total_cpus']}")
            print(f"  Total GPUs  : {cluster_info['total_gpus']}")
            for nd in cluster_info["node_details"]:
                print(f"    {nd['hostname']}  CPUs={nd['cpus']}  GPUs={nd['gpus']}")

            # Build and run pipeline
            pipeline = Pipeline(
                name="verify_slurm",
                description="Verify SlurmRayClient + XennaExecutor",
                stages=[
                    TaskCreationStage(num_sentences_per_task=4, num_tasks=num_tasks),
                    WordCountStage(),
                    NodeReporterStage(),
                ],
            )

            executor = XennaExecutor()
            results = pipeline.run(executor)

            # Summarise results
            node_counts: dict[str, int] = {}
            total_rows = 0
            for task in results:
                total_rows += len(task.data)
                for node in task.data["processed_by"].unique():
                    node_counts[node] = node_counts.get(node, 0) + int(
                        (task.data["processed_by"] == node).sum()
                    )

            report["results"] = {
                "num_output_tasks": len(results),
                "total_rows": total_rows,
                "rows_per_node": node_counts,
                "columns": list(results[0].data.columns) if results else [],
            }

            print(f"\nPipeline completed — {len(results)} output tasks, {total_rows} rows")
            print("Rows per node:")
            for node, count in sorted(node_counts.items()):
                print(f"  {node}: {count}")

            # Checks
            checks = {
                "task_count_ok": len(results) == num_tasks,
                "row_count_ok": total_rows == num_tasks * 4,
                "has_word_count": "word_count" in (results[0].data.columns if results else []),
                "has_processed_by": "processed_by" in (results[0].data.columns if results else []),
            }
            report["checks"] = checks
            all_ok = all(checks.values())
            report["status"] = "PASSED" if all_ok else "FAILED"

            for name, ok in checks.items():
                mark = "PASS" if ok else "FAIL"
                print(f"  [{mark}] {name}")

            print(f"\n{'=' * 60}")
            print(f"  RESULT: {report['status']}")
            print(f"{'=' * 60}")

    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        print(f"\nFATAL: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()

    # Always write the report
    with open(result_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {result_file}")


if __name__ == "__main__":
    main()
