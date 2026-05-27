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

"""Simple pipeline showing RayClient vs SlurmRayClient.

The pipeline is intentionally CPU-only and dependency-free so the focus
stays on the client switch rather than model setup.

Stages:
    1. TaskCreationStage  (_EmptyTask -> list[SampleTask])
       Generates ``num_tasks`` tasks, each holding a small DataFrame of sentences.

    2. WordCountStage     (SampleTask -> SampleTask)
       Adds a ``word_count`` column to each task.

    3. NodeTagStage       (SampleTask -> SampleTask)
       Records which Ray-worker hostname processed the task.
       On a multi-node SLURM job this column will show different hostnames,
       proving that work is genuinely distributed.

Usage::

    # Local (single-node):
    python pipeline.py

    # SLURM (multi-node) — called via srun inside submit.sh:
    python pipeline.py --slurm

    # Limit tasks for a quick smoke test:
    python pipeline.py --num-tasks 4
"""

from __future__ import annotations

import argparse
import os
import random
import socket
from dataclasses import field

import pandas as pd
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task, _EmptyTask

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class SampleTask(Task[pd.DataFrame]):
    """A task holding a small DataFrame of sentences."""

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class TaskCreationStage(ProcessingStage[_EmptyTask, SampleTask]):
    """Generate ``num_tasks`` tasks, each with ``sentences_per_task`` rows."""

    name: str = "TaskCreationStage"

    def __init__(self, num_tasks: int = 20, sentences_per_task: int = 5) -> None:
        self.num_tasks = num_tasks
        self.sentences_per_task = sentences_per_task

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence"]

    def process(self, _: _EmptyTask) -> list[SampleTask]:
        tasks = []
        for i in range(self.num_tasks):
            sentences = random.choices(SAMPLE_SENTENCES, k=self.sentences_per_task)  # noqa: S311
            tasks.append(
                SampleTask(
                    data=pd.DataFrame({"sentence": sentences}),
                    task_id=f"task_{i:04d}",
                    dataset_name="slurm_demo",
                )
            )
        return tasks


class WordCountStage(ProcessingStage[SampleTask, SampleTask]):
    """Add a ``word_count`` column — pure CPU, no dependencies."""

    name: str = "WordCountStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count"]

    def process(self, task: SampleTask) -> SampleTask:
        task.data["word_count"] = task.data["sentence"].str.split().str.len()
        return task


class NodeTagStage(ProcessingStage[SampleTask, SampleTask]):
    """Tag each task with the hostname and GPU info of the worker that processed it.

    On a multi-node SLURM run the ``processed_by`` column will show
    different hostnames, confirming tasks are spread across nodes.
    ``gpu_info`` reports the GPUs visible to the Ray worker process.
    """

    name: str = "NodeTagStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count", "processed_by", "gpu_info"]

    def process(self, task: SampleTask) -> SampleTask:
        task.data["processed_by"] = socket.gethostname()
        task.data["gpu_info"] = _gpu_summary()
        return task


def _gpu_summary() -> str:
    """Return a short string describing GPUs visible to the current process."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return "no GPUs (nvidia-smi not found)"
    except Exception:  # noqa: BLE001
        return "gpu_info unavailable"
    if result.returncode == 0 and result.stdout.strip():
        gpus = [line.strip() for line in result.stdout.strip().splitlines()]
        return f"{len(gpus)} GPU(s): " + "; ".join(gpus)
    return "no GPUs (nvidia-smi failed)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_pipeline(num_tasks: int, sentences_per_task: int) -> Pipeline:
    pipeline = Pipeline(
        name="slurm_demo",
        description="Word-count + node-tag pipeline — no GPU required",
    )
    pipeline.add_stage(TaskCreationStage(num_tasks=num_tasks, sentences_per_task=sentences_per_task))
    pipeline.add_stage(WordCountStage())
    pipeline.add_stage(NodeTagStage())
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SLURM demo pipeline")
    parser.add_argument("--slurm", action="store_true", help="Use SlurmRayClient (set when running via srun)")
    parser.add_argument("--num-tasks", type=int, default=20, help="Number of tasks to generate")
    parser.add_argument("--sentences-per-task", type=int, default=5)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # The only change needed to go from local to SLURM is this one line.
    # -----------------------------------------------------------------------
    ray_client = SlurmRayClient() if args.slurm else RayClient()
    ray_client.start()
    # On worker nodes (SLURM_NODEID > 0), start() never returns —
    # they block running the Ray daemon. Only the head continues below.

    try:
        pipeline = build_pipeline(args.num_tasks, args.sentences_per_task)
        logger.info(f"\n{pipeline.describe()}")

        executor = XennaExecutor(config={"execution_mode": "streaming"})
        results = pipeline.run(executor=executor)
    finally:
        ray_client.stop()

    if not results:
        logger.warning("No results returned")
        return

    logger.info(f"Completed {len(results)} tasks")

    # Show which nodes + GPUs processed tasks
    node_gpu: dict[str, str] = {}
    for task in results:
        for _, row in task.data[["processed_by", "gpu_info"]].drop_duplicates().iterrows():
            node_gpu[row["processed_by"]] = row["gpu_info"]

    logger.info(f"Tasks processed by {len(node_gpu)} distinct node(s):")
    for node, gpu in sorted(node_gpu.items()):
        logger.info(f"  {node}: {gpu}")

    # Print a sample result
    sample = results[0].data
    logger.info(f"\nSample output (task '{results[0].task_id}'):\n{sample.to_string(index=False)}")

    slurm_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    if slurm_nodes > 1 and len(node_gpu) < 2:  # noqa: PLR2004
        logger.warning(
            f"Job allocated {slurm_nodes} nodes but only {len(node_gpu)} node(s) processed tasks. "
            "Check that --num-tasks is large enough to distribute across all workers."
        )


if __name__ == "__main__":
    main()
