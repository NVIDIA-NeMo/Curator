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

# ruff: noqa: E402

import json
import pickle
from pathlib import Path

from nemo_curator.backends.experimental.ray_actor_pool.executor import RayActorPoolExecutor
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under

_executor_map = {"ray_data": RayDataExecutor, "xenna": XennaExecutor, "ray_actors": RayActorPoolExecutor}


def setup_executor(executor_name: str) -> RayDataExecutor | XennaExecutor | RayActorPoolExecutor:
    """Setup the executor for the given name."""
    try:
        executor = _executor_map[executor_name]()
    except KeyError:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg) from None
    return executor


def load_dataset_files(
    dataset_path: Path,
    dataset_size_gb: float | None = None,
    dataset_ratio: float | None = None,
    keep_extensions: str = "parquet",
) -> list[str]:
    """Load the dataset files at the given path and return a subset of the files whose combined size is approximately the given size in GB."""
    input_files = get_all_file_paths_and_size_under(
        dataset_path, recurse_subdirectories=True, keep_extensions=keep_extensions
    )
    if (not dataset_size_gb and not dataset_ratio) or (dataset_size_gb and dataset_ratio):
        msg = "Either dataset_size_gb or dataset_ratio must be provided, but not both"
        raise ValueError(msg)
    if dataset_size_gb:
        desired_size_bytes = (1024**3) * dataset_size_gb
    else:
        total_file_size_bytes = sum(size for _, size in input_files)
        desired_size_bytes = total_file_size_bytes * dataset_ratio

    total_size = 0
    subset_files = []
    for file, size in input_files:
        if size + total_size > desired_size_bytes:
            break
        else:
            subset_files.append(file)
            total_size += size

    return subset_files


def write_benchmark_results(results: dict, output_path: str | Path) -> None:
    """Write benchmark results (params, metrics, tasks) to the appropriate files in the output directory.

    - Writes 'params.json' and 'metrics.json' (merging with existing file contents if present and updating values).
    - Writes 'tasks.pkl' as a pickle file if present in results.
    - The output directory is created if it does not exist.

    Typically used by benchmark scripts to persist results in the format expected by the benchmarking framework.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if "params" in results:
        params_path = output_path / "params.json"
        params_data = {}
        if params_path.exists():
            params_data = json.loads(params_path.read_text())
        params_data.update(results["params"])
        params_path.write_text(json.dumps(params_data, default=convert_paths_to_strings, indent=2))
    if "metrics" in results:
        metrics_path = output_path / "metrics.json"
        metrics_data = {}
        if metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text())
        metrics_data.update(results["metrics"])
        metrics_path.write_text(json.dumps(metrics_data, default=convert_paths_to_strings, indent=2))
    if "tasks" in results:
        (output_path / "tasks.pkl").write_bytes(pickle.dumps(results["tasks"]))


def convert_paths_to_strings(obj: object) -> object:
    """
    Convert Path objects to strings, support conversions in container types in a recursive manner.
    """
    if isinstance(obj, dict):
        retval = {convert_paths_to_strings(k): convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        retval = [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, Path):
        retval = str(obj)
    else:
        retval = obj
    return retval


########################################################
# Utils for supporting legacy Curator
from collections.abc import Mapping
from typing import Any

from nemo_curator.tasks.tasks import Task, _EmptyTask
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.performance_utils import StagePerfStats

# for pre-26.02 Curator
try:
    from nemo_curator.tasks.utils import WorkflowRunResult
except ImportError:
    WorkflowRunResult = type(None)


def normalize_pipeline_tasks(  # noqa: PLR0912, C901
    tasks: list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None,
) -> dict[str, list[Task]]:
    """Return a mapping of pipeline name -> list of tasks from various input shapes."""
    if tasks is not None and isinstance(tasks, WorkflowRunResult):
        source: Mapping[str, Any] = tasks.pipeline_tasks
    elif isinstance(tasks, Mapping):
        if "pipeline_tasks" in tasks and isinstance(tasks["pipeline_tasks"], Mapping):
            source = tasks["pipeline_tasks"]
        else:
            source = tasks
    elif isinstance(tasks, list):
        return {"": list(tasks)}
    elif tasks is None:
        return {"": []}
    else:
        msg = (
            "tasks must be a list of Task objects, a mapping of pipeline_name -> tasks, "
            "a workflow result dict, or WorkflowRunResult instance."
        )
        raise TypeError(msg)

    normalized: dict[str, list[Task]] = {}
    for pipeline_name, pipeline_tasks in source.items():
        if pipeline_tasks is None:
            normalized[str(pipeline_name)] = []
        elif isinstance(pipeline_tasks, list):
            normalized[str(pipeline_name)] = pipeline_tasks
        elif isinstance(pipeline_tasks, Task):
            normalized[str(pipeline_name)] = [pipeline_tasks]
        elif hasattr(pipeline_tasks, "__iter__") and not isinstance(pipeline_tasks, (str, bytes, dict)):
            normalized[str(pipeline_name)] = list(pipeline_tasks)
        else:
            # If here, the tasks obj passed in is just a map of metadata:values.
            # Convert it to a map of pipeline names -> list of tasks with one task
            t = _EmptyTask(task_id="", dataset_name="", data=None)
            custom_metrics = {metadata: value for metadata, value in source.items() if isinstance(value, (int, float))}
            t.add_stage_perf(StagePerfStats(stage_name="", custom_metrics=custom_metrics))
            normalized = {"": [t]}
            break

    return normalized or {"": []}


def aggregate_task_metrics(
    tasks: list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Aggregate task metrics by computing mean/std/sum."""
    import numpy as np

    metrics: dict[str, float] = {}
    pipeline_task_map = normalize_pipeline_tasks(tasks)
    multiple_pipelines = len(pipeline_task_map) > 1

    for pipeline_name, pipeline_tasks in pipeline_task_map.items():
        stage_metrics = TaskPerfUtils.collect_stage_metrics(pipeline_tasks)
        if prefix:
            stage_prefix = f"{prefix}_{pipeline_name}" if pipeline_name else prefix
        elif pipeline_name and multiple_pipelines:
            stage_prefix = pipeline_name
        else:
            stage_prefix = None

        for stage_name, stage_data in stage_metrics.items():
            resolved_stage_name = stage_name if stage_prefix is None else f"{stage_prefix}_{stage_name}"
            for metric_name, values in stage_data.items():
                for agg_name, agg_func in [("sum", np.sum), ("mean", np.mean), ("std", np.std)]:
                    metric_key = f"{resolved_stage_name}_{metric_name}_{agg_name}"
                    if len(values) > 0:
                        metrics[metric_key] = float(agg_func(values))
                    else:
                        metrics[metric_key] = 0.0
    return metrics
