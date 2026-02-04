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
