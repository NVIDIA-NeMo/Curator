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


def load_dataset_files(dataset_path: Path, dataset_size_gb: float, keep_extensions: str = "parquet") -> list[str]:
    """Load the dataset files at the given path and return a subset of the files whose combined size is approximately the given size in GB."""
    input_files = get_all_file_paths_and_size_under(
        dataset_path, recurse_subdirectories=True, keep_extensions=keep_extensions
    )
    desired_size_bytes = (1024**3) * dataset_size_gb
    total_size = 0
    subset_files = []
    for file, size in input_files:
        if size + total_size > desired_size_bytes:
            break
        else:
            subset_files.append(file)
            total_size += size

    return subset_files


def write_benchmark_results(results: dict, output_path: Path | str) -> None:
    """Write results to the standard files expected by the benchmark framework.

    This utility is typically used by developer-written benchmark scripts to write results
    to the standard files expected by the benchmark framework.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if "params" in results:
        (output_path / "params.json").write_text(json.dumps(results["params"], indent=2))
    if "metrics" in results:
        (output_path / "metrics.json").write_text(json.dumps(results["metrics"], indent=2))
    if "tasks" in results:
        (output_path / "tasks.pkl").write_bytes(pickle.dumps(results["tasks"]))
