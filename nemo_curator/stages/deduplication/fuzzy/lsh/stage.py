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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.fuzzy.lsh.lsh import LSHActor
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_DEFAULT_MINHASH_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs


@dataclass
class LSHStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that performs LSH on a FileGroupTask containing minhash data.

    The executor will process this stage in iterations based on bands_per_iteration.

    Parameters
    ----------
    num_bands
        Number of LSH bands.
    minhashes_per_band
        Number of minhashes per band.
    id_field
        Name of the ID field in input data.
    minhash_field
        Name of the minhash field in input data.
    output_path
        Base path to write output files.
    read_kwargs
        Keyword arguments for the read method.
    write_kwargs
        Keyword arguments for the write method.
    rmm_pool_size
        Size of the RMM GPU memory pool in bytes.
        If "auto", the memory pool is set to 90% of the free GPU memory.
        If None, the memory pool is set to 50% of the free GPU memory that can expand if needed.
    spill_memory_limit
        Device memory limit in bytes for spilling to host.
        If "auto", the limit is set to 80% of the RMM pool size.
        If None spilling is disabled.
    enable_statistics
        Whether to collect statistics.
    bands_per_iteration
        Number of bands to process per shuffle iteration. Between 1 and num_bands.
        Higher values reduce the number of shuffle iterations but increase the memory usage.
    total_nparts
        Total number of partitions to write during the shuffle.
        If None, the number of partitions will be decided automatically by the executor as the closest power of 2 <= number of input tasks.
    """

    _name = "LSHStage"
    _resources = Resources(gpus=1.0)

    # Core Algo objects
    actor_class = LSHActor

    # LSH parameters
    num_bands: int
    minhashes_per_band: int
    # Data parameters
    id_field: str = CURATOR_DEDUP_ID_STR
    minhash_field: str = CURATOR_DEFAULT_MINHASH_FIELD
    output_path: str = "./"
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Shuffle parameters
    rmm_pool_size: int | Literal["auto"] | None = "auto"
    spill_memory_limit: int | Literal["auto"] | None = "auto"
    enable_statistics: bool = False
    bands_per_iteration: int = 5  # number of bands to process in each iteration
    total_nparts: int | None = None

    def __post_init__(self):
        super().__init__()

        self.read_kwargs = self.read_kwargs if self.read_kwargs is not None else {}
        self.write_kwargs = self.write_kwargs if self.write_kwargs is not None else {}
        self.output_paths = []

        self.actor_kwargs = {
            "num_bands": self.num_bands,
            "minhashes_per_band": self.minhashes_per_band,
            "id_field": self.id_field,
            "minhash_field": self.minhash_field,
            "rmm_pool_size": self.rmm_pool_size,
            "spill_memory_limit": self.spill_memory_limit,
            "enable_statistics": self.enable_statistics,
            "read_kwargs": self.read_kwargs,
            "write_kwargs": self.write_kwargs,
            "total_nparts": self.total_nparts,  # Can be None, executor will set it
        }

        if self.bands_per_iteration < 1 or self.bands_per_iteration > self.num_bands:
            err_msg = (
                f"Invalid bands_per_iteration: {self.bands_per_iteration}, must be in range [1, {self.num_bands}]"
            )
            raise ValueError(err_msg)

        # Handle output directory and subdirectories
        output_fs = get_fs(self.output_path, storage_options=self.write_kwargs.get("storage_options"))
        output_base_path = output_fs.sep.join([self.output_path, self.name])

        create_or_overwrite_dir(output_base_path, fs=output_fs)

        for band_range in self.get_band_iterations():
            output_path = output_fs.sep.join([output_base_path, f"band_{band_range[0]}-band_{band_range[1]}"])
            create_or_overwrite_dir(output_path, fs=output_fs)
            self.output_paths.append(output_path)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        err_msg = "LSHProcessingStage does not support the process method."
        raise NotImplementedError(err_msg)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_LSH_STAGE: True,
        }

    def _check_actor_obj(self) -> None:
        if not hasattr(self, "_actor_obj") or not isinstance(self._actor_obj, self.actor_class):
            error = "Actor object not initialized. This might be because an incorrect executor was used or it failed to setup the stage properly."
            raise RuntimeError(error)

    def read_and_insert(self, task: FileGroupTask, band_range: tuple[int, int]) -> FileGroupTask:
        self._check_actor_obj()
        result = self._actor_obj.read_and_insert(task.data, band_range)
        self._current_band_range = band_range
        self.output_columns = result
        self.dataset_name = task.dataset_name
        return task

    def insert_finished(self) -> None:
        self._check_actor_obj()
        self._actor_obj.insert_finished()

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        current_band_min, current_band_max = self._current_band_range
        partition_dicts = self._actor_obj.extract_and_write()
        return [
            FileGroupTask(
                task_id=f"b{current_band_min}_b{current_band_max}_{partition_info['partition_id']}",
                dataset_name=self.dataset_name + f"{self.name}",
                data=[partition_info["path"]],
                _metadata={
                    "partition_index": partition_info["partition_id"],
                    "total_partitions": len(partition_dicts),
                    "num_docs": partition_info["num_docs"],
                    "output_columns": self.output_columns,
                },
            )
            for partition_info in partition_dicts
        ]

    def teardown(self) -> None:
        self._check_actor_obj()
        self._actor_obj.cleanup()

    def get_band_iterations(self) -> Iterator[tuple[int, int]]:
        """Get all band ranges for iteration."""
        for band_start in range(0, self.num_bands, self.bands_per_iteration):
            band_range = (band_start, min(band_start + self.bands_per_iteration, self.num_bands))
            yield band_range
