from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from ray_curator.backends.experimental.utils import RayStageSpecKeys
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.fuzzy.lsh.lsh import LSHActor
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask
from ray_curator.utils.file_utils import delete_dir, get_fs, is_not_empty


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
    output_dir
        Directory to write output files.
    read_kwargs
        Keyword arguments for the read method.
    write_kwargs
        Keyword arguments for the write method.
    rmm_pool_size
        Size of the RMM GPU memory pool in bytes.
    spill_memory_limit
        Device memory limit in bytes for spilling to host.
        If "auto", the limit is set to 80% of the RMM pool size.
        If None spilling is disabled.
    enable_statistics
        Whether to collect statistics.
    bands_per_iteration
        Number of bands to process per shuffle iteration. Between 1 and num_bands.
        Higher values reduce the number of shuffle iterations but increase the memory usage.
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
    minhash_field: str = "_minhash_signature"
    output_dir: str = "./"
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Shuffle parameters
    rmm_pool_size: int = 1024 * 1024 * 1024
    spill_memory_limit: int | Literal["auto"] | None = "auto"
    enable_statistics: bool = False
    bands_per_iteration: int = 5  # number of bands to process in each iteration

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
        }

        if self.bands_per_iteration < 1 or self.bands_per_iteration > self.num_bands:
            err_msg = (
                f"Invalid bands_per_iteration: {self.bands_per_iteration}, must be in range [1, {self.num_bands}]"
            )
            raise ValueError(err_msg)

        # Handle output directory and subdirectories
        output_fs = get_fs(self.output_dir, storage_options=self.write_kwargs.get("storage_options", {}))
        output_base_dir = output_fs.sep.join([self.output_dir, self.name])

        if is_not_empty(output_base_dir, output_fs):
            logger.warning(f"Output directory {output_base_dir} is not empty. Deleting it.")
            delete_dir(output_base_dir, output_fs)

        for band_range in self.get_band_iterations():
            output_dir = output_fs.sep.join([output_base_dir, f"band_{band_range[0]}-band_{band_range[1]}"])
            output_fs.makedirs(output_dir)
            self.output_paths.append(output_dir)

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
        self.output_columns = result
        self.dataset_name = task.dataset_name
        return task

    def insert_finished(self) -> None:
        self._check_actor_obj()
        self._actor_obj.insert_finished()

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        partition_paths = self._actor_obj.extract_and_write()
        return [
            FileGroupTask(
                task_id=partition_id,
                dataset_name=self.dataset_name + f"{self.name}",
                data=path,
                _metadata={
                    "partition_index": partition_id,
                    "total_partitions": len(partition_paths),
                    "output_columns": self.output_columns,
                },
            )
            for partition_id, path in partition_paths
        ]

    def teardown(self) -> None:
        self._check_actor_obj()
        self._actor_obj.cleanup()

    def get_band_iterations(self) -> Iterator[tuple[int, int]]:
        """Get all band ranges for iteration."""
        for band_start in range(0, self.num_bands, self.bands_per_iteration):
            band_range = (band_start, min(band_start + self.bands_per_iteration, self.num_bands))
            yield band_range
