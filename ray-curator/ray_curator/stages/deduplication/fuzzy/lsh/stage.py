import os
from dataclasses import dataclass

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.deduplication.fuzzy.lsh.lsh import LSHActor
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask


@dataclass
class LSHProcessingStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that performs LSH on a FileGroupTask containing minhash data.
    """

    _name = "LSHProcessingStage"
    _resources = Resources(gpus=1.0)

    # Core Algo objects
    actor_class = LSHActor

    # LSH parameters
    num_bands: int
    minhashes_per_band: int
    id_column: str = CURATOR_DEDUP_ID_STR
    minhash_field: str = "_minhash_signature"
    output_dir: str = "./"
    rmm_pool_size: int = 1024 * 1024 * 1024
    spill_device: int | str | None = "auto"
    enable_statistics: bool = False
    band_range: tuple[int, int] = None  # (start_band, end_band)

    def __post_init__(self):
        super().__init__()
        # TODO: do outside driver
        self.output_path = os.path.join(
            self.output_dir, f"{self.name}", f"band_{self.band_range[0]}-band_{self.band_range[1]}"
        )
        os.makedirs(self.output_path, exist_ok=True)

        self.actor_kwargs = {
            "num_bands": self.num_bands,
            "minhashes_per_band": self.minhashes_per_band,
            "id_column": self.id_column,
            "minhash_field": self.minhash_field,
            "output_path": self.output_path,
            "rmm_pool_size": self.rmm_pool_size,
            "spill_device": self.spill_device,
            "enable_statistics": self.enable_statistics,
        }
        if self.band_range is None:
            self.band_range = (0, self.num_bands)

        if self.band_range[0] < 0 or self.band_range[1] > self.num_bands or self.band_range[0] >= self.band_range[1]:
            err_msg = f"Invalid band range: {self.band_range}, must be in range [0, {self.num_bands})"
            raise ValueError(err_msg)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        err_msg = "LSHProcessingStage does not support the process method."
        raise NotImplementedError(err_msg)

    def _check_actor_obj(self) -> None:
        if not hasattr(self, "_actor_obj"):
            error = "Actor object not initialized. This might be because an incorrect executor was used or it failed to setup the stage properly."
            raise RuntimeError(error)

    def read_and_insert(self, task: FileGroupTask) -> FileGroupTask:
        self._check_actor_obj()
        result = self._actor_obj.read_and_insert(task.data, self.band_range)
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


@dataclass
class LSHStage(CompositeStage[FileGroupTask, FileGroupTask]):
    """
    Stage that performs LSH on a FileGroupTask containing minhash data.
    """

    _name = "LSHStage"

    # LSH parameters
    num_bands: int
    minhashes_per_band: int
    id_column: str = CURATOR_DEDUP_ID_STR
    minhash_field: str = "_minhash_signature"
    output_path: str = "./"
    rmm_pool_size: int = 1024 * 1024 * 1024
    spill_device: int | str | None = "auto"
    enable_statistics: bool = False
    # number of bands to process in a single LSH shuffle. [1, num_bands]
    # Higher values will improve performance but will also increase memory usage. Risking out of memory errors.
    bands_per_iteration: int = 5

    def __post_init__(self):
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        stages = []
        for i, band_start_id in enumerate(range(0, self.num_bands, self.bands_per_iteration)):
            band_range = (band_start_id, min(band_start_id + self.bands_per_iteration, self.num_bands))
            stages.append(
                LSHProcessingStage(
                    num_bands=self.num_bands,
                    minhashes_per_band=self.minhashes_per_band,
                    id_column=self.id_column,
                    minhash_field=self.minhash_field,
                    output_path=self.output_path,
                    rmm_pool_size=self.rmm_pool_size,
                    spill_device=self.spill_device,
                    enable_statistics=self.enable_statistics,
                    band_range=band_range,
                ).with_(name=f"LSHProcessingStage_{i}")
            )
        return stages
