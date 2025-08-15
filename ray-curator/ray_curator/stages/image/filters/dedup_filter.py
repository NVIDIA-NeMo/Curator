from dataclasses import dataclass
import os

from loguru import logger
import pyarrow.parquet as pq

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch


@dataclass
class DedupFilterStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Filter stage that removes images whose IDs appear in a Parquet file.

    The Parquet file must contain a column with image identifiers; by default this
    column is assumed to be ``image_id`` to match writer metadata. You can change
    the column name via ``id_column``.
    """

    remove_parquet_path: str
    id_column: str = "image_id"
    verbose: bool = False

    _name: str = "image_dedup_filter"

    # Internal cache
    _ids_to_remove: set[str] | None = None

    @property
    def resources(self) -> Resources:
        # CPU-only filter
        return Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata=None) -> None:  # noqa: ANN001
        if not os.path.isfile(self.remove_parquet_path):
            msg = f"Parquet file not found: {self.remove_parquet_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        table = pq.read_table(self.remove_parquet_path, columns=[self.id_column])

        # Convert to set of strings
        ids_array = table[self.id_column].to_pylist()
        # Normalize to strings (ImageObject.image_id is str)
        self._ids_to_remove = {str(x) for x in ids_array if x is not None}

        if self.verbose:
            logger.info(
                f"Loaded {len(self._ids_to_remove)} IDs to remove from '{self.remove_parquet_path}'"
            )

    def process(self, task: ImageBatch) -> ImageBatch:
        original_count = len(task.data)
        ids_to_remove = self._ids_to_remove  # local reference for faster lookups
        filtered_images = [img for img in task.data if img.image_id not in ids_to_remove]

        removed_count = original_count - len(filtered_images)
        if self.verbose:
            logger.info(
                f"Dedup filtering: kept {len(filtered_images)}/{original_count} images, "
                f"removed {removed_count} by ID"
            )

        return ImageBatch(
            data=filtered_images,
            dataset_name=task.dataset_name,
            task_id=f"{task.task_id}_{self._name}",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )