
from dataclasses import dataclass

from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.io.reader.file_partitioning import FilePartitioningStage
from ray_curator.tasks import FileGroupTask, _EmptyTask
from ray_curator.utils.storage_utils import get_files_relative, get_storage_client, read_json_file


@dataclass
class ClientPartitioningStage(FilePartitioningStage):
    """Stage that partitions input file paths from a client into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.
    """

    input_s3_profile_name: str | None = None
    input_list_json_path: str | None = None
    _name: str = "client_partitioning"

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self.client_input = get_storage_client(self.file_paths, profile_name=self.input_s3_profile_name)

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        assert self.file_paths is not None

        if self.input_list_json_path is not None:
            inputs = _read_list_json(self.file_paths, self.input_list_json_path, self.input_s3_profile_name)
        else:
            inputs = get_files_relative(self.file_paths, self.client_input)

        # FILTER BY EXTENSIONS
        if self.file_extensions is not None:
            inputs = [x for x in inputs if any(x.endswith(ext) for ext in self.file_extensions)]

        # FILTER BY LIMIT
        if self.limit is not None and self.limit > 0:
            inputs = inputs[:self.limit]

        if self.files_per_partition is not None:
            inputs = self._partition_by_count(inputs, self.files_per_partition)

        # Create FileGroupTasks for each partition
        tasks = []
        dataset_name = self.file_paths
        for i, file_group in enumerate(inputs):
            file_task = FileGroupTask(
                task_id=f"file_group_{i}",
                dataset_name=dataset_name,
                data=file_group,
                _metadata={
                    "partition_index": i,
                    "total_partitions": len(inputs),
                    "storage_options": self.storage_options,
                    "source_files": file_group,  # Add source files for deterministic naming during write stage
                },
                reader_config={},  # Empty - will be populated by reader stage
            )
            tasks.append(file_task)

        return tasks

def _read_list_json(
    input_path: str,
    input_video_list_json_path: str,
    input_video_list_s3_profile_name: str,
) -> list[str]:
    input_videos = []
    client = get_storage_client(input_video_list_json_path, profile_name=input_video_list_s3_profile_name)
    try:
        data = read_json_file(input_video_list_json_path, client)
        listed_input_videos = [str(x) for x in data]
    except Exception as e:
        logger.exception(f"Failed to read input video list from {input_video_list_json_path}: {e}")
        raise

    for video_path in listed_input_videos:
        _input_path = input_path.rstrip("/") + "/"
        if not video_path.startswith(_input_path):
            error_msg = f"Input video {video_path} is not in {_input_path}"
            logger.exception(error_msg)
            raise ValueError(error_msg)
        input_videos.append(video_path[len(_input_path) :])

    return input_videos
