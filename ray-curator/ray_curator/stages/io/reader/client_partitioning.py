
from dataclasses import dataclass
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import FileGroupTask
from ray_curator.tasks import _EmptyTask
from ray_curator.backends.base import WorkerMetadata
from ray_curator.utils.storage_utils import get_storage_client, read_json_file, get_files_relative
from loguru import logger


@dataclass
class ClientPartitioningStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Stage that partitions input file paths from a client into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.
    """

    input_path: str
    input_s3_profile_name: str | None = None
    files_per_partition: int | None = None
    input_list_json_path: str | None = None
    limit: int | None = None
    _name: str = "client_partitioning"

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self.client_input = get_storage_client(self.input_path, profile_name=self.input_s3_profile_name)
    
    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        if self.input_list_json_path is not None:
            inputs = _read_list_json(self.input_path, self.input_list_json_path, self.input_s3_profile_name)
        else:
            inputs = get_files_relative(self.input_path, self.client_input)
        
        print(inputs)

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