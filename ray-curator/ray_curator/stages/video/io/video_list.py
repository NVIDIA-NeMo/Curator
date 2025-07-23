import pathlib
from dataclasses import dataclass

from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import Video, VideoTask, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class VideoListStage(ProcessingStage[_EmptyTask, VideoTask]):
    """Stage that discovers video files in a directory and creates VideoTask objects.

    This stage scans a specified directory (recursively) for video files with supported
    extensions and creates individual VideoTask objects for each discovered file. It serves
    as the entry point for video processing pipelines by converting file paths into
    structured task objects.

    The stage performs the following operations:
    1. Recursively scans the input directory for video files
    2. Filters files by supported extensions (.mp4, .mov, .avi, .mkv, .webm)
    3. Optionally limits the number of files processed
    4. Creates VideoTask objects with unique task IDs for each video file

    Args:
        input_video_path: Path to the directory containing video files to process
        video_limit: Maximum number of video files to process (-1 for unlimited)
        
    Note:
        This stage only discovers and lists files - actual video reading and metadata
        extraction is performed by subsequent stages like VideoReaderStage.
    """
    input_video_path: str
    video_limit: int = -1
    _name: str = "video_list"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define the input attributes required by this stage.
        
        Returns:
            Tuple of ([], []) - this stage requires no input attributes as it
            generates tasks from filesystem discovery rather than processing existing tasks.
        """
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage.
        
        Returns:
            Tuple of (top_level_attrs, data_attrs) where:
            - top_level_attrs: ["data"] - produces VideoTask.data
            - data_attrs: ["input_video"] - populates Video.input_video with file paths
        """
        return ["data"], ["input_video"]

    def process(self, _: _EmptyTask) -> list[VideoTask]:
        """Discover video files in the directory and create VideoTask objects.

        Recursively scans the input directory for video files with supported extensions,
        applies the video limit if specified, and creates a VideoTask object for each
        discovered file with appropriate task ID and dataset name.

        Args:
            _: Empty task (unused, as this stage generates tasks from filesystem scan).

        Returns:
            List of VideoTask objects, one for each discovered video file.

        Raises:
            ValueError: If input_video_path is None or not set.
        """
        if self.input_video_path is None:
            msg = "input_video_path is not set"
            raise ValueError(msg)
        files = get_all_files_paths_under(
            self.input_video_path,
            recurse_subdirectories=True,
            keep_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
        )
        logger.info(f"Found {len(files)} files under {self.input_video_path}")

        if self.video_limit > 0:
            files = files[:self.video_limit]
            logger.info(f"Using first {len(files)} files under {self.input_video_path} since video_limit is set to {self.video_limit}")

        video_tasks = []
        for fp in files:

            file_path = fp
            if isinstance(file_path, str):
                file_path = pathlib.Path(file_path)

            video = Video(input_video=file_path)
            video_task = VideoTask(
                task_id=f"{file_path}_processed",
                dataset_name=self.input_video_path,
                data=video,
            )
            video_tasks.append(video_task)

        return video_tasks
