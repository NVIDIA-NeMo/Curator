import pathlib
from dataclasses import dataclass

from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import Video, VideoTask, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class VideoReaderStage(ProcessingStage[_EmptyTask, VideoTask]):
    """Stage that reads video files from storage and extracts metadata."""
    input_video_path: str
    video_limit: int = -1
    _name: str = "video_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["input_video"]

    def process(self, _: _EmptyTask) -> list[VideoTask]:
        """Process a single group of video files."""
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
