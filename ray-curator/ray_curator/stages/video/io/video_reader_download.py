from dataclasses import dataclass

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.video.io.video_download import VideoDownloadStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.tasks import VideoTask, _EmptyTask


@dataclass
class VideoReaderDownloadStage(CompositeStage[_EmptyTask, VideoTask]):
    """Composite stage that reads video files from storage and downloads/processes them.

    This stage combines VideoReaderStage and VideoDownloadStage into a single
    high-level operation for reading video files from a directory and processing
    them with metadata extraction.

    Args:
        input_video_path: Path to the directory containing video files
        video_limit: Maximum number of videos to process (-1 for unlimited)
        verbose: Whether to enable verbose logging during download/processing
    """
    input_video_path: str
    video_limit: int = -1
    verbose: bool = False

    def __post_init__(self):
        """Initialize the parent CompositeStage after dataclass initialization."""
        super().__init__()

    @property
    def name(self) -> str:
        return "video_reader_download"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent execution stages.

        Returns:
            List of processing stages: [VideoReaderStage, VideoDownloadStage]
        """
        reader_stage = VideoReaderStage(
            input_video_path=self.input_video_path,
            video_limit=self.video_limit
        )

        download_stage = VideoDownloadStage(
            verbose=self.verbose
        )

        return [reader_stage, download_stage]

    def get_description(self) -> str:
        """Get a description of what this composite stage does."""
        return (
            f"Reads video files from '{self.input_video_path}' "
            f"(limit: {self.video_limit if self.video_limit > 0 else 'unlimited'}) "
            f"and downloads/processes them with metadata extraction"
        )
