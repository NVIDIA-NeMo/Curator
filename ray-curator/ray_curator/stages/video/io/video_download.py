import os
import pathlib
from dataclasses import dataclass

from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import Video, VideoTask, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class VideoDownloadStage(ProcessingStage[_EmptyTask, VideoTask]):
    """Stage that downloads video files from storage and extracts metadata.

    This class processes video files through a series of steps including downloading,
    extracting metadata, and storing the results in the task.
    """
    folder_path: str = None
    debug: bool = False

    @property
    def name(self) -> str:
        return "video_download"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, _: _EmptyTask) -> list[VideoTask]:
        """Process a single group of video files.

        Args:
            task (FileGroupTask): FileGroupTask containing file paths and configuration

        Returns:
            VideoTask: VideoTask with the data from these files
        """
        # Get list of files
        files = self._get_file_list()
        logger.info(f"Found {len(files)} files")

        video_tasks = []
        for fp in files:
            if not os.path.exists(fp):
                msg = f"Video file {fp} does not exist"
                raise FileNotFoundError(msg)

            file_path =fp
            if isinstance(file_path, str):
                file_path = pathlib.Path(file_path)

            video = Video(input_video=file_path)
            video_task = VideoTask(
                task_id=f"{file_path}_processed",
                dataset_name=self.folder_path,
                data=video,
            )

            # Download video bytes
            if not self._download_video_bytes(video):
                continue

            # Extract metadata and validate video properties
            if not self._extract_and_validate_metadata(video):
                continue

            # Log video information
            self._log_video_info(video)

            video_tasks.append(video_task)
        return video_tasks

    def _download_video_bytes(self, video: Video) -> bool:
        """Download video bytes from storage.

        Args:
            video: Video object to download bytes for.

        Returns:
            True if download successful, False otherwise.

        """
        def _raise_s3_error() -> None:
            msg = "S3 client is required for S3 destination"
            raise TypeError(msg)

        try:
            if isinstance(video.input_video, pathlib.Path):
                with video.input_video.open("rb") as fp:
                    video.source_bytes = fp.read()
            # TODO: Add support for S3
            else:
                _raise_s3_error()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Got an exception {e!s} when trying to read {video.input_video}")
            video.errors["download"] = str(e)
            return False

        if video.source_bytes is None:
            # should never happen, but log it just in case
            logger.error(f"video.source_bytes is None for {video.input_video} without exceptions ???")
            video.source_bytes = b""

        return True

    def _extract_and_validate_metadata(self, video: Video) -> bool:
        """Extract metadata and validate video properties.

        Args:
            video: Video object to extract metadata for.

        Returns:
            True if metadata extraction successful, False otherwise.

        """
        try:
            video.populate_metadata()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to extract metadata for {video.input_video}: {e}")
            return False

        # Log warnings for missing metadata
        if video.metadata.video_codec is None:
            logger.warning(f"Codec could not be extracted for {video.input_video}!")
        if video.metadata.pixel_format is None:
            logger.warning(f"Pixel format could not be extracted for {video.input_video}!")

        return True

    def _log_video_info(self, video: Video) -> None:
        """Log video information after successful download and metadata extraction.

        Args:
            video: Video object with metadata.

        """
        meta = self._format_metadata_for_logging(video)
        logger.info(
            f"Downloaded {video.input_video} "
            f"size={meta['size']} "
            f"res={meta['res']} "
            f"fps={meta['fps']} "
            f"duration={meta['duration']} "
            f"weight={meta['weight']} "
            f"bit_rate={meta['bit_rate']}.",
        )

    def _format_metadata_for_logging(self, video: Video) -> dict[str, str]:
        """Format video metadata for logging, handling None values gracefully.

        Args:
            video: Video object with metadata.

        Returns:
            Dictionary of formatted metadata strings.

        """
        metadata = video.metadata

        # Format each field, using "unknown" for None values
        return {
            "size": f"{len(video.source_bytes):,}B" if video.source_bytes else "0B",
            "res": f"{metadata.width or 'unknown'}x{metadata.height or 'unknown'}",
            "fps": f"{metadata.framerate:.1f}" if metadata.framerate is not None else "unknown",
            "duration": f"{metadata.duration / 60:.0f}m" if metadata.duration is not None else "unknown",
            "weight": f"{video.weight:.2f}" if metadata.duration is not None else "unknown",
            "bit_rate": f"{metadata.bit_rate_k}K" if metadata.bit_rate_k is not None else "unknown",
        }

    def _get_file_list(self) -> list[str]:
        """Get the list of files to process."""
        if self.folder_path is None:
            msg = "folder_path is not set"
            raise ValueError(msg)
        files = get_all_files_paths_under(
            self.folder_path,
            recurse_subdirectories=True,
            keep_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
        )
        if self.debug:
            logger.info("DEBUG mode: Using 2 video files for DEBUG")
            return files[:2]
        return files
