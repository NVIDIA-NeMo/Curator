import pathlib
from dataclasses import dataclass

from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks.video import Video, VideoTask


@dataclass
class VideoReaderStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that reads video files from local filesystem and extracts metadata.

    This stage processes video files by reading their binary content from the local
    filesystem and extracting comprehensive metadata including dimensions, frame rate,
    duration, codecs, and other technical properties. The stage handles both the file
    I/O operations and metadata extraction, storing results in the VideoTask.

    The stage performs the following operations:
    1. Reads video file bytes from the local filesystem
    2. Extracts technical metadata using video analysis tools
    3. Validates metadata completeness and logs warnings for missing fields
    4. Optionally logs detailed video information when verbose mode is enabled

    Args:
        verbose: If True, logs detailed video information after successful processing
        
    Note:
        Currently supports local filesystem paths only. S3 support is planned for future releases.
    """
    verbose: bool = False
    _name: str = "video_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define the input attributes required by this stage.
        
        Returns:
            Tuple of (top_level_attrs, data_attrs) where:
            - top_level_attrs: ["data"] - requires VideoTask.data to be populated
            - data_attrs: ["input_video"] - requires Video.input_video path to be set
        """
        return ["data"], ["input_video"]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage.
        
        Returns:
            Tuple of (top_level_attrs, data_attrs) where:
            - top_level_attrs: ["data"] - populates VideoTask.data
            - data_attrs: ["source_bytes", "metadata"] - populates Video.source_bytes and Video.metadata
        """
        return ["data"], ["source_bytes", "metadata"]

    def process(self, task: VideoTask) -> VideoTask:
        """Process a video task by reading file bytes and extracting metadata.

        Performs the complete video processing workflow including reading the video
        file from disk, extracting technical metadata, and optionally logging
        detailed information. Returns the same task with populated data.

        Args:
            task: VideoTask containing a Video object with input_video path set.

        Returns:
            The same VideoTask with video.source_bytes and video.metadata populated.
            If errors occur, the task is returned with error information stored.
        """
        video = task.data
        # Download video bytes
        if not self._download_video_bytes(video):
            return task

        # Extract metadata and validate video properties
        if not self._extract_and_validate_metadata(video):
            return task

        # Log video information
        if self.verbose:
            self._log_video_info(video)

        return task

    def _download_video_bytes(self, video: Video) -> bool:
        """Read video file bytes from the local filesystem.

        Reads the complete binary content of the video file and stores it in the
        video.source_bytes attribute. Handles file I/O errors gracefully and logs
        appropriate error messages.

        Args:
            video: Video object containing the input_video path to read from.

        Returns:
            True if file reading was successful, False if an error occurred.

        Note:
            Errors are logged and stored in video.errors["download"] for debugging.
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
        """Extract comprehensive metadata from video file and validate completeness.

        Uses video analysis tools to extract technical metadata including dimensions,
        frame rate, duration, codecs, bit rate, and other properties. Logs warnings
        for critical missing metadata fields like codec and pixel format.

        Args:
            video: Video object with source_bytes populated for metadata extraction.

        Returns:
            True if metadata extraction completed successfully, False if extraction
            failed due to corrupted file or unsupported format.

        Note:
            Warnings are logged for missing critical fields, but the method may still
            return True if partial metadata was extracted successfully.
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
        """Log comprehensive video information after successful processing.

        Outputs detailed information about the processed video including file size,
        resolution, frame rate, duration, weight, and bit rate. This method is only
        called when verbose mode is enabled.

        Args:
            video: Video object with populated metadata fields.
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
        """Format video metadata into human-readable strings for logging output.

        Converts raw metadata values into formatted strings with appropriate units
        and handles None values gracefully by substituting "unknown" placeholders.
        Used by _log_video_info for consistent log formatting.

        Args:
            video: Video object with populated metadata fields.

        Returns:
            Dictionary mapping metadata field names to formatted string values,
            including size (bytes), resolution, fps, duration (minutes), weight, 
            and bit rate (Kbps).
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