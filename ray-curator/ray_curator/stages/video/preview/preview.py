import pathlib
import subprocess
from dataclasses import dataclass

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks.video import VideoTask, _Window, Video
from ray_curator.stages.resources import Resources
from ray_curator.utils.operation_utils import make_pipeline_temporary_dir
from loguru import logger


@dataclass
class PreviewStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that generates webp previews from video clips.

    This class processes video clips through a series of steps including reading,
    generating webp previews, and writing to storage.
    """
    target_fps: float = 1.0
    target_height: int = 240
    verbose: bool = False
    num_cpus_per_worker: float = 4.0

    @property
    def resources(self) -> Resources:
        return Resources(cpus=self.num_cpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data
        for clip in video.clips:
            for window in clip.windows:
                self._generate_preview(window)
        return task

    def _generate_preview(self, window: _Window) -> None:
        """Generate webp preview for a video window.

        Args:
            window: Window containing video data to generate preview for.

        """
        with make_pipeline_temporary_dir(sub_dir="preview") as tmp_dir:
            input_mp4 = pathlib.Path(tmp_dir, "input.mp4")

            assert window.mp4_bytes is not None
            input_mp4.write_bytes(window.mp4_bytes)
            output_webp = pathlib.Path(tmp_dir, "output.webp")
            command = [
                "ffmpeg",
                "-threads",
                str(int(self.resources.cpus)),
                "-y",
                "-i",
                input_mp4.as_posix(),
                "-loglevel",
                "error",
                "-vf",
                f"fps={self.target_fps},scale=-1:{self.target_height}",
                "-c:v",
                "libwebp",
                "-lossless",
                str(0),
                "-compression_level",
                str(6),
                "-q:v",
                str(50),
                "-loop",
                "0",
                "-threads",
                str(int(self.resources.cpus)),
                output_webp.as_posix(),
            ]

            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)  # noqa: S603
                if output:
                    logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg command failed with return code {e.returncode}")
                logger.warning(f"ffmpeg command: {' '.join(command)}")
                if e.output:
                    logger.warning(f"ffmpeg output: {e.output.decode('utf-8')}")
                return

            window.webp_bytes = output_webp.read_bytes()