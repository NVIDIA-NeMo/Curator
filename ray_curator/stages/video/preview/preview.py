# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import subprocess
from dataclasses import dataclass

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Video, VideoTask, _Window
from nemo_curator.utils.operation_utils import make_pipeline_temporary_dir


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
    compression_level: int = 6  # 0-6, 0 is lossless, 6 is lossy
    quality: int = 50  # 0-100, 0 is worst quality, 100 is best quality

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def __post_init__(self) -> None:
        self._resources = Resources(cpus=self.num_cpus_per_worker)

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data

        if video.metadata.framerate < self.target_fps:
            logger.warning(
                f"Video {video.input_video} has framerate {video.metadata.framerate} < {self.target_fps}, preview generation quality will be degraded."
            )

        if video.metadata.height < self.target_height:
            logger.warning(
                f"Video {video.input_video} has height {video.metadata.height} < {self.target_height}, preview generation quality will be degraded."
            )

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
                str(self.compression_level),
                "-q:v",
                str(self.quality),
                "-loop",
                "0",
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
