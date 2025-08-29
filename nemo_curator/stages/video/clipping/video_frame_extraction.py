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

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import VideoTask
from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file

try:
    from nemo_curator.utils.nvcodec_utils import PyNvcFrameExtractor

    _PYNVC_AVAILABLE = True
except (ImportError, RuntimeError):
    logger.warning("PyNvcFrameExtractor not available, PyNvCodec mode will fall back to FFmpeg")
    PyNvcFrameExtractor = None
    _PYNVC_AVAILABLE = False


def get_frames_from_ffmpeg(
    video_file: Path,
    width: int,
    height: int,
    *,
    use_gpu: bool = False,
) -> npt.NDArray[np.uint8] | None:
    """Fetch resized frames for video."""
    if use_gpu:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "1",
            "-hwaccel",
            "auto",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            video_file.as_posix(),
            "-vf",
            f"scale_npp={width}:{height},hwdownload,format=nv12",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "4",
            "-i",
            video_file.as_posix(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-",
        ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    video_stream, err = process.communicate()
    if process.returncode != 0:
        if use_gpu:
            logger.warning("Caught ffmpeg runtime error with `use_gpu=True` option, falling back to CPU.")
            return get_frames_from_ffmpeg(video_file, width, height, use_gpu=False)
        logger.exception(f"FFmpeg error: {err.decode('utf-8')}")
        return None
    return np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])


@dataclass
class VideoFrameExtractionStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that extracts frames from videos into numpy arrays.

    This stage handles video frame extraction using either FFmpeg (CPU/GPU) or PyNvCodec,
    converting video content into standardized frame arrays for downstream processing.
    """

    output_hw: tuple[int, int] = (27, 48)
    pyncv_batch_size: int = 64
    decoder_mode: str = "pynvc"
    verbose: bool = False
    _name: str = "video_frame_extraction"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """
        if self.decoder_mode == "pynvc":
            if _PYNVC_AVAILABLE and PyNvcFrameExtractor is not None:
                self.pynvc_frame_extractor = PyNvcFrameExtractor(
                    width=self.output_hw[1],
                    height=self.output_hw[0],
                    batch_size=self.pyncv_batch_size,
                )
            else:
                logger.warning("PyNvcFrameExtractor not available, will fall back to FFmpeg for video processing")
                self.pynvc_frame_extractor = None

    def __post_init__(self) -> None:
        if self.decoder_mode == "pynvc":
            self._resources = Resources(gpu_memory_gb=10)
        else:
            self._resources = Resources(cpus=4.0)

    def process(self, task: VideoTask) -> VideoTask:
        width, height = self.output_hw
        video = task.data

        if video.source_bytes is None:
            msg = "Video source bytes are not available"
            raise ValueError(msg)

        if not video.has_metadata():
            logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
            video.errors["metadata"] = "incomplete"
            return task

        with make_pipeline_named_temporary_file(sub_dir="frame_extraction") as video_path:
            with video_path.open("wb") as fp:
                fp.write(video.source_bytes)
            if self.decoder_mode == "pynvc":
                if self.pynvc_frame_extractor is not None:
                    try:
                        video.frame_array = self.pynvc_frame_extractor(video_path).cpu().numpy().astype(np.uint8)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Got exception {e} with PyNvVideoCodec decode, trying ffmpeg CPU fallback")
                        video.frame_array = get_frames_from_ffmpeg(
                            video_path,
                            width=width,
                            height=height,
                            use_gpu=False,
                        )
                else:
                    logger.info("PyNvcFrameExtractor not available, using FFmpeg CPU fallback")
                    video.frame_array = get_frames_from_ffmpeg(
                        video_path,
                        width=width,
                        height=height,
                        use_gpu=False,
                    )
            else:
                logger.info(f"Decoding video {video.input_video} with FFmpeg")
                video.frame_array = get_frames_from_ffmpeg(
                    video_path,
                    width=width,
                    height=height,
                    use_gpu=self.decoder_mode == "ffmpeg_gpu",
                )
                logger.info(f"Decoded video {video.input_video} with FFmpeg successfully")
            if video.frame_array is None:
                logger.error("Frame extraction failed, exiting...")
                return None
            if self.verbose:
                logger.info(f"Loaded video as numpy uint8 array with shape {video.frame_array.shape}")
        return task
