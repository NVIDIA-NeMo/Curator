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

import copy
import pathlib
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, Video, VideoTask
from nemo_curator.utils import grouping
from nemo_curator.utils.operation_utils import make_pipeline_temporary_dir

SUPPORTED_ENCODERS = ("h264_nvenc", "libvpx-vp9", "libopenh264")

_BYO_H264_DOCS_URL = (
    "https://github.com/NVIDIA-NeMo/Curator/blob/main/fern/versions/main/pages/get-started/installation.mdx"
    "#software-h264hevcav1-codec-support-advanced"
)


@dataclass
class ClipTranscodingStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that transcodes video clips into a standardized format.

    This stage handles the conversion of video clips using FFmpeg. Supported
    encoders:

    - ``h264_nvenc`` — hardware H.264 via NVENC (recommended; requires an
      NVENC-equipped NVIDIA GPU — note that A100/H100 do not include NVENC).
    - ``libvpx-vp9`` — royalty-free VP9 software encoder (CPU fallback for
      non-NVENC GPUs). Significantly slower; emits a perf advisory.
    - ``libopenh264`` — H.264 software encoder. Not bundled with Curator's
      FFmpeg build for licensing reasons; users must install it themselves.
      The stage probes for it at setup time and raises a clear error pointing
      to the docs if it is not available.

    Args:
        num_cpus_per_worker: Number of CPUs per worker for Xenna scheduling. Does not affect Ray Data CPU scheduling; use ray_data_num_cpus for that.
        encoder: Video encoder to use.
        encoder_threads: Number of threads per encoder.
        encode_batch_size: Number of clips to encode in parallel.
        nb_streams_per_gpu: Number of streams per GPU.
        use_hwaccel: Whether to use hardware acceleration. Only valid with `h264_nvenc`.
        use_input_bit_rate: Whether to use input video bit rate.
        num_clips_per_chunk: Number of clips per chunk. If the number of clips is larger than this, the clips will be split into chunks, and created VideoTasks for each chunk.
        verbose: Whether to print verbose logs.
        ffmpeg_verbose: Whether to print FFmpeg verbose logs.
        ray_data_num_cpus: CPU cores reserved per Ray Data actor for this stage. Defaults to 1.0 on the CPU encoder path to enable stage fusion with upstream stages. Set to None to fall back to resources.cpus. Does not affect Xenna scheduling.
    """

    num_cpus_per_worker: float = 6.0
    encoder: str = "h264_nvenc"
    encoder_threads: int = 1
    encode_batch_size: int = 16
    nb_streams_per_gpu: int = 3
    use_hwaccel: bool = False
    use_input_bit_rate: bool = False
    num_clips_per_chunk: int = 32
    ffmpeg_verbose: bool = False
    verbose: bool = False
    name: str = "clip_transcoding"
    ray_data_num_cpus: float | None = (
        None  # CPU reservation for Ray Data scheduler; set to 1.0 on CPU path to enable stage fusion
    )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """
        if not shutil.which("ffmpeg"):
            msg = (
                "Could not find `ffmpeg` on PATH. ClipTranscodingStage requires "
                "FFmpeg built with supported video encoders. See docker/common/install_ffmpeg.sh."
            )
            raise RuntimeError(msg)
        if self.encoder not in SUPPORTED_ENCODERS:
            error_msg = f"Expected encoder in {SUPPORTED_ENCODERS}. Got {self.encoder}"
            raise ValueError(error_msg)
        if self.encoder == "libvpx-vp9" and self.use_hwaccel:
            error_msg = "use_hwaccel is not supported with libvpx-vp9 (CPU encoder)"
            raise ValueError(error_msg)
        if self.encoder == "libopenh264":
            self._verify_libopenh264_available()

    @staticmethod
    def _verify_libopenh264_available() -> None:
        """Probe the local FFmpeg build for libopenh264 support."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            error_msg = (
                "Could not find `ffmpeg` on PATH while verifying libopenh264 support. "
                f"Install FFmpeg and ensure it is on PATH. See {_BYO_H264_DOCS_URL}"
            )
            raise RuntimeError(error_msg)
        try:
            result = subprocess.run(  # noqa: S603
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired as e:
            error_msg = f"`ffmpeg -encoders` timed out while verifying libopenh264 support. See {_BYO_H264_DOCS_URL}"
            raise RuntimeError(error_msg) from e
        if "libopenh264" not in result.stdout:
            error_msg = (
                "encoder='libopenh264' was requested but the local FFmpeg build "
                "does not include it. Curator does not ship libopenh264 due to "
                "its patent-license redistribution model. To enable it, install "
                f"a libopenh264-enabled FFmpeg yourself — see {_BYO_H264_DOCS_URL}"
            )
            raise RuntimeError(error_msg)

    def __post_init__(self) -> None:
        """Post-initialization method called after all fields are set."""
        if self.encoder == "h264_nvenc" or self.use_hwaccel:
            if self.nb_streams_per_gpu > 0:
                # Assume that we have same type of GPUs
                self.resources = Resources(gpus=1.0 / self.nb_streams_per_gpu)
            else:
                self.resources = Resources(gpus=1)
        else:
            self.resources = Resources(cpus=self.num_cpus_per_worker)
            if self.ray_data_num_cpus is None:
                # Default to 1.0 so Ray Data fuses this stage with VideoReaderStage
                # and FixedStrideExtractorStage. Kept separate from resources.cpus
                # so Xenna scheduling is unaffected.
                self.ray_data_num_cpus = 1.0

        if self.encoder == "libvpx-vp9":
            logger.warning(
                "ClipTranscodingStage: libvpx-vp9 is significantly slower than "
                "h264_nvenc and libopenh264. If your GPU has NVENC, prefer "
                "encoder='h264_nvenc'. To use libopenh264 instead, see "
                f"{_BYO_H264_DOCS_URL}"
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["source_bytes"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        spec: dict[str, Any] = {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        if self.ray_data_num_cpus is not None:
            spec[RayStageSpecKeys.RAY_NUM_CPUS] = self.ray_data_num_cpus
        return spec

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data

        if not video.clips:
            logger.warning(f"No clips to transcode for {video.input_video}. Skipping...")
            video.source_bytes = None
            return task

        with make_pipeline_temporary_dir(sub_dir="transcode") as tmp_dir:
            # write video to file
            video_file = tmp_dir / "input.mp4"
            video_file.write_bytes(video.source_bytes)
            force_pix_fmt = video.is_10_bit_color() or False

            # use input video bit-rate
            use_bit_rate = None
            if self.use_input_bit_rate:
                use_bit_rate = str(video.metadata.bit_rate_k) + "K"

            # extract clips in batches
            for i in range(0, len(video.clips), self.encode_batch_size):
                batch = video.clips[i : i + self.encode_batch_size]
                self._extract_clips(
                    tmp_dir,
                    video_file.name,
                    force_pix_fmt=force_pix_fmt,
                    use_bit_rate=use_bit_rate,
                    clips=batch,
                )

        # we are done with source_bytes
        video.source_bytes = None

        # Consider craking into smaller chunks of clips
        output_tasks = []
        clip_durations = [clip.duration for clip in video.clips]
        if len(clip_durations) > 0:
            logger.info(
                f"video {video.input_video} has {len(video.clips)} "
                f"clips and weight={video.weight:.2f}; "
                f"min-clip={min(clip_durations):.2f}s, "
                f"max-clip={max(clip_durations):.1f}s.",
            )
        clip_chunks = list(
            grouping.split_by_chunk_size(
                video.clips,
                self.num_clips_per_chunk * 8,
                lambda x: int(x.span[1] - x.span[0]),
            ),
        )
        for idx in range(len(clip_chunks)):
            # create subtask for each video task
            subtask = VideoTask(
                dataset_name=task.dataset_name,
                data=Video(
                    input_video=video.input_video,
                    metadata=video.metadata,
                    clips=clip_chunks[idx],
                    num_total_clips=len(video.clips),
                    num_clip_chunks=len(clip_chunks),
                    clip_chunk_index=idx,
                    errors=copy.deepcopy(video.errors),
                ),
                _stage_perf=copy.deepcopy(task._stage_perf),
                _metadata=copy.deepcopy(task._metadata),
            )

            if self.verbose:
                logger.info(
                    f"Spawning subtask {idx} with {len(subtask.data.clips)} clips and weight={subtask.data.weight:.2f}",
                )
            output_tasks.append(subtask)
        logger.info(f"Creating {len(clip_chunks)} tasks for downstream from {video.input_video}.")

        return output_tasks

    def _extract_clips(
        self,
        working_dir: pathlib.Path,
        video_filename: str,
        *,
        force_pix_fmt: bool,
        use_bit_rate: str | None,
        clips: list[Clip],
    ) -> None:
        """Extract clips using FFmpeg."""
        # Construct FFmpeg command
        command = self._build_ffmpeg_command(video_filename, clips, force_pix_fmt, use_bit_rate)

        # Run FFmpeg command
        self._run_ffmpeg_command(command, working_dir, clips)

        # Read clips back into memory
        self._read_clips_to_memory(working_dir, clips)

    def _build_ffmpeg_command(
        self,
        video_filename: str,
        clips: list[Clip],
        force_pix_fmt: bool,
        use_bit_rate: str | None,
    ) -> list[str]:
        """Build the FFmpeg command for extracting clips."""
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning" if self.ffmpeg_verbose else "error",
        ]

        for i, clip in enumerate(clips):
            # Add decoder threads
            self._add_decoder_threads(command)

            # Add hardware acceleration if needed
            self._add_hwaccel_options(command)

            # Add input options
            self._add_input_options(command, clip, video_filename, i)

            # Add video encoding options
            self._add_video_encoding_options(command, use_bit_rate, force_pix_fmt)

            # Add output options
            self._add_output_options(command, clip, i)

        return command

    def _add_decoder_threads(self, command: list[str]) -> None:
        """Add decoder thread options to command."""
        thread_count = str(self.encoder_threads)
        command.extend(["-threads", thread_count])

    def _add_hwaccel_options(self, command: list[str]) -> None:
        """Add hardware acceleration options to command."""
        if self.use_hwaccel and self.encoder == "h264_nvenc":
            command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

    def _add_input_options(self, command: list[str], clip: Clip, video_filename: str, index: int) -> None:
        """Add input options to command."""
        start_s, end_s = clip.span
        command.extend(
            [
                "-ss",
                str(start_s),
                "-to",
                str(end_s),
                "-i",
                video_filename,
                "-map",
                f"{index}:v:0",
                "-c:v",
                self.encoder,
            ]
        )

    def _add_video_encoding_options(self, command: list[str], use_bit_rate: str | None, force_pix_fmt: bool) -> None:
        """Add video encoding options to command."""
        if use_bit_rate is not None:
            command.extend(["-b:v", use_bit_rate])

        if self.encoder == "h264_nvenc":
            self._add_nvenc_options(command, force_pix_fmt)
        elif self.encoder == "libvpx-vp9":
            self._add_libvpx_vp9_options(command, use_bit_rate, force_pix_fmt)

    def _add_nvenc_options(self, command: list[str], force_pix_fmt: bool) -> None:
        """Add NVENC-specific encoding options."""
        command.extend(
            [
                "-rc:v",
                "vbr",
                "-cq:v",
                "21",
                "-tune",
                "hq",
                "-b_ref_mode",
                "middle",
                "-temporal-aq",
                "1",
                "-rc-lookahead",
                "20",
                "-spatial-aq",
                "1",
            ]
        )

        if force_pix_fmt:
            command.extend(["-pix_fmt", "yuv420p"])

    def _add_libvpx_vp9_options(self, command: list[str], use_bit_rate: str | None, force_pix_fmt: bool) -> None:
        """Add libvpx-vp9 (CPU) encoding options."""
        # Constant-quality mode when no explicit bitrate is requested.
        # libvpx-vp9 requires `-b:v 0` to honor `-crf` exactly.
        if use_bit_rate is None:
            command.extend(["-b:v", "0", "-crf", "31"])
        command.extend(
            [
                "-deadline",
                "good",
                "-cpu-used",
                "4",
                "-row-mt",
                "1",
                "-tile-columns",
                "2",
            ]
        )

        if force_pix_fmt:
            command.extend(["-pix_fmt", "yuv420p"])

    def _add_output_options(self, command: list[str], clip: Clip, index: int) -> None:
        """Add output options to command."""
        # Add encoder threads
        thread_count = str(self.encoder_threads)
        command.extend(["-threads", thread_count])

        # Add audio and output filename
        command.extend(
            [
                "-map",
                f"{index}:a:0?",
                "-c:a",
                "copy",
                f"{clip.uuid}.mp4",
            ]
        )

    def _run_ffmpeg_command(self, command: list[str], working_dir: pathlib.Path, clips: list[Clip]) -> None:
        """Run the FFmpeg command and handle errors."""
        try:
            if self.verbose:
                logger.info(f"Executing FFmpeg command: {' '.join(command)}")
            output = subprocess.check_output(  # noqa: S603
                command, cwd=working_dir, stderr=subprocess.STDOUT
            )
            if output and self.ffmpeg_verbose:
                logger.warning(f"FFmpeg output: {output.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            self._handle_ffmpeg_error(e, command, clips)

    def _handle_ffmpeg_error(
        self, error: subprocess.CalledProcessError, command: list[str], clips: list[Clip]
    ) -> None:
        """Handle FFmpeg command errors."""
        logger.error(f"FFmpeg command failed with return code {error.returncode}")
        logger.error(f"Error: {error}")
        logger.warning(f"Command: {' '.join(command)}")
        if error.output:
            logger.warning(f"Error output: {error.output.decode('utf-8')}")

        for clip in clips:
            clip.errors["transcode"] = error.output.decode("utf-8") if error.output else str(error)

    def _read_clips_to_memory(self, working_dir: pathlib.Path, clips: list[Clip]) -> None:
        """Read extracted clips back into memory."""
        for clip in clips:
            clip.buffer = (working_dir / f"{clip.uuid}.mp4").read_bytes()


@dataclass
class FixedStrideExtractorStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that extracts video clips using fixed-length intervals.

    This stage splits videos into clips of specified length and stride, ensuring
    each clip meets minimum length requirements and optionally limiting total clips.
    """

    clip_len_s: float
    clip_stride_s: float
    min_clip_length_s: float
    limit_clips: int
    verbose: bool = False
    name: str = "fixed_stride_extractor"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        if video.source_bytes is None:
            msg = "Video source bytes are not available"
            raise ValueError(msg)

        if not video.has_metadata():
            logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
            video.errors["metadata"] = "incomplete"
            return task

        if self.limit_clips > 0 and len(video.clips) >= self.limit_clips:
            logger.warning(f"Skipping {video.input_video} because it has already been clipped")
            return task

        file = video.input_video
        if video.metadata.num_frames is None or video.metadata.framerate is None:
            msg = f"Incomplete metadata for {video.input_video}: Either metadata.num_frames or metadata.framerate is None."
            raise ValueError(msg)

        duration = video.metadata.num_frames / video.metadata.framerate if video.metadata.framerate > 0 else -1

        # create clip bounds based on clip_len_s and clip_stride_s
        clip_start = 0.0
        clip_bounds: list[tuple[float, float]] = []
        while clip_start < duration:
            clip_end = min(clip_start + self.clip_len_s, duration)
            if (clip_end - clip_start) >= self.min_clip_length_s:
                clip_bounds.append((clip_start, clip_end))
            clip_start += self.clip_stride_s

        for span in clip_bounds:
            start_event = int(span[0] * video.metadata.framerate)
            end_event = int(span[1] * video.metadata.framerate)
            clip = Clip(
                uuid=uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{file}_{start_event}_{end_event}",
                ),
                source_video=str(file),
                span=span,
            )
            video.clips.append(clip)

        logger.info(f"Extracted {len(task.data.clips)} clips from {task.data.input_video}")
        return task
