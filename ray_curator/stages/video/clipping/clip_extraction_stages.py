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
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

from cosmos_xenna.ray_utils.resources import _get_local_gpu_info, _make_gpu_resources_from_gpu_name
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources, _get_gpu_memory_gb
from nemo_curator.tasks.video import Clip, Video, VideoTask
from nemo_curator.utils import grouping
from nemo_curator.utils.operation_utils import make_pipeline_temporary_dir


@dataclass
class ClipTranscodingStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that transcodes video clips into a standardized format.

    This stage handles the conversion of video clips using FFmpeg, supporting both
    software (libx264, libopenh264) and hardware (NVENC) encoding with configurable parameters.

    Args:
        num_cpus_per_worker: Number of CPUs per worker.
        encoder: Video encoder to use.
        encoder_threads: Number of threads per encoder.
        encode_batch_size: Number of clips to encode in parallel.
        nb_streams_per_gpu: Number of streams per GPU.
        use_hwaccel: Whether to use hardware acceleration.
        use_input_bit_rate: Whether to use input video bit rate.
        num_clips_per_chunk: Number of clips per chunk. If the number of clips is larger than this, the clips will be split into chunks, and created VideoTasks for each chunk.
        verbose: Whether to print verbose logs.
        ffmpeg_verbose: Whether to print FFmpeg verbose logs.
    """

    num_cpus_per_worker: float = 6.0
    encoder: str = "libx264"
    encoder_threads: int = 1
    encode_batch_size: int = 16
    nb_streams_per_gpu: int = 3
    use_hwaccel: bool = False
    use_input_bit_rate: bool = False
    num_clips_per_chunk: int = 32
    ffmpeg_verbose: bool = False
    verbose: bool = False
    _name: str = "clip_transcoding"

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """
        if self.encoder not in {"libopenh264", "libx264", "h264_nvenc"}:
            error_msg = f"Expected encoder of `libopenh264`, `libx264`, or `h264_nvenc`. Got {self.encoder}"
            raise ValueError(error_msg)

    def __post_init__(self) -> None:
        """Post-initialization method called after all fields are set."""
        if self.encoder == "h264_nvenc" or self.use_hwaccel:
            if self.nb_streams_per_gpu > 0:
                # Assume that we have same type of GPUs
                gpu_info = _get_local_gpu_info()[0]
                nvencs = _make_gpu_resources_from_gpu_name(gpu_info.name).num_nvencs
                gpu_memory_gb = _get_gpu_memory_gb()
                self._resources = Resources(
                    nvencs=nvencs // self.nb_streams_per_gpu, gpu_memory_gb=gpu_memory_gb // self.nb_streams_per_gpu
                )
            else:
                self._resources = Resources(gpus=1)
        else:
            self._resources = Resources(cpus=self.num_cpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["source_bytes"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
        }

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
                task_id=f"{task.task_id}_chunk_{idx}",
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
        """Extract clips using ffmpeg."""
        # construct ffmpeg command
        command = self._build_ffmpeg_command(video_filename, clips, force_pix_fmt, use_bit_rate)

        # run ffmpeg command
        self._run_ffmpeg_command(command, working_dir, clips)

        # read clips back into memory
        self._read_clips_to_memory(working_dir, clips)

    def _build_ffmpeg_command(
        self,
        video_filename: str,
        clips: list[Clip],
        force_pix_fmt: bool,
        use_bit_rate: str | None,
    ) -> list[str]:
        """Build the ffmpeg command for extracting clips."""
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
        if self.use_hwaccel:
            if self.encoder == "h264_nvenc":
                command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
            else:
                command.extend(["-hwaccel", "auto"])

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
        """Run the ffmpeg command and handle errors."""
        try:
            if self.verbose:
                logger.info(f"Executing ffmpeg command: {' '.join(command)}")
            output = subprocess.check_output(  # noqa: S603
                command, cwd=working_dir, stderr=subprocess.STDOUT
            )
            if output and self.ffmpeg_verbose:
                logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            self._handle_ffmpeg_error(e, command, clips)

    def _handle_ffmpeg_error(
        self, error: subprocess.CalledProcessError, command: list[str], clips: list[Clip]
    ) -> None:
        """Handle ffmpeg command errors."""
        logger.error(f"ffmpeg command failed with return code {error.returncode}")
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
    _name: str = "fixed_stride_extractor"

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
