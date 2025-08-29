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

import io
from dataclasses import dataclass

from loguru import logger

import nemo_curator.stages.video.filtering.motion_vector_backend as motion_backend
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Video, VideoTask


@dataclass
class MotionVectorDecodeStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for decoding motion vector information from video files.

    This class processes video files through a series of steps including decoding,
    filtering by side length, and storing the results in the task.
    """

    num_cpus_per_worker: float = 6.0
    verbose: bool = False
    target_fps: float = 2.0
    target_duration_ratio: float = 0.5
    _name: str = "motion_vector_decoding"

    def __post_init__(self) -> None:
        self._resources = Resources(cpus=self.num_cpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["decoded_motion_data"]

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data

        for clip in video.clips:
            if not clip.buffer:
                logger.warning(f"Clip {clip.uuid} has no buffer. Skipping...")
                clip.errors["buffer"] = "empty"
                continue

            with io.BytesIO(clip.buffer) as fp:
                try:
                    clip.decoded_motion_data = motion_backend.decode_for_motion(
                        fp,
                        thread_count=int(self.num_cpus_per_worker),
                        target_fps=self.target_fps,
                        target_duration_ratio=self.target_duration_ratio,
                    )
                except motion_backend.VideoResolutionTooSmallError:
                    if self.verbose:
                        logger.warning(f"Clip {clip.uuid} has too small resolution.")
                    clip.decoded_motion_data = None
                    clip.errors["motion_decode"] = "resolution_too_small"
                except Exception as e:  # noqa: BLE001
                    if self.verbose:
                        logger.exception(f"Clip {clip.uuid} failed to decode motion data: {e}")
                    clip.decoded_motion_data = None
                    clip.errors["motion_decode"] = "decode_failed"
                else:
                    if clip.decoded_motion_data is None or len(clip.decoded_motion_data.frames) == 0:
                        logger.error(f"Clip {clip.uuid} has no motion frames.")
                        clip.decoded_motion_data = None
                        clip.errors["motion_decode"] = "no_motion_frames"

        failed_cnt = sum(1 for clip in video.clips if clip.decoded_motion_data is None)
        logger.info(
            f"MotionVectorDecodeStage: Processed {len(video.clips)} clips for {task.task_id}, {failed_cnt} failed."
        )

        return task


@dataclass
class MotionFilterStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for filtering video clips based on motion score.

    This class processes video clips through a series of steps including motion score
    computation and filtering based on thresholds.
    """

    score_only: bool = False
    global_mean_threshold: float = 0.00098
    per_patch_min_256_threshold: float = 0.000001
    num_gpus_per_worker: float = 0
    motion_filter_batch_size: int = 256
    verbose: bool = False
    _name: str = "motion_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "decoded_motion_data",
            "motion_score_global_mean",
            "motion_score_per_patch_min_256",
            "filtered_clips",
            "clip_stats",
        ]

    def __post_init__(self) -> None:
        self._resources = Resources(gpus=self.num_gpus_per_worker)

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data

        passing_clips = []
        for clip in video.clips:
            if not clip.decoded_motion_data:
                if self.verbose:
                    logger.warning(f"Clip {clip.uuid} has no decoded motion data.")
                fake_score = -1.0
                motion_info = motion_backend.MotionInfo(
                    fake_score < self.global_mean_threshold or fake_score < self.per_patch_min_256_threshold,
                    fake_score,
                    fake_score,
                )
            else:
                motion_info = motion_backend.check_if_small_motion(
                    clip.decoded_motion_data.frames,
                    clip.decoded_motion_data.frame_size,
                    global_mean_threshold=self.global_mean_threshold,
                    per_patch_min_256_threshold=self.per_patch_min_256_threshold,
                    use_gpu=self.num_gpus_per_worker > 0,
                    batch_size=self.motion_filter_batch_size,
                )

            clip.decoded_motion_data = None
            clip.motion_score_global_mean = motion_info.global_mean
            clip.motion_score_per_patch_min_256 = motion_info.per_patch_min_256
            if motion_info.is_small_motion:
                if self.score_only:
                    passing_clips.append(clip)
                else:
                    video.filtered_clips.append(clip)
                    video.clip_stats.num_filtered_by_motion += 1
                if self.verbose:
                    logger.info(
                        f"Clip {clip.uuid} has motion score global mean {clip.motion_score_global_mean:.5f}"
                        f"<{self.global_mean_threshold} or per-patch min 256 "
                        f"{clip.motion_score_per_patch_min_256:.6f}<{self.per_patch_min_256_threshold}, "
                        f"skipped.",
                    )
            else:
                passing_clips.append(clip)
                if self.verbose:
                    logger.info(
                        f"Clip {clip.uuid} has motion score global mean {clip.motion_score_global_mean:.5f}"
                        f">={self.global_mean_threshold} and per-patch min 256 "
                        f"{clip.motion_score_per_patch_min_256:.6f}>={self.per_patch_min_256_threshold}, "
                        f"kept.",
                    )
        video.clips = passing_clips

        logger.info(
            f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
            f"{len(video.clips)}/{len(video.filtered_clips) + len(video.clips)} clips "
            "passed/filtered",
        )

        # @aot TODO: free memory periodically

        return task
