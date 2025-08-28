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
import math
from dataclasses import dataclass

from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Video, VideoTask
from nemo_curator.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    FramePurpose,
    extract_frames,
)


@dataclass
class ClipFrameExtractionStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for extracting frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    target frame rate selection, and frame extraction signature creation.
    """

    extraction_policies: tuple[FrameExtractionPolicy, ...] = (FrameExtractionPolicy.sequence,)
    extract_purposes: list[FramePurpose] | None = None
    target_res: tuple[int, int] | None = None
    verbose: bool = False
    num_cpus: int = 3
    target_fps: list[float | int] | None = None
    _name: str = "clip_frame_extraction"

    def __post_init__(self) -> None:
        self._resources = Resources(cpus=self.num_cpus)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips.extracted_frames"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.target_fps is None:
            if self.extract_purposes is not None:
                self.target_fps = [purpose.value for purpose in self.extract_purposes]
            else:
                self.target_fps = [2]  # default fallback

        if self.target_res is None:
            self.target_res = (-1, -1)
        logger.info(f"ClipFrameExtractionStage will extract frames at {self.target_fps} FPS")

    def lcm_multiple(self, fps: list[float | int]) -> float | int:
        """Compute LCM of a list of fps targets."""
        fps = [int(fps) for fps in fps]
        return math.lcm(*fps)

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data
        for clip in video.clips:
            if clip.buffer is None:
                logger.error(f"Clip {clip.uuid} has no buffer")
                clip.errors["buffer"] = "empty"
                continue

            try:
                for policy in self.extraction_policies:
                    """
                    To save on decode costs, calculate the least-common-multiple(LCM) of fps
                    targets and apply decord.get_batch on this LCM fps
                    """
                    use_lcm_fps = len(self.target_fps) > 1 and all(
                        (fps.is_integer() if isinstance(fps, float) else isinstance(fps, int))
                        for fps in self.target_fps
                    )
                    if use_lcm_fps:
                        lcm = self.lcm_multiple(self.target_fps)
                        with io.BytesIO(clip.buffer) as fp:
                            frames = extract_frames(
                                fp,
                                extraction_policy=policy,
                                sample_rate_fps=lcm,
                                target_res=self.target_res,
                                num_threads=self.num_cpus,
                            )
                            for fps in self.target_fps:
                                signature = FrameExtractionSignature(
                                    extraction_policy=policy,
                                    target_fps=fps,
                                ).to_str()
                                clip.extracted_frames[signature] = frames[:: int(lcm / fps)]
                    else:
                        for fps in self.target_fps:
                            with io.BytesIO(clip.buffer) as fp:
                                frames = extract_frames(
                                    fp,
                                    extraction_policy=policy,
                                    sample_rate_fps=fps,
                                    target_res=self.target_res,
                                    num_threads=self.num_cpus,
                                )
                                signature = FrameExtractionSignature(
                                    extraction_policy=policy,
                                    target_fps=fps,
                                ).to_str()
                                clip.extracted_frames[signature] = frames
                                if self.verbose:
                                    logger.info(f"Extracted {len(frames)} frames from clip {clip.uuid} at {fps} fps")
            except (ValueError, OSError, RuntimeError) as e:
                logger.exception(f"Error extracting frames for clip {clip.uuid}: {e}")
                clip.errors["frame_extraction"] = "video_decode_failed"
                # reset the buffer to disable further operations on this clip
                clip.buffer = None
                continue

        if self.verbose:
            logger.info(f"ClipFrameExtractionStage extracted frames for {len(video.clips)} clips")
        return task
