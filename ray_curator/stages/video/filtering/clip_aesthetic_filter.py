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

from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.clip import CLIPAestheticScorer
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import VideoTask
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature


@dataclass
class ClipAestheticFilterStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for filtering video clips based on CLIP aesthetic score.

    This class processes video clips through a series of steps including aesthetic score
    calculation and filtering based on thresholds.
    """

    model_dir: str = "models/clip_aesthetic"
    score_threshold: float = 0.5
    reduction: Literal["mean", "min"] = "min"
    target_fps: float = 1.0
    num_gpus_per_worker: float = 0.25
    verbose: bool = False
    _name: str = "clip_aesthetic_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["decoded_motion_data"]

    def __post_init__(self) -> None:
        self._resources = Resources(gpus=self.num_gpus_per_worker)

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the CLIPAestheticScorer model on the node."""
        CLIPAestheticScorer.download_weights_on_node(self.model_dir)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self.model = CLIPAestheticScorer(model_dir=self.model_dir)
        self.model.setup()
        self.frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=self.target_fps,
        ).to_str()
        if self.reduction == "mean":
            self.reduction_fn = np.mean
        elif self.reduction == "min":
            self.reduction_fn = np.min
        else:
            msg = f"Invalid reduction: {self.reduction}"
            raise ValueError(msg)

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        passed_clips = []
        for clip in video.clips:
            if not clip.buffer:
                logger.warning(f"Clip {clip.uuid} has no buffer.")
                clip.errors["buffer"] = "empty"
                clip.aesthetic_score = -1.0
            elif self.frame_extraction_signature not in clip.extracted_frames:
                clip.errors[f"frames-{self.frame_extraction_signature}"] = "missing"
                error_msg = (
                    f"Clip {clip.uuid} has buffer but no extracted frames for {self.frame_extraction_signature}"
                )
                logger.error(error_msg)
                clip.aesthetic_score = -1.0
            else:
                frames = clip.extracted_frames.pop(self.frame_extraction_signature)
                scores = self.model(frames).cpu().numpy()
                del frames
                clip.aesthetic_score = float(self.reduction_fn(scores))

            # Filtering
            if clip.aesthetic_score < self.score_threshold:
                video.filtered_clips.append(clip)
                video.clip_stats.num_filtered_by_aesthetic += 1
                if self.verbose:
                    logger.info(
                        f"Clip {clip.uuid} has aesthetic score {clip.aesthetic_score:.3f} below threshold "
                        f"{self.score_threshold}, skipped.",
                    )
            else:
                passed_clips.append(clip)
                if self.verbose:
                    logger.info(
                        f"Clip {clip.uuid} has aesthetic score {clip.aesthetic_score:.3f} above threshold "
                        f"{self.score_threshold}, kept.",
                    )

        video.clips = passed_clips
        if self.verbose:
            logger.info(
                f"Video {video.input_video} chunk-{video.clip_chunk_index} has "
                f"{len(video.clips)}/{len(video.filtered_clips)} clips "
                "passed/filtered",
            )

        # @aot TODO: free memory periodically
        return task
