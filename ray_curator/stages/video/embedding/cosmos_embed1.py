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
from typing import Literal

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.cosmos_embed1 import CosmosEmbed1
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import VideoTask
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature, extract_frames


@dataclass
class CosmosEmbed1FrameCreationStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for creating Cosmos-Embed1 input frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    model initialization, and input frame creation.
    """

    model_dir: str = "models/cosmos_embed1"
    variant: Literal["224p", "336p", "448p"] = "336p"
    target_fps: float = 2.0
    verbose: bool = False
    num_cpus: int = 3
    _name: str = "cosmos_embed1"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips.cosmos_embed1_frames"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self.frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=self.target_fps,
        ).to_str()
        self.model = CosmosEmbed1(variant=self.variant, utils_only=True, model_dir=self.model_dir)
        self.model.setup()

    def process(self, task: VideoTask) -> VideoTask:
        max_fps: int = 20
        video = task.data
        for clip in video.clips:
            if clip.buffer is None:
                clip.errors["buffer"] = "empty"
                continue
            if self.frame_extraction_signature not in clip.extracted_frames:
                clip.errors[f"frames-{self.frame_extraction_signature}"] = "missing"
                logger.error(
                    f"Clip {clip.uuid} has buffer but no extracted frames for {self.frame_extraction_signature}"
                )
                continue

            frames = clip.extracted_frames[self.frame_extraction_signature]
            # check if we need re-extract
            target_num_frames = self.model.get_target_num_frames()
            regen_fps = self.target_fps
            while frames.shape[0] < target_num_frames:
                regen_fps *= 2
                if regen_fps > max_fps:
                    logger.error(f"Clip {clip.uuid} is too short to extract enough frames.")
                    break
                if self.verbose:
                    logger.warning(
                        f"Clip {clip.uuid} has <{target_num_frames} frames. "
                        f"Re-extracting with higher target_fps={regen_fps}. "
                        f"Current # frames={frames.shape[0]}.",
                    )
                with io.BytesIO(clip.buffer) as fp:
                    frames = extract_frames(
                        fp,
                        extraction_policy=FrameExtractionPolicy.sequence,
                        sample_rate_fps=regen_fps,
                    )
            # create input frames for Cosmos-Embed1 model
            clip.cosmos_embed1_frames = self.model.formulate_input_frames(list(frames))
            # done with extracted_frames
            clip.extracted_frames.clear()

        if self.verbose:
            logger.info(
                f"Cosmos-Embed1 frame creation stage completed for {len(video.clips)} clips in {video.input_video}"
            )
        return task

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the CosmosEmbed1 model on the node."""
        CosmosEmbed1.download_processor_config_on_node(self.model_dir, self.variant)


@dataclass
class CosmosEmbed1EmbeddingStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for embedding Cosmos-Embed1 frames into a vector space.

    This class processes video clips through a series of steps including frame extraction,
    model initialization, and input frame creation.
    """

    model_dir: str = "models/cosmos_embed1"
    variant: Literal["224p", "336p", "448p"] = "336p"
    texts_to_verify: list[str] | None = None
    gpu_memory_gb: int = 20
    verbose: bool = False
    _name: str = "cosmos_embed1_embedding"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips.cosmos_embed1_embeddings", "clips.cosmos_embed1_embedding"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self.model = CosmosEmbed1(variant=self.variant, utils_only=False, model_dir=self.model_dir)
        self.model.setup()

    def __post_init__(self) -> None:
        self._resources = Resources(gpu_memory_gb=self.gpu_memory_gb)

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data

        for clip in video.clips:
            if clip.cosmos_embed1_frames is None:
                clip.errors["cosmos_embed1_frames"] = "empty"
                continue
            embedding = self.model.encode_video_frames(clip.cosmos_embed1_frames)
            if embedding.numel() == 0:
                logger.error(f"Unable to compute cosmos-embed1 embedding for clip={clip.uuid}")
                clip.errors["cosmos_embed1_embedding"] = "failed"
            else:
                clip.cosmos_embed1_embedding = embedding.cpu().numpy()
            if self.texts_to_verify:
                text_embeddings = [self.model.get_text_embedding(x) for x in self.texts_to_verify]
                probs, idxs = self.model.evaluate(embedding, text_embeddings)
                clip.cosmos_embed1_text_match = (
                    self.texts_to_verify[idxs[0]],
                    probs[0],
                )
            # done with cosmos_embed1_frames
            clip.cosmos_embed1_frames = None

        # TODO: free memory periodically
        if self.verbose:
            logger.info(f"Cosmos-Embed1 embedding stage completed for {len(video.clips)} clips in {video.input_video}")
        return task

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the CosmosEmbed1 model on the node."""
        CosmosEmbed1.download_weights_on_node(self.model_dir, self.variant)
