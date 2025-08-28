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

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.internvideo2_mm import InternVideo2MultiModality
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Video, VideoTask
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FrameExtractionSignature, extract_frames


@dataclass
class InternVideo2FrameCreationStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for creating InternVideo2 input frames from video clips.

    This class processes video clips through a series of steps including frame extraction,
    model initialization, and input frame creation.
    """

    target_fps: float = 2.0
    verbose: bool = False
    model_dir: str = "InternVideo2"
    _name: str = "internvideo2_embedding_frame_creation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        # utils_only set to true to skip initializing the actual model
        self._model: InternVideo2MultiModality = InternVideo2MultiModality(model_dir=self.model_dir, utils_only=True)
        self._model.setup()
        self._extraction_policy = FrameExtractionPolicy.sequence
        self._frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=self.target_fps,
        ).to_str()

    def process(self, task: VideoTask) -> VideoTask:
        max_fps: int = 20
        video: Video = task.data

        for clip in video.clips:
            # check if buffer is empty
            if clip.buffer is None:
                clip.errors["buffer"] = "empty"
                continue
            # check if frames are extracted
            if self._frame_extraction_signature not in clip.extracted_frames:
                clip.errors[f"frames-{self._frame_extraction_signature}"] = "missing"
                logger.error(f"Clip {clip.uuid} has buffer but no extracted frames for ???")
                continue

            frames = clip.extracted_frames[self._frame_extraction_signature]
            # check if we need re-extract
            target_num_frames = self._model.get_target_num_frames()
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
            # create input frames for InternVideo2 model
            clip.intern_video_2_frames = self._model.formulate_input_frames(list(frames))

            # clear extracted frames to save memory
            clip.extracted_frames.clear()

        return task


@dataclass
class InternVideo2EmbeddingStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage for generating embeddings from InternVideo2 input frames.

    This class processes video clips through a series of steps including embedding generation,
    text verification, and memory management.
    """

    num_gpus_per_worker: float = 1.0
    texts_to_verify: list[str] | None = None
    verbose: bool = False
    gpu_memory_gb: float = 10.0
    model_dir: str = "InternVideo2"
    _name: str = "internvideo2_embedding"

    def __post_init__(self) -> None:
        self._resources = Resources(gpu_memory_gb=self.gpu_memory_gb)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self._model: InternVideo2MultiModality = InternVideo2MultiModality(model_dir=self.model_dir)
        self._model.setup()
        if self.verbose:
            logger.info("InternVideo2 model setup completed.")

    def process(self, task: VideoTask) -> VideoTask:
        video: Video = task.data
        for clip in video.clips:
            if clip.intern_video_2_frames is None:
                clip.errors["iv2_frames"] = "empty"
                continue
            # generate embeddings
            embedding = self._model.encode_video_frames(clip.intern_video_2_frames)
            if embedding.numel() == 0:
                logger.error(f"Unable to compute internvideo embedding for clip={clip.uuid}")
                clip.errors["iv2_embedding"] = "failed"
            else:
                clip.intern_video_2_embedding = embedding.cpu().numpy()

            if self.texts_to_verify is not None:
                text_embeddings = [self._model.get_text_embedding(x) for x in self._texts_to_verify]
                probs, idxs = self._model.evaluate(embedding, text_embeddings)
                clip.intern_video_2_text_match = (
                    self._texts_to_verify[idxs[0]],
                    probs[0],
                )

            # clear frames to save memory
            clip.intern_video_2_frames = None

            if self.verbose:
                logger.info(
                    f"InternVideo2 embedding generated for clip {clip.uuid}. Embedding shape: {clip.intern_video_2_embedding.shape}"
                )

        return task

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the InternVideo2 model on the node."""
        InternVideo2MultiModality.download_weights_on_node(self.model_dir)
