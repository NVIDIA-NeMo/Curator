# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from abc import abstractmethod
from typing import Generic, TypeVar

from loguru import logger
from PIL import Image

from nemo_curator.models.omni.base import NVInferenceModel
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.image import SingleDataTask

T = TypeVar("T")


class SkipSample(Exception):  # noqa: N818
    """Exception to be raised in build_prompt to skip a sample."""


class ModelProcessingStage(ProcessingStage[SingleDataTask[T], SingleDataTask[T]], Generic[T]):
    """Base stage for cloud-API VLM inference.

    Concrete subclasses implement ``build_prompt`` and ``handle_response``;
    image loading and batch dispatch are handled here. Sampling kwargs
    (temperature, top_p, max_tokens, etc.) live on the model instance —
    set them at construction.

    GPU placement is delegated to the executor (Xenna/Ray) via the
    ``resources`` declaration; the stage itself never touches CUDA.
    """

    name: str = "model_base_stage"
    resources: Resources = Resources(cpus=8.0)
    batch_size: int = 8
    multimodal: bool = True

    def __init__(
        self,
        model: NVInferenceModel,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.batch_size = batch_size

    def setup(self, _worker_metadata: dict | None = None) -> None:
        """Initialize the API client (idempotent)."""
        self.model.setup()

    @abstractmethod
    def build_prompt(self, task: SingleDataTask[T]) -> str:
        """Build the text prompt for a task.

        Raises:
            SkipSample: skip without marking the task invalid.
            Exception: any other error sets is_valid=False and records the error.
        """
        ...

    @abstractmethod
    def handle_response(self, task: SingleDataTask[T], response: str) -> SingleDataTask[T]:
        """Apply the model response to the task."""
        ...

    def load_image(self, task: SingleDataTask[T]) -> Image.Image:
        return Image.open(task.data.image_path)

    def process(self, task: SingleDataTask[T]) -> SingleDataTask[T]:
        return self.process_batch([task])[0]

    def _handle_response_one(self, tasks: list[SingleDataTask[T]], idx: int, response: str) -> None:
        """Call handle_response for one task, catching and logging errors."""
        try:
            self.handle_response(tasks[idx], response)
        except SkipSample:
            logger.debug(f"{self.name}: skipping sample {idx}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.name}: error handling response for task {idx}: {e}")
            tasks[idx].data.error = f"{self.name}: {e}"
            tasks[idx].data.is_valid = False

    def _dispatch_responses(
        self,
        tasks: list[SingleDataTask[T]],
        valid_indices: list[int],
        responses: list[str],
    ) -> None:
        """Hand each response to its task after validating the length contract.

        Raising on a length mismatch *before* any per-task write keeps the
        outer batch-error handler from clobbering tasks that have already been
        successfully scored — which a strict=True zip would not do, since it
        only raises after the shorter sequence has been consumed.
        """
        if len(responses) != len(valid_indices):
            msg = f"model returned {len(responses)} responses for {len(valid_indices)} prompts"
            raise RuntimeError(msg)
        for idx, response in zip(valid_indices, responses, strict=False):
            self._handle_response_one(tasks, idx, response)

    def process_batch(self, tasks: list[SingleDataTask[T]]) -> list[SingleDataTask[T]]:
        """Process a batch, skipping tasks already failed in previous stages."""
        if not tasks:
            return []

        valid_indices: list[int] = []
        prompts: list[str] = []
        images: list[Image.Image] = []

        for i, task in enumerate(tasks):
            if not task.data.is_valid:
                logger.debug(f"{self.name}: skipping invalid task {i}")
                continue

            try:
                image = self.load_image(task) if self.multimodal else None
                prompt = self.build_prompt(task)
                valid_indices.append(i)
                prompts.append(prompt)
                images.append(image)
            except SkipSample:
                logger.debug(f"{self.name}: skipping sample {i}")
                continue
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self.name}: error preparing task {i}: {e}")
                task.data.error = f"{self.name}: {e}"
                task.data.is_valid = False

        if not valid_indices:
            return tasks

        try:
            responses = self.model.generate(prompts, images if self.multimodal else None)
            self._dispatch_responses(tasks, valid_indices, responses)
            logger.info(f"{self.name}: processed batch of {len(valid_indices)} items")

        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.name}: batch processing error: {e}")
            for task in tasks:
                task.data.error = f"{self.name}: {e}"
                task.data.is_valid = False

        return tasks

    def teardown(self) -> None:
        """Release the API client."""
        self.model.unload()
