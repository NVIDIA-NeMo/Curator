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

import asyncio
import base64
import concurrent.futures
from abc import abstractmethod
from io import BytesIO
from typing import Any, TypeVar

from loguru import logger
from PIL import Image

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.image import ImageSampleTask, ImageTaskData

T = TypeVar("T", bound=ImageTaskData)


class SkipSample(Exception):  # noqa: N818
    """Exception to be raised in build_prompt to skip a sample."""


class ModelProcessingStage(ProcessingStage[ImageSampleTask[T], ImageSampleTask[T]]):
    """Base stage for batched VLM inference over an OpenAI-compatible client.

    Subclasses implement ``build_prompt`` and ``handle_response``; image loading,
    multimodal message assembly, and concurrent batch dispatch are handled here.
    The stage takes an :class:`AsyncLLMClient` (e.g. ``NVInferenceClient``) and
    calls its ``query_model`` — the client owns concurrency + retry, exactly like
    the other SDG stages (e.g. ``QAMultilingualSyntheticStage``).
    """

    name: str = "model_base_stage"
    resources: Resources = Resources(cpus=8.0)
    batch_size: int = 8
    multimodal: bool = True

    def __init__(  # noqa: PLR0913
        self,
        client: AsyncLLMClient,
        model_name: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 8,
    ) -> None:
        self.client = client
        self.model_name = model_name
        # NVInferenceClient always streams (reasoning models require it); no stream flag here.
        self.generation_config = GenerationConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        self.batch_size = batch_size

    def setup(self, _worker_metadata: dict | None = None) -> None:
        self.client.setup()

    @abstractmethod
    def build_prompt(self, task: ImageSampleTask[T]) -> str:
        """Build the text prompt for a task.

        Raises:
            SkipSample: skip without marking the task invalid.
            Exception: any other error sets is_valid=False and records the error.
        """
        ...

    @abstractmethod
    def handle_response(self, task: ImageSampleTask[T], response: str) -> ImageSampleTask[T]: ...

    def load_image(self, task: ImageSampleTask[T]) -> Image.Image:
        return Image.open(task.data.image_path)

    def process(self, task: ImageSampleTask[T]) -> ImageSampleTask[T]:
        msg = f"{self.name} does not support single-task processing; use process_batch"
        raise NotImplementedError(msg)

    @staticmethod
    def _encode_image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_messages(self, prompt: str, image: Image.Image | str | None) -> list[dict[str, Any]]:
        """Build the OpenAI-compatible ``messages`` payload for one prompt (+ optional image)."""
        content: list[dict[str, Any]] = []
        if image is not None:
            if isinstance(image, str):
                image_url = image
            else:
                image_url = f"data:image/png;base64,{self._encode_image_to_base64(image)}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _handle_response_one(self, tasks: list[ImageSampleTask[T]], idx: int, response: str) -> None:
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
        tasks: list[ImageSampleTask[T]],
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

    def process_batch(self, tasks: list[ImageSampleTask[T]]) -> list[ImageSampleTask[T]]:
        """Process a batch, skipping tasks already failed in previous stages."""
        if not tasks:
            return []

        valid_indices: list[int] = []
        messages_batch: list[list[dict[str, Any]]] = []

        for i, task in enumerate(tasks):
            if not task.data.is_valid:
                logger.debug(f"{self.name}: skipping invalid task {i}")
                continue

            try:
                image = self.load_image(task) if self.multimodal else None
                prompt = self.build_prompt(task)
                valid_indices.append(i)
                messages_batch.append(self._build_messages(prompt, image))
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
            responses = self._generate(messages_batch)
            self._dispatch_responses(tasks, valid_indices, responses)
            logger.info(f"{self.name}: processed batch of {len(valid_indices)} items")

        except Exception as e:  # noqa: BLE001
            logger.error(f"{self.name}: batch processing error: {e}")
            for task in tasks:
                task.data.error = f"{self.name}: {e}"
                task.data.is_valid = False

        return tasks

    def _generate(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        """Run the batch concurrently through the async client.

        Uses ``asyncio.run`` normally; if a loop is already running (e.g. inside a
        Ray async actor) we run in a dedicated thread with its own loop, mirroring
        ``QAMultilingualSyntheticStage``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._agenerate(messages_batch))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, self._agenerate(messages_batch)).result()

    async def _agenerate(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        """Fan the batch out via ``client.query_model`` (concurrency + retry live in the client)."""

        async def _one(idx: int, messages: list[dict[str, Any]]) -> str:
            try:
                out = await self.client.query_model(
                    messages=messages, model=self.model_name, generation_config=self.generation_config
                )
                return out[0] if out else ""
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self.name}: error generating response for prompt {idx}: {e}")
                return ""

        return list(await asyncio.gather(*(_one(i, m) for i, m in enumerate(messages_batch))))
