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

import asyncio
import base64
import io
from dataclasses import dataclass

import numpy as np
from loguru import logger
from PIL import Image

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.services.model_client import (
    AsyncLLMClient,
    GenerationConfig,
    LLMClient,
)
from ray_curator.tasks import ImageBatch, ImageObject


@dataclass
class VLMCaptioningStage(ProcessingStage[ImageBatch, ImageBatch]):
    """
    A stage for generating image captions using VLM (Vision Language Model).
    Supports both async and sync processing based on the client type.
    """

    client: AsyncLLMClient | LLMClient
    model_name: str
    prompt: str = "Describe this image in detail."
    max_tokens: int = 150
    temperature: float = 0.7
    verbose: bool = False
    fail_on_error: bool = True
    _name: str = "vlm_captioning"

    def __post_init__(self):
        self.is_async_client = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["caption"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Setup the VLM client."""
        self.client.setup()
        if self.verbose:
            client_type = "async" if self.is_async_client else "sync"
            logger.info(
                "VLM captioning stage setup complete. Using %s client.",
                client_type,
            )

    def _image_to_base64(self, image_data: np.ndarray) -> str:
        """Convert numpy image array to base64 string."""
        # Enforce expected dtype
        if image_data.dtype != np.uint8:
            msg = f"Expected uint8 image array, got {image_data.dtype}"
            if self.fail_on_error:
                if self.verbose:
                    logger.error(msg)
                raise ValueError(msg)
            if self.verbose:
                logger.warning("%s; coercing to uint8", msg)
            image_data = image_data.astype(np.uint8)

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_data)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_base64}"

    def _create_rest_api_messages(self, image_data: np.ndarray, prompt: str) -> list[dict]:
        """Create messages for OpenAI-compatible REST API calls."""
        image_base64 = self._image_to_base64(image_data)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64,
                        },
                    },
                ],
            }
        ]

    def _generate_caption_sync(self, image_data: np.ndarray) -> str:
        """Generate caption for a single image synchronously."""
        messages = self._create_rest_api_messages(image_data, self.prompt)
        response = self.client.query_model(
            model=self.model_name,
            messages=messages,
            generation_config=GenerationConfig(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        return response[0] if response else ""

    async def _generate_caption_async(self, image_data: np.ndarray) -> str:
        """Generate caption for a single image asynchronously."""
        messages = self._create_rest_api_messages(image_data, self.prompt)
        response = await self.client.query_model(
            model=self.model_name,
            messages=messages,
            generation_config=GenerationConfig(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        return response[0] if response else ""

    def _process_sync(self, image_batch: ImageBatch) -> ImageBatch:
        """Process image batch synchronously."""
        if self.verbose:
            logger.info(
                "Processing %d images synchronously",
                len(image_batch.data),
            )

        for image_obj in image_batch.data:
            if image_obj.image_data is not None:
                caption = self._generate_caption_sync(image_obj.image_data)
                image_obj.metadata["caption"] = caption

                if self.verbose:
                    logger.debug(
                        "Generated caption for %s: %s...",
                        image_obj.image_id,
                        caption[:100],
                    )
            else:
                if self.verbose:
                    logger.warning(
                        "No image data found for %s",
                        image_obj.image_id,
                    )
                image_obj.metadata["caption"] = ""

        return image_batch

    async def _process_async(self, image_batch: ImageBatch) -> ImageBatch:
        """Process image batch asynchronously."""
        if self.verbose:
            logger.info(
                "Processing %d images asynchronously",
                len(image_batch.data),
            )

        tasks, valid_indices = self._create_async_tasks(image_batch)

        if tasks:
            captions = await asyncio.gather(*tasks, return_exceptions=True)
            self._handle_async_results(image_batch, captions, valid_indices)

        return image_batch

    def _create_async_tasks(self, image_batch: ImageBatch) -> tuple[list, list[int]]:
        """Create async tasks for images with valid data."""
        tasks = []
        valid_indices = []

        for i, image_obj in enumerate(image_batch.data):
            if image_obj.image_data is not None:
                task = self._generate_caption_async(image_obj.image_data)
                tasks.append(task)
                valid_indices.append(i)
            else:
                self._handle_missing_image_data(image_obj)

        return tasks, valid_indices

    def _handle_missing_image_data(self, image_obj: ImageObject) -> None:
        """Handle cases where image data is missing."""
        if self.verbose:
            logger.warning(
                "No image data found for %s",
                image_obj.image_id,
            )
        image_obj.metadata["caption"] = ""

    def _handle_async_results(
        self,
        image_batch: ImageBatch,
        captions: list,
        valid_indices: list[int],
    ) -> None:
        """Handle results from async caption generation."""
        has_exceptions = any(isinstance(c, Exception) for c in captions)
        if has_exceptions and self.fail_on_error:
            first_exc = next(c for c in captions if isinstance(c, Exception))
            if self.verbose:
                logger.error(
                    "Caption generation failed for at least one image: %s",
                    first_exc,
                )
            raise first_exc

        # Assign captions back to image objects
        for idx, caption in zip(valid_indices, captions, strict=False):
            if isinstance(caption, Exception):
                self._handle_caption_error(image_batch, idx, caption)
            else:
                self._handle_caption_success(image_batch, idx, caption)

    def _handle_caption_error(self, image_batch: ImageBatch, idx: int, caption: Exception) -> None:
        """Handle caption generation errors."""
        if self.verbose:
            logger.error(
                "Error generating caption for %s: %s",
                image_batch.data[idx].image_id,
                caption,
            )
        image_batch.data[idx].metadata["caption"] = ""

    def _handle_caption_success(self, image_batch: ImageBatch, idx: int, caption: str) -> None:
        """Handle successful caption generation."""
        image_batch.data[idx].metadata["caption"] = caption

        if self.verbose:
            logger.debug(
                "Generated caption for %s: %s...",
                image_batch.data[idx].image_id,
                caption[:100],
            )

    def process(self, image_batch: ImageBatch) -> ImageBatch:
        """Process an ImageBatch to generate captions."""
        if self.is_async_client:
            try:
                # If there's an active loop, avoid asyncio.run
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._process_async(image_batch))
            else:
                if self.verbose:
                    logger.warning("Event loop already running; falling back to sync processing.")
                return self._process_sync(image_batch)
        else:
            return self._process_sync(image_batch)
