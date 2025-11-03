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

"""
This module contains a simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
"""

import asyncio
import secrets
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, _EmptyTask


@dataclass
class QAMultilingualSyntheticStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """
    A simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    prompt: str
    languages: list[str]
    client: AsyncLLMClient | LLMClient
    model_name: str
    num_samples: int
    generation_config: GenerationConfig | None = None
    _name: str = "QAMultilingualSyntheticStage"

    def __post_init__(self) -> None:
        self.is_async_client = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.client.setup()

    def process(self, _: _EmptyTask) -> DocumentBatch:
        responses = self._process_async() if self.is_async_client else self._process_sync()

        return DocumentBatch(data=pd.DataFrame({"text": responses}), dataset_name="simple_synthetic_data", task_id=1)

    def _process_llm_response(self, response: list[str]) -> str:
        """Process a single response from the LLM."""
        # Extract only the generated text content (first element of the response list)
        generated_text = response[0] if response else ""

        # Some models add ** bolding for the generated text
        if "*" in generated_text:
            generated_text = generated_text.replace("*", "")

        return generated_text

    def _process_sync(self) -> list[str]:
        """Process samples using synchronous client (sequential)."""
        responses = []
        for i in range(self.num_samples):
            logger.info(f"Generating sample {i + 1}/{self.num_samples} (sync)...")
            language = secrets.choice(self.languages)
            prompt = self.prompt.format(language=language)
            response = self.client.query_model(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                generation_config=self.generation_config,
            )
            generated_text = self._process_llm_response(response)
            responses.append(generated_text)
        return responses

    def _process_async(self) -> list[str]:
        """Process samples using async client (concurrent).

        This method handles both cases:
        - Normal case: No event loop exists, creates one with asyncio.run()
        - Edge case: Called from async context, runs in separate thread
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running - this is the expected/normal case
            # Safe to use asyncio.run() which creates its own loop
            return asyncio.run(self._generate_responses_async())

        # If we get here, there's already a loop running
        # This is an edge case (e.g., Ray async actors), but we can handle it
        # by running in a new thread with its own loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._generate_responses_async())
            return future.result()

    async def _generate_responses_async(self) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""

        async def generate_single_response(_i: int) -> str:
            language = secrets.choice(self.languages)
            prompt = self.prompt.format(language=language)
            response = await self.client.query_model(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                generation_config=self.generation_config,
            )
            return self._process_llm_response(response)

        # Create tasks for all samples and execute concurrently
        tasks = [generate_single_response(i) for i in range(self.num_samples)]
        return await asyncio.gather(*tasks)
