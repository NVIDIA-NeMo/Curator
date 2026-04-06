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

"""
This module contains a simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
"""

import asyncio
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass
class OmniLLMRequestStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for requesting LLM with messages and producing the predicted text in form of a DocumentBatch.

    Example input messages:
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}},
        {"type": "text", "text": "What can you see and hear? Answer in one sentence."}
    ]}
    ]
    Example output predicted text:
    predicted_text = "I can see a car and hear a cough."
    It drops the 'messages' column and adds the 'predicted_text' column after processing.

    """

    client: AsyncLLMClient | LLMClient
    model_name: str
    generation_config: GenerationConfig | None = None
    name: str = "OmniLLMRequestStage"
    fields_to_drop: list[str] = field(default_factory=lambda: ["messages"])
    predicted_text_key: str = "predicted_text"

    def __post_init__(self) -> None:
        self.is_async_client = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.client.setup()

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        """
        Process the input messages and produce the predicted text.
        Drop the 'messages' column and add the 'predicted_text' column after processing.
        """
        input_df = input_batch.to_pandas().copy()
        responses = self._process_async(input_df) if self.is_async_client else self._process_sync(input_df)
        input_df[self.predicted_text_key] = responses
        input_batch.data = input_df.drop(columns=self.fields_to_drop)
        return input_batch

    def _process_llm_response(self, response: list[str]) -> str:
        """Process a single response from the LLM."""
        # Extract only the generated text content (first element of the response list)
        generated_text = response[0] if response else ""

        # Some models add ** bolding for the generated text
        if "*" in generated_text:
            generated_text = generated_text.replace("*", "")

        return generated_text

    def _process_sync(self, input_df: pd.DataFrame) -> list[str]:
        """Process samples using synchronous client (sequential)."""
        batch_size = len(input_df)
        responses = []
        for i in range(batch_size):
            logger.info(f"Generating sample {i + 1}/{batch_size} (sync)...")
            messages = input_df.iloc[i]["messages"]
            response = self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            generated_text = self._process_llm_response(response)
            responses.append(generated_text)
        return responses

    def _process_async(self, input_df: pd.DataFrame) -> list[str]:
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
            return asyncio.run(self._generate_responses_async(input_df))

        # If we get here, there's already a loop running
        # This is an edge case (e.g., Ray async actors), but we can handle it
        # by running in a new thread with its own loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._generate_responses_async(input_df))
            return future.result()

    async def _generate_responses_async(self, input_df: pd.DataFrame) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""
        batch_size = len(input_df)

        async def generate_single_response(_i: int) -> str:
            messages = input_df.iloc[_i]["messages"]
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            return self._process_llm_response(response)

        tasks = [generate_single_response(i) for i in range(batch_size)]
        return await asyncio.gather(*tasks)
