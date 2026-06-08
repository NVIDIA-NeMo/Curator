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
import concurrent.futures
from dataclasses import dataclass

import pandas as pd

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass
class BaseSyntheticStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    system_prompt: str = None
    prompt: str = None
    input_field: str = None
    output_field: str = None
    client: AsyncLLMClient | LLMClient = None
    model_name: str = None
    generation_config: GenerationConfig | None = None
    name: str = "NemotronCCBaseStage"

    def __post_init__(self) -> None:
        self.is_async_client = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.output_field is None:
            msg = "output_field must be set before calling outputs()."
            raise ValueError(msg)
        return ["data"], [self.output_field]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self.client is not None:
            self.client.setup()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        responses = self._process_async(df) if self.is_async_client else self._process_sync(df)

        df[self.output_field] = responses
        return DocumentBatch(
            data=df,
            dataset_name=batch.dataset_name,
            task_id=f"{batch.task_id}_{self.name}",
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _process_llm_prompt(self, sample: dict) -> str:
        """Process the input sample to create the LLM prompt."""
        if self.input_field not in sample:
            msg = f"Expected input field '{self.input_field}' in sample."
            raise KeyError(msg)
        return self.prompt.format(document=sample[self.input_field])

    def _process_llm_response(self, response: list[str]) -> str:
        """Process a single response from the LLM."""
        # Extract only the generated text content (first element of the response list)
        return response[0] if response else ""

    def _process_sync(self, df: pd.DataFrame) -> list[str]:
        """Process DataFrame using synchronous sequential processing."""
        def generate_response(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]
            response = self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            return self._process_llm_response(response)

        # Sequential processing row by row
        return df.apply(generate_response, axis=1).tolist()

    def _process_async(self, df: pd.DataFrame) -> list[str]:
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
            return asyncio.run(self._generate_responses_async(df))

        # If we get here, there's already a loop running
        # This is an edge case (e.g., Ray async actors), but we can handle it
        # by running in a new thread with its own loop

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._generate_responses_async(df))
            return future.result()

    async def _generate_responses_async(self, df: pd.DataFrame) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""

        async def generate_response_async(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            return self._process_llm_response(response)

        # Create tasks for all rows and execute concurrently
        tasks = [generate_response_async(row) for _, row in df.iterrows()]
        return await asyncio.gather(*tasks)
