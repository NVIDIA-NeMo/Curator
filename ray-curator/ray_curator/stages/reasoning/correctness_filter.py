# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the specific language for the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time

import pandas as pd
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.reasoning.prompts import DEFAULT_GRADING_PROMPT_TEMPLATE
from ray_curator.stages.services.model_client import AsyncLLMClient, LLMClient
from ray_curator.tasks import DocumentBatch

# Constants for magic values
MAX_RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 10


class LLMBasedGrader(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for filtering reasoning traces based on correctness. It automatically detects whether to use
    async (concurrent) or sync (sequential) processing based on the client type.
    """

    def __init__(  # noqa: PLR0913
        self,
        prompt: str,
        client: AsyncLLMClient | LLMClient,
        model_name: str,
        input_problem_field: str,
        input_attempt_field: str,
        input_solution_field: str,
        output_field: str,
    ):
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = DEFAULT_GRADING_PROMPT_TEMPLATE
        self.client = client
        self.model_name = model_name
        self.input_problem_field = input_problem_field
        self.input_attempt_field = input_attempt_field
        self.input_solution_field = input_solution_field
        self.output_field = output_field
        self.is_async_client = isinstance(client, AsyncLLMClient)
        self.allowed_output_values = ["Yes", "No"]
        self._name = "LLMBasedCorrectnessFilter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_field]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.client.setup()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        responses = self._process_async(df) if self.is_async_client else self._process_sync(df)

        df[self.output_field] = responses

        logger.info(f"[Stage finished] - LLMBasedGrader - Number of samples - {len(df)}")

        return DocumentBatch(data=df, dataset_name="reasoning_traces_synthetic_data", task_id=1)

    def _process_llm_response(self, response: list[str]) -> str:
        """Process LLM response to extract the content."""
        processed_response = response[0].strip().split("\n")[-1].strip()
        # Some models add ** bolding for the generated text
        if "*" in processed_response:
            processed_response = processed_response.replace("*", "")
        return processed_response

    def _process_llm_prompt(self, sample: dict) -> str:
        """Process the input sample to create the LLM prompt."""
        return self.prompt.format(
            problem=sample[self.input_problem_field],
            attempt=sample[self.input_attempt_field],
            solution=sample[self.input_solution_field],
        )

    def _process_sync(self, df: pd.DataFrame) -> list[str]:
        """Process DataFrame using synchronous sequential processing."""
        def generate_response(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]

            for attempt in range(MAX_RETRY_ATTEMPTS):  # Try up to 3 times
                response = self.client.query_model(model=self.model_name, messages=messages)
                processed_response = self._process_llm_response(response)

                if processed_response in self.allowed_output_values:
                    return processed_response

                # If not the last attempt, wait 30 seconds before retrying
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' (expected 'Yes' or 'No'). Retrying in 30 seconds... (Attempt {attempt + 1}/3)")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' after 3 attempts. Returning 'Failed to grade'.")

            # If after 3 attempts we still don't have a valid response
            return "Failed to grade"

        # Sequential processing row by row
        return df.apply(generate_response, axis=1).tolist()

    def _process_async(self, df: pd.DataFrame) -> list[str]:
        """Process DataFrame using asynchronous concurrent processing."""
        return asyncio.run(self._generate_responses_async(df))

    async def _generate_responses_async(self, df: pd.DataFrame) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""
        async def generate_response_async(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]

            # logic to make sure generated response is in the allowed output values
            for attempt in range(MAX_RETRY_ATTEMPTS):  # Try up to 3 times
                response = await self.client.query_model(model=self.model_name, messages=messages)
                processed_response = self._process_llm_response(response)

                if processed_response in self.allowed_output_values:
                    return processed_response

                # If not the last attempt, wait 30 seconds before retrying
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' (expected 'Yes' or 'No'). Retrying in 30 seconds... (Attempt {attempt + 1}/3)")
                    await asyncio.sleep(RETRY_SLEEP_SECONDS)
                else:
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' after 3 attempts. Returning 'Failed to grade'.")

            # If after 3 attempts we still don't have a valid response
            return "Failed to grade"

        # Create tasks for all rows and execute concurrently
        tasks = [generate_response_async(row) for _, row in df.iterrows()]
        return await asyncio.gather(*tasks)


class LLMBasedCorrectnessFilter(DocumentFilter):

    def __init__(self):
        self._name = "llm_based_correctness_filter"

    def score_document(self, text: str) -> int:
        return 1 if text == "Yes" else 0

    def keep_document(self, score: int) -> bool:
        return score == 1
