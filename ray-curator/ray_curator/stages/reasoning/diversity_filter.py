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

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.reasoning.prompts import DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE
from ray_curator.stages.services.model_client import AsyncLLMClient, LLMClient
from ray_curator.tasks import DocumentBatch

# Constants for magic values
MAX_RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 10


class LLMBasedDomainClassifier(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for classifying reasoning traces by domain. It automatically detects whether to use
    async (concurrent) or sync (sequential) processing based on the client type.
    """

    def __init__(  # noqa: PLR0913
        self,
        prompt: str,
        client: AsyncLLMClient | LLMClient,
        model_name: str,
        domains_file_path: str,
        input_problem_field: str,
        output_field: str,
    ):
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE
        self.client = client
        self.model_name = model_name
        self.domains = pd.read_json(domains_file_path, dtype=str)
        self.domains_prompt = "\n\n".join(self.domains["prompt"].tolist())
        self.input_problem_field = input_problem_field
        self.output_field = output_field
        self.is_async_client = isinstance(client, AsyncLLMClient)
        self.allowed_output_values = self.domains["code"].tolist()
        self._name = "LLMBasedDomainClassifier"

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

        logger.info(f"[Stage finished] - LLMBasedDomainClassifier - Number of samples - {len(df)}")

        return DocumentBatch(data=df, dataset_name="domain_classification_data", task_id=1)

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
            question=sample[self.input_problem_field],
            domains_prompt=self.domains_prompt,
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
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' (expected one of {self.allowed_output_values}). Retrying in 30 seconds... (Attempt {attempt + 1}/3)")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' after 3 attempts. Returning 'Failed to classify'.")

            # If after 3 attempts we still don't have a valid response
            return "Failed to classify"

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
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' (expected one of {self.allowed_output_values}). Retrying in 30 seconds... (Attempt {attempt + 1}/3)")
                    await asyncio.sleep(RETRY_SLEEP_SECONDS)
                else:
                    print(f"⚠️  WARNING: Invalid response '{processed_response}' after 3 attempts. Returning 'Failed to classify'.")

            # If after 3 attempts we still don't have a valid response
            return "Failed to classify"

        # Create tasks for all rows and execute concurrently
        tasks = [generate_response_async(row) for _, row in df.iterrows()]
        return await asyncio.gather(*tasks)


class DiversitySampler(ProcessingStage[DocumentBatch, DocumentBatch]):

    def __init__(self, sampling_size: int, input_problem_field: str, input_domain_field: str):
        self.sampling_size = sampling_size
        self.input_problem_field = input_problem_field
        self.input_domain_field = input_domain_field
        self._name = "DiversitySampler"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        pass

    def _sample_uniformly(self, df: pd.DataFrame) -> pd.DataFrame:

        # Get unique domains from the dataframe
        all_domains = df[self.input_domain_field].unique().tolist()

        # Group dataframe by domain
        domain_groups = {domain: group for domain, group in df.groupby(self.input_domain_field)}  # noqa: C416

        # Initialize selected indices
        selected_indices = set()

        # Progress bar for sampling
        pbar = tqdm(initial=len(selected_indices), total=self.sampling_size, desc="Sampling questions")

        # Create random number generator
        rng = np.random.default_rng()

        while len(selected_indices) < self.sampling_size:
            # Sample uniformly from all available domains
            random_domain = rng.choice(all_domains)

            # Get examples from the selected domain
            domain_examples = domain_groups[random_domain]

            # Get indices of examples in this domain that haven't been selected yet
            available_indices = domain_examples.index.difference(selected_indices)

            if len(available_indices) > 0:
                # Get available examples and their lengths
                available_examples = domain_examples.loc[available_indices]
                lengths = available_examples[self.input_problem_field].apply(len).to_numpy()

                # Calculate ranks based on length (longer = higher rank)
                ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))

                # Calculate length weights (exponential decay based on rank)
                length_weights = np.power(2.0, -ranks)
                length_weights = length_weights / length_weights.sum()

                # Select example based on length weights
                selected_idx = rng.choice(available_indices, p=length_weights)
                selected_indices.add(selected_idx)
                pbar.update(1)

            # Remove domain if no more examples available
            if len(domain_examples.index.difference(selected_indices)) == 0:
                all_domains.remove(random_domain)
                if len(all_domains) == 0:
                    break

        pbar.close()

        # Return sampled dataframe
        return df.loc[list(selected_indices)]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        df = self._sample_uniformly(df)
        logger.info(f"[Stage finished] - DiversitySampler - Number of samples - {len(df)}")
        return DocumentBatch(data=df, dataset_name="diversity_sampling_data", task_id=1)
