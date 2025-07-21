"""
This module contains a simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
"""

import asyncio
import random
import pandas as pd
from typing import Union

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.services.model_client import AsyncLLMClient, LLMClient
from ray_curator.tasks import DocumentBatch, _EmptyTask


class QAMultilingualSyntheticStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """
    A simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    def __init__(self, prompt: str, languages: list[str], client: Union[AsyncLLMClient, LLMClient], model_name: str, num_samples: int):
        self.prompt = prompt
        self.languages = languages
        self.client = client
        self.num_samples = num_samples
        self.model_name = model_name
        self.is_async_client = isinstance(client, AsyncLLMClient)

    @property
    def name(self) -> str:
        return "QAMultilingualSyntheticStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.client.setup()

    def process(self, _: _EmptyTask) -> DocumentBatch:
        if self.is_async_client:
            responses = self._process_async()
        else:
            responses = self._process_sync()
        
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
            print(f"Generating sample {i+1}/{self.num_samples} (sync)...")
            language = random.choice(self.languages)
            prompt = self.prompt.format(language=language)
            response = self.client.query_model(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,  # Add randomness for variety
                seed=i,  # Different seed for each generation
            )
            generated_text = self._process_llm_response(response)
            responses.append(generated_text)
        return responses

    def _process_async(self) -> list[str]:
        """Process samples using async client (concurrent)."""
        return asyncio.run(self._generate_responses_async())

    async def _generate_responses_async(self) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""
        async def generate_single_response(i: int) -> str:
            language = random.choice(self.languages)
            prompt = self.prompt.format(language=language)
            response = await self.client.query_model(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._process_llm_response(response)
                
        # Create tasks for all samples and execute concurrently
        tasks = [generate_single_response(i) for i in range(self.num_samples)]
        return await asyncio.gather(*tasks)


class LanguageFilter(DocumentFilter):

    def __init__(self, languages: list[str]):
        self._name = "language_filter"
        self.languages = languages

    def score_document(self, text: str) -> float:
        return 1.0 if text.startswith(tuple(self.languages)) else 0.0

    def keep_document(self, score: float) -> bool:
        return score == 1.0
