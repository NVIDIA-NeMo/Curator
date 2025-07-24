import asyncio

import pandas as pd

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.reasoning.prompts import DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE
from ray_curator.stages.services.model_client import AsyncLLMClient, LLMClient
from ray_curator.tasks import DocumentBatch


class ReasoningTracesSyntheticStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for generating synthetic reasoning traces. It automatically detects whether to use
    async (concurrent) or sync (sequential) processing based on the client type.
    """

    def __init__(self, prompt: str, client: AsyncLLMClient | LLMClient, model_name: str, input_problem_field: str, output_field: str):
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE
        self.client = client
        self.model_name = model_name
        self.input_problem_field = input_problem_field
        self.output_field = output_field
        self.is_async_client = isinstance(client, AsyncLLMClient)

    @property
    def name(self) -> str:
        return "ReasoningTracesSyntheticStage"

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
        return DocumentBatch(data=df, dataset_name="reasoning_traces_synthetic_data", task_id=1)

    def _process_llm_response(self, response: list[str]) -> str:
        """Process LLM response to extract the content."""
        return response[0]

    def _process_llm_prompt(self, sample: dict) -> str:
        """Process the input sample to create the LLM prompt."""
        return self.prompt.format(problem=sample[self.input_problem_field])

    def _process_sync(self, df: pd.DataFrame) -> list[str]:
        """Process DataFrame using synchronous sequential processing."""
        def generate_response(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]
            response = self.client.query_model(model=self.model_name, messages=messages)
            return self._process_llm_response(response)

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
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages
            )
            return self._process_llm_response(response)

        # Create tasks for all rows and execute concurrently
        tasks = [generate_response_async(row) for _, row in df.iterrows()]
        return await asyncio.gather(*tasks)
