import asyncio
import pandas as pd
from typing import Union

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.services.model_client import AsyncLLMClient, LLMClient
from ray_curator.tasks import DocumentBatch
from ray_curator.tasks import _EmptyTask
from ray_curator.stages.reasoning.prompts import DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE

class ReasoningTracesSyntheticStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for generating synthetic reasoning traces. It automatically detects whether to use 
    async (concurrent) or sync (sequential) processing based on the client type.
    """

    def __init__(self, prompt: str, client: Union[AsyncLLMClient, LLMClient], model_name: str, input_problem_field: str, output_field: str):
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

        if self.is_async_client:
            responses = self._process_async(df)
        else:
            responses = self._process_sync(df)
        
        df[self.output_field] = responses
        return DocumentBatch(data=df, dataset_name="reasoning_traces_synthetic_data", task_id=1)

    def _process_llm_response(self, response: list[str]) -> str:
        """Process LLM response to extract the content."""
        processed_response = response[0]
        return processed_response

    def _process_llm_prompt(self, sample: dict) -> str:
        """Process the input sample to create the LLM prompt."""
        processed_prompt = self.prompt.format(problem=sample[self.input_problem_field])
        return processed_prompt

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
        async def generate_response_async(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.query_model(model=self.model_name, messages=messages)
            return self._process_llm_response(response)
        
        async def generate_all_responses_async(dataframe: pd.DataFrame) -> list[str]:
            """Generate responses for all rows with controlled concurrency"""
            tasks = [generate_response_async(row) for _, row in dataframe.iterrows()]
            return await asyncio.gather(*tasks)
        
        # Run the async function and get all responses
        return asyncio.run(generate_all_responses_async(df))