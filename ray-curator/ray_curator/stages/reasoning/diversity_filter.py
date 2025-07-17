import asyncio
import pandas as pd
from typing import Union

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.reasoning.prompts import DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE
from ray_curator.stages.services.model_client import AsyncLLMClient, LLMClient
from ray_curator.tasks import DocumentBatch


class LLMBasedDomainClassifier(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for classifying reasoning traces by domain. It automatically detects whether to use 
    async (concurrent) or sync (sequential) processing based on the client type.
    """

    def __init__(
        self,
        prompt: str,
        client: Union[AsyncLLMClient, LLMClient],
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
        self.domains = pd.read_json(domains_file_path)
        self.domains_prompt = "\n\n".join(self.domains["prompt"].tolist())
        self.input_problem_field = input_problem_field
        self.output_field = output_field
        self.is_async_client = isinstance(client, AsyncLLMClient)

    @property
    def name(self) -> str:
        return "LLMBasedDomainClassifier"

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

        # DEBUGGING
        print(f"[Stage finished] - LLMBasedDomainClassifier - Number of samples - {len(df)}")

        return DocumentBatch(data=df, dataset_name="domain_classification_data", task_id=1)

    def _process_llm_response(self, response: list[str]) -> str:
        """Process LLM response to extract the content."""
        processed_response = response[0].strip().split("\n")[-1].strip()
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


class DiversitySampler(ProcessingStage[DocumentBatch, DocumentBatch]):

    def __init__(self, sampling_size: int, input_problem_field: str, input_domain_field: str):
        self.sampling_size = sampling_size
        self.input_problem_field = input_problem_field
        self.input_domain_field = input_domain_field

    @property
    def name(self) -> str:
        return "DiversitySampler"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        pass

    def _sample_uniformly(self, df: pd.DataFrame) -> pd.DataFrame:
        
        import numpy as np
        from tqdm import tqdm
        
        # Get unique domains from the dataframe
        all_domains = df[self.input_domain_field].unique().tolist()
        
        # Group dataframe by domain
        domain_groups = {domain: group for domain, group in df.groupby(self.input_domain_field)}
        
        # Initialize selected indices
        selected_indices = set()
        
        # Progress bar for sampling
        pbar = tqdm(initial=len(selected_indices), total=self.sampling_size, desc="Sampling questions")
        
        while len(selected_indices) < self.sampling_size:
            # Sample uniformly from all available domains
            random_domain = np.random.choice(all_domains)
            
            # Get examples from the selected domain
            domain_examples = domain_groups[random_domain]
            
            # Get indices of examples in this domain that haven't been selected yet
            available_indices = domain_examples.index.difference(selected_indices)
            
            if len(available_indices) > 0:
                # Get available examples and their lengths
                available_examples = domain_examples.loc[available_indices]
                lengths = available_examples[self.input_problem_field].apply(len).values
                
                # Calculate ranks based on length (longer = higher rank)
                ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))
                
                # Calculate length weights (exponential decay based on rank)
                length_weights = np.power(2.0, -ranks)
                length_weights = length_weights / length_weights.sum()
                
                # Select example based on length weights
                selected_idx = np.random.choice(available_indices, p=length_weights)
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

        # DEBUGGING
        print(f"[Stage finished] - DiversitySampler - Number of samples - {len(df)}")

        return DocumentBatch(data=df, dataset_name="diversity_sampling_data", task_id=1)