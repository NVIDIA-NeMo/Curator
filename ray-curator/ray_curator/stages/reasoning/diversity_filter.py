import pandas as pd

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.services.model_client import LLMClient
from ray_curator.tasks import DocumentBatch
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.reasoning.prompts import DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE


class LLMBasedDomainClassifier(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for filtering reasoning traces based on correctness. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    def __init__(self, prompt: str, client: LLMClient, model_name: str, domains_file_path: str, input_problem_field: str, output_field: str):
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

        def process_llm_response(response: list[str]) -> str:
            processed_response = response[0].strip().split('\n')[-1].strip()
            # assert processed_response in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"], "Response must contain a two-digit code"
            return processed_response

        def process_llm_prompt(sample: dict) -> str:
            processed_prompt = self.prompt.format(question=sample[self.input_problem_field], domains_prompt=self.domains_prompt)
            return processed_prompt

        def generate_response(row: pd.Series) -> str:
            prompt = process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]
            response = self.client.query_model(model=self.model_name, messages=messages)
            return process_llm_response(response)
        
        df[self.output_field] = df.apply(generate_response, axis=1)

        # DEBUGGING
        print(f"[ray_curator/stages/reasoning/diversity_filter.py - LLMBasedDomainClassifier] Number of rows in df: {len(df)}")

        return DocumentBatch(data=df, dataset_name="domain_classification_data", task_id=1)


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

    def outputs(self) -> [list[str]]:
        return ["data"]

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
        print(f"[ray_curator/stages/reasoning/diversity_filter.py - DiversitySampler] Number of rows in df: {len(df)}")

        return DocumentBatch(data=df, dataset_name="diversity_sampling_data", task_id=1)
