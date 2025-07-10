import pandas as pd

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.services.model_client import LLMClient
from ray_curator.tasks import DocumentBatch
from ray_curator.tasks import _EmptyTask
from ray_curator.stages.reasoning.prompts import DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE

class ReasoningTracesSyntheticStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for generating synthetic reasoning traces. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    def __init__(self, prompt: str, client: LLMClient, model_name: str, input_problem_field: str, output_field: str):
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE
        self.client = client
        self.model_name = model_name
        self.input_problem_field = input_problem_field
        self.output_field = output_field

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

        def process_llm_response(response: list[str]) -> str:
            processed_response = response[0]
            return processed_response

        def process_llm_prompt(sample: dict) -> str:
            processed_prompt = self.prompt.format(problem=sample[self.input_problem_field])
            return processed_prompt

        def generate_response(row: pd.Series) -> str:
            prompt = process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]
            response = self.client.query_model(model=self.model_name, messages=messages)
            return process_llm_response(response)
        
        df[self.output_field] = df.apply(generate_response, axis=1)

        # DEBUGGING
        print(f"[ray_curator/stages/reasoning/reasoning_traces_synthetic.py - ReasoningTracesSyntheticStage] Number of rows in df: {len(df)}")

        return DocumentBatch(data=df, dataset_name="reasoning_traces_synthetic_data", task_id=1)