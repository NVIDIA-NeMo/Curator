import pandas as pd

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.services.model_client import LLMClient
from ray_curator.tasks import DocumentBatch
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.reasoning.prompts import DEFAULT_GRADING_PROMPT_TEMPLATE

class LLMBasedGrader(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    A stage for filtering reasoning traces based on correctness. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    def __init__(self, prompt: str, client: LLMClient, model_name: str, input_problem_field: str, input_attempt_field: str, input_solution_field: str, output_field: str):
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


    @property
    def name(self) -> str:
        return "LLMBasedCorrectnessFilter"

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
            # assert processed_response in ["Yes", "No"], "Response must contain Yes or No"

            # DEBUGGING
            processed_response = response[0]

            return processed_response

        def process_llm_prompt(sample: dict) -> str:
            processed_prompt = self.prompt.format(problem=sample[self.input_problem_field], attempt=sample[self.input_attempt_field], solution=sample[self.input_solution_field])
            return processed_prompt

        def generate_response(row: pd.Series) -> str:
            prompt = process_llm_prompt(row)
            messages = [{"role": "user", "content": prompt}]
            response = self.client.query_model(model=self.model_name, messages=messages)
            return process_llm_response(response)
        
        df[self.output_field] = df.apply(generate_response, axis=1)

        # DEBUGGING
        print(f"[ray_curator/stages/reasoning/correctness_filter.py - LLMBasedCorrectnessFilter] Number of rows in df: {len(df)}")

        return DocumentBatch(data=df, dataset_name="reasoning_traces_synthetic_data", task_id=1)

class LLMBasedCorrectnessFilter(DocumentFilter):

    def __init__(self):
        self._name = "llm_based_correctness_filter"

    def score_document(self, text: str) -> float:
        # assert text in ["Yes", "No"], "Response must contain Yes or No"
        document_score = 1.0 if text == "Yes" else 0.0
        return document_score

    def keep_document(self, score: float) -> bool:
        return score == 1.0

