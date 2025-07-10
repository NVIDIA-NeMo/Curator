import os
import argparse
import logging

# DEBUGGING
# Configure loguru (which is what ray_curator actually uses)
from loguru import logger
logger.disable("ray_curator")  # Disable all loguru logs from ray_curator
logger.disable("cosmos_xenna")  # Disable all loguru logs from cosmos_xenna

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.modules.filter import ScoreFilter
from ray_curator.stages.services.openai_client import OpenAIClient
from ray_curator.stages.reasoning.reasoning_traces_synthetic import ReasoningTracesSyntheticStage
from ray_curator.stages.reasoning.correctness_filter import LLMBasedGrader, LLMBasedCorrectnessFilter
from ray_curator.stages.reasoning.difficulty_filter import ReasoningLengthDifficultyFilter, LLMBasedDifficultyFilterFunction, LLMBasedDifficultyFilter
from ray_curator.stages.reasoning.diversity_filter import LLMBasedDomainClassifier, DiversitySampler
from ray_curator.tasks import DocumentBatch
import pandas as pd


def main(args: argparse.Namespace) -> None:
    """Main function to run the reasoning curation pipeline."""

    # Create pipeline
    pipeline = Pipeline(name="reasoning_curation", description="Curation pipeline for reasoning traces")

    # Create NeMo Curator LLM client
    # You can get your API key from https://build.nvidia.com/settings/api-keys
    llm_client = OpenAIClient(
        api_key=os.environ.get("NVIDIA_API_KEY", "<your-nvidia-api-key>"),
        base_url="https://integrate.api.nvidia.com/v1",
    )
    llm_reasoning_model = "nvidia/llama-3.1-nemotron-70b-instruct" # "deepseek-ai/deepseek-r1"
    llm_domain_classifier_model = "nvidia/llama-3.1-nemotron-70b-instruct"
    llm_grader_model = "nvidia/llama-3.1-nemotron-70b-instruct"
    llm_difficulty_model_1 = "microsoft/phi-3-mini-4k-instruct"
    llm_difficulty_model_2 = "microsoft/phi-3-medium-4k-instruct"

    # Read samples for reasoning traces
    input_data = pd.read_csv(args.input_path)

    # Create a DocumentBatch from the input data
    input_batch = DocumentBatch(
        data=input_data,
        task_id="input_questions",
        dataset_name="reasoning_traces_synthetic",
    )

    # Add stages to the pipeline
    # 1. Generate reasoning traces
    pipeline.add_stage(
        ReasoningTracesSyntheticStage(
            prompt=None,
            client=llm_client,
            model_name=llm_reasoning_model,
            input_problem_field="question",
            output_field="reasoning_trace_attempt",
        ),
    )
    # 2. Correctness filter
    pipeline.add_stage(
        LLMBasedGrader(
            client=llm_client,
            model_name=llm_grader_model,
            prompt=None,
            input_problem_field="question",
            input_solution_field="answer",
            input_attempt_field="reasoning_trace_attempt",
            output_field="reasoning_trace_correctness",
        ),
    )
    # pipeline.add_stage(
    #     ScoreFilter(
    #         LLMBasedCorrectnessFilter(),
    #         text_field="reasoning_trace_correctness",
    #     ),
    # )
    # # 3. Difficulty filter
    # # 3.1. Length difficulty filter
    # pipeline.add_stage(
    #     ScoreFilter(
    #         ReasoningLengthDifficultyFilter(
    #             min_length=100,
    #         ),
    #         text_field="reasoning_trace_attempt",
    #     ),
    # )
    # # 3.2. LLM-based difficulty filter
    # pipeline.add_stage(
    #     ReasoningTracesSyntheticStage(
    #         prompt=None,
    #         client=llm_client,
    #         model_name=llm_difficulty_model_1,
    #         input_problem_field="question",
    #         output_field="llm_difficulty_1_attempt",
    #     ),
    # )
    # pipeline.add_stage(
    #     ReasoningTracesSyntheticStage(
    #         prompt=None,
    #         client=llm_client,
    #         model_name=llm_difficulty_model_2,
    #         input_problem_field="question",    
    #         output_field="llm_difficulty_2_attempt",
    #     ),
    # )
    # pipeline.add_stage(
    #     LLMBasedGrader(
    #         client=llm_client,
    #         model_name=llm_grader_model,
    #         prompt=None,
    #         input_problem_field="question",
    #         input_attempt_field="llm_difficulty_1_attempt",
    #         input_solution_field="answer",
    #         output_field="llm_difficulty_1_correctness",
    #     ),
    # )
    # pipeline.add_stage(
    #     LLMBasedGrader(
    #         client=llm_client,
    #         model_name=llm_grader_model,
    #         prompt=None,
    #         input_problem_field="question",
    #         input_attempt_field="llm_difficulty_2_attempt",
    #         input_solution_field="answer",
    #         output_field="llm_difficulty_2_correctness",
    #     ),
    # )
    # pipeline.add_stage(
    #     LLMBasedDifficultyFilter(
    #         filter_obj=LLMBasedDifficultyFilterFunction(
    #             llm_correctness_fields=["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"],
    #         ),
    #         llm_correctness_fields=["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"],
    #     ),
    # )
    # # 4. Diversity filter
    # # 4.1. Domain classifier
    # pipeline.add_stage(
    #     LLMBasedDomainClassifier(
    #         client=llm_client,
    #         model_name=llm_domain_classifier_model,
    #         prompt=None,
    #         domains_file_path=args.domains_file_path,
    #         input_problem_field="question",
    #         output_field="domain",
    #     ),
    # )
    # # 4.2. Diversity sampler
    # pipeline.add_stage(
    #     DiversitySampler(
    #         sampling_size=20,
    #         input_problem_field="question",
    #         input_domain_field="domain",
    #     ),
    # )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting synthetic data generation pipeline...")
    results = pipeline.run(executor, [input_batch])

    # Print results
    print("\nPipeline completed!")
    print(f"Total output documents: {len(results) if results else 0}")

    if results:
        for i, document_batch in enumerate(results):
            print(f"\nDocument Batch {i}:")
            print(f"Number of documents: {len(document_batch.data)}")
            print("\nGenerated text:")
            
            # Print the DataFrame data directly
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(document_batch.data)

            print("-" * 40)

    # Write to output file
    print(f"Writing to output file: {args.output_path}")
    combined_data = pd.concat([document_batch.data for document_batch in results])
    combined_data.to_json(args.output_path, orient="records", lines=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--domains_file_path", type=str, default=None)
    args = parser.parse_args()

    main(args)