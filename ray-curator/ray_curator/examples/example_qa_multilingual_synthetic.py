"""
Quick synthetic data generation example for Ray Curator

This example shows how to use the SimpleSyntheticStage to generate synthetic data:
1. SimpleSyntheticStage: EmptyTask -> DocumentBatch : This generates synthetic text using an LLM
"""

import os

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.services.openai_client import AsyncOpenAIClient
from ray_curator.stages.synthetic.example_qa_multilingual_synthetic import LanguageFilter, QAMultilingualSyntheticStage


def main() -> None:
    """Main function to run the synthetic data generation pipeline."""

    # Create pipeline
    pipeline = Pipeline(name="synthetic_data_generation", description="Generate synthetic text data using LLM")

    # Create NeMo Curator Async LLM client for faster concurrent generation
    # You can get your API key from https://build.nvidia.com/settings/api-keys
    llm_client = AsyncOpenAIClient(
        api_key=os.environ.get("NVIDIA_API_KEY", "<your-nvidia-api-key>"),
        base_url="https://integrate.api.nvidia.com/v1",
        max_concurrent_requests=3,  # Limit concurrent requests to avoid rate limits
        max_retries=3,
        base_delay=1.0
    )

    # Define a prompt for synthetic data generation
    languages = ["English", "French", "German", "Spanish", "Italian"]
    prompt = """
    Generate a short question and a short answer in the general science domain in the language {language}.
    Begin with the language name using the 2-letter code, which is in square brackets, e.g. [EN] for English, [FR] for French, [DE] for German, [ES] for Spanish, [IT] for Italian.
    """

    # Add the synthetic data generation stage
    pipeline.add_stage(
        QAMultilingualSyntheticStage(
            prompt=prompt,
            languages=languages,
            client=llm_client,
            model_name="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
            num_samples=100,  # Increased to better show async performance benefits
        )
    )
    pipeline.add_stage(
        ScoreFilter(
            LanguageFilter(
                languages=["[EN]"], # ['[EN]', '[FR]', '[DE]', '[ES]', '[IT]']
            ),
            text_field="text",
        ),
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting synthetic data generation pipeline...")
    results = pipeline.run(executor)

    # Print results
    print("\nPipeline completed!")
    print(f"Total output documents: {len(results) if results else 0}")

    if results:
        for i, document_batch in enumerate(results):
            print(f"\nDocument Batch {i}:")
            print(f"Number of documents: {len(document_batch.data)}")
            print("\nGenerated text:")
            for j, text in enumerate(document_batch.data["text"]):
                print(f"Document {j + 1}:")
                print(f"'{text}'")
                print("-" * 40)


if __name__ == "__main__":
    main()
