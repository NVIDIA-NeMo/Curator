"""
Quick synthetic data generation example for Ray Curator

This example shows how to use the SimpleSyntheticStage to generate synthetic data:
1. SimpleSyntheticStage: EmptyTask -> DocumentBatch : This generates synthetic text using an LLM
"""

import os

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.services.openai_client import OpenAIClient
from ray_curator.stages.synthetic.simple import SimpleSyntheticStage


def main() -> None:
    """Main function to run the synthetic data generation pipeline."""

    # Create pipeline
    pipeline = Pipeline(name="synthetic_data_generation", description="Generate synthetic text data using LLM")

    # Create NeMo Curator LLM client
    # You can get your API key from https://build.nvidia.com/settings/api-keys
    llm_client = OpenAIClient(
        api_key=os.environ.get("NVIDIA_API_KEY", "<your-nvidia-api-key>"),
        base_url="https://integrate.api.nvidia.com/v1",
    )

    # Define a prompt for synthetic data generation
    prompt = """Generate a creative and engaging short story about artificial intelligence.
    The story should be between 100-200 words and include elements of:
    - A futuristic setting
    - An AI character with unique personality
    - A meaningful interaction between AI and humans
    - A positive message about technology and humanity

    Make the story engaging and suitable for educational purposes."""

    # Add the synthetic data generation stage
    pipeline.add_stage(
        SimpleSyntheticStage(
            prompt=prompt,
            client=llm_client,
            model_name="nvidia/llama-3.1-nemotron-70b-instruct",  # or "mistralai/mixtral-8x7b-instruct-v0.1" for NVIDIA API
        )
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
