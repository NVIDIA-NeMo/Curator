# NOTE: This file will probably get removed after the PR is merged.
# Please don't review this. It's just for FYI and making sure that reviewers can run.
# Some benchmarks at the end of the file.
"""
Simple example demonstrating how to use the Wikipedia download and extract stage.

This example shows how to:
1. Download Wikipedia dump files for a specific language
2. Extract and clean Wikipedia articles
3. Save the results to a dataset

Requirements:
- mwparserfromhell: pip install mwparserfromhell
- beautifulsoup4: pip install beautifulsoup4
- lxml: pip install lxml

Usage:
    python wikipedia_example.py
"""

import os
import tempfile

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.download.text.wikipedia import WikipediaDownloadExtractStage
from ray_curator.stages.io.writer import JsonlWriter

# Constants
MAX_DISPLAY_TASKS = 3


def create_directories() -> tuple[str, str, str]:
    """Create and return temporary directories for the example."""
    temp_dir = tempfile.mkdtemp()
    download_dir = os.path.join(temp_dir, "downloads")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return temp_dir, download_dir, output_dir


def create_pipeline(download_dir: str, output_dir: str) -> Pipeline:
    """Create and configure the Wikipedia processing pipeline."""
    # Configuration
    language = "en"  # Language code for Wikipedia (e.g., "en", "es", "fr", "de")
    url_limit = 2  # Limit to 2 dump files for this example
    record_limit = 100  # Limit to 100 articles per dump file

    # Create the Wikipedia download and extract stage
    wikipedia_stage = WikipediaDownloadExtractStage(
        language=language,
        download_dir=download_dir,
        dump_date=None,  # Use latest dump
        verbose=True,
        url_limit=url_limit,
        record_limit=record_limit,
        log_frequency=10,  # Log every 10 articles
    )

    # Create JSONL writer to save results
    jsonl_writer = JsonlWriter(output_dir=output_dir)

    # Create pipeline
    pipeline = Pipeline(
        name="wikipedia_example",
        description="Download and extract Wikipedia articles",
    )

    # Add stages
    pipeline.add_stage(wikipedia_stage)
    pipeline.add_stage(jsonl_writer)

    return pipeline, wikipedia_stage


def display_task_info(task: any, task_index: int) -> None:
    """Display information about a single task."""
    print(f"\nTask {task_index + 1}:")
    print(f"  Task ID: {task.task_id}")
    print(f"  Dataset: {task.dataset_name}")
    print(f"  Data type: {type(task.data)}")

    # If the task has data, show some sample content
    if hasattr(task, "data") and task.data is not None:
        if hasattr(task.data, "head"):  # pandas DataFrame
            print("  Sample data:")
            print(task.data.head(3))
        elif isinstance(task.data, list) and len(task.data) > 0:
            print(f"  Sample items ({len(task.data)} total):")
            for j, item in enumerate(task.data[:3]):
                print(f"    Item {j + 1}: {item}")
        else:
            print(f"  Data: {task.data}")


def display_results(results: list, output_dir: str) -> None:
    """Display results from the pipeline execution."""
    if results:
        print("\nResults from pipeline:")
        for i, task in enumerate(results):
            display_task_info(task, i)
            if i >= MAX_DISPLAY_TASKS - 1:  # Show only first 3 tasks
                break
    else:
        print("No results returned from pipeline")

    # Check if JSONL files were created
    if os.path.exists(output_dir):
        jsonl_files = [f for f in os.listdir(output_dir) if f.endswith(".jsonl")]
        if jsonl_files:
            print(f"\nGenerated JSONL files: {jsonl_files}")
        else:
            print("\nNo JSONL files were generated")


def main() -> None:
    """Run the Wikipedia download and extract example."""
    # Create directories
    temp_dir, download_dir, output_dir = create_directories()

    print(f"Temporary directory: {temp_dir}")
    print(f"Download directory: {download_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Create pipeline
        pipeline, wikipedia_stage = create_pipeline(download_dir, output_dir)

        print("\nStarting Wikipedia download for language: en")
        print(f"Stage description: {wikipedia_stage.get_description()}")

        # Print pipeline description
        print("\nPipeline description:")
        print(pipeline.describe())
        print("\n" + "=" * 50 + "\n")

        # Run the pipeline
        executor = XennaExecutor()
        print("Starting pipeline execution...")
        results = pipeline.run(executor)

        print("\nPipeline completed successfully!")
        print(f"Total output tasks: {len(results) if results else 0}")
        print(f"Results saved to: {output_dir}")

        # Display results
        display_results(results, output_dir)

    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    # Execution time for Xenna: 97s
    # Execution time for RayData: 96s
    # Both produce identical results (md5sums match)
    # 199 records in total across 2 output files.
