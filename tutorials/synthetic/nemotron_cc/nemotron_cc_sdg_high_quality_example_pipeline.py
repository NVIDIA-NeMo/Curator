# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import time

import pandas as pd
from nemotron_cc_pipelines import (
    add_distill_postprocessing_pipeline,
    add_diverse_qa_postprocessing_pipeline,
    add_extract_knowledge_postprocessing_pipeline,
    add_knowledge_list_postprocessing_pipeline,
    add_preprocessing_pipeline,
)
from transformers import AutoTokenizer

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import (
    DistillStage,
    DiverseQAStage,
    ExtractKnowledgeStage,
    KnowledgeListStage,
)
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
)
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.modules.score_filter import Filter
from nemo_curator.tasks.document import DocumentBatch

# Threshold used to bucket and filter input examples
BUCKETED_RESULTS_THRESHOLD = 11

TASK_CONFIGS = {
    "diverse_qa": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": DIVERSE_QA_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 600,
    },
    "distill": {
        "system_prompt": NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
        "prompt_template": DISTILL_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 10,
        "max_input_tokens": 2000,
        "max_output_tokens": 1600,
    },
    "extract_knowledge": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1400,
        "max_output_tokens": 1400,
    },
    "knowledge_list": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": KNOWLEDGE_LIST_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 600,
    },
}

GENERATION_CONFIG = {
    "MAX_INPUT_TOKENS": 2000,
    "MAX_OUTPUT_TOKENS": 1600,
    "TOP_K": 0,
    "TOP_P": 0.9,
    "END_STRINGS": "['</s>']",
    "TEMPERATURE": 0.5,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # API Configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable)",
    )
    parser.add_argument(
        "--base-url", type=str, default="https://integrate.api.nvidia.com/v1", help="Base URL for the API endpoint"
    )
    parser.add_argument(
        "--max-concurrent-requests", type=int, default=3, help="Maximum number of concurrent API requests"
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed requests")
    parser.add_argument("--base-delay", type=float, default=1.0, help="Base delay between retries (in seconds)")

    # Task settings
    # Model Configuration
    parser.add_argument(
        "--task",
        type=str,
        default="diverse_qa",
        help="Task to run",
        choices=["diverse_qa", "distill", "extract_knowledge", "knowledge_list"],
    )

    # Model Configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta/llama-3.3-70b-instruct",
        help="Name of the model to use for generation",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name of the tokenizer to use for generation",
    )

    # Generation Configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default="./synthetic_output",
        help="Directory path to save the generated synthetic data in JSONL format",
    )

    # LLM Sampling Parameters (for diversity)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (higher = more random/diverse, lower = more deterministic). Range: 0.0-2.0",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter (considers tokens with cumulative probability top_p). Range: 0.0-1.0",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top k tokens to consider for sampling. Range: 0-1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for non-deterministic generation)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate per sample",
    )
    parser.add_argument(
        "--end-strings",
        type=str,
        default=None,
        help="End strings to stop generation",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0915
    """Main function to run the synthetic data generation pipeline."""
    client = RayClient(include_dashboard=False)
    client.start()

    args = parse_args()

    # Set tokenizer
    if args.tokenizer is None:
        msg = "Tokenizer is required"
        raise ValueError(msg)
    args.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.hf_token = os.environ.get("HF_TOKEN", "")

    # Validate API key
    if not args.api_key:
        msg = (
            "API key is required. Set NVIDIA_API_KEY environment variable or use --api-key argument. "
            "Get your API key from https://build.nvidia.com/settings/api-keys"
        )
        raise ValueError(msg)

    # Set task config
    task_config = TASK_CONFIGS[args.task]

    # Create pipeline
    pipeline = Pipeline(name=f"nemotron_cc_{args.task}", description=f"Generate {args.task} data using Nemotron-CC")

    # Create NeMo Curator Async LLM client for faster concurrent generation
    llm_client = AsyncOpenAIClient(
        api_key=args.api_key,
        base_url=args.base_url,
        max_concurrent_requests=args.max_concurrent_requests,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
    )

    generation_config = GenerationConfig(
        temperature=args.temperature if args.temperature is not None else GENERATION_CONFIG["TEMPERATURE"],
        top_p=args.top_p if args.top_p is not None else GENERATION_CONFIG["TOP_P"],
        top_k=args.top_k if args.top_k is not None else GENERATION_CONFIG["TOP_K"],
        max_tokens=args.max_tokens if args.max_tokens is not None else GENERATION_CONFIG["MAX_OUTPUT_TOKENS"],
        stop=args.end_strings if args.end_strings is not None else GENERATION_CONFIG["END_STRINGS"],
        seed=args.seed,
    )

    input_data = [
        {"text": "The Amazon rainforest contains an unparalleled diversity of plant and animal species.", "bucketed_results": 12},
        {"text": "Isaac Newton formulated the laws of motion and universal gravitation.", "bucketed_results": 4},
        {"text": "The Great Wall of China is a historic fortification built to protect ancient Chinese states.", "bucketed_results": 17},
        {"text": "Mercury is the smallest planet in the Solar System and orbits closest to the Sun.", "bucketed_results": 1},
        {"text": "The Parthenon is a classical Greek temple dedicated to the goddess Athena.", "bucketed_results": 9},
        {"text": "Giraffes are the tallest living terrestrial animals, native to African savannas.", "bucketed_results": 6},
        {"text": "Marie Curie made pioneering contributions to the study of radioactivity.", "bucketed_results": 14},
        {"text": "The Pacific Ocean covers more area than all landmasses combined.", "bucketed_results": 3},
        {"text": "The Rosetta Stone provided the key to deciphering ancient Egyptian hieroglyphs.", "bucketed_results": 18},
        {"text": "The cheetah is capable of reaching speeds over 100 kilometers per hour.", "bucketed_results": 8},
        {"text": "Mount Everest is the highest peak on Earth, located in the Himalayas.", "bucketed_results": 2},
        {"text": "The Sahara Desert spans much of North Africa and is the largest hot desert in the world.", "bucketed_results": 5},
        {"text": "Leonardo da Vinci was an influential artist and inventor during the Italian Renaissance.", "bucketed_results": 19},
        {"text": "Photosynthesis enables plants to convert sunlight into chemical energy.", "bucketed_results": 7},
        {"text": "The Taj Mahal is an iconic mausoleum built by Mughal emperor Shah Jahan.", "bucketed_results": 0},
        {"text": "The human brain contains billions of neurons that communicate through electrical signals.", "bucketed_results": 11},
        {"text": "The Roman Empire was one of the most powerful civilizations of the ancient world.", "bucketed_results": 10},
        {"text": "The Hubble Space Telescope has captured detailed images of distant galaxies.", "bucketed_results": 15},
        {"text": "The Eiffel Tower was constructed for the 1889 Exposition Universelle in Paris.", "bucketed_results": 4},
        {"text": "Antarctica contains the vast majority of the Earth's freshwater ice.", "bucketed_results": 9},
        {"text": "The Library of Alexandria was a major center of scholarship in the ancient world.", "bucketed_results": 13},
        {"text": "Saturn is distinguished by its extensive system of icy rings.", "bucketed_results": 2},
        {"text": "The Nile River is often considered the longest river in the world.", "bucketed_results": 16},
        {"text": "Penguins are flightless birds that are highly adapted to marine life.", "bucketed_results": 3},
        {"text": "The Maya civilization developed advanced knowledge of astronomy and mathematics.", "bucketed_results": 18},
        {"text": "Pluto is a dwarf planet located in the Kuiper Belt beyond Neptune.", "bucketed_results": 6},
        {"text": "The Andes Mountains stretch along the western edge of South America.", "bucketed_results": 12},
        {"text": "The Renaissance was a cultural movement that profoundly influenced European art and science.", "bucketed_results": 8},
        {"text": "The Blue Whale is the largest known animal to have lived on Earth.", "bucketed_results": 14},
        {"text": "The Silk Road connected merchants and cultures across Asia, Africa, and Europe.", "bucketed_results": 5},
        {"text": "Gravity is a fundamental force that governs the attraction between masses.", "bucketed_results": 3},
        {"text": "The Mona Lisa is a celebrated portrait painted by Leonardo da Vinci.", "bucketed_results": 17},
        {"text": "Jupiter is the largest planet in the Solar System and has dozens of known moons.", "bucketed_results": 7},
        {"text": "The Colosseum in Rome hosted gladiatorial contests and public spectacles.", "bucketed_results": 10},
        {"text": "DNA contains the hereditary information necessary for biological development.", "bucketed_results": 15},
        {"text": "The Mariana Trench is the deepest known region of the Earth's oceans.", "bucketed_results": 19},
        {"text": "The Great Barrier Reef is the world's largest coral reef system.", "bucketed_results": 11},
        {"text": "Koalas are marsupials native to Australia that primarily eat eucalyptus leaves.", "bucketed_results": 1},
        {"text": "The Andes form the longest continental mountain range on the planet.", "bucketed_results": 16},
        {"text": "The Moon orbits Earth and influences ocean tides through gravitational forces.", "bucketed_results": 8},
    ]

    # Divide input_data into batches of 20 each
    batch_size = 20
    input_batches = [input_data[i:i + batch_size] for i in range(0, len(input_data), batch_size)]
    input_tasks = []
    for i, batch in enumerate(input_batches):
        df = pd.DataFrame(batch)
        input_task = DocumentBatch(
            data=df,
            task_id=f"input_batch_{i}",
            dataset_name="data_for_sdg",
        )
        input_tasks.append(input_task)
    # simulate larger data size
    input_tasks = input_tasks * 1000

    ### Extract high quality data
    # Filtering the input data, only run with low quality data
    pipeline.add_stage(
        Filter(
            filter_fn=lambda x: int(x) > BUCKETED_RESULTS_THRESHOLD,
            filter_field="bucketed_results",
        ),
    )

    ### Preprocessing Stages
    # Add preprocessing stages
    pipeline = add_preprocessing_pipeline(
        pipeline=pipeline,
        text_field="text",
        system_prompt=task_config["system_prompt"],
        user_prompt_template=task_config["prompt_template"],
        # DEBUGGING
        min_document_tokens=task_config["min_document_tokens"],
        min_segment_tokens=task_config["min_segment_tokens"],
        max_input_tokens=task_config["max_input_tokens"],
        args=args,
    )

    ######################## Diverse QA Stage ########################
    if args.task == "diverse_qa":
        # Add diverse QA stage
        pipeline.add_stage(
            DiverseQAStage(
                client=llm_client,
                model_name=args.model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="diverse_qa",
            )
        )

        # Add diverse QA postprocessing stages
        pipeline = add_diverse_qa_postprocessing_pipeline(
            pipeline=pipeline,
            llm_response_field="diverse_qa",
            args=args,
        )

    ######################## Distill Stage ########################
    elif args.task == "distill":
        # Add distill stage
        pipeline.add_stage(
            DistillStage(
                client=llm_client,
                model_name=args.model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="distill",
            )
        )

        # Add distill postprocessing stages
        pipeline = add_distill_postprocessing_pipeline(
            pipeline=pipeline,
            llm_response_field="distill",
            args=args,
        )

    # ######################## Extract Knowledge Stage ########################
    elif args.task == "extract_knowledge":
        # Add knowledge extraction stage
        pipeline.add_stage(
            ExtractKnowledgeStage(
                client=llm_client,
                model_name=args.model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="extract_knowledge",
            )
        )

        # Add extract knowledge postprocessing stages
        pipeline = add_extract_knowledge_postprocessing_pipeline(
            pipeline=pipeline,
            llm_response_field="extract_knowledge",
            args=args,
        )

    ######################## Knowledge List Stage ########################
    elif args.task == "knowledge_list":
        # Add knowledge list stage
        pipeline.add_stage(
            KnowledgeListStage(
                client=llm_client,
                model_name=args.model_name,
                generation_config=generation_config,
                input_field="text",
                output_field="knowledge_list",
            )
        )

        # Add knowledge list postprocessing stages
        pipeline = add_knowledge_list_postprocessing_pipeline(
            pipeline=pipeline,
            llm_response_field="knowledge_list",
            args=args,
        )

    ### Write output
    # Add JSONL writer to save the generated data
    pipeline.add_stage(
        JsonlWriter(
            path=args.output_path,
        )
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")


    # Create executor
    executor = XennaExecutor()

    # Execute pipeline with timing
    print("Starting synthetic data generation pipeline...")
    start_time = time.time()
    results = pipeline.run(executor, input_tasks)
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Print results
    print("\nPipeline completed!")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    # DEBUGGING
    print("results: ", results)

    # Collect output file paths and read generated data
    output_files = []
    all_data_frames = []
    if results:
        print(f"\nGenerated data saved to: {args.output_path}")
        for result in results:
            if hasattr(result, "data") and result.data:
                for file_path in result.data:
                    print(f"  - {file_path}")
                    output_files.append(file_path)
                    # Read the JSONL file to get the actual data
                    df = pd.read_json(file_path, lines=True)
                    all_data_frames.append(df)

    # Display sample of generated documents
    print("\n" + "=" * 50)
    print("Sample of generated documents:")
    print("=" * 50)
    for i, df in enumerate(all_data_frames):
        print(f"\nFile {i + 1}: {output_files[i]}")
        print(f"Number of documents: {len(df)}")
        print("\nGenerated text (showing first 5):")
        for j, text in enumerate(df["text"].head(5)):
            print(f"Document {j + 1}:")
            print(f"'{text}'")
            print("-" * 40)

    client.stop()


if __name__ == "__main__":
    main()
