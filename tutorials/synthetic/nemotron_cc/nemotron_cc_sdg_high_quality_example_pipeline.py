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
from loguru import logger
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
from nemo_curator.stages.text.filters import Filter
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
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
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["jsonl", "parquet"],
        help="Output format for generated data (jsonl or parquet)",
    )
    parser.add_argument(
        "--input-parquet-path",
        type=str,
        default=None,
        help="If set, read inputs from Parquet path/glob via Curator ParquetReader",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use preset in-script input_data instead of reading Parquet input",
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
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

    input_tasks = None
    if args.mock:
        input_data = [
            {
                "text": "The Amazon rainforest contains an unparalleled diversity of plant and animal species. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 12,
            },
            {
                "text": "Isaac Newton formulated the laws of motion and universal gravitation. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 4,
            },
            {
                "text": "The Great Wall of China is a historic fortification built to protect ancient Chinese states. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 17,
            },
            {
                "text": "Mercury is the smallest planet in the Solar System and orbits closest to the Sun. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 1,
            },
            {
                "text": "The Parthenon is a classical Greek temple dedicated to the goddess Athena. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 9,
            },
            {
                "text": "Giraffes are the tallest living terrestrial animals, native to African savannas. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 6,
            },
            {
                "text": "Marie Curie made pioneering contributions to the study of radioactivity. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 14,
            },
            {
                "text": "The Pacific Ocean covers more area than all landmasses combined. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 3,
            },
            {
                "text": "The Rosetta Stone provided the key to deciphering ancient Egyptian hieroglyphs. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 18,
            },
            {
                "text": "The cheetah is capable of reaching speeds over 100 kilometers per hour. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 8,
            },
            {
                "text": "Mount Everest is the highest peak on Earth, located in the Himalayas. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 2,
            },
            {
                "text": "The Sahara Desert spans much of North Africa and is the largest hot desert in the world. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 5,
            },
            {
                "text": "Leonardo da Vinci was an influential artist and inventor during the Italian Renaissance. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 19,
            },
            {
                "text": "Photosynthesis enables plants to convert sunlight into chemical energy. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 7,
            },
            {
                "text": "The Taj Mahal is an iconic mausoleum built by Mughal emperor Shah Jahan. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 0,
            },
            {
                "text": "The human brain contains billions of neurons that communicate through electrical signals. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 11,
            },
            {
                "text": "The Roman Empire was one of the most powerful civilizations of the ancient world. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 10,
            },
            {
                "text": "The Hubble Space Telescope has captured detailed images of distant galaxies. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 15,
            },
            {
                "text": "The Eiffel Tower was constructed for the 1889 Exposition Universelle in Paris. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 4,
            },
            {
                "text": "Antarctica contains the vast majority of the Earth's freshwater ice. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields. This topic is widely studied and holds significant relevance in scientific and historical contexts. It illustrates important principles, involves complex interactions, and helps researchers develop a deeper understanding of natural systems and cultural developments. Over many years, scholars, explorers, and scientists have contributed insights that enrich our collective knowledge, enabling future generations to continue studying and appreciating its broader importance across different fields.",
                "bucketed_results": 9,
            },
        ]
        # Divide input_data into batches of `batch_size` each
        # Simulate `num_input_tasks` input tasks
        batch_size = 10
        num_input_tasks = 100
        input_batches = [input_data[i : i + batch_size] for i in range(0, len(input_data), batch_size)]
        input_tasks = []
        id_counter = 0
        for i in range(num_input_tasks // len(input_batches)):
            for j, batch in enumerate(input_batches):
                df = pd.DataFrame(batch)
                # Ensure a stable document identifier required by DocumentJoiner
                df["id"] = [id_counter + j for j in range(len(df))]
                id_counter += len(df)
                input_task = DocumentBatch(
                    data=df,
                    task_id=f"input_batch_{i * batch_size + j}",
                    dataset_name="data_for_sdg",
                )
                input_tasks.append(input_task)
        logger.info("Number of input tasks: ", len(input_tasks))
        logger.info("Size of each input task: ", input_tasks[0].data.shape)
    else:
        if not args.input_parquet_path:
            msg = "When not using --mock, you must provide --input-parquet-path to read inputs."
            raise ValueError(msg)
        # Use ParquetReader to load input data from parquet path/glob
        pipeline.add_stage(
            ParquetReader(
                file_paths=[args.input_parquet_path],
                # Optional: select columns if needed by downstream stages
                # fields=["text", "bucketed_results"]  # noqa: ERA001
                read_kwargs={"engine": "pyarrow", "dtype_backend": "pyarrow"},
            )
        )

    ### Extract high quality data
    # Filtering the input data, only run with high quality data
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
    # Add output writer based on selected format
    if args.output_format == "jsonl":
        pipeline.add_stage(JsonlWriter(path=args.output_path))
    else:
        pipeline.add_stage(ParquetWriter(path=args.output_path))

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
                    # Read the output file to get the actual data
                    if file_path.endswith(".jsonl"):
                        df = pd.read_json(file_path, lines=True)
                    elif file_path.endswith(".parquet"):
                        df = pd.read_parquet(file_path)
                    else:
                        continue
                    all_data_frames.append(df)

    # Display sample of generated documents
    print("\n" + "=" * 50)
    print("Sample of generated documents:")
    print("=" * 50)
    for i, df in enumerate(all_data_frames):
        out_path = output_files[i]
        print(f"\nFile {i + 1}: {out_path}")
        print(f"Number of documents: {len(df)}")
        print("\nFirst 5 rows:")
        for j, row in enumerate(df.head(5).to_dict(orient="records")):
            print(f"Document {j + 1}: {row}")
            print("-" * 40)

    client.stop()


if __name__ == "__main__":
    main()
