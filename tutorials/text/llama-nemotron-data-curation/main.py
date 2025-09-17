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

from filters.heuristic_filters import (
    ContainsThinkOpenTagFilter,
    EmptyThinkTagsFilter,
    MissingThinkCloseTagFilter,
    MissingThinkOpenTagFilter,
    NanoFilter,
    ThinkingOnFilter,
    malformed_filter,
)
from filters.model_filters import ApplyChatTemplate, CompletionTokenCountFilter, NonEnglishFilter, TokenCountFilter

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.file_utils import get_all_file_paths_under


def main(args: argparse.Namespace) -> None:
    # Initialize and start Ray client with the number of CPUs specified by the user
    ray_client = RayClient(num_cpus=args.num_cpus)
    ray_client.start()

    # Initialize pipelines
    pipeline_thinking_on = Pipeline(
        name="curriculum_learning_thinking_on", description="Prepare dataset for curriculum learning with thinking ON."
    )
    pipeline_thinking_off = Pipeline(
        name="curriculum_learning_thinking_off",
        description="Prepare dataset for curriculum learning with thinking OFF.",
    )

    start_time = time.time()

    # Handle input path
    input_files = list(get_all_file_paths_under(args.input_dir, recurse_subdirectories=True, keep_extensions="jsonl"))
    if args.filename_filter:
        # Filter out files that don't contain any of the provided substrings
        input_files = [filename for filename in input_files if any(s in filename for s in args.filename_filter)]

    # If neither is set, set the default blocksize to 100MB
    if args.json_blocksize is None and args.json_files_per_partition is None:
        args.json_blocksize = "100mb"

    pipeline_thinking_on.add_stage(
        JsonlReader(
            file_paths=input_files, blocksize=args.json_blocksize, files_per_partition=args.json_files_per_partition
        )
    )
    pipeline_thinking_off.add_stage(
        JsonlReader(
            file_paths=input_files, blocksize=args.json_blocksize, files_per_partition=args.json_files_per_partition
        )
    )

    # Split into thinking ON and OFF
    pipeline_thinking_on.add_stage(ScoreFilter(ThinkingOnFilter(), text_field="reasoning"))
    pipeline_thinking_off.add_stage(ScoreFilter(ThinkingOnFilter(), text_field="reasoning", invert=True))

    # Filter out samples based on various criteria
    filter_steps = [
        ScoreFilter(
            NanoFilter(),
            text_field="used_in_training",
        ),
        ScoreFilter(
            EmptyThinkTagsFilter(),
            text_field="output",
        ),
        malformed_filter,
        ScoreFilter(
            MissingThinkCloseTagFilter(),
            text_field="output",
        ),
    ]
    for filter_step in filter_steps:
        pipeline_thinking_on.add_stage(filter_step)
        pipeline_thinking_off.add_stage(filter_step)

    # Filter out samples in thinking OFF that contain think tags
    pipeline_thinking_off.add_stage(
        ScoreFilter(
            ContainsThinkOpenTagFilter(),
            text_field="output",
        )
    )
    # Filter out samples in thinking OFF that do not contain think tags
    pipeline_thinking_off.add_stage(
        ScoreFilter(
            MissingThinkOpenTagFilter(),
            text_field="output",
        )
    )

    # Filter out samples based on token count
    tokenizer_steps = [
        NonEnglishFilter(
            tokenizer_identifier=args.tokenizer,
            hf_token=args.hf_token,
            lang_id_model_path=args.lang_id_model_path,
            input_field="input",
            output_field="output",
            system_prompt_field="system_prompt",
        ),
        TokenCountFilter(
            tokenizer_identifier=args.tokenizer,
            hf_token=args.hf_token,
            max_token_count=args.max_token_count,
            input_field="input",
            output_field="output",
            system_prompt_field="system_prompt",
        ),
        CompletionTokenCountFilter(
            tokenizer_identifier=args.tokenizer,
            hf_token=args.hf_token,
            max_completion_token_count=args.max_completion_token_count,
            output_field="output",
        ),
        ApplyChatTemplate(
            tokenizer_identifier=args.tokenizer,
            hf_token=args.hf_token,
            input_field="input",
            output_field="output",
            system_prompt_field="system_prompt",
        ),
    ]
    for tokenizer_step in tokenizer_steps:
        pipeline_thinking_on.add_stage(tokenizer_step)
        pipeline_thinking_off.add_stage(tokenizer_step)

    # No specific columns are accessed after this point, so we can drop any that the user specifies
    if args.remove_columns:
        # Use processing_stage decorator to remove columns
        @processing_stage(name="remove_columns", resources=Resources(cpus=1.0), batch_size=1)
        def remove_columns(task: DocumentBatch) -> DocumentBatch:
            task.data = task.data.drop(columns=args.remove_columns, axis=1)
            return task

        pipeline_thinking_on.add_stage(remove_columns)
        pipeline_thinking_off.add_stage(remove_columns)

    # Save intermediate datasets
    pipeline_thinking_on.add_stage(JsonlWriter(os.path.join(args.output_dir, "thinking_on")))
    pipeline_thinking_off.add_stage(JsonlWriter(os.path.join(args.output_dir, "thinking_off")))

    # Run pipelines
    _thinking_on_output = pipeline_thinking_on.run()
    _thinking_off_output = pipeline_thinking_off.run()

    # TODO: Sort datasets
    # TODO: Interleave datasets
    # TODO: Combine datasets into single file
    # TODO: Save datasets

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Prepare dataset for curriculum learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="Number of CPUs to use.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the input directory containing JSONL files.",
        required=True,
    )
    parser.add_argument(
        "--filename-filter",
        nargs="+",
        type=str,
        help="If specified, only files with names containing one or more of the provided substrings will be processed.",
    )
    parser.add_argument(
        "--remove-columns",
        nargs="+",
        type=str,
        help="Columns to remove from the dataset.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face tokenizer",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token",
    )
    parser.add_argument(
        "--lang-id-model-path",
        type=str,
        help="Path to the FastText model",
        required=True,
    )
    parser.add_argument(
        "--max-token-count",
        type=int,
        default=16384,
        help="Optional maximum token count. Rows exceeding this count will be filtered out.",
    )
    parser.add_argument(
        "--max-completion-token-count",
        type=int,
        default=8192,
        help="Optional maximum completion token count. Rows exceeding this count will be filtered out.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output directory.",
        required=True,
    )

    parser.add_argument(
        "--json-blocksize",
        type=str,
        help="Blocksize to use for reading the JSONL files.",
        required=False,
    )
    parser.add_argument(
        "--json-files-per-partition",
        type=int,
        help="The number of JSONL files to read for each DocumentBatch.",
        required=False,
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
