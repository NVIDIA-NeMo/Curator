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
import time
from dataclasses import dataclass

import pandas as pd
from filters.heuristic_filters import (
    ContainsThinkOpenTagFilter,
    EmptyThinkTagsFilter,
    MissingThinkCloseTagFilter,
    MissingThinkOpenTagFilter,
    NanoFilter,
    ThinkingOnFilter,
    malformed_filter,
)
from transformers import AutoTokenizer

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.file_utils import get_all_file_paths_under


# Modifier for input and output chat templates
def format_input_output(system_prompt: str, inpt: list[dict], outpt: str, tokenizer: AutoTokenizer) -> tuple[str, str]:
    prompt_and_completion = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            *inpt,
            {"role": "assistant", "content": outpt},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            *inpt,
        ],
        tokenize=False,
        # We expect the model to start predicting tokens after it sees the start of the assistant response turn
        add_generation_prompt=True,
    )

    # Remove the prompt from prompt_and_completion via string manipulation to extract the completion part
    completion = prompt_and_completion[len(prompt) :]

    # input, output
    return prompt, completion


# Apply format_input_output to each row in the batch and overwrite the input and output columns
def format_batch(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    input_field: str = "input",
    output_field: str = "output",
    system_prompt_field: str = "system_prompt",
) -> pd.DataFrame:
    new_inputs = []
    new_outputs = []

    for _, row in df.iterrows():
        prompt, completion = format_input_output(
            row[system_prompt_field], row[input_field], row[output_field], tokenizer
        )
        new_inputs.append(prompt)
        new_outputs.append(completion)

    df[input_field] = new_inputs
    df[output_field] = new_outputs

    return df


@dataclass
class FormatBatch(ProcessingStage[DocumentBatch, DocumentBatch]):
    tokenizer: str
    input_field: str = "input"
    output_field: str = "output"
    system_prompt_field: str = "system_prompt"
    _name: str = "format_batch"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_field, self.output_field, self.system_prompt_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_field, self.output_field]

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # Easy hack to only download the tokenizer once if it is not already downloaded
        _ = AutoTokenizer.from_pretrained(self.tokenizer, local_files_only=False)

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True, local_files_only=True)

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        batch.data = format_batch(
            batch.data, self._tokenizer, self.input_field, self.output_field, self.system_prompt_field
        )
        return batch


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

    # TODO: Filter out samples based on token count
    """
    tokenizer_steps = [
        ScoreFilter(
            NonEnglishFilter(args.tokenizer, args.lang_id_model_path),
            text_field=["system_prompt", "input", "output"],
        ),
        ScoreFilter(
            TokenCountFilter(args.tokenizer, args.max_token_count),
            text_field=["system_prompt", "input", "output"],
            score_field="token_count",
        ),
        ScoreFilter(
            CompletionTokenCountFilter(args.tokenizer, args.max_completion_token_count),
            text_field=["output"],
            score_field="completion_token_count",
        ),
    ]
    for tokenizer_step in tokenizer_steps:
        pipeline_thinking_on.add_stage(tokenizer_step)
        pipeline_thinking_off.add_stage(tokenizer_step)
    """

    pipeline_thinking_on.add_stage(FormatBatch(args.tokenizer))
    pipeline_thinking_off.add_stage(FormatBatch(args.tokenizer))

    # No specific columns are accessed after this point, so we can drop any that the user specifies
    if args.remove_columns:
        # Use processing_stage decorator to remove columns
        @processing_stage(name="remove_columns", resources=Resources(cpus=1.0), batch_size=1)
        def remove_columns(task: DocumentBatch) -> DocumentBatch:
            task.data = task.data.drop(columns=args.remove_columns, axis=1)
            return task

        pipeline_thinking_on.add_stage(remove_columns)
        pipeline_thinking_off.add_stage(remove_columns)

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
