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

import huggingface_hub
import pandas as pd
from transformers import AutoTokenizer

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.filters import DocumentFilter
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.file_utils import get_all_file_paths_under


# Filter by thinking ON or OFF
class ThinkingOnFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return "on" in text.lower()

    def keep_document(self, score: bool) -> bool:
        return score


# Skip if not used for Nano training
class NanoFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return "nano" in text.lower()

    def keep_document(self, score: bool) -> bool:
        return score


# Filter out samples with empty think tags
class EmptyThinkTagsFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>\n\n</think>" in text or "<think>\n</think>" in text or "<think></think>" in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Skip if malformed
class MalformedFilter(DocumentFilter):
    def __init__(self, text_fields: list[str] | None = None):
        if text_fields is None:
            self.text_fields = ["input", "output"]
        else:
            self.text_fields = text_fields

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        inpt = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        has_boxed_in_input = inpt.str.contains(r"\\boxed", na=False)
        has_boxed_in_output = outpt.str.contains(r"\\boxed", na=False)
        return ~(has_boxed_in_input & ~has_boxed_in_output)

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# Doesn't contain think close tag
class MissingThinkCloseTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" in text and "</think>" not in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Reasoning off and contains think open tag
class ContainsThinkOpenTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" in text or "</think>" in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Reasoning on and doesn't contain think open tag
class MissingThinkOpenTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" not in text or "</think>" not in text)

    def keep_document(self, score: bool) -> bool:
        return score


# TODO: Fix NonEnglishFilter
# Tokenize and filter out non-English text
class NonEnglishFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        lang_id_model_path: str,
        text_fields: list[str] | None = None,
    ):
        self._name = "non_english_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lang_id_model_path = lang_id_model_path
        if text_fields is None:
            self.text_fields = ["system_prompt", "input", "output"]
        else:
            self.text_fields = text_fields

    def is_english(self, system: str, inpt: list[dict], outpt: str) -> bool:
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                *inpt,
                {"role": "assistant", "content": outpt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = str(text).replace("\n", " ").strip()
        return self.model.predict(text)[0][0] == "__label__en"

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(
            lambda row: self.is_english(
                row[self.text_fields[0]],
                row[self.text_fields[1]],
                row[self.text_fields[2]],
            ),
            axis=1,
        )

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# TODO: Fix TokenCountFilter
# Tokenize system_prompt, input, and output and filter out samples with too many tokens
class TokenCountFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_token_count: int = 16384,
        text_fields: list[str] | None = None,
    ):
        super().__init__()
        self._name = "token_count_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_token_count = max_token_count
        if text_fields is None:
            self.text_fields = ["system_prompt", "input", "output"]
        else:
            self.text_fields = text_fields

    def apply_chat_template(self, system: str, inpt: list[dict], outpt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                *inpt,
                {"role": "assistant", "content": outpt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        templates_list = df.apply(
            lambda row: self.apply_chat_template(
                row[self.text_fields[0]],
                row[self.text_fields[1]],
                row[self.text_fields[2]],
            ),
            axis=1,
        ).tolist()
        tokenized = self.tokenizer(templates_list)
        return pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=df.index)

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return (scores > 0) & (scores <= self.max_token_count)


# TODO: Fix CompletionTokenCountFilter
# Tokenize text and filter out samples with too many tokens
class CompletionTokenCountFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_completion_token_count: int = 8192,
        text_fields: list[str] | None = None,
    ):
        super().__init__()
        self._name = "completion_token_count_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_completion_token_count = max_completion_token_count
        if text_fields is None:
            self.text_fields = ["output"]
        else:
            self.text_fields = text_fields

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        outpt = df[self.text_fields[0]]

        outpt_copy = outpt.copy()
        templates_list = outpt_copy.apply(
            lambda text: self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                truncation=False,
            )
        ).tolist()
        tokenized = self.tokenizer(templates_list)
        return pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=outpt_copy.index)

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return (scores > 0) & (scores <= self.max_completion_token_count)


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
        # Use snapshot_download to download all files without loading the model into memory.
        huggingface_hub.snapshot_download(
            repo_id=self.tokenizer,
            local_files_only=False,  # Download if not cached
            resume_download=True,  # Resume interrupted downloads
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True, local_files_only=True)

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        df = batch.to_pandas()
        df = format_batch(df, self._tokenizer, self.input_field, self.output_field, self.system_prompt_field)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class RemoveColumns(ProcessingStage[DocumentBatch, DocumentBatch]):
    remove_columns: list[str]
    _name: str = "remove_columns"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.remove_columns

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        df = batch.to_pandas()
        df = df.drop(columns=self.remove_columns, axis=1)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def main(args: argparse.Namespace) -> None:
    # Initialize and start Ray client
    ray_client = RayClient()
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
        # TODO: Fix MalformedFilter
        #ScoreFilter(
        #    MalformedFilter(),
        #    text_field=["input", "output"],
        #),
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
        pipeline_thinking_on.add_stage(RemoveColumns(args.remove_columns))
        pipeline_thinking_off.add_stage(RemoveColumns(args.remove_columns))

    # Run pipelines
    thinking_on_output = pipeline_thinking_on.run()
    thinking_off_output = pipeline_thinking_off.run()

    # TODO: Sort datasets
    # TODO: Interleave datasets
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
    # TODO: Fix NonEnglishFilter
    """
    parser.add_argument(
        "--lang-id-model-path",
        type=str,
        help="Path to the FastText model",
        required=True,
    )
    """
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
    # TODO: Write output files
    """
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output directory.",
        required=True,
    )
    """

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
