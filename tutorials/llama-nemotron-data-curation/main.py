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

from __future__ import annotations

import argparse
import os
import time
from itertools import zip_longest
from typing import TYPE_CHECKING

import dask.dataframe as dd
import fasttext
import pandas as pd
from dask.delayed import delayed
from transformers import AutoTokenizer

if TYPE_CHECKING:
    import cudf
    import dask_cudf

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import (
    NoWorkerError,
    get_client,
    load_object_on_worker,
)
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper


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

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        inpt = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        has_boxed_in_input = inpt.str.contains(r"\\boxed", na=False)
        has_boxed_in_output = outpt.str.contains(r"\\boxed", na=False)
        return ~(has_boxed_in_input & ~has_boxed_in_output)

    @batched
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
    def __init__(self, text_fields: list[str] | None = None):
        if text_fields is None:
            self.text_fields = ["reasoning", "output"]
        else:
            self.text_fields = text_fields

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        reasoning = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        is_off = reasoning == "off"
        has_think_tags = outpt.str.contains(r"<think>|</think>", na=False)
        return ~(is_off & has_think_tags)

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# Reasoning on and doesn't contain think open tag
class MissingThinkOpenTagFilter(DocumentFilter):
    def __init__(self, text_fields: list[str] | None = None):
        if text_fields is None:
            self.text_fields = ["reasoning", "output"]
        else:
            self.text_fields = text_fields

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        reasoning = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        is_on = reasoning == "on"
        has_think = outpt.str.contains(r"<think>", na=False)
        has_end_think = outpt.str.contains(r"</think>", na=False)
        return ~(is_on & (~has_think | ~has_end_think))

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


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

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        try:
            self.tokenizer = load_object_on_worker(
                attr=f"{self._name}.tokenizer",
                load_object_function=AutoTokenizer.from_pretrained,
                load_object_kwargs={"pretrained_model_name_or_path": self.pretrained_model_name_or_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading tokenizer: {e}"
            raise RuntimeError(msg) from e

        try:
            self.model = load_object_on_worker(
                attr=f"{self._name}.model",
                load_object_function=fasttext.load_model,
                load_object_kwargs={"path": self.lang_id_model_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading model: {e}"
            raise RuntimeError(msg) from e

        return df.apply(
            lambda row: self.is_english(
                row[self.text_fields[0]],
                row[self.text_fields[1]],
                row[self.text_fields[2]],
            ),
            axis=1,
        )

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


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

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        try:
            self.tokenizer = load_object_on_worker(
                attr=f"{self._name}.tokenizer",
                load_object_function=AutoTokenizer.from_pretrained,
                load_object_kwargs={"pretrained_model_name_or_path": self.pretrained_model_name_or_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading tokenizer: {e}"
            raise RuntimeError(msg) from e

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

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return (scores > 0) & (scores <= self.max_token_count)


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

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        outpt = df[self.text_fields[0]]

        try:
            tokenizer = load_object_on_worker(
                attr=f"{self._name}.tokenizer",
                load_object_function=AutoTokenizer.from_pretrained,
                load_object_kwargs={"pretrained_model_name_or_path": self.pretrained_model_name_or_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading tokenizer: {e}"
            raise RuntimeError(msg) from e

        outpt_copy = outpt.copy()
        templates_list = outpt_copy.apply(
            lambda text: tokenizer.apply_chat_template(
                [{"role": "assistant", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                truncation=False,
            )
        ).tolist()
        tokenized = tokenizer(templates_list)
        return pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=outpt_copy.index)

    @batched
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


# Apply format_input_output to each row in the partition and overwrite the input and output columns
def format_partition(df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    new_inputs = []
    new_outputs = []

    for _, row in df.iterrows():
        prompt, completion = format_input_output(row["system_prompt"], row["input"], row["output"], tokenizer)
        new_inputs.append(prompt)
        new_outputs.append(completion)

    df["input"] = new_inputs
    df["output"] = new_outputs

    return df


def interleave_partitions(
    df1: dd.DataFrame | dask_cudf.DataFrame, df2: dd.DataFrame | dask_cudf.DataFrame, gpu: bool = False
) -> dd.DataFrame | dask_cudf.DataFrame:
    parts1 = df1.to_delayed()
    parts2 = df2.to_delayed()

    merged_parts = []
    for p1, p2 in zip_longest(parts1, parts2):
        if p1 is not None:
            merged_parts.append(p1)
        if p2 is not None:
            merged_parts.append(p2)

    if gpu:
        import dask_cudf

        return dask_cudf.from_delayed(merged_parts, meta=df1._meta)  # noqa: SLF001
    else:
        return dd.from_delayed(merged_parts, meta=df1._meta)  # noqa: SLF001


def _interleave_rows(
    df1: pd.DataFrame | cudf.DataFrame, df2: pd.DataFrame | cudf.DataFrame, gpu: bool = False
) -> pd.DataFrame | cudf.DataFrame:
    max_len = max(len(df1), len(df2))
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    rows = []
    for i in range(max_len):
        if i < len(df1):
            rows.append(df1.iloc[i])
        if i < len(df2):
            rows.append(df2.iloc[i])

    if gpu:
        import cudf

        return cudf.DataFrame(rows)
    else:
        return pd.DataFrame(rows)


def interleave_rows(
    df1: dd.DataFrame | dask_cudf.DataFrame, df2: dd.DataFrame | dask_cudf.DataFrame, gpu: bool = False
) -> dd.DataFrame | dask_cudf.DataFrame:
    df1_parts = df1.to_delayed()
    df2_parts = df2.to_delayed()

    interleaved_parts = []
    for part1, part2 in zip_longest(df1_parts, df2_parts):
        if part1 is not None and part2 is not None:
            interleaved = delayed(_interleave_rows)(part1, part2, gpu)
        elif part1 is not None:
            interleaved = part1
        elif part2 is not None:
            interleaved = part2
        interleaved_parts.append(interleaved)

    if gpu:
        import dask_cudf

        return dask_cudf.from_delayed(interleaved_parts, meta=df1._meta)  # noqa: SLF001
    else:
        return dd.from_delayed(interleaved_parts, meta=df1._meta)  # noqa: SLF001


def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915
    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841

    start_time = time.time()

    # Handle input path
    input_files = list(get_all_files_paths_under(args.input_dir, keep_extensions="jsonl"))
    if args.filename_filter:
        # Filter out files that don't contain any of the provided substrings
        input_files = [filename for filename in input_files if any(s in filename for s in args.filename_filter)]

    # If neither is set, set the default blocksize to 100MB
    if args.json_blocksize is None and args.json_files_per_partition is None:
        args.json_blocksize = "100mb"

    dataset = DocumentDataset.read_json(
        input_files, blocksize=args.json_blocksize, files_per_partition=args.json_files_per_partition
    )

    if args.generate_statistics:
        initial_count = dataset.df.shape[0].compute()
        print("Initial number of samples:", initial_count)

    # Filter out samples based on token count
    print("Applying filters and counting tokens")
    filter_steps = Sequential(
        [
            ScoreFilter(
                NanoFilter(),
                text_field="used_in_training",
                score_type=bool,
            ),
            ScoreFilter(
                EmptyThinkTagsFilter(),
                text_field="output",
                score_type=bool,
            ),
            ScoreFilter(
                MalformedFilter(),
                text_field=["input", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                MissingThinkCloseTagFilter(),
                text_field="output",
                score_type=bool,
            ),
            ScoreFilter(
                ContainsThinkOpenTagFilter(),
                text_field=["reasoning", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                MissingThinkOpenTagFilter(),
                text_field=["reasoning", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                NonEnglishFilter(args.tokenizer, args.lang_id_model_path),
                text_field=["system_prompt", "input", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                TokenCountFilter(args.tokenizer, args.max_token_count),
                text_field=["system_prompt", "input", "output"],
                score_field="token_count",
                score_type=int,
            ),
            ScoreFilter(
                CompletionTokenCountFilter(args.tokenizer, args.max_completion_token_count),
                text_field=["output"],
                score_field="completion_token_count",
                score_type=int,
            ),
        ]
    )
    dataset_df = filter_steps(dataset).df

    print("Reformatting input and output columns")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    meta = dataset_df._meta.copy()  # noqa: SLF001
    meta["input"] = meta["input"].astype(str)
    dataset_df = dataset_df.map_partitions(lambda df: format_partition(df, tokenizer), meta=meta)

    dataset_df = dataset_df.persist()

    if args.generate_statistics:
        removed_count = initial_count - dataset_df.shape[0].compute()
        print(f"Removed {removed_count} samples")

    # Convert to GPU if requested
    if args.device == "gpu":
        import cudf

        print("Converting to GPU")
        dataset_df = dataset_df.map_partitions(lambda partition: cudf.from_pandas(partition))

    # Sort dataset by token count
    print("Sorting...")
    sorted_dataset_df = dataset_df.sort_values("completion_token_count")
    sorted_dataset_df = sorted_dataset_df.persist()

    # Split into thinking ON and OFF
    print("Splitting dataset")
    sorted_thinking_on = sorted_dataset_df.map_partitions(lambda df: df[df["reasoning"] == "on"])
    sorted_thinking_off = sorted_dataset_df.map_partitions(lambda df: df[df["reasoning"] == "off"])

    # No specific columns are accessed after this point, so we can drop any that the user specifies
    if args.remove_columns:
        sorted_thinking_on = sorted_thinking_on.drop(columns=args.remove_columns, axis=1)
        sorted_thinking_off = sorted_thinking_off.drop(columns=args.remove_columns, axis=1)

    if args.generate_statistics:
        thinking_on_count = sorted_thinking_on.shape[0].compute()
        thinking_off_count = sorted_thinking_off.shape[0].compute()
        print(f"Number of samples in thinking ON: {thinking_on_count}")
        print(f"Number of samples in thinking OFF: {thinking_off_count}")

    if not args.global_interleave:
        print("Approximate interleaving...")
        # Interleave the sorted DataFrame partitions
        interleaved_df = interleave_partitions(sorted_thinking_on, sorted_thinking_off, gpu=args.device == "gpu")
    else:
        if args.device == "gpu":
            msg = "Global interleaving on GPU is not supported. Please use --global-interleave or CPU."
            raise RuntimeError(msg)

        print("Global interleaving...")
        # Interleave the sorted DataFrame rows
        interleaved_df = interleave_rows(sorted_thinking_on, sorted_thinking_off, gpu=args.device == "gpu")

    if args.generate_statistics:
        print(f"Final dataset size: {interleaved_df.shape[0].compute()}")

    # Save dataset
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_dir + "/part_*.jsonl"
    interleaved_df.to_json(output_path, orient="records", lines=True)
    print(f"Saved sorted dataset to {output_path}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Prepare dataset for curriculum learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_helper = ArgumentHelper(parser)
    arg_helper.add_distributed_args()
    arg_helper.attach_bool_arg(
        parser,
        "global-interleave",
        default=False,
        help="""
        If True, the datasets will be interleaved globally, i.e., row by row.
        If False, the datasets will be interleaved approximately, i.e., partition by partition.
        Default is False.
        """,
    )
    arg_helper.attach_bool_arg(
        parser,
        "generate-statistics",
        default=False,
        help="""
        If True, statistics about the number of rows filtered from the original dataset will be displayed.
        Generating statistics will slow down the script and is not recommended for large datasets.
        Default is False.
        """,
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
        help="The number of JSONL files for each partition of the DocumentDataset.",
        required=False,
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
