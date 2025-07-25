# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import random
from typing import Any

import dask
from docbuilder import LawQADownloader, LawQAIterator
from filters import FilterLowScores
from modifiers import CleanHTML
from openai import AsyncOpenAI
from synthetic_gen import SyntheticGenerator

from nemo_curator import AsyncOpenAIClient, ScoreFilter, SemDedup, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import WordCountFilter
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
TEMP_DIR = os.path.join(SCRIPT_DIR_PATH, "_temp")
CONFIG_DIR = os.path.join(SCRIPT_DIR_PATH, "config")
DATASET_URL = (
    "https://huggingface.co/datasets/ymoslem/Law-StackExchange/resolve/main/law-stackexchange-questions-answers.json"
)


def pre_imports() -> None:
    import cudf  # noqa: F401


def random_split_rows(
    rows: list[Any], train_ratio: float, val_ratio: float, seed: int = 42
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Randomly splits a list of rows into training, validation, and test sets.

    Args:
        rows: The list of rows to be split.
        train_ratio: The ratio of rows to be allocated for training.
        val_ratio: The ratio of rows to be allocated for validation.
        seed: The seed value for random shuffling.

    Returns:
        A tuple containing the training, validation, and test sets.
    """
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    train_rows = rows[:train_size]
    val_rows = rows[train_size : train_size + val_size]
    test_rows = rows[train_size + val_size :]
    return train_rows, val_rows, test_rows


def download_and_convert_to_jsonl() -> str:
    """
    Downloads the Law Q&A dataset dataset and converts it to JSONL format.

    Returns:
        str: The path to the JSONL file.
    """
    download_dir = os.path.join(DATA_DIR, "raw", "downloads")
    splits_dir = os.path.join(DATA_DIR, "raw", "splits")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    # Download the dataset in raw format and convert it to JSONL.
    downloader = LawQADownloader(download_dir)
    raw_fp = downloader.download(DATASET_URL)

    iterator = LawQAIterator()
    rows = []

    for record in iterator.iterate(raw_fp):
        json_record = json.dumps(record, ensure_ascii=False)
        rows.append(json_record)

    # Randomly split the rows into train, validation, and test sets.
    train_rows, val_rows, test_rows = random_split_rows(rows, 0.8, 0.1)

    # Write the split rows to separate JSONL files.
    for split_name, split_rows in zip(
        ["train", "val", "test"],
        [train_rows, val_rows, test_rows],
        strict=False,
    ):
        split_fp = os.path.join(splits_dir, f"law-qa-{split_name}.jsonl")

        with open(split_fp, "w") as f:
            for row in split_rows:
                f.write(row + "\n")

    return (
        os.path.join(splits_dir, "law-qa-train.jsonl"),
        os.path.join(splits_dir, "law-qa-val.jsonl"),
        os.path.join(splits_dir, "law-qa-test.jsonl"),
    )


def semantic_dedupe(dataset: DocumentDataset) -> DocumentDataset:
    """
    Perform semantic deduplication on the given dataset.

    Args:
        dataset: The input DocumentDataset.

    Returns:
        The deduplicated DocumentDataset.
    """
    # Clean up the temporary directory to ensure everything is clean.
    if os.path.isdir(TEMP_DIR):
        os.system(f"rm -rf {TEMP_DIR}")  # noqa: S605

    semdedup_config = SemDedupConfig.from_yaml(
        os.path.join(CONFIG_DIR, "sem_dedup_config.yaml"),
    )
    expand_outdir_and_mkdir(semdedup_config.cache_dir)
    semdup = SemDedup(
        config=semdedup_config,
        input_column="text",
        id_column="id",
        perform_removal=True,
    )
    return semdup(dataset)


def run_curation_pipeline(
    args: argparse.Namespace,
    input_dir: str,
) -> DocumentDataset:
    """
    Run the curation pipeline on the dataset.

    Args:
        args: Command-line arguments.
        input_dir: The path to the uncurated JSONL file.

    Returns:
        The resulting dataset.
    """
    orig_dataset = DocumentDataset.read_json(input_dir, backend="pandas")
    dataset = orig_dataset

    cpu_curation_steps = Sequential(
        [
            #
            # Modifications
            #
            # Clean the HTML tags from all the records.
            Modify(CleanHTML(), text_field="title"),
            Modify(CleanHTML(), text_field="question"),
            Modify(CleanHTML(), text_field="answer"),
            # Unify the text encoding to Unicode.
            Modify(UnicodeReformatter(), text_field="title"),
            Modify(UnicodeReformatter(), text_field="question"),
            Modify(UnicodeReformatter(), text_field="answer"),
            #
            # Filtering
            #
            # Filter out records based on the question or answer word counts.
            ScoreFilter(
                WordCountFilter(min_words=50, max_words=500),
                text_field="question",
                score_type=int,
            ),
            ScoreFilter(
                WordCountFilter(min_words=50, max_words=500),
                text_field="answer",
                score_type=int,
            ),
            ScoreFilter(
                FilterLowScores(score_threshold=0),
                text_field="question_score",
                score_type=bool,
            ),
            ScoreFilter(
                FilterLowScores(score_threshold=0),
                text_field="answer_score",
                score_type=bool,
            ),
        ],
    )

    # Run the CPU curation steps.
    dataset = cpu_curation_steps(dataset)

    # Define and run the GPU curation steps.
    if args.device == "gpu":
        # Create a text field comprised of the title, question, and answer.
        # This field is used for finding semantically similar records and deduping them.
        dataset.df["text"] = dataset.df["title"] + "\n" + dataset.df["question"] + "\n" + dataset.df["answer"]
        dataset.df = dataset.df.to_backend("cudf")
        gpu_curation_steps = Sequential(
            [
                semantic_dedupe,
            ],
        )

        dataset = gpu_curation_steps(dataset)
        # Delete the text field as it is no longer needed.
        del dataset.df["text"]
        dataset.df = dataset.df.to_backend("pandas")

    dataset = dataset.persist()
    df = dataset.to_pandas()
    orig_len = len(orig_dataset.df)
    new_len = len(df)

    return df, orig_len, new_len


def run_pipeline(args: argparse.Namespace, jsonl_fp: str) -> str:
    """
    Run the curation pipeline.

    Args:
        args: The command-line arguments.
        jsonl_fp: The file path to the JSONL file.

    Returns:
        The file path to the final curated JSONL file.
    """
    # Disable synthetic data generation if the necessary arguments are not provided.
    if not args.synth_gen_endpoint:
        print(
            "No synthetic data generation endpoint provided. Skipping synthetic data generation.",
        )
        args.synth_gen_rounds = 0
    if not args.synth_gen_model:
        print(
            "No synthetic data generation model provided. Skipping synthetic data generation.",
        )
        args.synth_gen_rounds = 0
    if not args.api_key:
        print(
            "No synthetic data generation API key provided. Skipping synthetic data generation.",
        )
        args.synth_gen_rounds = 0

    if args.synth_gen_rounds:
        print(
            f"Using {args.synth_gen_endpoint}/{args.synth_gen_model} for synthetic data generation.",
        )

    synth_gen_ratio = args.synth_gen_ratio
    synth_gen_rounds = args.synth_gen_rounds
    synth_n_variants = args.synth_n_variants

    if synth_gen_ratio < 0 or synth_gen_ratio > 1:
        msg = "The synthetic generation ratio must be between 0 and 1."
        raise ValueError(msg)
    if synth_gen_rounds < 0:
        msg = "The number of synthetic generation rounds must be a non-negative integer."
        raise ValueError(msg)
    if synth_n_variants < 1:
        msg = "The number of synthetic variants must be a positive integer."
        raise ValueError(msg)

    backend = "cudf" if args.device == "gpu" else "pandas"

    out_dir_base = os.path.join(DATA_DIR, "curated")
    jsonl_filename = os.path.basename(jsonl_fp)

    # Create the synthetic data generator.
    llm_client = AsyncOpenAIClient(
        AsyncOpenAI(
            base_url=args.synth_gen_endpoint,
            api_key=args.api_key or "",
            timeout=args.api_timeout,
        ),
    )
    synth_gen = SyntheticGenerator(
        llm_client,
        sdg_model=args.synth_gen_model,
        sdg_model_kwargs={
            "top_p": 0.7,
            "max_tokens": 1024,
            "seed": 1234,
        },
        reward_model="nvidia/nemotron-4-340b-reward",
        n_variants=synth_n_variants,
    )

    with dask.config.set({"dataframe.backend": backend}):
        dask_client = get_client(**ArgumentHelper.parse_client_args(args))

        if args.device == "gpu":
            dask_client.run(pre_imports)

        print(f"Running the initial curation pipeline on '{jsonl_fp}'...")
        dataset_df, n_rows_before, n_rows_after = run_curation_pipeline(args, jsonl_fp)
        print(
            f"After the initial curation, the dataset has {n_rows_after} records (originally {n_rows_before}).",
        )

        for i in range(1, synth_gen_rounds + 1):
            print(
                "--------------------------------------------------------------------------------",
            )
            print(
                f"Running synthetic data generation -- round {i} (out of {synth_gen_rounds})...",
            )
            out_dir = out_dir_base + f"/round-{i}"
            os.makedirs(out_dir, exist_ok=True)
            # Save the base dataset to disk.
            dataset_df.to_json(
                f"{out_dir}/{jsonl_filename}",
                orient="records",
                lines=True,
            )

            #
            # Synthetic data generation
            #
            synth_prefix = f"{os.path.splitext(jsonl_filename)[0]}-synth-round-{i}"
            out_dir = synth_gen.run(dataset_df, out_dir, synth_prefix, synth_gen_ratio)
            #
            # Curation of the combined real and synthetic data
            #
            dataset_df, n_rows_before, n_rows_after = run_curation_pipeline(
                args,
                out_dir,
            )

            print(
                f"After round {i}, the dataset has {n_rows_after} records (originally {n_rows_before}).",
            )

        dask_client.cancel(dask_client.futures, force=True)
        dask_client.close()

    final_out_path = f"{out_dir_base}/final/{jsonl_filename}"
    os.makedirs(os.path.dirname(final_out_path), exist_ok=True)
    dataset_df.to_json(final_out_path, orient="records", lines=True)
    return final_out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser = ArgumentHelper(parser).add_distributed_args()
    parser.add_argument(
        "--synth-gen-endpoint",
        type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="The API endpoint to use for synthetic data generation. Any endpoint compatible with the OpenAI API can be used.",
    )
    parser.add_argument(
        "--synth-gen-model",
        type=str,
        default="nvidia/nemotron-4-340b-instruct",
        help="The model from the provided API endpoint to use for synthetic data generation. Leave blank to skip synthetic data generation.",
    )
    parser.add_argument(
        "--synth-gen-ratio",
        type=float,
        default=0.001,  # Use 0.1% of the real data for synthetic data generation to keep LLM calls low.
        help="The ratio of synthetic data to real data to generate. Synthetic data generation will be skipped if the value is 0.",
    )
    parser.add_argument(
        "--synth-gen-rounds",
        type=int,
        default=1,
        help="How many rounds of synthetic data generation to run. Will be ignored if --synth-gen-ratio is 0.",
    )
    parser.add_argument(
        "--synth-n-variants",
        type=int,
        default=1,
        help="The number of synthetic variants to generate for each record.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The API key to use for the synthetic data generation LLM client.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=120,
        help="The timeout value for API calls in seconds.",
    )

    args = parser.parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)
    # Don't use RMM to prevent models from running out of memory.
    args.set_torch_to_use_rmm = False

    # Prepare the download and JSONL directories.
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    train_fp, val_fp, test_fp = download_and_convert_to_jsonl()
    train_fp_curated = run_pipeline(args, train_fp)

    curated_dir = os.path.dirname(train_fp_curated)
    os.system(f"cp {val_fp} {curated_dir}")  # noqa: S605
    os.system(f"cp {test_fp} {curated_dir}")  # noqa: S605
    print(
        "--------------------------------------------------------------------------------",
    )
    print(f"Curated files are saved in '{curated_dir}'.")


if __name__ == "__main__":
    main()
