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
import os
import shutil
from typing import Any

from docbuilder import TinyStoriesDownloader
from filters import IncompleteStoryFilter
from helpers import write_jsonl
from modifiers import QuotationUnifier

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import RepeatingTopNGramsFilter, WordCountFilter
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules import ExactDuplicates
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
JSONL_ROOT_DIR = os.path.join(DATA_DIR, "jsonl")
# The TinyStories dataset is split into two files, one for training and one for validation.
# For the purposes of this tutorial, we will use the smaller validation file to demonstrate the curation pipeline.
TINY_STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"


def download_and_convert_to_jsonl() -> str:
    """
    Downloads the TinyStories dataset and converts it to JSONL format.

    Returns:
        str: The directory path where the JSONL files are saved.
    """

    # Download the TinyStories dataset.
    downloader = TinyStoriesDownloader(DATA_DIR)
    tinystories_val_fp = downloader.download(TINY_STORIES_URL)

    # Convert to JSONL files.
    jsonl_dir = os.path.join(JSONL_ROOT_DIR, "val")
    write_jsonl(tinystories_val_fp, jsonl_dir)

    return jsonl_dir


def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    """
    Cleans and unifies the given dataset using a set of predefined cleaners.

    Args:
        dataset (DocumentDataset): The dataset to be cleaned and unified.

    Returns:
        DocumentDataset: The cleaned and unified dataset.
    """
    cleaners = Sequential(
        [
            # Unify all the quotation marks
            Modify(QuotationUnifier()),
            # Unify all unicode
            Modify(UnicodeReformatter()),
        ]
    )
    return cleaners(dataset)


def filter_dataset(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given dataset based on various criteria.

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            ScoreFilter(
                WordCountFilter(min_words=80),
                text_field="text",
                score_field="word_count",
                score_type=int,
            ),
            ScoreFilter(IncompleteStoryFilter(), text_field="text", score_type=bool),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
                text_field="text",
                score_type=float,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def redact_pii(dataset: DocumentDataset) -> DocumentDataset:
    """
    Redacts personally identifiable information (PII) from a given dataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents with PII.

    Returns:
        DocumentDataset: The redacted dataset with PII replaced by a generic value.
    """
    redactor = Modify(
        PiiModifier(
            supported_entities=["PERSON"],
            anonymize_action="replace",
            device="cpu",
        ),
    )
    return redactor(dataset)


def dedupe(dataset: DocumentDataset) -> DocumentDataset:
    """
    Remove exact duplicates from the given DocumentDataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents.

    Returns:
        DocumentDataset: The deduplicated dataset.
    """
    deduplicator = ExactDuplicates(id_field="id", text_field="text", hash_method="md5")
    # Find the duplicates
    duplicates = deduplicator(dataset)
    docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")]
    )
    # Remove the duplicates using their IDs.
    duplicate_ids = list(docs_to_remove.compute().id)
    dataset_df = dataset.df
    deduped = dataset_df[~dataset_df.id.isin(duplicate_ids)]
    return DocumentDataset(deduped)


def run_curation_pipeline(args: Any, jsonl_dir: str) -> None:
    """
    Run the curation pipeline on the TinyStories dataset.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(f"Running curation pipeline on '{jsonl_dir}'...")
    files = [
        fp
        for fp in get_all_files_paths_under(jsonl_dir, recurse_subdirectories=False)
        if fp.endswith(".jsonl")
    ]
    print("Reading the data...")
    orig_dataset = DocumentDataset.read_json(files, add_filename=True)
    dataset = orig_dataset

    curation_steps = Sequential(
        [
            clean_and_unify,
            filter_dataset,
            dedupe,
            redact_pii,
        ]
    )
    dataset = curation_steps(dataset)
    print("Executing the pipeline...")
    dataset = dataset.persist()

    print(f"Original dataset length: {len(orig_dataset.df)}")
    print(f"After dataprep: {len(dataset.df)}")
    print("Writing the results to disk...")

    # Overwrite existing files in the curated directory.
    out_path = os.path.join(jsonl_dir, "curated")

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    dataset.to_json(out_path, write_to_filename=True)
    client.close()


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 4)

    # Prepare the download and JSONL directories.
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.isdir(JSONL_ROOT_DIR):
        os.makedirs(JSONL_ROOT_DIR)

    jsonl_val_dir = download_and_convert_to_jsonl()

    run_curation_pipeline(args, jsonl_val_dir)


if __name__ == "__main__":
    main()
