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
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
from downloaders import (
    download_github_sources,
    download_pdf_sources,
    download_wikipedia_sources,
)
from utils import (
    CodeLineCountFilter,
    TextLineCountFilter,
    clean_and_unify,
    exact_dedupe,
    filter_code,
    filter_text,
    fuzzy_dedupe,
    redact_code,
    rm_dir,
    semantic_dedupe,
)

import nemo_curator as nc
from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import (
    get_all_files_paths_under,
    separate_by_metadata,
)
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
CONFIG_DIR = os.path.join(SCRIPT_DIR_PATH, "configs")


def download_sources(
    wikipedia_limit: Optional[int] = None,
    github_limit: Optional[int] = None,
    pdf_limit: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Downloads all the dataset sources and converts them to the JSONL format.
    Args:
        wikipedia_limit (int): Maximum number of wiki urls to be downloaded
        github_limit (int): Maximum number of github repos to be downloaded
        pdf_limit (int): Maximum number of pdf to be downloaded


    Returns:
        tuple: the list of text files and the list of code files.
    """

    wikipedia_dir = download_wikipedia_sources(
        "sources/wikipedia_urls.jsonl", limit=wikipedia_limit
    )
    github_dir = download_github_sources(
        "sources/github_repos.jsonl", limit=github_limit
    )
    pdf_dir = download_pdf_sources("sources/arxiv_urls.jsonl", limit=pdf_limit)

    wiki_files = get_all_files_paths_under(wikipedia_dir)
    code_files = get_all_files_paths_under(github_dir)
    pdf_files = get_all_files_paths_under(pdf_dir)

    text_files = wiki_files + pdf_files

    return text_files, code_files


def plot_data(orig_dataset: DocumentDataset, filename: str):
    """
    Plot histogram of different file types and corresponding sizes

    Args:
        dataset (DocumentDataset): Dataset
        filename (str): Name of the plot to be saved ('sample.png')
    Returns:
        None (saves the plotted file in current directory)
    """
    # visualize file types and sizes
    orig_df = orig_dataset.df.compute()
    orig_df = orig_df.reset_index()

    # Create a histogram for different file types -text
    fig, ax = plt.subplots(figsize=(10, 6))
    orig_df.groupby("file_extension")["size_in_bytes"].sum().plot(kind="bar", ax=ax)
    ax.set_xlabel("file_extension")
    ax.set_ylabel("size_in_bytes")
    ax.set_title("File Size Histogram by File Extension")

    # Save the histogram to a file
    fig.savefig(filename, bbox_inches="tight")


def run_curation_pipeline(args: Any, text_files: str, code_files: str) -> None:
    """
    Run the curation pipeline on the Wiki+Arxiv+Github datasets.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Define data curation steps for text and pdf files
    curation_steps_text = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                TextLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_text,
            exact_dedupe,
        ]
    )

    # Define data curation steps for code files
    curation_steps_code = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                CodeLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_code,
            exact_dedupe,
            redact_code,
        ]
    )

    orig_dataset_text = DocumentDataset.read_json(text_files, add_filename=True)
    orig_dataset_code = DocumentDataset.read_json(code_files, add_filename=True)

    # Create a histogram for different file types -text
    plot_data(orig_dataset_text, "file_size_histogram_txt.png")

    # Create a histogram for different file types - code
    plot_data(orig_dataset_code, "file_size_histogram_code.png")

    # create a field combining fields file type and line count
    orig_dataset_text.df["file_type_count"] = (
        orig_dataset_text.df["file_type"]
        + " : "
        + orig_dataset_text.df["line_count"].astype(str)
    )
    orig_dataset_code.df["file_type_count"] = (
        orig_dataset_code.df["file_type"]
        + " : "
        + orig_dataset_code.df["line_count"].astype(str)
    )

    print("Executing the curation pipeline...")
    dataset_text = curation_steps_text(orig_dataset_text)
    dataset_code = curation_steps_code(orig_dataset_code)

    print(f"Original dataset length for text files: {len(orig_dataset_text.df)}")
    print(f"After dataprep for text files: {len(dataset_text.df)}")
    print(f"Original dataset length for code files: {len(orig_dataset_code.df)}")
    print(f"After dataprep length for code files: {len(dataset_code.df)}")

    if args.device == "gpu":
        print("Executing the semantic dedupe pipeline...")
        gpu_dataset_text = DocumentDataset(dataset_text.df.to_backend("cudf"))
        gpu_dataset_code = DocumentDataset(dataset_code.df.to_backend("cudf"))
        sem_dedupe_config_yaml_path = os.path.join(
            CONFIG_DIR, "text_semantic_dedupe_config.yaml"
        )
        CACHE_DIR = os.path.join(SCRIPT_DIR_PATH, "cache", "semantic_dedupe", "text")
        rm_dir(CACHE_DIR)
        duplicates = semantic_dedupe(
            dataset=gpu_dataset_text,
            sem_dedupe_config_yaml_path=sem_dedupe_config_yaml_path,
            cache_dir=CACHE_DIR,
        )
        unique_ids = duplicates.df.to_backend("pandas").compute()["id"]
        semantic_dataset_text = DocumentDataset(
            gpu_dataset_text.df[gpu_dataset_text.df.id.isin(unique_ids)]
        )
        print(f"After semantic dedupe for text files: {len(semantic_dataset_text.df)}")

        print("Executing the fuzzy dedupe pipeline...")
        CACHE_DIR = os.path.join(SCRIPT_DIR_PATH, "cache", "fuzzy_dedupe", "text")
        rm_dir(CACHE_DIR)
        fuzzy_dataset_text = fuzzy_dedupe(
            dataset=semantic_dataset_text, cache=CACHE_DIR
        )
        CACHE_DIR = os.path.join(SCRIPT_DIR_PATH, "cache", "fuzzy_dedupe", "code")
        rm_dir(CACHE_DIR)
        fuzzy_dataset_code = fuzzy_dedupe(dataset=gpu_dataset_code, cache=CACHE_DIR)

        dataset_text.df = fuzzy_dataset_text.df.to_backend("pandas")
        dataset_code.df = fuzzy_dataset_code.df.to_backend("pandas")
        print(f"After fuzzy dedupe for text files: {len(dataset_text.df)}")
        print(f"After fuzzy dedupe: {len(dataset_code.df)}")

    final_dataset_text = dataset_text.persist()
    final_dataset_code = dataset_code.persist()

    print("Writing the results to disk...")

    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    final_dataset_text.to_json(out_path, write_to_filename=True)
    final_dataset_code.to_json(out_path, write_to_filename=True)

    print("Writing results to disk completed")

    # Split the dataset by file category and save curated files (optional - to create blended datasets)
    print("Split dataset by metadata")
    separated_data_text = separate_by_metadata(
        final_dataset_text.df, out_path, "category"
    ).compute()
    separated_data_code = separate_by_metadata(
        final_dataset_code.df, out_path, "category"
    ).compute()

    client.close()


def blend_and_shuffle(
    args: Any, dataset_paths: list, dataset_weights: list, target_size: int
) -> None:
    """
    Blend and shuffle curated data based on file paths for continued pre-training

    Args:
        args (Any): Command-line arguments.
        dataset_paths (list): List containing directory paths where the different JSONL files are stored.
        dataset_weights (list): List setting weights for each directory path
        target_size (int): Target number of data samples after blending
    """
    root_path = os.path.join(DATA_DIR, "curated")
    output_path = root_path + "/data_blended"
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Blend the datasets
    datasets = [DocumentDataset.read_json(path) for path in dataset_paths]
    blended_dataset = nc.blend_datasets(target_size, datasets, dataset_weights)

    shuffle = nc.Shuffle(seed=42)
    blended_dataset = shuffle(blended_dataset)

    # Save the blend
    blended_dataset.to_json(output_path)


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)
    print("Args: ", args)

    # Download all the sources and get the list of text and code files.
    text_files, code_files = download_sources(100, 100, 100)
    run_curation_pipeline(args, text_files, code_files)
    print("Data Curation completed")

    # blend and shuffle datasets
    root_path = os.path.join(DATA_DIR, "curated")
    dataset_paths = [
        root_path + "/CPP",
        root_path + "/VerilogVHDL",
        root_path + "/text",
        root_path + "/Python",
    ]
    dataset_weights = [1.0, 4.0, 4.0, 1.0]
    target_size = 20

    blend_and_shuffle(args, dataset_paths, dataset_weights, target_size)
    print("Data Blending completed")


if __name__ == "__main__":
    main()
