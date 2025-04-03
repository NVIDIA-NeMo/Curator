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
import importlib
import os
import shutil
import time
from typing import Any, List

import numpy as np
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from retriever_evalset_generator import RetrieverEvalSetGenerator

from nemo_curator import AsyncOpenAIClient, ScoreFilter, Sequential, get_client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    AnswerabilityFilter,
    EasinessFilter,
    NonAlphaNumericFilter,
)
from nemo_curator.modules.filter import Score, ScoreFilter
from nemo_curator.utils.file_utils import get_all_files_paths_under

config = importlib.import_module(
    "tutorials.nemo-retriever-synthetic-data-generation.config.config"
)
RetrieverEvalSDGConfig = config.RetrieverEvalSDGConfig


def get_pipeline(args: Any) -> Any:

    cfg = RetrieverEvalSDGConfig.from_yaml(args.pipeline_config)
    # update api_key from input args
    cfg.api_key = args.api_key

    if args.pipeline_type == "generate":
        sdg_pipeline = Sequential(
            [
                RetrieverEvalSetGenerator(cfg),
            ]
        )
    else:
        sdg_pipeline = None

    filters = []

    if args.pipeline_type == "filter":
        if cfg.easiness_filter:
            filters.append(
                ScoreFilter(
                    EasinessFilter(
                        cfg.base_url,
                        cfg.api_key,
                        cfg.easiness_filter,
                        cfg.percentile,
                        cfg.truncate,
                        cfg.batch_size,
                    ),
                    text_field=["text", "question"],
                    score_field="easiness_scores",
                )
            )
        if cfg.answerability_filter:
            filters.append(
                ScoreFilter(
                    AnswerabilityFilter(
                        cfg.base_url,
                        cfg.api_key,
                        cfg.answerability_filter,
                        cfg.answerability_system_prompt,
                        cfg.answerability_user_prompt_template,
                        cfg.num_criteria,
                    ),
                    text_field=["text", "question"],
                    score_field="answerability_scores",
                )
            )
    if filters:
        filtering_pipeline = Sequential(filters)
    else:
        filtering_pipeline = None

    return sdg_pipeline, filtering_pipeline


def write_to_beir(args: Any, dataset: DocumentDataset, input_dataset: DocumentDataset):

    df = dataset.df
    df = df.compute()

    save_dir = os.path.join(args.output_dir, "beir")
    qrels_save_dir = os.path.join(args.output_dir, "beir", "qrels")

    os.makedirs(save_dir)
    os.makedirs(qrels_save_dir)

    corpus_save_path = os.path.join(args.output_dir, "beir", "corpus.jsonl")
    queries_save_path = os.path.join(args.output_dir, "beir", "queries.jsonl")
    df[["question-id", "question"]].rename(
        columns={"question-id": "_id", "question": "text"}
    ).to_json(queries_save_path, lines=True, orient="records")

    df[["question-id", "_id", "score"]].rename(
        columns={"question-id": "query-id", "_id": "corpus-id"}
    ).to_csv(os.path.join(qrels_save_dir, "test.tsv"), sep="\t", index=False)

    if args.pipeline_type == "filter":
        input_df = input_dataset.df.compute()
        input_df = input_df.groupby("_id").agg({"text": set}).reset_index()
        input_df["text"] = input_df["text"].map(lambda x: x.pop())
        input_df[["_id", "text"]].to_json(
            corpus_save_path, lines=True, orient="records"
        )
    elif args.pipeline_type == "generate":
        df = df.groupby("_id").agg({"text": set}).reset_index()
        df["text"] = df["text"].map(lambda x: x.pop())
        df[["_id", "text"]].to_json(corpus_save_path, lines=True, orient="records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Input directory containing jsonl files that have the document/text chunks for query & answer generation",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="jsonl",
        help="The input files must be in jsonl format",
    )
    parser.add_argument(
        "--pipeline-config",
        type=str,
        default="",
        help="Pipeline configuartion yaml file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output dir for generated data",
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
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of partitions for parallel processing of data.",
    )
    parser.add_argument(
        "--pipeline-type",
        type=str,
        default="generate",
        help="Choices: 1. 'generate', 2. 'filter'",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="jsonl",
        help="Save format choices 1. beir 2. jsonl",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif not any(os.scandir(args.output_dir)):
        print("Provided directory exists but is empty, using the empty directory")
    else:
        raise ValueError("Output directory exists already, use a new directory!")

    if args.input_format == "jsonl":
        if args.pipeline_type == "filter":
            input_files = get_all_files_paths_under(
                args.input_dir, keep_extensions="part"
            )
        elif args.pipeline_type == "generate":
            input_files = get_all_files_paths_under(
                args.input_dir, keep_extensions="jsonl"
            )
        else:
            raise ValueError(
                "Error only two pipelines supported: 'generate' & 'filter'"
            )

        input_dataset = DocumentDataset.read_json(input_files)
    else:
        raise ValueError("Error: Only jsonl format supported")

    if args.n_partitions:
        ddf = input_dataset.df
        n_data = len(ddf)
        if args.n_partitions < n_data:
            ddf = ddf.repartition(npartitions=args.n_partitions)
            input_dataset = DocumentDataset(ddf)
        else:
            print("Number of partitions greater than data size, using 1 partition")

    sdg_pipeline, filtering_pipeline = get_pipeline(args)

    if sdg_pipeline:
        print("Generating data ...")
        st_time = time.time()
        generated_dataset = sdg_pipeline(input_dataset)
        generated_dataset.persist()

        if args.save_format == "jsonl":
            print("Writing all generated data to disk ...")
            all_save_dir = os.path.join(args.output_dir, "jsonl")
            os.makedirs(all_save_dir)
            generated_dataset.to_json(all_save_dir)

        # saving in beir format
        if args.save_format == "beir":
            print("Write all data in beir format")
            write_to_beir(args, generated_dataset, input_dataset)

        print("Time taken to generate data = {:.2f} s".format(time.time() - st_time))

    if filtering_pipeline:
        print("Filtering data ...")
        st_time = time.time()
        filtered_dataset = filtering_pipeline(input_dataset)
        filtered_dataset.persist()

        if args.save_format == "jsonl":
            print("Writing filtered data to disk ...")
            all_save_dir = os.path.join(args.output_dir, "jsonl")
            os.makedirs(all_save_dir)
            filtered_dataset.to_json(all_save_dir)

        if args.save_format == "beir":
            print("Writing filtered data in beir format")
            # saving in beir format
            write_to_beir(args, filtered_dataset, input_dataset)

        print("Time taken to filter data = {:.2f} s".format(time.time() - st_time))

    print("RUN complete!")
    print("------------------------")


if __name__ == "__main__":
    dask_client = get_client()
    main()
