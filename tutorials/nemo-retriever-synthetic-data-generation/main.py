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
from typing import Any, List

from retriever_evalset_generator import RetrieverEvalSetGenerator
from tqdm.dask import TqdmCallback

from config.config import RetrieverEvalSDGConfig
from nemo_curator import AsyncOpenAIClient, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import AnswerabilityFilter, EasinessFilter
from nemo_curator.modules.filter import Score, ScoreFilter


def get_pipeline(args: Any) -> Any:

    cfg = RetrieverEvalSDGConfig.from_yaml(args.pipeline_config)
    # update api_key from input args
    cfg.api_key = args.api_key

    sdg_pipeline = Sequential(
        [
            RetrieverEvalSetGenerator(cfg),
        ]
    )
    filters = []
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


def write_to_beir(args: Any, dataset: DocumentDataset, filtered: bool = False):

    df = dataset.df
    df = df.compute()
    if filtered:
        save_dir = os.path.join(args.output_dir, "beir", "filtered")
        qrels_save_dir = os.path.join(args.output_dir, "beir", "filtered", "qrels")
        corpus_save_path = os.path.join(
            args.output_dir, "beir", "filtered", "corpus.jsonl"
        )
        queries_save_path = os.path.join(
            args.output_dir, "beir", "filtered", "queries.jsonl"
        )
    else:
        save_dir = os.path.join(args.output_dir, "beir", "all")
        qrels_save_dir = os.path.join(args.output_dir, "beir", "all", "qrels")
        corpus_save_path = os.path.join(args.output_dir, "beir", "all", "corpus.jsonl")
        queries_save_path = os.path.join(
            args.output_dir, "beir", "all", "queries.jsonl"
        )

    os.makedirs(save_dir)
    os.makedirs(qrels_save_dir)

    df[["question-id", "question"]].rename(
        columns={"question-id": "_id", "question": "text"}
    ).to_json(queries_save_path, lines=True, orient="records")

    if filtered:
        corpus_file_path = os.path.join(args.output_dir, "beir", "all", "corpus.jsonl")
        if os.path.exists(corpus_file_path):
            shutil.copy(corpus_file_path, corpus_save_path)
        else:
            raise ValueError("Generate data first")
    else:
        df[["_id", "text"]].to_json(corpus_save_path, lines=True, orient="records")

    df[["question-id", "_id", "score"]].rename(
        columns={"question-id": "query-id", "_id": "corpus-id"}
    ).to_csv(os.path.join(qrels_save_dir, "test.tsv"), sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help="File path of input file containing document chunks for synthetic data generation",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="rawdoc",
        help="The synthetic data generation framework supports two input formats rawdoc or squad.",
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
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        raise ValueError("Output directory exists already, use a new directory!")

    if args.input_format == "rawdoc":
        input_dataset = DocumentDataset.read_json(args.input_file)
    else:
        raise ValueError("Error: Only rawdoc format supported")

    sdg_pipeline, filtering_pipeline = get_pipeline(args)

    print("Generating data ...")
    with TqdmCallback(desc="apply"):
        generated_dataset = sdg_pipeline(input_dataset)
        generated_dataset.persist()
    print("Writing all generated data to disk ...")
    # saving in beir format
    write_to_beir(args, generated_dataset, filtered=False)

    print("Filtering data ...")
    with TqdmCallback(desc="apply"):
        filtered_dataset = filtering_pipeline(generated_dataset)
        filtered_dataset.persist()
    print("Writing filtered data to disk ...")
    # saving in beir format
    write_to_beir(args, filtered_dataset, filtered=True)

    # saving in jsonl format
    generated_dataset.to_json(
        os.path.join(args.output_dir, "jsonl", "all_generated_data.jsonl")
    )


if __name__ == "__main__":
    main()
