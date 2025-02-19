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

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import FastTextLangId
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import (
    get_all_files_paths_under,
    separate_by_metadata,
)
from nemo_curator.utils.script_utils import ArgumentHelper


def load_dataset(input_data_dir):
    files = list(get_all_files_paths_under(input_data_dir, keep_extensions="jsonl"))
    raw_data = read_data(files, file_type="jsonl", backend="pandas", add_filename=True)
    dataset = DocumentDataset(raw_data)

    return dataset


def main(args):
    # Params
    multilingual_data_path = "/path/to/multilingual"
    language_separated_output_path = "/path/to/lang_separated"
    cleaned_data_output_path = "/path/to/cleaned"

    # Download a fastText language identification model
    # and see a list of supported languages here:
    # https://fasttext.cc/docs/en/language-identification.html
    model_path = "/path/to/model.bin"
    language_field = "language"

    # Prepare samples for the classifier
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Filter data
    multilingual_dataset = load_dataset(multilingual_data_path)
    language_id_pipeline = nc.ScoreFilter(
        FastTextLangId(model_path), score_field=language_field, score_type="object"
    )
    filtered_dataset = language_id_pipeline(multilingual_dataset)

    # Remove the language score
    filtered_dataset.df[language_field] = filtered_dataset.df[language_field].apply(
        lambda score: score[1], meta=(None, str)
    )

    # Split the dataset by language
    language_stats = separate_by_metadata(
        filtered_dataset.df,
        language_separated_output_path,
        metadata_field=language_field,
    ).compute()


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
