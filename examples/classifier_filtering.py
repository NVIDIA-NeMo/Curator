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
import random

import fasttext

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import FastTextQualityFilter
from nemo_curator.modifiers import FastTextLabelModifier
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper


def load_dataset(input_data_dir):
    files = list(get_all_files_paths_under(input_data_dir))
    raw_data = read_data(files, file_type="jsonl", backend="pandas", add_filename=True)
    dataset = DocumentDataset(raw_data)

    return dataset


def create_samples(data_path, label, num_samples):
    raw_dataset = load_dataset(data_path)
    label_quality = nc.Modify(FastTextLabelModifier(label))

    labeled_dataset = label_quality(raw_dataset)
    labeled_samples = labeled_dataset.df.sample(
        frac=num_samples / len(labeled_dataset.df)
    )

    return labeled_samples["text"].compute().values.tolist()


def main(args):
    # Params
    low_quality_data_path = "/path/to/low_quality"
    high_quality_data_path = "/path/to/high_quality"
    num_low_quality_samples = 1000
    num_high_quality_samples = 1000
    filtered_output = "/path/to/output"

    # Prepare samples for the classifier
    client = get_client(**ArgumentHelper.parse_client_args(args))
    low_quality_samples = create_samples(
        low_quality_data_path, "__label__lq", num_low_quality_samples
    )
    high_quality_samples = create_samples(
        high_quality_data_path, "__label__hq", num_high_quality_samples
    )

    train_samples = low_quality_samples + high_quality_samples
    random.shuffle(train_samples)
    train_file = "./fasttext.train"
    model_path = "./fasttext_model.bin"
    with open(train_file, "w") as f:
        for sample in train_samples:
            f.write(sample)
            f.write("\n")

    # Train fastText classifier
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.01,
        dim=100,
        epoch=5,
        wordNgrams=2,
    )
    model.save_model(model_path)

    # Filter data
    target_dataset = load_dataset(low_quality_data_path)
    filter_pipeline = nc.ScoreFilter(
        FastTextQualityFilter(model_path),
        score_field="quality_score",
        score_type=float,
    )
    filtered_dataset = filter_pipeline(target_dataset)

    # Write filtered dataset
    write_to_disk(filtered_dataset.df, filtered_output, write_to_filename=True)


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
