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

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.tasks import (
    ANLI,
    CB,
    PIQA,
    RTE,
    WSC,
    ArcChallenge,
    ArcEasy,
    BoolQ,
    Copa,
    Drop,
    MultiRC,
    OpenBookQA,
    Quac,
    Race,
    Record,
    Squad,
    TriviaQA,
    WebQA,
    WiC,
    Winogrande,
)
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper


def load_dataset(input_data_dir):
    files = list(get_all_files_paths_under(input_data_dir))
    raw_data = read_data(files, file_type="jsonl", backend="pandas", add_filename=True)
    dataset = DocumentDataset(raw_data)

    return dataset


def main(args):
    # Params
    contaminated_dataset_path = "/path/to/input"
    decontaminated_output_path = "/path/to/output"

    downstream_tasks = [
        Winogrande(),
        Squad(),
        TriviaQA(),
        Quac(),
        WebQA(),
        Race(),
        Drop(),
        WiC(),
        PIQA(),
        ArcEasy(),
        ArcChallenge(),
        OpenBookQA(),
        BoolQ(),
        Copa(),
        RTE(),
        MultiRC(),
        WSC(),
        CB(),
        ANLI(),
        Record(),
    ]

    # Prepare samples for the classifier
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Filter data
    target_dataset = load_dataset(contaminated_dataset_path)
    decontaminator = nc.TaskDecontamination(downstream_tasks)
    decontaminated_dataset = decontaminator(target_dataset)

    # Write filtered dataset
    write_to_disk(
        decontaminated_dataset.df, decontaminated_output_path, write_to_filename=True
    )


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    return ArgumentHelper(parser).add_distributed_args()


if __name__ == "__main__":
    main(attach_args().parse_args())
