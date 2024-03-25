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
import time

from nemo_curator import DomainClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client


def main(args):
    global_st = time.time()

    labels = [
        "Adult",
        "Arts_and_Entertainment",
        "Autos_and_Vehicles",
        "Beauty_and_Fitness",
        "Books_and_Literature",
        "Business_and_Industrial",
        "Computers_and_Electronics",
        "Finance",
        "Food_and_Drink",
        "Games",
        "Health",
        "Hobbies_and_Leisure",
        "Home_and_Garden",
        "Internet_and_Telecom",
        "Jobs_and_Education",
        "Law_and_Government",
        "News",
        "Online_Communities",
        "People_and_Society",
        "Pets_and_Animals",
        "Real_Estate",
        "Science",
        "Sensitive_Subjects",
        "Shopping",
        "Sports",
        "Travel_and_Transportation",
    ]

    model_file_name = "/path/to/pytorch_model_file.pth"

    # Input can be a string or list
    input_file_path = "/path/to/data"
    output_file_path = "./"

    client = get_client(args, cluster_type=args.device)

    input_dataset = DocumentDataset.read_json(
        input_file_path, backend="cudf", add_filename=True
    )

    domain_classifier = DomainClassifier(
        model_file_name=model_file_name,
        labels=labels,
        filter_by=["Games", "Sports"],
    )
    result_dataset = domain_classifier(dataset=input_dataset)

    result_dataset.to_json(output_file_dir=output_file_path, write_to_filename=True)

    global_et = time.time()
    print(
        f"Total time taken for domain classifier inference: {global_et-global_st} s",
        flush=True,
    )

    client.close()


def attach_args(
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ),
):
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=None,
        help="Address to the scheduler of a created dask cluster. If not provided"
        "a single node LocalCUDACluster will be started.",
    )
    parser.add_argument(
        "--scheduler-file",
        type=str,
        default=None,
        help="Path to the scheduler file of a created dask cluster. If not provided"
        " a single node LocalCUDACluster will be started.",
    )
    parser.add_argument(
        "--nvlink-only",
        action="store_true",
        help="Start a local cluster with only NVLink enabled."
        "Only applicable when protocol=ucx and no scheduler file/address is specified",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="ucx",
        help="Protcol to use for dask cluster"
        "Note: This only applies to the localCUDACluster. If providing an user created "
        "cluster refer to"
        "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol",  # noqa: E501
    )
    parser.add_argument(
        "--rmm-pool-size",
        type=str,
        default="14GB",
        help="Initial pool size to use for the RMM Pool Memory allocator"
        "Note: This only applies to the localCUDACluster. If providing an user created "
        "cluster refer to"
        "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size",  # noqa: E501
    )
    parser.add_argument("--enable-spilling", action="store_true")
    parser.add_argument("--set-torch-to-use-rmm", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Device to run the script on. Either 'cpu' or 'gpu'.",
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
