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

import os
import time
import warnings

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

from nemo_curator.classifiers import PromptTaskComplexityClassifier
from nemo_curator.datasets import DocumentDataset

# Get relevant args
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_remaining_files
from nemo_curator.utils.script_utils import ArgumentHelper

warnings.filterwarnings("ignore")


def main():
    args = ArgumentHelper.parse_distributed_classifier_args(
        description="Run prompt task and complexity classifier inference."
    ).parse_args()
    print(f"Arguments parsed = {args}", flush=True)
    client_args = ArgumentHelper.parse_client_args(args)
    client_args["cluster_type"] = "gpu"
    client = get_client(**client_args)
    print("Starting prompt task and complexity classifier inference", flush=True)
    global_st = time.time()
    files_per_run = len(client.scheduler_info()["workers"]) * 2

    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    # Some times jsonl files are stored as .json
    # So to handle that case we can pass the input_file_extension
    if args.input_file_extension is not None:
        input_file_extension = args.input_file_extension
    else:
        input_file_extension = args.input_file_type

    input_files = get_remaining_files(
        args.input_data_dir, args.output_data_dir, input_file_extension
    )
    print(f"Total input files {len(input_files)}", flush=True)

    if args.input_file_type == "pickle":
        add_filename = False
    else:
        add_filename = True

    prompt_task_complexity_classifier = PromptTaskComplexityClassifier(
        text_field=args.input_text_field,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        autocast=args.autocast,
        max_mem_gb=args.max_mem_gb_classifier,
    )

    for file_batch_id, i in enumerate(range(0, len(input_files), files_per_run)):
        batch_st = time.time()
        current_batch_files = input_files[i : i + files_per_run]
        print(
            f"File Batch ID {file_batch_id}: total input files {len(current_batch_files)}",
            flush=True,
        )
        df = read_data(
            input_files=current_batch_files,
            file_type=args.input_file_type,
            add_filename=add_filename,
        )
        df = prompt_task_complexity_classifier(DocumentDataset(df)).df
        print(f"Total input Dask DataFrame partitions {df.npartitions}", flush=True)

        write_to_disk(
            df=df,
            output_path=args.output_data_dir,
            write_to_filename=add_filename,
            output_type=args.output_file_type,
        )
        batch_et = time.time()
        print(
            f"File Batch ID {file_batch_id}: completed in {batch_et-batch_st} seconds",
            flush=True,
        )

    global_et = time.time()
    print(
        f"Total time taken for prompt task and complexity classifier inference: {global_et-global_st} s",
        flush=True,
    )
    client.close()


def console_script():
    main()


if __name__ == "__main__":
    console_script()
