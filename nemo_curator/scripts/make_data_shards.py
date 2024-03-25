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

from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import reshard_jsonl
from nemo_curator.utils.script_utils import add_distributed_args


def main(args):
    client = get_client(args, args.device)

    reshard_jsonl(
        args.input_data_dir,
        args.output_resharded_dir,
        output_file_size=args.output_file_size,
        start_index=args.start_index,
        file_prefix=args.prefix,
    )


def attach_args(
    parser=argparse.ArgumentParser(
        """
Makes balanced text files of output size "--block-size" from
a directory of input files. The output files will be renamed
as output_dir/000.jsonl, output_dir/001.jsonl, ... etc. Users
may specify the desired number of output files and the block size
will be computed from this specified quantity.

The size of the input files must be larger than the specified
"--block-size"
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        required=True,
        help="Input directory consisting of .jsonl file(s)",
    )
    parser.add_argument(
        "--output-resharded-dir",
        type=str,
        default=None,
        required=True,
        help="Output directory to where the sharded " ".jsonl files will be written",
    )
    parser.add_argument(
        "--output-file-size",
        type=str,
        default="100M",
        help="Approximate size of output files. Must specify with a string and "
        "with the unit K, M or G for kilo, mega or gigabytes",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for naming the output files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to use to prepend to output file number",
    )

    parser = add_distributed_args(parser)

    return parser


def console_script():
    main(attach_args().parse_args())
