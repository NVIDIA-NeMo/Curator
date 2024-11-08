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

from nemo_curator.download.doc_builder import batch_download, download_and_extract
from nemo_curator.utils.config_utils import build_downloader
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
)
from nemo_curator.utils.script_utils import ArgumentHelper


def read_urls(file_path):
    with open(file_path, "r") as fp:
        urls = fp.readlines()
    return [url.strip() for url in urls]


def main(args):
    client = get_client(**ArgumentHelper.parse_client_args(args))

    if args.input_url_file:
        urls = read_urls(args.input_url_file)
        outdir = os.path.abspath(os.path.expanduser(args.output_json_dir))
        output_paths = list(
            map(lambda url: os.path.join(outdir, url.split("/")[-1] + ".jsonl"), urls)
        )
    elif args.input_data_dir:
        # If input_data_dir is specified, we operate in extraction only mode.
        urls = get_all_files_paths_under(args.input_data_dir)
        output_paths = urls
    else:
        raise ValueError(
            "One of --input-url-file or --input-data-dir must be specified"
        )

    expand_outdir_and_mkdir(args.output_json_dir)
    if args.output_download_dir:
        raw_download_dir = args.output_download_dir
    else:
        raw_download_dir = os.path.join(args.output_json_dir, "downloads")

    downloader, iterator, extractor, output_format = build_downloader(
        args.builder_config_file, default_download_dir=raw_download_dir
    )

    if args.download_only:
        output_paths = batch_download(urls, downloader)
        print(f"{len(output_paths)} were downloaded")
        return

    dataset = download_and_extract(
        urls,
        output_paths,
        downloader,
        iterator,
        extractor,
        output_format,
        keep_raw_download=args.keep_downloaded_files,
        force_download=args.overwrite_existing_json,
        input_meta=args.input_meta,
    )

    # Sample to trigger the dask computation
    sample = dataset.df.sample(frac=10 / len(dataset)).compute()


def attach_args(
    parser=argparse.ArgumentParser(
        """
Takes an input list of URLs, downloads the data,
and then extracts the text from the downloaded data. Using
the --builder-config-file argument, users must provide a YAML file
that points to implementations of the downloader, iterator, and extractor
classes that will be used to construct the documents that will "
"make up the dataset. Examples of these config files for the "
"Common Crawl, Wikipedia and ArXiv datasets can be found in the root "
"config directory of this repository.

In the case that users have data that have been pre-downloaded,
this utility can also be used for "extraction-only" purposes.
For this scenario, users should either provide a valid path
to the --input-data-dir/--input-local-data-dir directories
(depending on if the data are globally or locally available to each
MPI rank). Additionally, the downloader class should be implemented
such that it simply returns the pre-downloaded file.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_input_data_dir(help="Path to input data directory.")
    argumentHelper.add_arg_input_meta()
    argumentHelper.add_distributed_args()
    parser.add_argument(
        "--builder-config-file",
        type=str,
        required=True,
        help="YAML file that contains paths to implementations of a downloader, "
        "iterator, and extractor that will be used in this program "
        "to build the documents that make up the output dataset.",
    )
    ArgumentHelper.attach_bool_arg(
        parser,
        "download-only",
        help="Specify this flag if you desire to only download the data "
        "files and not extract text from the downloaded files.",
    )
    parser.add_argument(
        "--input-url-file",
        type=str,
        default=None,
        help="Input directory consisting of .jsonl files that are accessible "
        "to all nodes. Use this for a distributed file system.",
    )
    ArgumentHelper.attach_bool_arg(
        parser,
        "keep-downloaded-files",
        help="If this flag is set to true, the downloaded data files "
        "will be kept on disk and not removed after extraction.",
    )
    parser.add_argument(
        "--output-download-dir",
        type=str,
        default=None,
        help="The directory to where data files will be written "
        'in "download-only" mode. Specify this argument only when '
        "the --download-only flag is specified.",
    )
    parser.add_argument(
        "--output-json-dir",
        type=str,
        default=None,
        help="Output directory to store the extracted text in JSONL files.",
    )
    ArgumentHelper.attach_bool_arg(
        parser,
        "overwrite-existing-json",
        help="If this flag is specified, then the JSON data will be "
        "overwritten if downloading from the the same file.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())
