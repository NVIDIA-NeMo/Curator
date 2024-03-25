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

from nemo_curator.utils.download_utils import get_common_crawl_urls
from nemo_curator.utils.script_utils import attach_bool_arg


def main(args):
    urls = get_common_crawl_urls(
        args.starting_snapshot,
        args.ending_snapshot,
        data_domain_prefix=args.cc_data_domain_prefix,
        index_prefix=args.cc_index_prefix,
        news=args.cc_news,
    )

    with open(args.output_warc_url_file, "w") as fp:
        for url in urls:
            fp.write(url)
            fp.write("\n")


def attach_args(
    parser=argparse.ArgumentParser(
        """
Pulls URLs of WARC files stored within the common crawl data repository
and writes them to file so that they can be used to subsequently
download the WARC files.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    parser.add_argument(
        "--cc-data-domain-prefix",
        type=str,
        default="https://data.commoncrawl.org",
        help="The prefix that will be prepended to each WARC "
        "file to create the URL. By default this value is "
        " 'https://data.commoncrawl.org'",
    )
    parser.add_argument(
        "--cc-index-prefix",
        type=str,
        default="https://index.commoncrawl.org",
        help="The prefix of the URL to the Common Crawl index. "
        "By default this value is 'https://index.commoncrawl.org'",
    )
    parser.add_argument(
        "--output-warc-url-file",
        type=str,
        default=None,
        required=True,
        help="The output file to which the WARC urls will be written",
    )
    parser.add_argument(
        "--starting-snapshot",
        type=str,
        default="2020-50",
        help="The starting snapshot to download. All WARC urls will be written "
        "between the dates specified by --starting-snapshot "
        "and --ending-snapshot. Snapshots must be specified by YYYY-WeekNumber "
        "(e.g., '2020-50' or '2021-04'). For the CC-NEWS dataset, "
        "(specified with the '--cc-news' flag) this changes to "
        "Year-Month (YYYY-MM)",
    )
    parser.add_argument(
        "--ending-snapshot",
        type=str,
        default="2020-50",
        help="The last snapshot for which WARC urls will be retrieved. "
        "Snapshots must be specified by YYYY-WeekNumber "
        "(e.g., '2020-50' or '2021-04')",
    )
    attach_bool_arg(
        parser,
        "cc-news",
        help_str="Specify --cc-news in order to download WARC URLs for "
        "the CC-NEWS dataset instead of the CC-MAIN datasets. If this "
        "is specified, then it is assumed that the format for the start "
        "and end snapshots is 'YYYY-MM' (Year-Month). All WARC URLs between "
        "the specified years and months will be download",
    )
    return parser


def console_script():
    main(attach_args().parse_args())
