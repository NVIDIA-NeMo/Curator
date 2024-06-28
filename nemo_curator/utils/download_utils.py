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

import json
import os
import subprocess
import zlib
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def get_main_warc_paths(
    snapshot_index,
    start_snapshot,
    end_snapshot,
    prefix="https://data.commoncrawl.org",
):
    beg_year, beg_week = list(map(int, start_snapshot.split("-")))
    end_year, end_week = list(map(int, end_snapshot.split("-")))
    start_date = datetime.fromisocalendar(beg_year, beg_week, 1)
    end_date = datetime.fromisocalendar(end_year, end_week, 1)

    if start_date > end_date:
        raise ValueError(
            f"Start snapshot '{start_snapshot}' is after end snapshot '{end_snapshot}'"
        )

    if beg_year < 2013 or end_year < 2013:
        print("Warning: Only snapshots after 2013 are supported by this script")

    total_prefix = urljoin(prefix, "crawl-data/CC-MAIN")

    warc_paths = []
    for snapshot in snapshot_index:
        date = list(map(int, snapshot["id"].split("-")[2:]))

        if len(date) == 2:
            year, week = date
        else:
            continue

        if year >= 2013:
            curr_date = datetime.fromisocalendar(year, week, 1)
            if curr_date >= start_date and curr_date <= end_date:
                warc_path = f"{total_prefix}-{year}-{week:02d}/warc.paths.gz"
                warc_paths.append(warc_path)

    return warc_paths


def get_news_warc_paths(
    start_date,
    end_date,
    prefix="https://data.commoncrawl.org",
):
    beg = datetime.strptime(start_date, "%Y-%m")
    end = datetime.strptime(end_date, "%Y-%m")

    # Get current year and month
    today = datetime.now()

    if start_date > end_date:
        raise ValueError(
            f"Start snapshot '{start_date}' is after end snapshot '{end_date}'"
        )

    if beg.year < 2016 or end.year > today.year:
        print(
            "Warning: WARC paths exist only from 2016-8 to "
            f"{today.year}-{today.month}"
        )
    total_prefix = urljoin(prefix, "crawl-data/CC-NEWS")

    # Generate all valid YYYY-MM strings in range
    dates = OrderedDict()
    for day in range((end - beg).days + 1):
        new_date = beg + timedelta(day)
        dates[(new_date.year, new_date.month)] = None

    dates = list(dates.keys())

    warc_paths = []
    for year, month in dates:
        warc_path = f"{total_prefix}/{year}/{month:02d}/warc.paths.gz"
        warc_paths.append(warc_path)

    return warc_paths


def get_common_crawl_snapshot_index(index_prefix):
    index_url = urljoin(index_prefix, "collinfo.json")
    index_response = requests.get(index_url)

    return json.loads(index_response.content)


def get_common_crawl_urls(
    starting_snapshot: str,
    ending_snapshot: str,
    data_domain_prefix="https://data.commoncrawl.org",
    index_prefix="https://index.commoncrawl.org",
    news=False,
) -> List[str]:
    """
    Retrieves the URLs for all the compressed WARC files between given Common Crawl snapshots

    Args:
    starting_snapshot: The first common crawl snapshot to include. Snapshots must be
        specified by YYYY-WeekNumber (e.g., '2020-50' or '2021-04'). For the CC-NEWS dataset,
        (specified with news=True flag) this changes to Year-Month (YYYY-MM).
    ending_snapshot: The last common crawl snapshot to include. Must be chronologically
        after the starting snapshot.
    data_domain_prefix: The prefix that will be prepended to each WARC file to create the URL.
    index_prefix: The prefix of the URL to the Common Crawl index.
    news: If True, gets WARC URLs for the CC-NEWS dataset instead of the CC-MAIN datasets.
        Also assumes that the format for the start and end snapshots is 'YYYY-MM' (Year-Month).
    """
    if news:
        warc_paths = get_news_warc_paths(
            starting_snapshot, ending_snapshot, prefix=data_domain_prefix
        )
    else:
        index = get_common_crawl_snapshot_index(index_prefix)
        warc_paths = get_main_warc_paths(
            index, starting_snapshot, ending_snapshot, prefix=data_domain_prefix
        )

    common_crawl_urls = []
    for path in warc_paths:
        try:
            response = requests.get(path.rstrip(), stream=True)
            data = zlib.decompress(response.content, zlib.MAX_WBITS | 32)
            for warc in data.decode("utf-8").split("\n"):
                if warc != "":
                    warc_url = urljoin(data_domain_prefix, warc)
                    common_crawl_urls.append(warc_url)
        except Exception as e:
            print(f"Could not get URLs for snapshot {path}")
            print(response.content)
            print(e)

    return common_crawl_urls


def get_wikipedia_urls(
    language="en",
    wikidumps_index_prefix="https://dumps.wikimedia.org",
    dump_date: Optional[str] = None,
) -> List[str]:
    """
    Retrieves all urls pointing to the latest Wikipedia dumps

    Args:
        language: Desired language of the Wikipedia dump.
        wikidumps_index_prefix: The base url for all wikipedia dumps
        dump_date: A string formatted as "YYYYMMDD" for the wikipedia dump to use.
          If None, latest dump is used.
    """
    wiki_index_url = urljoin(wikidumps_index_prefix, f"{language}wiki")
    if not dump_date:
        # First get the index
        raw_wiki_index = requests.get(wiki_index_url)
        wiki_index = raw_wiki_index.content.decode("utf-8")
        wiki_index_parsed = BeautifulSoup(wiki_index, "lxml")

        # Get all dumps available in the index
        dumps = wiki_index_parsed.find_all("a")
        dump_date = dumps[-2].text
    else:
        # A trailing / is needed for the url
        dump_date = dump_date + "/"

    # Get the json dump data
    wiki_latest_dump = urljoin(wiki_index_url + "/", dump_date)
    wiki_latest_dump_status = urljoin(wiki_latest_dump, "dumpstatus.json")
    raw_dump_data = requests.get(wiki_latest_dump_status)
    try:
        dump_data = json.loads(raw_dump_data.content)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"No wikipedia dump found for {dump_date[:-1]}")

    # Get all multistream files within the dump data
    wikipedia_urls = []
    for ifile in dump_data["jobs"]["articlesmultistreamdump"]["files"]:
        if "xml" in ifile:
            url = urljoin(wiki_latest_dump, ifile)
            wikipedia_urls.append(url)

    return wikipedia_urls


def get_arxiv_urls():
    command = "s5cmd --request-payer=requester ls s3://arxiv/src/ | grep '.tar'"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Unable to get arxiv urls: {result.stderr}")

    urls = result.stdout.split()[3::4]
    urls.sort()

    return urls
