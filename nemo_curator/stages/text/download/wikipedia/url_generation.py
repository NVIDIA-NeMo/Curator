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

import json
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from nemo_curator.stages.text.download import URLGenerator

# Request timeout in seconds
REQUEST_TIMEOUT = 30


@dataclass
class WikipediaUrlGenerator(URLGenerator):
    """Generates URLs for Wikipedia dump files."""

    language: str = "en"
    dump_date: str | None = None
    wikidumps_index_prefix: str = "https://dumps.wikimedia.org"

    def generate_urls(self) -> list[str]:
        """Generate Wikipedia dump URLs.

        Returns:
            List of URLs pointing to Wikipedia dump files
        """
        return self._get_wikipedia_urls()

    def _get_data_for_dump(self, dump_date: str, wiki_index_url: str) -> dict | None:
        """Get the JSON dump data for a given dump date. Returns None if the dump is not found."""
        wiki_latest_dump = urljoin(wiki_index_url + "/", dump_date)
        wiki_latest_dump_status = urljoin(wiki_latest_dump, "dumpstatus.json")

        raw_dump_data = requests.get(wiki_latest_dump_status, timeout=REQUEST_TIMEOUT)
        try:
            dump_data = json.loads(raw_dump_data.content)
        except json.JSONDecodeError as e:
            logger.warning(f"Unable to load dump data for {wiki_latest_dump_status}: {e}")
            return None
        return dump_data

    def _get_wikipedia_urls(self) -> list[str]:
        """
        Retrieves all URLs pointing to Wikipedia dumps for the specified language and date.

        Returns:
            List of URLs for Wikipedia dump files
        """
        wiki_index_url = urljoin(self.wikidumps_index_prefix, f"{self.language}wiki")

        dump_date = self.dump_date
        if not dump_date:
            # Get the latest dump date from the index
            logger.info(f"Fetching latest dump date from {wiki_index_url}")
            raw_wiki_index = requests.get(wiki_index_url, timeout=REQUEST_TIMEOUT)
            wiki_index = raw_wiki_index.content.decode("utf-8")
            wiki_index_parsed = BeautifulSoup(wiki_index, "lxml")

            # Get all dumps available in the index
            dumps = wiki_index_parsed.find_all("a")
            for dump in reversed(dumps[:-1]):
                if dump.text.strip("/").isdigit():
                    candidate_dump_date = dump.text
                    dump_data = self._get_data_for_dump(candidate_dump_date, wiki_index_url)
                    if dump_data is None:
                        logger.warning(f"Cannot load dump data for {candidate_dump_date[:-1]}")
                        continue

                    if dump_data["jobs"].get("articlesmultistreamdump", {}).get("status") == "done":
                        dump_date = candidate_dump_date
                        break
                    else:
                        logger.warning(f"Dump {candidate_dump_date[:-1]} is not finished, trying next dump")
                        continue

            logger.info(f"Found latest dump date: {dump_date[:-1]}")
        else:
            # A trailing / is needed for the URL
            dump_date = dump_date + "/"
            dump_data = self._get_data_for_dump(dump_date, wiki_index_url)
            if dump_data is None:
                error_msg = f"Unable to load dump data for {dump_date[:-1]}"
                raise ValueError(error_msg)
            if dump_data["jobs"]["articlesmultistreamdump"]["status"] != "done":
                error_msg = f"Dump {dump_date[:-1]} is not finished"
                raise ValueError(error_msg)

        wiki_latest_dump = urljoin(wiki_index_url + "/", dump_date)

        # Get all multistream files within the dump data
        wikipedia_urls = []
        for file_name in dump_data["jobs"]["articlesmultistreamdump"]["files"]:
            if "xml" in file_name:
                url = urljoin(wiki_latest_dump, file_name)
                wikipedia_urls.append(url)

        logger.info(f"Found {len(wikipedia_urls)} Wikipedia dump files")
        return wikipedia_urls
