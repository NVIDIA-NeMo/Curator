# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from .constants import WIKIMEDIA_REQUEST_HEADERS

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

        raw_dump_data = requests.get(
            wiki_latest_dump_status,
            headers=WIKIMEDIA_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if raw_dump_data.status_code == requests.codes.not_found:
            logger.warning(f"Dump status is not available at {wiki_latest_dump_status}")
            return None
        raw_dump_data.raise_for_status()
        try:
            dump_data = json.loads(raw_dump_data.content)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Unable to load dump data for {wiki_latest_dump_status}: {e}")
            return None
        return dump_data

    def _get_wikipedia_urls(self) -> list[str]:  # noqa: C901, PLR0912, PLR0915
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
            raw_wiki_index = requests.get(
                wiki_index_url,
                headers=WIKIMEDIA_REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            raw_wiki_index.raise_for_status()
            wiki_index = raw_wiki_index.content.decode("utf-8")
            wiki_index_parsed = BeautifulSoup(wiki_index, "lxml")

            # Extract and sort date directories instead of relying on link order or
            # assuming that "latest/" is the final link in the index.
            dump_dates = {
                link.text.strip("/")
                for link in wiki_index_parsed.find_all("a")
                if len(link.text.strip("/")) == 8 and link.text.strip("/").isdigit()  # noqa: PLR2004
            }
            dump_data = None
            for candidate_date in sorted(dump_dates, reverse=True):
                candidate_dump_date = f"{candidate_date}/"
                try:
                    candidate_dump_data = self._get_data_for_dump(candidate_dump_date, wiki_index_url)
                except requests.HTTPError as e:
                    status_code = e.response.status_code if e.response is not None else None
                    if status_code is not None and (
                        status_code in (requests.codes.request_timeout, requests.codes.too_many_requests)
                        or status_code >= requests.codes.internal_server_error
                    ):
                        logger.warning(
                            f"Unable to load dump data for {candidate_date} due to HTTP {status_code}; "
                            "trying next dump"
                        )
                        continue
                    raise
                except requests.RequestException as e:
                    logger.warning(
                        f"Unable to load dump data for {candidate_date} due to {type(e).__name__}: {e}; "
                        "trying next dump"
                    )
                    continue
                if candidate_dump_data is None:
                    logger.warning(f"Cannot load dump data for {candidate_date}")
                    continue

                if candidate_dump_data.get("jobs", {}).get("articlesmultistreamdump", {}).get("status") == "done":
                    dump_date = candidate_dump_date
                    dump_data = candidate_dump_data
                    break

                logger.warning(f"Dump {candidate_date} is not finished, trying next dump")

            if dump_date is None or dump_data is None:
                error_msg = f"Unable to find a completed Wikipedia dump at {wiki_index_url}"
                raise ValueError(error_msg)

            logger.info(f"Found latest dump date: {dump_date[:-1]}")
        else:
            # A trailing / is needed for the URL
            dump_date = dump_date.rstrip("/") + "/"
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
