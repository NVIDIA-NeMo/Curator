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
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import cached_property
from urllib.parse import urljoin

import requests
from loguru import logger

from nemo_curator.stages.text.download import URLGenerator


@dataclass
class BaseCommonCrawlUrlGenerator(URLGenerator, ABC):
    """Get URLs for Common Crawl data
    Each concrete implementation must implement `_parse_datetime_from_snapshot_string` and `generate_path_urls`
    """

    start_snapshot_str: str
    end_snapshot_str: str
    data_prefix: str = "https://data.commoncrawl.org"
    limit: int | None = None

    @abstractmethod
    def _parse_datetime_from_snapshot_string(self, snapshot_str: str, for_start: bool) -> datetime:
        """Parses a snapshot string (YYYY-WW or YYYY-MM) into a datetime object."""

    @abstractmethod
    def generate_path_urls(self) -> list[str]:
        """Generates the list of URLs pointing to warc.paths.gz files."""

    def __post_init__(self):
        self._start_date, self._end_date = self._start_end_dates()

    def _start_end_dates(self) -> tuple[date, date]:
        """
        Parses the start and end snapshot strings into date objects.
        For 'news' (YYYY-MM), the day is set to 1 for start_date, and the last day of the month for end_date
        to ensure the full month is covered.
        """
        start_dt_obj = self._parse_datetime_from_snapshot_string(self.start_snapshot_str, for_start=True)
        end_dt_obj = self._parse_datetime_from_snapshot_string(self.end_snapshot_str, for_start=False)

        start_date = start_dt_obj.date()
        end_date = end_dt_obj.date()

        if start_date > end_date:
            msg = f"Start snapshot '{self.start_snapshot_str}' is after end snapshot '{self.end_snapshot_str}'"
            raise ValueError(msg)

        today_utc_date = datetime.now(tz=timezone.utc).date()
        if end_date > today_utc_date:
            logger.warning(
                f"Requested end date {end_date} is in the future. Adjusting end date to today's date ({today_utc_date})."
            )
            end_date = today_utc_date
        return start_date, end_date

    def generate_data_urls(self, path_urls: str | list[str] | None = None) -> list[str]:  # noqa: C901
        """
        Fetches all relevant warc.paths.gz files, decompresses them,
        and returns a list of all individual WARC file URLs.
        """
        if path_urls is None:
            gz_path_urls = self.generate_path_urls()
        else:
            gz_path_urls = [path_urls] if isinstance(path_urls, str) else path_urls

        all_individual_warc_urls = []

        if not gz_path_urls:
            return []

        for gz_path_url in gz_path_urls:
            try:
                response = requests.get(gz_path_url, stream=True, timeout=30)  # Added timeout
                response.raise_for_status()
                decompressed_data = zlib.decompress(response.content, zlib.MAX_WBITS | 32)
                warc_relative_paths = decompressed_data.decode("utf-8").splitlines()

                for rel_path in warc_relative_paths:
                    if rel_path.strip():
                        full_warc_url = urljoin(self.data_prefix, rel_path.strip())
                        all_individual_warc_urls.append(full_warc_url)
            except requests.RequestException as e:  # noqa: PERF203
                logger.error(f"Failed to download or access {gz_path_url}: {e}")
            except zlib.error as e:
                logger.error(f"Failed to decompress {gz_path_url}: {e}")
            except UnicodeDecodeError as e:
                logger.error(f"Failed to decode content from {gz_path_url}: {e}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"An unexpected error occurred while processing {gz_path_url}: {e}")

        if self.limit:
            logger.warning(f"Limiting the number of WARC URLs to {self.limit} from {len(all_individual_warc_urls)}")
            all_individual_warc_urls = all_individual_warc_urls[: self.limit]

        return all_individual_warc_urls

    def generate_urls(self) -> list[str]:
        """Process the task and return a list of WARC URLs"""
        path_urls = self.generate_path_urls()
        return self.generate_data_urls(path_urls)


@dataclass
class MainCommonCrawlUrlGenerator(BaseCommonCrawlUrlGenerator):
    index_prefix: str = "https://index.commoncrawl.org"

    def _parse_datetime_from_snapshot_string(self, snapshot_str: str, for_start: bool) -> datetime:  # noqa: ARG002
        try:
            year, week = map(int, snapshot_str.split("-"))
        except ValueError as e:
            msg = f"Invalid Main CC snapshot format. Use YYYY-WW (e.g., '2020-50'). Provided: '{snapshot_str}'"
            raise ValueError(msg) from e

        if not (1 <= week <= 53):  # noqa: PLR2004
            msg = f"Week number must be between 1 and 53. Provided: '{snapshot_str}'"
            raise ValueError(msg)
        return datetime.fromisocalendar(year, week, 1)

    @cached_property
    def _snapshot_index(self) -> list[dict]:
        collinfo_url = urljoin(self.index_prefix, "collinfo.json")
        try:
            response = requests.get(collinfo_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            msg = f"Failed to fetch Common Crawl index from {collinfo_url}: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Failed to decode JSON from Common Crawl index {collinfo_url}: {e}"
            raise RuntimeError(msg) from e

    def generate_path_urls(self) -> list[str]:
        start_date = self._start_date
        end_date = self._end_date
        if start_date.year < 2013:  # noqa: PLR2004
            logger.warning(
                "Only snapshots after 2013 are supported by this script. Adjusting start date to 2013-01-01"
            )
            start_date = date(2013, 1, 1)
        snapshot_data = self._snapshot_index
        warc_gz_urls = []
        crawl_data_main_prefix = urljoin(self.data_prefix, "crawl-data/CC-MAIN")

        for snapshot_info in snapshot_data:
            snapshot_id = snapshot_info.get("id")
            # 2008-2010 are old snapshots and not supported by this script
            if not snapshot_id or snapshot_id in {"CC-MAIN-2009-2010", "CC-MAIN-2008-2009"}:
                logger.warning(f"Skipping old snapshot for Common Crawl: {snapshot_id}")
                continue
            try:
                parts = snapshot_id.split("-")
                if len(parts) == 4 and parts[0] == "CC" and parts[1] == "MAIN":  # noqa: PLR2004
                    year_str, week_str = parts[2], parts[3]
                    year, week = int(year_str), int(week_str)
                    current_snapshot_dt = datetime.fromisocalendar(year, week, 1)
                    if start_date <= current_snapshot_dt.date() <= end_date:
                        path = f"{crawl_data_main_prefix}-{year}-{week:02d}/warc.paths.gz"
                        warc_gz_urls.append(path)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse Main CC snapshot ID: {snapshot_id}")
                continue
        if not warc_gz_urls:
            logger.error(
                f"No Main CC snapshots found for the period {self.start_snapshot_str} to {self.end_snapshot_str}."
            )
        return warc_gz_urls


@dataclass
class NewsCommonCrawlUrlGenerator(BaseCommonCrawlUrlGenerator):
    def _parse_datetime_from_snapshot_string(self, snapshot_str: str, for_start: bool) -> datetime:
        try:
            year, month = map(int, snapshot_str.split("-"))
            if not (1 <= month <= 12):  # noqa: PLR2004
                msg = f"Month must be between 1 and 12. Provided: '{snapshot_str}'"
                raise ValueError(msg)  # noqa : TRY301
            if for_start:
                return datetime(year, month, 1, tzinfo=timezone.utc)
            # For end_date, set to the last day of the month to ensure full month coverage
            next_month = month + 1
            next_year = year
            if next_month > 12:  # noqa: PLR2004
                next_month = 1
                next_year += 1
            return datetime(next_year, next_month, 1, tzinfo=timezone.utc) - timedelta(days=1)

        except ValueError as e:
            msg = f"Invalid News CC snapshot format. Use YYYY-MM (e.g., '2020-08'). Provided: '{snapshot_str}'"
            raise ValueError(msg) from e

    def generate_path_urls(self) -> list[str]:
        start_date = self._start_date
        end_date = self._end_date

        # CC-NEWS specific constraints
        min_news_date = date(2016, 8, 1)
        if start_date < min_news_date:
            logger.warning(
                f"Requested start date {start_date} is before the earliest available news data {min_news_date}. Adjusting start date to {min_news_date}."
            )
            start_date = min_news_date

        warc_gz_urls = []
        crawl_data_news_prefix = urljoin(self.data_prefix, "crawl-data/CC-NEWS")

        current_year = start_date.year
        current_month = start_date.month

        while True:
            current_dt = date(current_year, current_month, 1)
            if current_dt > end_date:  # Check against the start of the month for effective_end_date
                break

            # Check if current_dt is within the original unadjusted end_date's month and year
            # This is to correctly handle the case where end_snapshot_str might be for a month
            # that is partially in the future, but effective_end_date got capped to today.
            # We only want to include months up to the original end_snapshot_str's month.
            if current_year > end_date.year or (current_year == end_date.year and current_month > end_date.month):
                break

            path = f"{crawl_data_news_prefix}/{current_year}/{current_month:02d}/warc.paths.gz"
            warc_gz_urls.append(path)

            if current_month == 12:  # noqa: PLR2004
                current_month = 1
                current_year += 1
            else:
                current_month += 1

        if not warc_gz_urls:
            logger.warning(
                f"No News CC snapshots found for the period {self.start_snapshot_str} to {self.end_snapshot_str}."
            )
        # We reverse here to be consistent with the Main CC crawls which outputs the latest snapshots first
        return warc_gz_urls[::-1]
