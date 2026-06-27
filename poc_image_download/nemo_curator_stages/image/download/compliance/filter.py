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

"""Compliance filter stage: removes URLs that fail robots.txt or TDMRep checks.

This replaces cc-img-dl's compliance/checker.py and the compliance portion of worker.py.
Modeled as a ProcessingStage filter: takes FileGroupTask of URLs, returns FileGroupTask
with only compliant URLs.

The compliance checkers are initialized in setup() (once per worker) so each worker
has its own cache. For high-scale deployments, swap InMemoryCache for Redis/DynamoDB.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any

import requests
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask

from .cache import InMemoryCache
from .robots import RobotsChecker
from .tdm import TDMChecker


@dataclass
class ComplianceFilterStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Filter stage that removes non-compliant URLs from a FileGroupTask.

    Checks both robots.txt and TDMRep for each URL. URLs that fail either
    check are removed. The compliance results are cached per-origin with
    configurable TTL.

    This is an opt-in stage — pipelines that don't need compliance checking
    can simply omit it.
    """

    user_agent: str = "nemo-curator/1.0"
    robots_ttl: int = 86400
    tdm_ttl: int = 86400
    compliance_failure_policy: str = "conservative"
    max_workers: int = 8
    name: str = "ComplianceFilterStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def __post_init__(self) -> None:
        self._robots_checker: RobotsChecker | None = None
        self._tdm_checker: TDMChecker | None = None
        self._session: requests.Session | None = None

    def setup(self, worker_metadata: Any = None) -> None:
        """Initialize per-worker compliance checkers with fresh caches."""
        cache = InMemoryCache()
        self._robots_checker = RobotsChecker(
            cache, self.user_agent, self.robots_ttl, self.compliance_failure_policy
        )
        self._tdm_checker = TDMChecker(cache, self.tdm_ttl, self.compliance_failure_policy)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = self.user_agent
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def teardown(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _check_one(self, url: str) -> bool:
        """Check if a single URL passes both robots.txt and TDMRep."""
        assert self._robots_checker is not None
        assert self._tdm_checker is not None
        assert self._session is not None

        robots_ok = self._robots_checker.allowed(url, self._session)
        if not robots_ok:
            return False
        tdm_ok = self._tdm_checker.allowed(url, self._session)
        return tdm_ok

    def process(self, task: FileGroupTask) -> FileGroupTask | None:
        """Filter URLs by compliance checks. Returns None if all URLs are filtered out."""
        if self._robots_checker is None:
            self.setup()

        urls = task.data
        allowed_urls: list[str] = []
        blocked_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self._check_one, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    if future.result():
                        allowed_urls.append(url)
                    else:
                        blocked_count += 1
                except Exception:
                    blocked_count += 1

        if blocked_count:
            logger.info(
                f"Task {task.task_id}: compliance filtered {blocked_count}/{len(urls)} URLs"
            )

        if not allowed_urls:
            return None

        # Preserve metadata but update url_records to only include allowed URLs
        metadata = dict(task._metadata)
        url_records = metadata.get("url_records")
        if url_records:
            allowed_url_set = set(allowed_urls)
            metadata["url_records"] = [r for r in url_records if r["asset_url"] in allowed_url_set]

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=allowed_urls,
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )
