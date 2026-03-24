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

"""Manifest reading stage: reads a Parquet URL manifest and produces FileGroupTasks.

This replaces cc-img-dl's manifest.py (iter_records, write_shards, shard_id_for).
Sharding is eliminated because Curator's executor distributes tasks automatically.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, _EmptyTask


def validate_url(url: str) -> bool:
    """Validate URL is safe HTTP/HTTPS with public IP (SSRF protection)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.hostname:
            return False
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False
        except ValueError:
            pass
        return True
    except (ValueError, AttributeError):
        return False


@dataclass
class ManifestReaderStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Stage that reads a Parquet URL manifest and produces FileGroupTasks.

    Each output FileGroupTask contains a batch of URLs (controlled by urls_per_task).
    This replaces cc-img-dl's manifest reading + sharding logic.

    The Parquet manifest must contain an 'asset_url' column (configurable via url_column).
    Additional columns (url_id, asset_mime, target_path, source_ref) are preserved
    in task metadata for downstream stages.
    """

    manifest_path: str = ""
    urls_per_task: int = 500
    limit: int | None = None
    url_column: str = "asset_url"
    id_column: str = "url_id"
    mime_column: str = "asset_mime"
    target_path_column: str = "target_path"
    name: str = "ManifestReaderStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: _EmptyTask) -> list[FileGroupTask]:
        """Read manifest and produce FileGroupTasks with URL batches."""
        pf = pq.ParquetFile(self.manifest_path)
        records: list[dict[str, Any]] = []
        count = 0

        for batch in pf.iter_batches():
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                if self.limit and count >= self.limit:
                    break
                url = getattr(row, self.url_column)
                if not validate_url(url):
                    continue
                records.append({
                    "url_id": getattr(row, self.id_column, f"url_{count}"),
                    "asset_url": url,
                    "asset_mime": getattr(row, self.mime_column, ""),
                    "target_path": getattr(row, self.target_path_column, ""),
                })
                count += 1
            if self.limit and count >= self.limit:
                break

        logger.info(f"ManifestReaderStage: read {len(records)} valid URLs from {self.manifest_path}")

        tasks = []
        for i in range(0, len(records), self.urls_per_task):
            batch_records = records[i : i + self.urls_per_task]
            urls = [r["asset_url"] for r in batch_records]
            tasks.append(
                FileGroupTask(
                    task_id=f"{task.task_id}_manifest_batch_{i // self.urls_per_task}",
                    dataset_name=task.dataset_name,
                    data=urls,
                    _metadata={
                        "url_records": batch_records,
                        "batch_index": i // self.urls_per_task,
                    },
                )
            )

        logger.info(f"ManifestReaderStage: created {len(tasks)} tasks ({self.urls_per_task} URLs each)")
        return tasks

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}
