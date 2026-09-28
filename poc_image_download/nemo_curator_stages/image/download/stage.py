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

"""Composite stage that combines manifest reading, compliance filtering, and image downloading.

This is the user-facing API: a single stage that replaces the entire cc-img-dl pipeline.
It decomposes into 2 or 3 execution stages depending on whether compliance checking is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import FileGroupTask, _EmptyTask

from .compliance.filter import ComplianceFilterStage
from .downloader import ImageDownloaderStage
from .manifest import ManifestReaderStage


@dataclass
class ImageDownloadCompositeStage(CompositeStage[_EmptyTask, FileGroupTask]):
    """End-to-end image download pipeline as a single Curator CompositeStage.

    Decomposes into:
    1. ManifestReaderStage — reads Parquet manifest, produces FileGroupTasks with URL batches
    2. ComplianceFilterStage (optional) — filters out non-compliant URLs
    3. ImageDownloaderStage — downloads images to local storage

    This is the equivalent of cc-img-dl's entire pipeline (cli.py + worker.py +
    manifest.py + compliance/ + downloader.py + storage.py + checkpoint.py)
    collapsed into a single Curator stage that can be used in any Pipeline.

    Example:
        pipeline = Pipeline(
            name="image_download",
            stages=[
                ImageDownloadCompositeStage(
                    manifest_path="url_manifest.parquet",
                    output_dir="/data/images",
                    enable_compliance=True,
                ),
            ],
        )
        pipeline.run()
    """

    manifest_path: str = ""
    output_dir: str = ""
    urls_per_task: int = 500
    limit: int | None = None

    # Compliance settings
    enable_compliance: bool = True
    user_agent: str = "nemo-curator/1.0"
    compliance_failure_policy: str = "conservative"
    robots_ttl: int = 86400
    tdm_ttl: int = 86400

    # Download settings
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 3
    max_image_bytes: int = 50_000_000
    download_concurrency: int = 16

    # Manifest column names
    url_column: str = "asset_url"
    id_column: str = "url_id"
    mime_column: str = "asset_mime"
    target_path_column: str = "target_path"

    def __post_init__(self) -> None:
        reader = ManifestReaderStage(
            manifest_path=self.manifest_path,
            urls_per_task=self.urls_per_task,
            limit=self.limit,
            url_column=self.url_column,
            id_column=self.id_column,
            mime_column=self.mime_column,
            target_path_column=self.target_path_column,
        )

        stages: list[ProcessingStage] = [reader]

        if self.enable_compliance:
            compliance = ComplianceFilterStage(
                user_agent=self.user_agent,
                robots_ttl=self.robots_ttl,
                tdm_ttl=self.tdm_ttl,
                compliance_failure_policy=self.compliance_failure_policy,
            )
            stages.append(compliance)

        downloader = ImageDownloaderStage(
            output_dir=self.output_dir,
            user_agent=self.user_agent,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            max_retries=self.max_retries,
            max_image_bytes=self.max_image_bytes,
            max_workers=self.download_concurrency,
        )
        stages.append(downloader)

        self.stages = stages
        self.name = "ImageDownloadCompositeStage"
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        compliance_str = "with" if self.enable_compliance else "without"
        return (
            f"Image download pipeline {compliance_str} compliance checking. "
            f"Reads manifest from {self.manifest_path}, downloads to {self.output_dir}."
        )
