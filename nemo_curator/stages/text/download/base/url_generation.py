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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, _EmptyTask


class URLGenerator(ABC):
    """Abstract base class for URL generators - generates URLs from minimal input."""

    @abstractmethod
    def generate_urls(self) -> list[str]:
        """Generate a list of URLs to download."""
        ...


@dataclass
class URLGenerationStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Stage that generates URLs from minimal input parameters.

    This allows pipelines to start with URL generation (like Common Crawl).
    """

    url_generator: URLGenerator
    limit: int | None = None
    _resources = Resources(cpus=0.5)

    def __post_init__(self):
        self._name = f"url_generation_{self.url_generator.__class__.__name__.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects empty task."""
        return ([], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces FileGroupTask with URLs."""
        return (["data"], [])

    def process(self, task: _EmptyTask) -> list[FileGroupTask]:
        """Generate URLs and create FileGroupTasks.

        Args:
            task (_EmptyTask): Empty input task

        Returns:
            list[FileGroupTask]: List of tasks containing URLs
        """

        # Create one task per URL for better parallelization
        urls = self.url_generator.generate_urls()
        if self.limit is not None:
            urls = urls[: self.limit]

        return [
            FileGroupTask(
                task_id=f"{task.task_id}_{i}",
                dataset_name=task.dataset_name,
                data=[url],
                _metadata={"source_url": url},
            )
            for i, url in enumerate(urls)
        ]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            "is_fanout_stage": True,
        }
