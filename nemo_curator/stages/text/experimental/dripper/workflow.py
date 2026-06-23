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

"""DripperHTMLWorkflow — end-to-end HTML content extraction pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.stages.text.experimental.dripper._base_stages import (
    DripperHTMLInferenceStage,
    DripperHTMLPostprocessStage,
    DripperHTMLPreprocessStage,
)
from nemo_curator.stages.text.experimental.dripper.layout_template import DripperHTMLLayoutTemplateStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor
    from nemo_curator.models.client.llm_client import AsyncLLMClient
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import Task


@dataclass(kw_only=True)
class DripperHTMLWorkflow:
    """End-to-end HTML content extraction pipeline (layout clustering + LLM inference)."""

    client: AsyncLLMClient | None
    model_name: str
    html_col: str = "html"
    url_col: str | None = "url"
    output_col: str = "dripper_content"
    perform_layout_clustering: bool = True
    layout_cluster_threshold: float = 0.95
    fallback: str = "trafilatura"
    output_format: str = "mm_md"
    max_concurrent_requests: int = 64
    health_check: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.client is None:
            msg = "DripperHTMLWorkflow requires a non-None 'client' (AsyncLLMClient)"
            raise ValueError(msg)
        self.model_name = self.model_name.strip()
        if not self.model_name:
            msg = "DripperHTMLWorkflow requires a non-empty 'model_name'"
            raise ValueError(msg)
        if not (0.0 < self.layout_cluster_threshold <= 1.0):
            msg = "layout_cluster_threshold must be in (0, 1]"
            raise ValueError(msg)
        if self.max_concurrent_requests <= 0:
            msg = "max_concurrent_requests must be positive"
            raise ValueError(msg)

    def run(self, executor: BaseExecutor, initial_tasks: list[Task] | None = None) -> WorkflowRunResult:
        start = time.time()

        if self.verbose:
            logger.info(
                "DripperHTMLWorkflow starting — model={}, layout_clustering={}",
                self.model_name,
                self.perform_layout_clustering,
            )

        stages = self._build_stages()
        pipeline = Pipeline(name="dripper_html_extraction")
        for stage in stages:
            pipeline.add_stage(stage)

        output_tasks = pipeline.run(executor=executor, initial_tasks=initial_tasks)

        elapsed = time.time() - start

        if self.verbose:
            logger.info(
                "DripperHTMLWorkflow complete in {:.1f}s",
                elapsed,
            )

        result = WorkflowRunResult(workflow_name="dripper_html_extraction")
        result.add_metadata("elapsed_s", elapsed)
        result.add_metadata("stages", [s.name for s in stages])
        result.add_pipeline_tasks("dripper_html_extraction", output_tasks)
        return result

    def _build_stages(self) -> list[ProcessingStage]:
        preprocess = DripperHTMLPreprocessStage(html_col=self.html_col, url_col=self.url_col)

        if self.perform_layout_clustering:
            # Preprocess → LayoutTemplate handles clustering + representative LLM + sibling propagation
            # (DripperHTMLLayoutTemplateStage also handles singletons/standalone pages internally)
            return [
                preprocess,
                DripperHTMLLayoutTemplateStage(
                    client=self.client,
                    model_name=self.model_name,
                    html_col=self.html_col,
                    url_col=self.url_col,
                    layout_cluster_threshold=self.layout_cluster_threshold,
                    layout_template_fallback_llm=True,
                    fallback=self.fallback,
                    output_format=self.output_format,
                    max_concurrent_requests=self.max_concurrent_requests,
                    health_check=self.health_check,
                ),
            ]

        # Standalone extraction path: Preprocess → Inference → Postprocess
        return [
            preprocess,
            DripperHTMLInferenceStage(
                client=self.client,
                model_name=self.model_name,
                max_concurrent_requests=self.max_concurrent_requests,
            ),
            DripperHTMLPostprocessStage(
                html_col=self.html_col,
                url_col=self.url_col,
                fallback=self.fallback,
                output_format=self.output_format,
                output_content_col=self.output_col,
            ),
        ]
