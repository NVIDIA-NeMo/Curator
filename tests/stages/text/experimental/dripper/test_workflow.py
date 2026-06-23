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

"""Tests for DripperHTMLWorkflow — no GPU, Ray, or LLM server required."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper import DripperHTMLWorkflow


class _StubLLMClient(AsyncLLMClient):
    def __init__(self) -> None:
        super().__init__(max_concurrent_requests=1, max_retries=0, base_delay=0.0)

    def setup(self) -> None:
        pass

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: object = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        return [""]


@pytest.fixture
def stub_client() -> _StubLLMClient:
    return _StubLLMClient()


@pytest.fixture
def base_workflow(stub_client: _StubLLMClient) -> DripperHTMLWorkflow:
    return DripperHTMLWorkflow(
        client=stub_client, model_name="test-model", perform_layout_clustering=False, health_check=False
    )


class TestDripperHTMLWorkflow:
    def test_instantiation_defaults(self, stub_client: _StubLLMClient) -> None:
        wf = DripperHTMLWorkflow(client=stub_client, model_name="test-model")
        assert wf.perform_layout_clustering is True
        assert wf.layout_cluster_threshold == pytest.approx(0.95)
        assert wf.fallback == "trafilatura"
        assert wf.output_format == "mm_md"
        assert wf.max_concurrent_requests == 64
        assert wf.health_check is True
        assert wf.verbose is True
        assert wf.html_col == "html"
        assert wf.url_col == "url"
        assert wf.output_col == "dripper_content"

    def test_custom_fields_stored(self, stub_client: _StubLLMClient) -> None:
        wf = DripperHTMLWorkflow(
            client=stub_client,
            model_name="custom-model",
            layout_cluster_threshold=0.85,
            perform_layout_clustering=False,
            fallback="bypass",
            output_format="text",
            max_concurrent_requests=32,
            health_check=False,
            verbose=False,
        )
        assert wf.model_name == "custom-model"
        assert wf.layout_cluster_threshold == pytest.approx(0.85)
        assert wf.fallback == "bypass"
        assert wf.output_format == "text"
        assert wf.max_concurrent_requests == 32

    @pytest.mark.parametrize("with_clustering", [True, False])
    def test_build_stages_returns_processing_stages(self, stub_client: _StubLLMClient, with_clustering: bool) -> None:
        wf = DripperHTMLWorkflow(
            client=stub_client, model_name="test-model", perform_layout_clustering=with_clustering, health_check=False
        )
        stages = wf._build_stages()
        assert len(stages) > 0
        assert all(isinstance(s, ProcessingStage) for s in stages)
        assert all(s.name.strip() for s in stages)

    def test_layout_clustering_toggle(self, stub_client: _StubLLMClient) -> None:
        with_clust = DripperHTMLWorkflow(
            client=stub_client, model_name="test-model", perform_layout_clustering=True, health_check=False
        )
        without_clust = DripperHTMLWorkflow(
            client=stub_client, model_name="test-model", perform_layout_clustering=False, health_check=False
        )
        assert len(with_clust._build_stages()) > len(without_clust._build_stages())
        with_names = [s.name for s in with_clust._build_stages()]
        without_names = [s.name for s in without_clust._build_stages()]
        assert any("Layout" in n for n in with_names)
        assert not any("Layout" in n for n in without_names)

    def test_core_stage_order(self, base_workflow: DripperHTMLWorkflow) -> None:
        names = [s.name for s in base_workflow._build_stages()]
        assert "DripperHTMLPreprocessStage" in names
        assert "DripperHTMLInferenceStage" in names
        assert "DripperHTMLPostprocessStage" in names
        assert names.index("DripperHTMLPreprocessStage") < names.index("DripperHTMLInferenceStage")
        assert names.index("DripperHTMLInferenceStage") < names.index("DripperHTMLPostprocessStage")

    def test_custom_column_names_propagate(self, stub_client: _StubLLMClient) -> None:
        wf = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            html_col="raw_html",
            url_col="page_url",
            output_col="extracted_text",
            perform_layout_clustering=False,
            health_check=False,
        )
        stages = wf._build_stages()
        preprocess = next(s for s in stages if s.name == "DripperHTMLPreprocessStage")
        postprocess = next(s for s in stages if s.name == "DripperHTMLPostprocessStage")
        assert preprocess.html_col == "raw_html"
        assert preprocess.url_col == "page_url"
        assert postprocess.output_content_col == "extracted_text"

    def test_post_init_validation_raises_for_none_client(self) -> None:
        with pytest.raises(ValueError, match="non-None"):
            DripperHTMLWorkflow(client=None, model_name="test-model")

    def test_post_init_validation_raises_for_empty_model(self, stub_client: _StubLLMClient) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            DripperHTMLWorkflow(client=stub_client, model_name="  ")

    def test_post_init_validation_raises_for_bad_threshold(self, stub_client: _StubLLMClient) -> None:
        with pytest.raises(ValueError, match="layout_cluster_threshold"):
            DripperHTMLWorkflow(client=stub_client, model_name="m", layout_cluster_threshold=1.5)

    def test_run_returns_workflow_run_result(
        self, base_workflow: DripperHTMLWorkflow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_curator.pipeline import Pipeline

        monkeypatch.setattr(Pipeline, "run", lambda _self, _executor, _initial_tasks=None: [])

        from nemo_curator.backends.xenna import XennaExecutor

        result = base_workflow.run(executor=XennaExecutor())
        assert isinstance(result, WorkflowRunResult)
        assert result.get_metadata("elapsed_s") >= 0.0
        assert isinstance(result.get_metadata("stages"), list)
        assert len(result.get_metadata("stages")) > 0
