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
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytest_httpserver

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.base import _FORMATTED_PROMPT_COL
from nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.nemotron_cc import (
    DistillStage,
    DiverseQAStage,
    ExtractKnowledgeStage,
    KnowledgeListStage,
    WikipediaParaphrasingStage,
)
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask

pytest.importorskip("data_designer")

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults

ALL_STAGES = [
    (WikipediaParaphrasingStage, "rephrased", NEMOTRON_CC_SYSTEM_PROMPT, WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE),
    (DiverseQAStage, "diverse_qa", NEMOTRON_CC_SYSTEM_PROMPT, DIVERSE_QA_PROMPT_TEMPLATE),
    (DistillStage, "distill", NEMOTRON_CC_DISTILL_SYSTEM_PROMPT, DISTILL_PROMPT_TEMPLATE),
    (ExtractKnowledgeStage, "extract_knowledge", NEMOTRON_CC_SYSTEM_PROMPT, EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE),
    (KnowledgeListStage, "knowledge_list", NEMOTRON_CC_SYSTEM_PROMPT, KNOWLEDGE_LIST_PROMPT_TEMPLATE),
]


def _model_configs() -> list:
    return [dd.ModelConfig(alias="test_model", model="test/model")]


def _make_stage(stage_cls: type, **kwargs: object) -> object:
    defaults = {"model_alias": "test_model", "model_configs": _model_configs()}
    defaults.update(kwargs)
    return stage_cls(**defaults)


class TestNemotronCCNDDStages:
    """Unit tests for NDD-backed NemotronCC concrete stages."""

    @pytest.mark.parametrize(
        ("stage_cls", "output_field", "expected_sys_prompt", "expected_prompt"),
        ALL_STAGES,
    )
    def test_defaults_and_io(
        self, stage_cls: type, output_field: str, expected_sys_prompt: str | None, expected_prompt: str
    ) -> None:
        """Each stage has correct default prompts, fields, and inputs/outputs."""
        stage = _make_stage(stage_cls)
        assert stage.system_prompt == expected_sys_prompt
        assert stage.prompt == expected_prompt
        assert stage.input_field == "text"
        assert stage.output_field == output_field
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [output_field])

    @pytest.mark.parametrize(
        ("stage_cls", "output_field"),
        [(s[0], s[1]) for s in ALL_STAGES],
    )
    def test_process_smoke(self, stage_cls: type, output_field: str) -> None:
        """process() with mocked preview produces correct output and drops temp column."""
        stage = _make_stage(stage_cls)
        stage.setup()

        output_df = pd.DataFrame(
            [
                {
                    "text": "doc",
                    _FORMATTED_PROMPT_COL: "prompt",
                    output_field: "generated",
                }
            ]
        )
        stage.data_designer.preview = MagicMock(
            return_value=PreviewResults(config_builder=stage.config_builder, dataset=output_df)
        )

        batch = DocumentBatch(data=pd.DataFrame([{"text": "doc"}]), dataset_name="ds", task_id="t1")
        out = stage.process(batch)

        assert isinstance(out, DocumentBatch)
        assert out.data[output_field].iloc[0] == "generated"
        assert _FORMATTED_PROMPT_COL not in out.data.columns

    def test_process_with_mock_llm_endpoint(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        """One representative stage (WikipediaParaphrasing) against a fake HTTP endpoint."""
        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "mock output"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(mock_completion)

        base_url = httpserver.url_for("/v1")
        mock_provider = dd.ModelProvider(
            name="mock_llm",
            endpoint=base_url,
            provider_type="openai",
            api_key="sk-test",  # pragma: allowlist secret
        )
        stage = WikipediaParaphrasingStage(
            model_alias="mock_model",
            model_configs=[dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")],
            model_providers=[mock_provider],
            verbose=False,
        )
        stage.setup()

        batch = DocumentBatch(data=pd.DataFrame([{"text": "hello"}]), dataset_name="ds", task_id="t1")
        out = stage.process(batch)

        assert isinstance(out, DocumentBatch)
        assert "rephrased" in out.data.columns
        assert _FORMATTED_PROMPT_COL not in out.data.columns


@pytest.mark.gpu
class TestNemotronCCNDDPipelineIntegration:
    """Integration tests: pipeline.run() with concrete NDD NemotronCC stages."""

    def _mock_llm_setup(self, httpserver: pytest_httpserver.HTTPServer) -> tuple:
        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "generated"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(mock_completion)

        base_url = httpserver.url_for("/v1")
        mock_provider = dd.ModelProvider(
            name="mock_llm",
            endpoint=base_url,
            provider_type="openai",
            api_key="sk-test",  # pragma: allowlist secret
        )
        model_configs = [dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")]
        return mock_provider, model_configs

    @pytest.mark.parametrize(
        ("stage_cls", "output_field"),
        [(s[0], s[1]) for s in ALL_STAGES],
    )
    def test_pipeline_run_end_to_end(
        self, httpserver: pytest_httpserver.HTTPServer, stage_cls: type, output_field: str
    ) -> None:
        from nemo_curator.backends.xenna import XennaExecutor

        mock_provider, model_configs = self._mock_llm_setup(httpserver)

        stage = stage_cls(
            model_alias="mock_model",
            model_configs=model_configs,
            model_providers=[mock_provider],
            verbose=False,
        )
        pipeline = Pipeline(
            name=f"ndd_{stage_cls.__name__}_pipeline",
            description=f"{stage_cls.__name__} via pipeline.run()",
            stages=[stage],
        )
        initial_tasks = [
            DocumentBatch(data=pd.DataFrame([{"text": "hello"}]), dataset_name="integration", task_id="e2e-1")
        ]
        result_tasks = pipeline.run(XennaExecutor(config={"execution_mode": "streaming"}), initial_tasks=initial_tasks)

        assert len(result_tasks) == 1
        out = result_tasks[0]
        assert isinstance(out, DocumentBatch)
        assert out.dataset_name == "integration"
        assert output_field in out.data.columns
        assert _FORMATTED_PROMPT_COL not in out.data.columns
        assert len(out.data) == 1

    def test_pipeline_e2e_reader_ndd_writer(self, httpserver: pytest_httpserver.HTTPServer, tmp_path: Path) -> None:
        """JsonlReader -> WikipediaParaphrasingStage -> JsonlWriter. Verifies files, _metadata, _stage_perf."""
        from nemo_curator.backends.xenna import XennaExecutor

        n_rows, m_files = 3, 2
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for fi in range(m_files):
            with open(input_dir / f"doc_{fi}.jsonl", "w") as f:
                f.writelines(json.dumps({"text": f"file{fi}_row{ri}"}) + "\n" for ri in range(n_rows))

        mock_provider, model_configs = self._mock_llm_setup(httpserver)

        pipeline = Pipeline(
            name="ndd_wikipedia_e2e",
            description="Reader -> WikipediaParaphrasingStage -> Writer",
            stages=[
                JsonlReader(file_paths=str(input_dir), files_per_partition=1),
                WikipediaParaphrasingStage(
                    model_alias="mock_model",
                    model_configs=model_configs,
                    model_providers=[mock_provider],
                    verbose=False,
                ),
                JsonlWriter(path=str(output_dir)),
            ],
        )
        result_tasks = pipeline.run(XennaExecutor(config={"execution_mode": "streaming"}))

        assert len(result_tasks) == m_files
        assert all(isinstance(t, FileGroupTask) for t in result_tasks)

        output_paths = [p for t in result_tasks for p in t.data]
        assert len(output_paths) == m_files
        for out_path in output_paths:
            with open(out_path) as f:
                lines = f.readlines()
            assert len(lines) == n_rows
            for line in lines:
                obj = json.loads(line)
                assert "rephrased" in obj
                assert "text" in obj
                assert _FORMATTED_PROMPT_COL not in obj

        expected_stage_names = ["jsonl_reader", "DataDesignerStage", "jsonl_writer"]
        for task in result_tasks:
            assert "source_files" in task._metadata
            assert len(task._stage_perf) == len(expected_stage_names)
            for idx, perf in enumerate(task._stage_perf):
                assert perf.stage_name == expected_stage_names[idx]
                assert perf.process_time >= 0
