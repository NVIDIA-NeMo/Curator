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
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytest_httpserver

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.base import (
    _FORMATTED_PROMPT_COL,
    NDDBaseSyntheticStage,
)
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats

pytest.importorskip("data_designer")

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner


def _model_configs() -> list:
    return [dd.ModelConfig(alias="test_model", model="test/model")]


def _make_stage(**kwargs: object) -> NDDBaseSyntheticStage:
    defaults = {
        "prompt": "Rephrase: {document}",
        "output_field": "result",
        "input_field": "text",
        "model_alias": "test_model",
        "model_configs": _model_configs(),
        "verbose": False,
    }
    defaults.update(kwargs)
    return NDDBaseSyntheticStage(**defaults)


class TestNDDBaseSyntheticStage:
    """Unit tests for NDDBaseSyntheticStage; only preview() is mocked."""

    def test_init_validation(self) -> None:
        """_build_config_from_prompt raises when neither config nor prompt+output_field+input_field is given."""
        with pytest.raises(ValueError, match=r"Either provide .* or set"):
            NDDBaseSyntheticStage(prompt=None, output_field=None)
        with pytest.raises(ValueError, match=r"Either provide .* or set"):
            NDDBaseSyntheticStage(prompt="Test: {document}", output_field=None)
        with pytest.raises(ValueError, match=r"Either provide .* or set"):
            NDDBaseSyntheticStage(prompt=None, output_field="result")
        with pytest.raises(ValueError, match=r"Either provide .* or set"):
            NDDBaseSyntheticStage(prompt="Test: {document}", output_field="result", input_field=None)

    def test_auto_build_config(self) -> None:
        """prompt+output_field auto-builds config; system_prompt and model_alias are included."""
        stage = NDDBaseSyntheticStage(
            system_prompt="Be helpful.",
            prompt="Test: {document}",
            output_field="result",
            input_field="text",
            model_alias="test_model",
            model_configs=_model_configs(),
        )
        assert isinstance(stage.config_builder, dd.DataDesignerConfigBuilder)
        assert isinstance(stage.data_designer, DataDesigner)
        assert stage.system_prompt == "Be helpful."
        assert stage.model_alias == "test_model"

    def test_skip_auto_build_with_config_builder(self) -> None:
        """Providing config_builder directly bypasses auto-build."""
        real_builder = dd.DataDesignerConfigBuilder(model_configs=_model_configs())
        stage = NDDBaseSyntheticStage(config_builder=real_builder, output_field="result", input_field="text")
        assert stage.config_builder is real_builder

    def test_skip_auto_build_with_config_file(self) -> None:
        """Providing data_designer_config_file bypasses auto-build and calls from_config."""
        real_builder = dd.DataDesignerConfigBuilder(model_configs=_model_configs())
        with patch.object(dd.DataDesignerConfigBuilder, "from_config", return_value=real_builder) as mock_fc:
            stage = NDDBaseSyntheticStage(data_designer_config_file="/some/config.yaml", output_field="r")
        mock_fc.assert_called_once_with("/some/config.yaml")
        assert stage.config_builder is real_builder

    def test_model_providers_forwarded(self) -> None:
        provider = dd.ModelProvider(name="p", endpoint="https://example.com/v1", provider_type="openai")
        stage = _make_stage(model_providers=[provider])
        assert stage.model_providers == [provider]
        assert isinstance(stage.data_designer, DataDesigner)

    def test_properties(self) -> None:
        """inputs, outputs, name, resources defaults."""
        stage = _make_stage()
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["result"])
        assert stage.name == "DataDesignerStage"
        assert stage.resources == Resources(gpus=0.0)

    def test_outputs_raises_without_output_field(self) -> None:
        real_builder = dd.DataDesignerConfigBuilder(model_configs=_model_configs())
        stage = NDDBaseSyntheticStage(config_builder=real_builder, output_field=None)
        with pytest.raises(ValueError, match="output_field must be set"):
            stage.outputs()

    def test_process_llm_prompt_and_response(self) -> None:
        stage = _make_stage()
        assert stage._process_llm_prompt({"text": "hello"}) == "Rephrase: hello"
        with pytest.raises(KeyError, match="Expected input field 'text'"):
            stage._process_llm_prompt({"wrong": "val"})
        assert stage._process_llm_response(["first"]) == "first"
        assert stage._process_llm_response([]) == ""

    def test_process_llm_prompt_raises_when_input_field_is_none(self) -> None:
        real_builder = dd.DataDesignerConfigBuilder(model_configs=_model_configs())
        stage = NDDBaseSyntheticStage(
            config_builder=real_builder,
            prompt="Test: {document}",
            output_field="result",
            input_field=None,
        )
        with pytest.raises(ValueError, match="'input_field' is None"):
            stage._process_llm_prompt({"text": "hello"})

    def test_process(self) -> None:
        """Covers pre-process prompt formatting, NDD call, post-process response, temp column
        removal, task_id appending, and metadata/stage_perf preservation."""
        stage = _make_stage()
        stage.setup()

        output_df = pd.DataFrame(
            [
                {"text": "a", _FORMATTED_PROMPT_COL: "Rephrase: a", "result": "out_a"},
                {"text": "b", _FORMATTED_PROMPT_COL: "Rephrase: b", "result": "out_b"},
            ]
        )
        stage.data_designer.preview = MagicMock(
            return_value=PreviewResults(config_builder=stage.config_builder, dataset=output_df)
        )

        original_metadata = {"source": "test"}
        original_stage_perf = [StagePerfStats(stage_name="reader", process_time=0.1, num_items_processed=1)]
        batch = DocumentBatch(
            data=pd.DataFrame([{"text": "a"}, {"text": "b"}]),
            dataset_name="ds",
            task_id="t1",
            _metadata=original_metadata,
            _stage_perf=original_stage_perf,
        )
        out = stage.process(batch)

        assert isinstance(out, DocumentBatch)
        assert out.task_id == "t1_DataDesignerStage"
        assert out.dataset_name == "ds"
        assert out.data["result"].tolist() == ["out_a", "out_b"]
        assert _FORMATTED_PROMPT_COL not in out.data.columns
        assert out._metadata == original_metadata
        assert out._stage_perf == original_stage_perf

    def test_process_no_output_field_in_result(self) -> None:
        """When NDD result lacks the output_field column, post-processing is skipped gracefully."""
        stage = _make_stage()
        stage.setup()

        output_df = pd.DataFrame([{"text": "hello", _FORMATTED_PROMPT_COL: "Rephrase: hello"}])
        stage.data_designer.preview = MagicMock(
            return_value=PreviewResults(config_builder=stage.config_builder, dataset=output_df)
        )
        batch = DocumentBatch(data=pd.DataFrame([{"text": "hello"}]), dataset_name="ds", task_id="t1")
        out = stage.process(batch)

        assert isinstance(out, DocumentBatch)
        assert "result" not in out.data.columns

    def test_process_with_mock_llm_endpoint(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        """process() against a fake HTTP LLM endpoint (no preview mock)."""
        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ndd output"}, "finish_reason": "stop"}
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
        stage = NDDBaseSyntheticStage(
            prompt="Rephrase: {document}",
            output_field="result",
            input_field="text",
            model_alias="mock_model",
            model_configs=[dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")],
            model_providers=[mock_provider],
            verbose=False,
        )
        stage.setup()

        batch = DocumentBatch(data=pd.DataFrame([{"text": "hello"}]), dataset_name="ds", task_id="t1")
        out = stage.process(batch)

        assert isinstance(out, DocumentBatch)
        assert "result" in out.data.columns
        assert _FORMATTED_PROMPT_COL not in out.data.columns


@pytest.mark.gpu
class TestNDDBaseSyntheticStagePipelineIntegration:
    """Integration tests: pipeline.run() with NDDBaseSyntheticStage and mock LLM."""

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

    def test_pipeline_run_end_to_end(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        from nemo_curator.backends.xenna import XennaExecutor

        mock_provider, model_configs = self._mock_llm_setup(httpserver)

        stage = NDDBaseSyntheticStage(
            prompt="Echo: {document}",
            output_field="result",
            input_field="text",
            model_alias="mock_model",
            model_configs=model_configs,
            model_providers=[mock_provider],
            verbose=False,
        )
        pipeline = Pipeline(
            name="ndd_base_pipeline",
            description="NDDBaseSyntheticStage via pipeline.run()",
            stages=[stage],
        )
        initial_tasks = [
            DocumentBatch(data=pd.DataFrame([{"text": "hello"}]), dataset_name="integration", task_id="e2e-1")
        ]
        executor = XennaExecutor(config={"execution_mode": "streaming"})
        result_tasks = pipeline.run(executor, initial_tasks=initial_tasks)

        assert result_tasks is not None
        assert len(result_tasks) == 1
        out = result_tasks[0]
        assert isinstance(out, DocumentBatch)
        assert out.dataset_name == "integration"
        assert "result" in out.data.columns
        assert _FORMATTED_PROMPT_COL not in out.data.columns
        assert len(out.data) == 1

    def test_pipeline_e2e_reader_ndd_writer(
        self,
        httpserver: pytest_httpserver.HTTPServer,
        tmp_path: Path,
    ) -> None:
        """JsonlReader -> NDDBaseSyntheticStage -> JsonlWriter. Verifies files, _metadata, _stage_perf."""
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
            name="ndd_base_e2e",
            description="Reader -> NDDBaseSyntheticStage -> Writer",
            stages=[
                JsonlReader(file_paths=str(input_dir), files_per_partition=1),
                NDDBaseSyntheticStage(
                    prompt="Rephrase: {document}",
                    output_field="result",
                    input_field="text",
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
                assert "result" in obj
                assert "text" in obj
                assert _FORMATTED_PROMPT_COL not in obj

        expected_stage_names = ["jsonl_reader", "DataDesignerStage", "jsonl_writer"]
        for task in result_tasks:
            assert "source_files" in task._metadata
            assert len(task._stage_perf) == len(expected_stage_names)
            for idx, perf in enumerate(task._stage_perf):
                assert perf.stage_name == expected_stage_names[idx]
                assert perf.process_time >= 0
