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

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import pytest_httpserver

from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

# Optional: skip entire module if data_designer not installed (e.g. optional dep)
pytest.importorskip("data_designer")

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage


def _minimal_config_builder() -> dd.DataDesignerConfigBuilder:
    """Real minimal DataDesignerConfigBuilder (avoids 'model configs required' where no local defaults)."""
    return dd.DataDesignerConfigBuilder(
        model_configs=[dd.ModelConfig(alias="test_model", model="test/model")]
    )


class TestBaseDataDesignerStage:
    """Unit tests for DataDesignerStage using real data_designer objects; only preview() is mocked."""

    def test_post_init_validation(self) -> None:
        """Either config_builder or data_designer_config_file must be set; only one can be set."""
        real_builder = _minimal_config_builder()

        with pytest.raises(ValueError, match="Either .* must be set"):
            DataDesignerStage(config_builder=None, data_designer_config_file=None)

        with pytest.raises(ValueError, match="Only one of .* can be set"):
            DataDesignerStage(
                config_builder=real_builder,
                data_designer_config_file="/path/to/config.yaml",
            )

        stage_builder = DataDesignerStage(config_builder=real_builder)
        assert stage_builder.config_builder is real_builder
        assert stage_builder.data_designer_config_file is None

        stage_file = DataDesignerStage(data_designer_config_file="/some/config.yaml")
        assert stage_file.config_builder is None
        assert stage_file.data_designer_config_file == "/some/config.yaml"

    def test_properties(self) -> None:
        """Stage name, default/custom resources, and inputs/outputs."""
        stage = DataDesignerStage(config_builder=_minimal_config_builder())
        assert stage.name == "DataDesignerStage"
        assert stage.resources == Resources(gpus=0.0)
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

        stage_custom = DataDesignerStage(config_builder=_minimal_config_builder()).with_(
            resources=Resources(gpus=2.0)
        )
        assert stage_custom.resources == Resources(gpus=2.0)

    def test_setup_with_config_builder(self) -> None:
        """When config_builder is set, setup does not load from file; DataDesigner is created."""
        real_builder = _minimal_config_builder()
        stage = DataDesignerStage(config_builder=real_builder)
        stage.setup()
        assert stage.config_builder is real_builder
        assert isinstance(stage.data_designer, DataDesigner)

    def test_setup_with_config_file(self, tmp_path: Path) -> None:
        """When data_designer_config_file is set, setup calls from_config and uses the returned builder."""
        config_path = tmp_path / "config.yaml"
        real_builder = _minimal_config_builder()
        with patch.object(
            dd.DataDesignerConfigBuilder, "from_config", return_value=real_builder
        ) as mock_from_config:
            stage = DataDesignerStage(data_designer_config_file=str(config_path))
            stage.setup()
        mock_from_config.assert_called_once_with(str(config_path))
        assert stage.config_builder is real_builder
        assert isinstance(stage.data_designer, DataDesigner)

    def test_process(self) -> None:
        """process uses real config_builder and DataFrameSeedSource; only preview return is stubbed."""
        real_builder = _minimal_config_builder()
        with patch.object(
            real_builder, "with_seed_dataset", wraps=real_builder.with_seed_dataset
        ) as spy_with_seed:
            stage = DataDesignerStage(config_builder=real_builder, verbose=False)
            stage.setup()

            input_df = pd.DataFrame([{"text": "hello"}])
            output_df = pd.DataFrame([{"text": "hello", "generated": "world"}])
            stage.data_designer.preview = MagicMock(
                return_value=PreviewResults(config_builder=real_builder, dataset=output_df)
            )

            batch = DocumentBatch(
                data=input_df,
                dataset_name="ds1",
                task_id="task-1",
                _metadata={"k": "v"},
                _stage_perf=[],
            )
            out_batch = stage.process(batch)

            spy_with_seed.assert_called_once()
            seed_arg = spy_with_seed.call_args[0][0]
            assert isinstance(seed_arg, dd.DataFrameSeedSource)
            assert seed_arg.df is not None
            assert len(seed_arg.df) == 1
            pd.testing.assert_frame_equal(seed_arg.df, input_df)

        stage.data_designer.preview.assert_called_once_with(real_builder, num_records=1)

        assert isinstance(out_batch, DocumentBatch)
        assert out_batch.task_id == "task-1"
        assert out_batch.dataset_name == "ds1"
        assert out_batch.data is output_df
        assert out_batch._metadata == {"k": "v"}
        assert out_batch._stage_perf == []

    def test_process_empty_batch(self) -> None:
        """process handles empty dataframe."""
        real_builder = _minimal_config_builder()
        stage = DataDesignerStage(config_builder=real_builder, verbose=False)
        stage.setup()

        output_df = pd.DataFrame()
        stage.data_designer.preview = MagicMock(
            return_value=PreviewResults(config_builder=real_builder, dataset=output_df)
        )

        batch = DocumentBatch(data=pd.DataFrame(), dataset_name="ds", task_id="t1")
        out_batch = stage.process(batch)

        stage.data_designer.preview.assert_called_once_with(real_builder, num_records=0)
        assert len(out_batch.data) == 0
        assert out_batch.task_id == "t1"

    def test_process_logs_metrics(self) -> None:
        """process logs ndd_running_time, num_input_records, num_output_records."""
        real_builder = _minimal_config_builder()
        stage = DataDesignerStage(config_builder=real_builder, verbose=False)
        stage.setup()

        input_df = pd.DataFrame([{"a": 1}, {"a": 2}])
        output_df = pd.DataFrame(
            [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
        )
        stage.data_designer.preview = MagicMock(
            return_value=PreviewResults(config_builder=real_builder, dataset=output_df)
        )

        batch = DocumentBatch(data=input_df, dataset_name="ds", task_id="t1")
        stage.process(batch)

        assert hasattr(stage, "_custom_metrics")
        assert "ndd_running_time" in stage._custom_metrics
        assert stage._custom_metrics["num_input_records"] == 2.0
        assert stage._custom_metrics["num_output_records"] == 3.0

    def test_process_with_mock_llm_endpoint(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        """Run process() against a fake HTTP LLM endpoint (OpenAI-style) instead of mocking preview()."""
        # Minimal OpenAI chat-completions response so the engine gets valid JSON.
        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "mock output"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        # Allow multiple completion requests (one per generated row).
        for _ in range(10):
            httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
                mock_completion
            )

        base_url = httpserver.url_for("/v1")
        mock_provider = dd.ModelProvider(
            name="mock_llm",
            endpoint=base_url,
            provider_type="openai",
            api_key="sk-test",
        )
        designer = DataDesigner(model_providers=[mock_provider])

        config_builder = dd.DataDesignerConfigBuilder(
            model_configs=[dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")]
        )
        config_builder.add_column(
            dd.LLMTextColumnConfig(name="out", prompt="Say one word", model_alias="mock_model")
        )

        with patch(
            "nemo_curator.stages.synthetic.nemo_data_designer.data_designer.DataDesigner",
            return_value=designer,
        ):
            stage = DataDesignerStage(config_builder=config_builder, verbose=False)
            stage.setup()

        batch = DocumentBatch(
            data=pd.DataFrame([{"x": 1}]),
            dataset_name="ds",
            task_id="t1",
        )
        out_batch = stage.process(batch)

        assert isinstance(out_batch, DocumentBatch)
        assert out_batch.task_id == "t1"
        assert out_batch.data is not None
        assert hasattr(stage, "_custom_metrics")
        assert "ndd_running_time" in stage._custom_metrics


class TestBaseDataDesignerStageIntegration:
    """End-to-end: pipeline → DataDesignerStage → process() with mock LLM endpoint."""

    def test_pipeline_process_end_to_end(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        """Run a simple pipeline (single DataDesignerStage) from build → setup → process."""
        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "pipeline mock"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        for _ in range(10):
            httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
                mock_completion
            )

        mock_provider = dd.ModelProvider(
            name="mock_llm",
            endpoint=httpserver.url_for("/v1"),
            provider_type="openai",
            api_key="sk-test",
        )
        designer = DataDesigner(model_providers=[mock_provider])
        config_builder = dd.DataDesignerConfigBuilder(
            model_configs=[dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")]
        )
        config_builder.add_column(
            dd.LLMTextColumnConfig(name="out", prompt="One word", model_alias="mock_model")
        )

        with patch(
            "nemo_curator.stages.synthetic.nemo_data_designer.data_designer.DataDesigner",
            return_value=designer,
        ):
            stage = DataDesignerStage(config_builder=config_builder, verbose=False)
            pipeline = Pipeline(
                name="ndd_integration",
                description="DataDesigner stage integration",
                stages=[stage],
            )
            pipeline.build()
            assert len(pipeline.stages) == 1
            stage = pipeline.stages[0]
            stage.setup()

            batch = DocumentBatch(
                data=pd.DataFrame([{"x": 1}]),
                dataset_name="integration",
                task_id="e2e-1",
            )
            result = stage.process(batch)

        assert isinstance(result, DocumentBatch)
        assert result.task_id == "e2e-1"
        assert result.dataset_name == "integration"
        assert result.data is not None
        assert hasattr(stage, "_custom_metrics")
        assert "ndd_running_time" in stage._custom_metrics
