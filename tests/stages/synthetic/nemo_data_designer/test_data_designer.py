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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
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
        assert stage_builder.model_providers is None

        # Optional model_providers is stored when provided.
        custom_provider = dd.ModelProvider(
            name="custom",
            endpoint="https://example.com/v1",
            provider_type="openai",
        )
        stage_with_providers = DataDesignerStage(
            config_builder=real_builder,
            model_providers=[custom_provider],
        )
        assert stage_with_providers.model_providers == [custom_provider]
        assert isinstance(stage_with_providers.data_designer, DataDesigner)

        # When only data_designer_config_file is set, __post_init__ calls from_config();
        # patch it so we don't need a real file, and assert the path is stored and builder set.
        with patch.object(
            dd.DataDesignerConfigBuilder, "from_config", return_value=real_builder
        ) as mock_from_config:
            stage_file = DataDesignerStage(data_designer_config_file="/some/config.yaml")
        mock_from_config.assert_called_once_with("/some/config.yaml")
        assert stage_file.config_builder is real_builder
        assert stage_file.data_designer_config_file == "/some/config.yaml"

    def test_properties(self) -> None:
        """Stage name, default resources, and inputs/outputs."""
        stage = DataDesignerStage(config_builder=_minimal_config_builder())
        assert stage.name == "DataDesignerStage"
        assert stage.resources == Resources(gpus=0.0)
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

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

    def test_setup_with_model_providers(self) -> None:
        """When model_providers is set, the stage creates DataDesigner with those providers."""
        real_builder = _minimal_config_builder()
        custom_provider = dd.ModelProvider(
            name="test_provider",
            endpoint="https://test.example/v1",
            provider_type="openai",
        )
        stage = DataDesignerStage(
            config_builder=real_builder,
            model_providers=[custom_provider],
        )
        stage.setup()
        assert stage.model_providers == [custom_provider]
        assert isinstance(stage.data_designer, DataDesigner)
        # DataDesigner was constructed with our provider (process would use it; we only check setup).
        assert stage.data_designer is not None

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
            api_key="sk-test",  # pragma: allowlist secret
        )
        config_builder = dd.DataDesignerConfigBuilder(
            model_configs=[dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")]
        )
        config_builder.add_column(
            dd.LLMTextColumnConfig(name="out", prompt="Say one word", model_alias="mock_model")
        )

        # Tutorial-style: config_builder references provider "mock_llm"; pass model_providers
        # so the stage uses our fake endpoint instead of default providers (no patch needed).
        stage = DataDesignerStage(
            config_builder=config_builder,
            model_providers=[mock_provider],
            verbose=False,
        )
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


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestDataDesignerStagePipelineIntegration:
    """Integration tests: pipeline.run(executor, initial_tasks=...) with DataDesignerStage and mock LLM."""

    def test_pipeline_run_end_to_end(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        """Run pipeline.run(executor, initial_tasks=...) so the executor drives setup and process."""
        from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "e2e"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        for _ in range(10):
            httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
                mock_completion
            )

        base_url = httpserver.url_for("/v1")
        mock_provider = dd.ModelProvider(
            name="mock_llm",
            endpoint=base_url,
            provider_type="openai",
            api_key="sk-test",  # pragma: allowlist secret
        )
        config_builder = dd.DataDesignerConfigBuilder(
            model_configs=[dd.ModelConfig(alias="mock_model", model="test", provider="mock_llm")]
        )
        config_builder.add_column(
            dd.LLMTextColumnConfig(name="out", prompt="One word", model_alias="mock_model")
        )

        # Same as test_process_with_mock_llm_endpoint: pass model_providers so the stage
        # uses the fake httpserver (tutorial-style config, no patch).
        stage = DataDesignerStage(
            config_builder=config_builder,
            model_providers=[mock_provider],
            verbose=False,
        )
        pipeline = Pipeline(
            name="ndd_pipeline_integration",
            description="DataDesigner via pipeline.run()",
            stages=[stage],
        )
        initial_tasks = [
            DocumentBatch(
                data=pd.DataFrame([{"x": 1}]),
                dataset_name="integration",
                task_id="e2e-1",
            )
        ]
        executor = RayActorPoolExecutor()
        result_tasks = pipeline.run(executor, initial_tasks=initial_tasks)

        assert result_tasks is not None
        assert len(result_tasks) == 1
        out = result_tasks[0]
        assert isinstance(out, DocumentBatch)
        assert out.task_id == "e2e-1"
        assert out.dataset_name == "integration"
        assert out.data is not None
        assert len(out.data) >= 1
