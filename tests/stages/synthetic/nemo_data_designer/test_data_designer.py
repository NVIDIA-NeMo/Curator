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
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats

# Optional: skip entire module if data_designer not installed (e.g. optional dep)
pytest.importorskip("data_designer")

import data_designer.config as dd
from data_designer.config.preview_results import PreviewResults
from data_designer.interface import DataDesigner


def _minimal_config_builder() -> dd.DataDesignerConfigBuilder:
    """Real minimal DataDesignerConfigBuilder (avoids 'model configs required' where no local defaults)."""
    return dd.DataDesignerConfigBuilder(
        model_configs=[dd.ModelConfig(alias="test_model", model="test/model")]
    )


class TestBaseDataDesignerStage:
    """Unit tests for DataDesignerStage using real data_designer objects; only preview() is mocked.

    Note: Verification of _metadata and _stage_perf for each output task is done in the
    integration test (test_pipeline_e2e_reader_ndd_writer). That makes the mocked
    preservation checks here (e.g. in test_process) somewhat redundant; we keep them
    for fast unit-level feedback only.
    """

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

            # Use non-empty _metadata and _stage_perf like other stages (video reader, text writers)
            original_metadata = {"k": "v"}
            original_stage_perf = [
                StagePerfStats(
                    stage_name="jsonl_reader",
                    process_time=0.1,
                    num_items_processed=1,
                ),
            ]
            batch = DocumentBatch(
                data=input_df,
                dataset_name="ds1",
                task_id="task-1",
                _metadata=original_metadata,
                _stage_perf=original_stage_perf,
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
            # Preserve metadata and stage_perf (same assertion style as video reader, URL generation, image convert)
            assert out_batch._metadata == original_metadata
            assert out_batch._stage_perf == original_stage_perf

    def test_process_preserves_metadata(self) -> None:
        """Test process preserves task _metadata and _stage_perf (same pattern as VideoReaderStage.test_process_preserves_metadata)."""
        real_builder = _minimal_config_builder()
        stage = DataDesignerStage(config_builder=real_builder, verbose=False)
        stage.setup()
        stage.data_designer.preview = MagicMock(
            return_value=PreviewResults(
                config_builder=real_builder,
                dataset=pd.DataFrame([{"text": "hi", "out": "generated"}]),
            )
        )

        original_metadata = {"source": "test", "source_files": ["/path/to/file.jsonl"]}
        original_stage_perf = [
            StagePerfStats(stage_name="jsonl_reader", process_time=0.1, num_items_processed=1),
        ]
        batch = DocumentBatch(
            data=pd.DataFrame([{"text": "hello"}]),
            dataset_name="ds1",
            task_id="task-1",
            _metadata=original_metadata,
            _stage_perf=original_stage_perf,
        )
        result = stage.process(batch)

        assert result._metadata == original_metadata
        assert result._stage_perf == original_stage_perf

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
        # Permanent handler: respond to any number of /v1/chat/completions requests.
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
class TestDataDesignerStagePipelineIntegration:
    """Integration tests: pipeline.run(executor, initial_tasks=...) with DataDesignerStage and mock LLM."""

    def test_pipeline_run_end_to_end(self, httpserver: pytest_httpserver.HTTPServer) -> None:
        """Run pipeline with streaming executor so the executor drives setup and process."""
        from nemo_curator.backends.xenna import XennaExecutor

        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "e2e"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
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
        executor = XennaExecutor(config={"execution_mode": "streaming"})
        result_tasks = pipeline.run(executor, initial_tasks=initial_tasks)

        assert result_tasks is not None
        assert len(result_tasks) == 1
        out = result_tasks[0]
        assert isinstance(out, DocumentBatch)
        assert out.task_id == "e2e-1"
        assert out.dataset_name == "integration"
        assert out.data is not None
        expected_rows = len(initial_tasks[0].data)
        assert len(out.data) == expected_rows, (
            f"Output row count {len(out.data)} should match input {expected_rows}"
        )
        expected_columns = {"x", "out"}
        assert expected_columns.issubset(out.data.columns), (
            f"Output should have columns {expected_columns}, got {list(out.data.columns)}"
        )

    def test_pipeline_e2e_reader_ndd_writer(
        self,
        httpserver: pytest_httpserver.HTTPServer,
        tmp_path: Path,
    ) -> None:
        """Realistic e2e: N rows x M files → JsonlReader(files_per_partition=1) → NDD → JsonlWriter.

        Asserts: M output files, each with same row count as input file and new column from NDD;
        for each output task, verifies _metadata (e.g. source_files) and _stage_perf for all stages.
        This is the canonical check for metadata/stage_perf; the mocked unit tests above are
        redundant with it and kept only for fast unit-level feedback.
        """
        from nemo_curator.backends.xenna import XennaExecutor

        n_rows_per_file = 3
        m_files = 4

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # 1. Create M JSONL files with N rows each
        input_files = []
        for fi in range(m_files):
            path = input_dir / f"doc_{fi}.jsonl"
            input_files.append(str(path))
            with open(path, "w") as f:
                for ri in range(n_rows_per_file):
                    rec = {"text": f"file{fi}_row{ri}"}
                    f.write(json.dumps(rec) + "\n")

        # 2. Mock LLM
        mock_completion = {
            "id": "mock-id",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "generated"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
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

        # 3. Three-stage pipeline: JsonlReader → NDD → JsonlWriter
        pipeline = Pipeline(
            name="ndd_e2e_reader_ndd_writer",
            description="Reader → DataDesigner → Writer",
            stages=[
                JsonlReader(file_paths=str(input_dir), files_per_partition=1),
                DataDesignerStage(
                    config_builder=config_builder,
                    model_providers=[mock_provider],
                    verbose=False,
                ),
                JsonlWriter(path=str(output_dir)),
            ],
        )
        executor = XennaExecutor(config={"execution_mode": "streaming"})
        result_tasks = pipeline.run(executor)

        # 4. Output is M tasks (FileGroupTask from writer), one per input file
        assert result_tasks is not None
        assert len(result_tasks) == m_files, (
            f"Expected {m_files} output tasks (one per file), got {len(result_tasks)}"
        )
        assert all(isinstance(t, FileGroupTask) for t in result_tasks)

        # 5. Verify output files: M files, each with N rows and column "out"
        output_paths = []
        for task in result_tasks:
            assert task.data, f"Task {task.task_id} should have written file path(s)"
            output_paths.extend(task.data)
        assert len(output_paths) == m_files
        for out_path in output_paths:
            with open(out_path) as f:
                lines = f.readlines()
            assert len(lines) == n_rows_per_file, (
                f"Output file {out_path} should have {n_rows_per_file} rows, got {len(lines)}"
            )
            for line in lines:
                obj = json.loads(line)
                assert "out" in obj, f"Output should have NDD column 'out', keys: {list(obj.keys())}"
                assert "text" in obj

        # 6. Verify metadata and stage_perf_stats for each output task
        expected_stage_names = ["jsonl_reader", "DataDesignerStage", "jsonl_writer"]
        # jsonl_reader reports 1 item (one file-group task); NDD and writer report row count
        expected_items_per_stage = {
            "jsonl_reader": 1,
            "DataDesignerStage": n_rows_per_file,
            "jsonl_writer": n_rows_per_file,
        }
        for task in result_tasks:
            assert task._metadata, "Output task should have _metadata"
            assert "source_files" in task._metadata, (
                "Writer should preserve source_files from reader for deterministic naming"
            )
            assert len(task._stage_perf) == len(expected_stage_names), (
                f"Expected {len(expected_stage_names)} stage perf entries, got {len(task._stage_perf)}"
            )
            for idx, perf in enumerate(task._stage_perf):
                assert perf.stage_name == expected_stage_names[idx], (
                    f"Stage {idx}: expected {expected_stage_names[idx]}, got {perf.stage_name}"
                )
                expected_items = expected_items_per_stage[perf.stage_name]
                assert perf.num_items_processed == expected_items, (
                    f"Stage {perf.stage_name}: expected {expected_items} items, got {perf.num_items_processed}"
                )
                assert perf.process_time >= 0
